from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from skill_moe.base import SkillRequest, SkillResponse, SkillMetadata


_MEM_TRANSCRIPTS: dict[str, str] = {}
_LOCAL_MODELS: dict[str, Any] = {}
_WINDOW_EXTEND_SEC = float(os.getenv("ASR_WINDOW_EXTEND_SEC", "10"))
_MAX_WINDOW_FOR_EXTENSION_SEC = float(os.getenv("ASR_MAX_WINDOW_FOR_EXTENSION_SEC", "25"))
_MIN_INFO_GAIN = float(os.getenv("ASR_MIN_INFO_GAIN", "0.10"))
_MAX_EXTENSION_ROUNDS = max(1, int(os.getenv("ASR_MAX_EXTENSION_ROUNDS", "2")))


def _is_incomplete_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if len(s) < 12:
        return True

    # Ends with a conjunction or unfinished connector.
    tail = s.lower()
    if re.search(r"(?:\b(?:and|but|or|so|because|if|when|then|that|to|for)\b)\s*$", tail):
        return True

    # No sentence-final punctuation usually means cut-off transcript.
    if not re.search(r"[.!?。！？]\s*$", s):
        return True
    return False


def _token_set(text: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text or "")
        if t.strip()
    }


def _information_gain(old_text: str, new_text: str) -> float:
    old_tokens = _token_set(old_text)
    new_tokens = _token_set(new_text)
    if not new_tokens:
        return 0.0
    if not old_tokens:
        return 1.0
    novel = len(new_tokens - old_tokens)
    return novel / max(1, len(new_tokens))


def _extended_end(
    end_time: Optional[float],
    duration: float,
) -> Optional[float]:
    if end_time is None or _WINDOW_EXTEND_SEC <= 0:
        return None
    new_end = end_time + _WINDOW_EXTEND_SEC
    if duration > 0:
        new_end = min(new_end, duration)
    if new_end <= end_time:
        return None
    return new_end


def _should_try_extension(start_time: Optional[float], end_time: Optional[float]) -> bool:
    if end_time is None:
        return False
    if start_time is None:
        return True
    return (end_time - start_time) <= _MAX_WINDOW_FOR_EXTENSION_SEC


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _detect_local_device() -> str:
    override = os.getenv("FAST_WHISPER_DEVICE")
    if override:
        return override
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_local_settings() -> dict[str, Any]:
    device = _detect_local_device()
    compute_type = os.getenv("FAST_WHISPER_COMPUTE")
    if not compute_type:
        compute_type = "float16" if device.startswith("cuda") else "int8"

    return {
        "model_size": os.getenv("FAST_WHISPER_MODEL", "large-v3"),
        "device": device,
        "compute_type": compute_type,
        "beam_size": int(os.getenv("FAST_WHISPER_BEAM_SIZE", "5")),
        "vad_filter": _env_flag("FAST_WHISPER_VAD_FILTER", default=True),
    }


def _cache_dir() -> Path:
    return Path(os.getenv("ASR_CACHE_DIR", ".cache/asr")).resolve()


def _cache_key(
    file_path: str,
    local_settings: dict[str, Any],
    start_time: Optional[float],
    end_time: Optional[float],
) -> str:
    st = os.stat(file_path)
    payload = {
        "v": 3,
        "path": str(Path(file_path).resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "local_settings": local_settings,
        "start_time": start_time,
        "end_time": end_time,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _read_disk_cache(key: str) -> str | None:
    fp = _cache_dir() / f"{key}.json"
    if not fp.is_file():
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        transcript = data.get("transcript")
        return transcript if isinstance(transcript, str) else None
    except Exception:
        return None


def _write_disk_cache(key: str, transcript: str) -> None:
    cache_root = _cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    fp = cache_root / f"{key}.json"
    fp.write_text(json.dumps({"transcript": transcript}, ensure_ascii=False), encoding="utf-8")


def _transcribe_with_new_sdk(file_path: str, model: str, base_url: str | None) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f)
    # New SDK returns an object with `.text`
    return getattr(resp, "text", str(resp))


def _transcribe_with_legacy_sdk(file_path: str, model: str, base_url: str | None) -> str:
    import openai  # type: ignore

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        openai.base_url = base_url
    with open(file_path, "rb") as f:
        resp: Any = openai.Audio.transcribe(model, f)
    return resp.get("text", str(resp))


def _try_transcribe(file_path: str, model: str, base_url: str | None) -> str:
    """
    Attempt transcription with the provided base_url; if connection fails and a base_url
    was set, fall back to the default OpenAI endpoint.
    """
    import openai  # type: ignore

    # First attempt with provided base_url
    if hasattr(openai, "OpenAI"):
        try:
            return _transcribe_with_new_sdk(file_path, model, base_url)
        except Exception:
            if base_url:
                # Retry without base_url
                return _transcribe_with_new_sdk(file_path, model, None)
            raise
    else:
        try:
            return _transcribe_with_legacy_sdk(file_path, model, base_url)
        except Exception:
            if base_url:
                return _transcribe_with_legacy_sdk(file_path, model, None)
            raise


def _get_local_model(local_settings: dict[str, Any]) -> Any:
    key = (
        f"{local_settings['model_size']}|{local_settings['device']}|"
        f"{local_settings['compute_type']}"
    )
    model = _LOCAL_MODELS.get(key)
    if model is not None:
        return model

    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(
        local_settings["model_size"],
        device=local_settings["device"],
        compute_type=local_settings["compute_type"],
    )
    _LOCAL_MODELS[key] = model
    return model


def _try_local_faster_whisper(
    file_path: str,
    local_settings: dict[str, Any],
    start_time: Optional[float],
    end_time: Optional[float],
) -> str:
    """
    Local faster-whisper path (GPU preferred when available).
    """
    model = _get_local_model(local_settings)
    kwargs = {
        "beam_size": local_settings["beam_size"],
        "vad_filter": local_settings["vad_filter"],
    }
    use_window = start_time is not None and end_time is not None and end_time > start_time
    if use_window:
        # Newer faster-whisper versions support clip_timestamps for partial decoding.
        kwargs["clip_timestamps"] = [start_time, end_time]

    try:
        segments, _info = model.transcribe(file_path, **kwargs)
    except TypeError:
        # Compatibility fallback for older faster-whisper builds.
        kwargs.pop("clip_timestamps", None)
        segments, _info = model.transcribe(file_path, **kwargs)

    text_parts = []
    for seg in segments:
        if not seg.text:
            continue
        if start_time is not None and getattr(seg, "end", 0.0) < start_time:
            continue
        if end_time is not None and getattr(seg, "start", 0.0) > end_time:
            continue
        text_parts.append(seg.text.strip())
    return " ".join(text_parts).strip()


def _backend_order() -> list[str]:
    backend = os.getenv("ASR_BACKEND", "local").strip().lower()
    allow_remote_fallback = _env_flag("ASR_ALLOW_REMOTE_FALLBACK", default=False)

    if backend == "cloud":
        return ["cloud"]
    if backend == "cloud_first":
        return ["cloud", "local"]
    if backend in {"local_first", "auto"}:
        return ["local", "cloud"] if allow_remote_fallback else ["local"]
    # Default: local only (zero API spend for ASR)
    return ["local"]


def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    """
    ASR runner.

    Default behavior is local faster-whisper on GPU (when available), which
    avoids ASR API cost. Cloud ASR is opt-in via ASR_BACKEND / fallback flags.
    """
    cloud_model = os.getenv("OPENAI_ASR_MODEL", "whisper-1")
    file_path = request.video_path
    base_url = os.getenv("OPENAI_BASE_URL")
    local_settings = _resolve_local_settings()
    backends = _backend_order()
    start_time, end_time = request.normalized_window()
    effective_end = end_time
    extension_applied = False
    extension_reason = ""
    extension_rounds = 0
    info_gain = 0.0
    window_desc = (
        f"{start_time:.1f}-{end_time:.1f}s"
        if start_time is not None and end_time is not None
        else "full video"
    )

    if not os.path.isfile(file_path):
        summary = f"[ASR] File not found: {file_path}"
        return SkillResponse(skill_name=metadata.name, summary=summary, artifacts={"error": "missing_file"})

    key = _cache_key(file_path, local_settings, start_time, end_time)
    if key in _MEM_TRANSCRIPTS:
        return SkillResponse(
            skill_name=metadata.name,
            summary=(
                "[ASR] Loaded transcript from cache "
                f"(backend=local, model={local_settings['model_size']}, "
                f"device={local_settings['device']}, window={window_desc})"
            ),
            content=_MEM_TRANSCRIPTS[key],
            artifacts={"transcript": _MEM_TRANSCRIPTS[key], "cached": True, "backend": "local"},
        )
    cached = _read_disk_cache(key)
    if cached:
        _MEM_TRANSCRIPTS[key] = cached
        return SkillResponse(
            skill_name=metadata.name,
            summary=(
                "[ASR] Loaded transcript from disk cache "
                f"(backend=local, model={local_settings['model_size']}, "
                f"device={local_settings['device']}, window={window_desc})"
            ),
            content=cached,
            artifacts={"transcript": cached, "cached": True, "backend": "local"},
        )

    errors: list[str] = []

    if "local" in backends:
        try:
            transcript = _try_local_faster_whisper(
                file_path, local_settings, start_time, end_time
            )
            while (
                _should_try_extension(start_time, effective_end)
                and extension_rounds < _MAX_EXTENSION_ROUNDS
                and _is_incomplete_text(transcript)
            ):
                new_end = _extended_end(effective_end, request.video_duration)
                if new_end is None:
                    break
                extended_transcript = _try_local_faster_whisper(
                    file_path, local_settings, start_time, new_end
                )
                gain = _information_gain(transcript, extended_transcript)
                if not extended_transcript:
                    break
                if gain < _MIN_INFO_GAIN and len(extended_transcript) <= len(transcript):
                    break
                transcript = extended_transcript
                effective_end = new_end
                extension_applied = True
                extension_reason = "incomplete_text"
                extension_rounds += 1
                info_gain = gain
            if transcript:
                _MEM_TRANSCRIPTS[key] = transcript
                _write_disk_cache(key, transcript)
            effective_window_desc = (
                f"{start_time:.1f}-{effective_end:.1f}s"
                if start_time is not None and effective_end is not None
                else window_desc
            )
            summary = (
                "[ASR] Transcribed locally with faster-whisper "
                f"(model={local_settings['model_size']}, "
                f"device={local_settings['device']}, "
                f"compute={local_settings['compute_type']}, "
                f"window={effective_window_desc})"
            )
            if extension_applied:
                summary += " [Dynamic Window Extension applied]"
            return SkillResponse(
                skill_name=metadata.name,
                summary=summary,
                content=transcript,
                artifacts={
                    "transcript": transcript,
                    "cached": False,
                    "backend": "local",
                    "requested_start": start_time,
                    "requested_end": end_time,
                    "window_start": start_time,
                    "window_end": effective_end,
                    "window_extended": extension_applied,
                    "window_extension_reason": extension_reason,
                    "window_extension_rounds": extension_rounds,
                    "info_gain": round(info_gain, 4),
                },
            )
        except Exception as exc_local:  # pragma: no cover
            errors.append(f"local:{exc_local}")

    if "cloud" in backends:
        try:
            import openai  # type: ignore
        except Exception as exc:
            errors.append(f"cloud:openai module unavailable ({exc})")
        else:
            try:
                transcript = _try_transcribe(file_path, cloud_model, base_url)
                used_base = base_url if base_url else "openai-default"
                summary = (
                    f"[ASR] Transcribed with {cloud_model} via {used_base} "
                    f"(window request={window_desc}, cloud may decode full audio)"
                )
                return SkillResponse(
                    skill_name=metadata.name,
                    summary=summary,
                    content=transcript,
                    artifacts={"transcript": transcript, "cached": False, "backend": "cloud"},
                )
            except Exception as exc_remote:  # pragma: no cover
                errors.append(f"cloud:{exc_remote}")

    summary = f"[ASR] Transcription failed ({'; '.join(errors) if errors else 'no backend attempted'})"
    return SkillResponse(
        skill_name=metadata.name,
        summary=summary,
        artifacts={"error": "; ".join(errors) if errors else "no_backend"},
    )
