"""
OCR skill — Temporal Grid OCR with TF-IDF Evidence Filtering.

Algorithm:
  1. Sample video at ~1fps (capped at MAX_GRID_FRAMES frames).
  2. Try local OCR (EasyOCR → PaddleOCR) on each frame individually.
     If no local OCR installed, build a grid image and send to gpt-4o-mini vision.
  3. Collect all (timestamp, text) pairs; deduplicate near-identical lines.
  4. TF-IDF RAG filter: rank deduplicated lines by relevance to the question.
     Keep top TOP_K_LINES; if best similarity < MIN_SIM_THRESHOLD → no_match=True.
  5. Return concise evidence. Prevents garbage injection into the VLM prompt.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------
_SAMPLE_FPS        = float(os.getenv("OCR_SAMPLE_FPS",    "1.0"))
_MAX_GRID_FRAMES   = int(os.getenv("OCR_MAX_FRAMES",       "40"))
_GRID_THUMB_W      = int(os.getenv("OCR_THUMB_W",          "320"))
_GRID_THUMB_H      = int(os.getenv("OCR_THUMB_H",          "240"))
_TOP_K_LINES       = int(os.getenv("OCR_TOP_K",            "3"))
_MIN_SIM_THRESHOLD = float(os.getenv("OCR_MIN_SIM",        "0.06"))
_DEDUP_RATIO       = float(os.getenv("OCR_DEDUP_RATIO",    "0.82"))

METADATA = SkillMetadata(
    name="ocr",
    description=(
        "Extracts text visible in the video (signs, scoreboards, captions, labels). "
        "Returns the top-3 most question-relevant text snippets found across the video timeline."
    ),
    parameters={"start_time": "float", "end_time": "float"},
)

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames_1fps(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_frames: int = _MAX_GRID_FRAMES,
) -> List[Tuple[float, np.ndarray]]:
    """Return list of (timestamp_sec, BGR_frame) at ~1fps, capped at max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    t_start = max(0.0, start_time)
    t_end   = min(duration, end_time) if end_time else duration

    # Compute timestamps at 1fps, then subsample to max_frames.
    raw_ts = [t_start + i * (1.0 / _SAMPLE_FPS)
              for i in range(int((t_end - t_start) * _SAMPLE_FPS) + 1)
              if t_start + i * (1.0 / _SAMPLE_FPS) <= t_end]
    if len(raw_ts) > max_frames:
        step = len(raw_ts) / max_frames
        raw_ts = [raw_ts[int(i * step)] for i in range(max_frames)]

    result: List[Tuple[float, np.ndarray]] = []
    for ts in raw_ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if ok:
            result.append((ts, frame))
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Local OCR backend
# ---------------------------------------------------------------------------

def _get_easyocr():
    if "easyocr" in _model_cache:
        return _model_cache["easyocr"]
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        _model_cache["easyocr"] = reader
        logger.info("[OCR] EasyOCR loaded (GPU)")
        return reader
    except Exception:
        try:
            import easyocr  # type: ignore
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _model_cache["easyocr"] = reader
            logger.info("[OCR] EasyOCR loaded (CPU)")
            return reader
        except Exception as e:
            logger.debug("[OCR] EasyOCR not available: %s", e)
            _model_cache["easyocr"] = None
            return None


def _get_paddleocr():
    if not os.getenv("OCR_ENABLE_PADDLE_FALLBACK"):
        return None
    if "paddleocr" in _model_cache:
        return _model_cache["paddleocr"]
    try:
        from paddleocr import PaddleOCR  # type: ignore
        reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        _model_cache["paddleocr"] = reader
        logger.info("[OCR] PaddleOCR loaded")
        return reader
    except Exception as e:
        logger.debug("[OCR] PaddleOCR not available: %s", e)
        _model_cache["paddleocr"] = None
        return None


def _local_ocr_frame(frame_bgr: np.ndarray) -> List[str]:
    """Run local OCR on a single frame. Returns list of text strings found."""
    reader = _get_easyocr()
    if reader is not None:
        try:
            results = reader.readtext(frame_bgr, detail=0, paragraph=True)
            return [str(r).strip() for r in results if str(r).strip()]
        except Exception as e:
            logger.debug("[OCR] EasyOCR frame error: %s", e)

    paddle = _get_paddleocr()
    if paddle is not None:
        try:
            results = paddle.ocr(frame_bgr, cls=True)
            texts = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        txt = str(line[1][0]).strip() if isinstance(line[1], (list, tuple)) else str(line[1]).strip()
                        if txt:
                            texts.append(txt)
            return texts
        except Exception as e:
            logger.debug("[OCR] PaddleOCR frame error: %s", e)

    return []


# ---------------------------------------------------------------------------
# Vision API grid OCR
# ---------------------------------------------------------------------------

def _vision_api_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"))


def _encode_frame_jpeg(frame_bgr: np.ndarray, max_side: int = 512) -> str:
    """Resize frame to max_side and encode as base64 JPEG."""
    from PIL import Image as PILImage  # type: ignore
    h, w = frame_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_side:
        scale = max_side / long_edge
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _build_grid_image(
    frame_ts_pairs: List[Tuple[float, np.ndarray]],
    thumb_w: int = _GRID_THUMB_W,
    thumb_h: int = _GRID_THUMB_H,
) -> Tuple[str, List[float]]:
    """Arrange frames into a grid image. Returns (base64_jpeg, list_of_timestamps)."""
    from PIL import Image as PILImage, ImageDraw, ImageFont  # type: ignore

    n = len(frame_ts_pairs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid_w = cols * thumb_w
    grid_h = rows * thumb_h
    grid = PILImage.new("RGB", (grid_w, grid_h), color=(20, 20, 20))

    timestamps: List[float] = []
    for idx, (ts, frame_bgr) in enumerate(frame_ts_pairs):
        timestamps.append(ts)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cell = PILImage.fromarray(rgb).resize((thumb_w, thumb_h - 20), PILImage.LANCZOS)
        row, col = divmod(idx, cols)
        x, y = col * thumb_w, row * thumb_h
        grid.paste(cell, (x, y))
        # Draw timestamp label
        draw = ImageDraw.Draw(grid)
        draw.rectangle([x, y + thumb_h - 20, x + thumb_w, y + thumb_h],
                       fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_h - 17), f"#{idx+1} {ts:.1f}s",
                  fill=(255, 255, 0))

    buf = io.BytesIO()
    # Limit max grid dimension to keep API cost low
    max_dim = 2048
    if max(grid_w, grid_h) > max_dim:
        scale = max_dim / max(grid_w, grid_h)
        grid = grid.resize((int(grid_w * scale), int(grid_h * scale)),
                            PILImage.LANCZOS)
    grid.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode(), timestamps


def _vision_grid_ocr(
    frame_ts_pairs: List[Tuple[float, np.ndarray]],
) -> List[Tuple[float, str]]:
    """Send a grid image to gpt-4o-mini vision and parse per-frame text."""
    import litellm  # type: ignore

    grid_b64, timestamps = _build_grid_image(frame_ts_pairs)
    n = len(timestamps)

    prompt = (
        f"This image is a grid of {n} video frames, each labeled with a frame number "
        f"(#1, #2, ...) and timestamp (e.g., '12.5s'). "
        "For EVERY frame that contains visible text (signs, scoreboards, captions, labels, "
        "numbers, titles, banners), extract the text. "
        "Format your response as:\n"
        "Frame #N (at Xs): <extracted text>\n"
        "If a frame has no visible text, skip it. "
        "Be precise — copy text exactly as written."
    )

    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key  = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    model_id = os.getenv("VISUAL_ANSWERER_MODEL", "gpt-4o-mini")
    model    = f"openai/{model_id}" if api_base else model_id

    resp = litellm.completion(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{grid_b64}", "detail": "low"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=800,
        api_base=api_base,
        api_key=api_key,
        timeout=45,
    )
    raw = resp.choices[0].message.content.strip()
    logger.info("[OCR] Vision grid API: %d chars returned", len(raw))

    # Parse "Frame #N (at Xs): text" format
    results: List[Tuple[float, str]] = []
    for line in raw.splitlines():
        m = re.match(r"Frame\s*#?(\d+)\s*\(at\s*([\d.]+)s\)\s*:\s*(.+)", line.strip(), re.I)
        if m:
            idx = int(m.group(1)) - 1
            ts = float(m.group(2)) if m.group(2) else (timestamps[idx] if idx < len(timestamps) else 0.0)
            text = m.group(3).strip()
            if text:
                results.append((ts, text))
        else:
            # Fallback: plain text lines (not timestamped)
            stripped = line.strip()
            if stripped and not stripped.startswith("Frame"):
                results.append((0.0, stripped))
    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(
    raw_entries: List[Tuple[float, str]],
    ratio: float = _DEDUP_RATIO,
) -> List[str]:
    """Merge near-duplicate text lines. Returns deduplicated list."""
    if not raw_entries:
        return []

    seen: List[str] = []
    for _, text in raw_entries:
        text = text.strip()
        if not text or len(text) < 2:
            continue
        is_dup = any(
            SequenceMatcher(None, text.lower(), s.lower()).ratio() >= ratio
            for s in seen
        )
        if not is_dup:
            seen.append(text)

    return seen


# ---------------------------------------------------------------------------
# TF-IDF RAG filter
# ---------------------------------------------------------------------------

def _tfidf_filter(
    question: str,
    ocr_lines: List[str],
    top_k: int = _TOP_K_LINES,
    min_sim: float = _MIN_SIM_THRESHOLD,
) -> Tuple[List[str], float]:
    """
    Rank OCR lines by TF-IDF cosine similarity to the question.
    Returns (top_k_lines, best_similarity_score).
    Lines below min_sim are excluded (prevents garbage injection).
    """
    if not ocr_lines:
        return [], 0.0
    if len(ocr_lines) <= top_k:
        # Still compute similarity to check if we should no_match
        best = _best_keyword_overlap(question, ocr_lines)
        return (ocr_lines if best >= min_sim else []), best

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity as cs  # type: ignore

        corpus = [question] + ocr_lines
        vec = TfidfVectorizer(min_df=1, stop_words="english", sublinear_tf=True)
        tfidf = vec.fit_transform(corpus)
        sims = cs(tfidf[0:1], tfidf[1:]).flatten()
        ranked = sorted(zip(sims, ocr_lines), reverse=True)
        best_sim = float(ranked[0][0]) if ranked else 0.0
        filtered = [(sim, line) for sim, line in ranked if sim >= min_sim]
        return [line for _, line in filtered[:top_k]], best_sim

    except ImportError:
        # Fallback: simple keyword overlap ratio
        score, top = _best_keyword_overlap_ranked(question, ocr_lines, top_k, min_sim)
        return top, score


def _best_keyword_overlap(question: str, lines: List[str]) -> float:
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOPWORDS
    if not q_tokens:
        return 0.0
    best = 0.0
    for line in lines:
        line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
        overlap = len(q_tokens & line_tokens) / max(1, len(q_tokens))
        best = max(best, overlap)
    return best


def _best_keyword_overlap_ranked(
    question: str,
    lines: List[str],
    top_k: int,
    min_sim: float,
) -> Tuple[float, List[str]]:
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - _STOPWORDS
    if not q_tokens:
        return 0.0, []
    scored = []
    for line in lines:
        line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
        overlap = len(q_tokens & line_tokens) / max(1, len(q_tokens))
        scored.append((overlap, line))
    scored.sort(reverse=True)
    best = scored[0][0] if scored else 0.0
    filtered = [line for score, line in scored[:top_k] if score >= min_sim]
    return best, filtered


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "for", "of", "in",
    "on", "and", "or", "with", "that", "this", "it", "be", "as", "at", "by",
    "from", "not", "what", "which", "how", "when", "where", "does", "did",
    "do", "video", "show", "shown", "during",
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    video_path = request.video_path
    question   = request.question or ""
    start_time = request.start_time or 0.0
    end_time   = request.end_time

    # ── 1. Sample frames at 1fps ──────────────────────────────────────────
    frame_ts_pairs = _sample_frames_1fps(
        video_path, start_time=start_time, end_time=end_time,
        max_frames=_MAX_GRID_FRAMES,
    )
    if not frame_ts_pairs:
        return SkillResponse(
            skill_name="ocr",
            summary="Could not extract frames from video.",
            artifacts={"no_match": True},
        )

    logger.info("[OCR] Sampled %d frames (1fps, max=%d)", len(frame_ts_pairs), _MAX_GRID_FRAMES)

    # ── 2. Extract text ────────────────────────────────────────────────────
    raw_entries: List[Tuple[float, str]] = []

    # Try local OCR first
    local_available = (_get_easyocr() is not None) or (_get_paddleocr() is not None)
    if local_available:
        for ts, frame_bgr in frame_ts_pairs:
            texts = _local_ocr_frame(frame_bgr)
            for t in texts:
                raw_entries.append((ts, t))
        logger.info("[OCR] Local OCR: %d raw text entries from %d frames",
                    len(raw_entries), len(frame_ts_pairs))

    # Fall back to vision API grid if local produced nothing usable
    if not raw_entries and _vision_api_available():
        logger.info("[OCR] Falling back to vision API grid OCR (%d frames)", len(frame_ts_pairs))
        try:
            raw_entries = _vision_grid_ocr(frame_ts_pairs)
        except Exception as exc:
            logger.warning("[OCR] Vision API grid OCR failed: %s", exc)

    if not raw_entries:
        return SkillResponse(
            skill_name="ocr",
            summary="No text found in video frames.",
            artifacts={"no_match": True},
        )

    # ── 3. Deduplicate ─────────────────────────────────────────────────────
    deduped = _deduplicate(raw_entries)
    logger.info("[OCR] After dedup: %d unique text lines (from %d raw)", len(deduped), len(raw_entries))

    # ── 4. TF-IDF RAG filter ───────────────────────────────────────────────
    relevant_lines, best_sim = _tfidf_filter(question, deduped,
                                             top_k=_TOP_K_LINES,
                                             min_sim=_MIN_SIM_THRESHOLD)
    logger.info("[OCR] TF-IDF filter: best_sim=%.3f, kept %d/%d lines",
                best_sim, len(relevant_lines), len(deduped))

    if not relevant_lines:
        return SkillResponse(
            skill_name="ocr",
            summary=(
                f"OCR found {len(deduped)} text segments but none are relevant to the question "
                f"(best similarity={best_sim:.3f}, threshold={_MIN_SIM_THRESHOLD})."
            ),
            artifacts={
                "no_match": True,
                "total_extracted": len(deduped),
                "best_similarity": round(best_sim, 3),
            },
        )

    # ── 5. Format output ───────────────────────────────────────────────────
    evidence = "\n".join(f"• {line}" for line in relevant_lines)
    summary = (
        f"OCR found relevant text (similarity={best_sim:.2f}):\n{evidence}"
    )
    return SkillResponse(
        skill_name="ocr",
        summary=summary,
        artifacts={
            "ocr_lines": relevant_lines,
            "total_extracted": len(deduped),
            "best_similarity": round(best_sim, 3),
            "frames_sampled": len(frame_ts_pairs),
        },
    )
