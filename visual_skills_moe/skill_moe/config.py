from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class RouterConfig(BaseModel):
    strategy: str = "auto"  # "llm", "rules", or "auto" (try llm, fall back to rules)
    model: str = "gpt-5.2-codex"
    max_tokens: int = 220


class AnswererConfig(BaseModel):
    model: str = "gpt-5.2-codex"
    max_tokens: int = 256


class VideoLLMConfig(BaseModel):
    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-Omni-7B"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_frames: int = 64
    total_pixels: int = 20_971_520  # ~26,800 visual tokens, safe within 32K context
    use_audio: bool = False


class VerifierConfig(BaseModel):
    enabled: bool = False        # opt-in; set true to gate skill evidence quality
    model: str = "gpt-5.2-codex"
    max_tokens: int = 10         # only YES/NO needed
    timeout_s: float = 8.0       # per-call timeout; verifier skipped on timeout


class PipelineConfig(BaseModel):
    skills_root: str = "skills"
    max_turns: int = 5  # max active-perception turns before forced FINISH
    router: RouterConfig = RouterConfig()
    answerer: AnswererConfig = AnswererConfig()
    video_llm: VideoLLMConfig = VideoLLMConfig()
    verifier: VerifierConfig = VerifierConfig()


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    """Load config from YAML file if it exists, otherwise return defaults."""
    p = Path(path)
    if not p.is_file():
        return PipelineConfig()
    try:
        import yaml  # optional dep
    except ImportError:
        return PipelineConfig()
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PipelineConfig(**data)
