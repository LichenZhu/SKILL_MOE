"""
Skill-MoE skeleton package.

Lightweight, research-oriented framework for video multiple-choice QA.
Routes questions to specialized skills (ASR, OCR, tracking, CLIP-based scoring,
focus_vqa, etc.) and uses a local VLM (Qwen2.5-Omni-7B) for baseline answering
and evidence-augmented re-answering.
"""

from .base import (
    ActionType,
    ReasoningStep,
    ReasoningTrace,
    RouterDecision,
    SkillMetadata,
    SkillRequest,
    SkillResponse,
)
from .config import PipelineConfig, RouterConfig, AnswererConfig, VideoLLMConfig, load_config
from .registry import SkillRegistry
from .router import SkillRouter
from .pipeline import VideoUnderstandingPipeline
from .answerer import answer

__all__ = [
    "ActionType",
    "ReasoningStep",
    "ReasoningTrace",
    "RouterDecision",
    "SkillRequest",
    "SkillResponse",
    "SkillMetadata",
    "PipelineConfig",
    "RouterConfig",
    "AnswererConfig",
    "VideoLLMConfig",
    "load_config",
    "SkillRegistry",
    "SkillRouter",
    "VideoUnderstandingPipeline",
    "answer",
]
