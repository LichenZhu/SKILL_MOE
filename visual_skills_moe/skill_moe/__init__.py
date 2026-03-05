"""
Skill-MoE skeleton package.

Lightweight, research-oriented framework for routing video questions to
specialized skills (OCR, ASR, motion tracking, etc.) instead of running a
monolithic model on every frame.
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
