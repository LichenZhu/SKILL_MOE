from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class SkillMetadata:
    """Parsed from SKILL.md front matter."""

    name: str
    description: str
    path: str
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None


@dataclass
class SkillRequest:
    """Envelope passed into runners."""

    question: str
    video_path: str
    video_duration: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_window(self) -> tuple[Optional[float], Optional[float]]:
        """Return a valid [start, end] time window, or (None, None)."""
        start = self.start_time
        end = self.end_time

        if start is None and end is None:
            return None, None

        # Clamp to valid range when duration is known.
        if self.video_duration and self.video_duration > 0:
            if start is not None:
                start = max(0.0, min(start, self.video_duration))
            if end is not None:
                end = max(0.0, min(end, self.video_duration))
        else:
            if start is not None:
                start = max(0.0, start)
            if end is not None:
                end = max(0.0, end)

        if start is not None and end is not None and end <= start:
            return None, None
        return start, end


@dataclass
class SkillResponse:
    """Result produced by a skill runner."""

    skill_name: str
    summary: str
    content: str = ""  # Optional unstructured evidence text from the skill.
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def evidence_text(self) -> str:
        """Best-effort evidence text for prompting."""
        return (self.content or self.summary or "").strip()


# ---------------------------------------------------------------------------
# Iterative reasoning (ReAct) data structures
# ---------------------------------------------------------------------------

class ActionType(Enum):
    CALL_SKILL = "CALL_SKILL"
    FINISH = "FINISH"


@dataclass
class RouterDecision:
    """A single decision produced by the iterative router."""

    action: ActionType
    skill_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""



@dataclass
class ReasoningStep:
    """One turn in the reasoning trace: a decision and its result."""

    step: int
    decision: RouterDecision
    response: Optional[SkillResponse] = None


@dataclass
class ReasoningTrace:
    """Accumulates the full multi-turn reasoning trace for one query."""

    question: str
    video_duration: float = 0.0
    steps: List[ReasoningStep] = field(default_factory=list)
    # Phase-1 answer from Video LLM (before skills) for with/without comparison.
    initial_answer: str = ""
    initial_confidence: str = ""  # "high", "medium", "low"
    # Final answer after skill augmentation (set by pipeline).
    final_answer: str = ""

    @property
    def responses(self) -> List[SkillResponse]:
        """All skill responses collected so far (convenience accessor)."""
        return [s.response for s in self.steps if s.response is not None]

    @property
    def executed_skills(self) -> List[str]:
        """Names of skills already executed."""
        return [s.decision.skill_name for s in self.steps
                if s.decision.action == ActionType.CALL_SKILL and s.decision.skill_name]

    def history_text(self) -> str:
        """Concise text summary of the trace for the router prompt."""
        if not self.steps:
            return "No skills have been executed yet."
        lines: list[str] = []
        for s in self.steps:
            d = s.decision
            if d.action == ActionType.CALL_SKILL:
                start = d.parameters.get("start_time")
                end = d.parameters.get("end_time")
                try:
                    start_f = float(start) if start is not None else None
                except Exception:
                    start_f = None
                try:
                    end_f = float(end) if end is not None else None
                except Exception:
                    end_f = None
                if start_f is not None and end_f is not None:
                    window = f" [{start_f:.1f}-{end_f:.1f}s]"
                elif start_f is not None:
                    window = f" [{start_f:.1f}s-]"
                elif end_f is not None:
                    window = f" [0.0-{end_f:.1f}s]"
                else:
                    window = ""
                result = s.response.evidence_text() if s.response else "(no result)"
                low_note = ""
                if s.response and isinstance(s.response.artifacts, dict):
                    if s.response.artifacts.get("low_confidence"):
                        reason = str(s.response.artifacts.get("low_confidence_reason") or "unspecified")
                        low_note = f"\n  Confidence: LOW ({reason})"
                lines.append(
                    f"Step {s.step}: [Thought] {d.thought}\n"
                    f"  Action: CALL_SKILL({d.skill_name}){window}\n"
                    f"  Result: {result}{low_note}"
                )
            else:
                lines.append(f"Step {s.step}: Action: FINISH")
        return "\n".join(lines)
