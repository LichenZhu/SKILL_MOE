"""
Agentic router for Skill-MoE (Active Perception Loop / ReAct).

The router decides one action at a time:
1) CALL_SKILL with parameters (e.g. start/end time), or
2) FINISH when enough evidence exists.

The primary interface is ``decide_next_step(trace)``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Sequence

from .base import (
    ActionType,
    ReasoningTrace,
    RouterDecision,
    SkillMetadata,
)
from .llm_clients import LLMClient, default_llm_client
from .registry import SkillRegistry

logger = logging.getLogger(__name__)
_MIN_ROUTER_WINDOW_SEC = max(1.0, float(os.getenv("ROUTER_MIN_WINDOW_SEC", "5.0")))


class SkillRouter:
    """
    Sequential decision router.

    Strategy:
    - ``llm``: only LLM decisions.
    - ``rules``: only regex-based decision on the first turn.
    - ``auto``: LLM first, fallback to rules when LLM output is unavailable.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        llm_client: LLMClient | None = None,
        strategy: str = "auto",
        llm_max_tokens: int = 220,
    ) -> None:
        self.registry = registry
        self.strategy = (strategy or "auto").strip().lower()
        if self.strategy not in {"auto", "llm", "rules"}:
            logger.warning("Unknown router strategy '%s'; using 'auto'.", strategy)
            self.strategy = "auto"

        self.llm_max_tokens = max(1, llm_max_tokens)
        if llm_client is not None:
            self.llm_client = llm_client
        elif self.strategy in {"auto", "llm"}:
            self.llm_client = default_llm_client()
        else:
            self.llm_client = None

        # First-turn fallback rules (used when LLM router is unavailable).
        # Maps skill name → list of regex patterns to match against lowercased question.
        # Only lists active skills; deleted skills (action/motion/scene/face/spatial/
        # shape_count/object_detect/frame_vqa) have been removed.
        self._rules: Dict[str, list[str]] = {
            "rag_asr": [
                r"\bsay\b", r"\bsaid\b", r"\bspoken\b", r"\bspeech\b",
                r"\bwords\b", r"\bdialogue\b", r"\bnarrat", r"\bvoice\b",
                r"\bmentioned\b", r"\btells?\b", r"\btalk", r"\bannounce",
                r"\bcheer", r"\bapplaud",
                r"说了什么", r"旁白", r"解说",
            ],
            "ocr": [
                r"\btext\b", r"\bread\b", r"\bplate\b",
                r"\bsign\b", r"\bdisplay", r"\bwritten\b",
                r"\bcaption\b", r"\bsubtitle\b", r"\btitle\b", r"\blabel\b",
                r"\bprice\b", r"\bscore\b", r"\bclock\b", r"\btimer\b",
                r"\bscoreboard\b", r"\bjersey\b", r"\bodometer\b",
                r"车牌", r"文字", r"字幕",
            ],
            "tracking": [
                r"\bhow many\b", r"\bcount\b", r"\bnumber of\b",
                r"\blargest number\b", r"\bmaximum number\b", r"\bexact number\b",
                r"有几个", r"多少个",
            ],
            "grounding": [
                r"\bwhich (?:item|object|tool)\b", r"\bwhat (?:item|object|tool)\b",
                r"\bwhat is the (?:blue|red|green|yellow|white|black)\b",
                r"\bshows? up\b", r"\bappears? as\b",
            ],
            "focus_vqa": [
                r"\bwritten on\b", r"\btext on\b", r"\bsays on\b",
                r"\blogo (?:on|of)\b", r"\bbrand\b",
                r"\bholding\b", r"\bheld in\b", r"\bin his hand\b", r"\bin her hand\b",
                r"\bwhat is (?:he|she|they) holding\b",
                r"\bcolor of the\b", r"\bcolour of the\b",
            ],
            "visual_option_match": [
                r"\bwear(?:ing)?\b", r"\bclothing\b", r"\boutfit\b",
                r"\bwhat is .* wearing\b", r"\bwho is .* (?:in|wearing)\b",
                r"\bdressed in\b", r"\bclothed in\b",
                r"穿着", r"衣服",
            ],
        }

    # ------------------------------------------------------------------
    # Agentic interface
    # ------------------------------------------------------------------

    def decide_next_step(self, trace: ReasoningTrace) -> RouterDecision:
        """Decide next step based on question + history."""
        if self.strategy == "rules":
            return self._decide_with_rules_or_finish(trace)

        decision = self._decide_with_llm(trace)
        if decision is not None:
            return decision

        if self.strategy == "llm":
            return RouterDecision(
                action=ActionType.FINISH,
                thought="LLM strategy selected but no parsable router output.",
            )
        return self._decide_with_rules_or_finish(trace)

    def _decide_with_rules_or_finish(self, trace: ReasoningTrace) -> RouterDecision:
        # Rules only run on the first step to remain sequential.
        if trace.steps:
            return RouterDecision(
                action=ActionType.FINISH,
                thought="Rule router only applies to the first step.",
            )
        return self._decide_with_rules(trace.question, trace.video_duration)

    def _decide_with_llm(self, trace: ReasoningTrace) -> RouterDecision | None:
        if not self.llm_client:
            return None
        skills = list(self.registry)
        if not skills:
            return RouterDecision(action=ActionType.FINISH, thought="No skills available.")

        prompt = self._build_react_prompt(trace, skills)
        try:
            raw = self.llm_client.complete(prompt, max_tokens=self.llm_max_tokens)
        except Exception as exc:
            logger.warning("LLM router call failed: %s", exc)
            return None

        decision = self._parse_react_response(raw, skills, trace.video_duration)
        logger.info(
            "Router decision: %s skill=%s params=%s",
            decision.action.value,
            decision.skill_name,
            decision.parameters,
        )
        return decision

    def _decide_with_rules(self, question: str, duration: float = 0.0) -> RouterDecision:
        q = question.lower()
        is_counting = self._is_counting_question(q)
        for skill_name, patterns in self._rules.items():
            if skill_name not in self.registry.list():
                continue
            if any(re.search(pat, q) for pat in patterns):
                params = {}
                if duration > 0:
                    if is_counting and skill_name in {"tracking", "temporal_action_counter"}:
                        params = {"start_time": 0.0, "end_time": duration}
                    else:
                        params = {"start_time": 0.0, "end_time": min(duration, 12.0)}
                return RouterDecision(
                    action=ActionType.CALL_SKILL,
                    skill_name=skill_name,
                    parameters=params,
                    thought=f"Rule match for skill '{skill_name}'.",
                )
        return RouterDecision(action=ActionType.FINISH, thought="No rule matched.")

    # ------------------------------------------------------------------
    # Prompt / parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_react_prompt(trace: ReasoningTrace, skills: Sequence[SkillMetadata]) -> str:
        lines: list[str] = []
        for s in skills:
            tags = ", ".join(s.tags) if s.tags else "none"
            line = f"- {s.name}: {s.description} (tags: {tags})"
            if s.when_to_use:
                line += f" | when_to_use: {s.when_to_use}"
            lines.append(line)

        duration_text = (
            f"{trace.video_duration:.2f} seconds" if trace.video_duration and trace.video_duration > 0 else "unknown"
        )
        history = trace.history_text()
        executed = ", ".join(trace.executed_skills) if trace.executed_skills else "none"

        return (
            "You are the Brain of an Active Perception Loop for video QA.\n"
            "At each turn, inspect the question and current trace, then decide ONE next action.\n\n"
            "Inputs you must use:\n"
            f"- User Question: {trace.question}\n"
            f"- Video Duration: {duration_text}\n"
            f"- Executed Skills: {executed}\n"
            f"- Trace History:\n{history}\n\n"
            "Available skills:\n"
            + "\n".join(lines)
            + "\n\nDecision policy:\n"
            "- Choose CALL_SKILL for exactly one skill when additional evidence is needed.\n"
            "- If evidence is sufficient, choose FINISH.\n"
            "- For counting/tracking tasks, always pass start_time=0 end_time=video_duration.\n\n"
            "Skill selection guide:\n"
            "- rag_asr: speech/dialogue questions ('what was said', 'mentioned', 'narrated', 'announced').\n"
            "  NEVER use for timestamp-anchored questions ('at the end', 'first appears', 'last shown') —\n"
            "  ASR captures the ENTIRE video and misleads on moment-specific questions. Use FINISH instead.\n"
            "- ocr: on-screen text/numbers ('what does the sign say', 'what time', 'jersey number',\n"
            "  'score', 'license plate', 'scoreboard', 'odometer').\n"
            "- tracking: entity-count questions ('how many people', 'total athletes', 'number of cars').\n"
            "  Uses YOLOv8+ByteTrack. Full video duration required.\n"
            "  DO NOT use for: action-event counting (→ temporal_action_counter), or scoreboard\n"
            "  numbers (→ ocr), or abstract/non-COCO objects.\n"
            "- temporal_action_counter: action-event-frequency questions ('how many times did X happen',\n"
            "  'how many tricks are performed', 'how many dives/jumps/flips'). Uses CLIP frame transitions.\n"
            "- grounding: identify specific object by visual appearance ('which tool', 'what is the blue\n"
            "  item', 'what object shows up'). Returns bounding-box detections.\n"
            "- focus_vqa: fine-grained visual detail ('what is written on', 'what brand/logo',\n"
            "  'what is the person holding'). Crops + vision LLM. Supersedes grounding and ocr.\n"
            "- visual_option_match: attribute-constrained counting ('how many dressed in yellow',\n"
            "  'who is wearing red'). Scores MCQ options against CLIP keyframes.\n"
            "- zero_shot_identity: person identity ('who is the person with X').\n\n"
            "Output strictly as JSON object with this schema:\n"
            "{\n"
            '  "thought": "short reasoning",\n'
            '  "action": "CALL_SKILL" or "FINISH",\n'
            '  "skill_name": "ocr",\n'
            '  "parameters": {"start_time": 0, "end_time": 5}\n'
            "}\n"
            "For FINISH, set skill_name to empty string and parameters to {}.\n"
            "Return JSON only, no markdown."
        )

    @staticmethod
    def _is_counting_question(question: str) -> bool:
        q = (question or "").lower()
        if any(k in q for k in ("how many", "total number", "count", "number of")):
            return True
        return any(
            k in q
            for k in (
                "largest number",
                "maximum number",
                "exact number",
                "correctly states the number",
            )
        )

    @staticmethod
    def _extract_json_blob(raw: str) -> str | None:
        text = (raw or "").strip()
        if not text:
            return None

        # Direct JSON.
        if text.startswith("{") and text.endswith("}"):
            return text

        # Fenced markdown JSON.
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()

        # First object-like span.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1].strip()
        return None

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return None

    @classmethod
    def _normalize_parameters(cls, params: object, duration: float) -> Dict[str, object]:
        if not isinstance(params, dict):
            return {}

        normalized: Dict[str, object] = {}
        start = cls._safe_float(params.get("start_time", params.get("start")))
        end = cls._safe_float(params.get("end_time", params.get("end")))

        if start is not None:
            start = max(0.0, start)
        if end is not None:
            end = max(0.0, end)

        if duration and duration > 0:
            if start is not None:
                start = min(start, duration)
            if end is not None:
                end = min(end, duration)

        # Stabilize tiny or degenerate windows to avoid repeated near-zero clips.
        if start is not None and end is None:
            end = start + _MIN_ROUTER_WINDOW_SEC
            if duration and duration > 0:
                end = min(end, duration)

        if start is not None and end is not None and (end - start) < _MIN_ROUTER_WINDOW_SEC:
            end = start + _MIN_ROUTER_WINDOW_SEC
            if duration and duration > 0:
                end = min(end, duration)

        if start is not None and end is not None and end <= start:
            start, end = None, None

        if start is not None:
            normalized["start_time"] = round(start, 3)
        if end is not None:
            normalized["end_time"] = round(end, 3)

        # Keep optional additional parameters for compatible skills.
        passthrough_keys = ("target_classes", "candidate_labels")
        for key in passthrough_keys:
            if key in params:
                normalized[key] = params[key]

        return normalized

    @classmethod
    def _parse_react_json(
        cls,
        payload: dict,
        valid_names: set[str],
        duration: float,
    ) -> RouterDecision:
        thought = str(payload.get("thought", "")).strip()
        action = str(payload.get("action", "FINISH")).strip().upper()
        skill_name = str(payload.get("skill_name", "")).strip().lower()
        params = cls._normalize_parameters(payload.get("parameters", {}), duration)

        if action == ActionType.FINISH.value:
            return RouterDecision(action=ActionType.FINISH, thought=thought, parameters={})

        if action != ActionType.CALL_SKILL.value:
            return RouterDecision(
                action=ActionType.FINISH,
                thought=thought or f"Unknown action '{action}', defaulting to FINISH.",
            )

        if skill_name not in valid_names:
            # Fuzzy fallback: any known skill name embedded in thought.
            hay = f"{skill_name} {thought}".lower()
            for vn in valid_names:
                if vn in hay:
                    skill_name = vn
                    break

        if skill_name not in valid_names:
            return RouterDecision(
                action=ActionType.FINISH,
                thought=thought or "Invalid skill_name in router output.",
            )

        return RouterDecision(
            action=ActionType.CALL_SKILL,
            skill_name=skill_name,
            parameters=params,
            thought=thought,
        )

    @classmethod
    def _parse_react_response(
        cls,
        raw: str,
        skills: Sequence[SkillMetadata],
        duration: float,
    ) -> RouterDecision:
        valid_names = {s.name.lower() for s in skills}
        blob = cls._extract_json_blob(raw)
        if blob:
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    return cls._parse_react_json(parsed, valid_names, duration)
            except Exception:
                pass

        # Soft fallback for old Thought/Action format.
        thought_match = re.search(r"[Tt]hought:\s*(.+?)(?=\n\s*Action:|\Z)", raw or "", re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if re.search(r"Action:\s*FINISH", raw or "", re.IGNORECASE):
            return RouterDecision(action=ActionType.FINISH, thought=thought)

        skill_match = re.search(r"CALL_SKILL\(\s*([a-z_]+)\s*\)", raw or "", re.IGNORECASE)
        if skill_match:
            skill_name = skill_match.group(1).lower()
            if skill_name in valid_names:
                return RouterDecision(
                    action=ActionType.CALL_SKILL,
                    skill_name=skill_name,
                    thought=thought or "Parsed from legacy Action format.",
                )

        return RouterDecision(
            action=ActionType.FINISH,
            thought=thought or "Could not parse router output; defaulting to FINISH.",
        )

