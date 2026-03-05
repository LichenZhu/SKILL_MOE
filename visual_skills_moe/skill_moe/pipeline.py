"""
Video-understanding pipeline – Skill-MoE architecture.

    Question + Video
         ↓
    Router (rule-based skill selection)
         ↓
    Selected Skills run in parallel (ASR, OCR, tracking, grounding, …)
         ↓
    Video LLM answers with video + all skill evidence as context
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Callable, Dict, List, Optional

import cv2

from .base import (
    ActionType,
    RouterDecision,
    ReasoningStep,
    ReasoningTrace,
    SkillRequest,
    SkillResponse,
)
from .llm_clients import LLMClient
from .registry import SkillRegistry
from .router import SkillRouter
from .verifier import EvidenceVerifier

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 5


class VideoUnderstandingPipeline:
    """
    Orchestrates triage-based video understanding with optional skill augmentation.

    Runner convention: ``skills/<name>/runner.py`` implementing
    ``run(request, metadata) -> SkillResponse``.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        router: SkillRouter,
        max_turns: int = _DEFAULT_MAX_TURNS,
        video_llm: Optional[object] = None,
        verifier: Optional[EvidenceVerifier] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.max_turns = max_turns
        self.video_llm = video_llm
        self._verifier = verifier
        self._llm_client = llm_client
        self._runner_cache: Dict[str, Optional[Callable]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, request: SkillRequest) -> List[SkillResponse]:
        """Run the pipeline and return all collected SkillResponse objects."""
        trace = self.run_trace(request)
        return trace.responses

    def run_trace(self, request: SkillRequest) -> ReasoningTrace:
        """Run the pipeline, returning the complete ReasoningTrace."""
        effective_request = self._normalize_request(request)
        trace = ReasoningTrace(
            question=effective_request.question,
            video_duration=effective_request.video_duration,
        )
        if self.max_turns <= 0:
            logger.info("max_turns=%d; skipping skill loop.", self.max_turns)
            return trace

        if self.video_llm is not None:
            return self._run_trace_triage(effective_request, trace)

        # Fallback: sequential ReAct loop via the router.
        return self._run_trace_react(effective_request, trace)

    # ------------------------------------------------------------------
    # Triage pipeline (primary mode)
    # ------------------------------------------------------------------

    def _run_trace_triage(
        self,
        request: SkillRequest,
        trace: ReasoningTrace,
    ) -> ReasoningTrace:
        """Skill-MoE pipeline.

        Step 1 – Router selects skills based on question type.
        Step 2 – VLM baseline answer (no skills) runs IN PARALLEL with
                 all selected skills.
        Step 3 – If skill evidence exists, VLM answers again with video +
                 skill context → final_answer.  Otherwise final = initial.
        """
        from concurrent.futures import ThreadPoolExecutor

        # ── Step 1: Route — select skills ─────────────────────────
        selected_skills = self._route_skills(request.question)
        logger.info(
            "Router selected skills: %s for question: %s",
            selected_skills or "(none)", request.question[:80],
        )

        # ── Step 1b: Cross-Modal Disambiguation ────────────────────
        # Cheap LLM check: when ASR is selected, verify the trigger phrase
        # isn't actually describing a visual action (e.g., "says no" → head shake).
        # Returns {skill_name: hint_text} for any flagged ambiguities.
        _disambig_hints = self._cross_modal_disambiguate(request.question, selected_skills)

        # ── Step 2: VLM baseline + skills in parallel ─────────────
        baseline_answer: str = ""
        responses: list[SkillResponse] = []

        if selected_skills:
            # Run VLM baseline and skills concurrently.
            with ThreadPoolExecutor(max_workers=2) as pool:
                vlm_future = pool.submit(
                    self.video_llm.answer,
                    question=request.question,
                    video_path=request.video_path,
                )
                skills_future = pool.submit(
                    self._execute_skills_parallel,
                    selected_skills,
                    request,
                )
                baseline_answer = vlm_future.result()
                responses = skills_future.result()
        else:
            # No skills selected — single VLM call is the final answer.
            baseline_answer = self.video_llm.answer(
                question=request.question,
                video_path=request.video_path,
            )

        trace.initial_answer = baseline_answer

        # ── Step 2b: Evidence verification (optional) ─────────────
        if self._verifier and responses:
            responses = self._verify_responses(request.question, responses)

        for resp in responses:
            trace.steps.append(ReasoningStep(
                step=len(trace.steps) + 1,
                decision=RouterDecision(
                    action=ActionType.CALL_SKILL,
                    skill_name=resp.skill_name,
                    parameters={"start_time": request.start_time,
                                "end_time": request.end_time},
                    thought=f"Router selected: {resp.skill_name}",
                ),
                response=resp,
            ))

        # ── Step 3: VLM answers with evidence (if any) ────────────
        evidence_text  = self._build_evidence_text(responses, disambig_hints=_disambig_hints)
        visual_crops   = self._extract_visual_evidence(responses)

        if visual_crops:
            # Spatio-Temporal Attention path: feed high-res crops DIRECTLY into
            # the VLM alongside the video — no text conversion, no info loss.
            target_desc = self._extract_crop_target(responses)
            logger.info(
                "[Pipeline] Visual-evidence path: %d crop(s) for target='%s'",
                len(visual_crops), target_desc,
            )
            final_answer = self._answer_with_visual_crops(
                request=request,
                crops_b64=visual_crops,
                target_desc=target_desc,
                skill_context=evidence_text,  # also pass any text evidence
            )
        elif evidence_text:
            logger.info(
                "VLM re-answering with %d skill(s) evidence (%d chars)",
                len(responses), len(evidence_text),
            )
            final_answer = self.video_llm.answer(
                question=request.question,
                video_path=request.video_path,
                skill_context=evidence_text,
            )
        else:
            final_answer = baseline_answer

        trace.final_answer = final_answer

        logger.info(
            "Answer: initial=%s  final=%s  (%r)",
            self._extract_letter(baseline_answer),
            self._extract_letter(final_answer),
            final_answer[:80],
        )

        trace.steps.append(ReasoningStep(
            step=len(trace.steps) + 1,
            decision=RouterDecision(
                action=ActionType.FINISH,
                thought="VLM re-answered with skill evidence."
                if evidence_text else "VLM answered directly (no skills).",
            ),
        ))

        return trace

    # ------------------------------------------------------------------
    # Fallback ReAct loop (when no video_llm)
    # ------------------------------------------------------------------

    def _run_trace_react(
        self,
        request: SkillRequest,
        trace: ReasoningTrace,
    ) -> ReasoningTrace:
        """Simple sequential routing via the external LLM router."""
        for turn in range(1, self.max_turns + 1):
            decision = self.router.decide_next_step(trace)

            if decision.action == ActionType.FINISH:
                logger.info("Turn %d: Router decided FINISH.", turn)
                trace.steps.append(ReasoningStep(step=turn, decision=decision))
                break

            skill_name = decision.skill_name
            logger.info("Turn %d: CALL_SKILL(%s)", turn, skill_name)

            resp = self._execute_skill_single(skill_name, request, decision)
            step = ReasoningStep(step=turn, decision=decision, response=resp)
            trace.steps.append(step)

            logger.info("Turn %d: %s returned: %s", turn, skill_name, resp.summary[:200])

        return trace

    def _execute_skill_single(
        self,
        skill_name: str | None,
        request: SkillRequest,
        decision: RouterDecision,
    ) -> SkillResponse:
        if not skill_name:
            return SkillResponse(
                skill_name="unknown",
                summary="Router returned CALL_SKILL without a skill name.",
            )
        meta = self.registry.get(skill_name)
        if not meta:
            return SkillResponse(
                skill_name=skill_name,
                summary=f"Skill '{skill_name}' not found in registry.",
                artifacts={"error": "skill_not_found"},
            )
        params = dict(decision.parameters or {})
        start_time = self._coerce_float(params.get("start_time", request.start_time))
        end_time = self._coerce_float(params.get("end_time", request.end_time))
        enriched = SkillRequest(
            question=request.question,
            video_path=request.video_path,
            video_duration=request.video_duration,
            start_time=start_time,
            end_time=end_time,
            metadata={**request.metadata, **params},
        )
        s, e = enriched.normalized_window()
        enriched.start_time = s
        enriched.end_time = e
        runner = self._get_runner(meta.path)
        if not runner:
            return SkillResponse(
                skill_name=skill_name,
                summary=f"No runner found for '{skill_name}'.",
            )
        try:
            return runner(enriched, meta)
        except Exception as exc:
            return SkillResponse(
                skill_name=skill_name,
                summary=f"Runner error: {exc}",
                artifacts={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Spatio-Temporal Attention helpers (visual_evidence path)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_visual_evidence(responses: list[SkillResponse]) -> list[str]:
        """Collect base64 crop strings from any response with visual_evidence."""
        crops: list[str] = []
        for resp in responses:
            arts = resp.artifacts if isinstance(resp.artifacts, dict) else {}
            if arts.get("no_match") or arts.get("error"):
                continue
            ve = arts.get("visual_evidence")
            if isinstance(ve, list):
                crops.extend(str(c) for c in ve if c)
        return crops

    @staticmethod
    def _extract_crop_target(responses: list[SkillResponse]) -> str:
        """Return the target description from the focus_vqa response."""
        for resp in responses:
            arts = resp.artifacts if isinstance(resp.artifacts, dict) else {}
            if arts.get("visual_evidence") and arts.get("target"):
                return str(arts["target"])
        return "target region"

    def _answer_with_visual_crops(
        self,
        request: SkillRequest,
        crops_b64: list[str],
        target_desc: str,
        skill_context: str | None,
    ) -> str:
        """Answer using high-res crops.

        Priority order:
          1. video_llm.answer_with_crops()  — Qwen sees video + crops (best)
          2. visual_answerer (gpt-4o vision) — crops only (API fallback)
          3. video_llm.answer()             — text evidence only (last resort)
        """
        # Free skill runner VRAM (GroundingDINO, CLIP, etc.) before Qwen does its
        # augmented forward pass — crop injection requires extra activation memory
        # and OOMs on shared GPUs if skill models remain resident.
        try:
            import gc, torch
            self.clear_caches()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Primary: Qwen with video + crops.
        if self.video_llm is not None and hasattr(self.video_llm, "answer_with_crops"):
            try:
                return self.video_llm.answer_with_crops(
                    question=request.question,
                    video_path=request.video_path,
                    crops_b64=crops_b64,
                    skill_context=skill_context,
                    target_desc=target_desc,
                )
            except Exception as exc:
                logger.warning(
                    "[Pipeline] answer_with_crops failed (%s); trying visual_answerer.", exc
                )

        # Secondary: gpt-4o-vision with crops only.
        try:
            from .visual_answerer import answer_with_crops as _va_answer
            return _va_answer(
                question=request.question,
                crops_b64=crops_b64,
                target_desc=target_desc,
                skill_context=skill_context,
            )
        except Exception as exc:
            logger.warning(
                "[Pipeline] visual_answerer failed (%s); falling back to text evidence.", exc
            )

        # Last resort: text evidence only.
        if skill_context and self.video_llm is not None:
            return self.video_llm.answer(
                question=request.question,
                video_path=request.video_path,
                skill_context=skill_context,
            )
        return request.metadata.get("baseline_answer", "")

    # ------------------------------------------------------------------
    # Evidence verification
    # ------------------------------------------------------------------

    def _verify_responses(
        self,
        question: str,
        responses: list[SkillResponse],
    ) -> list[SkillResponse]:
        """Run EvidenceVerifier on each response that has not already been discarded.

        Responses with artifacts["no_match"] or artifacts["error"] are passed
        through unchanged — they are already filtered out by _build_evidence_text.
        Responses rejected by the verifier have no_match=True injected into their
        artifacts dict so _build_evidence_text skips them.
        """
        for resp in responses:
            arts = resp.artifacts if isinstance(resp.artifacts, dict) else {}
            if arts.get("no_match") or arts.get("error"):
                continue  # already filtered; skip

            keep = self._verifier.verify(question, resp)
            if not keep:
                resp.artifacts = {**arts, "no_match": True, "verifier_rejected": True}

        return responses

    # ------------------------------------------------------------------
    # Cross-Modal Disambiguation
    # ------------------------------------------------------------------

    _DISAMBIG_PROMPT = (
        "A video-understanding system triggered the {skill} skill because the phrase "
        "'{trigger}' appeared in this question:\n\"{question}\"\n\n"
        "Could this phrase actually describe a VISUAL action or gesture rather than "
        "audible speech or motion?\n"
        "Examples: 'says no' → head shake (visual), 'nods' → visual, "
        "'waves' → waving arm (visual), 'shouts' → might be a visual explosive moment.\n\n"
        "Answer ONLY 'YES' if this is potentially visual, or 'NO' if it is clearly audio."
    )

    def _cross_modal_disambiguate(
        self,
        question: str,
        selected_skills: list[str],
    ) -> dict[str, str]:
        """Lightweight LLM check for modality ambiguity in ASR/tracking triggers.

        When 'says no', 'nods', 'waves', etc. trigger ASR, the action may actually
        be a VISUAL gesture, not speech. This method detects that and returns a
        {skill_name: warning_hint} dict to be injected into the VLM prompt.

        Fail-open: returns {} on any error or if llm_client is unavailable.
        """
        if not self._llm_client:
            return {}

        asr_skills = [s for s in selected_skills if s in ("asr", "rag_asr")]
        if not asr_skills:
            return {}

        stem = re.split(r"\n\s*A\.", question or "", maxsplit=1)[0].lower()

        # Extract the matched ASR trigger phrase and surrounding context.
        trigger = self._find_asr_trigger_context(stem)
        if not trigger:
            return {}

        skill_name = asr_skills[0]
        try:
            prompt = self._DISAMBIG_PROMPT.format(
                skill=skill_name,
                trigger=trigger,
                question=question[:200],
            )
            raw = self._llm_client.complete(prompt, max_tokens=5)
            is_visual = raw.strip().upper().startswith("YES")
        except Exception as exc:
            logger.debug("[DISAMBIG] LLM call failed (fail-open): %s", exc)
            return {}

        if not is_visual:
            return {}

        logger.info(
            "[DISAMBIG] ASR trigger '%s' flagged as potentially visual. Adding VLM hint.",
            trigger,
        )
        hint = (
            f"The phrase '{trigger}' triggered speech recognition, but this action "
            "may be a VISUAL gesture (e.g., a head shake, nod, or physical expression), "
            "NOT necessarily spoken words in the audio. "
            "Pay extreme attention to the character's physical gestures, facial expressions, "
            "and body language. Do not assume the action must be audible."
        )
        return {skill_name: hint}

    @staticmethod
    def _find_asr_trigger_context(stem: str) -> str:
        """Extract a short phrase around the ASR trigger keyword for disambiguation."""
        # Word-boundary triggers that are most likely to be visually ambiguous
        _ambiguous_boundaries = ("say", "said", "tell", "talk", "nod", "wave", "shake")
        for kw in _ambiguous_boundaries:
            m = re.search(rf'\b{kw}(s|d|ing)?\b(.{{0,30}})', stem)
            if m:
                ctx = (kw + (m.group(1) or "") + m.group(2)).strip()
                return ctx[:40]
        # Phrase triggers that appear near negation/gesture words
        _gesture_phrases = ("says no", "said no", "shakes head", "nods head", "thumbs up", "thumbs down")
        for p in _gesture_phrases:
            if p in stem:
                return p
        return ""

    # ------------------------------------------------------------------
    # Evidence text building
    # ------------------------------------------------------------------

    # Skeptical preamble — injected before all tool evidence to combat Tool Sycophancy.
    _SKEPTICAL_PREAMBLE = (
        "[TOOL EVIDENCE — CRITICAL WARNING] The following is output from automated "
        "analysis tools. These tools are often noise-sensitive and can be wrong:\n"
        "  • ASR (speech transcript) captures ALL audio including background noise, "
        "off-topic narration, and unrelated speech — it does NOT know which moment "
        "the question is about.\n"
        "  • Tracking/counting tools may mis-count, double-count, or fail on "
        "occluded objects.\n"
        "  • Ordering tools may hallucinate timestamps if the content does not "
        "visually appear on screen.\n"
        "RULE: If tool evidence contradicts what you directly observe in the video "
        "(a gesture, a facial expression, a visual event, common sense), you MUST "
        "ignore the tool and prioritize your own visual observation. Tool evidence "
        "is supplementary — never authoritative when it conflicts with what you see.\n"
    )

    @staticmethod
    def _build_evidence_text(
        responses: list[SkillResponse],
        char_limit: int = 1000,
        disambig_hints: dict[str, str] | None = None,
    ) -> str | None:
        """Format skill responses into concise evidence for prompt injection.

        Prepends:
          1. A skeptical reliability notice (combats Tool Sycophancy).
          2. Any cross-modal disambiguation hints (e.g., 'says no' may be visual).
        """
        parts: list[str] = []
        total = 0

        for resp in responses:
            arts = resp.artifacts if isinstance(resp.artifacts, dict) else {}
            if arts.get("error"):
                continue
            # Skip rag_asr NO_MATCH: no relevant speech found → don't inject confusing summary.
            if arts.get("no_match"):
                continue

            if resp.skill_name == "asr":
                ev = (arts.get("transcript") or resp.evidence_text() or "").strip()
            elif resp.skill_name == "ocr":
                ev = (resp.summary or resp.evidence_text() or "").strip()
            else:
                ev = (resp.evidence_text() or "").strip()

            if len(ev) < 10:
                continue

            max_per = 2000 if resp.skill_name == "asr" else 500
            per_skill = min(max_per, char_limit - total)
            if per_skill <= 20:
                break
            if len(ev) > per_skill:
                ev = ev[: per_skill - 3] + "..."

            part = f"[{resp.skill_name}] {ev}"
            parts.append(part)
            total += len(part)

        if not parts:
            return None

        # Build the final evidence string with skeptical preamble.
        header = VideoUnderstandingPipeline._SKEPTICAL_PREAMBLE

        # Inject cross-modal disambiguation hints after the preamble.
        if disambig_hints:
            for skill_name, hint in disambig_hints.items():
                header += f"[MODALITY ALERT — {skill_name}] {hint}\n"

        return header + "\n".join(parts)

    # ------------------------------------------------------------------
    # Answer extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_letter(answer: str) -> str:
        m = re.search(r"\b([A-D])\b", answer or "")
        return m.group(1) if m else ""

    # ------------------------------------------------------------------
    # Skill router (rule-based)
    # ------------------------------------------------------------------

    def _route_skills(self, question: str) -> list[str]:
        """Select skills to invoke based on question type.

        Returns a list of skill names to run in parallel.

        Defensive rules (applied before skill selection):
          - Negative-presence questions ("not appear", "absent", etc.) → no skills.
            Visual tools cannot reliably confirm absence and tend to hallucinate.
        """
        stem = re.split(r"\n\s*A\.", question or "", maxsplit=1)[0].lower()

        # ── Guard: negative-presence questions → no skills ──────────────────
        _negative_phrases = (
            "not appear", "does not appear", "didn't appear", "not shown",
            "not visible", "absent", "missing", "never shown", "not present",
            "not mentioned", "wasn't shown", "isn't shown", "not include",
            "never appears", "not feature",
        )
        if any(p in stem for p in _negative_phrases):
            logger.debug("_route_skills: negative-presence question → no skills")
            return []

        # ── Guard: emotion-reasoning questions → VLM bare intuition ──────────
        # "Why does she cry?", "Why is he upset?" — causal/emotional questions are
        # answered better by VLM direct observation than by any automated tool.
        # Tools add noise (ASR hears random audio, tracking counts wrong things).
        _emotion_words = (
            "cry", "cries", "crying", "weep", "sob",
            "sad", "sadness", "happy", "happiness", "joy",
            "angry", "anger", "upset", "frustrat",
            "afraid", "fear", "scared", "shock",
            "laugh", "smile", "excited", "disappoint",
            "decide", "decided", "react", "reacts", "reaction",
            "emotion", "emotional", "feel", "feels", "feeling",
            "stressed", "nervous", "anxious", "guilty",
        )
        _causal_words = ("why", "reason", "because", "what led", "what caused", "purpose")
        _is_emotion_reasoning = (
            any(k in stem for k in _causal_words)
            and any(k in stem for k in _emotion_words)
        )
        if _is_emotion_reasoning:
            logger.info("[GUARD:emotion_reasoning] emotional causal question → VLM bare intuition")
            return []

        skills: list[str] = []

        # ASR for speech/narration/reaction questions
        _asr_phrases = (
            "spoken", "narrat",
            # "introduce"/"introduced" too broad; only "introduced after" (temporal sequence) is safe
            "introduced after",
            "mention", "voice", "dialogue", "announce",
            "words", "speech", "what is the name", "company name",
            # cheering/reaction context: announcer narration explains the event
            "cheering", "cheered", "celebrating", "celebration", "cheer",
            "crowd react", "applaud", "applause",
        )
        _asr_word_boundary = ("said", "say", "talk", "tell", "hear", "heard", "sound")
        # Gesture/expression phrases that are answered visually, not by audio.
        # "says no" / "says yes" refers to a head-shake or nod, not spoken words.
        # "tells me" in context of visual demonstration is a figure of speech.
        _asr_gesture_exclusions = (
            "says no", "say no", "said no",
            "says yes", "say yes", "said yes",
            "nods", "shakes his head", "shakes her head",
            "tells me", "tell me by",
            "gestures", "points to", "signals",
            "shows by", "demonstrates by",
        )
        _is_gesture_question = any(k in stem for k in _asr_gesture_exclusions)
        if (
            not _is_gesture_question
            and (
                any(k in stem for k in _asr_phrases)
                or any(re.search(rf"\b{k}\b", stem) for k in _asr_word_boundary)
            )
        ):
            skills.append("asr")

        # OCR for text/number/score reading questions
        _ocr_phrases = (
            "text", "written", "display", "subtitle",
            "caption", "neon sign", "what time", "at what time",
            "stock price", "score of", "result of the match",
            "current score", "duration of", "number on", "lane number",
            # Implicit text: specific displayed-number/identifier queries
            "jersey number", "shirt number", "bib number",
            "what does the screen", "what does the counter",
            "what does the timer", "what does the meter",
            "odometer", "on the scoreboard", "final score",
            "episode number", "channel number",
            # Graphic/text expansion
            "signboard", "time on the clock", "on the board", "on the screen",
            "what number", "what score", "total score", "points scored",
        )
        _ocr_word_boundary = (
            "sign", "read", "title", "label", "price",
            "clock", "number does the person type",
            # "plate" removed: too broad, matches food "on the plate" — use specific phrases instead
            "license plate", "number plate",
            "scoreboard", "jersey", "odometer",
        )
        # Dual-trigger phrases: both OCR (global scan) and focus_vqa (hi-res crop) together
        _dual_ocr_focusvqa_phrases = (
            "brand name", "what brand", "what logo", "logo of the",
            "jersey number", "shirt number", "bib number",
            "on the scoreboard", "scoreboard", "signboard",
        )
        # Exclude OCR when "displayed/shown at/in the [video/start/end/beginning]" —
        # these mean "visually present in the video", not "text/number on a screen".
        _ocr_visual_exclusions = (
            "displayed at the", "displayed in the video", "displayed in the clip",
        )
        _ocr_blocked = any(k in stem for k in _ocr_visual_exclusions)
        if not _ocr_blocked and (
            any(k in stem for k in _ocr_phrases) or any(
                re.search(rf"\b{k}\b", stem) for k in _ocr_word_boundary
            )
        ):
            skills.append("ocr")

        # Dual-trigger: add focus_vqa alongside ocr for hi-res brand/jersey/scoreboard reads
        if not _ocr_blocked and any(k in stem for k in _dual_ocr_focusvqa_phrases):
            if "ocr" not in skills:
                skills.append("ocr")
            if "focus_vqa" not in skills and "focus_vqa" in self.registry.list():
                skills.append("focus_vqa")
                logger.info("[ROUTE:dual_ocr_focusvqa] brand/jersey/scoreboard → ocr+focus_vqa")

        # Grounding for fine-grained object identification questions.
        # Kept deliberately narrow: fire only when the question explicitly names
        # a physical object, tool, or item and asks to identify/locate it.
        _grounding_phrases = (
            "which item", "which object", "which tool",
            "what item", "what object", "what tool",
            "identify the object", "identify the item",
            "what is the blue", "what is the red", "what is the green",
            "what is the object", "what is the item", "what is the tool",
            "shows up", "show up", "appears as", "appear as",
            "which of the following appears", "which of the following shows",
        )
        _grounding_word_boundary = ("item", "object", "tool", "device", "equipment")
        # Temporal-visual exclusions: GroundingDINO-tiny grounds across all frames;
        # questions asking about "first/last/opening/start/end" require temporal
        # localisation that the model cannot provide → skip grounding.
        _grounding_temporal_exclusions = (
            "at the start", "at the beginning", "at the end",
            "in the beginning", "in the opening",
            "first magic", "first trick", "first scene",
            "fourth-to-last", "third-to-last", "second-to-last",
            "opening shot", "closing shot",
        )
        # Negation/necessity exclusions: grounding finds the most visible entity,
        # which is wrong for "which tool is NOT necessary" type questions.
        _grounding_negation_exclusions = (
            "not necessary", "not needed", "unnecessary", "not required",
            "not used", "not present", "not shown",
        )
        _grounding_temporal_blocked = any(k in stem for k in _grounding_temporal_exclusions)
        _grounding_negation_blocked = any(k in stem for k in _grounding_negation_exclusions)
        # Also block when a strong temporal word appears as a superlative position marker
        _grounding_positional_blocked = bool(
            re.search(r"\b(first|last)\b.{0,30}\b(object|item|tool|equipment|celestial|thing)\b", stem)
            or re.search(r"\b(object|item|tool|equipment|celestial|thing)\b.{0,40}\b(first|last)\b", stem)
        )
        if (
            (any(k in stem for k in _grounding_phrases) or any(
                re.search(rf"\b{k}\b", stem) for k in _grounding_word_boundary
            ))
            and not _grounding_temporal_blocked
            and not _grounding_negation_blocked
            and not _grounding_positional_blocked
        ):
            skills.append("grounding")

        # ── Focus VQA for fine-grained visual detail questions ────────────────
        # Fires when the question asks about a specific detail (text, logo, label,
        # what someone holds) that a globally-downsampled video frame may miss.
        # Routes to focus_vqa INSTEAD OF grounding for these patterns: focus_vqa
        # crops the high-res region and queries a vision LLM, giving much richer
        # evidence than bounding-box coordinates alone.
        _focus_vqa_phrases = (
            "written on", "write on", "text on", "says on",
            "what does the sign", "what does the label", "what does the banner",
            "what does the poster", "what does the board",
            "logo on", "logo of", "brand on", "brand name",
            "holding in", "holding a", "held in", "holds a",
            "what is in his hand", "what is in her hand", "what is in their hand",
            "what is he holding", "what is she holding", "what are they holding",
            "color of the", "colour of the",
            "what is written", "what text",
        )
        _focus_vqa_word_boundary = ("holding", "logo", "label", "sign", "banner", "badge", "brand")
        _focus_vqa_candidate = (
            any(k in stem for k in _focus_vqa_phrases)
            or any(re.search(rf"\b{k}\b", stem) for k in _focus_vqa_word_boundary)
        )
        # Don't fire focus_vqa for counting questions — those go to tracking/TAC
        _focus_vqa_excluded = bool(re.search(r"how many", stem))
        if _focus_vqa_candidate and not _focus_vqa_excluded and "focus_vqa" in self.registry.list():
            # focus_vqa supersedes grounding AND ocr for fine-grained visual questions:
            # it crops the region and queries a vision LLM which can read text directly.
            # Also remove asr/rag_asr triggered by "say" — these are visual questions, not audio.
            # ("What does the sign say?" → visual text reading, not speech recognition)
            skills = [s for s in skills if s not in ("grounding", "ocr", "asr", "rag_asr")]
            if "focus_vqa" not in skills:
                skills.append("focus_vqa")
            logger.info("[ROUTE:focus_vqa] fine-grained detail question → focus_vqa")

        # Tracking for counting questions — unique entity counts via YOLOv8+ByteTrack.
        # Excluded from: scoreboard/points questions (→ OCR instead) and "visible"
        # questions where the VLM answers directly by visual inspection.
        _counting_phrases = (
            "how many", "total number", "number of people", "number of times",
            "how many times", "count of", "how many unique", "how many different",
        )
        _counting_word_boundary = ("count",)
        _tracking_exclusions = (
            # Scoreboard-type: displayed numbers → OCR is the right tool
            "how many points", "how many goals", "score of", "how many runs",
            "how many sets", "how many games", "how many rounds",
            # "visible in the video" → VLM answers by direct visual inspection
            "visible in the video", "visible in the frame",
            # Event/action counting — sequential/repeated events, not entity presence
            "interviewed", "were interviewed", "participated",
            # NOTE: "how many times", "are performed", "how many tricks", "how many shots"
            # now handled by temporal_action_counter (Rule above). Only keep here as
            # fallback guard so tracking is never chosen even if tac unavailable.
            "are performed",        # event-count → temporal_action_counter
            "introductory shot",    # temporal: only the opening shot (063-1)
            "presenting on",        # spatial: subset of people on stage (181-1)
            # Non-COCO physical objects — tracker defaults to person, giving wrong counts
            "how many flags", "how many national flags",
            "how many spheres", "how many grenades",
            "how many tricks", "how many shots", "how many methods",
            # Abstract-entity questions — companies/brands can't be tracked visually
            "how many companies", "how many brands", "how many organizations",
            "how many countries", "how many nations", "how many teams",
            "how many steps", "how many stages", "how many episodes",
            "how many chapters", "how many sections",
            # Attribute-constrained clothing: ByteTrack tracks all persons, can't filter by attire
            "are wearing", "is wearing", "wearing a", "people wearing", "persons wearing",
            # Activity-constrained: ByteTrack counts all persons, can't filter by activity
            "eating", "eat insects", "exercising", "sitting at the table",
            "sitting at a table", "drinking", "sleeping", "running in",
            # Non-COCO abstract / measurement targets — standalone to handle word-order variants
            # (e.g. "how many timeout substitutions" has "timeout" between "many" and "substitut")
            "how many colors", "how many colours", "how many glaze",
            "how many meters", "how many metres", "how many kilometers",
            "how many points", "how many set points", "how many match points",
            "how many substitut", "substitution", "substitutions",  # "timeout substitutions"
            "how many sessions", "session", "sessions",            # "running sessions"
            "how many sports",
            "how many peppers", "how many vegetables", "how many ingredients",
            "how many minutes", "how many seconds", "how many hours",
            # "how many times does X" → event counting → TAC handles these, not tracking.
            # TAC will exclude "appear" cases; tracking must not fire on any of them.
            "how many times does",
            # Event/action nouns — these are repeated actions, not trackable entities
            "how many rallies", "how many laps", "how many passes",
            "how many serves", "how many strokes", "how many swings",
            "how many kicks", "how many catches", "how many throws",
            "how many sprints", "how many bounces", "how many spins",
            "how many pushups", "how many situps", "how many reps",
            "how many crosses", "how many shots on goal",
        )
        _is_tracking_candidate = (
            any(k in stem for k in _counting_phrases)
            or any(re.search(rf"\b{k}\b", stem) for k in _counting_word_boundary)
        )
        # Historical year filter: "in 2010", "in 1999" → factual question from
        # documentary narration, cannot be answered by tracking live video.
        _has_historical_year = bool(re.search(r"\bin (19|20)\d\d\b", stem))
        _is_tracking_excluded = (
            any(k in stem for k in _tracking_exclusions)
            or _has_historical_year
        )
        if _is_tracking_candidate and not _is_tracking_excluded:
            skills.append("tracking")
        elif _is_tracking_excluded and any(k in stem for k in ("goals", "score", "runs")):
            # Scoreboard question (goals/score/runs displayed on screen) → OCR if not already present.
            # NOTE: "points" intentionally excluded here — "how many match points/set points" are
            # observational counting questions, not scoreboard-reading tasks; OCR would not help.
            if "ocr" not in skills:
                skills.append("ocr")

        # ── Guard: temporal-visual questions → exclude ASR ───────────────────
        # ASR captures speech across the ENTIRE video. For questions that anchor
        # on a specific visual moment ("at the end", "first", "finally", etc.),
        # the globally-dominant speech content misleads the model away from what
        # it actually sees at the targeted timestamp (L1/L2/L3 Tool Sycophancy).
        # Only suppress if the question does NOT explicitly ask about audio.
        if "asr" in skills:
            _temporal_markers = (
                "at the end", "at the beginning", "at the start",
                "in the beginning", "in the first", "in the last",
                "at the end of the video", "at the beginning of the video",
                "first appear", "last appear", "finally appear",
                "introduced at the end", "introduced at the beginning",
                "first shown", "last shown", "first introduced", "last introduced",
                "first scene", "last scene", "opening scene", "closing scene",
                "end of the video", "beginning of the video", "start of the video",
            )
            _audio_keywords = ("say", "said", "speak", "spoken", "hear", "heard",
                               "sound", "cheer", "narrat", "announce", "dialogue")
            has_temporal = any(p in stem for p in _temporal_markers)
            has_audio = (
                any(re.search(rf"\b{k}\b", stem) for k in _audio_keywords)
            )
            if has_temporal and not has_audio:
                skills.remove("asr")
                logger.info(
                    "[OVERRIDE:temporal_visual_exclude_asr] removed asr — "
                    "question targets a specific visual moment; "
                    "speech transcript would mislead. Question: %s", stem[:120],
                )

        # ── ASR Offensive: "mentioned/discussed" → force ASR ─────────────────
        # Questions asking what was stated/mentioned in the video are audio questions.
        # ASR transcript evidence directly answers what was said; visual tools mislead.
        _mentioned_phrases = (
            "are mentioned", "is mentioned", "was mentioned", "were mentioned",
            "mentioned in the video", "mentioned in the clip",
            "talk about", "talks about", "talking about", "discussed",
            # "speaker"/"speakers" removed — too broad, matches visual questions
        )
        if any(k in stem for k in _mentioned_phrases):
            # Prefer rag_asr (retrieval-augmented) over vanilla asr when available.
            all_registered = self.registry.list()
            if "rag_asr" in all_registered:
                skills = ["rag_asr"]
                logger.info("[OVERRIDE:mentioned→rag_asr] forced rag_asr for dialogue/mention question")
            elif "asr" in all_registered:
                skills = ["asr"]
                logger.info("[OVERRIDE:mentioned→asr] forced asr for dialogue/mention question")
            else:
                skills = []
                logger.info("[OVERRIDE:mentioned→tools=[]] no asr available, stripping tools")

        # ── Identity questions → zero_shot_identity ──────────────────────────
        # "Who is the person with X" / "which player scored first" —
        # crop-then-CLIP person matcher.  Co-fires temporal_segment for
        # spatio-temporal identity questions ("who scores first", "who at the end").
        _identity_phrases = (
            "who is the person", "who is wearing", "who appears",
            "what role does the person", "who is standing", "who is sitting",
            "who is holding", "who is the one", "who is shown",
            # Expanded triggers for spatio-temporal identity
            "which player", "which person", "which athlete",
            "who scored", "who scores", "who wins", "who won",
            "who kicks", "who throws", "who catches",
            "who enters", "who exits", "who leaves",
            "who performs", "who demonstrates",
            "identity of",
        )
        _identity_counting_guard = bool(re.search(r"\bhow many\b", stem))
        if (
            not _identity_counting_guard
            and any(k in stem for k in _identity_phrases)
            and "zero_shot_identity" in self.registry.list()
        ):
            # Co-fire temporal_segment for time-anchored identity questions
            _identity_temporal_markers = (
                "first", "last", "at the end", "at the beginning",
                "at the start", "initially", "finally", "first to",
                "last to", "who scored first", "who wins",
            )
            _has_temporal_anchor = any(k in stem for k in _identity_temporal_markers)
            if _has_temporal_anchor and "temporal_segment" in self.registry.list():
                skills = ["zero_shot_identity", "temporal_segment"]
                logger.info(
                    "[OVERRIDE:identity→zsi+temporal_segment] co-fire for spatio-temporal identity"
                )
            else:
                skills = ["zero_shot_identity"]
                logger.info("[OVERRIDE:identity→zero_shot_identity] forced for identity question")
            return skills

        # ── Attribute/Color Offensive: fine-grained visual attributes → visual_option_match ─
        # CLIP-based option matching ranks all MCQ options against video frames —
        # works for clothing, object attributes, color, shape, and held items.
        _is_counting = (
            any(k in stem for k in _counting_phrases)
            or any(re.search(rf"\b{k}\b", stem) for k in _counting_word_boundary)
        )
        _clothing_phrases = ("dressed in", "clothed in")
        if _is_counting and any(k in stem for k in _clothing_phrases):
            skills = ["visual_option_match"]
            logger.info("[OVERRIDE:clothing→visual_option_match] forced for attribute-count question")
            return skills

        # Fine-grained visual attribute questions (non-counting) → add visual_option_match
        # "What is the man holding?", "What color is the bag?", "What is the person wearing?"
        # Kept narrow: must be about a PHYSICAL attribute (color/shape/held object/clothing).
        _vom_attribute_phrases = (
            "what is the man holding", "what is the woman holding",
            "what is the person holding", "what is he holding", "what is she holding",
            "what is he wearing", "what is she wearing", "what is the person wearing",
            "what color is", "what colour is",
            "what shape is",
        )
        _vom_attribute_word_boundary = ("colour", "color")
        # Exclude text-reading and temporal questions from VOM
        _vom_exclusions = (
            "scoreboard", "score", "text", "written", "display", "caption",
            "sign", "label", "time", "clock", "timer", "number",
            "order", "sequence", "chronological", "before", "after",
        )
        _vom_excluded = any(k in stem for k in _vom_exclusions) or _is_counting
        if (
            not _vom_excluded
            and (
                any(k in stem for k in _vom_attribute_phrases)
                or any(re.search(rf"\b{k}\b", stem) for k in _vom_attribute_word_boundary)
            )
            and "visual_option_match" in self.registry.list()
            and "visual_option_match" not in skills
        ):
            skills.append("visual_option_match")
            logger.info("[ROUTE:visual_option_match] fine-grained visual attribute question")

        # ── Action-event counting → temporal_action_counter ──────────────────
        # "How many times does X happen", "how many tricks are performed" —
        # sequential events cannot be counted by ByteTrack.
        # EXCLUDE: "how many times does X appear/appear in" — these are scene/topic
        # appearance questions (identity or content), not physical action transitions.
        # TAC uses NO→YES CLIP frame transitions, which work for physical actions
        # (tricks, dives) but fail for "X appears on screen" / identity presence.
        _action_counter_phrases = (
            "how many times",
            "are performed",
            "how many tricks",
            "how many shots",
            "how many dives", "how many jumps", "how many attempts",
            "how many flips", "how many turns",
            # Event/action nouns — repeated physical actions, not trackable entities
            "how many rallies", "how many laps", "how many passes",
            "how many serves", "how many strokes", "how many swings",
            "how many kicks", "how many catches", "how many throws",
            "how many bounces", "how many spins", "how many reps",
            "how many crosses",
        )
        _tac_appearance_exclusions = (
            "how many times does",    # + "appear" check below
            "times does.*appear",     # regex-style: handled separately
        )
        _tac_is_appearance = bool(
            re.search(r"how many times does .{1,60} appear", stem)
            or re.search(r"how many times .{1,40} appear(s)?\b", stem)
        )
        if (_is_counting and any(k in stem for k in _action_counter_phrases)
                and not _tac_is_appearance
                and "temporal_action_counter" in self.registry.list()):
            skills = ["temporal_action_counter"]
            logger.info("[OVERRIDE:action-event→temporal_action_counter] forced for action-frequency question")
            return skills

        # ── Narration-fact → rag_asr ─────────────────────────────────────────────
        # "According to the video, which of the following is correct?" — these are
        # factual retrieval questions answered by narration/speech, not by visual tools.
        _narration_fact_triggers = (
            "according to the video, which of the following is",
            "according to the video, in which",
            "according to the video, what is the main",
            "according to the video, how many",
            "according to the video, what did",
            "based on the information provided by the video",
            "based on the video, which of the following",
            "as stated in the video",
            "as mentioned in the video",
        )
        # Exclude clearly visual/spatial questions that happen to use "according to"
        _narration_visual_exclusions = (
            "appears", "visible", "is shown", "is seen", "what color",
            "which item", "which object", "which tool", "wearing",
        )
        if (
            any(k in stem for k in _narration_fact_triggers)
            and not any(k in stem for k in _narration_visual_exclusions)
        ):
            all_registered = self.registry.list()
            if "rag_asr" in all_registered:
                skills = ["rag_asr"]
                logger.info("[OVERRIDE:narration_fact→rag_asr] forced rag_asr for factual narration question")
            elif "asr" in all_registered:
                skills = ["asr"]
                logger.info("[OVERRIDE:narration_fact→asr] forced asr for factual narration question")

        # ── Ordering/Chronological → temporal_ordering ────────────────────────
        # "In what order did these happen?", "Which is listed before X?",
        # "Which happened first?" — find first appearance timestamp via CLIP.
        _ordering_phrases = (
            "in what order",
            "in which order",
            "listed before",
            "listed after",
            "happened first",
            "came first",
            "comes first",
            "chronological order",
            "chronological sequence",
            "what comes before",
            "what comes after",
            "which is first",
            "which was first",
            "what order",
            "correct sequence",
            "correct order",
            "sequence of events",
            "order of events",
            "order do the following",
            "order did the following",
        )
        # Guard: if MCQ options are sequence-permutation symbols (①②③④ or pure numeric
        # ordinal stacks like "(1)(2)(3)(4)"), CLIP cannot match these to video frames.
        # These questions need semantic symbol resolution, not raw CLIP scoring.
        # The runner.py will attempt GPT resolution; block here only when options are
        # pure symbol-ordering sequences with NO underlying physical event description.
        _full_question = question or ""
        _has_circled_numbers = bool(re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]", _full_question))
        # Detect options that are sequences of circled numbers: "②①③④" or "①③②④"
        _options_are_orderings = _has_circled_numbers and bool(
            re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]{3,}", _full_question)
        )
        if (
            any(k in stem for k in _ordering_phrases)
            and "temporal_ordering" in self.registry.list()
            and not _options_are_orderings
        ):
            skills = ["temporal_ordering"]
            logger.info("[OVERRIDE:ordering→temporal_ordering] forced for chronological ordering question")
            return skills
        if _options_are_orderings:
            logger.info("[SKIP:temporal_ordering] options are symbol-orderings; CLIP cannot match")

        # ── Temporal segment → temporal_segment ───────────────────────────────
        # Questions targeting a specific temporal window of the video.
        # Dense-sample that window + vision LLM description (additive with other skills).
        _temporal_segment_triggers = (
            "second half", "latter half", "latter part",
            "first half", "opening half",
            "last quarter", "final quarter",
            "last third", "final third",
            "last scene", "final scene", "closing scene",
            "opening scene", "first scene", "initial scene",
            "last part", "final part", "last section",
            "end of the video", "end of this video",
            "at the end", "towards the end", "near the end",
            "beginning of the video", "start of the video",
            "opening part", "at the beginning", "at the start",
            "early in the video",
        )
        # Don't add temporal_segment when the question is about text/numbers on screen —
        # OCR/focus_vqa already handle those; temporal_segment adds noisy visual narration.
        _temporal_segment_text_exclusions = (
            "score", "scoreboard", "text", "written", "sign", "caption",
            "number", "time on the clock", "jersey", "odometer",
        )
        _temporal_segment_blocked = any(k in stem for k in _temporal_segment_text_exclusions)
        if (
            any(k in stem for k in _temporal_segment_triggers)
            and not _temporal_segment_blocked
            and "temporal_segment" in self.registry.list()
            and "temporal_segment" not in skills
        ):
            skills.append("temporal_segment")
            logger.info("[ADD:temporal_segment] added dense window sampling for temporal position question")

        # ── Global ASR → RAG-ASR upgrade ────────────────────────────────────────
        # Whenever vanilla `asr` ends up in skills and `rag_asr` is registered,
        # replace it to prevent transcript sycophancy on all ASR-triggered questions.
        if "asr" in skills and "rag_asr" in self.registry.list():
            skills = [s if s != "asr" else "rag_asr" for s in skills]
            logger.info("[UPGRADE:asr→rag_asr] replaced asr with rag_asr for filtered transcript")

        # ── Semantic Meta-Router: GPT zero-shot gatekeeper ──────────────────────
        # For risky skills that tend to hurt when the VLM can answer directly,
        # ask the LLM: "Is this tool actually needed?" → block it if NO.
        _RISKY_SKILLS = {
            "grounding", "visual_option_match", "temporal_segment",
            "focus_vqa", "temporal_action_counter",
        }
        risky_present = [s for s in skills if s in _RISKY_SKILLS]
        if risky_present:
            skills = self._apply_semantic_gate(question, skills, risky_present)

        return skills

    # ------------------------------------------------------------------
    # Semantic Meta-Router
    # ------------------------------------------------------------------

    _SKILL_DESC = {
        "grounding": (
            "bounding-box object detection (GroundingDINO) — useful when the question asks to "
            "locate a specific physical object, tool, or item in a frame"
        ),
        "focus_vqa": (
            "spotlight crop tool — uses GroundingDINO to isolate the target region and "
            "feed a high-res crop to the VLM; useful for reading text/labels, identifying "
            "logos, or examining fine-grained object attributes that require zoomed-in detail"
        ),
        "visual_option_match": (
            "CLIP image-text similarity scoring of multiple-choice options against video frames — "
            "useful when the answer requires distinguishing between visually similar objects, "
            "clothing colors, or fine-grained visual attributes"
        ),
        "temporal_segment": (
            "dense frame sampling from a specific temporal window (e.g. 'second half', 'last scene') "
            "plus an LLM vision description of what happens in that window"
        ),
        "temporal_action_counter": (
            "CLIP-based event counter that counts NO→YES transition events in video frames — "
            "useful ONLY for visually distinct atomic pose actions (e.g., jumping, standing up, "
            "falling, diving) where each occurrence creates a distinct visual state change"
        ),
    }

    def _apply_semantic_gate(
        self,
        question: str,
        skills: list[str],
        risky: list[str],
    ) -> list[str]:
        """
        Ask the LLM whether each risky skill is genuinely needed.
        Block any skill the model says NO the VLM needs.
        Returns the filtered skill list.
        """
        from .llm_clients import default_llm_client
        client = default_llm_client()
        if client is None:
            return skills  # no LLM → keep all skills

        tool_lines = "\n".join(
            f"- {s}: {self._SKILL_DESC.get(s, s)}" for s in risky
        )
        prompt = (
            "A state-of-the-art multimodal video LLM (with full video access) will answer "
            "this question. Decide whether each specialized tool is genuinely needed.\n"
            f"Question: {question[:400]}\n\n"
            "=== CRITICAL BLOCKING RULES (apply FIRST, they OVERRIDE everything else) ===\n"
            "SUPERBLOCK COLOR RULE: If the question asks for the COLOR of ANY object, "
            "background, clothing, or item (e.g. 'what color is X', 'color of the laptop', "
            "'what colour') — return NO for ALL tools including temporal_segment, focus_vqa, "
            "grounding, and visual_option_match, regardless of any temporal constraint "
            "('at the beginning', 'at the end'). Color recognition is a strict zero-shot "
            "base-model task; any tool intervention destroys contextual lighting cues and "
            "causes hallucinations.\n\n"
            "ACTION TAXONOMY RULE (temporal_action_counter only): Is this an "
            "'Atomic Pose Action' (e.g., jumping, standing up, falling, diving — sudden "
            "visually distinct single-body poses) or a 'Continuous Complex Event' "
            "(e.g., rallies, game rounds, shots-on-goal, planet tours, match sequences — "
            "where the background is static and the action depends on rules or speed)? "
            "ALSO treat as Continuous Complex Event: any action involving INTERACTION "
            "between two people or objects (e.g., 'transfer the phone', 'pass the ball', "
            "'hand something to someone') or crossing a logical/spatial boundary "
            "('crossing the finish line', 'entering the room'). "
            "If it is a Continuous Complex Event → return NO for temporal_action_counter.\n\n"
            "TEMPORAL SEGMENT RULE (temporal_segment only): Return NO if the question asks "
            "for a specific IDENTITY (name of a person/thing), SPATIAL DIRECTION "
            "(which direction, left/right/up/down), or EXACT ENDING POSE/STATE. "
            "Only return YES if the question requires understanding a MACRO-NARRATIVE or "
            "sequential stages across multiple minutes of video.\n\n"
            "=== DECISION ===\n"
            "For each tool: reply YES if it provides essential evidence the VLM cannot obtain "
            "by watching the full video, or NO if the VLM can answer directly without it.\n"
            "Reply ONLY as 'TOOL_NAME: YES' or 'TOOL_NAME: NO', one per line.\n\n"
            f"Tools:\n{tool_lines}"
        )

        try:
            raw = client.complete(prompt, max_tokens=150)
            blocked: set[str] = set()
            for line in raw.strip().splitlines():
                line_l = line.lower()
                for s in risky:
                    if s in line_l and ": no" in line_l:
                        blocked.add(s)
                        logger.info("[MetaRouter] blocked '%s' (VLM can answer directly)", s)
            if blocked:
                return [s for s in skills if s not in blocked]
        except Exception as exc:
            logger.debug("[MetaRouter] failed, keeping all skills: %s", exc)

        return skills

    # ------------------------------------------------------------------
    # Parallel skill execution
    # ------------------------------------------------------------------

    def _execute_skills_parallel(
        self,
        skill_names: List[str],
        request: SkillRequest,
    ) -> List[SkillResponse]:
        """Execute multiple skills in parallel using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_one(skill_name: str) -> SkillResponse:
            meta = self.registry.get(skill_name)
            if not meta:
                return SkillResponse(
                    skill_name=skill_name,
                    summary=f"Skill '{skill_name}' not found in registry.",
                    artifacts={"error": "skill_not_found"},
                )

            params: Dict[str, object] = {}
            if request.video_duration and request.video_duration > 0:
                params["start_time"] = request.start_time if request.start_time is not None else 0.0
                params["end_time"] = request.end_time if request.end_time is not None else request.video_duration

            enriched = SkillRequest(
                question=request.question,
                video_path=request.video_path,
                video_duration=request.video_duration,
                start_time=self._coerce_float(params.get("start_time")),
                end_time=self._coerce_float(params.get("end_time")),
                metadata={**request.metadata, **params},
            )
            s, e = enriched.normalized_window()
            enriched.start_time = s
            enriched.end_time = e

            runner = self._get_runner(meta.path)
            if not runner:
                return SkillResponse(
                    skill_name=skill_name,
                    summary=f"No runner found for '{skill_name}'.",
                )
            try:
                return runner(enriched, meta)
            except Exception as exc:
                logger.warning("Skill '%s' failed: %s", skill_name, exc)
                return SkillResponse(
                    skill_name=skill_name,
                    summary=f"Runner error: {exc}",
                    artifacts={"error": str(exc)},
                )

        max_workers = min(len(skill_names), 3)
        responses: List[SkillResponse] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_name = {
                pool.submit(_run_one, name): name for name in skill_names
            }
            for future in as_completed(future_to_name):
                try:
                    resp = future.result()
                except Exception as exc:
                    name = future_to_name[future]
                    resp = SkillResponse(
                        skill_name=name,
                        summary=f"Thread error: {exc}",
                        artifacts={"error": str(exc)},
                    )
                responses.append(resp)

        return responses

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        """Release cached runner modules and their GPU-resident models."""
        for runner in self._runner_cache.values():
            if runner is None:
                continue
            mod_ns = getattr(runner, "__globals__", {})
            for key in list(mod_ns):
                if "cache" in key.lower() and isinstance(mod_ns.get(key), dict):
                    mod_ns[key].clear()
        self._runner_cache.clear()
        # Explicitly free GPU memory held by now-deleted skill model tensors.
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Request normalization
    # ------------------------------------------------------------------

    def _normalize_request(self, request: SkillRequest) -> SkillRequest:
        duration = request.video_duration if request.video_duration and request.video_duration > 0 else self._probe_video_duration(request.video_path)
        normalized = SkillRequest(
            question=request.question,
            video_path=request.video_path,
            video_duration=duration,
            start_time=request.start_time,
            end_time=request.end_time,
            metadata=dict(request.metadata),
        )
        start, end = normalized.normalized_window()
        normalized.start_time = start
        normalized.end_time = end
        return normalized

    @staticmethod
    def _probe_video_duration(video_path: str) -> float:
        if not os.path.isfile(video_path):
            return 0.0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps <= 0:
            return 0.0
        return float(total_frames / fps)

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Runner loading
    # ------------------------------------------------------------------

    def _get_runner(self, skill_dir: str) -> Optional[Callable]:
        if skill_dir not in self._runner_cache:
            self._runner_cache[skill_dir] = self._load_runner(skill_dir)
        return self._runner_cache[skill_dir]

    @staticmethod
    def _load_runner(skill_dir: str) -> Optional[Callable]:
        candidate = os.path.join(skill_dir, "runner.py")
        if not os.path.isfile(candidate):
            return None
        spec = importlib.util.spec_from_file_location("skill_runner", candidate)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return getattr(module, "run", None)
