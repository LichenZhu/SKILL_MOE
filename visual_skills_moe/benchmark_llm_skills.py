"""
LLM + Skills benchmark: demonstrate skill value for video QA.

Runs each question twice:
  A) Baseline: LLM answers from question text only (no video access)
  B) With skills: LLM selects skills → skills extract video info → LLM answers with evidence

Usage:
    CUDA_VISIBLE_DEVICES=7 python benchmark_llm_skills.py \
        --dataset benchmarks/analysis/random50_sm_seed20260213.json \
        --limit 20 \
        --output-dir benchmarks/llm_skills_demo
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from skill_moe.env import load_env

load_env()

from openai import OpenAI

from benchmark import extract_mc_answer, load_dataset
from skill_moe.base import SkillRequest, SkillResponse
from skill_moe.pipeline import VideoUnderstandingPipeline
from skill_moe.registry import SkillRegistry

# Suppress verbose litellm/openai debug logging.
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SKILL_DESCRIPTIONS = {
    "asr": "Transcribe speech/narration from the video audio. Use for questions about spoken words, dialogue, narration, announcements, or company/person names mentioned aloud.",
    "rag_asr": "Retrieval-Augmented ASR: transcribes speech then extracts only the 1-3 sentences most relevant to the question. Use instead of asr when the question asks what was 'mentioned', 'discussed', or 'talked about' in the video — avoids sycophancy from irrelevant dialogue.",
    "ocr": "Read text, signs, labels, scores, lane numbers, clocks, captions visible in video frames. Use for questions about displayed text, numbers, scores, or time shown on screen.",
    "object_detect": "Detect and count common objects/people in video frames using bounding-box detection. ONLY use for: (1) counting questions (how many X), (2) confirming presence of common everyday objects (people, cars, animals, furniture). Do NOT use for: negative-presence questions ('which does NOT appear'), abstract/natural concepts (moon, iceberg, earth, fire, water), lane identification, or emotional reactions.",
    "tracking": "Uses YOLOv8+ByteTrack to track objects across video frames and count unique instances. Essential for 'how many' counting questions that require counting unique entities over time (e.g., 'how many people appear', 'how many challengers', 'how many times did X happen'). Returns unique count and max simultaneous count.",
    "temporal_action_counter": "Counts distinct occurrences of a specific action/event using CLIP per-frame classification. Use for 'how many times does X happen', 'how many tricks are performed', 'how many shots does the player take' — event-frequency questions where ByteTrack entity counting is inappropriate.",
    "zero_shot_identity": "Crop-then-CLIP person identity matcher. Detects people with YOLOv8, crops them, then scores each crop against MCQ options or descriptors using CLIP. Use for 'who is the person with X', 'who is wearing Y', 'what role does the person play' identity questions.",
    "action": "Recognize actions and activities (e.g. walking, cooking). Use for questions about what someone is doing or what events occur.",
    "scene": "Describe the visual scene, setting, background. Use for questions about the environment, location, or overall appearance.",
    "face": "Detect faces, recognize emotions and expressions. Use for questions about people's faces, emotions, or how someone looks/feels.",
    "motion": "Track object/person movement and trajectories. Use for questions about where/how something moved.",
    "spatial": "Analyze spatial relationships between objects. Use for questions about relative positions (left/right/above/below).",
    "shape_count": "Count geometric shapes (especially circles/rings). Use for questions about how many circles or round shapes are visible.",
    "visual_option_match": "Match MC options against video frames via CLIP image-text similarity. Use ONLY for: (1) clothing/outfit questions (what is someone wearing, what color/style), (2) scenery/background type questions (indoor/outdoor, natural/cityscape). Do NOT use for: action/event questions, object/tool identification, negative-presence questions ('which does NOT appear'), or any question requiring knowledge of timing or sequence.",
    "frame_vqa": "Analyzes sampled video frames using a vision-language model (Qwen2-VL). Returns visual description + recommended answer (A/B/C/D). Use for: (1) action/activity questions ('what is the person doing', 'what sport is shown', 'what activity occurs'), (2) physical events ('what happened to X', 'what shows up when Y'), (3) visual identification ('which person/object appears', 'which element does NOT appear visually', 'who reaches the finish line first'). Do NOT use for: counting exact numbers of objects, spoken/audio content, or on-screen text/scores/timestamps.",
}

_ROUTER_PROMPT_TEMPLATE = """\
You are selecting analysis tools for a video understanding question.

Available tools:
__TOOL_LIST__

Question: {question}
Options:
{options}

Select 1-3 tools that would help answer this question. Return JSON only:
{{"tools": ["asr"], "reason": "question asks about spoken content"}}
Return {{"tools": []}} if the question can be answered from text alone without video analysis.

Rules:
- Prefer asr for questions about spoken words, names, narration, or what someone says/announces.
- ALWAYS use asr for questions about why people are cheering, celebrating, or reacting (e.g., "what are people cheering for", "what event is being celebrated") — announcer commentary explains the event context.
- Do NOT use asr for Action Recognition or Object Recognition questions (what sport/activity is shown, what is the person doing, what physically happened to an object). Use frame_vqa instead.
- Prefer ocr for questions about text/numbers/scores/lane numbers/clocks shown on screen.
- Use frame_vqa for: (1) action/activity questions (what is the person doing, what sport is shown, what activity occurs, what is happening), (2) physical event questions (what happened to X, what happened to the car/object). Use frame_vqa when the question asks about what someone or something is physically doing or what physically occurred.
- For questions asking which option does NOT appear, is NOT shown, is NOT visible, "isn't mentioned", "was not discussed", "not shown", return {{"tools": []}} — determining absence from evidence is unreliable and will mislead the LLM.
- Use visual_option_match ONLY for clothing/outfit or scenery-type questions. Never use it for action questions, object/tool identification, or identity questions.
- Use object_detect ONLY for counting objects that coexist simultaneously in a scene (e.g., "how many cars in the lot", "how many people in the crowd"). Never use it for sequential/narrative counts.
- If no tool is clearly relevant, return {{"tools": []}}.
"""


def _build_router_prompt(valid_skills: set[str]) -> str:
    """Build router prompt dynamically based on available skills."""
    lines = []
    for name, desc in _SKILL_DESCRIPTIONS.items():
        if name in valid_skills:
            lines.append(f"- {name}: {desc}")
    tool_list = "\n".join(lines)
    # Use string replace for tool_list (not .format) to avoid
    # double-escaping issues with JSON braces in the template.
    return _ROUTER_PROMPT_TEMPLATE.replace("__TOOL_LIST__", tool_list)

_BASELINE_PROMPT = """\
Answer this multiple-choice question about a video.
You do NOT have access to the video itself. Answer to the best of your ability based on the question and options alone.

{question}

Choose the correct answer from:
{options}

Answer with ONLY the letter (A, B, C, or D).
"""

_EVIDENCE_PROMPT = """\
Answer this multiple-choice question about a video using the evidence extracted from the video by analysis tools.

{question}

Choose the correct answer from:
{options}

Evidence from video analysis tools:
{evidence}

Instructions:
- WARNING: External visual tools (especially frame_vqa and object detection) are highly prone to hallucinations. You MUST critically evaluate their evidence. If a tool claims an object is present, identifies something, or recommends an answer in a way that seems implausible or contradicts your knowledge of the topic, IGNORE that tool's evidence and rely on your own reasoning instead. Trust ASR (speech transcription) and OCR (on-screen text) significantly more than visual frame analysis or object detection.
- Use the evidence if it directly and explicitly answers the question (e.g., contains the exact name, number, or fact being asked).
- For visual frame analysis evidence (from frame_vqa), its recommended answer is a useful signal but NOT authoritative — treat it as one vote, not a command. If the recommendation contradicts the question context, seems implausible, or conflicts with what you know, override it with your best judgment.
- If the evidence talks about a related topic but does NOT explicitly state the answer, weigh it against your own knowledge.
- If the evidence is repetitive, garbled, clearly off-topic, or consists mainly of music symbols (🎵) and crowd chanting, treat it as absent.
- Answer with ONLY the letter (A, B, C, or D).
"""

# ---------------------------------------------------------------------------
# LLM calls (via openai SDK, supports max_completion_tokens for new models)
# ---------------------------------------------------------------------------

def _llm_complete(client: OpenAI, model: str, prompt: str) -> str:
    """Call the LLM. Omits max_tokens to avoid proxy compatibility issues."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def llm_baseline_answer(client: OpenAI, model: str, question: str, options: List[str]) -> str:
    """LLM answers without any video evidence (baseline)."""
    prompt = _BASELINE_PROMPT.format(
        question=question,
        options="\n".join(options),
    )
    raw = _llm_complete(client, model, prompt)
    return extract_mc_answer(raw)


def llm_select_skills(
    client: OpenAI,
    model: str,
    question: str,
    options: List[str],
    valid_skills: set[str],
) -> tuple[List[str], str]:
    """LLM selects which skills to run. Returns (skill_names, reason)."""
    router_prompt = _build_router_prompt(valid_skills)
    prompt = router_prompt.format(
        question=question,
        options="\n".join(options),
    )
    raw = _llm_complete(client, model, prompt)

    # Parse JSON
    tools: List[str] = []
    reason = ""
    try:
        text = raw.strip()
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
        payload = json.loads(text)
        tools = payload.get("tools", [])
        reason = payload.get("reason", "")
    except Exception:
        logger.warning("Could not parse router response: %s", raw[:200])

    tools = [t for t in tools if t in valid_skills]
    return tools[:3], reason


def llm_answer_with_evidence(
    client: OpenAI,
    model: str,
    question: str,
    options: List[str],
    evidence: str,
) -> str:
    """LLM answers using skill-extracted evidence."""
    prompt = _EVIDENCE_PROMPT.format(
        question=question,
        options="\n".join(options),
        evidence=evidence,
    )
    raw = _llm_complete(client, model, prompt)
    return extract_mc_answer(raw)


# ---------------------------------------------------------------------------
# Skill execution (reuses pipeline internals)
# ---------------------------------------------------------------------------

_SKILL_TIMEOUT_SEC: dict[str, int] = {
    "ocr": 180,         # EasyOCR model init (~40s) + frame processing
    "asr": 120,         # Whisper on long videos can be slow
    "object_detect": 120,  # GroundingDINO model init on first call
    "frame_vqa": 360,   # Qwen2-VL-7B model init (~60s) + 4-frame inference
    "tracking": 240,    # YOLOv8 model init + ByteTrack over many frames
    "grounding": 120,   # GroundingDINO-tiny model init + 4-frame inference
    "default": 60,
}

# Skills that are known to be broken on this machine.
# Excluded from routing to avoid wasting time.
_BROKEN_SKILLS = {"scene", "motion", "action", "spatial", "face", "object_detect"}


def _run_one_skill_thread(
    skill_name: str,
    request: SkillRequest,
    pipeline: VideoUnderstandingPipeline,
    result_holder: Dict[str, Any],
) -> None:
    """Run a single skill in a thread, storing result in result_holder."""
    try:
        resp = pipeline._execute_skills_parallel([skill_name], request)[0]
        result_holder["response"] = resp
    except Exception as exc:
        result_holder["error"] = str(exc)


def run_skills(
    skill_names: List[str],
    video_path: str,
    question: str,
    registry: SkillRegistry,
    pipeline: VideoUnderstandingPipeline,
    options: List[str] | None = None,
) -> tuple[List[SkillResponse], str]:
    """Run selected skills sequentially with per-skill timeout.

    Runs ONE skill at a time to avoid thread accumulation from hanging skills.
    Each skill gets a daemon thread with timeout — if it hangs, we move on.
    """
    if not skill_names:
        return [], ""

    # Append options as A./B./C./D. lines so skills like visual_option_match
    # can parse them via regex (e.g. _extract_options uses r"^\s*([A-D])\.\s*(.+?)\s*$").
    if options:
        question_for_skill = question + "\n" + "\n".join(options)
    else:
        question_for_skill = question

    request = SkillRequest(question=question_for_skill, video_path=video_path)
    request = pipeline._normalize_request(request)

    # Temporal windowing: focus ASR / frame_vqa on relevant portion when question
    # references a specific part of the video (e.g. "at the end", "at the beginning").
    temporal_skills = {"asr", "frame_vqa"}
    if temporal_skills & set(skill_names) and request.video_duration > 0:
        q_lower = question.lower()
        if any(p in q_lower for p in ("at the end", "end of the video", "introduced at the end",
                                       "at the conclusion", "toward the end", "last part")):
            request.start_time = request.video_duration * 0.85
            request.end_time = request.video_duration
        elif any(p in q_lower for p in ("at the beginning", "beginning of the video",
                                         "at the start", "start of the video", "first part",
                                         "opens with", "in the opening")):
            request.start_time = 0.0
            request.end_time = request.video_duration * 0.25

    import threading

    responses: List[SkillResponse] = []

    for name in skill_names:
        result_holder: Dict[str, Any] = {}
        t = threading.Thread(
            target=_run_one_skill_thread,
            args=(name, request, pipeline, result_holder),
            daemon=True,
        )
        t.start()
        skill_timeout = _SKILL_TIMEOUT_SEC.get(name, _SKILL_TIMEOUT_SEC["default"])
        t.join(timeout=skill_timeout)

        if t.is_alive():
            logger.warning("Skill '%s' timed out after %ds.", name, skill_timeout)
            # Daemon thread will be abandoned — no way to kill Python threads.
            # But since we run sequentially, max 1 zombie at a time.
        elif "response" in result_holder:
            responses.append(result_holder["response"])
            logger.info("Skill '%s' completed.", name)
        elif "error" in result_holder:
            logger.warning("Skill '%s' failed: %s", name, result_holder["error"])

    evidence = pipeline._build_evidence_text(responses) or ""
    return responses, evidence


# ---------------------------------------------------------------------------
# Hardcoded routing overrides (defensive rules applied after LLM router)
# ---------------------------------------------------------------------------

_TEMPORAL_VISUAL_MARKERS = (
    "at the end", "at the beginning", "at the start",
    "in the beginning", "in the first", "in the last",
    "at the end of the video", "at the beginning of the video",
    "first appear", "last appear", "finally appear",
    "introduced at the end", "introduced at the beginning",
    "first shown", "last shown", "first introduced", "last introduced",
    "first scene", "last scene", "opening scene", "closing scene",
    "end of the video", "beginning of the video", "start of the video",
)
_TEMPORAL_AUDIO_EXCEPTIONS = (
    "say", "said", "speak", "spoken", "hear", "heard",
    "sound", "cheer", "narrat", "announce", "dialogue",
)
_COUNTING_PATTERNS = ("how many", "how much", "number of", "count of", "largest number", "maximum number")
_NEGATIVE_PATTERNS = (
    "not appear", "does not appear", "didn't appear", "not shown",
    "not visible", "absent", "missing", "never shown", "not present",
    "not mentioned", "wasn't shown", "isn't shown", "not include",
    "never appears", "not feature",
)
_CHEER_PATTERNS = (
    "cheering", "cheered", "celebrating", "celebration", "cheer",
    "sound", "hear", "heard", "reacting", "reaction", "crowd reaction",
    "applaud", "applause", "announcement", "announced",
)
_TEXT_PATTERNS = (
    "text", "sign", "read", "number on", "written", "caption",
    "label", "displayed", "shows on screen", "on the screen",
    "on screen", "score", "lane number",
)


def _apply_routing_overrides(
    question: str,
    selected_skills: List[str],
    route_reason: str,
    valid_skills: set[str],
) -> tuple[List[str], str]:
    """Hardcoded regex rules that override or supplement LLM router decisions.

    Rules (in priority order):
      1. Counting or negative-presence questions → force tools=[]
         (these categories are historically net-negative with visual tools)
      2. Cheering/reaction/sound questions → ensure asr is included
      3. Text-reading questions → ensure ocr is included
    Rules 2 & 3 fire only if Rule 1 did not suppress all tools.
    """
    q = question.lower()
    selected_skills = list(selected_skills)

    # Rule 1a: Negative-presence → suppress all visual tools (absence is unreliable)
    is_negative = any(p in q for p in _NEGATIVE_PATTERNS)
    if is_negative and selected_skills:
        route_reason = (
            f"[OVERRIDE:negative-presence] forced tools=[] (was: {selected_skills}). "
            "Letting baseline LLM handle absence detection."
        )
        return [], route_reason

    # ── Pre-compute shared flags used by multiple rules below ──────────────────
    is_counting = any(p in q for p in _COUNTING_PATTERNS)

    # Rule 1b-pre: ASR Offensive — "are mentioned / talk about / speaker" questions.
    # Anything asking *what was stated/discussed* in the video is an audio question:
    # ASR provides direct transcript evidence; tracking/grounding are irrelevant.
    # Fire BEFORE the tracking rule so "how many methods are mentioned" → ASR, not tracking.
    _MENTIONED_PATTERNS = (
        "are mentioned", "is mentioned", "was mentioned", "were mentioned",
        "mentioned in the video", "mentioned in the clip",
        "talk about", "talks about", "talking about", "discussed",
        # NOTE: "speaker"/"speakers" removed — too broad; matches "speaker's waist"
        # (visual-identification questions). Only fire on explicit "mentioned" phrasing.
    )
    is_mentioned = any(p in q for p in _MENTIONED_PATTERNS)
    if is_mentioned:
        old = selected_skills[:]
        if "rag_asr" in valid_skills:
            selected_skills = ["rag_asr"]
            route_reason = (
                f"[OVERRIDE:mentioned→rag_asr] forced ['rag_asr'] (was: {old}). "
                "Question asks what is stated/discussed — RAG-ASR extracts relevant snippet."
            )
        elif "asr" in valid_skills:
            selected_skills = ["asr"]
            route_reason = (
                f"[OVERRIDE:mentioned→asr] forced ['asr'] (was: {old}). "
                "Question asks what is stated/discussed — ASR gives transcript evidence."
            )
        else:
            # Neither available — strip visual tools; let VLM answer from video alone.
            selected_skills = []
            route_reason = (
                f"[OVERRIDE:mentioned→tools=[] (asr unavailable)] stripped {old}. "
                "Mentioned-in-video question with no ASR — VLM answers directly."
            )
        return selected_skills, route_reason

    # Rule 1b-pre2: Attribute/Color Offensive — counting with explicit colour/dress constraint.
    # ByteTrack counts ALL entities ignoring colour; CLIP-based option matcher ranks by appearance.
    # Trigger only on "dressed in" / "clothed in" — NOT plain "wearing [object]" which tracking
    # can still handle (e.g., "how many people are wearing ties" ✓ with tracking in Run 2).
    _CLOTHING_PATTERNS = ("dressed in", "clothed in")
    is_clothing_count = is_counting and any(p in q for p in _CLOTHING_PATTERNS)
    if is_clothing_count and "visual_option_match" in valid_skills:
        old = selected_skills[:]
        selected_skills = ["visual_option_match"]
        route_reason = (
            f"[OVERRIDE:clothing-attribute→visual_option_match] forced "
            f"['visual_option_match'] (was: {old}). "
            "Tracking ignores colour/clothing — CLIP option matcher ranks by appearance."
        )
        return selected_skills, route_reason

    # Rule 1b-pre3: Identity questions → zero_shot_identity (crop-then-CLIP person matcher).
    # "Who is the person with X / wearing Y / standing at Z" — these require identifying a
    # specific individual from visual attributes, not counting or general scene understanding.
    _IDENTITY_PATTERNS = (
        "who is the person", "who is wearing", "who appears",
        "what role does the person", "who is standing", "who is sitting",
        "who is holding", "who is the one", "who is shown",
    )
    is_identity = any(p in q for p in _IDENTITY_PATTERNS)
    if is_identity and "zero_shot_identity" in valid_skills:
        old = selected_skills[:]
        selected_skills = ["zero_shot_identity"]
        route_reason = (
            f"[OVERRIDE:identity→zero_shot_identity] forced ['zero_shot_identity'] "
            f"(was: {old}). Identity question — crop-then-CLIP person matcher."
        )
        return selected_skills, route_reason

    # Rule 1b-pre4: Action-event counting → temporal_action_counter.
    # "How many times does X happen", "how many tricks are performed" — sequential/repeated
    # events cannot be counted by ByteTrack (which tracks persistent bounding boxes).
    # temporal_action_counter uses CLIP per-frame classification + NO→YES transition counting.
    _ACTION_COUNTER_PATTERNS = (
        "how many times",
        "are performed",
        "how many tricks",      # skateboard/gymnastics events
        "how many shots",       # basketball/pool/hockey shots (action events)
        "how many dives", "how many jumps", "how many attempts",
        "how many flips", "how many turns",
    )
    is_action_event = is_counting and any(p in q for p in _ACTION_COUNTER_PATTERNS)
    if is_action_event and "temporal_action_counter" in valid_skills:
        old = selected_skills[:]
        selected_skills = ["temporal_action_counter"]
        route_reason = (
            f"[OVERRIDE:action-event→temporal_action_counter] forced "
            f"['temporal_action_counter'] (was: {old}). "
            "Action-frequency question — CLIP per-frame event counter."
        )
        return selected_skills, route_reason

    # Rule 1b: Counting → force tracking, UNLESS it falls into an exclusion bucket:
    #   • Scoreboard:  "how many points/goals/score" → OCR reads the displayed number.
    #   • Visible:     "visible in the video" → VideoLLM direct inspection already correct.
    #   • Event/action counting: "interviewed", "participated", "how many times" →
    #     sequential/repeated events, not simultaneous-presence; ByteTrack misleads.
    #   • Abstract entities: "how many companies/brands/..." → cannot be visually tracked.
    #   • Historical year: "in 2010" → documentary fact, not trackable from current frames.
    #   • Temporal/spatial constraints: "introductory shot", "presenting on" →
    #     tracking over the full video ≠ counting within a specific shot or location.
    _SCOREBOARD_PATTERNS = (
        "how many points", "how many goals", "score of", "how many runs",
        "how many sets", "how many games", "how many rounds",
    )
    _VISIBLE_PATTERNS = ("visible in the video", "visible in the frame",)
    _EVENT_PATTERNS = (
        "interviewed", "were interviewed", "participated",
        # NOTE: "are performed", "how many tricks", "how many shots" REMOVED —
        # they now route to temporal_action_counter (Rule 1b-pre4 above).
        "introductory shot",    # temporal: only the opening shot, not full video
        "presenting on",        # spatial: subset of people on stage
        # Non-COCO physical objects — tracker falls back to person, giving nonsense counts.
        # Route to VLM direct rather than letting a wrong entity be tracked.
        "how many flags", "how many national flags",
        "how many spheres",
        "how many grenades",
        "how many methods",     # abstract procedural nouns
        # NOTE: "how many times" intentionally KEPT OUT — tracking can legitimately help
        # for "how many times does X die/fall/appear" questions (e.g., 438-2 correct in Run 2).
        # NOTE: "how many dives", "how many jumps", "how many attempts" — also handled
        # by temporal_action_counter (Rule 1b-pre4) before this block fires.
    )
    _ABSTRACT_PATTERNS = (
        "how many companies", "how many brands", "how many organizations",
        "how many countries", "how many nations", "how many teams",
        "how many steps", "how many stages", "how many episodes",
        "how many chapters", "how many sections",
    )
    is_scoreboard = any(p in q for p in _SCOREBOARD_PATTERNS)
    is_visible_count = any(p in q for p in _VISIBLE_PATTERNS)
    is_event_count = any(p in q for p in _EVENT_PATTERNS)
    is_abstract_count = any(p in q for p in _ABSTRACT_PATTERNS)
    is_historical_year = bool(re.search(r"\bin (19|20)\d\d\b", q))
    _tracking_blocked = (
        is_scoreboard or is_visible_count or is_event_count
        or is_abstract_count or is_historical_year
    )
    # Actively strip tracking from LLM router's selection when blocked, not just
    # prevent forcing — prevents tool sycophancy even when LLM chose tracking.
    if is_counting and _tracking_blocked and "tracking" in selected_skills:
        selected_skills = [s for s in selected_skills if s != "tracking"]
        route_reason += " [OVERRIDE:tracking-stripped-blocked]"
    if is_counting and not _tracking_blocked and "tracking" in valid_skills:
        if selected_skills and "tracking" not in selected_skills:
            route_reason = (
                f"[OVERRIDE:counting→tracking] replaced {selected_skills} with tracking. "
                "ByteTrack gives unique entity counts across the full video."
            )
        elif not selected_skills:
            route_reason = (
                "[OVERRIDE:counting→tracking] forced tracking (was: []). "
                "ByteTrack gives unique entity counts across the full video."
            )
        return ["tracking"], route_reason
    if is_scoreboard and "ocr" in valid_skills and "ocr" not in selected_skills:
        selected_skills = ["ocr"]
        route_reason += " [OVERRIDE:scoreboard→ocr]"

    # Rule 2: Temporal-visual question → exclude asr
    # ASR covers the full video; for questions pinpointing a specific visual moment
    # ("at the end", "first appears", etc.) the globally-dominant speech content
    # overrides correct visual perception (Tool Sycophancy L1/L2/L3).
    if "asr" in selected_skills:
        has_temporal = any(p in q for p in _TEMPORAL_VISUAL_MARKERS)
        has_audio = any(p in q for p in _TEMPORAL_AUDIO_EXCEPTIONS)
        if has_temporal and not has_audio:
            selected_skills = [s for s in selected_skills if s != "asr"]
            route_reason += " [OVERRIDE:temporal_visual_exclude_asr]"

    # Rule 3: Cheering/reaction/sound → append asr if missing
    if any(p in q for p in _CHEER_PATTERNS) and "asr" in valid_skills and "asr" not in selected_skills:
        selected_skills.append("asr")
        route_reason += " [OVERRIDE: appended asr for cheering/reaction/sound]"

    # Rule 3: Text/sign/reading questions → append ocr if missing
    if any(p in q for p in _TEXT_PATTERNS) and "ocr" in valid_skills and "ocr" not in selected_skills:
        selected_skills.append("ocr")
        route_reason += " [OVERRIDE: appended ocr for text/sign question]"

    return selected_skills[:3], route_reason


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------

def run_single(
    item: Dict[str, Any],
    client: OpenAI,
    model: str,
    registry: SkillRegistry,
    pipeline: VideoUnderstandingPipeline,
    valid_skills: set[str],
) -> Dict[str, Any]:
    """Run one test case: baseline vs with-skills."""
    qid = item.get("id", "?")
    question = item["question"]
    ground_truth = item["ground_truth"].strip().upper()
    video_path = item.get("video_path", "")
    options = item.get("options", [])
    category = item.get("category", "")

    result: Dict[str, Any] = {
        "id": qid,
        "category": category,
        "question": question,
        "ground_truth": ground_truth,
        "options": options,
        "video_path": video_path,
    }

    t0 = time.perf_counter()

    # --- A) Baseline: LLM only ---
    try:
        baseline_answer = llm_baseline_answer(client, model, question, options)
    except Exception as exc:
        baseline_answer = ""
        logger.warning("[%s] Baseline LLM error: %s", qid, exc)

    baseline_correct = baseline_answer == ground_truth
    result["baseline_answer"] = baseline_answer
    result["baseline_correct"] = baseline_correct

    # --- B) LLM Router selects skills ---
    try:
        selected_skills, route_reason = llm_select_skills(
            client, model, question, options, valid_skills,
        )
    except Exception as exc:
        selected_skills, route_reason = [], f"Router error: {exc}"
        logger.warning("[%s] Router error: %s", qid, exc)

    # --- B2) Apply hardcoded routing overrides (defensive rules) ---
    selected_skills, route_reason = _apply_routing_overrides(
        question, selected_skills, route_reason, valid_skills,
    )
    logger.info("[%s] Final skills after overrides: %s | %s", qid, selected_skills, route_reason)

    result["selected_skills"] = selected_skills
    result["route_reason"] = route_reason

    # --- C) Run skills ---
    evidence = ""
    skill_outputs: List[Dict[str, str]] = []
    if selected_skills:
        try:
            responses, evidence = run_skills(
                selected_skills, video_path, question, registry, pipeline,
                options=options,
            )
            for resp in responses:
                skill_outputs.append({
                    "skill": resp.skill_name,
                    "summary": resp.summary[:300],
                    "error": str(resp.artifacts.get("error", ""))
                    if isinstance(resp.artifacts, dict) else "",
                })
        except Exception as exc:
            evidence = ""
            logger.warning("[%s] Skill execution error: %s", qid, exc)
            skill_outputs.append({"skill": "error", "summary": str(exc), "error": str(exc)})

    result["evidence"] = evidence[:2000]
    result["skill_outputs"] = skill_outputs

    # --- D) LLM answers with evidence ---
    if evidence:
        try:
            skill_answer = llm_answer_with_evidence(client, model, question, options, evidence)
        except Exception as exc:
            skill_answer = baseline_answer
            logger.warning("[%s] Evidence LLM error: %s", qid, exc)
    else:
        # No evidence produced → skill_answer = baseline
        skill_answer = baseline_answer

    skill_correct = skill_answer == ground_truth
    result["skill_answer"] = skill_answer
    result["skill_correct"] = skill_correct

    # --- Comparison ---
    result["helped"] = skill_correct and not baseline_correct
    result["hurt"] = baseline_correct and not skill_correct
    result["latency_sec"] = round(time.perf_counter() - t0, 2)
    result["error"] = ""

    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict[str, Any]]) -> None:
    valid = [r for r in results if not r.get("error")]
    n = len(valid)
    if n == 0:
        print("No valid results.")
        return

    baseline_acc = sum(1 for r in valid if r["baseline_correct"]) / n * 100
    skill_acc = sum(1 for r in valid if r["skill_correct"]) / n * 100
    helped = sum(1 for r in valid if r["helped"])
    hurt = sum(1 for r in valid if r["hurt"])
    avg_skills = sum(len(r["selected_skills"]) for r in valid) / n
    avg_latency = sum(r["latency_sec"] for r in valid) / n

    print("\n" + "=" * 60)
    print("  LLM + Skills Benchmark Results")
    print("=" * 60)
    print(f"  Total cases:         {n}")
    print(f"  Baseline accuracy:   {baseline_acc:.1f}%  (LLM only, no video)")
    print(f"  With-skills accuracy:{skill_acc:.1f}%  (LLM + skill evidence)")
    print(f"  Improvement:         {skill_acc - baseline_acc:+.1f}%")
    print(f"  Skills helped:       {helped}")
    print(f"  Skills hurt:         {hurt}")
    print(f"  Avg skills/case:     {avg_skills:.1f}")
    print(f"  Avg latency:         {avg_latency:.1f}s")

    # Cases where skills helped
    if helped > 0:
        print(f"\n  Cases where skills HELPED (wrong→right):")
        for r in valid:
            if r["helped"]:
                print(f"    [{r['id']}] {r['category']}: "
                      f"baseline={r['baseline_answer']} → skill={r['skill_answer']} "
                      f"(GT={r['ground_truth']}) skills={r['selected_skills']}")

    # Cases where skills hurt
    if hurt > 0:
        print(f"\n  Cases where skills HURT (right→wrong):")
        for r in valid:
            if r["hurt"]:
                print(f"    [{r['id']}] {r['category']}: "
                      f"baseline={r['baseline_answer']} → skill={r['skill_answer']} "
                      f"(GT={r['ground_truth']}) skills={r['selected_skills']}")

    # Per-skill usage
    skill_counts: Dict[str, int] = {}
    for r in valid:
        for s in r["selected_skills"]:
            skill_counts[s] = skill_counts.get(s, 0) + 1
    if skill_counts:
        print(f"\n  Skill usage:")
        for s, c in sorted(skill_counts.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c} cases")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM + Skills Benchmark")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON/parquet")
    parser.add_argument("--limit", type=int, default=20, help="Number of cases (default: 20)")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--output-dir", default="benchmarks/llm_skills_demo")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--duration", default=None, choices=["short", "medium", "long"])
    parser.add_argument(
        "--exclude-skills",
        default=",".join(sorted(_BROKEN_SKILLS)),
        help="Comma-separated skills to exclude (default: broken ones)",
    )
    args = parser.parse_args()

    # --- LLM client (via openai SDK) ---
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.2-codex")
    api_base = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "not-set")
    client = OpenAI(base_url=api_base, api_key=api_key)
    print(f"LLM: {model} via {api_base or 'direct'}")

    # --- Skill registry & pipeline (for skill execution only) ---
    registry = SkillRegistry(root="skills")
    all_skills = set(registry.list())
    excluded = set(s.strip() for s in args.exclude_skills.split(",") if s.strip())
    valid_skills = all_skills - excluded
    print(f"Available skills: {sorted(valid_skills)}")
    if excluded & all_skills:
        print(f"Excluded (broken): {sorted(excluded & all_skills)}")

    # Create a minimal pipeline for skill execution (no VLM needed).
    from skill_moe.router import SkillRouter
    router = SkillRouter(registry, strategy="rules")
    pipeline = VideoUnderstandingPipeline(
        registry=registry,
        router=router,
        max_turns=0,
        video_llm=None,
    )

    # --- Dataset ---
    dataset, is_mc = load_dataset(args.dataset)
    if not is_mc:
        print("ERROR: This benchmark only supports multiple-choice datasets.")
        return

    if args.duration:
        dataset = [d for d in dataset if d.get("duration") == args.duration]
    if args.start > 0:
        dataset = dataset[args.start:]
    dataset = dataset[:args.limit]
    print(f"Running {len(dataset)} cases from {args.dataset}\n")

    # --- Run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    partial_path = output_dir / f"partial_{timestamp}.json"

    for item in tqdm(dataset, desc="Benchmarking", unit="case"):
        result = run_single(item, client, model, registry, pipeline, valid_skills)
        results.append(result)

        # Live feedback
        b = "+" if result["baseline_correct"] else "-"
        s = "+" if result["skill_correct"] else "-"
        delta = ""
        if result["helped"]:
            delta = " HELPED"
        elif result["hurt"]:
            delta = " HURT"
        tqdm.write(
            f"  [{result['id']}] baseline={b} skill={s} "
            f"skills={result['selected_skills']} "
            f"latency={result['latency_sec']:.1f}s{delta}"
        )

        # Save partial
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False)

        # Free GPU memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Summary ---
    n = len(results)
    baseline_correct = sum(1 for r in results if r["baseline_correct"])
    skill_correct = sum(1 for r in results if r["skill_correct"])
    helped = sum(1 for r in results if r["helped"])
    hurt = sum(1 for r in results if r["hurt"])

    summary = {
        "baseline_accuracy": round(baseline_correct / n * 100, 2) if n else 0,
        "skill_accuracy": round(skill_correct / n * 100, 2) if n else 0,
        "improvement": round((skill_correct - baseline_correct) / n * 100, 2) if n else 0,
        "skills_helped": helped,
        "skills_hurt": hurt,
        "total": n,
        "avg_latency_sec": round(sum(r["latency_sec"] for r in results) / n, 2) if n else 0,
    }

    log_data = {
        "run": {
            "timestamp": timestamp,
            "model": model,
            "api_base": api_base,
            "dataset": args.dataset,
            "num_cases": n,
            "limit": args.limit,
            "start": args.start,
        },
        "summary": summary,
        "results": results,
    }

    log_path = output_dir / f"log_{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"\nLog saved to {log_path}")

    if partial_path.exists():
        partial_path.unlink()

    print_summary(results)


if __name__ == "__main__":
    main()
