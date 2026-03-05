"""
Temporal Action Counter skill.

Counts the number of distinct times a specific action/event occurs in a video,
using OpenCLIP for per-frame binary classification followed by transition counting.

Approach:
  1. Extract the action phrase from the question via gpt-4o-mini.
  2. Sample _NUM_FRAMES frames uniformly across the video segment.
  3. For each frame, compute OpenCLIP cosine similarity to:
       positive: "a video frame where {action_phrase}"
       negative: "a video frame without {action_phrase}"
  4. Label each frame as active (positive > negative + gap threshold).
  5. Count NO→YES transitions → number of distinct event instances.

This is designed for "how many times does X happen" event-frequency questions
where ByteTrack entity counting is inappropriate (no persistent bounding boxes).
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

_NUM_FRAMES = int(os.getenv("TAC_NUM_FRAMES", "16"))
_THRESHOLD_GAP = float(os.getenv("TAC_THRESHOLD_GAP", "0.02"))  # pos must beat neg by this margin

# ── Lazy-load CLIP from the visual_option_match module (shared model cache) ──
_VISOPT_MODULE = None


def _get_visopt_module():
    global _VISOPT_MODULE
    if _VISOPT_MODULE is not None:
        return _VISOPT_MODULE
    p = Path(__file__).parent.parent / "visual_option_match" / "runner.py"
    spec = importlib.util.spec_from_file_location("_tac_visopt_impl", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _VISOPT_MODULE = mod
    return mod


def _sample_frames(
    video_path: str,
    n: int,
    start_time: Optional[float],
    end_time: Optional[float],
) -> Tuple[List[np.ndarray], float]:
    """Sample n frames uniformly from [start_time, end_time]. Returns (frames, duration)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_f = int((start_time or 0.0) * fps)
    end_f = int((end_time or duration) * fps)
    start_f = max(0, min(start_f, total_frames - 1))
    end_f = max(start_f + 1, min(end_f, total_frames))

    indices = np.linspace(start_f, end_f - 1, num=min(n, end_f - start_f), dtype=int)
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames, duration


_ACTION_EXTRACT_PROMPT = """\
Extract the specific action or event that is being counted in this question.
Output ONLY a short action description (2-8 words, starting with a noun or verb phrase).

Examples:
  Q: "How many times does the player dive into the water?"
  A: player diving into water

  Q: "How many tricks are performed in the video?"
  A: trick or stunt being performed

  Q: "How many times does the ball hit the net?"
  A: ball hitting the net

Question: {question}
Action:"""


def _regex_extract_action(question: str) -> str:
    """Regex-only fallback to extract a meaningful action phrase from counting questions."""
    q = question.lower().strip().rstrip("?")
    # Pattern 1: "how many times did/does/do X [VERB]..."
    m = re.search(r"how many times (?:did|does|do|is|are|was|were)\s+(.+?)(?:\s+in\b|\s+at\b|\s+during\b|\s+throughout\b|$)", q)
    if m:
        phrase = m.group(1).strip()
        return phrase[:60] if phrase else ""
    # Pattern 2: "how many [NOUN] are/were performed/done..."
    m = re.search(r"how many (\w+(?:\s+\w+){0,3})\s+(?:are|were|is|was)\s+(?:performed|done|executed|completed|made|taken|attempted)", q)
    if m:
        return m.group(1).strip() + " being performed"
    # Pattern 3: "how many [NOUN]" — grab the first noun after "how many"
    m = re.search(r"how many (\w+)\b", q)
    if m:
        noun = m.group(1).strip()
        if noun not in ("times", "different", "total", "distinct", "more", "less"):
            return noun
    return ""


def _extract_action(question: str, llm) -> str:
    """Use LLM to extract a concise action phrase from the question."""
    try:
        if llm is not None:
            result = llm.complete(
                _ACTION_EXTRACT_PROMPT.format(question=question[:300]),
                max_tokens=20,
            ).strip()
            result = re.sub(r"^[Aa]:\s*", "", result).strip('" \'')
            if result:
                return result
    except Exception:
        pass
    # Regex fallback (used when LLM is None OR LLM call fails)
    fallback = _regex_extract_action(question)
    return fallback or "action or event happening"


def _clip_score_frames(
    frames: List[np.ndarray],
    action_phrase: str,
) -> List[bool]:
    """Return per-frame boolean: True if action is actively happening."""
    import torch
    from PIL import Image

    visopt = _get_visopt_module()
    model, preprocess, tokenizer, device = visopt._get_model()

    pos_text = f"a video frame where {action_phrase}"
    neg_text = f"a video frame without {action_phrase}"

    texts = tokenizer([pos_text, neg_text]).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(texts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    labels: List[bool] = []
    for frame in frames:
        rgb = frame[..., ::-1].copy()  # BGR → RGB
        pil_img = Image.fromarray(rgb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_feats.T).squeeze(0)  # [pos_sim, neg_sim]
        pos_sim = sims[0].item()
        neg_sim = sims[1].item()
        labels.append(pos_sim > neg_sim + _THRESHOLD_GAP)

    return labels


def _count_events(labels: List[bool]) -> int:
    """Count NO→YES transitions = number of distinct event instances."""
    count = 0
    in_event = False
    for active in labels:
        if active and not in_event:
            count += 1
            in_event = True
        elif not active:
            in_event = False
    return count


def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    if not os.path.isfile(request.video_path):
        return SkillResponse(
            skill_name=metadata.name,
            summary="[ActionCounter] Error: video file not found.",
            artifacts={"error": "missing_file"},
        )

    llm = default_llm_client()
    q = request.question

    # ── Step 1: extract action phrase ────────────────────────────────────────
    action_phrase = _extract_action(q, llm)

    # ── Step 2: sample frames ─────────────────────────────────────────────────
    start_time, end_time = request.normalized_window()
    frames, duration = _sample_frames(
        request.video_path, _NUM_FRAMES, start_time, end_time,
    )

    if not frames:
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[ActionCounter] Could not read frames from video.",
            content=f"Unable to sample frames for action counting.",
            artifacts={"action": action_phrase, "error": "no_frames"},
        )

    # ── Step 3: CLIP per-frame classification ─────────────────────────────────
    try:
        labels = _clip_score_frames(frames, action_phrase)
    except Exception as exc:
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[ActionCounter] CLIP scoring failed: {exc!r}",
            artifacts={"action": action_phrase, "error": str(exc)},
        )

    # ── Step 4: count distinct events ────────────────────────────────────────
    event_count = _count_events(labels)
    n_active = sum(labels)
    n_total = len(labels)

    content = (
        f"[ActionCounter] The action '{action_phrase}' occurs "
        f"{event_count} distinct time(s) in the video "
        f"({n_active}/{n_total} sampled frames show the action active)."
    )

    return SkillResponse(
        skill_name=metadata.name,
        summary=(
            f"[ActionCounter] action='{action_phrase}' | "
            f"count={event_count} | active_frames={n_active}/{n_total} | "
            f"video_duration={duration:.1f}s"
        ),
        content=content,
        artifacts={
            "action": action_phrase,
            "event_count": event_count,
            "n_active_frames": n_active,
            "n_total_frames": n_total,
            "frame_labels": labels,
            "duration_sec": duration,
            "threshold_gap": _THRESHOLD_GAP,
        },
    )
