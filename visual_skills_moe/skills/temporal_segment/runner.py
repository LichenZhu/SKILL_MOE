"""
Temporal Chronicle Expert — dense frame sampling + narrative-arc description.

Problem: VLM gets 64 frames over the full video. For a 5-minute video, the
"last scene" might have only 1-2 frames, causing wrong answers on temporal
position questions ("second half", "latter part", "last scene").

Additionally, causal/reasoning questions ("why does X happen") need a narrative
arc rather than a list of per-frame observations — per-frame detail misleads the
VLM into ignoring what it directly observes in the video.

Solution:
- VISUAL mode (what/who/which): dense 24-frame grid → concise description of
  what is seen.
- CHRONICLE mode (why/reason/purpose/cause): same frames → structured Video
  Chronicle with semantic narrative beats (cause-and-effect arcs), not frame
  details.
"""
from __future__ import annotations

import base64
import math
import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

_SAMPLE_FRAMES = 24
_THUMB_W = 480
_THUMB_H = 360
_MAX_GRID_SIDE = 2048

# Triggers that indicate causal/narrative understanding is needed → Chronicle mode
_CHRONICLE_TRIGGERS = (
    "why", "reason", "purpose", "because", "cause",
    "what led to", "how did", "what happened to",
)


# ---------------------------------------------------------------------------
# Temporal window parsing
# ---------------------------------------------------------------------------

def _parse_window_fractions(question: str) -> Tuple[float, float]:
    """Return (frac_start, frac_end) in [0, 1] for the target temporal window."""
    q = question.lower()

    if any(k in q for k in ("second half", "latter half", "second part of the video")):
        return 0.5, 1.0
    if any(k in q for k in ("first half", "first part of the video", "opening half")):
        return 0.0, 0.5
    if any(k in q for k in ("last quarter", "final quarter", "last 25%")):
        return 0.75, 1.0
    if any(k in q for k in ("last third", "final third", "last 33%")):
        return 0.67, 1.0

    _end_markers = (
        "last scene", "final scene", "closing scene",
        "latter part", "final part", "last part", "last section",
        "end of the video", "end of this video", "at the end",
        "towards the end", "near the end",
    )
    if any(k in q for k in _end_markers):
        return 0.75, 1.0

    _start_markers = (
        "opening scene", "first scene", "opening part",
        "at the beginning", "beginning of the video",
        "start of the video", "at the start", "initial scene",
        "early in the video",
    )
    if any(k in q for k in _start_markers):
        return 0.0, 0.25

    return 0.0, 1.0


def _window_label(frac_start: float, frac_end: float, duration: float) -> str:
    t_start = frac_start * duration
    t_end = frac_end * duration
    pct_s, pct_e = int(frac_start * 100), int(frac_end * 100)
    return f"t={t_start:.0f}s–{t_end:.0f}s ({pct_s}%–{pct_e}% of {duration:.0f}s video)"


def _is_chronicle_mode(question: str) -> bool:
    """Return True when question asks for causal/narrative understanding."""
    q = question.lower()
    return any(k in q for k in _CHRONICLE_TRIGGERS)


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(
    video_path: str,
    start_t: float,
    end_t: float,
    n: int,
) -> List[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    n = max(1, n)
    timestamps = [
        start_t + (end_t - start_t) * i / max(n - 1, 1) for i in range(n)
    ]

    result: List[Tuple[float, np.ndarray]] = []
    for ts in timestamps:
        idx = max(0, min(int(ts * fps), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            result.append((ts, frame))
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Grid image
# ---------------------------------------------------------------------------

def _build_grid(
    frames: List[Tuple[float, np.ndarray]],
    thumb_w: int,
    thumb_h: int,
) -> str:
    """Arrange frames in a grid, timestamp-labelled. Returns base64 JPEG."""
    n = len(frames)
    if n == 0:
        return ""

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, (ts, frame) in enumerate(frames):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        label = f"{ts:.1f}s"
        cv2.rectangle(thumb, (0, thumb_h - 20), (70, thumb_h), (0, 0, 0), -1)
        cv2.putText(thumb, label, (3, thumb_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y0, x0 = r * thumb_h, c * thumb_w
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

    h, w = canvas.shape[:2]
    if max(h, w) > _MAX_GRID_SIDE:
        scale = _MAX_GRID_SIDE / max(h, w)
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Vision API — two modes
# ---------------------------------------------------------------------------

def _vision_describe(
    grid_b64: str,
    question: str,
    window_label: str,
    chronicle_mode: bool,
    duration: float,
) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        if chronicle_mode:
            # Narrative-arc / causal mode: Video Chronicle
            sys_msg = (
                "You are a video narrative analyst. You receive chronologically-ordered "
                "frames from a video segment. Produce a VIDEO CHRONICLE — a structured "
                "timeline of semantic narrative beats, NOT a frame-by-frame list.\n\n"
                "Rules:\n"
                "- Group frames into 3-5 meaningful narrative segments.\n"
                "- Format each as: [MM:SS -- MM:SS] Description of what happens and why it matters.\n"
                "- Focus on CAUSES, CONSEQUENCES, and emotional/narrative arcs.\n"
                "- NEVER use angle brackets < or > in your output.\n"
                "- NEVER list individual frame observations — synthesize into story beats.\n"
                "- Each segment description must be 10-25 words."
            )
            user_msg = (
                f"Video segment: {window_label} (total video: {duration:.0f}s)\n"
                f"Question: {question[:150]}\n\n"
                "Produce the Video Chronicle. Start directly with [MM:SS -- MM:SS] ..."
            )
            max_tokens = 512
        else:
            # Visual description mode
            sys_msg = (
                "You are analyzing chronologically-ordered frames from a specific temporal "
                "segment of a video (labeled with timestamps). "
                "Describe CONCISELY (2-4 sentences) what is happening: actions, events, "
                "visible objects, people, and any on-screen text. Be factual and specific. "
                "Do NOT use angle brackets < or > in your response."
            )
            user_msg = (
                f"Video segment: {window_label}\n"
                f"Question context: {question[:150]}\n\n"
                "What is shown in this segment? Focus on the main events and visual content."
            )
            max_tokens = 300

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{grid_b64}",
                                   "detail": "low"}},
                    {"type": "text", "text": user_msg},
                ]},
            ],
            max_tokens=max_tokens,
            timeout=30,
        )
        text = resp.choices[0].message.content.strip()
        return text if len(text) >= 10 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    video_path = request.video_path
    question = request.question
    duration = request.video_duration or 0.0

    if not os.path.isfile(video_path) or duration <= 0:
        return SkillResponse(
            skill_name="temporal_segment",
            summary="Video unavailable or duration unknown.",
            artifacts={"error": "no_video"},
        )

    frac_start, frac_end = _parse_window_fractions(question)
    start_t = frac_start * duration
    end_t = frac_end * duration
    label = _window_label(frac_start, frac_end, duration)
    chronicle_mode = _is_chronicle_mode(question)

    frames = _sample_frames(video_path, start_t, end_t, _SAMPLE_FRAMES)
    if not frames:
        return SkillResponse(
            skill_name="temporal_segment",
            summary="Could not sample frames from target window.",
            artifacts={"error": "frame_sample_failed"},
        )

    grid_b64 = _build_grid(frames, _THUMB_W, _THUMB_H)
    if not grid_b64:
        return SkillResponse(
            skill_name="temporal_segment",
            summary="Grid construction failed.",
            artifacts={"error": "grid_failed"},
        )

    desc = _vision_describe(grid_b64, question, label, chronicle_mode, duration)
    if not desc:
        return SkillResponse(
            skill_name="temporal_segment",
            summary="Vision API unavailable.",
            artifacts={"no_match": True},
        )

    mode_label = "Video Chronicle" if chronicle_mode else "Visual Description"
    summary = f"[{label}] {mode_label}: {desc}"
    return SkillResponse(
        skill_name="temporal_segment",
        summary=summary,
        artifacts={
            "window": label,
            "description": desc,
            "mode": "chronicle" if chronicle_mode else "visual",
        },
    )
