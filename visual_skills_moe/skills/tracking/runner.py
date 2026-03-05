"""Tracking skill — multi-object tracking for counting and temporal continuity.

Uses YOLOv8 + ByteTrack (via ultralytics) to assign persistent IDs across frames
and report unique entity counts and max simultaneous count.
"""
from __future__ import annotations

import gc
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

logger = logging.getLogger(__name__)

_MODEL_ID = os.getenv("TRACKING_MODEL", "yolov8n.pt")
_SAMPLE_FPS = float(os.getenv("TRACKING_SAMPLE_FPS", "2.0"))   # frames/sec to process
_MAX_FRAMES = int(os.getenv("TRACKING_MAX_FRAMES", "600"))      # hard cap for long videos

# Full COCO-80 name→class-id mapping for YOLOv8.
_COCO_NAME_TO_ID: Dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33,
    "baseball bat": 34, "baseball glove": 35, "skateboard": 36,
    "surfboard": 37, "tennis racket": 38, "bottle": 39, "wine glass": 40,
    "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59,
    "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64,
    "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77,
    "hair drier": 78, "toothbrush": 79,
}

# Common question-entity → COCO-class synonyms (supplements LLM mapping).
_SYNONYM_MAP: Dict[str, Optional[str]] = {
    "people": "person", "human": "person", "humans": "person",
    "man": "person", "woman": "person", "men": "person", "women": "person",
    "athlete": "person", "athletes": "person", "player": "person",
    "players": "person", "diver": "person", "divers": "person",
    "challenger": "person", "challengers": "person", "swimmer": "person",
    "runner": "person", "runners": "person", "competitor": "person",
    "competitors": "person", "participant": "person", "participants": "person",
    "child": "person", "children": "person", "kid": "person", "kids": "person",
    "individual": "person", "individuals": "person", "figure": "person",
    "spectator": "person", "spectators": "person", "audience": "person",
    "reporter": "person", "reporters": "person", "presenter": "person",
    "vehicles": "car", "automobile": "car", "automobiles": "car",
    # ── Explicitly non-trackable: abstract or inanimate entities ─────────────
    "earring": None, "earrings": None,  # Not in COCO
    # Inanimate objects commonly asked about in counting questions
    "disc": None, "discs": None, "disk": None, "disks": None,
    "lens": None, "lenses": None, "glass": None, "glasses": None,
    "layer": None, "layers": None, "component": None, "components": None,
    "piece": None, "pieces": None, "part": None, "parts": None,
    "item": None, "items": None, "object": None, "objects": None,
    # Abstract entities — cannot be physically tracked in video frames
    "company": None, "companies": None, "brand": None, "brands": None,
    "organization": None, "organizations": None, "firm": None, "firms": None,
    "country": None, "countries": None, "nation": None, "nations": None,
    "team": None, "teams": None, "group": None, "groups": None,
    "symbol": None, "symbols": None, "icon": None, "icons": None,
    "topic": None, "topics": None, "subject": None, "subjects": None,
    "step": None, "steps": None, "stage": None, "stages": None,
    "episode": None, "episodes": None, "scene": None, "scenes": None,
    "chapter": None, "chapters": None, "section": None, "sections": None,
    # Physical objects not in COCO-80 — fall back to person when unset, giving wrong counts
    "sphere": None, "spheres": None,        # 113-2: man holds spheres
    "flag": None, "flags": None,            # 003-1: national flags
    "grenade": None, "grenades": None,      # 438-1: in-game throws
    # Action/event nouns — ByteTrack counts entities in space, not repeated temporal events
    "trick": None, "tricks": None,          # 191-1: tricks performed
    "shot": None, "shots": None,            # 137-3: shots taken by player
    "method": None, "methods": None,        # 085-2: methods mentioned
    # Colors / visual properties — not physical trackable entities
    "color": None, "colors": None, "colour": None, "colours": None,
    "glaze": None, "glazes": None,
    # Distances / measurements — numeric values, not entities
    "meter": None, "meters": None, "metre": None, "metres": None,
    "kilometer": None, "kilometers": None, "mile": None, "miles": None,
    # Match score / game state — read from scoreboard, not tracked in video
    "point": None, "points": None,
    "substitution": None, "substitutions": None,
    # Activity/event occurrences — temporal events, not spatial entities
    "session": None, "sessions": None,
    "sport": None, "sports": None,
    # Food items commonly asked about that are NOT in COCO-80
    "pepper": None, "peppers": None,
    "vegetable": None, "vegetables": None,
    "fruit": None, "fruits": None,
    "ingredient": None, "ingredients": None,
}

# Module-level model cache — loads once per process.
_model_cache: Dict[str, Any] = {}


def _load_model() -> Any:
    if "model" not in _model_cache:
        from ultralytics import YOLO
        _model_cache["model"] = YOLO(_MODEL_ID)
    return _model_cache["model"]


def _get_target_class(question: str) -> Tuple[str, Optional[int]]:
    """Determine the COCO class to track from the question.

    Priority:
    1. Direct synonym table lookup on question words.
    2. LLM maps question → COCO class name.
    3. Default fallback to "person" (class 0).

    Returns (class_name, class_id) where class_id is None if the entity
    is not trackable (e.g. earrings).
    """
    q_lower = question.lower()

    # ── Strategy 1: synonym table lookup, two-pass ─────────────────────────
    # Pass 1a: check non-trackable (None) entries first, longest alias first.
    # This ensures action/event nouns (tricks, shots) take priority over
    # co-occurring person-synonyms (player, athlete) in the same question.
    _sorted_map = sorted(_SYNONYM_MAP.items(), key=lambda x: -len(x[0]))
    for alias, coco_name in _sorted_map:
        if coco_name is None and re.search(rf"\b{re.escape(alias)}\b", q_lower):
            return alias, None  # explicitly non-trackable; skip trackable synonyms
    # Pass 1b: now check trackable synonyms (nothing non-trackable fired above).
    for alias, coco_name in _sorted_map:
        if coco_name is not None and re.search(rf"\b{re.escape(alias)}\b", q_lower):
            class_id = _COCO_NAME_TO_ID.get(coco_name)
            if class_id is not None:
                return coco_name, class_id

    # ── Strategy 2: LLM mapping ────────────────────────────────────────────
    client = default_llm_client()
    if client:
        stem = re.split(r"\n\s*[A-D]\.", question)[0].strip()
        valid_names = ", ".join(sorted(_COCO_NAME_TO_ID.keys()))
        prompt = (
            "What is the primary object that needs to be counted in this question? "
            f"Map it to the closest COCO dataset class name from this list:\n{valid_names}\n"
            "Return ONLY the exact COCO class name (lowercase).\n"
            f"Question: {stem}"
        )
        try:
            raw = client.complete(prompt, max_tokens=16).strip().lower()
            raw_clean = re.sub(r"[^a-z0-9 ]", "", raw)
            for name in sorted(_COCO_NAME_TO_ID.keys(), key=len, reverse=True):
                if name in raw_clean:
                    return name, _COCO_NAME_TO_ID[name]
        except Exception:
            pass

    # ── Strategy 3: default to "person" ───────────────────────────────────
    return "person", 0


_SCENE_CUT_THRESHOLD = float(os.getenv("TRACKING_SCENE_CUT_THRESHOLD", "0.35"))
# Audience filter: ignore tracked boxes whose area < this fraction of the frame.
# Challengers/fighters occupy a significant portion of the frame; audience members
# and seated judges are smaller.
# 5% of 1280×720 = ~46 000 px² ≈ 215×215 px minimum — removes seated judges and
# background spectators while keeping standing fighters in the active area.
_MIN_BOX_AREA_FRACTION = float(os.getenv("TRACKING_MIN_BOX_AREA", "0.05"))


def _run_tracking(
    model: Any,
    video_path: str,
    class_id: int,
    start: Optional[float],
    end: Optional[float],
) -> Tuple[int, int, int, int]:
    """Run YOLOv8+ByteTrack on a video segment, feeding frames one at a time.

    Also detects scene cuts via grayscale histogram Bhattacharyya distance to
    flag when unique_count is inflated by ID recycling across camera cuts.

    Returns (unique_count, max_simultaneous, frames_processed, n_scene_cuts).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0, 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = int((start or 0.0) * src_fps)
    end_f = int((end or (total / src_fps)) * src_fps)
    start_f = max(0, min(start_f, total - 1))
    end_f = max(start_f, min(end_f, total - 1))
    step = max(1, int(src_fps / _SAMPLE_FPS))
    # Evenly subsample if the segment is too long (avoids timeout on hour-long videos).
    n_candidate = max(1, (end_f - start_f) // step + 1)
    if n_candidate > _MAX_FRAMES:
        step = max(step, (end_f - start_f) // _MAX_FRAMES)

    unique_ids: set[int] = set()
    max_simultaneous = 0
    frames_processed = 0
    n_scene_cuts = 0
    prev_hist: Optional[np.ndarray] = None

    idx = start_f
    while idx <= end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            break

        # ── Scene-cut detection (grayscale histogram Bhattacharyya) ──────
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > _SCENE_CUT_THRESHOLD:
                n_scene_cuts += 1
        prev_hist = hist

        # ── ByteTrack tracking ───────────────────────────────────────────
        frame_h, frame_w = bgr.shape[:2]
        frame_area = frame_h * frame_w
        results = model.track(
            source=bgr,
            classes=[class_id],
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            ids = results[0].boxes.id
            xyxy = results[0].boxes.xyxy  # [N, 4] pixel coords
            if ids is not None and xyxy is not None:
                # Audience filter: drop any box whose area is too small relative
                # to the frame.  This removes background crowd / referee detections
                # and keeps only the prominent on-field subjects.
                valid_ids = []
                for box, tid in zip(xyxy.cpu().tolist(), ids.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area / frame_area >= _MIN_BOX_AREA_FRACTION:
                        valid_ids.append(int(tid))
                unique_ids.update(valid_ids)
                max_simultaneous = max(max_simultaneous, len(valid_ids))
        frames_processed += 1
        idx += step

    cap.release()
    gc.collect()
    return len(unique_ids), max_simultaneous, frames_processed, n_scene_cuts


def run(request: SkillRequest, meta: SkillMetadata) -> SkillResponse:
    # ── Step 1: load model ─────────────────────────────────────────────
    try:
        model = _load_model()
    except Exception as exc:
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[tracking] Model load error: {exc}",
            artifacts={"error": str(exc)},
        )

    # ── Step 2: determine target class ─────────────────────────────────
    target_name, class_id = _get_target_class(request.question)
    if class_id is None:
        return SkillResponse(
            skill_name=meta.name,
            summary=(
                f"[tracking] '{target_name}' is not a trackable COCO class. "
                "Cannot count this type of entity with ByteTrack."
            ),
            artifacts={"error": "non_trackable_class", "target": target_name},
        )

    # ── Step 3: run tracking over time window ──────────────────────────
    start, end = request.normalized_window()
    try:
        unique_count, max_simultaneous, n_frames, n_cuts = _run_tracking(
            model, request.video_path, class_id, start, end
        )
    except Exception as exc:
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[tracking] Tracking error: {exc}",
            artifacts={"error": str(exc)},
        )

    if n_frames == 0:
        return SkillResponse(
            skill_name=meta.name,
            summary="[tracking] No frames extracted from the video segment.",
            artifacts={"error": "no_frames"},
        )

    # ── Step 4: format evidence ────────────────────────────────────────
    window_str = ""
    if start is not None or end is not None:
        s = f"{start:.1f}s" if start is not None else "0s"
        e = f"{end:.1f}s" if end is not None else "end"
        window_str = f" ({s}–{e})"

    # When scene cuts are detected ByteTrack re-assigns IDs on each cut,
    # inflating unique_count.  In that case, max_simultaneous is the more
    # reliable count (highest number of entities visible at once).
    multi_cut = n_cuts >= 2
    if multi_cut:
        reliable_count = max_simultaneous
        cut_note = (
            f" WARNING: {n_cuts} scene cuts detected — ByteTrack IDs reset "
            f"across cuts, so unique_count ({unique_count}) is inflated. "
            f"The reliable count is max_simultaneous = {max_simultaneous}."
        )
    else:
        reliable_count = unique_count
        cut_note = f" ({n_cuts} scene cut(s) detected, unique_count reliable)."

    summary = (
        f"[tracking] Tracked '{target_name}' across {n_frames} sampled frames"
        f"{window_str}. "
        f"unique_count={unique_count}, max_simultaneous={max_simultaneous}."
        f"{cut_note}"
    )

    return SkillResponse(
        skill_name=meta.name,
        summary=summary,
        artifacts={
            "target_class": target_name,
            "class_id": class_id,
            "unique_count": unique_count,
            "max_simultaneous": max_simultaneous,
            "reliable_count": reliable_count,
            "n_scene_cuts": n_cuts,
            "frames_processed": n_frames,
        },
    )
