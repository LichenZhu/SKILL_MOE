"""
Kinematic Storyboard Prompting — 4-frame motion grid with trajectory arrows.

Paradigm: "Tools as Spatio-Temporal Attention" (Motion Edition)
──────────────────────────────────────────────────────────────
For questions about motion direction, trajectory, or action sequences
(e.g., "which direction does X move", "how does Y approach Z"):

  1. Extract 4 equidistant keyframes from the relevant temporal window.
  2. Compute dense optical flow between consecutive frame pairs
     (Farneback, CPU-based, fast).
  3. Identify the centroid of the highest-magnitude motion blob in each frame.
  4. Apply Spotlight encoding: darken the whole frame, brighten a bounding box
     around the motion region, draw a red rectangle — same as focus_vqa.
  5. Arrange the 4 spotlight frames in a 2×2 grid.
  6. Overlay red trajectory arrows connecting the motion centroids across
     frames, making the object's path spatially explicit.
  7. Return the storyboard grid as visual_evidence for direct VLM injection.

This eliminates the temporal-order ambiguity that VLMs face when sampling
64 uniform frames from a full video: the 4-frame progression directly encodes
the kinematic history of the most dynamic object in the clip.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
_NUM_FRAMES     = int(os.getenv("KSB_NUM_FRAMES",     "4"))    # always 4 for 2×2 grid
_MOTION_PAD     = float(os.getenv("KSB_MOTION_PAD",   "0.15")) # spotlight padding fraction
_THUMB_SIZE     = int(os.getenv("KSB_THUMB_SIZE",     "512"))  # each cell is THUMB_SIZE px (long edge)
_JPEG_QUALITY   = int(os.getenv("KSB_JPEG_QUALITY",   "88"))
_FLOW_BLUR      = int(os.getenv("KSB_FLOW_BLUR",      "5"))    # Gaussian blur before flow (px)
_MIN_MOTION_PX  = int(os.getenv("KSB_MIN_MOTION_PX",  "20"))   # ignore blobs smaller than this


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(
    video_path: str,
    start: Optional[float],
    end: Optional[float],
    n: int,
) -> List[Tuple[float, np.ndarray]]:
    """Return [(timestamp_s, bgr_frame), ...] at n equidistant positions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s_f = int(start * fps) if start is not None else 0
    e_f = int(end   * fps) if end   is not None else max(total - 1, 0)
    e_f = min(e_f, total - 1)
    if s_f >= e_f:
        s_f = max(0, e_f - n)
    indices = np.linspace(s_f, e_f, n, dtype=int)
    result: List[Tuple[float, np.ndarray]] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if ok:
            result.append((fi / fps, bgr))
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Motion detection via dense optical flow
# ---------------------------------------------------------------------------

def _motion_centroid(
    bgr_prev: np.ndarray,
    bgr_curr: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute Farneback optical flow, find the bounding box of the largest
    high-magnitude motion blob.  Returns (x1, y1, x2, y2) or None.
    """
    h, w = bgr_curr.shape[:2]
    # Downscale for speed — flow is estimated at 1/4 resolution, upscaled.
    scale = 0.25
    sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
    small_prev = cv2.resize(bgr_prev, (sw, sh))
    small_curr = cv2.resize(bgr_curr, (sw, sh))

    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)

    if _FLOW_BLUR > 1:
        gray_prev = cv2.GaussianBlur(gray_prev, (_FLOW_BLUR, _FLOW_BLUR), 0)
        gray_curr = cv2.GaussianBlur(gray_curr, (_FLOW_BLUR, _FLOW_BLUR), 0)

    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # Threshold at 75th percentile to focus on highest motion.
    thresh = np.percentile(mag, 75)
    motion_mask = (mag > thresh).astype(np.uint8) * 255

    # Find the largest connected component.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(motion_mask, connectivity=8)
    if n_labels <= 1:
        return None

    # Skip label 0 (background); find largest foreground blob.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1  # +1 because we skipped label 0
    if stats[best, cv2.CC_STAT_AREA] < _MIN_MOTION_PX:
        return None

    # Scale bounding box back to full resolution.
    inv = 1.0 / scale
    x1 = int(stats[best, cv2.CC_STAT_LEFT]   * inv)
    y1 = int(stats[best, cv2.CC_STAT_TOP]    * inv)
    bw = int(stats[best, cv2.CC_STAT_WIDTH]  * inv)
    bh = int(stats[best, cv2.CC_STAT_HEIGHT] * inv)
    x2 = min(w - 1, x1 + bw)
    y2 = min(h - 1, y1 + bh)
    return x1, y1, x2, y2


def _fallback_centroid(bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """Fallback: use the centre quarter of the frame as the motion region."""
    h, w = bgr.shape[:2]
    return w // 4, h // 4, 3 * w // 4, 3 * h // 4


# ---------------------------------------------------------------------------
# Spotlight encoding (same style as focus_vqa)
# ---------------------------------------------------------------------------

def _spotlight_frame(
    bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    thumb_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Apply Spotlight effect to a BGR frame and resize to thumb_size.

    Returns (result_bgr, centroid_px) where centroid_px is in thumbnail coords.
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box

    arr = bgr.astype(np.float32)
    dark = (arr * 0.45).astype(np.uint8)

    pad = max(6, int(min(h, w) * 0.03))
    rx1 = max(0, x1 - pad);  ry1 = max(0, y1 - pad)
    rx2 = min(w, x2 + pad);  ry2 = min(h, y2 + pad)

    result = dark.copy()
    result[ry1:ry2, rx1:rx2] = arr[ry1:ry2, rx1:rx2].astype(np.uint8)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 220), 3)

    # Resize so long edge = thumb_size.
    long_edge = max(h, w)
    scale = thumb_size / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    thumb = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Centroid of box in thumbnail coords.
    cx = int(((x1 + x2) / 2) * scale)
    cy = int(((y1 + y2) / 2) * scale)
    return thumb, (cx, cy)


# ---------------------------------------------------------------------------
# 2×2 grid composition + trajectory arrows
# ---------------------------------------------------------------------------

def _compose_storyboard(
    thumbnails: List[np.ndarray],
    centroids: List[Tuple[int, int]],
    timestamps: List[float],
    thumb_size: int,
) -> str:
    """
    Assemble thumbnails into a 2×2 grid (BGR) and draw red trajectory arrows
    connecting centroids across frames (left→right, top→bottom reading order).

    Returns base64 JPEG string.
    """
    # Pad to exactly 4 if fewer thumbnails were provided (shouldn't happen, but guard).
    while len(thumbnails) < 4:
        thumbnails.append(np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))
        centroids.append((thumb_size // 2, thumb_size // 2))
        timestamps = list(timestamps) + [timestamps[-1] if timestamps else 0.0]
    thumbnails  = thumbnails[:4]
    centroids   = centroids[:4]
    timestamps  = list(timestamps)[:4]

    # Pad each thumbnail to a square of thumb_size×thumb_size.
    padded: List[np.ndarray] = []
    for i, thumb in enumerate(thumbnails):
        h, w = thumb.shape[:2]
        canvas = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
        canvas[:h, :w] = thumb
        # Timestamp label.
        ts = timestamps[i]
        label = f"t={int(ts // 60):02d}:{int(ts % 60):02d}"
        cv2.putText(canvas, label, (6, thumb_size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 80), 1, cv2.LINE_AA)
        # Frame index badge.
        cv2.putText(canvas, f"[{i+1}]", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        padded.append(canvas)

    top    = np.hstack([padded[0], padded[1]])
    bottom = np.hstack([padded[2], padded[3]])
    grid   = np.vstack([top, bottom])

    # Absolute centroid positions on the 2×2 grid.
    offsets = [(0, 0), (thumb_size, 0), (0, thumb_size), (thumb_size, thumb_size)]
    abs_centroids = [
        (cx + ox, cy + oy)
        for (cx, cy), (ox, oy) in zip(centroids, offsets)
    ]

    # Draw trajectory arrows connecting consecutive frames.
    # Route: frame 1→2 (top row), frame 3→4 (bottom row), frame 2→3 (cross).
    connections = [(0, 1), (2, 3), (1, 2)]
    for a_idx, b_idx in connections:
        pt_a = abs_centroids[a_idx]
        pt_b = abs_centroids[b_idx]
        cv2.arrowedLine(
            grid, pt_a, pt_b,
            color=(0, 0, 220),     # red in BGR
            thickness=3,
            tipLength=0.15,
            line_type=cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(request: SkillRequest, meta: SkillMetadata) -> SkillResponse:
    """
    Build a 4-frame kinematic storyboard with motion spotlight + trajectory arrows.
    Returns visual_evidence = [storyboard_b64] for direct VLM injection.
    """
    start, end = request.normalized_window()

    # ── Step 1: Extract 4 equidistant frames ─────────────────────────────
    frames = _extract_frames(request.video_path, start, end, _NUM_FRAMES)
    if len(frames) < 2:
        return SkillResponse(
            skill_name=meta.name,
            summary="[ActionStoryboard] Not enough frames extracted.",
            artifacts={"error": "no_frames", "no_match": True},
        )

    # Pad to exactly 4 by repeating the last frame if needed.
    while len(frames) < _NUM_FRAMES:
        frames.append(frames[-1])
    frames = frames[:_NUM_FRAMES]

    timestamps = [ts for ts, _ in frames]
    bgr_frames = [bgr for _, bgr in frames]

    # ── Step 2: Motion detection via optical flow ─────────────────────────
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(_NUM_FRAMES):
        prev_bgr = bgr_frames[i - 1] if i > 0 else bgr_frames[i]
        curr_bgr = bgr_frames[i]
        try:
            box = _motion_centroid(prev_bgr, curr_bgr)
        except Exception as exc:
            logger.debug("[action_storyboard] Flow failed at frame %d: %s", i, exc)
            box = None
        boxes.append(box or _fallback_centroid(curr_bgr))

    # Smooth box positions: use the union bounding box across all frames so the
    # spotlight remains spatially consistent (avoids disorienting jumps).
    all_x1 = [b[0] for b in boxes];  all_y1 = [b[1] for b in boxes]
    all_x2 = [b[2] for b in boxes];  all_y2 = [b[3] for b in boxes]
    # Median position for stability (not union — union may cover whole frame).
    med_x1 = int(np.median(all_x1));  med_y1 = int(np.median(all_y1))
    med_x2 = int(np.median(all_x2));  med_y2 = int(np.median(all_y2))

    # Keep per-frame centroids for arrows but use median box for spotlight.
    uniform_box = (med_x1, med_y1, med_x2, med_y2)

    # ── Step 3: Spotlight + thumbnail for each frame ───────────────────────
    thumbnails: List[np.ndarray] = []
    centroids:  List[Tuple[int, int]] = []
    for i, bgr in enumerate(bgr_frames):
        try:
            thumb, centroid = _spotlight_frame(bgr, uniform_box, _THUMB_SIZE)
        except Exception as exc:
            logger.warning("[action_storyboard] Spotlight failed frame %d: %s — using raw resize", i, exc)
            h_s, w_s = bgr.shape[:2]
            long_e = max(h_s, w_s)
            sc = _THUMB_SIZE / long_e
            thumb = cv2.resize(bgr, (int(w_s * sc), int(h_s * sc)), interpolation=cv2.INTER_AREA)
            centroid = (_THUMB_SIZE // 2, _THUMB_SIZE // 2)
        thumbnails.append(thumb)
        centroids.append(centroid)

    # ── Step 4: Compose 2×2 grid with trajectory arrows ───────────────────
    try:
        storyboard_b64 = _compose_storyboard(thumbnails, centroids, timestamps, _THUMB_SIZE)
    except Exception as exc:
        logger.error("[action_storyboard] Grid composition failed: %s", exc)
        storyboard_b64 = ""
    if not storyboard_b64:
        return SkillResponse(
            skill_name=meta.name,
            summary="[ActionStoryboard] Grid encoding failed.",
            artifacts={"error": "encode_failed", "no_match": True},
        )

    t_str = ", ".join(f"{ts:.1f}s" for ts in timestamps)
    summary = (
        f"[ActionStoryboard] 4-frame kinematic storyboard built "
        f"({t_str}). Motion region highlighted; trajectory arrows added."
    )

    # ── Debug disk dump (visual sanity check) ─────────────────────────────
    # Saves the storyboard to ./debug_outputs/ for manual inspection.
    # The directory is created if absent; errors are silently ignored.
    try:
        import base64 as _b64
        _debug_dir = os.path.join(os.getcwd(), "debug_outputs")
        os.makedirs(_debug_dir, exist_ok=True)
        _vid_stem = os.path.splitext(os.path.basename(request.video_path))[0]
        _debug_path = os.path.join(_debug_dir, f"action_storyboard_{_vid_stem}.jpg")
        _raw_jpg = _b64.b64decode(storyboard_b64)
        with open(_debug_path, "wb") as _f:
            _f.write(_raw_jpg)
        logger.info("[action_storyboard] Debug image saved → %s", _debug_path)
    except Exception as _exc:
        logger.debug("[action_storyboard] Debug dump skipped: %s", _exc)

    return SkillResponse(
        skill_name=meta.name,
        summary=summary,
        content=(
            f"A 4-frame storyboard (2×2 grid) of the main motion in this clip "
            f"({t_str}). Red arrows show the spatial trajectory of the moving "
            "object across consecutive frames. Use this to determine direction, "
            "approach/retreat, or action sequence."
        ),
        artifacts={
            "target":          "kinematic trajectory",
            "crops_found":     1,
            "frame_timestamps": [round(ts, 2) for ts in timestamps],
            "visual_evidence": [storyboard_b64],
        },
    )
