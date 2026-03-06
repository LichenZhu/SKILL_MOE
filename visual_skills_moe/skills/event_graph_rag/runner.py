"""
Event-Graph RAG — Hierarchical Semantic Scene Retrieval for Long Videos.

Architecture: CLIP-based Visual Scene Indexing
─────────────────────────────────────────────
For long videos where the VLM sees ≤64 frames (sparse sampling), important
events are easily missed.  This skill builds a lightweight in-memory scene
index and retrieves exactly the segments most relevant to the question.

Pipeline:
  1. Sample frames at 0.5 fps (up to MAX_SCENES frames).
  2. Detect scene cuts by RGB histogram L1 distance (fast, no neural net).
  3. For each scene, encode its representative keyframe with CLIP (ViT-H-14).
  4. Encode the question stem with CLIP text encoder.
  5. Retrieve top-K (default 3) scenes by cosine similarity.
  6. Build a storyboard grid image from the top-K keyframes with timestamp
     annotations.  Return as visual_evidence for direct VLM injection.

No per-scene LLM call → no API cost, fully local, scales to 2-hour videos.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs (overridable via env vars)
# ---------------------------------------------------------------------------
_SAMPLE_FPS      = float(os.getenv("EGR_SAMPLE_FPS",       "0.5"))  # frames/sec
_MAX_SCENES      = int(os.getenv("EGR_MAX_SCENES",          "240"))  # cap scene count
_CUT_THRESHOLD   = float(os.getenv("EGR_CUT_THRESHOLD",    "0.35"))  # histogram L1 delta
_TOP_K           = int(os.getenv("EGR_TOP_K",                 "3"))  # scenes returned
_GRID_THUMB_W    = int(os.getenv("EGR_THUMB_W",             "480"))  # thumbnail width
_GRID_THUMB_H    = int(os.getenv("EGR_THUMB_H",             "270"))  # thumbnail height
_CLIP_MODEL      = os.getenv("EGR_CLIP_MODEL",
                             "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
_CLIP_PRETRAINED = os.getenv("EGR_CLIP_PRETRAINED", "")

_CLIP_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# CLIP model loader (shares cache with visual_option_match when same model)
# ---------------------------------------------------------------------------

def _load_clip() -> Tuple[Any, Any, Any]:
    """Return (model, preprocess, tokenizer).  Cached after first load."""
    key = f"{_CLIP_MODEL}|{_CLIP_PRETRAINED}"
    if key in _CLIP_CACHE:
        return _CLIP_CACHE[key]["model"], _CLIP_CACHE[key]["preprocess"], _CLIP_CACHE[key]["tokenizer"]

    import torch
    import open_clip  # type: ignore

    if _CLIP_PRETRAINED:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL, pretrained=_CLIP_PRETRAINED
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(_CLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    _CLIP_CACHE[key] = dict(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)
    return model, preprocess, tokenizer


def _clip_device() -> str:
    key = f"{_CLIP_MODEL}|{_CLIP_PRETRAINED}"
    return _CLIP_CACHE.get(key, {}).get("device", "cpu")


# ---------------------------------------------------------------------------
# Frame sampling and scene cut detection
# ---------------------------------------------------------------------------

def _sample_video_frames(
    video_path: str,
    sample_fps: float,
    max_frames: int,
) -> List[Tuple[float, np.ndarray]]:
    """Return [(timestamp_s, bgr_frame), ...] at sample_fps, capped at max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(native_fps / sample_fps))
    indices = list(range(0, total_frames, step))[:max_frames]
    result: List[Tuple[float, np.ndarray]] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if ok:
            result.append((fi / native_fps, bgr))
    cap.release()
    return result


def _detect_scene_boundaries(
    frames: List[Tuple[float, np.ndarray]],
    threshold: float,
) -> List[int]:
    """Return indices of frames that start a new scene (via histogram L1 delta)."""
    if not frames:
        return []
    boundaries = [0]  # first frame always starts a scene
    prev_hist = _compute_hist(frames[0][1])
    for i in range(1, len(frames)):
        curr_hist = _compute_hist(frames[i][1])
        delta = float(np.sum(np.abs(curr_hist - prev_hist)))
        if delta > threshold:
            boundaries.append(i)
        prev_hist = curr_hist
    return boundaries


def _compute_hist(bgr: np.ndarray) -> np.ndarray:
    """Normalised 3-channel RGB histogram (16 bins per channel)."""
    hists = []
    for ch in range(3):
        h = cv2.calcHist([bgr], [ch], None, [16], [0, 256]).flatten()
        h /= h.sum() + 1e-6
        hists.append(h)
    return np.concatenate(hists)


# ---------------------------------------------------------------------------
# CLIP embedding helpers
# ---------------------------------------------------------------------------

def _embed_images(images: List[Image.Image], model: Any, preprocess: Any) -> np.ndarray:
    """Return (N, D) float32 array of CLIP image embeddings (L2-normalised)."""
    import torch
    device = _clip_device()
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(tensors).float()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
    return feats.cpu().numpy()


def _embed_text(text: str, model: Any, tokenizer: Any) -> np.ndarray:
    """Return (D,) float32 CLIP text embedding (L2-normalised)."""
    import torch
    device = _clip_device()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens).float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
    return feat.cpu().numpy()[0]


def _cosine_sim(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """(N, D) @ (D,) → (N,) cosine similarities."""
    return mat @ vec


# ---------------------------------------------------------------------------
# Storyboard grid builder
# ---------------------------------------------------------------------------

def _build_storyboard(
    keyframes: List[Tuple[float, np.ndarray]],
    thumb_w: int,
    thumb_h: int,
) -> str:
    """
    Compose selected keyframes into a horizontal storyboard strip.

    Each panel: thumbnail + black bar at bottom with timestamp label.
    Returns base64 JPEG string.
    """
    panels: List[Image.Image] = []
    bar_h = 24
    for ts, bgr in keyframes:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((thumb_w, thumb_h), Image.LANCZOS)
        # Add timestamp bar
        canvas = Image.new("RGB", (thumb_w, thumb_h + bar_h), (0, 0, 0))
        canvas.paste(pil, (0, 0))
        draw = ImageDraw.Draw(canvas)
        label = f"{int(ts // 60):02d}:{int(ts % 60):02d}"
        draw.text((4, thumb_h + 4), label, fill=(255, 220, 50))
        panels.append(canvas)

    if not panels:
        return ""

    total_w = thumb_w * len(panels)
    total_h = thumb_h + bar_h
    grid = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    for i, panel in enumerate(panels):
        grid.paste(panel, (i * thumb_w, 0))

    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(request: SkillRequest, meta: SkillMetadata) -> SkillResponse:
    """
    Build a CLIP scene index and retrieve the top-K question-relevant scenes.

    Returns visual_evidence = [storyboard_b64] so the pipeline injects the
    storyboard grid directly into Qwen2.5-Omni alongside the video.
    """
    # ── Step 1: Sample frames ──────────────────────────────────────────────
    frames = _sample_video_frames(request.video_path, _SAMPLE_FPS, _MAX_SCENES)
    if not frames:
        return SkillResponse(
            skill_name=meta.name,
            summary="[EventGraphRAG] No frames extracted.",
            artifacts={"error": "no_frames", "no_match": True},
        )

    logger.info("[event_graph_rag] Sampled %d frames from %s", len(frames), request.video_path)

    # ── Step 2: Detect scene boundaries ───────────────────────────────────
    try:
        boundaries = _detect_scene_boundaries(frames, _CUT_THRESHOLD)
    except Exception as exc:
        logger.warning("[event_graph_rag] Scene detection failed (%s); treating all frames as scenes", exc)
        boundaries = list(range(len(frames)))
    # One representative frame per scene (the first frame of the scene).
    scene_frames: List[Tuple[float, np.ndarray]] = [frames[b] for b in boundaries]
    logger.info("[event_graph_rag] Detected %d scenes", len(scene_frames))

    # ── Step 3: Load CLIP and embed scene keyframes ────────────────────────
    try:
        model, preprocess, tokenizer = _load_clip()
    except Exception as exc:
        logger.warning("[event_graph_rag] CLIP unavailable: %s", exc)
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[EventGraphRAG] CLIP unavailable: {exc}",
            artifacts={"error": str(exc), "no_match": True},
        )

    try:
        pil_frames = [
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for _, bgr in scene_frames
        ]
    except Exception as exc:
        logger.error("[event_graph_rag] Frame conversion failed: %s", exc)
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[EventGraphRAG] Frame conversion failed: {exc}",
            artifacts={"error": str(exc), "no_match": True},
        )

    # Process in batches of 32 to avoid OOM on large scene counts.
    all_feats: List[np.ndarray] = []
    batch = 32
    for i in range(0, len(pil_frames), batch):
        try:
            all_feats.append(_embed_images(pil_frames[i:i + batch], model, preprocess))
        except Exception as exc:
            logger.warning("[event_graph_rag] CLIP batch %d failed (%s); skipping", i // batch, exc)
    if not all_feats:
        return SkillResponse(
            skill_name=meta.name,
            summary="[EventGraphRAG] All CLIP embedding batches failed.",
            artifacts={"error": "clip_embed_failed", "no_match": True},
        )
    scene_feats = np.vstack(all_feats)  # (N_scenes, D) — may be fewer than scene_frames

    # Trim scene_frames to match the number of successfully embedded frames.
    scene_frames = scene_frames[:len(scene_feats)]

    # ── Step 4: Embed question (stem only, strip MCQ options) ─────────────
    q_stem = re.split(r"\n\s*A\.", request.question or "", maxsplit=1)[0].strip()
    try:
        q_feat = _embed_text(q_stem[:200], model, tokenizer)  # (D,)
    except Exception as exc:
        logger.warning("[event_graph_rag] Question embedding failed (%s); using uniform retrieval", exc)
        # Fallback: pick scenes evenly spaced across the video.
        k = min(_TOP_K, len(scene_frames))
        step = max(1, len(scene_frames) // k)
        top_indices_list = list(range(0, len(scene_frames), step))[:k]
        retrieved = [scene_frames[i] for i in top_indices_list]
        top_indices = np.array(top_indices_list)
        sims = np.zeros(len(scene_frames))
        timestamps_str = ", ".join(f"{int(ts // 60):02d}:{int(ts % 60):02d}" for ts, _ in retrieved)
        logger.info("[event_graph_rag] Fallback uniform retrieval: %s", timestamps_str)
    else:
        # ── Step 5: Retrieve top-K scenes ─────────────────────────────────
        sims = _cosine_sim(scene_feats, q_feat)  # (N_scenes,)
        k = min(_TOP_K, len(scene_frames))
        top_indices = np.argsort(sims)[::-1][:k]
        # Sort retrieved scenes chronologically so the storyboard reads left→right.
        top_indices = sorted(top_indices, key=lambda i: scene_frames[i][0])
        retrieved = [scene_frames[i] for i in top_indices]
        timestamps_str = ", ".join(f"{int(ts // 60):02d}:{int(ts % 60):02d}" for ts, _ in retrieved)

    logger.info(
        "[event_graph_rag] Top-%d scenes at %s",
        k, timestamps_str,
    )

    # ── Step 6: Build storyboard grid ─────────────────────────────────────
    try:
        storyboard_b64 = _build_storyboard(retrieved, _GRID_THUMB_W, _GRID_THUMB_H)
    except Exception as exc:
        logger.error("[event_graph_rag] Storyboard build failed: %s", exc)
        storyboard_b64 = ""
    if not storyboard_b64:
        return SkillResponse(
            skill_name=meta.name,
            summary="[EventGraphRAG] Storyboard encoding failed.",
            artifacts={"error": "encode_failed", "no_match": True},
        )

    summary = (
        f"[EventGraphRAG] Retrieved {k} most question-relevant scene(s) from "
        f"{len(scene_frames)} total scenes.  Timestamps: {timestamps_str}. "
        f"Storyboard injected as visual evidence."
    )
    content = (
        f"The {k} most semantically relevant scenes from this video "
        f"(at {timestamps_str}) are provided as a storyboard image. "
        "Use these frames to reason about events across the full video."
    )

    # ── Debug disk dump ────────────────────────────────────────────────────
    try:
        import base64 as _b64
        _debug_dir = os.path.join(os.getcwd(), "debug_outputs")
        os.makedirs(_debug_dir, exist_ok=True)
        _vid_stem = os.path.splitext(os.path.basename(request.video_path))[0]
        _debug_path = os.path.join(_debug_dir, f"event_graph_rag_{_vid_stem}.jpg")
        with open(_debug_path, "wb") as _f:
            _f.write(_b64.b64decode(storyboard_b64))
        logger.info("[event_graph_rag] Debug storyboard saved → %s", _debug_path)
    except Exception as _exc:
        logger.debug("[event_graph_rag] Debug dump skipped: %s", _exc)

    return SkillResponse(
        skill_name=meta.name,
        summary=summary,
        content=content,
        artifacts={
            "target":          f"top-{k} scenes for: {q_stem[:60]}",
            "crops_found":     k,
            "scene_timestamps": [round(ts, 2) for ts, _ in retrieved],
            "sim_scores":      [round(float(sims[i]), 3) for i in top_indices],
            "visual_evidence": [storyboard_b64],
        },
    )
