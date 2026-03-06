"""
Focus VQA — Spatio-Temporal Attention Slicer.

Paradigm: "Tools as Spatio-Temporal Attention"
──────────────────────────────────────────────
Instead of converting visual information to text (which destroys detail),
this skill acts as a *microscope*:

  1. LLM extracts the target object/region from the question.
  2. GroundingDINO localises the target in N evenly-spaced keyframes.
  3. The best 1-3 high-resolution crops are stored in
     artifacts['visual_evidence'] as base64 JPEG strings.
  4. The pipeline injects these crops DIRECTLY into Qwen2.5-Omni alongside
     the original video, letting the VLM reason with both temporal context
     (from the video) and spatial detail (from the crops).

No text intermediary → no information loss → no Tool Sycophancy.
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
from PIL import Image

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
_NUM_FRAMES       = int(os.getenv("FOCUS_VQA_NUM_FRAMES",      "8"))   # keyframes to scan
_MAX_CROPS        = int(os.getenv("FOCUS_VQA_MAX_CROPS",        "3"))   # crops sent to VLM
_CROP_PAD_FRAC    = float(os.getenv("FOCUS_VQA_PAD",           "0.25")) # context padding
_BOX_THRESHOLD    = float(os.getenv("FOCUS_VQA_BOX_THRESHOLD", "0.22")) # GDINO confidence
_MAX_BOX_AREA     = float(os.getenv("FOCUS_VQA_MAX_BOX_AREA",  "0.60")) # giant-box filter
_MIN_CROP_PX      = int(os.getenv("FOCUS_VQA_MIN_CROP_PX",     "64"))   # too-small filter
_CROP_MAX_SIDE    = int(os.getenv("FOCUS_VQA_CROP_MAX_SIDE",   "1024")) # keep detail

# Module-level cache for GroundingDINO (shared with grounding skill when loaded).
_local_gdino_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

_TARGET_PROMPT = """\
Extract the main physical object or region the question needs examined visually.
Return ONLY 1-6 words, no explanation, no punctuation.

Examples:
Q: "What is written on the red sign?" → red sign
Q: "What is the man holding in his left hand?" → object in left hand
Q: "What logo appears on the shirt?" → shirt logo
Q: "What does the label say?" → label
Q: "What color is the small bottle on the shelf?" → small bottle

Question: {question}
Object:"""


def _extract_target(question: str, llm) -> str:
    if llm is not None:
        try:
            raw = llm.complete(
                _TARGET_PROMPT.format(question=question[:400]), max_tokens=16
            ).strip().strip('"\'').strip()
            if raw and len(raw) < 70:
                return raw
        except Exception:
            pass

    q = question.lower()
    for pat in [
        r"written on (?:the |a )?(.{3,30}?)(?:\?|$|\n)",
        r"(?:logo|text|label|sign|name) (?:on|of) (?:the |a )?(.{3,25}?)(?:\?|,|$)",
        r"holding (?:in (?:his|her|the) \w+ hand )?(?:a |the )?(.{3,25}?)(?:\?|$|\n)",
        r"what (?:is|does|are) (?:the |a |an )?(.{3,30}?) (?:say|show|display|read)\b",
        r"what (?:is|are) (?:the |a |an )?(.{3,30}?) (?:holding|wearing)\b",
    ]:
        m = re.search(pat, q)
        if m:
            return m.group(1).strip()

    m = re.search(r"what (?:is|are|does) (?:the |a |an )?(\w+(?:\s+\w+){0,2})", q)
    if m:
        return m.group(1).strip()
    return "object"


# ---------------------------------------------------------------------------
# GroundingDINO loader (reuses grounding skill GPU cache)
# ---------------------------------------------------------------------------

def _load_grounding() -> Tuple[Any, Any, str]:
    """Return (processor, gdino_model, device), reusing grounding skill cache."""
    try:
        import importlib
        grounding = importlib.import_module("skills.grounding.runner")
        return grounding._load_model()
    except Exception:
        pass

    if "model" in _local_gdino_cache:
        return (
            _local_gdino_cache["processor"],
            _local_gdino_cache["model"],
            _local_gdino_cache["device"],
        )

    import torch
    from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

    model_id = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = GroundingDinoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id).eval().to(device)
    _local_gdino_cache.update(processor=proc, model=model, device=device)
    return proc, model, device


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(
    video_path: str,
    start: Optional[float],
    end: Optional[float],
    n: int,
) -> List[Tuple[float, Image.Image]]:
    """Return [(timestamp_sec, PIL_Image), ...] at full native resolution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(start * fps) if start is not None else 0
    end_f   = int(end   * fps) if end   is not None else max(total - 1, 0)
    end_f   = min(end_f, total - 1)
    if start_f >= end_f:
        start_f = max(0, end_f - 1)
    indices = np.linspace(start_f, end_f, min(n, end_f - start_f + 1), dtype=int)
    result: List[Tuple[float, Image.Image]] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if ok:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result.append((fi / fps, pil))
    cap.release()
    return result


# ---------------------------------------------------------------------------
# Detection + cropping
# ---------------------------------------------------------------------------

def _detect_and_crop(
    pil_img: Image.Image,
    target: str,
    processor: Any,
    model: Any,
    device: str,
) -> Optional[Tuple[float, Image.Image, Tuple[float, float, float, float]]]:
    """Detect target, return (confidence, padded_crop, orig_box) or None.

    orig_box is the raw GroundingDINO bounding box (before padding), used by
    _spotlight_encode to determine the highlight region on the full frame.
    """
    import torch

    text_q = target.strip().rstrip(".") + " ."
    inputs = processor(images=pil_img, text=text_q, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            outputs = model(**inputs)

    W, H = pil_img.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=_BOX_THRESHOLD,
        text_threshold=_BOX_THRESHOLD,
        target_sizes=[(H, W)],
    )[0]

    if len(results["boxes"]) == 0:
        return None

    # Filter giant false-positive boxes (usually full-scene GDINO artefacts).
    valid: List[Tuple[float, List[float]]] = []
    for i, box in enumerate(results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        area_frac = (x2 - x1) * (y2 - y1) / (W * H + 1e-6)
        if area_frac < _MAX_BOX_AREA:
            valid.append((float(results["scores"][i]), box.tolist()))

    if not valid:
        return None

    score, best_box = max(valid, key=lambda t: t[0])
    orig_box = tuple(best_box)  # save original GDINO box before padding

    x1, y1, x2, y2 = best_box

    # Add padding to retain context around the object.
    pw = (x2 - x1) * _CROP_PAD_FRAC
    ph = (y2 - y1) * _CROP_PAD_FRAC
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(W, int(x2 + pw))
    y2 = min(H, int(y2 + ph))

    crop = pil_img.crop((x1, y1, x2, y2))
    if crop.width < _MIN_CROP_PX or crop.height < _MIN_CROP_PX:
        return None
    return score, crop, orig_box


# ---------------------------------------------------------------------------
# Spotlight encoding  (replaces plain crop)
# ---------------------------------------------------------------------------

def _spotlight_encode(pil_img: Image.Image, box: Tuple[float, float, float, float]) -> str:
    """
    Context-Preserving Spotlight Effect:
    1. Darken the entire frame to 50% brightness (alpha-blend with black).
    2. Restore the target bounding-box region to 100% brightness.
    3. Draw a red rectangle around the target.
    4. Resize the full annotated frame so its long edge ≤ _CROP_MAX_SIDE.
    Returns base64 JPEG, or "" on failure.

    This lets the VLM see both fine-grained spatial detail (bright target)
    AND global context (dimmed surroundings), eliminating crop-induced
    color/lighting hallucinations.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    arr = np.array(pil_img, dtype=np.float32)

    # Darken entire frame to 50%
    dark = (arr * 0.5).astype(np.uint8)

    # Restore target region (with a small padding so the bright area is generous)
    h_img, w_img = arr.shape[:2]
    pad = max(8, int(min(h_img, w_img) * 0.03))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(w_img, x2 + pad)
    ry2 = min(h_img, y2 + pad)
    result = dark.copy()
    result[ry1:ry2, rx1:rx2] = arr[ry1:ry2, rx1:rx2].astype(np.uint8)

    # Draw red bounding box (BGR for OpenCV)
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Resize so long edge ≤ _CROP_MAX_SIDE
    fh, fw = bgr.shape[:2]
    long_edge = max(fw, fh)
    if long_edge > _CROP_MAX_SIDE:
        scale = _CROP_MAX_SIDE / long_edge
        bgr = cv2.resize(bgr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_LANCZOS4)

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(request: SkillRequest, meta: SkillMetadata) -> SkillResponse:
    """
    Locate target object in video → store high-res crops in visual_evidence.

    The pipeline (pipeline.py _answer_with_visual_crops) picks these up and
    feeds them DIRECTLY into Qwen2.5-Omni alongside the video, so the model
    sees both temporal context AND fine-grained spatial detail — no text
    conversion, no information loss.
    """
    llm = default_llm_client()

    # ── Step 1: extract target phrase ────────────────────────────────────────
    target = _extract_target(request.question, llm)
    logger.info("[focus_vqa] target='%s'", target)

    # ── Step 2: load GroundingDINO ────────────────────────────────────────────
    try:
        processor, gdino_model, device = _load_grounding()
    except Exception as exc:
        logger.warning("[focus_vqa] GroundingDINO unavailable: %s", exc)
        return SkillResponse(
            skill_name=meta.name,
            summary="[Focus-VQA] GroundingDINO unavailable.",
            artifacts={"error": str(exc), "no_match": True},
        )

    # ── Step 3: sample keyframes ──────────────────────────────────────────────
    start, end = request.normalized_window()
    frame_ts_pairs = _sample_frames(request.video_path, start, end, _NUM_FRAMES)
    if not frame_ts_pairs:
        return SkillResponse(
            skill_name=meta.name,
            summary="[Focus-VQA] No frames extracted.",
            artifacts={"error": "no_frames", "no_match": True},
        )

    # ── Step 4: Pre-downscale frames to ≤1024px before GroundingDINO ─────────
    # 4K frames (3840×2160) are ~16× larger than 1024px — GroundingDINO on them
    # is disproportionately slow and yields no detection quality improvement.
    # Pre-resize here; _spotlight_encode receives the same resized frame so box
    # coordinates remain consistent without any extra scaling step.
    _MAX_DETECT_PX = _CROP_MAX_SIDE  # reuse the 1024px constant
    scaled_pairs: List[Tuple[float, Image.Image]] = []
    for ts, pil_img in frame_ts_pairs:
        w, h = pil_img.size
        long_edge = max(w, h)
        if long_edge > _MAX_DETECT_PX:
            scale = _MAX_DETECT_PX / long_edge
            pil_img = pil_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        scaled_pairs.append((ts, pil_img))

    # ── Step 5: detect & rank ─────────────────────────────────────────────────
    # Store (score, ts, crop, orig_box, full_pil) for spotlight encoding.
    detections: List[Tuple[float, float, Image.Image, Any, Image.Image]] = []
    for ts, pil_img in scaled_pairs:
        try:
            result = _detect_and_crop(pil_img, target, processor, gdino_model, device)
        except Exception as exc:
            logger.debug("[focus_vqa] Detection failed at %.1fs: %s", ts, exc)
            continue
        if result is not None:
            score, crop, orig_box = result
            detections.append((score, ts, crop, orig_box, pil_img))

    if not detections:
        logger.info("[focus_vqa] '%s' not found in any frame", target)
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[Focus-VQA] '{target}' not detected in video.",
            content="",
            artifacts={"target": target, "crops_found": 0, "no_match": True},
        )

    # ── Step 6: pick top-K crops by detection confidence ─────────────────────
    detections.sort(key=lambda t: t[0], reverse=True)
    top = detections[:_MAX_CROPS]

    # ── Step 7: encode as spotlight images (full-frame with darkened surround) ─
    visual_evidence: List[str] = []
    crop_meta: List[Dict[str, Any]] = []
    for score, ts, crop, orig_box, full_pil in top:
        b64 = _spotlight_encode(full_pil, orig_box)
        if not b64:
            continue
        visual_evidence.append(b64)
        crop_meta.append({
            "timestamp_sec": round(ts, 2),
            "gdino_score":   round(score, 3),
            "crop_w":        crop.width,
            "crop_h":        crop.height,
        })

    timestamps_str = ", ".join(f"{m['timestamp_sec']:.1f}s" for m in crop_meta)
    summary = (
        f"[Focus-VQA] '{target}' found in {len(visual_evidence)} frame(s) "
        f"({timestamps_str}). High-res crops ready for VLM inspection."
    )

    logger.info(
        "[focus_vqa] %d crops encoded for '%s' — pipeline will inject into VLM",
        len(visual_evidence), target,
    )

    return SkillResponse(
        skill_name=meta.name,
        summary=summary,
        # content is intentionally sparse — the real evidence is visual, not textual.
        content=(
            f"Visual attention focused on '{target}' at {timestamps_str}. "
            "High-resolution crops will be injected directly into the VLM for inspection."
        ),
        artifacts={
            "target":          target,
            "crops_found":     len(visual_evidence),
            "crop_meta":       crop_meta,
            # ← The key payload: base64 JPEG strings consumed by pipeline.py
            "visual_evidence": visual_evidence,
        },
    )
