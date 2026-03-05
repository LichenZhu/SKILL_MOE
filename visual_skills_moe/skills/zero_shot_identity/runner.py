"""
Zero-Shot Identity skill — Crop-then-CLIP person matcher.

Answers "who is the person with X" or "what role does Y person play" questions
by:
  1. Sampling a few keyframes from the video.
  2. Running YOLOv8n to detect people and crop their bounding boxes.
  3. Scoring each crop against the MCQ options (or LLM-extracted descriptors)
     using OpenCLIP image-text similarity.
  4. Returning the best-matching option/descriptor with confidence.

Falls back to full-frame scoring (visual_option_match behaviour) when no
persons are detected.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

_NUM_FRAMES = int(os.getenv("ZSI_NUM_FRAMES", "4"))
_MAX_CROPS_PER_FRAME = int(os.getenv("ZSI_MAX_CROPS_PER_FRAME", "3"))
_YOLO_CONF = float(os.getenv("ZSI_YOLO_CONF", "0.30"))
_YOLO_MODEL = os.getenv("ZSI_YOLO_MODEL", "yolov8n.pt")

# ── Shared model caches ───────────────────────────────────────────────────────
_YOLO_CACHE: Dict[str, object] = {}
_VISOPT_MODULE = None


def _get_yolo():
    if _YOLO_MODEL in _YOLO_CACHE:
        return _YOLO_CACHE[_YOLO_MODEL]
    from ultralytics import YOLO  # type: ignore
    m = YOLO(_YOLO_MODEL)
    _YOLO_CACHE[_YOLO_MODEL] = m
    return m


def _get_visopt_module():
    global _VISOPT_MODULE
    if _VISOPT_MODULE is not None:
        return _VISOPT_MODULE
    p = Path(__file__).parent.parent / "visual_option_match" / "runner.py"
    spec = importlib.util.spec_from_file_location("_zsi_visopt_impl", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _VISOPT_MODULE = mod
    return mod


# ── Frame sampling ────────────────────────────────────────────────────────────

def _sample_frames(
    video_path: str,
    n: int,
    start_time: Optional[float],
    end_time: Optional[float],
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
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
    return frames


# ── Person crop extraction ────────────────────────────────────────────────────

def _detect_person_crops(
    frames: List[np.ndarray],
) -> List[np.ndarray]:
    """Run YOLOv8 person detector and return top crops across all frames."""
    yolo = _get_yolo()
    crops: List[Tuple[float, np.ndarray]] = []  # (confidence, crop_bgr)

    for frame in frames:
        results = yolo(frame, classes=[0], conf=_YOLO_CONF, verbose=False)
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        # Sort by confidence descending, take top _MAX_CROPS_PER_FRAME
        confs = boxes.conf.cpu().tolist()
        xywhn = boxes.xyxy.cpu().tolist()
        ranked = sorted(zip(confs, xywhn), key=lambda x: -x[0])
        for conf, xyxy in ranked[:_MAX_CROPS_PER_FRAME]:
            x1, y1, x2, y2 = map(int, xyxy)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append((conf, frame[y1:y2, x1:x2]))

    return [c for _, c in crops]


# ── Descriptor extraction ─────────────────────────────────────────────────────

_DESCRIPTOR_PROMPT = """\
List the visual descriptors used to identify a person in this question.
Output a comma-separated list of 2-5 short descriptions (e.g. "white hair", \
"blue jacket", "reporter", "standing at podium").

Question: {question}
Descriptors:"""


def _extract_options(question: str) -> List[Tuple[str, str]]:
    """Parse MCQ options A/B/C/D from the question text."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", question or ""):
        out.append((m.group(1).upper(), m.group(2).strip()))
    return out


def _extract_descriptors(question: str, llm) -> List[Tuple[str, str]]:
    """Fall back: use LLM to extract descriptors; return as (label, text) pairs."""
    if llm is None:
        # Very naive: grab nouns after "with" or "wearing"
        m = re.search(r"\b(?:with|wearing|has|in)\s+([a-z][a-z\s]{1,30})", question.lower())
        if m:
            return [("Desc0", m.group(1).strip())]
        return [("Desc0", "the person being asked about")]
    try:
        raw = llm.complete(
            _DESCRIPTOR_PROMPT.format(question=question[:400]), max_tokens=60
        ).strip()
        parts = [p.strip().strip('"\'') for p in raw.split(",") if p.strip()]
        return [(f"D{i}", p) for i, p in enumerate(parts) if p]
    except Exception:
        return [("D0", "the person being asked about")]


# ── CLIP scoring ─────────────────────────────────────────────────────────────

def _score_images_against_descriptors(
    images: List[np.ndarray],
    descriptors: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    For each descriptor, compute max CLIP similarity across all image crops.
    Returns dict {label: max_score}.
    """
    import torch
    from PIL import Image

    if not images or not descriptors:
        return {}

    visopt = _get_visopt_module()
    model, preprocess, tokenizer, device = visopt._get_model()

    # Build prompts for each descriptor: two formulations averaged.
    prompt_groups: List[List[str]] = []
    for _, text in descriptors:
        prompt_groups.append([
            f"a person who is {text}",
            f"a person with {text}",
        ])

    # Encode all prompts at once.
    all_prompts = [p for grp in prompt_groups for p in grp]
    text_tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Average the two prompts per descriptor → (n_descriptors, D)
    n_desc = len(descriptors)
    text_feats = text_feats.view(n_desc, 2, -1).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Score each image crop.
    max_scores: Dict[str, float] = {label: -1.0 for label, _ in descriptors}
    for img_bgr in images:
        rgb = img_bgr[..., ::-1].copy()
        pil_img = Image.fromarray(rgb)
        img_t = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_feats.T).squeeze(0).cpu().tolist()  # (n_desc,)
        for i, (label, _) in enumerate(descriptors):
            if sims[i] > max_scores[label]:
                max_scores[label] = sims[i]

    return max_scores


# ── Main entry point ─────────────────────────────────────────────────────────

def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    if not os.path.isfile(request.video_path):
        return SkillResponse(
            skill_name=metadata.name,
            summary="[ZeroShotIdentity] Error: video file not found.",
            artifacts={"error": "missing_file"},
        )

    llm = default_llm_client()
    q = request.question
    start_time, end_time = request.normalized_window()

    # ── Step 1: sample frames ─────────────────────────────────────────────────
    frames = _sample_frames(request.video_path, _NUM_FRAMES, start_time, end_time)
    if not frames:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[ZeroShotIdentity] Could not read frames.",
            artifacts={"error": "no_frames"},
        )

    # ── Step 2: detect person crops ──────────────────────────────────────────
    crops = _detect_person_crops(frames)
    using_full_frames = False
    if not crops:
        # Fall back to full frames (visual_option_match behaviour)
        crops = frames
        using_full_frames = True

    # ── Step 3: build descriptors / options ──────────────────────────────────
    options = _extract_options(q)
    if options:
        descriptors = options
    else:
        descriptors = _extract_descriptors(q, llm)

    if not descriptors:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[ZeroShotIdentity] Could not extract descriptors from question.",
            artifacts={"error": "no_descriptors"},
        )

    # ── Step 4: CLIP scoring ──────────────────────────────────────────────────
    try:
        scores = _score_images_against_descriptors(crops, descriptors)
    except Exception as exc:
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[ZeroShotIdentity] CLIP scoring failed: {exc!r}",
            artifacts={"error": str(exc)},
        )

    if not scores:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[ZeroShotIdentity] Scoring returned no results.",
            artifacts={"error": "empty_scores"},
        )

    # ── Step 5: pick best match ───────────────────────────────────────────────
    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]
    best_text = next((t for lbl, t in descriptors if lbl == best_label), best_label)

    # Sort all for display
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    score_str = ", ".join(f"{lbl}:{s:.3f}" for lbl, s in sorted_scores)

    # If MCQ options, second-best margin is informative.
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_score
    strong = margin >= 0.05

    content = (
        f"Best match: {best_label} — \"{best_text}\" (score={best_score:.3f}, "
        f"margin={margin:.3f}{'  ✓ strong' if strong else '  ⚠ weak'}). "
        f"All scores: {score_str}."
    )

    return SkillResponse(
        skill_name=metadata.name,
        summary=(
            f"[ZeroShotIdentity] best={best_label}({best_text!r}) "
            f"score={best_score:.3f} margin={margin:.3f} "
            f"crops={len(crops)} full_frame_fallback={using_full_frames}"
        ),
        content=content,
        artifacts={
            "best_label": best_label,
            "best_text": best_text,
            "best_score": best_score,
            "margin": margin,
            "strong": strong,
            "option_scores": scores,
            "n_crops": len(crops),
            "n_frames": len(frames),
            "full_frame_fallback": using_full_frames,
        },
    )
