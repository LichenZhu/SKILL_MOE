"""Grounding skill — zero-shot object detection with GroundingDINO.

Uses IDEA-Research/grounding-dino-tiny (cached locally, ~0.7 GB GPU) to localise
specific objects mentioned in the question.  Operates on evenly-sampled frames
and returns bounding-box evidence with confidence scores.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

_MODEL_ID = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
_NUM_FRAMES = int(os.getenv("GROUNDING_NUM_FRAMES", "4"))
_BOX_THRESHOLD = float(os.getenv("GROUNDING_BOX_THRESHOLD", "0.25"))
_TEXT_THRESHOLD = float(os.getenv("GROUNDING_TEXT_THRESHOLD", "0.20"))
_MAX_NEW_TOKENS = int(os.getenv("GROUNDING_ENTITY_TOKENS", "32"))
# Giant box filter: discard detections that cover more than this fraction of the
# frame.  A "peanut butter jar" or "dremel" does not fill the entire screen; a box
# covering 60%+ of the frame is almost certainly a false positive from GroundingDINO
# hallucinating over the full scene.
_MAX_BOX_AREA_FRACTION = float(os.getenv("GROUNDING_MAX_BOX_AREA", "0.60"))

# ── Color hint detection & HSV re-ranking ──────────────────────────────────
_COLOR_HINT_RE = re.compile(
    r"\b(red|orange|yellow|green|blue|purple|violet|pink|white|black|gray|grey|brown)\b",
    re.IGNORECASE,
)

# HSV ranges per color: list of (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)
# OpenCV HSV: H in [0, 180], S in [0, 255], V in [0, 255]
_HSV_RANGES: Dict[str, List[Tuple[int, int, int, int, int, int]]] = {
    "red":    [(0, 10, 80, 255, 40, 255), (170, 180, 80, 255, 40, 255)],
    "orange": [(10, 25, 100, 255, 80, 255)],
    "yellow": [(25, 35, 100, 255, 100, 255)],
    "green":  [(35, 85, 50, 255, 40, 255)],
    "blue":   [(90, 130, 50, 255, 40, 255)],
    "purple": [(130, 160, 50, 255, 40, 255)],
    "violet": [(130, 160, 50, 255, 40, 255)],
    "pink":   [(160, 180, 30, 255, 100, 255)],
    "white":  [(0, 180, 0, 30, 200, 255)],
    "black":  [(0, 180, 0, 255, 0, 50)],
    "gray":   [(0, 180, 0, 40, 50, 200)],
    "grey":   [(0, 180, 0, 40, 50, 200)],
    "brown":  [(10, 20, 100, 255, 30, 150)],
}


def _parse_color_hint(question: str) -> Optional[str]:
    """Return the first color word in the question stem, or None."""
    stem = re.split(r"\n\s*[A-D]\.", question)[0]
    m = _COLOR_HINT_RE.search(stem)
    return m.group(1).lower() if m else None


def _crop_color_score(pil_img: Image.Image, box: List[float], color_name: str) -> float:
    """Return [0, 1] fraction of crop pixels matching the target color in HSV space."""
    ranges = _HSV_RANGES.get(color_name)
    if not ranges:
        return 0.0
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    w, h = pil_img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = np.array(pil_img.crop((x1, y1, x2, y2)), dtype=np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) in ranges:
        lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)
    # cv2.inRange outputs 255 (not 1) for matching pixels; use np.count_nonzero
    # to get the true fraction in [0, 1].
    return float(np.count_nonzero(mask)) / float(mask.size + 1e-6)


# Module-level cache — model loads once per process.
_model_cache: Dict[str, Any] = {}


def _load_model() -> tuple:
    if "model" not in _model_cache:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        proc = GroundingDinoProcessor.from_pretrained(_MODEL_ID)
        # GroundingDINO has mixed-precision internals; load fp32 and use autocast.
        model = GroundingDinoForObjectDetection.from_pretrained(_MODEL_ID)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _model_cache["processor"] = proc
        _model_cache["model"] = model
        _model_cache["device"] = device
    return _model_cache["processor"], _model_cache["model"], _model_cache["device"]


def _extract_entities(question: str) -> List[str]:
    """Extract grounding query labels from the question.

    Strategy (in priority order):
    1. For MCQ questions, parse each option text and use the concrete nouns as
       candidate labels — these are always the best grounding targets since they
       name exactly the objects the model must discriminate between.
    2. LLM-assisted extraction from the question stem (if endpoint is available).
    3. Naive noun fallback from the question stem.
    """
    # ── Strategy 1: parse MCQ options ─────────────────────────────────────
    # Options are embedded in the question as "A. ...\nB. ...\n..."
    option_blocks = re.findall(r"(?:^|\n)\s*[A-D]\.\s*(.+?)(?=\n\s*[A-D]\.|$)", question)
    if not option_blocks:
        # Also try the "Choose the correct answer from:\nA. ..." format
        after_marker = re.split(r"Choose the correct answer from:", question, flags=re.IGNORECASE)
        if len(after_marker) > 1:
            option_blocks = re.findall(r"[A-D]\.\s*(.+?)(?=[A-D]\.|$)", after_marker[1])

    if option_blocks:
        _stopwords = {"a", "an", "the", "of", "in", "at", "on", "to", "for",
                      "and", "or", "it", "be", "is", "are", "was", "were"}
        entities: List[str] = []
        for block in option_blocks:
            # Strip trailing punctuation and split into words
            words = re.findall(r"\b[a-zA-Z][a-zA-Z\- ]{1,}\b", block.strip())
            # Pick longest meaningful phrase from each option
            for w in sorted(words, key=len, reverse=True):
                w_clean = w.strip().lower()
                if w_clean not in _stopwords and len(w_clean) >= 3:
                    entities.append(w_clean)
                    break
        entities = entities[:4]  # GroundingDINO handles up to ~4 labels well
        if entities:
            return entities

    # ── Strategy 2: LLM-assisted extraction ───────────────────────────────
    client = default_llm_client()
    if client:
        stem = re.split(r"\n\s*[A-D]\.", question)[0].strip()
        prompt = (
            "Extract 1 to 3 short physical object phrases from this question that we "
            "need to visually locate in a video frame. "
            "IMPORTANT: include descriptive adjectives, especially colors "
            "(e.g. if the question asks about a 'blue item', return 'blue item' or "
            "'blue dremel', NOT just 'dremel'). "
            "Return ONLY the object phrases, lowercase, comma-separated, nothing else.\n"
            f"Question: {stem}"
        )
        try:
            raw = client.complete(prompt, max_tokens=_MAX_NEW_TOKENS)
            candidates = [re.sub(r"[^a-z0-9 \-]", "", e.strip().lower())
                          for e in raw.split(",")]
            candidates = [c for c in candidates if len(c) >= 2][:3]
            if candidates:
                return candidates
        except Exception:
            pass

    # ── Strategy 3: naive noun fallback ───────────────────────────────────
    q = re.split(r"\n\s*[A-D]\.", question)[0].lower()
    stopwords = {"what", "which", "who", "where", "when", "how", "does", "the",
                 "a", "an", "is", "are", "was", "were", "do", "did", "has",
                 "have", "in", "on", "at", "of", "to", "for", "and", "or", "that"}
    tokens = re.findall(r"\b[a-z]{3,}\b", q)
    nouns = [t for t in tokens if t not in stopwords][:3]
    return nouns if nouns else ["object"]


def _extract_frames(
    video_path: str,
    start: Optional[float],
    end: Optional[float],
    n: int,
) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    start_f = int(start * fps) if start is not None else 0
    end_f = int(end * fps) if end is not None else total - 1
    start_f = max(0, min(start_f, total - 1))
    end_f = max(start_f, min(end_f, total - 1))

    indices = np.linspace(start_f, end_f, n, dtype=int)
    frames: List[Dict] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames.append({"img": pil, "time": int(idx) / fps})
    cap.release()
    return frames


def _detect(
    processor: Any,
    model: Any,
    device: str,
    image: Image.Image,
    entities: List[str],
) -> List[Dict]:
    """Run GroundingDINO on a single frame, return filtered detections."""
    # GroundingDINO text format: "entity1 . entity2 . entity3 ."
    text_query = " . ".join(entities) + " ."

    inputs = processor(images=image, text=text_query, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), torch.autocast(device_type="cuda" if device != "cpu" else "cpu"):
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=_BOX_THRESHOLD,
        text_threshold=_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )

    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    img_w, img_h = image.size  # PIL: (width, height)
    frame_area = img_w * img_h

    detections = []
    for score, label, box in zip(
        results[0]["scores"].cpu().tolist(),
        results[0]["labels"],
        results[0]["boxes"].cpu().tolist(),
    ):
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        # Giant box filter: skip detections that blanket the entire frame.
        # GroundingDINO sometimes returns a full-scene box as a false positive;
        # such boxes make the HSV color crop meaningless and also mislead the VLM.
        if frame_area > 0 and box_area / frame_area > _MAX_BOX_AREA_FRACTION:
            continue
        detections.append({
            "label": str(label).strip(),
            "score": round(score, 3),
            "box": [round(c, 1) for c in box],
        })
    return detections


def _annotate_and_encode(pil_img: Image.Image, detections: List[Dict], max_boxes: int = 3) -> Optional[str]:
    """Draw red bounding boxes on top detections and return base64 JPEG."""
    if not detections:
        return None
    top = sorted(detections, key=lambda d: d.get("composite_score", d.get("score", 0)), reverse=True)[:max_boxes]
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for d in top:
        x1, y1, x2, y2 = [int(c) for c in d["box"]]
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{d['label']} {d.get('composite_score', d['score']):.2f}"
        cv2.putText(arr, label, (x1, max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode()


def run(request: SkillRequest, meta: SkillMetadata) -> SkillResponse:
    try:
        processor, model, device = _load_model()
    except Exception as exc:
        return SkillResponse(
            skill_name=meta.name,
            summary=f"[grounding] Model load error: {exc}",
            artifacts={"error": str(exc)},
        )

    # ── Step 1: extract entity names and color hint from question ──────
    entities = _extract_entities(request.question)
    color_hint = _parse_color_hint(request.question)

    # ── Step 2: sample frames ──────────────────────────────────────────
    start, end = request.normalized_window()
    frames = _extract_frames(request.video_path, start, end, _NUM_FRAMES)
    if not frames:
        return SkillResponse(
            skill_name=meta.name,
            summary="[grounding] No frames extracted.",
            artifacts={"error": "no_frames"},
        )

    # Build time→image lookup for color re-ranking step.
    time_to_img = {f["time"]: f["img"] for f in frames}

    # ── Step 3: detect on each frame ───────────────────────────────────
    all_detections: List[Dict] = []
    frame_lines: List[str] = []

    for frame_info in frames:
        dets = _detect(processor, model, device, frame_info["img"], entities)
        t = frame_info["time"]
        if dets:
            det_strs = [f"'{d['label']}' (score: {d['score']:.2f})" for d in dets[:3]]
            frame_lines.append(f"At {t:.1f}s: found {', '.join(det_strs)}")
            for d in dets:
                all_detections.append({"time": t, **d})
        else:
            frame_lines.append(f"At {t:.1f}s: no detections above threshold")

    # ── Step 3.5: HSV crop re-ranking when a color hint is present ─────
    if color_hint and all_detections:
        for d in all_detections:
            img = time_to_img.get(d["time"])
            if img is not None:
                cs = _crop_color_score(img, d["box"], color_hint)
                d["color_score"] = round(cs, 3)
                d["composite_score"] = round(d["score"] + 0.5 * cs, 3)
            else:
                d["color_score"] = 0.0
                d["composite_score"] = d["score"]

    # ── Step 4: format evidence ────────────────────────────────────────
    score_key = "composite_score" if color_hint else "score"
    if not all_detections:
        summary = (
            f"[grounding] Searched for: {', '.join(entities)}. "
            "No objects detected above confidence threshold in sampled frames."
        )
    else:
        best = max(all_detections, key=lambda d: d.get(score_key, d["score"]))
        color_note = f" (color-reranked by '{color_hint}')" if color_hint else ""
        extra = ""
        if color_hint:
            extra = (
                f", color_score: {best.get('color_score', 0):.2f}"
                f", composite: {best.get('composite_score', best['score']):.2f}"
            )
        summary = (
            f"[grounding] Searched for: {', '.join(entities)}{color_note}. "
            + " | ".join(frame_lines)
            + f". Best match: '{best['label']}' at {best['time']:.1f}s "
            f"(score: {best['score']:.2f}{extra})."
        )

    # ── Step 5: build visual evidence — red-box annotated frames ──────
    # Group detections by frame timestamp, pick the top-2 frames by best
    # composite/detection score, annotate them with red bounding boxes,
    # and store as base64 JPEG so the pipeline can inject them into the VLM.
    visual_evidence: List[str] = []
    if all_detections:
        frame_det_map: Dict[float, List[Dict]] = {}
        for d in all_detections:
            frame_det_map.setdefault(d["time"], []).append(d)
        frame_scores = [
            (t, max(d.get("composite_score", d.get("score", 0)) for d in dets))
            for t, dets in frame_det_map.items()
        ]
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        for t, _ in frame_scores[:2]:
            img = time_to_img.get(t)
            if img is None:
                continue
            b64 = _annotate_and_encode(img, frame_det_map[t])
            if b64:
                visual_evidence.append(b64)

    gc.collect()

    return SkillResponse(
        skill_name=meta.name,
        summary=summary,
        artifacts={
            "entities_queried": entities,
            "color_hint": color_hint,
            "detections": all_detections,
            "frame_count": len(frames),
            "target": ", ".join(entities),
            **({"visual_evidence": visual_evidence} if visual_evidence else {}),
        },
    )
