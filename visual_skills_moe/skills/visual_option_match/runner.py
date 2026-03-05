"""
Visual option matcher for multiple-choice QA.

Given options A/B/C/D in the question text, score each option against
video keyframes using OpenCLIP image-text similarity, then return the
best option with confidence diagnostics.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

_CLIP_MODEL = os.getenv("VISOPT_CLIP_MODEL", "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
_CLIP_PRETRAINED = os.getenv("VISOPT_CLIP_PRETRAINED", "")
_SAMPLE_FPS = float(os.getenv("VISOPT_SAMPLE_FPS", "1.0"))   # 1 frame/sec over full video
_MAX_FRAMES = int(os.getenv("VISOPT_MAX_FRAMES", "120"))     # cap at 2 minutes
_MIN_CONF_FOR_STRONG = float(os.getenv("VISOPT_MIN_CONF_STRONG", "0.45"))

_CACHE: Dict[str, Any] = {}


def _extract_options(question: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", question or ""):
        out.append((m.group(1).upper(), m.group(2).strip()))
    return out


def _is_clothing_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        k in q
        for k in ("wear", "clothing", "dress", "shirt", "suit", "color", "colour", "outfit", "appearance")
    )


def _build_prompts(question: str, option_text: str) -> List[str]:
    prompts = [f"a video frame showing {option_text}"]
    if _is_clothing_question(question):
        prompts.append(f"a person wearing {option_text}")
    return prompts


def _get_model():
    key = f"{_CLIP_MODEL}|{_CLIP_PRETRAINED}"
    if key in _CACHE:
        return _CACHE[key]

    import torch
    import open_clip  # type: ignore

    if _CLIP_PRETRAINED:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL, pretrained=_CLIP_PRETRAINED,
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(_CLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _CACHE[key] = (model, preprocess, tokenizer, device)
    return _CACHE[key]


def _extract_frames_1fps(
    video_path: str,
    start_time: float | None,
    end_time: float | None,
    sample_fps: float = 1.0,
    max_frames: int = 120,
) -> Tuple[List[np.ndarray], float]:
    """Sample video at sample_fps across [start_time, end_time], capped at max_frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / vid_fps if vid_fps else 0.0
    if total <= 0:
        cap.release()
        return [], duration

    t_start = start_time if start_time is not None else 0.0
    t_end = end_time if end_time is not None else duration

    frames: List[np.ndarray] = []
    t = t_start
    step = 1.0 / sample_fps
    while t <= t_end and len(frames) < max_frames:
        idx = max(0, min(int(t * vid_fps), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        t += step
    cap.release()
    return frames, duration


def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    video_path = request.video_path
    if not os.path.isfile(video_path):
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[VisualOption] File not found: {video_path}",
            artifacts={"error": "missing_file"},
        )

    options = _extract_options(request.question)
    if len(options) < 2:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[VisualOption] No valid multiple-choice options found in question.",
            artifacts={"error": "missing_options"},
        )

    start_time, end_time = request.normalized_window()
    frames, duration = _extract_frames_1fps(video_path, start_time, end_time, _SAMPLE_FPS, _MAX_FRAMES)
    if not frames:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[VisualOption] Cannot extract frames from video.",
            artifacts={"error": "no_frames"},
        )

    try:
        import torch
        from PIL import Image
    except Exception as exc:
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[VisualOption] Missing runtime dependencies: {exc}",
            artifacts={"error": f"deps:{exc}"},
        )

    try:
        model, preprocess, tokenizer, device = _get_model()
    except Exception as exc:
        return SkillResponse(
            skill_name=metadata.name,
            summary=f"[VisualOption] Failed to initialize OpenCLIP: {exc}",
            artifacts={"error": f"clip_init:{exc}"},
        )

    # Build option text features (average multi-prompt embeddings per option).
    option_letters = [letter for letter, _ in options]
    option_labels = [text for _, text in options]
    text_features = []
    with torch.no_grad():
        for label in option_labels:
            prompts = _build_prompts(request.question, label)
            text_tokens = tokenizer(prompts).to(device)
            feats = model.encode_text(text_tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean_feat = feats.mean(dim=0, keepdim=True)
            mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)
            text_features.append(mean_feat)
    text_features = torch.cat(text_features, dim=0)  # [N, D]

    # Score all frames; take MAX similarity per option across all frames.
    # Max-pooling finds the best moment each option is visible, outperforming
    # mean-pooling when the relevant content appears in only a subset of frames.
    logits_all = []
    batch_size = 8
    with torch.no_grad():
        for b in range(0, len(frames), batch_size):
            batch_frames = frames[b: b + batch_size]
            imgs = []
            for frame in batch_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                imgs.append(preprocess(image))
            img_tensor = torch.stack(imgs).to(device)
            img_feats = model.encode_image(img_tensor)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = img_feats @ text_features.T   # [B, N_options]
            logits_all.append(logits)
    all_logits = torch.cat(logits_all, dim=0)          # [T, N_options]
    max_logits = all_logits.max(dim=0).values          # [N_options] — best frame per option
    probs = torch.softmax(max_logits, dim=-1).detach().cpu().numpy().tolist()

    ranked = sorted(
        [(option_letters[i], option_labels[i], float(probs[i])) for i in range(len(option_letters))],
        key=lambda x: x[2],
        reverse=True,
    )
    best_letter, best_label, best_prob = ranked[0]
    second_prob = ranked[1][2] if len(ranked) > 1 else 0.0
    margin = float(best_prob - second_prob)

    option_scores = {letter: round(float(prob), 4) for letter, _, prob in ranked}
    ranking_text = ", ".join(f"{letter}:{prob:.3f}" for letter, _, prob in ranked)
    window_desc = (
        f"{start_time:.1f}-{end_time:.1f}s"
        if start_time is not None and end_time is not None
        else "full video"
    )
    top2 = [(letter, label, prob) for letter, label, prob in ranked[:2]]
    top2_str = "; ".join(f"{l}='{lbl}'({p:.3f})" for l, lbl, p in top2)
    summary = (
        f"[VisualOption] Top-2 options: {top2_str} | margin={margin:.3f} | "
        f"frames={len(frames)} ({window_desc})"
    )

    return SkillResponse(
        skill_name=metadata.name,
        summary=summary,
        content=f"best_option={best_letter}; confidence={best_prob:.3f}; margin={margin:.3f}; top2={top2_str}",
        artifacts={
            "best_option": best_letter,
            "best_label": best_label,
            "confidence": round(best_prob, 4),
            "margin": round(margin, 4),
            "strong": bool(best_prob >= _MIN_CONF_FOR_STRONG),
            "top2": [{"letter": l, "text": lbl, "score": round(p, 4)} for l, lbl, p in top2],
            "option_scores": option_scores,
            "options": [{"letter": letter, "text": label} for letter, label in options],
            "duration_sec": duration,
            "window_start": start_time,
            "window_end": end_time,
            "num_frames": len(frames),
        },
    )
