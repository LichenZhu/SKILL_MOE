"""
Temporal Ordering skill — chronological sequence of MCQ options.

Problem: Questions like "What is listed BEFORE Semaphore?", "In what order
did these movements occur?", "Which happened first?" require knowing when
each option appears in the video timeline.

Solution: Score each MCQ option text against sampled frames via CLIP,
find the first timestamp each option clearly appears, sort chronologically.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse

_SAMPLE_FPS = 1.0         # 1 fps — enough for ordering
_MAX_FRAMES = 180          # cap at 3 minutes of scanning
_CLIP_THRESHOLD = 0.22     # min similarity to count as "clearly present"
_CLIP_FALLBACK_THR = 0.15  # use best-frame fallback if nothing exceeds threshold

_model_cache: dict = {}


# ---------------------------------------------------------------------------
# CLIP model
# ---------------------------------------------------------------------------

def _get_clip():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["meta"]
    try:
        import open_clip
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _model_cache["model"] = model
        _model_cache["meta"] = (preprocess, tokenizer, device)
        return model, (preprocess, tokenizer, device)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# MCQ option extraction
# ---------------------------------------------------------------------------

def _extract_options(question: str) -> Dict[str, str]:
    """Parse MCQ options A/B/C/D from question text."""
    opts: Dict[str, str] = {}
    # Match "A. text", "A) text", "A: text"
    for m in re.finditer(
        r"(?m)^\s*([A-D])[.):\s]\s*(.+?)(?=\n\s*[A-D][.):\s]|\Z)",
        question,
        re.DOTALL,
    ):
        letter = m.group(1)
        text = m.group(2).strip().replace("\n", " ")
        # Trim option to meaningful noun phrase (max 60 chars)
        opts[letter] = text[:60]
    return opts


def _extract_symbol_map(question: str) -> Dict[str, str]:
    """Extract ①→description mapping from question stem, e.g. '① Study ② Making bed'."""
    symbol_map: Dict[str, str] = {}
    for m in re.finditer(
        r"([①②③④⑤⑥⑦⑧⑨⑩])\s*([^①②③④⑤⑥⑦⑧⑨⑩\n]{2,50}?)(?=[①②③④⑤⑥⑦⑧⑨⑩]|\n|$)",
        question,
    ):
        sym = m.group(1)
        desc = m.group(2).strip().rstrip(".,;")
        if desc:
            symbol_map[sym] = desc
    return symbol_map


def _gpt_resolve_symbols(question: str) -> Dict[str, str]:
    """Use GPT-4o-mini to extract circled-number→description mapping from the question."""
    import os as _os
    import json as _json
    api_key = _os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}
    base_url = _os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        prompt = (
            "The following question uses circled numbers (①②③④...) to label events.\n"
            "Extract the mapping from each circled number to its event description.\n"
            "Return JSON only: {\"①\": \"event name\", \"②\": \"event name\", ...}\n\n"
            f"Question:\n{question[:600]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            timeout=15,
        )
        raw = resp.choices[0].message.content.strip()
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            data = _json.loads(raw[i:j + 1])
            return {k: str(v).strip() for k, v in data.items() if str(v).strip()}
    except Exception:
        pass
    return {}


def _resolve_symbol_options(
    options: Dict[str, str],
    question: str,
) -> Optional[Dict[str, str]]:
    """
    If options are symbol-ordering sequences (e.g. "②①③④"), resolve them into
    individual event descriptions that CLIP can match against video frames.

    Returns:
      - Original options if no circled-number sequences detected.
      - Dict of individual events {A/B/C/D: description} if resolved.
      - None if sequences detected but resolution failed → caller returns no_match.
    """
    has_sequence = any(
        bool(re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]{2,}", opt))
        for opt in options.values()
    )
    if not has_sequence:
        return options  # No symbols → use options as-is

    # Try regex extraction first, then GPT fallback
    symbol_map = _extract_symbol_map(question)
    if not symbol_map:
        symbol_map = _gpt_resolve_symbols(question)

    if not symbol_map:
        return None  # Cannot resolve → no_match

    # Build deduplicated ordered set of individual events
    ordered_syms = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    letters = list("ABCDEFGHIJ")
    resolved: Dict[str, str] = {}
    for i, sym in enumerate(ordered_syms):
        if sym in symbol_map and i < len(letters):
            resolved[letters[i]] = symbol_map[sym][:60]

    return resolved if len(resolved) >= 2 else None


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def _sample_frames(
    video_path: str,
    start_t: float,
    end_t: float,
    fps: float,
    max_frames: int,
) -> List[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    result: List[Tuple[float, np.ndarray]] = []
    t = start_t
    while t <= end_t and len(result) < max_frames:
        idx = max(0, min(int(t * vid_fps), total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            result.append((t, frame))
        t += 1.0 / fps
    cap.release()
    return result


# ---------------------------------------------------------------------------
# CLIP scoring
# ---------------------------------------------------------------------------

def _score_options(
    frames: List[Tuple[float, np.ndarray]],
    options: Dict[str, str],
) -> Dict[str, List[Tuple[float, float]]]:
    """Return {letter: [(timestamp, cosine_similarity), ...]} for all frames."""
    model, meta = _get_clip()
    if model is None or not options or not frames:
        return {}

    preprocess, tokenizer, device = meta

    import torch
    from PIL import Image

    letters = sorted(options.keys())
    texts = [options[k] for k in letters]

    try:
        tokens = tokenizer(texts).to(device)
        with torch.no_grad():
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    except Exception:
        return {}

    scores: Dict[str, List[Tuple[float, float]]] = {k: [] for k in letters}

    # Process in batches of 8 frames
    batch_size = 8
    for b in range(0, len(frames), batch_size):
        batch = frames[b: b + batch_size]
        imgs, tss = [], []
        for ts, frame_bgr in batch:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            imgs.append(preprocess(pil))
            tss.append(ts)
        try:
            img_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                img_feats = model.encode_image(img_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # sims: (n_frames, n_options)
            sims = (img_feats @ text_feats.T).cpu().float().numpy()
            for i, ts in enumerate(tss):
                for j, letter in enumerate(letters):
                    scores[letter].append((ts, float(sims[i, j])))
        except Exception:
            continue

    return scores


# ---------------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------------

def _first_occurrence(
    scores: Dict[str, List[Tuple[float, float]]],
    threshold: float,
    fallback_thr: float,
) -> Dict[str, Optional[float]]:
    """For each option, find its first frame above threshold."""
    result: Dict[str, Optional[float]] = {}
    for letter, ts_scores in scores.items():
        if not ts_scores:
            result[letter] = None
            continue
        above = [(ts, s) for ts, s in ts_scores if s >= threshold]
        if above:
            result[letter] = above[0][0]
        else:
            # fallback: use the peak-scoring frame if it clears the lower bar
            best_ts, best_s = max(ts_scores, key=lambda x: x[1])
            result[letter] = best_ts if best_s >= fallback_thr else None
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    video_path = request.video_path
    question = request.question
    duration = request.video_duration or 0.0
    start_t = request.start_time or 0.0
    end_t = request.end_time or duration

    if not os.path.isfile(video_path):
        return SkillResponse(
            skill_name="temporal_ordering",
            summary="Video not found.",
            artifacts={"error": "no_video"},
        )

    options = _extract_options(question)
    if len(options) < 2:
        return SkillResponse(
            skill_name="temporal_ordering",
            summary="Could not extract ≥2 MCQ options from question.",
            artifacts={"no_match": True},
        )

    # Resolve circled-number symbol sequences (e.g. "②①③④") to actual event
    # descriptions before CLIP scoring. If options ARE symbol orderings but
    # resolution fails, return no_match — CLIP cannot score abstract symbols.
    resolved_options = _resolve_symbol_options(options, question)
    if resolved_options is None:
        return SkillResponse(
            skill_name="temporal_ordering",
            summary="Options contain symbol-ordering sequences that could not be resolved to event descriptions.",
            artifacts={"no_match": True},
        )
    options = resolved_options

    frames = _sample_frames(video_path, start_t, end_t, _SAMPLE_FPS, _MAX_FRAMES)
    if not frames:
        return SkillResponse(
            skill_name="temporal_ordering",
            summary="Could not sample frames.",
            artifacts={"error": "frame_sample_failed"},
        )

    scores = _score_options(frames, options)
    if not scores:
        return SkillResponse(
            skill_name="temporal_ordering",
            summary="CLIP model unavailable (install open-clip-torch).",
            artifacts={"no_match": True},
        )

    first_occ = _first_occurrence(scores, _CLIP_THRESHOLD, _CLIP_FALLBACK_THR)

    # Build sorted output
    with_ts = [(l, t) for l, t in first_occ.items() if t is not None]
    no_ts = [l for l, t in first_occ.items() if t is None]
    with_ts.sort(key=lambda x: x[1])

    order_letters = [l for l, _ in with_ts] + no_ts
    order_str = " → ".join(order_letters)

    lines = []
    for letter, ts in with_ts:
        text = options.get(letter, "")[:45]
        lines.append(f"  {letter} ('{text}'): first appears at t={ts:.1f}s")
    for letter in no_ts:
        text = options.get(letter, "")[:45]
        lines.append(f"  {letter} ('{text}'): not clearly detected")

    summary = f"Chronological order: {order_str}\n" + "\n".join(lines)

    return SkillResponse(
        skill_name="temporal_ordering",
        summary=summary,
        artifacts={
            "chronological_order": order_letters,
            "first_occurrences": {k: v for k, v in first_occ.items()},
        },
    )
