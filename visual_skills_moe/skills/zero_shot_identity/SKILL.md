---
name: zero_shot_identity
description: Crop-then-CLIP person identity matcher. Detects people with YOLOv8n, crops bounding boxes, and scores crops against MCQ options or LLM-extracted descriptors via OpenCLIP (ViT-H-14). Falls back to full-frame scoring when no persons are detected.
tags: identity, person, clip, yolo, detection
when_to_use: Use for "who is the person with X", "who is wearing Y", "what role does the person play" identity questions requiring matching visual attributes of named individuals across multiple candidates.
skill_type: override
---

## How it works

1. Samples 8 keyframes evenly across the video (or requested window).
2. Runs YOLOv8n person detector (conf=0.20) to extract up to 4 person crops per frame (32 crops max).
3. Scores all crops against MCQ options (A/B/C/D) or LLM-extracted visual descriptors using OpenCLIP ViT-H-14.
4. Each descriptor is evaluated with two prompt formulations ("a person who is X", "a person with X") and averaged.
5. Falls back to full-frame scoring when no persons are detected by YOLO.

## Output
- **Type**: override — returns structured option_scores for direct answer selection.
- `artifacts["option_scores"]`: dict of `{letter: max_score}` across all crops.
- `artifacts["best_label"]`: letter/descriptor of highest-scoring match.
- `artifacts["margin"]`: score gap between best and second-best (≥0.05 considered strong).
- `artifacts["strong"]`: `true` if margin ≥ 0.05.
- `artifacts["full_frame_fallback"]`: `true` if YOLO detected no persons and full frames were used instead.
