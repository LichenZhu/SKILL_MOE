---
name: visual_option_match
description: Scores MCQ options A/B/C/D against video keyframes using OpenCLIP (ViT-H-14, LAION-2B) image-text similarity at 1 FPS. Returns per-option confidence scores and the best-matching option.
tags: appearance, clothing, color, attribute, multiple-choice, clip
when_to_use: Use when options describe visual attributes (clothing color/style, object appearance, outfit, pattern) and a direct A/B/C/D visual discrimination against video frames is needed.
skill_type: override
---

## How it works

1. Parses options A/B/C/D from the question text.
2. Samples keyframes at 1 FPS within the requested time window (up to 120 frames).
3. Scores each option against all frames using OpenCLIP ViT-H-14 (hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K).
4. For clothing questions, also evaluates "a person wearing {option}" prompts.
5. Returns the best-scoring option and per-option confidence.

## Output
- **Type**: override — returns structured option_scores and best_option for direct answer selection.
- `artifacts["option_scores"]`: dict of `{letter: score}` for all options.
- `artifacts["best_option"]`: letter of the highest-scoring option.
- `artifacts["margin"]`: score gap between best and second-best option.
- `artifacts["strong"]`: `true` if margin ≥ 0.45 (high-confidence answer).
