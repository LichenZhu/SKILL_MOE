---
name: focus_vqa
description: Spotlight attention skill — locates the target object in video frames via GroundingDINO-tiny, encodes spotlight images (darkened frame + red-boxed target region), and injects them directly into Qwen2.5-Omni alongside the original video for fine-grained visual inspection. No external API calls; no text intermediary.
tags: fine-grained, crop, vqa, visual, text, logo, holding, sign, spotlight, grounding
when_to_use: Use when the question asks about fine-grained visual details that a globally-downsampled video frame cannot resolve — text written on an object, logo or label on a product, what a person is holding, what accessory or clothing item is visible, or the color/pattern of a small specific item.
skill_type: support
---

## How it works
1. **Target extraction**: LLM extracts a 1–6 word object/region phrase from the question (regex fallback if LLM unavailable).
2. **Frame sampling**: Samples 8 evenly-spaced keyframes from the video (or the requested temporal window). Frames are pre-scaled to ≤1024px long edge before detection.
3. **GroundingDINO localisation**: Runs GroundingDINO-tiny (box threshold 0.22) to detect the target in each frame. Shares model cache with the `grounding` skill to avoid double-loading.
4. **Spotlight encoding**: For the top-3 highest-confidence detections, produces a spotlight image: the full frame is darkened to 50% brightness, the target bounding-box region is restored to 100% brightness, and a red rectangle is drawn around it. This preserves global context while highlighting the region of interest.
5. **Visual injection**: The spotlight images are stored as base64 JPEG strings in `artifacts["visual_evidence"]`. The pipeline (`_answer_with_visual_crops`) injects them directly into Qwen2.5-Omni alongside the video — no text conversion, no information loss.
6. **No external API**: Does not call GPT or any external vision API. Inference stays fully local.

## Output
- **Type**: support — provides visual crops, does not output `option_scores` or a direct answer.
- `artifacts["visual_evidence"]`: list of base64 JPEG spotlight images (up to 3).
- `artifacts["target"]`: extracted object phrase.
- `artifacts["crops_found"]`: number of frames where target was detected.
- Returns `no_match=True` if target is not detected in any frame, so the pipeline falls back to baseline without confusion.
