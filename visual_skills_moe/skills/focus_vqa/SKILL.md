---
name: focus_vqa
description: Crop-then-Answer VQA — locates the target object in video frames via GroundingDINO, crops the high-resolution region, and sends the crop to gpt-4o-mini vision for fine-grained visual details (text, logos, labels, small objects).
tags: fine-grained, crop, vqa, ocr, visual, text, logo, holding, sign
when_to_use: Use when the question asks about fine-grained visual details that a globally-downsampled video frame cannot resolve — text written on an object, logo or label on a product, what a person is holding, or the color of a small specific item.
---

## Guidance
- Step 1: Extract the target object/region phrase from the question via LLM (fallback: regex).
- Step 2: Sample 5 evenly-spaced keyframes from the video at full resolution.
- Step 3: Run GroundingDINO-tiny to localise the target in each frame.
- Step 4: Crop the highest-confidence bounding box with 20% padding for context.
- Step 5: Send up to 4 high-resolution crops + original question to gpt-4o-mini vision.
- If target is not found or the vision LLM call fails, returns no_match=True so the VLM answers from full video without confusion.
- Reuses GroundingDINO model cache with the grounding skill to avoid double-loading.
