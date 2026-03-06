---
name: grounding
description: Zero-shot object detection via GroundingDINO-tiny. Localises specific physical objects, clothing items, or tools in video keyframes and returns bounding boxes with confidence scores.
tags: object, detection, grounding, spatial, fine-grained, visual
when_to_use: Use when the question asks to identify a specific object, tool, or item by visual appearance (e.g. "what is the blue item", "which tool is used"), or requires fine-grained spatial localisation of named entities.
skill_type: support
---

## How it works

Extracts 1–3 entity names from the question via LLM client (regex fallback),
samples 4 evenly-spaced frames in the requested time window, and runs
GroundingDINO-tiny (IDEA-Research/grounding-dino-tiny, ~0.7 GB GPU) on each
frame. Reports detections above threshold as bounding boxes with confidence scores.
Model cache is shared with `focus_vqa` to avoid double-loading.

## Output
- **Type**: support — provides detection evidence as text; does not output option_scores.
- `artifacts["detections"]`: list of `{entity, frame_idx, timestamp, box, score}` dicts.
- Returns `no_match=True` if no entities are detected above threshold.
