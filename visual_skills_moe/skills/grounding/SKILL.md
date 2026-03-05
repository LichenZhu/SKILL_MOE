---
name: grounding
description: Uses GroundingDINO zero-shot object detection to precisely locate and confirm the presence of specific physical objects, clothing items, or tools in video frames. Returns bounding boxes and confidence scores.
tags: object, detection, grounding, spatial, fine-grained, visual
when_to_use: Use when the question asks to identify a specific object, tool, or item by visual appearance (e.g., "what is the blue item", "which tool is used"), or requires fine-grained visual discrimination between similar-looking objects.
---

## Guidance
- Extracts 1-3 entity names from the question using the LLM client, then queries GroundingDINO.
- Samples 4 evenly-spaced frames in the requested time window.
- Reports bounding boxes and confidence scores for each detection above threshold.
- Model: IDEA-Research/grounding-dino-tiny (~0.7 GB GPU, cached locally).
- Uses float32 with autocast for stable inference.
