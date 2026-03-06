---
name: event_graph_rag
description: Hierarchical Event-Graph RAG for long videos. Detects scene cuts via histogram difference, encodes each scene keyframe with CLIP (ViT-H-14), retrieves the top-3 most question-relevant scenes by cosine similarity, and returns a 3-panel storyboard grid as visual evidence.
tags: temporal, retrieval, long-video, scene-index, clip
when_to_use: Long videos (>3 min) where the question spans multiple scenes or requires identifying which part of the video contains the relevant event. Suitable for temporal retrieval, cross-scene causal reasoning, and "what is this video about" questions.
skill_type: support
---

## How it works

1. Detects scene boundaries via histogram difference between consecutive frames.
2. Encodes one keyframe per scene with CLIP (ViT-H-14).
3. Retrieves the top-3 scenes most similar to the question embedding by cosine similarity.
4. Composes a horizontal 3-panel storyboard stored as `visual_evidence` for direct VLM injection.

## Output
- **Type**: support — provides a scene-retrieval storyboard as visual evidence; does not output option_scores.
- `artifacts["visual_evidence"]`: list containing one base64 JPEG 3-panel storyboard.
- `artifacts["retrieved_scenes"]`: list of `{scene_idx, timestamp, similarity}` dicts.
