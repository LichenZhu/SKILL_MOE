---
name: temporal_action_counter
description: Counts distinct occurrences of a specific action/event using CLIP per-frame binary classification and NO→YES transition counting.
tags: counting, action, event, temporal, clip
when_to_use: Use for "how many times does X happen", "how many tricks are performed", "how many shots does the player take" — event-frequency questions where ByteTrack entity tracking is inappropriate.
---

## Guidance
- Extracts the action phrase from the question via gpt-4o-mini.
- Samples 16 frames uniformly from the video segment.
- Scores each frame with OpenCLIP against positive/negative action prompts.
- Counts NO→YES transitions as distinct event instances.
- Appropriate for sequential/repeated events, not persistent-entity counting.
