---
name: temporal_action_counter
description: Counts distinct occurrences of a specific action/event using OpenCLIP (ViT-H-14) per-frame binary classification and NO→YES transition counting.
tags: counting, action, event, temporal, clip
when_to_use: Use for "how many times does X happen", "how many tricks are performed", "how many shots does the player take" — event-frequency questions where ByteTrack entity tracking is inappropriate.
skill_type: override
---

## How it works

1. Extracts the action phrase from the question via LLM client (regex fallback).
2. Samples 16 frames uniformly from the video segment.
3. Scores each frame with OpenCLIP (ViT-H-14) against positive/negative action prompts.
4. Labels each frame active/inactive; counts NO→YES transitions as distinct event instances.

Appropriate for sequential/repeated events, not persistent-entity counting (use `tracking` for that).

## Output
- **Type**: override — returns a deterministic event count that can directly answer counting questions.
- `artifacts["event_count"]`: integer count of distinct action occurrences.
- `artifacts["action"]`: extracted action phrase used for classification.
