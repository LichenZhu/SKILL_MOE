---
name: tracking
description: Multi-object tracking via YOLOv8 + ByteTrack to count unique entities and max simultaneous presence across video frames. Returns reliable_count that accounts for scene cuts inflating unique_count.
tags: counting, temporal, objects, people, bytetrack, yolo
when_to_use: Use when the question asks how many unique objects or people appear over time (e.g. "how many challengers", "how many people appear in total"). For event-frequency counting ("how many times does X happen") use temporal_action_counter instead.
skill_type: override
---

## How it works

1. Samples video at 2 FPS (configurable via `TRACKING_SAMPLE_FPS`, up to 600 frames).
2. Extracts the target entity class from the question via LLM client; maps to COCO-80 class ID.
3. Runs YOLOv8n + ByteTrack to assign persistent IDs across frames.
4. Detects scene cuts; when cuts are present, uses `max_simultaneous` as `reliable_count` to avoid ID inflation. Otherwise uses `unique_count`.

## Output
- **Type**: override — returns a deterministic entity count that can directly answer counting questions.
- `artifacts["reliable_count"]`: recommended answer (scene-cut-aware).
- `artifacts["unique_count"]`: total distinct track IDs observed (may be inflated across cuts).
- `artifacts["max_simultaneous"]`: peak entity count in any single frame.
- `artifacts["target"]`: entity class searched for (e.g. "person", "car").
