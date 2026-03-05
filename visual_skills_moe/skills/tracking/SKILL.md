---
name: tracking
description: Track objects or people across video frames to answer counting and temporal continuity questions.
tags: counting, temporal, objects, people
when_to_use: Use when the question asks how many unique objects or people appear over time, or requires following a specific entity across the video (e.g., "how many challengers", "how many times did X happen").
---

## Guidance
- Use multi-object tracking (e.g., ByteTrack, BoTrack) to assign persistent IDs across frames.
- Frame-level max count ≠ total unique entities — track IDs give the correct total.
- Report: (a) total unique track IDs observed, (b) max simultaneous count per frame.
- Prefer GPU inference; fall back to CPU if unavailable.
- When a time window is provided, restrict tracking to that segment only.

## Planned Implementation
- Backend: ByteTrack (byte_tracker) or BoTrack on top of YOLO detections.
- Input: video segment (start_time, end_time), optional target class.
- Output: `{"unique_count": N, "max_simultaneous": M, "track_summary": [...]}`.
- Key bottleneck cases: 147-1 (challengers), 169-3 (synchronized dives), 206-1 (total people).
