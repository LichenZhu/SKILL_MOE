---
name: temporal_ordering
description: Determines the chronological order of MCQ options in a video using CLIP frame-level similarity scoring. Finds when each option first appears (visually or textually) and sorts by timestamp.
tags: ordering, chronological, sequence, temporal, clip
when_to_use: Use for "in what order", "listed before/after", "which happened first", "chronological order", "what comes before/after X" questions where the answer requires knowing the sequence of events or items in the video.
---

## Guidance
- Extracts MCQ options A/B/C/D from the question text.
- Scores each option text against video frames using CLIP (ViT-B-32).
- Finds the timestamp of each option's first clear appearance.
- Returns a sorted chronological sequence: "A first appears at 12s, B at 28s → order: A→B".
- Requires open-clip-torch (`pip install open-clip-torch`).
