---
name: temporal_ordering
description: Determines the chronological order of MCQ options in a video using OpenCLIP (ViT-H-14) frame-level similarity scoring at 2 FPS. Finds the first timestamp each option clearly appears and sorts them chronologically.
tags: ordering, chronological, sequence, temporal, clip
when_to_use: Use for "in what order", "listed before/after", "which happened first", "chronological order", "what comes before/after X" questions where the answer requires knowing the sequence of visually-identifiable events in the video.
skill_type: support
risky: true
---

## How it works

1. Extracts MCQ options A/B/C/D from the question text. Resolves circled-number symbol sequences (①②③④) to event descriptions via regex or LLM fallback.
2. Samples frames at 2 FPS (up to 300 frames) using the shared OpenCLIP ViT-H-14 model (reused from `visual_option_match` cache).
3. Computes cosine similarity between each option text and each frame embedding.
4. Finds each option's first appearance above threshold (0.28); falls back to peak-frame if above 0.22.
5. Returns a sorted chronological sequence: "A first appears at 12s, B at 28s → order: A→B".

**Note**: Gated by MetaRouter as a risky skill. Not invoked for questions with abstract symbol options that cannot be matched visually.

## Output
- **Type**: support — provides chronological ordering as text evidence; accuracy varies by question type.
- `artifacts["chronological_order"]`: list of option letters sorted by first appearance.
- `artifacts["first_occurrences"]`: dict of `{letter: timestamp_seconds}`.
