---
name: temporal_segment
description: Dense frame sampling from a specific temporal window of the video, with vision-LLM description. Solves questions where VLM's 64-frame global sampling leaves the target segment with only 1-2 frames.
tags: temporal, segment, dense-sampling, vision, description
when_to_use: Use when the question targets a specific temporal portion of the video — "second half", "last scene", "latter part", "opening scene", "first half". Returns a text description of what happens in that window.
---

## Guidance
- Parses temporal reference from the question to identify a time fraction.
- Densely samples 24 frames from that window only.
- Sends a timestamped grid image to gpt-4o-mini vision for description.
- Returns a concise description of the visual content in the segment.
- Fails gracefully (no_match) when vision API is unavailable.
