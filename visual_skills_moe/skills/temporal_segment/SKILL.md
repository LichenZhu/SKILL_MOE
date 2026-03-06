---
name: temporal_segment
description: Dense frame sampling from a specific temporal window of the video, with vision-LLM description. Solves questions where the VLM's global 64-frame sampling leaves the target segment with only 1-2 frames. Supports both visual description mode and narrative chronicle mode for causal questions.
tags: temporal, segment, dense-sampling, vision, description
when_to_use: Use when the question targets a specific temporal portion of the video — "second half", "last scene", "latter part", "opening scene", "first half", or causal/narrative questions about why something happened.
skill_type: support
---

## How it works

1. Parses the temporal reference in the question to identify a fractional window (e.g. "last scene" → 75%–100%).
2. Densely samples 24 frames from that window only.
3. Builds a timestamped 5×5 grid image at 480×360 px per thumbnail (JPEG quality 85).
4. Sends the grid to the configured vision LLM (via `OPENAI_MODEL` / `OPENAI_BASE_URL`) in one of two modes:
   - **Visual mode** (what/who/which): concise 2–4 sentence description of visible content.
   - **Chronicle mode** (why/reason/cause/how did): structured narrative arc with 3–5 semantic beats.
5. Fails gracefully with `no_match=True` when the vision API is unavailable.

## Output
- **Type**: support — provides a text description of the temporal segment; does not output option_scores.
- `artifacts["description"]`: vision-LLM description of the segment.
- `artifacts["window"]`: human-readable label of the sampled time range (e.g. "t=45s–60s (75%–100%)").
- `artifacts["mode"]`: `"visual"` or `"chronicle"`.
