---
name: zero_shot_identity
description: Crop-then-CLIP person identity matcher. Detects people with YOLOv8, crops them, scores crops against MCQ options or LLM-extracted descriptors via OpenCLIP.
tags: identity, person, clip, yolo, detection
when_to_use: Use for "who is the person with X", "who is wearing Y", "what role does the person play" identity questions requiring matching visual attributes of individuals.
---

## Guidance
- Samples 4 keyframes, detects persons with YOLOv8n (conf=0.30).
- Crops up to 3 persons per frame (12 crops max).
- Scores crops against MCQ options (A/B/C/D) or LLM-extracted descriptors using OpenCLIP.
- Falls back to full-frame scoring when no persons are detected.
- Returns best-matching label + confidence + margin.
