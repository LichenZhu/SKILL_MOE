---
name: visual_option_match
description: Match multiple-choice options against video keyframes via CLIP-style image-text similarity.
tags: appearance, clothing, color, attribute, multiple-choice
when_to_use: Use when options describe visual attributes (e.g., clothing color/style) and a direct A/B/C/D visual discrimination is needed.
---

## Guidance
- Parses options A/B/C/D directly from the question text.
- Samples keyframes within the requested time window.
- Uses OpenCLIP image-text similarity to score each option.
- Returns option-level confidence scores and best option.
