---
name: ocr
description: Detect and recognize text (e.g., license plates, signage, subtitles) from frames in the video.
tags: text, numbers, plate, signage
when_to_use: Use when the question asks to read, detect, or extract text/number sequences from the video.
---

## Guidance
- Uses EasyOCR with English + Chinese support.
- Samples video at 1 FPS by default (configurable via OCR_SAMPLE_FPS env var).
- Deduplicates text across frames to avoid repetitive output.
- Runs on CPU to avoid CUDA conflicts with Video LLM.
- Returns text segments with start/end timestamps.
