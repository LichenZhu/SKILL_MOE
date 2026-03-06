---
name: ocr
description: Detect and recognize on-screen text from video frames using EasyOCR (English + Chinese). Returns deduplicated timestamped text lines for questions about signage, scores, subtitles, license plates, or any visible text.
tags: text, numbers, plate, signage, easyocr, timestamp
when_to_use: Use when the question asks to read, detect, or extract text or number sequences from the video — scores, timers, labels, signs, captions, or any on-screen text.
skill_type: support
---

## How it works

Samples video at 1 FPS (configurable via `OCR_SAMPLE_FPS`), runs EasyOCR on each
frame on CPU (to avoid CUDA conflicts with the Video LLM), and deduplicates text
lines across frames. Returns timestamped text segments.

## Output
- **Type**: support — provides raw OCR text as evidence; does not output option_scores.
- `artifacts["text_lines"]`: list of `{text, timestamp, confidence}` dicts.
- `artifacts["full_text"]`: deduplicated concatenated text string.
