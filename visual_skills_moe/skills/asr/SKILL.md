---
name: asr
description: Transcribe speech in the referenced video segment using local faster-whisper. Returns a timestamped transcript for questions about spoken content, dialogue, or audio narration.
tags: speech, transcript, audio, whisper
when_to_use: Use when the question asks what was said, spoken words, or dialogue. For "what was mentioned/discussed" questions, prefer rag_asr which filters to question-relevant sentences.
skill_type: support
---

## How it works

Runs faster-whisper locally (GPU if available, CPU fallback) on the specified
video segment. Returns deduplicated timestamped segments. Supports optional
cloud fallback via `ASR_ALLOW_REMOTE_FALLBACK=1`.

## Output
- **Type**: support — provides transcript text as evidence; does not output option_scores.
- `artifacts["transcript"]`: full timestamped transcript string.
- `artifacts["segments"]`: list of `{start, end, text}` dicts.
