---
name: asr
description: Transcribe speech in the referenced video segment to answer questions about spoken content.
tags: speech, transcript, audio
when_to_use: Use when the question asks what was said, spoken words, or dialogue.
---

## Guidance
- Accept a clip range when available to trim inference cost.
- Default to local faster-whisper (GPU when available) to avoid ASR API cost.
- Enable cloud fallback only when explicitly needed (`ASR_ALLOW_REMOTE_FALLBACK=1`).
- Return timestamped words when possible.
