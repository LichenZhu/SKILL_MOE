---
name: rag_asr
description: Retrieval-Augmented ASR — transcribes speech then extracts the 1-3 sentences most relevant to the question using gpt-4o-mini. Avoids sycophancy from irrelevant dialogue.
tags: speech, transcript, audio, rag, retrieval
when_to_use: Use when the question asks what was mentioned, discussed, talked about, or stated in the video. Preferred over asr for mentioned-in-video questions.
---

## Guidance
- Internally calls the ASR runner to obtain a full transcript.
- Filters transcript with a lightweight LLM to extract only relevant sentences.
- Returns NO_MATCH (empty evidence) if nothing is relevant — does not inject noise.
- Falls back to full transcript if LLM is unavailable.
