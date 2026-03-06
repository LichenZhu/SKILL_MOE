---
name: rag_asr
description: Retrieval-Augmented ASR — transcribes speech then extracts the 1-3 sentences most relevant to the question via a lightweight LLM client. Avoids injecting irrelevant dialogue noise into the VLM context.
tags: speech, transcript, audio, rag, retrieval
when_to_use: Use when the question asks what was mentioned, discussed, talked about, or stated in the video. Preferred over plain asr for "mentioned-in-video" questions where only part of the transcript is relevant.
skill_type: support
---

## How it works

Internally calls the `asr` runner to obtain a full transcript, then sends
the transcript + question to the configured LLM client (via `OPENAI_MODEL` /
`OPENAI_BASE_URL`) to extract the 1–3 most relevant sentences. Falls back to
the full transcript if no LLM is available.

## Output
- **Type**: support — provides filtered transcript text as evidence; does not output option_scores.
- `artifacts["relevant_text"]`: extracted relevant sentences (empty if nothing matched).
- Returns `no_match=True` if the filtered result is empty, preventing noise injection.
