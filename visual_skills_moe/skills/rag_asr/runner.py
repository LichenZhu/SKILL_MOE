"""
Retrieval-Augmented ASR skill.

Runs Whisper to obtain the full transcript (reusing the ASR runner's model
cache and caching infrastructure), then uses gpt-4o-mini to extract only
the 1-3 sentences most relevant to the question.

This avoids the "Tool Sycophancy" failure mode of vanilla ASR where the
VideoLLM gets distracted by irrelevant dialogue and overrides a correct
visual answer.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from skill_moe.base import SkillMetadata, SkillRequest, SkillResponse
from skill_moe.llm_clients import default_llm_client

# ── Lazy-load the ASR runner to reuse its Whisper model cache ────────────────
_ASR_MODULE = None


def _get_asr_module():
    global _ASR_MODULE
    if _ASR_MODULE is not None:
        return _ASR_MODULE
    asr_path = Path(__file__).parent.parent / "asr" / "runner.py"
    spec = importlib.util.spec_from_file_location("_rag_asr_impl", str(asr_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ASR_MODULE = mod
    return mod


# Fake metadata for the inner ASR call (name must be "asr" so caching keys match).
_ASR_META = SkillMetadata(
    name="asr",
    description="Internal ASR call from rag_asr",
    path=str(Path(__file__).parent.parent / "asr"),
)

_EXTRACTION_PROMPT = """\
Below is a transcript from a video, followed by a question about the video.

TRANSCRIPT:
{transcript}

QUESTION:
{question}

Your task: Extract ONLY the 1 to 3 sentences from the transcript that \
directly answer or are most relevant to the question.
Rules:
- Copy sentences verbatim from the transcript.
- Do NOT paraphrase or summarize.
- If no sentence is relevant, output exactly: NO_MATCH

Relevant sentences:"""


def run(request: SkillRequest, metadata: SkillMetadata) -> SkillResponse:
    if not os.path.isfile(request.video_path):
        return SkillResponse(
            skill_name=metadata.name,
            summary="[RAG-ASR] Error: video file not found.",
            artifacts={"error": "missing_file"},
        )

    # ── Step 1: transcribe with Whisper (reuse ASR model cache) ──────────────
    asr_mod = _get_asr_module()
    asr_resp = asr_mod.run(request, _ASR_META)
    transcript = (asr_resp.content or "").strip()

    if not transcript:
        return SkillResponse(
            skill_name=metadata.name,
            summary="[RAG-ASR] No speech detected in the video segment.",
            content="",
            artifacts={"transcript_len": 0, "snippet": "", "no_match": True},
        )

    # ── Step 2: filter with LLM ───────────────────────────────────────────────
    llm = default_llm_client()
    if llm is None:
        # No LLM available — fall back to returning the full transcript (vanilla ASR).
        return SkillResponse(
            skill_name=metadata.name,
            summary=(
                f"[RAG-ASR] LLM unavailable — returning full transcript "
                f"({len(transcript)} chars)."
            ),
            content=transcript,
            artifacts={"transcript_len": len(transcript), "snippet": transcript,
                       "fallback": "full_transcript"},
        )

    prompt = _EXTRACTION_PROMPT.format(
        transcript=transcript[:4000],   # cap to avoid exceeding context
        question=request.question[:500],
    )
    try:
        snippet = llm.complete(prompt, max_tokens=300).strip()
    except Exception as exc:
        # LLM call failed — fall back to full transcript.
        return SkillResponse(
            skill_name=metadata.name,
            summary=(
                f"[RAG-ASR] LLM extraction failed ({exc!r}); returning full transcript."
            ),
            content=transcript,
            artifacts={"transcript_len": len(transcript), "snippet": transcript,
                       "error": str(exc)},
        )

    # ── Step 3: handle NO_MATCH ───────────────────────────────────────────────
    if "NO_MATCH" in snippet.upper():
        return SkillResponse(
            skill_name=metadata.name,
            summary=(
                f"[RAG-ASR] No relevant speech found for this question "
                f"(transcript {len(transcript)} chars examined)."
            ),
            content="",
            artifacts={"transcript_len": len(transcript), "snippet": "",
                       "no_match": True},
        )

    return SkillResponse(
        skill_name=metadata.name,
        summary=(
            f"[RAG-ASR] Extracted {len(snippet)} char snippet from "
            f"{len(transcript)} char transcript."
        ),
        content=snippet,
        artifacts={
            "transcript_len": len(transcript),
            "snippet": snippet,
            "no_match": False,
        },
    )
