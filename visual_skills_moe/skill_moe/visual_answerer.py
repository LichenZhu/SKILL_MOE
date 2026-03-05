"""
VisualAnswerer — gpt-4o-vision fallback for the Spatio-Temporal Attention path.

Used when:
  - The pipeline has high-res crops in visual_evidence but video_llm is None, OR
  - video_llm.answer_with_crops() raises an OOM / decoding error.

Sends crops (base64 JPEG) directly to gpt-4o via the OpenAI-compatible API.
No video is passed — the crops carry the spatially precise evidence.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

_VISION_MODEL = os.getenv("VISUAL_ANSWERER_MODEL", "gpt-4o")
_MAX_CROPS    = int(os.getenv("VISUAL_ANSWERER_MAX_CROPS", "3"))


def answer_with_crops(
    question: str,
    crops_b64: List[str],
    target_desc: str = "target region",
    skill_context: Optional[str] = None,
    options_text: str = "",
) -> str:
    """
    Query gpt-4o vision with high-res crops + the question.

    Returns the model's raw text response (caller extracts answer letter).
    Raises on API error so the pipeline can catch and fall back further.
    """
    import litellm  # type: ignore

    content: list[dict] = []

    # Inject crops as high-detail images.
    for b64_str in crops_b64[:_MAX_CROPS]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_str}",
                "detail": "high",  # use high detail for fine-grained reading
            },
        })

    # Compose the prompt.
    prompt_parts = [
        f"The image(s) above are high-resolution crops from a video, "
        f"zoomed in on: '{target_desc}'.",
        "",
        "Question about the video:",
        question,
    ]
    if options_text:
        prompt_parts += ["", "Options:", options_text]
    if skill_context:
        prompt_parts += [
            "",
            "Additional context from other analysis tools:",
            skill_context,
        ]
    prompt_parts += [
        "",
        "Based on these high-resolution crops, answer the multiple-choice question.",
        "Reason briefly, then output ONLY the answer letter (A, B, C, or D).",
    ]
    content.append({"type": "text", "text": "\n".join(prompt_parts)})

    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key  = os.getenv("OPENAI_API_KEY", "sk-placeholder")
    model    = f"openai/{_VISION_MODEL}" if api_base else _VISION_MODEL

    logger.info(
        "[VisualAnswerer] gpt-4o vision  crops=%d  target='%s'",
        min(len(crops_b64), _MAX_CROPS), target_desc,
    )

    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=300,
        api_base=api_base,
        api_key=api_key,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()
