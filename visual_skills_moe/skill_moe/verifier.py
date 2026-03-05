"""
Evidence Verifier — lightweight LLM gate on skill output quality.

Prevents Tool Sycophancy: if a skill returns output that is unrelated or
misleading to the question, the verifier marks it as no_match=True so
_build_evidence_text skips it before VLM re-answer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SkillResponse
    from .llm_clients import LLMClient

logger = logging.getLogger(__name__)

# Truncate long skill outputs to keep the verification prompt cheap.
_MAX_OUTPUT_CHARS = 400

# Prompt is deliberately terse — we only need YES/NO.
_PROMPT = """\
Question: {question}
Skill: {skill_name}
Skill output: {output}

Does this skill output contain information that is genuinely relevant and \
helpful for answering the question?
Answer YES if the output directly addresses the question or provides clear \
supporting evidence.
Answer NO if the output is unrelated, ambiguous, or would likely mislead \
the answer.
Respond with only YES or NO."""


class EvidenceVerifier:
    """Calls a cheap LLM to decide whether to inject a skill response as evidence.

    Design principles:
    - Fail-open: any exception (timeout, API error, parse failure) → keep evidence.
      It is safer to inject questionable evidence than to silently discard
      potentially useful evidence due to a transient API error.
    - Cheap: max_tokens=10 (only YES/NO needed); prompt <600 chars total.
    - Transparent: all rejections are logged at INFO with the LLM's raw reply.
    """

    def __init__(self, llm: "LLMClient", max_tokens: int = 10) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def verify(self, question: str, response: "SkillResponse") -> bool:
        """Return True (keep) or False (reject) for a skill response.

        A rejected response has its artifacts["no_match"] set to True by
        the caller so _build_evidence_text skips it.
        """
        try:
            output = (response.evidence_text() or response.summary or "").strip()
            if not output:
                # No evidence text at all — nothing to inject.
                return False

            output_trunc = output[:_MAX_OUTPUT_CHARS]
            prompt = _PROMPT.format(
                question=question[:300],
                skill_name=response.skill_name,
                output=output_trunc,
            )
            raw = self._llm.complete(prompt, max_tokens=self._max_tokens)
            keep = raw.strip().upper().startswith("YES")

            if not keep:
                logger.info(
                    "[VERIFIER:REJECT] skill=%s  q=%r  llm_reply=%r",
                    response.skill_name,
                    question[:60],
                    raw.strip()[:30],
                )
            else:
                logger.debug(
                    "[VERIFIER:KEEP] skill=%s  q=%r",
                    response.skill_name,
                    question[:60],
                )

            return keep

        except Exception as exc:
            logger.warning(
                "[VERIFIER:ERROR] %s — failing open (keeping evidence): %s",
                response.skill_name,
                exc,
            )
            return True  # fail-open: keep evidence on any error
