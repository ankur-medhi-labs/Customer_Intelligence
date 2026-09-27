"""
Use Case 4 - RAG Question & Answer with grounding and abstention.

Two-layer abstention:
    1. Retrieval gate - if the best chunk score < threshold, abstain before generation.
    2. Model sentinel - the prompt instructs the LLM to return "NOT_SUPPORTED" if the context
       does not support the answer; detected and surfaced as answered=False.
"""
from __future__ import annotations

from pathlib import Path

from ..bedrock import bedrock
from ..config import settings
from ..schemas import Citation, RagAnswer
from .retriever import retrieve
from ..logging_setup import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "rag_answer.txt"
).read_text(encoding="utf-8")

ABSTAIN_MSG = (
    "I'm sorry, I do not have information in my knowledge base to answer that question. "
    "Please contact support directly for assistance."
)

NOT_SUPPORTED_TOKEN = "NOT_SUPPORTED"

def _format_context(citations: list[Citation]) -> str:
    """Format retrieved chunks as numbered context for the prompt."""
    lines: list[str] = []
    for i, cit in enumerate(citations, 1):
        lines.append(f"[{i}] Source: {cit.title} ({cit.source})")
        lines.append(cit.snippet)
        lines.append("")
    return "\n".join(lines)

def answer(question: str) -> RagAnswer:
    """
    Answer a question using the RAG knowledge base.

    Returns a RagAnswer with:
    - answered=True  -> grounded answer + citations
    - answered=False -> abstention message, no fabricated content
    """
    logger.debug("[rag.answer] question=%r", question)

    # step 1: retrieve
    citations = retrieve(question, k=settings.retrieval_top_k)
    logger.debug(
        "[rag.answer] retrieved %d chunks: %s",
        len(citations),
        [(c.title[:30], round(c.score, 3)) for c in citations],
    )

    # step 2: relevance gate
    if not citations or citations[0].score < settings.retrieval_min_score:
        logger.info(
            "[rag.answer] ABSTAIN - best score=%.3f < threshold=%.3f",
            citations[0].score if citations else 0.0,
            settings.retrieval_min_score,
        )
        return RagAnswer(
            question=question,
            answer=ABSTAIN_MSG,
            answered=False,
            citations=[],
        )

    # step 3: build prompt and call LLM
    context = _format_context(citations)
    prompt = _PROMPT_TEMPLATE.replace("{context}", context).replace("{question}", question)

    logger.debug("[rag.answer] context_len=%d prompt_len=%d", len(context), len(prompt))

    system = (
        "You are a customer support knowledge assistant. "
        "Answer ONLY from the provided context. "
        f"If the context does not support an answer, reply exactly: {NOT_SUPPORTED_TOKEN}"
    )

    try:
        result = bedrock.converse(
            system=system,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        raw_answer = result.text.strip()
        logger.debug("[rag.answer] raw_answer first(200): %s", raw_answer[:200])
    except Exception as exc:
        logger.error("[rag.answer] generation failed: %s", exc)
        return RagAnswer(
            question=question,
            answer="An error occurred. Please try again.",
            answered=False,
            citations=citations,
        )

    # step 4: detect model-level abstention
    if NOT_SUPPORTED_TOKEN in raw_answer:
        logger.info("[rag.answer] model returned NOT_SUPPORTED - abstaining")
        return RagAnswer(
            question=question,
            answer=ABSTAIN_MSG,
            answered=False,
            citations=citations,
        )

    logger.debug("[rag.answer] answered=True citations=%d", len(citations))
    return RagAnswer(
        question=question,
        answer=raw_answer,
        answered=True,
        citations=citations,
    )


