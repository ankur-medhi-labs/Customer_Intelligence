"""Use Case 1 - structured customer interaction analysis (summary, topic, sentiment, temperature,
escalation, and recommended next action), grounded in the conversation text.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bedrock import bedrock
from .logging_setup import get_logger
from .schemas import AnalyzedConversation, Conversation, InteractionAnalysis
from .taxonomy import format_taxonomy_for_prompt, band_for_score
from .temperature import rule_based_score, reconcile

logger = get_logger(__name__)

_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "interaction_analysis.txt"
).read_text(encoding="utf-8")

_SYSTEM_PROMPT = (
    "You are a precise, grounded customer-support analytics engine. "
    "Only use facts present in the conversation provided."
)


def _check_grounding(evidence: list[str], conversation_text: str) -> list[str]:
    """Return the subset of evidence quotes that are NOT verbatim substrings of the conversation."""
    ungrounded = []
    haystack = conversation_text.lower()
    for quote in evidence:
        if quote.strip() and quote.strip().lower() not in haystack:
            ungrounded.append(quote)
    return ungrounded


def analyze_interaction(conversation_text: str) -> InteractionAnalysis:
    """Analyze a single customer conversation and return a validated, grounded result."""
    prompt = _PROMPT_TEMPLATE.replace("{taxonomy}", format_taxonomy_for_prompt()).replace(
        "{chat}", conversation_text
    )

    raw = bedrock.converse_json(
        system=_SYSTEM_PROMPT,
        user=prompt,
        schema=InteractionAnalysis.tool_schema(),
        tool_name="emit_analysis",
    )
    analysis = InteractionAnalysis(**raw)

    # Grounding check: flag any evidence quote that isn't verbatim in the source conversation.
    ungrounded = _check_grounding(analysis.evidence, conversation_text)
    if ungrounded:
        logger.warning("Ungrounded evidence quotes detected: %s", ungrounded)
    analysis.ungrounded_evidence = ungrounded

    # Reliability cross-check: reconcile the LLM's self-reported score with a rule-based score.
    rule_score = rule_based_score(conversation_text, analysis.sentiment)
    reliability = reconcile(analysis.temperature_score, rule_score)
    analysis.temperature_reliability = reliability
    analysis.temperature_score = reliability.final_score
    analysis.temperature_band = band_for_score(reliability.final_score)

    return analysis


def analyze_batch(conversations: list[Conversation]) -> list[AnalyzedConversation]:
    """Run `analyze_interaction` over a batch of conversations, skipping any that error out."""
    results: list[AnalyzedConversation] = []
    for conv in conversations:
        try:
            analysis = analyze_interaction(conv.raw_text)
        except Exception as exc:  # noqa: BLE001 - one bad row shouldn't stop the batch
            logger.error("Failed to analyze conversation %s: %s", conv.id, exc)
            continue
        results.append(AnalyzedConversation(conversation=conv, analysis=analysis))
    return results


def save_analyzed(records: list[AnalyzedConversation], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.info("Saved %d analyzed conversations -> %s", len(records), out_path)
    return out_path


def load_analyzed(path: str | Path) -> list[AnalyzedConversation]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Analyzed conversations not found at {in_path}")
    records: list[AnalyzedConversation] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(AnalyzedConversation(**json.loads(line)))
    return records
