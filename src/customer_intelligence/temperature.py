"""Customer temperature scoring: a 0-100 escalation-risk measure with a reliability cross-check.

Design: the LLM assigns a temperature_band/temperature_score as part of structured interaction
analysis (grounded in conversation content). Independently, a deterministic rule-based scorer
computes a second opinion from sentiment + escalation keyword signals. Comparing the two gives a
practical reliability signal: when they diverge sharply, the score is flagged as low-confidence
and the analysis should be reviewed by a human rather than trusted blindly.
"""
from __future__ import annotations

from .schemas import TemperatureReliability

# Base score contribution from the LLM-assigned sentiment.
_SENTIMENT_BASE: dict[str, int] = {
    "very_negative": 65,
    "negative": 40,
    "neutral": 15,
    "positive": 5,
    "very_positive": 0,
}

ESCALATION_KEYWORDS = [
    "cancel", "cancelling", "cancellation", "legal", "lawyer", "sue", "refund now",
    "unacceptable", "never again", "third time", "furious", "disgusted", "scam", "fraud",
    "formal complaint", "complaint", "regulator", "ombudsman", "compensation", "disgraceful",
]

REPEAT_KEYWORDS = ["again", "still not", "keeps", "every time", "multiple times", "over and over"]


def rule_based_score(text: str, sentiment: str) -> int:
    """Deterministic 0-100 escalation-risk score from sentiment + keyword/punctuation signals."""
    text_lower = text.lower()
    score = _SENTIMENT_BASE.get(sentiment, 15)

    escalation_hits = sum(1 for kw in ESCALATION_KEYWORDS if kw in text_lower)
    score += escalation_hits * 12

    repeat_hits = sum(1 for kw in REPEAT_KEYWORDS if kw in text_lower)
    score += repeat_hits * 6

    exclamations = text.count("!")
    score += min(exclamations * 3, 15)

    return max(0, min(100, score))


def reconcile(llm_score: int, rule_score: int) -> TemperatureReliability:
    """Compare the LLM's score with the rule-based score and produce a reconciled result."""
    llm_score = max(0, min(100, llm_score))
    rule_score = max(0, min(100, rule_score))
    diff = abs(llm_score - rule_score)
    agreement = diff <= 20
    final_score = llm_score if agreement else round((llm_score + rule_score) / 2)
    return TemperatureReliability(
        llm_score=llm_score, rule_score=rule_score, agreement=agreement, final_score=final_score,
    )
