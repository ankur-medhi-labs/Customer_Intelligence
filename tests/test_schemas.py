"""Unit tests for pydantic schemas: validation, defensive remapping, and clamping."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_intelligence.schemas import Citation, InteractionAnalysis, RagAnswer


def _base_analysis_kwargs(**overrides):
    kwargs = dict(
        formal_summary="The customer reported an issue.",
        primary_topic_family="Billing",
        primary_topic="Overcharge",
        topic_confidence=0.8,
        customer_intent="Get a refund for a duplicate charge.",
        sentiment="negative",
        temperature_band="frustrated",
        temperature_score=55,
        escalation_required=False,
        escalation_reason="",
        recommended_next_action="Issue a refund.",
        evidence=["That's not right."],
    )
    kwargs.update(overrides)
    return kwargs


def test_topic_prefix_is_coerced():
    analysis = InteractionAnalysis(**_base_analysis_kwargs(primary_topic="Overcharge"))
    assert analysis.primary_topic == "Billing - Overcharge"


def test_temperature_score_is_clamped():
    analysis = InteractionAnalysis(**_base_analysis_kwargs(temperature_score=150))
    assert analysis.temperature_score == 100
    analysis = InteractionAnalysis(**_base_analysis_kwargs(temperature_score=-10))
    assert analysis.temperature_score == 0


def test_sentiment_remaps_known_temperature_band_confusion():
    analysis = InteractionAnalysis(**_base_analysis_kwargs(sentiment="concerned"))
    assert analysis.sentiment == "negative"
    analysis = InteractionAnalysis(**_base_analysis_kwargs(sentiment="calm"))
    assert analysis.sentiment == "neutral"


def test_temperature_band_remaps_known_sentiment_confusion():
    analysis = InteractionAnalysis(**_base_analysis_kwargs(temperature_band="negative"))
    assert analysis.temperature_band == "frustrated"


def test_invalid_sentiment_still_raises():
    with pytest.raises(ValidationError):
        InteractionAnalysis(**_base_analysis_kwargs(sentiment="totally_unknown_value"))


def test_citation_and_rag_answer_construction():
    citation = Citation(source="Billing Policy", title="Refunds", chunk_id="refunds#0", score=0.9, snippet="...")
    answer = RagAnswer(question="What is your refund policy?", answer="...", answered=True, citations=[citation])
    assert answer.answered is True
    assert answer.citations[0].chunk_id == "refunds#0"
