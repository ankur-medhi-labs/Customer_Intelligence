"""Tests for the interaction-analysis pipeline, using the fake_bedrock fixture (no network calls)."""
from __future__ import annotations

from customer_intelligence.interaction_analysis import analyze_interaction


def test_analyze_interaction_returns_grounded_result(fake_bedrock, sample_conversation_text):
    result = analyze_interaction(sample_conversation_text)

    assert result.primary_topic_family == "Technical Issue"
    assert result.evidence == ["Super annoying!"]
    assert result.ungrounded_evidence == []
    assert result.temperature_reliability is not None
    assert result.temperature_reliability.agreement is True


def test_analyze_interaction_flags_ungrounded_evidence(monkeypatch, fake_bedrock, sample_conversation_text):
    from customer_intelligence.bedrock import bedrock

    original_converse_json = bedrock.converse_json

    def fake_with_bad_evidence(system, user, schema, tool_name):
        result = original_converse_json(system, user, schema, tool_name)
        result["evidence"] = ["this quote was never said by the customer"]
        return result

    monkeypatch.setattr(bedrock, "converse_json", fake_with_bad_evidence)

    result = analyze_interaction(sample_conversation_text)

    assert result.ungrounded_evidence == ["this quote was never said by the customer"]
