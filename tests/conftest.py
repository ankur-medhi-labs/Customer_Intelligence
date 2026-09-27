"""Shared pytest fixtures. All tests run offline - the shared `bedrock` client singleton is
monkeypatched so no real network/API calls are made."""
from __future__ import annotations

import numpy as np
import pytest

SAMPLE_CONVERSATION = (
    'My Wi-Fi is acting up again, keeps dropping the connection like every 5 minutes. '
    'Super annoying! What gives?'
)


@pytest.fixture
def sample_conversation_text() -> str:
    return SAMPLE_CONVERSATION


@pytest.fixture
def fake_bedrock(monkeypatch):
    """Patch the shared `bedrock` singleton's LLM/embedding calls with deterministic fakes."""
    from customer_intelligence.bedrock import bedrock

    def fake_converse_json(system: str, user: str, schema: dict, tool_name: str) -> dict:
        properties = schema.get("properties", {})
        if "formal_summary" in properties:
            return {
                "formal_summary": "The customer reported an intermittent Wi-Fi connectivity issue.",
                "primary_topic_family": "Technical Issue",
                "primary_topic": "Technical Issue - Connectivity",
                "topic_confidence": 0.9,
                "customer_intent": "Resolve repeated Wi-Fi disconnections.",
                "sentiment": "negative",
                "temperature_band": "frustrated",
                "temperature_score": 55,
                "escalation_required": False,
                "escalation_reason": "",
                "recommended_next_action": "Restart the router and check for regional outages.",
                "evidence": ["Super annoying!"],
            }
        if "label" in properties and "description" in properties:
            return {"label": "Fake Theme", "description": "A fake theme for testing."}
        if "label" in properties and "summary" in properties:
            return {"label": "Fake Emerging Issue", "summary": "A fake emerging issue for testing."}
        raise AssertionError(f"Unexpected schema in fake_converse_json: {properties.keys()}")

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # Deterministic pseudo-embedding: hash each text into a fixed-size vector.
        dim = 16
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.normal(size=dim).tolist())
        return vectors

    monkeypatch.setattr(bedrock, "converse_json", fake_converse_json)
    monkeypatch.setattr(bedrock, "embed", fake_embed)
    return bedrock
