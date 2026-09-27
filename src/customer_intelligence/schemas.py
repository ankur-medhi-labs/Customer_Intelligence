"""Pydantic data models shared across the application."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .taxonomy import TOPIC_FAMILIES, SENTIMENTS

TopicFamily = Literal["Technical Issue", "Billing", "Delivery", "Account Management", "Other"]
Sentiment = Literal["very_negative", "negative", "neutral", "positive", "very_positive"]
TemperatureBand = Literal["calm", "concerned", "frustrated", "angry", "critical"]

# The two scales are conceptually adjacent, so the model occasionally emits a temperature-band
# word where a sentiment word is expected (or vice versa). Remap known confusions defensively
# rather than hard-failing the whole analysis.
_BAND_TO_SENTIMENT = {
    "calm": "neutral", "concerned": "negative", "frustrated": "negative",
    "angry": "very_negative", "critical": "very_negative",
}
_SENTIMENT_TO_BAND = {
    "very_negative": "critical", "negative": "frustrated", "neutral": "concerned",
    "positive": "calm", "very_positive": "calm",
}


class Conversation(BaseModel):
    """A single raw customer chat record loaded from the dataset."""

    id: str
    raw_text: str
    given_topic_label: str | None = None
    expected_resolution: str | None = None
    source_row: int | None = None


class TemperatureReliability(BaseModel):
    """Cross-check between the LLM's self-reported temperature score and a rule-based score."""

    llm_score: int
    rule_score: int
    agreement: bool
    final_score: int


class InteractionAnalysis(BaseModel):
    """Structured result for a single customer interaction (Use Case 1)."""

    formal_summary: str
    primary_topic_family: TopicFamily
    primary_topic: str
    topic_confidence: float = Field(ge=0.0, le=1.0)
    customer_intent: str
    sentiment: Sentiment
    temperature_band: TemperatureBand
    temperature_score: int
    escalation_required: bool
    escalation_reason: str = ""
    recommended_next_action: str
    evidence: list[str] = Field(default_factory=list)

    # populated after LLM output is validated, not part of the LLM's tool schema
    temperature_reliability: TemperatureReliability | None = None
    ungrounded_evidence: list[str] = Field(default_factory=list)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _remap_sentiment(cls, v: object) -> object:
        return _BAND_TO_SENTIMENT.get(v, v) if isinstance(v, str) else v

    @field_validator("temperature_band", mode="before")
    @classmethod
    def _remap_temperature_band(cls, v: object) -> object:
        return _SENTIMENT_TO_BAND.get(v, v) if isinstance(v, str) else v

    @field_validator("temperature_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: int) -> int:
        return max(0, min(100, v))

    @model_validator(mode="after")
    def _coerce_topic_prefix(self) -> "InteractionAnalysis":
        if not self.primary_topic.startswith(self.primary_topic_family):
            self.primary_topic = f"{self.primary_topic_family} - {self.primary_topic}".strip()
        return self

    @classmethod
    def tool_schema(cls) -> dict:
        """JSON schema for the `emit_analysis` tool, excluding post-hoc fields."""
        schema = cls.model_json_schema()
        for key in ("temperature_reliability", "ungrounded_evidence"):
            schema.get("properties", {}).pop(key, None)
        schema["required"] = [
            f for f in schema.get("required", [])
            if f not in ("temperature_reliability", "ungrounded_evidence")
        ]
        return schema


class AnalyzedConversation(BaseModel):
    """A raw conversation paired with its structured interaction analysis (cached to disk)."""

    conversation: Conversation
    analysis: InteractionAnalysis


class ThemeLabelOutput(BaseModel):
    label: str
    description: str


class EmergingSummaryOutput(BaseModel):
    label: str
    summary: str


class Citation(BaseModel):
    source: str
    title: str
    chunk_id: str
    score: float
    snippet: str


class RagAnswer(BaseModel):
    question: str
    answer: str
    answered: bool
    citations: list[Citation] = Field(default_factory=list)


class Theme(BaseModel):
    label: str
    description: str
    count: int
    avg_temperature_score: float
    temperature_band_distribution: dict[str, int] = Field(default_factory=dict)
    example_conversation_ids: list[str] = Field(default_factory=list)


class EmergingIssue(BaseModel):
    label: str
    summary: str
    count: int
    example_conversation_ids: list[str] = Field(default_factory=list)


class ThemeReport(BaseModel):
    themes: list[Theme]
    emerging_issues: list[EmergingIssue] = Field(default_factory=list)


__all__ = [
    "TOPIC_FAMILIES", "SENTIMENTS",
    "Conversation", "TemperatureReliability", "InteractionAnalysis", "AnalyzedConversation",
    "ThemeLabelOutput", "EmergingSummaryOutput",
    "Citation", "RagAnswer", "Theme", "EmergingIssue", "ThemeReport",
]
