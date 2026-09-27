"""Unit tests for the rule-based temperature scorer and reliability reconciliation."""
from __future__ import annotations

from customer_intelligence.temperature import rule_based_score, reconcile


def test_escalation_keywords_raise_the_score():
    calm_text = "Could you please help me update my email address?"
    angry_text = "This is unacceptable, I want to cancel and I'm speaking to my lawyer!"
    calm_score = rule_based_score(calm_text, sentiment="neutral")
    angry_score = rule_based_score(angry_text, sentiment="very_negative")
    assert angry_score > calm_score


def test_score_is_clamped_to_0_100():
    text = "cancel " * 20 + "!!!!!!!!!!"
    score = rule_based_score(text, sentiment="very_negative")
    assert 0 <= score <= 100


def test_reconcile_agrees_when_scores_are_close():
    reliability = reconcile(llm_score=55, rule_score=60)
    assert reliability.agreement is True
    assert reliability.final_score == 55


def test_reconcile_disagrees_and_averages_when_scores_diverge():
    reliability = reconcile(llm_score=10, rule_score=90)
    assert reliability.agreement is False
    assert reliability.final_score == 50


def test_reconcile_clamps_out_of_range_inputs():
    reliability = reconcile(llm_score=150, rule_score=-20)
    assert reliability.llm_score == 100
    assert reliability.rule_score == 0
