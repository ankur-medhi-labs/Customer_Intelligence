"""Use Case 5 - practical evaluation harness.

Covers: schema/grounding validity, topic-classification correctness against real gold labels,
RAG answer correctness (including abstention on out-of-scope questions), analysis consistency
under re-runs, and temperature reliability (LLM vs rule-based agreement rate).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import settings
from .data_loader import load_processed
from .interaction_analysis import analyze_interaction
from .logging_setup import get_logger
from .rag.qa import answer as rag_answer

logger = get_logger(__name__)

_GOLD_DIR = Path("eval/gold")


def _check(name: str, passed: int, total: int, details: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "total": total, "details": details}


def eval_grounding_and_topics() -> dict[str, Any]:
    """Re-analyze a gold sample of real conversations and check: (a) no ungrounded evidence
    quotes, (b) the LLM's primary_topic_family matches the family embedded in the dataset."""
    gold = json.loads((_GOLD_DIR / "interaction_gold.json").read_text(encoding="utf-8"))
    all_conversations = {c.id: c for c in load_processed()}

    details: list[dict[str, Any]] = []
    passed = 0
    for conv_id in gold["conversation_ids"]:
        conv = all_conversations.get(conv_id)
        if conv is None or not conv.given_topic_label:
            continue
        expected_family = conv.given_topic_label.split(" - ")[0].strip()
        try:
            result = analyze_interaction(conv.raw_text)
        except Exception as exc:  # noqa: BLE001
            details.append({"id": conv_id, "ok": False, "error": str(exc)})
            continue
        family_ok = result.primary_topic_family == expected_family
        grounded_ok = not result.ungrounded_evidence
        ok = family_ok and grounded_ok
        passed += int(ok)
        details.append({
            "id": conv_id, "ok": ok,
            "expected_family": expected_family, "got_family": result.primary_topic_family,
            "ungrounded_evidence": result.ungrounded_evidence,
        })
    return _check("interaction_grounding_and_topic", passed, len(details), details)


def eval_consistency() -> dict[str, Any]:
    """Re-run analysis twice on the same conversations and check temperature_band /
    escalation_required agree between runs (a proxy for output stability/reliability)."""
    gold = json.loads((_GOLD_DIR / "interaction_gold.json").read_text(encoding="utf-8"))
    all_conversations = {c.id: c for c in load_processed()}

    details: list[dict[str, Any]] = []
    passed = 0
    # Keep this check cheap: only re-run a handful of conversations twice each.
    for conv_id in gold["conversation_ids"][:5]:
        conv = all_conversations.get(conv_id)
        if conv is None:
            continue
        try:
            r1 = analyze_interaction(conv.raw_text)
            r2 = analyze_interaction(conv.raw_text)
        except Exception as exc:  # noqa: BLE001
            details.append({"id": conv_id, "ok": False, "error": str(exc)})
            continue
        ok = r1.temperature_band == r2.temperature_band and r1.escalation_required == r2.escalation_required
        passed += int(ok)
        details.append({
            "id": conv_id, "ok": ok,
            "band_1": r1.temperature_band, "band_2": r2.temperature_band,
            "escalation_1": r1.escalation_required, "escalation_2": r2.escalation_required,
        })
    return _check("analysis_consistency", passed, len(details), details)


def eval_temperature_reliability() -> dict[str, Any]:
    """Report the LLM-vs-rule-based temperature agreement rate over the cached analyzed batch."""
    analyzed_path = Path(settings.processed_dir) / "analyzed.jsonl"
    if not analyzed_path.exists():
        return _check("temperature_reliability", 0, 0, [])

    from .interaction_analysis import load_analyzed
    records = load_analyzed(analyzed_path)
    details = []
    passed = 0
    for r in records:
        rel = r.analysis.temperature_reliability
        if rel is None:
            continue
        passed += int(rel.agreement)
        details.append({
            "id": r.conversation.id, "agreement": rel.agreement,
            "llm_score": rel.llm_score, "rule_score": rel.rule_score,
        })
    return _check("temperature_reliability", passed, len(details), details)


def eval_rag() -> dict[str, Any]:
    """Check RAG answers against gold Q&A pairs, including expected abstentions."""
    gold = json.loads((_GOLD_DIR / "rag_gold.json").read_text(encoding="utf-8"))
    details: list[dict[str, Any]] = []
    passed = 0
    for item in gold:
        result = rag_answer(item["question"])
        answered_ok = result.answered == item["expect_answered"]
        source_ok = True
        if item.get("expect_source_contains") and result.answered:
            source_ok = any(
                item["expect_source_contains"].lower() in c.source.lower()
                or item["expect_source_contains"].lower() in c.title.lower()
                for c in result.citations
            )
        ok = answered_ok and source_ok
        passed += int(ok)
        details.append({
            "question": item["question"], "ok": ok,
            "expected_answered": item["expect_answered"], "got_answered": result.answered,
            "source_ok": source_ok,
        })
    return _check("rag_answer_correctness", passed, len(details), details)


def run_evaluation() -> dict[str, Any]:
    checks = [
        eval_grounding_and_topics(),
        eval_consistency(),
        eval_temperature_reliability(),
        eval_rag(),
    ]
    return {"checks": checks}
