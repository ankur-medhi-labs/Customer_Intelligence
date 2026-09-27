"""Use Case 3 - thematic analysis of customer conversations and emerging-issue detection.

Design notes (see README for the full rationale):
- Main themes are grouped by `primary_topic_family`, the already-grounded, validated
  classification produced by Use Case 1 (interaction_analysis) for every conversation. This is
  more robust than re-clustering raw embeddings from scratch: the family field comes from an LLM
  that read the full conversation, not just a similarity heuristic, and it keeps the "theme" view
  consistent with the per-interaction analysis.
- Emerging-issue detection deliberately does NOT use the fixed taxonomy (a taxonomy-constrained
  classifier would just force a genuinely new issue into one of the existing families, defeating
  the purpose). Instead it is a relative embedding-similarity test: a candidate conversation is
  flagged as novel when it looks more similar to its own batch peers than to anything in the known
  corpus - i.e. it is part of a new *recurring* pattern rather than an existing one. Thresholds are
  derived from the known corpus's own similarity scale at runtime rather than a hardcoded constant,
  since absolute cosine-similarity magnitudes vary a lot between embedding models/endpoints.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bedrock import bedrock
from .logging_setup import get_logger
from .schemas import (
    AnalyzedConversation, Conversation, EmergingIssue, EmergingSummaryOutput,
    Theme, ThemeLabelOutput, ThemeReport,
)

logger = get_logger(__name__)

_THEME_LABEL_PROMPT = (Path(__file__).parent / "prompts" / "theme_label.txt").read_text(encoding="utf-8")
_EMERGING_PROMPT = (Path(__file__).parent / "prompts" / "emerging_summary.txt").read_text(encoding="utf-8")

_SYSTEM_PROMPT = "You are a precise, grounded customer support analytics engine."

# How much more similar a candidate must be to its own batch peers than to the known corpus
# before it's treated as a new recurring pattern rather than noise.
NOVELTY_MARGIN = 0.05


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalize a list of texts so dot products give cosine similarity."""
    raw = np.array(bedrock.embed(texts), dtype="float32")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-10
    return raw / norms


def _greedy_cluster(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    """Single-pass clustering: assign each item to its most-similar existing centroid if above
    `threshold`, otherwise start a new cluster. Centroids are re-normalized running means."""
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for i, emb in enumerate(embeddings):
        best_idx, best_sim = -1, -1.0
        for c_idx, centroid in enumerate(centroids):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim, best_idx = sim, c_idx
        if best_idx >= 0 and best_sim >= threshold:
            clusters[best_idx].append(i)
            members = embeddings[clusters[best_idx]]
            mean = members.mean(axis=0)
            centroids[best_idx] = mean / (np.linalg.norm(mean) + 1e-10)
        else:
            clusters.append([i])
            centroids.append(emb)
    return clusters


def _label_theme(sample_texts: list[str]) -> ThemeLabelOutput:
    sample = "\n---\n".join(sample_texts)
    prompt = _THEME_LABEL_PROMPT.replace("{sample}", sample)
    raw = bedrock.converse_json(
        system=_SYSTEM_PROMPT, user=prompt,
        schema=ThemeLabelOutput.model_json_schema(), tool_name="emit_theme_label",
    )
    return ThemeLabelOutput(**raw)


def _label_emerging(sample_texts: list[str]) -> EmergingSummaryOutput:
    sample = "\n---\n".join(sample_texts)
    prompt = _EMERGING_PROMPT.replace("{sample}", sample)
    raw = bedrock.converse_json(
        system=_SYSTEM_PROMPT, user=prompt,
        schema=EmergingSummaryOutput.model_json_schema(), tool_name="emit_emerging_summary",
    )
    return EmergingSummaryOutput(**raw)


def _build_themes_by_family(records: list[AnalyzedConversation]) -> list[Theme]:
    groups: dict[str, list[AnalyzedConversation]] = {}
    for r in records:
        groups.setdefault(r.analysis.primary_topic_family, []).append(r)

    themes: list[Theme] = []
    for members in groups.values():
        sample_texts = [m.conversation.raw_text for m in members[:5]]
        labeled = _label_theme(sample_texts)
        temp_scores = [m.analysis.temperature_score for m in members]
        band_counts: dict[str, int] = {}
        for m in members:
            band_counts[m.analysis.temperature_band] = band_counts.get(m.analysis.temperature_band, 0) + 1
        themes.append(Theme(
            label=labeled.label,
            description=labeled.description,
            count=len(members),
            avg_temperature_score=round(sum(temp_scores) / len(temp_scores), 1),
            temperature_band_distribution=band_counts,
            example_conversation_ids=[m.conversation.id for m in members[:5]],
        ))
    themes.sort(key=lambda t: t.count, reverse=True)
    return themes


def _detect_emerging_issues(
    known_records: list[AnalyzedConversation], candidates: list[Conversation],
) -> list[EmergingIssue]:
    if not candidates:
        return []

    known_embeddings = _embed_texts([r.conversation.raw_text for r in known_records])
    candidate_embeddings = _embed_texts([c.raw_text for c in candidates])

    known_sims = known_embeddings @ known_embeddings.T
    np.fill_diagonal(known_sims, -1.0)
    # Typical similarity between a known item and its closest known peer - our clustering scale.
    cluster_scale = float(np.median(known_sims.max(axis=1))) if len(known_records) > 1 else 0.5

    cross_sims = candidate_embeddings @ known_embeddings.T
    peer_sims = candidate_embeddings @ candidate_embeddings.T
    np.fill_diagonal(peer_sims, -1.0)

    novel_indices: list[int] = []
    for i in range(len(candidates)):
        best_known = float(cross_sims[i].max())
        best_peer = float(peer_sims[i].max()) if len(candidates) > 1 else -1.0
        if best_peer > best_known + NOVELTY_MARGIN:
            novel_indices.append(i)
        else:
            logger.debug(
                "Candidate %s not flagged as emerging (best_known=%.3f best_peer=%.3f)",
                candidates[i].id, best_known, best_peer,
            )

    if not novel_indices:
        logger.info("No emerging issues detected - all candidates matched the known corpus")
        return []

    sub_embeddings = candidate_embeddings[novel_indices]
    sub_clusters = _greedy_cluster(sub_embeddings, threshold=cluster_scale)

    emerging_issues: list[EmergingIssue] = []
    for sub in sub_clusters:
        member_convs = [candidates[novel_indices[i]] for i in sub]
        sample_texts = [c.raw_text for c in member_convs[:5]]
        labeled = _label_emerging(sample_texts)
        emerging_issues.append(EmergingIssue(
            label=labeled.label,
            summary=labeled.summary,
            count=len(member_convs),
            example_conversation_ids=[c.id for c in member_convs],
        ))
    return emerging_issues


def build_theme_report(
    records: list[AnalyzedConversation],
    emerging_conversations: list[Conversation] | None = None,
) -> ThemeReport:
    """Group analyzed conversations into themes (by grounded topic family), and flag any
    emerging issue among `emerging_conversations` that doesn't fit the known corpus."""
    if not records:
        return ThemeReport(themes=[], emerging_issues=[])

    themes = _build_themes_by_family(records)
    emerging_issues = _detect_emerging_issues(records, emerging_conversations or [])
    return ThemeReport(themes=themes, emerging_issues=emerging_issues)
