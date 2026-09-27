"""Canonical topic taxonomy, sentiment scale, and temperature bands.

Single source of truth shared by the interaction-analysis prompt, the pydantic schemas,
and the RAG knowledge base structure, so all three stay aligned.
"""
from __future__ import annotations

TOPIC_TAXONOMY: dict[str, list[str]] = {
    "Technical Issue": [
        "Connectivity", "Hardware Fault", "Login Failure", "Audio Output", "Data Access",
        "Peripheral Fault", "Website Performance", "Streaming Quality", "Device Connectivity",
        "System Instability", "Network Outage", "Content Access", "License Renewal Inquiry",
        "Session Timeout", "Software Crash", "Other",
    ],
    "Billing": [
        "Overcharge", "Subscription Cancellation", "Payment Deferral", "Payment Method Update",
        "Invoice Request", "Late Fee Dispute", "Unidentified Charge", "Price Change Inquiry",
        "Credit/Promotion Issue", "Discount Request", "Payment Rejection", "Receipt Request",
        "Tax Dispute", "Other",
    ],
    "Delivery": [
        "Late Shipment", "Address Change", "Order Modification", "Damaged Goods",
        "Wrong Product Sent", "Service Complaint", "Stalled Tracking", "Shipping Fee Dispute",
        "Other",
    ],
    "Account Management": [
        "Personal Data Change", "Password Reset", "Service Upgrade", "User Addition",
        "Account Lockout", "Subscription Pause", "Communication Preferences", "Form Correction",
        "Transfer Error", "Security Update", "Username Change", "Account Consolidation", "Other",
    ],
    "Other": ["General Inquiry", "Feedback", "Other"],
}

TOPIC_FAMILIES = list(TOPIC_TAXONOMY.keys())

SENTIMENTS = ["very_negative", "negative", "neutral", "positive", "very_positive"]

# (band, min_score, max_score)
TEMPERATURE_BANDS: list[tuple[str, int, int]] = [
    ("calm", 0, 20),
    ("concerned", 21, 40),
    ("frustrated", 41, 60),
    ("angry", 61, 80),
    ("critical", 81, 100),
]


def band_for_score(score: int) -> str:
    """Map a 0-100 temperature score to its band label."""
    score = max(0, min(100, score))
    for band, lo, hi in TEMPERATURE_BANDS:
        if lo <= score <= hi:
            return band
    return TEMPERATURE_BANDS[-1][0]


def format_taxonomy_for_prompt() -> str:
    """Render the taxonomy as an indented bullet list for the {taxonomy} prompt placeholder."""
    lines: list[str] = []
    for family, subtopics in TOPIC_TAXONOMY.items():
        lines.append(f"- {family}:")
        for sub in subtopics:
            lines.append(f"    - {family} - {sub}")
    return "\n".join(lines)
