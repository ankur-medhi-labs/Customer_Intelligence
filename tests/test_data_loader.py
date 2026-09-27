"""Unit tests for the raw dataset loader. Uses a small temp xlsx (not the real dataset) so the
test stays fast and self-contained."""
from __future__ import annotations

import pandas as pd

from customer_intelligence.data_loader import load_raw_conversations


def _write_sample_xlsx(path):
    df = pd.DataFrame([
        {
            "ID": 1,
            "Input (Informal Chat & Topic)": (
                'Chat: "My internet keeps dropping!" Topic: Technical Issue - Connectivity'
            ),
            "Expected Output (Formal, Summarized Resolution Note)": (
                "Resolution: Customer reported intermittent connectivity issues. Resolved via router reset."
            ),
        },
        {
            "ID": 2,
            "Input (Informal Chat & Topic)": (
                'Chat: "I was charged twice, please fix it." Topic: Billing - Overcharge'
            ),
            "Expected Output (Formal, Summarized Resolution Note)": (
                "Resolution: Duplicate charge identified and refunded."
            ),
        },
    ])
    df.to_excel(path, index=False)


def test_load_raw_conversations_splits_chat_and_topic(tmp_path):
    xlsx_path = tmp_path / "sample.xlsx"
    _write_sample_xlsx(xlsx_path)

    conversations = load_raw_conversations(xlsx_path)

    assert len(conversations) == 2
    first = conversations[0]
    assert first.raw_text == "My internet keeps dropping!"
    assert first.given_topic_label == "Technical Issue - Connectivity"
    resolution = conversations[1].expected_resolution
    assert resolution is not None and "refunded" in resolution.lower()


def test_load_raw_conversations_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.xlsx"
    try:
        load_raw_conversations(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
