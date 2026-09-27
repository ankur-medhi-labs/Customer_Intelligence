"""Load and normalize the raw customer chat dataset.

The source spreadsheet has 3 columns: ID, "Input (Informal Chat & Topic)" (which embeds both
the informal chat text and a topic label in one cell, e.g. `Chat: "..." Topic: Billing - Overcharge`),
and "Expected Output (Formal, Summarized Resolution Note)". Both extra fields are kept only as
reference/gold data for evaluation - never fed into the analysis prompt as ground truth.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import settings
from .logging_setup import get_logger
from .schemas import Conversation

logger = get_logger(__name__)

_INPUT_PATTERN = re.compile(r'Chat:\s*"(.*)"\s*Topic:\s*(.+)', re.DOTALL)
_RESOLUTION_PATTERN = re.compile(r'^\s*Resolution:\s*(.*)', re.DOTALL)

# Candidate header keywords used to auto-detect columns if the exact headers differ.
_ID_HINTS = ["id"]
_INPUT_HINTS = ["input", "chat", "conversation", "question", "message"]
_OUTPUT_HINTS = ["expected output", "resolution", "answer", "output"]


def _find_column(columns: list[str], hints: list[str]) -> str | None:
    lower = {c: c.lower() for c in columns}
    for hint in hints:
        for col, low in lower.items():
            if hint in low:
                return col
    return None


def _split_input_cell(raw: str) -> tuple[str, str | None]:
    """Split a combined "Chat: ... Topic: ..." cell into (chat_text, topic_label)."""
    match = _INPUT_PATTERN.search(raw)
    if not match:
        return raw.strip(), None
    return match.group(1).strip(), match.group(2).strip()


def _clean_resolution(raw: str | float | None) -> str | None:
    if raw is None or (isinstance(raw, float)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    match = _RESOLUTION_PATTERN.match(text)
    return match.group(1).strip() if match else text


def load_raw_conversations(path: str | Path | None = None) -> list[Conversation]:
    """Read the raw xlsx dataset and normalize it into `Conversation` records."""
    xlsx_path = Path(path) if path else Path(settings.raw_dataset_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Dataset not found at {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    columns = list(df.columns)

    id_col = _find_column(columns, _ID_HINTS) or columns[0]
    input_col = _find_column(columns, _INPUT_HINTS)
    output_col = _find_column(columns, _OUTPUT_HINTS)
    if input_col is None:
        raise ValueError(f"Could not detect the chat/input column among {columns}")
    logger.info(
        "Detected columns: id=%r input=%r output=%r", id_col, input_col, output_col
    )

    conversations: list[Conversation] = []
    for row_idx, row in df.iterrows():
        raw_input = str(row[input_col])
        chat_text, topic_label = _split_input_cell(raw_input)
        resolution = _clean_resolution(row[output_col]) if output_col else None
        conv_id = str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else str(row_idx)
        conversations.append(Conversation(
            id=conv_id,
            raw_text=chat_text,
            given_topic_label=topic_label,
            expected_resolution=resolution,
            source_row=int(row_idx),  # type: ignore[arg-type] - pandas index is always int here
        ))

    logger.info("Loaded %d conversations from %s", len(conversations), xlsx_path)
    return conversations


def save_processed(conversations: list[Conversation], out_path: str | Path | None = None) -> Path:
    """Persist normalized conversations as JSONL for reuse without re-parsing the xlsx."""
    out_path = Path(out_path) if out_path else Path(settings.processed_dir) / "conversations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")
    logger.info("Saved %d conversations -> %s", len(conversations), out_path)
    return out_path


def load_processed(path: str | Path | None = None) -> list[Conversation]:
    """Load previously-saved normalized conversations from JSONL."""
    in_path = Path(path) if path else Path(settings.processed_dir) / "conversations.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Processed conversations not found at {in_path}")
    conversations: list[Conversation] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation(**json.loads(line)))
    return conversations


def load_synthetic(path: str | Path | None = None) -> list[Conversation]:
    """Load hand-authored synthetic conversations (e.g. for emerging-issue demonstration)."""
    in_path = Path(path) if path else Path(settings.synthetic_dir) / "emerging_issue.jsonl"
    if not in_path.exists():
        return []
    conversations: list[Conversation] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation(**json.loads(line)))
    return conversations


def get_or_build_processed() -> list[Conversation]:
    """Load processed conversations, building them from the raw xlsx on first run."""
    processed_path = Path(settings.processed_dir) / "conversations.jsonl"
    if processed_path.exists():
        return load_processed(processed_path)
    conversations = load_raw_conversations()
    save_processed(conversations, processed_path)
    return conversations
