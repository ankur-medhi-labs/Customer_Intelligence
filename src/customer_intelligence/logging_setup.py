"""Central logging configuration shared by the CLI and library code."""
from __future__ import annotations

import logging
from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: str = "", force: bool = False) -> None:
    """Configure the root logger with a Rich console handler and optional file handler."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
