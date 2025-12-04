from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a configured logger.

    The logger prints messages with level INFO and above,
    using a concise format that includes time, level and logger name.
    """

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def append_metrics_csv(path: Path, row: Mapping[str, Any]) -> None:
    """Append one metrics row to CSV, writing header on first use.

    Ensures the file/parent directory exist and flushes after each write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.is_file()

    # newline="" to avoid blank lines on Windows; utf-8 for safety.
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))
        f.flush()
