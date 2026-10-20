"""Atomic publication of complete text files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_text"]


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a file with complete UTF-8 text through a temporary sibling.

    The parent directory must already exist. Filesystem errors propagate and
    an unpublished temporary file is removed. Atomic replacement does not
    promise fsync-backed crash durability.
    """

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
