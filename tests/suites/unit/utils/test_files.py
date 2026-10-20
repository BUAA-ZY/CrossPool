from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from xpool.utils import files


@pytest.mark.parametrize("replacement_fails", (False, True))
def test_atomic_text_publication_preserves_complete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_fails: bool
) -> None:
    path = tmp_path / "result.txt"
    path.write_text("previous", encoding="utf-8")
    content = "complete UTF-8 text: λ\n"

    def replace(source: Path, destination: Path) -> None:
        assert destination == path
        assert path.read_text(encoding="utf-8") == "previous"
        assert source.read_text(encoding="utf-8") == content
        if replacement_fails:
            raise OSError("publication denied")
        os.replace(source, destination)

    monkeypatch.setattr(files, "os", SimpleNamespace(replace=replace))
    if replacement_fails:
        with pytest.raises(OSError, match="publication denied"):
            files.atomic_write_text(path, content)
        assert path.read_text(encoding="utf-8") == "previous"
    else:
        files.atomic_write_text(path, content)
        assert path.read_text(encoding="utf-8") == content
    assert set(tmp_path.iterdir()) == {path}
