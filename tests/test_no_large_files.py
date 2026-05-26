from __future__ import annotations

from pathlib import Path


def test_repo_does_not_track_large_data_extensions() -> None:
    root = Path(__file__).parents[1]
    forbidden = {".parquet", ".zip", ".db", ".pt", ".pkl"}
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if ".git" not in path.parts and path.is_file() and path.suffix in forbidden
    ]
    assert offenders == []
