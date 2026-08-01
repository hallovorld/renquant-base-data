"""Validation for the momentum-prereg frozen-input snapshot manifest.

Required by the codex review of base-data#59: the manifest must be self-checkable —
schema/version, the 292-name file set, and the combined OHLCV digest RECOMPUTED from
the listed per-file entries (not trusted as stated). Pure manifest-content tests: no
filesystem access beyond the manifest itself, so they run on any machine including CI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = (Path(__file__).parent.parent
            / "manifests/momentum-prereg-inputs-20260801.json")

# The frozen §2 pins from renquant-model#164 — restated here so a manifest edit that
# drifts an identity digest fails loudly against an independent copy.
PREREG_PANEL = "55811f6387e67411fe11a20eb1d5d929086c5a9dc2675496f3d8592fed2c0dba"
PREREG_SECTOR = "ec26bb1efcf8463519366478ae72c933f93c9d110d65f8af1634e2fcbb578d3b"
PREREG_OHLCV = "4d4638a9f0d69f940fb36a73c28e92883d51b686ab032aebedf559c174c2c1d0"


def _load() -> dict:
    return json.loads(MANIFEST.read_text())


def test_schema_and_version():
    m = _load()
    assert m["dataset_id"] == "momentum-prereg-inputs-20260801"
    assert m["schema_version"] == "frozen-input-snapshot-v1"
    assert m["immutable"] is True
    for key in ("identity", "resolver", "resolution_contract",
                "combined_ohlcv_digest", "provenance", "files"):
        assert key in m, key


def test_file_set_is_panel_plus_sectors_plus_292_ohlcv_names():
    m = _load()
    files = m["files"]
    ohlcv = [k for k in files if k.startswith("ohlcv/")]
    assert "panel.parquet" in files and "ticker_sectors.json" in files
    assert len(files) == 294
    assert len(ohlcv) == 292
    names = {k.split("/")[1] for k in ohlcv}
    assert len(names) == 292, "duplicate ticker directories"
    assert all(k == f"ohlcv/{k.split('/')[1]}/1d.parquet" for k in ohlcv)
    assert "SPY" in names, "the benchmark file must be inside the pinned universe"
    for entry in files.values():
        assert set(entry) == {"sha256", "bytes"}
        assert len(entry["sha256"]) == 64 and int(entry["bytes"]) > 0


def test_combined_ohlcv_digest_recomputes_from_the_listed_entries():
    """The §2 arithmetic, re-derived from the per-file entries rather than trusted:
    sha256 over lines '<ticker>:<file sha256>\n' for the sorted ticker set."""
    m = _load()
    ohlcv = {k.split("/")[1]: v["sha256"]
             for k, v in m["files"].items() if k.startswith("ohlcv/")}
    h = hashlib.sha256()
    for t in sorted(ohlcv):
        h.update(f"{t}:{ohlcv[t]}\n".encode())
    assert h.hexdigest() == m["combined_ohlcv_digest"]["value"]
    assert h.hexdigest() == m["identity"]["combined_ohlcv_sha256"]
    assert h.hexdigest() == PREREG_OHLCV


def test_identity_block_matches_files_and_the_frozen_prereg_pins():
    m = _load()
    ident = m["identity"]
    assert ident["panel_sha256"] == m["files"]["panel.parquet"]["sha256"] == PREREG_PANEL
    assert (ident["sector_sha256"]
            == m["files"]["ticker_sectors.json"]["sha256"] == PREREG_SECTOR)


def test_no_path_is_part_of_the_identity():
    """Paths live only under resolver.candidate_roots (non-normative cache hints);
    the identity block must contain digests only, so relocation never touches it."""
    m = _load()
    ident_text = json.dumps(m["identity"])
    assert "/Users/" not in ident_text and "path" not in m["identity"]
    roots = m["resolver"]["candidate_roots"]
    assert isinstance(roots, list) and all("path" in r for r in roots)
    assert m["resolver"]["scheme"] == "content-addressed-v1"
