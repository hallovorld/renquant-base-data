"""Tests for the AC-1 (x) sanity-contract migration precondition scan."""
import json

from renquant_base_data.sidecar_sanity_contract_scan import (
    DEFAULT_SURFACES,
    main,
    scan_contracts,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_clean_tree_precondition_holds(tmp_path):
    _write(
        tmp_path / "prod" / "scorer.json",
        {"feature_cols": ["KMID", "roe"], "label_col": "fwd_60d_excess"},
    )
    _write(tmp_path / "prod" / "calibration.json", {"method": "isotonic"})
    report = scan_contracts([str(tmp_path)])
    assert report["precondition_holds"] is True
    assert report["n_contracts"] == 1
    assert report["n_no_feature_cols"] == 1
    assert report["n_violations"] == 0


def test_sentiment_naming_contract_fails_closed(tmp_path):
    _write(
        tmp_path / "prod" / "scorer.json",
        {
            "feature_cols": ["KMID", "mean_sentiment", "n_articles_log"],
            "label_col": "fwd_60d_excess",
        },
    )
    report = scan_contracts([str(tmp_path)])
    assert report["precondition_holds"] is False
    assert report["n_violations"] == 1
    (rec,) = report["violations"].values()
    assert rec["sentiment_named"] == ["mean_sentiment", "n_articles_log"]
    # No training_contract.dataset recorded -> the wf_gate sanity run would
    # take the SIDECAR path for this artifact (the flip-exposed population).
    assert report["sidecar_exposed_violations"] == list(report["violations"])


def test_dataset_recorded_violation_still_fails_but_is_split_out(tmp_path):
    _write(
        tmp_path / "shadow" / "scorer.json",
        {
            "feature_cols": ["sentiment_pos_share"],
            "training_contract": {"dataset": "data/transformer_v4_wl200_clean.parquet"},
        },
    )
    report = scan_contracts([str(tmp_path)])
    assert report["precondition_holds"] is False  # strict per RFC AC-1 (x)
    assert report["sidecar_exposed_violations"] == []  # but not sidecar-path exposed


def test_unparseable_payload_fails_closed(tmp_path):
    bad = tmp_path / "prod" / "corrupt.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    report = scan_contracts([str(tmp_path)])
    assert report["precondition_holds"] is False
    assert report["unparseable"] == [str(bad)]


def test_manifest_retrains_are_chased_to_ckpt_metadata(tmp_path):
    ckpt = tmp_path / "artifacts" / "model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"\x00")
    _write(
        tmp_path / "artifacts" / "model.pt.metadata.json",
        {
            "feature_cols": ["KMID", "mean_sentiment"],
            "training_contract": {"dataset": "data/corpus.parquet"},
        },
    )
    _write(
        tmp_path / "manifest.json",
        {"cadence_days": 147, "retrains": [{"artifact_uri": str(ckpt)}]},
    )
    report = scan_contracts([str(tmp_path / "manifest.json")])
    assert report["n_manifests"] == 1
    assert report["n_violations"] == 1
    assert report["precondition_holds"] is False


def test_manifest_entry_without_metadata_surface_fails_closed(tmp_path):
    _write(
        tmp_path / "manifest.json",
        {"retrains": [{"artifact_uri": str(tmp_path / "missing" / "model.pt")}]},
    )
    report = scan_contracts([str(tmp_path / "manifest.json")])
    assert report["precondition_holds"] is False
    assert len(report["unresolved_manifest_entries"]) == 1


def test_cli_exit_codes_and_json_out(tmp_path, capsys):
    _write(tmp_path / "clean" / "scorer.json", {"feature_cols": ["KMID"]})
    out = tmp_path / "report.json"
    rc = main(["--root", str(tmp_path / "clean"), "--json-out", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["precondition_holds"] is True
    _write(tmp_path / "dirty" / "scorer.json", {"feature_cols": ["n_articles_log"]})
    rc = main(["--root", str(tmp_path / "dirty")])
    assert rc == 1
    captured = capsys.readouterr().out
    assert "PRECONDITION FAILS" in captured
    assert "VIOLATION" in captured


def test_default_surfaces_exclude_archived_diagnostics():
    joined = " ".join(DEFAULT_SURFACES)
    assert "diagnostics" not in joined
    assert "modal_sweep" not in joined
