"""SIP feature-table tests use synthetic rows; optional source-file audit is separate."""

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu2text import sip, sip_data


@pytest.fixture
def frame():
    rng = np.random.default_rng(24)
    values = rng.normal(size=(100, len(sip_data.COLUMNS)))
    result = pd.DataFrame(values, columns=sip_data.COLUMNS)
    result["year"] = np.tile(np.arange(1, 6), 20)
    result["Z"] = np.tile([-2.0, 0.0], 50)
    result["ExecutionTime"] = 60.0
    return result


def test_schema_keeps_target_and_context_out_of_predictors():
    assert len(sip_data.COLUMNS) == 189
    assert len(sip_data.FEATURES) == 178
    assert len(set(sip_data.COLUMNS)) == 189
    assert not set(sip_data.FEATURES) & {"Z", "year", "sex", "age", "ExecutionTime"}


def test_validation_rejects_bad_schema_and_targets(frame):
    with pytest.raises(ValueError, match="189-column"):
        sip_data.validate_frame(frame.drop(columns=["Z"]))
    for field, value in (("year", 1.5), ("year", np.nan), ("StrokeNum", np.inf)):
        changed = frame.copy()
        changed[field] = changed[field].astype(float)
        changed.loc[0, field] = value
        with pytest.raises(ValueError):
            sip_data.validate_frame(changed)


def test_descriptive_report_has_aggregates_not_rows(frame):
    frame.loc[0, "StrokeNum"] = np.nan
    result = sip_data.describe(frame)
    assert result["rows"] == 100
    assert result["low_speed_rows"] == 50
    assert result["missing_by_column"]["StrokeNum"] == 1
    assert result["subject_ids_available"] is False
    json.dumps(result, allow_nan=False)


def test_no_baseline_without_identity_decision(frame):
    with pytest.raises(ValueError, match="Subject IDs are absent"):
        sip.baseline(frame, "UNO_B")


def test_row_baseline_is_deterministic_and_labelled(frame):
    first, _ = sip.baseline(frame, "UNO_B", exploratory=True)
    second, _ = sip.baseline(frame, "UNO_B", exploratory=True)
    assert first == second
    assert first["train_rows"] == 80
    assert first["test_rows"] == 20
    assert "row" in first["split"]
    assert any("not writer-independent" in message for message in first["warnings"])
    assert first["most_frequent_dummy"]["recall"] == 0
    json.dumps(first, allow_nan=False)


def test_preprocessing_is_fitted_on_training_rows_only(frame):
    labels = (frame["Z"].to_numpy() <= -1.5).astype(int)
    train, test = sip.partition(labels, None, 0, True)
    frame.loc[test, "StrokeNum"] = 1e6
    frame.loc[train[0], "StrokeNum"] = np.nan
    _, model = sip.baseline(frame, "UNO_B", exploratory=True)
    position = sip_data.FEATURES.index("StrokeNum")
    expected = frame.iloc[train]["StrokeNum"].median()
    assert model.named_steps["simpleimputer"].statistics_[position] == pytest.approx(
        expected
    )
    assert abs(model.named_steps["standardscaler"].mean_[position]) < 10


def test_group_split_is_disjoint_and_cursive_exclusion_is_explicit(frame):
    groups = np.repeat(np.arange(50), 2)
    labels = (frame["Z"].to_numpy() <= -1.5).astype(int)
    train, test = sip.partition(labels, groups, 0, False)
    assert set(groups[train]).isdisjoint(groups[test])
    report, _ = sip.baseline(frame, "UNO_C", groups=groups)
    assert report["excluded_first_grade_cursive_rows"] == 20
    assert report["train_rows"] + report["test_rows"] == 80
    assert report["split"] == "subject-grouped 80/20"


def test_baseline_rejects_duplicate_feature_records_across_split(frame):
    frame.loc[:, sip_data.FEATURES] = 1.0
    with pytest.raises(ValueError, match="Identical feature rows"):
        sip.baseline(frame, "UNO_B", exploratory=True)


def test_partition_rejects_single_class():
    with pytest.raises(ValueError, match="Both classes"):
        sip.partition(np.zeros(20, dtype=int), None, 0, True)


def test_group_mapping_requires_complete_rows_and_nonblank_ids(tmp_path):
    path = tmp_path / "groups.csv"
    path.write_text("row_index,subject_id\n1,b\n0,a\n")
    assert sip.read_groups(path, 2).tolist() == ["a", "b"]
    path.write_text("row_index,subject_id\n0,a\n0,b\n")
    with pytest.raises(ValueError, match="exactly once"):
        sip.read_groups(path, 2)
    path.write_text("row_index,subject_id\n0,a\n1,\n")
    with pytest.raises(ValueError, match="blank"):
        sip.read_groups(path, 2)


def test_download_is_checksum_checked_and_keeps_provenance(tmp_path, monkeypatch):
    payload = b"synthetic test bytes"
    checksum = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(sip_data, "FILES", {"UNO_B.xlsx": (len(payload), checksum)})
    monkeypatch.setattr(
        sip_data.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(payload)
    )
    report = sip_data.download(tmp_path)
    assert report["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["license"].endswith("/by/4.0/")
    assert json.loads((tmp_path / "provenance.json").read_text()) == report
    (tmp_path / "UNO_B.xlsx").write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="checksum"):
        sip_data.download(tmp_path)


def test_download_failure_does_not_publish_bad_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sip_data, "FILES", {"UNO_B.xlsx": (10, "unused")})
    monkeypatch.setattr(
        sip_data.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="size"):
        sip_data.download(tmp_path)
    assert not (tmp_path / "UNO_B.xlsx").exists()
    assert not (tmp_path / "provenance.json").exists()


def test_workbook_loader_and_cli_end_to_end(tmp_path, frame, monkeypatch):
    path = tmp_path / "UNO_B.xlsx"
    frame.to_excel(path, index=False, sheet_name="Sheet1")
    payload = path.read_bytes()
    monkeypatch.setattr(
        sip_data,
        "FILES",
        {
            path.name: (
                len(payload),
                hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            )
        },
    )
    actual = sip_data.load_table(path)
    assert actual.shape == (100, 189)
    output = tmp_path / "audit.json"
    sip.main(["inspect", str(path), "--output", str(output)])
    assert json.loads(output.read_text())["rows"] == 100
    sip.main(
        ["baseline", str(path), "--output", str(output), "--exploratory-row-split"]
    )
    assert json.loads(output.read_text())["feature_count"] == 178


@pytest.mark.skipif(not os.environ.get("SIP_DATA_DIR"), reason="SIP_DATA_DIR not set")
def test_actual_publisher_workbooks_match_the_loader():
    directory = Path(os.environ["SIP_DATA_DIR"])
    for name in sip_data.FILES:
        if name.endswith(".xlsx"):
            actual = sip_data.load_table(directory / name)
            assert actual.shape[1] == 189
            assert len(actual) > 100
            assert sip_data.describe(actual)["subject_ids_available"] is False


def test_missing_targets_are_counted_and_never_imputed(frame):
    frame.loc[2, "Z"] = np.nan
    assert sip_data.describe(frame)["missing_target_rows"] == 1
    report, _ = sip.baseline(frame, "UNO_B", exploratory=True)
    assert report["excluded_missing_target_rows"] == 1
    assert report["train_rows"] + report["test_rows"] == 99


def test_nonconvergence_does_not_produce_a_score(frame, monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    def unconverged(*_args, **_kwargs):
        warnings.warn("did not converge", ConvergenceWarning)

    monkeypatch.setattr(sip.LogisticRegression, "fit", unconverged)
    with pytest.raises(ConvergenceWarning):
        sip.baseline(frame, "UNO_B", exploratory=True)


def test_cli_cannot_overwrite_input_via_symlink(tmp_path):
    source = tmp_path / "UNO_B.xlsx"
    source.write_bytes(b"original input")
    output = tmp_path / "output.json"
    output.symlink_to(source)
    with pytest.raises(SystemExit):
        sip.main(["inspect", str(source), "--output", str(output)])
    assert source.read_bytes() == b"original input"
