"""Tests for scripts/score_numbers.py and the equation grouping it reads."""

import pickle

import numpy as np
import pytest

from imu2text import symbols as S
from scripts import score_numbers as N


def test_number_runs_split_at_operators_and_equations():
    # "12+3" then "45": three numbers, the last two touch but are separate
    # equations.
    true = np.array([1, 2, 10, 3, 4, 5])
    groups = np.array([0, 0, 0, 0, 1, 1])
    runs = [r.tolist() for r in N.number_runs(true, groups)]
    assert runs == [[0, 1], [3], [4, 5]]


def test_a_number_is_right_only_if_every_digit_is():
    true = np.array([1, 2, 10, 3, 4, 5])
    pred = np.array([1, 7, 10, 3, 4, 5])
    groups = np.array([0, 0, 0, 0, 1, 1])
    out = N.score(true, pred, groups)
    assert out["digit_acc"] == pytest.approx(80.0)
    assert out["operator_acc"] == 100.0
    assert out["number_acc"] == pytest.approx(66.67)
    assert out["number_acc_by_length"]["2"] == {"n": 2, "acc": 50.0}
    assert out["equation_acc"] == 50.0


def test_score_without_groups_has_no_number_fields():
    out = N.score(np.array([0, 11]), np.array([0, 12]))
    assert out["digit_acc"] == 100.0 and out["operator_acc"] == 0.0
    assert "number_acc" not in out


def _write_archive(tmp_path, index_text, n_slices):
    (tmp_path / "all_val_indices_e.txt").write_text(index_text, encoding="utf-8")
    with open(tmp_path / "all_val_gt_e.pkl", "wb") as f:
        pickle.dump([0] * n_slices, f)
    return str(tmp_path)


def test_equation_groups_number_recordings_and_skipped_indices(tmp_path):
    base = _write_archive(tmp_path, "rec/a\n0 0 2 2 2\nrec/b\n0 1\n", 7)
    assert S.load_equation_groups(base, "val").tolist() == [0, 0, 1, 1, 1, 2, 3]


def test_equation_groups_reject_a_count_mismatch(tmp_path):
    base = _write_archive(tmp_path, "rec/a\n0 0 1\n", 4)
    with pytest.raises(ValueError, match="maps 3 slices"):
        S.load_equation_groups(base, "val")


def test_equation_groups_reject_a_split_equation(tmp_path):
    base = _write_archive(tmp_path, "rec/a\n0 1 0\n", 3)
    with pytest.raises(ValueError, match="not contiguous"):
        S.load_equation_groups(base, "val")
