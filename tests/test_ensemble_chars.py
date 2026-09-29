"""Tests for scripts/ensemble_chars.py (probability averaging and calibration)."""

import numpy as np
import pytest

from scripts import ensemble_chars as E

CLASSES = np.array(list("ABab"))


def _member(path, seed, proba, true, *, handedness=None, split_seed=0):
    np.savez(
        path,
        true=np.asarray(true),
        pred=np.argmax(proba, 1),
        proba=np.asarray(proba, dtype=np.float32),
        handedness=(
            np.full(len(true), -1) if handedness is None else np.asarray(handedness)
        ),
        classes=CLASSES,
        model="cnn_bilstm_attn",
        test_acc=0.0,
        val_acc=0.0,
        seed=seed,
        split_seed=split_seed,
        split="official both/indep/fold0",
    )


def test_ensemble_can_be_right_where_every_member_is_wrong_somewhere(tmp_path):
    """Averaging fixes a sample that one confident-but-wrong member misreads."""
    true = [0, 1]
    good = [[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]]
    _member(tmp_path / "m0.npz", 0, [[0.4, 0.6, 0, 0], [0.1, 0.9, 0, 0]], true)
    _member(tmp_path / "m1.npz", 1, [[0.9, 0.1, 0, 0], [0.6, 0.4, 0, 0]], true)
    _member(tmp_path / "m2.npz", 2, good, true)
    g = E.summarise("x", E.load_members(str(tmp_path / "m*.npz")))
    assert [m["all"]["accuracy"] for m in g["members"]] == [50.0, 50.0, 100.0]
    assert g["ensemble"]["all"]["accuracy"] == 100.0
    assert g["seeds"] == [0, 1, 2]


def test_members_on_different_splits_are_refused(tmp_path):
    p = [[1.0, 0, 0, 0]]
    _member(tmp_path / "m0.npz", 0, p, [0], split_seed=0)
    _member(tmp_path / "m1.npz", 1, p, [0], split_seed=1)
    with pytest.raises(SystemExit, match="split_seed"):
        E.load_members(str(tmp_path / "m*.npz"))


def test_duplicate_seeds_are_refused(tmp_path):
    p = [[1.0, 0, 0, 0]]
    _member(tmp_path / "m0.npz", 3, p, [0])
    _member(tmp_path / "m1.npz", 3, p, [0])
    with pytest.raises(SystemExit, match="duplicate seeds"):
        E.load_members(str(tmp_path / "m*.npz"))


def test_populations_are_scored_separately(tmp_path):
    proba = [[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]]
    _member(tmp_path / "m0.npz", 0, proba, [0, 0, 1], handedness=[0, 0, 1])
    g = E.summarise("x", E.load_members(str(tmp_path / "m*.npz")))
    assert g["ensemble"]["right-handed"]["accuracy"] == 100.0
    assert g["ensemble"]["left-handed"]["accuracy"] == 0.0
    assert g["ensemble"]["all"]["n"] == 3


def test_case_insensitive_accuracy_forgives_only_case():
    # true A, predicted a (case error); true B, predicted A (real error)
    proba = np.array([[0, 0, 1.0, 0], [1.0, 0, 0, 0]])
    s = E.scores(proba, np.array([0, 1]), CLASSES)
    assert s["accuracy"] == 0.0
    assert s["case_insensitive_accuracy"] == 50.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    # Four predictions at confidence 0.75, three of them right.
    proba = np.tile([0.75, 0.25, 0.0, 0.0], (4, 1))
    assert E.expected_calibration_error(proba, np.array([0, 0, 0, 1])) == 0.0


def test_ece_measures_overconfidence():
    proba = np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))
    assert E.expected_calibration_error(proba, np.array([0, 1])) == pytest.approx(0.5)


def test_case_confidence_separates_case_errors_and_abstention():
    classes = np.array(["a", "A", "b"])
    true = np.array([0, 0, 0, 2])
    proba = np.array(
        [
            [0.90, 0.05, 0.05],  # right, confident
            [0.35, 0.60, 0.05],  # case error: A for a, second choice right
            [0.30, 0.30, 0.40],  # other error: b for a, least confident
            [0.20, 0.10, 0.70],  # right
        ]
    )
    out = E.case_confidence(proba, true, classes)
    assert out["right"]["n"] == 2
    assert out["case_errors"]["n"] == 1 and out["other_errors"]["n"] == 1
    assert out["case_share_of_errors"] == 50.0
    assert out["case_errors_second_choice_right"] == 100.0
    assert out["case_errors"]["median_confidence"] == 0.60
    # At 70% coverage one prediction (the 0.40 other error) is withheld.
    a70 = next(a for a in out["abstain"] if a["coverage"] == 0.7)
    assert a70["other_errors_withheld"] == 100.0
    assert a70["case_errors_withheld"] == 0.0
    assert a70["case_share_of_kept_errors"] == 100.0
