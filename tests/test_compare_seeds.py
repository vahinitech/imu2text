"""Tests for scripts/compare_seeds.py."""

import numpy as np
import pytest

from scripts import compare_seeds as C

CLASSES = np.array(["A", "a", "B", "b"])


def _run(pred, seed, split_seed=None):
    true = np.array([0, 1, 2, 3])
    proba = np.eye(4, dtype=np.float32)[pred]
    return {
        "true": true,
        "pred": np.array(pred),
        "proba": proba,
        "classes": CLASSES,
        "split": np.array("x"),
        "split_seed": np.array(seed if split_seed is None else split_seed),
        "val_acc": np.array(80.0),
        "train_acc": np.array(90.0),
    }


def test_paired_difference_and_case_insensitive_accuracy():
    a = {0: _run([0, 0, 2, 3], 0), 1: _run([0, 1, 3, 3], 1)}
    b = {0: _run([0, 1, 2, 3], 0), 1: _run([0, 1, 2, 3], 1)}
    res = C.compare(a, b)
    assert res["difference"]["per_seed"] == [25.0, 25.0]
    assert res["difference"]["b_better"] == 2
    assert res["rows"][0]["a"]["case_insensitive"] == 100.0  # a read as A
    assert res["b"]["vote"] == 100.0


def test_pairs_must_share_the_split():
    a = {0: _run([0, 1, 2, 3], 0), 1: _run([0, 1, 2, 3], 1)}
    b = {0: _run([0, 1, 2, 3], 0), 1: _run([0, 1, 2, 3], 1, split_seed=7)}
    with pytest.raises(SystemExit, match="split_seed"):
        C.compare(a, b)
