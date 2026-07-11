"""Headless tests for the capture app core (no display / no tkinter needed)."""
import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

spec = importlib.util.spec_from_file_location(
    "capture_app", os.path.join(os.path.dirname(__file__), "..", "tools",
                                "capture_app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def _simulated_sample(char="S", seed=3):
    """Full recording for one character via the simulator (rest + stroke + rest)."""
    pen = app.SimulatedPen(seed=seed)
    rest = [pen.bias + pen.rng.normal(0, 0.01, app.N_CHANNELS)
            for _ in range(2 * app.FS)]
    pen.begin_stroke(char)
    stroke = list(pen.queue)
    tail = [pen.bias + pen.rng.normal(0, 0.01, app.N_CHANNELS)
            for _ in range(app.FS // 2)]
    rows = np.asarray(rest + stroke + tail)
    counters = np.arange(len(rows))
    return pen, np.asarray(rest), rows, counters


def _calib_from_rest(rest):
    return dict(force_mean=float(rest[:, app.FORCE].mean()),
                force_std=float(rest[:, app.FORCE].std()),
                bias=[float(v) for v in rest.mean(axis=0)],
                accel_std=[float(v) for v in rest[:, 1:4].std(axis=0)])


def _shape_correlation(char, seed):
    """Correlation between the glyph path and its reconstruction from accel."""
    pen, rest, rows, _ = _simulated_sample(char, seed)
    calib = _calib_from_rest(rest)
    down = app.pen_down_mask(rows[:, app.FORCE], calib["force_mean"],
                             calib["force_std"])
    first, last = np.flatnonzero(down)[[0, -1]]
    lo = max(first - 10, 0)
    seg, seg_mask = rows[lo: last + 10], down[lo: last + 10]
    x, y, _ = app.reconstruct_xy(seg, bias=np.asarray(calib["bias"]),
                                 mask=seg_mask)
    ref, _ = pen.glyph_path(char, len(x))
    # Procrustes similarity: PCA leaves an arbitrary in-plane rotation /
    # reflection, which is irrelevant to shape fidelity — align optimally,
    # then measure. 1.0 = perfect shape recovery.
    A = np.c_[x, y] - np.c_[x, y].mean(axis=0)
    B = ref - ref.mean(axis=0)
    A /= np.linalg.norm(A) + 1e-12
    B /= np.linalg.norm(B) + 1e-12
    u, _, vt = np.linalg.svd(A.T @ B)
    return float(((A @ (u @ vt)) * B).sum())


def test_reconstruction_recovers_glyph_shape():
    """The on-screen stroke must resemble what was written (Phase B, in code)."""
    scores = [_shape_correlation(c, s) for s, c in enumerate("SLoZ3AgWk8")]
    assert np.mean(scores) > 0.9, scores
    assert min(scores) > 0.75, scores


def test_pen_down_mask_and_qc_accept():
    pen, rest, rows, counters = _simulated_sample("A")
    calib = _calib_from_rest(rest)
    down = app.pen_down_mask(rows[:, app.FORCE], calib["force_mean"],
                             calib["force_std"])
    assert 0.05 < down.mean() < 0.6          # stroke, surrounded by rest
    first, last = np.flatnonzero(down)[[0, -1]]
    seg, cnt = rows[first:last + 10], counters[first:last + 10]
    ok, reason = app.qc_check(seg, cnt, calib)
    assert ok, reason


def test_qc_rejects_bad_samples():
    pen, rest, rows, counters = _simulated_sample("A")
    calib = _calib_from_rest(rest)
    ok, _ = app.qc_check(rows[:20], counters[:20], calib)      # too short
    assert not ok
    gap = counters.copy()
    gap[100:] += 5                                             # dropped packets
    ok, _ = app.qc_check(rows, gap, calib)
    assert not ok
    ok, _ = app.qc_check(rest[: app.FS], np.arange(app.FS), calib)  # no writing
    assert not ok


def test_session_storage_and_resume(tmp_path):
    prompts = tmp_path / "prompts.csv"
    with open(prompts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["box", "label"])
        w.writerows([(1, "a"), (2, "B"), (3, "7")])
    sess = app.Session(str(tmp_path / "raw"), "school01", "S0001", "session_01",
                       str(prompts), pen_serial="SIM")
    n_practice = len(app.PRACTICE)
    assert len(sess.pending()) == 3 + n_practice
    rows = np.zeros((100, 2 + app.N_CHANNELS))
    rows[:, 1] = np.arange(100)
    for box, label in list(sess.pending())[: n_practice + 2]:
        sess.accept(box, label, rows, redo_count=0,
                    qc="practice" if box == "P" else "ok")
    # a fresh Session over the same directory resumes at the right box
    sess2 = app.Session(str(tmp_path / "raw"), "school01", "S0001", "session_01",
                        str(prompts), pen_serial="SIM")
    assert [b for b, _ in sess2.pending()] == [3]
    d = tmp_path / "raw" / "school01" / "S0001" / "session_01"
    assert (d / "rec_00001.csv").exists() and (d / "meta.json").exists()
