"""Tests for scripts/length_study.py (what --max-len cuts), on made-up data."""

import numpy as np

from scripts.length_study import FORCE, cut_summary, writing_lost


def _rec(steps, down_from, down_to):
    """A recording of ``steps`` steps with the pen down in [down_from, down_to)."""
    r = np.zeros((steps, 13))
    r[down_from:down_to, FORCE] = 1000.0
    return r


def test_cut_summary_counts_by_case_and_split():
    s = cut_summary([50, 120, 150, 90], ["A", "B", "a", "b"], [0, 1, 0, 0], 100)
    assert s["truncated"] == 2 and s["share"] == 50.0
    assert s["share_upper"] == 50.0 and s["share_lower"] == 50.0
    assert s["upper_share_of_truncated"] == 50.0
    assert s["truncated_test"] == 1
    assert round(s["steps_lost_share"], 4) == round(70 / 410 * 100, 4)
    assert s["top_classes"][0][1] == 1


def test_writing_lost_finds_letters_written_after_the_cut():
    recs = [
        _rec(80, 5, 40),  # short: never cut
        _rec(300, 250, 280),  # idle start, writing after step 100: all lost
        _rec(150, 60, 140),  # half the writing kept
        _rec(200, 0, 0),  # no pen-down step at all: not counted either way
    ]
    w = writing_lost(recs, [False, True, False, False], 100, threshold=500.0)
    assert w["truncated"] == 3
    assert w["all_writing_lost"] == 1 and w["all_writing_lost_test"] == 1
    assert w["median_writing_kept"] == 50.0
