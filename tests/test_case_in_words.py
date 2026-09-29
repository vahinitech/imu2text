"""Tests for scripts/case_in_words.py (case accuracy inside read words)."""

from scripts.case_in_words import case_in_words


def test_counts_case_on_read_words_and_twin_spellings():
    refs = ["Wie", "wie", "Haus", "Tag", "zur"]
    hyps = ["wie", "wie", "Haus", "Mg", "Zur"]
    r = case_in_words(refs, hyps)
    assert r["n"] == 5
    assert r["case_insensitive"] == 80.0  # all but Tag
    assert r["exact"] == 40.0  # wie, Haus
    assert r["case_right_in_read_words"] == 50.0  # 2 of 4
    # Letters in read words: Wie 3 + wie 3 + Haus 4 + zur 3 = 13, two wrong.
    assert round(r["case_right_chars_in_read_words"], 2) == round(11 / 13 * 100, 2)
    # "Wie"/"wie" are the only word in two spellings.
    assert r["twin_words"] == 2 and r["twin_read"] == 2 and r["twin_case_right"] == 1
    assert r["case_errors"] == 2 and r["case_errors_on_twin_words"] == 1


def test_empty_decoding_is_not_a_division_error():
    r = case_in_words(["a"], [""])
    assert r["case_insensitive"] == 0.0 and r["case_right_in_read_words"] == 0.0
