"""Independent checks of benchmark metric and interval reporting."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

# pylint: disable=wrong-import-position
from scripts.summarize_ctc import (
    metrics,
    writer_bootstrap,
    decoder_changes,
    audit_source,
)
from scripts.refit_ctc import file_hash


def test_saved_prediction_metrics_use_edit_counts_and_exact_word_matches():
    measured = metrics(["ab", "cd"], ["ab", "c"])
    assert measured == {"cer": 0.25, "wer": 0.5, "word_accuracy": 0.5}
    with pytest.raises(ValueError, match="paired"):
        metrics(["a", "b"], ["a"])


def test_writer_interval_has_positive_sign_for_lower_error():
    result = writer_bootstrap(
        ["aa", "bbbb"], ["", ""], ["aa", "bbbb"], np.array([0, 1]), repeats=100
    )
    assert result["cer_reduction_pp_95_interval"] == [100.0, 100.0]
    assert result["test_writers"] == 2


def test_decoder_report_counts_recoveries_and_spoiled_words_separately():
    result = decoder_changes(["ab", "cd", "ef"], ["a", "cd", "e"], ["ab", "ce", "ef"])
    assert result == {
        "gained_exact_words": 2,
        "lost_exact_words": 1,
        "greedy_empty_decodes": 0,
        "lexicon_empty_decodes": 0,
        "new_empty_decodes": 0,
        "character_edit_change": -1,
    }
    empty = decoder_changes(["abcd"], ["abc"], [""])
    assert empty["new_empty_decodes"] == 1
    assert empty["character_edit_change"] == 3


def test_source_audit_preserves_runtime_hash_and_rejects_training_changes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "imu2text" / "seq2seq.py"
    source.parent.mkdir()
    original = "def train():\n    return 1\ndef main():\n    return 2\n"
    source.write_text(original)
    runtime_hash = file_hash(source)
    report = {"source_sha256": {"imu2text/seq2seq.py": runtime_hash}}
    result = tmp_path / "run.json"
    audit_source(report, result)
    source.write_text(original.replace("return 2", "return 3"))
    audit_source(report, result)
    assert report["source_sha256"]["imu2text/seq2seq.py"] == runtime_hash
    source.write_text(original.replace("return 1", "return 3"))
    with pytest.raises(ValueError, match="computation changed"):
        audit_source(report, result)
