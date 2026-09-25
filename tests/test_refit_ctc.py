"""The final refit budget must come only from validation."""

import pytest

pytest.importorskip("tensorflow")

# pylint: disable=wrong-import-position
from scripts.refit_ctc import selected_budget, executable_hash


def test_test_metrics_cannot_change_the_refit_budget():
    report = {
        "validation_split": "writer",
        "deterministic": True,
        "selected_epoch": 2,
        "epochs_run": 3,
        "history": {"val_loss": [3, 1, 2]},
        "cer": 0.9,
    }
    assert selected_budget(report) == 2
    report["cer"] = 0.01
    assert selected_budget(report) == 2
    report["selected_epoch"] = 3
    with pytest.raises(ValueError, match="minimize validation"):
        selected_budget(report)


def test_source_fingerprint_allows_docs_but_detects_computation_changes():
    first = '"""Old description."""\ndef value():\n    return 1\n'
    documentation = '"""New description."""\n# Note\ndef value():\n    return 1\n'
    changed = '"""Old description."""\ndef value():\n    return 2\n'
    assert executable_hash(first) == executable_hash(documentation)
    assert executable_hash(first) != executable_hash(changed)


def test_library_fingerprint_omits_cli_but_keeps_training_functions():
    source = "def train():\n    return 1\ndef main():\n    return 2\n"
    cli_change = source.replace("return 2", "return 3")
    training_change = source.replace("return 1", "return 3")
    assert executable_hash(source) != executable_hash(cli_change)
    assert executable_hash(source, True) == executable_hash(cli_change, True)
    assert executable_hash(source, True) != executable_hash(training_change, True)
