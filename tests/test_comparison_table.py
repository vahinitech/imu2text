"""The table runner must consume the validation-selected CLI result."""

from types import SimpleNamespace

from scripts import make_comparison_table as comparison


def test_table_reads_validation_selected_result(monkeypatch):
    result = SimpleNamespace(
        returncode=0,
        stderr="",
        stdout="Selected by validation: cnn_bilstm @ 70.25% test (52-class)\n",
    )
    monkeypatch.setattr(comparison.subprocess, "run", lambda *args, **kwargs: result)
    assert comparison.run_cell("data", "both", "indep", 0, 1, 0, [], True) == 70.25
