
from unittest.mock import patch

from scripts.run_capture_learning_probe import (
    run_initial_capture_probe,
)


def test_initial_capture_probe_reports_accuracy(capsys):
    with patch(
        "scripts.run_capture_learning_probe.evaluate_capture_accuracy",
        return_value=0.625,
    ) as evaluate:
        result = run_initial_capture_probe(seed=42)

    output = capsys.readouterr().out

    assert result == 0.625
    assert evaluate.call_count == 1
    assert "Training positions reserved: 100" in output
    assert "Evaluation positions: 40" in output
    assert "Untrained accuracy: 62.5%" in output
