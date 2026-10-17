
from unittest.mock import patch
import math
import pytest
from scripts.run_capture_learning_probe import (
    run_initial_capture_probe,
)
from unittest.mock import MagicMock


def test_train_capture_batch_uses_existing_agent_update(monkeypatch):
    from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
    from chess_rl.training.capture_learning_task import (
        make_capture_positions,
        train_capture_batch,
    )

    agent = StateActionDQNAgent()
    received = []

    def fake_train_step(batch):
        received.extend(batch)
        return 0.25

    monkeypatch.setattr(agent, "train_step", fake_train_step)

    loss = train_capture_batch(agent, make_capture_positions()[:2])

    assert loss == 0.25
    assert len(received) == 4
    assert all(transition.done for transition in received)


def test_train_capture_batch_rejects_empty_positions():
    from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
    from chess_rl.training.capture_learning_task import train_capture_batch

    agent = StateActionDQNAgent()

    with pytest.raises(ValueError, match="cannot be empty"):
        train_capture_batch(agent, [])


def test_initial_capture_probe_reports_accuracy(capsys):
    with patch(
        "scripts.run_capture_learning_probe.train_capture_batch",
        return_value=0.01,
    ) as train:
        with patch(
            "scripts.run_capture_learning_probe.evaluate_capture_accuracy",
            side_effect=[0.50, 0.625, 0.875, 0.95],
        ) as evaluate:
            with patch(
                "scripts.run_capture_learning_probe.capture_q_gap_summary",
                side_effect=[(0.02, 0.24), (-0.01, 0.40)],
            ) as gaps:
                result = run_initial_capture_probe(seed=42, updates=3)

    output = capsys.readouterr().out

    assert result == 0.625
    assert train.call_count == 3
    assert evaluate.call_count == 4
    assert "Initial training accuracy: 50.0%" in output
    assert "Final training accuracy: 95.0%" in output
    assert "Training positions reserved: 300" in output
    assert "Evaluation positions: 100" in output
    assert "Untrained accuracy: 62.5%" in output
    assert "Training updates: 3" in output
    assert "Final accuracy: 87.5%" in output
    assert gaps.call_count == 2
    assert "Training mean Q gap: 0.020000" in output
    assert "Evaluation mean Q gap: -0.010000" in output
    assert "Training Q error rate: 24.0%" in output
    assert "Evaluation Q error rate: 40.0%" in output


def test_capture_probe_runs_one_training_update(capsys):
    with patch(
        "scripts.run_capture_learning_probe.train_capture_batch",
        return_value=0.0125,
    ) as train:
        with patch(
            "scripts.run_capture_learning_probe.evaluate_capture_accuracy",
            side_effect=[0.50, 0.575, 0.65, 0.75],
        ) as evaluate:
            with patch(
                "scripts.run_capture_learning_probe.capture_q_gap_summary",
                side_effect=[(0.02, 0.24), (-0.01, 0.40)],
            ) as gaps:
                run_initial_capture_probe(seed=42, updates=1)

    output = capsys.readouterr().out

    assert train.call_count == 1
    assert len(train.call_args.args[1]) == 300
    assert evaluate.call_count == 4
    assert "First training loss: 0.012500" in output
    assert "Final training loss: 0.012500" in output
    assert "Final accuracy: 65.0%" in output
    assert gaps.call_count == 2
    assert "Training mean Q gap: 0.020000" in output
    assert "Evaluation mean Q gap: -0.010000" in output
    assert "Training Q error rate: 24.0%" in output
    assert "Evaluation Q error rate: 40.0%" in output
    assert len(evaluate.call_args_list[0].args[1]) == 300
    assert len(evaluate.call_args_list[1].args[1]) == 100

def test_capture_probe_rejects_nonpositive_updates():
    with pytest.raises(ValueError, match="must be positive"):
        run_initial_capture_probe(updates=0)


def test_capture_feature_probe_rejects_invalid_updates():
    from scripts.run_capture_learning_probe import (
        run_capture_feature_probe,
    )

    with pytest.raises(ValueError, match="positive"):
        run_capture_feature_probe(updates=0)


def test_capture_feature_probe_returns_valid_accuracy(capsys):
    from scripts.run_capture_learning_probe import (
        run_capture_feature_probe,
    )

    accuracy = run_capture_feature_probe(seed=42, updates=2)

    assert 0.0 <= accuracy <= 1.0
    assert "Final evaluation accuracy:" in capsys.readouterr().out


def test_spatial_comparison_uses_both_models():
    from scripts.run_capture_learning_probe import (
        run_spatial_comparison,
    )

    results = run_spatial_comparison(
        seed=42,
        updates=1,
        train_count=4,
        evaluation_count=4,
    )

    assert set(results) == {"original", "spatial"}

    for result in results.values():
        for key in (
            "initial_train",
            "initial_eval",
            "final_train",
            "final_eval",
        ):
            assert 0.0 <= result[key] <= 1.0

        for key in (
            "first_loss",
            "final_loss",
            "train_q_gap",
            "eval_q_gap",
        ):
            assert math.isfinite(result[key])
