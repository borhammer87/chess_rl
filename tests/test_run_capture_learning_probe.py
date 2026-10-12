
from unittest.mock import patch
import pytest
from scripts.run_capture_learning_probe import (
    run_initial_capture_probe,
)



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
            result = run_initial_capture_probe(seed=42, updates=3)

    output = capsys.readouterr().out

    assert result == 0.625
    assert train.call_count == 3
    assert evaluate.call_count == 4
    assert "Initial training accuracy: 50.0%" in output
    assert "Final training accuracy: 95.0%" in output
    assert "Training positions reserved: 100" in output
    assert "Evaluation positions: 40" in output
    assert "Untrained accuracy: 62.5%" in output
    assert "Training updates: 3" in output
    assert "Final accuracy: 87.5%" in output


def test_capture_probe_runs_one_training_update(capsys):
    with patch(
        "scripts.run_capture_learning_probe.train_capture_batch",
        return_value=0.0125,
    ) as train:
        with patch(
            "scripts.run_capture_learning_probe.evaluate_capture_accuracy",
            side_effect=[0.50, 0.575, 0.65, 0.75],
        ) as evaluate:
            run_initial_capture_probe(seed=42, updates=1)

    output = capsys.readouterr().out

    assert train.call_count == 1
    assert len(train.call_args.args[1]) == 100
    assert evaluate.call_count == 4
    assert "First training loss: 0.012500" in output
    assert "Final training loss: 0.012500" in output
    assert "Final accuracy: 65.0%" in output


def test_capture_probe_rejects_nonpositive_updates():
    with pytest.raises(ValueError, match="must be positive"):
        run_initial_capture_probe(updates=0)
