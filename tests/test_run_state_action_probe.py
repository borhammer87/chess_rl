from types import SimpleNamespace

import chess
import pytest
import torch

from scripts.run_state_action_probe import TerminalTransitionTracker
from chess_rl.utils.replay_buffer import ReplayBuffer


def add_transition(buffer, *, done=True):
    state = torch.zeros((18, 8, 8))

    buffer.push(
        state=state,
        action=0,
        reward=0.0,
        next_state=state,
        done=done,
        next_legal_actions=[],
    )


def episode_result(*, truncated=False, result="1-0"):
    return SimpleNamespace(
        truncated=truncated,
        done=not truncated,
        final_info={"result": result},
        agent_color=chess.WHITE,
    )


def test_tracker_classifies_real_win_and_truncation():
    buffer = ReplayBuffer(capacity=10)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(
        1, 2, episode_result(result="1-0")
    )

    add_transition(buffer)
    tracker.record_episode(
        2, 2, episode_result(truncated=True)
    )

    assert list(tracker.get_final_labels().values()) == [
        "win",
        "truncated",
    ]


def test_tracker_ignores_evicted_transitions():
    buffer = ReplayBuffer(capacity=1)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(
        1, 2, episode_result(result="1-0")
    )

    add_transition(buffer)
    tracker.record_episode(
        2, 2, episode_result(result="0-1")
    )

    assert list(tracker.get_final_labels().values()) == [
        "loss",
    ]


def test_tracker_rejects_unlabelled_terminal_transition():
    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)

    with pytest.raises(ValueError, match="Unlabelled"):
        tracker.get_final_labels()


def test_tracker_rejects_nonterminal_last_transition():
    buffer = ReplayBuffer(capacity=2)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer, done=False)

    with pytest.raises(ValueError, match="not terminal"):
        tracker.record_episode(
            1, 1, episode_result()
        )

def test_terminal_per_categories(capsys):
    from scripts.run_state_action_probe import (
        analyze_terminal_per_categories,
    )

    buffer = ReplayBuffer(capacity=4)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(1, 2, episode_result(result="1-0"))

    add_transition(buffer)
    tracker.record_episode(2, 2, episode_result(truncated=True))

    analyze_terminal_per_categories(
        buffer,
        tracker.get_final_labels(),
    )

    output = capsys.readouterr().out

    assert "win: count=1" in output
    assert "truncated: count=1" in output
    assert "loss: count=0" in output
    assert "draw: count=0" in output
    assert "oversampling=1.000x" in output


def test_terminal_per_categories_with_unequal_priorities(capsys):
    from scripts.run_state_action_probe import (
        analyze_terminal_per_categories,
    )
    from chess_rl.training.episodes import PER_ALPHA

    buffer = ReplayBuffer(capacity=3)
    tracker = TerminalTransitionTracker(buffer)

    add_transition(buffer)
    tracker.record_episode(1, 2, episode_result(result="1-0"))

    add_transition(buffer)
    tracker.record_episode(2, 2, episode_result(truncated=True))

    add_transition(buffer, done=False)

    buffer.update_priorities(
        indices=[0, 1, 2],
        priorities=[1.0, 4.0, 9.0],
    )

    analyze_terminal_per_categories(
        buffer,
        tracker.get_final_labels(),
    )

    output = capsys.readouterr().out

    total = 1.0**PER_ALPHA + 4.0**PER_ALPHA + 9.0**PER_ALPHA
    win_share = 1.0**PER_ALPHA / total
    truncated_share = 4.0**PER_ALPHA / total

    assert (
        f"PER share={win_share:.6f}" in output
    )
    assert (
        f"PER share={truncated_share:.6f}" in output
    )
    assert (
        f"oversampling={3 * win_share:.3f}x" in output
    )
    assert (
        f"oversampling={3 * truncated_share:.3f}x" in output
    )


def test_sample_observer_preserves_original_result(monkeypatch):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
    )

    buffer = ReplayBuffer(capacity=3)
    add_transition(buffer, done=False)
    add_transition(buffer, done=True)

    original_transition = buffer.buffer[1]
    expected_indices = [1, 1]
    expected_weights = torch.tensor([0.5, 0.5])
    calls = []

    def fake_sample(*, batch_size, alpha, beta):
        calls.append((batch_size, alpha, beta))
        return (
            [original_transition, original_transition],
            expected_indices,
            expected_weights,
        )

    monkeypatch.setattr(
        buffer,
        "sample_prioritized",
        fake_sample,
    )

    observer = PrioritizedSampleObserver(buffer)

    result = observer.sample_prioritized(
        batch_size=2,
        alpha=0.6,
        beta=0.4,
    )

    assert calls == [(2, 0.6, 0.4)]
    assert result[0][0] is original_transition
    assert result[0][1] is original_transition
    assert result[1] is expected_indices
    assert result[2] is expected_weights
    assert observer.sampled_transitions == [
        original_transition,
        original_transition,
    ]


def test_sample_observer_does_not_sample_until_called(monkeypatch):
    from scripts.run_state_action_probe import (
        PrioritizedSampleObserver,
    )

    buffer = ReplayBuffer(capacity=2)
    calls = []

    def fake_sample(*, batch_size, alpha, beta):
        calls.append((batch_size, alpha, beta))
        return ([], [], torch.tensor([]))

    monkeypatch.setattr(
        buffer,
        "sample_prioritized",
        fake_sample,
    )

    observer = PrioritizedSampleObserver(buffer)

    assert calls == []
    assert observer.sampled_transitions == []

    observer.sample_prioritized(
        batch_size=1,
        alpha=0.6,
        beta=0.4,
    )

    assert len(calls) == 1
