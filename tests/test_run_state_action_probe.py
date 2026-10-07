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