from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.utils.board_encoder import BOARD_CHANNELS
import random

def test_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=10)

    assert len(buffer) == 0

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_push_adds_transition():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    next_state = torch.ones((BOARD_CHANNELS, 8, 8))

    buffer.push(
        state=state,
        action=123,
        reward=1.0,
        next_state=next_state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    assert len(buffer) == 1

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    for i in range(5):
        buffer.push(
            state=state,
            action=i,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    assert len(buffer) == 3

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_sample_returns_correct_batch_size():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    for i in range(5):
        buffer.push(
            state=state,
            action=i,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    batch = buffer.sample(batch_size=3)

    assert len(batch) == 3


import torch

from chess_rl.utils.replay_buffer import ReplayBuffer, Transition


def test_sample_returns_transition_objects():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    buffer.push(
        state=state,
        action=1,
        reward=1.0,
        next_state=state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    batch = buffer.sample(batch_size=1)

    assert isinstance(batch[0], Transition)

def test_replay_buffer_restores_state():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    next_state = torch.ones((BOARD_CHANNELS, 8, 8))

    buffer.push(
        state=state,
        action=123,
        reward=1.0,
        next_state=next_state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    saved_state = buffer.state_dict()

    restored_buffer = ReplayBuffer(capacity=1)
    restored_buffer.load_state_dict(saved_state)

    assert len(restored_buffer) == 1

    transition = restored_buffer.buffer[0]

    assert torch.equal(transition.state, state)
    assert transition.action == 123
    assert transition.reward == 1.0
    assert torch.equal(
        transition.next_state,
        next_state,
    )
    assert transition.done is False
    assert transition.next_legal_actions == [
        1,
        2,
        3,
    ]

def test_replay_buffer_restores_capacity():
    buffer = ReplayBuffer(capacity=10)

    saved_state = buffer.state_dict()

    restored_buffer = ReplayBuffer(capacity=1)
    restored_buffer.load_state_dict(saved_state)

    assert restored_buffer.buffer.maxlen == 10

def test_new_transition_receives_initial_priority():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    buffer.push(
        state=state,
        action=1,
        reward=0.0,
        next_state=state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    assert list(buffer.priorities) == [1.0]

def test_new_transition_receives_current_max_priority():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(2):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 1],
        priorities=[0.5, 3.0],
    )

    buffer.push(
        state=state,
        action=2,
        reward=0.0,
        next_state=state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    assert buffer.priorities[-1] == 3.0

def test_update_priorities_changes_selected_priorities():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(3):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 2],
        priorities=[0.25, 4.0],
    )

    assert list(buffer.priorities) == [
        0.25,
        1.0,
        4.0,
    ]

def test_prioritized_sampling_with_zero_alpha_has_uniform_weights():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(3):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 1, 2],
        priorities=[1.0, 10.0, 100.0],
    )

    _, _, weights = buffer.sample_prioritized(
        batch_size=3,
        alpha=0.0,
        beta=1.0,
    )

    assert torch.allclose(
        weights,
        torch.ones(3),
    )

def test_prioritized_sampling_prefers_high_priority_transition():
    random.seed(0)

    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(2):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 1],
        priorities=[1.0, 100.0],
    )

    counts = [0, 0]

    for _ in range(1_000):
        _, indices, _ = buffer.sample_prioritized(
            batch_size=1,
            alpha=1.0,
            beta=0.0,
        )

        counts[indices[0]] += 1

    assert counts[1] > counts[0]

def test_prioritized_sampling_normalizes_importance_weights():
    random.seed(0)

    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(3):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 1, 2],
        priorities=[1.0, 2.0, 3.0],
    )

    _, _, weights = buffer.sample_prioritized(
        batch_size=10,
        alpha=1.0,
        beta=1.0,
    )

    assert weights.max().item() == 1.0
    assert torch.all(weights > 0)
    assert torch.all(weights <= 1)

def test_replay_buffer_restores_priorities():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    for action in range(2):
        buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )

    buffer.update_priorities(
        indices=[0, 1],
        priorities=[0.25, 2.5],
    )

    saved_state = buffer.state_dict()

    restored_buffer = ReplayBuffer(capacity=1)
    restored_buffer.load_state_dict(saved_state)

    assert list(restored_buffer.priorities) == [
        0.25,
        2.5,
    ]

def test_replay_buffer_loads_old_state_without_priorities():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    buffer.push(
        state=state,
        action=1,
        reward=0.0,
        next_state=state,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    old_state = {
        "capacity": buffer.buffer.maxlen,
        "transitions": list(buffer.buffer),
    }

    restored_buffer = ReplayBuffer(capacity=1)
    restored_buffer.load_state_dict(old_state)

    assert list(restored_buffer.priorities) == [1.0]