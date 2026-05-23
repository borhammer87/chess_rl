from chess_rl.utils.replay_buffer import ReplayBuffer


def test_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=10)

    assert len(buffer) == 0

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_push_adds_transition():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((12, 8, 8))
    next_state = torch.ones((12, 8, 8))

    buffer.push(
        state=state,
        action=123,
        reward=1.0,
        next_state=next_state,
        done=False,
    )

    assert len(buffer) == 1

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=3)

    state = torch.zeros((12, 8, 8))

    for i in range(5):
        buffer.push(
            state=state,
            action=i,
            reward=0.0,
            next_state=state,
            done=False,
        )

    assert len(buffer) == 3

import torch

from chess_rl.utils.replay_buffer import ReplayBuffer


def test_sample_returns_correct_batch_size():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((12, 8, 8))

    for i in range(5):
        buffer.push(
            state=state,
            action=i,
            reward=0.0,
            next_state=state,
            done=False,
        )

    batch = buffer.sample(batch_size=3)

    assert len(batch) == 3


import torch

from chess_rl.utils.replay_buffer import ReplayBuffer, Transition


def test_sample_returns_transition_objects():
    buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((12, 8, 8))

    buffer.push(
        state=state,
        action=1,
        reward=1.0,
        next_state=state,
        done=False,
    )

    batch = buffer.sample(batch_size=1)

    assert isinstance(batch[0], Transition)