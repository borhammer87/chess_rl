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

def test_replay_buffer_restores_state():
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

def test_replay_buffer_restores_capacity():
    buffer = ReplayBuffer(capacity=10)

    saved_state = buffer.state_dict()

    restored_buffer = ReplayBuffer(capacity=1)
    restored_buffer.load_state_dict(saved_state)

    assert restored_buffer.buffer.maxlen == 10