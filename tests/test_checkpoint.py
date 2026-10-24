import torch
import random
import numpy as np
import pytest
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    load_checkpoint_metadata,
)
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.agents.state_action_dqn_agent import (
    StateActionDQNAgent,
)

def test_training_checkpoint_restores_agent_and_replay_buffer(
    tmp_path,
):
    agent = DQNAgent(epsilon=0.4)
    replay_buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    next_state = torch.ones((BOARD_CHANNELS, 8, 8))

    replay_buffer.push(
        state=state,
        action=123,
        reward=1.0,
        next_state=next_state,
        done=False,
        next_legal_actions=[1, 2, 3]
    )

    checkpoint_path = tmp_path / "training_checkpoint.pt"

    save_training_checkpoint(
        path=str(checkpoint_path),
        agent=agent,
        replay_buffer=replay_buffer,
    )

    restored_agent = DQNAgent(epsilon=0.9)
    restored_buffer = ReplayBuffer(capacity=1)

    load_training_checkpoint(
        path=str(checkpoint_path),
        agent=restored_agent,
        replay_buffer=restored_buffer,
    )

    assert restored_agent.epsilon == 0.4
    assert len(restored_buffer) == 1
    assert restored_buffer.buffer.maxlen == 10

    transition = restored_buffer.buffer[0]

    assert transition.action == 123
    assert transition.reward == 1.0
    assert torch.equal(
        transition.state,
        state,
    )
    assert torch.equal(
        transition.next_state,
        next_state,
    )

def test_training_checkpoint_restores_metadata(
    tmp_path,
):
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path=str(checkpoint_path),
        agent=agent,
        replay_buffer=replay_buffer,
        metadata={
            "evaluation_score": 0.75,
        },
    )

    metadata = load_checkpoint_metadata(
        str(checkpoint_path)
    )

    assert metadata["evaluation_score"] == 0.75

def test_checkpoint_without_metadata_returns_empty_dict(
    tmp_path,
):
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        path=str(checkpoint_path),
        agent=agent,
        replay_buffer=replay_buffer,
    )

    metadata = load_checkpoint_metadata(
        str(checkpoint_path)
    )

    assert metadata == {}

def test_training_checkpoint_supports_state_action_agent(
    tmp_path,
):
    agent = StateActionDQNAgent(
        epsilon=0.4
    )

    replay_buffer = ReplayBuffer(
        capacity=10
    )

    checkpoint_path = (
        tmp_path / "state_action_checkpoint.pt"
    )

    save_training_checkpoint(
        path=str(checkpoint_path),
        agent=agent,
        replay_buffer=replay_buffer,
    )

    restored_agent = StateActionDQNAgent(
        epsilon=0.9
    )

    restored_buffer = ReplayBuffer(
        capacity=1
    )

    load_training_checkpoint(
        path=str(checkpoint_path),
        agent=restored_agent,
        replay_buffer=restored_buffer,
    )

    assert restored_agent.epsilon == 0.4

    for original, restored in zip(
        agent.policy_net.parameters(),
        restored_agent.policy_net.parameters(),
    ):
        assert torch.equal(
            original,
            restored,
        )

def test_checkpoint_restores_random_generators(tmp_path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    path = tmp_path / "rng_checkpoint.pt"

    save_training_checkpoint(
        path=str(path),
        agent=DQNAgent(),
        replay_buffer=ReplayBuffer(capacity=10),
    )

    expected = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    load_training_checkpoint(
        path=str(path),
        agent=DQNAgent(),
        replay_buffer=ReplayBuffer(capacity=10),
        restore_rng_state=True,
    )

    actual = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )

    assert actual == expected


def test_old_checkpoint_rejects_strict_rng_restore(tmp_path):
    path = tmp_path / "old_checkpoint.pt"

    save_training_checkpoint(
        path=str(path),
        agent=DQNAgent(),
        replay_buffer=ReplayBuffer(capacity=10),
    )

    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["rng_state"]
    torch.save(checkpoint, path)

    with pytest.raises(
        ValueError,
        match="Checkpoint has no RNG state",
    ):
        load_training_checkpoint(
            path=str(path),
            agent=DQNAgent(),
            replay_buffer=ReplayBuffer(capacity=10),
            restore_rng_state=True,
        )