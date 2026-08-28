import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    load_checkpoint_metadata,
)
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.utils.board_encoder import BOARD_CHANNELS

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