import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.utils.replay_buffer import ReplayBuffer


def save_training_checkpoint(
    path: str,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
    metadata: dict | None = None,
) -> None:
    """
    Save the complete training state to disk.
    """
    checkpoint = {
        "agent": agent.state_dict(),
        "replay_buffer": replay_buffer.state_dict(),
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint_metadata(
    path: str,
) -> dict:
    """
    Return checkpoint metadata without restoring training state.
    """
    checkpoint = torch.load(
        path,
        weights_only=False,
    )

    return checkpoint.get("metadata", {})

def load_training_checkpoint(
    path: str,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
) -> None:
    """
    Restore a complete training state from disk.
    """
    checkpoint = torch.load(
        path,
        weights_only=False,
    )

    agent.load_state_dict(
        checkpoint["agent"]
    )

    replay_buffer.load_state_dict(
        checkpoint["replay_buffer"]
    )

