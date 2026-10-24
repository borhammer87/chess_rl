import torch
import random
import numpy as np
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
    checkpoint["rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
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
    restore_rng_state: bool = False,
) -> None:
    """
    Restore a complete training state from disk.
    """
    checkpoint = torch.load(
        path,
        weights_only=False,
    )

    if restore_rng_state:
        rng_state = checkpoint.get("rng_state")
        if rng_state is None:
            raise ValueError("Checkpoint has no RNG state.")

        if (
            rng_state["cuda"] is not None
            and not torch.cuda.is_available()
        ):
            raise ValueError(
                "Checkpoint requires CUDA to restore RNG state."
            )

    agent.load_state_dict(checkpoint["agent"])
    replay_buffer.load_state_dict(checkpoint["replay_buffer"])

    if restore_rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])

        if rng_state["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])