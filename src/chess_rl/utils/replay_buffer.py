from collections import deque
from dataclasses import dataclass
import random

import torch


@dataclass
class Transition:
    """
    Single experience transition used in DQN training.
    """

    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and allows random batch sampling.
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: Maximum number of transitions stored.
        """

        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Add a transition to the replay buffer.
        """

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Randomly sample a batch of transitions.
        """

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        Current number of stored transitions.
        """

        return len(self.buffer)