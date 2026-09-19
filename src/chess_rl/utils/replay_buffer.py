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
    next_legal_actions: list[int]


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and supports both uniform and prioritized
    batch sampling.
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: Maximum number of transitions stored.
        """

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        next_legal_actions: list[int],
    ) -> None:
        """
        Add a transition to the replay buffer.

        New transitions receive the current maximum priority so they
        have a good chance of being sampled at least once.
        """

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_legal_actions=next_legal_actions,
        )

        if self.priorities:
            initial_priority = max(self.priorities)
        else:
            initial_priority = 1.0

        self.buffer.append(transition)
        self.priorities.append(initial_priority)

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Randomly sample a batch of transitions uniformly.
        """

        return random.sample(
            self.buffer,
            batch_size,
        )

    def sample_prioritized(
        self,
        batch_size: int,
        alpha: float,
        beta: float,
    ) -> tuple[
        list[Transition],
        list[int],
        torch.Tensor,
    ]:
        """
        Sample transitions according to their priorities.

        Args:
            batch_size:
                Number of transitions to sample.

            alpha:
                Strength of prioritization.
                alpha=0 gives uniform probabilities.

            beta:
                Strength of importance-sampling correction.
                beta=1 fully applies the correction.

        Returns:
            transitions:
                Sampled replay transitions.

            indices:
                Positions of the sampled transitions in the buffer.

            weights:
                Normalized importance-sampling weights.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not self.buffer:
            raise ValueError(
                "Cannot sample from an empty replay buffer."
            )

        if alpha < 0:
            raise ValueError(
                "alpha must be non-negative."
            )

        if not 0 <= beta <= 1:
            raise ValueError(
                "beta must be between zero and one."
            )

        scaled_priorities = [
            priority ** alpha
            for priority in self.priorities
        ]

        total_priority = sum(
            scaled_priorities
        )

        probabilities = [
            priority / total_priority
            for priority in scaled_priorities
        ]

        indices = random.choices(
            range(len(self.buffer)),
            weights=probabilities,
            k=batch_size,
        )

        transitions = [
            self.buffer[index]
            for index in indices
        ]

        sample_probabilities = torch.tensor(
            [
                probabilities[index]
                for index in indices
            ],
            dtype=torch.float32,
        )

        weights = (
            len(self.buffer)
            * sample_probabilities
        ) ** (-beta)

        weights = weights / weights.max()

        return (
            transitions,
            indices,
            weights,
        )

    def update_priorities(
        self,
        indices: list[int],
        priorities: list[float],
    ) -> None:
        """
        Update priorities for previously sampled transitions.
        """

        if len(indices) != len(priorities):
            raise ValueError(
                "indices and priorities must have the same length."
            )

        for index, priority in zip(
            indices,
            priorities,
        ):
            if priority <= 0:
                raise ValueError(
                    "priorities must be greater than zero."
                )

            self.priorities[index] = priority

    def __len__(self) -> int:
        """
        Current number of stored transitions.
        """

        return len(self.buffer)

    def state_dict(self) -> dict:
        """
        Return the current replay-buffer state.
        """

        return {
            "capacity": self.buffer.maxlen,
            "transitions": list(self.buffer),
            "priorities": list(self.priorities),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore a previously saved replay-buffer state.

        Older checkpoints without stored priorities remain supported.
        Their transitions receive the default priority 1.0.
        """

        capacity = state["capacity"]
        transitions = state["transitions"]

        priorities = state.get(
            "priorities",
            [1.0] * len(transitions),
        )

        self.buffer = deque(
            transitions,
            maxlen=capacity,
        )

        self.priorities = deque(
            priorities,
            maxlen=capacity,
        )