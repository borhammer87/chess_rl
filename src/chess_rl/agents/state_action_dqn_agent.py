import random

import chess
import torch

from chess_rl.models.state_action_dqn import StateActionDQN
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import (
    select_state_action_greedy_action,
)


class StateActionDQNAgent:
    """
    DQN agent using an explicit state-action Q function.

    The state encoder is evaluated once per position and the shared
    Q-head scores only the supplied legal actions.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
    ):
        self.policy_net = StateActionDQN()
        self.target_net = StateActionDQN()

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
        )

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(
        self,
        state: torch.Tensor,
        legal_moves: list[chess.Move],
    ) -> int:
        """
        Select an action using an epsilon-greedy policy.
        """
        if not legal_moves:
            raise ValueError(
                "Cannot select an action without legal moves."
            )

        if random.random() < self.epsilon:
            return encode_move(
                random.choice(legal_moves)
            )

        return select_state_action_greedy_action(
            network=self.policy_net,
            state=state,
            legal_moves=legal_moves,
        )

    def update_target_network(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )