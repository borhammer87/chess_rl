import random

import chess
import torch

from chess_rl.models.state_action_dqn import StateActionDQN
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import (
    select_state_action_greedy_action,
)
import torch.nn.functional as F

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

    def train_step(
        self,
        batch,
        weights: torch.Tensor | None = None,
        return_td_errors: bool = False,
    ):
        """
        Perform one State-Action DQN update using a batch of transitions.
        """
        states = torch.stack([
            transition.state
            for transition in batch
        ])

        actions = [
            transition.action
            for transition in batch
        ]

        rewards = torch.tensor(
            [
                transition.reward
                for transition in batch
            ],
            dtype=torch.float32,
        )

        # Q(s, a) for the action actually taken in each transition.
        q_values = self.policy_net.evaluate_state_action_pairs(
            states,
            actions,
        )

        # Bellman future values. Terminal transitions keep zero.
        with torch.no_grad():
            next_q_values = torch.zeros(
                len(batch),
                dtype=torch.float32,
            )

            non_terminal_indices = [
                index
                for index, transition in enumerate(batch)
                if not transition.done
            ]

            for index in non_terminal_indices:
                if not batch[index].next_legal_actions:
                    raise ValueError(
                        "Non-terminal transition must have legal next actions."
                    )

            if non_terminal_indices:
                non_terminal_next_states = torch.stack([
                    batch[index].next_state
                    for index in non_terminal_indices
                ])

                non_terminal_legal_actions = [
                    batch[index].next_legal_actions
                    for index in non_terminal_indices
                ]

                non_terminal_next_q_values = (
                    self.target_net.evaluate_legal_action_maxes(
                        non_terminal_next_states,
                        non_terminal_legal_actions,
                    )
                )

                next_q_values[
                    non_terminal_indices
                ] = non_terminal_next_q_values

            targets = (
                rewards
                + self.gamma * next_q_values
            )

        td_errors = targets - q_values

        elementwise_losses = F.mse_loss(
            q_values,
            targets,
            reduction="none",
        )

        if weights is not None:
            if weights.shape != elementwise_losses.shape:
                raise ValueError(
                    "weights must match the batch size."
                )

            elementwise_losses = (
                elementwise_losses * weights
            )

        loss = elementwise_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if return_td_errors:
            return (
                loss.item(),
                td_errors.detach().abs().tolist(),
            )

        return loss.item()

    def update_target(self):
        """
        Sync policy network weights to the target network.
        """
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        """
        Save the current agent training state.
        """
        torch.save(
            self.state_dict(),
            path,
        )


    def load_checkpoint(
        self,
        path: str,
    ) -> None:
        """
        Restore a previously saved agent training state.
        """
        checkpoint = torch.load(
            path,
            weights_only=False,
        )

        self.load_state_dict(
            checkpoint
        )


    def state_dict(self) -> dict:
        """
        Return the current State-Action DQN training state.
        """
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }


    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        """
        Restore a State-Action DQN training state.
        """
        self.policy_net.load_state_dict(
            state["policy_net"]
        )

        self.target_net.load_state_dict(
            state["target_net"]
        )

        self.optimizer.load_state_dict(
            state["optimizer"]
        )

        self.epsilon = state["epsilon"]