import random
import torch
import torch.nn.functional as F
import chess

from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import select_greedy_action



class DQNAgent:
    """
    DQN Agent for Chess RL.

    Responsibilities:
    - Select actions (epsilon-greedy)
    - Maintain policy + target networks
    - Perform training step (Bellman update)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
    ):
        self.policy_net = DQNCNN()
        self.target_net = DQNCNN()

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # -------------------------
    # ACTION SELECTION
    # -------------------------
    def select_action(
    self,
    state: torch.Tensor,
    legal_moves: list[chess.Move],
        ) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Exploration:
            Select a random legal move.

        Exploitation:
            Predict all Q-values, mask illegal moves and select
            the legal action with the highest Q-value.
        """
        if not legal_moves:
            raise ValueError("Cannot select an action without legal moves.")

        # Exploration: choose one legal move and encode it.
        if random.random() < self.epsilon:
            random_move = random.choice(legal_moves)
            return encode_move(random_move)

        return select_greedy_action(
            network=self.policy_net,
            state=state,
            legal_moves=legal_moves,
        )
    # -------------------------
    # TRAINING STEP
    # -------------------------
    def train_step(self, batch):
        """
        Perform one DQN update using a batch of transitions.
        """

        states = torch.stack([t.state for t in batch])
        actions = torch.tensor([t.action for t in batch])
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])

        # current Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q values
        with torch.no_grad():
            all_next_q_values = self.target_net(
                next_states
            )

            next_q_values = torch.zeros(
                len(batch),
                dtype=torch.float32,
            )

            for index, transition in enumerate(batch):
                if transition.done:
                    continue

                if not transition.next_legal_actions:
                    raise ValueError(
                        "Non-terminal transition must have legal next actions."
                    )

                legal_actions = torch.tensor(
                    transition.next_legal_actions,
                    dtype=torch.long,
                )

                next_q_values[index] = all_next_q_values[
                    index,
                    legal_actions,
                ].max()

            targets = (
                rewards
                + self.gamma * next_q_values
            )

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # -------------------------
    # TARGET NETWORK UPDATE
    # -------------------------
    def update_target(self):
        """
        Sync policy network to target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # -------------------------
    # EPSILON DECAY
    # -------------------------
    def decay_epsilon(self):
        """
        Reduce exploration over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, path: str) -> None:
        """
        Save the current training state to disk.
        """
        torch.save(
            self.state_dict(),
            path,
        )
        

    def load_checkpoint(self, path: str) -> None:
        """
        Restore a previously saved training state.
        """
        checkpoint = torch.load(
            path,
            weights_only = False,
        )

        self.load_state_dict(checkpoint)

    def state_dict(self) -> dict:
        """
        Return the current DQN training state.
        """
        return {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore a previously saved DQN training state.
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