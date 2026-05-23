import random
import torch
import torch.nn.functional as F

from chess_rl.models.dqn_cnn import DQNCNN


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
    def select_action(self, state: torch.Tensor, legal_mask: torch.Tensor = None) -> int:
        """
        Epsilon-greedy action selection with optional legal move masking.
        """

        # random exploration
        if random.random() < self.epsilon:
            return random.randint(0, 4095)

        with torch.no_grad():
            q_values = self.policy_net(state.unsqueeze(0))[0]

        # apply legal mask (if provided)
        if legal_mask is not None:
            q_values = q_values + legal_mask  # illegal moves should be -inf or large negative

        return int(torch.argmax(q_values).item())

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
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        # current Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

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