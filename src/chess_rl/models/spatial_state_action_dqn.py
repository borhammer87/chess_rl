
import torch
from torch import nn

from chess_rl.models.state_action_dqn import (
    ACTION_FEATURE_SIZE,
    Q_HEAD_HIDDEN_SIZE,
    STATE_FEATURE_SIZE,
    StateActionDQN,
    destination_square_features,
)
from chess_rl.utils.board_encoder import BOARD_CHANNELS


class SpatialStateActionDQN(StateActionDQN):
    """State-Action DQN with explicit destination-square features."""

    def __init__(self):
        super().__init__()
        self.q_head = nn.Sequential(
            nn.Linear(
                STATE_FEATURE_SIZE + ACTION_FEATURE_SIZE + BOARD_CHANNELS,
                Q_HEAD_HIDDEN_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(Q_HEAD_HIDDEN_SIZE, 1),
        )

    def evaluate_state_action_pairs(self, states, action_ids):
        from chess_rl.utils.action_encoder import decode_action_components

        if states.ndim != 4 or len(states) != len(action_ids):
            raise ValueError("State and action batch sizes must match.")
        if not action_ids:
            raise ValueError("Cannot evaluate an empty batch.")

        components = [decode_action_components(a) for a in action_ids]
        from_squares, to_squares, promotions = (
            torch.tensor(values, device=states.device)
            for values in zip(*components)
        )

        state_features = self.encode_state(states)
        action_features = self.encode_actions(
            from_squares, to_squares, promotions
        )
        destination_features = destination_square_features(
            states, to_squares
        )

        return self.q_head(torch.cat(
            (state_features, action_features, destination_features),
            dim=-1,
        )).squeeze(-1)
