import torch
import torch.nn as nn

from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.utils.action_encoder import (
    decode_action_components,
)

STATE_FEATURE_SIZE = 256
SQUARE_EMBEDDING_SIZE = 16
PROMOTION_EMBEDDING_SIZE = 4

PROMOTION_TYPE_COUNT = 5

ACTION_FEATURE_SIZE = (
    2 * SQUARE_EMBEDDING_SIZE
    + PROMOTION_EMBEDDING_SIZE
)
Q_HEAD_HIDDEN_SIZE = 256

class StateActionDQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=BOARD_CHANNELS,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                STATE_FEATURE_SIZE,
            ),
            nn.ReLU(),
        )

        self.from_square_embedding = nn.Embedding(
            num_embeddings=64,
            embedding_dim=SQUARE_EMBEDDING_SIZE,
        )

        self.to_square_embedding = nn.Embedding(
            num_embeddings=64,
            embedding_dim=SQUARE_EMBEDDING_SIZE,
        )

        self.promotion_embedding = nn.Embedding(
            num_embeddings=PROMOTION_TYPE_COUNT,
            embedding_dim=PROMOTION_EMBEDDING_SIZE,
        )

        self.q_head = nn.Sequential(
            nn.Linear(
                STATE_FEATURE_SIZE + ACTION_FEATURE_SIZE,
                Q_HEAD_HIDDEN_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(
                Q_HEAD_HIDDEN_SIZE,
                1,
            ),
        )

    def encode_state(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of board states once.

        The resulting state features can later be reused to
        evaluate multiple legal actions without rerunning the CNN.
        """
        return self.state_encoder(
            states.float()
        )

    def encode_actions(
        self,
        from_squares: torch.Tensor,
        to_squares: torch.Tensor,
        promotion_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode structured chess actions.

        All input tensors must contain one value per action.
        """
        from_features = self.from_square_embedding(
            from_squares.long()
        )

        to_features = self.to_square_embedding(
            to_squares.long()
        )

        promotion_features = self.promotion_embedding(
            promotion_types.long()
        )

        return torch.cat(
            (
                from_features,
                to_features,
                promotion_features,
            ),
            dim=-1,
        )

    def score_actions(
        self,
        state_features: torch.Tensor,
        from_squares: torch.Tensor,
        to_squares: torch.Tensor,
        promotion_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score multiple actions for one already-encoded state.
        """
        action_features = self.encode_actions(
            from_squares,
            to_squares,
            promotion_types,
        )

        action_count = action_features.shape[0]

        expanded_state_features = state_features.expand(
            action_count,
            -1,
        )

        combined_features = torch.cat(
            (
                expanded_state_features,
                action_features,
            ),
            dim=-1,
        )

        return self.q_head(
            combined_features
        ).squeeze(-1)

    def evaluate_actions(
        self,
        state: torch.Tensor,
        from_squares: torch.Tensor,
        to_squares: torch.Tensor,
        promotion_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one state once and score all supplied actions.
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)

        state_features = self.encode_state(
            state
        )

        return self.score_actions(
            state_features,
            from_squares,
            to_squares,
            promotion_types,
        )

    def evaluate_action_ids(
        self,
        state: torch.Tensor,
        action_ids: list[int],
    ) -> torch.Tensor:
        """
        Encode one state once and score encoded chess actions.
        """
        if not action_ids:
            raise ValueError(
                "Cannot evaluate an empty action list."
            )

        components = [
            decode_action_components(action_id)
            for action_id in action_ids
        ]

        from_squares = torch.tensor(
            [
                from_square
                for from_square, _, _ in components
            ],
            dtype=torch.long,
            device=state.device,
        )

        to_squares = torch.tensor(
            [
                to_square
                for _, to_square, _ in components
            ],
            dtype=torch.long,
            device=state.device,
        )

        promotion_types = torch.tensor(
            [
                promotion_type
                for _, _, promotion_type in components
            ],
            dtype=torch.long,
            device=state.device,
        )

        return self.evaluate_actions(
            state,
            from_squares,
            to_squares,
            promotion_types,
        )