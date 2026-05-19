# src/chess_rl/utils/action_masking.py

import torch
import chess
from chess_rl.utils.action_encoder import encode_move


def mask_illegal_moves(
    q_values: torch.Tensor,
    legal_moves
) -> torch.Tensor:
    """
    Masks illegal chess moves by setting their Q-values
    to a very negative number.
    """

    masked_q_values = q_values.clone()

    # Convert legal moves → action indices
    legal_indices = [
        encode_move(move)
        for move in legal_moves
    ]

    # Create mask (all illegal initially)
    illegal_mask = torch.ones_like(
        masked_q_values,
        dtype=torch.bool,
        device=masked_q_values.device
    )

    # Mark legal actions as allowed
    illegal_mask[legal_indices] = False

    # Apply masking
    masked_q_values = masked_q_values.masked_fill(
        illegal_mask,
        -1e9
    )

    return masked_q_values