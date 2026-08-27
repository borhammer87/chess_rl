import chess
import torch

from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.action_masking import mask_illegal_moves


def select_greedy_action(
    network: DQNCNN,
    state: torch.Tensor,
    legal_moves: list[chess.Move],
) -> int:
    """
    Select the legal action with the highest Q-value.

    The network is used only for inference. Illegal actions are masked
    before selecting the maximum Q-value.
    """
    if not legal_moves:
        raise ValueError(
            "Cannot select an action without legal moves."
        )

    with torch.no_grad():
        q_values = network(
            state.unsqueeze(0)
        )[0]

    masked_q_values = mask_illegal_moves(
        q_values=q_values,
        legal_moves=legal_moves,
    )

    return int(
        torch.argmax(
            masked_q_values
        ).item()
    )