import chess
import torch

from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.action_masking import mask_illegal_moves
from chess_rl.models.state_action_dqn import StateActionDQN
from chess_rl.utils.action_encoder import encode_move

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

def select_state_action_greedy_action(
    network: StateActionDQN,
    state: torch.Tensor,
    legal_moves: list[chess.Move],
) -> int:
    """
    Select the legal action with the highest state-action Q-value.
    """
    if not legal_moves:
        raise ValueError(
            "Cannot select an action without legal moves."
        )

    legal_action_ids = [
        encode_move(move)
        for move in legal_moves
    ]

    with torch.no_grad():
        q_values = network.evaluate_action_ids(
            state,
            legal_action_ids,
        )

    best_local_index = int(
        torch.argmax(q_values).item()
    )

    return legal_action_ids[
        best_local_index
    ]