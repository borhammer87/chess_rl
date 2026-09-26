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

def evaluate_state_action_greedy_choice(
    network: StateActionDQN,
    state: torch.Tensor,
    legal_moves: list[chess.Move],
) -> tuple[int, float, float | None]:
    """
    Return the greedy legal action and its Q-value diagnostics.

    The returned tuple contains:
    - selected encoded action;
    - highest legal Q-value;
    - gap between the highest and second-highest legal Q-values,
      or None when only one legal action exists.
    """
    if not legal_moves:
        raise ValueError(
            "Cannot evaluate a greedy choice without legal moves."
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

    best_q = float(
        q_values[best_local_index].item()
    )

    if len(legal_action_ids) == 1:
        q_gap = None
    else:
        top_q_values = torch.topk(
            q_values,
            k=2,
        ).values

        q_gap = float(
            (
                top_q_values[0]
                - top_q_values[1]
            ).item()
        )

    return (
        legal_action_ids[best_local_index],
        best_q,
        q_gap,
    )