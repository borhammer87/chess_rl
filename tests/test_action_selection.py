import chess
import pytest
import torch

from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.action_encoder import (
    ACTION_SIZE,
    encode_move,
)
from chess_rl.utils.action_selection import (
    select_greedy_action,
)
from chess_rl.utils.board_encoder import BOARD_CHANNELS


class FakeNetwork(DQNCNN):
    def __init__(
        self,
        q_values: torch.Tensor,
    ):
        super().__init__()
        self.q_values = q_values

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.q_values.unsqueeze(0)


def test_select_greedy_action_selects_best_legal_action():
    board = chess.Board()

    legal_moves = list(board.legal_moves)

    preferred_move = chess.Move.from_uci(
        "e2e4"
    )

    preferred_action = encode_move(
        preferred_move
    )

    q_values = torch.zeros(
        ACTION_SIZE
    )

    q_values[preferred_action] = 10.0

    network = FakeNetwork(
        q_values
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = select_greedy_action(
        network=network,
        state=state,
        legal_moves=legal_moves,
    )

    assert action == preferred_action

def test_select_greedy_action_ignores_higher_illegal_action():
    board = chess.Board()

    legal_moves = list(board.legal_moves)

    preferred_move = chess.Move.from_uci(
        "e2e4"
    )

    preferred_action = encode_move(
        preferred_move
    )

    illegal_move = chess.Move.from_uci(
        "e2e5"
    )

    illegal_action = encode_move(
        illegal_move
    )

    q_values = torch.zeros(
        ACTION_SIZE
    )

    q_values[preferred_action] = 10.0
    q_values[illegal_action] = 100.0

    network = FakeNetwork(
        q_values
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = select_greedy_action(
        network=network,
        state=state,
        legal_moves=legal_moves,
    )

    assert action == preferred_action

def test_select_greedy_action_rejects_empty_legal_moves():
    q_values = torch.zeros(
        ACTION_SIZE
    )

    network = FakeNetwork(
        q_values
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    with pytest.raises(
        ValueError,
        match="without legal moves",
    ):
        select_greedy_action(
            network=network,
            state=state,
            legal_moves=[],
        )