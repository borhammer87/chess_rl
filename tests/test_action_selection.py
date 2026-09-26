import chess
import pytest
import torch

from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.action_encoder import (
    ACTION_SIZE,
    encode_move,
)
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.models.state_action_dqn import StateActionDQN
from chess_rl.utils.action_selection import (
    select_greedy_action,
    select_state_action_greedy_action,
    evaluate_state_action_greedy_choice,
)

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

class FakeStateActionNetwork(StateActionDQN):
    def __init__(
        self,
        q_values: torch.Tensor,
    ):
        super().__init__()
        self.q_values = q_values

    def evaluate_action_ids(
        self,
        state: torch.Tensor,
        action_ids: list[int],
    ) -> torch.Tensor:
        return self.q_values

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

def test_select_state_action_greedy_action_selects_best_legal_action():
    board = chess.Board()

    legal_moves = list(board.legal_moves)

    preferred_move = chess.Move.from_uci(
        "e2e4"
    )

    preferred_index = legal_moves.index(
        preferred_move
    )

    q_values = torch.zeros(
        len(legal_moves)
    )

    q_values[preferred_index] = 10.0

    network = FakeStateActionNetwork(
        q_values
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = select_state_action_greedy_action(
        network=network,
        state=state,
        legal_moves=legal_moves,
    )

    assert action == encode_move(
        preferred_move
    )

def test_select_state_action_greedy_action_rejects_empty_legal_moves():
    network = FakeStateActionNetwork(
        torch.tensor([])
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    with pytest.raises(
        ValueError,
        match="without legal moves",
    ):
        select_state_action_greedy_action(
            network=network,
            state=state,
            legal_moves=[],
        )

def test_evaluate_state_action_greedy_choice_returns_best_action_and_q_gap():
    board = chess.Board()

    legal_moves = list(board.legal_moves)

    preferred_move = chess.Move.from_uci(
        "e2e4"
    )

    preferred_index = legal_moves.index(
        preferred_move
    )

    q_values = torch.zeros(
        len(legal_moves)
    )

    q_values[preferred_index] = 10.0
    q_values[
        (preferred_index + 1) % len(legal_moves)
    ] = 7.5

    network = FakeStateActionNetwork(
        q_values
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action, best_q, q_gap = (
        evaluate_state_action_greedy_choice(
            network=network,
            state=state,
            legal_moves=legal_moves,
        )
    )

    assert action == encode_move(
        preferred_move
    )
    assert best_q == pytest.approx(10.0)
    assert q_gap == pytest.approx(2.5)

def test_evaluate_state_action_greedy_choice_returns_none_gap_for_one_move():
    legal_moves = [
        chess.Move.from_uci("e2e4")
    ]

    network = FakeStateActionNetwork(
        torch.tensor([3.0])
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action, best_q, q_gap = (
        evaluate_state_action_greedy_choice(
            network=network,
            state=state,
            legal_moves=legal_moves,
        )
    )

    assert action == encode_move(
        legal_moves[0]
    )
    assert best_q == pytest.approx(3.0)
    assert q_gap is None

def test_evaluate_state_action_greedy_choice_rejects_empty_legal_moves():
    network = FakeStateActionNetwork(
        torch.tensor([])
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    with pytest.raises(
        ValueError,
        match="without legal moves",
    ):
        evaluate_state_action_greedy_choice(
            network=network,
            state=state,
            legal_moves=[],
        )