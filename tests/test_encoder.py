import chess
import torch

from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import (
    encode_move,
    decode_move,
    get_legal_actions
)
from chess_rl.utils.action_masking import mask_illegal_moves

import pytest

from chess_rl.utils.action_encoder import (
    decode_legal_action,
    encode_move,
)


# =========================================================
# BOARD ENCODER
# =========================================================

def test_board_encoder_shape():
    board = chess.Board()

    tensor = encode_board(board)

    assert tensor.shape == (12, 8, 8)


def test_board_encoder_is_tensor():
    board = chess.Board()

    tensor = encode_board(board)

    assert isinstance(tensor, (torch.Tensor,))


# =========================================================
# SIMPLE SANITY CHECK
# =========================================================

def test_board_encoder_not_empty():
    board = chess.Board()

    tensor = encode_board(board)

    assert tensor.sum() != 0



def test_decode_legal_action_returns_matching_move():
    board = chess.Board()

    expected_move = chess.Move.from_uci("e2e4")
    action = encode_move(expected_move)

    decoded_move = decode_legal_action(
        action=action,
        legal_moves=list(board.legal_moves),
    )

    assert decoded_move == expected_move


def test_decode_legal_action_prefers_queen_promotion():
    board = chess.Board(
        "4k3/6P1/8/8/8/8/8/4K3 w - - 0 1"
    )

    queen_promotion = chess.Move.from_uci("g7g8q")
    action = encode_move(queen_promotion)

    decoded_move = decode_legal_action(
        action=action,
        legal_moves=list(board.legal_moves),
    )

    assert decoded_move == queen_promotion
    assert decoded_move.promotion == chess.QUEEN

def test_decode_legal_action_rejects_non_legal_action():
    board = chess.Board()
    illegal_move = chess.Move.from_uci("e2e5")
    action = encode_move(illegal_move)

    with pytest.raises(
        ValueError,
        match="does not correspond to any legal move",
    ):
        decode_legal_action(
            action=action,
            legal_moves=list(board.legal_moves),
        )