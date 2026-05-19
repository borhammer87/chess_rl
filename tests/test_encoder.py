import chess
import torch

from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import (
    encode_move,
    decode_move,
    get_legal_actions
)
from chess_rl.utils.action_masking import mask_illegal_moves


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