import chess
import torch
from chess_rl.utils.board_encoder import (
    BLACK_KINGSIDE_CASTLING_CHANNEL,
    BLACK_QUEENSIDE_CASTLING_CHANNEL,
    BOARD_CHANNELS,
    EN_PASSANT_CHANNEL,
    TURN_CHANNEL,
    WHITE_KINGSIDE_CASTLING_CHANNEL,
    WHITE_QUEENSIDE_CASTLING_CHANNEL,
    encode_board,
)
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

    assert tensor.shape == (
        BOARD_CHANNELS,
        8,
        8,
    )


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

def test_board_encoder_encodes_castling_rights():
    board = chess.Board(
        "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"
    )

    tensor = encode_board(board)

    assert (
        tensor[
            WHITE_KINGSIDE_CASTLING_CHANNEL
        ].sum()
        == 64
    )

    assert (
        tensor[
            WHITE_QUEENSIDE_CASTLING_CHANNEL
        ].sum()
        == 0
    )

    assert (
        tensor[
            BLACK_KINGSIDE_CASTLING_CHANNEL
        ].sum()
        == 0
    )

    assert (
        tensor[
            BLACK_QUEENSIDE_CASTLING_CHANNEL
        ].sum()
        == 64
    )

def test_board_encoder_encodes_en_passant_square():
    board = chess.Board()

    board.push(
        chess.Move.from_uci("e2e4")
    )

    tensor = encode_board(board)

    expected_square = chess.E3

    row = 7 - chess.square_rank(
        expected_square
    )
    col = chess.square_file(
        expected_square
    )

    assert tensor[
        EN_PASSANT_CHANNEL
    ].sum() == 1

    assert tensor[
        EN_PASSANT_CHANNEL,
        row,
        col,
    ] == 1

def test_board_encoder_encodes_white_to_move():
    board = chess.Board()

    tensor = encode_board(board)

    assert tensor[
        TURN_CHANNEL
    ].sum() == 64

def test_board_encoder_encodes_black_to_move():
    board = chess.Board()

    board.push(
        chess.Move.from_uci("e2e4")
    )

    tensor = encode_board(board)

    assert tensor[
        TURN_CHANNEL
    ].sum() == 0