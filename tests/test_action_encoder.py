import chess

from chess_rl.utils.action_encoder import (
    ACTION_SIZE,
    encode_move,
    decode_move,
    get_legal_actions,
    action_to_uci,
    uci_to_action,
)


def test_encode_decode_consistency():
    move = chess.Move.from_uci("e2e4")

    action = encode_move(move)
    decoded_move = decode_move(action)

    assert move == decoded_move


def test_action_range():
    move = chess.Move.from_uci("a1h8")

    action = encode_move(move)

    assert 0 <= action < ACTION_SIZE


def test_uci_to_action():
    action = uci_to_action("e2e4")

    decoded_move = decode_move(action)

    assert decoded_move.uci() == "e2e4"


def test_action_to_uci():
    move = chess.Move.from_uci("d2d4")

    action = encode_move(move)

    assert action_to_uci(action) == "d2d4"


def test_get_legal_actions():
    board = chess.Board()

    legal_actions = get_legal_actions(board)

    assert len(legal_actions) == 20

    for action in legal_actions:
        move = decode_move(action)

        assert move in board.legal_moves


def test_invalid_action():
    try:
        decode_move(-1)
        assert False

    except ValueError:
        assert True

    try:
        decode_move(ACTION_SIZE)
        assert False

    except ValueError:
        assert True