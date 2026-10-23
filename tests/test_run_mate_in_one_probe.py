
import chess

from scripts.run_mate_in_one_probe import find_mate_in_one_moves


def test_finds_fools_mate():
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)

    mates = find_mate_in_one_moves(board)

    assert chess.Move.from_uci("d8h4") in mates
    assert all(move in board.legal_moves for move in mates)


def test_initial_position_has_no_mate_in_one():
    board = chess.Board()

    assert find_mate_in_one_moves(board) == []

def test_mate_detection_does_not_modify_board():
    board = chess.Board()

    for move in ("f2f3", "e7e5", "g2g4"):
        board.push_uci(move)

    original_fen = board.fen()
    original_moves = list(board.move_stack)

    find_mate_in_one_moves(board)

    assert board.fen() == original_fen
    assert board.move_stack == original_moves