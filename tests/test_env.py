# tests/test_env.py

import chess

from chess_rl.env.chess_env import ChessEnv


# =========================================================
# ENV RESET
# =========================================================

def test_env_reset_returns_board():
    env = ChessEnv()

    board = env.reset()

    assert isinstance(board, chess.Board)


def test_env_starts_in_initial_position():
    env = ChessEnv()

    board = env.reset()

    assert board.fen() == chess.Board().fen()


# =========================================================
# ENV STEP
# =========================================================

def test_env_step_changes_turn():
    env = ChessEnv()

    env.reset()

    initial_turn = env.board.turn

    legal_move = next(iter(env.board.legal_moves))

    env.step(legal_move)

    assert env.board.turn != initial_turn


def test_env_step_updates_board():
    env = ChessEnv()

    env.reset()

    move = chess.Move.from_uci("e2e4")

    env.step(move)

    piece = env.board.piece_at(chess.E4)

    assert piece is not None
    assert piece.symbol() == "P"


# =========================================================
# LEGAL MOVES
# =========================================================

def test_env_has_legal_moves():
    env = ChessEnv()

    env.reset()

    legal_moves = list(env.board.legal_moves)

    assert len(legal_moves) > 0


# =========================================================
# TERMINAL STATES
# =========================================================

def test_new_game_is_not_done():
    env = ChessEnv()

    env.reset()

    assert env.board.is_game_over() is False