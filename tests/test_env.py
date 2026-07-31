# tests/test_env.py

import chess
import pytest

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



def test_env_rejects_illegal_move():
    env = ChessEnv()
    env.reset()

    illegal_move = chess.Move.from_uci("e2e5")

    with pytest.raises(ValueError, match="Illegal move"):
        env.step(illegal_move)


def test_env_rejects_non_move_objects():
    env = ChessEnv()
    env.reset()

    with pytest.raises(TypeError):
        env.step("e2e4")


def test_get_state_returns_independent_board():
    env = ChessEnv()
    state = env.reset()

    state.push(chess.Move.from_uci("e2e4"))

    assert env.board.fen() == chess.Board().fen()


def test_env_raises_when_game_is_already_done():
    env = ChessEnv()
    env.done = True

    move = chess.Move.from_uci("e2e4")

    with pytest.raises(RuntimeError):
        env.step(move)


def test_step_returns_info_dictionary():
    env = ChessEnv()
    env.reset()

    move = chess.Move.from_uci("e2e4")

    _, _, _, info = env.step(move)

    assert isinstance(info, dict)
    assert info["result"] is None
    assert info["termination"] is None