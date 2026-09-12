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

def test_env_detects_checkmate():
    env = ChessEnv()
    env.reset()

    env.step(chess.Move.from_uci("f2f3"))
    env.step(chess.Move.from_uci("e7e5"))
    env.step(chess.Move.from_uci("g2g4"))

    _, reward, done, info = env.step(
        chess.Move.from_uci("d8h4")
    )

    assert done is True
    assert reward == -1.0
    assert info["result"] == "0-1"
    assert info["termination"] == "CHECKMATE"


def test_env_detects_stalemate():
    env = ChessEnv()

    env.board = chess.Board(
        "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"
    )

    _, reward, done, info = env.step(
        chess.Move.from_uci("g5g6")
    )

    assert done is True
    assert reward == 0.0
    assert info["result"] == "1/2-1/2"
    assert info["termination"] == "STALEMATE"


def test_env_detects_insufficient_material():
    env = ChessEnv()

    env.board = chess.Board(
        "7k/8/8/8/8/8/1B6/K7 w - - 0 1"
    )

    _, reward, done, info = env.step(
        chess.Move.from_uci("b2c3")
    )

    assert done is True
    assert reward == 0.0
    assert info["result"] == "1/2-1/2"
    assert info["termination"] == "INSUFFICIENT_MATERIAL"


def test_env_does_not_end_on_claimable_threefold_repetition():
    env = ChessEnv()
    env.reset()

    moves = [
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    ]

    for uci in moves:
        _, _, done, _ = env.step(
            chess.Move.from_uci(uci)
        )

    assert env.board.can_claim_threefold_repetition() is True
    assert done is False


def test_env_detects_automatic_fivefold_repetition():
    env = ChessEnv()
    env.reset()

    moves = [
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    ] * 4

    for uci in moves:
        _, reward, done, info = env.step(
            chess.Move.from_uci(uci)
        )

    assert done is True
    assert reward == 0.0
    assert info["result"] == "1/2-1/2"
    assert info["termination"] == "FIVEFOLD_REPETITION"


def test_env_distinguishes_claimable_and_automatic_move_rule_draws():
    claimable_env = ChessEnv()
    claimable_env.board = chess.Board(
        "8/8/8/8/8/8/R6k/K7 w - - 99 50"
    )

    _, _, claimable_done, _ = claimable_env.step(
        chess.Move.from_uci("a2a3")
    )

    assert claimable_env.board.can_claim_fifty_moves() is True
    assert claimable_done is False

    automatic_env = ChessEnv()
    automatic_env.board = chess.Board(
        "8/8/8/8/8/8/R6k/K7 w - - 149 75"
    )

    _, reward, automatic_done, info = automatic_env.step(
        chess.Move.from_uci("a2a3")
    )

    assert automatic_done is True
    assert reward == 0.0
    assert info["result"] == "1/2-1/2"
    assert info["termination"] == "SEVENTYFIVE_MOVES"