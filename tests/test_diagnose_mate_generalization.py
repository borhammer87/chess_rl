
import chess
import chess.pgn

from scripts.diagnose_mate_generalization import (
    diagnose,
    game_positions,
    position_key,
)


def test_position_key_ignores_move_counters():
    board = chess.Board()
    other = chess.Board(board.fen())
    other.fullmove_number = 40

    assert position_key(board) == position_key(other)


def test_game_positions_only_uses_dqn_turns(tmp_path):
    board = chess.Board()

    for san in (
        "e4", "e5", "Qh5", "Nc6",
        "Bc4", "Nf6", "Qxf7#",
    ):
        board.push_san(san)

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "DQN"
    game.headers["Black"] = "RandomAgent"

    (tmp_path / "game_1.pgn").write_text(
        str(game), encoding="utf-8"
    )

    positions = list(game_positions(tmp_path))

    assert len(positions) == 1
    assert [
        move.uci()
        for move in positions[0].legal_moves
        if _is_mate(positions[0], move)
    ] == ["h5f7"]


def _is_mate(board, move):
    board.push(move)
    result = board.is_checkmate()
    board.pop()
    return result


def test_diagnose_deduplicates_positions(monkeypatch):
    board = chess.Board(
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    )

    monkeypatch.setattr(
        "scripts.diagnose_mate_generalization.evaluate_mate_position",
        lambda board, network: (
            None, True, 1.0, 0.2, 1.0, 0.5, 0.5
        ),
    )

    result = diagnose(None, [board, board.copy()])

    assert result == (
        1,
        1,
        [0.5],
        [1.0],
        [0.5],
    )