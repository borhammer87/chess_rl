
import chess
import chess.pgn

from scripts.evaluate_checkpoint_games import audit_mates


def test_audit_mates_counts_missed_and_converted(tmp_path):
    # Scholar's mate: white can mate on f7, and takes it.
    board = chess.Board()

    for san in (
        "e4",
        "e5",
        "Qh5",
        "Nc6",
        "Bc4",
        "Nf6",
        "Qxf7#",
    ):
        board.push_san(san)

    path = tmp_path / "game.pgn"

    path.write_text(
        str(chess.pgn.Game.from_board(board)),
        encoding="utf-8",
    )

    assert audit_mates(path, chess.WHITE) == (1, 1)
    assert audit_mates(path, chess.BLACK) == (0, 0)
