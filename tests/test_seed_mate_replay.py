
import csv

import chess
import chess.pgn
import pytest

from scripts.extract_mate_csv import extract_mates
from scripts.run_state_action_probe import (
    TerminalTransitionTracker,
    seed_mate_replay,
)
from chess_rl.utils.replay_buffer import ReplayBuffer


def test_extract_and_seed_mates(tmp_path):
    board = chess.Board()

    for san in (
        "e4", "e5", "Qh5", "Nc6",
        "Bc4", "Nf6", "Qxf7#",
    ):
        board.push_san(san)

    pgn = tmp_path / "game.pgn"
    pgn.write_text(
        str(chess.pgn.Game.from_board(board)),
        encoding="utf-8",
    )

    csv_path = tmp_path / "mates.csv"

    assert extract_mates([pgn, pgn], csv_path) == 1

    with csv_path.open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["mate_moves"] == "h5f7"

    replay = ReplayBuffer(capacity=100)
    tracker = TerminalTransitionTracker(replay)

    seed_mate_replay(csv_path, replay, tracker)

    assert len(replay) == 1
    assert replay.buffer[0].reward == 1.0
    assert replay.buffer[0].done is True
    assert tracker.state_dict()["terminal_labels"] == ["win"]


def test_seed_rejects_empty_mate_csv(tmp_path):
    path = tmp_path / "mates.csv"
    path.write_text(
        "fen,mate_moves\n",
        encoding="utf-8",
    )

    replay = ReplayBuffer(capacity=100)
    tracker = TerminalTransitionTracker(replay)

    with pytest.raises(ValueError, match="No mate examples"):
        seed_mate_replay(path, replay, tracker)
