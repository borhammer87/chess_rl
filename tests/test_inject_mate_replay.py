import csv
import chess
import pytest
from scripts.inject_mate_replay import load_mate_examples


def test_load_mate_examples_validates_and_deduplicates(tmp_path):
    board = chess.Board("rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq g6 0 3")
    path = tmp_path / "mates.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fen", "mate_moves"])
        writer.writeheader()
        writer.writerows([{"fen": board.fen(), "mate_moves": "d1h5"}] * 2)
    transitions, unique = load_mate_examples(path)
    assert unique == 1
    assert len(transitions) == 1
    assert transitions[0].reward == 1.0
    assert transitions[0].done
    path.write_text("fen,mate_moves\n" + board.fen() + ",a2a3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incorrect mate labels"):
        load_mate_examples(path)
