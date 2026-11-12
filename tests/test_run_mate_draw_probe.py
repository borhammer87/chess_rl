import chess
import pytest

from scripts import run_mate_draw_probe as probe


@pytest.mark.parametrize("black", [False, True])
def test_all_terminal_actions_and_deterministic_pair(black):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    if black:
        board = board.mirror()
    pair = probe.find_terminal_pair(board)
    assert pair is not None
    for field, termination in (("mate_moves", chess.Termination.CHECKMATE),
                               ("stalemate_moves", chess.Termination.STALEMATE)):
        expected = []
        for move in board.legal_moves:
            end = board.copy()
            end.push(move)
            if end.outcome() and end.outcome().termination == termination:
                expected.append(move)
        assert pair[field] == sorted(expected, key=lambda move: move.uci())
        assert expected
    assert pair["mate_move"] == pair["mate_moves"][0]
    assert pair["stalemate_move"] == pair["stalemate_moves"][0]


def test_preserves_board_and_history():
    board = chess.Board("7k/8/6K1/8/8/8/5Q2/8 w - - 0 1")
    board.push_uci("f2f7")
    board.push(chess.Move.null())
    before = board.fen(), list(board.move_stack), board.root().fen()
    assert probe.find_terminal_pair(board) is not None
    assert (board.fen(), list(board.move_stack), board.root().fen()) == before


@pytest.mark.parametrize("fen", [chess.STARTING_FEN, "8/8/8/8/8/8/8/8 w - - 0 1",
                               "7k/8/5KQ1/8/8/8/8/8 b - - 0 1"])
def test_no_pair(fen):
    assert probe.find_terminal_pair(chess.Board(fen)) is None


def test_generation_reproducible_and_counts():
    def signature(pairs):
        return [(p["board"].fen(), p["mate_moves"], p["stalemate_moves"],
                 p["mate_move"], p["stalemate_move"]) for p in pairs]
    first, counts = probe.generate_terminal_pairs(20, seed=42)
    second, other = probe.generate_terminal_pairs(20, seed=42)
    assert signature(first) == signature(second)
    assert counts == other
    assert counts["candidates"] == 20
    assert counts["pairs"] == len(first) == counts["white"] + counts["black"]
    assert first
    for pair in first:
        board = pair["board"]
        assert board.is_valid() and len(board.piece_map()) == 3
        for field, check in (("mate_move", "is_checkmate"), ("stalemate_move", "is_stalemate")):
            end = board.copy()
            assert pair[field] in end.legal_moves
            end.push(pair[field])
            assert getattr(end, check)()


def test_existing_generator_called_once_and_empty_filter(monkeypatch, capsys):
    calls = []
    def generate(count, seed):
        calls.append((count, seed))
        return [chess.Board()]
    monkeypatch.setattr(probe, "generate_mate_positions", generate)
    probe.main(["--candidates", "1", "--seed", "7"])
    assert calls == [(1, 7)]
    output = capsys.readouterr().out
    assert "candidates=1 pairs=0 white=0 black=0" in output
    assert "No mate/stalemate pairs found" in output


def test_zero_and_negative_candidates():
    assert probe.generate_terminal_pairs(0) == ([], dict(candidates=0, pairs=0, white=0, black=0))
    with pytest.raises(ValueError, match="negative"):
        probe.generate_terminal_pairs(-1)
