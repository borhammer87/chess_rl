
import chess

from scripts.run_mate_in_one_probe import (
    MATE_POSITIONS,
    find_mate_in_one_moves,
)

import torch

from chess_rl.utils.action_encoder import encode_move
from scripts.run_mate_in_one_probe import evaluate_mate_position

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


def test_all_mate_positions_have_legal_mate_in_one():
    assert len(MATE_POSITIONS) == 4

    colors = set()

    for name, fen in MATE_POSITIONS.items():
        board = chess.Board(fen)

        assert board.is_valid(), name
        assert not board.is_game_over(), name

        mates = find_mate_in_one_moves(board)

        assert mates, name
        assert all(move in board.legal_moves for move in mates)
        colors.add(board.turn)

    assert colors == {chess.WHITE, chess.BLACK}


def test_evaluate_mate_position_detects_greedy_checkmate():
    board = chess.Board(MATE_POSITIONS["fools_mate_black"])
    mating_move = chess.Move.from_uci("d8h4")

    class FakeNetwork:
        def evaluate_action_ids(self, state, action_ids):
            assert state.shape == (18, 8, 8)
            return torch.tensor([
                1.0 if action == encode_move(mating_move) else 0.0
                for action in action_ids
            ])

    original_fen = board.fen()
    original_history = list(board.move_stack)

    move, is_mate, best_q, q_gap = evaluate_mate_position(
        board, FakeNetwork()
    )

    assert move == mating_move
    assert is_mate is True
    assert best_q == 1.0
    assert q_gap == 1.0
    assert board.fen() == original_fen
    assert board.move_stack == original_history


def test_run_mate_in_one_probe_summarizes_results(monkeypatch, capsys):
    from scripts import run_mate_in_one_probe as probe

    outcomes = iter([True, False, True, False])
    visited_colors = []

    def fake_evaluate(board, network):
        visited_colors.append(board.turn)
        return chess.Move.from_uci("a2a3"), next(outcomes), 0.5, 0.1

    monkeypatch.setattr(probe, "evaluate_mate_position", fake_evaluate)

    network = object()
    correct, total = probe.run_mate_in_one_probe(network)

    output = capsys.readouterr().out

    assert (correct, total) == (2, 4)
    assert "Mate-in-one accuracy: 2/4 (50.0%)" in output
    assert output.count("PASS") == 2
    assert output.count("FAIL") == 2
    assert set(visited_colors) == {chess.WHITE, chess.BLACK}



def test_main_uses_untrained_spatial_agent(monkeypatch, capsys):
    from scripts import run_mate_in_one_probe as probe

    captured = {}

    class FakeNetwork:
        def eval(self):
            captured["eval_called"] = True

    class FakeAgent:
        def __init__(self, spatial, device):
            captured["spatial"] = spatial
            captured["device"] = device
            self.policy_net = FakeNetwork()

    def fake_run(network):
        captured["network"] = network
        return 0, 4

    monkeypatch.setattr(probe, "StateActionDQNAgent", FakeAgent)
    monkeypatch.setattr(probe, "run_mate_in_one_probe", fake_run)

    probe.main()

    assert captured["spatial"] is True
    assert captured["device"] == "cpu"
    assert captured["eval_called"] is True
    assert isinstance(captured["network"], FakeNetwork)
    assert "untrained - seed: 12345" in capsys.readouterr().out
