
import chess
import pytest
from chess_rl.training.episodes import material_reward


def test_capture_reward_prefers_rook_over_pawn():
    board = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )

    rook_capture = chess.Move.from_uci("e3d5")
    pawn_capture = chess.Move.from_uci("e3f5")

    assert board.is_valid()
    assert rook_capture in board.legal_moves
    assert pawn_capture in board.legal_moves

    rewards = []

    for move in (rook_capture, pawn_capture):
        next_board = board.copy()
        next_board.push(move)

        rewards.append(
            material_reward(
                board,
                next_board,
                chess.WHITE,
            )
        )

    assert rewards[0] > rewards[1]


def test_best_material_capture_is_independent_of_candidate_order():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
    )

    board = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )
    rook_capture = chess.Move.from_uci("e3d5")
    pawn_capture = chess.Move.from_uci("e3f5")

    assert best_material_capture(
        board, rook_capture, pawn_capture
    ) == rook_capture

    assert best_material_capture(
        board, pawn_capture, rook_capture
    ) == rook_capture


def test_best_material_capture_rejects_non_capture():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
    )

    board = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )

    with pytest.raises(ValueError):
        best_material_capture(
            board,
            chess.Move.from_uci("e3d5"),
            chess.Move.from_uci("e3g4"),
        )


def test_generated_capture_positions_are_valid():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
        make_capture_positions,
    )

    positions = make_capture_positions()

    assert len(positions) == 8
    assert len({board.fen() for board, _, _ in positions}) == 8

    for board, first, second in positions:
        assert board.is_valid()
        assert first in board.legal_moves
        assert second in board.legal_moves
        assert board.is_capture(first)
        assert board.is_capture(second)

        best = best_material_capture(board, first, second)
        assert best in (first, second)


def test_generated_capture_positions_balance_candidate_order():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
        make_capture_positions,
    )

    positions = make_capture_positions()

    first_wins = sum(
        best_material_capture(board, first, second) == first
        for board, first, second in positions
    )

    assert first_wins == 4


def test_random_capture_positions_are_reproducible():
    from chess_rl.training.capture_learning_task import (
        generate_capture_positions,
    )

    def signatures(positions):
        return [
            (board.fen(), first.uci(), second.uci())
            for board, first, second in positions
        ]

    assert signatures(generate_capture_positions(30, 42)) == signatures(
        generate_capture_positions(30, 42)
    )


def test_random_capture_positions_are_valid_and_distinct():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
        generate_capture_positions,
    )

    positions = generate_capture_positions(40, 123)
    assert len({board.fen() for board, _, _ in positions}) == 40

    for board, first, second in positions:
        assert board.is_valid()
        assert first in board.legal_moves
        assert second in board.legal_moves
        assert board.is_capture(first)
        assert board.is_capture(second)
        assert best_material_capture(board, first, second) in (first, second)

    assert {board.turn for board, _, _ in positions} == {
        chess.WHITE, chess.BLACK
    }


def test_random_capture_positions_reject_negative_count():
    from chess_rl.training.capture_learning_task import (
        generate_capture_positions,
    )

    with pytest.raises(ValueError):
        generate_capture_positions(-1, 42)


def test_capture_split_has_no_shared_positions():
    from chess_rl.training.capture_learning_task import (
        split_capture_positions,
    )

    train, evaluation = split_capture_positions(30, 20, seed=42)

    train_fens = {board.fen() for board, _, _ in train}
    evaluation_fens = {board.fen() for board, _, _ in evaluation}

    assert len(train) == 30
    assert len(evaluation) == 20
    assert train_fens.isdisjoint(evaluation_fens)


def test_capture_split_balances_correct_candidate_order():
    from chess_rl.training.capture_learning_task import (
        best_material_capture,
        split_capture_positions,
    )

    train, evaluation = split_capture_positions(30, 20, seed=42)

    for examples in (train, evaluation):
        first_is_best = sum(
            best_material_capture(board, first, second) == first
            for board, first, second in examples
        )

        assert first_is_best == len(examples) // 2


def test_capture_split_is_reproducible():
    from chess_rl.training.capture_learning_task import (
        split_capture_positions,
    )

    def signatures(examples):
        return [
            (board.fen(), first.uci(), second.uci())
            for board, first, second in examples
        ]

    train_a, evaluation_a = split_capture_positions(30, 20, seed=42)
    train_b, evaluation_b = split_capture_positions(30, 20, seed=42)

    assert signatures(train_a) == signatures(train_b)
    assert signatures(evaluation_a) == signatures(evaluation_b)

def test_capture_generator_rejects_impossible_count():
    from chess_rl.training.capture_learning_task import (
        generate_capture_positions,
    )

    with pytest.raises(ValueError, match="theoretical maximum"):
        generate_capture_positions(121, seed=42)