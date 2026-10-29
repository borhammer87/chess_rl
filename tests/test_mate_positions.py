import chess
import pytest
from chess_rl.training.mate_positions import (
    generate_mate_positions,
    split_mate_positions,
)


def test_generated_positions_are_valid_mates_in_one():
    positions = generate_mate_positions(8, seed=12345)

    assert len(positions) == 8
    assert len({board.fen() for board in positions}) == 8

    for board in positions:
        assert board.is_valid()
        assert not board.is_game_over()
        assert len(board.piece_map()) == 3

        original = board.fen()
        mate_found = False

        for move in board.legal_moves:
            board.push(move)
            mate_found |= board.is_checkmate()
            board.pop()

        assert mate_found
        assert board.fen() == original


def test_generation_is_reproducible():
    first = generate_mate_positions(5, seed=42)
    second = generate_mate_positions(5, seed=42)

    assert [b.fen() for b in first] == [b.fen() for b in second]


def test_generation_rejects_negative_count():
    with pytest.raises(ValueError, match="cannot be negative"):
        generate_mate_positions(-1)


def test_generation_accepts_zero():
    assert generate_mate_positions(0) == []

def test_split_mate_positions_has_no_overlap():
    train, test = split_mate_positions(
        train_count=20,
        test_count=10,
        seed=12345,
    )

    train_fens = {board.fen() for board in train}
    test_fens = {board.fen() for board in test}

    assert len(train) == 20
    assert len(test) == 10
    assert train_fens.isdisjoint(test_fens)


def test_split_mate_positions_is_reproducible():
    first_train, first_test = split_mate_positions(5, 3, seed=42)
    second_train, second_test = split_mate_positions(5, 3, seed=42)

    assert [b.fen() for b in first_train] == [
        b.fen() for b in second_train
    ]
    assert [b.fen() for b in first_test] == [
        b.fen() for b in second_test
    ]


def test_split_mate_positions_rejects_negative_counts():
    with pytest.raises(ValueError, match="cannot be negative"):
        split_mate_positions(-1, 10)