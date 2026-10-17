
import chess
import torch

from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.board_encoder import encode_board


def test_spatial_model_scores_state_action_pairs():
    model = SpatialStateActionDQN()

    board = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )
    states = torch.stack([encode_board(board), encode_board(board)])
    actions = [
        encode_move(chess.Move.from_uci("e3d5")),
        encode_move(chess.Move.from_uci("e3f5")),
    ]

    scores = model.evaluate_state_action_pairs(states, actions)

    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_spatial_individual_and_batched_scores_match():
    torch.manual_seed(42)
    model = SpatialStateActionDQN()
    model.eval()

    board = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )
    state = encode_board(board)
    actions = [
        encode_move(chess.Move.from_uci("e3d5")),
        encode_move(chess.Move.from_uci("e3f5")),
    ]

    with torch.no_grad():
        individual = model.evaluate_action_ids(state, actions)
        batched = model.evaluate_state_action_pairs(
            torch.stack([state, state]), actions
        )

    torch.testing.assert_close(individual, batched)


def test_spatial_legal_maxes_match_individual_evaluation():
    torch.manual_seed(42)
    model = SpatialStateActionDQN()
    model.eval()

    board_a = chess.Board()
    board_b = chess.Board(
        "4k3/8/8/3r1p2/8/4N3/8/4K3 w - - 0 1"
    )

    states = torch.stack([
        encode_board(board_a),
        encode_board(board_b),
    ])

    actions_a = [
        encode_move(chess.Move.from_uci("e2e4")),
        encode_move(chess.Move.from_uci("d2d4")),
    ]
    actions_b = [
        encode_move(chess.Move.from_uci("e3d5")),
        encode_move(chess.Move.from_uci("e3f5")),
        encode_move(chess.Move.from_uci("e1e2")),
    ]

    with torch.no_grad():
        maximums = model.evaluate_legal_action_maxes(
            states, [actions_a, actions_b]
        )
        expected = torch.stack([
            model.evaluate_action_ids(states[0], actions_a).max(),
            model.evaluate_action_ids(states[1], actions_b).max(),
        ])

    assert maximums.shape == (2,)
    torch.testing.assert_close(maximums, expected)


def test_spatial_legal_maxes_reject_empty_actions():
    import pytest

    model = SpatialStateActionDQN()
    states = torch.zeros((2, 18, 8, 8))
    action = encode_move(chess.Move.from_uci("e2e4"))

    with pytest.raises(ValueError, match="legal actions"):
        model.evaluate_legal_action_maxes(
            states, [[action], []]
        )
