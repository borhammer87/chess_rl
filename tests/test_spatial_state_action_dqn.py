
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
