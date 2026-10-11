
import chess

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
