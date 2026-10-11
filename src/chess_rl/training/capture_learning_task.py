
import chess

from chess_rl.training.episodes import material_reward


def best_material_capture(
    board: chess.Board,
    first_move: chess.Move,
    second_move: chess.Move,
) -> chess.Move:
    """Return the capture with the greater immediate material reward."""
    if first_move == second_move:
        raise ValueError("Candidate moves must be different.")

    rewards = []

    for move in (first_move, second_move):
        if move not in board.legal_moves or not board.is_capture(move):
            raise ValueError("Both moves must be legal captures.")

        next_board = board.copy()
        next_board.push(move)

        rewards.append(
            material_reward(board, next_board, board.turn)
        )

    if rewards[0] == rewards[1]:
        raise ValueError("Captures must have different material rewards.")

    return first_move if rewards[0] > rewards[1] else second_move
