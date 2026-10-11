
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


def make_capture_positions() -> list[
    tuple[chess.Board, chess.Move, chess.Move]
]:
    """Build a small deterministic set of capture comparisons."""
    positions = []

    for knight_file in (2, 3, 4, 5):
        knight_square = chess.square(knight_file, 2)
        left_target = chess.square(knight_file - 1, 4)
        right_target = chess.square(knight_file + 1, 4)

        for rook_on_left in (True, False):
            board = chess.Board(None)
            board.turn = chess.WHITE
            board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(
                knight_square, chess.Piece(chess.KNIGHT, chess.WHITE)
            )
            board.set_piece_at(
                left_target,
                chess.Piece(
                    chess.ROOK if rook_on_left else chess.PAWN,
                    chess.BLACK,
                ),
            )
            board.set_piece_at(
                right_target,
                chess.Piece(
                    chess.PAWN if rook_on_left else chess.ROOK,
                    chess.BLACK,
                ),
            )

            positions.append((
                board,
                chess.Move(knight_square, left_target),
                chess.Move(knight_square, right_target),
            ))

    return positions
