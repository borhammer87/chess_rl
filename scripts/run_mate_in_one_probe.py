
import chess


def find_mate_in_one_moves(board: chess.Board) -> list[chess.Move]:
    """Return all legal moves that immediately checkmate."""
    mates = []

    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            mates.append(move)
        board.pop()

    return mates
