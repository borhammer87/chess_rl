
import chess


MATE_POSITIONS = {
    "fools_mate": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "back_rank": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
}


def find_mate_in_one_moves(board: chess.Board) -> list[chess.Move]:
    """Return all legal moves that immediately checkmate."""
    mates = []

    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            mates.append(move)
        board.pop()

    return mates
