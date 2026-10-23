
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


MATE_POSITIONS = {
    "fools_mate_black": (
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/"
        "PPPPP2P/RNBQKBNR b KQkq g3 0 2"
    ),
    "fools_mate_white": (
        "rnbqkbnr/ppppp2p/5p2/6p1/4P3/8/"
        "PPPP1PPP/RNBQKBNR w KQkq g6 0 3"
    ),
    "queen_rook_mate_white": (
        "6k1/5ppp/8/8/8/8/6PP/4Q1KR w - - 0 1"
    ),
    "queen_rook_mate_black": (
        "4q1kr/6pp/8/8/8/8/5PPP/6K1 b - - 0 1"
    ),
}
