import random

import chess


def generate_mate_positions(
    count: int,
    seed: int = 12345,
) -> list[chess.Board]:
    """Generate distinct, legal KQK positions with mate in one."""
    if count < 0:
        raise ValueError("Count cannot be negative.")

    rng = random.Random(seed)
    positions = []
    seen = set()
    attempts = 0
    max_attempts = max(5000, count * 2000)

    edge = [
        square for square in chess.SQUARES
        if chess.square_file(square) in (0, 7)
        or chess.square_rank(square) in (0, 7)
    ]

    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        attacker = rng.choice((chess.WHITE, chess.BLACK))
        defender_king = rng.choice(edge)

        nearby = [
            square for square in chess.SQUARES
            if 2 <= chess.square_distance(square, defender_king) <= 3
        ]
        attacker_king = rng.choice(nearby)
        queen = rng.choice(list(chess.SQUARES))

        if len({defender_king, attacker_king, queen}) != 3:
            continue

        board = chess.Board(None)
        board.turn = attacker
        board.set_piece_at(
            defender_king, chess.Piece(chess.KING, not attacker)
        )
        board.set_piece_at(
            attacker_king, chess.Piece(chess.KING, attacker)
        )
        board.set_piece_at(
            queen, chess.Piece(chess.QUEEN, attacker)
        )

        if not board.is_valid() or board.is_game_over():
            continue

        key = board.board_fen() + (" w" if attacker else " b")
        if key in seen:
            continue

        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                seen.add(key)
                positions.append(board)
                break

    if len(positions) != count:
        raise ValueError(
            f"Generated only {len(positions)} of {count} "
            f"requested positions."
        )

    return positions