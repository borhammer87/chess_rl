import chess


BASE_ACTION_SIZE = 4096

PROMOTION_PIECES = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)


def _build_promotion_pairs() -> list[tuple[int, int]]:
    """
    Build all possible promotion origin/destination square pairs.

    White promotes from rank 7 to rank 8.
    Black promotes from rank 2 to rank 1.

    For each origin square, a pawn can move straight ahead or
    capture one file to either side when that destination exists.
    """
    pairs: list[tuple[int, int]] = []

    for from_rank, to_rank in (
        (6, 7),
        (1, 0),
    ):
        for from_file in range(8):
            from_square = chess.square(
                from_file,
                from_rank,
            )

            for file_offset in (-1, 0, 1):
                to_file = from_file + file_offset

                if not 0 <= to_file < 8:
                    continue

                to_square = chess.square(
                    to_file,
                    to_rank,
                )

                pairs.append(
                    (
                        from_square,
                        to_square,
                    )
                )

    return pairs


PROMOTION_PAIRS = _build_promotion_pairs()

PROMOTION_PAIR_TO_INDEX = {
    pair: index
    for index, pair in enumerate(PROMOTION_PAIRS)
}

PROMOTION_ACTION_COUNT = (
    len(PROMOTION_PAIRS)
    * len(PROMOTION_PIECES)
)

ACTION_SIZE = (
    BASE_ACTION_SIZE
    + PROMOTION_ACTION_COUNT
)


def encode_move(move: chess.Move) -> int:
    """
    Convert a chess move into a unique action index.

    Non-promotion moves use the original 4096-action encoding:

        from_square * 64 + to_square

    Promotion moves use dedicated actions after the base action space
    so queen, rook, bishop, and knight promotions remain distinct.
    """
    if move.promotion is None:
        return (
            move.from_square * 64
            + move.to_square
        )

    pair = (
        move.from_square,
        move.to_square,
    )

    if pair not in PROMOTION_PAIR_TO_INDEX:
        raise ValueError(
            "Invalid promotion move squares."
        )

    if move.promotion not in PROMOTION_PIECES:
        raise ValueError(
            "Unsupported promotion piece."
        )

    pair_index = PROMOTION_PAIR_TO_INDEX[pair]
    promotion_index = PROMOTION_PIECES.index(
        move.promotion
    )

    return (
        BASE_ACTION_SIZE
        + pair_index * len(PROMOTION_PIECES)
        + promotion_index
    )


def decode_move(action: int) -> chess.Move:
    """
    Convert an action index back into a chess move.
    """
    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(
            f"Invalid action index: {action}"
        )

    if action < BASE_ACTION_SIZE:
        from_square = action // 64
        to_square = action % 64

        return chess.Move(
            from_square,
            to_square,
        )

    promotion_action = (
        action - BASE_ACTION_SIZE
    )

    pair_index = (
        promotion_action
        // len(PROMOTION_PIECES)
    )

    promotion_index = (
        promotion_action
        % len(PROMOTION_PIECES)
    )

    from_square, to_square = (
        PROMOTION_PAIRS[pair_index]
    )

    promotion_piece = (
        PROMOTION_PIECES[promotion_index]
    )

    return chess.Move(
        from_square,
        to_square,
        promotion=promotion_piece,
    )


def get_legal_actions(
    board: chess.Board,
) -> list[int]:
    """
    Return all legal action indices for the current board.
    """
    return [
        encode_move(move)
        for move in board.legal_moves
    ]


def action_to_uci(action: int) -> str:
    """
    Convert an action index into UCI notation.
    """
    return decode_move(action).uci()


def uci_to_action(uci_move: str) -> int:
    """
    Convert a UCI move string into an action index.
    """
    move = chess.Move.from_uci(
        uci_move
    )

    return encode_move(move)


def decode_legal_action(
    action: int,
    legal_moves: list[chess.Move],
) -> chess.Move:
    """
    Convert an encoded action into a legal chess move.
    """
    decoded_move = decode_move(action)

    if decoded_move not in legal_moves:
        raise ValueError(
            f"Action {action} "
            f"({decoded_move.uci()}) "
            "does not correspond to any legal move."
        )

    return decoded_move