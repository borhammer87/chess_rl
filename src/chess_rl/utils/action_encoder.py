import chess

ACTION_SIZE = 4096


def encode_move(move: chess.Move) -> int:
    """
    Converts a chess move into a unique action index.

    Formula:
        action = from_square * 64 + to_square

    Total action space:
        64 * 64 = 4096
    """

    return move.from_square * 64 + move.to_square


def decode_move(action: int) -> chess.Move:
    """
    Converts an action index back into a chess move.
    """

    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(f"Invalid action index: {action}")

    from_square = action // 64
    to_square = action % 64

    return chess.Move(from_square, to_square)


def get_legal_actions(board: chess.Board) -> list[int]:
    """
    Returns all legal action indices for the current board.
    """

    return [
        encode_move(move)
        for move in board.legal_moves
    ]


def action_to_uci(action: int) -> str:
    """
    Converts an action index into UCI notation.
    """

    return decode_move(action).uci()


def uci_to_action(uci_move: str) -> int:
    """
    Converts a UCI move string into an action index.
    """

    move = chess.Move.from_uci(uci_move)

    return encode_move(move)


def decode_legal_action(
    action: int,
    legal_moves: list[chess.Move],
    ) -> chess.Move:
    """
    Convert an encoded action into a legal chess move.

    Several promotion moves may share the same action because the
    4096-action encoding only stores the origin and destination
    squares. In that case, queen promotion is preferred.
    """
    matching_moves = [
        move
        for move in legal_moves
        if encode_move(move) == action
    ]

    if not matching_moves:
        decoded_move = decode_move(action)

        raise ValueError(
            f"Action {action} ({decoded_move.uci()}) "
            "does not correspond to any legal move."
        )

    if len(matching_moves) == 1:
        return matching_moves[0]

    queen_promotions = [
        move
        for move in matching_moves
        if move.promotion == chess.QUEEN
    ]

    if queen_promotions:
        return queen_promotions[0]

    return matching_moves[0]