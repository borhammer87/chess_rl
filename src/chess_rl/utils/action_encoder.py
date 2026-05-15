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