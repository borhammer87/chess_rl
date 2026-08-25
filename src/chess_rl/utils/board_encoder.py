import chess
import numpy as np
import torch


BOARD_CHANNELS = 18

WHITE_KINGSIDE_CASTLING_CHANNEL = 12
WHITE_QUEENSIDE_CASTLING_CHANNEL = 13
BLACK_KINGSIDE_CASTLING_CHANNEL = 14
BLACK_QUEENSIDE_CASTLING_CHANNEL = 15
EN_PASSANT_CHANNEL = 16
TURN_CHANNEL = 17


PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,

    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode the current chess position as an 18x8x8 tensor.

    Channels 0-11 contain piece locations.

    Channels 12-15 contain castling rights.

    Channel 16 contains the en passant target square.

    Channel 17 contains the side to move:
    all ones for White and all zeros for Black.
    """
    tensor = np.zeros(
        (8, 8, BOARD_CHANNELS),
        dtype=np.float32,
    )

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        channel = PIECE_TO_CHANNEL[
            (piece.piece_type, piece.color)
        ]

        tensor[row, col, channel] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[
            :, :, WHITE_KINGSIDE_CASTLING_CHANNEL
        ] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[
            :, :, WHITE_QUEENSIDE_CASTLING_CHANNEL
        ] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[
            :, :, BLACK_KINGSIDE_CASTLING_CHANNEL
        ] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[
            :, :, BLACK_QUEENSIDE_CASTLING_CHANNEL
        ] = 1.0

    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)

        tensor[
            row,
            col,
            EN_PASSANT_CHANNEL,
        ] = 1.0

    if board.turn == chess.WHITE:
        tensor[:, :, TURN_CHANNEL] = 1.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    return torch.tensor(
        tensor,
        dtype=torch.float32,
    )