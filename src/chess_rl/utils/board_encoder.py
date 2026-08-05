import chess
import numpy as np
import torch


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
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]

        tensor[row, col, channel] = 1.0

    """return tensor"""
    tensor = np.transpose(tensor, (2, 0, 1))

    return torch.tensor(
        tensor,
        dtype=torch.float32
    )