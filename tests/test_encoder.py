import chess
from chess_rl.utils.board_encoder import encode_board

board = chess.Board()

tensor = encode_board(board)

print(tensor.shape)
print(tensor.sum())