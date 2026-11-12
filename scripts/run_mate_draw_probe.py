"""Identify KQK mate/stalemate pairs without networks or training."""
import argparse

import chess

from chess_rl.training.mate_positions import generate_mate_positions


def find_terminal_pair(board):
    """List all mating/stalemating moves and select the first UCI of each."""
    if not board.is_valid() or board.is_game_over():
        return None
    mates, stalemates = [], []
    for move in sorted(board.legal_moves, key=lambda move: move.uci()):
        successor = board.copy(stack=True)
        successor.push(move)
        if successor.is_checkmate():
            mates.append(move)
        elif successor.is_stalemate():
            stalemates.append(move)
    if not mates or not stalemates:
        return None
    return dict(board=board, mate_moves=mates, stalemate_moves=stalemates,
                mate_move=mates[0], stalemate_move=stalemates[0])


def generate_terminal_pairs(candidate_count, seed=12345):
    """Filter the existing generator; count means candidates, not pairs."""
    candidates = generate_mate_positions(candidate_count, seed=seed)
    pairs = [pair for board in candidates
             if (pair := find_terminal_pair(board)) is not None]
    return pairs, dict(candidates=len(candidates), pairs=len(pairs),
                       white=sum(pair["board"].turn == chess.WHITE for pair in pairs),
                       black=sum(pair["board"].turn == chess.BLACK for pair in pairs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    try:
        _, counts = generate_terminal_pairs(args.candidates, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f"seed={args.seed} selection=lexicographic UCI")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    if not counts["pairs"]:
        print("No mate/stalemate pairs found among the candidates.")


if __name__ == "__main__":
    main()
