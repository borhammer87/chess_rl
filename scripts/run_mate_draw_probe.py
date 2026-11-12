"""Identify KQK mate/stalemate pairs without networks or training."""
import argparse
import hashlib
from pathlib import Path

import chess
import torch

from chess_rl.training.mate_positions import generate_mate_positions
from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import evaluate_state_action_greedy_choice


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


def canonical_kqk_key(board):
    """Grouping only: D4 symmetries, with pieces identified by attacker role."""
    queens = list(board.pieces(chess.QUEEN, board.turn))
    if (not board.is_valid() or len(board.piece_map()) != 3 or len(queens) != 1
            or board.castling_rights or board.ep_square is not None):
        raise ValueError("Expected valid KQK with attacker to move and no special rights")
    squares = (board.king(board.turn), queens[0], board.king(not board.turn))
    variants = [[] for _ in range(8)]
    for square in squares:
        x, y = chess.square_file(square), chess.square_rank(square)
        for variant, (a, b) in zip(variants, (
            (x, y), (7-y, x), (7-x, 7-y), (y, 7-x),
            (7-x, y), (x, 7-y), (y, x), (7-y, 7-x),
        )):
            variant.append(chess.square(a, b))
    return min(map(tuple, variants))


def split_terminal_pairs(pairs, seed=12345):
    """Reserve nearest 20% of families, at least one if two or more exist.

    Empty or single-family inputs have no evaluation set. Original records,
    boards and selected actions are retained, in input order within each set.
    """
    keys = [canonical_kqk_key(pair["board"]) for pair in pairs]
    families = sorted(set(keys), key=lambda key: (
        hashlib.sha256(f"{seed}:{key[0]},{key[1]},{key[2]}".encode("ascii")).digest(), key))
    count = max(1, (len(families) + 2) // 5) if len(families) >= 2 else 0
    evaluation_keys = set(families[:count])
    adjustment, evaluation = [], []
    for pair, key in zip(pairs, keys):
        (evaluation if key in evaluation_keys else adjustment).append(pair)
    return adjustment, evaluation


def load_policy(path):
    """Load only spatial policy weights on CPU without initializing RNG."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("metadata", {}).get("architecture") != "spatial":
        raise ValueError("Expected spatial checkpoint")
    with torch.device("meta"):
        policy = SpatialStateActionDQN()
    policy.load_state_dict(checkpoint["agent"]["policy_net"], strict=True, assign=True)
    return policy


def evaluate_terminal_pairs(pairs, policy):
    """Score the fixed pair and separately evaluate greedy over all legal moves."""
    values = {key: [] for key in ("q_mate", "q_stalemate", "mae_mate", "mae_stalemate", "margin")}
    counts = dict(positive=0, negative=0, zero=0, greedy_mate=0)
    modes = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        with torch.no_grad():
            for pair in pairs:
                board = pair["board"]
                legal = list(board.legal_moves)
                selected = (pair["mate_move"], pair["stalemate_move"])
                if any(move not in legal for move in selected):
                    raise ValueError("Pair actions must be legal")
                state = encode_board(board).to(next(policy.parameters()).device)
                q_mate, q_draw = policy.evaluate_action_ids(
                    state, [encode_move(move) for move in selected]).tolist()
                action, _, _ = evaluate_state_action_greedy_choice(policy, state, legal)
                margin = q_mate - q_draw
                for key, value in zip(values, (q_mate, q_draw, abs(q_mate - 1), abs(q_draw), margin)):
                    values[key].append(value)
                counts["positive" if margin > 0 else "negative" if margin < 0 else "zero"] += 1
                counts["greedy_mate"] += int(action in {encode_move(m) for m in pair["mate_moves"]})
    finally:
        for module, training in modes:
            module.training = training
    n = len(pairs)
    return dict(positions=n, families=len({canonical_kqk_key(p["board"]) for p in pairs}),
                **{key: sum(items) / n if n else None for key, items in values.items()},
                **counts, **{key + "_percent": 100 * count / n if n else None
                             for key, count in counts.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        pairs, counts = generate_terminal_pairs(args.candidates, args.seed)
        adjustment, evaluation = split_terminal_pairs(pairs, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(f"seed={args.seed} selection=lexicographic UCI")
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    if not counts["pairs"]:
        print("No mate/stalemate pairs found among the candidates.")
    for name, subset in (("adjustment", adjustment), ("evaluation", evaluation)):
        families = len({canonical_kqk_key(pair["board"]) for pair in subset})
        white = sum(pair["board"].turn == chess.WHITE for pair in subset)
        print(f"{name}: families={families} positions={len(subset)} "
              f"white={white} black={len(subset)-white}")
    if not evaluation:
        print("No family-disjoint evaluation possible: fewer than two families.")
    if args.checkpoint is not None:
        policy = load_policy(args.checkpoint)
        print(f"checkpoint={args.checkpoint} device=cpu")
        print("Qmate/Qstalemate refer to the fixed UCI pair; greedy considers all legal actions.")
        print("Family separation does not establish independence from original training.")
        for name, subset in (("adjustment", adjustment), ("evaluation", evaluation)):
            metrics = evaluate_terminal_pairs(subset, policy)
            print(name + ": " + " ".join(
                f"{key}={value:.6f}" if isinstance(value, float) else
                f"{key}={value}" if value is not None else f"{key}=N/A"
                for key, value in metrics.items()))


if __name__ == "__main__":
    main()
