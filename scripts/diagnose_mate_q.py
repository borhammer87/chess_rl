"""Static mate-Q diagnostics; Q estimates are not observed returns."""

import argparse
import csv
from itertools import chain
from pathlib import Path

import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.checkpoint import load_training_checkpoint
from chess_rl.utils.replay_buffer import ReplayBuffer
from scripts.diagnose_mate_generalization import (
    game_positions, position_key, seeded_positions,
)

from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.episodes import material_reward, reward_for_color
from scripts.run_mate_in_one_probe import evaluate_mate_actions


def move_outcome(board, move):
    """Apply a move to an independent board, retaining repetition history."""
    env = ChessEnv()
    env.board = board.copy(stack=True)
    env.done = env.board.is_game_over()
    next_board, reward, done, info = env.step(move)
    category = (
        "non_terminal" if not done else
        "terminal_draw" if info["result"] == "1/2-1/2" else
        "other_terminal"
    )
    return category, next_board, reward, info


def analyze_position(board, network):
    """Return one mate-position record, or None when no mate is available.

    Errors use Q(mate) - 1, the exact seeded target. Game reward is
    reported separately and includes material shaping. No opponent reply
    or non-terminal Bellman target is evaluated.
    """
    if board.is_game_over():
        return None
    network.eval()
    with torch.no_grad():
        chosen, is_mate, _, _, mate, alternative = evaluate_mate_actions(
            board, network
        )
    if mate is None:
        return None

    mate_move, mate_q = mate
    _, next_board, reward, _ = move_outcome(board, mate_move)
    terminal_reward = reward_for_color(reward, board.turn)
    shaping = material_reward(board, next_board, board.turn)
    category, termination = "no_alternative", None
    alternative_move, alternative_q = None, None
    if alternative is not None:
        alternative_move, alternative_q = alternative
        category, _, _, info = move_outcome(board, alternative_move)
        termination = info["termination"]

    return {
        "fen": board.fen(),
        "chosen_move": chosen.uci(),
        "mate_selected": is_mate,
        "mate_move": mate_move.uci(),
        "mate_q": mate_q,
        "alternative_move": alternative_move.uci() if alternative_move else None,
        "alternative_q": alternative_q,
        "alternative_category": category,
        "alternative_termination": termination,
        "q_margin": mate_q - alternative_q if alternative_q is not None else None,
        "seeded_target": 1.0,
        "seeded_error": mate_q - 1.0,
        "seeded_absolute_error": abs(mate_q - 1.0),
        "game_terminal_reward": terminal_reward,
        "game_material_reward": shaping,
        "game_mate_reward": terminal_reward + shaping,
    }


CATEGORIES = ("non_terminal", "terminal_draw", "other_terminal", "no_alternative")


def analyze_sets(network, seeded, games):
    """Deduplicate by the existing key; keep each set's first occurrence."""
    records, counts, keys = [], {}, {}
    for label, positions in (("seeded", seeded), ("games", games)):
        seen, retained = set(), set()
        total = duplicates = skipped = 0
        for board in positions:
            total += 1
            key = position_key(board)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            record = analyze_position(board, network)
            if record is None:
                skipped += 1
                continue
            retained.add(key)
            records.append(dict(record, dataset=label, overlap=False))
        keys[label] = retained
        counts[label] = dict(total=total, duplicates=duplicates,
                             skipped=skipped, analyzed=len(retained))
    overlap = keys["seeded"] & keys["games"]
    # FEN retains the fields used by position_key; history is not part of it.
    import chess
    for record in records:
        record["overlap"] = position_key(chess.Board(record["fen"])) in overlap
    return records, counts, len(overlap)


def print_summary(records, counts, overlap):
    """Print aggregate estimates, never implying Q is an observed return."""
    print("Q values are network estimates, not observed returns.")
    print("Deduplication ignores history/counters; first occurrence retained.")
    print(f"Overlap: {overlap} unique analyzed mate positions")
    for label, count in counts.items():
        rows = [r for r in records if r["dataset"] == label]
        print(f"{label}: input={count['total']} duplicates={count['duplicates']} "
              f"skipped={count['skipped']} analyzed={count['analyzed']} "
              f"exclusive={count['analyzed'] - overlap}")
        for category in ("all", *CATEGORIES):
            group = rows if category == "all" else [
                r for r in rows if r["alternative_category"] == category
            ]
            def mean(field):
                values = [r[field] for r in group if r[field] is not None]
                return f"{sum(values) / len(values):.6f}(n={len(values)})" if values else "N/A(n=0)"
            margins = [r["q_margin"] for r in group if r["q_margin"] is not None]
            signs = [sum(v > 0 for v in margins), sum(v == 0 for v in margins),
                     sum(v < 0 for v in margins)]
            print(f"  {category}: n={len(group)} mates={sum(r['mate_selected'] for r in group)} "
                  f"Qmate={mean('mate_q')} Qalt={mean('alternative_q')} "
                  f"margin={mean('q_margin')} signs(+/0/-)={signs} "
                  f"error={mean('seeded_error')} MAE={mean('seeded_absolute_error')} "
                  f"game_reward={mean('game_mate_reward')}")


def export_csv(path, records):
    """Export per-position details; an empty result still gets a header."""
    fields = list(records[0]) if records else ["dataset", "fen", "overlap"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("games_directory", type=Path)
    parser.add_argument("--seed-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    for path in [args.checkpoint, *args.seed_csv]:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if not args.games_directory.is_dir() or not any(args.games_directory.glob("*.pgn")):
        parser.error("Games directory must contain PGN files")
    if args.output_csv and args.output_csv.resolve() in {
        p.resolve() for p in [args.checkpoint, *args.seed_csv, *args.games_directory.glob("*.pgn")]
    }:
        parser.error("Output must differ from input files")
    agent = StateActionDQNAgent(spatial=True, device=args.device)
    load_training_checkpoint(str(args.checkpoint), agent, ReplayBuffer(capacity=10_000))
    records, counts, overlap = analyze_sets(
        agent.policy_net,
        chain.from_iterable(seeded_positions(path) for path in args.seed_csv),
        game_positions(args.games_directory),
    )
    print_summary(records, counts, overlap)
    if args.output_csv:
        export_csv(args.output_csv, records)
        print(f"Detail CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
