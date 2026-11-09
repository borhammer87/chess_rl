"""Static mate-Q diagnostics; Q estimates are not observed returns."""

import argparse
import csv
import hashlib
import json
import math

import chess
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
from chess_rl.training.episodes import STEP_PENALTY, material_reward, reward_for_color
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import get_legal_actions
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
    return category, env.board.copy(stack=True), reward, info


def expected_alternative_target(board, move, target_net, gamma):
    """Exact expectation under uniform RandomAgent, outside artificial horizons.

    Accept an already selected move; never evaluate current-policy Q values.
    Immediate terminals have zero opponent replies and no reply fraction.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    category, after_move, reward, _ = move_outcome(board, move)
    replies = list(after_move.legal_moves) if category == "non_terminal" else []
    outcomes = [move_outcome(after_move, reply)[:3] for reply in replies]
    if not replies:
        outcomes = [(category, after_move, reward)]
    targets, terminal_count = [], 0
    rewards, bootstraps = [], []
    was_training = target_net.training
    try:
        target_net.eval()
        with torch.no_grad():
            for category, next_board, reward in outcomes:
                done = category != "non_terminal"
                terminal_count += int(done)
                target = reward_for_color(reward, board.turn)
                target += material_reward(board, next_board, board.turn)
                immediate_reward, bootstrap = target, 0.0
                if not done:
                    state = encode_board(next_board).to(next(target_net.parameters()).device)
                    future = target_net.evaluate_legal_action_maxes(
                        state.unsqueeze(0), [get_legal_actions(next_board)]
                    )[0].item()
                    bootstrap = gamma * future
                    immediate_reward += STEP_PENALTY
                    target += STEP_PENALTY + bootstrap
                targets.append(target)
                rewards.append(immediate_reward)
                bootstraps.append(bootstrap)
    finally:
        target_net.train(was_training)
    return {
        "alternative_expected_target": sum(targets) / len(targets),
        "alternative_expected_reward": sum(rewards) / len(rewards),
        "alternative_expected_bootstrap": sum(bootstraps) / len(bootstraps),
        "opponent_reply_count": len(replies),
        "opponent_terminal_fraction": terminal_count / len(replies) if replies else None,
    }


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


def select_positions(positions, sample_size=25):
    """Stable hash order, first occurrence per key; None selects all."""
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive or None")
    unique = {}
    for board in positions:
        unique.setdefault(position_key(board), board)
    def rank(key):
        serialized = json.dumps(key, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), serialized
    return [unique[key] for key in sorted(unique, key=rank)[:sample_size]]


def enrich_existing_detail(path, seeded, games, target_net, gamma, sample_size=25):
    """Reuse recorded Q values; reconstruct history from original inputs.

    The CSV does not certify checkpoint identity. The caller must supply the
    checkpoint that produced it. No policy network is accepted or evaluated.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {}
    for row in rows:
        key = (row["dataset"], row["fen"])
        if key[0] not in ("seeded", "games") or key in indexed:
            raise ValueError("Unknown dataset or duplicate detail position")
        indexed[key] = row
    result = []
    for label, positions in (("seeded", seeded), ("games", games)):
        # Match only positions included in the earlier mate diagnostic.
        unique = {}
        for board in positions:
            unique.setdefault(position_key(board), board)
        by_fen = {}
        for board in unique.values():
            by_fen.setdefault(board.fen(), []).append(board)
        matched = []
        for dataset, fen in indexed:
            if dataset != label:
                continue
            candidates = by_fen.get(fen, [])
            if not candidates:
                raise ValueError(f"Detail FEN missing from {label} first occurrences: {fen}")
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous detail FEN in {label}: {fen}")
            matched.append(candidates[0])
        selected = select_positions(matched, sample_size)
        for board in selected:
            record = dict(indexed[(label, board.fen())])
            # Retain all existing fields verbatim, including recorded Q values.
            record.update(alternative_expected_target=None,
                          alternative_expected_reward=None,
                          alternative_expected_bootstrap=None,
                          alternative_q_minus_target=None,
                          opponent_reply_count=None, opponent_terminal_fraction=None)
            uci = record["alternative_move"]
            if not uci:
                if record["alternative_category"] != "no_alternative" or record["alternative_q"]:
                    raise ValueError("Inconsistent missing alternative")
            else:
                move = chess.Move.from_uci(uci)
                category, after, _, _ = move_outcome(board, move)
                if category != record["alternative_category"] or after.is_checkmate():
                    raise ValueError("Detail alternative classification mismatch")
                q = float(record["alternative_q"])
                if not math.isfinite(q):
                    raise ValueError("Alternative Q must be finite")
                record.update(expected_alternative_target(board, move, target_net, gamma))
                record["alternative_q_minus_target"] = q - record["alternative_expected_target"]
            result.append(record)
    return result


def print_summary(records, counts, overlap):
    """Print aggregate estimates, never implying Q is an observed return."""
    print("Q values are network estimates, not observed returns.")
    print("Deduplication ignores history/counters; first occurrence retained.")
    print(f"Overlap: {overlap} unique analyzed mate positions")
    for label, count in counts.items():
        rows = [r for r in records if r["dataset"] == label]
        if "sampled" in count:
            print(f"{label}: sampled={count['sampled']}")
        else:
            print(f"{label}: input={count['total']} duplicates={count['duplicates']} "
                  f"skipped={count['skipped']} analyzed={count['analyzed']} "
                  f"exclusive={count['analyzed'] - overlap}")
        for category in ("all", *CATEGORIES):
            group = rows if category == "all" else [
                r for r in rows if r["alternative_category"] == category
            ]
            def mean(field):
                values = [float(r[field]) for r in group if r.get(field) not in (None, "")]
                return f"{sum(values) / len(values):.6f}(n={len(values)})" if values else "N/A(n=0)"
            margins = [float(r["q_margin"]) for r in group if r.get("q_margin") not in (None, "")]
            signs = [sum(v > 0 for v in margins), sum(v == 0 for v in margins),
                     sum(v < 0 for v in margins)]
            print(f"  {category}: n={len(group)} mates={sum(r.get('mate_selected') in (True, 'True') for r in group)} "
                  f"Qmate={mean('mate_q')} Qalt={mean('alternative_q')} "
                  f"margin={mean('q_margin')} signs(+/0/-)={signs} "
                  f"error={mean('seeded_error')} MAE={mean('seeded_absolute_error')} "
                  f"game_reward={mean('game_mate_reward')}")
            if any("alternative_expected_target" in r for r in records):
                residuals = [r["alternative_q_minus_target"] for r in group
                             if r["alternative_q_minus_target"] is not None]
                signs = [sum(v > 0 for v in residuals), sum(v == 0 for v in residuals),
                         sum(v < 0 for v in residuals)]
                print(f"    E[y]={mean('alternative_expected_target')} "
                      f"Qalt-E[y]={mean('alternative_q_minus_target')} "
                      f"signs(+/0/-)={signs} "
                      f"mean_reply_terminal_fraction={mean('opponent_terminal_fraction')}")



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
    parser.add_argument("--existing-detail", type=Path, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--sample-size", type=int, default=25,
                        help="Positions per dataset; 0 selects all")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not 0 <= args.gamma <= 1 or args.sample_size < 0:
        parser.error("gamma must be in [0, 1] and sample-size must be non-negative")
    for path in [args.checkpoint, args.existing_detail, *args.seed_csv]:
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if not args.games_directory.is_dir() or not any(args.games_directory.glob("*.pgn")):
        parser.error("Games directory must contain PGN files")
    if args.output_csv and args.output_csv.resolve() in {
        p.resolve() for p in [args.checkpoint, args.existing_detail, *args.seed_csv, *args.games_directory.glob("*.pgn")]
    }:
        parser.error("Output must differ from input files")
    agent = StateActionDQNAgent(spatial=True, device=args.device)
    load_training_checkpoint(str(args.checkpoint), agent, ReplayBuffer(capacity=10_000))
    try:
        records = enrich_existing_detail(
            args.existing_detail,
            chain.from_iterable(seeded_positions(path) for path in args.seed_csv),
            game_positions(args.games_directory), agent.target_net, args.gamma,
            sample_size=args.sample_size or None,
        )
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Checkpoint: {args.checkpoint} | existing detail: {args.existing_detail}")
    print(f"gamma={args.gamma} (explicit CLI value, not stored in checkpoint)")
    print("Uniform RandomAgent expectation; artificial truncation excluded.")
    print("CSV checkpoint identity is user-supplied, not independently verified.")
    print(f"SHA-256 sample per dataset: {args.sample_size or 'all'}; first occurrence retained.")
    keys = {label: {position_key(chess.Board(r["fen"])) for r in records
                    if r["dataset"] == label} for label in ("seeded", "games")}
    counts = {label: {"sampled": len(value)} for label, value in keys.items()}
    print_summary(records, counts, len(keys["seeded"] & keys["games"]))
    if args.output_csv:
        export_csv(args.output_csv, records)
        print(f"Detail CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
