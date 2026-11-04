"""Identify replay content compatible with generated mate examples, not provenance."""
from collections import defaultdict
import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.inject_mate_replay import load_mate_examples
from scripts.run_state_action_probe import (
    TerminalTransitionTracker, calculate_replay_learning_signal,
)
from scripts.diagnose_mate_q import CATEGORIES, print_summary
from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN
from chess_rl.utils.replay_buffer import ReplayBuffer


def _transition_key(transition):
    """Use complete encoded content, including tensor shape and dtype."""
    def tensor_key(tensor):
        return (tuple(tensor.shape), str(tensor.dtype),
                tensor.detach().cpu().contiguous().numpy().tobytes())

    return (
        tensor_key(transition.state), transition.action, transition.reward,
        tensor_key(transition.next_state), transition.done,
        tuple(transition.next_legal_actions),
    )


def match_mate_replay(csv_path, replay_buffer):
    """Return all content matches and duplicate groups, using zero-based indices.

    CSV indices refer to transitions returned by load_mate_examples, which
    already deduplicates CSV positions. Multiple matches are retained rather
    than assigned an arbitrary one-to-one correspondence. Compatibility does
    not establish whether a replay transition was originally preloaded.
    """
    examples, unique_positions = load_mate_examples(csv_path)
    example_groups = defaultdict(list)
    replay_groups = defaultdict(list)
    for index, transition in enumerate(examples):
        example_groups[_transition_key(transition)].append(index)
    for index, transition in enumerate(replay_buffer.buffer):
        replay_groups[_transition_key(transition)].append(index)

    example_matches = [[] for _ in examples]
    replay_matches = [[] for _ in replay_buffer.buffer]
    for key, indices in example_groups.items():
        matches = replay_groups.get(key, [])
        for index in indices:
            example_matches[index] = list(matches)
        for index in matches:
            replay_matches[index] = list(indices)

    matched = sum(bool(matches) for matches in example_matches)
    return {
        "unique_csv_positions": unique_positions,
        "generated_count": len(examples),
        "matched_count": matched,
        "missing_count": len(examples) - matched,
        "compatible_replay_indices": [
            i for i, matches in enumerate(replay_matches) if matches
        ],
        "compatible_replay_count": sum(bool(matches) for matches in replay_matches),
        "example_matches": example_matches,
        "replay_matches": replay_matches,
        "example_duplicate_groups": [
            indices for indices in example_groups.values() if len(indices) > 1
        ],
        "replay_duplicate_groups": [
            indices for indices in replay_groups.values() if len(indices) > 1
        ],
        "ambiguous_example_indices": [
            i for i, matches in enumerate(example_matches) if len(matches) > 1
        ],
        "ambiguous_replay_indices": [
            i for i, matches in enumerate(replay_matches) if len(matches) > 1
        ],
    }


def summarize_replay(agent, replay, tracker, mate_csv):
    """Compute TD once and partition every retained transition exactly once."""
    if agent.gamma != 0.99:
        raise ValueError("This audit requires explicit gamma=0.99")
    labels = tracker.get_final_labels()
    matches = match_mate_replay(mate_csv, replay)
    compatible = set(matches["compatible_replay_indices"])
    groups = {name: [] for name in (
        "compatible_with_seeded_mate", "other_win", "draw", "loss",
        "truncated", "non_terminal",
    )}
    if not compatible.issubset(range(len(replay.buffer))):
        raise ValueError("Compatible replay index out of range")
    for index, transition in enumerate(replay.buffer):
        label = labels.get(id(transition), "non_terminal")
        if label not in ("win", "draw", "loss", "truncated", "non_terminal"):
            raise ValueError("Unknown tracker category")
        if label != "truncated" and transition.done != (label != "non_terminal"):
            raise ValueError("Tracker category and terminal flag disagree")
        if index in compatible:
            if label != "win" or not transition.done:
                raise ValueError("Compatible mate must be a terminal win")
            category = "compatible_with_seeded_mate"
        else:
            category = "other_win" if label == "win" else label
        groups[category].append(index)
    if sum(map(len, groups.values())) != len(replay.buffer):
        raise ValueError("Replay categories do not partition the replay")
    signal = calculate_replay_learning_signal(agent, replay, terminal_labels=labels)
    return {
        name: {"count": len(indices), **{
            field: signal[field][indices].mean().item() if indices else None
            for field in ("q", "target", "td_error", "abs_td_error")
        }} for name, indices in groups.items()
    }


def read_evaluation_detail(path):
    """Read cached estimates only; validate the fields used by print_summary."""
    numeric = ("mate_q", "alternative_q", "q_margin", "seeded_target",
               "seeded_error", "seeded_absolute_error", "game_mate_reward")
    required = {"fen", "dataset", "mate_selected", "overlap",
                "alternative_category", *numeric}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing evaluation CSV columns")
        rows, seen = [], set()
        for original in reader:
            row = {key: original[key] for key in required}
            key = (row["dataset"], row["fen"])
            if row["dataset"] not in ("seeded", "games") or not row["fen"] or key in seen:
                raise ValueError("Invalid dataset, FEN or duplicate evaluation row")
            seen.add(key)
            if any(row[field] not in ("True", "False") for field in ("mate_selected", "overlap")):
                raise ValueError("Invalid evaluation boolean")
            if row["alternative_category"] not in CATEGORIES:
                raise ValueError("Invalid alternative category")
            for field in numeric:
                if field in ("alternative_q", "q_margin") and row["alternative_category"] == "no_alternative":
                    if row[field] != "":
                        raise ValueError("Unexpected value without alternative")
                    continue
                try:
                    value = float(row[field])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid numeric column: {field}") from error
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite numeric column: {field}")
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mate-csv", type=Path, required=True)
    parser.add_argument("--existing-detail", type=Path, required=True)
    parser.add_argument("--gamma", type=float, choices=[0.99], required=True)
    args = parser.parse_args(argv)
    rows = read_evaluation_detail(args.existing_detail)
    counts = {}
    overlaps = []
    for dataset in ("seeded", "games"):
        selected = [row for row in rows if row["dataset"] == dataset]
        counts[dataset] = dict(total=len(selected), analyzed=len(selected), duplicates=0, skipped=0)
        overlaps.append(sum(row["overlap"] == "True" for row in selected))
    if overlaps[0] != overlaps[1]:
        raise ValueError("Inconsistent cached overlap counts")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["metadata"]["architecture"] != "spatial":
        raise ValueError("Expected spatial checkpoint")
    # Meta construction loads saved weights without consuming initialization RNG.
    with torch.device("meta"):
        policy, target = SpatialStateActionDQN(), SpatialStateActionDQN()
    policy.load_state_dict(checkpoint["agent"]["policy_net"], assign=True)
    target.load_state_dict(checkpoint["agent"]["target_net"], assign=True)
    policy.eval()
    target.eval()
    agent = SimpleNamespace(policy_net=policy, target_net=target,
                            device=torch.device("cpu"), gamma=args.gamma)
    replay = ReplayBuffer(capacity=1)
    replay.load_state_dict(checkpoint["replay_buffer"])
    tracker = TerminalTransitionTracker(replay)
    tracker.load_state_dict(checkpoint["metadata"]["terminal_tracker"])
    summary = summarize_replay(agent, replay, tracker, args.mate_csv)
    print(f"Replay: gamma={args.gamma}; Q is for each stored action; TD = target - Q.")
    print("compatible_with_seeded_mate = compatible con mate precargado; not proven provenance.")
    for name, values in summary.items():
        metrics = " ".join(f"{key}={value:.6f}" if value is not None else f"{key}=N/A"
                           for key, value in values.items() if key != "count")
        print(f"{name}: n={values['count']} {metrics}")
    print("Evaluation CSV: Qmate is the maximum over mating actions per position.")
    print("Evaluation error = Qmate - seeded target (+1); game reward is separate.")
    print("Counts refer to cached CSV rows. Cached overlap is between evaluation sets only.")
    print("Independence from training/replay has not been verified; cached Q is not recomputed.")
    print_summary(rows, counts, overlaps[0])


if __name__ == "__main__":
    main()
