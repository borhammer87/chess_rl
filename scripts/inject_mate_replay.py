"""Fine-tune a complete spatial checkpoint with verified mate-in-one examples."""
import argparse
import csv
import random
from pathlib import Path

import chess
import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.checkpoint import (load_checkpoint_metadata, load_training_checkpoint, save_training_checkpoint)
from chess_rl.utils.replay_buffer import ReplayBuffer
from scripts.run_mate_in_one_probe import build_mate_transitions


def load_mate_examples(path: Path):
    """Validate CSV labels against chess rules and deduplicate positions."""
    examples = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            board = chess.Board(row["fen"])
            key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
            if key in seen:
                continue
            if not board.is_valid() or board.is_game_over():
                raise ValueError(f"Invalid mate position: {row['fen']}")
            transitions = build_mate_transitions(board)
            actual = {move.uci() for move in board.legal_moves if _is_mate(board, move)}
            claimed = set(row["mate_moves"].split())
            if not transitions or actual != claimed:
                raise ValueError(f"Incorrect mate labels: {row['fen']}")
            examples.extend(transitions)
            seen.add(key)
    if not examples:
        raise ValueError("No mate examples found")
    return examples, len(seen)


def _is_mate(board, move):
    board.push(move)
    result = board.is_checkmate()
    board.pop()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=54321)
    args = parser.parse_args(argv)
    if args.updates < 1 or args.output == args.checkpoint:
        parser.error("Updates must be positive and output must differ from input")
    if load_checkpoint_metadata(str(args.checkpoint)).get("architecture") != "spatial":
        parser.error("Expected spatial checkpoint")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = StateActionDQNAgent(spatial=True, device=args.device)
    replay = ReplayBuffer(capacity=10_000)
    load_training_checkpoint(str(args.checkpoint), agent, replay)
    original = list(replay.buffer)
    mates, unique = load_mate_examples(args.csv)
    if len(original) < 29 or len(mates) < 3:
        parser.error("Need at least 29 original and 3 mate transitions")
    for transition in mates:
        replay.push(transition.state, transition.action, transition.reward,
                    transition.next_state, transition.done, transition.next_legal_actions)
    losses = []
    for _ in range(args.updates):
        batch = random.sample(original, 29) + random.sample(mates, 3)
        random.shuffle(batch)
        losses.append(agent.train_step(batch))
    agent.update_target()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(load_checkpoint_metadata(str(args.checkpoint)))
    metadata["mate_finetune"] = {"source": str(args.csv), "unique_positions": unique,
                                 "transitions": len(mates), "updates": args.updates}
    save_training_checkpoint(str(args.output), agent, replay, metadata)
    print(f"Original replay: {len(original)} | unique positions: {unique} | mate transitions: {len(mates)}")
    print(f"Updates: {args.updates} | loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
