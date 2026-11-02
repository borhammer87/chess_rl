
"""Compare mate-in-one decisions on seeded and game positions."""

import argparse
import csv
from pathlib import Path

import chess
import chess.pgn

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.checkpoint import load_training_checkpoint
from chess_rl.utils.replay_buffer import ReplayBuffer
from scripts.run_mate_in_one_probe import (
    evaluate_mate_position,
    find_mate_in_one_moves,
)


def position_key(board):
    return (
        board.board_fen(),
        board.turn,
        board.castling_rights,
        board.ep_square,
    )


def seeded_positions(path):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield chess.Board(row["fen"])


def game_positions(directory):
    for path in sorted(directory.glob("*.pgn")):
        with path.open(encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)

        if game is None:
            continue

        agent_color = chess.WHITE if game.headers.get("White") == "DQN" else chess.BLACK
        board = game.board()

        for move in game.mainline_moves():
            if board.turn == agent_color and find_mate_in_one_moves(board):
                yield board.copy()
            board.push(move)


def diagnose(network, positions):
    seen = set()
    opportunities = converted = 0
    advantages = []
    mate_values = []
    non_mate_values = []

    for board in positions:
        key = position_key(board)
        if key in seen:
            continue
        seen.add(key)

        if not find_mate_in_one_moves(board):
            continue

        _, correct, _, _, mate_q, non_mate_q, advantage = (
            evaluate_mate_position(board, network)
        )
        opportunities += 1
        converted += int(correct)

        if advantage is not None:
            advantages.append(advantage)
        if mate_q is not None:
            mate_values.append(mate_q)
        if non_mate_q is not None:
            non_mate_values.append(non_mate_q)

    return opportunities, converted, advantages, mate_values, non_mate_values

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("seed_csv", type=Path)
    parser.add_argument("games_directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    agent = StateActionDQNAgent(spatial=True, device=args.device)
    load_training_checkpoint(
        str(args.checkpoint), agent, ReplayBuffer(capacity=10_000)
    )
    agent.policy_net.eval()

    def mean(values):
        return sum(values) / len(values) if values else None

    for label, positions in (
        ("Seeded positions", seeded_positions(args.seed_csv)),
        ("Game positions", game_positions(args.games_directory)),
    ):
        total, correct, advantages, mate_values, non_mate_values = (
            diagnose(agent.policy_net, positions)
        )
        print(
            f"{label}: {correct}/{total} mates selected "
            f"| mean Q advantage: {mean(advantages)} "
            f"| mean mate Q: {mean(mate_values)} "
            f"| mean best non-mate Q: {mean(non_mate_values)}"
        )


if __name__ == "__main__":
    main()
