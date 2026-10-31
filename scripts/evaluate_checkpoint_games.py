
"""Save three greedy games and audit mate-in-one opportunities."""

import argparse
import random
from pathlib import Path

import chess
import chess.pgn

from chess_rl.agents.random_agent import RandomAgent
from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.checkpoint import (
    load_checkpoint_metadata,
    load_training_checkpoint,
)
from chess_rl.training.train_dqn import save_greedy_evaluation_game
from chess_rl.utils.replay_buffer import ReplayBuffer


def audit_mates(path: Path, agent_color: chess.Color) -> tuple[int, int]:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)

    if game is None:
        raise ValueError(f"Empty PGN: {path}")

    board = game.board()
    opportunities = converted = 0

    for move in game.mainline_moves():
        if board.turn == agent_color:
            mates = []

            for candidate in list(board.legal_moves):
                board.push(candidate)
                is_mate = board.is_checkmate()
                board.pop()

                if is_mate:
                    mates.append(candidate)

            if mates:
                opportunities += 1
                converted += move in mates

                print(
                    f"  Mate available at {board.fullmove_number}: "
                    f"{', '.join(m.uci() for m in mates)}; "
                    f"played {move.uci()}; FEN {board.fen()}"
                )

        board.push(move)

    return opportunities, converted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/checkpoint_games"),
    )
    parser.add_argument("--seed", type=int, default=54321)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    metadata = load_checkpoint_metadata(str(args.checkpoint))

    if metadata.get("architecture") != "spatial":
        parser.error("Expected a spatial State-Action checkpoint")

    print(
        f"Checkpoint: {args.checkpoint} "
        f"| episodes: {metadata.get('episodes')}"
    )

    agent = StateActionDQNAgent(
        spatial=True,
        device=args.device,
    )

    load_training_checkpoint(
        str(args.checkpoint),
        agent,
        ReplayBuffer(capacity=10_000),
    )

    agent.policy_net.eval()
    random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    total_opportunities = 0
    total_converted = 0

    for number in range(1, 51):
        color = chess.WHITE if number % 2 else chess.BLACK
        path = args.output / f"game_{number}.pgn"

        result = save_greedy_evaluation_game(
            env=ChessEnv(),
            agent=agent,
            opponent=RandomAgent(),
            path=path,
            max_agent_steps=150,
            agent_color=color,
        )

        opportunities, converted = audit_mates(path, color)

        total_opportunities += opportunities
        total_converted += converted

        print(
            f"Game {number}: {path} "
            f"| agent={'White' if color else 'Black'} "
            f"| plies={result.total_plies} "
            f"| truncated={result.truncated} "
            f"| mate chances={opportunities} "
            f"| converted={converted}"
        )

    print(
        f"TOTAL: 50 games "
        f"| mate chances={total_opportunities} "
        f"| converted={total_converted} "
        f"| missed={total_opportunities - total_converted}"
    )


if __name__ == "__main__":
    main()
