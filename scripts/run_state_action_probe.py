import random
from time import perf_counter

import torch
import chess
import chess.pgn
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.train_dqn import (
    evaluate_against_random_both_colors,
    score_evaluation,
    summarize_training,
    train_against_random,
    save_greedy_evaluation_game,
    save_first_truncated_greedy_evaluation_game,

)
from pathlib import Path
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.action_selection import (
    evaluate_state_action_greedy_choice,
)
def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)

def analyze_greedy_pgn(
    path: Path,
    agent: StateActionDQNAgent,
    agent_color: chess.Color,
) -> None:
    """Compare recorded learner moves with reconstructed greedy choices."""
    with path.open(
        encoding="utf-8",
    ) as pgn_file:
        game = chess.pgn.read_game(
            pgn_file
        )

    if game is None:
        raise ValueError(
            "PGN does not contain a game."
        )

    board = game.board()
    learner_move_number = 0

    for move in game.mainline_moves():
        if board.turn == agent_color:
            learner_move_number += 1

            state = encode_board(
                board
            )
            legal_moves = list(
                board.legal_moves
            )

            (
                greedy_action,
                best_q,
                q_gap,
            ) = evaluate_state_action_greedy_choice(
                network=agent.policy_net,
                state=state,
                legal_moves=legal_moves,
            )

            recorded_action = encode_move(
                move
            )

            if greedy_action != recorded_action:
                raise RuntimeError(
                    "Reconstructed greedy action does not "
                    "match recorded PGN move "
                    f"at learner move {learner_move_number}: "
                    f"recorded={move.uci()}, "
                    f"greedy_action={greedy_action}."
                )

            q_gap_text = (
                f"{q_gap:.6f}"
                if q_gap is not None
                else "N/A"
            )

            print(
                f"{learner_move_number:3d}. "
                f"{move.uci()} "
                f"- Q: {best_q:.6f} "
                f"- gap: {q_gap_text}"
            )

        board.push(
            move
        )

def main() -> None:
    """
    Run a short CPU training probe for the State-Action DQN.

    The goal is to verify sustained end-to-end training and measure
    its wall-clock cost before deciding whether longer CPU training
    or explicit CUDA support should come next.
    """
    training_episodes = 100
    evaluation_episodes_per_color = 20
    max_agent_steps = 150
    batch_size = 32
    min_replay_size = 1_000
    target_update_frequency = 10


    initialization_seed = 12345
    evaluation_seed = 54321
    training_seed = 67890

    set_random_seed(initialization_seed)

    env = ChessEnv()

    agent = StateActionDQNAgent(
        epsilon=1.0,
    )

    opponent = RandomAgent()

    replay_buffer = ReplayBuffer(
        capacity=10_000,
    )

    print("State-Action DQN CPU probe")
    print(
        f"Training episodes: {training_episodes} "
        f"- max agent steps: {max_agent_steps} "
        f"- batch size: {batch_size} "
        f"- min replay size: {min_replay_size}"
    )

    print("\nInitial greedy evaluation...")

    set_random_seed(evaluation_seed)    
    initial_evaluation = evaluate_against_random_both_colors(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes_per_color=evaluation_episodes_per_color,
        max_agent_steps=max_agent_steps,
    )

    initial_score = score_evaluation(
        initial_evaluation
    )

    print(
        f"Initial evaluation "
        f"- games: {initial_evaluation.episodes} "
        f"- wins: {initial_evaluation.wins} "
        f"- draws: {initial_evaluation.draws} "
        f"- losses: {initial_evaluation.losses} "
        f"- truncated: {initial_evaluation.truncated} "
        f"- score: {initial_score:.3f}"
    )

    print(
        f"Initial truncation diagnostics "
        f"- claimable threefold: "
        f"{initial_evaluation.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{initial_evaluation.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{initial_evaluation.truncated_without_claimable_draw}"
    )

    print(
        f"Initial material diagnostics "
        f"- avg total material: "
        f"{initial_evaluation.truncated_average_total_material} "
        f"- avg material balance: "
        f"{initial_evaluation.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{initial_evaluation.truncated_average_absolute_material_balance}"
    )

    print("\nTraining...")

    set_random_seed(training_seed)
    start_time = perf_counter()

    results = train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=training_episodes,
        max_agent_steps=max_agent_steps,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        target_update_frequency=target_update_frequency,
        alternate_colors=True,
    )

    elapsed_seconds = (
        perf_counter() - start_time
    )

    summary = summarize_training(
        results
    )

    print(
        f"Training completed "
        f"- time: {elapsed_seconds:.2f}s "
        f"- seconds/episode: "
        f"{elapsed_seconds / training_episodes:.2f}"
    )

    print(
        f"Training summary "
        f"- wins: {summary.wins} "
        f"- draws: {summary.draws} "
        f"- losses: {summary.losses} "
        f"- truncated: {summary.truncated} "
        f"- avg plies: {summary.average_plies:.2f} "
        f"- avg reward: {summary.average_reward:.4f} "
        f"- avg loss: "
        f"{summary.average_loss if summary.average_loss is not None else 'N/A'} "
        f"- epsilon: {summary.final_epsilon:.4f} "
        f"- replay: {summary.replay_size}"
    )

    print(
        f"Truncation diagnostics "
        f"- claimable threefold: "
        f"{summary.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{summary.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{summary.truncated_without_claimable_draw}"
    )

    print(
        f"Truncated material diagnostics "
        f"- avg total material: "
        f"{summary.truncated_average_total_material} "
        f"- avg material balance: "
        f"{summary.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{summary.truncated_average_absolute_material_balance}"
    )

    print("\nFinal greedy evaluation...")

    set_random_seed(evaluation_seed)

    final_evaluation = evaluate_against_random_both_colors(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes_per_color=evaluation_episodes_per_color,
        max_agent_steps=max_agent_steps,
    )

    final_score = score_evaluation(
        final_evaluation
    )

    print(
        f"Final evaluation "
        f"- games: {final_evaluation.episodes} "
        f"- wins: {final_evaluation.wins} "
        f"- draws: {final_evaluation.draws} "
        f"- losses: {final_evaluation.losses} "
        f"- truncated: {final_evaluation.truncated} "
        f"- score: {final_score:.3f}"
    )

    print(
        f"Final truncation diagnostics "
        f"- claimable threefold: "
        f"{final_evaluation.truncated_claimable_threefold} "
        f"- claimable fifty-move: "
        f"{final_evaluation.truncated_claimable_fifty_moves} "
        f"- without claimable draw: "
        f"{final_evaluation.truncated_without_claimable_draw}"
    )

    print(
        f"Final material diagnostics "
        f"- avg total material: "
        f"{final_evaluation.truncated_average_total_material} "
        f"- avg material balance: "
        f"{final_evaluation.truncated_average_material_balance} "
        f"- avg absolute material balance: "
        f"{final_evaluation.truncated_average_absolute_material_balance}"
    )

    print("\nSaving final greedy diagnostic game...")

    set_random_seed(evaluation_seed)

    diagnostic_path = Path(
        "state_action_evaluation_game.pgn"
    )
    diagnostic_result = (
        save_first_truncated_greedy_evaluation_game(
            env=env,
            agent=agent,
            opponent=opponent,
            path=diagnostic_path,
            max_attempts=evaluation_episodes_per_color,
            max_agent_steps=max_agent_steps,
            agent_color=chess.WHITE,
        )
    )

    if diagnostic_result is None:
        print(
            "No truncated diagnostic game found "
            f"after {evaluation_episodes_per_color} attempts."
        )
    else:
        print(
            f"Truncated diagnostic game "
            f"- plies: {diagnostic_result.total_plies} "
            f"- path: {diagnostic_path}"
        )

        print(
            "\nAnalyzing reconstructed greedy choices..."
        )

        analyze_greedy_pgn(
            path=diagnostic_path,
            agent=agent,
            agent_color=chess.WHITE,
        )

if __name__ == "__main__":
    main()