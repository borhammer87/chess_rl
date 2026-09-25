import random
from time import perf_counter

import torch

from chess_rl.agents.random_agent import RandomAgent
from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.train_dqn import (
    evaluate_against_random_both_colors,
    score_evaluation,
    summarize_training,
    train_against_random,
)
from chess_rl.utils.replay_buffer import ReplayBuffer
def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)

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

if __name__ == "__main__":
    main()