from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.agents.random_agent import RandomAgent
from collections.abc import Callable
from pathlib import Path   
from chess_rl.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from chess_rl.training.results import (
    EvaluationSummary,
    TrainingSummary,
    VsRandomEpisodeResult,
)
from chess_rl.training.episodes import (
    run_dqn_vs_random_episode,
)
def train_against_random(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: RandomAgent,
    replay_buffer: ReplayBuffer,
    episodes: int,
    max_agent_steps: int = 150,
    batch_size: int = 32,
    min_replay_size: int = 1_000,
    target_update_frequency: int = 10,
    progress_callback: Callable[
        [int, int, VsRandomEpisodeResult],
        None,
    ] | None = None,
    checkpoint_frequency: int | None = None,
    checkpoint_callback: Callable[
        [int, DQNAgent],
        None,
    ] | None = None,
    evaluation_frequency: int | None = None,
    evaluation_callback: Callable[
        [int, DQNAgent],
        None,
    ] | None = None,
) -> list[VsRandomEpisodeResult]:
    """
    Run multiple DQN-versus-random episodes.

    The same agent and replay buffer are reused across all episodes,
    allowing replay memory to accumulate transitions from different
    games.

    The target network is periodically synchronized with the policy
    network after the configured number of episodes.

    Args:
        env: Chess environment reused across episodes.
        agent: DQN agent trained during the episodes.
        opponent: Random opponent playing Black.
        replay_buffer: Shared replay memory.
        episodes: Number of episodes to run.
        max_agent_steps: Maximum DQN decisions per episode.
        batch_size: Number of transitions sampled per training update.
        min_replay_size: Minimum replay size before training begins.
        target_update_frequency: Number of completed episodes between
            target-network synchronizations.
        progress_callback: Optional function called after each episode.
            It receives completed episodes, total episodes, and the
            latest episode result.
        checkpoint_frequency: Number of completed episodes between
            checkpoint saves. None disables periodic checkpointing.
        checkpoint_callback: Optional function called when a checkpoint
            should be saved. It receives the completed episode count
            and the agent.
        evaluation_frequency: Number of completed episodes between
            evaluations. None disables periodic evaluation.
        evaluation_callback: Optional function called when an evaluation
            should be performed. It receives the completed episode count
            and the agent.

    Returns:
        One result for each completed or truncated episode.
    """
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    if target_update_frequency <= 0:
        raise ValueError(
            "target_update_frequency must be greater than zero."
        )

    if (
        checkpoint_frequency is not None
        and checkpoint_frequency <= 0
    ):
        raise ValueError(
            "checkpoint_frequency must be greater than zero."
        )

    if (
        checkpoint_frequency is not None
        and checkpoint_callback is None
    ):
        raise ValueError(
            "checkpoint_callback is required when "
            "checkpoint_frequency is set."
        )

    if (
        evaluation_frequency is not None
        and evaluation_frequency <= 0
    ):
        raise ValueError(
            "evaluation_frequency must be greater than zero."
        )

    if (
        evaluation_frequency is not None
        and evaluation_callback is None
    ):
        raise ValueError(
            "evaluation_callback is required when "
            "evaluation_frequency is set."
        )

    results: list[VsRandomEpisodeResult] = []

    for episode_index in range(episodes):
        result = run_dqn_vs_random_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=max_agent_steps,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
        )

        results.append(result)

        completed_episodes = episode_index + 1

        if progress_callback is not None:
            progress_callback(
                completed_episodes,
                episodes,
                result,
            )

        if completed_episodes % target_update_frequency == 0:
            agent.update_target()

        if (
            checkpoint_frequency is not None
            and completed_episodes % checkpoint_frequency == 0
        ):
            checkpoint_callback(
                completed_episodes,
                agent,
            )

        if (
            evaluation_frequency is not None
            and completed_episodes % evaluation_frequency == 0
        ):
            evaluation_callback(
                completed_episodes,
                agent,
            )

    return results
    
def summarize_training(
    results: list[VsRandomEpisodeResult],
) -> TrainingSummary:
    """
    Aggregate metrics collected across multiple training episodes.
    """
    if not results:
        raise ValueError(
            "results must contain at least one episode."
        )

    average_reward = sum(
        result.total_reward
        for result in results
    ) / len(results)

    losses = [
        loss
        for result in results
        for loss in result.training_losses
    ]

    average_loss = (
        sum(losses) / len(losses)
        if losses
        else None
    )

    final_result = results[-1]

    return TrainingSummary(
        episodes=len(results),
        average_reward=average_reward,
        average_loss=average_loss,
        final_epsilon=final_result.final_epsilon,
        replay_size=final_result.replay_size,
    )

def evaluate_against_random(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: RandomAgent,
    episodes: int,
    max_agent_steps: int = 150,
) -> EvaluationSummary:
    """
    Evaluate the current greedy DQN policy against RandomAgent.

    Evaluation does not train the agent, decay epsilon, synchronize
    networks, or modify the training replay buffer.
    """
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    if max_agent_steps <= 0:
        raise ValueError(
            "max_agent_steps must be greater than zero."
        )

    original_epsilon = agent.epsilon

    evaluation_buffer = ReplayBuffer(
        capacity=max_agent_steps,
    )

    results: list[VsRandomEpisodeResult] = []

    try:
        agent.epsilon = 0.0

        for _ in range(episodes):
            result = run_dqn_vs_random_episode(
                env=env,
                agent=agent,
                opponent=opponent,
                replay_buffer=evaluation_buffer,
                max_agent_steps=max_agent_steps,
                batch_size=1,
                min_replay_size=max_agent_steps + 1,
            )

            results.append(result)

    finally:
        agent.epsilon = original_epsilon

    wins = sum(
        result.final_info.get("result") == "1-0"
        for result in results
    )

    draws = sum(
        result.final_info.get("result") == "1/2-1/2"
        for result in results
    )

    losses = sum(
        result.final_info.get("result") == "0-1"
        for result in results
    )

    truncated = sum(
        result.truncated
        for result in results
    )

    return EvaluationSummary(
        episodes=len(results),
        wins=wins,
        draws=draws,
        losses=losses,
        truncated=truncated,
    )



def main() -> None:
    """
    Run multi-episode DQN training against RandomAgent
    and print a concise training summary.
    """
    def print_training_progress(
        completed_episodes: int,
        total_episodes: int,
        result: VsRandomEpisodeResult,
    ) -> None:
        print(
            f"Episode {completed_episodes}/{total_episodes} "
            f"- reward: {result.total_reward:.4f} "
            f"- epsilon: {result.final_epsilon:.4f} "
            f"- replay: {result.replay_size}"
        )

    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10_000)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "latest.pt"

    def save_checkpoint_callback(
        completed_episodes: int,
        agent: DQNAgent,
    ) -> None:
        save_training_checkpoint(
            path=str(checkpoint_path),
            agent=agent,
            replay_buffer=replay_buffer,
        )

        print(
            f"Checkpoint saved after episode "
            f"{completed_episodes}: {checkpoint_path}"
        )

    def evaluation_callback(
        completed_episodes: int,
        agent: DQNAgent,
    ) -> None:
        evaluation = evaluate_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            episodes=20,
            max_agent_steps=150,
        )

        print(
            f"Evaluation after episode {completed_episodes}: "
            f"{evaluation.wins}W / "
            f"{evaluation.draws}D / "
            f"{evaluation.losses}L / "
            f"{evaluation.truncated} truncated"
        )


    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        load_training_checkpoint(
            path=str(checkpoint_path),
            agent=agent,
            replay_buffer=replay_buffer,
        )
    else:
        print("No checkpoint found. Starting from scratch.")

    results = train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=100,
        max_agent_steps=150,
        batch_size=32,
        min_replay_size=1_000,
        target_update_frequency=10,
        progress_callback=print_training_progress,
        checkpoint_frequency=25,
        checkpoint_callback=save_checkpoint_callback,
        evaluation_frequency=25,
        evaluation_callback=evaluation_callback,
    )

    summary = summarize_training(results)

    print(f"Episodes: {summary.episodes}")
    print(f"Average reward: {summary.average_reward:.4f}")

    if summary.average_loss is None:
        print("Average loss: N/A")
    else:
        print(f"Average loss: {summary.average_loss:.4f}")

    print(f"Final epsilon: {summary.final_epsilon:.4f}")
    print(f"Replay buffer size: {summary.replay_size}")

if __name__ == "__main__":
    main()