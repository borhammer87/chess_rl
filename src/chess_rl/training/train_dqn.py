from dataclasses import dataclass

import chess
import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.agents.random_agent import RandomAgent
from collections.abc import Callable
from pathlib import Path   

@dataclass
class StepResult:
    """
    Result of one complete agent-environment interaction.
    """

    state: torch.Tensor
    action: int
    move: chess.Move
    reward: float
    next_state: torch.Tensor
    done: bool
    info: dict

@dataclass
class EpisodeResult:
    """
    Result of one complete episode.

    An episode normally represents one chess game, although it can
    also stop early when the configured step limit is reached.
    """

    steps: int
    total_reward: float
    done: bool
    truncated: bool
    final_info: dict

@dataclass
class VsRandomEpisodeResult:
    """
    Result of one episode where the DQN plays White
    and RandomAgent plays Black.
    """

    agent_steps: int
    total_plies: int
    total_reward: float
    done: bool
    truncated: bool
    final_info: dict
    training_losses: list[float]
    final_epsilon: float
    replay_size: int

@dataclass
class TrainingSummary:
    """
    Aggregate metrics from a multi-episode training run.
    """

    episodes: int
    average_reward: float
    average_loss: float | None
    final_epsilon: float
    replay_size: int

@dataclass
class EvaluationSummary:
    """
    Aggregate results from evaluation games against RandomAgent.
    """

    episodes: int
    wins: int
    draws: int
    losses: int
    truncated: int

def run_single_step(
    env: ChessEnv,
    agent: DQNAgent,
) -> StepResult:
    """
    Execute one complete interaction between the DQN agent
    and the chess environment.

    Flow:
        Board
        -> encoded tensor
        -> action index
        -> chess.Move
        -> environment step
        -> next encoded tensor
    """

    board = env.get_state()

    if env.done:
        raise RuntimeError(
            "Cannot run a training step: the game has already ended."
        )

    legal_moves = env.legal_moves()

    if not legal_moves:
        raise RuntimeError(
            "Cannot run a training step without legal moves."
        )

    state = encode_board(board)

    action = agent.select_action(
        state=state,
        legal_moves=legal_moves,
    )

    move = decode_legal_action(
        action=action,
        legal_moves=legal_moves,
    )   

    next_board, reward, done, info = env.step(move)

    next_state = encode_board(next_board)

    return StepResult(
        state=state,
        action=action,
        move=move,
        reward=reward,
        next_state=next_state,
        done=done,
        info=info,
    )

def run_and_store_step(
    env: ChessEnv,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
) -> StepResult:
    """
    Execute one agent-environment interaction and store the resulting
    transition in the replay buffer.

    The replay buffer stores the information required by DQN:

        state
        action
        reward
        next_state
        done

    The decoded chess.Move and the info dictionary are useful for
    inspection, but they are not required for the Bellman update.
    """
    result = run_single_step(
        env=env,
        agent=agent,
    )

    replay_buffer.push(
        state=result.state,
        action=result.action,
        reward=result.reward,
        next_state=result.next_state,
        done=result.done,
    )

    return result

def train_from_replay(
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    min_replay_size: int,
) -> float | None:
    """
    Train the DQN agent using one random batch from replay memory.

    Training starts only when the replay buffer contains at least
    min_replay_size transitions.

    Args:
        agent: DQN agent to train.
        replay_buffer: Memory containing previous transitions.
        batch_size: Number of transitions sampled for one update.
        min_replay_size: Minimum number of stored transitions required
            before training begins.

    Returns:
        The training loss, or None when replay memory is not ready.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if min_replay_size <= 0:
        raise ValueError(
            "min_replay_size must be greater than zero."
        )

    if min_replay_size < batch_size:
        raise ValueError(
            "min_replay_size must be at least batch_size."
        )

    if len(replay_buffer) < min_replay_size:
        return None

    batch = replay_buffer.sample(batch_size)

    loss = agent.train_step(batch)

    agent.decay_epsilon()

    return loss

def run_episode(
    env: ChessEnv,
    agent: DQNAgent,
    replay_buffer: ReplayBuffer,
    max_steps: int = 300,
) -> EpisodeResult:
    """
    Run one episode and store every transition in replay memory.

    The episode finishes when:

    - the chess game reaches a terminal state, or
    - max_steps is reached.

    No neural-network training is performed here yet.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")

    env.reset()

    total_reward = 0.0
    steps = 0
    final_info: dict = {}

    while not env.done and steps < max_steps:
        result = run_and_store_step(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
        )

        steps += 1
        total_reward += result.reward
        final_info = result.info

    truncated = not env.done and steps >= max_steps

    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        done=env.done,
        truncated=truncated,
        final_info=final_info,
    )

def run_dqn_vs_random_episode(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: RandomAgent,
    replay_buffer: ReplayBuffer,
    max_agent_steps: int = 150,
    batch_size: int = 32,
    min_replay_size: int = 1_000,
) -> VsRandomEpisodeResult:
    """
    Run one episode with:

        DQNAgent    -> White
        RandomAgent -> Black

    One replay transition spans:

        state before the DQN move
        -> DQN action
        -> opponent response
        -> next state for the DQN

    Only DQN decisions are stored in replay memory.

    After each stored transition, one training update is attempted.
    Training begins only when replay memory reaches min_replay_size.
    """

    if max_agent_steps <= 0:
        raise ValueError(
            "max_agent_steps must be greater than zero."
        )

    env.reset()

    agent_steps = 0
    total_plies = 0
    total_reward = 0.0
    final_info: dict = {}
    training_losses: list[float] = []

    while not env.done and agent_steps < max_agent_steps:
        # The DQN currently plays only as White.
        if env.board.turn != chess.WHITE:
            raise RuntimeError(
                "Expected White to move at the start "
                "of a DQN decision."
            )

        # State observed before the DQN action.
        state = encode_board(env.get_state())
        legal_moves = env.legal_moves()

        action = agent.select_action(
            state=state,
            legal_moves=legal_moves,
        )

        move = decode_legal_action(
            action=action,
            legal_moves=legal_moves,
        )

        # White/DQN move.
        next_board, reward, done, info = env.step(move)

        agent_steps += 1
        total_plies += 1
        final_info = info

        # If White ended the game, there is no opponent response.
        if done:
            next_state = encode_board(next_board)

            replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=True,
            )

            loss = train_from_replay(
                agent=agent,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                min_replay_size=min_replay_size,
            )

            if loss is not None:
                training_losses.append(loss)

            total_reward += reward
            break

        # Black/RandomAgent response.
        opponent_move = opponent.select_move(
            env.legal_moves()
        )

        next_board, reward, done, info = env.step(
            opponent_move
        )

        total_plies += 1
        final_info = info

        # This is now either:
        # - the next position where White will decide, or
        # - the terminal position after Black's move.
        next_state = encode_board(next_board)

        replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        loss = train_from_replay(
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
        )

        if loss is not None:
            training_losses.append(loss)

        total_reward += reward

    truncated = (
        not env.done
        and agent_steps >= max_agent_steps
    )

    return VsRandomEpisodeResult(
        agent_steps=agent_steps,
        total_plies=total_plies,
        total_reward=total_reward,
        done=env.done,
        truncated=truncated,
        final_info=final_info,
        training_losses=training_losses,
        final_epsilon=agent.epsilon,
        replay_size=len(replay_buffer),
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

    def save_training_checkpoint(
        completed_episodes: int,
        agent: DQNAgent,
    ) -> None:
        agent.save_checkpoint(str(checkpoint_path))
        print(
            f"Checkpoint saved after episode "
            f"{completed_episodes}: {checkpoint_path}"
        )

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        agent.load_checkpoint(str(checkpoint_path))
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
        checkpoint_callback=save_training_checkpoint
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