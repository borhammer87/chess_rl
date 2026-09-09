from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
import chess
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.action_selection import select_greedy_action
from chess_rl.utils.board_encoder import encode_board
from chess_rl.training.episodes import (
    OpponentMoveSelector,
    run_dqn_vs_opponent_episode,
    get_episode_agent_color,
)
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.training.results import (
    VsRandomEpisodeResult,
    EvaluationSummary,
)
from chess_rl.env.chess_env import ChessEnv
from collections.abc import Callable

def create_frozen_opponent(
    agent: DQNAgent,
) -> DQNCNN:
    """
    Create an independent frozen copy of the agent's current policy.

    The returned network can be used as a stable self-play opponent.
    It does not share parameters with the learning policy and its
    parameters do not require gradients.
    """
    opponent = DQNCNN()

    opponent.load_state_dict(
        agent.policy_net.state_dict()
    )

    opponent.eval()

    for parameter in opponent.parameters():
        parameter.requires_grad_(False)

    return opponent

def select_frozen_opponent_move(
    opponent: DQNCNN,
    board: chess.Board,
    legal_moves: list[chess.Move],
) -> chess.Move:
    """
    Select the frozen opponent's best legal move greedily.
    """
    state = encode_board(board)

    action = select_greedy_action(
        network=opponent,
        state=state,
        legal_moves=legal_moves,
    )

    return decode_legal_action(
        action=action,
        legal_moves=legal_moves,
    )

def create_frozen_opponent_selector(
    opponent: DQNCNN,
) -> OpponentMoveSelector:
    """
    Adapt a frozen DQN network to the common opponent selector interface.
    """

    def select_move(
        board: chess.Board,
        legal_moves: list[chess.Move],
    ) -> chess.Move:
        return select_frozen_opponent_move(
            opponent=opponent,
            board=board,
            legal_moves=legal_moves,
        )

    return select_move

def run_dqn_vs_frozen_episode(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: DQNCNN,
    replay_buffer: ReplayBuffer,
    max_agent_steps: int = 150,
    batch_size: int = 32,
    min_replay_size: int = 1_000,
    agent_color: chess.Color = chess.WHITE,
) -> VsRandomEpisodeResult:
    """
    Run one training episode against a frozen DQN opponent.
    """
    opponent_move_selector = (
        create_frozen_opponent_selector(
            opponent
        )
    )

    return run_dqn_vs_opponent_episode(
        env=env,
        agent=agent,
        opponent_move_selector=opponent_move_selector,
        replay_buffer=replay_buffer,
        max_agent_steps=max_agent_steps,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        agent_color=agent_color,
    )

def evaluate_against_frozen(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: DQNCNN,
    episodes: int,
    max_agent_steps: int = 150,
    agent_color: chess.Color = chess.WHITE,
) -> EvaluationSummary:
    """
    Evaluate the current greedy DQN policy against a frozen opponent.

    Evaluation does not train the agent or modify the training
    replay buffer.
    """
    if episodes <= 0:
        raise ValueError(
            "episodes must be greater than zero."
        )

    if max_agent_steps <= 0:
        raise ValueError(
            "max_agent_steps must be greater than zero."
        )

    if agent_color not in (
        chess.WHITE,
        chess.BLACK,
    ):
        raise ValueError(
            "agent_color must be chess.WHITE or chess.BLACK."
        )

    original_epsilon = agent.epsilon

    evaluation_buffer = ReplayBuffer(
        capacity=max_agent_steps,
    )

    results: list[VsRandomEpisodeResult] = []

    try:
        agent.epsilon = 0.0

        for _ in range(episodes):
            result = run_dqn_vs_frozen_episode(
                env=env,
                agent=agent,
                opponent=opponent,
                replay_buffer=evaluation_buffer,
                max_agent_steps=max_agent_steps,
                batch_size=1,
                min_replay_size=max_agent_steps + 1,
                agent_color=agent_color,
            )

            results.append(result)

    finally:
        agent.epsilon = original_epsilon

    if agent_color == chess.WHITE:
        win_result = "1-0"
        loss_result = "0-1"
    else:
        win_result = "0-1"
        loss_result = "1-0"

    wins = sum(
        result.final_info.get("result") == win_result
        for result in results
    )

    draws = sum(
        result.final_info.get("result") == "1/2-1/2"
        for result in results
    )

    losses = sum(
        result.final_info.get("result") == loss_result
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

def evaluate_against_frozen_both_colors(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: DQNCNN,
    episodes_per_color: int,
    max_agent_steps: int = 150,
) -> EvaluationSummary:
    """
    Evaluate the DQN equally as White and Black
    against a frozen opponent.
    """
    if episodes_per_color <= 0:
        raise ValueError(
            "episodes_per_color must be greater than zero."
        )

    white_summary = evaluate_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes=episodes_per_color,
        max_agent_steps=max_agent_steps,
        agent_color=chess.WHITE,
    )

    black_summary = evaluate_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes=episodes_per_color,
        max_agent_steps=max_agent_steps,
        agent_color=chess.BLACK,
    )

    return EvaluationSummary(
        episodes=(
            white_summary.episodes
            + black_summary.episodes
        ),
        wins=(
            white_summary.wins
            + black_summary.wins
        ),
        draws=(
            white_summary.draws
            + black_summary.draws
        ),
        losses=(
            white_summary.losses
            + black_summary.losses
        ),
        truncated=(
            white_summary.truncated
            + black_summary.truncated
        ),
    )

def train_against_frozen(
    env: ChessEnv,
    agent: DQNAgent,
    opponent: DQNCNN,
    replay_buffer: ReplayBuffer,
    episodes: int,
    max_agent_steps: int = 150,
    batch_size: int = 32,
    min_replay_size: int = 1_000,
    target_update_frequency: int = 10,
    opponent_update_frequency: int | None = None,
    progress_callback: Callable[
        [int, int, VsRandomEpisodeResult],
        None,
    ] | None = None,
    agent_color: chess.Color = chess.WHITE,
    alternate_colors: bool = True,
    checkpoint_frequency: int | None = None,
    checkpoint_callback: Callable[
        [int, DQNAgent],
        None,
    ] | None = None,
) -> list[VsRandomEpisodeResult]:
    """
    Run multiple training episodes against one frozen DQN opponent.

    The learning agent alternates between White and Black.
    The same frozen opponent is reused for every episode.
    """
    if episodes <= 0:
        raise ValueError(
            "episodes must be greater than zero."
        )

    if (
        opponent_update_frequency is not None
        and opponent_update_frequency <= 0
    ):
        raise ValueError(
            "opponent_update_frequency must be greater than zero."
        )
    results: list[VsRandomEpisodeResult] = []

    if target_update_frequency <= 0:
        raise ValueError(
            "target_update_frequency must be greater than zero."
        )

    if agent_color not in (
        chess.WHITE,
        chess.BLACK,
    ):
        raise ValueError(
            "agent_color must be chess.WHITE or chess.BLACK."
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

    for episode_index in range(episodes):
        episode_agent_color = get_episode_agent_color(
            initial_color=agent_color,
            episode_index=episode_index,
            alternate_colors=alternate_colors,
        )

        result = run_dqn_vs_frozen_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=max_agent_steps,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            agent_color=episode_agent_color,
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
            opponent_update_frequency is not None
            and completed_episodes % opponent_update_frequency == 0
        ):
            update_frozen_opponent(
                agent=agent,
                opponent=opponent,
            )

    return results

def update_frozen_opponent(
    agent: DQNAgent,
    opponent: DQNCNN,
) -> None:
    """
    Replace the frozen opponent weights with the agent's current policy.
    """
    opponent.load_state_dict(
        agent.policy_net.state_dict()
    )

    opponent.eval()

    for parameter in opponent.parameters():
        parameter.requires_grad_(False)