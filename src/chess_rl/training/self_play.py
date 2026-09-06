from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
import chess
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.action_selection import select_greedy_action
from chess_rl.utils.board_encoder import encode_board
from chess_rl.training.episodes import (
    OpponentMoveSelector,
    run_dqn_vs_opponent_episode,
)
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.training.results import VsRandomEpisodeResult
from chess_rl.env.chess_env import ChessEnv

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

    for episode_index in range(episodes):
        agent_color = (
            chess.WHITE
            if episode_index % 2 == 0
            else chess.BLACK
        )

        result = run_dqn_vs_frozen_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=max_agent_steps,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            agent_color=agent_color,
        )

        results.append(result)

        completed_episodes = episode_index + 1

        if completed_episodes % target_update_frequency == 0:
            agent.update_target()

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