from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
import chess
from chess_rl.utils.action_encoder import decode_legal_action
from chess_rl.utils.action_selection import select_greedy_action
from chess_rl.utils.board_encoder import encode_board
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.episodes import (
    reward_for_color,
    train_from_replay,
)
from chess_rl.training.results import VsRandomEpisodeResult
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import ReplayBuffer

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
) -> chess.Move:
    """
    Select the frozen opponent's best legal move greedily.
    """
    legal_moves = list(board.legal_moves)

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
    Run one episode between the learning DQN and a frozen DQN opponent.

    Only decisions made by the learning agent are stored in replay.
    """
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

    env.reset()

    agent_steps = 0
    total_plies = 0
    total_reward = 0.0
    final_info: dict = {}
    training_losses: list[float] = []

    if agent_color == chess.BLACK:
        opponent_move = select_frozen_opponent_move(
            opponent=opponent,
            board=env.get_state(),
        )

        _, _, _, info = env.step(
            opponent_move
        )

        total_plies = 1
        final_info = info

    while (
        not env.done
        and agent_steps < max_agent_steps
    ):
        if env.board.turn != agent_color:
            raise RuntimeError(
                "Expected agent color to move at the start "
                "of a DQN decision."
            )

        state = encode_board(
            env.get_state()
        )

        legal_moves = env.legal_moves()

        action = agent.select_action(
            state=state,
            legal_moves=legal_moves,
        )

        move = decode_legal_action(
            action=action,
            legal_moves=legal_moves,
        )

        next_board, reward, done, info = env.step(
            move
        )

        agent_steps += 1
        total_plies += 1
        final_info = info

        if not done:
            opponent_move = select_frozen_opponent_move(
                opponent=opponent,
                board=env.get_state(),
            )

            next_board, reward, done, info = env.step(
                opponent_move
            )

            total_plies += 1
            final_info = info

        agent_reward = reward_for_color(
            reward,
            agent_color,
        )

        next_state = encode_board(
            next_board
        )

        if done:
            next_legal_actions = []
        else:
            next_legal_actions = [
                encode_move(move)
                for move in env.legal_moves()
            ]

        replay_buffer.push(
            state=state,
            action=action,
            reward=agent_reward,
            next_state=next_state,
            done=done,
            next_legal_actions=next_legal_actions,
        )

        loss = train_from_replay(
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
        )

        if loss is not None:
            training_losses.append(
                loss
            )

        total_reward += agent_reward

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