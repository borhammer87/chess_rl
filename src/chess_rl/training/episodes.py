import chess

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.results import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
)
from chess_rl.utils.action_encoder import (
    decode_legal_action,
    encode_move,
)
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.replay_buffer import ReplayBuffer
from collections.abc import Callable

OpponentMoveSelector = Callable[
    [chess.Board, list[chess.Move]],
    chess.Move,
]

def reward_for_color(
    reward: float,
    color: chess.Color,
) -> float:
    """
    Convert a White-perspective reward to the requested color.

    ChessEnv returns rewards from White's perspective:

    - White win: +1
    - Black win: -1
    - Draw or unfinished game: 0

    For a Black agent, the sign is reversed so positive reward
    always means a good outcome for the agent.
    """
    if color == chess.WHITE:
        return reward

    if color == chess.BLACK:
        return -reward

    raise ValueError(
        "color must be chess.WHITE or chess.BLACK."
    )

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

    if result.done:
        next_legal_actions = []
    else:
        next_legal_actions = [
            encode_move(move)
            for move in env.legal_moves()
        ]

    replay_buffer.push(
        state=result.state,
        action=result.action,
        reward=result.reward,
        next_state=result.next_state,
        done=result.done,
        next_legal_actions=next_legal_actions,
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
    agent_color: chess.Color = chess.WHITE,
) -> VsRandomEpisodeResult:
    """
    Run one episode with:

        DQNAgent    -> agent_color 
        RandomAgent -> opposite color

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
        opponent_move = opponent.select_move(
            env.legal_moves()
        )

        _, _, _, info = env.step(
            opponent_move
        )

        total_plies = 1
        final_info = info

    while not env.done and agent_steps < max_agent_steps:
        # The DQN must move when it is the agent's turn.
        if env.board.turn != agent_color:
            raise RuntimeError(
                "Expected agent color to move at the start "
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

        # DQN move.
        next_board, reward, done, info = env.step(move)

        agent_steps += 1
        total_plies += 1
        final_info = info

        # If the DQN agent ended the game, there is no opponent response.
        if done:
            agent_reward = reward_for_color(
                reward,
                agent_color,
            )
            next_state = encode_board(next_board)

            replay_buffer.push(
                state=state,
                action=action,
                reward=agent_reward,
                next_state=next_state,
                done=True,
                next_legal_actions=[],
            )

            loss = train_from_replay(
                agent=agent,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                min_replay_size=min_replay_size,
            )

            if loss is not None:
                training_losses.append(loss)

            total_reward += agent_reward
            break

        # RandomAgent response.
        opponent_move = opponent.select_move(
            env.legal_moves()
        )

        next_board, reward, done, info = env.step(
            opponent_move
        )

        agent_reward = reward_for_color(
            reward,
            agent_color,
        )

        total_plies += 1
        final_info = info

        # This is now either:
        # - the next position where White will decide, or
        # - the terminal position after Black's move.
        next_state = encode_board(next_board)

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
            training_losses.append(loss)

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

