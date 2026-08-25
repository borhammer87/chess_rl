import chess
import pytest
import torch

import chess_rl.training.episodes as episodes_module
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.episodes import (
    run_and_store_step,
    run_dqn_vs_random_episode,
    run_episode,
    run_single_step,
    train_from_replay,
    reward_for_color,
)
from chess_rl.training.results import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
)
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.board_encoder import encode_board
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.utils.board_encoder import BOARD_CHANNELS

def test_run_single_step_returns_step_result():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    env.reset()

    result = run_single_step(
        env=env,
        agent=agent,
    )

    assert isinstance(result, StepResult)

def test_run_single_step_returns_correct_tensor_shapes():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    env.reset()

    result = run_single_step(
        env=env,
        agent=agent,
    )

    assert result.state.shape == (BOARD_CHANNELS, 8, 8)
    assert result.next_state.shape == (BOARD_CHANNELS, 8, 8)

    assert isinstance(result.state, torch.Tensor)
    assert isinstance(result.next_state, torch.Tensor)

def test_run_single_step_selects_legal_move():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    board_before = env.reset()
    legal_moves_before = list(board_before.legal_moves)

    result = run_single_step(
        env=env,
        agent=agent,
    )

    assert result.move in legal_moves_before
    assert result.action == encode_move(result.move)

def test_run_single_step_changes_board():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    board_before = env.reset()
    initial_fen = board_before.fen()

    run_single_step(
        env=env,
        agent=agent,
    )

    assert env.get_state().fen() != initial_fen

def test_run_single_step_returns_non_terminal_reward_initially():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    env.reset()

    result = run_single_step(
        env=env,
        agent=agent,
    )

    assert result.reward == 0.0
    assert result.done is False
    assert result.info["result"] is None

def test_run_single_step_rejects_finished_game():
    env = ChessEnv()
    agent = DQNAgent()

    env.done = True

    with pytest.raises(
        RuntimeError,
        match="game has already ended",
    ):
        run_single_step(
            env=env,
            agent=agent,
        )

def test_run_and_store_step_adds_transition_to_buffer():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=10)

    env.reset()

    assert len(replay_buffer) == 0

    run_and_store_step(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
    )

    assert len(replay_buffer) == 1

def test_stored_transition_matches_step_result():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=10)

    env.reset()

    result = run_and_store_step(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
    )

    transition = replay_buffer.buffer[0]

    assert torch.equal(
        transition.state,
        result.state,
    )
    assert transition.action == result.action
    assert transition.reward == result.reward
    assert torch.equal(
        transition.next_state,
        result.next_state,
    )
    assert transition.done == result.done

def test_run_and_store_step_returns_step_result():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=10)

    env.reset()

    result = run_and_store_step(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
    )

    assert isinstance(result, StepResult)

def test_multiple_steps_grow_replay_buffer():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=10)

    env.reset()

    for _ in range(4):
        run_and_store_step(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
        )

    assert len(replay_buffer) == 4

def test_run_episode_returns_episode_result():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_episode(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_steps=5,
    )

    assert isinstance(result, EpisodeResult)

def test_run_episode_stores_one_transition_per_step():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_episode(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_steps=5,
    )

    assert len(replay_buffer) == result.steps

def test_run_episode_respects_max_steps():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_episode(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_steps=4,
    )

    assert result.steps <= 4

    if not result.done:
        assert result.steps == 4
        assert result.truncated is True

def test_run_episode_marks_completed_game_as_not_truncated():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=1_000)

    result = run_episode(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_steps=1_000,
    )

    if result.done:
        assert result.truncated is False

def test_run_episode_rejects_zero_max_steps():
    env = ChessEnv()
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        run_episode(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            max_steps=0,
        )

def test_run_episode_resets_environment():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    replay_buffer = ReplayBuffer(capacity=100)

    env.step(chess.Move.from_uci("e2e4"))

    run_episode(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_steps=1,
    )

    assert len(env.board.move_stack) == 1

def test_dqn_vs_random_returns_expected_result():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
    )

    assert isinstance(
        result,
        VsRandomEpisodeResult,
    )

def test_dqn_vs_random_stores_only_dqn_transitions():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=3,
    )

    assert len(replay_buffer) == result.agent_steps

def test_one_agent_step_contains_white_and_black_moves():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
    )

    assert result.agent_steps == 1
    assert result.total_plies == 2
    assert len(replay_buffer) == 1

    # After White and Black have moved, it is White's turn again.
    assert env.board.turn == chess.WHITE

def test_dqn_vs_random_transition_spans_opponent_response():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
    )

    transition = replay_buffer.buffer[0]

    assert transition.state.shape == (BOARD_CHANNELS, 8, 8)
    assert transition.next_state.shape == (BOARD_CHANNELS, 8, 8)

    assert not torch.equal(
        transition.state,
        transition.next_state,
    )

    assert transition.reward == 0.0
    assert transition.done is False

def test_dqn_vs_random_respects_agent_step_limit():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=2,
    )

    assert result.agent_steps <= 2

    if not result.done:
        assert result.agent_steps == 2
        assert result.truncated is True

def test_dqn_vs_random_rejects_invalid_step_limit():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        run_dqn_vs_random_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=0,
            
        )

def test_train_from_replay_waits_for_minimum_replay_size():
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    initial_epsilon = agent.epsilon

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    for action in range(3):
        replay_buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
        )

    loss = train_from_replay(
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=2,
        min_replay_size=4,
    )

    assert loss is None
    assert agent.epsilon == initial_epsilon

    loss = train_from_replay(
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=2,
        min_replay_size=4,
    )

    assert loss is None

def test_dqn_vs_random_attempts_training_after_storing_transition(
    monkeypatch,
    ):
        env = ChessEnv()
        agent = DQNAgent(epsilon=1.0)
        opponent = RandomAgent()
        replay_buffer = ReplayBuffer(capacity=100)

        received_calls = []

        def fake_train_from_replay(
            agent,
            replay_buffer,
            batch_size,
            min_replay_size,
        ):
            received_calls.append(
                {
                    "buffer_size": len(replay_buffer),
                    "batch_size": batch_size,
                    "min_replay_size": min_replay_size,
                }
            )
            return None

        monkeypatch.setattr(
            episodes_module,
            "train_from_replay",
            fake_train_from_replay,
        )

        result = run_dqn_vs_random_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=1,
            batch_size=2,
            min_replay_size=4,
        )

        assert result.agent_steps == 1
        assert len(received_calls) == 1
        assert received_calls[0]["buffer_size"] == 1
        assert received_calls[0]["batch_size"] == 2
        assert received_calls[0]["min_replay_size"] == 4

def test_train_from_replay_decays_epsilon_once_after_training(
        
    monkeypatch,
):
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    for action in range(4):
        replay_buffer.push(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
        )

    decay_calls = []

    def fake_decay_epsilon():
        decay_calls.append(True)

    monkeypatch.setattr(
        agent,
        "decay_epsilon",
        fake_decay_epsilon,
    )

    loss = train_from_replay(
        agent=agent,
        replay_buffer=replay_buffer,
        batch_size=2,
        min_replay_size=4,
    )

    assert isinstance(loss, float)
    assert len(decay_calls) == 1

def test_dqn_vs_random_records_training_loss(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    def fake_train_from_replay(
        agent,
        replay_buffer,
        batch_size,
        min_replay_size,
    ):
        return 0.25

    monkeypatch.setattr(
        episodes_module,
        "train_from_replay",
        fake_train_from_replay,
    )

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert result.training_losses == [0.25]

def test_dqn_vs_random_ignores_missing_training_loss(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    def fake_train_from_replay(
        agent,
        replay_buffer,
        batch_size,
        min_replay_size,
    ):
        return None

    monkeypatch.setattr(
        episodes_module,
        "train_from_replay",
        fake_train_from_replay,
    )

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert result.training_losses == []

def test_dqn_vs_random_records_final_epsilon(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    def fake_train_from_replay(
        agent,
        replay_buffer,
        batch_size,
        min_replay_size,
    ):
        agent.decay_epsilon()
        return 0.25

    monkeypatch.setattr(
        episodes_module,
        "train_from_replay",
        fake_train_from_replay,
    )

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert result.final_epsilon == 0.5

def test_dqn_vs_random_records_replay_size():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert result.replay_size == 1
    assert result.replay_size == len(replay_buffer)

def test_reward_for_white_keeps_white_perspective():
    assert reward_for_color(
        1.0,
        chess.WHITE,
    ) == 1.0

    assert reward_for_color(
        -1.0,
        chess.WHITE,
    ) == -1.0

def test_reward_for_black_reverses_white_perspective():
    assert reward_for_color(
        1.0,
        chess.BLACK,
    ) == -1.0

    assert reward_for_color(
        -1.0,
        chess.BLACK,
    ) == 1.0

def test_reward_for_color_preserves_draw_reward():
    assert reward_for_color(
        0.0,
        chess.WHITE,
    ) == 0.0

    assert reward_for_color(
        0.0,
        chess.BLACK,
    ) == 0.0

def test_reward_for_color_rejects_invalid_color():
    with pytest.raises(
        ValueError,
        match="chess.WHITE or chess.BLACK",
    ):
        reward_for_color(
            1.0,
            None,
        )

def test_dqn_vs_random_defaults_to_white():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
    )

    assert result.agent_steps == 1
    assert result.total_plies == 2

def test_dqn_vs_random_black_waits_for_white_move():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        agent_color=chess.BLACK,
    )

    assert result.agent_steps == 1
    assert result.total_plies >= 2

def test_dqn_vs_random_rejects_invalid_agent_color():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="agent_color must be",
    ):
        run_dqn_vs_random_episode(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            max_agent_steps=1,
            agent_color=None,
        )

def test_dqn_vs_random_black_stores_reward_from_agent_perspective(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    rewards = iter([
        0.0,
        -1.0,
    ])

    original_step = env.step

    def fake_step(move):
        board, _, done, info = original_step(move)

        reward = next(rewards)

        if reward == -1.0:
            env.done = True
            done = True

        return board, reward, done, info

    monkeypatch.setattr(
        env,
        "step",
        fake_step,
    )

    run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        agent_color=chess.BLACK,
    )

    transition = replay_buffer.buffer[0]

    assert transition.reward == 1.0