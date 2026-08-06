import chess
import pytest
import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import ReplayBuffer

from chess_rl.training.train_dqn import (
    EpisodeResult,
    StepResult,
    run_and_store_step,
    run_episode,
    run_single_step,
)

from chess_rl.agents.random_agent import RandomAgent

from chess_rl.training.train_dqn import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
    run_and_store_step,
    run_dqn_vs_random_episode,
    run_episode,
    run_single_step,
)

from chess_rl.training.train_dqn import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
    run_and_store_step,
    run_dqn_vs_random_episode,
    run_episode,
    run_single_step,
    train_from_replay,
)

import chess_rl.training.train_dqn as train_dqn_module

from chess_rl.training.train_dqn import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
    run_and_store_step,
    run_dqn_vs_random_episode,
    run_episode,
    run_single_step,
    train_against_random,
    train_from_replay,
)

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

    assert result.state.shape == (12, 8, 8)
    assert result.next_state.shape == (12, 8, 8)

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

    assert transition.state.shape == (12, 8, 8)
    assert transition.next_state.shape == (12, 8, 8)

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

    state = torch.zeros((12, 8, 8))

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
            train_dqn_module,
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

def test_train_against_random_returns_one_result_per_episode(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    fake_result = VsRandomEpisodeResult(
        agent_steps=1,
        total_plies=2,
        total_reward=0.0,
        done=False,
        truncated=True,
        final_info={},
    )

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        return fake_result

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    results = train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert len(results) == 3
    assert all(
        isinstance(result, VsRandomEpisodeResult)
        for result in results
    )

def test_train_against_random_reuses_same_replay_buffer(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    received_buffers = []

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        received_buffers.append(replay_buffer)

        return VsRandomEpisodeResult(
            agent_steps=1,
            total_plies=2,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
        )

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
    )

    assert len(received_buffers) == 3
    assert all(
        received_buffer is replay_buffer
        for received_buffer in received_buffers
    )

def test_train_against_random_passes_training_configuration(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    received_calls = []

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        received_calls.append(
            {
                "max_agent_steps": max_agent_steps,
                "batch_size": batch_size,
                "min_replay_size": min_replay_size,
            }
        )

        return VsRandomEpisodeResult(
            agent_steps=1,
            total_plies=2,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
        )

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=2,
        max_agent_steps=20,
        batch_size=8,
        min_replay_size=50,
    )

    assert len(received_calls) == 2

    for call in received_calls:
        assert call["max_agent_steps"] == 20
        assert call["batch_size"] == 8
        assert call["min_replay_size"] == 50

def test_train_against_random_rejects_zero_episodes():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="episodes must be greater than zero",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=0,
        )