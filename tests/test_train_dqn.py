import chess
import pytest
import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import ReplayBuffer
from chess_rl.agents.random_agent import RandomAgent

import chess_rl.training.train_dqn as train_dqn_module

from chess_rl.training.train_dqn import (
    EpisodeResult,
    StepResult,
    VsRandomEpisodeResult,
    TrainingSummary,
    run_and_store_step,
    run_dqn_vs_random_episode,
    run_episode,
    run_single_step,
    train_against_random,
    train_from_replay,
    summarize_training,
    EvaluationSummary,
    evaluate_against_random, 
    main,
)
from pathlib import Path

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

    initial_epsilon = agent.epsilon

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
        training_losses=[],
        final_epsilon=agent.epsilon,
        replay_size=len(replay_buffer),
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
            training_losses=[],
            final_epsilon=agent.epsilon,
            replay_size=len(replay_buffer),
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
            training_losses=[],
            final_epsilon=agent.epsilon,
            replay_size=len(replay_buffer),
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

def test_train_against_random_updates_target_at_configured_frequency(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    target_update_calls = []

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        return VsRandomEpisodeResult(
            agent_steps=1,
            total_plies=2,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
            training_losses=[],
            final_epsilon=agent.epsilon,
            replay_size=len(replay_buffer),
        )

    def fake_update_target():
        target_update_calls.append(True)

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    monkeypatch.setattr(
        agent,
        "update_target",
        fake_update_target,
    )

    train_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=5,
        max_agent_steps=1,
        batch_size=2,
        min_replay_size=4,
        target_update_frequency=2,
    )

    assert len(target_update_calls) == 2

def test_train_against_random_does_not_update_target_too_early(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    target_update_calls = []

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        return VsRandomEpisodeResult(
            agent_steps=1,
            total_plies=2,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
            training_losses=[],
            final_epsilon=agent.epsilon,
            replay_size=len(replay_buffer),
        )

    def fake_update_target():
        target_update_calls.append(True)

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    monkeypatch.setattr(
        agent,
        "update_target",
        fake_update_target,
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
        target_update_frequency=5,
    )

    assert target_update_calls == []

def test_train_against_random_rejects_invalid_target_update_frequency(
    monkeypatch
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="target_update_frequency must be greater than zero",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            target_update_frequency=0,
        )


    monkeypatch,

def test_train_from_replay_decays_epsilon_once_after_training(
        
    monkeypatch,
):
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=10)

    state = torch.zeros((12, 8, 8))

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

def test_train_against_random_does_not_decay_epsilon_without_training(
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

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        return VsRandomEpisodeResult(
            agent_steps=1,
            total_plies=2,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
            training_losses=[],
            final_epsilon=agent.epsilon,
            replay_size=len(replay_buffer),
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

    assert agent.epsilon == 1.0

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

def test_summarize_training_returns_expected_metrics():
    results = [
        VsRandomEpisodeResult(
            agent_steps=2,
            total_plies=4,
            total_reward=1.0,
            done=True,
            truncated=False,
            final_info={},
            training_losses=[0.8, 0.6],
            final_epsilon=0.8,
            replay_size=10,
        ),
        VsRandomEpisodeResult(
            agent_steps=3,
            total_plies=6,
            total_reward=-1.0,
            done=True,
            truncated=False,
            final_info={},
            training_losses=[0.4],
            final_epsilon=0.7,
            replay_size=20,
        ),
    ]

    summary = summarize_training(results)

    assert isinstance(summary, TrainingSummary)
    assert summary.episodes == 2
    assert summary.average_reward == 0.0
    assert summary.average_loss == pytest.approx(0.6)
    assert summary.final_epsilon == 0.7
    assert summary.replay_size == 20

def test_summarize_training_returns_none_without_losses():
    results = [
        VsRandomEpisodeResult(
            agent_steps=2,
            total_plies=4,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={},
            training_losses=[],
            final_epsilon=1.0,
            replay_size=2,
        ),
    ]

    summary = summarize_training(results)

    assert summary.average_loss is None

def test_summarize_training_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match="at least one episode",
    ):
        summarize_training([])

def test_main_runs_multi_episode_training(
    monkeypatch,
    capsys,
):
    fake_results = [
        VsRandomEpisodeResult(
            agent_steps=10,
            total_plies=20,
            total_reward=1.0,
            done=True,
            truncated=False,
            final_info={},
            training_losses=[0.4, 0.2],
            final_epsilon=0.8,
            replay_size=100,
        ),
    ]

    received_training_args = {}

    def fake_train_against_random(
        env,
        agent,
        opponent,
        replay_buffer,
        episodes,
        max_agent_steps,
        batch_size,
        min_replay_size,
        target_update_frequency,
        progress_callback,
        checkpoint_frequency,
        checkpoint_callback,
    ):
        received_training_args["episodes"] = episodes
        received_training_args["max_agent_steps"] = max_agent_steps
        received_training_args["batch_size"] = batch_size
        received_training_args["min_replay_size"] = min_replay_size
        received_training_args["target_update_frequency"] = (
            target_update_frequency
        )
        received_training_args["progress_callback"] = progress_callback
        received_training_args["checkpoint_frequency"] = checkpoint_frequency
        received_training_args["checkpoint_callback"] = checkpoint_callback

        checkpoint_callback(
            checkpoint_frequency,
            agent,
        )

        return fake_results

    saved_paths = []

    def fake_save_checkpoint(self, path):
        saved_paths.append(path)

    monkeypatch.setattr(
        DQNAgent,
        "save_checkpoint",
        fake_save_checkpoint,
    )

    monkeypatch.setattr(
        train_dqn_module,
        "train_against_random",
        fake_train_against_random,
    )

    main()

    assert received_training_args["episodes"] == 100
    assert received_training_args["max_agent_steps"] == 150
    assert received_training_args["batch_size"] == 32
    assert received_training_args["min_replay_size"] == 1_000
    assert received_training_args["target_update_frequency"] == 10
    assert callable(received_training_args["progress_callback"])
    assert received_training_args["checkpoint_frequency"] == 25
    assert callable(received_training_args["checkpoint_callback"])

    output = capsys.readouterr().out

    assert "Episodes: 1" in output
    assert "Average reward: 1.0000" in output
    assert "Average loss: 0.3000" in output
    assert "Final epsilon: 0.8000" in output
    assert "Replay buffer size: 100" in output
    assert len(saved_paths) == 1
    assert saved_paths[0].endswith("latest.pt")

def test_train_against_random_reports_progress(
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
        training_losses=[],
        final_epsilon=1.0,
        replay_size=1,
    )

    def fake_run_dqn_vs_random_episode(*args, **kwargs):
        return fake_result

    received_progress = []

    def progress_callback(
        completed_episodes,
        total_episodes,
        result,
    ):
        received_progress.append(
            (
                completed_episodes,
                total_episodes,
                result,
            )
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
        progress_callback=progress_callback,
    )

    assert len(received_progress) == 3
    assert received_progress[0][0] == 1
    assert received_progress[-1][0] == 3
    assert all(
        progress[1] == 3
        for progress in received_progress
    )

def test_evaluate_against_random_summarizes_results(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent(epsilon=0.4)
    opponent = RandomAgent()

    fake_results = iter([
        VsRandomEpisodeResult(
            agent_steps=10,
            total_plies=20,
            total_reward=1.0,
            done=True,
            truncated=False,
            final_info={"result": "1-0"},
            training_losses=[],
            final_epsilon=0.0,
            replay_size=10,
        ),
        VsRandomEpisodeResult(
            agent_steps=10,
            total_plies=20,
            total_reward=0.0,
            done=True,
            truncated=False,
            final_info={"result": "1/2-1/2"},
            training_losses=[],
            final_epsilon=0.0,
            replay_size=20,
        ),
        VsRandomEpisodeResult(
            agent_steps=10,
            total_plies=20,
            total_reward=-1.0,
            done=True,
            truncated=False,
            final_info={"result": "0-1"},
            training_losses=[],
            final_epsilon=0.0,
            replay_size=30,
        ),
        VsRandomEpisodeResult(
            agent_steps=10,
            total_plies=20,
            total_reward=0.0,
            done=False,
            truncated=True,
            final_info={"result": None},
            training_losses=[],
            final_epsilon=0.0,
            replay_size=40,
        ),
    ])

    observed_epsilons = []

    def fake_run_dqn_vs_random_episode(
        env,
        agent,
        opponent,
        replay_buffer,
        max_agent_steps,
        batch_size,
        min_replay_size,
    ):
        observed_epsilons.append(agent.epsilon)
        return next(fake_results)

    monkeypatch.setattr(
        train_dqn_module,
        "run_dqn_vs_random_episode",
        fake_run_dqn_vs_random_episode,
    )

    summary = evaluate_against_random(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes=4,
    )

    assert isinstance(summary, EvaluationSummary)
    assert summary.episodes == 4
    assert summary.wins == 1
    assert summary.draws == 1
    assert summary.losses == 1
    assert summary.truncated == 1

    assert observed_epsilons == [0.0, 0.0, 0.0, 0.0]
    assert agent.epsilon == 0.4

def test_evaluate_against_random_rejects_zero_episodes():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()

    with pytest.raises(
        ValueError,
        match="episodes must be greater than zero",
    ):
        evaluate_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            episodes=0,
        )

def test_main_loads_existing_checkpoint(
    monkeypatch,
):
    loaded_paths = []

    def fake_load_checkpoint(self, path):
        loaded_paths.append(path)

    def fake_save_checkpoint(self, path):
        pass

    def fake_exists(self):
        return True

    def fake_train_against_random(*args, **kwargs):
        return [
            VsRandomEpisodeResult(
                agent_steps=1,
                total_plies=2,
                total_reward=0.0,
                done=False,
                truncated=True,
                final_info={},
                training_losses=[],
                final_epsilon=1.0,
                replay_size=1,
            )
        ]

    monkeypatch.setattr(
        DQNAgent,
        "load_checkpoint",
        fake_load_checkpoint,
    )

    monkeypatch.setattr(
        DQNAgent,
        "save_checkpoint",
        fake_save_checkpoint,
    )

    monkeypatch.setattr(
        Path,
        "exists",
        fake_exists,
    )

    monkeypatch.setattr(
        train_dqn_module,
        "train_against_random",
        fake_train_against_random,
    )

    main()

    assert len(loaded_paths) == 1
    assert loaded_paths[0].endswith("latest.pt")


def test_main_does_not_load_missing_checkpoint(
    monkeypatch,
):
    load_calls = []

    def fake_load_checkpoint(self, path):
        load_calls.append(path)

    def fake_save_checkpoint(self, path):
        pass

    def fake_exists(self):
        return False

    def fake_train_against_random(*args, **kwargs):
        return [
            VsRandomEpisodeResult(
                agent_steps=1,
                total_plies=2,
                total_reward=0.0,
                done=False,
                truncated=True,
                final_info={},
                training_losses=[],
                final_epsilon=1.0,
                replay_size=1,
            )
        ]

    monkeypatch.setattr(
        DQNAgent,
        "load_checkpoint",
        fake_load_checkpoint,
    )

    monkeypatch.setattr(
        DQNAgent,
        "save_checkpoint",
        fake_save_checkpoint,
    )

    monkeypatch.setattr(
        Path,
        "exists",
        fake_exists,
    )

    monkeypatch.setattr(
        train_dqn_module,
        "train_against_random",
        fake_train_against_random,
    )

    main()

    assert load_calls == []

def test_train_against_random_checkpoints_at_configured_frequency(
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
        training_losses=[],
        final_epsilon=1.0,
        replay_size=1,
    )

    def fake_run_dqn_vs_random_episode(*args, **kwargs):
        return fake_result

    checkpoint_calls = []

    def checkpoint_callback(
        completed_episodes,
        received_agent,
    ):
        checkpoint_calls.append(
            (completed_episodes, received_agent)
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
        episodes=5,
        checkpoint_frequency=2,
        checkpoint_callback=checkpoint_callback,
    )

    assert [call[0] for call in checkpoint_calls] == [2, 4]
    assert all(
        call[1] is agent
        for call in checkpoint_calls
    )

def test_train_against_random_rejects_invalid_checkpoint_frequency():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="checkpoint_frequency must be greater than zero",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            checkpoint_frequency=0,
        )

def test_train_against_random_requires_checkpoint_callback():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="checkpoint_callback is required",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            checkpoint_frequency=10,
        )