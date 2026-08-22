from pathlib import Path

import pytest

import chess_rl.training.train_dqn as train_dqn_module
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.results import (
    EvaluationSummary,
    TrainingSummary,
    VsRandomEpisodeResult,
)
from chess_rl.training.train_dqn import (
    evaluate_against_random,
    main,
    summarize_training,
    train_against_random,
    score_evaluation,
)
from chess_rl.utils.replay_buffer import ReplayBuffer


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
        evaluation_frequency,
        evaluation_callback,
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
        received_training_args["evaluation_frequency"] = (
            evaluation_frequency
        )
        received_training_args["evaluation_callback"] = (
            evaluation_callback
        )

        checkpoint_callback(
            checkpoint_frequency,
            agent,
        )

        return fake_results

    saved_paths = []

    def fake_save_training_checkpoint(
        path,
        agent,
        replay_buffer,
    ):
        saved_paths.append(path)

    monkeypatch.setattr(
        train_dqn_module,
        "save_training_checkpoint",
        fake_save_training_checkpoint,
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
    assert received_training_args["evaluation_frequency"] == 25
    assert callable(received_training_args["evaluation_callback"])
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

    def fake_load_training_checkpoint(
        path,
        agent,
        replay_buffer,
    ):
        loaded_paths.append(path)

    def fake_save_training_checkpoint(
        path,
        agent,
        replay_buffer,
    ):
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
        train_dqn_module,
        "load_training_checkpoint",
        fake_load_training_checkpoint,
    )

    monkeypatch.setattr(
        train_dqn_module,
        "save_training_checkpoint",
        fake_save_training_checkpoint,
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

def test_train_against_random_evaluates_at_configured_frequency(
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

    evaluation_calls = []

    def evaluation_callback(
        completed_episodes,
        received_agent,
    ):
        evaluation_calls.append(
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
        evaluation_frequency=2,
        evaluation_callback=evaluation_callback,
    )

    assert [call[0] for call in evaluation_calls] == [2, 4]

    assert all(
        call[1] is agent
        for call in evaluation_calls
    )

def test_train_against_random_rejects_invalid_evaluation_frequency():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="evaluation_frequency must be greater than zero",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            evaluation_frequency=0,
        )

def test_train_against_random_requires_evaluation_callback():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(capacity=100)

    with pytest.raises(
        ValueError,
        match="evaluation_callback is required",
    ):
        train_against_random(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            evaluation_frequency=10,
        )

def test_score_evaluation_uses_chess_points():
    evaluation = EvaluationSummary(
        episodes=4,
        wins=2,
        draws=1,
        losses=1,
        truncated=0,
    )

    score = score_evaluation(evaluation)

    assert score == pytest.approx(0.625)

def test_score_evaluation_penalizes_truncated_games():
    evaluation = EvaluationSummary(
        episodes=4,
        wins=1,
        draws=0,
        losses=0,
        truncated=3,
    )

    score = score_evaluation(evaluation)

    assert score == pytest.approx(0.25)

def test_score_evaluation_rejects_zero_episodes():
    evaluation = EvaluationSummary(
        episodes=0,
        wins=0,
        draws=0,
        losses=0,
        truncated=0,
    )

    with pytest.raises(
        ValueError,
        match="at least one episode",
    ):
        score_evaluation(evaluation)