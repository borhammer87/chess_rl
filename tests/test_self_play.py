import torch
import pytest
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN

from chess_rl.training.self_play import (
    create_frozen_opponent,
    run_dqn_vs_frozen_episode,
    select_frozen_opponent_move,
    create_frozen_opponent_selector,
    update_frozen_opponent,
    train_against_frozen,
    evaluate_against_frozen,
)

import chess
import chess_rl.training.self_play as self_play_module
from chess_rl.utils.action_encoder import encode_move
from chess_rl.env.chess_env import ChessEnv
from chess_rl.utils.replay_buffer import ReplayBuffer



def test_create_frozen_opponent_copies_policy_weights():
    agent = DQNAgent()

    opponent = create_frozen_opponent(agent)

    assert isinstance(opponent, DQNCNN)

    for policy_parameter, opponent_parameter in zip(
        agent.policy_net.parameters(),
        opponent.parameters(),
    ):
        assert torch.equal(
            policy_parameter,
            opponent_parameter,
        )


def test_frozen_opponent_is_independent_from_policy():
    agent = DQNAgent()

    opponent = create_frozen_opponent(agent)

    opponent_parameters_before = [
        parameter.detach().clone()
        for parameter in opponent.parameters()
    ]

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.add_(1.0)

    for opponent_parameter, original_parameter in zip(
        opponent.parameters(),
        opponent_parameters_before,
    ):
        assert torch.equal(
            opponent_parameter,
            original_parameter,
        )


def test_frozen_opponent_parameters_do_not_require_gradients():
    agent = DQNAgent()

    opponent = create_frozen_opponent(agent)

    assert all(
        not parameter.requires_grad
        for parameter in opponent.parameters()
    )

def test_frozen_opponent_selects_legal_move(
    monkeypatch,
):
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)

    board = chess.Board()

    expected_move = chess.Move.from_uci(
        "e2e4"
    )

    expected_action = encode_move(
        expected_move
    )

    def fake_select_greedy_action(
        network,
        state,
        legal_moves,
    ):
        assert network is opponent
        assert expected_move in legal_moves

        return expected_action

    monkeypatch.setattr(
        self_play_module,
        "select_greedy_action",
        fake_select_greedy_action,
    )

    legal_moves = list(
        board.legal_moves
    )

    move = select_frozen_opponent_move(
        opponent=opponent,
        board=board,
        legal_moves=legal_moves,
    )

    assert move == expected_move

def test_dqn_vs_frozen_stores_only_learner_transitions():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    opponent = create_frozen_opponent(
        agent
    )

    replay_buffer = ReplayBuffer(
        capacity=100
    )

    result = run_dqn_vs_frozen_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=3,
    )

    assert (
        len(replay_buffer)
        == result.agent_steps
    )

def test_one_self_play_agent_step_contains_both_moves():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    opponent = create_frozen_opponent(
        agent
    )

    replay_buffer = ReplayBuffer(
        capacity=10
    )

    result = run_dqn_vs_frozen_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
    )

    assert result.agent_steps == 1
    assert result.total_plies == 2
    assert len(replay_buffer) == 1

def test_black_learner_starts_after_frozen_white_move():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)

    opponent = create_frozen_opponent(
        agent
    )

    replay_buffer = ReplayBuffer(
        capacity=10
    )

    result = run_dqn_vs_frozen_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=1,
        agent_color=chess.BLACK,
    )

    assert result.agent_steps == 1
    assert result.total_plies == 3
    assert len(replay_buffer) == 1

def test_frozen_opponent_selector_returns_legal_move():
    agent = DQNAgent()

    opponent = create_frozen_opponent(
        agent
    )

    selector = create_frozen_opponent_selector(
        opponent
    )

    board = chess.Board()

    legal_moves = list(
        board.legal_moves
    )

    move = selector(
        board,
        legal_moves,
    )

    assert move in legal_moves

def test_train_against_frozen_runs_multiple_episodes():
    env = ChessEnv()
    agent = DQNAgent(epsilon=1.0)
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=100)

    results = train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
    )

    assert len(results) == 3
    assert len(replay_buffer) == 3

def test_train_against_frozen_alternates_agent_color(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    colors = []

    original_run_episode = (
        self_play_module.run_dqn_vs_frozen_episode
    )

    def recording_run_episode(*args, **kwargs):
        colors.append(kwargs["agent_color"])

        return original_run_episode(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        self_play_module,
        "run_dqn_vs_frozen_episode",
        recording_run_episode,
    )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
    )

    assert colors == [
        chess.WHITE,
        chess.BLACK,
        chess.WHITE,
    ]

def test_train_against_frozen_rejects_non_positive_episodes():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="episodes must be greater than zero",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=0,
        )

def test_update_frozen_opponent_copies_current_policy():
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.add_(1.0)

    update_frozen_opponent(
        agent=agent,
        opponent=opponent,
    )

    for policy_parameter, opponent_parameter in zip(
        agent.policy_net.parameters(),
        opponent.parameters(),
    ):
        assert torch.equal(
            policy_parameter,
            opponent_parameter,
        )

def test_train_against_frozen_updates_opponent_periodically(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    updates = []

    def fake_update_frozen_opponent(
        agent,
        opponent,
    ):
        updates.append(True)

    monkeypatch.setattr(
        self_play_module,
        "update_frozen_opponent",
        fake_update_frozen_opponent,
    )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=5,
        max_agent_steps=1,
        opponent_update_frequency=2,
    )

    assert len(updates) == 2

def test_train_against_frozen_rejects_invalid_update_frequency():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="opponent_update_frequency must be greater than zero",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            opponent_update_frequency=0,
        )

def test_train_against_frozen_updates_target_periodically(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    updates = []

    def fake_update_target():
        updates.append(True)

    monkeypatch.setattr(
        agent,
        "update_target",
        fake_update_target,
    )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=5,
        max_agent_steps=1,
        target_update_frequency=2,
    )

    assert len(updates) == 2


def test_train_against_frozen_rejects_invalid_target_update_frequency():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="target_update_frequency must be greater than zero",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            target_update_frequency=0,
        )

def test_train_against_frozen_can_alternate_starting_with_black(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    colors = []

    original_run_episode = (
        self_play_module.run_dqn_vs_frozen_episode
    )

    def recording_run_episode(*args, **kwargs):
        colors.append(kwargs["agent_color"])

        return original_run_episode(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        self_play_module,
        "run_dqn_vs_frozen_episode",
        recording_run_episode,
    )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
        agent_color=chess.BLACK,
    )

    assert colors == [
        chess.BLACK,
        chess.WHITE,
        chess.BLACK,
    ]

def test_train_against_frozen_can_keep_same_color(
    monkeypatch,
):
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    colors = []

    original_run_episode = (
        self_play_module.run_dqn_vs_frozen_episode
    )

    def recording_run_episode(*args, **kwargs):
        colors.append(kwargs["agent_color"])

        return original_run_episode(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        self_play_module,
        "run_dqn_vs_frozen_episode",
        recording_run_episode,
    )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=2,
        max_agent_steps=1,
        agent_color=chess.BLACK,
        alternate_colors=False,
    )

    assert colors == [
        chess.BLACK,
        chess.BLACK,
    ]

def test_train_against_frozen_rejects_invalid_agent_color():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="agent_color must be chess.WHITE or chess.BLACK",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            agent_color=None,
        )

def test_train_against_frozen_reports_progress():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    progress_calls = []

    def progress_callback(
        completed,
        total,
        result,
    ):
        progress_calls.append(
            (completed, total, result)
        )

    results = train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=3,
        max_agent_steps=1,
        progress_callback=progress_callback,
    )

    assert len(progress_calls) == 3

    assert [
        (completed, total)
        for completed, total, _ in progress_calls
    ] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]

    assert [
        result
        for _, _, result in progress_calls
    ] == results

def test_train_against_frozen_calls_checkpoint_callback():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    checkpoint_episodes = []

    def checkpoint_callback(
        completed_episodes,
        callback_agent,
    ):
        checkpoint_episodes.append(
            (completed_episodes, callback_agent)
        )

    train_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        episodes=4,
        max_agent_steps=1,
        checkpoint_frequency=2,
        checkpoint_callback=checkpoint_callback,
    )

    assert checkpoint_episodes == [
        (2, agent),
        (4, agent),
    ]

def test_train_against_frozen_rejects_invalid_checkpoint_frequency():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="checkpoint_frequency must be greater than zero",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            checkpoint_frequency=0,
            checkpoint_callback=lambda *_: None,
        )

def test_train_against_frozen_requires_checkpoint_callback():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)
    replay_buffer = ReplayBuffer(capacity=10)

    with pytest.raises(
        ValueError,
        match="checkpoint_callback is required",
    ):
        train_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            replay_buffer=replay_buffer,
            episodes=1,
            checkpoint_frequency=1,
        )

def test_evaluate_against_frozen_returns_summary():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)

    summary = evaluate_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes=3,
        max_agent_steps=1,
    )

    assert summary.episodes == 3

def test_evaluate_against_frozen_restores_epsilon():
    env = ChessEnv()
    agent = DQNAgent(epsilon=0.7)
    opponent = create_frozen_opponent(agent)

    evaluate_against_frozen(
        env=env,
        agent=agent,
        opponent=opponent,
        episodes=1,
        max_agent_steps=1,
    )

    assert agent.epsilon == pytest.approx(0.7)

def test_evaluate_against_frozen_rejects_zero_episodes():
    env = ChessEnv()
    agent = DQNAgent()
    opponent = create_frozen_opponent(agent)

    with pytest.raises(
        ValueError,
        match="episodes must be greater than zero",
    ):
        evaluate_against_frozen(
            env=env,
            agent=agent,
            opponent=opponent,
            episodes=0,
        )