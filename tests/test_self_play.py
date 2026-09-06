import torch
import pytest

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN

from chess_rl.training.self_play import (
    create_frozen_opponent,
    run_dqn_vs_frozen_episode,
    select_frozen_opponent_move,
    create_frozen_opponent_selector,
)

import chess

import chess_rl.training.self_play as self_play_module
from chess_rl.training.self_play import train_against_frozen

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