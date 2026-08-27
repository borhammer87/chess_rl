import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.training.self_play import (
    create_frozen_opponent,
    select_frozen_opponent_move,
)

import chess

import chess_rl.training.self_play as self_play_module

from chess_rl.utils.action_encoder import encode_move

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

    move = select_frozen_opponent_move(
        opponent=opponent,
        board=board,
    )

    assert move == expected_move