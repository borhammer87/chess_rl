import chess
import pytest
import torch

from chess_rl.agents.state_action_dqn_agent import (
    StateActionDQNAgent,
)
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.board_encoder import BOARD_CHANNELS

def test_state_action_agent_random_exploration_selects_only_legal_actions():
    agent = StateActionDQNAgent(
        epsilon=1.0
    )

    board = chess.Board()
    legal_moves = list(board.legal_moves)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    legal_actions = {
        encode_move(move)
        for move in legal_moves
    }

    for _ in range(100):
        action = agent.select_action(
            state,
            legal_moves,
        )

        assert action in legal_actions

def test_state_action_agent_greedy_policy_selects_only_legal_actions():
    agent = StateActionDQNAgent(
        epsilon=0.0
    )

    board = chess.Board()
    legal_moves = list(board.legal_moves)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = agent.select_action(
        state,
        legal_moves,
    )

    assert action in {
        encode_move(move)
        for move in legal_moves
    }

def test_state_action_agent_rejects_empty_legal_move_list():
    agent = StateActionDQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    with pytest.raises(
        ValueError,
        match="without legal moves",
    ):
        agent.select_action(
            state,
            [],
        )

def test_state_action_agent_target_network_syncs_policy_weights():
    agent = StateActionDQNAgent()

    with torch.no_grad():
        parameter = next(
            agent.policy_net.parameters()
        )
        parameter.add_(1.0)

    policy_parameter = next(
        agent.policy_net.parameters()
    )
    target_parameter = next(
        agent.target_net.parameters()
    )

    assert not torch.equal(
        policy_parameter,
        target_parameter,
    )

    agent.update_target_network()

    assert torch.equal(
        policy_parameter,
        target_parameter,
    )

def test_state_action_agent_epsilon_decay_respects_minimum():
    agent = StateActionDQNAgent(
        epsilon=0.11,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.1