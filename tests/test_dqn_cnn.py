import chess
import torch
import pytest

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import Transition
from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.utils.action_encoder import ACTION_SIZE

def test_agent_selects_action_in_valid_range():
    agent = DQNAgent()

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    board = chess.Board()

    action = agent.select_action(
        state=state,
        legal_moves=list(board.legal_moves),
    )

    assert isinstance(action, int)
    assert 0 <= action < ACTION_SIZE


def test_random_exploration_selects_only_legal_actions():
    agent = DQNAgent(epsilon=1.0)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    board = chess.Board()

    legal_moves = list(board.legal_moves)
    legal_actions = {
        encode_move(move)
        for move in legal_moves
    }

    for _ in range(100):
        action = agent.select_action(
            state=state,
            legal_moves=legal_moves,
        )

        assert action in legal_actions


def test_greedy_policy_selects_only_legal_actions():
    agent = DQNAgent(epsilon=0.0)

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    board = chess.Board()

    legal_moves = list(board.legal_moves)
    legal_actions = {
        encode_move(move)
        for move in legal_moves
    }

    action = agent.select_action(
        state=state,
        legal_moves=legal_moves,
    )

    assert action in legal_actions


def test_select_action_rejects_empty_legal_move_list():
    agent = DQNAgent()

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    try:
        agent.select_action(
            state=state,
            legal_moves=[],
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass


def test_agent_train_step_returns_float():
    agent = DQNAgent()

    state = torch.zeros((BOARD_CHANNELS, 8, 8))

    batch = [
        Transition(
            state=state,
            action=1,
            reward=1.0,
            next_state=state,
            done=False,
            next_legal_actions=[1, 2, 3],
        )
        for _ in range(4)
    ]

    loss = agent.train_step(batch)

    assert isinstance(loss, float)


def test_target_network_syncs_policy_weights():
    agent = DQNAgent()

    old_target_weights = (
        agent.target_net.features[0].weight.detach().clone()
    )

    with torch.no_grad():
        agent.policy_net.features[0].weight.add_(1.0)

    agent.update_target()

    new_target_weights = agent.target_net.features[0].weight

    assert not torch.allclose(
        old_target_weights,
        new_target_weights,
    )

    assert torch.allclose(
        agent.policy_net.features[0].weight,
        agent.target_net.features[0].weight,
    )


def test_epsilon_decay_respects_minimum():
    agent = DQNAgent(
        epsilon=0.11,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert agent.epsilon == 0.1

def test_dqn_cnn_accepts_encoded_board_shape():
    model = DQNCNN()

    states = torch.zeros(
        (2, BOARD_CHANNELS, 8, 8)
    )

    output = model(states)

    assert output.shape == (
        2,
        ACTION_SIZE,
    )