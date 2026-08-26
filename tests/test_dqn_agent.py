import chess
import torch
import pytest
import copy
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import Transition
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

    with pytest.raises(
        ValueError,
        match="without legal moves",
    ):
        agent.select_action(
            state=state,
            legal_moves=[],
        )


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

def test_checkpoint_restores_network_parameters(tmp_path):
    agent = DQNAgent()

    checkpoint_path = tmp_path / "checkpoint.pt"

    original_parameters = [
        parameter.detach().clone()
        for parameter in agent.policy_net.parameters()
    ]

    agent.save_checkpoint(str(checkpoint_path))

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.add_(1.0)

    agent.load_checkpoint(str(checkpoint_path))

    for restored, original in zip(
        agent.policy_net.parameters(),
        original_parameters,
    ):
        assert torch.equal(restored, original)

def test_checkpoint_restores_epsilon(tmp_path):
    agent = DQNAgent(epsilon=0.4)

    checkpoint_path = tmp_path / "checkpoint.pt"

    agent.save_checkpoint(str(checkpoint_path))

    agent.epsilon = 0.9

    agent.load_checkpoint(str(checkpoint_path))

    assert agent.epsilon == 0.4

def test_agent_state_dict_restores_training_state():
    agent = DQNAgent(epsilon=0.4)

    original_parameters = [
        parameter.detach().clone()
        for parameter in agent.policy_net.parameters()
    ]

    saved_state = copy.deepcopy(agent.state_dict())

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.add_(1.0)

    agent.epsilon = 0.9

    agent.load_state_dict(saved_state)

    for restored, original in zip(
        agent.policy_net.parameters(),
        original_parameters,
    ):
        assert torch.equal(restored, original)

    assert agent.epsilon == 0.4