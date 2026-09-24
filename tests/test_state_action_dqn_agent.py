import chess
import pytest
import torch

from chess_rl.agents.state_action_dqn_agent import (
    StateActionDQNAgent,
)
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.utils.replay_buffer import Transition

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

    agent.update_target()

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

def test_state_action_train_step_uses_zero_future_value_for_terminal_transition(
    monkeypatch,
):
    agent = StateActionDQNAgent(
        gamma=0.99
    )

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = encode_move(
        chess.Move.from_uci("e2e4")
    )

    batch = [
        Transition(
            state=state,
            action=action,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        )
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Target network must not evaluate terminal next states."
        )

    monkeypatch.setattr(
        agent.target_net,
        "evaluate_legal_action_maxes",
        fail_if_called,
    )

    loss = agent.train_step(batch)

    assert isinstance(loss, float)

def test_state_action_train_step_rejects_non_terminal_without_legal_actions():
    agent = StateActionDQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = encode_move(
        chess.Move.from_uci("e2e4")
    )

    batch = [
        Transition(
            state=state,
            action=action,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[],
        )
    ]

    with pytest.raises(
        ValueError,
        match="must have legal next actions",
    ):
        agent.train_step(batch)

def test_state_action_train_step_can_return_td_errors():
    agent = StateActionDQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = encode_move(
        chess.Move.from_uci("e2e4")
    )

    batch = [
        Transition(
            state=state,
            action=action,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        )
    ]

    loss, td_errors = agent.train_step(
        batch,
        return_td_errors=True,
    )

    assert isinstance(loss, float)
    assert len(td_errors) == 1
    assert td_errors[0] >= 0.0

def test_state_action_train_step_rejects_wrong_number_of_weights():
    agent = StateActionDQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = encode_move(
        chess.Move.from_uci("e2e4")
    )

    batch = [
        Transition(
            state=state,
            action=action,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        )
    ]

    weights = torch.ones(2)

    with pytest.raises(
        ValueError,
        match="weights must match",
    ):
        agent.train_step(
            batch,
            weights=weights,
        )

def test_state_action_train_step_batches_policy_and_target_state_encoding(
    monkeypatch,
):
    agent = StateActionDQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action = encode_move(
        chess.Move.from_uci("e2e4")
    )

    next_legal_actions = [
        encode_move(
            chess.Move.from_uci("e7e5")
        ),
        encode_move(
            chess.Move.from_uci("d7d5")
        ),
    ]

    # 8 transitions:
    # 5 non-terminal -> must enter target CNN
    # 3 terminal     -> must NOT enter target CNN
    batch = []

    for index in range(8):
        done = index >= 5

        batch.append(
            Transition(
                state=state,
                action=action,
                reward=0.0,
                next_state=state,
                done=done,
                next_legal_actions=(
                    []
                    if done
                    else next_legal_actions
                ),
            )
        )

    policy_encode_state = (
        agent.policy_net.encode_state
    )
    target_encode_state = (
        agent.target_net.encode_state
    )

    policy_batch_sizes = []
    target_batch_sizes = []

    def counting_policy_encode_state(states):
        policy_batch_sizes.append(
            states.shape[0]
        )

        return policy_encode_state(states)

    def counting_target_encode_state(states):
        target_batch_sizes.append(
            states.shape[0]
        )

        return target_encode_state(states)

    monkeypatch.setattr(
        agent.policy_net,
        "encode_state",
        counting_policy_encode_state,
    )

    monkeypatch.setattr(
        agent.target_net,
        "encode_state",
        counting_target_encode_state,
    )

    loss = agent.train_step(batch)

    assert isinstance(loss, float)

    assert policy_batch_sizes == [8]
    assert target_batch_sizes == [5]

def test_state_action_agent_state_dict_restores_training_state():
    agent = StateActionDQNAgent(
        epsilon=0.4
    )

    with torch.no_grad():
        parameter = next(
            agent.policy_net.parameters()
        )
        parameter.add_(1.0)

    saved_state = agent.state_dict()

    restored_agent = StateActionDQNAgent(
        epsilon=0.9
    )

    restored_agent.load_state_dict(
        saved_state
    )

    assert restored_agent.epsilon == 0.4

    for original, restored in zip(
        agent.policy_net.parameters(),
        restored_agent.policy_net.parameters(),
    ):
        assert torch.equal(
            original,
            restored,
        )

    for original, restored in zip(
        agent.target_net.parameters(),
        restored_agent.target_net.parameters(),
    ):
        assert torch.equal(
            original,
            restored,
        )