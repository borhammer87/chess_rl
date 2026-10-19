import chess
import pytest
import torch
from chess_rl.models.spatial_state_action_dqn import SpatialStateActionDQN
from chess_rl.models.state_action_dqn import StateActionDQN
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


def test_state_action_train_step_bootstraps_non_terminal_transition(
    monkeypatch,
):
    agent = StateActionDQNAgent(gamma=0.9)

    parameter = next(agent.policy_net.parameters())

    def fake_policy_q_values(states, actions):
        return parameter.sum() * 0 + torch.zeros(
            len(actions),
        )

    monkeypatch.setattr(
        agent.policy_net,
        "evaluate_state_action_pairs",
        fake_policy_q_values,
    )

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    action = encode_move(chess.Move.from_uci("e2e4"))

    batch = [
        Transition(
            state=state,
            action=action,
            reward=-0.1,
            next_state=state,
            done=False,
            next_legal_actions=[action],
        )
    ]

    calls = []

    def fake_next_q_values(states, legal_actions):
        calls.append((states, legal_actions))
        return torch.tensor([0.5])

    monkeypatch.setattr(
        agent.target_net,
        "evaluate_legal_action_maxes",
        fake_next_q_values,
    )

    _, td_errors = agent.train_step(
        batch,
        return_td_errors=True,
    )

    assert len(calls) == 1
    assert calls[0][1] == [[action]]
    assert len(td_errors) == 1
    assert td_errors[0] == pytest.approx(0.35)


def test_state_action_train_step_uses_exact_terminal_targets(
    monkeypatch,
):
    agent = StateActionDQNAgent()

    state = torch.zeros((BOARD_CHANNELS, 8, 8))
    rook_action = encode_move(chess.Move.from_uci("e3d5"))
    pawn_action = encode_move(chess.Move.from_uci("e3f5"))

    batch = [
        Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=state,
            done=True,
            next_legal_actions=[],
        )
        for action, reward in (
            (rook_action, 0.05),
            (pawn_action, 0.01),
        )
    ]

    parameter = next(agent.policy_net.parameters())

    def fake_policy_q_values(states, actions):
        assert actions == [rook_action, pawn_action]
        return parameter.sum() * 0 + torch.tensor([0.02, 0.03])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Terminal transitions must not bootstrap.")

    monkeypatch.setattr(
        agent.policy_net,
        "evaluate_state_action_pairs",
        fake_policy_q_values,
    )
    monkeypatch.setattr(
        agent.target_net,
        "evaluate_legal_action_maxes",
        fail_if_called,
    )

    loss, td_errors = agent.train_step(
        batch,
        return_td_errors=True,
    )

    assert loss == pytest.approx(0.00065)
    assert td_errors == pytest.approx([0.03, 0.02])


def test_state_action_agent_uses_original_network_by_default():
    agent = StateActionDQNAgent()

    assert type(agent.policy_net) is StateActionDQN
    assert type(agent.target_net) is StateActionDQN


def test_state_action_agent_can_use_spatial_networks():
    agent = StateActionDQNAgent(spatial=True)

    assert isinstance(agent.policy_net, SpatialStateActionDQN)
    assert isinstance(agent.target_net, SpatialStateActionDQN)

    for policy_parameter, target_parameter in zip(
        agent.policy_net.parameters(),
        agent.target_net.parameters(),
    ):
        torch.testing.assert_close(
            policy_parameter,
            target_parameter,
        )


def test_spatial_agent_trains_non_terminal_and_syncs_target():
    from chess_rl.utils.board_encoder import encode_board

    torch.manual_seed(42)
    agent = StateActionDQNAgent(spatial=True, gamma=0.9)

    board = chess.Board()
    next_board = board.copy()
    next_board.push_uci("e2e4")

    action = encode_move(chess.Move.from_uci("e2e4"))
    next_actions = [
        encode_move(move)
        for move in next_board.legal_moves
    ]

    transition = Transition(
        state=encode_board(board),
        action=action,
        reward=0.05,
        next_state=encode_board(next_board),
        done=False,
        next_legal_actions=next_actions,
    )

    policy_before = {
        name: value.detach().clone()
        for name, value in agent.policy_net.state_dict().items()
    }
    target_before = {
        name: value.detach().clone()
        for name, value in agent.target_net.state_dict().items()
    }

    loss, td_errors = agent.train_step(
        [transition],
        return_td_errors=True,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert len(td_errors) == 1
    assert torch.isfinite(torch.tensor(td_errors)).all()

    policy_after = agent.policy_net.state_dict()
    target_after = agent.target_net.state_dict()

    assert any(
        not torch.equal(policy_before[name], policy_after[name])
        for name in policy_before
    )
    assert all(
        torch.equal(target_before[name], target_after[name])
        for name in target_before
    )

    agent.update_target()

    for name, value in agent.policy_net.state_dict().items():
        torch.testing.assert_close(
            value,
            agent.target_net.state_dict()[name],
        )

def test_state_action_agent_uses_cpu_by_default():
    agent = StateActionDQNAgent()

    assert agent.device == torch.device("cpu")
    assert next(agent.policy_net.parameters()).device.type == "cpu"
    assert next(agent.target_net.parameters()).device.type == "cpu"


def test_state_action_agent_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        StateActionDQNAgent(device="cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_state_action_agent_creates_cuda_networks():
    agent = StateActionDQNAgent(device="cuda", spatial=True)

    assert next(agent.policy_net.parameters()).device.type == "cuda"
    assert next(agent.target_net.parameters()).device.type == "cuda"