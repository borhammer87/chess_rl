import chess
import torch
import pytest
import copy
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.utils.action_encoder import encode_move
from chess_rl.utils.replay_buffer import Transition
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.utils.action_encoder import (
    ACTION_SIZE,
    encode_move,
)
import chess_rl.agents.dqn_agent as dqn_agent_module
from chess_rl.agents.random_agent import RandomAgent
from chess_rl.env.chess_env import ChessEnv
from chess_rl.training.episodes import run_dqn_vs_random_episode
from chess_rl.utils.replay_buffer import ReplayBuffer


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

def test_train_step_uses_only_legal_next_actions(
    monkeypatch,
):
    agent = DQNAgent(gamma=1.0)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    legal_action = 10
    illegal_action = 20

    batch = [
        Transition(
            state=state,
            action=1,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[legal_action],
        )
    ]

    def fake_target_forward(states):
        q_values = torch.zeros(
            (states.shape[0], ACTION_SIZE)
        )

        q_values[:, legal_action] = 10.0
        q_values[:, illegal_action] = 100.0

        return q_values

    monkeypatch.setattr(
        agent.target_net,
        "forward",
        fake_target_forward,
    )

    captured_targets = []

    def fake_mse_loss(
        q_values,
        targets,
        reduction="mean",
    ):
        captured_targets.append(
            targets.detach().clone()
        )

        return q_values.sum() * 0

    monkeypatch.setattr(
        dqn_agent_module.F,
        "mse_loss",
        fake_mse_loss,
    )

    agent.train_step(batch)

    assert captured_targets[0].item() == 10.0

def test_train_step_uses_zero_future_value_for_terminal_transition(
    monkeypatch,
):
    agent = DQNAgent(gamma=1.0)

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    batch = [
        Transition(
            state=state,
            action=1,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        )
    ]

    captured_targets = []

    def fake_mse_loss(
        q_values,
        targets,
        reduction="mean",
    ):
        captured_targets.append(
            targets.detach().clone()
        )

        return q_values.sum() * 0

    monkeypatch.setattr(
        dqn_agent_module.F,
        "mse_loss",
        fake_mse_loss,
    )

    agent.train_step(batch)

    assert captured_targets[0].item() == 1.

def test_train_step_rejects_non_terminal_transition_without_legal_actions():
    agent = DQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    batch = [
        Transition(
            state=state,
            action=1,
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

def test_agent_learns_to_prefer_rewarded_action():
    torch.manual_seed(0)

    agent = DQNAgent(
        lr=1e-3,
        epsilon=0.0,
    )

    board = chess.Board()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    good_move = chess.Move.from_uci("e2e4")
    bad_move = chess.Move.from_uci("d2d4")

    good_action = encode_move(good_move)
    bad_action = encode_move(bad_move)

    batch = [
        Transition(
            state=state,
            action=good_action,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
        Transition(
            state=state,
            action=bad_action,
            reward=-1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
    ]

    for _ in range(100):
        agent.train_step(batch)

    with torch.no_grad():
        q_values = agent.policy_net(
            state.unsqueeze(0)
        )[0]

    assert q_values[good_action] > q_values[bad_action]

    selected_action = agent.select_action(
        state=state,
        legal_moves=[
            good_move,
            bad_move,
        ],
    )

    assert selected_action == good_action

def mean_absolute_td_error(
    agent: DQNAgent,
    batch: list[Transition],
) -> float:
    states = torch.stack(
        [transition.state for transition in batch]
    )

    actions = torch.tensor(
        [transition.action for transition in batch]
    )

    rewards = torch.tensor(
        [transition.reward for transition in batch],
        dtype=torch.float32,
    )

    next_states = torch.stack(
        [transition.next_state for transition in batch]
    )

    with torch.no_grad():
        q_values = agent.policy_net(states)
        selected_q_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        all_next_q_values = agent.target_net(
            next_states
        )

        next_q_values = torch.zeros(
            len(batch),
            dtype=torch.float32,
        )

        for index, transition in enumerate(batch):
            if transition.done:
                continue

            legal_actions = torch.tensor(
                transition.next_legal_actions,
                dtype=torch.long,
            )

            next_q_values[index] = all_next_q_values[
                index,
                legal_actions,
            ].max()

        targets = (
            rewards
            + agent.gamma * next_q_values
        )

        td_errors = torch.abs(
            targets - selected_q_values
        )

    return td_errors.mean().item()

def test_agent_can_reduce_td_error_on_real_chess_transitions():
    torch.manual_seed(0)

    agent = DQNAgent(
        lr=1e-3,
        epsilon=1.0,
    )

    env = ChessEnv()
    opponent = RandomAgent()
    replay_buffer = ReplayBuffer(
        capacity=100
    )

    result = run_dqn_vs_random_episode(
        env=env,
        agent=agent,
        opponent=opponent,
        replay_buffer=replay_buffer,
        max_agent_steps=10,
        batch_size=32,
        min_replay_size=100,
    )

    assert result.training_losses == []
    assert len(replay_buffer) == 10

    batch = list(
        replay_buffer.buffer
    )

    initial_td_error = mean_absolute_td_error(
        agent,
        batch,
    )

    for _ in range(200):
        agent.train_step(batch)

    final_td_error = mean_absolute_td_error(
        agent,
        batch,
    )

    assert final_td_error < initial_td_error

def test_train_step_can_return_td_errors():
    agent = DQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    batch = [
        Transition(
            state=state,
            action=1,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
        Transition(
            state=state,
            action=2,
            reward=-1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
    ]

    loss, td_errors = agent.train_step(
        batch,
        return_td_errors=True,
    )

    assert isinstance(loss, float)
    assert isinstance(td_errors, list)
    assert len(td_errors) == len(batch)
    assert all(
        td_error >= 0
        for td_error in td_errors
    )

def test_train_step_applies_importance_sampling_weights(
    monkeypatch,
):
    agent = DQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    batch = [
        Transition(
            state=state,
            action=1,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
        Transition(
            state=state,
            action=2,
            reward=-1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
    ]

    captured_reduction = []

    original_mse_loss = dqn_agent_module.F.mse_loss

    def capture_mse_loss(
        input,
        target,
        reduction="mean",
    ):
        captured_reduction.append(reduction)

        return original_mse_loss(
            input,
            target,
            reduction=reduction,
        )

    monkeypatch.setattr(
        dqn_agent_module.F,
        "mse_loss",
        capture_mse_loss,
    )

    weights = torch.tensor(
        [1.0, 0.5],
        dtype=torch.float32,
    )

    loss = agent.train_step(
        batch,
        weights=weights,
    )

    assert isinstance(loss, float)
    assert captured_reduction == ["none"]

def test_train_step_rejects_wrong_number_of_weights():
    agent = DQNAgent()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    batch = [
        Transition(
            state=state,
            action=1,
            reward=1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
        Transition(
            state=state,
            action=2,
            reward=-1.0,
            next_state=state,
            done=True,
            next_legal_actions=[],
        ),
    ]

    weights = torch.tensor(
        [1.0],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="weights must match the batch size",
    ):
        agent.train_step(
            batch,
            weights=weights,
        )