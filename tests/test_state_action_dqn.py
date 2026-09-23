import torch
from chess_rl.utils.board_encoder import BOARD_CHANNELS
from chess_rl.models.state_action_dqn import (
    ACTION_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    StateActionDQN,
)
import chess
import pytest

from chess_rl.utils.action_encoder import (
    encode_move,
)

def test_state_encoder_accepts_encoded_board_batch():
    model = StateActionDQN()

    states = torch.zeros(
        (2, BOARD_CHANNELS, 8, 8)
    )

    state_features = model.encode_state(states)

    assert state_features.shape == (
        2,
        STATE_FEATURE_SIZE,
    )


def test_state_encoder_returns_different_features_for_different_states():
    model = StateActionDQN()

    states = torch.zeros(
        (2, BOARD_CHANNELS, 8, 8)
    )

    states[1, 0, 0, 0] = 1.0

    state_features = model.encode_state(states)

    assert not torch.allclose(
        state_features[0],
        state_features[1],
    )

def test_action_encoder_returns_one_feature_vector_per_action():
    model = StateActionDQN()

    from_squares = torch.tensor(
        [12, 6, 52]
    )

    to_squares = torch.tensor(
        [28, 21, 60]
    )

    promotion_types = torch.tensor(
        [0, 0, 1]
    )

    action_features = model.encode_actions(
        from_squares,
        to_squares,
        promotion_types,
    )

    assert action_features.shape == (
        3,
        ACTION_FEATURE_SIZE,
    )

def test_action_encoder_distinguishes_different_promotions():
    model = StateActionDQN()

    from_squares = torch.tensor(
        [52, 52]
    )

    to_squares = torch.tensor(
        [60, 60]
    )

    promotion_types = torch.tensor(
        [1, 4]
    )

    action_features = model.encode_actions(
        from_squares,
        to_squares,
        promotion_types,
    )

    assert not torch.allclose(
        action_features[0],
        action_features[1],
    )

def test_evaluate_actions_returns_one_q_value_per_action():
    model = StateActionDQN()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    from_squares = torch.tensor(
        [12, 6, 1, 11]
    )

    to_squares = torch.tensor(
        [28, 21, 18, 27]
    )

    promotion_types = torch.tensor(
        [0, 0, 0, 0]
    )

    q_values = model.evaluate_actions(
        state,
        from_squares,
        to_squares,
        promotion_types,
    )

    assert q_values.shape == (4,)

def test_evaluate_actions_encodes_state_only_once(
    monkeypatch,
):
    model = StateActionDQN()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action_count = 40

    from_squares = torch.zeros(
        action_count,
        dtype=torch.long,
    )

    to_squares = torch.ones(
        action_count,
        dtype=torch.long,
    )

    promotion_types = torch.zeros(
        action_count,
        dtype=torch.long,
    )

    original_encode_state = model.encode_state

    call_count = 0

    def counting_encode_state(states):
        nonlocal call_count

        call_count += 1

        return original_encode_state(states)

    monkeypatch.setattr(
        model,
        "encode_state",
        counting_encode_state,
    )

    q_values = model.evaluate_actions(
        state,
        from_squares,
        to_squares,
        promotion_types,
    )

    assert q_values.shape == (
        action_count,
    )

    assert call_count == 1

def test_evaluate_action_ids_returns_one_q_value_per_action():
    model = StateActionDQN()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action_ids = [
        encode_move(
            chess.Move.from_uci("e2e4")
        ),
        encode_move(
            chess.Move.from_uci("g1f3")
        ),
        encode_move(
            chess.Move.from_uci("d2d4")
        ),
    ]

    q_values = model.evaluate_action_ids(
        state,
        action_ids,
    )

    assert q_values.shape == (3,)

def test_evaluate_action_ids_encodes_state_only_once(
    monkeypatch,
):
    model = StateActionDQN()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    action_ids = [
        encode_move(
            chess.Move(
                from_square=index,
                to_square=(index + 1) % 64,
            )
        )
        for index in range(40)
    ]

    original_encode_state = model.encode_state
    call_count = 0

    def counting_encode_state(states):
        nonlocal call_count
        call_count += 1

        return original_encode_state(states)

    monkeypatch.setattr(
        model,
        "encode_state",
        counting_encode_state,
    )

    q_values = model.evaluate_action_ids(
        state,
        action_ids,
    )

    assert q_values.shape == (40,)
    assert call_count == 1

def test_evaluate_action_ids_rejects_empty_action_list():
    model = StateActionDQN()

    state = torch.zeros(
        (BOARD_CHANNELS, 8, 8)
    )

    with pytest.raises(
        ValueError,
        match="empty action list",
    ):
        model.evaluate_action_ids(
            state,
            [],
        )