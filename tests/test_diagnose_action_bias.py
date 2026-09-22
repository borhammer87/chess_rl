import torch

from chess_rl.training.diagnose_action_bias import (
    calculate_global_action_bias,
    get_bias_rank,
)
from chess_rl.utils.replay_buffer import Transition


class FixedQNetwork(torch.nn.Module):

    def forward(self, states):
        return torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ]
        )


class FakeAgent:

    def __init__(self):
        self.policy_net = FixedQNetwork()


def test_calculate_global_action_bias():
    agent = FakeAgent()

    state = torch.zeros((18, 8, 8))

    transitions = [
        Transition(
            state=state,
            action=0,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[0],
        ),
        Transition(
            state=state,
            action=0,
            reward=0.0,
            next_state=state,
            done=False,
            next_legal_actions=[0],
        ),
    ]

    bias = calculate_global_action_bias(
        agent,
        transitions,
    )

    assert torch.allclose(
        bias,
        torch.tensor(
            [2.0, 3.0, 4.0]
        ),
    )


def test_get_bias_rank():
    legal_global_bias = torch.tensor(
        [0.2, 0.8, 0.5, 0.1]
    )

    assert get_bias_rank(
        legal_global_bias,
        greedy_local_index=1,
    ) == 1

    assert get_bias_rank(
        legal_global_bias,
        greedy_local_index=2,
    ) == 2

    assert get_bias_rank(
        legal_global_bias,
        greedy_local_index=0,
    ) == 3