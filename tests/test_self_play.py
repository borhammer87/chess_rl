import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN
from chess_rl.training.self_play import create_frozen_opponent


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