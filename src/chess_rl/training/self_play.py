from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.models.dqn_cnn import DQNCNN


def create_frozen_opponent(
    agent: DQNAgent,
) -> DQNCNN:
    """
    Create an independent frozen copy of the agent's current policy.

    The returned network can be used as a stable self-play opponent.
    It does not share parameters with the learning policy and its
    parameters do not require gradients.
    """
    opponent = DQNCNN()

    opponent.load_state_dict(
        agent.policy_net.state_dict()
    )

    opponent.eval()

    for parameter in opponent.parameters():
        parameter.requires_grad_(False)

    return opponent