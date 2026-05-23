import torch
from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.utils.replay_buffer import Transition


# -----------------------------
# 1. ACTION RANGE TEST
# -----------------------------
def test_agent_select_action_shape():
    agent = DQNAgent()

    state = torch.zeros(12, 8, 8)

    action = agent.select_action(state)

    assert 0 <= action < 4096


# -----------------------------
# 2. GREEDY MODE TEST
# -----------------------------
def test_agent_greedy_action_runs():
    agent = DQNAgent(epsilon=0.0)

    state = torch.zeros(12, 8, 8)

    action = agent.select_action(state)

    assert isinstance(action, int)
    assert 0 <= action < 4096


# -----------------------------
# 3. TRAINING STEP TEST
# -----------------------------
def test_agent_train_step_runs():

    agent = DQNAgent()

    state = torch.zeros(12, 8, 8)

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


# -----------------------------
# 4. TARGET NETWORK SYNC TEST
# -----------------------------
def test_target_sync_changes_weights():

    agent = DQNAgent()

    old_weights = agent.target_net.features[0].weight.clone()

    agent.policy_net.features[0].weight.data += 1.0

    agent.update_target()

    new_weights = agent.target_net.features[0].weight

    assert not torch.allclose(old_weights, new_weights)