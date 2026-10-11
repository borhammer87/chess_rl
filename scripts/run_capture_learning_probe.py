
import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.capture_learning_task import (
    evaluate_capture_accuracy,
    split_capture_positions,
    train_capture_batch,
)


def run_initial_capture_probe(seed: int = 42) -> float:
    """Evaluate an untrained model on held-out capture positions."""
    torch.manual_seed(seed)

    train, evaluation = split_capture_positions(
        train_count=100,
        evaluation_count=40,
        seed=seed,
    )

    agent = StateActionDQNAgent()
    accuracy = evaluate_capture_accuracy(agent.policy_net, evaluation)
    loss = train_capture_batch(agent, train)
    final_accuracy = evaluate_capture_accuracy(agent.policy_net, evaluation)

    print("State-Action capture learning probe")
    print(f"Training positions reserved: {len(train)}")
    print(f"Evaluation positions: {len(evaluation)}")
    print(f"Untrained accuracy: {accuracy:.1%}")
    print(f"Training loss: {loss:.6f}")
    print(f"Accuracy after one update: {final_accuracy:.1%}")

    return accuracy


if __name__ == "__main__":
    run_initial_capture_probe()
