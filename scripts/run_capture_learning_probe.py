
import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.capture_learning_task import (
    evaluate_capture_accuracy,
    split_capture_positions,
    train_capture_batch,
)



def run_initial_capture_probe(
    seed: int = 42,
    updates: int = 50,
) -> float:
    """Train on fixed capture examples and evaluate held-out accuracy."""
    if updates < 1:
        raise ValueError("Updates must be positive.")

    torch.manual_seed(seed)

    train, evaluation = split_capture_positions(
        train_count=100,
        evaluation_count=40,
        seed=seed,
    )

    agent = StateActionDQNAgent()
    initial_train_accuracy = evaluate_capture_accuracy(
    agent.policy_net, train
    )
    accuracy = evaluate_capture_accuracy(agent.policy_net, evaluation)

    losses = [
        train_capture_batch(agent, train)
        for _ in range(updates)
    ]

    final_accuracy = evaluate_capture_accuracy(
        agent.policy_net, evaluation
    )
    final_train_accuracy = evaluate_capture_accuracy(
    agent.policy_net, train
    )
    
    print("State-Action capture learning probe")
    print(f"Training positions reserved: {len(train)}")
    print(f"Evaluation positions: {len(evaluation)}")
    print(f"Untrained accuracy: {accuracy:.1%}")
    print(f"Training updates: {updates}")
    print(f"First training loss: {losses[0]:.6f}")
    print(f"Final training loss: {losses[-1]:.6f}")
    print(f"Final accuracy: {final_accuracy:.1%}")
    print(f"Initial training accuracy: {initial_train_accuracy:.1%}")
    print(f"Final training accuracy: {final_train_accuracy:.1%}")
    return accuracy


if __name__ == "__main__":
    run_initial_capture_probe()
