
import torch

from chess_rl.models.state_action_dqn import StateActionDQN
from chess_rl.training.capture_learning_task import (
    evaluate_capture_accuracy,
    split_capture_positions,
)


def run_initial_capture_probe(seed: int = 42) -> float:
    """Evaluate an untrained model on held-out capture positions."""
    torch.manual_seed(seed)

    train, evaluation = split_capture_positions(
        train_count=100,
        evaluation_count=40,
        seed=seed,
    )

    model = StateActionDQN()
    accuracy = evaluate_capture_accuracy(model, evaluation)

    print("State-Action capture learning probe")
    print(f"Training positions reserved: {len(train)}")
    print(f"Evaluation positions: {len(evaluation)}")
    print(f"Untrained accuracy: {accuracy:.1%}")

    return accuracy


if __name__ == "__main__":
    run_initial_capture_probe()
