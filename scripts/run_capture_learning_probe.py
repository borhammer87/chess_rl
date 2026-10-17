
import torch

from chess_rl.agents.state_action_dqn_agent import StateActionDQNAgent
from chess_rl.training.capture_learning_task import (
    evaluate_capture_accuracy,
    split_capture_positions,
    train_capture_batch,
    capture_q_gap_summary,
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
        train_count=300,
        evaluation_count=100,
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

    train_gap, train_error_rate = capture_q_gap_summary(
    agent.policy_net, train
    )
    eval_gap, eval_error_rate = capture_q_gap_summary(
        agent.policy_net, evaluation
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
    print(f"Training mean Q gap: {train_gap:.6f}")
    print(f"Evaluation mean Q gap: {eval_gap:.6f}")
    print(f"Training Q error rate: {train_error_rate:.1%}")
    print(f"Evaluation Q error rate: {eval_error_rate:.1%}")    
    return accuracy


def run_capture_feature_probe(seed: int = 42, updates: int = 50):
    """Train the feature-only Q-network on terminal capture rewards."""
    from chess_rl.models.capture_feature_q import CaptureFeatureQ
    from chess_rl.training.capture_learning_task import (
        captured_piece_features,
        best_material_capture,
    )

    if updates < 1:
        raise ValueError("Updates must be positive.")

    torch.manual_seed(seed)
    train, evaluation = split_capture_positions(300, 100, seed)

    def examples(positions):
        features, rewards = [], []
        for board, first, second in positions:
            for move in (first, second):
                features.append(captured_piece_features(board, move))
                rewards.append(
                    0.05 if board.piece_at(move.to_square).piece_type == 4
                    else 0.01
                )
        return torch.stack(features), torch.tensor(rewards)

    def accuracy(positions):
        correct = 0
        with torch.no_grad():
            for board, first, second in positions:
                features = torch.stack([
                    captured_piece_features(board, first),
                    captured_piece_features(board, second),
                ])
                scores = model(features)
                chosen = first if scores[0] > scores[1] else second
                correct += chosen == best_material_capture(
                    board, first, second
                )
        return correct / len(positions)

    model = CaptureFeatureQ()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features, targets = examples(train)

    initial_accuracy = accuracy(evaluation)
    for _ in range(updates):
        loss = torch.nn.functional.mse_loss(model(features), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_accuracy = accuracy(evaluation)
    print("Capture-feature Q probe")
    print(f"Initial evaluation accuracy: {initial_accuracy:.1%}")
    print(f"Final evaluation accuracy: {final_accuracy:.1%}")
    print(f"Final training loss: {loss.item():.6f}")
    return final_accuracy


def run_spatial_comparison(
    seed: int = 42,
    updates: int = 50,
    train_count: int = 300,
    evaluation_count: int = 100,
) -> dict:
    """Compare original and spatial models on identical capture data."""
    from chess_rl.models.spatial_state_action_dqn import (
        SpatialStateActionDQN,
    )

    if updates < 1:
        raise ValueError("Updates must be positive.")

    train, evaluation = split_capture_positions(
        train_count, evaluation_count, seed
    )
    results = {}

    for name, model_type in (
        ("original", None),
        ("spatial", SpatialStateActionDQN),
    ):
        torch.manual_seed(seed)
        agent = StateActionDQNAgent()

        if model_type is not None:
            agent.policy_net = model_type()
            agent.optimizer = torch.optim.Adam(
                agent.policy_net.parameters(), lr=1e-3
            )

        model = agent.policy_net
        initial_train = evaluate_capture_accuracy(model, train)
        initial_eval = evaluate_capture_accuracy(model, evaluation)

        losses = [
            train_capture_batch(agent, train)
            for _ in range(updates)
        ]

        final_train = evaluate_capture_accuracy(model, train)
        final_eval = evaluate_capture_accuracy(model, evaluation)
        train_gap, _ = capture_q_gap_summary(model, train)
        eval_gap, _ = capture_q_gap_summary(model, evaluation)

        results[name] = {
            "initial_train": initial_train,
            "initial_eval": initial_eval,
            "final_train": final_train,
            "final_eval": final_eval,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "train_q_gap": train_gap,
            "eval_q_gap": eval_gap,
        }

        print(f"\n{name.upper()} ({updates} updates)")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.6f}")

    return results


if __name__ == "__main__":
    run_initial_capture_probe()
