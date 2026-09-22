import random
from pathlib import Path

import torch

from chess_rl.agents.dqn_agent import DQNAgent
from chess_rl.training.checkpoint import load_training_checkpoint
from chess_rl.utils.replay_buffer import ReplayBuffer


CHECKPOINT_PATH = Path("checkpoints/latest.pt")

BIAS_STATE_COUNT = 1000
TEST_STATE_COUNT = 500
REPLAY_CAPACITY = 10_000
SEED = 0


def calculate_global_action_bias(
    agent: DQNAgent,
    transitions,
) -> torch.Tensor:
    states = torch.stack(
        [
            transition.state
            for transition in transitions
        ]
    )

    with torch.no_grad():
        q_values = agent.policy_net(states)

    return q_values.mean(dim=0)


def get_bias_rank(
    legal_global_bias: torch.Tensor,
    greedy_local_index: int,
) -> int:
    bias_order = torch.argsort(
        legal_global_bias,
        descending=True,
    )

    return (
        (
            bias_order == greedy_local_index
        )
        .nonzero(as_tuple=False)[0]
        .item()
        + 1
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    agent = DQNAgent()
    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY
    )

    load_training_checkpoint(
        path=str(CHECKPOINT_PATH),
        agent=agent,
        replay_buffer=replay_buffer,
    )

    agent.policy_net.eval()

    transitions = list(
        replay_buffer.buffer
    )

    required = (
        BIAS_STATE_COUNT
        + TEST_STATE_COUNT
    )

    if len(transitions) < required:
        raise ValueError(
            f"Need at least {required} replay transitions."
        )

    sampled = random.sample(
        transitions,
        required,
    )

    bias_transitions = sampled[
        :BIAS_STATE_COUNT
    ]

    test_transitions = sampled[
        BIAS_STATE_COUNT:
    ]

    global_action_bias = (
        calculate_global_action_bias(
            agent,
            bias_transitions,
        )
    )

    agreements = 0
    total = 0

    greedy_ranks = []

    original_gaps = []
    residual_gaps = []

    for transition in test_transitions:
        if transition.done:
            continue

        legal_actions = (
            transition.next_legal_actions
        )

        if not legal_actions:
            continue

        state = transition.next_state.unsqueeze(
            0
        )

        with torch.no_grad():
            q_values = agent.policy_net(
                state
            )[0]

        legal_indices = torch.tensor(
            legal_actions,
            dtype=torch.long,
        )

        legal_q_values = q_values[
            legal_indices
        ]

        legal_global_bias = (
            global_action_bias[
                legal_indices
            ]
        )

        greedy_local_index = torch.argmax(
            legal_q_values
        ).item()

        bias_local_index = torch.argmax(
            legal_global_bias
        ).item()

        if (
            greedy_local_index
            == bias_local_index
        ):
            agreements += 1

        greedy_ranks.append(
            get_bias_rank(
                legal_global_bias,
                greedy_local_index,
            )
        )

        if len(legal_actions) >= 2:
            top_original = torch.topk(
                legal_q_values,
                2,
            ).values

            original_gaps.append(
                (
                    top_original[0]
                    - top_original[1]
                ).item()
            )

            residual_q_values = (
                legal_q_values
                - legal_global_bias
            )

            top_residual = torch.topk(
                residual_q_values,
                2,
            ).values

            residual_gaps.append(
                (
                    top_residual[0]
                    - top_residual[1]
                ).item()
            )

        total += 1

    if total == 0:
        raise ValueError(
            "No valid non-terminal test transitions."
        )

    ranks = torch.tensor(
        greedy_ranks,
        dtype=torch.float32,
    )

    print(
        "=== LEGAL ACTION-BIAS DIAGNOSTIC ==="
    )
    print()

    print(
        f"Bias states: {BIAS_STATE_COUNT}"
    )

    print(
        f"Legal test states analysed: {total}"
    )

    print()
    print(
        "--- GREEDY ACTION EXPLAINED "
        "BY GLOBAL BIAS ---"
    )

    print(
        "Exact agreement: "
        f"{agreements}/{total} "
        f"({agreements / total * 100:.2f}%)"
    )

    print(
        "Mean bias-rank of actual greedy action: "
        f"{ranks.mean().item():.3f}"
    )

    print(
        "Median bias-rank: "
        f"{ranks.median().item():.0f}"
    )

    for max_rank in (1, 2, 3, 5, 10):
        count = (
            ranks <= max_rank
        ).sum().item()

        print(
            f"Actual greedy action in bias "
            f"top-{max_rank}: "
            f"{count}/{total} "
            f"({count / total * 100:.2f}%)"
        )

    if original_gaps:
        original = torch.tensor(
            original_gaps
        )

        residual = torch.tensor(
            residual_gaps
        )

        print()
        print(
            "--- ORIGINAL vs "
            "STATE-RESIDUAL GAPS ---"
        )

        print(
            "Mean original top gap: "
            f"{original.mean().item():.6f}"
        )

        print(
            "Median original top gap: "
            f"{original.median().item():.6f}"
        )

        print(
            "Mean residual top gap: "
            f"{residual.mean().item():.6f}"
        )

        print(
            "Median residual top gap: "
            f"{residual.median().item():.6f}"
        )

    print()
    print(
        "=== END DIAGNOSTIC ==="
    )


if __name__ == "__main__":
    main()