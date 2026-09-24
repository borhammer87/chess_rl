# NEXT CHAT PROMPT

We are continuing the Chess Reinforcement Learning project.

IMPORTANT:

The attached ZIP repository is the only source of truth.

Do not rely on previous conversations.

Before proposing any modification:

1. Extract the ZIP repository.
2. Inspect the complete repository structure.
3. Read all project Markdown files.
4. Read every source file relevant to the requested change.
5. Determine the current project state from the repository itself.

If the ZIP cannot be inspected, state that limitation instead of making
assumptions.

## Repository version

The current project version is 0.9.0.

## Current implementation state

The original DQN implementation remains available.

It uses:

- an 18 × 8 × 8 board representation;
- `DQNCNN`;
- a fixed 4272-action output vector;
- legal-action masking;
- prioritized replay;
- target-network synchronization;
- reward shaping;
- checkpointing;
- RandomAgent evaluation;
- frozen-opponent self-play.

A second State-Action DQN implementation now exists in parallel.

It includes:

- `StateActionDQN`;
- a CNN state encoder;
- structured action decoding into from square, to square, and promotion;
- action embeddings;
- a shared Q(s, a) head;
- vectorized evaluation of multiple actions for one state;
- batched evaluation of state-action pairs;
- batched maximum legal-action evaluation for next states;
- `StateActionDQNAgent`;
- epsilon-greedy action selection;
- policy and target networks;
- Bellman training updates;
- terminal future-value handling;
- PER importance-sampling weights;
- TD-error reporting;
- target-network synchronization;
- state serialization;
- compatibility with the existing checkpoint infrastructure.

The State-Action agent has passed an end-to-end CPU smoke test through the
existing RandomAgent training workflow.

This smoke test establishes pipeline compatibility only. It does not
demonstrate that the State-Action model learns a stronger policy.

## Important current limitations

- Explicit CUDA/device management has not yet been implemented.
- The State-Action agent is not yet integrated into frozen-opponent
  self-play.
- No long State-Action training experiment has yet been completed.
- No comparative playing-strength result between StateActionDQN and
  DQNCNN has yet been established.

## Current decision point

Determine the next smallest development step from the repository.

One open question is whether explicit device/CUDA support should be added
before larger State-Action training experiments.

Do not assume that CUDA must be the next step. Inspect the repository and
justify the recommendation.

The target machines for future training must remain practical consumer
hardware.

Follow PROJECT_RULES.md.

Prefer the smallest correct modification.

Update tests together with code.

Do not remove the original DQN implementation merely because the
State-Action implementation now exists.