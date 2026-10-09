# CURRENT STATE

## Purpose

This document is the concise snapshot of the repository's current state.

It should answer:

- what is implemented now;
- what training path is currently active;
- what has been validated;
- what problem is currently being investigated;
- what the immediate next task is.

Detailed architecture belongs in `ARCHITECTURE.md`.

Long-lived design decisions belong in `DECISIONS.md`.

Diagnostic hypotheses and experimental evidence belong in `DIAGNOSTICS.md`.

Planned future work belongs in `ROADMAP.md`.

---

## Project status

The project contains a complete DQN chess training pipeline and an experimental
State-Action DQN path.

The current development focus is not adding basic DQN functionality.

The active objective is to understand why the State-Action DQN trains
successfully but has not yet demonstrated reliable greedy chess play before
scaling training or integrating it into the main self-play workflow.

---

## Implemented core pipeline

The repository currently includes:

- chess environment based on `python-chess`;
- board encoding;
- fixed external action encoding;
- legal-action masking;
- replay buffer;
- Prioritized Experience Replay;
- original CNN DQN;
- experimental State-Action DQN;
- DQN / State-Action DQN agents;
- RandomAgent;
- multi-episode training;
- replay sampling and optimization;
- target-network synchronization;
- epsilon decay after successful training updates;
- episode metrics;
- training summaries;
- greedy evaluation against RandomAgent;
- checkpoint infrastructure;
- self-play infrastructure for the original DQN path;
- reproducible State-Action CPU diagnostic probe.

See `ARCHITECTURE.md` for implementation details.

---

## State-Action DQN

The experimental State-Action architecture evaluates legal actions directly
instead of producing one Q-value for every action in the full external action
space.

It is currently exercised through:

`scripts/run_state_action_probe.py`

The probe provides a reproducible CPU test bed for training and diagnostic
experiments.

The State-Action path has not yet replaced the original DQN in the main
training/self-play workflow.

---

## Current reproducible baseline

The controlled State-Action probe uses:

- 100 training episodes;
- maximum 150 learner decisions per episode;
- batch size 32;
- minimum replay size 1000;
- target-network update frequency 10;
- baseline `gamma = 0.99`;
- baseline truncation penalty `-0.1`;
- fixed initialization, training and evaluation seeds.

The reproducible 100-episode baseline produced:

### Training

- wins: 7;
- draws: 11;
- losses: 4;
- truncated: 78;
- average plies: 278.11;
- average reward: -0.1064;
- average loss: approximately 0.00035667;
- final epsilon: approximately 0.6274;
- replay size: 10,000.

### Final greedy evaluation

Across 40 balanced games against `RandomAgent`:

- wins: 2;
- draws: 5;
- losses: 4;
- truncated: 29;
- score: 0.113.

The same score was also observed after the earlier 500-episode State-Action
probe.

Therefore longer training has not yet established improved greedy performance.

---

## Current problem

The State-Action DQN:

- trains end to end;
- updates its networks;
- produces non-zero TD errors;
- produces non-trivial legal-action Q-value distributions;
- maintains a highly diverse replay at the exact encoded-state level;

but greedy evaluation remains weak and artificial truncation remains frequent.

The project is therefore in a diagnostic phase.

Several simple explanations have already been weakened or rejected by
controlled measurements.

See `DIAGNOSTICS.md` for the hypothesis tree, evidence and experimental history.

---


## Current diagnostic boundary

The reproducible 100-episode probe now distinguishes real chess endings from
artificial truncations in the final replay without changing `Transition` or
training semantics. Among 10,000 retained transitions, 73 had `done=True`:

- 54 artificial truncations: PER probability share 1.5488%, oversampling 2.868x;
- 6 wins: 0.1615%, 2.691x;
- 3 losses: 0.0928%, 3.093x;
- 10 draws: 0.1152%, 1.152x.

Together the categories account for approximately 1.9182% of PER sampling
probability (rounding differences only). PER relatively favors wins and losses,
but their absolute representation remains very small. These are probabilities
at the end of training, not counts of actual training samples.



## Immediate next task

Determine whether the existing training/replay interfaces allow the diagnostic
probe to count **actual PER draws by outcome category during optimization**,
without altering the sampling distribution, random-number sequence or network
updates.

Inspect source and tests before proposing instrumentation. Add a
controlled unequal-priority unit test for the existing category-probability
calculation.

Do not change hyperparameters or scale training yet.


## Development constraints

Before any new repository-specific modification:

1. inspect the current repository ZIP;
2. read all project Markdown files;
3. inspect the relevant source and tests;
4. follow `PROJECT_RULES.md`;
5. prefer the smallest correct change;
6. update tests together with production behavior when appropriate;
7. do not infer repository state from previous conversations.

Do not scale training, add CUDA-specific work, or integrate State-Action into
the main self-play workflow while the current learning-behavior investigation
remains unresolved.