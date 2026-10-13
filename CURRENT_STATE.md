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

The PER investigation measured final-replay sampling probabilities
and actual training selections, including historical transitions
subsequently evicted from replay.

A controlled 100-episode experiment compared the baseline
artificial-truncation semantics (`done=True`) with optional
bootstrapping (`bootstrap_truncation=True`).

With bootstrapping enabled:

- training outcomes: 8 wins, 13 draws, 3 losses, 76 truncated;
- final greedy evaluation: 3 wins, 3 draws, 4 losses, 30 truncated;
- final balanced greedy score: 0.113;
- final replay: 10,000 transitions, including 21 Bellman-terminal
  transitions and 54 separately labelled artificial truncations.

The baseline final greedy score was also 0.113, with 29 truncated
evaluation games. The experiment therefore did not demonstrate
improved greedy performance.

Artificial truncations can now retain legal next actions and
bootstrap future values. The default remains
`bootstrap_truncation=False`.

The final greedy diagnostic-game recording now accepts the same
option, independently of greedy evaluation scoring. Its corrected
TD measurements have not yet been collected.


## Current capture-learning diagnostic (October 2026)

A controlled one-step capture probe is implemented in
`scripts/run_capture_learning_probe.py`, with deterministic training/evaluation
splits, terminal capture transitions, repeated updates through the existing
State-Action agent, and accuracy/Q-gap diagnostics. The position generator
was expanded to produce 400 distinct valid positions.

After 50 updates, the 100/40 probe achieved 76% training and 60% evaluation
accuracy; the expanded 300/100 probe achieved 64.3% training and 50%
evaluation accuracy. Loss fell sharply in both runs. These experiments are
not directly controlled comparisons of dataset size because the generator,
evaluation sets and experiences processed per update changed. Full results
are recorded in `TRAINING_STATUS.md`; competing explanations are in
`DIAGNOSTICS.md`.

**Immediate next task:** audit the capture transitions, encoded actions,
rewards, terminal Bellman targets and predicted Q-values using the current
State-Action training path. Do not change the architecture or scale training
until that audit clarifies the failure mode.

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