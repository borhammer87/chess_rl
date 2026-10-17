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

The active objective is to preserve and evaluate the promising 100-episode spatial State-Action result. The diagnostic probe now supports command-line selection of architecture and episode count, and prints its effective experiment configuration. Chess-playing strength remains unproven.

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


## Spatial capture-learning diagnosis (October 2026)

The repository contains `SpatialStateActionDQN`, a parallel experimental
State-Action variant that explicitly supplies the destination-square board
channels to the Q-head. It also contains a paired capture-learning comparison
and a five-seed runner in `scripts/run_capture_learning_probe.py`.

The controlled task compares terminal one-step knight captures of a rook
(reward `0.05`) and pawn (`0.01`). Each seed uses 300 training positions,
100 held-out evaluation positions and 50 full-set updates per model.
Across seeds 42–46, the original State-Action model reached **50.2% ± 2.8
percentage points** final evaluation accuracy; the spatial variant reached
**99.4% ± 0.9 percentage points**, winning in all five paired runs.
The mean held-out correct-minus-incorrect Q gap was approximately `-0.00001`
(original) versus `+0.03964` (spatial); the reward difference is `0.04`.

This supports the practical value of explicit action-conditioned spatial
features for this narrow task, but does not establish better full-game chess
play or isolate feature access from the slightly larger Q-head.

The spatial model's legal-next-action maxima, matching policy/target networks
and non-terminal training have been implemented and tested. A 100-episode
full-game probe now provides an initial comparison; see `TRAINING_STATUS.md`.

**Immediate next task:** Completed workflow improvement: scripts/run_state_action_probe.py supports --architecture (spatial or standard) and --episodes (1–100000), prints the effective experiment configuration, and includes CLI and configuration-output tests.

Immediate next task: design a controlled, multi-seed spatial-versus-standard experiment that keeps truncation bootstrapping and other training settings identical. Use the existing RandomAgent benchmark and distinguish measured results from hypotheses before changing the training architecture.

## Development constraints

Before any new repository-specific modification:

1. inspect the current repository ZIP;
2. read all project Markdown files;
3. inspect the relevant source and tests;
4. follow `PROJECT_RULES.md`;
5. prefer the smallest correct change;
6. update tests together with production behavior when appropriate;
7. do not infer repository state from previous conversations.

Do not infer general playing strength from a single seed or replace the main
self-play model without further controlled evaluation.

## Spatial full-game probe — October 2026

With `spatial=True`, 100 episodes, 150 maximum learner steps, batch size 32,
minimum replay size 1,000, target synchronization every 10 episodes,
`bootstrap_truncation=True`, and fixed probe seeds, training completed in
340.30 seconds (CPU). Training outcomes: 11 wins, 29 draws, 4 losses,
56 truncations; final epsilon 0.6274; replay size 10,000.

Balanced greedy evaluation against `RandomAgent` (20 games per color) changed
from 0 wins / 1 draw / 11 losses / 28 truncations (score 0.0125) to
8 wins / 27 draws / 1 loss / 4 truncations (score 0.5375).
This is encouraging **within this run**, not evidence of reproducibility or
strong chess play. Repetitive move sequences remain in the diagnostic game.
The original non-spatial 100-episode baseline ended at score 0.1125, but the
spatial run also enabled truncation bootstrapping, so it is **not** a strict
one-variable architecture A/B comparison.
