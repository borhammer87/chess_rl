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


The active objective is to diagnose why the experimental spatial
State-Action DQN fails to convert available checkmates during
full-game greedy evaluation.

A fresh 50-episode CUDA training run initialized replay with
295 verified mate transitions from 199 unique positions,
without mate-only pretraining. Final greedy evaluation scored
0.375 over 40 games, but a separate 50-game audit converted
only 2 of 121 available mates.

The checkpoint subsequently selected mate in 29/199 seeded
positions and 2/89 unique game positions. Mean mate Q
advantage was negative in both datasets.

The current task is to evaluate competing explanations
involving mate replay sampling and reward incentives,
without changing multiple training mechanisms at once.


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
- reproducible State-Action CPU/CUDA diagnostic probe.

See `ARCHITECTURE.md` for implementation details.

---

## State-Action DQN

The experimental State-Action architecture evaluates legal actions directly
instead of producing one Q-value for every action in the full external action
space.

It is currently exercised through:

`scripts/run_state_action_probe.py`

The probe provides a reproducible CPU/CUDA test bed for training and diagnostic
experiments. Device selection does not change the original DQNCNN `main()` workflow.

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

**Completed workflow improvement:** `scripts/run_state_action_probe.py` supports `--architecture`, `--episodes` and `--device` (CPU/CUDA), prints the experiment configuration and profiles training time. A 10-episode CUDA run completed 350 neural updates without errors; the CPU comparison completed 377 updates. CUDA took 4.13 s total versus 9.67 s on CPU, an indicative 2.34× total-time speedup, not a strict equal-work benchmark. See `TRAINING_STATUS.md`.

**Completed checkpoint workflow improvement:** 

**Completed checkpoint resume workflow (October 2026):** `scripts/run_state_action_probe.py` supports explicit `--resume`. Checkpoints are saved every 25 cumulative episodes under `checkpoints/state_action_probe/<run_id>/`. Resume restores the agent, optimizer, epsilon, replay buffer, terminal-transition labels and RNG state. The probe validates architecture and training configuration before resuming. The initial evaluation does not consume the restored training RNG sequence.

Integration tests cover checkpoint restoration, cumulative episode numbering, reuse of the original checkpoint directory, configuration validation and single-load RNG restoration.

A real CUDA resume test completed successfully: 25 initial episodes followed by 25 additional episodes, producing `episode_50.pt` in the original run directory. The resumed segment took 26.50 seconds, with final epsilon 0.8061 and replay size 6,602. Final balanced greedy evaluation scored 0.525 over 40 games (6 wins, 30 draws, 0 losses, 4 truncations). Mate-in-one accuracy remained 0/4.

**Current limitation:** practical checkpoint continuation is validated, but exact numerical equivalence between uninterrupted and resumed training has not been demonstrated.


**Mate-in-one learning milestone (October 2026):**

The spatial State-Action DQN has been trained on generated KQK
mate-in-one positions using agent-selected actions and terminal
rewards (+1 for checkmate, 0 otherwise).

With 1,000 training positions, 200 held-out positions and seed
12345, a 50-round CUDA experiment achieved:

- Training greedy accuracy: 90.4%.
- Held-out greedy accuracy: 83.5%.
- Held-out mean mate Q advantage: +0.2951.
- Total attempts: 50,000.
- Final epsilon: 0.1.

The earlier four-position experiment fitted its training examples
but did not solve its held-out position.

The larger experiment provides evidence of learning within KQK,
not general chess strength. Multi-seed validation and evaluation
on other position types remain pending.

The current mate probe does not persist a reusable trained
checkpoint.



**Immediate next development task:** validate mate-in-one learning
across independent seeds and determine whether the learned policy
generalizes beyond KQK. Then assess epsilon-decay sensitivity
and persistence of the trained mate policy before considering
integration with full-game training.


**Next learning-validation milestone:** run controlled multi-seed spatial-versus-standard comparisons with matching training settings and the existing RandomAgent benchmark.

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


## Current milestone — November 2026

Completed:

- Added mate-in-one CSV extraction and replay initialization
  to the experimental full-game training workflow.
- Completed a fresh 50-episode CUDA run with mate replay seeding.
- Evaluated the checkpoint in 40 balanced greedy games.
- Audited mate opportunities across 50 additional greedy games.
- Measured mate-selection accuracy and Q advantage on seeded
  and encountered positions.

Current evidence:

- Seeded positions: 29/199 mate selections (14.57%).
- Unique game positions: 2/89 mate selections (2.25%).
- Independent 50-game audit: 2/121 mates converted.
- Repetitive play remains observable.

The mate-generalization diagnostic was executed locally,
but its script is not present in the repository ZIP
inspected for this documentation update.

Immediate next task:

Choose the next controlled experiment after reviewing
mate-transition sampling and the existing reward structure.

Candidate changes include a fixed mate sampling proportion
or a stronger long-game/truncation penalty. No choice has
yet been approved.

See `DIAGNOSTICS.md` for interpretation and
`TRAINING_STATUS.md` for measured results.
