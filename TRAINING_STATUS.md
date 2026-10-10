# TRAINING STATUS

## Purpose

This document preserves the concise experimental history of the project's
training pipelines.

It records significant completed training runs and their measured results.

It does not determine why an experiment behaved as it did.

For interpretation, active hypotheses and diagnostic conclusions, see
`DIAGNOSTICS.md`.

For the current repository snapshot and immediate next task, see
`CURRENT_STATE.md`.

---

## Original DQN pipeline

The original fixed-output DQN training path reached a complete working training
pipeline including:

- replay-based optimization;
- target-network synchronization;
- epsilon decay after successful updates;
- multi-episode training;
- episode metrics and training summaries;
- evaluation against `RandomAgent`;
- checkpoint support;
- self-play infrastructure.

Diagnostic work on this path showed that successful optimization alone did not
establish reliable greedy chess performance.

The original DQN remains part of the repository and provides the established
production/self-play path.

---

## State-Action DQN introduction

The State-Action DQN was introduced as an experimental parallel architecture.

Instead of predicting Q-values for the complete fixed action space, it evaluates
state/action pairs and scores legal actions directly.

The architecture is currently exercised through:

`scripts/run_state_action_probe.py`

The probe is intentionally separate from the main self-play workflow while the
learning behavior is being validated.

---

## Reproducible State-Action probe configuration

The current controlled CPU probe uses:

- 100 training episodes;
- 40 initial greedy evaluation games;
- 40 final greedy evaluation games;
- maximum 150 learner decisions per episode;
- batch size 32;
- minimum replay size 1000;
- target-network update frequency 10;
- fixed initialization, training and evaluation seeds.

Unless an experiment explicitly states otherwise, the baseline includes:

- `gamma = 0.99`;
- truncation penalty `-0.1`.

---

## 100-episode baseline

### Initial greedy evaluation

Across 40 balanced games against `RandomAgent`:

- wins: 0;
- draws: 0;
- losses: 10;
- truncated: 30;
- score: 0.000.

Initial truncated-position material diagnostics:

- average total material: 20.2;
- average learner material balance: approximately -6.27;
- average absolute material balance: approximately 9.13.

### Training

Across 100 training episodes:

- wins: 7;
- draws: 11;
- losses: 4;
- truncated: 78;
- average plies: 278.11;
- average reward: -0.1064;
- average loss: approximately 0.00035667;
- final epsilon: approximately 0.6274;
- final replay size: 10,000.

### Final greedy evaluation

Across 40 balanced games against `RandomAgent`:

- wins: 2;
- draws: 5;
- losses: 4;
- truncated: 29;
- score: 0.113.

Final truncated-position material diagnostics:

- average total material: approximately 10.83;
- average learner material balance: approximately -2.62;
- average absolute material balance: approximately 6.28.

---

## 500-episode State-Action probe

A longer 500-episode CPU probe produced:

### Training

- wins: 46;
- draws: 70;
- losses: 25;
- truncated: 359;
- average plies: 274.44;
- average reward: -0.0895;
- average loss: approximately 0.00052025;
- final epsilon: 0.1000;
- replay size: 10,000.

### Final greedy evaluation

Across 40 balanced games against `RandomAgent`:

- wins: 3;
- draws: 3;
- losses: 2;
- truncated: 32;
- score: 0.113.

The longer run therefore did not produce a higher balanced RandomAgent score
than the reproducible 100-episode baseline.

Interpretation of this result belongs in `DIAGNOSTICS.md`.

---

## Controlled gamma experiment

A controlled 100-episode experiment changed:

`gamma: 0.99 -> 0.995`

while retaining the other baseline settings.

### Training

- wins: 7;
- draws: 12;
- losses: 7;
- truncated: 74;
- average plies: 275.54;
- average reward: -0.1412;
- average loss: approximately 0.00039254;
- final epsilon: approximately 0.6274.

### Final greedy evaluation

- wins: 3;
- draws: 2;
- losses: 4;
- truncated: 31;
- score: 0.100.

The baseline `gamma = 0.99` was restored after the experiment.

---

## Controlled truncation-penalty experiment

A controlled 100-episode experiment changed:

`TRUNCATION_PENALTY: -0.1 -> -0.2`

while retaining the baseline `gamma = 0.99`.

### Training

- wins: 3;
- draws: 11;
- losses: 9;
- truncated: 77;
- average plies: 277.99;
- average reward: -0.2831;
- average loss: approximately 0.00042269;
- final epsilon: approximately 0.6274;
- replay size: 10,000.

### Final greedy evaluation

- wins: 4;
- draws: 3;
- losses: 3;
- truncated: 30;
- score: 0.138.

The truncation penalty was restored to `-0.1` after the experiment.

Because the reward function itself changed, average reward from this experiment
must not be directly compared with baseline average reward as a measure of
policy quality.

---

## Baseline replay learning-signal diagnostics

The reproducible 10,000-transition baseline replay contained:

- non-terminal transitions: 9,927;
- terminal (`done=True`) transitions: 73.

Whole replay:

- reward mean: approximately -0.000736;
- reward standard deviation: approximately 0.03681;
- Bellman-target mean: approximately 0.35914;
- Bellman-target standard deviation: approximately 0.08710;
- selected-Q mean: approximately 0.34914;
- selected-Q standard deviation: approximately 0.07730;
- mean absolute TD error: approximately 0.02034.

Non-terminal transitions:

- reward mean: approximately -0.000492;
- Bellman-target mean: approximately 0.36203;
- selected-Q mean: approximately 0.35172;
- mean absolute TD error: approximately 0.02005.

Terminal transitions:

- reward / target mean: approximately -0.03397;
- selected-Q mean: approximately -0.00274;
- mean absolute TD error: approximately 0.05903.

---

## Baseline PER diagnostics

For the same reproducible 10,000-transition replay:

- terminal (`done=True`) replay share: 0.007300;
- terminal PER probability share: 0.019182;
- aggregate terminal oversampling factor: approximately 2.628x.

Priority statistics:

- terminal mean priority: 0.049633;
- terminal median priority: 0.017887;
- non-terminal mean priority: 0.007798;
- non-terminal median priority: 0.004111.


These aggregate measurements combine real chess endings and artificial
truncations. The category-resolved measurement below uses external labels in
the diagnostic probe; `Transition` itself remains unchanged.


---


## Baseline PER categories (100-episode diagnostic rerun)

The reproducible rerun took 132.86 seconds and retained 10,000 transitions,
including 73 with `done=True`. Episode outcomes and final greedy evaluation
matched the baseline: training 7/11/4/78 (win/draw/loss/truncated), final
greedy 2/5/4/29, score 0.113.

| Category | Count | Replay share | Mean priority | Median priority | PER share | Factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Truncated | 54 | 0.005400 | 0.059052 | 0.018387 | 0.015488 | 2.868x |
| Win | 6 | 0.000600 | 0.031964 | 0.032367 | 0.001615 | 2.691x |
| Loss | 3 | 0.000300 | 0.042702 | 0.034406 | 0.000928 | 3.093x |
| Draw | 10 | 0.001000 | 0.011450 | 0.002838 | 0.001152 | 1.152x |

These are probabilities for the final replay, not observed counts of sampled
training transitions. Category shares sum to the aggregate PER share within
rounding tolerance.

---
## Observed PER sampling during training

A reproducible 100-episode baseline rerun recorded actual PER
selections during optimization, including repeated selections
and transitions subsequently evicted from replay.

- Total selections: 412,416.
- Non-terminal: 406,265 (98.5085%).
- Artificial truncation: 4,686 (1.1362%).
- Win: 482 (0.1169%).
- Loss: 443 (0.1074%).
- Draw: 540 (0.1309%).

Wins and losses together accounted for 925 selections,
approximately 0.2243% of all selections.

The run retained the baseline training outcomes:
7 wins, 11 draws, 4 losses and 78 truncations.

Final greedy evaluation remained:
2 wins, 5 draws, 4 losses and 29 truncations,
with balanced score 0.113.

The run took 189.70 seconds.

These are accumulated selection counts, not unique transitions,
distinct optimization batches or importance-weighted gradient
contributions.

## Baseline replay diversity

The final 10,000-transition training replay contained:

- 1,685 unique action IDs;
- 9,910 unique encoded states;
- 9,977 unique encoded-state/action pairs;
- 90 repeated encoded states;
- 23 repeated encoded-state/action pairs.

Action concentration:

- top 10 action IDs: 0.036;
- top 25: 0.077;
- top 50: 0.135.

---

## Final-policy diagnostic game

The reproducible final-policy diagnostic trajectory reached the artificial
horizon after:

- 300 total plies;
- 150 learner transitions.

It contained:

- 149 non-terminal transitions;
- 1 terminal artificial-truncation transition;
- 105 unique action IDs;
- 149 unique encoded states;
- 149 unique encoded-state/action pairs.

For the final artificial-truncation transition:

- reward: -0.100000;
- Bellman target: -0.100000;
- selected Q-value: 0.284372;
- absolute TD error: 0.384372.

The reconstructed greedy decisions matched all 150 recorded learner moves.

The trajectory contained many small top-two legal-action Q gaps, while some
positions showed substantially stronger preferences.

Two queen-promotion decisions produced top-two gaps of approximately:

- 0.06317;
- 0.03497.

---

## Current experimental status

The baseline configuration has been restored after the controlled gamma and
truncation-penalty experiments.

No current result establishes reliable greedy State-Action DQN chess play.

The project remains in diagnostic validation before larger-scale training or
State-Action self-play integration.

For the active hypothesis tree and next diagnostic question, see
`DIAGNOSTICS.md`.