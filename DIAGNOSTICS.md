# DIAGNOSTICS

## Purpose

This document tracks the active investigation into why the experimental
State-Action DQN has not yet demonstrated reliable greedy chess play.

Its purpose is not to describe the architecture or preserve every raw
experiment output. It records:

- observed problems;
- hypotheses that could explain them;
- evidence supporting or weakening each hypothesis;
- conclusions that are actually justified by the evidence;
- questions that remain open;
- the next diagnostic experiment and why it has priority.

A hypothesis must not be presented as established merely because it is
plausible.

Negative experimental results are useful when they eliminate or weaken a
possible explanation.

## Observed problem

The State-Action DQN trains successfully end to end and produces non-trivial
Q-value differentiation, but greedy evaluation against `RandomAgent` remains
weak and frequently reaches the artificial episode horizon.

The controlled 100-episode and 500-episode probes both produced a balanced
greedy RandomAgent score of `0.113`.

Increasing training from 100 to 500 episodes therefore did not establish
improved greedy performance.

High truncation also remained a persistent feature of evaluation.

The immediate objective is to understand this behavior before scaling training,
adding CUDA support or integrating State-Action into self-play.

## Established diagnostic evidence

### Claimable draws do not explain most truncations

Truncation diagnostics distinguish positions in which threefold repetition or
the fifty-move rule could be claimed.

Most truncated games do not have either claim available.

Therefore the high truncation rate cannot be explained primarily as ordinary
chess draws that `ChessEnv` happens not to claim automatically.

Status: strong version rejected.

### Truncated games are materially simplified

Material diagnostics show that truncated games frequently reach positions with
substantially less material than the starting position.

Therefore truncation is not simply the result of games remaining completely
undeveloped for 150 learner decisions.

This does not establish that the resulting play is good or strategically
coherent.

Status: descriptive evidence established; cause remains open.

### Global Q-value collapse is not supported

The exact saved truncated greedy game was reconstructed against the final
policy.

All 150 recorded learner moves matched the reconstructed greedy choices.

Complete legal-action Q distributions showed meaningful ranges in many
positions. The network can also form strong preferences in some situations:
queen promotions in the diagnostic trajectory produced top-two Q gaps of
approximately `0.063` and `0.035`.

However, many ordinary positions had extremely small top-two gaps, including
values near zero.

Therefore the strongest hypothesis that the network assigns effectively the
same Q-value to every legal action is not supported.

A narrower observation remains: the policy frequently has weak separation
between its highest-ranked ordinary actions.

Status: strong global-collapse hypothesis rejected; weak ordinary-action
differentiation remains observed.

### Bellman targets and TD errors have non-trivial variation

The final 10,000-transition training replay showed:

- non-terminal Bellman-target standard deviation: approximately `0.07397`;
- non-terminal selected-Q standard deviation: approximately `0.06323`;
- non-terminal mean absolute TD error: approximately `0.02005`.

The exact 150-transition diagnostic-game replay also showed non-zero target,
Q-value and TD-error variation.

Its final artificial truncation transition had:

- reward / Bellman target: `-0.100000`;
- predicted Q-value: `0.284372`;
- absolute TD error: `0.384372`.

Therefore vanishing TD error or globally constant Bellman targets are not
supported as sufficient explanations.

Status: strong vanishing-learning-signal hypothesis rejected.

### Transition construction and Bellman semantics have been audited

A State-Action learner transition spans:

1. the learner move;
2. the opponent response when the game continues;
3. the next learner decision state.

Reward shaping covers the same interval.

Real chess terminal transitions disable bootstrapping.

Artificial truncations can either disable bootstrapping (the default)
or retain legal next actions for bootstrapping, controlled by
`bootstrap_truncation`.

No implementation error has been identified in this transition path.

Status: audited; no bug established.

### Exact replay duplication is low

The final 10,000-transition training replay contained:

- `1,685` unique action IDs;
- `9,910` unique encoded states;
- `9,977` unique encoded-state/action pairs.

The exact 150-transition truncated greedy diagnostic game contained:

- `105` unique action IDs;
- `149` unique encoded states;
- `149` unique encoded-state/action pairs.

The greedy trajectory nevertheless showed substantially greater action-ID
concentration and visible repeated movement patterns.

Therefore strong exact state/state-action replay redundancy is not supported as
a sufficient explanation.

This does not prove that exploration or chess-position coverage is sufficient.

Status: strong exact-redundancy hypothesis weakened substantially.

## Controlled parameter experiments

### Increasing gamma from 0.99 to 0.995

A controlled probe changed only `gamma` from the baseline `0.99` to `0.995`.

The experiment did not improve greedy evaluation and produced greater
action concentration in the diagnostic trajectory.

This does not prove that `gamma=0.99` is optimal.

It does provide evidence against the specific hypothesis that a small increase
to `0.995` is sufficient to solve the observed delayed-credit problem.

Status: tested; hypothesis not supported by this experiment.

### Increasing truncation penalty from -0.1 to -0.2

A controlled probe changed only the artificial truncation penalty from `-0.1`
to `-0.2`.

The stronger penalty did not reduce greedy truncation and did not establish a
meaningful improvement in policy behavior.

Average episode reward from this experiment must not be directly interpreted as
better or worse than baseline because the reward function itself changed.

The final diagnostic truncation transition still showed a large discrepancy
between its negative terminal target and its positive predicted Q-value.

The production value was restored to `-0.1`.

Status: tested; simply increasing the penalty is not supported as the next
step.

## Prioritized Experience Replay diagnostic

The current training path uses Prioritized Experience Replay with
`PER_ALPHA = 0.6`.

The reproducible baseline replay contained:

- `10,000` transitions;
- `73` transitions with `done=True`;
- `9,927` transitions with `done=False`.

The `done=True` transitions therefore represented only `0.73%` of the retained
replay.

Their priorities were nevertheless larger:

- terminal priority mean: `0.049633`;
- terminal priority median: `0.017887`;
- non-terminal priority mean: `0.007798`;
- non-terminal priority median: `0.004111`.

After applying PER priority scaling, `done=True` transitions received
approximately `1.9182%` of total sampling probability.

Relative to their `0.73%` replay share, this is an aggregate oversampling factor
of approximately `2.628x`.

This establishes that PER does not treat all `done=True` transitions as though
they were sampled uniformly.


### Category-resolved final-replay measurement

The baseline probe now labels retained `done=True` transitions externally by
actual episode outcome, without changing the replay representation:

| Outcome | Count | Replay share | PER probability share | Oversampling |
| --- | ---: | ---: | ---: | ---: |
| Truncation | 54 | 0.5400% | 1.5488% | 2.868x |
| Win | 6 | 0.0600% | 0.1615% | 2.691x |
| Loss | 3 | 0.0300% | 0.0928% | 3.093x |
| Draw | 10 | 0.1000% | 0.1152% | 1.152x |

All 73 terminal transitions are accounted for; category PER shares total
approximately 1.9183% (rounding). Wins and losses are relatively oversampled,
so the strong H1 claim that PER ignores genuine decisive endings is weakened.

However, only nine decisive terminal transitions remain in the replay, and
the combined win/loss probability mass is approximately 0.2543% per draw.
This is not evidence that the learner sees enough decisive endings to learn.

**Measurement boundary:** the category table describes sampling
probabilities in the final replay, not historical sampling counts.

### Actual PER selections during optimization

The controlled 100-episode rerun recorded 412,416 PER selections.

- Non-terminal: 406,265 (98.5085%).
- Artificial truncation: 4,686 (1.1362%).
- Win: 482 (0.1169%).
- Loss: 443 (0.1074%).
- Draw: 540 (0.1309%).

Decisive endings accounted for 925 selections (0.2243%).

This confirms that PER repeatedly samples genuine decisive endings,
but their absolute exposure remains small.

These counts do not measure distinct batches, gradient contribution
or successful propagation of terminal rewards.

Status: final-replay probabilities and historical sampling
frequencies measured; learning impact remains unresolved.


## Active hypotheses


### H1 — Genuine chess endings receive limited absolute exposure

The final replay contains six wins and three losses, compared with
54 artificial truncations.

PER relatively favors decisive endings, and actual sampling records
confirm 482 win selections and 443 loss selections during training.

These 925 selections represent approximately 0.2243% of all
412,416 selections.

The hypothesis that PER ignores decisive endings is not supported.
However, the limited absolute exposure may still constrain learning.

Selection frequency alone cannot establish whether the terminal
signal propagates effectively to earlier decisions.

Status: sampling behavior measured; causal contribution to weak
greedy performance remains OPEN.

### H2 — Artificial horizon terminality may affect learning

Status: experimentally tested; no improvement demonstrated.

The baseline stores artificial truncations as Bellman-terminal
transitions. An optional implementation now permits bootstrapping
while retaining the existing truncation penalty.

A controlled 100-episode experiment enabled
`bootstrap_truncation=True`, preserving gamma, PER configuration,
training horizon and truncation penalty.

The baseline and bootstrapping runs both achieved a final balanced
greedy RandomAgent score of 0.113.

Greedy truncations were 29/40 in the baseline and 30/40 with
bootstrapping.

The experiment therefore did not establish improved greedy
performance. It also does not prove that bootstrapping is harmful
or ineffective under other training conditions.

The final diagnostic-game TD recording was subsequently updated
to support the same semantics, but corrected TD measurements
have not yet been collected.

This hypothesis is no longer the highest-priority next experiment.

### H3 — Ordinary-action credit assignment remains too weak or noisy

The network demonstrates strong preferences in some positions but frequent
near-ties among its highest-valued ordinary actions.

Material shaping, terminal reward, step penalty, discounting and replay
prioritization jointly determine the signal from which these distinctions must
be learned.

Evidence for:

- repeated near-ties among top ordinary legal actions;
- persistent weak greedy behavior despite successful optimization.

Evidence against / limitations:

- TD errors and Bellman targets are not globally collapsed;
- increasing gamma to `0.995` did not solve the behavior;
- stronger truncation penalty did not solve the behavior;
- no isolated experiment has yet established reward/credit assignment as the
  primary cause.

Status: OPEN.

### H4 — Exploration may still be insufficient in a chess-relevant sense

Exact replay duplication is low, but state uniqueness alone does not establish
useful chess coverage.

A replay containing many unique positions can still fail to expose the learner
to sufficiently informative tactical or terminal situations.

Evidence for:

- weak greedy policy despite large observable replay diversity;
- relatively few real episode endings compared with truncations.

Evidence against / limitations:

- the replay is not dominated by exact duplicates;
- no chess-semantic coverage metric has been established.

Status: OPEN.

## Hypotheses currently deprioritized

The following explanations should not currently drive the next experiment
without new evidence:

- claimable draws as the main cause of truncation;
- global Q-value collapse;
- vanishing TD errors;
- strong exact replay duplication;
- simply increasing gamma from `0.99` to `0.995`;
- simply making the truncation penalty more negative.

They are not necessarily impossible contributors. They have merely received
enough contrary evidence that another immediate experiment on them would have
lower diagnostic value.


## Next diagnostic question

Can the State-Action DQN learn a simple chess-relevant
state-action preference under controlled conditions?

Before further full-game training, define a small task with:

1. a measurable correct action or action preference;
2. controlled training and evaluation positions;
3. a fixed training budget;
4. a success criterion established before running the experiment.

A positive result would demonstrate limited learning capability,
not general chess-playing strength.

A negative result would motivate examining representation,
optimization and reward propagation before scaling training.

## Diagnostic discipline

For subsequent investigations:

1. State the hypothesis before changing code.
2. State what observation would support it.
3. State what observation would weaken it.
4. Change one meaningful variable or measurement at a time.
5. Reuse the reproducible probe whenever possible.
6. Preserve the baseline configuration unless the experiment explicitly changes
   it.
7. Separate observed measurements from interpretation.
8. Do not declare a hypothesis solved or rejected more strongly than the
   evidence permits.
9. Record negative results when they eliminate a plausible explanation.
10. Prefer measurements that distinguish competing hypotheses before adding new
    algorithmic complexity.

  
---

## Controlled one-step capture-learning evidence (October 2026)

The State-Action network was trained on terminal, one-step knight captures
where the correct preference is rook over pawn. The material rewards are
`0.05` and `0.01`, respectively. The existing agent optimizer was reused.

With 100 training and 40 held-out positions, 50 updates increased training
accuracy from 48% to 76%, but evaluation accuracy only from 57.5% to 60%.
The final mean correct-minus-incorrect Q gaps were +0.013290 on training
and +0.007471 on evaluation.

With an expanded generator, 300 training and 100 held-out positions,
50 updates increased training accuracy from 50.7% to 64.3%, while evaluation
accuracy changed from 52% to 50%. The final mean Q gaps were +0.004421
on training and +0.000597 on evaluation. Both runs sharply reduced
training loss. See `TRAINING_STATUS.md` for the full measurements.

The runs are not a controlled comparison of dataset size: the generator,
evaluation set and transitions per update changed. These results show
limited learning of the capture preference and no reliable held-out
generalization; they do not prove overfitting or a specific defect.

### Additional hypotheses to discriminate

- **Data/target correctness:** the terminal transitions, action IDs,
  material rewards, Bellman targets and predicted Q-values must be audited
  end to end before changing optimization.
- **Representation/optimization difficulty:** the model receives board
  planes and origin/destination embeddings, not an explicit captured-piece
  feature; learning the spatial relationship may be difficult.
- **Capacity relative to the task:** the CNN, embeddings and Q-head may
  have more capacity than needed for this narrow exercise. A train/eval gap
  is compatible with overfitting but does not establish that capacity is
  the cause. A controlled capacity comparison is premature until the
  data/target audit is complete.

**Highest-value next diagnostic:** verify the construction of a small
set of terminal capture transitions and the exact Q targets and predictions
used by the existing training path. Do not change the architecture, reward
scale or optimizer during that audit.
