# TRAINING STATUS

## Purpose

This document records what has actually been validated about training and what remains experimental. It is not a feature inventory; `CURRENT_STATE.md` and `ARCHITECTURE.md` describe the implementation itself.

## Current training paths

### Original DQNCNN path

The original `DQNAgent` supports training against `RandomAgent` and against a periodically refreshed frozen `DQNCNN` opponent. The executable `main()` uses frozen-opponent self-play for experience generation and keeps `RandomAgent` as the stable evaluation benchmark.

Training alternates learner color. Rewards are converted to learner perspective while the board tensor remains absolute.

### State-Action path

`StateActionDQNAgent` is compatible with the existing
`train_against_random()` path, including PER, target synchronization and
epsilon decay.

The repository also contains `scripts/run_state_action_probe.py`, a dedicated
CPU validation runner that reuses the existing training, evaluation and
`TrainingSummary` infrastructure rather than implementing a separate training
system.

The probe:

- initializes the State-Action model from a fixed seed;
- performs a balanced greedy evaluation against `RandomAgent`;
- trains against `RandomAgent` with alternating learner colors;
- measures wall-clock training time;
- performs a second balanced greedy evaluation;
- resets the evaluation RNG to the same seed before both evaluations;
- uses separate fixed seeds for model initialization, evaluation and training.

Two consecutive 100-episode runs with the same seeds produced identical
training and evaluation metrics on the development machine, confirming
reproducibility for that configuration. Wall-clock time differed slightly, as
expected.

Controlled CPU probes have now gone beyond smoke-test level.

For the reproducible 100-episode probe:

- 40-game initial greedy evaluation:
  `0 wins / 0 draws / 10 losses / 30 truncations`, score `0.000`;
- training:
  `7 wins / 11 draws / 4 losses / 78 truncations`;
- final epsilon: `0.6274`;
- replay reached its configured capacity of `10,000`;
- 40-game final greedy evaluation:
  `2 wins / 5 draws / 4 losses / 29 truncations`, score `0.113`;
- training time was approximately 124–125 seconds in two consecutive runs.

The two consecutive runs reproduced the same discrete metrics, average plies,
average reward, average loss, epsilon and replay size.

For the reproducible 500-episode probe, using the same initialization,
evaluation and training seeds and otherwise retaining the controlled
configuration:

- 40-game initial greedy evaluation:
  `0 wins / 0 draws / 10 losses / 30 truncations`, score `0.000`;
- training:
  `46 wins / 70 draws / 25 losses / 359 truncations`;
- average plies: `274.44`;
- average reward: `-0.0895`;
- average loss: approximately `0.00052025`;
- final epsilon reached the configured floor of `0.1000`;
- replay remained at its configured capacity of `10,000`;
- 40-game final greedy evaluation:
  `3 wins / 3 draws / 2 losses / 32 truncations`, score `0.113`;
- training time was approximately 870 seconds (14.5 minutes).

These results provide evidence that sustained State-Action training executes
correctly and is computationally practical on the current CPU. They also show
some change in greedy behavior after training: among non-truncated evaluation
games, the trained policies produced wins and draws where the initial policy
lost all ten completed games.

However, the results do **not** establish reliable learning or playing
strength. The balanced greedy score was `0.113` after both 100 and 500
training episodes, while truncation remained very high and increased from
`29/40` after the 100-episode run to `32/40` after the 500-episode run.
Therefore the 500-episode result does not support the hypothesis that simply
training the current configuration for longer produces continuing improvement.

The State-Action agent is still not connected to frozen-opponent self-play or
`main()`.

### State-Action truncation diagnostics

The reproducible 100-episode State-Action probe was extended with targeted
diagnostics to investigate the persistent truncation rate.

Claimable-draw diagnostics showed:

- initial greedy evaluation: all `30/30` truncations ended without a claimable
  threefold repetition or fifty-move draw;
- training: all `78/78` truncations ended without either claimable draw;
- final greedy evaluation: `29` truncations, with `1` position claimable by
  threefold repetition, `1` claimable by the fifty-move rule, and `28`
  truncations without either claimable draw.

These counters are diagnostic properties and are not required to be mutually
exclusive. The result nevertheless shows that claimable draws do not explain
the high truncation rate.

Material diagnostics use `78` as the initial total non-king material value.
For truncated games, the same reproducible 100-episode probe produced:

| Diagnostic | Initial greedy | Training | Final greedy |
| --- | ---: | ---: | ---: |
| Truncated games | 30 | 78 | 29 |
| Average final total material | 20.20 | 10.55 | 10.83 |
| Average learner material balance | -6.27 | +0.45 | -2.62 |
| Average absolute material balance | 9.13 | 5.81 | 6.28 |

These diagnostics materially narrow the problem. The truncated games are not
generally reaching the artificial horizon with most material still on the
board. After training, final greedy truncated games retain only about `10.83`
of the initial `78` non-king material points on average.

The final greedy policy also ends truncated games less far behind on average
than the initial greedy policy (`-2.62` versus `-6.27`), while the balanced
evaluation score changes from `0.000` to `0.113`.

This is reproducible evidence of a behavioral change after training, but it is
not sufficient evidence of reliable chess strength or general improvement.
The remaining diagnostic question is what prevents heavily simplified greedy
games from terminating.

### Bellman-target and TD-error diagnostic

The final replay learning signal has now been measured directly without
changing the training algorithm.

For the non-terminal transitions in the final 10,000-transition training
replay:

- Bellman-target standard deviation: `0.073968`
- selected-action Q-value standard deviation: `0.063227`
- mean absolute TD error: `0.020054`
- absolute-TD-error standard deviation: `0.023346`

The same diagnostic was then applied to the exact replay buffer belonging to
the saved 300-ply truncated greedy game. Its 149 non-terminal learner
transitions produced:

- Bellman-target standard deviation: `0.035757`
- selected-action Q-value standard deviation: `0.028926`
- mean absolute TD error: `0.023659`
- absolute-TD-error standard deviation: `0.026904`

The terminalized truncation transition had reward and Bellman target
`-0.100000`, while its selected-action Q-value remained `0.284372`, giving an
absolute TD error of `0.384372`.

These results do not support global Bellman-target collapse or vanishing TD
error as a sufficient explanation for the weak greedy policy.

### State-Action transition audit

The temporal semantics of the State-Action replay transition have been checked
against the episode runner and training implementation.

A non-terminal transition represents one complete learner decision interval:

1. the learner observes state `s_t`;
2. the learner selects action `a_t`;
3. the opponent replies if the learner move did not terminate the game;
4. `s_(t+1)` is encoded when it is again the learner's turn.

The material-shaping reward measures the material change over that same
interval. The stored legal next actions therefore correspond to the learner's
next decision. Terminal and truncated transitions are stored without legal
next actions and do not bootstrap.

No implementation defect has been identified in this transition construction
or its Bellman target semantics.

With `gamma = 0.99`, however, a signal occurring many learner decisions later
is necessarily discounted substantially. This is a credit-assignment
property, not evidence by itself that gamma should be changed.

### Replay-diversity diagnostic

Observable diversity in the final 10,000-transition State-Action training
replay was measured without changing the training algorithm.

The replay contained:

- `1,685` unique action IDs;
- `9,910` unique encoded states;
- `9,977` unique encoded-state/action pairs;
- `90` repeated encoded states;
- `23` repeated state-action pairs.

Action-ID concentration was:

- top 10 actions: `3.6%`;
- top 25 actions: `7.7%`;
- top 50 actions: `13.5%`.

This does not prove that the training experience provides sufficient chess
coverage, because no objective threshold for sufficient coverage has been
established. It does show that the retained replay is not dominated by exact
state or state-action repetition.

The same diagnostic was applied to the exact 150-transition truncated greedy
game. It contained:

- `105` unique action IDs;
- `149` unique encoded states;
- `149` unique encoded-state/action pairs;
- `1` repeated encoded state;
- `1` repeated state-action pair.

Its action-ID concentration was substantially higher:

- top 10 actions: `24.0%`;
- top 25 actions: `46.0%`;
- top 50 actions: `63.3%`.

The greedy game therefore does not consist primarily of exact state cycles.
Almost every encoded state is different, even though the policy repeatedly
uses a narrower set of move IDs. This is compatible with the observed local
back-and-forth move patterns because opponent moves and other board changes can
produce a new encoded state while the learner reuses the same movement between
squares.

The comparison should not be interpreted as proof of policy collapse: a
single 150-transition game and a 10,000-transition multi-episode
epsilon-greedy replay have intrinsically different action-availability and
sampling properties.

Strong replay redundancy is therefore not supported as a sufficient
explanation for the current learning plateau. Reward and credit assignment are
reasonable next hypotheses to inspect, but they have not yet been established
as the cause.

### Concrete truncated-game diagnostic

The State-Action probe was extended to search for the first truncated greedy
White-vs-RandomAgent evaluation game within a bounded number of attempts,
reusing the existing PGN export path.

A truncated game was successfully captured at the configured artificial
horizon of `150` learner steps / `300` total plies. This provides a concrete
trajectory for inspecting policy behavior rather than relying only on aggregate
truncation statistics.

The repository also contains a tested
`evaluate_state_action_greedy_choice()` utility. Given a State-Action network,
encoded state and legal moves, it returns the selected greedy action, the
highest legal Q-value and the gap between the highest and second-highest legal
Q-values.

The probe now reconstructs every learner position in the captured truncated
PGN and evaluates it using the exact final policy that generated the game.
All `150` recorded learner moves matched the reconstructed greedy actions, so
the Q-value diagnostics correspond to the actual recorded trajectory.

The diagnostic also measures the complete legal-action Q-value distribution
for each learner position: legal-action count, best Q-value, top-two gap, mean,
minimum, standard deviation and full range.

The results rule out the strongest version of the Q-value-collapse hypothesis.
The network does not simply assign the same value to every legal action.
Full legal-action ranges are commonly on the order of hundredths and can be
substantially larger.

However, many positions still show extremely small separation between the two
highest-valued actions. Examples include top-two gaps of `0.000069`,
`0.000106`, `0.000039` and a minimum observed gap of `0.000002`.

The behavior is not uniform. Queen promotions were strongly preferred in the
captured trajectory, with top-two gaps of `0.063167` and `0.034973`. This
provides concrete evidence that the network can learn strong action
preferences in at least some positions.

The current interpretation is therefore narrower than global Q-value collapse:
the State-Action policy has learned some differentiation across legal actions,
but frequently produces near-ties among its highest-valued ordinary choices.
The cause of that behavior is not yet established.

## Learning signal

The environment supplies terminal reward from White's perspective:

- White win: `+1`;
- Black win: `-1`;
- draw/unfinished: `0`.

The training layer converts this to learner perspective and adds experimental shaping:

- material change: `0.01 * learner-perspective net material change`;
- ordinary non-terminal step penalty: `-0.0005`;
- artificial truncation penalty on the final transition: `-0.1`.

For an artificial horizon termination, the environment remains non-terminal but the stored final replay transition is terminal for Bellman learning. The ordinary step penalty is not also added to that final truncated transition.

Because reward is shaped, accumulated `total_reward` is a learning diagnostic only. Win/draw/loss classification uses the real chess result plus learner color.

## Replay and optimization

The current main learning path uses Prioritized Experience Replay.

- `PER_ALPHA = 0.6`;
- `PER_BETA = 0.4`;
- `PER_PRIORITY_EPSILON = 1e-6`;
- new transitions receive current maximum priority;
- prioritized samples are drawn with replacement;
- normalized importance-sampling weights scale per-transition MSE;
- sampled priorities are updated from absolute TD errors.

The policy and target networks are separate. Epsilon decay occurs once after an episode in which replay training actually occurred, rather than once per individual optimizer update.

## Evaluation and checkpoint validation

Evaluation against `RandomAgent` is greedy (`epsilon=0`) and restores the original epsilon afterward. It does not intentionally train the agent or use the training replay buffer.

Balanced evaluation combines equal numbers of White and Black games. Model-selection score is:

`(wins + 0.5 * draws) / episodes`

Truncations score zero.

`latest.pt` is the resumable training checkpoint. `best.pt` is the training state associated with the highest balanced RandomAgent evaluation score observed by the executable workflow. Replay state and priorities are persisted with the agent state.

## Diagnostic history preserved by the repository

The repository documentation records the following sequence of learning-signal investigations for the original fixed-output DQN:

1. high truncation rates were observed;
2. diagnostics separated claimable draws from other truncations and found that claimable-draw rules did not explain most truncations;
3. a `-0.1` artificial-truncation penalty was added and was insufficient by itself;
4. small material-based shaping was added, while chess outcomes were separated from shaped reward;
5. replay TD-error diagnostics motivated comparing standard DQN and Double-DQN targets; the analysed sample did not show a practical target difference;
6. PER was added to emphasize high-error experiences;
7. the first documented PER experiment did not show a clear greedy RandomAgent improvement;
8. legal-action Q diagnostics showed small gaps among top actions during non-progressing play;
9. a small `-0.0005` ordinary step penalty was added;
10. the documented PER + step-penalty experiment still did not establish reliable greedy play;
11. an alternative explicit State-Action DQN was then implemented in parallel and brought to CPU smoke-test level.
12. State-Action was subsequently validated with reproducible 100-episode and 500-episode CPU probes;
13. the balanced greedy RandomAgent score was `0.113` after both probes, while truncation remained very high;
14. the 500-episode result therefore shifted the immediate focus from simply scaling training to diagnosing the persistent truncation and apparent learning plateau.

These are qualitative repository-recorded conclusions. This HEAD does not include raw experiment logs sufficient to independently reproduce numerical historical results, so this document does not invent episode counts or scores that are not preserved.

## Fresh validation status

A preserved automated-suite run from an earlier point in the State-Action
probe work reported:

`257 passed in 11.29s`

The suite was subsequently reported green after later diagnostic and
replay-buffer changes. The repository has gained additional test coverage since
the preserved 257-test run, so that number should be treated as historical
rather than as the exact current test count.

The controlled State-Action CPU experiments and subsequent diagnostics
documented above were executed in the actual project environment.

## Open validation questions

- What prevents heavily simplified greedy State-Action games from terminating,
  despite final truncated positions retaining only a small fraction of the
  initial non-king material on average?
- Is the current material-shaping / terminal-reward signal sufficiently
  informative to distinguish ordinary actions whose consequences may only
  become useful many learner decisions later?
- Can a small controlled experiment isolate reward or credit assignment without
  simultaneously changing gamma, PER, exploration, representation or network
  architecture?  
- Why did the controlled greedy score remain `0.113` when training increased
  from 100 to 500 episodes?
- Is the current limitation primarily related to reward design, replay/PER
  behavior, epsilon scheduling, state representation, Bellman targets, model
  capacity, or another part of the learning setup?
- How does State-Action compare with the original fixed-output DQNCNN under a
  controlled equivalent experiment?
- Does either model improve materially against a benchmark stronger or more
  informative than `RandomAgent`?
- Should State-Action eventually be integrated into frozen-opponent self-play?

Explicit CUDA/device support remains unimplemented, but the measured CPU cost
does not currently justify treating CUDA as the immediate priority. The
500-episode State-Action probe completed in approximately 14.5 minutes on the
development machine. The more immediate problem is diagnosing the persistent
learning/truncation behavior before scaling training further.