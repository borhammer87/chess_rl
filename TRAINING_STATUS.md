# TRAINING STATUS

## Purpose

This document records what has actually been validated about training and what remains experimental. It is not a feature inventory; `CURRENT_STATE.md` and `ARCHITECTURE.md` describe the implementation itself.

## Current training paths

### Original DQNCNN path

The original `DQNAgent` supports training against `RandomAgent` and against a periodically refreshed frozen `DQNCNN` opponent. The executable `main()` uses frozen-opponent self-play for experience generation and keeps `RandomAgent` as the stable evaluation benchmark.

Training alternates learner color. Rewards are converted to learner perspective while the board tensor remains absolute.

### State-Action path

`StateActionDQNAgent` is compatible with the existing `train_against_random()` path, including PER, target synchronization and epsilon decay. The repository contains an end-to-end CPU smoke test that runs two short RandomAgent episodes and verifies that replay is populated and epsilon decays.

That smoke test validates plumbing only. It is not evidence of convergence, playing strength, stability over long runs or superiority to `DQNCNN`.

The State-Action agent is not currently connected to frozen-opponent self-play or `main()`.

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

These are qualitative repository-recorded conclusions. This HEAD does not include raw experiment logs sufficient to independently reproduce numerical historical results, so this document does not invent episode counts or scores that are not preserved.

## Fresh validation status for this audit

The repository contains 246 pytest test functions. A fresh `pytest -q` attempt in the documentation-audit sandbox failed during test collection because that interpreter lacks `python-chess` and does not have the `chess_rl` package installed/importable. Consequently there is no fresh green-suite claim from this audit.

The source and tests were inspected to reconcile documentation with implementation.

## Open validation questions

- Does the State-Action model improve learning quality beyond the short CPU smoke test?
- How does it compare with the original fixed-output DQN under controlled training/evaluation conditions?
- Does either model improve materially against a benchmark stronger or more informative than `RandomAgent`?
- Should State-Action eventually be integrated into frozen-opponent self-play?
- Are the current reward/PER hyperparameters useful beyond the diagnostic experiments that introduced them?

Explicit CUDA/device support is not implemented, but this audit does not select it as the next development task.
