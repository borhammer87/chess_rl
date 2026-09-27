# CURRENT STATE

## Status

Repository version: `0.9.0`.

The HEAD contains a mature original DQN training/evaluation/self-play pipeline plus a second experimental State-Action DQN implementation. The original `DQNCNN` remains the model used by `main()` and frozen-opponent self-play. The State-Action model coexists with it and has progressed beyond smoke-test level to reproducible 100-episode and 500-episode CPU training probes against `RandomAgent`; it has not replaced the original workflow.

The State-Action probes establish that sustained end-to-end training works and is computationally practical on CPU, but they do not establish reliable chess-playing strength. Greedy evaluation score remained `0.113` after both 100 and 500 training episodes, with high truncation rates.

## Implemented at HEAD

### Environment and representation

- `ChessEnv` based on `python-chess`.
- Independent board copies returned to callers.
- 18-channel absolute board tensor.
- Side to move, castling rights and en-passant target encoded.
- Fixed 4272-action external action space.
- Explicit queen, rook, bishop and knight promotion actions.
- Legal action masking/filtering.

### Original DQN

- `DQNCNN` fixed-output CNN.
- `DQNAgent` with policy/target networks, Adam optimizer and epsilon-greedy exploration.
- Bellman updates restricted to legal next actions.
- Target-network synchronization.
- Serialization/checkpoint support.

### State-Action DQN

- `StateActionDQN` with a 256-dimensional state representation.
- Structured from-square, to-square and promotion embeddings.
- Shared `Q(s, a)` head.
- Reuse of one state encoding across candidate actions.
- Batched selected-action scoring and batched legal-next-action maxima.
- `StateActionDQNAgent` with policy/target networks, Bellman updates, PER weights, TD-error reporting, epsilon scheduling and serialization.
- End-to-end CPU training through `train_against_random()`.
- Dedicated reproducible CPU probe in `scripts/run_state_action_probe.py`.
- Fixed seeds for model initialization, training and balanced pre/post evaluation.
- Controlled 100-episode and 500-episode RandomAgent training probes.
- Claimable-draw diagnostics for State-Action training and greedy evaluation
  truncations.
- Final-material diagnostics for truncated episodes, including total remaining
  material, learner material balance and absolute material balance.

### Replay and learning

- Replay buffer persistence.
- Prioritized Experience Replay with `alpha=0.6`, `beta=0.4` and priority epsilon `1e-6`.
- Importance-sampling weighted loss.
- Priority updates from absolute TD error.
- Agent-perspective rewards.
- Material shaping at scale `0.01`.
- Ordinary step penalty `-0.0005`.
- Artificial truncation penalty `-0.1`.
- Artificially truncated final replay transitions treated as terminal for Bellman learning.
- Chess outcomes classified from the actual game result, independently of shaped reward.

### Training, evaluation and self-play

- DQN-vs-RandomAgent episodes.
- Generic DQN-vs-opponent episode engine.
- Multi-episode training.
- Alternating White/Black learner color.
- Frozen `DQNCNN` opponent self-play.
- Independent target and opponent refresh schedules.
- Balanced White/Black evaluation against `RandomAgent`.
- Periodic checkpoint and evaluation callbacks.
- Normalized chess score for best-checkpoint selection.
- Truncation diagnostics for claimable threefold/fifty-move draws.
- Greedy diagnostic PGN export.
- Aggregate `TrainingSummary` and `EvaluationSummary` data structures.

## Exact current `main()` workflow

`python -m chess_rl.training.train_dqn` currently:

1. creates `ChessEnv`, original `DQNAgent(epsilon=1.0)`, `RandomAgent` and a replay buffer of capacity 10,000;
2. reads the previous best evaluation score from `checkpoints/best.pt` metadata when available;
3. restores agent and replay state from `checkpoints/latest.pt` when available;
4. creates the frozen training opponent only after checkpoint loading;
5. trains against that frozen opponent for 100 episodes;
6. uses a maximum of 150 learner steps per episode;
7. uses batch size 32 and minimum replay size 1000;
8. alternates learner color;
9. synchronizes the target network every 10 completed episodes;
10. saves `latest.pt` every 25 completed episodes;
11. evaluates against `RandomAgent` every 25 completed episodes using 10 games per color;
12. refreshes the frozen opponent every 25 completed episodes;
13. replaces `best.pt` only when the balanced evaluation score strictly improves;
14. calls `summarize_training(results)` and prints the resulting summary fields;
15. saves one final greedy White-vs-RandomAgent game to `checkpoints/evaluation_game.pgn`.

This corrects older documentation that stated a 15-episode target synchronization interval.

## Test status

The automated suite was run successfully in the actual `chess-rl` development environment after the State-Action CPU probe runner was added:

`257 passed in 11.29s`

A later experimental test around the standalone probe script was intentionally removed rather than changing the project package structure merely to make `scripts/` importable. The suite was subsequently reported green again before the reproducible State-Action probes continued.

The controlled 100-episode and 500-episode State-Action experiments were executed in the actual project environment.


Targeted diagnostics on the reproducible 100-episode configuration further
showed:

- initial greedy evaluation: `30/30` truncations without a claimable threefold
  or fifty-move draw;
- training: `78/78` truncations without either claimable draw;
- final greedy evaluation: `29` truncations, of which `28` had neither
  claimable draw; one was claimable by threefold and one by the fifty-move
  rule;
- initial greedy truncated games: average final total material `20.20/78`,
  average learner material balance `-6.27`, average absolute balance `9.13`;
- training truncations: average final total material `10.55/78`, average
  learner material balance `+0.45`, average absolute balance `5.81`;
- final greedy truncated games: average final total material `10.83/78`,
  average learner material balance `-2.62`, average absolute balance `6.28`.

These results rule out claimable draws as the main explanation for truncation
and show that truncated games are generally heavily simplified. They do not
establish reliable chess-playing strength.


## Current limitations

- No current evidence in the repository establishes reliable chess-playing strength.
- Historical diagnostics documented in the repository report many truncations and non-progressing greedy behavior for the original DQN.
- RandomAgent is a weak provisional benchmark.
- State-Action has reproducible 100-episode and 500-episode CPU validation, but neither experiment establishes reliable playing strength.
- Greedy State-Action evaluation remained heavily truncated: `29/40` games after 100 training episodes and `32/40` after 500.
- Balanced greedy RandomAgent score was `0.113` after both 100 and 500 training episodes, so simply extending the current training configuration did not show continuing improvement.
- State-Action is not integrated into frozen-opponent self-play or `main()`.
- Repetition history and move counters are absent from the board tensor.
- Board encoding remains absolute.
- Checkpoints omit RNG state and a lifetime episode counter.
- Explicit device/CUDA management is absent.
- `pyproject.toml` does not declare runtime dependencies.

## Current focus

The current focus is diagnosing why the experimental State-Action policy still
produces heavily truncated greedy games despite substantial material reduction.

The alternative architecture has demonstrated sustained end-to-end training,
reproducibility under fixed seeds and practical CPU execution time. Extending
training from 100 to 500 episodes did not improve the balanced greedy
RandomAgent score beyond `0.113`.

Targeted diagnostics show that claimable draws are not the main cause of
truncation and that final truncated positions are usually heavily simplified.

The State-Action probe can search for and save a truncated greedy
White-vs-RandomAgent game and analyze every learner decision against the final
policy network while that exact policy is still in memory.

In the reproducible 100-episode diagnostic run, all `150` recorded learner
moves in the `300`-ply truncated game matched the reconstructed greedy action.
This validates the saved trajectory for Q-value analysis.

The legal-action Q-value distributions do not support a simple global
Q-value-collapse explanation. The network often assigns meaningfully different
values across the complete legal action set, while the best and second-best
actions are nevertheless frequently separated by very small gaps.

Some decisions are clearly distinguished. In particular, queen promotions
showed large top-two gaps (`0.063167` and `0.034973`). Other ordinary decisions
were effectively near-ties, including gaps as small as `0.000002` and
`0.000039`.

The current evidence therefore suggests that the State-Action network has
learned some action preferences but often lacks strong separation among its
top candidate actions. This is evidence about policy behavior, not yet an
explanation of its cause.

## Next milestone

Investigate the learning signal that produces the observed State-Action
Q-value preferences before changing reward design, state representation,
network architecture or training hyperparameters.

The next diagnostic should determine whether Bellman targets and TD errors used
during State-Action training are themselves weakly differentiated or whether
the training signal contains stronger distinctions that the learned policy is
failing to represent.

Prefer diagnostic instrumentation over changes to the learning algorithm.
Use the existing training and State-Action abstractions where possible and do
not generalize temporary diagnostics into production APIs without a concrete
architectural reason.