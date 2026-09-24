# CURRENT STATE

## Status

Repository version: `0.9.0`.

The HEAD contains a mature original DQN training/evaluation/self-play pipeline plus a second experimental State-Action DQN implementation. The original `DQNCNN` remains the model used by `main()` and frozen-opponent self-play. The State-Action model coexists with it and has reached a short CPU end-to-end training smoke test against `RandomAgent`; it has not replaced the original workflow.

No source-code change was made during the current documentation audit.

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
- End-to-end CPU smoke test through `train_against_random()`.

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

The repository contains 246 pytest test functions across 13 test modules. They cover the environment, encoders, legal action selection, both DQN model/agent families, replay/PER, episode logic, RandomAgent training, frozen-opponent self-play, checkpointing and diagnostics.

During this documentation audit, `pytest -q` was attempted in the available sandbox but collection could not start because the execution environment does not have the project dependencies/import setup (`python-chess` is missing and `chess_rl` is not installed on that interpreter). Therefore this audit does **not** claim a fresh passing test run. The test files themselves were inspected and the documentation only claims that the tests exist, not that they passed in this sandbox.

## Current limitations

- No current evidence in the repository establishes reliable chess-playing strength.
- Historical diagnostics documented in the repository report many truncations and non-progressing greedy behavior for the original DQN.
- RandomAgent is a weak provisional benchmark.
- State-Action validation is limited to a short CPU smoke test; no long training or comparative strength experiment is recorded.
- State-Action is not integrated into frozen-opponent self-play or `main()`.
- Repetition history and move counters are absent from the board tensor.
- Board encoding remains absolute.
- Checkpoints omit RNG state and a lifetime episode counter.
- Explicit device/CUDA management is absent.
- `pyproject.toml` does not declare runtime dependencies.

## Current focus

The current session is a documentation reconciliation milestone. The repository had accumulated contradictory snapshots: for example, some documents still described the PER + step-penalty analysis as the next milestone even though State-Action DQN had already been implemented and smoke-tested, and one document disagreed with the code about target synchronization frequency.

The documentation is now intended to describe one consistent HEAD: original DQNCNN self-play remains the executable main workflow; State-Action exists in parallel and has pipeline-level CPU validation only.

## Next milestone

Do not infer the next implementation change from stale documentation. After this audit, choose the smallest next development step from the repository and fresh validation evidence. CUDA is an open future possibility, not an already selected next task.
