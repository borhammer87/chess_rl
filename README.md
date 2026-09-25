# Chess RL

Educational chess reinforcement-learning project built around DQN variants and `python-chess`.

The repository currently keeps two model approaches in parallel: the original fixed-output `DQNCNN`, which is the model used by the main frozen-opponent self-play workflow, and an experimental explicit `StateActionDQN`. The State-Action path supports sustained end-to-end RandomAgent training and has completed reproducible 100-episode and 500-episode CPU probes, but it has not replaced the original model and has not demonstrated reliable playing strength.

## Current capabilities

- Chess environment backed by `python-chess`.
- Absolute 18 × 8 × 8 board encoding: 12 piece planes plus four castling-right planes, en-passant target and side to move.
- Fixed 4272-action encoding with distinct queen, rook, bishop and knight promotions.
- Legal-action filtering for exploration and greedy action selection.
- Original CNN DQN with policy and target networks.
- Experimental explicit State-Action DQN with structured action features and a shared `Q(s, a)` head.
- Reproducible State-Action CPU probe with balanced greedy pre/post evaluation and separate fixed seeds for initialization, training and evaluation.
- Replay buffer with Prioritized Experience Replay (PER), importance-sampling weights and TD-error priority updates.
- DQN-vs-RandomAgent episodes and generic DQN-vs-opponent episode execution.
- Multi-episode training with alternating learner color.
- Frozen-policy self-play for the original `DQNCNN` agent.
- Independent target-network and frozen-opponent synchronization.
- Agent-perspective learning rewards while the board representation remains absolute.
- Material reward shaping, a small non-terminal step penalty and a separate artificial-truncation penalty.
- Training summaries that classify chess outcomes from the actual game result rather than from shaped reward.
- Periodic checkpoints, replay-buffer persistence and automatic resume from `checkpoints/latest.pt`.
- Balanced White/Black evaluation against `RandomAgent` and evaluation-based `checkpoints/best.pt` selection.
- Truncation diagnostics and greedy PGN export to `checkpoints/evaluation_game.pgn`.
- Automated tests covering the environment, encoders, agents, replay, training, self-play, checkpointing, diagnostics and both DQN model families.

## Installation

The package uses the `src/` layout and is configured by `pyproject.toml` as version `0.9.0`.

The repository metadata currently does not declare runtime dependencies. A working development environment therefore needs the project installed/importable plus the libraries used by the source and tests, notably PyTorch, `python-chess`, NumPy and pytest.

## Running training

From an environment in which `chess_rl` and its dependencies are available:

```bash
python -m chess_rl.training.train_dqn
```

`main()` currently:

1. creates the original `DQNAgent`, a `RandomAgent` benchmark and a replay buffer;
2. loads `checkpoints/latest.pt` when it exists;
3. creates a frozen copy of the loaded/current policy for self-play;
4. trains for 100 episodes with a 150 learner-step horizon, alternating White and Black;
5. uses batch size 32 and starts replay training at 1000 stored transitions;
6. synchronizes the target network every 10 completed episodes;
7. saves `latest.pt`, evaluates against `RandomAgent`, and refreshes the frozen opponent every 25 completed episodes;
8. keeps `best.pt` only when the balanced RandomAgent evaluation score strictly improves;
9. builds the final console metrics through the existing `TrainingSummary` infrastructure;
10. writes one final greedy diagnostic game to `checkpoints/evaluation_game.pgn`.

The balanced evaluation score is `(wins + 0.5 * draws) / episodes`; truncated games score zero.

## Current limitations

- The repository does not establish that either model has learned reliable chess-playing strength.
- The documented diagnostic history reports a high truncation rate and non-progressing greedy play for the original DQN despite several learning-signal experiments.
- `RandomAgent` is only a provisional stable benchmark, not a strong chess benchmark.
- Reproducible State-Action probes at 100 and 500 training episodes do not establish reliable playing strength.
- The balanced greedy RandomAgent score was `0.113` after both 100 and 500 training episodes, while truncation remained very high (`29/40` and `32/40` respectively).
- The State-Action agent is not integrated into the frozen-opponent self-play path used by `main()`.
- Board encoding omits repetition history and move counters and remains absolute rather than agent-relative.
- Checkpoints do not preserve random-number-generator state or a lifetime episode counter.
- Explicit device/CUDA management is not implemented.
- Runtime dependencies are not declared in `pyproject.toml`.

## Current objective

The immediate development objective is to diagnose the persistent truncation and apparent learning plateau observed in the reproducible State-Action CPU experiments.

The State-Action path is now known to train end-to-end at practical CPU cost, so explicit CUDA support is not the immediate priority. Likewise, simply extending the same training configuration is not currently justified by the evidence: increasing training from 100 to 500 episodes left the balanced greedy RandomAgent score at `0.113` and did not reduce the high truncation rate.

The next development step should therefore inspect the learning setup and identify the smallest diagnostic capable of distinguishing among plausible causes before changing hyperparameters, scaling training further or integrating State-Action into frozen-opponent self-play.