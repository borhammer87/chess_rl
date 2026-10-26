# Chess RL

Educational chess reinforcement-learning project built around DQN variants and `python-chess`.

The repository currently keeps two model approaches in parallel: the original fixed-output `DQNCNN`, which is the model used by the main frozen-opponent self-play workflow, and an experimental explicit `StateActionDQN`. The State-Action path supports sustained end-to-end RandomAgent training and has completed reproducible 100-episode and 500-episode CPU probes, but it has not replaced the original model and has not demonstrated reliable playing strength.

## Current capabilities

- Chess environment backed by `python-chess`.
- Absolute 18 × 8 × 8 board encoding: 12 piece planes plus four castling-right planes, en-passant target and side to move.
- Fixed 4272-action encoding with distinct queen, rook, bishop and knight promotions.
- Legal-action filtering for exploration and greedy action selection.
- Original CNN DQN with policy and target networks.
- Experimental explicit State-Action DQN and parallel spatial variant with destination-square features, both using a shared `Q(s, a)` head.
- Reproducible State-Action probe with balanced greedy pre/post evaluation, fixed seeds, selectable standard/spatial architecture and explicit CPU/CUDA device selection.
- Replay buffer with Prioritized Experience Replay (PER), importance-sampling weights and TD-error priority updates.
- DQN-vs-RandomAgent episodes and generic DQN-vs-opponent episode execution.
- Multi-episode training with alternating learner color.
- Frozen-policy self-play for the original `DQNCNN` agent.
- Independent target-network and frozen-opponent synchronization.
- Agent-perspective learning rewards while the board representation remains absolute.
- Material reward shaping, a small non-terminal step penalty and a separate artificial-truncation penalty.
- Training summaries that classify chess outcomes from the actual game result rather than from shaped reward.
- Periodic checkpoints and replay-buffer persistence. The original DQN main workflow supports automatic resume from `checkpoints/latest.pt`.
- The experimental State-Action probe saves checkpoints every 25 completed episodes in a unique directory per execution under `checkpoints/state_action_probe/<run_id>/`, avoiding overwrites between runs. - The experimental State-Action probe saves checkpoints every 25 completed episodes in `checkpoints/state_action_probe/<run_id>/`. It supports explicit resume with `--resume`, restoring the agent, optimizer, epsilon, replay buffer, terminal labels, RNG state and cumulative episode numbering. Resume requires matching architecture and training configuration. `--episodes` specifies additional episodes when resuming.

- Balanced White/Black evaluation against `RandomAgent` and evaluation-based `checkpoints/best.pt` selection.
- Truncation diagnostics and greedy PGN export to `checkpoints/evaluation_game.pgn`.
- Automated tests covering the environment, encoders, agents, replay, training, self-play, checkpointing, diagnostics and both DQN model families.
- Truncation diagnostics covering claimable draws and final material state.
- One final State-Action greedy diagnostic game, regardless of outcome,
  included as a delimited PGN block in console output (and captured logs).
- Diagnostic State-Action greedy-choice inspection with best-Q and
  best-vs-second-best Q-value gap reporting.
- Final-replay and exact diagnostic-game Bellman-target / TD-error analysis.
- Verified reconstruction of all 150 learner decisions in the saved truncated
  greedy game against the final policy.
- Replay-diversity diagnostics covering unique encoded states, unique actions,
  unique state-action pairs, exact repetition and action concentration.

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
- Checkpoints preserve random-number-generator state or a lifetime episode counter.
- Explicit CPU/CUDA selection is available for the experimental State-Action probe, not the original `main()` self-play workflow. A CUDA-enabled PyTorch build and compatible NVIDIA GPU are required for `--device cuda`.
- Runtime dependencies are not declared in `pyproject.toml`.

## Current objective

The immediate development focus is diagnosing why the experimental
State-Action policy still produces heavily truncated greedy games and weak
separation among many top-ranked ordinary legal actions.

Q-value, Bellman-target, TD-error, transition-semantics and replay-diversity
diagnostics have not identified global value collapse, a Bellman-transition
implementation defect or strong replay redundancy as sufficient explanations.

The final 10,000-transition training replay contained 9,910 unique encoded
states and 9,977 unique state-action pairs. The saved 150-transition greedy
diagnostic game also contained 149 unique encoded states and 149 unique
state-action pairs, although its action usage was substantially more
concentrated than the training replay.

The PER investigation has distinguished genuine chess endings
from artificial truncations and measured their sampling exposure.

A controlled experiment enabling Bellman bootstrapping for artificial
truncations did not improve the final balanced greedy RandomAgent
score, which remained 0.113.

The next objective is to test whether State-Action DQN can learn
a small, controlled chess-relevant state-action preference before
committing additional resources to full-game training.

See `DIAGNOSTICS.md` for the evidence and active hypotheses.

## Experimental spatial diagnostic probe

Run the experimental State-Action probe as a Python module from the repository root:

```powershell
# New CUDA training run (25 episodes)
python -m scripts.run_state_action_probe --architecture spatial --device cuda --episodes 25

# Resume for 25 additional episodes
python -m scripts.run_state_action_probe --architecture spatial --device cuda --episodes 25 --resume "checkpoints/state_action_probe/<run_id>/episode_25.pt"
```

Replace `<run_id>` with the actual run directory. A CUDA-enabled PyTorch installation and compatible NVIDIA GPU are required for CUDA execution. Use `--device cpu` for CPU execution.

A resumed run saves checkpoints in the original run directory and preserves cumulative episode numbering. Resume compatibility checks cover the architecture and training configuration. Exact numerical equivalence between uninterrupted and resumed training has not yet been established.

The probe supports command-line configuration of architecture (spatial or standard), training episode count and device (cpu or cuda). The defaults are spatial, 100 episodes and CPU.

For example:

python scripts/run_state_action_probe.py --episodes 100 --architecture spatial --device cpu

# CUDA-enabled PyTorch and compatible NVIDIA GPU required:
python scripts/run_state_action_probe.py --episodes 100 --architecture spatial --device cuda

The probe prints the effective architecture, truncation-bootstrapping setting, random seeds, evaluation settings, replay capacity and training parameters before execution.

This diagnostic workflow is separate from the original DQNCNN self-play main() workflow. Other experiment parameters remain defined in the script.

The 100-episode spatial result and its limitations are recorded in TRAINING_STATUS.md and DIAGNOSTICS.md.