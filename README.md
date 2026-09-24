# Chess RL

Chess reinforcement-learning project using a CNN-based DQN.

## Current capabilities

The project currently implements a complete DQN training, evaluation,
checkpointing, and balanced-color workflow.

- 18-channel board representation including turn, castling rights, and
  en passant state.
- 4272-action output space with explicit queen, rook, bishop, and knight
  promotions.

Implemented features:

- Chess environment based on python-chess.
- Board encoding using tensors.
- CNN-based DQN.
- Legal action masking.
- Replay Buffer.
- Prioritized Experience Replay with importance-sampling correction.
- TD-error-based replay-priority updates.
- Multi-episode training.
- Target network synchronization.
- Epsilon decay once per episode when replay training occurred.
- Training metrics collection.
- Aggregated training summaries.
- Console progress reporting.
- DQN training as White and Black.
- Alternating White/Black training episodes.
- Agent-perspective rewards.
- Greedy evaluation against RandomAgent.
- Balanced evaluation as White and Black.
- Periodic evaluation during training.
- Periodic training checkpoints.
- Automatic checkpoint loading.
- Replay-buffer persistence.
- Evaluation-based best-checkpoint selection.
- Frozen-policy self-play opponent.
- Multi-episode self-play with alternating colors.
- Independent target-network and frozen-opponent synchronization.
- Periodic evaluation of the learner against RandomAgent.
- Truncation diagnostics for claimable draws.
- Explicit `-0.1` penalty for artificial training truncation.
- Material-based reward shaping using scaled net material change.
- Small per-step penalty for non-terminal, non-truncated training transitions.
- Chess-outcome tracking independent of shaped training reward.
- Terminal replay treatment at the artificial training horizon.
- Greedy diagnostic evaluation game against RandomAgent.
- PGN export to `checkpoints/evaluation_game.pgn`.
- Experimental explicit state-action DQN architecture.
- Structured action features using origin square, destination square, and
  promotion type.
- Shared state-action Q-value head.
- Batched state-action evaluation without re-encoding a state for every
  legal action.
- State-action DQN training with Prioritized Experience Replay.
- State-action agent checkpoint compatibility.
- End-to-end StateActionDQNAgent CPU training smoke test against RandomAgent.

## Running training

Run:

`python -m chess_rl.training.train_dqn`

Training currently alternates the DQN between White and Black.

The program uses:

`checkpoints/latest.pt`

as the current resumable training checkpoint.

If the checkpoint exists, the agent and replay buffer are restored before
training continues.

The program also uses:

`checkpoints/best.pt`

to retain the training state with the highest balanced evaluation score.

Periodic evaluation currently uses equal numbers of games as White and
Black.

## Current limitations

- RandomAgent is retained as a stable evaluation benchmark and is no longer
the opponent used for the main training workflow.
- The board representation remains absolute rather than agent-relative.
- Some chess state such as repetition state, and move counters is not encoded.
- The experimental State-Action DQN is not yet integrated into the main
  frozen-opponent self-play workflow.
- Explicit CUDA/device management is not yet implemented.
- The State-Action DQN has only been validated with a short end-to-end CPU
  training smoke test; long-run learning quality has not yet been established.

## Next goal

## Next goal

Diagnostics of the original fixed-output DQN motivated development of an
alternative explicit State-Action DQN while preserving the original model.

The State-Action implementation now supports batched legal-action
evaluation, DQN training with Prioritized Experience Replay, checkpointing,
and a short end-to-end CPU training smoke test against RandomAgent.

The next step is to validate this architecture beyond the smoke-test level.

Before substantially longer training experiments, the project must decide
whether explicit CUDA/device support should be added so that training can
efficiently use available GPU hardware.

The original DQN remains available while the State-Action architecture is
evaluated.