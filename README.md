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

## Next goal

Real training has now tested truncation penalties, material-based reward
shaping, Prioritized Experience Replay, and a small non-terminal step
penalty.

The learned greedy policy still shows a high truncation rate and can enter
long non-progressing move cycles.

Q-value diagnostics show that legal actions are not globally assigned the
same value, but the highest-ranked actions are often separated by small
Q-value gaps.

The next goal is therefore to determine whether further progress should come
from reward design, network architecture, or a more fundamental change to
the current DQN learning formulation before running substantially longer
training experiments.