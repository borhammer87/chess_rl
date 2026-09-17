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
- Random replay sampling.
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

Real training and diagnostic PGN inspection showed that terminal-only
learning plus a truncation penalty was not sufficient to produce useful
chess behavior. The greedy policy continued to show long non-progressing
move cycles and a very high truncation rate.

The current experiment therefore keeps the `-0.1` artificial-truncation
penalty and adds a small material-based shaping signal:

`0.01 * net material-balance change`

The experiment is being run from scratch so the replay buffer contains only
transitions generated under the new reward semantics.

The main observations are:

- truncation frequency,
- average episode length,
- balanced RandomAgent evaluation,
- greedy diagnostic PGN behavior,
- whether materially sensible play begins to emerge.

No additional reward shaping or per-move penalty should be introduced until
this experiment has been evaluated.