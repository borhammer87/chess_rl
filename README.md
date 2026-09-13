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
- Epsilon decay once per episode when     replay training occurred
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
- keeps frozen-opponent refresh independent from target-network synchronization,
- periodically evaluates the learner against RandomAgent.
- Truncation diagnostics for claimable draws.
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

## Next goal

Real training runs and diagnostic PGN inspection have shown that the current
DQN policy does not yet produce reliable chess behaviour.

Most truncated games are not caused by claimable draw rules, and the greedy
policy can enter long non-progressing move sequences even without
exploration.

The next goal is to design the smallest useful improvement to the learning
signal.

Current candidates are:

- penalizing episodes that reach the truncation limit,
- introducing minimal material-based reward shaping,
- or keeping the sparse terminal reward while training substantially longer.

No reward-model change has been selected yet.

Champion-vs-challenger opponent management remains intentionally postponed.