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
- Epsilon decay after successful training updates.
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
- trains against a frozen copy of the learner policy,
- alternates the learner between White and Black,
- periodically refreshes the frozen opponent,
- keeps frozen-opponent refresh independent from target-network synchronization,
- periodically evaluates the learner against RandomAgent.

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

- Run and inspect real multi-episode self-play training from the main program,
using RandomAgent as the provisional stable benchmark.

- Use the resulting training and evaluation metrics to validate the workflow
before introducing more advanced opponent-selection mechanisms.

- A champion-vs-challenger system and its promotion criterion are intentionally
postponed until the current self-play workflow has been evaluated.