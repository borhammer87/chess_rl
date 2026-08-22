# Chess RL

Chess reinforcement-learning project using a CNN-based DQN.

## Current capabilities

The project currently implements a complete DQN training, evaluation, and
checkpoint workflow.

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
- Greedy evaluation against RandomAgent.
- Periodic evaluation during training.
- Periodic training checkpoints.
- Automatic checkpoint loading.
- Replay-buffer persistence.

## Running training

Run:

`python -m chess_rl.training.train_dqn`

The program uses:

`checkpoints/latest.pt`

as the current resumable training checkpoint.

If the checkpoint exists, the agent and replay buffer are restored before
training continues.

During training, progress is reported to the console and the latest
training state is saved periodically.

## Current limitations

- The DQN currently plays only White.
- RandomAgent always plays Black.
- Self-play is not implemented yet.

## Next goal

Use periodic evaluation results to identify and retain the best-performing checkpoint separately from the latest resumable training state.