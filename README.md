## Current capabilities

The project currently implements a complete DQN training pipeline.

Implemented features:

- Chess environment based on python-chess.
- Board encoding using tensors.
- CNN-based DQN.
- Replay Buffer.
- Random replay sampling.
- Multi-episode training.
- Target network synchronization.
- Epsilon decay after successful training updates.
- Training metrics collection.
- Aggregated training summaries.

The next goal is to execute complete training sessions, monitor their
progress and begin evaluating the learned policy.