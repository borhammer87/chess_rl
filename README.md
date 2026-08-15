## Current capabilities

The project currently implements a complete DQN training and evaluation
pipeline.

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
- Evaluation against RandomAgent.

The next goal is adding model persistence and longer training workflows.