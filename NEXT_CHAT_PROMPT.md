We are continuing a Chess Reinforcement Learning project.

IMPORTANT

The attached ZIP repository is the ONLY source of truth.

Do NOT rely on previous conversations.

------------------------------------------------------------
FIRST TASK
------------------------------------------------------------

Before proposing any modification:

1. Extract the ZIP repository.

2. Inspect the repository structure.

3. Read ALL project markdown files.

4. Read every source file relevant to the requested change.

5. Determine the current project status from the repository itself.

Only then propose the next development step.

If you cannot inspect the ZIP contents, explicitly state that limitation
instead of making assumptions.

------------------------------------------------------------
CURRENT PROJECT STATUS
------------------------------------------------------------

The project contains a complete DQN training, evaluation, and resumable
checkpoint workflow.

Implemented:

- Chess environment
- Board encoder
- Action encoder
- Legal action masking
- Replay Buffer
- DQN CNN
- DQN vs RandomAgent
- Replay sampling
- Multi-episode training
- Target network synchronization
- Epsilon decay after successful training updates
- Episode metrics
- Training summary generation
- Console progress reporting
- Evaluation against RandomAgent
- Periodic checkpointing
- Automatic checkpoint loading
- Replay-buffer persistence
- Evaluation scoring
- Best-checkpoint selection
- Checkpoint metadata

Training checkpoints currently preserve:

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions
- Agent-perspective rewards
- DQN training as White and Black
- Alternating White/Black training episodes
- Balanced White/Black evaluation

The DQN currently alternates between White and Black against RandomAgent.

Rewards stored in replay memory are expressed from the DQN's perspective.

Board encoding remains absolute rather than agent-relative.
------------------------------------------------------------
WORKFLOW
------------------------------------------------------------

Follow the development rules defined in PROJECT_RULES.md.

Prefer the smallest correct modification.

Always update tests together with code.

Do not rewrite large sections without architectural justification.

------------------------------------------------------------
NEXT OBJECTIVE
------------------------------------------------------------

The next recommended task is:

Design the first self-play training workflow.

Before implementing it, determine how the opponent side should be
controlled and updated while experience is generated.

Reuse the existing color, reward, replay, evaluation, and checkpoint
infrastructure instead of duplicating game logic.

Checkpoints created before v0.8.0 are incompatible with the current
network architecture because both the CNN input shape and action output
size changed.