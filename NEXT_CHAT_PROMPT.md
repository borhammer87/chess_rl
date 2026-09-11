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
The project contains a complete DQN training, evaluation, checkpoint, and
frozen-opponent self-play workflow.

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
- DQN training as White and Black
- Alternating White/Black training episodes
- Balanced White/Black evaluation
- Frozen-policy self-play
- Greedy frozen-opponent action selection
- Periodic frozen-opponent synchronization
- Main-program self-play integration

The main executable training workflow now trains the DQN learner against
an independent frozen copy of its `policy_net`.

The learner alternates between White and Black.

The frozen opponent plays greedily and is periodically refreshed from the
current learner policy.

When resuming training, `latest.pt` is loaded before the frozen opponent
is created. This ensures that the frozen opponent starts from the restored
learner policy rather than newly initialized weights.

RandomAgent remains the provisional stable evaluation benchmark.

It is used for periodic balanced White/Black evaluation and for deciding
whether `best.pt` should be replaced. It is no longer the opponent used
by the main training workflow.

The current main-program configuration uses:

- 100 training episodes per execution
- maximum 150 learner steps per episode
- batch size 32
- minimum replay size 1000
- target-network synchronization every 10 episodes
- checkpoint saving every 25 episodes
- RandomAgent evaluation every 25 episodes
- frozen-opponent synchronization every 25 episodes
- 10 evaluation games per color

Training checkpoints preserve:

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions

Rewards stored in replay memory are expressed from the learner's
perspective.

Board encoding remains absolute rather than agent-relative.

The frozen opponent and the target network serve different purposes even
though both are periodically copied from `policy_net`.

The target network stabilizes Bellman targets.

The frozen opponent provides a temporarily stable adversary during
self-play.

Their update schedules are independent.
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

Run the integrated multi-episode self-play workflow from the main program
and inspect the resulting training and RandomAgent evaluation behaviour.

The immediate objective is to validate the current workflow before adding
more opponent-management architecture.

Use RandomAgent as the provisional stable benchmark.

Do not implement champion-vs-challenger promotion yet.

The future champion-vs-challenger system should remain modular, and its
promotion criterion must be explicitly designed before implementation.

Do not change the current frozen-opponent architecture unless repository
evidence or training results provide a concrete reason to do so.

Checkpoints created before v0.8.0 are incompatible with the current
network architecture because both the CNN input shape and action output
size changed.