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
- Epsilon decay once per episode when replay training occurred
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
- Truncation diagnostics for claimable draws
- Material-based reward shaping from net material change
- Independent chess-outcome classification using game result and agent color
- Greedy diagnostic evaluation game against RandomAgent
- PGN diagnostic export
- Explicit `-0.1` truncation penalty
- Terminal replay treatment for artificial truncation
- Prioritized Experience Replay
- Importance-sampling weighted loss
- TD-error-based replay-priority updates
- Replay-priority persistence
- Small `-0.0005` non-terminal step penalty

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

The current training reward combines:

The current training reward combines:

- canonical chess terminal reward from the DQN's perspective,
- `0.01 * net material-balance change`,
- `-0.0005` on ordinary non-terminal, non-truncated learner transitions,
- an additional `-0.1` penalty on artificial truncation.

The current training path uses Prioritized Experience Replay with:

- `alpha = 0.6`
- `beta = 0.4`
- priority epsilon `1e-6`

Replay sampling is performed with replacement and sampled priorities are
updated from absolute TD error.

The material change is measured across the complete DQN transition,
including the opponent response.

Artificial truncation remains distinct from real chess termination, but the
final replay transition is stored as terminal for Bellman learning.

Chess win/draw/loss classification is independent of `total_reward` and is
derived from `final_info["result"]` together with `agent_color`.

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

Determine the next learning-direction change from the accumulated diagnostic
evidence before modifying the implementation again.

Completed experiments have shown:

- truncation penalty alone was insufficient;
- material shaping did not produce reliable greedy chess play;
- a DQN-vs-Double-DQN diagnostic showed no target difference in the analysed
  replay;
- Prioritized Experience Replay did not clearly improve greedy RandomAgent
  performance;
- Q-value diagnostics showed small gaps between many top-ranked legal
  actions during non-progressing play;
- an initial `-0.0005` step-penalty experiment still produced a high
  truncation rate.

Before another training run or learning modification, decide whether the
next experiment should address:

1. reward/learning objective,
2. model architecture/capacity,
3. or the suitability of the current DQN formulation.

Do not simply add another DQN enhancement or extend training without a
specific hypothesis supported by the existing diagnostics.

RandomAgent remains the provisional stable evaluation benchmark.

Champion-vs-challenger remains future work. Do not choose promotion criteria
yet.