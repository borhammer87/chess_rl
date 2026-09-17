# CURRENT STATE

## Status

The project currently contains a complete DQN training, evaluation,
checkpoint-selection, frozen-opponent self-play, and diagnostic workflow.

The main training workflow uses frozen-policy self-play.

RandomAgent remains the provisional stable evaluation benchmark.

The DQN can train and evaluate as both White and Black.

Implemented components:

- Chess environment
- Board encoder
- Action encoder
- Legal action masking
- DQN CNN
- Replay Buffer
- DQN vs RandomAgent training
- Multi-episode training
- Replay sampling
- Target network synchronization
- Epsilon decay once per episode when replay training occurred
- Training metrics collection
- Training summary generation
- Console progress reporting
- Agent-perspective rewards
- Training as White and Black
- Alternating colors between training episodes
- Evaluation against RandomAgent as White and Black
- Balanced two-color evaluation
- Periodic evaluation during training
- Evaluation scoring
- Periodic checkpointing
- Training checkpoint loading
- Replay-buffer persistence
- Best-checkpoint selection
- Extended 18-channel board representation
- Side-to-move encoding
- Castling-rights encoding
- En passant target-square encoding
- Explicit queen, rook, bishop, and knight promotion actions
- 4272-action DQN output space
- Generic opponent episode engine
- Frozen DQN self-play opponent
- Greedy frozen-opponent move selection
- Multi-episode self-play training
- Alternating White/Black self-play episodes
- Periodic frozen-opponent synchronization
- Truncation diagnostics for claimable threefold repetition
- Truncation diagnostics for claimable fifty-move draws
- Truncation classification without claimable draw
- Explicit truncation penalty for the final replay transition
- Material-based reward shaping from net material change
- Separation between chess outcome classification and shaped training reward
- Truncated replay transitions treated as terminal for Bellman targets
- Greedy diagnostic evaluation game against RandomAgent
- PGN export of the diagnostic evaluation game



## State representation

The board encoder returns a tensor with shape:

`(18, 8, 8)`

Channels:

- 0–5: White pieces
- 6–11: Black pieces
- 12: White kingside castling right
- 13: White queenside castling right
- 14: Black kingside castling right
- 15: Black queenside castling right
- 16: En passant target square
- 17: Side to move

The representation remains absolute: the board is not rotated when the
DQN plays Black.

## Action representation

The DQN uses a fixed action space of 4272 actions.

- Actions 0–4095 preserve the original `from_square * 64 + to_square`
  encoding for non-promotion moves.
- Actions 4096–4271 represent explicit promotion actions.
- Queen, rook, bishop, and knight promotions have distinct action indices.

The agent can therefore learn underpromotions rather than implicitly
defaulting every promotion to a queen.

## Training package structure

Training responsibilities are separated into focused modules:

- `results.py` — training and evaluation result data structures.
- `episodes.py` — step, generic opponent episode, reward-perspective,
  and replay-training operations.
- `checkpoint.py` — training-state persistence and checkpoint metadata.
- `train_dqn.py` — multi-episode workflow, color alternation, evaluation
  scheduling, model selection, summaries, and main program execution.
- `self_play.py` — frozen DQN opponents, self-play episode wrappers,
  multi-episode self-play, and opponent-network synchronization.

## Color and reward semantics

`ChessEnv` keeps a canonical White-perspective terminal reward:

- White win: `+1`
- Black win: `-1`
- Draw or unfinished game: `0`

The training layer converts this terminal reward to the DQN agent's
perspective.

Training transitions additionally receive material-based reward shaping
from the DQN's perspective:

`0.01 * net material-balance change`

using piece values:

- pawn: 1
- knight: 3
- bishop: 3
- rook: 5
- queen: 9

Artificial truncation adds a `-0.1` penalty to the final replay transition.

These components are additive.

Therefore, `total_reward` is now a learning signal and must not be used to
infer whether the DQN won, drew, or lost the chess game.

Chess outcomes are classified independently using the actual chess result
in `final_info["result"]` together with `agent_color`.

A truncated chess game remains non-terminal from the environment's
perspective, but its final replay transition is stored as terminal for
Bellman learning.

## Current training workflow

Running the training module:

1. Creates the environment, DQN learner, RandomAgent benchmark, and replay buffer.
2. Reads the previous best evaluation score from `best.pt` when available.
3. Loads `latest.pt` when available to resume training.
4. Creates a frozen training opponent from the restored learner `policy_net`.
5. Runs multi-episode self-play with `train_against_frozen()`.
6. Alternates the DQN learner between White and Black.
7. Stores rewards from the DQN's perspective.
8. Adds scaled net material change to the replay reward.
9. Reports progress during training.
10. Synchronizes the target network every 10 episodes.
11. Saves `latest.pt` every 25 episodes.
12. Evaluates the current learner against RandomAgent every 25 episodes.
13. Evaluates equally as White and Black.
14. Calculates a normalized evaluation score.
15. Replaces `best.pt` only when the new score is strictly better.
16. Refreshes the frozen self-play opponent every 25 episodes.
17. Prints an aggregated training summary using the existing `TrainingSummary` infrastructure.
18. Classifies truncated episodes according to whether a draw could have
    been claimed by threefold repetition or the fifty-move rule.
19. Applies a `-0.1` penalty to the final replay transition when an episode
    reaches the artificial training horizon, and stores that transition as
    terminal for DQN learning.
20. Runs one additional greedy diagnostic game against RandomAgent after
    training.
21. Saves that diagnostic game to `checkpoints/evaluation_game.pgn`.

RandomAgent remains the provisional stable evaluation benchmark. It is used
for periodic evaluation and best-checkpoint selection, but it is no longer
the opponent used by the main training workflow.

Frozen-opponent synchronization is independent from target-network
synchronization. They currently use different frequencies and serve
different purposes.

- Training outcomes are classified from the actual chess result and agent
  color, independently of `total_reward`.

## Current evaluation metrics

- Wins
- Draws
- Losses
- Truncated games
- Normalized evaluation score

Evaluation score is:

`(wins + 0.5 * draws) / episodes`

Periodic model-selection evaluation currently uses:

- 10 games as White
- 10 games as Black

## Current checkpoint roles

- `checkpoints/latest.pt` — most recent resumable training state.
- `checkpoints/best.pt` — best balanced evaluation score observed so far.

Training checkpoints can preserve:

- Policy network
- Target network
- Optimizer
- Epsilon
- Replay-buffer capacity
- Replay-buffer transitions
- Optional checkpoint metadata

## Tests

The repository currently defines 170 tests covering the environment,
encoding, DQN agent, replay buffer, generic episode execution,
RandomAgent training, frozen-opponent self-play, evaluation,
checkpointing, reward perspective, and color alternation.

## Current limitations

- The current policy has not yet demonstrated useful chess-playing
  behaviour.
- Training and RandomAgent evaluation still produce a very high proportion
  of truncated games.
- Diagnostic runs showed that most truncations occur without a claimable
  threefold repetition or fifty-move draw.
- Greedy diagnostic evaluation can enter long non-progressing move cycles
  even with epsilon set to zero.
- The learning signal remains mostly sparse: normal chess outcomes use
  win `+1`, loss `-1`, and draw `0`, while artificial truncation adds a
  `-0.1` terminal replay penalty.
- The `-0.1` truncation penalty is an initial experimental value and has not
  yet been validated as optimal.
- Board encoding does not include repetition state or move counters.
- Self-play agents are not yet evaluated against frozen or historical
  policies.
- Board encoding remains absolute rather than agent-relative.
- Checkpoints do not preserve random-number-generator state.
- Checkpoints do not store a global lifetime episode counter.
- Evaluation currently uses only RandomAgent as the stable benchmark.
- Evaluation scores may be noisy because they use a finite number of games.


## Current focus

The integrated frozen-opponent self-play workflow has now been exercised
with real training runs.

Truncation diagnostics and a greedy PGN evaluation were added to investigate
the high number of unfinished games.

The diagnostics indicate that claimable draw rules are not the main cause
of truncation.

The current greedy policy can make non-progressing move sequences even when
exploration is disabled, indicating that useful chess behaviour has not yet
emerged reliably.

## Next milestone

Validate the combined learning signal in a fresh real training run.

The current experiment uses:

- chess terminal reward: `+1 / 0 / -1`
- material shaping: `0.01 * net material-balance change`
- artificial truncation penalty: `-0.1`

The run must start from a fresh learner and empty replay buffer because
previous checkpoints contain experience generated under different reward
semantics.

Observe:

1. training win/draw/loss/truncation counts,
2. truncation frequency,
3. average episode length,
4. balanced RandomAgent evaluation every 25 episodes,
5. greedy diagnostic PGN behavior,
6. whether the policy begins to prefer materially sensible actions,
7. whether material shaping reduces the previously observed non-progressing
   move cycles.

Do not introduce additional reward components during this experiment.

In particular, do not add a per-move living penalty or change the material
scale or truncation penalty until the current combined reward has been
evaluated.