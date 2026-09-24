# ARCHITECTURE

This document describes the architecture implemented at the current repository HEAD.

## Repository layout

- `src/chess_rl/env/` — chess environment.
- `src/chess_rl/utils/` — board/action encoding, legal-action selection/masking and replay memory.
- `src/chess_rl/models/` — original fixed-output CNN and experimental State-Action network.
- `src/chess_rl/agents/` — `RandomAgent`, `DQNAgent` and `StateActionDQNAgent`.
- `src/chess_rl/training/` — episode execution, training orchestration, self-play, summaries, checkpoints and diagnostics.
- `scripts/` — standalone utility script(s).
- `tests/` — automated test suite.

## Chess environment

`ChessEnv` owns a `python-chess` board, exposes legal moves, applies `chess.Move` objects and returns independent board copies. Environment reward is deliberately canonical and color-neutral for the training layer: White win `+1`, Black win `-1`, draw or unfinished game `0`.

The environment uses `board.is_game_over()` without automatically claiming threefold repetition or the fifty-move rule. Artificial training truncation is handled above the environment.

## State representation

`encode_board()` returns a float tensor of shape `(18, 8, 8)`.

- channels 0–5: White pawn, knight, bishop, rook, queen, king;
- channels 6–11: Black pawn, knight, bishop, rook, queen, king;
- channel 12: White kingside castling right;
- channel 13: White queenside castling right;
- channel 14: Black kingside castling right;
- channel 15: Black queenside castling right;
- channel 16: en-passant target square;
- channel 17: side to move, all ones for White and zeros for Black.

The representation is absolute. It is not rotated or recolored when the learner plays Black.

## Action representation

The external action space contains 4272 integer actions.

- 0–4095: `from_square * 64 + to_square` for non-promotion moves.
- 4096–4271: explicit promotion actions for every legal promotion origin/destination geometry and each of queen, rook, bishop and knight.

`decode_action_components()` maps an action ID to `(from_square, to_square, promotion_type)` for the State-Action model. Existing replay transitions therefore use the same action IDs for both model families.

## Original fixed-output DQN

`DQNCNN` implements:

`18×8×8 state → Conv(32) → Conv(64) → Flatten → Linear(512) → 4272 Q-values`

The final layer has no bias. `DQNAgent` owns independent policy and target networks, an Adam optimizer and epsilon-greedy exploration state.

Greedy selection computes the full action vector and restricts the choice to legal encoded actions. Exploration samples only legal moves.

This is the model family used by the current `main()` frozen-opponent self-play workflow.

## Experimental State-Action DQN

`StateActionDQN` replaces the dedicated 4272-output head with an explicit state-action function.

State path:

`18×8×8 → Conv(32) → Conv(64) → Flatten → Linear(256)`

Action path:

- from-square embedding: 16 dimensions;
- to-square embedding: 16 dimensions;
- promotion embedding: 4 dimensions.

The resulting 36 action features are concatenated with the 256 state features. The shared Q-head is:

`292 → 256 → 1`

The network supports:

- scoring multiple legal actions after encoding one state once;
- batched scoring of one selected action per state;
- batched maximum legal-action evaluation for non-terminal next states.

`StateActionDQNAgent` mirrors the policy/target, epsilon-greedy, optimizer, Bellman-update and serialization interfaces needed by the existing RandomAgent training path. Terminal next states receive future value zero and are excluded from next-state legal-action evaluation.

The State-Action architecture is not used by `main()` and is not integrated into the current frozen-opponent self-play helper, whose opponent type is `DQNCNN`.

## Replay and learning

`ReplayBuffer` stores `Transition` objects containing state, action, reward, next state, terminal flag and legal next actions. It also stores one priority per transition.

The main learning path uses Prioritized Experience Replay:

- `alpha = 0.6`;
- `beta = 0.4`;
- priority epsilon `1e-6`;
- new transitions start at the current maximum priority;
- prioritized sampling is with replacement;
- importance-sampling weights are normalized by their maximum;
- priorities are updated from absolute TD errors plus the epsilon.

Uniform `sample()` remains available but is not the main training sampler.

## Episode and reward semantics

`episodes.py` separates low-level environment interaction from complete learner-vs-opponent episodes.

A learner transition in DQN-vs-opponent play spans the learner move and, when the game continues, the opponent response. Rewards stored for learning are from the learner's color perspective.

Learning reward can contain:

1. canonical terminal chess reward converted to learner perspective;
2. material shaping: `0.01 * change in learner-perspective material balance` across the complete learner transition;
3. `STEP_PENALTY = -0.0005` on ordinary non-terminal, non-truncated learner transitions;
4. `TRUNCATION_PENALTY = -0.1` on the final artificially truncated transition.

The step and truncation penalties are not both applied to the final truncated transition. For Bellman learning, an artificially truncated final transition is stored as terminal with no legal next actions, while the chess environment itself remains non-terminal.

Because rewards are shaped, chess outcome is never inferred from the sign of accumulated reward. Outcome summaries use the actual chess result and learner color.

## Training orchestration

`train_dqn.py` provides:

- multi-episode training against `RandomAgent`;
- alternating learner colors;
- target synchronization;
- progress callbacks;
- checkpoint/evaluation scheduling;
- balanced White/Black RandomAgent evaluation;
- normalized evaluation scoring;
- training summaries;
- greedy diagnostic PGN generation;
- the executable `main()` workflow.

`TrainingSummary` and `EvaluationSummary` live in `results.py` so aggregate metrics are represented explicitly rather than recalculated ad hoc by callers.

## Frozen-opponent self-play

`self_play.py` creates an independent frozen `DQNCNN` copied from the original agent's policy network. Its parameters do not require gradients and it selects greedily.

The frozen opponent and Bellman target network are distinct:

- target network: stabilizes Bellman targets;
- frozen opponent: stabilizes experience generation against a changing learner.

`train_against_frozen()` can alternate learner color and independently schedule target synchronization, frozen-opponent refresh, checkpoints and evaluation callbacks.

## Evaluation and model selection

RandomAgent remains the provisional stable benchmark. Evaluation sets epsilon to zero temporarily, does not train the agent and uses a separate temporary replay buffer.

Balanced evaluation combines equal numbers of White and Black games. The score is:

`(wins + 0.5 * draws) / episodes`

Losses and truncations contribute zero points.

`main()` uses this score to retain `checkpoints/best.pt` when a strictly better score is observed.

## Persistence

Both DQN agent classes expose compatible `state_dict()` / `load_state_dict()` structures containing policy network, target network, optimizer and epsilon.

`ReplayBuffer.state_dict()` stores capacity, transitions and priorities and can load older replay states without priorities by assigning default priority `1.0`.

`checkpoint.py` composes agent state and replay-buffer state into a training checkpoint and optionally stores metadata. Current checkpointing does not include RNG state or a global lifetime episode counter.
