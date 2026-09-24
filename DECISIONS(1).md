# DECISIONS.md

## Purpose

Chronological architectural decision record. Older decisions remain here when later decisions supersede them so that the evolution of the project stays understandable. `CURRENT_STATE.md` is the authority for the present HEAD snapshot.

## D-001 — Use a CNN-based DQN

### Status
Accepted.

### Decision
Use a convolutional Deep Q-Network as the original learning architecture for board-state evaluation and discrete chess actions.

### Motivation
The project is educational and intentionally develops the DQN pipeline component by component while using a spatial model suitable for board tensors.

### Alternatives considered
Tabular Q-learning, non-convolutional networks and policy-gradient-first designs.

### Consequences
The project requires a fixed state tensor, action encoding, replay memory, policy/target networks and legal-action handling.

## D-002 — Board representation uses 12 piece channels

### Status
Superseded by D-016.

### Decision
Initially encode only the six White and six Black piece types as 12 planes.

### Motivation
Start with the smallest clear board representation while building the pipeline incrementally.

### Alternatives considered
Richer chess-state features from the start.

### Consequences
The initial representation omitted turn, castling and en-passant state. D-016 later expanded it to 18 channels.

## D-003 — Fixed action space of 4096 origin/destination actions

### Status
Superseded by D-017.

### Decision
Initially encode actions as `from_square * 64 + to_square`.

### Motivation
A simple fixed output space made the first DQN and legal masking straightforward.

### Alternatives considered
Move lists of variable length and structured action models.

### Consequences
Moves sharing origin/destination, especially different promotion pieces, could not be distinguished. D-017 later expanded the action space.

## D-004 — Automatically prefer queen promotion

### Status
Superseded by D-017.

### Decision
Under the original 4096-action representation, resolve ambiguous promotion actions as queen promotion.

### Motivation
The original action ID could not encode promotion type.

### Alternatives considered
Expanding the action space immediately or introducing a separate promotion head.

### Consequences
Underpromotion could not be learned. D-017 removed the need for this fallback by representing promotion types explicitly.

## D-005 — Apply legal action masking after inference

### Status
Accepted.

### Decision
Restrict greedy fixed-output DQN selection to actions legal in the current chess position rather than asking the network to learn legality from reward.

### Motivation
Chess legality is known exactly and should not consume learning capacity.

### Alternatives considered
Penalizing illegal actions or changing the network output dynamically.

### Consequences
The network may output arbitrary values for illegal actions, but they cannot be selected greedily.

## D-006 — Exploration samples only legal moves

### Status
Accepted.

### Decision
Epsilon exploration chooses among legal chess moves only.

### Motivation
Exploration should investigate valid game behavior rather than waste transitions on impossible actions.

### Alternatives considered
Uniform sampling across the full action space followed by rejection/penalty.

### Consequences
Every exploratory action can be executed by the environment.

## D-007 — ChessEnv accepts `chess.Move` objects

### Status
Accepted.

### Decision
Keep action IDs outside the environment; `ChessEnv.step()` receives legal `chess.Move` objects.

### Motivation
Separate chess rules/environment concerns from neural-network action representation.

### Alternatives considered
Making the environment decode integer DQN actions.

### Consequences
Training/action-selection code is responsible for encoding and decoding actions.

## D-008 — ChessEnv returns independent board copies

### Status
Accepted.

### Decision
`reset()`, `get_state()` and `step()` expose copied board states rather than the mutable internal board object.

### Motivation
Prevent accidental external mutation of environment state.

### Alternatives considered
Returning the internal board directly.

### Consequences
Callers can inspect returned boards safely at a small copying cost.

## D-009 — ReplayBuffer uses bounded deque storage

### Status
Accepted.

### Decision
Store replay transitions in a bounded `deque`, later paired with a bounded priority deque.

### Motivation
Automatic FIFO eviction gives simple fixed-capacity replay memory.

### Alternatives considered
Manual ring buffers or unbounded lists.

### Consequences
Old experiences are discarded when capacity is reached. PER priorities must stay index-aligned with transitions.

## D-010 — Use policy and target networks

### Status
Accepted.

### Decision
Maintain a trainable policy network and a separately synchronized target network.

### Motivation
Reduce instability from using the same rapidly changing network for both current estimates and Bellman targets.

### Alternatives considered
A single network for both roles.

### Consequences
Training orchestration must schedule target synchronization independently of other updates.

## D-011 — Environment reward is terminal and from White's perspective

### Status
Accepted for the environment; extended by D-015 and later shaping decisions in the training layer.

### Decision
`ChessEnv` returns `+1` for a White win, `-1` for a Black win and `0` for draws or unfinished games.

### Motivation
Keep the environment simple and color-neutral from the trainer's point of view.

### Alternatives considered
Agent-relative environment rewards or dense environment shaping.

### Consequences
Training code must convert reward to learner perspective. Dense shaping belongs above the environment.

## D-012 — Build training incrementally

### Status
Accepted.

### Decision
Add training capabilities in small tested steps rather than implementing a large subsystem at once.

### Motivation
The project prioritizes understanding, testability and architectural clarity.

### Alternatives considered
A single end-to-end implementation pass.

### Consequences
Training responsibilities are split into episode operations, orchestration, results, self-play and persistence modules.

## D-013 — Separate checkpoint state ownership from checkpoint scheduling

### Status
Accepted.

### Decision
Agents/replay objects expose serializable state, `checkpoint.py` composes/restores complete training checkpoints, and training loops decide when checkpoint callbacks fire.

### Motivation
Persistence format and scheduling are different responsibilities.

### Alternatives considered
Hard-code checkpoint timing inside agents or checkpoint helpers.

### Consequences
The same checkpoint helpers can be reused by different training workflows and callbacks.

## D-014 — Select the best checkpoint using normalized chess score

### Status
Accepted.

### Decision
Use balanced evaluation score `(wins + 0.5 * draws) / episodes` and replace `best.pt` only on strict improvement.

### Motivation
Model selection should use chess results, not shaped training reward or loss.

### Alternatives considered
Average reward, training loss, win rate only, or always replacing the best checkpoint.

### Consequences
Truncations and losses score zero; finite evaluation sets can still be noisy. RandomAgent remains a provisional benchmark rather than a final strength measure.

## D-015 — Use agent-perspective rewards with absolute board encoding

### Status
Accepted.

### Decision
Keep board tensors in a fixed absolute orientation while converting White-perspective environment reward to the learner's color perspective.

### Motivation
Support one learner playing either color without changing the established board encoding.

### Alternatives considered
Agent-relative board rotation/recoloring or separate agents by color.

### Consequences
The network must learn color-dependent behavior from absolute features, but training and evaluation can alternate White and Black.

## D-016 — Expand board state representation to 18 channels

### Status
Accepted; supersedes D-002.

### Decision
Keep the 12 piece planes and add White/Black kingside and queenside castling rights, en-passant target and side to move.

### Motivation
Piece placement alone does not uniquely determine legal chess state.

### Alternatives considered
Keep 12 channels or introduce a larger history-based representation immediately.

### Consequences
The model input is 18 × 8 × 8. Older 12-channel model/replay data is architecture-incompatible. Repetition history and move counters remain absent.

## D-017 — Expand action space to represent promotions explicitly

### Status
Accepted; supersedes D-003 and D-004.

### Decision
Preserve actions 0–4095 for non-promotions and add 176 explicit promotion actions, producing `ACTION_SIZE = 4272`.

### Motivation
The original action representation could not distinguish promotion piece type and therefore could not learn underpromotion.

### Alternatives considered
Keep automatic queen promotion or redesign the entire action representation.

### Consequences
All four standard promotion choices are representable. The original DQNCNN output layer changed to 4272 values and older 4096-output checkpoints are incompatible.

## D-018 — Use a periodically updated frozen policy for self-play

### Status
Accepted for the original DQNCNN workflow.

### Decision
Create an independent frozen copy of the learner's policy network as the opponent, select its moves greedily, and refresh it periodically on a schedule independent of target-network synchronization.

### Motivation
A frozen adversary changes more slowly than the live learner and serves a different purpose from the Bellman target network.

### Alternatives considered
Play against the live policy, reuse the target network as opponent, or maintain a second trainable agent.

### Consequences
Opponent and target update frequencies can be tuned independently. Historical pools and champion/challenger promotion remain possible future extensions.

## D-019 — Keep current chess draw termination semantics after truncation diagnosis

### Status
Accepted.

### Decision
Do not automatically end games merely because threefold repetition or the fifty-move rule is claimable. Record those claimable states when the artificial training horizon is reached and retain greedy PGN diagnostics.

### Motivation
Repository-recorded diagnostics found that claimable draws explained only a minority of truncations; non-progressing policy behavior remained the larger issue.

### Alternatives considered
Automatically claim draws, increase the horizon, or treat all truncations as equivalent.

### Consequences
Environment semantics stay close to the existing `python-chess` game-over behavior, while training diagnostics distinguish draw-related from other truncations.

## D-020 — Penalize artificial training truncation in replay

### Status
Accepted experimentally.

### Decision
When `max_agent_steps` is reached before real chess termination, add `TRUNCATION_PENALTY = -0.1` to the final learner transition and store that replay transition as terminal with no legal next actions.

### Motivation
Previously, artificial horizon termination supplied no direct negative signal and still allowed Bellman bootstrapping beyond the generated trajectory.

### Alternatives considered
Zero truncation reward, a per-move penalty, material shaping, or simply increasing the horizon.

### Consequences
Chess environment termination remains unchanged, reporting still distinguishes truncation from loss, and the `-0.1` value remains experimental.

## D-021 — Add small material-based reward shaping

### Status
Accepted experimentally.

### Decision
Add `0.01 *` the learner-perspective change in conventional material balance across a complete learner transition, while preserving canonical terminal reward and the separate truncation penalty.

### Motivation
The repository-recorded truncation-penalty experiment did not provide enough learning signal to establish reliable play.

### Alternatives considered
Stronger terminal/truncation rewards, other dense chess heuristics, or no shaping.

### Consequences
Training reward no longer equals chess outcome. Outcome classification must use the actual game result. Replay generated under different reward semantics should not be mixed casually.

## D-022 — Use Prioritized Experience Replay

### Status
Accepted experimentally and implemented.

### Decision
Use PER in the main replay-training path with `alpha=0.6`, `beta=0.4` and priority epsilon `1e-6`. Sample with replacement, apply normalized importance-sampling weights and update priorities from absolute TD error.

### Motivation
Repository diagnostics found rare experiences with substantially larger TD errors, while a diagnostic standard-DQN vs Double-DQN target comparison did not show a practical difference in the analysed sample.

### Alternatives considered
Keep uniform replay, implement Double DQN next, enlarge the network, or change several mechanisms at once.

### Consequences
Replay persistence includes priorities and agents expose TD errors for updates. The documented initial PER experiment did not establish a clear greedy RandomAgent improvement, so PER is retained without claiming it solved policy quality.

## D-023 — Preserve the original DQN while developing an explicit State-Action DQN

### Status
Accepted experimentally.

### Decision
Add `StateActionDQN` and `StateActionDQNAgent` alongside, not instead of, the original `DQNCNN`/`DQNAgent`.

Represent each action through from-square, to-square and promotion type, reuse the existing external action IDs, and evaluate `Q(s, a)` through a shared head. Encode each state once when scoring multiple candidate actions and batch state/action evaluation where possible. Terminal next states receive future value zero and are excluded from legal-next-action max evaluation.

Keep PER compatibility and the existing checkpoint state interface. Validate integration first with a short CPU end-to-end RandomAgent training smoke test.

### Motivation
Original-DQN diagnostics suggested that the fixed 4272-output formulation might not discriminate useful legal actions well enough. An explicit state-action function is a controlled architectural alternative while preserving a known baseline and the rest of the training infrastructure.

### Alternatives considered
Replace the original DQN immediately, keep modifying only reward/PER, enlarge the fixed-output network, or redesign the entire training stack simultaneously.

### Consequences
Two model families now coexist. The State-Action model has pipeline-level CPU smoke-test coverage but no demonstrated long-run or comparative strength. It is not yet integrated into frozen-opponent self-play or `main()`. Any future promotion to the primary architecture requires evidence rather than assumption.

## Current unresolved decisions

The repository does not yet decide:

- whether State-Action should replace or remain alongside the original DQNCNN;
- whether/when State-Action should enter frozen-opponent self-play;
- whether explicit device/CUDA support should precede larger experiments;
- what stronger evaluation or champion-vs-challenger promotion criterion should eventually be used.

These should be decided from future repository evidence, not inherited as already-set next steps.
