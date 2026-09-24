# NEXT CHAT PROMPT

We are continuing the Chess Reinforcement Learning project.

## Source-of-truth rule

The attached ZIP repository is the primary and authoritative source of truth for the current code.

Before proposing any modification:

1. extract the ZIP;
2. inspect the complete repository tree;
3. read **all** project Markdown files;
4. read all source files and tests relevant to the requested change;
5. reconstruct the current state from code and tests before using prior conversation context.

If repository contents contradict prior chat context, trust the repository. Use prior project context only to recover rationale or chronology that the repository itself does not preserve, never to claim that unimplemented code exists.

If the ZIP cannot actually be inspected, state that limitation and do not guess.

Follow `PROJECT_RULES.md`. Prefer the smallest correct change and update tests with code.

## Repository snapshot at the documentation audit

`pyproject.toml` reports version `0.9.0`.

The repository contains two DQN approaches in parallel.

### Original DQNCNN path

Implemented:

- `ChessEnv` using `python-chess`;
- absolute 18 × 8 × 8 board representation;
- 4272-action encoding with explicit underpromotions;
- legal-action filtering/masking;
- `DQNCNN` and `DQNAgent` policy/target networks;
- replay buffer and Prioritized Experience Replay;
- agent-perspective reward handling;
- material shaping, ordinary step penalty and artificial-truncation penalty;
- DQN-vs-RandomAgent training;
- alternating White/Black training;
- frozen-policy self-play;
- independent target and opponent synchronization;
- balanced RandomAgent evaluation;
- checkpoint resume, replay persistence and best-checkpoint selection;
- training summaries, truncation diagnostics and diagnostic PGN export.

The executable `main()` uses this original DQNCNN path for frozen-opponent self-play. Its current hard-coded schedule is 100 episodes, 150 learner steps, batch size 32, minimum replay size 1000, target synchronization every 10 episodes, and checkpoint/evaluation/frozen-opponent refresh every 25 episodes. RandomAgent evaluation uses 10 games per color.

### Experimental State-Action path

Implemented in parallel:

- `StateActionDQN`;
- 256-dimensional CNN state features;
- from-square, to-square and promotion embeddings;
- shared `Q(s, a)` head;
- vectorized legal-action scoring without re-encoding one state per action;
- batched selected state-action evaluation;
- batched legal-next-action maxima;
- `StateActionDQNAgent` with policy/target networks, epsilon-greedy selection, Bellman updates, PER weights, TD-error reporting and serialization;
- short end-to-end CPU training smoke test through the RandomAgent training workflow.

Do **not** infer from that smoke test that State-Action learns better chess. It is not integrated into frozen-opponent self-play or `main()` and no long-run/comparative strength result is established.

## Important limitations

- No reliable playing strength has been demonstrated in the repository.
- Historical repository diagnostics report high truncation and non-progressing greedy behavior for the original DQN.
- RandomAgent is a provisional weak benchmark.
- Board encoding omits repetition history and move counters and remains absolute.
- Checkpoints omit RNG state and a lifetime episode counter.
- Explicit device/CUDA management is not implemented.
- `pyproject.toml` does not declare runtime dependencies.

## Test caution

At the documentation-audit session the repository contained 246 pytest test functions. A fresh test run could not be collected in that sandbox because `python-chess` was missing and `chess_rl` was not installed/importable there. Therefore do not inherit a claim that all tests passed from this prompt. Run them in the actual project environment before treating the suite as green.

## Next-step rule

Do not assume CUDA, longer training, State-Action self-play, champion-vs-challenger, or any other feature is automatically next. Inspect the new ZIP and the user's current objective, then justify the smallest next step from the repository evidence.
