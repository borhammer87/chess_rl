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

## Development workflow with the user

The user implements code changes locally.

Do **not** normally generate replacement source files for download and do not make broad changes on the user's behalf.

For each development step:

1. inspect the relevant current source and tests first;
2. explain briefly what should change and why;
3. propose the smallest correct modification;
4. tell the user exactly which file to open;
5. identify the existing code/location to find;
6. provide the exact code to add, remove or replace;
7. have the user make the change locally;
8. provide the exact test or command to run;
9. wait for the user's actual result before assuming the change works or moving to the next development step.
10. after the user modifies the local repository, do not propose further
code changes from the previously inspected repository snapshot if the next
step depends on the modified code;
11. ask the user for a fresh ZIP of the current repository;
12. inspect that fresh ZIP before continuing development.
13. actively track whether accumulated development changes make the project
Markdown documentation materially stale;

14. do not ask the user to update documentation after every small intermediate
change. Prefer documenting coherent milestones rather than creating
documentation churn;

15. explicitly remind the user when the accumulated changes are significant
enough that the relevant Markdown files should be updated. This should normally
happen when a development step or experiment establishes a new capability,
changes the documented architecture or workflow, resolves an open question,
changes the known project status, or makes existing documentation materially
inaccurate;

16. when recommending a documentation update, identify which Markdown files
actually need changes and why. Do not modify unrelated documentation merely for
consistency or completeness;

17. actively watch for signs that conversation context may be degrading. Signs
include uncertainty about code that was previously inspected, confusing old
and new repository states, contradicting established project decisions,
forgetting changes made during the current development sequence, or relying
increasingly on assumptions instead of repository evidence;

18. if there is a meaningful risk that context has degraded, tell the user
explicitly before proposing further code changes. Recommend starting a fresh
chat when appropriate, using an updated `NEXT_CHAT_PROMPT.md` and a fresh ZIP
of the repository. Do not continue confidently from potentially corrupted or
mixed context.

A previously inspected ZIP becomes stale as soon as the user changes the
local repository. Test output or pasted console output may be used to analyse
the result of a change, but it does not replace inspection of the updated
repository before proposing the next code modification.

Do not claim that tests pass unless they have actually been run successfully in an appropriate environment.

Do not bundle unrelated refactors or cleanup into a functional change.

The user is deliberately learning through this project, so explain important concepts and the reason for changes rather than only providing code.

Be technically critical. Do not agree with the user automatically. If the user's assumption, proposed implementation or interpretation is incorrect, incomplete or technically weaker than another option, say so clearly and explain why. Likewise, do not invent objections merely to be contrarian: disagreement should be based on repository evidence or sound technical reasoning.

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

## Current development focus

The current focus is validating the experimental State-Action DQN before deciding how far to integrate it.

Repository inspection established that the State-Action agent already plugs into the existing RandomAgent training workflow. Therefore a new training system should **not** be implemented.

The next intended experiment is a small real CPU training probe that reuses the existing infrastructure:

1. greedy evaluation against RandomAgent before training, using both colors;
2. short State-Action training against RandomAgent with alternating colors;
3. existing training-summary infrastructure;
4. wall-clock timing of the training;
5. greedy evaluation against RandomAgent after training, again using both colors.

The purpose of the first probe is primarily to validate sustained training and measure its CPU cost. A change in RandomAgent score after only a handful of episodes must not be treated as evidence that the network has learned useful chess.

A provisional first probe discussed was approximately:

- 10 training episodes;
- 150 maximum learner steps per episode;
- batch size 32;
- minimum replay size around 100 so actual updates occur during the short probe;
- target synchronization around every 10 episodes;
- alternating White/Black;
- small balanced greedy evaluations before and after training.

These are experimental parameters, not permanent architecture decisions. Reinspect the repository and justify them before implementation.

Do not modify the stable original `main()` merely to run this experiment if a smaller isolated experimental entry point is sufficient.

After measuring actual CPU training cost, decide from evidence whether a larger experiment (for example 100 episodes) is reasonable on CPU or whether explicit device/CUDA support should be implemented first.

## Important limitations

- No reliable playing strength has been demonstrated in the repository.
- Historical repository diagnostics report high truncation and non-progressing greedy behavior for the original DQN.
- RandomAgent is a provisional weak benchmark.
- Board encoding omits repetition history and move counters and remains absolute.
- Checkpoints omit RNG state and a lifetime episode counter.
- Explicit device/CUDA management is not implemented.
- `pyproject.toml` does not declare runtime dependencies.
- State-Action is not integrated into frozen-opponent self-play or the executable `main()`.

## Test caution

At the documentation-audit session the repository contained 246 pytest test functions. A fresh test run could not be collected in that sandbox because `python-chess` was missing and `chess_rl` was not installed/importable there. Therefore do not inherit a claim that all tests passed from this prompt. Run them in the actual project environment before treating the suite as green.

A later repository version may of course contain a different number of tests. Always inspect the attached ZIP rather than assuming this count remains current.

## Next-step rule

Do not assume CUDA, longer training, State-Action self-play, champion-vs-challenger, or any other feature is automatically next.

Inspect the new ZIP and the user's current objective, then justify the smallest next step from repository evidence.

At the current documented point, the intended immediate task is the small State-Action CPU training probe described above, unless the newly attached repository shows that it has already been implemented or reveals a reason not to do it.