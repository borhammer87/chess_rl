# NEXT CHAT PROMPT

We are continuing the Chess Reinforcement Learning project.

## Source-of-truth rule

The attached ZIP repository is the primary and authoritative source of truth for the current code.

Before proposing any modification, perform a repository audit. This is a
mandatory development step, not a quick orientation pass.

1. extract the ZIP into a fresh location;
2. inspect the complete repository tree and identify every tracked project
   file present in the snapshot;
3. read **every project Markdown file completely, from beginning to end**;
4. read `pyproject.toml` and any other project/configuration files that can
   affect the current implementation;
5. read **every source file relevant to the requested change completely**,
   not only matching functions or search excerpts;
6. read **every test file relevant to those source files and the requested
   change completely**;
7. follow imports, callers, result/data structures, constants and shared
   utilities far enough to understand the complete execution path affected by
   the proposed change;
8. inspect existing tests before proposing new tests, so that the assistant
   does not invent test names, locations, fixtures, assertions or coverage
   that already exists;
9. reconstruct the current project status from the inspected code and tests;
10. cross-check that reconstruction against the Markdown documentation and
    explicitly notice material contradictions or stale documentation;
11. only after completing the above may prior conversation/project context be
    used to recover rationale, chronology or experimental results that the
    repository itself does not preserve.

Repository inspection must be substantive. Merely listing files, searching for
keywords, reading selected snippets, relying on extracted summaries, or
checking only the function expected to change does **not** count as having read
or inspected the repository as required above.

Do not claim to have read the complete repository, all source code, all tests,
or all Markdown unless that work was actually performed in the current
repository snapshot. Be precise about scope instead, for example: "I read all
Markdown and the complete training/evaluation execution path plus its related
tests."

When giving an exact modification, verify the target against the current
snapshot immediately before answering. Exact function names, existing code,
test names, assertions, line/location descriptions and surrounding context
must come from the inspected repository, not from memory or inference. If an
exact location cannot be verified, say so and ask for the missing/current file
rather than approximating it.

Never write phrases such as "approximately this test", "you should have",
"it should look like", or otherwise reconstruct existing repository content
from expectation when the current ZIP makes verification possible. Quote or
identify the actual current code.

If a proposed change depends on a file that has not yet been read completely,
read it before proposing the change.

If repository contents contradict prior chat context, trust the repository.
Use prior project context only to recover rationale or chronology that the
repository itself does not preserve, never to claim that unimplemented code
exists.

If the ZIP cannot actually be extracted or inspected, state that limitation
and do not guess or propose repository-specific code changes.

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

## Current repository snapshot

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

The executable `main()` still uses this original DQNCNN path for frozen-opponent self-play. Its documented schedule is 100 episodes, 150 learner steps, batch size 32, minimum replay size 1000, target synchronization every 10 episodes, and checkpoint/evaluation/frozen-opponent refresh every 25 episodes. RandomAgent evaluation uses 10 games per color.

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
- integration with the existing RandomAgent training workflow;
- dedicated reproducible CPU probe in `scripts/run_state_action_probe.py`;
- controlled model-initialization, training and evaluation seeds;
- balanced greedy evaluation before and after training;
- controlled 100-episode and 500-episode CPU training probes.

State-Action is not integrated into frozen-opponent self-play or the executable `main()`.

## State-Action experimental evidence

The reproducible 100-episode probe produced:

- initial greedy evaluation: `0W / 0D / 10L / 30 truncated`, score `0.000`;
- final greedy evaluation: `2W / 5D / 4L / 29 truncated`, score `0.113`;
- final epsilon `0.6274`;
- approximately 124–125 seconds of CPU training.

Two consecutive executions reproduced the same training and evaluation metrics.

The reproducible 500-episode probe produced:

- the same initial evaluation: `0W / 0D / 10L / 30 truncated`, score `0.000`;
- final greedy evaluation: `3W / 3D / 2L / 32 truncated`, score `0.113`;
- final epsilon `0.1000`;
- approximately 870 seconds / 14.5 minutes of CPU training.

Interpret these results conservatively.

They establish that State-Action can perform sustained end-to-end training reproducibly at practical CPU cost. They do **not** establish reliable chess-playing strength.

In particular, increasing training from 100 to 500 episodes did not improve the balanced greedy RandomAgent score beyond `0.113`, and greedy truncation remained extremely high. Do not claim that State-Action has solved the original learning problem merely because some completed games after training were wins or draws.

Targeted diagnostics on the reproducible 100-episode configuration established:

- initial greedy truncations: `30/30` without a claimable threefold or
  fifty-move draw;
- training truncations: `78/78` without either claimable draw;
- final greedy evaluation: `29` truncations, with `28` lacking either
  claimable draw, `1` claimable by threefold and `1` claimable by the
  fifty-move rule;
- initial greedy truncated games: average final total material `20.20/78`,
  average learner material balance `-6.27`, average absolute balance `9.13`;
- training truncations: average final total material `10.55/78`, average
  learner material balance `+0.45`, average absolute balance `5.81`;
- final greedy truncated games: average final total material `10.83/78`,
  average learner material balance `-2.62`, average absolute balance `6.28`.

These results rule out claimable draws as the main explanation for truncation
and show that truncated games are generally heavily simplified rather than
simply reaching the artificial horizon with most material intact.

Treat this as reproducible evidence of behavioral change after training, not
as proof of reliable chess strength.

A concrete truncated greedy White-vs-RandomAgent game has now been captured
through the existing PGN infrastructure. The probe searches for the first
truncated game within a bounded number of attempts rather than assuming that an
arbitrary diagnostic game will truncate. The captured game reached the full
configured artificial horizon of `150` learner steps / `300` plies.

A tested `evaluate_state_action_greedy_choice()` utility exists and is now
connected to the diagnostic PGN analysis.

The reproducible 100-episode probe reconstructs every learner position from the
captured truncated game while the exact final policy remains in memory. All
`150` recorded learner moves matched the reconstructed greedy action.

For each learner position the probe reports the legal-action count, best
Q-value, top-two Q-value gap, mean Q-value, minimum Q-value, standard deviation
and full legal-action Q-value range.

The resulting evidence does not support simple global Q-value collapse. The
network often separates the complete legal action set by meaningful amounts,
while nevertheless assigning extremely similar values to its two highest
candidates in many ordinary positions.

Observed top-two gaps reached as low as `0.000002` and `0.000039`. Conversely,
queen promotions were strongly distinguished, with gaps of `0.063167` and
`0.034973`.

The current interpretation is that State-Action has learned some real action
preferences but frequently lacks strong separation among its highest-valued
ordinary choices. This does not yet identify the cause.

## Current diagnostic status

The Q-value, Bellman-target, TD-error, transition-semantics and observable
replay-diversity investigations are complete for the current milestone.

The final 10,000-transition training replay does not show globally collapsed
Bellman targets or vanishing TD errors. For non-terminal transitions,
Bellman-target standard deviation was `0.073968`, selected-action Q-value
standard deviation was `0.063227`, and mean absolute TD error was `0.020054`.

The exact 150-transition replay belonging to the saved truncated greedy
diagnostic game was also analyzed. Its non-terminal Bellman-target standard
deviation was approximately `0.03576`, selected-action Q-value standard
deviation approximately `0.02893`, and mean absolute TD error approximately
`0.02366`.

The final truncation transition had target `-0.100000` against a predicted
Q-value of `0.284372`, leaving absolute TD error `0.384372`.

The State-Action Bellman transition path has been audited from episode
construction through replay and `train_step()`. A learner transition spans the
learner action plus the opponent reply, so its next state correctly represents
the learner's next decision state. Reward shaping covers the same interval,
and terminal/truncated transitions correctly disable bootstrapping. No
implementation bug has been identified in this path.

The final 10,000-transition training replay contained `1,685` unique action
IDs, `9,910` unique encoded states and `9,977` unique state-action pairs. Its
top 10, 25 and 50 action IDs accounted for `3.6%`, `7.7%` and `13.5%` of the
replay.

The exact 150-transition truncated greedy diagnostic game contained `105`
unique action IDs, `149` unique encoded states and `149` unique state-action
pairs. Its top 10, 25 and 50 action IDs accounted for `24.0%`, `46.0%` and
`63.3%` of the trajectory.

The retained training replay therefore does not show strong exact
state/state-action redundancy. The greedy trajectory also visits almost
entirely distinct encoded states, despite visibly reusing some movement
patterns and showing substantially greater action-ID concentration.

Do not interpret this as proof that exploration is sufficient or that action
concentration causes the weak policy. The training replay and a single greedy
game have different sampling and action-availability properties.

The strongest behavioral finding remains that the network frequently assigns
extremely similar values to its highest-ranked ordinary legal actions, despite
meaningful differentiation across the complete legal-action set and strong
preferences in some cases such as queen promotions.

Reward and credit assignment are now plausible hypotheses to investigate, but
they have not been established as the cause.

## Current development focus

The next task is to inspect the existing State-Action reward and
credit-assignment design before changing it.

Read the complete reward calculation, episode construction, Bellman update and
related tests. Determine exactly what learning signal an ordinary learner
action receives immediately and how delayed consequences propagate through the
current transition interval and `gamma`.

Do not assume in advance that reward design or gamma is wrong. Use the
repository implementation and the accumulated diagnostic evidence to formulate
one concrete hypothesis and identify the smallest controlled experiment capable
of testing it.

Do not simultaneously change reward shaping, gamma, PER, exploration,
representation or network architecture. Reuse the existing reproducible
State-Action probe and diagnostics wherever possible.

## Important limitations

- Neither model has demonstrated reliable chess-playing strength.
- State-Action greedy evaluation still has a very high truncation rate.
- `RandomAgent` remains a weak provisional benchmark.
- State-Action has not been compared with the original DQNCNN under a controlled equivalent experiment.
- State-Action is not integrated into frozen-opponent self-play or `main()`.
- Board encoding omits repetition history and move counters and remains absolute.
- Checkpoints omit RNG state and a lifetime episode counter.
- Explicit device/CUDA management is not implemented.
- `pyproject.toml` does not declare runtime dependencies.

## Test status

A preserved automated-suite run from an earlier point in the State-Action
probe work reported:

`257 passed in 11.29s`

Additional tests and diagnostic changes have been added since that run. The
suite was subsequently reported green after those changes, but the exact
current test count and duration are not preserved in this repository snapshot.

Treat `257 passed` as historical evidence rather than the current test count.
Always run the relevant tests after code changes and do not inherit a green
status blindly into a newer repository state.

## Next-step rule

Before proposing any new code modification, inspect the fresh ZIP completely
according to the source-of-truth and inspection rules above.

At the current documented milestone, inspect the existing State-Action reward
and credit-assignment path before proposing an experiment. Read the relevant
reward calculation, episode runner, agent Bellman update and tests completely.

The immediate objective is not to redesign the reward function. It is to
identify one concrete, repository-supported hypothesis about whether the
current learning signal can explain weak differentiation among ordinary actions
and persistent failure to terminate heavily simplified greedy games.

Then propose the smallest controlled experiment capable of testing that
hypothesis. Change one meaningful variable at a time and reuse the existing
reproducible State-Action probe and diagnostic infrastructure.

Do not automatically change gamma, PER, exploration, board encoding, network
architecture, CUDA/device handling or self-play integration.

The user implements all code changes locally. Give exact file names, exact
placement and complete replacement/addition blocks. Do not claim a modification
has been implemented merely because it was proposed in chat.

After the user modifies the repository, require a fresh ZIP before proposing
the next code modification.