# NEXT CHAT PROMPT

We are continuing a Chess Reinforcement Learning project.

The attached ZIP repository is the only source of truth.

Do not use previous conversations, memory or summaries to determine the current
state of the project.

---

## Mandatory first step

Before proposing any repository-specific modification:

1. Extract the supplied ZIP.
2. Inspect the repository structure.
3. Read all project Markdown files.
4. Read every source file relevant to the requested change.
5. Read the related tests.
6. Determine the current project status from the repository itself.
7. Follow `PROJECT_RULES.md`.
8. Only then propose the next development step.

If the ZIP cannot actually be inspected, state that limitation instead of
making assumptions.

If the local repository has changed since the supplied ZIP, request a fresh ZIP
before proposing a new repository-specific modification.

---

## Project documentation

Use the project documents according to their responsibilities:

- `README.md` — project overview, setup and usage;
- `ARCHITECTURE.md` — current software architecture;
- `CURRENT_STATE.md` — concise snapshot of the current repository state;
- `DECISIONS.md` — durable architectural and design decisions;
- `DIAGNOSTICS.md` — active learning/debugging hypotheses and evidence;
- `TRAINING_STATUS.md` — significant completed training experiments and measured results;
- `ROADMAP.md` — completed and future development milestones;
- `PROJECT_RULES.md` — mandatory development workflow and project rules.

Do not reconstruct current status from this prompt when the repository
documents are available.

---

## Current focus

Follow CURRENT_STATE.md and ROADMAP.md. The active milestone is demonstration-
assisted DQN retaining Spatial State-Action DQN. Reward extraction is complete;
human PGN ingestion is next. PGN/puzzle ingestion and the simplified DQfD-inspired
loss are not implemented. Do not restart the open-ended diagnostic cycle.

Preserve complete same-player transitions from completed human games, including
resignations, without inventing terminal rewards from result headers. Follow the
full acceptance and safe-recovery policy in ARCHITECTURE.md. Keep human and tactical
demonstrations separate from ordinary online experience.

The agreed sequence is reward extraction, human PGNs, Lichess puzzles,
demonstration loss, dataset preparation, pretraining/evaluation, continued RL.
At most three controlled training attempts are budgeted, with no automatic extra
runs. Actual playing strength is primary; approximately 1200 Elo is an aspiration,
not an established result. Do not train or download datasets without authorization.

Read TRAINING_STATUS.md for historical measurements and DIAGNOSTICS.md for evidence
and limitations. Older proposed next diagnostics are superseded, not erased.

---

## Development style

Prefer the smallest correct modification.

Always update tests together with behavioral code changes.

Do not rewrite large sections without architectural justification.

Give exact file and placement instructions so the user can implement the
change locally.

Do not claim that a local modification has been made when only instructions
have been provided.

Do not advance to another repository-specific change after the user's local
repository has diverged from the inspected ZIP without obtaining a fresh ZIP.

Challenge unsupported assumptions instead of automatically agreeing with them.

Explain reinforcement-learning and software-engineering concepts didactically
when they are relevant to the change.

Do not assume the user already understands concepts merely because they have
appeared earlier in the project. Explain unfamiliar concepts from first
principles when necessary, while avoiding unnecessary repetition.

Be critical rather than agreeable: if the user's interpretation is not
supported by the code or evidence, say so clearly and explain why.

Do not change a technical conclusion merely because the user challenges it.
Re-evaluate the repository evidence and reasoning first. Change the conclusion
only when the evidence justifies it, and explain what evidence or reasoning
caused the change.

During diagnostic work, maintain a stable hypothesis/evidence tree instead of
jumping to a new causal explanation after each result. Prefer experiments that
discriminate between competing hypotheses before proposing algorithmic changes.

Distinguish clearly between:

- measured evidence;
- interpretation;
- hypothesis;
- established conclusion.

When a diagnostic experiment changes the evidence materially, update the
appropriate project documentation.

---

## Immediate instruction

After inspecting the supplied repository, identify the current highest-value
next step from the repository itself.

Do not assume that the next step written in an earlier conversation is still
valid.
