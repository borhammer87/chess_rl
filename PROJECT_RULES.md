# PROJECT_RULES.md

# Chess Reinforcement Learning Project Rules

This document defines the development workflow for the project.

It is as important as the source code.

Whenever contributing to this repository, follow these rules.

---

# 1. PROJECT PHILOSOPHY

The objective is NOT simply to build a chess AI.

The objective is to understand every component of a DQN-based Reinforcement Learning system.

Understanding is more important than speed.

Architecture is more important than features.

Code quality is more important than quantity.

---

# 2. REPOSITORY IS THE SOURCE OF TRUTH

Never rely on memory.

Never rely on previous conversations.

Always inspect the repository first.

If the repository contradicts the chat, trust the repository.

---

# 3. BEFORE WRITING CODE

Before proposing any modification:

1. Inspect the repository.

2. Read all project markdown files.

3. Understand the current implementation.

4. Understand the current architecture.

Only then propose changes.

---

# 4. DEVELOPMENT STYLE

Work like a senior software engineer performing code review.

Never generate an entire subsystem in one response.

Instead:

1.
Explain the objective.

2.
Explain WHY the modification is needed.

3.
Explain possible alternatives.

4.
Modify only a small amount of code.

Approximately:

20–40 lines whenever possible.

5.
Immediately update or create tests.

6.
Wait until pytest has been executed.

7.
Only continue once the tests pass.

---

# 5. TEACHING STYLE

Always explain:

- what problem we are solving
- why this solution is preferable
- consequences for future architecture
- possible future improvements

Never assume Reinforcement Learning knowledge.

Teach every important concept before implementing it.

---

# 6. TESTING

Every code modification should include tests whenever appropriate.

Prefer many small tests over few large ones.

Do not continue implementing new functionality while tests are failing.

---

# 7. DOCUMENTATION

Documentation is part of the project.

It is not optional.

Whenever a significant milestone is completed:

Review every markdown.

Determine whether each one requires updating.

Never modify documentation unnecessarily.

---

# 8. MARKDOWN RESPONSIBILITIES

README.md

Purpose:

- explain the project
- installation
- execution

Update only when user-facing behaviour changes.

---

ARCHITECTURE.md

Purpose:

- architecture
- package responsibilities
- component interaction

Update only when architecture changes.

---

DECISIONS.md

Purpose:

Record architectural decisions.

Each decision should include:

- Decision
- Motivation
- Alternatives
- Consequences

Update only when a new design decision is made.

---

ROADMAP.md

Purpose:

Track project milestones.

Update when a milestone starts or finishes.

---

CURRENT_STATE.md

Purpose:

Store the current development state.

Update after every important work session.

Include:

- completed functionality
- current milestone
- current tests
- next task

---

NEXT_CHAT_PROMPT.md

Purpose:

Store the prompt used for future conversations.

Update only if the development workflow changes.

---

# 9. GIT

Git history is part of the documentation.

Whenever a stable milestone is completed:

- suggest a commit message
- explain why
- suggest a version tag if appropriate

---

# 10. CONTEXT MANAGEMENT

If the conversation context may no longer reflect the repository:

STOP writing code.

Explain why.

Recommend opening a new chat.

Recommend updating the documentation.

Only continue after inspecting the current repository.

---

# 11. SMALL CHANGES

Prefer:

small

incremental

well-tested

changes.

Avoid large rewrites whenever possible.

---

# 12. LONG-TERM OBJECTIVE

The finished repository should be understandable by someone reading only:

README.md

ARCHITECTURE.md

DECISIONS.md

ROADMAP.md

CURRENT_STATE.md

without needing previous conversations.

"If the repository is available, never propose code based on memory. Always inspect the relevant file before suggesting modifications."