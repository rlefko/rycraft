---
name: perf-review
description: Review a diff or feature against rycraft's performance conventions (frame/tick budgets, hot-path allocation, lock discipline, bounded populations and caches, seeded determinism). Use for the pre-commit performance agent check, when the user asks for a "perf review" / "frame rate check", when investigating hitches or stalls, or before committing changes to the frame loop, gameTick, meshing, world generation, chunk streaming, or any mutex. Reads docs/performance-conventions.md as the source of truth.
---

# Performance Review

Check a change against rycraft's performance conventions and report where it complies, where it violates a rule, and how to fix each violation.

This skill is a review tool, not a rubber stamp. Every rule in the conventions doc was earned by a real defect (string-allocating chunk keys on the hottest path, chunk generation inside the render thread's mutex, 5,915 simultaneously ticking animals); hold changes to those rules with concrete cost accounting. Equally, do not invent violations: a rule that does not apply to the diff is simply not in play.

## Step 1: Load the source of truth

Read `docs/performance-conventions.md`. Its budgets table, rule sections, and the review checklist at the bottom are authoritative; if the doc has changed, follow the changed version.

## Step 2: Gather the change under review

In priority order:

1. A target the user named (a PR number, branch, file, or feature description).
2. The current working diff: `git diff origin/main...HEAD`, plus `git diff HEAD` for uncommitted changes.

If the diff is empty and no target was named, say so and stop. If the diff touches no per-frame, per-tick, or lock-holding code path, say so and stop.

## Step 3: Walk the review checklist

Go through the checklist at the bottom of the conventions doc in order. Mechanical sweeps worth running on every diff:

- `git diff origin/main... | grep -E '^\+.*(std::string|ostringstream|new |make_shared|make_unique)'` — inspect each hit that lands on a per-frame/per-tick path.
- `git diff origin/main... | grep -E '^\+.*lock_guard'` — for each new lock scope, confirm no generation, I/O, or allocation-heavy work happens inside it.
- New loops over entities, chunks, or particles: confirm the bound (simulation distance, cap, or eviction) is stated in the code.

Account costs structurally: calls × frequency × thread. "One extra `getLoadedChunks()` copy per frame at 625 chunks" decides priorities; "it feels fine" does not. A claimed speedup should show in the F3 HUD numbers or a before/after measurement.

## Step 4: Report

Output, in this order:

1. **Verdict**: clean, clean with notes, or violations found.
2. **Violations**, each with file:line, the rule broken, the structural cost, and the compliant fix — ordered by player impact.
3. **Risks worth a look**.
4. **Confirmed clean**: the checklist areas the diff actually exercised and passed.

Keep the report tight: findings first, no praise, no restating the diff.
