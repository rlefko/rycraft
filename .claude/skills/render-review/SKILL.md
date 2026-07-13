---
name: render-review
description: Review a diff or feature against rycraft's rendering conventions (C++/MSL shared layouts, pipeline/pass parity, storage modes, coordinate conventions, run-with-validation). Use for the pre-commit rendering agent check, when the user asks for a "render review" / "Metal review" / "why does the frame look wrong", or before committing changes under src/render/, shaders/, or anything touching Metal APIs, pixel formats, sample counts, or GPU-shared structs. Reads docs/rendering-conventions.md as the source of truth.
---

# Render Review

Check a change against rycraft's rendering conventions and report where it complies, where it violates a rule, and how to fix each violation.

This skill is a review tool, not a rubber stamp. Every rule in the conventions doc was earned by a shipped defect (the renderer once carried twelve at the same time); a violating change should be told so with the concrete corruption it risks and the compliant fix. Equally, do not invent violations: a rule the diff doesn't touch is not in play.

## Step 1: Load the source of truth

Read `docs/rendering-conventions.md`. Its rule sections and the review checklist at the bottom are authoritative; if the doc has changed, follow the changed version, not any summary reproduced elsewhere (including this skill).

## Step 2: Gather the change under review

In priority order:

1. A target the user named (a PR number, branch, file, or feature description).
2. The current working diff: `git diff origin/main...HEAD`, plus `git diff HEAD` for uncommitted changes.

If the diff is empty and no target was named, say so and stop. If the diff touches no rendering surface (no `src/render/`, `shaders/`, `include/render/`, or Metal API call sites), say so and stop.

## Step 3: Walk the review checklist

Go through the checklist at the bottom of the conventions doc in order, applying each item to the diff. Two items are mechanical — run them, don't eyeball:

- Struct parity: any struct in the diff appearing in both a `.metal` file and C++ must live in `include/render/shader_types.hpp`; grep the diff for `struct` in shader files.
- Verification: the change must have been exercised in the running game with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, ideally with an `RYCRAFT_CAPTURE` frame inspected (the `playtest` skill does this end to end). Treat missing evidence as a finding, not a gap.

For each violation report: **file:line**, **the rule it breaks** (by section from the doc), **the concrete corruption risked** (what the player would see — a black frame, a shifted uniform, a quarter-screen scene), and **the compliant fix**.

## Step 4: Report

Output, in this order:

1. **Verdict**: clean, clean with notes, or violations found.
2. **Violations**, ordered by how badly the frame breaks.
3. **Risks worth a look**: rules the diff comes close to breaking.
4. **Confirmed clean**: the checklist areas the diff actually exercised and passed.

Keep the report tight: findings first, no praise, no restating the diff.
