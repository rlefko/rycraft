# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rycraft is a from-scratch Minecraft-like voxel game for Apple Silicon Macs: C++23, direct Metal rendering, Cocoa windowing, Core Audio sound, Meson + Ninja builds, and procedurally generated everything (terrain, textures, sound). Full concept in [docs/game-concept.md](docs/game-concept.md).

## Documentation Map

| Read this | Before |
|-----------|--------|
| [docs/architecture.md](docs/architecture.md) | changing ownership, threading, error handling, or module boundaries |
| [docs/rendering-conventions.md](docs/rendering-conventions.md) | touching `src/render/`, `shaders/`, or any Metal API |
| [docs/performance-conventions.md](docs/performance-conventions.md) | touching the frame loop, `gameTick`, meshing, generation, or locks |
| [docs/world-generation.md](docs/world-generation.md) | changing worldgen, block types, or the save format |
| [docs/code-conventions.md](docs/code-conventions.md) | writing any code (naming, ownership, one-source-of-truth rules) |

Update the matching doc in the same PR as a behavior change. Every new conventions rule carries a one-line "why" naming the real defect that earned it.

## Git Workflow

- All changes go through pull requests into `main` (protected — no direct pushes)
- Commits: one line, emoji-prefixed, no ending punctuation (e.g., `🐛 Fix inverted WASD movement relative to the camera`). Prefixes in use: `🏗️` build, `✨` feature, `🐛` fix, `♻️` refactor, `📝` docs, `✅` tests, `🔧` tooling

## Development Commands

Run from the repo root.

| Task | Command |
|------|---------|
| Configure | `meson setup build` |
| Build | `ninja -C build` |
| Test | `ninja -C build test` |
| Format | `clang-format -i <touched files>` |
| Lint | `clang-tidy -p build <files>` |
| Run | `./build/src/rycraft` |
| Run with Metal validation | `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft` |
| Capture a frame headlessly | `RYCRAFT_CAPTURE=/tmp/f.png ./build/src/rycraft` |

## Code Review Before Commits

Before committing any change:

1. Run a **reuse** subagent: does the change re-implement something with an existing home? Source of truth: `docs/code-conventions.md` (the one-source-of-truth table).
2. Run a **simplification** subagent: is the diff as small as the change allows? Same source of truth.
3. Run a **readability** subagent: names, comments, and structure consistent with `docs/code-conventions.md`.
4. Run the **`render-review` skill** when the diff touches `src/render/`, `shaders/`, or Metal APIs — it walks `docs/rendering-conventions.md`, including the run-with-validation requirement.
5. Run the **`perf-review` skill** when the diff touches the frame loop, `gameTick`, meshing, world generation, or locking — it walks `docs/performance-conventions.md`.
6. Use the **`playtest` skill** to verify any player-visible change in the running game (build → run with validation → capture frames → inspect).
7. Apply the findings before committing.

## Error Handling Policy

- **Metal device/queue/pipeline failures:** fatal — `RY_LOG_FATAL` logs and terminates
- **Generator v4 authority, cubes, and far terrain:** latch a typed, user-visible failure, preserve resident geometry, and keep missing collision closed. Never publish a blank cube or silently enter v3. The explicit diagnostic v3 path retains its separate legacy behavior. Why: blank fallback cubes created visible holes and allowed failed learned authority to become saved world state.
- **File I/O:** `std::optional`/`bool` returns + `RY_LOG_*` — no exceptions for I/O, and no `Result` type
- **Optional subsystems (audio):** initialize non-fatally; the game runs without them

## CI Checklist (GitHub Actions)

Every PR must pass: **format** (clang-format check) → **lint** (clang-tidy) → **build + test** (Meson/Ninja + Catch2, werror) → **CodeQL**. Config in `.github/workflows/ci.yml`. Run format and tests locally before pushing.

## Performance Targets

60 FPS at native resolution and 4x MSAA on the documented Apple M4 Max route, a 20 Hz simulation, view distance 512, and at most 64 GB unified memory. Exact cubic simulation remains capped at radius 32. Cold far coverage begins with step-32 voxel parents, then each connected coordinate advances through adjacent tiers toward its absolute distance target before exact ownership. The budgets table in `docs/performance-conventions.md` is authoritative.
