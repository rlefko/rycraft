# Architecture

How rycraft's six modules fit together, who owns what, and which rules keep it that way. Every rule below carries the defect that earned it.

## Subsystem map

```
src/engine   Engine singleton (ObjC++): app lifecycle, MTKView-driven frame
             loop, 20 Hz fixed tick, game flow (title/pause), input, camera
src/render   RenderPipeline: one MSAA scene pass + bloom + UI; mesher,
             block textures, mega-buffer, UI overlay, entity renderer
src/world    World: chunk storage/streaming/generation pipeline, save/load
src/entity   Player + animals: physics, AI state machines, flocking, spawner
src/audio    Core Audio output unit, 16-voice mixer, procedural SFX
src/common   Vocabulary types: math, seeded randomness, thread pool, logging
```

Dependency direction: `engine → render/world/entity/audio → common`. The render module reads the world; it never mutates it beyond chunk mesh flags.

## Ownership and lifetimes

- The `Engine` singleton owns everything through `EngineState`: the `World` (shared with nothing that outlives it), the `SaveManager`, the `Spawner`, the `AudioEngine`, and the `InputManager` — all `unique_ptr`/`shared_ptr`.
- `RenderPipeline` owns its six subcomponents as `unique_ptr`. **Why:** they were raw `new`/`delete` and the `InputManager` leaked outright; ownership you can't see is ownership you forget.
- `World` holds a **non-owning** `SaveManager*` (the engine owns it and constructs it first). Set via `setSaveManager`; without it, chunks generate fresh and saved edits never load.
- `World::~World` waits on in-flight generation futures. **Why:** generation workers capture `this` and insert into `chunks_`; destroying the World underneath them was a use-after-free that corrupted unrelated tests.

## Threading model

- **Main thread:** the entire frame — input, ticks, meshing, encoding. `drawInMTKView:` drives everything.
- **Generation pool:** four workers (`World::genPool_`) build chunks and insert them into `chunks_` under `chunksMutex_`.
- **Save thread:** `SaveManager` drains an async write queue; `flush()` blocks until it empties.
- **Audio render thread:** Core Audio calls `AudioEngine::audioCallback`; the voice table is mutex-guarded.

Lock discipline (see also performance-conventions):
- Never generate, load, or do I/O while holding `chunksMutex_` — `getChunk` releases it around `loadOrGenerateChunk` and re-inserts with `try_emplace`, accepting rare duplicate generation over a render-thread stall. **Why:** the original `getChunk` generated a whole chunk (milliseconds) inside the lock the render thread takes every frame.
- Worker insertions use `try_emplace`, never `operator[]`. **Why:** an async worker overwriting an existing chunk discards player edits that landed while it generated.

## Error handling policy

Three tiers, uniformly applied:

1. **Metal device/queue/pipeline failures are fatal** — `RY_LOG_FATAL` logs and aborts. A machine that can't create a pipeline state can't play the game.
2. **Chunk generation failures fall back** — a `try/catch` in `loadOrGenerateChunk` logs and produces a blank chunk rather than crashing the world.
3. **File I/O returns `std::optional`** — a missing save is not an error; a corrupt chunk logs, returns `nullopt`, and regenerates.

There is deliberately **no `Result<T,E>` type**. One existed, was used by exactly one class, and coexisted with the two idioms above; three error styles is two too many. Non-fatal subsystems (audio) initialize with a `bool` and the game continues without them.

## The GPU boundary

Every struct both C++ and Metal read lives in [`include/render/shader_types.hpp`](../include/render/shader_types.hpp), compiled by both toolchains with `static_assert` layout pins. This is a hard rule — see [rendering-conventions.md](rendering-conventions.md) for the five separate layout-drift bugs that earned it.

## Persistence

Region files (256 chunks each), LZ4-compressed, format `RYCH` v2 — layout in [world-generation.md](world-generation.md). Writes happen on block edits (async); metadata (seed, player position, world time) writes on quit through `applicationShouldTerminate:`, so the window close button and the QUIT menu item share one save path. Loads happen chunk-by-chunk in `loadOrGenerateChunk`, disk before generator.

## Testing layout

Six hermetic Catch2 modules under `tests/` (common, world, render, entity, engine, audio) linked into one binary. Filesystem tests use the `TempDir` RAII fixture from `test_helpers.hpp` — unique per process, cleaned on destruction. **Why:** tests used to share fixed `/tmp` paths and shell out to `rm -rf`, so two concurrent runs corrupted each other. Everything runs headless; no test creates a Metal device.
