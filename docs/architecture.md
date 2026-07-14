# Architecture

How rycraft's six modules fit together, who owns what, and which rules keep it that way. Every rule below carries the defect that earned it.

## Subsystem map

```
src/engine   Engine singleton (ObjC++): app lifecycle, MTKView-driven frame
             loop, 20 Hz fixed tick, game flow (title/pause), input, camera
src/render   RenderPipeline: HDR scene pass + shadow cascades, SSAO,
             volumetric light & clouds, water SSR, bloom, post/tonemap, UI;
             mesher, block textures, mega-buffer, frame ring, entity renderer
src/world    World: chunk storage/streaming/generation pipeline, save/load,
             LightEngine (derived block light)
src/entity   Player + animals: physics, AI state machines, flocking, spawner
src/audio    Core Audio output unit, 16-voice mixer, procedural SFX
src/common   Vocabulary types: math, seeded randomness, thread pool, logging
```

Dependency direction: `engine → render/world/entity/audio → common`. The render module reads the world; it never mutates it (mesh staleness lives in the renderer's own registry, keyed on each chunk's edit version).

## Ownership and lifetimes

- The `Engine` singleton owns everything through `EngineState`: the `World` (shared with nothing that outlives it), the `SaveManager`, the `Spawner`, the `AudioEngine`, and the `InputManager` — all `unique_ptr`/`shared_ptr`.
- `RenderPipeline` owns its rendering subcomponents as `unique_ptr` (mesher, mega-buffer, bloom, shadow map, SSAO, volumetrics, volumetric clouds, post stack, entity and UI renderers), each owning its own PSOs and targets and composed like `Bloom`. **Why:** they were raw `new`/`delete` and the `InputManager` leaked outright; ownership you can't see is ownership you forget.
- `World` holds a **non-owning** `SaveManager*` (the engine owns it and constructs it first). Set via `setSaveManager`; without it, chunks generate fresh and saved edits never load.
- `World::~World` waits on in-flight generation futures — moved out of the map first, waited on outside `pendingMutex_`, in a loop. **Why:** generation workers capture `this` and insert into `chunks_` (use-after-free if the World dies first), and a finishing worker pumps the backlog under `pendingMutex_` (deadlock if the destructor waits while holding it).
- `RenderPipeline` owns the `MeshScheduler`, whose workers reference the World. The engine's quit path calls `shutdownMeshWorkers()` **before** the world can be destroyed. **Why:** ObjC ivar destruction order at teardown is nothing to bet a use-after-free on.

## Threading model

- **Main thread:** input, ticks, mesh uploads (memcpy into the mega-buffer), encoding. `drawInMTKView:` drives everything. It keeps **three frames in flight** behind a `dispatch_semaphore(3)` — `render()` waits at the top and the completed handler signals — so per-frame uniforms ride a 3-slot ring and the MegaBuffer frees ranges through a deferred queue drained once the GPU completes the frame the range was freed in (`completedFrame >= N`; in-order completion means that one wait covers every command buffer still referencing it). **Why:** the GPU reads buffers at execution time, not encode time; without the ring the next frame overwrites constants the GPU is still reading, and a freed vertex range gets recycled under a live draw.
- **Generation pool:** four workers (`World::genPool_`) build chunks nearest-first through a bounded submission window, insert them into `chunks_` under `chunksMutex_`, and pump the next backlog entry themselves.
- **Mesh workers:** two threads owned by the renderer's `MeshScheduler` snapshot chunks (one bounded copy under `chunksMutex_` — see `snapshotForMeshing`), mesh lock-free, and hand version-stamped results back to the render thread. **Why off the main thread:** a streaming burst of 16 full-chunk builds consumed an entire frame.
- **Block light is derived state, computed on two paths.** `LightEngine::computeSelfLight` runs on the gen/load worker *before* a chunk is inserted (lock-free, no borders), and cross-chunk light is reconciled by `World::reconcileLight` on the tick thread under a per-call chunk budget: it pulls light across the four face borders, bumps the `version`/`needsMeshUpdate` of any chunk whose stored light changed, and re-enqueues until quiescent (queue guarded by `lightMutex_`, the leaf below `chunksMutex_`). Edits run removal+addition BFS in world space while already holding `chunksMutex_`. **Why derived, not serialized:** propagation is monotone over fixed blocks, so the flood has a unique fixed point regardless of chunk-insertion order — light can be recomputed deterministically and never needs to touch the save format (see world-generation.md).
- **Save thread:** `SaveManager` serializes, compresses, and writes queued chunks; `flush()` blocks until it empties. Queued chunks stay readable through a pending map so unload-then-return never reads a stale file.
- **Audio render thread:** Core Audio calls `AudioEngine::audioCallback`; it holds `_voiceMutex` for the whole mix, so every read of the voice table is serialized against `playSound`/`stopVoice`. **Why:** the callback used to read each voice's `samples`/`active`/`readPosition` without the lock while `playSound` reassigned them under it — a torn `std::vector` read / use-after-free on the real-time thread that corrupted the heap and made libmalloc trap.

Lock discipline (see also performance-conventions):
- Never generate, load, or do I/O while holding `chunksMutex_` — `getChunk` releases it around `loadOrGenerateChunk` and re-inserts with `try_emplace`, accepting rare duplicate generation over a render-thread stall. **Why:** the original `getChunk` generated a whole chunk (milliseconds) inside the lock the render thread takes every frame.
- Worker insertions use `try_emplace`, never `operator[]`. **Why:** an async worker overwriting an existing chunk discards player edits that landed while it generated.

Rendering vs. simulation: the sim runs at a fixed 20 Hz and `drawInMTKView:` renders the camera at the **latest** tick position — no inter-tick interpolation. **Why:** interpolation (rendering `lerp(prevTick, curTick, alpha)`) trails the simulation by up to one tick (50 ms) and read as floaty/laggy; the 20 Hz camera step is preferred over that added latency. Falls no longer look instantaneous because vertical velocity is reset on the ground instead of saturating (see `Player::tick`), not because of render smoothing.

## Error handling policy

Three tiers, uniformly applied:

1. **Metal device/queue/pipeline failures are fatal** — `RY_LOG_FATAL` logs and aborts. A machine that can't create a pipeline state can't play the game.
2. **Chunk generation failures fall back** — a `try/catch` in `loadOrGenerateChunk` logs and produces a blank chunk rather than crashing the world.
3. **File I/O returns `std::optional`** — a missing save is not an error; a corrupt chunk logs, returns `nullopt`, and regenerates.

There is deliberately **no `Result<T,E>` type**. One existed, was used by exactly one class, and coexisted with the two idioms above; three error styles is two too many. Non-fatal subsystems (audio) initialize with a `bool` and the game continues without them.

## The GPU boundary

Every struct both C++ and Metal read lives in [`include/render/shader_types.hpp`](../include/render/shader_types.hpp), compiled by both toolchains with `static_assert` layout pins. This is a hard rule — see [rendering-conventions.md](rendering-conventions.md) for the five separate layout-drift bugs that earned it.

## Persistence

One LZ4-compressed file per chunk, sharded into 32×32-chunk region directories, format `RYCH` v3 — layout and the why in [world-generation.md](world-generation.md). Edited chunks (`modifiedSinceSave`) save on unload and in a quit-path sweep; metadata (seed, player position, world time) writes on quit through `applicationShouldTerminate:`, so the window close button and the QUIT menu item share one save path. Loads happen chunk-by-chunk in `loadOrGenerateChunk`, disk before generator.

`GraphicsSettings` (per-effect toggles + quality steppers, defaults all max) serializes with the input bindings into one `~/Library/Preferences/rycraft/settings.json`, **loaded before `RenderPipeline` is constructed** (the pipeline sizes its targets and PSOs from the settings) and saved on settings-close and quit. Env overrides (`RYCRAFT_SHADOWS`/`VL`/`CLOUDS`/`SSAO`/`SSR`/`WAVING`/`LENS_FLARE`/`VIBRANCE`/`SHARPEN`/`BLOOM`) apply *after* the JSON load and are never written back, so a playtest override can't corrupt the saved config. **Why one file:** two preference files (bindings + graphics) would drift and double the load-order surface; the settings menu and the renderer read the same struct.

## Testing layout

Six hermetic Catch2 modules under `tests/` (common, world, render, entity, engine, audio) linked into one binary. Filesystem tests use the `TempDir` RAII fixture from `test_helpers.hpp` — unique per process, cleaned on destruction. **Why:** tests used to share fixed `/tmp` paths and shell out to `rm -rf`, so two concurrent runs corrupted each other. Everything runs headless; no test creates a Metal device.
