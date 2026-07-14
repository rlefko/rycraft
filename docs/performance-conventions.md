# Performance Conventions

Budgets and mechanics rules. As with the rendering doc, each rule's "why" is a defect found in this codebase. The `perf-review` skill walks this file's checklist.

## Budgets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS sustained at native (Retina) resolution at view distance 12–16 |
| GPU frame time | ≤ 12 ms rendering budget at native Retina, vd 16, M4 Max (Sildur-style effects all on); measured ~4 ms steady state — the effects are not the frame-rate limiter |
| Simulation tick | 20 Hz fixed timestep; a tick must fit well inside 50 ms |
| Chunk generation | ~4–5 ms per chunk on the 4-worker pool, `MAX_INFLIGHT_GEN = 32` submission window |
| Chunk meshing | 2 mesh workers, `MAX_INFLIGHT_MESH = 8`; render thread uploads only (24 meshes / 8 MB per frame) |
| View distance | 12 chunks default, settings up to 32; the mega-buffer sizes itself from the setting (~128 KB vertices per visible chunk + 30% headroom) |
| Memory | < 4 GB RSS |
| Animals | 64 spawned cap, 96-block simulation distance |

The F3 debug HUD shows the real numbers (EMA frame time, gen/mesh ms, pending count, mega-buffer usage), and the 60-frame diagnostic log line mirrors them so headless playtests can measure. **Why:** it used to hardcode `fps = 60.0` — a perf HUD that lies is worse than none.

Measured (M4 Max, 2026-07): vd 12 → 16.7 ms/frame (vsync), vd 16 → 16.7 ms, vd 24 → ~29 ms (opt-in; far-ring LOD is the known lever).

GPU frame time is timed always-on from `GPUEndTime − GPUStartTime` (EMA in `PerformanceStats.gpuFrameMs`, F3 line + diagnostic log). With every Sildur-style effect on, the whole frame measures ~4 ms GPU on an M4 Max at native Retina steady state — a large margin under the 12 ms budget, so the streaming/upload path (not the shaders) is what a spike traces to. Per-pass `MTLCounterSampleBuffer` stage timestamps are an opt-in diagnostic (`RYCRAFT_GPU_COUNTERS=1`, resolved three frames later, mirrored to the log; zero overhead when off): scene ~2.4, three shadow cascades ~0.1/0.2/0.8, water+SSR ~2.2, UI ~2.0 ms. These stage timestamps **overlap** — the TBDR GPU pipelines consecutive passes, so per-pass numbers sum above the ~4 ms wall-clock total; use the always-on total as the budget figure and the per-pass counters only to attribute a *change*. The modeled per-pass envelope the design was sized against (cascades 1.3–1.8 · scene 3.6–4.0 · SSAO 0.5–0.7 · clouds 0.8–1.5 · water+SSR 1.2–1.4 · VL 0.9–1.1 · bloom 0.6–1.0 · composite 0.5 → ≈10.3–11.8 ms) is the ceiling, not the measurement. **Degrade order** if a slower GPU breaches budget, each a one-line constant: clouds → quarter-res / fewer steps, VL 16→8 steps, cascade-2 alternate-frame refresh, SSAO → quarter-res, SSR 24→12 steps, RGBA16F resolve → RG11B10.

## 1. Hot-path allocation

- **No heap allocation per block or per chunk access.** Chunk keys are the packed `ChunkPos` value type. **Why:** `World` built a `std::string` key via `ostringstream` on every `getBlock` — an allocation on the hottest path in the game.
- **Per-frame scratch is reused members**, e.g. the renderer's `_liveChunkKeys` sweep set, `_waterDraws`, `_pendingResults`, and the UI overlay's vertex vector. Mesh workers keep `thread_local` snapshot + scratch buffers. **Why:** each mesh build used to reserve ~2.25 MB of fresh vectors — ~36 MB of allocation traffic per frame during streaming bursts.
- The mesher's block accessor is a **template parameter**, not `std::function`. **Why:** ~1M indirect calls per chunk build was the documented hot-path debt; templating removed it.

## 2. Lock discipline

Lock ordering (a thread may only take locks left → right; the leaves are never held together):

```
pendingMutex_ → chunksMutex_        (World)
World::lightMutex_                   leaf (below chunksMutex_)
MeshScheduler::jobMutex_             leaf
MeshScheduler::completedMutex_       leaf
MegaBuffer::_mutex                   leaf
SaveManager::saveMutex_              leaf
```

- **Never generate, load, or perform I/O under `chunksMutex_`** — it's on the render thread's path. Release, do the work, re-insert with `try_emplace`; a rare duplicate generation beats a guaranteed stall. **Why:** `getChunk` once generated whole chunks inside the lock.
- The one sanctioned exception: `World::snapshotForMeshing` performs a **bounded ~165 KB memcpy** under `chunksMutex_` (tens of microseconds) — a padded 18×18×256 block array (the chunk plus its eight neighbors' walls and corner columns the baked AO needs) and a parallel same-size block-light ring. Blocks and stored light mutate only before a chunk is inserted or under this mutex, so the copy is always internally consistent — after it, meshing runs lock-free on private data. **Why the growth (~83 → ~165 KB):** corner AO reads diagonal neighbors and block light reads a one-block ring, so the snapshot widened from four face-walls to all eight neighbors and gained a full-byte light plane beside the block plane (the chunk stores light nibble-packed; the snapshot unpacks it for the mesher); still tens of microseconds, still the only lock-held copy.
- **`World::lightMutex_` is a leaf below `chunksMutex_`** — the block-light reconcile queue (chunks whose stored light a neighbor load or an edit may have staled) is only ever taken innermost, never held while acquiring another lock. `reconcileLight` drains it on the tick thread under a per-call budget. **Why:** block light is derived state that crosses chunk borders; a queue drained on one thread with a bounded budget keeps the flood off the render path without a new place in the lock order that could invert against `chunksMutex_`.
- **Never hold a lock while waiting on a future whose task takes that lock.** **Why:** `~World` once waited on generation futures under `pendingMutex_` while finishing workers pumped the backlog under the same mutex — move the futures out, then wait, and loop for stragglers.
- Accepted trade-off: two threads may occasionally generate the same chunk; `try_emplace` keeps one and drops the other. Document any new instance of this pattern.
- **The audio render callback holds `_voiceMutex` for the entire mix, and `playSound` must not allocate under it** — copy the incoming buffer outside the lock, then `std::swap` it into the slot so the critical section is O(1). **Why:** one-sided locking (callback read the voice table lock-free while `playSound` reallocated it) let the real-time thread read a torn `std::vector` and trap; a `malloc`/`free` under the lock would instead stall that thread.

## 3. Simulation cost scales with what the player can perceive

- Chunk generation streams **nearest-first** through a sorted backlog and a bounded submission window that finishing workers refill (`World::pumpGeneration`). A boundary cross rebuilds the backlog — re-prioritizing and implicitly cancelling everything not yet submitted. **Why:** raster-order submission of the whole radius meant the chunk in front of the player queued behind hundreds behind them.
- Generation reaches **view distance + 1** (visible chunks always have generated neighbors to mesh against); unloading waits until **view distance + 2** for hysteresis. Completed futures are pruned by readiness (`wait_for(0)`), not `valid()` — **why:** `valid()` stays true until `get()`, so the pending map only ever grew.
- Meshing runs on 2 workers with **version-stamped results**: `Chunk::version` bumps on every edit (self + boundary neighbors); the registry re-requests whenever `builtVersion` differs, and a stale result still uploads (a newer mesh beats a hole). Edits within 2 chunks of the camera re-mesh synchronously so breaking a block never shows a stale frame.
- Animals beyond 96 blocks skip AI and physics entirely; the initial population is capped at 64 in a 7×7-chunk ring, gated on `getPendingChunkCount() == 0` (which counts the backlog too). **Why:** biome densities across the full view distance once spawned 5,915 animals and the tick collapsed.
- Meshes of unloaded chunks are swept every frame and their mega-buffer allocations freed. **Why:** the cache never evicted, so long play sessions exhausted the buffer.
- Worldgen evaluates noise on a **world-aligned 4×4×4 lattice** and interpolates (~13× fewer evals than per-voxel); lattice columns provably above the terrain skip cave noise entirely. Cross-chunk features re-roll bounded per-chunk attempt counts (12 trees, 4 regions) — never unbounded scans.

## 4. GPU workload shape

- **One MSAA scene pass** still carries the bulk of shading — sky, terrain, entities, highlight, particles, and the flat cloud tier share its encoder, and shadow PCF, block light, corner AO, wind sway, emissive, and wetness/SSS all resolve inside its chunk fragment. The frame graph around it is a **fixed list of sanctioned passes**, each earning its place by needing a resource tile memory cannot provide mid-pass — the resolved color or depth as a *sampled* texture, a different camera, or a lower-resolution target. A new pass needs the same kind of reason; absent one, fold the work into the scene fragment.

  | Pass(es) | Why it can't fold into the scene pass |
  |----------|----------------------------------------|
  | Shadow cascades ×3 (+ water-shadow slices) | rendered from the light's camera, depth-only — a different view frustum |
  | SSAO gen + bilateral blur | screen-space, samples the resolved depth as a texture, half-res target |
  | Volumetric clouds march | quarter-res, samples depth; too costly at full res in-line |
  | Scene-apply (AO × color + cloud composite) | reads the half-res AO/cloud textures back onto the resolved HDR |
  | Blit `_sceneHDR` → copy | the water pass can't sample the target it renders into |
  | Water + SSR | the stated original exception — resolved opaque color + depth as textures (refraction, SSR march, manual depth test) |
  | Volumetric light march + composite | half-res, marches the shadow cascades against reconstructed depth, additive |
  | Exposure + flare probe (compute) | GPU reductions into persistent state, no raster target |
  | Bloom pyramid | the multi-pass HDR consumer; skips wholesale at zero intensity |
  | Final composite | the one tonemap/grade; always runs (see rendering-conventions §6) |
  | UI overlay | display-resolution, drawn after everything |

- Bloom skips entirely at zero intensity (a static black fallback binds so the composite PSO never forks). Every other effect is a **pass-skip or a uniform flag** driven by `GraphicsSettings`; consumers bind a 4×4 white/black fallback texture when an effect is off, so no pipeline forks per quality level and a toggled-off effect costs nothing.
- Uniform-sized state (`ChunkOrigin`, `EntityModel`) rides `setVertexBytes`; real buffers are for real data.
- The UI batches every quad and glyph of a frame into one draw call.

## 5. Determinism enables measurement

All gameplay randomness flows from the world seed through `common/random.hpp` (`SeededRng`, `hashCoords`), and worldgen sub-seeds through the table in `gen_seeds.hpp`. **Why:** weather particles seeded from `std::random_device` and AI shared a mutating global `mt19937`, so no two runs of "the same world" behaved alike — unmeasurable and unreproducible.

## Known debts

- LOD levels 1/2 exist but don't render (geometry emitted at grid scale, cache misses on level switches). The scheduler and registry are LOD-ready; reintroduce only with world-scaled coarse meshes and per-level cache eviction. This is the lever for view distance 24+ (measured ~29 ms/frame full-detail).
- Frustum culling is per-chunk AABB only; no occlusion culling.
- `getLoadedChunks` copies the shared_ptr vector under `chunksMutex_` every frame; fine at vd ≤ 16, a candidate for an epoch/revision cache beyond that.

## Review checklist

For any diff touching the frame loop, `gameTick`, meshing, generation, or locks:

1. New work on the per-frame or per-tick path: what is its cost as calls × frequency, and does it fit the budgets table?
2. Any allocation, string building, or I/O on a hot path or under `chunksMutex_`?
3. New unbounded collection (cache, queue, entity list): where is its eviction or cap?
4. New randomness: seeded through `common/random.hpp`, or is determinism deliberately broken (say why)?
5. New render pass or per-draw buffer: justified against the sanctioned-pass list in §4 (needs a resolved texture, a different camera, or a lower-res target), does it skip cleanly when its setting is off, and does its cost keep the frame inside the ≤ 12 ms GPU budget (`RYCRAFT_GPU_COUNTERS=1`)?
6. New lock: where does it sit in the ordering table above, and can it ever be held while waiting on another thread?
7. Claimed speedup: shown in the F3 HUD numbers or a measured before/after, not vibes?
