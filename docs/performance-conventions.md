# Performance Conventions

Budgets and mechanics rules. As with the rendering doc, each rule's "why" is a defect found in this codebase. The `perf-review` skill walks this file's checklist.

## Budgets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS sustained at native (Retina) resolution at view distance 12–16 |
| Simulation tick | 20 Hz fixed timestep; a tick must fit well inside 50 ms |
| Chunk generation | ~4–5 ms per chunk on the 4-worker pool, `MAX_INFLIGHT_GEN = 32` submission window |
| Chunk meshing | 2 mesh workers, `MAX_INFLIGHT_MESH = 8`; render thread uploads only (24 meshes / 8 MB per frame) |
| View distance | 12 chunks default, settings up to 32; the mega-buffer sizes itself from the setting (~128 KB vertices per visible chunk + 30% headroom) |
| Memory | < 4 GB RSS |
| Animals | 64 spawned cap, 96-block simulation distance |

The F3 debug HUD shows the real numbers (EMA frame time, gen/mesh ms, pending count, mega-buffer usage), and the 60-frame diagnostic log line mirrors them so headless playtests can measure. **Why:** it used to hardcode `fps = 60.0` — a perf HUD that lies is worse than none.

Measured (M4 Max, 2026-07): vd 12 → 16.7 ms/frame (vsync), vd 16 → 16.7 ms, vd 24 → ~29 ms (opt-in; far-ring LOD is the known lever).

## 1. Hot-path allocation

- **No heap allocation per block or per chunk access.** Chunk keys are the packed `ChunkPos` value type. **Why:** `World` built a `std::string` key via `ostringstream` on every `getBlock` — an allocation on the hottest path in the game.
- **Per-frame scratch is reused members**, e.g. the renderer's `_liveChunkKeys` sweep set, `_waterDraws`, `_pendingResults`, and the UI overlay's vertex vector. Mesh workers keep `thread_local` snapshot + scratch buffers. **Why:** each mesh build used to reserve ~2.25 MB of fresh vectors — ~36 MB of allocation traffic per frame during streaming bursts.
- The mesher's block accessor is a **template parameter**, not `std::function`. **Why:** ~1M indirect calls per chunk build was the documented hot-path debt; templating removed it.

## 2. Lock discipline

Lock ordering (a thread may only take locks left → right; the leaves are never held together):

```
pendingMutex_ → chunksMutex_        (World)
MeshScheduler::jobMutex_             leaf
MeshScheduler::completedMutex_       leaf
MegaBuffer::_mutex                   leaf
SaveManager::saveMutex_              leaf
```

- **Never generate, load, or perform I/O under `chunksMutex_`** — it's on the render thread's path. Release, do the work, re-insert with `try_emplace`; a rare duplicate generation beats a guaranteed stall. **Why:** `getChunk` once generated whole chunks inside the lock.
- The one sanctioned exception: `World::snapshotForMeshing` performs a **bounded ~83 KB memcpy** under `chunksMutex_` (microseconds). Blocks mutate only before a chunk is inserted or under this mutex, so the copy is always internally consistent — after it, meshing runs lock-free on private data.
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

- One MSAA scene pass; sky, terrain, entities, particles, and clouds share its encoder. The **water pass** is the stated exception: it needs the resolved opaque color + depth as textures (refraction, manual depth test), which tile memory cannot provide mid-pass. Adding any other pass needs the same kind of reason.
- Bloom is the only multi-pass consumer and skips entirely at zero intensity (a plain blit replaces it).
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
5. New render pass or per-draw buffer: justified against the single-scene-pass shape (the water pass documents the accepted exception)?
6. New lock: where does it sit in the ordering table above, and can it ever be held while waiting on another thread?
7. Claimed speedup: shown in the F3 HUD numbers or a measured before/after, not vibes?
