# Performance Conventions

Budgets and mechanics rules. As with the rendering doc, each rule's "why" is a defect found in this codebase. The `perf-review` skill walks this file's checklist.

## Budgets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS sustained at native (Retina) resolution |
| Simulation tick | 20 Hz fixed timestep; a tick must fit well inside 50 ms |
| Chunk mesh build | ≤ 16 builds per frame (`MAX_MESH_BUILDS_PER_FRAME`); bursts amortize |
| View distance | 12 chunks default (25×25 grid ≈ 60 MB of the 128 MB mega-buffer) |
| Memory | < 4 GB RSS |
| Animals | 64 spawned cap, 96-block simulation distance |

The F3 debug HUD shows the real numbers (EMA frame time, live chunk/entity counts). **Why:** it used to hardcode `fps = 60.0` — a perf HUD that lies is worse than none.

## 1. Hot-path allocation

- **No heap allocation per block or per chunk access.** Chunk keys are the packed `ChunkPos` value type. **Why:** `World` built a `std::string` key via `ostringstream` on every `getBlock` — an allocation on the hottest path in the game.
- **Per-frame scratch is reused members**, e.g. the renderer's `_liveChunkKeys` sweep set and the UI overlay's vertex vector (which `reserve`s once).

## 2. Lock discipline

- **Never generate, load, or perform I/O under `chunksMutex_`** — it's on the render thread's path. Release, do the work, re-insert with `try_emplace`; a rare duplicate generation beats a guaranteed stall. **Why:** `getChunk` once generated whole chunks inside the lock.
- Accepted trade-off: two threads may occasionally generate the same chunk; `try_emplace` keeps one and drops the other. Document any new instance of this pattern.

## 3. Simulation cost scales with what the player can perceive

- Chunk streaming is asynchronous on the four-worker pool; the tick never blocks on generation. Completed futures are pruned by readiness (`wait_for(0)`), not `valid()` — **why:** `valid()` stays true until `get()`, so the pending map only ever grew.
- Animals beyond 96 blocks skip AI and physics entirely; the initial population is capped at 64 in a 7×7-chunk ring. **Why:** biome densities across the full view distance once spawned 5,915 animals and the tick collapsed.
- Meshes of unloaded chunks are swept every frame and their mega-buffer allocations freed. **Why:** the cache never evicted, so long play sessions exhausted the buffer.

## 4. GPU workload shape

- One MSAA scene pass; sky, terrain, entities, particles, and clouds share its encoder. Adding a render pass needs a stated reason the scene pass can't absorb it.
- Bloom is the only multi-pass consumer and skips entirely at zero intensity (a plain blit replaces it).
- Uniform-sized state (`ChunkOrigin`, `EntityModel`) rides `setVertexBytes`; real buffers are for real data.
- The UI batches every quad and glyph of a frame into one draw call.

## 5. Determinism enables measurement

All gameplay randomness flows from the world seed through `common/random.hpp` (`SeededRng`, `hashCoords`). **Why:** weather particles seeded from `std::random_device` and AI shared a mutating global `mt19937`, so no two runs of "the same world" behaved alike — unmeasurable and unreproducible.

## Known debts

- The mesher routes block reads through `std::function` accessors (~1M indirect calls per chunk build). Within budget at view distance 12; the fix if it surfaces is templating the accessor.
- LOD levels 1/2 exist but don't render (geometry emitted at grid scale, cache misses on level switches). Reintroduce only with world-scaled coarse meshes and per-level cache eviction.
- Frustum culling is per-chunk AABB only; no occlusion culling.

## Review checklist

For any diff touching the frame loop, `gameTick`, meshing, generation, or locks:

1. New work on the per-frame or per-tick path: what is its cost as calls × frequency, and does it fit the budgets table?
2. Any allocation, string building, or I/O on a hot path or under `chunksMutex_`?
3. New unbounded collection (cache, queue, entity list): where is its eviction or cap?
4. New randomness: seeded through `common/random.hpp`, or is determinism deliberately broken (say why)?
5. New render pass or per-draw buffer: justified against the single-scene-pass shape?
6. Claimed speedup: shown in the F3 HUD numbers or a measured before/after, not vibes?
