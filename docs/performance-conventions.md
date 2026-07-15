# Performance Conventions

These budgets and mechanics apply to the cubic world, the 256-chunk visible horizon, and the HDR renderer. The `perf-review` skill walks the checklist at the end of this document.

## Reference target and evidence standard

The acceptance target is a lowest sustained one-second rate of at least 60 FPS at native display resolution with 4x MSAA and a 256-chunk visible horizon on an Apple M4 Max. Exact editable cubic simulation stops at radius 32; immutable far-terrain LOD fills the half-open annulus `[32, 256)`. Total process and Metal consumption must remain at or below 64 GB of Apple Silicon unified memory.

Record the exact M4 Max configuration, display resolution, macOS version, optimized build commit, seed, spawn, route, settings, and measurement duration. The canonical regression begins from the user-reported seed-764891 view:

```bash
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=256 \
./build/src/rycraft
```

After the static view settles, repeat it with `RYCRAFT_AUTOPILOT=fly` and a recorded warmup, movement interval, stop frame, and performance window so the same route crosses new chunks and then measures queue recovery. Metal validation runs separately because its overhead does not belong in a performance measurement. Hardware timing and memory are acceptance evidence from the reference machine, not pass or fail conditions for portable CI. CI enforces deterministic work limits, queue caps, cache bounds, and allocation invariants.

The normal Meson build defaults to `debugoptimized`, retaining symbols and assertions while avoiding an O0 generation path that is not representative or practical for interactive play. Existing build directories retain their configured mode, so reconfigure an older directory explicitly. Final M4 Max acceptance still uses a separate `release` build.

| Metric | Target or hard limit |
|---|---|
| Frame rate | Lowest sustained one-second rate of at least 60 FPS on the settled moving route at view distance 256 |
| Frame spikes | No sustained generation-related frame time above 20 ms |
| Simulation | Fixed 20 Hz; ordinary work fits well inside 50 ms |
| Unified memory | At most 64 GB total process and Metal use on the reference machine |
| Warm cube generation | At most 2 ms p95 |
| Cold basin construction | At most 100 ms p95 on a worker |
| Streaming recovery | Exact generation, exact mesh, and far-tile queues settle within 5 seconds after movement stops |
| Exact simulation radius | At most 32 chunks horizontally |
| Loaded exact cubes | Hard cap of 32,768, including retained halo cubes |
| Exact mesh candidates and residency | Hard cap of 16,384 |
| Exact generation work | Four workers, at most two cold plans and 64 cube jobs in flight |
| Exact mesh work | Two workers and at most 64 total queued, building, completed, or renderer-pending items |
| Far terrain | Four workers, 64 pending, 32 completed, and 1,024 cached tiles |
| Fluid work | 1,024 cells per fluid tick, 65,536 pending, and 65,536 deferred |
| Animals | 64 living animals including babies; 96-block active and 112-block despawn radii |

Apple Silicon uses unified memory. Process RSS, current Metal allocation, peak Metal allocation, and resident-set counters can overlap. Report each observed counter and the highest credible unified-memory total without adding overlapping counters as if they were separate physical pools.

The F3 HUD and periodic diagnostic line report CPU and GPU frame time, exact generation and meshing, far residency and culling, queues, entities, fluids, caches, and arena use. A bounded performance capture also reports fixed-tick p50, p95, and maximum duration, or explicitly states that a paused capture ran no gameplay ticks. Record frame p50 and p95, the lowest sustained one-second frame rate, fixed-tick timings, unified-memory peak, queue maxima, loaded-cube maximum, exact mesh maximum, far resident and drawn maxima, culling counts, and settle time.

At the same settled camera and settings, record a playing interval and then a paused interval. `RYCRAFT_AUTOPAUSE_FRAME` enters the real paused screen at a fixed frame so both windows can start from the same streamed scene. Attribute fixed-tick CPU p50, p95, and maximum with the bounded performance capture. Pausing may remove simulation work, but it must not conceal a material main-thread bottleneck or produce an unexplained frame-rate multiplication. Investigate any large gap before treating the moving-route result as accepted.

## Measured baselines and current validation status

Measurements must be labeled by the path they exercised. None of the following CPU microbenchmarks proves the final 60 FPS view-distance-256 target.

On July 14, 2026, a local standalone seed-42 CPU microbenchmark ran on a 16-core Apple M4 Max MacBook Pro with 128 GB installed memory and macOS 26.6. It constructed 16 far tiles at each step serially:

| Far sample step | Geometry samples per tile | Mean CPU build | Mean vertices | Mean retained bytes |
|---:|---:|---:|---:|---:|
| 4 blocks | 4,225 | 1.479 ms | 17,262 | 721,096 |
| 8 blocks | 1,089 | 1.028 ms | 4,624 | 180,424 |
| 16 blocks | 289 | 0.855 ms | 1,292 | 45,256 |

The same workspace's coarse hydrology loop sampled 263,169 points at 16-block spacing in 0.338 seconds, about 779,000 samples per second, with approximately 35 MB maximum RSS. These are standalone CPU characterizations. They exclude scheduling, GPU upload, exact cubes, shadows, SSAO, volumetrics, water, post processing, entities, the game loop, and camera movement. The benchmark executable's compile flags were not captured, so use the results to understand relative tile cost, not as a release acceptance result.

The latest renderer on `origin/main`, before the cubic and far-terrain integration, measured about 4 ms total GPU time at native Retina resolution and view distance 16 with its Sildur-style effects enabled. Its full-detail view-distance-24 path measured about 29 ms per frame. Those numbers preserve a useful renderer baseline, but they do not validate the new integrated path or a 256-chunk horizon.

On July 14, 2026, the release build of code commit `2bb5b62` passed the integrated acceptance route on a Mac16,5 MacBook Pro with a 16-core CPU, 40-core Apple M4 Max GPU, 128 GB of unified memory, and macOS 26.6. The native drawable was 3456 by 2234 with 4x MSAA and view distance 256. After exact and far residency settled, the seed-764891 aerial route started at X=23029, Y=400, Z=-111726, traveled 141 blocks during frames 9000 through 10199, stopped, and measured 2,400 frames from frame 9000. The route reported:

| Integrated metric | Release result |
|---|---:|
| Frame time p50 / p95 / maximum | 8.321 / 8.980 / 14.199 ms |
| Lowest sustained one-second frame rate | 118.68 FPS |
| Frames above 20 ms | 0 |
| GPU frame-time EMA maximum | 6.404 ms |
| Fixed tick p50 / p95 / maximum | 0.029 / 2.447 / 4.460 ms |
| Loaded / meshed cube maximum | 27,348 / 16,384 |
| Generation / mesh queue maximum | 449 / 64 |
| Far wanted / resident maximum | 865 / 865 |
| Far drawn / frustum / horizon maximum | 239 / 610 / 56 |
| Far pending maximum | 3 |
| Queue settling after movement | 0.000 s |
| Process RSS / Metal allocation | 2,867.0 / 4,964.4 MiB |
| Highest credible unified-memory use | 4,964.4 MiB |

The same release binary also reproduced the user's seed-764891 X=23029, Y=225, Z=-111726 camera at the native drawable. A short bounded probe measured 8.333 ms p50, 8.829 ms p95, 12.454 ms maximum, and 119.57 lowest sustained one-second FPS. A paired settled paused capture measured 8.333 ms p50, 8.340 ms p95, and 119.82 lowest sustained one-second FPS with no gameplay ticks, so pausing did not conceal the previously reported twofold frame-rate gap. The deterministic component inspector measured a 0.879 ms warm-cube p95 and a 10.761 ms cold basin build. Metal API and GPU validation were disabled for timing and enabled separately for the visual acceptance captures.

## 1. Exact cubic active-set cost

World simulation cost must scale with visible or explorable three-dimensional space, not with the 256-chunk horizon or the complete 40-section vertical range.

- Exact streaming uses `min(viewDistance, 32)` plus its bounded apron.
- Surface streaming selects exposed sections around sampled terrain, water, cliffs, vegetation, and waterfalls.
- The camera exploration band is bounded to radius six horizontally and four cubes vertically. It is the highest-priority mesh and retention class, ahead of exposed surface sections, so an underground player cannot outrun nearby cube loading when global caps apply.
- Visible column manifests add saved edited sections.
- A targeted one-cube, 26-neighbor halo supplies collision, six-face lighting, the 18 by 18 by 18 block, fluid, and block-light snapshot, and separate 18 by 18 generated-surface and skylight cutoffs. If a scheduled neighbor has not arrived, cells below its generated surface remain conservatively opaque while cells above it remain air. Only below-surface openings receive dark inward caps; visible uphill continuations receive lit surface-material faces. Load completion dirties every affected mesh.
- The exact mesh-candidate set is capped at 16,384 and the retained loaded set at 32,768.
- Priority is camera exploration and collision, exposed surface, saved edits, then nearest full three-dimensional distance.
- Existing cubes remain retained through two extra horizontal chunks and one extra vertical cube, preventing unload oscillation without expanding exact simulation to the far horizon.

Every active-set change must state the maximum number of columns sampled, plans touched, cubes retained, halo cubes added, and jobs submitted. Multiplying the full 256-chunk horizon by the complete vertical range is a violation.

One rebuild gathers the unique horizontal exact-column set and expands its fixed plan apron once. It must not issue the same apron-plan request separately for every surface or halo cube. Pending plans register dependent active-set columns in an index. Completion wakes only those dependents, and multiple completion notifications coalesce into one pending rebuild instead of scanning up to the complete retained-cube cap for every plan. Cold column plans are constructed on the four-worker pool with at most two active at once. The main thread may query the cache and assemble bounded sets, but it must not construct a missing plan or run cold hydrology while rebuilding the active set.

Visible edited sections use one bulk manifest lookup for the unique visible-column list. The manifest map lock covers only in-memory lookup and copying. Plan construction, file access, manifest serialization, and active-set sorting remain outside it.

## 2. Far-terrain workload

Far terrain is an immutable rendering approximation, not an expansion of exact simulation.

- Tiles cover 256 by 256 blocks in the half-open annulus `[32, 256)` chunks.
- A narrow two-block sampling tier immediately outside radius 32 samples exact emitted density heights as the topology bridge.
- Farther-out tiles select among four-, eight-, and sixteen-block topology tiers.
- Distance supplies a tunable baseline rather than rigid rings, and immutable tile complexity from maximum sampled slope and hydrology biases detailed geometry outward within a fixed bound.
- The previously selected tier applies asymmetric refine and coarsen thresholds, preventing camera motion from chattering at a boundary.
- A resident tile stays visible until its replacement uploads. A bounded 0.4-second transition fades the old topology into fog, swaps at the hidden midpoint, and fades the new topology back out.
- At most 64 topology transitions are active, with one tier per coordinate drawn at a time.
- Tile borders are globally aligned and transition skirts hide cracks.
- Equal flat terrain and fully wet water cells merge greedily. Partially wet cells use contour-clipped shoreline triangles.
- Every tile performs one bounded canopy query while it is built. Steps two and four reconstruct all exact accepted tree anchors. Steps eight and sixteen query globally anchored 32- or 64-block aggregate forest cells whose climate, substrate, slope, water, and counter-addressed acceptance preserve distant canopy mass without exact priority competition. The result remains part of the cached tile and never runs per frame.
- Four workers build coordinate-pure tiles.
- The scheduler caps pending work at 64, completed work at 32, cache entries at 1,024, and cache bytes at 512 MiB.
- A camera jump advances the scheduler epoch so queued and completed stale work is discarded.
- A coordinate keeps its resident LOD until the desired replacement uploads, then uses the fog-hidden transition without a hole or topology pop.
- The render thread uploads at most 12 far tiles and 32 MiB in one frame.
- The far GPU arena reserves 256 MiB of vertex storage and 128 MiB of index storage.

Candidate tiles are sorted front to back. Conservative AABB frustum culling runs before a 256-azimuth-bin terrain-horizon test. Each visible tile contributes sixteen 64 by 64-block heightfield patches, whose minimum elevation is a conservative lower horizon. Only bins fully covered by a nearer patch can contribute an occlusion decision, and bin iteration uses fixed stack state with no per-tile allocation. This design is an adaptive tiled LOD informed by geometry clipmaps and CDLOD, not a literal geometry clipmap. The terrain-horizon test is not hierarchical Z and allocates no depth pyramid. Draw submission uses bounded direct indexed draws, not Metal indirect command buffers.

Far terrain may reduce visible detail but cannot affect collision, edits, fauna, fluids, saves, or deterministic exact cube output. Its tree boxes are visual-only exact-anchor or aggregate forest summaries. A far-cache eviction changes cost only. Rebuilding the same key must reproduce the same mesh and canopy hash.

The two-block sampling tier is a rendering dependency, not an expansion of exact simulation. Its cost must remain inside the same far cache, scheduler, upload, and GPU-arena bounds. Tests compare its boundary samples directly against exact emitted density heights. Exact opaque terrain draws before depth-biased far tops, which remain as lit fallback while exact meshes are cold. Water and canopies retain exact ownership through radius 32, then use a stable world-space dither over the following 16 blocks. Per-draw edge masks expose skirts only on resident finer-to-coarser boundaries outside that band. Playtests inspect these mechanisms for cracks, duplicated ownership, water walls, cold-residency rings, and topology pops.

Canopy reconstruction is likewise a far-build dependency. Review its anchor-cell fanout, priority comparisons, emitted impostor quads, and retained bytes at every tier. Bit 28 retains canopy classification, while the shared exact-to-far predicate prevents far impostors from doubling exact trees without creating a second mesh or draw list.

## 3. Hot-path allocation

- Do not allocate per block lookup, density sample, fluid neighbor, coordinate conversion, or far surface sample.
- Coordinate keys are `ColumnPos`, `ChunkPos`, `BlockPos`, and `FarTerrainKey`, never strings or lossy packed values.
- Uniform air and stone cubes remain allocation-light until edited. Do not materialize 4,096 block or fluid values only to discover that every value is identical.
- Per-frame and per-worker scratch is reused. This includes loaded-key sets, water draws, mesh candidates, completed results, snapshots, mesher masks, far candidates, and far results.
- The exact mesher's accessor remains a template or inline callable. Do not introduce `std::function` into a per-voxel loop.
- The renderer consumes a revision-cached loaded-world snapshot. Do not add another chunk-map copy or lock-taking traversal in the same frame.
- Exact and far mega-buffer growth is a settings or capacity event, never routine per-frame behavior.

Count allocations structurally. One allocation per cube across 16,384 meshes or one allocation per far sample across hundreds of tiles is not a small change. A canopy result vector allocated once per cold far-tile build belongs in the measured tile construction cost and cache payload, not in the frame loop.

## 4. Lock discipline

Lock ordering is:

```text
World::activeSetBuildMutex_ -> World::pendingMutex_ -> World::chunksMutex_
World::activeSetRequestMutex_                    leaf outside active-set publication
World::lightMutex_                              leaf below chunksMutex_
MeshScheduler::jobMutex_                        leaf
MeshScheduler::completedMutex_                  leaf
FarTerrainScheduler job/completed/cache mutexes leaf and never nested
MegaBuffer::_mutex                              leaf
SaveManager::saveMutex_                         leaf
SaveManager::manifestWriteMutex_ -> SaveManager::manifestMutex_
ColumnPlanCache mutex                           leaf
BasinSolver cache mutex                         leaf
ColdBasinBuildLimiter permit mutex              leaf and never nested
```

- Never generate, build a column plan, solve drainage, load, serialize, compress, or perform I/O under `chunksMutex_`.
- `snapshotForMeshing` may copy one bounded 18-cube-edge block, fluid, and block-light halo plus a separate 18 by 18 sky-cutoff array while holding `chunksMutex_`. Meshing itself runs after release.
- Never hold a lock while waiting for a future whose task can acquire that lock.
- Duplicate generation is an accepted escape from a main-thread stall. Reinsert through `try_emplace` so a late worker never overwrites edits.
- A single-flight cache may publish a future under its own mutex, but expensive construction happens outside that mutex. Concurrent callers for one key share the future.
- The basin solver releases its cache mutex before waiting on an existing future or the process-wide cold-build permit. The permit mutex is never nested with the basin-cache mutex.
- The far scheduler never constructs a tile while holding its job, completion, or cache mutex.
- The Core Audio callback holds `_voiceMutex` for mixing. `playSound` prepares allocations before that lock and swaps in constant time.

Any new lock must be placed in the ordering table and reviewed for waits, callbacks, allocation, and file access inside its scope.

## 5. Bounded generation and caches

Every coordinate query must have a compile-time or configuration bound.

- Plate lookup searches a 3 by 3 neighborhood of 8,192-block cells.
- Hotspot lookup searches a 3 by 3 neighborhood of 16,384-block cells.
- Basin keys cover 2,048 by 2,048 blocks and route through shared cardinal portals.
- Portal discharge uses four exact upstream site rings plus one deterministic coarse distant term.
- Basin fields use a 16-block raster with a two-cell apron. Input callbacks sample a globally aligned 64-block grid and interpolate onto that raster. Shared boundaries use a four-cell blend and exact portal reconstruction.
- Priority-Flood and angular two-neighbor D-infinity-inspired routing run before and after erosion.
- Erosion has exactly eight stream-power, sediment-capacity, deposition, and thermal-relaxation passes.
- Strahler ordering, qualifying spill lakes, waterfall flags, outlet-fall primitives, and two through four delta distributaries are extracted from finished routing.
- Lake-body depth interpolation retains dry and different-body weights as zero-depth contributors rather than renormalizing the wet subset. Finished lake floors equal the flat water surface minus positive depth, and a supported dry rim leaves named outlets and active channels open.
- A crater lake uses a warped absolute local profile, validates its complete rim in 96 directions with one block of freeboard, supports the full dry bank, and is rejected when a safe wet radius cannot fit. Ordinary routed lakes may retain named outlets.
- Each column plan adds one compact 17 by 17 canonical lake authority to its nine macro samples and 256 exact surface values. Ambiguous lattice cells may make a bounded exact hydrology query during cold plan construction; per-frame and per-cube hot paths may not reconstruct that authority.
- Moisture transport evaluates 17 points over 16 intervals of 256 blocks. Ocean, lake, and river recharge weights are 1.0, 0.65, and 0.18.
- Column plans retain nine full macro samples on a 3 by 3 lattice at eight-block spacing and a 256-column exact density surface grid. Construction performs 16 transient height-only perimeter queries for neighboring feature reach. Their cache holds at most 8,112 plans under a compile-time 64 MiB payload bound.
- Basin solutions use a separate byte-accounted 64 MiB single-flight LRU cache. Exact and far generators share one process-wide gate that permits at most two cold solution builds, while cache hits and same-key future waiters consume no additional permit.
- Far terrain uses a separate 1,024-entry, 512 MiB CPU cache.
- Ore, structure, tree, and flora candidate counts are fixed.

Cache payload limits do not include transient solver storage, hash-table overhead, shared ownership, or allocator fragmentation. Include those costs in the unified-memory measurement.

The basin solver contract requires fixed catchment extent, apron, input grid, numerical spacing, pass count, no more than two process-wide cold solution builds across exact and far generators, single-flight construction, byte-accounted LRU eviction, finite-field and outlet validation, an un-eroded base fallback with deterministic outlet metadata, and no wait under the world or basin-cache mutex. A cold caller may wait only on the dedicated construction permit or an existing shared future. One `BasinSolver` instance also represents one immutable callback context. Its coordinate-pure elevation, rainfall, and rock-resistance fields must not change during its lifetime because the cache key does not include callback identity. Do not describe its angular two-neighbor routing as a verbatim Tarboton triangular-facet implementation.

## 6. Queue, fluid, and population bounds

- Exact generation is nearest-first. Rebuilding the backlog reprioritizes work, and at most 64 cube jobs may be in flight.
- Exact mesh candidates sort by three-dimensional distance. At most 64 items exist across queued, building, completed, and renderer-pending states, and duplicate cube results coalesce to the newest revision. Render-thread exact uploads stop after 64 meshes or 32 MiB in a frame, with two uploads and 4 MiB reserved for nearby edits.
- Far candidates sort front to back. The scheduler caps pending, completed, cache, and per-frame upload work as described above.
- Finished results, pending uploads, save queues, and GPU registries require eviction or backpressure. A producer cap alone does not bound an unconsumed completion vector. The save queue caps at 32,768 cubic positions, coalesces repeated snapshots for one queued position to its newest revision, and applies backpressure rather than accepting a 32,769th unique position.
- Runtime water is deduplicated by block position, delayed five ticks, and limited to 1,024 processed cells per tick. Pending updates and deferred frontiers each cap at 65,536. Long-frame recovery runs no more than eight catch-up ticks.
- Every standing generated water voxel from its supported floor through its surface is an implicit source, including across cube boundaries. Those volumes allocate no explicit fluid state until runtime disturbance changes a cell.
- Ordinary generation and loading enqueue no fluid work. Only a gameplay edit or a matching previously activated frontier can introduce work.
- Fluids query loaded cubes only. Propagation must never call a force-loading world accessor.
- Deferred frontiers are indexed by unavailable destination cube. A fixed-tick resume budget considers only newly available index buckets and cannot scan all 65,536 frontiers once for every loaded cube.
- Stable source and flowing cells emit top geometry only. Vertical sides are exclusive to explicit falling cells, and far shorelines use contour-clipped triangles instead of rectangular water sheets.
- A lake outlet fall is a separate receiver-centered primitive with top, bottom, width, flow, and anchor data. Exact emission produces only its short falling footprint. The anchor's half-open far tile owns one five-quad prism. Neither representation raises the receiving body's standing water or runs generation-time fluid ticks.
- Far generated source water uses the same 0.875-block surface plane as the exact implicit source voxel.
- Wild territories reevaluate once per second or after meaningful travel. AI and physics stop beyond 96 blocks, despawn begins beyond 112 blocks, and babies count toward the exact 64-animal limit.

Pending-update and deferred-frontier drop counts must appear in F3 diagnostics. Save and mesh coalescing counts must remain observable through their statistics interfaces and regression coverage. Silently exceeding a bound and growing anyway defeats the bound.

## 7. Main-thread and GPU workload

Terrain generation, cold macro construction, active-set reconstruction, exact unload scans, far tile construction, save compression, and ordinary exact mesh builds run off the render thread and fixed-tick thread. Player movement submits a latest-wins active-set request in bounded time. The utility-priority planner cancels superseded work before publication; generation and far-terrain workers also use utility priority, while exact mesh workers use user-initiated priority. Plan completions notify only after 128 results or backlog drain, and fixed-tick consumption has a four-tick cooldown. The main thread may upload finished geometry only within its count and byte budgets. The edit fast path may rebuild at most two already-meshed near-camera cubes synchronously; first-time streaming never uses it.

Collision, ray targeting, breaking, and placement query loaded cubes only. Missing collision stays closed, and a ray aborts when it reaches an unavailable cube. Player interaction must never synchronously generate a cube or convert a dark temporary boundary into editable world content.

Opaque exact and far terrain uses greedy meshing, counterclockwise front faces, and back-face culling. The frame first rejects exact cubes and far tiles by frustum. Far survivors additionally pass through conservative terrain-horizon culling. Water and shadow casters use cull-none for correctness.

The block-texture array eagerly uploads all five mip levels from 16 by 16 through 1 by 1. Alpha-aware downsampling preserves representable cutout coverage. Sampling uses nearest magnification plus linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy. Do not regenerate mips per frame or introduce a second texture array for far terrain.

The bulk of shading remains one 4x MSAA scene pass. The fixed frame graph permits extra passes only when they need a different camera, resolved texture input, persistent compute state, or a lower-resolution target:

| Passes | Required dependency |
|---|---|
| Three shadow cascades and water-shadow slices | Light camera and depth-only targets |
| SSAO and bilateral reconstruction | Resolved depth and half-resolution targets |
| Volumetric clouds | Resolved depth and quarter-resolution march |
| Scene application | AO and cloud textures applied to resolved HDR |
| Opaque-color copy | Water cannot sample its active render target |
| Water and screen-space reflection | Resolved opaque color and depth |
| Volumetric light | Reconstructed depth and shadow cascades at half resolution |
| Exposure and flare probes | GPU reductions into persistent state |
| Bloom | Multi-resolution HDR pyramid |
| Final composite | Single tonemap, grade, flare, sharpen, and dither |
| UI | Display-resolution overlay after the graded scene |

GPU total frame time comes from `GPUEndTime - GPUStartTime`. Optional `MTLCounterSampleBuffer` timestamps attribute pass changes when `RYCRAFT_GPU_COUNTERS=1`; overlapping tile-based GPU stages can sum above wall-clock time, so do not add pass times to infer total time.

If the full path exceeds budget, preserve correctness and reduce cost in this order: lower cloud resolution or steps, reduce volumetric-light steps, refresh the far shadow cascade less often, lower SSAO resolution, reduce screen-space reflection steps, then evaluate a smaller HDR resolve format. Record the visual and timing effect of any downgrade.

## 8. Determinism enables measurement

Discrete stochastic generation choices are counter-addressed by seed, subsystem, full-width coordinates, and candidate index. Continuous Simplex fields use an immutable seed-derived permutation. Neither has mutable query-order state. Worker count, generation order, cache eviction, duplicate work, and far scheduler epoch must not change blocks, fluid states, plans, basin samples, feature anchors, territory IDs, or far mesh hashes.

Benchmark routes use fixed seeds and spawn coordinates. If a performance change alters generated work, compare work counts and golden hashes before comparing timing.

## Rendering research boundary

The far visibility design is informed by [Geometry Clipmaps](https://hhoppe.com/geomclipmap.pdf) and [Continuous Distance-Dependent Level of Detail](https://doi.org/10.1080/2151237X.2009.10129287). Rycraft uses adaptive immutable tile tiers selected from distance and terrain complexity, not either algorithm verbatim. Its conservative angular occlusion follows the front-to-back hierarchy principle described by [Hierarchical Z-Buffer Visibility](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/greene93_hierarchicalz.pdf), but it stores 256 horizon bins rather than a hierarchical depth image.

Apple's [indirect command buffer](https://developer.apple.com/documentation/metal/creating-an-indirect-command-buffer) documentation describes a future GPU-driven submission option. The present renderer deliberately uses bounded direct tile draws. Command-buffer lifetime and batching follow Apple's [Metal command-buffer best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html).

## Review checklist

For any change touching the frame loop, fixed tick, generation, streaming, meshing, far terrain, fluids, fauna, caches, or locks:

1. What is the new cost as calls times frequency times active cubes, tiles, cells, samples, or entities?
2. Does exact simulation remain within radius 32, 16,384 mesh candidates, and 32,768 retained cubes, with the radius-six by vertical-four exploration band outranking surface work and a targeted complete halo?
3. Can the 256-chunk visible setting accidentally expand exact generation, collision, entity, fluid, or save work?
4. Does exact unload retain two extra horizontal chunks and one extra vertical cube, and does far LOD preserve asymmetric hysteresis, or will ordinary movement churn a frontier or topology tier?
5. Does any main-thread path build a cold column plan, solve hydrology, generate a cube, construct a far tile, compress a save, or perform I/O?
6. Does any hot loop allocate, build a string, copy the loaded map, or materialize uniform storage?
7. Does every cache state its entry or byte cap, single-flight behavior, eviction policy, transient-memory caveat, and cache-eviction determinism test?
8. Does every solver state its domain, apron, resolution, iteration count, concurrency limit, validation, and fallback?
9. Does every queue state producer cap, consumer budget, deduplication or coalescing, overflow behavior, cancellation behavior, and diagnostic counter?
10. Can fluid work begin during generation or normal loading, cross an unloaded boundary without an indexed frontier, scan every frontier for one cube load, or force-load a cube?
11. Does a new lock fit the ordering table, and can its scope call a waiter, generator, allocator-heavy operation, or file API?
12. Do exact and far mesh residency, CPU caches, GPU arenas, transient solvers, and post targets fit the 64 GB unified-memory ceiling together?
13. Do the two-block exact-density topology tier, depth-backed opaque fallback, shared 16-block dithered handoff for water and canopies, resident finer-to-coarser skirt masks, adaptive distance-and-complexity 4/8/16 LOD, exact near canopies, aggregate distant forest clusters, frustum culling, back-face culling, conservative horizon occlusion, greedy exact meshing, bounded fog-hidden transitions, and mipmapped anisotropic texture sampling each remain enabled and covered?
14. Is any claim of HZB, geometry clipmaps, indirect command buffers, or GPU-driven visibility accurate to the implementation?
15. Does entity work stay inside the 96-block activation radius and exact 64-living cap?
16. Is new randomness coordinate-addressed or explicitly seed-derived, and do shuffled order and cache eviction preserve output?
17. Is a speed claim backed by a fixed-seed measurement with build, machine, p50, p95, residency, culling, memory, and queue-settle evidence?
18. Are stable exact and far water free of unsupported vertical walls, with full generated standing volumes represented by implicit sources, side geometry restricted to explicit falling columns, far shorelines contour-clipped, and the seed-764891 caldera enclosed by its validated irregular dry rim?
19. Was the canonical seed-764891 route measured at native resolution, 4x MSAA, and view distance 256 on the identified M4 Max, with its lowest sustained one-second rate at least 60 FPS and the 64 GB unified-memory ceiling checked?
20. Do unavailable in-range sections stay closed for collision and interaction, use lit planned silhouettes above ground and dark inward caps only underground, and block underground skylight until the vertical loaded path is continuous?
21. Does active-set rebuild expand each unique horizontal plan apron once, use indexed completion dependencies, coalesce rebuild notifications, and obtain visible saved sections through one bulk short-lock manifest read?
22. Do canonical lake samples retain the full 17 by 17 authority, taper dry contributions to zero, keep shore water supported, and represent a valid lake outlet as a narrow falling connection rather than a discarded body or raised receiver?
23. Does gameplay submit active-set work without executing it on the fixed tick, does the planner retain only the latest request and cancel stale epochs before publication, and do request, coalescing, cancellation, and build-time metrics remain observable?
24. At the same settled camera, does a playing-versus-paused comparison include fixed-tick CPU attribution and rule out an unexplained frame-rate multiplication caused by simulation or other main-thread work?
