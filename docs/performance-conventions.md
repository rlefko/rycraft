# Performance Conventions

These budgets and mechanics apply to the cubic world, the 512-chunk visible horizon, and the HDR renderer. The `perf-review` skill walks the checklist at the end of this document.

## Reference target and evidence standard

The acceptance target is a lowest sustained one-second rate of at least 60 FPS at native display resolution with 4x MSAA and a 512-chunk visible horizon on an Apple M4 Max. Exact editable cubic simulation has a nominal radius of 32, while immutable step-32 far-terrain parents cover the complete visible disk as residency dependencies. Every far-owned fragment in the exact overlap remains protected, including fragments in a fully ready tile with only partial boundary requirements. The camera exploration band requires visible step-2 fallback, and every other protected overlap fragment requires step 8 or finer. Total process and Metal consumption must remain at or below 64 GB of Apple Silicon unified memory.

Record the exact M4 Max configuration, display resolution, macOS version, optimized build commit, seed, spawn, route, settings, and measurement duration. The canonical regression begins from the user-reported seed-764891 view:

```bash
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=512 \
./build-release/src/rycraft
```

After the static view settles, repeat it with `RYCRAFT_AUTOPILOT=fly` and a recorded warmup, movement interval, stop frame, and performance window so the same route crosses new chunks and then measures queue recovery. Metal validation runs separately because its overhead does not belong in a performance measurement. Hardware timing and memory are acceptance evidence from the reference machine, not pass or fail conditions for portable CI. CI enforces deterministic work limits, queue caps, cache bounds, and allocation invariants.

The normal Meson build defaults to `debugoptimized`, retaining symbols and assertions while avoiding an O0 generation path that is not representative or practical for interactive play. Existing build directories retain their configured mode, so reconfigure an older directory explicitly. Configure the final M4 Max acceptance binary with `meson setup build-release --buildtype=release` and build it with `ninja -C build-release`.

| Metric | Target or hard limit |
|---|---|
| Frame rate | Lowest sustained one-second rate of at least 60 FPS on the settled moving route at view distance 512 |
| Frame spikes | No sustained generation-related frame time above 20 ms |
| Simulation | Fixed 20 Hz; ordinary work fits well inside 50 ms |
| Unified memory | At most 64 GB total process and Metal use on the reference machine |
| Warm cube generation | At most 2 ms p95 |
| Cold basin construction | At most 100 ms p95 on a worker |
| Cold horizon coverage | Every step-32 parent GPU resident within 2 seconds on the reference route |
| Streaming recovery | Exact generation, exact mesh, and far-tile queues settle within 5 seconds after movement stops |
| Exact simulation radius | At most 32 chunks horizontally |
| Loaded exact cubes | Hard cap of 32,768, including retained halo cubes, enforced at insertion |
| Exact mesh candidates and residency | Hard cap of 16,384 |
| Exact generation work | Six workers, at most two cold plans, and seven submitted cube tasks beneath the 64-job hard ceiling; remaining active-set work stays in the prioritized backlog |
| Exact mesh work | Four workers and at most 64 total queued, building, completed, or renderer-pending items |
| Far terrain | Eight workers, with four parent slots and four progressive slots while coverage is incomplete; 64 pending, 32 completed, and 9,280 cached meshes under 3 GiB; residency cleanup scans 64 records and retires 32 MiB per worker pass |
| Fluid work | 1,024 cells per fluid tick, 65,536 pending, and 65,536 deferred |
| Animals | 64 living animals including babies; 96-block active and 112-block despawn radii |
| Dropped items | 128 live item entities, 96-block active radius, 6,000-tick despawn; resting items skip the sweep, a merge pass runs once per second, and the oldest grounded item is evicted at the cap |
| Boats | 40 live boats, 128-block active radius (the ridden boat always ticks), oldest evicted at the cap; each is one buoyancy + drag + terrain-sweep step per tick, no aging or despawn |
| Furnaces | Player-placed only, ticked once per gameTick (and once per accumulated tick while a container screen is open); a handful expected, unbounded only by placement |

Apple Silicon uses unified memory. Process RSS, current Metal allocation, peak Metal allocation, and resident-set counters can overlap. Report each observed counter and the highest credible unified-memory total without adding overlapping counters as if they were separate physical pools.

The six exact-generation, four exact-mesh, and eight far-terrain workers expose up to 18 construction threads in addition to the planner and main thread. Each generation pool splits latency-sensitive and utility work, while bounded admission prevents those workers from exceeding the fixed pool sizes. This does not by itself prove frame isolation on the 16-core reference machine. Performance acceptance must record active worker maxima, CPU saturation, frame p95, and queue recovery while all three pools are busy.

The F3 HUD and periodic diagnostic line report CPU and GPU frame time, exact generation and meshing, exact required and ready coverage, the conservative nearest-gap distance, far base and refinement residency, the drawable coverage frontier, culling, separate queue lanes, entities, fluids, caches, and arena use. A bounded performance capture also reports fixed-tick p50, p95, and maximum duration, or explicitly states that a paused capture ran no gameplay ticks. Record frame p50 and p95, the lowest sustained one-second frame rate, fixed-tick timings, unified-memory peak, queue maxima, loaded-cube maximum, exact mesh maximum, base and refinement wanted, resident, drawn, and missing maxima, culling counts, frontier distance, cold parent completion time, and settle time.

At the same settled camera and settings, record a playing interval and then a paused interval. `RYCRAFT_AUTOPAUSE_FRAME` enters the real paused screen at a fixed frame so both windows can start from the same streamed scene. Attribute fixed-tick CPU p50, p95, and maximum with the bounded performance capture. Pausing may remove simulation work, but it must not conceal a material main-thread bottleneck or produce an unexplained frame-rate multiplication. Investigate any large gap before treating the moving-route result as accepted.

## Measured baselines and current validation status

Measurements must be labeled by the path they exercised. None of the following CPU microbenchmarks proves the final 60 FPS view-distance-512 target.

On July 14, 2026, a local standalone seed-42 CPU microbenchmark ran on a 16-core Apple M4 Max MacBook Pro with 128 GB installed memory and macOS 26.6. It constructed 16 far tiles at each step serially:

| Far sample step | Geometry samples per tile | Mean CPU build | Mean vertices | Mean retained bytes |
|---:|---:|---:|---:|---:|
| 4 blocks | 4,225 | 1.479 ms | 17,262 | 721,096 |
| 8 blocks | 1,089 | 1.028 ms | 4,624 | 180,424 |
| 16 blocks | 289 | 0.855 ms | 1,292 | 45,256 |

The same workspace's coarse hydrology loop sampled 263,169 points at 16-block spacing in 0.338 seconds, about 779,000 samples per second, with approximately 35 MB maximum RSS. These are standalone CPU characterizations. They exclude scheduling, GPU upload, exact cubes, shadows, SSAO, volumetrics, water, post processing, entities, the game loop, and camera movement. The benchmark executable's compile flags were not captured, so use the results to understand relative tile cost, not as a release acceptance result.

The latest renderer on `origin/main`, before the cubic and far-terrain integration, measured about 4 ms total GPU time at native Retina resolution and view distance 16 with its Sildur-style effects enabled. Its full-detail view-distance-24 path measured about 29 ms per frame. Those numbers preserve a useful renderer baseline, but they do not validate the new integrated path or a 512-chunk horizon.

On July 14, 2026, the release build of code commit `2bb5b62` passed the then-current integrated route on a Mac16,5 MacBook Pro with a 16-core CPU, 40-core Apple M4 Max GPU, 128 GB of unified memory, and macOS 26.6. The native drawable was 3456 by 2234 with 4x MSAA and view distance 256. After exact and far residency settled, the seed-764891 aerial route started at X=23029, Y=400, Z=-111726, traveled 141 blocks during frames 9000 through 10199, stopped, and measured 2,400 frames from frame 9000. The route reported:

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

The same release binary also reproduced the user's seed-764891 X=23029, Y=225, Z=-111726 camera at the native drawable. A short bounded probe measured 8.333 ms p50, 8.829 ms p95, 12.454 ms maximum, and 119.57 lowest sustained one-second FPS. A paired settled paused capture measured 8.333 ms p50, 8.340 ms p95, and 119.82 lowest sustained one-second FPS with no gameplay ticks, so pausing did not conceal the previously reported twofold frame-rate gap. The deterministic component inspector measured a 0.879 ms warm-cube p95 and a 10.761 ms cold basin build. Metal API and GPU validation were disabled for timing and enabled separately for the visual captures. This view-distance-256 result is a historical baseline, not acceptance evidence for the current view-distance-512 target.

Procedural field continuity is also not accepted on the current branch. Categorical former-control-line runs pass with a longest run of 9 blocks against the 24-block limit, but the continuous-field matrix has 15 failing assertions. Terrain derivative-energy ratios reach 0.105649 at the former 2,048-block spacing and 0.076197 at 8,192 blocks, the aggregate shoreline ratio is 0.194688, and shoreline structured orientation is 1.674842 against the 1.5 limit. Biome suitability fails at multiple spacings. Run the preserved acceptance matrix explicitly with `./build-release/tests/test_rycraft "[.known-continuity-debt]"`. Its expected failure is deferred technical debt, not evidence that any render-residency boundary failed.

## 1. Exact cubic active-set cost

World simulation cost must scale with visible or explorable three-dimensional space, not with the 512-chunk horizon or the complete 40-section vertical range.

- Exact streaming uses `min(viewDistance, 32)` plus its bounded apron.
- The camera exploration band is bounded to radius six horizontally and four cubes vertically. It is the highest-priority mesh and retention class, ahead of exposed surface sections, so an underground player cannot outrun nearby cube loading when global caps apply.
- Visible column manifests add saved edited sections.
- Every visible column reserves one primary surface section before additional exposed, water, cliff, vegetation, and waterfall sections.
- A targeted one-cube, 26-neighbor halo supplies collision, six-face lighting, the 18 by 18 by 18 block, fluid, and block-light snapshot, and separate 18 by 18 generated-surface and skylight cutoffs. If a scheduled neighbor has not arrived, cells below its generated surface remain conservatively opaque while cells above it remain air. A cheap boundary scan gates the bounded exterior-air flood. Sky-connected lateral openings receive lit provisional surface-material faces, enclosed lateral openings receive dark stone, and missing vertical openings receive bedrock. Complete loaded roof cutoffs are authoritative, while a distinct incomplete-path marker uses generated authority to avoid black aboveground frontiers. At most one nonlinear surface-material sample is taken for each missing lateral face. Load completion dirties every affected mesh.
- The exact mesh-candidate set is capped at 16,384 and the retained loaded set at 32,768. A published retention change unloads obsolete cubes before submitting replacements, and both asynchronous and synchronous insertion recheck the loaded cap under the chunk-map mutex.
- Four latency-sensitive and two utility-priority exact generation workers submit at most seven cube tasks at once, one per running worker plus one look-ahead task. This submission limit stays beneath the 64-job hard ceiling while remaining active-set work stays in the prioritized backlog, with no more than two cold column plans active. Epoch-first priority lets a camera jump pass queued work from its prior active set. A worker skips a cube that is stale for current retention; completion processing sends a still-required cube back through its current plan dependencies. Within one epoch, the camera column and six-chunk exploration band pass the broad radius-32 surface disk, and distance orders each lane.
- Four user-initiated exact mesh workers retain at most 64 total scheduler and renderer items. Broad surface work stops admitting at 32 items so camera-band jobs can enter later, and queued camera-band jobs run before broad work even when their plans complete later.
- Priority is camera exploration and collision, saved edits, one primary surface section per visible column, additional exposed and cliff sections, then the required halo. Nearest full three-dimensional distance resolves ties.
- The planner publishes planned exact surface and boundary requirements plus unresolved columns before caps. The renderer derives one ownership bit for each 16 by 16-block column from those requirements and its currently published exact meshes. Missing and unresolved requirements keep that column far-owned, and a previously published exact mesh may retain ownership during an ordinary replacement.
- Existing cubes remain retained through two extra horizontal chunks and one extra vertical cube, preventing unload oscillation without expanding exact simulation to the far horizon.

Every active-set change must state the maximum number of columns sampled, plans touched, cubes retained, halo cubes added, and jobs submitted. Multiplying the full 512-chunk horizon by the complete vertical range is a violation.

One rebuild gathers the unique horizontal exact-column set and expands its fixed plan apron once. It must not issue the same apron-plan request separately for every surface or halo cube. Pending plans register dependent active-set columns in an index. Completion wakes only those dependents, and multiple completion notifications coalesce into one pending rebuild instead of scanning up to the complete retained-cube cap for every plan. Cold column plans are constructed on the six-worker pool with at most two active at once. Plan and cube submissions carry an epoch, lane, and distance so a newly available camera dependency passes queued broad work and a cold plan never waits behind an entire cube burst. The main thread may query the cache and assemble bounded sets, but it must not construct a missing plan or run cold hydrology while rebuilding the active set.

Visible edited sections use one bulk manifest lookup for the unique visible-column list. The manifest map lock covers only in-memory lookup and copying. Plan construction, file access, manifest serialization, and active-set sorting remain outside it.

## 2. Far-terrain workload

Far terrain is an immutable rendering approximation, not an expansion of exact simulation.

- Tiles cover 256 by 256 blocks across the complete visible disk, including coordinates wholly inside the nominal exact radius.
- Every selected coordinate requests a step-32 parent before optional refinement. Missing parents use a broad nearest-first lane. Connected coordinates request their distance-selected step-16, step-8, step-4, or step-2 target before the complete parent disk is resident. Optional broad intermediates remain gated behind complete parent coverage.
- Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks missing step-32 dependencies. The drawable frontier also treats protected base-only tiles as missing, suppresses resident islands at or beyond the nearest such gap, and fades the preceding 256 blocks into fog.
- Active parents remain pinned while displayed, desired, or transitioning. Pressure evicts the farthest refinement first.
- Published requirements and unresolved columns produce a 16 by 16 per-column ownership mask for each far tile. Empty completed meshes count as ready. A column acquires ownership from currently published exact meshes, while an unresolved column remains far-owned. Every fragment that remains far-owned in the exact overlap stays protected, including fragments in a fully ready tile whose requirements cover only part of a boundary. Far ownership does not authorize step 32 there. The nearest-gap distance remains a conservative fallback and diagnostic rather than the primary fragment boundary.
- One-, two-, four-, eight-, sixteen-, and thirty-two-block `SurfaceFootprint` queries filter sub-Nyquist detail without changing hydrology topology, water elevation, plate ownership, or feature anchors.
- Distance supplies a tunable baseline rather than rigid rings, and immutable tile complexity from maximum sampled slope and hydrology biases detailed geometry outward within a fixed bound.
- The previously selected tier applies asymmetric refine and coarsen thresholds, preventing camera motion from chattering at a boundary.
- A far-owned fragment in the camera exploration band does not display until step 2 is ready. Every other far-owned fragment in the exact overlap does not display until step 8 or finer is ready. This includes a fully ready partial boundary tile, not only a tile with an unresolved requirement. Its step-32 parent remains resident but hidden. These protected jobs bypass ordinary grace and topology-transition admission. Ordinary resident tiles stay visible until a replacement uploads. Independently filtered tiers are not assumed to be a nested height pyramid, so a narrow terrain-only fog pulse hides one complete ordinary voxel topology swap and prevents mismatched partial sheets. Normal far-tier terrain swaps at the temporal midpoint. Unswayed world coordinates keep wind from moving the pulse or coverage boundary.
- At most 64 topology transitions are active. Canopies appear during the first half and source crowns retire during the second half. Skirts follow the currently visible terrain topology, while source water retains ownership until completion.
- Tile borders are globally aligned, and transition skirts join a resident finer far tier to a resident coarser far tier. They are separate from the exact mesher's implemented lit, dark, and bedrock missing-halo closure caps.
- Equal flat terrain and fully wet water cells merge greedily. Partially wet cells use contour-clipped shoreline triangles.
- Every tile performs one bounded canopy query while it is built. Step 2 queries accepted exact tree anchors without constructing ColumnPlans. Steps 4 through 32 query globally anchored 64-block aggregate forest cells with six fixed candidates and block-8 habitat and ground authority. Coarser aggregate tiers form strict stable subsets. At step 32, block-resolution collector habitat and root-water authority wins over the coarse cell's aggregate water bit. Water elsewhere in a 32 by 32 cell does not suppress an accepted tree, and the emitted trunk grounds on the displayed voxel. The result remains part of the cached tile and never runs per frame.
- Four latency-sensitive and four utility workers build coordinate-pure tiles. While any parent is queued, dispatch reserves four worker slots for base coverage and admits at most four urgent connected refinements. Already running base jobs finish without preemption. This bounded 4/4 admission policy keeps the coverage frontier moving while populating all visible 16/8/4/2 distance tiers.
- Exact generation and meshing retain their separate utility and user-initiated priorities without reducing the eight-worker far pool. While exact streaming is busy, refinement uploads are limited to four per frame while up to 32 parent uploads remain available. Otherwise up to 12 refinement uploads may advance. The complete far upload budget remains 32 MiB per frame.
- The scheduler caps pending work at 64, completed work at 32, cache entries at 9,280, and cache bytes at 3 GiB.
- A changed residency filter cancels only the bounded job and completion queues on the caller. Utility workers rebuild cache priorities and scan at most 64 cache records per maintenance pass. They retire at most 32 MiB of mesh payload per pass, except that one individually oversized mesh may retire alone to guarantee progress. Retired meshes and membership tables are destroyed after cache locks are released.
- A camera jump advances the scheduler epoch so queued and completed stale work is discarded.
- An ordinary coordinate keeps its resident LOD until the next staged replacement uploads, finishes that monotonic transition before reevaluating the desired tier, and cannot redirect mid-flight. Cached reentry initializes directly to the finest resident acceptable tier once its parent is resident. Protected exact-loading requests lead the urgent order: step 2 for every far-owned exploration-band fragment, then step 8 for every other far-owned exact-overlap fragment, including fragments in fully ready partial boundary tiles. A protected coordinate never displays step 32. Its base-only state counts as missing in the drawable frontier, preventing a farther resident island from appearing through the gap.
- The render thread uploads at most 32 parents or 12 refinements and 32 MiB in one frame. The exact-busy refinement limit is four.
- The far GPU arena allocates paired 256 MiB vertex and 128 MiB index slabs lazily, up to 2 GiB of vertex storage and 1 GiB of index storage.

Candidate tiles are sorted front to back. Conservative AABB frustum culling runs before a 256-azimuth-bin terrain-horizon test. Each visible tile contributes sixteen 64 by 64-block heightfield patches, whose minimum elevation is a conservative lower horizon. Only bins fully covered by a nearer patch can contribute an occlusion decision, and bin iteration uses fixed stack state with no per-tile allocation. This design is an adaptive tiled LOD informed by geometry clipmaps and CDLOD, not a literal geometry clipmap. The terrain-horizon test is not hierarchical Z and allocates no depth pyramid. Draw submission uses bounded direct indexed draws, not Metal indirect command buffers.

Far terrain may reduce visible detail but cannot affect collision, edits, fauna, fluids, saves, or deterministic exact cube output. Its canopy clusters are visual-only aggregate forest summaries. A far-cache eviction changes cost only. Rebuilding the same key must reproduce the same mesh and canopy hash.

Every far tier is a rendering dependency, not an expansion of exact simulation. Its cost must remain inside the same far cache, scheduler, upload, and GPU-arena bounds. Tests compare footprint samples against exact ownership and verify conservative parent bounds. Exact opaque terrain draws before depth-biased far tops. Step-32 tops remain available outside protected exact-loading tiles, while protected tiles use step 2 or step 8 as their lit fallback. Terrain, water, and canopies use destination-column bits from one 256-bit ownership mask per far tile, plus the eight neighboring masks for geometry crossing tile faces. Per-draw edge masks and paired ownership samples govern LOD skirts on resident finer-to-coarser boundaries. Any terrain-horizon patch intersecting an exact-owned column is excluded from occluders because fragment masking makes the patch incomplete. Complementary terrain and canopy reveal never changes source water ownership mid-transition. Playtests inspect these mechanisms for cracks, orphan skirts, duplicated ownership, water walls, cold-residency rings, disconnected islands, and topology pops.

Known performance gap: the current payload couples base terrain, water, and canopy construction. Measured cold canopy work ranges from 250 to 1,165 milliseconds per tile, so a canopy-heavy tile cannot publish an otherwise ready step-32 terrain and water parent until the complete payload finishes. Staged canopy attachment remains a follow-up. Exact halo closure is already explicit and is not part of this deferred work: missing lateral faces receive lit planned continuations or dark inward caps, and missing vertical faces receive bedrock caps until halo arrival invalidates the mesh. Do not report the two-second cold-horizon target as passing until the staged publication work and its reference-route measurement are complete.

Canopy reconstruction is likewise a far-build dependency. Review its anchor-cell fanout, priority comparisons, emitted impostor quads, and retained bytes at every tier. Bit 28 retains canopy classification, while per-column masks and their 3 by 3 tile neighborhood prevent far impostors from doubling exact trees, including when a crown crosses a tile face, without creating a second mesh or draw list.

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
ShorelineContour cache mutex                   leaf
MacroControl cache mutex                       leaf
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
- Lake-body depth interpolation retains dry and different-body weights as zero-depth contributors rather than renormalizing the wet subset. Competing authorities preserve distinct flat levels behind a bounded supported watershed, except where an outlet or channel corridor owned by either body must stay open.
- Incoming and outgoing channel halves interpolate monotonically between one deterministic junction level and immutable shared portal levels. Ordinary profiles cannot rise downstream; explicit falling water is limited to tagged analytical drops and outlet falls.
- A crater lake uses a warped absolute local profile, validates its complete rim in 96 directions with one block of freeboard, supports the full dry bank, and is rejected when a safe wet radius cannot fit. Ordinary routed lakes may retain named outlets.
- Each column plan adds compact 17 by 17 canonical water and lithology authority to its nine macro samples and 256 exact surface values. Water authority includes stable identity, level, depth, endorheic state, and ocean, river, lake, delta, waterfall, and supported-bank topology. Ambiguous lattice cells may make bounded exact hydrology or geology queries during cold plan construction; per-frame and per-cube hot paths may not reconstruct that authority.
- Moisture transport evaluates 17 points over 16 intervals of 256 blocks. Ocean, lake, and river recharge weights are 1.0, 0.65, and 0.18.
- Column plans retain nine full macro samples on a 3 by 3 lattice at eight-block spacing and a 256-column exact density surface grid. Construction performs 16 transient height-only perimeter queries for neighboring feature reach. Their cache holds at most 8,112 plans under a compile-time 128 MiB payload bound.
- Basin solutions use a separate byte-accounted 64 MiB single-flight LRU cache. Exact and far generators share one process-wide gate that permits at most two cold solution builds, while cache hits and same-key future waiters consume no additional permit.
- Shoreline contours use a separate byte-accounted 64 MiB single-flight cache keyed by stable water body and 256-block global page. Shared aprons make adjacent pages reconstruct the same boundary.
- Continuous geology, slope, climate, soil, and suitability fields use a 1,024-entry, 128 MiB single-flight cache of immutable 64 by 64-block tiles with eight-block controls and a one-control apron. Tensor cubic B-splines reconstruct C2 fields. A separate 1,024-entry, 8 MiB far-climate cache uses 256-block tiles with 128-block controls.
- Each generation worker retains at most 1,024 hotspot-cell and 2,048 volcanic-arc-cell primitive results in thread-local storage. Generator-instance tokens invalidate the primitive cache after explicit macro-cache clearing, while coordinate-pure reconstruction keeps eviction and worker count from changing output.
- Far terrain uses a separate 9,280-entry, 3 GiB CPU cache with active step-32 parent pinning.
- Ore, structure, tree, and flora candidate counts are fixed. Tree candidates read continuous biome suitability, climate, soil, slope, light, elevation, lithology, tectonic stress, hydrology, and ecotopes. Only mangrove and willow traits admit suitable shallow water, and their emitted roots or trunks reach the solid floor.

Cache payload limits do not include transient solver storage, hash-table overhead, shared ownership, or allocator fragmentation. Include those costs in the unified-memory measurement.

The basin solver contract requires fixed catchment extent, apron, input grid, numerical spacing, pass count, no more than two process-wide cold solution builds across exact and far generators, single-flight construction, byte-accounted LRU eviction, finite-field and outlet validation, an un-eroded base fallback with deterministic outlet metadata, and no wait under the world or basin-cache mutex. A cold caller may wait only on the dedicated construction permit or an existing shared future. Scalar and grid requests retain shared pointers to all candidate solutions until authority selection and shoreline reconstruction complete, so concurrent clear or eviction cannot invalidate a borrowed solution. A request keeps the cache generation observed before lookup or construction, preventing a clear that overlaps a cold build from making stale work eligible for future fast hits. One `BasinSolver` instance also represents one immutable callback context. Its coordinate-pure elevation, rainfall, and rock-resistance fields must not change during its lifetime because the cache key does not include callback identity. Do not describe its angular two-neighbor routing as a verbatim Tarboton triangular-facet implementation.

## 6. Queue, fluid, and population bounds

- Exact generation is nearest-first. Rebuilding the backlog reprioritizes work. Six workers plus one look-ahead slot cap submitted cube tasks at seven beneath the 64-job hard ceiling, while the remaining active-set demand stays in the separate prioritized backlog. Stale retained-set work is skipped, and completion processing requeues still-relevant work through current plan dependencies.
- Exact mesh candidates sort by three-dimensional distance. At most 64 items exist across queued, building, completed, and renderer-pending states, and duplicate cube results coalesce to the newest revision. Render-thread exact uploads stop after 64 meshes or 32 MiB in a frame, with two uploads and 4 MiB reserved for nearby edits.
- Far candidates sort front to back. The scheduler caps pending, completed, cache, and per-frame upload work as described above. Exact streaming reduces the refinement upload budget, while the eight far workers remain available under the four-parent and four-progressive reservation policy.
- Finished results, pending uploads, save queues, and GPU registries require eviction or backpressure. A producer cap alone does not bound an unconsumed completion vector. The save queue caps at 32,768 cubic positions, coalesces repeated snapshots for one queued position to its newest revision, and applies backpressure rather than accepting a 32,769th unique position.
- Runtime water is deduplicated by block position, delayed five ticks, and limited to 1,024 processed cells per tick. Pending updates and deferred frontiers each cap at 65,536. Long-frame recovery runs no more than eight catch-up ticks.
- Dropped items cap at 128 live entities with a swap-remove and an explicit oldest-grounded eviction at the cap. Physics and aging run only within the 96-block active radius; resting grounded items skip the collision sweep entirely, and the O(n^2) merge pass runs once per second. Pickup inflates the player AABB, adds absorbed stacks to the inventory, and compacts emptied entries. All lookups are non-loading, so a scattered item freezes at an unloaded boundary rather than force-generating or falling through the world. While a container screen is open the fixed loop drives at most eight catch-up furnace ticks so a stall never runs the sim.
- Boats cap at 40 live craft. Each tick is one buoyancy-or-gravity step, planar drag, a speed clamp, and a single per-axis terrain sweep; unmanned boats beyond the 128-block active radius freeze entirely while the ridden boat always steps. Boats carry no AI, aging, or despawn, and all block lookups are non-loading like the dropped-item path. A ridden boat replaces the player physics tick for that tick (the rider is slaved to the seat), so riding never doubles the movement cost.
- Canonical column authority covers oceans, rivers, lakes, deltas, waterfalls, and supported banks. Every standing generated wet voxel from the lowest one above solid support through its surface is an implicit source, including across cube boundaries. Those volumes allocate no explicit fluid state until runtime disturbance changes a cell.
- Ordinary generation and loading enqueue no fluid work. Only a gameplay edit or a matching previously activated frontier can introduce work.
- Fluids query loaded cubes only. Propagation must never call a force-loading world accessor.
- Deferred frontiers are indexed by unavailable destination cube. A fixed-tick resume budget considers only newly available index buckets and cannot scan all 65,536 frontiers once for every loaded cube.
- Stable source and flowing cells emit planar top geometry only. Vertical sides are exclusive to explicit falling cells, and far shorelines use body-aware contour-clipped triangles instead of rectangular water sheets or connections between incompatible authorities. Analytic fragment normals provide visual waves without displacing geometry.
- A lake outlet fall is a separate receiver-centered primitive with top, bottom, width, flow, and anchor data. Exact emission produces only its short falling footprint. The anchor's half-open far tile owns one five-quad prism. Neither representation raises the receiving body's standing water or runs generation-time fluid ticks.
- Far generated source water uses the same full-block surface plane as the exact implicit source voxel.
- Wild territories reevaluate once per second or after meaningful travel. AI and physics stop beyond 96 blocks, despawn begins beyond 112 blocks, and babies count toward the exact 64-animal limit.

Pending-update and deferred-frontier drop counts must appear in F3 diagnostics. Save and mesh coalescing counts must remain observable through their statistics interfaces and regression coverage. Silently exceeding a bound and growing anyway defeats the bound.

## 7. Main-thread and GPU workload

Terrain generation, cold macro construction, active-set reconstruction, exact unload scans, far tile construction, save compression, and ordinary exact mesh builds run off the render thread and fixed-tick thread. Player movement submits a latest-wins active-set request in bounded time. The utility-priority planner cancels superseded work before publication. Exact generation uses four latency-sensitive and two utility-priority workers, with at most seven cube tasks submitted for six workers and one look-ahead slot under the hard 64-job ceiling. Far construction uses four workers of each priority, and exact mesh workers use user-initiated priority. The far pool remains at eight workers during exact streaming, with four slots reserved for missing parents and four bounded urgent slots for connected refinements while base work exists. Exact-busy state limits refinement uploads rather than construction workers. Plan completions notify only after 128 results or backlog drain, and fixed-tick consumption has a four-tick cooldown. The main thread may upload finished geometry only within its count and byte budgets. Progressive selection uses retained vectors, fixed tier counters, and a flat grace-record buffer reserved for 4,096 coordinates during renderer reset rather than allocating a node per tile or per frame. The edit fast path may rebuild at most two already-meshed near-camera cubes synchronously; first-time streaming never uses it.

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
3. Can the 512-chunk visible setting accidentally expand exact generation, collision, entity, fluid, or save work?
4. Does exact unload retain two extra horizontal chunks and one extra vertical cube, and does far LOD preserve asymmetric hysteresis, or will ordinary movement churn a frontier or topology tier?
5. Does any main-thread path build a cold column plan, solve hydrology, generate a cube, construct a far tile, compress a save, or perform I/O?
6. Does any hot loop allocate, build a string, copy the loaded map, or materialize uniform storage?
7. Does every cache state its entry or byte cap, single-flight behavior, eviction policy, transient-memory caveat, and cache-eviction determinism test?
8. Does every solver state its domain, apron, resolution, iteration count, concurrency limit, validation, and fallback?
9. Does every queue state producer cap, consumer budget, deduplication or coalescing, overflow behavior, cancellation behavior, and diagnostic counter?
10. Can fluid work begin during generation or normal loading, cross an unloaded boundary without an indexed frontier, scan every frontier for one cube load, or force-load a cube?
11. Does a new lock fit the ordering table, and can its scope call a waiter, generator, allocator-heavy operation, or file API?
12. Do exact and far mesh residency, CPU caches, GPU arenas, transient solvers, and post targets fit the 64 GB unified-memory ceiling together?
13. Does every visible tile request a step-32 parent before refinement, does dispatch preserve four parent slots while capping urgent connected 16/8/4/2 work at four, does every far-owned exploration-band fragment require step 2, does every other far-owned exact-overlap fragment require step 8 or finer, does that protection include fully ready partial boundary tiles, do protected jobs bypass ordinary grace and transition admission, does the separate drawable frontier treat protected base-only tiles as missing, do 16 by 16 per-column masks control exact ownership with 3 by 3 neighboring masks for crossing geometry, do LOD skirts use displayed-neighbor and paired ownership samples, do partially masked patches stay out of occluders, and do ordinary atomic terrain swaps, two-phase canopy exchange, single-owner water, frustum culling, back-face culling, conservative horizon occlusion, greedy meshing, and mipmapped anisotropic texture sampling remain enabled and covered? Separately, does every missing exact halo emit its explicit lit, dark, or bedrock closure cap, and is synchronous canopy work still measured as a terrain and water parent publication delay?
14. Is any claim of HZB, geometry clipmaps, indirect command buffers, or GPU-driven visibility accurate to the implementation?
15. Does entity work stay inside the 96-block activation radius and exact 64-living cap?
16. Is new randomness coordinate-addressed or explicitly seed-derived, and do shuffled order and cache eviction preserve output?
17. Is a speed claim backed by a fixed-seed measurement with build, machine, p50, p95, residency, culling, memory, and queue-settle evidence?
18. Are stable exact and far water free of unsupported vertical walls or coarse joins between incompatible bodies, with full generated standing volumes represented by implicit sources, side geometry restricted to explicit falling columns, far shorelines body-aware and contour-clipped, the seed-42 X=-557, Z=379 source-volume regression passing, and the seed-764891 caldera enclosed by its validated irregular dry rim?
19. Was the canonical seed-764891 route measured at native resolution, 4x MSAA, and view distance 512 on the identified M4 Max, with its lowest sustained one-second rate at least 60 FPS and the 64 GB unified-memory ceiling checked?
20. Do unavailable in-range sections stay closed for collision and interaction, use lit planned silhouettes above ground and dark inward caps only underground, and block underground skylight until the vertical loaded path is continuous?
21. Does active-set rebuild expand each unique horizontal plan apron once, use indexed completion dependencies, coalesce rebuild notifications, and obtain visible saved sections through one bulk short-lock manifest read?
22. Do canonical 17 by 17 samples retain ocean, river, lake, delta, waterfall, and supported-bank authority, keep full-depth implicit sources supported, preserve distinct lake levels behind a competitive watershed except at owned outlet or channel corridors, and represent a valid abrupt drop as a narrow falling connection rather than a discarded body or raised receiver?
23. Does gameplay submit active-set work without executing it on the fixed tick, does the planner retain only the latest request and cancel stale epochs before publication, and do request, coalescing, cancellation, and build-time metrics remain observable?
24. At the same settled camera, does a playing-versus-paused comparison include fixed-tick CPU attribution and rule out an unexplained frame-rate multiplication caused by simulation or other main-thread work?
25. Do former 8-, 16-, 32-, 64-, 2,048-, and 8,192-block control lines stay within the documented derivative and orientation limits for hydrology, climate, material, lithology, and strata fields? If the preserved continuous-field check remains deferred, report its exact failures separately from exact-to-far render residency and do not mark continuity accepted.
