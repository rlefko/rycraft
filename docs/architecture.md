# Architecture

This document describes module ownership, concurrency, and the boundaries that keep rycraft's cubic world deterministic while rendering a long visible horizon.

## Subsystem map

```text
src/engine   Cocoa lifecycle, MTKView frame loop, fixed 20 Hz simulation,
             input, camera, game flow, diagnostics, and developer hooks
src/render   HDR Metal frame graph, exact cubic and far-terrain meshing,
             shadows, SSAO, volumetrics, water, post effects, UI, particles,
             textures, GPU arenas, and entity rendering
src/world    Cubic storage, macro generation, bounded basin solving, density,
             features, streaming, lighting, runtime fluids, and persistence
src/entity   Player physics, fauna physics and AI, habitat territories,
             spawning, and spatial queries
src/audio    Core Audio mixer and procedural sound effects
src/common   Math, counter randomness, non-generation seeded randomness,
             worker pool, and logging
```

The dependency direction is `engine -> render/world/entity/audio -> common`. Rendering reads immutable world snapshots and never owns or mutates terrain. Gameplay edits enter through `World`.

## World data layers

World state is separated by lifetime and scale:

```text
world seed
  -> coordinate-pure MacroGenerationSampler
      -> immutable 2,048-block BasinSolver solutions in a 64 MiB cache
      -> immutable ColumnPlan, cached per horizontal column
          -> independent 16 x 16 x 16 Chunk emission
              -> optional player edits and explicit FluidState
                  -> 18 x 18 x 18 MeshSnapshot
                      -> version-stamped exact GPU mesh

world seed
  -> coordinate-pure coarse geology, hydrology, and canopy queries
      -> immutable 256 x 256-block FarTerrainMesh
          -> exact near anchors or aggregate distant forest clusters
          -> bounded CPU cache and far GPU arena
```

`ColumnPos`, `ChunkPos`, and `BlockPos` are the only coordinate keys. X and Z are 64-bit; section and block Y are 32-bit. `ChunkPos` includes Y, so storage, pending work, renderer caches, lighting, and persistence cannot alias vertical sections.

`MacroGenerationSampler` contains no mutable simulation. It reconstructs geology, hydrology, climate, soil, and biome descriptors from a seed and coordinates. Its `BasinSolver` builds 16-block Priority-Flood, angular two-neighbor D-infinity-inspired routing, erosion, lake, channel, and delta rasters with a two-cell apron. Lake depth interpolation gives dry and different-body contributors a value of zero without renormalizing their weights. Retained lake members reconstruct their floors from one flat water level and positive depth, while a supported dry rim leaves named outlets and active channels open. A lake that drains into sufficiently lower receiving water owns a separate immutable `OutletFall` with top, bottom, width, flow, and receiver-anchor data. The receiver keeps its own standing `waterSurface`. Basin solutions are immutable and share construction through a byte-accounted 64 MiB single-flight cache. A process-wide permit gate shared by exact and far generators limits distinct cold construction to two simultaneous builds. One solver instance has one immutable callback context: elevation, rainfall, and rock-resistance callbacks must remain coordinate-pure and describe the same fields throughout that solver's lifetime because cache keys contain only seed-space catchment coordinates.

`ColumnPlan` retains nine full macro samples on a world-aligned 3 by 3 lattice at eight-block spacing and is immutable after construction. It also stores a 256-column exact density surface grid used for exposure and skylight plus a compact 17 by 17 canonical lake authority shared across positive column faces. Ambiguous macro interpolation is replaced by exact hydrology membership, water level, depth, and endorheic state. Exact density and cube emission consume the same authority, so every shore-water voxel occupies a supported wet column instead of a floating interpolation fringe. Construction makes 16 transient height-only perimeter queries for neighboring feature reach, but those samples are not retained. Its 8,112-entry single-flight cache has a compile-time 64 MiB payload bound. `Chunk` is the mutable unit. It starts as a uniform block value, materializes 4,096 values only when needed, and optionally stores 4,096 fluid bytes. Generation clips every global feature to the current cube and never writes to another loaded cube.

The renderer keeps exact, editable cubic terrain through radius 32 chunks. A separate immutable far-terrain branch fills the visible annulus through radius 256 chunks. Far tiles contain coarse surface and water geometry plus visual-only canopy impostors. The two- and four-block tiers reconstruct exact accepted tree anchors. The eight- and sixteen-block tiers query deterministic, globally anchored aggregate forest clusters, which preserve canopy mass without running exact local-priority competition over the full horizon. Far tiles never contain collision, entities, edits, runtime fluid state, caves, per-block flora, or saved ownership.

## Ownership and lifetimes

- `EngineState` owns the `World`, `SaveManager`, `Spawner`, audio, input, player state, and graphics settings.
- `World` holds a non-owning `SaveManager*`. The engine constructs the save manager first and destroys it after the world.
- `World` owns the cubic map, one active-set planner, four-worker generation pool, bounded generation backlog, column generator, block-light reconciliation queue, and fluid scheduler.
- `RenderPipeline` owns the exact two-worker `MeshScheduler`, four-worker far-terrain scheduler, exact and far mesh registries, exact and far mega-buffers, frame ring, block textures, shadow map, SSAO, volumetrics, volumetric clouds, water, bloom, post stack, entity renderer, and UI renderer.
- `Spawner` owns entity shared pointers and the spatial hash. Stable territory IDs identify wild fauna independently of visit order.
- `World::~World` marks shutdown, cancels and joins the active-set planner, moves generation futures out of their map, and waits without holding the pending-work mutex. Workers capture `this`, so teardown cannot finish while one is still inserting a cube.
- The engine shuts down exact and far mesh workers before destroying the world because exact jobs read snapshots from it and both schedulers own active threads.

## Threading model

| Context | Work |
|---|---|
| Main thread | Input, fixed ticks, player and fauna simulation, fluid scheduling, bounded GPU uploads, command encoding, UI |
| Active-set planner | Coalesce camera requests, select exact cubes and plan dependencies, publish retention, and unload stale cubes |
| Four generation workers | Load or generate cubic chunks and construct cold column plans and basin solutions |
| Two exact mesh workers | Copy a bounded snapshot, build greedy cubic geometry without the world lock, return versioned results |
| Four far-terrain workers | Construct immutable coarse surface tiles from coordinate-pure macro samples |
| Save thread | Serialize, LZ4-compress, and atomically replace edited cube files and manifests |
| Core Audio callback | Mix active voices under the audio voice mutex |

Gameplay submits active-set requests without rebuilding on the fixed-tick thread. One utility-priority planner retains only the latest camera request, checks its epoch between expensive phases, and refuses to publish superseded work. Column-plan completions notify after 128 results or backlog drain, and the fixed tick admits at most one notification every four ticks. Planner request, coalescing, cancellation, and build-time counters remain visible in performance logs. Generation and far-terrain workers also run at utility priority, while exact mesh workers use user-initiated priority for visible near-camera work.

Generation streams nearest first through at most two cold column-plan jobs and 64 in-flight cube jobs. Exact meshing admits at most 64 total items across queued, building, completed, and renderer-pending states. The far scheduler admits 64 pending jobs, retains at most 32 completed results, and discards stale-epoch output. A revision-cached loaded-world snapshot prevents a fresh chunk-map copy when the set has not changed.

The simulation ticks at 20 Hz. Runtime water uses the same fixed rate and a five-tick delay. Rendering uses the latest simulation state without an additional interpolation frame.

The render thread keeps three frames in flight behind a semaphore. Per-frame constants use three frame-ring slots. Exact and far mega-buffer ranges enter a deferred-free queue and are not reused until the GPU completes the frame that last referenced them.

## Lock discipline

The global order is:

```text
World::pendingMutex_ -> World::chunksMutex_
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

- Never generate, build a column plan or basin, load, compress, or perform file I/O while holding `chunksMutex_`.
- Synchronous `getChunk` checks under the mutex, releases it for load or generation, then uses `try_emplace`. Duplicate work is allowed; overwriting a loaded edit is not.
- `snapshotForMeshing` is the bounded exception. It copies an 18 by 18 by 18 block, fluid, and block-light halo plus a separate 18 by 18 array of per-column sky cutoffs under `chunksMutex_`, then releases the lock before meshing.
- Never wait on work that can take a mutex currently held by the waiting thread.
- Fluid queries use only loaded cubes. They do not call the loading `getBlock` or `getChunk` path.
- Save serialization and compression run on the save thread. The queue is capped at 32,768 positions, and the pending-save map coalesces repeated snapshots for one position while keeping the latest queued cube visible to loads. Manifest disk writes serialize under `manifestWriteMutex_` after copying or updating state under the short-lived lookup mutex.
- Single-flight caches publish shared work under their own mutex and perform expensive construction after releasing it.
- The basin solver releases its cache mutex before waiting on either an existing future or the process-wide cold-build permit. The permit mutex is never nested with the cache mutex.

See [performance-conventions.md](performance-conventions.md) for caps and review questions.

## Exact cubic streaming and mesh consistency

The exact active set never follows the 256-chunk visible horizon. Gameplay rebuilds it on the dedicated planner, not during a fixed tick or render pass. Camera movement replaces any queued request, and an in-progress request checks the latest epoch before every major phase and immediately before publication. Each rebuild gathers one unique set of horizontal columns, expands the fixed plan apron once, and registers pending-plan dependents in an index. A plan completion addresses only that dependency bucket, and completion notifications coalesce before the next rebuild instead of scanning the retained cube map once per result. It is bounded by `min(viewDistance, 32)` and combines:

- exposed column-plan sections in the horizontal disk;
- a camera exploration band with radius six chunks and four cubes above and below the camera;
- saved edited sections copied for the unique visible-column set through one bulk, short-lock manifest query;
- a complete one-cube halo for collision, light, and meshing.

The mesh-candidate set is capped at 16,384 cubes. The retained set, including halo cubes, is capped at 32,768. The camera exploration and collision band is the highest-priority class, followed by exposed surface, saved edits, and then full three-dimensional distance. This keeps every cube within six horizontal chunks and four vertical sections of an underground player ahead of global cap pressure. Existing cubes remain retained through two extra horizontal chunks and one extra vertical cube, giving unload a concrete hysteresis boundary outside the current target.

Every exact mesh is identified by `ChunkPos` and a cube revision. An edit increments the owning cube and every face, edge, or corner neighbor whose one-block halo intersects the changed block. A stale worker result may be uploaded to avoid a hole, but its revision mismatch immediately schedules a newer mesh.

`MeshSnapshot` carries one block of padding on all axes. Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real faces, edges, and corners. If an in-range neighbor is still unavailable, its halo follows the immutable generated surface cutoff. Cells above that terrain silhouette remain air, cells below it remain conservatively opaque, and a missing cardinal face seals only the unresolved cells. An exposed uphill continuation uses the arriving column's normally lit surface material, while a cave opening below the local surface uses an unlit inward bedrock cap. This prevents full 16 by 16 black panels above ground without opening an underground void. Loading or unloading any halo cube invalidates every neighboring mesh that sampled it. World-floor and world-ceiling cells use the documented bedrock and air boundary values.

Skylight uses the exact surface cutoff only after every vertical section from the meshed cube through that cutoff is loaded. The world maintains a compact loaded-section mask per horizontal column. An incomplete path forces a fully occluded cutoff and dirties lower meshes when vertical continuity changes, preventing sunlight from crossing unloaded space above an underground camera.

Collision also treats missing cubes as closed. DDA targeting stops at the first unavailable cube, and breaking or placement revalidates loaded ownership before mutation. Interaction therefore cannot force generation or edit through a temporary mesh cap.

Meshes keep vertices cube-local. `ChunkOrigin` restores X, Y, and Z world position in the vertex shader. AABBs, frustum tests, candidate distance, water sorting, and buffer ownership are all three-dimensional.

## Far-terrain visibility architecture

The far renderer uses immutable 256 by 256-block tiles in the half-open annulus `[32, 256)` chunks. A narrow two-block sampling tier immediately outside radius 32 is the topology bridge: it samples exact emitted density heights rather than the coarser macro surface, so the first far vertices agree with exact terrain. Whole far tiles overlap the exact disk. Exact opaque terrain draws first, and a small positive depth bias keeps overlapping far tops behind resident exact surfaces while retaining a lit fallback for cold exact meshes. Water and canopy summaries keep exact ownership through radius 32, then use one stable world-space dither across the following 16 blocks. Farther out, distance and immutable tile complexity derived from maximum sampled slope and hydrology select among four-, eight-, and sixteen-block tiers. The distance thresholds are implementation parameters rather than fixed rings. The previously selected tier applies asymmetric refine and coarsen thresholds, so ordinary camera movement does not make a coordinate chatter between topologies.

Globally aligned tile borders and transition skirts hide level boundaries. Bit 29 marks every skirt vertex. Per-draw metadata enables an edge only when its displayed neighbor is resident at a coarser step, so absent and same-LOD neighbors cannot expose full-height panels. The fragment shader suppresses every skirt throughout the exact-to-far handoff. Direct exact-to-far tests compare the two-block topology tier with exact emitted surface heights at shared samples, while captures validate terrain and shoreline ownership across the handoff. A resident topology remains visible until its replacement uploads. The renderer then performs a 0.4-second fog-hidden transition, fading the old tile into fog, swapping at the obscured midpoint, and fading the new tile back out. At most 64 topology transitions are active at once, and only one tier for a coordinate is drawn at any instant. The mesher greedily combines equal flat terrain cells. Far standing water uses contour-clipped shoreline triangles and top geometry only, so a partially wet coarse cell cannot become a rectangular water ledge. A separate `OutletFall` is one narrow receiver-centered prism with four side quads and one top quad. Its anchor's half-open tile owns all five quads, even across a tile face, and the prism overlaps the lower body's top source voxel before reaching the upper lip without raising the receiving water. Tile coordinates and sampling remain 64-bit on the CPU, while a per-draw origin restores world space from tile-local half-precision vertices.

The far mesher queries the same accepted tree anchors as exact cube generation at two- and four-block steps, rendering each as one grounded trunk-and-crown box impostor. At eight- and sixteen-block steps it samples deterministic 32- or 64-block forest cells. Climate suitability, substrate, slope, water, and a cell-addressed priority decide one aggregate trunk-and-crown cluster per accepted cell. Half-open cell and tile ownership prevents duplicates across boundaries. Bit 28 of `faceAttr` marks both forms for the shared vertex contract and diagnostics. The exact-to-far predicate clips canopy fragments inside the exact radius and dithers them through the 16-block handoff, so an exact tree and its far summary never overlap. Opaque far tops remain independently available as depth-tested fallback.

The design is an adaptive tiled terrain LOD informed by geometry clipmaps and CDLOD. It is not a literal geometry clipmap. Visibility first uses a conservative tile AABB frustum test, then processes surviving tiles front to back through a conservative 256-bin terrain-horizon culler. Sixteen 64 by 64-block patches per visible tile contribute lower-bound horizons without per-frame heap allocation. The horizon test rejects a tile only when fully covered angular bins establish a higher lower-bound horizon. This is not a hierarchical Z buffer and does not use a depth pyramid. Exact opaque and far terrain use counterclockwise front faces with back-face culling. Cross and flat flora emit both windings. Shadow casters and water remain cull-none for their separate correctness requirements.

Far terrain is encoded through bounded direct indexed draws. The implementation does not build Metal indirect command buffers. Exact opaque geometry draws first and shares depth with the far annulus, making far tops a depth-backed cold-residency fallback. Water samples resolved depth without a depth attachment, so water and canopies retain the shared 16-block fragment handoff. Far water joins the same three-dimensional back-to-front water list as exact water.

CPU far-tile retention is capped at 1,024 entries and 512 MiB. The far GPU arena reserves 256 MiB for vertices and 128 MiB for indices. These are independent of exact mesh residency. All renderer, world, transient, and Metal allocations together must remain below the 64 GB unified-memory acceptance ceiling.

## Frame graph and graphics settings

The renderer preserves a linear HDR frame graph:

1. Three shadow cascades and water-shadow slices
2. One 4x MSAA `RGBA16Float` scene pass for sky, exact and far terrain, entities, highlight, particles, and flat clouds
3. SSAO and depth-aware bilateral reconstruction
4. Scene application and opaque-color copy
5. Post-resolve water with screen-space reflection and manual resolved-depth occlusion
6. Volumetric light and volumetric clouds
7. GPU-resident exposure and flare probes
8. Bloom
9. One final tonemap, vibrance, sharpening, flare, and dithering composite
10. UI

The single final composite is the only linear-HDR to display conversion. Toggled-off effects skip work or bind fixed fallback textures without changing the frame graph's resource contracts.

The procedural block-texture array contains all five mip levels from 16 by 16 through 1 by 1. Deterministic alpha-aware downsampling preserves representable cutout coverage. The terrain sampler uses nearest magnification to retain the block aesthetic and linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy to suppress distant aliasing.

`GraphicsSettings` serializes effect toggles, quality values, view distance, and input bindings into `~/Library/Preferences/rycraft/settings.json`. Settings load before `RenderPipeline` construction because they size targets and arenas. Environment overrides apply after JSON load and are never saved, so a playtest cannot overwrite preferences.

## Derived block light

Block light is derived state, like a mesh. `LightEngine::computeSelfLight` runs before a generated or loaded cube is published. The tick-thread reconciliation queue pulls light across all six cube faces under a bounded cube budget. Edits run removal and addition propagation against loaded cubes.

The propagation is monotone over fixed blocks and converges to one fixed point independent of load order. Light is recomputed, never serialized. A changed light field increments the cube revision and dirties its mesh.

## Runtime water boundary

World generation writes complete ocean, river, lake, aquifer, delta, crater-lake, channel-waterfall, and outlet-fall blocks directly. Every standing water voxel from its supported floor through its top voxel is an implicit source, including across cube boundaries, so generated standing volumes need no explicit fluid array. Generation never asks the fluid scheduler to settle terrain, and loading an ordinary generated cube also schedules no water. Stable generated and runtime water emits top geometry only. Vertical water sides are reserved for cells explicitly marked falling, including the short receiver-centered outlet overlay, so lake, river, and ocean boundaries cannot render as unsupported walls. The far sampled representation places generated source water on the same 0.875-block plane as the exact implicit source voxel.

A gameplay edit activates one cell and its six neighbors. The scheduler applies downward-first source and level rules to loaded cells only. If active flow reaches a missing cube, it persists a frontier indexed by that destination cube. A later load makes only the matching index bucket eligible, and the fixed tick resumes a bounded number rather than scanning all 65,536 possible frontiers. Runtime fluid writes bypass the player activation entry point to avoid recursive scheduling. Pending-update and frontier-overflow counters remain visible in diagnostics.

This division keeps generation order-independent while localized player disturbance can evolve over time.

## Persistence boundary

RYCH v4 stores one edited cubic section per LZ4 file using 64-bit X and Z plus section Y. Its packed 44-byte header includes an IEEE CRC-32 of the uncompressed block and optional fluid payload. A per-column manifest records edited section Y values and deferred fluid frontiers. Bulk visible-column reads copy only the required manifest data under one short lock. Manifest disk I/O uses a separate serialized write lock, and the bounded save queue coalesces repeated snapshots for a cubic position to the newest revision. Metadata remains separate and preserves the seed, player position and orientation, health, selected hotbar slot, nine-slot hotbar inventory, and world time.

v3 chunk payloads are not converted. The loader rejects their version and regenerates v4 terrain while leaving old files untouched. Compatible metadata still loads. Detailed layout and paths are in [world-generation.md](world-generation.md).

`GraphicsSettings` are preferences, not world metadata, and follow the separate settings path described above.

## Error policy

1. Metal device, queue, and pipeline failures are fatal because the game cannot render without them.
2. Cube or far-tile generation failures log and omit that result so streaming can continue.
3. Missing files return `std::nullopt`. Corrupt or incompatible cube data reports that cube's failure once, returns no cube, and regenerates deterministically.
4. Audio initialization may fail without preventing play.

There is no project-wide `Result<T, E>` wrapper. These boundaries use the smallest established mechanism for their failure mode.

## GPU boundary

Every structure shared by C++ and Metal lives in `include/render/shader_types.hpp` with size and offset assertions. Exact and far terrain preserve the 16-byte vertex format. Fluid direction occupies bits 24 through 26, falling state occupies bit 27, far-canopy impostors use bit 28, far-terrain boundary skirts use bit 29, and bits 30 through 31 remain reserved.

See [rendering-conventions.md](rendering-conventions.md) for coordinate, pass, halo, culling, and water rules.

## Tests and diagnostics

One hermetic Catch2 executable is built from eleven source modules covering common, world, advanced world generation, geology and ecology, render, entity, fauna habitats, fluid persistence, render concurrency, engine, and audio code. Cubic tests include negative floor conversion, vertical limits, uniform storage, full-width counter addressing, generation order, the complete 26-neighbor mesh halo, block-light convergence, indexed and budgeted fluid frontiers, bounded coalesced v4 persistence, stable territory IDs, runtime approach-triggered fleeing, and fauna movement. Basin tests pin canonical lake support, the seed-42 supported lake lip at X=-8235, Z=2976, the incised river across X=-12288 at Z=2653 and Z=2654, the canyon ecotope at X=-23904, Z=0, receiver-centered outlet falls into lower standing water, distributary deltas, bounded single-flight construction, the process-wide two-build cap across independent solver instances, reverse-order reconstruction, and cache-eviction determinism. The seed-764891 caldera sample verifies a complete irregular enclosing rim, one-block freeboard, supported dry banks, full-depth implicit source water across a cube face, and deterministic reverse-order reconstruction. A separate aquifer fixture verifies a sealed pocket.

Far-terrain tests compare the two-block topology tier with exact emitted density heights at the radius-32 seam, pin the shared 16-block water and canopy coverage, verify skirt marker and displayed-neighbor masks, preserve the exact 0.875 source-water plane, preserve exact canopy ownership at steps two and four, and verify deterministic aggregate forest ownership at steps eight and sixteen. They also cover adaptive tier selection, hysteresis, transition phases, border agreement, contour-clipped shorelines, outward winding, deterministic tile hashes, cache and queue bounds, epoch cancellation, and conservative horizon behavior. Streaming tests distinguish lit generated surface silhouettes from dark underground closures, then pin hard exploration-band priority, closed missing collision, aborted ray traversal, and skylight occlusion across unloaded vertical gaps. Headless tests do not create a Metal device, so actual culling, frame rate, memory, and image quality require the playtest workflow.

Developer runs can fix the seed and spawn with `RYCRAFT_WORLD_SEED` and `RYCRAFT_SPAWN`. `RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, or `biome`. The inspector executable provides repeatable feature locations, sample data, timing, hashes, and separate column-plan and basin-cache metrics. F3 displays cubic, far-tile, macro-cache, fluid, queue, coalescing, and dropped-work diagnostics. Visual changes must still run with Metal validation and be inspected in captured frames.
