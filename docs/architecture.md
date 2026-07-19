# Architecture

This document describes module ownership, concurrency, and the boundaries that keep rycraft's cubic world deterministic while rendering a long visible horizon.

## Subsystem map

```text
src/engine   Cocoa lifecycle, MTKView frame loop, fixed 20 Hz simulation,
             input, camera, game flow, diagnostics, and developer hooks
src/render   HDR Metal frame graph, exact cubic and far-terrain meshing,
             shadows, screen-space lighting, atmosphere, clouds, froxels,
             water, lightning, post effects, UI, particles, textures, GPU
             arenas, and entity rendering
src/world    Cubic storage, macro generation, bounded basin solving, density,
             features, streaming, derived lighting, regional weather, runtime
             fluids, and persistence
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
      -> immutable water-body shoreline pages in a 64 MiB cache
      -> filtered SurfaceFootprint samples at 1, 2, 4, 8, 16, or 32 blocks
      -> immutable ColumnPlan, cached per horizontal column in 128 MiB
          -> independent 16 x 16 x 16 Chunk emission
              -> optional player edits and explicit FluidState
                  -> 18 x 18 x 18 MeshSnapshot
                      -> version-stamped exact GPU mesh

world seed
  -> coordinate-pure FarSurfaceSample and canopy queries
      -> immutable step-32 parent for every visible 256 x 256-block tile
          -> connected step-16, step-8, step-4, and step-2 voxel refinement
      -> immutable 256 x 256-block FarTerrainMesh
          -> exact near anchors or aggregate distant forest clusters
          -> bounded CPU cache and far GPU arena

world seed + saved world time + static climate
  -> latest-wins WeatherSystem utility worker
      -> immutable 81 x 81 regional snapshots at 256-block spacing
          -> two deterministic slices 200 ticks apart
              -> weather, clouds, fog, precipitation, lightning, and thunder
```

`ColumnPos`, `ChunkPos`, and `BlockPos` are the only coordinate keys. X and Z are 64-bit; section and block Y are 32-bit. `ChunkPos` includes Y, so storage, pending work, renderer caches, lighting, and persistence cannot alias vertical sections.

`MacroGenerationSampler` contains no mutable simulation. It reconstructs geology, hydrology, climate, soil, and biome descriptors from a seed and coordinates. Immutable 64 by 64-block macro-control tiles retain eight-block controls plus a one-control apron. Tensor cubic B-splines reconstruct the continuous C2 fields, while rotated filtered Simplex bands restore footprint-appropriate detail without exposing a tile edge. The main macro-control cache holds at most 1,024 entries and 128 MiB. A separate 1,024-entry, 8 MiB far-climate cache uses 256-block tiles and 128-block controls to avoid repeating the bounded moisture integration at every coarse vertex. Categorical plate, water-body, and feature ownership remains a direct coordinate query rather than an interpolated category.

Its `BasinSolver` builds 16-block Priority-Flood, angular two-neighbor D-infinity-inspired routing, erosion, lake, channel, and delta rasters with a two-cell apron. Lake depth interpolation gives dry and different-body contributors a value of zero without renormalizing their weights. Each stable water body reconstructs a signed shoreline from immutable 256 by 256-block pages. Four-block controls, shared aprons, and two-block refinement in the narrow contour band make adjacent pages agree without exposing the routing raster. These pages share a separate byte-accounted 64 MiB single-flight cache. When two lake authorities overlap, a competitive signed-distance watershed preserves both flat levels behind a supported divider. An owned outlet or active channel corridor exempts the connector from that divider. Curved channel halves interpolate monotonically from one deterministic junction level to their shared portal levels; ordinary profiles never rise downstream, while tagged falls own explicit falling water. A lake that drains into sufficiently lower receiving water owns a separate immutable `OutletFall` with top, bottom, width, flow, and receiver-anchor data. The receiver keeps its own standing `waterSurface`. Basin solutions are immutable and share construction through another byte-accounted 64 MiB single-flight cache. Scalar and grid sampling retain shared ownership of every candidate solution for the complete authority and contour query, so concurrent clear or LRU eviction cannot invalidate a referenced neighbor. A cache generation stamp prevents work started before a clear from becoming a current fast hit afterward. A process-wide permit gate shared by exact and far generators limits distinct cold construction to two simultaneous builds. One solver instance has one immutable callback context: elevation, rainfall, and rock-resistance callbacks must remain coordinate-pure and describe the same fields throughout that solver's lifetime because cache keys contain only seed-space catchment coordinates.

`ColumnPlan` retains nine full macro samples on a world-aligned 3 by 3 lattice at eight-block spacing and is immutable after construction. It also stores a 256-column exact density surface grid plus compact 17 by 17 water and lithology authority shared across positive column faces. Water entries carry stable `WaterBodyId`, level, depth, endorheic state, and canonical ocean, river, lake, delta, waterfall, and supported-bank topology. The plan also retains the maximum waterfall reach needed for exposed-section discovery. Lithology entries carry the two facies, transition weight, and contact distance needed to avoid nearest-corner geological walls. Exact density and cube emission consume these canonical arrays, so every generated standing-water column has a solid floor and implicit source blocks at every voxel through its top, while rock contacts do not reset at a chunk edge. Construction makes 16 transient height-only perimeter queries for neighboring feature reach, but those samples are not retained. Its 8,112-entry single-flight cache has a compile-time 128 MiB payload bound. `Chunk` is the mutable unit. It starts as a uniform block value, materializes 4,096 values only when needed, and optionally stores 4,096 fluid bytes. Generation clips every global feature to the current cube and never writes to another loaded cube.

The renderer targets exact, editable cubic terrain through a nominal radius of 32 chunks. `ExactSurfaceCoverageSnapshot` publishes the planned surface and boundary requirements before caps, unresolved columns, and the active-set epoch. All requirements currently published for one 16 by 16-block column must be owned by exact GPU meshes before that column's bit masks the far overlap; an empty completed mesh is ready. A previously published exact mesh can retain ownership while its replacement is pending. Each 256 by 256-block far tile receives a 256-bit column mask, and a 3 by 3 tile neighborhood accompanies every draw so a canopy or waterfall crossing a tile face queries the destination column. Every fragment that remains far-owned inside the exact overlap retains fine-fallback protection. This includes fragments in a fully ready boundary tile whose published exact requirements cover only part of that tile. The global nearest-gap distance remains a conservative coverage fallback and diagnostic, not the primary visible ownership rule.

A separate immutable far-terrain branch selects every tile intersecting the visible radius-512 disk, including tiles wholly inside the exact disk. Each selected coordinate requests a step-32 parent before optional refinement, and a resident active parent is pinned. Four of the eight far workers remain reserved for missing parents while four bounded urgent slots construct distance-selected step-16, step-8, step-4, and step-2 targets in the connected visible prefix. Those targets do not wait for the complete parent disk. Every far-owned fragment in the camera exploration band requires step 2 before it can display far geometry. Every other far-owned fragment in the exact overlap requires step 8 or finer, including fragments of fully ready partial boundary tiles. Protected fallback jobs bypass ordinary grace and topology-transition admission, while their step-32 parents remain resident but hidden. A refinement still requires its own parent to be resident. Far tiles contain coarse voxel surface and water geometry plus visual-only canopy impostors. Step 2 reuses accepted exact tree anchors, species, and dimensions. Steps 4 through 32 query deterministic, globally anchored 64-block aggregate forest cells with six fixed candidates and block-8 habitat and ground authority, and those aggregate tiers form strict stable subsets. Far tiles never contain collision, entities, edits, runtime fluid state, caves, per-block flora, or saved ownership.

## Ownership and lifetimes

- `EngineState` owns the `World`, `WeatherSystem`, `SaveManager`, `Spawner`, audio, input, player state, wetness, and graphics settings. It retains one immutable weather snapshot for each frame and fixed tick.
- `World` holds a non-owning `SaveManager*`. The engine constructs the save manager first and destroys it after the world.
- `World` owns the cubic map, one active-set planner, six-worker generation pool, bounded generation backlog, column generator, packed voxel-light reconciliation queue, and fluid scheduler.
- `WeatherSystem` owns one joinable utility worker and publishes immutable camera-centered snapshots with latest-wins admission. Gameplay retains the prior valid snapshot while a replacement is built.
- `RenderPipeline` owns the exact four-worker `MeshScheduler`, eight-worker far-terrain scheduler, exact and far mesh registries, exact and far mega-buffers, frame ring, block textures, five-cascade shadow map, `ScreenSpaceLighting`, `AtmosphereRenderer`, `CloudRenderer`, froxel `Volumetrics`, lightning, water, bloom, post stack, entity renderer, and UI renderer.
- `Spawner` owns entity shared pointers and the spatial hash. Stable territory IDs identify wild fauna independently of visit order.
- `World::~World` marks shutdown, cancels and joins the active-set planner, moves generation futures out of their map, and waits without holding the pending-work mutex. Workers capture `this`, so teardown cannot finish while one is still inserting a cube.
- The engine shuts down exact and far mesh workers before destroying the world because exact jobs read snapshots from it and both schedulers own active threads.

## Threading model

| Context | Work |
|---|---|
| Main thread | Input, fixed ticks, player and fauna simulation, fluid scheduling, bounded GPU uploads, command encoding, UI |
| Active-set planner | Coalesce camera requests, select exact cubes and plan dependencies, publish retention, and unload stale cubes |
| Six generation workers | Load or generate cubic chunks and construct cold column plans and basin solutions |
| Four exact mesh workers | Copy a bounded snapshot, build greedy cubic geometry without the world lock, return versioned results |
| Eight far-terrain workers | Construct immutable coarse surface tiles from coordinate-pure macro samples |
| Weather utility worker | Batch static climate samples and publish the latest regional weather request |
| Save thread | Serialize, LZ4-compress, and atomically replace edited cube files and manifests |
| Core Audio callback | Mix active voices under the audio voice mutex |

Gameplay submits active-set requests without rebuilding on the fixed-tick thread. One utility-priority planner retains only the latest camera request, checks its epoch between expensive phases, and refuses to publish superseded work. Column-plan completions notify after 128 results or backlog drain, and the fixed tick admits at most one notification every four ticks. Planner request, coalescing, cancellation, and build-time counters remain visible in performance logs. Exact generation splits four latency-sensitive workers from two utility workers, far construction splits four from four, and exact mesh workers use user-initiated priority for visible near-camera work.

Generation streams through four latency-sensitive and two utility workers with at most two cold column-plan jobs. The pump submits at most seven cube tasks at once, six running tasks plus one look-ahead task, beneath the hard 64-job ceiling. Remaining active-set demand stays in the separate prioritized backlog. Each queued plan and cube carries its active-set epoch, gameplay lane, and distance. A worker skips a task that is stale for the current retention set, and completion processing requeues a still-required cube through its current plan dependencies. New camera work therefore passes stale queued work, the camera column and six-chunk exploration band pass the broad exact surface disk, and the next required plan cannot remain behind a full FIFO burst of cubes. Exact meshing admits at most 64 total items across queued, building, completed, and renderer-pending states. Thirty-two of those slots remain available to camera-band work, and four user-initiated mesh workers order that lane before broad surfaces. The eight-worker far scheduler splits four latency-sensitive workers from four utility workers, admits 64 pending jobs, retains at most 32 completed results, and discards stale-epoch output. While parents are missing, it reserves four worker slots for the base lane and four for connected progressive refinement. Exact work retains its dedicated pools and lane priority, and exact-busy state limits refinement uploads to four per frame without reducing the far construction pool. The steady-state render loop reuses reserved far candidate, request, key, result, and flat grace-record buffers plus fixed tier counters. A revision-cached loaded-world snapshot prevents a fresh chunk-map copy when the set has not changed.

The simulation ticks at 20 Hz. Runtime water uses the same fixed rate and a five-tick delay. Weather snapshots contain two slices ten real seconds apart and interpolate by saved world time. Rendering uses the latest simulation and immutable weather state without an additional gameplay interpolation frame.

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
- Synchronous `getChunk` checks both presence and loaded capacity under the mutex, releases it for load or generation, then rechecks both before `try_emplace`. Duplicate work is allowed; overwriting a loaded edit or exceeding the loaded cap is not. A capacity rejection returns no cube, and a loading block query treats that unavailable cube as closed.
- `snapshotForMeshing` is the bounded exception. It copies an 18 by 18 by 18 block, fluid, and packed voxel-light halo plus separate 18 by 18 generated-surface and sky-cutoff authority under `chunksMutex_`, then releases the lock before meshing.
- Never wait on work that can take a mutex currently held by the waiting thread.
- Fluid queries use only loaded cubes. They do not call the loading `getBlock` or `getChunk` path.
- Save serialization and compression run on the save thread. The queue is capped at 32,768 positions, and the pending-save map coalesces repeated snapshots for one position while keeping the latest queued cube visible to loads. Manifest disk writes serialize under `manifestWriteMutex_` after copying or updating state under the short-lived lookup mutex.
- Single-flight caches publish shared work under their own mutex and perform expensive construction after releasing it.
- The basin solver releases its cache mutex before waiting on either an existing future or the process-wide cold-build permit. The permit mutex is never nested with the cache mutex. A sample retains shared pointers to all authority candidates until selection and shoreline reconstruction finish, even if another thread clears or evicts the cache.

See [performance-conventions.md](performance-conventions.md) for caps and review questions.

## Exact cubic streaming and mesh consistency

The exact active set never follows the 512-chunk visible horizon. Gameplay rebuilds it on the dedicated planner, not during a fixed tick or render pass. Camera movement replaces any queued request, and an in-progress request checks the latest epoch before every major phase and immediately before publication. Each rebuild gathers one unique set of horizontal columns, expands the fixed plan apron once, and registers pending-plan dependents in an index. A plan completion addresses only that dependency bucket, and completion notifications coalesce before the next rebuild instead of scanning the retained cube map once per result. It is bounded by `min(viewDistance, 32)` and combines:

- a camera exploration band with radius six chunks and four cubes above and below the camera;
- saved edited sections copied for the unique visible-column set through one bulk, short-lock manifest query;
- one primary surface section for each resolved or unresolved visible column;
- additional exposed and cliff-wall sections from the resolved column plans;
- a complete one-cube halo for collision, light, and meshing.

The mesh-candidate set is capped at 16,384 cubes. The retained set, including halo cubes, is capped at 32,768. Capacity is reserved for the exploration and collision band first, then edited sections, one primary surface section per visible column, additional exposed or cliff sections, and halos. Distance resolves ties inside each class. An omitted surface fragment remains represented by its far parent dependency and by step-2 or step-8 display geometry according to its protected overlap class. Existing cubes remain retained through two extra horizontal chunks and one extra vertical cube, giving unload a concrete hysteresis boundary outside the current target. When retention changes, obsolete cubes unload before replacement jobs are submitted. Asynchronous and synchronous publication both enforce the same cap under `chunksMutex_`.

Every exact mesh is identified by `ChunkPos` and a cube revision. An edit increments the owning cube and every face, edge, or corner neighbor whose one-block halo intersects the changed block. Worker jobs retain both the renderer request revision and the revision captured by their snapshot. Coalescing preserves the completion for the newest request, including a failed snapshot, but the render thread publishes only a result that matches the live cube revision and is newer than any resident mesh. A rejected or failed completion clears only its matching request, so it cannot replace newer geometry or cancel a newer request. Any previously published exact mesh remains available while the current revision is rebuilt.

`MeshSnapshot` carries one block of padding on all axes. Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real faces, edges, corners, and packed voxel light. If an in-range neighbor is still unavailable, its halo follows the immutable generated surface cutoff. Cells above that terrain silhouette remain air, cells below it remain conservatively opaque, and a missing cardinal face seals only unresolved boundary cells. For a lateral face with a possible cap, the mesher floods transparent cells through the bounded 18 by 18 by 18 snapshot from sky-exposed seeds. Complete column authority is authoritative for both raised and lowered roofs. The distinct incomplete-path marker instead uses the generated cutoff only to keep an aboveground provisional boundary classified as sky-connected while propagated skylight remains dark. Sky-connected continuations receive a lit provisional face using one representative arriving-surface material per missing face. Enclosed lateral openings receive dark stone, while missing vertical openings receive bedrock. This prevents full 16 by 16 black panels above ground without opening an underground void. The flood runs only after a cheap boundary-candidate scan. Loading or unloading any halo cube reconciles the affected light faces and invalidates neighboring meshes that sampled changed halo data. World-floor and world-ceiling cells use the documented bedrock and air boundary values.

The world maintains a compact loaded-section mask per horizontal column. Only a continuous loaded path to the open sky may seed level-15 skylight. An incomplete path remains dark and dirties lower light fields when vertical continuity changes, preventing sunlight from crossing unloaded space above an underground camera.

Collision also treats missing cubes as closed. DDA targeting stops at the first unavailable cube, and breaking or placement revalidates loaded ownership before mutation. Interaction therefore cannot force generation or edit through a temporary mesh cap.

Meshes keep vertices cube-local. `ChunkOrigin` restores X, Y, and Z world position in the vertex shader. AABBs, frustum tests, candidate distance, water sorting, and buffer ownership are all three-dimensional.

## Far-terrain visibility architecture

The far renderer selects every immutable 256 by 256-block tile intersecting the radius-512 visible disk, not just a configured annulus. Every selected coordinate requests a step-32 parent in nearest-first order. A broad parent lane advances the connected horizon. The next bounded lane requests each nearby coordinate's distance-selected target plus every protected exact-loading fallback. Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks missing step-32 dependencies. The drawable frontier also treats a protected base-only tile as missing, suppresses geometry at or beyond the nearest such gap, and fades the preceding 256 blocks. Distance and immutable tile complexity derived from maximum sampled slope and hydrology select the final two-, four-, eight-, or sixteen-block tier. The previously selected tier applies asymmetric refine and coarsen thresholds, so ordinary camera movement does not make a coordinate chatter between topologies.

Exact opaque terrain draws first, and a small positive depth bias keeps overlapping far tops behind resident exact surfaces while retaining a lit fallback for cold exact meshes. The exact coverage snapshot becomes one 16 by 16 per-column ownership mask for every overlapping far tile. A column acquires its bit only when every required exact section has a current mesh, then retains that bit through ordinary stale revisions while the previous exact mesh stays published. Each draw carries the center mask and all eight neighboring tile masks, which extends the same ownership test across tile faces for canopies and waterfalls. Any fragment that remains far-owned in the exact overlap is protected, even when its tile is fully ready for a partial set of boundary requirements. Step 32 is not an acceptable visible fallback there. The camera exploration band must have step 2, and the rest of the exact overlap must have step 8 or finer. Any horizon patch touching an exact-owned column is excluded from the occluder set because fragment masking makes it nonconservative. Coarse parent geometry uses conservative footprint bounds beneath exact surfaces. The nearest unready distance remains available for conservative parent selection, fog behavior, and diagnostics, but it does not clip a complete radial ring.

Globally aligned tile borders and transition skirts join resident far LOD tiers. Bit 29 marks every skirt vertex. Per-draw metadata enables an edge only when its displayed neighbor is resident at a coarser step, and the shader evaluates the exact-ownership masks on both sides of that join. These skirts address finer-to-coarser far-tile transitions. Missing exact halos use separate explicit closure geometry: lit planned surface continuations aboveground, dark inward caps underground, and bedrock caps vertically. Direct exact-to-far tests compare the two-block topology tier with exact emitted surface heights at shared samples, while captures validate terrain and shoreline ownership across the overlap. Production lake and caldera fixtures also prove that independently filtered tiers are not a strict height-min pyramid. Ordinary terrain replacements therefore swap as one complete topology beneath a narrow terrain-only fog pulse. Protected step-2 and step-8 fallbacks publish directly when ready, bypassing ordinary grace and the 64-transition cap because their coarser parents are not displayable. Ordinary canopies use a monotonic target-in, source-out exchange over the complete 0.65 seconds. Their transitions use union frustum bounds and do not redirect mid-flight. Skirts follow the complete terrain topology currently on screen, while water remains source-owned until completion so there is exactly one refractive water owner.

The mesher greedily combines equal flat terrain cells. Far standing water carries water-body identity and kind. A coarse cell that observes distinct authorities refines against the canonical contour instead of joining their levels. Contour-clipped shoreline triangles and top-only planar geometry prevent a partially wet cell from becoming a rectangular water ledge. Water waves remain an analytic fragment-shading effect and never displace the voxel surface. A separate `OutletFall` is one narrow receiver-centered prism with four side quads and one top quad. Its anchor's half-open tile owns all five quads, even across a tile face, and the prism overlaps the lower body's top source voxel before reaching the upper lip without raising the receiving water. Tile coordinates and sampling remain 64-bit on the CPU, while a per-draw origin restores world space from tile-local half-precision vertices.

The feature layer evaluates tree cover and species against continuous biome suitability, temperature, precipitation, soil moisture and fertility, light, slope, elevation, lithology, tectonic stress, hydrology, and ecotopes. Its accepted world-space anchors reconstruct grounded oak, large oak, birch, spruce, acacia, jungle, mangrove, palm, willow, alpine scrub, and fallen-log forms across cube boundaries. Ordinary trees reject submerged roots. Mangroves and willows alone accept suitable shallow water, and their trunks or roots extend to the sampled solid floor. Dense forest climates increase accepted canopy cover without changing the deterministic local-priority rule.

The far mesher reuses accepted exact tree anchors for step 2 without constructing ColumnPlans. Steps 4 through 32 use globally anchored 64-block aggregate forest cells. Six fixed candidates use block-8 habitat and ground authority, and coarser per-cell limits form strict stable subsets. Climate suitability, substrate, slope, water, species, and a cell-addressed priority decide compact grounded trunk-and-crown clusters rather than scaling one generic box. At step 32, the collector's block-resolution habitat and root-water authority wins. Water elsewhere in the coarse 32 by 32 cell no longer suppresses an accepted tree, and its trunk grounds on the displayed voxel. Half-open cell and tile ownership prevents duplicates across boundaries. Bit 28 of `faceAttr` marks both forms for the shared vertex contract and diagnostics. Per-column masks clip each canopy fragment only after its exact destination column is owned. The 3 by 3 mask neighborhood handles crowns crossing a far-tile face. A target-in, source-out exchange handles both nested aggregate tiers and the intentionally unrelated step-2 anchor representation without passing through an empty forest. Terrain, water, and canopy data currently share one cold-build and residency payload. Measured cold canopy work ranges from 250 to 1,165 milliseconds and can delay publication of an otherwise ready parent. Staged canopy attachment remains a follow-up performance improvement.

Exact-to-far ownership and exact-face closure are separate mechanisms. Per-column masks decide whether exact or far fragments draw. When an exact mesh lacks a current halo, its boundary scan emits the appropriate lit, dark, or bedrock closure cap until the halo arrives and invalidates that mesh. The seed-42 frontier fixture near X=69.7936, Y=85.7918, Z=-1472.94 exercises both contracts during cold streaming. The remaining deferred work is staged canopy attachment, not missing exact-face closure.

The design is an adaptive tiled terrain LOD informed by geometry clipmaps and CDLOD. It is not a literal geometry clipmap. Visibility first uses a conservative tile AABB frustum test, then processes surviving tiles front to back through a conservative 256-bin terrain-horizon culler. Sixteen 64 by 64-block patches per visible tile contribute lower-bound horizons without per-frame heap allocation. The horizon test rejects a tile only when fully covered angular bins establish a higher lower-bound horizon. This is not a hierarchical Z buffer and does not use a depth pyramid; the renderer's only depth pyramid lives in screen-space indirect lighting and serves ray traversal, not visibility. Exact opaque and far terrain use counterclockwise front faces with back-face culling. Cross and flat flora emit both windings. Shadow casters and water remain cull-none for their separate correctness requirements.

Far terrain is encoded through bounded direct indexed draws. The implementation does not build Metal indirect command buffers. Exact opaque geometry draws first and shares depth with overlapping far geometry. Resident step-32 tops provide depth-backed cold-residency fallback only outside protected exact-loading tiles; protected tiles require step 2 or step 8. Water samples resolved opaque depth for refraction and absorption while hardware-testing and writing the nearest visible interface through the media depth attachment, so water and canopies use the same current per-column ownership masks as opaque far terrain. Far water joins the same three-dimensional back-to-front water list as exact water.

CPU far-tile retention is capped at 9,280 entries and 3 GiB. Active step-32 parents are pinned under pressure, and the farthest refinement is evicted first. A residency change cancels the bounded job and completion queues immediately, then utility workers rebuild priority state and retire obsolete cache records. One maintenance pass scans at most 64 records and retires at most 32 MiB, except that one oversized record may retire alone. Mesh and membership destruction occurs outside cache locks and never on the render thread. The far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertex storage and 1 GiB of index storage. These are independent of exact mesh residency. All renderer, world, transient, and Metal allocations together must remain below the 64 GB unified-memory acceptance ceiling.

## Frame graph and graphics settings

The renderer preserves a linear HDR frame graph:

1. Upload the immutable weather snapshot, refresh atmosphere LUTs when their slow parameters change, and generate the snapped cloud-shadow transmittance map.
2. Refresh selected shadow depth targets: four detailed cascades plus one coarse horizon cascade.
3. Render the physical sky, exact and far terrain, entities, and highlights into 4x MSAA `RGBA16Float`, depth, and `RGBA8Unorm` surface data, then resolve once.
4. Build the linear min-depth pyramid, ray trace GTAO and near-field diffuse SSGI through it, accumulate with age and variance clamping, denoise with a-trous passes, and apply ambient irradiance without modifying direct, block, or emissive radiance.
5. March and temporally composite quarter-resolution volumetric clouds, then render cloud-aware lightning.
6. Copy opaque color and render post-resolve water while sampling resolved opaque depth and hardware-testing and writing media depth.
7. Inject and integrate the unified air froxel volume, or apply low-cost atmosphere-LUT aerial perspective when volumetric lighting is disabled.
8. Render depth-tested wind-driven rain or snow with atmospheric attenuation.
9. Update exposure and flare probes, build bloom, apply the single final tonemap and grade, then render UI.

The single final composite is the only linear-HDR to display conversion. Toggled-off effects skip work or bind fixed fallback textures without changing the frame graph's resource contracts.

The active direct radiance is exclusive. Sun radiance reaches zero once the solar disc is below the horizon, and moon radiance remains suppressed through civil twilight before fading in across nautical twilight. Terrain, entities, water highlights, shadows, clouds, and froxels consume that shared direct source. Atmosphere and the solar-only flare retain true solar direction and visibility for physical twilight, but neither can reintroduce below-horizon direct sunlight. `CelestialState` keeps the true solar and selected-direct roles in one world-time authority, so a clear daylight surface cannot render beneath a dark or night-like sky. A deterministic 708,734-tick mean synodic cycle derives lunar phase from saved world time. The physical phase response scales lunar diffuse light and shadows, while water specular applies one additional phase factor so a crescent cannot produce a full-Moon highlight.

The procedural block-texture array contains all five mip levels from 16 by 16 through 1 by 1. Deterministic alpha-aware downsampling preserves representable cutout coverage. The terrain sampler uses nearest magnification to retain the block aesthetic and linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy to suppress distant aliasing.

`GraphicsSettings` serializes effect toggles, quality values, view distance, and input bindings into `~/Library/Preferences/rycraft/settings.json`. `SHADOWS`, `CLOUDS`, and `INDIRECT LIGHT` expose Off, Medium, and High. The loader accepts legacy `cloudMode` and `ssao`, while the writer emits only `cloudQuality` and `indirectLightingQuality`. Settings load before `RenderPipeline` construction because they size targets and arenas. Environment overrides apply after JSON load and are never saved, so a playtest cannot overwrite preferences. `RYCRAFT_CLOUDS` and `RYCRAFT_SSAO` remain legacy aliases.

## Derived voxel light

Skylight and block light share one derived byte per materialized voxel. The high nibble stores 4-bit skylight and the low nibble stores 4-bit block light. Level-15 sky seeds exist only where complete full-height column authority proves an unobstructed path to the sky. `LightEngine::floodChunk` then propagates both channels through transparent cells, loses one level at each non-seeded step, and reports both changed state and a changed-face mask. The bounded tick-thread reconciliation queue uses those face masks to revisit and remesh only affected loaded neighbors.

Missing chunks and incomplete vertical paths remain conservatively dark until their authority arrives. Generated surface cutoffs remain the source for direct-sky seeding and provisional missing-boundary classification, but they are not a binary final light value. The propagation is monotone over fixed blocks and converges to one fixed point independent of load order. Packed light is recomputed after generation, load, unload, or edits and is never serialized. A changed field increments the cube revision and dirties its mesh.

Lighting composition keeps independent physical terms. Cascaded visibility controls only direct sun or moon radiance. Propagated skylight and baked corner accessibility control ambient irradiance. Block light and emissive radiance are never darkened by skylight, baked accessibility, GTAO, or SSGI. Outside valid shadow coverage, direct light fades toward propagated exterior visibility instead of becoming unconditionally bright.

## Runtime water boundary

World generation writes complete ocean, river, lake, aquifer, delta, crater-lake, channel-waterfall, and outlet-fall blocks directly. The canonical column plan supplies ocean, river, lake, delta, waterfall, and supported-bank topology rather than reconstructing only a surface sheet. Every generated standing-water column is a full-height source-water volume: each wet voxel from the first one above solid support through the top water voxel is an implicit source, including across cube boundaries. These generated source voxels need no explicit fluid array. An implicit source fills its complete voxel and places its visible top one block above the voxel floor. Routed rapids and outlet approaches materialize explicit flowing levels 1 through 7 at their exposed stages, while covered water and receiving pools remain sources. Waterfall curtains and outlet falls carry explicit falling state. Generation never asks the fluid scheduler to settle terrain, and loading an ordinary generated cube also schedules no water. Stable generated and runtime water emits planar top geometry only. Analytic fragment normals provide distant ripple detail without changing geometry or ownership. Vertical water sides are reserved for cells explicitly marked falling, including the short receiver-centered outlet overlay, so lake, river, and ocean boundaries cannot render as unsupported walls. The far sampled representation places generated source water on the same full-block plane as the exact implicit source voxel.

A gameplay edit activates one cell and its six neighbors. The scheduler applies downward-first source and level rules to loaded cells only. If active flow reaches a missing cube, it persists a frontier indexed by that destination cube. A later load makes only the matching index bucket eligible, and the fixed tick resumes a bounded number rather than scanning all 65,536 possible frontiers. Runtime fluid writes bypass the player activation entry point to avoid recursive scheduling. Pending-update and frontier-overflow counters remain visible in diagnostics.

This division keeps generation order-independent while localized player disturbance can evolve over time.

## Persistence boundary

Generator version three stores RYCH v4 edited cubic sections beneath `regions-v3` using 64-bit X and Z plus section Y. Its packed 44-byte header includes an IEEE CRC-32 of the uncompressed block and optional fluid payload. A per-column manifest records edited section Y values and deferred fluid frontiers. Bulk visible-column reads copy only the required manifest data under one short lock. Manifest disk I/O uses a separate serialized write lock, and the bounded save queue coalesces repeated snapshots for a cubic position to the newest revision. Metadata remains separate and preserves the seed, player position and orientation, health, selected hotbar slot, nine-slot hotbar inventory, settings, and world time. Regional weather, lunar phase, lightning buckets, and packed voxel light reconstruct from the seed, coordinates, static climate, and saved time, so they add no save migration.

Generator-version-two cube files and manifests remain beneath `regions` and are never loaded into generator-version-three terrain. They are not deleted or converted. Compatible metadata still loads, and its generator version advances after the next successful metadata save. Detailed layout and paths are in [world-generation.md](world-generation.md).

`GraphicsSettings` are preferences, not world metadata, and follow the separate settings path described above.

## Error policy

1. Metal device, queue, and pipeline failures are fatal because the game cannot render without them.
2. Cube or far-tile generation failures log and omit that result so streaming can continue.
3. Missing files return `std::nullopt`. Corrupt or incompatible cube data reports that cube's failure once, returns no cube, and regenerates deterministically.
4. Audio initialization may fail without preventing play.

There is no project-wide `Result<T, E>` wrapper. These boundaries use the smallest established mechanism for their failure mode.

## GPU boundary

Every structure shared by C++ and Metal lives in `include/render/shader_types.hpp` with size and offset assertions. Exact and far terrain preserve the 16-byte vertex format. Fluid direction occupies bits 24 through 26, falling state occupies bit 27, far-canopy impostors use bit 28, far-terrain boundary skirts use bit 29, water exterior-sky authority uses bit 30, and bit 31 remains reserved.

See [rendering-conventions.md](rendering-conventions.md) for coordinate, pass, halo, culling, and water rules.

## Tests and diagnostics

One hermetic Catch2 executable is built from twenty-one source modules covering common, world, weather, advanced world generation, geology and ecology, organic fields, rendering, entities, fauna habitats, fluid persistence, render concurrency, engine, audio, lightning, and thunder code. Cubic tests include negative floor conversion, vertical limits, uniform storage, full-width counter addressing, generation order, the complete 26-neighbor mesh halo, packed skylight and block-light convergence, indexed and budgeted fluid frontiers, bounded coalesced v4 persistence, stable territory IDs, runtime approach-triggered fleeing, and fauna movement. Weather tests cover deterministic spatial and temporal continuity, presets, climate bias, physical wind units, precipitation type, cloud profiles, bounded latest-wins publication, lightning IDs, and thunder delay. Render tests cover five-cascade selection, blend continuity, refresh cadence, shared GPU layouts, temporal-history resets, atmosphere finiteness, cloud-noise tiling, Beer-Lambert extinction, and froxel slicing. Basin tests pin canonical water support at seed 42, X=-557, Z=379, the seed-42 supported lake lip at X=-8235, Z=2976, the incised river across X=-12288 at Z=2653 and Z=2654, the canyon ecotope at X=-23904, Z=0, competitive lake watersheds, open owned connectors, monotonic junction-to-portal profiles, receiver-centered outlet falls into lower standing water, distributary deltas, bounded single-flight construction, the process-wide two-build cap across independent solver instances, concurrent clear and eviction lifetimes, reverse-order reconstruction, and cache-eviction determinism. The seed-764891 caldera sample verifies a complete irregular enclosing rim, one-block freeboard, supported dry banks, full-depth implicit source water across a cube face, and deterministic reverse-order reconstruction. A separate aquifer fixture verifies a sealed pocket.

Far-terrain tests cover full-disk parent selection, base-before-refinement scheduling, the four-parent and four-urgent split, separate parent and drawable frontiers, protected step-2 and step-8 display floors for every far-owned overlap fragment, fully ready partial boundary tiles, protected-job grace and transition-cap bypass, 16 by 16 per-column exact ownership, paired skirt-mask evaluation, 3 by 3 neighbor-mask crossings, non-occluding partial masks, conservative distance fallback, adaptive tier selection, hysteresis, ordinary transition phases, border agreement, body-aware contour-clipped shorelines, outward winding, deterministic tile hashes, cache and queue bounds, epoch cancellation, and conservative horizon behavior. They also preserve full-volume implicit generated sources in exact cubes and the matching full-block visible source plane at every far tier. Streaming and meshing tests cover lit aboveground closure caps, dark underground closure caps, vertical bedrock caps, halo invalidation, hard exploration-band priority, the seven-task exact-generation submission limit, stale-task skipping and relevant-work requeue, closed missing collision, aborted ray traversal, and skylight occlusion across unloaded vertical gaps. Terrain-only base publication remains coupled to synchronous canopy construction and therefore requires performance measurement rather than a passing ownership test. Field regressions scan former control boundaries, water IDs, curved guides, shoreline support and page continuity, lithology contacts, material patch connectivity, and deformed strata across cube, macro-control, and plate boundaries. Headless tests do not create a Metal device, so actual culling, frame rate, memory, and image quality require the playtest workflow.

Developer runs can fix the seed and spawn with `RYCRAFT_WORLD_SEED` and `RYCRAFT_SPAWN`. `RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, or `biome`. `RYCRAFT_WEATHER` accepts `clear`, `overcast`, `rain`, `storm`, or `snow`. The inspector executable provides repeatable feature locations, footprint samples, former-grid artifact measurements, water IDs, shoreline distance, lithology, material palettes, timing, hashes, and separate cache metrics. F3 displays exact required and ready sections, unresolved columns, the conservative nearest-gap distance, parent and refinement wanted, resident, drawn, and queued counts, the drawable coverage frontier, cascades and refreshes, indirect-history validity, local weather and cloud type, storm or lightning ID, caches, arenas, fluids, coalescing, and dropped work. Visual changes must still run with Metal validation and be inspected in captured frames.
