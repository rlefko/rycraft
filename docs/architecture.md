# Architecture

This document describes ownership, concurrency, persistence, and failure boundaries for generator v4. It distinguishes code that is implemented in this branch from qualification work that must pass before v4 can be treated as production-ready.

## Subsystem map

```text
src/engine   Cocoa lifecycle, bootstrap UI, MTKView frame loop, fixed 20 Hz
             simulation, input, camera, game flow, diagnostics, and hooks
src/render   HDR Metal frame graph, exact cubic and far-terrain meshing,
             shadows, screen-space lighting, atmosphere, clouds, froxels,
             water, lightning, post effects, UI, particles, textures, GPU
             arenas, and entity rendering
src/world    Learned authority, runtime loader, canonical hydrology, cubic
             storage, generation, features, streaming, derived lighting,
             regional weather, runtime fluids, and persistence
src/entity   Player physics, fauna physics and AI, habitat territories,
             spawning, and spatial queries
src/audio    Core Audio mixer and procedural sound effects
src/common   Math, coordinate utilities, randomness, workers, logging
```

The dependency direction is `engine -> render/world/entity/audio -> common`. Rendering consumes immutable world snapshots. Gameplay edits enter through `World`.

## Startup boundary

Ordinary startup reaches the title screen without selecting a world or starting terrain setup. Selecting a compatible v4 profile, creating a fresh profile, or explicitly creating a successor begins the qualified startup path. That path does not construct `SaveManager`, `World`, `Spawner`, generation workers, or far workers until terrain setup succeeds.

```text
ModelRequired
  -> Downloading
  -> Verifying
  -> Compiling
  -> Loading and qualification
  -> Ready
```

Any failure enters a latched `Failed` state. Retry, repair, cancellation, and quit are explicit UI actions. A failure does not create `rycraft_world_v4`, generate placeholder cubes, or select the v3 generator. `RYCRAFT_DIAGNOSTIC_V3=1` is the only legacy entry point, and it is intentionally no-save.

The installer owns a persistent revision-specific staging directory below `~/Library/Application Support/rycraft/terrain-models`. It continues short files with HTTP range requests and verifies every file's exact byte count and SHA-256 before an atomic directory rename publishes a fresh pack. Repair replaces only invalid assets in the installed directory, so valid model files, the extracted runtime, and Core ML caches survive. The runtime archive is extracted only after its archive verification. Model data and Core ML caches remain outside Git and outside Conductor workspaces.

An installed pack candidate starts verification and loading automatically after an explicit world request. Opening the application alone does not inspect the pack. Ordinary world open and Retry never download over an installed pack. If the completion marker is missing or stale but every pinned asset passes exact size and SHA-256 verification, startup restores the marker and continues without a download. A failed asset verification remains latched until the user explicitly chooses Repair, which authorizes replacement of only the assets that fail verification.

The production runtime loads the verified ONNX Runtime 1.27.1 dylib with `dlopen`. It requires an Apple Silicon process on macOS 14 or newer, C API version 27, and an available Core ML execution provider. The verified dylib has one process-lifetime mapping because ONNX static operator registries can outlive an individual runtime environment. Retry and world teardown release all sessions and the environment deterministically, but they never call `dlclose`; a process-local owner also prevents retries from loading duplicate images. Compilation creates exactly one static Core ML session for each pinned graph: coarse, base, and decoder. The three sessions remain resident until retry, repair, or process shutdown, so a graph switch never destroys and recompiles a session. Graph execution stays sequential and the inference coordinator permits only one active call. Every session uses static shapes, ML Program format, all compute units, and the persistent model cache. Coarse retains its qualified scalar batch. Base binds the published symbolic `batch` dimension to the pipeline's immutable four-window latent batch. Decoder binds batch four and the 256 by 256 spatial dimensions, repeats the last real window to fill a tail batch, and discards padded outputs. This provider choice is fingerprinted and uses `coreml-cache-v3-base4-decoder4x256`, separate from caches compiled under earlier provider metadata.

CPU fallback uses an explicit intra-op total equal to `min(hw.physicalcpu, 16)`, with a conservative one-thread fallback if physical-core detection fails. ONNX Runtime counts the caller in that total. Inter-op execution remains one because graphs do not run concurrently. Provider partition counts, CPU fallback counts, configured CPU threads, compiled sessions, current resident sessions, and peak resident sessions are recorded. Resident session count must remain at three or fewer.

The canonical qualification query runs all three pinned graphs and hashes provider-stable quantized output. Startup refuses to become ready if that digest differs from the compiled baseline. `CANONICAL_QUALIFICATION_HASH` records `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df` for the pinned graphs, runtime, scalar Coarse graph, and static four-window Base and Decoder batches on the documented M4 Max. This recorded digest is a startup comparison input, not evidence that fresh, reverse, concurrent, cache-cleared, entry, visual, or performance qualification has passed.

## Runtime ownership

- `EngineState` owns the `World`, `WeatherSystem`, `SaveManager`, `Spawner`, audio, input, player state, wetness, graphics settings, the 36-slot `Inventory`, survival stats, block entities, dropped items, boats, and per-world configuration. It retains one immutable weather snapshot for each frame and fixed tick.
- A world session is created and torn down at runtime, not only at process start. `-init` reaches the title with no world; the verified v4 open path builds `SaveManager`, `World`, `Spawner`, and, when enabled, `WeatherSystem`. `-stopWorld` first requires a successful save and then destroys them in strict order: `RenderPipeline::endWorldSession`, `Spawner`, `WeatherSystem`, `World`, and `SaveManager`. With no world session the frame renders a menu-only pass.
- `World` holds a non-owning `SaveManager*`. The engine constructs the save manager first and destroys it after the world.
- `World` uniquely owns the cubic map, one active-set planner, six-worker generation pool, bounded generation backlog, column generator, packed voxel-light reconciliation queue, and fluid scheduler. Generation jobs capture `World` but never retain the pool that executes them.
- `WeatherSystem` owns one joinable utility worker and publishes immutable camera-centered snapshots with latest-wins admission. Gameplay retains the prior valid snapshot while a replacement is built.
- `RenderPipeline` owns the exact four-worker `MeshScheduler`, sixteen-worker far-terrain scheduler, a bounded canopy scheduler with at most one admitted gameplay worker, exact and far mesh registries, GPU arenas, frame ring, filtered block textures and emission masks, five-cascade shadow map, `ScreenSpaceLighting`, `AtmosphereRenderer`, `CloudRenderer`, froxel `Volumetrics`, lightning, water, bloom, post stack, entity and item renderers, particles, and UI. Base terrain and water residency is independent of canopy attachment residency. Ordinary outer-parent submission and publication pause whenever exact publication through 32 chunks or connected visible desired-LOD debt remains. Local far work may admit 8 workers alongside exact debt or 12 after exact debt clears, and all 16 return only after both debts clear. Canopy admission stays at zero during preparation and until the connected 96-chunk prefix is drawable. Gameplay then guarantees exactly one low-priority canopy worker, including while stronger terrain debt continues. Missing PREVIEW attachments precede FINAL promotion, and no second gameplay canopy lane opens.
- `Spawner` owns entity shared pointers and the spatial hash. Stable territory IDs identify wild fauna independently of visit order.
- `World::~World` marks shutdown, cancels and joins the active-set planner, explicitly drains and joins its uniquely owned generation pool, then consumes ready generation futures without holding the pending-work mutex. The explicit pool boundary prevents the last pool owner from moving onto one of its workers and attempting a self-join during world teardown.
- AppKit application termination uses one idempotent quiescence sequence because `exit` does not release the shared `Engine` singleton. It completes the durable world save before canceling bootstrap admission, joins exact, far, canopy, and world generation workers while their dependencies remain live, releases every learned-generation context, then destroys the production runtime so its sessions, environment, and global worker pool stop before process static finalizers run. The verified dylib remains mapped. Repeated window-close, menu-quit, capture-quit, and termination-delegate callbacks perform no duplicate teardown, and a defensive `dealloc` path applies the same dependency order.
- The engine shuts down exact and far mesh workers before destroying the world because exact jobs read snapshots from it and both schedulers own active threads.

## Generation identity

`GenerationIdentity` is immutable and includes:

- Generator version and unsigned 64-bit seed
- Model-pack and runtime hashes
- Provider and ONNX Runtime version
- Four-block model scale
- RNG, quantization, hydrology, and postprocessing revisions
- Coarse, latent, and decoder window geometry

The SHA-256 fingerprint of the encoded identity is the authority namespace. `WorldGenerationContext` owns one identity, one `TerrainAuthority`, and one `AuthorityQuality`. It provides bounded native-grid and world-coordinate queries, metrics, and a latched `GenerationFailure`. Consumers must not catch a learned-authority failure and substitute synthetic macro terrain.

The v4-aware constructors for `World`, `ChunkGenerator`, `MacroGenerationSampler`, and `FarTerrainScheduler` accept the same shared context. A context-free constructor remains for legacy tests and the explicit v3 diagnostic path.

## Learned authority layers

```text
unsigned 64-bit seed plus generation identity
  -> globally addressed native model pixels
  -> preview or final TerrainAuthority pages
  -> learned elevation, four learned climate variables, and derived lapse rate
  -> MacroGenerationSampler physical-climate adapter
  -> canonical hydrology and learned terrain bed correction
  -> immutable ColumnPlan
  -> independent 16 x 16 x 16 cube emission
  -> exact meshes and far surface-stage meshes
```

Model rows map to world Z and model columns map to world X. One 30-meter native pixel covers four blocks, so the horizontal and positive-elevation scale is 7.5 meters per block. Negative coordinates use floor division and half-open ownership.

The authority foundation implements global window geometry, deterministic PCG64 and Marsaglia Gaussian primitives, lexicographically stable weighted accumulation, bounded page queries, and an ordinary-CI fake backend. `InfiniteDiffusionBackend` implements the scalar coarse denoising loop, fixed four-item latent batches, lexicographically grouped fixed four-item Decoder batches with deterministic repeated-tail padding, signed-square-root elevation transform, Laplacian cleanup, climate reconstruction, and a 384 MiB tensor-window cache through the typed runtime executor. Postprocessing revision 9 reconstructs PREVIEW from the FINAL latent low-frequency channel with the same cleanup operator while retaining the published half-pixel, `align_corners=false` registration at positive and negative coordinates. Python and Java compatibility goldens and real-model output hashes remain qualification requirements, so source coverage alone is not a passing paper-faithfulness result.

Learned elevation, four learned climate variables, and the derived lapse-rate field become the macro height and climate source whenever a v4 context is present. The context-free legacy path retains synthetic plate relief and climate. Geology, rock resistance, strata, caves, ores, aquifers, structures, and volcanic primitives remain bounded procedural systems. Deterministic shield, stratovolcano, and warped-caldera relief is added to learned physical elevation before native hydrology, so any crater lake is a canonical depression-hierarchy result rather than a post-routing overlay. The v4 path does not run legacy `BasinSolver` hydraulic erosion, alpine morphology, or the analytical crater-lake overlay. It permits a separate footprint-filtered dry residual of at most 1.5 blocks only after water, outlet, rim, coast, divide, and slope clearance gates pass. Far-water candidate markers remain conservative refinement evidence, not permission to introduce water. Biomes, flora, canopy, and fauna currently use the physical-climate adapter. The adapter derives soil moisture from the learned annual water balance and variability rather than the legacy normalized-rainfall formula. Plant-functional-type equilibrium is deferred to the follow-up PR.

## Authority persistence

Each `TerrainAuthorityPage` covers 256 by 256 native samples, or 1,024 by 1,024 blocks. Preview and final pages use separate directories.

An `RYTA` file contains a 92-byte header followed by an LZ4 payload. The header records schema, quality, compression, signed page row and column, unsigned 64-bit seed, 32-byte generation fingerprint, native edge, channel count and mask, payload sizes, payload CRC-32, and header CRC-32. Each 12-byte sample stores:

- Elevation in meters
- Mean temperature in centidegrees Celsius
- Temperature variability in centidegrees Celsius
- Annual precipitation in millimeters
- Precipitation coefficient of variation in basis points
- Lapse rate in microdegrees Celsius per meter

Writes use a unique sibling temporary file, `fsync`, rename, and directory synchronization. A persisted page is immutable for its identity. Every seed or fingerprint mismatch, including a same-seed stale PREVIEW page, remains fail-closed. World profile metadata must select a separate identity namespace before generation starts. A corrupt RYHY envelope is rebuilt and atomically replaced, while an outer-valid RYHY page is replaced only when the native router proves the exact opaque payload is corrupt. No consumer may mix old and current page identities.

Spawn and protected-handoff hydrology also retain their exact quantized FINAL input rectangles under
`terrain-authority-v1/transient-final-v1`. Each LZ4-compressed `RYTG` file binds its half-open
native rectangle, seed, and full generation fingerprint to payload and header CRC-32 values. The
same atomic publication and corruption-repair rules apply. Visible and speculative transient
rectangles remain memory-only. This distinction lets a finalized world restart from its canonical
hydrology inputs without reconstructing the same coarse, Base, and decoder windows.

`CachedTerrainAuthority` bounds one query to 64 pages and 1,048,576 samples. The default decoded page cache is capped at 1,024 entries and 512 MiB. At most 64 requests may be outstanding and one page build may run at a time. Equal cold requests share a single flight. `InfiniteDiffusionBackend` owns an independent 384 MiB least-recently-used tensor-window cache. The shared coordinator services spawn, exploration exact, protected exact handoff, visible final refinement, coarse preview, and speculative movement prefetch in that order. Final step-32 parents in the exact handoff use the protected lane. After visible preview authority is ready, movement may admit at most eight pages immediately beyond the leading horizon through the speculative lane.

## Cubic world ownership

The world spans Y=-128 through Y=1407, represented by section Y=-8 through Y=87. `VerticalSectionMask` uses two 64-bit words for all 96 sections. It supports set, reset, empty, single-section containment, inclusive range containment, and highest-section lookup without shifting by 64 or more.

`ColumnPos`, `ChunkPos`, and `BlockPos` are canonical keys. X and Z are 64-bit; block and section Y are 32-bit. `Chunk` stores one mutable 16 by 16 by 16 cube and materializes its dense block or fluid arrays only when needed.

V4 cold entry uses a zero nominal exact radius while it waits for a final safe spawn and the
connected coarse terrain-and-water entry horizon. The mandatory mesh halo still keeps the camera
column and its four cardinal neighboring columns in the exact active set, while plan dependencies
extend four chunks from the center. This prevents cold entry from opening a second hydrology owner
solely for optional pre-entry exact coverage. Spawn readiness waits for the accepted center column
plan, not every plan in that dependency footprint. Canonical spawn certification remains separate
from playable collision readiness. Before control is released, the already-retained one-cube mesh
halo must be resident across the three sections centered on the player's section. This 27-cube
residency gate overlaps horizon work, adds no model pages, and preserves the closed missing-cube
collision fallback. The user's configured visible radius remains selected, up to 512 chunks, but
gameplay entry requires a connected parent frontier through 96 chunks. That entry radius is six
far-tile widths even when the later gameplay view is configured differently. The renderer suppresses every tile at or beyond the nearest
missing parent, so a protected diagonal outside that radial prefix cannot appear as an island.
Readiness still validates the configured selection, world epoch, view epoch, and center tile, so a
smaller or stale selection cannot open gameplay. Exact mesh publication advances throughout
preparation. Once the connected prefix reaches the near band, preparation also opens the
camera-critical protected FINAL lane. Its camera-aware closure contains 4 targets at step 1, 8 at
step 2, 12 at step 4, 16 at step 8, and 20 at step 16. All 60 targets must publish atomically before
entry. Optional flora and ordinary refinement remain closed until the first gameplay frame while
the configured parent disk continues filling behind the connected frontier. After entry, every
required surface section through the full 32-chunk exact radius keeps a protected generation,
meshing, and upload lane until revision-ready. The camera column is first, the six-chunk exploration
band is second, and the rest of the exact disk is third. Optional exact flora and far canopy may use
capacity only after those terrain obligations.

Exact collision publication is a renderer-owned frame snapshot, not a side effect of a cube merely
being loaded. The snapshot names section positions and the matching `ExactSurfaceCoverageSnapshot`
epoch. A stale publication is rejected. A named section reads its loaded exact blocks and treats a
missing named cube as closed. Until a section is named for the current epoch, collision and
submersion use the immutable `ColumnPlan` terrain and canonical generated-fluid state. An unresolved
column remains closed. This keeps physics on the same ownership boundary as the visible handoff and
prevents partially loaded exact cubes from opening holes or invisible walls beneath a far surface.

A fresh v4 profile first samples the nonpersistent coarse model directly. One coarse cell spans
256 native pixels, or one 1,024-block authority page. At the 7.5-meter block scale, that is
7.68 kilometers. The selector queries one page-aligned 16-cell square, whose 61.44-kilometer
half-edge is the largest representable square below the 64-kilometer search contract. It does not
create preview authority pages while searching. Coarse proposals are grouped by aligned two-by-two
page hydrology owner, and only the best proposal for each owner is retained. Startup therefore
cannot route the same rejected owner repeatedly.

The canonical screen directly prepares one 2,048-block native hydrology owner. It checks the
requested chunk first, then a bounded 16-worker scan examines the owner's globally aligned
four-block native raster for the nearest deterministic center whose five-by-five safety buffer is
flat and dry. The screen first tries a certificate for the complete cold footprint, then the exact
113 by 113 radius-zero safety footprint. It may relocate within the same hydrology owner and already
materialized learned page after a four-block mask prefilter. At most 64 exact candidates are tried.
A successful exact-safety certificate lets the center plans validate collision and headroom without
opening cardinal hydrology owners. Wider canonical water rejects only this fast path, after which the
screen installs the original 25-sample local certificate. A continental owner with positive learned
elevation but no conservative page-local ocean-escape proof can return one provisional learned site
without installing a dry certificate. That provisional result may start radius-zero exact generation,
but it cannot start the far horizon or persist metadata. Exact `ColumnPlan` validation must still
prove canonical water absence, support, headroom, slope, and the nearby dry neighborhood. A rejection
advances to the next deduplicated owner. Once exact validation succeeds, the zero-radius cold exact
set and configured preview horizon stream normally. The full 32-chunk exact disk waits until entry.
Metadata records a spawn-safety revision. Older v4 metadata
is revalidated once and relocated through the same bounded process if it no longer satisfies the
dry-land contract. An all-ocean bounded result fails closed instead of falling back to an ocean
coordinate.

The 193-level density lattice is evaluated lazily for the vertical interval needed by a cube and its surface neighborhood. Height-sensitive loops must use `WORLD_MIN_Y`, `WORLD_MAX_Y`, `WORLD_MIN_CHUNK_Y`, `WORLD_MAX_CHUNK_Y`, or `WORLD_VERTICAL_CHUNKS`, not a former 40-section assumption.

## Canonical water boundary

Generated water is immutable authority, not a fluid simulation. Runtime fluids begin only after a gameplay edit.

The v4 water path removes the conflict-bank, bank-dilation, dry-wall, and shore-raising outcomes that caused long retaining walls. Hydrology may lower terrain to form a supported bed or open a route. It may not raise dry terrain only to contain water, and it may not delete a conflicting wet route. Adjacent stages must reconcile to one compatible body, a lowered natural spill, an opened outlet, or an explicitly owned rapid or fall. The retained legacy v3 `BasinSolver`, including its iterative hydraulic-erosion passes, remains isolated behind the explicit diagnostic path. V4 instead routes native learned elevation and applies only the resulting canonical bed cut.

Exact cube generation and far terrain consume the same `HydrologySample` fields, including stable body ID, water stage, bed relationship, flow, discharge, signed shoreline distance, seasonality-related climate inputs, and explicit transition ownership where implemented. Standing water remains a full source volume above solid support. Vertical discontinuities are legal only for an explicit rapid or waterfall.

The implemented v4 router uses deterministic local native-page Priority-Flood and explicit shared-edge handoff reconciliation. It routes the learned physical elevation field in meters, with 30-meter native cells, before emitting block-height authority for terrain, stages, and beds. It does not approximate a global lake by applying page-edge minima to every lower cell, because that would merge disconnected depressions and make the step-32 topology reduction disagree with sampled water. Wet opposing edge summaries enter a deterministic tiled spill hierarchy. A persisted receiving-edge escape hint also detects a basin that is locally wet on only one side of a page boundary; the router then reruns the minimax flood on a combined immutable rectangle. Both paths share a 64-page fail-closed bound, flat stage and mass authority, stable identity, and exact plus far signed shoreline output.

Shallow, stable-groundwater candidates form a deterministic downstream graph through both significant D-infinity receivers. A candidate can inherit through any number of other candidates until the bounded graph reaches a finished ocean, lake, river, or wetland. The solve crosses half-open page owners, admits at most 64 pages and 262,144 native cells, caches every resolved node, and fails closed at either bound. Every connected wetland cell inherits the selected parent stage and body ID, promotes hydraulic head to that stage, and lowers its bed by one eighth block. It never raises terrain or invents an isolated wetland stage.

A low-gradient river mouth establishes a persisted estuary and brackish identity. Sea backwater is followed across immutable page owners for at most 64 native cells, never crosses an explicit fall, and keeps the river bed below the sea-backed stage. At the mouth, the router retains the physical D-infinity outlet and adds one deterministic, discharge-conserving distributary into a separated ocean target within four native cells. Both branch weights, the delta and estuary flags, the river or ocean body identity, the receiver graph, and the frozen two-bit fall-branch partition are part of the schema-6 RYHY payload. Exact columns, dense sampling grids, and far topology all consume that same graph. The remaining water qualification work is real-model visual evidence and bounded-overflow reporting, not a separate delta, estuary, connected-wetland, or crater-lake implementation.

## Far terrain publication

`FarTerrainMesh` is the immutable base payload for terrain, standing water, and falls. `FarCanopyAttachment` is a separate optional vegetation payload with its own queue, cache, upload, and allocation. It contains tree forms and deterministic ground-flora aggregates selected from the same habitat rules as exact flora. Construction is lower priority than local terrain work. Its worker budget is zero during preparation and until the connected 96-chunk terrain-and-water prefix is drawable. Gameplay then keeps exactly one low-priority canopy worker available even while stronger terrain debt continues. Missing nearby vegetation publishes from PREVIEW ecology before FINAL ecology promotion, using the currently displayed PREVIEW or FINAL terrain for grounding. The provisional attachment remains drawable until its FINAL replacement is resident. Cancellation and attachment-cache replacement never alter the already drawable base mesh. Presence of an attachment entry, including an explicit empty attachment, is the sole flora completion state.

Step-32 water discovery does not depend only on four corners. `waterTopologyPossible` requests bounded interior probes and finer shoreline reconstruction for a coarse cell that may contain a channel or contour. Half-open tile ownership prevents duplicate water and waterfall faces.

Production far meshes emit no downward crack-hiding skirt quads. The outer cell ring triangulates a canonical two-block boundary lattice into the coarse interior, giving adjacent tiers identical edge heights and half-open positive-area ownership. Runtime selection keeps displayed neighbors within one power-of-two tier and advances replacements one tier at a time. Automated tests cover topology and ownership, while visual LOD qualification must still inspect every 2:1 join and report any residual crack, panel, or ledge.

Parent and refinement residency remain separate. A coarse parent stays available until an eligible replacement is resident. Exact ownership uses per-column masks, and a delayed canopy cannot invalidate drawable terrain or water.

While the entry screen is visible, `RenderPipeline::renderV4Preparation` advances exact mesh publication, far-authority polling, base-mesh scheduling, result draining, and shared-buffer publication inside the menu frame without issuing the full world scene. Once the connected frontier reaches the near band, the same preparation path starts the camera-critical protected FINAL closure while entry-prefix parents continue. Gameplay opens after the connected step-32 terrain-and-water frontier reaches 96 chunks and the FINAL spawn, 27-cube collision, revision-ready exact mesh, and 60-target protected FINAL gates are ready. Those targets are distributed 4, 8, 12, 16, and 20 across steps 1, 2, 4, 8, and 16. After that prefix, exact publication through 32 chunks and the nearest connected desired-LOD misses pause ordinary outer-parent submission and publication. Near jobs run nearest-first, rank distance before projected error within the nearby visible class, and may displace queued or dependency-parked outer parents. Outer submission and publication resume only after both debts clear. Gameplay keeps one low-priority canopy worker after the connected prefix is drawable. This keeps preparation work from being paced by the increasingly expensive scene that it is trying to prepare.

## Persistence profiles

The default v4 world path is:

```text
~/Library/Application Support/rycraft/rycraft_world_v4/
  metadata.json
  regions-v4/
  terrain-authority-v1/
    transient-final-v1/
  hydrology-authority-v1/
```

The Worlds screen reserves a fresh collision-free directory, but publishes it only after model and runtime qualification. Metadata records generator version 4, chunk format version, unsigned 64-bit seed, full generation fingerprint, player state, and world time. Opening an existing profile requires its exact seed and identity. A mismatch returns to world selection without inventing or opening a sibling. An explicit fresh creation or successor action selects a separate identity-named path and never overwrites an occupied directory. The selected path is reported in startup diagnostics, and the terrain and hydrology authority roots are rebound to that profile before generation begins.

Ordinary startup does not inspect or migrate legacy world data. The Worlds screen lists legacy and incompatible profiles as read-only successor sources. The explicit successor action copies compatible seed, game mode, generation settings, player metadata, and world time into a separate current-v4 profile. It does not rewrite the source regions, manifests, edits, or fluid frontiers. `SaveManager::Profile` keeps the paths distinct.

## Threading and failure policy

| Context | Allowed work |
|---|---|
| Main and fixed-tick threads | Completed-authority reads, bounded enqueueing, input, player and fauna simulation, survival stats, mining progress, dropped-item and boat physics, boat riding, furnace ticking, fluid scheduling, bounded GPU uploads, command encoding, UI |
| Active-set planner | Coalesce camera requests, select exact cubes and plan dependencies, publish retention, and unload stale cubes |
| Six generation workers | Load or generate cubic chunks and construct cold column plans and native hydrology pages |
| Four exact mesh workers | Copy a bounded snapshot, build greedy cubic geometry without the world lock, return versioned results |
| Sixteen far-terrain workers | Pause ordinary outer submission and publication during exact or local desired-LOD debt; admit 8 local workers alongside exact debt, 12 for local debt after exact clears, and all 16 only after both clear |
| Canopy scheduler | Use zero workers during preparation and until the connected 96-chunk prefix is drawable; guarantee exactly one low-priority gameplay worker thereafter |
| Terrain bootstrap thread | Download, verify, compile, load, and qualify the model pack |
| Learned inference coordinator | Execute one active model call with bounded outstanding page work |
| Weather utility worker | Batch static climate samples and publish the latest regional weather request |
| Save thread | Serialize, LZ4-compress, and atomically replace edited cube files and manifests |
| Core Audio callback | Mix active voices under the audio voice mutex |
| Render thread | Snapshot reads, bounded result drains, Metal encoding |

Render and fixed-tick threads must not perform or wait on model inference. `NativeHydrologyRouter` uses one process-wide admission gate for preview and final contexts. It admits at most 16 page builds, further limited by reported hardware concurrency and a one-GiB scratch reservation, rather than allowing each context to claim its own 16-build budget. Generation failure is latched and user-visible. Existing resident geometry stays available, missing collision remains conservatively closed, and new generation does not publish an empty cube.

## Qualification status

Ordinary CI uses the deterministic fake authority and must not download the model pack. It covers identity encoding, negative coordinates, window intersection, accumulation order, portable randomness, bounded queries, single flight, cache behavior, page CRC and corruption repair, fingerprint rejection, bootstrap states, v4 path isolation, both words of the vertical mask, and selected far water and canopy regressions.

The simulation ticks at 20 Hz. Runtime water uses the same fixed rate and a five-tick delay. Weather snapshots contain two slices ten real seconds apart and interpolate by saved world time. Rendering uses the latest simulation and immutable weather state without an additional gameplay interpolation frame.

The following are required evidence, not established facts in this document:

- Identical real-model hashes across fresh, reverse, concurrent, and cache-cleared runs
- A safe final spawn, connected coarse terrain and canonical water through 96 chunks, revision-ready exact spawn meshes, and the 60-target protected FINAL closure within 30 seconds
- Five-minute cold settlement of the complete configured 512-chunk parent disk and all remaining required queues
- Native-resolution M4 Max performance at 4x MSAA and view distance 512
- Total unified memory at or below 64 GiB
- Metal validation with inspected captures
- Zero cracks at every far LOD join after skirts are removed
- Complete canonical hydrology behavior for every claimed water-body type

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
TerrainAuthority cache mutex                    leaf
NativeHydrologyRouter cache mutex               leaf
NativeHydrologyBuildGate permit mutex           leaf and never nested
```

- Never generate, build a column plan or hydrology page, load, compress, or perform file I/O while holding `chunksMutex_`.
- Synchronous `getChunk` checks both presence and loaded capacity under the mutex, releases it for load or generation, then rechecks both before `try_emplace`. Duplicate work is allowed; overwriting a loaded edit or exceeding the loaded cap is not. A capacity rejection returns no cube, and a loading block query treats that unavailable cube as closed.
- `snapshotForMeshing` is the bounded exception. It copies an 18 by 18 by 18 block, fluid, and packed voxel-light halo plus separate 18 by 18 generated-surface and sky-cutoff authority under `chunksMutex_`, then releases the lock before meshing.
- Never wait on work that can take a mutex currently held by the waiting thread.
- Fluid queries use only loaded cubes. They do not call the loading `getBlock` or `getChunk` path.
- Save serialization and compression run on the save thread. The queue is capped at 32,768 positions, and the pending-save map coalesces repeated snapshots for one position while keeping the latest queued cube visible to loads. Manifest disk writes serialize under `manifestWriteMutex_` after copying or updating state under the short-lived lookup mutex.
- Single-flight caches publish shared work under their own mutex and perform expensive construction after releasing it.
- The terrain and native-hydrology authorities release their cache mutexes before waiting on an existing flight, acquiring the process-wide build permit, running inference, or performing file I/O. Permit mutexes are never nested with authority caches. The retained `BasinSolver` follows its separate legacy diagnostic lock contract.

See [performance-conventions.md](performance-conventions.md) for caps and review questions.

## Exact cubic streaming and mesh consistency

The exact active set never follows the 512-chunk visible horizon. Gameplay rebuilds it on the dedicated planner, not during a fixed tick or render pass. Camera movement replaces any queued request, and an in-progress request checks the latest epoch before every major phase and immediately before publication. Each rebuild gathers one unique set of horizontal columns, expands the fixed plan apron once, and registers pending-plan dependents in an index. A plan completion addresses only that dependency bucket, and completion notifications coalesce before the next rebuild instead of scanning the retained cube map once per result. Surface and exposed ownership beneath the current camera column use the highest camera lane even during flight. Surface ownership inside the six-chunk exploration disk uses the exploration lane. Every other required surface section through the complete 32-chunk exact disk uses the next protected terrain lane, ahead of optional flora and broad primary work. The same camera, exploration, full-disk surface, flora, and broad order governs generation submission, mesh queue admission, completed-result upload, and deferred first-publication lighting. This changes ordering only and opens no additional model pages. The active set is bounded by `min(viewDistance, 32)` and combines:

- a camera exploration band with radius six chunks and four cubes above and below the camera;
- saved edited sections copied for the unique visible-column set through one bulk, short-lock manifest query;
- one primary surface section for each resolved or unresolved visible column;
- additional exposed and cliff-wall sections from the resolved column plans;
- a complete one-cube halo for collision, light, and meshing.

The mesh-candidate set is capped at 16,384 cubes. The retained set, including halo cubes, is capped at 32,768. Capacity is reserved for the exploration and collision band first, then every required surface section through 32 chunks, edited sections, optional flora-bearing sections, remaining primary or exposed sections, and halos. Distance resolves ties inside each class. An omitted surface fragment remains represented by its far parent dependency and by step-2 or step-8 display geometry according to its protected overlap class. Existing cubes remain retained through two extra horizontal chunks and one extra vertical cube, giving unload a concrete hysteresis boundary outside the current target. When retention changes, obsolete cubes unload before replacement jobs are submitted. Asynchronous and synchronous publication both enforce the same cap under `chunksMutex_`.

Every exact mesh is identified by `ChunkPos` and a cube revision. An edit increments the owning cube and every face, edge, or corner neighbor whose one-block halo intersects the changed block. Worker jobs retain both the renderer request revision and the revision captured by their snapshot. Coalescing preserves the completion for the newest request, including a failed snapshot, but the render thread publishes only a result that matches the live cube revision and is newer than any resident mesh. A rejected or failed completion clears only its matching request, so it cannot replace newer geometry or cancel a newer request. Any previously published exact mesh remains available while the current revision is rebuilt.

`MeshSnapshot` carries one block of padding on all axes. Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real faces, edges, corners, and packed voxel light. If an in-range neighbor is still unavailable, its halo follows the immutable generated surface cutoff. Cells above that terrain silhouette remain air, cells below it remain conservatively opaque, and a missing cardinal face seals only unresolved boundary cells. For a lateral face with a possible cap, the mesher floods transparent cells through the bounded 18 by 18 by 18 snapshot from sky-exposed seeds. Complete column authority is authoritative for both raised and lowered roofs. The distinct incomplete-path marker instead uses the generated cutoff only to keep an aboveground provisional boundary classified as sky-connected while propagated skylight remains dark. Sky-connected continuations receive a lit provisional face using one representative arriving-surface material per missing face. Enclosed lateral openings receive dark stone, while missing vertical openings receive bedrock. This prevents full 16 by 16 black panels above ground without opening an underground void. The flood runs only after a cheap boundary-candidate scan. Loading or unloading any halo cube reconciles the affected light faces and invalidates neighboring meshes that sampled changed halo data. World-floor and world-ceiling cells use the documented bedrock and air boundary values.

The world maintains a compact loaded-section mask per horizontal column. Only a continuous loaded path to the open sky may seed level-15 skylight. An incomplete path remains dark, preventing sunlight from crossing unloaded space above an underground camera. Cube publication compares the prior and new masks and runs one bounded lighting transaction under the world lock. That transaction settles the arriving cube, changed sky paths, and affected face neighbors. Work beyond the per-transaction flood cap enters the camera-ranked publication queue, and `snapshotForMeshing` refuses every pending cube. The first visible mesh therefore observes one settled packed-light revision without making an unbounded generation-worker transaction.

Collision also treats unresolved cubes as closed. For ordinary movement and fluid-height queries, the renderer publishes revision-ready exact section ownership with the same coverage epoch used for visual suppression. A matching owned section reads exact blocks and fluid cells. An unowned section reads canonical planned terrain and generated water, while a missing plan stays closed. DDA targeting still stops at the first unavailable exact cube, and breaking or placement revalidates loaded ownership before mutation. Interaction therefore cannot force generation or edit through a temporary mesh cap.

Meshes keep vertices cube-local. `ChunkOrigin` restores X, Y, and Z world position in the vertex shader. AABBs, frustum tests, candidate distance, water sorting, and buffer ownership are all three-dimensional.

## Far-terrain visibility architecture

The far renderer selects every immutable 256 by 256-block tile intersecting the radius-512 visible disk, not just a configured annulus. Every selected coordinate requests a step-32 parent in nearest-first order. A broad parent lane advances the connected 96-chunk entry prefix. A missing PREVIEW parent in the current protected closure uses urgent coverage admission and may displace a lower-ranked queued or dependency-parked ordinary parent at scheduler capacity. As soon as the connected prefix reaches the near band during preparation, a protected lane advances the camera closure while entry-prefix parents continue. After the prefix is drawable, exact publication through 32 chunks and every connected desired-LOD miss become local debt and pause ordinary outer-parent submission and publication. Near jobs run nearest-first and may displace queued or dependency-parked outer parents, but displayed parents, the connected prefix, transitions, exact fallbacks, and protected lineage remain pinned. Every ordinary coordinate progresses through adjacent tiers, so a displayed step-32 parent requests step 16 before step 8, step 4, step 2, or step 1 even when a finer CPU cache entry is already ready. Base-lineage PREVIEW proxies may occupy steps 16 through 2 until their same-key FINAL replacements are ready. Step 1 is always FINAL. The protected anchor is the minimum corner of a position-aware two-by-two FINAL step-1 core. Manhattan distances zero through four use FINAL steps 1, 2, 4, 8, and 16, producing 4, 8, 12, 16, and 20 targets, for 60 total, and 100 matching internal boundaries before atomic publication. Current protected FINAL children and parents receive the first urgent capacity. No more than one third of the bounded frame allowance serves required PREVIEW bridges, unused bridge capacity returns to current FINAL parents, and only remaining capacity may pre-stage one directional predicted anchor. Predicted work remains CPU-only until the canonical anchor advances and cannot affect publication epochs, GPU residency, display, or closure statistics. The camera, protected, fallback, and connected-wavefront classes rank first. Within the nearby visible class, horizontal distance ranks before projected error, so screen-space error chooses desired quality without starving closer missing detail. At most four ordinary visible FINAL parents remain in flight, and sixteen learned-authority coordinator slots stay available beyond coarse preview admission. Local far admission is 8 workers alongside exact debt and 12 after exact debt clears. All 16 and ordinary outer submission and publication return only after exact and local debt clear. Canopy uses one low-priority gameplay worker after the prefix is drawable. Speculative work waits until visible canonical replacements advance.

Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks missing step-32 dependencies. Exact cubes own the first 32 chunks. Settled far bands require step 2 through 64 chunks, step 4 through 128, step 8 through 256, and step 16 through 512. Those bands are hard maximum-coarseness limits. Gameplay may retain a finer tier, including ordinary FINAL step 1, when the conservative projected error derived from distance, viewport height, vertical FOV, and step-32 parent relief exceeds 0.55 pixels. Step 1 is the irreducible voxel-grid floor. Outward screen-error hysteresis validates only one adjacent coarsening tier at a time and coarsens only below 0.45 pixels. PREVIEW scheduling expands relief by twice the measured 46-block residual and its vertical visibility bounds by 46 blocks in each direction. Step 32 remains a coverage parent. Preparation starts the protected FINAL closure when the connected parent frontier reaches the near band. Startup releases gameplay only after that closure, the connected 96-chunk parent prefix, and the exact entry gates are ready. Ordinary perceptual refinement does not wait for full configured parent coverage, but its nearest visible debt and unfinished exact publication prevent ordinary outer submission and publication until near quality catches up.

A same-key authority promotion retains the PREVIEW allocation as a real source until the FINAL terrain and shadow target completes its bounded transition. PREVIEW reconstructs the seeded Base latent's low-frequency terrain without a decoder residual, while FINAL adds that latent's decoded residual through the same cleanup path. Coarse conditioning is never drawable terrain. Residual refinement may still change local shorelines or water topology, so terrain, standing water, and falls change to one FINAL allocation together at the fog-covered midpoint, with exactly one connected-water owner in every frame. A topology difference neither waits for exact ownership nor latches a generation failure. Exact ownership remains gated until stable FINAL authority. This prevents water from pairing with the other authority's terrain and allows distant tiles to converge without exact coverage.

PREVIEW parent preparation retains the complete topology-page closure needed for horizon stability. Protected FINAL targets instead enumerate the smaller set of immutable terrain pages reached by geometry support, sort directly intersected native hydrology owners lexicographically, and group adjacent owners in sets of at most two by two. Every combined half-open FINAL rectangle remains within the 1,048,576-sample query bound. Cropping a grouped result back to one 517 by 517 owner is exactly equivalent to preparing that owner independently, including negative coordinates and shared aprons. The transient LRU can retain up to the coordinator's bounded request count, subject to the decoded-authority byte budget, so a parent crossing both axes of a native-page seam keeps all four direct inputs through hydrology construction. Successful scalar, grid, point, and topology queries record their prepared direct owners. Dynamic neighboring owners discovered by depression reconciliation advance an observable authority completion generation. The base job remains parked under its original epoch, priority, and cancellation membership until that generation advances, so a cold model flight cannot become a render-driven resubmission loop. `NativeHydrologyCacheMetrics::deferredBuilds` counts completed build attempts that returned typed `DEFERRED`; it is not the parked count, active-build gauge, or failure count. If reconciliation becomes observable after the last learned completion, the scheduler retries one parked FINAL parent only after learned inference, publication, hydrology, queued base work, and active mesh work are all quiescent. This liveness probe is bounded by the spill-summary page limit. Exhaustion latches repair state instead of leaving entry blocked indefinitely.

Exact opaque terrain draws first, and a small positive depth bias keeps overlapping far tops behind resident exact surfaces while retaining a lit fallback for cold exact meshes. The exact coverage snapshot becomes separate 16 by 16 surface and flora readiness masks for every overlapping far tile. Terrain, water, and falls acquire a column bit only when every narrow surface section has a current mesh and the displayed far surface is stable FINAL authority. Tree and ground-flora fragments use the flora mask, which remains far-owned until every conservative exact vegetation section is revision-ready. An active PREVIEW-to-FINAL exchange keeps the entire destination column far-owned until the exchange retires. Each draw carries both center masks and all eight neighboring pairs, extending the ownership tests across tile faces for vegetation and waterfalls. Any fragment that remains far-owned in the exact overlap is protected, even when its tile is fully ready for a partial set of boundary requirements. Step 32 is not an acceptable visible fallback there. The camera exploration band must have step 2, and the rest of the exact overlap must have step 8 or finer. The protected handoff closure publishes only after all 60 step-1 through step-16 targets and their parents are FINAL and every internal canonical boundary matches. Any horizon patch touching an exact-owned column is excluded from the occluder set because fragment masking makes it nonconservative. Coarse parent geometry uses conservative footprint bounds beneath exact surfaces. The nearest unready distance remains available for conservative parent selection, fog behavior, and diagnostics, but it does not clip a complete radial ring.

Globally aligned tile borders use a canonical two-block boundary lattice and shared transition strips. Both adjacent tiers sample identical boundary positions and heights, so production emits no downward skirt or tile-face wall. Missing exact halos use separate explicit closure geometry: lit planned surface continuations aboveground, dark inward caps underground, and bedrock caps vertically. Direct exact-to-far tests compare the finest far tier with exact emitted surface heights at shared samples, while captures validate terrain and shoreline ownership across the overlap. Ordinary terrain replacements swap one complete topology beneath a narrow terrain-only fog pulse. Preview parents may refine through adjacent preview tiers while final authority is cold, and a same-key final payload replaces preview atomically without a downgrade. Water retains one owner until a replacement completes.

The mesher greedily combines equal flat terrain cells. Far standing water carries water-body identity and kind. A coarse cell that observes distinct authorities refines against the canonical contour instead of joining their levels. Contour-clipped shoreline triangles and top-only planar geometry prevent a partially wet cell from becoming a rectangular water ledge. Water waves remain an analytic fragment-shading effect and never displace the voxel surface. A separate `OutletFall` is one narrow receiver-centered prism with four side quads and one top quad. Its anchor's half-open tile owns all five quads, even across a tile face, and the prism overlaps the lower body's top source voxel before reaching the upper lip without raising the receiving water. Tile coordinates and sampling remain 64-bit on the CPU, while a per-draw origin restores world space from tile-local half-precision vertices.

The feature layer evaluates tree cover and species against continuous biome suitability, temperature, precipitation, soil moisture and fertility, light, slope, elevation, lithology, tectonic stress, hydrology, and ecotopes. Its accepted world-space anchors reconstruct grounded oak, large oak, birch, spruce, acacia, jungle, mangrove, palm, willow, alpine scrub, and fallen-log forms across cube boundaries. Ordinary trees reject submerged roots. Mangroves and willows alone accept suitable shallow water, and their trunks or roots extend to the sampled solid floor. Dense forest climates increase accepted canopy cover without changing the deterministic local-priority rule.

The far vegetation builder uses globally anchored aggregate forest cells and eight-block ground-flora cells. Climate suitability, substrate, slope, water, species, exact plant selection, and cell-addressed priorities decide compact grounded trunk-and-crown clusters and crossed plant clumps. LOD tiers retain nested anchor subsets. Coarse plant billboards widen with repeated texture coordinates to preserve projected cover. Half-open cell and tile ownership prevents duplicates across boundaries, and the separate flora-ready column masks clip each fragment only after its exact destination column is ready. Bit 28 of `faceAttr` marks far vegetation geometry for the shared vertex contract and diagnostics. Flora attachments have independent jobs, completion storage, cache entries, GPU allocations, and cancellation. A blocked, empty, or failed attachment cannot delay or replace its terrain, standing-water, and fall base mesh.

Exact-to-far ownership and exact-face closure are separate mechanisms. Per-column masks decide whether exact or far fragments draw. The renderer publishes exact readiness before drawing and suppresses required exact sections until the complete destination column is ready when a drawable parent covers it. When an exact mesh lacks a current halo, its boundary scan emits the appropriate lit, dark, or bedrock closure cap until the halo arrives and invalidates that mesh. One immutable coverage snapshot governs exact and far selection for the complete frame.

The design is an adaptive tiled terrain LOD informed by geometry clipmaps and CDLOD. It is not a literal geometry clipmap. Visibility first uses a conservative tile AABB frustum test, then processes surviving tiles front to back through a conservative 256-bin terrain-horizon culler. Sixteen 64 by 64-block patches per visible tile contribute lower-bound horizons without per-frame heap allocation. The horizon test rejects a tile only when fully covered angular bins establish a higher lower-bound horizon. This CPU culler is not a hierarchical Z buffer and does not use a depth pyramid. The renderer separately builds a GPU min-linear-depth Hi-Z pyramid after opaque resolve for screen-space indirect lighting; it does not participate in far-terrain selection. Exact opaque and far terrain use counterclockwise front faces with back-face culling. Cross and flat flora emit both windings. Shadow casters and water remain cull-none for their separate correctness requirements.

Far terrain is encoded through bounded direct indexed draws. The implementation does not build Metal indirect command buffers. Exact opaque geometry draws first and shares depth with overlapping far geometry. Resident step-32 tops provide depth-backed cold-residency fallback only outside protected exact-loading tiles;
protected tiles require step 2 or step 8. Water samples resolved opaque depth for refraction and absorption while hardware-testing and writing the nearest visible interface through the media depth attachment, so water and canopies use the same current per-column ownership masks as opaque far terrain. Far water joins the same three-dimensional back-to-front water list as exact water.

CPU far-tile retention is capped at 24,576 entries and 3 GiB. The entry cap can retain the conservative all-tier wanted set for a screen-error-refined radius-512 view, while the byte cap remains the effective memory bound. Active step-32 parents and camera-critical refinements are pinned against distant work under pressure. A requested protected FINAL refinement is the exceptional critical class: it may reclaim optional distant non-displayed CPU refinements or canopies, and its GPU admission may use the complete arena after the same optional distant reclamation. Active coverage, displayed surfaces, both transition endpoints, exact fallbacks, active protected lineage, and the requested critical keys remain pinned. Alternate LODs at the same protected coordinate do not inherit that exception. Distant work cannot evict a camera-critical cache entry, queued job, upload, or GPU allocation. A residency change cancels the bounded job and completion queues immediately, then utility workers rebuild priority state and retire obsolete cache records. One maintenance pass scans at most 64 records and retires at most 32 MiB, except that one oversized record may retire alone. Mesh and membership destruction occurs outside cache locks and never on the render thread. The far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertex storage and 1 GiB of index storage. These are independent of exact mesh residency. All renderer, world, transient, and Metal allocations together must remain below the 64 GiB unified-memory acceptance ceiling.

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

Opaque terrain and gameplay geometry share one material contract. Surface RGB stores diffuse albedo and alpha stores baked ambient accessibility. A separate reactive channel marks moving gameplay geometry, and a memoryless R32 color attachment preserves the full device-depth selection key. Entity, item-entity, and boat pipelines declare the same multisample formats as terrain. The 4x tile resolve averages HDR coverage but selects categorical surface and reactive data from the covered sample with minimum device depth. Clouds and weather particles use a separate single-sample contract, test against resolved depth, and do not write the opaque auxiliary attachments.

The block texture array owns a filtered five-level `R8Unorm` emission-mask array beside its color array. Lava is fully emissive. Only painted torch flames and the fixed mouth of an active furnace emit; the torch stick, inactive furnace, chest, and bed remain nonemissive. Emissive HDR radiance contributes to bloom, propagated block light, and near-field SSGI. A bed is a nonopaque 9/16-height box with matching collision, authored face culling, smooth packed light, baked corner accessibility, shadow participation, and indirect-light reception. Block raycasts and selection outlines use the same authored bed and torch bounds.

Indirect history is keyed by drawable size, camera state, world and session identity, quality, direct-light source, lighting-edit revision, time-discontinuity revision, and prior-depth validity. Resizes, teleports, material camera changes, world switches, sun-to-moon transitions, lighting edits, clock discontinuities, and invalid prior depth reset history. Ordinary motion reprojects through the previous view-projection matrix, while reactive history rejects moving entities and newly uncovered pixels. Preview-to-final terrain refinement, canopy attachment, parent replacement, and exact-to-far handoff preserve history because their ownership transitions retain drawable coverage.

Dynamic entities cache one packed sky and block-light probe during the fixed tick. Render frames consume that smooth packed light without querying `World`, and dedicated depth-only draws add dynamic objects to each intersecting shadow cascade. The v4 preparation path issues no scene pass and advances no temporal history. During gameplay, exact ownership, learned-authority provenance, connected-parent selection, 2:1 transition topology, and the zero-skirt contract decide published geometry before screen-space lighting runs.

## Derived voxel light

Skylight and block light share one derived byte per materialized voxel. The high nibble stores 4-bit skylight and the low nibble stores 4-bit block light. Level-15 sky seeds exist only where complete full-height column authority proves an unobstructed path to the sky. `LightEngine::floodChunk` then propagates both channels through transparent cells, loses one level at each non-seeded step, and reports both changed state and a changed-face mask. New cube publication consumes those face masks synchronously under the world lock so the first visible mesh is settled. The bounded tick-thread reconciliation queue remains for unloads and deferred fluid changes.

A resident gameplay edit that changes emission or opaque light transport increments `World::lightingRevision`, allowing temporal indirect lighting to reject history immediately without treating ordinary streaming publication as an edit.

Missing chunks and incomplete vertical paths remain conservatively dark until their authority arrives. Generated surface cutoffs remain the source for direct-sky seeding and provisional missing-boundary classification, but they are not a binary final light value. The propagation is monotone over fixed blocks and converges to one fixed point independent of load order. Packed light is recomputed after generation, load, unload, or edits and is never serialized. A changed field increments the cube revision and dirties its mesh.

Lighting composition keeps independent physical terms. Cascaded visibility controls only direct sun or moon radiance. Propagated skylight and baked corner accessibility control ambient irradiance. Block light and emissive radiance are never darkened by skylight, baked accessibility, GTAO, or SSGI. Outside valid shadow coverage, direct light fades toward propagated exterior visibility instead of becoming unconditionally bright.

## Runtime water boundary

World generation writes complete standing and falling water blocks directly. The v4 canonical column plan supplies ocean, river, tiled spill-reconciled lake, connected wetland, delta, estuary, brackish, rapid, channel-waterfall, outlet-fall, and naturally routed volcanic-crater topology rather than reconstructing only a surface sheet. A component that exceeds a bounded closure fails closed; the context-free v3 diagnostic path retains its legacy delta, aquifer, and analytical crater-lake systems. Every generated standing-water column is a full-height source-water volume: each wet voxel from the first one above solid support through the top water voxel is an implicit source, including across cube boundaries. These generated source voxels need no explicit fluid array. An implicit source fills its complete voxel and places its visible top one block above the voxel floor. Routed rapids and outlet approaches materialize explicit flowing levels 1 through 7 at their exposed stages, while covered water and receiving pools remain sources. Waterfall curtains and outlet falls carry explicit falling state. Generation never asks the fluid scheduler to settle terrain, and loading an ordinary generated cube also schedules no water. Stable generated and runtime water emits planar top geometry only. Analytic fragment normals provide distant ripple detail without changing geometry or ownership. Vertical water sides are reserved for cells explicitly marked falling, including the short receiver-centered outlet overlay, so lake, river, and ocean boundaries cannot render as unsupported walls. The far sampled representation places generated source water on the same full-block plane as the exact implicit source voxel.

A gameplay edit activates one cell and its six neighbors. The scheduler applies downward-first source and level rules to loaded cells only. If active flow reaches a missing cube, it persists a frontier indexed by that destination cube. A later load makes only the matching index bucket eligible, and the fixed tick resumes a bounded number rather than scanning all 65,536 possible frontiers. Runtime fluid writes bypass the player activation entry point to avoid recursive scheduling. Pending-update and frontier-overflow counters remain visible in diagnostics.

This division keeps generation order-independent while localized player disturbance can evolve over time.

## Persistence boundary

Generator version four stores RYCH v4 edited cubic sections beneath `regions-v4` using 64-bit X and Z plus section Y. Its packed header includes an IEEE CRC-32 of the uncompressed block and optional fluid payload. A per-column manifest records edited section Y values and deferred fluid frontiers. Bulk visible-column reads copy only required manifest data under one short lock. Manifest disk I/O uses a separate serialized write lock, and the bounded save queue coalesces repeated snapshots for a cubic position to the newest revision. Metadata preserves the generation fingerprint and seed, fixed safe spawn, current player and respawn state, inventory, game mode, generation toggles, timestamps, and world time. Regional weather, lunar phase, lightning buckets, and packed voxel light reconstruct from identity, coordinates, learned climate, and saved time, so they require no additional save payload.

Qualified v4 profiles live below Application Support and never share authority or edit directories across fingerprints. Metadata persists the safe world-start spawn separately from the current respawn anchor and records whether that anchor came from a bed. Player state includes the 36 inventory stacks plus the cursor and all nine crafting-input stacks. Furnaces and chests are the stateful blocks. Both persist in a per-world plaintext sidecar named `block_entities.dat`. Its `RYBE 1` format stores one `furnace` or `chest` record per full-width block position, uses atomic temporary-file replacement, tolerates unknown record types, and loads once at world start. Dropped item entities and boats are never persisted; they despawn on quit and on a world switch.

Legacy cube files and manifests remain isolated beneath their original paths. They are not deleted, converted, or loaded into generator-v4 terrain. Detailed layout and paths are in [world-generation.md](world-generation.md).

`GraphicsSettings` are preferences, not world metadata, and follow the separate settings path described above.

## Error policy

1. Metal device, queue, and pipeline failures are fatal because the game cannot render without them.
2. A v4 authority, cube, or far-base failure latches generation, preserves existing resident geometry, closes missing collision, and exposes retry or repair instead of publishing an empty result. Typed authority failures retain their original code. A standard or unknown exception from a critical far-base or final mesh becomes a retriable generation failure while any resident preview parent remains drawable.
3. Missing files return `std::nullopt`. Corrupt or incompatible cube data reports that cube's failure once, returns no cube, and regenerates deterministically.
4. Audio initialization may fail without preventing play.

There is no project-wide `Result<T, E>` wrapper. These boundaries use the smallest established mechanism for their failure mode.

Master audio volume follows the settings slider on every screen so interface clicks are audible in menus (`-playUiSfx:gain:` bypasses the playing-screen gate that world one-shots still respect). The paused-world feel is kept instead by stopping the looping wind bed off the playing screen; a frozen tick produces no world sounds.

## GPU boundary

Every structure shared by C++ and Metal lives in `include/render/shader_types.hpp` with size and offset assertions, including screen-space lighting, atmosphere, cloud, weather, lightning, and shadow records. Exact and far terrain preserve the 16-byte vertex format. Fluid direction occupies bits 24 through 26, falling state occupies bit 27, and far-canopy impostors use bit 28. Bit 29 is legacy compatibility metadata and may not create production skirts. Water exterior-sky authority uses bit 30, and bit 31 remains reserved unless its active transition use is documented without changing geometry ownership.

See [rendering-conventions.md](rendering-conventions.md) for coordinate, pass, halo, culling, and water rules.

## Tests and diagnostics

One hermetic Catch2 executable is built from the source modules listed in `tests/meson.build`, covering common utilities, learned and procedural world generation, canonical hydrology, weather, rendering, entities, persistence, concurrency, engine, audio, lightning, and thunder. Cubic tests cover negative coordinates, the 96-section vertical range, full-width counter addressing, generation order, mesh halos, smooth packed skylight and block-light convergence, bounded fluid frontiers, v4 persistence, and entity behavior. Weather tests cover deterministic spatial and temporal continuity, presets, climate bias, physical wind units, precipitation type, cloud profiles, latest-wins publication, lightning IDs, and thunder delay. Render tests cover five-cascade selection and refresh, shared layouts, temporal-history resets, atmosphere finiteness, cloud-noise tiling, Beer-Lambert extinction, and froxel slicing. Generator-v4 tests cover identity and fingerprint rejection, model-pipeline goldens, page CRC and repair, native hydrology, water continuity, exact and far equality, base-before-canopy publication, and zero production skirts.

Far-terrain tests cover full-disk parent selection, dynamic local 8, 12, and 16-worker admission, the ordinary-submission-and-publication pause during exact or local desired-LOD debt, distance-first nearby ordering before projected-error ranking, the nominal four-worker base reservation, the camera-critical bypass of that reservation, the one-worker gameplay canopy guarantee, exact 32-chunk ownership, step-2, step-4, step-8, and step-16 bands, coverage-only step-32 parents, near-job displacement of queued outer parents, urgent protected-parent displacement, current-FINAL-before-bridge scheduling, parent and drawable frontiers, adjacent-tier refinement, preview-to-final replacement, exact and far atomic handoff, critical protected CPU and GPU reclamation, grouped protected-authority exact equivalence, corrected native-hydrology deferred-build metrics, 16 by 16 per-column ownership, 3 by 3 neighbor-mask crossings, non-occluding partial masks, outward-only hysteresis, shared no-skirt transition topology, body-aware contour-clipped shorelines, deterministic tile hashes, cache and queue bounds, epoch cancellation, and conservative horizon behavior. They assert that cold entry requires the connected PREVIEW terrain-and-water prefix and the 60-target protected FINAL closure, movement retains the old closure until its complete replacement is ready, terrain and water publish while canopy construction is blocked, and every production mesh has zero skirt quads. Streaming and meshing tests cover lit aboveground closure caps, dark underground closure caps, vertical bedrock caps, halo invalidation, full 32-chunk required-surface priority, the seven-task exact-generation submission limit, stale-task skipping and relevant-work requeue, epoch-matched exact collision publication with canonical generated proxies, bounded first-publication lighting, aborted ray traversal, and skylight occlusion across unloaded vertical gaps. Focused render tests cover indirect uniform layout and bounded math, every history-reset reason, opaque versus resolved MRT contracts, emission-mask ownership, and nonemissive bed culling and corner accessibility under `[render][indirect]`, `[render][indirect][mrt][particles]`, `[render][textures][emissive]`, and `[render][mesher][bed]`. Field regressions scan learned window boundaries, water identities, shoreline support, page continuity, lithology contacts, material patches, and deformed strata. Headless tests do not create a Metal device, so actual culling, frame rate, memory, indirect lighting, and image quality require the playtest workflow.

Developer runs can fix the seed and spawn with `RYCRAFT_WORLD_SEED` and `RYCRAFT_SPAWN`. `RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, `biome`, `lod`, or `authority`. The LOD overlay distinguishes exact and step-1 through step-32 ownership, while the authority overlay distinguishes FINAL from PREVIEW surfaces during promotion. `RYCRAFT_WEATHER` accepts `clear`, `overcast`, `rain`, `storm`, or `snow`. The inspector executable provides repeatable feature locations, footprint samples, former-grid artifact measurements, water IDs, shoreline distance, lithology, material palettes, timing, hashes, and separate cache metrics. Its `--authority-delta` mode compares one isolated PREVIEW and FINAL page, including physical-height error, ocean-sign disagreement, boundary error, cold timing, and per-model call deltas. F3 displays exact required and ready sections, unresolved columns, the conservative nearest-gap distance, parent and refinement wanted, resident, drawn, and queued counts, the drawable coverage frontier, cascades and refreshes, indirect-history validity, local weather and cloud type, storm or lightning ID, caches, arenas, fluids, coalescing, and dropped work. Visual changes must still run with Metal validation and be inspected in captured frames.
Authority-delta JSON records both page hashes and separates PREVIEW and FINAL cold timing and per-model call deltas.

See [world-generation.md](world-generation.md), [performance-conventions.md](performance-conventions.md), and [rendering-conventions.md](rendering-conventions.md) for the corresponding contracts and commands.
