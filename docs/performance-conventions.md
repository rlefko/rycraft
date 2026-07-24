# Performance Conventions

Performance is part of generator correctness. Missing terrain, skipped water, synthetic walls, and crack-hiding skirts are not acceptable optimizations.

## Reference target

Generator v4 acceptance uses an identified Apple M4 Max, macOS 14 or newer, native drawable resolution, 4x MSAA, and view distance 512.

| Gate | Required result |
|---|---:|
| First entry | Safe final spawn, revision-ready exact spawn meshes, a connected coarse terrain-and-water parent frontier through 96 chunks, and the atomic 60-target protected FINAL closure within 30 seconds |
| Cold settlement | Configured 512-chunk parent coverage and all remaining authority, hydrology, exact, far, upload, and canopy queues within 5 minutes |
| Frame rate | Lowest sustained one-second rate at least 60 FPS |
| Frame time | p95 at most 16.67 ms and no sustained generation interval above 20 ms |
| Render and tick authority work | p95 below 0.25 ms and maximum below 1 ms |
| Movement recovery | One-chunk move recovers within 5 seconds without losing parent or water connectivity |
| Tensor-window cache | At most 384 MiB |
| Decoded authority cache | At most 512 MiB |
| Model, runtime, and scratch growth | At most 4 GiB |
| Total unified memory | At most 64 GiB |

These are merge gates, not measurements established by ordinary CI. Validation overhead does not belong in the frame-time run. Run Metal validation separately.

## Evidence standard

Every performance report records:

- Commit, build type, machine configuration, macOS version, drawable size, and settings
- Unsigned 64-bit seed, generation fingerprint, spawn, yaw, pitch, and route
- Cold and warm authority, hydrology, exact, far, upload, and canopy queue maxima
- First-entry and queue-settlement times
- Frame p50 and p95 and lowest sustained one-second frame rate
- Learned scheduling and cache-lookup p50, p95, and maximum on render and fixed-tick threads
- Process RSS and Metal allocated or resident memory as separate counters
- Highest credible unified-memory total without adding overlapping counters
- Core ML partition, CPU fallback, inference queue, and cache metrics
- Indirect prepare, trace, temporal, a-trous, and apply GPU times, persistent texture payload, quality, and history-reset counts by reason

Do not infer a passing cold horizon from queue caps. A canonical digest or one authority-page timing does not qualify the complete first-entry route.

Attribution uses one machine-readable critical-path trace (`common/trace.hpp`, enabled with `RYCRAFT_TRACE`) and the checked-in `rycraft_trace_summary` tool for p50, p95, maximum, queue depth, cache reuse, cancellation, duplicate model windows, critical-path ownership, and omitted water or flora products. Disabled tracing performs no per-frame allocation and its overhead is a single relaxed atomic load per emit site. The trace is observability only and may never change a measured number.

## Startup and inference

Download, initial SHA-256 verification, archive extraction, Core ML compilation, and canonical qualification run on the bootstrap thread before world construction. They do not run in a frame or fixed tick. Later launches check the installed pack's verified completion marker and current file stamps before reuse, so a normal launch does not reread the 2.3 GB pack or make a network request. A missing, legacy, or changed marker triggers a local full SHA-256 audit. Retry never redownloads a verified pack, and Repair is the only path that replaces an installed asset.

Production graph execution is sequential and protected by one inference mutex. ONNX Runtime does not execute independent graphs concurrently, and the inter-op setting remains one. CPU fallback sets an explicit intra-op total of `min(hw.physicalcpu, 16)`, including the calling thread. Coarse retains its qualified scalar batch. Base binds the symbolic batch dimension to the pipeline's fixed batch of four. Decoder binds batch four and the 256 by 256 spatial dimensions, with deterministic last-window repetition and padded-output removal for a short tail. All three sessions remain resident, with three as the hard maximum, under the fingerprinted `coreml-cache-v3-base4-decoder4x256` configuration and qualification digest `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df`. Metrics expose queued calls, active calls, failures, maximum call concurrency, compiled and resident sessions, provider partitions, and CPU fallback threads. Maximum active model calls must remain one.

Preparation evidence attributes every graph call and its model split to qualification, dry-spawn
coarse search, final spawn certification, exploration exact generation, horizon preview, protected
FINAL handoff, visible FINAL refinement, or other work. The attribution is diagnostic and
fingerprint-neutral. A warm open with finalized safe-spawn metadata and persisted protected
transient authority performs only the three qualification calls before the horizon is available.
It does not repeat dry-land search or reconstruct the same exact hydrology rectangles.

This distinction follows the official [ONNX Runtime thread-management contract](https://onnxruntime.ai/docs/performance/tune-performance/threading.html): execution mode governs graph-node parallelism, while the intra-op pool governs parallelism inside an operator.

A qualifying first-entry measurement covers the safe final spawn, revision-ready exact spawn meshes, connected canonical terrain-and-water parents through the 96-chunk entry radius, and the atomic protected FINAL closure. That camera-aware closure contains 4 targets at step 1, 8 at step 2, 12 at step 4, 16 at step 8, and 20 at step 16. It begins during preparation when the connected parent frontier reaches the near band. The configured 512-chunk horizon stays selected after entry, but unfinished exact publication through 32 chunks or connected visible desired-LOD debt pauses ordinary outer submission and publication. Near jobs run nearest-first and may displace queued or dependency-parked outer parents. Within the nearby visible class, distance ranks before projected screen error. Outer submission and publication resume only after both debts clear and may not expose a farther island. The same cold run measures complete configured parent coverage and every remaining queue against the five-minute settlement gate. A partial-radius inspector run may help attribute cost, but it does not establish either gate, frame rate, or memory.

Normal dry-spawn ranking retains at most one proposal per aligned 2,048-block hydrology owner. The
canonical screen checks the requested chunk before a bounded globally aligned four-block native
raster scan across at most 16 workers. A proven center first attempts the complete cold footprint,
then the exact 113 by 113 radius-zero safety footprint formed by the five-by-five `ColumnPlan`
dependency apron and its 49 by 49 hydrology rasters. The screen may try at most 64 deterministic
relocations whose coarse proof lies inside the prepared hydrology owner and whose exact footprint
stays inside the proposal's already materialized learned page. It therefore adds no model page to
the successful seed-42 cold path. Canonical water in either wider footprint rejects only this fast
path. The semantic fallback remains the original 25-sample five-by-five dry certificate. A
positive-elevation continental fallback installs no certificate and may start only radius-zero
exact validation. It cannot admit far-horizon work or metadata publication until that exact plan
proves canonical water absence, support, headroom, slope, and the nearby dry neighborhood.
Neighboring semantic hydrology owners and the wider exact band are post-construction streaming work.
Prequeueing their complete authority closure is reserved for explicit repair and qualification, not
ordinary entry. Entry separately requires the already-retained three-by-three horizontal by
three-section collision halo around the finalized spawn. Measure this residency check concurrently
with the horizon; it must not widen the nominal exact radius or add learned-model calls.

`CachedTerrainAuthority` has these hard bounds:

| Resource | Bound |
|---|---:|
| Outstanding page requests | 64 |
| Speculative movement pages | 8 |
| Concurrent page builds | 1 |
| Pages per bounded query | 64 |
| Samples per query | 1,048,576 |
| Decoded cache entries | 1,024 |
| Decoded cache bytes | 512 MiB |
| Tensor-window cache | 384 MiB |

Equal cold requests share one flight. Page construction and disk I/O do not occur under the cache mutex. Render and fixed-tick threads may enqueue or read completed authority. They may not execute inference, block on a flight, verify model files, decompress authority pages, or construct cold pages. SPAWN and EXPLORATION_EXACT admission may displace unstarted lower-priority coordinator work at the hard request cap. Decoded terrain and native-hydrology caches reject a weaker insertion before evicting a stronger entry. The shared native-hydrology build gate gives its next free CPU and scratch slot to the strongest waiter, including a flight promoted by a duplicate exact request. Running model calls, page solves, and atomic publications are not interrupted. The renderer updates movement prediction after one chunk of travel and submits its bounded page row only after visible preview authority is ready. Protected exact-handoff parents and all visible work remain ahead of prediction in the shared coordinator.

## Exact cubic active set

Exact simulation remains bounded independently of the radius-512 visible horizon.

- After entry, nominal exact radius is `min(viewDistance, 32)`.
- During cold entry, the nominal exact radius is zero. The mandatory mesh halo keeps the camera
  column and its four cardinal neighbors active, and plan dependencies extend four chunks from the
  center without authorizing a broader visible exact band.
- Loaded exact cubes are capped at 32,768.
- Exact mesh candidates and residency are capped at 16,384.
- Six generation workers submit at most seven cube tasks at once.
- Four exact mesh workers retain bounded queue and upload budgets.
- Every required surface section through the full 32-chunk exact disk ranks ahead of optional flora
  and broad primary work in generation, mesh admission, and upload. The camera column ranks first,
  the six-chunk exploration band second, and the remaining required disk third.
- Exact collision owns only renderer-published sections whose coverage epoch matches the active
  `ExactSurfaceCoverageSnapshot`. Other planned sections use canonical generated terrain and fluid
  proxies, unresolved columns remain closed, and raycasts do not force-load cubes.

The vertical world has 96 sections. An active-set scan that multiplies every horizontal column by all 96 sections is a violation. Use planned exposed sections, saved edits, the camera band, and a bounded halo. `VerticalSectionMask` must inspect both words for full sky-range queries.

The 193-level density lattice is lazy. A requested cube may evaluate its own vertical interval and required surface neighborhood, not all levels in every density column.

## Far horizon

Far terrain uses 256 by 256-block tiles. Settled geometry follows fixed distance bands:

| Chunk distance | Required settled tier |
|---:|---:|
| 0 through 32 | Exact cubes |
| 32 through 64 | Step 2 |
| 64 through 128 | Step 4 |
| 128 through 256 | Step 8 |
| 256 through 512 | Step 16 |

Step 32 is a coverage tier only. Exact cubes own the first 32 chunks, followed by maximum-coarseness limits of step 2 through 64 chunks, step 4 through 128, step 8 through 256, and step 16 through 512. During gameplay, conservative projected geometric error may retain a finer tier based on distance, viewport height, vertical FOV, and parent relief. Refinement begins above 0.55 pixels and reaches FINAL step 1 when step 2 remains perceptible. Step 1 is the physical voxel-grid floor. Outward coarsening validates one adjacent tier at a time below 0.45 pixels. Moving inward adopts the finer required result immediately. Cold entry completes the connected step-32 terrain-and-water prefix through 96 chunks and a camera-aware protected FINAL closure. Its position-aware two-by-two step-1 core and Manhattan rings contain 4, 8, 12, 16, and 20 targets at steps 1, 2, 4, 8, and 16, for 60 total. The protected lane begins during preparation as soon as the connected parent frontier reaches the near band.

- Every selected coordinate requests a step-32 parent.
- A resident parent stays available until a connected replacement is resident.
- Cold entry emits only the protected FINAL refinement closure after the connected frontier reaches the near band. Ordinary perceptual refinement and flora work remain disabled until gameplay.
- Step-2 residency begins at the exact 32-chunk boundary and remains warm while exact ownership advances.
- Once the connected 96-chunk prefix is drawable, exact publication through 32 chunks and every
  connected visible desired-LOD miss pause ordinary outer-parent submission and publication. Near
  work is nearest-first, ranks distance before projected error within the nearby visible class, and
  may displace queued or dependency-parked outer parents. Displayed parents, the connected prefix,
  transitions, exact fallbacks, and protected lineage remain pinned.
- Sixteen terrain workers cap pending jobs at 64 and completed results at 32. Admission is dynamic:
  local far work uses 8 workers alongside exact generation or meshing debt, 12 after exact debt
  clears, and all 16 only after both exact and local debt clear. Exact debt without a local far miss
  admits no ordinary far work.
- Four terrain workers form the nominal reservation for missing base coverage within each admitted
  budget. Ordinary urgent refinement observes that reservation.
- An urgent camera-critical refinement is selected before an unrelated distant parent even while
  the nominal reservation is active. The fixed total budget and twelve-job urgent cap still apply,
  and the base reservation resumes after the critical work drains.
- The lower-priority flora service remains separate from terrain work. Preparation and an
  incomplete connected 96-chunk prefix use zero workers. Gameplay guarantees exactly one
  low-priority flora worker after that prefix is drawable, including while exact or local terrain
  debt continues. No second gameplay flora lane opens. Fully exact-owned tiles do not consume the
  bounded canopy request batch, and missing PREVIEW attachments stay ahead of FINAL ecology
  promotion.
- The CPU far-mesh and canopy caches are capped at 24,576 entries each, with independent byte caps of 3 GiB and 512 MiB.
- The far GPU arena is capped at 2 GiB of vertices and 1 GiB of indices.
- Frustum culling precedes conservative terrain-horizon culling.

Surface-stage work includes terrain, standing water, and falls. It publishes before vegetation enrichment. Flora jobs have independent completion storage, caches, uploads, and GPU allocations. A missing attachment uses PREVIEW ecology first, grounded against the displayed PREVIEW or FINAL surface. All PREVIEW publications sort nearest-first and form a phase barrier ahead of FINAL ecology promotions, including active and dependency-parked work. A successful provisional publication retains the same bounded logical slot for its lower-priority FINAL replacement. A blocked, canceled, failed, or empty attachment cannot prevent a drawable base mesh. Camera-order revisions reprioritize queued, parked, and follow-up flora work, and a new nearby request may displace the least-important queued or parked request at the bounded pending cap.

The far GPU arena reserves 64 MiB of vertices and 32 MiB of indices for parent coverage. Below that, another 64 MiB of vertices and 32 MiB of indices remains available to visible flora but not broad refinements. Coverage retains authority to consume the complete arena when gap-free publication requires it. A ready nearby flora attachment receives one upload opportunity after protected and urgent local terrain and before distant base or broad refinement uploads. A camera-near terrain upload may reclaim farthest-first optional non-displayed refinement and flora allocations before admission fails. Coverage parents, displayed surfaces, active transition endpoints, and protected replacements remain pinned through that reclamation.

Camera-near terrain is capacity protected across the complete pipeline. A missing PREVIEW step-32 parent in the current protected closure may displace the worst queued or dependency-parked ordinary parent at the pending cap. A newly requested camera-near bridge may similarly displace noncritical refinement work, bypass the nominal distant-parent worker reservation, and reclaim optional CPU-cache or GPU residency. Current protected FINAL parents and children receive capacity before their bounded PREVIEW bridge prerequisites; unused bridge capacity returns to current FINAL, and directional prediction receives only the remainder. Prediction is CPU-only until the canonical anchor changes. A requested protected FINAL refinement may reclaim optional distant non-displayed CPU or GPU residency and use the complete GPU arena. Coverage, displayed surfaces, transition endpoints, exact fallbacks, active protected lineage, and the requested critical keys remain pinned. Distant work cannot evict a camera-critical job, CPU cache entry, pending upload, or GPU allocation. Moving the camera refreshes that classification and prevents the old center from keeping a permanent reservation. These rules preserve a valid drawable parent throughout replacement and do not increase the fixed queue, worker, cache, or arena bounds.

Exact cubic streaming applies the same rule before far ownership retires. Broad and medium plan or cube work leave at least one physical exact worker available to a newer camera epoch. All required surface sections through 32 chunks use the protected terrain lane; optional exact flora follows them, and broad primary work follows flora. At the 64-result mesh cap, a better camera, exploration, or required-surface request may transfer the worst queued slot without canceling running or completed work. A changed immutable candidate snapshot cancels queued meshes outside the new active set and clears only the matching requested revision. Deferred first-publication lighting is reranked in the same camera, exploration, required-surface, flora, and broad order whenever the active center moves. Each synchronous publication transaction is capped at 32 floods. Remaining work stays pending in the bounded queue, and no exact mesh may publish until its transaction settles.

Exact terrain and exact flora use separate readiness sets. Terrain and water hand off after the narrow surface set is revision-ready. Far flora remains visible until every conservative exact tree and ground-flora section is ready. Required terrain through 32 chunks always ranks ahead of flora. Within 16 chunks, flora sections then rank ahead of distant broad primary surfaces once local terrain debt clears. Flora-only exact sections publish visual and collision ownership together only when their complete flora column is ready. Farther exact flora remains broad because its drawable far attachment continues to own the vegetation.

Step-32 horizon parents establish PREVIEW coverage. During gameplay, steps 16, 8, 4, and 2 may also use the same Base-lineage PREVIEW authority as temporary geometric proxies, which removes large projected cells without pretending that the omitted Decoder residual is canonical detail. Step 1 always uses FINAL authority. Visible PREVIEW surfaces carry the measured revision-9 residual bound into their projected-error diagnostics and request same-key FINAL parents and children ahead of canopy or speculative work. Ready FINAL children, proxy bridges, and visible FINAL parents each receive four ordinary urgent slots, with unused child slots loaned to proxies. At most four ordinary visible FINAL parents remain in flight, so parked decoder work cannot strand either nearby proxy geometry or a same-key FINAL child. A base promotion retains its PREVIEW source while a visible proxy depends on it. Once that source retires, further children use FINAL authority. A FINAL child cannot display until its same-tile parent is also FINAL or its retained PREVIEW source matches the child during promotion. A same-key preview-to-final replacement retains the complete old surface until its FINAL terrain payload is ready. Matching water topology switches ownership at the transition midpoint without interpolation. A topology-changing water promotion is not accepted visual evidence until the complete hydrology owner and perimeter can publish as one unit. Preview failure may not latch the final world context.

Protected FINAL targets prepare only the terrain pages touched by their geometry callbacks. Directly
intersected native hydrology owners are sorted lexicographically and grouped in adjacent sets of at
most two by two. Each combined half-open FINAL rectangle stays within the 1,048,576-sample query
bound. Cropping a grouped result back to any 517 by 517 owner is exactly equivalent to preparing
that owner independently, including negative coordinates and shared aprons. They do not materialize
an owner's full 16-page topology closure. Spawn and protected rectangles publish fingerprinted
`RYTG` files after quantization, while visible and speculative rectangles remain memory-only. The
byte-bounded transient LRU uses the coordinator's request-count bound, which retains all four inputs
when a parent crosses two native-page seams. Successful batched hydrology sampling records each
direct owner so later parents do not repeat transient preparation after cache turnover.
Fill-Spill-Merge reconciliation may discover a bounded neighboring owner after routing starts. In
that case, the worker retains its scheduler ownership and parks on the learned authority's
observable completion generation. It resumes once for each completed dynamic dependency, never
once per frame, and neither waits on nor executes model inference.

`NativeHydrologyCacheMetrics::deferredBuilds` is a monotonic count of completed cache build attempts
that returned typed `DEFERRED` because learned authority was not ready. It is not the current number
of parked pages, the active-build gauge, or the failure count. Interval logs subtract the preceding
snapshot from `deferredBuilds`; they report `activeBuilds` directly.

While the preparation screen is active, the renderer advances exact mesh publication, authority polling, base scheduling, result drains, shared-buffer publication, and the protected FINAL closure without issuing the full exact-world scene draw. The protected lane opens when the connected step-32 frontier reaches the near band. The preparation cadence may not depend on drawing an increasingly large incomplete scene.

Step-32 water recovery uses `waterTopologyPossible` and bounded interior probes. A shortcut may reduce a route's apparent width at distance, but it may not delete the route, join unrelated stages, or infer water only from dry corners. Topology-certified partial pages remain sparse even when more than half of their samples are required. The selected mask is decomposed into deterministic rectangles for canonical grid queries, and capture diagnostics report grid calls, sampled cells, complete-page calls, and point samples.

Downward skirts are forbidden. `skirtQuadCount` remains zero in production. Shared transition strips use canonical two-block boundary samples and maintain a 2:1 displayed-neighbor ratio. A fog pulse may hide one atomic topology swap, not a missing parent or static crack.

Near step-2 refinement requests name the required target directly. The camera tile, protected
handoff, exact fallback, and connected-wavefront classes rank first. Within the nearby visible
class, smaller horizontal distance ranks before greater projected error. Screen-space error chooses
the desired tier but does not let a farther tile delay closer missing detail. Distant replacements
use adjacent bridges. Publication retains every coarse parent until the complete connected child
patch and its legal shell are resident.

## Hydrology cost

Hydrology pages are 2,048 blocks wide. At 7.5 meters per block, one page is 15.36 kilometers wide and 235.9296 square kilometers.

The water solver remains a bounded tiled query. Every change identifies domain size, apron, native spacing, fixed passes, cache key, single flight, construction concurrency, and edge reconciliation. Whole-world walks and unbounded upstream traversal are prohibited.

Preview and final native routers share one process-wide build gate. It permits at most 16 simultaneous page builds and also obeys hardware concurrency and a one-GiB aggregate scratch reservation. This is one total process budget, not 16 builds per context. Record active builds, peak concurrency, admission waits, and scratch use. The gate ranks waiters by the shared `AuthorityRequestPriority` and reserves lanes for the exact band the way the learned queue does: distant `COARSE_PREVIEW` and `SPECULATIVE_PREFETCH` builds take at most half the lanes and visible-or-lower builds at most three quarters, so a distant owner cannot occupy every hydrology lane while the player's exact band is unresolved.

No-raising reconciliation must not introduce an unbounded flood. Stage relaxation or dry-cell wetting needs a fixed rectangular extent, deterministic ordering, and scalar, reverse-order, and batched equality.

## Gameplay and lighting bounds

- Runtime fluid ticks process at most 1,024 cells, with 65,536 pending cells and 65,536 deferred frontiers. Fluids query loaded cubes only.
- Wild fauna cap at 64 living animals, with bounded activation and despawn radii.
- Dropped items cap at 128, skip collision sweeps while resting, and run their merge pass once per second.
- Boats cap at 40. Unridden boats outside the active radius freeze, while the ridden boat replaces the player's movement step.
- Furnaces are player-placed and tick on the fixed simulation path. Container catch-up remains bounded to the existing fixed-tick recovery limit.
- Block-light propagation is derived state. A gameplay edit may synchronously settle only the bounded affected neighborhood needed for a same-frame near-camera mesh; background reconciliation remains budgeted.

Emissive lava, torches, and active furnaces share one material and block-light contract. A filtered `R8Unorm` texture array limits radiance to all lava texels, torch flames, and active furnace mouths without a per-block frame scan. Beds, inactive furnaces, furnace tops and shells, and torch sticks remain nonemissive.

Screen-space indirect lighting consumes resolved HDR radiance, resolved `RGBA8Unorm` albedo and ambient accessibility, resolved depth, resolved `R8Unorm` reactive data, and a full-resolution normal guide. High quality traces four rays at half resolution with 24 Hi-Z iterations and three a-trous passes. The internal medium mode traces two rays at quarter resolution with 16 Hi-Z iterations and two a-trous passes. The retained compatibility Boolean maps false to off and true to high quality. Both active modes cap Hi-Z lookup at mip 7 and temporal age at 32 frames. Two reduced-resolution `R8Unorm` histories retain the current and previous reactive masks. Report the current helper-calculated persistent payload for High, Medium, and Off at the tested drawable; Off retains only the neutral texture and skips indirect compute.

The coherent scene target helper accounts for single-sample HDR resolve, surface resolve, reactive resolve, depth resolve, and the water-refraction copy. The 4x HDR, surface, reactive, R32 resolve-depth-key, and depth attachments use tile memory and contribute no persistent multisample payload. Target allocation logs both the calculated payload and Metal's reported sizes for persistent and memoryless resources. The tile kernel averages covered HDR samples, selects surface and reactive data with the full-precision device-depth key, and stores HDR alpha as one before the ordinary resolve.

Indirect targets allocate only at construction, resize, or quality change. A frame performs bounded full-resolution depth and normal preparation, one dispatch per Hi-Z mip, bounded reduced-resolution trace and temporal dispatches, two or three reduced-resolution filter dispatches, one fullscreen additive apply, and one in-place full-resolution opaque-fog dispatch. The fog dispatch adds no persistent texture and runs after tracing so its camera-dependent color does not enter bounce history. Clouds, rain, and snow follow in one single-sample resolved pass. They depth-test against resolved opaque depth and do not allocate or write an MRT material target.

Temporal history resets for resize, teleport, FOV or quality change, world or session change, direct-light source change, invalid prior depth, resident emission or opaque-light-transport edits, clock rewind, and a time jump greater than eight ticks. Air and water transitions preserve history because water renders after indirect lighting and neither material changes derived light. A bounded reactive mask rejects history at moving gameplay objects and their prior pixels. Far preview-to-final replacement, canopy arrival, connected-parent refinement, and exact-to-far handoff preserve history. The `ssao` settings key and `RYCRAFT_SSAO` variable remain compatibility names for the indirect-light toggle.

## Atmosphere, weather, shadows, and screen-space workload

The physical rendering path keeps one bounded frame graph. Every pass must justify a distinct camera, resolved input, persistent history, or reduced-resolution target.

| Passes | Required dependency |
|---|---|
| Weather upload, atmosphere LUTs, and cloud shadows | Immutable regional snapshot, slow optical state, and pre-opaque transmittance |
| Four detailed shadow cascades and one horizon cascade | Light camera, grouped depth targets, cached refresh state, and exact or far caster ownership |
| Screen-space lighting | Resolved HDR radiance, depth, albedo, accessibility, normals, reactive data, Hi-Z state, and temporal history |
| Volumetric clouds | Resolved depth, fixed 3D noise, regional weather maps, quarter-resolution march, and temporal hit depth |
| Lightning | Resolved depth, cloud hit depth, and deterministic visible storm events |
| Water | An opaque-color copy plus resolved opaque color and depth |
| Froxel air medium | Reconstructed depth, atmosphere LUTs, terrain and cloud shadows, weather extinction, and half-resolution history |
| Weather particles | Resolved depth, local precipitation, physical wind, and meter-scaled atmospheric attenuation |
| Exposure, flare, bloom, and final composite | GPU reductions, a bounded HDR pyramid, then one tonemap and grade |
| UI | Display-resolution overlay after the graded scene |

- Packed light allocates one byte only after a cube receives nonzero skylight or block light. Flooding is capped at 4,096 cells per reconciled cube and reports a six-bit changed-face mask. Player edits converge only their bounded 3 by 3 by 3 affected cube neighborhood synchronously; fluid lighting remains on the deferred budget.
- Five shadow depth targets are grouped as two near array slices, two far array slices, and one horizon texture. Their maximum refresh intervals are one, one, two, four, and eight frames unless a snapped projection, caster revision, or light-direction change invalidates a cache earlier. A skipped cascade retains the matrix paired with its existing depth.
- High screen-space lighting traces four cosine-weighted rays per half-resolution pixel with a 24-iteration Hi-Z cap and three a-trous passes. Medium traces two rays per quarter-resolution pixel with a 16-iteration cap and two passes. Persistent targets reallocate only on resize or quality change.
- `WeatherSystem` owns one utility worker, at most one running build, and at most one latest pending request. Each build samples learned elevation and climate for two 81 by 81 slices at 256-block spacing without routing hydrology or constructing surface ecology. Recenter work starts after 1,024 blocks of travel, readers retain the previous immutable snapshot without waiting, deferred authority retries without escaping the worker, and lightning queries return at most eight events.
- Atmosphere owns fixed 256 by 64 transmittance, 32 by 32 multiple-scattering, and 192 by 108 sky-view LUTs. Slow LUTs refresh only when optical parameters change.
- Cloud noise is generated after world entry on one cancellable utility worker as one 128-cubed base volume, one 32-cubed erosion volume, and one two-dimensional curl map. The render thread polls a completed CPU payload and uploads it only when its world instance and seed still match. Quality Off starts no noise work or allocation, and High or Medium uses neutral cloud shadow until the upload is ready. Cancellation does not consume a retry. Failed builds retain diagnostic timing and use bounded 100 and 500 millisecond backoff before latching on the third failure, so a deterministic failure cannot create a per-frame worker storm. High and Medium clouds march at quarter resolution with 48 or 24 view steps and 6 or 3 light steps. The snapped cloud shadow covers 16,384 blocks.
- The froxel volume is fixed at 160 by 104 by 64 and integrates at half resolution. Disabling volumetric lighting skips injection and marching but retains the lower-cost LUT aerial-perspective pass.
- Lightning geometry has a bounded segment count. Thunder admission keeps at most 16 pending events and 64 remembered IDs, schedules at the physical speed of sound of 343 meters per second, and performs no file I/O or asset decoding. Generator v4 converts block distance through its shared 7.5-meter scale before delay and attenuation.

GPU total frame time comes from `GPUEndTime - GPUStartTime`. Optional Metal counter timestamps attribute pass changes, but overlapping tile-based stages can sum above wall-clock time. Do not add pass durations to infer the total. Record current persistent allocations from the renderer's accounting helpers and Metal counters for the exact drawable and settings under test. A historical byte total is not evidence for a changed build.

If this path exceeds budget, preserve correctness and reduce cost in this order: cloud view steps, froxel resolution, horizon-cascade refresh rate or resolution, then screen-space ray or denoiser counts. Do not reduce the 1,536-block detailed-shadow reach, the first two 4,096-square High shadow targets, packed-light correctness, or direct, emissive, and indirect-light separation. Record the visual and timing effect of every downgrade.

## Persistence and I/O

Model verification and authority-page I/O are bootstrap or worker operations. RYTA writes use compression, `fsync`, and atomic rename. They must not run on render or fixed-tick threads.

The decoded page cache counts uncompressed 12-byte samples. Report file-cache and allocator overhead separately. Persisted explored pages are not automatically evicted, so disk growth remains a capacity-planning concern.

The save queue permits at most 32,768 unique cube positions and coalesces repeated snapshots. V4 regions, terrain pages, hydrology pages, gameplay metadata, and block-entity sidecars live under the selected v4 profile only.

## Hot-path allocation and locks

Classify every changed path by thread, frequency, fanout, and hard bound. Pay particular attention to:

- Per-frame far candidate, transition, and screen-space-lighting work
- Per-fixed-tick authority, fluid, furnace, boat, and entity work
- The one-lock packed-light batch for animals, dropped items, and boats, with scratch capacity reused across ticks
- Per-block density, water, cave, material, and light loops
- Per-cold-page tensor and hydrology scratch allocations
- Cache and scheduler mutex scopes
- Model calls or waits nested under another lock

The inference mutex is a leaf. Terrain-authority and native-hydrology cache mutexes are released before inference, compression, file I/O, or waiting on a flight. The world chunk-map mutex may not contain generation, hydrology, inference, save compression, or blocking waits.

## Qualification commands

Portable tests:

```bash
meson setup build --buildtype=debugoptimized
ninja -C build tests/test_rycraft
ninja -C build test
./build/tests/test_rycraft "[learned]"
./build/tests/test_rycraft "[bootstrap]"
./build/tests/test_rycraft "[reported-water-continuity]"
./build/tests/test_rycraft "[render][far-terrain]"
./build/tests/test_rycraft "[render][indirect]"
./build/tests/test_rycraft "[render][textures][emissive]"
./build/tests/test_rycraft "[render][mesher][bed]"
```

Ordinary CI remains network-free and must not link the downloaded runtime. Cross-compilation or native ARM64 coverage retains a macOS 14 deployment target.

Reference hardware run:

```bash
meson setup build-release --buildtype=release
ninja -C build-release
RYCRAFT_NATIVE_WINDOW=1 \
RYCRAFT_WORLD_SEED=764891 \
RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 \
RYCRAFT_VIEW_DISTANCE=512 \
RYCRAFT_PERF_WARMUP_FRAMES=1200 \
RYCRAFT_PERF_FRAMES=1200 \
./build-release/src/rycraft
```

Record cold and cache-warm runs, then move one chunk and record recovery. Metal validation is separate:

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft
```

Open every captured frame. A successful process exit is not visual evidence.

## Review checklist

- Is every new operation assigned to a thread, frequency, fanout, and hard bound?
- Can render or fixed-tick work wait on learned inference or page I/O?
- Does the unsigned 64-bit seed survive every constructor and cache key?
- Are quality and full fingerprint part of every persisted authority lookup?
- Does a verified installed model start without hashing or downloading the complete pack?
- Does dry-spawn selection remain bounded and validate the final exact location?
- Does exact streaming avoid scanning all 96 vertical sections?
- Do required exact surfaces through 32 chunks retain generation, mesh, upload, and publication-light
  priority after camera and exploration work but before flora and broad work?
- Does exact collision require the current coverage epoch and use canonical generated terrain and
  fluid proxies before exact ownership?
- Does terrain and water publish before canopy completion?
- Does exact or connected desired-LOD debt pause ordinary outer submission and publication, while
  local far workers follow the 8/12/16 combined-debt, local-debt, and clear budgets?
- Does canopy stay at zero before the connected 96-chunk prefix and use exactly one low-priority
  gameplay worker afterward, without a second lane?
- Can nearby preview terrain refine while final authority remains cold?
- Do fixed absolute LOD bands prevent a coarse tier from persisting near the camera?
- Can step-32 water disappear when its corners are dry?
- Does every production far mesh have zero skirt quads?
- Are coarse parents retained until connected replacements are resident?
- Can urgent protected PREVIEW coverage displace lower-ranked queued or parked ordinary coverage?
- Can every nearer desired-LOD job displace queued or dependency-parked outer parents without
  evicting displayed coverage, transitions, exact fallback, or protected lineage?
- Does current protected FINAL work receive capacity before bridges and directional prediction?
- Can only a requested protected FINAL role-selected key reclaim optional distant residency and use
  the complete GPU arena while structural owners stay pinned?
- Do preview and final hydrology builds share one process-wide budget?
- Are fluid, entity, boat, furnace, and light-update costs bounded?
- Do emissive gameplay blocks feed both block light and the indirect-light surface contract?
- Are screen-space-lighting resolution, ray, Hi-Z, denoiser, history, and persistent-memory bounds unchanged or remeasured?
- Do packed light and edit-time convergence remain bounded without delaying torch or furnace emission?
- Does first-visible streaming light use no more than 32 floods per transaction and block mesh
  snapshots until deferred publication completes?
- Do the five shadow targets retain their one, one, two, four, and eight-frame maximum refresh intervals and matching cached matrices?
- Does weather retain one running plus one latest pending build, two 81 by 81 slices, and nonblocking snapshot reads?
- Are atmosphere LUT, cloud, froxel, lightning, and thunder dimensions, queues, refreshes, and allocations bounded and measured?
- Do active furnaces and torch flames contribute emissive radiance while beds, torch sticks, and furnace shells remain nonemissive?
- Does resolved weather remain outside the indirect material pass without adding another multisample target?
- Do far refinement and canopy arrival preserve history while real camera, world, time, quality, and lighting discontinuities reset it?
- Are cold horizon, queue settlement, FPS, GPU lighting cost, and memory claims backed by recorded hardware evidence?
- Were performance and Metal validation runs kept separate?
