# Generator v4 Follow-up Roadmap

This document specifies work intentionally deferred from the first generator v4 pull request. It is
a set of implementation contracts, not permission to weaken the current correctness, entry,
rendering, or performance requirements.

The existing exact cubic world, learned terrain identity, canonical hydrology, and far surface
renderer remain the source of truth until a replacement passes the same tests and visual
qualification. New caches and GPU products are derived data. They may be discarded and rebuilt
without changing a world.

## Goals and invariants

Follow-up work must improve cold entry, movement recovery, memory efficiency, and distant visual
quality while preserving these rules:

- Exact terrain, far terrain, collision, water, materials, and ecology derive from one generation
  identity and one unsigned 64-bit seed.
- The highest-fidelity required terrain around the player wins CPU workers, model requests, cache
  entries, upload space, and GPU residency before optional distant work.
- A coarse parent stays drawable until a connected, revision-ready child replacement is resident.
- Screen-space error selects desired terrain detail. Fixed distance rings are minimum quality
  floors, not a reason to discard a finer resident representation. Within the nearby visible work
  class, horizontal distance ranks before projected error so a farther high-error tile cannot delay
  closer missing detail.
- Water topology, body identity, stage, bed, and shoreline ownership are canonical. A renderer or
  accelerator may not infer a different water system.
- Flora may enrich a drawable surface but may never delay terrain or water publication.
- No optimization may reintroduce retaining walls, deleted wet routes, downward skirts, empty
  cubes, synthetic v3 macro terrain, or legacy hydraulic erosion on a v4 path.
- Render and fixed-tick threads may submit or consume completed work, but they may not wait for
  inference, hydrology, disk I/O, compression, or mesh construction.
- Every fast path has a bounded queue, a cancellation boundary, and a deterministic CPU reference
  path.

## Delivery order

The work is split into independently reviewable programs. Results from an earlier program become
measured inputs to the next one.

| Program | Primary outcome | Depends on |
|---|---|---|
| 0. Baseline and traces | Reproducible cost and quality attribution | Current generator v4 |
| 1. Authority and hydrology work graph | Less repeated cold work with equal output | Program 0 |
| 2. Paged hybrid hierarchy | One persistent multiresolution spatial index | Programs 0 and 1 |
| 3. GPU construction and submission | CPU relief and bounded GPU-driven rendering | Program 2 |
| 4. Distant water and flora | Stable medium and far visual richness | Programs 2 and 3 |
| 5. Ecosystem authority | Plant-functional-type capacity and richer habitat | Program 4 |

Programs 1 through 4 should not be combined into one pull request. Each program must leave the
existing path available for comparison until its qualification matrix passes.

## Primary-source LOD evaluation

The deferred hierarchy design is based on current official source trees rather than a visual
comparison with other renderers:

| Reference | Exact revision | Relevant implementation | Adopt | Avoid |
|---|---|---|---|---|
| [Distant Horizons](https://gitlab.com/distant-horizons-team/distant-horizons/-/tree/8ab8e790edfcec77e668f548b0234cc5cf4acd8e) | Superproject `8ab8e790edfcec77e668f548b0234cc5cf4acd8e`, which pins [core `f61bc5297d4a2f7e6855fd194733cffda09bb313`](https://gitlab.com/distant-horizons-team/distant-horizons-core/-/tree/f61bc5297d4a2f7e6855fd194733cffda09bb313) | A moving sparse XZ quadtree whose nodes contain 64 by 64 grids of vertically run-length-encoded columns | Retain a parent until its complete child family is uploaded, persist bottom-up aggregates, store boundary summaries, replace GPU buffers atomically, and apply queue backpressure | Fixed logarithmic distance rings, modal 2 by 2 reduction for water or flora, border overdraw that hides cracks with boxes, and a purely horizontal hierarchy for all cubic content |
| [Voxy](https://github.com/MCRcortex/voxy/tree/b164a6d98c378ebb6bbb3e538770d76d527c001f) | `b164a6d98c378ebb6bbb3e538770d76d527c001f` | A finite forest of dense 32 by 32 by 32 sections, bottom-up voxel mips, flat GPU nodes, GPU traversal, Hi-Z culling, and compact request feedback | Use a paged forest instead of one global root, flat pool-allocated nodes, child masks, two-phase child publication, GPU request compaction, and memory-pressure eviction | Dense voxel payloads at every level, highest-opacity representative-voxel mips, projected node area as the only quality metric, and unstitched discrete swaps |
| [terrain-diffusion-mc](https://github.com/xandergos/terrain-diffusion-mc/tree/23d3f50e5108882bb88a03c3ab048aa63633a02f) | `23d3f50e5108882bb88a03c3ab048aa63633a02f` | The paper-compatible coarse, latent, and decoder window pipeline behind a blocking heightfield provider | Keep its window geometry, accumulation, reconstruction, and compatibility vectors as learned-authority references | Its single-threaded blocking request service, hydrology-unaware Perlin relief, vanilla sea fill, generic fluid reduction, categorical biome classification, and vanilla biome-driven vegetation |

The supporting source paths make the distinctions explicit:

- Distant Horizons selects and publishes quadtree families in
  [`LodQuadTree.java`](https://gitlab.com/distant-horizons-team/distant-horizons-core/-/blob/f61bc5297d4a2f7e6855fd194733cffda09bb313/core/src/main/java/com/seibel/distanthorizons/core/render/QuadTree/LodQuadTree.java),
  aggregates vertical columns in
  [`FullDataSourceV2.java`](https://gitlab.com/distant-horizons-team/distant-horizons-core/-/blob/f61bc5297d4a2f7e6855fd194733cffda09bb313/core/src/main/java/com/seibel/distanthorizons/core/dataObjects/fullData/sources/FullDataSourceV2.java),
  and deliberately overdraws column borders in
  [`ColumnBox.java`](https://gitlab.com/distant-horizons-team/distant-horizons-core/-/blob/f61bc5297d4a2f7e6855fd194733cffda09bb313/core/src/main/java/com/seibel/distanthorizons/core/dataObjects/render/bufferBuilding/ColumnBox.java).
- Voxy defines its finite three-dimensional section forest in
  [`WorldSection.java`](https://github.com/MCRcortex/voxy/blob/b164a6d98c378ebb6bbb3e538770d76d527c001f/src/main/java/me/cortex/voxy/common/world/WorldSection.java),
  stores compact GPU nodes in
  [`NodeStore.java`](https://github.com/MCRcortex/voxy/blob/b164a6d98c378ebb6bbb3e538770d76d527c001f/src/main/java/me/cortex/voxy/client/core/rendering/hierachical/NodeStore.java),
  and selects missing children while retaining a parent in
  [`traversal_dev.comp`](https://github.com/MCRcortex/voxy/blob/b164a6d98c378ebb6bbb3e538770d76d527c001f/src/main/resources/assets/voxy/shaders/lod/hierarchical/traversal_dev.comp).
- The pinned Minecraft reference implements learned windows in
  [`WorldPipeline.java`](https://github.com/xandergos/terrain-diffusion-mc/blob/23d3f50e5108882bb88a03c3ab048aa63633a02f/src/main/java/com/github/xandergos/terraindiffusionmc/pipeline/WorldPipeline.java),
  exposes its blocking request path in
  [`LocalTerrainProvider.java`](https://github.com/xandergos/terrain-diffusion-mc/blob/23d3f50e5108882bb88a03c3ab048aa63633a02f/src/main/java/com/github/xandergos/terraindiffusionmc/pipeline/LocalTerrainProvider.java),
  converts learned elevation and climate into categorical Minecraft labels in
  [`BiomeClassifier.java`](https://github.com/xandergos/terrain-diffusion-mc/blob/23d3f50e5108882bb88a03c3ab048aa63633a02f/src/main/java/com/github/xandergos/terraindiffusionmc/pipeline/BiomeClassifier.java),
  and configures ordinary sea fill rather than canonical hydrology in
  [`terrain_diffusion.json`](https://github.com/xandergos/terrain-diffusion-mc/blob/23d3f50e5108882bb88a03c3ab048aa63633a02f/src/main/resources/data/terrain-diffusion-mc/worldgen/noise_settings/terrain_diffusion.json).

Neither reference renderer supplies Rycraft's canonical water identities or a stable flora
hierarchy. Distant Horizons stores liquid and vegetation as ordinary vertical block spans. Voxy
stores them as ordinary voxel materials and models. Their generic parent reducers may erase a
narrow river or thin vegetation, so Rycraft keeps water and flora as independent semantic
attachments throughout aggregation, residency, and handoff.

## Nine delivery work packages

These packages are the review and merge boundaries for Programs 0 through 4:

1. Build reproducible cold, warm, movement, hover, reversal, water, and flora traces with
   critical-path and visible screen-error-debt attribution.
2. Implement signed page keys, flat node pools, the surface-column payload, optional sparse
   volumetric bricks, boundary summaries, a versioned derived cache, and corruption-safe rebuild.
3. Implement deterministic semantic aggregation for occupancy, geometric error, materials, water,
   flora, lighting, and emissives, with differential and wet-route-preservation tests.
4. Implement the CPU reference selector, projected-error hysteresis, camera epochs, preemptive
   scheduling, complete-family publication, and eviction by screen benefit per byte and rebuild
   cost.
5. Implement 2:1 topology-correct transition meshes and half-open shared ownership, including a
   separate canonical-water transition path.
6. Move selection, Hi-Z culling, request compaction, and indirect command construction to Metal
   with bounded feedback, frame-safe heaps, overflow recovery, and a CPU fallback.
7. Implement canonical far-water contours, river ribbons, shorelines, and falls with exact-to-far
   identity and connectivity tests.
8. Implement independent flora and stable-object tiers for rooted instances, crown clusters,
   impostors, canopy aggregates, shadows, and explicit emissive landmarks.
9. Qualify persistence, cache replacement, arbitrary readiness order, camera jitter, high-speed
   flight, queue saturation, memory pressure, narrow rivers, broad water, and nonzero flora
   coverage at every drawable level.

## Program 0: baseline and trace contract

Implemented by the generator v4 preparation PR: `common/trace.hpp` plus the checked-in
`rycraft_trace_summary` tool. Tracing is disabled by default with a single-atomic emit guard, no
per-frame allocation, and a bounded buffer; enabling it writes a Chrome Trace Event JSON and a
binary trace. See [architecture.md](architecture.md) for the record layout and controls.

Optimization begins with one trace format shared by startup, authority, hydrology, exact streaming,
far streaming, upload, and rendering. Each work item records its generation fingerprint, immutable
spatial key, camera epoch, requested quality, dependency reason, queue timestamps, execution time,
bytes retained, and cancellation result.

The trace must answer:

- Which coarse, Base, and decoder windows were requested, reused, persisted, or evicted?
- Which learned rectangles and hydrology owners overlap?
- How much time was spent in inference, low-frequency reconstruction, routing, ecology, meshing,
  compression, disk I/O, upload, and waiting for a dependency?
- Which visible tile had the greatest projected error, and why was its desired representation not
  resident?
- Which optional item occupied a worker, cache entry, or GPU allocation while protected work was
  waiting?
- How many flora anchors and canonical water features were omitted by each displayed tier?

Acceptance requires:

- A Chrome Trace Event or equivalent machine-readable export for a cold entry, warm entry,
  one-chunk movement, flight across an LOD boundary, and five-minute settlement.
- Signposts around every CPU and GPU phase, with Metal counter samples collected only in the
  dedicated validation run.
- No per-frame heap allocation or unbounded trace growth when tracing is disabled.
- A checked-in summary tool that reports p50, p95, maximum, queue depth, cache reuse, cancellation,
  and critical-path attribution from one trace.

### Recorded starting point

The retained generator v4 implementation already uses static four-window Base and Decoder batches
with deterministic repeated-tail padding. Its far scheduler lets an urgent camera-critical
refinement run before the worker reservation for distant coverage parents. Follow-up work must
preserve both behaviors unless a qualified replacement is demonstrably better.

The connected 96-chunk terrain-and-water prefix is the complete ordinary coverage obligation for
entry. After that prefix is drawable, unfinished exact publication through 32 chunks or any
connected visible desired-LOD debt pauses ordinary outer-horizon submission and publication. Near
work proceeds nearest-first and may displace a queued or dependency-parked outer parent. Within the
nearby visible class, distance ranks before projected screen error. It may not evict a displayed
parent, the connected entry prefix, a transition endpoint, exact fallback, or protected lineage.
Ordinary outer submission and publication resume only after both exact publication debt and nearest
desired-LOD debt clear.

Canopy service is also bounded deliberately. It uses zero workers during preparation and until the
connected entry prefix is drawable. Gameplay then guarantees exactly one low-priority canopy worker
while terrain continues to take precedence. There is no second gameplay canopy lane. The worker
serves missing drawable PREVIEW attachments before FINAL promotions and keeps the source attachment
resident until its replacement uploads.

The latest cold audit reached first playable at 30.0823 seconds, missing the strict 30-second gate
by 82.3 milliseconds. Full radius-512 settlement took another 42.4057 seconds, for 72.4903 seconds
total. It reported zero inference or hydrology failures, all 100 protected boundaries matched, and
maximum RSS was 8.05 GiB. This is a measured miss, not a 30-second pass, and it does not visually
qualify the deferred hierarchy. Program 0 must still produce reproducible cold, warm, and
cache-cleared distributions on the canonical seed-764891 route and the required opened images
before claiming complete performance or visual qualification.

The July 23 seed-764891 aerial capture also exposes a specific transition defect. Bright,
axis-aligned seams follow the displayed step-1 to step-2 boundary, with nearby water breaking into
regular strips and apparent gaps. This is failing evidence. The current mesher gives every step-2
or coarser tile an independently constructed outer-cell fan that converges on two-block canonical
boundary samples, while step 1 emits its ordinary block tops and positive-face risers without that
fan. Equal boundary-height hashes prove matching edge samples, but they do not prove positive-area
coverage, compatible triangulation, or compatible material and water ownership across two
different payloads. Existing topology tests inspect isolated meshes and synthetic paired edges;
they do not raster-qualify the complete real-model step-1 and step-2 union under culling, exact
ownership masks, temporal replacement, and 4x MSAA.

Program 2 must replace this implicit agreement with an explicit neighbor-family product. Persist
one canonical fine-edge record per hierarchy node, derive the parent edge from the same child
samples, and build one half-open transition strip for each unequal neighbor pair from the fine edge
vertices to the coarse interior. The pair key includes both content hashes, levels, authority
quality, exact-ownership revision, and canonical-water edge summary. It publishes only after both
endpoints, the complete positive-area strip, and its paired water contour are resident. One
lexicographic side owns the strip, and the other side emits no overlapping boundary triangles.
Raster tests must render every orientation and all four child phases with back-face culling, 4x
MSAA, exact masks, PREVIEW-to-FINAL replacement, camera jitter, and reversed readiness. They must
assert zero uncovered samples, zero double coverage, equal depth along the shared edge, one water
owner, and no axis-aligned color or depth line in the resolved image.

## Program 1: authority and hydrology work graph

### Learned-authority request planner

Implemented by the generator v4 preparation PR. `world/learned_authority_graph.hpp` plans the unique
coarse, Base, and Decoder windows once, computes the working-set bound before inference, pins the
protected closure so no referenced window is recomputed or evicted, and cost-gates 2x2 owner
grouping by unique-window count. The enumeration is cross-checked against the windows the backend
actually computes, and authority bytes and qualification hashes are unchanged.

Replace independent rectangular requests with an immutable request graph before inference begins.
The planner unions overlapping native rectangles, enumerates their exact coarse, latent, and decoder
window dependencies, and schedules each unique window once. Consumers receive views into the
completed shared product.

The planner must:

- Preserve lexicographic accumulation order, the existing static four-window Base and Decoder
  batches, lexicographic missing-window grouping, and deterministic repeated-tail Decoder padding.
- Keep Coarse execution scalar under the current generation identity. A Coarse batch or shape
  change is allowed only under a new qualified identity because the evaluated static batch changed
  raw output for the pinned graph.
- Treat any future Base or Decoder batch-shape change as a provider and generation-identity change.
  The pinned graph and Core ML path must accept it, and repeated, reverse-order, concurrent, and
  cache-cleared runs must reproduce the newly qualified quantized hashes.
- Prefer a larger already-required rectangle only when it costs fewer unique model windows than the
  independent requests. Union by bounding box alone is not sufficient.
- Keep spawn, exact exploration, protected handoff, visible refinement, preview, and prediction as
  explicit priority classes.
- Persist only fingerprinted authority products that already meet the existing persistence
  contract. A scheduling change cannot make a partially accumulated page publishable.
- Retain completed window products long enough to cover the entire cold-entry dependency graph.
  The planner computes this working-set bound before work starts instead of discovering it through
  cache churn.
- Group directly adjacent protected hydrology-owner rectangles in deterministic groups of at most
  two by two when the combined half-open rectangle stays within the 1,048,576-sample query bound.
  Cropping the grouped FINAL product back to each owner must produce exactly the same quantized and
  dequantized samples as preparing every 517 by 517 owner independently, including negative
  coordinates and shared aprons. Grouping is a request optimization, not a new terrain authority.

Acceptance requires:

- Identical authority page bytes and canonical qualification hashes before and after the planner.
- No duplicate model execution for one window in a cold-entry trace.
- No cache eviction of a window still referenced by the active protected dependency graph.
- Cold and warm entry measurements that meet the gates in
  [performance conventions](performance-conventions.md).
- Fault injection for cancellation, failed inference, stale epochs, interrupted persistence, and a
  request larger than every configured bound.

### Hydrology hierarchy and scheduling

Partially implemented. The generator v4 preparation PR adds camera-aware admission: the process-wide
build gate ranks waiters by the shared `AuthorityRequestPriority` and reserves lanes so a distant
owner cannot occupy every hydrology lane while the player's exact band is unresolved. The two-level
owner-summary and on-demand-detail split, the structure-of-arrays scratch layout, the benchmarked
radix or bucket Priority-Flood, and the vectorized local kernels remain deferred, together with
their 2x cold-CPU and 50-percent repeated-read gates, which require the retained reference path and
real-model benchmarking.

Canonical hydrology should become a two-level immutable product:

1. A compact owner summary stores boundary elevations, spill candidates, depression roots,
   accumulated runoff, outgoing flow fractions, water-body identities, and a hash of the native
   authority input.
2. A block-resolution detail product expands only the rectangles required by exact columns, far
   geometry, water rendering, or ecology.

Independent owners may prepare local summaries concurrently. Boundary reconciliation remains a
deterministic ordered operation. The finalized summary DAG then permits detail products to execute
in parallel without revisiting upstream terrain.

Implementation requirements:

- Keep D-infinity multi-receiver flow, Fill-Spill-Merge semantics, explicit falls, stable
  `WaterBodyId` values, and current quantization.
- Replace repeated rectangle scans with one owner-local structure-of-arrays layout for elevation,
  climate, flow fractions, runoff, stage, bed, identity, and flags.
- Use a radix or bucket queue for quantized Priority-Flood elevations after a benchmark proves it
  improves the real distribution. Keep the reference heap path for differential tests.
- Vectorize pure local kernels such as slope, D-infinity receivers, evapotranspiration, runoff, and
  shoreline distance. Boundary union, spill ownership, and stable ID assignment remain explicitly
  ordered.
- Store the summary in a fingerprinted, versioned envelope. Detail rasters are cacheable derived
  products and may be evicted independently.
- Admit hydrology tasks through the same camera-aware work graph as learned authority. A distant
  owner cannot occupy every hydrology lane while the player's exact band is unresolved.

Acceptance requires:

- Bit-identical body IDs and quantized stage, bed, flow, discharge, seasonality, hydroperiod, and
  shoreline fields against the CPU reference corpus.
- Zero wet-route deletions, zero hydrology-driven dry-terrain raising, monotone ordinary river
  stages, flat lake interiors, supported beds, and explicit ownership for every legal stage jump.
- Cross-owner results independent of request order, worker count, cache state, and page origin,
  including negative coordinates.
- At least a 2x reduction in cold protected-route hydrology CPU time and at least a 50 percent
  reduction in repeated native-sample reads on the documented M4 Max.
- Peak scratch memory bounded before execution and reported separately from persistent summaries.
- `deferredBuilds` counts completed native-hydrology cache build attempts that returned typed
  `DEFERRED` because learned authority was not ready. It is a monotonic attempt counter, not the
  current parked count, active-build gauge, or failure count. Interval diagnostics subtract the
  preceding snapshot, while `activeBuilds` is reported directly.

## Program 2: paged hybrid hierarchy

### Chosen structure

A classic pointer-heavy sparse voxel octree is not the default choice for Rycraft. It duplicates
metadata for a mostly surface-dominated far world, has poor cache locality, and forces an artificial
single root on a horizontally unbounded coordinate space.

| Candidate | Benefit | Limitation | Decision |
|---|---|---|---|
| Current surface tiles | Small migration | Repeated metadata at every tier | Reject |
| Geometry clipmap | Predictable work | Duplicate rings and weak cubic ownership | Reject |
| Pointer octree | Direct 3D hierarchy | Pointer cost, weak locality, and no infinite root | Reject |
| Horizontal quadtree only | Compact surface ownership | Cannot refine one local vertical feature | Reject |
| Paged surface quadtree with sparse volumetric bricks | Compact common case with local cubic detail | More involved derived-cache construction | Select |

The selected design keeps the practical goals associated with long-distance Minecraft renderers:
long-lived coarse parents, camera-centered refinement, hierarchical reuse, and screen-space
selection. It adapts those goals to Rycraft's cubic vertical range and canonical water instead of
copying another renderer's storage format.

The proposed structure is a surface-first paged hierarchy with sparse volumetric attachments:

- The mutable near world remains 16 by 16 by 16 exact cubes through the current exact radius.
- One top-level surface region is 1,024 by 1,024 blocks in XZ and aligns with one
  terrain-authority page. Its seven quadtree levels run from 16 by 16-block leaves through the
  1,024-block root.
- A sparse radix map addresses surface regions by a signed `(regionX, regionZ)` tuple. A node uses a
  level and region-local Morton index. There is no world-sized root, lossy packed global key, or
  coordinate narrowing.
- A surface leaf stores ordered vertical runs, height intervals, and discontinuity masks. Ordinary
  learned terrain, shorelines, vegetation grounding, and most distant silhouettes therefore avoid
  allocating nodes through solid rock or empty sky.
- A surface node references sparse 16 by 16 by 16 volumetric bricks only where independent Y
  refinement pays for itself, including overhangs, cave mouths, floating terrain, structures,
  deep cuts, and edited cubic exceptions. Volumetric pages add signed region Y and local Morton
  coordinates without imposing a world-scale octree.
- Each surface leaf points to a sorted cold-table range of `(signedBrickY, brickIndex)` attachment
  entries. Multiple disjoint Y ranges therefore remain sparse, and a brick cannot be found through
  an implicit unsigned height or a scan of unrelated elevations.
- Volumetric mips reduce canonical 2 by 2 by 2 child groups bottom-up. Empty, uniform-solid, and
  uniform-material groups collapse; mixed groups preserve conservative occupancy, material,
  emissive, lighting, and geometric-error bounds. Canonical water remains a semantic attachment
  and is never inferred from a representative solid voxel.
- Surface runs own the ordinary heightfield top. A volumetric brick owns only exception faces inside
  its half-open 16-cubed bounds. The derivation removes coplanar duplicates at the handoff and pins
  the surface boundary until the complete brick family is resident.
- Each region stores breadth-first node arrays. A logical surface node owns a four-bit child mask,
  occupancy class, content hash, axis-aligned bounds, conservative geometric-error bound, and
  optional material, water, flora, lighting, emissive, and volumetric-attachment summaries.
- Keep the hot traversal record at or below 32 bytes. It stores the child mask, compact child base
  index, occupancy class, bounds, error, and cold-table indices. Content hashes and optional
  semantic records live in parallel cold tables addressed by those indices.
- Empty, uniform, surface-only, water-only, mixed-run, and volumetric-exception nodes have distinct
  payload encodings. Mixed volumetric bricks may store palette-compressed 4 by 4 by 4 microblocks.
- Parent error is measured against its canonical children. A parent is therefore a real
  approximation of the same authority, not terrain sampled from a different generator path.
- Water summaries include body IDs, minimum and maximum stage, shoreline crossing masks, river
  contour and ribbon references, signed-distance ranges, crossing edges, and explicit-fall masks.
  A node with possible topology either carries clipped vector geometry or forces refinement. It
  cannot collapse to a dry parent.
- Flora summaries include capacity, crown-height bounds, dominant functional groups, anchor count,
  deterministic representative reservoirs, and an anchor-range reference. Flora payloads remain
  separate from base terrain and water residency. Dynamic entities and gameplay collision remain
  outside the hierarchy.
- Each page persists four canonical edge summaries with terrain samples, material transitions,
  water crossings, flora crown bounds, and source hashes. A mesh or transition can therefore
  validate its neighbor without loading the neighbor's complete page.

Flat arrays and relative indices make traversal cache-friendly on the CPU and directly consumable
by Metal. Surface pages and optional volumetric pages are immutable and content-addressed. A
copy-on-write overlay represents player edits only where far visualization of those edits is later
required.

### Ownership and traversal

The hierarchy answers spatial selection, conservative culling, error estimation, and derived-mesh
dependency queries. It does not replace canonical authority. Gameplay physics continues to use
exact cubes or the existing canonical column-plan proxy, never renderer hierarchy payloads.

Selection begins with the current camera and projection:

1. Reject a node only when its conservative bounds are outside the frustum or a qualified
   hierarchical depth test proves it hidden.
2. Project the node's geometric error into pixels.
3. Refine while error exceeds the quality threshold, the node intersects the protected near field,
   water topology requires a child, or a transition neighbor requires a child.
4. Apply separate refine and coarsen thresholds plus a minimum residency time to prevent temporal
   oscillation.
5. Keep the displayed parent until the complete selected child family, exterior transition owners,
   and required canonical-water products are GPU resident, then publish the family atomically.

The exact radius is a hard desired-quality and admission floor. A valid parent may remain drawable
inside unresolved exact ownership only as transient, measured error debt while the required exact
replacement runs first. It cannot count as settled coverage, displace exact work, or retire before
the exact replacement is revision-ready and resident. The hierarchy may select finer far detail
outside the exact radius whenever screen-space error requires it.

Flora publication is a separate transaction. The source-level flora attachment remains drawable
until the target attachment is resident, even when terrain and water have already changed level.
Temporal dithering or a bounded cross-fade may soften flora and material replacement only after both
representations are resident. It may not alter geometry, water stage, body identity, or collision.

Every displayed terrain edge is either equal-level or 2:1. Unequal edges use one shared transition
strip built from the coarser edge at the finer edge's canonical sample positions. Half-open
ownership assigns every positive-area triangle once. Rycraft does not adopt Distant Horizons'
border overdraw, Voxy's unstitched discrete edge, a downward skirt, or a vertical crack-hiding box.

Publication is one family transaction. The parent remains the terrain, water, shadow, and occlusion
owner while any selected child, transition strip, or canonical-water product is missing. The frame
that commits a complete family retires the parent through frame-safe reclamation. A stale camera
epoch or failed allocation discards the incomplete family and leaves the parent drawable.

Acceptance requires:

- Identical selected nodes for equal camera snapshots regardless of traversal worker count.
- No 2:1 neighbor violation, missing parent coverage, transition crack, downward panel, or
  preview-to-final geometry mismatch.
- Maximum displayed projected terrain error at or below the configured threshold for every settled
  capture, with a lower threshold for silhouettes and water crossings. Cold entry and movement
  captures separately bound the duration and distance of any transient parent error debt.
- No visible refine/coarsen oscillation during a stationary 60-second capture or a repeated
  boundary flight.
- At least a 40 percent reduction in CPU residency metadata and selection time relative to an
  equivalently covered all-tier tile set.
- A bounded region load, build, eviction, and GPU-upload budget. The camera's protected lineage and
  displayed transition endpoints remain pinned.

### Persistence and migration

Hybrid hierarchy pages live in a versioned derived-cache directory below the world profile. Their
header records the complete generation fingerprint, source authority hashes, hierarchy schema,
surface-run and volumetric-brick encoding revisions, region coordinates, payload sizes, and CRC-32
values.

A missing, corrupt, or old hierarchy page is rebuilt from canonical authority. It does not make the
world incompatible and never changes world metadata. Migration is therefore cache replacement, not
save migration.

Persist enough immutable hierarchy and renderer-neutral terrain-and-water payload data to restore
the last connected parent frontier without rerunning learned inference or canonical hydrology.
Optional device-specific mesh and command products use a separate cache key containing their source
hashes, renderer revision, pixel format, and GPU feature set. Those products remain rebuildable and
never become world authority.

## Program 3: GPU construction and Metal-native submission

### GPU terrain work

GPU work starts with derived products whose failure cannot change world authority:

1. Far vertex and index emission from selected surface nodes and volumetric bricks
2. Shared transition-strip construction
3. Normal, material-weight, and shoreline attribute generation
4. Flora-anchor visibility compaction and distant instance generation
5. Water contour tessellation and river-ribbon expansion

After those derived stages pass, traces must separately evaluate GPU implementations for bounded
exact-cube lazy density evaluation, strata and surface-material classification, block occupancy
compaction, and other pure generation kernels. A kernel moves only when its CPU fallback remains
available, its complete output is byte-identical for the differential corpus, and it reduces the
protected generation critical path rather than merely moving optional work.

Hydrology's pure local fields may move to compute under the same rule. Learned elevation
reconstruction and quantization remain on the qualified CPU path until a separate differential
suite proves every persisted sample identical. ONNX Runtime and Core ML continue to own model
execution. Program 0 also records every Core ML CPU-fallback node. Static-shape specialization,
supported operator substitution, and graph partition changes may be evaluated to reduce fallback,
but any graph, partition, output, or provider change creates a new generation identity and requires
full qualification. Rycraft does not reimplement model operators in an unversioned custom Metal
kernel.

Each compute job reads immutable, fingerprinted input buffers and writes into a bounded output
slice. An asynchronous completion record carries the spatial key, source hashes, camera epoch,
counts, and overflow status. The CPU publishes the result only after validating those fields. An
overflow, device loss, validation error, or canceled epoch discards the slice and retains the
current drawable parent.

Acceptance requires:

- Byte-identical topology and quantized vertex attributes against the CPU mesher for the
  differential corpus, or a versioned rendering-only encoding with mathematically bounded error
  and identical half-open ownership.
- No command-buffer wait on the render or fixed-tick thread.
- No GPU write into a range referenced by an in-flight frame.
- Bounded dispatch dimensions and an explicit overflow path for adversarial water, transition, and
  flora cases.
- At least a 3x reduction in far construction CPU time without increasing p95 frame time or the
  five-minute total unified-memory peak.
- Every accepted exact-generation or hydrology kernel remains byte-identical to its CPU reference
  across seeds, negative coordinates, vertical extrema, worker counts, cancellation, and restart.
- A generation kernel is adopted only if repeated traces show at least a 25 percent reduction in
  its protected CPU phase and no regression in entry, movement recovery, frame time, or memory.

### Metal resource and draw path

The first implementation should use flat hierarchy buffers, argument buffers, and indirect command
buffers rather than adopting mesh or object shaders as a prerequisite. Feature-gated experiments
may evaluate newer pipeline stages later, but the required macOS 14 Apple Silicon path must remain
complete.

The Metal work includes:

- Place far vertex, index, hierarchy, instance, and command allocations in explicit heap classes
  with separate transient and persistent lifetimes.
- Use one frame-safe allocation protocol for compute output, blit copies, and draws. Fence or
  shared-event ownership must be visible in diagnostics.
- Compact selected and visible nodes on the GPU, then encode bounded indirect indexed draws by
  material and pass. Exact terrain retains direct ownership until the indirect path proves equal
  culling and handoff.
- Reuse the depth pyramid for conservative far occlusion. A node rejected by occlusion keeps its
  residency and can return without regeneration.
- Use threadgroup tiling, `simdgroup` reductions, and coalesced structure-of-arrays access for
  height, occupancy, water, and instance kernels.
- Evaluate binary pipeline archives for startup compilation only after measuring archive creation,
  invalidation, and fallback behavior on the minimum supported operating system.
- Use `MTLCounterSampleBuffer` and signposts for attribution. Do not add overlapping per-pass GPU
  durations to infer wall-clock frame time.
- Evaluate Accelerate or vectorized standard-library kernels on CPU reference paths before adding
  custom assembly or architecture-specific intrinsics.

Acceptance requires:

- Direct and indirect paths produce the same visible ownership, material grouping, shadow
  participation, water order, and exact handoff across the capture matrix.
- Indirect command count, hierarchy traversal work, heap bytes, fragmentation, and reclaimed bytes
  appear in diagnostics.
- No Metal validation, API validation, shader validation, uninitialized-resource, or
  use-after-free finding.
- At least a 50 percent reduction in CPU draw-encoding time for the settled radius-512 scene.
- The lowest sustained one-second frame rate, p95 frame time, movement recovery, and total memory
  still meet the existing gates at native resolution with 4x MSAA.

## Program 4: distant water and flora

### Distant water

Terrain LOD and water LOD use the same selected hierarchy nodes, but water has an additional topology
constraint. A water feature may refine farther than nearby dry terrain when its projected width,
stage discontinuity, specular silhouette, or shoreline curvature would otherwise be lost.

### Representation

- Oceans and broad lakes use clipped canonical body contours with one stage per standing body.
- Rivers and distributaries use canonical centerline ribbons with stage, width, depth, flow,
  discharge, and junction identity. Width is clamped for antialiasing only in the renderer, never
  in authority.
- Wetlands use shallow coverage polygons plus hydroperiod and groundwater flags.
- Falls use explicit top and bottom stages, falling-column bounds, and mist eligibility.
- Every representation retains half-open ownership and stable boundary vertices across region and
  LOD edges.
- A signed shoreline-distance field controls edge placement, wet material blending, foam, and
  subpixel coverage. Corner-only terrain sampling is never a water-presence test.
- Extract body contours from the signed-distance field with deterministic marching squares in
  global cell coordinates. Resolve saddle cases from the canonical body-side samples, quantize
  crossings to 1/256 block, and assign every page-edge segment to one half-open owner.
- Simplify contours only with a topology-preserving constrained pass. Pin page crossings, extrema,
  reach junctions, fall endpoints, and any vertex whose removal could change connectivity; bound
  remaining displacement by both projected error and half the local feature width.
- Build river junctions as shared polygons keyed by stable reach IDs. Interior reaches end at the
  shared junction and never cap independently. Only true network termini receive caps, while page
  clips remain open and transfer ownership to the adjacent page.
- Quantized tessellation must reproduce pinned contour and ribbon endpoints exactly at every LOD.
  Interior deviation may not exceed 1/256 block in authority space or the active screen-space
  error budget, whichever is stricter.

### Rendering

The distant water shader should lower geometric subdivision only when projected curvature and
normal variation are below error bounds. It keeps analytic waves, body-scale flow direction, depth
color, sky and sun reflection, and atmosphere attenuation coherent with near water. Reflection
quality may fall with projected size, but the surface, shoreline, and body connection remain.

Temporal transitions reuse body identity and canonical stage. The parent remains the sole water
owner until the complete connected child contour, ribbons, falls, transition geometry, and bed
support are resident. Publication then changes ownership atomically. A renderer may dither
overlapping material response, but it may not interpolate stage, merge distinct identities, expose
both topologies, place a new surface above unsupported terrain, or leave the old surface floating.

Acceptance requires:

- Every canonical route in the water corpus remains connected from exact geometry through step 32.
- Exact and far queries agree on wet or dry state, body ID, quantized stage, and explicit-fall
  ownership at sampled block centers and all shared edges.
- Zero floating surfaces, horizon cutoffs, unrelated-level joins, regular phase-aligned
  staircases, or dry gaps in opened river, lake, delta, estuary, wetland, coast, and waterfall
  captures.
- Rivers narrower than one pixel remain temporally stable through analytic coverage rather than
  blinking between present and absent.
- Distant water adds no authority query to the render thread and stays within a separately reported
  GPU memory and frame-time budget.

### Distant flora

Flora uses deterministic anchors and ecosystem capacity, not a second terrain generator. Its
payload remains independently cancellable and evictable from terrain and water.

### Representation tiers

| Projected form | Representation |
|---|---|
| Block-scale | Existing exact rooted tree and ground-flora geometry |
| Crown-scale | Voxel crown, trunk, and major branch cluster grounded to the displayed surface |
| Several pixels | Species-group crown impostor with height, width, color, wind, and shadow data |
| Subpixel crown | Stochastic coverage cluster preserving canopy density and average height |
| Horizon | Low-frequency canopy height and coverage participating in color and terrain shadow |

The screen-space selector chooses these forms by projected crown size, not fixed distance alone.
Nearby and middle-distance terrain must not form a bare ring while optional individual plants are
still loading. PREVIEW ecology may provide a temporary density product, but FINAL promotion keeps
the same anchor identities and replaces it without a population jump.

Requirements:

- Generate one coordinate-addressed anchor stream per ecology cell. Every tier consumes ranges from
  that stream, so selection order and visit order cannot change density or species.
- Ground anchors against the displayed terrain quality and reject canonical water unless the
  species explicitly supports that habitat.
- Store crown bounds, trunk height, functional group, seasonal-static color, wind response, and
  shadow importance in the hierarchy's independent flora summary.
- Let the terrain base publish with an empty or pending flora attachment. The nearest unfinished
  flora is first only after all protected terrain obligations are admitted.
- Cast aggregated distant shadows only when their projected contribution justifies the update.
  Exact flora owns collision; every farther representation is visual only.
- Feed the later plant-functional-type authority into capacity without changing anchor addressing.

Acceptance requires:

- No visible bare flora ring from the edge of exact vegetation through the middle distance in the
  forest, wetland, riparian, savanna, and alpine capture set.
- Canopy cover and height histograms within 5 percent of exact reference aggregation for each
  ecology test region.
- No floating crowns, submerged ordinary trunks, density pulse, or anchor movement during
  PREVIEW-to-FINAL and far-to-exact transitions.
- Flora blockage cannot reduce terrain-and-water coverage, protected exact readiness, or movement
  recovery.
- Settled radius-512 flora stays within an explicit CPU cache, GPU allocation, command-count, and
  shadow-update budget recorded in the performance report.

### Stable object attachments

Dynamic entities, block entities, collision, and gameplay state stay outside the immutable far
hierarchy. A later renderer may attach explicit, versioned visual proxies for stable landmarks such
as a beacon, torch flame, active furnace mouth, or other emissive source. Those proxies use their
own residency and lighting revision, preserve the exact material's emissive versus nonemissive
parts, and retire only when exact ownership is ready. They never become save authority, generate
collision, or justify scanning every distant block entity per frame.

## Program 5: equilibrium ecosystem authority

The learned pipeline does not emit biome IDs. The published model pack produces elevation and four
coarse climate variables; paper-compatible postprocessing derives lapse rate, so Rycraft's
authority exposes five climate fields: mean temperature, temperature variability, annual
precipitation, precipitation variability, and lapse rate. The pinned Minecraft mod converts a
subset of those outputs plus slope and fixed local noise into categorical labels with
`BiomeClassifier`. Generator v4 already makes all learned physical channels authoritative and feeds
them through Rycraft's physical-climate adapter. Replacing that temporary adapter, rather than
merely copying Minecraft labels, remains a separate generator change.

The plant-functional-type work adds continuous capacity for
tropical evergreen broadleaf, tropical drought-deciduous, temperate evergreen broadleaf, temperate
deciduous broadleaf, boreal evergreen needleleaf, boreal deciduous needleleaf, xeric shrub, cold
shrub, C3 grass, C4 grass, emergent wetland, aquatic vegetation, barren ground, and ice.

It derives growing-season length, aridity, water stress, productivity, leaf-area proxy, root-depth
class, fuel moisture, riparian capacity, aquatic capacity, and static hydroperiod from learned
climate, soil, and canonical water. It does not invent monthly climate phases.

The independent flora hierarchy consumes these fields as immutable capacity inputs. Existing biome names
remain material and diagnostic labels. This program requires its own generation revision,
continuity tests, visual matrix, performance report, and documentation.

Implementation begins with a native-grid crosswalk against the pinned mod classifier. The corpus
records its temperature, seasonality, precipitation, precipitation-variability, slope,
growing-season, aridity, snow, ocean, and final-label decisions beside Rycraft's physical fields.
This is a compatibility diagnostic, not a requirement to reproduce its fixed Perlin perturbations,
vanilla sea test, or categorical tree-density thresholds. Rycraft substitutes canonical ocean,
wetland, groundwater, shoreline, hydroperiod, and surface-slope authority, uses the derived lapse
rate, and computes continuous PFT fractions before deriving append-only biome labels. Golden tests
cover every classifier boundary, negative coordinates, native-page seams, elevation bands, cold
and arid extremes, and request-order independence. Flora, canopy, and fauna consume the continuous
fractions; a label change alone may not move an anchor, alter water, or change terrain.

Reference fixtures execute the classifier at commit
`23d3f50e5108882bb88a03c3ab048aa63633a02f` once. Ordinary CI checks recorded inputs,
intermediate decisions, and labels without a network or Java dependency.

### Ecosystem authority schema and competition

Before implementation, a decision record pins the scientific equilibrium formulation, parameter
table, calibration dataset, and exact source revisions. The Minecraft classifier remains a
compatibility fixture, not the scientific basis for productivity, competition, root depth, or fuel
moisture. A parameter change advances the ecosystem revision in the generation fingerprint.

Persist `ecosystem-authority-v1` pages on the native model grid. Each page records the complete
generation identity, ecosystem revision, source terrain and hydrology hashes, units, channel mask,
quantization table, payload size, and CRC-32. Quantized unsigned 16-bit PFT, barren, and ice
fractions sum to exactly 65,535 at every sample. Largest-remainder correction uses the fixed PFT
enum as its tie break, so platform and request order cannot change the result.

Competition proceeds in three explicit stages: environmental gates produce nonnegative raw
suitability; the pinned annual water, energy, soil, disturbance-free productivity, and hydroperiod
formulation produces resource-limited capacity; deterministic simplex normalization produces the
stored fractions. Ice and barren ground compete in the same normalization. Aquatic and emergent
fractions require canonical water or hydroperiod support, while ordinary terrestrial fractions are
zero under standing water. Supporting productivity, water-stress, leaf-area, root-depth, fuel,
riparian, and aquatic fields each receive documented units, closed ranges, quantization, and
missing-data rules.

Qualification covers the complete temperature, variability, precipitation, aridity, elevation,
soil, groundwater, and hydroperiod edge matrix; native-page seams; negative coordinates; reversed
and concurrent requests; cache-cleared rebuilds; and recorded regional aggregates from the pinned
validation dataset. It requires exact fraction-sum and habitat-mask invariants, byte-identical page
hashes, no seam jump beyond one quantization unit, and documented tolerances for regional PFT,
productivity, canopy, and fauna-capacity comparisons.

## Additional required iterations

### Cold entry and warm restart

- Plan the complete protected entry dependency graph before executing its first model window.
- Reuse finalized spawn metadata, persisted transient FINAL authority, hydrology summaries, and
  compiled Core ML products on a warm restart.
- Reuse valid hybrid-hierarchy pages and the last connected parent frontier on a warm restart.
  Reopening an unchanged world must not reconstruct the full far horizon or resubmit learned model
  windows merely because device-local meshes were evicted.
- Separate local file-stamp validation from full pack hashing. Only Repair may fetch or replace an
  installed model asset.
- Report the critical path for dry-land selection, exact certification, parent horizon, protected
  FINAL closure, collision halo, and first-publication lighting.
- Keep the 30-second first-entry and five-minute cold-settlement gates. If a build misses either
  gate, the pull request reports the miss and remains unqualified.

### LOD stability and quality

- Derive geometric error from canonical child deviation, water topology, silhouette variance,
  material frequency, and flora height variance.
- Preserve the current rule that an urgent camera-critical refinement bypasses the distant-parent
  worker reservation. Extend that priority through authority, hydrology, hierarchy construction,
  upload, and GPU residency rather than weakening it at a later stage.
- Use separate refine and coarsen thresholds, camera-velocity prediction, and minimum residency
  time. Hysteresis cannot be implemented by hiding a missing child.
- Reserve CPU and GPU capacity for the camera column, exploration band, complete required exact
  surface disk, visible error violations, and current transition dependencies in that order.
- Add a stationary test, hover test, high-speed flight test, water-edge test, and 180-degree camera
  reversal test. Each test records maximum visible error, LOD changes per second, parent fallback
  duration, and nearest coarse violation.

### Derived lighting and exact handoff

- First-visible exact meshes remain blocked on their bounded lighting transaction.
- Give saved-section manifest snapshots a monotonically increasing revision. Active-set work must
  reject or retry a bulk snapshot when an edit or save advances that revision before publication,
  so a late background read cannot replace newer sky authority.
- Far and aggregate flora representations receive stable exterior light, terrain shadow, and
  atmosphere terms without pretending that unloaded block light exists.
- Torches, active furnaces, lava, and other exact emissive sources retain their HDR, bloom,
  propagated block-light, and near-field indirect-light behavior across a far-to-exact handoff.
- Beds and other nonemissive partial models keep their authored geometry, collision, skylight, and
  indirect-light behavior.

### Disk and memory control

- Publish separate budgets for model/runtime memory, tensor windows, decoded authority, hydrology
  summaries, hybrid hierarchy pages, CPU geometry, GPU heaps, flora, water, and temporary compute
  output.
- Evict speculative and optional products before displayed parents, protected lineage, transition
  endpoints, exact fallbacks, or current camera-critical work.
- Add cache compaction and integrity tools. They may delete rebuildable derived cache pages but may
  not modify regions, metadata, edits, inventories, block entities, or fluid frontiers.
- Measure write amplification and disk growth over a fixed 100-kilometer exploration route.

## Qualification matrix

Every program adds ordinary fake-backend tests plus explicit real-model and rendering suites.

| Area | Required evidence |
|---|---|
| Determinism | Fresh, reverse, concurrent, cache-cleared, worker-count, and restart hashes |
| Coordinates | Negative origins, page edges, region edges, vertical extrema, and half-open bounds |
| Water | Every body type across hydrology owner, sparse region, exact cube, and LOD boundaries |
| LOD | Exact, handoff, and every far tier while stationary, moving, reversing, and hovering |
| Flora | Empty, sparse, dense, riparian, wetland, alpine, and forest-horizon scenes |
| Failure | Cancellation, stale epoch, corruption, allocation overflow, device loss, and retry |
| Performance | Entry, settlement, movement recovery, frame distribution, and memory |
| Metal | API validation, GPU validation, shader validation, counters, and opened captured frames |

Visual evidence must include the generation identity, seed, camera, drawable size, settings,
selected nodes, desired and displayed quality, worst projected error, exact coverage epoch, water
identity, queue state, and memory totals. A captured PNG that has not been opened and inspected is
not qualification evidence.

## Non-goals

This roadmap does not:

- Replace the learned model, its pinned scale, or the canonical generation identity.
- Restore synthetic continents, legacy hydraulic erosion, analytical crater lakes, retaining
  walls, bank dilation, or raster water repairs.
- Make exact cubic simulation follow the full 512-chunk visible horizon.
- Add far caves, far structures, editable far voxels, or far collision.
- Allow GPU floating-point results to become persisted authority without bit-identical
  qualification.
- Reduce native resolution, 4x MSAA, view distance, near exact ownership, water connectivity, or
  visual quality to make a performance number pass.
- Add dynamic succession, animated seasons, terrain-changing runoff, floods, fire spread,
  migration, predators, food webs, eruptions, or climate change.
- Make renderer output, cache state, traversal order, worker count, or camera history affect world
  generation.

## Definition of complete

The roadmap is complete only when:

1. The paged hybrid hierarchy is the sole far spatial selection structure and the current tile path has
   been removed after differential qualification.
2. Protected exact terrain and visible error violations cannot be starved by distant terrain,
   water, flora, prediction, or cache maintenance.
3. Canonical water and ecology remain continuous through every representation.
4. Cold entry, cold settlement, movement recovery, frame time, and memory meet the documented M4
   Max gates without qualification exceptions.
5. Metal validation is clean, every required capture has been opened and inspected, and automated
   regressions reproduce the formerly reported LOD, floating-water, flat-terrain, bare-flora,
   invisible-wall, and lighting failures.
6. Derived caches can be deleted and rebuilt without changing the world, its edits, or its
   fingerprinted authority.
