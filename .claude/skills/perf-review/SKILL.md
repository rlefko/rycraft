---
name: perf-review
description: Review a Rycraft diff against generator v4 learned-authority, bootstrap, cubic streaming, canonical water, far terrain, canopy staging, smooth lighting, physical atmosphere and weather, queue, cache, frame-time, and unified-memory budgets. Use before committing changes to generation, inference, hydrology, rendering, persistence, workers, queues, caches, or locks. Reads docs/performance-conventions.md as the source of truth.
---

# Performance Review

Review the requested change against Rycraft's documented performance contract. Report structural costs and measured evidence separately. Never treat a queue cap or fake-backend test as proof of a real-model performance gate.

## 1. Read the source of truth

Read `docs/performance-conventions.md` completely. Read `docs/generator-v4-follow-up.md` when a
change touches the deferred hierarchy, GPU construction, distant water, or distant flora. Also
read the changed generator, runtime, hydrology, far-terrain, and persistence interfaces before
interpreting the diff.

## 2. Establish scope

Use the user's target. Otherwise inspect committed and uncommitted changes:

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
git diff HEAD
```

If no frame, tick, inference, generation, hydrology, streaming, mesh, save, cache, queue, or lock path changed, state that performance review is not in scope.

## 3. Classify every cost

For each changed path, record:

- Thread: bootstrap, inference, generation, far worker, mesh worker, render, fixed tick, save, or audio
- Frequency: startup, per model call, per page, per cube, per tile, per frame, per tick, or per block
- Fanout: windows, tensors, pages, samples, hydrology cells, cubes, tiles, entities, or draws
- Hard bound: queue cap, cache bytes, iteration count, radius, worker count, or upload budget

A page allocation once per cold inference is different from a vector allocation per block or per frame. Show the multiplication.

## 4. Run mechanical sweeps

```bash
git diff origin/main...HEAD | rg '^\+.*(new |make_shared|make_unique|resize\(|reserve\(|vector|unordered_)'
git diff HEAD | rg '^\+.*(new |make_shared|make_unique|resize\(|reserve\(|vector|unordered_)'
git diff origin/main...HEAD | rg '^\+.*(lock_guard|unique_lock|scoped_lock|\.wait\(|\.get\()'
git diff HEAD | rg '^\+.*(lock_guard|unique_lock|scoped_lock|\.wait\(|\.get\()'
git diff origin/main...HEAD | rg '^\+.*(for |while |inference|queryNative|preparePage)'
git diff HEAD | rg '^\+.*(for |while |inference|queryNative|preparePage)'
```

Classify every hit by execution path. Do not report startup-only allocation as frame-path allocation.

## 5. Audit bootstrap and inference

Confirm:

1. Download, verification, extraction, compilation, and qualification occur before `SaveManager`, `World`, and worker construction.
2. Model files and Core ML caches stay under `~/Library/Application Support/rycraft`, outside Git and workspaces.
3. The model remains pinned to revision `ad2df557eca5645f588766101cf3bc3682455c3e`, Coarse remains scalar, Base and Decoder remain static batches of four, Decoder uses deterministic repeated-tail padding, the production window and batching contract remains compatible with the InfiniteDiffusion paper and pinned Minecraft reference commit `23d3f50e5108882bb88a03c3ab048aa63633a02f`, and ONNX Runtime remains pinned to 1.27.1 and loaded through `dlopen`.
4. Core ML requires static shapes, ML Program, all compute units, model caching, and sequential execution.
5. At most one model call is active. The coarse, base, and decoder sessions remain resident after compile, with no more than three residents. The provider identity uses `coreml-cache-v3-base4-decoder4x256` and qualification digest `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df`. Queue, failure, partition, CPU fallback, configured intra-op thread, and resident-session metrics remain observable.
6. CPU fallback uses at most `min(hw.physicalcpu, 16)` intra-op threads, including the calling thread. It must not use inter-op graph parallelism to bypass the one-call rule.
7. Render and fixed-tick threads neither execute nor wait on inference.
8. The canonical qualification hash matches the pinned recorded digest, and any new runtime or provider configuration is requalified before a real-model result is called passing.
9. The unsigned 64-bit seed and full identity reach every runtime, authority, and world constructor without narrowing.
10. A normal launch and Retry reuse a verified installed pack without a network transfer. Repair alone may replace a failed asset, and a missing completion marker is restored locally after all assets verify.
11. Preparation logs attribute coarse, Base, and decoder calls to qualification, dry-spawn search,
    final-spawn certification, horizon preview, and protected FINAL handoff. A finalized warm profile
    does not repeat dry-spawn calls or protected transient reconstruction.
12. Spawn and protected transient FINAL rectangles use fingerprinted, checksummed atomic
    persistence. Corrupt payloads repair through inference, while identity mismatches fail closed.

## 6. Audit learned-authority bounds

Confirm each cache and queue against the source of truth:

- At most 64 outstanding page requests
- One page build at a time
- At most 64 pages and 1,048,576 samples per query
- At most 1,024 decoded entries and 512 MiB
- At most 384 MiB if a tensor-window cache is present
- Equal cold page requests share a single flight
- Inference, compression, `fsync`, and file I/O occur outside the cache mutex

Protected FINAL hydrology owners may be grouped lexicographically in sets of at most two by two,
provided every combined half-open rectangle remains within the 1,048,576-sample query bound. Require
each 517 by 517 owner crop to match an independent request exactly, including negative coordinates
and shared aprons. A lower fake-executor call count is useful structural evidence, not real-model
latency evidence.

Identify whether the production backend actually implements coarse, latent, and decoder windows, fixed batch four, reconstruction, and cache reuse. The deterministic fake backend is correctness infrastructure, not production throughput evidence.

Confirm that v4 learned elevation and climate remain the sole macro authority. Legacy `BasinSolver` hydraulic erosion, alpine postprocessing, and the analytical crater-lake overlay are v3-only. V4 volcanic primitives enter physical elevation before native hydrology, and the separate post-routing dry residual stays within 1.5 blocks and is zero for water, channels, outlets, lake rims, coasts, divides, transition owners, and uncleared slopes. Treat any topology-changing residual or v4 call into legacy erosion as a correctness and performance violation.

The PR 1 ecology boundary remains the physical-climate adapter feeding existing biome, flora, canopy, and fauna consumers. Plant-functional-type equilibrium belongs to PR 2 and must not be described as a completed PR 1 performance path.

## 7. Audit exact cubic work

Confirm:

1. Exact simulation reaches `min(viewDistance, 32)` after entry. A v4 cold start uses a zero
   nominal exact radius with the mandatory one-chunk active halo and four-chunk plan-dependency
   footprint. During preparation it advances exact mesh publication, connected coarse terrain and
   canonical water through 96 chunks, and the protected FINAL closure. The protected lane begins
   when the connected parent frontier reaches the near band. The configured horizon stays selected
   and fills after entry without exposing a farther island.
2. Loaded cubes stay at or below 32,768 and exact mesh residency at or below 16,384.
3. Six generation workers submit no more than seven cube jobs at once.
4. Every required surface section through the full 32-chunk exact disk stays ahead of optional flora
   and broad primary work in generation submission, mesh admission, completed-result upload, and
   deferred first-publication lighting. The camera column ranks first, the exploration band second,
   and the remaining required disk third.
5. The 96-section range does not become a full vertical scan for every visible column.
6. `VerticalSectionMask` checks both words for sky and loaded-range queries.
7. The 193 density levels are lazy and limited to the requested cube and surface neighborhood.
8. Exact collision accepts loaded block and fluid authority only from a renderer-published section
   whose epoch matches the active exact coverage. Unowned planned sections use canonical generated
   terrain and fluid proxies, unresolved sections stay closed, and raycasts do not force-load.
9. A learned failure freezes new generation instead of publishing an empty cube.
10. World teardown stops admission and drains its uniquely owned generation pool before releasing futures; queued work may not retain pool ownership or make a worker join itself.
11. AppKit termination completes persistence, then joins render, far, canopy, and world workers,
    releases generation contexts, and finally destroys the inference runtime. A capture or
    performance run that writes results and then exits by signal is a failed run.

State worst-case counts from code, not only constant names.

## 8. Audit canonical hydrology

For every changed solver, identify domain size, spacing, apron, fixed passes, queue behavior, cache key, single flight, construction thread, and edge reconciliation.

Reject:

- Whole-world or unbounded upstream walks
- Query-rectangle-dependent water results
- A dry-terrain raising pass
- Wet-route deletion to resolve conflicts
- Unbounded stage relaxation or flood queues
- Scalar and batch paths that use different authority

At 7.5 meters per block, one 2,048-block hydrology page is 15.36 kilometers wide and 235.9296 square kilometers. Use that scale for rainfall, runoff, and discharge calculations.

Preview and final routers must share one process-wide native-hydrology admission gate. Confirm the total is at most 16 concurrent page builds, additionally constrained by hardware concurrency and the one-GiB scratch reservation, rather than 16 builds per router.

Treat `NativeHydrologyCacheMetrics::deferredBuilds` as a monotonic count of completed cache build
attempts that returned typed `DEFERRED`. It is not a parked-work count, an active-build gauge, or a
failure count. Interval logs subtract the preceding snapshot for `deferredBuilds` and report
`activeBuilds` directly.

Connected wetland traversal must stop at 64 owner pages or 262,144 native cells, cache resolved and rejected native nodes, and avoid running for a cell without the persisted candidate flag. Estuary backwater must stop at 64 native cells and 64 owner pages, reject falls and steep channels, and cache its source result. Dense grid sampling may invoke either resolver only for matching candidate or channel cells; one low-gradient reach must not disable the direct immutable-page path for every dry point in the page.

## 9. Audit far terrain and canopy staging

Confirm:

1. Every selected coordinate requests a step-32 parent.
2. A valid parent remains drawable until a connected replacement is resident.
3. Sixteen terrain workers cap pending work at 64 and completed results at 32. After the connected
   96-chunk prefix, exact publication through 32 chunks or any connected visible desired-LOD debt
   pauses ordinary outer-parent submission and publication. Local far work admits 8 workers
   alongside exact debt, 12 after exact debt clears, and all 16 only after exact and local debt
   clear. Exact debt without a local far miss admits no ordinary far work. Canopy uses zero workers
   during preparation and until the connected prefix is drawable. Gameplay then guarantees exactly
   one low-priority canopy worker, including while stronger debt continues. No second gameplay
   canopy lane opens. Missing PREVIEW attachments remain ahead of FINAL promotion.
4. Missing base work reserves four admitted workers. No more than twelve urgent refinements run when
   the complete 16-worker budget is available.
5. Surface-stage terrain, standing water, and falls publish before vegetation enrichment.
6. A blocked or canceled flora callback cannot consume the only in-flight slot indefinitely or invalidate resident surface geometry.
7. Displayed PREVIEW and FINAL surfaces publish nearest-first PREVIEW ecology before any FINAL ecology promotion, grounded against their current surface authority.
8. Camera movement refreshes every queued, parked, and follow-up flora priority, and near work can displace the least-important queued or parked request at capacity.
9. Nearby flora gets one upload opportunity after protected and urgent local terrain, and broad refinements cannot consume its GPU residency floor.
10. Step-32 dry corners do not suppress a cell with `waterTopologyPossible`.
11. Production meshes have `skirtQuadCount == 0`.
12. Displayed neighbors, including active replacement endpoints, remain within a 2:1 step ratio.
13. Four canonical boundary strips add 516 samples per production tile, using four batched calls when available.
14. Transition payload stays within the fixed 65,024-byte per-tile test bound before genuine interior terrain-discontinuity faces.
15. Exact cubes own the first 32 chunks. Settled far bands are step 2 through 64 chunks, step 4 through 128, step 8 through 256, and step 16 through 512. These are maximum-coarseness limits. Gameplay retains finer tiers above the 0.55-pixel projected-error target, including FINAL step 1 when required, and coarsens outward below 0.45 pixels. Step 1 is the irreducible voxel-grid floor. Step 32 remains a coverage parent. Cold entry schedules only the required protected FINAL refinements; ordinary perceptual refinements remain closed.
16. Except for the mandatory step-32 coverage parent, a displayed parent and child use matching authority quality or the child matches a retained parent source during promotion. Base-lineage PREVIEW steps 16, 8, 4, and 2 may reduce visible cell size while FINAL is cold, but their projected-error debt includes the measured revision-9 omitted residual. PREVIEW relief and vertical scheduling bounds include the measured 46-block maximum. Four urgent slots each serve ready FINAL children, proxy bridges, and visible FINAL parents, with unused child capacity loaned to proxies. Coarse preview leaves sixteen learned-authority admissions available for visible or protected FINAL work. Same-key promotion retains its preview source while a visible preview child depends on it. Exact ownership stays gated until stable FINAL authority. Matched water topology switches atomically. Treat a topology-changing per-tile promotion as unresolved until one complete hydrology owner can be perimeter-checked and published together.
17. CPU terrain and canopy caches stay at or below 24,576 entries each, 3 GiB for terrain, and 512 MiB for canopy.
18. GPU storage stays at or below 2 GiB of vertices and 1 GiB of indices, with the documented coverage and flora floors.
19. During cold entry, preparation polling, base scheduling, result drains, and shared-buffer publication do not depend on a full exact-world scene draw.
20. Entry uses the connected parent frontier, not the full configured parent count. At a 512-chunk view it requires the step-32 parent prefix through 96 chunks and the complete camera-aware protected FINAL closure. The closure begins during preparation when the connected frontier reaches the near band. It contains 4, 8, 12, 16, and 20 targets at steps 1, 2, 4, 8, and 16, for 60 total. Test all four camera-within-tile corners and reject any protected diagonal drawn beyond the missing-parent frontier.
21. Authority requests preserve the production priority order: spawn, exploration exact,
    protected exact handoff, visible final refinement, coarse preview, then speculative
    movement prefetch. Final parents in the exact-handoff prefix use the protected lane. Current
    protected FINAL children and parents receive urgent capacity before bounded PREVIEW bridge
    prerequisites, unused bridge capacity returns to current FINAL, and directional prediction uses
    only the remaining CPU-only capacity.
22. Movement prefetch starts only after the current visible preview closure is ready, remains
    outside that closure, and admits at most eight pages beyond the leading horizon.
23. A standard or unknown exception from a critical far-base or final mesh latches a
    retriable generation failure, exposes recovery through the existing bootstrap UI, and
    leaves any valid resident parent drawable.
24. Each refinement request advances one adjacent tier. Camera and protected work rank first,
    followed by connected-wavefront eligibility. Within the nearby visible class, smaller
    horizontal distance ranks before greater projected error. Screen-space error chooses desired
    quality, but it cannot let a farther high-error tile delay closer missing detail. A finer cached
    result cannot starve a missing bridge.
25. Camera-near terrain is a capacity-protected class. At saturation an urgent protected PREVIEW
    parent may displace the worst queued or parked ordinary coverage request, while a near bridge may
    displace noncritical refinement work. An urgent camera-critical refinement bypasses the nominal
    four-worker reservation for an unrelated distant parent, while ordinary urgent work observes
    that reservation. Camera-critical work sheds protected status immediately after camera movement.
    Distant work cannot evict either job or CPU cache entry.
26. Every connected visible desired-LOD miss, not only a protected target, is local debt. It runs
    nearest-first and may displace a queued or dependency-parked outer parent. It may not evict a
    displayed parent, the connected 96-chunk prefix, a transition endpoint, exact fallback, active
    protected lineage, or a requested critical key. While this debt or exact publication debt
    remains, ordinary outer-parent submission and publication stay paused.
27. A requested protected FINAL role-selected key may reclaim optional non-displayed distant
    refinement or flora residency from both CPU and GPU storage, and it may use the complete GPU
    arena. Alternate LODs at the same coordinate do not inherit that class. Never reclaim coverage,
    a displayed surface, either transition endpoint, exact fallback, active protected lineage, or a
    requested critical key. Distant work cannot evict the critical pending upload or allocation, and
    the old drawable parent remains resident until commit.
28. Broad and medium exact plan or cube submissions preserve physical capacity for a newer camera
    epoch. At exact mesh saturation, a better near request displaces only the worst queued request,
    and camera movement cancels queued meshes outside the new immutable candidate snapshot.
29. Deferred first-publication lighting reranks in camera, exploration, required-surface, flora, and
    broad order. One synchronous transaction performs no more than 32 floods, overflow remains in a
    bounded queue, and no exact mesh publishes until its revision-matched transaction has settled.

Measure surface-stage and vegetation-enrichment time separately.

When the deferred paged hierarchy is in scope, also confirm:

- The primary structure is a signed, surface-first page forest with sparse volumetric bricks only
  for cubic exceptions. It is not a global pointer SVO or a dense voxel payload at every level.
- Parent aggregates are deterministic products of canonical children and carry conservative
  geometric, silhouette, water, flora, lighting, and emissive error summaries.
- Selection projects canonical error into pixels, uses asymmetric refine and coarsen thresholds,
  and retains the parent until the complete child terrain, transition, and water family is GPU
  resident. Screen-space error chooses desired quality, while the nearby visible scheduler ranks
  horizontal distance before projected error.
- Water and flora retain separate semantic caches, queues, memory budgets, and residency. Generic
  material or representative-voxel reduction cannot erase a wet route or distant canopy.
- GPU traversal, request feedback, indirect commands, and eviction remain bounded and retain a CPU
  reference path. Record hierarchy bytes, page faults, selected nodes, request overflow, command
  count, and benefit-per-byte eviction decisions.

## 10. Audit persistence and locks

RYTA verification, compression, `fsync`, and rename stay off render and fixed-tick threads. Page corruption repair may reinfer only under the same fingerprint. A fingerprint mismatch cannot be repaired in place.

The inference mutex is a leaf. Authority-cache locks are released before model calls and I/O. The world chunk-map mutex may not contain generation, hydrology, inference, compression, or waits.

For gameplay rendering, confirm that torch-flame and active-furnace-mouth emission reaches derived block light, emissive radiance in resolved HDR, bloom, and the Hi-Z indirect-light input without a per-block frame scan. The surface MRT supplies diffuse albedo and ambient accessibility, not emitted radiance. Confirm lava is fully emissive and that beds, inactive furnaces, furnace shells and tops, and torch sticks remain nonemissive.

Measure `indirectPrepare`, `indirectTrace`, `indirectTemporal`, `indirectAtrous`, `indirectApply`, and `resolvedFog` separately. High quality must remain half resolution with four rays, a 24-iteration Hi-Z cap, and three a-trous passes. Medium must remain quarter resolution with two rays, a 16-iteration cap, and two a-trous passes. The retained compatibility Boolean must map false to off and true to high quality. Report persistent payload bytes for high, medium, and off. Confirm opaque fog runs after tracing without an extra persistent HDR target, and clouds and weather particles remain in the single-sample resolved post-indirect pass with no surface attachment.

Audit history invalidation as bounded state, not a publication counter. Authoritative resize, teleport, FOV, quality, world, session, direct-light source, prior-depth, resident-light-edit, and time discontinuities reset it. Far refinement, preview-to-final replacement, canopy attachment, and exact-to-far handoff preserve it.

## 11. Audit lighting, atmosphere, and weather cost

For packed light, record reconciled cubes, the 4,096-cell per-cube flood bound, queue budget, changed-face fanout, changed meshes, and materialized bytes. Confirm stable cubes do not requeue neighbors, derived light is never serialized, player-edit convergence stays inside its documented bounded neighborhood, and fluid writes remain on the deferred budget.

For shadows, account for the two-slice near array, two-slice far array, and horizon texture. High resolutions are 4,096, 4,096, 2,048, 2,048, and 2,048. Medium resolutions are 2,048, 2,048, 1,024, 1,024, and 1,024. Refresh intervals remain bounded at one, one, two, four, and eight frames, with early refresh only for snapped projection, caster revision, or light direction. Record refreshed cascades and caster draw counts instead of treating every allocated target as every-frame work.

For screen-space lighting, account for the full-resolution min-depth pyramid and normal guide, reduced trace and denoise targets, color and depth histories, and moments-and-age histories. There is no radiance pyramid. High remains half resolution with four rays, a 24-iteration Hi-Z cap, and three a-trous passes. Medium remains quarter resolution with two rays, a 16-iteration cap, and two passes. Off retains ambient application. Target allocation occurs only on construction, resize, or quality change, and every history reset is constant-time.

For weather, prove one utility worker, one running request, and at most one latest pending request. Each build has two fixed 81 by 81 slices, recentering starts only after 1,024 blocks, and readers retain the previous immutable snapshot without waiting. Record requests, coalescing, starts, publications, stale discards, pending count, and busy state. Lightning queries return no more than eight events. Thunder retains no more than 16 pending events and 64 remembered IDs.

For dry-spawn screening, require at most one ranked proposal and one directly prepared 2,048-block owner per aligned hydrology owner. Check the requested chunk first, then use bounded globally aligned four-block native-raster batches across at most 16 workers. A canonically proven center must atomically install its complete 5 by 5 native safety certificate, or 25 dry samples at four-block spacing. A positive-elevation continental fallback may install no certificate and begin only radius-zero exact validation. Confirm that it does not start the far horizon, report dry land as located, or publish metadata before exact support, headroom, slope, water-absence, and nearby-dry validation pass. Record coarse selection, owner preparation, native samples, canonical proof calls, provisional fallbacks, exact rejections, and total dry-spawn latency. Reject any normal-entry path that prepares neighboring semantic topology owners or the wider exact band before world construction. A complete wide prequeue remains legal only for explicit repair or qualification.

For atmosphere, count the fixed 256 by 64, 32 by 32, and 192 by 108 LUTs and distinguish slow optical refreshes from sky-view refreshes. For clouds, count the 128-cubed base noise, 32-cubed erosion noise, curl map, weather slices, quarter-resolution histories, hit-depth histories, and 2,048 or 1,024-square cloud shadow. Confirm noise generation runs on its cancellable utility worker, that quality Off starts no worker or allocation, and that the render thread uploads only a completed payload for the current world instance and seed. Force deterministic noise failures and confirm that 100 and 500 millisecond backoff plus the third-failure latch prevent a per-frame worker storm while preserving one diagnostic record per attempt. High uses 48 view steps and 6 light steps; Medium uses 24 and 3. Wind updates remain constant work in physical blocks per second.

For froxels, account for the 160 by 104 by 64 volume, integration target, half-resolution result, and ping-pong histories. Disabling volumetric lighting skips froxel injection and integration while retaining LUT aerial perspective. Underwater gating may not add another air-medium march.

Recalculate all persistent High-tier Metal resources for the tested drawable and require no more than 768 MiB. Keep credible unified-memory use below 64 GiB without adding overlapping RSS and Metal counters. If the route misses its frame target, follow the documented reduction order for cloud steps, froxel resolution, horizon-cascade work, then screen-space work. Do not reduce detailed shadow reach, the first two High shadow target resolutions, lighting-term separation, or packed-light correctness.

## 12. Run portable evidence

```bash
meson setup build --buildtype=debugoptimized
ninja -C build tests/test_rycraft
ninja -C build test
./build/tests/test_rycraft "[learned]"
./build/tests/test_rycraft "[bootstrap]"
./build/tests/test_rycraft "[reported-water-continuity]"
./build/tests/test_rycraft "[render][indirect]"
./build/tests/test_rycraft "[render][textures][emissive]"
./build/tests/test_rycraft "[render][mesher][bed]"
```

Report exact commands and results. Ordinary CI must not download the model pack.

## 13. Measure hardware gates when claimed

Use a release build on the documented M4 Max, native resolution, 4x MSAA, view distance 512, seed 764891, spawn `23029,225,-111726`, yaw 0, and pitch -17.

Record first entry, five-minute settlement, frame p50 and p95, lowest sustained one-second frame rate, movement recovery, authority lookup p95 and maximum on render and fixed-tick threads, inference queues, all streaming queues, the five indirect-light GPU timers, indirect persistent payload, RSS, Metal memory, and highest credible unified-memory total. Measure high, medium, and off separately without changing the acceptance resolution, MSAA, or view distance.

Require at least 60 FPS, p95 at most 16.67 ms, no sustained generation interval above 20 ms, lookup p95 below 0.25 ms and maximum below 1 ms, and total unified memory at or below 64 GiB.

Run Metal validation separately. Do not report an unrun or failed real-model gate as passing.

## 14. Report

Output:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, rule, structural cost, player impact, and fix
3. **Measurement-required risks:** uncertain items and the exact needed evidence
4. **Confirmed clean:** only areas exercised by the diff
5. **Evidence:** commands, hashes, route, measurements, and observed bounds

Keep findings actionable and do not restate the diff.
