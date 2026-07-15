---
name: perf-review
description: Review a rycraft diff against frame and tick budgets, sparse cubic active-set limits, adaptive far-terrain LOD and culling bounds, mipmapped texture cost, unified-memory limits, cache and solver bounds, queue backpressure, lock discipline, deterministic generation, fluid work and geometry, and fauna caps. Use before committing changes to rendering, simulation, world generation, cubic or far streaming, meshing, persistence, fluids, entities, caches, worker pools, or mutexes. Reads docs/performance-conventions.md as the source of truth.
---

# Performance Review

Review the requested change against rycraft's documented performance contract. Report concrete structural costs and violations. Do not accept an unmeasured timing claim or invent a violation outside the changed paths.

## 1. Read the source of truth

Read `docs/performance-conventions.md` completely. Its current budgets and checklist override summaries in this skill.

## 2. Establish the review scope

Use the target named by the user. Otherwise inspect both committed and uncommitted work:

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
git diff HEAD
```

If the diff is empty, report that and stop. If no frame, tick, generation, streaming, mesh, fluid, entity, cache, persistence, or lock path changed, state that the performance checklist is not in scope and stop.

## 3. Map changed work to frequency and bounds

For every changed path, record:

- Thread: render, fixed tick, generation worker, mesh worker, save thread, or audio callback
- Frequency: per frame, per tick, per cube boundary, per cold plan, per block, or one-time
- Fanout: active cubes, halo cubes, far tiles, columns, lattice samples, fluid cells, entities, or draw calls
- Bound: hard cap, eviction rule, fixed iteration count, radius, or queue budget

Express cost as calls times frequency times fanout. A new operation once per block across 4,096 cells and thousands of cubes is not a constant-cost change.

## 4. Run the mechanical sweeps

Inspect added allocation and lock sites:

```bash
git diff origin/main...HEAD | rg '^\+.*(std::string|ostringstream|new |make_shared|make_unique|resize\(|reserve\()'
git diff HEAD | rg '^\+.*(std::string|ostringstream|new |make_shared|make_unique|resize\(|reserve\()'
git diff origin/main...HEAD | rg '^\+.*(lock_guard|unique_lock|scoped_lock|\.wait\(|\.get\()'
git diff HEAD | rg '^\+.*(lock_guard|unique_lock|scoped_lock|\.wait\(|\.get\()'
git diff origin/main...HEAD | rg '^\+.*(for |while |unordered_map|unordered_set|queue|vector)'
git diff HEAD | rg '^\+.*(for |while |unordered_map|unordered_set|queue|vector)'
```

Classify each hit by its execution path. An allocation during startup is not a hot-path finding. An allocation in density, fluid-neighbor, mesh-candidate, or per-frame code is.

## 5. Audit the cubic active set

Verify all of the following when streaming or rendering changes:

1. Exact simulation uses `min(viewDistance, 32)` and never follows the 256-chunk visible horizon.
2. Surface sections, the radius-six and vertical-four exploration band, visible saved edits, and the targeted halo are represented. The exploration band is the highest-priority mesh and retention class before either global cap. Mesh snapshots carry an 18 by 18 by 18 block, fluid, and block-light halo plus separate 18 by 18 sky cutoffs.
3. Loaded exact cubes cannot exceed 32,768.
4. Exact mesh candidates and renderer mesh residency cannot exceed 16,384.
5. Unload hysteresis retains existing cubes through two extra horizontal chunks and one extra vertical cube.
6. Priority includes visibility, collision, edited state, and full three-dimensional distance.
7. A view-distance scan does not multiply by the complete 40-section vertical range.
8. Active-set reconstruction does not perform cold plan construction, hydrology solving, or file I/O on the render thread.
9. Missing collision remains closed. Ray targeting and edits use non-loading lookups and stop at an unavailable cube instead of turning interaction into synchronous generation.
10. A compact loaded-section mask proves vertical continuity before an underground mesh receives skylight. A missing cardinal neighbor follows the generated terrain cutoff: air above it, opaque below it, a lit surface-material continuation when visible above ground, and a dark inward cap only for an underground opening. Halo residency changes invalidate every affected mesh.
11. One rebuild gathers unique horizontal columns and expands the fixed plan apron once. It does not repeat the complete apron query for every surface or halo cube.
12. Pending plans use indexed active-set dependencies, and completion notifications coalesce before rebuild. A completed plan does not scan all retained cubes or schedule one rebuild per dependent.
13. Saved edited sections for the unique visible-column list come from one bulk in-memory manifest query under a short lock, with no file I/O.
14. Gameplay submits a latest-wins request rather than rebuilding on the fixed tick. A utility-priority planner checks for stale epochs between expensive phases and before publication. Plan completions batch at 128 results or backlog drain with a four-tick consumption cooldown.
15. Request, coalescing, cancellation, and build-time metrics make planner churn visible in performance logs.

State the worst-case counts found in code. A constant named `MAX_MESH_RESIDENT_CUBES` does not prove compliance if the renderer's cache can grow around it.

## 6. Audit the far-terrain horizon

When view distance, culling, LOD, far meshing, or draw submission changes, confirm:

1. Exact simulation stops at radius 32 and immutable far tiles alone fill the half-open annulus `[32, 256)`.
2. Tiles cover 256 by 256 blocks. A narrow two-block sampling tier immediately outside radius 32 samples exact emitted density heights as the topology bridge.
3. Farther out, distance and immutable maximum slope and hydrology complexity select among four-, eight-, and sixteen-block tiers using bounded, tunable thresholds rather than rigid rings.
4. The previous tier supplies asymmetric refine and coarsen thresholds, and a bounded 0.4-second fog transition hides topology replacement after the target is resident.
5. At most 64 topology transitions are active and only one tier per coordinate is drawn at once.
6. The two-block topology tier compares directly with exact surface samples. Globally aligned borders, greedy merging, and skirts prevent cracks without multiplying exact cubes. Partially wet cells use contour-clipped shoreline triangles.
7. One bounded canopy query runs per cold tile. Steps two and four reconstruct every exact accepted tree anchor. Steps eight and sixteen use globally anchored 32- or 64-block aggregate forest cells, with coordinate-pure climate, substrate, slope, water, and acceptance tests. Canopy work is cached with the tile and never runs per frame.
8. Canopy vertices use bit 28 for classification. Depth-biased far opaque tops remain as lit fallback until resident exact depth replaces them. Water and canopies retain exact ownership through radius 32, then use a stable world-space dither over the following 16 blocks without a second mesh or draw list. Bit-29 skirts are visible only on a finer edge next to a resident coarser tile outside that band.
9. Four workers cap pending work at 64 and completed work at 32.
10. The CPU cache caps at 1,024 entries and 512 MiB with deterministic rebuild after eviction, including canopy geometry.
11. Far uploads cap at 12 tiles and 32 MiB per frame.
12. Far GPU arenas remain 256 MiB for vertices and 128 MiB for indices.
13. Frustum culling precedes conservative front-to-back 256-bin terrain-horizon culling.
14. Sixteen 64 by 64-block patches per visible tile contribute lower horizons with fixed storage, and LOD replacement keeps the last resident tile until the new one uploads.
15. Opaque exact and far terrain uses outward counterclockwise winding and back-face culling.
16. Submission uses bounded direct indexed draws. Do not credit HZB, a literal geometry clipmap, indirect command buffers, or GPU-driven submission unless the implementation actually adds them.

State wanted, resident, drawn, frustum-culled, horizon-culled, pending, cache-byte, and far-arena maxima. A full circular resident set is intentional; an unbounded GPU registry is not.

## 7. Audit caches and solvers

For each changed macro, basin, plan, feature, or renderer cache, identify:

- Key and value size
- Entry or byte cap
- Eviction policy
- Single-flight behavior for duplicate cold requests
- Construction thread
- Lock scope
- Result behavior after eviction

For each solver, identify domain size, apron, input grid, numerical spacing, fixed pass count, concurrency limit, stale-result behavior, validation, and fallback. The implemented basin contract is a 2,048-block catchment, a 16-block raster with a two-cell apron, globally aligned 64-block callback inputs, a four-cell shared-boundary blend with exact portal reconstruction, Priority-Flood, angular two-neighbor D-infinity-inspired routing, eight erosion and relaxation passes, Strahler ordering, and validated lake, waterfall, outlet-fall, and distributary outputs. Lake-depth interpolation retains dry and different-body contributors at zero depth rather than renormalizing the wet weights. Finished lake geometry reconstructs its floor from one flat level and positive depth, then supports the dry shore while leaving named outlets and active channels open. Crater lakes use a warped absolute local profile, validate all 96 rim directions with one block of freeboard, support the complete dry bank, and are rejected when a safe wet radius cannot fit. Invalid basin construction falls back to un-eroded base terrain with deterministic outlet metadata. Streaming admits at most two cold column-plan jobs at once, while one process-wide permit separately caps cold basin constructions across exact and far generators at two. Do not credit the routing as a verbatim Tarboton triangular-facet implementation.

Treat one `BasinSolver` as one immutable callback context. Elevation, rainfall, and rock-resistance callbacks must be coordinate-pure and describe the same fields for the solver's entire lifetime because the catchment cache key does not contain callback identity. A changed field requires a new solver, not reuse after an ordinary cache clear.

Confirm each column plan retains nine full macro samples, a 256-column exact density surface grid, and a compact 17 by 17 canonical lake authority while making only 16 transient height-only perimeter queries. Ambiguous lake cells may use bounded exact hydrology during cold construction, but ordinary cube and frame paths must reuse the retained authority. Confirm the 8,112-entry column-plan cache stays under its compile-time 64 MiB payload bound, the separate basin cache stays within its byte-accounted 64 MiB single-flight LRU bound, and the far cache stays within both 1,024 entries and 512 MiB. F3 aggregates macro entries and MiB while reporting far metrics separately; the inspector reports column-plan and basin metrics separately, including active, peak, and throttled cold-build counters. Cache payload bounds do not replace measurement of transient construction storage and allocator overhead.

Run or cite determinism tests that clear caches between equivalent requests. Eviction may change latency but never bytes, fluid states, plans, or feature anchors.

## 8. Audit queues and runtime fluids

For generation, mesh, save, fluid, and completion queues, verify producer cap, consumer budget, deduplication or coalescing, overflow behavior, and diagnostics.

Exact generation permits 64 in-flight cube jobs and at most two cold plans. Exact meshing permits 64 total items across queued, building, completed, and renderer-pending states, with coalescing by cube revision, and 64 uploads or 32 MiB per frame. Confirm its observed queue high-water never exceeds 64. Far terrain permits 64 pending jobs, 32 completed results, and 12 uploads or 32 MiB per frame. Confirm stale far epochs are discarded rather than published after a large move.

The save queue permits at most 32,768 unique cubic positions. Repeated snapshots for one already queued position replace its pending snapshot with the newest revision and increment an observable coalescing count. A unique producer at the cap waits for backpressure. Bulk manifest reads lock the in-memory index once for the requested columns, while manifest serialization and file replacement use a separate writer lock outside the lookup lock.

For water specifically, confirm:

- Five-tick delay on the 20 Hz clock
- At most 1,024 processed cells per fluid tick
- At most 65,536 pending updates and 65,536 deferred frontiers
- At most eight catch-up ticks
- Generation and ordinary loading enqueue zero work
- Reads never force-load a cube
- Missing boundaries create only activated frontiers
- Deferred frontiers are indexed by unavailable destination cube, and a fixed resume budget touches only matching newly available buckets rather than scanning all 65,536 entries for each cube load
- Restoring or loading a frontier does not activate unrelated generated water
- Stable source and flowing cells emit top geometry only; vertical sides belong only to explicit falling columns
- Every generated standing water voxel from the supported floor through the surface is an implicit source, including across cube faces, without an explicit fluid array until runtime disturbance changes a cell
- Far partially wet cells use contour-clipped shorelines rather than rectangular sheets
- A lake `OutletFall` retains independent top, bottom, width, flow, and receiver-anchor data; exact generation emits one short receiver-centered footprint and the owning half-open far tile emits one five-quad prism without raising the receiving water
- Generated outlet falls enqueue zero fluid ticks, like every other generated water body
- Far generated source water uses the exact implicit source plane 0.875 blocks above its voxel floor

Flag any pending-update or frontier drop count missing from F3. Confirm save and mesh coalescing remain observable through their statistics interfaces and regression coverage.

## 9. Audit locks and main-thread work

Place every new mutex in the documented order. Inspect the complete critical section, including called functions, for generation, plan construction, solver work, I/O, compression, waits, or allocation-heavy operations. `SaveManager::manifestWriteMutex_` precedes the short-lived `manifestMutex_` in lock order. The inner lookup lock may copy or publish manifest state, but it must be released before file I/O while the outer writer lock serializes replacement. The process-wide cold-basin permit mutex is a nonnested leaf, and the basin-cache mutex must be released before a caller waits on that permit or an existing future.

Confirm ordinary cube generation and mesh construction run on workers. The existing near-camera edit rebuild is limited to already-meshed cubes and two builds per frame. First-time streaming must not enter that synchronous path.

## 10. Measure when the claim requires it

Build and run the fixed-seed diagnostic or playtest route when the diff makes a speed, memory, queue, or frame-time claim. Record:

- Build type and commit
- Machine and macOS version
- Seed, spawn, route, and view distance
- Frame p50 and p95
- Lowest sustained one-second frame rate
- Warm cube p95
- Cold macro or basin p95 when applicable
- Peak process RSS and peak Metal allocated or resident memory
- Queue settle time and maximum queue sizes
- Loaded and mesh-resident maxima
- Far wanted, resident, drawn, frustum-culled, horizon-culled, pending, cache, and arena maxima

Acceptance evidence uses an optimized build on an identified Apple M4 Max at native resolution, 4x MSAA, and `RYCRAFT_VIEW_DISTANCE=256`. Begin with seed 764891 at spawn `23029,225,-111726`, yaw 0, and pitch -17, then use the documented autopilot interval to exercise a repeatable moving route. After streaming settles, require the lowest sustained one-second rate to remain at or above 60 FPS with total unified-memory use at or below 64 GB. Record the exact M4 Max configuration, macOS version, display resolution, seed, and route. Run Metal validation separately because validation overhead does not belong in the performance measurement.

At the same settled camera, record playing and paused intervals and attribute fixed-tick CPU p50, p95, and maximum with a profiler or scoped measurement. Pausing may remove simulation work, but it must not conceal a material main-thread bottleneck or produce an unexplained frame-rate multiplication.

Apple Silicon memory counters can overlap. Report process RSS, Metal allocation, and resident counters separately, then state the highest credible unified-memory total without adding overlapping counters. Hardware timings and memory are evidence, not portable CI gates. CI should enforce deterministic work limits, queue caps, cache bounds, and allocation invariants.

When texture sampling changed, confirm the single block-texture array eagerly owns all five 16-to-1 mip levels, alpha-aware downsampling preserves representable cutout coverage, and the sampler uses nearest magnification with linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy. Account for the complete mip allocation once, not per tile or frame.

When canopy LOD changed, measure exact anchor-query and priority-comparison fanout at steps 2 and 4, aggregate cell count at steps 8 and 16, impostor quad count, retained bytes, and build time. Confirm cell ownership is deterministic across split queries and that all canopy work remains on far workers under the existing tile queue and cache bounds.

## 11. Report

Output in this order:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, rule, structural cost, player impact, and compliant fix, ordered by impact
3. **Risks worth a look:** uncertain or measurement-dependent items
4. **Confirmed clean:** only checklist areas exercised by the diff
5. **Evidence:** commands, deterministic route, measurements, and limits observed

Keep findings actionable and do not restate the whole diff.
