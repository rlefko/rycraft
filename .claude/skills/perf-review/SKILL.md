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

1. Exact simulation uses `min(viewDistance, 32)` and never follows the 512-chunk visible horizon.
2. The radius-six and vertical-four exploration band, visible saved edits, one primary section per visible column, additional exposed and cliff sections, and the targeted halo are represented. Reservation follows that order before either global cap. Mesh snapshots carry an 18 by 18 by 18 block, fluid, and block-light halo plus separate 18 by 18 sky cutoffs.
3. Loaded exact cubes cannot exceed 32,768. Obsolete cubes unload before replacement generation starts, and every insertion rechecks the cap while holding the chunk-map mutex.
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

1. Exact simulation has a nominal radius of 32, and immutable far-parent requests cover the complete visible disk through radius 512, including tiles wholly inside that nominal radius.
2. Every selected 256 by 256-block coordinate requests a step-32 parent before optional refinement. Missing parents submit and upload nearest-first through a broad parent lane. Each connected coordinate gives its distance-selected step-16, step-8, step-4, or step-2 target one bounded urgent lane before the complete parent disk is resident. Optional broad intermediates remain behind complete coverage. Resident active parents stay pinned.
3. Distance and immutable maximum slope and hydrology complexity select the desired refinement using bounded, tunable thresholds rather than rigid rings. Every far-owned fragment in the camera exploration band requires a block-scale step-2 fallback, while every other far-owned fragment in the exact overlap requires step 8 or finer. Protection includes a fully ready boundary tile whose published exact requirements cover only part of the tile. Its step-32 parent remains resident as a dependency but is never displayed. Protected fallback jobs bypass ordinary grace and topology-transition limits so these minimum tiers can publish as soon as they are ready.
4. The previous tier supplies asymmetric refine and coarsen thresholds. Production fixtures prove that filtered voxel tiers are not a nested height pyramid, so a narrow terrain-only fog pulse hides one atomic complete-topology swap for ordinary replacements. Canopies retain the full 0.65-second target-in, source-out exchange. Transition and coverage decisions use unswayed world coordinates. Bit-29 skirts follow the complete terrain topology currently visible, while water remains source-owned until completion.
5. At most 64 topology transitions are active. Exactly one complete terrain topology is visible outside the narrow fog pulse. The canopy exchange never passes through an empty phase, and generated water has exactly one owner throughout a transition.
6. Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks missing step-32 dependencies. The drawable frontier additionally treats a protected tile with only its parent ready as missing, hides tiles at or beyond the nearest such gap, and fades the preceding 256 blocks. A protected step-2 or step-8 fallback advances that frontier only when it satisfies the tile's minimum display tier. Partially faded patches never enter the occluder horizon.
7. One bounded canopy query runs per cold tile. Step 2 reuses accepted exact tree anchors, species, and dimensions without constructing ColumnPlans. Steps 4 through 32 use globally anchored 64-block aggregate cells with six fixed candidates and block-8 habitat and ground authority. The aggregate tiers form strict stable subsets, while the two-phase exchange safely handles the unrelated exact-anchor and aggregate representations. At step 32, block-resolution collector habitat and root-water authority wins over the coarse cell water bit, so unrelated water elsewhere in a 32 by 32 cell cannot suppress an accepted canopy. Its trunk grounds on the displayed voxel. Canopy work is cached with the tile and never runs per frame. Measure the current synchronous canopy stage separately: observed cold canopy work ranges from 250 to 1,165 milliseconds and delays terrain and water parent publication. Treat staged canopy attachment as follow-up debt.
8. Published exact requirements and unresolved columns produce one 256-bit ownership mask for each far tile, with one bit per 16 by 16-block chunk column. Missing and unresolved requirements keep the column far-owned; empty completed meshes count as ready. Every far-owned exact-overlap fragment remains protected, including fragments in a fully ready partial boundary tile. Terrain, water, and canopies use the destination-column bit, and the eight neighboring tile masks cover fragments that cross tile faces. LOD skirts use displayed-neighbor state plus ownership samples on both sides of their joins. Any partially masked horizon patch is not an occluder. The nearest-gap distance remains a conservative parent-selection fallback and diagnostic. Separately verify the existing exact missing-halo closures: lit planned surface continuations aboveground, dark inward caps underground, and bedrock caps vertically. Halo arrival must invalidate the affected mesh.
9. Four latency-sensitive and four utility far workers cap pending work at 64 and completed work at 32. While parents are queued, dispatch reserves four worker slots for base coverage and admits at most four urgent connected refinements. Already running base jobs finish without preemption. Exact streaming does not reduce this worker pool; it limits refinement uploads to four per frame while preserving up to 32 parent uploads. Otherwise up to 12 refinement uploads may advance. All far uploads share the 32 MiB frame cap. Record the combined active maximum across six exact-generation, four exact-mesh, and eight far workers because QoS assignment alone does not prove frame isolation on the 16-core reference machine.
10. The CPU cache caps at 9,280 entries and 3 GiB. Active parents are pinned, the farthest refinement evicts first, and every key rebuilds deterministically, including canopy geometry. Residency changes never scan or destroy the cache on the render thread. Utility-worker maintenance scans at most 64 records and retires at most 32 MiB per pass, with one oversized record allowed alone for progress.
11. One-, two-, four-, eight-, sixteen-, and thirty-two-block footprint sampling filters sub-Nyquist detail without changing hydrology, water, plate, or feature ownership. Step-32 coverage geometry uses conservative minima, and one weighted palette resolves per active LOD cell.
12. Far uploads cap at 32 parents, 12 refinements, and 32 MiB per frame, with the busy-state refinement cap described above.
13. The segmented far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertex storage and 1 GiB of index storage.
14. Frustum culling precedes conservative front-to-back 256-bin terrain-horizon culling.
15. Sixteen 64 by 64-block patches per visible tile contribute lower horizons with fixed storage, and LOD replacement keeps the last resident tile until the staged replacement uploads.
16. Opaque exact and far terrain uses outward counterclockwise winding and back-face culling.
17. Submission uses bounded direct indexed draws. Do not credit HZB, a literal geometry clipmap, indirect command buffers, or GPU-driven submission unless the implementation actually adds them.

The current branch retains one performance violation: terrain, water, and canopy geometry share one synchronous far build and residency payload. Measured cold canopy work ranges from 250 to 1,165 milliseconds, so an otherwise ready parent cannot publish until canopy discovery finishes. Record staged canopy attachment as follow-up debt and do not report two-second cold-horizon residency as passing from queue policy or headless tests alone. Do not report exact-face closure as absent: missing exact halos already emit explicit lit, dark, or bedrock caps and rebuild when real halo data arrives.

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

For each solver, identify domain size, apron, input grid, numerical spacing, fixed pass count, concurrency limit, stale-result behavior, validation, and fallback. The implemented basin contract is a 2,048-block catchment, a 16-block raster with a two-cell apron, globally aligned 64-block callback inputs, a four-cell shared-boundary blend with exact portal reconstruction, Priority-Flood, angular two-neighbor D-infinity-inspired routing, eight erosion and relaxation passes, Strahler ordering, and validated lake, waterfall, outlet-fall, and distributary outputs. Lake-depth interpolation retains dry and different-body contributors at zero depth rather than renormalizing the wet weights. Distinct overlapping lake authorities retain both flat levels behind a bounded supported watershed, except where an outlet or channel corridor owned by either body must remain open. Incoming and outgoing channel halves follow monotonic quintic junction-to-portal water profiles, while explicit falling state is limited to tagged drops. Crater lakes use a warped absolute local profile, validate all 96 rim directions with one block of freeboard, support the complete dry bank, and are rejected when a safe wet radius cannot fit. Invalid basin construction falls back to un-eroded base terrain with deterministic outlet metadata. Streaming admits at most two cold column-plan jobs at once, while one process-wide permit separately caps cold basin constructions across exact and far generators at two. Do not credit the routing as a verbatim Tarboton triangular-facet implementation.

Treat one `BasinSolver` as one immutable callback context. Elevation, rainfall, and rock-resistance callbacks must be coordinate-pure and describe the same fields for the solver's entire lifetime because the catchment cache key does not contain callback identity. A changed field requires a new solver, not reuse after an ordinary cache clear. Scalar and grid sampling must retain shared ownership of every candidate basin solution until authority selection and shoreline reconstruction finish. Concurrent clear or LRU eviction must not invalidate a referenced neighbor, and work started under an older cache generation must not become a current fast hit after clear.

Confirm each column plan retains nine full macro samples, a 256-column exact density surface grid, and compact 17 by 17 canonical water and lithology authority while making only 16 transient height-only perimeter queries. Water authority includes stable identity, level, depth, endorheic state, and ocean, river, lake, delta, waterfall, and supported-bank topology. Every standing column has a solid floor and a full-height source-water volume from the first wet voxel above that support through the top. Routed rapid and outlet stages carry their explicit flowing levels, and waterfall curtains carry falling state, without turning covered volume or receiving pools into flow. Ambiguous cells may use bounded exact hydrology or geology during cold construction, but ordinary cube and frame paths must reuse the retained authority. Confirm the 8,112-entry column-plan cache stays under its compile-time 128 MiB payload bound, the separate basin cache stays within its byte-accounted 64 MiB single-flight LRU bound, shoreline pages stay within their separate byte-accounted 64 MiB single-flight cache, the shared macro-control cache stays within 1,024 entries and 128 MiB, the far-climate cache stays within 1,024 entries and 8 MiB, and the far mesh cache stays within 9,280 entries and 3 GiB. Shoreline pages must be keyed by water body and global 256-block page, reconstruct broad controls at four-block spacing with shared aprons, and refine only their narrow contour band at two-block spacing. F3 reports exact readiness, the conservative nearest-gap distance, base and refinement residency, queue lanes, the drawable coverage frontier, and cache use. The inspector reports footprint, grid-artifact, material, hydrology, and cache metrics, including active, peak, and throttled cold-build counters. Cache payload bounds do not replace measurement of transient construction storage and allocator overhead.

For tree generation, bound the global feature-cell fanout and local-priority comparisons. Habitat must remain coordinate-pure while consuming continuous biome suitability, temperature, precipitation, soil moisture and fertility, light, slope, emitted altitude, lithology, tectonic stress, hydrology, and ecotopes. Dense suitable forests may raise acceptance and tighten deterministic spacing, but dry, steep, barren, geothermal, and actively volcanic ground must suppress it. Ordinary species reject standing generated water. Only suitable mangroves through depth three and non-ocean willows through depth two may root while submerged, and their emitted trunks must replace intervening source water down to a rechecked solid floor without adding fluid work.

Run or cite determinism tests that clear caches between equivalent requests. Eviction may change latency but never bytes, fluid states, plans, or feature anchors.

Treat procedural field continuity as a separate acceptance concern from exact-to-far residency. The current categorical former-line check passes with a longest run of 9 blocks against the 24-block limit. The preserved continuous-field case is an explicitly deferred expected failure with 15 failing assertions: terrain derivative-energy ratios reach 0.105649 at 2,048 blocks and 0.076197 at 8,192 blocks, aggregate shoreline energy is 0.194688, shoreline structured orientation is 1.674842 against the 1.5 limit, and biome suitability fails multiple spacings. Run `./build-release/tests/test_rycraft "[.known-continuity-debt]"` when reviewing world fields. Report the result as deferred debt, never as a passing gate or a render-residency failure.

## 8. Audit queues and runtime fluids

For generation, mesh, save, fluid, and completion queues, verify producer cap, consumer budget, deduplication or coalescing, overflow behavior, and diagnostics.

Six exact generation workers split into four latency-sensitive and two utility workers. Confirm the pump submits at most seven cube tasks, six running plus one look-ahead, beneath the 64-job hard ceiling while remaining active-set work stays in the prioritized backlog. At most two cold plans may run. A worker must skip a cube that is stale for current retention, and completion processing must requeue a still-required cube through its current plan dependencies. Confirm queued work retains active-set epoch, gameplay lane, and distance priority so the camera column and six-chunk exploration band cannot wait behind the broad exact disk or a previous camera position. Four exact mesh workers permit 64 total items across queued, building, completed, and renderer-pending states, with 32 slots reserved from broad admission for later camera-band work, coalescing by requested cube revision, and 64 uploads or 32 MiB per frame. Confirm a completion can publish only when its build revision matches the live cube and is newer than the resident mesh. Rejecting or failing a result must clear only its matching request, while coalescing must retain the completion for the newest request even when that snapshot failed. Confirm the observed exact-mesh queue high-water never exceeds 64. Far terrain uses four latency-sensitive and four utility workers and permits 64 pending jobs and 32 completed results. While parents are queued, confirm four slots remain reserved for base work and no more than four urgent connected refinements run. It uploads at most 32 parents, 12 refinements, or 32 MiB per frame, with the refinement count reduced to four while exact streaming is busy. Confirm stale far epochs are discarded rather than published after a large move.

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
- Stable source and flowing cells emit planar top geometry only; the vertex path does not displace the source plane, analytic fragment shading supplies motion, and vertical sides belong only to explicit falling columns
- Every generated standing body is a full-height implicit source-water volume from the first wet voxel above solid support through the surface, including across cube faces, without an explicit fluid array until runtime disturbance changes a cell
- Routed rapid and outlet stages retain their explicit eighth-block flowing levels, and waterfall curtains retain falling state, while covered volume and receivers remain sources
- Far partially wet cells use body-aware contour-clipped shorelines rather than rectangular sheets or joins between incompatible water authorities
- Seed 42 at X=-557, Z=379 preserves direct, footprint, column-plan, solid-floor, and full implicit-source-volume agreement
- A lake `OutletFall` retains independent top, bottom, width, flow, and receiver-anchor data; exact generation emits one short receiver-centered footprint and the owning half-open far tile emits one five-quad prism without raising the receiving water
- Generated outlet falls enqueue zero fluid ticks, like every other generated water body
- Far generated source water uses the exact implicit full-block source plane

Flag any pending-update or frontier drop count missing from F3. Confirm save and mesh coalescing remain observable through their statistics interfaces and regression coverage.

## 9. Audit locks and main-thread work

Place every new mutex in the documented order. Inspect the complete critical section, including called functions, for generation, plan construction, solver work, I/O, compression, waits, or allocation-heavy operations. `SaveManager::manifestWriteMutex_` precedes the short-lived `manifestMutex_` in lock order. The inner lookup lock may copy or publish manifest state, but it must be released before file I/O while the outer writer lock serializes replacement. The process-wide cold-basin permit mutex is a nonnested leaf, and the basin-cache mutex must be released before a caller waits on that permit or an existing future.

Confirm ordinary cube generation and mesh construction run on workers. The existing near-camera edit rebuild is limited to already-meshed cubes and two builds per frame. First-time streaming must not enter that synchronous path. The steady far render loop must reuse its reserved candidate, request, key, cache-result, upload, and 4,096-entry flat grace-record buffers plus fixed tier counters. Ordinary unprotected replacements may use the grace buffer, but protected exact-loading fallbacks must bypass grace and transition-cap admission. Flag a progressive scheduler change that allocates per tile or during ordinary command encoding, while distinguishing one-time renderer-reset capacity growth from recurring frame work.

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

Acceptance evidence uses an optimized build on an identified Apple M4 Max at native resolution, 4x MSAA, and `RYCRAFT_VIEW_DISTANCE=512`. Begin with seed 764891 at spawn `23029,225,-111726`, yaw 0, and pitch -17, then use the documented autopilot interval to exercise a repeatable moving route. After streaming settles, require the lowest sustained one-second rate to remain at or above 60 FPS with total unified-memory use at or below 64 GB. Record the exact M4 Max configuration, macOS version, display resolution, seed, and route. Run Metal validation separately because validation overhead does not belong in the performance measurement.

At the same settled camera, record playing and paused intervals and attribute fixed-tick CPU p50, p95, and maximum with a profiler or scoped measurement. Pausing may remove simulation work, but it must not conceal a material main-thread bottleneck or produce an unexplained frame-rate multiplication.

Apple Silicon memory counters can overlap. Report process RSS, Metal allocation, and resident counters separately, then state the highest credible unified-memory total without adding overlapping counters. Hardware timings and memory are evidence, not portable CI gates. CI should enforce deterministic work limits, queue caps, cache bounds, and allocation invariants.

When texture sampling changed, confirm the single block-texture array eagerly owns all five 16-to-1 mip levels, alpha-aware downsampling preserves representable cutout coverage, and the sampler uses nearest magnification with linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy. Account for the complete mip allocation once, not per tile or frame.

When canopy LOD changed, measure the accepted exact-anchor count at step 2, globally anchored 64-block aggregate cell count at steps 4 through 32, six-candidate habitat fanout, impostor quad count, retained bytes, and build time. Confirm aggregate tiers are strict stable subsets, exact anchors retain their species and dimensions, cell ownership is deterministic across split queries, the exchange never exposes an empty forest, and all canopy work remains on far workers under the existing tile queue and cache bounds. At step 32, verify that the exact collector's habitat and root-water decision survives unrelated water elsewhere in the coarse cell and that the trunk grounds on displayed terrain. Measure terrain and water parent work separately from the synchronous 250-to-1,165-millisecond cold canopy stage. Treat staged canopy attachment as follow-up debt even when the combined mesh remains within its byte budget.

## 11. Report

Output in this order:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, rule, structural cost, player impact, and compliant fix, ordered by impact
3. **Risks worth a look:** uncertain or measurement-dependent items
4. **Confirmed clean:** only checklist areas exercised by the diff
5. **Evidence:** commands, deterministic route, measurements, and limits observed

Keep findings actionable and do not restate the whole diff.
