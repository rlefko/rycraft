---
name: render-review
description: Review a rycraft rendering diff for C++ and MSL layout parity, HDR pass compatibility, cubic and far-tile origins, 18-cube-edge halos, six-face seams, partial water, adaptive far LOD, mipmapped textures, frustum and conservative horizon culling, back-face winding, residency, unified-memory limits, and Metal validation evidence. Use before committing changes under src/render, include/render, shaders, mesh snapshots, fluid rendering, far terrain, textures, GPU-shared structs, or Metal call sites. Reads docs/rendering-conventions.md as the source of truth.
---

# Render Review

Review the requested change against rycraft's rendering contract. Treat missing runtime evidence as a finding for player-visible work. Do not manufacture a finding for a rule the diff does not exercise.

## 1. Read the source of truth

Read `docs/rendering-conventions.md` completely. Its current rules and checklist override summaries in this skill.

## 2. Establish the review scope

Use the target named by the user. Otherwise inspect:

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
git diff HEAD
```

Stop with a short explanation if the diff is empty or has no rendering, shader, mesh, snapshot, fluid-geometry, texture, cubic-origin, or Metal surface.

## 3. Check shared layouts mechanically

Run:

```bash
git diff origin/main...HEAD -- '*.metal' '*.hpp' '*.h' '*.mm' '*.cpp' | rg '^\+.*struct '
git diff HEAD -- '*.metal' '*.hpp' '*.h' '*.mm' '*.cpp' | rg '^\+.*struct '
rg -n 'struct (Uniforms|ChunkOrigin|WaterUniforms|CloudUniforms|GPUParticle|ParticleUniforms|BloomUniforms)' include shaders src tests
```

Any type read by both C++ and MSL must be declared once in `include/render/shader_types.hpp`. Verify `sizeof` and `offsetof` assertions in `tests/test_render.mm`, and verify every buffer length uses `sizeof`.

Confirm `Vertex` remains 16 bytes. Bits 0 through 23 retain face, texture, skylight, ambient occlusion, block light, emissive, and sway semantics. Fluid direction stays in bits 24 through 26, falling stays in bit 27, far-canopy impostors use bit 28, far-terrain boundary skirts use bit 29, and bits 30 through 31 remain reserved unless the entire vertex contract, descriptor, shaders, tests, and memory budgets are deliberately revised together.

## 4. Check pipelines, attachments, and storage

For every changed pipeline, list the pass that binds it and verify sample count, color formats, depth format, blending, and cull mode. Confirm fullscreen sampling orientation.

For every changed texture or buffer, record its size source, storage mode, lifetime, and write synchronization. MSAA targets are memoryless, CPU-written data is shared, and per-frame data cannot be overwritten before GPU execution.

For block textures, require one complete five-level chain from 16 by 16 through 1 by 1, deterministic alpha-aware downsampling, and representable alpha-cutout coverage preservation. The bound sampler must use nearest magnification, linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy.

## 5. Check cubic coordinate restoration

Trace one cube from `ChunkPos` to vertices and draw:

1. Mesh vertices stay in local 0 through 16 coordinates on X, Y, and Z.
2. `ChunkOrigin` restores `chunkX * 16`, `chunkY * 16`, and `chunkZ * 16`.
3. The origin is bound for opaque and water draws.
4. AABBs span the correct cube Y section.
5. Frustum tests, candidate distance, edit-near-camera checks, and water sort distance include Y.
6. Mesh caches use full `ChunkPos`, not a packed X/Z key.

Flag any world-coordinate half-precision vertex or a zeroed cube Y origin.

Trace one far tile separately:

1. The CPU key uses 64-bit tile X and Z plus its sample step.
2. X and Z vertices remain local to the 256 by 256-block tile.
3. The per-draw origin restores the tile's world X and Z once.
4. Sampled-surface bounds carry real minimum and maximum Y for visibility.
5. Far water receives the same origin and three-dimensional sort key.

## 6. Check the mesh halo and all six seams

Trace every accessor used during meshing. The 18 by 18 by 18 snapshot must provide blocks, fluid state, and block light for each possible coordinate. Skylight uses a separate 18 by 18 array of per-column sky cutoffs derived from exact density surfaces plus loaded opaque feature and edit overrides.

Verify tests or add review findings for:

- Negative and positive X solid-face culling
- Negative and positive Y solid-face culling
- Negative and positive Z solid-face culling
- Transparent blocks and flora on each boundary
- Skylight continuity across every face
- Edit invalidation of every face, edge, or corner neighbor whose halo intersects the edit
- Partial water top, explicit falling-side, and water-to-water continuity across every face
- Vertical world floor and ceiling halo behavior
- Missing in-range neighbors on all six cardinal faces, including lit sky-connected lateral caps, dark enclosed lateral caps, and vertical bedrock caps
- Generated overhangs, added and removed opaque roofs, the valid world-ceiling cutoff, and the distinct incomplete-path marker
- Vertical loaded-path continuity before an underground cube receives skylight

Corner-smoothed water samples X/Z diagonals. Confirm that loaded diagonals provide symmetric real data, missing diagonals use the same conservative opaque fallback, and an edge or corner edit or halo load transition invalidates every affected mesh.

## 7. Check partial water geometry

Verify source, levels 1 through 7, falling state, full-height water with water above, corner averaging, horizontal flow bits, bottom visibility, and culling against adjacent water. Stable source and flowing cells emit planar top geometry only. Their vertex path must not displace the source plane. Filtered analytic fragment normals and caustics may animate shading without changing geometry. Vertical sides are exclusive to explicit falling columns and must not appear at lake, river, ocean, delta, cube, or unloaded boundaries.

Trace generated water occupancy back to the `ColumnPlan` 17 by 17 canonical authority. It must retain stable body identity, level, depth, endorheic state, and ocean, river, lake, delta, waterfall, and supported-bank topology. Dry contributors must taper lake depth to zero without wet-weight renormalization, and every exact shore-water voxel must have a solid support below its lowest water cell. Every standing generated wet voxel from that lowest supported water cell through the surface must resolve as an implicit source, including across cube faces, without an explicit fluid array. A rendered top without canonical occupancy, full-depth source water, or support is a floating sheet even if its side faces were removed.

Where two lake authorities overlap, require a bounded supported competitive watershed that preserves both flat levels. The divider must yield to an outlet or channel corridor owned by either body, so a valid rapid or waterfall remains connected. Channel water must follow monotonic junction-to-portal profiles unless a tagged analytical drop supplies explicit falling state.

For a crater lake, require a coordinate-warped irregular shoreline, a complete dry enclosing rim with at least one block of freeboard, supported banks, and rejection when a safe wet radius does not fit. A routed nonendorheic lake may instead keep its named outlet; do not close that outlet merely to satisfy the rim check.

Review a lake `OutletFall` separately from both standing bodies. Its sample retains top, bottom, width, flow, and one receiver anchor while the lower sample keeps its own `waterSurface`. Exact generation must emit only a short, narrow receiver-centered falling footprint. The anchor's half-open far tile must own exactly one complete five-quad prism, with four sides and one top, even across a tile face. No neighbor may duplicate it, no long slab may replace it, and neither path may raise the receiving water or enqueue a generation-time fluid tick.

Compare renderer height with `fluidSurfaceHeight`, buoyancy, and camera-submersion logic. Sub-block coordinates must be binary-exact over the 0 through 16 cube-local range. Current eighth-block levels and 1/32-block corner averages satisfy that rule.

For generated source water, compare exact and far geometry at the same coordinate. Both must place the visible top one block above the water voxel floor.

Confirm water still renders after scene resolve, samples a color copy and resolved depth, binds no depth attachment, sorts back to front in three dimensions, and preserves underwater overlay ordering.

Trace resolved-depth reconstruction at large coordinates. Water surface, opaque floor, refracted floor, screen-space reflection, and underwater positions must share a camera-relative world frame whose view matrix omits camera translation. Absolute camera position may be added only after thickness math to anchor caustics and fog. Require a focused round-trip test beyond 100,000 blocks and across both sides of a cubic chunk face. Reject an absolute inverse view-projection for water depth because its precision loss appears as 16-block absorption and brightness tiles.

## 8. Check far LOD, winding, and visibility

For far terrain, verify:

- Exact editable cubes have a nominal radius of 32, while every 256 by 256-block tile intersecting the radius-512 visible disk requests a far step-32 parent before optional refinement, including tiles wholly inside the nominal exact disk.
- Missing parents use a broad nearest-first job and upload lane. Each connected coordinate gives its distance-selected step-16, step-8, step-4, or step-2 target one bounded urgent lane before the complete parent disk is resident. While base work remains, eight-worker dispatch reserves four slots for parents and admits no more than four connected refinements. Already running parents finish without preemption. Resident active parents stay pinned while their coordinates are visible, and pressure evicts the farthest refinement first.
- Parent residency and drawable coverage use separate connected frontiers. The drawable frontier treats a protected exact-loading tile with only step 32 ready as missing, suppresses tiles at or beyond the nearest such gap, and fades the preceding 256 blocks. Every far-owned fragment in the exact overlap stays protected, including fragments in a fully ready tile with only partial boundary requirements. It advances only after the camera exploration band has step 2 and every other protected overlap tile has step 8 or finer. Partially faded patches cannot become occluders.
- Published exact requirements and unresolved columns define a 256-bit ownership mask for each far tile, with one bit per 16 by 16-block chunk column. Missing and unresolved requirements keep a column far-owned, while empty completed meshes count as ready. A previously published exact mesh may retain ownership while an ordinary replacement is pending. The nearest-gap distance is only a conservative fallback and diagnostic. Separately verify the implemented exact missing-halo closure: lit planned surface continuations aboveground, dark inward caps underground, and bedrock caps vertically, followed by invalidation when real halo data arrives.
- Direct tests compare one-, two-, four-, eight-, sixteen-, and thirty-two-block footprint samples. Hydrology topology, water-body identity, water elevation, plate ownership, and feature anchors remain fixed across footprints.
- A far-owned fragment in the camera exploration band requires a block-scale step-2 fallback, and every other far-owned fragment in the exact overlap requires step 8 or finer. This includes fully ready partial boundary tiles. Step 32 remains resident but is not displayable for either protected class. Protected fallback uploads bypass ordinary grace and the 64-transition admission cap, so their first acceptable topology appears directly when ready. Ordinary replacements still use asymmetric refine and coarsen thresholds and no more than 64 simultaneous transitions. Production fixtures prove the filtered voxel tiers are not nested, so a narrow terrain-only fog pulse hides each ordinary atomic topology swap. Canopies retain a full 0.65-second target-in, source-out exchange. Transition and coverage ownership use unswayed world coordinates. Skirts follow the complete terrain topology currently visible, while water remains source-owned until completion.
- Adjacent equal-LOD tiles reproduce identical border heights, mixed-LOD transitions use bit-29-marked skirts on the finer edge of a resident coarser neighbor with displayed-neighbor and paired ownership checks, and greedy merging does not change sampled shape. Same-LOD and absent-neighbor edges carry no visible skirt. Do not confuse an LOD skirt with the explicit lit, dark, or bedrock closure cap already emitted for a missing exact halo.
- `FarSurfaceSample` carries filtered terrain and water, conservative footprint bounds, and a four-entry material palette. Step-32 coverage uses conservative minima, material resolves once per active LOD cell, and no aligned 32- or 64-block material cache remains.
- Partially wet far cells use body-aware contour-clipped shoreline triangles. A cell observing incompatible standing-water authorities refines from canonical samples instead of joining their levels. Stable far water emits top geometry only and never rectangular vertical sheets.
- Lake outlet falls remain separate receiver-centered five-quad prisms owned by one half-open anchor tile. Their bottom overlaps the lower body's top source voxel, their top reaches the upper lip, and their bounds remain narrow in the flow direction.
- Step 2 reuses accepted exact tree anchors, species, and dimensions without constructing ColumnPlans. Steps 4 through 32 query globally anchored 64-block aggregate forest cells with six fixed candidates and block-8 habitat and ground authority. The aggregate tiers form strict stable subsets. At step 32, the collector's block-resolution habitat and root-water decision wins over water elsewhere in the 32 by 32 cell, and the accepted trunk grounds on the displayed voxel. The two-phase canopy exchange safely handles the unrelated exact-anchor and aggregate representations without an empty phase.
- One half-open tile owns each grounded trunk-and-crown aggregate, even when the crown crosses a tile face. Every canopy vertex carries bit 28 for classification. Water, canopies, and terrain use destination-column exact ownership, and each draw binds the eight neighboring tile masks so crossing crowns and waterfalls query the correct columns. Opaque step-32 tops remain depth-biased behind resident exact surfaces only where the coordinate has no protected far-owned overlap fragment. Measure terrain and water parent work separately from the synchronous canopy stage, whose observed cold cost ranges from 250 to 1,165 milliseconds. Record staged canopy attachment as follow-up debt.
- Exact cube faces and far opaque faces are outward counterclockwise and use back-face culling.
- Cross and flat flora emit both windings; water and shadow casters stay cull-none.
- Tile AABB frustum culling runs before front-to-back 256-bin horizon culling.
- Sixteen 64 by 64-block patches per visible tile contribute conservative lower horizons without per-tile heap allocation. Any patch intersecting a revision-ready exact column is partially masked and cannot become an occluder.
- The horizon test uses only fully covered azimuth bins, selects distance extrema correctly above and below the camera, and cannot hide a taller or partially covered tile.
- A coordinate retains its old resident LOD until the selected replacement uploads successfully. One complete terrain topology and its skirts swap beneath the narrow fog pulse, while the target canopy becomes established before the source canopy retires. Source water retains ownership until transition completion.
- Exact opaque terrain draws before positively depth-biased far tops, which remain available as cold-residency fallback. Water has no depth attachment, so water and canopies use the same per-column fragment masks as opaque far terrain.
- Submission uses bounded direct indexed draws. Do not report hierarchical Z, a literal geometry clipmap, an indirect command buffer, or GPU-driven submission unless it exists in code.

The current branch has one known far-residency performance debt: terrain, water, and canopies publish as one synchronous payload. Measured cold canopy work ranges from 250 to 1,165 milliseconds, so an otherwise ready parent waits for canopy discovery. Record staged canopy attachment as follow-up work and require reference-route timing. Do not report exact-face closure as deferred: missing exact halos already emit explicit lit, dark, or bedrock caps and rebuild when real halo data arrives.

Require focused CPU tests for full-disk parent selection, four-parent and four-progressive worker reservation, connected 16/8/4/2 selected-target scheduling before full parent completion, step-2 protection for every far-owned exploration-band fragment, step-8 protection throughout the remaining exact overlap including fully ready partial boundary tiles, protected-job grace and transition-cap bypass, active-parent eviction protection, separate parent and drawable frontiers, drawable-frontier suppression and fade, 16 by 16 per-column exact ownership under missing, stale, capped, unresolved, and empty-completed requirements, 3 by 3 neighbor-mask crossings, paired skirt-mask evaluation, non-occluding partial masks, conservative distance fallback, footprint topology invariants through step 32, production nonnesting fixtures, atomic terrain topology selection, ordinary fog-pulse timing, unswayed reveal coordinates, conservative parent bounds, palette selection, identical per-column ownership for water and canopies, protected step-32 suppression, exact full-block source-water agreement, full-depth implicit generated sources across cube faces, canonical ocean, river, lake, delta, waterfall, and bank authority, supported shore occupancy, distinct-body far refinement, competitive lake watersheds, owned connector exemptions, monotonic junction-to-portal profiles, the seed-42 X=-557, Z=379 regression, complete caldera rim and freeboard, exact and far outlet-fall agreement, five-quad half-open fall ownership, exact-anchor step-2 canopies, strict aggregate subsets at steps 4 through 32, step-32 collector-authority preservation beside water, displayed-voxel trunk grounding, overlap-safe two-phase canopy exchange, stable species silhouettes, bit-28 canopy coverage, bit-29 skirt coverage, displayed-neighbor edge masks, bounded threshold selection, complexity shifts, asymmetric hysteresis, transition cap, single-owner transition water, negative tile coordinates, deterministic hashes, same-LOD borders, mixed-LOD skirts, body-aware contour-clipped shorelines, outward winding, scheduler caps, exact priority, epoch cancellation, and conservative ridge and peak cases. Missing-halo tests must prove lit aboveground continuations, dark underground caps, vertical bedrock caps, and halo-arrival invalidation. Measure terrain and water parent publication separately from the synchronous canopy stage. Inspect the opaque and water fragment paths for their shared ownership mask and distinct shading contracts.

## 9. Check residency and frame work

Confirm exact mesh residency never exceeds 16,384 and loaded halo cubes never exceed 32,768. Confirm the far CPU cache stays within 9,280 entries and 3 GiB. Confirm the segmented far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs and remains within 2 GiB of vertices and 1 GiB of indices. Verify unload and movement sweeps defer arena frees until the referencing frame completes. All exact, far, post-target, world, and transient use must fit the 64 GB unified-memory acceptance limit.

Inspect first-time exact builds, edit rebuilds, far builds, and uploads for their worker or main-thread path. Exact generation uses four latency-sensitive and two utility workers and submits at most seven cube tasks, six running plus one look-ahead, beneath the 64-job hard ceiling. Stale retained-set jobs are skipped, and still-required completions reenter through current plan dependencies. Four exact mesh workers permit 64 total queued, building, completed, or renderer-pending items and 64 uploads or 32 MiB per frame. Broad work may occupy at most 32 scheduler items before the camera-band reserve, and queued camera-band meshes outrank broad surfaces. Far terrain uses four latency-sensitive and four utility workers and permits 64 pending and 32 completed results. While parents are queued, four worker slots remain reserved for base coverage and at most four urgent connected refinements run. Exact streaming leaves all eight construction workers available but limits refinement uploads to four per frame while preserving 32 parent uploads. Otherwise up to 12 refinement uploads may advance within the 32 MiB frame cap. Confirm the steady render path reuses reserved progressive request, key, cache-result, upload, and 4,096-entry flat grace-record buffers plus fixed tier counters, with no per-tile heap allocation during command encoding after renderer reset.

## 10. Run validation and inspect captures

Use the `playtest` workflow. At minimum:

```bash
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MTL_SHADER_VALIDATION=1
```

For cubic changes, require representative captures from above terrain, below terrain, across a horizontal seam, and across a top or bottom cube face. Capture active aboveground streaming and reject any full black loading panel; every protected far-owned overlap fragment must receive lit step-2 or step-8 fallback instead of visible step-32 terrain. Underground captures must include lateral and vertical travel through the hard-priority exploration band, dark closed temporary openings, no skylight through a missing section above, and no bright void at a missing boundary below. Expose an oblique underground wall across a cube face, a macro-control boundary, and a plate contact. Ordinary strata must retain curved dip and variable thickness across all three; only a tagged fault may make a sharp offset. For water changes, include stable lake, river, ocean, and delta edges with no vertical walls or incompatible level joins, planar source tops without vertex displacement, supported full-depth occupancy, explicit falling columns with sides, the seed-42 source-volume regression at X=-557, Z=379, the seed-42 lake lip at X=-8235, Z=2976, the separate incised river face at X=-12288, Z=2653, the canyon at X=-23904, Z=0, a receiver-centered lake-to-river outlet fall joining both standing levels without a slab or gap, exact-to-far full-block source-water agreement, body-aware far contour-clipped shorelines, and an underwater view. Capture the seed-764891 caldera at X=23029, Z=-111486 around its complete perimeter and from water level, verifying an irregular supported bank, one block of freeboard, and water filling every voxel from crater floor through its flat surface. For far-terrain changes, capture cold startup, a rapid camera jump, both parent and drawable coverage frontiers, connected 16/8/4/2 targets appearing before the full parent disk completes, step-2 fallback for every far-owned exploration-band fragment, step-8-or-finer fallback for every other far-owned exact-overlap fragment, a fully ready partial boundary tile, a mixed ready-and-incomplete exact tile under active loading, crossing canopies and waterfalls at tile faces, direct agreement across every footprint through step 32, ordinary step-32 coverage outside the protected overlap, a forest spanning the exact overlap and every tier, exact-anchor step-2 trees, strict aggregate forest cover at steps 4 through 32, gradual refinement over both flat and complex terrain, refine and coarsen hysteresis crossings, an atomic terrain replacement beneath its narrow fog pulse, the full two-phase canopy exchange, the radius-512 horizon, a ridge occluding lower ground while preserving a taller peak, and a turn that reveals already resident tiles. Verify that no protected far-owned overlap fragment displays step 32, a protected base-only tile blocks the drawable frontier, latched exact columns do not flicker during halo remeshing, foliage sway does not move transition or coverage ownership, and a partially masked patch cannot act as an occluder. Verify lit, dark, and bedrock exact closure caps while halos are missing, then confirm arrival removes the cap. Reject disconnected islands, any same-LOD or absent-neighbor skirt panel, and any orphan skirt left after only one joined column becomes exact-owned. For texture changes, inspect distant textured slopes and alpha-cutout flora for aliasing, shimmer, coverage loss, and visible mip transitions. World-generation rendering should also capture former macro boundaries, curved shorelines and channels, lithology transitions, exposed strata, volcanic interiors, crater water, aquifers, or distributary channels when relevant.

For tree-rendering changes, capture dense suitable forest, ordinary dry ground beside standing water, a shallow mangrove, and a shallow non-ocean willow. Ordinary trunks must not float in water. Each allowed wet-rooted exception must preserve its species silhouette and visibly connect through replaced source-water voxels to the solid floor. Compare every far tier for stable surviving species geometry and strict nested canopy subsets. At step 32, confirm that a narrow channel elsewhere in the coarse cell does not erase an exact-approved canopy and that its trunk reaches the displayed voxel surface.

A zero exit code is not visual evidence. Read the validation log and inspect each PNG.

Run the view-distance-512 acceptance route without validation, require its lowest sustained one-second rate to remain at or above 60 FPS, and report exact and far residency, drawn and culling counts, queue settle time, and total unified memory separately from visual validation.

## 11. Report

Output in this order:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, convention section, concrete visible or GPU risk, and compliant fix
3. **Risks worth a look:** uncertain or capture-dependent issues
4. **Confirmed clean:** only rules exercised and passed
5. **Evidence:** build, validation-message count, tests, capture paths, and what each frame shows

Order findings by how badly they can break the frame.
