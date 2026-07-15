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
- Missing in-range neighbors on all six cardinal faces, including inward-facing unlit caps
- Vertical loaded-path continuity before an underground cube receives skylight

Corner-smoothed water samples X/Z diagonals. Confirm that loaded diagonals provide symmetric real data, missing diagonals use the same conservative opaque fallback, and an edge or corner edit or halo load transition invalidates every affected mesh.

## 7. Check partial water geometry

Verify source, levels 1 through 7, falling state, full-height water with water above, corner averaging, horizontal flow bits, bottom visibility, and culling against adjacent water. Stable source and flowing cells emit top geometry only. Vertical sides are exclusive to explicit falling columns and must not appear at lake, river, ocean, delta, cube, or unloaded boundaries.

Trace generated lake occupancy back to the `ColumnPlan` 17 by 17 canonical authority. Dry contributors must taper lake depth to zero without wet-weight renormalization, and every exact shore-water voxel must have a solid support below its lowest water cell. Every standing generated water voxel from that support through the surface must resolve as an implicit source, including across cube faces, without an explicit fluid array. A rendered top without canonical occupancy, full-depth source water, or support is a floating sheet even if its side faces were removed.

For a crater lake, require a coordinate-warped irregular shoreline, a complete dry enclosing rim with at least one block of freeboard, supported banks, and rejection when a safe wet radius does not fit. A routed nonendorheic lake may instead keep its named outlet; do not close that outlet merely to satisfy the rim check.

Review a lake `OutletFall` separately from both standing bodies. Its sample retains top, bottom, width, flow, and one receiver anchor while the lower sample keeps its own `waterSurface`. Exact generation must emit only a short, narrow receiver-centered falling footprint. The anchor's half-open far tile must own exactly one complete five-quad prism, with four sides and one top, even across a tile face. No neighbor may duplicate it, no long slab may replace it, and neither path may raise the receiving water or enqueue a generation-time fluid tick.

Compare renderer height with `fluidSurfaceHeight`, buoyancy, and camera-submersion logic. Sub-block coordinates must be binary-exact over the 0 through 16 cube-local range. Current eighth-block levels and 1/32-block corner averages satisfy that rule.

For generated source water, compare exact and far geometry at the same coordinate. Both must place the visible top 0.875 blocks above the water voxel floor.

Confirm water still renders after scene resolve, samples a color copy and resolved depth, binds no depth attachment, sorts back to front in three dimensions, and preserves underwater overlay ordering.

Trace resolved-depth reconstruction at large coordinates. Water surface, opaque floor, refracted floor, screen-space reflection, and underwater positions must share a camera-relative world frame whose view matrix omits camera translation. Absolute camera position may be added only after thickness math to anchor caustics and fog. Require a focused round-trip test beyond 100,000 blocks and across both sides of a cubic chunk face. Reject an absolute inverse view-projection for water depth because its precision loss appears as 16-block absorption and brightness tiles.

## 8. Check far LOD, winding, and visibility

For far terrain, verify:

- Exact editable cubes stop at radius 32 and far tiles alone extend the visible horizon through radius 256.
- Tiles are 256 by 256 blocks in the half-open annulus `[32, 256)`. A narrow two-block sampling tier immediately outside radius 32 samples exact emitted density heights as the topology bridge.
- Direct tests compare the two-block topology tier with exact surface samples. Farther out, distance and immutable maximum slope and hydrology complexity select among globally aligned four-, eight-, and sixteen-block tiers using bounded, tunable thresholds.
- The previous tier applies asymmetric refine and coarsen thresholds, and no more than 64 resident replacements use the 0.4-second fog-hidden transition at once.
- Adjacent equal-LOD tiles reproduce identical border heights, mixed-LOD transitions use bit-29-marked skirts only on the finer edge of a resident coarser neighbor outside the exact handoff band, and greedy merging does not change sampled shape. Same-LOD and absent-neighbor edges carry no visible skirt.
- Partially wet far cells use contour-clipped shoreline triangles. Stable far water emits top geometry only and never rectangular vertical sheets.
- Lake outlet falls remain separate receiver-centered five-quad prisms owned by one half-open anchor tile. Their bottom overlaps the lower body's top source voxel, their top reaches the upper lip, and their bounds remain narrow in the flow direction.
- Two- and four-block far tiers query the same exact accepted tree anchors as cubic generation. Eight- and sixteen-block tiers instead query globally anchored aggregate forest cells at 32- and 64-block spacing, preserving climate- and substrate-suitable canopy mass without exact local-priority competition.
- One half-open tile owns each grounded trunk-and-crown impostor or aggregate cluster, even when the crown crosses a tile face. Every impostor vertex carries bit 28 for classification. Water and canopies share a stable world-space dither over the 16-block handoff. Opaque far tops remain depth-biased behind resident exact surfaces so cold exact meshes never open a dark ring.
- Exact cube faces and far opaque faces are outward counterclockwise and use back-face culling.
- Cross and flat flora emit both windings; water and shadow casters stay cull-none.
- Tile AABB frustum culling runs before front-to-back 256-bin horizon culling.
- Sixteen 64 by 64-block patches per visible tile contribute conservative lower horizons without per-tile heap allocation.
- The horizon test uses only fully covered azimuth bins, selects distance extrema correctly above and below the camera, and cannot hide a taller or partially covered tile.
- A coordinate retains its old resident LOD until the desired adaptive replacement uploads successfully, then draws only one topology at a time through the fog-hidden transition.
- Exact opaque terrain draws before positively depth-biased far tops, which remain available as cold-residency fallback. Water has no depth attachment, so water and canopies retain the shared 16-block fragment ownership predicate.
- Submission uses bounded direct indexed draws. Do not report hierarchical Z, a literal geometry clipmap, an indirect command buffer, or GPU-driven submission unless it exists in code.

Require focused CPU tests for direct exact-to-far two-block topology agreement, identical 16-block handoff coverage for water and canopies, opaque cold-residency fallback, exact 0.875 source-water agreement, full-depth implicit generated sources across cube faces, 17 by 17 canonical lake authority, supported shore occupancy, complete caldera rim and freeboard, exact and far outlet-fall agreement, five-quad half-open fall ownership, exact canopy ownership at steps 2 and 4, aggregate forest ownership at steps 8 and 16, bit-28 canopy coverage, bit-29 skirt coverage, displayed-neighbor edge masks, and handoff suppression, bounded threshold selection, complexity shifts, asymmetric hysteresis, fog-transition phases and cap, negative tile coordinates, deterministic hashes, same-LOD borders, mixed-LOD skirts, contour-clipped shorelines, outward winding, scheduler caps, epoch cancellation, and conservative ridge and peak cases. Missing-halo tests must distinguish a lit aboveground terrain silhouette from a dark underground opening. Inspect the opaque and water fragment paths for their distinct ownership contracts.

## 9. Check residency and frame work

Confirm exact mesh residency never exceeds 16,384 and loaded halo cubes never exceed 32,768. Confirm the far CPU cache stays within 1,024 entries and 512 MiB, while far GPU arenas remain 256 MiB for vertices and 128 MiB for indices. Verify unload and movement sweeps defer arena frees until the referencing frame completes. All exact, far, post-target, world, and transient use must fit the 64 GB unified-memory acceptance limit.

Inspect first-time exact builds, edit rebuilds, far builds, and uploads for their worker or main-thread path. Exact meshing permits 64 total queued, building, completed, or renderer-pending items and 64 uploads or 32 MiB per frame. Far terrain permits 64 pending, 32 completed, and 12 uploads or 32 MiB per frame.

## 10. Run validation and inspect captures

Use the `playtest` workflow. At minimum:

```bash
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MTL_SHADER_VALIDATION=1
```

For cubic changes, require representative captures from above terrain, below terrain, across a horizontal seam, and across a top or bottom cube face. Capture active aboveground streaming and reject any full black loading panel; depth-backed far tops must provide a lit fallback until exact surfaces arrive. Underground captures must include lateral and vertical travel through the hard-priority exploration band, dark closed temporary openings, no skylight through a missing section above, and no bright void at a missing boundary below. For water changes, include stable lake, river, ocean, and delta edges with no vertical walls, supported lake-shore occupancy, explicit falling columns with sides, the seed-42 lake lip at X=-8235, Z=2976, the separate incised river face at X=-12288, Z=2653, the canyon at X=-23904, Z=0, a receiver-centered lake-to-river outlet fall joining both standing levels without a slab or gap, exact-to-far 0.875 source-water agreement, far contour-clipped shorelines, and an underwater view. Capture the seed-764891 caldera at X=23029, Z=-111486 around its complete perimeter and from water level, verifying an irregular supported bank, one block of freeboard, and water filling every voxel from crater floor through its flat surface. For far-terrain changes, capture direct agreement between exact surfaces and the two-block topology tier, the shared 16-block water and canopy dither, opaque fallback during cold exact residency, a forest spanning the exact handoff and every far tier, exact canopies at steps 2 and 4, stable aggregate forest cover at steps 8 and 16, the gradual taper into 4/8/16 tiers over both flat and complex terrain, refine and coarsen hysteresis crossings, a topology replacement, the radius-256 horizon, a ridge occluding lower ground while preserving a taller peak, and a turn that reveals already resident tiles. Reject any same-LOD or absent-neighbor skirt panel. For texture changes, inspect distant textured slopes and alpha-cutout flora for aliasing, shimmer, coverage loss, and visible mip transitions. World-generation rendering should also capture the features changed by the diff, including volcanic interiors, crater water, aquifers, or distributary channels when relevant.

A zero exit code is not visual evidence. Read the validation log and inspect each PNG.

Run the view-distance-256 acceptance route without validation, require its lowest sustained one-second rate to remain at or above 60 FPS, and report exact and far residency, drawn and culling counts, queue settle time, and total unified memory separately from visual validation.

## 11. Report

Output in this order:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, convention section, concrete visible or GPU risk, and compliant fix
3. **Risks worth a look:** uncertain or capture-dependent issues
4. **Confirmed clean:** only rules exercised and passed
5. **Evidence:** build, validation-message count, tests, capture paths, and what each frame shows

Order findings by how badly they can break the frame.
