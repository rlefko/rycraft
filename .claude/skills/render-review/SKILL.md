---
name: render-review
description: Review Rycraft rendering changes for generator v4 bootstrap UI, expanded vertical range, exact and far ownership, surface-before-canopy publication, canonical water at every LOD, shared no-skirt transition topology, smooth packed light, five-cascade shadows, Hi-Z indirect lighting, physical atmosphere and weather, emissive gameplay materials, Metal layouts, culling, residency, and validation evidence. Reads docs/rendering-conventions.md as the source of truth.
---

# Render Review

Review the requested change against Rycraft's rendering contract. Prefer concrete file and line findings. Do not accept fog, walls, ledges, or skirts as substitutes for missing topology.

## 1. Read the source of truth

Read `docs/rendering-conventions.md` completely. Read `docs/generator-v4-follow-up.md` when the
paged hierarchy, GPU traversal, distant water, or distant flora is in scope. Read changed shared
layouts, far mesh structures, scheduler code, render-pipeline replacement logic, and water shaders
before reaching a verdict.

## 2. Establish scope

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD
git diff HEAD
```

If no shader, layout, mesh, water, culling, residency, upload, UI, or render-pipeline path changed, state that render review is not in scope.

## 3. Check shared layouts

For every changed C++ or MSL structure:

- Compare field order, width, signedness, and alignment
- Check size and offset assertions
- Check buffer and texture indices in pipeline creation and encoding
- Check all shader stages that consume the structure
- Confirm compatibility fields are not mistaken for active geometry behavior

Run:

```bash
rg -n 'struct .*Uniform|struct .*Vertex|\[\[buffer\(|\[\[texture\(' include src shaders
```

Production `FarTerrainMesh::skirtQuadCount` must remain zero even if legacy bit-29 or `skirtBottom` fields remain in a shared layout.

## 4. Check bootstrap rendering

The title screen must render model-required, downloading, verifying, compiling, loading, ready, and failed states before a `World` exists. `renderMenuOnly` must not dereference world, camera, entity, or streaming state. Once a qualified world exists but entry is still blocked, `RenderPipeline::renderV4Preparation` advances exact mesh publication, far-authority polling, base scheduling, result drains, shared-buffer publication, and protected FINAL work after the connected frontier reaches the near band. It must not issue the full exact-world scene draw.

Confirm byte progress, cancel, retry, repair, and quit remain reachable and hit-tested through the same layout used for drawing. After dry-spawn relocation, entry completion must require the connected 96-chunk parent prefix plus fresh matching world, view, protected, and exact-mesh epochs rather than a smaller selection or stale resident counts from the former camera position.

## 5. Check coordinates and height

Audit every changed origin, AABB, frustum, shadow, fog, cloud, water-sort, and culling calculation for Y=-128 through Y=1407 and section Y=-8 through Y=87.

CPU tile and chunk coordinates remain 64-bit in X and Z. Convert to camera-relative floating point only after subtracting the exact origin.

## 6. Check exact ownership

Confirm:

1. Exact meshes use a complete six-face halo.
2. Every required surface through the full 32-chunk exact disk receives mesh admission and upload
   priority after the camera and exploration lanes but before optional flora and broad work.
3. Exact collision uses loaded blocks and fluid only for renderer-published sections whose coverage
   epoch matches the active visual snapshot. Unowned planned sections use canonical generated
   terrain and fluid proxies, and unresolved sections stay closed.
4. Temporary boundary closure disappears when the halo arrives.
5. Far suppression is destination-column based and waits for every required exact section.
6. Terrain, water, and falls use the surface-ready ownership neighborhood, while tree and ground-flora fragments use the parallel flora-ready neighborhood across tile faces.
7. A partially owned patch does not enter conservative horizon occlusion.
8. Exact FINAL ownership waits for a stable FINAL far surface and never cuts a column into PREVIEW terrain.
9. A first-visible exact mesh cannot snapshot a cube while its bounded publication-light transaction
   remains pending.

## 7. Check surface-before-canopy publication

Trace a far key through base and attachment construction, completion, caches, uploads, residency, and refresh.

Require:

- `FarTerrainMesher::build` contains complete terrain, standing water, and falls only
- `buildCanopyAttachment` owns all tree and ground-flora counts and geometry
- Flora enrichment is lower priority than protected and urgent local terrain, but nearby ready flora has one opportunity before distant base and broad refinement work
- Canopy worker admission is zero during preparation and until the connected 96-chunk prefix is
  drawable, then exactly one low-priority gameplay worker remains available even during stronger
  terrain debt
- PREVIEW ecology publishes nearest-first, including queued, active, and parked work, before FINAL ecology promotion and remains until the grounded replacement is resident
- Missing PREVIEW ecology remains ahead of FINAL promotion on the single gameplay worker
- No second gameplay canopy lane opens after terrain debt clears
- Ground-flora LOD tiers retain nested half-open anchors and preserve projected cover
- Cancellation or failure retains the drawable base mesh
- Flora upload never replaces or mutates the base allocation
- Attachment-cache presence is the sole completion state, including an explicit empty attachment
- A blocked flora callback cannot prevent another parent from reaching the renderer
- Camera movement reprioritizes queued, parked, and follow-up flora work even when wanted membership is unchanged
- A nearby flora request can displace the least-important queued or parked request at the pending cap
- Broad refinements preserve the documented GPU flora-residency floor
- Surface and flora ownership masks retire their payloads independently

Do not call the queues fully independent if they still share storage, active-key bookkeeping, or worker capacity. Describe the actual scheduler.

## 8. Check canonical water

Inspect exact, far geometry-grid, point-probe, cell-bounds, meshing, and water-pass paths.

Require:

- One body identity and stage for standing water
- Exact and far visible planes match
- Step-32 cells consult `waterTopologyPossible` when corners are dry
- Interior probes and contour refinement are bounded and deterministic
- Narrow routes remain connected across tile and LOD boundaries
- Different stages do not join without explicit transition ownership
- Partially wet cells are contour-clipped, not rectangular sheets
- Vertical sides belong only to explicit falling water
- Half-open ownership prevents duplicate contour, ribbon, and fall faces

Reject a path that samples only terrain-grid corners or changes water authority by footprint.

## 9. Check no-skirt transitions

Search mechanically:

```bash
rg -n 'skirt|Skirt|bit 29|1U << 29' include src shaders tests
```

Classify each hit as compatibility metadata, test coverage, shader marker, or active geometry. Any production downward quad is a violation.

Trace the outer transition ring and require:

- Identical canonical boundary positions and heights for both tiles
- Positive winding and complete projected area on all four edges and corner pairs
- Half-open positive-area ownership with no duplicate top triangles
- At most a 2:1 displayed-neighbor ratio, including both endpoints of active replacements
- Adjacent-tier replacement rather than a direct multi-tier jump
- No transition-marked vertical face on a tile boundary
- Any interior transition-marked vertical face corresponds to a real source-column discontinuity

A fog pulse may hide a brief atomic swap, but not a static missing strip. Portable topology tests do not replace visual inspection of every adjacent LOD pair.

## 10. Check residency and culling

Confirm:

- Every visible coordinate retains a step-32 parent dependency
- Coarse parents remain drawable until connected replacements are resident
- Protected overlap never displays an ineligible coarse tier
- Entry requires both the connected PREVIEW parent prefix through 96 chunks and the atomic protected FINAL closure. The closure begins during preparation when the connected frontier reaches the near band and publishes only when its 4, 8, 12, 16, and 20 targets at steps 1, 2, 4, 8, and 16, their parents, and every internal canonical boundary match
- Parent and drawable frontiers remain connected
- Frustum culling precedes conservative terrain-horizon culling
- Partially faded, masked, or transitioning patches do not occlude
- Upload and GPU arena bounds remain enforced
- Water has one owner during an LOD transition
- Final parents in the exact-handoff prefix use protected authority priority, while movement
  prefetch waits for visible preview closure and remains bounded to eight leading pages
- A parked FINAL parent may use one-at-a-time quiescent liveness retries only when learned,
  publication, hydrology, base, and mesh producers are all idle. The retries retain epoch,
  priority, cancellation, and resident preview coverage, and latch repair after the bounded
  spill-summary limit instead of leaving entry blocked forever
- Same-key preview-to-final promotion draws both terrain and shadow allocations until completion,
  switches terrain and connected water together at the fog-covered midpoint, gates exact ownership
  until stable FINAL authority, and admits differing water identity or connectivity without waiting
  for exact coverage or latching recovery
- Protected FINAL work outranks ordinary refinement, but both advance through one adjacent tier
  at a time. A cached fine cap cannot bypass or starve a missing displayable bridge
- A camera-near PREVIEW bridge outranks distant FINAL work and may display inside a pending
  protected closure. The complete FINAL closure still publishes atomically and retires that bridge
- Saturated queues displace the worst queued or parked noncritical request for camera-near terrain,
  and movement transfers the protected class away from the old center. Distant work cannot evict
  a camera-critical job or CPU cache entry
- An urgent camera-critical refinement runs before the nominal four-worker reservation for an
  unrelated distant parent. Ordinary urgent work continues to observe that reservation, and the
  fixed total worker and urgent-job caps do not change
- A missing protected PREVIEW step-32 parent may displace lower-ranked queued or dependency-parked
  ordinary coverage at the cap
- Current protected FINAL children and parents receive bounded urgent capacity before required
  PREVIEW bridges, unused bridge capacity returns to current FINAL, and directional prediction uses
  only spare CPU-only capacity without changing display or publication state
- CPU and GPU pressure reclaim only optional non-displayed distant refinement or flora residency
  for camera-near work. Coverage, displayed, transitioning, and protected terrain stay pinned, and
  distant work cannot evict a camera-critical pending upload or GPU allocation
- Only the requested protected FINAL role-selected key may reclaim optional distant CPU and GPU
  residency and use the complete GPU arena. Alternate steps at the same coordinate do not inherit
  the class, and coverage, displayed surfaces, transitions, exact fallbacks, active protected lineage,
  and requested critical keys remain pinned
- After the connected 96-chunk prefix, exact publication or any connected visible desired-LOD miss
  pauses ordinary outer-parent submission and publication. Near work proceeds nearest-first,
  ranks horizontal distance before projected error within the nearby visible class, and may
  displace queued or dependency-parked outer parents without evicting displayed or structural
  owners
- Grouped protected FINAL owner crops match independent 517 by 517 requests exactly, including
  negative coordinates and shared aprons. The hydrology `deferredBuilds` counter records completed
  typed deferrals, not parked, active, or failed work
- Local far work admits 8 workers alongside exact debt, 12 after exact debt clears, and 16 after
  both exact and local debt clear. Canopy uses exactly one low-priority gameplay worker after the
  prefix is drawable
- Exact mesh saturation transfers the worst queued slot to a better camera request, camera movement
  cancels obsolete queued meshes, and first-publication lighting reranks before ownership changes
- Absolute distance bands are maximum-coarseness limits. Screen-space error may retain finer
  geometry above 0.55 pixels and outward coarsening waits until the next tier is below 0.45 pixels
- A standard or unknown exception from a critical far-base or final mesh latches retriable
  generation failure without evicting its valid resident parent

Use `RYCRAFT_WORLDGEN_OVERLAY=lod` to verify exact and step-1 through step-32 ownership in a
capture. Use `RYCRAFT_WORLDGEN_OVERLAY=authority` to verify that FINAL exact geometry never cuts
through a PREVIEW surface or an active authority exchange.

When the deferred paged hierarchy is in scope, require:

- A signed surface-first page forest with sparse volumetric bricks for cubic exceptions, not a
  world-scale pointer SVO or a dense voxel allocation at every level
- Flat bounded node and semantic buffers whose CPU and Metal layouts have matching assertions
- Projected canonical geometric and silhouette error rather than fixed distance or projected node
  area alone
- Atomic replacement only after a complete child terrain, transition, and canonical-water family
  is resident, with the parent retained in terrain, water, shadow, and occlusion ownership until
  commit
- Independent water contours, river ribbons, falls, body IDs, and stage summaries that generic
  voxel or modal aggregation cannot erase
- Independent flora attachments at exact-instance, crown-cluster, impostor, and canopy-aggregate
  tiers, with source flora retained until target flora uploads
- A 2:1 neighboring-node limit, half-open shared boundary topology, and no border overdraw, skirts,
  or vertical crack-hiding boxes
- Bounded GPU selection feedback, request overflow recovery, indirect command buffers, frame-safe
  heap ownership, and a matching CPU traversal fallback

## 11. Check indirect lighting and gameplay materials

Trace every pipeline descriptor, render-pass attachment, shader output, texture binding, history key, and frame-order transition used by screen-space lighting.

Require:

- Exact and far terrain, canopies, entities, item entities, and boats declare the same 4x HDR, `RGBA8Unorm` surface, `R8Unorm` reactive, memoryless R32 resolve-key, and depth contract
- Sky and block-highlight pipelines disable surface, reactive, and resolve-key writes
- Surface RGB stores diffuse albedo and alpha stores baked ambient accessibility
- The five-level filtered `R8Unorm` emission array uses the same layer and UV ownership as block color
- Lava is fully emissive, only torch flames and the single fixed -Z active furnace mouth emit, and beds, inactive furnaces, furnace shells and tops, chest surfaces, and torch sticks remain nonemissive
- A bed stops at 9/16 height in both geometry and collision, transmits skylight, culls supported faces, and remains in the normal shadow and indirect-light receiver path
- Bed and torch targeting intersects authored occupied bounds, ignores empty voxel space, and gives the selection wireframe those same bounds
- Furnace and chest fronts use fixed world -Z until facing metadata exists; the single-block bed has no persisted facing state and must not be described as a fixed-facing fixture
- Hi-Z preparation, trace, temporal accumulation, variance filtering, and additive apply run after opaque resolve and before clouds, weather particles, water, volumetrics, exposure, and bloom
- Opaque atmospheric fog runs after the indirect trace and apply, leaves clear-depth sky unchanged, and uses no additional persistent HDR target
- Clouds, rain, and snow use a single-sample resolved pass with opaque-depth testing and no surface attachment
- Direct and emitted radiance are independent of baked accessibility, while the traced accessibility correction changes only ambient response
- Resize, teleport, FOV, quality, world, session, direct-light source, prior-depth, resident-light-edit, and time-discontinuity changes reset history
- Far preview-to-final replacement, canopy arrival, connected-parent refinement, exact-to-far handoff, and preparation polling do not reset or advance history incorrectly
- No legacy SSAO source, shader, or render pass remains; `ssao` and `RYCRAFT_SSAO` are compatibility names for the Hi-Z indirect-light toggle only
- The retained Boolean maps false to off and true to high quality; medium remains an internal bounded mode

Reject a material path that makes a complete furnace shell or torch stick glow, lets a bed emit, feeds atmospheric fog or translucent weather into indirect history, reduces the material resolve key to fp16, or clears history on ordinary far publication.

## 12. Check integrated lighting, atmosphere, and weather

Trace one terrain surface, entity, water highlight, cloud ray, and froxel through the active direct-radiance authority. The sun contributes no direct light below the horizon, the moon stays suppressed through civil twilight, and deterministic synodic phase scales the moon disc, lunar radiance, shadows, and water specular. Water, clouds, froxels, and shadows use the same active source. Atmosphere and solar flare retain explicit true-solar state without reintroducing direct sunlight. Reject a clear daytime frame whose physical sky is dark while terrain receives daylight.

Verify all five shadow records and grouped textures. High endpoints are 48, 160, 512, 1,536, and 8,192 blocks; Medium endpoints are 40, 128, 384, 768, and 8,192. High resolutions are 4,096, 4,096, 2,048, 2,048, and 2,048; Medium resolutions are 2,048, 2,048, 1,024, 1,024, and 1,024. Selection uses camera-forward view depth and the final 12.5 percent of adjacent ranges blends continuously. The first two records refresh every frame. Farther records refresh on invalidation or maximum intervals of two, four, and eight frames, and a skipped record retains the matrix paired with its depth texture.

Trace packed skylight and block light from the 18 by 18 by 18 mesh snapshot through smooth per-vertex corner values, merge keys, diagonal selection, and shader interpolation. A torch placement or furnace-state change must relight and remesh the bounded affected neighborhood in time for the first visible changed frame. A streaming publication transaction performs at most 32 floods, queues overflow by current-camera priority, and blocks its exact mesh snapshot until the revision-matched transaction settles. Direct visibility, propagated skylight, baked accessibility, block light, emissive radiance, ambient occlusion, and screen-space bounce remain independent terms.

For atmosphere, require 256 by 64 transmittance, 32 by 32 multiple-scattering, and 192 by 108 sky-view LUTs with Rayleigh, Mie, ozone, solar angular radius, altitude, aerosols, and humidity. Generator v4 altitude is sea-relative from Y=64 at 7.5 meters per positive-elevation block. Cloud layers, cloud-march bounds, lightning, and atmosphere must remain valid above terrain through Y=1407. Directional cloud attenuation belongs to the volumetric cloud composite, not a camera-local scalar that darkens clear gaps twice.

For clouds, require deterministic 128-cubed base noise, 32-cubed erosion noise, and a two-dimensional curl map built off the render thread; cancellation and world-instance plus seed validation before upload; no worker or allocation at quality Off; bounded 100 and 500 millisecond failure backoff; a third-failure latch reset only by quality re-enable or a new world; retained failure timing and one log per attempt; the documented stratus, cumulus, cumulonimbus, and cirrus profiles; physical extinction and phase response; bounded light marches; and wind in blocks per second. High uses 48 view and 6 light steps, Medium uses 24 and 3, both at quarter resolution. Inspect temporal rejection, neighborhood clamp, bilateral upscale, and the snapped cloud-shadow authority.

For froxels, require a 160 by 104 by 64 logarithmic grid. Injection consumes atmosphere, weather aerosols, humidity, precipitation fog, the active directional source, terrain shadows, and cloud shadows. Underwater absorption remains separate, air fog stops at the surface, and disabling volumetric lighting retains only LUT aerial perspective.

For weather and storms, prove one immutable snapshot feeds every frame consumer. The v4 snapshot samples learned elevation and climate without routing hydrology across its horizon grid. Deferred authority must retain the prior snapshot and may not escape the utility worker. Stable `clear`, `overcast`, `rain`, `storm`, and `snow` overrides reproduce captures and reset affected temporal histories. Lightning derives from deterministic event IDs, depth-tests against terrain and cloud hit depth, and never mutates blocks or fluids. Thunder remains procedural, de-duplicated, bounded, and delayed at 343 meters per second after applying the active world's physical scale.

## 13. Run portable tests

```bash
ninja -C build tests/test_rycraft
./build/tests/test_rycraft "[render][far-terrain]"
./build/tests/test_rycraft "[reported-water-continuity]"
./build/tests/test_rycraft "[render][indirect]"
./build/tests/test_rycraft "[render][indirect][mrt][particles]"
./build/tests/test_rycraft "[render][textures][emissive]"
./build/tests/test_rycraft "[render][mesher][bed]"
./build/tests/test_rycraft "[render][lighting]"
./build/tests/test_rycraft "[weather]"
```

Focused tests must include zero skirt quads, all transition edges and corner pairs, mixed 2:1 negative-coordinate neighbors, half-open duplicate rejection, fixed topology budgets, dry-corner step-32 water recovery, exact and far stage equality, full-disk exact surface priority, epoch-matched exact collision ownership and proxies, bounded first-publication lighting, dynamic 8/12/16 local worker admission, the ordinary outer submission-and-publication pause during exact or desired-LOD debt, distance-first nearby scheduling before projected-error ranking, near-job displacement of queued outer parents, grouped protected-owner exact equivalence, corrected deferred-build metrics, zero canopy workers before connected gameplay coverage, the one-worker service guarantee under continuing publication debt, missing-flora priority over promotions, monotonic canopy fallback during LOD transitions, urgent protected-parent displacement, current-FINAL-before-bridge ordering, critical protected full-arena admission, surface publication while flora work is blocked, PREVIEW-grounded flora, nested ground-flora anchors, camera-order reprioritization, parked-work displacement, and separate exact flora handoff.

## 14. Validate and inspect captures

Run Metal validation separately from performance:

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft
```

Capture exact, handoff, step 1, step 2, step 4, step 8, step 16, and step 32 views. Include narrow rivers, lakes, coasts, waterfalls, high terrain, cold horizon loading, delayed canopy enrichment, delta distributaries, brackish estuaries, and connected wetlands at page seams. Also capture lava, a torch in darkness, inactive and active furnaces, every visible bed face, rain, and snow with screen-space indirect lighting enabled and disabled. Exercise a light edit, teleport, FOV change, sleep time jump, and far-only refinement while watching temporal convergence.

Capture motion through all four detailed shadow ranges and the horizon range. Capture clear, overcast, rain, storm, and snow at dawn, noon, dusk, and night; views below, inside, and above each cloud family; lightning in front of and behind clouds; representative lunar phases; and air-to-water transitions. Record the active direct source, cascade refreshes, atmosphere refreshes, temporal reset reasons, weather snapshot identity, and storm event IDs.

Open every frame. Inspect for:

- Empty horizon gaps
- Artificial vertical banks or ledges
- Straight or sawtooth shorelines
- Step-32 water deletion
- Duplicate or joined water levels
- Downward panels
- Fine-to-coarse cracks
- Canopy-delayed terrain or water
- False ridge occlusion
- Missing emissive radiance, indirect bounce, or bloom from a torch or active furnace
- Incorrect bed geometry, texture ownership, culling, or baked corner accessibility
- Temporal ghosts after a required reset or needless convergence loss during far refinement
- Rain or snow contributing indirect light or material history
- Shadow blend rings, stale cached projections, or exact and far double-casting
- Dark physical daytime sky over daylight-lit terrain
- Cloud or froxel temporal trails, repeated slabs, or air fog below the water surface
- Duplicate direct sun and moon terms or below-horizon sunlight
- Lightning without delayed thunder, with backlog replay, or with world mutation

Record generation fingerprint, seed, camera, tier, frame, queue state, and validation-message count. A file that exists but was not opened is not evidence.

## 15. Report

Output:

1. **Verdict:** clean, clean with notes, or violations found
2. **Violations:** file and line, visible impact, and compliant fix
3. **Unverified visual risks:** exact capture needed
4. **Confirmed clean:** only exercised paths
5. **Evidence:** tests, validation logs, captures, seed, fingerprint, and settings

Do not claim a visually qualified transition strip, complete Core ML result, or inspected frame without direct evidence.
