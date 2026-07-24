# Rendering Conventions

This is the Metal, cubic-mesh, far-terrain, and HDR frame-graph rulebook. The `render-review` skill walks the checklist at the end.

## Prime directive

The frame must be correct before it is optimized. Shared layouts are compile-checked, pipeline state matches the pass that uses it, cubic coordinates include Y everywhere, all six exact mesh boundaries use symmetric halo data, visibility tests are conservative, and every player-visible change is run with Metal validation and inspected in captured frames.

## 1. C++ and MSL share one layout

- Every structure read by C++ and Metal lives in `include/render/shader_types.hpp` and has matching `sizeof` and `offsetof` assertions in `tests/test_render.mm`.
- Use `simd` types for GPU-shared vectors and matrices. Do not copy a structure declaration into a `.metal` file or invent float-array padding.
- Buffer length comes from `sizeof(TheStruct)`, never a byte literal.
- A per-frame writable buffer is ring-buffered or otherwise protected from encode-versus-execute overwrite.

The exact and far terrain vertex remains exactly 16 bytes:

```text
uint32 faceAttr
half3  local position
half2  UV
```

`faceAttr` uses one exhaustive layout:

| Bits | Meaning |
|---|---|
| 0 through 2 | Face normal |
| 3 through 10 | Texture array layer |
| 11 through 14 | Skylight (per-vertex smooth-lit corner value) |
| 15 through 16 | Baked corner ambient occlusion |
| 17 through 20 | Block light (per-vertex smooth-lit corner value) |
| 21 | Emissive flag |
| 22 through 23 | Wind sway class |
| 24 through 26 | Water flow direction |
| 27 | Falling water |
| 28 | Far-canopy impostor marker |
| 29 | Legacy far-terrain boundary-skirt marker; production geometry leaves it clear |
| 30 | Water exterior-sky authority |
| 31 | Reserved |

Renderer metadata must stay in the assigned high bits unless the vertex descriptor, shaders, tests, arena budgets, and every producer change together. Bit 28 identifies visual-only far canopy geometry so the fragment shader can suppress it in exact-owned columns. Bit 29 remains reserved for compatibility with legacy far meshes, but production geometry must not emit boundary skirts or set the bit. Bit 30 is a binary water-interface classification derived from propagated skylight or the complete edited column cutoff. It must not alter ambient skylight, and it prevents incomplete streaming authority from making exact water disagree with far water.

## 2. Pipelines match their passes

- A pipeline's `rasterSampleCount` equals the attachment sample count for every pass that uses it.
- A pipeline's depth format equals the bound depth texture format. Scene depth is `Depth32Float`.
- Fullscreen sampling passes flip V because Metal texture coordinates and NDC Y run in opposite directions.
- Anything encoded in the 4x MSAA scene pass declares sample count 4. Post-resolve water and final composites declare sample count 1.
- Color attachment formats match the linear HDR or display stage in which the pipeline runs.

## 3. Resolution, storage, and GPU lifetime

- Scene targets use drawable pixel dimensions, not Cocoa point dimensions. Recheck drawable size every frame.
- The scene target is linear `RGBA16Float`. Its 4x MSAA color and depth textures are memoryless and resolve into single-sample textures.
- Opaque shading also resolves one `RGBA8Unorm` surface attachment. RGB stores diffuse albedo and alpha stores baked ambient accessibility. Direct, block, and emissive radiance remain only in the HDR scene source.
- CPU-written buffers use `StorageModeShared` on Apple Silicon. Never call `contents` on a private buffer.
- Per-frame constants use three frame-ring slots behind the three-frame semaphore.
- Exact and far mega-buffer ranges enter deferred-free queues. A range cannot return to an allocator until the GPU completes the frame that last referenced it.
- Exact mesh residency cannot exceed 16,384. The far CPU terrain and canopy caches cannot exceed 24,576 entries each, 3 GiB for terrain, or 512 MiB for canopy. The far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertices and 1 GiB of indices. Parent coverage reserves the last 64 MiB of vertices and 32 MiB of indices. Visible flora can use a separate 64 MiB and 32 MiB floor that broad refinements cannot consume. A role-selected requested protected FINAL refinement may reclaim optional distant non-displayed refinement or flora allocations and use the complete arena. Coverage, displayed surfaces, transition endpoints, exact fallbacks, active protected lineage, and the requested critical keys stay pinned.
- All renderer targets, arenas, caches, world state, transient work, and Metal allocations together must remain below the 64 GiB unified-memory acceptance ceiling.
- Persistent High-tier allocations added for integrated scene targets, shadows, indirect lighting, atmosphere, clouds, lightning, and froxels must remain at or below 768 MiB. Recalculate the payload for the exact drawable and settings under test, then record both the helper result and Metal's reported allocation. Include the complete HDR water-reflection mip pyramid, full-resolution SSGI guides, SSGI histories, and froxel histories. Exclude memoryless MSAA attachments from persistent payload and report resized histories and shadow groups once, not once per frame.
- The procedural block-texture array has a complete five-level mip chain from 16 by 16 through 1 by 1. Alpha-aware downsampling preserves representable cutout coverage. Its sampler uses nearest magnification, linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy.

## 4. Coordinate conventions and precision

- Matrices are column-major with column vectors, a right-handed world, and Metal depth in [0, 1]. `perspective`, `lookAt`, frustum extraction, and CPU-to-MSL copies remain covered by tests.
- Exact vertices are local to one 16 by 16 by 16 cube. Valid exact block geometry lies in 0 through 16 on X, Y, and Z. `ChunkOrigin` restores `chunkX * 16`, `chunkY * 16`, and `chunkZ * 16` in the vertex shader. Generator v4 spans section Y=-8 through Y=87, or world Y=-128 through Y=1407, and every culling, shadow, fog, cloud, entity, and water path must accept that complete range. Atmosphere altitude, natural and forced cloud layers, cloud-march bounds, lightning, and thunder use the shared v4 datum at Y=64 and the 7.5-meter positive-elevation scale. High terrain may not clip clouds or place storm layers underground.
- Far vertices are local to one 256 by 256-block tile in X and Z. Their Y remains relative to world Y=0. The far `ChunkOrigin` restores the tile's 64-bit CPU origin at draw time.
- Never bake world X or Z directly into half-precision vertices. Exact local half values preserve partial-fluid geometry. Far geometry accepts the local half precision appropriate to its two-, four-, eight-, sixteen-, or thirty-two-block sample step.
- Face planes sit on block boundaries. The negative X face of local block `x` is at `x`; its positive X face is at `x + 1`. The same rule applies on Y and Z.
- Exact cube AABBs, frustum bounds, camera distance, candidate priority, opaque origins, entity relationships, and water sorting include Y.
- Far tile AABBs use sampled surface minimum and maximum Y for visibility. The full tile bound remains available for transition and allocation ownership.
- Water reconstructs resolved depth in a camera-relative world frame. Remove the camera translation before inverting view-projection, compare surface and floor positions in that frame, and add the absolute camera position only after depth math for world-anchored caustics. An absolute inverse view-projection loses visible precision at large coordinates and can reset absorption at cubic chunk faces.

Exact sub-block geometry uses binary-exact fractions. Flora insets and lily-pad height use 0.125. Fluid cell heights use eighth blocks, and four-cell corner smoothing can produce multiples of 0.03125. Both remain exactly representable through the exact cube-local range.

## 5. Exact mesh snapshots and six-face seams

The game meshes exact terrain from `MeshSnapshot`, not directly from a mutable cube. A snapshot contains the 16-cube-edge interior plus a one-block halo, for 18 by 18 by 18 block, fluid, and packed voxel-light data. The high light nibble is 4-bit skylight and the low nibble is 4-bit block light. Separate 18 by 18 arrays retain generated-surface and complete-column sky-cutoff authority. A valid loaded cutoff may raise or lower the generated surface. The incomplete-path sentinel is distinct from every valid cutoff, including the world-ceiling cutoff.

- Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real face, edge, and corner samples. An unavailable in-range halo follows the immutable generated surface cutoff. Cells above the silhouette remain air and cells below it remain opaque. When a lateral cap candidate exists, a bounded six-connected flood marks transparent cells reachable from sky-exposed cells within the padded snapshot. A complete loaded cutoff honors added roofs and removed surfaces. An incomplete vertical path uses generated authority for this provisional classification while remaining fully occluded for ordinary skylight. Sky-connected lateral openings emit normally lit provisional faces using one representative arriving-surface material per missing face. Enclosed lateral openings emit unlit stone, and missing vertical openings emit bedrock. These faces are temporary boundaries, not generated world content.
- Solid faces, transparent faces, packed skylight and block light, flora ownership, ambient occlusion, fluid corners, and explicit falling sides read real data whenever the corresponding neighbor is loaded. Loading or unloading any halo cube reconciles the affected light faces and dirties only neighboring meshes whose sampled halo or boundary light changed.
- Cube publication initializes the arriving cube from every available face neighbor, compares vertical sky authority before and after insertion, and starts one bounded affected-light transaction while the world map is still locked. A transaction performs at most 32 floods. Overflow remains pending in the camera-ranked publication queue, and `snapshotForMeshing` rejects every pending cube. The arriving cube and any resident neighbor therefore expose one settled packed-light version to the first eligible mesh snapshot without turning generation into an unbounded light flood. Missing neighbors remain conservatively dark until their own publication.
- Edits dirty the owning mesh and every face, edge, or corner neighbor mesh whose one-block halo intersects the changed block. A player edit also synchronously converges its whole affected light neighborhood, so any neighbor whose sampled halo light the edit changed is both relit and dirtied in the same tick rather than a tick later. Why: a placed or broken torch used to light adjacent cubes a tick or more late.
- Water corner smoothing at an X/Z edge reads the diagonal cube when it is loaded. A missing diagonal is the same closed opaque boundary on both participating meshes.
- Complete full-height column authority seeds level-15 skylight only where the path to the sky is unobstructed. `LightEngine::floodChunk` propagates both light nibbles through transparent cells across all six faces and loses one level at each non-seeded step. Generated cutoffs remain seed and provisional-boundary authority, not a final binary light value. An incomplete vertical path remains fully dark, so an unloaded section above an underground view cannot admit sunlight.
- At the vertical world floor, missing halo cells are bedrock. Above the ceiling they are air.
- Test positive and negative X, Y, and Z separately. A horizontal-only seam test cannot catch a wrong cube Y origin or top and bottom halo defect.

Exact opaque cube faces use greedy merging with Minecraft-style smooth lighting: skylight and block light are averaged per vertex over the non-opaque cells touching each corner in the outward plane (the diagonal drops when both sides are opaque, matching the ambient-occlusion short-circuit), so light interpolates across a face instead of stepping per voxel. The 64-bit merge key includes block and texture identity, face direction, the four 4-bit skylight corners, the four 4-bit block-light corners, and the four 2-bit ambient-accessibility corners, so cells merge only where every per-vertex value matches and a light gradient keeps its per-vertex detail. A quad splits along its brighter diagonal when corners are not planar. The water surface's top face smooths skylight per corner the same way; its sides and bottom stay flat. Why: flat per-face block light stepped abruptly from voxel to voxel at night.

## 6. Block shapes, winding, and emissive materials

`BlockDefinition::renderShape` is exhaustive. Cubes participate in greedy opaque meshing, flora crosses emit two diagonal inset planes, the distinct floor-torch cross retains that silhouette without inheriting flora replacement rules, low boxes emit authored partial-height faces, flat shapes emit a horizontal two-sided plane, liquids enter the water path, and none emits no geometry. Exact opaque merge keys include block and texture identity, face direction, skylight, block light, emissive state, sway state, and baked-accessibility corners.

Torches use their authored centered noncube shape. Their dedicated cross vertex class remains double-sided without inheriting randomized flora poses, wind, plant-facing shading, or plant subsurface response. Floor placement requires a full solid support, support loss drops the torch, and fluid rules do not treat it as replaceable flora. Beds use a single-block 9/16-height box with the same partial collision volume and remain transparent to skylight. Furnaces use the inactive or active texture set that matches the persisted furnace state. Furnace and chest fronts face fixed world -Z until saves gain facing metadata. Lava, torches, and active furnaces set both light emission and emissive material state. Their light propagates through the same block-light field used by exact geometry, while their surface radiance stays in linear HDR for bloom and indirect-light consumers.

The block-level emissive bit is only a fast shader gate. A five-level filtered `R8Unorm` array supplies the actual per-texel mask with the same layer and UV ownership as the color array. Lava is fully emissive. Only the painted flame of a torch and the one fixed front mouth of an active furnace emit. The torch stick, active-furnace side and top layers, inactive furnace, chest, and bed have a zero emission mask. Low beds stay in the opaque geometry and shadow index range while their block opacity remains false; their authored faces use ordinary culling, smooth packed light, and baked corner accessibility. The screen-space indirect path consumes resolved emissive radiance and may not maintain a separate CPU block scan or gameplay allowlist.

Main-pass cube faces and far terrain use counterclockwise outward winding with back-face culling. Cross and flat geometry emits both windings. Alpha-cutout blocks and their shadow casters use matching discard and wind-sway rules. Water and shadow casters retain their established culling rules.

## 7. Far-terrain LOD, water, and visibility

### Far base and canopy payloads

A far key has two independently published payloads:

- Base mesh: terrain, standing water, and falls
- Flora attachment: tree forms plus deterministic ground-flora clumps and impostors, or an explicit empty attachment

The base mesh is safe to draw immediately. Flora construction is lower priority and uploads into a separate allocation without replacing the base. Its worker budget remains zero during entry preparation and until the connected 96-chunk parent prefix is ready. Gameplay then guarantees exactly one low-priority flora worker even while exact or protected local terrain debt persists, which prevents continuously replenished terrain work from starving every far attachment. No second gameplay flora lane opens. Missing nearby flora publishes from PREVIEW ecology before any FINAL ecology promotion consumes the worker. The provisional attachment grounds against the displayed surface and remains resident until its FINAL ecology replacement uploads atomically. A delayed, blocked, canceled, or failed flora callback may not remove or delay terrain and water.

Presence of the render pipeline's flora-attachment entry is the sole completion state, including when the attachment has no allocation because it is empty. Refresh scans are bounded, rank missing drawable attachments before provisional FINAL promotions, and remain nearest-first within each class. Camera movement refreshes queued, parked, and follow-up priorities even when wanted membership is unchanged, and nearer work can replace the least-important queued or parked request at capacity.

Exact surface and flora ownership use separate per-column readiness masks carried through the same center-plus-eight-neighbor uniform. Terrain, water, and falls retire against the surface mask. Far tree and ground-flora fragments retire only after the exact column's conservative flora section set is revision-ready. Every required exact surface section through 32 chunks receives generation, mesh, and upload priority before optional flora. The camera column and exploration band remain first. Nearby flora-bearing exact sections then run ahead of distant broad terrain after local terrain debt clears.

The renderer publishes a separate exact collision-section snapshot after applying the same per-column
handoff predicate, tagged with the active surface-coverage epoch. A stale epoch is rejected by
`World`. A matching owned section uses loaded exact block and fluid data; an unowned planned section
uses canonical generated terrain and fluid proxies, and an unresolved plan remains closed. Empty but
revision-ready exact sections still publish exact air. Collision therefore changes authority with
the visible handoff rather than when a partially loaded cube happens to arrive.

### Parent coverage and refinement

Every selected 256 by 256-block coordinate needs a resident step-32 parent before refinement can display. A coarse parent remains drawable until a connected final replacement is resident. The drawable coverage frontier suppresses out-of-order islands beyond the nearest missing required tile.

The runtime uses a hybrid spatial hierarchy. Mutable exact cubes remain sparse and hash-indexed because the near simulation region is dense and write-heavy. Distant terrain and water use the two-dimensional tile hierarchy because their drawable authority is a single-valued surface. Unified frame-level coverage publication applies the parent-retention and connected-child readiness principles used by Distant Horizons and Voxy. Rycraft does not implement a literal sparse voxel octree.

Final parents that protect the exact handoff use the learned authority's protected-handoff lane.
Directly intersected native hydrology owners may share one lexicographically grouped FINAL
rectangle of at most two by two owners. Each 517 by 517 owner crop must remain exactly equivalent
to a separate FINAL request. Movement prefetch begins only after visible preview authority is ready,
updates after one chunk of travel, and warms at most eight pages beyond the leading edge through the
lowest-priority lane. A parked FINAL parent normally resumes when its learned completion becomes
observable. If every learned, publication, hydrology, base, and mesh producer is idle, one bounded
liveness probe resumes reconciliation without waiting for an impossible future completion.
Exhausting the spill-summary bound latches generator recovery without removing the valid resident
preview parent. The hydrology `deferredBuilds` metric counts completed typed deferrals, not parked or
active work.

A same-key preview-to-final promotion retains two real GPU allocations until its bounded terrain and shadow transition completes. PREVIEW reconstructs the seeded Base latent's low-frequency terrain without a decoder residual, while FINAL adds that latent's decoded residual through the same cleanup path. Coarse conditioning is never rendered. PREVIEW water is temporary coarse coverage, while FINAL water is canonical. Preview and final meshes record water-body, transition, and connectivity signatures for diagnostics, and residual refinement may still change those signatures locally. Matching terrain, standing water, and falls switch authority together at the fog-covered midpoint. The CPU submits only the matching connected-water owner, and the shader enforces the same source-or-target cut. A topology-changing per-tile promotion is not accepted visual evidence until its complete 2,048-block hydrology owner and perimeter can publish together. A ready exact column may own its sections only when the displayed far surface is FINAL and no same-key authority transition remains active. Until then, the far parent retains the whole destination column. This prevents decoded FINAL exact geometry from cutting rectangular holes into a low-frequency PREVIEW surface.

Cold entry uses the same base-parent ownership rule as gameplay. The preparation renderer publishes PREVIEW terrain and canonical water for the connected step-32 prefix through 96 chunks while also advancing exact spawn meshes. As soon as the connected frontier reaches the near band, it opens the camera-critical protected FINAL lane. The gate validates the selected radius, protected anchor, world epoch, view epoch, protected epoch, collision halo, and exact mesh revision, so it cannot declare entry ready from a smaller or stale selection after spawn relocation. The protected closure contains 4 targets at step 1, 8 at step 2, 12 at step 4, 16 at step 8, and 20 at step 16. All 60 publish only after their matching FINAL parents and every internal shared boundary are ready together. Adjacent PREVIEW bridges may refine those coordinates while the FINAL closure is incomplete. They never weaken the atomic FINAL publication rule, and the final commit retires them through the ordinary frame-safe path. The first gameplay frame opens ordinary refinement and the single low-priority canopy worker.

Far terrain can refine through steps 16, 8, 4, 2, and 1. The bounded protected near closure requires FINAL step 1 beneath unresolved exact ownership, while ordinary visible terrain may also select FINAL step 1 when step 2 exceeds the screen-error target. A movement request retains its old published closure until the complete replacement closure is resident. Ordinary topology swaps may use the existing bounded fog transition, but fog may not cover a missing parent or a persistent crack.

The settled absolute bands are exact cubes through 32 chunks, step 2 through 64, step 4 through 128, step 8 through 256, and step 16 through 512. Step 32 is coverage-only. These bands are maximum-coarseness limits, not permission to discard visible detail. During gameplay, a conservative screen-space error estimate uses distance, viewport height, vertical FOV, and parent-tile relief to retain a finer tier while its projected geometric error exceeds 0.55 pixels. This refinement reaches step 1 wherever required. Step 1 is the irreducible voxel-grid floor, so its projected error may exceed 0.55 pixels immediately outside the exact handoff. Outward coarsening uses a 0.45-pixel threshold to avoid thrashing.

The bounded urgent lane orders the camera tile, protected handoff, exact fallback, and connected-wavefront classes before optional work. Within the nearby visible class, horizontal distance ranks before projected error. Screen-space error selects the desired tier but cannot let a farther tile delay closer missing detail. Camera-near critical work is a hard admission class, not a sorting hint. A missing protected PREVIEW step-32 parent may displace the worst queued or dependency-parked ordinary parent at the terrain cap. Current protected FINAL children and parents consume the first urgent capacity. At most one third of a cold frame's bounded submissions serve required PREVIEW bridge prerequisites, unused bridge capacity returns to current FINAL, and only remaining capacity may stage one predicted anchor. Prediction remains CPU-only and cannot enter GPU residency, display, or closure statistics before the canonical anchor changes. A camera move transfers protection instead of allowing work for the old anchor to retain it. CPU cache, upload, and GPU residency preserve the same class and may retire optional distant refinement or canopy to admit it. A requested protected FINAL role-selected key may use the complete GPU arena after that optional reclamation; alternate keys at the same coordinate do not inherit the exception. Distant work cannot evict a camera-critical job, cache entry, upload, or allocation. A camera-near PREVIEW bridge outranks distant FINAL refinement because it can reduce visible error immediately. The atomic 60-payload protected closure names its 4 step-1, 8 step-2, 12 step-4, 16 step-8, and 20 step-16 targets directly. Its 100 internal canonical boundaries must match before FINAL publication. Ordinary refinement retains adjacent bridge progression, including inside the pending protected closure and including a FINAL step-1 target when screen error requires it. Base-lineage PREVIEW steps 16, 8, 4, and 2 may temporarily reduce visible cell size while decoded authority is cold. Their displayed-error metric includes the measured revision-9 46-block maximum omitted residual, so visible FINAL replacements outrank flora and speculative work. A base promotion retains its PREVIEW source while a visible PREVIEW child depends on it, and a retired source routes the next bridge through FINAL authority. A coarse tier may preserve coverage while work is cold, but it cannot become the settled near surface after the required target and legal shell are resident.

After the connected 96-chunk prefix is drawable, unfinished exact publication through 32 chunks or
any connected visible desired-LOD miss pauses ordinary outer-parent submission and publication.
Near jobs proceed nearest-first and may displace queued or dependency-parked outer parents.
Displayed parents, the connected prefix, transition endpoints, exact fallbacks, and protected
lineage remain pinned.

Local far work admits 8 of the 16 workers alongside exact generation or meshing debt, 12 after exact
debt clears, and all 16 only after both exact and local debt clear. Four admitted workers remain
available to connected base coverage when it is queued. The single gameplay canopy worker remains
lower priority and independently bounded.

CPU-cache and GPU-arena pressure must not turn a ready camera-near refinement into a permanent coarse tile. Before rejecting that upload, residency maintenance reclaims the farthest optional non-displayed refinement or flora allocation. It never evicts a step-32 coverage parent, a displayed surface, either endpoint of an active transition, or the current protected replacement. The valid coarse parent remains drawable until the reclaimed space has accepted the replacement, so preemption cannot create a hole.

`RYCRAFT_WORLDGEN_OVERLAY=lod` colors exact terrain cyan and far steps 1, 2, 4, 8, 16, and 32 blue, green, yellow, orange, red, and purple. `RYCRAFT_WORLDGEN_OVERLAY=authority` colors FINAL surfaces green and PREVIEW surfaces magenta. Capture both views when diagnosing a crack, rectangular hole, or unexpected coarse tile.

### No skirts

Production far meshes must emit zero downward skirt quads. A vertical panel is not terrain and must not hide an LOD mismatch. Tests assert `skirtQuadCount == 0`, including negative coordinates and the supplied seed-42 handoff.

Every production far tile replaces its outer cell ring with a shared transition topology. Tile faces use the same canonical two-block boundary lattice, which contains every sample position used by the adjacent step-2 through step-32 tiers. The strip triangulates from those shared positions to each tile's coarse interior with positive winding. Positive-area terrain remains half-open to one tile, and no cross-tile vertical face is emitted. Interior closure geometry is legal only where two source terrain columns actually differ, never to bridge an LOD mismatch.

Automated coverage exercises all four edges, all four corner pairs, steps 2 through 32, negative coordinates, fixed topology budgets, positive winding, exact projected area, duplicate-triangle rejection, and mixed step-16 to step-8 neighbors. These tests establish the topology contract, but accepted visual qualification still requires inspecting every adjacent join for cracks, ledges, or false faces.

Legacy `skirtBottom`, bit-29, and skirt helper fields may remain in compatibility structures or tests, but production geometry may not create downward panels from them. Any future cleanup must preserve shader-layout compatibility deliberately.

### Canonical water geometry

Exact and far water consume one canonical hydrology authority. A render path may not infer a body only from terrain corners.

Far cell bounds carry `waterTopologyPossible`. At step 32, a possible narrow route or contour triggers bounded interior probes and recursive or contour-based recovery. This prevents a river or lake inlet from disappearing between dry corners. Half-open tile ownership applies to standing-water contours, river ribbons, falls, and transition geometry.

Water invariants:

- A standing body has one `WaterBodyId` and stage.
- Stable standing water emits planar top geometry.
- Exact implicit sources and far water use the same visible plane.
- Different body stages are not joined by one coarse polygon.
- An abrupt stage change requires explicit rapid or waterfall ownership.
- Vertical water faces are reserved for explicit falling columns.
- A partially wet coarse cell follows the canonical shoreline rather than becoming a rectangular sheet.
- Water must remain present at steps 2, 4, 8, 16, and 32 when topology crosses the cell.
- A wetland is a shallow fringe owned by an existing ocean, lake, or river body. It may connect only to that exact body ID and stage, never to an unrelated standing surface.

Analytic fragment normals, refraction, absorption, caustics, and reflections may animate appearance. They may not displace the ownership plane or fill a topology gap.

### Culling and winding

Exact and far opaque terrain uses outward counterclockwise winding and back-face culling. Water and shadow paths keep their established culling rules. Frustum culling runs before conservative front-to-back terrain-horizon culling.

A partially owned, partially faded, or transitioning patch is not a conservative occluder. Canopy bounds are visual bounds and must not make absent terrain occlude a farther mountain.

## 8. Partial water geometry

Water indices follow `opaqueIndexCount` in exact and far mesh allocations and draw in a dedicated pass. Exact runtime water obeys these rules:

- Source water has a top height of one block.
- Flow levels 1 through 7 descend in eighth-block steps from 0.875 to 0.125.
- Water with water directly above is full height.
- Every standing generated wet voxel from the lowest one above solid support through the surface is an implicit source, including across cube faces. Those cells require no explicit fluid array.
- Falling water is full height and sets the falling face-attribute bit.
- Four adjacent cell heights are averaged at each top corner.
- Flow direction follows the lowest horizontal neighboring surface and occupies bits 24 through 26.
- Stable source and flowing cells emit planar top geometry only, even at a shoreline or unloaded boundary.
- Vertical side geometry is emitted exclusively for explicit falling columns and reaches from the cell floor to its two matching corner heights.
- Water-to-water side geometry is absent, including across every cube face.
- A generated `OutletFall` overlays only its short receiver-centered footprint with explicit falling states. Its top, bottom, width, and flow do not replace the receiving body's standing `waterSurface`, and generation enqueues no runtime fluid tick.

Rendering and physics use the same `fluidSurfaceHeight` rules. A camera is underwater only below the actual local fluid height, and buoyancy tests the same state. Far water is intentionally a coarse contour-clipped sampled representation and does not participate in runtime fluid physics. Its generated source plane is nevertheless quantized to the same full-block height as the exact implicit source voxel. The water vertex shader does not displace stable surfaces. Filtered analytic fragment normals, caustics, and waterfall streaks provide movement without bending the source plane or changing ownership.

## 9. Water pass

Water renders after the opaque scene resolves into `_colorResolve`:

- The fragment shader samples level zero of a blit copy of resolved opaque color for refraction, uses its complete HDR mip pyramid for grazing SSR, and samples resolved opaque depth for camera-relative reconstruction.
- Refraction is a close, stable-receiver effect. A missing, distant, under-sampled, or grazing receiver falls back continuously to reflection instead of transmitting one coarse scene sample across a large water pane. The exterior sky gate uses the binary water-interface authority from bit 30 to distinguish sealed water from exterior water. It never turns fractional skylight into a reflection multiplier or raises propagated ambient light.
- The pass binds media depth, hardware-tests each water surface, and writes the nearest visible interface while sampling `_depthResolve`.
- The shader owns its composite pixel, so the surface pipeline has no blend state.
- Exact and far water draws sort back to front by full three-dimensional distance.
- Screen-space reflection, depth absorption, procedural caustics, Fresnel sky reflection, sun sparkle, and shared fog run in this pass when enabled.
- The underwater overlay, god rays, and caustics render last when the camera is submerged.
- Anything that belongs behind water renders in the scene pass. Anything intentionally above water renders afterward.
- **The wave field is one table.** `WATER_WAVES` in `shader_types.hpp` drives the filtered analytic fragment normal (`waterSurfaceNormal`), with the phase advected by the packed flow direction. Stable water geometry remains planar. Why: independent wave formulas once desynchronized the surface detail, and geometric displacement broke exact and far ownership boundaries.
- **The SSR march starts at an angle-adaptive IGN-jittered stride.** Nearby and non-grazing rays retain the narrow original range; long grazing rays reduce that jitter, sample a bounded lower HDR mip, and fade only their unstable tail to the analytic sky reflection. When the camera is submerged, reflected hits attenuate per channel by `WATER_SIGMA_A` over the reflected path. On a thick-occluder reject the ray keeps marching instead of falling back. Why: a coherent stride turned the coarse march into stair bands, while full-resolution stochastic far hits alternated with sky misses into a black-and-bright checker pattern at grazing angles. The fallback is preferable once screen-space depth has no stable reflected surface.
- **From below, the surface is physical.** Water-to-air Fresnel uses total internal reflection past the critical angle, eased near it so per-quad wave normals do not flip whole cells into hard panels. SSR mirrors the underwater scene with the deep tint as fallback. Foam, refraction distortion, and the floor-caustic add are above-water-only. Why: each of those painted above-water effects onto the from-below view, including white waterline streaks and misoriented caustic bands.
- **The Snell window transmits without absorption.** From below, the distance behind the surface is air (sky or shore), and the eye-to-surface water segment already belongs to the underwater overlay. Why: absorbing that air distance as if it were water saturated the whole window into opaque flat blue instead of a view of the world above.
- **The interface side is a per-fragment geometric fact.** The water pass classifies each fragment by the sign of dot(V, N) (an elevated lake's underside seen from dry land is still the water-to-air interface; the camera flag only decides which medium rays travel through) and evaluates Schlick Fresnel against the transmitted angle on the water side, where cosT reaching zero at the critical angle rises continuously into total internal reflection with no hand-tuned ease. Why: branching on the camera flag saturated dot(V, N) to zero for every submerged pixel, which read as past-critical total internal reflection everywhere and turned the whole surface into a permanent mirror that never transmitted.
- **The internal mirror reflects luminous water.** Where the underside SSR misses, the fallback is the shared water volume scatter terms (`WATER_SCATTER`, `WATER_AMBIENT`, the same constants the overlay inscatters with). Why: a near-black fallback read as flat dark panels, where a real internal mirror reflects the sunlit volume.
- **The sun glint obeys Fresnel.** The sparkle term multiplies by the same Fresnel factor as the sky reflection. Why: about two percent reflects at normal incidence, so an unscaled glint under a zenith sun mirrored in every up-facing wave and bloomed into one giant white blob on the surface.
- **Intrinsic water tints are lit responses, not emission.** The shallow and deep tints scale by the surface's sky access and the day-night sky level before mixing with refraction or reflection. Why: constant tints glowed teal through moonlit and covered water, drawing a bright ring on the lake floor around the camera where refraction still outweighed the dark night reflection.
- **Caustics track the waves that focus them.** The web's cell scale sits at the ripple wavelength, its arms are warped by the shared wave normal, a slow rotated modulator octave breaks the wrapped tile's exact periodicity, and the web defocuses into broad swell-scale patches with floor depth. Why: one wrapped octave repeated identical ~2-block cells across every floor, and a crisp fixed-depth web read as painted on rather than focused by the surface.
- **Stable water surfaces remain planar.** The canonical source plane, corner heights, and exact or far ownership never move with wave animation. Why: displaced crests could cross shoreline blocks and bend matching exact and far source planes apart, producing seams and unstable refraction receivers.

### The underwater overlay is physically based

When the camera is submerged, the fullscreen overlay owns the entire water tint (the scene passes apply no fog below the surface, and the rain sheen turns off):

- **Per-channel Beer-Lambert absorption through dual-source blending.** The fragment outputs inscatter at `color(0) index(0)` and per-channel transmittance at `index(1)`; the pipeline blends `result = inscatter + scene * transmit`. Why: a single alpha cannot express spectral absorption, and red must die faster than blue for distance to read as water rather than flat fog.
- **Absorption counts only the in-water path.** Upward rays stop accumulating at the water surface (`waterSurfaceY`, scanned up from the camera cell on the CPU), and the shaded point's own depth below the surface attenuates the light that reached it. Why: fogging by the opaque distance behind the surface (the sky is far) drowned every upward view in murk, and depth-independent lighting made deep floors look daylit.
- **Sunlight is gated by sky exposure.** Covered water (sealed aquifers, roofed lakes, checked against the surface-height map like rain spawning) zeroes the caustics, the sun-driven inscatter, and the submerged volumetric shafts. Why: the shadow cascades cannot occlude terrain hundreds of blocks up, so sealed pockets grew impossible sun caustics and shafts.
- **Caustics modulate, never add.** The web multiplies the transmittance, so it rides each floor's own shading; the pattern is the iterative wave-warped web (`causticPattern`, warped by the shared wave normal so light moves with the waves) and is clamped, because an unclamped HDR caustic crossed the bloom threshold across whole floors and whited them out.
- **Caustics require a stable, strictly upward screen-space normal.** Best-of-both-sides depth taps reconstruct the receiver, a silhouette feather rejects discontinuities, and only the positive world-up hemisphere passes. Why: using an absolute normal-Y value admitted walls, ceilings, and corrupted edge normals as false floors, painting moving vertical caustic bands onto submerged block sides.
- **Inscatter is anisotropic.** A capped Henyey-Greenstein lobe brightens the view toward the sun. Why: isotropic murk lost the underwater silver lining that makes the volume read as sunlit water.

## 10. HDR frame graph and the one tonemap

The scene renders in linear HDR and is graded exactly once.

1. Upload the frame's immutable weather snapshot, refresh slow atmosphere LUTs when their optical parameters change, refresh sky view, and generate the snapped cloud-shadow transmittance map.
2. Render selected depth targets for four detailed shadow cascades and one coarse horizon cascade.
3. Render the physical sky, exact terrain, far terrain, entities, and highlights into the 4x MSAA HDR scene, depth, and surface-data attachments, then resolve them.
4. Build the linear min-depth pyramid. Ray trace GTAO and one-bounce diffuse SSGI through it, accumulate with age and a variance clamp, denoise with edge-aware a-trous passes, and apply ambient irradiance.
5. March and temporally composite quarter-resolution volumetric clouds, then render depth-tested, cloud-aware lightning.
6. Copy opaque color and render water against copied HDR and resolved opaque depth while hardware-testing and writing media depth.
7. Inject and integrate the air froxel volume, temporally resolve its scattering and transmittance, and composite `scattering + scene * transmittance`. When volumetric lighting is disabled, apply atmosphere-LUT aerial perspective instead. Underwater absorption remains a separate medium.
8. Render depth-tested weather particles with atmospheric attenuation.
9. GPU compute updates persistent exposure and flare state without CPU readback. Exposure meters a highlight-weighted mean of log luminance with asymmetric adaptation (fast down, slow up). Why: a flat mean barely moves when the small bright sun enters the frame, so facing the sun never stopped the scene down; with no highlights the weighted mean equals the plain mean, so caves and night keep their lift.
10. Bloom builds its HDR pyramid when enabled. One always-on final composite applies exposure, the Hable filmic tonemap, vibrance, contrast, lens flare, optional CAS sharpening, and dithering. UI draws at display resolution.

Toggled-off effects skip work or bind static fallback textures. They do not fork the scene into an untonemapped path. Screen-space lighting, clouds, and froxels own explicit temporal histories and reject them after resize, teleport, world change, FOV discontinuity, quality change, forced time or weather change, or invalid prior depth.

The UI overlay pass on the single-sample LDR drawable draws three fixed z-phases per frame: solid-color base quads, then textured item icons, then solid-color top quads (stack counts, tooltips, carets). This ordering keeps counts and tooltips above every icon and the cursor-held stack above everything. The icon phase uses a second pipeline (`uiIconVertexMain`/`uiIconFragmentMain`) with its own ring buffer and the shared `UIIconVertex` layout in `shader_types.hpp` (size and offset asserted in `tests/test_render.mm`); it samples the block texture array, which carries item-icon layers appended after the block-face layers and addressed by a constexpr offset from the contiguous non-block `ItemType` range. When no world session is live the frame is a single clear-plus-overlay menu-only pass (`renderMenuOnly`) with no HDR, depth, or world reads. Dropped items and boats render inside the MSAA scene pass through siblings of the entity renderer that reuse the entity shader and vertex format, so they introduce no new GPU-shared struct and their pipelines declare both scene color attachments through `configureScenePassPipeline`. The boat renderer bakes one hull mesh once and draws it per boat with a yaw rotation.

## 11. Lighting, shadows, atmosphere, and weather response

### Directional light and shadow cascades

- One active direct radiance authority feeds terrain, entities, water highlights, shadows, clouds, and froxels. Sun radiance is zero when the solar disc is below the horizon. Moon radiance remains suppressed through civil twilight, fades in across nautical twilight, and follows a deterministic 708,734-tick mean synodic phase. The physical phase response scales diffuse lunar radiance and shadow strength. Water specular applies the phase response once more, so a hidden sun cannot remain reflected after sunset and a crescent cannot produce a full-Moon glint. Atmosphere and the solar-only flare retain true solar direction and visibility for physical twilight without reintroducing direct sunlight.
- `CelestialState` is the only time authority for direct receivers and the physical sky. The atmosphere may retain true solar direction after direct sunlight reaches zero, but it must use the same world time, solar elevation, weather snapshot, and exposure path as the rest of the frame. Clear daylight cannot present a dark or night-like sky above terrain receiving daylight illumination. Conversely, a night sky cannot coexist with stale daytime direct light. This is a frame-coherence requirement, not a style choice.
- Five cascade records use camera-forward view depth, matching their projections. High endpoints are 48, 160, 512, 1,536, and 8,192 blocks. Medium endpoints are 40, 128, 384, 768, and 8,192 blocks. The first two High targets are 4,096 square, the next two and horizon target are 2,048 square. Medium uses 2,048 square for the first two and 1,024 square for the rest.
- Every projection is texel-snapped in light-space X, Y, and depth. Adjacent cascades blend over the final 12.5 percent of the current range, including the detailed-to-horizon transition. The final 12.5 percent of the horizon cascade also fades to propagated exterior visibility, so 8,192-block coverage cannot end in a hard ring. Each record carries view-depth range, blend start, valid coverage, texel scale, normal bias, filter scale, and matrix.
- A shared Metal helper selects and filters terrain, entities, water-related shading, and volumetrics. The first two cascades use contact-hardening filtering, the farther detailed cascades use rotated 9-tap PCF, and the horizon map uses a stable 4-tap filter.
- The first two cascades refresh every frame. Cascades two, three, and the horizon refresh when snapped projection, caster revision, light direction, or receiver-depth coverage changes, or after maximum intervals of two, four, and eight frames. A skipped cascade keeps its last rendered matrix and matching depth texture. Its depth center has a bounded texel-scale guard, so current receivers cannot be sampled against a retained map whose depth range excludes them.
- Exact terrain, cutout foliage, entities, current far terrain, and current far canopies cast only from their displayed ownership. The same per-column exact and far masks prevent overlapping representations from double-casting.
- Outside valid shadow coverage, direct lighting fades toward propagated exterior visibility. It never becomes unconditional full light.

### Lighting composition and screen-space indirect light

- Cascaded visibility controls direct sun or moon only, then a sky-access cap (`smoothstep(0.5, 0.9, propagated skylight)`, full sun only within about one block of a genuine sky path) removes direct light from covered cells. Why: inside cascade coverage the shadow map alone leaked stray sun onto opaque-covered surfaces, painting moving bands across dug tunnel walls; the earlier near-zero threshold left the leak visible everywhere propagated skylight was nonzero. Non-opaque cover keeps skylight 15, so the cap never adds a second shadow under a leaf canopy, and interiors receive light through the unrestricted indirect bounce instead. Propagated skylight and baked corner accessibility control ambient irradiance. Block light and emissive radiance remain independent and are never multiplied by skylight, baked accessibility, GTAO, or SSGI.
- The resolved surface attachment stores diffuse albedo and baked ambient accessibility. Direct, block, and emissive HDR remain in the scene source.
- High indirect lighting ray traces four cosine-weighted screen-space rays per half-resolution pixel through a stackless min-depth Hi-Z march capped at 24 iterations, with a 24-block bounce reach and an 8-block GTAO radius. Medium traces two rays capped at 16 iterations at quarter resolution with a 16-block reach. Off still applies the inexpensive ambient pass without GTAO or bounced radiance.
- The additive one-bounce is gated in the apply pass (only, so temporal history stays in un-gated units and needs no reset) by the day-night sky level (`filterParams.w`, 1 in daylight and ramping to 0 through twilight), so night auto-exposure cannot amplify the near-field bounce on open ground; a small floor keeps HDR emissive spill readable. It is deliberately NOT gated by the receiver's sky access: filling caves and tunnels from their genuinely lit spots is the point of indirect light, and the camera-following interior artifacts were direct-sun shadow-map leaks now capped at the source in the terrain pass. The GTAO/ambient-correction term is already scaled by the night ambient and is left alone. Water tints fade by the same celestial signal so terrain indirect and night water darken together. Why: a sky-access gate on the bounce made caves with bright openings read absurdly dark, while the night disk and ring came from the missing day-night scaling and the water tint floor.
- The min-depth pyramid exists solely to accelerate ray traversal; bounce radiance samples the scene source at the exact hit texel. Temporal accumulation tracks per-pixel age and luminance moments: the blend weight ramps with age to a 0.9 cap, a hue-preserving firefly clamp bounds each raw sample, and reprojected history clamps to the spatial mean within a variance-scaled range. A stale bright ghost over a converged neighborhood collapses to the clamp floor within a frame, while a genuinely sparse bright source survives through its accumulated variance. Why: an unclamped sparse-history special case let bright ghost splotches smear across walls for seconds after their source left the frame.
- Edge-aware a-trous wavelet passes denoise the accumulated result after the temporal pass, guided by linear depth, guide normals, and variance, with three iterations on High and two on Medium. History feedback stays pre-blur, and a young-age variance floor opens the filter on disocclusion so fresh regions fill with a smooth spatial estimate instead of black or speckle.
- Indirect history resets after resize, teleport, world change, FOV discontinuity, quality change, forced time or weather change, or invalid prior depth. Cloud and froxel histories follow the same discontinuity authority.
- SSGI is near-field and screen-space. Offscreen geometry cannot contribute colored bounce. Propagated skylight remains the view-independent ambient authority for caves and overhangs.

### Physical atmosphere, clouds, and froxels

- Daylight comes from an Earth-like LUT atmosphere, not a stylized daytime gradient. The renderer uses a 256 by 64 transmittance LUT, a 32 by 32 multiple-scattering LUT, and a 192 by 108 sky-view LUT with Rayleigh, Mie, ozone, altitude response, physical solar angular radius, and weather aerosols. Directional attenuation comes from the later volumetric cloud composite instead of a camera-local coverage scalar, so clear gaps remain physically bright and cloudy pixels are not darkened twice. Stars and the phase-shaped moon remain night overlays.
- `CloudRenderer` builds deterministic 128-cubed Perlin-Worley base noise, 32-cubed erosion noise, and 2D curl noise on one cancellable utility worker after world entry. The render thread uploads only a completed payload whose world instance and seed still match, and cloud quality Off starts no noise work or allocation. Cancellation is distinct from failure. A failed build retains its message, count, age, duration, and next retry time, retries after 100 and 500 milliseconds, then latches after the third failure until quality is re-enabled or a new world is bound. Each failed attempt is logged once. Weather blends stratus, cumulus, cumulonimbus, and cirrus profiles. Beer-Lambert extinction, dual-lobe phase scattering, ground and sky irradiance, short sun transmittance marches, erosion, and bounded multiple-scattering compensation define lighting.
- Cloud motion uses weather wind in blocks per second with independent layer response and wrapped double-precision offsets. High and Medium both render true quarter-resolution volumetric clouds. High uses 48 view steps and 6 light steps; Medium uses 24 and 3.
- Cloud color and hit depth use ping-pong temporal histories with invalid-sample rejection and neighborhood clamping, then upscale bilaterally against resolved depth. The snapped cloud-shadow map covers a 16,384-block footprint at 2,048 square on High and 1,024 square on Medium. Terrain, entities, volumetrics, the sun disc, and flare visibility consume the same transmittance authority.
- The air medium uses a 160 by 104 by 64 frustum-aligned froxel grid with logarithmic depth slices. It injects aerosols, humidity, precipitation fog, atmospheric extinction, active directional scattering, blended terrain shadows, and cloud shadows. Half-resolution scattering and transmittance are temporally reprojected before composition.
- Underwater absorption remains separate. The air medium is gated at the water surface and for a submerged camera so air fog cannot leak into water. Disabling volumetric lighting retains atmosphere-LUT aerial perspective but omits froxel shafts.

### Weather and storms

- One immutable weather snapshot feeds terrain, entities, foliage, particles, clouds, atmosphere, froxels, wetness, lightning, and audio for the frame. Wind is expressed in blocks per second everywhere. Rain or snow follows temperature rather than biome. Particle billboard size remains in blocks, while its Beer-Lambert view distance converts to meters through the active world's physical scale.
- Lightning IDs, positions, branches, and flashes derive from world seed, storm cell, and fixed time bucket. Lightning is depth-tested and cloud-aware, changes no blocks, and creates no fire. Thunder is procedural, bounded, de-duplicated, and delayed by physical strike distance at 343 meters per second.
- `RYCRAFT_WEATHER` accepts stable `clear`, `overcast`, `rain`, `storm`, and `snow` presets. Forced time or weather invalidates every affected temporal history.
- Lava remains emissive in linear HDR and seeds derived block light into neighboring transparent cells.

The implementation is informed by [Practical Realtime Strategies for Accurate Indirect Occlusion](https://research.activision.com/publications/archives/atvi-tr-16-01practical-realtime-strategies-for-accurate-indirect-occlusion), [Screen-Space Diffuse Global Illumination](https://pure.mpg.de/pubman/item/item_1324270), [Stochastic Screen-Space Reflections](https://www.ea.com/frostbite/news/stochastic-screen-space-reflections), [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced-global), [Cascaded Shadow Maps](https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps), [Nubis volumetric cloudscapes](https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine), [Frostbite unified volumetrics](https://advances.realtimerendering.com/s2015/), and [production-ready atmosphere rendering](https://sebh.github.io/publications/egsr2020.pdf). Rycraft uses bounded Metal implementations of those principles, not verbatim copies of any production renderer.

The celestial response follows production patterns documented by the [Sildur's Vibrant Shaders changelog](https://sildurs-shaders.github.io/changelogs/) and visible in the current implementations of [Complementary Reimagined phase influence](https://github.com/ComplementaryDevelopment/ComplementaryReimagined/blob/08e1c2ada5eaf2fc36f08516c316b3d3c3677d8e/shaders/lib/colors/moonPhaseInfluence.glsl), [Complementary Reimagined light selection](https://github.com/ComplementaryDevelopment/ComplementaryReimagined/blob/08e1c2ada5eaf2fc36f08516c316b3d3c3677d8e/shaders/lib/colors/lightAndAmbientColors.glsl), and [Bliss direct-light selection](https://github.com/X0nk/Bliss-Shader/blob/81e403ed308141039a09d792a36f8eb328898a60/shaders/dimensions/composite.vsh). Those shaders gate the active source around the horizon, keep moonlight much dimmer than sunlight, apply lunar phase to lighting or reflections, and reuse the selected light direction across direct receivers. Rycraft centralizes direct-source decisions in one CPU state and keeps true solar state explicit for atmosphere and solar flare, so the two roles cannot silently diverge.

The orbital geometry follows [NASA's Moon phase explanation](https://science.nasa.gov/moon/moon-phases/) and uses the mean 29.53059-day synodic period published by the [U.S. Naval Observatory](https://aa.usno.navy.mil/faq/moon_phases), rounded to one deterministic world tick.

## Verification is part of rendering work

Run Metal API and shader validation separately from performance:

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft
```

Capture and open all of the following after queues settle:

- Exact terrain at the v4 safe spawn
- Exact-to-far handoff
- Step 2, 4, 8, 16, and 32 terrain joins
- A narrow river crossing a step-32 cell whose corners are dry
- Lake, coast, waterfall, and elevated-to-lower transition views, plus delta views when an
  implemented authority exposes them
- A cold horizon before and after canopy enrichment
- A blocked-canopy test scene with visible terrain and water
- High terrain near the expanded ceiling and low terrain near Y=-128
- A broad turn that exposes newly selected parent tiles
- A supported floor torch in darkness, the single fixed front of an active and inactive furnace pair, the other nonemissive furnace faces, a chest front, lava, and a low bed from every exposed face, with indirect light both enabled and disabled. `RYCRAFT_SPAWN_MATERIALS=1` creates the block lineup only when `RYCRAFT_CAPTURE` also selects the disposable no-save path.
- Rain and snow crossing an opaque silhouette and a reflective water surface
- Settled and moving views through all four detailed shadow cascades and the horizon cascade, with no ownership double-cast, stale projection, or blend ring
- Clear, overcast, rain, storm, and snow at dawn, noon, dusk, and night, including temperature-based rain-to-snow selection, shared foliage and particle wind, wetness, cloud shadows, fog, and aerosol response
- Views below, inside, and above stratus, cumulus, cumulonimbus, and cirrus layers, including mountain intersections and motion long enough to expose noise tiling or temporal trails
- Lightning in front of and behind clouds, repeatable event IDs, atmospheric flashes, delayed thunder, and no world or fluid mutation
- Sun and moon transitions through civil twilight and representative lunar phases, with one exclusive direct-light source shared by terrain, water, clouds, froxels, and shadows
- Physical atmosphere at dawn, noon, dusk, and night, with finite LUTs, stable horizon luminance, weather aerosol response, and no dark sky over daylight-lit terrain
- A stable indirect-history view before and after a block-light edit, teleport, FOV change, sleep time jump, and far-only refinement

Inspect for gaps, vertical panels, ledges, straight shoreline runs, cardinal sawteeth, missing narrow water, duplicate water surfaces, unrelated joined levels, canopy-triggered horizon holes, false occlusion, indirect-light leaks through closed walls, stale temporal ghosts, emission across a torch stick or furnace shell, emissive beds, and weather particles entering the opaque material history.

Record exact model revision, generation fingerprint, seed, camera coordinates, LOD step, viewport, graphics settings, frame number, queue state, and validation-message count for each accepted capture.

## Review checklist

- Does bootstrap UI render without constructing a world?
- Do shared C++ and MSL layouts still match exactly?
- Do all height paths accept Y=-128 through Y=1407?
- Does exact ownership use the destination column for every fragment type?
- Do generation, mesh admission, and upload prioritize every required surface section through the
  full 32-chunk exact disk, with camera and exploration work first?
- Does exact collision publish only from the matching coverage epoch and otherwise use canonical
  generated terrain and fluid proxies?
- Can any first-visible exact mesh bypass a pending bounded lighting transaction?
- Can base terrain and water upload before canopy construction?
- Can a canopy attachment invalidate, replace, or delay a resident parent?
- Does canopy remain at zero during entry preparation and guarantee exactly one bounded gameplay
  worker after the connected 96-chunk prefix is drawable, without a second lane?
- Does step-32 water use topology probes when corners are dry?
- Are water identities and stages preserved across every LOD?
- Is every production `skirtQuadCount` zero?
- Did any change reintroduce a downward panel under another name?
- Do displayed neighbors, including active transition endpoints, stay within a 2:1 step ratio?
- Do shared strips use identical boundary samples with half-open positive-area ownership?
- Are parents retained until replacements are resident?
- Can urgent protected PREVIEW coverage displace lower-ranked queued or parked ordinary coverage?
- Does exact or connected desired-LOD debt pause ordinary outer submission and publication, and can
  nearer work displace queued outer parents without evicting structural owners?
- Does the pause cover both outer submission and outer publication, with nearby visible distance
  ranked before projected error?
- Do grouped protected FINAL owner crops exactly match independent owner requests, and does
  `deferredBuilds` count completed typed deferrals rather than parked or active work?
- Does current protected FINAL work precede bridge and predicted work?
- Can only the requested protected FINAL key reclaim optional distant residency and use the complete
  GPU arena while structural owners remain pinned?
- Do every opaque terrain, entity, item, and boat pipeline declare the same HDR, surface, reactive, depth, and sample-count contract?
- Does the 4x tile resolve average HDR and choose material and reactive data from the minimum-depth sample through a memoryless R32 key while resolved HDR alpha remains one? Does the single-sample contract bypass that tile work?
- Do sky and highlight paths leave the surface and reactive attachments untouched?
- Does the Hi-Z pass run after opaque resolve and before clouds, weather particles, water, volumetrics, exposure, and bloom?
- Do clouds, rain, and snow use resolved-depth testing without a surface attachment?
- Are emission masks filtered across all five mips and limited to lava, torch flames, and exactly one fixed -Z active furnace mouth?
- Do low beds and inactive furnaces remain nonemissive receivers with authored culling and baked corner accessibility?
- Do bed and torch raycasts ignore empty voxel space and draw selection outlines around the same authored bounds used for interaction?
- Do every documented history discontinuity reset temporal state while far refinement and canopy arrival preserve it?
- Do five shadow records retain their depth splits, blend bands, cached matrix and texture pairs, and one, one, two, four, and eight-frame maximum refresh intervals?
- Do atmosphere LUT dimensions, cloud noise and march bounds, froxel dimensions, weather snapshot ownership, physical wind units, and underwater air-medium gating match their contracts?
- Are lightning and thunder deterministic, bounded, delayed by distance, non-destructive, and free of file I/O on admission?
- Does one active sun or moon authority prevent duplicate direct shading and below-horizon sunlight across shadows, water, clouds, and froxels while true solar state remains explicit for atmosphere and flare?
- Do water-only edits preserve indirect history while opacity, lava, torch, and active-furnace changes reset it?
- Do animals, dropped items, and boats use cached packed lighting, receive and cast cascaded shadows, and reject temporal history around motion?
- Were Metal validation logs checked and captures opened?
