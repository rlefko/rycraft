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
| 11 through 14 | Skylight |
| 15 through 16 | Baked corner ambient occlusion |
| 17 through 20 | Block light |
| 21 | Emissive flag |
| 22 through 23 | Wind sway class |
| 24 through 26 | Water flow direction |
| 27 | Falling water |
| 28 | Far-canopy impostor marker |
| 29 | Far-terrain boundary-skirt marker |
| 30 | Water exterior-sky authority |
| 31 | Reserved |

Renderer metadata must stay in the assigned high bits unless the vertex descriptor, shaders, tests, arena budgets, and every producer change together. Bit 28 identifies visual-only far canopy geometry so the fragment shader can suppress it in exact-owned columns. Bit 29 identifies far boundary skirts so displayed-neighbor state and paired emitting-and-receiving column ownership can suppress unnecessary walls. Bit 30 is a binary water-interface classification derived from propagated skylight or the complete edited column cutoff. It must not alter ambient skylight, and it prevents incomplete streaming authority from making exact water disagree with far water.

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
- Exact mesh residency cannot exceed 16,384. The far CPU cache cannot exceed 9,280 entries or 3 GiB. The far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertices and 1 GiB of indices.
- All renderer targets, arenas, caches, world state, transient work, and Metal allocations together must remain below the 64 GB unified-memory acceptance ceiling.
- Persistent High-tier allocations added for the integrated scene targets, shadows, indirect lighting, atmosphere, clouds, lightning, and froxels must remain at or below 768 MiB. At native 3456 by 2234, the byte-accounted High contract is 715,822,207 bytes, leaving 89,484,161 bytes below that ceiling. The accounting includes the water reflection's complete HDR mip pyramid, a full-resolution RG16F SSGI normal guide, the SSGI luminance-moments and age history pair, and R32 froxel linear-depth histories, but excludes memoryless MSAA attachments and device-specific page alignment. Report resized histories and shadow groups once, not once per frame.
- The procedural block-texture array has a complete five-level mip chain from 16 by 16 through 1 by 1. Alpha-aware downsampling preserves representable cutout coverage. Its sampler uses nearest magnification, linear minification, linear mip interpolation, repeat addressing, and 8x anisotropy.

## 4. Coordinate conventions and precision

- Matrices are column-major with column vectors, a right-handed world, and Metal depth in [0, 1]. `perspective`, `lookAt`, frustum extraction, and CPU-to-MSL copies remain covered by tests.
- Exact vertices are local to one 16 by 16 by 16 cube. Valid exact block geometry lies in 0 through 16 on X, Y, and Z. `ChunkOrigin` restores `chunkX * 16`, `chunkY * 16`, and `chunkZ * 16` in the vertex shader.
- Far vertices are local to one 256 by 256-block tile in X and Z. Their Y remains relative to world Y=0. The far `ChunkOrigin` restores the tile's 64-bit CPU origin at draw time.
- Never bake world X or Z directly into half-precision vertices. Exact local half values preserve partial-fluid geometry. Far geometry accepts the local half precision appropriate to its two-, four-, eight-, sixteen-, or thirty-two-block sample step.
- Face planes sit on block boundaries. The negative X face of local block `x` is at `x`; its positive X face is at `x + 1`. The same rule applies on Y and Z.
- Exact cube AABBs, frustum bounds, camera distance, candidate priority, opaque origins, entity relationships, and water sorting include Y.
- Far tile AABBs use sampled surface minimum and maximum Y for visibility. The full tile bound remains available for skirt and allocation ownership.
- Water reconstructs resolved depth in a camera-relative world frame. Remove the camera translation before inverting view-projection, compare surface and floor positions in that frame, and add the absolute camera position only after depth math for world-anchored caustics. An absolute inverse view-projection loses visible precision at large coordinates and can reset absorption at cubic chunk faces.

Exact sub-block geometry uses binary-exact fractions. Flora insets and lily-pad height use 0.125. Fluid cell heights use eighth blocks, and four-cell corner smoothing can produce multiples of 0.03125. Both remain exactly representable through the exact cube-local range.

## 5. Exact mesh snapshots and six-face seams

The game meshes exact terrain from `MeshSnapshot`, not directly from a mutable cube. A snapshot contains the 16-cube-edge interior plus a one-block halo, for 18 by 18 by 18 block, fluid, and packed voxel-light data. The high light nibble is 4-bit skylight and the low nibble is 4-bit block light. Separate 18 by 18 arrays retain generated-surface and complete-column sky-cutoff authority. A valid loaded cutoff may raise or lower the generated surface. The incomplete-path sentinel is distinct from every valid cutoff, including the world-ceiling cutoff.

- Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real face, edge, and corner samples. An unavailable in-range halo follows the immutable generated surface cutoff. Cells above the silhouette remain air and cells below it remain opaque. When a lateral cap candidate exists, a bounded six-connected flood marks transparent cells reachable from sky-exposed cells within the padded snapshot. A complete loaded cutoff honors added roofs and removed surfaces. An incomplete vertical path uses generated authority for this provisional classification while remaining fully occluded for ordinary skylight. Sky-connected lateral openings emit normally lit provisional faces using one representative arriving-surface material per missing face. Enclosed lateral openings emit unlit stone, and missing vertical openings emit bedrock. These faces are temporary boundaries, not generated world content.
- Solid faces, transparent faces, packed skylight and block light, flora ownership, ambient occlusion, fluid corners, and explicit falling sides read real data whenever the corresponding neighbor is loaded. Loading or unloading any halo cube reconciles the affected light faces and dirties only neighboring meshes whose sampled halo or boundary light changed.
- Edits dirty the owning mesh and every face, edge, or corner neighbor mesh whose one-block halo intersects the changed block.
- Water corner smoothing at an X/Z edge reads the diagonal cube when it is loaded. A missing diagonal is the same closed opaque boundary on both participating meshes.
- Complete full-height column authority seeds level-15 skylight only where the path to the sky is unobstructed. `LightEngine::floodChunk` propagates both light nibbles through transparent cells across all six faces and loses one level at each non-seeded step. Generated cutoffs remain seed and provisional-boundary authority, not a final binary light value. An incomplete vertical path remains fully dark, so an unloaded section above an underground view cannot admit sunlight.
- At the vertical world floor, missing halo cells are bedrock. Above the ceiling they are air.
- Test positive and negative X, Y, and Z separately. A horizontal-only seam test cannot catch a wrong cube Y origin or top and bottom halo defect.

Exact opaque cube faces use greedy merging. The merge key includes block and texture identity, face direction, skylight, block light, emissive state, sway state, and the four 2-bit ambient-accessibility corners. A quad splits along its brighter diagonal when corners are not planar.

## 6. Block shapes, winding, and back-face culling

`BlockDefinition::renderShape` is exhaustive:

- `CUBE` participates in greedy opaque meshing.
- `CROSS` emits two diagonal, inset flora planes.
- `FLAT` emits a horizontal plane, currently used by lily pads.
- `LIQUID` enters the water section.
- `NONE` emits no geometry.

Main-pass exact cube faces and far terrain use counterclockwise outward winding with back-face culling. Cross and flat flora emit both windings so they remain visible from either side. Water uses cull-none because surfaces are visible from below. Shadow casters also use cull-none because visible-face-only greedy meshes are not closed solids.

Alpha-cutout leaves, glass, flora, and their shadow casters use texture alpha discard. The first two cascades refresh every frame and call the same shared wind-sway helper as the scene. Cascades two through four intentionally hold foliage casters static while their retained maps defer, preventing a visible 2/4/8-frame shadow-phase pop at distance.

## 7. Far-terrain LOD and visibility

The far renderer selects every immutable 256 by 256-block tile intersecting the visible radius-512 disk, including tiles wholly inside the nominal exact radius. Every coordinate requests a step-32 parent before optional refinement. Missing parents enter a broad nearest-first job and upload lane. Each connected nearby coordinate then gives its distance-selected step-16, step-8, step-4, or step-2 target one bounded lane before the complete parent disk is resident. While parents remain queued, eight-worker dispatch reserves four slots for coverage and admits at most four urgent connected refinements. Already running parent work is not preempted. A resident parent remains pinned while its coordinate is active. Cache pressure evicts the farthest refinement first.

Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks the nearest missing step-32 dependency. The drawable frontier also treats a protected exact-loading tile with only step 32 ready as missing. Tiles and fragments at or beyond the drawable frontier are suppressed, and the preceding 256 blocks fade into frame fog. A partially faded patch cannot become an occluder. Every far-owned fragment in the exact overlap remains protected, including a fully ready boundary tile whose exact requirements cover only part of the tile. The drawable frontier advances only after the camera exploration band has step 2 and every other protected overlap tile has step 8 or finer. This prevents out-of-order completion from exposing a disconnected distant island through a protected base-only tile.

Exact ownership comes from `ExactSurfaceCoverageSnapshot`, not the configured radius alone. One 256-bit mask describes the 16 by 16 exact chunk columns overlapping each far tile. A bit is set only when all requirements currently published for that column are owned by exact meshes; empty completed meshes count as ready, while missing and unresolved requirements keep the column far-owned. A previously published exact mesh can retain ownership while an ordinary replacement is pending. Each far draw binds the center mask and all eight neighboring tile masks so a canopy crown or waterfall crossing a tile face queries its destination column. Exact opaque terrain draws first, and overlapping far terrain uses a small positive depth bias so resident exact surfaces win. Far ownership does not make step 32 displayable anywhere inside the exact overlap, even in a fully ready partial boundary tile. A partially masked patch cannot contribute to the terrain-horizon occluder. The nearest-gap handoff distance remains a conservative parent-selection fallback and diagnostic, not a radial fragment-ownership boundary.

Tile construction requests one-, two-, four-, eight-, sixteen-, or thirty-two-block `SurfaceFootprint` values through one `FarSurfaceSample` contract. The contract carries filtered terrain and water, conservative footprint bounds, and a compact material palette. Hydrology topology, water-body identity, water elevation, plate ownership, and feature anchors do not vary by footprint. Step-32 coverage geometry uses conservative minima so a parent cannot protrude through resident exact terrain. Material resolves once per active LOD cell, and greedy meshing joins only equal resolved materials. There is no aligned 32- or 64-block material cache.

Tile construction emits terrain tops, body-aware contour-clipped standing-water tops, explicit outlet-fall prisms, boundary skirts, and visual-only canopy impostors. Far water carries stable body identity and water kind. A coarse cell observing incompatible standing-water authorities refines from canonical samples rather than joining their levels. Bit 29 marks every boundary-skirt vertex. Per-draw metadata enables an edge only when its displayed neighbor is resident at a coarser step, and the fragment shader samples exact ownership on both sides of that far-LOD join. Far-LOD skirts are separate from exact missing-halo closure. Exact lateral openings already emit either a lit planned surface continuation or a dark inward cap, and exact vertical openings emit bedrock caps until the halo arrives. Partially wet cells use clipped shoreline triangles rather than rectangular sheets. The far mesh represents the visible top of a generated standing-water volume whose exact column is source-filled from its solid support through the surface, and it places that top on the exact full-block source plane.

A lake outlet fall is not part of the receiving body's standing-water mesh. Its immutable sample carries top and bottom surfaces, width, normalized flow, and one receiver anchor. The half-open tile containing that anchor owns one complete narrow prism centered on the receiver, even if its footprint crosses a tile face. The prism has four vertical sides and one top, extends into the lower body's top source voxel so it overlaps the visible water plane, reaches the upper lip, and does not raise or cover the receiving surface. Neighbor tiles emit no duplicate prism.

Step-2 far canopies reuse accepted exact tree anchors, species, and dimensions without constructing ColumnPlans. Steps 4 through 32 use globally anchored 64-block aggregate forest cells with six fixed candidates and block-8 habitat and ground authority. One stable rank makes those aggregate tiers strict subsets. The collector's block-resolution habitat and root-water decision remains authoritative at step 32, so water elsewhere in the coarse 32 by 32 cell cannot suppress an accepted canopy. The trunk grounds on the displayed voxel. Half-open cells and tiles retain unique ownership. Bit 28 marks every impostor vertex for the shared vertex contract and diagnostics. The per-column exact mask clips each canopy fragment only when its destination exact column is owned. The 3 by 3 neighboring mask set covers crowns crossing tile faces. During a far topology transition, target canopies appear monotonically during the first half before source canopies retire monotonically during the second half. This overlap-safe exchange also handles the intentionally unrelated step-2 and aggregate anchor sets. Terrain, water, and canopy geometry currently share one cold-build and residency payload. Measured cold canopy work ranges from 250 to 1,165 milliseconds and can delay publication of an otherwise ready terrain and water parent. Staged canopy attachment remains a follow-up.

Far tiles do not carry caves, structures, per-block flora, entities, collision, edits, runtime fluid state, save ownership, or exact biome transition detail. Canopy impostors are immutable visual summaries only.

Exact opaque terrain draws first and shares the HDR scene and depth attachments with far opaque terrain. Outside protected exact-loading tiles, biased parent tops remain behind exact depth and can fill cold-residency gaps after those parents are resident. Every far-owned overlap fragment is protected. It requires step 2 in the camera exploration band or step 8 or finer elsewhere in the exact overlap, including within a fully ready partial boundary tile. Water samples resolved opaque depth while hardware-testing and writing the nearest visible interface through media depth, so water and canopies use the same current per-column masks as opaque far terrain. Far water is appended to the same back-to-front water list as exact water. An ordinary far LOD replacement submits only the source water until completion, guaranteeing one refractive owner for every tile.

Known performance limitation: terrain, water, and canopy construction remains synchronous within one far-tile payload. Measured canopy work ranges from 250 to 1,165 milliseconds on cold tiles, delaying parent residency even when terrain and water are ready. Staged canopy attachment remains a follow-up, so do not describe the two-second cold-horizon target as accepted until that work and its reference-route validation are complete. Exact missing-halo faces already use explicit lit, dark, or bedrock closure caps and are invalidated when real halo data arrives.

Visibility is conservative and ordered:

1. Reject tiles outside the circular visible horizon.
2. Require the coordinate's step-32 parent, apply the protected step-2 or step-8 display floor, and suppress coordinates beyond the drawable coverage gap.
3. Select a displayed refinement from distance, retained immutable complexity, and the previous-tier hysteresis state.
4. Reject the tile's sampled-surface AABB outside the view frustum.
5. Sort survivors front to back.
6. Reject a tile only when every fully covered bin in a 256-bin azimuth horizon has a nearer lower-bound horizon above that tile's maximum elevation angle.

Each visible tile contributes sixteen 64 by 64-block terrain patches to the horizon. A patch uses its minimum sampled height, so it cannot claim coverage above any part of its continuous heightfield. Candidate maxima and occluder minima choose the near or far distance endpoint according to the sign of the vertical delta, which prevents false rejection from high viewpoints. The culler iterates only fully covered fixed bins and allocates no heap storage per tile.

An ordinary coordinate keeps its displayed tier until the next staged replacement is resident. Reentering a coordinate whose selected refinement and parent are already resident initializes directly to that refinement instead of displaying the parent again. Protected exact-loading requests lead urgent scheduling. Every far-owned camera-exploration-band fragment requires step 2, and every other far-owned fragment in the exact overlap requires step 8 or finer. Fully ready partial boundary tiles remain protected because their nonrequired fragments are still far-owned. Their step-32 parents remain resident but hidden. Protected results publish directly when ready and bypass ordinary grace and the 64-transition admission cap. A refinement still cannot display before its coordinate's parent is resident. Ordinary replacements perform the 0.65-second monotonic canopy exchange. Production fixtures prove that filtered terrain tiers can cross by as many as five blocks, so ordinary terrain does not use a partial source and target mix. A narrow terrain-only fog pulse hides each ordinary atomic topology swap, with normal far-tier terrain swapping at the temporal midpoint. Canopies use the full target-in, source-out two-phase exchange, and unswayed world coordinates keep wind from changing transition or coverage cells. Union bounds keep both ordinary topologies in the frustum, transition geometry does not become a horizon occluder, and a transition finishes before the desired tier is reevaluated. Skirts swap with the complete terrain topology that is actually visible, while source water remains the only water draw until completion. This preserves the voxel silhouette and 16-byte vertex ABI without a slope morph or extra vertex stream.

The steady-state render path reuses pre-reserved candidate, progressive-request, key, cache-result, upload, and 4,096-entry flat grace-record buffers. Tier promotion uses fixed counters, and draw culling uses fixed horizon storage. Far mesh construction, cache priority rebuilds, and cache retirement remain on utility workers. The connected refinement path adds no per-tile heap allocation to command encoding after renderer reset.

The implementation is an adaptive tiled LOD inspired by [Geometry Clipmaps](https://hhoppe.com/geomclipmap.pdf) and [CDLOD](https://doi.org/10.1080/2151237X.2009.10129287). It is not a literal geometry clipmap. The angular test is informed by conservative front-to-back occlusion work such as [Hierarchical Z-Buffer Visibility](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/greene93_hierarchicalz.pdf), but it is not HZB and owns no depth pyramid. The renderer's only depth pyramid belongs to the screen-space indirect pass, which uses its min-depth hierarchy solely for ray traversal, never for visibility culling or draw submission.

The renderer uses bounded direct indexed tile draws. It does not use the indirect command-buffer path described in Apple's [indirect command buffer documentation](https://developer.apple.com/documentation/metal/creating-an-indirect-command-buffer). Command-buffer and frame lifetime follow Apple's [Metal command-buffer best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html).

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
- **From below, the surface is physical.** Water-to-air Fresnel with total internal reflection past the critical angle (eased near it so per-quad wave normals do not flip whole cells into hard panels); SSR mirrors the underwater scene with the deep tint as fallback; foam, refraction distortion, and the floor-caustic add are above-water-only. Why: each of those painted above-water effects onto the from-below view, including white waterline streaks and mis-oriented caustic bands.
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

- Cascaded visibility controls direct sun or moon only. Propagated skylight and baked corner accessibility control ambient irradiance. Block light and emissive radiance remain independent and are never multiplied by skylight, baked accessibility, GTAO, or SSGI.
- The resolved surface attachment stores diffuse albedo and baked ambient accessibility. Direct, block, and emissive HDR remain in the scene source.
- High indirect lighting ray traces four cosine-weighted screen-space rays per half-resolution pixel through a stackless min-depth Hi-Z march capped at 24 iterations, with a 24-block bounce reach and an 8-block GTAO radius. Medium traces two rays capped at 16 iterations at quarter resolution with a 16-block reach. Off still applies the inexpensive ambient pass without GTAO or bounced radiance.
- The min-depth pyramid exists solely to accelerate ray traversal; bounce radiance samples the scene source at the exact hit texel. Temporal accumulation tracks per-pixel age and luminance moments: the blend weight ramps with age to a 0.9 cap, a hue-preserving firefly clamp bounds each raw sample, and reprojected history clamps to the spatial mean within a variance-scaled range. A stale bright ghost over a converged neighborhood collapses to the clamp floor within a frame, while a genuinely sparse bright source survives through its accumulated variance. Why: an unclamped sparse-history special case let bright ghost splotches smear across walls for seconds after their source left the frame.
- Edge-aware a-trous wavelet passes denoise the accumulated result after the temporal pass, guided by linear depth, guide normals, and variance, with three iterations on High and two on Medium. History feedback stays pre-blur, and a young-age variance floor opens the filter on disocclusion so fresh regions fill with a smooth spatial estimate instead of black or speckle.
- Indirect history resets after resize, teleport, world change, FOV discontinuity, quality change, forced time or weather change, or invalid prior depth. Cloud and froxel histories follow the same discontinuity authority.
- SSGI is near-field and screen-space. Offscreen geometry cannot contribute colored bounce. Propagated skylight remains the view-independent ambient authority for caves and overhangs.

### Physical atmosphere, clouds, and froxels

- Daylight comes from an Earth-like LUT atmosphere, not a stylized daytime gradient. The renderer uses a 256 by 64 transmittance LUT, a 32 by 32 multiple-scattering LUT, and a 192 by 108 sky-view LUT with Rayleigh, Mie, ozone, altitude response, physical solar angular radius, and weather aerosols. Directional attenuation comes from the later volumetric cloud composite instead of a camera-local coverage scalar, so clear gaps remain physically bright and cloudy pixels are not darkened twice. Stars and the phase-shaped moon remain night overlays.
- `CloudRenderer` owns deterministic 128-cubed Perlin-Worley base noise, 32-cubed erosion noise, and 2D curl noise. Weather blends stratus, cumulus, cumulonimbus, and cirrus profiles. Beer-Lambert extinction, dual-lobe phase scattering, ground and sky irradiance, short sun transmittance marches, erosion, and bounded multiple-scattering compensation define lighting.
- Cloud motion uses weather wind in blocks per second with independent layer response and wrapped double-precision offsets. High and Medium both render true quarter-resolution volumetric clouds. High uses 48 view steps and 6 light steps; Medium uses 24 and 3.
- Cloud color and hit depth use ping-pong temporal histories with invalid-sample rejection and neighborhood clamping, then upscale bilaterally against resolved depth. The snapped cloud-shadow map covers a 16,384-block footprint at 2,048 square on High and 1,024 square on Medium. Terrain, entities, volumetrics, the sun disc, and flare visibility consume the same transmittance authority.
- The air medium uses a 160 by 104 by 64 frustum-aligned froxel grid with logarithmic depth slices. It injects aerosols, humidity, precipitation fog, atmospheric extinction, active directional scattering, blended terrain shadows, and cloud shadows. Half-resolution scattering and transmittance are temporally reprojected before composition.
- Underwater absorption remains separate. The air medium is gated at the water surface and for a submerged camera so air fog cannot leak into water. Disabling volumetric lighting retains atmosphere-LUT aerial perspective but omits froxel shafts.

### Weather and storms

- One immutable weather snapshot feeds terrain, entities, foliage, particles, clouds, atmosphere, froxels, wetness, lightning, and audio for the frame. Wind is expressed in blocks per second everywhere. Rain or snow follows temperature rather than biome.
- Lightning IDs, positions, branches, and flashes derive from world seed, storm cell, and fixed time bucket. Lightning is depth-tested and cloud-aware, changes no blocks, and creates no fire. Thunder is procedural, bounded, de-duplicated, and delayed by strike distance at 343 blocks per second.
- `RYCRAFT_WEATHER` accepts stable `clear`, `overcast`, `rain`, `storm`, and `snow` presets. Forced time or weather invalidates every affected temporal history.
- Lava remains emissive in linear HDR and seeds derived block light into neighboring transparent cells.

The implementation is informed by [Practical Realtime Strategies for Accurate Indirect Occlusion](https://research.activision.com/publications/archives/atvi-tr-16-01practical-realtime-strategies-for-accurate-indirect-occlusion), [Screen-Space Diffuse Global Illumination](https://pure.mpg.de/pubman/item/item_1324270), [Stochastic Screen-Space Reflections](https://www.ea.com/frostbite/news/stochastic-screen-space-reflections), [Spatiotemporal Variance-Guided Filtering](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced-global), [Cascaded Shadow Maps](https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps), [Nubis volumetric cloudscapes](https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine), [Frostbite unified volumetrics](https://advances.realtimerendering.com/s2015/), and [production-ready atmosphere rendering](https://sebh.github.io/publications/egsr2020.pdf). Rycraft uses bounded Metal implementations of those principles, not verbatim copies of any production renderer.

The celestial response follows production patterns documented by the [Sildur's Vibrant Shaders changelog](https://sildurs-shaders.github.io/changelogs/) and visible in the current implementations of [Complementary Reimagined phase influence](https://github.com/ComplementaryDevelopment/ComplementaryReimagined/blob/08e1c2ada5eaf2fc36f08516c316b3d3c3677d8e/shaders/lib/colors/moonPhaseInfluence.glsl), [Complementary Reimagined light selection](https://github.com/ComplementaryDevelopment/ComplementaryReimagined/blob/08e1c2ada5eaf2fc36f08516c316b3d3c3677d8e/shaders/lib/colors/lightAndAmbientColors.glsl), and [Bliss direct-light selection](https://github.com/X0nk/Bliss-Shader/blob/81e403ed308141039a09d792a36f8eb328898a60/shaders/dimensions/composite.vsh). Those shaders gate the active source around the horizon, keep moonlight much dimmer than sunlight, apply lunar phase to lighting or reflections, and reuse the selected light direction across direct receivers. Rycraft centralizes direct-source decisions in one CPU state and keeps true solar state explicit for atmosphere and solar flare, so the two roles cannot silently diverge.

The orbital geometry follows [NASA's Moon phase explanation](https://science.nasa.gov/moon/moon-phases/) and uses the mean 29.53059-day synodic period published by the [U.S. Naval Observatory](https://aa.usno.navy.mil/faq/moon_phases), rounded to one deterministic world tick.

## 12. Verification is part of rendering work

Run every rendering change with:

```bash
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MTL_SHADER_VALIDATION=1
```

Capture settled frames through `RYCRAFT_CAPTURE` and inspect the PNG, not only the exit status. Cubic and far-world work requires captures that exercise:

- high and low exact cube Y;
- positive and negative X, Y, and Z seams;
- partial water above and below the surface;
- cold startup and a camera jump before all full-disk step-32 parents are resident, with no disconnected far island and with connected 16/8/4/2 refinement visible before full parent completion;
- per-column exact ownership acquisition at missing, capped, and empty-completed requirements, plus ownership retention during stale halo remeshing, including a far tile with both owned and never-ready columns;
- the 256-block drawable-coverage-frontier fade without a partially faded occluder, including a protected base-only tile that blocks farther islands;
- direct seam agreement between exact emitted surfaces and every filtered footprint;
- the visible distance taper across step-32, step-16, step-8, step-4, step-2, and exact voxel terrain, including step-2 fallback for every far-owned exploration-band fragment, step-8-or-finer fallback for every other far-owned exact-overlap fragment, a fully ready partial boundary tile, and flat and complex terrain at similar distances;
- a forest spanning the exact overlap and all far tiers, with exact step-2 anchors, strict coarser aggregate subsets, two-phase canopy exchange, and no duplicate impostors in exact-owned columns or across tile faces;
- a tier replacement while moving across both refine and coarsen hysteresis thresholds, with an atomic terrain swap beneath narrow fog and exactly one water owner;
- a horizon-facing view at radius 512;
- back-face culling from above, below, and inside overhangs;
- conservative occlusion with a ridge in front of a taller distant peak;
- mountains, cliffs, rivers, lakes, waterfalls, deltas, volcanoes, caves, aquifers, flora, land fauna, and underwater fish;
- stable lake, river, ocean, and delta shorelines without vertical water walls or unrelated levels joined by coarse geometry, plus the seed-42 canonical source-volume regression at X=-557, Z=379, the supported seed-42 lake lip at X=-8235, Z=2976, the incised river across X=-12288 at Z=2653 and Z=2654, the canyon at X=-23904, Z=0, and the separate lake-to-river outlet fall at X=-8256, Z=3072;
- the seed-764891 caldera at X=23029, Z=-111486 with a complete irregular dry rim, at least one block of freeboard, supported banks, and source water filling every voxel from the crater floor through its flat surface;
- a receiver-centered outlet fall whose exact cells and half-open owned five-quad far prism join the upper lip to lower standing water without a long slab, duplicate tile ownership, a vertical gap, or a raised receiver;
- active aboveground streaming with no full black loading panels and a lit generated terrain silhouette;
- the seed-42 exact-to-far frontier near X=69.7936, Y=85.7918, Z=-1472.94, confirming protected fine fallback plus lit, dark, or bedrock exact closure caps while parent residency timing records the synchronous canopy cost;
- underground travel across the hard-priority exploration band, including dark closed temporary openings, no skylight through missing vertical sections, and no block interaction through an unloaded cube;
- crooked forests, isolated logs, broad overhangs, cave entrances, sealed caves, and lava-lit interiors at multiple light angles, verifying that direct shadows do not suppress ambient, block, or emissive light;
- settled and moving views through every detailed cascade blend and into horizon coverage, with no ring, resolution jump, ownership double-cast, or stale projection;
- SSGI history acceptance and rejection during ordinary motion, disocclusion, teleport, resize, FOV change, quality change, forced time, and forced weather;
- `clear`, `overcast`, `rain`, `storm`, and `snow` at dawn, noon, dusk, and night, including rain-to-snow temperature selection, shared foliage and particle wind, wetness, cloud shadows, fog, and aerosol response;
- views below, inside, and above stratus, cumulus, cumulonimbus, and cirrus layers, including mountain intersections and motion long enough to expose noise tiling or temporal trails;
- lightning in front of and behind clouds, atmospheric flashes, repeatable event IDs, delayed thunder, and no world or fluid mutation;
- sun and moon transition captures through civil twilight, every representative lunar phase, water reflections after sunset, phase-scaled moon brightness, and no competing directional lights;
- physical atmosphere captures at dawn, noon, dusk, and night, with finite LUTs, stable horizon luminance, weather aerosol response, sun or moon occlusion, shafts, fog silhouettes, and no stylized daytime gradient authority;
- clear daytime sky and terrain together, with frozen time, weather preset, cloud quality, and the relevant F3 weather and atmosphere diagnostics recorded. Reject a dark or night-like sky over terrain receiving daylight illumination, or a daylight sky paired with stale nighttime direct state;
- distant textured slopes and alpha-cutout flora without shimmering, moire patterns, or disappearing coverage;
- shadows, indirect lighting, atmosphere, clouds, froxels, weather particles, lightning, water, bloom, flare, tonemapping, and UI in the integrated frame.

Zero Metal validation messages is required. Log errors, successful capture, plausible frame, culling counts, and frame time are separate checks. Run the final 60 FPS view-distance-512 measurement without validation and verify total unified memory remains at or below 64 GB.

## Review checklist

For a diff touching rendering, shaders, meshing, fluid geometry, culling, LOD, or GPU-shared state:

1. Is every GPU-shared structure defined once in `shader_types.hpp`, with size and offset assertions updated?
2. Do sample count, color format, depth format, blending, and cull mode match every pass using each pipeline?
3. Is each texture or buffer storage mode correct, and is size derived from the drawable, `sizeof`, or a bounded capacity?
4. Does every fullscreen sampler flip V where required?
5. Is dynamic per-frame data ring-buffered or otherwise safe from GPU overwrite, and are arena frees deferred until frame completion?
6. Are matrices and depth conventions unchanged or covered by focused tests?
7. Are exact mesh positions local in X, Y, and Z, with a full three-dimensional `ChunkOrigin`?
8. Are far mesh positions tile-local, with the 64-bit CPU tile origin restored once per draw?
9. Are exact cube AABBs, frustum tests, candidate distance, mesh keys, and water ordering three-dimensional?
10. Does the snapshot contain every 18 by 18 by 18 block, fluid, packed voxel-light, and diagonal sample the exact mesher reads, plus separate 18 by 18 generated-surface and sky-cutoff authority?
11. Do solid, transparent, packed lighting, changed-face reconciliation, edit invalidation, missing-neighbor caps, and water tests cover negative and positive X, Y, and Z faces, including cave mouths, isolated logs, broad roofs, sealed caves, overhangs, added and removed roofs, the real world-ceiling cutoff, and an incomplete sky path?
12. Do loaded diagonal halo samples and conservative missing-halo fallbacks produce the same water corner on either side of a cube face?
13. Are exact sub-block values binary-exact through the 0 through 16 local range?
14. Does partial water honor levels, sources, stable top-only geometry, full-volume implicit generated sources, explicit falling sides, corners, flow bits, physics height, and water-to-water culling?
15. Does the 16-byte vertex layout remain unchanged with fluid direction in bits 24 through 26, falling water in bit 27, far-canopy marking in bit 28, far-skirt marking in bit 29, water exterior-sky authority in bit 30, and bit 31 reserved?
16. Do exact and far opaque faces use outward counterclockwise winding and back-face culling, while cross and flat flora emit both windings?
17. Does every tile intersecting the visible disk request a step-32 parent before refinement, with a broad nearest-first base lane, a four-worker parent reservation, four urgent connected refinement slots spanning 16/8/4/2 tiers before full parent completion, and resident active parents pinned under pressure?
18. Do separate parent and drawable frontiers suppress farther resident islands, does the drawable frontier treat protected base-only tiles as missing, does it fade the preceding 256 blocks, and are partially faded patches excluded from occluders?
19. Does revision-aware readiness derive a 16 by 16 per-column ownership bit only from currently published requirements and exact meshes, count empty completed meshes as ready, protect every far-owned exact-overlap fragment from step-32 display including fully ready partial boundary tiles, bind eight neighboring masks for crossing geometry, and exclude partially masked patches from occluders? Separately, does every missing exact halo emit its explicit lit, dark, or bedrock closure cap and invalidate that mesh when the real halo arrives?
20. Do filtered footprint samples preserve hydrology and feature ownership, use conservative parent minima, carry one weighted material palette, and avoid aligned material caches?
21. Do refinements use 256-block alignment, distance-and-complexity selection, asymmetric hysteresis, step-2 protection for every far-owned exploration-band fragment, step-8 protection for every other far-owned exact-overlap fragment including fully ready partial boundary tiles, protected-job grace and transition-cap bypass, a narrow terrain-only swap pulse for ordinary replacements, monotonic two-phase canopy exchange, single-owner water, greedy merging, deterministic borders, and LOD skirts only on resident finer-to-coarser edges with displayed-neighbor and paired ownership checks?
22. Do depth-biased opaque parents remain available behind exact terrain only outside protected exact-loading tiles, is synchronous canopy cost measured separately from terrain and water parent work, and do step 2 plus steps 4 through 32 retain their accepted anchors, deterministic aggregate subsets, species-shaped silhouettes, authoritative root-water placement, displayed-voxel grounding, and half-open ownership?
23. Are frustum and 256-bin horizon culling conservative, front to back, and incapable of hiding a taller visible feature?
24. Does documentation accurately avoid claiming HZB, literal geometry clipmaps, indirect command buffers, or GPU-driven submission, while crediting the screen-space indirect min-depth pyramid to ray traversal only?
25. Are far shorelines contour-clipped and body-aware, with incompatible water authorities refined rather than joined, and do their source-water tops match the exact full-block plane without unsupported vertical walls?
26. Does the block-texture array contain every 16-to-1 mip, preserve alpha-tested coverage, and use nearest magnification with trilinear 8x-anisotropic minification?
27. Is translucent and post-resolve geometry ordered correctly across indirect light, clouds, lightning, water, froxels, weather particles, post effects, and UI?
28. Do exact and far residency plus all post targets remain within the 64 GB unified-memory ceiling, and do the new persistent High-tier allocations remain within 768 MiB?
29. Do missing boundaries use bounded exterior connectivity to keep sky-connected lateral caps lit, keep enclosed lateral and vertical openings dark and closed, honor complete edited roof cutoffs, seed skylight only from proven full-height paths, and prevent raycasts or edits from crossing unloaded cubes?
30. Was the game run with Metal validation, were the required exact, far, surface, cave-GI, cascade-motion, weather, cloud-layer, atmosphere, lightning, celestial-transition, and underwater captures inspected, and was the final view-distance-512 performance run recorded separately?
31. Does canonical ocean, river, lake, delta, waterfall, and bank authority produce supported full-depth implicit source water, preserve distinct lake levels behind a supported competitive watershed except at owned connectors, and keep each outlet fall separate from the receiving level?
32. Do the five shadow records use view depth, exact Medium and High splits, 12.5 percent blend bands, per-cascade bias and filter scale, valid-coverage fallback, and refresh intervals of one, one, two, four, and eight frames?
33. Do direct visibility, propagated skylight, baked accessibility, block light, emissive radiance, GTAO, and SSGI remain independent, with Off still applying ambient and with all temporal reset reasons covered?
34. Do atmosphere LUT dimensions, optical coefficients, cloud-noise dimensions, step counts, physical wind units, cloud histories, shadow footprint, froxel dimensions, logarithmic slices, and underwater gating match the documented contracts?
35. Does one immutable weather snapshot feed the frame, do stable presets cover every weather kind, and are lightning and thunder deterministic, bounded, delayed, non-destructive, and free of backlog replay?
36. Is directional radiance exclusive across twilight, with below-horizon sun suppression, moon suppression through civil twilight, deterministic synodic phases, phase-scaled lunar light and shadows, and matching water glint, clouds, froxels, atmosphere, and flare?
37. Does the physical sky use the same celestial time, true solar elevation, weather snapshot, and exposure path as the scene, so clear daylight cannot show a dark or night-like sky over a daylight-lit ground?
