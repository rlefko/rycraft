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
| 30 through 31 | Reserved |

Renderer metadata must stay in the assigned high bits unless the vertex descriptor, shaders, tests, arena budgets, and every producer change together. Bit 28 identifies visual-only far canopy geometry so the fragment shader can suppress it in exact-owned columns. Bit 29 identifies far boundary skirts so displayed-neighbor state and paired emitting-and-receiving column ownership can suppress unnecessary walls.

## 2. Pipelines match their passes

- A pipeline's `rasterSampleCount` equals the attachment sample count for every pass that uses it.
- A pipeline's depth format equals the bound depth texture format. Scene depth is `Depth32Float`.
- Fullscreen sampling passes flip V because Metal texture coordinates and NDC Y run in opposite directions.
- Anything encoded in the 4x MSAA scene pass declares sample count 4. Post-resolve water and final composites declare sample count 1.
- Color attachment formats match the linear HDR or display stage in which the pipeline runs.

## 3. Resolution, storage, and GPU lifetime

- Scene targets use drawable pixel dimensions, not Cocoa point dimensions. Recheck drawable size every frame.
- The scene target is linear `RGBA16Float`. Its 4x MSAA color and depth textures are memoryless and resolve into single-sample textures.
- CPU-written buffers use `StorageModeShared` on Apple Silicon. Never call `contents` on a private buffer.
- Per-frame constants use three frame-ring slots behind the three-frame semaphore.
- Exact and far mega-buffer ranges enter deferred-free queues. A range cannot return to an allocator until the GPU completes the frame that last referenced it.
- Exact mesh residency cannot exceed 16,384. The far CPU cache cannot exceed 9,280 entries or 3 GiB. The far GPU arena grows lazily in paired 256 MiB vertex and 128 MiB index slabs, up to 2 GiB of vertices and 1 GiB of indices.
- All renderer targets, arenas, caches, world state, transient work, and Metal allocations together must remain below the 64 GB unified-memory acceptance ceiling.
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

The game meshes exact terrain from `MeshSnapshot`, not directly from a mutable cube. A snapshot contains the 16-cube-edge interior plus a one-block halo, for 18 by 18 by 18 block, fluid, and block-light data. Skylight is represented separately by an 18 by 18 array of per-column sky cutoffs derived from the exact density surface and loaded opaque features or edits. A valid loaded cutoff may raise or lower the generated surface. The incomplete-path sentinel is distinct from every valid cutoff, including the world-ceiling cutoff.

- Loaded cubes in the surrounding 3 by 3 by 3 neighborhood supply real face, edge, and corner samples. An unavailable in-range halo follows the immutable generated surface cutoff. Cells above the silhouette remain air and cells below it remain opaque. When a lateral cap candidate exists, a bounded six-connected flood marks transparent cells reachable from sky-exposed cells within the padded snapshot. A complete loaded cutoff honors added roofs and removed surfaces. An incomplete vertical path uses generated authority for this provisional classification while remaining fully occluded for ordinary skylight. Sky-connected lateral openings emit normally lit provisional faces using one representative arriving-surface material per missing face. Enclosed lateral openings emit unlit stone, and missing vertical openings emit bedrock. These faces are temporary boundaries, not generated world content.
- Solid faces, transparent faces, skylight, block light, flora ownership, ambient occlusion, fluid corners, and explicit falling sides read real data whenever the corresponding neighbor is loaded. Loading or unloading any halo cube dirties all affected neighboring meshes so the temporary boundary disappears or returns coherently.
- Edits dirty the owning mesh and every face, edge, or corner neighbor mesh whose one-block halo intersects the changed block.
- Water corner smoothing at an X/Z edge reads the diagonal cube when it is loaded. A missing diagonal is the same closed opaque boundary on both participating meshes.
- A generated surface cutoff grants skylight only when every vertical section from the meshed cube through that cutoff is loaded. An incomplete vertical path receives a fully occluded cutoff, so an unloaded section above an underground view cannot admit sunlight.
- At the vertical world floor, missing halo cells are bedrock. Above the ceiling they are air.
- Test positive and negative X, Y, and Z separately. A horizontal-only seam test cannot catch a wrong cube Y origin or top and bottom halo defect.

Exact opaque cube faces use greedy merging. The merge key includes block and texture identity, face direction, skylight, block light, emissive state, sway state, and the four 2-bit ambient-occlusion corners. A quad splits along its brighter diagonal when corners are not planar.

## 6. Block shapes, winding, and back-face culling

`BlockDefinition::renderShape` is exhaustive:

- `CUBE` participates in greedy opaque meshing.
- `CROSS` emits two diagonal, inset flora planes.
- `FLAT` emits a horizontal plane, currently used by lily pads.
- `LIQUID` enters the water section.
- `NONE` emits no geometry.

Main-pass exact cube faces and far terrain use counterclockwise outward winding with back-face culling. Cross and flat flora emit both windings so they remain visible from either side. Water uses cull-none because surfaces are visible from below. Shadow casters also use cull-none because visible-face-only greedy meshes are not closed solids.

Alpha-cutout leaves, glass, flora, and their shadow casters use texture alpha discard. Scene and shadow vertex stages call the same shared wind-sway helper so animated geometry and shadows cannot separate.

## 7. Far-terrain LOD and visibility

The far renderer selects every immutable 256 by 256-block tile intersecting the visible radius-512 disk, including tiles wholly inside the nominal exact radius. Every coordinate requests a step-32 parent before optional refinement. Missing parents enter a broad nearest-first job and upload lane. Each connected nearby coordinate then gives its distance-selected step-16, step-8, step-4, or step-2 target one bounded lane before the complete parent disk is resident. While parents remain queued, eight-worker dispatch reserves four slots for coverage and admits at most four urgent connected refinements. Already running parent work is not preempted. A resident parent remains pinned while its coordinate is active. Cache pressure evicts the farthest refinement first.

Parent residency and drawable coverage use separate connected frontiers. The parent frontier tracks the nearest missing step-32 dependency. The drawable frontier also treats a protected exact-loading tile with only step 32 ready as missing. Tiles and fragments at or beyond the drawable frontier are suppressed, and the preceding 256 blocks fade into frame fog. A partially faded patch cannot become an occluder. Every far-owned fragment in the exact overlap remains protected, including a fully ready boundary tile whose exact requirements cover only part of the tile. The drawable frontier advances only after the camera exploration band has step 2 and every other protected overlap tile has step 8 or finer. This prevents out-of-order completion from exposing a disconnected distant island through a protected base-only tile.

Exact ownership comes from `ExactSurfaceCoverageSnapshot`, not the configured radius alone. One 256-bit mask describes the 16 by 16 exact chunk columns overlapping each far tile. A bit is set only when all requirements currently published for that column are owned by exact meshes; empty completed meshes count as ready, while missing and unresolved requirements keep the column far-owned. A previously published exact mesh can retain ownership while an ordinary replacement is pending. Each far draw binds the center mask and all eight neighboring tile masks so a canopy crown or waterfall crossing a tile face queries its destination column. Exact opaque terrain draws first, and overlapping far terrain uses a small positive depth bias so resident exact surfaces win. Far ownership does not make step 32 displayable anywhere inside the exact overlap, even in a fully ready partial boundary tile. A partially masked patch cannot contribute to the terrain-horizon occluder. The nearest-gap handoff distance remains a conservative parent-selection fallback and diagnostic, not a radial fragment-ownership boundary.

Tile construction requests one-, two-, four-, eight-, sixteen-, or thirty-two-block `SurfaceFootprint` values through one `FarSurfaceSample` contract. The contract carries filtered terrain and water, conservative footprint bounds, and a compact material palette. Hydrology topology, water-body identity, water elevation, plate ownership, and feature anchors do not vary by footprint. Step-32 coverage geometry uses conservative minima so a parent cannot protrude through resident exact terrain. Material resolves once per active LOD cell, and greedy meshing joins only equal resolved materials. There is no aligned 32- or 64-block material cache.

Tile construction emits terrain tops, body-aware contour-clipped standing-water tops, explicit outlet-fall prisms, boundary skirts, and visual-only canopy impostors. Far water carries stable body identity and water kind. A coarse cell observing incompatible standing-water authorities refines from canonical samples rather than joining their levels. Bit 29 marks every boundary-skirt vertex. Per-draw metadata enables an edge only when its displayed neighbor is resident at a coarser step, and the fragment shader samples exact ownership on both sides of that far-LOD join. Far-LOD skirts are separate from exact missing-halo closure. Exact lateral openings already emit either a lit planned surface continuation or a dark inward cap, and exact vertical openings emit bedrock caps until the halo arrives. Partially wet cells use clipped shoreline triangles rather than rectangular sheets. The far mesh represents the visible top of a generated standing-water volume whose exact column is source-filled from its solid support through the surface, and it places that top on the exact full-block source plane.

A lake outlet fall is not part of the receiving body's standing-water mesh. Its immutable sample carries top and bottom surfaces, width, normalized flow, and one receiver anchor. The half-open tile containing that anchor owns one complete narrow prism centered on the receiver, even if its footprint crosses a tile face. The prism has four vertical sides and one top, extends into the lower body's top source voxel so it overlaps the visible water plane, reaches the upper lip, and does not raise or cover the receiving surface. Neighbor tiles emit no duplicate prism.

Step-2 far canopies reuse accepted exact tree anchors, species, and dimensions without constructing ColumnPlans. Steps 4 through 32 use globally anchored 64-block aggregate forest cells with six fixed candidates and block-8 habitat and ground authority. One stable rank makes those aggregate tiers strict subsets. The collector's block-resolution habitat and root-water decision remains authoritative at step 32, so water elsewhere in the coarse 32 by 32 cell cannot suppress an accepted canopy. The trunk grounds on the displayed voxel. Half-open cells and tiles retain unique ownership. Bit 28 marks every impostor vertex for the shared vertex contract and diagnostics. The per-column exact mask clips each canopy fragment only when its destination exact column is owned. The 3 by 3 neighboring mask set covers crowns crossing tile faces. During a far topology transition, target canopies appear monotonically during the first half before source canopies retire monotonically during the second half. This overlap-safe exchange also handles the intentionally unrelated step-2 and aggregate anchor sets. Terrain, water, and canopy geometry currently share one cold-build and residency payload. Measured cold canopy work ranges from 250 to 1,165 milliseconds and can delay publication of an otherwise ready terrain and water parent. Staged canopy attachment remains a follow-up.

Far tiles do not carry caves, structures, per-block flora, entities, collision, edits, runtime fluid state, save ownership, or exact biome transition detail. Canopy impostors are immutable visual summaries only.

Exact opaque terrain draws first and shares the HDR scene and depth attachments with far opaque terrain. Outside protected exact-loading tiles, biased parent tops remain behind exact depth and can fill cold-residency gaps after those parents are resident. Every far-owned overlap fragment is protected. It requires step 2 in the camera exploration band or step 8 or finer elsewhere in the exact overlap, including within a fully ready partial boundary tile. Water samples resolved depth without binding a depth attachment, so water and canopies use the same current per-column masks as opaque far terrain. Far water is appended to the same back-to-front water list as exact water. An ordinary far LOD replacement submits only the source water until completion, guaranteeing one refractive owner for every tile.

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

The implementation is an adaptive tiled LOD inspired by [Geometry Clipmaps](https://hhoppe.com/geomclipmap.pdf) and [CDLOD](https://doi.org/10.1080/2151237X.2009.10129287). It is not a literal geometry clipmap. The angular test is informed by conservative front-to-back occlusion work such as [Hierarchical Z-Buffer Visibility](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/greene93_hierarchicalz.pdf), but it is not HZB and owns no depth pyramid.

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

- The fragment shader samples a blit copy of resolved opaque color for refraction and samples resolved depth for manual occlusion and camera-relative reconstruction.
- No depth attachment is bound while the shader samples `_depthResolve`.
- The shader owns its composite pixel, so the surface pipeline has no blend state.
- Exact and far water draws sort back to front by full three-dimensional distance.
- Screen-space reflection, depth absorption, procedural caustics, Fresnel sky reflection, sun sparkle, and shared fog run in this pass when enabled.
- The underwater overlay, god rays, and caustics render last when the camera is submerged.
- Anything that belongs behind water renders in the scene pass. Anything intentionally above water renders afterward.
- **The wave field is one table.** `WATER_WAVES` in `shader_types.hpp` drives both the vertex displacement (`waterWaveHeight`) and the fragment normal (`waterSurfaceNormal`), with the phase advected by the packed flow direction. Why: the two lived as separate formulas once, and editing the sea in one silently desynced the shading from the geometry.
- **The SSR march starts at an IGN-jittered stride** (a deliberately narrow range) and, when the camera is submerged, attenuates the reflected hit per channel by `WATER_SIGMA_A` over the reflected path. On a thick-occluder reject the ray keeps marching instead of falling back. Why: a coherent stride turned the coarse march into stair bands across every reflection, wide jitter traded the bands for salt-and-pepper speckle with no temporal history to average it, bailing out on rejects flipped neighboring pixels between hit and sky along every silhouette, and unabsorbed hits mirrored crisp daylight colors from deep underwater.
- **From below, the surface is physical.** Water-to-air Fresnel with total internal reflection past the critical angle (eased near it so per-quad wave normals do not flip whole cells into hard panels); SSR mirrors the underwater scene with the deep tint as fallback; foam, refraction distortion, and the floor-caustic add are above-water-only. Why: each of those painted above-water effects onto the from-below view, including white waterline streaks and mis-oriented caustic bands.
- **The Snell window transmits without absorption.** From below, the distance behind the surface is air (sky or shore), and the eye-to-surface water segment already belongs to the underwater overlay. Why: absorbing that air distance as if it were water saturated the whole window into opaque flat blue instead of a view of the world above.
- **The interface side is a per-fragment geometric fact.** The water pass classifies each fragment by the sign of dot(V, N) (an elevated lake's underside seen from dry land is still the water-to-air interface; the camera flag only decides which medium rays travel through) and evaluates Schlick Fresnel against the transmitted angle on the water side, where cosT reaching zero at the critical angle rises continuously into total internal reflection with no hand-tuned ease. Why: branching on the camera flag saturated dot(V, N) to zero for every submerged pixel, which read as past-critical total internal reflection everywhere and turned the whole surface into a permanent mirror that never transmitted.
- **The internal mirror reflects luminous water.** Where the underside SSR misses, the fallback is the shared water volume scatter terms (`WATER_SCATTER`, `WATER_AMBIENT`, the same constants the overlay inscatters with). Why: a near-black fallback read as flat dark panels, where a real internal mirror reflects the sunlit volume.
- **The sun glint obeys Fresnel.** The sparkle term multiplies by the same Fresnel factor as the sky reflection. Why: about two percent reflects at normal incidence, so an unscaled glint under a zenith sun mirrored in every up-facing wave and bloomed into one giant white blob on the surface.
- **Caustics track the waves that focus them.** The web's cell scale sits at the ripple wavelength, its arms are warped by the shared wave normal, a slow rotated modulator octave breaks the wrapped tile's exact periodicity, and the web defocuses into broad swell-scale patches with floor depth. Why: one wrapped octave repeated identical ~2-block cells across every floor, and a crisp fixed-depth web read as painted on rather than focused by the surface.
- **Displaced crests stay below block tops.** The swell midline is biased down and scaled by each cell's fluid level. Why: the rest plane sits at 0.875, so unbiased crests washed over adjacent shoreline blocks, and a thin flowing sheet would otherwise displace below its own floor.

### The underwater overlay is physically based

When the camera is submerged, the fullscreen overlay owns the entire water tint (the scene passes apply no fog below the surface, and the rain sheen turns off):

- **Per-channel Beer-Lambert absorption through dual-source blending.** The fragment outputs inscatter at `color(0) index(0)` and per-channel transmittance at `index(1)`; the pipeline blends `result = inscatter + scene * transmit`. Why: a single alpha cannot express spectral absorption, and red must die faster than blue for distance to read as water rather than flat fog.
- **Absorption counts only the in-water path.** Upward rays stop accumulating at the water surface (`waterSurfaceY`, scanned up from the camera cell on the CPU), and the shaded point's own depth below the surface attenuates the light that reached it. Why: fogging by the opaque distance behind the surface (the sky is far) drowned every upward view in murk, and depth-independent lighting made deep floors look daylit.
- **Sunlight is gated by sky exposure.** Covered water (sealed aquifers, roofed lakes, checked against the surface-height map like rain spawning) zeroes the caustics, the sun-driven inscatter, and the submerged volumetric shafts. Why: the shadow cascades cannot occlude terrain hundreds of blocks up, so sealed pockets grew impossible sun caustics and shafts.
- **Caustics modulate, never add.** The web multiplies the transmittance, so it rides each floor's own shading; the pattern is the iterative wave-warped web (`causticPattern`, warped by the shared wave normal so light moves with the waves) and is clamped, because an unclamped HDR caustic crossed the bloom threshold across whole floors and whited them out.
- **Screen-space normals for the caustic gate use best-of-both-sides depth taps** with a silhouette feather, the same reconstruction rationale as `ssao.metal`. Why: one-sided derivatives straddle block silhouettes and lit dashed lines along every oblique edge.
- **Inscatter is anisotropic.** A capped Henyey-Greenstein lobe brightens the view toward the sun. Why: isotropic murk lost the underwater silver lining that makes the volume read as sunlit water.

## 10. HDR frame graph and the one tonemap

The scene renders in linear HDR and is graded exactly once.

1. Texel-snapped shadow cascades render from the active sun or moon.
2. Sky, exact terrain, far terrain, entities, highlight, particles, and flat clouds render into the 4x MSAA HDR scene.
3. SSAO and volumetric clouds reconstruct from resolved depth at reduced resolution and use spatial dithering plus depth-aware bilateral reconstruction.
4. Scene application combines opaque HDR, AO, and clouds.
5. Water composites against copied opaque color and resolved depth.
6. Volumetric light marches the cascades and composites into HDR.
7. GPU compute updates persistent exposure and flare state without CPU readback. Exposure meters a highlight-weighted mean of log luminance with asymmetric adaptation (fast down, slow up). Why: a flat mean barely moves when the small bright sun enters the frame, so facing the sun never stopped the scene down; with no highlights the weighted mean equals the plain mean, so caves and night keep their lift.
8. Bloom builds its HDR pyramid when enabled.
9. One always-on final composite applies exposure, the Hable filmic tonemap (its shoulder keeps compressing across many stops, so the HDR sun lands on a gradient where an earlier curve plateaued to flat white), vibrance, contrast, lens flare, optional CAS sharpening, and dithering.
10. UI draws at display resolution.

Toggled-off effects skip work or bind static fallback textures. They do not fork the scene into an untonemapped path. Half- and quarter-resolution effects dither in space because this renderer uses memoryless 4x MSAA rather than a temporal history.

## 11. Shadows, ambient occlusion, block light, and weather response

- Shadow cascade projections are texel-snapped to prevent crawling.
- Depth bias is slope-scaled and tuned per cascade. Near flora needs a small clamp, while far terrain still needs enough range-scaled bias to prevent acne.
- Shadow casters render cull-none with alpha-cutout discard.
- Baked corner ambient occlusion, derived block light, emissive state, and skylight ride `faceAttr`.
- SSAO rejects samples behind the receiver tangent plane rather than using raw depth difference alone.
- SSAO is bilateral-blurred at half resolution before it multiplies the scene, weighted by view-space depth so it never bleeds across block silhouettes, and its strength fades at grazing view angles. Why: the generate pass rotates its kernel with per-pixel IGN and MSAA keeps no temporal history to hide that noise under, so unblurred AO printed diagonal scan lines on ceilings and grazing ground.
- Lava remains emissive in linear HDR and seeds derived block light into neighboring transparent cells.
- Rain wetness, foliage subsurface response, wind sway, cloud shadowing, fog, and post effects consume the same world and sky uniforms in exact and far geometry where applicable.

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
- distant textured slopes and alpha-cutout flora without shimmering, moire patterns, or disappearing coverage;
- shadows, SSAO, volumetrics, weather, water, bloom, flare, tonemapping, and UI in the integrated frame.

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
10. Does the snapshot contain every 18 by 18 by 18 block, fluid, block-light, and diagonal sample the exact mesher reads, plus the separate 18 by 18 sky cutoffs?
11. Do solid, transparent, lighting, edit invalidation, missing-neighbor caps, and water tests cover negative and positive X, Y, and Z faces, including overhangs, added and removed roofs, the real world-ceiling cutoff, and an incomplete sky path?
12. Do loaded diagonal halo samples and conservative missing-halo fallbacks produce the same water corner on either side of a cube face?
13. Are exact sub-block values binary-exact through the 0 through 16 local range?
14. Does partial water honor levels, sources, stable top-only geometry, full-volume implicit generated sources, explicit falling sides, corners, flow bits, physics height, and water-to-water culling?
15. Does the 16-byte vertex layout remain unchanged with fluid direction in bits 24 through 26, falling water in bit 27, far-canopy marking in bit 28, far-skirt marking in bit 29, and bits 30 through 31 reserved?
16. Do exact and far opaque faces use outward counterclockwise winding and back-face culling, while cross and flat flora emit both windings?
17. Does every tile intersecting the visible disk request a step-32 parent before refinement, with a broad nearest-first base lane, a four-worker parent reservation, four urgent connected refinement slots spanning 16/8/4/2 tiers before full parent completion, and resident active parents pinned under pressure?
18. Do separate parent and drawable frontiers suppress farther resident islands, does the drawable frontier treat protected base-only tiles as missing, does it fade the preceding 256 blocks, and are partially faded patches excluded from occluders?
19. Does revision-aware readiness derive a 16 by 16 per-column ownership bit only from currently published requirements and exact meshes, count empty completed meshes as ready, protect every far-owned exact-overlap fragment from step-32 display including fully ready partial boundary tiles, bind eight neighboring masks for crossing geometry, and exclude partially masked patches from occluders? Separately, does every missing exact halo emit its explicit lit, dark, or bedrock closure cap and invalidate that mesh when the real halo arrives?
20. Do filtered footprint samples preserve hydrology and feature ownership, use conservative parent minima, carry one weighted material palette, and avoid aligned material caches?
21. Do refinements use 256-block alignment, distance-and-complexity selection, asymmetric hysteresis, step-2 protection for every far-owned exploration-band fragment, step-8 protection for every other far-owned exact-overlap fragment including fully ready partial boundary tiles, protected-job grace and transition-cap bypass, a narrow terrain-only swap pulse for ordinary replacements, monotonic two-phase canopy exchange, single-owner water, greedy merging, deterministic borders, and LOD skirts only on resident finer-to-coarser edges with displayed-neighbor and paired ownership checks?
22. Do depth-biased opaque parents remain available behind exact terrain only outside protected exact-loading tiles, is synchronous canopy cost measured separately from terrain and water parent work, and do step 2 plus steps 4 through 32 retain their accepted anchors, deterministic aggregate subsets, species-shaped silhouettes, authoritative root-water placement, displayed-voxel grounding, and half-open ownership?
23. Are frustum and 256-bin horizon culling conservative, front to back, and incapable of hiding a taller visible feature?
24. Does documentation accurately avoid claiming HZB, literal geometry clipmaps, indirect command buffers, or GPU-driven submission?
25. Are far shorelines contour-clipped and body-aware, with incompatible water authorities refined rather than joined, and do their source-water tops match the exact full-block plane without unsupported vertical walls?
26. Does the block-texture array contain every 16-to-1 mip, preserve alpha-tested coverage, and use nearest magnification with trilinear 8x-anisotropic minification?
27. Is translucent and post-resolve geometry ordered correctly relative to water?
28. Do exact and far residency plus all post targets remain within the 64 GB unified-memory ceiling?
29. Do missing boundaries use bounded exterior connectivity to keep sky-connected lateral caps lit, keep enclosed lateral and vertical openings dark and closed, honor complete edited roof cutoffs, and prevent raycasts or edits from crossing unloaded cubes?
30. Was the game run with Metal validation, were the required exact, far, surface, cave, weather, and underwater captures inspected, and was the final view-distance-512 performance run recorded separately?
31. Does canonical ocean, river, lake, delta, waterfall, and bank authority produce supported full-depth implicit source water, preserve distinct lake levels behind a supported competitive watershed except at owned connectors, and keep each outlet fall separate from the receiving level?
