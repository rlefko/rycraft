# World Generation and Persistence

This document is the source of truth for cubic coordinates, procedural generation, runtime water, and world persistence. The implementation synthesizes a static world from bounded coordinate queries. It does not simulate a finite planet through geological time.

## World coordinates and cubic chunks

The horizontal world is procedurally unbounded. Renderer and entity positions still use floating-point values, so very long travel eventually reaches practical precision limits. The supported vertical range is finite.

| Constant | Value |
|---|---:|
| Cube edge | 16 blocks |
| Cube volume | 4,096 blocks |
| Minimum world Y | -128 |
| Maximum world Y | 511 |
| Vertical cube range | -8 through 31 |
| Sea level | 64 |

Coordinates use three canonical types from `include/world/chunk_pos.hpp`:

```cpp
struct ColumnPos { int64_t x; int64_t z; };
struct ChunkPos  { int64_t x; int32_t y; int64_t z; };
struct BlockPos  { int64_t x; int32_t y; int64_t z; };
```

Each type has explicit equality and hashing. X and Z remain 64-bit throughout generation and persistence. `world_coord::floorDiv` and `floorMod` define all chunk and local-coordinate conversion, including negative boundaries. A block below Y=-128 reads as bedrock, a block above Y=511 reads as air, and edits outside the range are ignored.

`Chunk` stores one 16 by 16 by 16 cube. A new cube is a uniform value without a 4,096-element block allocation. The first nonuniform edit materializes dense storage, and `compactStorage` returns a dense cube to uniform storage when possible. Fluid states are another optional 4,096-byte array. A generated source-water cube needs no fluid array unless a cell has an explicit runtime level or falling state.

All 58 `BlockType` values are append-only because saves persist raw bytes. A single compile-time `BlockDefinition` table exhaustively defines every block's render shape, collision solidity, opacity, targeting, liquid behavior, leaf behavior, sound, material, light emission, emissive state, and wind-sway class. `BlockType::COUNT` and compile-time assertions prevent a new block from omitting its traits.

| Trait | Meaning |
|---|---|
| `renderShape` | Cube, cross, flat, liquid, or none |
| `solid` | Participates in collision |
| `opaque` | Hides a neighboring solid face and blocks light |
| `targetable` | Stops the block-edit raycast |
| `liquid` | Participates in water or static-lava physics |
| `sound` and `material` | Shared audio and material classification |
| `lightEmission` and `emissive` | Derived block-light source and self-lit HDR surface |
| `sway` | Static, root-bending flora, or whole-block leaves and reeds |

Cross flora is walk-through and targetable. Lily pads use flat geometry. Leaves and glass remain solid but nonopaque. Lava is non-solid, opaque, static, emissive, and the current level-15 block-light source.

Biome and surface metadata do not live in every vertical cube. An immutable `ColumnPlan` retains nine full `SurfaceSample` values on a world-aligned 3 by 3 lattice at eight-block spacing for one horizontal chunk column. It also stores the exact density surface Y for all 256 local X/Z columns and a compact canonical lake authority on the complete 17 by 17 world-coordinate grid. Each lake entry records known membership, flat water level, positive depth, and endorheic state. Ambiguous lattice cells query exact hydrology, while uniform cells interpolate one lake signature. Shared positive faces therefore receive the same lake classification as the adjacent column. The plan records the vertical sections that may expose terrain, water, cliffs, vegetation, or waterfalls. Construction makes 16 transient height-only perimeter queries to cover neighboring feature anchors, but those samples are not retained. `ColumnPlanCache` is a bounded 8,112-entry single-flight cache under a compile-time 64 MiB payload budget, so concurrent requests for one column share its construction.

## Public generation contracts

`ChunkGenerator` exposes three forms of procedural access:

- `getColumnPlan(ColumnPos)` returns the immutable plan for a horizontal column.
- `generateCube(Chunk&)` emits one cube and writes nowhere else.
- `sampleSurface(x, z)`, `baseHeightAt(x, z)`, `surfaceYAt(x, z)`, and `biomeAt(x, z)` provide coordinate-pure queries.

The macro sampler returns the data shared by terrain and ecology:

- `GeologySample`: plate, crust, rock, plate velocity, boundary classification, uplift, rift, fault, hotspot, and volcanic values.
- `HydrologySample`: flow direction, terrain and water elevations, discharge, sediment, channel geometry, erosion, lake depth, stream order, and water-body flags.
- `ClimateFields`: wind, temperature, annual precipitation, potential evapotranspiration, aridity, and relative humidity.
- `SoilSample`: moisture, fertility, drainage, and water table.
- `BiomeBlend`: the two strongest biome suitability results and the secondary transition weight.
- `Ecotope`: riverbank, floodplain, delta, lakeshore, coast, cliff, scree, canyon, geothermal, cave, and aquifer overlays.

These values are deterministic descriptors for world synthesis. Units such as annual precipitation and climate temperature are calibrated procedural values, not forecasts or a global climate simulation.

## Generation pipeline

The pipeline is acyclic:

1. Geology and provisional relief
2. Provisional precipitation for drainage discharge
3. Bounded basin routing, erosion, channel incision, and water bodies
4. Final climate over the incised elevation
5. Soil and continuous biome suitability
6. Cubic density, materials, water, ores, structures, and vegetation

This order prevents a biome ID from feeding back into terrain shape. Biomes choose surface materials and ecology after height, water, and climate are established.

### Geology and provisional relief

`MacroGenerationSampler` places one jittered plate seed in every 8,192-block lattice cell after low-frequency domain warping. A bounded 3 by 3 search finds the two nearest sites. Each site has coordinate-counter-derived crust type, age, thickness, density, rock, and velocity.

Relative velocity is decomposed into normal and tangential motion near the nearest-site boundary:

- Closing normal motion produces convergent uplift.
- Opening normal motion produces divergent rift relief.
- Dominant tangential motion produces transform fault strength.
- Boundaries with little normal motion also receive a weaker transform classification.

The result controls broad relief rather than replaying plate evolution. Pairwise candidate relationships are evaluated across the bounded 3 by 3 neighborhood, so convergent, divergent, and transform signals remain continuous through plate edges and triple points even though the persisted plate identifier changes discretely. Continental fraction blends oceanic and continental base elevations. Uplift, rifts, faults, ridged noise, and smaller relief noise form the provisional height.

Relief is intentionally amplified for a fantastical game scale. Continental convergence can build broad folded massifs above Y=300, while oceanic convergence favors trenches and lower island arcs. Hotspots, faults, and rifts have distinct bounded relief responses. Smooth noise envelopes and unit-slope hyperbolic compression keep this exaggeration continuous instead of creating a height step, solitary wall, or flat clipped summit.

Hotspot candidates occupy a 16,384-block lattice and have a 14 percent acceptance chance. An accepted hotspot emits four through seven volcanoes along the direction opposite the nearest plate velocity, which makes older relief trail away from the source. Separate 1,024-block candidates place arc-associated stratovolcano candidates near active convergent boundaries. Broad shield volcanoes and steeper stratovolcanoes lift oceanic islands, carve summit calderas or craters, optionally fill settled crater lakes, and emit basalt or ash fields. A crater lake uses an absolute local volcanic profile and a coordinate-warped radial distance, so its shoreline is irregular without inheriting an unrelated macro-height tilt. Generation samples the complete rim in 96 directions and accepts the lake only when every direction retains at least one full block of freeboard. It then emits a supported dry bank around the standing water and rejects the lake if a safe wet radius cannot fit inside the rim. These crater lakes are endorheic; ordinary routed lakes can instead retain the named outlets, rapids, and falls described below. The same global primitives reconstruct obsidian or lava-bearing conduits and three or four curving lava tubes inside every intersecting cube. These are static generated landforms; eruptions and propagating lava remain outside this version.

Provisional elevations are smoothly compressed and clamped to Y=-112 through Y=480. This leaves bedrock space below and headroom above generated terrain.

### Drainage, incision, lakes, and coastal deposition

`BasinSolver` keys one immutable solution to each 2,048-block catchment. A jittered site represents the catchment. Each non-ocean site chooses a lower cardinal neighbor under a stable elevation, random tie, and coordinate ordering. Adjacent catchments reconstruct the same boundary portal. Portal discharge combines exact provisional rainfall from four upstream site rings with a deterministic coarse contribution for more distant drainage.

Each catchment holds a 16-block raster plus a two-cell apron. Relief, rainfall, and rock-resistance callbacks are evaluated on a globally aligned 64-block input grid and interpolated onto the numerical raster. A four-cell boundary blend and exact reconstruction of the shared portal make terrain, water elevation, flow, discharge, and channel shape agree across catchment boundaries. Priority-Flood fills routing depressions, then an angular D-infinity-inspired routing step splits flow between the two downhill raster receivers around the local aspect, with a strict lower-neighbor fallback. Provisional rainfall and portal contributions accumulate through that acyclic routing.

Eight fixed passes apply stream-power incision, sediment-capacity transport, deposition, and thermal hillslope relaxation while keeping the shared boundary locked. Priority-Flood and D-infinity run again over the eroded surface before Strahler order is assigned. Discharge, order, gradient, sediment, and rock resistance determine channel width, depth, floodplain reach, canyon incision, and gorge shape. High-gradient site-to-portal guide corridors use coordinate-addressed curved centerlines and bounded width variation rather than one straight segment. Their displacement and tangent return smoothly to the canonical portal, and the locked seam band remains unchanged, so long gorges bend without introducing a catchment boundary or breaking bilateral portal values. Nearly level guides remain undisplaced because a lateral cut there can manufacture an invalid spill basin.

Routed channel identity takes precedence over the below-sea fallback, so incision below sea level does not reclassify a river as ocean. A one-raster-cell support band raises the immediate dry channel bank to the routed water level, while named outlets remain open. Legacy knickpoint flags shape rapids, but only bounded analytical drops emit explicit falling-water state. Outlet-fall lookup uses immutable local raster buckets, so sampling cost does not grow with the number of falls in a basin.

Qualifying depression components become flat lakes at their spill surface; small or shallow depressions remain filled only for routing. Lake-body sampling groups contributors by flat water level and endorheic state. A dry or different-body contributor contributes zero lake depth and keeps its original interpolation weight instead of causing the wet weights to be normalized back to full depth. Depth therefore tapers to zero at a real shoreline rather than projecting a floating sheet beyond it. The solver reconstructs every retained lake floor from its flat level minus positive depth, removes negligible members, and raises the neighboring dry rim to the spill level. Named outlet receivers and established channels remain open, so this support rule does not dam valid streams, rapids, or falls. The `ColumnPlan` 17 by 17 authority carries the resulting membership and depth into exact density and cube emission. Every generated shore-water voxel consequently occupies a canonical wet column with terrain below its top and a solid supporting block below its lowest water voxel.

Routed basins continue toward an outlet, while terminal catchments receive bounded endorheic lakes. A high-discharge, steep channel over resistant rock can form an ordinary channel waterfall. A nonendorheic lake whose selected receiver is at least 2.5 blocks lower additionally records one immutable `OutletFall`. Its top and bottom surfaces, width, start-to-receiver flow direction, and receiver anchor are separate from the receiver's standing `waterSurface`. Exact generation overlays a short, narrow falling footprint centered on that receiver and marks its cells as explicit falling water from the lower surface through the upper lip. This connects the two bodies without extending a long water slab, discarding the elevated lake, or raising the lower receiving body. Sediment-rich, low-gradient rivers entering shallow ocean water build depositional fans with two through four distributaries.

Channel, lake, waterfall, and delta geometry is carved directly into terrain before cubes are emitted. Ocean, river, lake, crater-lake, and outlet-fall cells receive finished block and fluid state during generation and enter no runtime fluid queue. The far renderer reconstructs each outlet fall as one receiver-centered five-quad prism, with four sides and one top. Exactly one half-open anchor tile owns the complete prism, even when it crosses a tile face. The prism extends into the lower body's top source voxel so it overlaps the visible surface, reaches the upper lip, and does not alter or cover the receiving body's standing surface. Basin solutions are stored in a byte-accounted 64 MiB single-flight LRU cache. Exact and far terrain share a process-wide permit gate that admits no more than two distinct cold basin constructions at once; cache hits and callers sharing an existing future do not consume another permit. That budget accounts for retained immutable solution payloads; transient construction and allocator overhead are measured separately. Validation rejects nonfinite fields, invalid sinks or receivers, uphill channel water, lakes without positive depth, invalid lake outlets, and malformed outlet-fall extents. Construction failure falls back to un-eroded base terrain with deterministic outlet metadata instead of publishing an invalid solution.

Each `BasinSolver` instance represents one immutable callback context. Its elevation, rainfall, and rock-resistance callbacks must be coordinate-pure and must describe the same fields for the solver's entire lifetime because the catchment cache key contains seed-space coordinates, not callback identity. Changing any field requires a new solver instance. Clearing the cache does not change the callback context.

### Final climate and soil

Climate uses nonrepeating noise fields rather than latitude bands. A pressure-field gradient plus a stable rotational term determines wind. Moisture evaluates 17 points connected by 16 upwind intervals of 256 blocks, for a bounded 4,096-block path. Each point samples the incised hydrology result. Oceans, lakes, and rivers recharge moisture with relative weights 1.0, 0.65, and 0.18. Rising terrain removes moisture as precipitation, descending terrain creates a lee-side drying response, and accumulated water exposure moderates temperature.

Temperature begins with a synthetic insolation field. Elevation cooling uses a lapse rate of 6.5 C per 1,000 climate meters with eight climate meters per vertical block. Annual precipitation and temperature produce potential evapotranspiration and aridity. Rock type, soil noise, water proximity, drainage, and sediment context then produce soil moisture, fertility, and the water table.

The final climate uses the incised local terrain height and generated water along the bounded upwind profile. Discrete volcanic relief is applied after that macro integration. Its emitted-height lapse correction and any crater-water moderation are local corrections only; they do not rerun the 4,096-block atmospheric path.

### Biomes and ecotopes

All 33 persisted biome values are append-only. The original 14 values remain unchanged, the first climate expansion occupies values 14 through 26, and values 27 through 32 add montane grassland, flooded grassland, Mediterranean woodland, temperate conifer forest, tropical conifer forest, and tropical dry forest.

The land-biome palette has at least one reachable representative for each of the 14 terrestrial biome classes used by the [One Earth Bioregions Framework](https://www.oneearth.org/bioregions-2023/) and the World Wildlife Fund's foundational [Terrestrial Ecoregions of the World](https://doi.org/10.1641/0006-3568%282001%29051%5B0933%3ATEOTWA%5D2.0.CO%3B2) classification. Desert, montane grassland, steppe, savanna, flooded grassland, mangrove, Mediterranean woodland, temperate broadleaf forest, temperate conifer forest, tropical conifer forest, tropical dry forest, tropical rainforest, taiga, and tundra provide that correspondence. Additional game biomes represent oceans, river and coast context, unusual geology, local floristic variation, ice, and glaciers. This is a procedural climate classification, not a claim to reproduce a specific real ecoregion.

Continuous suitability scores combine temperature, precipitation, aridity, fertility, elevation, slope, geology, and water context. The score fields are evaluated continuously and `BiomeBlend` retains the strongest and second-strongest results with normalized influence. Counter-addressed multiscale dithering chooses between them for surface and subsurface materials, tree candidates, and per-column flora. Each use has a stable subsystem stream, so broad transition zones mix palettes at several spatial scales without creating a hard biome line, changing terrain shape, or depending on cube order.

Ecotopes describe local landforms without creating a combinatorial set of biome IDs. Riverbanks, floodplains, deltas, lakeshores, coasts, cliffs, scree, canyons, geothermal areas, caves, and aquifers can overlap a climate biome. Eight continuous elevation ecotopes add valleys, foothills, montane slopes, subalpine terrain, alpine zones, snowfields, glaciers, and exposed peaks. Their overlapping weights vary surface cover and ecology gradually across elevation rather than imposing one fixed snow line.

Snow and ice are climate outputs. A surface freezes at or below -1 C when annual precipitation is at least 120 mm or when its primary biome is ice spikes, glacier, or frozen ocean. Tundra keeps its cold surface palette but does not independently force water to freeze. There is no fixed snow line.

### Cubic density and materials

The solid test is `D(x, y, z) > 0`. Macro relief supplies the column height and detail amplitude. Existing three-dimensional density contributes anisotropic ledges and overhangs, cheese caverns, spaghetti tunnels, and deep noodle caves. Density is evaluated on a world-aligned 4 by 4 by 4 lattice, then interpolated with one fixed operation order. Neighboring cubes therefore sample the same density at the same world position.

Water-covered columns seal caves near the surface. Dry high ground can expose cave entrances. Bedrock fills the bottom two layers and deterministically dithers part of the third. Aquifer ecotopes can emit bounded ellipsoidal water pockets below the water table. A clay or limestone shell seals each pocket, so aquifers do not flood an arbitrary connected cave. Static lava may occupy deep open cells at or below Y=-96. Lava does not flow.

Geology, the dithered biome blend, elevation ecotopes, and surface context select stone, andesite, basalt, obsidian, limestone, sandstone, volcanic ash, mud, clay, silt, gravel, sand, dirt, grass, snow, and ice. A plate-keyed smooth stratal field tilts and folds material layers without resetting at arbitrary material cells. Continental convergent arcs expose andesite intrusions, oceanic and older basaltic crust retain basalt-rich layers, volcanic conduits can expose obsidian, and limestone and sandstone regions keep their sedimentary sequences. Delta beds use silt, inland water beds use mud or clay, and volcanic surfaces use basalt or ash. Exact terrain and every far LOD call the same coordinate-pure material evaluator, so geology does not pop at the cubic handoff.

All block identifiers are append-only because save payloads store raw bytes. `BlockType::COUNT` is 58. A compile-time `BlockDefinition` table defines render shape, collision solidity, opacity, targeting, liquid behavior, and leaf behavior for every value. Render shapes are cube, cross-quad, flat, liquid, or none. Lily pads use flat geometry.

### Ores, structures, and plants

Every feature is anchored in global horizontal cells and clipped to the cube being generated. A generator never writes into a neighboring loaded cube.

- Ore random walks are re-evaluated from nearby source columns and clipped in X, Y, and Z. Bands now extend from Y=-120 through Y=160 depending on ore.
- Ruins, wells, and houses are anchored once per 8 by 8 horizontal chunk region. The writer clips each structure to the current cube, so a vertical or horizontal boundary does not transfer ownership.
- Trees place one counter-addressed candidate per 8 by 8 global feature cell. A candidate first passes species traits for temperature, soil moisture, fertility, light, slope, altitude, flooding, and biome density. It is accepted only when its stable priority beats every candidate inside the larger of their species spacing radii. Spacing ranges from 8 through 14 blocks and is independent of cube request order.
- The tree query expands six blocks beyond the target cube footprint, reconstructs every winning anchor, and clips output to that cube. The ten rooted forms are oak, branched large oak, birch, conical spruce, bent acacia, four-trunk jungle trees with buttress roots and branches, rooted mangroves, leaning palms, hanging willows, and alpine scrub. Fallen logs use the same anchored placement system as a separate form.
- Per-column flora reads soil moisture, fertility, relative humidity, slope, ecotopes, water, and a counter-dithered primary or secondary biome. It emits tall grass, three flower colors, mushrooms, ferns, shrubs, cattails, reeds, cacti, dead bushes, succulents, and lily pads. Riparian plants use riverbank, lakeshore, floodplain, and mangrove context.
- Tree and flora candidates also query the emitted substrate. Rock, scree, volcanic ash, deep snow, and water reject ordinary roots unless the selected species explicitly tolerates that surface. Exact and far tree summaries therefore agree on where vegetation can plausibly stand.

This is a locally evaluated priority distribution inspired by infinite Poisson-disk work. It does not precompute Poisson or Wang tiles. Global cells, counter randomness, radius competition, and clipped reconstruction provide the required order-independent spacing directly.

## Runtime water

Generated water is settled geometry. Every standing generated water block from the supported floor through the top water voxel is an implicit source, including when that vertical volume crosses cube boundaries. A cube allocates no fluid array for those source cells; only an explicit runtime level or falling state materializes an entry. Generation and ordinary cube loading enqueue no water updates. Only a gameplay block edit activates the changed cell and its six face neighbors.

`FluidState` occupies one byte:

- Bits 0 through 2 hold source level 0 or flowing levels 1 through 7.
- Bit 3 marks falling water.
- Other bits are reserved.
- `0xFF` inside an explicit cube array means the generated source state remains implicit.

The scheduler advances at 20 Hz with a five-tick update delay. One fluid tick processes at most 1,024 deduplicated cells. Pending updates and deferred frontiers are each capped at 65,536, and catch-up is limited to eight ticks. Deferred frontiers are indexed by the unavailable destination cube, and each tick resumes only a bounded number whose destination has become loaded. Loading one cube therefore does not scan or activate the full frontier set. Water attempts downward flow before horizontal flow, increases horizontal distance levels, removes unsupported flow, and forms a source only from two horizontal sources over a solid block or source water.

Fluid reads never load a cube. Missing collision cubes are closed, while a water update that reaches a missing cube records an activated frontier under the destination-cube index. Loading that cube makes only matching frontiers eligible for the bounded resume budget. Deferred frontiers are stored in column manifests and restored with the world. Static generated water without a frontier remains inactive on load. Pending-update and deferred-frontier overflow counts are surfaced in diagnostics instead of being silent.

The mesh snapshot includes blocks, fluid bytes, and block light in an 18 by 18 by 18 halo, plus a separate 18 by 18 array of per-column sky cutoffs. Water uses eighth-block cell heights, corner averaging, flow-direction bits, and a falling flag while preserving the 16-byte vertex layout. Stable source and flowing cells emit top geometry only. Vertical sides are emitted exclusively for explicit falling columns, including the narrow `OutletFall` overlay, which prevents lake, river, and ocean edges from becoming walls. Buoyancy and head-submersion checks query the same runtime fluid height used by rendering.

An implicit generated source has its visible surface 0.875 blocks above the water voxel floor. The far sampled representation quantizes its source-water top to that same plane, so exact and far water do not meet at different heights.

## Habitat-driven fauna

Nine entity types are exhaustive through `EntityType::COUNT`: sheep, cow, pig, chicken, deer, goat, rabbit, frog, and fish.

Wild populations use deterministic 64-block territories. Each territory has a stable seed-derived anchor and stable entity IDs. A habitat score chooses one species and a carrying capacity of up to four land animals or six fish. The hard cap is 64 living animals, including babies. Territories activate within 96 blocks and wild animals despawn beyond 112 blocks.

Habitat reads `SurfaceSample` directly. Temperature, soil moisture, fertility, slope, precipitation, biome suitability, channel width, discharge, water-body geometry, and ecotopes produce food, cover, river size, and water depth. Loaded water can refine the procedural depth estimate. Deer favor cover, goats favor steep high ground, rabbits favor fertile open ground, frogs require warm wet ground near water, and fish require sufficiently deep generated water. Habitat suitability does not claim a global water-connectivity solve.

Movement modes add climbing for goats, hopping for rabbits, amphibious hopping for frogs, and three-dimensional swimming for fish. Flocking supplies herd and schooling steering, fish physics confines fish to loaded water, and nearby wildlife flee when the player's actual horizontal displacement closes the distance to them.

## Streaming and meshing

The active cubic set is rebuilt when the player crosses a cube boundary or relevant cold column plans complete. Gameplay only submits a request; one utility-priority planner retains the latest camera position and cancels superseded work before publication, so entering a cold area cannot execute selection, sorting, apron expansion, or unload scans on the fixed-tick and render thread. One rebuild gathers the unique horizontal columns in the exact disk, expands that set through the fixed plan apron once, and requests each resulting plan no more than once. Pending plans register their dependent active-set columns in an index. A plan completion wakes only its indexed dependents instead of scanning every retained cube. Completion notifications batch after 128 plans or backlog drain, and the fixed tick consumes at most one every four ticks. Exact simulation uses `min(viewDistance, 32)`, even when the visible horizon is 256 chunks:

- Every `ColumnPlan::exposedSections` result in the exact-radius-plus-one disk, with a sea-level placeholder while a cold plan is still building
- A radius-six exploration band extending four cubes above and below the camera
- Every saved edited section listed by a visible column manifest
- A complete one-cube, 26-neighbor halo needed by collision and 18 by 18 by 18 meshing

The exact mesh-candidate set is capped at 16,384 cubes and the retained set at 32,768 cubes. The camera exploration and collision band is a hard highest priority, followed by exposed surface, saved edits, and then nearest three-dimensional distance. Cap pressure therefore cannot discard the radius-six by vertical-four band around an underground player in favor of distant surface sections. Unload hysteresis retains an existing cube through two extra horizontal chunks and one extra vertical cube beyond the current target. This hysteresis is distinct from the targeted one-cube mesh and collision halo. Four utility-priority generation workers process nearest-first column-plan and cube backlogs. At most two cold plans and 64 cube jobs are in flight. Exact mesh workers use user-initiated priority so near-camera results outrank speculative generation. Completed cube insertion uses `try_emplace`, which preserves an already loaded edited cube if duplicate work finishes later. Performance logs report planner requests, coalesced replacements, canceled stale builds, and build-time EMA.

Rendering reads a revision-cached immutable vector of loaded cube pointers. Two mesh workers admit at most 64 total items across queued, building, completed, and renderer-pending states and produce version-stamped results from 18-cube-edge snapshots. Snapshot construction copies loaded boundary block, fluid, and block-light data plus separate generated-surface and conservative skylight cutoffs under the world mutex. An unavailable in-range halo follows its generated terrain silhouette. Cells above the cutoff remain air and cells below it remain opaque. A visible uphill continuation uses the arriving column's lit surface material, while an opening below the local surface receives an inward-facing, unlit bedrock cap. Loading or unloading a halo cube dirties all affected neighboring meshes. Mesh work then runs without the world lock. An edit dirties the touched cube and every face, edge, or corner neighbor whose one-block halo intersects the changed block. Exact upload work stops after 64 meshes or 32 MiB in one frame.

Cube mesh origins, AABBs, candidate distance, frustum tests, and water sorting include Y. Missing collision cubes behave as bedrock and do not force generation. A generated skylight cutoff applies only when every vertical section through that cutoff is loaded; an incomplete path is fully occluded so sunlight cannot pass through an unloaded gap above an underground camera. Block raycasts stop at the first missing cube, and break or placement operations revalidate that their destination is loaded without generating it. The renderer retains a synchronous near-camera rebuild path for up to two already-meshed edited cubes per frame; initial meshes use workers.

### Far visible terrain

Immutable 256 by 256-block surface tiles fill the half-open annulus `[32, 256)` chunks outside the exact radius. A two-block sampling tier immediately outside radius 32 is the topology bridge: it samples the exact emitted density height used to choose exact exposed cubes. Independently, whole far tiles overlap the exact disk. Exact opaque terrain draws first, and a small positive depth bias leaves overlapping far tops behind it as lit fallback while exact meshes are cold. Water and canopy summaries retain exact ownership through radius 32 and use a stable world-space dither across the following 16 blocks. Farther out, distance plus immutable maximum sampled slope and hydrology complexity select among four-, eight-, and sixteen-block tiers. Their threshold values remain tunable implementation parameters rather than rigid rings. The previous tier applies asymmetric refine and coarsen thresholds to stop boundary chatter. Once a replacement is resident, a 0.4-second transition fades the old topology into fog, swaps at the hidden midpoint, and fades the target out of fog. Tile construction uses coordinate-pure geology and hydrology, globally aligned borders, finer-to-coarser transition skirts, and greedy merging of equal flat terrain.

Far tiles are rendering only. They carry no caves, structures, per-block flora, fauna, collision, edits, persistence, runtime water, or exact biome transition detail. Steps two and four reconstruct deterministic visual canopy impostors from the same accepted tree anchors as cubic generation. Steps eight and sixteen use globally anchored aggregate forest cells at 32- and 64-block spacing. Each accepted cell emits one larger grounded trunk-and-crown cluster after coordinate-pure climate, substrate, slope, and water checks. Half-open cells and tiles own complete impostors even when a canopy crosses a tile face. Bit 28 preserves the canopy classification in the shared vertex contract and diagnostics. The exact-to-far predicate clips canopy fragments through the exact radius and dithers them through the handoff band without affecting opaque fallback terrain.

Far shorelines use contour-clipped triangles so a partially wet coarse cell cannot create a rectangular ledge, and far water emits top geometry only at the same 0.875-block source plane as exact generated water. Exact and far opaque terrain share depth, while water samples resolved depth without a depth attachment. Opaque terrain therefore uses depth-backed cold-residency fallback, and water and canopies use the 16-block fragment handoff. Per-draw edge metadata exposes a marked skirt only where a resident finer tile borders a resident coarser tile, and the shader suppresses all skirts inside the handoff. Frustum culling, counterclockwise back-face culling, and a conservative 256-bin terrain-horizon test reduce the bounded direct draw list. This adaptive tiled LOD is inspired by geometry clipmaps and CDLOD, but it is not a literal geometry clipmap or hierarchical Z buffer.

Four far workers retain at most 64 pending jobs and 32 completed results. The CPU cache holds at most 1,024 tiles and 512 MiB. The far GPU arena reserves 256 MiB of vertices and 128 MiB of indices. Per-frame far uploads stop after 12 tiles or 32 MiB. All exact, far, post-processing, world, and transient allocations share a 64 GB unified-memory acceptance ceiling.

## Determinism

Discrete stochastic world-generation choices use `CounterRng`, a 10-round Philox-style counter generator keyed by the world seed, subsystem stream, full-width coordinates, and candidate index. This includes geology candidates, catchments, volcanoes, aquifers, materials, ores, structures, trees, and flora. Continuous Simplex fields instead use an immutable permutation derived from the world seed. Neither mechanism has mutable query-order state, so generation does not depend on worker order.

The determinism contract is:

- A cube writes only its own storage.
- Coordinate conversion is identical for generation, streaming, meshing, physics, and saves.
- Density interpolation has one operation order.
- Cross-cube features reconstruct the same anchor from the same seed.
- Generated water never runs a settling simulation.
- Exact and far cache eviction may change cost, but not results.
- Far tile construction depends only on its key, seed, coordinate-pure samples, and exact or aggregate canopy cells selected for its LOD.

Block light is derived state, not saved world data. `LightEngine` floods from `blockLightEmission` sources through transparent cells, losing one level per block, and the reconciliation queue pulls it across all six cube faces until quiescent. The monotone flood over fixed blocks has one fixed point regardless of cube load and reconciliation order. Light is recomputed after generation or load, never serialized. Tests pin falloff, opacity, generation-order independence, and cross-cube agreement.

Regression coverage checks negative coordinate boundaries, vertical limits, uniform and dense storage, full-width counter addresses, seed and request-order independence, surface query consistency, and exact column-plan surfaces. Canonical-lake tests cover the complete 17 by 17 authority, dry-to-wet taper, supported shore occupancy, negative column faces, flat interior depth, and the absence of floating water or artificial shoreline walls. The seed-42 probe at X=-8235, Z=2976 is a shallow supported nonendorheic lake lip. Separate fixtures pin the incised river across the X=-12288 cube face at Z=2653 and Z=2654 and the canyon ecotope at X=-23904, Z=0. Another seed-42 fixture routes the elevated lake at X=-8272, Z=3056 through the receiver-centered outlet fall at X=-8256, Z=3072 into a lower standing river. It pins distinct top and bottom surfaces, bounded width and flow footprint, supported exact falling cells, the unchanged receiving-water level, five-quad far ownership at every LOD, and deterministic cache rebuilds.

The fixed seed-764891 caldera fixture samples all 96 rim directions. It requires an irregular shoreline with at least six distinct radii spanning at least six blocks, no emitted terrain step above two blocks, a solid dry bank around the full endorheic perimeter, and a water surface at least one block below the validated rim. It also walks every generated water voxel from floor through surface at the center and across a cube face, requiring implicit source state without an explicit fluid array. Cache clearing and reverse traversal reproduce the same perimeter.

Mesh tests exercise real and missing neighbors on all six faces, lit aboveground silhouettes, dark underground caps, halo invalidation, and skylight blocking across unloaded vertical sections. Streaming and entity tests pin the hard exploration-band priority, exact unload hysteresis, indexed plan completion, coalesced rebuilds, closed collision, and raycasts and edits that never cross a missing cube. Basin regressions also fix a multi-distributary delta sample, bound straight runs along high-gradient curved guides, pin both sides of their shared portal, prove single-flight construction, limit cold construction to two producers across independent solver instances, enforce the immutable callback-context invariant and 64 MiB bound, and compare concurrent results with reverse-order cache rebuilds. Fixed volcano and aquifer samples verify conduits, the enclosed settled crater lake, sealed pockets, and volumetric implicit generated source water. Far tests compare the two-block topology tier directly with exact emitted density heights and source-water planes at the radius-32 handoff. They also cover depth-backed opaque fallback, one shared 16-block dithered handoff for water and canopies, exact canopy reconstruction at steps two and four, aggregate forest ownership at steps eight and sixteen, shader marker ownership, tunable distance thresholds, complexity-sensitive selection, asymmetric hysteresis, fog-transition phases, negative tile coordinates, deterministic hashes, same-LOD borders, resident finer-to-coarser skirt masks, handoff skirt suppression, contour-clipped shorelines, outward winding, scheduler bounds, epoch cancellation, and conservative ridge occlusion. Persistence tests cover checksums, bounded save coalescing, and bulk manifest reads. Runtime-fluid tests cover indexed and budgeted frontier resumption plus surfaced overflow counts. Deterministic territory IDs, runtime approach-triggered fleeing, fauna movement, and the global entity cap have focused coverage as well.

For repeatable visual diagnostics, `RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, or `biome`. The overlay is a rendering aid and does not alter generation. `RYCRAFT_WORLD_SEED`, `RYCRAFT_SPAWN`, `RYCRAFT_YAW`, `RYCRAFT_PITCH`, and `rycraft_worldgen_inspect [seed] [sample_x sample_z]` provide the corresponding deterministic setup and samples. F3 reports one combined cache entry count and MiB total for column plans plus basin solutions, then separate far resident, drawn, culling, pending, cache, arena, fluid-work, and dropped-fluid metrics. The inspector reports feature coordinates with surface samples, optional requested coordinates, and column-plan and basin cache metrics separately, including basin bytes, hits, misses, builds, failures, active and peak cold constructions, throttled construction requests, and deterministic route timing.

## Save format and migration

RYCH v4 stores one LZ4-compressed file per edited cube:

```text
rycraft_world/regions/r.<regionX>.<regionZ>/c.<chunkX>.<chunkY>.<chunkZ>.dat
```

Region X and Z use floor division by 32. Writes use a temporary file followed by rename. The uncompressed payload begins with a packed 44-byte header:

```text
uint32 magic              0x52594348, "RYCH"
uint32 version            4
int64  chunkX
int32  chunkY
uint32 flags              uniform blocks, explicit fluid states
int64  chunkZ
uint32 blockCount         1 or 4,096
uint32 fluidStateCount    0 or 4,096
uint32 payloadChecksum    IEEE CRC-32 over following uncompressed bytes
uint8  blocks             one uniform value or 4,096 dense values
uint8  fluidStates        optional 4,096 packed values
```

Each saved horizontal column also has an atomically replaced manifest:

```text
rycraft_world/regions/r.<regionX>.<regionZ>/m.<chunkX>.<chunkZ>.manifest
```

The manifest lists edited section Y values and activated fluid frontiers. `SaveManager` indexes manifests at startup. Active-set reconstruction submits its unique visible-column list to one bulk manifest query, which copies the requested edited-section vectors under one short manifest lock and performs no disk access there. Its pending map keeps a queued cube readable until the save thread finishes, so an unload followed by an immediate return cannot expose stale disk data.

Only modified cubes persist. Procedural cubes, far tiles, meshes, skylight, and block light regenerate from the seed and current loaded neighborhood. The loader verifies the CRC-32 before deserializing block or fluid state. Unloading queues serialization and compression on the save thread. The queue is bounded to 32,768 cubic positions, and repeated snapshots for one queued position coalesce to its newest revision instead of growing the queue. Backpressure applies at the bound, while the quit path sweeps still-loaded modified cubes before flushing. Manifest serialization is ordered separately from the short in-memory manifest lock, so file I/O never holds the lookup lock. Metadata remains in `metadata.json` and records the seed, player position and orientation, health, selected hotbar slot, nine-slot hotbar inventory, world time, chunk format version 4, and generator version 2.

RYCH v4 deliberately rejects v3 chunk payloads. Existing v3 files are left in place and ignored, while compatible metadata is still read. Terrain and old chunk edits therefore regenerate under the v4 generator. Corrupt or incompatible cube data also returns no cube, reports that cube's failure once, and falls back to deterministic generation.

The streaming active-set builder includes manifest `savedSections` for visible horizontal columns, so an off-surface build is rediscovered without generating every vertical section.

## Research basis and implementation boundary

The generator uses the following work as design guidance:

- [Procedural Tectonic Planets](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf) motivates crust attributes, plate motion, boundary-driven relief, subduction arcs, and volcanic chains. Rycraft uses a flat coordinate field and bounded volcano primitives rather than a simulated spherical crust.
- [Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf) couples uplift, stream networks, fluvial incision, sediment, and hillslope response. Rycraft applies these relationships in eight fixed passes inside each bounded raster catchment.
- [Terrain Generation Using Procedural Models Based on Hydrology](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Genevaux-ACM_Trans_Graph-2013-Terrain_Generation_Using_Procedural_Models_Based_on_Hydrology.pdf) motivates hierarchical river features and terrain constructed around drainage. Rycraft combines catchment sites, shared portals, Strahler channels, lakes, waterfalls, and distributary fans.
- [Priority-Flood](https://richard.science/sci/2014_barnes_depressions_published.pdf) supplies depression handling for each catchment raster. [D-infinity flow routing](https://digitalcommons.usu.edu/cee_facpub/2507/) motivates aspect-based flow split between adjacent downhill receivers. Rycraft runs Priority-Flood and an angular two-neighbor D-infinity-inspired routing step before erosion and again on the eroded surface; the routing step is not a verbatim triangular-facet implementation of Tarboton's method.
- [Orographic Precipitation](https://earthweb.ess.washington.edu/roe/GerardWeb/Publications_files/MinderRoe_OrogPrecEncyc.pdf) motivates upwind moisture recharge, ascent precipitation, and lee drying. The implementation is a bounded procedural approximation, not a weather model.
- [AutoBiomes](https://cgvr.cs.uni-bremen.de/papers/cgi20/AutoBiomes.pdf) motivates climate-driven multi-biome suitability instead of discrete terrain switches.
- [Terrestrial Ecoregions of the World](https://doi.org/10.1641/0006-3568%282001%29051%5B0933%3ATEOTWA%5D2.0.CO%3B2), produced by World Wildlife Fund conservation scientists, supplies the 14 terrestrial biome classes also carried forward by One Earth. Rycraft maps reachable climate biomes to those broad classes without claiming to reproduce ecoregion boundaries.
- [Random123](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) motivates stateless counter-addressed randomness. Discrete stochastic world-generation choices use a `CounterRng` Philox-style 4 by 32-bit, 10-round construction. Continuous Simplex fields use an immutable seed-derived permutation instead.
- [A Procedural Object Distribution Function](https://graphics.cs.kuleuven.be/publications/LD05PODF/LD05PODF_paper.pdf) motivates locally evaluable infinite object distributions. Rycraft uses priority competition among global feature-cell candidates instead of precomputed Poisson-disk tiles.

The bounded tradeoff is deliberate. Each basin is an immutable 2,048-block local solve with a fixed raster, apron, and pass count. Every other sample searches a fixed neighborhood or integrates a fixed number of steps. This keeps random access finite and generation order irrelevant without claiming to simulate a complete evolving planet.
