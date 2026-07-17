# rycraft Game Concept

rycraft is a from-scratch Minecraft-like voxel game for Apple Silicon Macs. It uses direct Metal rendering, Cocoa windowing, Core Audio, and procedural terrain, textures, models, and sound effects without a conventional game engine or external art assets.

## Vision

Build a native voxel sandbox whose systems remain readable end to end while supporting long exploration, deep vertical terrain, persistent building, responsive movement, and a world whose geology, water, climate, and ecology reinforce one another.

The world is not a simulated planet. It is an infinite horizontal procedural field made from bounded, deterministic local solutions. That tradeoff keeps random access fast and makes the same seed stable regardless of streaming order.

## Pillars

1. **Native performance.** Target a lowest sustained one-second rate of at least 60 FPS at native resolution with 4x MSAA and a 512-chunk visible horizon on an Apple M4 Max, within 64 GB of total unified memory. Exact editable cubic simulation has a nominal radius of 32. Every visible far-tile coordinate requests a step-32 voxel parent before optional refinement, including coordinates inside that nominal disk. Eight far workers reserve four slots for missing parents while four urgent slots populate connected 16/8/4/2 targets before the complete parent disk is ready. Every far-owned fragment in the camera exploration band requires visible step-2 fallback, and every other far-owned fragment in the exact overlap requires step 8 or finer, including fragments in fully ready partial boundary tiles. Their step-32 parents remain resident but hidden. A separate drawable frontier treats protected base-only tiles as missing and suppresses farther resident islands, while revision-aware per-column masks govern the exact overlap. Ordinary atomic terrain swaps beneath narrow fog, two-phase canopy exchange, single-owner water, asymmetric hysteresis, LOD skirts, frustum culling, back-face culling, conservative terrain-horizon occlusion, greedy meshing, bounded worker lanes, and bounded queues pursue the eight-kilometer view without expanding simulation beyond the near exact disk.
2. **Procedural content.** Block textures, terrain, caves, water bodies, plants, voxel fauna, weather, and sound are synthesized from code.
3. **Deterministic worlds.** The same seed and coordinates produce the same geology, terrain, water, blocks, feature anchors, and wild territory IDs. Discrete stochastic choices use counter-addressed streams, while continuous Simplex fields use an immutable seed-derived permutation. Neither depends on mutable query-order state.
4. **Honest simulation boundaries.** A visible solid block collides, edits persist, missing collision cubes stay closed, and raycasts cannot cross unloaded space. Aboveground loading fronts follow a lit generated terrain silhouette; unresolved underground openings stay dark. Generated water is already settled, and Java-style flow begins only after a gameplay edit disturbs it.
5. **Research-informed shape.** Plate relationships, drainage, orographic moisture, and climate suitability guide the generator, while documentation states where the bounded procedural approximation differs from the cited research.

## Core loop

Explore a horizontally unbounded world from Y=-128 through Y=511. Cross oceans, plains, forests, deserts, wetlands, mountains, volcanic ground, river valleys, and cold highlands. Descend through cubic caves, mine depth-dependent ores, build with persistent block edits, disturb water, and encounter wildlife selected from local habitat.

Day cycles into night over twenty minutes. Weather, procedural sound, animals, water, and the live world continue behind the normal playing view. Pause menus stop simulation.

## World character

- **Terrain:** domain-warped plate regions, blended lithology contacts, folded and faulted implicit strata, convergent uplift, divergent rifts, transform faults, hotspot chains, volcanic arcs, overhangs, cliffs, curved eroded Strahler channels, signed-distance shorelines, lakes, waterfalls, distributary deltas, islands, calderas, validated irregular crater-lake rims, conduits, lava tubes, aquifers, and caves.
- **Climate:** rotated synthetic pressure and insolation fields, bounded upwind moisture, coastal moderation, elevation cooling, soil moisture, fertility, 33 continuously blended biomes, organic weighted material patches, and land representatives for all 14 terrestrial biome classes shared by the One Earth and World Wildlife Fund frameworks.
- **Ecology:** append-only terrain and plant materials, ten rooted tree forms plus fallen logs, dense climate-suitable forests, moisture- and ecotope-driven flora, and habitat territories for sheep, cows, pigs, chickens, deer, goats, rabbits, frogs, and fish. Tree cover and species composition blend continuously through biome, climate, soil, geological, tectonic, slope, elevation, light, and hydrological conditions. Ordinary trees reject submerged roots. Mangroves and willows are limited to suitable shallow water and remain connected to a supporting floor.
- **Water:** generated oceans, rivers, lakes, deltas, and waterfalls are stable at creation. Canonical column authority gives every standing body a supported full-height source-water volume from the first wet voxel above its floor through the surface, including across cube boundaries. Competing lakes retain distinct flat levels behind an irregular supported watershed, except where an owned outlet or channel corridor remains open. Routed rapids and outlet approaches use explicit eighth-block flowing levels, while covered volume and receiving pools remain sources. Body-aware far geometry keeps unrelated water bodies separate. Monotonic junction-to-portal channel profiles handle ordinary descent, while explicit falling columns connect valid abrupt drops and supply waterfall sides. Edited water follows delayed source, falling, and level rules without loading unavailable cubes.
- **Scale:** exact caves, buildings, fluids, and wildlife occupy the near nominal 32-chunk radius. Coarse immutable surface and water parents are requested across the visible disk through radius 512 so mountain chains, drainage, coasts, islands, and biome-scale relief can remain visible beyond exact simulation. The step-32 parent and selected 16-, 8-, 4-, or 2-block voxel refinements preserve the block aesthetic across the eight-kilometer horizon. Step-2 canopies retain exact tree anchors, while coarser aggregate forest tiers preserve stable species silhouettes and canopy mass.

The implementation boundary for each item is documented in [world-generation.md](world-generation.md). Feature labels describe procedural cues, not a claim of geological or ecological simulation.

## Feel

- **Look:** blocky procedural textures with a complete alpha-aware mip chain, varied geology, cubic cliffs and caves, alpha-cutout leaves and flora, partial-height water, a granular adaptive terrain horizon, texel-snapped shadows, ambient occlusion, volumetric clouds and light, weather response, and one linear-HDR grade.
- **Sound:** generated footsteps, block impacts, ambient wind, and animal calls mixed through Core Audio.
- **Movement:** pointer-locked mouse look, camera-aligned WASD, sprinting, repeated jumping, swimming, and creative-style flight. Flying and the larger vertical range make aerial and subterranean exploration first-class.
- **Wildlife:** herds and schools use local movement modes, flee and flock behavior, deterministic territory anchors, and strict population bounds.
- **Menus:** the title, pause, settings, hotbar, and debug HUD use the game's bitmap UI over the live world.

## Screens

| Screen | Purpose |
|---|---|
| Title | Play or quit over a live world view |
| Playing | Captured cursor, crosshair, hotbar, world, and entities |
| Paused | Resume, settings, or quit while simulation is frozen |
| Settings | View distance through 512, graphics quality, controls, sensitivity, and volume |
| Debug HUD | Frame, exact coverage and conservative gap distance, far parent and refinement residency, drawable coverage frontier, culling, queue, cache, fluid, fauna, and world-generation diagnostics |

## Current scope

Shipping systems include sparse 16 by 16 by 16 chunks, the finite vertical range, deterministic macro geology, continuous C2 climate and biome controls, organic lithology blends, deformed implicit strata, smoothly exaggerated tectonic relief, bounded Priority-Flood and D-infinity-inspired basin erosion with curved high-gradient guides, stable water-body identities, continuous shoreline distance, climate-driven biome suitability with organic weighted material transitions, elevation ecotopes, density caves, aquifers and volcanic interiors, structures, procedural materials and models, generator-version-three cubic v4 saves, planar full-height generated source-water volumes, explicit generated rapid and waterfall states, Java-style activated runtime water, nine fauna types, habitat-driven trees and dense forests, hard-priority underground streaming with closed dark missing boundaries, full-disk step-32 far-parent requests, revision-aware per-column exact ownership, separate parent and drawable coverage frontiers, progressive 16/8/4/2 refinement, protected step-2 exploration fallback, protected step-8 exact-disk fallback, ordinary atomic terrain swaps beneath narrow fog, two-phase canopy exchange, single-owner water transitions, frustum and terrain-horizon visibility, complete block-texture mipmaps, day and night, weather, audio, HDR shadows, SSAO, volumetrics, water reflections, bloom, grading, and the complete menu flow.

Missing exact halos close explicitly while their real neighbors load. Aboveground openings receive lit planned surface continuations, enclosed underground openings receive dark inward caps, and missing vertical openings receive bedrock caps. The remaining horizon performance debt is the synchronous far payload: terrain, water, and canopy work publish together, and measured cold canopy construction ranges from 250 to 1,165 milliseconds. Staged canopy attachment is the planned follow-up so an otherwise ready terrain and water parent does not wait for forest geometry.

Deliberately outside this version are a dynamic planet, moving plates, erosion after generation, seasons, climate change, eruptions, lava propagation and mixing, predators, food webs, migration, multiplayer, crafting, far-terrain caves or structures, hierarchical Z-buffer occlusion, indirect command buffers, and GPU-driven draw submission. The implemented tectonics, basin erosion, climate, and volcanoes synthesize a static world. The far renderer uses adaptive immutable tile tiers, not a literal geometry clipmap.
