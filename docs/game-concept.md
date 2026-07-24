# rycraft Game Concept

Rycraft is a native voxel sandbox for Apple Silicon Macs. It uses C++23, Cocoa, Metal, Core Audio, deterministic world generation, and persistent cubic edits without a conventional game engine.

## Vision

Build an explorable world whose terrain, water, climate, geology, and ecology reinforce one another at both block scale and a long visible horizon. The same world must remain recognizable through exact cubes and every far LOD, with no artificial walls, ledges, straight shoreline repairs, or missing-water gaps.

The world is horizontally unbounded and vertically finite. Generator v4 uses a learned InfiniteDiffusion-compatible macro authority rather than synthetic continents. Bounded procedural systems add geology, caves, materials, structures, water, and life without making query order part of the result.

## Pillars

1. **One authoritative world.** Exact terrain, far terrain, water, materials, and ecology share one generation identity and unsigned 64-bit seed. A model, runtime, provider, or algorithm mismatch fails closed.
2. **Native performance.** Qualification targets a lowest sustained one-second rate of at least 60 FPS at native resolution with 4x MSAA and a 512-chunk visible horizon on the documented M4 Max, within 64 GiB of total unified memory. Exact editable cubic simulation remains bounded, and every required surface through its 32-chunk radius takes generation, mesh, upload, CPU, and GPU capacity before optional distant detail or canopy enrichment.
3. **Physical continuity.** Learned elevation and climate establish macro form. Canonical water may carve a bed but may not raise a retaining wall or delete a wet route. Coarse LODs retain narrow water topology.
4. **Honest loading.** Coarse parents remain visible until connected replacements are resident. Exact collision changes ownership with the matching visual coverage epoch; otherwise canonical generated terrain and water remain the physics proxy. A renderer may not hide a crack with a downward skirt or a persistent fog band, and it may not publish an exact mesh before its bounded lighting transaction settles.
5. **Persistent authorship.** Player edits, inventory, and fluid frontiers belong to the exact cubic world. Immutable learned and hydrology authority can be regenerated only under the same fingerprint.
6. **Research-informed boundaries.** Documentation states which paper-compatible mechanisms are implemented, which are Rycraft extensions, and which acceptance work is still pending.
7. **Procedural breadth.** Block textures, geology, caves, plants, voxel fauna, regional weather, atmosphere, clouds, lightning, thunder, and sound are synthesized from code around the pinned learned terrain authority.
8. **Deterministic simulation.** Coordinate-addressed generation, saved world time, and bounded weather fields remain query-order independent. Generated water is settled at creation, while runtime fluid rules begin only after gameplay disturbs it.

## Core loop

Explore from Y=-128 through Y=1407 across oceans, coasts, rivers, wetlands, plains, forests, mountains, volcanic ground, caves, and high cold terrain. Mine, build, disturb local water, fly across the long horizon, and encounter wildlife selected from physical habitat.

A native 30-meter model pixel covers four blocks. One horizontal or positive-elevation block represents 7.5 meters. Sea level remains Y=64.

Day cycles into night over twenty minutes. One directional radiance authority crosses from sun to moon without competing lights: the sun stops contributing below the horizon, the moon stays subdued through civil twilight, and its deterministic 29.53058-day mean synodic phase scales the visible disc, light, shadows, and reflections. Regional weather, procedural sound, animals, water, and the live world continue behind the normal playing view. Pause menus stop simulation.

## World character

- **Terrain:** learned broad elevation, procedural lithology, folded strata, faults, pre-routed volcanic forms, cubic caves, aquifers, ores, and structures. V4 does not apply legacy hydraulic erosion; its bounded dry residual is slope and water-clearance gated after routing.
- **Climate and weather:** learned mean temperature, temperature variability, annual precipitation, precipitation variability, and lapse rate feed existing biome and habitat consumers. A separate deterministic regional system advects pressure, moisture, temperature, and instability fronts through saved world time to drive wind, cloud cover and type, precipitation, fog, aerosols, and storms without changing terrain or gameplay rules.
- **Water:** oceans, rivers, tiled spill-reconciled lakes, falls, connected groundwater-fed wetlands, deltas, brackish estuaries, and naturally filled volcanic craters use canonical authority for identity, stage, bed, flow, shoreline data, and bounded cross-page connectivity. Runtime flow begins only after a gameplay edit. A component that exceeds its page or cell bound fails closed.
- **Ecology:** append-only terrain and plant materials, rooted tree forms, fallen logs, flora, and bounded fauna territories remain climate and habitat consumers in PR 1. Ordinary roots reject water and unsupported substrates; wet-habitat species require suitable shallow water and a supporting floor. PR 2 introduces equilibrium plant-functional-type capacity for a more physical ecosystem authority.
- **Scale:** exact editable cubes occupy the near world. Immutable terrain-and-water parents and selected refinements cover the visible disk through view distance 512.

## Startup experience

The default title screen opens without selecting or mutating a world. The Worlds screen opens a compatible v4 profile, creates a fresh one, or explicitly creates a separate v4 successor from a legacy or incompatible profile. That request shows model required, downloading, verifying, compiling, loading, ready, and failed states with byte progress and retry, repair, cancel, or quit actions. A verified installed model is reused on later world opens. Retry reuses it in place, while a missing, stale, or changed completion marker causes a local full audit. Repair fetches only an asset that fails verification.

Verified models, ONNX Runtime, Core ML caches, v4 saves, and authority pages live under `~/Library/Application Support/rycraft`. They do not belong in Git or Conductor workspaces. Before first entry, a bounded coarse search chooses dry inland terrain and final exact plans verify a safe standing location. The legacy v3 generator is available only through `RYCRAFT_DIAGNOSTIC_V3=1` and does not save.

## Rendering feel

- Block-scale exact geometry and granular far voxel tiers beneath physical atmosphere and regional weather
- Canonical water that remains connected through step 32
- Terrain and water visible while optional tree and ground-flora enrichment is still pending
- Vegetated nearby and middle-distance terrain while exact flora sections converge
- Maximum-detail terrain around the player before distant refinement or vegetation consumes scarce
  workers, cache entries, upload bandwidth, or GPU arena space
- No downward LOD skirts or artificial shoreline walls
- Linear HDR, five shadow cascades, baked corner accessibility, propagated smooth voxel light, Hi-Z screen-space GTAO and denoised near-field SSGI, atmosphere LUTs, volumetric clouds and fog, water reflections, bloom, and a native bitmap UI
- Emissive lava, torch flames, and active furnace mouths that seed the same block-light, HDR, bloom, and indirect-bounce response
- Nonemissive 9/16-height beds with partial collision, authored face culling, corner accessibility, shadow casting, skylight transmission, and indirect-light reception
- Generated footsteps, block impacts, weather-driven wind and precipitation, delayed procedural thunder, and animal calls mixed through Core Audio

Shared transition rings keep adjacent displayed tiers within a 2:1 ratio and give both tiles identical canonical boundary heights without downward panels. The full LOD capture matrix still requires visual crack qualification.

## Screens

| Screen | Purpose |
|---|---|
| Title | Play or quit over the menu backdrop |
| World select | List, play, or delete saved worlds |
| World create | Name, seed, generation toggles, and starting game mode for a new world |
| Playing | Captured cursor, crosshair, hotbar, health and hunger, world, and entities |
| Paused | Resume, switch game mode, save and quit to title, or quit while simulation is frozen |
| Settings | View distance through 512, graphics quality, controls, sensitivity, and volume |
| Inventory | 36 item slots with a 2x2 crafting grid, or the paged creative palette |
| Crafting | 3x3 crafting grid opened from a crafting table |
| Furnace | Smelting input, fuel, output, and progress opened from a furnace |
| Chest | 27 storage slots opened from a placed chest, persisted with the world |
| Death | Respawn or return to title after health reaches zero |
| Debug HUD | Frame, exact coverage and conservative gap distance, far parent and refinement residency, drawable coverage frontier, culling, queue, cache, fluid, fauna, and world-generation diagnostics |

## Current scope and deferrals

The generator v4 foundation includes verified model installation and reuse, a fail-closed Core ML runtime boundary, an independent coarse, latent, and decoder backend, generation identities, fingerprinted terrain pages, unsigned 64-bit seed interfaces, learned macro adaptation, dry-land spawn validation, the 96-section vertical range, no-raising water corrections, step-32 topology probes, hard full-disk exact-surface priority, epoch-matched exact collision publication, bounded first-visible lighting, surface-before-canopy publication, no-draw cold-horizon preparation, protected near-field CPU and GPU reclamation, shared far transition topology, and zero production skirt quads.

Rendering and simulation systems include deterministic regional weather, non-destructive lightning and delayed thunder, propagated smooth voxel light, four blended detailed shadow cascades plus a terrain-horizon cascade, Hi-Z screen-space GTAO and temporally denoised near-field SSGI, physical atmosphere LUTs, volumetric clouds and cloud shadows, unified froxel fog and shafts, water reflections, bloom, grading, audio, and the complete menu flow. Missing exact halos close explicitly while their real neighbors load: aboveground openings receive lit planned surface continuations, enclosed underground openings receive dark inward caps, and missing vertical openings receive bedrock caps. Far terrain and water publish as the base payload; deterministic tree and ground-flora attachments arrive independently, remain visible through the exact flora handoff, and cannot delay or invalidate drawable base coverage.

The survival experience borrows Minecraft's structure as a placeholder to be reshaped later. Multiple named worlds live under a saves directory (the legacy `rycraft_world` is adopted in place), each with its own seed, game mode, and generation toggles for structures, fauna, weather, and the day cycle. Survival adds a stack-based 36-slot inventory, 2x2 and 3x3 crafting, furnace smelting, block hardness with wood, stone, and iron tool tiers, block drops as collectible item entities, health and hunger with Minecraft-matched saturation-fast and food-slow regeneration, fall and drowning damage, starvation, death with an inventory scatter, respawn, and huntable animals that drop meat. The interface borrows Minecraft's stack handling too: left and right drag distribute a held stack evenly or one-per-slot, and a double-click gathers a matching stack. Utility content matches Minecraft placeholders as well: buckets scoop and pour water and lava, chests store items and persist, torches emit steady propagated block light, shears cut wool from sheep, beds set the spawn and sleep through the night, and craftable boats float on water and carry the player with WASD steering. Creative keeps free flight, instant breaking, and an infinite item palette. Game mode switches from the pause menu.

The repository records a canonical startup digest and one provider-bound final authority-page reference hash. These are comparison inputs, not proof of complete real-model qualification. Fresh, reverse, concurrent, and cache-cleared real-model determinism, the 96-chunk connected entry route, complete 512-chunk cold settlement, the complete real-model water-body matrix, opened mixed-LOD captures, Metal validation, and full M4 Max acceptance still require evidence before production v4 can be called complete.

PR 2 adds static plant-functional-type fractions and connects flora, canopy, and fauna capacity. Dynamic succession, animated seasons, terrain-changing weather, snow accumulation, runoff, flooding, storm fire, migration, predators, food webs, moving plates, post-generation erosion, climate change, eruptions, lava propagation and mixing, hostile mobs, combat, experience, multiplayer, far-terrain caves or structures, hierarchical Z-buffer occlusion, indirect command buffers, and GPU-driven draw submission remain outside the current scope. Weather affects presentation and surface wetness only. The far renderer uses adaptive immutable tile tiers, not a literal geometry clipmap.
