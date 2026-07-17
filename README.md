# rycraft

A Minecraft-like voxel game for macOS, built from scratch in C++23 on Metal. Terrain, block textures, voxel models, and sound effects are generated procedurally at runtime.

![Platform](https://img.shields.io/badge/platform-macOS%2013%2B-blue)
![Apple Silicon](https://img.shields.io/badge/CPU-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-GPLv3-green)
![C++](https://img.shields.io/badge/C%2B%2B-23-lightgrey)

## Quick start

```bash
brew install meson ninja
meson setup build --buildtype=debugoptimized
ninja -C build
./build/src/rycraft
```

Xcode command line tools are required. Run the tests with `ninja -C build test`.

Controls: use WASD to move and the mouse to look. Space jumps, Ctrl or a double-tap of W sprints, and left or right click breaks or places blocks. Use 1 through 9 or the scroll wheel for the hotbar, F3 for diagnostics, and Escape to pause. In water, hold Space to rise or double-tap W to swim along the view direction. Double-tap Space to toggle flight; Space rises and Shift descends.

## What is in the game

- **Sparse cubic world:** 16 by 16 by 16 chunks stream through a world that is horizontally unbounded and vertically spans Y=-128 through Y=511.
- **Advanced procedural terrain:** deterministic plate relationships, hotspot chains, volcanic arcs, smoothly amplified mountain massifs, bounded Priority-Flood and D-infinity-inspired basins, curved channel and shoreline reconstruction, erosion, climate, soil, 33 blended biomes, organic lithology and material transitions, folded strata, elevation ecotopes, cubic caves, aquifers, ores, structures, vegetation, rivers, lakes, waterfalls, distributary deltas, islands, calderas, crater lakes, conduits, and lava tubes. Shared C2 macro-control tiles and rotated filtered detail reconstruct continuous climate, soil, biome, and geological fields without exposing their storage lattice. The terrestrial climate set includes representatives for all 14 biome classes shared by the One Earth and World Wildlife Fund terrestrial frameworks.
- **Eight-kilometer voxel horizon:** the scheduler requests a step-32 parent for every visible 256 by 256-block tile and pins active parents once they are resident. Eight far workers reserve four slots for missing parents while four urgent slots build connected step-16, step-8, step-4, and step-2 targets before the complete parent disk is ready. Every far-owned fragment that intersects the exact overlap is protected, including a fully ready boundary tile whose published exact requirements cover only part of the tile. The camera exploration band requires step-2 fallback, while every other protected overlap fragment requires step 8 or finer. Step-32 parents remain hidden in both protected regions, and protected fallback jobs bypass ordinary grace and transition limits. A separate drawable coverage frontier treats protected base-only tiles as missing, suppresses out-of-order distant islands, and fades the preceding 256 blocks into fog. Exact cubic terrain then takes ownership through revision-aware per-column masks derived from the current exact-coverage snapshot. Filtered voxel tiers can cross by several blocks, so a narrow terrain-only fog pulse hides ordinary complete-topology swaps instead of exposing partial mismatched surfaces. Canopies keep their full 650-millisecond target-in, source-out exchange. Unswayed coordinates make transition and coverage decisions immune to foliage motion. Greedy meshing, frustum culling, back-face culling, conservative terrain-horizon culling, and parent-pinned residency bound the radius-512 view.
- **Generated and runtime water:** world-generated water starts settled and never runs generation-time ticks. Canonical 17 by 17 column authority covers oceans, rivers, lakes, deltas, waterfalls, and supported banks. Generated standing water is a full-height source-water volume: every wet voxel from the first one above solid support through the visible surface is an implicit source, including across cube boundaries. Steep routed channel tops use deterministic eighth-block flowing levels 1 through 7, outlet throats carry a complete source-to-level-seven approach, and waterfall curtains carry explicit falling state into a source-filled receiver. Competing lakes retain their distinct flat levels behind a supported watershed divider, except where an owned outlet or channel corridor must remain open. Body-aware far geometry never joins unrelated standing surfaces, while monotonic channel profiles and explicit falls connect valid higher and lower water. Stable water emits planar top surfaces without artificial vertical walls; analytic fragment shading supplies ripples without moving the voxel mesh. Gameplay edits activate the same delayed Java-style source, falling, and flow-level rules, and an undisturbed generated arrangement is already a fixed point.
- **Climate-driven forests:** oak, large oak, birch, spruce, acacia, jungle, mangrove, palm, willow, alpine scrub, and fallen-log forms use continuous biome suitability, temperature, precipitation, soil, slope, elevation, lithology, tectonic stress, light, and hydrology. Dense forest biomes receive high canopy cover. Ordinary roots reject water and unsupported substrates, while mangroves and willows accept only their suitable shallow-water habitats and extend trunks or roots to the solid floor. Coarse step-32 geometry trusts that block-resolution habitat and root-water decision instead of rejecting a tree because unrelated water occurs elsewhere in its 32 by 32 cell, then grounds the accepted trunk on the displayed voxel.
- **Building and persistence:** loaded-only raycast block editing, lit planned silhouettes at aboveground loading fronts, closed and dark unresolved openings underground, an outline highlight, LZ4-compressed RYCH v4 cubic saves beneath the generator-version-three region root, bounded coalesced save work, and manifests for edited vertical sections and indexed fluid frontiers.
- **Habitat fauna:** sheep, cows, pigs, chickens, deer, goats, rabbits, frogs, and fish use deterministic territories, bounded populations, procedural voxel models, and movement-specific AI.
- **A full day:** a twenty-minute day and night cycle with moving sun, dawn and dusk skies, procedural clouds, weather, and skylight shading.
- **Native presentation:** linear HDR with texel-snapped shadow cascades, baked and screen-space ambient occlusion, volumetric clouds and light, weather, post-resolve water and screen-space reflections, exposure, bloom, lens flare, tonemapping, sharpening, complete alpha-aware block-texture mipmaps, trilinear 8x-anisotropic minification, and a native Metal UI.

The generator uses bounded research-informed approximations so cubes remain fast to query in any order. See [world generation and persistence](docs/world-generation.md) for the implemented algorithms, limitations, and research references.

Known limitation: far terrain, water, and canopy geometry still publish as one synchronous tile payload. Measured canopy work on cold tiles ranges from 250 to 1,165 milliseconds, so canopy-heavy construction can delay an otherwise ready terrain and water parent. Staged canopy attachment is a follow-up performance improvement. Missing exact halos already close with lit surface continuations, dark inward underground caps, or vertical bedrock caps while the real neighbor loads.

## Project structure

```text
include/, src/
  engine/      Game loop, game flow, input, camera, diagnostics
  render/      Metal pipelines, cubic mesher, water, textures, UI, entities
  world/       Macro generation, cubic chunks, fluids, saving and loading
  entity/      Player physics, fauna AI, habitats, spawning
  audio/       Core Audio mixer and procedural sound effects
  common/      Math, deterministic randomness, thread pool, logging
shaders/       Metal shaders compiled into one metallib
tests/         Headless Catch2 modules
docs/          Architecture, conventions, and domain references
```

More references: [architecture](docs/architecture.md), [rendering conventions](docs/rendering-conventions.md), [performance conventions](docs/performance-conventions.md), [code conventions](docs/code-conventions.md), and [game concept](docs/game-concept.md).

## Technology

| Component | Technology |
|---|---|
| Language | C++23 and Objective-C++ at the Cocoa and Metal boundary |
| Rendering | Metal 3 and MSL with shared C++ layout headers |
| Windowing and input | Cocoa, MTKView, NSEvent, and CG pointer lock |
| Audio | Core Audio DefaultOutput unit |
| Build | Meson and Ninja |
| Tests | Catch2 |
| Compression | LZ4 |

## Development

```bash
meson setup build --buildtype=debugoptimized
ninja -C build
ninja -C build test
meson setup build-release --buildtype=release
ninja -C build-release
clang-format -i <touched files>
```

`debugoptimized` keeps debug symbols and assertions while enabling the optimization required by world generation. Reconfigure an older unoptimized build directory with `meson setup --reconfigure build --buildtype=debugoptimized` before judging streaming or frame rate. Use a separate `release` build for the M4 Max acceptance run.

Useful deterministic and playtest hooks:

```bash
RYCRAFT_WORLD_SEED=42 ./build/src/rycraft
RYCRAFT_SPAWN=100,180,-240 ./build/src/rycraft
RYCRAFT_WORLDGEN_OVERLAY=geology ./build/src/rycraft
RYCRAFT_SHOW_DEBUG=1 ./build/src/rycraft
RYCRAFT_VIEW_DISTANCE=512 ./build/src/rycraft
./build/src/rycraft_worldgen_inspect 42
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft
RYCRAFT_CAPTURE=/tmp/frame.png RYCRAFT_CAPTURE_FRAME=500 ./build/src/rycraft
RYCRAFT_BLOOM=0 ./build/src/rycraft
RYCRAFT_NATIVE_WINDOW=1 RYCRAFT_PERF_WARMUP_FRAMES=1200 RYCRAFT_PERF_FRAMES=1200 ./build-release/src/rycraft
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=512 ./build-release/src/rycraft
```

`rycraft_worldgen_inspect [seed] [sample_x sample_z]` reports deterministic feature locations, surface footprints, water-body and shoreline data, lithology and material palettes, former-grid artifact measurements, far-parent coverage counts, separate column-plan, basin, shoreline-contour, and macro-control cache information, benchmark timing, and a route hash as JSON. The positional seed is optional and takes precedence over `RYCRAFT_WORLD_SEED`.

`RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, or `biome`. Performance acceptance is measured at native resolution with 4x MSAA and view distance 512 on an Apple M4 Max. The target is a lowest sustained one-second rate of at least 60 FPS, with exact cubic simulation capped at radius 32 and total unified-memory use capped at 64 GB. Hardware timing is reported separately from the deterministic limits enforced in CI. See [performance conventions](docs/performance-conventions.md) for the canonical seed-764891 route, measured CPU microbenchmarks, and the current validation status.

## Troubleshooting

- If Meson cannot find `xcrun metal`, install Xcode command line tools with `xcode-select --install`.
- The first setup may download Catch2 and LZ4 Meson wraps into `subprojects`; later setups reuse the cache.
- After a large build-definition change, run `meson setup --wipe build`.
- World data is relative to the launch directory at `./rycraft_world/`. Use a scratch launch directory for playtests that should not touch the normal save.
- Generator-version-two cube edits and fluid frontiers under `regions` are not migrated. They remain on disk, while generator-version-three terrain writes new RYCH v4 cubes beneath `regions-v3`.
- Generator-version-three metadata preserves the seed, player transform, health, selected hotbar slot, hotbar inventory, settings, and world time.

## Author

Ryan Lefkowitz ([rlefkowitz1800@yahoo.com](mailto:rlefkowitz1800@yahoo.com))

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
