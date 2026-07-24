# rycraft

A Minecraft-like voxel game for macOS, built from scratch in C++23 on Metal. Terrain, block textures, voxel models, and sound effects are generated procedurally at runtime.

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B-blue)
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

The default generator v4 profile requires Apple Silicon and macOS 14 or newer. Ordinary startup reaches the title screen without opening, creating, migrating, or verifying a world. Selecting a compatible v4 world, creating a fresh v4 world, or explicitly creating a v4 successor starts verification of the pinned terrain model pack and ONNX Runtime before any world or save object is created. The 2,308,318,566-byte required download set is stored under `~/Library/Application Support/rycraft`, outside Git and outside Conductor workspaces. Later world opens reuse that verified installed directory with a local marker and file-stamp check, not another download or full 2.3 GB read. If the marker is missing, old, or changed, the installer rechecks every pinned asset locally. A successful audit restores the marker without transferring the files; an invalid asset fails closed until Repair. Interrupted transfers continue from persistent staged bytes. Download, verification, Core ML compilation, and qualification may be retried or canceled from the title screen. Repair downloads only missing or invalid assets and retains valid model files and compiled Core ML caches. The scalar Coarse graph and fixed four-window Base and Decoder batches use the identity-bound `coreml-cache-v3-base4-decoder4x256` cache. Decoder tail batches repeat the last real window deterministically and discard padded outputs, so normal restarts neither redownload the ONNX files nor rebuild a different provider configuration.

Generator v4 implements the pipeline described in [InfiniteDiffusion paper v4](https://arxiv.org/abs/2512.08309v4). It pins the published [30-meter ONNX model revision](https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/tree/ad2df557eca5645f588766101cf3bc3682455c3e), uses the [Minecraft implementation at commit `23d3f50`](https://github.com/xandergos/terrain-diffusion-mc/tree/23d3f50e5108882bb88a03c3ab048aa63633a02f) as a compatibility reference, and runs through [ONNX Runtime 1.27.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.27.1). Rycraft's canonical water, geology, ecology, persistence, and rendering are extensions around that learned authority.

Controls: use WASD to move and the mouse to look. Space jumps, Ctrl or a double-tap of W sprints, and left or right click breaks or places blocks. Use 1 through 9 or the scroll wheel for the hotbar, F3 for diagnostics, and Escape to pause. In water, hold Space to rise or double-tap W to swim along the view direction. Double-tap Space to toggle flight; Space rises and Shift descends.

## What is in the game

- **Sparse cubic world:** 16 by 16 by 16 cubes stream through a horizontally unbounded world spanning Y=-128 through Y=1407. A two-word `VerticalSectionMask` covers all 96 vertical sections, including sky paths that cross section bit 64.
- **Learned macro authority:** generator v4 identifies a world with an unsigned 64-bit seed, pinned model and runtime hashes, Core ML provider settings, scale, window geometry, and algorithm revisions. The learned authority supplies elevation, four learned physical climate variables, and one derived lapse-rate field on a 30-meter native grid. Four blocks represent one native pixel, giving a 7.5-meter horizontal and positive-elevation block scale. Geology, caves, ores, and structures remain bounded procedural consumers. Deterministic volcanic primitives modify learned physical elevation before canonical hydrology, and a separately bounded dry residual may add at most 1.5 blocks after routing only where water and terrain-clearance gates allow it. V4 does not run the legacy synthetic post-hydrology relief or hydraulic-erosion paths.
- **Canonical generated water:** generation may lower terrain to form a bed, but it may not raise dry terrain to hold water or delete a wet route to resolve a level conflict. Exact cubes and far tiles consume stable water identity, stage, bed, flow, and shoreline data. Standing water remains source-filled, and abrupt stage transitions are legal only when explicit rapid or waterfall ownership accounts for them.
- **Gap-free far publication:** a terrain-and-water surface stage is drawable before canopy enrichment finishes. Cold entry requires safe FINAL spawn terrain, a connected step-32 parent frontier through 96 chunks, revision-ready exact spawn meshes, and one atomic 60-target FINAL near closure. The camera-aware closure contains 4 targets at step 1, 8 at step 2, 12 at step 4, 16 at step 8, and 20 at step 16. It begins during preparation as soon as the connected parent frontier reaches the near band. Ordinary perceptual refinements become eligible after gameplay opens. The configured horizon, including the 512-chunk default, remains selected, but unfinished exact publication through 32 chunks or connected visible desired-LOD debt pauses ordinary outer submission and publication after the 96-chunk prefix. Near jobs run nearest-first, rank horizontal distance before projected error within the nearby visible class, and may displace queued or dependency-parked outer parents. Outer submission and publication resume after both debts clear, and an out-of-order distant tile cannot appear as an island. Exact cubes own the first 32 chunks, and every required surface section in that complete disk retains generation, meshing, and upload priority over optional flora and distant work. The camera column ranks first, followed by the exploration band and then the rest of the required disk. Canopy workers remain at zero during entry preparation and until the connected drawable prefix exists. Gameplay then guarantees exactly one low-priority canopy worker even while protected, local, or exact publication debt continues. No second gameplay canopy lane opens. Missing drawable attachments rank ahead of FINAL ecology promotions, and a source attachment remains visible through an LOD transition until its compatible target arrives. Local far terrain uses 8 workers alongside exact debt, 12 after exact debt clears, and all 16 only after exact and local debt clear. Settled far tiers are step 2 through 64 chunks, step 4 through 128, step 8 through 256, and step 16 through 512. Step 32 remains coverage-only. Screen-space error may retain any finer tier. A nearer desired-LOD job can displace lower-ranked queued or parked outer parents. Current protected FINAL parents and children run before bridge prerequisites, and unused current capacity precedes directional prediction. A requested protected FINAL refinement may reclaim optional distant CPU or GPU residency and use the complete GPU arena, but coverage, displayed surfaces, transitions, and protected lineage remain pinned. Coarse parents remain available until a connected replacement and its legal exterior shell are resident. Step-32 cells use conservative topology probes so a narrow water route is not discarded merely because all sampled corners are dry. Downward crack-hiding skirts are disabled. Shared transition rings give adjacent tiers identical canonical boundary heights and half-open positive-area ownership.
- **Climate-driven vegetation:** oak, large oak, birch, spruce, acacia, jungle, mangrove, palm, willow, alpine scrub, fallen logs, grasses, flowers, reeds, cattails, ferns, shrubs, mushrooms, succulents, and dead bushes use continuous biome suitability, temperature, precipitation, soil, slope, elevation, lithology, tectonic stress, light, and hydrology. Dense forest biomes receive high canopy cover. Ordinary roots reject water and unsupported substrates, while mangroves and willows accept only their suitable shallow-water habitats and extend trunks or roots to the solid floor. Optional far attachments retain deterministic tree and ground-flora anchors through every displayed LOD, including PREVIEW-grounded surfaces, until exact flora sections are ready. Required exact terrain through 32 chunks always precedes optional flora in its own generation and upload lanes. After connected terrain becomes drawable, one dedicated low-priority canopy worker fills nearby and middle-distance vegetation without waiting for continuously replenished exact debt to reach zero. Coarse geometry trusts block-resolution habitat and water decisions, then grounds accepted vegetation on the displayed voxel surface.
- **Building and persistence:** loaded-only raycast block editing, conservative unresolved boundaries, LZ4-compressed cubic saves, bounded coalesced save work, and manifests for edited vertical sections and indexed fluid frontiers. The Worlds screen explicitly creates each v4 profile beneath `~/Library/Application Support/rycraft`, with `regions-v4`, `terrain-authority-v1`, and `hydrology-authority-v1`. Fingerprinted protected FINAL rectangles persist beside ordinary terrain pages so a finalized warm open does not reconstruct the same learned hydrology inputs. A fresh creation always reserves a collision-free directory. An incompatible v4 or legacy profile remains read-only and offers an explicit v4 successor action that copies compatible metadata into a separate current-identity profile without rewriting the source.
- **Dry-land entry:** a fresh v4 world searches the nonpersistent coarse model through a page-aligned 61.44-kilometer half-edge and ranks at most one proposal per 2,048-block hydrology owner. Startup checks the requested chunk first, then scans the owner's globally aligned four-block native raster across up to 16 workers for a flat center with a dry five-by-five safety buffer. It first tries to certify the complete cold footprint, then the exact 113 by 113 radius-zero safety footprint. A bounded relocation stays within the already materialized learned page so exact spawn validation does not open cardinal hydrology owners or request more model pages. If wider canonical water prevents that optimization, startup retains the strict 25-sample local certificate. A continental owner that cannot produce a conservative page-local proof may instead offer one learned positive-elevation provisional site, but it cannot start horizon work or persist metadata until the radius-zero exact plan independently rejects water, unsupported ground, blocked headroom, and excessive slope. World construction starts with a zero nominal exact radius; the mandatory mesh halo still keeps the camera column and its four cardinal neighbors active. Gameplay waits separately for the already-retained three-by-three-by-three collision halo around the finalized spawn, so closed missing-cube collision cannot become an invisible wall. During streaming, exact collision owns only renderer-published sections from the matching surface-coverage epoch. Every other planned column uses its canonical generated terrain and fluid profile as the collision proxy, and an unresolved column stays closed. A generated or loaded cube cannot become mesh-visible until its bounded first-publication lighting transaction settles. This residency check overlaps horizon preparation and adds no learned-authority footprint. The full 32-chunk exact disk streams after entry. Older v4 metadata is rechecked once under the current rule.
- **Habitat fauna:** sheep, cows, pigs, chickens, deer, goats, rabbits, frogs, and fish use deterministic territories, bounded populations, procedural voxel models, and movement-specific AI.
- **A full day and regional weather:** a twenty-minute day and night cycle drives an Earth-like atmosphere, one active sun or moon light, moon phases, stars, and physically scaled twilight. Deterministic pressure, moisture, temperature, instability, and front fields produce local wind, cloud type, rain, snow, fog, and non-destructive thunderstorms without changing blocks or gameplay rules.
- **Native presentation:** linear HDR with four blended detailed shadow cascades through 1,536 blocks, a coarse terrain horizon shadow through 8,192 blocks, propagated smooth voxel light, ray traced screen-space GTAO and temporally denoised near-field diffuse SSGI, physical atmosphere LUTs, true volumetric cloud layers and cloud shadows, unified froxel fog and shafts, post-resolve water and screen-space reflections, exposure, bloom, lens flare, tonemapping, sharpening, complete alpha-aware block-texture mipmaps, trilinear 8x-anisotropic minification, and a native Metal UI. Lava, torch flames, and active furnace mouths emit through filtered material masks and contribute to HDR, bloom, voxel light, and near-field indirect response. Beds remain ordinarily lit and nonemissive.

See [world generation and persistence](docs/world-generation.md) for the implemented v4 foundation and its fail-closed qualification boundary. The [generator v4 follow-up roadmap](docs/generator-v4-follow-up.md) specifies the sparse hierarchy, GPU, hydrology, distant-water, distant-flora, startup, and ecosystem work intentionally deferred from this PR. The plant-functional-type ecosystem layer remains a separate generator change. Existing biome, flora, canopy, and fauna systems currently consume a physical-climate adapter. The model supplies no biome IDs. The ecosystem follow-up begins with a golden native-grid crosswalk against the pinned mod's hand-written `BiomeClassifier`, then replaces the adapter with continuous PFT capacity informed by canonical water and soils instead of copying its fixed noise or vanilla sea rules.

Screen-space indirect lighting is intentionally near-field. Geometry or radiance outside the current frame cannot contribute colored bounce, while propagated skylight remains view-independent and supplies ambient accessibility through cave entrances and around loaded overhangs.

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
| Learned terrain | InfiniteDiffusion-compatible authority and pinned ONNX Runtime 1.27.1 Core ML runtime |

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
RYCRAFT_APPLICATION_SUPPORT_ROOT=/tmp/rycraft-v4-playtest ./build/src/rycraft
RYCRAFT_DIAGNOSTIC_V3=1 ./build/src/rycraft
RYCRAFT_SPAWN=100,180,-240 ./build/src/rycraft
RYCRAFT_WORLDGEN_OVERLAY=geology ./build/src/rycraft
RYCRAFT_SHOW_DEBUG=1 ./build/src/rycraft
RYCRAFT_VIEW_DISTANCE=512 ./build/src/rycraft
./build/src/rycraft_worldgen_inspect 42
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft
RYCRAFT_CAPTURE=/tmp/frame.png RYCRAFT_CAPTURE_FRAME=500 ./build/src/rycraft
RYCRAFT_CAPTURE=/tmp/materials.png RYCRAFT_SPAWN_MATERIALS=1 RYCRAFT_CAPTURE_FRAME=500 ./build/src/rycraft
RYCRAFT_START_SCREEN=playing RYCRAFT_WEATHER=storm RYCRAFT_TIME=6002 RYCRAFT_TIME_FREEZE=1 \
  RYCRAFT_CAPTURE=/tmp/lightning.png RYCRAFT_CAPTURE_LIGHTNING=23080,-111650,17,2 ./build/src/rycraft
RYCRAFT_BLOOM=0 ./build/src/rycraft
RYCRAFT_WEATHER=storm RYCRAFT_TIME=6000 RYCRAFT_TIME_FREEZE=1 ./build/src/rycraft
RYCRAFT_CLOUD_QUALITY=2 RYCRAFT_INDIRECT_LIGHT=2 RYCRAFT_SHADOWS=2 ./build/src/rycraft
RYCRAFT_NATIVE_WINDOW=1 RYCRAFT_PERF_WARMUP_FRAMES=1200 RYCRAFT_PERF_FRAMES=1200 ./build-release/src/rycraft
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=512 ./build-release/src/rycraft
RYCRAFT_WORLD_SEED=42 RYCRAFT_SPAWN=3200.5,215.05,-5307.5 RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 ci/run-v4-capture.sh cold-handoff /absolute/model-pack
```

`RYCRAFT_WORLD_SEED` accepts the full unsigned 64-bit range for generator v4. `RYCRAFT_APPLICATION_SUPPORT_ROOT` accepts an absolute external directory for isolated model qualification and disposable v4 playtests. Keep that directory outside the repository and Conductor workspace. `RYCRAFT_DIAGNOSTIC_V3=1` starts the legacy synthetic generator without opening or saving `rycraft_world`. `rycraft_worldgen_inspect [seed] [sample_x sample_z]` reports deterministic feature locations, surface footprints, water-body and shoreline data, cache information, benchmark timing, and a route hash as JSON.

`ci/run-v4-capture.sh` APFS-clones a verified model pack into a disposable `/tmp` Application Support root, redirects settings reads to that root without changing `HOME`, records once-per-second bootstrap and preparation progress plus the complete captured identity, camera, graphics, authority, and streaming state beside the PNG, and never writes to a user world, user preferences, or the source model pack. The first cloned launch performs one local SHA-256 audit because the cloned files have new stamps, then refreshes the marker inside the clone. Reuse the printed qualification root as its third argument to skip that audit and share cloned Core ML caches and generated authority pages across related captures. The helper never removes that directory.

Set `RYCRAFT_CAPTURE_CAMERA=x,y,z` when a visual check needs an aerial or water-centered view. The override moves only the camera after entry; the player remains at the canonical validated spawn, and the capture-only session still does not save.

The capture helper returns a failing status when a completed frame log contains a generation,
runtime, or Metal validation error. The PNG is retained for diagnosis but does not count as passing
visual evidence.

`RYCRAFT_SPAWN_MATERIALS=1` is honored only beside `RYCRAFT_CAPTURE`. It places a bed, a fixed -Z-facing chest, a supported floor torch, and fixed -Z-facing inactive and actively burning furnaces near the validated spawn, then relies on the existing capture no-save path so the fixture cannot modify normal saves. Beds have no persisted facing state.

For an installed real model, `rycraft_worldgen_inspect --v4-model MODEL_PACK SEED X Z final --dry-spawn` runs the same bounded coarse-to-final learned-land search and canonical-water screen used before v4 world entry. The screen prepares one proposed 2,048-block hydrology owner, checks the requested chunk, and searches its globally aligned four-block native raster for a center with a dry five-by-five safety buffer. It prefers a complete cold or 113 by 113 exact-safety certificate, including bounded same-page relocation, and retains the 25-sample local certificate as the shoreline-safe fallback. An all-positive continental owner may return a provisional learned site without a certificate, but the result remains nonpersistent until the radius-zero exact plan validates canonical water, solid support, headroom, and slope. The inspector binds authority pages to a deterministic temporary profile under the system temporary directory, never to `rycraft_world_v4`. Use `--profile /absolute/external/directory` to retain an explicit inspector cache. The JSON records that authority profile, candidate, ordinal, canonical-water rejections, local relocations, total selection time, learned selection time, canonical water-screen time, and the identity-bound static Base and Decoder batches. Add `--horizon-mesh` for the full configured 512-chunk settlement route. `--horizon-radius 64` is a shorter diagnostic only, and the JSON always reports the selected radius as `horizon.radius_chunks`.

`RYCRAFT_WORLDGEN_OVERLAY` accepts exactly `geology`, `hydrology`, `climate`, `biome`, `lod`, or `authority`. The LOD view colors exact terrain cyan and far steps 1, 2, 4, 8, 16, and 32 blue, green, yellow, orange, red, and purple. The authority view colors FINAL green and PREVIEW magenta. Performance acceptance is measured at native resolution with 4x MSAA and view distance 512 on an Apple M4 Max. The target is a lowest sustained one-second rate of at least 60 FPS, with exact cubic simulation capped at radius 32 and total unified-memory use capped at 64 GiB. Hardware timing is reported separately from the deterministic limits enforced in CI. See [performance conventions](docs/performance-conventions.md) for the canonical seed-764891 route, measurement procedure, and current validation status.

`RYCRAFT_WEATHER` accepts `clear`, `overcast`, `rain`, `storm`, or `snow` for stable captures. `RYCRAFT_CLOUD_QUALITY` and `RYCRAFT_INDIRECT_LIGHT` accept 0 through 2 for Off, Medium, and High. `RYCRAFT_CLOUDS` and `RYCRAFT_SSAO` remain compatibility aliases for older scripts.

`RYCRAFT_SSAO` now controls the Hi-Z screen-space indirect-lighting pass through its compatibility alias. Propagated smooth skylight and block light, baked corner accessibility, emissive HDR radiance, and bloom remain active when screen-space indirect lighting is disabled.

`RYCRAFT_TIME` accepts an absolute unsigned 64-bit decimal world tick. The remainder modulo 24,000 selects time of day, while the complete saved age selects the lunar phase. For a PNG capture, `RYCRAFT_CAPTURE_LIGHTNING=x,z,id,ageTicks` injects one deterministic visual strike after regional weather is available. Its age must not exceed `RYCRAFT_TIME`; values from 0 through 11 select the visible flash interval.

## Troubleshooting

- If Meson cannot find `xcrun metal`, install Xcode command line tools with `xcode-select --install`.
- The first setup may download Catch2 and LZ4 Meson wraps into `subprojects`; later setups reuse the cache.
- After a large build-definition change, run `meson setup --wipe build`.
- Generator v4 model assets, Core ML caches, authority pages, and saves live under `~/Library/Application Support/rycraft`. Do not copy them into Git or a Conductor workspace.
- A normal restart reuses the installed model pack. If verification fails, Retry does not download over it. Repair verifies every asset, fetches only missing or invalid files, and preserves the extracted runtime and Core ML caches.
- A new v4 world does not accept an arbitrary ocean coordinate as its final spawn. It first selects a bounded inland coarse-model candidate and then validates a dry final exact location. If that bounded search cannot find one, startup fails closed instead of placing the player in water.
- A size, SHA-256, provider, qualification, seed, or generation-fingerprint mismatch fails closed. Use the title-screen repair or retry action. The game does not silently substitute v3 terrain.
- Production qualification requires canonical digest `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df` and the matching scalar Coarse, static Base-four, and static Decoder-four provider configuration. A passing authority-page hash does not by itself qualify the complete first-entry and performance route.
- Ordinary startup never migrates a profile. The Worlds screen can explicitly create a current-v4 successor from a legacy or incompatible profile, while the source directory, regions, manifests, edits, and fluid frontiers remain untouched.

## Author

Ryan Lefkowitz ([rlefkowitz1800@yahoo.com](mailto:rlefkowitz1800@yahoo.com))

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
