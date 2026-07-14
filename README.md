# rycraft

A Minecraft-like voxel game for macOS, built from scratch in C++23 on Metal. No engine, no asset files — the terrain, the block textures, and even the sound effects are all generated procedurally at runtime.

![Platform](https://img.shields.io/badge/platform-macOS%2013%2B-blue)
![Apple Silicon](https://img.shields.io/badge/CPU-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-GPLv3-green)
![C++](https://img.shields.io/badge/C%2B%2B-23-lightgrey)

## Quick start

```bash
brew install meson ninja        # requires Xcode command line tools
meson setup build
ninja -C build
./build/src/rycraft
```

Run the tests with `ninja -C build test`.

**Controls:** WASD to move, mouse to look, Space to jump (hold it to keep hopping), double-tap W or hold Ctrl to sprint, left/right click to break/place blocks, 1–9 or scroll for the hotbar, F3 for the debug HUD, ESC to pause (and to resume). In water, hold Space to float up or double-tap W to swim wherever you're looking. Double-tap Space to toggle flying — Space rises, Shift sinks, and landing with Shift (or another double-tap) ends the flight. The window opens on a title screen; click PLAY to capture the mouse.

## What's in the game

- **Infinite procedural world** — simplex-noise terrain across 10 biomes, three kinds of caves, ore veins, trees, and grid-placed structures, streaming around the player on a worker pool
- **A day** — twenty-minute day/night cycle with a moving sun, dawn/dusk skies, drifting procedural clouds, and column-skylight shadows under trees and inside caves
- **Building** — raycast block breaking and placing with a highlight wireframe; edits persist to LZ4-compressed region files and load back next session
- **Animals** — sheep, cows, pigs, and chickens with state-machine AI, flocking, and ambient calls, rendered from procedural voxel models
- **Sound** — procedural block, footstep, wind, and animal sounds through a 16-voice Core Audio mixer
- **A real game shell** — title screen, ESC pause menu, settings (render distance, fog, sensitivity, volume), pointer lock that never loses your cursor, and save-on-quit
- **A lean renderer** — one 4x MSAA scene pass at native resolution plus bloom and fog, batched UI, ~60 FPS

Full design notes live in [docs/game-concept.md](docs/game-concept.md).

## Project structure

```
include/, src/
  engine/      Game loop, game flow (title/pause), input, camera
  render/      Metal pipeline, mesher, textures, UI, entities
  world/       Terrain generation, chunks, saving/loading
  entity/      Player physics, animal AI, spawning
  audio/       Core Audio mixer, procedural sound effects
  common/      Math, seeded randomness, thread pool
shaders/       Metal shaders (compiled to one metallib)
tests/         Catch2 suite — six hermetic modules, all headless
docs/          Architecture, conventions, and domain references
```

The deeper references: [architecture](docs/architecture.md) · [world generation & saves](docs/world-generation.md) · [rendering conventions](docs/rendering-conventions.md) · [performance conventions](docs/performance-conventions.md) · [code conventions](docs/code-conventions.md)

## Tech stack

| Component | Technology |
|-----------|------------|
| Language | C++23 (+ Objective-C++ at the Cocoa/Metal boundary) |
| Rendering | Metal 3, MSL shaders shared with C++ via one types header |
| Windowing & input | Cocoa, MTKView, NSEvent with CG pointer lock |
| Audio | Core Audio (DefaultOutput unit) |
| Build | Meson + Ninja |
| Tests | Catch2 (via Meson wrap) |
| Compression | LZ4 (system copy or wrap) |

## Development

```bash
meson setup build                  # once (wraps download on first setup)
ninja -C build                     # build (werror, warning_level=3)
ninja -C build test                # run the test suite
clang-format -i <files>            # format touched files (config in .clang-format)
```

Playtest hooks (used by CI-less visual verification and the `playtest` skill):

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/src/rycraft   # Metal validation
RYCRAFT_CAPTURE=/tmp/frame.png ./build/src/rycraft              # dump a frame as PNG
RYCRAFT_CAPTURE_FRAME=300 RYCRAFT_START_SCREEN=paused ...       # pick frame/screen
RYCRAFT_BLOOM=0 ...                                             # scale/disable bloom
```

## Troubleshooting

- **`meson setup` fails finding `xcrun metal`** — install the Xcode command line tools (`xcode-select --install`); the shader pipeline compiles `.metal` sources with the Metal CLI.
- **First `meson setup` downloads things** — Catch2 (and LZ4, when no system copy exists) come from Meson wraps into `subprojects/`; that's expected and cached.
- **Stale build directory after big changes** — `meson setup --wipe build`.
- **The game saves next to where you run it** — world data lands in `./rycraft_world/`; delete it for a fresh world.

## Author

Ryan Lefkowitz ([rlefkowitz1800@yahoo.com](mailto:rlefkowitz1800@yahoo.com))

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
