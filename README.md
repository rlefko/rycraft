# rycraft

A full Minecraft-like voxel game for macOS, built from the ground up with Metal, C++23, and Meson.

![Platform](https://img.shields.io/badge/platform-macOS%2013%2B-blue)
![Apple Silicon](https://img.shields.io/badge/CPU-Apple%20Silicon-orange)
![License](https://img.shields.io/badge/license-GPLv3-green)
![C++](https://img.shields.io/badge/C%2B%2B-23-lightgrey)

## Features

- **Infinite procedural world** - Simplex noise terrain with range crossfading, 10 biomes, caves (cheese/spaghetti/noodle), ore deposits, trees, and grid-placed structures
- **Day/night cycle** - Dynamic sun position, sky gradient with twilight transitions, star field at night, directional lighting with color temperature shifts
- **Block interaction** - Raycast-based block breaking and placing with hotbar (9 slots), block highlight wireframe, world persistence with LZ4 compression
- **AABB physics** - Per-axis sweep collision (Y-first), gravity, drag, terminal velocity, jump, sprint, fall damage, step assist, water buoyancy
- **Animal AI** - State machine with 6 states (idle/wander/flee/eat/breed/follow), flocking behavior (separation/alignment/cohesion), edge detection, spatial hash partitioning
- **Weather** - Rain and snow particle system (4096 particles), biome-aware spawning, CPU-simulated physics, GPU billboard rendering
- **Post-processing** - Bloom (extract + Kawase blur + ACES tone mapping), exponential distance fog, procedural clouds with wind animation
- **Audio** - Core Audio RemoteIO engine, 16-voice mixer, procedural sound effects (block break/place, footsteps, ambient wind, animal sounds)
- **Performance** - Binary greedy meshing (50-200μs/chunk), 16-byte packed vertices, mega-buffer allocator (128MB vertex + 64MB index), chunk LOD (3 levels), frustum culling, MSAA 4x

## Requirements

- macOS 13 (Ventura) or later
- Apple Silicon (M1 or later)
- Xcode command line tools
- Meson build system

## Build Instructions

```bash
# Install dependencies
brew install meson

# Configure and build
meson setup build
ninja -C build

# Run tests
ninja -C build test
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | C++23 |
| Build System | Meson + Ninja |
| Rendering | Metal API (MSL 3.0) |
| Windowing | Direct Cocoa + MTKView |
| Input | Direct Cocoa (NSEvent) + GameKit |
| Audio | Core Audio RemoteIO |
| Math | Custom Vec2/Vec3/Vec4/Mat4/AABB |
| Testing | Catch2 3.14.0 |
| Compression | LZ4 |

## Performance Targets

| Metric | Target |
|--------|--------|
| Frame rate | 60 FPS at 1024×768 |
| View distance | 32 chunks default, 64 chunks max |
| Memory | <4GB RAM |
| Chunk generation | <200ms/chunk (async) |
| Mesh building | <200μs/chunk |
| Physics tick | 20Hz fixed (50ms budget) |

## Project Structure

```
rycraft/
├── include/
│   ├── engine/      # Game loop, input, camera, hotbar
│   ├── render/      # Metal pipeline, meshing, bloom, particles
│   ├── world/       # Terrain, biomes, chunks, save/load
│   ├── entity/      # Player, physics, AI, spawning
│   ├── audio/       # Core Audio engine, procedural SFX
│   └── common/      # Math, error handling, thread pool
├── src/             # Implementation files
├── shaders/         # Metal shader sources (.metal)
├── tests/           # Catch2 test suite (260+ tests)
└── docs/            # Project documentation
```

## Author

Ryan Lefkowitz ([rlefkowitz1800@yahoo.com](mailto:rlefkowitz1800@yahoo.com))

## License

GNU General Public License v3.0 (GPLv3)
