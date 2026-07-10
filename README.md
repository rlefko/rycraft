# rycraft

A Minecraft-like voxel game for macOS.

## Features

- Infinite procedural world generation with biomes, caves, ore deposits, trees, and structures
- Day/night cycle with dynamic lighting
- Block placement and destruction
- AABB collision physics
- Animal AI with behaviors: idle, wander, flee, eat, breed, follow
- Metal rendering with bloom post-processing
- Fully procedural textures (no external assets)
- Procedural audio generation

## Requirements

- macOS 13 (Ventura) or later
- Apple Silicon (M1 or later)
- Xcode command line tools

## Build Instructions

```bash
# Install Meson
brew install meson

# Configure and build
meson setup build
ninja -C build
```

## Author

Ryan Lefkowitz

## License

GNU General Public License v3.0 (GPLv3)
