# rycraft

## Project Overview

rycraft is a full Minecraft-like voxel game for macOS, built from the ground up using Metal, C++23, and Meson. It targets Apple Silicon exclusively and leverages Direct Cocoa, GameKit, and Core Audio for native performance.

## Tech Stack

- **Platform:** macOS only (Apple Silicon required)
- **Rendering:** Metal API with custom shaders
- **Build System:** Meson + Ninja
- **Language:** C++23
- **Input:** Direct Cocoa + GameKit
- **Audio:** Direct Core Audio
- **Textures:** Fully procedural (no external assets)

## Git Workflow

- **PR-based development:** All changes go through pull requests
- **Emoji-prefixed commits:** Use semantic commit prefixes (e.g., `🏗️`, `🐛`, `✨`)
- **Protected main branch:** No direct pushes to `main`

## Development Commands

```bash
# Setup build directory
meson setup build

# Build the project
ninja -C build

# Run tests
ninja -C build test
```

## CI Checklist

| Stage | Tool |
|-------|------|
| Lint | clang-tidy |
| Build | Meson + Ninja |
| Tests | Catch2 |
| Security | CodeQL |

## Performance Targets

- **Frame rate:** 60 FPS sustained
- **View distance:** 32 chunks default, 64 chunks maximum
- **Memory:** Under 4GB RAM

## Error Handling Policy

- **Metal device/queue failures:** Fatal — log and terminate
- **Chunk generation:** `try/catch` with flat terrain fallback
- **File I/O:** Returns `std::optional` — no exceptions for I/O
