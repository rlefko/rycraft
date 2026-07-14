# rycraft — Game Concept

rycraft is a from-scratch Minecraft-like voxel game for Apple Silicon Macs. No engine, no external assets: direct Metal rendering, Cocoa windowing, Core Audio sound, and procedurally generated everything — terrain, textures, and sound effects alike.

## Vision

A complete, native, self-contained voxel sandbox that demonstrates what a modern Mac can do with a few thousand lines of C++23. Every system is built here and understandable end to end: you can read the whole renderer in an afternoon.

## Pillars

1. **Native-fast.** 60 FPS sustained at native resolution with 4x MSAA. The renderer is one MSAA scene pass plus bloom; the simulation is a fixed 20 Hz tick. Performance budgets live in [performance-conventions.md](performance-conventions.md).
2. **Procedural everything.** No asset files ship with the game. Block textures are painted at startup ([`block_texture_array.mm`](../src/render/block_texture_array.mm)), terrain flows from seeded simplex noise, and sound effects are synthesized PCM ([`sfx.cpp`](../src/audio/sfx.cpp)).
3. **Deterministic worlds.** The same seed always produces the same world — trees, ores, structures, weather, and animal spawns included. All randomness derives from the seed through `common/random.hpp`; `std::random_device` does not appear in gameplay code.
4. **Honest simulation.** What renders is what exists: a block that looks solid collides ([`block_properties.hpp`](../include/world/block_properties.hpp)), edits persist across sessions, and menus genuinely freeze the world.

## The core loop

Explore an infinite terrain of ten biomes; mine blocks; place blocks; build. Day cycles into night over twenty minutes. Animals wander the surface, flee when startled, and call out nearby. The world streams around you and saves behind you.

## Feel

- **Look:** classic blocky voxels with per-face procedural textures, column-skylight shadows under trees and inside caves, alpha-cutout foliage, drifting procedural clouds, and dawn-to-dusk sky colors.
- **Sound:** understated procedural audio — soft footsteps, block thunks, ambient wind, and the occasional animal call. Menus mute the world because a paused world is silent.
- **Input:** pointer-locked mouse look that never lets the cursor escape mid-play, WASD aligned exactly with the camera, ESC always one keypress away from the pause menu. Movement reads like Minecraft: double-tap W (or hold Ctrl) to sprint with a subtle FOV widening, hold Space to keep hopping or to float up in water, double-tap W in water to swim along the look direction, double-tap Space for creative-style flight (Space/Shift rise/sink; landing with Shift ends it).
- **Menus:** bitmap-font panels in the game's own 8×8 pixel face, rendered by the same UI batcher as the HUD. Title → Play; ESC → Paused → Settings.

## Screens

| Screen | Purpose |
|--------|---------|
| Title | PLAY / QUIT over a live view of the world |
| Playing | The game: captured cursor, crosshair, hotbar |
| Paused (ESC) | RESUME / SETTINGS / QUIT; the simulation freezes |
| Settings | Render distance, fog, mouse sensitivity, volume |
| Debug HUD (F3) | Real FPS, frame time, chunk and entity counts |

## Current scope

Shipping: infinite terrain (10 biomes, caves, ores, trees, structures), block breaking/placing with persistence, day/night, weather particles, animals with state-machine AI and flocking, sprint/swim/fly movement with auto-jump, procedural audio, bloom + fog + clouds, the full menu suite.

Deliberately not yet built: water rendering (blocks exist, the transparent pass does not), chunk LOD (the mesher supports it; the renderer draws full detail — see the note in [`lod_mesher.hpp`](../include/render/lod_mesher.hpp)), crafting/inventory, and multiplayer.
