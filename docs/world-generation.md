# World Generation & Persistence

The domain reference: what a chunk is, how the world generates, and how it saves. Numbers here are the source of truth for tests and tools.

## Chunks

- Dimensions: **16 × 256 × 16** (`CHUNK_WIDTH × CHUNK_HEIGHT × CHUNK_DEPTH`), 65,536 blocks in a flat `std::vector<BlockType>`.
- Addressed by **`ChunkPos {x, z}`** (`include/world/chunk_pos.hpp`) — the one chunk key type, used by world storage and the renderer's mesh cache alike. `packed()` gives a stable `uint64_t`.
- Per-chunk metadata: a 16×16 `Biome` map and a 16×16 `heightMap` (highest solid block per column, filled by surface generation, consumed by tree/structure placement).
- Flags: `generated`, `needsMeshUpdate` (`markDirty()` on edits), `meshed`.

## Block types

`BlockType` (`include/world/block_properties.hpp`): AIR, STONE, GRASS, DIRT, SAND, GRAVEL, WATER, BEDROCK, LOG, LEAVES, SNOW, COAL_ORE, IRON_ORE, GOLD_ORE, DIAMOND_ORE, PLANKS, GLASS — 17 values.

Properties come from the **single table** in that header:

| Predicate | Meaning | False for |
|-----------|---------|-----------|
| `isSolid` | collision + casts skylight shadow | AIR, WATER |
| `isOpaque` | fully hides neighbors' faces | AIR, WATER, LEAVES, GLASS |
| `isTransparent` | see-through(ish) | everything except those four |

Leaves and glass are **alpha-cutout** blocks: solid to walk on, hole-punched textures, and they don't occlude what's behind them. **Why one table:** four diverging `isSolid` copies once made glass render solid while entities fell through it.

## The generation pipeline

`World::generateChunk` runs, in order:

1. **Terrain + biome** — per column: simplex fBm height (7 octaves, base frequency 0.005, heights ~20–128, sea level 64) and a temperature/moisture/elevation biome lookup. Ten biomes: DeepOcean, Ocean, Plains, Forest, Taiga, Desert, ExtremeHills, Swamp, MushroomIsland, IceSpikes.
2. **Surface** — fills each column (surface block, subsurface, stone core, bedrock floor) and records `heightMap`.
3. **Caves** — three carvers: cheese (threshold 0.05), spaghetti (0.10, two noises), noodle (0.08, ridged); ceiling 128, floor 4.
4. **Ores** — trapezoidal vertical distributions per ore with per-chunk cluster counts.
5. **Trees** — oak and pine, density-noise gated, grass-only placement with canopy clearance.
6. **Structures** — grid-hashed placement (houses).

## Determinism

Everything derives from the world seed via `common/random.hpp` (`hash64`, `hashCoords`, `SeededRng`). Weather is seeded; animal spawning is seeded; AI jitter is a pure per-entity hash. A new randomness source must take its seed from this chain — see the determinism pillar in [game-concept.md](game-concept.md).

## Streaming

`updatePlayerPosition` (every tick) triggers `generateAroundPlayer` on chunk-boundary crossings — and on the very first call, so the spawn area streams even when the player starts in chunk (0,0) (**why:** the position tracker initialized to the spawn chunk and the first-load never fired). Generation runs on the worker pool; `unloadDistantChunks` drops chunks outside the view distance; changing the view distance re-streams immediately.

## Save format — RYCH v2

Region files hold 256 chunks (16×16 regions) under `rycraft_world/regions/`, each chunk LZ4-compressed. Chunk payload:

```
Header (20 bytes):
  uint32  magic       0x52594348 "RYCH"
  uint32  version     2
  int32   chunkX, chunkZ
  uint32  blockCount  (= 65,536)
Body:
  blockCount × uint8   block types
  256        × uint8   biomes
  256        × int16   height map (little-endian)
```

v2 widened heights from int8 to int16. **Why:** terrain reaches height 128, which overflowed int8 to -128 on load and corrupted tree/structure placement. Pre-v2 chunks deserialize to `nullopt` and simply regenerate (pre-release migration policy: regenerate, don't convert). Bump the version for any layout change.

Metadata (`metadata.json`-equivalent binary): seed, player spawn position, world time — written on quit, read at launch to resume the same world.

**Load-before-generate:** `loadOrGenerateChunk` consults the `SaveManager` first, so player edits survive sessions. **Why:** the world once only ever *wrote* saves; every launch regenerated pristine terrain over your build.
