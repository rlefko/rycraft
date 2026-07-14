# World Generation & Persistence

The domain reference: what a chunk is, how the world generates, and how it saves. Numbers here are the source of truth for tests and tools.

## Chunks

- Dimensions: **16 × 256 × 16** (`CHUNK_WIDTH × CHUNK_HEIGHT × CHUNK_DEPTH`), 65,536 blocks in a flat `std::vector<BlockType>`.
- Addressed by **`ChunkPos {x, z}`** (`include/world/chunk_pos.hpp`) — the one chunk key type, used by world storage and the renderer's mesh cache alike. `packed()` gives a stable `uint64_t`.
- Per-chunk metadata: a 16×16 `Biome` map and a 16×16 `heightMap` — the **raw terrain surface** (topmost solid block the density fill produced, before decoration). It equals `ChunkGenerator::surfaceYAt` for every column; trees, ice caps, and structure walls may rise above it, and structures may carve or pave the exact cell.
- Flags: `generated`, `needsMeshUpdate`, `meshed`, `modifiedSinceSave`; plus `version`, an atomic revision bumped by every block edit (self + boundary neighbors) that drives the async mesher's staleness protocol.

## Block types

`BlockType` (`include/world/block_properties.hpp`) — 34 values. The original 17 (AIR … GLASS) plus the worldgen-overhaul set: COBBLESTONE, MOSSY_COBBLESTONE, SANDSTONE, BIRCH_LOG, BIRCH_LEAVES, SPRUCE_LOG, SPRUCE_LEAVES, CACTUS, DEAD_BUSH, TALL_GRASS, FLOWER_YELLOW, FLOWER_RED, MUSHROOM_BROWN, MUSHROOM_RED, REED, LAVA, ICE. Values are persisted as raw bytes in saves: **only append, never renumber.**

Properties come from the **single table** in that header:

| Predicate | Meaning | Notes |
|-----------|---------|-------|
| `isFlora` | cross-quad plants (walk-through decoration) | DEAD_BUSH, TALL_GRASS, both flowers, both mushrooms, REED |
| `isLiquid` | swimmable | WATER, LAVA |
| `isSolid` | collision + casts skylight shadow | false for AIR, liquids, flora |
| `isOpaque` | fully hides neighbors' faces | false for AIR, WATER, leaf variants, GLASS, flora; **LAVA is opaque** |
| `rendersAsCube` | emitted by the opaque chunk mesh pass | `isSolid` plus LAVA (swim-through emissive cube); WATER draws in the water pass |
| `isTargetable` | crosshair raycast stops here | `isSolid` plus flora (breakable in place); liquids are click-through |

Leaf variants and glass are **alpha-cutout** blocks: solid to walk on, hole-punched textures, and they don't occlude what's behind them. **Why one table:** four diverging `isSolid` copies once made glass render solid while entities fell through it.

## The generation pipeline

`ChunkGenerator::generate` (`src/world/chunk_generator.cpp`) fills each chunk completely and **independently** — there is no cross-chunk ordering, which is what makes the world infinite and the streaming order irrelevant.

### Design pillars

1. **One density function.** A voxel is solid iff `D(x,y,z) > 0`. Terrain shape AND every cave type combine into `D` (caves contribute negative density through `min()`), so a carve pass can never orphan a floating surface layer or hollow out the world — the two bugs that motivated this design (the old carver's inverted threshold deleted ~99% of underground stone; the old surface fill left a one-block air gap under every surface block).
2. **World-aligned lattice.** `D` is evaluated on a 4×4×4 world-aligned lattice (`LATTICE_XZ`/`LATTICE_Y`, `density_field.hpp`) and trilinearly interpolated to voxels — ~13× fewer noise evaluations than per-voxel sampling, and because the lattice is world-aligned, interpolated density is a pure function of world position: neighboring chunks agree on shared columns bit-for-bit. **There is exactly one interpolation code path** (`lerpDensity`/`bilerpDensity` + `voxelDensity`) shared by the bulk fill and the single-column queries; a second one with reordered float ops would produce different bits and visible seams.
3. **Purity contract.** `baseHeightAt(x,z)`, `biomeAt(x,z)`, and `surfaceYAt(x,z)` (post-cave surface) are pure functions of world coordinates + seed. Cross-chunk features may consume ONLY these plus their own feature RNG — never another chunk's block state.
4. **Parameters, not biome IDs.** Terrain height reads continuous climate splines; the biome ID derives from the same fields and drives only materials/flora/trees. **Why:** the old design added discrete per-biome height offsets (+15 hills) after the biome pick, producing walls at biome borders.

### Climate & terrain shaping (`climate.cpp`)

Five 2D fBm fields sampled at lattice columns and bilinearly interpolated per block column:

| Field | Wavelength | Octaves | Drives |
|-------|-----------|---------|--------|
| continentalness | 1200 | 4 | base height spline (ocean 38 → coast 64 → interior 96), land factor, mushroom-island boost |
| erosion | 900 | 4 | mountain amplitude spline (70 → 2) |
| ridges | 500 | 3 | peaks-and-valleys fold `1 − \|3\|R\| − 2\|` (remapped to [0,1] so valleys sit at the continental base) and river channels along `\|R\| < 0.06` |
| temperature | 1400 | 3 | biome choice, ice caps, snow tops |
| humidity | 1000 | 3 | biome choice |

Rivers carve to y≈59 with smoothstep banks; a 3D detail field (2 octaves, anisotropic 70/48/70) adds crags and overhangs, with amplitude scaled by mountain uplift and damped to zero at/below sea level so coasts and banks stay clean. Height caps at 240.

### Biomes

14 values (`Biome` in `chunk.hpp` — append-only for saves): the original ten plus BEACH, RIVER, BIRCH_FOREST, FLOWER_FIELD. `ClimateSampler::selectBiome` applies ordered rules on (temperature, humidity, continentalness, ridges, height); rivers classify before the ocean bands because their channels sit below 62 too.

### Caves (inside the density function, `density_field.cpp`)

| Type | Formulation | Character |
|------|-------------|-----------|
| Cheese | fBm 2-oct wl 90/60, carve where `n > 0.42 − 0.16·depthFrac` | caverns, growing with depth |
| Spaghetti | two noises wl 68/44, carve where `max(\|n1\|,\|n2\|) < (0.055 + 0.025·depthFrac)` | long winding tunnels |
| Noodle | two noises wl 34/22, width 0.04, y < 64 only | tight crawl spaces |
| Ravines | 2D ridged wl 280, edge `smoothstep(0.82, 0.96)`, floor down to `H − 46·edge` (min 12) | rare canyons; the `(1−edge)` offset tapers lips into gullies instead of one-block trenches |

**Near-surface sealing:** cave strength ramps 0→1 over the 12 blocks below the surface — except where the entrance mask (2D, wl 140, > 0.4) keeps full strength on land ≥ 70, producing natural cave mouths. Water-covered columns (oceans, rivers, flooded ravines) always keep ≥ 8 blocks of cover: no drained seas. Sealed cave air at y ≤ 10 becomes LAVA. Bedrock: y 0–1 solid, y 2 dithered by `hashCoords`.

### Surface & liquids (single top-down pass per column)

Open-to-sky air below 64 floods with WATER (ICE at y 63 when frozen: temperature < −0.45). Each sky-visible air→solid transition gets its biome surface set — grass/dirt, sand → sandstone at depth (desert/beach), snow tops above y 108 or in frozen biomes, stone in extreme hills, hash-mixed sand/gravel beds under water — with 2–4 blocks of subsoil (per-column hash). Under-overhang and cave transitions stay stone.

### Decoration (all cross-chunk deterministic)

The **neighborhood pattern**: when generating chunk C, re-roll the feature attempts of every chunk in a fixed neighborhood from each source chunk's own `hashCoords` seed, and write only the blocks that land inside C. Both sides of every border agree without reading each other. The **RNG-order rule**: every attempt draws its randomness in a fixed order *before* any accept/reject decision, so skipped attempts consume identical draws from every chunk's perspective.

1. **Ores** (`ores.cpp`, radius 1): random-walk blobs capped at 12 steps (what makes radius 1 sufficient), replacing STONE only. Bands: coal 48–131, iron 8–71, gold 4–35, diamond 2–17.
2. **Structures** (`structures.cpp`): one attempt per 8×8-chunk region — RUIN 40% / WELL 30% / HOUSE 30%, rotated, anchored inside the region with a footprint cap that keeps spill within one chunk. Validated purely: 5 `surfaceYAt` probes (spread ≤ 2, ruins ≤ 5), dry land, land biome; foundations fill per-column down to the real terrain, interiors force-carve. Placements are cached in `GenScratch` and queried by tree placement for rejection.
3. **Trees** (`features.cpp`, radius 1): 12 attempts per chunk; acceptance by biome (forest/birch 0.55, taiga 0.45, swamp 0.20, plains 0.06, flower field 0.04). Base `y = surfaceYAt` — post-cave, so a tree at a cave mouth sits on the real lip, never floating. Kinds: OAK, LARGE_OAK, BIRCH, SPRUCE (conical), each rebuilt identically from a private RNG stream by every chunk it touches; canopies span borders. Logs claim their cell (over leaves/flora), leaves fill only air — overlapping canopies commute.
4. **Flora** (chunk-local): tall grass, flowers (flower fields at 25%), mushrooms (swamp/taiga/mushroom island), dead bushes + 1–3-tall cacti (desert sand), 2–3-tall reeds beside water. Placed on the chunk's real final surface, above-cell must be air.

## Determinism

Everything derives from the world seed via `common/random.hpp` (`hash64`, `hashCoords`, `SeededRng`), routed through the **seed-offset table** in `include/world/gen_seeds.hpp` (`subSeed(worldSeed, key)`): 0x1xx climate fields, 0x2xx terrain detail, 0x3xx caves, 0x4xx bedrock/surface, 0x5xx ores, 0x6xx trees/flora, 0x7xx structures. Weather is seeded; animal spawning is seeded; AI jitter is a pure per-entity hash. A new randomness source must take its seed from this chain — see the determinism pillar in [game-concept.md](game-concept.md).

Tests pin the contract: same seed → bit-identical chunks; generation-order independence; `surfaceYAt` ↔ `heightMap` agreement; carve fraction bounds; column invariants (no terrain above `heightMap`, no water resting on air, lava only at the bottom); ore bands; grounded flora.

## Streaming

`updatePlayerPosition` (every tick) rebuilds the **generation backlog** on chunk-boundary crossings — and on the very first call, so the spawn area streams even when the player starts in chunk (0,0). The backlog covers a **generation radius of view distance + 1** (visible chunks always have generated neighbors for the neighbor-aware mesher), sorted nearest-first, and drains through a bounded submission window (`MAX_INFLIGHT_GEN = 32`) that finishing workers refill themselves. **Why the window:** submitting everything at once meant a boundary cross had to fight hundreds of already-queued stale-priority tasks; rebuilding a sorted backlog re-prioritizes (and implicitly cancels) everything not yet submitted. `unloadDistantChunks` drops chunks beyond **view distance + 2** (hysteresis so strafing along a boundary doesn't churn the frontier), queueing edited ones for saving; changing the view distance re-streams immediately.

## Save format — RYCH v3

**One file per chunk**, sharded into 32×32-chunk region directories: `rycraft_world/regions/r.X.Z/c.<cx>.<cz>.dat`, LZ4-compressed, written atomically (temp file + rename). **Why per-chunk files:** the old packed region format wrote ONE chunk per region file — `writeFile` clobbered the whole file on every save, silently losing every other edited chunk in the region. Chunk payload:

```
Header (20 bytes):
  uint32  magic       0x52594348 "RYCH"
  uint32  version     3
  int32   chunkX, chunkZ
  uint32  blockCount  (= 65,536)
Body:
  blockCount × uint8   block types
  256        × uint8   biomes
  256        × int16   height map (little-endian)
```

v3 keeps the v2 layout but marks the worldgen-overhaul epoch: v2 worlds were generated by the broken carver, so they regenerate (pre-release migration policy: regenerate, don't convert). Pre-v3 chunks deserialize to `nullopt`. Bump the version for any layout change.

**Save-on-unload:** only chunks with `modifiedSinceSave` persist — everything else is a pure function of the seed. Unloading queues the chunk (serialization + compression happen on the save thread); the quit path sweeps still-loaded modified chunks and flushes. A queued chunk stays readable through the SaveManager's **pending map**, so walking straight back to a just-unloaded chunk sees the queued edits, not the stale file.

Metadata (`metadata.json`): seed, player spawn position, world time — written on quit, read at launch to resume the same world.

**Load-before-generate:** `loadOrGenerateChunk` consults the `SaveManager` first, so player edits survive sessions. **Why:** the world once only ever *wrote* saves; every launch regenerated pristine terrain over your build.
