#include "world/features.hpp"

#include "world/chunk_generator.hpp"
#include "world/gen_seeds.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr int TREE_ATTEMPTS_PER_CHUNK = 12;

enum class TreeKind : uint8_t { OAK, LARGE_OAK, BIRCH, SPRUCE };

struct TreeProfile {
    float density = 0.f; // per-attempt acceptance probability
};

TreeProfile treeProfile(Biome biome) {
    switch (biome) {
        case Biome::FOREST:
        case Biome::BIRCH_FOREST:
            return {0.55f};
        case Biome::TAIGA:
            return {0.45f};
        case Biome::SWAMP:
            return {0.20f};
        case Biome::PLAINS:
            return {0.06f};
        case Biome::FLOWER_FIELD:
            return {0.04f};
        default:
            return {0.f};
    }
}

TreeKind pickKind(Biome biome, float roll) {
    switch (biome) {
        case Biome::FOREST:
            return roll < 0.05f   ? TreeKind::LARGE_OAK
                   : roll < 0.85f ? TreeKind::OAK
                                  : TreeKind::BIRCH;
        case Biome::BIRCH_FOREST:
            return roll < 0.85f ? TreeKind::BIRCH : TreeKind::OAK;
        case Biome::TAIGA:
            return TreeKind::SPRUCE;
        case Biome::PLAINS:
            return roll < 0.10f ? TreeKind::LARGE_OAK : TreeKind::OAK;
        default:
            return TreeKind::OAK;
    }
}

// Writes one tree's blocks into the chunk, dropping everything outside it.
// Logs claim their cell outright (over leaves/flora); leaves fill only air,
// which makes overlapping canopies commute — emission order can't matter.
struct TreeWriter {
    Chunk& chunk;
    int baseX;
    int baseZ;

    void log(int x, int y, int z, BlockType logBlock) const {
        int lx = x - baseX;
        int lz = z - baseZ;
        if (lx < 0 || lx >= CHUNK_WIDTH || lz < 0 || lz >= CHUNK_DEPTH || y < 0 ||
            y >= CHUNK_HEIGHT)
            return;
        BlockType current = chunk.getBlock(lx, y, lz);
        if (current == BlockType::AIR || isLeafBlock(current) || isFlora(current)) {
            chunk.setBlock(lx, y, lz, logBlock);
        }
    }

    void leaves(int x, int y, int z, BlockType leafBlock) const {
        int lx = x - baseX;
        int lz = z - baseZ;
        if (lx < 0 || lx >= CHUNK_WIDTH || lz < 0 || lz >= CHUNK_DEPTH || y < 0 ||
            y >= CHUNK_HEIGHT)
            return;
        if (chunk.getBlock(lx, y, lz) == BlockType::AIR) {
            chunk.setBlock(lx, y, lz, leafBlock);
        }
    }
};

// The whole tree is rebuilt identically by every chunk it touches; rng is
// the tree's private stream, so shapes can draw freely.
void buildTree(TreeKind kind, SeededRng rng, int x, int baseY, int z, const TreeWriter& out) {
    switch (kind) {
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK: {
            bool large = kind == TreeKind::LARGE_OAK;
            int height = large ? rng.nextInt(7, 9) : rng.nextInt(4, 6);
            int radius = large ? 3 : 2;
            int top = baseY + height;
            // Canopy: rounded blob around the crown
            for (int dy = -2; dy <= 1; ++dy) {
                int r = dy >= 0 ? radius - 1 : radius;
                for (int dz = -r; dz <= r; ++dz) {
                    for (int dx = -r; dx <= r; ++dx) {
                        bool corner = std::abs(dx) == r && std::abs(dz) == r;
                        if (corner && (r > 1 || rng.nextFloat() < 0.5f)) continue;
                        out.leaves(x + dx, top + dy, z + dz, BlockType::LEAVES);
                    }
                }
            }
            out.leaves(x, top + 2, z, BlockType::LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::LOG);
            break;
        }

        case TreeKind::BIRCH: {
            int height = rng.nextInt(5, 7);
            int top = baseY + height;
            for (int dy = -2; dy <= 0; ++dy) {
                int r = dy == 0 ? 1 : 2;
                for (int dz = -r; dz <= r; ++dz) {
                    for (int dx = -r; dx <= r; ++dx) {
                        if (std::abs(dx) == r && std::abs(dz) == r && r == 2) continue;
                        out.leaves(x + dx, top + dy, z + dz, BlockType::BIRCH_LEAVES);
                    }
                }
            }
            out.leaves(x, top + 1, z, BlockType::BIRCH_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::BIRCH_LOG);
            break;
        }

        case TreeKind::SPRUCE: {
            int height = rng.nextInt(6, 9);
            int top = baseY + height;
            // Conical rings widening toward the bottom of the crown
            for (int dy = 0; dy <= height - 2; ++dy) {
                int r = std::min(3, 1 + dy / 2);
                if (dy % 2 == 1) r = std::max(1, r - 1); // stepped silhouette
                int y = top - dy;
                for (int dz = -r; dz <= r; ++dz) {
                    for (int dx = -r; dx <= r; ++dx) {
                        if (std::abs(dx) == r && std::abs(dz) == r && r > 1) continue;
                        out.leaves(x + dx, y, z + dz, BlockType::SPRUCE_LEAVES);
                    }
                }
            }
            out.leaves(x, top + 1, z, BlockType::SPRUCE_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::SPRUCE_LOG);
            break;
        }
    }
}

// Per-biome flora odds: r1 buckets into tall grass → flowers → mushrooms.
struct FloraOdds {
    float tallGrass = 0.f;
    float flowers = 0.f;
    float mushrooms = 0.f;
};

FloraOdds floraOdds(Biome biome) {
    switch (biome) {
        case Biome::PLAINS:
            return {0.18f, 0.03f, 0.f};
        case Biome::FLOWER_FIELD:
            return {0.20f, 0.25f, 0.f};
        case Biome::FOREST:
            return {0.10f, 0.02f, 0.01f};
        case Biome::BIRCH_FOREST:
            return {0.10f, 0.03f, 0.01f};
        case Biome::SWAMP:
            return {0.12f, 0.01f, 0.04f};
        case Biome::TAIGA:
            return {0.05f, 0.f, 0.03f};
        case Biome::MUSHROOM_ISLAND:
            return {0.02f, 0.f, 0.20f};
        default:
            return {};
    }
}

} // namespace

FeaturePlacer::FeaturePlacer(uint32_t worldSeed) : seed_(worldSeed) {}

void FeaturePlacer::placeTrees(Chunk& chunk, const ChunkGenerator& gen,
                               const StructurePlacer& structures, GenScratch& scratch) const {
    TreeWriter out{chunk, chunk.chunkX * CHUNK_WIDTH, chunk.chunkZ * CHUNK_DEPTH};
    uint32_t treeSeed = genseed::subSeed(seed_, genseed::TREES);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            int sourceChunkX = chunk.chunkX + dx;
            int sourceChunkZ = chunk.chunkZ + dz;
            SeededRng rng(hashCoords(sourceChunkX, sourceChunkZ, treeSeed));

            for (int attempt = 0; attempt < TREE_ATTEMPTS_PER_CHUNK; ++attempt) {
                // Fixed draw order; decisions only after all draws
                int x = sourceChunkX * CHUNK_WIDTH + rng.nextInt(0, CHUNK_WIDTH - 1);
                int z = sourceChunkZ * CHUNK_DEPTH + rng.nextInt(0, CHUNK_DEPTH - 1);
                float acceptRoll = rng.nextFloat();
                float kindRoll = rng.nextFloat();
                uint64_t shapeSeed = rng.next();

                Biome biome = gen.biomeAt(x, z, scratch);
                TreeProfile profile = treeProfile(biome);
                if (acceptRoll >= profile.density) continue;

                int surfaceY = gen.surfaceYAt(x, z, scratch);
                if (surfaceY < SEA_LEVEL || surfaceY >= SNOW_LINE) continue;
                if (structures.insideStructure(x, z, chunk.chunkX, chunk.chunkZ, gen, scratch, 1))
                    continue;

                buildTree(pickKind(biome, kindRoll), SeededRng(shapeSeed), x, surfaceY + 1, z, out);
            }
        }
    }
}

void FeaturePlacer::placeFlora(Chunk& chunk) const {
    uint32_t floraSeed = genseed::subSeed(seed_, genseed::FLORA);
    int baseX = chunk.chunkX * CHUNK_WIDTH;
    int baseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int lz = 0; lz < CHUNK_DEPTH; ++lz) {
        for (int lx = 0; lx < CHUNK_WIDTH; ++lx) {
            int idx = lx + lz * CHUNK_WIDTH;
            int surfaceY = chunk.heightMap[idx];
            if (surfaceY <= 2 || surfaceY >= CHUNK_HEIGHT - 4) continue;

            SeededRng rng(hashCoords(baseX + lx, baseZ + lz, floraSeed));
            float roll = rng.nextFloat();
            float kindRoll = rng.nextFloat();
            int heightRoll = rng.nextInt(0, 2);

            BlockType ground = chunk.getBlock(lx, surfaceY, lz);
            if (chunk.getBlock(lx, surfaceY + 1, lz) != BlockType::AIR) continue;

            Biome biome = chunk.biomes[idx];

            // Reeds hug the waterline: grassy/sandy ground at sea level with
            // water directly beside it (in-chunk neighbors only — border
            // columns simply skip, deterministically).
            bool nearWater = false;
            if (surfaceY >= 62 && surfaceY <= 66 &&
                (ground == BlockType::GRASS || ground == BlockType::SAND ||
                 ground == BlockType::DIRT)) {
                for (auto [nx, nz] :
                     {std::pair{lx - 1, lz}, {lx + 1, lz}, {lx, lz - 1}, {lx, lz + 1}}) {
                    if (nx >= 0 && nx < CHUNK_WIDTH && nz >= 0 && nz < CHUNK_DEPTH &&
                        chunk.getBlock(nx, surfaceY, nz) == BlockType::WATER) {
                        nearWater = true;
                        break;
                    }
                }
            }
            if (nearWater && roll < 0.30f) {
                int reedHeight = 2 + heightRoll % 2;
                for (int dy = 1; dy <= reedHeight; ++dy) {
                    if (chunk.getBlock(lx, surfaceY + dy, lz) != BlockType::AIR) break;
                    chunk.setBlock(lx, surfaceY + dy, lz, BlockType::REED);
                }
                continue;
            }

            if (ground == BlockType::GRASS) {
                FloraOdds odds = floraOdds(biome);
                if (roll < odds.tallGrass) {
                    chunk.setBlock(lx, surfaceY + 1, lz, BlockType::TALL_GRASS);
                } else if (roll < odds.tallGrass + odds.flowers) {
                    chunk.setBlock(lx, surfaceY + 1, lz,
                                   kindRoll < 0.5f ? BlockType::FLOWER_YELLOW
                                                   : BlockType::FLOWER_RED);
                } else if (roll < odds.tallGrass + odds.flowers + odds.mushrooms) {
                    chunk.setBlock(lx, surfaceY + 1, lz,
                                   kindRoll < 0.5f ? BlockType::MUSHROOM_BROWN
                                                   : BlockType::MUSHROOM_RED);
                }
            } else if (ground == BlockType::SAND && biome == Biome::DESERT) {
                if (roll < 0.015f) {
                    int cactusHeight = 1 + heightRoll;
                    for (int dy = 1; dy <= cactusHeight; ++dy) {
                        if (chunk.getBlock(lx, surfaceY + dy, lz) != BlockType::AIR) break;
                        chunk.setBlock(lx, surfaceY + dy, lz, BlockType::CACTUS);
                    }
                } else if (roll < 0.035f) {
                    chunk.setBlock(lx, surfaceY + 1, lz, BlockType::DEAD_BUSH);
                }
            }
        }
    }
}
