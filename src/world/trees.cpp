#include "world/trees.hpp"

#include "common/random.hpp"

#include <cmath>

uint32_t TreeGenerator::treeRand(uint32_t& state) {
    state = static_cast<uint32_t>(hash64(state));
    return (state >> 16) & 0x7fff;
}

TreeGenerator::TreeGenerator(uint32_t seed)
    : treeDensity_(seed)
{
}

bool TreeGenerator::canPlaceTree(Chunk& chunk, int localX, int localY, int localZ) const {
    // Must be on GRASS block
    if (localY < 0 || localY >= CHUNK_HEIGHT) return false;
    if (chunk.getBlock(localX, localY, localZ) != BlockType::GRASS) return false;

    // Check clearance: need at least 8 blocks of AIR above
    for (int dy = 1; dy <= 8; ++dy) {
        int checkY = localY + dy;
        if (checkY >= CHUNK_HEIGHT) return false;
        BlockType above = chunk.getBlock(localX, checkY, localZ);
        if (above != BlockType::AIR) return false;
    }

    return true;
}

void TreeGenerator::generateOak(Chunk& chunk, int localX, int localY, int localZ) const {
    // Total height: 4-7 blocks
    uint32_t state = static_cast<uint32_t>(localX * 7919u + localY * 104729u + localZ * 130021u);
    int totalHeight = 4 + (treeRand(state) % 4);

    // Golden ratio: trunk = 0.618 * total height
    int trunkHeight = static_cast<int>(totalHeight * 0.618);
    trunkHeight = std::max(2, trunkHeight);

    // Place trunk
    for (int dy = 0; dy < trunkHeight; ++dy) {
        int y = localY + 1 + dy;
        if (y < CHUNK_HEIGHT) {
            chunk.setBlock(localX, y, localZ, BlockType::LOG);
        }
    }

    // Place leaves in sphere around top of trunk
    int leafCenterY = localY + trunkHeight + 1;
    int leafRadius = 2 + (treeRand(state) % 2);  // Radius 2-3

    for (int dx = -leafRadius; dx <= leafRadius; ++dx) {
        for (int dy = -leafRadius; dy <= leafRadius; ++dy) {
            for (int dz = -leafRadius; dz <= leafRadius; ++dz) {
                int distSq = dx * dx + dy * dy + dz * dz;
                if (distSq > leafRadius * leafRadius) continue;

                int tx = localX + dx;
                int ty = leafCenterY + dy;
                int tz = localZ + dz;

                if (tx < 0 || tx >= CHUNK_WIDTH) continue;
                if (ty < 0 || ty >= CHUNK_HEIGHT) continue;
                if (tz < 0 || tz >= CHUNK_DEPTH) continue;

                BlockType existing = chunk.getBlock(tx, ty, tz);
                if (existing == BlockType::AIR) {
                    chunk.setBlock(tx, ty, tz, BlockType::LEAVES);
                }
            }
        }
    }
}

void TreeGenerator::generatePine(Chunk& chunk, int localX, int localY, int localZ) const {
    // Total height: 5-8 blocks
    uint32_t state = static_cast<uint32_t>(localX * 7919u + localY * 104729u + localZ * 130021u);
    int totalHeight = 5 + (treeRand(state) % 4);

    // Trunk takes most of the height
    int trunkHeight = totalHeight - 1;

    // Place trunk
    for (int dy = 0; dy < trunkHeight; ++dy) {
        int y = localY + 1 + dy;
        if (y < CHUNK_HEIGHT) {
            chunk.setBlock(localX, y, localZ, BlockType::LOG);
        }
    }

    // Place cone canopy: 3-5 layers, each 1 block smaller than previous
    int numLayers = 3 + (treeRand(state) % 3);
    int baseRadius = numLayers;

    for (int layer = 0; layer < numLayers; ++layer) {
        int layerY = localY + trunkHeight - layer;
        int radius = baseRadius - layer;

        for (int dx = -radius; dx <= radius; ++dx) {
            for (int dz = -radius; dz <= radius; ++dz) {
                // Diamond shape for cone
                if (std::abs(dx) + std::abs(dz) > radius) continue;

                int tx = localX + dx;
                int ty = layerY;
                int tz = localZ + dz;

                if (tx < 0 || tx >= CHUNK_WIDTH) continue;
                if (ty < 0 || ty >= CHUNK_HEIGHT) continue;
                if (tz < 0 || tz >= CHUNK_DEPTH) continue;

                BlockType existing = chunk.getBlock(tx, ty, tz);
                if (existing == BlockType::AIR) {
                    chunk.setBlock(tx, ty, tz, BlockType::LEAVES);
                }
            }
        }
    }
}

void TreeGenerator::generate(Chunk& chunk, const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes) const {
    int worldBaseX = chunk.chunkX * CHUNK_WIDTH;
    int worldBaseZ = chunk.chunkZ * CHUNK_DEPTH;

    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            int xzIndex = x + z * CHUNK_WIDTH;
            Biome biome = biomes[xzIndex];

            // Determine tree density based on biome
            double density;
            switch (biome) {
                case Biome::Forest:       density = 0.15; break;
                case Biome::Plains:       density = 0.03; break;
                case Biome::Taiga:        density = 0.12; break;
                case Biome::Swamp:        density = 0.05; break;
                default:                  density = 0.0;  break;  // Desert, Ocean, etc.
            }

            if (density <= 0.0) continue;

            // Use noise to determine if a tree spawns at this position
            double noiseVal = treeDensity_.noise2D(
                static_cast<double>(worldBaseX + x) * 0.5,
                static_cast<double>(worldBaseZ + z) * 0.5
            );
            // Map from [-1, 1] to [0, 1]
            double spawnChance = (noiseVal + 1.0) * 0.5;

            if (spawnChance > density) continue;

            // Find the surface height at this position
            int surfaceY = chunk.heightMap[xzIndex];

            // Check if we can place a tree here
            if (!canPlaceTree(chunk, x, surfaceY, z)) continue;

            // Choose tree type based on biome
            bool isPine = (biome == Biome::Taiga);

            if (isPine) {
                generatePine(chunk, x, surfaceY, z);
            } else {
                generateOak(chunk, x, surfaceY, z);
            }
        }
    }
}
