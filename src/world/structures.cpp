#include "world/structures.hpp"

#include "common/random.hpp"

#include <cmath>

StructureGenerator::StructureGenerator(uint32_t seed)
    : seed_(seed)
{
}

uint32_t StructureGenerator::hashCoords(int x, int z, uint32_t seed) {
    return static_cast<uint32_t>(::hashCoords(x, z, seed));
}

bool StructureGenerator::shouldPlaceStructure(int chunkX, int chunkZ, const std::string& type) const {
    // Grid-based placement using hash
    uint32_t hash = hashCoords(chunkX, chunkZ, seed_);

    // Different structure types have different spacing
    int spacing;
    if (type == "house") {
        spacing = 16;  // One house every ~16 chunks
    } else {
        spacing = 32;  // Default spacing
    }

    return (hash % spacing) < 2;  // 2 in spacing chance
}

void StructureGenerator::generateHouse(Chunk& chunk, int localX, int localY, int localZ) const {
    // 5x5x4 house made of PLANKS
    int houseWidth = 5;
    int houseDepth = 5;
    int houseHeight = 4;

    for (int dz = 0; dz < houseDepth; ++dz) {
        for (int dx = 0; dx < houseWidth; ++dx) {
            for (int dy = 0; dy < houseHeight; ++dy) {
                int tx = localX + dx;
                int ty = localY + dy;
                int tz = localZ + dz;

                // Bounds check
                if (tx < 0 || tx >= CHUNK_WIDTH) continue;
                if (ty < 0 || ty >= CHUNK_HEIGHT) continue;
                if (tz < 0 || tz >= CHUNK_DEPTH) continue;

                // Build walls, floor, and roof
                bool isFloor = (dy == 0);
                bool isRoof = (dy == houseHeight - 1);
                bool isWall = (dx == 0 || dx == houseWidth - 1 ||
                               dz == 0 || dz == houseDepth - 1);

                // Place blocks for walls, floor, and roof
                if (isFloor || isRoof || isWall) {
                    // Don't overwrite non-air blocks except at ground level
                    BlockType existing = chunk.getBlock(tx, ty, tz);
                    if (existing == BlockType::AIR || (isFloor && existing != BlockType::BEDROCK)) {
                        chunk.setBlock(tx, ty, tz, BlockType::PLANKS);
                    }
                }
            }
        }
    }
}

bool biomeAllowsStructure(Biome biome) {
    switch (biome) {
        case Biome::PLAINS:
        case Biome::FOREST:
            return true;
        default:
            return false;
    }
}

void StructureGenerator::generate(Chunk& chunk, const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes) const {
    // Check if any structure should spawn in this chunk
    if (!shouldPlaceStructure(chunk.chunkX, chunk.chunkZ, "house")) {
        return;
    }

    // Check if the biome allows structures
    // Use the biome at the center of the chunk
    int centerX = CHUNK_WIDTH / 2;
    int centerZ = CHUNK_DEPTH / 2;
    Biome centerBiome = biomes[centerX + centerZ * CHUNK_WIDTH];

    if (!biomeAllowsStructure(centerBiome)) {
        return;
    }

    // Find a suitable placement position
    // Scan from center outward for a GRASS block with clearance
    int startX = CHUNK_WIDTH / 2 - 2;
    int startZ = CHUNK_DEPTH / 2 - 2;

    for (int searchZ = 0; searchZ < CHUNK_DEPTH; ++searchZ) {
        for (int searchX = 0; searchX < CHUNK_WIDTH; ++searchX) {
            int testX = startX + searchX;
            int testZ = startZ + searchZ;

            if (testX < 0 || testX >= CHUNK_WIDTH - 5) continue;
            if (testZ < 0 || testZ >= CHUNK_DEPTH - 5) continue;

            int xzIndex = testX + testZ * CHUNK_WIDTH;
            int surfaceY = chunk.heightMap[xzIndex];

            // Must be on grass
            if (surfaceY < 0 || surfaceY >= CHUNK_HEIGHT) continue;
            if (chunk.getBlock(testX, surfaceY, testZ) != BlockType::GRASS) continue;

            // Check clearance above (need 4 blocks of AIR)
            bool clear = true;
            for (int dy = 1; dy <= 4; ++dy) {
                int checkY = surfaceY + dy;
                if (checkY >= CHUNK_HEIGHT) { clear = false; break; }
                if (chunk.getBlock(testX, checkY, testZ) != BlockType::AIR) {
                    clear = false;
                    break;
                }
            }

            if (!clear) continue;

            // Place the house
            generateHouse(chunk, testX, surfaceY + 1, testZ);
            return;  // Only one structure per chunk
        }
    }
}
