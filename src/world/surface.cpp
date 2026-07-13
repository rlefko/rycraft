#include "world/surface.hpp"

#include <cmath>

constexpr int SEA_LEVEL = 64;
constexpr int BEDROCK_LEVEL = 1;

int SurfaceGenerator::getSurfaceHeight(Biome biome, double baseHeight) {
    int height = static_cast<int>(std::round(baseHeight));

    // Apply biome-specific height adjustments
    switch (biome) {
        case Biome::EXTREME_HILLS: height += 15; break;
        case Biome::ICE_SPIKES:    height += 10; break;
        case Biome::TAIGA:        height += 8;  break;
        case Biome::FOREST:       height += 5;  break;
        case Biome::DESERT:       height += 3;  break;
        case Biome::SWAMP:        height -= 3;  break;
        default:                  break;
    }

    // Clamp to valid range
    return std::max(2, std::min(CHUNK_HEIGHT - 1, height));
}

BlockType SurfaceGenerator::getSurfaceBlockType(Biome biome) {
    switch (biome) {
        case Biome::DESERT:       return BlockType::SAND;
        case Biome::ICE_SPIKES:    return BlockType::SNOW;
        case Biome::EXTREME_HILLS: return BlockType::STONE;
        case Biome::PLAINS:
        case Biome::FOREST:
        case Biome::TAIGA:
        case Biome::SWAMP:
        case Biome::MUSHROOM_ISLAND:
            return BlockType::GRASS;
        default:
            return BlockType::DIRT;
    }
}

BlockType SurfaceGenerator::getSubsurfaceBlockType(Biome biome) {
    switch (biome) {
        case Biome::DESERT:       return BlockType::SAND;
        case Biome::TAIGA:        return BlockType::GRAVEL;
        default:
            return BlockType::DIRT;
    }
}

bool SurfaceGenerator::isDesert(Biome biome) {
    return biome == Biome::DESERT;
}

void SurfaceGenerator::generateSurface(Chunk& chunk,
                                       const std::vector<double>& heights,
                                       const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes) {
    // Ensure heights array is sized correctly
    if (heights.size() < static_cast<size_t>(CHUNK_WIDTH * CHUNK_DEPTH)) {
        return;
    }

    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            int xzIndex = x + z * CHUNK_WIDTH;
            Biome biome = biomes[xzIndex];
            double baseHeight = heights[xzIndex];
            int surfaceY = getSurfaceHeight(biome, baseHeight);

            BlockType surfaceBlock = getSurfaceBlockType(biome);
            BlockType subsurfaceBlock = getSubsurfaceBlockType(biome);
            bool desert = isDesert(biome);

            for (int y = 0; y < CHUNK_HEIGHT; ++y) {
                BlockType block = BlockType::AIR;

                if (y <= BEDROCK_LEVEL) {
                    // Bedrock layer at the bottom
                    block = BlockType::BEDROCK;
                } else if (y < surfaceY - 4) {
                    // Deep stone
                    block = BlockType::STONE;
                } else if (y < surfaceY - 1) {
                    // Subsurface layer (dirt/sand/gravel)
                    block = subsurfaceBlock;
                } else if (y == surfaceY) {
                    // Surface block
                    block = surfaceBlock;
                } else if (y < SEA_LEVEL && !desert) {
                    // Water fill up to sea level (non-desert biomes)
                    block = BlockType::WATER;
                }
                // Above surface and above sea level: AIR (already default)

                chunk.setBlock(x, y, z, block);
            }

            // Update height map
            chunk.heightMap[xzIndex] = surfaceY;
        }
    }

    chunk.needsMeshUpdate = true;
}
