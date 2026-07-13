#pragma once
#include "world/chunk.hpp"
#include <array>
#include <vector>

class SurfaceGenerator {
public:
    // Fill surface blocks for a chunk based on terrain height and biome
    static void generateSurface(Chunk& chunk, const std::vector<double>& heights,
                                const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes);

private:
    static int getSurfaceHeight(Biome biome, double baseHeight);
    static BlockType getSurfaceBlockType(Biome biome);
    static BlockType getSubsurfaceBlockType(Biome biome);
    static bool isDesert(Biome biome);
};
