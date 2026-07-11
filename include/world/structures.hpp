#pragma once
#include "world/chunk.hpp"
#include <array>
#include <cstdint>
#include <string>

class StructureGenerator {
public:
    explicit StructureGenerator(uint32_t seed);

    // Generate structures in a chunk
    void generate(Chunk& chunk, const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes) const;

private:
    uint32_t seed_;

    // Grid-based placement
    bool shouldPlaceStructure(int chunkX, int chunkZ, const std::string& type) const;

    // Simple structure: small house
    void generateHouse(Chunk& chunk, int localX, int localY, int localZ) const;

    // Hash helper
    static uint32_t hashCoords(int x, int z, uint32_t seed);
};
