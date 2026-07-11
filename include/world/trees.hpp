#pragma once
#include "world/chunk.hpp"
#include "world/noise.hpp"
#include <array>
#include <cstdint>

class TreeGenerator {
public:
    explicit TreeGenerator(uint32_t seed);

    // Generate trees in a chunk
    void generate(Chunk& chunk, const std::array<Biome, CHUNK_WIDTH * CHUNK_DEPTH>& biomes) const;

private:
    SimplexNoise treeDensity_;

    void generateOak(Chunk& chunk, int localX, int localY, int localZ) const;
    void generatePine(Chunk& chunk, int localX, int localY, int localZ) const;

    bool canPlaceTree(Chunk& chunk, int localX, int localY, int localZ) const;

    // Deterministic PRNG helper
    static uint32_t treeRand(uint32_t& state);
};
