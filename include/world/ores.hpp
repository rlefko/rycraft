#pragma once
#include "world/chunk.hpp"
#include "world/noise.hpp"
#include <cstdint>
#include <vector>

struct OreConfig {
    struct OreDistribution {
        BlockType ore;
        int minVeinSize;
        int maxVeinSize;
        double minHeight;
        double maxHeight;
        double falloffHeight;  // where density starts decreasing (trapezoid peak)
        int clustersPerChunk;
        double discardThreshold; // noise threshold to discard
    };

    std::vector<OreDistribution> ores = {
        {BlockType::COAL_ORE, 4, 8, 0, 128, 80, 16, 0.7},
        {BlockType::IRON_ORE, 4, 8, 0, 96, 64, 8, 0.8},
        {BlockType::GOLD_ORE, 2, 4, 0, 32, 16, 4, 0.85},
        {BlockType::DIAMOND_ORE, 1, 3, 0, 16, 8, 2, 0.9}
    };
};

class OreGenerator {
public:
    explicit OreGenerator(uint32_t seed);

    // Generate ore deposits in a chunk
    void generate(Chunk& chunk, const OreConfig& config = {}) const;

private:
    SimplexNoise oreNoise_;

    void generateOreVein(Chunk& chunk, int x, int y, int z,
                         int veinSize, BlockType ore) const;
    double getOreDensity(double y, const OreConfig::OreDistribution& dist) const;
};
