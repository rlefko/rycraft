#pragma once

#include "world/noise.hpp"

#include <cstdint>

struct TerrainConfig {
    // Base noise
    int octaves = 7;
    double persistence = 0.5;
    double lacunarity = 2.0;
    double baseFrequency = 0.005;

    // Height range
    double minHeight = 20.0;  // Ocean floor
    double maxHeight = 128.0; // Mountain peaks
    double seaLevel = 64.0;

    // Range noise for dramatic transitions
    double rangeFrequency = 0.002;
    double rangePersistence = 0.4;
    double rangeThreshold = 0.3;
};

class TerrainGenerator {
public:
    explicit TerrainGenerator(uint32_t seed);

    // Get terrain height at world position
    double getHeight(double x, double z, const TerrainConfig& config = {}) const;

    // Get raw noise value (for debugging)
    double getNoise(double x, double z) const;

private:
    SimplexNoise heightNoise_;
    SimplexNoise rangeLow_;
    SimplexNoise rangeHigh_;
    SimplexNoise rangeSelector_;

    double generateBase(double x, double z, const TerrainConfig& config) const;
    double generateRange(double x, double z, const TerrainConfig& config) const;
};
