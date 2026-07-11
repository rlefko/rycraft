#pragma once

#include "world/noise.hpp"
#include "world/chunk.hpp"

#include <cstdint>

struct BiomeConfig {
    double temperatureFrequency = 0.025;
    double moistureFrequency = 0.05;
    int temperatureOctaves = 4;
    int moistureOctaves = 4;
};

class BiomeGenerator {
public:
    explicit BiomeGenerator(uint32_t seed);

    // Get biome at world position
    Biome getBiome(double x, double z, double elevation, const BiomeConfig& config = {}) const;

    // Get temperature/moisture at world position
    double getTemperature(double x, double z) const;
    double getMoisture(double x, double z) const;

    // Get biome-specific terrain height modifier
    double getBiomeHeightModifier(Biome biome) const;

    // Get biome-specific surface block
    BlockType getSurfaceBlock(Biome biome) const;

    // Direct biome lookup from temperature/moisture/elevation
    Biome lookupBiome(double temperature, double moisture, double elevation) const;

private:
    SimplexNoise temperatureNoise_;
    SimplexNoise moistureNoise_;
};
