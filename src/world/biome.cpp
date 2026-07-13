#include "world/biome.hpp"

#include <cmath>

BiomeGenerator::BiomeGenerator(uint32_t seed) : temperatureNoise_(seed), moistureNoise_(seed + 1) {}

double BiomeGenerator::getTemperature(double x, double z) const {
    double val = temperatureNoise_.octave2D(x * 0.025, z * 0.025, 4, 0.5, 2.0);
    // Map from [-1, 1] to [0, 1]
    return (val + 1.0) * 0.5;
}

double BiomeGenerator::getMoisture(double x, double z) const {
    double val = moistureNoise_.octave2D(x * 0.05, z * 0.05, 4, 0.5, 2.0);
    return (val + 1.0) * 0.5;
}

Biome BiomeGenerator::lookupBiome(double temperature, double moisture, double elevation) const {
    constexpr double seaLevel = 64.0;

    // Below sea level: ocean biomes
    if (elevation < seaLevel - 8.0) {
        return Biome::DEEP_OCEAN;
    }
    if (elevation < seaLevel) {
        return Biome::OCEAN;
    }

    // Swamp: low elevation, very wet
    if (elevation < seaLevel + 4.0 && moisture > 0.6) {
        return Biome::SWAMP;
    }

    // Cold biomes
    if (temperature < 0.3) {
        if (moisture < 0.3) {
            return Biome::EXTREME_HILLS;
        }
        if (moisture < 0.5) {
            return Biome::ICE_SPIKES;
        }
        return Biome::TAIGA;
    }

    // Hot + dry = Desert
    if (temperature > 0.7 && moisture < 0.3) {
        return Biome::DESERT;
    }

    // Warm + wet = Forest
    if (moisture > 0.5) {
        return Biome::FOREST;
    }

    // Default: Plains
    return Biome::PLAINS;
}

Biome BiomeGenerator::getBiome(double x, double z, double elevation,
                               const BiomeConfig& config) const {
    double temp =
        temperatureNoise_.octave2D(x * config.temperatureFrequency, z * config.temperatureFrequency,
                                   config.temperatureOctaves, 0.5, 2.0);
    temp = (temp + 1.0) * 0.5;

    double moist =
        moistureNoise_.octave2D(x * config.moistureFrequency, z * config.moistureFrequency,
                                config.moistureOctaves, 0.5, 2.0);
    moist = (moist + 1.0) * 0.5;

    return lookupBiome(temp, moist, elevation);
}

double BiomeGenerator::getBiomeHeightModifier(Biome biome) const {
    switch (biome) {
        case Biome::EXTREME_HILLS:
            return 30.0;
        case Biome::DESERT:
            return 5.0;
        case Biome::FOREST:
            return 10.0;
        case Biome::TAIGA:
            return 15.0;
        case Biome::ICE_SPIKES:
            return 20.0;
        case Biome::SWAMP:
            return -5.0;
        case Biome::PLAINS:
        case Biome::OCEAN:
        case Biome::DEEP_OCEAN:
        case Biome::MUSHROOM_ISLAND:
        default:
            return 0.0;
    }
}

BlockType BiomeGenerator::getSurfaceBlock(Biome biome) const {
    switch (biome) {
        case Biome::DESERT:
        case Biome::DEEP_OCEAN:
        case Biome::OCEAN:
            return BlockType::AIR; // Water fills ocean, sand below

        case Biome::PLAINS:
        case Biome::FOREST:
        case Biome::TAIGA:
        case Biome::SWAMP:
        case Biome::MUSHROOM_ISLAND:
            return BlockType::GRASS;

        case Biome::EXTREME_HILLS:
        case Biome::ICE_SPIKES:
            return BlockType::STONE;

        default:
            return BlockType::DIRT;
    }
}
