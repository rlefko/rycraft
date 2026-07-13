#include "world/terrain.hpp"

#include <algorithm>
#include <cmath>

TerrainGenerator::TerrainGenerator(uint32_t seed)
    : heightNoise_(seed)
    , rangeLow_(seed + 1)
    , rangeHigh_(seed + 2)
    , rangeSelector_(seed + 3) {}

double TerrainGenerator::getNoise(double x, double z) const {
    return heightNoise_(x, z);
}

double TerrainGenerator::generateBase(double x, double z, const TerrainConfig& config) const {
    double val = heightNoise_.octave2D(x * config.baseFrequency, z * config.baseFrequency,
                                       config.octaves, config.persistence, config.lacunarity);
    // Map from [-1, 1] to [0, 1]
    return (val + 1.0) * 0.5;
}

double TerrainGenerator::generateRange(double x, double z, const TerrainConfig& config) const {
    // Low terrain field (plains)
    double low = rangeLow_.octave2D(x * config.rangeFrequency, z * config.rangeFrequency, 4,
                                    config.rangePersistence, config.lacunarity);
    low = (low + 1.0) * 0.5; // [0, 1]

    // High terrain field (mountains) — offset frequency to decorrelate
    double high =
        rangeHigh_.octave2D(x * config.rangeFrequency * 1.5, z * config.rangeFrequency * 1.5, 4,
                            config.rangePersistence, config.lacunarity);
    high = (high + 1.0) * 0.5; // [0, 1]

    // Selector determines which field dominates
    double selector = rangeSelector_.noise2D(x * config.rangeFrequency, z * config.rangeFrequency);
    // Map from [-1, 1] to [0, 1]
    selector = (selector + 1.0) * 0.5;

    // Smoothstep interpolation around threshold for seamless transitions
    double t = selector - config.rangeThreshold;
    // Clamp to [0, 1]
    t = std::max(0.0, std::min(1.0, t));
    // Smoothstep: t * t * (3 - 2 * t)
    double smooth = t * t * (3.0 - 2.0 * t);

    // Crossfade between low and high terrain
    return low * (1.0 - smooth) + high * smooth;
}

double TerrainGenerator::getHeight(double x, double z, const TerrainConfig& config) const {
    double base = generateBase(x, z, config);
    double range = generateRange(x, z, config);

    // Blend base and range noise — range adds dramatic variation
    double combined = base * 0.6 + range * 0.4;

    // Map to [minHeight, maxHeight]
    double height = config.minHeight + combined * (config.maxHeight - config.minHeight);

    return height;
}
