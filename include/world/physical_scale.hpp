#pragma once

#include "world/learned_terrain.hpp"

#include <cmath>

struct WorldPhysicalScale {
    double horizontalMetersPerBlock = 1.0;
    double positiveVerticalMetersPerBlock = 1.0;
    double altitudeDatumY = 0.0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return horizontalMetersPerBlock > 0.0 && positiveVerticalMetersPerBlock > 0.0;
    }
};

inline constexpr WorldPhysicalScale LEGACY_WORLD_PHYSICAL_SCALE{};
inline constexpr WorldPhysicalScale GENERATOR_V4_PHYSICAL_SCALE{
    .horizontalMetersPerBlock = worldgen::learned::WORLD_METERS_PER_BLOCK,
    .positiveVerticalMetersPerBlock = worldgen::learned::WORLD_METERS_PER_BLOCK,
    .altitudeDatumY = static_cast<double>(worldgen::learned::LEARNED_SEA_LEVEL),
};

[[nodiscard]] constexpr WorldPhysicalScale worldPhysicalScale(bool usesLearnedAuthority) noexcept {
    return usesLearnedAuthority ? GENERATOR_V4_PHYSICAL_SCALE : LEGACY_WORLD_PHYSICAL_SCALE;
}

[[nodiscard]] constexpr bool usesGeneratorV4PhysicalScale(WorldPhysicalScale scale) noexcept {
    return scale.horizontalMetersPerBlock == GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock &&
           scale.positiveVerticalMetersPerBlock ==
               GENERATOR_V4_PHYSICAL_SCALE.positiveVerticalMetersPerBlock &&
           scale.altitudeDatumY == GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY;
}

[[nodiscard]] inline double altitudeMetersFromWorldY(double worldY,
                                                     WorldPhysicalScale scale) noexcept {
    return std::max(0.0, (worldY - scale.altitudeDatumY) * scale.positiveVerticalMetersPerBlock);
}

[[nodiscard]] inline double worldYFromAltitudeMeters(double altitudeMeters,
                                                     WorldPhysicalScale scale) noexcept {
    return scale.altitudeDatumY +
           std::max(0.0, altitudeMeters) / scale.positiveVerticalMetersPerBlock;
}

[[nodiscard]] inline double worldYAboveTerrainMeters(double terrainWorldY, double heightMeters,
                                                     WorldPhysicalScale scale) noexcept {
    return terrainWorldY + std::max(0.0, heightMeters) / scale.positiveVerticalMetersPerBlock;
}

[[nodiscard]] inline double worldDistanceMeters(double deltaX, double deltaY, double deltaZ,
                                                WorldPhysicalScale scale) noexcept {
    return std::hypot(std::hypot(deltaX * scale.horizontalMetersPerBlock,
                                 deltaZ * scale.horizontalMetersPerBlock),
                      deltaY * scale.positiveVerticalMetersPerBlock);
}
