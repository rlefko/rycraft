#pragma once

#include "common/math.hpp"

#include <cstdint>

inline constexpr uint64_t CELESTIAL_TICKS_PER_DAY = 24'000;
// The mean Earth synodic month rounded to one deterministic game tick.
inline constexpr uint64_t CELESTIAL_SYNODIC_PERIOD_TICKS = 708'734;
inline constexpr uint64_t CELESTIAL_FULL_MOON_REFERENCE_TICK = 18'000;
inline constexpr float SOLAR_ANGULAR_RADIUS_RADIANS = 0.004675F;
inline constexpr float LUNAR_ANGULAR_RADIUS_RADIANS = 0.00452F;

enum class CelestialLightSource : uint8_t {
    NONE,
    SUN,
    MOON,
};

// One deterministic authority for sky discs, direct lighting, shadows,
// clouds, water, and volumetrics. The true solar state remains separate from
// the selected direct source so twilight atmosphere never inherits moonlight.
struct CelestialState {
    Vec3 sunDirection;
    Vec3 moonDirection;
    Vec3 directLightDirection;
    Vec3 directLightRadiance;
    Vec3 solarDiscRadiance;
    Vec3 lunarDiscRadiance;
    Vec3 ambientRadiance;
    float sunVisibility = 0.0F;
    // Direct solar irradiance rises much more slowly than the apparent disc.
    // Near the horizon, the long atmospheric path can leave the Sun visible
    // while its contribution to terrain lighting is still weak.
    float sunDirectVisibility = 0.0F;
    float moonVisibility = 0.0F;
    float moonDirectVisibility = 0.0F;
    float illuminatedFraction = 0.0F;
    float phaseEnergy = 0.0F;
    float phaseCycle = 0.0F;
    float starVisibility = 0.0F;
    float shadowStrength = 0.0F;
    // Diffuse lunar radiance already follows the physical phase function.
    // Specular receivers multiply this once more to keep thin phases from
    // producing a full-Moon highlight.
    float directSpecularFactor = 0.0F;
    CelestialLightSource directSource = CelestialLightSource::NONE;
};

CelestialState computeCelestialState(uint64_t worldTime) noexcept;
