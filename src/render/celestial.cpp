#include "render/celestial.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float PI = 3.14159265358979323846F;
constexpr float TWO_PI = 2.0F * PI;

float smoothstep(float edge0, float edge1, float value) noexcept {
    const float amount = std::clamp((value - edge0) / (edge1 - edge0), 0.0F, 1.0F);
    return amount * amount * (3.0F - 2.0F * amount);
}

Vec3 tiltedOrbitDirection(float angle) noexcept {
    constexpr float ORBIT_TILT = 0.20F;
    return Vec3{std::cos(angle), std::sin(angle) * std::cos(ORBIT_TILT),
                std::sin(angle) * std::sin(ORBIT_TILT)}
        .normalize();
}

Vec3 mix(const Vec3& from, const Vec3& to, float amount) noexcept {
    return from.lerp(to, amount);
}

} // namespace

CelestialState computeCelestialState(uint64_t worldTime) noexcept {
    CelestialState result;
    const double dayFraction = static_cast<double>(worldTime % CELESTIAL_TICKS_PER_DAY) /
                               static_cast<double>(CELESTIAL_TICKS_PER_DAY);
    const float solarAngle = static_cast<float>(dayFraction * static_cast<double>(TWO_PI));
    const double phasePeriod = static_cast<double>(CELESTIAL_SYNODIC_PERIOD_TICKS);
    const uint64_t phaseTick =
        (worldTime % CELESTIAL_SYNODIC_PERIOD_TICKS + CELESTIAL_SYNODIC_PERIOD_TICKS -
         CELESTIAL_FULL_MOON_REFERENCE_TICK) %
        CELESTIAL_SYNODIC_PERIOD_TICKS;
    result.phaseCycle = static_cast<float>(static_cast<double>(phaseTick) / phasePeriod);

    result.sunDirection = tiltedOrbitDirection(solarAngle);
    const float lunarAngle = solarAngle + PI - TWO_PI * result.phaseCycle;
    result.moonDirection = tiltedOrbitDirection(lunarAngle);

    const float separation = std::clamp(result.sunDirection.dot(result.moonDirection), -1.0F, 1.0F);
    result.illuminatedFraction = std::clamp(0.5F * (1.0F - separation), 0.0F, 1.0F);
    const float phaseAngle = std::acos(std::clamp(-separation, -1.0F, 1.0F));
    result.phaseEnergy = std::clamp(
        (std::sin(phaseAngle) + (PI - phaseAngle) * std::cos(phaseAngle)) / PI, 0.0F, 1.0F);

    const float solarElevation = result.sunDirection.y;
    const float lunarElevation = result.moonDirection.y;
    result.sunVisibility =
        smoothstep(-SOLAR_ANGULAR_RADIUS_RADIANS, SOLAR_ANGULAR_RADIUS_RADIANS, solarElevation);
    // Disc visibility is not irradiance. At sunrise and sunset the direct
    // beam crosses far more atmosphere than it does overhead, so turning a
    // half-visible disc directly into half-strength noon lighting makes the
    // ground look daylit under a twilight sky. Ease the beam over the first
    // ten degrees while leaving the independently rendered disc visible.
    constexpr float DIRECT_SUN_FULL_ELEVATION = 10.0F * PI / 180.0F;
    result.sunDirectVisibility =
        result.sunVisibility *
        smoothstep(0.0F, std::sin(DIRECT_SUN_FULL_ELEVATION), std::max(solarElevation, 0.0F));
    const float moonAboveHorizon =
        smoothstep(-LUNAR_ANGULAR_RADIUS_RADIANS, LUNAR_ANGULAR_RADIUS_RADIANS, lunarElevation);
    // Keep the Moon from becoming a second bright authority during civil
    // twilight. It begins contributing after the Sun reaches -6 degrees and
    // reaches full strength at the end of nautical twilight (-12 degrees).
    const float nightWeight = 1.0F - smoothstep(std::sin(-12.0F * PI / 180.0F),
                                                std::sin(-6.0F * PI / 180.0F), solarElevation);
    result.moonVisibility = moonAboveHorizon * (0.01F + 0.99F * nightWeight);
    result.moonDirectVisibility = moonAboveHorizon * nightWeight * result.phaseEnergy;
    result.starVisibility = 1.0F - smoothstep(std::sin(-12.0F * PI / 180.0F),
                                              std::sin(-4.0F * PI / 180.0F), solarElevation);

    const float daylightColor = smoothstep(0.0F, 0.35F, solarElevation);
    result.solarDiscRadiance = mix({1.0F, 0.46F, 0.18F}, {1.0F, 1.0F, 0.96F}, daylightColor);
    // The sunlit face peaks just above the 1.0 bloom threshold, so a full
    // moon carries a slight glow while staying an order of magnitude below
    // the sun disc's 18x on-screen radiance. The sphere-lit terminator and
    // surface variation keep the disc from flattening into a white circle.
    result.lunarDiscRadiance = {1.60F, 1.80F, 2.10F};

    if (result.sunDirectVisibility > 0.0001F) {
        result.directSource = CelestialLightSource::SUN;
        result.directLightDirection = result.sunDirection;
        result.directLightRadiance = result.solarDiscRadiance * result.sunDirectVisibility;
        result.shadowStrength = result.sunDirectVisibility;
        result.directSpecularFactor = 1.0F;
    } else if (result.moonDirectVisibility > 0.0001F) {
        result.directSource = CelestialLightSource::MOON;
        result.directLightDirection = result.moonDirection;
        // Playability over strict photometry: real moonlight is orders of
        // magnitude below this, but the exposure ceiling stays fixed so caves
        // read dark, and a full-moon surface must stay legible in motion.
        result.directLightRadiance = Vec3{0.016F, 0.022F, 0.034F} * result.moonDirectVisibility;
        result.shadowStrength = 0.07F * std::sqrt(result.moonDirectVisibility);
        result.directSpecularFactor = result.phaseEnergy;
    } else {
        result.directLightDirection = result.sunDirection;
        result.directLightRadiance = Vec3::zero();
    }

    const float ambientDay = smoothstep(-0.20F, 0.40F, solarElevation);
    // Keep propagated skylight an accessibility signal, not a phase-blind
    // nighttime light source. A small stellar floor preserves silhouettes;
    // the Moon adds cool ambient irradiance only while it is above the
    // horizon and after twilight, following the same physical phase energy
    // as its directional contribution.
    const Vec3 nightAmbient =
        Vec3{0.0050F, 0.0080F, 0.0150F} +
        Vec3{0.0110F, 0.0160F, 0.0240F} * (moonAboveHorizon * nightWeight * result.phaseEnergy);
    result.ambientRadiance = mix(nightAmbient, {0.35F, 0.35F, 0.40F}, ambientDay);
    return result;
}
