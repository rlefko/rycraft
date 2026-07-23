#include "world/weather.hpp"

#include "common/counter_rng.hpp"
#include "common/error.hpp"
#include "common/thread_priority.hpp"
#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace {

constexpr double TICKS_PER_SECOND = 20.0;
constexpr double SPEED_OF_SOUND_METERS_PER_SECOND = 343.0;
constexpr uint64_t PRESSURE_STREAM = 0x5745415448455201ULL;
constexpr uint64_t MOISTURE_STREAM = 0x5745415448455202ULL;
constexpr uint64_t AIR_MASS_STREAM = 0x5745415448455203ULL;
constexpr uint64_t TEMPERATURE_STREAM = 0x5745415448455204ULL;
constexpr uint64_t CLOUD_FLOW_STREAM = 0x5745415448455211ULL;
constexpr uint64_t LIGHTNING_STREAM = 0x57454154484552A1ULL;
constexpr uint64_t LIGHTNING_ID_STREAM = 0x57454154484552A2ULL;

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double smoothstep(double edge0, double edge1, double value) {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    const double t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

float smoothstepFloat(float edge0, float edge1, float value) {
    if (edge0 == edge1) return value < edge0 ? 0.0f : 1.0f;
    const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

double quintic(double value) {
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
}

double latticeNoise(const CounterRng& random, uint64_t stream, double x, double z) {
    const double floorX = std::floor(x);
    const double floorZ = std::floor(z);
    const int64_t x0 = static_cast<int64_t>(floorX);
    const int64_t z0 = static_cast<int64_t>(floorZ);
    const double tx = quintic(x - floorX);
    const double tz = quintic(z - floorZ);
    const double n00 = random.signedUnit(stream, x0, 0, z0);
    const double n10 = random.signedUnit(stream, x0 + 1, 0, z0);
    const double n01 = random.signedUnit(stream, x0, 0, z0 + 1);
    const double n11 = random.signedUnit(stream, x0 + 1, 0, z0 + 1);
    const double nx0 = std::lerp(n00, n10, tx);
    const double nx1 = std::lerp(n01, n11, tx);
    return std::lerp(nx0, nx1, tz);
}

double fractalNoise(const CounterRng& random, uint64_t stream, double x, double z,
                    double wavelength) {
    double value = 0.0;
    double amplitude = 0.58;
    double normalizer = 0.0;
    double frequency = 1.0;
    for (uint32_t octave = 0; octave < 4; ++octave) {
        value += amplitude * latticeNoise(random, stream + octave, x * frequency / wavelength,
                                          z * frequency / wavelength);
        normalizer += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value / normalizer;
}

Vec2 normalizedOrFallback(Vec2 value) {
    const float length = value.length();
    if (length > 1.0e-5f) return value / length;
    return Vec2{0.936329f, 0.351123f};
}

Vec2 canonicalCloudVelocity(const CounterRng& random) {
    constexpr double TWO_PI = 6.28318530717958647692;
    const double angle = random.uniform01(CLOUD_FLOW_STREAM, 0, 0, 0) * TWO_PI;
    const float speed =
        static_cast<float>(4.5 + random.uniform01(CLOUD_FLOW_STREAM + 1U, 0, 0, 0) * 1.5);
    return {static_cast<float>(std::cos(angle)) * speed,
            static_cast<float>(std::sin(angle)) * speed};
}

CloudType classifyCloudType(float coverage, float stormPotential, float front, float instability) {
    if (coverage < 0.10f) return CloudType::CLEAR;
    if (stormPotential >= 0.58f) return CloudType::CUMULONIMBUS;
    if (coverage < 0.34f) return CloudType::CIRRUS;
    if (coverage > 0.78f && instability < 0.38f) return CloudType::STRATUS;
    if (front > 0.62f && instability < 0.45f) return CloudType::STRATUS;
    return CloudType::CUMULUS;
}

void applyPreset(WeatherSample& sample, WeatherPreset preset) {
    switch (preset) {
        case WeatherPreset::NATURAL:
            return;
        case WeatherPreset::CLEAR:
            sample.windBlocksPerSecond = {2.0f, 0.5f};
            sample.pressureHpa = 1023.0f;
            sample.relativeHumidity = 0.30f;
            sample.temperatureC = 18.0f;
            sample.cloudCoverage = 0.02f;
            sample.precipitationIntensity = 0.0f;
            sample.stormPotential = 0.0f;
            sample.fogExtinction = 0.00008f;
            sample.aerosolDensity = 0.07f;
            sample.cloudBaseY = 300.0f;
            sample.cloudTopY = 340.0f;
            sample.precipitationKind = PrecipitationKind::NONE;
            sample.cloudType = CloudType::CLEAR;
            return;
        case WeatherPreset::OVERCAST:
            sample.windBlocksPerSecond = {3.5f, 1.0f};
            sample.pressureHpa = 1008.0f;
            sample.relativeHumidity = 0.86f;
            sample.temperatureC = 11.0f;
            sample.cloudCoverage = 0.94f;
            sample.precipitationIntensity = 0.0f;
            sample.stormPotential = 0.12f;
            sample.fogExtinction = 0.0012f;
            sample.aerosolDensity = 0.16f;
            sample.cloudBaseY = 152.0f;
            sample.cloudTopY = 198.0f;
            sample.precipitationKind = PrecipitationKind::NONE;
            sample.cloudType = CloudType::STRATUS;
            return;
        case WeatherPreset::RAIN:
            sample.windBlocksPerSecond = {5.0f, 1.5f};
            sample.pressureHpa = 1003.0f;
            sample.relativeHumidity = 0.96f;
            sample.temperatureC = 10.0f;
            sample.cloudCoverage = 0.98f;
            sample.precipitationIntensity = 0.78f;
            sample.stormPotential = 0.32f;
            sample.fogExtinction = 0.0024f;
            sample.aerosolDensity = 0.20f;
            sample.cloudBaseY = 148.0f;
            sample.cloudTopY = 232.0f;
            sample.precipitationKind = PrecipitationKind::RAIN;
            sample.cloudType = CloudType::STRATUS;
            return;
        case WeatherPreset::STORM:
            sample.windBlocksPerSecond = {10.0f, 3.0f};
            sample.pressureHpa = 990.0f;
            sample.relativeHumidity = 1.0f;
            sample.temperatureC = 14.0f;
            sample.cloudCoverage = 1.0f;
            sample.precipitationIntensity = 1.0f;
            sample.stormPotential = 1.0f;
            sample.fogExtinction = 0.0040f;
            sample.aerosolDensity = 0.26f;
            sample.cloudBaseY = 144.0f;
            sample.cloudTopY = 456.0f;
            sample.precipitationKind = PrecipitationKind::RAIN;
            sample.cloudType = CloudType::CUMULONIMBUS;
            return;
        case WeatherPreset::SNOW:
            sample.windBlocksPerSecond = {4.0f, 1.0f};
            sample.pressureHpa = 1005.0f;
            sample.relativeHumidity = 0.94f;
            sample.temperatureC = -7.0f;
            sample.cloudCoverage = 0.97f;
            sample.precipitationIntensity = 0.72f;
            sample.stormPotential = 0.18f;
            sample.fogExtinction = 0.0020f;
            sample.aerosolDensity = 0.12f;
            sample.cloudBaseY = 142.0f;
            sample.cloudTopY = 220.0f;
            sample.precipitationKind = PrecipitationKind::SNOW;
            sample.cloudType = CloudType::STRATUS;
            return;
    }
}

WeatherSample interpolateWeather(const WeatherSample& first, const WeatherSample& second,
                                 float amount) {
    const auto mix = [amount](float a, float b) { return std::lerp(a, b, amount); };
    WeatherSample result;
    result.windBlocksPerSecond = first.windBlocksPerSecond.lerp(second.windBlocksPerSecond, amount);
    result.cloudOffsetBlocks = {std::lerp(first.cloudOffsetBlocks.x, second.cloudOffsetBlocks.x,
                                          static_cast<double>(amount)),
                                std::lerp(first.cloudOffsetBlocks.z, second.cloudOffsetBlocks.z,
                                          static_cast<double>(amount))};
    result.highCloudOffsetBlocks = {
        std::lerp(first.highCloudOffsetBlocks.x, second.highCloudOffsetBlocks.x,
                  static_cast<double>(amount)),
        std::lerp(first.highCloudOffsetBlocks.z, second.highCloudOffsetBlocks.z,
                  static_cast<double>(amount))};
    result.pressureHpa = mix(first.pressureHpa, second.pressureHpa);
    result.relativeHumidity = mix(first.relativeHumidity, second.relativeHumidity);
    result.temperatureC = mix(first.temperatureC, second.temperatureC);
    result.cloudCoverage = mix(first.cloudCoverage, second.cloudCoverage);
    result.precipitationIntensity =
        mix(first.precipitationIntensity, second.precipitationIntensity);
    result.stormPotential = mix(first.stormPotential, second.stormPotential);
    result.fogExtinction = mix(first.fogExtinction, second.fogExtinction);
    result.aerosolDensity = mix(first.aerosolDensity, second.aerosolDensity);
    result.cloudBaseY = mix(first.cloudBaseY, second.cloudBaseY);
    result.cloudTopY = mix(first.cloudTopY, second.cloudTopY);
    result.terrainHeight = mix(first.terrainHeight, second.terrainHeight);
    if (result.precipitationIntensity < 0.01f) {
        result.precipitationKind = PrecipitationKind::NONE;
    } else {
        result.precipitationKind =
            result.temperatureC <= 0.0f ? PrecipitationKind::SNOW : PrecipitationKind::RAIN;
    }
    const WeatherSample& categorical = amount < 0.5f ? first : second;
    result.cloudType =
        classifyCloudType(result.cloudCoverage, result.stormPotential,
                          categorical.cloudType == CloudType::STRATUS ? 1.0f : 0.0f,
                          categorical.cloudType == CloudType::CUMULONIMBUS ? 1.0f : 0.0f);
    return result;
}

worldgen::SurfaceSample fallbackClimate() {
    worldgen::SurfaceSample surface;
    surface.terrainHeight = 64.0;
    surface.climate.wind = {0.8, 0.3};
    surface.climate.temperatureC = 15.0;
    surface.climate.annualPrecipitationMm = 900.0;
    surface.climate.potentialEvapotranspirationMm = 760.0;
    surface.climate.aridity = 760.0 / 900.0;
    surface.climate.relativeHumidity = 0.55;
    return surface;
}

int64_t snappedWeatherCenter(int64_t coordinate) {
    constexpr int64_t SPACING = WeatherSnapshot::GRID_SPACING;
    constexpr int64_t HALF_EXTENT = static_cast<int64_t>(WeatherSnapshot::GRID_EDGE / 2) * SPACING;
    const int64_t minimumCenter = std::numeric_limits<int64_t>::min() + HALF_EXTENT;
    const int64_t maximumCenter = std::numeric_limits<int64_t>::max() - HALF_EXTENT;
    const int64_t clamped = std::clamp(coordinate, minimumCenter, maximumCenter);
    return world_coord::floorDiv(clamped, SPACING) * SPACING;
}

} // namespace

std::optional<WeatherPreset> weatherPresetFromString(std::string_view value) noexcept {
    if (value == "clear") return WeatherPreset::CLEAR;
    if (value == "overcast") return WeatherPreset::OVERCAST;
    if (value == "rain") return WeatherPreset::RAIN;
    if (value == "storm") return WeatherPreset::STORM;
    if (value == "snow") return WeatherPreset::SNOW;
    return std::nullopt;
}

Vec2 weatherAdvectionVelocity(const worldgen::ClimateFields& climate) noexcept {
    const Vec2 direction = normalizedOrFallback(
        {static_cast<float>(climate.wind.x), static_cast<float>(climate.wind.z)});
    const float climateWind =
        std::clamp(static_cast<float>(std::hypot(climate.wind.x, climate.wind.z)), 0.0f, 1.0f);
    const float speed = 2.0f + climateWind * 2.5f;
    return direction * speed;
}

CloudLayerBounds cloudLayerBounds(CloudType type, float terrainHeight, float relativeHumidity,
                                  WorldPhysicalScale physicalScale) noexcept {
    const float humidity = std::clamp(relativeHumidity, 0.0f, 1.0f);
    CloudLayerBounds result;
    if (usesGeneratorV4PhysicalScale(physicalScale)) {
        const auto aboveTerrain = [&](double meters) {
            return static_cast<float>(
                worldYAboveTerrainMeters(terrainHeight, meters, physicalScale));
        };
        const auto aboveSea = [&](double meters) {
            return static_cast<float>(worldYFromAltitudeMeters(meters, physicalScale));
        };
        switch (type) {
            case CloudType::CLEAR:
                result.baseY = std::max(aboveSea(3'000.0), aboveTerrain(1'000.0));
                result.topY =
                    result.baseY +
                    300.0F / static_cast<float>(physicalScale.positiveVerticalMetersPerBlock);
                break;
            case CloudType::CIRRUS:
                result.baseY = std::max(aboveSea(6'000.0), aboveTerrain(1'500.0));
                result.topY = result.baseY +
                              (450.0F + humidity * 300.0F) /
                                  static_cast<float>(physicalScale.positiveVerticalMetersPerBlock);
                break;
            case CloudType::STRATUS:
                result.baseY = std::max(aboveSea(1'000.0), aboveTerrain(300.0));
                result.topY = result.baseY +
                              (225.0F + humidity * 180.0F) /
                                  static_cast<float>(physicalScale.positiveVerticalMetersPerBlock);
                break;
            case CloudType::CUMULUS:
                result.baseY = std::max(aboveSea(1'500.0), aboveTerrain(450.0));
                result.topY = result.baseY +
                              (435.0F + humidity * 390.0F) /
                                  static_cast<float>(physicalScale.positiveVerticalMetersPerBlock);
                break;
            case CloudType::CUMULONIMBUS:
                result.baseY = std::max(aboveSea(1'200.0), aboveTerrain(375.0));
                result.topY =
                    std::max(aboveSea(8'000.0),
                             result.baseY + (3'000.0F + humidity * 2'000.0F) /
                                                static_cast<float>(
                                                    physicalScale.positiveVerticalMetersPerBlock));
                break;
        }
        result.baseY = std::max(result.baseY, aboveTerrain(75.0));
        result.topY = std::max(result.topY, result.baseY + 8.0F);
        return result;
    }
    switch (type) {
        case CloudType::CLEAR:
            result = {300.0f, 340.0f};
            break;
        case CloudType::CIRRUS:
            result.baseY = std::clamp(std::max(300.0f, terrainHeight + 120.0f), 300.0f, 410.0f);
            result.topY = std::min(480.0f, result.baseY + 42.0f + humidity * 18.0f);
            break;
        case CloudType::STRATUS:
            result.baseY = std::clamp(terrainHeight + 48.0f, 132.0f, 260.0f);
            result.topY = std::min(360.0f, result.baseY + 30.0f + humidity * 24.0f);
            break;
        case CloudType::CUMULUS:
            result.baseY = std::clamp(terrainHeight + 66.0f, 148.0f, 290.0f);
            result.topY = std::min(420.0f, result.baseY + 58.0f + humidity * 52.0f);
            break;
        case CloudType::CUMULONIMBUS:
            result.baseY = std::clamp(terrainHeight + 58.0f, 140.0f, 270.0f);
            result.topY =
                std::min(500.0f, std::max(420.0f, result.baseY + 150.0f + humidity * 70.0f));
            break;
    }
    result.topY = std::max(result.topY, result.baseY + 8.0f);
    return result;
}

float cloudProfileDensity(CloudType type, float normalizedHeight) noexcept {
    if (normalizedHeight <= 0.0f || normalizedHeight >= 1.0f) return 0.0f;
    const float h = normalizedHeight;
    switch (type) {
        case CloudType::CLEAR:
            return 0.0f;
        case CloudType::CIRRUS:
            return smoothstepFloat(0.0f, 0.16f, h) * (1.0f - smoothstepFloat(0.72f, 1.0f, h)) *
                   (0.45f + 0.55f * h);
        case CloudType::STRATUS:
            return smoothstepFloat(0.0f, 0.12f, h) * (1.0f - smoothstepFloat(0.78f, 1.0f, h));
        case CloudType::CUMULUS:
            return smoothstepFloat(0.0f, 0.24f, h) * (1.0f - smoothstepFloat(0.62f, 1.0f, h));
        case CloudType::CUMULONIMBUS:
            return smoothstepFloat(0.0f, 0.10f, h) * (1.0f - smoothstepFloat(0.88f, 1.0f, h));
    }
    return 0.0f;
}

WeatherSample deriveWeatherSample(uint64_t worldSeed, double worldX, double worldZ,
                                  uint64_t worldTick, const worldgen::SurfaceSample& staticClimate,
                                  WeatherPreset preset, WorldPhysicalScale physicalScale) noexcept {
    const CounterRng random(worldSeed);
    const Vec2 advection = weatherAdvectionVelocity(staticClimate.climate);
    const double seconds = static_cast<double>(worldTick) / TICKS_PER_SECOND;
    const double pressureX = worldX - static_cast<double>(advection.x) * seconds;
    const double pressureZ = worldZ - static_cast<double>(advection.y) * seconds;
    const double pressureSignal =
        fractalNoise(random, PRESSURE_STREAM, pressureX, pressureZ, 5'600.0);

    const double moistureX = worldX - static_cast<double>(advection.x) * seconds * 0.76;
    const double moistureZ = worldZ - static_cast<double>(advection.y) * seconds * 0.76;
    const double moistureSignal =
        fractalNoise(random, MOISTURE_STREAM, moistureX, moistureZ, 3'200.0);
    const double airMass = fractalNoise(
        random, AIR_MASS_STREAM, worldX - static_cast<double>(advection.x) * seconds * 1.14,
        worldZ - static_cast<double>(advection.y) * seconds * 1.14, 6'400.0);
    const double front = 1.0 - smoothstep(0.025, 0.30, std::abs(airMass));
    const double lowPressure = clamp01((-pressureSignal + 0.18) / 1.18);
    const double humidity =
        clamp01(staticClimate.climate.relativeHumidity * 0.68 +
                (moistureSignal * 0.5 + 0.5) * 0.27 + lowPressure * 0.13 + front * 0.08 - 0.06);
    const double temperatureAnomaly = fractalNoise(
        random, TEMPERATURE_STREAM, worldX - static_cast<double>(advection.x) * seconds * 0.46,
        worldZ - static_cast<double>(advection.y) * seconds * 0.46, 7'200.0);
    const double temperature = staticClimate.climate.temperatureC + temperatureAnomaly * 4.5 -
                               lowPressure * 1.5 + front * airMass * 2.0;
    const double warmInstability = smoothstep(2.0, 28.0, temperature);
    const double instability = clamp01(warmInstability * humidity * (0.45 + front * 0.75));
    const double storm =
        clamp01(smoothstep(0.42, 0.92, humidity) * smoothstep(0.22, 0.82, instability) *
                (0.32 + lowPressure * 0.48 + front * 0.42));
    const double coverage = clamp01((humidity - 0.34) * 1.26 + lowPressure * 0.28 + front * 0.22);
    const double condensation = smoothstep(0.62, 0.94, humidity) * smoothstep(0.56, 0.90, coverage);
    double precipitation =
        clamp01(condensation * (0.24 + front * 0.36 + lowPressure * 0.32 + storm * 0.58));
    if (precipitation < 0.035) precipitation = 0.0;

    // A world-stable backbone plus bounded regional flow and gust terms has a
    // closed-form integral. The spatial terms never multiply world age, so
    // neighboring weather cells remain phase-continuous in old saved worlds.
    // Weather consumers receive the exact low-layer derivative in blocks per
    // second, while the high layer follows its own faster analytic flow.
    constexpr double FLOW_PERIOD_SECONDS = 300.0;
    constexpr double HIGH_FLOW_PERIOD_SECONDS = 210.0;
    constexpr double GUST_PERIOD_SECONDS = 120.0;
    constexpr double GUST_DISPLACEMENT_BLOCKS = 18.0;
    constexpr double HIGH_GUST_DISPLACEMENT_BLOCKS = 16.0;
    constexpr double HIGH_SPEED_SCALE = 1.35;
    constexpr double TWO_PI = 6.28318530717958647692;
    const Vec2 baseVelocity = canonicalCloudVelocity(random);
    const Vec2 regionalVelocity = (advection - baseVelocity) * 0.20f;
    const Vec2 windDirection = normalizedOrFallback(baseVelocity);
    const Vec2 crosswind{-windDirection.y, windDirection.x};
    const double flowPhase =
        fractalNoise(random, CLOUD_FLOW_STREAM + 2U, worldX, worldZ, 12'000.0) * 3.141592653589793;
    const double flowFrequency = TWO_PI / FLOW_PERIOD_SECONDS;
    const double flowAngle = seconds * flowFrequency + flowPhase;
    const Vec2 regionalWind = regionalVelocity * static_cast<float>(std::cos(flowAngle));
    const double regionalScale = std::sin(flowAngle) / flowFrequency;
    const double gustPhase =
        fractalNoise(random, AIR_MASS_STREAM + 17U, worldX, worldZ, 12'000.0) * 3.141592653589793;
    const double gustAngle = seconds * TWO_PI / GUST_PERIOD_SECONDS + gustPhase;
    const float gustVelocity = static_cast<float>(GUST_DISPLACEMENT_BLOCKS * TWO_PI /
                                                  GUST_PERIOD_SECONDS * std::cos(gustAngle));
    const double gustOffset = GUST_DISPLACEMENT_BLOCKS * std::sin(gustAngle);
    const Vec2 analyticWind = baseVelocity + regionalWind + crosswind * gustVelocity;

    const double highFlowPhase =
        fractalNoise(random, CLOUD_FLOW_STREAM + 7U, worldX, worldZ, 16'000.0) * 3.141592653589793;
    const double highFlowFrequency = TWO_PI / HIGH_FLOW_PERIOD_SECONDS;
    const double highFlowAngle = seconds * highFlowFrequency + highFlowPhase;
    const double highRegionalScale = std::sin(highFlowAngle) / highFlowFrequency;
    const double highGustAngle =
        seconds * TWO_PI / (GUST_PERIOD_SECONDS * 0.75) + gustPhase + 1.047197551196598;
    const double highGustOffset = HIGH_GUST_DISPLACEMENT_BLOCKS * std::sin(highGustAngle);

    WeatherSample result;
    result.windBlocksPerSecond = analyticWind;
    result.cloudOffsetBlocks = {static_cast<double>(baseVelocity.x) * seconds +
                                    static_cast<double>(regionalVelocity.x) * regionalScale +
                                    static_cast<double>(crosswind.x) * gustOffset,
                                static_cast<double>(baseVelocity.y) * seconds +
                                    static_cast<double>(regionalVelocity.y) * regionalScale +
                                    static_cast<double>(crosswind.y) * gustOffset};
    result.highCloudOffsetBlocks = {
        static_cast<double>(baseVelocity.x) * HIGH_SPEED_SCALE * seconds +
            static_cast<double>(regionalVelocity.x) * 0.60 * highRegionalScale +
            static_cast<double>(crosswind.x) * highGustOffset,
        static_cast<double>(baseVelocity.y) * HIGH_SPEED_SCALE * seconds +
            static_cast<double>(regionalVelocity.y) * 0.60 * highRegionalScale +
            static_cast<double>(crosswind.y) * highGustOffset};
    result.pressureHpa = static_cast<float>(1013.25 + pressureSignal * 21.0);
    result.relativeHumidity = static_cast<float>(humidity);
    result.temperatureC = static_cast<float>(temperature);
    result.cloudCoverage = static_cast<float>(coverage);
    result.precipitationIntensity = static_cast<float>(precipitation);
    result.stormPotential = static_cast<float>(storm);
    result.fogExtinction = static_cast<float>(
        std::clamp(0.00006 + smoothstep(0.72, 1.0, humidity) * 0.0022 + precipitation * 0.0018,
                   0.00004, 0.0060));
    const double dryness = clamp01(staticClimate.climate.aridity / 2.2);
    result.aerosolDensity =
        static_cast<float>(std::clamp(0.05 + dryness * 0.24 + (1.0 - humidity) * 0.08, 0.03, 0.45));
    result.terrainHeight = static_cast<float>(staticClimate.terrainHeight);
    result.cloudType =
        classifyCloudType(result.cloudCoverage, result.stormPotential, static_cast<float>(front),
                          static_cast<float>(instability));
    const CloudLayerBounds bounds = cloudLayerBounds(result.cloudType, result.terrainHeight,
                                                     result.relativeHumidity, physicalScale);
    result.cloudBaseY = bounds.baseY;
    result.cloudTopY = bounds.topY;
    if (precipitation == 0.0) {
        result.precipitationKind = PrecipitationKind::NONE;
    } else {
        result.precipitationKind =
            temperature <= 0.0 ? PrecipitationKind::SNOW : PrecipitationKind::RAIN;
    }
    applyPreset(result, preset);
    const CloudLayerBounds resolvedBounds = cloudLayerBounds(
        result.cloudType, result.terrainHeight, result.relativeHumidity, physicalScale);
    result.cloudBaseY = resolvedBounds.baseY;
    result.cloudTopY = resolvedBounds.topY;
    if (preset != WeatherPreset::NATURAL) {
        result.cloudOffsetBlocks = {static_cast<double>(result.windBlocksPerSecond.x) * seconds,
                                    static_cast<double>(result.windBlocksPerSecond.y) * seconds};
        result.highCloudOffsetBlocks = {
            static_cast<double>(result.windBlocksPerSecond.x) * HIGH_SPEED_SCALE * seconds,
            static_cast<double>(result.windBlocksPerSecond.y) * HIGH_SPEED_SCALE * seconds};
    }
    return result;
}

std::optional<LightningEvent> lightningEventForCell(uint64_t worldSeed, int64_t stormCellX,
                                                    int64_t stormCellZ, uint64_t timeBucket,
                                                    const WeatherSample& weather) noexcept {
    if (weather.stormPotential < 0.55f || weather.cloudType != CloudType::CUMULONIMBUS) {
        return std::nullopt;
    }
    const CounterRng random(worldSeed);
    const int32_t bucketLow = static_cast<int32_t>(timeBucket);
    const uint32_t bucketHigh = static_cast<uint32_t>(timeBucket >> 32);
    const float chance = 0.025f * weather.stormPotential * weather.stormPotential *
                         weather.stormPotential * (0.35f + 0.65f * weather.precipitationIntensity);
    const double roll =
        random.uniform01(LIGHTNING_STREAM, stormCellX, bucketLow, stormCellZ, bucketHigh);
    if (roll >= static_cast<double>(chance)) return std::nullopt;

    const auto block =
        random.block(LIGHTNING_STREAM + 1, stormCellX, bucketLow, stormCellZ, bucketHigh);
    const double unitX = static_cast<double>(block[0]) / 4'294'967'296.0;
    const double unitZ = static_cast<double>(block[1]) / 4'294'967'296.0;
    const float unitHeight = static_cast<float>(block[2]) / 4'294'967'296.0f;
    LightningEvent event;
    event.id = random.u64(LIGHTNING_ID_STREAM, stormCellX, bucketLow, stormCellZ, bucketHigh);
    event.tick = WeatherSystem::lightningTickForBucket(
        timeBucket, static_cast<uint64_t>(block[3] % WeatherSystem::LIGHTNING_BUCKET_TICKS));
    event.x = (static_cast<double>(stormCellX) + 0.15 + unitX * 0.70) *
              WeatherSystem::LIGHTNING_CELL_EDGE;
    event.z = (static_cast<double>(stormCellZ) + 0.15 + unitZ * 0.70) *
              WeatherSystem::LIGHTNING_CELL_EDGE;
    event.y = weather.terrainHeight;
    event.cloudY = std::lerp(weather.cloudBaseY, weather.cloudTopY, 0.68f + unitHeight * 0.24f);
    event.intensity = 0.75f + unitHeight * 0.25f;
    return event;
}

double thunderDelaySeconds(const LightningEvent& event, double listenerX, double listenerY,
                           double listenerZ, WorldPhysicalScale physicalScale) noexcept {
    const double dx = event.x - listenerX;
    const double dy = static_cast<double>(event.y) - listenerY;
    const double dz = event.z - listenerZ;
    return worldDistanceMeters(dx, dy, dz, physicalScale) / SPEED_OF_SOUND_METERS_PER_SECOND;
}

WeatherSnapshot::WeatherSnapshot(uint64_t requestId, int64_t centerX, int64_t centerZ,
                                 uint64_t firstTick, WeatherPreset preset,
                                 std::vector<WeatherSample> first,
                                 std::vector<WeatherSample> second)
    : requestId_(requestId)
    , centerX_(centerX)
    , centerZ_(centerZ)
    , originX_(centerX - static_cast<int64_t>(GRID_EDGE / 2) * GRID_SPACING)
    , originZ_(centerZ - static_cast<int64_t>(GRID_EDGE / 2) * GRID_SPACING)
    , firstTick_(weatherTimeSliceStart(firstTick, TIME_SLICE_TICKS))
    , secondTick_(firstTick_ + TIME_SLICE_TICKS)
    , preset_(preset)
    , first_(std::move(first))
    , second_(std::move(second)) {
    if (first_.size() != GRID_SAMPLE_COUNT || second_.size() != GRID_SAMPLE_COUNT) {
        RY_LOG_FATAL("Weather snapshot grids have the wrong size");
    }
}

bool WeatherSnapshot::covers(double worldX, double worldZ) const noexcept {
    const double maximumX =
        static_cast<double>(originX_) + static_cast<double>((GRID_EDGE - 1) * GRID_SPACING);
    const double maximumZ =
        static_cast<double>(originZ_) + static_cast<double>((GRID_EDGE - 1) * GRID_SPACING);
    return worldX >= static_cast<double>(originX_) && worldX <= maximumX &&
           worldZ >= static_cast<double>(originZ_) && worldZ <= maximumZ;
}

const WeatherSample& WeatherSnapshot::gridSample(int sampleX, int sampleZ,
                                                 int timeSliceIndex) const {
    if (sampleX < 0 || sampleX >= GRID_EDGE || sampleZ < 0 || sampleZ >= GRID_EDGE ||
        timeSliceIndex < 0 || timeSliceIndex > 1) {
        RY_LOG_FATAL("Weather grid sample is outside the snapshot");
    }
    const size_t index = static_cast<size_t>(sampleZ * GRID_EDGE + sampleX);
    return timeSliceIndex == 0 ? first_[index] : second_[index];
}

std::span<const WeatherSample> WeatherSnapshot::timeSlice(int timeSliceIndex) const {
    if (timeSliceIndex == 0) return first_;
    if (timeSliceIndex == 1) return second_;
    RY_LOG_FATAL("Weather time slice is outside the snapshot");
}

WeatherSample WeatherSnapshot::sample(double worldX, double worldZ,
                                      uint64_t worldTick) const noexcept {
    const double gridX = std::clamp((worldX - static_cast<double>(originX_)) / GRID_SPACING, 0.0,
                                    static_cast<double>(GRID_EDGE - 1));
    const double gridZ = std::clamp((worldZ - static_cast<double>(originZ_)) / GRID_SPACING, 0.0,
                                    static_cast<double>(GRID_EDGE - 1));
    const int x0 = static_cast<int>(std::floor(gridX));
    const int z0 = static_cast<int>(std::floor(gridZ));
    const int x1 = std::min(x0 + 1, GRID_EDGE - 1);
    const int z1 = std::min(z0 + 1, GRID_EDGE - 1);
    const float amountX = static_cast<float>(gridX - x0);
    const float amountZ = static_cast<float>(gridZ - z0);
    float amountTime = 0.0f;
    if (worldTick >= secondTick_) {
        amountTime = 1.0f;
    } else if (worldTick > firstTick_) {
        amountTime = static_cast<float>(worldTick - firstTick_) /
                     static_cast<float>(secondTick_ - firstTick_);
    }
    const auto sliceSample = [&](const std::vector<WeatherSample>& slice) {
        const WeatherSample& s00 = slice[static_cast<size_t>(z0 * GRID_EDGE + x0)];
        const WeatherSample& s10 = slice[static_cast<size_t>(z0 * GRID_EDGE + x1)];
        const WeatherSample& s01 = slice[static_cast<size_t>(z1 * GRID_EDGE + x0)];
        const WeatherSample& s11 = slice[static_cast<size_t>(z1 * GRID_EDGE + x1)];
        return interpolateWeather(interpolateWeather(s00, s10, amountX),
                                  interpolateWeather(s01, s11, amountX), amountZ);
    };
    return interpolateWeather(sliceSample(first_), sliceSample(second_), amountTime);
}

bool WeatherSystem::Request::sameBuild(const Request& other) const noexcept {
    return centerX == other.centerX && centerZ == other.centerZ && firstTick == other.firstTick &&
           preset == other.preset;
}

WeatherSystem::WeatherSystem(const ChunkGenerator& generator)
    : WeatherSystem(
          generator.seed(),
          [&generator](int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                       std::span<worldgen::SurfaceSample> output) {
              // The generator outlives this system. The destructor joins the
              // worker before that non-owning reference can become invalid.
              generator.sampleWeatherClimateGrid(originX, originZ, spacing, sampleEdge, output);
          },
          WeatherPreset::NATURAL, worldPhysicalScale(generator.usesLearnedAuthority())) {
    if (const char* environment = std::getenv("RYCRAFT_WEATHER")) {
        if (const auto environmentPreset = weatherPresetFromString(environment)) {
            setPreset(*environmentPreset);
        }
    }
}

WeatherSystem::WeatherSystem(uint64_t worldSeed, ClimateGridSampler climateGridSampler,
                             WeatherPreset preset, WorldPhysicalScale physicalScale)
    : worldSeed_(worldSeed)
    , climateGridSampler_(std::move(climateGridSampler))
    , physicalScale_(physicalScale.valid() ? physicalScale : LEGACY_WORLD_PHYSICAL_SCALE)
    , preset_(preset) {
    if (!climateGridSampler_) {
        RY_LOG_FATAL("Weather system requires a climate grid sampler");
    }
    worker_ = std::thread([this] { workerMain(); });
}

WeatherSystem::~WeatherSystem() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        pendingRequest_.reset();
    }
    requestCondition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

uint64_t WeatherSystem::enqueueLocked(int64_t cameraX, int64_t cameraZ, uint64_t worldTick,
                                      WeatherPreset requestedPreset) {
    ++stats_.requests;
    int64_t centerX = snappedWeatherCenter(cameraX);
    int64_t centerZ = snappedWeatherCenter(cameraZ);
    const Request* reference = desiredRequest_ ? &*desiredRequest_ : nullptr;
    if (reference == nullptr && snapshot_) {
        centerX = snapshot_->centerX();
        centerZ = snapshot_->centerZ();
    } else if (reference != nullptr) {
        centerX = reference->centerX;
        centerZ = reference->centerZ;
    }
    const bool hasCenterAuthority = reference != nullptr || snapshot_ != nullptr;
    if (!hasCenterAuthority ||
        std::abs(static_cast<long double>(cameraX) - static_cast<long double>(centerX)) >=
            RECENTER_DISTANCE ||
        std::abs(static_cast<long double>(cameraZ) - static_cast<long double>(centerZ)) >=
            RECENTER_DISTANCE) {
        centerX = snappedWeatherCenter(cameraX);
        centerZ = snappedWeatherCenter(cameraZ);
    }

    Request candidate;
    candidate.centerX = centerX;
    candidate.centerZ = centerZ;
    candidate.firstTick = weatherTimeSliceStart(worldTick, WeatherSnapshot::TIME_SLICE_TICKS);
    candidate.preset = requestedPreset;
    if (desiredRequest_ && candidate.sameBuild(*desiredRequest_)) {
        ++stats_.coalescedRequests;
        return desiredRequest_->id;
    }

    candidate.id = nextRequestId_++;
    if (pendingRequest_) ++stats_.coalescedRequests;
    latestRequestId_ = candidate.id;
    desiredRequest_ = candidate;
    pendingRequest_ = candidate;
    requestCondition_.notify_one();
    return candidate.id;
}

uint64_t WeatherSystem::requestSnapshot(int64_t cameraX, int64_t cameraZ, uint64_t worldTick) {
    std::lock_guard lock(mutex_);
    return enqueueLocked(cameraX, cameraZ, worldTick, preset_);
}

void WeatherSystem::setPreset(WeatherPreset presetValue) {
    std::lock_guard lock(mutex_);
    if (preset_ == presetValue) return;
    preset_ = presetValue;
    if (desiredRequest_) {
        enqueueLocked(desiredRequest_->centerX, desiredRequest_->centerZ,
                      desiredRequest_->firstTick, preset_);
    } else if (snapshot_) {
        enqueueLocked(snapshot_->centerX(), snapshot_->centerZ(), snapshot_->firstTick(), preset_);
    }
}

WeatherPreset WeatherSystem::preset() const {
    std::lock_guard lock(mutex_);
    return preset_;
}

std::shared_ptr<const WeatherSnapshot> WeatherSystem::buildSnapshot(const Request& request) const {
    constexpr int64_t HALF_EXTENT =
        static_cast<int64_t>(WeatherSnapshot::GRID_EDGE / 2) * WeatherSnapshot::GRID_SPACING;
    const int64_t originX = request.centerX - HALF_EXTENT;
    const int64_t originZ = request.centerZ - HALF_EXTENT;
    std::vector<worldgen::SurfaceSample> staticClimate(WeatherSnapshot::GRID_SAMPLE_COUNT);
    climateGridSampler_(originX, originZ, WeatherSnapshot::GRID_SPACING, WeatherSnapshot::GRID_EDGE,
                        staticClimate);
    {
        std::lock_guard lock(mutex_);
        if (stopping_ || request.id != latestRequestId_) return nullptr;
    }

    std::vector<WeatherSample> first(WeatherSnapshot::GRID_SAMPLE_COUNT);
    std::vector<WeatherSample> second(WeatherSnapshot::GRID_SAMPLE_COUNT);
    for (int sampleZ = 0; sampleZ < WeatherSnapshot::GRID_EDGE; ++sampleZ) {
        if ((sampleZ & 7) == 0) {
            std::lock_guard lock(mutex_);
            if (stopping_ || request.id != latestRequestId_) return nullptr;
        }
        for (int sampleX = 0; sampleX < WeatherSnapshot::GRID_EDGE; ++sampleX) {
            const size_t index =
                static_cast<size_t>(sampleZ * WeatherSnapshot::GRID_EDGE + sampleX);
            const double worldX = static_cast<double>(originX + static_cast<int64_t>(sampleX) *
                                                                    WeatherSnapshot::GRID_SPACING);
            const double worldZ = static_cast<double>(originZ + static_cast<int64_t>(sampleZ) *
                                                                    WeatherSnapshot::GRID_SPACING);
            first[index] =
                deriveWeatherSample(worldSeed_, worldX, worldZ, request.firstTick,
                                    staticClimate[index], request.preset, physicalScale_);
            second[index] = deriveWeatherSample(
                worldSeed_, worldX, worldZ, request.firstTick + WeatherSnapshot::TIME_SLICE_TICKS,
                staticClimate[index], request.preset, physicalScale_);
        }
    }
    return std::make_shared<WeatherSnapshot>(request.id, request.centerX, request.centerZ,
                                             request.firstTick, request.preset, std::move(first),
                                             std::move(second));
}

void WeatherSystem::workerMain() {
    setCurrentThreadPriority(ThreadPriority::UTILITY);
    while (true) {
        Request request;
        {
            std::unique_lock lock(mutex_);
            requestCondition_.wait(lock,
                                   [this] { return stopping_ || pendingRequest_.has_value(); });
            if (stopping_) return;
            request = *pendingRequest_;
            pendingRequest_.reset();
            workerBusy_ = true;
            ++stats_.buildsStarted;
        }

        std::shared_ptr<const WeatherSnapshot> built;
        bool deferred = false;
        bool failed = false;
        try {
            built = buildSnapshot(request);
        } catch (const worldgen::learned::GenerationFailureException& exception) {
            deferred = exception.status() == worldgen::learned::AuthorityStatus::DEFERRED;
            failed = !deferred;
            if (failed) {
                RY_LOG_ERROR(std::string("Weather climate authority failed: ") + exception.what());
            }
        } catch (const std::exception& exception) {
            failed = true;
            RY_LOG_ERROR(std::string("Weather snapshot construction failed: ") + exception.what());
        } catch (...) {
            failed = true;
            RY_LOG_ERROR("Weather snapshot construction failed with an unknown exception");
        }

        {
            std::lock_guard lock(mutex_);
            workerBusy_ = false;
            if (stopping_) return;
            if (deferred) ++stats_.buildsDeferred;
            if (failed) ++stats_.buildsFailed;
            if (built && request.id == latestRequestId_) {
                snapshot_ = std::move(built);
                ++stats_.snapshotsPublished;
                snapshotCondition_.notify_all();
            } else if (request.id != latestRequestId_) {
                ++stats_.staleBuildsDiscarded;
            } else if (!built || deferred || failed) {
                // The fixed tick requests the same immutable snapshot again.
                // Releasing the desired request gives that retry a fresh ID;
                // the worker never spins or waits on inference itself.
                desiredRequest_.reset();
            }
        }
    }
}

std::shared_ptr<const WeatherSnapshot> WeatherSystem::latestSnapshot() const {
    std::lock_guard lock(mutex_);
    return snapshot_;
}

WeatherSample WeatherSystem::sample(double worldX, double worldZ, uint64_t worldTick) const {
    std::shared_ptr<const WeatherSnapshot> current;
    WeatherPreset currentPreset;
    {
        std::lock_guard lock(mutex_);
        current = snapshot_;
        currentPreset = preset_;
    }
    if (current) return current->sample(worldX, worldZ, worldTick);
    return deriveWeatherSample(worldSeed_, worldX, worldZ, worldTick, fallbackClimate(),
                               currentPreset, physicalScale_);
}

std::vector<LightningEvent> WeatherSystem::lightningEventsForBucket(const WeatherSnapshot& snapshot,
                                                                    uint64_t timeBucket) const {
    std::lock_guard lock(lightningDiscoveryMutex_);
    const auto cached =
        std::find_if(lightningDiscoveryCache_.begin(), lightningDiscoveryCache_.end(),
                     [&snapshot, timeBucket](const LightningDiscoveryCacheEntry& entry) {
                         return entry.snapshotRequestId == snapshot.requestId() &&
                                entry.timeBucket == timeBucket;
                     });
    if (cached != lightningDiscoveryCache_.end()) {
        ++lightningDiscoveryCacheHits_;
        return cached->events;
    }

    const int64_t maximumX =
        snapshot.originX() +
        static_cast<int64_t>(WeatherSnapshot::GRID_EDGE - 1) * WeatherSnapshot::GRID_SPACING;
    const int64_t maximumZ =
        snapshot.originZ() +
        static_cast<int64_t>(WeatherSnapshot::GRID_EDGE - 1) * WeatherSnapshot::GRID_SPACING;
    const int64_t minimumCellX =
        world_coord::floorDiv(snapshot.originX(), static_cast<int64_t>(LIGHTNING_CELL_EDGE));
    const int64_t maximumCellX =
        world_coord::floorDiv(maximumX, static_cast<int64_t>(LIGHTNING_CELL_EDGE));
    const int64_t minimumCellZ =
        world_coord::floorDiv(snapshot.originZ(), static_cast<int64_t>(LIGHTNING_CELL_EDGE));
    const int64_t maximumCellZ =
        world_coord::floorDiv(maximumZ, static_cast<int64_t>(LIGHTNING_CELL_EDGE));
    const uint64_t sampleTick = lightningTickForBucket(timeBucket, LIGHTNING_BUCKET_TICKS / 2);

    std::vector<LightningEvent> discovered;
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const double sampleX = (static_cast<double>(cellX) + 0.5) * LIGHTNING_CELL_EDGE;
            const double sampleZ = (static_cast<double>(cellZ) + 0.5) * LIGHTNING_CELL_EDGE;
            const WeatherSample weather = snapshot.sample(sampleX, sampleZ, sampleTick);
            const auto event = lightningEventForCell(worldSeed_, cellX, cellZ, timeBucket, weather);
            if (!event) continue;
            LightningEvent resolved = *event;
            const WeatherSample strikeWeather =
                snapshot.sample(resolved.x, resolved.z, resolved.tick);
            resolved.y = strikeWeather.terrainHeight;
            discovered.push_back(resolved);
        }
    }
    std::sort(discovered.begin(), discovered.end(),
              [](const LightningEvent& left, const LightningEvent& right) {
                  if (left.tick != right.tick) return left.tick < right.tick;
                  return left.id < right.id;
              });

    if (lightningDiscoveryCache_.size() >= MAX_LIGHTNING_DISCOVERY_CACHE_ENTRIES) {
        lightningDiscoveryCache_.erase(lightningDiscoveryCache_.begin());
    }
    lightningDiscoveryCache_.push_back({
        .snapshotRequestId = snapshot.requestId(),
        .timeBucket = timeBucket,
        .events = discovered,
    });
    ++lightningDiscoveryBuilds_;
    return discovered;
}

std::vector<LightningEvent> WeatherSystem::lightningEvents(uint64_t previousTick,
                                                           uint64_t currentTick) const {
    const std::shared_ptr<const WeatherSnapshot> current = latestSnapshot();
    if (!current || currentTick <= previousTick) return {};
    if (currentTick - previousTick > LIGHTNING_BUCKET_TICKS) {
        previousTick = currentTick - LIGHTNING_BUCKET_TICKS;
    }

    const uint64_t firstBucket = previousTick / LIGHTNING_BUCKET_TICKS;
    const uint64_t lastBucket = currentTick / LIGHTNING_BUCKET_TICKS;
    std::vector<LightningEvent> events;
    for (uint64_t bucket = firstBucket; bucket <= lastBucket; ++bucket) {
        const std::vector<LightningEvent> bucketEvents = lightningEventsForBucket(*current, bucket);
        for (const LightningEvent& event : bucketEvents) {
            if (event.tick > previousTick && event.tick <= currentTick) {
                events.push_back(event);
            }
        }
        if (bucket == std::numeric_limits<uint64_t>::max()) break;
    }
    std::sort(events.begin(), events.end(),
              [](const LightningEvent& left, const LightningEvent& right) {
                  if (left.tick != right.tick) return left.tick < right.tick;
                  return left.id < right.id;
              });
    if (events.size() > MAX_LIGHTNING_EVENTS_PER_QUERY) {
        events.resize(MAX_LIGHTNING_EVENTS_PER_QUERY);
    }
    return events;
}

WeatherSystemStats WeatherSystem::stats() const {
    WeatherSystemStats result;
    {
        std::lock_guard lock(mutex_);
        result = stats_;
        result.pendingRequests = pendingRequest_ ? 1 : 0;
        result.workerBusy = workerBusy_;
    }
    {
        std::lock_guard lock(lightningDiscoveryMutex_);
        result.lightningDiscoveryBuilds = lightningDiscoveryBuilds_;
        result.lightningDiscoveryCacheHits = lightningDiscoveryCacheHits_;
    }
    return result;
}

bool WeatherSystem::waitForSnapshot(uint64_t requestId, std::chrono::milliseconds timeout) const {
    std::unique_lock lock(mutex_);
    return snapshotCondition_.wait_for(
               lock, timeout,
               [this, requestId] {
                   return stopping_ || (snapshot_ && snapshot_->requestId() >= requestId);
               }) &&
           snapshot_ && snapshot_->requestId() >= requestId;
}
