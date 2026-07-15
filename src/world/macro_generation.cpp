#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>

namespace worldgen {
namespace {

constexpr double PLATE_SCALE = 8192.0;
constexpr double DRAINAGE_SCALE = 2048.0;
constexpr int MOISTURE_STEPS = 16;
constexpr double MOISTURE_STEP_DISTANCE = 256.0;

namespace stream {
constexpr uint64_t PLATE_POSITION = 0x1101;
constexpr uint64_t PLATE_PROPERTIES = 0x1102;
constexpr uint64_t PLATE_ROCK = 0x1103;
constexpr uint64_t HOTSPOT_POSITION = 0x1201;
constexpr uint64_t HOTSPOT_PROPERTIES = 0x1202;
constexpr uint64_t DRAINAGE_POSITION = 0x2101;
constexpr uint64_t DRAINAGE_PROPERTIES = 0x2102;
constexpr uint64_t CONTINENTAL_NOISE = 0x3101;
constexpr uint64_t WARP_X_NOISE = 0x3102;
constexpr uint64_t WARP_Z_NOISE = 0x3103;
constexpr uint64_t RELIEF_NOISE = 0x3104;
constexpr uint64_t PRESSURE_NOISE = 0x3105;
constexpr uint64_t INSOLATION_NOISE = 0x3106;
constexpr uint64_t SOIL_NOISE = 0x3107;
} // namespace stream

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double smoothstep(double edge0, double edge1, double value) {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    double t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

double bell(double value, double center, double radius) {
    if (radius <= 0.0) return value == center ? 1.0 : 0.0;
    double normalized = (value - center) / radius;
    return std::exp(-2.0 * normalized * normalized);
}

double length(Vector2d value) {
    return std::hypot(value.x, value.z);
}

Vector2d normalized(Vector2d value) {
    double magnitude = length(value);
    if (magnitude < 1.0e-12) return {1.0, 0.0};
    return {value.x / magnitude, value.z / magnitude};
}

double dot(Vector2d lhs, Vector2d rhs) {
    return lhs.x * rhs.x + lhs.z * rhs.z;
}

int64_t floorToInt64(double value) {
    constexpr double MIN_VALUE = static_cast<double>(std::numeric_limits<int64_t>::min());
    constexpr double MAX_VALUE = static_cast<double>(std::numeric_limits<int64_t>::max());
    return static_cast<int64_t>(std::floor(std::clamp(value, MIN_VALUE, MAX_VALUE)));
}

uint32_t noiseSeed(uint64_t seed, uint64_t noiseStream) {
    return CounterRng(seed).u32(noiseStream, 0, 0, 0);
}

double counterValueNoise(const CounterRng& random, uint64_t noiseStream, double x, double z,
                         double baseScale, int octaves) {
    double result = 0.0;
    double amplitude = 1.0;
    double amplitudeSum = 0.0;
    double scale = baseScale;
    for (int octave = 0; octave < octaves; ++octave) {
        const double scaledX = x / scale;
        const double scaledZ = z / scale;
        const int64_t cellX = floorToInt64(scaledX);
        const int64_t cellZ = floorToInt64(scaledZ);
        const double fractionX = scaledX - static_cast<double>(cellX);
        const double fractionZ = scaledZ - static_cast<double>(cellZ);
        const double blendX = fractionX * fractionX * (3.0 - 2.0 * fractionX);
        const double blendZ = fractionZ * fractionZ * (3.0 - 2.0 * fractionZ);
        const uint32_t counter = static_cast<uint32_t>(octave);
        const double northwest = random.signedUnit(noiseStream, cellX, 0, cellZ, counter);
        const double northeast = random.signedUnit(noiseStream, cellX + 1, 0, cellZ, counter);
        const double southwest = random.signedUnit(noiseStream, cellX, 0, cellZ + 1, counter);
        const double southeast = random.signedUnit(noiseStream, cellX + 1, 0, cellZ + 1, counter);
        const double north = northwest + (northeast - northwest) * blendX;
        const double south = southwest + (southeast - southwest) * blendX;
        result += (north + (south - north) * blendZ) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= 0.5;
        scale *= 0.5;
    }
    return amplitudeSum > 0.0 ? result / amplitudeSum : 0.0;
}

double distanceToSegment(double px, double pz, double ax, double az, double bx, double bz,
                         double& t) {
    double dx = bx - ax;
    double dz = bz - az;
    double lengthSquared = dx * dx + dz * dz;
    if (lengthSquared < 1.0e-12) {
        t = 0.0;
        return std::hypot(px - ax, pz - az);
    }
    t = std::clamp(((px - ax) * dx + (pz - az) * dz) / lengthSquared, 0.0, 1.0);
    return std::hypot(px - (ax + dx * t), pz - (az + dz * t));
}

size_t biomeIndex(Biome biome) {
    return static_cast<size_t>(biome);
}

void setScore(BiomeSuitability& suitability, Biome biome, double score) {
    suitability.scores[biomeIndex(biome)] = static_cast<float>(std::max(0.0, score));
}

double channelInfluence(const HydrologySample& hydrology, double outerWidth = 2.5) {
    if (hydrology.channelWidth <= 0.0 || !std::isfinite(hydrology.channelDistance)) return 0.0;
    const double width = std::max(1.0, hydrology.channelWidth);
    return 1.0 - smoothstep(width * 0.35, width * outerWidth, hydrology.channelDistance);
}

double lakeInfluence(const HydrologySample& hydrology) {
    if (!hydrology.lake) return 0.0;
    return smoothstep(0.0, 3.5, std::max(0.0, hydrology.lakeDepth));
}

double oceanInfluence(const HydrologySample& hydrology) {
    if (!hydrology.ocean) return 0.0;
    const double depth = std::max(0.0, static_cast<double>(SEA_LEVEL) - hydrology.surfaceElevation);
    return smoothstep(0.0, 12.0, depth);
}

double geologyInteriorInfluence(const GeologySample& geology) {
    return smoothstep(0.0, 768.0, geology.distanceToBoundary);
}

} // namespace

double biomeBlendWeight(const BiomeBlend& blend, Biome biome) noexcept {
    if (blend.primary == blend.secondary) return biome == blend.primary ? 1.0 : 0.0;
    const double secondaryWeight = clamp01(blend.transition);
    if (biome == blend.primary) return 1.0 - secondaryWeight;
    if (biome == blend.secondary) return secondaryWeight;
    return 0.0;
}

double multiscaleDitherThreshold(const CounterRng& random, uint64_t stream, int64_t x, int64_t z,
                                 uint32_t index) noexcept {
    constexpr std::array<int64_t, 4> SCALES = {64, 16, 4, 1};
    uint32_t rank = 0;
    for (uint32_t level = 0; level < SCALES.size(); ++level) {
        const int64_t cellX = world_coord::floorDiv(x, SCALES[level]);
        const int64_t cellZ = world_coord::floorDiv(z, SCALES[level]);
        const uint32_t digit =
            random.u32(stream, cellX, static_cast<int32_t>(level), cellZ, index, level) & 3U;
        rank = (rank << 2U) | digit;
    }
    return (static_cast<double>(rank) + 0.5) / 256.0;
}

double climateWaterInfluence(const HydrologySample& hydrology) noexcept {
    return std::max({oceanInfluence(hydrology), lakeInfluence(hydrology) * 0.65,
                     channelInfluence(hydrology) * 0.18});
}

struct MacroGenerationSampler::PlateSite {
    int64_t cellX = 0;
    int64_t cellZ = 0;
    uint64_t id = 0;
    double x = 0.0;
    double z = 0.0;
    CrustType crust = CrustType::CONTINENTAL;
    RockType rock = RockType::GRANITE;
    Vector2d velocity;
    double age = 0.0;
    double thickness = 0.0;
    double density = 0.0;
};

struct MacroGenerationSampler::DrainageNode {
    int64_t cellX = 0;
    int64_t cellZ = 0;
    double x = 0.0;
    double z = 0.0;
    double elevation = 0.0;
    double potential = 0.0;
    double rainfall = 0.0;
    double meander = 0.0;
    bool ocean = false;
};

MacroGenerationSampler::MacroGenerationSampler(uint64_t worldSeed)
    : cacheTag_(worldSeed)
    , random_(worldSeed)
    , basinSolver_(worldSeed)
    , continentalNoise_(noiseSeed(worldSeed, stream::CONTINENTAL_NOISE))
    , warpXNoise_(noiseSeed(worldSeed, stream::WARP_X_NOISE))
    , warpZNoise_(noiseSeed(worldSeed, stream::WARP_Z_NOISE))
    , reliefNoise_(noiseSeed(worldSeed, stream::RELIEF_NOISE))
    , soilNoise_(noiseSeed(worldSeed, stream::SOIL_NOISE)) {}

MacroGenerationSampler::PlateSite MacroGenerationSampler::plateSite(int64_t cellX,
                                                                    int64_t cellZ) const {
    struct CacheEntry {
        const MacroGenerationSampler* owner = nullptr;
        uint64_t tag = 0;
        int64_t cellX = 0;
        int64_t cellZ = 0;
        PlateSite site;
    };
    constexpr size_t CACHE_SIZE = 128;
    thread_local std::array<CacheEntry, CACHE_SIZE> cache;
    const uint64_t mixedX = static_cast<uint64_t>(cellX) * 0x9E37'79B9'7F4A'7C15ULL;
    const uint64_t mixedZ = static_cast<uint64_t>(cellZ) * 0xBF58'476D'1CE4'E5B9ULL;
    CacheEntry& entry = cache[static_cast<size_t>((mixedX ^ mixedZ) & (CACHE_SIZE - 1))];
    if (entry.owner == this && entry.tag == cacheTag_ && entry.cellX == cellX &&
        entry.cellZ == cellZ) {
        return entry.site;
    }

    PlateSite site;
    site.cellX = cellX;
    site.cellZ = cellZ;
    site.id = random_.u64(stream::PLATE_PROPERTIES, cellX, 0, cellZ);
    double jitterX = 0.12 + random_.uniform01(stream::PLATE_POSITION, cellX, 0, cellZ, 0) * 0.76;
    double jitterZ = 0.12 + random_.uniform01(stream::PLATE_POSITION, cellX, 0, cellZ, 1) * 0.76;
    site.x = (static_cast<double>(cellX) + jitterX) * PLATE_SCALE;
    site.z = (static_cast<double>(cellZ) + jitterZ) * PLATE_SCALE;

    site.crust = random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 1) < 0.57
                     ? CrustType::CONTINENTAL
                     : CrustType::OCEANIC;
    site.age = random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 2);
    site.thickness =
        site.crust == CrustType::CONTINENTAL
            ? 30.0 + 18.0 * random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 3)
            : 6.0 + 7.0 * random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 3);
    site.density =
        site.crust == CrustType::CONTINENTAL ? 2.62 + site.age * 0.18 : 2.90 + site.age * 0.20;

    double angle =
        2.0 * std::numbers::pi * random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 4);
    double speed = 0.25 + random_.uniform01(stream::PLATE_PROPERTIES, cellX, 0, cellZ, 5);
    site.velocity = {std::cos(angle) * speed, std::sin(angle) * speed};

    double rockChoice = random_.uniform01(stream::PLATE_ROCK, cellX, 0, cellZ);
    if (site.crust == CrustType::OCEANIC) {
        site.rock = rockChoice < 0.82 ? RockType::BASALT : RockType::VOLCANIC;
    } else if (rockChoice < 0.42) {
        site.rock = RockType::GRANITE;
    } else if (rockChoice < 0.67) {
        site.rock = RockType::LIMESTONE;
    } else if (rockChoice < 0.90) {
        site.rock = RockType::SANDSTONE;
    } else {
        site.rock = RockType::VOLCANIC;
    }
    entry = {this, cacheTag_, cellX, cellZ, site};
    return site;
}

Vector2d MacroGenerationSampler::plateVelocityAt(double x, double z) const {
    const double warpedX = x + warpXNoise_.octave2D(x / 24000.0, z / 24000.0, 3) * 900.0;
    const double warpedZ = z + warpZNoise_.octave2D(x / 24000.0, z / 24000.0, 3) * 900.0;
    const int64_t baseCellX = floorToInt64(warpedX / PLATE_SCALE);
    const int64_t baseCellZ = floorToInt64(warpedZ / PLATE_SCALE);

    Vector2d velocity{1.0, 0.0};
    double nearestDistanceSquared = std::numeric_limits<double>::max();
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            const PlateSite candidate = plateSite(baseCellX + dx, baseCellZ + dz);
            const double offsetX = warpedX - candidate.x;
            const double offsetZ = warpedZ - candidate.z;
            const double distanceSquared = offsetX * offsetX + offsetZ * offsetZ;
            if (distanceSquared >= nearestDistanceSquared) continue;
            nearestDistanceSquared = distanceSquared;
            velocity = candidate.velocity;
        }
    }
    return velocity;
}

HotspotChainPrimitive MacroGenerationSampler::hotspotChain(int64_t cellX, int64_t cellZ) const {
    struct CacheEntry {
        const MacroGenerationSampler* owner = nullptr;
        uint64_t tag = 0;
        int64_t cellX = 0;
        int64_t cellZ = 0;
        HotspotChainPrimitive primitive;
    };
    constexpr size_t CACHE_SIZE = 64;
    thread_local std::array<CacheEntry, CACHE_SIZE> cache;
    const uint64_t mixedX = static_cast<uint64_t>(cellX) * 0x9E37'79B9'7F4A'7C15ULL;
    const uint64_t mixedZ = static_cast<uint64_t>(cellZ) * 0xBF58'476D'1CE4'E5B9ULL;
    CacheEntry& entry = cache[static_cast<size_t>((mixedX ^ mixedZ) & (CACHE_SIZE - 1))];
    if (entry.owner == this && entry.tag == cacheTag_ && entry.cellX == cellX &&
        entry.cellZ == cellZ) {
        return entry.primitive;
    }

    HotspotChainPrimitive result;
    if (random_.uniform01(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ) < 0.14) {
        result.active = true;
        result.sourceX = (static_cast<double>(cellX) + 0.1 +
                          random_.uniform01(stream::HOTSPOT_POSITION, cellX, 0, cellZ, 0) * 0.8) *
                         HOTSPOT_LATTICE_EDGE;
        result.sourceZ = (static_cast<double>(cellZ) + 0.1 +
                          random_.uniform01(stream::HOTSPOT_POSITION, cellX, 0, cellZ, 1) * 0.8) *
                         HOTSPOT_LATTICE_EDGE;
        result.length =
            3200.0 + random_.uniform01(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ, 1) * 6200.0;
        result.sourcePlateVelocity = plateVelocityAt(result.sourceX, result.sourceZ);
        result.direction =
            normalized({-result.sourcePlateVelocity.x, -result.sourcePlateVelocity.z});
    }
    entry = {this, cacheTag_, cellX, cellZ, result};
    return result;
}

GeologySample MacroGenerationSampler::sampleGeology(double x, double z) const {
    double warpedX = x + warpXNoise_.octave2D(x / 24000.0, z / 24000.0, 3) * 900.0;
    double warpedZ = z + warpZNoise_.octave2D(x / 24000.0, z / 24000.0, 3) * 900.0;
    int64_t baseCellX = floorToInt64(warpedX / PLATE_SCALE);
    int64_t baseCellZ = floorToInt64(warpedZ / PLATE_SCALE);

    std::array<PlateSite, 9> candidates;
    std::array<double, 9> candidateDistanceSquared;
    size_t candidateCount = 0;
    PlateSite nearest;
    double nearestDistanceSquared = std::numeric_limits<double>::max();
    double secondDistanceSquared = std::numeric_limits<double>::max();
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            const PlateSite candidate = plateSite(baseCellX + dx, baseCellZ + dz);
            const double offsetX = warpedX - candidate.x;
            const double offsetZ = warpedZ - candidate.z;
            const double distanceSquared = offsetX * offsetX + offsetZ * offsetZ;
            candidates[candidateCount] = candidate;
            candidateDistanceSquared[candidateCount] = distanceSquared;
            ++candidateCount;
            if (distanceSquared < nearestDistanceSquared) {
                secondDistanceSquared = nearestDistanceSquared;
                nearest = candidate;
                nearestDistanceSquared = distanceSquared;
            } else if (distanceSquared < secondDistanceSquared) {
                secondDistanceSquared = distanceSquared;
            }
        }
    }

    GeologySample result;
    result.plateId = nearest.id;
    result.crust = nearest.crust;
    result.rock = nearest.rock;
    result.plateVelocity = nearest.velocity;
    result.crustAge = nearest.age;
    result.crustThickness = nearest.thickness;
    result.crustDensity = nearest.density;
    result.distanceToBoundary =
        std::max(0.0, (std::sqrt(secondDistanceSquared) - std::sqrt(nearestDistanceSquared)) * 0.5);

    std::array<double, 9> candidateDistances;
    std::array<size_t, 9> nearbyCandidates;
    size_t nearbyCandidateCount = 0;
    const double nearestDistance = std::sqrt(nearestDistanceSquared);
    const double nearbyDistanceLimitSquared =
        (nearestDistance + 2700.0) * (nearestDistance + 2700.0);
    double oppositeCrustDistanceSquared = std::numeric_limits<double>::max();
    for (size_t index = 0; index < candidateCount; ++index) {
        const PlateSite& candidate = candidates[index];
        if (candidate.crust != nearest.crust) {
            oppositeCrustDistanceSquared =
                std::min(oppositeCrustDistanceSquared, candidateDistanceSquared[index]);
        }
        if (candidateDistanceSquared[index] <= nearbyDistanceLimitSquared) {
            candidateDistances[index] = std::sqrt(candidateDistanceSquared[index]);
            nearbyCandidates[nearbyCandidateCount++] = index;
        }
    }

    double arcActivity = 0.0;
    for (size_t firstNearby = 0; firstNearby < nearbyCandidateCount; ++firstNearby) {
        const size_t firstIndex = nearbyCandidates[firstNearby];
        const PlateSite& first = candidates[firstIndex];
        for (size_t secondNearby = firstNearby + 1; secondNearby < nearbyCandidateCount;
             ++secondNearby) {
            const size_t secondIndex = nearbyCandidates[secondNearby];
            const PlateSite& second = candidates[secondIndex];
            const double distanceToCandidateBoundary =
                std::abs(candidateDistances[firstIndex] - candidateDistances[secondIndex]) * 0.5;
            const double pairExcess =
                std::max(candidateDistances[firstIndex], candidateDistances[secondIndex]) -
                nearestDistance;
            const double locality = 1.0 - smoothstep(900.0, 2700.0, pairExcess);
            const double boundaryInfluence =
                (1.0 - smoothstep(180.0, 1350.0, distanceToCandidateBoundary)) * locality;
            if (boundaryInfluence <= 0.0) continue;

            const Vector2d boundaryNormal = normalized({second.x - first.x, second.z - first.z});
            const Vector2d relativeVelocity = {first.velocity.x - second.velocity.x,
                                               first.velocity.z - second.velocity.z};
            const double closingMotion = dot(relativeVelocity, boundaryNormal);
            const double tangentialMotion =
                std::abs(dot(relativeVelocity, {-boundaryNormal.z, boundaryNormal.x}));
            if (tangentialMotion > std::abs(closingMotion) * 1.35) {
                result.faultStrength = std::max(
                    result.faultStrength, boundaryInfluence * clamp01(tangentialMotion / 1.5));
            } else if (closingMotion > 0.08) {
                const double collision =
                    first.crust == CrustType::CONTINENTAL && second.crust == CrustType::CONTINENTAL
                        ? 1.0
                        : 0.72;
                const double uplift = boundaryInfluence * collision * clamp01(closingMotion / 1.4);
                result.uplift = std::max(result.uplift, uplift);
                if (first.crust == CrustType::OCEANIC || second.crust == CrustType::OCEANIC) {
                    arcActivity = std::max(arcActivity, uplift * 0.75);
                }
            } else if (closingMotion < -0.08) {
                result.rift =
                    std::max(result.rift, boundaryInfluence * clamp01(-closingMotion / 1.4));
            } else {
                result.faultStrength = std::max(result.faultStrength, boundaryInfluence * 0.35);
            }
        }
    }

    const double crustInterior =
        oppositeCrustDistanceSquared < std::numeric_limits<double>::max()
            ? smoothstep(0.0, 1350.0,
                         std::max(0.0, (std::sqrt(oppositeCrustDistanceSquared) -
                                        std::sqrt(nearestDistanceSquared)) *
                                           0.5))
            : 1.0;
    result.continentalFraction = nearest.crust == CrustType::CONTINENTAL
                                     ? 0.5 + crustInterior * 0.5
                                     : 0.5 - crustInterior * 0.5;

    result.boundary = PlateBoundary::NONE;
    double dominantBoundaryStrength = 0.0;
    const auto retainDominantBoundary = [&](PlateBoundary boundary, double strength) {
        if (strength <= dominantBoundaryStrength) return;
        dominantBoundaryStrength = strength;
        result.boundary = boundary;
    };
    retainDominantBoundary(PlateBoundary::CONVERGENT, result.uplift);
    retainDominantBoundary(PlateBoundary::DIVERGENT, result.rift);
    retainDominantBoundary(PlateBoundary::TRANSFORM, result.faultStrength);

    int64_t hotspotCellX = floorToInt64(x / HOTSPOT_LATTICE_EDGE);
    int64_t hotspotCellZ = floorToInt64(z / HOTSPOT_LATTICE_EDGE);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            const HotspotChainPrimitive chain = hotspotChain(hotspotCellX + dx, hotspotCellZ + dz);
            if (!chain.active) continue;
            double chainEndX = chain.sourceX + chain.direction.x * chain.length;
            double chainEndZ = chain.sourceZ + chain.direction.z * chain.length;
            double along = 0.0;
            double distance =
                distanceToSegment(x, z, chain.sourceX, chain.sourceZ, chainEndX, chainEndZ, along);
            double radius = 420.0 + 560.0 * (1.0 - along);
            double influence =
                (1.0 - smoothstep(radius * 0.35, radius, distance)) * (1.0 - along * 0.62);
            result.hotspotInfluence = std::max(result.hotspotInfluence, influence);
        }
    }

    result.volcanicActivity = clamp01(std::max(result.hotspotInfluence, arcActivity));
    if (result.volcanicActivity > 0.52) result.rock = RockType::VOLCANIC;
    return result;
}

double MacroGenerationSampler::preliminaryElevation(double x, double z) const {
    const GeologySample geology = sampleGeology(x, z);
    const double continentalness = continentalNoise_.octave2D(x / 6200.0, z / 6200.0, 5);
    const double detail = reliefNoise_.octave2D(x / 760.0, z / 760.0, 4);
    const double ridge = clamp01(reliefNoise_.ridged2D(x / 1650.0, z / 1650.0, 4));
    const double tectonicSignal =
        std::max({geology.uplift, geology.rift, geology.faultStrength, geology.hotspotInfluence});
    const double broadRidge =
        tectonicSignal > 0.0 ? clamp01(reliefNoise_.ridged2D(x / 2800.0, z / 2800.0, 3)) : ridge;
    const double foldedPeak = smoothstep(0.38, 0.86, broadRidge);

    const double oceanicBase = 47.0 + continentalness * 15.0;
    const double continentalBase = 73.0 + continentalness * 24.0;
    const double base = oceanicBase + (continentalBase - oceanicBase) * geology.continentalFraction;
    const double continentalWeight = clamp01(geology.continentalFraction);
    const double oceanicWeight = 1.0 - continentalWeight;
    const double continentalUplift = 72.0 + broadRidge * 46.0 + foldedPeak * 148.0;
    const double oceanicUplift = 12.0 + broadRidge * 18.0;
    const double upliftRelief =
        geology.uplift * (oceanicUplift * oceanicWeight + continentalUplift * continentalWeight);
    const double subductionTrench = geology.uplift * oceanicWeight * (45.0 + broadRidge * 32.0);
    const double divergentRelief = geology.rift * (oceanicWeight * (30.0 + broadRidge * 44.0) -
                                                   continentalWeight * (28.0 + broadRidge * 16.0));
    const double faultRelief = geology.faultStrength * (detail * 34.0 + (broadRidge - 0.35) * 30.0);
    const double volcanicRelief =
        geology.hotspotInfluence * (68.0 + broadRidge * 36.0 + foldedPeak * 64.0);
    const double boundaryStrength =
        std::max({geology.uplift, geology.faultStrength * 0.72, geology.rift * 0.55});
    const double foldScale = oceanicWeight * 10.0 + continentalWeight * 18.0;
    const double boundaryFolds = boundaryStrength * (broadRidge - 0.32) * foldScale;
    double elevation = base + detail * 10.0 + ridge * 7.0 + upliftRelief - subductionTrench +
                       divergentRelief + faultRelief + volcanicRelief + boundaryFolds;

    // Unit slope at both knees keeps the bounded mapping smooth. The upper
    // asymptote preserves headroom for emitted volcanoes without flattening
    // ordinary mountain crests into a visible plateau.
    if (elevation > 300.0) elevation = 300.0 + std::tanh((elevation - 300.0) / 180.0) * 180.0;
    if (elevation < -80.0) elevation = -80.0 + std::tanh((elevation + 80.0) / 32.0) * 32.0;
    return std::clamp(elevation, -112.0, 480.0);
}

double MacroGenerationSampler::provisionalRainfall(double x, double z, double elevation) const {
    double pressure = counterValueNoise(random_, stream::PRESSURE_NOISE, x, z, 5400.0, 3);
    double maritime = 1.0 - smoothstep(SEA_LEVEL - 4.0, SEA_LEVEL + 80.0, elevation);
    double uplift = clamp01((elevation - SEA_LEVEL) / 180.0);
    return std::clamp(520.0 + pressure * 330.0 + maritime * 760.0 + uplift * 210.0, 80.0, 2400.0);
}

MacroGenerationSampler::DrainageNode MacroGenerationSampler::drainageNode(int64_t cellX,
                                                                          int64_t cellZ) const {
    DrainageNode node;
    node.cellX = cellX;
    node.cellZ = cellZ;
    double jitterX = 0.18 + random_.uniform01(stream::DRAINAGE_POSITION, cellX, 0, cellZ, 0) * 0.64;
    double jitterZ = 0.18 + random_.uniform01(stream::DRAINAGE_POSITION, cellX, 0, cellZ, 1) * 0.64;
    node.x = (static_cast<double>(cellX) + jitterX) * DRAINAGE_SCALE;
    node.z = (static_cast<double>(cellZ) + jitterZ) * DRAINAGE_SCALE;
    node.elevation = preliminaryElevation(node.x, node.z);
    node.potential = node.elevation;
    node.rainfall = provisionalRainfall(node.x, node.z, node.elevation);
    node.meander = random_.signedUnit(stream::DRAINAGE_PROPERTIES, cellX, 0, cellZ);
    node.ocean = node.elevation < SEA_LEVEL - 2.0;
    return node;
}

MacroGenerationSampler::DrainageNode
MacroGenerationSampler::downstreamNode(const DrainageNode& node) const {
    if (node.ocean) return node;
    DrainageNode best = node;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dz == 0) continue;
            DrainageNode candidate = drainageNode(node.cellX + dx, node.cellZ + dz);
            bool lower = candidate.potential < best.potential - 0.01;
            bool tie = std::abs(candidate.potential - best.potential) <= 0.01 &&
                       (candidate.cellX < best.cellX ||
                        (candidate.cellX == best.cellX && candidate.cellZ < best.cellZ));
            if (lower || (best.cellX == node.cellX && best.cellZ == node.cellZ && tie)) {
                best = candidate;
            }
        }
    }
    return best;
}

HydrologySample MacroGenerationSampler::sampleHydrology(double x, double z) const {
    const BasinSample basin = basinSolver_.sample(
        x, z,
        [this](double sampleX, double sampleZ) { return preliminaryElevation(sampleX, sampleZ); },
        [this](double sampleX, double sampleZ, double elevation) {
            return provisionalRainfall(sampleX, sampleZ, elevation);
        },
        [this](double sampleX, double sampleZ) {
            switch (sampleGeology(sampleX, sampleZ).rock) {
                case RockType::GRANITE:
                    return 0.92;
                case RockType::BASALT:
                    return 1.12;
                case RockType::LIMESTONE:
                    return 0.56;
                case RockType::SANDSTONE:
                    return 0.42;
                case RockType::VOLCANIC:
                    return 1.20;
            }
            return 0.90;
        });
    if (!basin.valid) return sampleHydrologyFallback(x, z);

    HydrologySample result;
    result.flowDirection = {basin.flowX, basin.flowZ};
    result.surfaceElevation = basin.surfaceElevation;
    result.waterSurface = basin.waterSurface;
    result.discharge = basin.discharge;
    result.sediment = basin.sediment;
    result.channelDistance = basin.channelDistance;
    result.channelWidth = basin.channelWidth;
    result.channelDepth = basin.channelDepth;
    result.channelGradient = basin.channelGradient;
    result.erosionDepth = basin.erosionDepth;
    result.lakeDepth = basin.lakeDepth;
    result.lakeShoreDistance = basin.lakeShoreDistance;
    result.shoreWaterSurface = basin.shoreWaterSurface;
    result.lakeBankTarget = basin.lakeBankTarget;
    result.lakeBankInfluence = basin.lakeBankInfluence;
    result.waterfallTop = basin.waterfallTop;
    result.waterfallBottom = basin.waterfallBottom;
    result.waterfallWidth = basin.waterfallWidth;
    result.streamOrder = basin.streamOrder;
    result.distributaryCount = basin.distributaryCount;
    result.ocean = basin.ocean;
    result.river = basin.river;
    result.lake = basin.lake;
    result.lakeBank = basin.lakeBank;
    result.endorheic = basin.endorheic;
    result.waterfall = basin.waterfall;
    result.waterfallAnchor = basin.waterfallAnchor;
    result.delta = basin.delta;
    return result;
}

HydrologySample MacroGenerationSampler::sampleHydrologyFallback(double x, double z) const {
    HydrologySample result;
    double baseElevation = preliminaryElevation(x, z);
    result.surfaceElevation = baseElevation;
    result.waterSurface = SEA_LEVEL;
    result.ocean = baseElevation < SEA_LEVEL;

    int64_t baseCellX = floorToInt64(x / DRAINAGE_SCALE);
    int64_t baseCellZ = floorToInt64(z / DRAINAGE_SCALE);
    DrainageNode closestStart;
    DrainageNode closestEnd;
    double closestDistance = std::numeric_limits<double>::max();
    double closestAlong = 0.0;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            DrainageNode start = drainageNode(baseCellX + dx, baseCellZ + dz);
            DrainageNode end = downstreamNode(start);
            if (start.cellX == end.cellX && start.cellZ == end.cellZ) continue;

            Vector2d segment = {end.x - start.x, end.z - start.z};
            Vector2d perpendicular = normalized({-segment.z, segment.x});
            double meanderOffset = start.meander * std::min(260.0, length(segment) * 0.12);
            double previousX = start.x;
            double previousZ = start.z;
            for (int piece = 1; piece <= 6; ++piece) {
                double t1 = static_cast<double>(piece) / 6.0;
                double wave = std::sin(t1 * std::numbers::pi) * meanderOffset;
                double nextX = start.x + segment.x * t1 + perpendicular.x * wave;
                double nextZ = start.z + segment.z * t1 + perpendicular.z * wave;
                double localAlong = 0.0;
                double distance =
                    distanceToSegment(x, z, previousX, previousZ, nextX, nextZ, localAlong);
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestAlong = (static_cast<double>(piece - 1) + localAlong) / 6.0;
                    closestStart = start;
                    closestEnd = end;
                }
                previousX = nextX;
                previousZ = nextZ;
            }
        }
    }

    result.channelDistance = closestDistance;
    if (closestDistance < std::numeric_limits<double>::max()) {
        int upstreamCount = 0;
        double upstreamRain = closestStart.rainfall;
        for (int dz = -2; dz <= 2; ++dz) {
            for (int dx = -2; dx <= 2; ++dx) {
                DrainageNode candidate =
                    drainageNode(closestStart.cellX + dx, closestStart.cellZ + dz);
                for (int step = 0; step < 2; ++step) {
                    DrainageNode next = downstreamNode(candidate);
                    if (next.cellX == closestStart.cellX && next.cellZ == closestStart.cellZ) {
                        ++upstreamCount;
                        upstreamRain += candidate.rainfall;
                        break;
                    }
                    if (next.cellX == candidate.cellX && next.cellZ == candidate.cellZ) break;
                    candidate = next;
                }
            }
        }

        result.streamOrder = static_cast<uint8_t>(
            std::clamp(1 + static_cast<int>(std::floor(std::log2(upstreamCount + 1.0))), 1, 6));
        result.discharge = (upstreamRain / 1000.0) * (18.0 + upstreamCount * 7.0);
        result.channelWidth = std::clamp(
            6.0 + result.streamOrder * 2.4 + std::sqrt(result.discharge) * 0.40, 7.0, 42.0);
        double segmentDrop = std::max(0.0, closestStart.elevation - closestEnd.elevation);
        double segmentLength =
            std::max(1.0, std::hypot(closestEnd.x - closestStart.x, closestEnd.z - closestStart.z));
        double gradient = segmentDrop / segmentLength;
        result.channelGradient = gradient;
        result.channelDepth = std::clamp(
            1.2 + result.streamOrder * 0.8 + std::sqrt(result.discharge) * 0.10, 2.0, 14.0);
        result.sediment = result.discharge * (0.025 + gradient * 5.0);
        double knickpoint = 0.30 + random_.uniform01(stream::DRAINAGE_PROPERTIES,
                                                     closestStart.cellX, 0, closestStart.cellZ, 3) *
                                       0.40;
        bool hasKnickpoint = result.streamOrder >= 2 && segmentDrop > 9.0 &&
                             random_.uniform01(stream::DRAINAGE_PROPERTIES, closestStart.cellX, 0,
                                               closestStart.cellZ, 4) < 0.55;
        double knickpointHalfWidth = std::max(6.0, result.channelWidth * 0.55) / segmentLength;
        if (hasKnickpoint) {
            double gradualDrop = segmentDrop * 0.45 * closestAlong;
            double suddenDrop = segmentDrop * 0.55 *
                                smoothstep(knickpoint - knickpointHalfWidth,
                                           knickpoint + knickpointHalfWidth, closestAlong);
            result.waterSurface = closestStart.elevation - 1.0 - gradualDrop - suddenDrop;
        } else {
            result.waterSurface = closestStart.elevation - 1.0 - segmentDrop * closestAlong;
        }

        Vector2d direction = {closestEnd.x - closestStart.x, closestEnd.z - closestStart.z};
        result.flowDirection = normalized(direction);
        double floodplainWidth = result.channelWidth * (2.2 + result.streamOrder * 0.35);
        double channelMask =
            1.0 - smoothstep(result.channelWidth * 0.45, floodplainWidth, closestDistance);
        double targetFloor = result.waterSurface - result.channelDepth;
        double incisionNeeded = std::max(0.0, baseElevation - targetFloor);
        result.erosionDepth = incisionNeeded * channelMask;
        result.river = closestDistance <= result.channelWidth * 0.55 && !result.ocean;
        result.waterfall = result.river && hasKnickpoint &&
                           std::abs(closestAlong - knickpoint) <= knickpointHalfWidth;
        result.delta = result.river && closestEnd.ocean && closestAlong > 0.60 &&
                       result.streamOrder >= 2 && gradient < 0.025;
        if (result.delta) {
            result.distributaryCount = static_cast<uint8_t>(
                2 + random_.uniformInt(stream::DRAINAGE_PROPERTIES, closestStart.cellX, 0,
                                       closestStart.cellZ, 5, 0, 2));
        }
    }

    // Local minima become bounded lakes. Searching neighboring catchments
    // makes the lake edge agree when a sample crosses a cell boundary.
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            DrainageNode node = drainageNode(baseCellX + dx, baseCellZ + dz);
            DrainageNode downstream = downstreamNode(node);
            if (node.ocean || downstream.cellX != node.cellX || downstream.cellZ != node.cellZ) {
                continue;
            }
            double radius = 100.0 + random_.uniform01(stream::DRAINAGE_PROPERTIES, node.cellX, 0,
                                                      node.cellZ, 2) *
                                        240.0;
            double distance = std::hypot(x - node.x, z - node.z);
            double lakeMask = 1.0 - smoothstep(radius * 0.72, radius, distance);
            if (lakeMask <= 0.0) continue;
            double depth = (4.0 + radius / 45.0) * lakeMask;
            if (depth > result.lakeDepth) {
                result.lake = true;
                result.endorheic = true;
                result.lakeDepth = depth;
                result.waterSurface = node.elevation + 1.5;
                double lakeFloor = result.waterSurface - depth;
                result.erosionDepth =
                    std::max(result.erosionDepth, std::max(0.0, baseElevation - lakeFloor));
                result.river = false;
                result.waterfall = false;
                result.delta = false;
            }
        }
    }

    result.surfaceElevation = baseElevation - result.erosionDepth;
    if (result.ocean) result.waterSurface = SEA_LEVEL;
    return result;
}

ClimateFields MacroGenerationSampler::sampleClimate(double x, double z,
                                                    double terrainHeight) const {
    ClimateFields result;
    constexpr double PRESSURE_DELTA = 96.0;
    double pressureEast =
        counterValueNoise(random_, stream::PRESSURE_NOISE, x + PRESSURE_DELTA, z, 6200.0, 3);
    double pressureWest =
        counterValueNoise(random_, stream::PRESSURE_NOISE, x - PRESSURE_DELTA, z, 6200.0, 3);
    double pressureNorth =
        counterValueNoise(random_, stream::PRESSURE_NOISE, x, z + PRESSURE_DELTA, 6200.0, 3);
    double pressureSouth =
        counterValueNoise(random_, stream::PRESSURE_NOISE, x, z - PRESSURE_DELTA, 6200.0, 3);
    Vector2d gradient = {(pressureEast - pressureWest) / (2.0 * PRESSURE_DELTA),
                         (pressureNorth - pressureSouth) / (2.0 * PRESSURE_DELTA)};
    Vector2d rotational = {-gradient.z, gradient.x};
    Vector2d windDirection = normalized(
        {-gradient.x + rotational.x * 0.62 + 0.00008, -gradient.z + rotational.z * 0.62 + 0.00003});
    double windSpeed = 0.45 + clamp01(length(gradient) * 1800.0) * 0.55;
    result.wind = {windDirection.x * windSpeed, windDirection.z * windSpeed};

    std::array<double, MOISTURE_STEPS + 1> elevations{};
    std::array<double, MOISTURE_STEPS + 1> waterRecharge{};
    double waterSteps = 0.0;
    for (int i = 0; i <= MOISTURE_STEPS; ++i) {
        double distance = static_cast<double>(MOISTURE_STEPS - i) * MOISTURE_STEP_DISTANCE;
        double sampleX = x - windDirection.x * distance;
        double sampleZ = z - windDirection.z * distance;
        const HydrologySample hydrology = sampleHydrology(sampleX, sampleZ);
        elevations[static_cast<size_t>(i)] = hydrology.surfaceElevation;
        const double recharge = climateWaterInfluence(hydrology);
        waterRecharge[static_cast<size_t>(i)] = recharge;
        waterSteps += recharge;
    }

    double moisture = 0.22;
    double precipitation = 0.0;
    for (int i = 0; i <= MOISTURE_STEPS; ++i) {
        double elevation = elevations[static_cast<size_t>(i)];
        moisture += (1.0 - moisture) * 0.34 * waterRecharge[static_cast<size_t>(i)];
        double rise =
            i == 0 ? 0.0 : std::max(0.0, elevation - elevations[static_cast<size_t>(i - 1)]);
        double descent =
            i == 0 ? 0.0 : std::max(0.0, elevations[static_cast<size_t>(i - 1)] - elevation);
        double stepRain = moisture * (0.010 + std::min(0.32, rise * 0.010));
        precipitation += stepRain;
        moisture = std::max(0.02, moisture - stepRain);
        moisture *= std::max(0.72, 1.0 - descent * 0.0025);
    }

    double localPressure = counterValueNoise(random_, stream::PRESSURE_NOISE, x, z, 6200.0, 3);
    result.annualPrecipitationMm =
        std::clamp(90.0 + precipitation * 3900.0 + (localPressure + 1.0) * 130.0, 60.0, 3600.0);
    result.relativeHumidity = clamp01(moisture + result.annualPrecipitationMm / 5200.0);

    double insolation = counterValueNoise(random_, stream::INSOLATION_NOISE, x, z, 8800.0, 4);
    double maritime = static_cast<double>(waterSteps) / (MOISTURE_STEPS + 1.0);
    double continentalTemperature = 15.0 + insolation * 26.0;
    double moderatedTemperature =
        continentalTemperature * (1.0 - maritime * 0.42) + 13.0 * maritime * 0.42;
    double lapseCooling = std::max(0.0, terrainHeight - SEA_LEVEL) * 8.0 * 0.0065;
    result.temperatureC = moderatedTemperature - lapseCooling;
    result.potentialEvapotranspirationMm = std::clamp(
        300.0 + std::max(-8.0, result.temperatureC) * 31.0 + windSpeed * 170.0, 120.0, 1800.0);
    result.aridity =
        result.potentialEvapotranspirationMm / std::max(1.0, result.annualPrecipitationMm);
    return result;
}

double MacroGenerationSampler::terrainSlope(double x, double z) const {
    constexpr double DELTA = 16.0;
    double east = sampleHydrology(x + DELTA, z).surfaceElevation;
    double west = sampleHydrology(x - DELTA, z).surfaceElevation;
    double north = sampleHydrology(x, z + DELTA).surfaceElevation;
    double south = sampleHydrology(x, z - DELTA).surfaceElevation;
    return std::hypot((east - west) / (2.0 * DELTA), (north - south) / (2.0 * DELTA));
}

SoilSample MacroGenerationSampler::sampleSoil(double x, double z, const GeologySample& geology,
                                              const HydrologySample& hydrology,
                                              const ClimateFields& climate) const {
    SoilSample result;
    const double textureNoise = soilNoise_.octave2D(x / 540.0, z / 540.0, 3);
    const double rawRockDrainage = geology.rock == RockType::LIMESTONE ? 0.82
                                   : geology.rock == RockType::BASALT  ? 0.62
                                                                       : 0.48;
    const double geologyInterior = geologyInteriorInfluence(geology);
    const double rockDrainage = 0.55 + (rawRockDrainage - 0.55) * geologyInterior;
    const double lakeWetness = lakeInfluence(hydrology);
    const double channelWetness = channelInfluence(hydrology);
    result.drainage =
        clamp01(rockDrainage + textureNoise * 0.18 - lakeWetness * 0.45 - channelWetness * 0.10);
    const double waterContribution =
        std::max({lakeWetness * 0.38, channelWetness * 0.34, oceanInfluence(hydrology) * 0.12});
    result.moisture = clamp01(climate.annualPrecipitationMm / 2100.0 - climate.aridity * 0.19 +
                              waterContribution - result.drainage * 0.12);
    const double rawMineralContribution =
        geology.rock == RockType::VOLCANIC || geology.rock == RockType::BASALT ? 0.22
        : geology.rock == RockType::LIMESTONE                                  ? 0.12
                                                                               : 0.0;
    const double mineralContribution = 0.08 + (rawMineralContribution - 0.08) * geologyInterior;
    const double sedimentFertility = clamp01(std::log1p(std::max(0.0, hydrology.sediment)) / 8.0);
    const double alluvialContribution = channelWetness * (0.16 + sedimentFertility * 0.08);
    result.fertility = clamp01(0.24 + result.moisture * 0.48 + mineralContribution +
                               alluvialContribution - climate.aridity * 0.10);
    const double waterDepth = hydrology.lake ? hydrology.lakeDepth : 0.0;
    result.waterTable = hydrology.waterSurface - waterDepth -
                        (4.0 + result.drainage * 22.0) * (1.0 - result.moisture * 0.65);
    return result;
}

BiomeSuitability MacroGenerationSampler::biomeSuitability(
    const GeologySample& geology, const HydrologySample& hydrology, const ClimateFields& climate,
    const SoilSample& soil, double terrainHeight, double slope) const {
    BiomeSuitability result;
    double temperature = climate.temperatureC;
    double rain = climate.annualPrecipitationMm;
    double dry = clamp01((climate.aridity - 0.65) / 1.4);
    double wet = clamp01(rain / 2200.0);
    double high = smoothstep(105.0, 185.0, terrainHeight);
    double steep = smoothstep(0.55, 1.45, slope);

    const double oceanDepth =
        std::max(0.0, static_cast<double>(SEA_LEVEL) - hydrology.surfaceElevation);
    const double oceanHabitat = oceanInfluence(hydrology);
    const double landHabitat = 1.0 - oceanHabitat;
    setScore(result, Biome::DEEP_OCEAN, oceanHabitat * smoothstep(7.0, 32.0, oceanDepth));
    setScore(result, Biome::OCEAN,
             oceanHabitat * (1.1 - smoothstep(18.0, 38.0, oceanDepth) * 0.35));
    setScore(result, Biome::FROZEN_OCEAN, oceanHabitat * bell(temperature, -8.0, 11.0) * 1.35);

    auto setLandScore = [&](Biome biome, double score) {
        setScore(result, biome, score * landHabitat);
    };
    const double coast = 1.0 - smoothstep(SEA_LEVEL + 2.0, SEA_LEVEL + 14.0, terrainHeight);
    const double riparian = channelInfluence(hydrology);
    setLandScore(Biome::BEACH, coast * bell(temperature, 20.0, 24.0));
    setLandScore(Biome::RIVER, riparian * 1.2);
    setLandScore(Biome::SWAMP, bell(temperature, 22.0, 15.0) * bell(rain, 1900.0, 1050.0) *
                                   bell(slope, 0.0, 0.32));
    setLandScore(Biome::MANGROVE,
                 coast * bell(temperature, 27.0, 10.0) * bell(rain, 2300.0, 900.0));
    setLandScore(Biome::TROPICAL_RAINFOREST,
                 bell(temperature, 28.0, 10.0) * bell(rain, 2850.0, 1000.0));
    setLandScore(Biome::TEMPERATE_RAINFOREST,
                 bell(temperature, 12.0, 9.0) * bell(rain, 2500.0, 950.0));
    setLandScore(Biome::TEMPERATE_CONIFER_FOREST, bell(temperature, 6.0, 11.0) *
                                                      bell(rain, 1450.0, 900.0) *
                                                      (0.72 + soil.moisture * 0.48));
    setLandScore(Biome::TROPICAL_CONIFER_FOREST,
                 bell(temperature, 22.0, 9.0) * bell(rain, 1450.0, 760.0) *
                     bell(terrainHeight, 118.0, 82.0) * (0.62 + soil.drainage * 0.58));
    setLandScore(Biome::TROPICAL_DRY_FOREST,
                 bell(temperature, 27.0, 10.0) * bell(rain, 850.0, 560.0) *
                     bell(climate.aridity, 0.90, 0.72) * (0.70 + soil.fertility * 0.30) * 1.30);
    setLandScore(Biome::FOREST,
                 bell(temperature, 16.0, 14.0) * bell(rain, 1250.0, 820.0) * soil.fertility);
    setLandScore(Biome::BIRCH_FOREST,
                 bell(temperature, 10.0, 8.0) * bell(rain, 1150.0, 650.0) * soil.fertility);
    setLandScore(Biome::TAIGA, bell(temperature, 1.0, 9.0) * bell(rain, 850.0, 600.0));
    setLandScore(Biome::PLAINS,
                 bell(temperature, 16.0, 18.0) * bell(rain, 750.0, 700.0) * (0.5 + soil.fertility));
    setLandScore(Biome::FLOWER_FIELD, bell(temperature, 17.0, 9.0) * bell(rain, 980.0, 420.0) *
                                          soil.fertility * (1.0 - steep));
    setLandScore(Biome::SAVANNA, bell(temperature, 27.0, 11.0) * bell(rain, 640.0, 500.0));
    setLandScore(Biome::FLOODED_GRASSLAND,
                 bell(temperature, 22.0, 18.0) * bell(rain, 1450.0, 1050.0) *
                     std::max(riparian, soil.moisture * 0.80) * (1.0 - steep) * 1.35);
    setLandScore(Biome::MEDITERRANEAN_WOODLAND,
                 bell(temperature, 18.0, 11.0) * bell(rain, 550.0, 380.0) *
                     bell(climate.aridity, 1.05, 0.75) * (0.80 + soil.fertility * 0.40) * 1.20);
    setLandScore(Biome::SHRUBLAND, bell(temperature, 14.0, 14.0) * bell(rain, 500.0, 430.0));
    setLandScore(Biome::STEPPE, bell(temperature, 7.0, 15.0) * bell(rain, 380.0, 300.0));
    setLandScore(Biome::DESERT,
                 bell(temperature, 30.0, 14.0) * bell(rain, 100.0, 250.0) * (0.65 + dry));
    setLandScore(Biome::COLD_DESERT,
                 bell(temperature, 0.0, 13.0) * bell(rain, 130.0, 260.0) * (0.55 + dry));
    const double geologyInterior = geologyInteriorInfluence(geology);
    const double sandstoneAffinity =
        geology.rock == RockType::SANDSTONE ? 0.55 + geologyInterior * 0.70 : 0.55;
    setLandScore(Biome::BADLANDS,
                 bell(temperature, 22.0, 14.0) * bell(rain, 260.0, 300.0) * sandstoneAffinity);
    setLandScore(Biome::TUNDRA,
                 bell(temperature, -5.0, 9.0) * bell(rain, 350.0, 400.0) * (1.0 - high * 0.4));
    setLandScore(Biome::ICE_SPIKES, bell(temperature, -15.0, 7.0) * bell(rain, 260.0, 300.0));
    setLandScore(Biome::EXTREME_HILLS, (0.38 + wet * 0.2) * std::max(high, steep));
    setLandScore(Biome::ALPINE, bell(temperature, -1.0, 12.0) * high * (0.5 + steep * 0.5));
    setLandScore(Biome::MONTANE_GRASSLAND, bell(temperature, 6.0, 12.0) * bell(rain, 720.0, 650.0) *
                                               bell(terrainHeight, 145.0, 68.0) *
                                               (0.62 + soil.moisture * 0.45) *
                                               (1.0 - steep * 0.35));
    setLandScore(Biome::GLACIER, bell(temperature, -18.0, 8.0) *
                                     smoothstep(135.0, 230.0, terrainHeight) *
                                     bell(rain, 1100.0, 1000.0));
    setLandScore(Biome::VOLCANIC_BARREN, geology.volcanicActivity * (0.65 + steep * 0.35));
    bool mushroomPlate = (geology.plateId & 0xFFU) < 8U;
    setLandScore(Biome::MUSHROOM_ISLAND, geology.crust == CrustType::OCEANIC &&
                                                 terrainHeight >= SEA_LEVEL &&
                                                 geology.volcanicActivity < 0.18 && mushroomPlate
                                             ? 1.15 * wet * geologyInterior
                                             : 0.0);

    // Ensure every land sample has a useful fallback even at unusual field
    // intersections.
    result.scores[biomeIndex(Biome::PLAINS)] += static_cast<float>(0.08 * landHabitat);
    return result;
}

BiomeBlend MacroGenerationSampler::selectBiome(const BiomeSuitability& suitability) {
    size_t primaryIndex = biomeIndex(Biome::PLAINS);
    size_t secondaryIndex = primaryIndex;
    double primaryScore = -1.0;
    double secondaryScore = -1.0;
    for (size_t index = 0; index < suitability.scores.size(); ++index) {
        double score = suitability.scores[index];
        if (score > primaryScore) {
            secondaryScore = primaryScore;
            secondaryIndex = primaryIndex;
            primaryScore = score;
            primaryIndex = index;
        } else if (score > secondaryScore) {
            secondaryScore = score;
            secondaryIndex = index;
        }
    }

    BiomeBlend result;
    result.primary = static_cast<Biome>(primaryIndex);
    result.secondary = static_cast<Biome>(secondaryIndex);
    double total = std::max(1.0e-12, primaryScore + std::max(0.0, secondaryScore));
    result.transition = clamp01(std::max(0.0, secondaryScore) / total);
    return result;
}

double MacroGenerationSampler::ecotopeInfluence(const SurfaceSample& surface, Ecotope ecotope) {
    const HydrologySample& hydrology = surface.hydrology;
    switch (ecotope) {
        case Ecotope::NONE:
            return 0.0;
        case Ecotope::RIVERBANK:
            return channelInfluence(hydrology, 1.6);
        case Ecotope::FLOODPLAIN:
            return channelInfluence(hydrology, 3.2) * (1.0 - smoothstep(0.28, 0.90, surface.slope));
        case Ecotope::DELTA:
            return hydrology.delta
                       ? smoothstep(0.0, 6.0, std::log1p(std::max(0.0, hydrology.sediment)))
                       : 0.0;
        case Ecotope::LAKESHORE:
            return hydrology.lake ? 1.0 - smoothstep(0.35, 3.5, std::max(0.0, hydrology.lakeDepth))
                                  : 0.0;
        case Ecotope::COAST:
            return 1.0 - smoothstep(2.0, 16.0, std::abs(surface.terrainHeight - SEA_LEVEL));
        case Ecotope::CLIFF:
            return smoothstep(0.45, 1.15, surface.slope);
        case Ecotope::SCREE:
            return smoothstep(0.35, 1.05, surface.slope) *
                   smoothstep(88.0, 155.0, surface.terrainHeight);
        case Ecotope::CANYON: {
            const double incision = smoothstep(2.0, 8.0, hydrology.erosionDepth);
            const double gradient = smoothstep(0.006, 0.024, hydrology.channelGradient);
            const double order = smoothstep(1.0, 3.0, hydrology.streamOrder);
            return incision * std::max(gradient, smoothstep(0.25, 0.75, surface.slope)) * order;
        }
        case Ecotope::GEOTHERMAL:
            return smoothstep(0.25, 0.75, surface.geology.volcanicActivity);
        case Ecotope::CAVE:
            return hasEcotope(surface.ecotopes, Ecotope::CAVE) ? 1.0 : 0.0;
        case Ecotope::AQUIFER:
            return smoothstep(surface.terrainHeight - 32.0, surface.terrainHeight - 8.0,
                              surface.soil.waterTable);
        case Ecotope::VALLEY:
            return bell(surface.terrainHeight, 68.0, 62.0) *
                   (1.0 - smoothstep(0.30, 0.95, surface.slope));
        case Ecotope::FOOTHILL:
            return bell(surface.terrainHeight, 108.0, 64.0) *
                   (0.72 + smoothstep(0.12, 0.65, surface.slope) * 0.28);
        case Ecotope::MONTANE:
            return bell(surface.terrainHeight, 158.0, 72.0) *
                   bell(surface.climate.temperatureC, 7.0, 22.0);
        case Ecotope::SUBALPINE:
            return bell(surface.terrainHeight, 214.0, 72.0) *
                   bell(surface.climate.temperatureC, 1.0, 17.0);
        case Ecotope::ALPINE_ZONE:
            return bell(surface.terrainHeight, 278.0, 92.0) *
                   bell(surface.climate.temperatureC, -5.0, 18.0);
        case Ecotope::SNOWFIELD:
            return smoothstep(118.0, 270.0, surface.terrainHeight) *
                   bell(surface.climate.temperatureC, -8.0, 14.0) *
                   smoothstep(100.0, 950.0, surface.climate.annualPrecipitationMm);
        case Ecotope::GLACIER:
            return smoothstep(145.0, 310.0, surface.terrainHeight) *
                   bell(surface.climate.temperatureC, -17.0, 11.0) *
                   smoothstep(350.0, 1500.0, surface.climate.annualPrecipitationMm);
        case Ecotope::EXPOSED_PEAK:
            return smoothstep(175.0, 340.0, surface.terrainHeight) *
                   std::max(smoothstep(0.38, 1.10, surface.slope),
                            smoothstep(0.35, 0.85, surface.geology.uplift));
        case Ecotope::ALL:
            return 1.0;
    }
    return 0.0;
}

Ecotope MacroGenerationSampler::classifyEcotopes(const SurfaceSample& surface) {
    Ecotope result = Ecotope::NONE;
    if (surface.hydrology.river) result |= Ecotope::RIVERBANK;
    if (!surface.hydrology.river &&
        surface.hydrology.channelDistance < surface.hydrology.channelWidth * 2.5) {
        result |= Ecotope::FLOODPLAIN;
    }
    if (surface.hydrology.delta) result |= Ecotope::DELTA;
    if (surface.hydrology.lake && surface.hydrology.lakeDepth < 2.2) {
        result |= Ecotope::LAKESHORE;
    }
    if (!surface.hydrology.ocean && surface.terrainHeight < SEA_LEVEL + 4.0) {
        result |= Ecotope::COAST;
    }
    // The numerical basin surface is sampled at 16-block spacing. A 0.75
    // rise-to-run slope at that scale already represents a sustained 37
    // degree face, while cube density adds the smaller ledges and overhangs.
    if (surface.slope > 0.75) result |= Ecotope::CLIFF;
    if (surface.slope > 0.50 && surface.terrainHeight > 105.0) result |= Ecotope::SCREE;
    if (surface.hydrology.erosionDepth > 4.5 && surface.hydrology.streamOrder >= 2 &&
        (surface.slope > 0.42 || surface.hydrology.channelGradient > 0.012)) {
        result |= Ecotope::CANYON;
    }
    if (surface.geology.volcanicActivity > 0.50) result |= Ecotope::GEOTHERMAL;
    if (surface.soil.waterTable > surface.terrainHeight - 18.0) result |= Ecotope::AQUIFER;
    if (!surface.hydrology.ocean) {
        constexpr std::array<Ecotope, 8> ELEVATION_ECOTOPES = {
            Ecotope::VALLEY,      Ecotope::FOOTHILL,  Ecotope::MONTANE, Ecotope::SUBALPINE,
            Ecotope::ALPINE_ZONE, Ecotope::SNOWFIELD, Ecotope::GLACIER, Ecotope::EXPOSED_PEAK,
        };
        for (const Ecotope ecotope : ELEVATION_ECOTOPES) {
            if (ecotopeInfluence(surface, ecotope) >= 0.28) result |= ecotope;
        }
    }
    return result;
}

SurfaceSample MacroGenerationSampler::sampleSurface(double x, double z) const {
    SurfaceSample result;
    result.geology = sampleGeology(x, z);
    result.hydrology = sampleHydrology(x, z);
    result.terrainHeight = result.hydrology.surfaceElevation;
    result.waterSurface = result.hydrology.waterSurface;
    result.slope = terrainSlope(x, z);
    result.climate = sampleClimate(x, z, result.terrainHeight);
    result.soil = sampleSoil(x, z, result.geology, result.hydrology, result.climate);
    result.suitability = biomeSuitability(result.geology, result.hydrology, result.climate,
                                          result.soil, result.terrainHeight, result.slope);
    result.biome = selectBiome(result.suitability);

    result.ecotopes = classifyEcotopes(result);
    return result;
}

} // namespace worldgen
