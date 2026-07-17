#pragma once

#include "common/counter_rng.hpp"
#include "world/noise.hpp"

#include <cstdint>

namespace worldgen {

inline constexpr int ALPINE_CIRQUE_CANDIDATE_LIMIT = 9;
inline constexpr int ALPINE_RIDGE_DETAIL_BAND_COUNT = 4;

// Inputs are continuous fields rather than categorical biome switches. This
// keeps the morphology coordinate-pure and lets exact and far terrain query
// the same landform at different filtered footprints.
struct AlpineTectonicContext {
    double x = 0.0;
    double z = 0.0;
    double uplift = 0.0;
    double rockResistance = 0.0;
    double continentalFraction = 0.0;
};

struct AlpineTectonicSample {
    double upliftGate = 0.0;
    double ridgeStrength = 0.0;
    double hornStrength = 0.0;
    double elevationOffset = 0.0;
};

struct AlpineSurfaceContext : AlpineTectonicContext {
    double terrainHeight = 0.0;
    double temperatureC = 0.0;
    double annualPrecipitationMm = 0.0;
    double flowX = 1.0;
    double flowZ = 0.0;
    double channelDistance = 1.0e9;
    double channelWidth = 0.0;
    double channelGradient = 0.0;
    double discharge = 0.0;
    double erosionDepth = 0.0;
    int footprintWidth = 1;
    bool ocean = false;
    bool lake = false;
};

struct AlpineMorphologySample {
    double ridgeStrength = 0.0;
    double hornStrength = 0.0;
    double glacialInfluence = 0.0;
    double periglacialInfluence = 0.0;
    double valleyInfluence = 0.0;
    double cirqueInfluence = 0.0;
    double talusInfluence = 0.0;
    double ridgeDetail = 0.0;
    double valleyCarve = 0.0;
    double cirqueCarve = 0.0;
    double talusDeposit = 0.0;
    double elevationOffset = 0.0;
};

// The basin solver uses this bounded response instead of treating every cell
// as the same soil-covered slope. Drainage convergence can still incise a
// resistant divide, while high-uplift bedrock is not uniformly diffused away.
struct AlpineErosionContext {
    double uplift = 0.0;
    double rockResistance = 0.0;
    double terrainHeight = 0.0;
    double temperatureC = 0.0;
    double annualPrecipitationMm = 0.0;
    double drainageConvergence = 0.0;
    double slope = 0.0;
};

struct AlpineErosionResponse {
    double ridgePreservation = 0.0;
    double glacialCompetition = 0.0;
    double periglacialWeathering = 0.0;
    double streamIncisionScale = 1.0;
    double thermalRelaxationScale = 1.0;
    double criticalSlope = 0.7;
};

AlpineErosionResponse alpineErosionResponse(const AlpineErosionContext& context) noexcept;

class AlpineMorphologySampler {
public:
    explicit AlpineMorphologySampler(uint64_t worldSeed);

    AlpineTectonicSample sampleTectonic(const AlpineTectonicContext& context) const noexcept;
    AlpineMorphologySample sampleSurface(const AlpineSurfaceContext& context) const noexcept;
    // Production sampling already carries the tectonic sample in its geology
    // result. Accepting it here prevents the exact and far hot paths from
    // evaluating the same broad ridge network twice.
    AlpineMorphologySample sampleSurface(const AlpineSurfaceContext& context,
                                         const AlpineTectonicSample& tectonic) const noexcept;
    // Hydrology owns the footprint-16 morphology. Exact and finer far tiers
    // add only the difference from this filtered crest term, so refinement
    // cannot move a drainage divide, lake, or channel.
    double sampleRidgeDetail(const AlpineSurfaceContext& context,
                             const AlpineTectonicSample& tectonic) const noexcept;

private:
    struct CirqueCandidate {
        double x = 0.0;
        double z = 0.0;
        double orientation = 0.0;
        double radius = 0.0;
        double strength = 0.0;
    };

    CirqueCandidate cirqueCandidate(int64_t cellX, int64_t cellZ) const noexcept;
    double filteredRidgeDetail(const AlpineSurfaceContext& context,
                               const AlpineTectonicSample& tectonic) const noexcept;

    uint64_t cacheTag_ = 0;
    CounterRng random_;
    SimplexNoise warpNoise_;
    SimplexNoise ridgeNoise_;
    SimplexNoise detailNoise_;
    SimplexNoise hornNoise_;
};

} // namespace worldgen
