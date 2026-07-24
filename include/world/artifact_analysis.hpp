#pragma once

#include "common/counter_rng.hpp"
#include "world/macro_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace worldgen::artifact_analysis {

constexpr int64_t HALF_WINDOW = 2'048;
constexpr int64_t CATEGORICAL_HALF_WINDOW = 128;
constexpr double DERIVATIVE_RATIO_MINIMUM = 0.85;
constexpr double DERIVATIVE_RATIO_MAXIMUM = 1.15;
constexpr double STRUCTURED_ORIENTATION_LIMIT = 1.5;
constexpr int CATEGORICAL_BOUNDARY_RUN_LIMIT = 24;
constexpr std::array<int, 8> NEARBY_OFFSETS = {-12, -9, -6, -3, 3, 6, 9, 12};
constexpr std::array<int64_t, 6> FORMER_GRID_SPACINGS = {8, 16, 32, 64, 2'048, 8'192};

enum class Field : uint8_t {
    PRELIMINARY_ELEVATION,
    PROVISIONAL_PRECIPITATION,
    LITHOLOGY_CONTACT,
};

constexpr std::array FIELDS = {
    Field::PRELIMINARY_ELEVATION,
    Field::PROVISIONAL_PRECIPITATION,
    Field::LITHOLOGY_CONTACT,
};

// These are the post-hydrology, post-volcanism fields exposed by
// ChunkGenerator::sampleSurface(). Keeping them separate from Field prevents
// the inexpensive macro diagnostic above from being mistaken for emitted
// world coverage.
enum class FinalField : uint8_t {
    TERRAIN_HEIGHT,
    ANNUAL_PRECIPITATION,
    LITHOLOGY_CONTACT,
    LAKE_SHORE_DISTANCE,
};

constexpr std::array FINAL_FIELDS = {
    FinalField::TERRAIN_HEIGHT,
    FinalField::ANNUAL_PRECIPITATION,
    FinalField::LITHOLOGY_CONTACT,
    FinalField::LAKE_SHORE_DISTANCE,
};

using OrientationBins = std::array<uint32_t, 8>;

struct OrientationHistogram {
    OrientationBins formerLine{};
    OrientationBins nearby{};
};

inline double fieldValue(MacroGenerationSampler& sampler, Field field, double x, double z) {
    switch (field) {
        case Field::PRELIMINARY_ELEVATION:
            return sampler.preliminaryElevation(x, z);
        case Field::PROVISIONAL_PRECIPITATION:
            return sampler.sampleProvisionalRainfall(x, z);
        case Field::LITHOLOGY_CONTACT:
            return sampler.sampleGeology(x, z).lithology.contactDistance;
    }
    return 0.0;
}

inline double fieldValue(const SurfaceSample& sample, FinalField field) {
    switch (field) {
        case FinalField::TERRAIN_HEIGHT:
            return sample.terrainHeight;
        case FinalField::ANNUAL_PRECIPITATION:
            return sample.climate.annualPrecipitationMm;
        case FinalField::LITHOLOGY_CONTACT:
            return sample.geology.lithology.contactDistance;
        case FinalField::LAKE_SHORE_DISTANCE:
            return sample.hydrology.lakeShoreDistance;
    }
    return 0.0;
}

inline double biomeSuitabilityValue(const SurfaceSample& sample, Biome biome) {
    return sample.suitability.scores[static_cast<size_t>(biome)];
}

template <typename ValueAt>
double derivativeEnergy(ValueAt&& valueAt, int64_t x, int64_t centerZ) {
    double energy = 0.0;
    size_t count = 0;
    for (int64_t offset = -HALF_WINDOW; offset <= HALF_WINDOW; offset += 2) {
        const double derivative =
            (valueAt(x + 1, centerZ + offset) - valueAt(x - 1, centerZ + offset)) * 0.5;
        if (!std::isfinite(derivative)) continue;
        energy += derivative * derivative;
        ++count;
    }
    return count == 0 ? 0.0 : energy / static_cast<double>(count);
}

inline double derivativeEnergy(MacroGenerationSampler& sampler, Field field, int64_t x,
                               int64_t centerZ) {
    const auto valueAt = [&](double sampleX, double sampleZ) {
        return fieldValue(sampler, field, sampleX, sampleZ);
    };
    return derivativeEnergy(valueAt, x, centerZ);
}

inline double nearbyDerivativeEnergy(MacroGenerationSampler& sampler, Field field, int64_t x,
                                     int64_t centerZ) {
    double energy = 0.0;
    for (const int offset : NEARBY_OFFSETS)
        energy += derivativeEnergy(sampler, field, x + offset, centerZ);
    return energy / static_cast<double>(NEARBY_OFFSETS.size());
}

inline double energyRatio(double boundary, double nearby) {
    constexpr double EPSILON = 1.0e-12;
    if (nearby > EPSILON) return boundary / nearby;
    return boundary <= EPSILON ? 1.0 : 1.0e9;
}

inline size_t orientationBin(double gradientX, double gradientZ) {
    double angle = std::atan2(gradientZ, gradientX);
    if (angle < 0.0) angle += std::numbers::pi;
    if (angle >= std::numbers::pi) angle -= std::numbers::pi;
    return static_cast<size_t>(static_cast<int>(std::floor(angle * 8.0 / std::numbers::pi + 0.5)) %
                               8);
}

template <typename ValueAt>
void accumulateOrientationLine(OrientationHistogram& histogram, ValueAt& valueAt, int64_t line,
                               int64_t alongCenter, bool vertical) {
    const auto record = [&](OrientationBins& bins, int64_t x, int64_t z) {
        const double gradientX = (valueAt(x + 1, z) - valueAt(x - 1, z)) * 0.5;
        const double gradientZ = (valueAt(x, z + 1) - valueAt(x, z - 1)) * 0.5;
        if (!std::isfinite(gradientX) || !std::isfinite(gradientZ) ||
            std::hypot(gradientX, gradientZ) <= 1.0e-8) {
            return;
        }
        ++bins[orientationBin(gradientX, gradientZ)];
    };

    for (int64_t offset = -HALF_WINDOW; offset <= HALF_WINDOW; offset += 8) {
        const int64_t x = vertical ? line : alongCenter + offset;
        const int64_t z = vertical ? alongCenter + offset : line;
        record(histogram.formerLine, x, z);
        for (const int nearbyOffset : NEARBY_OFFSETS) {
            record(histogram.nearby, x + (vertical ? nearbyOffset : 0),
                   z + (vertical ? 0 : nearbyOffset));
        }
    }
}

template <typename ValueAt>
OrientationHistogram orientationHistogram(ValueAt&& valueAt, int64_t line, int64_t alongCenter) {
    OrientationHistogram result;
    accumulateOrientationLine(result, valueAt, line, alongCenter, true);
    accumulateOrientationLine(result, valueAt, line, alongCenter, false);
    return result;
}

inline OrientationHistogram orientationHistogram(MacroGenerationSampler& sampler, Field field,
                                                 int64_t line, int64_t alongCenter) {
    // Evaluate both axes because a storage lattice can imprint either a
    // north-south or east-west line. The paired baseline removes legitimate
    // regional anisotropy from both measurements.
    auto valueAt = [&](double x, double z) { return fieldValue(sampler, field, x, z); };
    return orientationHistogram(valueAt, line, alongCenter);
}

inline void add(OrientationHistogram& destination, const OrientationHistogram& source) {
    for (size_t index = 0; index < destination.formerLine.size(); ++index) {
        destination.formerLine[index] += source.formerLine[index];
        destination.nearby[index] += source.nearby[index];
    }
}

inline std::array<double, 8> orientationBias(const OrientationBins& formerLine,
                                             const OrientationBins& nearby) {
    constexpr double PRIOR = 1.0;
    double formerObserved = 0.0;
    double nearbyObserved = 0.0;
    for (size_t index = 0; index < formerLine.size(); ++index) {
        formerObserved += static_cast<double>(formerLine[index]);
        nearbyObserved += static_cast<double>(nearby[index]);
    }

    // The nearby control samples eight offset lines for each former line, so
    // paired histograms ordinarily have different exposure. Scale the
    // Dirichlet pseudocount with that exposure: proportional observations
    // then remain proportional even when both histograms have empty bins.
    // Equal-exposure histograms retain the original one-count prior exactly.
    double referenceExposure = 0.0;
    if (formerObserved > 0.0 && nearbyObserved > 0.0)
        referenceExposure = std::min(formerObserved, nearbyObserved);
    else
        referenceExposure = std::max(formerObserved, nearbyObserved);
    const double formerPrior = referenceExposure > 0.0 && formerObserved > 0.0
                                   ? PRIOR * formerObserved / referenceExposure
                                   : PRIOR;
    const double nearbyPrior = referenceExposure > 0.0 && nearbyObserved > 0.0
                                   ? PRIOR * nearbyObserved / referenceExposure
                                   : PRIOR;
    const double formerTotal = formerObserved + formerPrior * 8.0;
    const double nearbyTotal = nearbyObserved + nearbyPrior * 8.0;

    std::array<double, 8> result{};
    for (size_t index = 0; index < result.size(); ++index) {
        const double formerFraction =
            (static_cast<double>(formerLine[index]) + formerPrior) / formerTotal;
        const double nearbyFraction =
            (static_cast<double>(nearby[index]) + nearbyPrior) / nearbyTotal;
        result[index] = formerFraction / nearbyFraction;
    }
    return result;
}

inline std::array<double, 8> orientationBias(const OrientationHistogram& histogram) {
    return orientationBias(histogram.formerLine, histogram.nearby);
}

inline double structuredOrientationRatio(const std::array<double, 8>& bias) {
    const double structured = std::max({bias[0], bias[2], bias[4], bias[6]});
    std::array<double, 4> unstructured = {bias[1], bias[3], bias[5], bias[7]};
    std::sort(unstructured.begin(), unstructured.end());
    const double median = (unstructured[1] + unstructured[2]) * 0.5;
    if (median > 0.0) return structured / median;
    return structured == 0.0 ? 1.0 : 1.0e9;
}

inline double structuredOrientationRatio(const OrientationHistogram& histogram) {
    return structuredOrientationRatio(orientationBias(histogram));
}

inline double structuredOrientationRatio(const OrientationBins& bins) {
    const uint32_t structured = std::max({bins[0], bins[2], bins[4], bins[6]});
    std::array<uint32_t, 4> unstructured = {bins[1], bins[3], bins[5], bins[7]};
    std::sort(unstructured.begin(), unstructured.end());
    const double median =
        (static_cast<double>(unstructured[1]) + static_cast<double>(unstructured[2])) * 0.5;
    if (median > 0.0) return static_cast<double>(structured) / median;
    return structured == 0 ? 1.0 : 1.0e9;
}

inline OrientationBins globalOrientationHistogram(MacroGenerationSampler& sampler) {
    constexpr size_t SAMPLE_COUNT = 4'096;
    constexpr int32_t SAMPLE_EXTENT = 8'000'000;
    constexpr uint64_t POSITION_STREAM = 0x4F52'4945'4E54'4154ULL;
    constexpr CounterRng positions(0xC011'71A5'B17E'5EEDULL);
    // The fixed counter-random route samples a square world area without
    // favoring either coordinate axis.
    OrientationBins result{};
    for (uint32_t index = 0; index < SAMPLE_COUNT; ++index) {
        const int64_t x =
            positions.uniformInt(POSITION_STREAM, index, 0, 0, 0, -SAMPLE_EXTENT, SAMPLE_EXTENT);
        const int64_t z =
            positions.uniformInt(POSITION_STREAM, index, 0, 0, 1, -SAMPLE_EXTENT, SAMPLE_EXTENT);
        for (const Field field : FIELDS) {
            const double gradientX =
                (fieldValue(sampler, field, x + 1, z) - fieldValue(sampler, field, x - 1, z)) * 0.5;
            const double gradientZ =
                (fieldValue(sampler, field, x, z + 1) - fieldValue(sampler, field, x, z - 1)) * 0.5;
            if (!std::isfinite(gradientX) || !std::isfinite(gradientZ) ||
                std::hypot(gradientX, gradientZ) <= 1.0e-8) {
                continue;
            }
            ++result[orientationBin(gradientX, gradientZ)];
        }
    }
    return result;
}

} // namespace worldgen::artifact_analysis
