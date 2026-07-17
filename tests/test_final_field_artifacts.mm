#include <catch2/catch_test_macros.hpp>

#include "world/artifact_analysis.hpp"
#include "world/chunk_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace {

namespace artifact = worldgen::artifact_analysis;

constexpr int64_t CONTINUOUS_HALF_WINDOW = 128;
constexpr int64_t CATEGORICAL_HALF_WINDOW = 64;
constexpr std::array<uint32_t, 6> CONTINUOUS_SEEDS = {1, 7, 42, 12'345, 764'891, 0xDEAD'BEEF};
constexpr std::array<uint32_t, 2> CATEGORICAL_SEEDS = {42, 764'891};

struct WindowAnchor {
    int64_t line;
    int64_t along;
};

constexpr std::array<WindowAnchor, 6> WINDOW_ANCHORS = {
    WindowAnchor{12'345, -9'327},    WindowAnchor{-54'321, 48'123},
    WindowAnchor{90'117, 32'761},    WindowAnchor{-130'019, -77'003},
    WindowAnchor{249'999, -180'011}, WindowAnchor{-333'337, 222'229},
};

constexpr std::array CONTINUOUS_WINDOW_ANCHORS = {
    WINDOW_ANCHORS[0],
    WINDOW_ANCHORS[3],
};

struct SamplePos {
    int64_t x = 0;
    int64_t z = 0;

    bool operator==(const SamplePos&) const = default;
};

struct SamplePosHash {
    size_t operator()(const SamplePos& position) const noexcept {
        size_t seed = world_coord::mix(0, static_cast<uint64_t>(position.x));
        return world_coord::mix(seed, static_cast<uint64_t>(position.z));
    }
};

struct CachedProbe {
    worldgen::SurfaceSample surface;
    std::optional<BlockType> material;
    std::optional<BlockType> emittedGround;
    std::optional<BlockType> emittedTop;
};

uint8_t waterClass(const worldgen::SurfaceSample& sample);

struct FinalSurfaceDigest {
    double terrainHeight = 0.0;
    double waterSurface = 0.0;
    double hydrologySurface = 0.0;
    double hydrologyWaterSurface = 0.0;
    double shoreDistance = 0.0;
    double discharge = 0.0;
    double sediment = 0.0;
    double channelDistance = 0.0;
    double channelWidth = 0.0;
    double channelDepth = 0.0;
    double precipitation = 0.0;
    double temperature = 0.0;
    double humidity = 0.0;
    double evapotranspiration = 0.0;
    double aridity = 0.0;
    double soilMoisture = 0.0;
    double soilFertility = 0.0;
    double soilDrainage = 0.0;
    double waterTable = 0.0;
    double slope = 0.0;
    double lithologyTransition = 0.0;
    double lithologyContact = 0.0;
    uint64_t plateId = 0;
    worldgen::WaterBodyId waterBodyId = worldgen::NO_WATER_BODY;
    worldgen::RockType primaryRock = worldgen::RockType::GRANITE;
    worldgen::RockType secondaryRock = worldgen::RockType::GRANITE;
    Biome primaryBiome = Biome::PLAINS;
    Biome secondaryBiome = Biome::PLAINS;
    double biomeTransition = 0.0;
    uint8_t water = 0;
    BlockType material = BlockType::AIR;
    std::array<float, static_cast<size_t>(Biome::COUNT)> suitability{};

    bool operator==(const FinalSurfaceDigest&) const = default;
};

FinalSurfaceDigest digest(const worldgen::SurfaceSample& sample,
                          BlockType material = BlockType::AIR) {
    return {
        sample.terrainHeight,
        sample.waterSurface,
        sample.hydrology.surfaceElevation,
        sample.hydrology.waterSurface,
        sample.hydrology.lakeShoreDistance,
        sample.hydrology.discharge,
        sample.hydrology.sediment,
        sample.hydrology.channelDistance,
        sample.hydrology.channelWidth,
        sample.hydrology.channelDepth,
        sample.climate.annualPrecipitationMm,
        sample.climate.temperatureC,
        sample.climate.relativeHumidity,
        sample.climate.potentialEvapotranspirationMm,
        sample.climate.aridity,
        sample.soil.moisture,
        sample.soil.fertility,
        sample.soil.drainage,
        sample.soil.waterTable,
        sample.slope,
        sample.geology.lithology.transition,
        sample.geology.lithology.contactDistance,
        sample.geology.plateId,
        sample.hydrology.waterBodyId,
        sample.geology.lithology.primary,
        sample.geology.lithology.secondary,
        sample.biome.primary,
        sample.biome.secondary,
        sample.biome.transition,
        waterClass(sample),
        material,
        sample.suitability.scores,
    };
}

bool hasStandingWater(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.ocean || sample.hydrology.lake;
}

uint8_t waterClass(const worldgen::SurfaceSample& sample) {
    return static_cast<uint8_t>(
        (sample.hydrology.ocean ? 1U : 0U) | (sample.hydrology.lake ? 2U : 0U) |
        (sample.hydrology.river ? 4U : 0U) | (sample.hydrology.waterfall ? 8U : 0U) |
        (sample.hydrology.delta ? 16U : 0U));
}

bool isOutletOrChannelJunction(const worldgen::SurfaceSample& sample) {
    return sample.hydrology.river || sample.hydrology.waterfall || sample.hydrology.delta;
}

bool isTaggedFault(const worldgen::SurfaceSample& first, const worldgen::SurfaceSample& second) {
    return first.geology.faultStrength > 0.45 || second.geology.faultStrength > 0.45 ||
           first.geology.boundary == worldgen::PlateBoundary::TRANSFORM ||
           second.geology.boundary == worldgen::PlateBoundary::TRANSFORM;
}

bool hasUsableShoreDistance(const worldgen::SurfaceSample& sample) {
    return std::isfinite(sample.hydrology.lakeShoreDistance) &&
           sample.hydrology.lakeShoreDistance > -32'000.0 &&
           (sample.hydrology.lake || sample.hydrology.lakeBank ||
            std::abs(sample.hydrology.lakeShoreDistance) <= 96.0);
}

bool isWaterLevelJunction(const worldgen::SurfaceSample& first,
                          const worldgen::SurfaceSample& second) {
    if (!hasStandingWater(first) || !hasStandingWater(second))
        return false;
    if (first.hydrology.waterBodyId == second.hydrology.waterBodyId)
        return false;
    return std::abs(first.hydrology.waterSurface - second.hydrology.waterSurface) > 0.125;
}

bool isIntentionalShoreline(const worldgen::SurfaceSample& first,
                            const worldgen::SurfaceSample& second) {
    if (waterClass(first) == waterClass(second))
        return false;
    if (isOutletOrChannelJunction(first) || isOutletOrChannelJunction(second))
        return true;

    if (first.hydrology.ocean != second.hydrology.ocean) {
        const double firstSeaDistance = first.terrainHeight - SEA_LEVEL;
        const double secondSeaDistance = second.terrainHeight - SEA_LEVEL;
        if (firstSeaDistance * secondSeaDistance <= 0.0 ||
            std::min(std::abs(firstSeaDistance), std::abs(secondSeaDistance)) <= 2.0) {
            return true;
        }
    }

    if (first.hydrology.lake != second.hydrology.lake && hasUsableShoreDistance(first) &&
        hasUsableShoreDistance(second)) {
        const double firstDistance = first.hydrology.lakeShoreDistance;
        const double secondDistance = second.hydrology.lakeShoreDistance;
        return firstDistance * secondDistance <= 0.0 ||
               std::min(std::abs(firstDistance), std::abs(secondDistance)) <= 2.0;
    }
    return false;
}

bool hasHardMaterialConstraint(const worldgen::SurfaceSample& sample) {
    return waterClass(sample) != 0 || sample.geology.volcanicActivity > 0.52 ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::CLIFF) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SCREE) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::SNOWFIELD) ||
           worldgen::hasEcotope(sample.ecotopes, worldgen::Ecotope::GLACIER);
}

class ProbeCache {
public:
    explicit ProbeCache(uint32_t seed) : generator_(seed) {}

    const worldgen::SurfaceSample& surface(int64_t x, int64_t z) { return probe(x, z).surface; }

    BlockType material(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        if (!cached.material)
            cached.material = generator_.surfaceMaterialAt(x, z);
        return *cached.material;
    }

    BlockType emittedGround(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        populateEmittedBlocks(cached, x, z);
        return *cached.emittedGround;
    }

    BlockType emittedTop(int64_t x, int64_t z) {
        CachedProbe& cached = probe(x, z);
        populateEmittedBlocks(cached, x, z);
        return *cached.emittedTop;
    }

private:
    CachedProbe& probe(int64_t x, int64_t z) {
        const SamplePos position{x, z};
        if (auto found = probes_.find(position); found != probes_.end())
            return found->second;
        CachedProbe result;
        result.surface = generator_.sampleSurface(x, z);
        return probes_.emplace(position, std::move(result)).first->second;
    }

    BlockType blockAt(int64_t x, int y, int64_t z) {
        const ChunkPos position{Chunk::worldToChunk(x), Chunk::worldToChunkY(y),
                                Chunk::worldToChunk(z)};
        auto [found, inserted] = cubes_.try_emplace(position);
        if (inserted) {
            found->second = std::make_unique<Chunk>(position);
            generator_.generate(*found->second);
        }
        return found->second->getBlock(Chunk::worldToLocal(x), Chunk::worldToLocalY(y),
                                       Chunk::worldToLocal(z));
    }

    void populateEmittedBlocks(CachedProbe& cached, int64_t x, int64_t z) {
        if (cached.emittedGround && cached.emittedTop)
            return;
        const ColumnPos column{Chunk::worldToChunk(x), Chunk::worldToChunk(z)};
        const std::shared_ptr<const ColumnPlan> plan = generator_.getColumnPlan(column);
        const int groundY = plan->surfaceY(Chunk::worldToLocal(x), Chunk::worldToLocal(z));
        int topY = groundY;
        if (waterClass(cached.surface) != 0) {
            topY = std::max(topY, static_cast<int>(std::ceil(cached.surface.waterSurface)) - 1);
        }
        cached.emittedGround = blockAt(x, groundY, z);
        cached.emittedTop = blockAt(x, topY, z);
    }

    ChunkGenerator generator_;
    std::unordered_map<SamplePos, CachedProbe, SamplePosHash> probes_;
    std::unordered_map<ChunkPos, std::unique_ptr<Chunk>> cubes_;
};

struct EnergyAccumulator {
    double squaredSum = 0.0;
    size_t count = 0;

    void add(double derivative) {
        if (!std::isfinite(derivative))
            return;
        squaredSum += derivative * derivative;
        ++count;
    }

    void merge(const EnergyAccumulator& other) {
        squaredSum += other.squaredSum;
        count += other.count;
    }

    double mean() const { return count == 0 ? 0.0 : squaredSum / static_cast<double>(count); }
};

struct ContinuousMetric {
    EnergyAccumulator formerLine;
    EnergyAccumulator nearby;
    artifact::OrientationHistogram orientation;

    void merge(const ContinuousMetric& other) {
        formerLine.merge(other.formerLine);
        nearby.merge(other.nearby);
        artifact::add(orientation, other.orientation);
    }
};

enum class MeasuredField : uint8_t {
    TERRAIN,
    PRECIPITATION,
    LITHOLOGY,
    SHORE_DISTANCE,
    BIOME_SUITABILITY,
    COUNT,
};

using ContinuousMetrics = std::array<ContinuousMetric, static_cast<size_t>(MeasuredField::COUNT)>;

bool derivativeAllowed(const worldgen::SurfaceSample& first, const worldgen::SurfaceSample& second,
                       MeasuredField field) {
    if (field == MeasuredField::LITHOLOGY)
        return !isTaggedFault(first, second);
    if (field == MeasuredField::SHORE_DISTANCE) {
        if (!hasUsableShoreDistance(first) || !hasUsableShoreDistance(second))
            return false;
        if (isOutletOrChannelJunction(first) || isOutletOrChannelJunction(second))
            return false;
        if (isWaterLevelJunction(first, second))
            return false;
        const worldgen::WaterBodyId firstId = first.hydrology.waterBodyId;
        const worldgen::WaterBodyId secondId = second.hydrology.waterBodyId;
        return firstId == secondId || firstId == worldgen::NO_WATER_BODY ||
               secondId == worldgen::NO_WATER_BODY;
    }
    return !isTaggedFault(first, second) && !isWaterLevelJunction(first, second) &&
           !isIntentionalShoreline(first, second);
}

double scalarValue(const worldgen::SurfaceSample& sample, MeasuredField field) {
    switch (field) {
        case MeasuredField::TERRAIN:
            return artifact::fieldValue(sample, artifact::FinalField::TERRAIN_HEIGHT);
        case MeasuredField::PRECIPITATION:
            return artifact::fieldValue(sample, artifact::FinalField::ANNUAL_PRECIPITATION);
        case MeasuredField::LITHOLOGY:
            return artifact::fieldValue(sample, artifact::FinalField::LITHOLOGY_CONTACT);
        case MeasuredField::SHORE_DISTANCE:
            return artifact::fieldValue(sample, artifact::FinalField::LAKE_SHORE_DISTANCE);
        case MeasuredField::BIOME_SUITABILITY:
        case MeasuredField::COUNT:
            return 0.0;
    }
    return 0.0;
}

void addOrientation(artifact::OrientationBins& bins, double gradientX, double gradientZ) {
    if (!std::isfinite(gradientX) || !std::isfinite(gradientZ) ||
        std::hypot(gradientX, gradientZ) <= 1.0e-8) {
        return;
    }
    ++bins[artifact::orientationBin(gradientX, gradientZ)];
}

struct AxialPoint {
    int64_t x = 0;
    int64_t z = 0;
};

AxialPoint axialPoint(bool vertical, int64_t normal, int64_t along) {
    return vertical ? AxialPoint{normal, along} : AxialPoint{along, normal};
}

void accumulateDerivative(ProbeCache& cache, ContinuousMetric& metric, MeasuredField field,
                          bool formerLine, bool vertical, int64_t normal, int64_t along) {
    const AxialPoint firstPosition = axialPoint(vertical, normal - 1, along);
    const AxialPoint secondPosition = axialPoint(vertical, normal + 1, along);
    const worldgen::SurfaceSample& first = cache.surface(firstPosition.x, firstPosition.z);
    const worldgen::SurfaceSample& second = cache.surface(secondPosition.x, secondPosition.z);
    if (!derivativeAllowed(first, second, field))
        return;
    EnergyAccumulator& destination = formerLine ? metric.formerLine : metric.nearby;
    if (field != MeasuredField::BIOME_SUITABILITY) {
        destination.add((scalarValue(second, field) - scalarValue(first, field)) * 0.5);
        return;
    }
    for (size_t biome = 0; biome < static_cast<size_t>(Biome::COUNT); ++biome) {
        destination.add((artifact::biomeSuitabilityValue(second, static_cast<Biome>(biome)) -
                         artifact::biomeSuitabilityValue(first, static_cast<Biome>(biome))) *
                        0.5);
    }
}

void accumulateOrientation(ProbeCache& cache, ContinuousMetric& metric, MeasuredField field,
                           bool formerLine, int64_t x, int64_t z) {
    const worldgen::SurfaceSample& west = cache.surface(x - 1, z);
    const worldgen::SurfaceSample& east = cache.surface(x + 1, z);
    const worldgen::SurfaceSample& north = cache.surface(x, z - 1);
    const worldgen::SurfaceSample& south = cache.surface(x, z + 1);
    if (!derivativeAllowed(west, east, field) || !derivativeAllowed(north, south, field))
        return;
    artifact::OrientationBins& destination =
        formerLine ? metric.orientation.formerLine : metric.orientation.nearby;
    if (field != MeasuredField::BIOME_SUITABILITY) {
        addOrientation(destination, (scalarValue(east, field) - scalarValue(west, field)) * 0.5,
                       (scalarValue(south, field) - scalarValue(north, field)) * 0.5);
        return;
    }
    for (size_t biome = 0; biome < static_cast<size_t>(Biome::COUNT); ++biome) {
        const Biome value = static_cast<Biome>(biome);
        addOrientation(destination,
                       (artifact::biomeSuitabilityValue(east, value) -
                        artifact::biomeSuitabilityValue(west, value)) *
                           0.5,
                       (artifact::biomeSuitabilityValue(south, value) -
                        artifact::biomeSuitabilityValue(north, value)) *
                           0.5);
    }
}

ContinuousMetrics measureContinuousWindow(ProbeCache& cache, int64_t line, int64_t alongCenter,
                                          bool vertical) {
    ContinuousMetrics result;
    constexpr std::array<int64_t, 4> nearbyOffsets = {-6, -3, 3, 6};
    for (int64_t offset = -CONTINUOUS_HALF_WINDOW; offset <= CONTINUOUS_HALF_WINDOW; offset += 2) {
        const int64_t along = alongCenter + offset;
        for (size_t field = 0; field < result.size(); ++field) {
            const MeasuredField measured = static_cast<MeasuredField>(field);
            accumulateDerivative(cache, result[field], measured, true, vertical, line, along);
            for (const int64_t nearbyOffset : nearbyOffsets) {
                accumulateDerivative(cache, result[field], measured, false, vertical,
                                     line + nearbyOffset, along);
            }
            if (offset % 4 == 0) {
                const AxialPoint former = axialPoint(vertical, line, along);
                accumulateOrientation(cache, result[field], measured, true, former.x, former.z);
                for (const int64_t nearbyOffset : nearbyOffsets) {
                    const AxialPoint nearby = axialPoint(vertical, line + nearbyOffset, along);
                    accumulateOrientation(cache, result[field], measured, false, nearby.x,
                                          nearby.z);
                }
            }
        }
    }
    return result;
}

struct CategoricalRuns {
    int unexplainedWater = 0;
    int material = 0;
    int emittedGround = 0;
    int emittedTop = 0;
    int lithology = 0;
    int biome = 0;

    void include(const CategoricalRuns& other) {
        unexplainedWater = std::max(unexplainedWater, other.unexplainedWater);
        material = std::max(material, other.material);
        emittedGround = std::max(emittedGround, other.emittedGround);
        emittedTop = std::max(emittedTop, other.emittedTop);
        lithology = std::max(lithology, other.lithology);
        biome = std::max(biome, other.biome);
    }
};

CategoricalRuns measureCategoricalWindow(ProbeCache& cache, int64_t line, int64_t alongCenter,
                                         bool vertical) {
    CategoricalRuns longest;
    CategoricalRuns current;
    for (int64_t offset = -CATEGORICAL_HALF_WINDOW; offset <= CATEGORICAL_HALF_WINDOW; ++offset) {
        const AxialPoint firstPosition = axialPoint(vertical, line - 1, alongCenter + offset);
        const AxialPoint secondPosition = axialPoint(vertical, line + 1, alongCenter + offset);
        const worldgen::SurfaceSample& first = cache.surface(firstPosition.x, firstPosition.z);
        const worldgen::SurfaceSample& second = cache.surface(secondPosition.x, secondPosition.z);
        const bool intentionalShore = isIntentionalShoreline(first, second);
        const bool waterJunction = isWaterLevelJunction(first, second);
        const bool hardMaterial = hasHardMaterialConstraint(first) ||
                                  hasHardMaterialConstraint(second) || intentionalShore ||
                                  waterJunction || isTaggedFault(first, second);

        const bool unexplainedWater =
            waterClass(first) != waterClass(second) && !intentionalShore && !waterJunction;
        const bool materialBoundary =
            !hardMaterial && cache.material(firstPosition.x, firstPosition.z) !=
                                 cache.material(secondPosition.x, secondPosition.z);
        const bool groundBoundary =
            !hardMaterial && cache.emittedGround(firstPosition.x, firstPosition.z) !=
                                 cache.emittedGround(secondPosition.x, secondPosition.z);
        const bool topBoundary =
            !hardMaterial && cache.emittedTop(firstPosition.x, firstPosition.z) !=
                                 cache.emittedTop(secondPosition.x, secondPosition.z);
        const bool lithologyBoundary =
            !isTaggedFault(first, second) &&
            first.geology.lithology.primary != second.geology.lithology.primary;
        const bool biomeBoundary = !hardMaterial && first.biome.primary != second.biome.primary;

        current.unexplainedWater = unexplainedWater ? current.unexplainedWater + 1 : 0;
        current.material = materialBoundary ? current.material + 1 : 0;
        current.emittedGround = groundBoundary ? current.emittedGround + 1 : 0;
        current.emittedTop = topBoundary ? current.emittedTop + 1 : 0;
        current.lithology = lithologyBoundary ? current.lithology + 1 : 0;
        current.biome = biomeBoundary ? current.biome + 1 : 0;
        longest.include(current);
    }
    return longest;
}

size_t binTotal(const artifact::OrientationBins& bins) {
    size_t total = 0;
    for (const uint32_t count : bins)
        total += count;
    return total;
}

const char* fieldName(MeasuredField field) {
    switch (field) {
        case MeasuredField::TERRAIN:
            return "terrain";
        case MeasuredField::PRECIPITATION:
            return "precipitation";
        case MeasuredField::LITHOLOGY:
            return "lithology contact";
        case MeasuredField::SHORE_DISTANCE:
            return "shore distance";
        case MeasuredField::BIOME_SUITABILITY:
            return "biome suitability";
        case MeasuredField::COUNT:
            return "invalid";
    }
    return "invalid";
}

} // namespace

// Keep the quantitative acceptance matrix runnable while the remaining
// continuous former-line signatures are explicitly deferred. Categorical
// boundary runs remain in the default suite below.
TEST_CASE("Final emitted world fields do not inherit former control-line derivative energy",
          "[.known-continuity-debt][worldgen][continuity][artifact][final-fields]") {
    std::array<ContinuousMetrics, artifact::FORMER_GRID_SPACINGS.size()> bySpacing{};
    for (const uint32_t seed : CONTINUOUS_SEEDS) {
        ProbeCache cache(seed);
        for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
             ++spacingIndex) {
            const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
            for (const WindowAnchor anchor : CONTINUOUS_WINDOW_ANCHORS) {
                const int64_t line = world_coord::floorDiv(anchor.line, spacing) * spacing;
                for (const bool vertical : {true, false}) {
                    const ContinuousMetrics window =
                        measureContinuousWindow(cache, line, anchor.along, vertical);
                    for (size_t field = 0; field < window.size(); ++field)
                        bySpacing[spacingIndex][field].merge(window[field]);
                }
            }
        }
    }

    ContinuousMetric aggregateShore;
    for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
         ++spacingIndex) {
        const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
        for (size_t field = 0; field < static_cast<size_t>(MeasuredField::COUNT); ++field) {
            const MeasuredField measured = static_cast<MeasuredField>(field);
            const ContinuousMetric& metric = bySpacing[spacingIndex][field];
            if (measured == MeasuredField::SHORE_DISTANCE) {
                aggregateShore.merge(metric);
                continue;
            }
            const double ratio =
                artifact::energyRatio(metric.formerLine.mean(), metric.nearby.mean());
            INFO("ratio " << ratio << " spacing " << spacing << " final field "
                          << fieldName(measured) << " former samples " << metric.formerLine.count
                          << " nearby samples " << metric.nearby.count << " former energy "
                          << metric.formerLine.mean() << " nearby energy " << metric.nearby.mean());
            REQUIRE(metric.formerLine.count >= 100);
            REQUIRE(metric.nearby.count >= 400);
            CHECK(ratio >= artifact::DERIVATIVE_RATIO_MINIMUM);
            CHECK(ratio <= artifact::DERIVATIVE_RATIO_MAXIMUM);

            // The compact exact lithology authority stores contact distance
            // at one-eighth-block precision. Its scalar gradient therefore
            // collapses into cardinal bins even when the categorical contact
            // itself is curved. Derivative energy and the categorical run
            // test remain meaningful; orientation does not.
            if (measured != MeasuredField::LITHOLOGY &&
                binTotal(metric.orientation.formerLine) >= 64 &&
                binTotal(metric.orientation.nearby) >= 256) {
                const double orientationRatio =
                    artifact::structuredOrientationRatio(metric.orientation);
                INFO("orientation " << orientationRatio << " spacing " << spacing << " final field "
                                    << fieldName(measured));
                CHECK(orientationRatio <= artifact::STRUCTURED_ORIENTATION_LIMIT);
            }
        }
    }

    const double shoreRatio =
        artifact::energyRatio(aggregateShore.formerLine.mean(), aggregateShore.nearby.mean());
    INFO("shore former samples " << aggregateShore.formerLine.count << " nearby samples "
                                 << aggregateShore.nearby.count << " ratio " << shoreRatio);
    REQUIRE(aggregateShore.formerLine.count >= 24);
    REQUIRE(aggregateShore.nearby.count >= 96);
    CHECK(shoreRatio >= artifact::DERIVATIVE_RATIO_MINIMUM);
    CHECK(shoreRatio <= artifact::DERIVATIVE_RATIO_MAXIMUM);
    if (binTotal(aggregateShore.orientation.formerLine) >= 32 &&
        binTotal(aggregateShore.orientation.nearby) >= 128) {
        CHECK(artifact::structuredOrientationRatio(aggregateShore.orientation) <=
              artifact::STRUCTURED_ORIENTATION_LIMIT);
    }
}

TEST_CASE("Final emitted categories do not run along former control lines",
          "[worldgen][continuity][artifact][final-fields][blocks]") {
    std::array<CategoricalRuns, artifact::FORMER_GRID_SPACINGS.size()> bySpacing{};
    for (const uint32_t seed : CATEGORICAL_SEEDS) {
        ProbeCache cache(seed);
        for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
             ++spacingIndex) {
            const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
            for (const WindowAnchor anchor : WINDOW_ANCHORS) {
                const int64_t line = world_coord::floorDiv(anchor.line, spacing) * spacing;
                for (const bool vertical : {true, false}) {
                    bySpacing[spacingIndex].include(
                        measureCategoricalWindow(cache, line, anchor.along, vertical));
                }
            }
        }
    }

    for (size_t spacingIndex = 0; spacingIndex < artifact::FORMER_GRID_SPACINGS.size();
         ++spacingIndex) {
        const int64_t spacing = artifact::FORMER_GRID_SPACINGS[spacingIndex];
        const CategoricalRuns& runs = bySpacing[spacingIndex];
        INFO("spacing " << spacing << " unexplained water " << runs.unexplainedWater << " material "
                        << runs.material << " emitted ground " << runs.emittedGround
                        << " emitted top " << runs.emittedTop << " lithology " << runs.lithology
                        << " biome " << runs.biome);
        REQUIRE(runs.unexplainedWater <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.material <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.emittedGround <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.emittedTop <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.lithology <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
        REQUIRE(runs.biome <= artifact::CATEGORICAL_BOUNDARY_RUN_LIMIT);
    }
}

TEST_CASE("Column plans share exact final fields across negative and positive faces",
          "[worldgen][continuity][artifact][final-fields][seam]") {
    ChunkGenerator generator(764'891);
    constexpr std::array<int64_t, 7> boundaries = {-64, -32, -16, 0, 16, 32, 64};
    constexpr std::array<int64_t, 8> alongPositions = {-65, -33, -17, -1, 0, 15, 31, 63};

    for (const int64_t boundary : boundaries) {
        const int64_t highColumn =
            world_coord::floorDiv(boundary, static_cast<int64_t>(CHUNK_EDGE));
        for (const int64_t along : alongPositions) {
            CAPTURE(boundary, along);
            const int64_t alongColumn =
                world_coord::floorDiv(along, static_cast<int64_t>(CHUNK_EDGE));
            const int localAlong = Chunk::worldToLocal(along);

            const auto west = generator.getColumnPlan(ColumnPos{highColumn - 1, alongColumn});
            const auto east = generator.getColumnPlan(ColumnPos{highColumn, alongColumn});
            REQUIRE(digest(west->sample(CHUNK_EDGE, localAlong)) ==
                    digest(east->sample(0, localAlong)));

            const auto north = generator.getColumnPlan(ColumnPos{alongColumn, highColumn - 1});
            const auto south = generator.getColumnPlan(ColumnPos{alongColumn, highColumn});
            REQUIRE(digest(north->sample(localAlong, CHUNK_EDGE)) ==
                    digest(south->sample(localAlong, 0)));
        }
    }
}

TEST_CASE("Final fields are identical after reverse-order macro cache eviction",
          "[worldgen][continuity][artifact][final-fields][determinism]") {
    constexpr std::array<SamplePos, 16> positions = {
        SamplePos{-8193, -2049}, SamplePos{-8192, -2048}, SamplePos{-8191, -2047},
        SamplePos{-2049, -65},   SamplePos{-2048, -64},   SamplePos{-2047, -63},
        SamplePos{-17, -17},     SamplePos{-16, -16},     SamplePos{-1, -1},
        SamplePos{0, 0},         SamplePos{15, 15},       SamplePos{16, 16},
        SamplePos{63, 65},       SamplePos{2048, -2048},  SamplePos{8192, 2048},
        SamplePos{8193, 8193},
    };

    ChunkGenerator generator(764'891);
    std::array<FinalSurfaceDigest, positions.size()> forward;
    for (size_t index = 0; index < positions.size(); ++index) {
        const SamplePos position = positions[index];
        const worldgen::SurfaceSample sample = generator.sampleSurface(position.x, position.z);
        forward[index] = digest(sample, generator.surfaceMaterialAt(position.x, position.z));
    }

    generator.clearMacroCaches();
    for (size_t reverseIndex = positions.size(); reverseIndex > 0; --reverseIndex) {
        const size_t index = reverseIndex - 1;
        const SamplePos position = positions[index];
        CAPTURE(position.x, position.z);
        const worldgen::SurfaceSample sample = generator.sampleSurface(position.x, position.z);
        REQUIRE(digest(sample, generator.surfaceMaterialAt(position.x, position.z)) ==
                forward[index]);
    }
}
