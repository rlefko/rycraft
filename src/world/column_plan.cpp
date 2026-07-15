#include "world/column_plan.hpp"

#include "world/features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

namespace {

constexpr uint8_t CANONICAL_LAKE_KNOWN = 1U << 0U;
constexpr uint8_t CANONICAL_LAKE_PRESENT = 1U << 1U;
constexpr uint8_t CANONICAL_LAKE_ENDORHEIC = 1U << 2U;
constexpr double CANONICAL_LAKE_DEPTH_SCALE = 64.0;

uint16_t encodeCanonicalLakeDepth(double depth) {
    return static_cast<uint16_t>(
        std::clamp(std::ceil(std::max(0.0, depth) * CANONICAL_LAKE_DEPTH_SCALE), 0.0,
                   static_cast<double>(std::numeric_limits<uint16_t>::max())));
}

double decodeCanonicalLakeDepth(uint16_t depth) {
    return static_cast<double>(depth) / CANONICAL_LAKE_DEPTH_SCALE;
}

double lerp(double first, double second, double amount) {
    return first + (second - first) * amount;
}

template <typename Getter>
double bilerp(const std::array<const worldgen::SurfaceSample*, 4>& samples, double fx, double fz,
              Getter getter) {
    if (fx <= 0.0) return lerp(getter(*samples[0]), getter(*samples[2]), fz);
    if (fx >= 1.0) return lerp(getter(*samples[1]), getter(*samples[3]), fz);
    if (fz <= 0.0) return lerp(getter(*samples[0]), getter(*samples[1]), fx);
    if (fz >= 1.0) return lerp(getter(*samples[2]), getter(*samples[3]), fx);
    return lerp(lerp(getter(*samples[0]), getter(*samples[1]), fx),
                lerp(getter(*samples[2]), getter(*samples[3]), fx), fz);
}

worldgen::Vector2d normalized(worldgen::Vector2d value) {
    double magnitude = std::hypot(value.x, value.z);
    if (magnitude < 1.0e-12) return {1.0, 0.0};
    return {value.x / magnitude, value.z / magnitude};
}

bool sameWaterBody(const worldgen::HydrologySample& sample,
                   const worldgen::HydrologySample& reference) {
    if (reference.ocean) return sample.ocean;
    if (reference.lake) return sample.lake;
    if (reference.river) return sample.river;
    return !sample.ocean && !sample.lake && !sample.river;
}

double waterSurface(const std::array<const worldgen::SurfaceSample*, 4>& samples,
                    const worldgen::HydrologySample& reference, double fx, double fz) {
    const std::array<double, 4> weights = {
        (1.0 - fx) * (1.0 - fz),
        fx * (1.0 - fz),
        (1.0 - fx) * fz,
        fx * fz,
    };
    double weightedSurface = 0.0;
    double totalWeight = 0.0;
    for (size_t index = 0; index < samples.size(); ++index) {
        if (!sameWaterBody(samples[index]->hydrology, reference)) continue;
        weightedSurface += samples[index]->waterSurface * weights[index];
        totalWeight += weights[index];
    }
    return totalWeight > 1.0e-9 ? weightedSurface / totalWeight : reference.waterSurface;
}

struct LakeBodySelection {
    double level = 0.0;
    double membership = 0.0;
    bool endorheic = false;
};

std::optional<LakeBodySelection>
dominantLakeBody(const std::array<const worldgen::SurfaceSample*, 4>& samples, double fx,
                 double fz) {
    const std::array<double, 4> weights = {
        (1.0 - fx) * (1.0 - fz),
        fx * (1.0 - fz),
        (1.0 - fx) * fz,
        fx * fz,
    };
    std::array<LakeBodySelection, 4> bodies{};
    size_t bodyCount = 0;
    for (size_t index = 0; index < samples.size(); ++index) {
        const worldgen::HydrologySample& hydrology = samples[index]->hydrology;
        if (!hydrology.lake || weights[index] <= 0.0) continue;
        size_t body = 0;
        while (body < bodyCount &&
               (std::abs(bodies[body].level - hydrology.waterSurface) > 1.0e-4 ||
                bodies[body].endorheic != hydrology.endorheic)) {
            ++body;
        }
        if (body == bodyCount) {
            bodies[bodyCount++] = {
                .level = hydrology.waterSurface,
                .membership = 0.0,
                .endorheic = hydrology.endorheic,
            };
        }
        LakeBodySelection& selected = bodies[body];
        const double combinedWeight = selected.membership + weights[index];
        selected.level =
            (selected.level * selected.membership + hydrology.waterSurface * weights[index]) /
            combinedWeight;
        selected.membership = combinedWeight;
    }
    if (bodyCount == 0) return std::nullopt;
    const auto best =
        std::max_element(bodies.begin(), bodies.begin() + static_cast<std::ptrdiff_t>(bodyCount),
                         [](const LakeBodySelection& first, const LakeBodySelection& second) {
                             if (first.membership != second.membership) {
                                 return first.membership < second.membership;
                             }
                             return first.level > second.level;
                         });
    constexpr double DOMINANT_MEMBERSHIP = 0.5;
    if (best->membership + 1.0e-9 < DOMINANT_MEMBERSHIP) return std::nullopt;
    return *best;
}

void addSection(std::set<int32_t>& sections, int blockY) {
    int clampedY = std::clamp(blockY, WORLD_MIN_Y, WORLD_MAX_Y);
    sections.insert(
        world_coord::floorDiv(static_cast<int32_t>(clampedY), static_cast<int32_t>(CHUNK_EDGE)));
}

void addSectionRange(std::set<int32_t>& sections, int minimumBlockY, int maximumBlockY) {
    const int low = std::clamp(minimumBlockY, WORLD_MIN_Y, WORLD_MAX_Y);
    const int high = std::clamp(maximumBlockY, WORLD_MIN_Y, WORLD_MAX_Y);
    if (low > high) return;
    const int32_t first =
        world_coord::floorDiv(static_cast<int32_t>(low), static_cast<int32_t>(CHUNK_EDGE));
    const int32_t last =
        world_coord::floorDiv(static_cast<int32_t>(high), static_cast<int32_t>(CHUNK_EDGE));
    for (int32_t section = first; section <= last; ++section)
        sections.insert(section);
}

} // namespace

ColumnPlan::ColumnPlan(ColumnPos chunkColumn, const ColumnPlanSurfaceSampler& sampleSurface,
                       const ColumnPlanHeightSampler& sampleHeight,
                       const ColumnPlanSurfaceGridSampler& sampleExactSurface,
                       const ColumnPlanHydrologySampler& sampleHydrology)
    : chunkColumn_(chunkColumn) {
    const int64_t baseX = chunkColumn.x * CHUNK_EDGE;
    const int64_t baseZ = chunkColumn.z * CHUNK_EDGE;
    for (int latticeZ = 0; latticeZ < COLUMN_PLAN_LATTICE_EDGE; ++latticeZ) {
        for (int latticeX = 0; latticeX < COLUMN_PLAN_LATTICE_EDGE; ++latticeX) {
            const int64_t worldX = baseX + latticeX * COLUMN_PLAN_LATTICE_SPACING;
            const int64_t worldZ = baseZ + latticeZ * COLUMN_PLAN_LATTICE_SPACING;
            lattice_[static_cast<size_t>(latticeZ * COLUMN_PLAN_LATTICE_EDGE + latticeX)] =
                sampleSurface(worldX, worldZ);
        }
    }
    if (sampleHydrology) {
        const auto storeCanonicalLake = [](CanonicalLakeSample& destination,
                                           const worldgen::HydrologySample& hydrology) {
            destination.waterSurface = static_cast<float>(hydrology.waterSurface);
            destination.encodedDepth = encodeCanonicalLakeDepth(hydrology.lakeDepth);
            destination.flags = CANONICAL_LAKE_KNOWN;
            if (hydrology.lake) destination.flags |= CANONICAL_LAKE_PRESENT;
            if (hydrology.endorheic) destination.flags |= CANONICAL_LAKE_ENDORHEIC;
        };
        const auto sameLakeSignature = [](const worldgen::HydrologySample& first,
                                          const worldgen::HydrologySample& second) {
            return first.lake == second.lake &&
                   (!first.lake || (first.endorheic == second.endorheic &&
                                    std::abs(first.waterSurface - second.waterSurface) <= 1.0e-4));
        };
        for (int localZ = 0; localZ <= CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX <= CHUNK_EDGE; ++localX) {
                CanonicalLakeSample& destination =
                    canonicalLakes_[static_cast<size_t>(localZ * (CHUNK_EDGE + 1) + localX)];
                const int latticeX =
                    std::min(localX / COLUMN_PLAN_LATTICE_SPACING, COLUMN_PLAN_LATTICE_EDGE - 2);
                const int latticeZ =
                    std::min(localZ / COLUMN_PLAN_LATTICE_SPACING, COLUMN_PLAN_LATTICE_EDGE - 2);
                const std::array<const worldgen::HydrologySample*, 4> corners = {
                    &lattice(latticeX, latticeZ).hydrology,
                    &lattice(latticeX + 1, latticeZ).hydrology,
                    &lattice(latticeX, latticeZ + 1).hydrology,
                    &lattice(latticeX + 1, latticeZ + 1).hydrology,
                };
                const bool uniform =
                    std::all_of(corners.begin() + 1, corners.end(), [&](const auto* value) {
                        return sameLakeSignature(*corners[0], *value);
                    });
                if (uniform) {
                    const double fx =
                        static_cast<double>(localX - latticeX * COLUMN_PLAN_LATTICE_SPACING) /
                        COLUMN_PLAN_LATTICE_SPACING;
                    const double fz =
                        static_cast<double>(localZ - latticeZ * COLUMN_PLAN_LATTICE_SPACING) /
                        COLUMN_PLAN_LATTICE_SPACING;
                    destination.flags = CANONICAL_LAKE_KNOWN;
                    if (corners[0]->lake) destination.flags |= CANONICAL_LAKE_PRESENT;
                    if (corners[0]->endorheic) {
                        destination.flags |= CANONICAL_LAKE_ENDORHEIC;
                    }
                    destination.waterSurface = static_cast<float>(
                        lerp(lerp(corners[0]->waterSurface, corners[1]->waterSurface, fx),
                             lerp(corners[2]->waterSurface, corners[3]->waterSurface, fx), fz));
                    destination.encodedDepth = encodeCanonicalLakeDepth(
                        lerp(lerp(corners[0]->lakeDepth, corners[1]->lakeDepth, fx),
                             lerp(corners[2]->lakeDepth, corners[3]->lakeDepth, fx), fz));
                } else {
                    storeCanonicalLake(destination,
                                       sampleHydrology(baseX + localX, baseZ + localZ));
                }
            }
        }
    }
    if (sampleExactSurface) {
        exactSurfaceY_ = sampleExactSurface(*this);
    } else {
        for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                exactSurfaceY_[static_cast<size_t>(localZ * CHUNK_EDGE + localX)] =
                    static_cast<int16_t>(std::clamp(
                        static_cast<int>(std::floor(sample(localX, localZ).terrainHeight)),
                        WORLD_MIN_Y, WORLD_MAX_Y));
            }
        }
    }
    buildExposedSections(sampleHeight);
}

const worldgen::SurfaceSample& ColumnPlan::lattice(int x, int z) const {
    return lattice_[static_cast<size_t>(z * COLUMN_PLAN_LATTICE_EDGE + x)];
}

int ColumnPlan::surfaceY(int localX, int localZ) const {
    const int clampedX = std::clamp(localX, 0, CHUNK_EDGE - 1);
    const int clampedZ = std::clamp(localZ, 0, CHUNK_EDGE - 1);
    return exactSurfaceY_[static_cast<size_t>(clampedZ * CHUNK_EDGE + clampedX)];
}

worldgen::SurfaceSample ColumnPlan::sample(int localX, int localZ) const {
    const int clampedX = std::clamp(localX, 0, CHUNK_EDGE);
    const int clampedZ = std::clamp(localZ, 0, CHUNK_EDGE);
    if (clampedX % COLUMN_PLAN_LATTICE_SPACING == 0 &&
        clampedZ % COLUMN_PLAN_LATTICE_SPACING == 0) {
        return lattice(clampedX / COLUMN_PLAN_LATTICE_SPACING,
                       clampedZ / COLUMN_PLAN_LATTICE_SPACING);
    }
    const int latticeX =
        std::min(clampedX / COLUMN_PLAN_LATTICE_SPACING, COLUMN_PLAN_LATTICE_EDGE - 2);
    const int latticeZ =
        std::min(clampedZ / COLUMN_PLAN_LATTICE_SPACING, COLUMN_PLAN_LATTICE_EDGE - 2);
    const double fx = static_cast<double>(clampedX - latticeX * COLUMN_PLAN_LATTICE_SPACING) /
                      COLUMN_PLAN_LATTICE_SPACING;
    const double fz = static_cast<double>(clampedZ - latticeZ * COLUMN_PLAN_LATTICE_SPACING) /
                      COLUMN_PLAN_LATTICE_SPACING;
    const std::array<const worldgen::SurfaceSample*, 4> samples = {
        &lattice(latticeX, latticeZ),
        &lattice(latticeX + 1, latticeZ),
        &lattice(latticeX, latticeZ + 1),
        &lattice(latticeX + 1, latticeZ + 1),
    };
    const int nearestX = fx <= 0.5 ? 0 : 1;
    const int nearestZ = fz <= 0.5 ? 0 : 1;
    const worldgen::SurfaceSample& nearest = *samples[static_cast<size_t>(nearestZ * 2 + nearestX)];

    worldgen::SurfaceSample result = nearest;
    auto interpolate = [&](auto getter) { return bilerp(samples, fx, fz, getter); };

    result.geology.plateVelocity = {
        interpolate([](const auto& value) { return value.geology.plateVelocity.x; }),
        interpolate([](const auto& value) { return value.geology.plateVelocity.z; }),
    };
    result.geology.continentalFraction =
        interpolate([](const auto& value) { return value.geology.continentalFraction; });
    result.geology.crustAge = interpolate([](const auto& value) { return value.geology.crustAge; });
    result.geology.crustThickness =
        interpolate([](const auto& value) { return value.geology.crustThickness; });
    result.geology.crustDensity =
        interpolate([](const auto& value) { return value.geology.crustDensity; });
    result.geology.distanceToBoundary =
        interpolate([](const auto& value) { return value.geology.distanceToBoundary; });
    result.geology.uplift = interpolate([](const auto& value) { return value.geology.uplift; });
    result.geology.rift = interpolate([](const auto& value) { return value.geology.rift; });
    result.geology.faultStrength =
        interpolate([](const auto& value) { return value.geology.faultStrength; });
    result.geology.hotspotInfluence =
        interpolate([](const auto& value) { return value.geology.hotspotInfluence; });
    result.geology.volcanicActivity =
        interpolate([](const auto& value) { return value.geology.volcanicActivity; });

    result.hydrology.flowDirection = normalized(
        {interpolate([](const auto& value) { return value.hydrology.flowDirection.x; }),
         interpolate([](const auto& value) { return value.hydrology.flowDirection.z; })});
    result.hydrology.surfaceElevation =
        interpolate([](const auto& value) { return value.hydrology.surfaceElevation; });
    result.hydrology.discharge =
        interpolate([](const auto& value) { return value.hydrology.discharge; });
    result.hydrology.sediment =
        interpolate([](const auto& value) { return value.hydrology.sediment; });
    result.hydrology.channelDistance =
        interpolate([](const auto& value) { return value.hydrology.channelDistance; });
    result.hydrology.channelWidth =
        interpolate([](const auto& value) { return value.hydrology.channelWidth; });
    result.hydrology.channelDepth =
        interpolate([](const auto& value) { return value.hydrology.channelDepth; });
    result.hydrology.channelGradient =
        interpolate([](const auto& value) { return value.hydrology.channelGradient; });
    result.hydrology.erosionDepth =
        interpolate([](const auto& value) { return value.hydrology.erosionDepth; });
    result.hydrology.lakeShoreDistance =
        interpolate([](const auto& value) { return value.hydrology.lakeShoreDistance; });
    result.hydrology.shoreWaterSurface = nearest.hydrology.shoreWaterSurface;
    const bool routedChannel =
        result.hydrology.streamOrder > 0 && result.hydrology.channelWidth > 0.0 &&
        result.hydrology.channelDistance <= result.hydrology.channelWidth * 0.55;
    result.hydrology.ocean = nearest.hydrology.ocean && !routedChannel;
    std::optional<LakeBodySelection> lakeBody;
    const CanonicalLakeSample& canonical =
        canonicalLakes_[static_cast<size_t>(clampedZ * (CHUNK_EDGE + 1) + clampedX)];
    if ((canonical.flags & CANONICAL_LAKE_KNOWN) != 0) {
        if ((canonical.flags & CANONICAL_LAKE_PRESENT) != 0) {
            lakeBody = LakeBodySelection{
                .level = canonical.waterSurface,
                .membership = 1.0,
                .endorheic = (canonical.flags & CANONICAL_LAKE_ENDORHEIC) != 0,
            };
            result.hydrology.ocean = false;
            result.hydrology.surfaceElevation =
                canonical.waterSurface - decodeCanonicalLakeDepth(canonical.encodedDepth);
        }
    } else {
        lakeBody = dominantLakeBody(samples, fx, fz);
    }
    result.hydrology.lake = !result.hydrology.ocean && lakeBody.has_value() &&
                            lakeBody->level > result.hydrology.surfaceElevation + 0.05;
    result.hydrology.lakeDepth =
        result.hydrology.lake ? lakeBody->level - result.hydrology.surfaceElevation : 0.0;
    result.hydrology.river = !result.hydrology.ocean && !result.hydrology.lake && routedChannel;
    result.hydrology.endorheic = result.hydrology.lake && lakeBody->endorheic;
    result.hydrology.waterfall = false;
    result.hydrology.waterfallAnchor = false;
    result.hydrology.waterfallTop = 0.0;
    result.hydrology.waterfallBottom = 0.0;
    result.hydrology.waterfallWidth = 0.0;
    for (const worldgen::SurfaceSample* value : samples) {
        if (!value->hydrology.waterfall) continue;
        result.hydrology.waterfall = true;
        if (value->hydrology.waterfallTop > result.hydrology.waterfallTop) {
            result.hydrology.waterfallTop = value->hydrology.waterfallTop;
            result.hydrology.waterfallBottom = value->hydrology.waterfallBottom;
            result.hydrology.waterfallWidth = value->hydrology.waterfallWidth;
        }
    }
    result.hydrology.delta = std::any_of(samples.begin(), samples.end(),
                                         [](const auto* value) { return value->hydrology.delta; });
    if (result.hydrology.river || result.hydrology.delta) {
        for (const worldgen::SurfaceSample* value : samples) {
            result.hydrology.streamOrder =
                std::max(result.hydrology.streamOrder, value->hydrology.streamOrder);
            result.hydrology.distributaryCount =
                std::max(result.hydrology.distributaryCount, value->hydrology.distributaryCount);
        }
    }
    result.hydrology.waterSurface =
        result.hydrology.lake ? lakeBody->level : waterSurface(samples, result.hydrology, fx, fz);
    const double routedWaterSurface =
        interpolate([](const auto& value) { return value.hydrology.waterSurface; });

    result.climate.wind = {
        interpolate([](const auto& value) { return value.climate.wind.x; }),
        interpolate([](const auto& value) { return value.climate.wind.z; }),
    };
    result.climate.temperatureC =
        interpolate([](const auto& value) { return value.climate.temperatureC; });
    result.climate.annualPrecipitationMm =
        interpolate([](const auto& value) { return value.climate.annualPrecipitationMm; });
    result.climate.potentialEvapotranspirationMm =
        interpolate([](const auto& value) { return value.climate.potentialEvapotranspirationMm; });
    result.climate.aridity = interpolate([](const auto& value) { return value.climate.aridity; });
    result.climate.relativeHumidity =
        interpolate([](const auto& value) { return value.climate.relativeHumidity; });

    result.soil.moisture = interpolate([](const auto& value) { return value.soil.moisture; });
    result.soil.fertility = interpolate([](const auto& value) { return value.soil.fertility; });
    result.soil.drainage = interpolate([](const auto& value) { return value.soil.drainage; });
    result.soil.waterTable = interpolate([](const auto& value) { return value.soil.waterTable; });
    for (size_t index = 0; index < result.suitability.scores.size(); ++index) {
        result.suitability.scores[index] = static_cast<float>(
            interpolate([index](const auto& value) { return value.suitability.scores[index]; }));
    }
    result.biome = worldgen::MacroGenerationSampler::selectBiome(result.suitability);
    result.terrainHeight = interpolate([](const auto& value) { return value.terrainHeight; });
    if (result.hydrology.lake && (canonical.flags & CANONICAL_LAKE_KNOWN) != 0) {
        result.terrainHeight = result.hydrology.surfaceElevation;
    }
    if (!result.hydrology.ocean && !result.hydrology.lake && !result.hydrology.river &&
        result.hydrology.channelWidth > 0.0 &&
        result.hydrology.channelDistance <=
            result.hydrology.channelWidth * 0.55 + worldgen::BASIN_RASTER_SPACING) {
        // Keep interpolated dry channel-edge samples on the same supported
        // bank contract as direct basin samples. This closes the sub-lattice
        // gaps that otherwise appear between a wet river column and lower,
        // categorically dry terrain.
        result.hydrology.surfaceElevation =
            std::max(result.hydrology.surfaceElevation, routedWaterSurface);
        result.terrainHeight = std::max(result.terrainHeight, routedWaterSurface);
    }
    result.hydrology.lakeBank = false;
    result.hydrology.lakeBankTarget = 0.0;
    result.hydrology.lakeBankInfluence = 0.0;
    if (!result.hydrology.ocean && !result.hydrology.lake && !result.hydrology.river &&
        result.hydrology.shoreWaterSurface > 0.0 && result.hydrology.lakeShoreDistance <= 0.0 &&
        result.hydrology.lakeShoreDistance > -16.0) {
        const double distance = -result.hydrology.lakeShoreDistance / 16.0;
        const double smoothDistance = distance * distance * (3.0 - 2.0 * distance);
        result.hydrology.lakeBankInfluence = 1.0 - smoothDistance;
        result.hydrology.lakeBankTarget =
            result.terrainHeight +
            std::max(0.0, std::ceil(result.hydrology.shoreWaterSurface) - result.terrainHeight) *
                result.hydrology.lakeBankInfluence;
        result.terrainHeight = std::max(result.terrainHeight, result.hydrology.lakeBankTarget);
        result.hydrology.surfaceElevation = result.terrainHeight;
        result.hydrology.lakeBank = result.hydrology.lakeBankInfluence > 1.0e-4;
    }
    result.waterSurface = result.hydrology.waterSurface;
    result.slope = interpolate([](const auto& value) { return value.slope; });

    result.ecotopes = worldgen::MacroGenerationSampler::classifyEcotopes(result);
    return result;
}

void ColumnPlan::buildExposedSections(const ColumnPlanHeightSampler& sampleHeight) {
    std::set<int32_t> sections;
    minimumSurfaceY_ = WORLD_MAX_Y;
    maximumSurfaceY_ = WORLD_MIN_Y;

    // A tree anchor can be six blocks outside the target column. Surface
    // samples are bilinearly reconstructed from the world-aligned 8-block
    // lattice, so one transient lattice apron on every side bounds every
    // possible anchor without a per-block climate scan. Only the central
    // 3 by 3 samples remain resident in the immutable plan.
    constexpr int apron =
        (feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH + COLUMN_PLAN_LATTICE_SPACING - 1) /
        COLUMN_PLAN_LATTICE_SPACING;
    double minimumAnchorSurface = std::numeric_limits<double>::infinity();
    double maximumAnchorSurface = -std::numeric_limits<double>::infinity();
    const int64_t baseX = chunkColumn_.x * CHUNK_EDGE;
    const int64_t baseZ = chunkColumn_.z * CHUNK_EDGE;
    for (int latticeZ = -apron; latticeZ < COLUMN_PLAN_LATTICE_EDGE + apron; ++latticeZ) {
        for (int latticeX = -apron; latticeX < COLUMN_PLAN_LATTICE_EDGE + apron; ++latticeX) {
            double height = 0.0;
            if (latticeX >= 0 && latticeX < COLUMN_PLAN_LATTICE_EDGE && latticeZ >= 0 &&
                latticeZ < COLUMN_PLAN_LATTICE_EDGE) {
                height = lattice(latticeX, latticeZ).terrainHeight;
            } else {
                height = sampleHeight(baseX + latticeX * COLUMN_PLAN_LATTICE_SPACING,
                                      baseZ + latticeZ * COLUMN_PLAN_LATTICE_SPACING);
            }
            minimumAnchorSurface = std::min(minimumAnchorSurface, height);
            maximumAnchorSurface = std::max(maximumAnchorSurface, height);
        }
    }

    // Tree emission uses the actual density surface. It accepts only a
    // bounded departure from the sampled terrain, then raises the root one
    // block. Include the complete geometry interval for every possible
    // neighboring anchor, regardless of this column's biome or shore state.
    const int minimumTreeY = static_cast<int>(std::floor(minimumAnchorSurface)) -
                             feature_generation::TREE_MAXIMUM_SURFACE_DEVIATION + 1 +
                             feature_generation::TREE_MINIMUM_VERTICAL_OFFSET;
    const int maximumTreeY = static_cast<int>(std::floor(maximumAnchorSurface)) +
                             feature_generation::TREE_MAXIMUM_SURFACE_DEVIATION + 1 +
                             feature_generation::TREE_MAXIMUM_VERTICAL_OFFSET;
    minimumSurfaceY_ = std::min(minimumSurfaceY_, minimumTreeY);
    maximumSurfaceY_ = std::max(maximumSurfaceY_, maximumTreeY);
    addSectionRange(sections, minimumTreeY, maximumTreeY);

    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const worldgen::SurfaceSample current = sample(localX, localZ);
            const int surfaceY = this->surfaceY(localX, localZ);
            minimumSurfaceY_ = std::min(minimumSurfaceY_, surfaceY);
            maximumSurfaceY_ = std::max(maximumSurfaceY_, surfaceY);
            addSection(sections, surfaceY);
            addSection(sections, surfaceY - 16);
            addSection(sections, surfaceY + 16);

            if (current.geology.volcanicActivity > 0.08 ||
                (current.geology.boundary == worldgen::PlateBoundary::CONVERGENT &&
                 current.geology.distanceToBoundary < 1800.0)) {
                addSection(sections, surfaceY - 32);
                addSection(sections, surfaceY + 32);
            }

            if (current.hydrology.ocean || current.hydrology.river || current.hydrology.lake) {
                addSection(sections, static_cast<int>(std::ceil(current.waterSurface)) - 1);
            }
            if (current.hydrology.waterfall) {
                const double waterfallTop =
                    std::max(current.waterSurface, current.hydrology.waterfallTop);
                const int waterY = static_cast<int>(std::ceil(waterfallTop)) - 1;
                for (int y = surfaceY; y <= waterY; y += CHUNK_EDGE)
                    addSection(sections, y);
                addSection(sections, waterY);
            }

            auto exposeVerticalRange = [&](int adjacentLocalX, int adjacentLocalZ,
                                           const worldgen::SurfaceSample& adjacent) {
                const int adjacentY = adjacentLocalX < CHUNK_EDGE && adjacentLocalZ < CHUNK_EDGE
                                          ? this->surfaceY(adjacentLocalX, adjacentLocalZ)
                                          : static_cast<int>(std::floor(adjacent.terrainHeight));
                const int lowY = std::min(surfaceY, adjacentY);
                const int highY = std::max(surfaceY, adjacentY);
                if (highY - lowY <= 1) return;
                for (int y = lowY; y <= highY; y += CHUNK_EDGE)
                    addSection(sections, y);
                addSection(sections, highY);
            };
            exposeVerticalRange(localX + 1, localZ, sample(localX + 1, localZ));
            exposeVerticalRange(localX, localZ + 1, sample(localX, localZ + 1));
        }
    }
    exposedSections_.assign(sections.begin(), sections.end());
}

bool ColumnPlan::exposesSection(int32_t chunkY) const {
    return std::binary_search(exposedSections_.begin(), exposedSections_.end(), chunkY);
}

class ColumnPlanCache::Impl {
public:
    using PlanPointer = std::shared_ptr<const ColumnPlan>;

    struct Entry {
        std::shared_future<PlanPointer> future;
        uint64_t lastAccess = 0;
        uint64_t token = 0;
    };

    explicit Impl(size_t requestedCapacity) : capacity(std::max<size_t>(1, requestedCapacity)) {}

    mutable std::mutex mutex;
    mutable std::unordered_map<ColumnPos, Entry> entries;
    mutable uint64_t accessClock = 0;
    mutable uint64_t tokenClock = 0;
    size_t capacity;
};

ColumnPlanCache::ColumnPlanCache(size_t capacity) : impl_(std::make_unique<Impl>(capacity)) {}

ColumnPlanCache::~ColumnPlanCache() = default;
ColumnPlanCache::ColumnPlanCache(ColumnPlanCache&&) noexcept = default;
ColumnPlanCache& ColumnPlanCache::operator=(ColumnPlanCache&&) noexcept = default;

std::shared_ptr<const ColumnPlan>
ColumnPlanCache::getOrCreate(ColumnPos chunkColumn, const ColumnPlanSurfaceSampler& sampleSurface,
                             const ColumnPlanHeightSampler& sampleHeight,
                             const ColumnPlanSurfaceGridSampler& sampleExactSurface,
                             const ColumnPlanHydrologySampler& sampleHydrology) const {
    using PlanPointer = Impl::PlanPointer;
    std::shared_future<PlanPointer> future;
    std::shared_ptr<std::promise<PlanPointer>> producer;
    uint64_t token = 0;

    while (true) {
        std::shared_future<PlanPointer> evictionWait;
        {
            std::lock_guard lock(impl_->mutex);
            auto found = impl_->entries.find(chunkColumn);
            if (found != impl_->entries.end()) {
                found->second.lastAccess = ++impl_->accessClock;
                future = found->second.future;
                break;
            }
            if (impl_->entries.size() >= impl_->capacity) {
                auto oldestReady = impl_->entries.end();
                auto oldestPending = impl_->entries.end();
                for (auto entry = impl_->entries.begin(); entry != impl_->entries.end(); ++entry) {
                    if (oldestPending == impl_->entries.end() ||
                        entry->second.lastAccess < oldestPending->second.lastAccess) {
                        oldestPending = entry;
                    }
                    if (entry->second.future.wait_for(std::chrono::seconds(0)) ==
                            std::future_status::ready &&
                        (oldestReady == impl_->entries.end() ||
                         entry->second.lastAccess < oldestReady->second.lastAccess)) {
                        oldestReady = entry;
                    }
                }
                if (oldestReady != impl_->entries.end()) {
                    impl_->entries.erase(oldestReady);
                } else if (oldestPending != impl_->entries.end()) {
                    evictionWait = oldestPending->second.future;
                }
            }
            if (!evictionWait.valid()) {
                producer = std::make_shared<std::promise<PlanPointer>>();
                future = producer->get_future().share();
                token = ++impl_->tokenClock;
                impl_->entries.emplace(chunkColumn,
                                       Impl::Entry{future, ++impl_->accessClock, token});
                break;
            }
        }
        evictionWait.wait();
    }

    if (!producer) return future.get();

    try {
        PlanPointer plan = std::make_shared<ColumnPlan>(chunkColumn, sampleSurface, sampleHeight,
                                                        sampleExactSurface, sampleHydrology);
        producer->set_value(plan);
        return plan;
    } catch (...) {
        producer->set_exception(std::current_exception());
        std::lock_guard lock(impl_->mutex);
        auto found = impl_->entries.find(chunkColumn);
        if (found != impl_->entries.end() && found->second.token == token) {
            impl_->entries.erase(found);
        }
        throw;
    }
}

std::shared_ptr<const ColumnPlan> ColumnPlanCache::find(ColumnPos chunkColumn) const {
    std::shared_future<Impl::PlanPointer> future;
    {
        std::lock_guard lock(impl_->mutex);
        auto found = impl_->entries.find(chunkColumn);
        if (found == impl_->entries.end() ||
            found->second.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return nullptr;
        }
        found->second.lastAccess = ++impl_->accessClock;
        future = found->second.future;
    }
    try {
        return future.get();
    } catch (...) {
        return nullptr;
    }
}

size_t ColumnPlanCache::size() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->entries.size();
}

void ColumnPlanCache::clear() {
    std::lock_guard lock(impl_->mutex);
    impl_->entries.clear();
}
