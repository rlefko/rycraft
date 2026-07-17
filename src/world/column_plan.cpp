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
constexpr uint8_t CANONICAL_LAKE_BANK = 1U << 3U;
constexpr uint8_t CANONICAL_WATER_OCEAN = 1U << 0U;
constexpr uint8_t CANONICAL_WATER_RIVER = 1U << 1U;
constexpr uint8_t CANONICAL_WATER_DELTA = 1U << 2U;
constexpr uint8_t CANONICAL_WATER_WATERFALL = 1U << 3U;
constexpr uint8_t CANONICAL_WATER_WATERFALL_ANCHOR = 1U << 4U;
constexpr uint8_t CANONICAL_WATER_CHANNEL_BANK = 1U << 5U;
constexpr uint8_t CANONICAL_GEOLOGY_SECONDARY_SELECTED = 1U << 7U;
constexpr uint8_t CANONICAL_GEOLOGY_BOUNDARY_MASK =
    static_cast<uint8_t>(~CANONICAL_GEOLOGY_SECONDARY_SELECTED);
constexpr double CANONICAL_GEOLOGY_CONTACT_SCALE = 8.0;

static_assert((CANONICAL_GEOLOGY_BOUNDARY_MASK & CANONICAL_GEOLOGY_SECONDARY_SELECTED) == 0U);
static_assert(static_cast<uint8_t>(worldgen::PlateBoundary::TRANSFORM) <=
              CANONICAL_GEOLOGY_BOUNDARY_MASK);

int16_t encodeShoreDistance(double distance) {
    if (!std::isfinite(distance) || distance <= -4095.0) {
        return std::numeric_limits<int16_t>::min();
    }
    return static_cast<int16_t>(std::clamp(
        std::lround(distance * 8.0), static_cast<long>(std::numeric_limits<int16_t>::min() + 1),
        static_cast<long>(std::numeric_limits<int16_t>::max())));
}

double decodeShoreDistance(int16_t distance) {
    return distance == std::numeric_limits<int16_t>::min() ? -1.0e9
                                                           : static_cast<double>(distance) / 8.0;
}

double lerp(double first, double second, double amount) {
    return first + (second - first) * amount;
}

double smootherstep(double value) {
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

double cubicHermite(double first, double second, double firstDerivative, double secondDerivative,
                    double amount) {
    const double t = std::clamp(amount, 0.0, 1.0);
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * first + (-2.0 * t3 + 3.0 * t2) * second +
           (t3 - 2.0 * t2 + t) * firstDerivative + (t3 - t2) * secondDerivative;
}

int16_t encodeGeologyContactDistance(double distance) {
    const double scaled = std::round(distance * CANONICAL_GEOLOGY_CONTACT_SCALE);
    return static_cast<int16_t>(
        std::clamp(scaled, static_cast<double>(std::numeric_limits<int16_t>::min()),
                   static_cast<double>(std::numeric_limits<int16_t>::max())));
}

int16_t encodeFlowComponent(double component) {
    constexpr double SCALE = static_cast<double>(std::numeric_limits<int16_t>::max());
    return static_cast<int16_t>(std::lround(std::clamp(component, -1.0, 1.0) * SCALE));
}

double decodeFlowComponent(int16_t component) {
    return static_cast<double>(component) /
           static_cast<double>(std::numeric_limits<int16_t>::max());
}

double decodeGeologyContactDistance(int16_t distance) {
    return static_cast<double>(distance) / CANONICAL_GEOLOGY_CONTACT_SCALE;
}

uint16_t encodeGeologyTransition(double transition) {
    return static_cast<uint16_t>(std::round(std::clamp(transition, 0.0, 0.5) * 131070.0));
}

double decodeGeologyTransition(uint16_t transition) {
    return static_cast<double>(transition) / 131070.0;
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
    if (reference.waterBodyId != worldgen::NO_WATER_BODY) {
        return sample.waterBodyId == reference.waterBodyId;
    }
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
    worldgen::WaterBodyId identity = worldgen::NO_WATER_BODY;
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
               ((hydrology.waterBodyId != worldgen::NO_WATER_BODY &&
                 bodies[body].identity != hydrology.waterBodyId) ||
                (hydrology.waterBodyId == worldgen::NO_WATER_BODY &&
                 std::abs(bodies[body].level - hydrology.waterSurface) > 1.0e-4) ||
                bodies[body].endorheic != hydrology.endorheic)) {
            ++body;
        }
        if (body == bodyCount) {
            bodies[bodyCount++] = {
                .identity = hydrology.waterBodyId,
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
                       const ColumnPlanHydrologySampler& sampleHydrology,
                       const ColumnPlanGeologySampler& sampleGeology,
                       worldgen::MacroControlView continuousFields)
    : chunkColumn_(chunkColumn)
    , continuousFields_(std::move(continuousFields)) {
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
    constexpr int TERRAIN_APRON_EDGE = COLUMN_PLAN_LATTICE_EDGE + 2;
    std::array<double, TERRAIN_APRON_EDGE * TERRAIN_APRON_EDGE> terrainApron{};
    const auto apronIndex = [](int latticeX, int latticeZ) {
        return static_cast<size_t>((latticeZ + 1) * TERRAIN_APRON_EDGE + latticeX + 1);
    };
    for (int latticeZ = -1; latticeZ <= COLUMN_PLAN_LATTICE_EDGE; ++latticeZ) {
        for (int latticeX = -1; latticeX <= COLUMN_PLAN_LATTICE_EDGE; ++latticeX) {
            const bool retained = latticeX >= 0 && latticeX < COLUMN_PLAN_LATTICE_EDGE &&
                                  latticeZ >= 0 && latticeZ < COLUMN_PLAN_LATTICE_EDGE;
            terrainApron[apronIndex(latticeX, latticeZ)] =
                retained ? lattice(latticeX, latticeZ).terrainHeight
                         : sampleSurface(baseX + latticeX * COLUMN_PLAN_LATTICE_SPACING,
                                         baseZ + latticeZ * COLUMN_PLAN_LATTICE_SPACING)
                               .terrainHeight;
        }
    }
    constexpr double FIRST_DERIVATIVE_SCALE = 0.5 / COLUMN_PLAN_LATTICE_SPACING;
    constexpr double MIXED_DERIVATIVE_SCALE =
        0.25 / (COLUMN_PLAN_LATTICE_SPACING * COLUMN_PLAN_LATTICE_SPACING);
    for (int latticeZ = 0; latticeZ < COLUMN_PLAN_LATTICE_EDGE; ++latticeZ) {
        for (int latticeX = 0; latticeX < COLUMN_PLAN_LATTICE_EDGE; ++latticeX) {
            TerrainDerivative& derivative = terrainDerivatives_[static_cast<size_t>(
                latticeZ * COLUMN_PLAN_LATTICE_EDGE + latticeX)];
            derivative.x = static_cast<float>((terrainApron[apronIndex(latticeX + 1, latticeZ)] -
                                               terrainApron[apronIndex(latticeX - 1, latticeZ)]) *
                                              FIRST_DERIVATIVE_SCALE);
            derivative.z = static_cast<float>((terrainApron[apronIndex(latticeX, latticeZ + 1)] -
                                               terrainApron[apronIndex(latticeX, latticeZ - 1)]) *
                                              FIRST_DERIVATIVE_SCALE);
            derivative.mixed =
                static_cast<float>((terrainApron[apronIndex(latticeX + 1, latticeZ + 1)] -
                                    terrainApron[apronIndex(latticeX + 1, latticeZ - 1)] -
                                    terrainApron[apronIndex(latticeX - 1, latticeZ + 1)] +
                                    terrainApron[apronIndex(latticeX - 1, latticeZ - 1)]) *
                                   MIXED_DERIVATIVE_SCALE);
        }
    }
    if (sampleHydrology) {
        const auto storeCanonicalLake = [](CanonicalLakeSample& destination,
                                           const worldgen::HydrologySample& hydrology) {
            destination.waterSurface = static_cast<float>(hydrology.waterSurface);
            destination.surfaceElevation = static_cast<float>(hydrology.surfaceElevation);
            destination.encodedShoreDistance = encodeShoreDistance(hydrology.lakeShoreDistance);
            destination.encodedBankInfluence = static_cast<uint8_t>(
                std::lround(std::clamp(hydrology.lakeBankInfluence, 0.0, 1.0) * 255.0));
            destination.flags = CANONICAL_LAKE_KNOWN;
            if (hydrology.lake) destination.flags |= CANONICAL_LAKE_PRESENT;
            if (hydrology.endorheic) destination.flags |= CANONICAL_LAKE_ENDORHEIC;
            if (hydrology.lakeBank) destination.flags |= CANONICAL_LAKE_BANK;
        };
        waterBodyPalette_.push_back(worldgen::NO_WATER_BODY);
        for (int localZ = 0; localZ <= CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX <= CHUNK_EDGE; ++localX) {
                const size_t index = static_cast<size_t>(localZ * (CHUNK_EDGE + 1) + localX);
                CanonicalLakeSample& destination = canonicalLakes_[index];
                const worldgen::HydrologySample hydrology =
                    sampleHydrology(baseX + localX, baseZ + localZ);
                storeCanonicalLake(destination, hydrology);
                generatedTopFluidStates_[index] =
                    FluidState::flowing(hydrology.generatedFluidLevel).packed();
                if (hydrology.generatedFluidLevel == 0)
                    generatedTopFluidStates_[index] = FluidState::source().packed();
                if (hydrology.transitionOwnerId != 0) {
                    auto owner =
                        std::find(transitionOwnerPalette_.begin(), transitionOwnerPalette_.end(),
                                  hydrology.transitionOwnerId);
                    if (owner == transitionOwnerPalette_.end()) {
                        transitionOwnerPalette_.push_back(hydrology.transitionOwnerId);
                        owner = std::prev(transitionOwnerPalette_.end());
                    }
                    canonicalTransitions_.push_back({
                        .localIndex = static_cast<uint16_t>(index),
                        .ownerPaletteIndex = static_cast<uint16_t>(
                            std::distance(transitionOwnerPalette_.begin(), owner)),
                        .ownerKind = hydrology.transitionOwnerKind,
                    });
                }
                auto body = std::find(waterBodyPalette_.begin(), waterBodyPalette_.end(),
                                      hydrology.waterBodyId);
                if (body == waterBodyPalette_.end()) {
                    waterBodyPalette_.push_back(hydrology.waterBodyId);
                    body = std::prev(waterBodyPalette_.end());
                }
                waterBodyIndices_[index] =
                    static_cast<uint16_t>(std::distance(waterBodyPalette_.begin(), body));
                uint8_t topology = 0;
                if (hydrology.ocean) topology |= CANONICAL_WATER_OCEAN;
                if (hydrology.river) topology |= CANONICAL_WATER_RIVER;
                if (hydrology.delta) topology |= CANONICAL_WATER_DELTA;
                if (hydrology.waterfall) topology |= CANONICAL_WATER_WATERFALL;
                if (hydrology.waterfallAnchor) topology |= CANONICAL_WATER_WATERFALL_ANCHOR;
                if (hydrology.channelBank) topology |= CANONICAL_WATER_CHANNEL_BANK;
                waterTopologyFlags_[index] = topology;
                if (hydrology.waterfall) {
                    const worldgen::Vector2d flow = normalized(hydrology.flowDirection);
                    canonicalWaterfalls_.push_back({
                        .localIndex = static_cast<uint16_t>(index),
                        .encodedFlowX = encodeFlowComponent(flow.x),
                        .encodedFlowZ = encodeFlowComponent(flow.z),
                        .top = static_cast<float>(hydrology.waterfallTop),
                        .bottom = static_cast<float>(hydrology.waterfallBottom),
                        .width = static_cast<float>(hydrology.waterfallWidth),
                    });
                    maximumWaterfallTop_ =
                        std::max(maximumWaterfallTop_, static_cast<float>(hydrology.waterfallTop));
                }
            }
        }
    }
    if (sampleGeology) {
        hasCanonicalGeology_ = true;
        for (int localZ = 0; localZ <= CHUNK_EDGE; ++localZ) {
            for (int localX = 0; localX <= CHUNK_EDGE; ++localX) {
                const worldgen::GeologySample geology =
                    sampleGeology(baseX + localX, baseZ + localZ);
                CanonicalGeologySample& destination =
                    canonicalGeology_[static_cast<size_t>(localZ * (CHUNK_EDGE + 1) + localX)];
                destination.plateId = geology.plateId;
                destination.encodedContactDistance =
                    encodeGeologyContactDistance(geology.lithology.contactDistance);
                destination.encodedTransition =
                    encodeGeologyTransition(geology.lithology.transition);
                destination.primaryRock = static_cast<uint8_t>(geology.lithology.primary);
                destination.secondaryRock = static_cast<uint8_t>(geology.lithology.secondary);
                destination.crust = static_cast<uint8_t>(geology.crust);
                destination.boundary = static_cast<uint8_t>(geology.boundary);
                if (geology.rock == geology.lithology.secondary &&
                    geology.lithology.secondary != geology.lithology.primary) {
                    destination.boundary |= CANONICAL_GEOLOGY_SECONDARY_SELECTED;
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

const ColumnPlan::TerrainDerivative& ColumnPlan::terrainDerivative(int x, int z) const {
    return terrainDerivatives_[static_cast<size_t>(z * COLUMN_PLAN_LATTICE_EDGE + x)];
}

int ColumnPlan::surfaceY(int localX, int localZ) const {
    const int clampedX = std::clamp(localX, 0, CHUNK_EDGE - 1);
    const int clampedZ = std::clamp(localZ, 0, CHUNK_EDGE - 1);
    return exactSurfaceY_[static_cast<size_t>(clampedZ * CHUNK_EDGE + clampedX)];
}

worldgen::SurfaceSample ColumnPlan::sample(int localX, int localZ) const {
    const int clampedX = std::clamp(localX, 0, CHUNK_EDGE);
    const int clampedZ = std::clamp(localZ, 0, CHUNK_EDGE);
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
    const double smoothFx = smootherstep(fx);
    const double smoothFz = smootherstep(fz);
    auto interpolate = [&](auto getter) { return bilerp(samples, smoothFx, smoothFz, getter); };

    if (hasCanonicalGeology_) {
        const CanonicalGeologySample& geology =
            canonicalGeology_[static_cast<size_t>(clampedZ * (CHUNK_EDGE + 1) + clampedX)];
        result.geology.plateId = geology.plateId;
        result.geology.crust = static_cast<worldgen::CrustType>(geology.crust);
        result.geology.boundary = static_cast<worldgen::PlateBoundary>(
            geology.boundary & CANONICAL_GEOLOGY_BOUNDARY_MASK);
        result.geology.lithology.primary = static_cast<worldgen::RockType>(geology.primaryRock);
        result.geology.lithology.secondary = static_cast<worldgen::RockType>(geology.secondaryRock);
        result.geology.lithology.transition = decodeGeologyTransition(geology.encodedTransition);
        result.geology.lithology.contactDistance =
            decodeGeologyContactDistance(geology.encodedContactDistance);
        // The high bit records the direct sampler's resolved facies without
        // growing the compact authority structure.
        result.geology.rock = (geology.boundary & CANONICAL_GEOLOGY_SECONDARY_SELECTED) != 0
                                  ? result.geology.lithology.secondary
                                  : result.geology.lithology.primary;
    }

    if (!continuousFields_) {
        result.geology.plateVelocity = {
            interpolate([](const auto& value) { return value.geology.plateVelocity.x; }),
            interpolate([](const auto& value) { return value.geology.plateVelocity.z; }),
        };
        result.geology.continentalFraction =
            interpolate([](const auto& value) { return value.geology.continentalFraction; });
        result.geology.crustAge =
            interpolate([](const auto& value) { return value.geology.crustAge; });
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
    }

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
    const bool interpolatedRoutedChannel =
        result.hydrology.streamOrder > 0 && result.hydrology.channelWidth > 0.0 &&
        result.hydrology.channelDistance <= result.hydrology.channelWidth * 0.55;
    std::optional<LakeBodySelection> lakeBody;
    const size_t canonicalIndex = static_cast<size_t>(clampedZ * (CHUNK_EDGE + 1) + clampedX);
    const CanonicalLakeSample& canonical = canonicalLakes_[canonicalIndex];
    const uint8_t canonicalTopology = waterTopologyFlags_[canonicalIndex];
    const FluidState canonicalTopFluidState =
        FluidState::fromPacked(generatedTopFluidStates_[canonicalIndex]);
    const uint16_t canonicalWaterBodyIndex = waterBodyIndices_[canonicalIndex];
    const worldgen::WaterBodyId canonicalWaterBodyId =
        canonicalWaterBodyIndex < waterBodyPalette_.size()
            ? waterBodyPalette_[canonicalWaterBodyIndex]
            : worldgen::NO_WATER_BODY;
    const bool canonicalWaterKnown = (canonical.flags & CANONICAL_LAKE_KNOWN) != 0;
    const bool canonicalLake = (canonical.flags & CANONICAL_LAKE_PRESENT) != 0;
    const bool canonicalOcean = (canonicalTopology & CANONICAL_WATER_OCEAN) != 0;
    const bool canonicalRiver = (canonicalTopology & CANONICAL_WATER_RIVER) != 0;
    const bool canonicalDelta = (canonicalTopology & CANONICAL_WATER_DELTA) != 0;
    const bool canonicalWaterfall = (canonicalTopology & CANONICAL_WATER_WATERFALL) != 0;
    const bool canonicalWaterfallAnchor =
        (canonicalTopology & CANONICAL_WATER_WATERFALL_ANCHOR) != 0;
    const bool canonicalChannelBank = (canonicalTopology & CANONICAL_WATER_CHANNEL_BANK) != 0;
    const bool canonicalLakeBank = (canonical.flags & CANONICAL_LAKE_BANK) != 0;
    if (canonicalWaterKnown) {
        result.hydrology.generatedFluidLevel =
            canonicalTopFluidState.isSource() ? 0 : canonicalTopFluidState.level();
        const auto transition = std::lower_bound(
            canonicalTransitions_.begin(), canonicalTransitions_.end(), canonicalIndex,
            [](const CanonicalTransitionSample& candidate, size_t index) {
                return candidate.localIndex < index;
            });
        if (transition != canonicalTransitions_.end() && transition->localIndex == canonicalIndex &&
            transition->ownerPaletteIndex < transitionOwnerPalette_.size()) {
            result.hydrology.transitionOwnerKind = transition->ownerKind;
            result.hydrology.transitionOwnerId =
                transitionOwnerPalette_[transition->ownerPaletteIndex];
        } else {
            result.hydrology.transitionOwnerKind = worldgen::WaterTransitionKind::NONE;
            result.hydrology.transitionOwnerId = 0;
        }
        result.hydrology.waterBodyId = canonicalWaterBodyId;
        result.hydrology.lakeShoreDistance = decodeShoreDistance(canonical.encodedShoreDistance);
        result.hydrology.shoreWaterSurface = canonical.waterSurface;
        // The 17 by 17 canonical authority is already evaluated at block
        // resolution. Use its unmodified macro floor for every category, not
        // only wet cells, so final terrain retains sub-control detail. The
        // generator applies coordinate-pure volcanic geometry afterward.
        result.hydrology.surfaceElevation = canonical.surfaceElevation;
        result.terrainHeight = canonical.surfaceElevation;
        result.hydrology.ocean = canonicalOcean;
        result.hydrology.channelBank = canonicalChannelBank;
        if (canonicalLake) {
            lakeBody = LakeBodySelection{
                .identity = canonicalWaterBodyId,
                .level = canonical.waterSurface,
                .membership = 1.0,
                .endorheic = (canonical.flags & CANONICAL_LAKE_ENDORHEIC) != 0,
            };
            result.hydrology.ocean = false;
        }
    } else {
        result.hydrology.ocean = nearest.hydrology.ocean && !interpolatedRoutedChannel;
        lakeBody = dominantLakeBody(samples, fx, fz);
    }
    result.hydrology.lake = canonicalWaterKnown
                                ? !result.hydrology.ocean && canonicalLake &&
                                      canonical.waterSurface > canonical.surfaceElevation + 0.05
                                : !result.hydrology.ocean && lakeBody.has_value() &&
                                      lakeBody->level > result.hydrology.surfaceElevation + 0.05;
    if (!canonicalWaterKnown) {
        result.hydrology.waterBodyId =
            result.hydrology.lake ? lakeBody->identity : nearest.hydrology.waterBodyId;
    }
    result.hydrology.lakeDepth =
        result.hydrology.lake ? lakeBody->level - result.hydrology.surfaceElevation : 0.0;
    result.hydrology.river = !result.hydrology.ocean && !result.hydrology.lake &&
                             (canonicalWaterKnown ? canonicalRiver : interpolatedRoutedChannel);
    result.hydrology.endorheic =
        result.hydrology.lake &&
        (canonicalWaterKnown ? (canonical.flags & CANONICAL_LAKE_ENDORHEIC) != 0
                             : lakeBody->endorheic);
    result.hydrology.waterfall = canonicalWaterKnown && canonicalWaterfall;
    result.hydrology.waterfallAnchor = canonicalWaterKnown && canonicalWaterfallAnchor;
    result.hydrology.waterfallTop = 0.0;
    result.hydrology.waterfallBottom = 0.0;
    result.hydrology.waterfallWidth = 0.0;
    if (canonicalWaterfall) {
        const auto waterfall = std::lower_bound(
            canonicalWaterfalls_.begin(), canonicalWaterfalls_.end(), canonicalIndex,
            [](const CanonicalWaterfallSample& candidate, size_t index) {
                return candidate.localIndex < index;
            });
        if (waterfall != canonicalWaterfalls_.end() && waterfall->localIndex == canonicalIndex) {
            result.hydrology.flowDirection =
                normalized({decodeFlowComponent(waterfall->encodedFlowX),
                            decodeFlowComponent(waterfall->encodedFlowZ)});
            result.hydrology.waterfallTop = waterfall->top;
            result.hydrology.waterfallBottom = waterfall->bottom;
            result.hydrology.waterfallWidth = waterfall->width;
        }
    } else if (!canonicalWaterKnown) {
        for (const worldgen::SurfaceSample* value : samples) {
            if (!value->hydrology.waterfall) continue;
            result.hydrology.waterfall = true;
            if (value->hydrology.waterfallTop > result.hydrology.waterfallTop) {
                result.hydrology.waterfallTop = value->hydrology.waterfallTop;
                result.hydrology.waterfallBottom = value->hydrology.waterfallBottom;
                result.hydrology.waterfallWidth = value->hydrology.waterfallWidth;
            }
        }
    }
    result.hydrology.delta =
        canonicalWaterKnown ? canonicalDelta
                            : std::any_of(samples.begin(), samples.end(),
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
        canonicalWaterKnown
            ? canonical.waterSurface
            : (result.hydrology.lake ? lakeBody->level
                                     : waterSurface(samples, result.hydrology, fx, fz));
    if (!continuousFields_) {
        result.climate.wind = {
            interpolate([](const auto& value) { return value.climate.wind.x; }),
            interpolate([](const auto& value) { return value.climate.wind.z; }),
        };
        result.climate.temperatureC =
            interpolate([](const auto& value) { return value.climate.temperatureC; });
        result.climate.annualPrecipitationMm =
            interpolate([](const auto& value) { return value.climate.annualPrecipitationMm; });
        result.climate.potentialEvapotranspirationMm = interpolate(
            [](const auto& value) { return value.climate.potentialEvapotranspirationMm; });
        result.climate.aridity =
            interpolate([](const auto& value) { return value.climate.aridity; });
        result.climate.relativeHumidity =
            interpolate([](const auto& value) { return value.climate.relativeHumidity; });

        result.soil.moisture = interpolate([](const auto& value) { return value.soil.moisture; });
        result.soil.fertility = interpolate([](const auto& value) { return value.soil.fertility; });
        result.soil.drainage = interpolate([](const auto& value) { return value.soil.drainage; });
        result.soil.waterTable =
            interpolate([](const auto& value) { return value.soil.waterTable; });
        for (size_t index = 0; index < result.suitability.scores.size(); ++index) {
            result.suitability.scores[index] = static_cast<float>(interpolate(
                [index](const auto& value) { return value.suitability.scores[index]; }));
        }
        result.biome = worldgen::MacroGenerationSampler::selectBiome(result.suitability);
    }
    const TerrainDerivative& derivative00 = terrainDerivative(latticeX, latticeZ);
    const TerrainDerivative& derivative10 = terrainDerivative(latticeX + 1, latticeZ);
    const TerrainDerivative& derivative01 = terrainDerivative(latticeX, latticeZ + 1);
    const TerrainDerivative& derivative11 = terrainDerivative(latticeX + 1, latticeZ + 1);
    constexpr double SPACING = COLUMN_PLAN_LATTICE_SPACING;
    const double lowerTerrain =
        cubicHermite(samples[0]->terrainHeight, samples[1]->terrainHeight, derivative00.x * SPACING,
                     derivative10.x * SPACING, fx);
    const double upperTerrain =
        cubicHermite(samples[2]->terrainHeight, samples[3]->terrainHeight, derivative01.x * SPACING,
                     derivative11.x * SPACING, fx);
    const double lowerZDerivative =
        cubicHermite(derivative00.z, derivative10.z, derivative00.mixed * SPACING,
                     derivative10.mixed * SPACING, fx);
    const double upperZDerivative =
        cubicHermite(derivative01.z, derivative11.z, derivative01.mixed * SPACING,
                     derivative11.mixed * SPACING, fx);
    if (!canonicalWaterKnown) {
        result.terrainHeight = cubicHermite(lowerTerrain, upperTerrain, lowerZDerivative * SPACING,
                                            upperZDerivative * SPACING, fz);
    }
    if (canonicalWaterKnown) {
        result.hydrology.lakeBank = canonicalLakeBank;
        result.hydrology.lakeBankInfluence =
            static_cast<double>(canonical.encodedBankInfluence) / 255.0;
        result.hydrology.lakeBankTarget =
            result.hydrology.lakeBank ? canonical.surfaceElevation : 0.0;
    } else if (result.hydrology.lakeBank) {
        // Without exact authority, retain the sampled bank as descriptive
        // metadata only. The reconstructed terrain is the physical substrate;
        // a water stage must never raise it after hydrology has been sampled.
        result.hydrology.lakeBankInfluence =
            std::clamp(result.hydrology.lakeBankInfluence, 0.0, 1.0);
        result.hydrology.lakeBankTarget = result.terrainHeight;
    }
    result.waterSurface = result.hydrology.waterSurface;
    if (!continuousFields_) {
        result.slope = interpolate([](const auto& value) { return value.slope; });
    }

    const int64_t worldX = chunkColumn_.x * CHUNK_EDGE + clampedX;
    const int64_t worldZ = chunkColumn_.z * CHUNK_EDGE + clampedZ;
    continuousFields_.reconstructContinuous(static_cast<double>(worldX),
                                            static_cast<double>(worldZ), result);

    result.ecotopes = worldgen::MacroGenerationSampler::classifyEcotopes(result);
    return result;
}

void ColumnPlan::buildExposedSections(const ColumnPlanHeightSampler& sampleHeight) {
    std::set<int32_t> sections;
    minimumSurfaceY_ = WORLD_MAX_Y;
    maximumSurfaceY_ = WORLD_MIN_Y;

    // A tree anchor can be six blocks outside the target column. Surface
    // samples are bicubically reconstructed from the world-aligned 8-block
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
                    std::max({current.waterSurface, current.hydrology.waterfallTop,
                              static_cast<double>(maximumWaterfallTop_)});
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
                             const ColumnPlanHydrologySampler& sampleHydrology,
                             const ColumnPlanGeologySampler& sampleGeology,
                             worldgen::MacroControlView continuousFields) const {
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
                                                        sampleExactSurface, sampleHydrology,
                                                        sampleGeology, std::move(continuousFields));
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
