#include "world/macro_generation.hpp"

#include "world/learned_terrain.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace worldgen {
namespace {

constexpr double PLATE_SCALE = 8192.0;
constexpr double DRAINAGE_SCALE = 2048.0;
constexpr int MOISTURE_STEPS = 16;
constexpr double MOISTURE_STEP_DISTANCE = 256.0;
constexpr int64_t V4_VOLCANIC_ARC_CELL_EDGE = 1'024;
constexpr int V4_HOTSPOT_QUERY_RADIUS = 1;
constexpr uint64_t V4_VOLCANIC_ARC_STREAM = 0x564F'4C43'4152'4331ULL;

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

struct V4VolcanicPrimitive {
    double centerX = 0.0;
    double centerZ = 0.0;
    double radius = 1.0;
    double coneHeightBlocks = 0.0;
    double craterRadius = 0.0;
    double craterDepthBlocks = 0.0;
    double radialPhase = 0.0;
    uint64_t id = 0;
    bool shield = false;
};

struct ThreadV4VolcanicPrimitiveCache {
    const MacroGenerationSampler* owner = nullptr;
    uint64_t tag = 0;
    std::unordered_map<ColumnPos, std::vector<V4VolcanicPrimitive>> hotspotCells;
    std::unordered_map<ColumnPos, std::vector<V4VolcanicPrimitive>> arcCells;
    ColumnPos hotspotNeighborhood{};
    ColumnPos arcNeighborhood{};
    std::vector<V4VolcanicPrimitive> hotspotCandidates;
    std::vector<V4VolcanicPrimitive> arcCandidates;
    bool hasHotspotNeighborhood = false;
    bool hasArcNeighborhood = false;
};

ThreadV4VolcanicPrimitiveCache& v4VolcanicPrimitiveCache(const MacroGenerationSampler* owner,
                                                         uint64_t tag) {
    thread_local ThreadV4VolcanicPrimitiveCache cache;
    if (cache.owner != owner || cache.tag != tag) {
        cache.owner = owner;
        cache.tag = tag;
        cache.hotspotCells.clear();
        cache.arcCells.clear();
        cache.hotspotCandidates.clear();
        cache.arcCandidates.clear();
        cache.hasHotspotNeighborhood = false;
        cache.hasArcNeighborhood = false;
    }
    return cache;
}

double v4VolcanoProfile(const V4VolcanicPrimitive& volcano, double radialDistance) {
    const double normalized = radialDistance / std::max(1.0, volcano.radius);
    return volcano.shield ? std::pow(std::max(0.0, 1.0 - normalized * normalized), 2.0)
                          : std::pow(std::max(0.0, 1.0 - normalized), 1.28);
}

double v4VolcanoCrater(const V4VolcanicPrimitive& volcano, double radialDistance) {
    return radialDistance < volcano.craterRadius
               ? 1.0 - smoothstep(0.12, 1.0, radialDistance / volcano.craterRadius)
               : 0.0;
}

double v4VolcanoWarpedDistance(const V4VolcanicPrimitive& volcano, double offsetX, double offsetZ) {
    const double radialDistance = std::hypot(offsetX, offsetZ);
    if (radialDistance < 1.0e-9) return 0.0;
    const double angle = std::atan2(offsetZ, offsetX);
    const double idPhase =
        static_cast<double>((volcano.id >> 17U) & 0xFFFFU) / 65535.0 * 2.0 * std::numbers::pi;
    const double radialPhase = radialDistance / std::max(24.0, volcano.craterRadius * 0.72);
    const double scale =
        std::clamp(1.0 + std::sin(angle * 3.0 + volcano.radialPhase) * 0.052 +
                       std::sin(angle * 5.0 + idPhase) * 0.027 +
                       std::sin(angle * 2.0 + radialPhase + idPhase * 0.61) * 0.018,
                   0.88, 1.12);
    return radialDistance / scale;
}

double dryReliefDetailScale(const HydrologySample& hydrology) {
    // A volcano or other final deformation can move a formerly dry coastal
    // column below sea level. Blend its relief amplitude before that
    // deformation so the new ocean floor meets the existing floor without a
    // categorical dry-to-ocean detail seam.
    constexpr double COASTAL_DETAIL_BLEND_HEIGHT = 16.0;
    const double inland =
        smoothstep(SEA_LEVEL, SEA_LEVEL + COASTAL_DETAIL_BLEND_HEIGHT, hydrology.surfaceElevation);
    return std::lerp(OCEAN_FLOOR_DETAIL_SCALE, DRY_RELIEF_DETAIL_SCALE, inland);
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
    constexpr double MINIMUM = -0x1p63;
    constexpr double MAXIMUM_EXCLUSIVE = 0x1p63;
    if (std::isnan(value)) return 0;
    // The positive int64 limit is not representable as a double: converting
    // INT64_MAX rounds up to 2^63 and makes the subsequent cast undefined.
    // Keep two cells of headroom because simplex and feature queries address
    // neighboring lattice points after this conversion.
    if (value <= MINIMUM) return std::numeric_limits<int64_t>::min() + 2;
    if (value >= MAXIMUM_EXCLUSIVE) return std::numeric_limits<int64_t>::max() - 2;
    return static_cast<int64_t>(std::floor(value));
}

constexpr int64_t MAXIMUM_EXACT_DOUBLE_INTEGER = 9'007'199'254'740'992LL;

bool nativeHydrologyCertificatePositionIsExact(ColumnPos position) noexcept {
    const auto coordinateIsExact = [](int64_t coordinate) {
        return coordinate >= -MAXIMUM_EXACT_DOUBLE_INTEGER &&
               coordinate <= MAXIMUM_EXACT_DOUBLE_INTEGER;
    };
    return coordinateIsExact(position.x) && coordinateIsExact(position.z);
}

bool nativeHydrologyCertificatePositionsAreExact(std::span<const ColumnPos> positions) noexcept {
    return std::ranges::all_of(positions, nativeHydrologyCertificatePositionIsExact);
}

learned::PhysicalTerrainSample
requireLearnedTerrain(const std::shared_ptr<learned::WorldGenerationContext>& context, double x,
                      double z) {
    auto result = context->sampleWorld(floorToInt64(x), floorToInt64(z));
    if (result.isReady()) return *result.value();
    learned::GenerationFailure failure =
        result.failure() != nullptr
            ? *result.failure()
            : learned::GenerationFailure{
                  .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                  .message = "Learned terrain query returned no value or failure",
                  .retriable = true,
              };
    if (result.status() == learned::AuthorityStatus::FAILED) context->latchFailure(failure);
    throw learned::GenerationFailureException(result.status(), std::move(failure));
}

std::vector<learned::PhysicalTerrainSample> requireLearnedTerrainPoints(
    const std::shared_ptr<learned::WorldGenerationContext>& context,
    std::span<const learned::WorldBlockPoint> points,
    std::optional<learned::AuthorityRequestPriority> priority = std::nullopt) {
    auto result = context->queryWorldPoints(points, priority);
    if (result.isReady()) return std::move(*result.value());
    learned::GenerationFailure failure =
        result.failure() != nullptr
            ? *result.failure()
            : learned::GenerationFailure{
                  .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                  .message = "Learned terrain point query returned no value or failure",
                  .retriable = true,
              };
    if (result.status() == learned::AuthorityStatus::FAILED) context->latchFailure(failure);
    throw learned::GenerationFailureException(result.status(), std::move(failure));
}

// The native four-block water raster lands on integer model-pixel boundaries.
// align_corners=false still places a block sample 0.375 native pixels before
// that boundary, so a direct native sample would move the final coastline.
// Query the compact (width + 1) by (height + 1) stencil instead and reproduce
// the ordinary world-point bilinear reconstruction without the per-point map
// and duplicate-point collection in WorldGenerationContext::queryWorldPoints.
// This is deliberately limited to the native water raster. Other callers keep
// their general world-point query contract and metrics unchanged.
bool isNativeAlignedWaterGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                              int sampleWidth, int sampleHeight) noexcept {
    constexpr int64_t SCALE = learned::MODEL_BLOCK_SCALE;
    if (spacingX != SCALE || spacingZ != SCALE || sampleWidth <= 0 || sampleHeight <= 0 ||
        originX % SCALE != 0 || originZ % SCALE != 0) {
        return false;
    }
    const uint64_t nativeWidth = static_cast<uint64_t>(sampleWidth) + 1U;
    const uint64_t nativeHeight = static_cast<uint64_t>(sampleHeight) + 1U;
    return nativeWidth <= learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES / nativeHeight;
}

std::vector<int64_t> nativeHydrologyOwnersOnGridAxis(int64_t origin, int spacing, int sampleCount) {
    std::vector<int64_t> owners;
    owners.reserve(std::min(sampleCount, 8));
    for (int index = 0; index < sampleCount; ++index) {
        const __int128 coordinate =
            static_cast<__int128>(origin) + static_cast<__int128>(index) * spacing;
        if (coordinate < std::numeric_limits<int64_t>::min() ||
            coordinate > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("Native hydrology grid owner coordinate is out of range");
        }
        const int64_t owner = world_coord::floorDiv(
            static_cast<int64_t>(coordinate), static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
        if (owners.empty() || owners.back() != owner) owners.push_back(owner);
    }
    return owners;
}

void recordPreparedNativeHydrologyPointOwners(
    const std::shared_ptr<learned::WorldGenerationContext>& context,
    std::span<const ColumnPos> positions, std::span<const uint8_t> certifiedDryHits = {});

void recordPreparedNativeHydrologyGridOwners(
    const std::shared_ptr<learned::WorldGenerationContext>& context, int64_t originX,
    int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<const uint8_t> certifiedDryHits = {}) {
    if (!context) return;
    if (!certifiedDryHits.empty()) {
        const size_t sampleCount = static_cast<size_t>(sampleWidth) * sampleHeight;
        if (certifiedDryHits.size() != sampleCount)
            throw std::logic_error("invalid native hydrology readiness mask");
        std::vector<ColumnPos> fallbackPositions;
        fallbackPositions.reserve(sampleCount);
        for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                const size_t index = static_cast<size_t>(sampleZ) * sampleWidth + sampleX;
                if (certifiedDryHits[index] != 0) continue;
                fallbackPositions.emplace_back(originX + static_cast<int64_t>(sampleX) * spacingX,
                                               originZ + static_cast<int64_t>(sampleZ) * spacingZ);
            }
        }
        recordPreparedNativeHydrologyPointOwners(context, fallbackPositions);
        return;
    }
    const std::vector<int64_t> ownerXs =
        nativeHydrologyOwnersOnGridAxis(originX, spacingX, sampleWidth);
    const std::vector<int64_t> ownerZs =
        nativeHydrologyOwnersOnGridAxis(originZ, spacingZ, sampleHeight);
    for (const int64_t ownerZ : ownerZs) {
        for (const int64_t ownerX : ownerXs)
            context->recordPreparedNativeHydrologyOwner(ownerX, ownerZ);
    }
}

void recordPreparedNativeHydrologyPointOwners(
    const std::shared_ptr<learned::WorldGenerationContext>& context,
    std::span<const ColumnPos> positions, std::span<const uint8_t> certifiedDryHits) {
    if (!context) return;
    if (!certifiedDryHits.empty() && certifiedDryHits.size() != positions.size())
        throw std::logic_error("invalid native hydrology readiness mask");
    std::vector<ColumnPos> owners;
    owners.reserve(positions.size());
    for (size_t index = 0; index < positions.size(); ++index) {
        if (!certifiedDryHits.empty() && certifiedDryHits[index] != 0) continue;
        const ColumnPos position = positions[index];
        owners.emplace_back(
            world_coord::floorDiv(position.x, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
            world_coord::floorDiv(position.z, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)));
    }
    std::ranges::sort(owners, [](ColumnPos left, ColumnPos right) {
        return left.x < right.x || (left.x == right.x && left.z < right.z);
    });
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
    for (const ColumnPos owner : owners)
        context->recordPreparedNativeHydrologyOwner(owner.x, owner.z);
}

std::vector<learned::PhysicalTerrainSample> requireLearnedTerrainNativeAlignedWaterGrid(
    const std::shared_ptr<learned::WorldGenerationContext>& context, int64_t originX,
    int64_t originZ, int sampleWidth, int sampleHeight) {
    constexpr int64_t SCALE = learned::MODEL_BLOCK_SCALE;
    constexpr double FRACTION = 0.625;
    const int64_t nativeColumn = originX / SCALE;
    const int64_t nativeRow = originZ / SCALE;
    const learned::NativeRect region{
        .rowBegin = nativeRow - 1,
        .columnBegin = nativeColumn - 1,
        .rowEnd = nativeRow + sampleHeight,
        .columnEnd = nativeColumn + sampleWidth,
    };
    auto result = context->queryNative(region);
    if (!result.isReady()) {
        learned::GenerationFailure failure =
            result.failure() != nullptr
                ? *result.failure()
                : learned::GenerationFailure{
                      .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                      .message = "Learned native water-grid query returned no value or failure",
                      .retriable = true,
                  };
        if (result.status() == learned::AuthorityStatus::FAILED) context->latchFailure(failure);
        throw learned::GenerationFailureException(result.status(), std::move(failure));
    }

    const learned::PhysicalTerrainGrid& grid = *result.value();
    if (!grid.valid()) {
        learned::GenerationFailure failure{
            .code = learned::GenerationFailureCode::CORRUPT_PAGE,
            .message = "Learned native water-grid query returned an invalid grid",
            .retriable = true,
        };
        context->latchFailure(failure);
        throw learned::GenerationFailureException(learned::AuthorityStatus::FAILED,
                                                  std::move(failure));
    }

    const size_t nativeWidth = static_cast<size_t>(sampleWidth) + 1U;
    std::vector<learned::PhysicalTerrainSample> output;
    output.reserve(static_cast<size_t>(sampleWidth) * static_cast<size_t>(sampleHeight));
    const auto sampleAt = [&](int row, int column) -> const learned::PhysicalTerrainSample& {
        return grid.samples[static_cast<size_t>(row) * nativeWidth + static_cast<size_t>(column)];
    };
    const auto bilinear = [&](int row, int column, auto member) {
        const double north =
            std::lerp(sampleAt(row, column).*member, sampleAt(row, column + 1).*member, FRACTION);
        const double south = std::lerp(sampleAt(row + 1, column).*member,
                                       sampleAt(row + 1, column + 1).*member, FRACTION);
        return std::lerp(north, south, FRACTION);
    };
    for (int row = 0; row < sampleHeight; ++row) {
        for (int column = 0; column < sampleWidth; ++column) {
            output.push_back({
                .elevationMeters =
                    bilinear(row, column, &learned::PhysicalTerrainSample::elevationMeters),
                .meanTemperatureC =
                    bilinear(row, column, &learned::PhysicalTerrainSample::meanTemperatureC),
                .temperatureVariabilityC =
                    bilinear(row, column, &learned::PhysicalTerrainSample::temperatureVariabilityC),
                .annualPrecipitationMm =
                    bilinear(row, column, &learned::PhysicalTerrainSample::annualPrecipitationMm),
                .precipitationCoefficientOfVariation =
                    bilinear(row, column,
                             &learned::PhysicalTerrainSample::precipitationCoefficientOfVariation),
                .lapseRateCPerMeter =
                    bilinear(row, column, &learned::PhysicalTerrainSample::lapseRateCPerMeter),
            });
        }
    }
    return output;
}

std::vector<learned::PhysicalTerrainSample>
requireLearnedNativeTerrainPoints(const std::shared_ptr<learned::WorldGenerationContext>& context,
                                  std::span<const learned::NativePoint> points) {
    auto result = context->queryNativePoints(points);
    if (result.isReady()) return std::move(*result.value());
    learned::GenerationFailure failure =
        result.failure() != nullptr
            ? *result.failure()
            : learned::GenerationFailure{
                  .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                  .message = "Learned native terrain query returned no value or failure",
                  .retriable = true,
              };
    if (result.status() == learned::AuthorityStatus::FAILED) context->latchFailure(failure);
    throw learned::GenerationFailureException(result.status(), std::move(failure));
}

// A native hydrology owner routes a complete 2,048-block raster. Its input
// callback used to admit learned pages one 32K row-major chunk at a time. A
// deferred first chunk then aborted the routing build before later rows could
// admit their pages, turning one 4-by-4 authority closure into serial waves.
// Admit the rectangular native-page closure first, in the same lexical order
// used by TerrainPageCoordinate, while retaining the bounded raster queries
// below for reconstruction and memory locality.
void prequeueNativeHydrologyAuthorityClosure(
    const std::shared_ptr<learned::WorldGenerationContext>& context,
    std::span<const NativeHydrologyPosition> positions) {
    if (positions.empty()) return;

    auto pageCoordinate = [](NativeHydrologyPosition position) {
        return learned::terrainPageCoordinateFor(
            learned::worldBlockToNative(position.x, position.z));
    };

    learned::TerrainPageCoordinate minimum = pageCoordinate(positions.front());
    learned::TerrainPageCoordinate maximum = minimum;
    for (const NativeHydrologyPosition position : positions) {
        const learned::TerrainPageCoordinate coordinate = pageCoordinate(position);
        minimum.row = std::min(minimum.row, coordinate.row);
        minimum.column = std::min(minimum.column, coordinate.column);
        maximum.row = std::max(maximum.row, coordinate.row);
        maximum.column = std::max(maximum.column, coordinate.column);
    }

    // Atomic closure admission shares the existing single-flight, queue
    // bound, and priority lanes. It keeps the coordinator from beginning a
    // singleton before the remaining hydrology apron pages are visible, so
    // the production backend can reuse their fixed latent batches. The
    // result is deliberately ignored because the normal learned raster query
    // below preserves the established deferred and failed-state behavior.
    std::vector<learned::TerrainPageCoordinate> closure;
    closure.reserve(static_cast<size_t>(maximum.row - minimum.row + 1) *
                    static_cast<size_t>(maximum.column - minimum.column + 1));
    for (int64_t row = minimum.row; row <= maximum.row; ++row) {
        for (int64_t column = minimum.column; column <= maximum.column; ++column) {
            closure.push_back({.row = row, .column = column});
        }
    }
    static_cast<void>(context->requestAuthorityPages(closure, context->requestPriority()));
}

void applyLearnedUndisturbedTerrain(HydrologySample& hydrology, double undisturbed,
                                    uint64_t worldSeed) {
    const bool learnedOcean = undisturbed < SEA_LEVEL;
    const bool explicitFall = hydrology.waterfall &&
                              hydrology.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
                              hydrology.transitionOwnerId != 0 &&
                              hydrology.waterfallTop >= hydrology.waterfallBottom + 0.5;
    const uint64_t fallOwner = hydrology.transitionOwnerId;
    const bool fallAnchor = hydrology.waterfallAnchor;
    const double fallTop = hydrology.waterfallTop;
    const double fallBottom = hydrology.waterfallBottom;
    const double fallWidth = hydrology.waterfallWidth;
    const bool coastalDelta = hydrology.delta;
    const bool coastalEstuary = hydrology.estuary;
    const bool coastalBrackish = hydrology.brackish;
    const uint8_t coastalStreamOrder = hydrology.streamOrder;
    const uint8_t coastalDistributaryCount = hydrology.distributaryCount;
    if (learnedOcean) {
        hydrology.waterBodyId = nativeOceanWaterBodyId(worldSeed);
        hydrology.generatedFluidLevel = explicitFall ? 7 : 0;
        hydrology.transitionOwnerKind =
            explicitFall ? WaterTransitionKind::EXPLICIT_FALL : WaterTransitionKind::NONE;
        hydrology.transitionOwnerId = explicitFall ? fallOwner : 0;
        hydrology.waterSurface = SEA_LEVEL;
        hydrology.channelDepth = 0.0;
        hydrology.lakeDepth = 0.0;
        hydrology.lakeShoreDistance = -1.0e9;
        hydrology.shoreWaterSurface = 0.0;
        hydrology.lakeBankTarget = 0.0;
        hydrology.lakeBankInfluence = 0.0;
        hydrology.waterfallTop = explicitFall ? fallTop : 0.0;
        hydrology.waterfallBottom = explicitFall ? fallBottom : 0.0;
        hydrology.waterfallWidth = explicitFall ? fallWidth : 0.0;
        hydrology.streamOrder = coastalDelta ? coastalStreamOrder : 0;
        hydrology.distributaryCount = coastalDelta ? coastalDistributaryCount : 0;
        hydrology.ocean = true;
        hydrology.river = false;
        hydrology.lake = false;
        hydrology.lakeBank = false;
        hydrology.channelBank = false;
        hydrology.endorheic = false;
        hydrology.waterfall = explicitFall;
        hydrology.waterfallAnchor = explicitFall && fallAnchor;
        hydrology.delta = coastalDelta;
        hydrology.estuary = coastalEstuary;
        hydrology.brackish = coastalBrackish;
        hydrology.perennial = false;
        hydrology.ephemeral = false;
        hydrology.wetland = false;
    } else if (hydrology.ocean) {
        // The routed authority and the block-resolution sample use the same
        // quality. Bilinear reconstruction can still move a coastline through
        // a column, so leave that column dry without inventing a bank.
        hydrology.waterBodyId = NO_WATER_BODY;
        hydrology.generatedFluidLevel = 0;
        hydrology.transitionOwnerKind = WaterTransitionKind::NONE;
        hydrology.transitionOwnerId = 0;
        hydrology.waterSurface = 0.0;
        hydrology.ocean = false;
        hydrology.delta = false;
        hydrology.estuary = false;
        hydrology.brackish = false;
        hydrology.distributaryCount = 0;
        hydrology.waterfall = false;
        hydrology.waterfallAnchor = false;
        hydrology.waterfallTop = 0.0;
        hydrology.waterfallBottom = 0.0;
        hydrology.waterfallWidth = 0.0;
    }
    quantizeGeneratedRiverSurface(hydrology);
    double corrected = undisturbed;
    const bool surfaceWater =
        hydrology.ocean || hydrology.river || hydrology.lake || hydrology.wetland;
    if (surfaceWater) {
        // Learned elevation is the v4 macro surface. Reapplying the legacy
        // 16-block eroded raster would replace that authority and imprint its
        // storage phase on every routed bed. Preserve the canonical stage and
        // discharge-derived river depth while cutting the learned substrate
        // directly at the queried coordinate.
        const double bedDepth = hydrology.river ? std::max(0.125, hydrology.channelDepth) : 0.125;
        corrected = std::min(corrected, hydrology.waterSurface - bedDepth);
    } else if (hydrology.waterfall) {
        corrected = std::min(corrected, hydrology.waterSurface - 0.125);
    }
    hydrology.surfaceElevation = corrected;
    hydrology.erosionDepth = std::max(0.0, undisturbed - corrected);
    if (hydrology.lake) hydrology.lakeDepth = std::max(0.0, hydrology.waterSurface - corrected);
    if (hydrology.river)
        hydrology.channelDepth =
            std::max(hydrology.channelDepth, hydrology.waterSurface - corrected);
    if (!hydrology.ocean && !hydrology.lake && !hydrology.wetland)
        hydrology.groundwaterHead = std::min(hydrology.groundwaterHead, corrected);
}

[[noreturn]] void
failInvalidNativeHydrology(const std::shared_ptr<learned::WorldGenerationContext>& context) {
    learned::GenerationFailure failure{
        .code = learned::GenerationFailureCode::INFERENCE_FAILED,
        .message = "Generator v4 hydrology returned an invalid canonical page sample",
        .retriable = true,
    };
    context->latchFailure(failure);
    throw learned::GenerationFailureException(learned::AuthorityStatus::FAILED, std::move(failure));
}

ClimateFields learnedClimateFields(const learned::PhysicalTerrainSample& physical,
                                   double terrainHeight) {
    const double nativeHeight =
        learned::learnedElevationMetersToWorldHeight(physical.elevationMeters);
    const double heightDeltaMeters =
        (terrainHeight - nativeHeight) * learned::WORLD_METERS_PER_BLOCK;

    ClimateFields result;
    result.temperatureC =
        physical.meanTemperatureC + physical.lapseRateCPerMeter * heightDeltaMeters;
    result.temperatureVariabilityC = physical.temperatureVariabilityC;
    result.annualPrecipitationMm = physical.annualPrecipitationMm;
    result.precipitationCoefficientOfVariation = physical.precipitationCoefficientOfVariation;
    result.lapseRateCPerMeter = physical.lapseRateCPerMeter;
    result.potentialEvapotranspirationMm = std::clamp(
        300.0 + std::max(-8.0, result.temperatureC) * 31.0 + result.temperatureVariabilityC * 8.0,
        120.0, 1'800.0);
    result.relativeHumidity =
        clamp01(result.annualPrecipitationMm /
                std::max(1.0, result.annualPrecipitationMm +
                                  result.potentialEvapotranspirationMm *
                                      (0.75 + result.precipitationCoefficientOfVariation)));
    result.aridity =
        result.potentialEvapotranspirationMm / std::max(1.0, result.annualPrecipitationMm);
    return result;
}

ClimateFields learnedClimateFields(const std::shared_ptr<learned::WorldGenerationContext>& context,
                                   double x, double z, double terrainHeight) {
    return learnedClimateFields(requireLearnedTerrain(context, x, z), terrainHeight);
}

uint32_t noiseSeed(uint64_t seed, uint64_t noiseStream) {
    return CounterRng(seed).u32(noiseStream, 0, 0, 0);
}

double counterSimplex(const CounterRng& random, uint64_t noiseStream, double x, double z,
                      uint32_t index) {
    constexpr double F2 = 0.36602540378443864676;
    constexpr double G2 = 0.21132486540518711775;
    constexpr std::array<Vector2d, 16> GRADIENTS = {{
        {1.0, 0.0},
        {0.9238795325, 0.3826834324},
        {0.7071067812, 0.7071067812},
        {0.3826834324, 0.9238795325},
        {0.0, 1.0},
        {-0.3826834324, 0.9238795325},
        {-0.7071067812, 0.7071067812},
        {-0.9238795325, 0.3826834324},
        {-1.0, 0.0},
        {-0.9238795325, -0.3826834324},
        {-0.7071067812, -0.7071067812},
        {-0.3826834324, -0.9238795325},
        {0.0, -1.0},
        {0.3826834324, -0.9238795325},
        {0.7071067812, -0.7071067812},
        {0.9238795325, -0.3826834324},
    }};
    const double skew = (x + z) * F2;
    const int64_t cellX = floorToInt64(x + skew);
    const int64_t cellZ = floorToInt64(z + skew);
    const double unskew = (static_cast<double>(cellX) + static_cast<double>(cellZ)) * G2;
    const double x0 = x - (static_cast<double>(cellX) - unskew);
    const double z0 = z - (static_cast<double>(cellZ) - unskew);
    const int64_t xStep = x0 > z0 ? 1 : 0;
    const int64_t zStep = x0 > z0 ? 0 : 1;
    const std::array<double, 3> offsetsX = {x0, x0 - static_cast<double>(xStep) + G2,
                                            x0 - 1.0 + 2.0 * G2};
    const std::array<double, 3> offsetsZ = {z0, z0 - static_cast<double>(zStep) + G2,
                                            z0 - 1.0 + 2.0 * G2};
    const std::array<int64_t, 3> latticeX = {cellX, cellX + xStep, cellX + 1};
    const std::array<int64_t, 3> latticeZ = {cellZ, cellZ + zStep, cellZ + 1};
    double value = 0.0;
    for (size_t corner = 0; corner < 3; ++corner) {
        double attenuation =
            0.5 - offsetsX[corner] * offsetsX[corner] - offsetsZ[corner] * offsetsZ[corner];
        if (attenuation <= 0.0) continue;
        const uint32_t hash = random.u32(noiseStream, latticeX[corner], 0, latticeZ[corner],
                                         index + static_cast<uint32_t>(corner));
        const Vector2d gradient = GRADIENTS[hash & 15U];
        attenuation *= attenuation;
        value += attenuation * attenuation *
                 (gradient.x * offsetsX[corner] + gradient.z * offsetsZ[corner]);
    }
    return value * 70.0;
}

double rotatedCounterSimplex(const CounterRng& random, uint64_t noiseStream, double x, double z,
                             double scale, double angle, uint32_t index) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double rotatedX = (x * cosine - z * sine) / scale;
    const double rotatedZ = (x * sine + z * cosine) / scale;
    return counterSimplex(random, noiseStream, rotatedX, rotatedZ, index);
}

double tabulatedSimplexCdf(double value) {
    constexpr std::array<double, 17> VALUES = {
        -1.0,  -0.875, -0.75, -0.625, -0.5,  -0.375, -0.25, -0.125, 0.0,
        0.125, 0.25,   0.375, 0.5,    0.625, 0.75,   0.875, 1.0,
    };
    constexpr std::array<double, 17> CDF = {
        0.0003, 0.0013, 0.0049, 0.0155, 0.0420, 0.0980, 0.1940, 0.3330, 0.5000,
        0.6670, 0.8060, 0.9020, 0.9580, 0.9845, 0.9951, 0.9987, 0.9997,
    };
    if (value <= VALUES.front()) return CDF.front();
    if (value >= VALUES.back()) return CDF.back();
    const double position = (value - VALUES.front()) / (VALUES.back() - VALUES.front()) *
                            static_cast<double>(VALUES.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const double fraction = position - static_cast<double>(lower);
    return std::lerp(CDF[lower], CDF[lower + 1], fraction);
}

double rotatedSimplex(const SimplexNoise& noise, double x, double z, double scale, int octaves,
                      double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double rotatedX = x * cosine - z * sine;
    const double rotatedZ = x * sine + z * cosine;
    return noise.octave2D(rotatedX / scale, rotatedZ / scale, octaves);
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

double wetlandInfluence(const HydrologySample& hydrology) {
    if (!hydrology.wetland) return 0.0;
    // The native authority only marks a wetland after assigning a parent
    // standing-water stage. Hydroperiod then gives the physical-climate
    // adapter a continuous saturated-ground signal without widening the
    // water body beyond that canonical fringe.
    return clamp01(hydrology.hydroperiod);
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
    // Independent rotated simplex-gradient bands make connected isotropic
    // patches. Their weighted distribution is transformed through a compact
    // symmetric CDF table so blend proportions remain unbiased.
    const double broad =
        rotatedCounterSimplex(random, stream ^ 0xA24BAED4963EE407ULL, static_cast<double>(x),
                              static_cast<double>(z), 47.0, 0.6180339887498948, index * 3U);
    const double medium =
        rotatedCounterSimplex(random, stream ^ 0x9FB21C651E98DF25ULL, static_cast<double>(x),
                              static_cast<double>(z), 17.0, 1.176005207095135, index * 3U + 1U);
    const double fine =
        rotatedCounterSimplex(random, stream ^ 0xC13FA9A902A6328FULL, static_cast<double>(x),
                              static_cast<double>(z), 7.0, 2.0344439357957027, index * 3U + 2U);
    const double rank = tabulatedSimplexCdf(broad * 0.58 + medium * 0.29 + fine * 0.13);
    return std::clamp(rank, std::numeric_limits<double>::epsilon(),
                      1.0 - std::numeric_limits<double>::epsilon());
}

double climateWaterInfluence(const HydrologySample& hydrology) noexcept {
    return std::max({oceanInfluence(hydrology), lakeInfluence(hydrology) * 0.65,
                     channelInfluence(hydrology) * 0.18, wetlandInfluence(hydrology) * 0.22});
}

double rockErosionResistance(RockType rock) noexcept {
    switch (rock) {
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
}

double lithologyErosionResistance(const GeologySample& geology) noexcept {
    return std::clamp(geology.erosionResistance, 0.35, 1.25);
}

void quantizeGeneratedRiverSurface(HydrologySample& hydrology) noexcept {
    if (!hydrology.river || hydrology.ocean || hydrology.lake || hydrology.wetland ||
        hydrology.waterfall || !std::isfinite(hydrology.waterSurface)) {
        return;
    }

    hydrology.waterSurface = std::round(hydrology.waterSurface * 8.0) / 8.0;
    const double wholePlane = std::floor(hydrology.waterSurface + 1.0e-6);
    const int eighths = static_cast<int>(
        std::lround(std::clamp(hydrology.waterSurface - wholePlane, 0.0, 0.875) * 8.0));
    hydrology.generatedFluidLevel =
        eighths > 0 ? static_cast<uint8_t>(8 - eighths) : static_cast<uint8_t>(0);

    // The top state is immutable generated geometry. Keep a complete source
    // volume underneath it and lower the analytical bed when needed, without
    // manufacturing a dry support wall around the route.
    hydrology.surfaceElevation =
        std::min(hydrology.surfaceElevation, hydrology.waterSurface - 0.125);
    hydrology.channelDepth = std::max(0.125, hydrology.waterSurface - hydrology.surfaceElevation);
}

GeneratedFluidColumn generatedFluidColumn(const SurfaceSample& surface) noexcept {
    GeneratedFluidColumn result;
    result.visibleSurface = surface.waterSurface;
    const double emittedTerrainHeight = geometryTerrainHeight(surface);
    result.wet = (surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake ||
                  surface.hydrology.wetland) &&
                 std::isfinite(surface.waterSurface) &&
                 surface.waterSurface > emittedTerrainHeight + 0.01;
    if (!result.wet) return result;

    result.topY =
        std::clamp(static_cast<int>(std::ceil(surface.waterSurface)) - 1, WORLD_MIN_Y, WORLD_MAX_Y);
    result.standing = surface.hydrology.ocean || surface.hydrology.lake ||
                      surface.hydrology.wetland || surface.hydrology.waterfall ||
                      surface.hydrology.generatedFluidLevel == 0;
    if (surface.hydrology.river && surface.hydrology.generatedFluidLevel > 0) {
        result.topState = FluidState::flowing(surface.hydrology.generatedFluidLevel);
    }
    result.visibleSurface = static_cast<double>(result.topY) + fluidSurfaceHeight(result.topState);
    if (surface.hydrology.waterfall) {
        result.fallingStartY =
            std::clamp(static_cast<int>(std::ceil(surface.hydrology.waterfallBottom)), WORLD_MIN_Y,
                       WORLD_MAX_Y + 1);
    }
    return result;
}

namespace {

HydrologySample hydrologyFromBasin(const BasinSample& basin) {
    HydrologySample result;
    result.waterBodyId = basin.waterBodyId;
    result.generatedFluidLevel = basin.generatedFluidLevel;
    result.transitionOwnerKind = basin.transitionOwnerKind;
    result.transitionOwnerId = basin.transitionOwnerId;
    result.flowDirection = {basin.flowX, basin.flowZ};
    result.surfaceElevation = basin.surfaceElevation;
    result.terrainSlope = basin.terrainSlope;
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
    result.lakeAreaSquareKilometers = basin.lakeAreaSquareKilometers;
    result.lakeVolumeCubicMeters = basin.lakeVolumeCubicMeters;
    result.lakeRunoffMmSquareKilometers = basin.lakeRunoffMmSquareKilometers;
    result.lakeLossMm = basin.lakeLossMm;
    result.lakeOverflowMmSquareKilometers = basin.lakeOverflowMmSquareKilometers;
    result.lakeSpillSurface = basin.lakeSpillSurface;
    result.baseflow = basin.baseflow;
    result.precipitationSeasonality = basin.precipitationSeasonality;
    result.groundwaterRechargeMm = basin.groundwaterRechargeMm;
    result.groundwaterHead = basin.groundwaterHead;
    result.hydroperiod = basin.hydroperiod;
    result.waterfallTop = basin.waterfallTop;
    result.waterfallBottom = basin.waterfallBottom;
    result.waterfallWidth = basin.waterfallWidth;
    result.streamOrder = basin.streamOrder;
    result.distributaryCount = basin.distributaryCount;
    result.ocean = basin.ocean;
    result.river = basin.river;
    result.lake = basin.lake;
    result.lakeBank = basin.lakeBank;
    result.channelBank = basin.channelBank;
    result.endorheic = basin.endorheic;
    result.waterfall = basin.waterfall;
    result.waterfallAnchor = basin.waterfallAnchor;
    result.delta = basin.delta;
    result.estuary = basin.estuary;
    result.brackish = basin.brackish;
    result.perennial = basin.perennial;
    result.ephemeral = basin.ephemeral;
    result.wetland = basin.wetland;
    return result;
}

HydrologySample hydrologyForGenerationAuthority(const BasinSample& basin, bool learnedAuthority) {
    HydrologySample result = hydrologyFromBasin(basin);
    if (learnedAuthority) return result;

    // Generator v3 predates the static ecosystem and wetland authority
    // fields. Keep its diagnostic profile byte-compatible by preventing
    // those fields from changing its climate controls, material choices, or
    // water topology while v4 remains the sole consumer of the new authority.
    result.baseflow = 0.0;
    result.precipitationSeasonality = 0.0;
    result.groundwaterRechargeMm = 0.0;
    result.groundwaterHead = 0.0;
    result.hydroperiod = 0.0;
    result.perennial = false;
    result.ephemeral = false;
    result.wetland = false;
    result.estuary = false;
    result.brackish = false;
    return result;
}

bool certifiedHydrologySampleRemainsDry(const HydrologySample& sample) noexcept {
    constexpr double WATER_DEPTH_EPSILON = 0.01;
    return std::isfinite(sample.surfaceElevation) && std::isfinite(sample.waterSurface) &&
           sample.generatedFluidLevel == 0 && sample.waterBodyId == NO_WATER_BODY &&
           sample.transitionOwnerKind == WaterTransitionKind::NONE &&
           sample.transitionOwnerId == 0 && !sample.ocean && !sample.river && !sample.lake &&
           !sample.wetland && !sample.waterfall && !sample.delta && !sample.estuary &&
           !sample.brackish && sample.waterSurface <= sample.surfaceElevation + WATER_DEPTH_EPSILON;
}

} // namespace

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

struct MacroGenerationSampler::MacroControlTile {
    ColumnPos key;
    std::array<SurfaceSample, MACRO_CONTROL_SAMPLE_COUNT> controls;
    SimplexNoise climateDetailNoise;

    const SurfaceSample& at(int x, int z) const {
        return controls[static_cast<size_t>(z * MACRO_CONTROL_GRID_EDGE + x)];
    }

    size_t byteSize() const noexcept { return sizeof(*this); }
};

struct MacroGenerationSampler::FarClimateControlTile {
    struct Control {
        ClimateFields climate;
        double terrainHeight = 0.0;
    };

    ColumnPos key;
    std::array<Control, FAR_CLIMATE_CONTROL_SAMPLE_COUNT> controls;
    SimplexNoise climateDetailNoise;

    const Control& at(int x, int z) const {
        return controls[static_cast<size_t>(z * FAR_CLIMATE_CONTROL_GRID_EDGE + x)];
    }

    size_t byteSize() const noexcept { return sizeof(*this); }
};

struct MacroControlView::Impl {
    static constexpr size_t LEARNED_STENCIL_EDGE = CHUNK_EDGE / learned::MODEL_BLOCK_SCALE + 2;
    static constexpr size_t LEARNED_STENCIL_SAMPLE_COUNT =
        LEARNED_STENCIL_EDGE * LEARNED_STENCIL_EDGE;
    static_assert(LEARNED_STENCIL_EDGE == 6);
    static_assert(LEARNED_STENCIL_SAMPLE_COUNT * sizeof(learned::QuantizedTerrainSample) == 432);

    std::shared_ptr<const MacroGenerationSampler::MacroControlTile> tile;
    std::shared_ptr<learned::WorldGenerationContext> generationContext;
    ColumnPos chunkColumn;
    mutable std::once_flag learnedStencilOnce;
    mutable std::array<learned::QuantizedTerrainSample, LEARNED_STENCIL_SAMPLE_COUNT>
        learnedStencil{};
    mutable int64_t learnedStencilRowBegin = 0;
    mutable int64_t learnedStencilColumnBegin = 0;
    mutable bool learnedStencilReady = false;

    [[nodiscard]] bool ownsReconstructionPoint(double x, double z) const noexcept {
        if (!std::isfinite(x) || !std::isfinite(z)) return false;
        const __int128 worldXBegin = static_cast<__int128>(chunkColumn.x) * CHUNK_EDGE;
        const __int128 worldZBegin = static_cast<__int128>(chunkColumn.z) * CHUNK_EDGE;
        const __int128 worldXEnd = worldXBegin + CHUNK_EDGE;
        const __int128 worldZEnd = worldZBegin + CHUNK_EDGE;
        if (worldXBegin < std::numeric_limits<int64_t>::min() ||
            worldXEnd > std::numeric_limits<int64_t>::max() ||
            worldZBegin < std::numeric_limits<int64_t>::min() ||
            worldZEnd > std::numeric_limits<int64_t>::max()) {
            return false;
        }
        return x >= static_cast<double>(worldXBegin) && x <= static_cast<double>(worldXEnd) &&
               z >= static_cast<double>(worldZBegin) && z <= static_cast<double>(worldZEnd);
    }

    [[nodiscard]] const learned::QuantizedTerrainSample*
    learnedSample(learned::NativePoint point) const noexcept {
        if (!learnedStencilReady || point.row < learnedStencilRowBegin ||
            point.column < learnedStencilColumnBegin ||
            point.row >= learnedStencilRowBegin + static_cast<int64_t>(LEARNED_STENCIL_EDGE) ||
            point.column >=
                learnedStencilColumnBegin + static_cast<int64_t>(LEARNED_STENCIL_EDGE)) {
            return nullptr;
        }
        const size_t row = static_cast<size_t>(point.row - learnedStencilRowBegin);
        const size_t column = static_cast<size_t>(point.column - learnedStencilColumnBegin);
        return std::addressof(learnedStencil[row * LEARNED_STENCIL_EDGE + column]);
    }
};

namespace {

std::array<double, 4> cubicSplineBasis(double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    return {
        (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
        (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
        t3 / 6.0,
    };
}

std::array<double, 4> cubicSplineDerivative(double t) {
    const double t2 = t * t;
    return {
        -0.5 * (1.0 - 2.0 * t + t2),
        -2.0 * t + 1.5 * t2,
        0.5 + t - 1.5 * t2,
        0.5 * t2,
    };
}

struct CubicControlStencil {
    int cellX = 0;
    int cellZ = 0;
    double spacing = 1.0;
    std::array<double, 4> weightsX{};
    std::array<double, 4> weightsZ{};
    std::array<double, 4> derivativeX{};
    std::array<double, 4> derivativeZ{};

    template <typename SampleAt>
    double interpolate(SampleAt&& sampleAt) const {
        double value = 0.0;
        for (int offsetZ = 0; offsetZ < 4; ++offsetZ) {
            for (int offsetX = 0; offsetX < 4; ++offsetX) {
                value += weightsX[static_cast<size_t>(offsetX)] *
                         weightsZ[static_cast<size_t>(offsetZ)] *
                         sampleAt(cellX + offsetX, cellZ + offsetZ);
            }
        }
        return value;
    }

    template <typename SampleAt>
    Vector2d gradient(SampleAt&& sampleAt) const {
        Vector2d result;
        for (int offsetZ = 0; offsetZ < 4; ++offsetZ) {
            for (int offsetX = 0; offsetX < 4; ++offsetX) {
                const double sample = sampleAt(cellX + offsetX, cellZ + offsetZ);
                result.x += derivativeX[static_cast<size_t>(offsetX)] *
                            weightsZ[static_cast<size_t>(offsetZ)] * sample;
                result.z += weightsX[static_cast<size_t>(offsetX)] *
                            derivativeZ[static_cast<size_t>(offsetZ)] * sample;
            }
        }
        result.x /= spacing;
        result.z /= spacing;
        return result;
    }
};

CubicControlStencil cubicControlStencil(double x, double z, double originX, double originZ,
                                        double spacing, int coreEdge) {
    const double controlX =
        std::clamp((x - originX) / spacing, 0.0, static_cast<double>(coreEdge - 1));
    const double controlZ =
        std::clamp((z - originZ) / spacing, 0.0, static_cast<double>(coreEdge - 1));
    CubicControlStencil result;
    result.cellX = std::clamp(static_cast<int>(std::floor(controlX)), 0, coreEdge - 2);
    result.cellZ = std::clamp(static_cast<int>(std::floor(controlZ)), 0, coreEdge - 2);
    result.spacing = spacing;
    result.weightsX = cubicSplineBasis(controlX - result.cellX);
    result.weightsZ = cubicSplineBasis(controlZ - result.cellZ);
    result.derivativeX = cubicSplineDerivative(controlX - result.cellX);
    result.derivativeZ = cubicSplineDerivative(controlZ - result.cellZ);
    return result;
}

template <typename TerrainAt, typename ClimateAt>
void reconstructClimate(SurfaceSample& result, const CubicControlStencil& stencil,
                        TerrainAt&& terrainAt, ClimateAt&& climateAt) {
    result.climate.wind = {
        stencil.interpolate([&](int x, int z) { return climateAt(x, z).wind.x; }),
        stencil.interpolate([&](int x, int z) { return climateAt(x, z).wind.z; }),
    };
    const double controlTerrain = stencil.interpolate(terrainAt);
    const double controlTemperature =
        stencil.interpolate([&](int x, int z) { return climateAt(x, z).temperatureC; });
    result.climate.temperatureC = controlTemperature + (result.terrainHeight - controlTerrain) *
                                                           LEGACY_WORLD_METERS_PER_BLOCK *
                                                           LEGACY_LAPSE_RATE_C_PER_METER;
    result.climate.annualPrecipitationMm = std::clamp(
        stencil.interpolate([&](int x, int z) { return climateAt(x, z).annualPrecipitationMm; }),
        60.0, 3600.0);
    result.climate.relativeHumidity = clamp01(
        stencil.interpolate([&](int x, int z) { return climateAt(x, z).relativeHumidity; }));
    result.climate.potentialEvapotranspirationMm =
        std::clamp(stencil.interpolate([&](int x, int z) {
            return climateAt(x, z).potentialEvapotranspirationMm;
        }) + (result.climate.temperatureC - controlTemperature) * 31.0,
                   120.0, 1800.0);
    result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                             std::max(1.0, result.climate.annualPrecipitationMm);
}

template <typename Tile>
void reconstructContinuousFields(const Tile& tile, double x, double z, SurfaceSample& result) {
    const double originX = static_cast<double>(tile.key.x * MACRO_CONTROL_TILE_EDGE);
    const double originZ = static_cast<double>(tile.key.z * MACRO_CONTROL_TILE_EDGE);
    const CubicControlStencil stencil =
        cubicControlStencil(x, z, originX, originZ, MACRO_CONTROL_SPACING, MACRO_CONTROL_CORE_EDGE);
    const auto interpolate = [&](auto getter) {
        return stencil.interpolate(
            [&](int controlX, int controlZ) { return getter(tile.at(controlX, controlZ)); });
    };

    result.geology.plateVelocity = {
        interpolate([](const SurfaceSample& sample) { return sample.geology.plateVelocity.x; }),
        interpolate([](const SurfaceSample& sample) { return sample.geology.plateVelocity.z; }),
    };
    result.geology.continentalFraction = clamp01(interpolate(
        [](const SurfaceSample& sample) { return sample.geology.continentalFraction; }));
    result.geology.crustAge =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.geology.crustAge; }));
    result.geology.crustThickness =
        interpolate([](const SurfaceSample& sample) { return sample.geology.crustThickness; });
    result.geology.crustDensity =
        interpolate([](const SurfaceSample& sample) { return sample.geology.crustDensity; });
    result.geology.erosionResistance = std::clamp(
        interpolate([](const SurfaceSample& sample) { return sample.geology.erosionResistance; }),
        0.35, 1.25);
    result.geology.distanceToBoundary = std::max(0.0, interpolate([](const SurfaceSample& sample) {
                                                     return sample.geology.distanceToBoundary;
                                                 }));
    result.geology.uplift =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.geology.uplift; }));
    result.geology.rift =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.geology.rift; }));
    result.geology.faultStrength = clamp01(
        interpolate([](const SurfaceSample& sample) { return sample.geology.faultStrength; }));
    result.geology.hotspotInfluence = clamp01(
        interpolate([](const SurfaceSample& sample) { return sample.geology.hotspotInfluence; }));
    result.geology.volcanicActivity = clamp01(
        interpolate([](const SurfaceSample& sample) { return sample.geology.volcanicActivity; }));

    result.slope =
        std::max(0.0, interpolate([](const SurfaceSample& sample) { return sample.slope; }));
    reconstructClimate(
        result, stencil,
        [&](int controlX, int controlZ) { return tile.at(controlX, controlZ).terrainHeight; },
        [&](int controlX, int controlZ) -> const ClimateFields& {
            return tile.at(controlX, controlZ).climate;
        });
    // The C2 controls carry the broad moisture transport solution. Two
    // oblique filtered bands restore local weather variation without making
    // the eight-block control phase visible in final precipitation or biome
    // suitability.
    result.climate.annualPrecipitationMm =
        std::clamp(result.climate.annualPrecipitationMm +
                       rotatedSimplex(tile.climateDetailNoise, x, z, 96.0, 1, 0.619) * 90.0 +
                       rotatedSimplex(tile.climateDetailNoise, x, z, 36.0, 1, 1.847) * 75.0,
                   60.0, 3600.0);
    const double temperatureDetail =
        rotatedSimplex(tile.climateDetailNoise, x, z, 80.0, 1, 2.311) * 1.8 +
        rotatedSimplex(tile.climateDetailNoise, x, z, 28.0, 1, 0.927) * 1.4;
    result.climate.temperatureC += temperatureDetail;
    result.climate.potentialEvapotranspirationMm = std::clamp(
        result.climate.potentialEvapotranspirationMm + temperatureDetail * 31.0, 120.0, 1800.0);
    result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                             std::max(1.0, result.climate.annualPrecipitationMm);
    result.soil.moisture =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.soil.moisture; }));
    result.soil.fertility =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.soil.fertility; }));
    result.soil.drainage =
        clamp01(interpolate([](const SurfaceSample& sample) { return sample.soil.drainage; }));
    result.soil.waterTable =
        interpolate([](const SurfaceSample& sample) { return sample.soil.waterTable; });
    for (size_t index = 0; index < result.suitability.scores.size(); ++index) {
        result.suitability.scores[index] = static_cast<float>(
            std::max(0.0, interpolate([index](const SurfaceSample& sample) {
                         return static_cast<double>(sample.suitability.scores[index]);
                     })));
    }
    result.biome = MacroGenerationSampler::selectBiome(result.suitability);
}

template <typename Tile>
class SingleFlightControlTileCache {
public:
    using TilePointer = std::shared_ptr<const Tile>;
    using Builder = std::function<TilePointer()>;

    SingleFlightControlTileCache(size_t capacity, size_t byteBudget)
        : capacity_(std::max<size_t>(1, capacity))
        , byteBudget_(std::max<size_t>(1, byteBudget)) {}

    TilePointer getOrCreate(ColumnPos key, const Builder& builder) const {
        std::shared_future<TilePointer> future;
        std::shared_ptr<std::promise<TilePointer>> producer;
        uint64_t token = 0;
        constexpr size_t TILE_BYTES = sizeof(Tile);
        const bool cacheable = TILE_BYTES <= byteBudget_;

        while (true) {
            std::shared_future<TilePointer> evictionWait;
            {
                std::lock_guard lock(mutex_);
                auto found = entries_.find(key);
                if (found != entries_.end()) {
                    found->second.lastAccess = ++accessClock_;
                    ++metrics_.hits;
                    if (found->second.future.wait_for(std::chrono::seconds(0)) !=
                        std::future_status::ready) {
                        ++metrics_.singleFlightWaits;
                    }
                    future = found->second.future;
                    break;
                }

                if (entries_.size() >= capacity_ ||
                    (cacheable && bytes_ + TILE_BYTES > byteBudget_)) {
                    auto oldestReady = entries_.end();
                    auto oldestPending = entries_.end();
                    for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
                        if (oldestPending == entries_.end() ||
                            entry->second.lastAccess < oldestPending->second.lastAccess) {
                            oldestPending = entry;
                        }
                        if (entry->second.future.wait_for(std::chrono::seconds(0)) ==
                                std::future_status::ready &&
                            (oldestReady == entries_.end() ||
                             entry->second.lastAccess < oldestReady->second.lastAccess)) {
                            oldestReady = entry;
                        }
                    }
                    if (oldestReady != entries_.end()) {
                        bytes_ -= oldestReady->second.bytes;
                        entries_.erase(oldestReady);
                        ++metrics_.evictions;
                    } else if (oldestPending != entries_.end()) {
                        evictionWait = oldestPending->second.future;
                    }
                }
                if (!evictionWait.valid()) {
                    producer = std::make_shared<std::promise<TilePointer>>();
                    future = producer->get_future().share();
                    token = ++tokenClock_;
                    const size_t retainedBytes = cacheable ? TILE_BYTES : 0;
                    entries_.emplace(key, Entry{future, ++accessClock_, token, retainedBytes});
                    bytes_ += retainedBytes;
                    ++metrics_.misses;
                    ++metrics_.activeBuilds;
                    metrics_.peakBuilds = std::max(metrics_.peakBuilds, metrics_.activeBuilds);
                    break;
                }
            }
            evictionWait.wait();
        }

        if (!producer) return future.get();

        try {
            TilePointer tile = builder();
            if (!cacheable) {
                std::lock_guard lock(mutex_);
                ++metrics_.builds;
                --metrics_.activeBuilds;
                producer->set_value(tile);
                auto found = entries_.find(key);
                if (found != entries_.end() && found->second.token == token) {
                    entries_.erase(found);
                }
                return tile;
            }
            producer->set_value(tile);
            std::lock_guard lock(mutex_);
            ++metrics_.builds;
            --metrics_.activeBuilds;
            return tile;
        } catch (...) {
            const std::exception_ptr exception = std::current_exception();
            if (!cacheable) {
                std::lock_guard lock(mutex_);
                --metrics_.activeBuilds;
                producer->set_exception(exception);
                auto found = entries_.find(key);
                if (found != entries_.end() && found->second.token == token) {
                    entries_.erase(found);
                }
                throw;
            }
            producer->set_exception(exception);
            std::lock_guard lock(mutex_);
            --metrics_.activeBuilds;
            auto found = entries_.find(key);
            if (found != entries_.end() && found->second.token == token) {
                bytes_ -= found->second.bytes;
                entries_.erase(found);
            }
            throw;
        }
    }

    MacroControlCacheMetrics metrics() const {
        std::lock_guard lock(mutex_);
        MacroControlCacheMetrics result = metrics_;
        result.entries = entries_.size();
        result.bytes = bytes_;
        result.capacity = capacity_;
        result.byteBudget = byteBudget_;
        return result;
    }

    void clear() const {
        std::lock_guard lock(mutex_);
        entries_.clear();
        bytes_ = 0;
    }

private:
    struct Entry {
        std::shared_future<TilePointer> future;
        uint64_t lastAccess = 0;
        uint64_t token = 0;
        size_t bytes = 0;
    };

    mutable std::mutex mutex_;
    mutable std::unordered_map<ColumnPos, Entry> entries_;
    mutable uint64_t accessClock_ = 0;
    mutable uint64_t tokenClock_ = 0;
    mutable size_t bytes_ = 0;
    mutable MacroControlCacheMetrics metrics_;
    const size_t capacity_;
    const size_t byteBudget_;
};

} // namespace

class MacroGenerationSampler::MacroControlTileCache
    : public SingleFlightControlTileCache<MacroControlTile> {
public:
    static_assert(sizeof(MacroControlTile) * MACRO_CONTROL_CACHE_CAPACITY <=
                  MACRO_CONTROL_CACHE_BYTE_BUDGET);
    using SingleFlightControlTileCache<MacroControlTile>::SingleFlightControlTileCache;
};

class MacroGenerationSampler::FarClimateControlTileCache
    : public SingleFlightControlTileCache<FarClimateControlTile> {
public:
    static_assert(sizeof(FarClimateControlTile) * FAR_CLIMATE_CONTROL_CACHE_CAPACITY <=
                  FAR_CLIMATE_CONTROL_CACHE_BYTE_BUDGET);
    using SingleFlightControlTileCache<FarClimateControlTile>::SingleFlightControlTileCache;
};

MacroGenerationSampler::MacroGenerationSampler(uint64_t worldSeed, size_t macroControlCacheCapacity,
                                               size_t macroControlCacheByteBudget,
                                               size_t farClimateControlCacheCapacity,
                                               size_t farClimateControlCacheByteBudget)
    : MacroGenerationSampler(worldSeed, nullptr, macroControlCacheCapacity,
                             macroControlCacheByteBudget, farClimateControlCacheCapacity,
                             farClimateControlCacheByteBudget) {}

MacroGenerationSampler::MacroGenerationSampler(
    uint64_t worldSeed, std::shared_ptr<learned::WorldGenerationContext> generationContext,
    size_t macroControlCacheCapacity, size_t macroControlCacheByteBudget,
    size_t farClimateControlCacheCapacity, size_t farClimateControlCacheByteBudget)
    : cacheTag_(worldSeed)
    , generationContext_(std::move(generationContext))
    , random_(worldSeed)
    , legacyAlpineMorphology_(
          generationContext_ ? nullptr : std::make_unique<AlpineMorphologySampler>(worldSeed))
    , legacyBasinSolver_(generationContext_ ? nullptr : std::make_unique<BasinSolver>(worldSeed))
    , nativeHydrologyRouter_(generationContext_ ? generationContext_->nativeHydrologyRouter()
                                                : nullptr)
    , macroControlCache_(std::make_unique<MacroControlTileCache>(macroControlCacheCapacity,
                                                                 macroControlCacheByteBudget))
    , farClimateControlCache_(std::make_unique<FarClimateControlTileCache>(
          farClimateControlCacheCapacity, farClimateControlCacheByteBudget))
    , continentalNoise_(noiseSeed(worldSeed, stream::CONTINENTAL_NOISE))
    , warpXNoise_(noiseSeed(worldSeed, stream::WARP_X_NOISE))
    , warpZNoise_(noiseSeed(worldSeed, stream::WARP_Z_NOISE))
    , reliefNoise_(noiseSeed(worldSeed, stream::RELIEF_NOISE))
    , pressureNoise_(noiseSeed(worldSeed, stream::PRESSURE_NOISE))
    , insolationNoise_(noiseSeed(worldSeed, stream::INSOLATION_NOISE))
    , soilNoise_(noiseSeed(worldSeed, stream::SOIL_NOISE))
    , climateDetailNoise_(noiseSeed(worldSeed, stream::PRESSURE_NOISE ^ 0x434C'494D'4445'5441ULL)) {
    if (generationContext_ &&
        (generationContext_->identity().seed != worldSeed ||
         generationContext_->identity().generatorVersion != 4 ||
         generationContext_->identity().modelBlockScale != learned::MODEL_BLOCK_SCALE)) {
        generationContext_->latchFailure({
            .code = learned::GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
            .message = "Learned macro sampler identity does not match the v4 generation request",
            .retriable = false,
        });
    }
}

MacroGenerationSampler::~MacroGenerationSampler() = default;

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
    result.lithology.primary = nearest.rock;
    result.lithology.secondary = nearest.rock;
    result.plateVelocity = nearest.velocity;
    result.crustAge = nearest.age;
    result.crustThickness = nearest.thickness;
    result.crustDensity = nearest.density;
    result.erosionResistance = rockErosionResistance(nearest.rock);
    result.distanceToBoundary =
        std::max(0.0, (std::sqrt(secondDistanceSquared) - std::sqrt(nearestDistanceSquared)) * 0.5);

    std::array<double, 9> candidateDistances;
    std::array<size_t, 9> nearbyCandidates;
    size_t nearbyCandidateCount = 0;
    const double nearestDistance = std::sqrt(nearestDistanceSquared);
    const double nearbyDistanceLimitSquared =
        (nearestDistance + 2700.0) * (nearestDistance + 2700.0);
    double oppositeCrustDistanceSquared = std::numeric_limits<double>::max();
    double secondaryRockDistanceSquared = std::numeric_limits<double>::max();
    RockType secondaryRock = nearest.rock;
    for (size_t index = 0; index < candidateCount; ++index) {
        const PlateSite& candidate = candidates[index];
        if (candidate.crust != nearest.crust) {
            oppositeCrustDistanceSquared =
                std::min(oppositeCrustDistanceSquared, candidateDistanceSquared[index]);
        }
        if (candidate.rock != nearest.rock &&
            candidateDistanceSquared[index] < secondaryRockDistanceSquared) {
            secondaryRockDistanceSquared = candidateDistanceSquared[index];
            secondaryRock = candidate.rock;
        }
        if (candidateDistanceSquared[index] <= nearbyDistanceLimitSquared) {
            candidateDistances[index] = std::sqrt(candidateDistanceSquared[index]);
            nearbyCandidates[nearbyCandidateCount++] = index;
        }
    }

    // Blend every facies close enough to influence the nearest contact.
    // Keeping one half-weight for the nearest site reproduces the ordinary
    // two-facies transition, while equal secondary candidates participate
    // symmetrically at triple points instead of introducing a rank seam.
    double resistanceWeight = 0.5;
    double weightedResistance = rockErosionResistance(nearest.rock) * resistanceWeight;
    for (size_t index = 0; index < candidateCount; ++index) {
        const PlateSite& candidate = candidates[index];
        if (candidate.id == nearest.id) continue;
        const double contactDistance =
            std::max(0.0, (std::sqrt(candidateDistanceSquared[index]) - nearestDistance) * 0.5);
        const double weight = 0.5 * (1.0 - smoothstep(0.0, 620.0, contactDistance));
        weightedResistance += rockErosionResistance(candidate.rock) * weight;
        resistanceWeight += weight;
    }
    result.erosionResistance = weightedResistance / resistanceWeight;

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

    if (secondaryRockDistanceSquared < std::numeric_limits<double>::max()) {
        result.lithology.secondary = secondaryRock;
        result.lithology.contactDistance = std::max(
            0.0,
            (std::sqrt(secondaryRockDistanceSquared) - std::sqrt(nearestDistanceSquared)) * 0.5);
        result.lithology.transition =
            0.5 * (1.0 - smoothstep(0.0, 620.0, result.lithology.contactDistance));
        const double rank = multiscaleDitherThreshold(
            random_, stream::PLATE_ROCK ^ 0x4C4954484F4C4F47ULL, floorToInt64(x), floorToInt64(z));
        if (rank < result.lithology.transition) result.rock = secondaryRock;
    }

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
    if (result.volcanicActivity > 0.52) {
        result.rock = RockType::VOLCANIC;
        result.lithology.primary = RockType::VOLCANIC;
        result.lithology.secondary = RockType::VOLCANIC;
        result.lithology.transition = 0.0;
    }
    result.erosionResistance =
        std::lerp(result.erosionResistance, rockErosionResistance(RockType::VOLCANIC),
                  smoothstep(0.52, 0.78, result.volcanicActivity));
    return result;
}

double MacroGenerationSampler::v4VolcanicReliefBlocks(double x, double z) const {
    if (!generationContext_) return 0.0;

    ThreadV4VolcanicPrimitiveCache& cache = v4VolcanicPrimitiveCache(this, cacheTag_);
    const auto hotspotPrimitives = [&](int64_t cellX,
                                       int64_t cellZ) -> const std::vector<V4VolcanicPrimitive>& {
        const ColumnPos key{cellX, cellZ};
        if (const auto found = cache.hotspotCells.find(key); found != cache.hotspotCells.end())
            return found->second;

        std::vector<V4VolcanicPrimitive> volcanoes;
        const HotspotChainPrimitive chain = hotspotChain(cellX, cellZ);
        if (chain.active) {
            const Vector2d transverse{-chain.direction.z, chain.direction.x};
            const int count =
                random_.uniformInt(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ, 2, 4, 7);
            volcanoes.reserve(static_cast<size_t>(count));
            for (int index = 0; index < count; ++index) {
                const uint32_t propertyIndex = static_cast<uint32_t>(16 + index * 12);
                const double chainFraction =
                    std::clamp((static_cast<double>(index) +
                                random_.signedUnit(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ,
                                                   propertyIndex) *
                                    0.10) /
                                   std::max(1, count - 1),
                               0.0, 1.0);
                const double transverseOffset =
                    random_.signedUnit(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ,
                                       propertyIndex + 1) *
                    150.0;
                const bool shield =
                    random_.uniform01(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ,
                                      propertyIndex + 2) < (index == 0 ? 0.72 : 0.48);
                const double sizeRoll = random_.uniform01(stream::HOTSPOT_PROPERTIES, cellX, 0,
                                                          cellZ, propertyIndex + 3);
                volcanoes.push_back({
                    .centerX = chain.sourceX + chain.direction.x * chain.length * chainFraction +
                               transverse.x * transverseOffset,
                    .centerZ = chain.sourceZ + chain.direction.z * chain.length * chainFraction +
                               transverse.z * transverseOffset,
                    .radius = shield ? 520.0 + sizeRoll * 430.0 : 260.0 + sizeRoll * 260.0,
                    .coneHeightBlocks = shield ? 11.0 + sizeRoll * 6.0 : 13.0 + sizeRoll * 5.0,
                    .craterRadius = shield ? 52.0 + sizeRoll * 58.0 : 22.0 + sizeRoll * 30.0,
                    .craterDepthBlocks = shield ? 30.0 + sizeRoll * 22.0 : 12.0 + sizeRoll * 10.0,
                    .radialPhase = random_.uniform01(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ,
                                                     propertyIndex + 6) *
                                   2.0 * std::numbers::pi,
                    .id =
                        random_.u64(stream::HOTSPOT_PROPERTIES, cellX, 0, cellZ, propertyIndex + 5),
                    .shield = shield,
                });
            }
        }
        if (cache.hotspotCells.size() >= 1'024) cache.hotspotCells.clear();
        return cache.hotspotCells.emplace(key, std::move(volcanoes)).first->second;
    };

    const auto arcPrimitives = [&](int64_t cellX,
                                   int64_t cellZ) -> const std::vector<V4VolcanicPrimitive>& {
        const ColumnPos key{cellX, cellZ};
        if (const auto found = cache.arcCells.find(key); found != cache.arcCells.end())
            return found->second;

        std::vector<V4VolcanicPrimitive> volcanoes;
        const double centerX =
            (static_cast<double>(cellX) + 0.28 +
             random_.uniform01(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 0) * 0.44) *
            V4_VOLCANIC_ARC_CELL_EDGE;
        const double centerZ =
            (static_cast<double>(cellZ) + 0.28 +
             random_.uniform01(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 1) * 0.44) *
            V4_VOLCANIC_ARC_CELL_EDGE;
        const GeologySample geology = sampleGeology(centerX, centerZ);
        const double acceptance = 0.22 + geology.volcanicActivity * 0.54;
        if (geology.boundary == PlateBoundary::CONVERGENT && geology.volcanicActivity > 0.16 &&
            random_.uniform01(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 2) < acceptance) {
            const double sizeRoll = random_.uniform01(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 3);
            volcanoes.push_back({
                .centerX = centerX,
                .centerZ = centerZ,
                .radius = 230.0 + sizeRoll * 210.0,
                .coneHeightBlocks = 12.0 + sizeRoll * 6.0,
                .craterRadius = 21.0 + sizeRoll * 26.0,
                .craterDepthBlocks = 8.0 + sizeRoll * 7.0,
                .radialPhase = random_.uniform01(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 4) * 2.0 *
                               std::numbers::pi,
                .id = random_.u64(V4_VOLCANIC_ARC_STREAM, cellX, 0, cellZ, 6),
                .shield = false,
            });
        }
        if (cache.arcCells.size() >= 2'048) cache.arcCells.clear();
        return cache.arcCells.emplace(key, std::move(volcanoes)).first->second;
    };

    double strongestProfile = 0.0;
    double retainedAdjustment = 0.0;
    const auto accumulate = [&](const V4VolcanicPrimitive& volcano) {
        const double offsetX = x + 0.5 - volcano.centerX;
        const double offsetZ = z + 0.5 - volcano.centerZ;
        if (std::hypot(offsetX, offsetZ) > volcano.radius * 1.20) return;
        const double radialDistance = v4VolcanoWarpedDistance(volcano, offsetX, offsetZ);
        const double profile = v4VolcanoProfile(volcano, radialDistance);
        if (profile <= strongestProfile) return;
        strongestProfile = profile;
        retainedAdjustment = volcano.coneHeightBlocks * profile -
                             volcano.craterDepthBlocks * v4VolcanoCrater(volcano, radialDistance);
    };

    const int64_t hotspotCellX = world_coord::floorDiv(floorToInt64(x), HOTSPOT_LATTICE_EDGE);
    const int64_t hotspotCellZ = world_coord::floorDiv(floorToInt64(z), HOTSPOT_LATTICE_EDGE);
    const ColumnPos hotspotNeighborhood{hotspotCellX, hotspotCellZ};
    if (!cache.hasHotspotNeighborhood || cache.hotspotNeighborhood != hotspotNeighborhood) {
        cache.hotspotNeighborhood = hotspotNeighborhood;
        cache.hasHotspotNeighborhood = true;
        cache.hotspotCandidates.clear();
        for (int offsetZ = -V4_HOTSPOT_QUERY_RADIUS; offsetZ <= V4_HOTSPOT_QUERY_RADIUS;
             ++offsetZ) {
            for (int offsetX = -V4_HOTSPOT_QUERY_RADIUS; offsetX <= V4_HOTSPOT_QUERY_RADIUS;
                 ++offsetX) {
                const auto& candidates =
                    hotspotPrimitives(hotspotCellX + offsetX, hotspotCellZ + offsetZ);
                cache.hotspotCandidates.insert(cache.hotspotCandidates.end(), candidates.begin(),
                                               candidates.end());
            }
        }
    }
    for (const V4VolcanicPrimitive& volcano : cache.hotspotCandidates)
        accumulate(volcano);

    const int64_t arcCellX = world_coord::floorDiv(floorToInt64(x), V4_VOLCANIC_ARC_CELL_EDGE);
    const int64_t arcCellZ = world_coord::floorDiv(floorToInt64(z), V4_VOLCANIC_ARC_CELL_EDGE);
    const ColumnPos arcNeighborhood{arcCellX, arcCellZ};
    if (!cache.hasArcNeighborhood || cache.arcNeighborhood != arcNeighborhood) {
        cache.arcNeighborhood = arcNeighborhood;
        cache.hasArcNeighborhood = true;
        cache.arcCandidates.clear();
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const auto& candidates = arcPrimitives(arcCellX + offsetX, arcCellZ + offsetZ);
                cache.arcCandidates.insert(cache.arcCandidates.end(), candidates.begin(),
                                           candidates.end());
            }
        }
    }
    for (const V4VolcanicPrimitive& volcano : cache.arcCandidates)
        accumulate(volcano);

    return std::clamp(retainedAdjustment, -18.0, 18.0);
}

double MacroGenerationSampler::preHydrologyElevationMeters(
    double x, double z, const learned::PhysicalTerrainSample& learnedTerrain) const {
    return learnedTerrain.elevationMeters +
           v4VolcanicReliefBlocks(x, z) * learned::WORLD_METERS_PER_BLOCK;
}

double MacroGenerationSampler::preliminaryElevation(double x, double z) const {
    if (generationContext_) {
        const learned::PhysicalTerrainSample learnedTerrain =
            requireLearnedTerrain(generationContext_, x, z);
        return learned::learnedElevationMetersToWorldHeight(
            preHydrologyElevationMeters(x, z, learnedTerrain));
    }
    const GeologySample geology = sampleGeology(x, z);
    const double continentalness = continentalNoise_.octave2D(x / 6200.0, z / 6200.0, 5);
    const double detail = reliefNoise_.octave2D(x / 760.0, z / 760.0, 4);
    const double ridge = clamp01(reliefNoise_.ridged2D(x / 1650.0, z / 1650.0, 4));
    const double tectonicSignal =
        std::max({geology.uplift, geology.rift, geology.faultStrength, geology.hotspotInfluence});
    const double broadRidge =
        tectonicSignal > 0.0 ? clamp01(reliefNoise_.ridged2D(x / 2800.0, z / 2800.0, 3)) : ridge;
    const double foldedPeak = smoothstep(0.38, 0.86, broadRidge);

    // Continental fraction is reconstructed continuously across the warped
    // plate contacts. Deriving the shelf and abyss from that field avoids a
    // sea-floor step when the diagnostic nearest-plate crust ID changes.
    const double oceanInterior = 1.0 - smoothstep(0.10, 0.58, geology.continentalFraction);
    const double shelfProgress = smoothstep(0.05, 0.30, oceanInterior);
    const double abyssProgress = smoothstep(0.22, 0.82, oceanInterior);
    // Plate-site age remains useful as a diagnostic, but nearest-site age is
    // categorical and would step the sea floor at Voronoi contacts. A smooth
    // spreading-age field makes young divergent crust shallow and lets older
    // oceanic crust subside continuously across plate ownership boundaries.
    const double broadSpreadingAge =
        clamp01(0.50 + rotatedSimplex(reliefNoise_, x, z, 14'000.0, 3, 2.173) * 0.36);
    const double ridgeYouth = smoothstep(0.04, 0.72, geology.rift);
    const double spreadingAge =
        clamp01(std::lerp(broadSpreadingAge, 0.03, ridgeYouth) + geology.uplift * 0.12);
    const double thermalSubsidence =
        shelfProgress * 25.0 + abyssProgress * (80.0 + spreadingAge * 40.0);
    const double abyssalUndulation =
        rotatedSimplex(reliefNoise_, x, z, 3100.0, 2, 0.947) * (2.0 + abyssProgress * 6.0);
    // The continuous shelf rolls through a continental slope into ordinary
    // abyssal floors around Y=-55 through Y=-85. Young divergent crust rises
    // above that floor, old crust subsides, and the existing subduction and
    // hotspot terms add trenches, ridges, and rooted seamount chains.
    const double oceanicBase = 49.0 + continentalness * (6.0 - abyssProgress * 3.0) -
                               thermalSubsidence + abyssalUndulation;
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

    const AlpineTectonicSample alpine = legacyAlpineMorphology_->sampleTectonic({
        .x = x,
        .z = z,
        .uplift = geology.uplift,
        .rockResistance = lithologyErosionResistance(geology),
        .continentalFraction = geology.continentalFraction,
    });
    // Hydrology sees the broad ridge and horn network. Finer glacial and
    // crest detail is added only after routing, so it cannot move divides,
    // lake ownership, channels, or shoreline elevations between footprints.
    // Vertical exaggeration gives resistant convergent ranges a fantastical
    // silhouette while the cap and smooth world-height compression preserve
    // headroom and avoid flat clipping at the upper generation bound.
    elevation += std::min(72.0, alpine.elevationOffset * 2.6);

    if (oceanInterior > 0.58 && geology.hotspotInfluence < 0.24 && geology.uplift < 0.42) {
        // Ordinary ridges and abyssal roughness remain submarine. Hotspots
        // and sufficiently strong subduction arcs may still found islands.
        elevation = std::min(elevation, SEA_LEVEL - 4.0 - oceanInterior * 6.0);
    }

    // Unit slope at both knees keeps the bounded mapping smooth. The upper
    // asymptote preserves headroom for emitted volcanoes without flattening
    // ordinary mountain crests into a visible plateau.
    if (elevation > 300.0) elevation = 300.0 + std::tanh((elevation - 300.0) / 180.0) * 180.0;
    if (elevation < -80.0) elevation = -80.0 + std::tanh((elevation + 80.0) / 32.0) * 32.0;
    return std::clamp(elevation, -112.0, 480.0);
}

double MacroGenerationSampler::reliefDetail(double x, double z, SurfaceFootprint footprint) const {
    constexpr std::array<double, 4> WAVELENGTHS = {64.0, 32.0, 16.0, 8.0};
    constexpr std::array<double, 4> AMPLITUDES = {2.20, 1.15, 1.10, 2.80};
    constexpr std::array<double, 4> ROTATIONS = {0.381, 1.073, 1.917, 2.548};
    const double support = static_cast<double>(surfaceFootprintWidth(footprint));
    double result = 0.0;
    for (size_t index = 0; index < WAVELENGTHS.size(); ++index) {
        // The transition band avoids a topology pop when a tile refines from
        // one supported footprint to the next.
        const double retained = smoothstep(support, support * 2.0, WAVELENGTHS[index]);
        if (retained <= 0.0) continue;
        result += rotatedSimplex(reliefNoise_, x, z, WAVELENGTHS[index], 1, ROTATIONS[index]) *
                  AMPLITUDES[index] * retained;
    }
    return result;
}

void MacroGenerationSampler::applyV4SurfaceDetail(double x, double z, SurfaceFootprint footprint,
                                                  SurfaceSample& surface) const {
    if (!generationContext_) return;
    const HydrologySample& hydrology = surface.hydrology;
    if (hydrology.ocean || hydrology.river || hydrology.lake || hydrology.wetland ||
        hydrology.waterfall || hydrology.channelBank || hydrology.lakeBank || hydrology.delta ||
        hydrology.estuary || hydrology.brackish ||
        hydrology.transitionOwnerKind != WaterTransitionKind::NONE ||
        hydrology.waterSurface > surface.terrainHeight + 0.01) {
        return;
    }

    // This is a visual residual on the already solved dry surface. Wide
    // clearance bands keep it away from beds, outlets, lake rims, coasts, and
    // low-gradient divides, so it cannot delete water or create a new dam.
    const double channelClearance = hydrology.channelDistance - hydrology.channelWidth * 3.0;
    const double channelGate = smoothstep(16.0, 64.0, channelClearance);
    const double shorelineClearance =
        hydrology.shoreWaterSurface > 0.0 ? std::max(0.0, -hydrology.lakeShoreDistance) : 64.0;
    const double shorelineGate = smoothstep(16.0, 64.0, shorelineClearance);
    const double coastGate = smoothstep(8.0, 40.0, surface.terrainHeight - SEA_LEVEL);
    const double divideGate = smoothstep(0.10, 0.45, hydrology.terrainSlope);
    const double gate = channelGate * shorelineGate * coastGate * divideGate;
    if (gate <= 1.0e-9) return;

    const double requested = std::clamp(reliefDetail(x, z, footprint) * 0.18 * gate, -1.5, 1.5);
    const double previous = surface.terrainHeight;
    surface.terrainHeight = std::clamp(previous + requested, static_cast<double>(SEA_LEVEL) + 0.125,
                                       static_cast<double>(WORLD_MAX_Y));
    surface.hydrology.surfaceElevation = surface.terrainHeight;
}

double MacroGenerationSampler::provisionalRainfall(double x, double z, double elevation) const {
    if (generationContext_)
        return learnedClimateFields(generationContext_, x, z, elevation).annualPrecipitationMm;
    const double pressure = rotatedSimplex(pressureNoise_, x, z, 5400.0, 3, 0.7137243789);
    double maritime = 1.0 - smoothstep(SEA_LEVEL - 4.0, SEA_LEVEL + 80.0, elevation);
    double uplift = clamp01((elevation - SEA_LEVEL) / 180.0);
    return std::clamp(520.0 + pressure * 330.0 + maritime * 760.0 + uplift * 210.0, 80.0, 2400.0);
}

double MacroGenerationSampler::provisionalPotentialEvapotranspiration(double x, double z,
                                                                      double elevation) const {
    if (generationContext_)
        return learnedClimateFields(generationContext_, x, z, elevation)
            .potentialEvapotranspirationMm;
    constexpr double PRESSURE_DELTA = 160.0;
    const auto pressureAt = [this](double sampleX, double sampleZ) {
        return rotatedSimplex(pressureNoise_, sampleX, sampleZ, 6200.0, 3, 0.7137243789);
    };
    const double pressureX =
        (pressureAt(x + PRESSURE_DELTA, z) - pressureAt(x - PRESSURE_DELTA, z)) /
        (2.0 * PRESSURE_DELTA);
    const double pressureZ =
        (pressureAt(x, z + PRESSURE_DELTA) - pressureAt(x, z - PRESSURE_DELTA)) /
        (2.0 * PRESSURE_DELTA);
    const double windSpeed = 0.45 + clamp01(std::hypot(pressureX, pressureZ) * 1800.0) * 0.55;
    const double insolation = rotatedSimplex(insolationNoise_, x, z, 8800.0, 4, 1.3211187536);
    const double temperature = 15.0 + insolation * 26.0 +
                               std::max(0.0, elevation - SEA_LEVEL) *
                                   LEGACY_WORLD_METERS_PER_BLOCK * LEGACY_LAPSE_RATE_C_PER_METER;
    return std::clamp(300.0 + std::max(-8.0, temperature) * 31.0 + windSpeed * 170.0, 120.0,
                      1800.0);
}

BasinSolver::HydroclimateFunction MacroGenerationSampler::learnedHydroclimateFunction() const {
    if (!generationContext_) return {};
    return [this](double x, double z, double elevation) {
        const ClimateFields climate = learnedClimateFields(generationContext_, x, z, elevation);
        return BasinHydroclimateSample{
            .meanTemperatureC = climate.temperatureC,
            .temperatureVariabilityC = climate.temperatureVariabilityC,
            .annualPrecipitationMm = climate.annualPrecipitationMm,
            .precipitationCoefficientOfVariation = climate.precipitationCoefficientOfVariation,
            .lapseRateCPerMeter = climate.lapseRateCPerMeter,
            .potentialEvapotranspirationMm = climate.potentialEvapotranspirationMm,
        };
    };
}

NativeHydrologyInputFunction MacroGenerationSampler::nativeHydrologyInputFunction() const {
    if (!generationContext_)
        throw std::logic_error("native hydrology requires learned terrain authority");
    return [this, context = generationContext_](std::span<const NativeHydrologyPosition> positions,
                                                std::span<NativeHydrologyInput> output) {
        if (positions.size() != output.size())
            throw std::invalid_argument("invalid native hydrology input batch");
        const auto copyInput = [&](size_t outputIndex,
                                   const learned::PhysicalTerrainSample& sample) {
            const double potentialEvapotranspiration =
                std::clamp(300.0 + std::max(-8.0, sample.meanTemperatureC) * 31.0 +
                               sample.temperatureVariabilityC * 8.0,
                           120.0, 1'800.0);
            const NativeHydrologyPosition position = positions[outputIndex];
            output[outputIndex] = {
                .elevationMeters = preHydrologyElevationMeters(
                    static_cast<double>(position.x), static_cast<double>(position.z), sample),
                .climate =
                    {
                        .meanTemperatureC = sample.meanTemperatureC,
                        .temperatureVariabilityC = sample.temperatureVariabilityC,
                        .annualPrecipitationMm = sample.annualPrecipitationMm,
                        .precipitationCoefficientOfVariation =
                            sample.precipitationCoefficientOfVariation,
                        .lapseRateCPerMeter = sample.lapseRateCPerMeter,
                        .potentialEvapotranspirationMm = potentialEvapotranspiration,
                    },
            };
        };

        // Native authority pages arrive as one complete row-major raster at
        // the model's four-block spacing. Ask the learned backend for exactly
        // that FINAL rectangle. This preserves quantized canonical samples
        // while avoiding twelve full persistence pages that contribute only
        // the two-cell hydrology apron.
        if (!positions.empty()) {
            size_t width = 1;
            while (width < positions.size() && positions[width].z == positions.front().z)
                ++width;
            const bool rectangular = width != 0 && positions.size() % width == 0;
            const size_t height = rectangular ? positions.size() / width : 0;
            const learned::NativePoint origin =
                learned::worldBlockToNative(positions.front().x, positions.front().z);
            bool regular =
                rectangular && width <= std::numeric_limits<int64_t>::max() &&
                height <= std::numeric_limits<int64_t>::max() &&
                origin.column <=
                    std::numeric_limits<int64_t>::max() - static_cast<int64_t>(width) &&
                origin.row <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(height);
            for (size_t row = 0; regular && row < height; ++row) {
                for (size_t column = 0; column < width; ++column) {
                    const size_t index = row * width + column;
                    const learned::NativePoint point =
                        learned::worldBlockToNative(positions[index].x, positions[index].z);
                    if (point.row != origin.row + static_cast<int64_t>(row) ||
                        point.column != origin.column + static_cast<int64_t>(column)) {
                        regular = false;
                        break;
                    }
                }
            }
            if (regular) {
                const learned::NativeRect region{
                    .rowBegin = origin.row,
                    .columnBegin = origin.column,
                    .rowEnd = origin.row + static_cast<int64_t>(height),
                    .columnEnd = origin.column + static_cast<int64_t>(width),
                };
                auto result = context->queryTransientFinalNativeGrid(region);
                if (!result.isReady()) {
                    learned::GenerationFailure failure =
                        result.failure() != nullptr
                            ? *result.failure()
                            : learned::GenerationFailure{
                                  .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                                  .message = "Learned transient hydrology input returned no value "
                                             "or failure",
                                  .retriable = true,
                              };
                    if (result.status() != learned::AuthorityStatus::FAILED ||
                        failure.code != learned::GenerationFailureCode::INVALID_REQUEST) {
                        if (result.status() == learned::AuthorityStatus::FAILED)
                            context->latchFailure(failure);
                        throw learned::GenerationFailureException(result.status(),
                                                                  std::move(failure));
                    }
                } else {
                    const learned::PhysicalTerrainGrid& grid = **result.value();
                    for (size_t index = 0; index < grid.samples.size(); ++index)
                        copyInput(index, grid.samples[index]);
                    return;
                }
            }
        }

        prequeueNativeHydrologyAuthorityClosure(context, positions);
        // A five-by-five bounded hierarchy spans more than the authority's
        // 64-page query cap. Row-major chunks keep each learned query bounded
        // while retaining one deterministic native routing build.
        constexpr size_t MAXIMUM_INPUT_QUERY_POINTS = 32'768;
        for (size_t begin = 0; begin < positions.size(); begin += MAXIMUM_INPUT_QUERY_POINTS) {
            const size_t count = std::min(MAXIMUM_INPUT_QUERY_POINTS, positions.size() - begin);
            std::vector<learned::NativePoint> points;
            points.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                const NativeHydrologyPosition position = positions[begin + index];
                points.push_back(learned::worldBlockToNative(position.x, position.z));
            }
            const std::vector<learned::PhysicalTerrainSample> physical =
                requireLearnedNativeTerrainPoints(context, points);
            for (size_t index = 0; index < count; ++index)
                copyInput(begin + index, physical[index]);
        }
    };
}

double MacroGenerationSampler::sampleProvisionalRainfall(double x, double z) const {
    return provisionalRainfall(x, z, preliminaryElevation(x, z));
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
    if (generationContext_) {
        BasinSample basin;
        bool certifiedDryHit = false;
        try {
            basin = nativeHydrologyRouter_->sample(x, z, nativeHydrologyInputFunction(),
                                                   &certifiedDryHit,
                                                   generationContext_->requestPriority());
        } catch (const learned::GenerationFailureException& failure) {
            if (failure.status() == learned::AuthorityStatus::FAILED)
                generationContext_->latchFailure(failure.failure());
            throw;
        }
        if (!basin.valid) failInvalidNativeHydrology(generationContext_);
        if (!certifiedDryHit) {
            generationContext_->recordPreparedNativeHydrologyOwner(
                world_coord::floorDiv(floorToInt64(x),
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
                world_coord::floorDiv(floorToInt64(z),
                                      static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)));
        }
        HydrologySample result = hydrologyFromBasin(basin);
        const learned::PhysicalTerrainSample learnedTerrain =
            requireLearnedTerrain(generationContext_, x, z);
        applyLearnedUndisturbedTerrain(result,
                                       learned::learnedElevationMetersToWorldHeight(
                                           preHydrologyElevationMeters(x, z, learnedTerrain)),
                                       generationContext_->identity().seed);
        return result;
    }
    const BasinSample basin = legacyBasinSolver_->sample(
        x, z,
        [this](double sampleX, double sampleZ) { return preliminaryElevation(sampleX, sampleZ); },
        [this](double sampleX, double sampleZ, double elevation) {
            return provisionalRainfall(sampleX, sampleZ, elevation);
        },
        [this](double sampleX, double sampleZ) {
            return lithologyErosionResistance(sampleGeology(sampleX, sampleZ));
        },
        [this](double sampleX, double sampleZ, double elevation) {
            return provisionalPotentialEvapotranspiration(sampleX, sampleZ, elevation);
        },
        learnedHydroclimateFunction());
    if (!basin.valid) return sampleHydrologyFallback(x, z);
    return hydrologyForGenerationAuthority(basin, false);
}

void MacroGenerationSampler::prepareNativeHydrologyOwner(int64_t ownerPageX,
                                                         int64_t ownerPageZ) const {
    if (!generationContext_ || !nativeHydrologyRouter_) {
        throw std::logic_error("Native hydrology owner preparation requires generator v4");
    }
    try {
        nativeHydrologyRouter_->prepareOwner(ownerPageX, ownerPageZ, nativeHydrologyInputFunction(),
                                             generationContext_->requestPriority());
    } catch (const learned::GenerationFailureException& failure) {
        if (failure.status() == learned::AuthorityStatus::FAILED)
            generationContext_->latchFailure(failure.failure());
        throw;
    }
}

void MacroGenerationSampler::sampleHydrologyGrid(int64_t originX, int64_t originZ, int spacingX,
                                                 int spacingZ, int sampleWidth, int sampleHeight,
                                                 std::span<HydrologySample> output) const {
    sampleHydrologyGridWithTerrain(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
                                   output, {});
}

void MacroGenerationSampler::sampleNativeHydrologyTopologyGrid(
    int64_t originX, int64_t originZ, int cellWidth, int cellHeight,
    std::span<NativeHydrologyTopologyCell> output) const {
    if (cellWidth <= 0 || cellHeight <= 0 ||
        output.size() != static_cast<size_t>(cellWidth) * static_cast<size_t>(cellHeight)) {
        throw std::invalid_argument("invalid native hydrology topology output");
    }
    if (!generationContext_ || !nativeHydrologyRouter_) {
        std::fill(output.begin(), output.end(), NativeHydrologyTopologyCell{});
        return;
    }
    std::vector<uint8_t> certifiedDryHits(output.size());
    try {
        nativeHydrologyRouter_->sampleTopologyGrid(
            originX, originZ, cellWidth, cellHeight, nativeHydrologyInputFunction(), output,
            certifiedDryHits, generationContext_->requestPriority());
    } catch (const learned::GenerationFailureException& failure) {
        if (failure.status() == learned::AuthorityStatus::FAILED)
            generationContext_->latchFailure(failure.failure());
        throw;
    }
    recordPreparedNativeHydrologyGridOwners(
        generationContext_, originX, originZ, NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE,
        NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE, cellWidth, cellHeight, certifiedDryHits);
}

void MacroGenerationSampler::sampleHydrologyGridWithTerrain(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    std::span<HydrologySample> output,
    std::span<learned::PhysicalTerrainSample> learnedTerrainOutput) const {
    if (!learnedTerrainOutput.empty() && learnedTerrainOutput.size() != output.size()) {
        throw std::invalid_argument("invalid learned terrain hydrology grid output");
    }
    std::vector<BasinSample> basins(output.size());
    std::vector<uint8_t> certifiedDryHits;
    if (generationContext_) {
        certifiedDryHits.resize(output.size());
        try {
            nativeHydrologyRouter_->sampleGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                               sampleHeight, nativeHydrologyInputFunction(), basins,
                                               certifiedDryHits,
                                               generationContext_->requestPriority());
        } catch (const learned::GenerationFailureException& failure) {
            if (failure.status() == learned::AuthorityStatus::FAILED)
                generationContext_->latchFailure(failure.failure());
            throw;
        }
    } else {
        legacyBasinSolver_->sampleGrid(
            originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
            [this](double sampleX, double sampleZ) {
                return preliminaryElevation(sampleX, sampleZ);
            },
            [this](double sampleX, double sampleZ, double elevation) {
                return provisionalRainfall(sampleX, sampleZ, elevation);
            },
            [this](double sampleX, double sampleZ) {
                return lithologyErosionResistance(sampleGeology(sampleX, sampleZ));
            },
            basins,
            [this](double sampleX, double sampleZ, double elevation) {
                return provisionalPotentialEvapotranspiration(sampleX, sampleZ, elevation);
            },
            learnedHydroclimateFunction());
    }
    std::vector<learned::PhysicalTerrainSample> learnedTerrain;
    if (generationContext_) {
        if (isNativeAlignedWaterGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                     sampleHeight)) {
            learnedTerrain = requireLearnedTerrainNativeAlignedWaterGrid(
                generationContext_, originX, originZ, sampleWidth, sampleHeight);
        } else {
            std::vector<learned::WorldBlockPoint> positions;
            positions.reserve(output.size());
            for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
                for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                    positions.push_back({
                        .x = originX + static_cast<int64_t>(sampleX) * spacingX,
                        .z = originZ + static_cast<int64_t>(sampleZ) * spacingZ,
                    });
                }
            }
            learnedTerrain = requireLearnedTerrainPoints(generationContext_, positions);
        }
        if (!learnedTerrainOutput.empty()) {
            std::copy(learnedTerrain.begin(), learnedTerrain.end(), learnedTerrainOutput.begin());
        }
    }
    for (size_t index = 0; index < output.size(); ++index) {
        if (basins[index].valid) {
            output[index] = hydrologyForGenerationAuthority(basins[index],
                                                            static_cast<bool>(generationContext_));
        } else if (generationContext_) {
            failInvalidNativeHydrology(generationContext_);
        } else {
            const int sampleX = static_cast<int>(index % static_cast<size_t>(sampleWidth));
            const int sampleZ = static_cast<int>(index / static_cast<size_t>(sampleWidth));
            output[index] = sampleHydrologyFallback(
                static_cast<double>(originX + static_cast<int64_t>(sampleX) * spacingX),
                static_cast<double>(originZ + static_cast<int64_t>(sampleZ) * spacingZ));
        }
        if (generationContext_) {
            const int sampleX = static_cast<int>(index % static_cast<size_t>(sampleWidth));
            const int sampleZ = static_cast<int>(index / static_cast<size_t>(sampleWidth));
            const double worldX =
                static_cast<double>(originX + static_cast<int64_t>(sampleX) * spacingX);
            const double worldZ =
                static_cast<double>(originZ + static_cast<int64_t>(sampleZ) * spacingZ);
            applyLearnedUndisturbedTerrain(
                output[index],
                learned::learnedElevationMetersToWorldHeight(
                    preHydrologyElevationMeters(worldX, worldZ, learnedTerrain[index])),
                generationContext_->identity().seed);
        }
    }
    if (generationContext_) {
        recordPreparedNativeHydrologyGridOwners(generationContext_, originX, originZ, spacingX,
                                                spacingZ, sampleWidth, sampleHeight,
                                                certifiedDryHits);
    }
}

void MacroGenerationSampler::sampleHydrologyPoints(std::span<const ColumnPos> positions,
                                                   std::span<HydrologySample> output) const {
    sampleHydrologyPointsWithTerrain(positions, output, {});
}

void MacroGenerationSampler::sampleCoarseHydrologyPoints(std::span<const ColumnPos> positions,
                                                         std::span<HydrologySample> output) const {
    sampleHydrologyPointsWithTerrain(positions, output, {}, false);
}

void MacroGenerationSampler::certifyNativeHydrologyDryPoints(std::span<const ColumnPos> positions,
                                                             std::span<HydrologySample> output,
                                                             std::span<uint8_t> certified) const {
    if (!generationContext_ || !nativeHydrologyRouter_) {
        throw std::logic_error("Native hydrology dry certification requires generator v4");
    }
    if (positions.size() != output.size() || positions.size() != certified.size()) {
        throw std::invalid_argument("invalid native hydrology dry certificate output");
    }
    if (positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
        throw std::invalid_argument("native hydrology dry certificate exceeds its sample bound");
    }
    std::fill(output.begin(), output.end(), HydrologySample{});
    std::fill(certified.begin(), certified.end(), uint8_t{0});
    if (positions.empty()) return;
    if (!nativeHydrologyCertificatePositionsAreExact(positions)) return;

    std::vector<BasinSamplePosition> basinPositions;
    basinPositions.reserve(positions.size());
    for (const ColumnPos position : positions) {
        basinPositions.push_back(
            {.x = static_cast<double>(position.x), .z = static_cast<double>(position.z)});
    }
    std::vector<BasinSample> basins(positions.size());
    try {
        nativeHydrologyRouter_->certifyDryPoints(basinPositions, nativeHydrologyInputFunction(),
                                                 basins, certified,
                                                 generationContext_->requestPriority());
    } catch (const learned::GenerationFailureException& failure) {
        if (failure.status() == learned::AuthorityStatus::FAILED)
            generationContext_->latchFailure(failure.failure());
        throw;
    }

    std::vector<learned::WorldBlockPoint> certifiedPoints;
    std::vector<size_t> certifiedIndexes;
    certifiedPoints.reserve(positions.size());
    certifiedIndexes.reserve(positions.size());
    for (size_t index = 0; index < positions.size(); ++index) {
        if (certified[index] == 0) continue;
        certifiedPoints.push_back({.x = positions[index].x, .z = positions[index].z});
        certifiedIndexes.push_back(index);
    }
    if (certifiedPoints.empty()) return;

    const std::vector<learned::PhysicalTerrainSample> learnedTerrain =
        requireLearnedTerrainPoints(generationContext_, certifiedPoints);
    for (size_t certifiedIndex = 0; certifiedIndex < certifiedIndexes.size(); ++certifiedIndex) {
        const size_t index = certifiedIndexes[certifiedIndex];
        if (!basins[index].valid) {
            certified[index] = 0;
            continue;
        }
        output[index] = hydrologyForGenerationAuthority(basins[index], true);
        const ColumnPos position = positions[index];
        applyLearnedUndisturbedTerrain(
            output[index],
            learned::learnedElevationMetersToWorldHeight(preHydrologyElevationMeters(
                static_cast<double>(position.x), static_cast<double>(position.z),
                learnedTerrain[certifiedIndex])),
            generationContext_->identity().seed);
        if (!certifiedHydrologySampleRemainsDry(output[index])) {
            output[index] = {};
            certified[index] = 0;
        }
    }
}

void MacroGenerationSampler::certifyNativeHydrologyDryMask(std::span<const ColumnPos> positions,
                                                           std::span<uint8_t> certified) const {
    if (!generationContext_ || !nativeHydrologyRouter_) {
        throw std::logic_error("Native hydrology dry certification requires generator v4");
    }
    if (positions.size() != certified.size() ||
        positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
        throw std::invalid_argument("invalid native hydrology dry certificate mask");
    }
    std::fill(certified.begin(), certified.end(), uint8_t{0});
    if (positions.empty()) return;
    if (!nativeHydrologyCertificatePositionsAreExact(positions)) return;

    std::vector<BasinSamplePosition> basinPositions;
    basinPositions.reserve(positions.size());
    for (const ColumnPos position : positions) {
        basinPositions.push_back(
            {.x = static_cast<double>(position.x), .z = static_cast<double>(position.z)});
    }
    std::vector<BasinSample> ignored(positions.size());
    try {
        nativeHydrologyRouter_->certifyDryPoints(basinPositions, nativeHydrologyInputFunction(),
                                                 ignored, certified,
                                                 generationContext_->requestPriority());
    } catch (const learned::GenerationFailureException& failure) {
        if (failure.status() == learned::AuthorityStatus::FAILED)
            generationContext_->latchFailure(failure.failure());
        throw;
    }
}

bool MacroGenerationSampler::replaceNativeHydrologyDryFootprint(
    std::span<const ColumnPos> positions, std::span<HydrologySample> output,
    const std::function<void()>& beforeInstall) const {
    if (!generationContext_ || !nativeHydrologyRouter_) {
        throw std::logic_error("Native hydrology dry certification requires generator v4");
    }
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid native hydrology dry footprint output");
    }
    if (positions.empty() || positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
        std::fill(output.begin(), output.end(), HydrologySample{});
        return false;
    }
    std::fill(output.begin(), output.end(), HydrologySample{});
    if (!nativeHydrologyCertificatePositionsAreExact(positions)) return false;

    std::vector<BasinSamplePosition> basinPositions;
    basinPositions.reserve(positions.size());
    std::vector<learned::WorldBlockPoint> learnedPositions;
    learnedPositions.reserve(positions.size());
    for (const ColumnPos position : positions) {
        basinPositions.push_back(
            {.x = static_cast<double>(position.x), .z = static_cast<double>(position.z)});
        learnedPositions.push_back({.x = position.x, .z = position.z});
    }

    std::optional<NativeHydrologyDryFootprintCertificate> certificate;
    try {
        certificate = nativeHydrologyRouter_->certifyDryFootprint(
            basinPositions, nativeHydrologyInputFunction(), generationContext_->requestPriority());
    } catch (const learned::GenerationFailureException& failure) {
        if (failure.status() == learned::AuthorityStatus::FAILED)
            generationContext_->latchFailure(failure.failure());
        throw;
    }
    if (!certificate) return false;

    const std::vector<learned::PhysicalTerrainSample> learnedTerrain =
        requireLearnedTerrainPoints(generationContext_, learnedPositions);
    std::vector<HydrologySample> replacement(positions.size());
    for (size_t index = 0; index < positions.size(); ++index) {
        const BasinSample& basin = certificate->samples()[index];
        if (!basin.valid) return false;
        replacement[index] = hydrologyForGenerationAuthority(basin, true);
        const ColumnPos position = positions[index];
        applyLearnedUndisturbedTerrain(
            replacement[index],
            learned::learnedElevationMetersToWorldHeight(preHydrologyElevationMeters(
                static_cast<double>(position.x), static_cast<double>(position.z),
                learnedTerrain[index])),
            generationContext_->identity().seed);
        if (!certifiedHydrologySampleRemainsDry(replacement[index])) return false;
    }
    if (beforeInstall) beforeInstall();
    if (!nativeHydrologyRouter_->replaceCertifiedDryFootprint(*certificate)) return false;
    std::copy(replacement.begin(), replacement.end(), output.begin());
    return true;
}

bool MacroGenerationSampler::nativeHydrologyDryFootprintContains(
    std::span<const ColumnPos> positions) const {
    if (!generationContext_ || !nativeHydrologyRouter_) return false;
    if (positions.empty() || positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
        return false;
    }
    if (!nativeHydrologyCertificatePositionsAreExact(positions)) return false;
    std::vector<BasinSamplePosition> basinPositions;
    basinPositions.reserve(positions.size());
    for (const ColumnPos position : positions) {
        basinPositions.push_back(
            {.x = static_cast<double>(position.x), .z = static_cast<double>(position.z)});
    }
    return nativeHydrologyRouter_->certifiedDryFootprintContains(basinPositions);
}

void MacroGenerationSampler::clearNativeHydrologyDryFootprint() const {
    if (nativeHydrologyRouter_) nativeHydrologyRouter_->clearCertifiedDryFootprint();
}

void MacroGenerationSampler::sampleHydrologyPointsWithTerrain(
    std::span<const ColumnPos> positions, std::span<HydrologySample> output,
    std::span<learned::PhysicalTerrainSample> learnedTerrainOutput,
    bool reconcileOrdinaryStage) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid hydrology sample points");
    }
    if (!learnedTerrainOutput.empty() && learnedTerrainOutput.size() != output.size()) {
        throw std::invalid_argument("invalid learned terrain hydrology point output");
    }
    std::vector<BasinSamplePosition> basinPositions;
    basinPositions.reserve(positions.size());
    for (const ColumnPos position : positions)
        basinPositions.push_back(
            {static_cast<double>(position.x), static_cast<double>(position.z)});
    std::vector<BasinSample> basins(output.size());
    std::vector<uint8_t> certifiedDryHits;
    if (generationContext_) {
        certifiedDryHits.resize(output.size());
        try {
            if (reconcileOrdinaryStage) {
                nativeHydrologyRouter_->samplePoints(basinPositions, nativeHydrologyInputFunction(),
                                                     basins, certifiedDryHits,
                                                     generationContext_->requestPriority());
            } else {
                nativeHydrologyRouter_->sampleCoarsePoints(
                    basinPositions, nativeHydrologyInputFunction(), basins, certifiedDryHits,
                    generationContext_->requestPriority());
            }
        } catch (const learned::GenerationFailureException& failure) {
            if (failure.status() == learned::AuthorityStatus::FAILED)
                generationContext_->latchFailure(failure.failure());
            throw;
        }
    } else {
        legacyBasinSolver_->samplePoints(
            basinPositions,
            [this](double sampleX, double sampleZ) {
                return preliminaryElevation(sampleX, sampleZ);
            },
            [this](double sampleX, double sampleZ, double elevation) {
                return provisionalRainfall(sampleX, sampleZ, elevation);
            },
            [this](double sampleX, double sampleZ) {
                return lithologyErosionResistance(sampleGeology(sampleX, sampleZ));
            },
            basins,
            [this](double sampleX, double sampleZ, double elevation) {
                return provisionalPotentialEvapotranspiration(sampleX, sampleZ, elevation);
            },
            learnedHydroclimateFunction());
    }
    std::vector<learned::PhysicalTerrainSample> learnedTerrain;
    if (generationContext_) {
        std::vector<learned::WorldBlockPoint> points;
        points.reserve(positions.size());
        for (const ColumnPos position : positions)
            points.push_back({.x = position.x, .z = position.z});
        learnedTerrain = requireLearnedTerrainPoints(generationContext_, points);
        if (!learnedTerrainOutput.empty()) {
            std::copy(learnedTerrain.begin(), learnedTerrain.end(), learnedTerrainOutput.begin());
        }
    }
    for (size_t index = 0; index < output.size(); ++index) {
        if (basins[index].valid) {
            output[index] = hydrologyForGenerationAuthority(basins[index],
                                                            static_cast<bool>(generationContext_));
        } else if (generationContext_) {
            failInvalidNativeHydrology(generationContext_);
        } else {
            output[index] = sampleHydrologyFallback(static_cast<double>(positions[index].x),
                                                    static_cast<double>(positions[index].z));
        }
        if (generationContext_) {
            applyLearnedUndisturbedTerrain(
                output[index],
                learned::learnedElevationMetersToWorldHeight(preHydrologyElevationMeters(
                    static_cast<double>(positions[index].x),
                    static_cast<double>(positions[index].z), learnedTerrain[index])),
                generationContext_->identity().seed);
        }
    }
    recordPreparedNativeHydrologyPointOwners(generationContext_, positions, certifiedDryHits);
}

void MacroGenerationSampler::sampleSurfacePoints(std::span<const ColumnPos> positions,
                                                 SurfaceFootprint footprint,
                                                 std::span<SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid surface sample points");
    }
    if (positions.empty()) return;

    std::vector<HydrologySample> hydrology(positions.size());
    std::vector<learned::PhysicalTerrainSample> learnedTerrain;
    if (generationContext_) {
        learnedTerrain.resize(positions.size());
        sampleHydrologyPointsWithTerrain(positions, hydrology, learnedTerrain);
        for (size_t index = 0; index < positions.size(); ++index) {
            const ColumnPos position = positions[index];
            output[index] =
                surfaceFootprintWidth(footprint) >= 2
                    ? sampleSurfaceFromFarClimateControlTile(
                          nullptr, static_cast<double>(position.x), static_cast<double>(position.z),
                          footprint, &hydrology[index], &learnedTerrain[index])
                    : sampleSurfaceFromControlTile(nullptr, static_cast<double>(position.x),
                                                   static_cast<double>(position.z), footprint,
                                                   &hydrology[index], &learnedTerrain[index]);
        }
        return;
    }
    sampleHydrologyPoints(positions, hydrology);

    std::unordered_map<ColumnPos, std::shared_ptr<const MacroControlTile>> retainedControlTiles;
    std::unordered_map<ColumnPos, std::shared_ptr<const FarClimateControlTile>>
        retainedFarClimateTiles;
    retainedControlTiles.reserve(std::min(positions.size(), MACRO_CONTROL_CACHE_CAPACITY));
    retainedFarClimateTiles.reserve(std::min(positions.size(), FAR_CLIMATE_CONTROL_CACHE_CAPACITY));

    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        if (surfaceFootprintWidth(footprint) >= 2) {
            const ColumnPos key{
                world_coord::floorDiv(position.x,
                                      static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
                world_coord::floorDiv(position.z,
                                      static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
            };
            auto [tile, inserted] = retainedFarClimateTiles.try_emplace(key);
            if (inserted) tile->second = farClimateControlTile(key.x, key.z);
            output[index] = sampleSurfaceFromFarClimateControlTile(
                tile->second.get(), static_cast<double>(position.x),
                static_cast<double>(position.z), footprint, &hydrology[index]);
            continue;
        }

        const ColumnPos key{
            world_coord::floorDiv(position.x, static_cast<int64_t>(MACRO_CONTROL_TILE_EDGE)),
            world_coord::floorDiv(position.z, static_cast<int64_t>(MACRO_CONTROL_TILE_EDGE)),
        };
        auto [tile, inserted] = retainedControlTiles.try_emplace(key);
        if (inserted) tile->second = controlTile(key.x, key.z);
        output[index] = sampleSurfaceFromControlTile(
            tile->second.get(), static_cast<double>(position.x), static_cast<double>(position.z),
            footprint, &hydrology[index]);
    }
}

void MacroGenerationSampler::samplePreviewEcologyPoints(std::span<const ColumnPos> positions,
                                                        SurfaceFootprint footprint,
                                                        std::span<SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid preview ecology sample points");
    }
    if (positions.empty()) return;
    if (!generationContext_ ||
        generationContext_->quality() != learned::AuthorityQuality::PREVIEW) {
        sampleSurfacePoints(positions, footprint, output);
        return;
    }

    constexpr int64_t SLOPE_SAMPLE_OFFSET = learned::MODEL_BLOCK_SCALE;
    const auto offsetCoordinate = [](int64_t coordinate, int64_t offset) {
        const __int128 result = static_cast<__int128>(coordinate) + offset;
        if (result < std::numeric_limits<int64_t>::min() ||
            result > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("preview ecology slope sample exceeds int64 range");
        }
        return static_cast<int64_t>(result);
    };
    std::vector<learned::WorldBlockPoint> learnedPositions;
    learnedPositions.reserve(positions.size() * 5);
    for (const ColumnPos position : positions) {
        learnedPositions.push_back({.x = position.x, .z = position.z});
        learnedPositions.push_back(
            {.x = offsetCoordinate(position.x, SLOPE_SAMPLE_OFFSET), .z = position.z});
        learnedPositions.push_back(
            {.x = offsetCoordinate(position.x, -SLOPE_SAMPLE_OFFSET), .z = position.z});
        learnedPositions.push_back(
            {.x = position.x, .z = offsetCoordinate(position.z, SLOPE_SAMPLE_OFFSET)});
        learnedPositions.push_back(
            {.x = position.x, .z = offsetCoordinate(position.z, -SLOPE_SAMPLE_OFFSET)});
    }
    const std::vector<learned::PhysicalTerrainSample> learnedTerrain =
        requireLearnedTerrainPoints(generationContext_, learnedPositions);
    const auto worldHeight = [&](size_t index, double x, double z) {
        return learned::learnedElevationMetersToWorldHeight(
            preHydrologyElevationMeters(x, z, learnedTerrain[index]));
    };

    for (size_t index = 0; index < positions.size(); ++index) {
        const ColumnPos position = positions[index];
        const size_t base = index * 5;
        const double x = static_cast<double>(position.x);
        const double z = static_cast<double>(position.z);
        const double center = worldHeight(base, x, z);
        const double east = worldHeight(base + 1, x + SLOPE_SAMPLE_OFFSET, z);
        const double west = worldHeight(base + 2, x - SLOPE_SAMPLE_OFFSET, z);
        const double north = worldHeight(base + 3, x, z + SLOPE_SAMPLE_OFFSET);
        const double south = worldHeight(base + 4, x, z - SLOPE_SAMPLE_OFFSET);

        HydrologySample provisional;
        provisional.surfaceElevation = center;
        provisional.waterSurface = SEA_LEVEL;
        provisional.shoreWaterSurface = SEA_LEVEL;
        provisional.flowDirection = {1.0, 0.0};
        provisional.terrainSlope = std::hypot((east - west) / (2.0 * SLOPE_SAMPLE_OFFSET),
                                              (north - south) / (2.0 * SLOPE_SAMPLE_OFFSET));
        provisional.ocean = learnedTerrain[base].elevationMeters < 0.0;
        provisional.lakeShoreDistance = provisional.ocean ? 0.0 : -1.0e9;
        provisional.generatedFluidLevel = provisional.ocean ? 7 : 0;
        provisional.groundwaterHead = center - 12.0;
        provisional.hydroperiod = provisional.ocean ? 1.0 : 0.0;

        output[index] = surfaceFootprintWidth(footprint) >= 2
                            ? sampleSurfaceFromFarClimateControlTile(
                                  nullptr, x, z, footprint, &provisional, &learnedTerrain[base])
                            : sampleSurfaceFromControlTile(nullptr, x, z, footprint, &provisional,
                                                           &learnedTerrain[base]);
    }
}

HydrologySample MacroGenerationSampler::sampleClimateHydrology(double x, double z) const {
    HydrologySample result;
    result.surfaceElevation = preliminaryElevation(x, z);
    result.waterSurface = SEA_LEVEL;
    result.ocean = result.surfaceElevation < SEA_LEVEL;
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
    return sampleClimateWithHydrology(x, z, terrainHeight, true);
}

ClimateFields
MacroGenerationSampler::sampleClimateWithHydrology(double x, double z, double terrainHeight,
                                                   bool canonicalHydrology,
                                                   const HydrologySample* localHydrology) const {
    if (generationContext_) return learnedClimateFields(generationContext_, x, z, terrainHeight);
    ClimateFields result;
    constexpr double PRESSURE_DELTA = 96.0;
    constexpr double PRESSURE_ROTATION = 0.7137243789;
    const auto pressureAt = [this](double sampleX, double sampleZ) {
        return rotatedSimplex(pressureNoise_, sampleX, sampleZ, 6200.0, 3, PRESSURE_ROTATION);
    };
    const double pressureEast = pressureAt(x + PRESSURE_DELTA, z);
    const double pressureWest = pressureAt(x - PRESSURE_DELTA, z);
    const double pressureNorth = pressureAt(x, z + PRESSURE_DELTA);
    const double pressureSouth = pressureAt(x, z - PRESSURE_DELTA);
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
        const HydrologySample hydrology = canonicalHydrology
                                              ? sampleHydrology(sampleX, sampleZ)
                                              : (i == MOISTURE_STEPS && localHydrology != nullptr
                                                     ? *localHydrology
                                                     : sampleClimateHydrology(sampleX, sampleZ));
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

    const double localPressure = pressureAt(x, z);
    result.annualPrecipitationMm =
        std::clamp(90.0 + precipitation * 3900.0 + (localPressure + 1.0) * 130.0, 60.0, 3600.0);
    result.relativeHumidity = clamp01(moisture + result.annualPrecipitationMm / 5200.0);

    const double insolation = rotatedSimplex(insolationNoise_, x, z, 8800.0, 4, 1.3211187536);
    double maritime = static_cast<double>(waterSteps) / (MOISTURE_STEPS + 1.0);
    double continentalTemperature = 15.0 + insolation * 26.0;
    double moderatedTemperature =
        continentalTemperature * (1.0 - maritime * 0.42) + 13.0 * maritime * 0.42;
    const double lapseDelta = std::max(0.0, terrainHeight - SEA_LEVEL) *
                              LEGACY_WORLD_METERS_PER_BLOCK * LEGACY_LAPSE_RATE_C_PER_METER;
    result.temperatureC = moderatedTemperature + lapseDelta;
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

double MacroGenerationSampler::alpineSurfaceDetail(double x, double z, SurfaceFootprint footprint,
                                                   const GeologySample& geology,
                                                   const HydrologySample& hydrology,
                                                   const ClimateFields& climate) const {
    if (!legacyAlpineMorphology_) return 0.0;
    if (hydrology.ocean || hydrology.lake || hydrology.river || hydrology.wetland ||
        hydrology.waterfall || hydrology.channelBank || hydrology.lakeBank ||
        !std::isfinite(hydrology.surfaceElevation) ||
        hydrology.waterSurface > hydrology.surfaceElevation + 0.01) {
        return 0.0;
    }

    AlpineSurfaceContext context;
    context.x = x;
    context.z = z;
    context.uplift = geology.uplift;
    context.rockResistance = lithologyErosionResistance(geology);
    context.continentalFraction = geology.continentalFraction;
    context.terrainHeight = hydrology.surfaceElevation;
    context.temperatureC = climate.temperatureC;
    context.annualPrecipitationMm = climate.annualPrecipitationMm;
    context.flowX = hydrology.flowDirection.x;
    context.flowZ = hydrology.flowDirection.z;
    context.channelDistance = hydrology.channelDistance;
    context.channelWidth = hydrology.channelWidth;
    context.channelGradient = hydrology.channelGradient;
    context.discharge = hydrology.discharge;
    context.erosionDepth = hydrology.erosionDepth;
    context.footprintWidth = surfaceFootprintWidth(footprint);

    const AlpineTectonicSample tectonic = legacyAlpineMorphology_->sampleTectonic(context);
    const AlpineMorphologySample morphology =
        legacyAlpineMorphology_->sampleSurface(context, tectonic);
    const double support = static_cast<double>(surfaceFootprintWidth(footprint));
    const auto retained = [support](double wavelength) {
        return smoothstep(support, support * 2.0, wavelength);
    };

    // Cirques and glacial valleys are broad enough to remain recognizable in
    // coarse voxel tiers. Talus and crest corrugation taper sooner. The same
    // analytical feature ownership is retained at every footprint.
    const double valleyWavelength = std::clamp(96.0 + hydrology.channelWidth * 4.0, 96.0, 256.0);
    const double hornAccent =
        tectonic.hornStrength * morphology.glacialInfluence * 9.0 * retained(256.0);
    const double morphologyDetail = morphology.ridgeDetail + hornAccent +
                                    morphology.talusDeposit * retained(64.0) -
                                    morphology.valleyCarve * retained(valleyWavelength) -
                                    morphology.cirqueCarve * retained(160.0);

    // Drainage is intentionally solved on a filtered 16-block routing
    // surface. Narrow resistant summits can fall between those samples, so
    // reconstruct the positive difference between the exact pre-erosion
    // relief and the routed pre-erosion surface on dry alpine divides. The
    // recorded erosion depth remains subtracted, which preserves incision,
    // while the channel-clearance gate keeps the correction out of valleys.
    const double routedPreErosion =
        hydrology.surfaceElevation + std::max(0.0, hydrology.erosionDepth);
    const double alpineGate = smoothstep(0.72, 0.94, geology.uplift) *
                              smoothstep(0.55, 0.86, geology.continentalFraction) *
                              smoothstep(150.0, 260.0, routedPreErosion);
    const double channelClearance = hydrology.channelDistance - hydrology.channelWidth * 2.5;
    const double channelDivideGate = smoothstep(12.0, 64.0, channelClearance);
    const double shorelineClearance =
        hydrology.shoreWaterSurface > 0.0 ? std::max(0.0, -hydrology.lakeShoreDistance) : 64.0;
    const double shorelineDivideGate = smoothstep(12.0, 64.0, shorelineClearance);
    double restoredRelief = 0.0;
    if (alpineGate > 1.0e-6 && channelDivideGate > 1.0e-6 && shorelineDivideGate > 1.0e-6) {
        const double preliminary = preliminaryElevation(x, z);
        const double filteredReliefResidual = std::clamp(preliminary - routedPreErosion, 0.0, 64.0);
        restoredRelief =
            filteredReliefResidual * alpineGate * channelDivideGate * shorelineDivideGate;
    }

    return std::clamp(morphologyDetail + restoredRelief, -42.0, 72.0);
}

void MacroGenerationSampler::applyAlpineSurfaceDetail(double x, double z,
                                                      SurfaceFootprint footprint,
                                                      SurfaceSample& surface) const {
    if (generationContext_) return;
    const double detail =
        alpineSurfaceDetail(x, z, footprint, surface.geology, surface.hydrology, surface.climate);
    if (std::abs(detail) <= 1.0e-12) return;

    const double previousHeight = surface.terrainHeight;
    surface.terrainHeight = std::clamp(previousHeight + detail, -112.0, 480.0);
    surface.hydrology.surfaceElevation = surface.terrainHeight;
    const double appliedDetail = surface.terrainHeight - previousHeight;
    const double lapseDelta =
        appliedDetail * LEGACY_WORLD_METERS_PER_BLOCK * LEGACY_LAPSE_RATE_C_PER_METER;
    surface.climate.temperatureC += lapseDelta;
    surface.climate.potentialEvapotranspirationMm = std::clamp(
        surface.climate.potentialEvapotranspirationMm + lapseDelta * 31.0, 120.0, 1800.0);
    surface.climate.aridity = surface.climate.potentialEvapotranspirationMm /
                              std::max(1.0, surface.climate.annualPrecipitationMm);
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
    // Channel projections remain useful immediately beside a coast, but a
    // submerged segment is not terrestrial bank soil. Ignoring it here keeps
    // a page-handoff's underwater routing detail from changing the ocean
    // ecology or creating a soil stripe along a water-authority edge.
    const double channelWetness = hydrology.ocean ? 0.0 : channelInfluence(hydrology);
    const double wetlandWetness = wetlandInfluence(hydrology);
    result.drainage = clamp01(rockDrainage + textureNoise * 0.18 - lakeWetness * 0.45 -
                              channelWetness * 0.10 - wetlandWetness * 0.55);
    const double waterContribution =
        std::max({lakeWetness * 0.38, channelWetness * 0.34, wetlandWetness * 0.46,
                  oceanInfluence(hydrology) * 0.12});
    if (generationContext_) {
        // The learned channels carry physical annual totals. The legacy
        // synthetic climate used a broad normalized rainfall term and then
        // subtracted aridity, which effectively charged physical
        // precipitation for evaporative demand twice. That collapsed cool,
        // semiarid learned regions to exactly zero soil moisture and removed
        // their grass, scrub, and otherwise viable open woodland. Use a
        // bounded annual water balance for the PR 1 compatibility adapter.
        // Variability raises effective demand without inventing a monthly
        // phase that the model does not provide.
        const double variabilityDemand =
            std::clamp(0.68 + climate.precipitationCoefficientOfVariation * 0.32, 0.68, 1.48);
        const double effectiveDemand = climate.potentialEvapotranspirationMm * variabilityDemand;
        const double climaticMoisture =
            climate.annualPrecipitationMm /
            std::max(1.0, climate.annualPrecipitationMm + effectiveDemand);
        result.moisture = clamp01(climaticMoisture + waterContribution - result.drainage * 0.12);
    } else {
        result.moisture = clamp01(climate.annualPrecipitationMm / 2100.0 - climate.aridity * 0.19 +
                                  waterContribution - result.drainage * 0.12);
    }
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
    const double modeledWaterTable =
        hydrology.waterSurface - waterDepth -
        (4.0 + result.drainage * 22.0) * (1.0 - result.moisture * 0.65);
    result.waterTable = modeledWaterTable;
    if (hydrology.wetland) {
        // A parent-owned wetland has a solved hydraulic head. Keep the soil
        // table at that head or its inherited surface stage rather than
        // reconstructing a dry table below the newly emitted water column.
        result.waterTable =
            std::max({modeledWaterTable, hydrology.groundwaterHead, hydrology.waterSurface});
    }
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
            // Alpine exposure begins above the subalpine transition and
            // persists onto summits. A bell centered at 278 incorrectly
            // removed the alpine ecotope from the highest snow peaks.
            return smoothstep(190.0, 300.0, surface.terrainHeight) *
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

SurfaceSample MacroGenerationSampler::sampleSurfaceDirect(double x, double z,
                                                          SurfaceFootprint footprint) const {
    SurfaceSample result;
    result.geology = sampleGeology(x, z);
    result.hydrology = sampleHydrology(x, z);
    result.terrainHeight = result.hydrology.surfaceElevation;
    result.waterSurface = result.hydrology.waterSurface;
    if (!generationContext_ && result.hydrology.ocean) {
        result.terrainHeight = std::min(result.terrainHeight + reliefDetail(x, z, footprint) *
                                                                   OCEAN_FLOOR_DETAIL_SCALE,
                                        static_cast<double>(SEA_LEVEL) - 0.5);
        result.hydrology.surfaceElevation = result.terrainHeight;
    } else if (!generationContext_ && !result.hydrology.lake && !result.hydrology.river &&
               !result.hydrology.channelBank && !result.hydrology.lakeBank) {
        const double channelClearance =
            result.hydrology.channelDistance - result.hydrology.channelWidth * 2.5;
        const double dryInfluence = smoothstep(0.0, 32.0, channelClearance);
        result.terrainHeight +=
            reliefDetail(x, z, footprint) * dryInfluence * dryReliefDetailScale(result.hydrology);
        result.hydrology.surfaceElevation = result.terrainHeight;
    }
    applyV4SurfaceDetail(x, z, footprint, result);
    result.slope =
        generationContext_ ? std::max(0.0, result.hydrology.terrainSlope) : terrainSlope(x, z);
    result.climate = sampleClimate(x, z, result.terrainHeight);
    applyAlpineSurfaceDetail(x, z, footprint, result);
    result.soil = sampleSoil(x, z, result.geology, result.hydrology, result.climate);
    result.suitability = biomeSuitability(result.geology, result.hydrology, result.climate,
                                          result.soil, result.terrainHeight, result.slope);
    result.biome = selectBiome(result.suitability);

    result.ecotopes = classifyEcotopes(result);
    return result;
}

std::shared_ptr<const MacroGenerationSampler::MacroControlTile>
MacroGenerationSampler::controlTile(int64_t tileX, int64_t tileZ) const {
    const ColumnPos key{tileX, tileZ};
    return macroControlCache_->getOrCreate(key, [this, key] {
        auto tile = std::make_shared<MacroControlTile>();
        tile->key = key;
        tile->climateDetailNoise = climateDetailNoise_;
        const int64_t originX = key.x * MACRO_CONTROL_TILE_EDGE;
        const int64_t originZ = key.z * MACRO_CONTROL_TILE_EDGE;

        // One exact control needs its own hydrology, four slope neighbors, and
        // the complete canonical upwind moisture path. Retaining those points
        // in one basin batch preserves the scalar BLOCK_1 result without
        // repeating cache traversal thousands of times while nearby tree
        // anchors and column plans warm the same immutable control tile.
        constexpr size_t SLOPE_EAST = 1;
        constexpr size_t SLOPE_WEST = 2;
        constexpr size_t SLOPE_NORTH = 3;
        constexpr size_t SLOPE_SOUTH = 4;
        constexpr size_t MOISTURE_BEGIN = 5;
        constexpr size_t DEPENDENCIES_PER_CONTROL = MOISTURE_BEGIN + MOISTURE_STEPS + 1;
        std::vector<BasinSamplePosition> dependencyPositions(MACRO_CONTROL_SAMPLE_COUNT *
                                                             DEPENDENCIES_PER_CONTROL);
        std::array<Vector2d, MACRO_CONTROL_SAMPLE_COUNT> windDirections{};
        std::array<double, MACRO_CONTROL_SAMPLE_COUNT> windSpeeds{};
        const auto dependencyIndex = [](size_t controlIndex, size_t dependency) {
            return controlIndex * DEPENDENCIES_PER_CONTROL + dependency;
        };
        constexpr double PRESSURE_DELTA = 96.0;
        constexpr double PRESSURE_ROTATION = 0.7137243789;
        const auto pressureAt = [this](double x, double z) {
            return rotatedSimplex(pressureNoise_, x, z, 6200.0, 3, PRESSURE_ROTATION);
        };

        for (int controlZ = 0; controlZ < MACRO_CONTROL_GRID_EDGE; ++controlZ) {
            for (int controlX = 0; controlX < MACRO_CONTROL_GRID_EDGE; ++controlX) {
                const int64_t worldX =
                    originX + static_cast<int64_t>(controlX - 1) * MACRO_CONTROL_SPACING;
                const int64_t worldZ =
                    originZ + static_cast<int64_t>(controlZ - 1) * MACRO_CONTROL_SPACING;
                const size_t controlIndex =
                    static_cast<size_t>(controlZ * MACRO_CONTROL_GRID_EDGE + controlX);
                const double x = static_cast<double>(worldX);
                const double z = static_cast<double>(worldZ);
                const double pressureEast = pressureAt(x + PRESSURE_DELTA, z);
                const double pressureWest = pressureAt(x - PRESSURE_DELTA, z);
                const double pressureNorth = pressureAt(x, z + PRESSURE_DELTA);
                const double pressureSouth = pressureAt(x, z - PRESSURE_DELTA);
                const Vector2d gradient = {
                    (pressureEast - pressureWest) / (2.0 * PRESSURE_DELTA),
                    (pressureNorth - pressureSouth) / (2.0 * PRESSURE_DELTA),
                };
                const Vector2d rotational = {-gradient.z, gradient.x};
                const Vector2d windDirection =
                    normalized({-gradient.x + rotational.x * 0.62 + 0.00008,
                                -gradient.z + rotational.z * 0.62 + 0.00003});
                windDirections[controlIndex] = windDirection;
                windSpeeds[controlIndex] = 0.45 + clamp01(length(gradient) * 1800.0) * 0.55;

                dependencyPositions[dependencyIndex(controlIndex, 0)] = {x, z};
                dependencyPositions[dependencyIndex(controlIndex, SLOPE_EAST)] = {x + 16.0, z};
                dependencyPositions[dependencyIndex(controlIndex, SLOPE_WEST)] = {x - 16.0, z};
                dependencyPositions[dependencyIndex(controlIndex, SLOPE_NORTH)] = {x, z + 16.0};
                dependencyPositions[dependencyIndex(controlIndex, SLOPE_SOUTH)] = {x, z - 16.0};
                for (int step = 0; step <= MOISTURE_STEPS; ++step) {
                    const double distance =
                        static_cast<double>(MOISTURE_STEPS - step) * MOISTURE_STEP_DISTANCE;
                    dependencyPositions[dependencyIndex(
                        controlIndex, MOISTURE_BEGIN + static_cast<size_t>(step))] = {
                        x - windDirection.x * distance,
                        z - windDirection.z * distance,
                    };
                }
            }
        }

        std::vector<BasinSample> basinDependencies(dependencyPositions.size());
        if (generationContext_) {
            std::vector<BasinSamplePosition> canonicalPositions;
            canonicalPositions.reserve(MACRO_CONTROL_SAMPLE_COUNT * MOISTURE_BEGIN);
            for (size_t controlIndex = 0; controlIndex < MACRO_CONTROL_SAMPLE_COUNT;
                 ++controlIndex) {
                for (size_t dependency = 0; dependency < MOISTURE_BEGIN; ++dependency) {
                    canonicalPositions.push_back(
                        dependencyPositions[dependencyIndex(controlIndex, dependency)]);
                }
            }
            std::vector<BasinSample> canonicalBasins(canonicalPositions.size());
            try {
                nativeHydrologyRouter_->samplePoints(
                    canonicalPositions, nativeHydrologyInputFunction(), canonicalBasins, {},
                    generationContext_->requestPriority());
            } catch (const learned::GenerationFailureException& failure) {
                if (failure.status() == learned::AuthorityStatus::FAILED)
                    generationContext_->latchFailure(failure.failure());
                throw;
            }
            for (size_t controlIndex = 0; controlIndex < MACRO_CONTROL_SAMPLE_COUNT;
                 ++controlIndex) {
                for (size_t dependency = 0; dependency < MOISTURE_BEGIN; ++dependency) {
                    const BasinSample basin =
                        canonicalBasins[controlIndex * MOISTURE_BEGIN + dependency];
                    if (!basin.valid) failInvalidNativeHydrology(generationContext_);
                    basinDependencies[dependencyIndex(controlIndex, dependency)] = basin;
                }
                for (size_t dependency = MOISTURE_BEGIN; dependency < DEPENDENCIES_PER_CONTROL;
                     ++dependency) {
                    basinDependencies[dependencyIndex(controlIndex, dependency)] =
                        basinDependencies[dependencyIndex(controlIndex, 0)];
                }
            }
        } else {
            legacyBasinSolver_->samplePoints(
                dependencyPositions,
                [this](double x, double z) { return preliminaryElevation(x, z); },
                [this](double x, double z, double elevation) {
                    return provisionalRainfall(x, z, elevation);
                },
                [this](double x, double z) {
                    return lithologyErosionResistance(sampleGeology(x, z));
                },
                basinDependencies,
                [this](double x, double z, double elevation) {
                    return provisionalPotentialEvapotranspiration(x, z, elevation);
                },
                learnedHydroclimateFunction());
        }
        std::vector<learned::PhysicalTerrainSample> learnedDependencyTerrain;
        if (generationContext_) {
            std::vector<learned::WorldBlockPoint> points;
            points.reserve(MACRO_CONTROL_SAMPLE_COUNT * MOISTURE_BEGIN);
            for (size_t controlIndex = 0; controlIndex < MACRO_CONTROL_SAMPLE_COUNT;
                 ++controlIndex) {
                for (size_t dependency = 0; dependency < MOISTURE_BEGIN; ++dependency) {
                    const BasinSamplePosition position =
                        dependencyPositions[dependencyIndex(controlIndex, dependency)];
                    points.push_back({
                        .x = floorToInt64(position.x),
                        .z = floorToInt64(position.z),
                    });
                }
            }
            learnedDependencyTerrain = requireLearnedTerrainPoints(generationContext_, points);
        }
        std::vector<HydrologySample> hydrologyDependencies(dependencyPositions.size());
        for (size_t index = 0; index < dependencyPositions.size(); ++index) {
            hydrologyDependencies[index] =
                basinDependencies[index].valid
                    ? hydrologyForGenerationAuthority(basinDependencies[index],
                                                      static_cast<bool>(generationContext_))
                    : sampleHydrologyFallback(dependencyPositions[index].x,
                                              dependencyPositions[index].z);
        }
        if (generationContext_) {
            for (size_t controlIndex = 0; controlIndex < MACRO_CONTROL_SAMPLE_COUNT;
                 ++controlIndex) {
                for (size_t dependency = 0; dependency < MOISTURE_BEGIN; ++dependency) {
                    const size_t dependencyPosition = dependencyIndex(controlIndex, dependency);
                    const size_t learnedIndex = controlIndex * MOISTURE_BEGIN + dependency;
                    const BasinSamplePosition position = dependencyPositions[dependencyPosition];
                    applyLearnedUndisturbedTerrain(
                        hydrologyDependencies[dependencyPosition],
                        learned::learnedElevationMetersToWorldHeight(preHydrologyElevationMeters(
                            position.x, position.z, learnedDependencyTerrain[learnedIndex])),
                        generationContext_->identity().seed);
                }
            }
        }

        for (int controlZ = 0; controlZ < MACRO_CONTROL_GRID_EDGE; ++controlZ) {
            for (int controlX = 0; controlX < MACRO_CONTROL_GRID_EDGE; ++controlX) {
                const int64_t worldX =
                    originX + static_cast<int64_t>(controlX - 1) * MACRO_CONTROL_SPACING;
                const int64_t worldZ =
                    originZ + static_cast<int64_t>(controlZ - 1) * MACRO_CONTROL_SPACING;
                const size_t controlIndex =
                    static_cast<size_t>(controlZ * MACRO_CONTROL_GRID_EDGE + controlX);
                const double x = static_cast<double>(worldX);
                const double z = static_cast<double>(worldZ);
                SurfaceSample& result = tile->controls[controlIndex];
                result.geology = sampleGeology(x, z);
                result.hydrology = hydrologyDependencies[dependencyIndex(controlIndex, 0)];
                result.terrainHeight = result.hydrology.surfaceElevation;
                result.waterSurface = result.hydrology.waterSurface;
                if (!generationContext_ && result.hydrology.ocean) {
                    result.terrainHeight = std::min(
                        result.terrainHeight + reliefDetail(x, z, SurfaceFootprint::BLOCK_1) *
                                                   OCEAN_FLOOR_DETAIL_SCALE,
                        static_cast<double>(SEA_LEVEL) - 0.5);
                    result.hydrology.surfaceElevation = result.terrainHeight;
                } else if (!generationContext_ && !result.hydrology.lake &&
                           !result.hydrology.river && !result.hydrology.channelBank &&
                           !result.hydrology.lakeBank) {
                    const double channelClearance =
                        result.hydrology.channelDistance - result.hydrology.channelWidth * 2.5;
                    const double dryInfluence = smoothstep(0.0, 32.0, channelClearance);
                    result.terrainHeight += reliefDetail(x, z, SurfaceFootprint::BLOCK_1) *
                                            dryInfluence * dryReliefDetailScale(result.hydrology);
                    result.hydrology.surfaceElevation = result.terrainHeight;
                }

                const double east = hydrologyDependencies[dependencyIndex(controlIndex, SLOPE_EAST)]
                                        .surfaceElevation;
                const double west = hydrologyDependencies[dependencyIndex(controlIndex, SLOPE_WEST)]
                                        .surfaceElevation;
                const double north =
                    hydrologyDependencies[dependencyIndex(controlIndex, SLOPE_NORTH)]
                        .surfaceElevation;
                const double south =
                    hydrologyDependencies[dependencyIndex(controlIndex, SLOPE_SOUTH)]
                        .surfaceElevation;
                result.slope = std::hypot((east - west) / 32.0, (north - south) / 32.0);

                const Vector2d windDirection = windDirections[controlIndex];
                const double windSpeed = windSpeeds[controlIndex];
                result.climate.wind = {windDirection.x * windSpeed, windDirection.z * windSpeed};
                std::array<double, MOISTURE_STEPS + 1> elevations{};
                std::array<double, MOISTURE_STEPS + 1> waterRecharge{};
                double waterSteps = 0.0;
                for (int step = 0; step <= MOISTURE_STEPS; ++step) {
                    const HydrologySample& hydrology = hydrologyDependencies[dependencyIndex(
                        controlIndex, MOISTURE_BEGIN + static_cast<size_t>(step))];
                    elevations[static_cast<size_t>(step)] = hydrology.surfaceElevation;
                    const double recharge = climateWaterInfluence(hydrology);
                    waterRecharge[static_cast<size_t>(step)] = recharge;
                    waterSteps += recharge;
                }

                double moisture = 0.22;
                double precipitation = 0.0;
                for (int step = 0; step <= MOISTURE_STEPS; ++step) {
                    const double elevation = elevations[static_cast<size_t>(step)];
                    moisture += (1.0 - moisture) * 0.34 * waterRecharge[static_cast<size_t>(step)];
                    const double rise =
                        step == 0
                            ? 0.0
                            : std::max(0.0, elevation - elevations[static_cast<size_t>(step - 1)]);
                    const double descent =
                        step == 0
                            ? 0.0
                            : std::max(0.0, elevations[static_cast<size_t>(step - 1)] - elevation);
                    const double stepRain = moisture * (0.010 + std::min(0.32, rise * 0.010));
                    precipitation += stepRain;
                    moisture = std::max(0.02, moisture - stepRain);
                    moisture *= std::max(0.72, 1.0 - descent * 0.0025);
                }

                const double localPressure = pressureAt(x, z);
                result.climate.annualPrecipitationMm = std::clamp(
                    90.0 + precipitation * 3900.0 + (localPressure + 1.0) * 130.0, 60.0, 3600.0);
                result.climate.relativeHumidity =
                    clamp01(moisture + result.climate.annualPrecipitationMm / 5200.0);
                const double insolation =
                    rotatedSimplex(insolationNoise_, x, z, 8800.0, 4, 1.3211187536);
                const double maritime = waterSteps / (MOISTURE_STEPS + 1.0);
                const double continentalTemperature = 15.0 + insolation * 26.0;
                const double moderatedTemperature =
                    continentalTemperature * (1.0 - maritime * 0.42) + 13.0 * maritime * 0.42;
                const double lapseDelta = std::max(0.0, result.terrainHeight - SEA_LEVEL) *
                                          LEGACY_WORLD_METERS_PER_BLOCK *
                                          LEGACY_LAPSE_RATE_C_PER_METER;
                result.climate.temperatureC = moderatedTemperature + lapseDelta;
                result.climate.potentialEvapotranspirationMm = std::clamp(
                    300.0 + std::max(-8.0, result.climate.temperatureC) * 31.0 + windSpeed * 170.0,
                    120.0, 1800.0);
                result.climate.aridity = result.climate.potentialEvapotranspirationMm /
                                         std::max(1.0, result.climate.annualPrecipitationMm);
                if (generationContext_)
                    result.climate =
                        learnedClimateFields(generationContext_, x, z, result.terrainHeight);

                applyAlpineSurfaceDetail(x, z, SurfaceFootprint::BLOCK_1, result);
                result.soil = sampleSoil(x, z, result.geology, result.hydrology, result.climate);
                result.suitability =
                    biomeSuitability(result.geology, result.hydrology, result.climate, result.soil,
                                     result.terrainHeight, result.slope);
                result.biome = selectBiome(result.suitability);
                result.ecotopes = classifyEcotopes(result);
            }
        }
        return std::shared_ptr<const MacroControlTile>(std::move(tile));
    });
}

MacroControlView MacroGenerationSampler::controlView(ColumnPos chunkColumn) const {
    constexpr int64_t CHUNK_COLUMNS_PER_CONTROL_TILE = MACRO_CONTROL_TILE_EDGE / CHUNK_EDGE;
    const int64_t tileX = world_coord::floorDiv(chunkColumn.x, CHUNK_COLUMNS_PER_CONTROL_TILE);
    const int64_t tileZ = world_coord::floorDiv(chunkColumn.z, CHUNK_COLUMNS_PER_CONTROL_TILE);
    auto impl = std::make_shared<MacroControlView::Impl>();
    // A learned v4 view must not construct a legacy C2 control tile only to
    // discard its terrain and climate fields. It copies the bounded native
    // stencil supporting this chunk instead.
    impl->tile = generationContext_ ? nullptr : controlTile(tileX, tileZ);
    impl->generationContext = generationContext_;
    impl->chunkColumn = chunkColumn;
    return MacroControlView(std::move(impl));
}

bool MacroControlView::usesLearnedAuthority() const noexcept {
    return impl_ && impl_->generationContext;
}

void MacroControlView::prepareLearnedAuthority() const {
    if (!impl_ || !impl_->generationContext) return;
    std::call_once(impl_->learnedStencilOnce, [&] {
        constexpr int64_t NATIVE_CHUNK_EDGE = CHUNK_EDGE / learned::MODEL_BLOCK_SCALE;
        static_assert(CHUNK_EDGE % learned::MODEL_BLOCK_SCALE == 0);
        const auto checkedNativeBase = [](int64_t chunkCoordinate) -> std::optional<int64_t> {
            const __int128 value = static_cast<__int128>(chunkCoordinate) * NATIVE_CHUNK_EDGE;
            if (value < static_cast<__int128>(std::numeric_limits<int64_t>::min()) + 1 ||
                value > static_cast<__int128>(std::numeric_limits<int64_t>::max()) -
                            NATIVE_CHUNK_EDGE) {
                return std::nullopt;
            }
            return static_cast<int64_t>(value);
        };
        const std::optional<int64_t> nativeColumnBase = checkedNativeBase(impl_->chunkColumn.x);
        const std::optional<int64_t> nativeRowBase = checkedNativeBase(impl_->chunkColumn.z);
        if (!nativeColumnBase || !nativeRowBase) {
            throw learned::GenerationFailureException(
                learned::AuthorityStatus::FAILED,
                {.code = learned::GenerationFailureCode::INVALID_REQUEST,
                 .message = "Exact learned climate page coordinates are out of range",
                 .retriable = false});
        }

        // An inclusive 17-block chunk edge reconstructs from six native
        // samples on each axis: base - 1 through base + 4. Those samples can
        // cross no more than four immutable authority pages.
        const learned::TerrainPageCoordinate first = learned::terrainPageCoordinateFor(
            {.row = *nativeRowBase - 1, .column = *nativeColumnBase - 1});
        const learned::TerrainPageCoordinate last =
            learned::terrainPageCoordinateFor({.row = *nativeRowBase + NATIVE_CHUNK_EDGE,
                                               .column = *nativeColumnBase + NATIVE_CHUNK_EDGE});
        std::array<learned::TerrainPageCoordinate, 4> coordinates{};
        size_t coordinateCount = 0;
        for (int64_t row = first.row; row <= last.row; ++row) {
            for (int64_t column = first.column; column <= last.column; ++column) {
                if (coordinateCount >= coordinates.size()) {
                    throw learned::GenerationFailureException(
                        learned::AuthorityStatus::FAILED,
                        {.code = learned::GenerationFailureCode::INVALID_REQUEST,
                         .message = "Exact learned climate view exceeded four authority pages",
                         .retriable = false});
                }
                coordinates[coordinateCount++] = {.row = row, .column = column};
            }
        }
        auto closure = impl_->generationContext->requestAuthorityPages(
            std::span<const learned::TerrainPageCoordinate>(coordinates).first(coordinateCount),
            impl_->generationContext->requestPriority());
        if (!closure.isReady()) {
            learned::GenerationFailure failure =
                closure.failure() != nullptr
                    ? *closure.failure()
                    : learned::GenerationFailure{
                          .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                          .message = "Exact learned climate page closure returned no failure",
                          .retriable = true,
                      };
            throw learned::GenerationFailureException(closure.status(), std::move(failure));
        }

        struct RetainedLearnedPage {
            learned::TerrainPageCoordinate coordinate;
            std::shared_ptr<const learned::TerrainAuthorityPage> page;
        };
        std::array<RetainedLearnedPage, 4> retained{};
        size_t retainedCount = 0;
        for (const learned::TerrainPageCoordinate coordinate :
             std::span<const learned::TerrainPageCoordinate>(coordinates).first(coordinateCount)) {
            auto page = impl_->generationContext->retainAuthorityPage(coordinate);
            if (!page.isReady()) {
                learned::GenerationFailure failure =
                    page.failure() != nullptr
                        ? *page.failure()
                        : learned::GenerationFailure{
                              .code = learned::GenerationFailureCode::INFERENCE_FAILED,
                              .message = "Exact learned climate page returned no value or failure",
                              .retriable = true,
                          };
                throw learned::GenerationFailureException(page.status(), std::move(failure));
            }
            retained[retainedCount++] = {
                .coordinate = coordinate,
                .page = std::move(*page.value()),
            };
        }

        const int64_t stencilRowBegin = *nativeRowBase - 1;
        const int64_t stencilColumnBegin = *nativeColumnBase - 1;
        std::array<learned::QuantizedTerrainSample, Impl::LEARNED_STENCIL_SAMPLE_COUNT> stencil{};
        for (size_t row = 0; row < Impl::LEARNED_STENCIL_EDGE; ++row) {
            for (size_t column = 0; column < Impl::LEARNED_STENCIL_EDGE; ++column) {
                const learned::NativePoint point{
                    .row = stencilRowBegin + static_cast<int64_t>(row),
                    .column = stencilColumnBegin + static_cast<int64_t>(column),
                };
                const learned::TerrainPageCoordinate coordinate =
                    learned::terrainPageCoordinateFor(point);
                const auto retainedPage = std::find_if(
                    retained.begin(), retained.begin() + static_cast<std::ptrdiff_t>(retainedCount),
                    [coordinate](const RetainedLearnedPage& candidate) {
                        return candidate.coordinate == coordinate;
                    });
                const learned::QuantizedTerrainSample* sample =
                    retainedPage != retained.begin() + static_cast<std::ptrdiff_t>(retainedCount) &&
                            retainedPage->page
                        ? retainedPage->page->sample(
                              learned::terrainPageLocalCoordinate(point.row),
                              learned::terrainPageLocalCoordinate(point.column))
                        : nullptr;
                if (sample == nullptr) {
                    const learned::GenerationFailure failure{
                        .code = learned::GenerationFailureCode::CORRUPT_PAGE,
                        .message = "Exact learned climate stencil lost an authority sample",
                        .retriable = true,
                    };
                    impl_->generationContext->latchFailure(failure);
                    throw learned::GenerationFailureException(learned::AuthorityStatus::FAILED,
                                                              failure);
                }
                stencil[row * Impl::LEARNED_STENCIL_EDGE + column] = *sample;
            }
        }
        impl_->learnedStencil = std::move(stencil);
        impl_->learnedStencilRowBegin = stencilRowBegin;
        impl_->learnedStencilColumnBegin = stencilColumnBegin;
        impl_->learnedStencilReady = true;
    });
}

void MacroControlView::reconstructContinuous(double x, double z, SurfaceSample& destination) const {
    if (!impl_) return;
    if (impl_->generationContext) {
        if (!impl_->ownsReconstructionPoint(x, z)) {
            throw learned::GenerationFailureException(
                learned::AuthorityStatus::FAILED,
                {.code = learned::GenerationFailureCode::INVALID_REQUEST,
                 .message = "Exact learned climate reconstruction is outside its associated chunk",
                 .retriable = false});
        }
        // Learned elevation, climate, and native hydrology are the complete
        // v4 macro authority. Do not pull a slope, soil, or biome ranking
        // back through the legacy eight-block control tile after those fields
        // have been solved at native resolution.
        destination.slope = std::max(0.0, destination.hydrology.terrainSlope);
        prepareLearnedAuthority();
        struct AxisInterpolation {
            int64_t lower = 0;
            int64_t upper = 0;
            double fraction = 0.0;
        };
        const auto interpolationAxis = [](int64_t worldCoordinate) {
            const int64_t containing = world_coord::floorDiv(
                worldCoordinate, static_cast<int64_t>(learned::MODEL_BLOCK_SCALE));
            const int64_t remainder =
                worldCoordinate - containing * static_cast<int64_t>(learned::MODEL_BLOCK_SCALE);
            const double offset =
                (static_cast<double>(remainder) + 0.5) / learned::MODEL_BLOCK_SCALE - 0.5;
            if (offset < 0.0) {
                return AxisInterpolation{
                    .lower = containing - 1,
                    .upper = containing,
                    .fraction = offset + 1.0,
                };
            }
            return AxisInterpolation{
                .lower = containing,
                .upper = containing + 1,
                .fraction = offset,
            };
        };
        const AxisInterpolation row = interpolationAxis(floorToInt64(z));
        const AxisInterpolation column = interpolationAxis(floorToInt64(x));
        const std::array<learned::NativePoint, 4> points{
            learned::NativePoint{.row = row.lower, .column = column.lower},
            learned::NativePoint{.row = row.lower, .column = column.upper},
            learned::NativePoint{.row = row.upper, .column = column.lower},
            learned::NativePoint{.row = row.upper, .column = column.upper},
        };
        std::array<learned::PhysicalTerrainSample, 4> samples{};
        for (size_t index = 0; index < points.size(); ++index) {
            const learned::QuantizedTerrainSample* quantized = impl_->learnedSample(points[index]);
            if (!quantized) {
                const learned::GenerationFailure failure{
                    .code = learned::GenerationFailureCode::INVALID_REQUEST,
                    .message = "Exact learned climate stencil does not cover the requested point",
                    .retriable = false,
                };
                throw learned::GenerationFailureException(learned::AuthorityStatus::FAILED,
                                                          failure);
            }
            samples[index] = learned::dequantizeTerrainSample(*quantized);
        }
        const auto bilinear = [&](auto member) {
            const double north = std::lerp(samples[0].*member, samples[1].*member, column.fraction);
            const double south = std::lerp(samples[2].*member, samples[3].*member, column.fraction);
            return std::lerp(north, south, row.fraction);
        };
        const learned::PhysicalTerrainSample physical{
            .elevationMeters = bilinear(&learned::PhysicalTerrainSample::elevationMeters),
            .meanTemperatureC = bilinear(&learned::PhysicalTerrainSample::meanTemperatureC),
            .temperatureVariabilityC =
                bilinear(&learned::PhysicalTerrainSample::temperatureVariabilityC),
            .annualPrecipitationMm =
                bilinear(&learned::PhysicalTerrainSample::annualPrecipitationMm),
            .precipitationCoefficientOfVariation =
                bilinear(&learned::PhysicalTerrainSample::precipitationCoefficientOfVariation),
            .lapseRateCPerMeter = bilinear(&learned::PhysicalTerrainSample::lapseRateCPerMeter),
        };
        destination.climate = learnedClimateFields(physical, destination.terrainHeight);
        return;
    }
    if (!impl_->tile) return;
    reconstructContinuousFields(*impl_->tile, x, z, destination);
}

SurfaceSample MacroGenerationSampler::sampleSurfaceFromControlTile(
    const MacroControlTile* tile, double x, double z, SurfaceFootprint footprint,
    const HydrologySample* hydrology, const learned::PhysicalTerrainSample* learnedTerrain) const {
    // Ownership and hydrologic topology remain direct coordinate queries.
    // Continuous geology, climate, soil, suitability, and slope fields share
    // the immutable tile reconstruction used by exact ColumnPlans.
    SurfaceSample result;
    result.geology = sampleGeology(x, z);
    result.hydrology = hydrology != nullptr ? *hydrology : sampleHydrology(x, z);
    result.terrainHeight = result.hydrology.surfaceElevation;
    result.waterSurface = result.hydrology.waterSurface;
    if (!generationContext_ && result.hydrology.ocean) {
        result.terrainHeight = std::min(result.terrainHeight + reliefDetail(x, z, footprint) *
                                                                   OCEAN_FLOOR_DETAIL_SCALE,
                                        static_cast<double>(SEA_LEVEL) - 0.5);
        result.hydrology.surfaceElevation = result.terrainHeight;
    } else if (!generationContext_ && !result.hydrology.lake && !result.hydrology.river &&
               !result.hydrology.channelBank && !result.hydrology.lakeBank) {
        const double channelClearance =
            result.hydrology.channelDistance - result.hydrology.channelWidth * 2.5;
        const double dryInfluence = smoothstep(0.0, 32.0, channelClearance);
        result.terrainHeight +=
            reliefDetail(x, z, footprint) * dryInfluence * dryReliefDetailScale(result.hydrology);
        result.hydrology.surfaceElevation = result.terrainHeight;
    }
    applyV4SurfaceDetail(x, z, footprint, result);

    if (generationContext_) {
        result.slope = std::max(0.0, result.hydrology.terrainSlope);
        result.climate = learnedTerrain != nullptr
                             ? learnedClimateFields(*learnedTerrain, result.terrainHeight)
                             : learnedClimateFields(generationContext_, x, z, result.terrainHeight);
    } else {
        if (tile == nullptr)
            throw std::logic_error("legacy surface sample is missing a control tile");
        reconstructContinuousFields(*tile, x, z, result);
    }
    applyAlpineSurfaceDetail(x, z, footprint, result);
    result.soil = sampleSoil(x, z, result.geology, result.hydrology, result.climate);
    result.suitability = biomeSuitability(result.geology, result.hydrology, result.climate,
                                          result.soil, result.terrainHeight, result.slope);
    result.biome = selectBiome(result.suitability);
    result.ecotopes = classifyEcotopes(result);
    return result;
}

std::shared_ptr<const MacroGenerationSampler::FarClimateControlTile>
MacroGenerationSampler::farClimateControlTile(int64_t tileX, int64_t tileZ) const {
    const ColumnPos key{tileX, tileZ};
    return farClimateControlCache_->getOrCreate(key, [this, key] {
        auto tile = std::make_shared<FarClimateControlTile>();
        tile->key = key;
        tile->climateDetailNoise = climateDetailNoise_;
        const int64_t originX = key.x * FAR_CLIMATE_CONTROL_TILE_EDGE;
        const int64_t originZ = key.z * FAR_CLIMATE_CONTROL_TILE_EDGE;
        std::array<HydrologySample, FAR_CLIMATE_CONTROL_SAMPLE_COUNT> hydrologyControls;
        std::array<learned::PhysicalTerrainSample, FAR_CLIMATE_CONTROL_SAMPLE_COUNT>
            learnedTerrainControls;
        const std::span<learned::PhysicalTerrainSample> learnedTerrainOutput =
            generationContext_ ? std::span<learned::PhysicalTerrainSample>(learnedTerrainControls)
                               : std::span<learned::PhysicalTerrainSample>{};
        sampleHydrologyGridWithTerrain(
            originX - FAR_CLIMATE_CONTROL_SPACING, originZ - FAR_CLIMATE_CONTROL_SPACING,
            FAR_CLIMATE_CONTROL_SPACING, FAR_CLIMATE_CONTROL_SPACING, FAR_CLIMATE_CONTROL_GRID_EDGE,
            FAR_CLIMATE_CONTROL_GRID_EDGE, hydrologyControls, learnedTerrainOutput);
        for (int controlZ = 0; controlZ < FAR_CLIMATE_CONTROL_GRID_EDGE; ++controlZ) {
            for (int controlX = 0; controlX < FAR_CLIMATE_CONTROL_GRID_EDGE; ++controlX) {
                const int64_t worldX =
                    originX + static_cast<int64_t>(controlX - 1) * FAR_CLIMATE_CONTROL_SPACING;
                const int64_t worldZ =
                    originZ + static_cast<int64_t>(controlZ - 1) * FAR_CLIMATE_CONTROL_SPACING;
                const double sampleX = static_cast<double>(worldX);
                const double sampleZ = static_cast<double>(worldZ);
                const size_t controlIndex =
                    static_cast<size_t>(controlZ * FAR_CLIMATE_CONTROL_GRID_EDGE + controlX);
                const HydrologySample& hydrology = hydrologyControls[controlIndex];
                double terrainHeight = hydrology.surfaceElevation;
                if (!generationContext_ && hydrology.ocean) {
                    terrainHeight = std::min(
                        terrainHeight + reliefDetail(sampleX, sampleZ, SurfaceFootprint::BLOCK_16) *
                                            OCEAN_FLOOR_DETAIL_SCALE,
                        static_cast<double>(SEA_LEVEL) - 0.5);
                } else if (!generationContext_ && !hydrology.lake && !hydrology.river &&
                           !hydrology.channelBank && !hydrology.lakeBank) {
                    const double channelClearance =
                        hydrology.channelDistance - hydrology.channelWidth * 2.5;
                    terrainHeight += reliefDetail(sampleX, sampleZ, SurfaceFootprint::BLOCK_16) *
                                     smoothstep(0.0, 32.0, channelClearance) *
                                     dryReliefDetailScale(hydrology);
                }
                FarClimateControlTile::Control& control = tile->controls[static_cast<size_t>(
                    controlZ * FAR_CLIMATE_CONTROL_GRID_EDGE + controlX)];
                SurfaceSample surface;
                surface.geology = sampleGeology(sampleX, sampleZ);
                surface.hydrology = hydrology;
                surface.terrainHeight = terrainHeight;
                surface.waterSurface = hydrology.waterSurface;
                applyV4SurfaceDetail(sampleX, sampleZ, SurfaceFootprint::BLOCK_16, surface);
                surface.climate = generationContext_
                                      ? learnedClimateFields(learnedTerrainControls[controlIndex],
                                                             surface.terrainHeight)
                                      : sampleClimateWithHydrology(sampleX, sampleZ, terrainHeight,
                                                                   false, &hydrology);
                applyAlpineSurfaceDetail(sampleX, sampleZ, SurfaceFootprint::BLOCK_16, surface);
                control.terrainHeight = surface.terrainHeight;
                control.climate = surface.climate;
            }
        }
        return std::shared_ptr<const FarClimateControlTile>(std::move(tile));
    });
}

SurfaceSample MacroGenerationSampler::sampleSurfaceFromFarClimateControlTile(
    const FarClimateControlTile* tile, double x, double z, SurfaceFootprint footprint,
    const HydrologySample* hydrology, const learned::PhysicalTerrainSample* learnedTerrain) const {
    SurfaceSample result;
    result.geology = sampleGeology(x, z);
    result.hydrology = hydrology != nullptr ? *hydrology : sampleHydrology(x, z);
    result.terrainHeight = result.hydrology.surfaceElevation;
    result.waterSurface = result.hydrology.waterSurface;
    if (!generationContext_ && result.hydrology.ocean) {
        result.terrainHeight = std::min(result.terrainHeight + reliefDetail(x, z, footprint) *
                                                                   OCEAN_FLOOR_DETAIL_SCALE,
                                        static_cast<double>(SEA_LEVEL) - 0.5);
        result.hydrology.surfaceElevation = result.terrainHeight;
    } else if (!generationContext_ && !result.hydrology.lake && !result.hydrology.river &&
               !result.hydrology.channelBank && !result.hydrology.lakeBank) {
        const double channelClearance =
            result.hydrology.channelDistance - result.hydrology.channelWidth * 2.5;
        const double dryInfluence = smoothstep(0.0, 32.0, channelClearance);
        result.terrainHeight +=
            reliefDetail(x, z, footprint) * dryInfluence * dryReliefDetailScale(result.hydrology);
        result.hydrology.surfaceElevation = result.terrainHeight;
    }
    applyV4SurfaceDetail(x, z, footprint, result);

    if (generationContext_) {
        result.slope = std::max(0.0, result.hydrology.terrainSlope);
        result.climate = learnedTerrain != nullptr
                             ? learnedClimateFields(*learnedTerrain, result.terrainHeight)
                             : learnedClimateFields(generationContext_, x, z, result.terrainHeight);
    } else {
        if (tile == nullptr)
            throw std::logic_error("legacy far surface sample is missing a climate control tile");
        const double originX = static_cast<double>(tile->key.x * FAR_CLIMATE_CONTROL_TILE_EDGE);
        const double originZ = static_cast<double>(tile->key.z * FAR_CLIMATE_CONTROL_TILE_EDGE);
        const CubicControlStencil stencil = cubicControlStencil(
            x, z, originX, originZ, FAR_CLIMATE_CONTROL_SPACING, FAR_CLIMATE_CONTROL_CORE_EDGE);
        const Vector2d terrainGradient = stencil.gradient(
            [&](int controlX, int controlZ) { return tile->at(controlX, controlZ).terrainHeight; });
        result.slope = std::hypot(terrainGradient.x, terrainGradient.z);
        reconstructClimate(
            result, stencil,
            [&](int controlX, int controlZ) { return tile->at(controlX, controlZ).terrainHeight; },
            [&](int controlX, int controlZ) -> const ClimateFields& {
                return tile->at(controlX, controlZ).climate;
            });
    }
    applyAlpineSurfaceDetail(x, z, footprint, result);
    result.soil = sampleSoil(x, z, result.geology, result.hydrology, result.climate);
    result.suitability = biomeSuitability(result.geology, result.hydrology, result.climate,
                                          result.soil, result.terrainHeight, result.slope);
    result.biome = selectBiome(result.suitability);
    result.ecotopes = classifyEcotopes(result);
    return result;
}

SurfaceSample MacroGenerationSampler::sampleSurface(double x, double z,
                                                    SurfaceFootprint footprint) const {
    if (generationContext_) return sampleSurfaceDirect(x, z, footprint);
    if (surfaceFootprintWidth(footprint) >= 2) {
        const int64_t tileX = floorToInt64(x / FAR_CLIMATE_CONTROL_TILE_EDGE);
        const int64_t tileZ = floorToInt64(z / FAR_CLIMATE_CONTROL_TILE_EDGE);
        std::array<HydrologySample, 1> hydrology;
        const bool integerCoordinate =
            static_cast<double>(floorToInt64(x)) == x && static_cast<double>(floorToInt64(z)) == z;
        if (integerCoordinate) {
            sampleHydrologyGrid(floorToInt64(x), floorToInt64(z), 1, 1, 1, 1, hydrology);
        }
        return sampleSurfaceFromFarClimateControlTile(
            farClimateControlTile(tileX, tileZ).get(), x, z, footprint,
            integerCoordinate ? hydrology.data() : nullptr);
    }
    const int64_t tileX = floorToInt64(x / MACRO_CONTROL_TILE_EDGE);
    const int64_t tileZ = floorToInt64(z / MACRO_CONTROL_TILE_EDGE);
    return sampleSurfaceFromControlTile(controlTile(tileX, tileZ).get(), x, z, footprint);
}

void MacroGenerationSampler::sampleSurfaceGrid(int64_t originX, int64_t originZ, int spacing,
                                               int sampleEdge, SurfaceFootprint footprint,
                                               std::span<SurfaceSample> output) const {
    if (spacing <= 0 || sampleEdge <= 0 ||
        output.size() != static_cast<size_t>(sampleEdge * sampleEdge)) {
        throw std::invalid_argument("invalid macro surface grid");
    }

    std::unordered_map<ColumnPos, std::shared_ptr<const MacroControlTile>> retainedTiles;
    std::unordered_map<ColumnPos, std::shared_ptr<const FarClimateControlTile>>
        retainedFarClimateTiles;
    // Group basin/page lookups once for every footprint. Calling the scalar
    // hydrology path for each of the 16,641 samples in a step-2 tile repeated
    // cache synchronization and made the nearest far ring take nearly a
    // second even after terrain generation moved off the render thread. The
    // final learned samples are retained alongside hydrology so climate does
    // not issue one scalar authority query per far vertex.
    std::vector<HydrologySample> gridHydrology(output.size());
    std::vector<learned::PhysicalTerrainSample> gridLearnedTerrain;
    if (generationContext_) gridLearnedTerrain.resize(output.size());
    sampleHydrologyGridWithTerrain(originX, originZ, spacing, spacing, sampleEdge, sampleEdge,
                                   gridHydrology,
                                   std::span<learned::PhysicalTerrainSample>(gridLearnedTerrain));
    if (generationContext_) {
        for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
                const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
                const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
                const size_t index = static_cast<size_t>(sampleZ * sampleEdge + sampleX);
                output[index] =
                    surfaceFootprintWidth(footprint) >= 2
                        ? sampleSurfaceFromFarClimateControlTile(
                              nullptr, static_cast<double>(worldX), static_cast<double>(worldZ),
                              footprint, &gridHydrology[index], &gridLearnedTerrain[index])
                        : sampleSurfaceFromControlTile(
                              nullptr, static_cast<double>(worldX), static_cast<double>(worldZ),
                              footprint, &gridHydrology[index], &gridLearnedTerrain[index]);
            }
        }
        return;
    }
    const int64_t gridWidth = static_cast<int64_t>(sampleEdge - 1) * spacing;
    const size_t approximateTileSpan = static_cast<size_t>(gridWidth / MACRO_CONTROL_TILE_EDGE + 2);
    retainedTiles.reserve(approximateTileSpan * approximateTileSpan);
    for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
            const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
            const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
            const size_t index = static_cast<size_t>(sampleZ * sampleEdge + sampleX);
            SurfaceSample& destination = output[index];
            const learned::PhysicalTerrainSample* learnedTerrain =
                generationContext_ ? &gridLearnedTerrain[index] : nullptr;
            if (surfaceFootprintWidth(footprint) >= 2) {
                const ColumnPos key{
                    world_coord::floorDiv(worldX,
                                          static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
                    world_coord::floorDiv(worldZ,
                                          static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
                };
                auto [tile, inserted] = retainedFarClimateTiles.try_emplace(key);
                if (inserted) tile->second = farClimateControlTile(key.x, key.z);
                destination = sampleSurfaceFromFarClimateControlTile(
                    tile->second.get(), static_cast<double>(worldX), static_cast<double>(worldZ),
                    footprint, &gridHydrology[index], learnedTerrain);
                continue;
            }
            const ColumnPos key{
                world_coord::floorDiv(worldX, static_cast<int64_t>(MACRO_CONTROL_TILE_EDGE)),
                world_coord::floorDiv(worldZ, static_cast<int64_t>(MACRO_CONTROL_TILE_EDGE)),
            };
            auto [tile, inserted] = retainedTiles.try_emplace(key);
            if (inserted) tile->second = controlTile(key.x, key.z);
            destination = sampleSurfaceFromControlTile(
                tile->second.get(), static_cast<double>(worldX), static_cast<double>(worldZ),
                footprint, &gridHydrology[index], learnedTerrain);
        }
    }
}

void MacroGenerationSampler::sampleClimateGrid(int64_t originX, int64_t originZ, int spacing,
                                               int sampleEdge,
                                               std::span<SurfaceSample> output) const {
    if (spacing <= 0 || sampleEdge <= 0 ||
        output.size() != static_cast<size_t>(sampleEdge) * static_cast<size_t>(sampleEdge)) {
        throw std::invalid_argument("invalid macro climate grid");
    }
    if (!generationContext_) {
        sampleSurfaceGrid(originX, originZ, spacing, sampleEdge, SurfaceFootprint::BLOCK_32,
                          output);
        return;
    }

    // Four weather rows span one 1,024-block authority page at the canonical
    // 256-block weather spacing. Limiting each tile to 32 columns keeps even a
    // bilinear page-boundary crossing below 24 pages, leaving coordinator room
    // for protected exact and visible refinement work. Weather is optional
    // presentation enrichment and therefore enters the lowest priority lane.
    constexpr int MAXIMUM_ROWS_PER_QUERY = 4;
    constexpr int MAXIMUM_COLUMNS_PER_QUERY = 32;
    for (int rowBegin = 0; rowBegin < sampleEdge; rowBegin += MAXIMUM_ROWS_PER_QUERY) {
        const int rowEnd = std::min(sampleEdge, rowBegin + MAXIMUM_ROWS_PER_QUERY);
        for (int columnBegin = 0; columnBegin < sampleEdge;
             columnBegin += MAXIMUM_COLUMNS_PER_QUERY) {
            const int columnEnd = std::min(sampleEdge, columnBegin + MAXIMUM_COLUMNS_PER_QUERY);
            std::vector<learned::WorldBlockPoint> points;
            points.reserve(static_cast<size_t>(rowEnd - rowBegin) *
                           static_cast<size_t>(columnEnd - columnBegin));
            for (int sampleZ = rowBegin; sampleZ < rowEnd; ++sampleZ) {
                for (int sampleX = columnBegin; sampleX < columnEnd; ++sampleX) {
                    points.push_back({
                        .x = originX + static_cast<int64_t>(sampleX) * spacing,
                        .z = originZ + static_cast<int64_t>(sampleZ) * spacing,
                    });
                }
            }
            const std::vector<learned::PhysicalTerrainSample> physical =
                requireLearnedTerrainPoints(
                    generationContext_, points,
                    learned::AuthorityRequestPriority::SPECULATIVE_PREFETCH);
            if (physical.size() != points.size())
                throw std::logic_error("learned climate query returned the wrong sample count");

            const size_t batchWidth = static_cast<size_t>(columnEnd - columnBegin);
            for (size_t batchIndex = 0; batchIndex < physical.size(); ++batchIndex) {
                const size_t sampleZ = static_cast<size_t>(rowBegin) + batchIndex / batchWidth;
                const size_t sampleX = static_cast<size_t>(columnBegin) + batchIndex % batchWidth;
                SurfaceSample& destination =
                    output[sampleZ * static_cast<size_t>(sampleEdge) + sampleX];
                destination = {};
                destination.terrainHeight = learned::learnedElevationMetersToWorldHeight(
                    physical[batchIndex].elevationMeters);
                destination.climate =
                    learnedClimateFields(physical[batchIndex], destination.terrainHeight);
            }
        }
    }
}

void MacroGenerationSampler::sampleGeometryGrid(int64_t originX, int64_t originZ, int spacingX,
                                                int spacingZ, int sampleWidth, int sampleHeight,
                                                SurfaceFootprint footprint,
                                                std::span<SurfaceSample> output) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid macro geometry grid");
    }
    std::vector<HydrologySample> hydrology(output.size());
    std::vector<learned::PhysicalTerrainSample> learnedTerrain;
    if (generationContext_) learnedTerrain.resize(output.size());
    sampleHydrologyGridWithTerrain(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
                                   hydrology,
                                   std::span<learned::PhysicalTerrainSample>(learnedTerrain));
    if (generationContext_) {
        for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
                const double x =
                    static_cast<double>(originX + static_cast<int64_t>(sampleX) * spacingX);
                const double z =
                    static_cast<double>(originZ + static_cast<int64_t>(sampleZ) * spacingZ);
                SurfaceSample& destination = output[index];
                destination.geology = sampleGeology(x, z);
                destination.hydrology = hydrology[index];
                destination.terrainHeight = destination.hydrology.surfaceElevation;
                destination.waterSurface = destination.hydrology.waterSurface;
                applyV4SurfaceDetail(x, z, footprint, destination);
                destination.slope = std::max(0.0, destination.hydrology.terrainSlope);
                destination.climate =
                    learnedClimateFields(learnedTerrain[index], destination.terrainHeight);
                applyAlpineSurfaceDetail(x, z, footprint, destination);
            }
        }
        return;
    }
    std::unordered_map<ColumnPos, std::shared_ptr<const FarClimateControlTile>> retainedTiles;
    const int64_t gridWidth = static_cast<int64_t>(sampleWidth - 1) * spacingX;
    const int64_t gridHeight = static_cast<int64_t>(sampleHeight - 1) * spacingZ;
    const size_t tileSpanX = static_cast<size_t>(gridWidth / FAR_CLIMATE_CONTROL_TILE_EDGE + 2);
    const size_t tileSpanZ = static_cast<size_t>(gridHeight / FAR_CLIMATE_CONTROL_TILE_EDGE + 2);
    retainedTiles.reserve(tileSpanX * tileSpanZ);
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * sampleWidth + sampleX);
            const double x =
                static_cast<double>(originX + static_cast<int64_t>(sampleX) * spacingX);
            const double z =
                static_cast<double>(originZ + static_cast<int64_t>(sampleZ) * spacingZ);
            SurfaceSample& destination = output[index];
            destination.geology = sampleGeology(x, z);
            destination.hydrology = hydrology[index];
            destination.terrainHeight = destination.hydrology.surfaceElevation;
            destination.waterSurface = destination.hydrology.waterSurface;
            if (!generationContext_ && footprint != SurfaceFootprint::BLOCK_1 &&
                destination.hydrology.ocean) {
                destination.terrainHeight =
                    std::min(destination.terrainHeight +
                                 reliefDetail(x, z, footprint) * OCEAN_FLOOR_DETAIL_SCALE,
                             static_cast<double>(SEA_LEVEL) - 0.5);
                destination.hydrology.surfaceElevation = destination.terrainHeight;
            } else if (!generationContext_ && footprint != SurfaceFootprint::BLOCK_1 &&
                       !destination.hydrology.lake && !destination.hydrology.river) {
                const double channelClearance = destination.hydrology.channelDistance -
                                                destination.hydrology.channelWidth * 2.5;
                destination.terrainHeight += reliefDetail(x, z, footprint) *
                                             smoothstep(0.0, 32.0, channelClearance) *
                                             dryReliefDetailScale(destination.hydrology);
                destination.hydrology.surfaceElevation = destination.terrainHeight;
            }

            const ColumnPos key{
                world_coord::floorDiv(static_cast<int64_t>(x),
                                      static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
                world_coord::floorDiv(static_cast<int64_t>(z),
                                      static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
            };
            auto [tile, inserted] = retainedTiles.try_emplace(key);
            if (inserted) tile->second = farClimateControlTile(key.x, key.z);
            const double tileOriginX = static_cast<double>(key.x * FAR_CLIMATE_CONTROL_TILE_EDGE);
            const double tileOriginZ = static_cast<double>(key.z * FAR_CLIMATE_CONTROL_TILE_EDGE);
            const CubicControlStencil stencil =
                cubicControlStencil(x, z, tileOriginX, tileOriginZ, FAR_CLIMATE_CONTROL_SPACING,
                                    FAR_CLIMATE_CONTROL_CORE_EDGE);
            reconstructClimate(
                destination, stencil,
                [&](int controlX, int controlZ) {
                    return tile->second->at(controlX, controlZ).terrainHeight;
                },
                [&](int controlX, int controlZ) -> const ClimateFields& {
                    return tile->second->at(controlX, controlZ).climate;
                });
            if (generationContext_)
                destination.climate =
                    learnedClimateFields(learnedTerrain[index], destination.terrainHeight);
            applyAlpineSurfaceDetail(x, z, footprint, destination);
        }
    }
}

void MacroGenerationSampler::sampleGeometryPoints(std::span<const ColumnPos> positions,
                                                  SurfaceFootprint footprint,
                                                  std::span<SurfaceSample> output) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid macro geometry sample points");
    }
    if (positions.empty()) return;

    std::vector<HydrologySample> hydrology(output.size());
    std::vector<learned::PhysicalTerrainSample> learnedTerrain;
    if (generationContext_) learnedTerrain.resize(output.size());
    sampleHydrologyPointsWithTerrain(positions, hydrology,
                                     std::span<learned::PhysicalTerrainSample>(learnedTerrain));
    if (generationContext_) {
        for (size_t index = 0; index < positions.size(); ++index) {
            const double x = static_cast<double>(positions[index].x);
            const double z = static_cast<double>(positions[index].z);
            SurfaceSample& destination = output[index];
            destination.geology = sampleGeology(x, z);
            destination.hydrology = hydrology[index];
            destination.terrainHeight = destination.hydrology.surfaceElevation;
            destination.waterSurface = destination.hydrology.waterSurface;
            applyV4SurfaceDetail(x, z, footprint, destination);
            destination.slope = std::max(0.0, destination.hydrology.terrainSlope);
            destination.climate =
                learnedClimateFields(learnedTerrain[index], destination.terrainHeight);
            applyAlpineSurfaceDetail(x, z, footprint, destination);
        }
        return;
    }
    std::unordered_map<ColumnPos, std::shared_ptr<const FarClimateControlTile>> retainedTiles;
    retainedTiles.reserve(std::min(positions.size(), FAR_CLIMATE_CONTROL_CACHE_CAPACITY));

    for (size_t index = 0; index < positions.size(); ++index) {
        const double x = static_cast<double>(positions[index].x);
        const double z = static_cast<double>(positions[index].z);
        SurfaceSample& destination = output[index];
        destination.geology = sampleGeology(x, z);
        destination.hydrology = hydrology[index];
        destination.terrainHeight = destination.hydrology.surfaceElevation;
        destination.waterSurface = destination.hydrology.waterSurface;
        if (!generationContext_ && footprint != SurfaceFootprint::BLOCK_1 &&
            destination.hydrology.ocean) {
            destination.terrainHeight =
                std::min(destination.terrainHeight +
                             reliefDetail(x, z, footprint) * OCEAN_FLOOR_DETAIL_SCALE,
                         static_cast<double>(SEA_LEVEL) - 0.5);
            destination.hydrology.surfaceElevation = destination.terrainHeight;
        } else if (!generationContext_ && footprint != SurfaceFootprint::BLOCK_1 &&
                   !destination.hydrology.lake && !destination.hydrology.river) {
            const double channelClearance =
                destination.hydrology.channelDistance - destination.hydrology.channelWidth * 2.5;
            destination.terrainHeight += reliefDetail(x, z, footprint) *
                                         smoothstep(0.0, 32.0, channelClearance) *
                                         dryReliefDetailScale(destination.hydrology);
            destination.hydrology.surfaceElevation = destination.terrainHeight;
        }

        const ColumnPos key{
            world_coord::floorDiv(positions[index].x,
                                  static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
            world_coord::floorDiv(positions[index].z,
                                  static_cast<int64_t>(FAR_CLIMATE_CONTROL_TILE_EDGE)),
        };
        auto [tile, inserted] = retainedTiles.try_emplace(key);
        if (inserted) tile->second = farClimateControlTile(key.x, key.z);
        const double tileOriginX = static_cast<double>(key.x * FAR_CLIMATE_CONTROL_TILE_EDGE);
        const double tileOriginZ = static_cast<double>(key.z * FAR_CLIMATE_CONTROL_TILE_EDGE);
        const CubicControlStencil stencil =
            cubicControlStencil(x, z, tileOriginX, tileOriginZ, FAR_CLIMATE_CONTROL_SPACING,
                                FAR_CLIMATE_CONTROL_CORE_EDGE);
        reconstructClimate(
            destination, stencil,
            [&](int controlX, int controlZ) {
                return tile->second->at(controlX, controlZ).terrainHeight;
            },
            [&](int controlX, int controlZ) -> const ClimateFields& {
                return tile->second->at(controlX, controlZ).climate;
            });
        if (generationContext_)
            destination.climate =
                learnedClimateFields(learnedTerrain[index], destination.terrainHeight);
        applyAlpineSurfaceDetail(x, z, footprint, destination);
    }
}

MacroControlCacheMetrics MacroGenerationSampler::macroControlCacheMetrics() const {
    return macroControlCache_->metrics();
}

MacroControlCacheMetrics MacroGenerationSampler::farClimateControlCacheMetrics() const {
    return farClimateControlCache_->metrics();
}

NativeHydrologyCacheMetrics MacroGenerationSampler::nativeHydrologyCacheMetrics() const {
    return nativeHydrologyRouter_ ? nativeHydrologyRouter_->cacheMetrics()
                                  : NativeHydrologyCacheMetrics{};
}

void MacroGenerationSampler::clearMacroControlCache() const {
    macroControlCache_->clear();
    farClimateControlCache_->clear();
}

void MacroGenerationSampler::clearBasinCache() const {
    if (legacyBasinSolver_) legacyBasinSolver_->clear();
    if (nativeHydrologyRouter_) nativeHydrologyRouter_->clear();
}

} // namespace worldgen
