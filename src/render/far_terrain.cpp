#include "render/far_terrain.hpp"

#include "common/thread_priority.hpp"
#include "render/block_textures.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {

enum class FarWaterKind : uint8_t {
    NONE,
    OCEAN,
    RIVER,
    LAKE,
};

struct FarWaterAuthority {
    worldgen::WaterBodyId bodyId = worldgen::NO_WATER_BODY;
    uint64_t transitionOwnerId = 0;
    float height = SEA_LEVEL;
    FarWaterKind kind = FarWaterKind::NONE;
    worldgen::WaterTransitionKind transitionOwnerKind = worldgen::WaterTransitionKind::NONE;
    uint8_t generatedFluidLevel = 0;
    bool delta = false;
};

constexpr size_t NEAR_WATER_MAX_POINT_BATCH = 4'096;
constexpr size_t NEAR_WATER_MAX_EXACT_POINTS =
    static_cast<size_t>(FAR_TERRAIN_TILE_EDGE * FAR_TERRAIN_TILE_EDGE);

struct FarCell {
    std::array<float, 4> terrain{};
    std::array<float, 4> waterSurface{};
    std::array<FarWaterAuthority, 4> waterAuthority{};
    FarWaterAuthority centerWaterAuthority{};
    float skirtBottom = 0.0F;
    float maximumTerrain = 0.0F;
    BlockType material = BlockType::GRASS;
    bool flat = false;
    bool water = false;
    bool flatWater = false;
    bool discontinuousWater = false;
    bool centerWet = false;
    uint8_t waterMask = 0;
    float waterHeight = SEA_LEVEL;
    float centerWaterHeight = SEA_LEVEL;
};

struct Corner {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float u = 0.0F;
    float v = 0.0F;
};

struct WaterPoint {
    float x = 0.0F;
    float z = 0.0F;
    float height = 0.0F;
    bool wet = false;
    FarWaterAuthority authority{};
};

using WaterEdgeRefiner =
    std::function<WaterPoint(const WaterPoint& first, const WaterPoint& second,
                             const FarWaterAuthority& target, float connectionSpan)>;

bool validStep(FarTerrainStep step) {
    return step == FarTerrainStep::ONE || step == FarTerrainStep::TWO ||
           step == FarTerrainStep::FOUR || step == FarTerrainStep::EIGHT ||
           step == FarTerrainStep::SIXTEEN || step == FarTerrainStep::THIRTY_TWO;
}

int64_t tileOrigin(int64_t tileCoordinate) {
    const __int128 product =
        static_cast<__int128>(tileCoordinate) * static_cast<__int128>(FAR_TERRAIN_TILE_EDGE);
    if (product < std::numeric_limits<int64_t>::min() ||
        product > std::numeric_limits<int64_t>::max() - FAR_TERRAIN_TILE_EDGE) {
        throw std::out_of_range("far terrain tile origin exceeds int64 range");
    }
    return static_cast<int64_t>(product);
}

uint64_t mix64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xBF58'476D'1CE4'E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

uint16_t halfBits(const float16_t& value) {
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t hashMesh(const FarTerrainMesh& mesh) {
    uint64_t hash = mix64(static_cast<uint64_t>(mesh.key.tileX));
    hash = mix64(hash ^ static_cast<uint64_t>(mesh.key.tileZ));
    hash = mix64(hash ^ static_cast<uint8_t>(mesh.key.step));
    for (const Vertex& vertex : mesh.vertices) {
        hash = mix64(hash ^ vertex.faceAttr);
        hash = mix64(hash ^ (static_cast<uint64_t>(halfBits(vertex.px)) << 0U) ^
                     (static_cast<uint64_t>(halfBits(vertex.py)) << 16U) ^
                     (static_cast<uint64_t>(halfBits(vertex.pz)) << 32U));
        hash = mix64(hash ^ static_cast<uint64_t>(halfBits(vertex.u)) ^
                     (static_cast<uint64_t>(halfBits(vertex.v)) << 16U));
    }
    for (uint32_t index : mesh.indices)
        hash = mix64(hash ^ index);
    hash = mix64(hash ^ std::bit_cast<uint32_t>(mesh.complexity));
    return hash;
}

worldgen::surface_material::SurfaceMaterialPalette
terrainMaterialPalette(const worldgen::SurfaceSample& sample) {
    const Biome biome = sample.biome.primary;
    return worldgen::surface_material::materialPalette(
        sample, {}, worldgen::surface_material::frozen(sample, biome),
        worldgen::surface_material::submerged(sample));
}

FarTerrainGeometrySample geometryFromSurface(const worldgen::SurfaceSample& sample) {
    FarTerrainGeometrySample result;
    result.waterBodyId = sample.hydrology.waterBodyId;
    result.transitionOwnerId = sample.hydrology.transitionOwnerId;
    result.terrainHeight = sample.terrainHeight;
    // Exact emission and far water share one finished top-state authority.
    // This preserves flat source-filled standing bodies while matching the
    // eighth-block flowing profile of rivers and outlet corridors.
    result.waterSurface = worldgen::generatedFluidColumn(sample).visibleSurface;
    result.discharge = sample.hydrology.discharge;
    result.sediment = sample.hydrology.sediment;
    result.waterfallTop = sample.hydrology.waterfallTop;
    result.waterfallBottom = sample.hydrology.waterfallBottom;
    result.waterfallWidth = sample.hydrology.waterfallWidth;
    result.flowX = sample.hydrology.flowDirection.x;
    result.flowZ = sample.hydrology.flowDirection.z;
    result.transitionOwnerKind = sample.hydrology.transitionOwnerKind;
    result.generatedFluidLevel = sample.hydrology.generatedFluidLevel;
    result.ocean = sample.hydrology.ocean;
    result.river = sample.hydrology.river;
    result.lake = sample.hydrology.lake;
    result.waterfall = sample.hydrology.waterfall;
    result.waterfallAnchor = sample.hydrology.waterfallAnchor;
    result.delta = sample.hydrology.delta;
    return result;
}

FarTerrainGeometrySample geometryFromHydrology(const worldgen::HydrologySample& hydrology) {
    worldgen::SurfaceSample surface;
    surface.hydrology = hydrology;
    surface.terrainHeight = hydrology.surfaceElevation;
    surface.waterSurface = hydrology.waterSurface;
    return geometryFromSurface(surface);
}

FarSurfaceSample farSampleFromSurface(const worldgen::SurfaceSample& sample) {
    return {
        .geometry = geometryFromSurface(sample),
        .footprintMinimumTerrainHeight = sample.terrainHeight,
        .footprintMaximumTerrainHeight = sample.terrainHeight,
        .materialPalette = terrainMaterialPalette(sample),
    };
}

FarSurfaceSample boundedFarSampleFromSurface(const worldgen::SurfaceSample& surface,
                                             worldgen::SurfaceFootprint footprint,
                                             double emittedDetailAmplitude = 0.0) {
    FarSurfaceSample result = farSampleFromSurface(surface);
    const double reliefEnvelope =
        footprint == worldgen::SurfaceFootprint::BLOCK_32   ? FAR_TERRAIN_STEP32_RELIEF_ENVELOPE
        : footprint == worldgen::SurfaceFootprint::BLOCK_16 ? FAR_TERRAIN_STEP16_RELIEF_ENVELOPE
                                                            : 0.28;
    const double envelope =
        reliefEnvelope + emittedDetailAmplitude + FAR_TERRAIN_EMITTED_SURFACE_ENVELOPE;
    result.footprintMinimumTerrainHeight = surface.terrainHeight - envelope;
    result.footprintMaximumTerrainHeight = surface.terrainHeight + envelope;
    return result;
}

void includeCanonicalWaterFloor(FarSurfaceSample& coverage,
                                const worldgen::SurfaceSample& canonical) {
    const bool waterTopology =
        canonical.hydrology.ocean || canonical.hydrology.river || canonical.hydrology.lake;
    const bool standingWater =
        waterTopology && canonical.terrainHeight < canonical.hydrology.waterSurface - 0.01;
    if (!standingWater && !canonical.hydrology.waterfall) {
        return;
    }

    // The filtered relief envelope bounds terrain detail, but exact cube
    // emission can additionally lower a wet column to its analytical channel
    // floor or receiving-water cap. Carry that block-footprint authority into
    // the coarse parent instead of hiding the incision behind a larger global
    // relief constant.
    double minimum = canonical.hydrology.surfaceElevation;
    if (standingWater) {
        minimum = std::min(minimum, std::ceil(canonical.hydrology.waterSurface) - 1.0);
    }
    if (canonical.hydrology.waterfall &&
        canonical.hydrology.waterfallTop >= canonical.hydrology.waterfallBottom + 0.5) {
        minimum = std::min(minimum, std::ceil(canonical.hydrology.waterfallBottom) - 1.0);
    }
    coverage.footprintMinimumTerrainHeight =
        std::min(coverage.footprintMinimumTerrainHeight, minimum);
}

struct GeneratorCellBoundsBatch {
    const ChunkGenerator* generator = nullptr;
    uint32_t generatorSeed = 0;
    int64_t originX = 0;
    int64_t originZ = 0;
    int step = 0;
    int cellWidth = 0;
    int cellHeight = 0;
    worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1;
    std::vector<worldgen::SurfaceSample> vertices;
    std::vector<worldgen::SurfaceSample> canonicalVertices;
    std::vector<worldgen::SurfaceSample> centers;

    bool matches(const ChunkGenerator* expectedGenerator, uint32_t expectedSeed,
                 int64_t expectedOriginX, int64_t expectedOriginZ, int expectedStep,
                 int expectedCellWidth, int expectedCellHeight,
                 worldgen::SurfaceFootprint expectedFootprint) const {
        return generator == expectedGenerator && originX == expectedOriginX &&
               generatorSeed == expectedSeed && originZ == expectedOriginZ &&
               step == expectedStep && cellWidth == expectedCellWidth &&
               cellHeight == expectedCellHeight && footprint == expectedFootprint;
    }

    const worldgen::SurfaceSample& vertex(int cellX, int cellZ) const {
        return vertices[static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth + 1) +
                        static_cast<size_t>(cellX)];
    }

    const worldgen::SurfaceSample& canonicalVertex(int cellX, int cellZ) const {
        return canonicalVertices[static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth + 1) +
                                 static_cast<size_t>(cellX)];
    }

    const worldgen::SurfaceSample* center(int cellX, int cellZ) const {
        if (centers.empty()) {
            return nullptr;
        }
        return &centers[static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth) +
                        static_cast<size_t>(cellX)];
    }
};

GeneratorCellBoundsBatch& threadGeneratorCellBoundsBatch() {
    thread_local GeneratorCellBoundsBatch batch;
    return batch;
}

int64_t checkedCoordinateOffset(int64_t coordinate, int64_t offset) {
    const __int128 result = static_cast<__int128>(coordinate) + offset;
    if (result < std::numeric_limits<int64_t>::min() ||
        result > std::numeric_limits<int64_t>::max()) {
        throw std::out_of_range("far terrain bounds coordinate exceeds int64 range");
    }
    return static_cast<int64_t>(result);
}

double footprintReliefSlopeEnvelope(worldgen::SurfaceFootprint footprint) {
    if (footprint == worldgen::SurfaceFootprint::BLOCK_32) {
        return FAR_TERRAIN_STEP32_RELIEF_SLOPE_ENVELOPE;
    }
    if (footprint == worldgen::SurfaceFootprint::BLOCK_16) {
        return FAR_TERRAIN_STEP16_RELIEF_SLOPE_ENVELOPE;
    }
    return 0.18;
}

double footprintInteriorEnvelope(worldgen::SurfaceFootprint footprint) {
    switch (footprint) {
        case worldgen::SurfaceFootprint::BLOCK_1:
            return 0.0;
        case worldgen::SurfaceFootprint::BLOCK_2:
            return 0.45;
        case worldgen::SurfaceFootprint::BLOCK_4:
            return 6.5;
        case worldgen::SurfaceFootprint::BLOCK_8:
            return 11.75;
        case worldgen::SurfaceFootprint::BLOCK_16:
            return 17.0;
        case worldgen::SurfaceFootprint::BLOCK_32:
            return 21.0;
    }
    return 21.0;
}

void prepareGeneratorCellBoundsBatch(GeneratorCellBoundsBatch& batch,
                                     const std::shared_ptr<ChunkGenerator>& generator,
                                     int64_t originX, int64_t originZ, int step, int cellWidth,
                                     int cellHeight, worldgen::SurfaceFootprint footprint,
                                     bool geometryOnly) {
    if (step <= 0 || cellWidth <= 0 || cellHeight <= 0 ||
        worldgen::surfaceFootprintWidth(footprint) != step) {
        throw std::invalid_argument("invalid generator far terrain cell bounds grid");
    }
    batch.generator = generator.get();
    batch.generatorSeed = generator->seed();
    batch.originX = originX;
    batch.originZ = originZ;
    batch.step = step;
    batch.cellWidth = cellWidth;
    batch.cellHeight = cellHeight;
    batch.footprint = footprint;
    batch.vertices.resize(static_cast<size_t>(cellWidth + 1) * static_cast<size_t>(cellHeight + 1));
    if (geometryOnly || cellWidth != cellHeight) {
        generator->sampleFarGeometryGrid(originX, originZ, step, step, cellWidth + 1,
                                         cellHeight + 1, footprint, batch.vertices);
    } else {
        generator->sampleFarSurfaceGrid(originX, originZ, step, cellWidth + 1, footprint,
                                        batch.vertices);
    }
    if (step == 32) {
        // The footprint preserves hydrology topology and levels. Reuse its
        // already volcanic surface for the emergency parent; the analytical
        // channel, lake, ocean, and relief envelopes below conservatively
        // cover omitted block-scale floor detail.
        batch.canonicalVertices = batch.vertices;
    } else {
        batch.canonicalVertices.resize(batch.vertices.size());
        generator->sampleFarGeometryGrid(originX, originZ, step, step, cellWidth + 1,
                                         cellHeight + 1, worldgen::SurfaceFootprint::BLOCK_1,
                                         batch.canonicalVertices);
    }
    if (step >= 4 && step <= 32 && (step & 1) == 0) {
        batch.centers.resize(static_cast<size_t>(cellWidth) * static_cast<size_t>(cellHeight));
        generator->sampleFarGeometryGrid(checkedCoordinateOffset(originX, step / 2),
                                         checkedCoordinateOffset(originZ, step / 2), step, step,
                                         cellWidth, cellHeight, footprint, batch.centers);
    } else {
        batch.centers.clear();
    }
}

struct GeneratorSurfaceBounds {
    double terrainHeight = std::numeric_limits<double>::quiet_NaN();
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    bool waterTopologyPossible = false;
    bool volcanicWaterPossible = false;
};

void includeGeneratorBoundsSample(GeneratorSurfaceBounds& bounds,
                                  const worldgen::SurfaceSample& filtered,
                                  const worldgen::SurfaceSample& canonical,
                                  worldgen::SurfaceFootprint footprint) {
    FarSurfaceSample bounded =
        boundedFarSampleFromSurface(filtered, footprint,
                                    ChunkGenerator::emittedSurfaceDetailAmplitude(
                                        filtered, footprintReliefSlopeEnvelope(footprint)));
    includeCanonicalWaterFloor(bounded, canonical);
    bounds.minimum = std::min(bounds.minimum, bounded.footprintMinimumTerrainHeight);
    bounds.maximum = std::max(bounds.maximum, bounded.footprintMaximumTerrainHeight);
    const worldgen::HydrologySample& hydrology = canonical.hydrology;
    const bool volcanicWaterPossible =
        canonical.geology.hotspotInfluence > 1.0e-6 || canonical.geology.volcanicActivity > 1.0e-6;
    bounds.volcanicWaterPossible = bounds.volcanicWaterPossible || volcanicWaterPossible;
    bounds.waterTopologyPossible = bounds.waterTopologyPossible || hydrology.ocean ||
                                   hydrology.river || hydrology.lake || hydrology.waterfall ||
                                   volcanicWaterPossible;
}

void includeInteriorHydrologyBound(GeneratorSurfaceBounds& bounds,
                                   const worldgen::SurfaceSample& sample, int step) {
    const worldgen::HydrologySample& hydrology = sample.hydrology;
    const double reach = static_cast<double>(step) * std::numbers::sqrt2 * 0.5;
    if (hydrology.streamOrder > 0 && hydrology.channelWidth > 0.0 &&
        hydrology.channelDistance <= hydrology.channelWidth * 0.55 + reach) {
        bounds.waterTopologyPossible = true;
        const double channelDepth = std::max({hydrology.channelDepth, hydrology.erosionDepth, 1.0});
        bounds.minimum =
            std::min(bounds.minimum,
                     std::min(hydrology.surfaceElevation, hydrology.waterSurface) - channelDepth);
    }
    const bool knownLake = hydrology.lake || hydrology.waterBodyId != worldgen::NO_WATER_BODY ||
                           hydrology.lakeAreaSquareKilometers > 0.0;
    if (knownLake && hydrology.lakeShoreDistance >= -reach &&
        std::isfinite(hydrology.waterSurface)) {
        bounds.waterTopologyPossible = true;
        const double areaSquareMeters = hydrology.lakeAreaSquareKilometers * 1'000'000.0;
        const double meanDepth =
            areaSquareMeters > 1.0 ? hydrology.lakeVolumeCubicMeters / areaSquareMeters : 0.0;
        const double lakeDepth =
            std::clamp(std::max(hydrology.lakeDepth, meanDepth * 3.0 + 2.0), 2.0, 48.0);
        bounds.minimum = std::min(bounds.minimum, hydrology.waterSurface - lakeDepth);
    }
    if (hydrology.waterfall && hydrology.waterfallTop >= hydrology.waterfallBottom + 0.5) {
        bounds.waterTopologyPossible = true;
        bounds.minimum = std::min(bounds.minimum, std::ceil(hydrology.waterfallBottom) - 1.0);
        bounds.maximum = std::max(bounds.maximum, std::ceil(hydrology.waterfallTop));
    }
}

GeneratorSurfaceBounds generatorCellSurfaceBounds(const GeneratorCellBoundsBatch& batch, int cellX,
                                                  int cellZ, bool includeCenter = true) {
    GeneratorSurfaceBounds bounds;
    if (batch.step == 1) {
        bounds.terrainHeight = batch.vertex(cellX, cellZ).terrainHeight;
        includeGeneratorBoundsSample(bounds, batch.vertex(cellX, cellZ),
                                     batch.canonicalVertex(cellX, cellZ), batch.footprint);
        return bounds;
    }
    constexpr std::array<std::array<int, 2>, 4> CORNERS = {
        {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}}};
    for (const auto& corner : CORNERS) {
        const worldgen::SurfaceSample& filtered =
            batch.vertex(cellX + corner[0], cellZ + corner[1]);
        const worldgen::SurfaceSample& canonical =
            batch.canonicalVertex(cellX + corner[0], cellZ + corner[1]);
        includeGeneratorBoundsSample(bounds, filtered, canonical, batch.footprint);
        includeInteriorHydrologyBound(bounds, canonical, batch.step);
    }
    if (includeCenter) {
        const worldgen::SurfaceSample* center = batch.center(cellX, cellZ);
        if (center != nullptr) {
            bounds.terrainHeight = center->terrainHeight;
            includeGeneratorBoundsSample(bounds, *center, *center, batch.footprint);
            includeInteriorHydrologyBound(bounds, *center, batch.step);
        }
    }
    if (!std::isfinite(bounds.terrainHeight)) {
        double terrainHeight = 0.0;
        for (const auto& corner : CORNERS) {
            terrainHeight += batch.vertex(cellX + corner[0], cellZ + corner[1]).terrainHeight;
        }
        bounds.terrainHeight = terrainHeight / static_cast<double>(CORNERS.size());
    }
    bounds.waterTopologyPossible = bounds.waterTopologyPossible || bounds.minimum < SEA_LEVEL;
    const double interiorEnvelope = footprintInteriorEnvelope(batch.footprint);
    bounds.minimum -= interiorEnvelope;
    bounds.maximum += interiorEnvelope;
    return bounds;
}

bool hasWater(const FarTerrainGeometrySample& sample) {
    return sample.ocean || sample.river || sample.lake;
}

double materialRank(int64_t worldX, int64_t worldZ) {
    static const CounterRng rankRandom{0x4C4F'445F'4D41'5445ULL};
    return worldgen::multiscaleDitherThreshold(rankRandom, 0x4641'525F'4D41'5445ULL, worldX,
                                               worldZ);
}

double footprintTerrainMinimum(const FarSurfaceSample& sample) {
    return std::isfinite(sample.footprintMinimumTerrainHeight)
               ? std::min(sample.footprintMinimumTerrainHeight, sample.geometry.terrainHeight)
               : sample.geometry.terrainHeight;
}

double footprintTerrainMaximum(const FarSurfaceSample& sample) {
    return std::isfinite(sample.footprintMaximumTerrainHeight)
               ? std::max(sample.footprintMaximumTerrainHeight, sample.geometry.terrainHeight)
               : sample.geometry.terrainHeight;
}

float vertexHeight(double height) {
    return static_cast<float>(static_cast<float16_t>(std::clamp(
        height, static_cast<double>(WORLD_MIN_Y), static_cast<double>(WORLD_MAX_Y + 1))));
}

float terrainCellHeight(const std::array<FarSurfaceSample*, 4>& corners, FarTerrainStep step) {
    if (step == FarTerrainStep::ONE) {
        // One far cell represents the exact block column at its lower-left
        // world coordinate. Averaging its four neighbors rounded shorelines
        // and cliffs into a different surface immediately outside exact
        // residency, despite both tiers having one-block horizontal detail.
        return vertexHeight(std::floor(corners.front()->geometry.terrainHeight + 0.5));
    }

    // Reduced-resolution cells display the filtered field itself. Bounds
    // remain conservative for culling and skirts, but they cannot depress the
    // visible top. The four-corner mean is the center-equivalent available to
    // callback-only sources that do not provide authoritative cell samples.
    double height = 0.0;
    for (const FarSurfaceSample* corner : corners)
        height += corner->geometry.terrainHeight;
    return vertexHeight(std::ceil(height / static_cast<double>(corners.size())));
}

struct ResolvedFarTerrainCellBounds {
    float top = 0.0F;
    float minimum = 0.0F;
    float maximum = 0.0F;
    float skirtBottom = 0.0F;
};

float downwardBound(double height) {
    return vertexHeight(std::floor(height));
}

float upwardBound(double height) {
    return vertexHeight(std::ceil(height));
}

ResolvedFarTerrainCellBounds fallbackCellBounds(const std::array<FarSurfaceSample*, 4>& corners,
                                                FarTerrainStep step) {
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    for (const FarSurfaceSample* corner : corners) {
        minimum = std::min(minimum, footprintTerrainMinimum(*corner));
        maximum = std::max(maximum, footprintTerrainMaximum(*corner));
    }
    const float top = terrainCellHeight(corners, step);
    return {
        .top = top,
        .minimum = std::min(top, downwardBound(minimum)),
        .maximum = std::max(top, upwardBound(maximum)),
        .skirtBottom = std::max(static_cast<float>(WORLD_MIN_Y), top - FAR_TERRAIN_SKIRT_DEPTH),
    };
}

ResolvedFarTerrainCellBounds authoritativeCellBounds(const FarTerrainCellBounds& bounds) {
    if (!std::isfinite(bounds.terrainHeight) || !std::isfinite(bounds.minimumTerrainHeight) ||
        !std::isfinite(bounds.maximumTerrainHeight) || !std::isfinite(bounds.skirtBottom) ||
        bounds.maximumTerrainHeight < bounds.minimumTerrainHeight ||
        bounds.skirtBottom > bounds.minimumTerrainHeight ||
        bounds.terrainHeight < bounds.minimumTerrainHeight ||
        bounds.terrainHeight > bounds.maximumTerrainHeight) {
        throw std::invalid_argument("invalid far terrain cell bounds");
    }
    const float minimum = downwardBound(bounds.minimumTerrainHeight);
    const float maximum = upwardBound(bounds.maximumTerrainHeight);
    const float top = upwardBound(bounds.terrainHeight);
    return {
        .top = top,
        .minimum = std::min(top, minimum),
        .maximum = std::max(top, maximum),
        .skirtBottom = std::min(top, downwardBound(bounds.skirtBottom)),
    };
}

FarWaterKind farWaterKind(const FarTerrainGeometrySample& sample) {
    if (sample.lake) return FarWaterKind::LAKE;
    if (sample.ocean) return FarWaterKind::OCEAN;
    if (sample.river) return FarWaterKind::RIVER;
    return FarWaterKind::NONE;
}

FarWaterAuthority waterAuthority(const FarTerrainGeometrySample& sample, bool wet) {
    const FarWaterKind kind = wet ? farWaterKind(sample) : FarWaterKind::NONE;
    return {
        .bodyId = sample.waterBodyId,
        .transitionOwnerId = sample.transitionOwnerId,
        .height = vertexHeight(sample.waterSurface),
        .kind = kind,
        .transitionOwnerKind = sample.transitionOwnerKind,
        .generatedFluidLevel = sample.generatedFluidLevel,
        .delta = sample.delta,
    };
}

bool waterAuthoritiesConnect(const FarWaterAuthority& first, const FarWaterAuthority& second,
                             float sampleDistance) {
    if (first.kind == FarWaterKind::NONE || second.kind == FarWaterKind::NONE) return false;
    const float heightDifference = std::abs(first.height - second.height);
    if (first.kind == FarWaterKind::LAKE && second.kind == FarWaterKind::LAKE) {
        return first.bodyId != worldgen::NO_WATER_BODY && first.bodyId == second.bodyId &&
               heightDifference <= 0.125F;
    }
    if (first.kind == FarWaterKind::OCEAN && second.kind == FarWaterKind::OCEAN) {
        return heightDifference <= 0.125F;
    }
    if (first.kind == FarWaterKind::RIVER && second.kind == FarWaterKind::RIVER) {
        // A routed channel may descend gradually between samples. Larger
        // drops remain separate top surfaces and are represented by the
        // explicitly owned waterfall prism instead of a sloped water wall.
        return heightDifference <= std::max(0.25F, sampleDistance * 0.20F);
    }
    const bool riverMouth =
        (first.kind == FarWaterKind::RIVER && second.kind == FarWaterKind::OCEAN) ||
        (first.kind == FarWaterKind::OCEAN && second.kind == FarWaterKind::RIVER);
    const bool lakeOutlet =
        (first.kind == FarWaterKind::RIVER && second.kind == FarWaterKind::LAKE) ||
        (first.kind == FarWaterKind::LAKE && second.kind == FarWaterKind::RIVER);
    if (riverMouth || lakeOutlet) {
        // A mouth or nonfalling outlet shares a surface only where its exact
        // authority is already level. Deltas retain the same tolerance; they
        // do not authorize a bridge to an unrelated standing body.
        return heightDifference <= 0.25F;
    }
    return false;
}

bool sharesFlowTransition(const FarWaterAuthority& first, const FarWaterAuthority& second,
                          float sampleDistance) {
    return first.kind == FarWaterKind::RIVER && second.kind == FarWaterKind::RIVER &&
           first.generatedFluidLevel > 0 && second.generatedFluidLevel > 0 &&
           first.transitionOwnerKind != worldgen::WaterTransitionKind::NONE &&
           first.transitionOwnerKind == second.transitionOwnerKind &&
           first.transitionOwnerId != 0 && first.transitionOwnerId == second.transitionOwnerId &&
           waterAuthoritiesConnect(first, second, sampleDistance);
}

void updateYBounds(FarTerrainMesh& mesh, float y) {
    mesh.bounds.minY = std::min(mesh.bounds.minY, y);
    mesh.bounds.maxY = std::max(mesh.bounds.maxY, y);
}

void updateSurfaceYBounds(FarTerrainMesh& mesh, float y) {
    mesh.surfaceBounds.minY = std::min(mesh.surfaceBounds.minY, y);
    mesh.surfaceBounds.maxY = std::max(mesh.surfaceBounds.maxY, y);
}

void pushQuad(FarTerrainMesh& mesh, uint32_t attribute, const std::array<Corner, 4>& corners,
              bool water) {
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (const Corner& corner : corners) {
        const Vertex vertex{attribute,
                            static_cast<float16_t>(corner.x),
                            static_cast<float16_t>(corner.y),
                            static_cast<float16_t>(corner.z),
                            static_cast<float16_t>(corner.u),
                            static_cast<float16_t>(corner.v)};
        mesh.vertices.push_back(vertex);
        updateYBounds(mesh, static_cast<float>(vertex.py));
    }
    mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
    if (water) {
        ++mesh.waterQuadCount;
    } else {
        ++mesh.terrainQuadCount;
    }
}

void pushTerrainTop(FarTerrainMesh& mesh, BlockType material, float x0, float z0, float x1,
                    float z1, float northwest, float northeast, float southeast, float southwest) {
    const uint32_t attribute =
        packFaceAttr(FaceNormal::PLUS_Y, textureLayerFor(material, FaceNormal::PLUS_Y), 15);
    const float width = x1 - x0;
    const float depth = z1 - z0;
    pushQuad(mesh, attribute,
             {{{x0, northwest, z0, 0.0F, 0.0F},
               {x0, southwest, z1, 0.0F, depth},
               {x1, southeast, z1, width, depth},
               {x1, northeast, z0, width, 0.0F}}},
             false);
    updateSurfaceYBounds(mesh, northwest);
    updateSurfaceYBounds(mesh, northeast);
    updateSurfaceYBounds(mesh, southeast);
    updateSurfaceYBounds(mesh, southwest);
}

void pushTerrainRiser(FarTerrainMesh& mesh, FaceNormal face, BlockType material, float x0, float z0,
                      float x1, float z1, float bottom, float top) {
    if (top <= bottom) return;
    const float width = std::hypot(x1 - x0, z1 - z0);
    const float height = top - bottom;
    const uint32_t attribute = packFaceAttr(face, textureLayerFor(material, face), 15);
    pushQuad(mesh, attribute,
             {{{x0, bottom, z0, 0.0F, height},
               {x1, bottom, z1, width, height},
               {x1, top, z1, width, 0.0F},
               {x0, top, z0, 0.0F, 0.0F}}},
             false);
    updateSurfaceYBounds(mesh, bottom);
    updateSurfaceYBounds(mesh, top);
}

void pushWaterTop(FarTerrainMesh& mesh, float x0, float z0, float x1, float z1,
                  const std::array<float, 4>& heights) {
    const uint32_t attribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, false);
    const float width = x1 - x0;
    const float depth = z1 - z0;
    pushQuad(mesh, attribute,
             {{{x0, heights[0], z0, 0.0F, 0.0F},
               {x0, heights[3], z1, 0.0F, depth},
               {x1, heights[2], z1, width, depth},
               {x1, heights[1], z0, width, 0.0F}}},
             true);
    for (float height : heights)
        updateSurfaceYBounds(mesh, height);
}

void pushWaterTop(FarTerrainMesh& mesh, float x0, float z0, float x1, float z1, float height) {
    pushWaterTop(mesh, x0, z0, x1, z1, {height, height, height, height});
}

void pushWaterRiser(FarTerrainMesh& mesh, FaceNormal face, float x0, float z0, float x1, float z1,
                    float bottom, float top) {
    if (top <= bottom + 1.0e-4F) return;
    const float width = std::hypot(x1 - x0, z1 - z0);
    const float height = top - bottom;
    const uint32_t attribute = packFluidFaceAttr(face, 15, 0, false);
    std::array<Corner, 4> corners{};
    switch (face) {
        case FaceNormal::MINUS_X:
            corners = {{{x0, bottom, z0, 0.0F, height},
                        {x0, bottom, z1, width, height},
                        {x0, top, z1, width, 0.0F},
                        {x0, top, z0, 0.0F, 0.0F}}};
            break;
        case FaceNormal::PLUS_X:
            corners = {{{x0, bottom, z1, 0.0F, height},
                        {x0, bottom, z0, width, height},
                        {x0, top, z0, width, 0.0F},
                        {x0, top, z1, 0.0F, 0.0F}}};
            break;
        case FaceNormal::MINUS_Z:
            corners = {{{x1, bottom, z0, 0.0F, height},
                        {x0, bottom, z0, width, height},
                        {x0, top, z0, width, 0.0F},
                        {x1, top, z0, 0.0F, 0.0F}}};
            break;
        case FaceNormal::PLUS_Z:
            corners = {{{x0, bottom, z0, 0.0F, height},
                        {x1, bottom, z0, width, height},
                        {x1, top, z0, width, 0.0F},
                        {x0, top, z0, 0.0F, 0.0F}}};
            break;
        default:
            throw std::invalid_argument("far water riser must be horizontal");
    }
    pushQuad(mesh, attribute, corners, true);
    updateSurfaceYBounds(mesh, bottom);
    updateSurfaceYBounds(mesh, top);
}

FaceNormal horizontalFace(float normalX, float normalZ) {
    if (std::abs(normalX) >= std::abs(normalZ)) {
        return normalX >= 0.0F ? FaceNormal::PLUS_X : FaceNormal::MINUS_X;
    }
    return normalZ >= 0.0F ? FaceNormal::PLUS_Z : FaceNormal::MINUS_Z;
}

void pushWaterfallPrism(FarTerrainMesh& mesh, float centerX, float centerZ,
                        const FarTerrainGeometrySample& sample) {
    const float bottom = vertexHeight(std::ceil(sample.waterfallBottom) - 1.0);
    const float top = vertexHeight(std::ceil(sample.waterfallTop));
    if (top <= bottom + 0.5F) return;

    float flowX = static_cast<float>(sample.flowX);
    float flowZ = static_cast<float>(sample.flowZ);
    const float flowLength = std::hypot(flowX, flowZ);
    if (flowLength <= 1.0e-5F) {
        flowX = 1.0F;
        flowZ = 0.0F;
    } else {
        flowX /= flowLength;
        flowZ /= flowLength;
    }
    const float crossX = -flowZ;
    const float crossZ = flowX;
    const float halfWidth =
        std::clamp(static_cast<float>(sample.waterfallWidth) * 0.5F, 1.5F, 8.0F);
    const float halfDepth = std::clamp(halfWidth * 0.18F, 0.75F, 1.5F);
    const std::array<std::array<float, 2>, 4> footprint = {{
        {{centerX - crossX * halfWidth - flowX * halfDepth,
          centerZ - crossZ * halfWidth - flowZ * halfDepth}},
        {{centerX + crossX * halfWidth - flowX * halfDepth,
          centerZ + crossZ * halfWidth - flowZ * halfDepth}},
        {{centerX + crossX * halfWidth + flowX * halfDepth,
          centerZ + crossZ * halfWidth + flowZ * halfDepth}},
        {{centerX - crossX * halfWidth + flowX * halfDepth,
          centerZ - crossZ * halfWidth + flowZ * halfDepth}},
    }};
    const float height = top - bottom;
    auto pushSide = [&](size_t firstIndex, size_t secondIndex, float normalX, float normalZ) {
        const auto& first = footprint[firstIndex];
        const auto& second = footprint[secondIndex];
        const float width = std::hypot(second[0] - first[0], second[1] - first[1]);
        const uint32_t attribute = packFluidFaceAttr(horizontalFace(normalX, normalZ), 15, 0, true);
        pushQuad(mesh, attribute,
                 {{{first[0], bottom, first[1], 0.0F, height},
                   {second[0], bottom, second[1], width, height},
                   {second[0], top, second[1], width, 0.0F},
                   {first[0], top, first[1], 0.0F, 0.0F}}},
                 true);
        ++mesh.waterfallQuadCount;
    };
    pushSide(0, 1, -flowX, -flowZ);
    pushSide(1, 2, crossX, crossZ);
    pushSide(2, 3, flowX, flowZ);
    pushSide(3, 0, -crossX, -crossZ);

    const uint32_t topAttribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, true);
    pushQuad(mesh, topAttribute,
             {{{footprint[0][0], top, footprint[0][1], 0.0F, 0.0F},
               {footprint[3][0], top, footprint[3][1], 0.0F, halfDepth * 2.0F},
               {footprint[2][0], top, footprint[2][1], halfWidth * 2.0F, halfDepth * 2.0F},
               {footprint[1][0], top, footprint[1][1], halfWidth * 2.0F, 0.0F}}},
             true);
    ++mesh.waterfallQuadCount;
    updateSurfaceYBounds(mesh, bottom);
    updateSurfaceYBounds(mesh, top);

    for (const auto& point : footprint) {
        const int64_t worldX = mesh.originX + static_cast<int64_t>(std::floor(point[0]));
        const int64_t worldZ = mesh.originZ + static_cast<int64_t>(std::floor(point[1]));
        const int64_t worldMaxX = mesh.originX + static_cast<int64_t>(std::ceil(point[0]));
        const int64_t worldMaxZ = mesh.originZ + static_cast<int64_t>(std::ceil(point[1]));
        mesh.bounds.minX = std::min(mesh.bounds.minX, worldX);
        mesh.bounds.maxX = std::max(mesh.bounds.maxX, worldMaxX);
        mesh.bounds.minZ = std::min(mesh.bounds.minZ, worldZ);
        mesh.bounds.maxZ = std::max(mesh.bounds.maxZ, worldMaxZ);
        mesh.surfaceBounds.minX = std::min(mesh.surfaceBounds.minX, worldX);
        mesh.surfaceBounds.maxX = std::max(mesh.surfaceBounds.maxX, worldMaxX);
        mesh.surfaceBounds.minZ = std::min(mesh.surfaceBounds.minZ, worldZ);
        mesh.surfaceBounds.maxZ = std::max(mesh.surfaceBounds.maxZ, worldMaxZ);
    }
}

// Clip one center-split cell triangle against the binary wet mask. The caller
// refines wet-to-dry edges against the coordinate-pure water predicate, which
// keeps a coarse LOD from extending a lake surface halfway across a dry cell.
void pushWaterContourTriangle(FarTerrainMesh& mesh, const std::array<WaterPoint, 3>& triangle,
                              const FarWaterAuthority& target, float connectionSpan,
                              const WaterEdgeRefiner& refineEdge) {
    std::array<WaterPoint, 4> polygon{};
    size_t polygonSize = 0;
    for (size_t index = 0; index < triangle.size(); ++index) {
        WaterPoint current = triangle[index];
        WaterPoint next = triangle[(index + 1) % triangle.size()];
        current.wet = waterAuthoritiesConnect(current.authority, target, connectionSpan);
        next.wet = waterAuthoritiesConnect(next.authority, target, connectionSpan);
        if (current.wet) polygon[polygonSize++] = current;
        if (current.wet != next.wet) {
            polygon[polygonSize++] = refineEdge(current, next, target, connectionSpan);
        }
    }
    if (polygonSize < 3) return;

    const uint32_t attribute = packFluidFaceAttr(FaceNormal::PLUS_Y, 15, 0, false);
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (size_t index = 0; index < polygonSize; ++index) {
        WaterPoint point = polygon[index];
        // Standing and routed water remains a voxel surface at every far tier.
        // Authority clipping chooses which finished level owns this polygon;
        // interpolating the participating samples would reintroduce the
        // walkable diagonal sheets that the voxel LOD contract forbids.
        point.height = target.height;
        const Vertex vertex{attribute,
                            static_cast<float16_t>(point.x),
                            static_cast<float16_t>(point.height),
                            static_cast<float16_t>(point.z),
                            static_cast<float16_t>(point.x),
                            static_cast<float16_t>(point.z)};
        mesh.vertices.push_back(vertex);
        updateYBounds(mesh, static_cast<float>(vertex.py));
        updateSurfaceYBounds(mesh, static_cast<float>(vertex.py));
    }
    for (size_t index = 1; index + 1 < polygonSize; ++index) {
        mesh.indices.insert(mesh.indices.end(), {base, base + static_cast<uint32_t>(index),
                                                 base + static_cast<uint32_t>(index + 1)});
        ++mesh.waterContourTriangleCount;
    }
}

void pushSkirt(FarTerrainMesh& mesh, FaceNormal face, BlockType material, float x0, float z0,
               float x1, float z1, float top0, float top1, float bottom0, float bottom1) {
    bottom0 = std::clamp(bottom0, static_cast<float>(WORLD_MIN_Y), top0);
    bottom1 = std::clamp(bottom1, static_cast<float>(WORLD_MIN_Y), top1);
    const float width = std::hypot(x1 - x0, z1 - z0);
    const uint32_t attribute =
        packFaceAttr(face, textureLayerFor(material, face), 15) | FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK;
    const std::array<Corner, 4> corners = {{{x0, bottom0, z0, 0.0F, top0 - bottom0},
                                            {x1, bottom1, z1, width, top1 - bottom1},
                                            {x1, top1, z1, width, 0.0F},
                                            {x0, top0, z0, 0.0F, 0.0F}}};
    pushQuad(mesh, attribute, corners, false);
    --mesh.terrainQuadCount;
    ++mesh.skirtQuadCount;
}

void pushCanopyQuad(FarTerrainMesh& mesh, FaceNormal face, BlockType material,
                    const std::array<Corner, 4>& corners) {
    const uint32_t attribute =
        packFaceAttr(face, textureLayerFor(material, face), 15) | FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK;
    pushQuad(mesh, attribute, corners, false);
    --mesh.terrainQuadCount;
    ++mesh.canopyImpostorQuadCount;
    for (const Corner& corner : corners) {
        updateSurfaceYBounds(mesh, corner.y);
        const int64_t worldX = mesh.originX + static_cast<int64_t>(std::floor(corner.x));
        const int64_t worldZ = mesh.originZ + static_cast<int64_t>(std::floor(corner.z));
        const int64_t worldMaxX = mesh.originX + static_cast<int64_t>(std::ceil(corner.x));
        const int64_t worldMaxZ = mesh.originZ + static_cast<int64_t>(std::ceil(corner.z));
        mesh.surfaceBounds.minX = std::min(mesh.surfaceBounds.minX, worldX);
        mesh.surfaceBounds.maxX = std::max(mesh.surfaceBounds.maxX, worldMaxX);
        mesh.surfaceBounds.minZ = std::min(mesh.surfaceBounds.minZ, worldZ);
        mesh.surfaceBounds.maxZ = std::max(mesh.surfaceBounds.maxZ, worldMaxZ);
    }
}

void pushCanopyPrism(FarTerrainMesh& mesh, BlockType material, float centerX, float centerZ,
                     float halfWidthX, float halfWidthZ, float bottom, float top, bool includeTop) {
    const float x0 = centerX - halfWidthX;
    const float x1 = centerX + halfWidthX;
    const float z0 = centerZ - halfWidthZ;
    const float z1 = centerZ + halfWidthZ;
    const float width = halfWidthX * 2.0F;
    const float depth = halfWidthZ * 2.0F;
    const float height = std::max(0.0F, top - bottom);
    if (height <= 0.0F || width <= 0.0F || depth <= 0.0F) return;
    pushCanopyQuad(mesh, FaceNormal::MINUS_X, material,
                   {{{x0, bottom, z0, 0.0F, height},
                     {x0, bottom, z1, depth, height},
                     {x0, top, z1, depth, 0.0F},
                     {x0, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::PLUS_X, material,
                   {{{x1, bottom, z1, 0.0F, height},
                     {x1, bottom, z0, depth, height},
                     {x1, top, z0, depth, 0.0F},
                     {x1, top, z1, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::MINUS_Z, material,
                   {{{x1, bottom, z0, 0.0F, height},
                     {x0, bottom, z0, width, height},
                     {x0, top, z0, width, 0.0F},
                     {x1, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(mesh, FaceNormal::PLUS_Z, material,
                   {{{x0, bottom, z1, 0.0F, height},
                     {x1, bottom, z1, width, height},
                     {x1, top, z1, width, 0.0F},
                     {x0, top, z1, 0.0F, 0.0F}}});
    if (includeTop) {
        pushCanopyQuad(mesh, FaceNormal::PLUS_Y, material,
                       {{{x0, top, z0, 0.0F, 0.0F},
                         {x0, top, z1, 0.0F, depth},
                         {x1, top, z1, width, depth},
                         {x1, top, z0, width, 0.0F}}});
    }
}

float steppedCanopyRadius(float radius, float scale) {
    return std::max(0.5F, std::floor(radius * scale * 2.0F + 0.5F) * 0.5F);
}

template <size_t LayerCount>
void pushCanopyStack(FarTerrainMesh& mesh, BlockType material, float centerX, float centerZ,
                     float radius, float bottom, float top,
                     const std::array<float, LayerCount>& radiusScales) {
    const float height = top - bottom;
    if (height <= 0.0F) return;
    for (size_t layer = 0; layer < LayerCount; ++layer) {
        const float layerBottom =
            bottom + std::floor(height * static_cast<float>(layer) / LayerCount);
        const float layerTop =
            layer + 1 == LayerCount
                ? top
                : bottom + std::floor(height * static_cast<float>(layer + 1) / LayerCount);
        const float layerRadius = steppedCanopyRadius(radius, radiusScales[layer]);
        pushCanopyPrism(mesh, material, centerX, centerZ, layerRadius, layerRadius, layerBottom,
                        layerTop, true);
    }
}

void pushSpeciesCanopy(FarTerrainMesh& mesh, const FarCanopy& canopy, float centerX, float centerZ,
                       float radius, float bottom, float top) {
    using feature_generation::TreeSpecies;
    const TreeSpecies species = canopy.species;
    const BlockType leafBlock = canopy.leafBlock;
    switch (species) {
        case TreeSpecies::SPRUCE:
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{1.0F, 0.80F, 0.58F, 0.32F});
            return;
        case TreeSpecies::ACACIA: {
            const float height = std::max(1.0F, top - bottom);
            const float directionX = canopy.formX == 0 ? 1.0F : static_cast<float>(canopy.formX);
            const float directionZ = canopy.formZ == 0 ? 1.0F : static_cast<float>(canopy.formZ);
            const float primaryShift = radius * 0.24F;
            const float secondaryShift = radius * 0.34F;
            const float split = bottom + std::max(1.0F, std::floor(height * 0.50F));
            const float primaryRadius = steppedCanopyRadius(radius, 0.68F);
            pushCanopyPrism(mesh, leafBlock, centerX + directionX * primaryShift,
                            centerZ + directionZ * primaryShift, primaryRadius, primaryRadius,
                            bottom, std::min(split, top), true);
            const float secondaryRadius = steppedCanopyRadius(radius, 0.54F);
            pushCanopyPrism(mesh, leafBlock, centerX - directionZ * secondaryShift,
                            centerZ + directionX * secondaryShift, secondaryRadius, secondaryRadius,
                            std::min(split, top), top, true);
            return;
        }
        case TreeSpecies::PALM: {
            const float height = top - bottom;
            if (height <= 0.0F) return;
            const float firstTop = bottom + std::max(1.0F, std::floor(height / 3.0F));
            const float secondTop =
                std::min(top, firstTop + std::max(1.0F, std::floor(height / 3.0F)));
            const float frondThickness = steppedCanopyRadius(radius, 0.24F);
            pushCanopyPrism(mesh, leafBlock, centerX, centerZ, radius, frondThickness, bottom,
                            std::min(firstTop, top), true);
            pushCanopyPrism(mesh, leafBlock, centerX, centerZ, frondThickness, radius,
                            std::min(firstTop, top), secondTop, true);
            const float tuftRadius = steppedCanopyRadius(radius, 0.38F);
            pushCanopyPrism(mesh, leafBlock, centerX, centerZ, tuftRadius, tuftRadius, secondTop,
                            top, true);
            return;
        }
        case TreeSpecies::WILLOW: {
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.78F, 1.0F, 0.68F});
            const float tendrilRadius = steppedCanopyRadius(radius, 0.18F);
            const float offset = radius * 0.68F;
            const float tendrilTop = bottom + std::max(1.0F, (top - bottom) * 0.58F);
            for (const auto [dx, dz] : std::array{std::pair{1.0F, 0.0F}, std::pair{-1.0F, 0.0F},
                                                  std::pair{0.0F, 1.0F}, std::pair{0.0F, -1.0F}}) {
                pushCanopyPrism(mesh, leafBlock, centerX + dx * offset, centerZ + dz * offset,
                                tendrilRadius, tendrilRadius, bottom, tendrilTop, true);
            }
            return;
        }
        case TreeSpecies::JUNGLE: {
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.72F, 1.0F, 0.82F});
            const float branchRadius = steppedCanopyRadius(radius, 0.34F);
            const float branchOffset = radius * 0.46F;
            const float branchBottom = bottom + std::max(0.0F, (top - bottom) * 0.22F);
            const float branchTop =
                std::min(top, branchBottom + std::max(1.0F, (top - bottom) * 0.32F));
            for (const auto [dx, dz] : std::array{std::pair{1.0F, 0.0F}, std::pair{-1.0F, 0.0F},
                                                  std::pair{0.0F, 1.0F}, std::pair{0.0F, -1.0F}}) {
                pushCanopyPrism(mesh, leafBlock, centerX + dx * branchOffset,
                                centerZ + dz * branchOffset, branchRadius, branchRadius,
                                branchBottom, branchTop, true);
            }
            return;
        }
        case TreeSpecies::MANGROVE:
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.82F, 1.0F, 0.72F});
            return;
        case TreeSpecies::ALPINE_SCRUB:
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{1.0F, 0.55F});
            return;
        case TreeSpecies::LARGE_OAK:
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.58F, 0.92F, 1.0F, 0.64F});
            return;
        case TreeSpecies::OAK:
        case TreeSpecies::BIRCH:
            pushCanopyStack(mesh, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.65F, 1.0F, 0.65F});
            return;
        case TreeSpecies::FALLEN_LOG:
        case TreeSpecies::COUNT:
            return;
    }
}

float canopyTrunkHalfWidth(BlockType logBlock) {
    switch (logBlock) {
        case BlockType::JUNGLE_LOG:
            return 0.85F;
        case BlockType::MANGROVE_LOG:
            return 0.55F;
        case BlockType::PALM_LOG:
            return 0.35F;
        default:
            return 0.45F;
    }
}

float canopyTrunkTop(BlockType leafBlock, float canopyBottom, float canopyTop) {
    const float height = std::max(0.0F, canopyTop - canopyBottom);
    switch (leafBlock) {
        case BlockType::SPRUCE_LEAVES:
        case BlockType::ACACIA_LEAVES:
        case BlockType::PALM_LEAVES:
            return canopyTop - std::min(1.0F, height * 0.25F);
        case BlockType::WILLOW_LEAVES:
            return canopyBottom + std::floor(height * 0.55F);
        default:
            return canopyBottom + std::floor(height * 0.70F);
    }
}

bool sameFlatTerrain(const FarCell& first, const FarCell& second) {
    return first.flat && second.flat && first.material == second.material &&
           first.terrain[0] == second.terrain[0];
}

bool sameWater(const FarCell& first, const FarCell& second) {
    return first.waterMask == 0x0FU && second.waterMask == 0x0FU && first.flatWater &&
           second.flatWater && !first.discontinuousWater && !second.discontinuousWater &&
           first.waterHeight == second.waterHeight &&
           waterAuthoritiesConnect(first.waterAuthority[0], second.waterAuthority[0], 0.0F);
}

void validateLimits(FarTerrainSchedulerLimits& limits) {
    limits.maxPending = std::max<size_t>(1, limits.maxPending);
    limits.maxCompleted = std::max<size_t>(1, limits.maxCompleted);
    limits.maxCacheEntries = std::max<size_t>(1, limits.maxCacheEntries);
    limits.maxCacheBytes = std::max<size_t>(1, limits.maxCacheBytes);
    limits.maxMaintenanceEntries = std::max<size_t>(1, limits.maxMaintenanceEntries);
    limits.maxMaintenanceBytes = std::max<size_t>(1, limits.maxMaintenanceBytes);
}

void recordAtomicMaximum(std::atomic<size_t>& destination, size_t candidate) {
    size_t previous = destination.load(std::memory_order_relaxed);
    while (previous < candidate &&
           !destination.compare_exchange_weak(previous, candidate, std::memory_order_relaxed)) {
    }
}

struct AzimuthCoverage {
    double start = 0.0;
    double end = 0.0;
    bool cameraInside = false;
};

double normalizedAngle(double angle) {
    const double fullCircle = 2.0 * std::numbers::pi;
    angle = std::fmod(angle, fullCircle);
    return angle < 0.0 ? angle + fullCircle : angle;
}

AzimuthCoverage azimuthCoverage(const FarTerrainBounds& bounds, TerrainHorizonViewpoint viewpoint) {
    AzimuthCoverage result;
    result.cameraInside = viewpoint.x >= static_cast<double>(bounds.minX) &&
                          viewpoint.x <= static_cast<double>(bounds.maxX) &&
                          viewpoint.z >= static_cast<double>(bounds.minZ) &&
                          viewpoint.z <= static_cast<double>(bounds.maxZ);
    if (result.cameraInside) return result;
    std::array<double, 4> angles = {
        normalizedAngle(std::atan2(static_cast<double>(bounds.minZ) - viewpoint.z,
                                   static_cast<double>(bounds.minX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.minZ) - viewpoint.z,
                                   static_cast<double>(bounds.maxX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.maxZ) - viewpoint.z,
                                   static_cast<double>(bounds.maxX) - viewpoint.x)),
        normalizedAngle(std::atan2(static_cast<double>(bounds.maxZ) - viewpoint.z,
                                   static_cast<double>(bounds.minX) - viewpoint.x)),
    };
    std::sort(angles.begin(), angles.end());
    const double fullCircle = 2.0 * std::numbers::pi;
    double largestGap = -1.0;
    size_t gapIndex = 0;
    for (size_t index = 0; index < angles.size(); ++index) {
        const double next = index + 1 < angles.size() ? angles[index + 1] : angles[0] + fullCircle;
        const double gap = next - angles[index];
        if (gap > largestGap) {
            largestGap = gap;
            gapIndex = index;
        }
    }
    result.start = gapIndex + 1 < angles.size() ? angles[gapIndex + 1] : angles[0];
    result.end = result.start + fullCircle - largestGap;
    return result;
}

template <typename Visitor>
size_t visitFullyCoveredBins(const AzimuthCoverage& coverage, Visitor&& visitor) {
    if (coverage.cameraInside || coverage.end <= coverage.start) return 0;
    constexpr double FULL_CIRCLE = 2.0 * std::numbers::pi;
    constexpr double BIN_WIDTH = FULL_CIRCLE / TerrainHorizonCuller::AZIMUTH_BIN_COUNT;
    constexpr double EPSILON = 1.0e-12;
    size_t count = 0;
    // Contract both ends so numerical tolerance can only omit an occluder
    // bin. Expanding here could classify a partially covered bin as fully
    // covered and create a false-positive horizon rejection.
    const int64_t first = static_cast<int64_t>(std::ceil((coverage.start + EPSILON) / BIN_WIDTH));
    const int64_t pastLast = static_cast<int64_t>(std::floor((coverage.end - EPSILON) / BIN_WIDTH));
    for (int64_t unwrapped = first; unwrapped < pastLast; ++unwrapped) {
        const size_t bin = static_cast<size_t>(world_coord::floorMod(
            unwrapped, static_cast<int32_t>(TerrainHorizonCuller::AZIMUTH_BIN_COUNT)));
        ++count;
        visitor(bin);
    }
    return count;
}

template <typename Visitor>
size_t visitIntersectedBins(const AzimuthCoverage& coverage, Visitor&& visitor) {
    if (coverage.cameraInside || coverage.end <= coverage.start) return 0;
    constexpr double FULL_CIRCLE = 2.0 * std::numbers::pi;
    constexpr double BIN_WIDTH = FULL_CIRCLE / TerrainHorizonCuller::AZIMUTH_BIN_COUNT;
    constexpr double EPSILON = 1.0e-12;
    const int64_t first = static_cast<int64_t>(std::floor(coverage.start / BIN_WIDTH));
    const int64_t pastLast =
        static_cast<int64_t>(std::floor((coverage.end - EPSILON) / BIN_WIDTH)) + 1;
    size_t count = 0;
    for (int64_t unwrapped = first; unwrapped < pastLast; ++unwrapped) {
        const size_t bin = static_cast<size_t>(world_coord::floorMod(
            unwrapped, static_cast<int32_t>(TerrainHorizonCuller::AZIMUTH_BIN_COUNT)));
        ++count;
        visitor(bin);
    }
    return count;
}

double farthestHorizontalDistance(const FarTerrainBounds& bounds,
                                  TerrainHorizonViewpoint viewpoint) {
    double farthestSquared = 0.0;
    for (int xSide = 0; xSide < 2; ++xSide) {
        for (int zSide = 0; zSide < 2; ++zSide) {
            const double x = static_cast<double>(xSide == 0 ? bounds.minX : bounds.maxX);
            const double z = static_cast<double>(zSide == 0 ? bounds.minZ : bounds.maxZ);
            farthestSquared = std::max(farthestSquared, (x - viewpoint.x) * (x - viewpoint.x) +
                                                            (z - viewpoint.z) * (z - viewpoint.z));
        }
    }
    return std::sqrt(farthestSquared);
}

} // namespace

uint32_t farTerrainSkirtEdgeMask(
    FarTerrainStep step,
    const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps) {
    uint32_t mask = 0;
    for (size_t edge = 0; edge < displayedNeighborSteps.size(); ++edge) {
        const std::optional<FarTerrainStep> neighbor = displayedNeighborSteps[edge];
        if (neighbor && farTerrainStepSize(step) < farTerrainStepSize(*neighbor)) {
            mask |= 1U << edge;
        }
    }
    return mask;
}

FarTerrainStep farTerrainNextDisplayedStep(FarTerrainStep displayed, FarTerrainStep desired) {
    return displayed == desired ? displayed : desired;
}

FarTerrainRefinementOrder farTerrainRefinementOrder(FarTerrainStep desired) {
    FarTerrainRefinementOrder result;
    const int desiredSize = farTerrainStepSize(desired);
    for (FarTerrainStep step : FAR_TERRAIN_REFINEMENT_STEPS) {
        if (farTerrainStepSize(step) >= desiredSize) result.steps[result.count++] = step;
    }
    return result;
}

FarTerrainStep farTerrainFinestReadyStep(FarTerrainStep displayed, FarTerrainStep desired,
                                         FarTerrainStepMask readySteps) {
    const auto ready = [&](FarTerrainStep step) {
        return (readySteps & farTerrainStepMask(step)) != 0;
    };
    const int displayedSize = farTerrainStepSize(displayed);
    const int desiredSize = farTerrainStepSize(desired);
    if (desiredSize > displayedSize) return ready(desired) ? desired : displayed;

    const FarTerrainRefinementOrder order = farTerrainRefinementOrder(desired);
    for (size_t index = order.count; index > 0; --index) {
        const FarTerrainStep step = order.steps[index - 1];
        if (farTerrainStepSize(step) < displayedSize && ready(step)) return step;
    }
    return displayed;
}

std::optional<FarTerrainStep> farTerrainInitialDisplayedStep(FarTerrainStep desired,
                                                             FarTerrainStepMask readySteps,
                                                             FarTerrainStep coarsestAllowed) {
    if ((readySteps & farTerrainStepMask(FAR_TERRAIN_BASE_STEP)) == 0) return std::nullopt;
    const FarTerrainStep finest =
        farTerrainFinestReadyStep(FAR_TERRAIN_BASE_STEP, desired, readySteps);
    if (farTerrainStepSize(finest) > farTerrainStepSize(coarsestAllowed)) return std::nullopt;
    return finest;
}

bool farTerrainDisplayedStepAllowed(FarTerrainStep step, FarTerrainStep coarsestAllowed) {
    return farTerrainStepSize(step) <= farTerrainStepSize(coarsestAllowed);
}

std::optional<FarTerrainStep> farTerrainReadyTransitionTarget(FarTerrainStep displayed,
                                                              FarTerrainStep desired,
                                                              FarTerrainStepMask readySteps,
                                                              bool transitionActive) {
    if (transitionActive) return std::nullopt;
    const FarTerrainStep best = farTerrainFinestReadyStep(displayed, desired, readySteps);
    return best == displayed ? std::nullopt : std::optional{best};
}

bool farTerrainDeferNearIntermediate(FarTerrainStep displayed, FarTerrainStep desired,
                                     FarTerrainStep target, float parentAgeSeconds) {
    if (displayed != FAR_TERRAIN_BASE_STEP || desired != FarTerrainStep::TWO || target == desired) {
        return false;
    }
    return std::isfinite(parentAgeSeconds) && parentAgeSeconds >= 0.0F &&
           parentAgeSeconds < FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS;
}

std::optional<FarTerrainStep> farTerrainStepForChunkDistance(double chunkDistance) {
    return farTerrainStepForMetrics(chunkDistance, 0.0F);
}

std::optional<FarTerrainStep> farTerrainStepForMetrics(double chunkDistance, float complexity,
                                                       std::optional<FarTerrainStep> previousStep) {
    if (!std::isfinite(chunkDistance) || chunkDistance < FAR_TERRAIN_NEAR_CHUNK_RADIUS ||
        chunkDistance >= FAR_TERRAIN_MAX_CHUNK_RADIUS) {
        return std::nullopt;
    }
    const double boundedComplexity = std::clamp(static_cast<double>(complexity), 0.0, 1.0);
    const double effectiveDistance = chunkDistance / (1.0 + boundedComplexity * 0.35);
    const auto nominalStep = [&] {
        if (effectiveDistance < FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS) return FarTerrainStep::TWO;
        if (effectiveDistance < FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS) return FarTerrainStep::FOUR;
        if (effectiveDistance < FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS) return FarTerrainStep::EIGHT;
        return FarTerrainStep::SIXTEEN;
    }();
    if (!previousStep || *previousStep == FarTerrainStep::ONE ||
        *previousStep == FarTerrainStep::THIRTY_TWO) {
        return nominalStep;
    }

    // A displayed tier survives a small band on both sides of its nominal
    // ring. Camera motion therefore cannot alternate two resident meshes at
    // a boundary, while a jump across more than one ring resolves directly
    // to the distance-selected tier.
    double lower = FAR_TERRAIN_NEAR_CHUNK_RADIUS;
    double upper = FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS;
    double lowerMargin = 0.0;
    double upperMargin = 12.0;
    switch (*previousStep) {
        case FarTerrainStep::ONE:
            break;
        case FarTerrainStep::TWO:
            upper = FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS;
            upperMargin = 12.0;
            break;
        case FarTerrainStep::FOUR:
            lower = FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS;
            upper = FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS;
            lowerMargin = 12.0;
            upperMargin = 16.0;
            break;
        case FarTerrainStep::EIGHT:
            lower = FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS;
            upper = FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS;
            lowerMargin = 16.0;
            upperMargin = 24.0;
            break;
        case FarTerrainStep::SIXTEEN:
            lower = FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS;
            upper = FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS;
            lowerMargin = 24.0;
            upperMargin = 0.0;
            break;
        case FarTerrainStep::THIRTY_TWO:
            break;
    }
    if (effectiveDistance >= lower - lowerMargin && effectiveDistance < upper + upperMargin) {
        return *previousStep;
    }
    return nominalStep;
}

FarTerrainTransitionSample sampleFarTerrainTransition(float elapsedSeconds) {
    if (!std::isfinite(elapsedSeconds) || elapsedSeconds <= 0.0F) return {};
    if (elapsedSeconds >= FAR_TERRAIN_LOD_TRANSITION_SECONDS) {
        return {.drawTarget = true, .complete = true, .fogBlend = 0.0F, .progress = 1.0F};
    }
    const float progress = farTerrainLodTransitionProgressAtSeconds(elapsedSeconds);
    return {
        .drawTarget = progress >= 0.5F, .complete = false, .fogBlend = 0.0F, .progress = progress};
}

FarTerrainLodAdvance advanceFarTerrainLod(FarTerrainStep displayed, FarTerrainStep desired,
                                          std::optional<FarTerrainStep> activeTarget,
                                          float activeElapsedSeconds) {
    if (activeTarget) {
        if (!sampleFarTerrainTransition(activeElapsedSeconds).complete) {
            return {
                .displayed = displayed,
                .transitionTarget = activeTarget,
            };
        }
        return {
            .displayed = *activeTarget,
            .transitionTarget = std::nullopt,
            .completedTransition = true,
        };
    }

    const FarTerrainStep next = farTerrainNextDisplayedStep(displayed, desired);
    return {
        .displayed = displayed,
        .transitionTarget = next == displayed ? std::nullopt : std::optional{next},
    };
}

void selectFarTerrainView(double cameraX, double cameraZ, int visibleChunkRadius,
                          std::vector<FarTerrainViewTile>& output) {
    output.clear();
    if (!std::isfinite(cameraX) || !std::isfinite(cameraZ)) return;
    visibleChunkRadius = std::clamp(visibleChunkRadius, 0, FAR_TERRAIN_MAX_CHUNK_RADIUS);
    if (visibleChunkRadius <= 0) return;

    const double visibleBlocks = static_cast<double>(visibleChunkRadius * CHUNK_EDGE);
    const double visibleSquared = visibleBlocks * visibleBlocks;
    const int64_t cameraBlockX = static_cast<int64_t>(std::floor(cameraX));
    const int64_t cameraBlockZ = static_cast<int64_t>(std::floor(cameraZ));
    const int64_t centerTileX =
        world_coord::floorDiv(cameraBlockX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    const int64_t centerTileZ =
        world_coord::floorDiv(cameraBlockZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
    const int tileRadius = static_cast<int>(std::ceil(visibleBlocks / FAR_TERRAIN_TILE_EDGE)) + 1;
    const size_t squareEdge = static_cast<size_t>(tileRadius * 2 + 1);
    output.reserve(squareEdge * squareEdge);

    auto horizontalDistanceSquared = [&](const FarTerrainBounds& bounds) {
        const double dx = cameraX < static_cast<double>(bounds.minX)
                              ? static_cast<double>(bounds.minX) - cameraX
                          : cameraX > static_cast<double>(bounds.maxX)
                              ? cameraX - static_cast<double>(bounds.maxX)
                              : 0.0;
        const double dz = cameraZ < static_cast<double>(bounds.minZ)
                              ? static_cast<double>(bounds.minZ) - cameraZ
                          : cameraZ > static_cast<double>(bounds.maxZ)
                              ? cameraZ - static_cast<double>(bounds.maxZ)
                              : 0.0;
        return dx * dx + dz * dz;
    };
    for (int dz = -tileRadius; dz <= tileRadius; ++dz) {
        for (int dx = -tileRadius; dx <= tileRadius; ++dx) {
            FarTerrainKey key{centerTileX + dx, centerTileZ + dz, FarTerrainStep::TWO};
            FarTerrainBounds bounds;
            bounds.minX = tileOrigin(key.tileX);
            bounds.maxX = bounds.minX + FAR_TERRAIN_TILE_EDGE;
            bounds.minZ = tileOrigin(key.tileZ);
            bounds.maxZ = bounds.minZ + FAR_TERRAIN_TILE_EDGE;
            bounds.minY = static_cast<float>(WORLD_MIN_Y);
            bounds.maxY = static_cast<float>(WORLD_MAX_Y + 1);
            const double nearestSquared = horizontalDistanceSquared(bounds);
            if (nearestSquared >= visibleSquared) continue;
            const double lodDistanceChunks =
                std::max(static_cast<double>(FAR_TERRAIN_NEAR_CHUNK_RADIUS),
                         std::sqrt(nearestSquared) / CHUNK_EDGE);
            const auto step = farTerrainStepForChunkDistance(lodDistanceChunks);
            if (!step) continue;
            key.step = *step;
            output.push_back({key, bounds, nearestSquared, lodDistanceChunks});
        }
    }
    std::sort(output.begin(), output.end(), [](const auto& first, const auto& second) {
        if (first.distanceSquared != second.distanceSquared) {
            return first.distanceSquared < second.distanceSquared;
        }
        if (first.key.tileX != second.key.tileX) return first.key.tileX < second.key.tileX;
        return first.key.tileZ < second.key.tileZ;
    });
}

FarTerrainCoverageFrontier
farTerrainCoverageFrontier(const std::vector<FarTerrainViewTile>& selected,
                           const FarTerrainResidencyFunction& isResident) {
    FarTerrainCoverageFrontier result;
    if (!isResident) {
        result.missingBaseTiles = static_cast<uint32_t>(selected.size());
        result.complete = selected.empty();
        return result;
    }

    double nearestMissingSquared = std::numeric_limits<double>::infinity();
    for (const FarTerrainViewTile& tile : selected) {
        const FarTerrainKey base{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        if (isResident(base)) continue;
        ++result.missingBaseTiles;
        nearestMissingSquared = std::min(nearestMissingSquared, tile.distanceSquared);
    }
    result.complete = result.missingBaseTiles == 0;
    if (!result.complete) {
        result.distanceSquaredBlocks = nearestMissingSquared;
        result.distanceBlocks = static_cast<float>(std::sqrt(nearestMissingSquared));
    }
    return result;
}

bool farTerrainCoverageDrawEligible(double tileDistanceSquared,
                                    const FarTerrainCoverageFrontier& frontier) {
    return frontier.complete || tileDistanceSquared < frontier.distanceSquaredBlocks;
}

bool farTerrainCoveragePatchMayOcclude(const FarTerrainBounds& patch,
                                       TerrainHorizonViewpoint viewpoint,
                                       const FarTerrainCoverageFrontier& frontier,
                                       double coverageFadeBlocks, bool lodTransitionActive) {
    if (lodTransitionActive) return false;
    if (frontier.complete) return true;
    const double opaqueRadius =
        std::max(0.0, static_cast<double>(frontier.distanceBlocks) - coverageFadeBlocks);
    const double opaqueRadiusSquared = opaqueRadius * opaqueRadius;
    double farthestSquared = 0.0;
    for (const int64_t x : {patch.minX, patch.maxX}) {
        for (const int64_t z : {patch.minZ, patch.maxZ}) {
            const double dx = static_cast<double>(x) - viewpoint.x;
            const double dz = static_cast<double>(z) - viewpoint.z;
            farthestSquared = std::max(farthestSquared, dx * dx + dz * dz);
        }
    }
    return farthestSquared < opaqueRadiusSquared;
}

bool farTerrainConnectedRefinementEligible(const FarTerrainViewTile& tile,
                                           float actualExactHandoffBlocks,
                                           const FarTerrainCoverageFrontier& frontier,
                                           bool baseResident, bool cameraTile) {
    if (!std::isfinite(actualExactHandoffBlocks) || actualExactHandoffBlocks < 0.0F ||
        (!baseResident && !cameraTile) ||
        (!cameraTile && !farTerrainCoverageDrawEligible(tile.distanceSquared, frontier))) {
        return false;
    }
    return tile.key.step == FarTerrainStep::TWO || tile.key.step == FarTerrainStep::FOUR ||
           tile.key.step == FarTerrainStep::EIGHT || tile.key.step == FarTerrainStep::SIXTEEN;
}

void buildFarTerrainProgressiveSubmissionOrder(
    std::span<const FarTerrainRefinementCacheRequest> requests,
    std::vector<FarTerrainKey>& output) {
    output.clear();
    output.reserve(requests.size() * 3 + FAR_TERRAIN_NEAR_FALLBACK_TILE_COUNT * 2 + 1);
    const auto append = [&](FarTerrainKey key) {
        if (std::ranges::find(output, key) == output.end()) output.push_back(key);
    };

    std::array<const FarTerrainRefinementCacheRequest*, FAR_TERRAIN_NEAR_FALLBACK_TILE_COUNT>
        nearStepTwo{};
    size_t nearStepTwoCount = 0;
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive || request.desired != FarTerrainStep::TWO ||
            request.requiresFineFallback) {
            continue;
        }
        nearStepTwo[nearStepTwoCount++] = &request;
        if (nearStepTwoCount == nearStepTwo.size()) break;
    }

    // The camera exploration band must not display a coarse approximation, so
    // its step-2 fallbacks lead the connected lane. Step 8 is both finer and
    // slightly faster than step 16 on the reference path and protects the
    // rest of the unresolved exact-loading disk.
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (!request.transitionActive && request.requiresBlockScaleFallback &&
            request.desired == FarTerrainStep::TWO) {
            append({request.coordinate.x, request.coordinate.z, FarTerrainStep::TWO});
        }
    }
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (!request.transitionActive && request.requiresFineFallback &&
            !request.requiresBlockScaleFallback && request.desired == FarTerrainStep::TWO) {
            append({request.coordinate.x, request.coordinate.z, FarTerrainStep::EIGHT});
        }
    }
    for (const auto* request : std::span(nearStepTwo).first(nearStepTwoCount)) {
        append({request->coordinate.x, request->coordinate.z, FarTerrainStep::EIGHT});
    }
    const auto firstStepTwo = std::ranges::find_if(requests, [](const auto& request) {
        return !request.transitionActive && request.desired == FarTerrainStep::TWO;
    });
    if (firstStepTwo != requests.end()) {
        const auto* request = &*firstStepTwo;
        append({request->coordinate.x, request->coordinate.z, FarTerrainStep::TWO});
    }
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (!request.transitionActive && request.requiresFineFallback &&
            !request.requiresBlockScaleFallback && request.desired == FarTerrainStep::TWO) {
            append({request.coordinate.x, request.coordinate.z, FarTerrainStep::FOUR});
        }
    }
    for (const auto* request : std::span(nearStepTwo).first(nearStepTwoCount)) {
        append({request->coordinate.x, request->coordinate.z, FarTerrainStep::FOUR});
    }
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive) continue;
        append({request.coordinate.x, request.coordinate.z, request.desired});
    }
}

size_t
reserveFarTerrainIntermediateTransitionSlots(std::span<FarTerrainRefinementCacheRequest> requests,
                                             size_t activeTransitions) {
    size_t remaining = FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS -
                       std::min(activeTransitions, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS);
    size_t reserved = 0;
    for (FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive || request.deferIntermediate || request.requiresFineFallback)
            continue;
        if (remaining == 0) {
            request.deferIntermediate = true;
            continue;
        }
        --remaining;
        ++reserved;
    }
    return reserved;
}

bool farTerrainRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier,
                                  bool allBaseCandidatesScanned) {
    return frontier.complete && allBaseCandidatesScanned;
}

bool farTerrainSubmissionBefore(FarTerrainKey first, uint32_t firstViewPriority,
                                FarTerrainKey second, uint32_t secondViewPriority) {
    const bool firstBase = farTerrainIsBaseStep(first.step);
    const bool secondBase = farTerrainIsBaseStep(second.step);
    if (firstBase != secondBase) return firstBase;
    if (firstViewPriority != secondViewPriority) return firstViewPriority < secondViewPriority;
    if (first.tileX != second.tileX) return first.tileX < second.tileX;
    if (first.tileZ != second.tileZ) return first.tileZ < second.tileZ;
    return farTerrainStepSize(first.step) < farTerrainStepSize(second.step);
}

void buildFarTerrainResidencyOrder(const std::vector<FarTerrainViewTile>& selected,
                                   std::vector<FarTerrainKey>& output) {
    output.clear();
    output.reserve(selected.size() * (FAR_TERRAIN_REFINEMENT_STEPS.size() + 1));
    for (const FarTerrainViewTile& tile : selected) {
        output.push_back({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    for (const FarTerrainViewTile& tile : selected) {
        output.push_back(tile.key);
    }
    for (const FarTerrainViewTile& tile : selected) {
        const FarTerrainRefinementOrder order = farTerrainRefinementOrder(tile.key.step);
        for (FarTerrainStep step : std::span(order.steps).first(order.count)) {
            if (step == tile.key.step) continue;
            output.push_back({tile.key.tileX, tile.key.tileZ, step});
        }
    }
}

bool farTerrainResidencyOrderMatches(const std::vector<FarTerrainViewTile>& selected,
                                     std::span<const FarTerrainKey> order) {
    size_t refinementCount = 0;
    for (const FarTerrainViewTile& tile : selected)
        refinementCount += farTerrainRefinementOrder(tile.key.step).count;
    if (order.size() != selected.size() + refinementCount) return false;
    for (size_t index = 0; index < selected.size(); ++index) {
        const FarTerrainKey expected{selected[index].key.tileX, selected[index].key.tileZ,
                                     FAR_TERRAIN_BASE_STEP};
        if (order[index] != expected) return false;
    }
    size_t orderIndex = selected.size();
    for (const FarTerrainViewTile& tile : selected) {
        if (order[orderIndex++] != tile.key) return false;
    }
    for (const FarTerrainViewTile& tile : selected) {
        const FarTerrainRefinementOrder refinement = farTerrainRefinementOrder(tile.key.step);
        for (FarTerrainStep step : std::span(refinement.steps).first(refinement.count)) {
            if (step == tile.key.step) continue;
            const FarTerrainKey expected{tile.key.tileX, tile.key.tileZ, step};
            if (order[orderIndex++] != expected) return false;
        }
    }
    return true;
}

bool farTerrainResidencyMembershipMatches(
    const std::vector<FarTerrainViewTile>& selected,
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted) {
    size_t expectedCount = selected.size();
    for (const FarTerrainViewTile& tile : selected) {
        expectedCount += farTerrainRefinementOrder(tile.key.step).count;
    }
    if (wanted.size() != expectedCount) return false;
    for (const FarTerrainViewTile& tile : selected) {
        if (!wanted.contains({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP})) return false;
        const FarTerrainRefinementOrder refinement = farTerrainRefinementOrder(tile.key.step);
        for (FarTerrainStep step : std::span(refinement.steps).first(refinement.count)) {
            if (!wanted.contains({tile.key.tileX, tile.key.tileZ, step})) return false;
        }
    }
    return true;
}

double farTerrainColumnDistanceSquared(double cameraX, double cameraZ, ColumnPos column) {
    const double minimumX = static_cast<double>(column.x) * CHUNK_EDGE;
    const double maximumX = minimumX + CHUNK_EDGE;
    const double minimumZ = static_cast<double>(column.z) * CHUNK_EDGE;
    const double maximumZ = minimumZ + CHUNK_EDGE;
    const double dx = cameraX < minimumX   ? minimumX - cameraX
                      : cameraX > maximumX ? cameraX - maximumX
                                           : 0.0;
    const double dz = cameraZ < minimumZ   ? minimumZ - cameraZ
                      : cameraZ > maximumZ ? cameraZ - maximumZ
                                           : 0.0;
    return dx * dx + dz * dz;
}

bool farTerrainExactSectionReady(uint32_t builtRevision, uint32_t currentRevision) {
    return builtRevision == currentRevision;
}

bool farTerrainExactSectionOwnsSurface(bool previouslyPublished, uint32_t builtRevision,
                                       uint32_t currentRevision) {
    return previouslyPublished || farTerrainExactSectionReady(builtRevision, currentRevision);
}

bool farTerrainExactStreamingBusy(size_t pendingChunks, size_t schedulerOwnedMeshes,
                                  size_t consumerPendingMeshes, size_t requiredSections,
                                  size_t readySections, size_t unresolvedColumns) {
    return pendingChunks != 0 || schedulerOwnedMeshes != 0 || consumerPendingMeshes != 0 ||
           readySections < requiredSections || unresolvedColumns != 0;
}

void FarTerrainExactCoverageCache::rebuild(uint64_t epoch, int nominalRadiusChunks,
                                           std::span<const ChunkPos> requiredSections,
                                           std::span<const ColumnPos> unresolvedColumns,
                                           const FarTerrainExactReadinessFunction& isReady) {
    clear();
    valid_ = true;
    epoch_ = epoch;
    nominalRadiusChunks_ = std::max(0, nominalRadiusChunks);
    handoff_.requiredSections = requiredSections.size();
    handoff_.unresolvedColumns = unresolvedColumns.size();
    columns_.reserve(requiredSections.size() / 2 + unresolvedColumns.size());
    columnIndices_.reserve(requiredSections.size() / 2 + unresolvedColumns.size());
    sectionColumns_.reserve(requiredSections.size());
    readySections_.reserve(requiredSections.size());

    const auto findOrAddColumn = [&](ColumnPos coordinate) {
        const auto [found, inserted] = columnIndices_.try_emplace(coordinate, columns_.size());
        if (inserted) columns_.push_back(ColumnState{.coordinate = coordinate});
        return found->second;
    };
    for (const ChunkPos section : requiredSections) {
        const ColumnPos column{section.x, section.z};
        const size_t columnIndex = findOrAddColumn(column);
        const auto [_, inserted] = sectionColumns_.try_emplace(section, columnIndex);
        if (!inserted) continue;
        ColumnState& state = columns_[columnIndex];
        if (state.requiredSections != std::numeric_limits<uint16_t>::max()) {
            ++state.requiredSections;
        }
        if (isReady && isReady(section)) {
            readySections_.insert(section);
            if (state.readySections != std::numeric_limits<uint16_t>::max()) {
                ++state.readySections;
            }
            ++handoff_.readySections;
        }
    }
    for (const ColumnPos column : unresolvedColumns) {
        columns_[findOrAddColumn(column)].unresolved = true;
    }

    constexpr int64_t COLUMNS_PER_TILE = FAR_TERRAIN_EXACT_COLUMNS_PER_TILE;
    static_assert(FAR_TERRAIN_TILE_EDGE / CHUNK_EDGE == COLUMNS_PER_TILE);
    handoff_.tileStateIndices.reserve(FarTerrainExactHandoff::MAX_TILE_STATES);
    for (size_t columnIndex = 0; columnIndex < columns_.size(); ++columnIndex) {
        ColumnState& column = columns_[columnIndex];
        const ColumnPos tileCoordinate{
            world_coord::floorDiv(column.coordinate.x, COLUMNS_PER_TILE),
            world_coord::floorDiv(column.coordinate.z, COLUMNS_PER_TILE),
        };
        auto tile = handoff_.tileStateIndices.find(tileCoordinate);
        if (tile == handoff_.tileStateIndices.end() &&
            handoff_.tileStateCount < FarTerrainExactHandoff::MAX_TILE_STATES) {
            const uint8_t index = static_cast<uint8_t>(handoff_.tileStateCount++);
            handoff_.tileStates[index].coordinate = tileCoordinate;
            handoff_.tileStates[index].ready = true;
            tile = handoff_.tileStateIndices.emplace(tileCoordinate, index).first;
        }
        if (tile != handoff_.tileStateIndices.end()) column.tileIndex = tile->second;
        const uint32_t localX =
            static_cast<uint32_t>(world_coord::floorMod(column.coordinate.x, COLUMNS_PER_TILE));
        const uint32_t localZ =
            static_cast<uint32_t>(world_coord::floorMod(column.coordinate.z, COLUMNS_PER_TILE));
        column.bit = static_cast<uint16_t>(localZ * COLUMNS_PER_TILE + localX);
        if (column.tileIndex != UINT8_MAX) {
            FarTerrainExactHandoff::TileState& state = handoff_.tileStates[column.tileIndex];
            state.hasRequirements = true;
            state.requiredColumns[column.bit / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD] |=
                1U << (column.bit % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD);
        }
    }
    for (size_t columnIndex = 0; columnIndex < columns_.size(); ++columnIndex) {
        const ColumnState& column = columns_[columnIndex];
        setColumnIncomplete(columnIndex,
                            column.unresolved || column.readySections < column.requiredSections);
    }
}

void FarTerrainExactCoverageCache::clear() {
    valid_ = false;
    epoch_ = 0;
    nominalRadiusChunks_ = 0;
    handoff_ = {};
    columns_.clear();
    columnIndices_.clear();
    sectionColumns_.clear();
    readySections_.clear();
    incompleteColumns_.clear();
    lastSampleColumnVisits_ = 0;
}

bool FarTerrainExactCoverageCache::matches(uint64_t epoch, int nominalRadiusChunks) const {
    return valid_ && epoch_ == epoch && nominalRadiusChunks_ == std::max(0, nominalRadiusChunks);
}

void FarTerrainExactCoverageCache::setColumnIncomplete(size_t columnIndex, bool incomplete) {
    ColumnState& column = columns_[columnIndex];
    const bool wasIncomplete = column.incompleteListIndex != std::numeric_limits<size_t>::max();
    if (incomplete != wasIncomplete) {
        if (incomplete) {
            column.incompleteListIndex = incompleteColumns_.size();
            incompleteColumns_.push_back(columnIndex);
        } else {
            const size_t removed = column.incompleteListIndex;
            const size_t movedColumn = incompleteColumns_.back();
            incompleteColumns_[removed] = movedColumn;
            columns_[movedColumn].incompleteListIndex = removed;
            incompleteColumns_.pop_back();
            column.incompleteListIndex = std::numeric_limits<size_t>::max();
        }
    }
    if (column.tileIndex == UINT8_MAX) return;
    FarTerrainExactHandoff::TileState& tile = handoff_.tileStates[column.tileIndex];
    const size_t word = column.bit / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD;
    const uint32_t mask = 1U << (column.bit % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD);
    if (incomplete) {
        tile.incompleteColumns[word] |= mask;
    } else {
        tile.incompleteColumns[word] &= ~mask;
    }
    tile.ready =
        tile.hasRequirements &&
        std::ranges::all_of(tile.incompleteColumns, [](uint32_t bits) { return bits == 0U; });
}

bool FarTerrainExactCoverageCache::setSectionReady(ChunkPos section, bool ready) {
    const auto found = sectionColumns_.find(section);
    if (found == sectionColumns_.end()) return false;
    const bool wasReady = readySections_.contains(section);
    if (wasReady == ready) return false;
    ColumnState& column = columns_[found->second];
    const bool wasIncomplete = column.unresolved || column.readySections < column.requiredSections;
    if (ready) {
        readySections_.insert(section);
        ++column.readySections;
        ++handoff_.readySections;
    } else {
        readySections_.erase(section);
        --column.readySections;
        --handoff_.readySections;
    }
    const bool incomplete = column.unresolved || column.readySections < column.requiredSections;
    if (incomplete != wasIncomplete) setColumnIncomplete(found->second, incomplete);
    return true;
}

const FarTerrainExactHandoff& FarTerrainExactCoverageCache::sample(double cameraX, double cameraZ) {
    const double nominalBlocks = static_cast<double>(nominalRadiusChunks_) * CHUNK_EDGE;
    double handoffSquared = nominalBlocks * nominalBlocks;
    for (size_t index = 0; index < handoff_.tileStateCount; ++index) {
        handoff_.tileStates[index].limitingDistanceBlocks = std::numeric_limits<float>::infinity();
    }
    lastSampleColumnVisits_ = incompleteColumns_.size();
    for (const size_t columnIndex : incompleteColumns_) {
        const ColumnState& column = columns_[columnIndex];
        const double distanceSquared =
            farTerrainColumnDistanceSquared(cameraX, cameraZ, column.coordinate);
        handoffSquared = std::min(handoffSquared, distanceSquared);
        if (column.tileIndex == UINT8_MAX) continue;
        FarTerrainExactHandoff::TileState& tile = handoff_.tileStates[column.tileIndex];
        tile.limitingDistanceBlocks =
            std::min(tile.limitingDistanceBlocks, static_cast<float>(std::sqrt(distanceSquared)));
    }
    handoff_.distanceBlocks = static_cast<float>(std::sqrt(handoffSquared));
    return handoff_;
}

FarTerrainExactHandoff farTerrainExactHandoff(double cameraX, double cameraZ,
                                              int nominalRadiusChunks,
                                              std::span<const ChunkPos> requiredSections,
                                              std::span<const ColumnPos> unresolvedColumns,
                                              const FarTerrainExactReadinessFunction& isReady) {
    FarTerrainExactCoverageCache cache;
    cache.rebuild(1, nominalRadiusChunks, requiredSections, unresolvedColumns, isReady);
    return cache.sample(cameraX, cameraZ);
}

bool FarTerrainExactHandoff::tileFullyReady(ColumnPos coordinate) const {
    const auto found = tileStateIndices.find(coordinate);
    if (found == tileStateIndices.end()) return false;
    const TileState& state = tileStates[found->second];
    return state.hasRequirements && state.ready;
}

bool FarTerrainExactHandoff::tileFullyOwned(ColumnPos coordinate) const {
    const ColumnMask mask = readyColumnMask(coordinate);
    return std::ranges::all_of(
        mask, [](uint32_t word) { return word == std::numeric_limits<uint32_t>::max(); });
}

FarTerrainExactHandoff::ColumnMask
FarTerrainExactHandoff::readyColumnMask(ColumnPos tileCoordinate) const {
    ColumnMask result{};
    const auto found = tileStateIndices.find(tileCoordinate);
    if (found == tileStateIndices.end()) return result;
    const TileState& state = tileStates[found->second];
    for (size_t word = 0; word < result.size(); ++word) {
        result[word] = state.requiredColumns[word] & ~state.incompleteColumns[word];
    }
    return result;
}

bool FarTerrainExactHandoff::columnFullyReady(ColumnPos chunkColumn) const {
    constexpr int64_t COLUMNS_PER_TILE = FAR_TERRAIN_TILE_EDGE / CHUNK_EDGE;
    const ColumnPos tileCoordinate{
        world_coord::floorDiv(chunkColumn.x, COLUMNS_PER_TILE),
        world_coord::floorDiv(chunkColumn.z, COLUMNS_PER_TILE),
    };
    const uint32_t localX = static_cast<uint32_t>(
        world_coord::floorMod(chunkColumn.x, static_cast<int64_t>(COLUMNS_PER_TILE)));
    const uint32_t localZ = static_cast<uint32_t>(
        world_coord::floorMod(chunkColumn.z, static_cast<int64_t>(COLUMNS_PER_TILE)));
    const uint32_t bit = localZ * static_cast<uint32_t>(COLUMNS_PER_TILE) + localX;
    const ColumnMask mask = readyColumnMask(tileCoordinate);
    return (mask[bit / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD] &
            (1U << (bit % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD))) != 0U;
}

float FarTerrainExactHandoff::distanceBlocksForTile(ColumnPos coordinate,
                                                    float nominalDistanceBlocks) const {
    const auto found = tileStateIndices.find(coordinate);
    if (found == tileStateIndices.end()) return distanceBlocks;
    const TileState& state = tileStates[found->second];
    if (state.hasRequirements && state.ready) return nominalDistanceBlocks;
    if (std::isfinite(state.limitingDistanceBlocks)) return state.limitingDistanceBlocks;
    return distanceBlocks;
}

bool farTerrainRequiresCoverageParent(double cameraX, double cameraZ, ColumnPos tile,
                                      float nominalDistanceBlocks,
                                      const FarTerrainExactHandoff& handoff) {
    if (nominalDistanceBlocks <= 0.0F || handoff.tileFullyOwned(tile)) return false;
    const int64_t tileMinimumX = tile.x * FAR_TERRAIN_TILE_EDGE;
    const int64_t tileMinimumZ = tile.z * FAR_TERRAIN_TILE_EDGE;
    const int64_t tileMaximumX = tileMinimumX + FAR_TERRAIN_TILE_EDGE;
    const int64_t tileMaximumZ = tileMinimumZ + FAR_TERRAIN_TILE_EDGE;
    const double dx =
        cameraX < static_cast<double>(tileMinimumX)   ? static_cast<double>(tileMinimumX) - cameraX
        : cameraX > static_cast<double>(tileMaximumX) ? cameraX - static_cast<double>(tileMaximumX)
                                                      : 0.0;
    const double dz =
        cameraZ < static_cast<double>(tileMinimumZ)   ? static_cast<double>(tileMinimumZ) - cameraZ
        : cameraZ > static_cast<double>(tileMaximumZ) ? cameraZ - static_cast<double>(tileMaximumZ)
                                                      : 0.0;
    const double handoffBand =
        static_cast<double>(nominalDistanceBlocks) + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS;
    return dx * dx + dz * dz < handoffBand * handoffBand;
}

size_t FarTerrainKeyHash::operator()(const FarTerrainKey& key) const noexcept {
    uint64_t hash = mix64(static_cast<uint64_t>(key.tileX));
    hash = mix64(hash ^ static_cast<uint64_t>(key.tileZ));
    hash = mix64(hash ^ static_cast<uint8_t>(key.step));
    return static_cast<size_t>(hash);
}

size_t FarTerrainMesh::byteSize() const {
    return sizeof(*this) + vertices.capacity() * sizeof(Vertex) +
           indices.capacity() * sizeof(uint32_t);
}

std::shared_ptr<const FarTerrainMesh> FarTerrainMesher::build(FarTerrainKey key,
                                                              const FarTerrainSource& source) {
    if (!source.sample) {
        throw std::invalid_argument("far terrain source is incomplete");
    }
    if (!validStep(key.step)) throw std::invalid_argument("unsupported far terrain LOD step");
    const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(key.step);
    const int step = farTerrainStepSize(key.step);
    const bool exactNearWater = key.step == FarTerrainStep::TWO || key.step == FarTerrainStep::FOUR;
    const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
    const int sampleEdge = cellEdge + 1;
    auto mesh = std::make_shared<FarTerrainMesh>();
    mesh->key = key;
    mesh->originX = tileOrigin(key.tileX);
    mesh->originZ = tileOrigin(key.tileZ);
    mesh->bounds.minX = mesh->originX;
    mesh->bounds.maxX = mesh->originX + FAR_TERRAIN_TILE_EDGE;
    mesh->bounds.minZ = mesh->originZ;
    mesh->bounds.maxZ = mesh->originZ + FAR_TERRAIN_TILE_EDGE;
    mesh->bounds.minY = std::numeric_limits<float>::max();
    mesh->bounds.maxY = std::numeric_limits<float>::lowest();
    mesh->surfaceBounds = mesh->bounds;
    constexpr int PATCHES_PER_EDGE = FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
    for (int patchZ = 0; patchZ < PATCHES_PER_EDGE; ++patchZ) {
        for (int patchX = 0; patchX < PATCHES_PER_EDGE; ++patchX) {
            FarTerrainBounds& patch =
                mesh->occluderPatches[static_cast<size_t>(patchZ * PATCHES_PER_EDGE + patchX)];
            patch.minX = mesh->originX + patchX * FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.maxX = patch.minX + FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.minZ = mesh->originZ + patchZ * FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.maxZ = patch.minZ + FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            patch.minY = std::numeric_limits<float>::max();
            patch.maxY = std::numeric_limits<float>::lowest();
        }
    }

    std::vector<FarSurfaceSample> samples(static_cast<size_t>(sampleEdge * sampleEdge));
    auto surfaceAt = [&](int x, int z) -> FarSurfaceSample& {
        return samples[static_cast<size_t>(z * sampleEdge + x)];
    };
    auto sampleAt = [&](int x, int z) -> FarTerrainGeometrySample& {
        return surfaceAt(x, z).geometry;
    };
    if (source.sampleGrid) {
        source.sampleGrid(mesh->originX, mesh->originZ, step, sampleEdge, footprint, samples);
    } else {
        for (int z = 0; z < sampleEdge; ++z) {
            for (int x = 0; x < sampleEdge; ++x) {
                const int64_t worldX = mesh->originX + static_cast<int64_t>(x * step);
                const int64_t worldZ = mesh->originZ + static_cast<int64_t>(z * step);
                surfaceAt(x, z) = source.sample(worldX, worldZ, footprint);
            }
        }
    }

    const int cellBoundsEdge = cellEdge + 2;
    std::vector<FarTerrainCellBounds> authoritativeBounds;
    if (source.cellBoundsGrid) {
        if (mesh->originX < std::numeric_limits<int64_t>::min() + step ||
            mesh->originZ < std::numeric_limits<int64_t>::min() + step) {
            throw std::out_of_range("far terrain cell bounds apron exceeds int64 range");
        }
        authoritativeBounds.resize(static_cast<size_t>(cellBoundsEdge * cellBoundsEdge));
        source.cellBoundsGrid(mesh->originX - step, mesh->originZ - step, step, cellBoundsEdge,
                              cellBoundsEdge, footprint, authoritativeBounds);
    }
    const auto authoritativeBoundsAt = [&](int cellX, int cellZ) -> const FarTerrainCellBounds& {
        return authoritativeBounds[static_cast<size_t>((cellZ + 1) * cellBoundsEdge + cellX + 1)];
    };

    double maximumSampleSlope = 0.0;
    bool hasWaterBoundary = false;
    bool hasChannelFeature = false;
    bool hasLakeFeature = false;
    auto sampleIsWet = [](const FarTerrainGeometrySample& sample) {
        return hasWater(sample) && sample.waterSurface > sample.terrainHeight + 0.01;
    };
    using GeometryFunction =
        std::function<FarTerrainGeometrySample(int64_t worldX, int64_t worldZ)>;
    auto makeWaterEdgeRefiner = [&](GeometryFunction sampler) {
        return WaterEdgeRefiner{
            [&, sampler = std::move(sampler)](const WaterPoint& first, const WaterPoint& second,
                                              const FarWaterAuthority& target,
                                              float connectionSpan) {
                WaterPoint wet = first.wet ? first : second;
                const WaterPoint dry = first.wet ? second : first;
                // The canonical two- and four-block contour pages already
                // bound the shoreline's prescribed narrow band. Their edge
                // crossing is the midpoint between unlike signs, exactly as
                // in marching squares, and needs no further block-by-block
                // hydrology probes.
                if (key.step == FarTerrainStep::THIRTY_TWO || connectionSpan <= 5.75F) {
                    // Step-32 shoreline cells are adaptively subdivided to a
                    // four-block canonical raster before reaching this path.
                    // Their two-block center probes therefore own the same
                    // midpoint crossing as every finer contour. Retaining the
                    // old 7/8 crossing stretched one coarse wet corner across
                    // dry exact columns during a cold handoff.
                    const float crossing = connectionSpan <= 5.75F ? 0.5F : 0.875F;
                    return WaterPoint{
                        .x = wet.x + (dry.x - wet.x) * crossing,
                        .z = wet.z + (dry.z - wet.z) * crossing,
                        .height = wet.height,
                        .wet = true,
                        .authority = target,
                    };
                }
                const int subdivisions =
                    std::max(1, static_cast<int>(std::ceil(
                                    std::max(std::abs(dry.x - wet.x), std::abs(dry.z - wet.z)))));
                WaterPoint lastWet = wet;
                for (int subdivision = 1; subdivision <= subdivisions; ++subdivision) {
                    const float amount = static_cast<float>(subdivision) / subdivisions;
                    const float localX = wet.x + (dry.x - wet.x) * amount;
                    const float localZ = wet.z + (dry.z - wet.z) * amount;
                    const int64_t worldX =
                        mesh->originX + static_cast<int64_t>(std::llround(localX));
                    const int64_t worldZ =
                        mesh->originZ + static_cast<int64_t>(std::llround(localZ));
                    const FarTerrainGeometrySample sample = sampler(worldX, worldZ);
                    const bool surfaceWet = sampleIsWet(sample);
                    const FarWaterAuthority authority = waterAuthority(sample, surfaceWet);
                    WaterPoint current{
                        .x = localX,
                        .z = localZ,
                        .height = vertexHeight(sample.waterSurface),
                        .wet = waterAuthoritiesConnect(authority, target, connectionSpan),
                        .authority = authority,
                    };
                    if (!current.wet) {
                        return WaterPoint{
                            .x = (lastWet.x + current.x) * 0.5F,
                            .z = (lastWet.z + current.z) * 0.5F,
                            .height = lastWet.height,
                            .wet = true,
                            .authority = target,
                        };
                    }
                    lastWet = current;
                }
                return WaterPoint{
                    .x = (wet.x + dry.x) * 0.5F,
                    .z = (wet.z + dry.z) * 0.5F,
                    .height = wet.height,
                    .wet = true,
                    .authority = target,
                };
            },
        };
    };
    std::unordered_map<ColumnPos, FarTerrainGeometrySample> canonicalWaterSamples;
    const worldgen::SurfaceFootprint canonicalWaterFootprint = worldgen::SurfaceFootprint::BLOCK_1;
    const auto preloadCanonicalWater = [&](std::span<const ColumnPos> requested,
                                           bool basinAuthorityOnly = false) {
        if (requested.empty()) return;
        std::vector<ColumnPos> positions;
        positions.reserve(requested.size());
        for (const ColumnPos position : requested) {
            if (canonicalWaterSamples.contains(position)) continue;
            const auto [_, inserted] = canonicalWaterSamples.try_emplace(position);
            if (inserted) positions.push_back(position);
        }
        if (positions.empty()) return;
        std::vector<FarTerrainGeometrySample> geometry(positions.size());
        if (basinAuthorityOnly && source.waterAuthorityPoints) {
            source.waterAuthorityPoints(positions, canonicalWaterFootprint, geometry);
        } else if (source.geometryPoints) {
            source.geometryPoints(positions, canonicalWaterFootprint, geometry);
        } else if (source.geometryGrid) {
            for (size_t index = 0; index < positions.size(); ++index) {
                source.geometryGrid(positions[index].x, positions[index].z, 1, 1, 1, 1,
                                    canonicalWaterFootprint,
                                    std::span<FarTerrainGeometrySample>(&geometry[index], 1));
            }
        } else {
            for (size_t index = 0; index < positions.size(); ++index) {
                geometry[index] =
                    source.sample(positions[index].x, positions[index].z, canonicalWaterFootprint)
                        .geometry;
            }
        }
        for (size_t index = 0; index < positions.size(); ++index) {
            canonicalWaterSamples.find(positions[index])->second = geometry[index];
        }
    };
    const GeometryFunction canonicalWaterGeometry = [&](int64_t worldX, int64_t worldZ) {
        const ColumnPos position{worldX, worldZ};
        if (const auto found = canonicalWaterSamples.find(position);
            found != canonicalWaterSamples.end()) {
            return found->second;
        }
        if (source.geometryPoints &&
            (key.step == FarTerrainStep::EIGHT || key.step == FarTerrainStep::SIXTEEN)) {
            throw std::logic_error("coarse canonical water probe was not included in a bulk batch");
        }
        const std::array positions{position};
        preloadCanonicalWater(positions);
        return canonicalWaterSamples.find(position)->second;
    };
    const WaterEdgeRefiner refineWaterEdge = makeWaterEdgeRefiner(canonicalWaterGeometry);

    // Every LOD samples tile-face water on the same two-block lattice. A
    // narrow river can be smaller than a coarse interior cell, but it cannot
    // disappear on one side of a fine/coarse tile seam.
    constexpr int SHARED_WATER_EDGE_STEP = 2;
    enum WaterEdge : size_t { WEST = 0, EAST = 1, NORTH = 2, SOUTH = 3 };
    std::array<bool, 4> refineWaterBoundary{};
    const bool refineSharedWaterBoundary =
        key.step != FarTerrainStep::TWO && key.step != FarTerrainStep::THIRTY_TWO;
    if (refineSharedWaterBoundary && source.geometryPoints) {
        constexpr int EDGE_SAMPLE_COUNT = FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP + 1;
        std::vector<ColumnPos> edgePositions;
        edgePositions.reserve(static_cast<size_t>(EDGE_SAMPLE_COUNT * 4 - 4));
        for (int coordinate = 0; coordinate <= FAR_TERRAIN_TILE_EDGE;
             coordinate += SHARED_WATER_EDGE_STEP) {
            edgePositions.push_back({mesh->originX, mesh->originZ + coordinate});
            edgePositions.push_back(
                {mesh->originX + FAR_TERRAIN_TILE_EDGE, mesh->originZ + coordinate});
            if (coordinate != 0 && coordinate != FAR_TERRAIN_TILE_EDGE) {
                edgePositions.push_back({mesh->originX + coordinate, mesh->originZ});
                edgePositions.push_back(
                    {mesh->originX + coordinate, mesh->originZ + FAR_TERRAIN_TILE_EDGE});
            }
        }
        preloadCanonicalWater(edgePositions);
    } else if (refineSharedWaterBoundary && source.geometryGrid) {
        constexpr int EDGE_SAMPLE_COUNT = FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP + 1;
        const auto retainStrip = [&](int64_t originX, int64_t originZ, int sampleWidth,
                                     int sampleHeight) {
            std::vector<FarTerrainGeometrySample> strip(
                static_cast<size_t>(sampleWidth * sampleHeight));
            source.geometryGrid(originX, originZ, SHARED_WATER_EDGE_STEP, SHARED_WATER_EDGE_STEP,
                                sampleWidth, sampleHeight, worldgen::SurfaceFootprint::BLOCK_1,
                                strip);
            for (int z = 0; z < sampleHeight; ++z) {
                for (int x = 0; x < sampleWidth; ++x) {
                    const ColumnPos position{
                        originX + static_cast<int64_t>(x * SHARED_WATER_EDGE_STEP),
                        originZ + static_cast<int64_t>(z * SHARED_WATER_EDGE_STEP),
                    };
                    canonicalWaterSamples.emplace(position,
                                                  strip[static_cast<size_t>(z * sampleWidth + x)]);
                }
            }
        };
        retainStrip(mesh->originX, mesh->originZ, 1, EDGE_SAMPLE_COUNT);
        retainStrip(mesh->originX + FAR_TERRAIN_TILE_EDGE, mesh->originZ, 1, EDGE_SAMPLE_COUNT);
        // The vertical strips own all four corners. Horizontal strips retain
        // only their interior so every face coordinate is sampled once.
        retainStrip(mesh->originX + SHARED_WATER_EDGE_STEP, mesh->originZ, EDGE_SAMPLE_COUNT - 2,
                    1);
        retainStrip(mesh->originX + SHARED_WATER_EDGE_STEP, mesh->originZ + FAR_TERRAIN_TILE_EDGE,
                    EDGE_SAMPLE_COUNT - 2, 1);
    }
    auto sharedWaterSample = [&](int localX, int localZ) -> const FarTerrainGeometrySample& {
        const ColumnPos key{mesh->originX + localX, mesh->originZ + localZ};
        auto found = canonicalWaterSamples.find(key);
        if (found != canonicalWaterSamples.end()) return found->second;
        canonicalWaterGeometry(key.x, key.z);
        return canonicalWaterSamples.find(key)->second;
    };
    auto probeWaterBoundary = [&](WaterEdge edge) {
        bool sawWet = false;
        bool sawDry = false;
        std::optional<FarWaterAuthority> previousWet;
        bool authorityDiscontinuity = false;
        for (int coordinate = 0; coordinate <= FAR_TERRAIN_TILE_EDGE;
             coordinate += SHARED_WATER_EDGE_STEP) {
            const int localX = edge == WEST ? 0 : edge == EAST ? FAR_TERRAIN_TILE_EDGE : coordinate;
            const int localZ = edge == NORTH   ? 0
                               : edge == SOUTH ? FAR_TERRAIN_TILE_EDGE
                                               : coordinate;
            const FarTerrainGeometrySample& sample = sharedWaterSample(localX, localZ);
            const bool wet = sampleIsWet(sample);
            sawWet = sawWet || wet;
            sawDry = sawDry || !wet;
            const FarWaterAuthority authority = waterAuthority(sample, wet);
            if (wet && previousWet &&
                !waterAuthoritiesConnect(*previousWet, authority,
                                         static_cast<float>(SHARED_WATER_EDGE_STEP))) {
                authorityDiscontinuity = true;
            }
            previousWet = wet ? std::optional<FarWaterAuthority>{authority} : std::nullopt;
            hasChannelFeature = hasChannelFeature || sample.river || sample.delta;
            hasLakeFeature = hasLakeFeature || sample.lake;
        }
        refineWaterBoundary[edge] = (sawWet && sawDry) || authorityDiscontinuity;
        hasWaterBoundary = hasWaterBoundary || refineWaterBoundary[edge];
    };
    if (refineSharedWaterBoundary) {
        probeWaterBoundary(WEST);
        probeWaterBoundary(EAST);
        probeWaterBoundary(NORTH);
        probeWaterBoundary(SOUTH);
    }
    if (refineSharedWaterBoundary && source.geometryPoints) {
        std::vector<ColumnPos> boundaryDecisionPoints;
        boundaryDecisionPoints.reserve(
            static_cast<size_t>((FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP) * 8));
        const auto appendBoundaryDecisionLine = [&](WaterEdge edge) {
            if (!refineWaterBoundary[edge]) return;
            for (int coordinate = 0; coordinate < FAR_TERRAIN_TILE_EDGE;
                 coordinate += SHARED_WATER_EDGE_STEP) {
                if (edge == WEST || edge == EAST) {
                    const int innerX = edge == WEST ? step : FAR_TERRAIN_TILE_EDGE - step;
                    const int centerX = edge == WEST ? step / 2 : FAR_TERRAIN_TILE_EDGE - step / 2;
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + innerX, mesh->originZ + coordinate});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + innerX,
                         mesh->originZ + coordinate + SHARED_WATER_EDGE_STEP});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + centerX,
                         mesh->originZ + coordinate + SHARED_WATER_EDGE_STEP / 2});
                } else {
                    const int innerZ = edge == NORTH ? step : FAR_TERRAIN_TILE_EDGE - step;
                    const int centerZ = edge == NORTH ? step / 2 : FAR_TERRAIN_TILE_EDGE - step / 2;
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + coordinate, mesh->originZ + innerZ});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + coordinate + SHARED_WATER_EDGE_STEP,
                         mesh->originZ + innerZ});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + coordinate + SHARED_WATER_EDGE_STEP / 2,
                         mesh->originZ + centerZ});
                }
            }
        };
        appendBoundaryDecisionLine(WEST);
        appendBoundaryDecisionLine(EAST);
        appendBoundaryDecisionLine(NORTH);
        appendBoundaryDecisionLine(SOUTH);
        preloadCanonicalWater(boundaryDecisionPoints);
    }
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            const FarTerrainGeometrySample& sample = sampleAt(x, z);
            hasChannelFeature = hasChannelFeature || sample.river || sample.delta;
            hasLakeFeature = hasLakeFeature || sample.lake;
            if (x + 1 < sampleEdge) {
                const FarTerrainGeometrySample& east = sampleAt(x + 1, z);
                maximumSampleSlope = std::max(
                    maximumSampleSlope, std::abs(east.terrainHeight - sample.terrainHeight) / step);
                const bool sampleWet = sampleIsWet(sample);
                const bool eastWet = sampleIsWet(east);
                hasWaterBoundary = hasWaterBoundary || sampleWet != eastWet ||
                                   (sampleWet && eastWet &&
                                    !waterAuthoritiesConnect(waterAuthority(sample, true),
                                                             waterAuthority(east, true),
                                                             static_cast<float>(step)));
            }
            if (z + 1 < sampleEdge) {
                const FarTerrainGeometrySample& south = sampleAt(x, z + 1);
                maximumSampleSlope =
                    std::max(maximumSampleSlope,
                             std::abs(south.terrainHeight - sample.terrainHeight) / step);
                const bool sampleWet = sampleIsWet(sample);
                const bool southWet = sampleIsWet(south);
                hasWaterBoundary = hasWaterBoundary || sampleWet != southWet ||
                                   (sampleWet && southWet &&
                                    !waterAuthoritiesConnect(waterAuthority(sample, true),
                                                             waterAuthority(south, true),
                                                             static_cast<float>(step)));
            }
        }
    }
    const float terrainComplexity =
        static_cast<float>(std::clamp(maximumSampleSlope / 1.5, 0.0, 1.0));
    const float hydrologyComplexity = hasWaterBoundary    ? 1.0F
                                      : hasChannelFeature ? 0.85F
                                      : hasLakeFeature    ? 0.45F
                                                          : 0.0F;
    mesh->complexity = std::max(terrainComplexity, hydrologyComplexity);

    if (source.geometryPoints && step >= 8 && key.step != FarTerrainStep::THIRTY_TWO) {
        std::vector<ColumnPos> centers;
        centers.reserve(static_cast<size_t>(cellEdge * cellEdge));
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                const bool anyWet =
                    key.step == FarTerrainStep::THIRTY_TWO || sampleIsWet(sampleAt(x, z)) ||
                    sampleIsWet(sampleAt(x + 1, z)) || sampleIsWet(sampleAt(x + 1, z + 1)) ||
                    sampleIsWet(sampleAt(x, z + 1));
                if (!anyWet) continue;
                centers.push_back({mesh->originX + static_cast<int64_t>(x * step + step / 2),
                                   mesh->originZ + static_cast<int64_t>(z * step + step / 2)});
            }
        }
        preloadCanonicalWater(centers);
    }
    std::vector<FarCell> cells(static_cast<size_t>(cellEdge * cellEdge));
    auto cellAt = [&](int x, int z) -> FarCell& {
        return cells[static_cast<size_t>(z * cellEdge + x)];
    };
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const std::array<FarSurfaceSample*, 4> cornerSamples = {
                &surfaceAt(x, z), &surfaceAt(x + 1, z), &surfaceAt(x + 1, z + 1),
                &surfaceAt(x, z + 1)};
            const ResolvedFarTerrainCellBounds bounds =
                source.cellBoundsGrid ? authoritativeCellBounds(authoritativeBoundsAt(x, z))
                                      : fallbackCellBounds(cornerSamples, key.step);
            cell.terrain.fill(bounds.top);
            cell.skirtBottom = bounds.skirtBottom;
            cell.maximumTerrain = bounds.maximum;
            updateSurfaceYBounds(*mesh, bounds.minimum);
            updateSurfaceYBounds(*mesh, bounds.maximum);
            const int patchX = x * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            const int patchZ = z * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
            FarTerrainBounds& patch =
                mesh->occluderPatches[static_cast<size_t>(patchZ * PATCHES_PER_EDGE + patchX)];
            patch.minY = std::min(patch.minY, bounds.minimum);
            patch.maxY = std::max(patch.maxY, bounds.maximum);
            cell.flat = true;
            const int64_t worldX = mesh->originX + static_cast<int64_t>(x * step);
            const int64_t worldZ = mesh->originZ + static_cast<int64_t>(z * step);
            const auto& palette = surfaceAt(x, z).materialPalette;
            const int64_t materialX = worldX + step / 2;
            const int64_t materialZ = worldZ + step / 2;
            const double rank = source.materialRank ? source.materialRank(materialX, materialZ)
                                                    : materialRank(materialX, materialZ);
            cell.material = worldgen::surface_material::selectMaterial(palette, rank);
            if (cell.material == BlockType::AIR) cell.material = BlockType::STONE;
            if (key.step != FarTerrainStep::THIRTY_TWO) {
                for (size_t corner = 0; corner < cornerSamples.size(); ++corner) {
                    const FarTerrainGeometrySample& geometry = cornerSamples[corner]->geometry;
                    cell.waterSurface[corner] = vertexHeight(geometry.waterSurface);
                    const bool wet = sampleIsWet(geometry);
                    cell.waterAuthority[corner] = waterAuthority(geometry, wet);
                    if (wet) cell.waterMask |= static_cast<uint8_t>(1U << corner);
                }
                const bool cornerWater = cell.waterMask != 0;
                cell.flatWater = cell.waterMask == 0x0FU &&
                                 cell.waterSurface[0] == cell.waterSurface[1] &&
                                 cell.waterSurface[0] == cell.waterSurface[2] &&
                                 cell.waterSurface[0] == cell.waterSurface[3];
                cell.waterHeight =
                    *std::max_element(cell.waterSurface.begin(), cell.waterSurface.end());
                constexpr std::array<std::array<float, 2>, 4> CORNER_POSITIONS = {{
                    {{0.0F, 0.0F}},
                    {{1.0F, 0.0F}},
                    {{1.0F, 1.0F}},
                    {{0.0F, 1.0F}},
                }};
                for (size_t first = 0; first < cell.waterAuthority.size(); ++first) {
                    if ((cell.waterMask & (1U << first)) == 0) continue;
                    for (size_t second = first + 1; second < cell.waterAuthority.size(); ++second) {
                        if ((cell.waterMask & (1U << second)) == 0) continue;
                        const float distance =
                            static_cast<float>(step) *
                            std::hypot(CORNER_POSITIONS[first][0] - CORNER_POSITIONS[second][0],
                                       CORNER_POSITIONS[first][1] - CORNER_POSITIONS[second][1]);
                        if (!waterAuthoritiesConnect(cell.waterAuthority[first],
                                                     cell.waterAuthority[second], distance)) {
                            cell.discontinuousWater = true;
                        }
                    }
                }
                if (!exactNearWater && cornerWater &&
                    (cell.waterMask != 0x0FU || cell.discontinuousWater || step >= 8)) {
                    const int64_t centerX = worldX + step / 2;
                    const int64_t centerZ = worldZ + step / 2;
                    const FarTerrainGeometrySample center =
                        canonicalWaterGeometry(centerX, centerZ);
                    cell.centerWet = sampleIsWet(center);
                    cell.centerWaterHeight = vertexHeight(center.waterSurface);
                    cell.centerWaterAuthority = waterAuthority(center, cell.centerWet);
                    if (!cell.centerWet && cell.waterMask == 0x0FU) {
                        cell.discontinuousWater = true;
                    } else if (cell.centerWet) {
                        const float centerDistance = static_cast<float>(step) * 0.70710678F;
                        for (size_t corner = 0; corner < cell.waterAuthority.size(); ++corner) {
                            if ((cell.waterMask & (1U << corner)) != 0 &&
                                !waterAuthoritiesConnect(cell.centerWaterAuthority,
                                                         cell.waterAuthority[corner],
                                                         centerDistance)) {
                                cell.discontinuousWater = true;
                            }
                        }
                    }
                }
                cell.water = cornerWater;
            }
        }
    }

    // Coarse parents resolve only ambiguous water rectangles at the
    // shoreline page's canonical two- or four-block spacing. Gather every
    // corner and center first, then let the generator group the immutable
    // point set by basin in one call. This preserves exact body identity and
    // shared tile faces without the former thousands of scalar samples.
    if (source.geometryPoints && step >= 8 && key.step != FarTerrainStep::THIRTY_TWO) {
        std::vector<ColumnPos> contourPositions;
        const auto appendContourRectangle = [&](int x0, int z0, int x1, int z1, int resolution) {
            for (int fineZ0 = z0; fineZ0 < z1; fineZ0 += resolution) {
                const int fineZ1 = std::min(fineZ0 + resolution, z1);
                for (int fineX0 = x0; fineX0 < x1; fineX0 += resolution) {
                    const int fineX1 = std::min(fineX0 + resolution, x1);
                    contourPositions.push_back({mesh->originX + fineX0, mesh->originZ + fineZ0});
                    contourPositions.push_back({mesh->originX + fineX1, mesh->originZ + fineZ0});
                    contourPositions.push_back({mesh->originX + fineX1, mesh->originZ + fineZ1});
                    contourPositions.push_back({mesh->originX + fineX0, mesh->originZ + fineZ1});
                    contourPositions.push_back({mesh->originX + (fineX0 + fineX1) / 2,
                                                mesh->originZ + (fineZ0 + fineZ1) / 2});
                }
            }
        };
        const auto sharedRectangleNeedsResolution = [&](int x0, int z0, int x1, int z1) {
            constexpr std::array<std::array<int, 2>, 4> CORNERS = {
                {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}}};
            std::array<FarWaterAuthority, 4> authorities{};
            std::array<float, 4> heights{};
            bool allWet = true;
            bool anyWet = false;
            for (size_t index = 0; index < CORNERS.size(); ++index) {
                const int localX = CORNERS[index][0] == 0 ? x0 : x1;
                const int localZ = CORNERS[index][1] == 0 ? z0 : z1;
                const FarTerrainGeometrySample sample =
                    canonicalWaterGeometry(mesh->originX + localX, mesh->originZ + localZ);
                const bool wet = sampleIsWet(sample);
                authorities[index] = waterAuthority(sample, wet);
                heights[index] = vertexHeight(sample.waterSurface);
                allWet = allWet && wet;
                anyWet = anyWet || wet;
            }
            const FarTerrainGeometrySample center = canonicalWaterGeometry(
                mesh->originX + (x0 + x1) / 2, mesh->originZ + (z0 + z1) / 2);
            const bool centerWet = sampleIsWet(center);
            anyWet = anyWet || centerWet;
            if (!anyWet) return false;
            FarWaterAuthority target = waterAuthority(center, centerWet);
            if (!centerWet) {
                const auto firstWet = std::find_if(authorities.begin(), authorities.end(),
                                                   [](const FarWaterAuthority& authority) {
                                                       return authority.kind != FarWaterKind::NONE;
                                                   });
                if (firstWet != authorities.end()) target = *firstWet;
            }
            const float connectionSpan =
                std::hypot(static_cast<float>(x1 - x0), static_cast<float>(z1 - z0));
            const bool allOwned = allWet && std::all_of(authorities.begin(), authorities.end(),
                                                        [&](const FarWaterAuthority& authority) {
                                                            return waterAuthoritiesConnect(
                                                                authority, target, connectionSpan);
                                                        });
            const bool centerOwned =
                waterAuthoritiesConnect(waterAuthority(center, centerWet), target, connectionSpan);
            const bool flatWater =
                std::all_of(heights.begin() + 1, heights.end(),
                            [&](float height) { return height == heights.front(); });
            return !allOwned || !centerOwned || !flatWater;
        };
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                const FarCell& cell = cellAt(x, z);
                const bool sharedBoundary = (x == 0 && refineWaterBoundary[WEST]) ||
                                            (x == cellEdge - 1 && refineWaterBoundary[EAST]) ||
                                            (z == 0 && refineWaterBoundary[NORTH]) ||
                                            (z == cellEdge - 1 && refineWaterBoundary[SOUTH]);
                const bool ambiguous =
                    cell.water && (cell.waterMask != 0x0FU || cell.discontinuousWater);
                if (sharedBoundary || !ambiguous) continue;
                const int resolution = 2;
                appendContourRectangle(x * step, z * step, (x + 1) * step, (z + 1) * step,
                                       resolution);
            }
        }
        const auto appendBoundaryRectangles = [&](WaterEdge edge) {
            if (!refineWaterBoundary[edge]) return;
            const bool horizontal = edge == NORTH || edge == SOUTH;
            const int begin = horizontal && refineWaterBoundary[WEST] ? step : 0;
            const int end = horizontal && refineWaterBoundary[EAST] ? FAR_TERRAIN_TILE_EDGE - step
                                                                    : FAR_TERRAIN_TILE_EDGE;
            for (int coordinate = begin; coordinate < end; coordinate += SHARED_WATER_EDGE_STEP) {
                int x0 = coordinate;
                int z0 = coordinate;
                int x1 = coordinate + SHARED_WATER_EDGE_STEP;
                int z1 = coordinate + SHARED_WATER_EDGE_STEP;
                if (edge == WEST) {
                    x0 = 0;
                    x1 = step;
                } else if (edge == EAST) {
                    x0 = FAR_TERRAIN_TILE_EDGE - step;
                    x1 = FAR_TERRAIN_TILE_EDGE;
                } else if (edge == NORTH) {
                    z0 = 0;
                    z1 = step;
                } else {
                    z0 = FAR_TERRAIN_TILE_EDGE - step;
                    z1 = FAR_TERRAIN_TILE_EDGE;
                }
                if (sharedRectangleNeedsResolution(x0, z0, x1, z1)) {
                    appendContourRectangle(x0, z0, x1, z1, 1);
                }
            }
        };
        appendBoundaryRectangles(WEST);
        appendBoundaryRectangles(EAST);
        appendBoundaryRectangles(NORTH);
        appendBoundaryRectangles(SOUTH);
        preloadCanonicalWater(contourPositions);
    }

    // Step-8 and step-16 shoreline polygons can refine a coarse corner-to-
    // center edge one block at a time. Gather every possible integer probe for
    // those polygons before emission. The geometry callback then receives one
    // deduplicated basin batch instead of a succession of one-point batches.
    if (source.geometryPoints &&
        (key.step == FarTerrainStep::EIGHT || key.step == FarTerrainStep::SIXTEEN)) {
        std::vector<ColumnPos> edgeRefinementPositions;
        const auto appendSegment = [&](int startX, int startZ, int endX, int endZ) {
            const int subdivisions = std::max(std::abs(endX - startX), std::abs(endZ - startZ));
            for (int subdivision = 1; subdivision <= subdivisions; ++subdivision) {
                const double amount = static_cast<double>(subdivision) / subdivisions;
                edgeRefinementPositions.push_back(
                    {mesh->originX +
                         static_cast<int64_t>(std::llround(std::lerp(
                             static_cast<double>(startX), static_cast<double>(endX), amount))),
                     mesh->originZ +
                         static_cast<int64_t>(std::llround(std::lerp(
                             static_cast<double>(startZ), static_cast<double>(endZ), amount)))});
            }
        };
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                const FarCell& cell = cellAt(x, z);
                if (!cell.water || cell.waterMask == 0x0FU || cell.discontinuousWater) continue;
                const int x0 = x * step;
                const int z0 = z * step;
                const int x1 = (x + 1) * step;
                const int z1 = (z + 1) * step;
                const int centerX = x0 + step / 2;
                const int centerZ = z0 + step / 2;
                appendSegment(x0, z0, x0, z1);
                appendSegment(x0, z1, x1, z1);
                appendSegment(x1, z1, x1, z0);
                appendSegment(x1, z0, x0, z0);
                appendSegment(x0, z0, centerX, centerZ);
                appendSegment(x0, z1, centerX, centerZ);
                appendSegment(x1, z1, centerX, centerZ);
                appendSegment(x1, z0, centerX, centerZ);
            }
        }
        preloadCanonicalWater(edgeRefinementPositions);
    }

    std::vector<uint8_t> merged(static_cast<size_t>(cellEdge * cellEdge), 0);
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const size_t cellIndex = static_cast<size_t>(z * cellEdge + x);
            if (merged[cellIndex] != 0) continue;
            if (!cell.flat) {
                const float x0 = static_cast<float>(x * step);
                const float z0 = static_cast<float>(z * step);
                pushTerrainTop(*mesh, cell.material, x0, z0, x0 + step, z0 + step, cell.terrain[0],
                               cell.terrain[1], cell.terrain[2], cell.terrain[3]);
                merged[cellIndex] = 1;
                continue;
            }
            int width = 1;
            while (x + width < cellEdge &&
                   merged[static_cast<size_t>(z * cellEdge + x + width)] == 0 &&
                   sameFlatTerrain(cell, cellAt(x + width, z))) {
                ++width;
            }
            int depth = 1;
            while (z + depth < cellEdge) {
                bool compatible = true;
                for (int offset = 0; offset < width; ++offset) {
                    const size_t candidate =
                        static_cast<size_t>((z + depth) * cellEdge + x + offset);
                    if (merged[candidate] != 0 ||
                        !sameFlatTerrain(cell, cellAt(x + offset, z + depth))) {
                        compatible = false;
                        break;
                    }
                }
                if (!compatible) break;
                ++depth;
            }
            for (int dz = 0; dz < depth; ++dz) {
                for (int dx = 0; dx < width; ++dx) {
                    merged[static_cast<size_t>((z + dz) * cellEdge + x + dx)] = 1;
                }
            }
            mesh->mergedTerrainCellCount += static_cast<uint32_t>(width * depth);
            const float x0 = static_cast<float>(x * step);
            const float z0 = static_cast<float>(z * step);
            const float x1 = static_cast<float>((x + width) * step);
            const float z1 = static_cast<float>((z + depth) * step);
            pushTerrainTop(*mesh, cell.material, x0, z0, x1, z1, cell.terrain[0], cell.terrain[0],
                           cell.terrain[0], cell.terrain[0]);
        }
    }

    // Each far cell is a larger voxel column, not a height-field triangle.
    // Emit cardinal risers wherever neighboring quantized columns differ so
    // lighting, silhouettes, and materials retain the block language at
    // every LOD. Only the higher column owns the shared face.
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            const FarCell& cell = cellAt(x, z);
            const float x0 = static_cast<float>(x * step);
            const float x1 = x0 + step;
            const float z0 = static_cast<float>(z * step);
            const float z1 = z0 + step;
            if (x + 1 < cellEdge) {
                const FarCell& east = cellAt(x + 1, z);
                if (cell.terrain[0] > east.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::PLUS_X, cell.material, x1, z1, x1, z0,
                                     east.terrain[0], cell.terrain[0]);
                } else if (east.terrain[0] > cell.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::MINUS_X, east.material, x1, z0, x1, z1,
                                     cell.terrain[0], east.terrain[0]);
                }
            }
            if (z + 1 < cellEdge) {
                const FarCell& south = cellAt(x, z + 1);
                if (cell.terrain[0] > south.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::PLUS_Z, cell.material, x0, z1, x1, z1,
                                     south.terrain[0], cell.terrain[0]);
                } else if (south.terrain[0] > cell.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::MINUS_Z, south.material, x1, z1, x0, z1,
                                     cell.terrain[0], south.terrain[0]);
                }
            }
        }
    }

    std::unordered_map<ColumnPos, FarSurfaceSample> exteriorTerrainSamples;
    const auto exteriorCellHeight = [&](int cellX, int cellZ) {
        if (source.cellBoundsGrid) {
            return authoritativeCellBounds(authoritativeBoundsAt(cellX, cellZ)).top;
        }
        std::array<FarSurfaceSample*, 4> corners{};
        constexpr std::array<std::array<int, 2>, 4> OFFSETS = {
            {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}}};
        for (size_t corner = 0; corner < OFFSETS.size(); ++corner) {
            const ColumnPos position{
                mesh->originX + static_cast<int64_t>((cellX + OFFSETS[corner][0]) * step),
                mesh->originZ + static_cast<int64_t>((cellZ + OFFSETS[corner][1]) * step),
            };
            auto [entry, inserted] = exteriorTerrainSamples.try_emplace(position);
            if (inserted) entry->second = source.sample(position.x, position.z, footprint);
            corners[corner] = &entry->second;
        }
        return terrainCellHeight(corners, key.step);
    };
    for (int coordinate = 0; coordinate < cellEdge; ++coordinate) {
        const float start = static_cast<float>(coordinate * step);
        const float end = start + step;

        const FarCell& west = cellAt(0, coordinate);
        const float westNeighbor = exteriorCellHeight(-1, coordinate);
        if (west.terrain[0] > westNeighbor) {
            pushTerrainRiser(*mesh, FaceNormal::MINUS_X, west.material, 0.0F, start, 0.0F, end,
                             westNeighbor, west.terrain[0]);
        }

        const FarCell& east = cellAt(cellEdge - 1, coordinate);
        const float eastNeighbor = exteriorCellHeight(cellEdge, coordinate);
        if (east.terrain[0] > eastNeighbor) {
            pushTerrainRiser(*mesh, FaceNormal::PLUS_X, east.material,
                             static_cast<float>(FAR_TERRAIN_TILE_EDGE), end,
                             static_cast<float>(FAR_TERRAIN_TILE_EDGE), start, eastNeighbor,
                             east.terrain[0]);
        }

        const FarCell& north = cellAt(coordinate, 0);
        const float northNeighbor = exteriorCellHeight(coordinate, -1);
        if (north.terrain[0] > northNeighbor) {
            pushTerrainRiser(*mesh, FaceNormal::MINUS_Z, north.material, end, 0.0F, start, 0.0F,
                             northNeighbor, north.terrain[0]);
        }

        const FarCell& south = cellAt(coordinate, cellEdge - 1);
        const float southNeighbor = exteriorCellHeight(coordinate, cellEdge);
        if (south.terrain[0] > southNeighbor) {
            pushTerrainRiser(*mesh, FaceNormal::PLUS_Z, south.material, start,
                             static_cast<float>(FAR_TERRAIN_TILE_EDGE), end,
                             static_cast<float>(FAR_TERRAIN_TILE_EDGE), southNeighbor,
                             south.terrain[0]);
        }
    }

    auto addEdgeSkirt = [&](int cellX, int cellZ, int x, int z, int nextX, int nextZ,
                            FaceNormal face) {
        const FarCell& materialCell = cellAt(cellX, cellZ);
        const float x0 = static_cast<float>(x * step);
        const float z0 = static_cast<float>(z * step);
        const float x1 = static_cast<float>(nextX * step);
        const float z1 = static_cast<float>(nextZ * step);
        pushSkirt(*mesh, face, materialCell.material, x0, z0, x1, z1, materialCell.terrain[0],
                  materialCell.terrain[0], materialCell.skirtBottom, materialCell.skirtBottom);
    };
    for (int coordinate = 0; coordinate < cellEdge; ++coordinate) {
        addEdgeSkirt(0, coordinate, 0, coordinate, 0, coordinate + 1, FaceNormal::MINUS_X);
        addEdgeSkirt(cellEdge - 1, coordinate, cellEdge, coordinate + 1, cellEdge, coordinate,
                     FaceNormal::PLUS_X);
        addEdgeSkirt(coordinate, 0, coordinate + 1, 0, coordinate, 0, FaceNormal::MINUS_Z);
        addEdgeSkirt(coordinate, cellEdge - 1, coordinate, cellEdge, coordinate + 1, cellEdge,
                     FaceNormal::PLUS_Z);
    }

    if (source.canopies) {
        const std::vector<FarCanopy> canopies =
            source.canopies(mesh->originX, mesh->originZ, mesh->originX + FAR_TERRAIN_TILE_EDGE,
                            mesh->originZ + FAR_TERRAIN_TILE_EDGE, key.step);
        for (const FarCanopy& canopy : canopies) {
            // The anchor's half-open tile owns the complete layered crown.
            // It may cross the tile face, matching exact tree ownership
            // without duplicate coplanar foliage.
            if (canopy.x < mesh->originX || canopy.x >= mesh->originX + FAR_TERRAIN_TILE_EDGE ||
                canopy.z < mesh->originZ || canopy.z >= mesh->originZ + FAR_TERRAIN_TILE_EDGE) {
                continue;
            }
            // Crown anchors use one canonical terrain footprint within each
            // hierarchy. The collector owns thinning, so every surviving
            // species remains identical while terrain changes tier.
            float ground = 0.0F;
            // Every silhouette stands on its resident voxel. This keeps
            // aggregate crowns grounded in the displayed parent and avoids
            // a cold scalar basin query for each forest cluster.
            const int canopyCellX = static_cast<int>((canopy.x - mesh->originX) / step);
            const int canopyCellZ = static_cast<int>((canopy.z - mesh->originZ) / step);
            // The canopy collector already applies block-resolution habitat,
            // substrate, and root-water rules at the anchor. A coarse cell's
            // water-topology bit means only that water occurs somewhere in
            // its 32 by 32 footprint. Suppressing every accepted anchor in
            // that cell creates square forest holes beside narrow channels.
            // Ground the accepted silhouette on the displayed parent voxel;
            // its trunk then remains connected even where the canonical
            // shoreline crosses the same coarse cell.
            ground = cellAt(canopyCellX, canopyCellZ).terrain[0];
            const float localAnchorX = static_cast<float>(canopy.x - mesh->originX);
            const float localAnchorZ = static_cast<float>(canopy.z - mesh->originZ);
            const feature_generation::TreeSpecies species = canopy.species;
            if (species == feature_generation::TreeSpecies::FALLEN_LOG) {
                const float length = std::max(1.0F, static_cast<float>(canopy.formExtent));
                const float directionX = canopy.formX == 0 ? 0.0F : 1.0F;
                const float directionZ = canopy.formZ == 0 ? 0.0F : 1.0F;
                const float centerX = localAnchorX + 0.5F + directionX * (length - 1.0F) * 0.5F;
                const float centerZ = localAnchorZ + 0.5F + directionZ * (length - 1.0F) * 0.5F;
                const float halfWidthX = directionX > 0.0F ? length * 0.5F : 0.5F;
                const float halfWidthZ = directionZ > 0.0F ? length * 0.5F : 0.5F;
                if (canopy.logBlock != BlockType::AIR) {
                    pushCanopyPrism(*mesh, canopy.logBlock, centerX, centerZ, halfWidthX,
                                    halfWidthZ, ground, ground + 1.0F, true);
                }
                ++mesh->canopyAnchorCount;
                continue;
            }
            const float canopyCenterX = localAnchorX + canopy.canopyOffsetX;
            const float canopyCenterZ = localAnchorZ + canopy.canopyOffsetZ;
            const float canopyBottom =
                ground + static_cast<float>(canopy.canopyMinimumY - canopy.baseY);
            const float canopyTop =
                ground + static_cast<float>(canopy.canopyMaximumY + 1 - canopy.baseY);
            const float canopyRadius =
                std::max(1.0F, static_cast<float>(canopy.canopyRadius) + 0.5F);
            const float trunkTop = canopyTrunkTop(canopy.leafBlock, canopyBottom, canopyTop);
            if (canopy.logBlock != BlockType::AIR && trunkTop > ground) {
                const float trunkHalfWidth = canopyTrunkHalfWidth(canopy.logBlock);
                pushCanopyPrism(*mesh, canopy.logBlock, localAnchorX + 0.5F, localAnchorZ + 0.5F,
                                trunkHalfWidth, trunkHalfWidth, ground, trunkTop, false);
            }
            if (canopy.leafBlock != BlockType::AIR) {
                pushSpeciesCanopy(*mesh, canopy, canopyCenterX + 0.5F, canopyCenterZ + 0.5F,
                                  canopyRadius, canopyBottom, canopyTop);
            }
            ++mesh->canopyAnchorCount;
        }
    }
    mesh->opaqueIndexCount = static_cast<uint32_t>(mesh->indices.size());

    // Lake outlets retain the downstream body's standing surface and add a
    // separately owned falling prism. The half-open tile containing the
    // outlet anchor owns the complete narrow prism, even when it crosses a
    // tile face, so no adjacent tile duplicates a coplanar waterfall wall.
    for (int z = 0; z < sampleEdge - 1; ++z) {
        for (int x = 0; x < sampleEdge - 1; ++x) {
            const FarTerrainGeometrySample& sample = sampleAt(x, z);
            if (!sample.waterfall || !sample.waterfallAnchor ||
                sample.waterfallTop < sample.waterfallBottom + 0.5) {
                continue;
            }
            pushWaterfallPrism(*mesh, static_cast<float>(x * step), static_cast<float>(z * step),
                               sample);
            mesh->complexity = 1.0F;
        }
    }

    if (exactNearWater) {
        // The first two fallback tiers meet exact cubic terrain while nearby
        // sections stream. Probe a lattice whose covering radius is below the
        // minimum routed-channel half width, then request block authority only
        // for cells containing a shoreline, channel, or level transition.
        const int sentinelSpacing = 2;
        const int sentinelOffset = step == 2 ? 1 : 0;
        const int sentinelEdge = step == 2 ? cellEdge : FAR_TERRAIN_TILE_EDGE / 2 + 1;
        std::vector<FarTerrainGeometrySample> sentinels(
            static_cast<size_t>(sentinelEdge * sentinelEdge));
        if (source.geometryGrid) {
            source.geometryGrid(mesh->originX + sentinelOffset, mesh->originZ + sentinelOffset,
                                sentinelSpacing, sentinelSpacing, sentinelEdge, sentinelEdge,
                                worldgen::SurfaceFootprint::BLOCK_1, sentinels);
        } else {
            for (int z = 0; z < sentinelEdge; ++z) {
                for (int x = 0; x < sentinelEdge; ++x) {
                    sentinels[static_cast<size_t>(z * sentinelEdge + x)] =
                        source
                            .sample(mesh->originX + sentinelOffset + x * sentinelSpacing,
                                    mesh->originZ + sentinelOffset + z * sentinelSpacing,
                                    worldgen::SurfaceFootprint::BLOCK_1)
                            .geometry;
                }
            }
        }
        const auto sentinelAt = [&](int x, int z) -> const FarTerrainGeometrySample& {
            return sentinels[static_cast<size_t>(z * sentinelEdge + x)];
        };
        const auto exactSentinelAt = [&](int localX,
                                         int localZ) -> const FarTerrainGeometrySample* {
            const int sentinelX = localX - sentinelOffset;
            const int sentinelZ = localZ - sentinelOffset;
            if (sentinelX < 0 || sentinelZ < 0 || (sentinelX & 1) != 0 || (sentinelZ & 1) != 0) {
                return nullptr;
            }
            const int sampleX = sentinelX / sentinelSpacing;
            const int sampleZ = sentinelZ / sentinelSpacing;
            if (sampleX >= sentinelEdge || sampleZ >= sentinelEdge) return nullptr;
            return &sentinelAt(sampleX, sampleZ);
        };
        const auto sameAuthorityState = [&](const FarWaterAuthority& first,
                                            const FarWaterAuthority& second) {
            if (first.kind == FarWaterKind::NONE || second.kind == FarWaterKind::NONE) {
                return first.kind == second.kind;
            }
            return first.kind == second.kind && first.bodyId == second.bodyId &&
                   first.height == second.height && first.delta == second.delta;
        };

        std::vector<uint32_t> ambiguousCells;
        ambiguousCells.reserve(static_cast<size_t>(cellEdge * cellEdge));
        std::vector<FarWaterAuthority> uniformAuthorities(static_cast<size_t>(cellEdge * cellEdge));
        // Zero is ambiguous, one is uniformly dry, and two is uniformly wet.
        std::vector<uint8_t> uniformStates(static_cast<size_t>(cellEdge * cellEdge), 0);
        for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
            for (int cellX = 0; cellX < cellEdge; ++cellX) {
                std::array<FarWaterAuthority, 9> authorities{};
                size_t authorityCount = 0;
                if (step == 2) {
                    const FarCell& cell = cellAt(cellX, cellZ);
                    for (const FarWaterAuthority authority : cell.waterAuthority) {
                        authorities[authorityCount++] = authority;
                    }
                    const FarTerrainGeometrySample& center = sentinelAt(cellX, cellZ);
                    authorities[authorityCount++] = waterAuthority(center, sampleIsWet(center));
                } else {
                    for (int sampleZ = 0; sampleZ < 3; ++sampleZ) {
                        for (int sampleX = 0; sampleX < 3; ++sampleX) {
                            const FarTerrainGeometrySample& sample =
                                sentinelAt(cellX * 2 + sampleX, cellZ * 2 + sampleZ);
                            authorities[authorityCount++] =
                                waterAuthority(sample, sampleIsWet(sample));
                        }
                    }
                }
                const FarWaterAuthority reference = authorities.front();
                const bool uniform =
                    std::all_of(authorities.begin() + 1, authorities.begin() + authorityCount,
                                [&](const FarWaterAuthority& authority) {
                                    return sameAuthorityState(reference, authority);
                                });
                if (uniform) {
                    const size_t cellIndex = static_cast<size_t>(cellZ * cellEdge + cellX);
                    uniformAuthorities[cellIndex] = reference;
                    uniformStates[cellIndex] =
                        reference.kind == FarWaterKind::NONE ? uint8_t{1} : uint8_t{2};
                    continue;
                }
                ambiguousCells.push_back(static_cast<uint32_t>(cellZ * cellEdge + cellX));
            }
        }
        std::vector<uint8_t> emittedUniform(static_cast<size_t>(cellEdge * cellEdge), 0);
        const auto boundaryNeedsSharedResolution = [&](int cellX, int cellZ) {
            return (cellX == 0 && refineWaterBoundary[WEST]) ||
                   (cellX == cellEdge - 1 && refineWaterBoundary[EAST]) ||
                   (cellZ == 0 && refineWaterBoundary[NORTH]) ||
                   (cellZ == cellEdge - 1 && refineWaterBoundary[SOUTH]);
        };
        // A mixed-LOD tile face uses block-resolution rectangles only where
        // the shared two-block probes found a shoreline or authority change.
        // Uniform open water remains one greedily merged quad.
        if (step == 4) {
            for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
                for (int cellX = 0; cellX < cellEdge; ++cellX) {
                    const size_t cellIndex = static_cast<size_t>(cellZ * cellEdge + cellX);
                    if (uniformStates[cellIndex] != 2 ||
                        !boundaryNeedsSharedResolution(cellX, cellZ)) {
                        continue;
                    }
                    const FarWaterAuthority authority = uniformAuthorities[cellIndex];
                    for (int offsetZ = 0; offsetZ < step; ++offsetZ) {
                        for (int offsetX = 0; offsetX < step; ++offsetX) {
                            const float x0 = static_cast<float>(cellX * step + offsetX);
                            const float z0 = static_cast<float>(cellZ * step + offsetZ);
                            pushWaterTop(*mesh, x0, z0, x0 + 1.0F, z0 + 1.0F, authority.height);
                        }
                    }
                    emittedUniform[cellIndex] = 1;
                }
            }
        }
        for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
            for (int cellX = 0; cellX < cellEdge; ++cellX) {
                const size_t cellIndex = static_cast<size_t>(cellZ * cellEdge + cellX);
                if (uniformStates[cellIndex] != 2 || emittedUniform[cellIndex] != 0) continue;
                const FarWaterAuthority authority = uniformAuthorities[cellIndex];
                int width = 1;
                while (cellX + width < cellEdge) {
                    const size_t candidate = static_cast<size_t>(cellZ * cellEdge + cellX + width);
                    if (uniformStates[candidate] != 2 || emittedUniform[candidate] != 0 ||
                        boundaryNeedsSharedResolution(cellX + width, cellZ) ||
                        !sameAuthorityState(authority, uniformAuthorities[candidate])) {
                        break;
                    }
                    ++width;
                }
                int depth = 1;
                while (cellZ + depth < cellEdge) {
                    bool compatible = true;
                    for (int offset = 0; offset < width; ++offset) {
                        const size_t candidate =
                            static_cast<size_t>((cellZ + depth) * cellEdge + cellX + offset);
                        if (uniformStates[candidate] != 2 || emittedUniform[candidate] != 0 ||
                            boundaryNeedsSharedResolution(cellX + offset, cellZ + depth) ||
                            !sameAuthorityState(authority, uniformAuthorities[candidate])) {
                            compatible = false;
                            break;
                        }
                    }
                    if (!compatible) break;
                    ++depth;
                }
                for (int offsetZ = 0; offsetZ < depth; ++offsetZ) {
                    for (int offsetX = 0; offsetX < width; ++offsetX) {
                        emittedUniform[static_cast<size_t>((cellZ + offsetZ) * cellEdge + cellX +
                                                           offsetX)] = 1;
                    }
                }
                pushWaterTop(*mesh, static_cast<float>(cellX * step),
                             static_cast<float>(cellZ * step),
                             static_cast<float>((cellX + width) * step),
                             static_cast<float>((cellZ + depth) * step), authority.height);
            }
        }
        const size_t sentinelPointsPerCell = static_cast<size_t>(step / sentinelSpacing) *
                                             static_cast<size_t>(step / sentinelSpacing);
        const size_t exactPointCount =
            ambiguousCells.size() *
            (static_cast<size_t>(step) * static_cast<size_t>(step) - sentinelPointsPerCell);
        if (exactPointCount > NEAR_WATER_MAX_EXACT_POINTS) {
            throw std::runtime_error("near far-water authority exceeded one tile");
        }
        std::vector<ColumnPos> exactPositions;
        exactPositions.reserve(exactPointCount);
        // Ambiguous cells partition the tile into disjoint half-open block
        // footprints. Iterating those cells twice in the same row-major order
        // makes every exact position unique and lets sampled authority stay in
        // a compact parallel array without sorting or hashing coordinates.
        for (const uint32_t cellIndex : ambiguousCells) {
            const int cellX = static_cast<int>(cellIndex % static_cast<uint32_t>(cellEdge));
            const int cellZ = static_cast<int>(cellIndex / static_cast<uint32_t>(cellEdge));
            for (int offsetZ = 0; offsetZ < step; ++offsetZ) {
                for (int offsetX = 0; offsetX < step; ++offsetX) {
                    const int localX = cellX * step + offsetX;
                    const int localZ = cellZ * step + offsetZ;
                    if (exactSentinelAt(localX, localZ) != nullptr) continue;
                    exactPositions.push_back({mesh->originX + localX, mesh->originZ + localZ});
                }
            }
        }
        if (exactPositions.size() != exactPointCount) {
            throw std::logic_error("near far-water sentinel partition diverged");
        }
        std::vector<FarTerrainGeometrySample> exactGeometry(exactPositions.size());
        if (!exactPositions.empty() && source.geometryPoints) {
            for (size_t first = 0; first < exactPositions.size();
                 first += NEAR_WATER_MAX_POINT_BATCH) {
                const size_t count =
                    std::min(NEAR_WATER_MAX_POINT_BATCH, exactPositions.size() - first);
                source.geometryPoints(
                    std::span<const ColumnPos>(exactPositions).subspan(first, count),
                    worldgen::SurfaceFootprint::BLOCK_1,
                    std::span<FarTerrainGeometrySample>(exactGeometry).subspan(first, count));
            }
        } else {
            for (size_t index = 0; index < exactPositions.size(); ++index) {
                exactGeometry[index] = source
                                           .sample(exactPositions[index].x, exactPositions[index].z,
                                                   worldgen::SurfaceFootprint::BLOCK_1)
                                           .geometry;
            }
        }
        size_t exactIndex = 0;
        for (const uint32_t cellIndex : ambiguousCells) {
            const int cellX = static_cast<int>(cellIndex % static_cast<uint32_t>(cellEdge));
            const int cellZ = static_cast<int>(cellIndex / static_cast<uint32_t>(cellEdge));
            for (int offsetZ = 0; offsetZ < step; ++offsetZ) {
                for (int offsetX = 0; offsetX < step; ++offsetX) {
                    const int localX = cellX * step + offsetX;
                    const int localZ = cellZ * step + offsetZ;
                    const FarTerrainGeometrySample* geometry = exactSentinelAt(localX, localZ);
                    if (geometry == nullptr) geometry = &exactGeometry[exactIndex++];
                    const FarWaterAuthority authority =
                        waterAuthority(*geometry, sampleIsWet(*geometry));
                    if (authority.kind == FarWaterKind::NONE) continue;
                    pushWaterTop(*mesh, static_cast<float>(localX), static_cast<float>(localZ),
                                 static_cast<float>(localX + 1), static_cast<float>(localZ + 1),
                                 authority.height);
                }
            }
        }
        if (exactIndex != exactGeometry.size()) {
            throw std::logic_error("near far-water authority traversal diverged");
        }
        if (!exactPositions.empty()) mesh->complexity = 1.0F;
    } else {
        // A coarse cell may straddle two independent lakes or a standing body
        // and a much lower channel. Subdivide only those ambiguous rectangles on
        // the shared two-block water lattice. Each fine cell chooses the exact
        // authority at its center (or its first wet corner at a shoreline), then
        // clips every other authority out instead of triangulating a false ramp.
        auto emitAuthorityResolvedWater = [&](int rectangleX0, int rectangleZ0, int rectangleX1,
                                              int rectangleZ1, int resolution) {
            for (int fineZ0 = rectangleZ0; fineZ0 < rectangleZ1; fineZ0 += resolution) {
                const int fineZ1 = std::min(fineZ0 + resolution, rectangleZ1);
                for (int fineX0 = rectangleX0; fineX0 < rectangleX1; fineX0 += resolution) {
                    const int fineX1 = std::min(fineX0 + resolution, rectangleX1);
                    if (resolution == 1) {
                        const auto pointAt = [&](int localX, int localZ) {
                            const FarTerrainGeometrySample sample = canonicalWaterGeometry(
                                mesh->originX + localX, mesh->originZ + localZ);
                            const bool wet = sampleIsWet(sample);
                            return WaterPoint{
                                .x = static_cast<float>(localX),
                                .z = static_cast<float>(localZ),
                                .height = vertexHeight(sample.waterSurface),
                                .wet = wet,
                                .authority = waterAuthority(sample, wet),
                            };
                        };
                        const std::array<WaterPoint, 4> corners = {
                            pointAt(fineX0, fineZ0), pointAt(fineX1, fineZ0),
                            pointAt(fineX1, fineZ1), pointAt(fineX0, fineZ1)};
                        const auto firstWet =
                            std::find_if(corners.begin(), corners.end(),
                                         [](const WaterPoint& point) { return point.wet; });
                        if (firstWet == corners.end()) continue;

                        const bool allWet =
                            std::all_of(corners.begin(), corners.end(),
                                        [](const WaterPoint& point) { return point.wet; });
                        if (!allWet && firstWet->authority.kind != FarWaterKind::RIVER) {
                            // Standing-water ownership is a continuous
                            // shoreline field. Clip its last unit rectangle
                            // at the midpoint between wet and dry samples;
                            // filling that complete rectangle leaves a
                            // conspicuous one-block shelf beyond the lake or
                            // ocean contour.
                            const WaterPoint center{
                                .x = static_cast<float>(fineX0 + fineX1) * 0.5F,
                                .z = static_cast<float>(fineZ0 + fineZ1) * 0.5F,
                                .height = firstWet->authority.height,
                                .wet = false,
                                .authority = {},
                            };
                            constexpr float UNIT_CONNECTION_SPAN = 1.41421356F;
                            pushWaterContourTriangle(*mesh, {corners[0], corners[3], center},
                                                     firstWet->authority, UNIT_CONNECTION_SPAN,
                                                     refineWaterEdge);
                            pushWaterContourTriangle(*mesh, {corners[3], corners[2], center},
                                                     firstWet->authority, UNIT_CONNECTION_SPAN,
                                                     refineWaterEdge);
                            pushWaterContourTriangle(*mesh, {corners[2], corners[1], center},
                                                     firstWet->authority, UNIT_CONNECTION_SPAN,
                                                     refineWaterEdge);
                            pushWaterContourTriangle(*mesh, {corners[1], corners[0], center},
                                                     firstWet->authority, UNIT_CONNECTION_SPAN,
                                                     refineWaterEdge);
                            continue;
                        }

                        // Flowing block states own half-open unit cells. For
                        // an all-wet authority transition on a positive tile
                        // face, use the exterior coordinate so both LODs pick
                        // the same body's level on their shared edge.
                        const int authorityX = fineX1 == FAR_TERRAIN_TILE_EDGE ? fineX1 : fineX0;
                        const int authorityZ = fineZ1 == FAR_TERRAIN_TILE_EDGE ? fineZ1 : fineZ0;
                        const FarTerrainGeometrySample sample = canonicalWaterGeometry(
                            mesh->originX + authorityX, mesh->originZ + authorityZ);
                        FarWaterAuthority authority = waterAuthority(sample, sampleIsWet(sample));
                        if (authority.kind == FarWaterKind::NONE) {
                            if (!allWet) continue;
                            authority = firstWet->authority;
                        }
                        pushWaterTop(*mesh, static_cast<float>(fineX0), static_cast<float>(fineZ0),
                                     static_cast<float>(fineX1), static_cast<float>(fineZ1),
                                     authority.height);
                        continue;
                    }
                    const auto pointAt = [&](int localX, int localZ) {
                        const FarTerrainGeometrySample sample =
                            canonicalWaterGeometry(mesh->originX + localX, mesh->originZ + localZ);
                        const bool wet = sampleIsWet(sample);
                        return WaterPoint{
                            .x = static_cast<float>(localX),
                            .z = static_cast<float>(localZ),
                            .height = vertexHeight(sample.waterSurface),
                            .wet = wet,
                            .authority = waterAuthority(sample, wet),
                        };
                    };
                    const std::array<WaterPoint, 4> corners = {
                        pointAt(fineX0, fineZ0), pointAt(fineX1, fineZ0), pointAt(fineX1, fineZ1),
                        pointAt(fineX0, fineZ1)};
                    WaterPoint center = pointAt((fineX0 + fineX1) / 2, (fineZ0 + fineZ1) / 2);
                    center.x = static_cast<float>(fineX0 + fineX1) * 0.5F;
                    center.z = static_cast<float>(fineZ0 + fineZ1) * 0.5F;
                    FarWaterAuthority target = center.authority;
                    if (!center.wet) {
                        const auto firstWet =
                            std::find_if(corners.begin(), corners.end(),
                                         [](const WaterPoint& point) { return point.wet; });
                        if (firstWet == corners.end()) continue;
                        target = firstWet->authority;
                    }
                    const float connectionSpan = std::hypot(static_cast<float>(fineX1 - fineX0),
                                                            static_cast<float>(fineZ1 - fineZ0));
                    const bool allCornersOwned =
                        std::all_of(corners.begin(), corners.end(), [&](const WaterPoint& point) {
                            return waterAuthoritiesConnect(point.authority, target, connectionSpan);
                        });
                    const bool centerOwned =
                        waterAuthoritiesConnect(center.authority, target, connectionSpan);
                    if (allCornersOwned && centerOwned) {
                        pushWaterTop(*mesh, static_cast<float>(fineX0), static_cast<float>(fineZ0),
                                     static_cast<float>(fineX1), static_cast<float>(fineZ1),
                                     target.height);
                        continue;
                    }
                    pushWaterContourTriangle(*mesh, {corners[0], corners[3], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[3], corners[2], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[2], corners[1], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[1], corners[0], center}, target,
                                             connectionSpan, refineWaterEdge);
                }
            }
        };

        std::fill(merged.begin(), merged.end(), 0);
        if (key.step == FarTerrainStep::THIRTY_TWO) {
            // Coverage water is a globally aligned raster of half-open 8x8
            // voxel cells. A cell's lower-left block-resolution authority is
            // its canonical representative at every tile and worker order.
            // The west/north halo lets the positive-side cell own each shared
            // stage riser without consulting a neighboring mesh.
            constexpr int WATER_STEP = 8;
            constexpr int WATER_CELLS = FAR_TERRAIN_TILE_EDGE / WATER_STEP;
            constexpr int WATER_PAGE_EDGE = WATER_CELLS + 1;
            constexpr int WATER_HALO = 1;
            const int64_t waterOriginX = checkedCoordinateOffset(mesh->originX, -WATER_STEP);
            const int64_t waterOriginZ = checkedCoordinateOffset(mesh->originZ, -WATER_STEP);
            std::vector<FarTerrainGeometrySample> waterPage(
                static_cast<size_t>(WATER_PAGE_EDGE * WATER_PAGE_EDGE));
            const auto pageIndex = [](int pageX, int pageZ) {
                return static_cast<size_t>(pageZ * WATER_PAGE_EDGE + pageX);
            };
            const auto waterPageAt = [&](int pageX, int pageZ) -> const FarTerrainGeometrySample& {
                return waterPage[pageIndex(pageX, pageZ)];
            };
            const auto pagePosition = [&](int pageX, int pageZ, int offset) {
                return ColumnPos{
                    checkedCoordinateOffset(waterOriginX,
                                            static_cast<int64_t>(pageX * WATER_STEP + offset)),
                    checkedCoordinateOffset(waterOriginZ,
                                            static_cast<int64_t>(pageZ * WATER_STEP + offset)),
                };
            };
            const auto samplePointBatch = [&](std::span<const ColumnPos> positions,
                                              std::span<FarTerrainGeometrySample> output,
                                              bool finalGeometry) {
                if (positions.empty()) return;
                if (finalGeometry && source.geometryPoints) {
                    source.geometryPoints(positions, canonicalWaterFootprint, output);
                    return;
                }
                if (!finalGeometry && source.waterAuthorityPoints) {
                    source.waterAuthorityPoints(positions, canonicalWaterFootprint, output);
                    return;
                }
                for (size_t index = 0; index < positions.size(); ++index) {
                    output[index] =
                        source
                            .sample(positions[index].x, positions[index].z, canonicalWaterFootprint)
                            .geometry;
                }
            };

            bool directAuthorityPage = false;
            if (source.waterAuthorityGrid) {
                source.waterAuthorityGrid(waterOriginX, waterOriginZ, WATER_STEP, WATER_STEP,
                                          WATER_PAGE_EDGE, WATER_PAGE_EDGE, canonicalWaterFootprint,
                                          waterPage);
                directAuthorityPage = true;
            } else if (source.waterAuthorityPoints) {
                std::vector<ColumnPos> positions;
                positions.reserve(waterPage.size());
                for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                    for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX)
                        positions.push_back(pagePosition(pageX, pageZ, 0));
                }
                source.waterAuthorityPoints(positions, canonicalWaterFootprint, waterPage);
                directAuthorityPage = true;
            } else {
                for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                    for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                        const ColumnPos position = pagePosition(pageX, pageZ, 0);
                        waterPage[pageIndex(pageX, pageZ)] =
                            source.sample(position.x, position.z, canonicalWaterFootprint).geometry;
                    }
                }
            }

            const auto parentBoundsFor = [&](int localX,
                                             int localZ) -> const FarTerrainCellBounds* {
                if (authoritativeBounds.empty()) return nullptr;
                const int parentX = static_cast<int>(world_coord::floorDiv(localX, step));
                const int parentZ = static_cast<int>(world_coord::floorDiv(localZ, step));
                return &authoritativeBoundsAt(parentX, parentZ);
            };
            if (directAuthorityPage && source.geometryPoints && !authoritativeBounds.empty()) {
                std::vector<size_t> volcanicIndices;
                std::vector<ColumnPos> volcanicPositions;
                for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                    for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                        const int localX = (pageX - WATER_HALO) * WATER_STEP;
                        const int localZ = (pageZ - WATER_HALO) * WATER_STEP;
                        const FarTerrainCellBounds* bounds = parentBoundsFor(localX, localZ);
                        if (bounds == nullptr || !bounds->volcanicWaterPossible) continue;
                        volcanicIndices.push_back(pageIndex(pageX, pageZ));
                        volcanicPositions.push_back(pagePosition(pageX, pageZ, 0));
                    }
                }
                std::vector<FarTerrainGeometrySample> volcanicGeometry(volcanicPositions.size());
                samplePointBatch(volcanicPositions, volcanicGeometry, true);
                for (size_t index = 0; index < volcanicIndices.size(); ++index)
                    waterPage[volcanicIndices[index]] = volcanicGeometry[index];
            }

            bool emittedWater = false;
            for (int pageZ = WATER_HALO; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                for (int pageX = WATER_HALO; pageX < WATER_PAGE_EDGE; ++pageX) {
                    const FarTerrainGeometrySample& sample = waterPageAt(pageX, pageZ);
                    if (!sampleIsWet(sample)) continue;
                    const FarWaterAuthority authority = waterAuthority(sample, true);
                    const float localX = static_cast<float>((pageX - WATER_HALO) * WATER_STEP);
                    const float localZ = static_cast<float>((pageZ - WATER_HALO) * WATER_STEP);
                    pushWaterTop(*mesh, localX, localZ, localX + WATER_STEP, localZ + WATER_STEP,
                                 authority.height);
                    emittedWater = true;
                }
            }

            const auto emitOwnedRiser = [&](const FarTerrainGeometrySample& current,
                                            const FarTerrainGeometrySample& neighbor,
                                            FaceNormal currentHighFace, FaceNormal neighborHighFace,
                                            float x0, float z0, float x1, float z1) {
                if (!sampleIsWet(current) || !sampleIsWet(neighbor)) return;
                const FarWaterAuthority currentAuthority = waterAuthority(current, true);
                const FarWaterAuthority neighborAuthority = waterAuthority(neighbor, true);
                if (!sharesFlowTransition(currentAuthority, neighborAuthority,
                                          static_cast<float>(WATER_STEP)) ||
                    currentAuthority.height == neighborAuthority.height) {
                    return;
                }
                const bool currentHigh = currentAuthority.height > neighborAuthority.height;
                pushWaterRiser(*mesh, currentHigh ? currentHighFace : neighborHighFace, x0, z0, x1,
                               z1, std::min(currentAuthority.height, neighborAuthority.height),
                               std::max(currentAuthority.height, neighborAuthority.height));
                emittedWater = true;
            };
            for (int pageZ = WATER_HALO; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                for (int pageX = WATER_HALO; pageX < WATER_PAGE_EDGE; ++pageX) {
                    const float localX = static_cast<float>((pageX - WATER_HALO) * WATER_STEP);
                    const float localZ = static_cast<float>((pageZ - WATER_HALO) * WATER_STEP);
                    const FarTerrainGeometrySample& current = waterPageAt(pageX, pageZ);
                    emitOwnedRiser(current, waterPageAt(pageX - 1, pageZ), FaceNormal::MINUS_X,
                                   FaceNormal::PLUS_X, localX, localZ, localX, localZ + WATER_STEP);
                    emitOwnedRiser(current, waterPageAt(pageX, pageZ - 1), FaceNormal::MINUS_Z,
                                   FaceNormal::PLUS_Z, localX, localZ, localX + WATER_STEP, localZ);
                }
            }
            std::fill(merged.begin(), merged.end(), 1);
            if (emittedWater) mesh->complexity = 1.0F;
        }
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                const bool replacedBySharedBoundary =
                    (x == 0 && refineWaterBoundary[WEST]) ||
                    (x == cellEdge - 1 && refineWaterBoundary[EAST]) ||
                    (z == 0 && refineWaterBoundary[NORTH]) ||
                    (z == cellEdge - 1 && refineWaterBoundary[SOUTH]);
                if (replacedBySharedBoundary) {
                    merged[static_cast<size_t>(z * cellEdge + x)] = 1;
                }
            }
        }
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                FarCell& cell = cellAt(x, z);
                const size_t cellIndex = static_cast<size_t>(z * cellEdge + x);
                if (!cell.water || merged[cellIndex] != 0) continue;
                const float x0 = static_cast<float>(x * step);
                const float z0 = static_cast<float>(z * step);
                const float x1 = static_cast<float>((x + 1) * step);
                const float z1 = static_cast<float>((z + 1) * step);
                if ((cell.discontinuousWater || (step >= 8 && cell.waterMask != 0x0FU)) &&
                    key.step != FarTerrainStep::THIRTY_TWO) {
                    emitAuthorityResolvedWater(x * step, z * step, (x + 1) * step, (z + 1) * step,
                                               2);
                    merged[cellIndex] = 1;
                    continue;
                }
                if (cell.waterMask != 0x0FU || cell.discontinuousWater) {
                    const std::array<WaterPoint, 4> corners = {{
                        {x0, z0, cell.waterSurface[0], (cell.waterMask & (1U << 0U)) != 0,
                         cell.waterAuthority[0]},
                        {x1, z0, cell.waterSurface[1], (cell.waterMask & (1U << 1U)) != 0,
                         cell.waterAuthority[1]},
                        {x1, z1, cell.waterSurface[2], (cell.waterMask & (1U << 2U)) != 0,
                         cell.waterAuthority[2]},
                        {x0, z1, cell.waterSurface[3], (cell.waterMask & (1U << 3U)) != 0,
                         cell.waterAuthority[3]},
                    }};
                    const WaterPoint center{(x0 + x1) * 0.5F, (z0 + z1) * 0.5F,
                                            cell.centerWaterHeight, cell.centerWet,
                                            cell.centerWaterAuthority};
                    FarWaterAuthority target = cell.centerWaterAuthority;
                    if (!cell.centerWet) {
                        const auto firstWet =
                            std::find_if(corners.begin(), corners.end(),
                                         [](const WaterPoint& point) { return point.wet; });
                        if (firstWet != corners.end()) target = firstWet->authority;
                    }
                    const float connectionSpan = static_cast<float>(step) * 1.41421356F;
                    pushWaterContourTriangle(*mesh, {corners[0], corners[3], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[3], corners[2], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[2], corners[1], center}, target,
                                             connectionSpan, refineWaterEdge);
                    pushWaterContourTriangle(*mesh, {corners[1], corners[0], center}, target,
                                             connectionSpan, refineWaterEdge);
                    merged[cellIndex] = 1;
                    continue;
                }
                if (!cell.flatWater) {
                    const float height = cell.centerWet ? cell.centerWaterHeight : cell.waterHeight;
                    pushWaterTop(*mesh, x0, z0, x1, z1, height);
                    merged[cellIndex] = 1;
                    continue;
                }
                int width = 1;
                while (x + width < cellEdge &&
                       merged[static_cast<size_t>(z * cellEdge + x + width)] == 0 &&
                       sameWater(cell, cellAt(x + width, z))) {
                    ++width;
                }
                int depth = 1;
                while (z + depth < cellEdge) {
                    bool compatible = true;
                    for (int offset = 0; offset < width; ++offset) {
                        const size_t candidate =
                            static_cast<size_t>((z + depth) * cellEdge + x + offset);
                        if (merged[candidate] != 0 ||
                            !sameWater(cell, cellAt(x + offset, z + depth))) {
                            compatible = false;
                            break;
                        }
                    }
                    if (!compatible) break;
                    ++depth;
                }
                for (int dz = 0; dz < depth; ++dz) {
                    for (int dx = 0; dx < width; ++dx) {
                        merged[static_cast<size_t>((z + dz) * cellEdge + x + dx)] = 1;
                    }
                }
                pushWaterTop(*mesh, x0, z0, static_cast<float>((x + width) * step),
                             static_cast<float>((z + depth) * step), cell.waterHeight);
            }
        }

        auto emitSharedWaterCell = [&](int x0, int z0, int x1, int z1) {
            const std::array<std::array<int, 2>, 4> coordinates = {{
                {{x0, z0}},
                {{x1, z0}},
                {{x1, z1}},
                {{x0, z1}},
            }};
            std::array<WaterPoint, 4> corners{};
            bool allWet = true;
            bool anyWet = false;
            bool flatWater = true;
            float firstHeight = 0.0F;
            for (size_t index = 0; index < coordinates.size(); ++index) {
                const auto [localX, localZ] = coordinates[index];
                const FarTerrainGeometrySample& sample = sharedWaterSample(localX, localZ);
                const bool wet = sampleIsWet(sample);
                corners[index] = {
                    .x = static_cast<float>(localX),
                    .z = static_cast<float>(localZ),
                    .height = vertexHeight(sample.waterSurface),
                    .wet = wet,
                    .authority = waterAuthority(sample, wet),
                };
                if (index == 0) firstHeight = corners[index].height;
                allWet = allWet && corners[index].wet;
                anyWet = anyWet || corners[index].wet;
                flatWater = flatWater && corners[index].height == firstHeight;
            }
            const int centerX = (x0 + x1) / 2;
            const int centerZ = (z0 + z1) / 2;
            const FarTerrainGeometrySample& centerSample = sharedWaterSample(centerX, centerZ);
            const bool centerWet = sampleIsWet(centerSample);
            const WaterPoint center{
                .x = static_cast<float>(centerX),
                .z = static_cast<float>(centerZ),
                .height = vertexHeight(centerSample.waterSurface),
                .wet = centerWet,
                .authority = waterAuthority(centerSample, centerWet),
            };
            anyWet = anyWet || center.wet;
            if (!anyWet) return;
            FarWaterAuthority target = center.authority;
            if (!center.wet) {
                const auto firstWet =
                    std::find_if(corners.begin(), corners.end(),
                                 [](const WaterPoint& point) { return point.wet; });
                if (firstWet != corners.end()) target = firstWet->authority;
            }
            const float connectionSpan =
                std::hypot(static_cast<float>(x1 - x0), static_cast<float>(z1 - z0));
            const bool allOwned =
                allWet && std::all_of(corners.begin(), corners.end(), [&](const WaterPoint& point) {
                    return waterAuthoritiesConnect(point.authority, target, connectionSpan);
                });
            const bool centerOwned =
                waterAuthoritiesConnect(center.authority, target, connectionSpan);
            if (allOwned && centerOwned && flatWater) {
                pushWaterTop(*mesh, static_cast<float>(x0), static_cast<float>(z0),
                             static_cast<float>(x1), static_cast<float>(z1), firstHeight);
                return;
            }
            emitAuthorityResolvedWater(x0, z0, x1, z1, 1);
        };
        auto emitSharedBoundary = [&](WaterEdge edge) {
            if (!refineWaterBoundary[edge]) return;
            // West and east own the four corner squares whenever their strips
            // are active. North and south omit those same squares so orthogonal
            // boundary refinement remains a disjoint partition instead of
            // emitting coplanar or conflicting water triangles twice.
            const bool horizontal = edge == NORTH || edge == SOUTH;
            const int begin = horizontal && refineWaterBoundary[WEST] ? step : 0;
            const int end = horizontal && refineWaterBoundary[EAST] ? FAR_TERRAIN_TILE_EDGE - step
                                                                    : FAR_TERRAIN_TILE_EDGE;
            for (int coordinate = begin; coordinate < end; coordinate += SHARED_WATER_EDGE_STEP) {
                if (edge == WEST) {
                    emitSharedWaterCell(0, coordinate, step, coordinate + SHARED_WATER_EDGE_STEP);
                } else if (edge == EAST) {
                    emitSharedWaterCell(FAR_TERRAIN_TILE_EDGE - step, coordinate,
                                        FAR_TERRAIN_TILE_EDGE, coordinate + SHARED_WATER_EDGE_STEP);
                } else if (edge == NORTH) {
                    emitSharedWaterCell(coordinate, 0, coordinate + SHARED_WATER_EDGE_STEP, step);
                } else {
                    emitSharedWaterCell(coordinate, FAR_TERRAIN_TILE_EDGE - step,
                                        coordinate + SHARED_WATER_EDGE_STEP, FAR_TERRAIN_TILE_EDGE);
                }
            }
        };
        emitSharedBoundary(WEST);
        emitSharedBoundary(EAST);
        emitSharedBoundary(NORTH);
        emitSharedBoundary(SOUTH);
    }

    if (mesh->vertices.empty()) {
        mesh->bounds.minY = 0.0F;
        mesh->bounds.maxY = 0.0F;
        mesh->surfaceBounds.minY = 0.0F;
        mesh->surfaceBounds.maxY = 0.0F;
    }
    for (FarTerrainBounds& patch : mesh->occluderPatches) {
        if (patch.minY > patch.maxY) {
            patch.minY = mesh->surfaceBounds.minY;
            patch.maxY = mesh->surfaceBounds.maxY;
        }
    }
    mesh->deterministicHash = hashMesh(*mesh);
    return mesh;
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainMesher::buildFromSurface(FarTerrainKey key, const SurfaceSampleFunction& sampleSurface) {
    return build(key, surfaceGeometrySource(sampleSurface));
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainMesher::buildFromSurface(FarTerrainKey key,
                                   const BlockSurfaceSampleFunction& sampleSurface) {
    return build(key, surfaceGeometrySource(sampleSurface));
}

FarTerrainSource FarTerrainMesher::surfaceGeometrySource(SurfaceSampleFunction sampleSurface) {
    if (!sampleSurface) throw std::invalid_argument("far terrain surface sampler is empty");
    FarTerrainSource source;
    source.sample = [sampleSurface = std::move(sampleSurface)](
                        int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        return farSampleFromSurface(sampleSurface(x, z, footprint));
    };
    return source;
}

FarTerrainSource FarTerrainMesher::surfaceGeometrySource(BlockSurfaceSampleFunction sampleSurface) {
    if (!sampleSurface) throw std::invalid_argument("far terrain surface sampler is empty");
    return surfaceGeometrySource(
        [sampleSurface = std::move(sampleSurface)](
            int64_t x, int64_t z, worldgen::SurfaceFootprint) { return sampleSurface(x, z); });
}

FarTerrainSource
FarTerrainMesher::tieredSurfaceGeometrySource(BlockSurfaceSampleFunction exactNearSurface,
                                              BlockSurfaceSampleFunction coarseSurface) {
    if (!exactNearSurface || !coarseSurface) {
        throw std::invalid_argument("far terrain exact near surface sampler is empty");
    }
    FarTerrainSource source;
    source.sample = [exactNearSurface = std::move(exactNearSurface),
                     coarseSurface = std::move(coarseSurface)](
                        int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        const worldgen::SurfaceSample canonical = coarseSurface(x, z);
        if (surfaceFootprintWidth(footprint) > 4) return farSampleFromSurface(canonical);

        const worldgen::SurfaceSample exact = exactNearSurface(x, z);
        FarSurfaceSample result = farSampleFromSurface(exact);
        result.geometry = geometryFromSurface(canonical);
        result.geometry.terrainHeight = exact.terrainHeight;
        result.footprintMinimumTerrainHeight = exact.terrainHeight;
        result.footprintMaximumTerrainHeight = exact.terrainHeight;
        return result;
    };
    return source;
}

FarTerrainSource
FarTerrainMesher::generatorGeometrySource(std::shared_ptr<ChunkGenerator> generator) {
    if (!generator) throw std::invalid_argument("far terrain generator is empty");
    FarTerrainSource source;
    source.sample = [generator](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        // Exact cubes are the only one-block terrain tier. Far step-2 and
        // step-4 relief remains a bounded macro sample; their water mask is
        // gathered separately in one block-authority batch without building
        // complete ColumnPlans for the tile.
        const bool exactHandoff = surfaceFootprintWidth(footprint) == 1;
        const worldgen::SurfaceSample surface = exactHandoff
                                                    ? generator->sampleExactSurface(x, z)
                                                    : generator->sampleFarSurface(x, z, footprint);
        FarSurfaceSample result = farSampleFromSurface(surface);
        if (exactHandoff) {
            // The exact density top owns the handoff geometry, but a direct
            // coordinate sample owns water topology and level. Column-plan
            // interpolation must never change a lake between LODs.
            worldgen::SurfaceSample canonical =
                generator->sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
            canonical.terrainHeight = surface.terrainHeight;
            canonical.hydrology.surfaceElevation = surface.terrainHeight;
            result = farSampleFromSurface(canonical);
            result.materialPalette = generator->farSurfaceMaterialPaletteAt(x, z, canonical);
        } else {
            // Exact amplitude sums for relief bands filtered below each
            // footprint plus one block for density-to-voxel quantization make
            // the coarse parent conservative at every shared sample.
            result = boundedFarSampleFromSurface(
                surface, footprint,
                ChunkGenerator::emittedSurfaceDetailAmplitude(
                    surface, footprint == worldgen::SurfaceFootprint::BLOCK_32
                                 ? FAR_TERRAIN_STEP32_RELIEF_SLOPE_ENVELOPE
                                 : FAR_TERRAIN_STEP16_RELIEF_SLOPE_ENVELOPE));
            includeCanonicalWaterFloor(result, generator->sampleFarGeometrySurface(
                                                   x, z, worldgen::SurfaceFootprint::BLOCK_1));
            // A filtered macro sample cannot predict every emitted density
            // opening at this coordinate. Point sampling is not the hot tile
            // path, so retain the exact voxel top explicitly. Batched mesh
            // construction uses cellBoundsGrid for the stronger interior
            // footprint guarantee without constructing column plans here.
            const double exactTerrainHeight = generator->sampleExactSurface(x, z).terrainHeight;
            result.footprintMinimumTerrainHeight =
                std::min(result.footprintMinimumTerrainHeight, exactTerrainHeight);
            result.footprintMaximumTerrainHeight =
                std::max(result.footprintMaximumTerrainHeight, exactTerrainHeight);
            result.materialPalette = generator->farSurfaceMaterialPaletteAt(x, z, surface);
        }
        return result;
    };
    source.sampleGrid = [generator](int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                                    worldgen::SurfaceFootprint footprint,
                                    std::span<FarSurfaceSample> output) {
        if (spacing <= 0 || sampleEdge <= 0 ||
            worldgen::surfaceFootprintWidth(footprint) != spacing ||
            output.size() != static_cast<size_t>(sampleEdge * sampleEdge)) {
            throw std::invalid_argument("invalid far terrain sample grid");
        }
        const bool exactHandoff = worldgen::surfaceFootprintWidth(footprint) == 1;
        const int expandedEdge = sampleEdge + 2;
        const int64_t expandedOriginX = checkedCoordinateOffset(originX, -spacing);
        const int64_t expandedOriginZ = checkedCoordinateOffset(originZ, -spacing);
        GeneratorCellBoundsBatch& batch = threadGeneratorCellBoundsBatch();
        batch.generator = generator.get();
        batch.generatorSeed = generator->seed();
        batch.originX = expandedOriginX;
        batch.originZ = expandedOriginZ;
        batch.step = spacing;
        batch.cellWidth = expandedEdge - 1;
        batch.cellHeight = expandedEdge - 1;
        batch.footprint = footprint;
        batch.vertices.resize(static_cast<size_t>(expandedEdge * expandedEdge));
        if (exactHandoff) {
            generator->sampleExactSurfaceGrid(expandedOriginX, expandedOriginZ, spacing,
                                              expandedEdge, batch.vertices);
        } else {
            generator->sampleFarSurfaceGrid(expandedOriginX, expandedOriginZ, spacing, expandedEdge,
                                            footprint, batch.vertices);
        }
        if (spacing == 32) {
            batch.canonicalVertices = batch.vertices;
        } else {
            batch.canonicalVertices.resize(batch.vertices.size());
            generator->sampleFarGeometryGrid(
                expandedOriginX, expandedOriginZ, spacing, spacing, expandedEdge, expandedEdge,
                worldgen::SurfaceFootprint::BLOCK_1, batch.canonicalVertices);
        }
        if (spacing >= 4 && spacing <= 32 && (spacing & 1) == 0) {
            batch.centers.resize(static_cast<size_t>(batch.cellWidth * batch.cellHeight));
            generator->sampleFarGeometryGrid(checkedCoordinateOffset(expandedOriginX, spacing / 2),
                                             checkedCoordinateOffset(expandedOriginZ, spacing / 2),
                                             spacing, spacing, batch.cellWidth, batch.cellHeight,
                                             footprint, batch.centers);
        } else {
            batch.centers.clear();
        }
        for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
                const size_t outputIndex = static_cast<size_t>(sampleZ * sampleEdge + sampleX);
                const size_t batchIndex =
                    static_cast<size_t>((sampleZ + 1) * expandedEdge + sampleX + 1);
                worldgen::SurfaceSample& retainedSurface = batch.vertices[batchIndex];
                const worldgen::SurfaceSample& canonicalWater = batch.canonicalVertices[batchIndex];
                worldgen::SurfaceSample materialSurface = retainedSurface;
                if (exactHandoff) {
                    const double emittedTop = materialSurface.terrainHeight;
                    materialSurface.hydrology = canonicalWater.hydrology;
                    materialSurface.waterSurface = canonicalWater.waterSurface;
                    materialSurface.terrainHeight = emittedTop;
                    materialSurface.hydrology.surfaceElevation = emittedTop;
                    if (materialSurface.hydrology.lake) {
                        materialSurface.hydrology.lakeDepth =
                            std::max(0.0, materialSurface.hydrology.waterSurface - emittedTop);
                    }
                    output[outputIndex] = farSampleFromSurface(materialSurface);
                    retainedSurface = materialSurface;
                } else {
                    output[outputIndex] = boundedFarSampleFromSurface(
                        materialSurface, footprint,
                        ChunkGenerator::emittedSurfaceDetailAmplitude(
                            materialSurface, footprintReliefSlopeEnvelope(footprint)));
                    includeCanonicalWaterFloor(output[outputIndex], canonicalWater);
                }
                const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
                const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
                output[outputIndex].materialPalette =
                    generator->farSurfaceMaterialPaletteAt(worldX, worldZ, materialSurface);
            }
        }
    };
    source.cellBoundsGrid = [generator](int64_t originX, int64_t originZ, int step, int cellWidth,
                                        int cellHeight, worldgen::SurfaceFootprint footprint,
                                        std::span<FarTerrainCellBounds> output) {
        if (step <= 0 || cellWidth <= 0 || cellHeight <= 0 ||
            worldgen::surfaceFootprintWidth(footprint) != step ||
            output.size() != static_cast<size_t>(cellWidth * cellHeight)) {
            throw std::invalid_argument("invalid far terrain cell bounds grid");
        }
        GeneratorCellBoundsBatch& batch = threadGeneratorCellBoundsBatch();
        if (!batch.matches(generator.get(), generator->seed(), originX, originZ, step, cellWidth,
                           cellHeight, footprint)) {
            prepareGeneratorCellBoundsBatch(batch, generator, originX, originZ, step, cellWidth,
                                            cellHeight, footprint, true);
        }

        std::vector<GeneratorSurfaceBounds> localBounds(output.size());
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                localBounds[static_cast<size_t>(z * cellWidth + x)] =
                    generatorCellSurfaceBounds(batch, x, z);
            }
        }

        GeneratorCellBoundsBatch ancestorBatch;
        std::vector<GeneratorSurfaceBounds> ancestorBounds;
        int64_t ancestorCellX = 0;
        int64_t ancestorCellZ = 0;
        int ancestorWidth = 0;
        int ancestorHeight = 0;
        if (step < 32) {
            const __int128 exclusiveX =
                static_cast<__int128>(originX) + static_cast<__int128>(cellWidth) * step;
            const __int128 exclusiveZ =
                static_cast<__int128>(originZ) + static_cast<__int128>(cellHeight) * step;
            if (exclusiveX <= std::numeric_limits<int64_t>::min() ||
                exclusiveX > std::numeric_limits<int64_t>::max() ||
                exclusiveZ <= std::numeric_limits<int64_t>::min() ||
                exclusiveZ > std::numeric_limits<int64_t>::max()) {
                throw std::out_of_range("far terrain ancestor bounds exceed int64 range");
            }
            ancestorCellX = world_coord::floorDiv(originX, int64_t{32}) - 1;
            ancestorCellZ = world_coord::floorDiv(originZ, int64_t{32}) - 1;
            const int64_t lastAncestorX =
                world_coord::floorDiv(static_cast<int64_t>(exclusiveX) - 1, int64_t{32}) + 1;
            const int64_t lastAncestorZ =
                world_coord::floorDiv(static_cast<int64_t>(exclusiveZ) - 1, int64_t{32}) + 1;
            ancestorWidth = static_cast<int>(lastAncestorX - ancestorCellX + 1);
            ancestorHeight = static_cast<int>(lastAncestorZ - ancestorCellZ + 1);
            const __int128 ancestorOriginX128 = static_cast<__int128>(ancestorCellX) * 32;
            const __int128 ancestorOriginZ128 = static_cast<__int128>(ancestorCellZ) * 32;
            if (ancestorOriginX128 < std::numeric_limits<int64_t>::min() ||
                ancestorOriginX128 > std::numeric_limits<int64_t>::max() ||
                ancestorOriginZ128 < std::numeric_limits<int64_t>::min() ||
                ancestorOriginZ128 > std::numeric_limits<int64_t>::max()) {
                throw std::out_of_range("far terrain ancestor origin exceeds int64 range");
            }
            prepareGeneratorCellBoundsBatch(
                ancestorBatch, generator, static_cast<int64_t>(ancestorOriginX128),
                static_cast<int64_t>(ancestorOriginZ128), 32, ancestorWidth, ancestorHeight,
                worldgen::SurfaceFootprint::BLOCK_32, true);
            ancestorBounds.resize(static_cast<size_t>(ancestorWidth * ancestorHeight));
            for (int z = 0; z < ancestorHeight; ++z) {
                for (int x = 0; x < ancestorWidth; ++x) {
                    ancestorBounds[static_cast<size_t>(z * ancestorWidth + x)] =
                        generatorCellSurfaceBounds(ancestorBatch, x, z, false);
                }
            }
        }

        const auto includeAdjacentAncestorMinimum = [&](int x, int z, double minimum) {
            if (step == 32) {
                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int neighborX = x + dx;
                        const int neighborZ = z + dz;
                        if (neighborX < 0 || neighborX >= cellWidth || neighborZ < 0 ||
                            neighborZ >= cellHeight) {
                            continue;
                        }
                        minimum = std::min(
                            minimum,
                            localBounds[static_cast<size_t>(neighborZ * cellWidth + neighborX)]
                                .minimum);
                    }
                }
                return minimum;
            }
            const int64_t worldX = checkedCoordinateOffset(originX, static_cast<int64_t>(x) * step);
            const int64_t worldZ = checkedCoordinateOffset(originZ, static_cast<int64_t>(z) * step);
            const int parentX =
                static_cast<int>(world_coord::floorDiv(worldX, int64_t{32}) - ancestorCellX);
            const int parentZ =
                static_cast<int>(world_coord::floorDiv(worldZ, int64_t{32}) - ancestorCellZ);
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int neighborX = parentX + dx;
                    const int neighborZ = parentZ + dz;
                    if (neighborX < 0 || neighborX >= ancestorWidth || neighborZ < 0 ||
                        neighborZ >= ancestorHeight) {
                        continue;
                    }
                    minimum = std::min(
                        minimum,
                        ancestorBounds[static_cast<size_t>(neighborZ * ancestorWidth + neighborX)]
                            .minimum);
                }
            }
            return minimum;
        };

        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const size_t index = static_cast<size_t>(z * cellWidth + x);
                GeneratorSurfaceBounds bounds = localBounds[index];
                const double minimum = std::clamp(bounds.minimum, static_cast<double>(WORLD_MIN_Y),
                                                  static_cast<double>(WORLD_MAX_Y + 1));
                const double maximum =
                    std::clamp(std::max(bounds.maximum, minimum), static_cast<double>(WORLD_MIN_Y),
                               static_cast<double>(WORLD_MAX_Y + 1));
                const double terrainHeight = std::clamp(bounds.terrainHeight, minimum, maximum);
                const double ancestorMinimum = includeAdjacentAncestorMinimum(x, z, minimum);
                const double skirtBottom =
                    std::clamp(std::min(minimum - FAR_TERRAIN_SKIRT_DEPTH, ancestorMinimum),
                               static_cast<double>(WORLD_MIN_Y), minimum);
                output[index] = {
                    .terrainHeight = terrainHeight,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = maximum,
                    .skirtBottom = skirtBottom,
                    .waterTopologyPossible = bounds.waterTopologyPossible,
                    .volcanicWaterPossible = bounds.volcanicWaterPossible,
                };
            }
        }
    };
    source.geometryGrid = [generator](int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                      int sampleWidth, int sampleHeight,
                                      worldgen::SurfaceFootprint footprint,
                                      std::span<FarTerrainGeometrySample> output) {
        if (output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
            throw std::invalid_argument("invalid far terrain geometry grid");
        }
        const bool nearWaterSentinels = spacingX == 2 && spacingZ == 2 &&
                                        sampleWidth == sampleHeight &&
                                        (sampleWidth == FAR_TERRAIN_TILE_EDGE / 2 ||
                                         sampleWidth == FAR_TERRAIN_TILE_EDGE / 2 + 1) &&
                                        footprint == worldgen::SurfaceFootprint::BLOCK_1;
        if (nearWaterSentinels) {
            std::vector<worldgen::SurfaceSample> surfaces(output.size());
            generator->sampleGeneratedWaterGeometryGrid(originX, originZ, spacingX, spacingZ,
                                                        sampleWidth, sampleHeight, surfaces);
            for (size_t index = 0; index < output.size(); ++index)
                output[index] = geometryFromSurface(surfaces[index]);
            return;
        }
        std::vector<worldgen::SurfaceSample> surfaces(output.size());
        generator->sampleFarGeometryGrid(originX, originZ, spacingX, spacingZ, sampleWidth,
                                         sampleHeight, footprint, surfaces);
        for (size_t index = 0; index < surfaces.size(); ++index) {
            output[index] = geometryFromSurface(surfaces[index]);
        }
    };
    source.geometryPoints = [generator](std::span<const ColumnPos> positions,
                                        worldgen::SurfaceFootprint footprint,
                                        std::span<FarTerrainGeometrySample> output) {
        if (positions.size() != output.size()) {
            throw std::invalid_argument("invalid far terrain geometry points");
        }
        if (footprint == worldgen::SurfaceFootprint::BLOCK_1 && !positions.empty()) {
            std::vector<worldgen::SurfaceSample> surfaces(output.size());
            generator->sampleGeneratedWaterGeometryPoints(positions, surfaces);
            for (size_t index = 0; index < surfaces.size(); ++index)
                output[index] = geometryFromSurface(surfaces[index]);
            return;
        }
        std::vector<worldgen::SurfaceSample> surfaces(output.size());
        generator->sampleFarGeometryPoints(positions, footprint, surfaces);
        for (size_t index = 0; index < surfaces.size(); ++index) {
            output[index] = geometryFromSurface(surfaces[index]);
        }
    };
    source.waterAuthorityGrid = [generator](int64_t originX, int64_t originZ, int spacingX,
                                            int spacingZ, int sampleWidth, int sampleHeight,
                                            worldgen::SurfaceFootprint footprint,
                                            std::span<FarTerrainGeometrySample> output) {
        if (spacingX <= 0 || spacingX != spacingZ || sampleWidth <= 0 ||
            sampleWidth != sampleHeight ||
            output.size() != static_cast<size_t>(sampleWidth * sampleHeight) ||
            footprint != worldgen::SurfaceFootprint::BLOCK_1) {
            throw std::invalid_argument("invalid far water authority grid");
        }
        std::vector<worldgen::HydrologySample> hydrology(output.size());
        generator->sampleGeneratedWaterAuthorityGrid(originX, originZ, spacingX, sampleWidth,
                                                     hydrology);
        for (size_t index = 0; index < hydrology.size(); ++index)
            output[index] = geometryFromHydrology(hydrology[index]);
    };
    source.waterAuthorityPoints = [generator](std::span<const ColumnPos> positions,
                                              worldgen::SurfaceFootprint footprint,
                                              std::span<FarTerrainGeometrySample> output) {
        if (positions.size() != output.size() || footprint != worldgen::SurfaceFootprint::BLOCK_1) {
            throw std::invalid_argument("invalid far water authority points");
        }
        std::vector<worldgen::HydrologySample> hydrology(output.size());
        generator->sampleGeneratedWaterAuthorityPoints(positions, hydrology);
        for (size_t index = 0; index < hydrology.size(); ++index)
            output[index] = geometryFromHydrology(hydrology[index]);
    };
    source.canopies = [generator](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                  int64_t maximumZ, FarTerrainStep step) {
        return generator->collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ,
                                                   farTerrainStepSize(step));
    };
    source.materialRank = [generator](int64_t x, int64_t z) {
        return generator->farSurfaceMaterialRankAt(x, z);
    };
    return source;
}

FarTerrainSource
FarTerrainMesher::macroGeometrySource(std::shared_ptr<worldgen::MacroGenerationSampler> sampler) {
    if (!sampler) throw std::invalid_argument("far terrain macro sampler is empty");
    FarTerrainSource source;
    source.sample = [sampler = std::move(sampler)](int64_t x, int64_t z,
                                                   worldgen::SurfaceFootprint footprint) {
        return farSampleFromSurface(
            sampler->sampleSurface(static_cast<double>(x), static_cast<double>(z), footprint));
    };
    return source;
}

TerrainHorizonCuller::TerrainHorizonCuller(TerrainHorizonViewpoint viewpoint) {
    reset(viewpoint);
}

void TerrainHorizonCuller::reset(TerrainHorizonViewpoint viewpoint) {
    viewpoint_ = viewpoint;
    horizonCounts_.fill(0);
}

double TerrainHorizonCuller::horizontalDistanceSquared(const FarTerrainBounds& bounds,
                                                       TerrainHorizonViewpoint viewpoint) {
    const double dx = viewpoint.x < static_cast<double>(bounds.minX)
                          ? static_cast<double>(bounds.minX) - viewpoint.x
                      : viewpoint.x > static_cast<double>(bounds.maxX)
                          ? viewpoint.x - static_cast<double>(bounds.maxX)
                          : 0.0;
    const double dz = viewpoint.z < static_cast<double>(bounds.minZ)
                          ? static_cast<double>(bounds.minZ) - viewpoint.z
                      : viewpoint.z > static_cast<double>(bounds.maxZ)
                          ? viewpoint.z - static_cast<double>(bounds.maxZ)
                          : 0.0;
    return dx * dx + dz * dz;
}

bool TerrainHorizonCuller::isOccluded(const FarTerrainBounds& surfaceBounds) const {
    const AzimuthCoverage coverage = azimuthCoverage(surfaceBounds, viewpoint_);
    const double nearestDistance = std::sqrt(horizontalDistanceSquared(surfaceBounds, viewpoint_));
    if (nearestDistance <= 1.0e-9) return false;
    const double farthestDistance = farthestHorizontalDistance(surfaceBounds, viewpoint_);
    const double verticalDelta = static_cast<double>(surfaceBounds.maxY) - viewpoint_.y;
    const double distance = verticalDelta >= 0.0 ? nearestDistance : farthestDistance;
    const double maximumElevation = std::atan2(verticalDelta, std::max(distance, 1.0e-9));
    bool occluded = true;
    const size_t count = visitIntersectedBins(coverage, [&](size_t bin) {
        bool binOccluded = false;
        for (uint8_t index = 0; index < horizonCounts_[bin]; ++index) {
            const HorizonEntry& horizon = horizons_[bin][index];
            if (horizon.farthestDistance < nearestDistance - 1.0e-9 &&
                maximumElevation < horizon.minimumElevation - 1.0e-12) {
                binOccluded = true;
                break;
            }
        }
        if (!binOccluded) {
            occluded = false;
        }
    });
    return count != 0 && occluded;
}

void TerrainHorizonCuller::addOccluder(const FarTerrainBounds& surfaceBounds) {
    const AzimuthCoverage coverage = azimuthCoverage(surfaceBounds, viewpoint_);
    const double nearestDistance = std::sqrt(horizontalDistanceSquared(surfaceBounds, viewpoint_));
    if (nearestDistance <= 1.0e-9) return;
    const double farthestDistance = farthestHorizontalDistance(surfaceBounds, viewpoint_);
    const double verticalDelta = static_cast<double>(surfaceBounds.minY) - viewpoint_.y;
    const double distance = verticalDelta >= 0.0 ? farthestDistance : nearestDistance;
    const double minimumElevation = std::atan2(verticalDelta, std::max(distance, 1.0e-9));
    visitFullyCoveredBins(coverage, [&](size_t bin) {
        auto& entries = horizons_[bin];
        uint8_t& count = horizonCounts_[bin];

        // Keep a bounded Pareto frontier. A nearer, higher horizon dominates
        // a farther, lower one for every possible candidate. Dropping an entry
        // when the fixed frontier is full can only reduce culling, never hide
        // visible terrain.
        for (uint8_t index = 0; index < count; ++index) {
            if (entries[index].farthestDistance <= farthestDistance &&
                entries[index].minimumElevation >= minimumElevation) {
                return;
            }
        }
        for (uint8_t index = 0; index < count;) {
            if (farthestDistance <= entries[index].farthestDistance &&
                minimumElevation >= entries[index].minimumElevation) {
                entries[index] = entries[--count];
            } else {
                ++index;
            }
        }
        if (count < MAX_HORIZONS_PER_BIN) {
            entries[count++] = {farthestDistance, minimumElevation};
        }
    });
}

bool TerrainHorizonCuller::testAndAdd(const FarTerrainBounds& surfaceBounds) {
    if (isOccluded(surfaceBounds)) return true;
    addOccluder(surfaceBounds);
    return false;
}

FarTerrainScheduler::FarTerrainScheduler(FarTerrainSource source, FarTerrainSchedulerLimits limits)
    : source_(std::move(source))
    , limits_(limits) {
    if (!source_.sample) {
        throw std::invalid_argument("far terrain scheduler source is incomplete");
    }
    validateLimits(limits_);
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        const bool latencySensitive = index < LATENCY_WORKER_COUNT;
        workers_.emplace_back([this, latencySensitive] { workerLoop(latencySensitive); });
    }
}

FarTerrainScheduler::FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits)
    : generator_(std::make_shared<ChunkGenerator>(static_cast<uint32_t>(worldSeed)))
    , source_(FarTerrainMesher::generatorGeometrySource(generator_))
    , limits_(limits) {
    validateLimits(limits_);
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        const bool latencySensitive = index < LATENCY_WORKER_COUNT;
        workers_.emplace_back([this, latencySensitive] { workerLoop(latencySensitive); });
    }
}

FarTerrainScheduler::~FarTerrainScheduler() {
    shutdown();
}

bool FarTerrainScheduler::enqueue(FarTerrainKey key, uint32_t viewPriority) {
    return enqueueInternal(key, viewPriority, false);
}

bool FarTerrainScheduler::enqueueUrgentRefinement(FarTerrainKey key, uint32_t viewPriority) {
    if (farTerrainIsBaseStep(key.step)) return false;
    return enqueueInternal(key, viewPriority, true);
}

bool FarTerrainScheduler::enqueueInternal(FarTerrainKey key, uint32_t viewPriority,
                                          bool urgentRefinement) {
    if (!validStep(key.step) || !running_.load(std::memory_order_acquire)) return false;
    if (findCached(key)) return false;
    const uint64_t current = epoch_.load(std::memory_order_acquire);
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            inFlight_.load(std::memory_order_relaxed) >= limits_.maxPending) {
            return false;
        }
        if (urgentRefinement && urgentRefinementInFlightCount_.load(std::memory_order_relaxed) >=
                                    FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT) {
            return false;
        }
        if (wantedMembership_ && !wantedMembership_->keys.contains(key)) return false;
        auto active = activeKeys_.find(key);
        if (active != activeKeys_.end() && active->second == current) return false;
        activeKeys_[key] = current;
        const Job job{key, current, viewPriority, urgentRefinement};
        const auto before = [](const Job& first, const Job& second) {
            if (first.urgentRefinement != second.urgentRefinement) {
                return first.urgentRefinement;
            }
            return farTerrainSubmissionBefore(first.key, first.viewPriority, second.key,
                                              second.viewPriority);
        };
        const auto insertion = std::find_if(jobs_.begin(), jobs_.end(),
                                            [&](const Job& queued) { return before(job, queued); });
        jobs_.insert(insertion, job);
        (farTerrainIsBaseStep(key.step) ? queuedBaseCount_ : queuedRefinementCount_)
            .fetch_add(1, std::memory_order_relaxed);
        if (urgentRefinement) {
            queuedUrgentRefinementCount_.fetch_add(1, std::memory_order_relaxed);
            urgentRefinementInFlightCount_.fetch_add(1, std::memory_order_relaxed);
        }
        inFlight_.fetch_add(1, std::memory_order_relaxed);
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }
    jobCv_.notify_one();
    return true;
}

bool FarTerrainScheduler::hasSubmissionCapacity() const noexcept {
    return running_.load(std::memory_order_acquire) &&
           inFlight_.load(std::memory_order_relaxed) < limits_.maxPending;
}

bool FarTerrainScheduler::hasUrgentRefinementCapacity() const noexcept {
    return hasSubmissionCapacity() &&
           urgentRefinementInFlightCount_.load(std::memory_order_relaxed) <
               FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT;
}

void FarTerrainScheduler::setWorkerBudget(size_t budget) {
    const size_t clamped = std::clamp<size_t>(budget, 1, WORKER_COUNT);
    {
        std::lock_guard lock(jobMutex_);
        if (workerBudget_ == clamped) return;
        workerBudget_ = clamped;
        workerBudgetSnapshot_.store(clamped, std::memory_order_release);
    }
    jobCv_.notify_all();
}

bool FarTerrainScheduler::retainWanted(
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
    const std::vector<FarTerrainKey>& nearestFirst) {
    {
        std::lock_guard lock(jobMutex_);
        if (wantedMembership_ && wantedMembership_->keys == wanted) {
            wantedNoops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    auto membership = std::make_shared<ResidencyMembership>();
    membership->keys = wanted;
    membership->nearestFirst = nearestFirst;
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    {
        std::lock_guard lock(jobMutex_);
        if (wantedMembership_ && wantedMembership_->keys == wanted) {
            wantedNoops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        membership->revision = ++nextWantedRevision_;
        if (wantedMembership_) retiredMemberships_.push_back(std::move(wantedMembership_));
        wantedMembership_ = membership;
        residencyMaintenanceRequested_ = true;
        std::erase_if(jobs_, [&](const Job& job) {
            if (membership->keys.contains(job.key)) return false;
            const auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
            if (farTerrainIsBaseStep(job.key.step)) {
                ++removedBase;
            } else {
                ++removedRefinement;
            }
            if (job.urgentRefinement) ++removedUrgentRefinement;
            ++removed;
            return true;
        });
    }
    wantedUpdates_.fetch_add(1, std::memory_order_relaxed);
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement,
                                                 std::memory_order_relaxed);
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(completedMutex_);
        size_t completedBaseRemoved = 0;
        size_t completedRefinementRemoved = 0;
        std::erase_if(completed_, [&](const FarTerrainResult& result) {
            if (membership->keys.contains(result.key)) return false;
            if (farTerrainIsBaseStep(result.key.step)) {
                ++completedBaseRemoved;
            } else {
                ++completedRefinementRemoved;
            }
            return true;
        });
        completedBaseCount_.fetch_sub(completedBaseRemoved, std::memory_order_relaxed);
        completedRefinementCount_.fetch_sub(completedRefinementRemoved, std::memory_order_relaxed);
    }
    maintenancePendingSnapshot_.store(1, std::memory_order_release);
    jobCv_.notify_one();
    return true;
}

uint64_t FarTerrainScheduler::advanceEpoch() {
    const uint64_t next = epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    {
        std::lock_guard lock(jobMutex_);
        removed = jobs_.size();
        for (const Job& job : jobs_) {
            if (farTerrainIsBaseStep(job.key.step)) {
                ++removedBase;
            } else {
                ++removedRefinement;
            }
            if (job.urgentRefinement) ++removedUrgentRefinement;
            auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
        }
        jobs_.clear();
    }
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement,
                                                 std::memory_order_relaxed);
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(completedMutex_);
        completed_.clear();
        completedBaseCount_.store(0, std::memory_order_relaxed);
        completedRefinementCount_.store(0, std::memory_order_relaxed);
    }
    return next;
}

void FarTerrainScheduler::drainCompleted(std::vector<FarTerrainResult>& output) {
    std::lock_guard lock(completedMutex_);
    while (!completed_.empty()) {
        if (farTerrainIsBaseStep(completed_.front().key.step)) {
            completedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        } else {
            completedRefinementCount_.fetch_sub(1, std::memory_order_relaxed);
        }
        output.push_back(std::move(completed_.front()));
        completed_.pop_front();
    }
}

std::shared_ptr<const FarTerrainMesh> FarTerrainScheduler::findCached(FarTerrainKey key) const {
    std::lock_guard lock(cacheMutex_);
    auto found = cache_.find(key);
    if (found == cache_.end()) return {};
    found->second.lastAccess = ++accessClock_;
    cacheHits_.fetch_add(1, std::memory_order_relaxed);
    return found->second.mesh;
}

void FarTerrainScheduler::findCachedBatch(
    std::span<const FarTerrainKey> keys, size_t maximumResults,
    std::vector<std::shared_ptr<const FarTerrainMesh>>& output) const {
    output.clear();
    output.reserve(std::min(keys.size(), maximumResults));
    if (maximumResults == 0) return;
    std::lock_guard lock(cacheMutex_);
    for (const FarTerrainKey key : keys) {
        auto found = cache_.find(key);
        if (found == cache_.end()) continue;
        found->second.lastAccess = ++accessClock_;
        output.push_back(found->second.mesh);
        if (output.size() >= maximumResults) break;
    }
    cacheHits_.fetch_add(output.size(), std::memory_order_relaxed);
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainScheduler::findFinestCached(ColumnPos coordinate, FarTerrainStep displayed,
                                      FarTerrainStep desired, FarTerrainStepMask residentSteps,
                                      bool transitionActive) const {
    if (transitionActive) return {};
    std::lock_guard lock(cacheMutex_);
    FarTerrainStepMask available = residentSteps;
    const FarTerrainRefinementOrder order = farTerrainRefinementOrder(desired);
    for (FarTerrainStep step : std::span(order.steps).first(order.count)) {
        if (cache_.contains({coordinate.x, coordinate.z, step}))
            available |= farTerrainStepMask(step);
    }
    const std::optional<FarTerrainStep> target =
        farTerrainReadyTransitionTarget(displayed, desired, available, false);
    if (!target || (residentSteps & farTerrainStepMask(*target)) != 0) return {};
    auto found = cache_.find({coordinate.x, coordinate.z, *target});
    if (found == cache_.end()) return {};
    found->second.lastAccess = ++accessClock_;
    cacheHits_.fetch_add(1, std::memory_order_relaxed);
    return found->second.mesh;
}

void FarTerrainScheduler::findFinestCachedBatch(
    std::span<const FarTerrainRefinementCacheRequest> requests, size_t maximumResults,
    std::vector<std::shared_ptr<const FarTerrainMesh>>& output) const {
    output.clear();
    output.reserve(std::min(requests.size(), maximumResults));
    if (maximumResults == 0) return;
    std::lock_guard lock(cacheMutex_);
    for (const FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive) continue;
        FarTerrainStepMask available = request.residentSteps;
        const FarTerrainRefinementOrder order = farTerrainRefinementOrder(request.desired);
        for (FarTerrainStep step : std::span(order.steps).first(order.count)) {
            if (cache_.contains({request.coordinate.x, request.coordinate.z, step})) {
                available |= farTerrainStepMask(step);
            }
        }
        const std::optional<FarTerrainStep> target =
            farTerrainReadyTransitionTarget(request.displayed, request.desired, available, false);
        if (!target || (request.residentSteps & farTerrainStepMask(*target)) != 0) continue;
        if (request.requiresBlockScaleFallback && *target != FarTerrainStep::TWO) continue;
        if (request.requiresFineFallback &&
            farTerrainStepSize(*target) > farTerrainStepSize(FarTerrainStep::EIGHT)) {
            continue;
        }
        if (request.deferIntermediate && *target != request.desired) continue;
        auto found = cache_.find({request.coordinate.x, request.coordinate.z, *target});
        if (found == cache_.end()) continue;
        found->second.lastAccess = ++accessClock_;
        output.push_back(found->second.mesh);
        if (output.size() >= maximumResults) break;
    }
    cacheHits_.fetch_add(output.size(), std::memory_order_relaxed);
}

void FarTerrainScheduler::clearCache() {
    std::lock_guard lock(cacheMutex_);
    cache_.clear();
    cacheMaintenanceQueue_.clear();
    cacheMaintenanceRemaining_ = 0;
    cacheBytes_ = 0;
    cacheBaseEntries_ = 0;
    maintenancePendingSnapshot_.store(0, std::memory_order_release);
    publishCacheStatsLocked();
}

FarTerrainSchedulerStats FarTerrainScheduler::stats() const {
    FarTerrainSchedulerStats result;
    result.inFlight = inFlight_.load(std::memory_order_acquire);
    result.epoch = epoch_.load(std::memory_order_relaxed);
    result.submitted = submitted_.load(std::memory_order_relaxed);
    result.built = built_.load(std::memory_order_relaxed);
    result.canceled = canceled_.load(std::memory_order_relaxed);
    result.failed = failed_.load(std::memory_order_relaxed);
    result.cacheHits = cacheHits_.load(std::memory_order_relaxed);
    result.wantedUpdates = wantedUpdates_.load(std::memory_order_relaxed);
    result.wantedNoops = wantedNoops_.load(std::memory_order_relaxed);
    result.maintenancePending = maintenancePendingSnapshot_.load(std::memory_order_relaxed);
    result.maintenancePasses = maintenancePasses_.load(std::memory_order_relaxed);
    result.maintenanceScanned = maintenanceScanned_.load(std::memory_order_relaxed);
    result.maintenanceEvicted = maintenanceEvicted_.load(std::memory_order_relaxed);
    result.maintenanceBytes = maintenanceBytes_.load(std::memory_order_relaxed);
    result.maximumMaintenanceScanned = maximumMaintenanceScanned_.load(std::memory_order_relaxed);
    result.maximumMaintenanceBytes = maximumMaintenanceBytes_.load(std::memory_order_relaxed);
    result.queuedBase = queuedBaseCount_.load(std::memory_order_relaxed);
    result.queuedRefinement = queuedRefinementCount_.load(std::memory_order_relaxed);
    result.queuedUrgentRefinement = queuedUrgentRefinementCount_.load(std::memory_order_relaxed);
    result.activeUrgentRefinement =
        activeUrgentRefinementCountSnapshot_.load(std::memory_order_relaxed);
    result.urgentRefinementInFlight =
        urgentRefinementInFlightCount_.load(std::memory_order_relaxed);
    result.queued = result.queuedBase + result.queuedRefinement;
    result.activeWorkers = activeWorkerCountSnapshot_.load(std::memory_order_relaxed);
    result.activeBaseWorkers = activeBaseWorkerCountSnapshot_.load(std::memory_order_relaxed);
    result.workerBudget = workerBudgetSnapshot_.load(std::memory_order_relaxed);
    result.reservedBaseWorkers =
        farTerrainBaseWorkerReservation(result.workerBudget, result.queuedBase > 0);
    result.completedBase = completedBaseCount_.load(std::memory_order_relaxed);
    result.completedRefinement = completedRefinementCount_.load(std::memory_order_relaxed);
    result.completed = result.completedBase + result.completedRefinement;
    result.cacheEntries = cacheEntryCount_.load(std::memory_order_relaxed);
    result.cacheBytes = cacheBytesSnapshot_.load(std::memory_order_relaxed);
    result.cacheBaseEntries = cacheBaseEntryCount_.load(std::memory_order_relaxed);
    return result;
}

FarTerrainGenerationCacheStats FarTerrainScheduler::generationCacheStats() const {
    if (!generator_) return {};
    const worldgen::BasinCacheMetrics basin = generator_->basinCacheMetrics();
    const worldgen::MacroControlCacheMetrics macroControl = generator_->macroControlCacheMetrics();
    const worldgen::MacroControlCacheMetrics farClimate =
        generator_->farClimateControlCacheMetrics();
    return {
        .entries =
            basin.entries + basin.shorelineEntries + macroControl.entries + farClimate.entries,
        .bytes = basin.bytes + basin.shorelineBytes + macroControl.bytes + farClimate.bytes,
    };
}

void FarTerrainScheduler::shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    {
        std::lock_guard lock(jobMutex_);
        removed = jobs_.size();
        for (const Job& job : jobs_) {
            if (farTerrainIsBaseStep(job.key.step)) {
                ++removedBase;
            } else {
                ++removedRefinement;
            }
            if (job.urgentRefinement) ++removedUrgentRefinement;
        }
        jobs_.clear();
        activeKeys_.clear();
        residencyMaintenanceRequested_ = false;
    }
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement,
                                                 std::memory_order_relaxed);
        inFlight_.fetch_sub(removed, std::memory_order_relaxed);
        canceled_.fetch_add(removed, std::memory_order_relaxed);
    }
    jobCv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
}

void FarTerrainScheduler::finishJob(const Job& job) {
    {
        std::lock_guard lock(jobMutex_);
        auto active = activeKeys_.find(job.key);
        if (active != activeKeys_.end() && active->second == job.epoch) {
            activeKeys_.erase(active);
        }
        if (activeWorkerCount_ > 0) {
            --activeWorkerCount_;
            activeWorkerCountSnapshot_.store(activeWorkerCount_, std::memory_order_relaxed);
        }
        if (farTerrainIsBaseStep(job.key.step) && activeBaseWorkerCount_ > 0) {
            --activeBaseWorkerCount_;
            activeBaseWorkerCountSnapshot_.store(activeBaseWorkerCount_, std::memory_order_relaxed);
        }
        if (job.urgentRefinement && activeUrgentRefinementCount_ > 0) {
            --activeUrgentRefinementCount_;
            activeUrgentRefinementCountSnapshot_.store(activeUrgentRefinementCount_,
                                                       std::memory_order_relaxed);
        }
    }
    if (job.urgentRefinement) {
        urgentRefinementInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    inFlight_.fetch_sub(1, std::memory_order_release);
    jobCv_.notify_all();
}

void FarTerrainScheduler::storeCompleted(FarTerrainResult result) {
    std::lock_guard lock(completedMutex_);
    while (completed_.size() >= limits_.maxCompleted) {
        if (farTerrainIsBaseStep(completed_.front().key.step)) {
            completedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        } else {
            completedRefinementCount_.fetch_sub(1, std::memory_order_relaxed);
        }
        completed_.pop_front();
    }
    if (farTerrainIsBaseStep(result.key.step)) {
        completedBaseCount_.fetch_add(1, std::memory_order_relaxed);
    } else {
        completedRefinementCount_.fetch_add(1, std::memory_order_relaxed);
    }
    completed_.push_back(std::move(result));
}

void FarTerrainScheduler::publishCacheStatsLocked() {
    cacheEntryCount_.store(cache_.size(), std::memory_order_relaxed);
    cacheBaseEntryCount_.store(cacheBaseEntries_, std::memory_order_relaxed);
    cacheBytesSnapshot_.store(cacheBytes_, std::memory_order_release);
}

void FarTerrainScheduler::storeCache(std::shared_ptr<const FarTerrainMesh> mesh) {
    const size_t bytes = mesh->byteSize();
    const FarTerrainKey key = mesh->key;
    if (bytes > limits_.maxCacheBytes) return;
    std::vector<std::shared_ptr<const FarTerrainMesh>> retired;
    retired.reserve(2);
    bool stored = true;
    bool requestMaintenance = false;
    {
        std::lock_guard lock(cacheMutex_);
        auto retire = [&](auto entry) {
            cacheBytes_ -= entry->second.bytes;
            if (farTerrainIsBaseStep(entry->first.step)) --cacheBaseEntries_;
            retired.push_back(std::move(entry->second.mesh));
            cache_.erase(entry);
        };
        if (cacheMembership_ && !cacheMembership_->keys.contains(key)) stored = false;
        if (stored) {
            if (auto existing = cache_.find(key); existing != cache_.end()) retire(existing);
        }
        while (stored && !cache_.empty() &&
               (cache_.size() >= limits_.maxCacheEntries ||
                cacheBytes_ + bytes > limits_.maxCacheBytes)) {
            auto victim = cache_.end();
            for (auto candidate = cache_.begin(); candidate != cache_.end(); ++candidate) {
                if (victim == cache_.end()) {
                    victim = candidate;
                    continue;
                }
                const bool candidatePinned = pinnedBaseKeys_.contains(candidate->first);
                const bool victimPinned = pinnedBaseKeys_.contains(victim->first);
                if (candidatePinned != victimPinned) {
                    if (!candidatePinned) victim = candidate;
                    continue;
                }
                const auto candidatePriority = cachePriorities_.find(candidate->first);
                const auto victimPriority = cachePriorities_.find(victim->first);
                const uint32_t candidateRank = candidatePriority == cachePriorities_.end()
                                                   ? std::numeric_limits<uint32_t>::max()
                                                   : candidatePriority->second;
                const uint32_t victimRank = victimPriority == cachePriorities_.end()
                                                ? std::numeric_limits<uint32_t>::max()
                                                : victimPriority->second;
                if (candidateRank > victimRank ||
                    (candidateRank == victimRank &&
                     candidate->second.lastAccess < victim->second.lastAccess)) {
                    victim = candidate;
                }
            }
            if (pinnedBaseKeys_.contains(victim->first) && !pinnedBaseKeys_.contains(key)) {
                stored = false;
                break;
            }
            retire(victim);
        }
        if (stored) {
            const uint64_t token = ++cacheMaintenanceTokenClock_;
            cacheBytes_ += bytes;
            cache_.emplace(key, CacheEntry{std::move(mesh), bytes, ++accessClock_, token});
            cacheMaintenanceQueue_.push_back({key, token});
            if (cacheMembership_ &&
                cacheMaintenanceQueue_.size() > cache_.size() + limits_.maxMaintenanceEntries) {
                cacheMaintenanceRemaining_ =
                    std::max(cacheMaintenanceRemaining_, cacheMaintenanceQueue_.size());
                maintenancePendingSnapshot_.store(cacheMaintenanceRemaining_,
                                                  std::memory_order_release);
                requestMaintenance = true;
            }
            if (farTerrainIsBaseStep(key.step)) ++cacheBaseEntries_;
        }
        publishCacheStatsLocked();
    }
    if (requestMaintenance) requestResidencyMaintenance();
}

void FarTerrainScheduler::requestResidencyMaintenance() {
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed)) return;
        residencyMaintenanceRequested_ = true;
    }
    jobCv_.notify_one();
}

bool FarTerrainScheduler::performResidencyMaintenance() {
    std::shared_ptr<const ResidencyMembership> membership;
    std::deque<std::shared_ptr<const ResidencyMembership>> retiredJobMemberships;
    {
        std::lock_guard lock(jobMutex_);
        membership = wantedMembership_;
        retiredJobMemberships.swap(retiredMemberships_);
    }
    if (!membership) {
        maintenancePendingSnapshot_.store(0, std::memory_order_release);
        return false;
    }

    bool needsPriorityBuild = false;
    {
        std::lock_guard lock(cacheMutex_);
        needsPriorityBuild =
            !cacheMembership_ || cacheMembership_->revision != membership->revision;
    }
    std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> priorities;
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> pinnedBases;
    if (needsPriorityBuild) {
        priorities.reserve(membership->nearestFirst.size());
        for (size_t index = 0; index < membership->nearestFirst.size(); ++index) {
            priorities.try_emplace(
                membership->nearestFirst[index],
                static_cast<uint32_t>(std::min(index, static_cast<size_t>(UINT32_MAX))));
        }
        pinnedBases.reserve(membership->keys.size());
        for (const FarTerrainKey key : membership->keys) {
            if (farTerrainIsBaseStep(key.step)) pinnedBases.insert(key);
        }
    }

    std::vector<std::shared_ptr<const FarTerrainMesh>> retired;
    retired.reserve(limits_.maxMaintenanceEntries);
    std::shared_ptr<const ResidencyMembership> retiredMembership;
    size_t scanned = 0;
    size_t evicted = 0;
    size_t retiredBytes = 0;
    bool more = false;
    {
        // Lock order is job then cache. The bounded cache pass never destroys
        // a mesh while either lock is held.
        std::lock_guard jobLock(jobMutex_);
        if (!wantedMembership_ || wantedMembership_->revision != membership->revision) return true;
        std::lock_guard cacheLock(cacheMutex_);
        if (!cacheMembership_ || cacheMembership_->revision != membership->revision) {
            retiredMembership = std::move(cacheMembership_);
            cacheMembership_ = membership;
            cachePriorities_.swap(priorities);
            pinnedBaseKeys_.swap(pinnedBases);
            cacheMaintenanceRemaining_ = cacheMaintenanceQueue_.size();
        }
        while (cacheMaintenanceRemaining_ > 0 && scanned < limits_.maxMaintenanceEntries &&
               !cacheMaintenanceQueue_.empty()) {
            const CacheMaintenanceItem item = cacheMaintenanceQueue_.front();
            auto found = cache_.find(item.key);
            const bool current =
                found != cache_.end() && found->second.maintenanceToken == item.token;
            if (current && !cacheMembership_->keys.contains(item.key) && evicted > 0 &&
                retiredBytes + found->second.bytes > limits_.maxMaintenanceBytes) {
                break;
            }
            cacheMaintenanceQueue_.pop_front();
            --cacheMaintenanceRemaining_;
            ++scanned;
            if (!current) continue;
            if (cacheMembership_->keys.contains(item.key)) {
                cacheMaintenanceQueue_.push_back(item);
                continue;
            }
            retiredBytes += found->second.bytes;
            cacheBytes_ -= found->second.bytes;
            if (farTerrainIsBaseStep(found->first.step)) --cacheBaseEntries_;
            retired.push_back(std::move(found->second.mesh));
            cache_.erase(found);
            ++evicted;
        }
        if (cacheMaintenanceQueue_.empty()) cacheMaintenanceRemaining_ = 0;
        maintenancePendingSnapshot_.store(cacheMaintenanceRemaining_, std::memory_order_release);
        publishCacheStatsLocked();
        more = cacheMaintenanceRemaining_ > 0;
    }

    maintenancePasses_.fetch_add(1, std::memory_order_relaxed);
    maintenanceScanned_.fetch_add(scanned, std::memory_order_relaxed);
    maintenanceEvicted_.fetch_add(evicted, std::memory_order_relaxed);
    maintenanceBytes_.fetch_add(retiredBytes, std::memory_order_relaxed);
    recordAtomicMaximum(maximumMaintenanceScanned_, scanned);
    recordAtomicMaximum(maximumMaintenanceBytes_, retiredBytes);
    return more;
}

FarTerrainScheduler::Job FarTerrainScheduler::takeNextJobLocked() {
    auto selected = jobs_.begin();
    const auto urgent = std::find_if(jobs_.begin(), jobs_.end(),
                                     [](const Job& queued) { return queued.urgentRefinement; });
    const auto base = std::find_if(jobs_.begin(), jobs_.end(), [](const Job& queued) {
        return farTerrainIsBaseStep(queued.key.step);
    });
    const bool baseQueued = base != jobs_.end();
    const size_t reservedBaseWorkers = farTerrainBaseWorkerReservation(workerBudget_, baseQueued);
    if (baseQueued && activeBaseWorkerCount_ < reservedBaseWorkers) {
        selected = base;
    } else if (urgent != jobs_.end()) {
        const size_t urgentWorkerLimit = farTerrainUrgentWorkerLimit(workerBudget_, baseQueued);
        // Missing connected parents claim their reserved workers before an
        // urgent child. Without queued base work, urgent refinement may use
        // all otherwise idle capacity immediately.
        if (!baseQueued || activeUrgentRefinementCount_ < urgentWorkerLimit) {
            selected = urgent;
        } else {
            selected = base;
        }
    }
    const Job job = *selected;
    jobs_.erase(selected);
    (farTerrainIsBaseStep(job.key.step) ? queuedBaseCount_ : queuedRefinementCount_)
        .fetch_sub(1, std::memory_order_relaxed);
    if (job.urgentRefinement) {
        queuedUrgentRefinementCount_.fetch_sub(1, std::memory_order_relaxed);
        ++activeUrgentRefinementCount_;
        activeUrgentRefinementCountSnapshot_.store(activeUrgentRefinementCount_,
                                                   std::memory_order_relaxed);
    }
    if (farTerrainIsBaseStep(job.key.step)) {
        ++activeBaseWorkerCount_;
        activeBaseWorkerCountSnapshot_.store(activeBaseWorkerCount_, std::memory_order_relaxed);
    }
    return job;
}

void FarTerrainScheduler::workerLoop(bool latencySensitive) {
    setCurrentThreadPriority(latencySensitive ? ThreadPriority::USER_INITIATED
                                              : ThreadPriority::UTILITY);
    while (true) {
        Job job;
        bool hasJob = false;
        bool runMaintenance = false;
        {
            std::unique_lock lock(jobMutex_);
            jobCv_.wait(lock, [this] {
                return (activeWorkerCount_ < workerBudget_ &&
                        (!jobs_.empty() ||
                         (residencyMaintenanceRequested_ && !residencyMaintenanceActive_))) ||
                       !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_relaxed)) return;
            if (activeWorkerCount_ >= workerBudget_) continue;
            if (residencyMaintenanceRequested_ && !residencyMaintenanceActive_) {
                residencyMaintenanceRequested_ = false;
                residencyMaintenanceActive_ = true;
                runMaintenance = true;
            } else if (!jobs_.empty()) {
                job = takeNextJobLocked();
                hasJob = true;
            } else {
                continue;
            }
            ++activeWorkerCount_;
            activeWorkerCountSnapshot_.store(activeWorkerCount_, std::memory_order_relaxed);
        }
        if (runMaintenance) {
            bool more = false;
            try {
                more = performResidencyMaintenance();
            } catch (...) {
                failed_.fetch_add(1, std::memory_order_relaxed);
            }
            {
                std::lock_guard lock(jobMutex_);
                residencyMaintenanceActive_ = false;
                if (activeWorkerCount_ > 0) --activeWorkerCount_;
                activeWorkerCountSnapshot_.store(activeWorkerCount_, std::memory_order_relaxed);
                if (more && running_.load(std::memory_order_relaxed)) {
                    residencyMaintenanceRequested_ = true;
                }
            }
            jobCv_.notify_all();
            continue;
        }
        if (!hasJob) continue;
        if (job.epoch != epoch_.load(std::memory_order_acquire)) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }

        FarTerrainResult result;
        result.key = job.key;
        result.epoch = job.epoch;
        try {
            result.mesh = FarTerrainMesher::build(job.key, source_);
        } catch (...) {
            result.failed = true;
        }

        if (job.epoch != epoch_.load(std::memory_order_acquire) ||
            !running_.load(std::memory_order_acquire)) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }
        bool noLongerWanted = false;
        {
            std::lock_guard lock(jobMutex_);
            if (wantedMembership_ && !wantedMembership_->keys.contains(job.key)) {
                noLongerWanted = true;
            }
        }
        if (noLongerWanted) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }
        if (result.failed || !result.mesh) {
            failed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            storeCache(result.mesh);
            built_.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard lock(jobMutex_);
            noLongerWanted = wantedMembership_ && !wantedMembership_->keys.contains(job.key);
        }
        if (noLongerWanted) {
            canceled_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
        }
        storeCompleted(std::move(result));
        finishJob(job);
    }
}
