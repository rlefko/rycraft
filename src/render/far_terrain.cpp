#include "render/far_terrain.hpp"

#include "common/error.hpp"
#include "common/thread_priority.hpp"
#include "common/trace.hpp"
#include "render/block_textures.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/learned_terrain.hpp"
#include "world/native_hydrology.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <numbers>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace {

class ScopedCanopyDiagnosticTimer {
public:
    explicit ScopedCanopyDiagnosticTimer(uint64_t& destination)
        : destination_(destination)
        , started_(std::chrono::steady_clock::now()) {}

    ~ScopedCanopyDiagnosticTimer() { finish(); }

    void finish() noexcept {
        if (finished_) return;
        const auto elapsed = std::chrono::steady_clock::now() - started_;
        destination_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
        finished_ = true;
    }

private:
    uint64_t& destination_;
    std::chrono::steady_clock::time_point started_;
    bool finished_ = false;
};

uint32_t boundedDiagnosticCount(size_t count) noexcept {
    return static_cast<uint32_t>(
        std::min(count, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
}

bool farCanopyJobDiagnosticsEnabled() noexcept {
    static const bool enabled = [] {
        const char* requested = std::getenv("RYCRAFT_FAR_CANOPY_DIAGNOSTICS");
        if (requested && *requested != '\0' && std::strcmp(requested, "0") != 0) return true;
        const char* capture = std::getenv("RYCRAFT_CAPTURE");
        return capture && *capture != '\0';
    }();
    return enabled;
}

constexpr int GENERATOR_FAR_GRID_APRON_CELLS = 1;
constexpr size_t FINAL_BASE_AUTHORITY_POLLS_PER_PUMP = 16;
static_assert(FAR_TERRAIN_TILE_EDGE % farTerrainStepSize(FAR_TERRAIN_BASE_STEP) == 0,
              "base far terrain must contain a whole number of cells");

enum class FarWaterKind : uint8_t {
    NONE,
    OCEAN,
    RIVER,
    LAKE,
    WETLAND,
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
    float maximumTerrain = 0.0F;
    BlockType material = BlockType::GRASS;
    bool flat = false;
    bool water = false;
    bool flatWater = false;
    bool discontinuousWater = false;
    bool waterTopologyPossible = false;
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
    hash = mix64(hash ^ static_cast<uint8_t>(mesh.authorityQuality));
    hash = mix64(hash ^ static_cast<uint8_t>(mesh.exactAuthorityCompatible));
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
    for (uint64_t boundaryHash : mesh.surfaceBoundary.heightHashes)
        hash = mix64(hash ^ boundaryHash);
    hash = mix64(hash ^ static_cast<uint8_t>(mesh.surfaceBoundary.valid));
    hash = mix64(hash ^ std::bit_cast<uint32_t>(mesh.complexity));
    return hash;
}

uint64_t hashSurfaceBoundary(std::span<const float> heights) {
    uint64_t hash = mix64(static_cast<uint64_t>(heights.size()));
    for (const float height : heights)
        hash = mix64(hash ^ std::bit_cast<uint32_t>(height));
    return hash;
}

uint64_t hashCanopyAttachment(const FarCanopyAttachment& attachment) {
    uint64_t hash = mix64(static_cast<uint64_t>(attachment.key.tileX));
    hash = mix64(hash ^ static_cast<uint64_t>(attachment.key.tileZ));
    hash = mix64(hash ^ static_cast<uint8_t>(attachment.key.step));
    hash = mix64(hash ^ static_cast<uint8_t>(attachment.authorityQuality));
    hash = mix64(hash ^ static_cast<uint8_t>(attachment.groundingQuality));
    hash = mix64(hash ^ attachment.anchorIdentityHash);
    for (const Vertex& vertex : attachment.vertices) {
        hash = mix64(hash ^ vertex.faceAttr);
        hash = mix64(hash ^ (static_cast<uint64_t>(halfBits(vertex.px)) << 0U) ^
                     (static_cast<uint64_t>(halfBits(vertex.py)) << 16U) ^
                     (static_cast<uint64_t>(halfBits(vertex.pz)) << 32U));
        hash = mix64(hash ^ static_cast<uint64_t>(halfBits(vertex.u)) ^
                     (static_cast<uint64_t>(halfBits(vertex.v)) << 16U));
    }
    for (uint32_t index : attachment.indices)
        hash = mix64(hash ^ index);
    return hash;
}

uint64_t hashCanopyAnchors(std::span<const FarCanopy> canopies) {
    uint64_t hash = mix64(static_cast<uint64_t>(canopies.size()));
    for (const FarCanopy& canopy : canopies) {
        hash = mix64(hash ^ canopy.anchorId);
        hash = mix64(hash ^ static_cast<uint64_t>(canopy.x));
        hash = mix64(hash ^ static_cast<uint64_t>(canopy.z));
        hash = mix64(hash ^ static_cast<uint32_t>(canopy.baseY));
        hash = mix64(hash ^ static_cast<uint32_t>(canopy.topY));
        hash = mix64(hash ^ static_cast<uint32_t>(canopy.canopyMinimumY));
        hash = mix64(hash ^ static_cast<uint32_t>(canopy.canopyMaximumY));
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.canopyOffsetX));
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.canopyOffsetZ));
        hash = mix64(hash ^ canopy.canopyRadius);
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.species));
        hash = mix64(hash ^ static_cast<uint16_t>(canopy.logBlock));
        hash = mix64(hash ^ static_cast<uint16_t>(canopy.leafBlock));
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.aggregate));
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.formX));
        hash = mix64(hash ^ static_cast<uint8_t>(canopy.formZ));
        hash = mix64(hash ^ canopy.formExtent);
    }
    return hash;
}

uint64_t hashFloraAnchors(std::span<const FarFlora> flora) {
    uint64_t hash = mix64(static_cast<uint64_t>(flora.size()));
    for (const FarFlora& plant : flora) {
        hash = mix64(hash ^ plant.anchorId);
        hash = mix64(hash ^ static_cast<uint64_t>(plant.x));
        hash = mix64(hash ^ static_cast<uint64_t>(plant.z));
        hash = mix64(hash ^ static_cast<uint32_t>(plant.baseY));
        hash = mix64(hash ^ static_cast<uint8_t>(plant.block));
        hash = mix64(hash ^ plant.height);
    }
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
    result.terrainHeight = worldgen::geometryTerrainHeight(sample);
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
    result.wetland = sample.hydrology.wetland;
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
    const FarTerrainGeometrySample geometry = geometryFromSurface(sample);
    return {
        .geometry = geometry,
        .footprintMinimumTerrainHeight = geometry.terrainHeight,
        .footprintMaximumTerrainHeight = geometry.terrainHeight,
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
    result.footprintMinimumTerrainHeight = result.geometry.terrainHeight - envelope;
    result.footprintMaximumTerrainHeight = result.geometry.terrainHeight + envelope;
    return result;
}

void includeCanonicalWaterFloor(FarSurfaceSample& coverage,
                                const worldgen::SurfaceSample& canonical) {
    const bool waterTopology = canonical.hydrology.ocean || canonical.hydrology.river ||
                               canonical.hydrology.lake || canonical.hydrology.wetland;
    const bool standingWater = waterTopology && worldgen::geometryTerrainHeight(canonical) <
                                                    canonical.hydrology.waterSurface - 0.01;
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
    uint64_t generatorSeed = 0;
    int64_t originX = 0;
    int64_t originZ = 0;
    int step = 0;
    int cellWidth = 0;
    int cellHeight = 0;
    worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1;
    std::vector<worldgen::SurfaceSample> vertices;
    std::vector<worldgen::SurfaceSample> canonicalVertices;
    std::vector<worldgen::SurfaceSample> centers;
    bool usesVolcanicWaterCandidates = false;
    std::vector<uint8_t> volcanicWaterCandidates;
    // A learned-authority query may defer after this thread-local batch has
    // received its request metadata but before every dependent grid is
    // populated. Reusing that partial batch would read empty canonical
    // vertices on the retry and could suppress the water parent entirely.
    bool prepared = false;

    bool matches(const ChunkGenerator* expectedGenerator, uint64_t expectedSeed,
                 int64_t expectedOriginX, int64_t expectedOriginZ, int expectedStep,
                 int expectedCellWidth, int expectedCellHeight,
                 worldgen::SurfaceFootprint expectedFootprint) const {
        return prepared && generator == expectedGenerator && originX == expectedOriginX &&
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

    bool volcanicWaterCandidate(int cellX, int cellZ) const {
        if (!usesVolcanicWaterCandidates) return false;
        return volcanicWaterCandidates[static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth) +
                                       static_cast<size_t>(cellX)] != 0;
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
    batch.prepared = false;
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
    if (generator->usesLearnedAuthority()) {
        // Load canonical native water through the coarse grid contract so
        // block-resolution stage reconciliation stays out of every step-2
        // and coarser bounds batch.
        std::vector<worldgen::HydrologySample> hydrology(batch.vertices.size());
        generator->sampleCoarseNativeHydrologyAuthorityGrid(
            originX, originZ, step, step, cellWidth + 1, cellHeight + 1, hydrology);
        batch.canonicalVertices.resize(batch.vertices.size());
        for (size_t index = 0; index < hydrology.size(); ++index) {
            batch.canonicalVertices[index].hydrology = hydrology[index];
            batch.canonicalVertices[index].terrainHeight = hydrology[index].surfaceElevation;
            batch.canonicalVertices[index].waterSurface = hydrology[index].waterSurface;
        }
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
    // Legacy canonical-water sampling already applies its volcanic overlay,
    // while v4's native raster deliberately remains pre-volcanic. Both modes
    // still need the exact crater-footprint marker: without it a narrow
    // caldera can lie entirely between step-32 terrain samples and suppress
    // the water page before its canonical cells are inspected.
    batch.usesVolcanicWaterCandidates = step == 32;
    if (batch.usesVolcanicWaterCandidates) {
        batch.volcanicWaterCandidates.resize(static_cast<size_t>(cellWidth) *
                                             static_cast<size_t>(cellHeight));
        generator->markVolcanicWaterCandidates(originX, originZ, step, cellWidth, cellHeight,
                                               batch.volcanicWaterCandidates);
    } else {
        batch.volcanicWaterCandidates.clear();
    }
    batch.prepared = true;
}

struct GeneratorSurfaceBounds {
    double terrainHeight = std::numeric_limits<double>::quiet_NaN();
    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    bool waterTopologyPossible = false;
    bool volcanicWaterPossible = false;
    bool waterfallPossible = false;
};

void includeGeneratorBoundsSample(GeneratorSurfaceBounds& bounds,
                                  const worldgen::SurfaceSample& filtered,
                                  const worldgen::SurfaceSample& canonical,
                                  worldgen::SurfaceFootprint footprint,
                                  bool includeBroadVolcanicWaterCandidate) {
    FarSurfaceSample bounded =
        boundedFarSampleFromSurface(filtered, footprint,
                                    ChunkGenerator::emittedSurfaceDetailAmplitude(
                                        filtered, footprintReliefSlopeEnvelope(footprint)));
    includeCanonicalWaterFloor(bounded, canonical);
    bounds.minimum = std::min(bounds.minimum, bounded.footprintMinimumTerrainHeight);
    bounds.maximum = std::max(bounds.maximum, bounded.footprintMaximumTerrainHeight);
    const worldgen::HydrologySample& hydrology = canonical.hydrology;
    const bool volcanicWaterPossible =
        includeBroadVolcanicWaterCandidate && (canonical.geology.hotspotInfluence > 1.0e-6 ||
                                               canonical.geology.volcanicActivity > 1.0e-6);
    bounds.volcanicWaterPossible = bounds.volcanicWaterPossible || volcanicWaterPossible;
    bounds.waterfallPossible = bounds.waterfallPossible || hydrology.waterfall;
    // Wet corners are represented directly by the far sample grid. This bit
    // is reserved for water that can cross a cell even when every coarse
    // representative is dry, or for an interior contour that must prevent a
    // uniform coarse-water fill from covering an island. Marking every ocean
    // cell here would make an otherwise uniform horizon expand to block
    // resolution at step 32.
    bounds.waterTopologyPossible =
        bounds.waterTopologyPossible || hydrology.waterfall || volcanicWaterPossible;
}

void includeInteriorHydrologyBound(GeneratorSurfaceBounds& bounds,
                                   const worldgen::SurfaceSample& sample, int step) {
    const worldgen::HydrologySample& hydrology = sample.hydrology;
    const double reach = static_cast<double>(step) * std::numbers::sqrt2 * 0.5;
    if (hydrology.streamOrder > 0 && hydrology.channelWidth > 0.0 &&
        hydrology.channelDistance <= hydrology.channelWidth * 0.55 + reach) {
        bounds.waterTopologyPossible = true;
        // A native routed channel is the only surface that can own a
        // waterfall anchor. Preserve this bounded candidate even if the
        // anchor itself falls between the displayed far-terrain samples.
        bounds.waterfallPossible = true;
        const double channelDepth = std::max({hydrology.channelDepth, hydrology.erosionDepth, 1.0});
        bounds.minimum =
            std::min(bounds.minimum,
                     std::min(hydrology.surfaceElevation, hydrology.waterSurface) - channelDepth);
    }
    // WaterBodyId is also assigned to oceans and routed channels. Treating
    // every nonzero identity as a lake forced whole open-water parents into
    // block-resolution shoreline recovery. Lake-specific signed distance is
    // authoritative only for standing bodies.
    const bool knownLake = hydrology.lake || hydrology.lakeAreaSquareKilometers > 0.0;
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
    if (hydrology.wetland && hydrology.waterSurface > hydrology.surfaceElevation + 0.01) {
        // A wetland never expands beyond a direct parent-owned native cell,
        // but a shallow fringe can still sit entirely between a coarse cell's
        // terrain representatives. Request the canonical water raster rather
        // than converting it into a square terrain-corner sheet.
        bounds.waterTopologyPossible = true;
        bounds.minimum = std::min(bounds.minimum, hydrology.surfaceElevation);
    }
    // A dry sample close to sea level can contain a narrow inlet that misses
    // every displayed corner. A wet sample does not need this conservative
    // bit: the canonical water page already represents its body and stage,
    // and marking shallow open ocean made whole horizons expand to blocks.
    const double seaContourReach = reach + FAR_TERRAIN_EMITTED_SURFACE_ENVELOPE;
    const double surfaceElevation = hydrology.surfaceElevation;
    if (!hydrology.ocean && surfaceElevation <= SEA_LEVEL + seaContourReach) {
        bounds.waterTopologyPossible = true;
    }
    if (hydrology.waterfall && hydrology.waterfallTop >= hydrology.waterfallBottom + 0.5) {
        bounds.waterTopologyPossible = true;
        bounds.waterfallPossible = true;
        bounds.minimum = std::min(bounds.minimum, std::ceil(hydrology.waterfallBottom) - 1.0);
        bounds.maximum = std::max(bounds.maximum, std::ceil(hydrology.waterfallTop));
    }
}

GeneratorSurfaceBounds generatorCellSurfaceBounds(const GeneratorCellBoundsBatch& batch, int cellX,
                                                  int cellZ, bool includeCenter = true) {
    GeneratorSurfaceBounds bounds;
    const bool includeBroadVolcanicWaterCandidate = !batch.usesVolcanicWaterCandidates;
    if (batch.step == 1) {
        bounds.terrainHeight = worldgen::geometryTerrainHeight(batch.vertex(cellX, cellZ));
        includeGeneratorBoundsSample(bounds, batch.vertex(cellX, cellZ),
                                     batch.canonicalVertex(cellX, cellZ), batch.footprint,
                                     includeBroadVolcanicWaterCandidate);
        return bounds;
    }
    constexpr std::array<std::array<int, 2>, 4> CORNERS = {
        {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}}};
    for (const auto& corner : CORNERS) {
        const worldgen::SurfaceSample& filtered =
            batch.vertex(cellX + corner[0], cellZ + corner[1]);
        const worldgen::SurfaceSample& canonical =
            batch.canonicalVertex(cellX + corner[0], cellZ + corner[1]);
        includeGeneratorBoundsSample(bounds, filtered, canonical, batch.footprint,
                                     includeBroadVolcanicWaterCandidate);
        includeInteriorHydrologyBound(bounds, canonical, batch.step);
    }
    if (includeCenter) {
        const worldgen::SurfaceSample* center = batch.center(cellX, cellZ);
        if (center != nullptr) {
            bounds.terrainHeight = worldgen::geometryTerrainHeight(*center);
            includeGeneratorBoundsSample(bounds, *center, *center, batch.footprint,
                                         includeBroadVolcanicWaterCandidate);
            includeInteriorHydrologyBound(bounds, *center, batch.step);
        }
    }
    if (!std::isfinite(bounds.terrainHeight)) {
        double terrainHeight = 0.0;
        for (const auto& corner : CORNERS) {
            terrainHeight +=
                worldgen::geometryTerrainHeight(batch.vertex(cellX + corner[0], cellZ + corner[1]));
        }
        bounds.terrainHeight = terrainHeight / static_cast<double>(CORNERS.size());
    }
    const double interiorEnvelope = footprintInteriorEnvelope(batch.footprint);
    bounds.minimum -= interiorEnvelope;
    bounds.maximum += interiorEnvelope;
    if (batch.volcanicWaterCandidate(cellX, cellZ)) {
        // A direct crater-footprint marker always requests water recovery.
        // V3's canonical raster already includes its crater overlay, whereas
        // v4 must promote the cell to final geometry because its inexpensive
        // native raster intentionally precedes volcanic postprocessing.
        bounds.volcanicWaterPossible =
            bounds.volcanicWaterPossible || batch.generator->usesLearnedAuthority();
        bounds.waterTopologyPossible = true;
    }
    return bounds;
}

bool hasWater(const FarTerrainGeometrySample& sample) {
    return sample.ocean || sample.river || sample.lake || sample.wetland;
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

float stepOneTerrainCellHeight(const FarSurfaceSample& sample) {
    // One far cell represents the exact block column at its lower-left world
    // coordinate. Averaging its neighbors rounded shorelines and cliffs into a
    // different surface immediately outside exact residency, despite both
    // tiers having one-block horizontal detail.
    return vertexHeight(std::floor(sample.geometry.terrainHeight + 0.5));
}

float terrainCellHeight(const std::array<FarSurfaceSample*, 4>& corners, FarTerrainStep step) {
    if (step == FarTerrainStep::ONE) return stepOneTerrainCellHeight(*corners.front());

    // Reduced-resolution cells display the filtered field itself. Bounds
    // remain conservative for culling, but they cannot depress the
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
    bool waterTopologyPossible = false;
    bool volcanicWaterPossible = false;
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
        .waterTopologyPossible = false,
        .volcanicWaterPossible = false,
    };
}

ResolvedFarTerrainCellBounds authoritativeCellBounds(const FarTerrainCellBounds& bounds) {
    if (!std::isfinite(bounds.terrainHeight) || !std::isfinite(bounds.minimumTerrainHeight) ||
        !std::isfinite(bounds.maximumTerrainHeight) ||
        bounds.maximumTerrainHeight < bounds.minimumTerrainHeight ||
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
        .waterTopologyPossible = bounds.waterTopologyPossible,
        .volcanicWaterPossible = bounds.volcanicWaterPossible,
    };
}

FarWaterKind farWaterKind(const FarTerrainGeometrySample& sample) {
    if (sample.lake) return FarWaterKind::LAKE;
    if (sample.ocean) return FarWaterKind::OCEAN;
    if (sample.river) return FarWaterKind::RIVER;
    if (sample.wetland) return FarWaterKind::WETLAND;
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

class WaterTopologyTracker {
public:
    void observe(const FarTerrainGeometrySample& sample) {
        const FarWaterKind kind = farWaterKind(sample);
        if (kind == FarWaterKind::NONE) return;
        const uint64_t body = mix64(sample.waterBodyId ^ (static_cast<uint64_t>(kind) << 56U));
        bodies_.insert(body);
        if (sample.transitionOwnerKind == worldgen::WaterTransitionKind::NONE ||
            sample.transitionOwnerId == 0) {
            return;
        }
        transitions_.insert(mix64(body ^ sample.transitionOwnerId ^
                                  (static_cast<uint64_t>(sample.transitionOwnerKind) << 48U)));
    }

    void observeConnection(const FarTerrainGeometrySample& first,
                           const FarTerrainGeometrySample& second) {
        const FarWaterKind firstKind = farWaterKind(first);
        const FarWaterKind secondKind = farWaterKind(second);
        if (firstKind == FarWaterKind::NONE || secondKind == FarWaterKind::NONE) return;
        const bool sameBody =
            first.waterBodyId != worldgen::NO_WATER_BODY && first.waterBodyId == second.waterBodyId;
        const bool sameOcean =
            firstKind == FarWaterKind::OCEAN && secondKind == FarWaterKind::OCEAN;
        const bool sameTransition =
            first.transitionOwnerId != 0 && first.transitionOwnerId == second.transitionOwnerId &&
            first.transitionOwnerKind != worldgen::WaterTransitionKind::NONE &&
            first.transitionOwnerKind == second.transitionOwnerKind;
        if (!sameBody && !sameOcean && !sameTransition) return;
        uint64_t firstBody = mix64(first.waterBodyId ^ (static_cast<uint64_t>(firstKind) << 56U));
        uint64_t secondBody =
            mix64(second.waterBodyId ^ (static_cast<uint64_t>(secondKind) << 56U));
        if (secondBody < firstBody) std::swap(firstBody, secondBody);
        connections_.insert(mix64(firstBody ^ std::rotl(secondBody, 17)));
    }

    FarTerrainWaterTopologySignature signature() const {
        FarTerrainWaterTopologySignature result;
        for (const uint64_t body : bodies_)
            result.bodyIdentityHash = mix64(result.bodyIdentityHash ^ body);
        for (const uint64_t transition : transitions_)
            result.transitionIdentityHash = mix64(result.transitionIdentityHash ^ transition);
        for (const uint64_t connection : connections_)
            result.connectivityHash = mix64(result.connectivityHash ^ connection);
        result.bodyIdentityCount = static_cast<uint32_t>(bodies_.size());
        result.transitionIdentityCount = static_cast<uint32_t>(transitions_.size());
        result.connectivityCount = static_cast<uint32_t>(connections_.size());
        return result;
    }

private:
    std::set<uint64_t> bodies_;
    std::set<uint64_t> transitions_;
    std::set<uint64_t> connections_;
};

bool waterAuthoritiesConnect(const FarWaterAuthority& first, const FarWaterAuthority& second,
                             float sampleDistance) {
    if (first.kind == FarWaterKind::NONE || second.kind == FarWaterKind::NONE) return false;
    const float heightDifference = std::abs(first.height - second.height);
    if (first.kind == FarWaterKind::WETLAND || second.kind == FarWaterKind::WETLAND) {
        // Wetlands do not define a new standing body. They may join only the
        // exact parent stage and identity assigned by native hydrology.
        return first.bodyId != worldgen::NO_WATER_BODY && first.bodyId == second.bodyId &&
               heightDifference <= 0.125F;
    }
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

template <typename HeightFunction>
void pushTerrainTransitionTop(FarTerrainMesh& mesh, BlockType material, float x0, float z0,
                              const FarTerrainTransitionTopology& topology,
                              HeightFunction&& heightAt) {
    const uint32_t attribute =
        packFaceAttr(FaceNormal::PLUS_Y, textureLayerFor(material, FaceNormal::PLUS_Y), 15) |
        FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK;
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    for (const FarTerrainTransitionVertex& point :
         std::span(topology.vertices).first(topology.vertexCount)) {
        const float x = x0 + static_cast<float>(point.x);
        const float z = z0 + static_cast<float>(point.z);
        const float height = heightAt(point);
        mesh.vertices.push_back(Vertex{attribute, static_cast<float16_t>(x),
                                       static_cast<float16_t>(height), static_cast<float16_t>(z),
                                       static_cast<float16_t>(x), static_cast<float16_t>(z)});
        updateYBounds(mesh, height);
        updateSurfaceYBounds(mesh, height);
    }
    for (const uint8_t index : std::span(topology.indices).first(topology.indexCount))
        mesh.indices.push_back(base + index);
    mesh.transitionTriangleCount += topology.indexCount / 3;
}

// Close only a genuine source-column discontinuity inside the tile's outer
// cell ring. This wedge never lies on a tile face and never bridges two LODs;
// the shared top strip is the sole owner of the cross-tile transition.
void pushTerrainDiscontinuityWedge(FarTerrainMesh& mesh, FaceNormal face, BlockType material,
                                   float x0, float z0, float x1, float z1, float low0, float low1,
                                   float high0, float high1) {
    constexpr float COLLAPSE_EPSILON = 1.0e-4F;
    const bool firstCollapsed = high0 <= low0 + COLLAPSE_EPSILON;
    const bool secondCollapsed = high1 <= low1 + COLLAPSE_EPSILON;
    if (firstCollapsed && secondCollapsed) return;
    const float width = std::hypot(x1 - x0, z1 - z0);
    const float height = std::max(high0 - low0, high1 - low1);
    const uint32_t attribute = packFaceAttr(face, textureLayerFor(material, face), 15) |
                               FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK;
    std::array<Corner, 4> corners = {{
        {x0, low0, z0, 0.0F, height},
        {x1, low1, z1, width, height},
        {x1, high1, z1, width, 0.0F},
        {x0, high0, z0, 0.0F, 0.0F},
    }};
    if (firstCollapsed != secondCollapsed) {
        // A transition strip converges to its shared boundary height at one
        // endpoint. Split the resulting triangle at the midpoint of the
        // opposite vertical edge. Splitting the diagonal from the collapsed
        // endpoint would leave one of the fixed two triangles at zero area.
        if (firstCollapsed) {
            const Corner highEndpoint = corners[2];
            corners[2] = {
                (corners[1].x + highEndpoint.x) * 0.5F,
                (corners[1].y + highEndpoint.y) * 0.5F,
                (corners[1].z + highEndpoint.z) * 0.5F,
                width,
                height * 0.5F,
            };
            corners[3] = highEndpoint;
        } else {
            const Corner collapsedEndpoint = corners[1];
            corners[1] = {
                (corners[0].x + corners[3].x) * 0.5F,
                (corners[0].y + corners[3].y) * 0.5F,
                (corners[0].z + corners[3].z) * 0.5F,
                0.0F,
                height * 0.5F,
            };
            corners[2] = collapsedEndpoint;
        }
    }
    pushQuad(mesh, attribute, corners, false);
    updateSurfaceYBounds(mesh, low0);
    updateSurfaceYBounds(mesh, low1);
    updateSurfaceYBounds(mesh, high0);
    updateSurfaceYBounds(mesh, high1);
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

struct FarCanopyGeometrySink {
    int64_t originX = 0;
    int64_t originZ = 0;
    FarTerrainBounds bounds;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t canopyAnchorCount = 0;
    uint32_t canopyImpostorQuadCount = 0;
    uint32_t floraAnchorCount = 0;
    uint32_t floraImpostorQuadCount = 0;
};

void pushCanopyQuad(FarCanopyGeometrySink& geometry, FaceNormal face, BlockType material,
                    const std::array<Corner, 4>& corners, bool flora = false) {
    const uint32_t attribute =
        packFaceAttr(face, textureLayerFor(material, face), 15) | FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK;
    const uint32_t base = static_cast<uint32_t>(geometry.vertices.size());
    for (const Corner& corner : corners) {
        const Vertex vertex{attribute,
                            static_cast<float16_t>(corner.x),
                            static_cast<float16_t>(corner.y),
                            static_cast<float16_t>(corner.z),
                            static_cast<float16_t>(corner.u),
                            static_cast<float16_t>(corner.v)};
        geometry.vertices.push_back(vertex);
        geometry.bounds.minY = std::min(geometry.bounds.minY, static_cast<float>(vertex.py));
        geometry.bounds.maxY = std::max(geometry.bounds.maxY, static_cast<float>(vertex.py));
        const int64_t worldX = geometry.originX + static_cast<int64_t>(std::floor(corner.x));
        const int64_t worldZ = geometry.originZ + static_cast<int64_t>(std::floor(corner.z));
        const int64_t worldMaxX = geometry.originX + static_cast<int64_t>(std::ceil(corner.x));
        const int64_t worldMaxZ = geometry.originZ + static_cast<int64_t>(std::ceil(corner.z));
        geometry.bounds.minX = std::min(geometry.bounds.minX, worldX);
        geometry.bounds.maxX = std::max(geometry.bounds.maxX, worldMaxX);
        geometry.bounds.minZ = std::min(geometry.bounds.minZ, worldZ);
        geometry.bounds.maxZ = std::max(geometry.bounds.maxZ, worldMaxZ);
    }
    geometry.indices.insert(geometry.indices.end(),
                            {base, base + 1, base + 2, base, base + 2, base + 3});
    if (flora) {
        // Crossed flora must remain visible from both sides even if an
        // attachment is rendered with back-face culling enabled.
        geometry.indices.insert(geometry.indices.end(),
                                {base + 2, base + 1, base, base + 3, base + 2, base});
        ++geometry.floraImpostorQuadCount;
    } else {
        ++geometry.canopyImpostorQuadCount;
    }
}

void pushCanopyPrism(FarCanopyGeometrySink& geometry, BlockType material, float centerX,
                     float centerZ, float halfWidthX, float halfWidthZ, float bottom, float top,
                     bool includeTop) {
    const float x0 = centerX - halfWidthX;
    const float x1 = centerX + halfWidthX;
    const float z0 = centerZ - halfWidthZ;
    const float z1 = centerZ + halfWidthZ;
    const float width = halfWidthX * 2.0F;
    const float depth = halfWidthZ * 2.0F;
    const float height = std::max(0.0F, top - bottom);
    if (height <= 0.0F || width <= 0.0F || depth <= 0.0F) return;
    pushCanopyQuad(geometry, FaceNormal::MINUS_X, material,
                   {{{x0, bottom, z0, 0.0F, height},
                     {x0, bottom, z1, depth, height},
                     {x0, top, z1, depth, 0.0F},
                     {x0, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(geometry, FaceNormal::PLUS_X, material,
                   {{{x1, bottom, z1, 0.0F, height},
                     {x1, bottom, z0, depth, height},
                     {x1, top, z0, depth, 0.0F},
                     {x1, top, z1, 0.0F, 0.0F}}});
    pushCanopyQuad(geometry, FaceNormal::MINUS_Z, material,
                   {{{x1, bottom, z0, 0.0F, height},
                     {x0, bottom, z0, width, height},
                     {x0, top, z0, width, 0.0F},
                     {x1, top, z0, 0.0F, 0.0F}}});
    pushCanopyQuad(geometry, FaceNormal::PLUS_Z, material,
                   {{{x0, bottom, z1, 0.0F, height},
                     {x1, bottom, z1, width, height},
                     {x1, top, z1, width, 0.0F},
                     {x0, top, z1, 0.0F, 0.0F}}});
    if (includeTop) {
        pushCanopyQuad(geometry, FaceNormal::PLUS_Y, material,
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
void pushCanopyStack(FarCanopyGeometrySink& geometry, BlockType material, float centerX,
                     float centerZ, float radius, float bottom, float top,
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
        pushCanopyPrism(geometry, material, centerX, centerZ, layerRadius, layerRadius, layerBottom,
                        layerTop, true);
    }
}

void pushSpeciesCanopy(FarCanopyGeometrySink& geometry, const FarCanopy& canopy, float centerX,
                       float centerZ, float radius, float bottom, float top) {
    using feature_generation::TreeSpecies;
    const TreeSpecies species = canopy.species;
    const BlockType leafBlock = canopy.leafBlock;
    switch (species) {
        case TreeSpecies::SPRUCE:
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
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
            pushCanopyPrism(geometry, leafBlock, centerX + directionX * primaryShift,
                            centerZ + directionZ * primaryShift, primaryRadius, primaryRadius,
                            bottom, std::min(split, top), true);
            const float secondaryRadius = steppedCanopyRadius(radius, 0.54F);
            pushCanopyPrism(geometry, leafBlock, centerX - directionZ * secondaryShift,
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
            pushCanopyPrism(geometry, leafBlock, centerX, centerZ, radius, frondThickness, bottom,
                            std::min(firstTop, top), true);
            pushCanopyPrism(geometry, leafBlock, centerX, centerZ, frondThickness, radius,
                            std::min(firstTop, top), secondTop, true);
            const float tuftRadius = steppedCanopyRadius(radius, 0.38F);
            pushCanopyPrism(geometry, leafBlock, centerX, centerZ, tuftRadius, tuftRadius,
                            secondTop, top, true);
            return;
        }
        case TreeSpecies::WILLOW: {
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.78F, 1.0F, 0.68F});
            const float tendrilRadius = steppedCanopyRadius(radius, 0.18F);
            const float offset = radius * 0.68F;
            const float tendrilTop = bottom + std::max(1.0F, (top - bottom) * 0.58F);
            for (const auto [dx, dz] : std::array{std::pair{1.0F, 0.0F}, std::pair{-1.0F, 0.0F},
                                                  std::pair{0.0F, 1.0F}, std::pair{0.0F, -1.0F}}) {
                pushCanopyPrism(geometry, leafBlock, centerX + dx * offset, centerZ + dz * offset,
                                tendrilRadius, tendrilRadius, bottom, tendrilTop, true);
            }
            return;
        }
        case TreeSpecies::JUNGLE: {
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.72F, 1.0F, 0.82F});
            const float branchRadius = steppedCanopyRadius(radius, 0.34F);
            const float branchOffset = radius * 0.46F;
            const float branchBottom = bottom + std::max(0.0F, (top - bottom) * 0.22F);
            const float branchTop =
                std::min(top, branchBottom + std::max(1.0F, (top - bottom) * 0.32F));
            for (const auto [dx, dz] : std::array{std::pair{1.0F, 0.0F}, std::pair{-1.0F, 0.0F},
                                                  std::pair{0.0F, 1.0F}, std::pair{0.0F, -1.0F}}) {
                pushCanopyPrism(geometry, leafBlock, centerX + dx * branchOffset,
                                centerZ + dz * branchOffset, branchRadius, branchRadius,
                                branchBottom, branchTop, true);
            }
            return;
        }
        case TreeSpecies::MANGROVE:
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.82F, 1.0F, 0.72F});
            return;
        case TreeSpecies::ALPINE_SCRUB:
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{1.0F, 0.55F});
            return;
        case TreeSpecies::LARGE_OAK:
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
                            std::array{0.58F, 0.92F, 1.0F, 0.64F});
            return;
        case TreeSpecies::OAK:
        case TreeSpecies::BIRCH:
            pushCanopyStack(geometry, leafBlock, centerX, centerZ, radius, bottom, top,
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

template <typename GroundFunction>
void appendCanopyGeometry(FarCanopyGeometrySink& geometry, std::span<const FarCanopy> canopies,
                          GroundFunction&& groundAt) {
    for (const FarCanopy& canopy : canopies) {
        // The anchor's half-open tile owns the complete layered crown. It may
        // cross the tile face without giving the neighboring tile duplicate
        // coplanar geometry.
        if (canopy.x < geometry.originX || canopy.x >= geometry.originX + FAR_TERRAIN_TILE_EDGE ||
            canopy.z < geometry.originZ || canopy.z >= geometry.originZ + FAR_TERRAIN_TILE_EDGE) {
            continue;
        }
        const float ground = groundAt(canopy);
        const float localAnchorX = static_cast<float>(canopy.x - geometry.originX);
        const float localAnchorZ = static_cast<float>(canopy.z - geometry.originZ);
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
                pushCanopyPrism(geometry, canopy.logBlock, centerX, centerZ, halfWidthX, halfWidthZ,
                                ground, ground + 1.0F, true);
            }
            ++geometry.canopyAnchorCount;
            continue;
        }
        const float canopyCenterX = localAnchorX + canopy.canopyOffsetX;
        const float canopyCenterZ = localAnchorZ + canopy.canopyOffsetZ;
        const float canopyBottom =
            ground + static_cast<float>(canopy.canopyMinimumY - canopy.baseY);
        const float canopyTop =
            ground + static_cast<float>(canopy.canopyMaximumY + 1 - canopy.baseY);
        const float canopyRadius = std::max(1.0F, static_cast<float>(canopy.canopyRadius) + 0.5F);
        const float trunkTop = canopyTrunkTop(canopy.leafBlock, canopyBottom, canopyTop);
        if (canopy.logBlock != BlockType::AIR && trunkTop > ground) {
            const float trunkHalfWidth = canopyTrunkHalfWidth(canopy.logBlock);
            pushCanopyPrism(geometry, canopy.logBlock, localAnchorX + 0.5F, localAnchorZ + 0.5F,
                            trunkHalfWidth, trunkHalfWidth, ground, trunkTop, false);
        }
        if (canopy.leafBlock != BlockType::AIR) {
            pushSpeciesCanopy(geometry, canopy, canopyCenterX + 0.5F, canopyCenterZ + 0.5F,
                              canopyRadius, canopyBottom, canopyTop);
        }
        ++geometry.canopyAnchorCount;
    }
}

template <typename GroundFunction>
void appendFarFloraGeometry(FarCanopyGeometrySink& geometry, std::span<const FarFlora> flora,
                            FarTerrainStep lodStep, GroundFunction&& groundAt) {
    constexpr std::array<std::array<int8_t, 2>, 4> AXES = {
        {{{5, 0}}, {{5, 3}}, {{5, 5}}, {{3, 5}}}};
    constexpr std::array<float, 4> JITTER = {-0.1875F, -0.0625F, 0.0625F, 0.1875F};
    constexpr std::array<std::array<float, 2>, 8> CLUMP_DIRECTIONS = {{{{0.75F, 0.0F}},
                                                                       {{0.5F, 0.5F}},
                                                                       {{0.0F, 0.75F}},
                                                                       {{-0.5F, 0.5F}},
                                                                       {{-0.75F, 0.0F}},
                                                                       {{-0.5F, -0.5F}},
                                                                       {{0.0F, -0.75F}},
                                                                       {{0.5F, -0.5F}}}};
    const int step = farTerrainStepSize(lodStep);
    const int clumpCount = step <= 4 ? 3 : step <= 8 ? 2 : 1;
    const float coarseWidthScale = step <= 8 ? 1.0F : step <= 16 ? 6.4F : 12.8F;
    for (const FarFlora& plant : flora) {
        if (plant.x < geometry.originX || plant.x >= geometry.originX + FAR_TERRAIN_TILE_EDGE ||
            plant.z < geometry.originZ || plant.z >= geometry.originZ + FAR_TERRAIN_TILE_EDGE ||
            !rendersAsCross(plant.block)) {
            continue;
        }
        const float localX = static_cast<float>(plant.x - geometry.originX);
        const float localZ = static_cast<float>(plant.z - geometry.originZ);
        const float bottom = groundAt(plant);
        const float height = static_cast<float>(std::max<uint8_t>(plant.height, 1));
        const float top = bottom + height;
        const uint64_t directionSeed = mix64(plant.anchorId ^ 0x434C554D50464C52ULL);
        const size_t directionIndex = static_cast<size_t>(directionSeed & 7U);
        for (int instance = 0; instance < clumpCount; ++instance) {
            const uint64_t pose = mix64(
                plant.anchorId ^ (static_cast<uint64_t>(instance + 1) * 0x9E37'79B9'7F4A'7C15ULL));
            const auto& axis = AXES[pose & 3U];
            float offsetX = 0.0F;
            float offsetZ = 0.0F;
            if (instance > 0) {
                const auto& direction =
                    CLUMP_DIRECTIONS[(directionIndex + static_cast<size_t>(instance - 1) * 3U) &
                                     7U];
                offsetX = direction[0];
                offsetZ = direction[1];
            }
            const float centerX = localX + 0.5F + offsetX + JITTER[(pose >> 8U) & 3U];
            const float centerZ = localZ + 0.5F + offsetZ + JITTER[(pose >> 16U) & 3U];
            const float axisX = static_cast<float>(axis[0]) * 0.0625F * coarseWidthScale;
            const float axisZ = static_cast<float>(axis[1]) * 0.0625F * coarseWidthScale;
            const float textureSpan =
                std::max(1.0F, 2.0F * std::sqrt(axisX * axisX + axisZ * axisZ));
            pushCanopyQuad(geometry, FaceNormal::CROSS, plant.block,
                           {{{centerX - axisX, bottom, centerZ - axisZ, 0.0F, height},
                             {centerX + axisX, bottom, centerZ + axisZ, textureSpan, height},
                             {centerX + axisX, top, centerZ + axisZ, textureSpan, 0.0F},
                             {centerX - axisX, top, centerZ - axisZ, 0.0F, 0.0F}}},
                           true);
            pushCanopyQuad(geometry, FaceNormal::CROSS, plant.block,
                           {{{centerX + axisZ, bottom, centerZ - axisX, 0.0F, height},
                             {centerX - axisZ, bottom, centerZ + axisX, textureSpan, height},
                             {centerX - axisZ, top, centerZ + axisX, textureSpan, 0.0F},
                             {centerX + axisZ, top, centerZ - axisX, 0.0F, 0.0F}}},
                           true);
        }
        ++geometry.floraAnchorCount;
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
    limits.maxCanopyPending = std::max<size_t>(1, limits.maxCanopyPending);
    limits.maxCanopyCompleted = std::max<size_t>(1, limits.maxCanopyCompleted);
    limits.maxCanopyCacheEntries = std::max<size_t>(1, limits.maxCanopyCacheEntries);
    limits.maxCanopyCacheBytes = std::max<size_t>(1, limits.maxCanopyCacheBytes);
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

bool farTerrainRefinementRequestBefore(const FarTerrainRefinementCacheRequest& first,
                                       const FarTerrainRefinementCacheRequest& second) {
    if (first.cameraTile != second.cameraTile) return first.cameraTile;
    if (first.protectedNearTarget != second.protectedNearTarget) {
        return first.protectedNearTarget;
    }
    if (first.requiresBlockScaleFallback != second.requiresBlockScaleFallback) {
        return first.requiresBlockScaleFallback;
    }
    if (first.requiresFineFallback != second.requiresFineFallback) {
        return first.requiresFineFallback;
    }
    if (first.displayableWavefront != second.displayableWavefront)
        return first.displayableWavefront;
    const double firstDistance = std::isfinite(first.distanceSquaredBlocks)
                                     ? std::max(0.0, first.distanceSquaredBlocks)
                                     : std::numeric_limits<double>::infinity();
    const double secondDistance = std::isfinite(second.distanceSquaredBlocks)
                                      ? std::max(0.0, second.distanceSquaredBlocks)
                                      : std::numeric_limits<double>::infinity();
    if (firstDistance != secondDistance) return firstDistance < secondDistance;
    if (first.visible != second.visible) return first.visible;
    const double firstError =
        std::isfinite(first.projectedErrorPixels) ? std::max(0.0, first.projectedErrorPixels) : 0.0;
    const double secondError = std::isfinite(second.projectedErrorPixels)
                                   ? std::max(0.0, second.projectedErrorPixels)
                                   : 0.0;
    if (firstError != secondError) return firstError > secondError;
    if (first.coordinate.x != second.coordinate.x) return first.coordinate.x < second.coordinate.x;
    return first.coordinate.z < second.coordinate.z;
}

} // namespace

bool farTerrainGpuUploadFitsArena(uint64_t vertexUsedBytes, uint64_t indexUsedBytes,
                                  uint64_t vertexCapacityBytes, uint64_t indexCapacityBytes,
                                  uint64_t candidateVertexBytes, uint64_t candidateIndexBytes,
                                  FarTerrainGpuArenaClass admissionClass) noexcept {
    const auto alignBytes = [](uint64_t bytes) {
        constexpr uint64_t ALIGNMENT = 256;
        if (bytes > std::numeric_limits<uint64_t>::max() - (ALIGNMENT - 1)) {
            return std::numeric_limits<uint64_t>::max();
        }
        return (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    };
    const auto reservedLimit = [](uint64_t capacity, uint64_t reserved) {
        return capacity > reserved ? capacity - reserved : uint64_t{0};
    };
    uint64_t vertexReserve = FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES +
                             FAR_TERRAIN_GPU_VERTEX_FLORA_RESERVE_BYTES +
                             FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES;
    uint64_t indexReserve = FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES +
                            FAR_TERRAIN_GPU_INDEX_FLORA_RESERVE_BYTES +
                            FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES;
    if (admissionClass == FarTerrainGpuArenaClass::FLORA) {
        vertexReserve = FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES +
                        FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES;
        indexReserve = FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES +
                       FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES;
    } else if (admissionClass == FarTerrainGpuArenaClass::NEAR_REFINEMENT) {
        vertexReserve = FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES;
        indexReserve = FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES;
    } else if (admissionClass == FarTerrainGpuArenaClass::COVERAGE) {
        vertexReserve = FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES;
        indexReserve = FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES;
    } else if (admissionClass == FarTerrainGpuArenaClass::CRITICAL_COVERAGE ||
               admissionClass == FarTerrainGpuArenaClass::CRITICAL_REFINEMENT) {
        vertexReserve = 0;
        indexReserve = 0;
    }
    const uint64_t vertexLimit = reservedLimit(vertexCapacityBytes, vertexReserve);
    const uint64_t indexLimit = reservedLimit(indexCapacityBytes, indexReserve);
    const uint64_t alignedVertices = alignBytes(candidateVertexBytes);
    const uint64_t alignedIndices = alignBytes(candidateIndexBytes);
    return vertexUsedBytes <= vertexLimit && indexUsedBytes <= indexLimit &&
           alignedVertices <= vertexLimit - vertexUsedBytes &&
           alignedIndices <= indexLimit - indexUsedBytes;
}

bool farTerrainStepCompatibleWithNeighbors(
    FarTerrainStep step,
    const std::array<std::optional<FarTerrainStep>, 4>& displayedNeighborSteps) noexcept {
    const int size = farTerrainStepSize(step);
    return std::all_of(displayedNeighborSteps.begin(), displayedNeighborSteps.end(),
                       [size](std::optional<FarTerrainStep> neighbor) {
                           if (!neighbor) return true;
                           const int neighborSize = farTerrainStepSize(*neighbor);
                           return std::max(size, neighborSize) <= std::min(size, neighborSize) * 2;
                       });
}

FarTerrainTransitionTopology farTerrainTransitionCellTopology(int coarseStep, int fineStep,
                                                              uint32_t boundaryEdgeMask) {
    constexpr uint32_t HORIZONTAL_EDGE_MASK = (1U << static_cast<uint8_t>(FaceNormal::PLUS_X)) |
                                              (1U << static_cast<uint8_t>(FaceNormal::MINUS_X)) |
                                              (1U << static_cast<uint8_t>(FaceNormal::PLUS_Z)) |
                                              (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z));
    const bool validSteps = coarseStep >= 2 && coarseStep <= 32 && fineStep > 0 &&
                            fineStep <= coarseStep && coarseStep % fineStep == 0 &&
                            std::has_single_bit(static_cast<unsigned int>(coarseStep)) &&
                            std::has_single_bit(static_cast<unsigned int>(fineStep));
    const uint32_t selectedEdges = boundaryEdgeMask & HORIZONTAL_EDGE_MASK;
    const bool adjacentPair =
        selectedEdges == ((1U << static_cast<uint8_t>(FaceNormal::MINUS_X)) |
                          (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z))) ||
        selectedEdges == ((1U << static_cast<uint8_t>(FaceNormal::MINUS_X)) |
                          (1U << static_cast<uint8_t>(FaceNormal::PLUS_Z))) ||
        selectedEdges == ((1U << static_cast<uint8_t>(FaceNormal::PLUS_X)) |
                          (1U << static_cast<uint8_t>(FaceNormal::MINUS_Z))) ||
        selectedEdges == ((1U << static_cast<uint8_t>(FaceNormal::PLUS_X)) |
                          (1U << static_cast<uint8_t>(FaceNormal::PLUS_Z)));
    if (!validSteps || selectedEdges == 0 || selectedEdges != boundaryEdgeMask ||
        (std::popcount(selectedEdges) == 2 && !adjacentPair) || std::popcount(selectedEdges) > 2) {
        throw std::invalid_argument("invalid far terrain transition topology");
    }

    FarTerrainTransitionTopology result;
    const auto edgeBit = [](FaceNormal face) { return 1U << static_cast<uint8_t>(face); };
    const auto boundaryMaskAt = [&](int x, int z) {
        uint8_t mask = 0;
        if (x == 0 && (selectedEdges & edgeBit(FaceNormal::MINUS_X)) != 0)
            mask |= static_cast<uint8_t>(edgeBit(FaceNormal::MINUS_X));
        if (x == coarseStep && (selectedEdges & edgeBit(FaceNormal::PLUS_X)) != 0)
            mask |= static_cast<uint8_t>(edgeBit(FaceNormal::PLUS_X));
        if (z == 0 && (selectedEdges & edgeBit(FaceNormal::MINUS_Z)) != 0)
            mask |= static_cast<uint8_t>(edgeBit(FaceNormal::MINUS_Z));
        if (z == coarseStep && (selectedEdges & edgeBit(FaceNormal::PLUS_Z)) != 0)
            mask |= static_cast<uint8_t>(edgeBit(FaceNormal::PLUS_Z));
        return mask;
    };
    const auto appendVertex = [&](int x, int z) {
        if (result.vertexCount >= result.vertices.size()) {
            throw std::logic_error("far terrain transition vertex capacity exceeded");
        }
        result.vertices[result.vertexCount++] = {
            .x = static_cast<int16_t>(x),
            .z = static_cast<int16_t>(z),
            .boundaryEdgeMask = boundaryMaskAt(x, z),
        };
    };

    appendVertex(0, 0);
    if ((selectedEdges & edgeBit(FaceNormal::MINUS_X)) != 0) {
        for (int z = fineStep; z < coarseStep; z += fineStep)
            appendVertex(0, z);
    }
    appendVertex(0, coarseStep);
    if ((selectedEdges & edgeBit(FaceNormal::PLUS_Z)) != 0) {
        for (int x = fineStep; x < coarseStep; x += fineStep)
            appendVertex(x, coarseStep);
    }
    appendVertex(coarseStep, coarseStep);
    if ((selectedEdges & edgeBit(FaceNormal::PLUS_X)) != 0) {
        for (int z = coarseStep - fineStep; z > 0; z -= fineStep)
            appendVertex(coarseStep, z);
    }
    appendVertex(coarseStep, 0);
    if ((selectedEdges & edgeBit(FaceNormal::MINUS_Z)) != 0) {
        for (int x = coarseStep - fineStep; x > 0; x -= fineStep)
            appendVertex(x, 0);
    }

    const uint8_t perimeterCount = static_cast<uint8_t>(result.vertexCount);
    appendVertex(coarseStep / 2, coarseStep / 2);
    const uint8_t center = static_cast<uint8_t>(result.vertexCount - 1);
    for (uint8_t index = 0; index < perimeterCount; ++index) {
        if (result.indexCount + 3 > result.indices.size()) {
            throw std::logic_error("far terrain transition index capacity exceeded");
        }
        result.indices[result.indexCount++] = center;
        result.indices[result.indexCount++] = index;
        result.indices[result.indexCount++] = static_cast<uint8_t>((index + 1) % perimeterCount);
    }
    return result;
}

FarTerrainStep farTerrainNextDisplayedStep(FarTerrainStep displayed, FarTerrainStep desired) {
    const int displayedSize = farTerrainStepSize(displayed);
    const int desiredSize = farTerrainStepSize(desired);
    if (displayedSize == desiredSize) return displayed;
    const int nextSize = desiredSize < displayedSize ? displayedSize / 2 : displayedSize * 2;
    return farTerrainStepForSize(nextSize).value_or(desired);
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

std::optional<FarTerrainStep> farTerrainInitialDisplayedStep(FarTerrainStepMask readySteps) {
    if ((readySteps & farTerrainStepMask(FAR_TERRAIN_BASE_STEP)) == 0) return std::nullopt;
    // Recreating display state must never discard a finer FINAL mesh that is
    // still resident. Start from the finest completed refinement and let the
    // ordinary adjacent-tier transition path perform any intentional outward
    // coarsening. This prevents a transient map rebuild from revealing step
    // 16 or step 32 over terrain that was already displayed at a finer tier.
    for (auto step = FAR_TERRAIN_REFINEMENT_STEPS.rbegin();
         step != FAR_TERRAIN_REFINEMENT_STEPS.rend(); ++step) {
        if ((readySteps & farTerrainStepMask(*step)) != 0) return *step;
    }
    return FAR_TERRAIN_BASE_STEP;
}

bool farTerrainDisplayedStepAllowed(FarTerrainStep step, FarTerrainStep coarsestAllowed,
                                    FarTerrainStepMask readySteps) {
    if (!validStep(step) || !validStep(coarsestAllowed)) return false;
    if (farTerrainStepSize(step) <= farTerrainStepSize(coarsestAllowed)) return true;
    for (const FarTerrainStep candidate :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN}) {
        if (farTerrainStepSize(candidate) <= farTerrainStepSize(coarsestAllowed) &&
            (readySteps & farTerrainStepMask(candidate)) != 0) {
            return false;
        }
    }
    return true;
}

bool farTerrainRetainsProgressiveStep(FarTerrainStep candidate, FarTerrainStep displayed,
                                      FarTerrainStep desired) noexcept {
    if (candidate == FAR_TERRAIN_BASE_STEP || displayed == desired) return false;
    const int candidateSize = farTerrainStepSize(candidate);
    const int minimum = std::min(farTerrainStepSize(displayed), farTerrainStepSize(desired));
    const int maximum = std::max(farTerrainStepSize(displayed), farTerrainStepSize(desired));
    return candidateSize >= minimum && candidateSize <= maximum;
}

std::optional<FarTerrainStep> farTerrainReadyTransitionTarget(FarTerrainStep displayed,
                                                              FarTerrainStep desired,
                                                              FarTerrainStepMask readySteps,
                                                              bool transitionActive) {
    if (transitionActive) return std::nullopt;
    const FarTerrainStep next = farTerrainNextDisplayedStep(displayed, desired);
    if (next == displayed || (readySteps & farTerrainStepMask(next)) == 0) return std::nullopt;
    return next;
}

FarTerrainStep farTerrainCoarsestDrawableFallback(FarTerrainStep desired, bool requiresFineFallback,
                                                  bool requiresBlockScaleFallback) noexcept {
    if (farTerrainStepSize(desired) <= farTerrainStepSize(FarTerrainStep::TWO) ||
        requiresBlockScaleFallback) {
        return FarTerrainStep::TWO;
    }
    if (requiresFineFallback) return FarTerrainStep::EIGHT;
    return FAR_TERRAIN_BASE_STEP;
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
    return farTerrainStepForMetrics(chunkDistance);
}

std::optional<FarTerrainStep> farTerrainStepForMetrics(double chunkDistance,
                                                       std::optional<FarTerrainStep> previousStep) {
    if (!std::isfinite(chunkDistance) || chunkDistance < FAR_TERRAIN_NEAR_CHUNK_RADIUS ||
        chunkDistance >= FAR_TERRAIN_MAX_CHUNK_RADIUS) {
        return std::nullopt;
    }
    // Exact cubic terrain owns the first 32 chunks. The far hierarchy begins
    // at step 2 and doubles its footprint with each doubling of distance.
    const double effectiveDistance = chunkDistance;
    const auto nominalStep = [&] {
        if (effectiveDistance < FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS) return FarTerrainStep::TWO;
        if (effectiveDistance < FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS) return FarTerrainStep::FOUR;
        if (effectiveDistance < FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS) return FarTerrainStep::EIGHT;
        return FarTerrainStep::SIXTEEN;
    }();
    if (!previousStep || *previousStep == FarTerrainStep::SIXTEEN ||
        *previousStep == FarTerrainStep::THIRTY_TWO) {
        return nominalStep;
    }
    if (*previousStep == FarTerrainStep::ONE) return nominalStep;

    // Moving inward adopts the finer absolute band immediately. Hysteresis
    // may retain extra detail while moving outward, but it may never preserve
    // a coarser tier inside a finer band.
    if (farTerrainStepSize(*previousStep) > farTerrainStepSize(nominalStep)) return nominalStep;

    // A displayed finer tier survives a small distance beyond its nominal
    // ring. Camera motion therefore cannot alternate two resident meshes at
    // a boundary, while inward movement adopts the absolute finer tier
    // immediately.
    double upper = FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS;
    double upperMargin = 8.0;
    switch (*previousStep) {
        case FarTerrainStep::ONE:
            upper = FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS;
            upperMargin = 8.0;
            break;
        case FarTerrainStep::TWO:
            upper = FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS;
            upperMargin = 12.0;
            break;
        case FarTerrainStep::FOUR:
            upper = FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS;
            upperMargin = 16.0;
            break;
        case FarTerrainStep::EIGHT:
            upper = FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS;
            upperMargin = 24.0;
            break;
        case FarTerrainStep::SIXTEEN:
            upper = FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS;
            upperMargin = 0.0;
            break;
        case FarTerrainStep::THIRTY_TWO:
            break;
    }
    if (effectiveDistance < upper + upperMargin) {
        return *previousStep;
    }
    return nominalStep;
}

double farTerrainProjectedBlockPixels(const FarTerrainScreenErrorMetrics& metrics) noexcept {
    if (!std::isfinite(metrics.distanceBlocks) || metrics.distanceBlocks <= 0.0) {
        return 0.0;
    }
    if (std::isfinite(metrics.projectionScalePixels) && metrics.projectionScalePixels > 0.0) {
        return metrics.projectionScalePixels / metrics.distanceBlocks;
    }
    if (!std::isfinite(metrics.viewportHeightPixels) || metrics.viewportHeightPixels <= 0.0 ||
        !std::isfinite(metrics.verticalFovRadians) || metrics.verticalFovRadians <= 0.0 ||
        metrics.verticalFovRadians >= std::numbers::pi) {
        return 0.0;
    }
    const double tangent = std::tan(metrics.verticalFovRadians * 0.5);
    if (!std::isfinite(tangent) || tangent <= 0.0) return 0.0;
    return metrics.viewportHeightPixels / (2.0 * tangent * metrics.distanceBlocks);
}

double
farTerrainProjectedGeometricErrorPixels(FarTerrainStep step,
                                        const FarTerrainScreenErrorMetrics& metrics) noexcept {
    const double projectedBlockPixels = farTerrainProjectedBlockPixels(metrics);
    if (projectedBlockPixels <= 0.0 || !validStep(step)) return 0.0;
    const double relief =
        std::isfinite(metrics.tileReliefBlocks) ? std::max(0.0, metrics.tileReliefBlocks) : 0.0;
    const double reliefWeight =
        std::clamp(relief / FAR_TERRAIN_SCREEN_ERROR_RELIEF_BLOCKS, 0.0, 1.0);
    // Even a flat sampled surface carries half its grid spacing as material,
    // triangulation, and silhouette uncertainty. Relief raises the bound to
    // five eighths of the spacing without making one mountain tile force
    // block-scale geometry across the distant horizon.
    const double geometricErrorBlocks =
        static_cast<double>(farTerrainStepSize(step)) * (0.5 + reliefWeight * 0.125);
    return projectedBlockPixels * geometricErrorBlocks;
}

double farTerrainProjectedDisplayErrorPixels(FarTerrainStep step,
                                             FarTerrainAuthorityQuality quality,
                                             const FarTerrainScreenErrorMetrics& metrics) noexcept {
    const double geometric = farTerrainProjectedGeometricErrorPixels(step, metrics);
    if (quality == FarTerrainAuthorityQuality::FINAL) return geometric;
    const double projectedBlockPixels = farTerrainProjectedBlockPixels(metrics);
    if (projectedBlockPixels <= 0.0) return geometric;
    return geometric + projectedBlockPixels * FAR_TERRAIN_PREVIEW_RESIDUAL_MAX_BLOCKS;
}

std::optional<FarTerrainStep>
farTerrainStepForScreenMetrics(double chunkDistance, const FarTerrainScreenErrorMetrics& metrics,
                               std::optional<FarTerrainStep> previousStep) {
    // Preserve the absolute distance hysteresis as the coarsest legal tier.
    // Screen error may only move inward from this result.
    const std::optional<FarTerrainStep> absolute =
        farTerrainStepForMetrics(chunkDistance, previousStep);
    if (!absolute) return std::nullopt;
    if (farTerrainProjectedBlockPixels(metrics) <= 0.0) return absolute;

    FarTerrainStep desired = *absolute;
    while (desired != FarTerrainStep::ONE &&
           farTerrainProjectedGeometricErrorPixels(desired, metrics) >
               FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS) {
        desired = farTerrainNextDisplayedStep(desired, FarTerrainStep::ONE);
    }
    if (!previousStep || *previousStep == FarTerrainStep::SIXTEEN ||
        *previousStep == FarTerrainStep::THIRTY_TWO) {
        return desired;
    }

    const int previousSize = farTerrainStepSize(*previousStep);
    const int desiredSize = farTerrainStepSize(desired);
    if (previousSize >= desiredSize) return desired;

    // When moving outward or widening FOV, retain the finer tier until the
    // next coarser surface is comfortably subpixel. Moving inward adopts a
    // finer tier immediately through the branch above.
    const FarTerrainStep nextCoarser = farTerrainNextDisplayedStep(*previousStep, desired);
    if (farTerrainProjectedGeometricErrorPixels(nextCoarser, metrics) >
        FAR_TERRAIN_SCREEN_ERROR_COARSEN_PIXELS) {
        return previousStep;
    }
    // Only the adjacent tier was validated against the coarsening threshold.
    // Returning a farther desired tier here lets the display state walk several
    // levels outward without testing those levels, which defeats hysteresis
    // after an FOV change or a large camera move.
    return nextCoarser;
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
            output.push_back({key, bounds, nearestSquared, lodDistanceChunks, std::nullopt});
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

std::vector<worldgen::learned::TerrainPageCoordinate>
farTerrainCoarseAuthorityPages(std::span<const FarTerrainViewTile> selected, double cameraX,
                               double cameraZ) {
    using worldgen::learned::TerrainPageCoordinate;
    if (!std::isfinite(cameraX) || !std::isfinite(cameraZ)) return {};

    const auto saturatingAdd = [](int64_t value, int64_t delta) {
        if (delta > 0 && value > std::numeric_limits<int64_t>::max() - delta)
            return std::numeric_limits<int64_t>::max();
        if (delta < 0 && value < std::numeric_limits<int64_t>::min() - delta)
            return std::numeric_limits<int64_t>::min();
        return value + delta;
    };
    // Learned v4 base meshes no longer reconstruct terrain or climate through
    // legacy far-climate controls. Their widest query is the one-base-step
    // geometry and topology apron: sampleGrid probes the outer vertex and
    // cellBoundsGrid probes the adjacent half-open parent cell. Prefetching
    // the former 384-block control halo made every cold horizon request
    // learned authority that the mesh could never consume.
    constexpr int64_t BASE_GEOMETRY_SUPPORT_APRON =
        static_cast<int64_t>(farTerrainStepSize(FAR_TERRAIN_BASE_STEP));
    static_assert(BASE_GEOMETRY_SUPPORT_APRON == worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
    std::set<TerrainPageCoordinate> pages;
    for (const FarTerrainViewTile& tile : selected) {
        // These are inclusive probe coordinates. The positive tile seam is
        // itself sampled by both base geometry and canonical water coverage.
        const int64_t minimumX = saturatingAdd(tile.bounds.minX, -BASE_GEOMETRY_SUPPORT_APRON);
        const int64_t maximumX = saturatingAdd(tile.bounds.maxX, BASE_GEOMETRY_SUPPORT_APRON);
        const int64_t minimumZ = saturatingAdd(tile.bounds.minZ, -BASE_GEOMETRY_SUPPORT_APRON);
        const int64_t maximumZ = saturatingAdd(tile.bounds.maxZ, BASE_GEOMETRY_SUPPORT_APRON);
        const int64_t firstOwnerX =
            worldgen::learned::floorDivide(minimumX, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE);
        const int64_t lastOwnerX =
            worldgen::learned::floorDivide(maximumX, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE);
        const int64_t firstOwnerZ =
            worldgen::learned::floorDivide(minimumZ, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE);
        const int64_t lastOwnerZ =
            worldgen::learned::floorDivide(maximumZ, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE);
        for (int64_t ownerZ = firstOwnerZ;; ++ownerZ) {
            for (int64_t ownerX = firstOwnerX;; ++ownerX) {
                const std::vector<TerrainPageCoordinate> required =
                    worldgen::nativeHydrologyRequiredAuthorityPages(ownerX, ownerZ);
                pages.insert(required.begin(), required.end());
                if (ownerX == lastOwnerX) break;
            }
            if (ownerZ == lastOwnerZ) break;
        }
    }

    std::vector<TerrainPageCoordinate> ordered(pages.begin(), pages.end());
    const auto distanceSquared = [cameraX, cameraZ](TerrainPageCoordinate coordinate) {
        const long double centerX = static_cast<long double>(coordinate.column) *
                                        worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE +
                                    worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE / 2.0L;
        const long double centerZ = static_cast<long double>(coordinate.row) *
                                        worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE +
                                    worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE / 2.0L;
        const long double dx = centerX - static_cast<long double>(cameraX);
        const long double dz = centerZ - static_cast<long double>(cameraZ);
        return dx * dx + dz * dz;
    };
    std::sort(ordered.begin(), ordered.end(),
              [&](TerrainPageCoordinate first, TerrainPageCoordinate second) {
                  const long double firstDistance = distanceSquared(first);
                  const long double secondDistance = distanceSquared(second);
                  if (firstDistance != secondDistance) return firstDistance < secondDistance;
                  if (first.row != second.row) return first.row < second.row;
                  return first.column < second.column;
              });
    return ordered;
}

FarTerrainFinalBaseAuthorityDependencies
farTerrainFinalBaseAuthorityDependencies(FarTerrainKey key) {
    using worldgen::learned::AUTHORITY_PAGE_NATIVE_EDGE;
    using worldgen::learned::MODEL_BLOCK_SCALE;
    using worldgen::learned::TerrainPageCoordinate;

    if (!farTerrainIsBaseStep(key.step))
        throw std::invalid_argument("FINAL base authority requires a step-32 parent");
    constexpr int64_t STEP = farTerrainStepSize(FAR_TERRAIN_BASE_STEP);
    const auto checkedAdd = [](int64_t value, int64_t delta) {
        const __int128 result = static_cast<__int128>(value) + delta;
        if (result < std::numeric_limits<int64_t>::min() ||
            result > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("FINAL base authority support exceeds int64 range");
        }
        return static_cast<int64_t>(result);
    };
    const int64_t originX = tileOrigin(key.tileX);
    const int64_t originZ = tileOrigin(key.tileZ);
    const int64_t apron = STEP * GENERATOR_FAR_GRID_APRON_CELLS;
    const int64_t minimumX = checkedAdd(originX, -apron);
    const int64_t minimumZ = checkedAdd(originZ, -apron);
    const int64_t maximumX = checkedAdd(checkedAdd(originX, FAR_TERRAIN_TILE_EDGE), apron);
    const int64_t maximumZ = checkedAdd(checkedAdd(originZ, FAR_TERRAIN_TILE_EDGE), apron);

    FarTerrainFinalBaseAuthorityDependencies result{
        .minimumWorldX = minimumX,
        .minimumWorldZ = minimumZ,
        .maximumWorldXExclusive = checkedAdd(maximumX, 1),
        .maximumWorldZExclusive = checkedAdd(maximumZ, 1),
    };

    // Match WorldGenerationContext::queryWorldPoints exactly. The base sample
    // lattice and complete native-water raster are rectangular and monotone,
    // so their extreme bilinear support determines the complete page set.
    const auto bilinearNativeAxis = [](int64_t worldCoordinate) {
        const int64_t containing =
            worldgen::learned::floorDivide(worldCoordinate, MODEL_BLOCK_SCALE);
        const int64_t remainder =
            worldCoordinate - containing * static_cast<int64_t>(MODEL_BLOCK_SCALE);
        const int64_t lower = remainder < MODEL_BLOCK_SCALE / 2 ? containing - 1 : containing;
        return std::pair{lower, lower + 1};
    };
    const auto [minimumNativeColumn, ignoredMinimumColumnUpper] = bilinearNativeAxis(minimumX);
    const auto [ignoredMaximumColumnLower, maximumNativeColumn] = bilinearNativeAxis(maximumX);
    const auto [minimumNativeRow, ignoredMinimumRowUpper] = bilinearNativeAxis(minimumZ);
    const auto [ignoredMaximumRowLower, maximumNativeRow] = bilinearNativeAxis(maximumZ);
    static_cast<void>(ignoredMinimumColumnUpper);
    static_cast<void>(ignoredMaximumColumnLower);
    static_cast<void>(ignoredMinimumRowUpper);
    static_cast<void>(ignoredMaximumRowLower);
    const int64_t firstPageColumn =
        worldgen::learned::floorDivide(minimumNativeColumn, AUTHORITY_PAGE_NATIVE_EDGE);
    const int64_t lastPageColumn =
        worldgen::learned::floorDivide(maximumNativeColumn, AUTHORITY_PAGE_NATIVE_EDGE);
    const int64_t firstPageRow =
        worldgen::learned::floorDivide(minimumNativeRow, AUTHORITY_PAGE_NATIVE_EDGE);
    const int64_t lastPageRow =
        worldgen::learned::floorDivide(maximumNativeRow, AUTHORITY_PAGE_NATIVE_EDGE);
    for (int64_t row = firstPageRow;; ++row) {
        for (int64_t column = firstPageColumn;; ++column) {
            result.geometryPages.push_back(TerrainPageCoordinate{.row = row, .column = column});
            if (column == lastPageColumn) break;
        }
        if (row == lastPageRow) break;
    }

    const int64_t firstOwnerX =
        world_coord::floorDiv(minimumX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t lastOwnerX =
        world_coord::floorDiv(maximumX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t firstOwnerZ =
        world_coord::floorDiv(minimumZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t lastOwnerZ =
        world_coord::floorDiv(maximumZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
    for (int64_t ownerZ = firstOwnerZ;; ++ownerZ) {
        for (int64_t ownerX = firstOwnerX;; ++ownerX) {
            result.nativeHydrology.push_back({
                .ownerPageX = ownerX,
                .ownerPageZ = ownerZ,
                .finalTerrainRegion = worldgen::nativeHydrologyFinalTerrainRegion(ownerX, ownerZ),
            });
            if (ownerX == lastOwnerX) break;
        }
        if (ownerZ == lastOwnerZ) break;
    }
    if (maximumNativeRow < std::numeric_limits<int64_t>::max() &&
        maximumNativeColumn < std::numeric_limits<int64_t>::max()) {
        const worldgen::learned::NativeRect geometryRegion{
            .rowBegin = minimumNativeRow,
            .columnBegin = minimumNativeColumn,
            .rowEnd = maximumNativeRow + 1,
            .columnEnd = maximumNativeColumn + 1,
        };
        const auto contains = [](worldgen::learned::NativeRect outer,
                                 worldgen::learned::NativeRect inner) {
            return outer.rowBegin <= inner.rowBegin && outer.columnBegin <= inner.columnBegin &&
                   outer.rowEnd >= inner.rowEnd && outer.columnEnd >= inner.columnEnd;
        };
        if (std::ranges::any_of(result.nativeHydrology, [&](const auto& dependency) {
                return contains(dependency.finalTerrainRegion, geometryRegion);
            })) {
            result.transientGeometryRegion = geometryRegion;
        }
    }
    return result;
}

std::vector<worldgen::learned::NativeRect>
farTerrainProtectedFinalTerrainRegions(std::span<const FarTerrainKey> targets) {
    using worldgen::learned::NativeRect;
    std::set<std::pair<int64_t, int64_t>> owners;
    for (const FarTerrainKey target : targets) {
        const FarTerrainFinalBaseAuthorityDependencies dependencies =
            farTerrainFinalBaseAuthorityDependencies(
                {target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
        for (const FarTerrainNativeHydrologyDependency& dependency : dependencies.nativeHydrology) {
            owners.emplace(dependency.ownerPageZ, dependency.ownerPageX);
        }
    }

    std::vector<NativeRect> regions;
    regions.reserve(owners.size());
    while (!owners.empty()) {
        const auto [originZ, originX] = *owners.begin();
        std::vector<std::pair<int64_t, int64_t>> group;
        group.reserve(4);
        group.emplace_back(originZ, originX);
        const bool mayExtendRight = originX != std::numeric_limits<int64_t>::max();
        const bool mayExtendDown = originZ != std::numeric_limits<int64_t>::max();
        const std::pair right{originZ, mayExtendRight ? originX + 1 : originX};
        const std::pair down{mayExtendDown ? originZ + 1 : originZ, originX};
        const std::pair diagonal{mayExtendDown ? originZ + 1 : originZ,
                                 mayExtendRight ? originX + 1 : originX};
        const bool hasRight = mayExtendRight && owners.contains(right);
        const bool hasDown = mayExtendDown && owners.contains(down);
        if (hasRight) group.push_back(right);
        if (hasDown) group.push_back(down);
        if ((hasRight || hasDown) && mayExtendRight && mayExtendDown && owners.contains(diagonal)) {
            group.push_back(diagonal);
        }

        NativeRect combined = worldgen::nativeHydrologyFinalTerrainRegion(originX, originZ);
        for (size_t index = 1; index < group.size(); ++index) {
            const auto [ownerZ, ownerX] = group[index];
            const NativeRect member = worldgen::nativeHydrologyFinalTerrainRegion(ownerX, ownerZ);
            combined.rowBegin = std::min(combined.rowBegin, member.rowBegin);
            combined.columnBegin = std::min(combined.columnBegin, member.columnBegin);
            combined.rowEnd = std::max(combined.rowEnd, member.rowEnd);
            combined.columnEnd = std::max(combined.columnEnd, member.columnEnd);
        }
        const uint64_t samples = combined.height() * combined.width();
        if (!combined.valid() || samples > worldgen::learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES) {
            throw std::logic_error(
                "Protected FINAL terrain grouping exceeds the bounded native query");
        }
        regions.push_back(combined);
        for (const auto owner : group)
            owners.erase(owner);
    }
    return regions;
}

std::vector<worldgen::learned::TerrainPageCoordinate> farTerrainSpeculativeAuthorityPages(
    std::span<const worldgen::learned::TerrainPageCoordinate> visiblePages, double previousCameraX,
    double previousCameraZ, double cameraX, double cameraZ) {
    using worldgen::learned::TerrainPageCoordinate;
    if (visiblePages.empty() || !std::isfinite(previousCameraX) ||
        !std::isfinite(previousCameraZ) || !std::isfinite(cameraX) || !std::isfinite(cameraZ)) {
        return {};
    }

    const double movementX = cameraX - previousCameraX;
    const double movementZ = cameraZ - previousCameraZ;
    const double movementLength = std::hypot(movementX, movementZ);
    if (!std::isfinite(movementLength) || movementLength <= 0.0) return {};
    const long double directionX = static_cast<long double>(movementX / movementLength);
    const long double directionZ = static_cast<long double>(movementZ / movementLength);
    constexpr long double PAGE_EDGE = worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE;
    const auto center = [](TerrainPageCoordinate coordinate) {
        return std::pair{static_cast<long double>(coordinate.column) * PAGE_EDGE + PAGE_EDGE / 2.0L,
                         static_cast<long double>(coordinate.row) * PAGE_EDGE + PAGE_EDGE / 2.0L};
    };

    const std::set<TerrainPageCoordinate> visible(visiblePages.begin(), visiblePages.end());
    long double maximumForward = std::numeric_limits<long double>::lowest();
    for (const TerrainPageCoordinate coordinate : visible) {
        const auto [centerX, centerZ] = center(coordinate);
        const long double relativeX = centerX - static_cast<long double>(cameraX);
        const long double relativeZ = centerZ - static_cast<long double>(cameraZ);
        maximumForward = std::max(maximumForward, relativeX * directionX + relativeZ * directionZ);
    }

    std::set<TerrainPageCoordinate> candidates;
    for (const TerrainPageCoordinate coordinate : visible) {
        for (int64_t rowOffset = -1; rowOffset <= 1; ++rowOffset) {
            if ((rowOffset < 0 && coordinate.row == std::numeric_limits<int64_t>::min()) ||
                (rowOffset > 0 && coordinate.row == std::numeric_limits<int64_t>::max())) {
                continue;
            }
            for (int64_t columnOffset = -1; columnOffset <= 1; ++columnOffset) {
                if (rowOffset == 0 && columnOffset == 0) continue;
                if ((columnOffset < 0 &&
                     coordinate.column == std::numeric_limits<int64_t>::min()) ||
                    (columnOffset > 0 &&
                     coordinate.column == std::numeric_limits<int64_t>::max())) {
                    continue;
                }
                const TerrainPageCoordinate candidate{.row = coordinate.row + rowOffset,
                                                      .column = coordinate.column + columnOffset};
                if (visible.contains(candidate)) continue;
                const auto [centerX, centerZ] = center(candidate);
                const long double relativeX = centerX - static_cast<long double>(cameraX);
                const long double relativeZ = centerZ - static_cast<long double>(cameraZ);
                if (relativeX * directionX + relativeZ * directionZ <= maximumForward) continue;
                candidates.insert(candidate);
            }
        }
    }

    const long double targetDistance = maximumForward + PAGE_EDGE;
    const long double targetX = static_cast<long double>(cameraX) + directionX * targetDistance;
    const long double targetZ = static_cast<long double>(cameraZ) + directionZ * targetDistance;
    std::vector<TerrainPageCoordinate> ordered(candidates.begin(), candidates.end());
    const auto targetDistanceSquared = [&](TerrainPageCoordinate coordinate) {
        const auto [centerX, centerZ] = center(coordinate);
        const long double deltaX = centerX - targetX;
        const long double deltaZ = centerZ - targetZ;
        return deltaX * deltaX + deltaZ * deltaZ;
    };
    std::sort(ordered.begin(), ordered.end(),
              [&](TerrainPageCoordinate first, TerrainPageCoordinate second) {
                  const long double firstDistance = targetDistanceSquared(first);
                  const long double secondDistance = targetDistanceSquared(second);
                  if (firstDistance != secondDistance) return firstDistance < secondDistance;
                  if (first.row != second.row) return first.row < second.row;
                  return first.column < second.column;
              });
    if (ordered.size() > FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES)
        ordered.resize(FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES);
    return ordered;
}

bool FarTerrainProtectedNearHandoff::request(ColumnPos anchor) noexcept {
    if (activeCenter_ && *activeCenter_ == anchor) {
        const bool changed = requestedCenter_.has_value();
        requestedCenter_.reset();
        return changed;
    }
    if (requestedCenter_ && *requestedCenter_ == anchor) return false;
    requestedCenter_ = anchor;
    return true;
}

bool FarTerrainProtectedNearHandoff::commitRequested(bool ready) noexcept {
    if (!ready || !requestedCenter_) return false;
    activeCenter_ = requestedCenter_;
    requestedCenter_.reset();
    return true;
}

void FarTerrainProtectedNearHandoff::clear() noexcept {
    activeCenter_.reset();
    requestedCenter_.reset();
}

FarTerrainProtectedNearRole farTerrainProtectedNearRole(ColumnPos anchor,
                                                        ColumnPos coordinate) noexcept {
    const auto axisDistance = [](int64_t value, int64_t minimum) {
        const __int128 coordinateValue = value;
        const __int128 lower = minimum;
        const __int128 upper = lower + FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES - 1;
        if (coordinateValue < lower) return lower - coordinateValue;
        if (coordinateValue > upper) return coordinateValue - upper;
        return __int128{0};
    };
    const __int128 distance =
        axisDistance(coordinate.x, anchor.x) + axisDistance(coordinate.z, anchor.z);
    if (distance == FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_DISTANCE_TILES) {
        return FarTerrainProtectedNearRole::STEP_ONE_CORE;
    }
    if (distance == FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_DISTANCE_TILES) {
        return FarTerrainProtectedNearRole::STEP_TWO_RING;
    }
    if (distance == FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_DISTANCE_TILES) {
        return FarTerrainProtectedNearRole::STEP_FOUR_RING;
    }
    if (distance == FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_DISTANCE_TILES) {
        return FarTerrainProtectedNearRole::STEP_EIGHT_RING;
    }
    if (distance == FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_DISTANCE_TILES) {
        return FarTerrainProtectedNearRole::STEP_SIXTEEN_RING;
    }
    return FarTerrainProtectedNearRole::NONE;
}

namespace {
std::optional<FarTerrainStep> protectedNearStepForRole(FarTerrainProtectedNearRole role) noexcept {
    switch (role) {
        case FarTerrainProtectedNearRole::STEP_ONE_CORE:
            return FarTerrainStep::ONE;
        case FarTerrainProtectedNearRole::STEP_TWO_RING:
            return FarTerrainStep::TWO;
        case FarTerrainProtectedNearRole::STEP_FOUR_RING:
            return FarTerrainStep::FOUR;
        case FarTerrainProtectedNearRole::STEP_EIGHT_RING:
            return FarTerrainStep::EIGHT;
        case FarTerrainProtectedNearRole::STEP_SIXTEEN_RING:
            return FarTerrainStep::SIXTEEN;
        case FarTerrainProtectedNearRole::NONE:
            return std::nullopt;
    }
    return std::nullopt;
}
} // namespace

bool farTerrainProtectedNearTargetKey(const std::optional<ColumnPos>& anchor,
                                      FarTerrainKey key) noexcept {
    if (!anchor || farTerrainIsBaseStep(key.step)) return false;
    const std::optional<FarTerrainStep> required = protectedNearStepForRole(
        farTerrainProtectedNearRole(*anchor, ColumnPos{key.tileX, key.tileZ}));
    return required && *required == key.step;
}

bool farTerrainCriticalProtectedRefinement(const std::optional<ColumnPos>& requestedAnchor,
                                           FarTerrainKey key,
                                           FarTerrainAuthorityQuality quality) noexcept {
    return quality == FarTerrainAuthorityQuality::FINAL &&
           farTerrainProtectedNearTargetKey(requestedAnchor, key);
}

std::optional<FarTerrainStep>
farTerrainProtectedNearRequiredStep(const FarTerrainProtectedNearHandoff& handoff,
                                    ColumnPos coordinate) noexcept {
    const auto stepForCenter =
        [&](const std::optional<ColumnPos>& center) -> std::optional<FarTerrainStep> {
        if (!center) return std::nullopt;
        return protectedNearStepForRole(farTerrainProtectedNearRole(*center, coordinate));
    };
    // The requested anchor owns overlap while moving. An old step-1 core can
    // therefore remain displayed while its replacement step-2 shell is built;
    // choosing the finer old role here would omit that required shell mesh and
    // prevent the atomic handoff from ever completing.
    if (const std::optional<FarTerrainStep> requested = stepForCenter(handoff.requestedCenter()))
        return requested;
    return stepForCenter(handoff.activeCenter());
}

void buildFarTerrainProtectedNearTargets(ColumnPos anchor,
                                         std::span<const FarTerrainViewTile> selected,
                                         std::vector<FarTerrainKey>& targets) {
    targets.clear();
    targets.reserve(FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    for (const FarTerrainViewTile& tile : selected) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const FarTerrainProtectedNearRole role = farTerrainProtectedNearRole(anchor, coordinate);
        const std::optional<FarTerrainStep> step = protectedNearStepForRole(role);
        if (step) targets.push_back({coordinate.x, coordinate.z, *step});
    }
    std::sort(targets.begin(), targets.end(), [](FarTerrainKey first, FarTerrainKey second) {
        if (first.step != second.step)
            return farTerrainStepSize(first.step) < farTerrainStepSize(second.step);
        if (first.tileX != second.tileX) return first.tileX < second.tileX;
        return first.tileZ < second.tileZ;
    });
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
}

bool farTerrainProtectedNearTargetsReady(std::span<const FarTerrainKey> targets,
                                         const FarTerrainResidencyFunction& isCompatibleResident) {
    if (targets.empty() || !isCompatibleResident) return false;
    return std::ranges::all_of(targets, [&](FarTerrainKey target) {
        return isCompatibleResident({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP}) &&
               isCompatibleResident(target);
    });
}

FarTerrainProtectedNearGeometryStatus
farTerrainProtectedNearGeometryStatus(ColumnPos anchor,
                                      std::span<const FarTerrainKey> expectedTargets,
                                      std::span<const FarTerrainProtectedNearSurface> surfaces) {
    FarTerrainProtectedNearGeometryStatus status;
    status.expectedTargets = expectedTargets.size();
    status.expectedFinalParents = expectedTargets.size();
    constexpr size_t WEST = static_cast<size_t>(FarTerrainBoundaryEdge::WEST);
    constexpr size_t EAST = static_cast<size_t>(FarTerrainBoundaryEdge::EAST);
    constexpr size_t NORTH = static_cast<size_t>(FarTerrainBoundaryEdge::NORTH);
    constexpr size_t SOUTH = static_cast<size_t>(FarTerrainBoundaryEdge::SOUTH);

    const auto findExpected = [&](ColumnPos coordinate) -> const FarTerrainKey* {
        const auto found = std::ranges::find_if(expectedTargets, [&](const FarTerrainKey& key) {
            return key.tileX == coordinate.x && key.tileZ == coordinate.z;
        });
        if (found == expectedTargets.end()) return nullptr;
        const auto duplicate =
            std::find_if(std::next(found), expectedTargets.end(), [&](const FarTerrainKey& key) {
                return key.tileX == coordinate.x && key.tileZ == coordinate.z;
            });
        return duplicate == expectedTargets.end() ? &*found : nullptr;
    };
    const auto findSurface =
        [&](const FarTerrainKey& expected) -> const FarTerrainProtectedNearSurface* {
        const auto found = std::ranges::find_if(
            surfaces, [&](const auto& surface) { return surface.key == expected; });
        if (found == surfaces.end()) return nullptr;
        const auto duplicate =
            std::find_if(std::next(found), surfaces.end(),
                         [&](const auto& value) { return value.key == found->key; });
        return duplicate == surfaces.end() ? &*found : nullptr;
    };

    for (const FarTerrainKey& expected : expectedTargets) {
        const ColumnPos coordinate{expected.tileX, expected.tileZ};
        const std::optional<FarTerrainStep> roleStep =
            protectedNearStepForRole(farTerrainProtectedNearRole(anchor, coordinate));
        if (!roleStep || *roleStep != expected.step) {
            ++status.incompatibleLodBoundaries;
            continue;
        }
        const FarTerrainProtectedNearSurface* surface = findSurface(expected);
        if (!surface || !surface->surfaceBoundary.valid) continue;
        ++status.presentTargets;
        if (surface->authorityQuality == FarTerrainAuthorityQuality::FINAL) ++status.finalTargets;
        if (surface->parentAuthorityQuality == FarTerrainAuthorityQuality::FINAL)
            ++status.finalParents;
        if (surface->exactAuthorityCompatible) ++status.exactCompatibleTargets;

        const auto compare = [&](ColumnPos neighbor, size_t edge, size_t opposite) {
            const FarTerrainKey* adjacentExpected = findExpected(neighbor);
            if (!adjacentExpected) return;
            ++status.expectedSharedBoundaries;
            const int smaller = std::min(farTerrainStepSize(expected.step),
                                         farTerrainStepSize(adjacentExpected->step));
            const int larger = std::max(farTerrainStepSize(expected.step),
                                        farTerrainStepSize(adjacentExpected->step));
            if (larger > smaller * 2) ++status.incompatibleLodBoundaries;
            const FarTerrainProtectedNearSurface* adjacent = findSurface(*adjacentExpected);
            if (!adjacent || !adjacent->surfaceBoundary.valid) return;
            if (surface->surfaceBoundary.heightHashes[edge] ==
                adjacent->surfaceBoundary.heightHashes[opposite]) {
                ++status.matchingSharedBoundaries;
            } else {
                ++status.mismatchedSharedBoundaries;
            }
        };
        compare({coordinate.x + 1, coordinate.z}, EAST, WEST);
        compare({coordinate.x, coordinate.z + 1}, SOUTH, NORTH);
    }
    return status;
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
    return tile.key.step == FarTerrainStep::ONE || tile.key.step == FarTerrainStep::TWO ||
           tile.key.step == FarTerrainStep::FOUR || tile.key.step == FarTerrainStep::EIGHT ||
           tile.key.step == FarTerrainStep::SIXTEEN;
}

bool farTerrainConnectedRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier) noexcept {
    if (frontier.complete) return true;
    constexpr float CONNECTED_REFINEMENT_START_BLOCK_RADIUS =
        static_cast<float>(FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS * CHUNK_EDGE);
    return frontier.missingBaseTiles != 0 &&
           frontier.distanceBlocks >= CONNECTED_REFINEMENT_START_BLOCK_RADIUS;
}

void buildFarTerrainProgressiveSubmissionOrder(
    std::span<const FarTerrainRefinementCacheRequest> requests, std::vector<FarTerrainKey>& output,
    size_t maximumResults) {
    output.clear();
    if (maximumResults == 0) return;

    struct Candidate {
        const FarTerrainRefinementCacheRequest* request = nullptr;
        FarTerrainKey key{};
    };
    const auto candidateFor =
        [](const FarTerrainRefinementCacheRequest& request) -> std::optional<Candidate> {
        if (request.transitionActive) return std::nullopt;
        const bool nearTarget =
            farTerrainStepSize(request.desired) <= farTerrainStepSize(FarTerrainStep::TWO);
        const bool desiredResident =
            (request.residentSteps & farTerrainStepMask(request.desired)) != 0;
        if (nearTarget && desiredResident) return std::nullopt;
        const FarTerrainStep next = farTerrainNextDisplayedStep(request.displayed, request.desired);
        if (next == request.displayed || (request.residentSteps & farTerrainStepMask(next)) != 0 ||
            (request.deferIntermediate && next != request.desired)) {
            return std::nullopt;
        }
        return Candidate{&request, {request.coordinate.x, request.coordinate.z, next}};
    };

    // Production asks for at most the scheduler's 64 available slots. Keep a
    // stable, bounded top-K directly on the stack so the 3,336-tile horizon is
    // one linear pass with a small constant and no render-thread allocation.
    if (maximumResults <= FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS) {
        std::array<Candidate, FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS> ranked{};
        size_t rankedCount = 0;
        for (const FarTerrainRefinementCacheRequest& request : requests) {
            const std::optional<Candidate> candidate = candidateFor(request);
            if (!candidate) continue;

            size_t duplicate = rankedCount;
            for (size_t index = 0; index < rankedCount; ++index) {
                if (ranked[index].key == candidate->key) {
                    duplicate = index;
                    break;
                }
            }
            if (duplicate != rankedCount) {
                if (!farTerrainRefinementRequestBefore(request, *ranked[duplicate].request))
                    continue;
                for (size_t index = duplicate + 1; index < rankedCount; ++index)
                    ranked[index - 1] = ranked[index];
                --rankedCount;
            }

            size_t insertion = 0;
            while (insertion < rankedCount &&
                   !farTerrainRefinementRequestBefore(request, *ranked[insertion].request)) {
                ++insertion;
            }
            if (insertion >= maximumResults) continue;
            const size_t newCount = std::min(rankedCount + 1, maximumResults);
            for (size_t index = newCount; index > insertion + 1; --index)
                ranked[index - 1] = ranked[index - 2];
            ranked[insertion] = *candidate;
            rankedCount = newCount;
        }

        output.reserve(maximumResults);
        for (const Candidate& candidate : std::span(ranked).first(rankedCount))
            output.push_back(candidate.key);
        return;
    }

    std::vector<const FarTerrainRefinementCacheRequest*> ordered;
    ordered.reserve(requests.size());
    for (const FarTerrainRefinementCacheRequest& request : requests)
        ordered.push_back(&request);
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto* first, const auto* second) {
        return farTerrainRefinementRequestBefore(*first, *second);
    });

    // Build only the next adjacent topology. A direct step-32 to step-2 job
    // consumes substantially more CPU and memory, but cannot become visible
    // while a cardinal neighbor still needs step 16 or step 8. Advancing the
    // displayable bridge first converts every completed job into immediate
    // visual progress and lets the same tile request its next tier after the
    // bounded transition completes.
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> emitted;
    emitted.reserve(std::min(requests.size(), maximumResults));
    output.reserve(std::min(requests.size(), maximumResults));
    for (const FarTerrainRefinementCacheRequest* requestPointer : ordered) {
        const FarTerrainRefinementCacheRequest& request = *requestPointer;
        const std::optional<Candidate> candidate = candidateFor(request);
        if (!candidate || !emitted.insert(candidate->key).second) continue;
        output.push_back(candidate->key);
        if (output.size() >= maximumResults) break;
    }
}

size_t
reserveFarTerrainIntermediateTransitionSlots(std::span<FarTerrainRefinementCacheRequest> requests,
                                             size_t activeTransitions) {
    size_t remaining = FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS -
                       std::min(activeTransitions, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS);
    std::array<FarTerrainRefinementCacheRequest*, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS>
        reserved{};
    size_t reservedCount = 0;
    for (FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive || request.deferIntermediate || request.requiresFineFallback ||
            request.protectedNearTarget)
            continue;
        size_t insertion = 0;
        while (insertion < reservedCount &&
               !farTerrainRefinementRequestBefore(request, *reserved[insertion])) {
            ++insertion;
        }
        if (insertion >= remaining) continue;
        const size_t newCount = std::min(reservedCount + 1, remaining);
        for (size_t index = newCount; index > insertion + 1; --index)
            reserved[index - 1] = reserved[index - 2];
        reserved[insertion] = &request;
        reservedCount = newCount;
    }
    for (FarTerrainRefinementCacheRequest& request : requests) {
        if (request.transitionActive || request.deferIntermediate || request.requiresFineFallback ||
            request.protectedNearTarget)
            continue;
        bool selected = false;
        for (FarTerrainRefinementCacheRequest* candidate :
             std::span(reserved).first(reservedCount)) {
            if (candidate == &request) {
                selected = true;
                break;
            }
        }
        request.deferIntermediate = !selected;
    }
    return reservedCount;
}

void FarTerrainPlannerTimingHistogram::clear() noexcept {
    bins_.fill(0);
    sampleCount_ = 0;
    maximumMilliseconds_ = 0.0F;
}

void FarTerrainPlannerTimingHistogram::record(double milliseconds) noexcept {
    if (!std::isfinite(milliseconds) || milliseconds < 0.0) return;
    const size_t bin =
        std::min(static_cast<size_t>(milliseconds / BIN_WIDTH_MILLISECONDS), BIN_COUNT - size_t{1});
    ++bins_[bin];
    ++sampleCount_;
    maximumMilliseconds_ = std::max(maximumMilliseconds_, static_cast<float>(milliseconds));
}

float FarTerrainPlannerTimingHistogram::percentile95Milliseconds() const noexcept {
    if (sampleCount_ == 0) return 0.0F;
    const uint64_t target = (sampleCount_ * 95 + 99) / 100;
    uint64_t cumulative = 0;
    for (size_t bin = 0; bin < bins_.size(); ++bin) {
        cumulative += bins_[bin];
        if (cumulative >= target) {
            return static_cast<float>((static_cast<double>(bin) + 1.0) * BIN_WIDTH_MILLISECONDS);
        }
    }
    return static_cast<float>(BIN_COUNT * BIN_WIDTH_MILLISECONDS);
}

bool farTerrainRefinementLaneOpen(const FarTerrainCoverageFrontier& frontier,
                                  bool allBaseCandidatesScanned) {
    return frontier.complete && frontier.missingBaseTiles == 0 && allBaseCandidatesScanned;
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
                                   std::vector<FarTerrainKey>& output,
                                   std::span<const ColumnPos> criticalCoordinates) {
    output.clear();
    output.reserve(selected.size() * (FAR_TERRAIN_REFINEMENT_STEPS.size() + 1));
    for (const FarTerrainViewTile& tile : selected) {
        output.push_back({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    const std::unordered_set<ColumnPos> criticalSet(criticalCoordinates.begin(),
                                                    criticalCoordinates.end());
    const auto critical = [&](const FarTerrainViewTile& tile) {
        return criticalSet.contains({tile.key.tileX, tile.key.tileZ});
    };
    for (const bool criticalPass : {true, false}) {
        for (const FarTerrainStep step : FAR_TERRAIN_REFINEMENT_STEPS) {
            for (const FarTerrainViewTile& tile : selected) {
                if (critical(tile) != criticalPass) continue;
                const FarTerrainStep target = farTerrainResidencyTarget(tile);
                if (farTerrainStepSize(step) < farTerrainStepSize(target)) continue;
                output.push_back({tile.key.tileX, tile.key.tileZ, step});
            }
        }
    }
}

void buildFarTerrainCriticalResidencyOrder(std::span<const FarTerrainKey> targets,
                                           std::vector<FarTerrainKey>& output) {
    output.clear();
    output.reserve(targets.size() * (FAR_TERRAIN_REFINEMENT_STEPS.size() + 1));
    output.insert(output.end(), targets.begin(), targets.end());
    for (const FarTerrainKey target : targets) {
        output.push_back({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    for (const FarTerrainStep bridge : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        for (const FarTerrainKey target : targets) {
            if (farTerrainStepSize(bridge) <= farTerrainStepSize(target.step)) continue;
            output.push_back({target.tileX, target.tileZ, bridge});
        }
    }
}

void buildFarTerrainTieredCriticalResidencyOrder(std::span<const FarTerrainKey> currentTargets,
                                                 std::span<const FarTerrainKey> predictedTargets,
                                                 std::vector<FarTerrainKey>& output) {
    buildFarTerrainCriticalResidencyOrder(currentTargets, output);
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> seen(output.begin(), output.end());
    std::vector<FarTerrainKey> predicted;
    buildFarTerrainCriticalResidencyOrder(predictedTargets, predicted);
    output.reserve(output.size() + predicted.size());
    for (const FarTerrainKey key : predicted) {
        if (seen.insert(key).second) output.push_back(key);
    }
}

bool farTerrainResidencyOrderMatches(const std::vector<FarTerrainViewTile>& selected,
                                     std::span<const FarTerrainKey> order,
                                     std::span<const ColumnPos> criticalCoordinates) {
    size_t refinementCount = 0;
    for (const FarTerrainViewTile& tile : selected)
        refinementCount += farTerrainRefinementOrder(farTerrainResidencyTarget(tile)).count;
    if (order.size() != selected.size() + refinementCount) return false;
    for (size_t index = 0; index < selected.size(); ++index) {
        const FarTerrainKey expected{selected[index].key.tileX, selected[index].key.tileZ,
                                     FAR_TERRAIN_BASE_STEP};
        if (order[index] != expected) return false;
    }
    size_t orderIndex = selected.size();
    const std::unordered_set<ColumnPos> criticalSet(criticalCoordinates.begin(),
                                                    criticalCoordinates.end());
    const auto critical = [&](const FarTerrainViewTile& tile) {
        return criticalSet.contains({tile.key.tileX, tile.key.tileZ});
    };
    for (const bool criticalPass : {true, false}) {
        for (const FarTerrainStep step : FAR_TERRAIN_REFINEMENT_STEPS) {
            for (const FarTerrainViewTile& tile : selected) {
                if (critical(tile) != criticalPass) continue;
                const FarTerrainStep target = farTerrainResidencyTarget(tile);
                if (farTerrainStepSize(step) < farTerrainStepSize(target)) continue;
                const FarTerrainKey expected{tile.key.tileX, tile.key.tileZ, step};
                if (order[orderIndex++] != expected) return false;
            }
        }
    }
    return true;
}

bool farTerrainResidencyMembershipMatches(
    const std::vector<FarTerrainViewTile>& selected,
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
    std::span<const FarTerrainKey> additionalKeys) {
    size_t expectedCount = selected.size();
    for (const FarTerrainViewTile& tile : selected) {
        expectedCount += farTerrainRefinementOrder(farTerrainResidencyTarget(tile)).count;
    }
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> uniqueAdditional;
    uniqueAdditional.reserve(additionalKeys.size());
    for (const FarTerrainKey key : additionalKeys) {
        if (!wanted.contains(key)) return false;
        uniqueAdditional.insert(key);
    }
    for (const FarTerrainViewTile& tile : selected) {
        const FarTerrainKey base{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        if (!wanted.contains(base)) return false;
        uniqueAdditional.erase(base);
        const FarTerrainRefinementOrder refinement =
            farTerrainRefinementOrder(farTerrainResidencyTarget(tile));
        for (FarTerrainStep step : std::span(refinement.steps).first(refinement.count)) {
            const FarTerrainKey key{tile.key.tileX, tile.key.tileZ, step};
            if (!wanted.contains(key)) return false;
            uniqueAdditional.erase(key);
        }
    }
    expectedCount += uniqueAdditional.size();
    return wanted.size() == expectedCount;
}

bool buildFarTerrainConnectedNearPatchHandoff(std::span<const FarTerrainViewTile> selected,
                                              const FarTerrainResidencyFunction& isResident,
                                              std::vector<FarTerrainKey>& targets) {
    targets.clear();
    if (!isResident) return false;
    targets.reserve(std::min<size_t>(selected.size(), 128));
    const auto selectedAt = [&](ColumnPos coordinate) -> const FarTerrainViewTile* {
        const auto found = std::ranges::find_if(selected, [&](const FarTerrainViewTile& tile) {
            return tile.key.tileX == coordinate.x && tile.key.tileZ == coordinate.z;
        });
        return found == selected.end() ? nullptr : &*found;
    };
    for (const FarTerrainViewTile& tile : selected) {
        if (tile.key.step != FarTerrainStep::ONE) continue;
        const FarTerrainKey parent{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        const FarTerrainKey target{tile.key.tileX, tile.key.tileZ, FarTerrainStep::ONE};
        if (!isResident(parent) || !isResident(target)) {
            targets.clear();
            return false;
        }
        targets.push_back(target);
    }
    if (targets.empty()) return false;

    // A complete fine patch is not publishable against a step-32 exterior.
    // Require one cardinal shell and publish it in the same frame-level unit.
    // The movement halo normally has step 1 ready; step 2 is the legal fallback
    // and still gives every visible patch edge a 2:1 or better neighbor.
    constexpr std::array<ColumnPos, 4> CARDINALS = {ColumnPos{1, 0}, ColumnPos{-1, 0},
                                                    ColumnPos{0, 1}, ColumnPos{0, -1}};
    const size_t fineTargetCount = targets.size();
    for (size_t index = 0; index < fineTargetCount; ++index) {
        const FarTerrainKey fine = targets[index];
        for (const ColumnPos offset : CARDINALS) {
            const ColumnPos coordinate{fine.tileX + offset.x, fine.tileZ + offset.z};
            const FarTerrainViewTile* neighbor = selectedAt(coordinate);
            if (!neighbor || neighbor->key.step == FarTerrainStep::ONE) continue;
            const FarTerrainKey parent{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
            if (!isResident(parent)) {
                targets.clear();
                return false;
            }
            FarTerrainKey shell{coordinate.x, coordinate.z, FarTerrainStep::ONE};
            if (!isResident(shell)) {
                shell.step = FarTerrainStep::TWO;
                if (!isResident(shell)) {
                    targets.clear();
                    return false;
                }
            }
            if (std::ranges::find(targets, shell) == targets.end()) targets.push_back(shell);
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

bool farTerrainExactSectionDrawAllowed(bool sectionRequired, bool columnFullyReady,
                                       bool coverageParentDrawable) noexcept {
    return farTerrainExactVisualOwnership(sectionRequired, columnFullyReady, coverageParentDrawable,
                                          true)
        .drawExact;
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

bool FarTerrainExactCoverageCache::sectionRequired(ChunkPos section) const {
    return sectionColumns_.contains(section);
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

uint32_t buildFarTerrainFinalParentUpgradeOrder(std::span<const FarTerrainViewTile> selected,
                                                double cameraX, double cameraZ,
                                                float nominalExactDistanceBlocks,
                                                const FarTerrainExactHandoff& handoff,
                                                const FarTerrainResidencyFunction& isFinalResident,
                                                std::vector<FarTerrainKey>& output) {
    output.clear();
    output.reserve(selected.size());
    const auto append = [&](bool exactHandoffRequired) {
        for (const FarTerrainViewTile& tile : selected) {
            const FarTerrainKey base{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
            if (isFinalResident && isFinalResident(base)) continue;
            const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
            if (farTerrainRequiresCoverageParent(cameraX, cameraZ, coordinate,
                                                 nominalExactDistanceBlocks,
                                                 handoff) != exactHandoffRequired) {
                continue;
            }
            output.push_back(base);
        }
    };
    append(true);
    const uint32_t requiredCount = static_cast<uint32_t>(
        std::min<size_t>(output.size(), static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    append(false);
    return requiredCount;
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

size_t FarCanopyAttachment::byteSize() const {
    return sizeof(*this) + vertices.capacity() * sizeof(Vertex) +
           indices.capacity() * sizeof(uint32_t);
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainMesher::build(FarTerrainKey key, const FarTerrainSource& source,
                        FarTerrainAuthorityQuality authorityQuality) {
    trace::Scope span(
        trace::Track::FarGeneration, trace::Name::FarTileBuild,
        {.spatialKey = trace::packCoord(key.tileX, key.tileZ, static_cast<uint8_t>(key.step)),
         .quality = static_cast<uint8_t>(authorityQuality)});
    return buildInternal(key, source, authorityQuality);
}

std::shared_ptr<const FarCanopyAttachment>
FarTerrainMesher::buildCanopyAttachment(FarTerrainKey key, const FarTerrainSource& source,
                                        FarTerrainAuthorityQuality authorityQuality) {
    return buildCanopyAttachment(key, source, source, authorityQuality, authorityQuality);
}

std::shared_ptr<const FarCanopyAttachment> FarTerrainMesher::buildCanopyAttachment(
    FarTerrainKey key, const FarTerrainSource& ecologySource,
    const FarTerrainSource& groundingSource, FarTerrainAuthorityQuality groundingQuality,
    FarTerrainAuthorityQuality authorityQuality, FarCanopyBuildDiagnostics* diagnostics) {
    if (!ecologySource.sample || !groundingSource.sample) {
        throw std::invalid_argument("far terrain source is incomplete");
    }
    if (!validStep(key.step)) throw std::invalid_argument("unsupported far terrain LOD step");

    FarCanopyBuildDiagnostics localDiagnostics;
    FarCanopyBuildDiagnostics& buildDiagnostics = diagnostics ? *diagnostics : localDiagnostics;
    buildDiagnostics = {};
    ScopedCanopyDiagnosticTimer totalTimer(buildDiagnostics.totalMicroseconds);

    auto attachment = std::make_shared<FarCanopyAttachment>();
    attachment->key = key;
    attachment->authorityQuality = authorityQuality;
    attachment->groundingQuality = groundingQuality;
    attachment->originX = tileOrigin(key.tileX);
    attachment->originZ = tileOrigin(key.tileZ);
    attachment->bounds = {attachment->originX,
                          attachment->originX + FAR_TERRAIN_TILE_EDGE,
                          attachment->originZ,
                          attachment->originZ + FAR_TERRAIN_TILE_EDGE,
                          0.0F,
                          0.0F};
    if (!ecologySource.canopies && !ecologySource.flora) {
        attachment->anchorIdentityHash = hashCanopyAnchors(std::span<const FarCanopy>{});
        attachment->deterministicHash = hashCanopyAttachment(*attachment);
        totalTimer.finish();
        attachment->buildDiagnostics = buildDiagnostics;
        return attachment;
    }

    std::vector<FarCanopy> candidates;
    {
        ScopedCanopyDiagnosticTimer collectionTimer(buildDiagnostics.canopyCollectionMicroseconds);
        candidates =
            ecologySource.canopies
                ? ecologySource.canopies(attachment->originX, attachment->originZ,
                                         attachment->originX + FAR_TERRAIN_TILE_EDGE,
                                         attachment->originZ + FAR_TERRAIN_TILE_EDGE, key.step)
                : std::vector<FarCanopy>{};
    }
    buildDiagnostics.canopyCandidateCount = boundedDiagnosticCount(candidates.size());
    std::vector<FarCanopy> canopies;
    canopies.reserve(candidates.size());
    for (const FarCanopy& canopy : candidates) {
        if (canopy.x >= attachment->originX &&
            canopy.x < attachment->originX + FAR_TERRAIN_TILE_EDGE &&
            canopy.z >= attachment->originZ &&
            canopy.z < attachment->originZ + FAR_TERRAIN_TILE_EDGE) {
            canopies.push_back(canopy);
        }
    }
    buildDiagnostics.acceptedCanopyCount = boundedDiagnosticCount(canopies.size());
    std::vector<FarFlora> floraCandidates;
    {
        ScopedCanopyDiagnosticTimer collectionTimer(buildDiagnostics.floraCollectionMicroseconds);
        floraCandidates =
            ecologySource.flora
                ? ecologySource.flora(attachment->originX, attachment->originZ,
                                      attachment->originX + FAR_TERRAIN_TILE_EDGE,
                                      attachment->originZ + FAR_TERRAIN_TILE_EDGE, key.step)
                : std::vector<FarFlora>{};
    }
    buildDiagnostics.floraCandidateCount = boundedDiagnosticCount(floraCandidates.size());
    std::vector<FarFlora> flora;
    flora.reserve(floraCandidates.size());
    for (const FarFlora& plant : floraCandidates) {
        if (plant.x >= attachment->originX &&
            plant.x < attachment->originX + FAR_TERRAIN_TILE_EDGE &&
            plant.z >= attachment->originZ &&
            plant.z < attachment->originZ + FAR_TERRAIN_TILE_EDGE) {
            flora.push_back(plant);
        }
    }
    buildDiagnostics.acceptedFloraCount = boundedDiagnosticCount(flora.size());
    attachment->anchorIdentityHash = hashCanopyAnchors(canopies);
    if (ecologySource.flora) {
        attachment->anchorIdentityHash =
            mix64(attachment->anchorIdentityHash ^ std::rotl(hashFloraAnchors(flora), 17));
    }

    std::vector<FarCanopy> groundingAnchors = canopies;
    groundingAnchors.reserve(canopies.size() + flora.size());
    for (const FarFlora& plant : flora) {
        groundingAnchors.push_back({.x = plant.x,
                                    .z = plant.z,
                                    .baseY = plant.baseY,
                                    .anchorId = plant.anchorId,
                                    .species = feature_generation::TreeSpecies::COUNT});
    }
    std::vector<float> groundHeights(groundingAnchors.size());
    if (!groundingAnchors.empty()) {
        ScopedCanopyDiagnosticTimer groundingTimer(buildDiagnostics.groundingMicroseconds);
        const int step = farTerrainStepSize(key.step);
        const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
        const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(key.step);
        const auto owningCell = [&](const FarCanopy& canopy) {
            return std::pair{
                static_cast<int>((canopy.x - attachment->originX) / step),
                static_cast<int>((canopy.z - attachment->originZ) / step),
            };
        };
        std::vector<uint32_t> occupiedCells;
        occupiedCells.reserve(groundingAnchors.size());
        for (const FarCanopy& canopy : groundingAnchors) {
            const auto [cellX, cellZ] = owningCell(canopy);
            occupiedCells.push_back(static_cast<uint32_t>(cellZ * cellEdge + cellX));
        }
        std::ranges::sort(occupiedCells);
        occupiedCells.erase(std::unique(occupiedCells.begin(), occupiedCells.end()),
                            occupiedCells.end());
        buildDiagnostics.occupiedGroundCellCount = boundedDiagnosticCount(occupiedCells.size());
        std::vector<float> cellTops(static_cast<size_t>(cellEdge * cellEdge),
                                    std::numeric_limits<float>::quiet_NaN());
        const size_t fullGridThreshold = static_cast<size_t>(cellEdge) * 4;
        const size_t denseCellCount = static_cast<size_t>(cellEdge * cellEdge);
        const bool preferSparseGrounding =
            groundingSource.terrainCellTopPoints && occupiedCells.size() * 2 < denseCellCount;
        if (preferSparseGrounding) {
            std::vector<float> sparseCellTops(occupiedCells.size());
            groundingSource.terrainCellTopPoints(attachment->originX, attachment->originZ, step,
                                                 cellEdge, footprint, occupiedCells,
                                                 sparseCellTops);
            buildDiagnostics.sparseGroundCellCount = boundedDiagnosticCount(occupiedCells.size());
            for (size_t index = 0; index < occupiedCells.size(); ++index) {
                if (!std::isfinite(sparseCellTops[index])) {
                    throw std::invalid_argument("invalid sparse far terrain cell top");
                }
                cellTops[occupiedCells[index]] = vertexHeight(sparseCellTops[index]);
            }
        } else if (groundingSource.terrainCellTopGrid &&
                   occupiedCells.size() >= fullGridThreshold) {
            std::vector<float> denseCellTops(static_cast<size_t>(cellEdge * cellEdge));
            groundingSource.terrainCellTopGrid(attachment->originX, attachment->originZ, step,
                                               cellEdge, footprint, denseCellTops);
            buildDiagnostics.denseGroundGridSampleCount =
                boundedDiagnosticCount(denseCellTops.size());
            for (const uint32_t occupied : occupiedCells) {
                if (!std::isfinite(denseCellTops[occupied])) {
                    throw std::invalid_argument("invalid far terrain cell top grid");
                }
                cellTops[occupied] = vertexHeight(denseCellTops[occupied]);
            }
        } else if (key.step == FarTerrainStep::ONE) {
            // Step 1 displays the exact lower-left block-column sample. A
            // conservative macro bounds callback may describe a much wider
            // culling envelope, so it cannot ground flora for this tier.
            for (const uint32_t occupied : occupiedCells) {
                const int cellX = static_cast<int>(occupied % static_cast<uint32_t>(cellEdge));
                const int cellZ = static_cast<int>(occupied / static_cast<uint32_t>(cellEdge));
                const FarSurfaceSample surface = groundingSource.sample(
                    checkedCoordinateOffset(attachment->originX, cellX),
                    checkedCoordinateOffset(attachment->originZ, cellZ), footprint);
                cellTops[occupied] = stepOneTerrainCellHeight(surface);
            }
        } else if (groundingSource.cellBoundsGrid) {
            if (occupiedCells.size() >= fullGridThreshold) {
                // Dense grassland must not turn one tile into hundreds of
                // separate bounds batches. One rectangular query shares the
                // same model pages and hydrology batch as the base mesh while
                // still grounding every aggregate on its displayed cell.
                std::vector<FarTerrainCellBounds> bounds(static_cast<size_t>(cellEdge * cellEdge));
                groundingSource.cellBoundsGrid(attachment->originX, attachment->originZ, step,
                                               cellEdge, cellEdge, footprint, bounds);
                for (const uint32_t occupied : occupiedCells) {
                    cellTops[occupied] = authoritativeCellBounds(bounds[occupied]).top;
                }
            } else {
                for (size_t first = 0; first < occupiedCells.size();) {
                    const int cellZ = static_cast<int>(occupiedCells[first] / cellEdge);
                    const int firstCellX = static_cast<int>(occupiedCells[first] % cellEdge);
                    size_t pastLast = first + 1;
                    int lastCellX = firstCellX;
                    while (pastLast < occupiedCells.size()) {
                        const int nextCellZ = static_cast<int>(occupiedCells[pastLast] / cellEdge);
                        const int nextCellX = static_cast<int>(occupiedCells[pastLast] % cellEdge);
                        if (nextCellZ != cellZ || nextCellX != lastCellX + 1) break;
                        lastCellX = nextCellX;
                        ++pastLast;
                    }
                    const int runWidth = lastCellX - firstCellX + 1;
                    std::vector<FarTerrainCellBounds> bounds(static_cast<size_t>(runWidth));
                    groundingSource.cellBoundsGrid(
                        checkedCoordinateOffset(attachment->originX,
                                                static_cast<int64_t>(firstCellX) * step),
                        checkedCoordinateOffset(attachment->originZ,
                                                static_cast<int64_t>(cellZ) * step),
                        step, runWidth, 1, footprint, bounds);
                    for (int offset = 0; offset < runWidth; ++offset) {
                        cellTops[static_cast<size_t>(cellZ * cellEdge + firstCellX + offset)] =
                            authoritativeCellBounds(bounds[static_cast<size_t>(offset)]).top;
                    }
                    first = pastLast;
                }
            }
        } else {
            // Callback-only sources use the same four-corner fallback as the
            // base mesher. Sampling an arbitrary root point would disagree
            // with the displayed flat cell on every nonconstant field.
            const int sampleEdge = cellEdge + 1;
            std::vector<std::optional<FarSurfaceSample>> samples(
                static_cast<size_t>(sampleEdge * sampleEdge));
            const auto sampleAt = [&](int x, int z) -> FarSurfaceSample* {
                std::optional<FarSurfaceSample>& sample =
                    samples[static_cast<size_t>(z * sampleEdge + x)];
                if (!sample) {
                    sample = groundingSource.sample(
                        checkedCoordinateOffset(attachment->originX,
                                                static_cast<int64_t>(x) * step),
                        checkedCoordinateOffset(attachment->originZ,
                                                static_cast<int64_t>(z) * step),
                        footprint);
                }
                return &*sample;
            };
            for (const uint32_t occupied : occupiedCells) {
                const int cellX = static_cast<int>(occupied % cellEdge);
                const int cellZ = static_cast<int>(occupied / cellEdge);
                const std::array<FarSurfaceSample*, 4> corners = {
                    sampleAt(cellX, cellZ), sampleAt(cellX + 1, cellZ),
                    sampleAt(cellX + 1, cellZ + 1), sampleAt(cellX, cellZ + 1)};
                cellTops[static_cast<size_t>(occupied)] = fallbackCellBounds(corners, key.step).top;
            }
        }

        // Tile-edge cells are not displayed as flat cell tops at reduced
        // LODs. Their visible surface is the same fan of triangles used to
        // join the cell interior to canonical two-block boundary samples.
        // Ground trunks on that fan so a canopy cannot float above or sink
        // through the transition strip.
        constexpr int transitionSampleCount =
            FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1;
        enum TransitionEdge : size_t {
            TRANSITION_WEST = 0,
            TRANSITION_EAST = 1,
            TRANSITION_NORTH = 2,
            TRANSITION_SOUTH = 3,
        };
        const bool transitionTopologyEnabled = step >= FAR_TERRAIN_TRANSITION_SAMPLE_STEP;
        std::array<bool, 4> transitionBoundaryNeeded{};
        if (transitionTopologyEnabled) {
            for (const uint32_t occupied : occupiedCells) {
                const int cellX = static_cast<int>(occupied % cellEdge);
                const int cellZ = static_cast<int>(occupied / cellEdge);
                transitionBoundaryNeeded[TRANSITION_WEST] |= cellX == 0;
                transitionBoundaryNeeded[TRANSITION_EAST] |= cellX == cellEdge - 1;
                transitionBoundaryNeeded[TRANSITION_NORTH] |= cellZ == 0;
                transitionBoundaryNeeded[TRANSITION_SOUTH] |= cellZ == cellEdge - 1;
            }
        }
        std::array<std::array<float, transitionSampleCount>, 4> transitionBoundaryHeights{};
        const auto retainTransitionBoundary = [&](TransitionEdge edge, int64_t originX,
                                                  int64_t originZ, int sampleWidth,
                                                  int sampleHeight) {
            if (!transitionBoundaryNeeded[edge]) return;
            std::vector<FarTerrainGeometrySample> geometry(
                static_cast<size_t>(sampleWidth * sampleHeight));
            if (groundingSource.geometryGrid) {
                groundingSource.geometryGrid(originX, originZ, FAR_TERRAIN_TRANSITION_SAMPLE_STEP,
                                             FAR_TERRAIN_TRANSITION_SAMPLE_STEP, sampleWidth,
                                             sampleHeight, worldgen::SurfaceFootprint::BLOCK_2,
                                             geometry);
            } else {
                for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
                    for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                        geometry[static_cast<size_t>(sampleZ * sampleWidth + sampleX)] =
                            groundingSource
                                .sample(checkedCoordinateOffset(
                                            originX, static_cast<int64_t>(sampleX) *
                                                         FAR_TERRAIN_TRANSITION_SAMPLE_STEP),
                                        checkedCoordinateOffset(
                                            originZ, static_cast<int64_t>(sampleZ) *
                                                         FAR_TERRAIN_TRANSITION_SAMPLE_STEP),
                                        worldgen::SurfaceFootprint::BLOCK_2)
                                .geometry;
                    }
                }
            }
            buildDiagnostics.transitionGroundSampleCount += boundedDiagnosticCount(geometry.size());
            for (int coordinate = 0; coordinate < transitionSampleCount; ++coordinate) {
                transitionBoundaryHeights[edge][static_cast<size_t>(coordinate)] =
                    upwardBound(geometry[static_cast<size_t>(coordinate)].terrainHeight);
            }
        };
        if (transitionTopologyEnabled) {
            retainTransitionBoundary(TRANSITION_WEST, attachment->originX, attachment->originZ, 1,
                                     transitionSampleCount);
            retainTransitionBoundary(
                TRANSITION_EAST,
                checkedCoordinateOffset(attachment->originX, FAR_TERRAIN_TILE_EDGE),
                attachment->originZ, 1, transitionSampleCount);
            retainTransitionBoundary(TRANSITION_NORTH, attachment->originX, attachment->originZ,
                                     transitionSampleCount, 1);
            retainTransitionBoundary(
                TRANSITION_SOUTH, attachment->originX,
                checkedCoordinateOffset(attachment->originZ, FAR_TERRAIN_TILE_EDGE),
                transitionSampleCount, 1);
        }
        const auto transitionEdgeMaskAt = [&](int cellX, int cellZ) {
            if (!transitionTopologyEnabled) return uint32_t{0};
            uint32_t mask = 0;
            if (cellX == 0) mask |= 1U << static_cast<uint8_t>(FaceNormal::MINUS_X);
            if (cellX == cellEdge - 1) mask |= 1U << static_cast<uint8_t>(FaceNormal::PLUS_X);
            if (cellZ == 0) mask |= 1U << static_cast<uint8_t>(FaceNormal::MINUS_Z);
            if (cellZ == cellEdge - 1) mask |= 1U << static_cast<uint8_t>(FaceNormal::PLUS_Z);
            return mask;
        };
        const auto transitionBoundaryHeight = [&](uint8_t edgeMask, int localX, int localZ) {
            const auto selected = [edgeMask](FaceNormal edge) {
                return (edgeMask & (1U << static_cast<uint8_t>(edge))) != 0;
            };
            if (selected(FaceNormal::MINUS_X)) {
                return transitionBoundaryHeights[TRANSITION_WEST][static_cast<size_t>(
                    localZ / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
            }
            if (selected(FaceNormal::PLUS_X)) {
                return transitionBoundaryHeights[TRANSITION_EAST][static_cast<size_t>(
                    localZ / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
            }
            if (selected(FaceNormal::MINUS_Z)) {
                return transitionBoundaryHeights[TRANSITION_NORTH][static_cast<size_t>(
                    localX / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
            }
            if (selected(FaceNormal::PLUS_Z)) {
                return transitionBoundaryHeights[TRANSITION_SOUTH][static_cast<size_t>(
                    localX / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
            }
            throw std::logic_error("transition vertex does not own a tile edge");
        };
        const auto transitionSurfaceHeight = [&](int cellX, int cellZ, float pointX, float pointZ,
                                                 float cellTop) {
            const uint32_t edgeMask = transitionEdgeMaskAt(cellX, cellZ);
            if (edgeMask == 0) return cellTop;
            const FarTerrainTransitionTopology topology = farTerrainTransitionCellTopology(
                step, FAR_TERRAIN_TRANSITION_SAMPLE_STEP, edgeMask);
            const int localOriginX = cellX * step;
            const int localOriginZ = cellZ * step;
            const auto vertexHeightAt = [&](const FarTerrainTransitionVertex& vertex) {
                if (vertex.boundaryEdgeMask == 0) return cellTop;
                return transitionBoundaryHeight(vertex.boundaryEdgeMask, localOriginX + vertex.x,
                                                localOriginZ + vertex.z);
            };
            const auto signedArea = [](float ax, float az, float bx, float bz, float cx, float cz) {
                return static_cast<double>(bz - az) * static_cast<double>(cx - ax) -
                       static_cast<double>(bx - ax) * static_cast<double>(cz - az);
            };
            constexpr double insideEpsilon = 1.0e-6;
            for (size_t offset = 0; offset < topology.indexCount; offset += 3) {
                const FarTerrainTransitionVertex& first =
                    topology.vertices[topology.indices[offset]];
                const FarTerrainTransitionVertex& second =
                    topology.vertices[topology.indices[offset + 1]];
                const FarTerrainTransitionVertex& third =
                    topology.vertices[topology.indices[offset + 2]];
                const double area =
                    signedArea(first.x, first.z, second.x, second.z, third.x, third.z);
                const double firstWeight =
                    signedArea(pointX, pointZ, second.x, second.z, third.x, third.z) / area;
                const double secondWeight =
                    signedArea(first.x, first.z, pointX, pointZ, third.x, third.z) / area;
                const double thirdWeight =
                    signedArea(first.x, first.z, second.x, second.z, pointX, pointZ) / area;
                if (firstWeight < -insideEpsilon || secondWeight < -insideEpsilon ||
                    thirdWeight < -insideEpsilon) {
                    continue;
                }
                return static_cast<float>(firstWeight * vertexHeightAt(first) +
                                          secondWeight * vertexHeightAt(second) +
                                          thirdWeight * vertexHeightAt(third));
            }
            throw std::logic_error("far-canopy root lies outside its transition topology");
        };
        for (size_t index = 0; index < groundingAnchors.size(); ++index) {
            const auto [cellX, cellZ] = owningCell(groundingAnchors[index]);
            const float top = cellTops[static_cast<size_t>(cellZ * cellEdge + cellX)];
            if (!std::isfinite(top)) {
                throw std::logic_error("far-canopy owning cell was not grounded");
            }
            const float pointX =
                static_cast<float>(groundingAnchors[index].x - attachment->originX -
                                   static_cast<int64_t>(cellX) * step) +
                0.5F;
            const float pointZ =
                static_cast<float>(groundingAnchors[index].z - attachment->originZ -
                                   static_cast<int64_t>(cellZ) * step) +
                0.5F;
            groundHeights[index] =
                vertexHeight(transitionSurfaceHeight(cellX, cellZ, pointX, pointZ, top));
        }
    }

    ScopedCanopyDiagnosticTimer geometryTimer(buildDiagnostics.geometryMicroseconds);
    FarCanopyGeometrySink geometry;
    geometry.originX = attachment->originX;
    geometry.originZ = attachment->originZ;
    geometry.bounds = {geometry.originX,
                       geometry.originX + FAR_TERRAIN_TILE_EDGE,
                       geometry.originZ,
                       geometry.originZ + FAR_TERRAIN_TILE_EDGE,
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::lowest()};
    size_t groundIndex = 0;
    appendCanopyGeometry(geometry, canopies,
                         [&](const FarCanopy&) { return groundHeights[groundIndex++]; });
    appendFarFloraGeometry(geometry, flora, key.step,
                           [&](const FarFlora&) { return groundHeights[groundIndex++]; });
    if (!geometry.vertices.empty()) {
        attachment->bounds = geometry.bounds;
    }
    attachment->vertices = std::move(geometry.vertices);
    attachment->indices = std::move(geometry.indices);
    attachment->canopyAnchorCount = geometry.canopyAnchorCount;
    attachment->canopyImpostorQuadCount = geometry.canopyImpostorQuadCount;
    attachment->floraAnchorCount = geometry.floraAnchorCount;
    attachment->floraImpostorQuadCount = geometry.floraImpostorQuadCount;
    attachment->deterministicHash = hashCanopyAttachment(*attachment);
    geometryTimer.finish();
    totalTimer.finish();
    attachment->buildDiagnostics = buildDiagnostics;
    return attachment;
}

std::shared_ptr<const FarTerrainMesh>
FarTerrainMesher::buildInternal(FarTerrainKey key, const FarTerrainSource& source,
                                FarTerrainAuthorityQuality authorityQuality) {
    if (!source.sample) {
        throw std::invalid_argument("far terrain source is incomplete");
    }
    if (!validStep(key.step)) throw std::invalid_argument("unsupported far terrain LOD step");
    const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(key.step);
    const int step = farTerrainStepSize(key.step);
    const bool exactNearWater = key.step == FarTerrainStep::ONE ||
                                key.step == FarTerrainStep::TWO || key.step == FarTerrainStep::FOUR;
    const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
    const int sampleEdge = cellEdge + 1;
    auto mesh = std::make_shared<FarTerrainMesh>();
    mesh->key = key;
    mesh->authorityQuality = authorityQuality;
    mesh->exactAuthorityCompatible = authorityQuality == FarTerrainAuthorityQuality::FINAL;
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

    WaterTopologyTracker waterTopology;
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
    for (const FarSurfaceSample& sample : samples)
        waterTopology.observe(sample.geometry);
    for (int z = 0; z < sampleEdge; ++z) {
        for (int x = 0; x < sampleEdge; ++x) {
            if (x + 1 < sampleEdge)
                waterTopology.observeConnection(sampleAt(x, z), sampleAt(x + 1, z));
            if (z + 1 < sampleEdge)
                waterTopology.observeConnection(sampleAt(x, z), sampleAt(x, z + 1));
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
    const bool hasCanonicalWaterPoints =
        static_cast<bool>(source.canonicalWaterPoints) || static_cast<bool>(source.geometryPoints);
    const bool hasCanonicalWaterGrid =
        static_cast<bool>(source.canonicalWaterGrid) || static_cast<bool>(source.geometryGrid);
    std::unordered_map<ColumnPos, FarTerrainGeometrySample> canonicalWaterSamples;
    const worldgen::SurfaceFootprint canonicalWaterFootprint =
        source.planFreeCoarseAuthority && key.step != FarTerrainStep::ONE
            ? footprint
            : worldgen::SurfaceFootprint::BLOCK_1;
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
        } else if (source.canonicalWaterPoints) {
            source.canonicalWaterPoints(positions, canonicalWaterFootprint, geometry);
        } else if (source.geometryPoints) {
            source.geometryPoints(positions, canonicalWaterFootprint, geometry);
        } else if (source.canonicalWaterGrid) {
            for (size_t index = 0; index < positions.size(); ++index) {
                source.canonicalWaterGrid(positions[index].x, positions[index].z, 1, 1, 1, 1,
                                          canonicalWaterFootprint,
                                          std::span<FarTerrainGeometrySample>(&geometry[index], 1));
            }
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
            waterTopology.observe(geometry[index]);
        }
    };
    const GeometryFunction canonicalWaterGeometry = [&](int64_t worldX, int64_t worldZ) {
        const ColumnPos position{worldX, worldZ};
        if (const auto found = canonicalWaterSamples.find(position);
            found != canonicalWaterSamples.end()) {
            return found->second;
        }
        if (hasCanonicalWaterPoints &&
            (key.step == FarTerrainStep::EIGHT || key.step == FarTerrainStep::SIXTEEN ||
             key.step == FarTerrainStep::THIRTY_TWO)) {
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
    const int sharedWaterBoundaryDepth =
        key.step == FarTerrainStep::THIRTY_TWO ? worldgen::NATIVE_HYDROLOGY_RASTER_SPACING : step;
    enum WaterEdge : size_t { WEST = 0, EAST = 1, NORTH = 2, SOUTH = 3 };
    std::array<bool, 4> refineWaterBoundary{};
    const bool refineSharedWaterBoundary = key.step != FarTerrainStep::TWO;
    std::array<bool, 4> inspectWaterBoundary{};
    if (refineSharedWaterBoundary) {
        inspectWaterBoundary.fill(true);
        if (key.step == FarTerrainStep::THIRTY_TWO && source.cellBoundsGrid) {
            const auto coarseEdgeMayNeedRefinement = [&](WaterEdge edge) {
                for (int cell = 0; cell < cellEdge; ++cell) {
                    const int cellX = edge == WEST ? 0 : edge == EAST ? cellEdge - 1 : cell;
                    const int cellZ = edge == NORTH ? 0 : edge == SOUTH ? cellEdge - 1 : cell;
                    const FarTerrainCellBounds& bounds = authoritativeBoundsAt(cellX, cellZ);
                    if (bounds.waterTopologyPossible || bounds.volcanicWaterPossible ||
                        bounds.waterfallPossible) {
                        return true;
                    }
                }

                bool sawWet = false;
                bool sawDry = false;
                std::optional<FarWaterAuthority> previousWet;
                for (int coordinate = 0; coordinate <= cellEdge; ++coordinate) {
                    const int sampleX = edge == WEST ? 0 : edge == EAST ? cellEdge : coordinate;
                    const int sampleZ = edge == NORTH ? 0 : edge == SOUTH ? cellEdge : coordinate;
                    const FarTerrainGeometrySample& sample = sampleAt(sampleX, sampleZ);
                    const bool wet = sampleIsWet(sample);
                    sawWet = sawWet || wet;
                    sawDry = sawDry || !wet;
                    const FarWaterAuthority authority = waterAuthority(sample, wet);
                    if (wet && previousWet &&
                        !waterAuthoritiesConnect(*previousWet, authority,
                                                 static_cast<float>(step))) {
                        return true;
                    }
                    previousWet = wet ? std::optional<FarWaterAuthority>{authority} : std::nullopt;
                }
                return sawWet && sawDry;
            };
            for (size_t edge = 0; edge < inspectWaterBoundary.size(); ++edge) {
                inspectWaterBoundary[edge] =
                    coarseEdgeMayNeedRefinement(static_cast<WaterEdge>(edge));
            }
        }
    }
    if (refineSharedWaterBoundary && hasCanonicalWaterGrid) {
        constexpr int EDGE_SAMPLE_COUNT = FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP + 1;
        const auto retainStrip = [&](int64_t originX, int64_t originZ, int sampleWidth,
                                     int sampleHeight) {
            std::vector<FarTerrainGeometrySample> strip(
                static_cast<size_t>(sampleWidth * sampleHeight));
            const auto& sampleGrid =
                source.canonicalWaterGrid ? source.canonicalWaterGrid : source.geometryGrid;
            sampleGrid(originX, originZ, SHARED_WATER_EDGE_STEP, SHARED_WATER_EDGE_STEP,
                       sampleWidth, sampleHeight, canonicalWaterFootprint, strip);
            for (int z = 0; z < sampleHeight; ++z) {
                for (int x = 0; x < sampleWidth; ++x) {
                    const ColumnPos position{
                        originX + static_cast<int64_t>(x * SHARED_WATER_EDGE_STEP),
                        originZ + static_cast<int64_t>(z * SHARED_WATER_EDGE_STEP),
                    };
                    canonicalWaterSamples.emplace(position,
                                                  strip[static_cast<size_t>(z * sampleWidth + x)]);
                    waterTopology.observe(strip[static_cast<size_t>(z * sampleWidth + x)]);
                }
            }
        };
        if (inspectWaterBoundary[WEST])
            retainStrip(mesh->originX, mesh->originZ, 1, EDGE_SAMPLE_COUNT);
        if (inspectWaterBoundary[EAST])
            retainStrip(mesh->originX + FAR_TERRAIN_TILE_EDGE, mesh->originZ, 1, EDGE_SAMPLE_COUNT);
        // Active vertical strips own their two corners. Horizontal strips
        // retain any corner whose vertical strip was proven unnecessary, so
        // every inspected face coordinate is sampled exactly once.
        const int horizontalOriginOffset = inspectWaterBoundary[WEST] ? SHARED_WATER_EDGE_STEP : 0;
        const int horizontalSampleCount = EDGE_SAMPLE_COUNT -
                                          static_cast<int>(inspectWaterBoundary[WEST]) -
                                          static_cast<int>(inspectWaterBoundary[EAST]);
        if (inspectWaterBoundary[NORTH])
            retainStrip(mesh->originX + horizontalOriginOffset, mesh->originZ,
                        horizontalSampleCount, 1);
        if (inspectWaterBoundary[SOUTH])
            retainStrip(mesh->originX + horizontalOriginOffset,
                        mesh->originZ + FAR_TERRAIN_TILE_EDGE, horizontalSampleCount, 1);
    } else if (refineSharedWaterBoundary && hasCanonicalWaterPoints) {
        constexpr int EDGE_SAMPLE_COUNT = FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP + 1;
        std::vector<ColumnPos> edgePositions;
        edgePositions.reserve(static_cast<size_t>(EDGE_SAMPLE_COUNT * 4 - 4));
        for (int coordinate = 0; coordinate <= FAR_TERRAIN_TILE_EDGE;
             coordinate += SHARED_WATER_EDGE_STEP) {
            if (inspectWaterBoundary[WEST])
                edgePositions.push_back({mesh->originX, mesh->originZ + coordinate});
            if (inspectWaterBoundary[EAST])
                edgePositions.push_back(
                    {mesh->originX + FAR_TERRAIN_TILE_EDGE, mesh->originZ + coordinate});
            const bool westCornerOwned = coordinate == 0 && inspectWaterBoundary[WEST];
            const bool eastCornerOwned =
                coordinate == FAR_TERRAIN_TILE_EDGE && inspectWaterBoundary[EAST];
            if (!westCornerOwned && !eastCornerOwned) {
                if (inspectWaterBoundary[NORTH])
                    edgePositions.push_back({mesh->originX + coordinate, mesh->originZ});
                if (inspectWaterBoundary[SOUTH])
                    edgePositions.push_back(
                        {mesh->originX + coordinate, mesh->originZ + FAR_TERRAIN_TILE_EDGE});
            }
        }
        preloadCanonicalWater(edgePositions);
    }
    auto sharedWaterSample = [&](int localX, int localZ) -> const FarTerrainGeometrySample& {
        const ColumnPos key{mesh->originX + localX, mesh->originZ + localZ};
        auto found = canonicalWaterSamples.find(key);
        if (found != canonicalWaterSamples.end()) return found->second;
        canonicalWaterGeometry(key.x, key.z);
        return canonicalWaterSamples.find(key)->second;
    };
    auto probeWaterBoundary = [&](WaterEdge edge) {
        if (!inspectWaterBoundary[edge]) return;
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
    const auto step32SharedBoundaryIntersects = [&](int x0, int z0, int x1, int z1) {
        if (key.step != FarTerrainStep::THIRTY_TWO) return false;
        return (refineWaterBoundary[WEST] && x0 < sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[EAST] &&
                x1 > FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[NORTH] && z0 < sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[SOUTH] &&
                z1 > FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth);
    };
    const auto step32SharedBoundaryOwns = [&](int x0, int z0, int x1, int z1) {
        if (key.step != FarTerrainStep::THIRTY_TWO) return false;
        return (refineWaterBoundary[WEST] && x1 <= sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[EAST] &&
                x0 >= FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[NORTH] && z1 <= sharedWaterBoundaryDepth) ||
               (refineWaterBoundary[SOUTH] &&
                z0 >= FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth);
    };
    if (refineSharedWaterBoundary && hasCanonicalWaterPoints) {
        std::vector<ColumnPos> boundaryDecisionPoints;
        boundaryDecisionPoints.reserve(
            static_cast<size_t>((FAR_TERRAIN_TILE_EDGE / SHARED_WATER_EDGE_STEP) * 8));
        const auto appendBoundaryDecisionLine = [&](WaterEdge edge) {
            if (!refineWaterBoundary[edge]) return;
            for (int coordinate = 0; coordinate < FAR_TERRAIN_TILE_EDGE;
                 coordinate += SHARED_WATER_EDGE_STEP) {
                if (edge == WEST || edge == EAST) {
                    const int innerX = edge == WEST
                                           ? sharedWaterBoundaryDepth
                                           : FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth;
                    const int centerX = edge == WEST
                                            ? sharedWaterBoundaryDepth / 2
                                            : FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth / 2;
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + innerX, mesh->originZ + coordinate});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + innerX,
                         mesh->originZ + coordinate + SHARED_WATER_EDGE_STEP});
                    boundaryDecisionPoints.push_back(
                        {mesh->originX + centerX,
                         mesh->originZ + coordinate + SHARED_WATER_EDGE_STEP / 2});
                } else {
                    const int innerZ = edge == NORTH
                                           ? sharedWaterBoundaryDepth
                                           : FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth;
                    const int centerZ = edge == NORTH
                                            ? sharedWaterBoundaryDepth / 2
                                            : FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth / 2;
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

    if (hasCanonicalWaterPoints && step >= 8) {
        std::vector<ColumnPos> centers;
        centers.reserve(static_cast<size_t>(cellEdge * cellEdge));
        for (int z = 0; z < cellEdge; ++z) {
            for (int x = 0; x < cellEdge; ++x) {
                const bool topologyPossible =
                    source.cellBoundsGrid && authoritativeBoundsAt(x, z).waterTopologyPossible;
                const bool anyWet =
                    sampleIsWet(sampleAt(x, z)) || sampleIsWet(sampleAt(x + 1, z)) ||
                    sampleIsWet(sampleAt(x + 1, z + 1)) || sampleIsWet(sampleAt(x, z + 1));
                if (!anyWet && !topologyPossible) continue;
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
    const auto surfaceMaterialAt = [&](const FarSurfaceSample& surface, int64_t worldX,
                                       int64_t worldZ) {
        const double rank = source.materialRank ? source.materialRank(worldX, worldZ)
                                                : materialRank(worldX, worldZ);
        BlockType material =
            worldgen::surface_material::selectMaterial(surface.materialPalette, rank);
        return material == BlockType::AIR ? BlockType::STONE : material;
    };
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const std::array<FarSurfaceSample*, 4> cornerSamples = {
                &surfaceAt(x, z), &surfaceAt(x + 1, z), &surfaceAt(x + 1, z + 1),
                &surfaceAt(x, z + 1)};
            ResolvedFarTerrainCellBounds bounds =
                source.cellBoundsGrid ? authoritativeCellBounds(authoritativeBoundsAt(x, z))
                                      : fallbackCellBounds(cornerSamples, key.step);
            if (key.step == FarTerrainStep::ONE) {
                // The dense sample grid is the emitted exact-column authority
                // at step 1. Cell bounds may carry a broader macro or native
                // envelope for culling and water recovery, but they must not
                // replace the visible top and intersect exact cube geometry.
                bounds.top = terrainCellHeight(cornerSamples, key.step);
                bounds.minimum = std::min(bounds.minimum, bounds.top);
                bounds.maximum = std::max(bounds.maximum, bounds.top);
            }
            cell.terrain.fill(bounds.top);
            cell.maximumTerrain = bounds.maximum;
            cell.waterTopologyPossible = bounds.waterTopologyPossible;
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
            const int64_t materialX = worldX + step / 2;
            const int64_t materialZ = worldZ + step / 2;
            cell.material = surfaceMaterialAt(surfaceAt(x, z), materialX, materialZ);
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
                if (!exactNearWater && (cornerWater || cell.waterTopologyPossible) &&
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
                // The cell-bounds authority can identify an analytical
                // channel or shoreline crossing even when all four displayed
                // corners and the center are dry. Keep the cell eligible for
                // canonical subdivision so that topology is not reduced to a
                // phase-dependent point sample.
                cell.water = cornerWater || cell.centerWet || cell.waterTopologyPossible;
            }
        }
    }

    const auto sharedRectangleTopologyPossible = [&](int x0, int z0, int x1, int z1) {
        if (authoritativeBounds.empty()) return false;
        const int centerX = (x0 + x1 - 1) / 2;
        const int centerZ = (z0 + z1 - 1) / 2;
        const int cellX = std::clamp(centerX / step, 0, cellEdge - 1);
        const int cellZ = std::clamp(centerZ / step, 0, cellEdge - 1);
        const FarTerrainCellBounds& bounds = authoritativeBoundsAt(cellX, cellZ);
        return bounds.waterTopologyPossible || bounds.volcanicWaterPossible ||
               bounds.waterfallPossible;
    };

    // Coarse parents resolve only ambiguous water rectangles at the
    // shoreline page's canonical two- or four-block spacing. Gather every
    // corner and center first, then let the generator group the immutable
    // point set by basin in one call. This preserves exact body identity and
    // shared tile faces without the former thousands of scalar samples.
    if (hasCanonicalWaterPoints && step >= 8) {
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
            if (sharedRectangleTopologyPossible(x0, z0, x1, z1)) return true;
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
        if (key.step != FarTerrainStep::THIRTY_TWO) {
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
                    const int resolution = cell.waterTopologyPossible ? 1 : 2;
                    appendContourRectangle(x * step, z * step, (x + 1) * step, (z + 1) * step,
                                           resolution);
                }
            }
        }
        const auto appendBoundaryRectangles = [&](WaterEdge edge) {
            if (!refineWaterBoundary[edge]) return;
            const bool horizontal = edge == NORTH || edge == SOUTH;
            const int begin =
                horizontal && refineWaterBoundary[WEST] ? sharedWaterBoundaryDepth : 0;
            const int end = horizontal && refineWaterBoundary[EAST]
                                ? FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth
                                : FAR_TERRAIN_TILE_EDGE;
            for (int coordinate = begin; coordinate < end; coordinate += SHARED_WATER_EDGE_STEP) {
                int x0 = coordinate;
                int z0 = coordinate;
                int x1 = coordinate + SHARED_WATER_EDGE_STEP;
                int z1 = coordinate + SHARED_WATER_EDGE_STEP;
                if (edge == WEST) {
                    x0 = 0;
                    x1 = sharedWaterBoundaryDepth;
                } else if (edge == EAST) {
                    x0 = FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth;
                    x1 = FAR_TERRAIN_TILE_EDGE;
                } else if (edge == NORTH) {
                    z0 = 0;
                    z1 = sharedWaterBoundaryDepth;
                } else {
                    z0 = FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth;
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
    if (hasCanonicalWaterPoints &&
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

    constexpr int TRANSITION_SAMPLE_COUNT =
        FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1;
    const bool transitionTopologyEnabled = step >= FAR_TERRAIN_TRANSITION_SAMPLE_STEP;
    std::array<std::array<float, TRANSITION_SAMPLE_COUNT>, 4> transitionBoundaryHeights{};
    const auto retainTransitionBoundary = [&](WaterEdge edge, int64_t originX, int64_t originZ,
                                              int sampleWidth, int sampleHeight) {
        std::vector<FarTerrainGeometrySample> geometry(
            static_cast<size_t>(sampleWidth * sampleHeight));
        if (source.geometryGrid) {
            source.geometryGrid(originX, originZ, FAR_TERRAIN_TRANSITION_SAMPLE_STEP,
                                FAR_TERRAIN_TRANSITION_SAMPLE_STEP, sampleWidth, sampleHeight,
                                worldgen::SurfaceFootprint::BLOCK_2, geometry);
        } else {
            for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
                for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                    geometry[static_cast<size_t>(sampleZ * sampleWidth + sampleX)] =
                        source
                            .sample(originX + sampleX * FAR_TERRAIN_TRANSITION_SAMPLE_STEP,
                                    originZ + sampleZ * FAR_TERRAIN_TRANSITION_SAMPLE_STEP,
                                    worldgen::SurfaceFootprint::BLOCK_2)
                            .geometry;
                }
            }
        }
        for (int coordinate = 0; coordinate < TRANSITION_SAMPLE_COUNT; ++coordinate) {
            transitionBoundaryHeights[edge][static_cast<size_t>(coordinate)] =
                upwardBound(geometry[static_cast<size_t>(coordinate)].terrainHeight);
        }
    };
    // Step 1 does not need a transition strip, but it still records the same
    // canonical two-block boundary as every coarser payload. The protected
    // entry patch can therefore validate its complete visual surface before
    // exact FINAL ownership clips any far fragments.
    retainTransitionBoundary(WEST, mesh->originX, mesh->originZ, 1, TRANSITION_SAMPLE_COUNT);
    retainTransitionBoundary(EAST, mesh->originX + FAR_TERRAIN_TILE_EDGE, mesh->originZ, 1,
                             TRANSITION_SAMPLE_COUNT);
    retainTransitionBoundary(NORTH, mesh->originX, mesh->originZ, TRANSITION_SAMPLE_COUNT, 1);
    retainTransitionBoundary(SOUTH, mesh->originX, mesh->originZ + FAR_TERRAIN_TILE_EDGE,
                             TRANSITION_SAMPLE_COUNT, 1);
    for (size_t edge = 0; edge < mesh->surfaceBoundary.heightHashes.size(); ++edge) {
        mesh->surfaceBoundary.heightHashes[edge] = hashSurfaceBoundary(std::span<const float>(
            transitionBoundaryHeights[edge].data(), transitionBoundaryHeights[edge].size()));
    }
    mesh->surfaceBoundary.valid = true;

    const auto transitionEdgeMaskAt = [&](int x, int z) {
        if (!transitionTopologyEnabled) return uint32_t{0};
        uint32_t mask = 0;
        if (x == 0) mask |= 1U << static_cast<uint8_t>(FaceNormal::MINUS_X);
        if (x == cellEdge - 1) mask |= 1U << static_cast<uint8_t>(FaceNormal::PLUS_X);
        if (z == 0) mask |= 1U << static_cast<uint8_t>(FaceNormal::MINUS_Z);
        if (z == cellEdge - 1) mask |= 1U << static_cast<uint8_t>(FaceNormal::PLUS_Z);
        return mask;
    };
    const auto transitionBoundaryHeight = [&](uint8_t edgeMask, int localX, int localZ) {
        const auto edgeSelected = [edgeMask](FaceNormal edge) {
            return (edgeMask & (1U << static_cast<uint8_t>(edge))) != 0;
        };
        if (edgeSelected(FaceNormal::MINUS_X)) {
            return transitionBoundaryHeights[WEST][static_cast<size_t>(
                localZ / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
        }
        if (edgeSelected(FaceNormal::PLUS_X)) {
            return transitionBoundaryHeights[EAST][static_cast<size_t>(
                localZ / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
        }
        if (edgeSelected(FaceNormal::MINUS_Z)) {
            return transitionBoundaryHeights[NORTH][static_cast<size_t>(
                localX / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
        }
        if (edgeSelected(FaceNormal::PLUS_Z)) {
            return transitionBoundaryHeights[SOUTH][static_cast<size_t>(
                localX / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
        }
        throw std::logic_error("transition vertex does not own a tile edge");
    };
    std::array<std::optional<FarTerrainTransitionTopology>, 16> transitionTopologies;
    const auto transitionTopology = [&](uint32_t edgeMask) -> const FarTerrainTransitionTopology& {
        std::optional<FarTerrainTransitionTopology>& cached = transitionTopologies[edgeMask];
        if (!cached) {
            cached = farTerrainTransitionCellTopology(step, FAR_TERRAIN_TRANSITION_SAMPLE_STEP,
                                                      edgeMask);
        }
        return *cached;
    };
    struct TransitionCell {
        int x = 0;
        int z = 0;
        uint32_t edgeMask = 0;
    };
    std::vector<TransitionCell> transitionCells;
    transitionCells.reserve(static_cast<size_t>(cellEdge * 4 - 4));

    std::vector<uint8_t> merged(static_cast<size_t>(cellEdge * cellEdge), 0);
    for (int z = 0; z < cellEdge; ++z) {
        for (int x = 0; x < cellEdge; ++x) {
            FarCell& cell = cellAt(x, z);
            const size_t cellIndex = static_cast<size_t>(z * cellEdge + x);
            if (merged[cellIndex] != 0) continue;
            const uint32_t transitionMask = transitionEdgeMaskAt(x, z);
            if (transitionMask != 0) {
                merged[cellIndex] = 1;
                ++mesh->mergedTerrainCellCount;
                transitionCells.push_back({x, z, transitionMask});
                continue;
            }
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
                   transitionEdgeMaskAt(x + width, z) == 0 &&
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
                        transitionEdgeMaskAt(x + offset, z + depth) != 0 ||
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

    for (const TransitionCell transition : transitionCells) {
        const FarCell& cell = cellAt(transition.x, transition.z);
        const int localOriginX = transition.x * step;
        const int localOriginZ = transition.z * step;
        const FarTerrainTransitionTopology& topology = transitionTopology(transition.edgeMask);
        pushTerrainTransitionTop(*mesh, cell.material, static_cast<float>(localOriginX),
                                 static_cast<float>(localOriginZ), topology,
                                 [&](const FarTerrainTransitionVertex& point) {
                                     if (point.boundaryEdgeMask == 0) return cell.terrain[0];
                                     return transitionBoundaryHeight(point.boundaryEdgeMask,
                                                                     localOriginX + point.x,
                                                                     localOriginZ + point.z);
                                 });
        const int patchX = localOriginX / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int patchZ = localOriginZ / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        FarTerrainBounds& patch =
            mesh->occluderPatches[static_cast<size_t>(patchZ * PATCHES_PER_EDGE + patchX)];
        for (const FarTerrainTransitionVertex& point :
             std::span(topology.vertices).first(topology.vertexCount)) {
            if (point.boundaryEdgeMask == 0) continue;
            const float height = transitionBoundaryHeight(
                point.boundaryEdgeMask, localOriginX + point.x, localOriginZ + point.z);
            patch.minY = std::min(patch.minY, height);
            patch.maxY = std::max(patch.maxY, height);
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
                const bool northTransition = z == 0;
                const bool southTransition = z == cellEdge - 1;
                if (transitionTopologyEnabled && (northTransition || southTransition)) {
                    const WaterEdge edge = northTransition ? NORTH : SOUTH;
                    const float outerZ =
                        northTransition ? 0.0F : static_cast<float>(FAR_TERRAIN_TILE_EDGE);
                    const float innerZ = northTransition ? static_cast<float>(step)
                                                         : outerZ - static_cast<float>(step);
                    const float outerHeight = transitionBoundaryHeights[edge][static_cast<size_t>(
                        static_cast<int>(x1) / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
                    const float southZ = std::max(outerZ, innerZ);
                    const float northZ = std::min(outerZ, innerZ);
                    const auto profileHeight = [&](float coordinate, float innerHeight) {
                        return coordinate == outerZ ? outerHeight : innerHeight;
                    };
                    if (cell.terrain[0] > east.terrain[0]) {
                        pushTerrainDiscontinuityWedge(*mesh, FaceNormal::PLUS_X, cell.material, x1,
                                                      southZ, x1, northZ,
                                                      profileHeight(southZ, east.terrain[0]),
                                                      profileHeight(northZ, east.terrain[0]),
                                                      profileHeight(southZ, cell.terrain[0]),
                                                      profileHeight(northZ, cell.terrain[0]));
                    } else if (east.terrain[0] > cell.terrain[0]) {
                        pushTerrainDiscontinuityWedge(*mesh, FaceNormal::MINUS_X, east.material, x1,
                                                      northZ, x1, southZ,
                                                      profileHeight(northZ, cell.terrain[0]),
                                                      profileHeight(southZ, cell.terrain[0]),
                                                      profileHeight(northZ, east.terrain[0]),
                                                      profileHeight(southZ, east.terrain[0]));
                    }
                } else if (cell.terrain[0] > east.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::PLUS_X, cell.material, x1, z1, x1, z0,
                                     east.terrain[0], cell.terrain[0]);
                } else if (east.terrain[0] > cell.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::MINUS_X, east.material, x1, z0, x1, z1,
                                     cell.terrain[0], east.terrain[0]);
                }
            }
            if (z + 1 < cellEdge) {
                const FarCell& south = cellAt(x, z + 1);
                const bool westTransition = x == 0;
                const bool eastTransition = x == cellEdge - 1;
                if (transitionTopologyEnabled && (westTransition || eastTransition)) {
                    const WaterEdge edge = westTransition ? WEST : EAST;
                    const float outerX =
                        westTransition ? 0.0F : static_cast<float>(FAR_TERRAIN_TILE_EDGE);
                    const float innerX = westTransition ? static_cast<float>(step)
                                                        : outerX - static_cast<float>(step);
                    const float outerHeight = transitionBoundaryHeights[edge][static_cast<size_t>(
                        static_cast<int>(z1) / FAR_TERRAIN_TRANSITION_SAMPLE_STEP)];
                    const float westX = std::min(outerX, innerX);
                    const float eastX = std::max(outerX, innerX);
                    const auto profileHeight = [&](float coordinate, float innerHeight) {
                        return coordinate == outerX ? outerHeight : innerHeight;
                    };
                    if (cell.terrain[0] > south.terrain[0]) {
                        pushTerrainDiscontinuityWedge(*mesh, FaceNormal::PLUS_Z, cell.material,
                                                      westX, z1, eastX, z1,
                                                      profileHeight(westX, south.terrain[0]),
                                                      profileHeight(eastX, south.terrain[0]),
                                                      profileHeight(westX, cell.terrain[0]),
                                                      profileHeight(eastX, cell.terrain[0]));
                    } else if (south.terrain[0] > cell.terrain[0]) {
                        pushTerrainDiscontinuityWedge(*mesh, FaceNormal::MINUS_Z, south.material,
                                                      eastX, z1, westX, z1,
                                                      profileHeight(eastX, cell.terrain[0]),
                                                      profileHeight(westX, cell.terrain[0]),
                                                      profileHeight(eastX, south.terrain[0]),
                                                      profileHeight(westX, south.terrain[0]));
                    }
                } else if (cell.terrain[0] > south.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::PLUS_Z, cell.material, x0, z1, x1, z1,
                                     south.terrain[0], cell.terrain[0]);
                } else if (south.terrain[0] > cell.terrain[0]) {
                    pushTerrainRiser(*mesh, FaceNormal::MINUS_Z, south.material, x1, z1, x0, z1,
                                     cell.terrain[0], south.terrain[0]);
                }
            }
        }
    }

    if (key.step == FarTerrainStep::ONE) {
        // Step 1 has no transition strip, so adjacent terminal block columns
        // can meet at different exact heights. The tile on the negative side
        // owns the shared vertical closure at its positive X and Z faces. Its
        // dense sample grid already contains the neighboring positive column,
        // which keeps the closure exact without another scalar generator walk.
        const float boundary = static_cast<float>(FAR_TERRAIN_TILE_EDGE);
        for (int z = 0; z < cellEdge; ++z) {
            const FarCell& cell = cellAt(cellEdge - 1, z);
            const FarSurfaceSample& east = surfaceAt(cellEdge, z);
            const float eastTop = stepOneTerrainCellHeight(east);
            const float z0 = static_cast<float>(z);
            const float z1 = z0 + 1.0F;
            if (cell.terrain[0] > eastTop) {
                pushTerrainRiser(*mesh, FaceNormal::PLUS_X, cell.material, boundary, z1, boundary,
                                 z0, eastTop, cell.terrain[0]);
            } else if (eastTop > cell.terrain[0]) {
                const BlockType eastMaterial =
                    surfaceMaterialAt(east, mesh->originX + FAR_TERRAIN_TILE_EDGE,
                                      mesh->originZ + static_cast<int64_t>(z));
                pushTerrainRiser(*mesh, FaceNormal::MINUS_X, eastMaterial, boundary, z0, boundary,
                                 z1, cell.terrain[0], eastTop);
            }
        }
        for (int x = 0; x < cellEdge; ++x) {
            const FarCell& cell = cellAt(x, cellEdge - 1);
            const FarSurfaceSample& south = surfaceAt(x, cellEdge);
            const float southTop = stepOneTerrainCellHeight(south);
            const float x0 = static_cast<float>(x);
            const float x1 = x0 + 1.0F;
            if (cell.terrain[0] > southTop) {
                pushTerrainRiser(*mesh, FaceNormal::PLUS_Z, cell.material, x0, boundary, x1,
                                 boundary, southTop, cell.terrain[0]);
            } else if (southTop > cell.terrain[0]) {
                const BlockType southMaterial =
                    surfaceMaterialAt(south, mesh->originX + static_cast<int64_t>(x),
                                      mesh->originZ + FAR_TERRAIN_TILE_EDGE);
                pushTerrainRiser(*mesh, FaceNormal::MINUS_Z, southMaterial, x1, boundary, x0,
                                 boundary, cell.terrain[0], southTop);
            }
        }
    }

    // Coarser tile faces terminate on the same two-block canonical polyline.
    // Their shared transition top supplies complete closure without an
    // exterior panel.

    mesh->opaqueIndexCount = static_cast<uint32_t>(mesh->indices.size());

    // Lake outlets retain the downstream body's standing surface and add a
    // separately owned falling prism. The half-open tile containing the
    // outlet anchor owns the complete narrow prism, even when it crosses a
    // tile face, so no adjacent tile duplicates a coplanar waterfall wall.
    //
    // A fall anchor can be narrower than a far LOD sample and sit between two
    // phase-aligned terrain vertices. Scan the native four-block hydrology
    // lattice in addition to the displayed lattice, then deduplicate exact
    // anchors before emitting. The scan is available only from batched
    // authority callbacks, so ordinary sample-only sources retain their
    // existing lightweight behavior.
    struct WaterfallAnchor {
        int localX = 0;
        int localZ = 0;
        FarTerrainGeometrySample sample;
    };
    std::vector<WaterfallAnchor> waterfallAnchors;
    const auto retainWaterfallAnchor = [&](int localX, int localZ,
                                           const FarTerrainGeometrySample& sample) {
        if (localX < 0 || localX >= FAR_TERRAIN_TILE_EDGE || localZ < 0 ||
            localZ >= FAR_TERRAIN_TILE_EDGE || !sample.waterfall || !sample.waterfallAnchor ||
            sample.waterfallTop < sample.waterfallBottom + 0.5) {
            return;
        }
        const auto duplicate =
            std::ranges::find_if(waterfallAnchors, [localX, localZ](const WaterfallAnchor& anchor) {
                return anchor.localX == localX && anchor.localZ == localZ;
            });
        if (duplicate == waterfallAnchors.end()) {
            waterfallAnchors.push_back({.localX = localX, .localZ = localZ, .sample = sample});
        }
    };
    for (int z = 0; z < sampleEdge - 1; ++z) {
        for (int x = 0; x < sampleEdge - 1; ++x) {
            retainWaterfallAnchor(x * step, z * step, sampleAt(x, z));
        }
    }
    constexpr int WATERFALL_ANCHOR_SPACING = worldgen::NATIVE_HYDROLOGY_RASTER_SPACING;
    static_assert(FAR_TERRAIN_TILE_EDGE % WATERFALL_ANCHOR_SPACING == 0);
    const bool hasBatchedWaterfallAuthority =
        source.waterAuthorityGrid || source.waterAuthorityPoints || source.canonicalWaterGrid ||
        source.canonicalWaterPoints || source.geometryGrid || source.geometryPoints;
    if (key.step != FarTerrainStep::THIRTY_TWO && step > WATERFALL_ANCHOR_SPACING &&
        source.cellBoundsGrid && hasBatchedWaterfallAuthority) {
        // Probe only channel-adjacent cells, not the full 256-block tile.
        // A native channel cell is four blocks wide, so this lattice includes
        // every possible canonical anchor while the half-open cell ownership
        // keeps a boundary anchor in exactly one probe.
        const int probeEdge = step / WATERFALL_ANCHOR_SPACING;
        for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
            for (int cellX = 0; cellX < cellEdge; ++cellX) {
                if (!authoritativeBoundsAt(cellX, cellZ).waterfallPossible) continue;
                const int localOriginX = cellX * step;
                const int localOriginZ = cellZ * step;
                const bool finalWaterfallGeometry =
                    authoritativeBoundsAt(cellX, cellZ).volcanicWaterPossible;
                std::vector<FarTerrainGeometrySample> anchorSamples(
                    static_cast<size_t>(probeEdge * probeEdge));
                // The native authority already owns legal falls. Only a
                // volcanic candidate needs emitted-terrain contact before
                // publication; requesting exact density for every routed
                // channel was the dominant cold-horizon cost.
                if (!finalWaterfallGeometry && source.canonicalWaterGrid) {
                    source.canonicalWaterGrid(
                        mesh->originX + localOriginX, mesh->originZ + localOriginZ,
                        WATERFALL_ANCHOR_SPACING, WATERFALL_ANCHOR_SPACING, probeEdge, probeEdge,
                        canonicalWaterFootprint, anchorSamples);
                } else if (source.waterAuthorityGrid) {
                    source.waterAuthorityGrid(
                        mesh->originX + localOriginX, mesh->originZ + localOriginZ,
                        WATERFALL_ANCHOR_SPACING, WATERFALL_ANCHOR_SPACING, probeEdge, probeEdge,
                        canonicalWaterFootprint, anchorSamples);
                } else if (source.waterAuthorityPoints || source.canonicalWaterPoints ||
                           source.geometryPoints) {
                    std::vector<ColumnPos> positions;
                    positions.reserve(anchorSamples.size());
                    for (int probeZ = 0; probeZ < probeEdge; ++probeZ) {
                        for (int probeX = 0; probeX < probeEdge; ++probeX) {
                            positions.push_back(
                                {mesh->originX + localOriginX + probeX * WATERFALL_ANCHOR_SPACING,
                                 mesh->originZ + localOriginZ + probeZ * WATERFALL_ANCHOR_SPACING});
                        }
                    }
                    if (!finalWaterfallGeometry && source.canonicalWaterPoints) {
                        source.canonicalWaterPoints(positions, canonicalWaterFootprint,
                                                    anchorSamples);
                    } else if (source.waterAuthorityPoints) {
                        source.waterAuthorityPoints(positions, canonicalWaterFootprint,
                                                    anchorSamples);
                    } else if (source.canonicalWaterPoints) {
                        source.canonicalWaterPoints(positions, canonicalWaterFootprint,
                                                    anchorSamples);
                    } else {
                        source.geometryPoints(positions, canonicalWaterFootprint, anchorSamples);
                    }
                } else if (source.canonicalWaterGrid) {
                    source.canonicalWaterGrid(
                        mesh->originX + localOriginX, mesh->originZ + localOriginZ,
                        WATERFALL_ANCHOR_SPACING, WATERFALL_ANCHOR_SPACING, probeEdge, probeEdge,
                        canonicalWaterFootprint, anchorSamples);
                } else {
                    source.geometryGrid(mesh->originX + localOriginX, mesh->originZ + localOriginZ,
                                        WATERFALL_ANCHOR_SPACING, WATERFALL_ANCHOR_SPACING,
                                        probeEdge, probeEdge, canonicalWaterFootprint,
                                        anchorSamples);
                }
                for (int probeZ = 0; probeZ < probeEdge; ++probeZ) {
                    for (int probeX = 0; probeX < probeEdge; ++probeX) {
                        retainWaterfallAnchor(
                            localOriginX + probeX * WATERFALL_ANCHOR_SPACING,
                            localOriginZ + probeZ * WATERFALL_ANCHOR_SPACING,
                            anchorSamples[static_cast<size_t>(probeZ * probeEdge + probeX)]);
                    }
                }
            }
        }
    }
    const auto emitWaterfallAnchors = [&] {
        std::ranges::sort(waterfallAnchors,
                          [](const WaterfallAnchor& first, const WaterfallAnchor& second) {
                              return first.localZ != second.localZ ? first.localZ < second.localZ
                                                                   : first.localX < second.localX;
                          });
        for (const WaterfallAnchor& anchor : waterfallAnchors) {
            pushWaterfallPrism(*mesh, static_cast<float>(anchor.localX),
                               static_cast<float>(anchor.localZ), anchor.sample);
            mesh->complexity = 1.0F;
        }
    };
    // Step-32 water pages below already cover every topology-marked parent at
    // the native four-block lattice. Reusing that page for falls avoids one
    // separate eight-by-eight authority request for every ordinary channel
    // parent. Finer tiers do not build a full coverage page, so they retain
    // the earlier bounded scan.
    if (key.step != FarTerrainStep::THIRTY_TWO) emitWaterfallAnchors();

    if (exactNearWater) {
        // The first three fallback tiers meet exact cubic terrain while nearby
        // sections stream. Probe a lattice whose covering radius is below the
        // minimum routed-channel half width, then request block authority only
        // for cells containing a shoreline, channel, or level transition.
        // Step 1 already owns every integer column, so its sentinel lattice is
        // the block lattice itself.
        const int sentinelSpacing = step == 1 ? 1 : 2;
        const int sentinelOffset = step == 2 ? 1 : 0;
        const int sentinelEdge = step == 2 ? cellEdge : FAR_TERRAIN_TILE_EDGE / sentinelSpacing + 1;
        std::vector<FarTerrainGeometrySample> sentinels(
            static_cast<size_t>(sentinelEdge * sentinelEdge));
        if (source.geometryGrid) {
            source.geometryGrid(mesh->originX + sentinelOffset, mesh->originZ + sentinelOffset,
                                sentinelSpacing, sentinelSpacing, sentinelEdge, sentinelEdge,
                                canonicalWaterFootprint, sentinels);
        } else {
            for (int z = 0; z < sentinelEdge; ++z) {
                for (int x = 0; x < sentinelEdge; ++x) {
                    sentinels[static_cast<size_t>(z * sentinelEdge + x)] =
                        source
                            .sample(mesh->originX + sentinelOffset + x * sentinelSpacing,
                                    mesh->originZ + sentinelOffset + z * sentinelSpacing,
                                    canonicalWaterFootprint)
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
            if (sentinelX < 0 || sentinelZ < 0 || sentinelX % sentinelSpacing != 0 ||
                sentinelZ % sentinelSpacing != 0) {
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
                    const int sentinelCellStride = step / sentinelSpacing;
                    const int sentinelSamplesPerAxis = sentinelCellStride + 1;
                    for (int sampleZ = 0; sampleZ < sentinelSamplesPerAxis; ++sampleZ) {
                        for (int sampleX = 0; sampleX < sentinelSamplesPerAxis; ++sampleX) {
                            const FarTerrainGeometrySample& sample =
                                sentinelAt(cellX * sentinelCellStride + sampleX,
                                           cellZ * sentinelCellStride + sampleZ);
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
                                }) &&
                    !(step == 1 && cellAt(cellX, cellZ).waterTopologyPossible);
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
                    canonicalWaterFootprint,
                    std::span<FarTerrainGeometrySample>(exactGeometry).subspan(first, count));
            }
        } else {
            for (size_t index = 0; index < exactPositions.size(); ++index) {
                exactGeometry[index] = source
                                           .sample(exactPositions[index].x, exactPositions[index].z,
                                                   canonicalWaterFootprint)
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
        // the shared two-block water lattice. Block-resolution recovery retains
        // half-open voxel ownership, while wider rectangles clip unrelated
        // authority instead of triangulating a false ramp.
        auto emitAuthorityResolvedWater = [&](int rectangleX0, int rectangleZ0, int rectangleX1,
                                              int rectangleZ1, int resolution,
                                              bool preserveVoxelColumns = false) {
            for (int fineZ0 = rectangleZ0; fineZ0 < rectangleZ1; fineZ0 += resolution) {
                const int fineZ1 = std::min(fineZ0 + resolution, rectangleZ1);
                for (int fineX0 = rectangleX0; fineX0 < rectangleX1; fineX0 += resolution) {
                    const int fineX1 = std::min(fineX0 + resolution, rectangleX1);
                    if (resolution == 1) {
                        if (preserveVoxelColumns) {
                            // A topology-certified block owns the complete
                            // half-open voxel column beginning at this coordinate.
                            // Clipping that final unit cell can sever a one-block
                            // route hidden inside a coarse tile-face parent.
                            const FarTerrainGeometrySample sample = canonicalWaterGeometry(
                                mesh->originX + fineX0, mesh->originZ + fineZ0);
                            if (!sampleIsWet(sample)) continue;
                            const FarWaterAuthority authority = waterAuthority(sample, true);
                            pushWaterTop(*mesh, static_cast<float>(fineX0),
                                         static_cast<float>(fineZ0), static_cast<float>(fineX1),
                                         static_cast<float>(fineZ1), authority.height);
                            continue;
                        }
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
        // The step-32 source grid already contains globally aligned standing
        // water samples. When every sample has the same finished standing
        // authority and the conservative cell bounds exclude an interior
        // contour, volcano, and fall, that grid proves the complete half-open
        // tile is one water surface. Do not rebuild its native four-block
        // raster merely to emit the same top as 64 adjacent quads.
        std::optional<FarWaterAuthority> step32UniformStandingAuthority;
        if (key.step == FarTerrainStep::THIRTY_TWO && source.cellBoundsGrid) {
            const auto sameStandingAuthority = [](const FarWaterAuthority& first,
                                                  const FarWaterAuthority& second) {
                return first.kind == second.kind && first.bodyId == second.bodyId &&
                       first.transitionOwnerId == second.transitionOwnerId &&
                       first.transitionOwnerKind == second.transitionOwnerKind &&
                       first.generatedFluidLevel == second.generatedFluidLevel &&
                       first.height == second.height && first.delta == second.delta;
            };
            bool uniform = true;
            for (const FarSurfaceSample& sample : samples) {
                if (!sampleIsWet(sample.geometry) || sample.geometry.waterfall ||
                    sample.geometry.waterfallAnchor) {
                    uniform = false;
                    break;
                }
                const FarWaterAuthority authority = waterAuthority(sample.geometry, true);
                if (authority.kind != FarWaterKind::OCEAN && authority.kind != FarWaterKind::LAKE) {
                    uniform = false;
                    break;
                }
                if (!step32UniformStandingAuthority) {
                    step32UniformStandingAuthority = authority;
                } else if (!sameStandingAuthority(*step32UniformStandingAuthority, authority)) {
                    uniform = false;
                    break;
                }
            }
            if (uniform) {
                for (int cellZ = 0; cellZ < cellEdge && uniform; ++cellZ) {
                    for (int cellX = 0; cellX < cellEdge; ++cellX) {
                        const FarTerrainCellBounds& bounds = authoritativeBoundsAt(cellX, cellZ);
                        if (bounds.waterTopologyPossible || bounds.volcanicWaterPossible ||
                            bounds.waterfallPossible) {
                            uniform = false;
                            break;
                        }
                    }
                }
            }
            if (!uniform || std::ranges::any_of(refineWaterBoundary, std::identity{})) {
                step32UniformStandingAuthority.reset();
            }
        }
        const bool step32NeedsWaterPage = [&] {
            if (key.step != FarTerrainStep::THIRTY_TWO || !source.cellBoundsGrid) return true;

            if (step32UniformStandingAuthority) return false;

            // The coarse representatives prove the visible samples are dry;
            // the conservative half-open cell bounds prove that no hidden
            // channel, shoreline, volcanic overlay, or waterfall candidate
            // can require canonical recovery. In that case the native page
            // below would classify every parent as dry and emit nothing.
            for (const FarSurfaceSample& sample : samples) {
                if (sampleIsWet(sample.geometry)) return true;
            }
            for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
                for (int cellX = 0; cellX < cellEdge; ++cellX) {
                    const FarTerrainCellBounds& bounds = authoritativeBoundsAt(cellX, cellZ);
                    if (bounds.waterTopologyPossible || bounds.volcanicWaterPossible ||
                        bounds.waterfallPossible) {
                        return true;
                    }
                }
            }
            return false;
        }();
        if (step32UniformStandingAuthority) {
            pushWaterTop(*mesh, 0.0F, 0.0F, static_cast<float>(FAR_TERRAIN_TILE_EDGE),
                         static_cast<float>(FAR_TERRAIN_TILE_EDGE),
                         step32UniformStandingAuthority->height);
            std::fill(merged.begin(), merged.end(), 1);
            mesh->complexity = 1.0F;
        } else if (key.step == FarTerrainStep::THIRTY_TWO && step32NeedsWaterPage) {
            // Broad, uniform water remains cheap, but a dry phase-zero sample
            // is not evidence that a 32-block parent is dry. Cell-bounds
            // authority marks parents that an analytical channel or shoreline
            // can cross. Mixed and topology-marked parents are then resolved
            // from their complete half-open block footprint, preserving narrow
            // crossings independently of raster phase and tile build order.
            // Native hydrology is routed on a four-block lattice. Distant
            // topology can therefore stay on that canonical lattice instead
            // of expanding every shallow coast or channel parent into 1,024
            // block queries. Volcanic cells retain their exact exception
            // path below because their geometry is not yet part of the
            // native pre-hydrology authority.
            constexpr int WATER_STEP = worldgen::NATIVE_HYDROLOGY_RASTER_SPACING;
            constexpr int WATER_CELLS = FAR_TERRAIN_TILE_EDGE / WATER_STEP;
            constexpr int WATER_PAGE_EDGE = WATER_CELLS + 2;
            constexpr int WATER_HALO = 1;
            constexpr int WATER_CELLS_PER_PARENT = 32 / WATER_STEP;
            const int64_t waterOriginX = checkedCoordinateOffset(mesh->originX, -WATER_STEP);
            const int64_t waterOriginZ = checkedCoordinateOffset(mesh->originZ, -WATER_STEP);
            std::vector<FarTerrainGeometrySample> waterPage(
                static_cast<size_t>(WATER_PAGE_EDGE * WATER_PAGE_EDGE));
            std::vector<uint8_t> waterPageLoaded(waterPage.size(), 0);
            const auto pageIndex = [](int pageX, int pageZ) {
                return static_cast<size_t>(pageZ * WATER_PAGE_EDGE + pageX);
            };
            const auto waterPageAt = [&](int pageX, int pageZ) -> const FarTerrainGeometrySample& {
                const size_t index = pageIndex(pageX, pageZ);
                if (waterPageLoaded[index] == 0)
                    throw std::logic_error("step-32 water raster read an unprepared sample");
                return waterPage[index];
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
                if (!finalGeometry && source.canonicalWaterPoints) {
                    source.canonicalWaterPoints(positions, canonicalWaterFootprint, output);
                    return;
                }
                if (!finalGeometry && source.waterAuthorityPoints) {
                    source.waterAuthorityPoints(positions, canonicalWaterFootprint, output);
                    return;
                }
                if (!finalGeometry && source.geometryPoints) {
                    source.geometryPoints(positions, canonicalWaterFootprint, output);
                    return;
                }
                for (size_t index = 0; index < positions.size(); ++index) {
                    output[index] =
                        source
                            .sample(positions[index].x, positions[index].z, canonicalWaterFootprint)
                            .geometry;
                }
            };

            const auto sameCoverageAuthority = [](const FarWaterAuthority& first,
                                                  const FarWaterAuthority& second) {
                return first.kind == second.kind && first.bodyId == second.bodyId &&
                       first.transitionOwnerId == second.transitionOwnerId &&
                       first.transitionOwnerKind == second.transitionOwnerKind &&
                       first.generatedFluidLevel == second.generatedFluidLevel &&
                       first.height == second.height && first.delta == second.delta;
            };
            const auto parentBoundsFor = [&](int localX,
                                             int localZ) -> const FarTerrainCellBounds* {
                if (authoritativeBounds.empty()) return nullptr;
                const int parentX = static_cast<int>(world_coord::floorDiv(localX, step));
                const int parentZ = static_cast<int>(world_coord::floorDiv(localZ, step));
                return &authoritativeBoundsAt(parentX, parentZ);
            };

            // The compact native topology proves which 32-block parents are
            // uniformly dry or uniformly owned standing water. Query the
            // four-block raster only for an ambiguous parent. Include its
            // positive edge and the one negative-side sample needed by
            // half-open west and north riser ownership. Shared shoreline
            // refinement can conservatively select a little more than half of
            // the page. Keep
            // that selection sparse because sampling the remaining certified
            // area dominated cold horizon construction.
            bool sparseWaterPage =
                source.sparseStep32Water && source.canonicalWaterPoints && source.cellBoundsGrid;
            std::vector<uint8_t> sparseParentNeedsRaster(static_cast<size_t>(cellEdge * cellEdge),
                                                         sparseWaterPage ? 0 : 1);
            if (sparseWaterPage) {
                for (int parentZ = 0; parentZ < cellEdge; ++parentZ) {
                    for (int parentX = 0; parentX < cellEdge; ++parentX) {
                        const size_t parentIndex =
                            static_cast<size_t>(parentZ * cellEdge + parentX);
                        const FarTerrainCellBounds* bounds =
                            parentBoundsFor(parentX * step, parentZ * step);
                        bool needsRaster = bounds == nullptr || bounds->waterTopologyPossible ||
                                           bounds->volcanicWaterPossible ||
                                           bounds->waterfallPossible;
                        needsRaster = needsRaster ||
                                      step32SharedBoundaryIntersects(parentX * step, parentZ * step,
                                                                     (parentX + 1) * step,
                                                                     (parentZ + 1) * step);
                        if (!needsRaster) {
                            const std::array<const FarTerrainGeometrySample*, 4> corners{{
                                &sampleAt(parentX, parentZ),
                                &sampleAt(parentX + 1, parentZ),
                                &sampleAt(parentX + 1, parentZ + 1),
                                &sampleAt(parentX, parentZ + 1),
                            }};
                            bool allDry = true;
                            bool allWet = true;
                            bool uniformWet = true;
                            std::optional<FarWaterAuthority> reference;
                            for (const FarTerrainGeometrySample* sample : corners) {
                                const bool wet = sampleIsWet(*sample);
                                allDry = allDry && !wet;
                                allWet = allWet && wet;
                                if (!wet) {
                                    uniformWet = false;
                                    continue;
                                }
                                const FarWaterAuthority authority = waterAuthority(*sample, true);
                                if (!reference) {
                                    reference = authority;
                                } else {
                                    uniformWet =
                                        uniformWet && sameCoverageAuthority(*reference, authority);
                                }
                            }
                            needsRaster = !allDry && !(allWet && uniformWet);
                        }
                        sparseParentNeedsRaster[parentIndex] = needsRaster ? 1 : 0;
                        if (!needsRaster) continue;

                        const int pageOriginX = WATER_HALO + parentX * WATER_CELLS_PER_PARENT;
                        const int pageOriginZ = WATER_HALO + parentZ * WATER_CELLS_PER_PARENT;
                        for (int pageZ = std::max(0, pageOriginZ - 1);
                             pageZ <=
                             std::min(WATER_PAGE_EDGE - 1, pageOriginZ + WATER_CELLS_PER_PARENT);
                             ++pageZ) {
                            for (int pageX = std::max(0, pageOriginX - 1);
                                 pageX <= std::min(WATER_PAGE_EDGE - 1,
                                                   pageOriginX + WATER_CELLS_PER_PARENT);
                                 ++pageX) {
                                waterPageLoaded[pageIndex(pageX, pageZ)] = 1;
                            }
                        }
                    }
                }
                const size_t sparseSampleCount = static_cast<size_t>(
                    std::ranges::count(waterPageLoaded, static_cast<uint8_t>(1)));
                if (sparseSampleCount == waterPage.size()) {
                    sparseWaterPage = false;
                    std::ranges::fill(sparseParentNeedsRaster, uint8_t{1});
                    std::ranges::fill(waterPageLoaded, uint8_t{0});
                }
            }

            if (sparseWaterPage && source.canonicalWaterGrid) {
                // Decompose the selected native mask into deterministic
                // rectangles. Production hydrology can evaluate each rectangle
                // as one grid instead of sorting the same points into dozens of
                // one-row calls, while no certified sample is reconstructed.
                std::vector<uint8_t> pending = waterPageLoaded;
                for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                    for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                        if (pending[pageIndex(pageX, pageZ)] == 0) continue;

                        int width = 1;
                        while (pageX + width < WATER_PAGE_EDGE &&
                               pending[pageIndex(pageX + width, pageZ)] != 0) {
                            ++width;
                        }
                        int height = 1;
                        while (pageZ + height < WATER_PAGE_EDGE) {
                            bool completeRow = true;
                            for (int offsetX = 0; offsetX < width; ++offsetX) {
                                completeRow =
                                    completeRow &&
                                    pending[pageIndex(pageX + offsetX, pageZ + height)] != 0;
                            }
                            if (!completeRow) break;
                            ++height;
                        }

                        std::vector<FarTerrainGeometrySample> rectangle(
                            static_cast<size_t>(width * height));
                        const ColumnPos origin = pagePosition(pageX, pageZ, 0);
                        source.canonicalWaterGrid(origin.x, origin.z, WATER_STEP, WATER_STEP, width,
                                                  height, canonicalWaterFootprint, rectangle);
                        ++mesh->step32WaterGridCallCount;
                        mesh->step32WaterGridSampleCount += static_cast<uint32_t>(rectangle.size());
                        for (int offsetZ = 0; offsetZ < height; ++offsetZ) {
                            for (int offsetX = 0; offsetX < width; ++offsetX) {
                                const size_t destination =
                                    pageIndex(pageX + offsetX, pageZ + offsetZ);
                                waterPage[destination] =
                                    rectangle[static_cast<size_t>(offsetZ * width + offsetX)];
                                pending[destination] = 0;
                            }
                        }
                    }
                }
            } else if (sparseWaterPage) {
                std::vector<ColumnPos> positions;
                std::vector<size_t> indexes;
                positions.reserve(waterPage.size());
                indexes.reserve(waterPage.size());
                for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                    for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                        const size_t index = pageIndex(pageX, pageZ);
                        if (waterPageLoaded[index] == 0) continue;
                        positions.push_back(pagePosition(pageX, pageZ, 0));
                        indexes.push_back(index);
                    }
                }
                std::vector<FarTerrainGeometrySample> geometry(positions.size());
                if (!positions.empty()) {
                    source.canonicalWaterPoints(positions, canonicalWaterFootprint, geometry);
                    mesh->step32WaterPointSampleCount += static_cast<uint32_t>(positions.size());
                }
                for (size_t index = 0; index < indexes.size(); ++index)
                    waterPage[indexes[index]] = geometry[index];
            } else {
                std::ranges::fill(waterPageLoaded, uint8_t{1});
                if (source.canonicalWaterGrid) {
                    source.canonicalWaterGrid(waterOriginX, waterOriginZ, WATER_STEP, WATER_STEP,
                                              WATER_PAGE_EDGE, WATER_PAGE_EDGE,
                                              canonicalWaterFootprint, waterPage);
                    ++mesh->step32WaterGridCallCount;
                    ++mesh->step32WaterDenseGridCallCount;
                    mesh->step32WaterGridSampleCount += static_cast<uint32_t>(waterPage.size());
                } else if (source.canonicalWaterPoints) {
                    std::vector<ColumnPos> positions;
                    positions.reserve(waterPage.size());
                    for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                        for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX)
                            positions.push_back(pagePosition(pageX, pageZ, 0));
                    }
                    source.canonicalWaterPoints(positions, canonicalWaterFootprint, waterPage);
                    mesh->step32WaterPointSampleCount += static_cast<uint32_t>(positions.size());
                } else if (source.waterAuthorityGrid) {
                    source.waterAuthorityGrid(waterOriginX, waterOriginZ, WATER_STEP, WATER_STEP,
                                              WATER_PAGE_EDGE, WATER_PAGE_EDGE,
                                              canonicalWaterFootprint, waterPage);
                } else if (source.waterAuthorityPoints) {
                    std::vector<ColumnPos> positions;
                    positions.reserve(waterPage.size());
                    for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                        for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX)
                            positions.push_back(pagePosition(pageX, pageZ, 0));
                    }
                    source.waterAuthorityPoints(positions, canonicalWaterFootprint, waterPage);
                } else {
                    for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                        for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                            const ColumnPos position = pagePosition(pageX, pageZ, 0);
                            waterPage[pageIndex(pageX, pageZ)] =
                                source.sample(position.x, position.z, canonicalWaterFootprint)
                                    .geometry;
                        }
                    }
                }
            }
            for (size_t index = 0; index < waterPage.size(); ++index) {
                if (waterPageLoaded[index] != 0) waterTopology.observe(waterPage[index]);
            }
            for (int pageZ = 0; pageZ < WATER_PAGE_EDGE; ++pageZ) {
                for (int pageX = 0; pageX < WATER_PAGE_EDGE; ++pageX) {
                    const size_t index = pageIndex(pageX, pageZ);
                    if (waterPageLoaded[index] == 0) continue;
                    if (pageX + 1 < WATER_PAGE_EDGE &&
                        waterPageLoaded[pageIndex(pageX + 1, pageZ)] != 0) {
                        waterTopology.observeConnection(waterPageAt(pageX, pageZ),
                                                        waterPageAt(pageX + 1, pageZ));
                    }
                    if (pageZ + 1 < WATER_PAGE_EDGE &&
                        waterPageLoaded[pageIndex(pageX, pageZ + 1)] != 0) {
                        waterTopology.observeConnection(waterPageAt(pageX, pageZ),
                                                        waterPageAt(pageX, pageZ + 1));
                    }
                }
            }

            // Any step-32 cell carrying waterfallPossible reaches this
            // canonical page through step32NeedsWaterPage. Scan the owned
            // native lattice once after its full water authority is ready,
            // rather than rebuilding an eight-by-eight page per channel
            // parent before this coverage pass. The half-open inner page
            // range gives the anchor to exactly one far tile.
            for (int pageZ = WATER_HALO; pageZ <= WATER_CELLS; ++pageZ) {
                for (int pageX = WATER_HALO; pageX <= WATER_CELLS; ++pageX) {
                    if (waterPageLoaded[pageIndex(pageX, pageZ)] == 0) continue;
                    const FarTerrainGeometrySample& sample = waterPageAt(pageX, pageZ);
                    retainWaterfallAnchor((pageX - WATER_HALO) * WATER_STEP,
                                          (pageZ - WATER_HALO) * WATER_STEP, sample);
                }
            }

            enum class CoverageWaterMode : uint8_t { DRY, UNIFORM, ADAPTIVE };
            struct AdaptiveCoveragePage {
                bool finalGeometry = false;
                bool nativeRaster = false;
                size_t offset = 0;
                FarWaterAuthority uniformAuthority{};
            };
            std::vector<CoverageWaterMode> parentModes(static_cast<size_t>(cellEdge * cellEdge),
                                                       CoverageWaterMode::DRY);
            std::vector<AdaptiveCoveragePage> adaptivePages(
                static_cast<size_t>(cellEdge * cellEdge));
            std::vector<ColumnPos> authorityPositions;
            std::vector<ColumnPos> finalPositions;
            const bool hasNativeTopologyRaster = static_cast<bool>(source.canonicalWaterGrid);
            for (int parentZ = 0; parentZ < cellEdge; ++parentZ) {
                for (int parentX = 0; parentX < cellEdge; ++parentX) {
                    const size_t parentIndex = static_cast<size_t>(parentZ * cellEdge + parentX);
                    const FarTerrainCellBounds* bounds =
                        parentBoundsFor(parentX * step, parentZ * step);
                    const bool topologyPossible = bounds && bounds->waterTopologyPossible;
                    const bool finalGeometry = bounds && bounds->volcanicWaterPossible;
                    const bool sharedBoundaryParent = step32SharedBoundaryIntersects(
                        parentX * step, parentZ * step, (parentX + 1) * step, (parentZ + 1) * step);
                    bool allDry = true;
                    bool allWet = true;
                    bool uniformWet = true;
                    std::optional<FarWaterAuthority> reference;
                    const bool nativeRasterPrepared = sparseParentNeedsRaster[parentIndex] != 0;
                    const int sampleCount = nativeRasterPrepared ? WATER_CELLS_PER_PARENT + 1 : 2;
                    for (int sampleZ = 0; sampleZ < sampleCount; ++sampleZ) {
                        for (int sampleX = 0; sampleX < sampleCount; ++sampleX) {
                            const FarTerrainGeometrySample& sample =
                                nativeRasterPrepared
                                    ? waterPageAt(
                                          WATER_HALO + parentX * WATER_CELLS_PER_PARENT + sampleX,
                                          WATER_HALO + parentZ * WATER_CELLS_PER_PARENT + sampleZ)
                                    : sampleAt(parentX + sampleX, parentZ + sampleZ);
                            const bool wet = sampleIsWet(sample);
                            allDry = allDry && !wet;
                            allWet = allWet && wet;
                            if (!wet) {
                                uniformWet = false;
                                continue;
                            }
                            const FarWaterAuthority authority = waterAuthority(sample, true);
                            if (!reference) {
                                reference = authority;
                            } else {
                                uniformWet =
                                    uniformWet && sameCoverageAuthority(*reference, authority);
                            }
                        }
                    }
                    // A cell-bounds topology signal also describes an
                    // interior dry barrier or island. Do not collapse an
                    // apparently uniform native raster to one water
                    // quad before the canonical half-open block footprint
                    // has had a chance to prove it is uniformly wet.
                    if (allWet && uniformWet && !topologyPossible && !finalGeometry &&
                        !sharedBoundaryParent) {
                        parentModes[parentIndex] = CoverageWaterMode::UNIFORM;
                        adaptivePages[parentIndex].uniformAuthority = *reference;
                        continue;
                    }
                    if (allDry && !topologyPossible && !finalGeometry) continue;

                    parentModes[parentIndex] = CoverageWaterMode::ADAPTIVE;
                    AdaptiveCoveragePage& page = adaptivePages[parentIndex];
                    page.finalGeometry = finalGeometry;
                    page.nativeRaster = hasNativeTopologyRaster && !finalGeometry;
                    if (page.nativeRaster) continue;
                    std::vector<ColumnPos>& positions =
                        finalGeometry ? finalPositions : authorityPositions;
                    page.offset = positions.size();
                    for (int offsetZ = 0; offsetZ < step; ++offsetZ) {
                        for (int offsetX = 0; offsetX < step; ++offsetX) {
                            positions.push_back({mesh->originX + parentX * step + offsetX,
                                                 mesh->originZ + parentZ * step + offsetZ});
                        }
                    }
                }
            }
            std::vector<FarTerrainGeometrySample> authorityGeometry(authorityPositions.size());
            std::vector<FarTerrainGeometrySample> finalGeometry(finalPositions.size());
            const auto sampleCoverageBatches = [&](std::span<const ColumnPos> positions,
                                                   std::span<FarTerrainGeometrySample> output,
                                                   bool useFinalGeometry) {
                for (size_t first = 0; first < positions.size();
                     first += NEAR_WATER_MAX_POINT_BATCH) {
                    const size_t count =
                        std::min(NEAR_WATER_MAX_POINT_BATCH, positions.size() - first);
                    samplePointBatch(positions.subspan(first, count), output.subspan(first, count),
                                     useFinalGeometry);
                }
            };
            sampleCoverageBatches(authorityPositions, authorityGeometry, false);
            sampleCoverageBatches(finalPositions, finalGeometry, true);

            bool emittedWater = false;
            for (int parentZ = 0; parentZ < cellEdge; ++parentZ) {
                for (int parentX = 0; parentX < cellEdge; ++parentX) {
                    const bool replacedBySharedBoundary =
                        key.step != FarTerrainStep::THIRTY_TWO &&
                        ((parentX == 0 && refineWaterBoundary[WEST]) ||
                         (parentX == cellEdge - 1 && refineWaterBoundary[EAST]) ||
                         (parentZ == 0 && refineWaterBoundary[NORTH]) ||
                         (parentZ == cellEdge - 1 && refineWaterBoundary[SOUTH]));
                    if (replacedBySharedBoundary) continue;
                    const size_t parentIndex = static_cast<size_t>(parentZ * cellEdge + parentX);
                    const CoverageWaterMode mode = parentModes[parentIndex];
                    if (mode == CoverageWaterMode::DRY) continue;
                    const int localOriginX = parentX * step;
                    const int localOriginZ = parentZ * step;
                    if (mode == CoverageWaterMode::UNIFORM) {
                        const FarWaterAuthority authority =
                            adaptivePages[parentIndex].uniformAuthority;
                        pushWaterTop(*mesh, static_cast<float>(localOriginX),
                                     static_cast<float>(localOriginZ),
                                     static_cast<float>(localOriginX + step),
                                     static_cast<float>(localOriginZ + step), authority.height);
                        emittedWater = true;
                        continue;
                    }

                    const AdaptiveCoveragePage& page = adaptivePages[parentIndex];
                    if (page.nativeRaster) {
                        const int pageOriginX = WATER_HALO + parentX * WATER_CELLS_PER_PARENT;
                        const int pageOriginZ = WATER_HALO + parentZ * WATER_CELLS_PER_PARENT;
                        for (int offsetZ = 0; offsetZ < step; offsetZ += WATER_STEP) {
                            int offsetX = 0;
                            while (offsetX < step) {
                                const int waterX0 = localOriginX + offsetX;
                                const int waterZ0 = localOriginZ + offsetZ;
                                if (step32SharedBoundaryOwns(waterX0, waterZ0, waterX0 + WATER_STEP,
                                                             waterZ0 + WATER_STEP)) {
                                    offsetX += WATER_STEP;
                                    continue;
                                }
                                const FarTerrainGeometrySample& sample =
                                    waterPageAt(pageOriginX + offsetX / WATER_STEP,
                                                pageOriginZ + offsetZ / WATER_STEP);
                                if (!sampleIsWet(sample)) {
                                    offsetX += WATER_STEP;
                                    continue;
                                }
                                const FarWaterAuthority authority = waterAuthority(sample, true);
                                int run = 1;
                                while (offsetX + run * WATER_STEP < step) {
                                    const int nextX0 = localOriginX + offsetX + run * WATER_STEP;
                                    if (step32SharedBoundaryOwns(nextX0, waterZ0,
                                                                 nextX0 + WATER_STEP,
                                                                 waterZ0 + WATER_STEP)) {
                                        break;
                                    }
                                    const FarTerrainGeometrySample& next =
                                        waterPageAt(pageOriginX + offsetX / WATER_STEP + run,
                                                    pageOriginZ + offsetZ / WATER_STEP);
                                    if (!sampleIsWet(next) ||
                                        !sameCoverageAuthority(authority,
                                                               waterAuthority(next, true))) {
                                        break;
                                    }
                                    ++run;
                                }
                                pushWaterTop(
                                    *mesh, static_cast<float>(localOriginX + offsetX),
                                    static_cast<float>(localOriginZ + offsetZ),
                                    static_cast<float>(localOriginX + offsetX + run * WATER_STEP),
                                    static_cast<float>(localOriginZ + offsetZ + WATER_STEP),
                                    authority.height);
                                emittedWater = true;
                                offsetX += run * WATER_STEP;
                            }
                        }
                        continue;
                    }
                    const std::vector<FarTerrainGeometrySample>& geometry =
                        page.finalGeometry ? finalGeometry : authorityGeometry;
                    for (int offsetZ = 0; offsetZ < step; ++offsetZ) {
                        int offsetX = 0;
                        while (offsetX < step) {
                            const int waterX0 = localOriginX + offsetX;
                            const int waterZ0 = localOriginZ + offsetZ;
                            if (step32SharedBoundaryOwns(waterX0, waterZ0, waterX0 + 1,
                                                         waterZ0 + 1)) {
                                ++offsetX;
                                continue;
                            }
                            const size_t sampleIndex =
                                page.offset + static_cast<size_t>(offsetZ * step + offsetX);
                            const FarTerrainGeometrySample& sample = geometry[sampleIndex];
                            if (!sampleIsWet(sample)) {
                                ++offsetX;
                                continue;
                            }
                            const FarWaterAuthority authority = waterAuthority(sample, true);
                            int run = 1;
                            while (offsetX + run < step) {
                                const int nextX0 = localOriginX + offsetX + run;
                                if (step32SharedBoundaryOwns(nextX0, waterZ0, nextX0 + 1,
                                                             waterZ0 + 1)) {
                                    break;
                                }
                                const FarTerrainGeometrySample& next =
                                    geometry[page.offset +
                                             static_cast<size_t>(offsetZ * step + offsetX + run)];
                                if (!sampleIsWet(next) ||
                                    !sameCoverageAuthority(authority, waterAuthority(next, true))) {
                                    break;
                                }
                                ++run;
                            }
                            pushWaterTop(*mesh, static_cast<float>(localOriginX + offsetX),
                                         static_cast<float>(localOriginZ + offsetZ),
                                         static_cast<float>(localOriginX + offsetX + run),
                                         static_cast<float>(localOriginZ + offsetZ + 1),
                                         authority.height);
                            emittedWater = true;
                            offsetX += run;
                        }
                    }
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
            for (int pageZ = WATER_HALO; pageZ <= WATER_CELLS; ++pageZ) {
                for (int pageX = WATER_HALO; pageX <= WATER_CELLS; ++pageX) {
                    const size_t currentIndex = pageIndex(pageX, pageZ);
                    if (waterPageLoaded[currentIndex] == 0) continue;
                    const float localX = static_cast<float>((pageX - WATER_HALO) * WATER_STEP);
                    const float localZ = static_cast<float>((pageZ - WATER_HALO) * WATER_STEP);
                    const FarTerrainGeometrySample& current = waterPageAt(pageX, pageZ);
                    if (waterPageLoaded[pageIndex(pageX - 1, pageZ)] != 0) {
                        emitOwnedRiser(current, waterPageAt(pageX - 1, pageZ), FaceNormal::MINUS_X,
                                       FaceNormal::PLUS_X, localX, localZ, localX,
                                       localZ + WATER_STEP);
                    }
                    if (waterPageLoaded[pageIndex(pageX, pageZ - 1)] != 0) {
                        emitOwnedRiser(current, waterPageAt(pageX, pageZ - 1), FaceNormal::MINUS_Z,
                                       FaceNormal::PLUS_Z, localX, localZ, localX + WATER_STEP,
                                       localZ);
                    }
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
                                               cell.waterTopologyPossible ? 1 : 2);
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
            if (sharedRectangleTopologyPossible(x0, z0, x1, z1)) {
                emitAuthorityResolvedWater(x0, z0, x1, z1, 1, true);
                return;
            }
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
            const int begin =
                horizontal && refineWaterBoundary[WEST] ? sharedWaterBoundaryDepth : 0;
            const int end = horizontal && refineWaterBoundary[EAST]
                                ? FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth
                                : FAR_TERRAIN_TILE_EDGE;
            for (int coordinate = begin; coordinate < end; coordinate += SHARED_WATER_EDGE_STEP) {
                if (edge == WEST) {
                    emitSharedWaterCell(0, coordinate, sharedWaterBoundaryDepth,
                                        coordinate + SHARED_WATER_EDGE_STEP);
                } else if (edge == EAST) {
                    emitSharedWaterCell(FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth,
                                        coordinate, FAR_TERRAIN_TILE_EDGE,
                                        coordinate + SHARED_WATER_EDGE_STEP);
                } else if (edge == NORTH) {
                    emitSharedWaterCell(coordinate, 0, coordinate + SHARED_WATER_EDGE_STEP,
                                        sharedWaterBoundaryDepth);
                } else {
                    emitSharedWaterCell(coordinate,
                                        FAR_TERRAIN_TILE_EDGE - sharedWaterBoundaryDepth,
                                        coordinate + SHARED_WATER_EDGE_STEP, FAR_TERRAIN_TILE_EDGE);
                }
            }
        };
        emitSharedBoundary(WEST);
        emitSharedBoundary(EAST);
        emitSharedBoundary(NORTH);
        emitSharedBoundary(SOUTH);
    }

    if (key.step == FarTerrainStep::THIRTY_TWO) emitWaterfallAnchors();

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
    mesh->waterTopology = waterTopology.signature();
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
        result.geometry.terrainHeight = worldgen::geometryTerrainHeight(exact);
        result.footprintMinimumTerrainHeight = result.geometry.terrainHeight;
        result.footprintMaximumTerrainHeight = result.geometry.terrainHeight;
        return result;
    };
    return source;
}

FarTerrainSource
FarTerrainMesher::generatorGeometrySource(std::shared_ptr<ChunkGenerator> generator) {
    if (!generator) throw std::invalid_argument("far terrain generator is empty");
    FarTerrainSource source;
    source.sparseStep32Water = generator->usesLearnedAuthority();
    source.planFreeCoarseAuthority = generator->usesLearnedAuthority();
    if (generator->usesLearnedAuthority()) {
        source.finalBaseAuthorityDependencies = [](FarTerrainKey key) {
            return farTerrainFinalBaseAuthorityDependencies(key);
        };
    }
    source.sample = [generator](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        // Exact cubes are the only one-block terrain tier. Far step-2 and
        // step-4 relief remains a bounded macro sample; their water mask is
        // gathered separately in one block-authority batch without building
        // complete ColumnPlans for the tile.
        const bool exactHandoff = surfaceFootprintWidth(footprint) == 1;
        worldgen::SurfaceSample surface;
        if (exactHandoff) {
            surface = generator->sampleExactSurface(x, z);
        } else {
            std::array<worldgen::SurfaceSample, 1> coarse{};
            generator->sampleFarSurfaceGrid(x, z, worldgen::surfaceFootprintWidth(footprint), 1,
                                            footprint, coarse);
            surface = coarse.front();
        }
        FarSurfaceSample result = farSampleFromSurface(surface);
        if (exactHandoff) {
            // The exact density top owns the handoff geometry, but a direct
            // coordinate sample owns water topology and level. Column-plan
            // interpolation must never change a lake between LODs.
            const worldgen::SurfaceSample canonical =
                generator->sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1);
            worldgen::SurfaceSample geometry = canonical;
            const double emittedTop = worldgen::geometryTerrainHeight(surface);
            geometry.terrainHeight = emittedTop;
            geometry.emittedTerrainHeight = static_cast<float>(emittedTop);
            geometry.hydrology.surfaceElevation = emittedTop;
            result = farSampleFromSurface(geometry);
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
            std::array<worldgen::SurfaceSample, 1> canonical{};
            generator->sampleCoarseNativeHydrologyGeometryGrid(x, z, 2, 2, 1, 1, canonical);
            includeCanonicalWaterFloor(result, canonical.front());
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
        const int expandedEdge = sampleEdge + GENERATOR_FAR_GRID_APRON_CELLS * 2;
        const int64_t expandedOriginX =
            checkedCoordinateOffset(originX, -spacing * GENERATOR_FAR_GRID_APRON_CELLS);
        const int64_t expandedOriginZ =
            checkedCoordinateOffset(originZ, -spacing * GENERATOR_FAR_GRID_APRON_CELLS);
        GeneratorCellBoundsBatch& batch = threadGeneratorCellBoundsBatch();
        // This scratch batch is not a cell-bounds cache entry. Its surface
        // query and apron can have the same metadata as a later bounds query,
        // but the bounds path requires geometry-only samples. Invalidate the
        // reusable marker before populating scratch storage so address reuse
        // and randomized test order cannot publish the wrong top authority.
        batch.prepared = false;
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
        std::vector<worldgen::HydrologySample> canonicalHydrology(batch.vertices.size());
        const bool nativeCoarseWater = generator->usesLearnedAuthority() && !exactHandoff;
        batch.canonicalVertices.resize(batch.vertices.size());
        if (nativeCoarseWater) {
            generator->sampleCoarseNativeHydrologyAuthorityGrid(expandedOriginX, expandedOriginZ,
                                                                spacing, spacing, expandedEdge,
                                                                expandedEdge, canonicalHydrology);
            for (size_t index = 0; index < canonicalHydrology.size(); ++index) {
                batch.canonicalVertices[index].hydrology = canonicalHydrology[index];
                batch.canonicalVertices[index].terrainHeight =
                    canonicalHydrology[index].surfaceElevation;
                batch.canonicalVertices[index].waterSurface =
                    canonicalHydrology[index].waterSurface;
            }
        } else {
            generator->sampleFarGeometryGrid(
                expandedOriginX, expandedOriginZ, spacing, spacing, expandedEdge, expandedEdge,
                worldgen::SurfaceFootprint::BLOCK_1, batch.canonicalVertices);
        }
        if (spacing == 32 && !nativeCoarseWater) {
            for (size_t index = 0; index < canonicalHydrology.size(); ++index)
                canonicalHydrology[index] = batch.canonicalVertices[index].hydrology;
        } else if (!nativeCoarseWater) {
            generator->sampleGeneratedWaterAuthorityGrid(expandedOriginX, expandedOriginZ, spacing,
                                                         expandedEdge, canonicalHydrology);
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
        batch.usesVolcanicWaterCandidates = spacing == 32;
        if (batch.usesVolcanicWaterCandidates) {
            batch.volcanicWaterCandidates.resize(static_cast<size_t>(batch.cellWidth) *
                                                 static_cast<size_t>(batch.cellHeight));
            generator->markVolcanicWaterCandidates(expandedOriginX, expandedOriginZ, spacing,
                                                   batch.cellWidth, batch.cellHeight,
                                                   batch.volcanicWaterCandidates);
        } else {
            batch.volcanicWaterCandidates.clear();
        }
        for (int sampleZ = 0; sampleZ < sampleEdge; ++sampleZ) {
            for (int sampleX = 0; sampleX < sampleEdge; ++sampleX) {
                const size_t outputIndex = static_cast<size_t>(sampleZ * sampleEdge + sampleX);
                const size_t batchIndex =
                    static_cast<size_t>((sampleZ + GENERATOR_FAR_GRID_APRON_CELLS) * expandedEdge +
                                        sampleX + GENERATOR_FAR_GRID_APRON_CELLS);
                const worldgen::SurfaceSample& retainedSurface = batch.vertices[batchIndex];
                const worldgen::SurfaceSample& canonicalWater = batch.canonicalVertices[batchIndex];
                const worldgen::SurfaceSample physicalMaterialSurface = retainedSurface;
                worldgen::SurfaceSample geometrySurface = retainedSurface;
                if (exactHandoff) {
                    const double emittedTop = worldgen::geometryTerrainHeight(retainedSurface);
                    geometrySurface.hydrology = canonicalWater.hydrology;
                    geometrySurface.waterSurface = canonicalWater.waterSurface;
                    geometrySurface.terrainHeight = emittedTop;
                    geometrySurface.emittedTerrainHeight = static_cast<float>(emittedTop);
                    geometrySurface.hydrology.surfaceElevation = emittedTop;
                    if (geometrySurface.hydrology.lake) {
                        geometrySurface.hydrology.lakeDepth =
                            std::max(0.0, geometrySurface.hydrology.waterSurface - emittedTop);
                    }
                    output[outputIndex] = farSampleFromSurface(geometrySurface);
                } else {
                    output[outputIndex] = boundedFarSampleFromSurface(
                        geometrySurface, footprint,
                        ChunkGenerator::emittedSurfaceDetailAmplitude(
                            geometrySurface, footprintReliefSlopeEnvelope(footprint)));
                    includeCanonicalWaterFloor(output[outputIndex], canonicalWater);
                    geometrySurface.hydrology = canonicalHydrology[batchIndex];
                    geometrySurface.waterSurface = geometrySurface.hydrology.waterSurface;
                    if (geometrySurface.hydrology.ocean || geometrySurface.hydrology.river ||
                        geometrySurface.hydrology.lake || geometrySurface.hydrology.wetland) {
                        geometrySurface.terrainHeight =
                            std::min(geometrySurface.terrainHeight,
                                     geometrySurface.hydrology.surfaceElevation);
                    }
                    geometrySurface.hydrology.surfaceElevation = geometrySurface.terrainHeight;
                }
                const int64_t worldX = originX + static_cast<int64_t>(sampleX) * spacing;
                const int64_t worldZ = originZ + static_cast<int64_t>(sampleZ) * spacing;
                output[outputIndex].materialPalette =
                    generator->farSurfaceMaterialPaletteAt(worldX, worldZ, physicalMaterialSurface);
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
        // Corner and center probes alone cannot see a four-block river or a
        // small lake lying entirely between the step-32 terrain samples.
        // Reduce the solved native page into compact 32-block topology cells
        // before deciding whether this parent needs canonical water. Uniform
        // standing water remains unmarked, so it retains the one-quad fast
        // path rather than turning every open-water horizon tile into a
        // dense raster.
        std::vector<worldgen::NativeHydrologyTopologyCell> nativeTopology;
        if (generator->usesLearnedAuthority() &&
            step == worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE &&
            originX % worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE == 0 &&
            originZ % worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE == 0) {
            nativeTopology.resize(output.size());
            std::span<worldgen::NativeHydrologyTopologyCell> topologyOutput(nativeTopology.data(),
                                                                            nativeTopology.size());
            generator->sampleNativeHydrologyTopologyGrid(originX, originZ, cellWidth, cellHeight,
                                                         topologyOutput);
        }

        // Topology is independent of the filtered terrain envelope, and must
        // be ready before the bounds batch begins its optional far-geometry
        // work. This also lets an authority deferral release the far worker
        // without spending time constructing a batch that cannot publish.
        GeneratorCellBoundsBatch& batch = threadGeneratorCellBoundsBatch();
        if (!batch.matches(generator.get(), generator->seed(), originX, originZ, step, cellWidth,
                           cellHeight, footprint)) {
            prepareGeneratorCellBoundsBatch(batch, generator, originX, originZ, step, cellWidth,
                                            cellHeight, footprint, true);
        }

        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const size_t index = static_cast<size_t>(z * cellWidth + x);
                GeneratorSurfaceBounds bounds = generatorCellSurfaceBounds(batch, x, z);
                if (!nativeTopology.empty()) {
                    bounds.waterTopologyPossible =
                        bounds.waterTopologyPossible || nativeTopology[index].waterTopologyPossible;
                    bounds.waterfallPossible =
                        bounds.waterfallPossible || nativeTopology[index].waterfallPossible;
                }
                const double minimum = std::clamp(bounds.minimum, static_cast<double>(WORLD_MIN_Y),
                                                  static_cast<double>(WORLD_MAX_Y + 1));
                const double maximum =
                    std::clamp(std::max(bounds.maximum, minimum), static_cast<double>(WORLD_MIN_Y),
                               static_cast<double>(WORLD_MAX_Y + 1));
                const double terrainHeight = std::clamp(bounds.terrainHeight, minimum, maximum);
                output[index] = {
                    .terrainHeight = terrainHeight,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = maximum,
                    .waterTopologyPossible = bounds.waterTopologyPossible,
                    .volcanicWaterPossible = bounds.volcanicWaterPossible,
                    .waterfallPossible = bounds.waterfallPossible,
                };
            }
        }
    };
    source.terrainCellTopGrid = [generator](int64_t originX, int64_t originZ, int step,
                                            int cellEdge, worldgen::SurfaceFootprint footprint,
                                            std::span<float> output) {
        if (step <= 0 || cellEdge <= 0 || worldgen::surfaceFootprintWidth(footprint) != step ||
            output.size() != static_cast<size_t>(cellEdge * cellEdge)) {
            throw std::invalid_argument("invalid far terrain cell top grid");
        }
        if (step >= 4) {
            std::vector<worldgen::SurfaceSample> centers(output.size());
            generator->sampleFarGeometryGrid(checkedCoordinateOffset(originX, step / 2),
                                             checkedCoordinateOffset(originZ, step / 2), step, step,
                                             cellEdge, cellEdge, footprint, centers);
            for (size_t index = 0; index < centers.size(); ++index) {
                output[index] = upwardBound(worldgen::geometryTerrainHeight(centers[index]));
            }
            return;
        }

        const int sampleEdge = cellEdge + 1;
        std::vector<worldgen::SurfaceSample> vertices(static_cast<size_t>(sampleEdge * sampleEdge));
        if (step == 1) {
            generator->sampleExactSurfaceGrid(originX, originZ, step, sampleEdge, vertices);
        } else {
            generator->sampleFarGeometryGrid(originX, originZ, step, step, sampleEdge, sampleEdge,
                                             footprint, vertices);
        }
        for (int cellZ = 0; cellZ < cellEdge; ++cellZ) {
            for (int cellX = 0; cellX < cellEdge; ++cellX) {
                const auto heightAt = [&](int x, int z) {
                    return worldgen::geometryTerrainHeight(
                        vertices[static_cast<size_t>(z * sampleEdge + x)]);
                };
                double height = heightAt(cellX, cellZ);
                if (step == 2) {
                    height = (height + heightAt(cellX + 1, cellZ) + heightAt(cellX + 1, cellZ + 1) +
                              heightAt(cellX, cellZ + 1)) /
                             4.0;
                }
                output[static_cast<size_t>(cellZ * cellEdge + cellX)] = upwardBound(height);
            }
        }
    };
    source.terrainCellTopPoints = [generator](int64_t originX, int64_t originZ, int step,
                                              int cellEdge, worldgen::SurfaceFootprint footprint,
                                              std::span<const uint32_t> occupiedCells,
                                              std::span<float> output) {
        if (step <= 0 || cellEdge <= 0 || worldgen::surfaceFootprintWidth(footprint) != step ||
            occupiedCells.size() != output.size()) {
            throw std::invalid_argument("invalid sparse far terrain cell top query");
        }
        if (step == 1) {
            const size_t cellCount = static_cast<size_t>(cellEdge * cellEdge);
            std::vector<ColumnPos> positions;
            positions.reserve(occupiedCells.size());
            for (size_t index = 0; index < occupiedCells.size(); ++index) {
                const uint32_t occupied = occupiedCells[index];
                if (occupied >= cellCount) {
                    throw std::out_of_range("sparse far terrain cell index exceeds its tile");
                }
                const int cellX = static_cast<int>(occupied % static_cast<uint32_t>(cellEdge));
                const int cellZ = static_cast<int>(occupied / static_cast<uint32_t>(cellEdge));
                positions.push_back({checkedCoordinateOffset(originX, cellX),
                                     checkedCoordinateOffset(originZ, cellZ)});
            }
            // The habitat point path is the sparse batched counterpart of the
            // plan-free exact grid. It reconstructs the emitted density top for
            // all occupied roots under shared macro and hydrology queries,
            // avoiding thousands of independent ColumnPlan lookups.
            std::vector<worldgen::SurfaceSample> samples(positions.size());
            generator->sampleFarHabitatPoints(positions, samples);
            for (size_t index = 0; index < samples.size(); ++index) {
                output[index] =
                    vertexHeight(std::floor(worldgen::geometryTerrainHeight(samples[index]) + 0.5));
            }
            return;
        }
        struct CellProbes {
            std::array<size_t, 4> indices{};
            uint8_t count = 0;
        };
        std::vector<CellProbes> cells(occupiedCells.size());
        std::unordered_map<ColumnPos, size_t> probeIndices;
        probeIndices.reserve(occupiedCells.size() * (step == 2 ? 4 : 1));
        std::vector<ColumnPos> probes;
        probes.reserve(occupiedCells.size() * (step == 2 ? 4 : 1));
        const auto retainProbe = [&](ColumnPos position) {
            const auto [found, inserted] = probeIndices.emplace(position, probes.size());
            if (inserted) probes.push_back(position);
            return found->second;
        };
        const size_t cellCount = static_cast<size_t>(cellEdge * cellEdge);
        for (size_t index = 0; index < occupiedCells.size(); ++index) {
            const uint32_t occupied = occupiedCells[index];
            if (occupied >= cellCount) {
                throw std::out_of_range("sparse far terrain cell index exceeds its tile");
            }
            const int cellX = static_cast<int>(occupied % static_cast<uint32_t>(cellEdge));
            const int cellZ = static_cast<int>(occupied / static_cast<uint32_t>(cellEdge));
            const int64_t cellOriginX =
                checkedCoordinateOffset(originX, static_cast<int64_t>(cellX) * step);
            const int64_t cellOriginZ =
                checkedCoordinateOffset(originZ, static_cast<int64_t>(cellZ) * step);
            CellProbes& cell = cells[index];
            if (step == 2) {
                cell.count = 4;
                cell.indices = {
                    retainProbe({cellOriginX, cellOriginZ}),
                    retainProbe({checkedCoordinateOffset(cellOriginX, step), cellOriginZ}),
                    retainProbe({checkedCoordinateOffset(cellOriginX, step),
                                 checkedCoordinateOffset(cellOriginZ, step)}),
                    retainProbe({cellOriginX, checkedCoordinateOffset(cellOriginZ, step)}),
                };
            } else {
                cell.count = 1;
                const int centerOffset = step >= 4 ? step / 2 : 0;
                cell.indices[0] = retainProbe({checkedCoordinateOffset(cellOriginX, centerOffset),
                                               checkedCoordinateOffset(cellOriginZ, centerOffset)});
            }
        }

        std::vector<worldgen::SurfaceSample> samples(probes.size());
        generator->sampleFarGeometryPoints(probes, footprint, samples);
        for (size_t index = 0; index < cells.size(); ++index) {
            const CellProbes& cell = cells[index];
            double height = 0.0;
            for (uint8_t probe = 0; probe < cell.count; ++probe) {
                height += worldgen::geometryTerrainHeight(samples[cell.indices[probe]]);
            }
            output[index] = upwardBound(height / static_cast<double>(cell.count));
        }
    };
    source.geometryGrid = [generator](int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                      int sampleWidth, int sampleHeight,
                                      worldgen::SurfaceFootprint footprint,
                                      std::span<FarTerrainGeometrySample> output) {
        if (output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
            throw std::invalid_argument("invalid far terrain geometry grid");
        }
        if (generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1 &&
            spacingX == 1 && spacingZ == 1) {
            std::vector<worldgen::SurfaceSample> surfaces(output.size());
            generator->sampleCoarseNativeHydrologyGeometryGrid(originX, originZ, spacingX, spacingZ,
                                                               sampleWidth, sampleHeight, surfaces);
            for (size_t index = 0; index < surfaces.size(); ++index)
                output[index] = geometryFromSurface(surfaces[index]);
            return;
        }
        const bool nearWaterSentinels = spacingX == 2 && spacingZ == 2 &&
                                        sampleWidth == sampleHeight &&
                                        (sampleWidth == FAR_TERRAIN_TILE_EDGE / 2 ||
                                         sampleWidth == FAR_TERRAIN_TILE_EDGE / 2 + 1) &&
                                        footprint == worldgen::SurfaceFootprint::BLOCK_1;
        if (nearWaterSentinels) {
            if (!generator->usesLearnedAuthority()) {
                // Legacy crater lakes are generated by the bounded volcanic
                // overlay rather than the macro hydrology router. Near-water
                // sentinels are authoritative for sub-sample coverage, so
                // they must include that same overlay instead of querying the
                // pre-volcanic generated-water helper.
                std::vector<worldgen::SurfaceSample> surfaces(output.size());
                generator->sampleNativeHydrologyGeometryGrid(originX, originZ, spacingX, spacingZ,
                                                             sampleWidth, sampleHeight, surfaces);
                for (size_t index = 0; index < surfaces.size(); ++index)
                    output[index] = geometryFromSurface(surfaces[index]);
                return;
            }
            std::vector<worldgen::HydrologySample> hydrology(output.size());
            generator->sampleGeneratedWaterAuthorityGrid(originX, originZ, spacingX, sampleWidth,
                                                         hydrology);
            for (size_t index = 0; index < output.size(); ++index)
                output[index] = geometryFromHydrology(hydrology[index]);
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
        if (generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1) {
            std::vector<worldgen::SurfaceSample> surfaces(output.size());
            generator->sampleCoarseNativeHydrologyGeometryPoints(positions, surfaces);
            for (size_t index = 0; index < surfaces.size(); ++index)
                output[index] = geometryFromSurface(surfaces[index]);
            return;
        }
        if (footprint == worldgen::SurfaceFootprint::BLOCK_1 && !positions.empty()) {
            if (!generator->usesLearnedAuthority()) {
                // Keep block-point water recovery consistent with the legacy
                // sentinel grid above. The macro-only authority intentionally
                // does not know about an analytical crater lake.
                std::vector<worldgen::SurfaceSample> surfaces(output.size());
                generator->sampleNativeHydrologyGeometryPoints(positions, surfaces);
                for (size_t index = 0; index < surfaces.size(); ++index)
                    output[index] = geometryFromSurface(surfaces[index]);
                return;
            }
            std::vector<worldgen::HydrologySample> hydrology(output.size());
            generator->sampleGeneratedWaterAuthorityPoints(positions, hydrology);
            for (size_t index = 0; index < hydrology.size(); ++index)
                output[index] = geometryFromHydrology(hydrology[index]);
            return;
        }
        std::vector<worldgen::SurfaceSample> surfaces(output.size());
        generator->sampleFarGeometryPoints(positions, footprint, surfaces);
        for (size_t index = 0; index < surfaces.size(); ++index) {
            output[index] = geometryFromSurface(surfaces[index]);
        }
    };
    source.canonicalWaterGrid = [generator](int64_t originX, int64_t originZ, int spacingX,
                                            int spacingZ, int sampleWidth, int sampleHeight,
                                            worldgen::SurfaceFootprint footprint,
                                            std::span<FarTerrainGeometrySample> output) {
        const bool coarseLearned =
            generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1;
        if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
            output.size() != static_cast<size_t>(sampleWidth * sampleHeight) ||
            (!coarseLearned && footprint != worldgen::SurfaceFootprint::BLOCK_1)) {
            throw std::invalid_argument("invalid canonical far water grid");
        }
        if (generator->usesLearnedAuthority()) {
            // Native v4 hydrology is the canonical pre-volcanic authority.
            // The caller promotes every volcanic candidate to final geometry,
            // so this broad raster must not pay geology and crater work for
            // its ordinary native-water samples.
            std::vector<worldgen::HydrologySample> hydrology(output.size());
            if (coarseLearned) {
                generator->sampleCoarseNativeHydrologyAuthorityGrid(
                    originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight, hydrology);
            } else {
                generator->sampleNativeHydrologyAuthorityGrid(originX, originZ, spacingX, spacingZ,
                                                              sampleWidth, sampleHeight, hydrology);
            }
            for (size_t index = 0; index < hydrology.size(); ++index)
                output[index] = geometryFromHydrology(hydrology[index]);
            return;
        }
        std::vector<worldgen::SurfaceSample> surfaces(output.size());
        generator->sampleNativeHydrologyGeometryGrid(originX, originZ, spacingX, spacingZ,
                                                     sampleWidth, sampleHeight, surfaces);
        for (size_t index = 0; index < surfaces.size(); ++index)
            output[index] = geometryFromSurface(surfaces[index]);
    };
    source.canonicalWaterPoints = [generator](std::span<const ColumnPos> positions,
                                              worldgen::SurfaceFootprint footprint,
                                              std::span<FarTerrainGeometrySample> output) {
        const bool coarseLearned =
            generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1;
        if (positions.size() != output.size() ||
            (!coarseLearned && footprint != worldgen::SurfaceFootprint::BLOCK_1)) {
            throw std::invalid_argument("invalid canonical far water points");
        }
        if (generator->usesLearnedAuthority()) {
            std::vector<worldgen::HydrologySample> hydrology(output.size());
            if (coarseLearned) {
                generator->sampleCoarseNativeHydrologyAuthorityPoints(positions, hydrology);
            } else {
                generator->sampleNativeHydrologyAuthorityPoints(positions, hydrology);
            }
            for (size_t index = 0; index < hydrology.size(); ++index)
                output[index] = geometryFromHydrology(hydrology[index]);
            return;
        }
        std::vector<worldgen::SurfaceSample> surfaces(output.size());
        generator->sampleNativeHydrologyGeometryPoints(positions, surfaces);
        for (size_t index = 0; index < surfaces.size(); ++index)
            output[index] = geometryFromSurface(surfaces[index]);
    };
    source.waterAuthorityGrid = [generator](int64_t originX, int64_t originZ, int spacingX,
                                            int spacingZ, int sampleWidth, int sampleHeight,
                                            worldgen::SurfaceFootprint footprint,
                                            std::span<FarTerrainGeometrySample> output) {
        const bool coarseLearned =
            generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1;
        if (spacingX <= 0 || spacingX != spacingZ || sampleWidth <= 0 ||
            sampleWidth != sampleHeight ||
            output.size() != static_cast<size_t>(sampleWidth * sampleHeight) ||
            (!coarseLearned && footprint != worldgen::SurfaceFootprint::BLOCK_1)) {
            throw std::invalid_argument("invalid far water authority grid");
        }
        std::vector<worldgen::HydrologySample> hydrology(output.size());
        if (coarseLearned) {
            generator->sampleCoarseNativeHydrologyAuthorityGrid(
                originX, originZ, spacingX, spacingX, sampleWidth, sampleHeight, hydrology);
        } else {
            generator->sampleGeneratedWaterAuthorityGrid(originX, originZ, spacingX, sampleWidth,
                                                         hydrology);
        }
        for (size_t index = 0; index < hydrology.size(); ++index)
            output[index] = geometryFromHydrology(hydrology[index]);
    };
    source.waterAuthorityPoints = [generator](std::span<const ColumnPos> positions,
                                              worldgen::SurfaceFootprint footprint,
                                              std::span<FarTerrainGeometrySample> output) {
        const bool coarseLearned =
            generator->usesLearnedAuthority() && footprint != worldgen::SurfaceFootprint::BLOCK_1;
        if (positions.size() != output.size() ||
            (!coarseLearned && footprint != worldgen::SurfaceFootprint::BLOCK_1)) {
            throw std::invalid_argument("invalid far water authority points");
        }
        std::vector<worldgen::HydrologySample> hydrology(output.size());
        if (coarseLearned) {
            generator->sampleCoarseNativeHydrologyAuthorityPoints(positions, hydrology);
        } else {
            generator->sampleGeneratedWaterAuthorityPoints(positions, hydrology);
        }
        for (size_t index = 0; index < hydrology.size(); ++index)
            output[index] = geometryFromHydrology(hydrology[index]);
    };
    source.canopies = [generator](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                  int64_t maximumZ, FarTerrainStep step) {
        return generator->collectFarCanopiesForLod(minimumX, minimumZ, maximumX, maximumZ,
                                                   farTerrainStepSize(step));
    };
    source.flora = [generator](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                               int64_t maximumZ, FarTerrainStep step) {
        return generator->collectFarFloraForLod(minimumX, minimumZ, maximumX, maximumZ,
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
    canopyWorkers_.reserve(CANOPY_WORKER_COUNT);
    for (size_t index = 0; index < CANOPY_WORKER_COUNT; ++index)
        canopyWorkers_.emplace_back([this] { canopyWorkerLoop(); });
}

FarTerrainScheduler::FarTerrainScheduler(
    FarTerrainSource source,
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
    FarTerrainSchedulerLimits limits)
    : source_(std::move(source))
    , limits_(limits) {
    if (!source_.sample) {
        throw std::invalid_argument("far terrain scheduler source is incomplete");
    }
    if (generationContext) {
        generationContext_ = generationContext->withRequestPriority(
            worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
        protectedSource_ = source_;
        if (generationContext->quality() == worldgen::learned::AuthorityQuality::FINAL) {
            previewGenerationContext_ =
                generationContext->withQuality(worldgen::learned::AuthorityQuality::PREVIEW)
                    ->withRequestPriority(
                        worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW);
            previewSource_ = source_;
        }
    }
    validateLimits(limits_);
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        const bool latencySensitive = index < LATENCY_WORKER_COUNT;
        workers_.emplace_back([this, latencySensitive] { workerLoop(latencySensitive); });
    }
    canopyWorkers_.reserve(CANOPY_WORKER_COUNT);
    for (size_t index = 0; index < CANOPY_WORKER_COUNT; ++index)
        canopyWorkers_.emplace_back([this] { canopyWorkerLoop(); });
}

FarTerrainScheduler::FarTerrainScheduler(uint64_t worldSeed, FarTerrainSchedulerLimits limits,
                                         GenerationSettings generation)
    : FarTerrainScheduler(worldSeed, nullptr, limits, generation) {}

FarTerrainScheduler::FarTerrainScheduler(
    uint64_t worldSeed,
    std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
    FarTerrainSchedulerLimits limits, GenerationSettings generation)
    : limits_(limits) {
    std::shared_ptr<worldgen::learned::WorldGenerationContext> finalContext = generationContext;
    if (finalContext) {
        finalContext = finalContext->withRequestPriority(
            worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    }
    generationContext_ = finalContext;
    generator_ = std::make_shared<ChunkGenerator>(worldSeed, finalContext, generation);
    source_ = FarTerrainMesher::generatorGeometrySource(generator_);
    if (generationContext) {
        const std::shared_ptr<worldgen::learned::WorldGenerationContext> protectedContext =
            generationContext->withQuality(worldgen::learned::AuthorityQuality::FINAL)
                ->withRequestPriority(
                    worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF);
        protectedSource_ = FarTerrainMesher::generatorGeometrySource(
            std::make_shared<ChunkGenerator>(worldSeed, protectedContext, generation));
        const std::shared_ptr<worldgen::learned::WorldGenerationContext> canopyContext =
            generationContext->withQuality(worldgen::learned::AuthorityQuality::FINAL)
                ->withRequestPriority(FAR_TERRAIN_CANOPY_AUTHORITY_PRIORITY);
        canopyGenerator_ = std::make_shared<ChunkGenerator>(worldSeed, canopyContext, generation);
        canopySource_ = FarTerrainMesher::generatorGeometrySource(canopyGenerator_);
    }
    if (generationContext &&
        generationContext->quality() == worldgen::learned::AuthorityQuality::FINAL) {
        previewGenerationContext_ =
            generationContext->withQuality(worldgen::learned::AuthorityQuality::PREVIEW)
                ->withRequestPriority(worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW);
        previewGenerator_ =
            std::make_shared<ChunkGenerator>(worldSeed, previewGenerationContext_, generation);
        previewSource_ = FarTerrainMesher::generatorGeometrySource(previewGenerator_);
    }
    validateLimits(limits_);
    workers_.reserve(WORKER_COUNT);
    for (size_t index = 0; index < WORKER_COUNT; ++index) {
        const bool latencySensitive = index < LATENCY_WORKER_COUNT;
        workers_.emplace_back([this, latencySensitive] { workerLoop(latencySensitive); });
    }
    canopyWorkers_.reserve(CANOPY_WORKER_COUNT);
    for (size_t index = 0; index < CANOPY_WORKER_COUNT; ++index)
        canopyWorkers_.emplace_back([this] { canopyWorkerLoop(); });
}

FarTerrainScheduler::~FarTerrainScheduler() {
    shutdown();
}

bool FarTerrainScheduler::enqueue(FarTerrainKey key, uint32_t viewPriority) {
    const FarTerrainAuthorityQuality quality =
        previewSource_.sample && key.step != FarTerrainStep::ONE
            ? FarTerrainAuthorityQuality::PREVIEW
            : FarTerrainAuthorityQuality::FINAL;
    const worldgen::learned::AuthorityRequestPriority authorityPriority =
        quality == FarTerrainAuthorityQuality::PREVIEW
            ? worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW
            : worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT;
    return enqueueInternal(key, viewPriority, false, false, quality, authorityPriority);
}

bool FarTerrainScheduler::enqueueUrgentCoverage(FarTerrainKey key, uint32_t viewPriority) {
    if (!farTerrainIsBaseStep(key.step)) return false;
    const FarTerrainAuthorityQuality quality = previewSource_.sample
                                                   ? FarTerrainAuthorityQuality::PREVIEW
                                                   : FarTerrainAuthorityQuality::FINAL;
    // Urgent coverage controls scheduler admission and displacement. Protected
    // handoff epochs are valid only for FINAL authority, so a missing PREVIEW
    // parent must remain in the coarse-preview authority lane.
    const worldgen::learned::AuthorityRequestPriority priority =
        quality == FarTerrainAuthorityQuality::PREVIEW
            ? worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW
            : worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF;
    return enqueueInternal(key, viewPriority, true, true, quality, priority);
}

bool FarTerrainScheduler::enqueueFinalBase(FarTerrainKey key, uint32_t viewPriority,
                                           bool protectedHandoff) {
    if (!farTerrainIsBaseStep(key.step) || !generationContext_ ||
        !finalStreamingWorkEnabled_.load(std::memory_order_acquire)) {
        return false;
    }
    return enqueueInternal(
        key, viewPriority, true, protectedHandoff, FarTerrainAuthorityQuality::FINAL,
        protectedHandoff ? worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF
                         : worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
}

bool FarTerrainScheduler::enqueueCanopy(FarTerrainKey key, uint32_t viewPriority,
                                        FarTerrainAuthorityQuality groundingQuality) {
    const std::shared_ptr<const FarCanopyAttachment> cached = findCachedCanopy(key);
    if (!validStep(key.step) || !running_.load(std::memory_order_acquire) ||
        (cached && cached->authorityQuality == FarTerrainAuthorityQuality::FINAL &&
         cached->groundingQuality == groundingQuality)) {
        return false;
    }
    return enqueueCanopyInternal(key, epoch_.load(std::memory_order_acquire), viewPriority,
                                 groundingQuality);
}

bool FarTerrainScheduler::enqueueUrgentRefinement(FarTerrainKey key, uint32_t viewPriority,
                                                  bool cameraNearCritical) {
    if (farTerrainIsBaseStep(key.step)) return false;
    const FarTerrainAuthorityQuality quality =
        previewSource_.sample && key.step != FarTerrainStep::ONE
            ? FarTerrainAuthorityQuality::PREVIEW
            : FarTerrainAuthorityQuality::FINAL;
    const worldgen::learned::AuthorityRequestPriority priority =
        quality == FarTerrainAuthorityQuality::PREVIEW
            ? worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW
            : worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT;
    return enqueueInternal(key, viewPriority, true, cameraNearCritical, quality, priority);
}

bool FarTerrainScheduler::enqueueUrgentFinalRefinement(FarTerrainKey key, uint32_t viewPriority,
                                                       bool cameraNearCritical) {
    if (farTerrainIsBaseStep(key.step) || !generationContext_ ||
        !finalStreamingWorkEnabled_.load(std::memory_order_acquire)) {
        return false;
    }
    return enqueueInternal(key, viewPriority, true, cameraNearCritical,
                           FarTerrainAuthorityQuality::FINAL,
                           worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF);
}

bool FarTerrainScheduler::enqueueFinalRefinement(FarTerrainKey key, uint32_t viewPriority,
                                                 bool cameraNearCritical) {
    if (farTerrainIsBaseStep(key.step) || !generationContext_ ||
        !finalStreamingWorkEnabled_.load(std::memory_order_acquire)) {
        return false;
    }
    return enqueueInternal(key, viewPriority, true, cameraNearCritical,
                           FarTerrainAuthorityQuality::FINAL,
                           worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
}

void FarTerrainScheduler::setFinalStreamingWorkEnabled(bool enabled) noexcept {
    finalStreamingWorkEnabled_.store(enabled, std::memory_order_release);
}

bool FarTerrainScheduler::jobBefore(const Job& first, const Job& second) noexcept {
    if (first.urgentRefinement != second.urgentRefinement) return first.urgentRefinement;
    if (first.authorityPriority == worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF &&
        second.authorityPriority ==
            worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF &&
        first.protectedHandoffEpoch != second.protectedHandoffEpoch) {
        return first.protectedHandoffEpoch > second.protectedHandoffEpoch;
    }
    if (first.urgentRefinement) {
        if (first.cameraNearCritical != second.cameraNearCritical) return first.cameraNearCritical;
        if (first.viewPriority != second.viewPriority)
            return first.viewPriority < second.viewPriority;
    }
    if (first.authorityPriority != second.authorityPriority) {
        return first.authorityPriority < second.authorityPriority;
    }
    if (first.authorityQuality != second.authorityQuality) {
        return first.authorityQuality == FarTerrainAuthorityQuality::FINAL;
    }
    return farTerrainSubmissionBefore(first.key, first.viewPriority, second.key,
                                      second.viewPriority);
}

bool FarTerrainScheduler::canopyJobBefore(const CanopyJob& first,
                                          const CanopyJob& second) noexcept {
    if (first.ecologyQuality != second.ecologyQuality) {
        return first.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW;
    }
    if (first.viewPriority != second.viewPriority) return first.viewPriority < second.viewPriority;
    const bool firstBase = farTerrainIsBaseStep(first.key.step);
    const bool secondBase = farTerrainIsBaseStep(second.key.step);
    if (firstBase != secondBase) return !firstBase;
    if (first.key.tileX != second.key.tileX) return first.key.tileX < second.key.tileX;
    if (first.key.tileZ != second.key.tileZ) return first.key.tileZ < second.key.tileZ;
    return farTerrainStepSize(first.key.step) < farTerrainStepSize(second.key.step);
}

FarTerrainScheduler::Job FarTerrainScheduler::mergeJobRequest(const Job& current,
                                                              const Job& requested) {
    Job merged = current;
    merged.viewPriority = std::min(current.viewPriority, requested.viewPriority);
    merged.urgentRefinement = current.urgentRefinement || requested.urgentRefinement;
    merged.cameraNearCritical = current.cameraNearCritical || requested.cameraNearCritical;
    merged.visibleFinalParent = current.visibleFinalParent || requested.visibleFinalParent;
    if (static_cast<uint8_t>(requested.authorityQuality) >
        static_cast<uint8_t>(merged.authorityQuality)) {
        merged.authorityQuality = requested.authorityQuality;
    }
    if (requested.authorityPriority < merged.authorityPriority) {
        merged.authorityPriority = requested.authorityPriority;
    }
    merged.protectedHandoffEpoch =
        std::max(current.protectedHandoffEpoch, requested.protectedHandoffEpoch);
    return merged;
}

bool FarTerrainScheduler::sameJobRequest(const Job& first, const Job& second) noexcept {
    return first.key == second.key && first.epoch == second.epoch &&
           first.viewPriority == second.viewPriority &&
           first.urgentRefinement == second.urgentRefinement &&
           first.cameraNearCritical == second.cameraNearCritical &&
           first.visibleFinalParent == second.visibleFinalParent &&
           first.authorityQuality == second.authorityQuality &&
           first.authorityPriority == second.authorityPriority &&
           first.protectedHandoffEpoch == second.protectedHandoffEpoch;
}

bool FarTerrainScheduler::executionUpgradeRequested(const Job& current,
                                                    const Job& requested) noexcept {
    return (!current.urgentRefinement && requested.urgentRefinement) ||
           (!current.cameraNearCritical && requested.cameraNearCritical) ||
           (!current.visibleFinalParent && requested.visibleFinalParent) ||
           static_cast<uint8_t>(requested.authorityQuality) >
               static_cast<uint8_t>(current.authorityQuality) ||
           requested.authorityPriority < current.authorityPriority ||
           requested.protectedHandoffEpoch > current.protectedHandoffEpoch;
}

bool FarTerrainScheduler::makeRoomForJobLocked(const Job& incoming, bool removeVictim) {
    const bool pendingFull = inFlight_.load(std::memory_order_relaxed) >= limits_.maxPending;
    const bool urgentFull = incoming.urgentRefinement &&
                            urgentRefinementInFlightCount_.load(std::memory_order_relaxed) >=
                                FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT;
    const bool visibleFinalParentFull =
        incoming.visibleFinalParent &&
        visibleFinalParentInFlightCount_.load(std::memory_order_relaxed) >=
            FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT;
    if (!pendingFull && !urgentFull && !visibleFinalParentFull) return true;
    if (!incoming.urgentRefinement) return false;

    enum class VictimKind : uint8_t {
        NONE,
        QUEUED,
        PARKED,
    };
    VictimKind victimKind = VictimKind::NONE;
    std::deque<Job>::iterator queuedVictim = jobs_.end();
    std::unordered_map<FarTerrainKey, ParkedBaseJob, FarTerrainKeyHash>::iterator parkedVictim =
        parkedBaseJobs_.end();
    const Job* victim = nullptr;
    const bool criticalPreviewCoverage =
        incoming.urgentRefinement && incoming.cameraNearCritical &&
        farTerrainIsBaseStep(incoming.key.step) &&
        incoming.authorityQuality == FarTerrainAuthorityQuality::PREVIEW;
    const auto eligible = [&](const Job& candidate) {
        // Preparation retains gap-free parent priority. During gameplay's
        // near-first phase, a connected desired-LOD request may reclaim a
        // queued ordinary horizon parent because the latter is less visible
        // than a nearby surface changing under the player.
        if (farTerrainIsBaseStep(candidate.key.step) && !candidate.urgentRefinement &&
            !nearFirstWorkEnabled_ && (!criticalPreviewCoverage || urgentFull)) {
            return false;
        }
        if (urgentFull && !candidate.urgentRefinement) return false;
        if (visibleFinalParentFull && !candidate.visibleFinalParent) return false;
        return jobBefore(incoming, candidate);
    };
    const auto consider = [&](const Job& candidate, VictimKind kind, auto iterator) {
        if (!eligible(candidate)) return;
        if (victim && !jobBefore(*victim, candidate)) return;
        victim = &candidate;
        victimKind = kind;
        if constexpr (std::is_same_v<decltype(iterator), std::deque<Job>::iterator>) {
            queuedVictim = iterator;
        } else {
            parkedVictim = iterator;
        }
    };
    for (auto candidate = jobs_.begin(); candidate != jobs_.end(); ++candidate)
        consider(*candidate, VictimKind::QUEUED, candidate);
    for (auto candidate = parkedBaseJobs_.begin(); candidate != parkedBaseJobs_.end(); ++candidate)
        consider(candidate->second.job, VictimKind::PARKED, candidate);
    if (!victim) return false;
    if (!removeVictim) return true;

    const Job displaced = *victim;
    if (victimKind == VictimKind::QUEUED) {
        (farTerrainIsBaseStep(displaced.key.step) ? queuedBaseCount_ : queuedRefinementCount_)
            .fetch_sub(1, std::memory_order_relaxed);
        if (displaced.urgentRefinement)
            queuedUrgentRefinementCount_.fetch_sub(1, std::memory_order_relaxed);
        jobs_.erase(queuedVictim);
    } else {
        removeParkedBaseWaitersLocked(parkedVictim->second);
        parkedBaseJobs_.erase(parkedVictim);
        parkedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (const auto active = activeKeys_.find(displaced.key);
        active != activeKeys_.end() && active->second == displaced.epoch) {
        activeKeys_.erase(active);
    }
    if (const auto followup = terrainFollowupJobs_.find(displaced.key);
        followup != terrainFollowupJobs_.end()) {
        terrainFollowupJobs_.erase(followup);
        terrainFollowupCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (displaced.urgentRefinement)
        urgentRefinementInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    if (displaced.visibleFinalParent)
        visibleFinalParentInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    inFlight_.fetch_sub(1, std::memory_order_relaxed);
    canceled_.fetch_add(1, std::memory_order_relaxed);
    if (incoming.cameraNearCritical) criticalDisplacements_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void FarTerrainScheduler::queueJobLocked(Job job) {
    const auto insertion = std::find_if(jobs_.begin(), jobs_.end(),
                                        [&](const Job& queued) { return jobBefore(job, queued); });
    const bool base = farTerrainIsBaseStep(job.key.step);
    const bool urgent = job.urgentRefinement;
    jobs_.insert(insertion, std::move(job));
    (base ? queuedBaseCount_ : queuedRefinementCount_).fetch_add(1, std::memory_order_relaxed);
    if (urgent) queuedUrgentRefinementCount_.fetch_add(1, std::memory_order_relaxed);
}

FarTerrainScheduler::ExistingJobResolution
FarTerrainScheduler::upgradeExistingJobLocked(const Job& requested) {
    const auto active = activeKeys_.find(requested.key);
    if (active == activeKeys_.end() || active->second != requested.epoch) {
        return ExistingJobResolution::NotFound;
    }

    const auto accountUpgrade = [&](const Job& previous, const Job& upgraded) {
        if (!previous.urgentRefinement && upgraded.urgentRefinement) {
            urgentRefinementInFlightCount_.fetch_add(1, std::memory_order_relaxed);
        }
        if (!previous.visibleFinalParent && upgraded.visibleFinalParent) {
            visibleFinalParentInFlightCount_.fetch_add(1, std::memory_order_relaxed);
        }
    };

    if (const auto queued = std::find_if(jobs_.begin(), jobs_.end(),
                                         [&](const Job& job) {
                                             return job.key == requested.key &&
                                                    job.epoch == requested.epoch;
                                         });
        queued != jobs_.end()) {
        const Job previous = *queued;
        const Job upgraded = mergeJobRequest(previous, requested);
        if (sameJobRequest(previous, upgraded)) return ExistingJobResolution::Unchanged;
        accountUpgrade(previous, upgraded);
        if (!previous.urgentRefinement && upgraded.urgentRefinement) {
            queuedUrgentRefinementCount_.fetch_add(1, std::memory_order_relaxed);
        }
        jobs_.erase(queued);
        const auto insertion = std::find_if(
            jobs_.begin(), jobs_.end(), [&](const Job& job) { return jobBefore(upgraded, job); });
        jobs_.insert(insertion, upgraded);
        jobCv_.notify_one();
        return ExistingJobResolution::Upgraded;
    }

    if (const auto parked = parkedBaseJobs_.find(requested.key);
        parked != parkedBaseJobs_.end() && parked->second.job.epoch == requested.epoch) {
        const Job previous = parked->second.job;
        const Job upgraded = mergeJobRequest(previous, requested);
        if (sameJobRequest(previous, upgraded)) return ExistingJobResolution::Unchanged;
        accountUpgrade(previous, upgraded);
        removeParkedBaseWaitersLocked(parked->second);
        parkedBaseJobs_.erase(parked);
        parkedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        queueJobLocked(upgraded);
        jobCv_.notify_one();
        return ExistingJobResolution::Upgraded;
    }

    const auto worker = activeWorkerJobs_.find(requested.key);
    if (worker == activeWorkerJobs_.end() || worker->second.epoch != requested.epoch) {
        return ExistingJobResolution::Unchanged;
    }
    const auto followup = terrainFollowupJobs_.find(requested.key);
    const Job& previous =
        followup == terrainFollowupJobs_.end() ? worker->second : followup->second;
    if (followup == terrainFollowupJobs_.end() && !executionUpgradeRequested(previous, requested)) {
        return ExistingJobResolution::Unchanged;
    }
    const Job upgraded = mergeJobRequest(previous, requested);
    if (sameJobRequest(previous, upgraded)) return ExistingJobResolution::Unchanged;
    accountUpgrade(previous, upgraded);
    if (followup == terrainFollowupJobs_.end()) {
        terrainFollowupJobs_.emplace(requested.key, upgraded);
        terrainFollowupCount_.fetch_add(1, std::memory_order_relaxed);
    } else {
        followup->second = upgraded;
    }
    return ExistingJobResolution::Upgraded;
}

bool FarTerrainScheduler::enqueueInternal(
    FarTerrainKey key, uint32_t viewPriority, bool urgentRefinement, bool cameraNearCritical,
    FarTerrainAuthorityQuality authorityQuality,
    worldgen::learned::AuthorityRequestPriority authorityPriority) {
    if (!validStep(key.step) || !running_.load(std::memory_order_acquire) ||
        (authorityQuality == FarTerrainAuthorityQuality::FINAL &&
         !finalStreamingWorkEnabled_.load(std::memory_order_acquire))) {
        return false;
    }
    if (const std::shared_ptr<const FarTerrainMesh> cached = findCached(key);
        cached && farTerrainAuthoritySatisfies(cached->authorityQuality, authorityQuality)) {
        return false;
    }
    const uint64_t current = epoch_.load(std::memory_order_acquire);
    const uint64_t protectedEpoch =
        authorityPriority == worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF
            ? protectedHandoffEpoch_.load(std::memory_order_acquire)
            : uint64_t{0};
    const bool visibleFinalParent =
        urgentRefinement && farTerrainIsBaseStep(key.step) &&
        authorityQuality == FarTerrainAuthorityQuality::FINAL &&
        authorityPriority == worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT;

    Job job{.key = key,
            .epoch = current,
            .viewPriority = viewPriority,
            .urgentRefinement = urgentRefinement,
            .cameraNearCritical = cameraNearCritical,
            .visibleFinalParent = visibleFinalParent,
            .authorityQuality = authorityQuality,
            .authorityPriority = authorityPriority,
            .protectedHandoffEpoch = protectedEpoch};
    const bool nonurgentBase = farTerrainIsBaseStep(key.step) && !urgentRefinement;

    // A stronger request upgrades the existing logical key before any capacity
    // checks. This lets protected FINAL work pass a parked preview without
    // allocating a duplicate scheduler slot.
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed)) return false;
        if (wantedMembership_ && !wantedMembership_->keys.contains(key)) return false;
        const ExistingJobResolution existing = upgradeExistingJobLocked(job);
        if (existing != ExistingJobResolution::NotFound) {
            return existing == ExistingJobResolution::Upgraded;
        }
        const size_t inFlight = inFlight_.load(std::memory_order_relaxed);
        if ((nonurgentBase &&
             inFlight >= farTerrainNonurgentBaseAdmissionLimit(limits_.maxPending)) ||
            !makeRoomForJobLocked(job, false))
            return false;
    }
    BaseAuthorityWaitSet waitingOn;
    BaseAuthorityPreparation authorityPreparation = BaseAuthorityPreparation::Ready;
    if (farTerrainIsBaseStep(key.step) && (previewGenerationContext_ || generationContext_)) {
        authorityPreparation = prepareBaseAuthority(job, waitingOn);
        if (authorityPreparation == BaseAuthorityPreparation::Failed) return false;
    }

    bool parked = false;
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            epoch_.load(std::memory_order_relaxed) != current) {
            return false;
        }
        if (wantedMembership_ && !wantedMembership_->keys.contains(key)) return false;
        const ExistingJobResolution existing = upgradeExistingJobLocked(job);
        if (existing != ExistingJobResolution::NotFound) {
            return existing == ExistingJobResolution::Upgraded;
        }
        const size_t inFlight = inFlight_.load(std::memory_order_relaxed);
        if ((nonurgentBase &&
             inFlight >= farTerrainNonurgentBaseAdmissionLimit(limits_.maxPending)) ||
            !makeRoomForJobLocked(job, true))
            return false;
        activeKeys_[key] = current;
        if (authorityPreparation == BaseAuthorityPreparation::Deferred) {
            parkBaseJobLocked(std::move(job), std::move(waitingOn));
            parked = true;
        } else {
            queueJobLocked(job);
        }
        if (urgentRefinement)
            urgentRefinementInFlightCount_.fetch_add(1, std::memory_order_relaxed);
        if (visibleFinalParent)
            visibleFinalParentInFlightCount_.fetch_add(1, std::memory_order_relaxed);
        inFlight_.fetch_add(1, std::memory_order_relaxed);
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }
    if (parked) {
        // Close the small race where a page becomes ready after the initial
        // status pass but before this key is registered as its waiter.
        static_cast<void>(
            refreshParkedBaseJob(key, worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS));
    } else {
        jobCv_.notify_one();
    }
    return true;
}

FarTerrainScheduler::BaseAuthorityPreparation
FarTerrainScheduler::prepareBaseAuthority(Job& job, BaseAuthorityWaitSet& waitingOn) {
    using worldgen::learned::AuthorityRequestPriority;
    using worldgen::learned::AuthorityStatus;
    using worldgen::learned::GenerationFailure;
    using worldgen::learned::GenerationFailureCode;
    using worldgen::learned::TerrainPageCoordinate;

    waitingOn.pages.clear();
    waitingOn.nativeHydrology.clear();
    waitingOn.minimumAuthorityCompletion.reset();
    job.authorityDependencies.clear();
    job.nativeHydrologyDependencies.clear();
    job.transientGeometryDependency.reset();
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context =
        job.authorityQuality == FarTerrainAuthorityQuality::PREVIEW ? previewGenerationContext_
                                                                    : generationContext_;
    if (!context) return BaseAuthorityPreparation::Ready;
    const AuthorityRequestPriority priority = job.authorityPriority;
    const worldgen::learned::ProtectedHandoffEpoch protectedEpoch{
        priority == AuthorityRequestPriority::PROTECTED_HANDOFF ? job.protectedHandoffEpoch
                                                                : uint64_t{0}};

    try {
        const FarTerrainSource& source = sourceFor(job.authorityQuality, job.authorityPriority);
        if (job.authorityQuality == FarTerrainAuthorityQuality::FINAL &&
            source.finalBaseAuthorityDependencies) {
            FarTerrainFinalBaseAuthorityDependencies dependencies =
                source.finalBaseAuthorityDependencies(job.key);
            if (dependencies.minimumWorldX >= dependencies.maximumWorldXExclusive ||
                dependencies.minimumWorldZ >= dependencies.maximumWorldZExclusive ||
                dependencies.geometryPages.empty() || dependencies.nativeHydrology.empty()) {
                throw std::invalid_argument("FINAL base authority dependency plan is empty");
            }
            std::ranges::sort(dependencies.geometryPages);
            dependencies.geometryPages.erase(
                std::unique(dependencies.geometryPages.begin(), dependencies.geometryPages.end()),
                dependencies.geometryPages.end());
            std::ranges::sort(dependencies.nativeHydrology);
            dependencies.nativeHydrology.erase(std::unique(dependencies.nativeHydrology.begin(),
                                                           dependencies.nativeHydrology.end()),
                                               dependencies.nativeHydrology.end());
            if (dependencies.geometryPages.size() >
                    worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS ||
                dependencies.nativeHydrology.size() >
                    worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS -
                        dependencies.geometryPages.size()) {
                throw std::invalid_argument(
                    "FINAL base authority dependency plan exceeds the bounded queue");
            }
            for (const FarTerrainNativeHydrologyDependency& dependency :
                 dependencies.nativeHydrology) {
                if (!dependency.finalTerrainRegion.valid() ||
                    dependency.finalTerrainRegion !=
                        worldgen::nativeHydrologyFinalTerrainRegion(dependency.ownerPageX,
                                                                    dependency.ownerPageZ)) {
                    throw std::invalid_argument(
                        "FINAL base authority contains an invalid hydrology owner rectangle");
                }
            }
            if (dependencies.transientGeometryRegion) {
                if (!dependencies.transientGeometryRegion->valid() ||
                    !std::ranges::any_of(
                        dependencies.nativeHydrology,
                        [&](const FarTerrainNativeHydrologyDependency& dependency) {
                            const auto outer = dependency.finalTerrainRegion;
                            const auto inner = *dependencies.transientGeometryRegion;
                            return outer.rowBegin <= inner.rowBegin &&
                                   outer.columnBegin <= inner.columnBegin &&
                                   outer.rowEnd >= inner.rowEnd &&
                                   outer.columnEnd >= inner.columnEnd;
                        })) {
                    throw std::invalid_argument(
                        "FINAL base transient geometry is outside its hydrology input");
                }
            }
            job.authorityDependencies = std::move(dependencies.geometryPages);
            job.nativeHydrologyDependencies = std::move(dependencies.nativeHydrology);
            job.transientGeometryDependency = dependencies.transientGeometryRegion;
        } else {
            const int64_t originX = tileOrigin(job.key.tileX);
            const int64_t originZ = tileOrigin(job.key.tileZ);
            std::array<FarTerrainViewTile, 1> selected{};
            selected.front().key = job.key;
            selected.front().bounds = {.minX = originX,
                                       .maxX = originX + FAR_TERRAIN_TILE_EDGE,
                                       .minZ = originZ,
                                       .maxZ = originZ + FAR_TERRAIN_TILE_EDGE,
                                       .minY = static_cast<float>(WORLD_MIN_Y),
                                       .maxY = static_cast<float>(WORLD_MAX_Y + 1)};
            job.authorityDependencies = farTerrainCoarseAuthorityPages(
                selected, static_cast<double>(originX) + FAR_TERRAIN_TILE_EDGE / 2.0,
                static_cast<double>(originZ) + FAR_TERRAIN_TILE_EDGE / 2.0);
        }
    } catch (const std::exception& error) {
        context->latchFailure(
            {GenerationFailureCode::INVALID_REQUEST,
             std::string("Could not resolve base terrain authority dependencies: ") + error.what(),
             false});
        return BaseAuthorityPreparation::Failed;
    }

    for (const FarTerrainNativeHydrologyDependency& dependency : job.nativeHydrologyDependencies) {
        const bool containsTransientGeometry =
            job.transientGeometryDependency &&
            dependency.finalTerrainRegion.rowBegin <= job.transientGeometryDependency->rowBegin &&
            dependency.finalTerrainRegion.columnBegin <=
                job.transientGeometryDependency->columnBegin &&
            dependency.finalTerrainRegion.rowEnd >= job.transientGeometryDependency->rowEnd &&
            dependency.finalTerrainRegion.columnEnd >= job.transientGeometryDependency->columnEnd;
        if (!containsTransientGeometry &&
            context->nativeHydrologyOwnerPrepared(dependency.ownerPageX, dependency.ownerPageZ)) {
            continue;
        }
        const auto prepared = context->queryTransientFinalNativeGrid(dependency.finalTerrainRegion,
                                                                     priority, protectedEpoch);
        if (prepared.status() == AuthorityStatus::FAILED) {
            context->latchFailure(
                prepared.failure()
                    ? *prepared.failure()
                    : GenerationFailure{.code = GenerationFailureCode::INFERENCE_FAILED,
                                        .message = "Base native hydrology input request failed",
                                        .retriable = true});
            return BaseAuthorityPreparation::Failed;
        }
        if (prepared.status() == AuthorityStatus::DEFERRED)
            waitingOn.nativeHydrology.push_back(dependency);
    }
    // A containing owner query is also the geometry dependency. Production
    // authority serves the smaller geometry rectangle as an immutable crop of
    // that cached grid, so a second coordinator request is neither necessary
    // nor desirable. Keep the geometry rectangle on the job for meshing after
    // every containing owner becomes observable.
    if (!waitingOn.nativeHydrology.empty()) return BaseAuthorityPreparation::Deferred;

    if (!job.transientGeometryDependency) {
        for (const TerrainPageCoordinate coordinate : job.authorityDependencies) {
            const auto prepared =
                context->requestAuthorityPage(coordinate, priority, protectedEpoch);
            if (prepared.status() == AuthorityStatus::FAILED) {
                context->latchFailure(
                    prepared.failure()
                        ? *prepared.failure()
                        : GenerationFailure{.code = GenerationFailureCode::INFERENCE_FAILED,
                                            .message = "Base terrain authority request failed",
                                            .retriable = true});
                return BaseAuthorityPreparation::Failed;
            }
            if (prepared.status() == AuthorityStatus::DEFERRED)
                waitingOn.pages.push_back(coordinate);
        }
    }
    return waitingOn.empty() ? BaseAuthorityPreparation::Ready : BaseAuthorityPreparation::Deferred;
}

void FarTerrainScheduler::parkBaseJobLocked(Job job, BaseAuthorityWaitSet waitingOn) {
    if (waitingOn.empty()) return;
    const FarTerrainKey key = job.key;
    const auto [parked, inserted] =
        parkedBaseJobs_.emplace(key, ParkedBaseJob{std::move(job), std::move(waitingOn)});
    if (!inserted) return;
    for (const worldgen::learned::TerrainPageCoordinate coordinate : parked->second.waitingOn.pages)
        parkedBaseWaiters_[coordinate].push_back(key);
    parkedBaseCount_.fetch_add(1, std::memory_order_relaxed);
}

void FarTerrainScheduler::removeParkedBaseWaitersLocked(const ParkedBaseJob& job) {
    for (const worldgen::learned::TerrainPageCoordinate coordinate : job.waitingOn.pages) {
        const auto waiting = parkedBaseWaiters_.find(coordinate);
        if (waiting == parkedBaseWaiters_.end()) continue;
        std::erase(waiting->second, job.job.key);
        if (waiting->second.empty()) parkedBaseWaiters_.erase(waiting);
    }
}

void FarTerrainScheduler::releaseActiveWorkerLocked(const Job& job) {
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

bool FarTerrainScheduler::parkActiveBaseJob(Job job, BaseAuthorityWaitSet waitingOn) {
    if (!farTerrainIsBaseStep(job.key.step) || waitingOn.empty()) return false;
    {
        std::lock_guard lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            job.epoch != epoch_.load(std::memory_order_relaxed) ||
            (wantedMembership_ && !wantedMembership_->keys.contains(job.key))) {
            return false;
        }
        const auto active = activeKeys_.find(job.key);
        if (active == activeKeys_.end() || active->second != job.epoch ||
            parkedBaseJobs_.contains(job.key)) {
            return false;
        }
        const Job activeJob = job;
        activeWorkerJobs_.erase(job.key);
        if (const auto followup = terrainFollowupJobs_.find(job.key);
            followup != terrainFollowupJobs_.end() && followup->second.epoch == job.epoch) {
            queueJobLocked(followup->second);
            terrainFollowupJobs_.erase(followup);
            terrainFollowupCount_.fetch_sub(1, std::memory_order_relaxed);
            submitted_.fetch_add(1, std::memory_order_relaxed);
        } else {
            parkBaseJobLocked(std::move(job), std::move(waitingOn));
        }
        releaseActiveWorkerLocked(activeJob);
    }
    jobCv_.notify_all();
    return true;
}

void FarTerrainScheduler::wakeParkedBaseJobsForReadyPage(
    worldgen::learned::TerrainPageCoordinate coordinate) {
    std::vector<FarTerrainKey> keys;
    {
        std::lock_guard lock(jobMutex_);
        const auto waiters = parkedBaseWaiters_.find(coordinate);
        if (waiters == parkedBaseWaiters_.end()) return;
        keys = std::move(waiters->second);
        parkedBaseWaiters_.erase(waiters);
        for (const FarTerrainKey key : keys) {
            const auto parked = parkedBaseJobs_.find(key);
            if (parked == parkedBaseJobs_.end()) continue;
            std::erase(parked->second.waitingOn.pages, coordinate);
        }
    }
    for (const FarTerrainKey key : keys)
        wakeParkedBaseJobIfReady(key);
}

void FarTerrainScheduler::wakeParkedBaseJobIfReady(FarTerrainKey key) {
    bool canceled = false;
    bool canceledUrgent = false;
    bool canceledVisibleFinalParent = false;
    bool queued = false;
    {
        std::lock_guard lock(jobMutex_);
        const auto parked = parkedBaseJobs_.find(key);
        if (parked == parkedBaseJobs_.end() || !parked->second.waitingOn.empty()) return;
        Job job = std::move(parked->second.job);
        removeParkedBaseWaitersLocked(parked->second);
        parkedBaseJobs_.erase(parked);
        parkedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        const bool stale = !running_.load(std::memory_order_relaxed) ||
                           job.epoch != epoch_.load(std::memory_order_relaxed) ||
                           (wantedMembership_ && !wantedMembership_->keys.contains(job.key));
        if (stale) {
            const auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch)
                activeKeys_.erase(active);
            canceledUrgent = job.urgentRefinement;
            canceledVisibleFinalParent = job.visibleFinalParent;
            canceled = true;
        } else {
            queueJobLocked(std::move(job));
            queued = true;
        }
    }
    if (canceled) {
        inFlight_.fetch_sub(1, std::memory_order_release);
        canceled_.fetch_add(1, std::memory_order_relaxed);
    }
    if (canceledUrgent) urgentRefinementInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    if (canceledVisibleFinalParent)
        visibleFinalParentInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    if (queued || canceled) jobCv_.notify_all();
}

size_t FarTerrainScheduler::refreshParkedBaseJob(FarTerrainKey key, size_t maximumWork) {
    BaseAuthorityWaitSet waitingOn;
    FarTerrainAuthorityQuality authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    worldgen::learned::AuthorityRequestPriority authorityPriority =
        worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW;
    uint64_t protectedHandoffEpoch = 0;
    {
        std::lock_guard lock(jobMutex_);
        const auto parked = parkedBaseJobs_.find(key);
        if (parked == parkedBaseJobs_.end()) return 0;
        waitingOn = parked->second.waitingOn;
        authorityQuality = parked->second.job.authorityQuality;
        authorityPriority = parked->second.job.authorityPriority;
        protectedHandoffEpoch = parked->second.job.protectedHandoffEpoch;
    }
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context =
        authorityQuality == FarTerrainAuthorityQuality::PREVIEW ? previewGenerationContext_
                                                                : generationContext_;
    if (!context) return 0;
    const worldgen::learned::ProtectedHandoffEpoch protectedEpoch{
        authorityPriority == worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF
            ? protectedHandoffEpoch
            : uint64_t{0}};
    size_t work = 0;
    for (const worldgen::learned::TerrainPageCoordinate coordinate : waitingOn.pages) {
        if (work == maximumWork) return work;
        const auto prepared =
            context->requestAuthorityPage(coordinate, authorityPriority, protectedEpoch);
        ++work;
        if (prepared.status() == worldgen::learned::AuthorityStatus::FAILED) {
            context->latchFailure(
                prepared.failure()
                    ? *prepared.failure()
                    : worldgen::learned::GenerationFailure{
                          .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                          .message = "Parked base terrain authority request failed",
                          .retriable = true});
            cancelParkedBaseJob(key);
            return work;
        }
        if (prepared.status() == worldgen::learned::AuthorityStatus::READY)
            wakeParkedBaseJobsForReadyPage(coordinate);
    }
    for (const FarTerrainNativeHydrologyDependency& dependency : waitingOn.nativeHydrology) {
        if (work == maximumWork) return work;
        const auto prepared = context->queryTransientFinalNativeGrid(
            dependency.finalTerrainRegion, authorityPriority, protectedEpoch);
        ++work;
        if (prepared.status() == worldgen::learned::AuthorityStatus::FAILED) {
            context->latchFailure(
                prepared.failure()
                    ? *prepared.failure()
                    : worldgen::learned::GenerationFailure{
                          .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                          .message = "Parked base native hydrology input request failed",
                          .retriable = true});
            cancelParkedBaseJob(key);
            return work;
        }
        const bool ready = prepared.status() == worldgen::learned::AuthorityStatus::READY;
        if (!ready) continue;
        {
            std::lock_guard lock(jobMutex_);
            const auto parked = parkedBaseJobs_.find(key);
            if (parked != parkedBaseJobs_.end()) {
                std::erase(parked->second.waitingOn.nativeHydrology, dependency);
            }
        }
        wakeParkedBaseJobIfReady(key);
    }
    if (waitingOn.minimumAuthorityCompletion && work < maximumWork) {
        ++work;
        const uint64_t completion = context->metrics().authorityCache.completionGeneration;
        if (completion >= *waitingOn.minimumAuthorityCompletion) {
            bool resumed = false;
            {
                std::lock_guard lock(jobMutex_);
                const auto parked = parkedBaseJobs_.find(key);
                if (parked != parkedBaseJobs_.end() &&
                    parked->second.waitingOn.minimumAuthorityCompletion &&
                    *parked->second.waitingOn.minimumAuthorityCompletion <= completion) {
                    parked->second.waitingOn.minimumAuthorityCompletion.reset();
                    resumed = true;
                }
            }
            if (resumed) {
                authorityCompletionResumes_.fetch_add(1, std::memory_order_relaxed);
                wakeParkedBaseJobIfReady(key);
            }
        }
    }
    return work;
}

void FarTerrainScheduler::cancelParkedBaseJob(FarTerrainKey key) {
    bool canceled = false;
    bool canceledUrgent = false;
    bool canceledVisibleFinalParent = false;
    {
        std::lock_guard lock(jobMutex_);
        const auto parked = parkedBaseJobs_.find(key);
        if (parked == parkedBaseJobs_.end()) return;
        const Job job = parked->second.job;
        removeParkedBaseWaitersLocked(parked->second);
        parkedBaseJobs_.erase(parked);
        parkedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        const auto active = activeKeys_.find(key);
        if (active != activeKeys_.end() && active->second == job.epoch) activeKeys_.erase(active);
        canceledUrgent = job.urgentRefinement;
        canceledVisibleFinalParent = job.visibleFinalParent;
        canceled = true;
    }
    if (canceled) {
        if (canceledUrgent) urgentRefinementInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
        if (canceledVisibleFinalParent)
            visibleFinalParentInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
        inFlight_.fetch_sub(1, std::memory_order_release);
        canceled_.fetch_add(1, std::memory_order_relaxed);
        jobCv_.notify_all();
    }
}

bool FarTerrainScheduler::enqueueCanopyInternal(FarTerrainKey key, uint64_t epoch,
                                                uint32_t viewPriority,
                                                FarTerrainAuthorityQuality groundingQuality) {
    const std::shared_ptr<const FarCanopyAttachment> cached = findCachedCanopy(key);
    if (!running_.load(std::memory_order_acquire) ||
        (cached && cached->authorityQuality == FarTerrainAuthorityQuality::FINAL &&
         cached->groundingQuality == groundingQuality)) {
        return false;
    }
    const FarTerrainAuthorityQuality ecologyQuality =
        previewSource_.sample && (!cached || cached->groundingQuality != groundingQuality)
            ? FarTerrainAuthorityQuality::PREVIEW
            : FarTerrainAuthorityQuality::FINAL;
    {
        std::lock_guard lock(jobMutex_);
        if (wantedMembership_ && !wantedMembership_->keys.contains(key)) return false;
    }
    {
        std::lock_guard lock(canopyJobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            epoch != epoch_.load(std::memory_order_relaxed)) {
            return false;
        }
        if (const auto current = canopyViewPriorities_.find(key);
            current != canopyViewPriorities_.end()) {
            viewPriority = current->second;
        }
        const auto active = activeCanopyKeys_.find(key);
        if (active != activeCanopyKeys_.end() && active->second == epoch) {
            // Preserve the job for the currently displayed surface until it
            // has published provisional vegetation. A simultaneous FINAL
            // terrain request becomes a followup instead of turning the first
            // visible attachment into more cold authority debt.
            const auto upgrade = [&](CanopyJob& job) {
                job.viewPriority = std::min(job.viewPriority, viewPriority);
                if (farTerrainAuthoritySatisfies(groundingQuality, job.groundingQuality))
                    job.groundingQuality = groundingQuality;
                if (farTerrainAuthoritySatisfies(ecologyQuality, job.ecologyQuality))
                    job.ecologyQuality = ecologyQuality;
            };
            const auto queueFollowup = [&] {
                auto [followup, inserted] = canopyFollowupJobs_.try_emplace(
                    key, CanopyJob{.key = key,
                                   .epoch = epoch,
                                   .viewPriority = viewPriority,
                                   .groundingQuality = groundingQuality,
                                   .ecologyQuality = ecologyQuality});
                if (!inserted) upgrade(followup->second);
            };
            if (auto parked = parkedCanopyJobs_.find(key);
                parked != parkedCanopyJobs_.end() && parked->second.provisionalPublished) {
                upgrade(parked->second);
            } else {
                queueFollowup();
            }
            return false;
        }
        const CanopyJob job{.key = key,
                            .epoch = epoch,
                            .viewPriority = viewPriority,
                            .groundingQuality = groundingQuality,
                            .ecologyQuality = ecologyQuality};

        enum class Replacement : uint8_t {
            NONE,
            QUEUED,
            PARKED,
        };
        Replacement replacement = Replacement::NONE;
        if (canopyInFlight_.load(std::memory_order_relaxed) >= limits_.maxCanopyPending) {
            // Parked authority work still owns a pending slot. Compare it with
            // the least important queued attachment so a newly visible tile
            // can replace stale distant work regardless of which state owns
            // the slot. Running work remains non-preemptive.
            CanopyJob displaced;
            auto parkedVictim = parkedCanopyJobs_.end();
            if (!canopyJobs_.empty()) {
                displaced = canopyJobs_.back();
                replacement = Replacement::QUEUED;
            }
            for (auto candidate = parkedCanopyJobs_.begin(); candidate != parkedCanopyJobs_.end();
                 ++candidate) {
                if (replacement == Replacement::NONE ||
                    canopyJobBefore(displaced, candidate->second)) {
                    displaced = candidate->second;
                    parkedVictim = candidate;
                    replacement = Replacement::PARKED;
                }
            }
            if (replacement == Replacement::NONE || !canopyJobBefore(job, displaced)) return false;
            if (replacement == Replacement::QUEUED) {
                canopyJobs_.pop_back();
            } else {
                parkedCanopyJobs_.erase(parkedVictim);
                parkedCanopyCount_.fetch_sub(1, std::memory_order_relaxed);
                queuedCanopyCount_.fetch_add(1, std::memory_order_relaxed);
            }
            const auto displacedActive = activeCanopyKeys_.find(displaced.key);
            if (displacedActive != activeCanopyKeys_.end() &&
                displacedActive->second == displaced.epoch) {
                activeCanopyKeys_.erase(displacedActive);
            }
            canopyFollowupJobs_.erase(displaced.key);
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
        }

        activeCanopyKeys_[key] = epoch;
        const auto insertion =
            std::find_if(canopyJobs_.begin(), canopyJobs_.end(),
                         [&](const CanopyJob& queued) { return canopyJobBefore(job, queued); });
        canopyJobs_.insert(insertion, job);
        if (replacement == Replacement::NONE) {
            canopyInFlight_.fetch_add(1, std::memory_order_relaxed);
            queuedCanopyCount_.fetch_add(1, std::memory_order_relaxed);
        }
        canopySubmitted_.fetch_add(1, std::memory_order_relaxed);
    }
    canopyJobCv_.notify_one();
    return true;
}

bool FarTerrainScheduler::parkCanopyJob(CanopyJob job) {
    {
        std::scoped_lock lock(jobMutex_, canopyJobMutex_);
        if (!running_.load(std::memory_order_relaxed) ||
            job.epoch != epoch_.load(std::memory_order_relaxed) ||
            (wantedMembership_ && !wantedMembership_->keys.contains(job.key))) {
            return false;
        }
        const auto active = activeCanopyKeys_.find(job.key);
        if (active == activeCanopyKeys_.end() || active->second != job.epoch ||
            parkedCanopyJobs_.contains(job.key)) {
            return false;
        }
        // Once the currently displayed surface has a provisional attachment,
        // a queued promotion may retarget only the parked FINAL retry. The
        // resident provisional geometry remains untouched until its successor
        // has completed and uploaded.
        if (job.provisionalPublished) {
            if (auto followup = canopyFollowupJobs_.find(job.key);
                followup != canopyFollowupJobs_.end() && followup->second.epoch == job.epoch &&
                farTerrainAuthoritySatisfies(followup->second.groundingQuality,
                                             job.groundingQuality)) {
                job.viewPriority = followup->second.viewPriority;
                job.groundingQuality = followup->second.groundingQuality;
                if (farTerrainAuthoritySatisfies(followup->second.ecologyQuality,
                                                 job.ecologyQuality)) {
                    job.ecologyQuality = followup->second.ecologyQuality;
                }
                canopyFollowupJobs_.erase(followup);
            }
        }
        const bool previewEcology = job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW;
        parkedCanopyJobs_.emplace(job.key, std::move(job));
        parkedCanopyCount_.fetch_add(1, std::memory_order_relaxed);
        activeCanopyWorkerCount_.fetch_sub(1, std::memory_order_relaxed);
        if (previewEcology) {
            if (activePreviewCanopyWorkerCount_ > 0) --activePreviewCanopyWorkerCount_;
        } else if (activeFinalCanopyWorkerCount_ > 0) {
            --activeFinalCanopyWorkerCount_;
        }
    }
    canopyJobCv_.notify_all();
    return true;
}

bool FarTerrainScheduler::hasSubmissionCapacity() const noexcept {
    return running_.load(std::memory_order_acquire) &&
           inFlight_.load(std::memory_order_relaxed) <
               farTerrainNonurgentBaseAdmissionLimit(limits_.maxPending);
}

bool FarTerrainScheduler::hasUrgentRefinementCapacity() const noexcept {
    if (!running_.load(std::memory_order_acquire)) return false;
    if (inFlight_.load(std::memory_order_relaxed) < limits_.maxPending &&
        urgentRefinementInFlightCount_.load(std::memory_order_relaxed) <
            FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT) {
        return true;
    }
    // The exact incoming rank is checked under jobMutex_ by enqueueInternal.
    // A conservative positive result lets camera-near callers attempt to
    // displace lower-ranked queued or parked optional work at a full cap.
    return queuedUrgentRefinementCount_.load(std::memory_order_relaxed) != 0 ||
           queuedRefinementCount_.load(std::memory_order_relaxed) != 0 ||
           parkedBaseCount_.load(std::memory_order_relaxed) != 0;
}

void FarTerrainScheduler::setWorkerBudget(size_t budget) {
    // Zero is a real pause state used when a stronger owner needs every
    // available construction lane. Queued jobs retain their bounded slots and
    // resume in priority order when the budget opens again.
    const size_t clamped = std::min(budget, WORKER_COUNT);
    {
        std::lock_guard lock(jobMutex_);
        if (workerBudget_ == clamped) return;
        workerBudget_ = clamped;
        workerBudgetSnapshot_.store(clamped, std::memory_order_release);
    }
    jobCv_.notify_all();
}

void FarTerrainScheduler::setNearFirstWorkEnabled(bool enabled) {
    {
        std::lock_guard lock(jobMutex_);
        if (nearFirstWorkEnabled_ == enabled) return;
        nearFirstWorkEnabled_ = enabled;
    }
    jobCv_.notify_all();
}

void FarTerrainScheduler::setCanopyWorkerBudget(size_t budget) {
    const size_t clamped = std::min(budget, CANOPY_WORKER_COUNT);
    {
        std::lock_guard lock(canopyJobMutex_);
        if (canopyWorkerBudget_ == clamped) return;
        canopyWorkerBudget_ = clamped;
    }
    canopyJobCv_.notify_all();
}

void FarTerrainScheduler::setCoarseAuthorityPrefetchPages(
    std::vector<worldgen::learned::TerrainPageCoordinate> pages) {
    if (coarseAuthorityPrefetchPages_ == pages) return;
    coarseAuthorityPrefetchPages_ = std::move(pages);
    coarseAuthorityPrefetchCursor_ = 0;
}

void FarTerrainScheduler::pumpCoarseAuthorityPrefetch() {
    if (!running_.load(std::memory_order_acquire) || !previewGenerationContext_ ||
        previewGenerationContext_->failure()) {
        return;
    }

    // A complete authority queue contains at most sixty-four pages. Polling
    // submitted pages before trying a replacement horizon is essential: a
    // completed failed flight remains in the cache's single-flight table
    // until a caller observes it. Without this pass, a queue filled just
    // before a terminal failure could reject every later request forever.
    // The shared work budget keeps this render-thread path bounded even when
    // a camera move replaces the entire horizon.
    static_assert(worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS >
                  FAR_TERRAIN_RESERVED_FINAL_AUTHORITY_REQUESTS);
    constexpr size_t MAX_PREFETCH_WORK_PER_PUMP =
        worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS -
        FAR_TERRAIN_RESERVED_FINAL_AUTHORITY_REQUESTS;
    size_t work = 0;
    for (size_t index = 0; index < coarseAuthorityPrefetchOutstandingPages_.size() &&
                           work < MAX_PREFETCH_WORK_PER_PUMP;) {
        const worldgen::learned::TerrainPageCoordinate coordinate =
            coarseAuthorityPrefetchOutstandingPages_[index];
        const auto result = previewGenerationContext_->requestAuthorityPage(
            coordinate, worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW);
        ++work;
        if (result.status() == worldgen::learned::AuthorityStatus::FAILED) {
            if (result.failure()) previewGenerationContext_->latchFailure(*result.failure());
            return;
        }
        if (result.status() == worldgen::learned::AuthorityStatus::READY) {
            coarseAuthorityPrefetchOutstandingPages_[index] =
                coarseAuthorityPrefetchOutstandingPages_.back();
            coarseAuthorityPrefetchOutstandingPages_.pop_back();
            wakeParkedBaseJobsForReadyPage(coordinate);
            continue;
        }
        ++index;
    }

    while (coarseAuthorityPrefetchCursor_ < coarseAuthorityPrefetchPages_.size() &&
           work < MAX_PREFETCH_WORK_PER_PUMP) {
        const auto coordinate = coarseAuthorityPrefetchPages_[coarseAuthorityPrefetchCursor_];
        ++work;
        if (std::find(coarseAuthorityPrefetchOutstandingPages_.begin(),
                      coarseAuthorityPrefetchOutstandingPages_.end(),
                      coordinate) != coarseAuthorityPrefetchOutstandingPages_.end()) {
            ++coarseAuthorityPrefetchCursor_;
            continue;
        }

        const auto result = previewGenerationContext_->requestAuthorityPage(
            coordinate, worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW);
        if (result.status() == worldgen::learned::AuthorityStatus::FAILED) {
            if (result.failure()) previewGenerationContext_->latchFailure(*result.failure());
            return;
        }
        if (result.status() == worldgen::learned::AuthorityStatus::DEFERRED && result.failure() &&
            result.failure()->code == worldgen::learned::GenerationFailureCode::QUEUE_FULL) {
            return;
        }
        if (result.status() == worldgen::learned::AuthorityStatus::DEFERRED)
            coarseAuthorityPrefetchOutstandingPages_.push_back(coordinate);
        else if (result.status() == worldgen::learned::AuthorityStatus::READY)
            wakeParkedBaseJobsForReadyPage(coordinate);
        ++coarseAuthorityPrefetchCursor_;
    }
}

void FarTerrainScheduler::setSpeculativeAuthorityPrefetchPages(
    std::vector<worldgen::learned::TerrainPageCoordinate> pages) {
    std::vector<worldgen::learned::TerrainPageCoordinate> canonical;
    canonical.reserve(std::min(pages.size(), FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES));
    for (const worldgen::learned::TerrainPageCoordinate coordinate : pages) {
        if (std::find(coarseAuthorityPrefetchPages_.begin(), coarseAuthorityPrefetchPages_.end(),
                      coordinate) != coarseAuthorityPrefetchPages_.end() ||
            std::find(coarseAuthorityPrefetchOutstandingPages_.begin(),
                      coarseAuthorityPrefetchOutstandingPages_.end(),
                      coordinate) != coarseAuthorityPrefetchOutstandingPages_.end() ||
            std::find(canonical.begin(), canonical.end(), coordinate) != canonical.end()) {
            continue;
        }
        canonical.push_back(coordinate);
        if (canonical.size() == FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES) break;
    }
    if (speculativeAuthorityPrefetchPages_ == canonical) return;
    speculativeAuthorityPrefetchPages_ = std::move(canonical);
    speculativeAuthorityPrefetchCursor_ = 0;
}

void FarTerrainScheduler::pumpSpeculativeAuthorityPrefetch() {
    if (!running_.load(std::memory_order_acquire) || !previewGenerationContext_ ||
        previewGenerationContext_->failure()) {
        return;
    }

    using worldgen::learned::AuthorityRequestPriority;
    using worldgen::learned::AuthorityStatus;
    using worldgen::learned::GenerationFailureCode;
    constexpr size_t MAX_WORK_PER_PUMP = FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES;
    size_t work = 0;

    // Always observe old speculative flights, even after a view replacement.
    // This releases completed single-flight records and surfaces a terminal
    // backend failure without admitting any new hint ahead of visible work.
    for (size_t index = 0; index < speculativeAuthorityPrefetchOutstandingPages_.size() &&
                           work < MAX_WORK_PER_PUMP;) {
        const worldgen::learned::TerrainPageCoordinate coordinate =
            speculativeAuthorityPrefetchOutstandingPages_[index];
        const auto result = previewGenerationContext_->requestAuthorityPage(
            coordinate, AuthorityRequestPriority::SPECULATIVE_PREFETCH);
        ++work;
        if (result.status() == AuthorityStatus::FAILED) {
            if (result.failure()) previewGenerationContext_->latchFailure(*result.failure());
            return;
        }
        if (result.status() == AuthorityStatus::READY) {
            speculativeAuthorityPrefetchOutstandingPages_[index] =
                speculativeAuthorityPrefetchOutstandingPages_.back();
            speculativeAuthorityPrefetchOutstandingPages_.pop_back();
            continue;
        }
        ++index;
    }

    if (coarseAuthorityPrefetchCursor_ < coarseAuthorityPrefetchPages_.size() ||
        !coarseAuthorityPrefetchOutstandingPages_.empty()) {
        return;
    }

    while (speculativeAuthorityPrefetchCursor_ < speculativeAuthorityPrefetchPages_.size() &&
           work < MAX_WORK_PER_PUMP) {
        const worldgen::learned::TerrainPageCoordinate coordinate =
            speculativeAuthorityPrefetchPages_[speculativeAuthorityPrefetchCursor_];
        if (std::find(speculativeAuthorityPrefetchOutstandingPages_.begin(),
                      speculativeAuthorityPrefetchOutstandingPages_.end(),
                      coordinate) != speculativeAuthorityPrefetchOutstandingPages_.end()) {
            ++speculativeAuthorityPrefetchCursor_;
            continue;
        }
        const auto result = previewGenerationContext_->requestAuthorityPage(
            coordinate, AuthorityRequestPriority::SPECULATIVE_PREFETCH);
        ++work;
        if (result.status() == AuthorityStatus::FAILED) {
            if (result.failure()) previewGenerationContext_->latchFailure(*result.failure());
            return;
        }
        if (result.status() == AuthorityStatus::DEFERRED && result.failure() &&
            result.failure()->code == GenerationFailureCode::QUEUE_FULL) {
            return;
        }
        if (result.status() == AuthorityStatus::DEFERRED)
            speculativeAuthorityPrefetchOutstandingPages_.push_back(coordinate);
        ++speculativeAuthorityPrefetchCursor_;
    }
}

void FarTerrainScheduler::pumpFinalBaseAuthority() {
    if (!running_.load(std::memory_order_acquire) || !generationContext_ ||
        !finalStreamingWorkEnabled_.load(std::memory_order_acquire) ||
        generationContext_->failure()) {
        return;
    }
    std::array<Job, FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT> parkedJobs{};
    size_t parkedCount = 0;
    {
        std::lock_guard lock(jobMutex_);
        for (const auto& [key, parked] : parkedBaseJobs_) {
            if (parked.job.authorityQuality != FarTerrainAuthorityQuality::FINAL) continue;
            parkedJobs[parkedCount++] = parked.job;
            if (parkedCount == parkedJobs.size()) break;
        }
    }
    std::sort(parkedJobs.begin(), parkedJobs.begin() + static_cast<std::ptrdiff_t>(parkedCount),
              [](const Job& first, const Job& second) {
                  if (first.authorityPriority != second.authorityPriority)
                      return first.authorityPriority < second.authorityPriority;
                  return farTerrainSubmissionBefore(first.key, first.viewPriority, second.key,
                                                    second.viewPriority);
              });
    size_t work = 0;
    for (const Job& job : std::span(parkedJobs).first(parkedCount)) {
        if (work == FINAL_BASE_AUTHORITY_POLLS_PER_PUMP || generationContext_->failure()) break;
        work += refreshParkedBaseJob(job.key, FINAL_BASE_AUTHORITY_POLLS_PER_PUMP - work);
    }

    // A native-hydrology build can publish shared reconciliation state and
    // still surface DEFERRED after the last learned rectangle completed. In
    // that case minimumAuthorityCompletion points one past a generation that
    // can no longer advance. Retry one parked parent only when every possible
    // producer is quiescent. The per-job spill-summary bound converts a true
    // no-progress defect into the normal repair UI instead of an infinite
    // Entry Horizon wait.
    const worldgen::learned::TerrainAuthorityCacheMetrics authorityMetrics =
        generationContext_->metrics().authorityCache;
    const worldgen::NativeHydrologyCacheMetrics hydrologyMetrics =
        generationContext_->nativeHydrologyRouter()->cacheMetrics();
    const bool producersQuiescent =
        authorityMetrics.activeBuilds == 0 && authorityMetrics.queuedBuilds == 0 &&
        authorityMetrics.activePublications == 0 && authorityMetrics.queuedPublications == 0 &&
        hydrologyMetrics.activeBuilds == 0 &&
        activeWorkerCountSnapshot_.load(std::memory_order_relaxed) == 0 &&
        queuedBaseCount_.load(std::memory_order_relaxed) == 0 &&
        queuedRefinementCount_.load(std::memory_order_relaxed) == 0;
    if (!producersQuiescent || generationContext_->failure()) return;

    std::optional<FarTerrainKey> resumeKey;
    std::optional<FarTerrainKey> exhaustedKey;
    {
        std::lock_guard lock(jobMutex_);
        for (const Job& ordered : std::span(parkedJobs).first(parkedCount)) {
            const auto parked = parkedBaseJobs_.find(ordered.key);
            if (parked == parkedBaseJobs_.end() || !parked->second.waitingOn.pages.empty() ||
                !parked->second.waitingOn.nativeHydrology.empty() ||
                !parked->second.waitingOn.minimumAuthorityCompletion) {
                continue;
            }
            if (parked->second.job.quiescentAuthorityRetries >=
                worldgen::NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES) {
                exhaustedKey = ordered.key;
                break;
            }
            ++parked->second.job.quiescentAuthorityRetries;
            parked->second.waitingOn.minimumAuthorityCompletion.reset();
            resumeKey = ordered.key;
            break;
        }
    }
    if (exhaustedKey) {
        generationContext_->latchFailure({
            .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
            .message = "Far-terrain FINAL parent remained deferred after bounded hydrology "
                       "reconciliation at tile " +
                       std::to_string(exhaustedKey->tileX) + "," +
                       std::to_string(exhaustedKey->tileZ),
            .retriable = true,
        });
        cancelParkedBaseJob(*exhaustedKey);
        return;
    }
    if (resumeKey) {
        quiescentAuthorityResumes_.fetch_add(1, std::memory_order_relaxed);
        wakeParkedBaseJobIfReady(*resumeKey);
    }
}

void FarTerrainScheduler::pumpCanopyAuthority() {
    if (!running_.load(std::memory_order_acquire) || !generationContext_ ||
        generationContext_->failure()) {
        return;
    }
    const uint64_t completion = generationContext_->metrics().authorityCache.completionGeneration;
    size_t resumed = 0;
    {
        std::lock_guard lock(canopyJobMutex_);
        for (auto parked = parkedCanopyJobs_.begin(); parked != parkedCanopyJobs_.end();) {
            if (parked->second.epoch != epoch_.load(std::memory_order_relaxed) ||
                completion < parked->second.minimumAuthorityCompletion) {
                ++parked;
                continue;
            }
            CanopyJob job = std::move(parked->second);
            job.minimumAuthorityCompletion = 0;
            parked = parkedCanopyJobs_.erase(parked);
            const auto insertion =
                std::find_if(canopyJobs_.begin(), canopyJobs_.end(),
                             [&](const CanopyJob& queued) { return canopyJobBefore(job, queued); });
            canopyJobs_.insert(insertion, std::move(job));
            ++resumed;
        }
        if (resumed != 0) {
            parkedCanopyCount_.fetch_sub(resumed, std::memory_order_relaxed);
            queuedCanopyCount_.fetch_add(resumed, std::memory_order_relaxed);
            canopyAuthorityCompletionResumes_.fetch_add(resumed, std::memory_order_relaxed);
        }
    }
    if (resumed != 0) canopyJobCv_.notify_all();
}

bool FarTerrainScheduler::retainWanted(
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash>& wanted,
    const std::vector<FarTerrainKey>& nearestFirst, std::span<const FarTerrainKey> criticalKeys) {
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> canonicalCriticalKeys;
    canonicalCriticalKeys.reserve(criticalKeys.size());
    std::vector<FarTerrainKey> canonicalCriticalNearestFirst;
    canonicalCriticalNearestFirst.reserve(criticalKeys.size());
    for (const FarTerrainKey key : criticalKeys) {
        if (wanted.contains(key) && canonicalCriticalKeys.insert(key).second)
            canonicalCriticalNearestFirst.push_back(key);
    }
    {
        std::lock_guard lock(jobMutex_);
        if (wantedMembership_ && wantedMembership_->keys == wanted &&
            wantedMembership_->nearestFirst == nearestFirst &&
            criticalNearestFirst_ == canonicalCriticalNearestFirst) {
            wantedNoops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    auto membership = std::make_shared<ResidencyMembership>();
    membership->keys = wanted;
    std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> canonicalCriticalPriorities;
    canonicalCriticalPriorities.reserve(canonicalCriticalNearestFirst.size());
    for (size_t index = 0; index < canonicalCriticalNearestFirst.size(); ++index) {
        canonicalCriticalPriorities.emplace(
            canonicalCriticalNearestFirst[index],
            static_cast<uint32_t>(std::min(index, static_cast<size_t>(UINT32_MAX))));
    }
    membership->nearestFirst = nearestFirst;
    membership->priorities.reserve(nearestFirst.size());
    for (size_t index = 0; index < nearestFirst.size(); ++index) {
        const FarTerrainKey key = nearestFirst[index];
        if (!wanted.contains(key)) continue;
        membership->priorities.try_emplace(
            key, static_cast<uint32_t>(std::min(index, static_cast<size_t>(UINT32_MAX))));
    }
    std::unordered_map<ColumnPos, uint32_t>& coordinatePriorities =
        membership->coordinatePriorities;
    coordinatePriorities.reserve(wanted.size());
    uint32_t nextCoordinatePriority = 0;
    for (const FarTerrainKey key : nearestFirst) {
        if (!wanted.contains(key)) continue;
        const ColumnPos coordinate{key.tileX, key.tileZ};
        if (coordinatePriorities.try_emplace(coordinate, nextCoordinatePriority).second &&
            nextCoordinatePriority != std::numeric_limits<uint32_t>::max()) {
            ++nextCoordinatePriority;
        }
    }
    std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> canopyViewPriorities;
    canopyViewPriorities.reserve(wanted.size());
    for (const FarTerrainKey key : wanted) {
        if (const auto priority = coordinatePriorities.find({key.tileX, key.tileZ});
            priority != coordinatePriorities.end()) {
            canopyViewPriorities.emplace(key, priority->second);
        }
    }
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    size_t removedVisibleFinalParents = 0;
    size_t parkedRemoved = 0;
    size_t parkedUrgentRemoved = 0;
    size_t parkedVisibleFinalParentsRemoved = 0;
    size_t followupUrgentReverted = 0;
    size_t followupVisibleFinalParentReverted = 0;
    size_t followupsRemoved = 0;
    size_t canopyRemoved = 0;
    size_t parkedCanopyRemoved = 0;
    {
        // A worker cannot take the old canopy front after the new membership
        // has become visible. Publish membership and refresh optional work as
        // one scheduler transaction.
        std::scoped_lock lock(jobMutex_, canopyJobMutex_);
        if (wantedMembership_ && wantedMembership_->keys == wanted &&
            wantedMembership_->nearestFirst == nearestFirst &&
            criticalNearestFirst_ == canonicalCriticalNearestFirst) {
            wantedNoops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        membership->revision = ++nextWantedRevision_;
        if (wantedMembership_) retiredMemberships_.push_back(std::move(wantedMembership_));
        wantedMembership_ = membership;
        criticalWantedKeys_ = std::move(canonicalCriticalKeys);
        criticalNearestFirst_ = std::move(canonicalCriticalNearestFirst);
        criticalPriorities_ = std::move(canonicalCriticalPriorities);
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
            if (job.visibleFinalParent) ++removedVisibleFinalParents;
            ++removed;
            return true;
        });
        for (auto parked = parkedBaseJobs_.begin(); parked != parkedBaseJobs_.end();) {
            if (membership->keys.contains(parked->first)) {
                ++parked;
                continue;
            }
            const Job job = parked->second.job;
            removeParkedBaseWaitersLocked(parked->second);
            const auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch)
                activeKeys_.erase(active);
            if (job.urgentRefinement) ++parkedUrgentRemoved;
            if (job.visibleFinalParent) ++parkedVisibleFinalParentsRemoved;
            parked = parkedBaseJobs_.erase(parked);
            ++parkedRemoved;
        }
        for (auto followup = terrainFollowupJobs_.begin();
             followup != terrainFollowupJobs_.end();) {
            if (membership->keys.contains(followup->first)) {
                ++followup;
                continue;
            }
            const auto worker = activeWorkerJobs_.find(followup->first);
            if (worker != activeWorkerJobs_.end()) {
                if (followup->second.urgentRefinement && !worker->second.urgentRefinement)
                    ++followupUrgentReverted;
                if (followup->second.visibleFinalParent && !worker->second.visibleFinalParent)
                    ++followupVisibleFinalParentReverted;
            }
            followup = terrainFollowupJobs_.erase(followup);
            ++followupsRemoved;
        }
        const auto refreshTerrainPriority = [&](Job& job) {
            job.cameraNearCritical = criticalWantedKeys_.contains(job.key);
            if (const auto priority = coordinatePriorities.find({job.key.tileX, job.key.tileZ});
                priority != coordinatePriorities.end()) {
                job.viewPriority = priority->second;
            }
        };
        for (Job& job : jobs_)
            refreshTerrainPriority(job);
        std::sort(jobs_.begin(), jobs_.end(), jobBefore);
        for (auto& entry : parkedBaseJobs_)
            refreshTerrainPriority(entry.second.job);
        for (auto& entry : terrainFollowupJobs_)
            refreshTerrainPriority(entry.second);
        for (auto& entry : activeWorkerJobs_)
            refreshTerrainPriority(entry.second);
        std::erase_if(canopyJobs_, [&](const CanopyJob& job) {
            if (membership->keys.contains(job.key)) return false;
            const auto active = activeCanopyKeys_.find(job.key);
            if (active != activeCanopyKeys_.end() && active->second == job.epoch)
                activeCanopyKeys_.erase(active);
            ++canopyRemoved;
            return true;
        });
        std::erase_if(parkedCanopyJobs_, [&](const auto& entry) {
            if (membership->keys.contains(entry.first)) return false;
            const auto active = activeCanopyKeys_.find(entry.first);
            if (active != activeCanopyKeys_.end() && active->second == entry.second.epoch)
                activeCanopyKeys_.erase(active);
            ++parkedCanopyRemoved;
            return true;
        });
        std::erase_if(canopyFollowupJobs_,
                      [&](const auto& entry) { return !membership->keys.contains(entry.first); });
        canopyViewPriorities_ = std::move(canopyViewPriorities);
        const auto refreshPriority = [&](CanopyJob& job) {
            if (const auto priority = canopyViewPriorities_.find(job.key);
                priority != canopyViewPriorities_.end()) {
                job.viewPriority = priority->second;
            }
        };
        for (CanopyJob& job : canopyJobs_)
            refreshPriority(job);
        std::sort(canopyJobs_.begin(), canopyJobs_.end(), canopyJobBefore);
        for (auto& entry : parkedCanopyJobs_)
            refreshPriority(entry.second);
        for (auto& entry : canopyFollowupJobs_)
            refreshPriority(entry.second);
    }
    wantedUpdates_.fetch_add(1, std::memory_order_relaxed);
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
    }
    if (removedUrgentRefinement + parkedUrgentRemoved != 0)
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement + parkedUrgentRemoved,
                                                 std::memory_order_relaxed);
    if (followupUrgentReverted != 0)
        urgentRefinementInFlightCount_.fetch_sub(followupUrgentReverted, std::memory_order_relaxed);
    if (removedVisibleFinalParents + parkedVisibleFinalParentsRemoved != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(removedVisibleFinalParents +
                                                       parkedVisibleFinalParentsRemoved,
                                                   std::memory_order_relaxed);
    }
    if (followupVisibleFinalParentReverted != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(followupVisibleFinalParentReverted,
                                                   std::memory_order_relaxed);
    }
    if (followupsRemoved != 0)
        terrainFollowupCount_.fetch_sub(followupsRemoved, std::memory_order_relaxed);
    if (parkedRemoved > 0) parkedBaseCount_.fetch_sub(parkedRemoved, std::memory_order_relaxed);
    const size_t totalRemoved = removed + parkedRemoved;
    if (totalRemoved > 0) {
        inFlight_.fetch_sub(totalRemoved, std::memory_order_relaxed);
        canceled_.fetch_add(totalRemoved, std::memory_order_relaxed);
    }
    {
        // Completion retention follows the newly published camera order in
        // the same lock order used by storeCompleted. A camera move can
        // therefore promote a ready near mesh before later distant results
        // reach the bounded render-thread handoff queue.
        std::lock_guard jobLock(jobMutex_);
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
        std::stable_sort(completed_.begin(), completed_.end(),
                         [this](const FarTerrainResult& first, const FarTerrainResult& second) {
                             return completedResultBeforeLocked(first, second);
                         });
    }
    if (canopyRemoved + parkedCanopyRemoved > 0) {
        queuedCanopyCount_.fetch_sub(canopyRemoved, std::memory_order_relaxed);
        parkedCanopyCount_.fetch_sub(parkedCanopyRemoved, std::memory_order_relaxed);
        canopyInFlight_.fetch_sub(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
        canopyCanceled_.fetch_add(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(canopyCompletedMutex_);
        const size_t before = canopyCompleted_.size();
        std::erase_if(canopyCompleted_, [&](const FarCanopyResult& result) {
            return !membership->keys.contains(result.key);
        });
        completedCanopyCount_.fetch_sub(before - canopyCompleted_.size(),
                                        std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(canopyCacheMutex_);
        canopyCacheMembership_ = membership;
    }
    maintenancePendingSnapshot_.store(1, std::memory_order_release);
    jobCv_.notify_one();
    canopyJobCv_.notify_all();
    return true;
}

bool FarTerrainScheduler::refreshCriticalPriorities(std::span<const FarTerrainKey> criticalKeys) {
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> canonicalKeys;
    std::vector<FarTerrainKey> canonicalNearestFirst;
    canonicalKeys.reserve(criticalKeys.size());
    canonicalNearestFirst.reserve(criticalKeys.size());

    {
        std::lock_guard lock(jobMutex_);
        for (const FarTerrainKey key : criticalKeys) {
            if (wantedMembership_ && !wantedMembership_->keys.contains(key)) continue;
            if (canonicalKeys.insert(key).second) canonicalNearestFirst.push_back(key);
        }
        if (criticalNearestFirst_ == canonicalNearestFirst) {
            criticalPriorityNoops_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::unordered_map<FarTerrainKey, uint32_t, FarTerrainKeyHash> priorities;
        priorities.reserve(canonicalNearestFirst.size());
        std::unordered_map<ColumnPos, uint32_t> coordinatePriorities;
        coordinatePriorities.reserve(canonicalNearestFirst.size());
        uint32_t nextCoordinatePriority = 0;
        for (size_t index = 0; index < canonicalNearestFirst.size(); ++index) {
            const FarTerrainKey key = canonicalNearestFirst[index];
            priorities.emplace(
                key, static_cast<uint32_t>(std::min(index, static_cast<size_t>(UINT32_MAX))));
            if (coordinatePriorities.try_emplace({key.tileX, key.tileZ}, nextCoordinatePriority)
                    .second &&
                nextCoordinatePriority != std::numeric_limits<uint32_t>::max()) {
                ++nextCoordinatePriority;
            }
        }

        criticalWantedKeys_ = std::move(canonicalKeys);
        criticalNearestFirst_ = std::move(canonicalNearestFirst);
        criticalPriorities_ = std::move(priorities);
        const auto refresh = [&](Job& job) {
            job.cameraNearCritical = criticalWantedKeys_.contains(job.key);
            if (const auto priority = coordinatePriorities.find({job.key.tileX, job.key.tileZ});
                priority != coordinatePriorities.end()) {
                job.viewPriority = priority->second;
            }
        };
        for (Job& job : jobs_)
            refresh(job);
        std::sort(jobs_.begin(), jobs_.end(), jobBefore);
        for (auto& entry : parkedBaseJobs_)
            refresh(entry.second.job);
        for (auto& entry : terrainFollowupJobs_)
            refresh(entry.second);
        for (auto& entry : activeWorkerJobs_)
            refresh(entry.second);
        {
            std::lock_guard completedLock(completedMutex_);
            std::stable_sort(completed_.begin(), completed_.end(),
                             [this](const FarTerrainResult& first, const FarTerrainResult& second) {
                                 return completedResultBeforeLocked(first, second);
                             });
        }
    }

    criticalPriorityUpdates_.fetch_add(1, std::memory_order_relaxed);
    jobCv_.notify_all();
    return true;
}

uint64_t FarTerrainScheduler::advanceProtectedHandoffEpoch() {
    const uint64_t next = protectedHandoffEpoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    size_t queuedRemoved = 0;
    size_t queuedBaseRemoved = 0;
    size_t queuedRefinementRemoved = 0;
    size_t queuedUrgentRemoved = 0;
    size_t queuedVisibleFinalParentRemoved = 0;
    size_t parkedRemoved = 0;
    size_t parkedUrgentRemoved = 0;
    size_t parkedVisibleFinalParentRemoved = 0;
    size_t followupsRemoved = 0;
    size_t followupUrgentReverted = 0;
    size_t followupVisibleFinalParentReverted = 0;
    std::vector<FarTerrainKey> retainedParked;
    const auto staleProtected = [&](const Job& job) {
        return job.authorityPriority ==
                   worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF &&
               job.protectedHandoffEpoch < next;
    };
    {
        std::lock_guard lock(jobMutex_);
        const auto remainsCritical = [&](FarTerrainKey key) {
            return wantedMembership_ && criticalWantedKeys_.contains(key);
        };
        std::erase_if(jobs_, [&](Job& job) {
            if (!staleProtected(job)) return false;
            if (remainsCritical(job.key)) {
                job.protectedHandoffEpoch = next;
                return false;
            }
            ++queuedRemoved;
            (farTerrainIsBaseStep(job.key.step) ? queuedBaseRemoved : queuedRefinementRemoved)++;
            if (job.urgentRefinement) ++queuedUrgentRemoved;
            if (job.visibleFinalParent) ++queuedVisibleFinalParentRemoved;
            if (const auto active = activeKeys_.find(job.key);
                active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
            return true;
        });
        for (auto parked = parkedBaseJobs_.begin(); parked != parkedBaseJobs_.end();) {
            const Job& job = parked->second.job;
            if (!staleProtected(job)) {
                ++parked;
                continue;
            }
            if (remainsCritical(job.key)) {
                parked->second.job.protectedHandoffEpoch = next;
                retainedParked.push_back(job.key);
                ++parked;
                continue;
            }
            removeParkedBaseWaitersLocked(parked->second);
            if (const auto active = activeKeys_.find(job.key);
                active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
            if (job.urgentRefinement) ++parkedUrgentRemoved;
            if (job.visibleFinalParent) ++parkedVisibleFinalParentRemoved;
            parked = parkedBaseJobs_.erase(parked);
            ++parkedRemoved;
        }
        for (auto followup = terrainFollowupJobs_.begin();
             followup != terrainFollowupJobs_.end();) {
            if (!staleProtected(followup->second)) {
                ++followup;
                continue;
            }
            if (remainsCritical(followup->first)) {
                followup->second.protectedHandoffEpoch = next;
                ++followup;
                continue;
            }
            if (const auto worker = activeWorkerJobs_.find(followup->first);
                worker != activeWorkerJobs_.end()) {
                if (followup->second.urgentRefinement && !worker->second.urgentRefinement)
                    ++followupUrgentReverted;
                if (followup->second.visibleFinalParent && !worker->second.visibleFinalParent)
                    ++followupVisibleFinalParentReverted;
            }
            followup = terrainFollowupJobs_.erase(followup);
            ++followupsRemoved;
        }
        std::sort(jobs_.begin(), jobs_.end(), jobBefore);
    }
    // Retagging a parked request is not sufficient by itself. Reissue its
    // nonblocking dependency probes with the new protected epoch so the
    // inference coordinator can promote the shared flight instead of leaving
    // the retained mesh waiting on a canceled movement epoch.
    for (const FarTerrainKey key : retainedParked) {
        static_cast<void>(
            refreshParkedBaseJob(key, worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS));
    }
    if (queuedRemoved != 0) {
        queuedBaseCount_.fetch_sub(queuedBaseRemoved, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(queuedRefinementRemoved, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(queuedUrgentRemoved, std::memory_order_relaxed);
    }
    const size_t urgentRemoved = queuedUrgentRemoved + parkedUrgentRemoved + followupUrgentReverted;
    if (urgentRemoved != 0)
        urgentRefinementInFlightCount_.fetch_sub(urgentRemoved, std::memory_order_relaxed);
    const size_t visibleFinalParentsRemoved = queuedVisibleFinalParentRemoved +
                                              parkedVisibleFinalParentRemoved +
                                              followupVisibleFinalParentReverted;
    if (visibleFinalParentsRemoved != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(visibleFinalParentsRemoved,
                                                   std::memory_order_relaxed);
    }
    if (parkedRemoved != 0) parkedBaseCount_.fetch_sub(parkedRemoved, std::memory_order_relaxed);
    if (followupsRemoved != 0)
        terrainFollowupCount_.fetch_sub(followupsRemoved, std::memory_order_relaxed);
    const size_t inFlightRemoved = queuedRemoved + parkedRemoved;
    if (inFlightRemoved != 0) {
        inFlight_.fetch_sub(inFlightRemoved, std::memory_order_release);
        canceled_.fetch_add(inFlightRemoved, std::memory_order_relaxed);
        criticalDisplacements_.fetch_add(inFlightRemoved, std::memory_order_relaxed);
    }
    if (inFlightRemoved != 0 || followupsRemoved != 0) jobCv_.notify_all();
    return next;
}

uint64_t FarTerrainScheduler::cancelViewPreparation() {
    coarseAuthorityPrefetchPages_.clear();
    coarseAuthorityPrefetchCursor_ = 0;
    coarseAuthorityPrefetchOutstandingPages_.clear();
    speculativeAuthorityPrefetchPages_.clear();
    speculativeAuthorityPrefetchCursor_ = 0;
    speculativeAuthorityPrefetchOutstandingPages_.clear();
    return advanceEpoch();
}

uint64_t FarTerrainScheduler::advanceEpoch() {
    const uint64_t next = epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    size_t removedVisibleFinalParents = 0;
    size_t parkedRemoved = 0;
    size_t parkedUrgentRemoved = 0;
    size_t parkedVisibleFinalParentsRemoved = 0;
    size_t followupUrgentReverted = 0;
    size_t followupVisibleFinalParentReverted = 0;
    size_t followupsRemoved = 0;
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
            if (job.visibleFinalParent) ++removedVisibleFinalParents;
            auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch) {
                activeKeys_.erase(active);
            }
        }
        jobs_.clear();
        for (auto parked = parkedBaseJobs_.begin(); parked != parkedBaseJobs_.end();) {
            const Job job = parked->second.job;
            removeParkedBaseWaitersLocked(parked->second);
            const auto active = activeKeys_.find(job.key);
            if (active != activeKeys_.end() && active->second == job.epoch)
                activeKeys_.erase(active);
            if (job.urgentRefinement) ++parkedUrgentRemoved;
            if (job.visibleFinalParent) ++parkedVisibleFinalParentsRemoved;
            parked = parkedBaseJobs_.erase(parked);
            ++parkedRemoved;
        }
        for (const auto& [key, followup] : terrainFollowupJobs_) {
            const auto worker = activeWorkerJobs_.find(key);
            if (worker == activeWorkerJobs_.end()) continue;
            if (followup.urgentRefinement && !worker->second.urgentRefinement)
                ++followupUrgentReverted;
            if (followup.visibleFinalParent && !worker->second.visibleFinalParent)
                ++followupVisibleFinalParentReverted;
        }
        followupsRemoved = terrainFollowupJobs_.size();
        terrainFollowupJobs_.clear();
    }
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
    }
    if (removedUrgentRefinement + parkedUrgentRemoved != 0)
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement + parkedUrgentRemoved,
                                                 std::memory_order_relaxed);
    if (followupUrgentReverted != 0)
        urgentRefinementInFlightCount_.fetch_sub(followupUrgentReverted, std::memory_order_relaxed);
    if (removedVisibleFinalParents + parkedVisibleFinalParentsRemoved != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(removedVisibleFinalParents +
                                                       parkedVisibleFinalParentsRemoved,
                                                   std::memory_order_relaxed);
    }
    if (followupVisibleFinalParentReverted != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(followupVisibleFinalParentReverted,
                                                   std::memory_order_relaxed);
    }
    if (followupsRemoved != 0)
        terrainFollowupCount_.fetch_sub(followupsRemoved, std::memory_order_relaxed);
    if (parkedRemoved > 0) parkedBaseCount_.fetch_sub(parkedRemoved, std::memory_order_relaxed);
    const size_t totalRemoved = removed + parkedRemoved;
    if (totalRemoved > 0) {
        inFlight_.fetch_sub(totalRemoved, std::memory_order_relaxed);
        canceled_.fetch_add(totalRemoved, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(completedMutex_);
        completed_.clear();
        completedBaseCount_.store(0, std::memory_order_relaxed);
        completedRefinementCount_.store(0, std::memory_order_relaxed);
    }
    size_t canopyRemoved = 0;
    size_t parkedCanopyRemoved = 0;
    {
        std::lock_guard lock(canopyJobMutex_);
        canopyRemoved = canopyJobs_.size();
        parkedCanopyRemoved = parkedCanopyJobs_.size();
        canopyJobs_.clear();
        parkedCanopyJobs_.clear();
        canopyFollowupJobs_.clear();
        activeCanopyKeys_.clear();
        coarseCanopyDispatchStreak_ = 0;
    }
    if (canopyRemoved + parkedCanopyRemoved > 0) {
        queuedCanopyCount_.fetch_sub(canopyRemoved, std::memory_order_relaxed);
        parkedCanopyCount_.fetch_sub(parkedCanopyRemoved, std::memory_order_relaxed);
        canopyInFlight_.fetch_sub(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
        canopyCanceled_.fetch_add(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(canopyCompletedMutex_);
        canopyCompleted_.clear();
        completedCanopyCount_.store(0, std::memory_order_relaxed);
    }
    canopyJobCv_.notify_all();
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

void FarTerrainScheduler::drainCanopyCompleted(std::vector<FarCanopyResult>& output) {
    std::lock_guard lock(canopyCompletedMutex_);
    while (!canopyCompleted_.empty()) {
        output.push_back(std::move(canopyCompleted_.front()));
        canopyCompleted_.pop_front();
        completedCanopyCount_.fetch_sub(1, std::memory_order_relaxed);
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
    std::vector<std::shared_ptr<const FarTerrainMesh>>& output,
    FarTerrainAuthorityQuality minimumQuality) const {
    output.clear();
    output.reserve(std::min(keys.size(), maximumResults));
    if (maximumResults == 0) return;
    std::lock_guard lock(cacheMutex_);
    for (const FarTerrainKey key : keys) {
        auto found = cache_.find(key);
        if (found == cache_.end()) continue;
        if (!farTerrainAuthoritySatisfies(found->second.mesh->authorityQuality, minimumQuality)) {
            continue;
        }
        found->second.lastAccess = ++accessClock_;
        output.push_back(found->second.mesh);
        if (output.size() >= maximumResults) break;
    }
    cacheHits_.fetch_add(output.size(), std::memory_order_relaxed);
}

std::shared_ptr<const FarCanopyAttachment>
FarTerrainScheduler::findCachedCanopy(FarTerrainKey key) const {
    std::lock_guard lock(canopyCacheMutex_);
    if (canopyCacheMembership_ && !canopyCacheMembership_->keys.contains(key)) return {};
    auto found = canopyCache_.find(key);
    if (found == canopyCache_.end()) return {};
    found->second.lastAccess = ++canopyAccessClock_;
    canopyCacheHits_.fetch_add(1, std::memory_order_relaxed);
    return found->second.attachment;
}

void FarTerrainScheduler::findCachedCanopyBatch(
    std::span<const FarTerrainKey> keys, size_t maximumResults,
    std::vector<std::shared_ptr<const FarCanopyAttachment>>& output) const {
    output.clear();
    output.reserve(std::min(keys.size(), maximumResults));
    if (maximumResults == 0) return;
    std::lock_guard lock(canopyCacheMutex_);
    for (const FarTerrainKey key : keys) {
        if (canopyCacheMembership_ && !canopyCacheMembership_->keys.contains(key)) continue;
        auto found = canopyCache_.find(key);
        if (found == canopyCache_.end()) continue;
        found->second.lastAccess = ++canopyAccessClock_;
        output.push_back(found->second.attachment);
        if (output.size() >= maximumResults) break;
    }
    canopyCacheHits_.fetch_add(output.size(), std::memory_order_relaxed);
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
        const auto cached = cache_.find({coordinate.x, coordinate.z, step});
        if (cached != cache_.end()) available |= farTerrainStepMask(step);
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
            const auto cached = cache_.find({request.coordinate.x, request.coordinate.z, step});
            if (cached != cache_.end()) available |= farTerrainStepMask(step);
        }
        const std::optional<FarTerrainStep> target =
            farTerrainReadyTransitionTarget(request.displayed, request.desired, available, false);
        if (!target || (request.residentSteps & farTerrainStepMask(*target)) != 0) continue;
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
    {
        std::lock_guard lock(cacheMutex_);
        cache_.clear();
        cacheMaintenanceQueue_.clear();
        cacheMaintenanceRemaining_ = 0;
        cacheBytes_ = 0;
        cacheBaseEntries_ = 0;
        maintenancePendingSnapshot_.store(0, std::memory_order_release);
        publishCacheStatsLocked();
    }
    {
        std::lock_guard lock(canopyCacheMutex_);
        canopyCache_.clear();
        canopyCacheBytes_ = 0;
        canopyCacheEntryCount_.store(0, std::memory_order_relaxed);
        canopyCacheBytesSnapshot_.store(0, std::memory_order_release);
    }
}

FarTerrainSchedulerStats FarTerrainScheduler::stats() const {
    FarTerrainSchedulerStats result;
    result.inFlight = inFlight_.load(std::memory_order_acquire);
    result.epoch = epoch_.load(std::memory_order_relaxed);
    result.submitted = submitted_.load(std::memory_order_relaxed);
    result.built = built_.load(std::memory_order_relaxed);
    result.canceled = canceled_.load(std::memory_order_relaxed);
    result.criticalDisplacements = criticalDisplacements_.load(std::memory_order_relaxed);
    result.failed = failed_.load(std::memory_order_relaxed);
    result.deferred = deferred_.load(std::memory_order_relaxed);
    result.step32WaterGridCalls = step32WaterGridCalls_.load(std::memory_order_relaxed);
    result.step32WaterGridSamples = step32WaterGridSamples_.load(std::memory_order_relaxed);
    result.step32WaterPointSamples = step32WaterPointSamples_.load(std::memory_order_relaxed);
    result.step32WaterDenseGridCalls = step32WaterDenseGridCalls_.load(std::memory_order_relaxed);
    result.authorityCompletionResumes = authorityCompletionResumes_.load(std::memory_order_relaxed);
    result.quiescentAuthorityResumes = quiescentAuthorityResumes_.load(std::memory_order_relaxed);
    result.cacheHits = cacheHits_.load(std::memory_order_relaxed);
    result.wantedUpdates = wantedUpdates_.load(std::memory_order_relaxed);
    result.wantedNoops = wantedNoops_.load(std::memory_order_relaxed);
    result.criticalPriorityUpdates = criticalPriorityUpdates_.load(std::memory_order_relaxed);
    result.criticalPriorityNoops = criticalPriorityNoops_.load(std::memory_order_relaxed);
    result.maintenancePending = maintenancePendingSnapshot_.load(std::memory_order_relaxed);
    result.maintenancePasses = maintenancePasses_.load(std::memory_order_relaxed);
    result.maintenanceScanned = maintenanceScanned_.load(std::memory_order_relaxed);
    result.maintenanceEvicted = maintenanceEvicted_.load(std::memory_order_relaxed);
    result.maintenanceBytes = maintenanceBytes_.load(std::memory_order_relaxed);
    result.maximumMaintenanceScanned = maximumMaintenanceScanned_.load(std::memory_order_relaxed);
    result.maximumMaintenanceBytes = maximumMaintenanceBytes_.load(std::memory_order_relaxed);
    result.queuedBase = queuedBaseCount_.load(std::memory_order_relaxed);
    result.parkedBase = parkedBaseCount_.load(std::memory_order_relaxed);
    result.terrainFollowups = terrainFollowupCount_.load(std::memory_order_relaxed);
    result.queuedRefinement = queuedRefinementCount_.load(std::memory_order_relaxed);
    result.queuedUrgentRefinement = queuedUrgentRefinementCount_.load(std::memory_order_relaxed);
    result.activeUrgentRefinement =
        activeUrgentRefinementCountSnapshot_.load(std::memory_order_relaxed);
    result.urgentRefinementInFlight =
        urgentRefinementInFlightCount_.load(std::memory_order_relaxed);
    result.visibleFinalParentInFlight =
        visibleFinalParentInFlightCount_.load(std::memory_order_relaxed);
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
    result.canopyInFlight = canopyInFlight_.load(std::memory_order_relaxed);
    result.activeCanopyWorkers = activeCanopyWorkerCount_.load(std::memory_order_relaxed);
    result.queuedCanopy = queuedCanopyCount_.load(std::memory_order_relaxed);
    result.parkedCanopy = parkedCanopyCount_.load(std::memory_order_relaxed);
    result.completedCanopy = completedCanopyCount_.load(std::memory_order_relaxed);
    result.canopyCacheEntries = canopyCacheEntryCount_.load(std::memory_order_relaxed);
    result.canopyCacheBytes = canopyCacheBytesSnapshot_.load(std::memory_order_relaxed);
    result.canopySubmitted = canopySubmitted_.load(std::memory_order_relaxed);
    result.canopyBuilt = canopyBuilt_.load(std::memory_order_relaxed);
    result.canopyCanceled = canopyCanceled_.load(std::memory_order_relaxed);
    result.canopyFailed = canopyFailed_.load(std::memory_order_relaxed);
    result.canopyDeferred = canopyDeferred_.load(std::memory_order_relaxed);
    result.canopyAuthorityCompletionResumes =
        canopyAuthorityCompletionResumes_.load(std::memory_order_relaxed);
    result.canopyCacheHits = canopyCacheHits_.load(std::memory_order_relaxed);
    return result;
}

FarTerrainGenerationCacheStats FarTerrainScheduler::generationCacheStats() const {
    if (!generator_) return {};
    const worldgen::BasinCacheMetrics basin = generator_->basinCacheMetrics();
    const worldgen::MacroControlCacheMetrics macroControl = generator_->macroControlCacheMetrics();
    const worldgen::MacroControlCacheMetrics farClimate =
        generator_->farClimateControlCacheMetrics();
    FarTerrainGenerationCacheStats result{
        .entries =
            basin.entries + basin.shorelineEntries + macroControl.entries + farClimate.entries,
        .bytes = basin.bytes + basin.shorelineBytes + macroControl.bytes + farClimate.bytes,
    };
    if (previewGenerator_) {
        const worldgen::BasinCacheMetrics previewBasin = previewGenerator_->basinCacheMetrics();
        const worldgen::MacroControlCacheMetrics previewMacro =
            previewGenerator_->macroControlCacheMetrics();
        const worldgen::MacroControlCacheMetrics previewClimate =
            previewGenerator_->farClimateControlCacheMetrics();
        result.entries += previewBasin.entries + previewBasin.shorelineEntries +
                          previewMacro.entries + previewClimate.entries;
        result.bytes += previewBasin.bytes + previewBasin.shorelineBytes + previewMacro.bytes +
                        previewClimate.bytes;
    }
    if (canopyGenerator_) {
        const worldgen::BasinCacheMetrics canopyBasin = canopyGenerator_->basinCacheMetrics();
        const worldgen::MacroControlCacheMetrics canopyMacro =
            canopyGenerator_->macroControlCacheMetrics();
        const worldgen::MacroControlCacheMetrics canopyClimate =
            canopyGenerator_->farClimateControlCacheMetrics();
        result.entries += canopyBasin.entries + canopyBasin.shorelineEntries + canopyMacro.entries +
                          canopyClimate.entries;
        result.bytes += canopyBasin.bytes + canopyBasin.shorelineBytes + canopyMacro.bytes +
                        canopyClimate.bytes;
    }
    return result;
}

void FarTerrainScheduler::shutdown() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    size_t removed = 0;
    size_t removedBase = 0;
    size_t removedRefinement = 0;
    size_t removedUrgentRefinement = 0;
    size_t removedVisibleFinalParents = 0;
    size_t parkedRemoved = 0;
    size_t parkedUrgentRemoved = 0;
    size_t parkedVisibleFinalParentsRemoved = 0;
    size_t followupUrgentReverted = 0;
    size_t followupVisibleFinalParentReverted = 0;
    size_t followupsRemoved = 0;
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
            if (job.visibleFinalParent) ++removedVisibleFinalParents;
        }
        jobs_.clear();
        for (auto parked = parkedBaseJobs_.begin(); parked != parkedBaseJobs_.end();) {
            if (parked->second.job.urgentRefinement) ++parkedUrgentRemoved;
            if (parked->second.job.visibleFinalParent) ++parkedVisibleFinalParentsRemoved;
            removeParkedBaseWaitersLocked(parked->second);
            parked = parkedBaseJobs_.erase(parked);
            ++parkedRemoved;
        }
        for (const auto& [key, followup] : terrainFollowupJobs_) {
            const auto worker = activeWorkerJobs_.find(key);
            if (worker == activeWorkerJobs_.end()) continue;
            if (followup.urgentRefinement && !worker->second.urgentRefinement)
                ++followupUrgentReverted;
            if (followup.visibleFinalParent && !worker->second.visibleFinalParent)
                ++followupVisibleFinalParentReverted;
        }
        followupsRemoved = terrainFollowupJobs_.size();
        terrainFollowupJobs_.clear();
        activeKeys_.clear();
        residencyMaintenanceRequested_ = false;
    }
    if (removed > 0) {
        queuedBaseCount_.fetch_sub(removedBase, std::memory_order_relaxed);
        queuedRefinementCount_.fetch_sub(removedRefinement, std::memory_order_relaxed);
        queuedUrgentRefinementCount_.fetch_sub(removedUrgentRefinement, std::memory_order_relaxed);
    }
    if (removedUrgentRefinement + parkedUrgentRemoved != 0)
        urgentRefinementInFlightCount_.fetch_sub(removedUrgentRefinement + parkedUrgentRemoved,
                                                 std::memory_order_relaxed);
    if (followupUrgentReverted != 0)
        urgentRefinementInFlightCount_.fetch_sub(followupUrgentReverted, std::memory_order_relaxed);
    if (removedVisibleFinalParents + parkedVisibleFinalParentsRemoved != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(removedVisibleFinalParents +
                                                       parkedVisibleFinalParentsRemoved,
                                                   std::memory_order_relaxed);
    }
    if (followupVisibleFinalParentReverted != 0) {
        visibleFinalParentInFlightCount_.fetch_sub(followupVisibleFinalParentReverted,
                                                   std::memory_order_relaxed);
    }
    if (followupsRemoved != 0)
        terrainFollowupCount_.fetch_sub(followupsRemoved, std::memory_order_relaxed);
    if (parkedRemoved > 0) parkedBaseCount_.fetch_sub(parkedRemoved, std::memory_order_relaxed);
    const size_t totalRemoved = removed + parkedRemoved;
    if (totalRemoved > 0) {
        inFlight_.fetch_sub(totalRemoved, std::memory_order_relaxed);
        canceled_.fetch_add(totalRemoved, std::memory_order_relaxed);
    }
    size_t canopyRemoved = 0;
    size_t parkedCanopyRemoved = 0;
    {
        std::lock_guard lock(canopyJobMutex_);
        canopyRemoved = canopyJobs_.size();
        parkedCanopyRemoved = parkedCanopyJobs_.size();
        canopyJobs_.clear();
        parkedCanopyJobs_.clear();
        canopyFollowupJobs_.clear();
        activeCanopyKeys_.clear();
        coarseCanopyDispatchStreak_ = 0;
    }
    if (canopyRemoved + parkedCanopyRemoved > 0) {
        queuedCanopyCount_.fetch_sub(canopyRemoved, std::memory_order_relaxed);
        parkedCanopyCount_.fetch_sub(parkedCanopyRemoved, std::memory_order_relaxed);
        canopyInFlight_.fetch_sub(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
        canopyCanceled_.fetch_add(canopyRemoved + parkedCanopyRemoved, std::memory_order_relaxed);
    }
    jobCv_.notify_all();
    canopyJobCv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    for (std::thread& worker : canopyWorkers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
    canopyWorkers_.clear();
}

void FarTerrainScheduler::finishJob(const Job& job) {
    bool queuedFollowup = false;
    Job logicalJob = job;
    {
        std::lock_guard lock(jobMutex_);
        auto active = activeKeys_.find(job.key);
        const auto followup = terrainFollowupJobs_.find(job.key);
        const bool wanted = !wantedMembership_ || wantedMembership_->keys.contains(job.key);
        if (active != activeKeys_.end() && active->second == job.epoch &&
            followup != terrainFollowupJobs_.end() && followup->second.epoch == job.epoch &&
            running_.load(std::memory_order_relaxed) &&
            epoch_.load(std::memory_order_relaxed) == job.epoch && wanted) {
            logicalJob = followup->second;
            queueJobLocked(logicalJob);
            terrainFollowupJobs_.erase(followup);
            terrainFollowupCount_.fetch_sub(1, std::memory_order_relaxed);
            submitted_.fetch_add(1, std::memory_order_relaxed);
            queuedFollowup = true;
        } else if (active != activeKeys_.end() && active->second == job.epoch) {
            activeKeys_.erase(active);
        }
        if (!queuedFollowup && followup != terrainFollowupJobs_.end() &&
            followup->second.epoch == job.epoch) {
            logicalJob = followup->second;
            terrainFollowupJobs_.erase(followup);
            terrainFollowupCount_.fetch_sub(1, std::memory_order_relaxed);
        }
        activeWorkerJobs_.erase(job.key);
        releaseActiveWorkerLocked(job);
    }
    if (!queuedFollowup && logicalJob.urgentRefinement) {
        urgentRefinementInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (!queuedFollowup && logicalJob.visibleFinalParent) {
        visibleFinalParentInFlightCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (!queuedFollowup) inFlight_.fetch_sub(1, std::memory_order_release);
    jobCv_.notify_all();
}

void FarTerrainScheduler::finishCanopyJob(const CanopyJob& job, bool allowFollowup) {
    bool queuedFollowup = false;
    {
        std::lock_guard lock(canopyJobMutex_);
        const auto active = activeCanopyKeys_.find(job.key);
        const auto followup = canopyFollowupJobs_.find(job.key);
        CanopyJob next = job;
        bool hasFollowup = false;
        if (allowFollowup && job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW) {
            next.ecologyQuality = FarTerrainAuthorityQuality::FINAL;
            next.minimumAuthorityCompletion = 0;
            next.provisionalPublished = true;
            hasFollowup = true;
        }
        if (allowFollowup && followup != canopyFollowupJobs_.end() &&
            followup->second.epoch == job.epoch) {
            const bool provisionalGroundingAdvance =
                followup->second.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW &&
                farTerrainAuthoritySatisfies(followup->second.groundingQuality,
                                             job.groundingQuality) &&
                followup->second.groundingQuality != job.groundingQuality;
            if (provisionalGroundingAdvance) {
                // Publish the successor on its promoted surface before asking
                // FINAL ecology to own that surface. The current provisional
                // allocation remains drawable throughout this queued phase.
                next = followup->second;
                next.minimumAuthorityCompletion = 0;
                hasFollowup = true;
            } else {
                next.viewPriority = std::min(next.viewPriority, followup->second.viewPriority);
                if (farTerrainAuthoritySatisfies(followup->second.groundingQuality,
                                                 next.groundingQuality)) {
                    next.groundingQuality = followup->second.groundingQuality;
                }
                if (farTerrainAuthoritySatisfies(followup->second.ecologyQuality,
                                                 next.ecologyQuality)) {
                    next.ecologyQuality = followup->second.ecologyQuality;
                }
                hasFollowup = hasFollowup || next.groundingQuality != job.groundingQuality ||
                              next.ecologyQuality != job.ecologyQuality;
            }
        }
        if (hasFollowup && active != activeCanopyKeys_.end() && active->second == job.epoch &&
            running_.load(std::memory_order_relaxed) &&
            job.epoch == epoch_.load(std::memory_order_relaxed)) {
            const auto insertion =
                std::find_if(canopyJobs_.begin(), canopyJobs_.end(), [&](const CanopyJob& queued) {
                    return canopyJobBefore(next, queued);
                });
            canopyJobs_.insert(insertion, std::move(next));
            queuedCanopyCount_.fetch_add(1, std::memory_order_relaxed);
            canopySubmitted_.fetch_add(1, std::memory_order_relaxed);
            queuedFollowup = true;
        } else if (active != activeCanopyKeys_.end() && active->second == job.epoch) {
            activeCanopyKeys_.erase(active);
        }
        if (followup != canopyFollowupJobs_.end() && followup->second.epoch == job.epoch)
            canopyFollowupJobs_.erase(followup);
        activeCanopyWorkerCount_.fetch_sub(1, std::memory_order_relaxed);
        if (job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW) {
            if (activePreviewCanopyWorkerCount_ > 0) --activePreviewCanopyWorkerCount_;
        } else if (activeFinalCanopyWorkerCount_ > 0) {
            --activeFinalCanopyWorkerCount_;
        }
    }
    if (!queuedFollowup) canopyInFlight_.fetch_sub(1, std::memory_order_release);
    canopyJobCv_.notify_all();
}

bool FarTerrainScheduler::completedResultBeforeLocked(
    const FarTerrainResult& first, const FarTerrainResult& second) const noexcept {
    const auto criticalRank = [&](FarTerrainKey key) {
        const auto found = criticalPriorities_.find(key);
        return found == criticalPriorities_.end() ? std::numeric_limits<uint32_t>::max()
                                                  : found->second;
    };
    const uint32_t firstCriticalRank = criticalRank(first.key);
    const uint32_t secondCriticalRank = criticalRank(second.key);
    const bool firstCritical = firstCriticalRank != std::numeric_limits<uint32_t>::max();
    const bool secondCritical = secondCriticalRank != std::numeric_limits<uint32_t>::max();
    if (firstCritical != secondCritical) return firstCritical;
    if (firstCritical && firstCriticalRank != secondCriticalRank)
        return firstCriticalRank < secondCriticalRank;

    const auto coordinateRank = [&](FarTerrainKey key) {
        if (!wantedMembership_) return std::numeric_limits<uint32_t>::max();
        const auto found = wantedMembership_->coordinatePriorities.find({key.tileX, key.tileZ});
        return found == wantedMembership_->coordinatePriorities.end()
                   ? std::numeric_limits<uint32_t>::max()
                   : found->second;
    };
    const uint32_t firstCoordinateRank = coordinateRank(first.key);
    const uint32_t secondCoordinateRank = coordinateRank(second.key);
    if (firstCoordinateRank != secondCoordinateRank)
        return firstCoordinateRank < secondCoordinateRank;

    if (first.key.tileX == second.key.tileX && first.key.tileZ == second.key.tileZ) {
        const bool firstBase = farTerrainIsBaseStep(first.key.step);
        const bool secondBase = farTerrainIsBaseStep(second.key.step);
        if (firstBase != secondBase) return firstBase;
        const int firstStep = farTerrainStepSize(first.key.step);
        const int secondStep = farTerrainStepSize(second.key.step);
        if (firstStep != secondStep) return firstStep > secondStep;
        if (first.mesh && second.mesh &&
            first.mesh->authorityQuality != second.mesh->authorityQuality) {
            return first.mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL;
        }
    }
    if (first.failed != second.failed) return !first.failed;
    if (first.key.tileX != second.key.tileX) return first.key.tileX < second.key.tileX;
    if (first.key.tileZ != second.key.tileZ) return first.key.tileZ < second.key.tileZ;
    return farTerrainStepSize(first.key.step) < farTerrainStepSize(second.key.step);
}

bool FarTerrainScheduler::storeCompleted(FarTerrainResult result) {
    std::lock_guard jobLock(jobMutex_);
    if (result.epoch != epoch_.load(std::memory_order_relaxed) ||
        (wantedMembership_ && !wantedMembership_->keys.contains(result.key))) {
        return false;
    }
    std::lock_guard lock(completedMutex_);
    const bool resultBase = farTerrainIsBaseStep(result.key.step);
    const auto insertion =
        std::find_if(completed_.begin(), completed_.end(), [&](const FarTerrainResult& queued) {
            return completedResultBeforeLocked(result, queued);
        });
    completed_.insert(insertion, std::move(result));
    if (resultBase) {
        completedBaseCount_.fetch_add(1, std::memory_order_relaxed);
    } else {
        completedRefinementCount_.fetch_add(1, std::memory_order_relaxed);
    }
    while (completed_.size() > limits_.maxCompleted) {
        if (farTerrainIsBaseStep(completed_.back().key.step)) {
            completedBaseCount_.fetch_sub(1, std::memory_order_relaxed);
        } else {
            completedRefinementCount_.fetch_sub(1, std::memory_order_relaxed);
        }
        completed_.pop_back();
    }
    return true;
}

bool FarTerrainScheduler::storeCanopyCompleted(FarCanopyResult result) {
    std::lock_guard jobLock(jobMutex_);
    if (result.epoch != epoch_.load(std::memory_order_relaxed) ||
        (wantedMembership_ && !wantedMembership_->keys.contains(result.key))) {
        return false;
    }
    std::lock_guard lock(canopyCompletedMutex_);
    while (canopyCompleted_.size() >= limits_.maxCanopyCompleted) {
        canopyCompleted_.pop_front();
        completedCanopyCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    canopyCompleted_.push_back(std::move(result));
    completedCanopyCount_.fetch_add(1, std::memory_order_relaxed);
    return true;
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
        // Membership and cache admission are one transaction. A camera move
        // cannot publish a new critical set between victim classification and
        // insertion, including while an older mesh finishes on a worker.
        std::scoped_lock lock(jobMutex_, cacheMutex_);
        const std::shared_ptr<const ResidencyMembership> membership = wantedMembership_;
        enum class ProtectionClass : uint8_t {
            Ordinary,
            Base,
            Critical,
        };
        const auto protectionClass = [&](FarTerrainKey candidate) {
            if (membership && criticalWantedKeys_.contains(candidate))
                return ProtectionClass::Critical;
            return farTerrainIsBaseStep(candidate.step) ? ProtectionClass::Base
                                                        : ProtectionClass::Ordinary;
        };
        const ProtectionClass incomingClass = protectionClass(key);
        const auto criticalPriorityRank = [&](FarTerrainKey candidate) {
            if (!membership) return std::numeric_limits<uint32_t>::max();
            const auto priority = criticalPriorities_.find(candidate);
            return priority == criticalPriorities_.end() ? std::numeric_limits<uint32_t>::max()
                                                         : priority->second;
        };
        const uint32_t incomingCriticalRank = criticalPriorityRank(key);
        const auto coordinatePriorityRank = [&](FarTerrainKey candidate) {
            if (!membership) return std::numeric_limits<uint32_t>::max();
            const auto priority =
                membership->coordinatePriorities.find({candidate.tileX, candidate.tileZ});
            return priority == membership->coordinatePriorities.end()
                       ? std::numeric_limits<uint32_t>::max()
                       : priority->second;
        };
        const auto broadPriorityRank = [&](FarTerrainKey candidate) {
            if (!membership) return std::numeric_limits<uint32_t>::max();
            const auto priority = membership->priorities.find(candidate);
            return priority == membership->priorities.end() ? std::numeric_limits<uint32_t>::max()
                                                            : priority->second;
        };
        const auto sameClassBefore = [&](FarTerrainKey first, FarTerrainKey second) {
            const ProtectionClass priorityClass = protectionClass(first);
            if (priorityClass == ProtectionClass::Critical) {
                return criticalPriorityRank(first) < criticalPriorityRank(second);
            }
            const uint32_t firstCoordinate = coordinatePriorityRank(first);
            const uint32_t secondCoordinate = coordinatePriorityRank(second);
            if (firstCoordinate != secondCoordinate) return firstCoordinate < secondCoordinate;
            return broadPriorityRank(first) < broadPriorityRank(second);
        };
        const auto sameClassMayReplace = [&](FarTerrainKey candidate) {
            if (!membership || sameClassBefore(key, candidate)) return true;
            const uint32_t unknown = std::numeric_limits<uint32_t>::max();
            return coordinatePriorityRank(key) == unknown &&
                   coordinatePriorityRank(candidate) == unknown &&
                   broadPriorityRank(key) == unknown && broadPriorityRank(candidate) == unknown;
        };
        const auto mayEvict = [&](FarTerrainKey candidate) {
            const ProtectionClass candidateClass = protectionClass(candidate);
            switch (incomingClass) {
                case ProtectionClass::Critical:
                    return candidateClass != ProtectionClass::Critical ||
                           criticalPriorityRank(candidate) > incomingCriticalRank;
                case ProtectionClass::Base:
                    return candidateClass == ProtectionClass::Ordinary ||
                           (candidateClass == ProtectionClass::Base &&
                            sameClassMayReplace(candidate));
                case ProtectionClass::Ordinary:
                    return candidateClass == ProtectionClass::Ordinary &&
                           sameClassMayReplace(candidate);
            }
            return false;
        };
        const auto priorityRank = [&](FarTerrainKey candidate) {
            if (!membership) return std::numeric_limits<uint32_t>::max();
            if (protectionClass(candidate) == ProtectionClass::Critical)
                return criticalPriorityRank(candidate);
            return broadPriorityRank(candidate);
        };
        const auto retire = [&](auto entry) {
            cacheBytes_ -= entry->second.bytes;
            if (farTerrainIsBaseStep(entry->first.step)) --cacheBaseEntries_;
            retired.push_back(std::move(entry->second.mesh));
            cache_.erase(entry);
        };
        if (membership && !membership->keys.contains(key)) stored = false;
        auto existing = cache_.end();
        if (stored) {
            existing = cache_.find(key);
            if (existing != cache_.end()) {
                if (!farTerrainAuthorityMayReplace(existing->second.mesh->authorityQuality,
                                                   mesh->authorityQuality)) {
                    stored = false;
                }
            }
        }
        size_t projectedEntries = cache_.size() - (existing != cache_.end() ? 1 : 0);
        size_t projectedResidentBytes =
            cacheBytes_ - (existing != cache_.end() ? existing->second.bytes : 0);
        const size_t availableResidentBytes = limits_.maxCacheBytes - bytes;
        std::vector<FarTerrainKey> victimKeys;
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash> selectedVictims;
        while (stored && !cache_.empty() &&
               (projectedEntries >= limits_.maxCacheEntries ||
                projectedResidentBytes > availableResidentBytes)) {
            auto victim = cache_.end();
            for (auto candidate = cache_.begin(); candidate != cache_.end(); ++candidate) {
                if (candidate == existing || selectedVictims.contains(candidate->first)) continue;
                const ProtectionClass candidateClass = protectionClass(candidate->first);
                if (!mayEvict(candidate->first)) continue;
                if (victim == cache_.end()) {
                    victim = candidate;
                    continue;
                }
                const ProtectionClass victimClass = protectionClass(victim->first);
                if (candidateClass != victimClass) {
                    if (candidateClass < victimClass) victim = candidate;
                    continue;
                }
                const uint32_t candidateCoordinateRank = coordinatePriorityRank(candidate->first);
                const uint32_t victimCoordinateRank = coordinatePriorityRank(victim->first);
                if (candidateClass != ProtectionClass::Critical &&
                    candidateCoordinateRank != victimCoordinateRank) {
                    if (candidateCoordinateRank > victimCoordinateRank) victim = candidate;
                    continue;
                }
                const uint32_t candidateRank = priorityRank(candidate->first);
                const uint32_t victimRank = priorityRank(victim->first);
                if (candidateRank > victimRank ||
                    (candidateRank == victimRank &&
                     candidate->second.lastAccess < victim->second.lastAccess)) {
                    victim = candidate;
                }
            }
            if (victim == cache_.end()) {
                stored = false;
                break;
            }
            victimKeys.push_back(victim->first);
            selectedVictims.insert(victim->first);
            --projectedEntries;
            projectedResidentBytes -= victim->second.bytes;
        }
        if (stored) {
            if (existing != cache_.end()) retire(existing);
            for (const FarTerrainKey victimKey : victimKeys) {
                const auto victim = cache_.find(victimKey);
                if (victim != cache_.end()) retire(victim);
            }
            const uint64_t token = ++cacheMaintenanceTokenClock_;
            cacheBytes_ += bytes;
            cache_.emplace(key, CacheEntry{std::move(mesh), bytes, ++accessClock_, token});
            cacheMaintenanceQueue_.push_back({key, token});
            if (membership &&
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

bool FarTerrainScheduler::storeCanopyCache(std::shared_ptr<const FarCanopyAttachment> attachment) {
    const size_t bytes = attachment->byteSize();
    const FarTerrainKey key = attachment->key;
    if (bytes > limits_.maxCanopyCacheBytes) return true;
    std::vector<std::shared_ptr<const FarCanopyAttachment>> retired;
    retired.reserve(2);
    {
        std::lock_guard lock(canopyCacheMutex_);
        if (canopyCacheMembership_ && !canopyCacheMembership_->keys.contains(key)) return true;
        const auto retire = [&](auto entry) {
            canopyCacheBytes_ -= entry->second.bytes;
            retired.push_back(std::move(entry->second.attachment));
            canopyCache_.erase(entry);
        };
        if (auto existing = canopyCache_.find(key); existing != canopyCache_.end()) {
            if (!farCanopyAnchorIdentityCompatible(existing->second.attachment->authorityQuality,
                                                   existing->second.attachment->anchorIdentityHash,
                                                   attachment->authorityQuality,
                                                   attachment->anchorIdentityHash)) {
                const std::string message =
                    "Far-canopy FINAL promotion changed stable ecology anchors at tile " +
                    std::to_string(key.tileX) + "," + std::to_string(key.tileZ) + " step " +
                    std::to_string(farTerrainStepSize(key.step));
                if (generationContext_) {
                    generationContext_->latchFailure({
                        .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                        .message = message,
                        .retriable = false,
                    });
                }
                RY_LOG_ERROR(message.c_str());
                return false;
            }
            // A provisional attachment rooted on the promoted surface is more
            // useful than FINAL ecology rooted on the retired surface. Its
            // parked retry restores FINAL ecology without exposing an empty
            // publication interval.
            if (!farCanopyMayReplace(existing->second.attachment->authorityQuality,
                                     existing->second.attachment->groundingQuality,
                                     attachment->authorityQuality, attachment->groundingQuality)) {
                return true;
            }
            retire(existing);
        }
        while (!canopyCache_.empty() && (canopyCache_.size() >= limits_.maxCanopyCacheEntries ||
                                         canopyCacheBytes_ + bytes > limits_.maxCanopyCacheBytes)) {
            auto victim =
                std::min_element(canopyCache_.begin(), canopyCache_.end(),
                                 [](const auto& first, const auto& second) {
                                     return first.second.lastAccess < second.second.lastAccess;
                                 });
            retire(victim);
        }
        canopyCacheBytes_ += bytes;
        canopyCache_.emplace(key,
                             CanopyCacheEntry{std::move(attachment), bytes, ++canopyAccessClock_});
        canopyCacheEntryCount_.store(canopyCache_.size(), std::memory_order_relaxed);
        canopyCacheBytesSnapshot_.store(canopyCacheBytes_, std::memory_order_release);
    }
    return true;
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

    bool needsMembershipRefresh = false;
    {
        std::lock_guard lock(cacheMutex_);
        needsMembershipRefresh =
            !cacheMembership_ || cacheMembership_->revision != membership->revision;
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
        if (needsMembershipRefresh &&
            (!cacheMembership_ || cacheMembership_->revision != membership->revision)) {
            retiredMembership = std::move(cacheMembership_);
            cacheMembership_ = membership;
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
    const auto cameraCritical = std::find_if(jobs_.begin(), jobs_.end(), [](const Job& queued) {
        return queued.urgentRefinement && queued.cameraNearCritical;
    });
    const auto urgent = std::find_if(jobs_.begin(), jobs_.end(),
                                     [](const Job& queued) { return queued.urgentRefinement; });
    const auto base = std::find_if(jobs_.begin(), jobs_.end(), [](const Job& queued) {
        return farTerrainIsBaseStep(queued.key.step);
    });
    const bool baseQueued = base != jobs_.end();
    // Camera-critical work is a hard admission class across the complete
    // pipeline. It includes missing protected parents as well as their
    // refinements, so reserving workers for unrelated distant parents here
    // would preserve coverage at the expense of visible near-player debt.
    if (cameraCritical != jobs_.end()) {
        selected = cameraCritical;
    } else if (nearFirstWorkEnabled_ && urgent != jobs_.end()) {
        selected = urgent;
    } else {
        const size_t reservedBaseWorkers =
            farTerrainBaseWorkerReservation(workerBudget_, baseQueued);
        if (baseQueued && activeBaseWorkerCount_ < reservedBaseWorkers) {
            selected = base;
        } else if (urgent != jobs_.end()) {
            const size_t urgentWorkerLimit = farTerrainUrgentWorkerLimit(workerBudget_, baseQueued);
            // Missing connected parents claim their reserved workers before
            // ordinary urgent work. Without queued base work, refinement may
            // use all otherwise idle capacity immediately.
            if (!baseQueued || activeUrgentRefinementCount_ < urgentWorkerLimit) {
                selected = urgent;
            } else {
                selected = base;
            }
        } else {
            selected = jobs_.begin();
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
    activeWorkerJobs_[job.key] = job;
    return job;
}

void FarTerrainScheduler::latchCriticalMeshFailure(const Job& job, std::string message) const {
    if (job.authorityQuality != FarTerrainAuthorityQuality::FINAL) return;
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context =
        generationContext_ ? generationContext_ : previewGenerationContext_;
    if (!context) return;
    context->latchFailure({.code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                           .message = std::move(message),
                           .retriable = true});
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
        bool authorityDeferred = false;
        const std::shared_ptr<worldgen::learned::WorldGenerationContext>& authorityContext =
            job.authorityQuality == FarTerrainAuthorityQuality::PREVIEW ? previewGenerationContext_
                                                                        : generationContext_;
        job.authorityCompletionAtDispatch =
            authorityContext ? authorityContext->metrics().authorityCache.completionGeneration
                             : uint64_t{0};
        try {
            const FarTerrainSource& source = sourceFor(job.authorityQuality, job.authorityPriority);
            result.mesh = FarTerrainMesher::build(job.key, source, job.authorityQuality);
        } catch (const worldgen::learned::GenerationFailureException& error) {
            authorityDeferred = error.status() == worldgen::learned::AuthorityStatus::DEFERRED;
            result.failed = !authorityDeferred;
            const bool criticalAuthorityFailure =
                job.authorityQuality == FarTerrainAuthorityQuality::FINAL;
            if (!authorityDeferred && criticalAuthorityFailure && generationContext_)
                generationContext_->latchFailure(error.failure());
        } catch (const std::exception& error) {
            result.failed = true;
            const std::string message =
                std::string("Far-terrain mesh generation failed at tile ") +
                std::to_string(job.key.tileX) + "," + std::to_string(job.key.tileZ) + " step " +
                std::to_string(farTerrainStepSize(job.key.step)) + ": " + error.what();
            latchCriticalMeshFailure(job, message);
            RY_LOG_ERROR(message.c_str());
        } catch (...) {
            result.failed = true;
            const std::string message =
                std::string("Far-terrain mesh generation failed at tile ") +
                std::to_string(job.key.tileX) + "," + std::to_string(job.key.tileZ) + " step " +
                std::to_string(farTerrainStepSize(job.key.step)) + " with an unknown exception";
            latchCriticalMeshFailure(job, message);
            RY_LOG_ERROR(message.c_str());
        }

        if (authorityDeferred) {
            // A base parent has a complete, known dependency plan. Preserve
            // its scheduler ownership while preview pages or exact FINAL
            // native-owner inputs are cold so the render thread cannot feed
            // the same tile back to a worker on every frame.
            if (farTerrainIsBaseStep(job.key.step) &&
                (previewGenerationContext_ || generationContext_)) {
                BaseAuthorityWaitSet waitingOn;
                BaseAuthorityPreparation preparation = prepareBaseAuthority(job, waitingOn);
                if (preparation == BaseAuthorityPreparation::Ready && authorityContext &&
                    job.authorityQuality == FarTerrainAuthorityQuality::FINAL &&
                    !job.nativeHydrologyDependencies.empty()) {
                    const uint64_t completion =
                        authorityContext->metrics().authorityCache.completionGeneration;
                    if (completion < job.authorityCompletionAtDispatch ||
                        (completion == job.authorityCompletionAtDispatch &&
                         completion == std::numeric_limits<uint64_t>::max())) {
                        authorityContext->latchFailure({
                            .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                            .message = "Far-terrain authority completion generation overflowed",
                            .retriable = false,
                        });
                        preparation = BaseAuthorityPreparation::Failed;
                    } else {
                        // Native Fill-Spill-Merge reconciliation may discover
                        // a bounded neighboring owner only after the direct
                        // owner is routed. Retry once for an authority result
                        // that completed during this dispatch; otherwise park
                        // until the newly enqueued request becomes observable.
                        waitingOn.minimumAuthorityCompletion =
                            completion > job.authorityCompletionAtDispatch
                                ? completion
                                : job.authorityCompletionAtDispatch + 1;
                        preparation = BaseAuthorityPreparation::Deferred;
                    }
                }
                if (preparation == BaseAuthorityPreparation::Deferred &&
                    parkActiveBaseJob(job, std::move(waitingOn))) {
                    static_cast<void>(refreshParkedBaseJob(
                        job.key, worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS));
                    continue;
                }
            }
            deferred_.fetch_add(1, std::memory_order_relaxed);
            finishJob(job);
            continue;
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
            if (!result.failed) {
                latchCriticalMeshFailure(
                    job, std::string("Far-terrain mesh generation returned no payload at tile ") +
                             std::to_string(job.key.tileX) + "," + std::to_string(job.key.tileZ) +
                             " step " + std::to_string(farTerrainStepSize(job.key.step)));
            }
            result.failed = true;
            failed_.fetch_add(1, std::memory_order_relaxed);
        } else {
            step32WaterGridCalls_.fetch_add(result.mesh->step32WaterGridCallCount,
                                            std::memory_order_relaxed);
            step32WaterGridSamples_.fetch_add(result.mesh->step32WaterGridSampleCount,
                                              std::memory_order_relaxed);
            step32WaterPointSamples_.fetch_add(result.mesh->step32WaterPointSampleCount,
                                               std::memory_order_relaxed);
            step32WaterDenseGridCalls_.fetch_add(result.mesh->step32WaterDenseGridCallCount,
                                                 std::memory_order_relaxed);
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
        if (!storeCompleted(std::move(result))) canceled_.fetch_add(1, std::memory_order_relaxed);
        finishJob(job);
    }
}

void FarTerrainScheduler::canopyWorkerLoop() {
    setCurrentThreadPriority(ThreadPriority::UTILITY);
    while (true) {
        CanopyJob job;
        {
            std::unique_lock lock(canopyJobMutex_);
            const auto dispatchReady = [this] {
                if (canopyJobs_.empty() || activeCanopyWorkerCount_.load(
                                               std::memory_order_relaxed) >= canopyWorkerBudget_) {
                    return false;
                }
                const bool queuedPreview =
                    std::ranges::any_of(canopyJobs_, [](const CanopyJob& queued) {
                        return queued.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW;
                    });
                if (queuedPreview) return true;
                const bool parkedPreview =
                    std::ranges::any_of(parkedCanopyJobs_, [](const auto& entry) {
                        return entry.second.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW;
                    });
                if (activePreviewCanopyWorkerCount_ != 0 || parkedPreview) return false;

                // Once every visible PREVIEW has published, FINAL promotion
                // may use one lane. The other remains available for a new
                // camera-visible PREVIEW because running work is not
                // preemptible.
                const size_t maximumFinalWorkers =
                    previewSource_.sample && canopyWorkerBudget_ > 1 ? 1 : canopyWorkerBudget_;
                return activeFinalCanopyWorkerCount_ < maximumFinalWorkers;
            };
            canopyJobCv_.wait(lock, [this, &dispatchReady] {
                return dispatchReady() || !running_.load(std::memory_order_acquire);
            });
            if (!running_.load(std::memory_order_relaxed)) return;
            if (!dispatchReady()) continue;
            const bool previewPhase = std::ranges::any_of(canopyJobs_, [](const CanopyJob& queued) {
                return queued.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW;
            });
            const FarTerrainAuthorityQuality selectedQuality =
                previewPhase ? FarTerrainAuthorityQuality::PREVIEW
                             : FarTerrainAuthorityQuality::FINAL;
            const auto firstInPhase =
                std::find_if(canopyJobs_.begin(), canopyJobs_.end(), [&](const CanopyJob& queued) {
                    return queued.ecologyQuality == selectedQuality;
                });
            // Step 1 evaluates every exact tree candidate in a full 256 by
            // 256 block tile. During cold streaming it can therefore occupy
            // the only optional lane long enough to leave every already
            // drawable coarse surface barren. Establish a short coarse
            // coverage wave first, but admit step 1 after four dispatches so
            // camera motion cannot starve fine flora indefinitely.
            const auto coarseCoverage =
                std::find_if(firstInPhase, canopyJobs_.end(), [&](const CanopyJob& queued) {
                    return queued.ecologyQuality == selectedQuality &&
                           queued.key.step != FarTerrainStep::ONE;
                });
            const auto stepOne =
                std::find_if(firstInPhase, canopyJobs_.end(), [&](const CanopyJob& queued) {
                    return queued.ecologyQuality == selectedQuality &&
                           queued.key.step == FarTerrainStep::ONE;
                });
            constexpr size_t COARSE_COVERAGE_DISPATCH_BURST = 4;
            const bool establishCoarseCoverage =
                previewPhase && coarseCoverage != canopyJobs_.end() &&
                stepOne != canopyJobs_.end() &&
                coarseCanopyDispatchStreak_ < COARSE_COVERAGE_DISPATCH_BURST;
            const auto selected = establishCoarseCoverage                        ? coarseCoverage
                                  : previewPhase && stepOne != canopyJobs_.end() ? stepOne
                                                                                 : firstInPhase;
            if (previewPhase && stepOne != canopyJobs_.end() &&
                selected->key.step != FarTerrainStep::ONE) {
                ++coarseCanopyDispatchStreak_;
            } else {
                coarseCanopyDispatchStreak_ = 0;
            }
            job = *selected;
            canopyJobs_.erase(selected);
            queuedCanopyCount_.fetch_sub(1, std::memory_order_relaxed);
            activeCanopyWorkerCount_.fetch_add(1, std::memory_order_relaxed);
            if (job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW) {
                ++activePreviewCanopyWorkerCount_;
            } else {
                ++activeFinalCanopyWorkerCount_;
            }
        }
        if (job.epoch != epoch_.load(std::memory_order_acquire)) {
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
            finishCanopyJob(job);
            continue;
        }

        bool noLongerWanted = false;
        {
            std::lock_guard lock(jobMutex_);
            noLongerWanted = wantedMembership_ && !wantedMembership_->keys.contains(job.key);
        }
        if (noLongerWanted) {
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
            finishCanopyJob(job);
            continue;
        }

        FarCanopyResult result;
        result.key = job.key;
        result.epoch = job.epoch;
        FarCanopyBuildDiagnostics buildDiagnostics;
        bool authorityDeferred = false;
        const std::shared_ptr<worldgen::learned::WorldGenerationContext>& authorityContext =
            job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW ? previewGenerationContext_
                                                                      : generationContext_;
        const uint64_t authorityCompletionAtDispatch =
            authorityContext ? authorityContext->metrics().authorityCache.completionGeneration
                             : uint64_t{0};
        if (farCanopyJobDiagnosticsEnabled()) {
            const char* ecologyQuality =
                job.ecologyQuality == FarTerrainAuthorityQuality::FINAL ? "final" : "preview";
            const char* groundingQuality =
                job.groundingQuality == FarTerrainAuthorityQuality::FINAL ? "final" : "preview";
            char line[256];
            std::snprintf(line, sizeof(line),
                          "Far flora job started tile %lld,%lld step %d ecology %s ground %s",
                          static_cast<long long>(job.key.tileX),
                          static_cast<long long>(job.key.tileZ), farTerrainStepSize(job.key.step),
                          ecologyQuality, groundingQuality);
            RY_LOG_INFO(line);
        }
        try {
            // The scheduler publishes PREVIEW ecology as a first-class phase
            // before it admits FINAL promotion work. Only the displayed
            // terrain source controls the anchors' ground Y in either phase.
            const FarTerrainSource& ecologySource =
                job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW
                    ? previewSource_
                    : (canopySource_.sample ? canopySource_ : source_);
            result.attachment = FarTerrainMesher::buildCanopyAttachment(
                job.key, ecologySource,
                sourceFor(job.groundingQuality,
                          worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT),
                job.groundingQuality, job.ecologyQuality, &buildDiagnostics);
        } catch (const worldgen::learned::GenerationFailureException& error) {
            authorityDeferred = error.status() == worldgen::learned::AuthorityStatus::DEFERRED;
            result.failed = !authorityDeferred;
            if (!authorityDeferred && job.ecologyQuality == FarTerrainAuthorityQuality::FINAL &&
                generationContext_) {
                generationContext_->latchFailure(error.failure());
            }
        } catch (const std::exception& error) {
            result.failed = true;
            RY_LOG_ERROR((std::string("Far-canopy attachment generation failed at tile ") +
                          std::to_string(job.key.tileX) + "," + std::to_string(job.key.tileZ) +
                          " step " + std::to_string(farTerrainStepSize(job.key.step)) + ": " +
                          error.what())
                             .c_str());
        } catch (...) {
            result.failed = true;
            RY_LOG_ERROR((std::string("Far-canopy attachment generation failed at tile ") +
                          std::to_string(job.key.tileX) + "," + std::to_string(job.key.tileZ) +
                          " step " + std::to_string(farTerrainStepSize(job.key.step)) +
                          " with an unknown exception")
                             .c_str());
        }

        if (farCanopyJobDiagnosticsEnabled()) {
            const char* outcome = authorityDeferred   ? "deferred"
                                  : result.failed     ? "failed"
                                  : result.attachment ? "ready"
                                                      : "empty";
            const auto qualityName = [](FarTerrainAuthorityQuality quality) {
                return quality == FarTerrainAuthorityQuality::FINAL ? "final" : "preview";
            };
            const size_t vertexCount =
                result.attachment ? result.attachment->vertices.size() : size_t{0};
            const size_t indexCount =
                result.attachment ? result.attachment->indices.size() : size_t{0};
            char line[768];
            std::snprintf(
                line, sizeof(line),
                "Far flora job tile %lld,%lld step %d ecology %s ground %s %s | "
                "%.2f ms total collect %.2f/%.2f ground %.2f mesh %.2f | "
                "candidates %u/%u accepted %u/%u cells %u sparse %u dense %u transition %u | "
                "vertices %zu indices %zu",
                static_cast<long long>(job.key.tileX), static_cast<long long>(job.key.tileZ),
                farTerrainStepSize(job.key.step), qualityName(job.ecologyQuality),
                qualityName(job.groundingQuality), outcome,
                static_cast<double>(buildDiagnostics.totalMicroseconds) / 1000.0,
                static_cast<double>(buildDiagnostics.canopyCollectionMicroseconds) / 1000.0,
                static_cast<double>(buildDiagnostics.floraCollectionMicroseconds) / 1000.0,
                static_cast<double>(buildDiagnostics.groundingMicroseconds) / 1000.0,
                static_cast<double>(buildDiagnostics.geometryMicroseconds) / 1000.0,
                buildDiagnostics.canopyCandidateCount, buildDiagnostics.floraCandidateCount,
                buildDiagnostics.acceptedCanopyCount, buildDiagnostics.acceptedFloraCount,
                buildDiagnostics.occupiedGroundCellCount, buildDiagnostics.sparseGroundCellCount,
                buildDiagnostics.denseGroundGridSampleCount,
                buildDiagnostics.transitionGroundSampleCount, vertexCount, indexCount);
            RY_LOG_INFO(line);
        }

        if (authorityDeferred) {
            canopyDeferred_.fetch_add(1, std::memory_order_relaxed);
            if (authorityContext) {
                const uint64_t completion =
                    authorityContext->metrics().authorityCache.completionGeneration;
                if (completion > authorityCompletionAtDispatch) {
                    job.minimumAuthorityCompletion = completion;
                    if (parkCanopyJob(job)) continue;
                } else if (authorityCompletionAtDispatch != std::numeric_limits<uint64_t>::max()) {
                    job.minimumAuthorityCompletion = authorityCompletionAtDispatch + 1;
                    if (parkCanopyJob(job)) continue;
                }
            }
            finishCanopyJob(job);
            continue;
        }
        if (job.epoch != epoch_.load(std::memory_order_acquire) ||
            !running_.load(std::memory_order_acquire)) {
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
            finishCanopyJob(job);
            continue;
        }
        {
            std::lock_guard lock(jobMutex_);
            noLongerWanted = wantedMembership_ && !wantedMembership_->keys.contains(job.key);
        }
        if (noLongerWanted) {
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
            finishCanopyJob(job);
            continue;
        }
        if (result.failed || !result.attachment) {
            result.failed = true;
            canopyFailed_.fetch_add(1, std::memory_order_relaxed);
        } else if (storeCanopyCache(result.attachment)) {
            canopyBuilt_.fetch_add(1, std::memory_order_relaxed);
        } else {
            result.failed = true;
            result.attachment.reset();
            canopyFailed_.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard lock(jobMutex_);
            noLongerWanted = wantedMembership_ && !wantedMembership_->keys.contains(job.key);
        }
        if (noLongerWanted) {
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
            finishCanopyJob(job);
            continue;
        }
        const bool succeeded = !result.failed && result.attachment != nullptr;
        if (succeeded && job.ecologyQuality == FarTerrainAuthorityQuality::PREVIEW)
            job.provisionalPublished = true;
        if (!storeCanopyCompleted(std::move(result)))
            canopyCanceled_.fetch_add(1, std::memory_order_relaxed);
        finishCanopyJob(job, succeeded);
    }
}

const FarTerrainSource& FarTerrainScheduler::sourceFor(
    FarTerrainAuthorityQuality authorityQuality,
    worldgen::learned::AuthorityRequestPriority authorityPriority) const {
    if (authorityQuality == FarTerrainAuthorityQuality::PREVIEW && previewSource_.sample) {
        return previewSource_;
    }
    if (authorityQuality == FarTerrainAuthorityQuality::FINAL &&
        authorityPriority == worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF &&
        protectedSource_.sample) {
        return protectedSource_;
    }
    return source_;
}
