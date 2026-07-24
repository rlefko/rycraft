#pragma once

#include "world/chunk.hpp"
#include "world/climate.hpp"
#include "world/column_plan.hpp"
#include "world/density_field.hpp"
#include "world/features.hpp"
#include "world/ores.hpp"
#include "world/structures.hpp"
#include "world/surface_material.hpp"
#include "world/world_config.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

// Column plans sample canonical hydrology beyond their 16-block footprint so
// interpolation, tree anchors, and water contacts share one immutable input
// field. Keep this geometry public because startup must prequeue every native
// hydrology owner a cold exact plan can touch.
inline constexpr int COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS = CHUNK_EDGE;
inline constexpr int COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE =
    CHUNK_EDGE + 1 + COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS * 2;

struct VolcanoPrimitive {
    double centerX = 0.0;
    double centerZ = 0.0;
    double radius = 0.0;
    double coneHeight = 0.0;
    double craterRadius = 0.0;
    double craterDepth = 0.0;
    double craterDatumElevation = 0.0;
    // V4 derives this immutable datum only when a column lies within the
    // primitive's bounded influence. Constructing distant hotspot chains must
    // not request learned authority pages that cannot affect the queried tile.
    bool craterDatumResolved = false;
    double craterLakeRadius = 0.0;
    double craterLakeSurface = 0.0;
    double craterRimElevation = 0.0;
    double craterRimWidth = 0.0;
    double tubePhase = 0.0;
    double conduitRadius = 0.0;
    uint64_t id = 0;
    bool shield = false;
    bool craterLake = false;
    bool lavaBearing = false;
};

struct VolcanicColumnSample {
    double heightAdjustment = 0.0;
    double strongestProfile = 0.0;
    double slopeContribution = 0.0;
    double strongestRadius = 1.0;
    double basaltField = 0.0;
    double craterFactor = 0.0;
    double craterRadius = 0.0;
    double craterProfileDistance = 1.0e9;
    double craterTerrainTarget = 0.0;
    double craterProfileInfluence = 0.0;
    double craterLakeRadius = 0.0;
    double craterLakeSurface = 0.0;
    worldgen::WaterBodyId craterWaterBodyId = worldgen::NO_WATER_BODY;
    double craterRimElevation = 0.0;
    double craterRimWidth = 0.0;
    double centerDistance = 1.0e9;
    double conduitRadius = 0.0;
    double conduitDepth = 0.0;
    double tubeDistance = 1.0e9;
    double tubeCenterOffset = 0.0;
    double tubeRadius = 0.0;
    bool craterLake = false;
    bool conduitLavaBearing = false;
    bool tubeLavaBearing = false;
};

struct LazyDensityColumn {
    DensityColumnContext context;
    std::vector<double> values;
};

struct GenScratch {
    const void* owner = nullptr;
    uint64_t ownerToken = 0;
    std::unordered_map<ColumnPos, ColumnShape> shapes;
    std::unordered_map<ColumnPos, LazyDensityColumn> densityColumns;
    std::unordered_map<ColumnPos, std::shared_ptr<const ColumnPlan>> columnPlans;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> volcanoCells;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> volcanicArcCells;
    std::unordered_map<const VolcanoPrimitive*, VolcanoPrimitive> resolvedVolcanoes;
    std::unordered_map<ColumnPos, VolcanicColumnSample> volcanicColumns;
    std::unordered_map<ColumnPos, StructurePlacement> structurePlacements;

    void reset(const void* newOwner, uint64_t newOwnerToken = 0) {
        owner = newOwner;
        ownerToken = newOwnerToken;
        shapes.clear();
        densityColumns.clear();
        columnPlans.clear();
        volcanoCells.clear();
        volcanicArcCells.clear();
        resolvedVolcanoes.clear();
        volcanicColumns.clear();
        structurePlacements.clear();
    }
};

class ChunkGenerator {
public:
    // Default settings generate byte-identical output to the settings-free
    // form; the structures toggle is the only knob that reaches the
    // generator.
    explicit ChunkGenerator(uint64_t worldSeed, GenerationSettings generation = {});
    ChunkGenerator(uint64_t worldSeed,
                   std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext,
                   GenerationSettings generation = {});

    void generate(Chunk& chunk) const;
    void generateCube(Chunk& chunk) const { generate(chunk); }

    std::shared_ptr<const ColumnPlan> getColumnPlan(ColumnPos chunkColumn) const;
    std::shared_ptr<const ColumnPlan> findColumnPlan(ColumnPos chunkColumn) const;
    worldgen::SurfaceSample sampleSurface(int64_t x, int64_t z) const;
    worldgen::SurfaceSample sampleSurface(int64_t x, int64_t z,
                                          worldgen::SurfaceFootprint footprint) const;
    // Geometry-only far samples omit climate, soil, and biome work that the
    // far mesh does not consume. Volcanic relief and canonical hydrology are
    // still applied so terrain and water remain identical.
    worldgen::SurfaceSample sampleFarGeometrySurface(int64_t x, int64_t z) const;
    worldgen::SurfaceSample sampleFarGeometrySurface(int64_t x, int64_t z,
                                                     worldgen::SurfaceFootprint footprint) const;
    void sampleFarGeometryGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                               int sampleWidth, int sampleHeight,
                               worldgen::SurfaceFootprint footprint,
                               std::span<worldgen::SurfaceSample> output) const;
    void sampleFarGeometryPoints(std::span<const ColumnPos> positions,
                                 worldgen::SurfaceFootprint footprint,
                                 std::span<worldgen::SurfaceSample> output) const;
    // Native hydrology geometry for distant water topology. Unlike the
    // general far-geometry path, this deliberately skips climate, soil, and
    // biome reconstruction because water meshes consume only solved terrain,
    // hydrology, and bounded volcanic geometry.
    void sampleNativeHydrologyGeometryGrid(int64_t originX, int64_t originZ, int spacingX,
                                           int spacingZ, int sampleWidth, int sampleHeight,
                                           std::span<worldgen::SurfaceSample> output) const;
    void sampleNativeHydrologyGeometryPoints(std::span<const ColumnPos> positions,
                                             std::span<worldgen::SurfaceSample> output) const;
    // The v4 far-water coverage raster is solved before volcanic overlays.
    // Nonvolcanic cells need only this canonical hydrology payload. Cells
    // marked as volcanic are promoted to the final geometry path separately,
    // so this avoids rebuilding geology and volcano data for every native
    // water sample in a distant parent.
    void sampleNativeHydrologyAuthorityGrid(int64_t originX, int64_t originZ, int spacingX,
                                            int spacingZ, int sampleWidth, int sampleHeight,
                                            std::span<worldgen::HydrologySample> output) const;
    void sampleNativeHydrologyAuthorityPoints(std::span<const ColumnPos> positions,
                                              std::span<worldgen::HydrologySample> output) const;
    // Coarse far meshes use the same routed native fields without invoking
    // the block-resolution stage reconciliation reserved for exact columns.
    // Every grid call keeps at least one spacing axis above one, including
    // one-row and one-column strips. Authority point batches retain their
    // immutable pages through one stage-free query; geometry point batches
    // are deterministically converted into stage-free runs before overlays.
    void
    sampleCoarseNativeHydrologyAuthorityGrid(int64_t originX, int64_t originZ, int spacingX,
                                             int spacingZ, int sampleWidth, int sampleHeight,
                                             std::span<worldgen::HydrologySample> output) const;
    void
    sampleCoarseNativeHydrologyAuthorityPoints(std::span<const ColumnPos> positions,
                                               std::span<worldgen::HydrologySample> output) const;
    void sampleCoarseNativeHydrologyGeometryGrid(int64_t originX, int64_t originZ, int spacingX,
                                                 int spacingZ, int sampleWidth, int sampleHeight,
                                                 std::span<worldgen::SurfaceSample> output) const;
    void sampleCoarseNativeHydrologyGeometryPoints(std::span<const ColumnPos> positions,
                                                   std::span<worldgen::SurfaceSample> output) const;
    // Compact v4 topology reduction for globally aligned 32-block far cells.
    // It conservatively marks only cells that can hide a native water feature
    // between their coarse terrain samples.
    void sampleNativeHydrologyTopologyGrid(
        int64_t originX, int64_t originZ, int cellWidth, int cellHeight,
        std::span<worldgen::NativeHydrologyTopologyCell> output) const;
    // Marks half-open legacy cells whose block-scale volcanic overlay can
    // introduce an analytical crater lake. V4 returns an empty mask because
    // explicit volcanic relief enters native routing before its compact
    // topology grid is built.
    void markVolcanicWaterCandidates(int64_t originX, int64_t originZ, int step, int cellWidth,
                                     int cellHeight, std::span<uint8_t> output) const;
    [[nodiscard]] bool usesLearnedAuthority() const noexcept { return learnedAuthority_; }
    [[nodiscard]] bool usesPreviewAuthority() const noexcept { return previewAuthority_; }
    worldgen::SurfaceSample sampleExactGeometrySurface(int64_t x, int64_t z) const;
    // Returns exact physical terrain, climate, hydrology, and material
    // authority. For v4, terrainHeight remains the continuous learned value;
    // emittedTerrainHeight is the exact density top, one block above
    // ColumnPlan::surfaceY(), used only by collision, fluids, and mesh
    // geometry so cubic and far terrain meet.
    worldgen::SurfaceSample sampleExactSurface(int64_t x, int64_t z) const;
    // Batched counterpart for fine far-voxel tiles. It retains each immutable
    // column plan once and reconstructs all of its block samples without
    // repeating cache lookups and direct macro solves at every vertex.
    void sampleExactSurfaceGrid(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                                std::span<worldgen::SurfaceSample> output) const;
    // Returns the exact local amplitude used by the near-surface density
    // field. Coarse coverage bounds combine this with their filtered relief
    // envelope instead of relying on one world-wide height allowance.
    static double emittedSurfaceDetailAmplitude(const worldgen::SurfaceSample& surface,
                                                double slopeEnvelope = 0.0);
    // Samples the globally aligned macro lattice without constructing a
    // 16-block plan around every sparse far-LOD vertex. At 8- and 16-block
    // aligned coordinates this matches the exact plan lattice, including
    // emitted volcanic relief.
    worldgen::SurfaceSample sampleFarSurface(int64_t x, int64_t z) const;
    worldgen::SurfaceSample sampleFarSurface(int64_t x, int64_t z,
                                             worldgen::SurfaceFootprint footprint) const;
    // Canonical pre-overlay hydrology used by ColumnPlan water ownership.
    // Tree control pages query it only for accepted roots, preventing a
    // volcanic relief sample from hiding water that exact cube emission keeps.
    worldgen::HydrologySample sampleGeneratedWaterAuthority(int64_t x, int64_t z) const;
    void sampleGeneratedWaterAuthorityGrid(int64_t originX, int64_t originZ, int spacing,
                                           int sampleEdge,
                                           std::span<worldgen::HydrologySample> output) const;
    void sampleGeneratedWaterAuthorityPoints(std::span<const ColumnPos> positions,
                                             std::span<worldgen::HydrologySample> output) const;
    // Applies coordinate-pure volcanic geometry to canonical hydrology
    // without reconstructing climate, soil, or biome fields. These batches
    // are the final water ownership authority used by far-mesh sentinels.
    void sampleGeneratedWaterGeometryGrid(int64_t originX, int64_t originZ, int spacingX,
                                          int spacingZ, int sampleWidth, int sampleHeight,
                                          std::span<worldgen::SurfaceSample> output) const;
    void sampleGeneratedWaterGeometryPoints(std::span<const ColumnPos> positions,
                                            std::span<worldgen::SurfaceSample> output) const;
    // Samples sparse final macro surfaces in one basin batch and applies the
    // same coordinate-pure volcanic geometry used by scalar far sampling.
    void sampleFarSurfacePoints(std::span<const ColumnPos> positions,
                                worldgen::SurfaceFootprint footprint,
                                std::span<worldgen::SurfaceSample> output) const;
    // Internal block-footprint habitat authority for far feature candidates.
    // Unlike the public exact sampler, this remains plan-free and batches all
    // basin and macro-control work across the requested points.
    void sampleFarHabitatPoints(std::span<const ColumnPos> positions,
                                std::span<worldgen::SurfaceSample> output) const;
    // Coarse PREVIEW attachments use learned macro ecology without forcing a
    // canonical block-scale hydrology owner. FINAL and legacy generators
    // retain the exact habitat path.
    void sampleFarEcologyPoints(std::span<const ColumnPos> positions, int lodStep,
                                std::span<worldgen::SurfaceSample> output) const;
    void sampleFarSurfaceGrid(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                              worldgen::SurfaceFootprint footprint,
                              std::span<worldgen::SurfaceSample> output) const;
    // Regional weather consumes only immutable macro elevation and climate.
    // It must not build canonical hydrology across the complete weather map.
    void sampleWeatherClimateGrid(int64_t originX, int64_t originZ, int spacing, int sampleEdge,
                                  std::span<worldgen::SurfaceSample> output) const;
    // Selects the visible top material through the exact column plan and the
    // same coordinate-pure geology, water, and weathering path used by cubic
    // emission. Fine far terrain samples this on its shared material lattice.
    BlockType surfaceMaterialAt(int64_t x, int64_t z) const;
    worldgen::surface_material::SurfaceMaterialPalette surfaceMaterialPaletteAt(int64_t x,
                                                                                int64_t z) const;
    // Macro counterpart for coarse far LODs. This avoids constructing column
    // plans where exact block-level shoreline parity is no longer visible.
    BlockType farSurfaceMaterialAt(int64_t x, int64_t z) const;
    BlockType farSurfaceMaterialAt(int64_t x, int64_t z,
                                   const worldgen::SurfaceSample& surface) const;
    worldgen::surface_material::SurfaceMaterialPalette
    farSurfaceMaterialPaletteAt(int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) const;
    worldgen::surface_material::SurfaceMaterialPalette
    farSurfaceMaterialPaletteAt(int64_t x, int64_t z, const worldgen::SurfaceSample& surface) const;
    double farSurfaceMaterialRankAt(int64_t x, int64_t z) const;
    // Height-only counterpart for conservative plan perimeters. This avoids
    // climate and soil work while using the exact emitted volcano primitive.
    double sampleFarTerrainHeight(int64_t x, int64_t z) const;
    // Coordinate-pure diagnostic view of the emitted cones for one hotspot
    // lattice cell.
    std::vector<VolcanoPrimitive> hotspotVolcanoesForCell(int64_t cellX, int64_t cellZ) const;
    // Read-only far-LOD source backed by the same accepted local-priority
    // anchors and exact density surface used by cubic tree emission. Bounds
    // are half-open and canopies rooted just outside them are included when
    // foliage intersects the requested rectangle.
    std::vector<FarCanopy> collectFarCanopies(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                              int64_t maximumZ) const;
    std::vector<FarCanopy> collectFarCanopiesForLod(int64_t minimumX, int64_t minimumZ,
                                                    int64_t maximumX, int64_t maximumZ,
                                                    int lodStep) const;
    std::vector<FarFlora> collectFarFloraForLod(int64_t minimumX, int64_t minimumZ,
                                                int64_t maximumX, int64_t maximumZ,
                                                int lodStep) const;
    size_t cachedColumnPlanCount() const { return columnPlanCache_.size(); }
    uint64_t columnPlanConstructionRequestCount() const noexcept {
        return columnPlanConstructionRequestCount_.load(std::memory_order_relaxed);
    }
    worldgen::BasinCacheMetrics basinCacheMetrics() const {
        return macroSampler_.basinCacheMetrics();
    }
    worldgen::MacroControlCacheMetrics macroControlCacheMetrics() const {
        return macroSampler_.macroControlCacheMetrics();
    }
    worldgen::MacroControlCacheMetrics farClimateControlCacheMetrics() const {
        return macroSampler_.farClimateControlCacheMetrics();
    }
    worldgen::NativeHydrologyCacheMetrics nativeHydrologyCacheMetrics() const {
        return macroSampler_.nativeHydrologyCacheMetrics();
    }
    void clearMacroCaches() const;

    double baseHeightAt(int64_t x, int64_t z, GenScratch& scratch) const;
    Biome biomeAt(int64_t x, int64_t z, GenScratch& scratch) const;
    int surfaceYAt(int64_t x, int64_t z, GenScratch& scratch) const;
    ColumnShape columnShapeAt(int64_t x, int64_t z, GenScratch& scratch) const;

    double baseHeightAt(int64_t x, int64_t z) const;
    Biome biomeAt(int64_t x, int64_t z) const;
    int surfaceYAt(int64_t x, int64_t z) const;

    uint64_t seed() const { return seed_; }
    uint64_t densityEvaluationCount() const noexcept {
        return densityEvaluationCount_.load(std::memory_order_relaxed);
    }
    uint64_t lastCubeDensityEvaluationCount() const noexcept {
        return lastCubeDensityEvaluationCount_.load(std::memory_order_relaxed);
    }
    int densityLatticeVerticalSpacing() const noexcept {
        return learnedAuthority_ ? LEARNED_DENSITY_LATTICE_Y : LEGACY_DENSITY_LATTICE_Y;
    }
    int densityLatticeLevelCount() const noexcept {
        return (WORLD_MAX_Y - WORLD_MIN_Y + 1) / densityLatticeVerticalSpacing() + 1;
    }

private:
    uint64_t seed_;
    mutable std::atomic<uint64_t> scratchToken_;
    mutable std::atomic<uint64_t> densityEvaluationCount_{0};
    mutable std::atomic<uint64_t> lastCubeDensityEvaluationCount_{0};
    mutable std::atomic<uint64_t> columnPlanConstructionRequestCount_{0};
    CounterRng random_;
    bool learnedAuthority_ = false;
    bool previewAuthority_ = false;
    worldgen::MacroGenerationSampler macroSampler_;
    mutable ColumnPlanCache columnPlanCache_;
    mutable ColumnPlanCache farWaterPlanCache_{1024};
    ClimateSampler climate_;
    DensityField density_;
    OrePlacer ores_;
    StructurePlacer structures_;
    FeaturePlacer features_;

    const ColumnShape& latticeShape(int64_t lx, int64_t lz, GenScratch& scratch) const;
    double latticeDensityAt(int64_t lx, int64_t lz, int level, GenScratch& scratch) const;
    std::shared_ptr<const ColumnPlan>
    constructColumnPlan(ColumnPos chunkColumn, bool retainInCache,
                        ColumnPlanCache* alternateCache = nullptr) const;
    void sampleCanonicalWaterSurfacePoints(std::span<const ColumnPos> positions,
                                           std::span<worldgen::SurfaceSample> output) const;
    worldgen::SurfaceSample surfaceSampleFromPlan(int64_t x, int64_t z, const ColumnPlan& plan,
                                                  GenScratch& scratch) const;
    worldgen::SurfaceSample surfaceSampleAt(int64_t x, int64_t z, GenScratch& scratch) const;
    const std::vector<VolcanoPrimitive>& volcanoesForCell(int64_t cellX, int64_t cellZ,
                                                          GenScratch& scratch) const;
    const std::vector<VolcanoPrimitive>& volcanicArcForCell(int64_t cellX, int64_t cellZ,
                                                            GenScratch& scratch) const;
    const VolcanoPrimitive& resolvedVolcano(const VolcanoPrimitive& volcano,
                                            GenScratch& scratch) const;
    const VolcanicColumnSample& volcanismAt(int64_t x, int64_t z,
                                            const worldgen::GeologySample& geology,
                                            GenScratch& scratch) const;
    worldgen::SurfaceSample applyVolcanism(int64_t x, int64_t z, worldgen::SurfaceSample surface,
                                           GenScratch& scratch) const;
    void applyVolcanicGeometry(worldgen::SurfaceSample& surface,
                               const VolcanicColumnSample& volcanism) const;
    worldgen::SurfaceSample refreshDependentSurface(int64_t x, int64_t z,
                                                    worldgen::SurfaceSample surface,
                                                    const VolcanicColumnSample& volcanism) const;
    worldgen::SurfaceSample emittedSurfaceAt(int64_t x, int64_t z, GenScratch& scratch) const;
    void samplePlanFreeExactSurfaceGrid(int64_t originX, int64_t originZ, int sampleEdge,
                                        std::span<worldgen::SurfaceSample> output) const;
    ColumnPlanSurfaceGrid exactSurfaceGrid(const ColumnPlan& plan) const;
    double evaluateDensity(double x, double y, double z, const ColumnShape& shape,
                           const DensityColumnContext& context) const;
    double interpolatedDensity(int64_t x, int y, int64_t z, GenScratch& scratch) const;
    void fillColumn(Chunk& chunk, int lx, int lz, const worldgen::SurfaceSample& surface,
                    int surfaceY, GenScratch& scratch) const;
    void prepareScratch(GenScratch& scratch) const;
    GenScratch& threadScratch() const;
};
