#pragma once

#include "world/chunk.hpp"
#include "world/climate.hpp"
#include "world/column_plan.hpp"
#include "world/density_field.hpp"
#include "world/features.hpp"
#include "world/ores.hpp"
#include "world/structures.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct VolcanoPrimitive {
    double centerX = 0.0;
    double centerZ = 0.0;
    double radius = 0.0;
    double coneHeight = 0.0;
    double craterRadius = 0.0;
    double craterDepth = 0.0;
    double craterDatumElevation = 0.0;
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

struct GenScratch {
    const void* owner = nullptr;
    uint64_t ownerToken = 0;
    std::unordered_map<ColumnPos, ColumnShape> shapes;
    std::unordered_map<ColumnPos, std::vector<double>> densityColumns;
    std::unordered_map<ColumnPos, std::shared_ptr<const ColumnPlan>> columnPlans;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> volcanoCells;
    std::unordered_map<ColumnPos, std::vector<VolcanoPrimitive>> volcanicArcCells;
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
        volcanicColumns.clear();
        structurePlacements.clear();
    }
};

class ChunkGenerator {
public:
    explicit ChunkGenerator(uint32_t worldSeed);

    void generate(Chunk& chunk) const;
    void generateCube(Chunk& chunk) const { generate(chunk); }

    std::shared_ptr<const ColumnPlan> getColumnPlan(ColumnPos chunkColumn) const;
    std::shared_ptr<const ColumnPlan> findColumnPlan(ColumnPos chunkColumn) const;
    worldgen::SurfaceSample sampleSurface(int64_t x, int64_t z) const;
    // Geometry-only far samples omit climate, soil, and biome work that the
    // far mesh does not consume. Volcanic relief and canonical hydrology are
    // still applied so terrain and water remain identical.
    worldgen::SurfaceSample sampleFarGeometrySurface(int64_t x, int64_t z) const;
    worldgen::SurfaceSample sampleExactGeometrySurface(int64_t x, int64_t z) const;
    // Returns the same climate, hydrology, and material inputs as
    // sampleSurface(), but replaces the provisional macro relief with the
    // exact emitted density top. The height is a mesh-plane coordinate, one
    // block above ColumnPlan::surfaceY(), so cubic and far terrain meet.
    worldgen::SurfaceSample sampleExactSurface(int64_t x, int64_t z) const;
    // Samples the globally aligned macro lattice without constructing a
    // 16-block plan around every sparse far-LOD vertex. At 8- and 16-block
    // aligned coordinates this matches the exact plan lattice, including
    // emitted volcanic relief.
    worldgen::SurfaceSample sampleFarSurface(int64_t x, int64_t z) const;
    // Selects the visible top material through the exact column plan and the
    // same coordinate-pure geology, water, and weathering path used by cubic
    // emission. Fine far terrain samples this on its shared material lattice.
    BlockType surfaceMaterialAt(int64_t x, int64_t z) const;
    // Macro counterpart for coarse far LODs. This avoids constructing column
    // plans where exact block-level shoreline parity is no longer visible.
    BlockType farSurfaceMaterialAt(int64_t x, int64_t z) const;
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
    size_t cachedColumnPlanCount() const { return columnPlanCache_.size(); }
    worldgen::BasinCacheMetrics basinCacheMetrics() const {
        return macroSampler_.basinCacheMetrics();
    }
    void clearMacroCaches() const;

    double baseHeightAt(int64_t x, int64_t z, GenScratch& scratch) const;
    Biome biomeAt(int64_t x, int64_t z, GenScratch& scratch) const;
    int surfaceYAt(int64_t x, int64_t z, GenScratch& scratch) const;
    ColumnShape columnShapeAt(int64_t x, int64_t z, GenScratch& scratch) const;

    double baseHeightAt(int64_t x, int64_t z) const;
    Biome biomeAt(int64_t x, int64_t z) const;
    int surfaceYAt(int64_t x, int64_t z) const;

    uint32_t seed() const { return seed_; }

private:
    uint32_t seed_;
    mutable std::atomic<uint64_t> scratchToken_;
    CounterRng random_;
    worldgen::MacroGenerationSampler macroSampler_;
    mutable ColumnPlanCache columnPlanCache_;
    ClimateSampler climate_;
    DensityField density_;
    OrePlacer ores_;
    StructurePlacer structures_;
    FeaturePlacer features_;

    const ColumnShape& latticeShape(int64_t lx, int64_t lz, GenScratch& scratch) const;
    const std::vector<double>& latticeDensityColumn(int64_t lx, int64_t lz,
                                                    GenScratch& scratch) const;
    worldgen::SurfaceSample surfaceSampleFromPlan(int64_t x, int64_t z, const ColumnPlan& plan,
                                                  GenScratch& scratch) const;
    worldgen::SurfaceSample surfaceSampleAt(int64_t x, int64_t z, GenScratch& scratch) const;
    const std::vector<VolcanoPrimitive>& volcanoesForCell(int64_t cellX, int64_t cellZ,
                                                          GenScratch& scratch) const;
    const std::vector<VolcanoPrimitive>& volcanicArcForCell(int64_t cellX, int64_t cellZ,
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
    ColumnPlanSurfaceGrid exactSurfaceGrid(const ColumnPlan& plan) const;
    double interpolatedDensity(int64_t x, int y, int64_t z, GenScratch& scratch) const;
    void fillColumn(Chunk& chunk, int lx, int lz, const worldgen::SurfaceSample& surface,
                    int surfaceY, GenScratch& scratch) const;
    void prepareScratch(GenScratch& scratch) const;
    GenScratch& threadScratch() const;
};
