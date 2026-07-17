#pragma once

#include "world/chunk_pos.hpp"
#include "world/macro_generation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

inline constexpr int COLUMN_PLAN_LATTICE_SPACING = 8;
inline constexpr int COLUMN_PLAN_LATTICE_EDGE = CHUNK_EDGE / COLUMN_PLAN_LATTICE_SPACING + 1;
inline constexpr size_t COLUMN_PLAN_LATTICE_SAMPLES =
    COLUMN_PLAN_LATTICE_EDGE * COLUMN_PLAN_LATTICE_EDGE;
inline constexpr size_t COLUMN_PLAN_CACHE_BYTE_BUDGET = 128 * 1024 * 1024;
// A 32-chunk exact disk plus its two-column construction apron contains
// roughly 4,000 plans. Retain two neighboring working sets so ordinary
// movement does not evict plans that the active-set rebuild still needs.
inline constexpr size_t DEFAULT_COLUMN_PLAN_CACHE_CAPACITY = 8112;

// Column plans deliberately depend on a bounded coordinate-pure callback
// instead of one concrete macro sampler. The generator can therefore supply
// its final emitted surface, including discrete terrain overlays, without the
// plan retaining a callback or mutable generator state.
using ColumnPlanSurfaceSampler =
    std::function<worldgen::SurfaceSample(int64_t worldX, int64_t worldZ)>;
using ColumnPlanHeightSampler = std::function<double(int64_t worldX, int64_t worldZ)>;
class ColumnPlan;
using ColumnPlanSurfaceGrid = std::array<int16_t, CHUNK_EDGE * CHUNK_EDGE>;
using ColumnPlanSurfaceGridSampler = std::function<ColumnPlanSurfaceGrid(const ColumnPlan& plan)>;
using ColumnPlanHydrologySampler =
    std::function<worldgen::HydrologySample(int64_t worldX, int64_t worldZ)>;
using ColumnPlanGeologySampler =
    std::function<worldgen::GeologySample(int64_t worldX, int64_t worldZ)>;

// Immutable macro data for one horizontal chunk column. A world-aligned 3 by
// 3 lattice and a shared C2 control view retain continuous fields, while
// compact 17 by 17 arrays retain block-exact hydrology and geology authority.
// Remaining per-block values are reconstructed on demand.
class ColumnPlan {
public:
    ColumnPlan(ColumnPos chunkColumn, const ColumnPlanSurfaceSampler& sampleSurface,
               const ColumnPlanHeightSampler& sampleHeight,
               const ColumnPlanSurfaceGridSampler& sampleExactSurface,
               const ColumnPlanHydrologySampler& sampleHydrology = {},
               const ColumnPlanGeologySampler& sampleGeology = {},
               worldgen::MacroControlView continuousFields = {});

    ColumnPos chunkColumn() const { return chunkColumn_; }
    worldgen::SurfaceSample sample(int localX, int localZ) const;
    int surfaceY(int localX, int localZ) const;
    const std::vector<int32_t>& exposedSections() const { return exposedSections_; }
    bool exposesSection(int32_t chunkY) const;
    int minimumSurfaceY() const { return minimumSurfaceY_; }
    int maximumSurfaceY() const { return maximumSurfaceY_; }

private:
    struct TerrainDerivative {
        float x = 0.0F;
        float z = 0.0F;
        float mixed = 0.0F;
    };

    struct CanonicalLakeSample {
        float waterSurface = 0.0F;
        float surfaceElevation = 0.0F;
        int16_t encodedShoreDistance = std::numeric_limits<int16_t>::min();
        uint8_t encodedBankInfluence = 0;
        uint8_t flags = 0;
    };

    // Waterfalls occupy only a narrow analytical footprint, so retaining them
    // sparsely is substantially smaller than three additional 17 by 17 float
    // fields. Entries are emitted in local-index order during plan construction
    // and remain immutable afterward.
    struct CanonicalWaterfallSample {
        uint16_t localIndex = 0;
        int16_t encodedFlowX = 0;
        int16_t encodedFlowZ = 0;
        float top = 0.0F;
        float bottom = 0.0F;
        float width = 0.0F;
    };

    struct CanonicalTransitionSample {
        uint16_t localIndex = 0;
        uint16_t ownerPaletteIndex = 0;
        worldgen::WaterTransitionKind ownerKind = worldgen::WaterTransitionKind::NONE;
    };

    // Separate arrays keep the block-resolution authority compact and avoid
    // padding a full GeologySample 289 times. Continuous geological signals
    // come from the shared C2 control view; only categorical ownership and
    // lithology contacts require exact block resolution.
    struct CanonicalGeologySample {
        uint64_t plateId = 0;
        int16_t encodedContactDistance = 0;
        uint16_t encodedTransition = 0;
        uint8_t primaryRock = 0;
        uint8_t secondaryRock = 0;
        uint8_t crust = 0;
        uint8_t boundary = 0;
    };

    ColumnPos chunkColumn_;
    worldgen::MacroControlView continuousFields_;
    std::array<worldgen::SurfaceSample, COLUMN_PLAN_LATTICE_SAMPLES> lattice_{};
    // Shared centered derivatives turn the retained 3 by 3 lattice into a C1
    // bicubic surface. They are built from a transient one-control apron, so
    // adjacent plans reconstruct identical values and slopes without keeping
    // the apron resident.
    std::array<TerrainDerivative, COLUMN_PLAN_LATTICE_SAMPLES> terrainDerivatives_{};
    ColumnPlanSurfaceGrid exactSurfaceY_{};
    std::array<CanonicalLakeSample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalLakes_{};
    std::array<uint8_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> generatedTopFluidStates_{};
    std::array<uint8_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> waterTopologyFlags_{};
    std::array<uint16_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> waterBodyIndices_{};
    std::vector<worldgen::WaterBodyId> waterBodyPalette_;
    std::vector<CanonicalWaterfallSample> canonicalWaterfalls_;
    std::vector<uint64_t> transitionOwnerPalette_;
    std::vector<CanonicalTransitionSample> canonicalTransitions_;
    std::array<CanonicalGeologySample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalGeology_{};
    std::vector<int32_t> exposedSections_;
    float maximumWaterfallTop_ = 0.0F;
    int minimumSurfaceY_ = SEA_LEVEL;
    int maximumSurfaceY_ = SEA_LEVEL;
    bool hasCanonicalGeology_ = false;

    const worldgen::SurfaceSample& lattice(int x, int z) const;
    const TerrainDerivative& terrainDerivative(int x, int z) const;
    void buildExposedSections(const ColumnPlanHeightSampler& sampleHeight);
};

static_assert((CHUNK_EDGE + 1) * (CHUNK_EDGE + 1) <=
              static_cast<size_t>(std::numeric_limits<uint16_t>::max()));

static_assert(sizeof(ColumnPlan) * DEFAULT_COLUMN_PLAN_CACHE_CAPACITY <=
              COLUMN_PLAN_CACHE_BYTE_BUDGET);

// Bounded single-flight cache. Construction happens outside the cache lock,
// while concurrent requests for the same column share one future.
class ColumnPlanCache {
public:
    explicit ColumnPlanCache(size_t capacity = DEFAULT_COLUMN_PLAN_CACHE_CAPACITY);
    ~ColumnPlanCache();

    ColumnPlanCache(const ColumnPlanCache&) = delete;
    ColumnPlanCache& operator=(const ColumnPlanCache&) = delete;
    ColumnPlanCache(ColumnPlanCache&&) noexcept;
    ColumnPlanCache& operator=(ColumnPlanCache&&) noexcept;

    std::shared_ptr<const ColumnPlan>
    getOrCreate(ColumnPos chunkColumn, const ColumnPlanSurfaceSampler& sampleSurface,
                const ColumnPlanHeightSampler& sampleHeight,
                const ColumnPlanSurfaceGridSampler& sampleExactSurface,
                const ColumnPlanHydrologySampler& sampleHydrology = {},
                const ColumnPlanGeologySampler& sampleGeology = {},
                worldgen::MacroControlView continuousFields = {}) const;
    std::shared_ptr<const ColumnPlan> find(ColumnPos chunkColumn) const;
    size_t size() const;
    void clear();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
