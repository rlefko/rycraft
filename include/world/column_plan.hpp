#pragma once

#include "world/chunk_pos.hpp"
#include "world/macro_generation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <vector>

inline constexpr int COLUMN_PLAN_LATTICE_SPACING = 8;
inline constexpr int COLUMN_PLAN_LATTICE_EDGE = CHUNK_EDGE / COLUMN_PLAN_LATTICE_SPACING + 1;
inline constexpr size_t COLUMN_PLAN_LATTICE_SAMPLES =
    COLUMN_PLAN_LATTICE_EDGE * COLUMN_PLAN_LATTICE_EDGE;
inline constexpr size_t COLUMN_PLAN_CACHE_BYTE_BUDGET = 128 * 1024 * 1024;
// This value is set from the measured ColumnPlan size below. It retains the
// largest radius-35 exact-plan disk while keeping the cache below its hard
// 128 MiB budget.
inline constexpr size_t DEFAULT_COLUMN_PLAN_CACHE_CAPACITY = 3'950;

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
// 3 lattice and a compact macro control view retain continuous fields, while
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
    std::span<const int32_t> exposedSections() const {
        return std::span<const int32_t>(exposedSections_).first(floraOwnershipOffset_);
    }
    // Sections that can contain exact tree geometry or ground flora. Keeping
    // this span separate from broad generation support lets exact streaming
    // finish visible vegetation without waiting for unrelated deep terrain,
    // volcanic support, or tall vertical walls.
    std::span<const int32_t> floraOwnershipSections() const {
        return std::span<const int32_t>(exposedSections_)
            .subspan(floraOwnershipOffset_,
                     surfaceOwnershipOffset_ - floraOwnershipOffset_);
    }
    // Sections that contain the drawable terrain top, standing water, falls,
    // or required vertical surface walls. This is deliberately narrower than
    // exposedSections(), which also retains optional tree and generation
    // support. Far terrain can retire once these sections own the surface;
    // waiting for unrelated support cubes leaves coarse grass and water
    // parents visibly intersecting an already rendered exact column.
    std::span<const int32_t> surfaceOwnershipSections() const {
        return std::span<const int32_t>(exposedSections_).subspan(surfaceOwnershipOffset_);
    }
    bool exposesSection(int32_t chunkY) const;
    int minimumSurfaceY() const { return minimumSurfaceY_; }
    int maximumSurfaceY() const { return maximumSurfaceY_; }

private:
    struct TerrainDerivative {
        float x = 0.0F;
        float z = 0.0F;
        float mixed = 0.0F;
    };

    // Learned authority is sampled at every block while the plan is built.
    // Retain routing fields that cannot be reconstructed from the eight-block
    // lattice, so exact columns, far habitats, and water meshes all observe
    // one canonical route. Flow components use the same signed 16-bit coding
    // as canonical waterfalls; the remaining fields retain source float
    // precision without growing a full HydrologySample for every cell.
    struct CanonicalRoutingSample {
        int16_t encodedFlowX = 0;
        int16_t encodedFlowZ = 0;
        float discharge = 0.0F;
        float sediment = 0.0F;
        float channelDistance = 0.0F;
        float channelWidth = 0.0F;
        float channelDepth = 0.0F;
        float channelGradient = 0.0F;
        float erosionDepth = 0.0F;
        float baseflow = 0.0F;
        float precipitationSeasonality = 0.0F;
        float groundwaterRechargeMm = 0.0F;
        float groundwaterHead = 0.0F;
        float hydroperiod = 0.0F;
        uint8_t streamOrder = 0;
        uint8_t distributaryCount = 0;
        uint8_t flags = 0;
        uint8_t reserved = 0;
    };
    static_assert(sizeof(CanonicalRoutingSample) == 56);

    struct CanonicalLakeSample {
        float waterSurface = 0.0F;
        float surfaceElevation = 0.0F;
        // This applies to dry cells as well. It is retained with the exact
        // hydrology grid because learned slope is an authority field, not a
        // value reconstructed from the plan's eight-block terrain lattice.
        float terrainSlope = 0.0F;
        int16_t encodedShoreDistance = std::numeric_limits<int16_t>::min();
        uint8_t encodedBankInfluence = 0;
        uint8_t flags = 0;
    };

    // Wetlands carry a parent-owned surface stage in CanonicalLakeSample and
    // waterBodyPalette_. Keep only their saturated-ground fields sparse so
    // ordinary dry columns do not pay for two more block-resolution floats.
    struct CanonicalWetlandSample {
        uint16_t localIndex = 0;
        float groundwaterHead = 0.0F;
        uint8_t encodedHydroperiod = 0;
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
    std::array<CanonicalRoutingSample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalRouting_{};
    std::array<CanonicalLakeSample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalLakes_{};
    std::array<uint8_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> generatedTopFluidStates_{};
    std::array<uint8_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> waterTopologyFlags_{};
    std::array<uint16_t, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> waterBodyIndices_{};
    std::vector<worldgen::WaterBodyId> waterBodyPalette_;
    std::vector<CanonicalWetlandSample> canonicalWetlands_;
    std::vector<CanonicalWaterfallSample> canonicalWaterfalls_;
    std::vector<uint64_t> transitionOwnerPalette_;
    std::vector<CanonicalTransitionSample> canonicalTransitions_;
    std::array<CanonicalGeologySample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalGeology_{};
    // One allocation stores [all exposed sections][flora ownership
    // sections][surface ownership sections]. The 16-bit split points retain
    // the exact-plan cache's measured 128 MiB object budget.
    std::vector<int32_t> exposedSections_;
    float maximumWaterfallTop_ = 0.0F;
    int minimumSurfaceY_ = SEA_LEVEL;
    int maximumSurfaceY_ = SEA_LEVEL;
    bool hasCanonicalRouting_ = false;
    bool hasCanonicalGeology_ = false;
    uint16_t floraOwnershipOffset_ = 0;
    uint16_t surfaceOwnershipOffset_ = 0;

    const worldgen::SurfaceSample& lattice(int x, int z) const;
    const TerrainDerivative& terrainDerivative(int x, int z) const;
    void buildExposedSections(const ColumnPlanHeightSampler& sampleHeight);
};

static_assert((CHUNK_EDGE + 1) * (CHUNK_EDGE + 1) <=
              static_cast<size_t>(std::numeric_limits<uint16_t>::max()));
static_assert(WORLD_VERTICAL_CHUNKS <= static_cast<int>(std::numeric_limits<uint16_t>::max()));

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
