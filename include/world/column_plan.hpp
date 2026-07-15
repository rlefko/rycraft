#pragma once

#include "world/chunk_pos.hpp"
#include "world/macro_generation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

inline constexpr int COLUMN_PLAN_LATTICE_SPACING = 8;
inline constexpr int COLUMN_PLAN_LATTICE_EDGE = CHUNK_EDGE / COLUMN_PLAN_LATTICE_SPACING + 1;
inline constexpr size_t COLUMN_PLAN_LATTICE_SAMPLES =
    COLUMN_PLAN_LATTICE_EDGE * COLUMN_PLAN_LATTICE_EDGE;
inline constexpr size_t COLUMN_PLAN_CACHE_BYTE_BUDGET = 64 * 1024 * 1024;
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

// Immutable macro data for one horizontal chunk column. Only a world-aligned
// 3 by 3 lattice is retained; per-block samples are interpolated on demand.
class ColumnPlan {
public:
    ColumnPlan(ColumnPos chunkColumn, const ColumnPlanSurfaceSampler& sampleSurface,
               const ColumnPlanHeightSampler& sampleHeight,
               const ColumnPlanSurfaceGridSampler& sampleExactSurface,
               const ColumnPlanHydrologySampler& sampleHydrology = {});

    ColumnPos chunkColumn() const { return chunkColumn_; }
    worldgen::SurfaceSample sample(int localX, int localZ) const;
    int surfaceY(int localX, int localZ) const;
    const std::vector<int32_t>& exposedSections() const { return exposedSections_; }
    bool exposesSection(int32_t chunkY) const;
    int minimumSurfaceY() const { return minimumSurfaceY_; }
    int maximumSurfaceY() const { return maximumSurfaceY_; }

private:
    struct CanonicalLakeSample {
        float waterSurface = 0.0F;
        uint16_t encodedDepth = 0;
        uint8_t flags = 0;
    };

    ColumnPos chunkColumn_;
    std::array<worldgen::SurfaceSample, COLUMN_PLAN_LATTICE_SAMPLES> lattice_{};
    ColumnPlanSurfaceGrid exactSurfaceY_{};
    std::array<CanonicalLakeSample, (CHUNK_EDGE + 1) * (CHUNK_EDGE + 1)> canonicalLakes_{};
    std::vector<int32_t> exposedSections_;
    int minimumSurfaceY_ = SEA_LEVEL;
    int maximumSurfaceY_ = SEA_LEVEL;

    const worldgen::SurfaceSample& lattice(int x, int z) const;
    void buildExposedSections(const ColumnPlanHeightSampler& sampleHeight);
};

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
                const ColumnPlanHydrologySampler& sampleHydrology = {}) const;
    std::shared_ptr<const ColumnPlan> find(ColumnPos chunkColumn) const;
    size_t size() const;
    void clear();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
