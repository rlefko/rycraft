#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace worldgen {

inline constexpr double BASIN_CATCHMENT_EDGE = 2048.0;
inline constexpr double BASIN_RASTER_SPACING = 16.0;
inline constexpr size_t BASIN_CACHE_BYTE_BUDGET = 64 * 1024 * 1024;
inline constexpr size_t MAX_CONCURRENT_COLD_BASIN_BUILDS = 2;

enum class BasinOutlet : uint8_t {
    NONE,
    OCEAN,
    ENDORHEIC,
    SHARED_PORTAL,
};

struct BasinSample {
    double flowX = 1.0;
    double flowZ = 0.0;
    double surfaceElevation = 0.0;
    double waterSurface = 0.0;
    double discharge = 0.0;
    double sediment = 0.0;
    double channelDistance = 0.0;
    double channelWidth = 0.0;
    double channelDepth = 0.0;
    double channelGradient = 0.0;
    double erosionDepth = 0.0;
    double lakeDepth = 0.0;
    double lakeShoreDistance = -1.0e9;
    double shoreWaterSurface = 0.0;
    double lakeBankTarget = 0.0;
    double lakeBankInfluence = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    uint8_t streamOrder = 0;
    uint8_t distributaryCount = 0;
    BasinOutlet outlet = BasinOutlet::NONE;
    double outletX = 0.0;
    double outletZ = 0.0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool lakeBank = false;
    bool endorheic = false;
    bool waterfall = false;
    bool waterfallAnchor = false;
    bool delta = false;
    bool valid = false;
};

struct BasinCacheMetrics {
    size_t entries = 0;
    size_t bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t builds = 0;
    uint64_t failures = 0;
    // These three counters describe the process-wide construction limiter
    // shared by every BasinSolver instance, including exact and far terrain.
    size_t activeColdBuilds = 0;
    size_t peakColdBuilds = 0;
    uint64_t throttledBuilds = 0;
};

// Builds immutable, bounded drainage solutions. Every callback must be
// coordinate-pure because cached solutions can be constructed on any worker.
// One solver instance also represents one immutable callback context: callers
// must use the same elevation, rainfall, and resistance fields for its entire
// lifetime because catchment cache keys contain only seed-space coordinates.
class BasinSolver {
public:
    using ElevationFunction = std::function<double(double, double)>;
    using RainfallFunction = std::function<double(double, double, double)>;
    using RockResistanceFunction = std::function<double(double, double)>;

    explicit BasinSolver(uint64_t worldSeed, size_t cacheByteBudget = BASIN_CACHE_BYTE_BUDGET);
    ~BasinSolver();

    BasinSolver(const BasinSolver&) = delete;
    BasinSolver& operator=(const BasinSolver&) = delete;
    BasinSolver(BasinSolver&&) noexcept;
    BasinSolver& operator=(BasinSolver&&) noexcept;

    BasinSample sample(double x, double z, const ElevationFunction& elevation,
                       const RainfallFunction& rainfall,
                       const RockResistanceFunction& rockResistance) const;

    BasinCacheMetrics cacheMetrics() const;
    void clear() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace worldgen
