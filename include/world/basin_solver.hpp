#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace worldgen {

inline constexpr double BASIN_CATCHMENT_EDGE = 2048.0;
inline constexpr double BASIN_RASTER_SPACING = 16.0;
inline constexpr size_t BASIN_CACHE_BYTE_BUDGET = 64 * 1024 * 1024;
inline constexpr double SHORELINE_PAGE_EDGE = 256.0;
inline constexpr double SHORELINE_COARSE_SPACING = 4.0;
inline constexpr double SHORELINE_REFINED_SPACING = 2.0;
inline constexpr size_t SHORELINE_CACHE_BYTE_BUDGET = 64 * 1024 * 1024;
inline constexpr size_t MAX_CONCURRENT_COLD_BASIN_BUILDS = 2;

enum class BasinOutlet : uint8_t {
    NONE,
    OCEAN,
    ENDORHEIC,
    SHARED_PORTAL,
};

enum class WaterTransitionKind : uint8_t {
    NONE = 0,
    EXPLICIT_FALL,
    OUTLET_CORRIDOR,
    CHANNEL_GUIDE,
    RASTER_CHANNEL,
    COUNT,
};

using WaterBodyId = uint64_t;
inline constexpr WaterBodyId NO_WATER_BODY = 0;

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
    // A dry channel bank still has a continuous eroded substrate below its
    // freeboard. Cross-catchment reconstruction uses this value when another
    // solution owns adjacent water so the bank cannot introduce a floor wall.
    double channelBankSubstrate = 0.0;
    double lakeAreaSquareKilometers = 0.0;
    double lakeVolumeCubicMeters = 0.0;
    double lakeRunoffMmSquareKilometers = 0.0;
    double lakeLossMm = 0.0;
    double lakeOverflowMmSquareKilometers = 0.0;
    double lakeSpillSurface = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallWidth = 0.0;
    WaterBodyId waterBodyId = NO_WATER_BODY;
    // Zero denotes generated source water. Levels one through seven are
    // reserved for immutable rapid and outlet approach patches whose cells
    // form a face-connected Java-style gradient.
    uint8_t generatedFluidLevel = 0;
    WaterTransitionKind transitionOwnerKind = WaterTransitionKind::NONE;
    uint64_t transitionOwnerId = 0;
    uint8_t streamOrder = 0;
    uint8_t distributaryCount = 0;
    BasinOutlet outlet = BasinOutlet::NONE;
    double outletX = 0.0;
    double outletZ = 0.0;
    bool ocean = false;
    bool river = false;
    bool lake = false;
    bool lakeBank = false;
    bool channelBank = false;
    bool channelBankSubstrateValid = false;
    bool endorheic = false;
    bool waterfall = false;
    bool waterfallAnchor = false;
    bool delta = false;
    bool valid = false;
};

struct BasinSamplePosition {
    double x = 0.0;
    double z = 0.0;
};

struct BasinCacheMetrics {
    size_t entries = 0;
    size_t bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t builds = 0;
    uint64_t failures = 0;
    // Failed detailed solves publish one immutable coordinate-pure fallback
    // under the same key. This counter proves repeated samples do not rebuild
    // a known failure for every column.
    uint64_t fallbackBuilds = 0;
    // Successful detailed solutions publish four two-pass erosion epochs.
    // These cumulative work counters make the fixed numerical budget and
    // terrain-dependent flow adaptation observable without timing-dependent
    // acceptance checks.
    uint64_t erosionEpochs = 0;
    uint64_t erosionReroutes = 0;
    uint64_t erosionReceiverChanges = 0;
    size_t shorelineEntries = 0;
    size_t shorelineBytes = 0;
    uint64_t shorelineHits = 0;
    uint64_t shorelineMisses = 0;
    uint64_t shorelineBuilds = 0;
    uint64_t shorelineFailures = 0;
    // Point queries construct a transient retained-solution map. Exact cube
    // emission should consume ColumnPlan authority instead, so this cumulative
    // counter can enforce that boundary in regression tests and diagnostics.
    uint64_t scalarSampleCalls = 0;
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
    using PotentialEvapotranspirationFunction = std::function<double(double, double, double)>;

    explicit BasinSolver(uint64_t worldSeed, size_t cacheByteBudget = BASIN_CACHE_BYTE_BUDGET);
    ~BasinSolver();

    BasinSolver(const BasinSolver&) = delete;
    BasinSolver& operator=(const BasinSolver&) = delete;
    BasinSolver(BasinSolver&&) noexcept;
    BasinSolver& operator=(BasinSolver&&) noexcept;

    BasinSample
    sample(double x, double z, const ElevationFunction& elevation, const RainfallFunction& rainfall,
           const RockResistanceFunction& rockResistance,
           const PotentialEvapotranspirationFunction& potentialEvapotranspiration = {}) const;
    // Reconstructs a regular integer-coordinate grid while retaining every
    // immutable basin and only the contour nodes referenced by the batch.
    // This keeps dense mesh sampling from repeatedly traversing the global
    // LRU and does not alter any canonical ownership decision.
    void
    sampleGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth,
               int sampleHeight, const ElevationFunction& elevation,
               const RainfallFunction& rainfall, const RockResistanceFunction& rockResistance,
               std::span<BasinSample> output,
               const PotentialEvapotranspirationFunction& potentialEvapotranspiration = {}) const;
    // Reconstructs arbitrary point coordinates while retaining the same
    // bounded basin and shoreline solutions across the whole batch.
    void
    samplePoints(std::span<const BasinSamplePosition> positions, const ElevationFunction& elevation,
                 const RainfallFunction& rainfall, const RockResistanceFunction& rockResistance,
                 std::span<BasinSample> output,
                 const PotentialEvapotranspirationFunction& potentialEvapotranspiration = {}) const;

    BasinCacheMetrics cacheMetrics() const;
    void clear() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace worldgen
