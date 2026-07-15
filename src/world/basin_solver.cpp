#include "world/basin_solver.hpp"

#include "common/counter_rng.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace worldgen {
namespace {

constexpr int RASTER_APRON = 2;
constexpr int INTERIOR_INTERVALS = static_cast<int>(BASIN_CATCHMENT_EDGE / BASIN_RASTER_SPACING);
constexpr int RASTER_EDGE = INTERIOR_INTERVALS + 1 + RASTER_APRON * 2;
constexpr int RASTER_CELLS = RASTER_EDGE * RASTER_EDGE;
constexpr int INPUT_SAMPLE_STRIDE = 4;
constexpr int EROSION_PASSES = 8;
constexpr int SHARED_EDGE_BAND_CELLS = 4;
constexpr int LAKE_SEAM_SHAPING_CELLS = 6;
constexpr double LAKE_SEAM_SETBACK = 48.0;
constexpr double LAKE_BANK_WIDTH = 16.0;
constexpr double LAKE_SHELF_WIDTH = 16.0;
constexpr int PORTAL_PROFILE_MAX_EDGES = 64;
constexpr int PORTAL_PROFILE_TERRAIN_SAMPLES = 17;
constexpr int CHANNEL_GUIDE_CURVE_SEGMENTS = 16;
constexpr double MIN_CHANNEL_DISCHARGE = 105.0;
constexpr uint64_t SITE_POSITION_STREAM = 0xB451'0001ULL;
constexpr uint64_t SITE_PROPERTY_STREAM = 0xB451'0002ULL;
constexpr uint64_t WATERFALL_STREAM = 0xB451'0003ULL;
constexpr uint64_t DELTA_STREAM = 0xB451'0004ULL;
constexpr uint64_t CHANNEL_MEANDER_STREAM = 0xB451'0005ULL;

struct ColdBasinBuildMetrics {
    size_t active = 0;
    size_t peak = 0;
    uint64_t throttled = 0;
};

class ColdBasinBuildLimiter {
public:
    void acquire() {
        std::unique_lock lock(mutex_);
        if (active_ >= MAX_CONCURRENT_COLD_BASIN_BUILDS) ++throttled_;
        available_.wait(lock, [this] { return active_ < MAX_CONCURRENT_COLD_BASIN_BUILDS; });
        ++active_;
        peak_ = std::max(peak_, active_);
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            --active_;
        }
        available_.notify_one();
    }

    ColdBasinBuildMetrics metrics() const {
        std::lock_guard lock(mutex_);
        return {active_, peak_, throttled_};
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable available_;
    size_t active_ = 0;
    size_t peak_ = 0;
    uint64_t throttled_ = 0;
};

ColdBasinBuildLimiter& coldBasinBuildLimiter() {
    static ColdBasinBuildLimiter limiter;
    return limiter;
}

class ColdBasinBuildPermit {
public:
    ColdBasinBuildPermit() { coldBasinBuildLimiter().acquire(); }
    ~ColdBasinBuildPermit() { coldBasinBuildLimiter().release(); }

    ColdBasinBuildPermit(const ColdBasinBuildPermit&) = delete;
    ColdBasinBuildPermit& operator=(const ColdBasinBuildPermit&) = delete;
};

constexpr std::array<int, 8> DX = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 8> DZ = {0, 1, 1, 1, 0, -1, -1, -1};
constexpr std::array<double, 8> STEP_LENGTH = {
    1.0, std::numbers::sqrt2, 1.0, std::numbers::sqrt2,
    1.0, std::numbers::sqrt2, 1.0, std::numbers::sqrt2,
};

struct BasinKey {
    int64_t x = 0;
    int64_t z = 0;

    bool operator==(const BasinKey&) const = default;
};

struct BasinKeyHash {
    size_t operator()(const BasinKey& key) const noexcept {
        auto mix = [](uint64_t value) {
            value ^= value >> 30U;
            value *= 0xBF58'476D'1CE4'E5B9ULL;
            value ^= value >> 27U;
            value *= 0x94D0'49BB'1331'11EBULL;
            return value ^ (value >> 31U);
        };
        const uint64_t first = mix(static_cast<uint64_t>(key.x));
        const uint64_t second = mix(static_cast<uint64_t>(key.z) ^ first);
        return static_cast<size_t>(first ^ (second + 0x9E37'79B9'7F4A'7C15ULL));
    }
};

struct Site {
    BasinKey cell;
    double x = 0.0;
    double z = 0.0;
    double elevation = 0.0;
    double rainfall = 0.0;
    uint64_t tie = 0;
    bool ocean = false;
};

struct Portal {
    double x = 0.0;
    double z = 0.0;
};

struct TerminalLake {
    double x = 0.0;
    double z = 0.0;
    double surface = 0.0;
    double radius = 0.0;
    bool standing = false;
};

struct Receiver {
    int32_t first = -1;
    int32_t second = -1;
    float firstWeight = 0.0F;
    float secondWeight = 0.0F;
};

struct DeltaBranch {
    double startX = 0.0;
    double startZ = 0.0;
    double endX = 0.0;
    double endZ = 0.0;
    float width = 0.0F;
    float waterSurface = SEA_LEVEL;
    uint8_t groupCount = 0;
    bool lakeEntry = false;
};

struct OutletFall {
    double startX = 0.0;
    double startZ = 0.0;
    double endX = 0.0;
    double endZ = 0.0;
    float topSurface = 0.0F;
    float bottomSurface = 0.0F;
    float halfWidth = 0.0F;
};

struct GuidePoint {
    double x = 0.0;
    double z = 0.0;
};

struct ChannelGuide {
    double startX = 0.0;
    double startZ = 0.0;
    double endX = 0.0;
    double endZ = 0.0;
    double discharge = 0.0;
    double portalX = 0.0;
    double portalZ = 0.0;
    double portalWater = SEA_LEVEL;
    double downstreamWater = SEA_LEVEL;
    double terrainUpper = SEA_LEVEL;
    double gradient = 0.0;
    double portalFlowX = 1.0;
    double portalFlowZ = 0.0;
    double widthPhase = 0.0;
    double widthVariation = 0.0;
    std::array<GuidePoint, CHANNEL_GUIDE_CURVE_SEGMENTS + 1> curve;
    bool outgoing = false;
    bool oceanMouth = false;
    bool backwater = false;
    bool profileFall = false;
};

struct PortalProfile {
    double water = SEA_LEVEL;
    double downstreamWater = SEA_LEVEL;
    double terrainUpper = SEA_LEVEL;
    bool backwater = false;
    bool profileFall = false;
};

enum CellFlags : uint8_t {
    CELL_OCEAN = 1U << 0U,
    CELL_LAKE = 1U << 1U,
    CELL_ENDORHEIC = 1U << 2U,
    CELL_WATERFALL = 1U << 3U,
    CELL_LAKE_OUTLET = 1U << 4U,
    CELL_BACKWATER = 1U << 5U,
};

struct BasinSolution {
    BasinKey key;
    double originX = 0.0;
    double originZ = 0.0;
    std::vector<float> surface;
    std::vector<float> waterSurface;
    std::vector<float> discharge;
    std::vector<float> sediment;
    std::vector<float> channelDistance;
    std::vector<float> channelWidth;
    std::vector<float> channelDepth;
    std::vector<float> channelGradient;
    std::vector<float> erosionDepth;
    std::vector<float> lakeDepth;
    std::vector<float> lakeShoreDistance;
    std::vector<float> shoreWaterSurface;
    std::vector<float> flowX;
    std::vector<float> flowZ;
    std::vector<uint8_t> streamOrder;
    std::vector<uint8_t> flags;
    std::vector<uint8_t> outletTypes;
    std::vector<ChannelGuide> guides;
    double namedOutletX = 0.0;
    double namedOutletZ = 0.0;
    std::vector<DeltaBranch> deltaBranches;
    std::vector<OutletFall> outletFalls;
    std::vector<int32_t> outletFallByCell;

    size_t byteSize() const {
        size_t total = sizeof(*this) + deltaBranches.capacity() * sizeof(DeltaBranch) +
                       outletFalls.capacity() * sizeof(OutletFall) +
                       guides.capacity() * sizeof(ChannelGuide);
        total += surface.capacity() * sizeof(float);
        total += waterSurface.capacity() * sizeof(float);
        total += discharge.capacity() * sizeof(float);
        total += sediment.capacity() * sizeof(float);
        total += channelDistance.capacity() * sizeof(float);
        total += channelWidth.capacity() * sizeof(float);
        total += channelDepth.capacity() * sizeof(float);
        total += channelGradient.capacity() * sizeof(float);
        total += erosionDepth.capacity() * sizeof(float);
        total += lakeDepth.capacity() * sizeof(float);
        total += lakeShoreDistance.capacity() * sizeof(float);
        total += shoreWaterSurface.capacity() * sizeof(float);
        total += flowX.capacity() * sizeof(float);
        total += flowZ.capacity() * sizeof(float);
        total += outletFallByCell.capacity() * sizeof(int32_t);
        total += streamOrder.capacity() * sizeof(uint8_t);
        total += flags.capacity() * sizeof(uint8_t);
        total += outletTypes.capacity() * sizeof(uint8_t);
        return total;
    }
};

int indexOf(int x, int z) {
    return z * RASTER_EDGE + x;
}

bool inRaster(int x, int z) {
    return x >= 0 && x < RASTER_EDGE && z >= 0 && z < RASTER_EDGE;
}

bool lockedBoundary(int x, int z) {
    return x <= RASTER_APRON || z <= RASTER_APRON || x >= RASTER_APRON + INTERIOR_INTERVALS ||
           z >= RASTER_APRON + INTERIOR_INTERVALS;
}

bool inSampledCatchment(int x, int z) {
    return x >= RASTER_APRON && x <= RASTER_APRON + INTERIOR_INTERVALS && z >= RASTER_APRON &&
           z <= RASTER_APRON + INTERIOR_INTERVALS;
}

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double smoothstep(double edge0, double edge1, double value) {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    const double t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

int64_t floorToCell(double coordinate) {
    constexpr double MIN_CELL = static_cast<double>(std::numeric_limits<int64_t>::min()) + 1.0;
    constexpr double MAX_CELL = static_cast<double>(std::numeric_limits<int64_t>::max()) - 1.0;
    return static_cast<int64_t>(
        std::floor(std::clamp(coordinate / BASIN_CATCHMENT_EDGE, MIN_CELL, MAX_CELL)));
}

bool sameCell(const Site& first, const Site& second) {
    return first.cell == second.cell;
}

bool siteLower(const Site& candidate, const Site& current) {
    // A pairwise epsilon comparison is not transitive and can create a ring
    // of individually "lower" catchments. Quantized potential followed by
    // immutable tie and coordinates is one strict total order, so every
    // downstream step decreases and the macro graph cannot cycle.
    auto drainageBand = [](double elevation) {
        if (!std::isfinite(elevation)) {
            return std::signbit(elevation) ? std::numeric_limits<int64_t>::min()
                                           : std::numeric_limits<int64_t>::max();
        }
        const double scaled = std::clamp(std::floor(elevation * 100.0), -9.0e15, 9.0e15);
        return static_cast<int64_t>(scaled);
    };
    const int64_t candidateBand = drainageBand(candidate.elevation);
    const int64_t currentBand = drainageBand(current.elevation);
    if (candidateBand != currentBand) return candidateBand < currentBand;
    if (candidate.tie != current.tie) return candidate.tie < current.tie;
    if (candidate.cell.x != current.cell.x) return candidate.cell.x < current.cell.x;
    return candidate.cell.z < current.cell.z;
}

double distanceToSegment(double px, double pz, double ax, double az, double bx, double bz,
                         double& along) {
    const double dx = bx - ax;
    const double dz = bz - az;
    const double lengthSquared = dx * dx + dz * dz;
    if (lengthSquared < 1.0e-12) {
        along = 0.0;
        return std::hypot(px - ax, pz - az);
    }
    along = std::clamp(((px - ax) * dx + (pz - az) * dz) / lengthSquared, 0.0, 1.0);
    return std::hypot(px - (ax + dx * along), pz - (az + dz * along));
}

double distanceToGuide(const ChannelGuide& guide, double worldX, double worldZ, double& along) {
    double bestDistanceSquared = std::numeric_limits<double>::max();
    along = 0.0;
    for (int segment = 0; segment < CHANNEL_GUIDE_CURVE_SEGMENTS; ++segment) {
        const GuidePoint& start = guide.curve[segment];
        const GuidePoint& end = guide.curve[segment + 1];
        const double directionX = end.x - start.x;
        const double directionZ = end.z - start.z;
        const double lengthSquared = directionX * directionX + directionZ * directionZ;
        double segmentAlong = 0.0;
        if (lengthSquared > 1.0e-12) {
            segmentAlong = std::clamp(
                ((worldX - start.x) * directionX + (worldZ - start.z) * directionZ) / lengthSquared,
                0.0, 1.0);
        }
        const double offsetX = worldX - (start.x + directionX * segmentAlong);
        const double offsetZ = worldZ - (start.z + directionZ * segmentAlong);
        const double distanceSquared = offsetX * offsetX + offsetZ * offsetZ;
        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            along = (segment + segmentAlong) / CHANNEL_GUIDE_CURVE_SEGMENTS;
        }
    }
    return std::sqrt(bestDistanceSquared);
}

float bilerp(const std::vector<float>& field, double gridX, double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double fx = std::clamp(gridX - x0, 0.0, 1.0);
    const double fz = std::clamp(gridZ - z0, 0.0, 1.0);
    const double north = field[indexOf(x0, z0)] * (1.0 - fx) + field[indexOf(x0 + 1, z0)] * fx;
    const double south =
        field[indexOf(x0, z0 + 1)] * (1.0 - fx) + field[indexOf(x0 + 1, z0 + 1)] * fx;
    return static_cast<float>(north * (1.0 - fz) + south * fz);
}

uint8_t nearestByte(const std::vector<uint8_t>& field, double gridX, double gridZ) {
    const int x = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int z = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    return field[indexOf(x, z)];
}

float nearestFloat(const std::vector<float>& field, double gridX, double gridZ) {
    const int x = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int z = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    return field[indexOf(x, z)];
}

struct ConditionedSample {
    double value = 0.0;
    double weight = 0.0;
};

struct LakeBodySample {
    ConditionedSample depth;
    double waterLevel = 0.0;
    bool endorheic = false;
    bool found = false;
};

ConditionedSample sampleLakeBodyField(const BasinSolution& solution,
                                      const std::vector<float>& field, double gridX, double gridZ,
                                      double waterLevel) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double fx = std::clamp(gridX - x0, 0.0, 1.0);
    const double fz = std::clamp(gridZ - z0, 0.0, 1.0);
    const std::array<int, 4> indices = {
        indexOf(x0, z0),
        indexOf(x0 + 1, z0),
        indexOf(x0, z0 + 1),
        indexOf(x0 + 1, z0 + 1),
    };
    const std::array<double, 4> weights = {
        (1.0 - fx) * (1.0 - fz),
        fx * (1.0 - fz),
        (1.0 - fx) * fz,
        fx * fz,
    };
    ConditionedSample result;
    for (size_t index = 0; index < indices.size(); ++index) {
        const int cell = indices[index];
        if ((solution.flags[cell] & CELL_LAKE) == 0 ||
            std::abs(solution.waterSurface[cell] - waterLevel) > 1.0e-4F) {
            continue;
        }
        result.value += field[cell] * weights[index];
        result.weight += weights[index];
    }
    return result;
}

LakeBodySample sampleDominantLakeBody(const BasinSolution& solution, double gridX, double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double fx = std::clamp(gridX - x0, 0.0, 1.0);
    const double fz = std::clamp(gridZ - z0, 0.0, 1.0);
    const std::array<int, 4> indices = {
        indexOf(x0, z0),
        indexOf(x0 + 1, z0),
        indexOf(x0, z0 + 1),
        indexOf(x0 + 1, z0 + 1),
    };
    const std::array<double, 4> weights = {
        (1.0 - fx) * (1.0 - fz),
        fx * (1.0 - fz),
        (1.0 - fx) * fz,
        fx * fz,
    };
    struct Candidate {
        double waterLevel = 0.0;
        double membership = 0.0;
        bool endorheic = false;
    };
    std::array<Candidate, 4> candidates{};
    size_t candidateCount = 0;
    for (size_t index = 0; index < indices.size(); ++index) {
        const int cell = indices[index];
        if ((solution.flags[cell] & CELL_LAKE) == 0 || weights[index] <= 0.0) continue;
        const double waterLevel = solution.waterSurface[cell];
        const bool endorheic = (solution.flags[cell] & CELL_ENDORHEIC) != 0;
        size_t candidate = 0;
        while (candidate < candidateCount &&
               (std::abs(candidates[candidate].waterLevel - waterLevel) > 1.0e-4 ||
                candidates[candidate].endorheic != endorheic)) {
            ++candidate;
        }
        if (candidate == candidateCount) {
            candidates[candidateCount++] = {
                .waterLevel = waterLevel,
                .membership = 0.0,
                .endorheic = endorheic,
            };
        }
        candidates[candidate].membership += weights[index];
    }
    if (candidateCount == 0) return {};
    const auto selected = std::max_element(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidateCount),
        [](const Candidate& first, const Candidate& second) {
            if (first.membership != second.membership) {
                return first.membership < second.membership;
            }
            return first.waterLevel > second.waterLevel;
        });
    return {
        .depth =
            sampleLakeBodyField(solution, solution.lakeDepth, gridX, gridZ, selected->waterLevel),
        .waterLevel = selected->waterLevel,
        .endorheic = selected->endorheic,
        .found = true,
    };
}

Site makeSite(const CounterRng& random, BasinKey cell,
              const BasinSolver::ElevationFunction& elevation,
              const BasinSolver::RainfallFunction& rainfall) {
    Site site;
    site.cell = cell;
    const double jitterX =
        0.16 + random.uniform01(SITE_POSITION_STREAM, cell.x, 0, cell.z, 0) * 0.68;
    const double jitterZ =
        0.16 + random.uniform01(SITE_POSITION_STREAM, cell.x, 0, cell.z, 1) * 0.68;
    site.x = (static_cast<double>(cell.x) + jitterX) * BASIN_CATCHMENT_EDGE;
    site.z = (static_cast<double>(cell.z) + jitterZ) * BASIN_CATCHMENT_EDGE;
    site.elevation = elevation(site.x, site.z);
    site.rainfall = rainfall(site.x, site.z, site.elevation);
    site.tie = random.u64(SITE_PROPERTY_STREAM, cell.x, 0, cell.z);
    site.ocean = site.elevation < SEA_LEVEL - 2.0;
    return site;
}

Site downstreamSite(const CounterRng& random, const Site& site,
                    const BasinSolver::ElevationFunction& elevation,
                    const BasinSolver::RainfallFunction& rainfall) {
    if (site.ocean) return site;
    Site best = site;
    constexpr std::array<int, 4> CARDINAL = {0, 2, 4, 6};
    for (int direction : CARDINAL) {
        const BasinKey neighbor{site.cell.x + DX[direction], site.cell.z + DZ[direction]};
        Site candidate = makeSite(random, neighbor, elevation, rainfall);
        if (siteLower(candidate, best)) best = candidate;
    }
    return best;
}

Portal sharedPortal(const Site& first, const Site& second) {
    Portal portal;
    const double dx = second.x - first.x;
    const double dz = second.z - first.z;
    if (first.cell.x != second.cell.x) {
        const int64_t eastCell = std::max(first.cell.x, second.cell.x);
        portal.x = static_cast<double>(eastCell) * BASIN_CATCHMENT_EDGE;
        const double along = std::abs(dx) > 1.0e-9 ? (portal.x - first.x) / dx : 0.5;
        portal.z = first.z + dz * std::clamp(along, 0.0, 1.0);
        const double south = static_cast<double>(first.cell.z) * BASIN_CATCHMENT_EDGE;
        portal.z = std::clamp(portal.z, south + BASIN_CATCHMENT_EDGE * 0.08,
                              south + BASIN_CATCHMENT_EDGE * 0.92);
    } else {
        const int64_t southCell = std::max(first.cell.z, second.cell.z);
        portal.z = static_cast<double>(southCell) * BASIN_CATCHMENT_EDGE;
        const double along = std::abs(dz) > 1.0e-9 ? (portal.z - first.z) / dz : 0.5;
        portal.x = first.x + dx * std::clamp(along, 0.0, 1.0);
        const double west = static_cast<double>(first.cell.x) * BASIN_CATCHMENT_EDGE;
        portal.x = std::clamp(portal.x, west + BASIN_CATCHMENT_EDGE * 0.08,
                              west + BASIN_CATCHMENT_EDGE * 0.92);
    }
    return portal;
}

double siteContribution(const Site& site) {
    const double squareKilometers = BASIN_CATCHMENT_EDGE * BASIN_CATCHMENT_EDGE / 1'000'000.0;
    return std::max(1.0, site.rainfall * squareKilometers);
}

double channelGuideIncision(double discharge) {
    const double dischargeRelief = smoothstep(MIN_CHANNEL_DISCHARGE, 4'800.0, discharge);
    return 1.2 + std::sqrt(discharge) * (0.012 + dischargeRelief * 0.006);
}

int channelGuideOrder(double discharge) {
    return std::clamp(1 + static_cast<int>(std::floor(std::log2(discharge / 500.0 + 1.0))), 1, 6);
}

double channelGuideDepth(double discharge) {
    return std::clamp(1.2 + std::sqrt(discharge) * 0.065 + channelGuideOrder(discharge) * 0.65, 1.8,
                      14.0);
}

class MacroContext {
public:
    MacroContext(const CounterRng& randomValue,
                 const BasinSolver::ElevationFunction& elevationValue,
                 const BasinSolver::RainfallFunction& rainfallValue)
        : random(randomValue)
        , elevation(elevationValue)
        , rainfall(rainfallValue) {}

    Site site(BasinKey key) {
        auto found = sites.find(key);
        if (found != sites.end()) return found->second;
        Site value = makeSite(random, key, elevation, rainfall);
        sites.emplace(key, value);
        return value;
    }

    Site downstream(const Site& value) {
        auto found = downstreamCells.find(value.cell);
        if (found != downstreamCells.end()) return site(found->second);
        Site best = value;
        if (!value.ocean) {
            constexpr std::array<int, 4> CARDINAL = {0, 2, 4, 6};
            for (int direction : CARDINAL) {
                Site candidate = site({value.cell.x + DX[direction], value.cell.z + DZ[direction]});
                if (siteLower(candidate, best)) best = candidate;
            }
        }
        downstreamCells.emplace(value.cell, best.cell);
        return best;
    }

    double discharge(const Site& target) {
        auto found = discharges.find(target.cell);
        if (found != discharges.end()) return found->second;
        double exact = siteContribution(target);
        int contributors = 1;
        for (int dz = -4; dz <= 4; ++dz) {
            for (int dx = -4; dx <= 4; ++dx) {
                if ((dx == 0 && dz == 0) || std::abs(dx) + std::abs(dz) > 4) continue;
                Site candidate = site({target.cell.x + dx, target.cell.z + dz});
                const int maxSteps = std::abs(dx) + std::abs(dz) + 1;
                for (int step = 0; step < maxSteps; ++step) {
                    Site next = downstream(candidate);
                    if (sameCell(next, target)) {
                        exact += siteContribution(candidate);
                        ++contributors;
                        break;
                    }
                    if (sameCell(next, candidate) || siteLower(target, next)) break;
                    candidate = next;
                }
            }
        }
        const double coarseFraction =
            0.18 +
            random.uniform01(SITE_PROPERTY_STREAM, target.cell.x, 0, target.cell.z, 7) * 0.34;
        const double result =
            exact + siteContribution(target) * coarseFraction * std::sqrt(contributors);
        discharges.emplace(target.cell, result);
        return result;
    }

    TerminalLake terminalLake(const Site& target) {
        float saddle = std::numeric_limits<float>::max();
        const double west = static_cast<double>(target.cell.x) * BASIN_CATCHMENT_EDGE;
        const double north = static_cast<double>(target.cell.z) * BASIN_CATCHMENT_EDGE;
        constexpr int BORDER_SAMPLES = 9;
        for (int sample = 0; sample < BORDER_SAMPLES; ++sample) {
            const double amount = static_cast<double>(sample) / (BORDER_SAMPLES - 1);
            const double x = west + BASIN_CATCHMENT_EDGE * amount;
            const double z = north + BASIN_CATCHMENT_EDGE * amount;
            saddle = std::min(saddle, static_cast<float>(elevation(west, z)));
            saddle =
                std::min(saddle, static_cast<float>(elevation(west + BASIN_CATCHMENT_EDGE, z)));
            saddle = std::min(saddle, static_cast<float>(elevation(x, north)));
            saddle =
                std::min(saddle, static_cast<float>(elevation(x, north + BASIN_CATCHMENT_EDGE)));
        }
        TerminalLake lake;
        lake.x = target.x;
        lake.z = target.z;
        constexpr double MINIMUM_DEPTH = 1.25;
        constexpr double MAXIMUM_RISE = 18.0;
        const double candidate =
            std::min(static_cast<double>(saddle) - 0.5, target.elevation + MAXIMUM_RISE);
        lake.standing = candidate >= target.elevation + MINIMUM_DEPTH;
        lake.surface = lake.standing ? candidate : target.elevation - 0.35;
        lake.radius = lake.standing ? std::clamp(110.0 + (lake.surface - target.elevation) * 26.0,
                                                 110.0, 360.0)
                                    : 0.0;
        return lake;
    }

    PortalProfile portalProfile(const Site& upstream) {
        if (const auto found = portalProfiles.find(upstream.cell); found != portalProfiles.end()) {
            return found->second;
        }

        std::vector<Site> path;
        path.reserve(32);
        std::unordered_set<BasinKey, BasinKeyHash> visited;
        visited.reserve(32);
        Site current = upstream;
        double receiverWater = SEA_LEVEL;
        bool resolved = false;
        for (int edge = 0; edge < PORTAL_PROFILE_MAX_EDGES; ++edge) {
            if (!visited.insert(current.cell).second) {
                throw std::runtime_error("canonical portal profile contains a drainage cycle");
            }
            if (const auto found = portalProfiles.find(current.cell);
                found != portalProfiles.end()) {
                receiverWater = found->second.water;
                resolved = true;
                break;
            }
            const Site next = downstream(current);
            if (sameCell(next, current)) {
                receiverWater = current.ocean ? static_cast<double>(SEA_LEVEL)
                                              : std::max(static_cast<double>(SEA_LEVEL),
                                                         terminalLake(current).surface);
                resolved = true;
                break;
            }
            path.push_back(current);
            if (next.ocean) {
                receiverWater = SEA_LEVEL;
                resolved = true;
                break;
            }
            current = next;
        }
        if (!resolved) {
            throw std::runtime_error("canonical portal profile exceeded its bounded horizon");
        }

        for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
            const Site edgeUpstream = *iterator;
            const Site edgeDownstream = downstream(edgeUpstream);
            const Portal portal = sharedPortal(edgeUpstream, edgeDownstream);
            const double fullLength = std::max(1.0, std::hypot(edgeDownstream.x - edgeUpstream.x,
                                                               edgeDownstream.z - edgeUpstream.z));
            const double portalAlong = std::clamp(
                std::hypot(portal.x - edgeUpstream.x, portal.z - edgeUpstream.z) / fullLength, 0.0,
                1.0);
            const double preferred =
                edgeUpstream.elevation +
                (edgeDownstream.elevation - edgeUpstream.elevation) * portalAlong - 0.35;
            const double edgeDischarge = discharge(edgeUpstream);
            double minimumCorridorTerrain = std::numeric_limits<double>::max();
            const double approachLength =
                std::max(1.0, std::hypot(portal.x - edgeUpstream.x, portal.z - edgeUpstream.z));
            const double directionX = (portal.x - edgeUpstream.x) / approachLength;
            const double directionZ = (portal.z - edgeUpstream.z) / approachLength;
            for (int sample = 0; sample < PORTAL_PROFILE_TERRAIN_SAMPLES; ++sample) {
                const double amount = static_cast<double>(sample) /
                                      static_cast<double>(PORTAL_PROFILE_TERRAIN_SAMPLES - 1);
                const double centerX = edgeUpstream.x + (portal.x - edgeUpstream.x) * amount;
                const double centerZ = edgeUpstream.z + (portal.z - edgeUpstream.z) * amount;
                for (const double lateral : {-256.0, -128.0, 0.0, 128.0, 256.0}) {
                    const double sampleX = centerX - directionZ * lateral;
                    const double sampleZ = centerZ + directionX * lateral;
                    minimumCorridorTerrain =
                        std::min(minimumCorridorTerrain, elevation(sampleX, sampleZ));
                }
            }
            const double terrainUpper = std::clamp(
                std::min(elevation(portal.x, portal.z) - 0.35,
                         minimumCorridorTerrain - channelGuideDepth(edgeDischarge) * 0.5),
                static_cast<double>(WORLD_MIN_Y + 2), static_cast<double>(WORLD_MAX_Y));

            PortalProfile profile;
            profile.downstreamWater = receiverWater;
            profile.terrainUpper = terrainUpper;
            if (edgeDownstream.ocean) {
                profile.water = SEA_LEVEL;
                profile.backwater = terrainUpper + 0.01 < SEA_LEVEL;
            } else if (receiverWater <= terrainUpper && preferred >= receiverWater) {
                profile.water = std::clamp(preferred, receiverWater, terrainUpper);
            } else {
                profile.water = receiverWater;
                profile.backwater = true;
            }
            profile.profileFall = profile.water >= profile.downstreamWater + 2.5;
            portalProfiles[edgeUpstream.cell] = profile;
            receiverWater = profile.water;
        }
        return portalProfiles.at(upstream.cell);
    }

private:
    const CounterRng& random;
    const BasinSolver::ElevationFunction& elevation;
    const BasinSolver::RainfallFunction& rainfall;
    std::unordered_map<BasinKey, Site, BasinKeyHash> sites;
    std::unordered_map<BasinKey, BasinKey, BasinKeyHash> downstreamCells;
    std::unordered_map<BasinKey, double, BasinKeyHash> discharges;
    std::unordered_map<BasinKey, PortalProfile, BasinKeyHash> portalProfiles;
};

void sampleInputs(const BasinSolution& solution, const BasinSolver::ElevationFunction& elevation,
                  const BasinSolver::RainfallFunction& rainfall,
                  const BasinSolver::RockResistanceFunction& rockResistance,
                  std::vector<float>& raw, std::vector<float>& rain,
                  std::vector<float>& resistance) {
    std::vector<int> coarseCoordinates;
    for (int coordinate = 0; coordinate < RASTER_EDGE - 1; coordinate += INPUT_SAMPLE_STRIDE) {
        coarseCoordinates.push_back(coordinate);
    }
    if (coarseCoordinates.back() != RASTER_EDGE - 1) {
        coarseCoordinates.push_back(RASTER_EDGE - 1);
    }
    const int coarseEdge = static_cast<int>(coarseCoordinates.size());
    std::vector<float> coarseRaw(static_cast<size_t>(coarseEdge * coarseEdge));
    std::vector<float> coarseRain(static_cast<size_t>(coarseEdge * coarseEdge));
    std::vector<float> coarseResistance(static_cast<size_t>(coarseEdge * coarseEdge));
    for (int cz = 0; cz < coarseEdge; ++cz) {
        for (int cx = 0; cx < coarseEdge; ++cx) {
            const int gx = coarseCoordinates[cx];
            const int gz = coarseCoordinates[cz];
            const double worldX = solution.originX + (gx - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (gz - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double height = elevation(worldX, worldZ);
            const size_t index = static_cast<size_t>(cz * coarseEdge + cx);
            coarseRaw[index] = static_cast<float>(height);
            coarseRain[index] = static_cast<float>(rainfall(worldX, worldZ, height));
            coarseResistance[index] =
                static_cast<float>(std::clamp(rockResistance(worldX, worldZ), 0.15, 1.75));
        }
    }

    auto interpolate = [&](const std::vector<float>& source, int gx, int gz) {
        const auto upperX =
            std::upper_bound(coarseCoordinates.begin(), coarseCoordinates.end(), gx);
        const auto upperZ =
            std::upper_bound(coarseCoordinates.begin(), coarseCoordinates.end(), gz);
        const int cx1 =
            std::clamp(static_cast<int>(upperX - coarseCoordinates.begin()), 1, coarseEdge - 1);
        const int cz1 =
            std::clamp(static_cast<int>(upperZ - coarseCoordinates.begin()), 1, coarseEdge - 1);
        const int cx0 = cx1 - 1;
        const int cz0 = cz1 - 1;
        const double fx = static_cast<double>(gx - coarseCoordinates[cx0]) /
                          (coarseCoordinates[cx1] - coarseCoordinates[cx0]);
        const double fz = static_cast<double>(gz - coarseCoordinates[cz0]) /
                          (coarseCoordinates[cz1] - coarseCoordinates[cz0]);
        const double north = source[static_cast<size_t>(cz0 * coarseEdge + cx0)] * (1.0 - fx) +
                             source[static_cast<size_t>(cz0 * coarseEdge + cx1)] * fx;
        const double south = source[static_cast<size_t>(cz1 * coarseEdge + cx0)] * (1.0 - fx) +
                             source[static_cast<size_t>(cz1 * coarseEdge + cx1)] * fx;
        return static_cast<float>(north * (1.0 - fz) + south * fz);
    };

    raw.resize(RASTER_CELLS);
    rain.resize(RASTER_CELLS);
    resistance.resize(RASTER_CELLS);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            raw[index] = interpolate(coarseRaw, x, z);
            rain[index] = interpolate(coarseRain, x, z);
            resistance[index] = interpolate(coarseResistance, x, z);
        }
    }
}

ChannelGuide makeChannelGuide(const CounterRng& random, const Site& upstream,
                              const Site& downstream, const Portal& portal,
                              const PortalProfile& profile, double discharge, bool outgoing) {
    const double fullLength =
        std::max(1.0, std::hypot(downstream.x - upstream.x, downstream.z - upstream.z));
    ChannelGuide guide;
    guide.startX = outgoing ? upstream.x : portal.x;
    guide.startZ = outgoing ? upstream.z : portal.z;
    guide.endX = outgoing ? portal.x : downstream.x;
    guide.endZ = outgoing ? portal.z : downstream.z;
    guide.discharge = discharge;
    guide.portalX = portal.x;
    guide.portalZ = portal.z;
    guide.portalWater = profile.water;
    guide.downstreamWater = profile.downstreamWater;
    guide.terrainUpper = profile.terrainUpper;
    guide.gradient = std::max(0.0, upstream.elevation - downstream.elevation) / fullLength;
    guide.portalFlowX = (downstream.x - upstream.x) / fullLength;
    guide.portalFlowZ = (downstream.z - upstream.z) / fullLength;
    const double segmentX = guide.endX - guide.startX;
    const double segmentZ = guide.endZ - guide.startZ;
    const double segmentLength = std::hypot(segmentX, segmentZ);
    const int directionX = static_cast<int>(downstream.cell.x - upstream.cell.x);
    const int directionZ = static_cast<int>(downstream.cell.z - upstream.cell.z);
    const int32_t segmentAddress = (outgoing ? 0 : 1) * 9 + (directionX + 1) * 3 + directionZ + 1;
    const double amplitudeRandom = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x,
                                                    segmentAddress, upstream.cell.z, 0);
    const double phase = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x, segmentAddress,
                                          upstream.cell.z, 1) *
                         2.0 * std::numbers::pi;
    const double secondaryPhase = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x,
                                                   segmentAddress, upstream.cell.z, 2) *
                                  2.0 * std::numbers::pi;
    // A lateral cut through nearly level relief can create an artificial spill basin. Reserve
    // full guide displacement for slopes that keep the routed water surface monotonic.
    const double meanderStrength = smoothstep(0.0045, 0.012, guide.gradient);
    const double amplitude =
        std::min(144.0, segmentLength * (0.075 + amplitudeRandom * 0.055)) * meanderStrength;
    const double bendDirection = amplitudeRandom < 0.5 ? -1.0 : 1.0;
    const double inverseLength = segmentLength > 1.0e-9 ? 1.0 / segmentLength : 0.0;
    const double normalX = -segmentZ * inverseLength;
    const double normalZ = segmentX * inverseLength;
    for (int point = 0; point <= CHANNEL_GUIDE_CURVE_SEGMENTS; ++point) {
        const double amount = static_cast<double>(point) / CHANNEL_GUIDE_CURVE_SEGMENTS;
        const double envelope = std::pow(std::sin(std::numbers::pi * amount), 2.0);
        const double wave = bendDirection * 0.72 +
                            std::sin(2.0 * std::numbers::pi * amount + phase) * 0.20 +
                            std::sin(4.0 * std::numbers::pi * amount + secondaryPhase) * 0.08;
        const double lateralOffset = amplitude * envelope * wave;
        guide.curve[point] = {
            .x = guide.startX + segmentX * amount + normalX * lateralOffset,
            .z = guide.startZ + segmentZ * amount + normalZ * lateralOffset,
        };
    }
    guide.curve.front() = {guide.startX, guide.startZ};
    guide.curve.back() = {guide.endX, guide.endZ};
    guide.widthPhase = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x, segmentAddress,
                                        upstream.cell.z, 3) *
                       2.0 * std::numbers::pi;
    guide.widthVariation = (0.08 + random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x,
                                                    segmentAddress, upstream.cell.z, 4) *
                                       0.08) *
                           meanderStrength;
    guide.outgoing = outgoing;
    guide.oceanMouth = downstream.ocean;
    guide.backwater = profile.backwater;
    guide.profileFall = profile.profileFall;
    return guide;
}

std::vector<ChannelGuide> channelGuides(const CounterRng& random, MacroContext& macro,
                                        const Site& center, const Site& downstream) {
    std::vector<ChannelGuide> guides;
    if (!sameCell(center, downstream)) {
        const Portal portal = sharedPortal(center, downstream);
        guides.push_back(makeChannelGuide(random, center, downstream, portal,
                                          macro.portalProfile(center), macro.discharge(center),
                                          true));
    }
    constexpr std::array<int, 4> CARDINAL = {0, 2, 4, 6};
    for (int direction : CARDINAL) {
        const BasinKey neighborCell{center.cell.x + DX[direction], center.cell.z + DZ[direction]};
        const Site neighbor = macro.site(neighborCell);
        const Site neighborDownstream = macro.downstream(neighbor);
        if (!sameCell(neighborDownstream, center)) continue;
        const Portal portal = sharedPortal(neighbor, center);
        guides.push_back(makeChannelGuide(random, neighbor, center, portal,
                                          macro.portalProfile(neighbor), macro.discharge(neighbor),
                                          false));
    }
    return guides;
}

void carveChannelGuides(const BasinSolution& solution, const std::vector<ChannelGuide>& guides,
                        std::vector<float>& raw) {
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            if (lockedBoundary(x, z)) continue;
            const double worldX = solution.originX + (x - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (z - RASTER_APRON) * BASIN_RASTER_SPACING;
            double guideCut = 0.0;
            for (const ChannelGuide& guide : guides) {
                double along = 0.0;
                const double distance = distanceToGuide(guide, worldX, worldZ, along);
                const double widthEnvelope = std::pow(std::sin(std::numbers::pi * along), 2.0);
                const double widthScale =
                    1.0 + guide.widthVariation * widthEnvelope *
                              std::sin(4.0 * std::numbers::pi * along + guide.widthPhase);
                const double width =
                    std::clamp((14.0 + std::sqrt(guide.discharge) * 0.10) * widthScale, 18.0, 68.0);
                const double mask = 1.0 - smoothstep(width, width * 3.2, distance);
                const double incision = channelGuideIncision(guide.discharge);
                guideCut = std::max(guideCut, mask * incision);
            }
            raw[indexOf(x, z)] -= static_cast<float>(guideCut);
        }
    }
}

std::vector<uint8_t> buildOutletConstraints(BasinSolution& solution, const Site& center,
                                            const Site& downstream,
                                            const std::vector<ChannelGuide>& guides,
                                            const std::vector<float>& raw) {
    std::vector<uint8_t> terminals(RASTER_CELLS, static_cast<uint8_t>(BasinOutlet::NONE));
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (raw[index] < SEA_LEVEL &&
            inSampledCatchment(index % RASTER_EDGE, index / RASTER_EDGE)) {
            terminals[index] = static_cast<uint8_t>(BasinOutlet::OCEAN);
        }
    }

    if (!sameCell(center, downstream)) {
        const auto outgoing = std::find_if(
            guides.begin(), guides.end(), [](const ChannelGuide& guide) { return guide.outgoing; });
        if (outgoing != guides.end()) {
            const int gridX =
                std::clamp(static_cast<int>(std::lround((outgoing->portalX - solution.originX) /
                                                        BASIN_RASTER_SPACING)) +
                               RASTER_APRON,
                           RASTER_APRON, RASTER_APRON + INTERIOR_INTERVALS);
            const int gridZ =
                std::clamp(static_cast<int>(std::lround((outgoing->portalZ - solution.originZ) /
                                                        BASIN_RASTER_SPACING)) +
                               RASTER_APRON,
                           RASTER_APRON, RASTER_APRON + INTERIOR_INTERVALS);
            const int portalIndex = indexOf(gridX, gridZ);
            if (terminals[portalIndex] != static_cast<uint8_t>(BasinOutlet::OCEAN)) {
                terminals[portalIndex] = static_cast<uint8_t>(BasinOutlet::SHARED_PORTAL);
            }
            solution.namedOutletX = outgoing->portalX;
            solution.namedOutletZ = outgoing->portalZ;
        }
    } else if (!center.ocean) {
        const int centerX = std::clamp(
            static_cast<int>(std::lround((center.x - solution.originX) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            RASTER_APRON + 1, RASTER_APRON + INTERIOR_INTERVALS - 1);
        const int centerZ = std::clamp(
            static_cast<int>(std::lround((center.z - solution.originZ) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            RASTER_APRON + 1, RASTER_APRON + INTERIOR_INTERVALS - 1);
        int sink = indexOf(centerX, centerZ);
        for (int z = std::max(RASTER_APRON + 1, centerZ - 3);
             z <= std::min(RASTER_APRON + INTERIOR_INTERVALS - 1, centerZ + 3); ++z) {
            for (int x = std::max(RASTER_APRON + 1, centerX - 3);
                 x <= std::min(RASTER_APRON + INTERIOR_INTERVALS - 1, centerX + 3); ++x) {
                const int candidate = indexOf(x, z);
                if (raw[candidate] < raw[sink] ||
                    (raw[candidate] == raw[sink] && candidate < sink)) {
                    sink = candidate;
                }
            }
        }
        terminals[sink] = static_cast<uint8_t>(BasinOutlet::ENDORHEIC);
        const int sinkX = sink % RASTER_EDGE;
        const int sinkZ = sink / RASTER_EDGE;
        solution.namedOutletX = solution.originX + (sinkX - RASTER_APRON) * BASIN_RASTER_SPACING;
        solution.namedOutletZ = solution.originZ + (sinkZ - RASTER_APRON) * BASIN_RASTER_SPACING;
    }
    return terminals;
}

void priorityFlood(const std::vector<float>& surface, const std::vector<uint8_t>& terminals,
                   std::vector<float>& filled, std::vector<uint32_t>& floodOrder) {
    using QueueValue = std::pair<float, int32_t>;
    std::priority_queue<QueueValue, std::vector<QueueValue>, std::greater<>> queue;
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    filled = surface;
    floodOrder.assign(RASTER_CELLS, std::numeric_limits<uint32_t>::max());
    uint32_t order = 0;
    auto seed = [&](int index) {
        if (visited[index] != 0) return;
        visited[index] = 1;
        floodOrder[index] = order++;
        queue.emplace(filled[index], index);
    };
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (terminals[index] != static_cast<uint8_t>(BasinOutlet::NONE)) seed(index);
    }

    while (!queue.empty()) {
        const auto [level, index] = queue.top();
        queue.pop();
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        for (int direction = 0; direction < 8; ++direction) {
            const int nx = x + DX[direction];
            const int nz = z + DZ[direction];
            if (!inSampledCatchment(nx, nz)) continue;
            const int neighbor = indexOf(nx, nz);
            if (visited[neighbor] != 0) continue;
            visited[neighbor] = 1;
            filled[neighbor] = std::max(surface[neighbor], level);
            floodOrder[neighbor] = order++;
            queue.emplace(filled[neighbor], neighbor);
        }
    }
}

bool routesLower(int from, int to, const std::vector<float>& filled,
                 const std::vector<uint32_t>& floodOrder) {
    if (to < 0) return false;
    if (filled[to] < filled[from]) return true;
    return filled[to] == filled[from] && floodOrder[to] < floodOrder[from];
}

void buildDInfinityRouting(const std::vector<float>& filled,
                           const std::vector<uint32_t>& floodOrder,
                           const std::vector<uint8_t>& terminals, std::vector<Receiver>& receivers,
                           std::vector<float>& flowX, std::vector<float>& flowZ,
                           std::vector<int32_t>& upstreamOrder) {
    receivers.assign(RASTER_CELLS, {});
    flowX.assign(RASTER_CELLS, 0.0F);
    flowZ.assign(RASTER_CELLS, 0.0F);
    upstreamOrder.resize(RASTER_CELLS);
    for (int index = 0; index < RASTER_CELLS; ++index)
        upstreamOrder[index] = index;
    std::sort(upstreamOrder.begin(), upstreamOrder.end(), [&](int lhs, int rhs) {
        if (filled[lhs] != filled[rhs]) return filled[lhs] > filled[rhs];
        if (floodOrder[lhs] != floodOrder[rhs]) return floodOrder[lhs] > floodOrder[rhs];
        return lhs > rhs;
    });

    for (int z = RASTER_APRON; z <= RASTER_APRON + INTERIOR_INTERVALS; ++z) {
        for (int x = RASTER_APRON; x <= RASTER_APRON + INTERIOR_INTERVALS; ++x) {
            const int index = indexOf(x, z);
            if (terminals[index] != static_cast<uint8_t>(BasinOutlet::NONE)) continue;
            const double gradientX = (filled[indexOf(x + 1, z)] - filled[indexOf(x - 1, z)]) * 0.5;
            const double gradientZ = (filled[indexOf(x, z + 1)] - filled[indexOf(x, z - 1)]) * 0.5;
            double angle = std::atan2(-gradientZ, -gradientX);
            if (angle < 0.0) angle += 2.0 * std::numbers::pi;
            const double sector = angle / (std::numbers::pi / 4.0);
            int firstDirection = static_cast<int>(std::floor(sector)) & 7;
            int secondDirection = (firstDirection + 1) & 7;
            double secondWeight = sector - std::floor(sector);
            int first = indexOf(x + DX[firstDirection], z + DZ[firstDirection]);
            int second = indexOf(x + DX[secondDirection], z + DZ[secondDirection]);
            const bool firstValid =
                inSampledCatchment(x + DX[firstDirection], z + DZ[firstDirection]) &&
                routesLower(index, first, filled, floodOrder);
            const bool secondValid =
                inSampledCatchment(x + DX[secondDirection], z + DZ[secondDirection]) &&
                routesLower(index, second, filled, floodOrder);

            if (!firstValid && !secondValid) {
                std::array<std::pair<double, int>, 8> candidates{};
                int candidateCount = 0;
                for (int direction = 0; direction < 8; ++direction) {
                    if (!inSampledCatchment(x + DX[direction], z + DZ[direction])) continue;
                    const int neighbor = indexOf(x + DX[direction], z + DZ[direction]);
                    if (!routesLower(index, neighbor, filled, floodOrder)) continue;
                    const double drop = filled[index] - filled[neighbor];
                    candidates[candidateCount++] = {drop / STEP_LENGTH[direction], direction};
                }
                if (candidateCount == 0) continue;
                std::sort(candidates.begin(), candidates.begin() + candidateCount,
                          [](const auto& lhs, const auto& rhs) {
                              if (lhs.first != rhs.first) return lhs.first > rhs.first;
                              return lhs.second < rhs.second;
                          });
                firstDirection = candidates[0].second;
                first = indexOf(x + DX[firstDirection], z + DZ[firstDirection]);
                second = -1;
                secondWeight = 0.0;
                if (candidateCount > 1 && candidates[1].first > candidates[0].first * 0.45) {
                    secondDirection = candidates[1].second;
                    second = indexOf(x + DX[secondDirection], z + DZ[secondDirection]);
                    secondWeight =
                        candidates[1].first / (candidates[0].first + candidates[1].first);
                }
            } else if (!firstValid) {
                first = second;
                firstDirection = secondDirection;
                second = -1;
                secondWeight = 0.0;
            } else if (!secondValid) {
                second = -1;
                secondWeight = 0.0;
            }

            Receiver& receiver = receivers[index];
            receiver.first = first;
            receiver.second = second;
            receiver.firstWeight = static_cast<float>(1.0 - secondWeight);
            receiver.secondWeight = static_cast<float>(secondWeight);
            double directionX = DX[firstDirection] * receiver.firstWeight;
            double directionZ = DZ[firstDirection] * receiver.firstWeight;
            if (second >= 0) {
                directionX += DX[secondDirection] * receiver.secondWeight;
                directionZ += DZ[secondDirection] * receiver.secondWeight;
            }
            const double magnitude = std::hypot(directionX, directionZ);
            if (magnitude > 1.0e-9) {
                flowX[index] = static_cast<float>(directionX / magnitude);
                flowZ[index] = static_cast<float>(directionZ / magnitude);
            }
        }
    }
}

std::vector<uint8_t> resolveOutletTypes(const std::vector<uint8_t>& terminals,
                                        const std::vector<Receiver>& receivers,
                                        const std::vector<int32_t>& upstreamOrder) {
    std::vector<uint8_t> result = terminals;
    for (auto iterator = upstreamOrder.rbegin(); iterator != upstreamOrder.rend(); ++iterator) {
        const int index = *iterator;
        if (result[index] != static_cast<uint8_t>(BasinOutlet::NONE)) continue;
        const Receiver& receiver = receivers[index];
        if (receiver.first >= 0) result[index] = result[receiver.first];
        if (result[index] == static_cast<uint8_t>(BasinOutlet::NONE) && receiver.second >= 0 &&
            receiver.secondWeight > 0.0F) {
            result[index] = result[receiver.second];
        }
    }
    return result;
}

void injectGuideDischarge(const BasinSolution& solution, const std::vector<ChannelGuide>& guides,
                          std::vector<float>& accumulation) {
    for (const ChannelGuide& guide : guides) {
        const double anchorX = guide.startX;
        const double anchorZ = guide.startZ;
        const int gridX = std::clamp(
            static_cast<int>(std::lround((anchorX - solution.originX) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            1, RASTER_EDGE - 2);
        const int gridZ = std::clamp(
            static_cast<int>(std::lround((anchorZ - solution.originZ) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            1, RASTER_EDGE - 2);
        accumulation[indexOf(gridX, gridZ)] += static_cast<float>(guide.discharge * 0.32);
    }
}

void accumulateFlow(const std::vector<float>& rain, const std::vector<Receiver>& receivers,
                    const std::vector<int32_t>& upstreamOrder, std::vector<float>& accumulation,
                    const BasinSolution& solution, const std::vector<ChannelGuide>& guides) {
    accumulation.resize(RASTER_CELLS);
    constexpr double CELL_SQUARE_KILOMETERS =
        BASIN_RASTER_SPACING * BASIN_RASTER_SPACING / 1'000'000.0;
    for (int index = 0; index < RASTER_CELLS; ++index) {
        accumulation[index] =
            static_cast<float>(std::max(0.001, rain[index] * CELL_SQUARE_KILOMETERS));
    }
    injectGuideDischarge(solution, guides, accumulation);
    for (int index : upstreamOrder) {
        const Receiver& receiver = receivers[index];
        if (receiver.first >= 0) {
            accumulation[receiver.first] += accumulation[index] * receiver.firstWeight;
        }
        if (receiver.second >= 0) {
            accumulation[receiver.second] += accumulation[index] * receiver.secondWeight;
        }
    }
}

void erodeTerrain(const std::vector<float>& base, const std::vector<float>& resistance,
                  const std::vector<Receiver>& receivers, const std::vector<int32_t>& upstreamOrder,
                  const std::vector<float>& accumulation, std::vector<float>& surface,
                  std::vector<float>& sediment) {
    surface = base;
    sediment.assign(RASTER_CELLS, 0.0F);
    std::vector<float> delta(RASTER_CELLS);
    std::vector<float> transported(RASTER_CELLS);
    for (int pass = 0; pass < EROSION_PASSES; ++pass) {
        std::fill(delta.begin(), delta.end(), 0.0F);
        std::fill(transported.begin(), transported.end(), 0.0F);
        for (int index : upstreamOrder) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            if (lockedBoundary(x, z)) continue;
            const int receiver = receivers[index].first;
            if (receiver < 0) continue;
            const int rx = receiver % RASTER_EDGE;
            const int rz = receiver / RASTER_EDGE;
            const double run = std::hypot(rx - x, rz - z) * BASIN_RASTER_SPACING;
            const double slope = std::max(0.0, (surface[index] - surface[receiver]) / run);
            const double capacity = std::pow(std::max(0.0F, accumulation[index]), 0.47) *
                                    std::sqrt(slope + 1.0e-5) *
                                    (0.90 / std::max(0.15F, resistance[index]));
            const double load = sediment[index] + transported[index];
            const double dischargeRelief =
                smoothstep(MIN_CHANNEL_DISCHARGE, 4'800.0, accumulation[index]);
            const double incisionLimit = 0.38 + dischargeRelief * 0.16;
            const double incisionRate = 0.045 + dischargeRelief * 0.013;
            const double incision =
                std::min(incisionLimit, std::max(0.0, capacity - load) * incisionRate);
            const double deposition = std::min(0.24, std::max(0.0, load - capacity) * 0.028);
            delta[index] += static_cast<float>(deposition - incision);
            const float moved = static_cast<float>(std::max(0.0, load + incision - deposition));
            transported[receiver] += moved * receivers[index].firstWeight;
            if (receivers[index].second >= 0) {
                transported[receivers[index].second] += moved * receivers[index].secondWeight;
            }
        }

        for (int z = RASTER_APRON + 1; z < RASTER_APRON + INTERIOR_INTERVALS; ++z) {
            for (int x = RASTER_APRON + 1; x < RASTER_APRON + INTERIOR_INTERVALS; ++x) {
                const int index = indexOf(x, z);
                double neighborMean = 0.0;
                for (int direction = 0; direction < 8; direction += 2) {
                    neighborMean += surface[indexOf(x + DX[direction], z + DZ[direction])];
                }
                neighborMean *= 0.25;
                const double difference = surface[index] - neighborMean;
                const double talus = 5.5 + resistance[index] * 5.0;
                if (std::abs(difference) > talus) {
                    delta[index] -= static_cast<float>(std::copysign(
                        std::min(0.20, (std::abs(difference) - talus) * 0.025), difference));
                }
            }
        }
        for (int index = 0; index < RASTER_CELLS; ++index) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            if (!lockedBoundary(x, z)) surface[index] += delta[index];
            sediment[index] = transported[index];
        }
    }
}

void computeStrahler(const std::vector<Receiver>& receivers,
                     const std::vector<int32_t>& upstreamOrder,
                     const std::vector<float>& accumulation, std::vector<uint8_t>& order) {
    order.assign(RASTER_CELLS, 0);
    std::vector<uint8_t> maximumIncoming(RASTER_CELLS, 0);
    std::vector<uint8_t> maximumCount(RASTER_CELLS, 0);
    for (int index : upstreamOrder) {
        if (accumulation[index] < MIN_CHANNEL_DISCHARGE) continue;
        uint8_t current = maximumIncoming[index];
        if (current == 0) {
            current = 1;
        } else if (maximumCount[index] >= 2 && current < std::numeric_limits<uint8_t>::max()) {
            ++current;
        }
        order[index] = current;
        const int receiver = receivers[index].first;
        if (receiver < 0 || accumulation[receiver] < MIN_CHANNEL_DISCHARGE) continue;
        if (current > maximumIncoming[receiver]) {
            maximumIncoming[receiver] = current;
            maximumCount[receiver] = 1;
        } else if (current == maximumIncoming[receiver] && maximumCount[receiver] < 255) {
            ++maximumCount[receiver];
        }
    }
}

void classifyLakes(const BasinSolution& solution, const Site& center, const Site& downstream,
                   const TerminalLake& terminalLake, const std::vector<float>& raw,
                   const std::vector<float>& filled, const std::vector<Receiver>& receivers,
                   const std::vector<uint8_t>& terminals, std::vector<uint8_t>& flags,
                   std::vector<float>& lakeSurface, std::vector<float>& lakeDepth) {
    flags.assign(RASTER_CELLS, 0);
    lakeSurface.assign(RASTER_CELLS, 0.0F);
    lakeDepth.assign(RASTER_CELLS, 0.0F);
    std::vector<int32_t> component(RASTER_CELLS, -1);
    std::vector<int> queue;
    int componentId = 0;
    for (int start = 0; start < RASTER_CELLS; ++start) {
        if (raw[start] < SEA_LEVEL) {
            flags[start] |= CELL_OCEAN;
            continue;
        }
        if (component[start] >= 0 || filled[start] - raw[start] <= 0.75F) continue;
        queue.clear();
        queue.push_back(start);
        component[start] = componentId;
        size_t cursor = 0;
        float spill = filled[start];
        float maximumDepth = 0.0F;
        while (cursor < queue.size()) {
            const int index = queue[cursor++];
            spill = std::max(spill, filled[index]);
            maximumDepth = std::max(maximumDepth, filled[index] - raw[index]);
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (component[neighbor] >= 0 || raw[neighbor] < SEA_LEVEL ||
                    filled[neighbor] - raw[neighbor] <= 0.75F) {
                    continue;
                }
                component[neighbor] = componentId;
                queue.push_back(neighbor);
            }
        }
        int outlet = -1;
        float outletLevel = std::numeric_limits<float>::max();
        bool containsEndorheicSink = false;
        for (int index : queue) {
            containsEndorheicSink =
                containsEndorheicSink ||
                terminals[index] == static_cast<uint8_t>(BasinOutlet::ENDORHEIC);
            const Receiver& receiver = receivers[index];
            for (int candidate : {receiver.first, receiver.second}) {
                if (candidate < 0 || component[candidate] == componentId) continue;
                if (filled[index] < outletLevel ||
                    (filled[index] == outletLevel && index < outlet)) {
                    outlet = index;
                    outletLevel = filled[index];
                }
            }
        }
        const bool retained = queue.size() >= 12 && maximumDepth >= 1.5F &&
                              queue.size() < RASTER_CELLS / 3 &&
                              (containsEndorheicSink || outlet >= 0);
        if (retained) {
            for (int index : queue) {
                flags[index] |= CELL_LAKE;
                if (containsEndorheicSink) flags[index] |= CELL_ENDORHEIC;
                lakeSurface[index] = spill;
                lakeDepth[index] = std::max(0.0F, spill - raw[index]);
            }
            if (!containsEndorheicSink) flags[outlet] |= CELL_LAKE_OUTLET;
        }
        ++componentId;
    }

    if (!sameCell(center, downstream) || center.ocean) return;
    for (int z = RASTER_APRON; z <= RASTER_APRON + INTERIOR_INTERVALS; ++z) {
        for (int x = RASTER_APRON; x <= RASTER_APRON + INTERIOR_INTERVALS; ++x) {
            const double worldX = solution.originX + (x - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (z - RASTER_APRON) * BASIN_RASTER_SPACING;
            if ((flags[indexOf(x, z)] & CELL_LAKE) != 0 &&
                std::hypot(worldX - center.x, worldZ - center.z) < 420.0) {
                return;
            }
        }
    }
    const float terminalSurface = static_cast<float>(terminalLake.surface);
    const double radius = terminalLake.radius;
    for (int z = RASTER_APRON; z <= RASTER_APRON + INTERIOR_INTERVALS; ++z) {
        for (int x = RASTER_APRON; x <= RASTER_APRON + INTERIOR_INTERVALS; ++x) {
            const double worldX = solution.originX + (x - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (z - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double radialMask =
                1.0 -
                smoothstep(radius * 0.72, radius, std::hypot(worldX - center.x, worldZ - center.z));
            if (radialMask <= 0.0) continue;
            const int index = indexOf(x, z);
            const float depth =
                static_cast<float>(std::max(0.0, (terminalSurface - raw[index]) * radialMask));
            if (depth <= 0.35F) continue;
            flags[index] =
                static_cast<uint8_t>((flags[index] & ~CELL_OCEAN) | CELL_LAKE | CELL_ENDORHEIC);
            lakeSurface[index] = terminalSurface;
            lakeDepth[index] = depth;
        }
    }
}

void nearestChannels(const std::vector<uint8_t>& streamOrder,
                     const std::vector<float>& accumulation, std::vector<float>& distance,
                     std::vector<int32_t>& nearest) {
    using QueueValue = std::pair<float, int32_t>;
    std::priority_queue<QueueValue, std::vector<QueueValue>, std::greater<>> queue;
    distance.assign(RASTER_CELLS, std::numeric_limits<float>::max());
    nearest.assign(RASTER_CELLS, -1);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (streamOrder[index] == 0 || accumulation[index] < MIN_CHANNEL_DISCHARGE) continue;
        distance[index] = 0.0F;
        nearest[index] = index;
        queue.emplace(0.0F, index);
    }
    while (!queue.empty()) {
        const auto [currentDistance, index] = queue.top();
        queue.pop();
        if (currentDistance != distance[index]) continue;
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        for (int direction = 0; direction < 8; ++direction) {
            const int nx = x + DX[direction];
            const int nz = z + DZ[direction];
            if (!inRaster(nx, nz)) continue;
            const int neighbor = indexOf(nx, nz);
            const float candidate =
                currentDistance + static_cast<float>(STEP_LENGTH[direction] * BASIN_RASTER_SPACING);
            if (candidate >= distance[neighbor]) continue;
            distance[neighbor] = candidate;
            nearest[neighbor] = nearest[index];
            queue.emplace(candidate, neighbor);
        }
    }
}

void buildDeltaBranches(const CounterRng& random, const Site& center, const Site& downstream,
                        double mouthX, double mouthZ, double waterSurface, bool lakeEntry,
                        double lakeRadius, double discharge, double sediment,
                        std::vector<DeltaBranch>& branches) {
    if (sameCell(center, downstream) || center.ocean || discharge < 480.0) return;
    const double length = std::hypot(downstream.x - center.x, downstream.z - center.z);
    const double gradient =
        std::max(0.0, center.elevation - downstream.elevation) / std::max(1.0, length);
    if (gradient > 0.035 || sediment < discharge * 0.012) return;
    double directionX = downstream.x - center.x;
    double directionZ = downstream.z - center.z;
    const double magnitude = std::hypot(directionX, directionZ);
    directionX /= magnitude;
    directionZ /= magnitude;
    const double perpendicularX = -directionZ;
    const double perpendicularZ = directionX;
    const int count = random.uniformInt(DELTA_STREAM, center.cell.x, 0, center.cell.z, 0, 2, 4);
    const double fanLength =
        lakeEntry
            ? std::clamp(70.0 + std::sqrt(discharge) * 1.4, 90.0, std::max(90.0, lakeRadius * 0.82))
            : std::clamp(180.0 + std::sqrt(discharge) * 5.0, 220.0, 620.0);
    for (int branch = 0; branch < count; ++branch) {
        const double centered =
            count == 1 ? 0.0 : (static_cast<double>(branch) / (count - 1) - 0.5) * 2.0;
        const double jitter =
            random.signedUnit(DELTA_STREAM, center.cell.x, branch + 1, center.cell.z) * 0.18;
        const double spread = centered * 0.58 + jitter;
        DeltaBranch result;
        result.startX = mouthX - directionX * fanLength * 0.32;
        result.startZ = mouthZ - directionZ * fanLength * 0.32;
        result.endX = mouthX + directionX * fanLength + perpendicularX * spread * fanLength;
        result.endZ = mouthZ + directionZ * fanLength + perpendicularZ * spread * fanLength;
        result.width =
            static_cast<float>(std::clamp(5.0 + std::sqrt(discharge / count) * 0.18, 7.0, 22.0));
        result.waterSurface = static_cast<float>(waterSurface);
        result.groupCount = static_cast<uint8_t>(count);
        result.lakeEntry = lakeEntry;
        branches.push_back(result);
    }
}

std::vector<float> chamferDistanceToLakeState(const std::vector<uint8_t>& flags,
                                              bool sourceIsLake) {
    constexpr float DIAGONAL_COST = static_cast<float>(std::numbers::sqrt2);
    constexpr float INFINITY_DISTANCE = std::numeric_limits<float>::max() / 8.0F;
    std::vector<float> distance(RASTER_CELLS, INFINITY_DISTANCE);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const bool lake = (flags[index] & CELL_LAKE) != 0;
        if (lake == sourceIsLake) distance[index] = 0.0F;
    }
    const auto relax = [&](int x, int z, int neighborX, int neighborZ, float cost) {
        if (!inRaster(neighborX, neighborZ)) return;
        float& value = distance[indexOf(x, z)];
        value = std::min(value, distance[indexOf(neighborX, neighborZ)] + cost);
    };
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            relax(x, z, x - 1, z, 1.0F);
            relax(x, z, x, z - 1, 1.0F);
            relax(x, z, x - 1, z - 1, DIAGONAL_COST);
            relax(x, z, x + 1, z - 1, DIAGONAL_COST);
        }
    }
    for (int z = RASTER_EDGE - 1; z >= 0; --z) {
        for (int x = RASTER_EDGE - 1; x >= 0; --x) {
            relax(x, z, x + 1, z, 1.0F);
            relax(x, z, x, z + 1, 1.0F);
            relax(x, z, x + 1, z + 1, DIAGONAL_COST);
            relax(x, z, x - 1, z + 1, DIAGONAL_COST);
        }
    }
    return distance;
}

void conditionLakeMembershipAtSeams(BasinSolution& solution, const std::vector<float>& raw) {
    const std::vector<uint8_t> originalFlags = solution.flags;
    const std::vector<float> distanceToLake = chamferDistanceToLakeState(originalFlags, true);
    const std::vector<float> distanceToDry = chamferDistanceToLakeState(originalFlags, false);
    std::vector<std::pair<int, int>> outletCells;
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if ((originalFlags[index] & CELL_LAKE_OUTLET) != 0) {
            outletCells.emplace_back(index % RASTER_EDGE, index / RASTER_EDGE);
        }
    }

    solution.lakeShoreDistance.resize(RASTER_CELLS, -1.0e9F);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            const bool originalLake = (originalFlags[index] & CELL_LAKE) != 0;
            const float oppositeDistance =
                originalLake ? distanceToDry[index] : distanceToLake[index];
            float signedDistance = originalLake ? 1.0e9F : -1.0e9F;
            if (oppositeDistance < 1.0e8F) {
                const float distanceBlocks = oppositeDistance * BASIN_RASTER_SPACING -
                                             static_cast<float>(BASIN_RASTER_SPACING * 0.5);
                signedDistance = originalLake ? distanceBlocks : -distanceBlocks;
            }

            if (originalLake && inSampledCatchment(x, z)) {
                const int edgeDistance = std::min({x - RASTER_APRON, z - RASTER_APRON,
                                                   RASTER_APRON + INTERIOR_INTERVALS - x,
                                                   RASTER_APRON + INTERIOR_INTERVALS - z});
                bool outletCorridor = false;
                for (const auto [outletX, outletZ] : outletCells) {
                    if (std::hypot(x - outletX, z - outletZ) <= 2.25) {
                        outletCorridor = true;
                        break;
                    }
                }
                if (!outletCorridor && edgeDistance <= LAKE_SEAM_SHAPING_CELLS) {
                    const double freeboard = solution.waterSurface[index] - raw[index];
                    const double terrainOffset = std::clamp((freeboard - 2.5) * 3.2, -8.0, 8.0);
                    const double seamDistance =
                        edgeDistance * BASIN_RASTER_SPACING - LAKE_SEAM_SETBACK + terrainOffset;
                    signedDistance = std::min(signedDistance, static_cast<float>(seamDistance));
                }
            }
            solution.lakeShoreDistance[index] = signedDistance;

            if (!originalLake) continue;
            if (signedDistance <= 0.0F) {
                solution.flags[index] &=
                    static_cast<uint8_t>(~(CELL_LAKE | CELL_ENDORHEIC | CELL_LAKE_OUTLET));
                solution.lakeDepth[index] = 0.0F;
                solution.waterSurface[index] = SEA_LEVEL;
                continue;
            }

            const float solvedDepth = std::max(solution.lakeDepth[index], 0.0F);
            const float shelfWeight =
                static_cast<float>(smoothstep(0.0, LAKE_SHELF_WIDTH, signedDistance));
            const float shelfDepth = 0.125F + std::max(0.0F, solvedDepth - 0.125F) * shelfWeight;
            solution.lakeDepth[index] = std::min(solvedDepth, shelfDepth);
            solution.surface[index] = solution.waterSurface[index] - solution.lakeDepth[index];
        }
    }
}

void propagateLakeShoreLevels(BasinSolution& solution) {
    using QueueValue = std::tuple<float, int32_t, int32_t>;
    constexpr float INFINITY_DISTANCE = std::numeric_limits<float>::max() / 8.0F;
    std::priority_queue<QueueValue, std::vector<QueueValue>, std::greater<>> queue;
    std::vector<float> distance(RASTER_CELLS, INFINITY_DISTANCE);
    std::vector<int32_t> source(RASTER_CELLS, -1);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if ((solution.flags[index] & CELL_LAKE) == 0) continue;
        distance[index] = 0.0F;
        source[index] = index;
        queue.emplace(0.0F, index, index);
    }
    while (!queue.empty()) {
        const auto [currentDistance, index, currentSource] = queue.top();
        queue.pop();
        if (currentDistance != distance[index] || currentSource != source[index]) continue;
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        for (int direction = 0; direction < 8; ++direction) {
            const int nx = x + DX[direction];
            const int nz = z + DZ[direction];
            if (!inRaster(nx, nz)) continue;
            const int neighbor = indexOf(nx, nz);
            const float candidate = currentDistance + static_cast<float>(STEP_LENGTH[direction]);
            if (candidate > distance[neighbor] + 1.0e-6F ||
                (std::abs(candidate - distance[neighbor]) <= 1.0e-6F && source[neighbor] >= 0 &&
                 currentSource >= source[neighbor])) {
                continue;
            }
            distance[neighbor] = candidate;
            source[neighbor] = currentSource;
            queue.emplace(candidate, neighbor, currentSource);
        }
    }

    solution.shoreWaterSurface.resize(RASTER_CELLS, 0.0F);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (source[index] >= 0) {
            solution.shoreWaterSurface[index] = solution.waterSurface[source[index]];
        }
    }
}

void enforceSharedBoundaryConstraints(BasinSolution& solution, const std::vector<float>& raw,
                                      const std::vector<ChannelGuide>& guides) {
    conditionLakeMembershipAtSeams(solution, raw);
    for (int z = RASTER_APRON; z <= RASTER_APRON + INTERIOR_INTERVALS; ++z) {
        for (int x = RASTER_APRON; x <= RASTER_APRON + INTERIOR_INTERVALS; ++x) {
            const int edgeDistance =
                std::min({x - RASTER_APRON, z - RASTER_APRON, RASTER_APRON + INTERIOR_INTERVALS - x,
                          RASTER_APRON + INTERIOR_INTERVALS - z});
            if (edgeDistance > SHARED_EDGE_BAND_CELLS) continue;
            const float sharedWeight =
                static_cast<float>(1.0 - smoothstep(0.0, SHARED_EDGE_BAND_CELLS, edgeDistance));
            const int index = indexOf(x, z);
            const double worldX = solution.originX + (x - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (z - RASTER_APRON) * BASIN_RASTER_SPACING;

            // Lake membership is conditioned by its continuous shoreline
            // field before shared channel constraints are applied. A lake
            // that remains this far into the seam band owns its flat level;
            // the shared portal corridor starts closer to the catchment face.
            if ((solution.flags[index] & CELL_LAKE) != 0) continue;

            float sharedSurface = raw[index];
            float sharedWater = SEA_LEVEL;
            float sharedDischarge = 0.0F;
            float sharedSediment = 0.0F;
            float sharedDistance = std::numeric_limits<float>::max();
            float sharedWidth = 0.0F;
            float sharedDepth = 0.0F;
            float sharedGradient = 0.0F;
            float sharedErosion = 0.0F;
            float sharedFlowX = 1.0F;
            float sharedFlowZ = 0.0F;
            uint8_t sharedOrder = 0;
            uint8_t sharedFlags = raw[index] < SEA_LEVEL ? CELL_OCEAN : 0;

            const ChannelGuide* closest = nullptr;
            double closestDistance = std::numeric_limits<double>::max();
            for (const ChannelGuide& guide : guides) {
                const double distance = std::hypot(worldX - guide.portalX, worldZ - guide.portalZ);
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closest = &guide;
                }
            }
            if (closest != nullptr) {
                const int order = std::clamp(
                    1 + static_cast<int>(std::floor(std::log2(closest->discharge / 500.0 + 1.0))),
                    1, 6);
                const double width =
                    std::clamp(3.0 + std::sqrt(closest->discharge) * 0.22 + order * 1.2, 4.0, 42.0);
                const double floodplainWidth = width * (2.4 + order * 0.28);
                if (closestDistance <= floodplainWidth) {
                    const double depth = std::clamp(
                        1.2 + std::sqrt(closest->discharge) * 0.065 + order * 0.65, 1.8, 14.0);
                    const double mask =
                        1.0 - smoothstep(width * 0.45, floodplainWidth, closestDistance);
                    const double bed = closest->portalWater - depth;
                    sharedFlowX = static_cast<float>(closest->portalFlowX);
                    sharedFlowZ = static_cast<float>(closest->portalFlowZ);
                    sharedSurface =
                        static_cast<float>(raw[index] - std::max(0.0, raw[index] - bed) * mask);
                    sharedWater = static_cast<float>(closest->portalWater);
                    sharedDischarge = static_cast<float>(closest->discharge);
                    sharedSediment =
                        static_cast<float>(closest->discharge * (0.018 + closest->gradient * 0.9));
                    sharedDistance = static_cast<float>(closestDistance);
                    sharedWidth = static_cast<float>(width);
                    sharedDepth = static_cast<float>(depth);
                    sharedGradient = static_cast<float>(closest->gradient);
                    sharedErosion = raw[index] - sharedSurface;
                    sharedOrder = static_cast<uint8_t>(order);
                    if (closest->oceanMouth) {
                        sharedFlags = CELL_OCEAN;
                    } else if (closest->backwater && sharedSurface + 0.05F < sharedWater) {
                        sharedFlags = CELL_BACKWATER;
                    } else {
                        sharedFlags = 0;
                    }
                }
            }

            const auto blend = [sharedWeight](float local, float shared) {
                return local + (shared - local) * sharedWeight;
            };
            solution.surface[index] = blend(solution.surface[index], sharedSurface);
            solution.waterSurface[index] = blend(solution.waterSurface[index], sharedWater);
            solution.discharge[index] = blend(solution.discharge[index], sharedDischarge);
            solution.sediment[index] = blend(solution.sediment[index], sharedSediment);
            solution.channelDistance[index] =
                blend(solution.channelDistance[index], sharedDistance);
            solution.channelWidth[index] = blend(solution.channelWidth[index], sharedWidth);
            solution.channelDepth[index] = blend(solution.channelDepth[index], sharedDepth);
            solution.channelGradient[index] =
                blend(solution.channelGradient[index], sharedGradient);
            solution.erosionDepth[index] = blend(solution.erosionDepth[index], sharedErosion);
            solution.lakeDepth[index] = blend(solution.lakeDepth[index], 0.0F);
            solution.flowX[index] = blend(solution.flowX[index], sharedFlowX);
            solution.flowZ[index] = blend(solution.flowZ[index], sharedFlowZ);
            if (sharedWeight > 0.0F) {
                solution.streamOrder[index] = sharedOrder;
                solution.flags[index] = sharedFlags;
            }
        }
    }
}

bool isActiveChannelCell(const BasinSolution& solution, int index) {
    return (solution.flags[index] & CELL_WATERFALL) != 0 ||
           (solution.streamOrder[index] > 0 && solution.channelWidth[index] > 0.0F &&
            solution.channelDistance[index] <= solution.channelWidth[index] * 0.55F);
}

void appendChannelFalls(BasinSolution& solution, const std::vector<Receiver>& receivers) {
    // A steep channel marker describes a possible knickpoint, not a volume of
    // falling water. Promote only an actual drop between routed channel cells
    // to an analytical fall. The receiver-centered, narrow footprint is then
    // shared by exact cube emission and far terrain instead of turning the
    // complete upstream river band into a vertical water sheet.
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const bool steepKnickpoint =
            (solution.flags[index] & CELL_WATERFALL) != 0 ||
            (solution.channelGradient[index] > 0.008F && solution.discharge[index] > 400.0F);
        if (!steepKnickpoint || solution.streamOrder[index] == 0 ||
            solution.channelDistance[index] > solution.channelWidth[index] * 0.60F) {
            continue;
        }
        const float topSurface = solution.waterSurface[index];
        const int sourceX = index % RASTER_EDGE;
        const int sourceZ = index / RASTER_EDGE;
        auto receivingSurface = [&](int target) {
            return (solution.flags[target] & CELL_OCEAN) != 0 ? static_cast<float>(SEA_LEVEL)
                                                              : solution.waterSurface[target];
        };
        auto validReceiver = [&](int target) {
            return target >= 0 && target != index &&
                   ((solution.flags[target] & (CELL_OCEAN | CELL_LAKE)) != 0 ||
                    isActiveChannelCell(solution, target)) &&
                   topSurface >= receivingSurface(target) + 2.5F;
        };

        int receiver = receivers[index].first;
        if (!validReceiver(receiver)) {
            receiver = -1;
            double bestAlignment = 0.20;
            float bestSurface = topSurface;
            for (int direction = 0; direction < 8; ++direction) {
                const int targetX = sourceX + DX[direction];
                const int targetZ = sourceZ + DZ[direction];
                if (!inRaster(targetX, targetZ)) continue;
                const int candidate = indexOf(targetX, targetZ);
                if (!validReceiver(candidate)) continue;
                const double inverseLength = 1.0 / STEP_LENGTH[direction];
                const double alignment = (solution.flowX[index] * DX[direction] +
                                          solution.flowZ[index] * DZ[direction]) *
                                         inverseLength;
                const float candidateSurface = receivingSurface(candidate);
                if (alignment > bestAlignment + 1.0e-6 ||
                    (std::abs(alignment - bestAlignment) <= 1.0e-6 &&
                     (candidateSurface < bestSurface ||
                      (candidateSurface == bestSurface && candidate < receiver)))) {
                    receiver = candidate;
                    bestAlignment = alignment;
                    bestSurface = candidateSurface;
                }
            }
        }
        if (receiver < 0) continue;

        const float bottomSurface = receivingSurface(receiver);
        if (topSurface < bottomSurface + 2.5F) continue;

        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;
        const float channelWidth =
            std::max(solution.channelWidth[index], solution.channelWidth[receiver]);
        solution.outletFalls.push_back({
            .startX = solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING,
            .startZ = solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING,
            .endX = solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING,
            .endZ = solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING,
            .topSurface = topSurface,
            .bottomSurface = bottomSurface,
            .halfWidth = std::clamp(channelWidth * 0.28F, 2.0F, 8.0F),
        });
    }
}

void indexOutletFalls(BasinSolution& solution) {
    solution.outletFallByCell.assign(RASTER_CELLS, -1);
    for (size_t fallIndex = 0; fallIndex < solution.outletFalls.size(); ++fallIndex) {
        const OutletFall& fall = solution.outletFalls[fallIndex];
        const int gridX = std::clamp(
            static_cast<int>(std::llround((fall.endX - solution.originX) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            0, RASTER_EDGE - 1);
        const int gridZ = std::clamp(
            static_cast<int>(std::llround((fall.endZ - solution.originZ) / BASIN_RASTER_SPACING)) +
                RASTER_APRON,
            0, RASTER_EDGE - 1);
        int32_t& indexed = solution.outletFallByCell[indexOf(gridX, gridZ)];
        if (indexed < 0 ||
            fall.topSurface > solution.outletFalls[static_cast<size_t>(indexed)].topSurface) {
            indexed = static_cast<int32_t>(fallIndex);
        }
    }
}

void restoreLakeOutletMarkers(BasinSolution& solution, const std::vector<Receiver>& receivers,
                              const std::vector<float>& filled,
                              const std::vector<uint32_t>& floodOrder,
                              bool recordOutletFalls = false) {
    if (recordOutletFalls) solution.outletFalls.clear();
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    std::vector<int> queue;
    for (int start = 0; start < RASTER_CELLS; ++start) {
        if (visited[start] != 0 || (solution.flags[start] & CELL_LAKE) == 0) continue;
        queue.clear();
        queue.push_back(start);
        visited[start] = 1;
        size_t cursor = 0;
        bool endorheic = false;
        int outlet = -1;
        int outletTarget = -1;
        bool outletActive = false;
        while (cursor < queue.size()) {
            const int index = queue[cursor++];
            solution.flags[index] &= static_cast<uint8_t>(~CELL_LAKE_OUTLET);
            endorheic = endorheic || (solution.flags[index] & CELL_ENDORHEIC) != 0;
            const Receiver& receiver = receivers[index];
            for (const auto [target, weight] :
                 {std::pair{receiver.first, receiver.firstWeight},
                  std::pair{receiver.second, receiver.secondWeight}}) {
                if (target < 0 || weight <= 0.0F || (solution.flags[target] & CELL_LAKE) != 0) {
                    continue;
                }
                const bool active = (solution.flags[target] & CELL_OCEAN) != 0 ||
                                    isActiveChannelCell(solution, target);
                const bool preferable =
                    outlet < 0 || (active && !outletActive) ||
                    (active == outletActive &&
                     (filled[target] < filled[outletTarget] ||
                      (filled[target] == filled[outletTarget] &&
                       (floodOrder[target] < floodOrder[outletTarget] ||
                        (floodOrder[target] == floodOrder[outletTarget] &&
                         (target < outletTarget || (target == outletTarget && index < outlet)))))));
                if (preferable) {
                    outlet = index;
                    outletTarget = target;
                    outletActive = active;
                }
            }
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (visited[neighbor] != 0 || (solution.flags[neighbor] & CELL_LAKE) == 0) continue;
                visited[neighbor] = 1;
                queue.push_back(neighbor);
            }
        }
        if (!endorheic && outlet >= 0 && outletActive) {
            solution.flags[outlet] |= CELL_LAKE_OUTLET;
            if (recordOutletFalls) {
                const float topSurface = solution.waterSurface[outlet];
                const float bottomSurface = (solution.flags[outletTarget] & CELL_OCEAN) != 0 ||
                                                    solution.surface[outletTarget] < SEA_LEVEL
                                                ? static_cast<float>(SEA_LEVEL)
                                                : solution.waterSurface[outletTarget];
                if (topSurface >= bottomSurface + 2.5F) {
                    const int outletX = outlet % RASTER_EDGE;
                    const int outletZ = outlet / RASTER_EDGE;
                    const int targetX = outletTarget % RASTER_EDGE;
                    const int targetZ = outletTarget / RASTER_EDGE;
                    const float channelWidth = std::max(solution.channelWidth[outletTarget],
                                                        solution.channelWidth[outlet]);
                    solution.outletFalls.push_back({
                        .startX =
                            solution.originX + (outletX - RASTER_APRON) * BASIN_RASTER_SPACING,
                        .startZ =
                            solution.originZ + (outletZ - RASTER_APRON) * BASIN_RASTER_SPACING,
                        .endX = solution.originX + (targetX - RASTER_APRON) * BASIN_RASTER_SPACING,
                        .endZ = solution.originZ + (targetZ - RASTER_APRON) * BASIN_RASTER_SPACING,
                        .topSurface = topSurface,
                        .bottomSurface = bottomSurface,
                        .halfWidth = std::clamp(channelWidth * 0.28F, 2.0F, 8.0F),
                    });
                }
            }
        } else if (!endorheic) {
            for (const int index : queue)
                solution.flags[index] |= CELL_ENDORHEIC;
        }
    }
}

void normalizeLakeGeometry(BasinSolution& solution) {
    // A retained lake owns one flat spill level and a positive depth at each
    // member cell. Reconstructing its floor from those two authoritative
    // values prevents interpolation with nearby dry or ocean terrain from
    // creating a deep phantom fringe.
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if ((solution.flags[index] & CELL_LAKE) == 0) continue;
        const float depth = std::max(solution.lakeDepth[index], 0.0F);
        if (depth <= 0.05F) {
            solution.flags[index] &=
                static_cast<uint8_t>(~(CELL_LAKE | CELL_ENDORHEIC | CELL_LAKE_OUTLET));
            solution.lakeDepth[index] = 0.0F;
            continue;
        }
        solution.surface[index] = solution.waterSurface[index] - depth;
    }

    // Dry shoreline support is reconstructed continuously from the signed
    // lake field at sample time. Keeping it out of this categorical raster
    // avoids a one-cell wall and lets the bank taper over sixteen blocks.
}

bool validateSolution(const BasinSolution& solution, const std::vector<float>& raw,
                      const std::vector<float>& filled, const std::vector<uint32_t>& floodOrder,
                      const std::vector<Receiver>& receivers, const std::vector<uint8_t>& terminals,
                      const std::vector<float>& accumulation) {
    const auto finiteField = [](const std::vector<float>& field) {
        return std::all_of(field.begin(), field.end(),
                           [](float value) { return std::isfinite(value); });
    };
    if (!finiteField(solution.surface) || !finiteField(solution.waterSurface) ||
        !finiteField(solution.discharge) || !finiteField(solution.sediment) ||
        !finiteField(solution.channelDistance) || !finiteField(solution.channelWidth) ||
        !finiteField(solution.channelDepth) || !finiteField(solution.channelGradient) ||
        !finiteField(solution.erosionDepth) || !finiteField(solution.lakeDepth) ||
        !finiteField(solution.lakeShoreDistance) || !finiteField(solution.shoreWaterSurface) ||
        !finiteField(solution.flowX) || !finiteField(solution.flowZ)) {
        return false;
    }
    if (solution.outletTypes.size() != RASTER_CELLS || terminals.size() != RASTER_CELLS ||
        accumulation.size() != RASTER_CELLS || solution.outletFallByCell.size() != RASTER_CELLS ||
        solution.lakeShoreDistance.size() != RASTER_CELLS ||
        solution.shoreWaterSurface.size() != RASTER_CELLS) {
        return false;
    }
    if (std::any_of(
            solution.outletFallByCell.begin(), solution.outletFallByCell.end(), [&](int32_t index) {
                return index < -1 ||
                       (index >= 0 && static_cast<size_t>(index) >= solution.outletFalls.size());
            })) {
        return false;
    }
    for (const ChannelGuide& guide : solution.guides) {
        if (!std::isfinite(guide.portalWater) || !std::isfinite(guide.downstreamWater) ||
            !std::isfinite(guide.terrainUpper) ||
            guide.portalWater + 1.0e-4 < guide.downstreamWater ||
            (guide.oceanMouth && std::abs(guide.portalWater - SEA_LEVEL) > 1.0e-6) ||
            (!guide.backwater && guide.portalWater > guide.terrainUpper + 1.0e-4) ||
            (guide.backwater && std::abs(guide.portalWater - guide.downstreamWater) > 1.0e-4) ||
            guide.profileFall != (guide.portalWater >= guide.downstreamWater + 2.5)) {
            return false;
        }
        const int portalX =
            std::clamp(static_cast<int>(std::llround((guide.portalX - solution.originX) /
                                                     BASIN_RASTER_SPACING)) +
                           RASTER_APRON,
                       0, RASTER_EDGE - 1);
        const int portalZ =
            std::clamp(static_cast<int>(std::llround((guide.portalZ - solution.originZ) /
                                                     BASIN_RASTER_SPACING)) +
                           RASTER_APRON,
                       0, RASTER_EDGE - 1);
        const int portalIndex = indexOf(portalX, portalZ);
        if ((solution.flags[portalIndex] & CELL_LAKE) != 0 ||
            solution.surface[portalIndex] > raw[portalIndex] + 1.0F) {
            return false;
        }
        if (solution.channelWidth[portalIndex] > 0.0F &&
            solution.channelDistance[portalIndex] <= solution.channelWidth[portalIndex] * 0.55F) {
            const float actualDepth =
                solution.waterSurface[portalIndex] - solution.surface[portalIndex];
            if ((!guide.backwater &&
                 std::abs(actualDepth - solution.channelDepth[portalIndex]) > 0.05F) ||
                (guide.backwater && actualDepth + 0.05F < solution.channelDepth[portalIndex])) {
                return false;
            }
        }
    }

    std::vector<double> incoming(RASTER_CELLS, 0.0);
    std::vector<uint16_t> incomingChannels(RASTER_CELLS, 0);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const Receiver& receiver = receivers[index];
        for (const auto [target, weight] : {std::pair{receiver.first, receiver.firstWeight},
                                            std::pair{receiver.second, receiver.secondWeight}}) {
            if (target < 0 || weight <= 0.0F) continue;
            incoming[target] += accumulation[index] * weight;
            if (accumulation[index] >= MIN_CHANNEL_DISCHARGE) ++incomingChannels[target];
            if (accumulation[target] + 0.01F < accumulation[index] * weight) {
                return false;
            }
        }
    }

    for (int index = 0; index < RASTER_CELLS; ++index) {
        const Receiver& receiver = receivers[index];
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        const bool sampled = x >= RASTER_APRON && x <= RASTER_APRON + INTERIOR_INTERVALS &&
                             z >= RASTER_APRON && z <= RASTER_APRON + INTERIOR_INTERVALS;
        const BasinOutlet outlet = static_cast<BasinOutlet>(solution.outletTypes[index]);
        if (sampled && outlet == BasinOutlet::NONE) {
            return false;
        }
        if (sampled && receiver.first < 0 &&
            terminals[index] == static_cast<uint8_t>(BasinOutlet::NONE)) {
            return false;
        }
        if (receiver.first >= 0 && !routesLower(index, receiver.first, filled, floodOrder)) {
            return false;
        }
        if (receiver.second >= 0 && receiver.secondWeight > 0.0F &&
            !routesLower(index, receiver.second, filled, floodOrder)) {
            return false;
        }
        if (receiver.first >= 0 &&
            solution.outletTypes[receiver.first] == static_cast<uint8_t>(BasinOutlet::NONE)) {
            return false;
        }
        if (receiver.second >= 0 && receiver.secondWeight > 0.0F &&
            solution.outletTypes[receiver.second] == static_cast<uint8_t>(BasinOutlet::NONE)) {
            return false;
        }
        if (receiver.first >= 0 && solution.streamOrder[index] > 0 &&
            solution.streamOrder[receiver.first] > 0 && solution.channelDistance[index] < 0.01F &&
            solution.channelDistance[receiver.first] < 0.01F &&
            (solution.flags[index] & (CELL_OCEAN | CELL_LAKE)) == 0 &&
            (solution.flags[receiver.first] & (CELL_OCEAN | CELL_LAKE)) == 0 &&
            !lockedBoundary(index % RASTER_EDGE, index / RASTER_EDGE) &&
            !lockedBoundary(receiver.first % RASTER_EDGE, receiver.first / RASTER_EDGE) &&
            solution.waterSurface[receiver.first] > solution.waterSurface[index] + 0.02F) {
            return false;
        }
        if ((solution.flags[index] & CELL_LAKE) != 0) {
            if (solution.lakeDepth[index] <= 0.0F || solution.lakeShoreDistance[index] <= 0.0F ||
                std::abs((solution.waterSurface[index] - solution.surface[index]) -
                         solution.lakeDepth[index]) > 1.0e-3F) {
                return false;
            }
        } else if ((solution.flags[index] & CELL_OCEAN) == 0 &&
                   solution.lakeShoreDistance[index] > 0.0F) {
            return false;
        }
        if (incomingChannels[index] >= 2 && accumulation[index] + 0.05 < incoming[index]) {
            return false;
        }
    }

    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    std::vector<int> queue;
    for (int start = 0; start < RASTER_CELLS; ++start) {
        if (visited[start] != 0 || (solution.flags[start] & CELL_LAKE) == 0) continue;
        queue.clear();
        queue.push_back(start);
        visited[start] = 1;
        size_t cursor = 0;
        bool endorheic = false;
        bool hasOutlet = false;
        bool hasActiveOutlet = false;
        size_t outletCount = 0;
        const float level = solution.waterSurface[start];
        while (cursor < queue.size()) {
            const int index = queue[cursor++];
            if (std::abs(solution.waterSurface[index] - level) > 1.0e-4F ||
                solution.lakeDepth[index] <= 0.0F) {
                return false;
            }
            endorheic = endorheic || (solution.flags[index] & CELL_ENDORHEIC) != 0;
            if ((solution.flags[index] & CELL_LAKE_OUTLET) != 0) {
                ++outletCount;
                const Receiver& receiver = receivers[index];
                const bool firstLeaves =
                    receiver.first >= 0 && (solution.flags[receiver.first] & CELL_LAKE) == 0;
                const bool secondLeaves = receiver.second >= 0 && receiver.secondWeight > 0.0F &&
                                          (solution.flags[receiver.second] & CELL_LAKE) == 0;
                const auto receivingWater = [&](int target) {
                    return (solution.flags[target] & CELL_OCEAN) != 0
                               ? static_cast<float>(SEA_LEVEL)
                               : solution.waterSurface[target];
                };
                if ((firstLeaves && receivingWater(receiver.first) > level + 0.02F) ||
                    (secondLeaves && receivingWater(receiver.second) > level + 0.02F)) {
                    return false;
                }
                hasOutlet = hasOutlet || firstLeaves || secondLeaves;
                if (firstLeaves && ((solution.flags[receiver.first] & CELL_OCEAN) != 0 ||
                                    isActiveChannelCell(solution, receiver.first))) {
                    hasActiveOutlet = true;
                }
                if (secondLeaves && ((solution.flags[receiver.second] & CELL_OCEAN) != 0 ||
                                     isActiveChannelCell(solution, receiver.second))) {
                    hasActiveOutlet = true;
                }
            }
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (visited[neighbor] != 0 || (solution.flags[neighbor] & CELL_LAKE) == 0) continue;
                visited[neighbor] = 1;
                queue.push_back(neighbor);
            }
        }
        if ((!endorheic && (outletCount != 1 || !hasOutlet || !hasActiveOutlet)) ||
            (endorheic && outletCount != 0)) {
            return false;
        }
    }

    for (const DeltaBranch& branch : solution.deltaBranches) {
        if (!std::isfinite(branch.startX) || !std::isfinite(branch.startZ) ||
            !std::isfinite(branch.endX) || !std::isfinite(branch.endZ) ||
            !std::isfinite(branch.waterSurface) || branch.width <= 0.0F || branch.groupCount < 2 ||
            branch.groupCount > 4) {
            return false;
        }
    }
    for (const OutletFall& fall : solution.outletFalls) {
        if (!std::isfinite(fall.startX) || !std::isfinite(fall.startZ) ||
            !std::isfinite(fall.endX) || !std::isfinite(fall.endZ) ||
            !std::isfinite(fall.topSurface) || !std::isfinite(fall.bottomSurface) ||
            !std::isfinite(fall.halfWidth) || fall.topSurface < fall.bottomSurface + 2.5F ||
            fall.halfWidth < 2.0F || fall.halfWidth > 8.0F) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<const BasinSolution>
buildSolution(const CounterRng& random, BasinKey key,
              const BasinSolver::ElevationFunction& elevation,
              const BasinSolver::RainfallFunction& rainfall,
              const BasinSolver::RockResistanceFunction& rockResistance) {
    auto solution = std::make_shared<BasinSolution>();
    solution->key = key;
    solution->originX = static_cast<double>(key.x) * BASIN_CATCHMENT_EDGE;
    solution->originZ = static_cast<double>(key.z) * BASIN_CATCHMENT_EDGE;

    MacroContext macro(random, elevation, rainfall);
    const Site center = macro.site(key);
    const Site downstream = macro.downstream(center);
    const TerminalLake terminalLake =
        sameCell(center, downstream) && !center.ocean ? macro.terminalLake(center) : TerminalLake{};
    const std::vector<ChannelGuide> guides = channelGuides(random, macro, center, downstream);
    solution->guides = guides;

    std::vector<float> raw;
    std::vector<float> rain;
    std::vector<float> resistance;
    sampleInputs(*solution, elevation, rainfall, rockResistance, raw, rain, resistance);
    carveChannelGuides(*solution, guides, raw);
    const std::vector<uint8_t> terminals =
        buildOutletConstraints(*solution, center, downstream, guides, raw);

    std::vector<float> filled;
    std::vector<uint32_t> floodOrder;
    priorityFlood(raw, terminals, filled, floodOrder);
    std::vector<Receiver> receivers;
    std::vector<float> routingFlowX;
    std::vector<float> routingFlowZ;
    std::vector<int32_t> upstreamOrder;
    buildDInfinityRouting(filled, floodOrder, terminals, receivers, routingFlowX, routingFlowZ,
                          upstreamOrder);
    std::vector<float> accumulation;
    accumulateFlow(rain, receivers, upstreamOrder, accumulation, *solution, guides);

    std::vector<float> eroded;
    std::vector<float> transportedSediment;
    erodeTerrain(raw, resistance, receivers, upstreamOrder, accumulation, eroded,
                 transportedSediment);

    priorityFlood(eroded, terminals, filled, floodOrder);
    buildDInfinityRouting(filled, floodOrder, terminals, receivers, routingFlowX, routingFlowZ,
                          upstreamOrder);
    accumulateFlow(rain, receivers, upstreamOrder, accumulation, *solution, guides);
    std::vector<uint8_t> strahler;
    computeStrahler(receivers, upstreamOrder, accumulation, strahler);

    std::vector<uint8_t> flags;
    std::vector<float> lakeSurface;
    std::vector<float> classifiedLakeDepth;
    classifyLakes(*solution, center, downstream, terminalLake, eroded, filled, receivers, terminals,
                  flags, lakeSurface, classifiedLakeDepth);

    std::vector<float> channelDistance;
    std::vector<int32_t> nearestChannel;
    nearestChannels(strahler, accumulation, channelDistance, nearestChannel);

    std::vector<float> channelWater(RASTER_CELLS, std::numeric_limits<float>::max());
    std::vector<float> localGradient(RASTER_CELLS, 0.0F);
    std::vector<float> localWidth(RASTER_CELLS, 0.0F);
    std::vector<float> localDepth(RASTER_CELLS, 0.0F);
    for (int index : upstreamOrder) {
        if (strahler[index] == 0) continue;
        const int receiver = receivers[index].first;
        double gradient = 0.0;
        if (receiver >= 0) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            const int rx = receiver % RASTER_EDGE;
            const int rz = receiver / RASTER_EDGE;
            gradient = std::max(0.0, (eroded[index] - eroded[receiver]) /
                                         (std::hypot(rx - x, rz - z) * BASIN_RASTER_SPACING));
        }
        localGradient[index] = static_cast<float>(gradient);
        localWidth[index] = static_cast<float>(std::clamp(
            3.0 + std::sqrt(accumulation[index]) * 0.22 + strahler[index] * 1.2, 4.0, 42.0));
        localDepth[index] = static_cast<float>(std::clamp(
            1.2 + std::sqrt(accumulation[index]) * 0.065 + strahler[index] * 0.65, 1.8, 14.0));
        channelWater[index] = std::min(channelWater[index], eroded[index] - 0.35F);
        if (receiver >= 0 && strahler[receiver] > 0) {
            channelWater[receiver] = std::min(channelWater[receiver], channelWater[index] - 0.01F);
        }
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        const bool hardCap = resistance[index] > 0.35F;
        const int64_t globalGridX = key.x * INTERIOR_INTERVALS + x - RASTER_APRON;
        const int64_t globalGridZ = key.z * INTERIOR_INTERVALS + z - RASTER_APRON;
        const bool selected =
            random.uniform01(WATERFALL_STREAM, globalGridX, 0, globalGridZ) < 0.97;
        if (gradient > 0.008 && accumulation[index] > 400.0F && hardCap && selected) {
            flags[index] |= CELL_WATERFALL;
            localDepth[index] = std::min(14.0F, localDepth[index] + 2.5F);
        }
    }

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            const Site deltaSite = macro.site({key.x + dx, key.z + dz});
            const Site deltaDownstream = macro.downstream(deltaSite);
            if (sameCell(deltaSite, deltaDownstream) || deltaSite.ocean) continue;

            bool lakeEntry = false;
            double lakeRadius = 0.0;
            double waterSurface = SEA_LEVEL;
            double mouthX = 0.0;
            double mouthZ = 0.0;
            if (deltaDownstream.ocean) {
                if (deltaDownstream.elevation < SEA_LEVEL - 28.0) continue;
                const Portal mouth = sharedPortal(deltaSite, deltaDownstream);
                mouthX = mouth.x;
                mouthZ = mouth.z;
            } else {
                const Site afterLake = macro.downstream(deltaDownstream);
                if (!sameCell(afterLake, deltaDownstream)) continue;
                const TerminalLake lake = macro.terminalLake(deltaDownstream);
                if (!lake.standing) continue;
                double directionX = deltaDownstream.x - deltaSite.x;
                double directionZ = deltaDownstream.z - deltaSite.z;
                const double directionLength = std::hypot(directionX, directionZ);
                if (directionLength < 1.0) continue;
                directionX /= directionLength;
                directionZ /= directionLength;
                lakeEntry = true;
                lakeRadius = lake.radius;
                waterSurface = lake.surface;
                mouthX = lake.x - directionX * lake.radius;
                mouthZ = lake.z - directionZ * lake.radius;
            }
            const double deltaDischarge = macro.discharge(deltaSite);
            const double deltaSediment =
                deltaDischarge *
                (0.016 + std::max(0.0, deltaSite.elevation - deltaDownstream.elevation) /
                             BASIN_CATCHMENT_EDGE * 0.9);
            buildDeltaBranches(random, deltaSite, deltaDownstream, mouthX, mouthZ, waterSurface,
                               lakeEntry, lakeRadius, deltaDischarge, deltaSediment,
                               solution->deltaBranches);
        }
    }

    solution->surface.resize(RASTER_CELLS);
    solution->waterSurface.resize(RASTER_CELLS);
    solution->discharge.resize(RASTER_CELLS);
    solution->sediment.resize(RASTER_CELLS);
    solution->channelDistance.resize(RASTER_CELLS);
    solution->channelWidth.resize(RASTER_CELLS);
    solution->channelDepth.resize(RASTER_CELLS);
    solution->channelGradient.resize(RASTER_CELLS);
    solution->erosionDepth.resize(RASTER_CELLS);
    solution->lakeDepth.resize(RASTER_CELLS);
    solution->flowX = routingFlowX;
    solution->flowZ = routingFlowZ;
    solution->streamOrder = strahler;
    solution->flags = flags;
    solution->outletTypes = resolveOutletTypes(terminals, receivers, upstreamOrder);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        solution->discharge[index] = accumulation[index];
        solution->sediment[index] =
            transportedSediment[index] + accumulation[index] * localGradient[index] * 0.025F;
        solution->channelDistance[index] = channelDistance[index];
        solution->erosionDepth[index] = std::max(0.0F, raw[index] - eroded[index]);
        solution->lakeDepth[index] = classifiedLakeDepth[index];
        if ((flags[index] & CELL_OCEAN) != 0) {
            solution->surface[index] = eroded[index];
            solution->waterSurface[index] = SEA_LEVEL;
            continue;
        }
        if ((flags[index] & CELL_LAKE) != 0) {
            solution->surface[index] = lakeSurface[index] - classifiedLakeDepth[index];
            solution->waterSurface[index] = lakeSurface[index];
            solution->erosionDepth[index] =
                std::max(solution->erosionDepth[index], raw[index] - solution->surface[index]);
            continue;
        }
        const int source = nearestChannel[index];
        if (source < 0) {
            solution->surface[index] = eroded[index];
            solution->waterSurface[index] = SEA_LEVEL;
            continue;
        }
        solution->channelWidth[index] = localWidth[source];
        solution->channelDepth[index] = localDepth[source];
        solution->channelGradient[index] = localGradient[source];
        solution->waterSurface[index] = channelWater[source];
        solution->discharge[index] = accumulation[source];
        solution->sediment[index] =
            transportedSediment[source] + accumulation[source] * localGradient[source] * 0.025F;
        solution->streamOrder[index] = strahler[source];
        solution->flowX[index] = routingFlowX[source];
        solution->flowZ[index] = routingFlowZ[source];
        const double floodplainWidth = localWidth[source] * (2.4 + strahler[source] * 0.28);
        const double mask =
            1.0 - smoothstep(localWidth[source] * 0.45, floodplainWidth, channelDistance[index]);
        const double bed = channelWater[source] - localDepth[source];
        solution->surface[index] =
            static_cast<float>(eroded[index] - std::max(0.0, eroded[index] - bed) * mask);
        if (channelDistance[index] > localWidth[source] * 0.55F &&
            channelDistance[index] <= localWidth[source] * 0.55F + BASIN_RASTER_SPACING) {
            // Dry floodplain samples immediately beside a river must own a
            // bank at the water surface. Otherwise the analytical bed remains
            // several blocks below the river while the categorical wet mask
            // has already ended, creating a floating sheet with an open side.
            solution->surface[index] =
                std::max(solution->surface[index], solution->waterSurface[index]);
        }
        solution->erosionDepth[index] =
            std::max(solution->erosionDepth[index], raw[index] - solution->surface[index]);
        if ((flags[source] & CELL_WATERFALL) != 0 &&
            channelDistance[index] <= localWidth[source] * 0.60F) {
            solution->flags[index] |= CELL_WATERFALL;
        }
    }

    enforceSharedBoundaryConstraints(*solution, raw, guides);
    restoreLakeOutletMarkers(*solution, receivers, filled, floodOrder);
    normalizeLakeGeometry(*solution);
    restoreLakeOutletMarkers(*solution, receivers, filled, floodOrder, true);
    propagateLakeShoreLevels(*solution);
    appendChannelFalls(*solution, receivers);
    indexOutletFalls(*solution);

    if (!validateSolution(*solution, raw, filled, floodOrder, receivers, terminals, accumulation)) {
        throw std::runtime_error("invalid bounded basin solution");
    }
    return solution;
}

BasinSample fallbackSample(const CounterRng& random, double x, double z,
                           const BasinSolver::ElevationFunction& elevation,
                           const BasinSolver::RainfallFunction& rainfall) {
    BasinSample result;
    const BasinKey key{floorToCell(x), floorToCell(z)};
    const Site center = makeSite(random, key, elevation, rainfall);
    const Site downstream = downstreamSite(random, center, elevation, rainfall);
    const double base = elevation(x, z);
    result.surfaceElevation = base;
    result.waterSurface = SEA_LEVEL;
    result.ocean = base < SEA_LEVEL;
    result.outlet = result.ocean ? BasinOutlet::OCEAN : BasinOutlet::NONE;
    result.valid = true;
    if (!sameCell(center, downstream)) {
        const Portal portal = sharedPortal(center, downstream);
        if (!result.ocean) {
            result.outlet = BasinOutlet::SHARED_PORTAL;
            result.outletX = portal.x;
            result.outletZ = portal.z;
        }
    } else if (!result.ocean) {
        result.outlet = BasinOutlet::ENDORHEIC;
        result.outletX = center.x;
        result.outletZ = center.z;
    }
    return result;
}

BasinSample sampleSolution(const BasinSolution& solution, double x, double z) {
    const double gridX = (x - solution.originX) / BASIN_RASTER_SPACING + RASTER_APRON;
    const double gridZ = (z - solution.originZ) / BASIN_RASTER_SPACING + RASTER_APRON;
    BasinSample result;
    result.flowX = bilerp(solution.flowX, gridX, gridZ);
    result.flowZ = bilerp(solution.flowZ, gridX, gridZ);
    const double flowMagnitude = std::hypot(result.flowX, result.flowZ);
    if (flowMagnitude > 1.0e-9) {
        result.flowX /= flowMagnitude;
        result.flowZ /= flowMagnitude;
    } else {
        result.flowX = 1.0;
        result.flowZ = 0.0;
    }
    result.surfaceElevation = bilerp(solution.surface, gridX, gridZ);
    result.waterSurface = bilerp(solution.waterSurface, gridX, gridZ);
    result.discharge = bilerp(solution.discharge, gridX, gridZ);
    result.sediment = bilerp(solution.sediment, gridX, gridZ);
    result.channelDistance = bilerp(solution.channelDistance, gridX, gridZ);
    result.channelWidth = bilerp(solution.channelWidth, gridX, gridZ);
    result.channelDepth = bilerp(solution.channelDepth, gridX, gridZ);
    result.channelGradient = bilerp(solution.channelGradient, gridX, gridZ);
    result.erosionDepth = bilerp(solution.erosionDepth, gridX, gridZ);
    result.lakeDepth = bilerp(solution.lakeDepth, gridX, gridZ);
    result.lakeShoreDistance = bilerp(solution.lakeShoreDistance, gridX, gridZ);
    result.shoreWaterSurface = nearestFloat(solution.shoreWaterSurface, gridX, gridZ);
    result.streamOrder = nearestByte(solution.streamOrder, gridX, gridZ);
    const uint8_t flags = nearestByte(solution.flags, gridX, gridZ);
    const bool categoricalBackwater = (flags & CELL_BACKWATER) != 0;
    const int nearestX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int nearestZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    const int nearestIndex = indexOf(nearestX, nearestZ);
    result.outlet = static_cast<BasinOutlet>(solution.outletTypes[nearestIndex]);
    if (result.outlet == BasinOutlet::SHARED_PORTAL || result.outlet == BasinOutlet::ENDORHEIC) {
        result.outletX = solution.namedOutletX;
        result.outletZ = solution.namedOutletZ;
    } else if (result.outlet == BasinOutlet::OCEAN) {
        result.outletX = x;
        result.outletZ = z;
    }
    const LakeBodySample lakeBody = sampleDominantLakeBody(solution, gridX, gridZ);
    const bool categoricalOcean = (flags & CELL_OCEAN) != 0;
    result.ocean = categoricalOcean;
    if (!result.ocean && lakeBody.found && result.lakeShoreDistance > 0.0) {
        result.lakeDepth = lakeBody.depth.value;
        result.lake = result.lakeDepth > 0.05;
        if (result.lake) {
            result.waterSurface = lakeBody.waterLevel;
            result.surfaceElevation = result.waterSurface - result.lakeDepth;
        }
    } else {
        result.lake = false;
        result.lakeDepth = 0.0;
    }
    const bool routedChannel = result.streamOrder > 0 && result.channelWidth > 0.0 &&
                               result.channelDistance <= result.channelWidth * 0.55;
    if (!result.lake && result.surfaceElevation < SEA_LEVEL && !routedChannel) {
        result.ocean = true;
    }
    result.endorheic = result.lake && lakeBody.endorheic;
    result.river = !result.ocean && !result.lake &&
                   (routedChannel ||
                    (categoricalBackwater && result.surfaceElevation + 0.05 < result.waterSurface));
    if (!result.ocean && !result.lake && !result.river && result.channelWidth > 0.0 &&
        result.channelDistance <= result.channelWidth * 0.55 + BASIN_RASTER_SPACING) {
        // Bilinear reconstruction can move the narrow categorical channel
        // edge between raster cells. Keep every dry edge sample at or above
        // its routed water level so the exact emitter sees the same supported
        // bank as the bounded solution.
        result.surfaceElevation = std::max(result.surfaceElevation, result.waterSurface);
    }
    result.waterfall = false;

    const double localX = x - solution.originX;
    const double localZ = z - solution.originZ;
    const double edgeDistance =
        std::min({std::abs(localX), std::abs(localZ), std::abs(BASIN_CATCHMENT_EDGE - localX),
                  std::abs(BASIN_CATCHMENT_EDGE - localZ)});
    if (edgeDistance <= 1.0e-3) {
        const ChannelGuide* closest = nullptr;
        double closestDistance = std::numeric_limits<double>::max();
        for (const ChannelGuide& guide : solution.guides) {
            const double distance = std::hypot(x - guide.portalX, z - guide.portalZ);
            if (distance < closestDistance) {
                closest = &guide;
                closestDistance = distance;
            }
        }
        result.discharge = 0.0;
        result.sediment = 0.0;
        result.channelDistance = std::numeric_limits<float>::max();
        result.channelWidth = 0.0;
        result.channelDepth = 0.0;
        result.channelGradient = 0.0;
        result.streamOrder = 0;
        result.river = false;
        result.waterfall = false;
        if (closest != nullptr) {
            const int order = std::clamp(
                1 + static_cast<int>(std::floor(std::log2(closest->discharge / 500.0 + 1.0))), 1,
                6);
            const double width =
                std::clamp(3.0 + std::sqrt(closest->discharge) * 0.22 + order * 1.2, 4.0, 42.0);
            const double floodplainWidth = width * (2.4 + order * 0.28);
            if (closestDistance <= floodplainWidth) {
                result.flowX = closest->portalFlowX;
                result.flowZ = closest->portalFlowZ;
                result.waterSurface = closest->portalWater;
                result.discharge = closest->discharge;
                result.sediment = closest->discharge * (0.018 + closest->gradient * 0.9);
                result.channelDistance = closestDistance;
                result.channelWidth = width;
                result.channelDepth = std::clamp(
                    1.2 + std::sqrt(closest->discharge) * 0.065 + order * 0.65, 1.8, 14.0);
                result.channelGradient = closest->gradient;
                result.streamOrder = static_cast<uint8_t>(order);
                result.river = !result.ocean && !result.lake &&
                               (closestDistance <= result.channelWidth * 0.55 ||
                                (closest->backwater && closestDistance <= floodplainWidth &&
                                 result.surfaceElevation + 0.05 < result.waterSurface));
            }
        }
    }

    for (const DeltaBranch& branch : solution.deltaBranches) {
        double along = 0.0;
        const double distance =
            distanceToSegment(x, z, branch.startX, branch.startZ, branch.endX, branch.endZ, along);
        const double fanWidth = branch.width * (2.4 + along * 2.2);
        if (distance > fanWidth || along <= 0.0 || along >= 1.0) continue;
        const double fanMask = 1.0 - smoothstep(branch.width * 0.65, fanWidth, distance);
        result.delta = true;
        result.distributaryCount = branch.groupCount;
        result.sediment += fanMask * 18.0;
        if (result.ocean || (branch.lakeEntry && result.lake)) {
            const double depositionalShelf = branch.waterSurface - (2.0 + along * 2.5);
            result.surfaceElevation =
                std::max(result.surfaceElevation,
                         result.surfaceElevation +
                             std::max(0.0, depositionalShelf - result.surfaceElevation) * fanMask);
        }
        if (distance > branch.width * 0.65) continue;
        result.river = !result.ocean;
        result.channelDistance = distance;
        result.channelWidth = branch.width;
        result.channelDepth = std::clamp(result.channelDepth * 0.58, 1.5, 5.0);
        result.waterSurface = branch.waterSurface;
        result.surfaceElevation =
            std::min(result.surfaceElevation, result.waterSurface - result.channelDepth);
        result.erosionDepth = std::max(result.erosionDepth, result.channelDepth);
    }
    if (result.ocean) result.waterSurface = SEA_LEVEL;
    if (result.lake) {
        result.lakeDepth = std::max(0.0, result.waterSurface - result.surfaceElevation);
        if (result.lakeDepth <= 0.05) {
            result.lake = false;
            result.endorheic = false;
        }
    }
    const int fallGridX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int fallGridZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    std::array<int32_t, 9> nearbyFallIndices{};
    nearbyFallIndices.fill(-1);
    size_t nearbyFallCount = 0;
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const int candidateX = fallGridX + offsetX;
            const int candidateZ = fallGridZ + offsetZ;
            if (!inRaster(candidateX, candidateZ)) continue;
            const int32_t fallIndex = solution.outletFallByCell[indexOf(candidateX, candidateZ)];
            if (fallIndex < 0 ||
                std::find(nearbyFallIndices.begin(),
                          nearbyFallIndices.begin() + static_cast<std::ptrdiff_t>(nearbyFallCount),
                          fallIndex) !=
                    nearbyFallIndices.begin() + static_cast<std::ptrdiff_t>(nearbyFallCount)) {
                continue;
            }
            nearbyFallIndices[nearbyFallCount++] = fallIndex;
        }
    }
    for (size_t nearbyIndex = 0; nearbyIndex < nearbyFallCount; ++nearbyIndex) {
        const OutletFall& fall =
            solution.outletFalls[static_cast<size_t>(nearbyFallIndices[nearbyIndex])];
        double flowX = fall.endX - fall.startX;
        double flowZ = fall.endZ - fall.startZ;
        const double flowLength = std::hypot(flowX, flowZ);
        if (flowLength <= 1.0e-6) continue;
        flowX /= flowLength;
        flowZ /= flowLength;
        const double offsetX = x - fall.endX;
        const double offsetZ = z - fall.endZ;
        const double longitudinal = offsetX * flowX + offsetZ * flowZ;
        const double crossStream = std::abs(-offsetX * flowZ + offsetZ * flowX);
        const double halfDepth = std::clamp(fall.halfWidth * 0.18, 0.75, 1.5);
        if (std::abs(longitudinal) > halfDepth || crossStream > fall.halfWidth) continue;
        if (!result.waterfall || fall.topSurface > result.waterfallTop) {
            result.waterfallTop = fall.topSurface;
            result.waterfallBottom = fall.bottomSurface;
            result.waterfallWidth = fall.halfWidth * 2.0F;
            result.flowX = flowX;
            result.flowZ = flowZ;
        }
        result.waterSurface =
            std::min(result.waterSurface, static_cast<double>(fall.bottomSurface));
        if (!result.lake && fall.bottomSurface <= SEA_LEVEL + 1.0e-4F &&
            result.surfaceElevation < SEA_LEVEL) {
            // The explicit fall owns the narrow transition into the lower
            // ocean. Channel incision below sea level remains a river
            // everywhere else, but the receiving footprint is ocean water.
            result.ocean = true;
            result.river = false;
        }
        result.waterfall = true;
        result.waterfallAnchor = result.waterfallAnchor || (std::abs(x - fall.endX) <= 1.0e-6 &&
                                                            std::abs(z - fall.endZ) <= 1.0e-6);
    }
    if (!result.ocean && !result.lake && !result.river && !result.waterfall &&
        result.shoreWaterSurface > 0.0 && result.lakeShoreDistance <= 0.0 &&
        result.lakeShoreDistance > -LAKE_BANK_WIDTH) {
        result.lakeBankInfluence =
            1.0 - smoothstep(0.0, LAKE_BANK_WIDTH, -result.lakeShoreDistance);
        const double bankCrest = std::ceil(result.shoreWaterSurface);
        result.lakeBankTarget =
            result.surfaceElevation +
            std::max(0.0, bankCrest - result.surfaceElevation) * result.lakeBankInfluence;
        result.surfaceElevation = std::max(result.surfaceElevation, result.lakeBankTarget);
        result.lakeBank = result.lakeBankInfluence > 1.0e-4;
    }
    result.valid = std::isfinite(result.surfaceElevation) && std::isfinite(result.waterSurface) &&
                   std::isfinite(result.discharge) && std::isfinite(result.erosionDepth) &&
                   std::isfinite(result.lakeShoreDistance) &&
                   std::isfinite(result.shoreWaterSurface) &&
                   std::isfinite(result.lakeBankTarget) && std::isfinite(result.waterfallTop) &&
                   std::isfinite(result.waterfallBottom);
    return result;
}

} // namespace

class BasinSolver::Impl {
public:
    using SolutionPointer = std::shared_ptr<const BasinSolution>;

    struct Entry {
        std::shared_future<SolutionPointer> future;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
        uint64_t token = 0;
    };

    explicit Impl(uint64_t seed, size_t requestedBudget)
        : random(seed)
        , byteBudget(std::max<size_t>(1, requestedBudget)) {}

    SolutionPointer getOrCreate(BasinKey key, const ElevationFunction& elevation,
                                const RainfallFunction& rainfall,
                                const RockResistanceFunction& rockResistance) const {
        std::shared_future<SolutionPointer> future;
        std::shared_ptr<std::promise<SolutionPointer>> producer;
        uint64_t token = 0;
        {
            std::lock_guard lock(mutex);
            auto found = entries.find(key);
            if (found != entries.end()) {
                found->second.lastAccess = ++accessClock;
                ++metrics.hits;
                future = found->second.future;
            } else {
                ++metrics.misses;
                producer = std::make_shared<std::promise<SolutionPointer>>();
                future = producer->get_future().share();
                token = ++tokenClock;
                entries.emplace(key, Entry{future, 0, ++accessClock, token});
            }
        }
        if (!producer) return future.get();

        try {
            SolutionPointer solution;
            {
                ColdBasinBuildPermit permit;
                solution = buildSolution(random, key, elevation, rainfall, rockResistance);
            }
            producer->set_value(solution);
            std::lock_guard lock(mutex);
            auto found = entries.find(key);
            if (found != entries.end() && found->second.token == token) {
                found->second.bytes = solution->byteSize();
                metrics.bytes += found->second.bytes;
                ++metrics.builds;
                while (metrics.bytes > byteBudget && entries.size() > 1) {
                    auto oldest = entries.end();
                    for (auto iterator = entries.begin(); iterator != entries.end(); ++iterator) {
                        if (iterator->first == key || iterator->second.bytes == 0) continue;
                        if (oldest == entries.end() ||
                            iterator->second.lastAccess < oldest->second.lastAccess) {
                            oldest = iterator;
                        }
                    }
                    if (oldest == entries.end()) break;
                    metrics.bytes -= oldest->second.bytes;
                    entries.erase(oldest);
                }
                metrics.entries = entries.size();
            }
            return solution;
        } catch (...) {
            producer->set_exception(std::current_exception());
            std::lock_guard lock(mutex);
            auto found = entries.find(key);
            if (found != entries.end() && found->second.token == token) entries.erase(found);
            ++metrics.failures;
            metrics.entries = entries.size();
            throw;
        }
    }

    mutable std::mutex mutex;
    mutable std::unordered_map<BasinKey, Entry, BasinKeyHash> entries;
    CounterRng random;
    size_t byteBudget = BASIN_CACHE_BYTE_BUDGET;
    mutable uint64_t accessClock = 0;
    mutable uint64_t tokenClock = 0;
    mutable BasinCacheMetrics metrics;
};

BasinSolver::BasinSolver(uint64_t worldSeed, size_t cacheByteBudget)
    : impl_(std::make_unique<Impl>(worldSeed, cacheByteBudget)) {}

BasinSolver::~BasinSolver() = default;
BasinSolver::BasinSolver(BasinSolver&&) noexcept = default;
BasinSolver& BasinSolver::operator=(BasinSolver&&) noexcept = default;

BasinSample BasinSolver::sample(double x, double z, const ElevationFunction& elevation,
                                const RainfallFunction& rainfall,
                                const RockResistanceFunction& rockResistance) const {
    const BasinKey key{floorToCell(x), floorToCell(z)};
    try {
        const std::shared_ptr<const BasinSolution> solution =
            impl_->getOrCreate(key, elevation, rainfall, rockResistance);
        const BasinSample result = sampleSolution(*solution, x, z);
        if (result.valid) return result;
    } catch (const std::exception&) {
    }
    return fallbackSample(impl_->random, x, z, elevation, rainfall);
}

BasinCacheMetrics BasinSolver::cacheMetrics() const {
    BasinCacheMetrics result;
    {
        std::lock_guard lock(impl_->mutex);
        result = impl_->metrics;
        result.entries = impl_->entries.size();
    }
    const ColdBasinBuildMetrics construction = coldBasinBuildLimiter().metrics();
    result.activeColdBuilds = construction.active;
    result.peakColdBuilds = construction.peak;
    result.throttledBuilds = construction.throttled;
    return result;
}

void BasinSolver::clear() const {
    std::lock_guard lock(impl_->mutex);
    impl_->entries.clear();
    impl_->metrics.entries = 0;
    impl_->metrics.bytes = 0;
}

} // namespace worldgen
