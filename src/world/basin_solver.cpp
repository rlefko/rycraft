#include "world/basin_solver.hpp"

#include "common/counter_rng.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <queue>
#include <ranges>
#include <stdexcept>
#include <tuple>
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
constexpr int EROSION_EPOCHS = 4;
constexpr int EROSION_PASSES_PER_EPOCH = 2;
static_assert(EROSION_EPOCHS * EROSION_PASSES_PER_EPOCH == 8);
constexpr int SHARED_EDGE_BAND_CELLS = 4;
constexpr double LAKE_BANK_WIDTH = 16.0;
constexpr double LAKE_SHELF_WIDTH = 16.0;
constexpr double LAKE_SHORE_PERTURBATION_BAND = 56.0;
constexpr double LAKE_SHORE_PERTURBATION_AMPLITUDE = 5.5;
constexpr int SHORELINE_COARSE_INTERVALS =
    static_cast<int>(SHORELINE_PAGE_EDGE / SHORELINE_COARSE_SPACING);
constexpr int SHORELINE_COARSE_EDGE = SHORELINE_COARSE_INTERVALS + 3;
constexpr double SHORELINE_DISTANCE_LIMIT = 96.0;
constexpr double SHORELINE_FLOOR_QUANTIZATION = 16.0;
constexpr double SHORELINE_DISTANCE_QUANTIZATION = 4.0;
constexpr int PORTAL_PROFILE_MAX_EDGES = 64;
constexpr int PORTAL_PROFILE_TERRAIN_SAMPLES = 17;
constexpr int CHANNEL_GUIDE_CURVE_SEGMENTS = 16;
constexpr double MIN_CHANNEL_DISCHARGE = 105.0;
// D-infinity routing is acyclic, so a complete receiver walk is bounded by
// the raster itself. Stopping after a catchment-width heuristic could strand
// a valid through-flow lake before its real portal or receiving water.
constexpr int MAX_OUTLET_CORRIDOR_CELLS = RASTER_CELLS;
constexpr uint16_t NO_LAKE_EQUILIBRIUM = std::numeric_limits<uint16_t>::max();
constexpr uint64_t SITE_POSITION_STREAM = 0xB451'0001ULL;
constexpr uint64_t SITE_PROPERTY_STREAM = 0xB451'0002ULL;
constexpr uint64_t WATERFALL_STREAM = 0xB451'0003ULL;
constexpr uint64_t DELTA_STREAM = 0xB451'0004ULL;
constexpr uint64_t CHANNEL_MEANDER_STREAM = 0xB451'0005ULL;

uint64_t nextCacheInstanceToken() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

size_t fastHitShard() {
    static std::atomic<size_t> next{0};
    static thread_local const size_t shard = next.fetch_add(1, std::memory_order_relaxed) % 16;
    return shard;
}

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

struct ShorelinePageKey {
    WaterBodyId body = NO_WATER_BODY;
    int64_t x = 0;
    int64_t z = 0;

    bool operator==(const ShorelinePageKey&) const = default;
};

struct ShorelinePageKeyHash {
    size_t operator()(const ShorelinePageKey& key) const noexcept {
        BasinKeyHash hash;
        const size_t position = hash({key.x, key.z});
        uint64_t body = key.body + static_cast<uint64_t>(position);
        body ^= body >> 30U;
        body *= 0xBF58'476D'1CE4'E5B9ULL;
        body ^= body >> 27U;
        body *= 0x94D0'49BB'1331'11EBULL;
        body ^= body >> 31U;
        return position ^ static_cast<size_t>(body);
    }
};

struct ShorelineNode {
    int16_t signedDistance = 0;
    int16_t floor = 0;
    int16_t bankWaterLevel = 0;
};

struct RefinedShorelineCell {
    uint16_t coarseCell = 0;
    std::array<ShorelineNode, 5> midpointNodes{};
};

struct ShorelineContourPage {
    ShorelinePageKey key;
    float waterLevel = 0.0F;
    bool endorheic = false;
    std::array<ShorelineNode, SHORELINE_COARSE_EDGE * SHORELINE_COARSE_EDGE> coarse{};
    std::vector<RefinedShorelineCell> refined;

    size_t byteSize() const {
        return sizeof(*this) + refined.capacity() * sizeof(RefinedShorelineCell);
    }
};

struct ShorelineContourSample {
    double signedDistance = -SHORELINE_DISTANCE_LIMIT;
    double floor = 0.0;
    double waterLevel = 0.0;
    double bankWaterLevel = 0.0;
    WaterBodyId identity = NO_WATER_BODY;
    bool endorheic = false;
    bool outletContinuation = false;
    bool valid = false;
};

struct LakeAuthoritySample {
    WaterBodyId identity = NO_WATER_BODY;
    double waterLevel = 0.0;
    double bankWaterLevel = 0.0;
    double signedDistance = -SHORELINE_DISTANCE_LIMIT;
    double floor = 0.0;
    bool endorheic = false;
    bool found = false;
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
    float discharge = 0.0F;
    float halfWidth = 0.0F;
    bool transitionPromotion = false;
};

using RapidOwnerKind = WaterTransitionKind;

struct RapidCell {
    int64_t x = 0;
    int64_t z = 0;
    int32_t topY = 0;
    int32_t ownerIndex = -1;
    uint64_t ownerId = 0;
    float flowX = 0.0F;
    float flowZ = 0.0F;
    float channelDistance = 0.0F;
    float channelWidth = 0.0F;
    uint8_t level = 0;
    RapidOwnerKind ownerKind = RapidOwnerKind::CHANNEL_GUIDE;
};

struct TransitionDescriptor {
    uint64_t ownerId = 0;
    int32_t ownerIndex = -1;
    int32_t topY = 0;
    float flowX = 0.0F;
    float flowZ = 0.0F;
    float channelWidth = 0.0F;
    RapidOwnerKind ownerKind = RapidOwnerKind::CHANNEL_GUIDE;
};

struct TransitionCell {
    int16_t localX = 0;
    int16_t localZ = 0;
    uint16_t descriptorIndex = 0;
    uint8_t level = 0;
    uint8_t distanceQuarterBlocks = 0;
};

static_assert(sizeof(TransitionDescriptor) == 32);
static_assert(sizeof(TransitionCell) == 8);

struct SettledWaterCell {
    int16_t localX = 0;
    int16_t localZ = 0;
    int16_t surfaceEighths = 0;
    int16_t bedEighths = 0;
    int32_t ownerIndex = -1;
    uint8_t fluidPacked = 0;
    RapidOwnerKind ownerKind = RapidOwnerKind::RASTER_CHANNEL;
    uint16_t reserved = 0;
};

static_assert(sizeof(SettledWaterCell) == 16);

struct SettledBankCell {
    int16_t localX = 0;
    int16_t localZ = 0;
    int16_t targetEighths = 0;
    uint16_t reserved = 0;
};

static_assert(sizeof(SettledBankCell) == 8);

// Guides, routed raster edges, and their eight-block approaches remain inside
// this fixed basin-relative band. A dense row table makes exact block lookup
// independent of the number of transitions elsewhere in the catchment.
constexpr int TRANSITION_MIN_LOCAL = -256;
constexpr int TRANSITION_MAX_LOCAL = 2304;
constexpr size_t TRANSITION_ROW_COUNT =
    static_cast<size_t>(TRANSITION_MAX_LOCAL - TRANSITION_MIN_LOCAL + 1);

struct LakeEquilibrium {
    WaterBodyId identity = NO_WATER_BODY;
    float surface = 0.0F;
    float spillSurface = 0.0F;
    float areaSquareKilometers = 0.0F;
    float volumeCubicMeters = 0.0F;
    float runoffMmSquareKilometers = 0.0F;
    float lossMm = 0.0F;
    float overflowMmSquareKilometers = 0.0F;
    float maximumDepth = 0.0F;
    bool endorheic = false;
};

struct GuidePoint {
    double x = 0.0;
    double z = 0.0;
};

struct LakeOutletCorridor {
    std::vector<GuidePoint> points;
    std::vector<GuidePoint> receivingRoute;
    std::vector<float> waterSurface;
    WaterBodyId body = NO_WATER_BODY;
    int32_t outletCell = -1;
    float discharge = 0.0F;
    float halfWidth = 0.0F;
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
    double startWater = SEA_LEVEL;
    double endWater = SEA_LEVEL;
    double terrainUpper = SEA_LEVEL;
    double gradient = 0.0;
    double portalFlowX = 1.0;
    double portalFlowZ = 0.0;
    double widthPhase = 0.0;
    double widthVariation = 0.0;
    double minX = 0.0;
    double maxX = 0.0;
    double minZ = 0.0;
    double maxZ = 0.0;
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

uint64_t transitionIdentityMix(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

uint64_t transitionIdentity(uint64_t domain, double x, double z, double directionX = 0.0,
                            double directionZ = 0.0) {
    const auto quantized = [](double value) {
        return static_cast<uint64_t>(static_cast<int64_t>(std::llround(value * 1024.0)));
    };
    uint64_t identity = transitionIdentityMix(domain ^ quantized(x));
    identity = transitionIdentityMix(identity ^ transitionIdentityMix(quantized(z)));
    identity = transitionIdentityMix(identity ^ transitionIdentityMix(quantized(directionX)));
    return transitionIdentityMix(identity ^ transitionIdentityMix(quantized(directionZ)));
}

uint64_t guideTransitionIdentity(const ChannelGuide& guide) {
    return transitionIdentity(0x47554944455F4544ULL, guide.portalX, guide.portalZ,
                              guide.portalFlowX, guide.portalFlowZ);
}

uint64_t corridorTransitionIdentity(const LakeOutletCorridor& corridor) {
    if (corridor.points.empty()) return transitionIdentityMix(corridor.body);
    return transitionIdentity(corridor.body ^ 0x434F525249444F52ULL, corridor.points.front().x,
                              corridor.points.front().z);
}

uint64_t rasterDrainageIdentity(const Site& root) {
    return transitionIdentity(0x5241535445525F52ULL, root.x, root.z,
                              static_cast<double>(root.cell.x), static_cast<double>(root.cell.z));
}

enum CellFlags : uint8_t {
    CELL_OCEAN = 1U << 0U,
    CELL_LAKE = 1U << 1U,
    CELL_ENDORHEIC = 1U << 2U,
    CELL_WATERFALL = 1U << 3U,
    CELL_LAKE_OUTLET = 1U << 4U,
    CELL_BACKWATER = 1U << 5U,
    CELL_LAKE_CONNECTOR = 1U << 6U,
};

struct BasinSolution {
    BasinKey key;
    uint64_t erosionEpochs = 0;
    uint64_t erosionReroutes = 0;
    uint64_t erosionReceiverChanges = 0;
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
    std::vector<WaterBodyId> waterBodyIds;
    std::vector<WaterBodyId> shoreWaterBodyIds;
    std::vector<uint8_t> shoreWaterEndorheic;
    std::vector<float> flowX;
    std::vector<float> flowZ;
    std::vector<int32_t> rasterReceivers;
    std::vector<float> rasterSeaBackwaterDistance;
    std::vector<uint8_t> streamOrder;
    std::vector<uint8_t> flags;
    std::vector<uint8_t> outletTypes;
    std::vector<uint16_t> lakeEquilibriumByCell;
    std::vector<LakeEquilibrium> lakeEquilibria;
    std::vector<ChannelGuide> guides;
    double namedOutletX = 0.0;
    double namedOutletZ = 0.0;
    uint64_t rasterDrainageOwnerId = 0;
    std::vector<DeltaBranch> deltaBranches;
    std::vector<OutletFall> outletFalls;
    std::vector<RapidCell> rapidCells;
    std::vector<TransitionDescriptor> transitionDescriptors;
    std::vector<TransitionCell> transitionCells;
    std::vector<uint32_t> transitionRowOffsets;
    std::vector<SettledWaterCell> settledWaterCells;
    std::vector<uint32_t> settledWaterRowOffsets;
    std::vector<SettledBankCell> settledBankCells;
    std::vector<uint32_t> settledBankRowOffsets;
    std::vector<LakeOutletCorridor> outletCorridors;
    std::vector<int32_t> outletFallByCell;

    size_t byteSize() const {
        size_t total = sizeof(*this) + deltaBranches.capacity() * sizeof(DeltaBranch) +
                       outletFalls.capacity() * sizeof(OutletFall) +
                       rapidCells.capacity() * sizeof(RapidCell) +
                       transitionDescriptors.capacity() * sizeof(TransitionDescriptor) +
                       transitionCells.capacity() * sizeof(TransitionCell) +
                       transitionRowOffsets.capacity() * sizeof(uint32_t) +
                       settledWaterCells.capacity() * sizeof(SettledWaterCell) +
                       settledWaterRowOffsets.capacity() * sizeof(uint32_t) +
                       settledBankCells.capacity() * sizeof(SettledBankCell) +
                       settledBankRowOffsets.capacity() * sizeof(uint32_t) +
                       outletCorridors.capacity() * sizeof(LakeOutletCorridor) +
                       lakeEquilibria.capacity() * sizeof(LakeEquilibrium) +
                       guides.capacity() * sizeof(ChannelGuide);
        for (const LakeOutletCorridor& corridor : outletCorridors) {
            total += corridor.points.capacity() * sizeof(GuidePoint);
            total += corridor.receivingRoute.capacity() * sizeof(GuidePoint);
            total += corridor.waterSurface.capacity() * sizeof(float);
        }
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
        total += waterBodyIds.capacity() * sizeof(WaterBodyId);
        total += shoreWaterBodyIds.capacity() * sizeof(WaterBodyId);
        total += shoreWaterEndorheic.capacity() * sizeof(uint8_t);
        total += flowX.capacity() * sizeof(float);
        total += flowZ.capacity() * sizeof(float);
        total += rasterReceivers.capacity() * sizeof(int32_t);
        total += rasterSeaBackwaterDistance.capacity() * sizeof(float);
        total += outletFallByCell.capacity() * sizeof(int32_t);
        total += streamOrder.capacity() * sizeof(uint8_t);
        total += flags.capacity() * sizeof(uint8_t);
        total += outletTypes.capacity() * sizeof(uint8_t);
        total += lakeEquilibriumByCell.capacity() * sizeof(uint16_t);
        return total;
    }
};

LakeAuthoritySample sampleLakeAuthority(const BasinSolution& solution, double x, double z);
double dominantPositiveLevel(const BasinSolution& solution, double gridX, double gridZ);

int indexOf(int x, int z) {
    return z * RASTER_EDGE + x;
}

int32_t rasterEdgeIndex(int source, int receiver) {
    const int sourceX = source % RASTER_EDGE;
    const int sourceZ = source / RASTER_EDGE;
    const int receiverX = receiver % RASTER_EDGE;
    const int receiverZ = receiver / RASTER_EDGE;
    for (int direction = 0; direction < 8; ++direction) {
        if (sourceX + DX[direction] == receiverX && sourceZ + DZ[direction] == receiverZ) {
            return static_cast<int32_t>(source * 8 + direction);
        }
    }
    return -1;
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

uint64_t mix64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xBF58'476D'1CE4'E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

uint64_t combine64(uint64_t first, uint64_t second) {
    return mix64(first ^ (mix64(second) + 0x9E37'79B9'7F4A'7C15ULL));
}

double hashUnit(uint64_t value) {
    constexpr double INVERSE_53_BITS = 1.0 / static_cast<double>(uint64_t{1} << 53U);
    return static_cast<double>(mix64(value) >> 11U) * INVERSE_53_BITS;
}

WaterBodyId makeWaterBodyId(int64_t anchorX, int64_t anchorZ, double waterLevel, bool endorheic) {
    const int64_t quantizedLevel = static_cast<int64_t>(std::llround(waterLevel * 1'024.0));
    uint64_t identity = combine64(static_cast<uint64_t>(anchorX), static_cast<uint64_t>(anchorZ));
    identity = combine64(identity, static_cast<uint64_t>(quantizedLevel));
    identity = combine64(identity, endorheic ? 0xE0D0'4E1CULL : 0x7A10'0E71ULL);
    return identity == NO_WATER_BODY ? 1 : identity;
}

float canonicalLakeSurface(double waterLevel) {
    return static_cast<float>(std::round(waterLevel * 1024.0) / 1024.0);
}

double organicShoreNoise(WaterBodyId identity, double worldX, double worldZ) {
    constexpr std::array<double, 3> WAVELENGTHS = {39.0, 67.0, 109.0};
    constexpr std::array<double, 3> WEIGHTS = {0.52, 0.31, 0.17};
    double result = 0.0;
    for (size_t octave = 0; octave < WAVELENGTHS.size(); ++octave) {
        const uint64_t key = combine64(identity, octave + 0x51A9U);
        // Keep every wave well away from the cardinal axes. Combining three
        // unrelated oblique directions avoids replacing square raster edges
        // with one visibly periodic shoreline direction.
        const double angle = (0.12 + hashUnit(key) * 0.76) * std::numbers::pi;
        const double phase = hashUnit(key ^ 0xA17E'5EEDULL) * 2.0 * std::numbers::pi;
        const double projected = worldX * std::cos(angle) + worldZ * std::sin(angle);
        result += WEIGHTS[octave] *
                  std::sin(projected * (2.0 * std::numbers::pi / WAVELENGTHS[octave]) + phase);
    }
    return result;
}

int64_t floorToCell(double coordinate) {
    return world_coord::floorToNeighborSafeInt64(coordinate / BASIN_CATCHMENT_EDGE);
}

BasinKey samplingKey(double x, double z) {
    return {floorToCell(x), floorToCell(z)};
}

bool sameCell(const Site& first, const Site& second) {
    return first.cell == second.cell;
}

bool siteLower(const Site& candidate, const Site& current) {
    // Exact finite elevation order is already a strict transitive potential.
    // The immutable tie and coordinates resolve only genuinely equal values;
    // coarser centimeter bands can otherwise route a slightly higher site
    // downhill by rank and publish a portal that does not belong to the
    // current catchment.
    if (candidate.elevation != current.elevation) {
        if (!std::isfinite(candidate.elevation) || !std::isfinite(current.elevation)) {
            if (std::isnan(candidate.elevation)) return false;
            if (std::isnan(current.elevation)) return true;
        }
        return candidate.elevation < current.elevation;
    }
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

struct GuideProjection {
    double distance = std::numeric_limits<double>::max();
    double along = 0.0;
    double x = 0.0;
    double z = 0.0;
    double tangentX = 1.0;
    double tangentZ = 0.0;
};

GuideProjection projectGuide(const ChannelGuide& guide, double worldX, double worldZ) {
    double bestDistanceSquared = std::numeric_limits<double>::max();
    GuideProjection result;
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
            result.along = (segment + segmentAlong) / CHANNEL_GUIDE_CURVE_SEGMENTS;
            result.x = start.x + directionX * segmentAlong;
            result.z = start.z + directionZ * segmentAlong;
            const double length = std::sqrt(lengthSquared);
            if (length > 1.0e-9) {
                result.tangentX = directionX / length;
                result.tangentZ = directionZ / length;
            }
        }
    }

    // Extend the canonical portal tangent through the numerical apron. The
    // neighboring catchment owns the other Hermite half, but apron sampling
    // must still see the same channel centerline instead of snapping to a
    // different nearby guide after the curve endpoint.
    const double portalOffsetX = worldX - guide.portalX;
    const double portalOffsetZ = worldZ - guide.portalZ;
    const double portalAlong =
        portalOffsetX * guide.portalFlowX + portalOffsetZ * guide.portalFlowZ;
    const bool beyondPortal = guide.outgoing ? portalAlong > 0.0 : portalAlong < 0.0;
    if (std::hypot(portalOffsetX, portalOffsetZ) <= 1.0e-3) {
        bestDistanceSquared = 0.0;
        result.along = guide.outgoing ? 1.0 : 0.0;
        result.x = guide.portalX;
        result.z = guide.portalZ;
        result.tangentX = guide.portalFlowX;
        result.tangentZ = guide.portalFlowZ;
    } else if (beyondPortal) {
        const double crossStream =
            std::abs(-portalOffsetX * guide.portalFlowZ + portalOffsetZ * guide.portalFlowX);
        if (crossStream * crossStream < bestDistanceSquared) {
            bestDistanceSquared = crossStream * crossStream;
            result.along = guide.outgoing ? 1.0 : 0.0;
            result.x = guide.portalX;
            result.z = guide.portalZ;
            result.tangentX = guide.portalFlowX;
            result.tangentZ = guide.portalFlowZ;
        }
    }
    result.distance = std::sqrt(bestDistanceSquared);
    return result;
}

double distanceToGuide(const ChannelGuide& guide, double worldX, double worldZ, double& along) {
    const GuideProjection projection = projectGuide(guide, worldX, worldZ);
    along = projection.along;
    return projection.distance;
}

double quinticWeight(double value) {
    const double t = clamp01(value);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

struct RasterWeights {
    int x0 = 0;
    int z0 = 0;
    std::array<int, 4> indices{};
    std::array<double, 4> values{};
};

RasterWeights reconstructionWeights(double gridX, double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double fx = quinticWeight(gridX - x0);
    const double fz = quinticWeight(gridZ - z0);
    return {
        .x0 = x0,
        .z0 = z0,
        .indices = {indexOf(x0, z0), indexOf(x0 + 1, z0), indexOf(x0, z0 + 1),
                    indexOf(x0 + 1, z0 + 1)},
        .values = {(1.0 - fx) * (1.0 - fz), fx * (1.0 - fz), (1.0 - fx) * fz, fx * fz},
    };
}

float reconstruct(const std::vector<float>& field, double gridX, double gridZ) {
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    double result = 0.0;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        result += field[weights.indices[index]] * weights.values[index];
    }
    return static_cast<float>(result);
}

double reconstructChannelDistance(const BasinSolution& solution, double gridX, double gridZ) {
    constexpr double MAX_CHANNEL_DISTANCE =
        (BASIN_CATCHMENT_EDGE + 2.0 * RASTER_APRON * BASIN_RASTER_SPACING) * std::numbers::sqrt2;
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    double result = 0.0;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        const double distance = solution.channelDistance[weights.indices[index]];
        result += std::clamp(distance, 0.0, MAX_CHANNEL_DISTANCE) * weights.values[index];
    }
    return result;
}

int nearestIndex(double gridX, double gridZ) {
    const int x = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int z = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    return indexOf(x, z);
}

double flagWeight(const BasinSolution& solution, double gridX, double gridZ, uint8_t flag) {
    const int cellX = static_cast<int>(std::floor(gridX));
    const int cellZ = static_cast<int>(std::floor(gridZ));
    const auto cubicWeights = [](double amount) {
        const double t = std::clamp(amount, 0.0, 1.0);
        const double t2 = t * t;
        const double t3 = t2 * t;
        return std::array{
            (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
            (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
            (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
            t3 / 6.0,
        };
    };
    const auto broaden = [](const std::array<double, 4>& cubic) {
        // Convolving the cubic basis with a three-control binomial kernel
        // gives categorical ocean and backwater masks a 96-block C2 support.
        // This avoids accelerating a smooth channel by several eighth-levels
        // in one block as it approaches a categorical control transition.
        return std::array{
            cubic[0] * 0.25,
            (cubic[1] + 2.0 * cubic[0]) * 0.25,
            (cubic[2] + 2.0 * cubic[1] + cubic[0]) * 0.25,
            (cubic[3] + 2.0 * cubic[2] + cubic[1]) * 0.25,
            (2.0 * cubic[3] + cubic[2]) * 0.25,
            cubic[3] * 0.25,
        };
    };
    const std::array<double, 6> weightsX = broaden(cubicWeights(gridX - cellX));
    const std::array<double, 6> weightsZ = broaden(cubicWeights(gridZ - cellZ));
    double result = 0.0;
    for (int offsetZ = 0; offsetZ < 6; ++offsetZ) {
        const int sampleZ = std::clamp(cellZ + offsetZ - 2, 0, RASTER_EDGE - 1);
        for (int offsetX = 0; offsetX < 6; ++offsetX) {
            const int sampleX = std::clamp(cellX + offsetX - 2, 0, RASTER_EDGE - 1);
            if ((solution.flags[indexOf(sampleX, sampleZ)] & flag) == 0) continue;
            result +=
                weightsX[static_cast<size_t>(offsetX)] * weightsZ[static_cast<size_t>(offsetZ)];
        }
    }
    return result;
}

uint8_t reconstructedStreamOrder(const BasinSolution& solution, double gridX, double gridZ) {
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    double weightedOrder = 0.0;
    double activeWeight = 0.0;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        const uint8_t order = solution.streamOrder[weights.indices[index]];
        if (order == 0) continue;
        weightedOrder += order * weights.values[index];
        activeWeight += weights.values[index];
    }
    if (activeWeight <= 1.0e-9) return 0;
    return static_cast<uint8_t>(std::clamp(std::lround(weightedOrder / activeWeight), 1L, 255L));
}

struct ConditionedSample {
    double value = 0.0;
    double weight = 0.0;
};

struct LakeBodySample {
    ConditionedSample depth;
    double waterLevel = 0.0;
    WaterBodyId identity = NO_WATER_BODY;
    bool endorheic = false;
    bool found = false;
};

ConditionedSample sampleLakeBodyField(const BasinSolution& solution,
                                      const std::vector<float>& field, double gridX, double gridZ,
                                      WaterBodyId identity) {
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    ConditionedSample result;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        const int cell = weights.indices[index];
        if ((solution.flags[cell] & CELL_LAKE) == 0 || solution.waterBodyIds[cell] != identity) {
            continue;
        }
        result.value += field[cell] * weights.values[index];
        result.weight += weights.values[index];
    }
    return result;
}

double authoritativeLakeSurface(const BasinSolution& solution, int cell, WaterBodyId identity,
                                double fallback) {
    if (cell < 0 || cell >= RASTER_CELLS || solution.lakeEquilibriumByCell.empty()) {
        return fallback;
    }
    const uint16_t equilibrium = solution.lakeEquilibriumByCell[static_cast<size_t>(cell)];
    if (equilibrium >= solution.lakeEquilibria.size() ||
        solution.lakeEquilibria[equilibrium].identity != identity) {
        return fallback;
    }
    return solution.lakeEquilibria[equilibrium].surface;
}

LakeBodySample sampleDominantLakeBody(const BasinSolution& solution, double gridX, double gridZ) {
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    struct Candidate {
        WaterBodyId identity = NO_WATER_BODY;
        double waterLevel = 0.0;
        double membership = 0.0;
        bool endorheic = false;
    };
    std::array<Candidate, 4> candidates{};
    size_t candidateCount = 0;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        const int cell = weights.indices[index];
        const WaterBodyId identity = solution.waterBodyIds[cell];
        if ((solution.flags[cell] & CELL_LAKE) == 0 || identity == NO_WATER_BODY ||
            weights.values[index] <= 0.0) {
            continue;
        }
        const double waterLevel =
            authoritativeLakeSurface(solution, cell, identity, solution.waterSurface[cell]);
        const bool endorheic = (solution.flags[cell] & CELL_ENDORHEIC) != 0;
        size_t candidate = 0;
        while (candidate < candidateCount && candidates[candidate].identity != identity) {
            ++candidate;
        }
        if (candidate == candidateCount) {
            candidates[candidateCount++] = {
                .identity = identity,
                .waterLevel = waterLevel,
                .membership = 0.0,
                .endorheic = endorheic,
            };
        }
        candidates[candidate].membership += weights.values[index];
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
            sampleLakeBodyField(solution, solution.lakeDepth, gridX, gridZ, selected->identity),
        .waterLevel = selected->waterLevel,
        .identity = selected->identity,
        .endorheic = selected->endorheic,
        .found = true,
    };
}

LakeBodySample sampleCandidateLakeBody(const BasinSolution& solution, double gridX, double gridZ) {
    LakeBodySample result = sampleDominantLakeBody(solution, gridX, gridZ);
    if (result.found) return result;
    if (solution.shoreWaterBodyIds.empty() || solution.shoreWaterSurface.empty() ||
        solution.shoreWaterEndorheic.empty()) {
        return {};
    }
    const int nearest = nearestIndex(gridX, gridZ);
    const WaterBodyId identity = solution.shoreWaterBodyIds[nearest];
    const double waterLevel =
        authoritativeLakeSurface(solution, nearest, identity, solution.shoreWaterSurface[nearest]);
    if (identity == NO_WATER_BODY || waterLevel <= 0.0) return {};
    result.identity = identity;
    result.waterLevel = waterLevel;
    result.endorheic = solution.shoreWaterEndorheic[nearest] != 0;
    result.found = true;
    return result;
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

    Site drainageRoot(const Site& value) {
        Site current = value;
        std::unordered_set<BasinKey, BasinKeyHash> visited;
        constexpr int MAX_DRAINAGE_ROOT_EDGES = 256;
        visited.reserve(MAX_DRAINAGE_ROOT_EDGES);
        for (int edge = 0; edge < MAX_DRAINAGE_ROOT_EDGES; ++edge) {
            if (!visited.insert(current.cell).second) {
                throw std::runtime_error("canonical drainage root contains a cycle");
            }
            const Site next = downstream(current);
            if (sameCell(next, current) || next.ocean) return next;
            current = next;
        }
        // Static coordinate fields can form exceptionally long downhill
        // paths. The bounded terminal remains deterministic and preserves
        // random access rather than invalidating an otherwise sound basin.
        return current;
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

    double junctionWater(const Site& junction) {
        if (const auto found = junctionWaters.find(junction.cell); found != junctionWaters.end()) {
            return found->second;
        }
        if (junction.ocean) {
            junctionWaters.emplace(junction.cell, static_cast<double>(SEA_LEVEL));
            return SEA_LEVEL;
        }

        const Site next = downstream(junction);
        if (sameCell(next, junction)) {
            const double water =
                std::max(static_cast<double>(SEA_LEVEL), terminalLake(junction).surface);
            junctionWaters.emplace(junction.cell, water);
            return water;
        }

        // Portal profiles constrain water at catchment faces. A separate
        // junction level is required at the catchment site where incoming
        // guides meet the outgoing guide. Derive it from local terrain, then
        // clamp it between the outgoing portal and every incoming portal so
        // all guide halves meet continuously and remain nonincreasing.
        const double outgoingWater = portalProfile(junction).water;
        double incomingWater = std::numeric_limits<double>::max();
        constexpr std::array<int, 4> CARDINAL = {0, 2, 4, 6};
        for (const int direction : CARDINAL) {
            const Site neighbor =
                site({junction.cell.x + DX[direction], junction.cell.z + DZ[direction]});
            if (!sameCell(downstream(neighbor), junction)) continue;
            incomingWater = std::min(incomingWater, portalProfile(neighbor).water);
        }
        const double preferred = std::max(outgoingWater, junction.elevation - 0.35);
        const double water = std::isfinite(incomingWater)
                                 ? std::clamp(preferred, outgoingWater, incomingWater)
                                 : preferred;
        junctionWaters.emplace(junction.cell, water);
        return water;
    }

private:
    const CounterRng& random;
    const BasinSolver::ElevationFunction& elevation;
    const BasinSolver::RainfallFunction& rainfall;
    std::unordered_map<BasinKey, Site, BasinKeyHash> sites;
    std::unordered_map<BasinKey, BasinKey, BasinKeyHash> downstreamCells;
    std::unordered_map<BasinKey, double, BasinKeyHash> discharges;
    std::unordered_map<BasinKey, PortalProfile, BasinKeyHash> portalProfiles;
    std::unordered_map<BasinKey, double, BasinKeyHash> junctionWaters;
};

void sampleInputs(
    const BasinSolution& solution, const BasinSolver::ElevationFunction& elevation,
    const BasinSolver::RainfallFunction& rainfall,
    const BasinSolver::RockResistanceFunction& rockResistance,
    const BasinSolver::PotentialEvapotranspirationFunction& potentialEvapotranspiration,
    std::vector<float>& raw, std::vector<float>& rain, std::vector<float>& pet,
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
    std::vector<float> coarsePet(static_cast<size_t>(coarseEdge * coarseEdge));
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
            coarsePet[index] =
                static_cast<float>(potentialEvapotranspiration
                                       ? potentialEvapotranspiration(worldX, worldZ, height)
                                       : std::clamp(1450.0 - coarseRain[index] * 0.35 +
                                                        std::max(0.0, height - SEA_LEVEL) * 1.2,
                                                    220.0, 1800.0));
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
    pet.resize(RASTER_CELLS);
    resistance.resize(RASTER_CELLS);
    // Inclusive catchment-edge derivatives read one neighbor from the apron.
    // Initialize the complete raster even though routing ownership itself is
    // restricted to the sampled catchment below.
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            raw[index] = interpolate(coarseRaw, x, z);
            rain[index] = interpolate(coarseRain, x, z);
            pet[index] = interpolate(coarsePet, x, z);
            resistance[index] = interpolate(coarseResistance, x, z);
        }
    }
}

ChannelGuide makeChannelGuide(const CounterRng& random, const Site& upstream,
                              const Site& downstream, const Portal& portal,
                              const PortalProfile& profile, double upstreamWater,
                              double downstreamWater, double discharge, bool outgoing) {
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
    guide.startWater = outgoing ? upstreamWater : profile.water;
    guide.endWater = outgoing ? profile.water : downstreamWater;
    guide.terrainUpper = profile.terrainUpper;
    guide.gradient = std::max(0.0, upstream.elevation - downstream.elevation) / fullLength;
    guide.portalFlowX = (downstream.x - upstream.x) / fullLength;
    guide.portalFlowZ = (downstream.z - upstream.z) / fullLength;
    const double segmentX = guide.endX - guide.startX;
    const double segmentZ = guide.endZ - guide.startZ;
    const double segmentLength = std::hypot(segmentX, segmentZ);
    const int directionX = static_cast<int>(downstream.cell.x - upstream.cell.x);
    const int directionZ = static_cast<int>(downstream.cell.z - upstream.cell.z);
    // The complete macro edge, rather than either catchment half, owns all
    // curve randomness. Both independently built solutions therefore derive
    // one portal position, tangent, and bend.
    const int32_t segmentAddress = (directionX + 1) * 3 + directionZ + 1;
    const double amplitudeRandom = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x,
                                                    segmentAddress, upstream.cell.z, 0);
    const double secondaryPhase = random.uniform01(CHANNEL_MEANDER_STREAM, upstream.cell.x,
                                                   segmentAddress, upstream.cell.z, 1) *
                                  2.0 * std::numbers::pi;
    // A lateral cut through nearly level relief can create an artificial spill basin. Reserve
    // full guide displacement for slopes that keep the routed water surface monotonic.
    const double meanderStrength = smoothstep(0.0045, 0.012, guide.gradient);
    const double amplitude =
        std::min(144.0, segmentLength * (0.075 + amplitudeRandom * 0.055)) * meanderStrength;
    const double bendDirection = amplitudeRandom < 0.5 ? -1.0 : 1.0;
    const double canonicalNormalX = -guide.portalFlowZ;
    const double canonicalNormalZ = guide.portalFlowX;
    const GuidePoint startTangent =
        outgoing ? GuidePoint{segmentX + canonicalNormalX * amplitude * bendDirection * 2.25,
                              segmentZ + canonicalNormalZ * amplitude * bendDirection * 2.25}
                 : GuidePoint{guide.portalFlowX * segmentLength, guide.portalFlowZ * segmentLength};
    const GuidePoint endTangent =
        outgoing ? GuidePoint{guide.portalFlowX * segmentLength, guide.portalFlowZ * segmentLength}
                 : GuidePoint{segmentX - canonicalNormalX * amplitude * bendDirection * 2.25,
                              segmentZ - canonicalNormalZ * amplitude * bendDirection * 2.25};
    const auto hermite = [&](double amount) {
        const double amount2 = amount * amount;
        const double amount3 = amount2 * amount;
        const double h00 = 2.0 * amount3 - 3.0 * amount2 + 1.0;
        const double h10 = amount3 - 2.0 * amount2 + amount;
        const double h01 = -2.0 * amount3 + 3.0 * amount2;
        const double h11 = amount3 - amount2;
        return GuidePoint{
            .x = h00 * guide.startX + h10 * startTangent.x + h01 * guide.endX + h11 * endTangent.x,
            .z = h00 * guide.startZ + h10 * startTangent.z + h01 * guide.endZ + h11 * endTangent.z,
        };
    };
    for (int point = 0; point <= CHANNEL_GUIDE_CURVE_SEGMENTS; ++point) {
        const double amount = static_cast<double>(point) / CHANNEL_GUIDE_CURVE_SEGMENTS;
        guide.curve[point] = hermite(amount);
        const double secondaryEnvelope = std::pow(std::sin(std::numbers::pi * amount), 2.0);
        const double secondaryBend = amplitude * 0.24 * secondaryEnvelope *
                                     std::sin(4.0 * std::numbers::pi * amount + secondaryPhase);
        guide.curve[point].x += canonicalNormalX * secondaryBend;
        guide.curve[point].z += canonicalNormalZ * secondaryBend;
    }
    guide.curve.front() = {guide.startX, guide.startZ};
    guide.curve.back() = {guide.endX, guide.endZ};
    guide.minX = guide.maxX = guide.curve.front().x;
    guide.minZ = guide.maxZ = guide.curve.front().z;
    for (const GuidePoint& point : guide.curve) {
        guide.minX = std::min(guide.minX, point.x);
        guide.maxX = std::max(guide.maxX, point.x);
        guide.minZ = std::min(guide.minZ, point.z);
        guide.maxZ = std::max(guide.maxZ, point.z);
    }
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
                                          macro.portalProfile(center), macro.junctionWater(center),
                                          macro.junctionWater(downstream), macro.discharge(center),
                                          true));
    }
    for (int direction = 0; direction < 8; ++direction) {
        const BasinKey neighborCell{center.cell.x + DX[direction], center.cell.z + DZ[direction]};
        const Site neighbor = macro.site(neighborCell);
        const Site neighborDownstream = macro.downstream(neighbor);
        if (!sameCell(neighborDownstream, center)) continue;
        const Portal portal = sharedPortal(neighbor, center);
        guides.push_back(
            makeChannelGuide(random, neighbor, center, portal, macro.portalProfile(neighbor),
                             macro.junctionWater(neighbor), macro.junctionWater(center),
                             macro.discharge(neighbor), false));
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
                                            const std::vector<float>& raw,
                                            std::vector<float>& terminalFloor) {
    std::vector<uint8_t> terminals(RASTER_CELLS, static_cast<uint8_t>(BasinOutlet::NONE));
    terminalFloor = raw;
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (raw[index] < SEA_LEVEL &&
            inSampledCatchment(index % RASTER_EDGE, index / RASTER_EDGE)) {
            terminals[index] = static_cast<uint8_t>(BasinOutlet::OCEAN);
            terminalFloor[index] = std::max(raw[index], static_cast<float>(SEA_LEVEL));
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
                terminalFloor[portalIndex] =
                    std::max(raw[portalIndex], static_cast<float>(outgoing->portalWater));
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
                   const std::vector<float>& terminalFloor, std::vector<float>& filled,
                   std::vector<uint32_t>& floodOrder) {
    using QueueValue = std::pair<float, int32_t>;
    std::priority_queue<QueueValue, std::vector<QueueValue>, std::greater<>> queue;
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    filled = surface;
    floodOrder.assign(RASTER_CELLS, std::numeric_limits<uint32_t>::max());
    uint32_t order = 0;
    auto seed = [&](int index) {
        if (visited[index] != 0) return;
        visited[index] = 1;
        filled[index] = std::max(surface[index], terminalFloor[index]);
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

    // Route only owned samples. The initialized apron above supplies safe
    // central differences at every inclusive catchment edge.
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

struct ErosionScratch {
    std::vector<float> fluvialDelta = std::vector<float>(RASTER_CELLS);
    std::vector<float> transported = std::vector<float>(RASTER_CELLS);
    std::vector<double> talusOutgoing = std::vector<double>(RASTER_CELLS);
    std::vector<double> talusDelta = std::vector<double>(RASTER_CELLS);
    std::vector<int32_t> orderPosition = std::vector<int32_t>(RASTER_CELLS);
};

struct TalusTransfer {
    int high = -1;
    int low = -1;
    double amount = 0.0;
};

TalusTransfer proposedTalusTransfer(int first, int second, double run,
                                    const std::vector<float>& surface,
                                    const std::vector<float>& resistance) {
    const int firstX = first % RASTER_EDGE;
    const int firstZ = first / RASTER_EDGE;
    const int secondX = second % RASTER_EDGE;
    const int secondZ = second / RASTER_EDGE;
    if (lockedBoundary(firstX, firstZ) || lockedBoundary(secondX, secondZ)) return {};

    const double drop = static_cast<double>(surface[first]) - surface[second];
    const double meanResistance =
        (static_cast<double>(resistance[first]) + resistance[second]) * 0.5;
    const double criticalSlope = (5.5 + 5.0 * meanResistance) / BASIN_RASTER_SPACING;
    const double excess = std::abs(drop) - criticalSlope * run;
    if (!(excess > 0.0)) return {};
    return {
        .high = drop > 0.0 ? first : second,
        .low = drop > 0.0 ? second : first,
        .amount = std::min(0.08, excess * 0.0125),
    };
}

void erodeTerrainEpoch(const std::vector<float>& resistance, const std::vector<Receiver>& receivers,
                       const std::vector<int32_t>& upstreamOrder,
                       const std::vector<float>& accumulation, int passCount,
                       std::vector<float>& surface, std::vector<float>& sediment,
                       ErosionScratch& scratch) {
    constexpr std::array<std::pair<int, int>, 4> TALUS_PAIRS = {std::pair{1, 0}, std::pair{0, 1},
                                                                std::pair{1, 1}, std::pair{-1, 1}};
    constexpr double MAXIMUM_TALUS_OUTGOING = 0.20;

    for (int pass = 0; pass < passCount; ++pass) {
        std::ranges::fill(scratch.fluvialDelta, 0.0F);
        std::ranges::fill(scratch.transported, 0.0F);
        for (int index : upstreamOrder) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            if (lockedBoundary(x, z)) continue;
            const Receiver& receiver = receivers[index];
            if (receiver.first < 0) continue;

            double slope = 0.0;
            const auto accumulateSlope = [&](int target, float weight) {
                if (target < 0 || weight <= 0.0F) return;
                const int targetX = target % RASTER_EDGE;
                const int targetZ = target / RASTER_EDGE;
                const double run = std::hypot(targetX - x, targetZ - z) * BASIN_RASTER_SPACING;
                slope += weight * std::max(0.0, (surface[index] - surface[target]) / run);
            };
            accumulateSlope(receiver.first, receiver.firstWeight);
            accumulateSlope(receiver.second, receiver.secondWeight);

            const double capacity = std::pow(std::max(0.0F, accumulation[index]), 0.47) *
                                    std::sqrt(slope + 1.0e-5) *
                                    (0.90 / std::max(0.15F, resistance[index]));
            const double load = sediment[index] + scratch.transported[index];
            const double dischargeRelief =
                smoothstep(MIN_CHANNEL_DISCHARGE, 4'800.0, accumulation[index]);
            const double incisionLimit = 0.38 + dischargeRelief * 0.16;
            const double incisionRate = 0.045 + dischargeRelief * 0.013;
            const double incision =
                std::min(incisionLimit, std::max(0.0, capacity - load) * incisionRate);
            const double deposition = std::min(0.24, std::max(0.0, load - capacity) * 0.028);
            scratch.fluvialDelta[index] += static_cast<float>(deposition - incision);
            const float moved = static_cast<float>(std::max(0.0, load + incision - deposition));
            scratch.transported[receiver.first] += moved * receiver.firstWeight;
            if (receiver.second >= 0) {
                scratch.transported[receiver.second] += moved * receiver.secondWeight;
            }
        }

        for (int index = 0; index < RASTER_CELLS; ++index) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            if (!lockedBoundary(x, z)) surface[index] += scratch.fluvialDelta[index];
            sediment[index] = scratch.transported[index];
        }

        std::ranges::fill(scratch.talusOutgoing, 0.0);
        for (int z = 0; z < RASTER_EDGE; ++z) {
            for (int x = 0; x < RASTER_EDGE; ++x) {
                const int first = indexOf(x, z);
                for (const auto [offsetX, offsetZ] : TALUS_PAIRS) {
                    const int secondX = x + offsetX;
                    const int secondZ = z + offsetZ;
                    if (!inRaster(secondX, secondZ)) continue;
                    const int second = indexOf(secondX, secondZ);
                    const TalusTransfer transfer = proposedTalusTransfer(
                        first, second, std::hypot(offsetX, offsetZ) * BASIN_RASTER_SPACING, surface,
                        resistance);
                    if (transfer.high >= 0) {
                        scratch.talusOutgoing[transfer.high] += transfer.amount;
                    }
                }
            }
        }

        std::ranges::fill(scratch.talusDelta, 0.0);
        for (int z = 0; z < RASTER_EDGE; ++z) {
            for (int x = 0; x < RASTER_EDGE; ++x) {
                const int first = indexOf(x, z);
                for (const auto [offsetX, offsetZ] : TALUS_PAIRS) {
                    const int secondX = x + offsetX;
                    const int secondZ = z + offsetZ;
                    if (!inRaster(secondX, secondZ)) continue;
                    const int second = indexOf(secondX, secondZ);
                    const TalusTransfer transfer = proposedTalusTransfer(
                        first, second, std::hypot(offsetX, offsetZ) * BASIN_RASTER_SPACING, surface,
                        resistance);
                    if (transfer.high < 0) continue;
                    const double outgoing = scratch.talusOutgoing[transfer.high];
                    const double scale =
                        outgoing > MAXIMUM_TALUS_OUTGOING ? MAXIMUM_TALUS_OUTGOING / outgoing : 1.0;
                    const double amount = transfer.amount * scale;
                    scratch.talusDelta[transfer.high] -= amount;
                    scratch.talusDelta[transfer.low] += amount;
                }
            }
        }

        double talusBalance = 0.0;
        for (double delta : scratch.talusDelta)
            talusBalance += delta;
        if (!std::isfinite(talusBalance) || std::abs(talusBalance) > 1.0e-6) {
            throw std::runtime_error("nonconservative basin talus relaxation");
        }
        for (int index = 0; index < RASTER_CELLS; ++index) {
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            if (!lockedBoundary(x, z)) {
                surface[index] = static_cast<float>(surface[index] + scratch.talusDelta[index]);
            }
            if (!std::isfinite(surface[index]) || !std::isfinite(sediment[index]) ||
                sediment[index] < 0.0F) {
                throw std::runtime_error("invalid basin erosion field");
            }
        }
    }
}

bool validateErosionEpoch(const std::vector<float>& raw, const std::vector<float>& surface,
                          const std::vector<float>& sediment, const std::vector<float>& filled,
                          const std::vector<uint32_t>& floodOrder,
                          const std::vector<uint8_t>& terminals,
                          const std::vector<Receiver>& receivers,
                          const std::vector<int32_t>& upstreamOrder,
                          std::vector<int32_t>& orderPosition) {
    if (raw.size() != RASTER_CELLS || surface.size() != RASTER_CELLS ||
        sediment.size() != RASTER_CELLS || filled.size() != RASTER_CELLS ||
        floodOrder.size() != RASTER_CELLS || terminals.size() != RASTER_CELLS ||
        receivers.size() != RASTER_CELLS || upstreamOrder.size() != RASTER_CELLS) {
        return false;
    }

    std::ranges::fill(orderPosition, -1);
    for (size_t position = 0; position < upstreamOrder.size(); ++position) {
        const int index = upstreamOrder[position];
        if (index < 0 || index >= RASTER_CELLS || orderPosition[index] >= 0) {
            return false;
        }
        orderPosition[index] = static_cast<int32_t>(position);
    }

    for (int index = 0; index < RASTER_CELLS; ++index) {
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        if (!std::isfinite(surface[index]) || !std::isfinite(sediment[index]) ||
            sediment[index] < 0.0F || (lockedBoundary(x, z) && surface[index] != raw[index])) {
            return false;
        }

        const Receiver& receiver = receivers[index];
        const auto validTarget = [&](int target) {
            return target >= 0 && target < RASTER_CELLS && target != index &&
                   routesLower(index, target, filled, floodOrder) &&
                   orderPosition[index] < orderPosition[target];
        };
        if (receiver.first < 0) {
            if (receiver.second >= 0 || receiver.firstWeight != 0.0F ||
                receiver.secondWeight != 0.0F) {
                return false;
            }
            if (inSampledCatchment(x, z) &&
                terminals[index] == static_cast<uint8_t>(BasinOutlet::NONE)) {
                return false;
            }
            continue;
        }
        if (!validTarget(receiver.first) || receiver.firstWeight <= 0.0F ||
            receiver.firstWeight > 1.0F || receiver.secondWeight < 0.0F ||
            receiver.secondWeight > 1.0F) {
            return false;
        }
        if (receiver.second >= 0 && !validTarget(receiver.second)) {
            return false;
        }
        if (receiver.second < 0 && receiver.secondWeight != 0.0F) {
            return false;
        }
        if (std::abs(receiver.firstWeight + receiver.secondWeight - 1.0F) > 1.0e-5F) {
            return false;
        }
    }
    return true;
}

uint64_t changedReceiverCount(const std::vector<Receiver>& before,
                              const std::vector<Receiver>& after) {
    uint64_t changes = 0;
    for (size_t index = 0; index < before.size(); ++index) {
        changes += before[index].first != after[index].first ||
                           before[index].second != after[index].second ||
                           before[index].firstWeight != after[index].firstWeight ||
                           before[index].secondWeight != after[index].secondWeight
                       ? 1U
                       : 0U;
    }
    return changes;
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

bool breachShallowDepressions(std::vector<float>& surface, const std::vector<float>& filled,
                              const std::vector<Receiver>& receivers,
                              const std::vector<uint8_t>& terminals, const std::vector<float>& rain,
                              const std::vector<float>& potentialEvapotranspiration,
                              const std::vector<float>& resistance,
                              const std::vector<float>& accumulation) {
    constexpr double CELL_SQUARE_KILOMETERS =
        BASIN_RASTER_SPACING * BASIN_RASTER_SPACING / 1'000'000.0;
    std::vector<double> incomingFlux(RASTER_CELLS, 0.0);
    for (int source = 0; source < RASTER_CELLS; ++source) {
        const Receiver& receiver = receivers[source];
        for (const auto [target, weight] : {std::pair{receiver.first, receiver.firstWeight},
                                            std::pair{receiver.second, receiver.secondWeight}}) {
            if (target >= 0 && weight > 0.0F) incomingFlux[target] += accumulation[source] * weight;
        }
    }
    std::vector<int32_t> component(RASTER_CELLS, -1);
    std::vector<int> cells;
    int componentId = 0;
    bool changed = false;
    for (int start = 0; start < RASTER_CELLS; ++start) {
        if (component[start] >= 0 || surface[start] < SEA_LEVEL ||
            filled[start] - surface[start] <= 0.35F) {
            continue;
        }
        cells.clear();
        cells.push_back(start);
        component[start] = componentId;
        size_t cursor = 0;
        float maximumDepth = 0.0F;
        double fillVolume = 0.0;
        bool terminal = false;
        int deepest = start;
        while (cursor < cells.size()) {
            const int index = cells[cursor++];
            maximumDepth = std::max(maximumDepth, filled[index] - surface[index]);
            fillVolume += filled[index] - surface[index];
            terminal = terminal || terminals[index] == static_cast<uint8_t>(BasinOutlet::ENDORHEIC);
            if (surface[index] < surface[deepest] ||
                (surface[index] == surface[deepest] && index < deepest)) {
                deepest = index;
            }
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (component[neighbor] >= 0 || surface[neighbor] < SEA_LEVEL ||
                    filled[neighbor] - surface[neighbor] <= 0.35F) {
                    continue;
                }
                component[neighbor] = componentId;
                cells.push_back(neighbor);
            }
        }
        if (terminal || cells.size() > 64 || maximumDepth > 2.75F || fillVolume > 96.0) {
            ++componentId;
            continue;
        }
        double runoff = 0.0;
        double fullLoss = 0.0;
        for (const int index : cells) {
            const double localSource =
                std::max(0.0, static_cast<double>(accumulation[index]) - incomingFlux[index]);
            runoff += std::max(0.0, localSource - rain[index] * CELL_SQUARE_KILOMETERS);
            const double seepage = std::clamp(390.0 - resistance[index] * 245.0, 55.0, 340.0);
            const double netLoss =
                std::max(40.0, potentialEvapotranspiration[index] + seepage - rain[index]);
            fullLoss += netLoss * CELL_SQUARE_KILOMETERS;
        }
        for (int source = 0; source < RASTER_CELLS; ++source) {
            if (component[source] == componentId) continue;
            const Receiver& receiver = receivers[source];
            for (const auto [target, weight] :
                 {std::pair{receiver.first, receiver.firstWeight},
                  std::pair{receiver.second, receiver.secondWeight}}) {
                if (target >= 0 && weight > 0.0F && component[target] == componentId) {
                    runoff += accumulation[source] * weight;
                }
            }
        }
        // A breach is justified only when the steady water balance reaches
        // the spill and the deterministic excavation limits above are cheap.
        if (runoff <= fullLoss + 1.0e-4) {
            ++componentId;
            continue;
        }

        std::vector<int> path;
        int current = deepest;
        std::vector<uint8_t> seen(RASTER_CELLS, 0);
        seen[current] = 1;
        while (path.size() <= 20) {
            const Receiver& receiver = receivers[current];
            int next = receiver.first;
            if (receiver.second >= 0 && receiver.secondWeight > 0.0F &&
                (next < 0 || filled[receiver.second] < filled[next] ||
                 (filled[receiver.second] == filled[next] && receiver.second < next))) {
                next = receiver.second;
            }
            if (next < 0 || seen[next] != 0) break;
            path.push_back(next);
            if (component[next] != componentId) break;
            seen[next] = 1;
            current = next;
        }
        if (path.empty() || path.size() > 20 || component[path.back()] == componentId) {
            ++componentId;
            continue;
        }

        const float startElevation = surface[deepest];
        const float endElevation = std::min(surface[path.back()], startElevation - 0.05F);
        for (size_t index = 0; index < path.size(); ++index) {
            const double amount = static_cast<double>(index + 1) / path.size();
            surface[path[index]] = std::min(
                surface[path[index]],
                static_cast<float>(std::lerp(startElevation, endElevation, quinticWeight(amount))));
        }
        changed = true;
        ++componentId;
    }
    return changed;
}

void classifyLakes(BasinSolution& solution, const Site& center, const Site& downstream,
                   const TerminalLake& terminalLake, const std::vector<float>& raw,
                   const std::vector<float>& filled, const std::vector<float>& rain,
                   const std::vector<float>& potentialEvapotranspiration,
                   const std::vector<float>& resistance, const std::vector<float>& accumulation,
                   const std::vector<Receiver>& receivers, const std::vector<uint8_t>& terminals,
                   std::vector<uint8_t>& flags, std::vector<float>& lakeSurface,
                   std::vector<float>& lakeDepth) {
    constexpr double CELL_SQUARE_KILOMETERS =
        BASIN_RASTER_SPACING * BASIN_RASTER_SPACING / 1'000'000.0;
    constexpr double CELL_SQUARE_METERS = BASIN_RASTER_SPACING * BASIN_RASTER_SPACING;
    flags.assign(RASTER_CELLS, 0);
    lakeSurface.assign(RASTER_CELLS, 0.0F);
    lakeDepth.assign(RASTER_CELLS, 0.0F);
    solution.lakeEquilibria.clear();
    solution.lakeEquilibriumByCell.assign(RASTER_CELLS, NO_LAKE_EQUILIBRIUM);
    std::vector<int32_t> component(RASTER_CELLS, -1);
    std::vector<int> queue;
    struct StageSelection {
        std::vector<int> cells;
        float surface = 0.0F;
    };
    const auto selectConnectedStage =
        [&](const std::vector<int>& candidates, int deepest, double availableRunoff,
            const std::vector<float>& netLossByCell, float maximumSurface) -> StageSelection {
        StageSelection result;
        if (candidates.empty() || availableRunoff <= 0.0) return result;
        std::vector<uint8_t> eligible(RASTER_CELLS, 0);
        std::vector<float> activation(RASTER_CELLS, std::numeric_limits<float>::max());
        for (const int index : candidates)
            eligible[index] = 1;
        using FloodCandidate = std::pair<float, int32_t>;
        std::priority_queue<FloodCandidate, std::vector<FloodCandidate>, std::greater<>> flood;
        activation[deepest] = raw[deepest];
        flood.emplace(activation[deepest], deepest);
        std::vector<std::pair<float, int>> activationOrder;
        activationOrder.reserve(candidates.size());
        while (!flood.empty()) {
            const float candidateActivation = flood.top().first;
            const int index = flood.top().second;
            flood.pop();
            if (candidateActivation != activation[index]) continue;
            activationOrder.emplace_back(candidateActivation, index);
            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (eligible[neighbor] == 0) continue;
                const float neighborActivation = std::max(candidateActivation, raw[neighbor]);
                if (neighborActivation >= activation[neighbor]) continue;
                activation[neighbor] = neighborActivation;
                flood.emplace(neighborActivation, neighbor);
            }
        }
        double consumedLoss = 0.0;
        float previousStage =
            activationOrder.empty() ? maximumSurface : activationOrder.front().first;
        size_t cursor = 0;
        while (cursor < activationOrder.size()) {
            const float groupStage = activationOrder[cursor].first;
            size_t groupEnd = cursor;
            double groupLoss = 0.0;
            while (groupEnd < activationOrder.size() &&
                   std::abs(activationOrder[groupEnd].first - groupStage) <= 1.0e-4F) {
                groupLoss +=
                    netLossByCell[activationOrder[groupEnd].second] * CELL_SQUARE_KILOMETERS;
                ++groupEnd;
            }
            if (consumedLoss + groupLoss > availableRunoff) {
                const double fraction =
                    groupLoss > 0.0
                        ? std::clamp((availableRunoff - consumedLoss) / groupLoss, 0.0, 1.0)
                        : 0.0;
                result.surface = static_cast<float>(std::lerp(
                    static_cast<double>(previousStage), static_cast<double>(groupStage), fraction));
                break;
            }
            for (size_t index = cursor; index < groupEnd; ++index)
                result.cells.push_back(activationOrder[index].second);
            consumedLoss += groupLoss;
            previousStage = groupStage;
            cursor = groupEnd;
        }
        if (cursor == activationOrder.size()) result.surface = maximumSurface;
        if (!result.cells.empty()) {
            float highestFloor = raw[result.cells.front()];
            for (const int index : result.cells)
                highestFloor = std::max(highestFloor, raw[index]);
            result.surface =
                std::min(maximumSurface, std::max(result.surface, highestFloor + 0.01F));
        }
        return result;
    };
    std::vector<double> incomingFlux(RASTER_CELLS, 0.0);
    for (int source = 0; source < RASTER_CELLS; ++source) {
        const Receiver& receiver = receivers[source];
        for (const auto [target, weight] : {std::pair{receiver.first, receiver.firstWeight},
                                            std::pair{receiver.second, receiver.secondWeight}}) {
            if (target >= 0 && weight > 0.0F) incomingFlux[target] += accumulation[source] * weight;
        }
    }
    std::vector<float> netLossByCell(RASTER_CELLS, 0.0F);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const double seepage = std::clamp(390.0 - resistance[index] * 245.0, 55.0, 340.0);
        netLossByCell[index] = static_cast<float>(
            std::max(40.0, potentialEvapotranspiration[index] + seepage - rain[index]));
    }
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
        double catchmentRunoff = 0.0;
        double fullLoss = 0.0;
        int deepest = queue.front();
        for (const int index : queue) {
            // Local rainfall is already represented in the stage-dependent
            // net loss. Retain only guide or coarse drainage injection here.
            const double localSource =
                std::max(0.0, static_cast<double>(accumulation[index]) - incomingFlux[index]);
            const double localRain = rain[index] * CELL_SQUARE_KILOMETERS;
            catchmentRunoff += std::max(0.0, localSource - localRain);
            fullLoss += netLossByCell[index] * CELL_SQUARE_KILOMETERS;
            if (raw[index] < raw[deepest] || (raw[index] == raw[deepest] && index < deepest)) {
                deepest = index;
            }
        }
        for (int source = 0; source < RASTER_CELLS; ++source) {
            if (component[source] == componentId) continue;
            const Receiver& receiver = receivers[source];
            for (const auto [target, weight] :
                 {std::pair{receiver.first, receiver.firstWeight},
                  std::pair{receiver.second, receiver.secondWeight}}) {
                if (target >= 0 && weight > 0.0F && component[target] == componentId) {
                    catchmentRunoff += accumulation[source] * weight;
                }
            }
        }
        const double overflow = std::max(0.0, catchmentRunoff - fullLoss);
        const bool fillsToSpill = !containsEndorheicSink && outlet >= 0 && overflow > 1.0e-4;

        StageSelection stage;
        if (fillsToSpill) {
            stage.cells = queue;
            stage.surface = spill;
        } else {
            stage = selectConnectedStage(queue, deepest, catchmentRunoff, netLossByCell,
                                         spill - 0.125F);
        }

        const float equilibriumSurface = canonicalLakeSurface(stage.surface);
        float equilibriumMaximumDepth = 0.0F;
        double volume = 0.0;
        double selectedLoss = 0.0;
        for (const int index : stage.cells) {
            const float depth = std::max(0.0F, equilibriumSurface - raw[index]);
            equilibriumMaximumDepth = std::max(equilibriumMaximumDepth, depth);
            volume += depth * CELL_SQUARE_METERS;
            selectedLoss += netLossByCell[index];
        }
        const double meanSelectedLoss =
            stage.cells.empty() ? 0.0 : selectedLoss / static_cast<double>(stage.cells.size());
        const bool retained = stage.cells.size() >= 12 && equilibriumMaximumDepth >= 1.5F &&
                              queue.size() < RASTER_CELLS / 3 &&
                              (containsEndorheicSink || outlet >= 0);
        if (retained) {
            const uint16_t equilibriumIndex = static_cast<uint16_t>(solution.lakeEquilibria.size());
            solution.lakeEquilibria.push_back({
                .surface = equilibriumSurface,
                .spillSurface = std::max(equilibriumSurface, canonicalLakeSurface(spill)),
                .areaSquareKilometers =
                    static_cast<float>(stage.cells.size() * CELL_SQUARE_KILOMETERS),
                .volumeCubicMeters = static_cast<float>(volume),
                .runoffMmSquareKilometers = static_cast<float>(catchmentRunoff),
                .lossMm = static_cast<float>(meanSelectedLoss),
                .overflowMmSquareKilometers = static_cast<float>(overflow),
                .maximumDepth = equilibriumMaximumDepth,
                .endorheic = !fillsToSpill,
            });
            for (int index : stage.cells) {
                flags[index] |= CELL_LAKE;
                if (!fillsToSpill) flags[index] |= CELL_ENDORHEIC;
                lakeSurface[index] = equilibriumSurface;
                lakeDepth[index] = std::max(0.0F, equilibriumSurface - raw[index]);
                solution.lakeEquilibriumByCell[index] = equilibriumIndex;
            }
            if (fillsToSpill && std::ranges::find(stage.cells, outlet) != stage.cells.end()) {
                flags[outlet] |= CELL_LAKE_OUTLET;
            }
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
    const float terminalSurface = canonicalLakeSurface(terminalLake.surface);
    const double radius = terminalLake.radius;
    std::vector<int> terminalCandidates;
    int terminalDeepest = -1;
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
            terminalCandidates.push_back(index);
            if (terminalDeepest < 0 || raw[index] < raw[terminalDeepest] ||
                (raw[index] == raw[terminalDeepest] && index < terminalDeepest)) {
                terminalDeepest = index;
            }
        }
    }
    if (!terminalCandidates.empty()) {
        std::vector<uint8_t> terminalMember(RASTER_CELLS, 0);
        for (const int index : terminalCandidates)
            terminalMember[index] = 1;
        double catchmentRunoff = 0.0;
        double fullLoss = 0.0;
        for (const int index : terminalCandidates) {
            const double localSource =
                std::max(0.0, static_cast<double>(accumulation[index]) - incomingFlux[index]);
            const double localRain = rain[index] * CELL_SQUARE_KILOMETERS;
            catchmentRunoff += std::max(0.0, localSource - localRain);
            fullLoss += netLossByCell[index] * CELL_SQUARE_KILOMETERS;
        }
        for (int source = 0; source < RASTER_CELLS; ++source) {
            if (terminalMember[source] != 0) continue;
            const Receiver& receiver = receivers[source];
            for (const auto [target, weight] :
                 {std::pair{receiver.first, receiver.firstWeight},
                  std::pair{receiver.second, receiver.secondWeight}}) {
                if (target >= 0 && weight > 0.0F && terminalMember[target] != 0) {
                    catchmentRunoff += accumulation[source] * weight;
                }
            }
        }
        StageSelection stage;
        if (catchmentRunoff >= fullLoss) {
            stage.cells = terminalCandidates;
            stage.surface = terminalSurface;
        } else {
            stage = selectConnectedStage(terminalCandidates, terminalDeepest, catchmentRunoff,
                                         netLossByCell, terminalSurface);
        }
        const float equilibriumSurface = canonicalLakeSurface(stage.surface);
        double terminalVolume = 0.0;
        double selectedLoss = 0.0;
        float terminalMaximumDepth = 0.0F;
        for (const int index : stage.cells) {
            const float depth = std::max(0.0F, equilibriumSurface - raw[index]);
            terminalVolume += depth * CELL_SQUARE_METERS;
            terminalMaximumDepth = std::max(terminalMaximumDepth, depth);
            selectedLoss += netLossByCell[index];
        }
        if (stage.cells.size() < 12 || terminalMaximumDepth < 1.5F) return;
        const double terminalArea = stage.cells.size() * CELL_SQUARE_KILOMETERS;
        const double meanSelectedLoss = selectedLoss / static_cast<double>(stage.cells.size());
        const uint16_t equilibriumIndex = static_cast<uint16_t>(solution.lakeEquilibria.size());
        solution.lakeEquilibria.push_back({
            .surface = equilibriumSurface,
            .spillSurface = terminalSurface,
            .areaSquareKilometers = static_cast<float>(terminalArea),
            .volumeCubicMeters = static_cast<float>(terminalVolume),
            .runoffMmSquareKilometers = static_cast<float>(catchmentRunoff),
            .lossMm = static_cast<float>(meanSelectedLoss),
            .overflowMmSquareKilometers = 0.0F,
            .maximumDepth = terminalMaximumDepth,
            .endorheic = true,
        });
        for (const int index : stage.cells) {
            flags[index] =
                static_cast<uint8_t>((flags[index] & ~CELL_OCEAN) | CELL_LAKE | CELL_ENDORHEIC);
            lakeSurface[index] = equilibriumSurface;
            lakeDepth[index] = std::max(0.0F, equilibriumSurface - raw[index]);
            solution.lakeEquilibriumByCell[index] = equilibriumIndex;
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

void rebuildLakeShoreDistance(BasinSolution& solution) {
    const std::vector<uint8_t> originalFlags = solution.flags;
    const std::vector<float> distanceToLake = chamferDistanceToLakeState(originalFlags, true);
    const std::vector<float> distanceToDry = chamferDistanceToLakeState(originalFlags, false);
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
            solution.lakeShoreDistance[index] = signedDistance;
        }
    }
}

void buildLakeShoreDistance(BasinSolution& solution) {
    rebuildLakeShoreDistance(solution);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            const float signedDistance = solution.lakeShoreDistance[index];

            if ((solution.flags[index] & CELL_LAKE) == 0) continue;
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

void shapeLakeBathymetry(BasinSolution& solution, const std::vector<float>& resistance) {
    if (solution.lakeEquilibria.empty()) return;
    std::vector<WaterBodyId> identities(solution.lakeEquilibria.size(), NO_WATER_BODY);
    std::vector<std::vector<int>> bodyCells(solution.lakeEquilibria.size());
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if ((solution.flags[index] & CELL_LAKE) == 0) continue;
        const uint16_t equilibriumIndex = solution.lakeEquilibriumByCell[index];
        if (equilibriumIndex == NO_LAKE_EQUILIBRIUM ||
            equilibriumIndex >= solution.lakeEquilibria.size()) {
            continue;
        }
        identities[equilibriumIndex] = solution.waterBodyIds[index];
        bodyCells[equilibriumIndex].push_back(index);
    }

    constexpr double CELL_SQUARE_METERS = BASIN_RASTER_SPACING * BASIN_RASTER_SPACING;
    for (size_t bodyIndex = 0; bodyIndex < solution.lakeEquilibria.size(); ++bodyIndex) {
        const std::vector<int>& cells = bodyCells[bodyIndex];
        if (cells.empty() || identities[bodyIndex] == NO_WATER_BODY) continue;
        LakeEquilibrium& equilibrium = solution.lakeEquilibria[bodyIndex];
        const double areaSquareMeters =
            std::max(CELL_SQUARE_METERS,
                     static_cast<double>(equilibrium.areaSquareKilometers) * 1'000'000.0);
        const double characteristicRadius = std::sqrt(areaSquareMeters / std::numbers::pi);
        const double targetDepthSum =
            static_cast<double>(equilibrium.volumeCubicMeters) / CELL_SQUARE_METERS;
        const double meanDepth = targetDepthSum / static_cast<double>(cells.size());
        const double minimumDepth = std::min(0.125, std::max(0.01, meanDepth * 0.25));
        const uint64_t directionBits = mix64(identities[bodyIndex] ^ 0x4C414B4542415448ULL);
        const uint64_t phaseBits = mix64(identities[bodyIndex] ^ 0x52494D534C4F5045ULL);
        const double directionUnit = static_cast<double>(directionBits >> 11U) * 0x1.0p-53;
        const double phaseUnit = static_cast<double>(phaseBits >> 11U) * 0x1.0p-53;
        const double angle = directionUnit * 2.0 * std::numbers::pi;
        const double profileExponent = equilibrium.endorheic ? 0.86 : 0.96;

        std::vector<double> weights;
        weights.reserve(cells.size());
        double weightSum = 0.0;
        for (const int cell : cells) {
            const float inward = std::max(0.0F, solution.lakeShoreDistance[cell]);
            const int gridX = cell % RASTER_EDGE;
            const int gridZ = cell / RASTER_EDGE;
            const double worldX = solution.originX + (gridX - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution.originZ + (gridZ - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double along = (worldX * std::cos(angle) + worldZ * std::sin(angle)) /
                                     std::max(64.0, characteristicRadius * 0.8) +
                                 phaseUnit * 2.0 * std::numbers::pi;
            const double inwardProfile =
                std::pow(1.0 - std::exp(-inward / std::max(24.0, characteristicRadius * 0.38)),
                         profileExponent);
            const double resistanceProfile = std::clamp(0.82 + resistance[cell] * 0.22, 0.82, 1.12);
            const double anisotropy = 1.0 + std::sin(along) * 0.08;
            const double solvedProfile =
                equilibrium.maximumDepth > 1.0e-6F
                    ? std::clamp(static_cast<double>(solution.lakeDepth[cell]) /
                                     equilibrium.maximumDepth,
                                 0.0, 1.0)
                    : 0.0;
            const double weight = std::max(0.01, (inwardProfile * 0.72 + solvedProfile * 0.28) *
                                                     resistanceProfile * anisotropy);
            weights.push_back(weight);
            weightSum += weight;
        }

        const double distributableDepth =
            std::max(0.0, targetDepthSum - minimumDepth * static_cast<double>(cells.size()));
        const double scale = distributableDepth / std::max(1.0e-9, weightSum);
        float realizedMaximum = 0.0F;
        for (size_t cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
            const int cell = cells[cellIndex];
            const float depth = static_cast<float>(minimumDepth + weights[cellIndex] * scale);
            solution.lakeDepth[cell] = depth;
            solution.surface[cell] = solution.waterSurface[cell] - depth;
            realizedMaximum = std::max(realizedMaximum, depth);
        }
        // Area, volume, runoff, loss, overflow, and spill level remain the
        // solved water-balance authority. Only the derived realized maximum
        // changes when the volume-conserving floor profile is redistributed.
        equilibrium.maximumDepth = realizedMaximum;
    }
}

void propagateLakeShoreAuthority(BasinSolution& solution) {
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
    solution.shoreWaterBodyIds.resize(RASTER_CELLS, NO_WATER_BODY);
    solution.shoreWaterEndorheic.resize(RASTER_CELLS, 0);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (source[index] >= 0) {
            solution.shoreWaterSurface[index] = solution.waterSurface[source[index]];
            solution.shoreWaterBodyIds[index] = solution.waterBodyIds[source[index]];
            solution.shoreWaterEndorheic[index] =
                (solution.flags[source[index]] & CELL_ENDORHEIC) != 0 ? 1 : 0;
        }
    }
}

void enforceSharedBoundaryConstraints(BasinSolution& solution, const std::vector<float>& raw,
                                      const std::vector<ChannelGuide>& guides) {
    buildLakeShoreDistance(solution);
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

            const bool wasLake = (solution.flags[index] & CELL_LAKE) != 0;

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
            bool portalCorridor = false;

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
                    portalCorridor = true;
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

            // A lake may cross a catchment face anywhere except the one
            // analytically shared portal corridor. This replaces the former
            // face-wide setback with one explicit, routed opening.
            if (wasLake && !portalCorridor) continue;
            if (wasLake && sharedWeight > 0.0F) {
                solution.lakeShoreDistance[index] =
                    std::min(solution.lakeShoreDistance[index], -0.125F);
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
    return (solution.flags[index] & (CELL_WATERFALL | CELL_LAKE_CONNECTOR)) != 0 ||
           (solution.streamOrder[index] > 0 && solution.channelWidth[index] > 0.0F &&
            solution.channelDistance[index] <= solution.channelWidth[index] * 0.55F);
}

float localMaximumDischarge(const BasinSolution& solution, int center, int radius = 2) {
    const int centerX = center % RASTER_EDGE;
    const int centerZ = center / RASTER_EDGE;
    float discharge = solution.discharge[center];
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int x = centerX + dx;
            const int z = centerZ + dz;
            if (!inRaster(x, z)) continue;
            discharge = std::max(discharge, solution.discharge[indexOf(x, z)]);
        }
    }
    return discharge;
}

void buildRasterSeaBackwater(BasinSolution& solution, const std::vector<Receiver>& receivers,
                             const std::vector<int32_t>& upstreamOrder) {
    constexpr int FINE_OCEAN_APRON = 16;
    constexpr int MAXIMUM_FINE_EDGE = 128;
    constexpr size_t MAXIMUM_FINE_CELLS =
        static_cast<size_t>(MAXIMUM_FINE_EDGE * MAXIMUM_FINE_EDGE);
    // Ocean and channel columns exchange water through a shared block face.
    // Their centers can therefore be one block apart even when the ocean
    // center lies just outside the analytical channel footprint. Import that
    // face-connected sea boundary into the routed backwater solve without
    // dilating the emitted channel itself.
    constexpr double FACE_CONNECTED_OCEAN_RADIUS = 1.0;
    solution.rasterSeaBackwaterDistance.assign(RASTER_CELLS, -1.0F);
    std::vector<float> contactAlong(RASTER_CELLS, -1.0F);

    for (int source = 0; source < RASTER_CELLS; ++source) {
        if (solution.streamOrder[source] == 0 || solution.channelWidth[source] <= 0.0F ||
            solution.channelDistance[source] > 1.0e-4F ||
            solution.waterSurface[source] <= SEA_LEVEL + 1.0e-4F) {
            continue;
        }
        const int receiver = receivers[source].first;
        if (receiver < 0 || receiver == source) continue;
        const bool receivingWater = (solution.flags[receiver] & (CELL_OCEAN | CELL_LAKE)) != 0;
        if (!receivingWater && !isActiveChannelCell(solution, receiver)) continue;

        const int sourceX = source % RASTER_EDGE;
        const int sourceZ = source / RASTER_EDGE;
        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;

        const double startX = solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double startZ = solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endX = solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endZ = solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        double crossSectionWidth = solution.channelWidth[source];
        for (int direction = 0; direction < 8; ++direction) {
            const int neighborX = sourceX + DX[direction];
            const int neighborZ = sourceZ + DZ[direction];
            if (!inRaster(neighborX, neighborZ)) continue;
            const int neighbor = indexOf(neighborX, neighborZ);
            if (solution.streamOrder[neighbor] == 0 ||
                solution.channelDistance[neighbor] > 1.0e-4F) {
                continue;
            }
            crossSectionWidth =
                std::max(crossSectionWidth, static_cast<double>(solution.channelWidth[neighbor]));
        }
        const double halfWidth = std::clamp(
            std::max(crossSectionWidth, static_cast<double>(solution.channelWidth[receiver])) *
                0.55,
            2.0, 24.0);
        const int margin = static_cast<int>(std::ceil(halfWidth)) + FINE_OCEAN_APRON;
        const int64_t minimumX = static_cast<int64_t>(std::floor(std::min(startX, endX))) - margin;
        const int64_t maximumX = static_cast<int64_t>(std::ceil(std::max(startX, endX))) + margin;
        const int64_t minimumZ = static_cast<int64_t>(std::floor(std::min(startZ, endZ))) - margin;
        const int64_t maximumZ = static_cast<int64_t>(std::ceil(std::max(startZ, endZ))) + margin;
        const int width = static_cast<int>(maximumX - minimumX + 1);
        const int height = static_cast<int>(maximumZ - minimumZ + 1);
        if (width <= 0 || height <= 0 || width > MAXIMUM_FINE_EDGE || height > MAXIMUM_FINE_EDGE ||
            static_cast<size_t>(width * height) > MAXIMUM_FINE_CELLS) {
            continue;
        }

        const auto localIndex = [=](int64_t worldX, int64_t worldZ) {
            return static_cast<size_t>((worldZ - minimumZ) * width + worldX - minimumX);
        };
        std::vector<uint8_t> submerged(static_cast<size_t>(width * height), 0);
        std::vector<uint8_t> connected(static_cast<size_t>(width * height), 0);
        std::vector<std::pair<int64_t, int64_t>> queue;
        queue.reserve(static_cast<size_t>(width * height));
        for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
            for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                const double gridX =
                    (static_cast<double>(worldX) - solution.originX) / BASIN_RASTER_SPACING +
                    RASTER_APRON;
                const double gridZ =
                    (static_cast<double>(worldZ) - solution.originZ) / BASIN_RASTER_SPACING +
                    RASTER_APRON;
                if (reconstruct(solution.surface, gridX, gridZ) >= SEA_LEVEL ||
                    flagWeight(solution, gridX, gridZ, CELL_LAKE) >= 0.5) {
                    continue;
                }
                const size_t index = localIndex(worldX, worldZ);
                submerged[index] = 1;
                if (flagWeight(solution, gridX, gridZ, CELL_OCEAN) >= 0.5) {
                    connected[index] = 1;
                    queue.emplace_back(worldX, worldZ);
                }
            }
        }
        if (queue.empty()) {
            for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
                for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                    if (worldX != minimumX && worldX != maximumX && worldZ != minimumZ &&
                        worldZ != maximumZ) {
                        continue;
                    }
                    const size_t index = localIndex(worldX, worldZ);
                    if (submerged[index] == 0 || connected[index] != 0) continue;
                    connected[index] = 1;
                    queue.emplace_back(worldX, worldZ);
                }
            }
        }
        constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                                 std::pair{0, -1}, std::pair{0, 1}};
        for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const auto [worldX, worldZ] = queue[cursor];
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int64_t neighborX = worldX + offsetX;
                const int64_t neighborZ = worldZ + offsetZ;
                if (neighborX < minimumX || neighborX > maximumX || neighborZ < minimumZ ||
                    neighborZ > maximumZ) {
                    continue;
                }
                const size_t neighbor = localIndex(neighborX, neighborZ);
                if (submerged[neighbor] == 0 || connected[neighbor] != 0) continue;
                connected[neighbor] = 1;
                queue.emplace_back(neighborX, neighborZ);
            }
        }

        double earliestContact = std::numeric_limits<double>::infinity();
        for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
            for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                if (connected[localIndex(worldX, worldZ)] == 0) continue;
                double along = 0.0;
                const double distance =
                    distanceToSegment(static_cast<double>(worldX), static_cast<double>(worldZ),
                                      startX, startZ, endX, endZ, along);
                if (distance <= halfWidth + FACE_CONNECTED_OCEAN_RADIUS) {
                    earliestContact = std::min(earliestContact, along);
                }
            }
        }
        if (std::isfinite(earliestContact)) {
            contactAlong[source] = static_cast<float>(earliestContact);
        }
    }

    for (int source = 0; source < RASTER_CELLS; ++source) {
        if (contactAlong[source] < 0.0F) continue;
        const int receiver = receivers[source].first;
        if (receiver < 0 || receiver == source) continue;
        const int sourceX = source % RASTER_EDGE;
        const int sourceZ = source / RASTER_EDGE;
        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;
        const float edgeLength = static_cast<float>(
            std::hypot(receiverX - sourceX, receiverZ - sourceZ) * BASIN_RASTER_SPACING);
        const float contactDistance = std::max(0.0F, contactAlong[source] * edgeLength);
        float& sourceDistance = solution.rasterSeaBackwaterDistance[source];
        if (sourceDistance < 0.0F || contactDistance < sourceDistance) {
            sourceDistance = contactDistance;
        }
        int downstream = receiver;
        for (int steps = 0; steps < RASTER_CELLS && downstream >= 0; ++steps) {
            float& distance = solution.rasterSeaBackwaterDistance[downstream];
            if (distance == 0.0F) break;
            distance = 0.0F;
            if ((solution.flags[downstream] & CELL_OCEAN) != 0) break;
            const int next = receivers[downstream].first;
            if (next < 0 || next == downstream) break;
            downstream = next;
        }
    }
    for (auto current = upstreamOrder.rbegin(); current != upstreamOrder.rend(); ++current) {
        const int source = *current;
        const int receiver = receivers[source].first;
        if (receiver < 0 || receiver == source ||
            solution.rasterSeaBackwaterDistance[receiver] < 0.0F) {
            continue;
        }
        const int sourceX = source % RASTER_EDGE;
        const int sourceZ = source / RASTER_EDGE;
        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;
        const float candidate =
            solution.rasterSeaBackwaterDistance[receiver] +
            static_cast<float>(std::hypot(receiverX - sourceX, receiverZ - sourceZ) *
                               BASIN_RASTER_SPACING);
        const float candidateSurface = static_cast<float>(
            SEA_LEVEL + std::max(0.0, std::ceil(static_cast<double>(candidate) - 1.0e-6)) * 0.125);
        if (solution.waterSurface[source] > candidateSurface + 2.5F) continue;
        float& distance = solution.rasterSeaBackwaterDistance[source];
        if (distance < 0.0F || candidate < distance) distance = candidate;
    }
    // Settle receivers before their sources. Each accepted edit therefore
    // observes the receiver's final backwater stage and cannot introduce an
    // uphill routed edge merely because its raster index happened to sort
    // first.
    for (auto current = upstreamOrder.rbegin(); current != upstreamOrder.rend(); ++current) {
        const int index = *current;
        const float distance = solution.rasterSeaBackwaterDistance[index];
        if (distance < 0.0F || !isActiveChannelCell(solution, index) ||
            (solution.flags[index] & (CELL_OCEAN | CELL_LAKE)) != 0) {
            continue;
        }
        float backwaterSurface = static_cast<float>(
            SEA_LEVEL + std::max(0.0, std::ceil(static_cast<double>(distance) - 1.0e-6)) * 0.125);
        const int receiver = receivers[index].first;
        if (receiver >= 0 && receiver != index) {
            const float receivingSurface = (solution.flags[receiver] & CELL_OCEAN) != 0
                                               ? static_cast<float>(SEA_LEVEL)
                                               : solution.waterSurface[receiver];
            backwaterSurface = std::max(backwaterSurface, receivingSurface);
        }
        if (solution.waterSurface[index] > backwaterSurface + 2.5F) continue;
        if (backwaterSurface >= solution.waterSurface[index] - 1.0e-4F) continue;
        solution.waterSurface[index] = backwaterSurface;
        const float depth = std::max(0.125F, solution.channelDepth[index]);
        solution.surface[index] =
            std::min(solution.surface[index], solution.waterSurface[index] - depth);
    }
}

void appendChannelFalls(BasinSolution& solution, const std::vector<Receiver>& receivers) {
    // A steep channel marker describes a possible knickpoint, not a volume of
    // falling water. Promote only an actual drop between routed channel cells
    // to an analytical fall. The receiver-centered, narrow footprint is then
    // shared by exact cube emission and far terrain instead of turning the
    // complete upstream river band into a vertical water sheet.
    const size_t lakeOutletFallCount = solution.outletFalls.size();
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (solution.streamOrder[index] == 0 || solution.channelDistance[index] > 1.0e-4F) {
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
            if (target < 0 || target == index) return false;
            const int targetX = target % RASTER_EDGE;
            const int targetZ = target / RASTER_EDGE;
            const double run =
                std::hypot(targetX - sourceX, targetZ - sourceZ) * BASIN_RASTER_SPACING;
            const double drop = topSurface - receivingSurface(target);
            return target >= 0 && target != index &&
                   ((solution.flags[target] & (CELL_OCEAN | CELL_LAKE)) != 0 ||
                    isActiveChannelCell(solution, target)) &&
                   drop >= 2.5 && drop / std::max(1.0, run) > 0.20;
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
        if (lockedBoundary(sourceX, sourceZ) || lockedBoundary(receiverX, receiverZ)) continue;
        const double fallEndX =
            solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double fallEndZ =
            solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        const bool duplicatesLakeOutlet = std::any_of(
            solution.outletFalls.begin(),
            solution.outletFalls.begin() + static_cast<std::ptrdiff_t>(lakeOutletFallCount),
            [&](const OutletFall& existing) {
                return std::abs(existing.topSurface - topSurface) <= 0.25F &&
                       std::hypot(existing.endX - fallEndX, existing.endZ - fallEndZ) <=
                           BASIN_RASTER_SPACING * 1.5;
            });
        if (duplicatesLakeOutlet) continue;
        const float fallDischarge = std::max(localMaximumDischarge(solution, index),
                                             localMaximumDischarge(solution, receiver));
        const int fallOrder = channelGuideOrder(fallDischarge);
        const float dischargeWidth = static_cast<float>(
            std::clamp(3.0 + std::sqrt(fallDischarge) * 0.22 + fallOrder * 1.2, 4.0, 42.0));
        const float channelWidth = std::max(
            {solution.channelWidth[index], solution.channelWidth[receiver], dischargeWidth});
        solution.outletFalls.push_back({
            .startX = solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING,
            .startZ = solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING,
            .endX = fallEndX,
            .endZ = fallEndZ,
            .topSurface = topSurface,
            .bottomSurface = bottomSurface,
            .discharge = fallDischarge,
            // The curtain must span the complete routed channel core. A
            // narrower visual fall leaves upper and lower wet shelves
            // touching around its sides, which exposes an untagged vertical
            // water edge even though the centerline has a valid fall.
            .halfWidth = std::clamp(channelWidth * 0.60F, 2.0F, 24.0F),
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
            fall.topSurface > solution.outletFalls[static_cast<size_t>(indexed)].topSurface ||
            (std::abs(fall.topSurface -
                      solution.outletFalls[static_cast<size_t>(indexed)].topSurface) <= 1.0e-4F &&
             fall.halfWidth > solution.outletFalls[static_cast<size_t>(indexed)].halfWidth)) {
            indexed = static_cast<int32_t>(fallIndex);
        }
    }
}

double rasterCrossSectionWidth(const BasinSolution& solution, int source) {
    const int sourceX = source % RASTER_EDGE;
    const int sourceZ = source / RASTER_EDGE;
    double width = solution.channelWidth[source];
    for (int direction = 0; direction < 8; ++direction) {
        const int neighborX = sourceX + DX[direction];
        const int neighborZ = sourceZ + DZ[direction];
        if (!inRaster(neighborX, neighborZ)) continue;
        const int neighbor = indexOf(neighborX, neighborZ);
        if (solution.streamOrder[neighbor] == 0 || solution.channelDistance[neighbor] > 1.0e-4F) {
            continue;
        }
        width = std::max(width, static_cast<double>(solution.channelWidth[neighbor]));
    }
    return width;
}

double rasterWaterProfileWeight(const BasinSolution& solution, int source, int receiver,
                                double along) {
    const bool seaBackwater =
        source >= 0 && receiver >= 0 &&
        static_cast<size_t>(source) < solution.rasterSeaBackwaterDistance.size() &&
        static_cast<size_t>(receiver) < solution.rasterSeaBackwaterDistance.size() &&
        solution.rasterSeaBackwaterDistance[static_cast<size_t>(source)] >= 0.0F &&
        solution.rasterSeaBackwaterDistance[static_cast<size_t>(receiver)] >= 0.0F;
    return seaBackwater ? std::clamp(along, 0.0, 1.0) : quinticWeight(along);
}

void buildSettledRasterWater(BasinSolution& solution, const std::vector<Receiver>& receivers) {
    constexpr int LOCAL_EDGE = TRANSITION_MAX_LOCAL - TRANSITION_MIN_LOCAL + 1;
    constexpr size_t LOCAL_CELLS = static_cast<size_t>(LOCAL_EDGE) * LOCAL_EDGE;
    constexpr size_t MAXIMUM_SEGMENT_VISITS = 2'097'152;
    constexpr size_t MAXIMUM_SETTLED_CELLS = 524'288;
    constexpr int MINIMUM_STAGE_EIGHTHS = (WORLD_MIN_Y - 1) * 8;
    constexpr int MAXIMUM_STAGE_EIGHTHS = (WORLD_MAX_Y + 1) * 8;
    constexpr int STAGE_BUCKETS = MAXIMUM_STAGE_EIGHTHS - MINIMUM_STAGE_EIGHTHS + 1;
    constexpr int16_t UNSET = std::numeric_limits<int16_t>::max();

    struct SettledOwner {
        int32_t index = -1;
        WaterTransitionKind kind = WaterTransitionKind::NONE;

        bool operator==(const SettledOwner&) const = default;
    };
    const auto connectedOwner = [](SettledOwner first, SettledOwner second) {
        return first == second || (first.kind == WaterTransitionKind::RASTER_CHANNEL &&
                                   second.kind == WaterTransitionKind::RASTER_CHANNEL);
    };

    std::vector<int16_t> stages(LOCAL_CELLS, UNSET);
    std::vector<int16_t> beds(LOCAL_CELLS, UNSET);
    std::vector<int16_t> conflictBankTargets(LOCAL_CELLS, UNSET);
    std::vector<SettledOwner> owners(LOCAL_CELLS);
    std::vector<uint32_t> touched;
    touched.reserve(65'536);
    size_t segmentVisits = 0;
    const int64_t originX = static_cast<int64_t>(std::llround(solution.originX));
    const int64_t originZ = static_cast<int64_t>(std::llround(solution.originZ));
    const auto packedIndex = [](int localX, int localZ) {
        return static_cast<uint32_t>((localZ - TRANSITION_MIN_LOCAL) * LOCAL_EDGE + localX -
                                     TRANSITION_MIN_LOCAL);
    };
    const auto retainCell = [&](uint32_t packed, int16_t candidateStage, int16_t candidateBed,
                                RapidOwnerKind ownerKind, int32_t ownerIndex) {
        if (ownerKind == RapidOwnerKind::CHANNEL_GUIDE && ownerIndex >= 0 &&
            static_cast<size_t>(ownerIndex) < solution.guides.size()) {
            const ChannelGuide& guide = solution.guides[static_cast<size_t>(ownerIndex)];
            const bool submergedGuideBed = candidateBed < SEA_LEVEL * 8;
            const bool seaControlled = submergedGuideBed || guide.oceanMouth ||
                                       (guide.backwater && guide.portalWater <= SEA_LEVEL + 1.0e-4);
            if (seaControlled) {
                // A submerged guide footprint belongs to the source-filled
                // ocean boundary. Publishing a sub-sea partial state makes
                // closure replace the cell with a dry sea-level bank and
                // exposes the unrelated analytical river stage beside it.
                candidateStage = static_cast<int16_t>(SEA_LEVEL * 8);
                candidateBed = std::min<int16_t>(candidateBed, candidateStage - 1);
                ownerKind = RapidOwnerKind::NONE;
                ownerIndex = -1;
            }
        }
        const SettledOwner candidateOwner{.index = ownerIndex, .kind = ownerKind};
        int16_t& retained = stages[packed];
        if (conflictBankTargets[packed] != UNSET) {
            conflictBankTargets[packed] = std::max<int16_t>(
                conflictBankTargets[packed], static_cast<int16_t>(candidateStage + 4));
            return;
        }
        if (retained == UNSET) {
            if (touched.size() >= MAXIMUM_SETTLED_CELLS) {
                throw std::runtime_error("settled raster water exceeded its cell budget");
            }
            touched.push_back(packed);
            retained = candidateStage;
            beds[packed] = std::min<int16_t>(candidateBed, candidateStage - 1);
            owners[packed] = candidateOwner;
            return;
        }
        if (owners[packed] == candidateOwner) {
            if (candidateStage < retained) {
                retained = candidateStage;
                beds[packed] = std::min<int16_t>(candidateBed, candidateStage - 1);
            } else {
                beds[packed] = std::min(beds[packed], candidateBed);
            }
            return;
        }
        const auto representationRank = [](WaterTransitionKind kind) {
            switch (kind) {
                case WaterTransitionKind::RASTER_CHANNEL:
                    return 1;
                case WaterTransitionKind::CHANNEL_GUIDE:
                    return 2;
                case WaterTransitionKind::OUTLET_CORRIDOR:
                    return 3;
                case WaterTransitionKind::NONE:
                case WaterTransitionKind::EXPLICIT_FALL:
                case WaterTransitionKind::COUNT:
                    return 0;
            }
            return 0;
        };
        const int retainedRank = representationRank(owners[packed].kind);
        const int candidateRank = representationRank(candidateOwner.kind);
        if (retainedRank > 0 && candidateRank > 0 && retainedRank != candidateRank) {
            // Raster reaches, shared Hermite guides, and outlet corridors are
            // progressively more specific representations of one routed
            // network. Where their footprints overlap, select the more
            // specific stage instead of manufacturing a dry conflict bank at
            // a portal or confluence.
            if (candidateRank > retainedRank) {
                retained = candidateStage;
                beds[packed] = std::min<int16_t>(candidateBed, candidateStage - 1);
                owners[packed] = candidateOwner;
            } else {
                beds[packed] = std::min(beds[packed], candidateBed);
            }
            return;
        }
        if (std::abs(static_cast<int>(candidateStage) - retained) <= 1) {
            if (candidateStage < retained ||
                (candidateStage == retained &&
                 std::pair{candidateOwner.kind, candidateOwner.index} <
                     std::pair{owners[packed].kind, owners[packed].index})) {
                retained = candidateStage;
                beds[packed] = std::min<int16_t>(candidateBed, candidateStage - 1);
                owners[packed] = candidateOwner;
            }
            return;
        }
        conflictBankTargets[packed] =
            static_cast<int16_t>(std::max<int>(retained, candidateStage) + 4);
        retained = UNSET;
        beds[packed] = UNSET;
        owners[packed] = {};
    };

    for (int source = 0; source < RASTER_CELLS; ++source) {
        if (solution.streamOrder[source] == 0 || solution.channelWidth[source] <= 0.0F ||
            solution.channelDistance[source] > 1.0e-4F ||
            (solution.flags[source] & (CELL_OCEAN | CELL_LAKE)) != 0) {
            continue;
        }
        const int receiver = receivers[source].first;
        if (receiver < 0 || receiver == source) continue;
        const bool receivingWater = (solution.flags[receiver] & (CELL_OCEAN | CELL_LAKE)) != 0;
        if (!receivingWater && !isActiveChannelCell(solution, receiver)) continue;

        const int sourceX = source % RASTER_EDGE;
        const int sourceZ = source / RASTER_EDGE;
        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;
        const double startX = solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double startZ = solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endX = solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endZ = solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        const bool seaBackwater =
            static_cast<size_t>(source) < solution.rasterSeaBackwaterDistance.size() &&
            static_cast<size_t>(receiver) < solution.rasterSeaBackwaterDistance.size() &&
            solution.rasterSeaBackwaterDistance[static_cast<size_t>(source)] >= 0.0F &&
            solution.rasterSeaBackwaterDistance[static_cast<size_t>(receiver)] >= 0.0F;
        const bool coastalBackwater =
            seaBackwater && std::min(solution.waterSurface[source],
                                     solution.waterSurface[receiver]) <= SEA_LEVEL + 8.0F;
        const double halfWidth =
            std::clamp(std::max(rasterCrossSectionWidth(solution, source),
                                static_cast<double>(solution.channelWidth[receiver])) *
                           0.55,
                       2.0, 24.0) +
            (coastalBackwater ? 8.0 : 0.0);
        const int64_t minimumX =
            static_cast<int64_t>(std::floor(std::min(startX, endX) - halfWidth - 0.125));
        const int64_t maximumX =
            static_cast<int64_t>(std::ceil(std::max(startX, endX) + halfWidth + 0.125));
        const int64_t minimumZ =
            static_cast<int64_t>(std::floor(std::min(startZ, endZ) - halfWidth - 0.125));
        const int64_t maximumZ =
            static_cast<int64_t>(std::ceil(std::max(startZ, endZ) + halfWidth + 0.125));
        const double startWater = solution.waterSurface[source];
        const double endWater =
            (solution.flags[receiver] & CELL_OCEAN) != 0
                ? static_cast<double>(SEA_LEVEL)
                : std::min(startWater, static_cast<double>(solution.waterSurface[receiver]));
        for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
            const int localZ = static_cast<int>(worldZ - originZ);
            if (localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL) continue;
            for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                if (++segmentVisits > MAXIMUM_SEGMENT_VISITS) {
                    throw std::runtime_error("settled raster water exceeded its visit budget");
                }
                const int localX = static_cast<int>(worldX - originX);
                if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL) continue;
                double along = 0.0;
                if (distanceToSegment(static_cast<double>(worldX), static_cast<double>(worldZ),
                                      startX, startZ, endX, endZ, along) > halfWidth + 0.125) {
                    continue;
                }
                const double stage =
                    std::lerp(startWater, endWater,
                              rasterWaterProfileWeight(solution, source, receiver, along));
                int16_t candidate =
                    static_cast<int16_t>(std::clamp(static_cast<int>(std::lround(stage * 8.0)),
                                                    MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                const double gridX =
                    (static_cast<double>(worldX) - solution.originX) / BASIN_RASTER_SPACING +
                    RASTER_APRON;
                const double gridZ =
                    (static_cast<double>(worldZ) - solution.originZ) / BASIN_RASTER_SPACING +
                    RASTER_APRON;
                if (seaBackwater && stage < SEA_LEVEL + 2.5 - 1.0e-4 &&
                    reconstruct(solution.surface, gridX, gridZ) < SEA_LEVEL) {
                    candidate = static_cast<int16_t>(SEA_LEVEL * 8);
                }
                const int16_t candidateBed = static_cast<int16_t>(std::clamp(
                    static_cast<int>(std::floor(
                        std::min(static_cast<double>(reconstruct(solution.surface, gridX, gridZ)),
                                 static_cast<double>(candidate - 1) / 8.0) *
                        8.0)),
                    MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                retainCell(packedIndex(localX, localZ), candidate, candidateBed,
                           RapidOwnerKind::RASTER_CHANNEL, rasterEdgeIndex(source, receiver));
            }
        }
    }

    // Outlet corridors participate in the same owner-aware settled field as
    // raster reaches and macro guides. Their large drops are omitted here
    // because the explicit fall descriptor owns that transition.
    for (size_t corridorIndex = 0; corridorIndex < solution.outletCorridors.size();
         ++corridorIndex) {
        const LakeOutletCorridor& corridor = solution.outletCorridors[corridorIndex];
        for (size_t segment = 0; segment + 1 < corridor.points.size(); ++segment) {
            const double startWater = corridor.waterSurface[segment];
            const double endWater = corridor.waterSurface[segment + 1];
            if (startWater >= endWater + 2.5) continue;
            const GuidePoint& start = corridor.points[segment];
            const GuidePoint& end = corridor.points[segment + 1];
            const double halfWidth = corridor.halfWidth;
            const int64_t minimumX =
                static_cast<int64_t>(std::floor(std::min(start.x, end.x) - halfWidth - 0.125));
            const int64_t maximumX =
                static_cast<int64_t>(std::ceil(std::max(start.x, end.x) + halfWidth + 0.125));
            const int64_t minimumZ =
                static_cast<int64_t>(std::floor(std::min(start.z, end.z) - halfWidth - 0.125));
            const int64_t maximumZ =
                static_cast<int64_t>(std::ceil(std::max(start.z, end.z) + halfWidth + 0.125));
            for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
                const int localZ = static_cast<int>(worldZ - originZ);
                if (localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL) continue;
                for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                    if (++segmentVisits > MAXIMUM_SEGMENT_VISITS) {
                        throw std::runtime_error("settled raster water exceeded its visit budget");
                    }
                    const int localX = static_cast<int>(worldX - originX);
                    if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL) continue;
                    double along = 0.0;
                    if (distanceToSegment(static_cast<double>(worldX), static_cast<double>(worldZ),
                                          start.x, start.z, end.x, end.z,
                                          along) > halfWidth + 0.125) {
                        continue;
                    }
                    const int16_t candidate = static_cast<int16_t>(std::clamp(
                        static_cast<int>(std::lround(
                            std::lerp(startWater, endWater, quinticWeight(along)) * 8.0)),
                        MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                    const double gridX =
                        static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
                    const double gridZ =
                        static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
                    const int16_t candidateBed = static_cast<int16_t>(
                        std::clamp(static_cast<int>(std::floor(
                                       std::min(static_cast<double>(
                                                    reconstruct(solution.surface, gridX, gridZ)),
                                                static_cast<double>(candidate - 1) / 8.0) *
                                       8.0)),
                                   MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                    retainCell(packedIndex(localX, localZ), candidate, candidateBed,
                               RapidOwnerKind::OUTLET_CORRIDOR,
                               static_cast<int32_t>(corridorIndex));
                }
            }
        }
    }

    // Macro guides are the continuous authority between raster catchments.
    // Settle their wettable footprint into the same cardinal stage field so a
    // connected inundation pass below continues through low natural relief.
    for (size_t guideIndex = 0; guideIndex < solution.guides.size(); ++guideIndex) {
        const ChannelGuide& guide = solution.guides[guideIndex];
        if (guide.profileFall) continue;
        const int order = channelGuideOrder(guide.discharge);
        for (int segment = 0; segment < CHANNEL_GUIDE_CURVE_SEGMENTS; ++segment) {
            const GuidePoint& start = guide.curve[segment];
            const GuidePoint& end = guide.curve[segment + 1];
            const double middleAlong =
                (static_cast<double>(segment) + 0.5) / CHANNEL_GUIDE_CURVE_SEGMENTS;
            const double widthEnvelope = std::pow(std::sin(std::numbers::pi * middleAlong), 2.0);
            const double widthScale =
                1.0 + guide.widthVariation * widthEnvelope *
                          std::sin(4.0 * std::numbers::pi * middleAlong + guide.widthPhase);
            const double channelWidth = std::clamp(
                (3.0 + std::sqrt(guide.discharge) * 0.22 + order * 1.2) * widthScale, 4.0, 42.0);
            constexpr double BACKWATER_WETTING_HALO = 8.0;
            const double halfWidth =
                channelWidth * 0.55 + (guide.backwater ? BACKWATER_WETTING_HALO : 0.0);
            const int64_t minimumX =
                static_cast<int64_t>(std::floor(std::min(start.x, end.x) - halfWidth - 0.125));
            const int64_t maximumX =
                static_cast<int64_t>(std::ceil(std::max(start.x, end.x) + halfWidth + 0.125));
            const int64_t minimumZ =
                static_cast<int64_t>(std::floor(std::min(start.z, end.z) - halfWidth - 0.125));
            const int64_t maximumZ =
                static_cast<int64_t>(std::ceil(std::max(start.z, end.z) + halfWidth + 0.125));
            for (int64_t worldZ = minimumZ; worldZ <= maximumZ; ++worldZ) {
                const int localZ = static_cast<int>(worldZ - originZ);
                if (localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL) continue;
                for (int64_t worldX = minimumX; worldX <= maximumX; ++worldX) {
                    if (++segmentVisits > MAXIMUM_SEGMENT_VISITS) {
                        throw std::runtime_error("settled raster water exceeded its visit budget");
                    }
                    const int localX = static_cast<int>(worldX - originX);
                    if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL) continue;
                    double segmentAlong = 0.0;
                    if (distanceToSegment(static_cast<double>(worldX), static_cast<double>(worldZ),
                                          start.x, start.z, end.x, end.z,
                                          segmentAlong) > halfWidth + 0.125) {
                        continue;
                    }
                    const double guideAlong = (static_cast<double>(segment) + segmentAlong) /
                                              CHANNEL_GUIDE_CURVE_SEGMENTS;
                    const double guideStage = guide.backwater
                                                  ? guide.portalWater
                                                  : std::lerp(guide.startWater, guide.endWater,
                                                              quinticWeight(guideAlong));
                    const double gridX =
                        (static_cast<double>(worldX) - solution.originX) / BASIN_RASTER_SPACING +
                        RASTER_APRON;
                    const double gridZ =
                        (static_cast<double>(worldZ) - solution.originZ) / BASIN_RASTER_SPACING +
                        RASTER_APRON;
                    const double terrain = reconstruct(solution.surface, gridX, gridZ);
                    const double stage =
                        terrain < SEA_LEVEL
                            ? static_cast<double>(SEA_LEVEL)
                            : std::min(guideStage, static_cast<double>(reconstruct(
                                                       solution.waterSurface, gridX, gridZ)));
                    const int16_t candidate = static_cast<int16_t>(
                        std::clamp(static_cast<int>(std::lround(stage * 8.0)),
                                   MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                    const int16_t candidateBed = static_cast<int16_t>(std::clamp(
                        static_cast<int>(std::floor(
                            std::min(terrain, static_cast<double>(candidate - 1) / 8.0) * 8.0)),
                        MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
                    retainCell(packedIndex(localX, localZ), candidate, candidateBed,
                               RapidOwnerKind::CHANNEL_GUIDE, static_cast<int32_t>(guideIndex));
                }
            }
        }
    }

    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};

    // Transition patches are emitted from the same routed edges and guides,
    // but their block-resolution footprint can extend one cell beyond the
    // analytical wet core. Fold every ordinary rapid cell into the settled
    // stage field before enforcing the cardinal gradient. Otherwise a rapid
    // that exists only in the transition index can meet a source-filled
    // neighbor several blocks higher or lower even though both belong to the
    // same route. Explicit falls remain disconnected here because their
    // vertical curtain is the intentional sharp transition.
    for (const RapidCell& rapid : solution.rapidCells) {
        if (rapid.ownerKind == RapidOwnerKind::EXPLICIT_FALL) continue;
        const int64_t localX64 = rapid.x - originX;
        const int64_t localZ64 = rapid.z - originZ;
        if (localX64 < TRANSITION_MIN_LOCAL || localX64 > TRANSITION_MAX_LOCAL ||
            localZ64 < TRANSITION_MIN_LOCAL || localZ64 > TRANSITION_MAX_LOCAL) {
            continue;
        }
        const int localX = static_cast<int>(localX64);
        const int localZ = static_cast<int>(localZ64);
        const uint32_t packed = packedIndex(localX, localZ);
        const int visibleEighths = rapid.topY * 8 + (rapid.level == 0 ? 8 : 8 - rapid.level);
        const int16_t candidate = static_cast<int16_t>(
            std::clamp(visibleEighths, MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
        const double gridX = static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        const int16_t candidateBed = static_cast<int16_t>(std::clamp(
            static_cast<int>(std::floor(
                std::min(static_cast<double>(reconstruct(solution.surface, gridX, gridZ)),
                         static_cast<double>(candidate - 1) / 8.0) *
                8.0)),
            MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS));
        retainCell(packed, candidate, candidateBed, rapid.ownerKind, rapid.ownerIndex);
    }

    // A routed reach that opens directly into below-sea relief needs a final
    // one-eighth step into source-filled ocean. Treat the sea as a downstream
    // boundary condition at the last owned wet cell. Lateral faces are not
    // lowered here; they receive a solid bank below so a river running beside
    // the coast does not turn into a freestanding water wall.
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const double gridX = static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        double flowX = reconstruct(solution.flowX, gridX, gridZ);
        double flowZ = reconstruct(solution.flowZ, gridX, gridZ);
        const double flowLength = std::hypot(flowX, flowZ);
        if (flowLength <= 1.0e-6) continue;
        flowX /= flowLength;
        flowZ /= flowLength;
        bool opensDownstream = false;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            if (stages[neighbor] != UNSET) continue;
            const double alignment = flowX * offsetX + flowZ * offsetZ;
            const double neighborGridX =
                static_cast<double>(neighborX) / BASIN_RASTER_SPACING + RASTER_APRON;
            const double neighborGridZ =
                static_cast<double>(neighborZ) / BASIN_RASTER_SPACING + RASTER_APRON;
            const bool belowSeaRelief =
                reconstruct(solution.surface, neighborGridX, neighborGridZ) < SEA_LEVEL;
            if (alignment > 0.55 && belowSeaRelief) {
                opensDownstream = true;
                break;
            }
        }
        if (opensDownstream) {
            stages[packed] =
                std::min<int16_t>(stages[packed], static_cast<int16_t>(SEA_LEVEL * 8 + 1));
            beds[packed] = std::min<int16_t>(beds[packed], stages[packed] - 1);
        }
    }

    // Routed cells may touch but cannot occupy adjacent cells at incompatible
    // elevations, even when two overlapping segments belong to the same
    // corridor. Retain the higher water and turn the lower contact into a
    // solid retaining bank. Equal and one-eighth handoffs are already valid
    // shared transitions and remain wet.
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            if (stages[neighbor] == UNSET ||
                std::abs(static_cast<int>(stages[neighbor]) - stages[packed]) <= 1) {
                continue;
            }
            const uint32_t lower = stages[packed] < stages[neighbor] ? packed : neighbor;
            const int16_t upperStage = std::max(stages[packed], stages[neighbor]);
            int16_t& bankTarget = conflictBankTargets[lower];
            const int16_t requiredTarget = static_cast<int16_t>(upperStage + 4);
            bankTarget = bankTarget == UNSET ? requiredTarget
                                             : std::max<int16_t>(bankTarget, requiredTarget);
            stages[lower] = UNSET;
            beds[lower] = UNSET;
            owners[lower] = {};
            if (lower == packed) break;
        }
    }

    std::array<std::vector<uint32_t>, STAGE_BUCKETS> buckets;
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        buckets[static_cast<size_t>(stages[packed] - MINIMUM_STAGE_EIGHTHS)].push_back(packed);
    }
    for (int bucket = 0; bucket < STAGE_BUCKETS; ++bucket) {
        const int16_t surface = static_cast<int16_t>(bucket + MINIMUM_STAGE_EIGHTHS);
        for (size_t cursor = 0; cursor < buckets[static_cast<size_t>(bucket)].size(); ++cursor) {
            const uint32_t packed = buckets[static_cast<size_t>(bucket)][cursor];
            if (stages[packed] != surface) continue;
            const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int neighborX = localX + offsetX;
                const int neighborZ = localZ + offsetZ;
                if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                    neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                    continue;
                }
                const uint32_t neighbor = packedIndex(neighborX, neighborZ);
                if (stages[neighbor] == UNSET || owners[neighbor] != owners[packed] ||
                    stages[neighbor] <= surface + 1) {
                    continue;
                }
                stages[neighbor] = static_cast<int16_t>(surface + 1);
                beds[neighbor] = std::min<int16_t>(beds[neighbor], stages[neighbor] - 1);
                buckets[static_cast<size_t>(stages[neighbor] - MINIMUM_STAGE_EIGHTHS)].push_back(
                    neighbor);
            }
        }
    }

    // Complete every partial state through its next source plane. Narrow
    // diagonal reaches can otherwise leave level seven beside only the lower
    // pool after an unrelated upper owner has been bank-separated.
    // A level-seven terminal at the sea plane already has its lower source
    // pool. It is not the beginning of an invented uphill chain across the
    // seafloor. Normalize unsupported sea terminals before closure so
    // iteration order cannot manufacture predecessors.
    for (const uint32_t packed : touched) {
        if (stages[packed] != SEA_LEVEL * 8 + 1) continue;
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        bool hasPredecessor = false;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            hasPredecessor = hasPredecessor || (owners[neighbor] == owners[packed] &&
                                                stages[neighbor] == stages[packed] + 1);
        }
        if (hasPredecessor) continue;
        stages[packed] = static_cast<int16_t>(SEA_LEVEL * 8);
        beds[packed] = std::min<int16_t>(beds[packed], static_cast<int16_t>(SEA_LEVEL * 8 - 1));
        owners[packed] = {};
    }
    const size_t closureCandidates = touched.size();
    for (size_t touchedIndex = 0; touchedIndex < closureCandidates; ++touchedIndex) {
        uint32_t packed = touched[touchedIndex];
        if (stages[packed] == UNSET) continue;
        const int32_t floorY =
            static_cast<int32_t>(std::floor(static_cast<double>(stages[packed]) / 8.0));
        if (stages[packed] - floorY * 8 == 0) continue;
        const SettledOwner owner = owners[packed];
        int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const auto hasPredecessor = [&](uint32_t cell, int x, int z) {
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int neighborX = x + offsetX;
                const int neighborZ = z + offsetZ;
                if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                    neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                    continue;
                }
                const uint32_t neighbor = packedIndex(neighborX, neighborZ);
                if (owners[neighbor] == owners[cell] && stages[neighbor] == stages[cell] + 1) {
                    return true;
                }
            }
            return false;
        };
        if (hasPredecessor(packed, localX, localZ)) continue;

        const double gridX = static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        double flowX = reconstruct(solution.flowX, gridX, gridZ);
        double flowZ = reconstruct(solution.flowZ, gridX, gridZ);
        const double flowLength = std::hypot(flowX, flowZ);
        if (flowLength > 1.0e-6) {
            flowX /= flowLength;
            flowZ /= flowLength;
        }
        std::array<std::pair<int, int>, 4> directions = CARDINAL;
        std::ranges::sort(directions, [&](const auto first, const auto second) {
            const double firstAlignment = flowX * first.first + flowZ * first.second;
            const double secondAlignment = flowX * second.first + flowZ * second.second;
            if (std::abs(firstAlignment - secondAlignment) > 1.0e-9)
                return firstAlignment < secondAlignment;
            return first < second;
        });

        int16_t stage = stages[packed];
        int16_t bed = beds[packed];
        while ((stage - static_cast<int16_t>(std::floor(static_cast<double>(stage) / 8.0) * 8.0)) !=
               0) {
            const int16_t predecessorStage = static_cast<int16_t>(stage + 1);
            bool placed = false;
            for (const auto [offsetX, offsetZ] : directions) {
                const int candidateX = localX + offsetX;
                const int candidateZ = localZ + offsetZ;
                if (candidateX < TRANSITION_MIN_LOCAL || candidateX > TRANSITION_MAX_LOCAL ||
                    candidateZ < TRANSITION_MIN_LOCAL || candidateZ > TRANSITION_MAX_LOCAL) {
                    continue;
                }
                const uint32_t candidate = packedIndex(candidateX, candidateZ);
                if (conflictBankTargets[candidate] != UNSET ||
                    (stages[candidate] != UNSET && owners[candidate] != owner) ||
                    (stages[candidate] != UNSET && stages[candidate] < predecessorStage)) {
                    continue;
                }
                bool touchesConflict = false;
                for (const auto [neighborOffsetX, neighborOffsetZ] : CARDINAL) {
                    const int neighborX = candidateX + neighborOffsetX;
                    const int neighborZ = candidateZ + neighborOffsetZ;
                    if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                        neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                        continue;
                    }
                    const uint32_t neighbor = packedIndex(neighborX, neighborZ);
                    touchesConflict =
                        stages[neighbor] != UNSET &&
                        std::abs(static_cast<int>(stages[neighbor]) - predecessorStage) > 1;
                    if (touchesConflict) break;
                }
                if (touchesConflict) continue;
                if (stages[candidate] == UNSET) {
                    if (touched.size() >= MAXIMUM_SETTLED_CELLS) {
                        throw std::runtime_error("settled raster water exceeded its cell budget");
                    }
                    touched.push_back(candidate);
                }
                stages[candidate] = predecessorStage;
                beds[candidate] = std::min<int16_t>(bed, predecessorStage - 1);
                owners[candidate] = owner;
                packed = candidate;
                localX = candidateX;
                localZ = candidateZ;
                stage = predecessorStage;
                bed = beds[candidate];
                placed = true;
                break;
            }
            if (!placed) {
                conflictBankTargets[packed] = static_cast<int16_t>(stage + 4);
                stages[packed] = UNSET;
                beds[packed] = UNSET;
                owners[packed] = {};
                break;
            }
        }
    }

    // Steep transition patches must already be fixed points of Java-style
    // infinite-source formation. Gentle standing reaches retain their solved
    // hydrologic stage; only a rapid whose partial top has two horizontal
    // sources can trigger this local discrete correction.
    struct SteepCorridorNeighborhood {
        GuidePoint start;
        GuidePoint end;
        double reach = 0.0;
        int32_t ownerIndex = -1;
    };
    std::vector<SteepCorridorNeighborhood> steepCorridors;
    for (size_t corridorIndex = 0; corridorIndex < solution.outletCorridors.size();
         ++corridorIndex) {
        const LakeOutletCorridor& corridor = solution.outletCorridors[corridorIndex];
        for (size_t segment = 0; segment + 1 < corridor.points.size(); ++segment) {
            if (corridor.waterSurface[segment] < corridor.waterSurface[segment + 1] + 2.5) {
                continue;
            }
            steepCorridors.push_back({
                .start = corridor.points[segment],
                .end = corridor.points[segment + 1],
                .reach = static_cast<double>(corridor.halfWidth) + 20.0,
                .ownerIndex = static_cast<int32_t>(corridorIndex),
            });
        }
    }
    const auto steepTransitionCell = [&](uint32_t packed) {
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const double gridX = static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        if (reconstruct(solution.channelGradient, gridX, gridZ) >= 0.125) return true;
        const SettledOwner owner = owners[packed];
        if (owner.kind != WaterTransitionKind::OUTLET_CORRIDOR || owner.index < 0) {
            return false;
        }
        const double worldX = originX + localX;
        const double worldZ = originZ + localZ;
        for (const SteepCorridorNeighborhood& corridor : steepCorridors) {
            if (corridor.ownerIndex != owner.index) continue;
            double along = 0.0;
            if (distanceToSegment(worldX, worldZ, corridor.start.x, corridor.start.z,
                                  corridor.end.x, corridor.end.z, along) <= corridor.reach) {
                return true;
            }
        }
        return false;
    };
    std::queue<uint32_t> sourceCandidates;
    for (const uint32_t packed : touched) {
        if (stages[packed] != UNSET && stages[packed] % 8 != 0 && steepTransitionCell(packed)) {
            sourceCandidates.push(packed);
        }
    }
    std::vector<uint32_t> formedSources;
    while (!sourceCandidates.empty()) {
        const uint32_t packed = sourceCandidates.front();
        sourceCandidates.pop();
        if (stages[packed] == UNSET || stages[packed] % 8 == 0) continue;

        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int32_t topY =
            static_cast<int32_t>(std::floor(static_cast<double>(stages[packed]) / 8.0));
        const int16_t sourceStage = static_cast<int16_t>((topY + 1) * 8);
        unsigned adjacentSources = 0;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            adjacentSources += stages[neighbor] == sourceStage;
        }
        if (adjacentSources < 2) continue;

        stages[packed] = sourceStage;
        formedSources.push_back(packed);
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            if (stages[neighbor] != UNSET && stages[neighbor] % 8 != 0 &&
                steepTransitionCell(neighbor)) {
                sourceCandidates.push(neighbor);
            }
        }
    }

    // Propagate only from newly formed sources. Existing gradients outside
    // this bounded rapid neighborhood remain the hydrologic solution. The
    // frontier may lay levels one through seven over the adjacent lower
    // source pool, which gives the corrected gradient a supported terminal
    // instead of an abrupt or floating edge.
    constexpr uint8_t UNREACHED_SOURCE_DISTANCE = std::numeric_limits<uint8_t>::max();
    constexpr int16_t UNREACHED_SOURCE_PLANE = std::numeric_limits<int16_t>::min();
    std::vector<uint8_t> sourceDistance(LOCAL_CELLS, UNREACHED_SOURCE_DISTANCE);
    std::vector<int16_t> sourcePlane(LOCAL_CELLS, UNREACHED_SOURCE_PLANE);
    std::queue<std::pair<uint32_t, int16_t>> sourceFrontier;
    const auto waterTopY = [&](uint32_t packed) {
        const int16_t stage = stages[packed];
        const int32_t floorY = static_cast<int32_t>(std::floor(static_cast<double>(stage) / 8.0));
        return stage % 8 == 0 ? floorY - 1 : floorY;
    };
    for (const uint32_t packed : formedSources) {
        const int16_t plane = static_cast<int16_t>(waterTopY(packed));
        sourceDistance[packed] = 0;
        sourcePlane[packed] = plane;
        sourceFrontier.emplace(packed, plane);
    }
    while (!sourceFrontier.empty()) {
        const auto [packed, plane] = sourceFrontier.front();
        sourceFrontier.pop();
        if (sourcePlane[packed] != plane) continue;
        const uint8_t distance = sourceDistance[packed];
        if (distance >= FluidState::LEVEL_MASK) continue;
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            if (stages[neighbor] == UNSET) continue;
            if (owners[neighbor] != owners[packed]) continue;
            const int32_t neighborTopY = waterTopY(neighbor);
            const bool samePlanePartial = neighborTopY == plane && stages[neighbor] % 8 != 0;
            const bool supportedLowerSource =
                stages[neighbor] % 8 == 0 && neighborTopY + 1 == plane;
            if (!samePlanePartial && !supportedLowerSource) continue;
            const uint8_t candidateDistance = static_cast<uint8_t>(distance + 1);
            if (samePlanePartial) {
                const int32_t floorY =
                    static_cast<int32_t>(std::floor(static_cast<double>(stages[neighbor]) / 8.0));
                const uint8_t existingLevel =
                    static_cast<uint8_t>(8 - (stages[neighbor] - floorY * 8));
                if (candidateDistance > existingLevel) continue;
            }
            if (sourcePlane[neighbor] > plane ||
                (sourcePlane[neighbor] == plane && sourceDistance[neighbor] <= candidateDistance)) {
                continue;
            }
            sourceDistance[neighbor] = candidateDistance;
            sourcePlane[neighbor] = plane;
            sourceFrontier.emplace(neighbor, plane);
        }
    }
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        const uint8_t distance = sourceDistance[packed];
        const int16_t plane = sourcePlane[packed];
        if (distance == UNREACHED_SOURCE_DISTANCE || plane == UNREACHED_SOURCE_PLANE ||
            distance == 0) {
            continue;
        }
        stages[packed] = static_cast<int16_t>(plane * 8 + 8 - distance);
    }

    // A standing lake is represented by source blocks whose emitted surface
    // is the next integer plane above the hydraulic lake elevation. Seed that
    // source plane into an adjacent routed owner before the final fixed-point
    // pass. Without this boundary condition, a fractional lake elevation can
    // meet level two or lower flowing water even though Java-style spreading
    // requires a cardinal level-one predecessor at the lake edge.
    std::vector<int16_t> standingPredecessorStages(LOCAL_CELLS, UNSET);
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET || owners[packed].kind == WaterTransitionKind::NONE ||
            owners[packed].kind == WaterTransitionKind::EXPLICIT_FALL) {
            continue;
        }
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const double neighborGridX =
                static_cast<double>(neighborX) / BASIN_RASTER_SPACING + RASTER_APRON;
            const double neighborGridZ =
                static_cast<double>(neighborZ) / BASIN_RASTER_SPACING + RASTER_APRON;
            if (dominantPositiveLevel(solution, neighborGridX, neighborGridZ) <= 0.0 ||
                reconstruct(solution.lakeShoreDistance, neighborGridX, neighborGridZ) <=
                    -LAKE_SHORE_PERTURBATION_BAND) {
                continue;
            }
            const double neighborWorldX = static_cast<double>(originX + neighborX);
            const double neighborWorldZ = static_cast<double>(originZ + neighborZ);
            const LakeAuthoritySample lake =
                sampleLakeAuthority(solution, neighborWorldX, neighborWorldZ);
            if (!lake.found || lake.signedDistance <= 0.0 || !std::isfinite(lake.waterLevel)) {
                continue;
            }
            const int sourceStage = static_cast<int>(std::ceil(lake.waterLevel - 1.0e-6) * 8.0);
            if (sourceStage < MINIMUM_STAGE_EIGHTHS || sourceStage > MAXIMUM_STAGE_EIGHTHS ||
                std::abs(sourceStage - static_cast<int>(stages[packed])) > 20) {
                continue;
            }
            const int16_t predecessorStage = static_cast<int16_t>(sourceStage);
            const int16_t firstFlowStage = static_cast<int16_t>(sourceStage - 1);
            standingPredecessorStages[packed] =
                standingPredecessorStages[packed] == UNSET
                    ? predecessorStage
                    : std::max(standingPredecessorStages[packed], predecessorStage);
            if (stages[packed] < firstFlowStage) {
                stages[packed] = firstFlowStage;
                beds[packed] = std::min<int16_t>(beds[packed], firstFlowStage - 1);
            }
        }
    }

    // Raster segments are independently indexed for deterministic feature
    // ownership, but connected cells still form one routed water surface. A
    // source correction or overlapping representation can otherwise leave a
    // multi-level step beside the next cell. Compute the least one-eighth
    // majorant of the retained stages, propagating higher physical boundary
    // conditions down instead of lowering their source predecessors.
    std::array<std::vector<uint32_t>, STAGE_BUCKETS> connectedStageBuckets;
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        connectedStageBuckets[static_cast<size_t>(stages[packed] - MINIMUM_STAGE_EIGHTHS)]
            .push_back(packed);
    }
    for (int bucket = STAGE_BUCKETS; bucket-- > 0;) {
        const int16_t surface = static_cast<int16_t>(bucket + MINIMUM_STAGE_EIGHTHS);
        auto& cells = connectedStageBuckets[static_cast<size_t>(bucket)];
        for (size_t cursor = 0; cursor < cells.size(); ++cursor) {
            const uint32_t packed = cells[cursor];
            if (stages[packed] != surface) continue;
            const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int neighborX = localX + offsetX;
                const int neighborZ = localZ + offsetZ;
                if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                    neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                    continue;
                }
                const uint32_t neighbor = packedIndex(neighborX, neighborZ);
                if (stages[neighbor] == UNSET ||
                    !connectedOwner(owners[packed], owners[neighbor]) ||
                    surface <= MINIMUM_STAGE_EIGHTHS || stages[neighbor] >= surface - 1) {
                    continue;
                }
                stages[neighbor] = static_cast<int16_t>(surface - 1);
                beds[neighbor] = std::min<int16_t>(beds[neighbor], stages[neighbor] - 1);
                connectedStageBuckets[static_cast<size_t>(stages[neighbor] - MINIMUM_STAGE_EIGHTHS)]
                    .push_back(neighbor);
            }
        }
    }

    // A higher source plane can supersede the predecessor of an older lower
    // overlay while the multi-plane frontier is still advancing. Retract any
    // resulting orphan into the adjacent lower source pool. This preserves a
    // one-eighth terminal step and prevents an unsupported partial top from
    // being rewritten on activation.
    std::queue<uint32_t> orphanCandidates;
    for (const uint32_t packed : touched) {
        if (stages[packed] != UNSET && stages[packed] % 8 != 0) orphanCandidates.push(packed);
    }
    while (!orphanCandidates.empty()) {
        const uint32_t packed = orphanCandidates.front();
        orphanCandidates.pop();
        if (stages[packed] == UNSET || stages[packed] % 8 == 0) continue;
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int16_t predecessorStage = static_cast<int16_t>(stages[packed] + 1);
        const int16_t lowerSourceStage = static_cast<int16_t>(waterTopY(packed) * 8);
        bool hasPredecessor = standingPredecessorStages[packed] == predecessorStage;
        bool touchesLowerSource = false;
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            hasPredecessor = hasPredecessor || (stages[neighbor] == predecessorStage &&
                                                connectedOwner(owners[packed], owners[neighbor]));
            touchesLowerSource = touchesLowerSource || stages[neighbor] == lowerSourceStage;
        }
        if (hasPredecessor) continue;
        if (touchesLowerSource) {
            stages[packed] = lowerSourceStage;
        } else {
            int16_t& bankTarget = conflictBankTargets[packed];
            const int16_t requiredTarget = static_cast<int16_t>(stages[packed] + 4);
            bankTarget = bankTarget == UNSET ? requiredTarget
                                             : std::max<int16_t>(bankTarget, requiredTarget);
            stages[packed] = UNSET;
            beds[packed] = UNSET;
            owners[packed] = {};
        }
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            orphanCandidates.push(packedIndex(neighborX, neighborZ));
        }
    }

    // Deepen abrupt same-owner bed contacts until every cardinal floor step
    // is at most one block. This retains the deeper physical bed while
    // removing storage-cell cliffs beneath otherwise continuous water.
    std::array<std::vector<uint32_t>, STAGE_BUCKETS> bedBuckets;
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        bedBuckets[static_cast<size_t>(beds[packed] - MINIMUM_STAGE_EIGHTHS)].push_back(packed);
    }
    for (int bucket = 0; bucket < STAGE_BUCKETS; ++bucket) {
        const int16_t bed = static_cast<int16_t>(bucket + MINIMUM_STAGE_EIGHTHS);
        for (size_t cursor = 0; cursor < bedBuckets[static_cast<size_t>(bucket)].size(); ++cursor) {
            const uint32_t packed = bedBuckets[static_cast<size_t>(bucket)][cursor];
            if (beds[packed] != bed || stages[packed] == UNSET) continue;
            const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int neighborX = localX + offsetX;
                const int neighborZ = localZ + offsetZ;
                if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                    neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                    continue;
                }
                const uint32_t neighbor = packedIndex(neighborX, neighborZ);
                if (stages[neighbor] == UNSET || owners[neighbor] != owners[packed] ||
                    beds[neighbor] <= bed + 8) {
                    continue;
                }
                beds[neighbor] = static_cast<int16_t>(bed + 8);
                bedBuckets[static_cast<size_t>(beds[neighbor] - MINIMUM_STAGE_EIGHTHS)].push_back(
                    neighbor);
            }
        }
    }

    std::vector<SettledBankCell> bankCandidates;
    bankCandidates.reserve(touched.size() / 3);
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) {
            if (conflictBankTargets[packed] != UNSET) {
                bankCandidates.push_back({
                    .localX = static_cast<int16_t>(packed % LOCAL_EDGE + TRANSITION_MIN_LOCAL),
                    .localZ = static_cast<int16_t>(packed / LOCAL_EDGE + TRANSITION_MIN_LOCAL),
                    .targetEighths = conflictBankTargets[packed],
                });
            }
            continue;
        }
        const int localX = static_cast<int>(packed % LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const int localZ = static_cast<int>(packed / LOCAL_EDGE) + TRANSITION_MIN_LOCAL;
        const double gridX = static_cast<double>(localX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = static_cast<double>(localZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        double flowX = reconstruct(solution.flowX, gridX, gridZ);
        double flowZ = reconstruct(solution.flowZ, gridX, gridZ);
        const double flowLength = std::hypot(flowX, flowZ);
        if (flowLength > 1.0e-6) {
            flowX /= flowLength;
            flowZ /= flowLength;
        }
        for (const auto [offsetX, offsetZ] : CARDINAL) {
            const int neighborX = localX + offsetX;
            const int neighborZ = localZ + offsetZ;
            if (neighborX < TRANSITION_MIN_LOCAL || neighborX > TRANSITION_MAX_LOCAL ||
                neighborZ < TRANSITION_MIN_LOCAL || neighborZ > TRANSITION_MAX_LOCAL) {
                continue;
            }
            const uint32_t neighbor = packedIndex(neighborX, neighborZ);
            if (stages[neighbor] != UNSET) continue;
            const double downstreamAlignment = flowX * offsetX + flowZ * offsetZ;
            const double neighborGridX =
                static_cast<double>(neighborX) / BASIN_RASTER_SPACING + RASTER_APRON;
            const double neighborGridZ =
                static_cast<double>(neighborZ) / BASIN_RASTER_SPACING + RASTER_APRON;
            const double neighborTerrain =
                reconstruct(solution.surface, neighborGridX, neighborGridZ);
            const double neighborWater =
                reconstruct(solution.waterSurface, neighborGridX, neighborGridZ);
            const double neighborShoreWater =
                reconstruct(solution.shoreWaterSurface, neighborGridX, neighborGridZ);
            const double stage = static_cast<double>(stages[packed]) / 8.0;
            const double neighborChannelWidth =
                reconstruct(solution.channelWidth, neighborGridX, neighborGridZ);
            const bool neighborRoutedWater =
                reconstruct(solution.discharge, neighborGridX, neighborGridZ) >=
                    MIN_CHANNEL_DISCHARGE &&
                reconstructChannelDistance(solution, neighborGridX, neighborGridZ) <=
                    neighborChannelWidth * 0.55;
            const bool neighborStandingWater =
                flagWeight(solution, neighborGridX, neighborGridZ, CELL_OCEAN) >= 0.5 ||
                (reconstruct(solution.lakeDepth, neighborGridX, neighborGridZ) > 0.05 &&
                 reconstruct(solution.lakeShoreDistance, neighborGridX, neighborGridZ) > 0.0);
            const bool compatibleStandingWater =
                (neighborRoutedWater || neighborStandingWater) &&
                neighborWater > neighborTerrain + 0.05 &&
                std::abs((neighborStandingWater && neighborShoreWater > 0.0 ? neighborShoreWater
                                                                            : neighborWater) -
                         stage) <= 0.125;
            const bool opensIntoOcean = stages[packed] <= SEA_LEVEL * 8 + 1 &&
                                        neighborTerrain < SEA_LEVEL && downstreamAlignment > 0.55;
            if (compatibleStandingWater || opensIntoOcean) {
                continue;
            }
            bankCandidates.push_back({
                .localX = static_cast<int16_t>(neighborX),
                .localZ = static_cast<int16_t>(neighborZ),
                .targetEighths =
                    static_cast<int16_t>(std::clamp(static_cast<int>(stages[packed]) + 4,
                                                    MINIMUM_STAGE_EIGHTHS, MAXIMUM_STAGE_EIGHTHS)),
            });
        }
    }
    std::ranges::sort(
        bankCandidates, [](const SettledBankCell& first, const SettledBankCell& second) {
            return std::pair{first.localZ, first.localX} < std::pair{second.localZ, second.localX};
        });
    solution.settledBankCells.clear();
    solution.settledBankCells.reserve(bankCandidates.size());
    for (const SettledBankCell& candidate : bankCandidates) {
        if (!solution.settledBankCells.empty() &&
            solution.settledBankCells.back().localX == candidate.localX &&
            solution.settledBankCells.back().localZ == candidate.localZ) {
            solution.settledBankCells.back().targetEighths =
                std::max(solution.settledBankCells.back().targetEighths, candidate.targetEighths);
            continue;
        }
        solution.settledBankCells.push_back(candidate);
    }
    solution.settledBankRowOffsets.resize(TRANSITION_ROW_COUNT + 1);
    size_t bankCursor = 0;
    for (size_t row = 0; row < TRANSITION_ROW_COUNT; ++row) {
        solution.settledBankRowOffsets[row] = static_cast<uint32_t>(bankCursor);
        const int16_t localZ = static_cast<int16_t>(TRANSITION_MIN_LOCAL + static_cast<int>(row));
        while (bankCursor < solution.settledBankCells.size() &&
               solution.settledBankCells[bankCursor].localZ == localZ) {
            ++bankCursor;
        }
    }
    solution.settledBankRowOffsets.back() = static_cast<uint32_t>(bankCursor);

    std::ranges::sort(touched);
    solution.settledWaterCells.clear();
    solution.settledWaterCells.reserve(touched.size());
    for (const uint32_t packed : touched) {
        if (stages[packed] == UNSET) continue;
        const SettledOwner owner = owners[packed];
        const int32_t floorY =
            static_cast<int32_t>(std::floor(static_cast<double>(stages[packed]) / 8.0));
        const int32_t remainder = stages[packed] - floorY * 8;
        const FluidState state = remainder == 0
                                     ? FluidState::source()
                                     : FluidState::flowing(static_cast<uint8_t>(8 - remainder));
        solution.settledWaterCells.push_back({
            .localX = static_cast<int16_t>(packed % LOCAL_EDGE + TRANSITION_MIN_LOCAL),
            .localZ = static_cast<int16_t>(packed / LOCAL_EDGE + TRANSITION_MIN_LOCAL),
            .surfaceEighths = stages[packed],
            .bedEighths = beds[packed],
            .ownerIndex = owner.index,
            .fluidPacked = state.packed(),
            .ownerKind = owner.kind,
        });
    }
    solution.settledWaterRowOffsets.resize(TRANSITION_ROW_COUNT + 1);
    size_t cursor = 0;
    for (size_t row = 0; row < TRANSITION_ROW_COUNT; ++row) {
        solution.settledWaterRowOffsets[row] = static_cast<uint32_t>(cursor);
        const int16_t localZ = static_cast<int16_t>(TRANSITION_MIN_LOCAL + static_cast<int>(row));
        while (cursor < solution.settledWaterCells.size() &&
               solution.settledWaterCells[cursor].localZ == localZ) {
            ++cursor;
        }
    }
    solution.settledWaterRowOffsets.back() = static_cast<uint32_t>(cursor);
}

const SettledWaterCell* settledWaterAt(const BasinSolution& solution, int64_t worldX,
                                       int64_t worldZ) {
    const int64_t localX = worldX - static_cast<int64_t>(std::llround(solution.originX));
    const int64_t localZ = worldZ - static_cast<int64_t>(std::llround(solution.originZ));
    if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL ||
        localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL ||
        solution.settledWaterRowOffsets.size() != TRANSITION_ROW_COUNT + 1) {
        return nullptr;
    }
    const size_t row = static_cast<size_t>(localZ - TRANSITION_MIN_LOCAL);
    const auto begin = solution.settledWaterCells.begin() + solution.settledWaterRowOffsets[row];
    const auto end = solution.settledWaterCells.begin() + solution.settledWaterRowOffsets[row + 1];
    const auto found =
        std::lower_bound(begin, end, static_cast<int16_t>(localX),
                         [](const SettledWaterCell& cell, int16_t x) { return cell.localX < x; });
    return found != end && found->localX == localX ? &*found : nullptr;
}

const SettledBankCell* settledBankAt(const BasinSolution& solution, int64_t worldX,
                                     int64_t worldZ) {
    const int64_t localX = worldX - static_cast<int64_t>(std::llround(solution.originX));
    const int64_t localZ = worldZ - static_cast<int64_t>(std::llround(solution.originZ));
    if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL ||
        localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL ||
        solution.settledBankRowOffsets.size() != TRANSITION_ROW_COUNT + 1) {
        return nullptr;
    }
    const size_t row = static_cast<size_t>(localZ - TRANSITION_MIN_LOCAL);
    const auto begin = solution.settledBankCells.begin() + solution.settledBankRowOffsets[row];
    const auto end = solution.settledBankCells.begin() + solution.settledBankRowOffsets[row + 1];
    const auto found =
        std::lower_bound(begin, end, static_cast<int16_t>(localX),
                         [](const SettledBankCell& cell, int16_t x) { return cell.localX < x; });
    return found != end && found->localX == localX ? &*found : nullptr;
}

void buildRapidPatches(BasinSolution& solution, const std::vector<Receiver>& receivers) {
    solution.rapidCells.clear();
    solution.transitionDescriptors.clear();
    solution.transitionCells.clear();
    solution.transitionRowOffsets.clear();
    constexpr std::array<std::pair<int, int>, 4> CARDINAL = {std::pair{-1, 0}, std::pair{1, 0},
                                                             std::pair{0, -1}, std::pair{0, 1}};
    const auto emitFallApproach = [&](size_t fallIndex) {
        const OutletFall& fall = solution.outletFalls[fallIndex];
        const double runX = fall.endX - fall.startX;
        const double runZ = fall.endZ - fall.startZ;
        const double run = std::hypot(runX, runZ);
        if (std::abs(runX) + std::abs(runZ) < 8.0 - 1.0e-6) return;
        const double flowX = runX / run;
        const double flowZ = runZ / run;
        // The vertical curtain owns exactly the receiving anchor column. The
        // preceding eight cardinal layers remain horizontal flow, which gives
        // the lip one level-seven predecessor instead of replacing the end of
        // the gradient with a second falling column.
        const int margin = static_cast<int>(std::ceil(fall.halfWidth)) + 2;
        const int64_t minimumX =
            static_cast<int64_t>(std::floor(std::min(fall.startX, fall.endX))) - margin;
        const int64_t maximumX =
            static_cast<int64_t>(std::ceil(std::max(fall.startX, fall.endX))) + margin;
        const int64_t minimumZ =
            static_cast<int64_t>(std::floor(std::min(fall.startZ, fall.endZ))) - margin;
        const int64_t maximumZ =
            static_cast<int64_t>(std::ceil(std::max(fall.startZ, fall.endZ))) + margin;
        const int width = static_cast<int>(maximumX - minimumX + 1);
        const int height = static_cast<int>(maximumZ - minimumZ + 1);
        if (width <= 0 || height <= 0 || width > 64 || height > 64) return;
        const auto insideStrip = [&](int64_t x, int64_t z) {
            const double fromStartX = static_cast<double>(x) - fall.startX;
            const double fromStartZ = static_cast<double>(z) - fall.startZ;
            const double along = (fromStartX * flowX + fromStartZ * flowZ) / run;
            const double cross = std::abs(-fromStartX * flowZ + fromStartZ * flowX);
            return along >= -1.0e-6 && along <= 1.0 + 1.0e-6 &&
                   cross <= static_cast<double>(fall.halfWidth) + 1.0e-6;
        };
        const auto remaining = [&](int64_t x, int64_t z) {
            return (fall.endX - static_cast<double>(x)) * flowX +
                   (fall.endZ - static_cast<double>(z)) * flowZ;
        };
        const double dominantCardinalComponent = std::max(std::abs(flowX), std::abs(flowZ));
        for (int64_t z = minimumZ; z <= maximumZ; ++z) {
            for (int64_t x = minimumX; x <= maximumX; ++x) {
                if (!insideStrip(x, z)) continue;
                const double remainingDistance = remaining(x, z);
                if (remainingDistance <= 1.0e-6) continue;
                const int reverseDistance = static_cast<int>(
                    std::ceil(remainingDistance / dominantCardinalComponent - 1.0e-6));
                if (reverseDistance <= 0 || reverseDistance > 8) continue;
                solution.rapidCells.push_back({
                    .x = x,
                    .z = z,
                    .topY = static_cast<int32_t>(std::ceil(fall.topSurface)) - 1,
                    .ownerIndex = static_cast<int32_t>(fallIndex),
                    .ownerId = transitionIdentity(0x46414C4C5F454447ULL, fall.endX, fall.endZ,
                                                  fall.endX - fall.startX, fall.endZ - fall.startZ),
                    .flowX = static_cast<float>(flowX),
                    .flowZ = static_cast<float>(flowZ),
                    .channelDistance = static_cast<float>(
                        std::abs(-(static_cast<double>(x) - fall.startX) * flowZ +
                                 (static_cast<double>(z) - fall.startZ) * flowX)),
                    .channelWidth = fall.halfWidth / 0.55F,
                    .level = static_cast<uint8_t>(8 - reverseDistance),
                    .ownerKind = RapidOwnerKind::EXPLICIT_FALL,
                });
            }
        }
    };
    const size_t initialFallCount = solution.outletFalls.size();
    for (size_t fallIndex = 0; fallIndex < initialFallCount; ++fallIndex) {
        emitFallApproach(fallIndex);
    }

    struct PendingApproach {
        double crossingX = 0.0;
        double crossingZ = 0.0;
        double flowX = 1.0;
        double flowZ = 0.0;
        double routeEndX = 0.0;
        double routeEndZ = 0.0;
        double halfWidth = 0.0;
        int32_t lowerPlane = 0;
        int32_t ownerIndex = -1;
        uint64_t ownerId = 0;
        RapidOwnerKind ownerKind = RapidOwnerKind::CHANNEL_GUIDE;
        bool suppressed = false;
    };
    std::vector<PendingApproach> pendingApproaches;
    const auto queueCrossingApproach = [&](double crossingX, double crossingZ, double flowX,
                                           double flowZ, double routeEndX, double routeEndZ,
                                           double halfWidth, int32_t lowerPlane,
                                           RapidOwnerKind ownerKind, int32_t ownerIndex,
                                           uint64_t ownerId) {
        pendingApproaches.push_back({crossingX, crossingZ, flowX, flowZ, routeEndX, routeEndZ,
                                     halfWidth, lowerPlane, ownerIndex, ownerId, ownerKind, false});
    };
    const auto emitCrossingApproach = [&](const PendingApproach& approach) {
        const double crossingX = approach.crossingX;
        const double crossingZ = approach.crossingZ;
        const double flowX = approach.flowX;
        const double flowZ = approach.flowZ;
        const double halfWidth = approach.halfWidth;
        constexpr int APPROACH_LENGTH = 8;
        const int margin = static_cast<int>(std::ceil(halfWidth)) + APPROACH_LENGTH + 2;
        const int64_t minimumX = static_cast<int64_t>(std::floor(crossingX)) - margin;
        const int64_t maximumX = static_cast<int64_t>(std::ceil(crossingX)) + margin;
        const int64_t minimumZ = static_cast<int64_t>(std::floor(crossingZ)) - margin;
        const int64_t maximumZ = static_cast<int64_t>(std::ceil(crossingZ)) + margin;
        const int width = static_cast<int>(maximumX - minimumX + 1);
        const int height = static_cast<int>(maximumZ - minimumZ + 1);
        if (width <= 0 || height <= 0 || width > 96 || height > 96) return;
        std::vector<uint8_t> distance(static_cast<size_t>(width * height), 0);
        std::vector<uint8_t> candidate(static_cast<size_t>(width * height), 0);
        const auto localIndex = [=](int64_t x, int64_t z) {
            return static_cast<size_t>((z - minimumZ) * width + x - minimumX);
        };
        const auto projection = [&](int64_t x, int64_t z) {
            const double offsetX = static_cast<double>(x) - crossingX;
            const double offsetZ = static_cast<double>(z) - crossingZ;
            return std::pair{offsetX * flowX + offsetZ * flowZ,
                             std::abs(-offsetX * flowZ + offsetZ * flowX)};
        };
        for (int64_t z = minimumZ; z <= maximumZ; ++z) {
            for (int64_t x = minimumX; x <= maximumX; ++x) {
                const auto [along, cross] = projection(x, z);
                if (along < -APPROACH_LENGTH - 0.5 || along >= -1.0e-6 || cross > halfWidth) {
                    continue;
                }
                candidate[localIndex(x, z)] = 1;
            }
        }
        std::vector<std::pair<int64_t, int64_t>> queue;
        for (int64_t z = minimumZ; z <= maximumZ; ++z) {
            for (int64_t x = minimumX; x <= maximumX; ++x) {
                if (candidate[localIndex(x, z)] == 0) continue;
                const bool touchesReceiver = std::ranges::any_of(CARDINAL, [&](const auto offset) {
                    const int64_t neighborX = x + offset.first;
                    const int64_t neighborZ = z + offset.second;
                    const auto [along, cross] = projection(neighborX, neighborZ);
                    return along >= -1.0e-6 && cross <= halfWidth;
                });
                if (!touchesReceiver) continue;
                distance[localIndex(x, z)] = 1;
                queue.emplace_back(x, z);
            }
        }
        for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const auto [x, z] = queue[cursor];
            const uint8_t current = distance[localIndex(x, z)];
            if (current >= APPROACH_LENGTH) continue;
            for (const auto [offsetX, offsetZ] : CARDINAL) {
                const int64_t neighborX = x + offsetX;
                const int64_t neighborZ = z + offsetZ;
                if (neighborX < minimumX || neighborX > maximumX || neighborZ < minimumZ ||
                    neighborZ > maximumZ) {
                    continue;
                }
                const size_t neighbor = localIndex(neighborX, neighborZ);
                if (candidate[neighbor] == 0 || distance[neighbor] != 0) continue;
                distance[neighbor] = static_cast<uint8_t>(current + 1);
                queue.emplace_back(neighborX, neighborZ);
            }
        }
        for (int64_t z = minimumZ; z <= maximumZ; ++z) {
            for (int64_t x = minimumX; x <= maximumX; ++x) {
                const uint8_t receiverDistance = distance[localIndex(x, z)];
                if (receiverDistance == 0 || receiverDistance > APPROACH_LENGTH) continue;
                solution.rapidCells.push_back({
                    .x = x,
                    .z = z,
                    .topY = approach.lowerPlane,
                    .ownerIndex = approach.ownerIndex,
                    .ownerId = approach.ownerId,
                    .flowX = static_cast<float>(flowX),
                    .flowZ = static_cast<float>(flowZ),
                    .channelDistance = static_cast<float>(projection(x, z).second),
                    .channelWidth = static_cast<float>(halfWidth / 0.55),
                    .level = static_cast<uint8_t>(APPROACH_LENGTH - receiverDistance),
                    .ownerKind = approach.ownerKind,
                });
                if (receiverDistance != 1) continue;
                for (const auto [offsetX, offsetZ] : CARDINAL) {
                    const int64_t sourceX = x + offsetX;
                    const int64_t sourceZ = z + offsetZ;
                    const auto [sourceAlong, sourceCross] = projection(sourceX, sourceZ);
                    if (sourceAlong < -1.0e-6 || sourceCross > halfWidth) continue;
                    solution.rapidCells.push_back({
                        .x = sourceX,
                        .z = sourceZ,
                        .topY = approach.lowerPlane - 1,
                        .ownerIndex = approach.ownerIndex,
                        .ownerId = approach.ownerId,
                        .flowX = static_cast<float>(flowX),
                        .flowZ = static_cast<float>(flowZ),
                        .channelDistance = static_cast<float>(sourceCross),
                        .channelWidth = static_cast<float>(halfWidth / 0.55),
                        .level = 0,
                        .ownerKind = approach.ownerKind,
                    });
                }
            }
        }
    };

    // Outlet corridors are immutable routed polylines just like macro
    // guides. Every integer crossing in a gradual corridor owns its own
    // eight-state approach; drops large enough to be vertical remain covered
    // by the explicit fall pass above.
    for (size_t corridorIndex = 0; corridorIndex < solution.outletCorridors.size();
         ++corridorIndex) {
        const LakeOutletCorridor& corridor = solution.outletCorridors[corridorIndex];
        for (size_t segment = 0; segment + 1 < corridor.points.size(); ++segment) {
            const double startWater = corridor.waterSurface[segment];
            const double endWater = corridor.waterSurface[segment + 1];
            const double drop = startWater - endWater;
            if (segment + 2 == corridor.points.size() && drop < 2.5) continue;
            if (drop <= 1.0e-4 || drop >= 2.5) continue;
            const GuidePoint& start = corridor.points[segment];
            const GuidePoint& end = corridor.points[segment + 1];
            double flowX = end.x - start.x;
            double flowZ = end.z - start.z;
            const double length = std::hypot(flowX, flowZ);
            if (length <= 1.0e-6) continue;
            flowX /= length;
            flowZ /= length;
            const int firstLowerPlane = static_cast<int>(std::ceil(startWater)) - 1;
            const int lastLowerPlane = static_cast<int>(std::ceil(endWater));
            for (int lowerPlane = firstLowerPlane; lowerPlane >= lastLowerPlane; --lowerPlane) {
                const double targetWater = static_cast<double>(lowerPlane);
                if (targetWater >= startWater - 1.0e-6 || targetWater <= endWater + 1.0e-6)
                    continue;
                double low = 0.0;
                double high = 1.0;
                for (int iteration = 0; iteration < 32; ++iteration) {
                    const double middle = (low + high) * 0.5;
                    const double water = std::lerp(startWater, endWater, middle);
                    if (water > targetWater)
                        low = middle;
                    else
                        high = middle;
                }
                const double along = (low + high) * 0.5;
                queueCrossingApproach(
                    std::lerp(start.x, end.x, along), std::lerp(start.z, end.z, along), flowX,
                    flowZ, end.x, end.z, corridor.halfWidth, lowerPlane,
                    RapidOwnerKind::OUTLET_CORRIDOR, static_cast<int32_t>(corridorIndex),
                    corridorTransitionIdentity(corridor));
            }
        }
    }

    // Raster streams below the macro-guide scale still own immutable routed
    // receiver edges. Index each gradual integer crossing once so finished
    // water remains source-filled between Java-stable rapid approaches.
    for (int source = 0; source < RASTER_CELLS; ++source) {
        if (solution.streamOrder[source] == 0 || solution.channelWidth[source] <= 0.0F ||
            solution.channelDistance[source] > 1.0e-4F ||
            (solution.flags[source] & (CELL_OCEAN | CELL_LAKE)) != 0) {
            continue;
        }
        const int receiver = receivers[source].first;
        if (receiver < 0 || receiver == source) continue;
        const bool activeReceiver =
            (solution.flags[receiver] & (CELL_WATERFALL | CELL_LAKE_CONNECTOR)) != 0 ||
            (solution.streamOrder[receiver] > 0 && solution.channelWidth[receiver] > 0.0F &&
             solution.channelDistance[receiver] <= solution.channelWidth[receiver] * 0.55F);
        if ((solution.flags[receiver] & (CELL_OCEAN | CELL_LAKE)) == 0 && !activeReceiver) {
            continue;
        }
        const double startWater = solution.waterSurface[source];
        const double endWater = (solution.flags[receiver] & CELL_OCEAN) != 0
                                    ? SEA_LEVEL
                                    : solution.waterSurface[receiver];
        const double drop = startWater - endWater;
        if (drop <= 1.0e-4) continue;
        const int sourceX = source % RASTER_EDGE;
        const int sourceZ = source / RASTER_EDGE;
        const int receiverX = receiver % RASTER_EDGE;
        const int receiverZ = receiver / RASTER_EDGE;
        const double startX = solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double startZ = solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endX = solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING;
        const double endZ = solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING;
        double flowX = endX - startX;
        double flowZ = endZ - startZ;
        const double length = std::hypot(flowX, flowZ);
        if (length <= 1.0e-6) continue;
        if (drop >= 2.5 && drop / length > 0.20) continue;
        flowX /= length;
        flowZ /= length;
        const double halfWidth =
            std::clamp(std::max(rasterCrossSectionWidth(solution, source),
                                static_cast<double>(solution.channelWidth[receiver])) *
                           0.55,
                       2.0, 24.0);
        const int firstLowerPlane = static_cast<int>(std::ceil(startWater)) - 1;
        const int lastLowerPlane = static_cast<int>(std::ceil(endWater));
        for (int lowerPlane = firstLowerPlane; lowerPlane >= lastLowerPlane; --lowerPlane) {
            const double targetWater = static_cast<double>(lowerPlane);
            if (targetWater >= startWater - 1.0e-6 || targetWater <= endWater + 1.0e-6) continue;
            double low = 0.0;
            double high = 1.0;
            for (int iteration = 0; iteration < 32; ++iteration) {
                const double middle = (low + high) * 0.5;
                const double water =
                    std::lerp(startWater, endWater,
                              rasterWaterProfileWeight(solution, source, receiver, middle));
                if (water > targetWater)
                    low = middle;
                else
                    high = middle;
            }
            const double along = (low + high) * 0.5;
            queueCrossingApproach(std::lerp(startX, endX, along), std::lerp(startZ, endZ, along),
                                  flowX, flowZ, endX, endZ, halfWidth, lowerPlane,
                                  RapidOwnerKind::RASTER_CHANNEL, rasterEdgeIndex(source, receiver),
                                  solution.rasterDrainageOwnerId);
        }
    }

    // Gentle routed water remains source-filled until the canonical guide
    // descends through the next integer source plane. Each such crossing owns
    // one eight-cell rapid approach. A cardinal breadth-first distance gives
    // every flowing level an in-plane predecessor and hands level seven to
    // the lower source pool with one final eighth-block step.
    for (size_t guideIndex = 0; guideIndex < solution.guides.size(); ++guideIndex) {
        const ChannelGuide& guide = solution.guides[guideIndex];
        if (guide.backwater || guide.profileFall || guide.startWater <= guide.endWater + 1.0e-4) {
            continue;
        }
        const int firstLowerPlane = static_cast<int>(std::ceil(guide.startWater)) - 1;
        const int lastLowerPlane = static_cast<int>(std::ceil(guide.endWater));
        for (int lowerPlane = firstLowerPlane; lowerPlane >= lastLowerPlane; --lowerPlane) {
            const double targetWater = static_cast<double>(lowerPlane);
            if (targetWater >= guide.startWater - 1.0e-6 ||
                targetWater <= guide.endWater + 1.0e-6) {
                continue;
            }
            double low = 0.0;
            double high = 1.0;
            for (int iteration = 0; iteration < 32; ++iteration) {
                const double middle = (low + high) * 0.5;
                const double water =
                    std::lerp(guide.startWater, guide.endWater, quinticWeight(middle));
                if (water > targetWater) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            const double crossingAlong = (low + high) * 0.5;
            const double scaled = crossingAlong * CHANNEL_GUIDE_CURVE_SEGMENTS;
            const int segment = std::clamp(static_cast<int>(std::floor(scaled)), 0,
                                           CHANNEL_GUIDE_CURVE_SEGMENTS - 1);
            const double segmentAlong = std::clamp(scaled - segment, 0.0, 1.0);
            const GuidePoint& start = guide.curve[segment];
            const GuidePoint& end = guide.curve[segment + 1];
            const double crossingX = std::lerp(start.x, end.x, segmentAlong);
            const double crossingZ = std::lerp(start.z, end.z, segmentAlong);
            double flowX = end.x - start.x;
            double flowZ = end.z - start.z;
            const double flowLength = std::hypot(flowX, flowZ);
            if (flowLength <= 1.0e-6) continue;
            flowX /= flowLength;
            flowZ /= flowLength;

            const double widthEnvelope = std::pow(std::sin(std::numbers::pi * crossingAlong), 2.0);
            const double widthScale =
                1.0 + guide.widthVariation * widthEnvelope *
                          std::sin(4.0 * std::numbers::pi * crossingAlong + guide.widthPhase);
            const int order = channelGuideOrder(guide.discharge);
            const double channelWidth = std::clamp(
                (3.0 + std::sqrt(guide.discharge) * 0.22 + order * 1.2) * widthScale, 4.0, 42.0);
            const double halfWidth = channelWidth * 0.60;
            queueCrossingApproach(crossingX, crossingZ, flowX, flowZ, end.x, end.z, halfWidth,
                                  lowerPlane, RapidOwnerKind::CHANNEL_GUIDE,
                                  static_cast<int32_t>(guideIndex), guideTransitionIdentity(guide));
        }
    }

    // Two or more integer source-plane crossings can fall less than eight
    // cells apart on a steep profile. Their Java-style approaches would
    // overlap, so collapse each complete conflicting cluster into one tagged
    // lip from the source plane above the first crossing to the source plane
    // below the last. The cluster gets one complete predecessor chain and no
    // partial approach can overwrite another stage.
    std::ranges::sort(pendingApproaches,
                      [](const PendingApproach& first, const PendingApproach& second) {
                          if (first.ownerKind != second.ownerKind)
                              return first.ownerKind < second.ownerKind;
                          if (first.ownerId != second.ownerId)
                              return first.ownerId < second.ownerId;
                          if (first.ownerIndex != second.ownerIndex)
                              return first.ownerIndex < second.ownerIndex;
                          if (first.lowerPlane != second.lowerPlane)
                              return first.lowerPlane > second.lowerPlane;
                          if (first.crossingX != second.crossingX)
                              return first.crossingX < second.crossingX;
                          return first.crossingZ < second.crossingZ;
                      });
    pendingApproaches.erase(
        std::unique(pendingApproaches.begin(), pendingApproaches.end(),
                    [](const PendingApproach& first, const PendingApproach& second) {
                        return first.ownerKind == second.ownerKind &&
                               first.ownerId == second.ownerId &&
                               first.ownerIndex == second.ownerIndex &&
                               first.lowerPlane == second.lowerPlane;
                    }),
        pendingApproaches.end());
    const auto approachDischarge = [&](const PendingApproach& approach) {
        if (approach.ownerKind == RapidOwnerKind::OUTLET_CORRIDOR && approach.ownerIndex >= 0 &&
            static_cast<size_t>(approach.ownerIndex) < solution.outletCorridors.size()) {
            return solution.outletCorridors[static_cast<size_t>(approach.ownerIndex)].discharge;
        }
        if (approach.ownerKind == RapidOwnerKind::CHANNEL_GUIDE && approach.ownerIndex >= 0 &&
            static_cast<size_t>(approach.ownerIndex) < solution.guides.size()) {
            return static_cast<float>(
                solution.guides[static_cast<size_t>(approach.ownerIndex)].discharge);
        }
        if (approach.ownerKind == RapidOwnerKind::RASTER_CHANNEL && approach.ownerIndex >= 0 &&
            static_cast<size_t>(approach.ownerIndex / 8) < solution.discharge.size()) {
            return solution.discharge[static_cast<size_t>(approach.ownerIndex / 8)];
        }
        return 1.0F;
    };
    bool promotedFall = false;
    for (size_t first = 0; first < pendingApproaches.size();) {
        size_t last = first + 1;
        while (last < pendingApproaches.size() &&
               pendingApproaches[last].ownerKind == pendingApproaches[first].ownerKind &&
               pendingApproaches[last].ownerId == pendingApproaches[first].ownerId &&
               pendingApproaches[last].ownerIndex == pendingApproaches[first].ownerIndex) {
            ++last;
        }
        for (size_t clusterFirst = first; clusterFirst < last;) {
            size_t clusterLast = clusterFirst + 1;
            while (clusterLast < last) {
                const PendingApproach& upstream = pendingApproaches[clusterLast - 1];
                const PendingApproach& downstream = pendingApproaches[clusterLast];
                const double cardinalSeparation =
                    std::abs(downstream.crossingX - upstream.crossingX) +
                    std::abs(downstream.crossingZ - upstream.crossingZ);
                if (cardinalSeparation >= 8.0 - 1.0e-6) break;
                ++clusterLast;
            }
            if (clusterLast - clusterFirst < 2) {
                clusterFirst = clusterLast;
                continue;
            }

            PendingApproach& upstream = pendingApproaches[clusterFirst];
            PendingApproach& downstream = pendingApproaches[clusterLast - 1];
            double clusterHalfWidth = 0.0;
            for (size_t index = clusterFirst; index < clusterLast; ++index) {
                pendingApproaches[index].suppressed = true;
                clusterHalfWidth = std::max(clusterHalfWidth, pendingApproaches[index].halfWidth);
            }

            const double cardinalFlow = std::abs(upstream.flowX) + std::abs(upstream.flowZ);
            if (cardinalFlow <= 1.0e-6) {
                clusterFirst = clusterLast;
                continue;
            }
            const double promotedRun = 8.0 / cardinalFlow;
            const double startX = upstream.crossingX - upstream.flowX * promotedRun;
            const double startZ = upstream.crossingZ - upstream.flowZ * promotedRun;
            const OutletFall promoted{
                .startX = startX,
                .startZ = startZ,
                .endX = downstream.crossingX,
                .endZ = downstream.crossingZ,
                .topSurface = static_cast<float>(upstream.lowerPlane + 1),
                .bottomSurface = static_cast<float>(downstream.lowerPlane),
                .discharge = std::max(1.0F, approachDischarge(downstream)),
                .halfWidth = static_cast<float>(std::clamp(clusterHalfWidth, 2.0, 24.0)),
                .transitionPromotion = true,
            };
            const bool duplicate =
                std::ranges::any_of(solution.outletFalls, [&](const OutletFall& existing) {
                    return std::abs(existing.startX - promoted.startX) <= 1.0e-4 &&
                           std::abs(existing.startZ - promoted.startZ) <= 1.0e-4 &&
                           std::abs(existing.endX - promoted.endX) <= 1.0e-4 &&
                           std::abs(existing.endZ - promoted.endZ) <= 1.0e-4 &&
                           std::abs(existing.topSurface - promoted.topSurface) <= 1.0e-4F &&
                           std::abs(existing.bottomSurface - promoted.bottomSurface) <= 1.0e-4F;
                });
            if (!duplicate) {
                solution.outletFalls.push_back(promoted);
                promotedFall = true;
            }
            clusterFirst = clusterLast;
        }
        first = last;
    }
    if (promotedFall) {
        indexOutletFalls(solution);
        for (size_t fallIndex = initialFallCount; fallIndex < solution.outletFalls.size();
             ++fallIndex) {
            emitFallApproach(fallIndex);
        }
    }
    for (const PendingApproach& approach : pendingApproaches) {
        if (!approach.suppressed) emitCrossingApproach(approach);
    }

    struct ClosureKey {
        int64_t x = 0;
        int64_t z = 0;
        uint64_t ownerId = 0;
        RapidOwnerKind ownerKind = RapidOwnerKind::CHANNEL_GUIDE;

        bool operator==(const ClosureKey&) const = default;
    };
    struct ClosureKeyHash {
        size_t operator()(const ClosureKey& key) const noexcept {
            uint64_t value = transitionIdentityMix(static_cast<uint64_t>(key.x));
            value = transitionIdentityMix(value ^ static_cast<uint64_t>(key.z));
            value = transitionIdentityMix(value ^ key.ownerId);
            return static_cast<size_t>(
                transitionIdentityMix(value ^ static_cast<uint8_t>(key.ownerKind)));
        }
    };
    const auto visibleSurfaceEighths = [](const RapidCell& cell) {
        if (cell.level == 0) return (cell.topY + 1) * 8;
        return cell.topY * 8 + 8 - static_cast<int32_t>(cell.level);
    };
    const auto applyVisibleSurfaceEighths = [](RapidCell& cell, int32_t surfaceEighths) {
        const int32_t floorY =
            static_cast<int32_t>(std::floor(static_cast<double>(surfaceEighths) / 8.0));
        const int32_t remainder = surfaceEighths - floorY * 8;
        if (remainder == 0) {
            cell.topY = floorY - 1;
            cell.level = 0;
        } else {
            cell.topY = floorY;
            cell.level = static_cast<uint8_t>(8 - remainder);
        }
    };
    std::unordered_map<ClosureKey, RapidCell, ClosureKeyHash> closedTransitions;
    closedTransitions.reserve(solution.rapidCells.size() * 2);
    std::vector<RapidCell> explicitFalls;
    explicitFalls.reserve(solution.rapidCells.size());
    for (const RapidCell& seed : solution.rapidCells) {
        if (seed.ownerKind == RapidOwnerKind::EXPLICIT_FALL) {
            explicitFalls.push_back(seed);
            continue;
        }
        const int32_t seedSurface = visibleSurfaceEighths(seed);
        const int32_t remainder =
            seedSurface -
            static_cast<int32_t>(std::floor(static_cast<double>(seedSurface) / 8.0)) * 8;
        const int maximumDistance = remainder == 0 ? 7 : 8 - remainder;
        for (int offsetZ = -maximumDistance; offsetZ <= maximumDistance; ++offsetZ) {
            const int remaining = maximumDistance - std::abs(offsetZ);
            for (int offsetX = -remaining; offsetX <= remaining; ++offsetX) {
                const int distance = std::abs(offsetX) + std::abs(offsetZ);
                RapidCell candidate = seed;
                candidate.x += offsetX;
                candidate.z += offsetZ;
                applyVisibleSurfaceEighths(candidate, seedSurface + distance);
                candidate.channelWidth += static_cast<float>(distance * 2.0 / 0.55);
                const ClosureKey key{
                    .x = candidate.x,
                    .z = candidate.z,
                    .ownerId = candidate.ownerId,
                    .ownerKind = candidate.ownerKind,
                };
                auto [entry, inserted] = closedTransitions.try_emplace(key, candidate);
                if (inserted) continue;
                const int32_t existingSurface = visibleSurfaceEighths(entry->second);
                const int32_t candidateSurface = visibleSurfaceEighths(candidate);
                if (candidateSurface < existingSurface ||
                    (candidateSurface == existingSurface &&
                     candidate.ownerIndex < entry->second.ownerIndex)) {
                    entry->second = candidate;
                }
            }
        }
    }
    solution.rapidCells = std::move(explicitFalls);
    solution.rapidCells.reserve(solution.rapidCells.size() + closedTransitions.size());
    for (const auto& [key, cell] : closedTransitions) {
        static_cast<void>(key);
        solution.rapidCells.push_back(cell);
    }

    std::ranges::sort(solution.rapidCells, [](const RapidCell& first, const RapidCell& second) {
        if (first.x != second.x) return first.x < second.x;
        if (first.z != second.z) return first.z < second.z;
        if (first.ownerKind != second.ownerKind) return first.ownerKind < second.ownerKind;
        if (first.ownerId != second.ownerId) return first.ownerId < second.ownerId;
        if (first.ownerIndex != second.ownerIndex) return first.ownerIndex < second.ownerIndex;
        if (first.topY != second.topY) return first.topY > second.topY;
        return first.level < second.level;
    });
    solution.rapidCells.erase(std::unique(solution.rapidCells.begin(), solution.rapidCells.end(),
                                          [](const RapidCell& first, const RapidCell& second) {
                                              return first.x == second.x && first.z == second.z &&
                                                     first.ownerKind == second.ownerKind &&
                                                     first.ownerId == second.ownerId &&
                                                     first.ownerIndex == second.ownerIndex &&
                                                     first.topY == second.topY;
                                          }),
                              solution.rapidCells.end());

    using DescriptorKey = std::tuple<RapidOwnerKind, uint64_t, int32_t, int32_t>;
    std::map<DescriptorKey, uint16_t> descriptorIndices;
    solution.transitionCells.reserve(static_cast<size_t>(std::count_if(
        solution.rapidCells.begin(), solution.rapidCells.end(),
        [](const RapidCell& rapid) { return rapid.ownerKind == RapidOwnerKind::EXPLICIT_FALL; })));
    const int64_t originX = static_cast<int64_t>(std::llround(solution.originX));
    const int64_t originZ = static_cast<int64_t>(std::llround(solution.originZ));
    for (const RapidCell& rapid : solution.rapidCells) {
        if (rapid.ownerKind != RapidOwnerKind::EXPLICIT_FALL) continue;
        const int64_t localX = rapid.x - originX;
        const int64_t localZ = rapid.z - originZ;
        if (localX < TRANSITION_MIN_LOCAL || localX > TRANSITION_MAX_LOCAL ||
            localZ < TRANSITION_MIN_LOCAL || localZ > TRANSITION_MAX_LOCAL) {
            throw std::runtime_error("generated water transition exceeded its basin index");
        }

        const DescriptorKey key{rapid.ownerKind, rapid.ownerId, rapid.ownerIndex, rapid.topY};
        auto descriptor = descriptorIndices.find(key);
        if (descriptor == descriptorIndices.end()) {
            if (solution.transitionDescriptors.size() >
                static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
                throw std::runtime_error("generated water transition descriptor overflow");
            }
            const uint16_t descriptorIndex =
                static_cast<uint16_t>(solution.transitionDescriptors.size());
            solution.transitionDescriptors.push_back({
                .ownerId = rapid.ownerId,
                .ownerIndex = rapid.ownerIndex,
                .topY = rapid.topY,
                .flowX = rapid.flowX,
                .flowZ = rapid.flowZ,
                .channelWidth = rapid.channelWidth,
                .ownerKind = rapid.ownerKind,
            });
            descriptor = descriptorIndices.emplace(key, descriptorIndex).first;
        }
        solution.transitionCells.push_back({
            .localX = static_cast<int16_t>(localX),
            .localZ = static_cast<int16_t>(localZ),
            .descriptorIndex = descriptor->second,
            .level = rapid.level,
            .distanceQuarterBlocks = static_cast<uint8_t>(std::clamp(
                std::lround(static_cast<double>(rapid.channelDistance) * 4.0), 0L, 255L)),
        });
    }
    std::ranges::sort(solution.transitionCells,
                      [](const TransitionCell& first, const TransitionCell& second) {
                          if (first.localZ != second.localZ) return first.localZ < second.localZ;
                          if (first.localX != second.localX) return first.localX < second.localX;
                          return first.descriptorIndex < second.descriptorIndex;
                      });
    solution.transitionRowOffsets.resize(TRANSITION_ROW_COUNT + 1);
    size_t cursor = 0;
    for (size_t row = 0; row < TRANSITION_ROW_COUNT; ++row) {
        solution.transitionRowOffsets[row] = static_cast<uint32_t>(cursor);
        const int16_t localZ = static_cast<int16_t>(TRANSITION_MIN_LOCAL + static_cast<int>(row));
        while (cursor < solution.transitionCells.size() &&
               solution.transitionCells[cursor].localZ == localZ) {
            ++cursor;
        }
    }
    solution.transitionRowOffsets.back() = static_cast<uint32_t>(cursor);

    // Expanded cells remain available until the settled cardinal stage field
    // folds in every ordinary transition. Cached solutions discard them after
    // that construction pass retains the compact descriptors and row offsets.
}

bool naturalOutletReceiver(const BasinSolution& solution, const std::vector<uint8_t>& terminals,
                           int index) {
    if ((solution.flags[index] & (CELL_OCEAN | CELL_LAKE | CELL_WATERFALL)) != 0) return true;
    if (solution.streamOrder[index] > 0 && solution.channelWidth[index] > 0.0F &&
        solution.channelDistance[index] <= solution.channelWidth[index] * 0.55F) {
        return true;
    }
    // outletTypes is propagated upstream, so it cannot identify a physical
    // receiver. Only a literal raster terminal may end a dry corridor.
    const BasinOutlet terminal = static_cast<BasinOutlet>(terminals[index]);
    return terminal == BasinOutlet::OCEAN || terminal == BasinOutlet::SHARED_PORTAL;
}

float receivingWaterSurface(const BasinSolution& solution, int index) {
    if ((solution.flags[index] & CELL_OCEAN) != 0 || solution.surface[index] < SEA_LEVEL) {
        return SEA_LEVEL;
    }
    return solution.waterSurface[index];
}

struct LakeOutletRoute {
    int outlet = -1;
    std::vector<int> path;
    std::vector<int> receivingPath;
};

std::vector<int> traceReceivingRoute(const BasinSolution& solution,
                                     const std::vector<Receiver>& receivers,
                                     const std::vector<uint8_t>& terminals, int start) {
    std::vector<int> result;
    if (start < 0 || start >= RASTER_CELLS) return result;
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    int current = start;
    while (current >= 0 && current < RASTER_CELLS && visited[current] == 0 &&
           result.size() <= static_cast<size_t>(MAX_OUTLET_CORRIDOR_CELLS)) {
        visited[current] = 1;
        result.push_back(current);
        const BasinOutlet terminal = static_cast<BasinOutlet>(terminals[current]);
        if ((result.size() > 1 && (solution.flags[current] & (CELL_OCEAN | CELL_LAKE)) != 0) ||
            terminal == BasinOutlet::OCEAN || terminal == BasinOutlet::SHARED_PORTAL) {
            break;
        }
        const Receiver& receiver = receivers[current];
        int next = receiver.first;
        if (next < 0 || receiver.firstWeight <= 0.0F) {
            next = receiver.secondWeight > 0.0F ? receiver.second : -1;
        }
        current = next;
    }
    return result;
}

LakeOutletRoute
findLakeOutletRoute(const BasinSolution& solution, const std::vector<Receiver>& receivers,
                    const std::vector<float>& filled, const std::vector<uint32_t>& floodOrder,
                    const std::vector<uint8_t>& terminals, const std::vector<int>& lakeCells) {
    std::vector<uint8_t> lakeMember(RASTER_CELLS, 0);
    for (const int index : lakeCells)
        lakeMember[index] = 1;

    using SearchNode = std::tuple<int, float, uint32_t, int32_t, int32_t>;
    std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<>> pending;
    std::vector<int32_t> predecessor(RASTER_CELLS, -2);
    std::vector<int32_t> originatingOutlet(RASTER_CELLS, -1);
    for (const int outlet : lakeCells) {
        const Receiver& receiver = receivers[outlet];
        for (const auto [target, weight] : {std::pair{receiver.first, receiver.firstWeight},
                                            std::pair{receiver.second, receiver.secondWeight}}) {
            if (target < 0 || weight <= 0.0F || lakeMember[target] != 0) continue;
            pending.emplace(0, filled[target], floodOrder[target], target, outlet);
        }
    }

    std::vector<uint8_t> settled(RASTER_CELLS, 0);
    int receiving = -1;
    while (!pending.empty()) {
        const auto [steps, ignoredFill, ignoredOrder, index, outlet] = pending.top();
        static_cast<void>(ignoredFill);
        static_cast<void>(ignoredOrder);
        pending.pop();
        if (settled[index] != 0 || steps > MAX_OUTLET_CORRIDOR_CELLS) continue;
        settled[index] = 1;
        if (originatingOutlet[index] < 0) originatingOutlet[index] = outlet;
        if (naturalOutletReceiver(solution, terminals, index) &&
            receivingWaterSurface(solution, index) <=
                solution.waterSurface[originatingOutlet[index]] + 0.02F) {
            receiving = index;
            break;
        }

        const Receiver& receiver = receivers[index];
        for (const auto [target, weight] : {std::pair{receiver.first, receiver.firstWeight},
                                            std::pair{receiver.second, receiver.secondWeight}}) {
            if (target < 0 || weight <= 0.0F || lakeMember[target] != 0 || settled[target] != 0) {
                continue;
            }
            if (predecessor[target] == -2) {
                predecessor[target] = index;
                originatingOutlet[target] = originatingOutlet[index];
            }
            pending.emplace(steps + 1, filled[target], floodOrder[target], target,
                            originatingOutlet[target]);
        }
    }

    LakeOutletRoute result;
    if (receiving < 0 || originatingOutlet[receiving] < 0) return result;
    result.outlet = originatingOutlet[receiving];
    for (int index = receiving; index >= 0; index = predecessor[index]) {
        result.path.push_back(index);
        if (predecessor[index] == -1 || predecessor[index] == -2) break;
    }
    std::ranges::reverse(result.path);
    return result;
}

void emitLakeOutletCorridor(BasinSolution& solution, const LakeOutletRoute& route) {
    if (route.outlet < 0 || route.path.empty()) return;
    const float topSurface = solution.waterSurface[route.outlet];
    const int receiving = route.path.back();
    float bottomSurface = std::min(topSurface, receivingWaterSurface(solution, receiving));
    if (!route.receivingPath.empty()) {
        // The first active raster reach can sit above its eventual standing
        // receiver. Use the complete bounded receiver walk as the hydraulic
        // control so the outlet fall lands at the lake or ocean stage instead
        // of ending in a perched pool that would immediately spill again.
        bottomSurface =
            std::min(bottomSurface, receivingWaterSurface(solution, route.receivingPath.back()));
    }
    float discharge = std::max(localMaximumDischarge(solution, route.outlet),
                               localMaximumDischarge(solution, receiving));
    const uint16_t equilibriumIndex = solution.lakeEquilibriumByCell[route.outlet];
    if (equilibriumIndex != NO_LAKE_EQUILIBRIUM &&
        equilibriumIndex < solution.lakeEquilibria.size()) {
        discharge = std::max(discharge,
                             solution.lakeEquilibria[equilibriumIndex].overflowMmSquareKilometers);
    }
    discharge = std::max(discharge, 1.0F);
    const int order =
        std::clamp(1 + static_cast<int>(std::floor(std::log2(discharge / 500.0F + 1.0F))), 1, 6);
    const float channelWidth =
        static_cast<float>(std::clamp(3.0 + std::sqrt(discharge) * 0.22 + order * 1.2, 4.0, 42.0));
    const float channelDepth = static_cast<float>(
        std::clamp(1.2 + std::sqrt(discharge) * 0.065 + order * 0.65, 1.8, 14.0));

    LakeOutletCorridor corridor;
    corridor.outletCell = route.outlet;
    corridor.discharge = discharge;
    corridor.halfWidth = std::clamp(channelWidth * 0.55F, 2.0F, 12.0F);
    corridor.points.reserve(route.path.size() + 1);
    corridor.receivingRoute.reserve(route.receivingPath.size());
    corridor.waterSurface.reserve(route.path.size() + 1);
    const auto pointFor = [&](int index) {
        const int gridX = index % RASTER_EDGE;
        const int gridZ = index / RASTER_EDGE;
        return GuidePoint{
            solution.originX + (gridX - RASTER_APRON) * BASIN_RASTER_SPACING,
            solution.originZ + (gridZ - RASTER_APRON) * BASIN_RASTER_SPACING,
        };
    };
    corridor.points.push_back(pointFor(route.outlet));
    corridor.waterSurface.push_back(topSurface);
    for (const int index : route.path)
        corridor.points.push_back(pointFor(index));
    for (const int index : route.receivingPath)
        corridor.receivingRoute.push_back(pointFor(index));

    std::vector<double> cumulative(corridor.points.size(), 0.0);
    for (size_t index = 1; index < corridor.points.size(); ++index) {
        cumulative[index] = cumulative[index - 1] +
                            std::hypot(corridor.points[index].x - corridor.points[index - 1].x,
                                       corridor.points[index].z - corridor.points[index - 1].z);
    }
    const double totalLength = std::max(1.0, cumulative.back());
    for (size_t index = 1; index < corridor.points.size(); ++index) {
        const double progress = quinticWeight(cumulative[index] / totalLength);
        corridor.waterSurface.push_back(
            static_cast<float>(std::lerp(topSurface, bottomSurface, progress)));
    }

    for (size_t pathIndex = 0; pathIndex < route.path.size(); ++pathIndex) {
        const int index = route.path[pathIndex];
        const float water = corridor.waterSurface[pathIndex + 1];
        if ((solution.flags[index] & (CELL_OCEAN | CELL_LAKE)) == 0) {
            solution.flags[index] |= CELL_LAKE_CONNECTOR;
            solution.waterSurface[index] = water;
            solution.surface[index] = std::min(solution.surface[index], water - channelDepth);
            solution.channelDistance[index] = 0.0F;
            solution.channelWidth[index] = std::max(solution.channelWidth[index], channelWidth);
            solution.channelDepth[index] = solution.waterSurface[index] - solution.surface[index];
            solution.discharge[index] = std::max(solution.discharge[index], discharge);
            solution.streamOrder[index] =
                std::max(solution.streamOrder[index], static_cast<uint8_t>(order));
        }
        const size_t currentPoint = pathIndex + 1;
        const size_t nextPoint =
            currentPoint + 1 < corridor.points.size() ? currentPoint + 1 : currentPoint;
        const size_t previousPoint = currentPoint > 0 ? currentPoint - 1 : currentPoint;
        const GuidePoint& current = corridor.points[currentPoint];
        const GuidePoint& next =
            nextPoint != currentPoint ? corridor.points[nextPoint] : corridor.points[previousPoint];
        const double length = std::hypot(next.x - current.x, next.z - current.z);
        if (length > 1.0e-6) {
            const double sign = nextPoint != currentPoint ? 1.0 : -1.0;
            solution.flowX[index] = static_cast<float>((next.x - current.x) / length * sign);
            solution.flowZ[index] = static_cast<float>((next.z - current.z) / length * sign);
        }
    }

    for (size_t index = 0; index + 1 < corridor.waterSurface.size(); ++index) {
        const float drop = corridor.waterSurface[index] - corridor.waterSurface[index + 1];
        if (drop < 2.5F) continue;
        solution.outletFalls.push_back({
            .startX = corridor.points[index].x,
            .startZ = corridor.points[index].z,
            .endX = corridor.points[index + 1].x,
            .endZ = corridor.points[index + 1].z,
            .topSurface = corridor.waterSurface[index],
            .bottomSurface = corridor.waterSurface[index + 1],
            .discharge = discharge,
            // The vertical curtain spans the complete routed outlet. A
            // narrower fall leaves the high and low corridor stages touching
            // around its sides as an untagged water wall.
            .halfWidth = std::clamp(corridor.halfWidth + 0.75F, 2.0F, 12.0F),
        });
    }
    solution.outletCorridors.push_back(std::move(corridor));
}

void restoreLakeOutletMarkers(BasinSolution& solution, const std::vector<Receiver>& receivers,
                              const std::vector<float>& filled,
                              const std::vector<uint32_t>& floodOrder,
                              const std::vector<uint8_t>& terminals,
                              bool recordOutletFalls = false) {
    if (recordOutletFalls) {
        solution.outletFalls.clear();
        solution.outletCorridors.clear();
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
        while (cursor < queue.size()) {
            const int index = queue[cursor++];
            solution.flags[index] &= static_cast<uint8_t>(~CELL_LAKE_OUTLET);
            endorheic = endorheic || (solution.flags[index] & CELL_ENDORHEIC) != 0;
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
        LakeOutletRoute route = endorheic ? LakeOutletRoute{}
                                          : findLakeOutletRoute(solution, receivers, filled,
                                                                floodOrder, terminals, queue);
        if (!endorheic && route.outlet >= 0 && !route.path.empty()) {
            solution.flags[route.outlet] |= CELL_LAKE_OUTLET;
            if (recordOutletFalls) {
                route.receivingPath =
                    traceReceivingRoute(solution, receivers, terminals, route.path.back());
                emitLakeOutletCorridor(solution, route);
            }
        } else if (!endorheic) {
            throw std::runtime_error("through-flow lake has no bounded outlet route");
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
            solution.lakeEquilibriumByCell[index] = NO_LAKE_EQUILIBRIUM;
            continue;
        }
        solution.surface[index] = solution.waterSurface[index] - depth;
    }

    // Dry shoreline support is reconstructed continuously from the signed
    // lake field at sample time. Keeping it out of this categorical raster
    // avoids a one-cell wall and lets the bank taper over sixteen blocks.
}

void assignWaterBodyIds(BasinSolution& solution, const Site& drainageRoot) {
    solution.waterBodyIds.assign(RASTER_CELLS, NO_WATER_BODY);
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    std::vector<int> queue;
    for (int start = 0; start < RASTER_CELLS; ++start) {
        if (visited[start] != 0 || (solution.flags[start] & CELL_LAKE) == 0) continue;
        queue.clear();
        queue.push_back(start);
        visited[start] = 1;
        size_t cursor = 0;
        int anchor = start;
        int anchorPriority = -1;
        bool endorheic = false;
        const float waterLevel = solution.waterSurface[start];
        while (cursor < queue.size()) {
            const int index = queue[cursor++];
            endorheic = endorheic || (solution.flags[index] & CELL_ENDORHEIC) != 0;
            int priority = 0;
            if (solution.outletTypes[index] == static_cast<uint8_t>(BasinOutlet::ENDORHEIC)) {
                priority = 2;
            } else if ((solution.flags[index] & CELL_LAKE_OUTLET) != 0) {
                priority = 1;
            }
            const int64_t globalX =
                solution.key.x * INTERIOR_INTERVALS + index % RASTER_EDGE - RASTER_APRON;
            const int64_t globalZ =
                solution.key.z * INTERIOR_INTERVALS + index / RASTER_EDGE - RASTER_APRON;
            const int64_t anchorGlobalX =
                solution.key.x * INTERIOR_INTERVALS + anchor % RASTER_EDGE - RASTER_APRON;
            const int64_t anchorGlobalZ =
                solution.key.z * INTERIOR_INTERVALS + anchor / RASTER_EDGE - RASTER_APRON;
            if (priority > anchorPriority ||
                (priority == anchorPriority &&
                 std::pair{globalX, globalZ} < std::pair{anchorGlobalX, anchorGlobalZ})) {
                anchor = index;
                anchorPriority = priority;
            }

            const int x = index % RASTER_EDGE;
            const int z = index / RASTER_EDGE;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + DX[direction];
                const int nz = z + DZ[direction];
                if (!inRaster(nx, nz)) continue;
                const int neighbor = indexOf(nx, nz);
                if (visited[neighbor] != 0 || (solution.flags[neighbor] & CELL_LAKE) == 0 ||
                    std::abs(solution.waterSurface[neighbor] - waterLevel) > 1.0e-4F) {
                    continue;
                }
                visited[neighbor] = 1;
                queue.push_back(neighbor);
            }
        }
        // Every independently built catchment in the same drainage tree uses
        // the immutable macro root and flat body level. Basin-local raster
        // anchors are deliberately excluded because they change at a storage
        // face even when both sides describe one physical lake.
        const WaterBodyId identity =
            makeWaterBodyId(drainageRoot.cell.x, drainageRoot.cell.z, waterLevel, endorheic);
        for (const int index : queue) {
            solution.waterBodyIds[index] = identity;
            const uint16_t equilibriumIndex = solution.lakeEquilibriumByCell[index];
            if (equilibriumIndex != NO_LAKE_EQUILIBRIUM &&
                equilibriumIndex < solution.lakeEquilibria.size()) {
                LakeEquilibrium& equilibrium = solution.lakeEquilibria[equilibriumIndex];
                if (equilibrium.identity == NO_WATER_BODY || equilibrium.identity == identity) {
                    equilibrium.identity = identity;
                }
            }
        }
    }
    for (LakeOutletCorridor& corridor : solution.outletCorridors) {
        if (corridor.outletCell >= 0 && corridor.outletCell < RASTER_CELLS) {
            corridor.body = solution.waterBodyIds[static_cast<size_t>(corridor.outletCell)];
        }
    }
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
        !finiteField(solution.flowX) || !finiteField(solution.flowZ) ||
        !finiteField(solution.rasterSeaBackwaterDistance)) {
        return false;
    }
    if (solution.outletTypes.size() != RASTER_CELLS || terminals.size() != RASTER_CELLS ||
        accumulation.size() != RASTER_CELLS || solution.outletFallByCell.size() != RASTER_CELLS ||
        solution.lakeShoreDistance.size() != RASTER_CELLS ||
        solution.shoreWaterSurface.size() != RASTER_CELLS ||
        solution.waterBodyIds.size() != RASTER_CELLS ||
        solution.shoreWaterBodyIds.size() != RASTER_CELLS ||
        solution.shoreWaterEndorheic.size() != RASTER_CELLS ||
        solution.rasterReceivers.size() != RASTER_CELLS ||
        solution.rasterSeaBackwaterDistance.size() != RASTER_CELLS ||
        solution.lakeEquilibriumByCell.size() != RASTER_CELLS) {
        return false;
    }
    for (const LakeEquilibrium& equilibrium : solution.lakeEquilibria) {
        if (!std::isfinite(equilibrium.surface) || !std::isfinite(equilibrium.spillSurface) ||
            !std::isfinite(equilibrium.areaSquareKilometers) ||
            !std::isfinite(equilibrium.volumeCubicMeters) ||
            !std::isfinite(equilibrium.runoffMmSquareKilometers) ||
            !std::isfinite(equilibrium.lossMm) ||
            !std::isfinite(equilibrium.overflowMmSquareKilometers) ||
            !std::isfinite(equilibrium.maximumDepth) ||
            equilibrium.surface > equilibrium.spillSurface + 1.0e-4F ||
            equilibrium.areaSquareKilometers <= 0.0F || equilibrium.volumeCubicMeters <= 0.0F ||
            equilibrium.lossMm <= 0.0F) {
            return false;
        }
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
            !std::isfinite(guide.startWater) || !std::isfinite(guide.endWater) ||
            !std::isfinite(guide.terrainUpper) || guide.startWater + 1.0e-4 < guide.endWater ||
            (guide.outgoing && (std::abs(guide.endWater - guide.portalWater) > 1.0e-4 ||
                                guide.startWater + 1.0e-4 < guide.portalWater)) ||
            (!guide.outgoing && (std::abs(guide.startWater - guide.portalWater) > 1.0e-4 ||
                                 guide.portalWater + 1.0e-4 < guide.endWater)) ||
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
        // Outlet corridors replace the raster channel profile with their own
        // explicitly validated monotonic stage. A tributary may enter that
        // connector inside its backwater reach, so validate ordinary raster
        // ascent only when neither endpoint belongs to the corridor.
        const bool connectorTransition =
            (solution.flags[index] & CELL_LAKE_CONNECTOR) != 0 ||
            (receiver.first >= 0 && (solution.flags[receiver.first] & CELL_LAKE_CONNECTOR) != 0);
        if (receiver.first >= 0 && solution.streamOrder[index] > 0 &&
            solution.streamOrder[receiver.first] > 0 && solution.channelDistance[index] < 0.01F &&
            solution.channelDistance[receiver.first] < 0.01F &&
            (solution.flags[index] & (CELL_OCEAN | CELL_LAKE)) == 0 &&
            (solution.flags[receiver.first] & (CELL_OCEAN | CELL_LAKE)) == 0 &&
            !connectorTransition && !lockedBoundary(index % RASTER_EDGE, index / RASTER_EDGE) &&
            !lockedBoundary(receiver.first % RASTER_EDGE, receiver.first / RASTER_EDGE) &&
            solution.waterSurface[receiver.first] > solution.waterSurface[index] + 0.02F) {
            return false;
        }
        if ((solution.flags[index] & CELL_LAKE) != 0) {
            if (solution.lakeDepth[index] <= 0.0F || solution.lakeShoreDistance[index] <= 0.0F ||
                solution.waterBodyIds[index] == NO_WATER_BODY ||
                std::abs((solution.waterSurface[index] - solution.surface[index]) -
                         solution.lakeDepth[index]) > 1.0e-3F) {
                return false;
            }
        } else if ((solution.flags[index] & CELL_OCEAN) == 0 &&
                   solution.lakeShoreDistance[index] > 0.0F) {
            return false;
        } else if (solution.waterBodyIds[index] != NO_WATER_BODY) {
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
        const float minimumDrop = fall.transitionPromotion ? 0.999F : 2.5F;
        if (!std::isfinite(fall.startX) || !std::isfinite(fall.startZ) ||
            !std::isfinite(fall.endX) || !std::isfinite(fall.endZ) ||
            !std::isfinite(fall.topSurface) || !std::isfinite(fall.bottomSurface) ||
            !std::isfinite(fall.discharge) || !std::isfinite(fall.halfWidth) ||
            fall.discharge <= 0.0F || fall.topSurface < fall.bottomSurface + minimumDrop ||
            fall.halfWidth < 2.0F || fall.halfWidth > 24.0F) {
            return false;
        }
    }
    for (const LakeOutletCorridor& corridor : solution.outletCorridors) {
        if (corridor.points.size() < 2 || corridor.waterSurface.size() != corridor.points.size() ||
            corridor.body == NO_WATER_BODY || corridor.outletCell < 0 ||
            corridor.outletCell >= RASTER_CELLS || !std::isfinite(corridor.discharge) ||
            !std::isfinite(corridor.halfWidth) || corridor.discharge <= 0.0F ||
            corridor.halfWidth <= 0.0F) {
            return false;
        }
        for (size_t index = 0; index < corridor.points.size(); ++index) {
            if (!std::isfinite(corridor.points[index].x) ||
                !std::isfinite(corridor.points[index].z) ||
                !std::isfinite(corridor.waterSurface[index]) ||
                (index > 0 &&
                 corridor.waterSurface[index] > corridor.waterSurface[index - 1] + 1.0e-4F)) {
                return false;
            }
            if (index == 0) continue;
            const double run = std::hypot(corridor.points[index].x - corridor.points[index - 1].x,
                                          corridor.points[index].z - corridor.points[index - 1].z);
            if (run < 1.0e-4) return false;
            const float drop = corridor.waterSurface[index - 1] - corridor.waterSurface[index];
            if (drop < 2.5F) {
                if (drop / run > 2.5F + 1.0e-4F) return false;
                continue;
            }
            const bool markedFall =
                std::ranges::any_of(solution.outletFalls, [&](const OutletFall& fall) {
                    return std::abs(fall.startX - corridor.points[index - 1].x) < 1.0e-4 &&
                           std::abs(fall.startZ - corridor.points[index - 1].z) < 1.0e-4 &&
                           std::abs(fall.endX - corridor.points[index].x) < 1.0e-4 &&
                           std::abs(fall.endZ - corridor.points[index].z) < 1.0e-4 &&
                           std::abs(fall.topSurface - corridor.waterSurface[index - 1]) < 1.0e-4F &&
                           std::abs(fall.bottomSurface - corridor.waterSurface[index]) < 1.0e-4F;
                });
            if (!markedFall) return false;
        }
    }
    return true;
}

std::shared_ptr<const BasinSolution>
buildSolution(const CounterRng& random, BasinKey key,
              const BasinSolver::ElevationFunction& elevation,
              const BasinSolver::RainfallFunction& rainfall,
              const BasinSolver::RockResistanceFunction& rockResistance,
              const BasinSolver::PotentialEvapotranspirationFunction& potentialEvapotranspiration) {
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
    std::vector<float> pet;
    std::vector<float> resistance;
    sampleInputs(*solution, elevation, rainfall, rockResistance, potentialEvapotranspiration, raw,
                 rain, pet, resistance);
    carveChannelGuides(*solution, guides, raw);
    std::vector<float> terminalFloor;
    const std::vector<uint8_t> terminals =
        buildOutletConstraints(*solution, center, downstream, guides, raw, terminalFloor);

    std::vector<float> filled;
    std::vector<uint32_t> floodOrder;
    priorityFlood(raw, terminals, terminalFloor, filled, floodOrder);
    std::vector<Receiver> receivers;
    std::vector<float> routingFlowX;
    std::vector<float> routingFlowZ;
    std::vector<int32_t> upstreamOrder;
    buildDInfinityRouting(filled, floodOrder, terminals, receivers, routingFlowX, routingFlowZ,
                          upstreamOrder);
    std::vector<float> accumulation;
    accumulateFlow(rain, receivers, upstreamOrder, accumulation, *solution, guides);

    std::vector<float> eroded = raw;
    std::vector<float> transportedSediment(RASTER_CELLS, 0.0F);
    std::vector<Receiver> reroutedReceivers;
    ErosionScratch erosionScratch;
    for (int epoch = 0; epoch < EROSION_EPOCHS; ++epoch) {
        erodeTerrainEpoch(resistance, receivers, upstreamOrder, accumulation,
                          EROSION_PASSES_PER_EPOCH, eroded, transportedSediment, erosionScratch);
        priorityFlood(eroded, terminals, terminalFloor, filled, floodOrder);
        buildDInfinityRouting(filled, floodOrder, terminals, reroutedReceivers, routingFlowX,
                              routingFlowZ, upstreamOrder);
        solution->erosionReceiverChanges += changedReceiverCount(receivers, reroutedReceivers);
        receivers.swap(reroutedReceivers);
        accumulateFlow(rain, receivers, upstreamOrder, accumulation, *solution, guides);
        if (!validateErosionEpoch(raw, eroded, transportedSediment, filled, floodOrder, terminals,
                                  receivers, upstreamOrder, erosionScratch.orderPosition)) {
            throw std::runtime_error("invalid staged basin erosion");
        }
        ++solution->erosionEpochs;
        ++solution->erosionReroutes;
    }

    if (breachShallowDepressions(eroded, filled, receivers, terminals, rain, pet, resistance,
                                 accumulation)) {
        priorityFlood(eroded, terminals, terminalFloor, filled, floodOrder);
        buildDInfinityRouting(filled, floodOrder, terminals, receivers, routingFlowX, routingFlowZ,
                              upstreamOrder);
    }
    accumulateFlow(rain, receivers, upstreamOrder, accumulation, *solution, guides);
    std::vector<uint8_t> strahler;
    computeStrahler(receivers, upstreamOrder, accumulation, strahler);

    std::vector<uint8_t> flags;
    std::vector<float> lakeSurface;
    std::vector<float> classifiedLakeDepth;
    classifyLakes(*solution, center, downstream, terminalLake, eroded, filled, rain, pet,
                  resistance, accumulation, receivers, terminals, flags, lakeSurface,
                  classifiedLakeDepth);

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
                const Portal mouth = sharedPortal(deltaSite, deltaDownstream);
                mouthX = mouth.x;
                mouthZ = mouth.z;
                // Continental interiors may drain toward an old, deep ocean
                // site while their actual mouth still crosses a shallow
                // shelf. Delta suitability belongs to that canonical mouth,
                // not to the downstream catchment center.
                if (elevation(mouthX, mouthZ) < SEA_LEVEL - 28.0) continue;
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
    solution->rasterReceivers.resize(RASTER_CELLS);
    std::ranges::transform(receivers, solution->rasterReceivers.begin(),
                           [](const Receiver& receiver) { return receiver.first; });
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
        solution->erosionDepth[index] =
            std::max(solution->erosionDepth[index], raw[index] - solution->surface[index]);
        if ((flags[source] & CELL_WATERFALL) != 0 &&
            channelDistance[index] <= localWidth[source] * 0.60F) {
            solution->flags[index] |= CELL_WATERFALL;
        }
    }

    enforceSharedBoundaryConstraints(*solution, raw, guides);
    normalizeLakeGeometry(*solution);
    restoreLakeOutletMarkers(*solution, receivers, filled, floodOrder, terminals, true);
    const bool hasLake = std::ranges::any_of(
        solution->flags, [](uint8_t flags) { return (flags & CELL_LAKE) != 0; });
    const Site drainageRoot = macro.drainageRoot(center);
    solution->rasterDrainageOwnerId = rasterDrainageIdentity(drainageRoot);
    assignWaterBodyIds(*solution, drainageRoot);
    if (hasLake) {
        shapeLakeBathymetry(*solution, resistance);
        normalizeLakeGeometry(*solution);
    }
    // Shared-boundary replacement and shallow-fringe normalization can
    // remove raster lake cells after the first signed-distance solve. Rebuild
    // the categorical sign from the final flags before publishing immutable
    // shoreline authority, otherwise a dry cell can retain a positive
    // interior distance and invalidate the complete basin.
    rebuildLakeShoreDistance(*solution);
    propagateLakeShoreAuthority(*solution);
    buildRasterSeaBackwater(*solution, receivers, upstreamOrder);
    appendChannelFalls(*solution, receivers);
    indexOutletFalls(*solution);
    buildRapidPatches(*solution, receivers);
    buildSettledRasterWater(*solution, receivers);
    solution->rapidCells.clear();
    solution->rapidCells.shrink_to_fit();

    if (!validateSolution(*solution, raw, filled, floodOrder, receivers, terminals, accumulation)) {
        throw std::runtime_error("invalid bounded basin solution");
    }
    return solution;
}

std::shared_ptr<const BasinSolution>
buildFallbackSolution(const CounterRng& random, BasinKey key,
                      const BasinSolver::ElevationFunction& elevation,
                      const BasinSolver::RainfallFunction& rainfall) {
    auto solution = std::make_shared<BasinSolution>();
    solution->key = key;
    solution->originX = static_cast<double>(key.x) * BASIN_CATCHMENT_EDGE;
    solution->originZ = static_cast<double>(key.z) * BASIN_CATCHMENT_EDGE;
    const auto floats = [&](float value) { return std::vector<float>(RASTER_CELLS, value); };
    solution->surface = floats(SEA_LEVEL + 8.0F);
    solution->waterSurface = floats(SEA_LEVEL);
    solution->discharge = floats(0.0F);
    solution->sediment = floats(0.0F);
    solution->channelDistance = floats(std::numeric_limits<float>::max());
    solution->channelWidth = floats(0.0F);
    solution->channelDepth = floats(0.0F);
    solution->channelGradient = floats(0.0F);
    solution->erosionDepth = floats(0.0F);
    solution->lakeDepth = floats(0.0F);
    solution->lakeShoreDistance = floats(-1.0e9F);
    solution->shoreWaterSurface = floats(0.0F);
    solution->waterBodyIds.assign(RASTER_CELLS, NO_WATER_BODY);
    solution->shoreWaterBodyIds.assign(RASTER_CELLS, NO_WATER_BODY);
    solution->shoreWaterEndorheic.assign(RASTER_CELLS, 0);
    solution->flowX = floats(0.0F);
    solution->flowZ = floats(0.0F);
    solution->rasterReceivers.assign(RASTER_CELLS, -1);
    solution->rasterSeaBackwaterDistance.assign(RASTER_CELLS, -1.0F);
    solution->streamOrder.assign(RASTER_CELLS, 0);
    solution->flags.assign(RASTER_CELLS, 0);
    const Site center = makeSite(random, key, elevation, rainfall);
    const Site downstream = downstreamSite(random, center, elevation, rainfall);
    BasinOutlet fallbackOutlet = BasinOutlet::ENDORHEIC;
    if (center.ocean) {
        fallbackOutlet = BasinOutlet::OCEAN;
        solution->namedOutletX = center.x;
        solution->namedOutletZ = center.z;
    } else if (!sameCell(center, downstream)) {
        fallbackOutlet = BasinOutlet::SHARED_PORTAL;
        const Portal portal = sharedPortal(center, downstream);
        solution->namedOutletX = portal.x;
        solution->namedOutletZ = portal.z;
    } else {
        solution->namedOutletX = center.x;
        solution->namedOutletZ = center.z;
    }
    solution->outletTypes.assign(RASTER_CELLS, static_cast<uint8_t>(fallbackOutlet));
    solution->lakeEquilibriumByCell.assign(RASTER_CELLS, NO_LAKE_EQUILIBRIUM);
    solution->outletFallByCell.assign(RASTER_CELLS, -1);
    solution->rasterDrainageOwnerId =
        transitionIdentity(0x5241535445525F52ULL, solution->namedOutletX, solution->namedOutletZ,
                           static_cast<double>(key.x), static_cast<double>(key.z));

    constexpr double CELL_SQUARE_KILOMETERS =
        BASIN_RASTER_SPACING * BASIN_RASTER_SPACING / 1'000'000.0;
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            const double worldX = solution->originX + (x - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double worldZ = solution->originZ + (z - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double sampledElevation = elevation(worldX, worldZ);
            const float base = std::isfinite(sampledElevation)
                                   ? static_cast<float>(std::clamp(sampledElevation, -112.0, 480.0))
                                   : SEA_LEVEL + 8.0F;
            solution->surface[index] = base;
            const double sampledRainfall = rainfall(worldX, worldZ, base);
            solution->discharge[index] = static_cast<float>(
                (std::isfinite(sampledRainfall) ? std::max(0.0, sampledRainfall) : 0.0) *
                CELL_SQUARE_KILOMETERS);
            if (base < SEA_LEVEL) {
                solution->flags[index] |= CELL_OCEAN;
                solution->outletTypes[index] = static_cast<uint8_t>(BasinOutlet::OCEAN);
            }
        }
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

double dominantPositiveLevel(const BasinSolution& solution, double gridX, double gridZ) {
    const RasterWeights weights = reconstructionWeights(gridX, gridZ);
    struct Candidate {
        double level = 0.0;
        double weight = 0.0;
    };
    std::array<Candidate, 4> candidates{};
    size_t candidateCount = 0;
    for (size_t index = 0; index < weights.indices.size(); ++index) {
        const double level = solution.shoreWaterSurface[weights.indices[index]];
        if (level <= 0.0 || weights.values[index] <= 0.0) continue;
        size_t candidate = 0;
        while (candidate < candidateCount &&
               std::abs(candidates[candidate].level - level) > 1.0e-4) {
            ++candidate;
        }
        if (candidate == candidateCount) candidates[candidateCount++] = {.level = level};
        candidates[candidate].weight += weights.values[index];
    }
    if (candidateCount == 0) return 0.0;
    return std::max_element(candidates.begin(),
                            candidates.begin() + static_cast<std::ptrdiff_t>(candidateCount),
                            [](const Candidate& first, const Candidate& second) {
                                if (first.weight != second.weight)
                                    return first.weight < second.weight;
                                return first.level > second.level;
                            })
        ->level;
}

double shorelineOutletProtection(const BasinSolution& solution, const LakeBodySample& lake,
                                 double x, double z) {
    double protection = 1.0;
    for (const OutletFall& fall : solution.outletFalls) {
        if (std::abs(fall.topSurface - lake.waterLevel) > 1.0e-3F) continue;
        protection = std::min(protection,
                              smoothstep(10.0, 64.0, std::hypot(x - fall.startX, z - fall.startZ)));
    }
    for (const ChannelGuide& guide : solution.guides) {
        if (std::abs(guide.portalWater - lake.waterLevel) > 1.0e-3) continue;
        protection = std::min(
            protection, smoothstep(10.0, 64.0, std::hypot(x - guide.portalX, z - guide.portalZ)));
    }
    return protection;
}

LakeAuthoritySample sampleLakeAuthority(const BasinSolution& solution, double x, double z) {
    const double gridX = (x - solution.originX) / BASIN_RASTER_SPACING + RASTER_APRON;
    const double gridZ = (z - solution.originZ) / BASIN_RASTER_SPACING + RASTER_APRON;
    const LakeBodySample lake = sampleCandidateLakeBody(solution, gridX, gridZ);
    LakeAuthoritySample result;
    result.floor = reconstruct(solution.surface, gridX, gridZ);
    if (!lake.found) return result;

    result.identity = lake.identity;
    result.waterLevel = lake.waterLevel;
    result.bankWaterLevel = lake.waterLevel;
    result.endorheic = lake.endorheic;
    result.signedDistance =
        std::clamp(static_cast<double>(reconstruct(solution.lakeShoreDistance, gridX, gridZ)),
                   -SHORELINE_DISTANCE_LIMIT, SHORELINE_DISTANCE_LIMIT);
    if (lake.depth.weight > 0.0) {
        result.floor = lake.waterLevel - lake.depth.value;
    }
    const double bandWeight =
        1.0 - smoothstep(8.0, LAKE_SHORE_PERTURBATION_BAND, std::abs(result.signedDistance));
    result.signedDistance += organicShoreNoise(lake.identity, x, z) *
                             LAKE_SHORE_PERTURBATION_AMPLITUDE * bandWeight *
                             shorelineOutletProtection(solution, lake, x, z);
    result.signedDistance =
        std::clamp(result.signedDistance, -SHORELINE_DISTANCE_LIMIT, SHORELINE_DISTANCE_LIMIT);
    result.found = true;
    return result;
}

bool ownedLakeConnectorAt(const BasinSolution& solution, WaterBodyId body, double x, double z) {
    if (body == NO_WATER_BODY) return false;
    for (const LakeOutletCorridor& corridor : solution.outletCorridors) {
        if (corridor.points.size() < 2 || corridor.body != body) continue;
        for (size_t segment = 0; segment + 1 < corridor.points.size(); ++segment) {
            double along = 0.0;
            const double distance = distanceToSegment(
                x, z, corridor.points[segment].x, corridor.points[segment].z,
                corridor.points[segment + 1].x, corridor.points[segment + 1].z, along);
            if (distance <= corridor.halfWidth) return true;
        }
        for (size_t segment = 0; segment + 1 < corridor.receivingRoute.size(); ++segment) {
            double along = 0.0;
            const double distance = distanceToSegment(
                x, z, corridor.receivingRoute[segment].x, corridor.receivingRoute[segment].z,
                corridor.receivingRoute[segment + 1].x, corridor.receivingRoute[segment + 1].z,
                along);
            if (distance <= corridor.halfWidth) return true;
        }
    }
    const int centerX =
        std::clamp(static_cast<int>(std::llround((x - solution.originX) / BASIN_RASTER_SPACING)) +
                       RASTER_APRON,
                   0, RASTER_EDGE - 1);
    const int centerZ =
        std::clamp(static_cast<int>(std::llround((z - solution.originZ) / BASIN_RASTER_SPACING)) +
                       RASTER_APRON,
                   0, RASTER_EDGE - 1);
    for (int offsetZ = -2; offsetZ <= 2; ++offsetZ) {
        for (int offsetX = -2; offsetX <= 2; ++offsetX) {
            const int gridX = centerX + offsetX;
            const int gridZ = centerZ + offsetZ;
            if (!inRaster(gridX, gridZ)) continue;
            const int index = indexOf(gridX, gridZ);
            if ((solution.flags[index] & CELL_LAKE_OUTLET) == 0 ||
                solution.waterBodyIds[index] != body) {
                continue;
            }
            const double outletX = solution.originX + (gridX - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double outletZ = solution.originZ + (gridZ - RASTER_APRON) * BASIN_RASTER_SPACING;
            const double flowMagnitude = std::hypot(solution.flowX[index], solution.flowZ[index]);
            const double flowX =
                flowMagnitude > 1.0e-6 ? solution.flowX[index] / flowMagnitude : 1.0;
            const double flowZ =
                flowMagnitude > 1.0e-6 ? solution.flowZ[index] / flowMagnitude : 0.0;
            double along = 0.0;
            const double distance =
                distanceToSegment(x, z, outletX - flowX * BASIN_RASTER_SPACING * 0.5,
                                  outletZ - flowZ * BASIN_RASTER_SPACING * 0.5,
                                  outletX + flowX * BASIN_RASTER_SPACING * 2.0,
                                  outletZ + flowZ * BASIN_RASTER_SPACING * 2.0, along);
            const double halfWidth = std::clamp(solution.channelWidth[index] * 0.55, 2.0, 8.0);
            if (distance <= halfWidth) return true;
        }
    }
    return false;
}

struct LakeAuthorityKeySet {
    std::array<BasinKey, 9> values{};
    size_t count = 0;

    const BasinKey* begin() const { return values.data(); }
    const BasinKey* end() const { return values.data() + count; }
};

LakeAuthorityKeySet lakeAuthorityKeys(double x, double z) {
    constexpr double DISCOVERY_BAND = SHORELINE_DISTANCE_LIMIT;
    const BasinKey primary = samplingKey(x, z);
    const double localX = x - static_cast<double>(primary.x) * BASIN_CATCHMENT_EDGE;
    const double localZ = z - static_cast<double>(primary.z) * BASIN_CATCHMENT_EDGE;

    std::array<int64_t, 3> xCells = {primary.x, primary.x, primary.x};
    std::array<int64_t, 3> zCells = {primary.z, primary.z, primary.z};
    size_t xCount = 1;
    size_t zCount = 1;
    if (localX <= DISCOVERY_BAND) xCells[xCount++] = primary.x - 1;
    if (BASIN_CATCHMENT_EDGE - localX <= DISCOVERY_BAND) xCells[xCount++] = primary.x + 1;
    if (localZ <= DISCOVERY_BAND) zCells[zCount++] = primary.z - 1;
    if (BASIN_CATCHMENT_EDGE - localZ <= DISCOVERY_BAND) zCells[zCount++] = primary.z + 1;

    LakeAuthorityKeySet result;
    result.values[result.count++] = primary;
    for (size_t zIndex = 0; zIndex < zCount; ++zIndex) {
        for (size_t xIndex = 0; xIndex < xCount; ++xIndex) {
            const BasinKey candidate{xCells[xIndex], zCells[zIndex]};
            if (candidate == primary ||
                std::find(result.begin(), result.end(), candidate) != result.end()) {
                continue;
            }
            result.values[result.count++] = candidate;
        }
    }
    return result;
}

LakeAuthoritySample extrapolatedLakeAuthority(const BasinSolution& solution, double x, double z,
                                              const BasinSolver::ElevationFunction& elevation) {
    LakeAuthoritySample result = sampleLakeAuthority(solution, x, z);
    if (!result.found) return result;

    constexpr double APRON_BLOCKS = RASTER_APRON * BASIN_RASTER_SPACING;
    const double minimumX = solution.originX - APRON_BLOCKS;
    const double maximumX = solution.originX + BASIN_CATCHMENT_EDGE + APRON_BLOCKS;
    const double minimumZ = solution.originZ - APRON_BLOCKS;
    const double maximumZ = solution.originZ + BASIN_CATCHMENT_EDGE + APRON_BLOCKS;
    const double outsideX = std::max({minimumX - x, 0.0, x - maximumX});
    const double outsideZ = std::max({minimumZ - z, 0.0, z - maximumZ});
    const double outsideDistance = std::hypot(outsideX, outsideZ);
    if (outsideDistance > 0.0) {
        // Continue the signed contour away from the last authoritative apron
        // control instead of ending a lake on a straight raster boundary.
        result.signedDistance =
            std::max(-SHORELINE_DISTANCE_LIMIT, result.signedDistance - outsideDistance);
        result.floor = elevation(x, z);
    }
    return result;
}

struct SelectedLakeAuthority {
    LakeAuthoritySample sample;
    BasinKey key;
    bool connector = false;
    bool exteriorOnly = false;
    bool found = false;
};

// Lake authority is a categorical ownership decision, so point samples,
// sparse batches, and contour-page construction must share the same ordering
// and tie break. The lookup stays caller-owned so a dense batch can retain its
// solutions without adding an extra cache traversal per candidate.
template <typename SolutionAt>
SelectedLakeAuthority
selectLakeAuthority(double x, double z, const BasinSolver::ElevationFunction& elevation,
                    SolutionAt&& solutionAt, WaterBodyId requiredBody = NO_WATER_BODY) {
    // Neighboring bounded solutions can independently retain depressions
    // whose perturbed contours overlap across a catchment apron. Picking the
    // largest signed distance directly makes the winner alternate at block
    // scale and joins two flat surfaces with an unsupported vertical step.
    // Collapse duplicate observations of one body first, then clip distinct
    // bodies against a narrow competitive watershed. Each lake retains its
    // non-overlapping interior, while the organic contour fields create a
    // supported dry divide wherever two otherwise valid bodies meet.
    struct Candidate {
        LakeAuthoritySample sample;
        BasinKey key;
        bool connector = false;
        bool exteriorOnly = false;
    };
    std::array<Candidate, 81> candidates{};
    size_t candidateCount = 0;
    const auto stronger = [](const Candidate& first, const Candidate& second) {
        if (first.connector != second.connector) return first.connector;
        if (first.sample.signedDistance > second.sample.signedDistance + 1.0e-6) return true;
        if (second.sample.signedDistance > first.sample.signedDistance + 1.0e-6) return false;
        if (first.exteriorOnly != second.exteriorOnly) return !first.exteriorOnly;
        return std::pair{first.sample.identity, std::pair{first.key.x, first.key.z}} <
               std::pair{second.sample.identity, std::pair{second.key.x, second.key.z}};
    };
    const auto retainCandidate = [&](const BasinSolution& solution, BasinKey candidateKey,
                                     const LakeAuthoritySample& candidate, bool exteriorOnly) {
        if (!candidate.found || candidate.identity == NO_WATER_BODY) return;
        const bool connector = ownedLakeConnectorAt(solution, candidate.identity, x, z);
        if (exteriorOnly && connector) return;
        const Candidate observation{candidate, candidateKey, connector, exteriorOnly};
        size_t body = 0;
        while (body < candidateCount && candidates[body].sample.identity != candidate.identity) {
            ++body;
        }
        if (body == candidateCount) {
            candidates[candidateCount++] = observation;
        } else if (stronger(observation, candidates[body])) {
            candidates[body] = observation;
        }
    };
    constexpr std::array<std::pair<double, double>, 8> NEARBY_BODY_DIRECTIONS = {
        std::pair{-1.0, 0.0},  std::pair{1.0, 0.0},  std::pair{0.0, -1.0}, std::pair{0.0, 1.0},
        std::pair{-1.0, -1.0}, std::pair{1.0, -1.0}, std::pair{-1.0, 1.0}, std::pair{1.0, 1.0}};
    const LakeAuthorityKeySet authorityKeys = lakeAuthorityKeys(x, z);
    const bool sharedCatchmentApron = authorityKeys.count > 1;
    for (const BasinKey candidateKey : authorityKeys) {
        const BasinSolution& solution = solutionAt(candidateKey);
        const LakeAuthoritySample direct = extrapolatedLakeAuthority(solution, x, z, elevation);
        retainCandidate(solution, candidateKey, direct, false);

        const double gridX = (x - solution.originX) / BASIN_RASTER_SPACING + RASTER_APRON;
        const double gridZ = (z - solution.originZ) / BASIN_RASTER_SPACING + RASTER_APRON;
        const RasterWeights weights = reconstructionWeights(gridX, gridZ);
        WaterBodyId firstBody = NO_WATER_BODY;
        bool mixedBodyControls = false;
        if (solution.shoreWaterBodyIds.size() == RASTER_CELLS) {
            for (const int index : weights.indices) {
                const WaterBodyId body = solution.shoreWaterBodyIds[index];
                if (body == NO_WATER_BODY) continue;
                if (firstBody == NO_WATER_BODY) {
                    firstBody = body;
                } else if (body != firstBody) {
                    mixedBodyControls = true;
                    break;
                }
            }
        }
        const bool withinContourBand =
            direct.found && std::abs(direct.signedDistance) <= LAKE_SHORE_PERTURBATION_BAND + 4.0;
        if (!withinContourBand && !mixedBodyControls && !sharedCatchmentApron) continue;

        for (const auto [offsetX, offsetZ] : NEARBY_BODY_DIRECTIONS) {
            LakeAuthoritySample nearby =
                extrapolatedLakeAuthority(solution, x + offsetX, z + offsetZ, elevation);
            if (!nearby.found || nearby.identity == NO_WATER_BODY) continue;
            // A neighbor observation only retains body identity and rim
            // authority. It is always exterior, so categorical discovery can
            // never enlarge the lake or move its wet contour.
            nearby.signedDistance =
                std::clamp(std::min(-0.125, nearby.signedDistance - std::hypot(offsetX, offsetZ)),
                           -SHORELINE_DISTANCE_LIMIT, -0.125);
            nearby.floor = elevation(x, z);
            retainCandidate(solution, candidateKey, nearby, true);
        }
    }

    SelectedLakeAuthority selected;
    selected.key = samplingKey(x, z);
    const Candidate* winner = nullptr;
    for (size_t index = 0; index < candidateCount; ++index) {
        const Candidate& candidate = candidates[index];
        if (requiredBody != NO_WATER_BODY && candidate.sample.identity != requiredBody) continue;
        if (winner == nullptr || stronger(candidate, *winner)) winner = &candidate;
    }
    if (winner == nullptr) return selected;
    selected.sample = winner->sample;
    selected.key = winner->key;
    selected.connector = winner->connector;
    selected.exteriorOnly = winner->exteriorOnly;
    selected.found = true;

    const Candidate* competitor = nullptr;
    for (size_t index = 0; index < candidateCount; ++index) {
        const Candidate& candidate = candidates[index];
        if (candidate.sample.identity == winner->sample.identity) continue;
        if (competitor == nullptr || stronger(candidate, *competitor)) competitor = &candidate;
    }
    if (competitor != nullptr && !winner->connector && !competitor->connector) {
        constexpr double WATERSHED_HALF_WIDTH = 2.5;
        const double competitiveDistance =
            (winner->sample.signedDistance - competitor->sample.signedDistance) * 0.5 -
            WATERSHED_HALF_WIDTH;
        if (competitiveDistance < selected.sample.signedDistance) {
            selected.sample.signedDistance = competitiveDistance;
            selected.sample.bankWaterLevel =
                std::max(selected.sample.waterLevel, competitor->sample.waterLevel);
        }
    }
    return selected;
}

ShorelineContourSample exteriorLakeContour(const SelectedLakeAuthority& authority,
                                           BasinKey primaryKey) {
    if (!authority.found) {
        return {};
    }
    if (!authority.exteriorOnly &&
        (authority.key == primaryKey || authority.sample.signedDistance <= 0.0)) {
        return {};
    }
    return {
        .signedDistance = authority.exteriorOnly ? std::min(-0.125, authority.sample.signedDistance)
                                                 : authority.sample.signedDistance,
        .floor = authority.sample.floor,
        .waterLevel = authority.sample.waterLevel,
        .bankWaterLevel = authority.sample.bankWaterLevel,
        .identity = authority.sample.identity,
        .endorheic = authority.sample.endorheic,
        .outletContinuation = authority.connector,
        .valid = true,
    };
}

int16_t encodeShorelineDistance(double value) {
    const long encoded =
        std::lround(std::clamp(value, -SHORELINE_DISTANCE_LIMIT, SHORELINE_DISTANCE_LIMIT) *
                    SHORELINE_DISTANCE_QUANTIZATION);
    return static_cast<int16_t>(std::clamp(encoded,
                                           static_cast<long>(std::numeric_limits<int16_t>::min()),
                                           static_cast<long>(std::numeric_limits<int16_t>::max())));
}

int16_t encodeShorelineFloor(double value) {
    const long encoded = std::lround(value * SHORELINE_FLOOR_QUANTIZATION);
    return static_cast<int16_t>(std::clamp(encoded,
                                           static_cast<long>(std::numeric_limits<int16_t>::min()),
                                           static_cast<long>(std::numeric_limits<int16_t>::max())));
}

int16_t encodeShorelineWaterLevel(double value) {
    const long encoded = std::lround(value * SHORELINE_FLOOR_QUANTIZATION);
    return static_cast<int16_t>(std::clamp(encoded,
                                           static_cast<long>(std::numeric_limits<int16_t>::min()),
                                           static_cast<long>(std::numeric_limits<int16_t>::max())));
}

double decodeShorelineWaterLevel(const ShorelineNode& node) {
    return static_cast<double>(node.bankWaterLevel) / SHORELINE_FLOOR_QUANTIZATION;
}

ShorelineNode shorelineNode(const LakeAuthoritySample& authority, WaterBodyId body,
                            double bodyWaterLevel) {
    const double signedDistance = authority.found && authority.identity == body
                                      ? authority.signedDistance
                                      : -SHORELINE_DISTANCE_LIMIT;
    const double bankWaterLevel = authority.found && authority.identity == body
                                      ? std::max(bodyWaterLevel, authority.bankWaterLevel)
                                      : bodyWaterLevel;
    return {
        .signedDistance = encodeShorelineDistance(signedDistance),
        .floor = encodeShorelineFloor(authority.floor),
        .bankWaterLevel = encodeShorelineWaterLevel(bankWaterLevel),
    };
}

double decodeShorelineDistance(const ShorelineNode& node) {
    return static_cast<double>(node.signedDistance) / SHORELINE_DISTANCE_QUANTIZATION;
}

double decodeShorelineFloor(const ShorelineNode& node) {
    return static_cast<double>(node.floor) / SHORELINE_FLOOR_QUANTIZATION;
}

using LakeAuthorityFunction = std::function<LakeAuthoritySample(double, double)>;

std::shared_ptr<const ShorelineContourPage>
buildShorelinePage(ShorelinePageKey key, double waterLevel, bool endorheic,
                   const LakeAuthorityFunction& sampleAuthority) {
    auto page = std::make_shared<ShorelineContourPage>();
    page->key = key;
    page->waterLevel = static_cast<float>(std::round(waterLevel * 1024.0) / 1024.0);
    page->endorheic = endorheic;
    const double originX = static_cast<double>(key.x) * SHORELINE_PAGE_EDGE;
    const double originZ = static_cast<double>(key.z) * SHORELINE_PAGE_EDGE;
    const auto nodeAt = [&](int x, int z) {
        const double worldX = originX + (x - 1) * SHORELINE_COARSE_SPACING;
        const double worldZ = originZ + (z - 1) * SHORELINE_COARSE_SPACING;
        return shorelineNode(sampleAuthority(worldX, worldZ), key.body, waterLevel);
    };
    for (int z = 0; z < SHORELINE_COARSE_EDGE; ++z) {
        for (int x = 0; x < SHORELINE_COARSE_EDGE; ++x) {
            page->coarse[static_cast<size_t>(z * SHORELINE_COARSE_EDGE + x)] = nodeAt(x, z);
        }
    }

    page->refined.reserve(512);
    for (int z = 1; z <= SHORELINE_COARSE_INTERVALS; ++z) {
        for (int x = 1; x <= SHORELINE_COARSE_INTERVALS; ++x) {
            const ShorelineNode& northwest =
                page->coarse[static_cast<size_t>(z * SHORELINE_COARSE_EDGE + x)];
            const ShorelineNode& northeast =
                page->coarse[static_cast<size_t>(z * SHORELINE_COARSE_EDGE + x + 1)];
            const ShorelineNode& southwest =
                page->coarse[static_cast<size_t>((z + 1) * SHORELINE_COARSE_EDGE + x)];
            const ShorelineNode& southeast =
                page->coarse[static_cast<size_t>((z + 1) * SHORELINE_COARSE_EDGE + x + 1)];
            const std::array<double, 4> distances = {
                decodeShorelineDistance(northwest), decodeShorelineDistance(northeast),
                decodeShorelineDistance(southwest), decodeShorelineDistance(southeast)};
            const bool crosses =
                std::ranges::any_of(distances, [](double value) { return value > 0.0; }) &&
                std::ranges::any_of(distances, [](double value) { return value <= 0.0; });
            const double closest = std::ranges::min(
                distances | std::views::transform([](double value) { return std::abs(value); }));
            if (!crosses && closest > 8.0) continue;

            const double worldX = originX + (x - 1) * SHORELINE_COARSE_SPACING;
            const double worldZ = originZ + (z - 1) * SHORELINE_COARSE_SPACING;
            RefinedShorelineCell refined;
            refined.coarseCell = static_cast<uint16_t>(z * SHORELINE_COARSE_EDGE + x);
            constexpr std::array<std::pair<double, double>, 5> OFFSETS = {
                std::pair{2.0, 0.0}, std::pair{0.0, 2.0}, std::pair{2.0, 2.0}, std::pair{4.0, 2.0},
                std::pair{2.0, 4.0}};
            for (size_t node = 0; node < OFFSETS.size(); ++node) {
                refined.midpointNodes[node] = shorelineNode(
                    sampleAuthority(worldX + OFFSETS[node].first, worldZ + OFFSETS[node].second),
                    key.body, waterLevel);
            }
            page->refined.push_back(refined);
        }
    }
    return page;
}

ShorelineContourSample sampleShorelinePage(const ShorelineContourPage& page, double x, double z) {
    const double originX = static_cast<double>(page.key.x) * SHORELINE_PAGE_EDGE;
    const double originZ = static_cast<double>(page.key.z) * SHORELINE_PAGE_EDGE;
    const double localGridX = (x - originX) / SHORELINE_COARSE_SPACING + 1.0;
    const double localGridZ = (z - originZ) / SHORELINE_COARSE_SPACING + 1.0;
    const int cellX =
        std::clamp(static_cast<int>(std::floor(localGridX)), 1, SHORELINE_COARSE_INTERVALS);
    const int cellZ =
        std::clamp(static_cast<int>(std::floor(localGridZ)), 1, SHORELINE_COARSE_INTERVALS);
    double fx = std::clamp(localGridX - cellX, 0.0, 1.0);
    double fz = std::clamp(localGridZ - cellZ, 0.0, 1.0);
    const size_t northwestIndex = static_cast<size_t>(cellZ * SHORELINE_COARSE_EDGE + cellX);
    std::array<ShorelineNode, 4> nodes = {page.coarse[northwestIndex],
                                          page.coarse[northwestIndex + 1],
                                          page.coarse[northwestIndex + SHORELINE_COARSE_EDGE],
                                          page.coarse[northwestIndex + SHORELINE_COARSE_EDGE + 1]};

    const uint16_t coarseCell = static_cast<uint16_t>(northwestIndex);
    const auto refined =
        std::lower_bound(page.refined.begin(), page.refined.end(), coarseCell,
                         [](const RefinedShorelineCell& candidate, uint16_t value) {
                             return candidate.coarseCell < value;
                         });
    if (refined != page.refined.end() && refined->coarseCell == coarseCell) {
        const bool east = fx > 0.5;
        const bool south = fz > 0.5;
        if (!east && !south) {
            nodes = {nodes[0], refined->midpointNodes[0], refined->midpointNodes[1],
                     refined->midpointNodes[2]};
        } else if (east && !south) {
            nodes = {refined->midpointNodes[0], nodes[1], refined->midpointNodes[2],
                     refined->midpointNodes[3]};
        } else if (!east && south) {
            nodes = {refined->midpointNodes[1], refined->midpointNodes[2], nodes[2],
                     refined->midpointNodes[4]};
        } else {
            nodes = {refined->midpointNodes[2], refined->midpointNodes[3],
                     refined->midpointNodes[4], nodes[3]};
        }
        fx = east ? fx * 2.0 - 1.0 : fx * 2.0;
        fz = south ? fz * 2.0 - 1.0 : fz * 2.0;
    }

    const auto interpolate = [&](auto decode) {
        const double north = std::lerp(decode(nodes[0]), decode(nodes[1]), fx);
        const double south = std::lerp(decode(nodes[2]), decode(nodes[3]), fx);
        return std::lerp(north, south, fz);
    };
    return {
        .signedDistance = interpolate(decodeShorelineDistance),
        .floor = interpolate(decodeShorelineFloor),
        .waterLevel = page.waterLevel,
        .bankWaterLevel = interpolate(decodeShorelineWaterLevel),
        .identity = page.key.body,
        .endorheic = page.endorheic,
        .valid = true,
    };
}

using SparseShorelineNodes =
    std::unordered_map<ShorelinePageKey, ShorelineNode, ShorelinePageKeyHash>;

ShorelineContourSample sampleSparseShoreline(WaterBodyId body, double waterLevel, bool endorheic,
                                             double x, double z,
                                             const LakeAuthorityFunction& sampleAuthority,
                                             SparseShorelineNodes& retainedNodes) {
    const int64_t pageX = static_cast<int64_t>(std::floor(x / SHORELINE_PAGE_EDGE));
    const int64_t pageZ = static_cast<int64_t>(std::floor(z / SHORELINE_PAGE_EDGE));
    const double originX = static_cast<double>(pageX) * SHORELINE_PAGE_EDGE;
    const double originZ = static_cast<double>(pageZ) * SHORELINE_PAGE_EDGE;
    const double localGridX = (x - originX) / SHORELINE_COARSE_SPACING + 1.0;
    const double localGridZ = (z - originZ) / SHORELINE_COARSE_SPACING + 1.0;
    const int cellX =
        std::clamp(static_cast<int>(std::floor(localGridX)), 1, SHORELINE_COARSE_INTERVALS);
    const int cellZ =
        std::clamp(static_cast<int>(std::floor(localGridZ)), 1, SHORELINE_COARSE_INTERVALS);
    double fx = std::clamp(localGridX - cellX, 0.0, 1.0);
    double fz = std::clamp(localGridZ - cellZ, 0.0, 1.0);
    const auto nodeAt = [&](double worldX, double worldZ) {
        const ShorelinePageKey nodeKey{body, static_cast<int64_t>(std::llround(worldX)),
                                       static_cast<int64_t>(std::llround(worldZ))};
        auto [node, inserted] = retainedNodes.try_emplace(nodeKey);
        if (inserted) {
            node->second = shorelineNode(sampleAuthority(worldX, worldZ), body, waterLevel);
        }
        return node->second;
    };
    const double cellWorldX = originX + (cellX - 1) * SHORELINE_COARSE_SPACING;
    const double cellWorldZ = originZ + (cellZ - 1) * SHORELINE_COARSE_SPACING;
    std::array<ShorelineNode, 4> nodes = {
        nodeAt(cellWorldX, cellWorldZ),
        nodeAt(cellWorldX + SHORELINE_COARSE_SPACING, cellWorldZ),
        nodeAt(cellWorldX, cellWorldZ + SHORELINE_COARSE_SPACING),
        nodeAt(cellWorldX + SHORELINE_COARSE_SPACING, cellWorldZ + SHORELINE_COARSE_SPACING),
    };
    const std::array<double, 4> distances = {
        decodeShorelineDistance(nodes[0]), decodeShorelineDistance(nodes[1]),
        decodeShorelineDistance(nodes[2]), decodeShorelineDistance(nodes[3])};
    const bool crosses = std::ranges::any_of(distances, [](double value) { return value > 0.0; }) &&
                         std::ranges::any_of(distances, [](double value) { return value <= 0.0; });
    const double closest = std::ranges::min(
        distances | std::views::transform([](double value) { return std::abs(value); }));
    if (crosses || closest <= 8.0) {
        constexpr double HALF_SPACING = SHORELINE_COARSE_SPACING * 0.5;
        const std::array<ShorelineNode, 5> midpointNodes = {
            nodeAt(cellWorldX + HALF_SPACING, cellWorldZ),
            nodeAt(cellWorldX, cellWorldZ + HALF_SPACING),
            nodeAt(cellWorldX + HALF_SPACING, cellWorldZ + HALF_SPACING),
            nodeAt(cellWorldX + SHORELINE_COARSE_SPACING, cellWorldZ + HALF_SPACING),
            nodeAt(cellWorldX + HALF_SPACING, cellWorldZ + SHORELINE_COARSE_SPACING),
        };
        const bool east = fx > 0.5;
        const bool south = fz > 0.5;
        if (!east && !south) {
            nodes = {nodes[0], midpointNodes[0], midpointNodes[1], midpointNodes[2]};
        } else if (east && !south) {
            nodes = {midpointNodes[0], nodes[1], midpointNodes[2], midpointNodes[3]};
        } else if (!east && south) {
            nodes = {midpointNodes[1], midpointNodes[2], nodes[2], midpointNodes[4]};
        } else {
            nodes = {midpointNodes[2], midpointNodes[3], midpointNodes[4], nodes[3]};
        }
        fx = east ? fx * 2.0 - 1.0 : fx * 2.0;
        fz = south ? fz * 2.0 - 1.0 : fz * 2.0;
    }

    const auto interpolate = [&](auto decode) {
        const double north = std::lerp(decode(nodes[0]), decode(nodes[1]), fx);
        const double south = std::lerp(decode(nodes[2]), decode(nodes[3]), fx);
        return std::lerp(north, south, fz);
    };
    return {
        .signedDistance = interpolate(decodeShorelineDistance),
        .floor = interpolate(decodeShorelineFloor),
        .waterLevel = static_cast<float>(std::round(waterLevel * 1024.0) / 1024.0),
        .bankWaterLevel = interpolate(decodeShorelineWaterLevel),
        .identity = body,
        .endorheic = endorheic,
        .valid = true,
    };
}

void quantizeGeneratedRiverSurface(BasinSample& sample, bool allowLakeBackwater) {
    if (!sample.river || sample.ocean || sample.lake || sample.waterfall ||
        !std::isfinite(sample.waterSurface)) {
        return;
    }
    const bool lakeControlledBackwater =
        sample.generatedFluidLevel == 0 && sample.shoreWaterSurface > 0.0 &&
        sample.lakeShoreDistance > -LAKE_SHORE_PERTURBATION_BAND &&
        std::abs(sample.waterSurface - sample.shoreWaterSurface) < 2.5;
    if (allowLakeBackwater && lakeControlledBackwater) {
        if (sample.lakeShoreDistance >= -8.0) {
            sample.waterSurface = sample.shoreWaterSurface;
        } else {
            const double backwaterFloor =
                sample.shoreWaterSurface - std::max(0.0, -sample.lakeShoreDistance) * 0.125;
            sample.waterSurface = std::max(sample.waterSurface, backwaterFloor);
        }
    }
    if (sample.generatedFluidLevel == 0) {
        if (sample.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL) {
            sample.waterSurface = std::ceil(sample.waterSurface - 1.0e-6);
            sample.surfaceElevation =
                std::min(sample.surfaceElevation, sample.waterSurface - 0.125);
            sample.channelDepth = std::max(0.125, sample.waterSurface - sample.surfaceElevation);
            return;
        }
        // Quantize the continuous routed profile to Java's eighth-block top
        // states. Integer planes remain source water, while the seven
        // intermediate heights become flowing levels. Rounding every routed
        // column up to a full source block manufactured a one-block wet wall
        // at every otherwise gentle integer contour.
        // Select the closest representable fluid height. Always rounding up
        // turns tiny reconstruction residuals into a two-level cardinal step
        // and systematically raises a routed profile above its solved stage.
        sample.waterSurface = std::round(sample.waterSurface * 8.0) / 8.0;
        const double wholePlane = std::floor(sample.waterSurface + 1.0e-6);
        const int eighths = static_cast<int>(
            std::lround(std::clamp(sample.waterSurface - wholePlane, 0.0, 0.875) * 8.0));
        if (eighths > 0) {
            sample.generatedFluidLevel = static_cast<uint8_t>(8 - eighths);
        }
    } else {
        sample.waterSurface = std::round(sample.waterSurface * 8.0) / 8.0;
        const FluidState state = FluidState::flowing(sample.generatedFluidLevel);
        const double height = fluidSurfaceHeight(state);
        const double topY = std::round(sample.waterSurface - height);
        const double emittedSurface = topY + height;
        if (std::abs(emittedSurface - sample.waterSurface) > 0.125001) {
            sample.generatedFluidLevel = 0;
            sample.waterSurface = std::ceil(sample.waterSurface - 1.0e-6);
        }
    }
    sample.surfaceElevation = std::min(sample.surfaceElevation, sample.waterSurface - 0.125);
    sample.channelDepth = std::max(0.125, sample.waterSurface - sample.surfaceElevation);
}

BasinSample sampleSolution(const BasinSolution& solution, double x, double z,
                           const ShorelineContourSample* shoreline = nullptr) {
    const double gridX = (x - solution.originX) / BASIN_RASTER_SPACING + RASTER_APRON;
    const double gridZ = (z - solution.originZ) / BASIN_RASTER_SPACING + RASTER_APRON;
    BasinSample result;
    result.flowX = reconstruct(solution.flowX, gridX, gridZ);
    result.flowZ = reconstruct(solution.flowZ, gridX, gridZ);
    const double flowMagnitude = std::hypot(result.flowX, result.flowZ);
    if (flowMagnitude > 1.0e-9) {
        result.flowX /= flowMagnitude;
        result.flowZ /= flowMagnitude;
    } else {
        result.flowX = 1.0;
        result.flowZ = 0.0;
    }
    result.surfaceElevation = reconstruct(solution.surface, gridX, gridZ);
    result.waterSurface = reconstruct(solution.waterSurface, gridX, gridZ);
    result.discharge = reconstruct(solution.discharge, gridX, gridZ);
    result.sediment = reconstruct(solution.sediment, gridX, gridZ);
    result.channelDistance = reconstructChannelDistance(solution, gridX, gridZ);
    result.channelWidth = reconstruct(solution.channelWidth, gridX, gridZ);
    result.channelDepth = reconstruct(solution.channelDepth, gridX, gridZ);
    result.channelGradient = reconstruct(solution.channelGradient, gridX, gridZ);
    result.erosionDepth = reconstruct(solution.erosionDepth, gridX, gridZ);
    result.lakeDepth = reconstruct(solution.lakeDepth, gridX, gridZ);
    result.lakeShoreDistance = reconstruct(solution.lakeShoreDistance, gridX, gridZ);
    result.shoreWaterSurface = dominantPositiveLevel(solution, gridX, gridZ);
    result.streamOrder = reconstructedStreamOrder(solution, gridX, gridZ);
    const double oceanMembership = flagWeight(solution, gridX, gridZ, CELL_OCEAN);
    const bool categoricalOcean = oceanMembership >= 0.5;
    const LakeBodySample lakeBody = sampleCandidateLakeBody(solution, gridX, gridZ);
    const WaterBodyId selectedLakeBody = shoreline != nullptr && shoreline->valid
                                             ? shoreline->identity
                                             : (lakeBody.found ? lakeBody.identity : NO_WATER_BODY);
    const bool protectedLakeFloor =
        (shoreline != nullptr && shoreline->valid && shoreline->signedDistance > 0.0) ||
        (lakeBody.depth.weight > 0.0 && result.lakeShoreDistance > 0.0);
    const int closestCell = nearestIndex(gridX, gridZ);
    result.outlet = static_cast<BasinOutlet>(solution.outletTypes[closestCell]);
    if (result.outlet == BasinOutlet::SHARED_PORTAL || result.outlet == BasinOutlet::ENDORHEIC) {
        result.outletX = solution.namedOutletX;
        result.outletZ = solution.namedOutletZ;
    } else if (result.outlet == BasinOutlet::OCEAN) {
        result.outletX = x;
        result.outletZ = z;
    }

    double rasterChannelWater = result.waterSurface;
    double rasterChannelSurface = result.surfaceElevation;
    double rasterChannelDischarge = result.discharge;
    // Distance and width are continuous reconstructions of the routed
    // centerline. Stream order is categorical diagnostic metadata and can
    // change at a raster control boundary, so it must not punch dry or ocean
    // squares through an otherwise continuous channel core.
    bool rasterChannelCore = result.channelWidth > 0.0 &&
                             result.discharge >= MIN_CHANNEL_DISCHARGE &&
                             result.channelDistance <= result.channelWidth * 0.55;
    if (rasterChannelCore && result.streamOrder == 0) {
        result.streamOrder = static_cast<uint8_t>(channelGuideOrder(result.discharge));
    }
    int32_t selectedRasterSource = -1;
    uint64_t selectedRasterOwnerId = 0;
    double rasterBankInfluence = 0.0;
    double rasterBankWater = 0.0;
    double lowlandChannelFloor = std::numeric_limits<double>::infinity();
    bool elevatedRasterChannelCore = rasterChannelCore && rasterChannelWater > SEA_LEVEL + 1.0e-4;
    const bool possibleRasterChannel =
        result.channelWidth > 0.0 && result.discharge >= MIN_CHANNEL_DISCHARGE;
    if ((rasterChannelCore || categoricalOcean || possibleRasterChannel) && !protectedLakeFloor) {
        struct RasterEdgeAuthority {
            int32_t source = -1;
            int32_t edgeIndex = -1;
            uint64_t ownerId = 0;
            double stage = 0.0;
            double distance = std::numeric_limits<double>::max();
            double width = 0.0;
            double flowX = 1.0;
            double flowZ = 0.0;
            double discharge = -1.0;
            double stageError = std::numeric_limits<double>::max();
        } authority;
        const int centerX = static_cast<int>(std::floor(gridX));
        const int centerZ = static_cast<int>(std::floor(gridZ));
        for (int offsetZ = -3; offsetZ <= 3; ++offsetZ) {
            for (int offsetX = -3; offsetX <= 3; ++offsetX) {
                const int sourceX = centerX + offsetX;
                const int sourceZ = centerZ + offsetZ;
                if (!inRaster(sourceX, sourceZ)) continue;
                const int source = indexOf(sourceX, sourceZ);
                if (solution.streamOrder[source] == 0 || solution.channelWidth[source] <= 0.0F ||
                    solution.channelDistance[source] > 1.0e-4F ||
                    (solution.flags[source] & (CELL_OCEAN | CELL_LAKE)) != 0) {
                    continue;
                }
                const double startX =
                    solution.originX + (sourceX - RASTER_APRON) * BASIN_RASTER_SPACING;
                const double startZ =
                    solution.originZ + (sourceZ - RASTER_APRON) * BASIN_RASTER_SPACING;
                double crossSectionWidth = solution.channelWidth[source];
                for (int neighborDirection = 0; neighborDirection < 8; ++neighborDirection) {
                    const int neighborX = sourceX + DX[neighborDirection];
                    const int neighborZ = sourceZ + DZ[neighborDirection];
                    if (!inRaster(neighborX, neighborZ)) continue;
                    const int neighbor = indexOf(neighborX, neighborZ);
                    if (solution.streamOrder[neighbor] == 0 ||
                        solution.channelDistance[neighbor] > 1.0e-4F) {
                        continue;
                    }
                    crossSectionWidth = std::max(
                        crossSectionWidth, static_cast<double>(solution.channelWidth[neighbor]));
                }
                for (int direction = 0; direction < 8; ++direction) {
                    const int receiverX = sourceX + DX[direction];
                    const int receiverZ = sourceZ + DZ[direction];
                    if (!inRaster(receiverX, receiverZ)) continue;
                    const int receiver = indexOf(receiverX, receiverZ);
                    if (source >= static_cast<int>(solution.rasterReceivers.size()) ||
                        solution.rasterReceivers[static_cast<size_t>(source)] != receiver) {
                        continue;
                    }
                    const bool receivingWater =
                        (solution.flags[receiver] & (CELL_OCEAN | CELL_LAKE)) != 0;
                    const bool activeReceiver = solution.streamOrder[receiver] > 0 &&
                                                solution.channelWidth[receiver] > 0.0F &&
                                                solution.channelDistance[receiver] <= 1.0e-4F;
                    if (!receivingWater && !activeReceiver) continue;
                    const double inverseLength = 1.0 / STEP_LENGTH[direction];
                    const double startWater = solution.waterSurface[source];
                    const double endWater = std::min(
                        startWater, (solution.flags[receiver] & CELL_OCEAN) != 0
                                        ? static_cast<double>(SEA_LEVEL)
                                        : static_cast<double>(solution.waterSurface[receiver]));
                    const double endX =
                        solution.originX + (receiverX - RASTER_APRON) * BASIN_RASTER_SPACING;
                    const double endZ =
                        solution.originZ + (receiverZ - RASTER_APRON) * BASIN_RASTER_SPACING;
                    double along = 0.0;
                    const double distance =
                        distanceToSegment(x, z, startX, startZ, endX, endZ, along);
                    const double halfWidth =
                        std::clamp(std::max(crossSectionWidth,
                                            static_cast<double>(solution.channelWidth[receiver])) *
                                       0.55,
                                   2.0, 24.0);
                    const double stage =
                        std::lerp(startWater, endWater,
                                  rasterWaterProfileWeight(solution, source, receiver, along));
                    if (stage <= SEA_LEVEL + 1.0 + 1.0e-6) {
                        constexpr double SUBMERGED_BANK_SLOPE = 0.75;
                        const double fullDepth = solution.channelDepth[source];
                        const double submergedOutsideCore = std::max(0.0, distance - halfWidth);
                        const double continuedDepth =
                            std::max(0.0, fullDepth - submergedOutsideCore * SUBMERGED_BANK_SLOPE);
                        if (continuedDepth > 0.0) {
                            lowlandChannelFloor =
                                std::min(lowlandChannelFloor, stage - continuedDepth);
                        }
                    }
                    const double discharge = solution.discharge[source];
                    const double stageError = std::abs(stage - rasterChannelWater);
                    const uint64_t ownerId = solution.rasterDrainageOwnerId;
                    const int32_t edgeIndex = source * 8 + direction;
                    const double outsideCore = std::max(0.0, distance - halfWidth);
                    // A routed cross section needs solid freeboard outside its
                    // wet core. Without this bank, a later broad floodplain
                    // deformation can expose the side of a deep river as a
                    // freestanding wall of water. Restrict the bank to the
                    // lateral body of the edge so routed endpoints remain open
                    // to their receiver, and warp its taper to avoid a straight
                    // storage-grid contour.
                    constexpr double CHANNEL_BANK_WIDTH = 12.0;
                    constexpr double CHANNEL_BANK_FREEBOARD = 0.5;
                    const double segmentX = endX - startX;
                    const double segmentZ = endZ - startZ;
                    const double segmentLengthSquared = segmentX * segmentX + segmentZ * segmentZ;
                    const double unboundedAlong =
                        segmentLengthSquared > 1.0e-9
                            ? ((x - startX) * segmentX + (z - startZ) * segmentZ) /
                                  segmentLengthSquared
                            : 0.0;
                    const double upstreamBankExtent =
                        segmentLengthSquared > 1.0e-9
                            ? CHANNEL_BANK_WIDTH / std::sqrt(segmentLengthSquared)
                            : 0.0;
                    if (outsideCore > 0.125 && unboundedAlong >= -upstreamBankExtent &&
                        unboundedAlong <= 1.02) {
                        const uint64_t bankIdentity =
                            combine64(ownerId, static_cast<uint64_t>(edgeIndex));
                        const double warpedOutside = std::max(
                            0.0, outsideCore + organicShoreNoise(bankIdentity, x, z) * 1.5);
                        const double influence =
                            1.0 - smoothstep(0.0, CHANNEL_BANK_WIDTH, warpedOutside);
                        if (influence > rasterBankInfluence + 1.0e-6 ||
                            (std::abs(influence - rasterBankInfluence) <= 1.0e-6 &&
                             stage + CHANNEL_BANK_FREEBOARD > rasterBankWater)) {
                            rasterBankInfluence = influence;
                            rasterBankWater = stage + CHANNEL_BANK_FREEBOARD;
                        }
                    }
                    if (distance > halfWidth + 0.125) continue;
                    const bool equalDistance = std::abs(distance - authority.distance) <= 1.0e-6;
                    const bool equalError = std::abs(stageError - authority.stageError) <= 1.0e-6;
                    const bool equalDischarge = std::abs(discharge - authority.discharge) <= 1.0e-6;
                    const bool better =
                        authority.source < 0 || distance < authority.distance - 1.0e-6 ||
                        (equalDistance && stageError < authority.stageError - 1.0e-6) ||
                        (equalDistance && equalError && discharge > authority.discharge + 1.0e-6) ||
                        (equalDistance && equalError && equalDischarge &&
                         edgeIndex < authority.edgeIndex);
                    if (!better) {
                        continue;
                    }
                    authority = {
                        .source = source,
                        .edgeIndex = edgeIndex,
                        .ownerId = ownerId,
                        .stage = stage,
                        .distance = distance,
                        .width = halfWidth / 0.55,
                        .flowX = (endX - startX) * inverseLength / BASIN_RASTER_SPACING,
                        .flowZ = (endZ - startZ) * inverseLength / BASIN_RASTER_SPACING,
                        .discharge = discharge,
                        .stageError = stageError,
                    };
                }
            }
        }
        if (authority.source >= 0) {
            const double hydraulicStage = authority.stage;
            const double fullChannelDepth =
                solution.channelDepth[static_cast<size_t>(authority.source)];
            const double halfWidth = authority.width * 0.55;
            const bool submergedMouth = hydraulicStage <= SEA_LEVEL + 1.0 + 1.0e-6;
            const double emittedChannelDepth =
                submergedMouth ? std::min(fullChannelDepth,
                                          std::max(0.125, (halfWidth - authority.distance) * 0.75))
                               : fullChannelDepth;
            selectedRasterSource = authority.source;
            selectedRasterOwnerId = authority.ownerId;
            rasterChannelWater = hydraulicStage;
            rasterChannelDischarge = authority.discharge;
            result.waterSurface = hydraulicStage;
            result.discharge = authority.discharge;
            result.sediment = solution.sediment[static_cast<size_t>(authority.source)];
            result.channelDistance = authority.distance;
            result.channelWidth = authority.width;
            result.channelDepth = fullChannelDepth;
            result.channelGradient =
                solution.channelGradient[static_cast<size_t>(authority.source)];
            result.flowX = authority.flowX;
            result.flowZ = authority.flowZ;
            result.streamOrder = solution.streamOrder[static_cast<size_t>(authority.source)];
            result.surfaceElevation =
                std::min(result.surfaceElevation, result.waterSurface - emittedChannelDepth);
            rasterChannelSurface = result.surfaceElevation;
            rasterChannelCore = true;
            elevatedRasterChannelCore = hydraulicStage > SEA_LEVEL + 1.0e-4;
        }
    }

    const ChannelGuide* analyticalGuide = nullptr;
    GuideProjection analyticalProjection;
    bool analyticalGuideCore = false;
    bool elevatedAnalyticalGuideCore = false;
    double analyticalGuideNormalizedDistance = std::numeric_limits<double>::max();
    for (const ChannelGuide& guide : solution.guides) {
        constexpr double MAX_GUIDE_INFLUENCE = 176.0;
        if (x < guide.minX - MAX_GUIDE_INFLUENCE || x > guide.maxX + MAX_GUIDE_INFLUENCE ||
            z < guide.minZ - MAX_GUIDE_INFLUENCE || z > guide.maxZ + MAX_GUIDE_INFLUENCE) {
            continue;
        }
        const GuideProjection projection = projectGuide(guide, x, z);
        const double widthEnvelope = std::pow(std::sin(std::numbers::pi * projection.along), 2.0);
        const double widthScale =
            1.0 + guide.widthVariation * widthEnvelope *
                      std::sin(4.0 * std::numbers::pi * projection.along + guide.widthPhase);
        const int order = channelGuideOrder(guide.discharge);
        const double width = std::clamp(
            (3.0 + std::sqrt(guide.discharge) * 0.22 + order * 1.2) * widthScale, 4.0, 42.0);
        const double floodplainWidth = width * (2.4 + order * 0.28);
        const double influenceWidth = floodplainWidth * (guide.backwater ? 2.0 : 1.0);
        if (projection.distance > influenceWidth) continue;
        const bool core = projection.distance <= width * 0.55;
        const double normalizedDistance = projection.distance / std::max(1.0, width);
        const bool strongerCore = core && !analyticalGuideCore;
        const bool strongerMainstem = core == analyticalGuideCore && core &&
                                      analyticalGuide != nullptr &&
                                      guide.discharge > analyticalGuide->discharge + 1.0e-4;
        const bool closerPeer =
            core == analyticalGuideCore &&
            (analyticalGuide == nullptr ||
             (std::abs(guide.discharge - analyticalGuide->discharge) <= 1.0e-4 &&
              normalizedDistance < analyticalGuideNormalizedDistance));
        if (analyticalGuide == nullptr || strongerCore || strongerMainstem || closerPeer) {
            analyticalGuide = &guide;
            analyticalProjection = projection;
            analyticalGuideCore = core;
            analyticalGuideNormalizedDistance = normalizedDistance;
        }
    }
    const auto guideWaterAt = [](const ChannelGuide& guide, const GuideProjection& projection) {
        if (guide.backwater) return guide.portalWater;
        return std::lerp(guide.startWater, guide.endWater, quinticWeight(projection.along));
    };
    if (analyticalGuide != nullptr && analyticalGuideCore && rasterChannelCore &&
        selectedRasterOwnerId != 0 &&
        std::abs(guideWaterAt(*analyticalGuide, analyticalProjection) - rasterChannelWater) >=
            2.5) {
        // A shared guide may overlap an unrelated raster reach in projection.
        // It cannot replace that reach with a radically different stage at
        // the edge of its strip. Keep the locally owned raster route here;
        // true guide junctions already share compatible portal levels, while
        // large routed drops are emitted by explicit fall ownership.
        analyticalGuide = nullptr;
        analyticalGuideCore = false;
    }
    if (analyticalGuide != nullptr && !protectedLakeFloor) {
        const double widthEnvelope =
            std::pow(std::sin(std::numbers::pi * analyticalProjection.along), 2.0);
        const double widthScale =
            1.0 + analyticalGuide->widthVariation * widthEnvelope *
                      std::sin(4.0 * std::numbers::pi * analyticalProjection.along +
                               analyticalGuide->widthPhase);
        const int guideOrder = channelGuideOrder(analyticalGuide->discharge);
        const double guideWidth = std::clamp(
            (3.0 + std::sqrt(analyticalGuide->discharge) * 0.22 + guideOrder * 1.2) * widthScale,
            4.0, 42.0);
        const double floodplainWidth = guideWidth * (2.4 + guideOrder * 0.28);
        const double influenceWidth = floodplainWidth * (analyticalGuide->backwater ? 2.0 : 1.0);
        const double guideWater = guideWaterAt(*analyticalGuide, analyticalProjection);
        const double guideInfluence =
            1.0 - smoothstep(guideWidth * 0.45, influenceWidth, analyticalProjection.distance);
        const bool sharedBackwater = analyticalGuide->backwater && guideInfluence > 1.0e-6;
        const bool compatibleSharedBackwater = sharedBackwater && !rasterChannelCore;
        const bool confluenceBackwater =
            !analyticalGuideCore && rasterChannelCore &&
            analyticalGuide->discharge > rasterChannelDischarge + 1.0e-4 &&
            guideWater > rasterChannelWater + 1.0e-4;
        if (confluenceBackwater) {
            // When a smaller raster tributary reaches the floodplain of a
            // higher mainstem, the receiving stage propagates upstream until
            // the tributary profile catches it. Keeping the lower raster
            // stage to the guide-core edge created a vertical wet face at the
            // confluence. Preserve tributary geometry and flow direction, but
            // let the stronger mainstem own this standing backwater level.
            result.waterSurface = guideWater;
            result.surfaceElevation =
                std::min(result.surfaceElevation, result.waterSurface - result.channelDepth);
        }
        if (analyticalProjection.distance <= influenceWidth &&
            (analyticalGuideCore || compatibleSharedBackwater || result.channelWidth <= 0.0 ||
             analyticalProjection.distance < result.channelDistance)) {
            // Every half edge owns a continuous junction-to-portal profile.
            // Incoming and outgoing curves share the same junction level,
            // while both catchments share the immutable portal level.
            const double guideDepth = channelGuideDepth(analyticalGuide->discharge);
            result.surfaceElevation =
                std::min(result.surfaceElevation,
                         result.surfaceElevation +
                             std::min(0.0, guideWater - guideDepth - result.surfaceElevation) *
                                 guideInfluence);
            result.flowX = analyticalProjection.tangentX;
            result.flowZ = analyticalProjection.tangentZ;
            result.discharge = std::max(result.discharge, analyticalGuide->discharge);
            result.sediment =
                std::max(result.sediment,
                         analyticalGuide->discharge * (0.018 + analyticalGuide->gradient * 0.9));
            result.channelDistance = analyticalProjection.distance;
            result.channelWidth = guideWidth;
            result.channelDepth = guideDepth;
            result.channelGradient = analyticalGuide->gradient;
            result.streamOrder = static_cast<uint8_t>(guideOrder);
            if (analyticalGuide->backwater) {
                // A backwater is a standing floodplain surface controlled by
                // the shared guide profile. Retaining an unrelated nearest
                // raster channel level across this broad band produced
                // meter-scale one-block jumps as ownership switched between
                // channel controls.
                result.waterSurface = analyticalGuide->portalWater;
            }
            if (analyticalProjection.distance <= guideWidth * 0.55) {
                // The immutable guide owns the routed core's stage. Raster
                // erosion still owns the bed, while globally anchored rapid
                // patches below represent each integer source-plane crossing.
                result.waterSurface = guideWater;
                elevatedAnalyticalGuideCore = guideWater > SEA_LEVEL + 1.0e-4;
                result.surfaceElevation =
                    std::min(result.surfaceElevation, result.waterSurface - guideDepth);
                result.erosionDepth = std::max(result.erosionDepth, guideDepth);
            }
        }
    }

    if (std::isfinite(lowlandChannelFloor)) {
        // This is an independent physical floor cap, not an input to guide
        // deformation. Applying it before the continuous guide blend let the
        // bounded raster search edge amplify a harmless higher cap into a
        // straight underwater terrace.
        result.surfaceElevation = std::min(result.surfaceElevation, lowlandChannelFloor);
    }
    const double unbankedChannelSurface = result.surfaceElevation;
    if (rasterBankInfluence > 1.0e-6 && !protectedLakeFloor && !analyticalGuideCore) {
        result.surfaceElevation =
            std::lerp(result.surfaceElevation, std::max(result.surfaceElevation, rasterBankWater),
                      rasterBankInfluence);
        if (!rasterChannelCore) {
            // Preserve the dry lateral ownership through exact density
            // emission. Without this flag a sub-block detail trough can be
            // reclassified as ocean after the routed bank has already been
            // solved, creating an ownerless sea cell beside an elevated
            // channel.
            result.channelBank = true;
            result.channelBankSubstrate = unbankedChannelSurface;
            result.channelBankSubstrateValid = true;
        }
    }

    if (shoreline != nullptr && shoreline->valid) {
        result.lakeShoreDistance = shoreline->signedDistance;
        result.shoreWaterSurface = std::max(shoreline->waterLevel, shoreline->bankWaterLevel);
    } else if (lakeBody.found) {
        const double bandWeight =
            1.0 - smoothstep(8.0, LAKE_SHORE_PERTURBATION_BAND, std::abs(result.lakeShoreDistance));
        result.lakeShoreDistance += organicShoreNoise(lakeBody.identity, x, z) *
                                    LAKE_SHORE_PERTURBATION_AMPLITUDE * bandWeight *
                                    shorelineOutletProtection(solution, lakeBody, x, z);
        result.shoreWaterSurface = lakeBody.waterLevel;
    }
    result.ocean = categoricalOcean && result.surfaceElevation < SEA_LEVEL + 0.5;
    if (elevatedAnalyticalGuideCore || elevatedRasterChannelCore) {
        // A continuous mouth guide remains a river until its canonical stage
        // actually reaches sea level. The coarse categorical ocean mask may
        // overlap the incised core, but it must not truncate that profile and
        // expose a multi-block wall against the next river cell.
        result.ocean = false;
    }
    const bool hasLakeAuthority = (shoreline != nullptr && shoreline->valid) || lakeBody.found;
    if (!result.ocean && hasLakeAuthority && result.lakeShoreDistance > 0.0) {
        if (shoreline != nullptr && shoreline->valid) {
            // A contour page owns the complete body geometry, including its
            // floor. Mixing in either catchment's independently eroded floor
            // would reintroduce a depth step at the storage face.
            result.surfaceElevation = std::min(shoreline->floor, shoreline->waterLevel - 0.125);
            result.lakeDepth = shoreline->waterLevel - result.surfaceElevation;
        } else {
            result.lakeDepth = lakeBody.depth.value;
        }
        result.lake = result.lakeDepth > 0.05;
        if (result.lake) {
            result.waterSurface = shoreline != nullptr && shoreline->valid ? shoreline->waterLevel
                                                                           : lakeBody.waterLevel;
            result.surfaceElevation = result.waterSurface - result.lakeDepth;
            result.waterBodyId =
                shoreline != nullptr && shoreline->valid ? shoreline->identity : lakeBody.identity;
        }
    } else {
        result.lake = false;
        result.lakeDepth = 0.0;
    }
    if (result.lake && rasterChannelCore && rasterChannelWater > result.waterSurface + 0.05 &&
        rasterChannelWater < result.waterSurface + 2.5) {
        // A shallow incoming reach remains a routed channel through the
        // reconstructed shoreline until its descending stage meets the flat
        // lake. Cutting ownership at the first positive contour sample left
        // an arbitrary one-block shelf above the standing source volume.
        result.waterSurface = rasterChannelWater;
        result.surfaceElevation =
            std::min(rasterChannelSurface, result.waterSurface - result.channelDepth);
        result.lake = false;
        result.endorheic = false;
        result.lakeDepth = 0.0;
        result.waterBodyId = NO_WATER_BODY;
        result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
        result.river = true;
    }
    const bool routedChannel = result.streamOrder > 0 && result.channelWidth > 0.0 &&
                               result.channelDistance <= result.channelWidth * 0.55;
    const bool canonicalRoutedChannel =
        routedChannel && (selectedRasterOwnerId != 0 || analyticalGuideCore);
    if (!result.lake && result.surfaceElevation < SEA_LEVEL && !canonicalRoutedChannel) {
        // Final continuous relief, not a raster ownership mask, defines the
        // coastline. A submerged non-channel column is source-filled ocean;
        // this also prevents single-cell categorical holes along irregular
        // shores.
        result.ocean = true;
    }
    result.endorheic =
        result.lake &&
        (shoreline != nullptr && shoreline->valid ? shoreline->endorheic : lakeBody.endorheic);
    bool outletConnector = false;
    bool ownedReceivingChannel = false;
    double closestConnectorDistance = std::numeric_limits<double>::max();
    bool corridorBackwater = false;
    double corridorBackwaterDistance = std::numeric_limits<double>::max();
    double corridorBackwaterSurface = 0.0;
    double corridorBackwaterWidth = 0.0;
    double corridorBackwaterFlowX = 0.0;
    double corridorBackwaterFlowZ = 0.0;
    uint64_t corridorBackwaterOwner = 0;
    for (size_t corridorIndex = 0; corridorIndex < solution.outletCorridors.size();
         ++corridorIndex) {
        const LakeOutletCorridor& corridor = solution.outletCorridors[corridorIndex];
        // Exterior contour authority may switch to a neighboring lake one
        // block outside its shore. Outlet corridors remain globally owned
        // route geometry there; only a sample actually inside another lake
        // may reject an unrelated corridor.
        if (result.lake && selectedLakeBody != NO_WATER_BODY && corridor.body != selectedLakeBody) {
            continue;
        }
        for (size_t segment = 0; segment + 1 < corridor.receivingRoute.size(); ++segment) {
            double along = 0.0;
            const double distance = distanceToSegment(
                x, z, corridor.receivingRoute[segment].x, corridor.receivingRoute[segment].z,
                corridor.receivingRoute[segment + 1].x, corridor.receivingRoute[segment + 1].z,
                along);
            ownedReceivingChannel = ownedReceivingChannel || distance <= corridor.halfWidth;
        }
        for (size_t segment = 0; segment + 1 < corridor.points.size(); ++segment) {
            double along = 0.0;
            const double distance = distanceToSegment(
                x, z, corridor.points[segment].x, corridor.points[segment].z,
                corridor.points[segment + 1].x, corridor.points[segment + 1].z, along);
            const double segmentTop = corridor.waterSurface[segment];
            const double segmentBottom = corridor.waterSurface[segment + 1];
            const double segmentX = corridor.points[segment + 1].x - corridor.points[segment].x;
            const double segmentZ = corridor.points[segment + 1].z - corridor.points[segment].z;
            const double segmentLength = std::hypot(segmentX, segmentZ);
            const bool steepFall = segmentTop >= segmentBottom + 2.5;
            const bool receivingBackwater =
                segment + 2 == corridor.points.size() && segmentTop < segmentBottom + 2.5;
            // A fall remains at its upstream level until the narrow vertical
            // curtain at the receiving control. Interpolating a large drop
            // across one raster segment produced a diagonal water sheet and
            // an unsupported block-scale jump where the connector met the
            // standing lake. Gradual rapids still use the smooth profile.
            const double connectorWater = steepFall ? segmentTop
                                          : receivingBackwater
                                              ? segmentBottom
                                              : std::lerp(segmentTop, segmentBottom, along);
            const double distanceOutside = distance - corridor.halfWidth;
            if (distanceOutside > 0.0 && distanceOutside <= 20.0 &&
                distanceOutside < corridorBackwaterDistance) {
                corridorBackwater = true;
                corridorBackwaterDistance = distanceOutside;
                corridorBackwaterSurface = connectorWater;
                corridorBackwaterWidth = corridor.halfWidth;
                if (segmentLength > 1.0e-6) {
                    corridorBackwaterFlowX = segmentX / segmentLength;
                    corridorBackwaterFlowZ = segmentZ / segmentLength;
                }
                corridorBackwaterOwner = corridorTransitionIdentity(corridor);
            }
            if (distance > corridor.halfWidth || distance > closestConnectorDistance + 1.0e-6) {
                continue;
            }
            closestConnectorDistance = distance;
            outletConnector = true;
            result.transitionOwnerKind = WaterTransitionKind::OUTLET_CORRIDOR;
            result.transitionOwnerId = corridorTransitionIdentity(corridor);
            const double connectorDepth =
                std::clamp(1.2 + std::sqrt(corridor.discharge) * 0.065, 1.8, 14.0);
            const double standingLakeWater = result.waterSurface;
            if (segmentLength > 1.0e-6) {
                result.flowX = segmentX / segmentLength;
                result.flowZ = segmentZ / segmentLength;
                result.channelGradient =
                    std::max(result.channelGradient,
                             static_cast<double>(corridor.waterSurface[segment] -
                                                 corridor.waterSurface[segment + 1]) /
                                 segmentLength);
            }
            result.waterSurface = connectorWater;
            result.surfaceElevation =
                std::min(result.surfaceElevation, connectorWater - connectorDepth);
            result.discharge = std::max(result.discharge, static_cast<double>(corridor.discharge));
            result.channelDistance = distance;
            result.channelWidth =
                std::max(result.channelWidth, static_cast<double>(corridor.halfWidth) / 0.55);
            result.channelDepth = std::max(result.channelDepth, connectorDepth);
            result.erosionDepth = std::max(result.erosionDepth, connectorDepth);
            result.streamOrder = std::max<uint8_t>(result.streamOrder, 1);
            if (result.lake && connectorWater < standingLakeWater - 0.05) {
                result.lake = false;
            }
            if (result.lake && shoreline != nullptr && shoreline->valid &&
                connectorWater < shoreline->waterLevel - 0.05) {
                result.lake = false;
            }
            if (!result.lake) {
                result.endorheic = false;
                result.lakeDepth = 0.0;
                result.waterBodyId = NO_WATER_BODY;
                result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
                if (connectorWater > SEA_LEVEL + 0.05) result.ocean = false;
            }
        }
    }
    result.river = !result.ocean && !result.lake && (canonicalRoutedChannel || outletConnector);
    if (result.ocean || result.river || result.lake) result.channelBank = false;
    result.waterfall = false;
    if (corridorBackwater && !outletConnector && result.river && !result.lake) {
        const double controllingSurface =
            result.shoreWaterSurface > 0.0
                ? std::min(corridorBackwaterSurface, result.shoreWaterSurface)
                : corridorBackwaterSurface;
        const double stageDifference = controllingSurface - result.waterSurface;
        if (stageDifference > 0.0 && stageDifference < 2.5 &&
            corridorBackwaterDistance <= stageDifference * 8.0 + 1.0) {
            const double backwaterSurface = controllingSurface - corridorBackwaterDistance * 0.125;
            if (result.surfaceElevation + 0.05 < backwaterSurface) {
                result.waterSurface = std::max(result.waterSurface, backwaterSurface);
                result.surfaceElevation =
                    std::min(result.surfaceElevation, result.waterSurface - 0.125);
                result.channelWidth = std::max(result.channelWidth, corridorBackwaterWidth / 0.55);
                result.flowX = corridorBackwaterFlowX;
                result.flowZ = corridorBackwaterFlowZ;
                result.transitionOwnerKind = WaterTransitionKind::OUTLET_CORRIDOR;
                result.transitionOwnerId = corridorBackwaterOwner;
            }
        }
    }
    const double localWetSurface =
        result.ocean || result.river || result.lake ? result.waterSurface : result.surfaceElevation;
    if (corridorBackwater && !outletConnector && !result.ocean && !result.lake &&
        corridorBackwaterDistance <= 1.5 && corridorBackwaterSurface >= localWetSurface + 2.5) {
        // A steep corridor retains its upstream stage until the explicit
        // curtain. Its immediately adjacent dry cells are the physical lip;
        // without this freeboard the source-filled approach can spill into
        // unrelated low relief before it reaches the owned fall.
        result.surfaceElevation = std::max(result.surfaceElevation, corridorBackwaterSurface + 0.5);
        result.waterSurface = 0.0;
        result.ocean = false;
        result.river = false;
        result.lake = false;
        result.lakeBank = false;
        result.channelBank = true;
        result.endorheic = false;
        result.lakeDepth = 0.0;
        result.waterBodyId = NO_WATER_BODY;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
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
    bool ownedFallApproach = false;
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
        const double strongerSelectedDischarge = std::max(
            selectedRasterOwnerId != 0 ? rasterChannelDischarge : 0.0,
            analyticalGuide != nullptr && analyticalGuideCore ? analyticalGuide->discharge : 0.0);
        const bool fallSuppressedByStrongerRoute =
            strongerSelectedDischarge > fall.discharge + 1.0e-4;
        const bool fallMatchesSelectedConnector =
            !outletConnector ||
            std::min(std::abs(static_cast<double>(fall.topSurface) - result.waterSurface),
                     std::abs(static_cast<double>(fall.bottomSurface) - result.waterSurface)) <=
                0.125001;
        if (!fallMatchesSelectedConnector) continue;
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
        // A fall plane may cross block centers at any sub-block offset and at
        // any angle. A three-half-block sampling slab guarantees a complete
        // face-connected digital curtain instead of leaving one untagged
        // upper-stage column beside the lower receiving apron.
        constexpr double halfDepth = 0.75;
        const double segmentAlong =
            ((x - fall.startX) * flowX + (z - fall.startZ) * flowZ) / flowLength;
        const double capAlong = halfDepth / flowLength;
        const double approachAlong = std::min(1.0, 8.0 / flowLength);
        constexpr double RECEIVING_CORRIDOR_LENGTH = BASIN_RASTER_SPACING;
        if (!fallSuppressedByStrongerRoute && segmentAlong >= 1.0 - approachAlong - capAlong &&
            segmentAlong <= 1.0 + capAlong && crossStream <= fall.halfWidth) {
            ownedFallApproach = true;
            result.transitionOwnerKind = WaterTransitionKind::EXPLICIT_FALL;
            result.transitionOwnerId =
                transitionIdentity(0x46414C4C5F454447ULL, fall.endX, fall.endZ,
                                   fall.endX - fall.startX, fall.endZ - fall.startZ);
            // Keep only the discrete eight-cell approach at its upstream
            // source plane. The indexed rapid states below descend by one
            // eighth block per face into the receiver-centered curtain. A
            // fall must not claim the rest of its raster edge because a
            // converging downstream reach can already occupy that footprint.
            const double connectorWater = fall.topSurface;
            const double connectorDepth =
                std::clamp(1.2 + std::sqrt(fall.discharge) * 0.065, 1.8, 14.0);
            result.waterSurface = connectorWater;
            result.surfaceElevation =
                std::min(result.surfaceElevation, connectorWater - connectorDepth);
            result.discharge = std::max(result.discharge, static_cast<double>(fall.discharge));
            result.channelDistance = crossStream;
            result.channelWidth =
                std::max(result.channelWidth, static_cast<double>(fall.halfWidth) / 0.55);
            result.channelDepth = std::max(result.channelDepth, connectorDepth);
            result.channelGradient =
                std::max(result.channelGradient,
                         static_cast<double>(fall.topSurface - fall.bottomSurface) / flowLength);
            result.erosionDepth = std::max(result.erosionDepth, connectorDepth);
            result.streamOrder = std::max<uint8_t>(result.streamOrder, 1);
            result.flowX = flowX;
            result.flowZ = flowZ;
            const bool inletCrossesLakeContour =
                result.lake && result.shoreWaterSurface > 0.0 &&
                std::abs(static_cast<double>(fall.topSurface) - result.shoreWaterSurface) > 0.05;
            if (inletCrossesLakeContour ||
                connectorWater < static_cast<double>(fall.topSurface) - 0.05) {
                result.lake = false;
                result.endorheic = false;
                result.lakeDepth = 0.0;
                result.waterBodyId = NO_WATER_BODY;
                result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
            }
            if (!result.lake) {
                if (connectorWater > SEA_LEVEL + 0.05) result.ocean = false;
                result.river = !result.ocean;
            }
        }
        if (fallSuppressedByStrongerRoute) continue;
        const bool receivingApron = longitudinal > halfDepth &&
                                    longitudinal <= RECEIVING_CORRIDOR_LENGTH &&
                                    crossStream <= fall.halfWidth;
        if (receivingApron) {
            // Carry the lower stage through the receiving control cell. A
            // one-block apron only moved the large stage jump one row beyond
            // the curtain instead of connecting it to the routed receiver.
            const double receivingDepth =
                std::clamp(1.2 + std::sqrt(fall.discharge) * 0.065, 1.8, 14.0);
            result.waterSurface = fall.bottomSurface;
            result.surfaceElevation =
                std::min(result.surfaceElevation, result.waterSurface - receivingDepth);
            result.lake = false;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
            result.ocean =
                fall.bottomSurface <= SEA_LEVEL + 1.0e-4F && result.surfaceElevation < SEA_LEVEL;
            result.river = !result.ocean;
            result.discharge = std::max(result.discharge, static_cast<double>(fall.discharge));
            result.channelDepth = std::max(result.channelDepth, receivingDepth);
            result.channelDistance = crossStream;
            result.channelWidth =
                std::max(result.channelWidth, static_cast<double>(fall.halfWidth) / 0.55);
            if (!outletConnector) {
                result.transitionOwnerKind = WaterTransitionKind::RASTER_CHANNEL;
                result.transitionOwnerId =
                    transitionIdentity(0x46414C4C5F524543ULL, fall.endX, fall.endZ,
                                       fall.endX - fall.startX, fall.endZ - fall.startZ);
            }
        }
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
        const double plungeDepth = std::max(1.8, result.channelDepth);
        result.surfaceElevation =
            std::min(result.surfaceElevation, result.waterSurface - plungeDepth);
        result.erosionDepth = std::max(result.erosionDepth, plungeDepth);
        result.discharge = std::max(result.discharge, static_cast<double>(fall.discharge));
        if (!result.lake && fall.bottomSurface <= SEA_LEVEL + 1.0e-4F &&
            result.surfaceElevation < SEA_LEVEL) {
            // The explicit fall owns the narrow transition into the lower
            // ocean. Channel incision below sea level remains a river
            // everywhere else, but the receiving footprint is ocean water.
            result.ocean = true;
            result.river = false;
        }
        result.waterfall = true;
        result.transitionOwnerKind = WaterTransitionKind::EXPLICIT_FALL;
        result.transitionOwnerId =
            transitionIdentity(0x46414C4C5F454447ULL, fall.endX, fall.endZ, fall.endX - fall.startX,
                               fall.endZ - fall.startZ);
        result.waterfallAnchor = result.waterfallAnchor || (std::abs(x - fall.endX) <= 1.0e-6 &&
                                                            std::abs(z - fall.endZ) <= 1.0e-6);
        // The explicit analytical fall is the only legal opening through a
        // lake rim. It owns this narrow footprint even when the continuous
        // contour remains positive on either side of the outlet.
        if (result.lake) {
            result.lake = false;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.river = !result.ocean;
            result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
        }
    }
    const bool authoritativeOutletContinuation =
        shoreline != nullptr && shoreline->valid && shoreline->outletContinuation;
    const bool finalRoutedChannel = result.streamOrder > 0 && result.channelWidth > 0.0 &&
                                    result.channelDistance <= result.channelWidth * 0.55;
    const bool connectedLakeOutlet =
        outletConnector || ownedReceivingChannel || authoritativeOutletContinuation;
    const bool finalChannelAuthority =
        selectedRasterOwnerId != 0 || analyticalGuideCore || connectedLakeOutlet;
    const bool stageCompatibleWithNearbyLake =
        result.shoreWaterSurface <= 0.0 || result.shoreWaterSurface <= result.waterSurface + 1.0;
    const bool protectedRoutedChannel = finalRoutedChannel && finalChannelAuthority &&
                                        (connectedLakeOutlet || stageCompatibleWithNearbyLake);

    const int64_t blockX = static_cast<int64_t>(std::llround(x));
    const int64_t blockZ = static_cast<int64_t>(std::llround(z));
    const bool exactBlockSample = std::abs(x - static_cast<double>(blockX)) <= 1.0e-6 &&
                                  std::abs(z - static_cast<double>(blockZ)) <= 1.0e-6;
    const SettledBankCell* settledBank =
        exactBlockSample ? settledBankAt(solution, blockX, blockZ) : nullptr;
    if (settledBank != nullptr &&
        static_cast<double>(settledBank->targetEighths) / 8.0 <= SEA_LEVEL) {
        // A submerged guard does not replace source-filled ocean. Treat it
        // as absent so the settled water field or continuous coast remains
        // authoritative at this coordinate.
        settledBank = nullptr;
    }
    const double settledBankTarget =
        settledBank != nullptr ? static_cast<double>(settledBank->targetEighths) / 8.0 : 0.0;
    const bool continuationMatchesSettledBank =
        settledBank != nullptr && result.waterSurface > 0.0 &&
        std::abs(result.waterSurface + 0.5 - settledBankTarget) <= 0.125;
    bool dominantRasterContinuation =
        settledBank != nullptr && result.transitionOwnerKind == WaterTransitionKind::RASTER_CHANNEL;
    if (dominantRasterContinuation) {
        constexpr std::array<std::pair<int64_t, int64_t>, 4> CARDINAL_BLOCKS = {
            std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}};
        for (const auto [offsetX, offsetZ] : CARDINAL_BLOCKS) {
            const SettledBankCell* neighborBank =
                settledBankAt(solution, blockX + offsetX, blockZ + offsetZ);
            if (neighborBank != nullptr &&
                neighborBank->targetEighths > settledBank->targetEighths) {
                dominantRasterContinuation = false;
                break;
            }
        }
    }
    const bool specificContinuationAuthority =
        result.transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR ||
        result.transitionOwnerKind == WaterTransitionKind::CHANNEL_GUIDE ||
        dominantRasterContinuation;
    if (settledBank != nullptr && continuationMatchesSettledBank && specificContinuationAuthority &&
        (outletConnector || ownedReceivingChannel ||
         (authoritativeOutletContinuation && analyticalGuideCore))) {
        // Raster and guide representations of the same shared outlet can
        // overlap at a catchment face with slightly different sampled stages.
        // That is one routed owner, not a retaining-bank conflict. Preserve
        // the analytical continuation selected by the shared contour.
        settledBank = nullptr;
    }
    const SettledWaterCell* settledWater = exactBlockSample && settledBank == nullptr
                                               ? settledWaterAt(solution, blockX, blockZ)
                                               : nullptr;
    const bool implicitOceanSourceAuthority = exactBlockSample && settledBank == nullptr &&
                                              settledWater == nullptr && result.ocean &&
                                              result.waterSurface <= SEA_LEVEL + 1.0e-6;
    const bool settledOutletAuthority =
        settledWater != nullptr &&
        (settledWater->ownerKind == WaterTransitionKind::OUTLET_CORRIDOR ||
         settledWater->ownerKind == WaterTransitionKind::CHANNEL_GUIDE);
    const bool ownedOutletContinuation =
        outletConnector || ownedReceivingChannel ||
        (authoritativeOutletContinuation &&
         (settledOutletAuthority ||
          result.transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR ||
          result.transitionOwnerKind == WaterTransitionKind::CHANNEL_GUIDE));
    const auto settledRasterGradientReachesCore = [&](auto&& self, int64_t candidateX,
                                                      int64_t candidateZ, int16_t stage,
                                                      unsigned depth) -> bool {
        const double candidateGridX =
            (static_cast<double>(candidateX) - solution.originX) / BASIN_RASTER_SPACING +
            RASTER_APRON;
        const double candidateGridZ =
            (static_cast<double>(candidateZ) - solution.originZ) / BASIN_RASTER_SPACING +
            RASTER_APRON;
        const double candidateWidth =
            reconstruct(solution.channelWidth, candidateGridX, candidateGridZ);
        const bool routedCore = candidateWidth > 0.0 &&
                                reconstruct(solution.discharge, candidateGridX, candidateGridZ) >=
                                    MIN_CHANNEL_DISCHARGE &&
                                reconstructChannelDistance(solution, candidateGridX,
                                                           candidateGridZ) <= candidateWidth * 0.55;
        if (routedCore) return true;
        if (depth >= 8) return false;
        constexpr std::array<std::pair<int64_t, int64_t>, 4> CARDINAL_BLOCKS = {
            std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}};
        for (const auto [offsetX, offsetZ] : CARDINAL_BLOCKS) {
            const SettledWaterCell* downstream =
                settledWaterAt(solution, candidateX + offsetX, candidateZ + offsetZ);
            if (downstream == nullptr ||
                downstream->ownerKind != WaterTransitionKind::RASTER_CHANNEL ||
                downstream->surfaceEighths != stage - 1) {
                continue;
            }
            if (self(self, candidateX + offsetX, candidateZ + offsetZ, downstream->surfaceEighths,
                     depth + 1)) {
                return true;
            }
        }
        return false;
    };
    const bool settledRasterGradientCap =
        settledWater != nullptr && settledWater->ownerKind == WaterTransitionKind::RASTER_CHANNEL &&
        settledRasterGradientReachesCore(settledRasterGradientReachesCore, blockX, blockZ,
                                         settledWater->surfaceEighths, 0);
    const bool settledOwnedTransitionFringe =
        settledWater != nullptr && ownedOutletContinuation && result.river &&
        (settledWater->ownerKind == WaterTransitionKind::OUTLET_CORRIDOR ||
         settledWater->ownerKind == WaterTransitionKind::CHANNEL_GUIDE) &&
        result.channelWidth > 0.0 && result.channelDistance <= result.channelWidth * 0.55 + 0.5;
    const bool detachedSettledApron =
        settledWater != nullptr && settledWater->ownerKind != WaterTransitionKind::NONE &&
        !finalRoutedChannel && !settledRasterGradientCap && !settledOwnedTransitionFringe;
    if (detachedSettledApron) {
        // The settled rapid raster includes a narrow construction apron so
        // predecessor chains can be solved without reading a neighboring
        // page. That apron is not water ownership. Publishing it outside the
        // final analytical channel produces isolated one-block water cells
        // and checkerboard shoreline gaps.
        settledWater = nullptr;
    }
    double settledConflictBankTarget = 0.0;
    if (settledWater != nullptr && settledWater->ownerKind == WaterTransitionKind::RASTER_CHANNEL &&
        !finalRoutedChannel && !ownedOutletContinuation && result.shoreWaterSurface > 0.0 &&
        result.lakeShoreDistance <= 0.0 && result.lakeShoreDistance > -(LAKE_BANK_WIDTH + 8.0)) {
        // A settled rapid may extend laterally beyond its analytical channel
        // core to complete a predecessor chain. Keep that halo dry where it
        // meets a standing shoreline so it cannot expose a side wall between
        // incompatible water stages.
        settledConflictBankTarget =
            std::max(result.shoreWaterSurface,
                     static_cast<double>(settledWater->surfaceEighths) / 8.0) +
            0.5;
        settledWater = nullptr;
    }
    if ((settledWater != nullptr || implicitOceanSourceAuthority) && shoreline != nullptr &&
        shoreline->valid && !ownedOutletContinuation &&
        (settledWater == nullptr ||
         settledWater->ownerKind != WaterTransitionKind::OUTLET_CORRIDOR) &&
        result.transitionOwnerKind != WaterTransitionKind::OUTLET_CORRIDOR &&
        result.transitionOwnerKind != WaterTransitionKind::EXPLICIT_FALL &&
        std::abs(result.lakeShoreDistance) <= LAKE_BANK_WIDTH + 8.0) {
        const double settledSurface = settledWater != nullptr
                                          ? static_cast<double>(settledWater->surfaceEighths) / 8.0
                                          : static_cast<double>(SEA_LEVEL);
        if (result.shoreWaterSurface >= settledSurface + 2.5) {
            // A distinct high standing lake cannot overlap a lower settled
            // route or ocean cell. Publish the lake's existing rim freeboard
            // at only this narrow contour conflict; its owned outlet remains
            // wet and explicit falls retain their sharp transition.
            settledConflictBankTarget = result.shoreWaterSurface + 0.5;
            settledWater = nullptr;
        }
    }
    double routedContactLakeStage = std::numeric_limits<double>::infinity();
    if (shoreline != nullptr && shoreline->valid) {
        routedContactLakeStage = shoreline->waterLevel;
    }
    if (lakeBody.found) {
        routedContactLakeStage = std::min(routedContactLakeStage, lakeBody.waterLevel);
    }
    const bool settledRasterContact =
        settledWater != nullptr && settledWater->ownerKind == WaterTransitionKind::RASTER_CHANNEL;
    const double routedContactStage = settledRasterContact
                                          ? static_cast<double>(settledWater->surfaceEighths) / 8.0
                                          : result.waterSurface;
    if (std::isfinite(routedContactLakeStage) && !ownedOutletContinuation &&
        (settledRasterContact ||
         (result.transitionOwnerKind == WaterTransitionKind::RASTER_CHANNEL && result.river)) &&
        !result.waterfall && result.lakeShoreDistance <= 0.0 &&
        result.lakeShoreDistance > -(LAKE_BANK_WIDTH + 8.0) &&
        routedContactStage > routedContactLakeStage + 0.125001) {
        // A raster reach that merely overlaps a different lake contour is not
        // that lake's outlet. Preserve a dry bank at the high route's edge;
        // only the globally owned corridor or explicit fall may connect the
        // two stages. This keeps both water bodies while preventing a lateral
        // wall of unsupported water at the signed shoreline.
        settledConflictBankTarget = std::max(settledConflictBankTarget, routedContactStage + 0.5);
        settledWater = nullptr;
    }
    const bool settledFieldAuthority =
        settledBank != nullptr || settledWater != nullptr || settledConflictBankTarget > SEA_LEVEL;
    if (protectedRoutedChannel && !result.lakeBank && !result.channelBank && !result.ocean &&
        !result.lake && !result.waterfall) {
        // Late analytical corridors and receiving aprons refine channel
        // distance after the initial raster classification. Reapply wet
        // topology from that final shared authority so a valid cardinal
        // predecessor cannot become a dry hole inside the rapid core.
        result.river = true;
    }
    const auto closeIncompatibleLakeBank = [&] {
        result.lakeBankTarget = result.shoreWaterSurface + 0.5;
        result.surfaceElevation = std::max(result.surfaceElevation, result.lakeBankTarget);
        result.waterSurface = 0.0;
        result.ocean = false;
        result.river = false;
        result.lakeBank = true;
        result.channelBank = false;
        result.endorheic = false;
        result.waterBodyId = NO_WATER_BODY;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
    };
    const double nearbyLakeStageDifference =
        result.shoreWaterSurface > 0.0 ? result.shoreWaterSurface - result.waterSurface : 0.0;
    const bool closesPartialLakeBankEdge = result.generatedFluidLevel > 0 &&
                                           nearbyLakeStageDifference >= 2.5 &&
                                           result.lakeShoreDistance > -(LAKE_BANK_WIDTH + 8.0);
    if (!settledFieldAuthority && !ownedOutletContinuation && !result.lake && !result.waterfall &&
        result.shoreWaterSurface > 0.0 && result.lakeShoreDistance <= 0.0 &&
        (result.lakeShoreDistance > -LAKE_BANK_WIDTH || closesPartialLakeBankEdge)) {
        result.lakeBankInfluence =
            1.0 - smoothstep(0.0, LAKE_BANK_WIDTH, -result.lakeShoreDistance);
        if (result.river && result.surfaceElevation + 0.05 < result.shoreWaterSurface) {
            const double shoreDrop = result.shoreWaterSurface - result.waterSurface;
            if (shoreDrop < 2.5) {
                // Small differences are a backwater, not a vertical fall.
                result.lakeBank = false;
                result.lakeBankInfluence = 0.0;
                result.lakeBankTarget = 0.0;
                result.waterSurface = result.shoreWaterSurface;
                result.surfaceElevation =
                    std::min(result.surfaceElevation, result.waterSurface - 0.125);
                result.generatedFluidLevel = 0;
            } else {
                // An unrelated routed owner cannot cut a vertical water face
                // through a standing lake contour. Only the lake's shared
                // outlet continuation or tagged fall may own that opening.
                closeIncompatibleLakeBank();
            }
        } else {
            const double rimTarget = result.shoreWaterSurface + 0.5;
            result.surfaceElevation =
                std::lerp(result.surfaceElevation, std::max(result.surfaceElevation, rimTarget),
                          result.lakeBankInfluence);
            result.lakeBankTarget = result.surfaceElevation;
            result.lakeBank = result.lakeBankInfluence > 1.0e-4 &&
                              result.surfaceElevation + 0.05 >= result.shoreWaterSurface;
        }
    }
    if (!result.lake && !result.lakeBank &&
        result.transitionOwnerKind != WaterTransitionKind::OUTLET_CORRIDOR &&
        result.lakeShoreDistance <= -LAKE_SHORE_PERTURBATION_BAND) {
        // A signed shoreline distance is authoritative only in the narrow
        // contour band. Basin-local distance transforms may choose different
        // remote lakes on opposite sides of a catchment face, but those
        // distant values have no physical or rendering meaning. Collapse the
        // exterior to one sentinel so an inactive contour cannot expose the
        // numerical catchment lattice to climate and diagnostic consumers.
        result.lakeShoreDistance = -1.0e9;
        result.shoreWaterSurface = 0.0;
    }
    if (result.lake) {
        const auto equilibrium =
            std::ranges::find_if(solution.lakeEquilibria, [&](const LakeEquilibrium& candidate) {
                return candidate.identity == result.waterBodyId;
            });
        if (equilibrium != solution.lakeEquilibria.end()) {
            result.waterSurface = equilibrium->surface;
            result.lakeDepth = std::max(0.0, result.waterSurface - result.surfaceElevation);
            result.lakeAreaSquareKilometers = equilibrium->areaSquareKilometers;
            result.lakeVolumeCubicMeters = equilibrium->volumeCubicMeters;
            result.lakeRunoffMmSquareKilometers = equilibrium->runoffMmSquareKilometers;
            result.lakeLossMm = equilibrium->lossMm;
            result.lakeOverflowMmSquareKilometers = equilibrium->overflowMmSquareKilometers;
            result.lakeSpillSurface = equilibrium->spillSurface;
        }
    }
    if (selectedRasterOwnerId != 0 && selectedRasterSource >= 0 && result.river &&
        !outletConnector && !analyticalGuideCore && !result.waterfall) {
        if (!ownedFallApproach) {
            result.transitionOwnerKind = WaterTransitionKind::RASTER_CHANNEL;
            result.transitionOwnerId = selectedRasterOwnerId;
        }
    }
    if (analyticalGuide != nullptr && analyticalGuideCore && result.river && !outletConnector &&
        !result.waterfall && !ownedFallApproach) {
        result.transitionOwnerKind = WaterTransitionKind::CHANNEL_GUIDE;
        result.transitionOwnerId = guideTransitionIdentity(*analyticalGuide);
    }
    bool explicitRapidAuthority = false;
    bool immutableSettledWaterAuthority = false;
    if (exactBlockSample) {
        RapidCell rapid;
        bool hasRapid = false;
        const int64_t localX = blockX - static_cast<int64_t>(std::llround(solution.originX));
        const int64_t localZ = blockZ - static_cast<int64_t>(std::llround(solution.originZ));
        if (localX >= TRANSITION_MIN_LOCAL && localX <= TRANSITION_MAX_LOCAL &&
            localZ >= TRANSITION_MIN_LOCAL && localZ <= TRANSITION_MAX_LOCAL &&
            solution.transitionRowOffsets.size() == TRANSITION_ROW_COUNT + 1) {
            const size_t row = static_cast<size_t>(localZ - TRANSITION_MIN_LOCAL);
            const auto rowBegin =
                solution.transitionCells.begin() + solution.transitionRowOffsets[row];
            const auto rowEnd =
                solution.transitionCells.begin() + solution.transitionRowOffsets[row + 1];
            const auto firstTransition = std::lower_bound(
                rowBegin, rowEnd, static_cast<int16_t>(localX),
                [](const TransitionCell& cell, int16_t targetX) { return cell.localX < targetX; });
            for (auto cell = firstTransition;
                 cell != rowEnd && cell->localX == static_cast<int16_t>(localX); ++cell) {
                const TransitionDescriptor& descriptor =
                    solution.transitionDescriptors[cell->descriptorIndex];
                const RapidCell candidate{
                    .x = blockX,
                    .z = blockZ,
                    .topY = descriptor.topY,
                    .ownerIndex = descriptor.ownerIndex,
                    .ownerId = descriptor.ownerId,
                    .flowX = descriptor.flowX,
                    .flowZ = descriptor.flowZ,
                    .channelDistance = static_cast<float>(cell->distanceQuarterBlocks) * 0.25F,
                    .channelWidth = descriptor.channelWidth,
                    .level = cell->level,
                    .ownerKind = descriptor.ownerKind,
                };
                if (candidate.ownerKind != RapidOwnerKind::EXPLICIT_FALL ||
                    result.transitionOwnerKind != WaterTransitionKind::EXPLICIT_FALL ||
                    candidate.ownerId != result.transitionOwnerId) {
                    continue;
                }
                if (!hasRapid || candidate.topY > rapid.topY) {
                    rapid = candidate;
                    hasRapid = true;
                }
            }
        }
        explicitRapidAuthority = hasRapid;
        if (explicitRapidAuthority) {
            result.transitionOwnerKind = rapid.ownerKind;
            result.transitionOwnerId = rapid.ownerId;
            result.flowX = rapid.flowX;
            result.flowZ = rapid.flowZ;
            result.channelDistance = rapid.channelDistance;
            result.channelWidth = rapid.channelWidth;
            const FluidState rapidState =
                rapid.level == 0 ? FluidState::source() : FluidState::flowing(rapid.level);
            // The transition descriptor owns both the packed state and its
            // emitted top plane. Retaining the interpolated raster stage here
            // caused quantization to discard levels one through five even
            // though their immutable predecessors were present.
            result.waterSurface = static_cast<double>(rapid.topY) + fluidSurfaceHeight(rapidState);
            if (rapid.ownerKind == RapidOwnerKind::EXPLICIT_FALL && rapid.ownerIndex >= 0 &&
                static_cast<size_t>(rapid.ownerIndex) < solution.outletFalls.size()) {
                const OutletFall& fall =
                    solution.outletFalls[static_cast<size_t>(rapid.ownerIndex)];
                // The indexed source and levels one through six are the
                // horizontal approach to the drop. Only the level-seven lip
                // is itself falling water; the analytical curtain below
                // retains waterfall ownership independently.
                result.waterfall = result.waterfall || rapid.level == 7;
                result.waterfallTop = fall.topSurface;
                result.waterfallBottom = fall.bottomSurface;
                result.waterfallWidth = fall.halfWidth * 2.0F;
                if (rapid.level == 7) {
                    const double plungeDepth = std::max(1.8, result.channelDepth);
                    result.surfaceElevation =
                        std::min(result.surfaceElevation,
                                 static_cast<double>(fall.bottomSurface) - plungeDepth);
                    result.erosionDepth = std::max(result.erosionDepth, plungeDepth);
                }
            }
        }
        if (explicitRapidAuthority && rapid.level > 0) {
            // The globally owned rapid patch is the legal outlet throat
            // through a standing lake contour. Its source predecessor remains
            // lake water, while levels one through seven become the routed
            // channel without leaving a categorical shoreline gap.
            result.lake = false;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.lakeShoreDistance = std::min(result.lakeShoreDistance, -0.125);
            result.lakeBank = false;
            result.lakeBankInfluence = 0.0;
            result.ocean = false;
            result.river = true;
            result.generatedFluidLevel = rapid.level;
        }
    }
    if (!explicitRapidAuthority && !result.waterfall && exactBlockSample) {
        // The immutable settled field is the block-resolution authority. A
        // conflict bank wins first; otherwise the settled water cell owns its
        // exact stage, bed, state, and routed identity. Legacy analytical
        // lake and bank classifications only survive where neither exists.
        if (!result.lake && (settledBank != nullptr || settledConflictBankTarget > SEA_LEVEL)) {
            double bankTarget = settledBank != nullptr
                                    ? static_cast<double>(settledBank->targetEighths) / 8.0
                                    : settledConflictBankTarget;
            if (!ownedOutletContinuation && result.shoreWaterSurface > 0.0 &&
                result.lakeShoreDistance <= 0.0 &&
                result.lakeShoreDistance > -(LAKE_BANK_WIDTH + 8.0)) {
                const double influence =
                    1.0 - smoothstep(0.0, LAKE_BANK_WIDTH + 8.0, -result.lakeShoreDistance);
                bankTarget = std::lerp(result.surfaceElevation, bankTarget, influence);
            }
            if ((result.ocean || result.river || result.lake) &&
                result.waterSurface > result.surfaceElevation + 0.05) {
                bankTarget = std::max(bankTarget, result.waterSurface + 0.5);
            }
            result.channelBankSubstrate = std::min(result.surfaceElevation, unbankedChannelSurface);
            if (std::isfinite(lowlandChannelFloor)) {
                result.channelBankSubstrate =
                    std::min(result.channelBankSubstrate, lowlandChannelFloor);
            }
            result.channelBankSubstrateValid = true;
            result.surfaceElevation = std::max(result.surfaceElevation, bankTarget);
            result.waterSurface = 0.0;
            result.ocean = false;
            result.river = false;
            result.lake = false;
            result.lakeBank = false;
            result.channelBank = true;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.generatedFluidLevel = 0;
            result.transitionOwnerKind = WaterTransitionKind::NONE;
            result.transitionOwnerId = 0;
        } else if (settledWater != nullptr && !result.lake) {
            immutableSettledWaterAuthority = true;
            const WaterTransitionKind analyticalOwnerKind = result.transitionOwnerKind;
            const uint64_t analyticalOwnerId = result.transitionOwnerId;
            int32_t surfaceEighths = settledWater->surfaceEighths;
            FluidState settledState = FluidState::fromPacked(settledWater->fluidPacked);
            if (!settledState.isSource()) {
                constexpr std::array<std::pair<int64_t, int64_t>, 4> CARDINAL_BLOCKS = {
                    std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}};
                const auto hasCompletePredecessorChain = [&](auto&& self, int64_t x, int64_t z,
                                                             int16_t stage) -> bool {
                    const int16_t predecessorStage = static_cast<int16_t>(stage + 1);
                    for (const auto [offsetX, offsetZ] : CARDINAL_BLOCKS) {
                        const int64_t predecessorX = x + offsetX;
                        const int64_t predecessorZ = z + offsetZ;
                        const SettledWaterCell* predecessor =
                            settledWaterAt(solution, predecessorX, predecessorZ);
                        if (predecessor == nullptr ||
                            predecessor->surfaceEighths != predecessorStage ||
                            predecessor->ownerKind != settledWater->ownerKind ||
                            (predecessor->ownerIndex != settledWater->ownerIndex &&
                             settledWater->ownerKind != WaterTransitionKind::RASTER_CHANNEL)) {
                            continue;
                        }
                        const SettledBankCell* predecessorBank =
                            settledBankAt(solution, predecessorX, predecessorZ);
                        if (predecessorBank != nullptr &&
                            static_cast<double>(predecessorBank->targetEighths) / 8.0 > SEA_LEVEL) {
                            continue;
                        }
                        const double predecessorGridX =
                            (static_cast<double>(predecessorX) - solution.originX) /
                                BASIN_RASTER_SPACING +
                            RASTER_APRON;
                        const double predecessorGridZ =
                            (static_cast<double>(predecessorZ) - solution.originZ) /
                                BASIN_RASTER_SPACING +
                            RASTER_APRON;
                        const double predecessorShore =
                            dominantPositiveLevel(solution, predecessorGridX, predecessorGridZ);
                        const double predecessorShoreDistance = reconstruct(
                            solution.lakeShoreDistance, predecessorGridX, predecessorGridZ);
                        const bool incompatibleRasterLakeContact =
                            predecessor->ownerKind == WaterTransitionKind::RASTER_CHANNEL &&
                            predecessorShore >
                                static_cast<double>(predecessor->surfaceEighths) / 8.0 + 2.5 &&
                            predecessorShoreDistance <= 0.0 &&
                            predecessorShoreDistance > -(LAKE_BANK_WIDTH + 8.0);
                        if (incompatibleRasterLakeContact) continue;
                        if (predecessorStage % 8 == 0 ||
                            self(self, predecessorX, predecessorZ, predecessorStage)) {
                            return true;
                        }
                    }
                    return false;
                };
                const bool completeChain = hasCompletePredecessorChain(
                    hasCompletePredecessorChain, blockX, blockZ, settledWater->surfaceEighths);
                if (!completeChain) {
                    surfaceEighths = static_cast<int32_t>(
                        std::floor(static_cast<double>(surfaceEighths) / 8.0) * 8.0);
                    settledState = FluidState::source();
                }
            }
            const double settledSurface = static_cast<double>(surfaceEighths) / 8.0;
            result.waterSurface = settledSurface;
            result.generatedFluidLevel = settledState.level();
            result.ocean =
                result.waterSurface <= SEA_LEVEL + 1.0e-6 && result.surfaceElevation < SEA_LEVEL;
            result.river = !result.ocean;
            result.lake = false;
            result.lakeBank = false;
            result.channelBank = false;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.transitionOwnerKind = settledWater->ownerKind;
            switch (settledWater->ownerKind) {
                case WaterTransitionKind::NONE:
                case WaterTransitionKind::COUNT:
                    result.transitionOwnerId = 0;
                    break;
                case RapidOwnerKind::EXPLICIT_FALL:
                    result.transitionOwnerId = 0;
                    break;
                case RapidOwnerKind::OUTLET_CORRIDOR:
                    result.transitionOwnerId =
                        settledWater->ownerIndex >= 0 &&
                                static_cast<size_t>(settledWater->ownerIndex) <
                                    solution.outletCorridors.size()
                            ? corridorTransitionIdentity(
                                  solution.outletCorridors[static_cast<size_t>(
                                      settledWater->ownerIndex)])
                            : 0;
                    break;
                case RapidOwnerKind::CHANNEL_GUIDE:
                    result.transitionOwnerId =
                        settledWater->ownerIndex >= 0 &&
                                static_cast<size_t>(settledWater->ownerIndex) <
                                    solution.guides.size()
                            ? guideTransitionIdentity(
                                  solution.guides[static_cast<size_t>(settledWater->ownerIndex)])
                            : 0;
                    break;
                case RapidOwnerKind::RASTER_CHANNEL:
                    result.transitionOwnerId = solution.rasterDrainageOwnerId;
                    break;
            }
            if (ownedOutletContinuation &&
                analyticalOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR &&
                analyticalOwnerId != 0) {
                result.transitionOwnerKind = analyticalOwnerKind;
                result.transitionOwnerId = analyticalOwnerId;
            }
            const double settledBed = static_cast<double>(settledWater->bedEighths) / 8.0;
            // The settled field owns the top state and provides a maximum
            // bed elevation. Preserve a deeper continuous physical bed so a
            // valid rapid emits a full water volume instead of a one-block
            // shelf suspended above an ocean trench or incised channel.
            result.surfaceElevation =
                std::min({result.surfaceElevation, unbankedChannelSurface, settledBed});
            result.channelDepth =
                std::max(result.channelDepth, result.waterSurface - result.surfaceElevation);
        }
    }
    if (exactBlockSample && (ownedOutletContinuation || settledOutletAuthority) &&
        shoreline != nullptr && shoreline->valid && result.river && !result.waterfall &&
        result.transitionOwnerKind != WaterTransitionKind::EXPLICIT_FALL &&
        result.lakeShoreDistance <= 0.0 && result.lakeShoreDistance > -28.0) {
        // A shared shoreline can belong to the neighboring catchment and is
        // therefore unavailable while either immutable basin field is built.
        // Complete that shared boundary here: a standing source surface owns
        // the integer plane, and the first exterior routed column must be
        // level one. Further exterior bands may descend by one eighth after
        // enough signed-distance clearance to absorb shoreline perturbation.
        // This is a deterministic refinement of the settled field, not a
        // generation-time fluid tick.
        const double sourceSurface = std::ceil(shoreline->waterLevel - 1.0e-6);
        constexpr double FLOW_LEVEL_DISTANCE = 4.0;
        const int level =
            std::clamp(static_cast<int>(std::floor(std::max(0.0, -result.lakeShoreDistance) /
                                                   FLOW_LEVEL_DISTANCE)) +
                           1,
                       1, 7);
        const double minimumRoutedSurface = sourceSurface - static_cast<double>(level) * 0.125;
        if (std::abs(result.waterSurface - sourceSurface) < 2.5 &&
            result.waterSurface < minimumRoutedSurface) {
            result.waterSurface = minimumRoutedSurface;
            const int stageEighths = static_cast<int>(std::lround(result.waterSurface * 8.0));
            const int floorEighths =
                static_cast<int>(std::floor(static_cast<double>(stageEighths) / 8.0) * 8.0);
            const int remainder = stageEighths - floorEighths;
            result.generatedFluidLevel = remainder == 0 ? 0 : static_cast<uint8_t>(8 - remainder);
            result.surfaceElevation =
                std::min(result.surfaceElevation, result.waterSurface - 0.125);
            result.channelDepth =
                std::max(result.channelDepth, result.waterSurface - result.surfaceElevation);
        }
    }
    if (!explicitRapidAuthority && !result.waterfall && exactBlockSample) {
        // A source or levels one through six spread horizontally when their
        // covered cell below is source water. Surround the sides and upstream
        // end of an explicit fall approach with physical lip terrain whenever
        // the neighboring column does not already carry a compatible state at
        // the same plane. Level seven remains open downstream because it
        // cannot spread horizontally and owns the falling curtain there.
        constexpr std::array<std::pair<int64_t, int64_t>, 4> CARDINAL_BLOCKS = {
            std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}};
        double explicitApproachBankTarget = 0.0;
        for (const auto [offsetX, offsetZ] : CARDINAL_BLOCKS) {
            const int64_t neighborX = blockX + offsetX;
            const int64_t neighborZ = blockZ + offsetZ;
            const int64_t neighborLocalX =
                neighborX - static_cast<int64_t>(std::llround(solution.originX));
            const int64_t neighborLocalZ =
                neighborZ - static_cast<int64_t>(std::llround(solution.originZ));
            if (neighborLocalX < TRANSITION_MIN_LOCAL || neighborLocalX > TRANSITION_MAX_LOCAL ||
                neighborLocalZ < TRANSITION_MIN_LOCAL || neighborLocalZ > TRANSITION_MAX_LOCAL ||
                solution.transitionRowOffsets.size() != TRANSITION_ROW_COUNT + 1) {
                continue;
            }

            const size_t row = static_cast<size_t>(neighborLocalZ - TRANSITION_MIN_LOCAL);
            const auto rowBegin =
                solution.transitionCells.begin() + solution.transitionRowOffsets[row];
            const auto rowEnd =
                solution.transitionCells.begin() + solution.transitionRowOffsets[row + 1];
            const auto firstTransition = std::lower_bound(
                rowBegin, rowEnd, static_cast<int16_t>(neighborLocalX),
                [](const TransitionCell& cell, int16_t targetX) { return cell.localX < targetX; });
            for (auto cell = firstTransition;
                 cell != rowEnd && cell->localX == static_cast<int16_t>(neighborLocalX); ++cell) {
                const TransitionDescriptor& descriptor =
                    solution.transitionDescriptors[cell->descriptorIndex];
                if (descriptor.ownerKind != RapidOwnerKind::EXPLICIT_FALL) {
                    continue;
                }
                const int rapidTopY = descriptor.topY;
                const bool currentWaterAtRapidPlane =
                    (result.ocean || result.river || result.lake) &&
                    static_cast<int>(std::ceil(result.waterSurface)) - 1 == rapidTopY;
                const double fromRapidToCurrentX = -static_cast<double>(offsetX);
                const double fromRapidToCurrentZ = -static_cast<double>(offsetZ);
                const double directionalAlignment =
                    descriptor.flowX * fromRapidToCurrentX + descriptor.flowZ * fromRapidToCurrentZ;
                const bool upstreamPredecessor = directionalAlignment < -0.55;
                const bool downstreamSuccessor = directionalAlignment > 0.55;
                const bool sourceCompatibility =
                    result.generatedFluidLevel == 0 &&
                    (cell->level == 0 || (cell->level == 1 && upstreamPredecessor));
                const bool flowingCompatibility =
                    result.generatedFluidLevel > 0 &&
                    ((upstreamPredecessor && result.generatedFluidLevel + 1 == cell->level) ||
                     (downstreamSuccessor &&
                      (result.generatedFluidLevel == cell->level + 1 ||
                       (cell->level == 7 && result.generatedFluidLevel == 7))));
                const bool compatibleState =
                    currentWaterAtRapidPlane && (sourceCompatibility || flowingCompatibility);
                const bool openFallingSuccessor = cell->level == 7 && downstreamSuccessor;
                if (compatibleState || openFallingSuccessor) continue;
                explicitApproachBankTarget =
                    std::max(explicitApproachBankTarget, static_cast<double>(rapidTopY) + 1.5);
            }
        }
        if (explicitApproachBankTarget > 0.0) {
            result.channelBankSubstrate = std::min(result.surfaceElevation, unbankedChannelSurface);
            if (std::isfinite(lowlandChannelFloor)) {
                result.channelBankSubstrate =
                    std::min(result.channelBankSubstrate, lowlandChannelFloor);
            }
            result.channelBankSubstrateValid = true;
            result.surfaceElevation = std::max(result.surfaceElevation, explicitApproachBankTarget);
            result.waterSurface = 0.0;
            result.ocean = false;
            result.river = false;
            result.lake = false;
            result.waterfall = false;
            result.lakeBank = false;
            result.channelBank = true;
            result.endorheic = false;
            result.lakeDepth = 0.0;
            result.waterBodyId = NO_WATER_BODY;
            result.generatedFluidLevel = 0;
            result.transitionOwnerKind = WaterTransitionKind::NONE;
            result.transitionOwnerId = 0;
        }
    }
    if (exactBlockSample && result.waterfall &&
        result.transitionOwnerKind == WaterTransitionKind::EXPLICIT_FALL &&
        !explicitRapidAuthority) {
        // The analytical receiver footprint still describes the plunge pool
        // and far-water curtain, but only an indexed level-seven lip has a
        // horizontal predecessor at the upper plane. Do not let a lower-pool
        // anchor emit a second unsupported voxel curtain beside that lip.
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
    }
    if ((result.transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR ||
         result.transitionOwnerKind == WaterTransitionKind::CHANNEL_GUIDE) &&
        result.channelDistance < 1.0e-3) {
        // Segment projection around an exact shared endpoint can retain a
        // few ten-thousandths of a block from either side. Publish the
        // mathematical endpoint as zero so face sampling is identical.
        result.channelDistance = 0.0;
    }
    quantizeGeneratedRiverSurface(result, !immutableSettledWaterAuthority);
    result.valid = std::isfinite(result.surfaceElevation) && std::isfinite(result.waterSurface) &&
                   std::isfinite(result.discharge) && std::isfinite(result.erosionDepth) &&
                   std::isfinite(result.lakeShoreDistance) &&
                   std::isfinite(result.shoreWaterSurface) &&
                   std::isfinite(result.lakeBankTarget) && std::isfinite(result.waterfallTop) &&
                   std::isfinite(result.waterfallBottom);
    return result;
}

BasinSample finalizeGeneratedWaterContact(
    BasinSample result,
    double unbankedSurfaceElevation = std::numeric_limits<double>::quiet_NaN()) {
    if (!result.valid) return result;

    const bool hasUnbankedSurface = std::isfinite(unbankedSurfaceElevation);
    const double physicalBed = hasUnbankedSurface
                                   ? std::min(unbankedSurfaceElevation, result.surfaceElevation)
                                   : result.surfaceElevation;
    const bool unresolvedSteepOutlet =
        result.transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR &&
        result.channelGradient >= 0.5 && result.shoreWaterSurface > result.waterSurface + 2.5;
    const double incompatibleLakeBankReach =
        unresolvedSteepOutlet ? LAKE_BANK_WIDTH + 8.0 : LAKE_BANK_WIDTH;
    if (!result.lake && !result.waterfall && result.river &&
        result.transitionOwnerKind != WaterTransitionKind::EXPLICIT_FALL &&
        result.transitionOwnerKind != WaterTransitionKind::OUTLET_CORRIDOR &&
        result.transitionOwnerKind != WaterTransitionKind::CHANNEL_GUIDE &&
        result.shoreWaterSurface > result.waterSurface + 2.5 && result.lakeShoreDistance <= 0.0 &&
        result.lakeShoreDistance > -incompatibleLakeBankReach) {
        // A catchment handoff cannot reopen an unrelated low corridor through
        // the side of a standing lake. The exact shared fall remains wet;
        // every other incompatible contact publishes supported rim terrain.
        result.lakeBankTarget = result.shoreWaterSurface + 0.5;
        result.surfaceElevation = std::max(result.surfaceElevation, result.lakeBankTarget);
        result.waterSurface = 0.0;
        result.ocean = false;
        result.river = false;
        result.lakeBank = true;
        result.channelBank = false;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
    }
    const bool submergedLegacyChannelGuard =
        result.channelBank && !result.lakeBank && hasUnbankedSurface &&
        result.surfaceElevation < SEA_LEVEL && physicalBed < SEA_LEVEL;
    if (submergedLegacyChannelGuard) {
        // A raster-side guard below the sea surface cannot remain dry. Its
        // neighboring solution supplies the same physical ocean bed, while
        // the exact settled field retains every above-water bank needed to
        // isolate an elevated route. Reopen only this obsolete submerged
        // guard so it cannot become a straight source-water floor wall.
        result.channelBank = false;
        result.surfaceElevation = physicalBed;
    }
    const bool submergedRoutedBackwater =
        !result.lake && !result.waterfall && !result.lakeBank && !result.channelBank &&
        result.river &&
        (result.transitionOwnerKind == WaterTransitionKind::CHANNEL_GUIDE ||
         result.transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR) &&
        physicalBed < SEA_LEVEL;
    if (submergedRoutedBackwater) {
        result.waterSurface = SEA_LEVEL;
        result.ocean = true;
        result.river = false;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
    }
    if (!result.lake && !result.waterfall && !result.lakeBank && !result.channelBank &&
        physicalBed < SEA_LEVEL && !result.river) {
        result.ocean = true;
        result.river = false;
        result.waterSurface = SEA_LEVEL;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
    }
    if (result.ocean) {
        if (hasUnbankedSurface) {
            result.surfaceElevation =
                std::min(unbankedSurfaceElevation, static_cast<double>(SEA_LEVEL) - 0.125);
        } else {
            result.surfaceElevation =
                std::min(result.surfaceElevation, static_cast<double>(SEA_LEVEL) - 0.125);
        }
        result.channelDepth = std::max(0.125, result.waterSurface - result.surfaceElevation);
        result.lakeBank = false;
        result.channelBank = false;
    }
    return result;
}

template <typename SolutionAt>
BasinSample sampleAcrossCatchmentFaces(BasinKey primary, double x, double z,
                                       const ShorelineContourSample* shoreline,
                                       SolutionAt&& solutionAt) {
    constexpr double BLEND_HALF_WIDTH = RASTER_APRON * BASIN_RASTER_SPACING;
    struct AxisBlend {
        int64_t low = 0;
        int64_t high = 0;
        double amount = 0.0;
    };
    const auto axisBlend = [](double coordinate, int64_t cell) {
        const double origin = static_cast<double>(cell) * BASIN_CATCHMENT_EDGE;
        const double local = coordinate - origin;
        AxisBlend result{cell, cell, 0.0};
        if (local < BLEND_HALF_WIDTH) {
            result.low = cell - 1;
            result.amount = quinticWeight((local + BLEND_HALF_WIDTH) / (BLEND_HALF_WIDTH * 2.0));
        } else if (BASIN_CATCHMENT_EDGE - local < BLEND_HALF_WIDTH) {
            result.high = cell + 1;
            const double signedDistance = local - BASIN_CATCHMENT_EDGE;
            result.amount =
                quinticWeight((signedDistance + BLEND_HALF_WIDTH) / (BLEND_HALF_WIDTH * 2.0));
        }
        return result;
    };

    const AxisBlend xBlend = axisBlend(x, primary.x);
    const AxisBlend zBlend = axisBlend(z, primary.z);
    const std::array<BasinKey, 4> keys = {
        BasinKey{xBlend.low, zBlend.low}, BasinKey{xBlend.high, zBlend.low},
        BasinKey{xBlend.low, zBlend.high}, BasinKey{xBlend.high, zBlend.high}};
    const std::array<double, 4> weights = {
        (1.0 - xBlend.amount) * (1.0 - zBlend.amount), xBlend.amount * (1.0 - zBlend.amount),
        (1.0 - xBlend.amount) * zBlend.amount, xBlend.amount * zBlend.amount};

    BasinSample result = sampleSolution(solutionAt(primary), x, z, shoreline);
    if (xBlend.low == xBlend.high && zBlend.low == zBlend.high) {
        return finalizeGeneratedWaterContact(result);
    }

    std::array<BasinSample, 4> samples;
    for (size_t index = 0; index < samples.size(); ++index) {
        samples[index] = keys[index] == primary
                             ? result
                             : sampleSolution(solutionAt(keys[index]), x, z, shoreline);
        if (!samples[index].valid) return finalizeGeneratedWaterContact(result);
    }
    const bool anyLake =
        std::ranges::any_of(samples, [](const BasinSample& sample) { return sample.lake; });
    const bool anyWaterfall =
        std::ranges::any_of(samples, [](const BasinSample& sample) { return sample.waterfall; });
    if (anyLake) {
        const WaterBodyId body = std::ranges::find_if(samples, [](const BasinSample& sample) {
                                     return sample.lake;
                                 })->waterBodyId;
        if (std::ranges::any_of(samples, [&](const BasinSample& sample) {
                return !sample.lake || sample.waterBodyId != body;
            })) {
            // Lake geometry is owned by its signed contour and one flat
            // WaterBodyId surface. At its globally owned outlet, select the
            // same immutable corridor or fall from both catchments instead of
            // returning whichever basin happened to be primary.
            const BasinSample* authority = nullptr;
            if (shoreline != nullptr && shoreline->valid && shoreline->outletContinuation) {
                for (size_t index = 0; index < samples.size(); ++index) {
                    const BasinSample& candidate = samples[index];
                    if (weights[index] <= 0.0 || candidate.lake ||
                        candidate.transitionOwnerId == 0 ||
                        (!candidate.river && !candidate.waterfall) ||
                        !ownedLakeConnectorAt(solutionAt(keys[index]), shoreline->identity, x, z)) {
                        continue;
                    }
                    const auto rank = [](const BasinSample& sample) {
                        return std::tuple{sample.transitionOwnerKind, -sample.discharge,
                                          sample.transitionOwnerId};
                    };
                    if (authority == nullptr || rank(candidate) < rank(*authority)) {
                        authority = &candidate;
                    }
                }
            }
            if (authority == nullptr && shoreline != nullptr && shoreline->valid &&
                shoreline->signedDistance > 0.0) {
                for (size_t index = 0; index < samples.size(); ++index) {
                    const BasinSample& candidate = samples[index];
                    if (weights[index] <= 0.0 || !candidate.lake ||
                        candidate.waterBodyId != shoreline->identity) {
                        continue;
                    }
                    if (authority == nullptr ||
                        std::pair{keys[index].x, keys[index].z} <
                            std::pair{keys[static_cast<size_t>(authority - samples.data())].x,
                                      keys[static_cast<size_t>(authority - samples.data())].z}) {
                        authority = &candidate;
                    }
                }
            }
            if (authority != nullptr) return finalizeGeneratedWaterContact(*authority);
            return finalizeGeneratedWaterContact(result);
        }
    }
    if (anyWaterfall &&
        std::ranges::any_of(samples, [](const BasinSample& sample) { return !sample.waterfall; })) {
        // Tagged falls are the only intentionally sharp water surface. Keep
        // their globally owned curtain instead of numerically smearing it
        // across a catchment face or changing ownership with the primary key.
        const BasinSample* authority = nullptr;
        for (const BasinSample& candidate : samples) {
            if (!candidate.waterfall || candidate.transitionOwnerId == 0) continue;
            if (authority == nullptr || candidate.discharge > authority->discharge + 1.0e-6 ||
                (std::abs(candidate.discharge - authority->discharge) <= 1.0e-6 &&
                 candidate.transitionOwnerId < authority->transitionOwnerId)) {
                authority = &candidate;
            }
        }
        if (authority != nullptr) return finalizeGeneratedWaterContact(*authority);
        return finalizeGeneratedWaterContact(result);
    }

    bool canonicalTopologySelected = false;
    bool canonicalBodyConnectorSelected = false;
    double canonicalBedElevation = std::numeric_limits<double>::quiet_NaN();
    if (!anyLake && !anyWaterfall) {
        const BasinSample* topologyAuthority = nullptr;
        for (size_t index = 0; index < samples.size(); ++index) {
            const BasinSample& candidate = samples[index];
            if (weights[index] <= 0.0 || candidate.transitionOwnerId == 0 ||
                (!candidate.ocean && !candidate.river) ||
                candidate.waterSurface <= candidate.surfaceElevation + 0.05) {
                continue;
            }
            const bool lowerStage =
                topologyAuthority == nullptr ||
                candidate.waterSurface < topologyAuthority->waterSurface - 1.0e-6;
            const bool equalStage =
                topologyAuthority != nullptr &&
                std::abs(candidate.waterSurface - topologyAuthority->waterSurface) <= 1.0e-6;
            const bool strongerEqualStage =
                equalStage && candidate.discharge > topologyAuthority->discharge + 1.0e-6;
            const bool stableEqualStage =
                equalStage &&
                std::abs(candidate.discharge - topologyAuthority->discharge) <= 1.0e-6 &&
                std::pair{candidate.transitionOwnerKind, candidate.transitionOwnerId} <
                    std::pair{topologyAuthority->transitionOwnerKind,
                              topologyAuthority->transitionOwnerId};
            if (lowerStage || strongerEqualStage || stableEqualStage) {
                topologyAuthority = &candidate;
            }
        }
        if (topologyAuthority != nullptr) {
            canonicalTopologySelected = true;
            const size_t topologyIndex = static_cast<size_t>(topologyAuthority - samples.data());
            canonicalBodyConnectorSelected =
                shoreline != nullptr && shoreline->valid && shoreline->outletContinuation &&
                topologyAuthority->transitionOwnerKind == WaterTransitionKind::OUTLET_CORRIDOR &&
                ownedLakeConnectorAt(solutionAt(keys[topologyIndex]), shoreline->identity, x, z);
            canonicalBedElevation = topologyAuthority->surfaceElevation;
            result.waterSurface = topologyAuthority->waterSurface;
            result.flowX = topologyAuthority->flowX;
            result.flowZ = topologyAuthority->flowZ;
            result.discharge = topologyAuthority->discharge;
            result.sediment = topologyAuthority->sediment;
            result.channelDistance = topologyAuthority->channelDistance;
            result.channelWidth = topologyAuthority->channelWidth;
            result.channelDepth = topologyAuthority->channelDepth;
            result.channelGradient = topologyAuthority->channelGradient;
            result.erosionDepth = topologyAuthority->erosionDepth;
            result.generatedFluidLevel = topologyAuthority->generatedFluidLevel;
            result.streamOrder = topologyAuthority->streamOrder;
            result.outlet = topologyAuthority->outlet;
            result.outletX = topologyAuthority->outletX;
            result.outletZ = topologyAuthority->outletZ;
            result.ocean = topologyAuthority->ocean;
            result.river = topologyAuthority->river;
            result.lakeBank = topologyAuthority->lakeBank;
            result.channelBank = topologyAuthority->channelBank;
            result.transitionOwnerKind = topologyAuthority->transitionOwnerKind;
            result.transitionOwnerId = topologyAuthority->transitionOwnerId;
        }
    }

    const auto blend = [&](auto getter) {
        double value = 0.0;
        for (size_t index = 0; index < samples.size(); ++index)
            value += getter(samples[index]) * weights[index];
        return value;
    };
    double exactChannelBankTarget = -std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < samples.size(); ++index) {
        if (weights[index] <= 0.0 || !samples[index].channelBank ||
            !samples[index].channelBankSubstrateValid) {
            continue;
        }
        exactChannelBankTarget = std::max(exactChannelBankTarget, samples[index].surfaceElevation);
    }
    result.surfaceElevation =
        blend([](const BasinSample& sample) { return sample.surfaceElevation; });
    double unbankedSurfaceElevation = 0.0;
    double unbankedWeight = 0.0;
    for (size_t index = 0; index < samples.size(); ++index) {
        if (samples[index].lakeBank) continue;
        if (samples[index].channelBank) {
            if (!samples[index].channelBankSubstrateValid) continue;
            unbankedSurfaceElevation += samples[index].channelBankSubstrate * weights[index];
            unbankedWeight += weights[index];
            continue;
        }
        unbankedSurfaceElevation += samples[index].surfaceElevation * weights[index];
        unbankedWeight += weights[index];
    }
    if (unbankedWeight > 1.0e-9) unbankedSurfaceElevation /= unbankedWeight;
    if (canonicalTopologySelected && (result.ocean || result.river)) {
        // The canonical water owner also owns its immutable physical bed.
        // Blending that bed and then applying a generic support floor can
        // lower a valid rapid below sea level, causing the shared finalizer
        // to relabel the partial state as ocean source water.
        result.surfaceElevation = canonicalBedElevation;
        result.channelDepth =
            std::max(result.channelDepth, result.waterSurface - result.surfaceElevation);
    }
    if (canonicalTopologySelected && !canonicalBodyConnectorSelected &&
        std::isfinite(exactChannelBankTarget) &&
        exactChannelBankTarget > result.waterSurface + 0.05) {
        // A settled bank is exact block authority at a cross-catchment
        // contact. The lower canonical water stage must not overwrite that
        // retaining wall merely because the neighboring solution reports an
        // ocean or routed channel at the same coordinate.
        result.surfaceElevation = std::max(result.surfaceElevation, exactChannelBankTarget);
        result.waterSurface = 0.0;
        result.ocean = false;
        result.river = false;
        result.lake = false;
        result.lakeBank = false;
        result.channelBank = true;
        result.endorheic = false;
        result.lakeDepth = 0.0;
        result.waterBodyId = NO_WATER_BODY;
        result.generatedFluidLevel = 0;
        result.transitionOwnerKind = WaterTransitionKind::NONE;
        result.transitionOwnerId = 0;
        canonicalTopologySelected = false;
    }
    if (!anyLake && !anyWaterfall) {
        // Hydrology has one canonical catchment owner at every coordinate.
        // Blend terrain fields across the apron, but never renormalize a tiny
        // wet observation from a neighboring catchment into full river
        // ownership when the primary solution is dry. Shared portals and
        // analytical guides already make genuine cross-face routes agree.
        const bool supportedWater =
            (result.ocean || result.river) && result.waterSurface > result.surfaceElevation + 0.05;
        result.lake = false;
        result.endorheic = false;
        result.lakeDepth = 0.0;
        result.waterBodyId = NO_WATER_BODY;
        result.ocean = supportedWater && result.ocean;
        result.river = supportedWater && result.river;
        if (supportedWater) {
            result.lakeBank = false;
            result.channelBank = false;
            result.lakeBankInfluence = 0.0;
            result.lakeBankTarget = 0.0;
        } else {
            result.generatedFluidLevel = 0;
            if (result.channelBank && std::isfinite(exactChannelBankTarget)) {
                // Exact settled banks are physical boundary authority, not a
                // scalar terrain field. Blending their freeboard down toward
                // another catchment's substrate opens a cave-backed drain
                // beside deep routed water.
                result.surfaceElevation = std::max(result.surfaceElevation, exactChannelBankTarget);
            }
        }
    } else {
        result.waterSurface = blend([](const BasinSample& sample) { return sample.waterSurface; });
    }
    if (!canonicalTopologySelected) {
        result.discharge = blend([](const BasinSample& sample) { return sample.discharge; });
        result.sediment = blend([](const BasinSample& sample) { return sample.sediment; });
        result.channelDistance =
            blend([](const BasinSample& sample) { return sample.channelDistance; });
        result.channelWidth = blend([](const BasinSample& sample) { return sample.channelWidth; });
        result.channelDepth = blend([](const BasinSample& sample) { return sample.channelDepth; });
        result.channelGradient =
            blend([](const BasinSample& sample) { return sample.channelGradient; });
        result.erosionDepth = blend([](const BasinSample& sample) { return sample.erosionDepth; });
    }
    if (result.lake) {
        result.lakeDepth = std::max(0.0, result.waterSurface - result.surfaceElevation);
        // Equilibrium values describe the whole WaterBodyId. They are not
        // spatial fields and must never be interpolated at a catchment face.
        // Choose the lexicographically first contributing observation so the
        // result is independent of which side initiated the sample.
        size_t authority = 0;
        for (size_t index = 1; index < samples.size(); ++index) {
            if (weights[index] > 0.0 && (weights[authority] <= 0.0 ||
                                         std::pair{keys[index].x, keys[index].z} <
                                             std::pair{keys[authority].x, keys[authority].z})) {
                authority = index;
            }
        }
        result.lakeAreaSquareKilometers = samples[authority].lakeAreaSquareKilometers;
        result.lakeVolumeCubicMeters = samples[authority].lakeVolumeCubicMeters;
        result.lakeRunoffMmSquareKilometers = samples[authority].lakeRunoffMmSquareKilometers;
        result.lakeLossMm = samples[authority].lakeLossMm;
        result.lakeOverflowMmSquareKilometers = samples[authority].lakeOverflowMmSquareKilometers;
        result.lakeSpillSurface = samples[authority].lakeSpillSurface;
    }
    return finalizeGeneratedWaterContact(result, unbankedWeight > 1.0e-9
                                                     ? unbankedSurfaceElevation
                                                     : std::numeric_limits<double>::quiet_NaN());
}

} // namespace

class BasinSolver::Impl {
public:
    using SolutionPointer = std::shared_ptr<const BasinSolution>;
    using ShorelinePointer = std::shared_ptr<const ShorelineContourPage>;

    struct Entry {
        std::shared_future<SolutionPointer> future;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
        uint64_t token = 0;
    };

    struct ShorelineEntry {
        std::shared_future<ShorelinePointer> future;
        size_t bytes = 0;
        uint64_t lastAccess = 0;
        uint64_t token = 0;
    };

    explicit Impl(uint64_t seed, size_t requestedBudget)
        : random(seed)
        , byteBudget(std::max<size_t>(1, requestedBudget))
        , instanceToken(nextCacheInstanceToken()) {}

    SolutionPointer
    getOrCreate(BasinKey key, const ElevationFunction& elevation, const RainfallFunction& rainfall,
                const RockResistanceFunction& rockResistance,
                const PotentialEvapotranspirationFunction& potentialEvapotranspiration) const {
        struct LocalBasin {
            const Impl* owner = nullptr;
            uint64_t instance = 0;
            uint64_t generation = 0;
            BasinKey key;
            std::weak_ptr<const BasinSolution> solution;
        };
        struct LocalBasinSet {
            std::array<LocalBasin, 4> entries{};
            size_t replacement = 0;
        };
        static thread_local LocalBasinSet local;
        const uint64_t observedGeneration = cacheGeneration.load(std::memory_order_acquire);
        for (const LocalBasin& candidate : local.entries) {
            if (candidate.owner == this && candidate.instance == instanceToken &&
                candidate.generation == observedGeneration && candidate.key == key) {
                const SolutionPointer result = candidate.solution.lock();
                if (!result) continue;
                if (cacheGeneration.load(std::memory_order_acquire) == observedGeneration) {
                    fastHits[fastHitShard()].basins.fetch_add(1, std::memory_order_relaxed);
                    return result;
                }
            }
        }
        const auto remember = [&](SolutionPointer value) {
            LocalBasin* destination = nullptr;
            for (LocalBasin& candidate : local.entries) {
                if (candidate.owner == this && candidate.instance == instanceToken &&
                    candidate.key == key) {
                    destination = &candidate;
                    break;
                }
            }
            if (destination == nullptr) {
                destination = &local.entries[local.replacement++ % local.entries.size()];
            }
            *destination = {
                .owner = this,
                .instance = instanceToken,
                // A clear or eviction that overlaps a cache wait or cold
                // build invalidates that result for future fast hits. Stamp
                // the request generation, not the generation observed after
                // the build, so stale work cannot become current again.
                .generation = observedGeneration,
                .key = key,
                .solution = value,
            };
            return value;
        };

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
        if (!producer) return remember(future.get());

        const auto publish = [&](const SolutionPointer& solution, bool fallback) {
            producer->set_value(solution);
            std::lock_guard lock(mutex);
            auto found = entries.find(key);
            if (found != entries.end() && found->second.token == token) {
                found->second.bytes = solution->byteSize();
                metrics.bytes += found->second.bytes;
                ++metrics.builds;
                if (fallback) {
                    ++metrics.failures;
                    ++metrics.fallbackBuilds;
                } else {
                    metrics.erosionEpochs += solution->erosionEpochs;
                    metrics.erosionReroutes += solution->erosionReroutes;
                    metrics.erosionReceiverChanges += solution->erosionReceiverChanges;
                }
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
                    cacheGeneration.fetch_add(1, std::memory_order_release);
                }
                metrics.entries = entries.size();
            }
            return remember(solution);
        };

        try {
            SolutionPointer solution;
            {
                ColdBasinBuildPermit permit;
                solution = buildSolution(random, key, elevation, rainfall, rockResistance,
                                         potentialEvapotranspiration);
            }
            return publish(solution, false);
        } catch (const std::exception&) {
            try {
                const SolutionPointer fallback =
                    buildFallbackSolution(random, key, elevation, rainfall);
                return publish(fallback, true);
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
    }

    ShorelinePointer getOrCreateShoreline(
        double x, double z, WaterBodyId body, double waterLevel, bool endorheic,
        const ElevationFunction& elevation, const RainfallFunction& rainfall,
        const RockResistanceFunction& rockResistance,
        const PotentialEvapotranspirationFunction& potentialEvapotranspiration) const {
        const auto pageCoordinate = [](double coordinate) {
            return world_coord::floorToNeighborSafeInt64(coordinate / SHORELINE_PAGE_EDGE);
        };
        const ShorelinePageKey key{body, pageCoordinate(x), pageCoordinate(z)};
        struct LocalShoreline {
            const Impl* owner = nullptr;
            uint64_t instance = 0;
            uint64_t generation = 0;
            ShorelinePageKey key;
            std::weak_ptr<const ShorelineContourPage> page;
        };
        static thread_local LocalShoreline local;
        const uint64_t observedGeneration = cacheGeneration.load(std::memory_order_acquire);
        if (local.owner == this && local.instance == instanceToken &&
            local.generation == observedGeneration && local.key == key) {
            const ShorelinePointer result = local.page.lock();
            if (result && cacheGeneration.load(std::memory_order_acquire) == observedGeneration) {
                fastHits[fastHitShard()].shorelines.fetch_add(1, std::memory_order_relaxed);
                return result;
            }
        }
        const auto remember = [&](ShorelinePointer value) {
            local = {
                .owner = this,
                .instance = instanceToken,
                .generation = observedGeneration,
                .key = key,
                .page = value,
            };
            return value;
        };

        std::shared_future<ShorelinePointer> future;
        std::shared_ptr<std::promise<ShorelinePointer>> producer;
        uint64_t token = 0;
        {
            std::lock_guard lock(mutex);
            auto found = shorelineEntries.find(key);
            if (found != shorelineEntries.end()) {
                found->second.lastAccess = ++shorelineAccessClock;
                ++metrics.shorelineHits;
                future = found->second.future;
            } else {
                ++metrics.shorelineMisses;
                producer = std::make_shared<std::promise<ShorelinePointer>>();
                future = producer->get_future().share();
                token = ++shorelineTokenClock;
                shorelineEntries.emplace(key,
                                         ShorelineEntry{future, 0, ++shorelineAccessClock, token});
            }
        }
        if (!producer) return remember(future.get());

        try {
            std::unordered_map<BasinKey, SolutionPointer, BasinKeyHash> localSolutions;
            localSolutions.reserve(4);
            const auto authorityAt = [&](double worldX, double worldZ) {
                const auto solutionAt = [&](BasinKey basin) -> const BasinSolution& {
                    auto found = localSolutions.find(basin);
                    if (found == localSolutions.end()) {
                        found = localSolutions
                                    .emplace(basin,
                                             getOrCreate(basin, elevation, rainfall, rockResistance,
                                                         potentialEvapotranspiration))
                                    .first;
                    }
                    return *found->second;
                };
                SelectedLakeAuthority selected =
                    selectLakeAuthority(worldX, worldZ, elevation, solutionAt, body);
                if (selected.found) return selected.sample;
                LakeAuthoritySample missing;
                missing.floor = elevation(worldX, worldZ);
                return missing;
            };
            ShorelinePointer page = buildShorelinePage(key, waterLevel, endorheic, authorityAt);
            producer->set_value(page);
            std::lock_guard lock(mutex);
            auto found = shorelineEntries.find(key);
            if (found != shorelineEntries.end() && found->second.token == token) {
                found->second.bytes = page->byteSize();
                metrics.shorelineBytes += found->second.bytes;
                ++metrics.shorelineBuilds;
                while (metrics.shorelineBytes > SHORELINE_CACHE_BYTE_BUDGET &&
                       shorelineEntries.size() > 1) {
                    auto oldest = shorelineEntries.end();
                    for (auto iterator = shorelineEntries.begin();
                         iterator != shorelineEntries.end(); ++iterator) {
                        if (iterator->first == key || iterator->second.bytes == 0) continue;
                        if (oldest == shorelineEntries.end() ||
                            iterator->second.lastAccess < oldest->second.lastAccess) {
                            oldest = iterator;
                        }
                    }
                    if (oldest == shorelineEntries.end()) break;
                    metrics.shorelineBytes -= oldest->second.bytes;
                    shorelineEntries.erase(oldest);
                    cacheGeneration.fetch_add(1, std::memory_order_release);
                }
                metrics.shorelineEntries = shorelineEntries.size();
            }
            return remember(page);
        } catch (...) {
            producer->set_exception(std::current_exception());
            std::lock_guard lock(mutex);
            auto found = shorelineEntries.find(key);
            if (found != shorelineEntries.end() && found->second.token == token) {
                shorelineEntries.erase(found);
            }
            ++metrics.shorelineFailures;
            metrics.shorelineEntries = shorelineEntries.size();
            throw;
        }
    }

    mutable std::mutex mutex;
    mutable std::unordered_map<BasinKey, Entry, BasinKeyHash> entries;
    mutable std::unordered_map<ShorelinePageKey, ShorelineEntry, ShorelinePageKeyHash>
        shorelineEntries;
    CounterRng random;
    size_t byteBudget = BASIN_CACHE_BYTE_BUDGET;
    mutable uint64_t accessClock = 0;
    mutable uint64_t tokenClock = 0;
    mutable uint64_t shorelineAccessClock = 0;
    mutable uint64_t shorelineTokenClock = 0;
    mutable BasinCacheMetrics metrics;
    mutable std::atomic<uint64_t> scalarSampleCalls{0};
    const uint64_t instanceToken = 0;
    mutable std::atomic<uint64_t> cacheGeneration{1};
    struct alignas(64) FastHitCounters {
        std::atomic<uint64_t> basins{0};
        std::atomic<uint64_t> shorelines{0};
    };
    mutable std::array<FastHitCounters, 16> fastHits{};
};

BasinSolver::BasinSolver(uint64_t worldSeed, size_t cacheByteBudget)
    : impl_(std::make_unique<Impl>(worldSeed, cacheByteBudget)) {}

BasinSolver::~BasinSolver() = default;
BasinSolver::BasinSolver(BasinSolver&&) noexcept = default;
BasinSolver& BasinSolver::operator=(BasinSolver&&) noexcept = default;

BasinSample
BasinSolver::sample(double x, double z, const ElevationFunction& elevation,
                    const RainfallFunction& rainfall, const RockResistanceFunction& rockResistance,
                    const PotentialEvapotranspirationFunction& potentialEvapotranspiration) const {
    impl_->scalarSampleCalls.fetch_add(1, std::memory_order_relaxed);
    try {
        std::unordered_map<BasinKey, Impl::SolutionPointer, BasinKeyHash> retainedSolutions;
        retainedSolutions.reserve(9);
        const auto solutionFor = [&](BasinKey candidateKey) -> const Impl::SolutionPointer& {
            auto [entry, inserted] = retainedSolutions.try_emplace(candidateKey);
            if (inserted) {
                try {
                    entry->second = impl_->getOrCreate(candidateKey, elevation, rainfall,
                                                       rockResistance, potentialEvapotranspiration);
                } catch (...) {
                    retainedSolutions.erase(entry);
                    throw;
                }
            }
            return entry->second;
        };
        const auto sampleRaw = [&](double sampleX, double sampleZ) {
            const BasinKey sampleKey = samplingKey(sampleX, sampleZ);
            const auto solutionAt = [&](BasinKey candidateKey) -> const BasinSolution& {
                return *solutionFor(candidateKey);
            };
            const SelectedLakeAuthority authority =
                selectLakeAuthority(sampleX, sampleZ, elevation, solutionAt);
            ShorelineContourSample contour;
            if (authority.exteriorOnly) {
                contour = exteriorLakeContour(authority, sampleKey);
            } else if (authority.found && std::abs(authority.sample.signedDistance) <=
                                              LAKE_SHORE_PERTURBATION_BAND + 4.0) {
                const auto page = impl_->getOrCreateShoreline(
                    sampleX, sampleZ, authority.sample.identity, authority.sample.waterLevel,
                    authority.sample.endorheic, elevation, rainfall, rockResistance,
                    potentialEvapotranspiration);
                contour = sampleShorelinePage(*page, sampleX, sampleZ);
            } else {
                contour = exteriorLakeContour(authority, sampleKey);
            }
            if (contour.valid) contour.outletContinuation = authority.connector;
            return sampleAcrossCatchmentFaces(sampleKey, sampleX, sampleZ,
                                              contour.valid ? &contour : nullptr, solutionAt);
        };
        BasinSample result = sampleRaw(x, z);
        if (result.valid) return result;
    } catch (const std::exception&) {
    }
    return fallbackSample(impl_->random, x, z, elevation, rainfall);
}

void BasinSolver::sampleGrid(
    int64_t originX, int64_t originZ, int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
    const ElevationFunction& elevation, const RainfallFunction& rainfall,
    const RockResistanceFunction& rockResistance, std::span<BasinSample> output,
    const PotentialEvapotranspirationFunction& potentialEvapotranspiration) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth * sampleHeight)) {
        throw std::invalid_argument("invalid basin sample grid");
    }

    std::unordered_map<BasinKey, Impl::SolutionPointer, BasinKeyHash> retainedSolutions;
    retainedSolutions.reserve(16);
    SparseShorelineNodes retainedShorelineNodes;
    retainedShorelineNodes.reserve(static_cast<size_t>(sampleWidth * sampleHeight));
    const auto solutionFor = [&](BasinKey key) -> const Impl::SolutionPointer& {
        auto [entry, inserted] = retainedSolutions.try_emplace(key);
        if (inserted) {
            try {
                entry->second = impl_->getOrCreate(key, elevation, rainfall, rockResistance,
                                                   potentialEvapotranspiration);
            } catch (...) {
                retainedSolutions.erase(entry);
                throw;
            }
        }
        return entry->second;
    };
    const auto sampleRaw = [&](double x, double z) {
        try {
            const BasinKey key = samplingKey(x, z);
            const auto solutionAt = [&](BasinKey candidateKey) -> const BasinSolution& {
                return *solutionFor(candidateKey);
            };
            const SelectedLakeAuthority authority =
                selectLakeAuthority(x, z, elevation, solutionAt);

            ShorelineContourSample contour;
            if (authority.exteriorOnly) {
                contour = exteriorLakeContour(authority, key);
            } else if (authority.found && std::abs(authority.sample.signedDistance) <=
                                              LAKE_SHORE_PERTURBATION_BAND + 4.0) {
                const LakeAuthorityFunction authorityAt =
                    [&, body = authority.sample.identity](double worldX, double worldZ) {
                        const auto contourSolutionAt = [&](BasinKey basin) -> const BasinSolution& {
                            return *solutionFor(basin);
                        };
                        SelectedLakeAuthority selected =
                            selectLakeAuthority(worldX, worldZ, elevation, contourSolutionAt, body);
                        if (selected.found) return selected.sample;
                        LakeAuthoritySample missing;
                        missing.floor = elevation(worldX, worldZ);
                        return missing;
                    };
                contour = sampleSparseShoreline(
                    authority.sample.identity, authority.sample.waterLevel,
                    authority.sample.endorheic, x, z, authorityAt, retainedShorelineNodes);
            } else {
                contour = exteriorLakeContour(authority, key);
            }
            if (contour.valid) contour.outletContinuation = authority.connector;
            const BasinSample result = sampleAcrossCatchmentFaces(
                key, x, z, contour.valid ? &contour : nullptr, solutionAt);
            if (result.valid) return result;
        } catch (const std::exception&) {
        }
        return fallbackSample(impl_->random, x, z, elevation, rainfall);
    };
    for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
        for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
            const double x =
                static_cast<double>(originX + static_cast<int64_t>(sampleX) * spacingX);
            const double z =
                static_cast<double>(originZ + static_cast<int64_t>(sampleZ) * spacingZ);
            output[static_cast<size_t>(sampleZ * sampleWidth + sampleX)] = sampleRaw(x, z);
        }
    }
}

void BasinSolver::samplePoints(
    std::span<const BasinSamplePosition> positions, const ElevationFunction& elevation,
    const RainfallFunction& rainfall, const RockResistanceFunction& rockResistance,
    std::span<BasinSample> output,
    const PotentialEvapotranspirationFunction& potentialEvapotranspiration) const {
    if (positions.size() != output.size()) {
        throw std::invalid_argument("invalid basin sample points");
    }

    std::unordered_map<BasinKey, Impl::SolutionPointer, BasinKeyHash> retainedSolutions;
    retainedSolutions.reserve(16);
    SparseShorelineNodes retainedShorelineNodes;
    retainedShorelineNodes.reserve(positions.size());
    const auto solutionFor = [&](BasinKey key) -> const Impl::SolutionPointer& {
        auto [entry, inserted] = retainedSolutions.try_emplace(key);
        if (inserted) {
            try {
                entry->second = impl_->getOrCreate(key, elevation, rainfall, rockResistance,
                                                   potentialEvapotranspiration);
            } catch (...) {
                retainedSolutions.erase(entry);
                throw;
            }
        }
        return entry->second;
    };
    const auto sampleRaw = [&](double x, double z) {
        try {
            const BasinKey key = samplingKey(x, z);
            const auto solutionAt = [&](BasinKey candidateKey) -> const BasinSolution& {
                return *solutionFor(candidateKey);
            };
            const SelectedLakeAuthority authority =
                selectLakeAuthority(x, z, elevation, solutionAt);

            ShorelineContourSample contour;
            if (authority.exteriorOnly) {
                contour = exteriorLakeContour(authority, key);
            } else if (authority.found && std::abs(authority.sample.signedDistance) <=
                                              LAKE_SHORE_PERTURBATION_BAND + 4.0) {
                const LakeAuthorityFunction authorityAt =
                    [&, body = authority.sample.identity](double worldX, double worldZ) {
                        const auto contourSolutionAt = [&](BasinKey basin) -> const BasinSolution& {
                            return *solutionFor(basin);
                        };
                        SelectedLakeAuthority selected =
                            selectLakeAuthority(worldX, worldZ, elevation, contourSolutionAt, body);
                        if (selected.found) return selected.sample;
                        LakeAuthoritySample missing;
                        missing.floor = elevation(worldX, worldZ);
                        return missing;
                    };
                contour = sampleSparseShoreline(
                    authority.sample.identity, authority.sample.waterLevel,
                    authority.sample.endorheic, x, z, authorityAt, retainedShorelineNodes);
            } else {
                contour = exteriorLakeContour(authority, key);
            }
            if (contour.valid) contour.outletContinuation = authority.connector;
            const BasinSample result = sampleAcrossCatchmentFaces(
                key, x, z, contour.valid ? &contour : nullptr, solutionAt);
            if (result.valid) return result;
        } catch (const std::exception&) {
        }
        return fallbackSample(impl_->random, x, z, elevation, rainfall);
    };
    for (size_t index = 0; index < positions.size(); ++index) {
        output[index] = sampleRaw(positions[index].x, positions[index].z);
    }
}

BasinCacheMetrics BasinSolver::cacheMetrics() const {
    BasinCacheMetrics result;
    {
        std::lock_guard lock(impl_->mutex);
        result = impl_->metrics;
        result.entries = impl_->entries.size();
        result.shorelineEntries = impl_->shorelineEntries.size();
    }
    for (const auto& shard : impl_->fastHits) {
        result.hits += shard.basins.load(std::memory_order_relaxed);
        result.shorelineHits += shard.shorelines.load(std::memory_order_relaxed);
    }
    result.scalarSampleCalls = impl_->scalarSampleCalls.load(std::memory_order_relaxed);
    const ColdBasinBuildMetrics construction = coldBasinBuildLimiter().metrics();
    result.activeColdBuilds = construction.active;
    result.peakColdBuilds = construction.peak;
    result.throttledBuilds = construction.throttled;
    return result;
}

void BasinSolver::clear() const {
    std::lock_guard lock(impl_->mutex);
    impl_->entries.clear();
    impl_->shorelineEntries.clear();
    impl_->metrics.entries = 0;
    impl_->metrics.bytes = 0;
    impl_->metrics.shorelineEntries = 0;
    impl_->metrics.shorelineBytes = 0;
    impl_->cacheGeneration.fetch_add(1, std::memory_order_release);
}

} // namespace worldgen
