#include "world/native_hydrology.hpp"

#include "world/chunk.hpp"
#include "world/learned_terrain.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace worldgen {

double native_hydrology_detail::receiverAxisProgress(double worldX, double worldZ, double sourceX,
                                                     double sourceZ, double receiverX,
                                                     double receiverZ) noexcept {
    const double directionX = receiverX - sourceX;
    const double directionZ = receiverZ - sourceZ;
    const double runSquared = directionX * directionX + directionZ * directionZ;
    if (!(runSquared > 1.0e-18) || !std::isfinite(runSquared)) return 0.0;
    const double amount =
        ((worldX - sourceX) * directionX + (worldZ - sourceZ) * directionZ) / runSquared;
    return std::isfinite(amount) ? std::clamp(amount, 0.0, 1.0) : 0.0;
}

double
native_hydrology_detail::explicitFallPublishedWaterSurface(bool waterfall, double waterfallBottom,
                                                           double ordinaryWaterSurface) noexcept {
    return waterfall ? waterfallBottom : ordinaryWaterSurface;
}

namespace {

constexpr int RASTER_APRON = 2;
constexpr int INTERIOR_INTERVALS = NATIVE_HYDROLOGY_PAGE_EDGE / NATIVE_HYDROLOGY_RASTER_SPACING;
constexpr int RASTER_EDGE = INTERIOR_INTERVALS + 1 + 2 * RASTER_APRON;
constexpr int RASTER_CELLS = RASTER_EDGE * RASTER_EDGE;
constexpr int CORE_BEGIN = RASTER_APRON;
constexpr int CORE_END = RASTER_APRON + INTERIOR_INTERVALS;
constexpr size_t CORE_EDGE_SAMPLES = static_cast<size_t>(INTERIOR_INTERVALS + 1);
constexpr int TOPOLOGY_CELL_RASTER_EDGE =
    NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE / NATIVE_HYDROLOGY_RASTER_SPACING;
constexpr int TOPOLOGY_EDGE = INTERIOR_INTERVALS / TOPOLOGY_CELL_RASTER_EDGE;
constexpr int TOPOLOGY_CELLS = TOPOLOGY_EDGE * TOPOLOGY_EDGE;
constexpr int64_t LOCAL_LAKE_COARSE_ANCHOR_BLOCKS = 256;
constexpr float ROUTING_ELEVATION_EPSILON_METERS = 1.0e-6F;
constexpr float LAKE_DEPTH_THRESHOLD_METERS =
    0.125F * static_cast<float>(learned::WORLD_METERS_PER_BLOCK);
constexpr float LAKE_COMPONENT_STAGE_EPSILON_METERS =
    0.01F * static_cast<float>(learned::WORLD_METERS_PER_BLOCK);
constexpr float MINIMUM_CHANNEL_DISCHARGE =
    static_cast<float>(NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE);
// Wetland candidates inherit a finished downstream water body's stage and
// identity through a bounded, deterministic native-cell graph. The graph may
// cross page seams, but never raises terrain or invents an isolated stage.
constexpr float WETLAND_MAX_GROUNDWATER_DEPTH_METERS = 1.0F;
constexpr float WETLAND_MAX_GROUNDWATER_DEPTH_BLOCKS =
    WETLAND_MAX_GROUNDWATER_DEPTH_METERS / static_cast<float>(learned::WORLD_METERS_PER_BLOCK);
constexpr float WETLAND_MAX_SEASONALITY = 0.75F;
constexpr float WETLAND_MAX_PARENT_STAGE_OFFSET_BLOCKS = 0.375F;
constexpr float WETLAND_BED_DEPTH_BLOCKS = 0.125F;
constexpr float WETLAND_PARENT_STAGE_EPSILON_BLOCKS = 1.0e-5F;
constexpr size_t MAXIMUM_CONNECTED_WETLAND_PAGES = NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES;
constexpr size_t MAXIMUM_CONNECTED_WETLAND_CELLS = 262'144;
constexpr float RIVER_NATURAL_STAGE_BELOW_TERRAIN_BLOCKS = 0.25F;
constexpr float RIVER_MINIMUM_BED_DEPTH_BLOCKS = 0.125F;
// Exposed ordinary river tops are quantized to eighth-block states. Bounding
// the analytical grade to half that interval prevents ordinary one-block
// movement from skipping a visible level after rounding.
constexpr float RIVER_MAXIMUM_ORDINARY_STAGE_DROP_PER_BLOCK = 0.0625F;
constexpr double MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS = 0.5;
constexpr float MINIMUM_CHANNEL_BRANCH_WEIGHT = 0.08F;
constexpr float WATERFALL_MINIMUM_DROP_METERS =
    4.0F * static_cast<float>(learned::WORLD_METERS_PER_BLOCK);
constexpr float ESTUARY_MAXIMUM_CHANNEL_GRADIENT = 0.02F;
constexpr int ESTUARY_MAXIMUM_BACKWATER_NATIVE_CELLS = 64;
constexpr int DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS = 4;
constexpr double WORLD_SAMPLE_NATIVE_OFFSET =
    0.5 / static_cast<double>(NATIVE_HYDROLOGY_RASTER_SPACING) - 0.5;
constexpr double NATIVE_SAMPLE_WORLD_CENTER_OFFSET =
    (static_cast<double>(NATIVE_HYDROLOGY_RASTER_SPACING) - 1.0) * 0.5;
constexpr uint8_t CELL_OCEAN = 1U << 0U;
constexpr uint8_t CELL_LAKE = 1U << 1U;
constexpr uint8_t CELL_RIVER = 1U << 2U;
constexpr uint8_t CELL_WETLAND = 1U << 3U;
constexpr uint8_t CELL_WATERFALL = 1U << 4U;
constexpr uint8_t CELL_WETLAND_CANDIDATE = 1U << 5U;
constexpr uint8_t CELL_ESTUARY = 1U << 6U;
constexpr uint8_t CELL_DELTA = 1U << 7U;
constexpr int CHANNEL_SEARCH_RADIUS = 8;
constexpr uint8_t TOPOLOGY_WET = 1U << 0U;
constexpr uint8_t TOPOLOGY_DRY = 1U << 1U;
constexpr uint8_t TOPOLOGY_CHANNEL = 1U << 2U;
constexpr uint8_t TOPOLOGY_WATERFALL = 1U << 3U;
constexpr uint8_t TOPOLOGY_AUTHORITY_DISCONTINUITY = 1U << 4U;
constexpr std::array<int, 8> DX{1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 8> DZ{0, 1, 1, 1, 0, -1, -1, -1};
constexpr double CELL_AREA_SQUARE_KILOMETERS = NATIVE_HYDROLOGY_CELL_AREA_SQUARE_KILOMETERS;

static_assert(NATIVE_HYDROLOGY_RASTER_SPACING == 4);
static_assert(NATIVE_HYDROLOGY_PAGE_EDGE % NATIVE_HYDROLOGY_RASTER_SPACING == 0);
static_assert(INTERIOR_INTERVALS % TOPOLOGY_CELL_RASTER_EDGE == 0);
static_assert(learned::LEARNED_SEA_LEVEL == SEA_LEVEL);

float worldHeightFromNativeMeters(double elevationMeters) noexcept {
    return static_cast<float>(learned::learnedElevationMetersToWorldHeight(elevationMeters));
}

int indexOf(int x, int z) {
    return z * RASTER_EDGE + x;
}

bool inside(int x, int z) {
    return x >= 0 && x < RASTER_EDGE && z >= 0 && z < RASTER_EDGE;
}

uint64_t mix64(uint64_t value) {
    value ^= value >> 30U;
    value *= 0xBF58'476D'1CE4'E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

WaterBodyId stableWaterBodyId(uint64_t seed, int64_t x, int64_t z, uint64_t discriminator) {
    uint64_t value = mix64(seed ^ discriminator);
    value = mix64(value ^ static_cast<uint64_t>(x));
    value = mix64(value ^ static_cast<uint64_t>(z));
    return value == NO_WATER_BODY ? 1 : value;
}

struct PageKey {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const PageKey&) const = default;
};

struct PageKeyHash {
    size_t operator()(PageKey key) const noexcept {
        return static_cast<size_t>(mix64(static_cast<uint64_t>(key.x)) ^
                                   std::rotl(mix64(static_cast<uint64_t>(key.z)), 29));
    }
};

size_t nativeHydrologyBuildConcurrency() noexcept {
    const size_t reported = std::thread::hardware_concurrency() == 0
                                ? 1U
                                : static_cast<size_t>(std::thread::hardware_concurrency());
    const size_t memoryBound =
        NATIVE_HYDROLOGY_PARALLEL_BUILD_MEMORY_BUDGET / NATIVE_HYDROLOGY_MAX_BUILD_BYTES;
    return std::max<size_t>(
        1, std::min({reported, NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS, memoryBound}));
}

// The native pages are spatially independent during their Priority-Flood and
// D-infinity solve. A router-wide mutex here used to serialize every cold
// horizon page, leaving nearly all CPU cores idle while far tiles waited. This
// admission gate bounds memory without imposing an artificial single-file
// build order. Per-page flights still keep duplicate requests deterministic.
class NativeHydrologyBuildGate {
public:
    class Lease {
    public:
        Lease() = default;
        explicit Lease(NativeHydrologyBuildGate* gate) : gate_(gate) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : gate_(std::exchange(other.gate_, nullptr)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();
                gate_ = std::exchange(other.gate_, nullptr);
            }
            return *this;
        }
        ~Lease() { release(); }

    private:
        void release() {
            if (gate_ == nullptr) return;
            gate_->release();
            gate_ = nullptr;
        }

        NativeHydrologyBuildGate* gate_ = nullptr;
    };

    [[nodiscard]] Lease acquire(std::atomic<learned::AuthorityRequestPriority>& priority,
                                bool& waited) {
        std::unique_lock lock(mutex_);
        waiters_.push_back(&priority);
        const auto higherPriorityWaiting = [&] {
            const learned::AuthorityRequestPriority requested =
                priority.load(std::memory_order_relaxed);
            return std::ranges::any_of(waiters_, [&](const auto* candidate) {
                return candidate != &priority &&
                       candidate->load(std::memory_order_relaxed) < requested;
            });
        };
        waited = active_ >= limit_ || higherPriorityWaiting();
        ready_.wait(lock, [&] { return active_ < limit_ && !higherPriorityWaiting(); });
        std::erase(waiters_, &priority);
        ++active_;
        return Lease(this);
    }

private:
    void release() {
        {
            std::lock_guard lock(mutex_);
            if (active_ == 0) std::terminate();
            --active_;
        }
        // A low-priority waiter may not be eligible while a later exact
        // waiter is. Wake the set so the strongest request always claims the
        // newly available build slot.
        ready_.notify_all();
    }

    friend class Lease;

    const size_t limit_ = nativeHydrologyBuildConcurrency();
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<std::atomic<learned::AuthorityRequestPriority>*> waiters_;
    size_t active_ = 0;
};

// PREVIEW and FINAL contexts intentionally own separate immutable page
// caches, but their Priority-Flood work competes for the same physical CPU
// and unified-memory bandwidth. One process-wide gate lets the cold horizon
// use all available cores without multiplying that budget for each context.
NativeHydrologyBuildGate& sharedNativeHydrologyBuildGate() {
    static NativeHydrologyBuildGate gate;
    return gate;
}

PageKey pageKeyAt(double x, double z) {
    if (!std::isfinite(x) || !std::isfinite(z))
        throw std::invalid_argument("native hydrology position is not finite");
    const long double pageX = std::floor(static_cast<long double>(x) / NATIVE_HYDROLOGY_PAGE_EDGE);
    const long double pageZ = std::floor(static_cast<long double>(z) / NATIVE_HYDROLOGY_PAGE_EDGE);
    if (pageX < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        pageX > static_cast<long double>(std::numeric_limits<int64_t>::max()) ||
        pageZ < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        pageZ > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("native hydrology page coordinate is out of range");
    }
    return {static_cast<int64_t>(pageX), static_cast<int64_t>(pageZ)};
}

int64_t checkedCoordinate(int64_t page, int rasterOffset) {
    const __int128 value = static_cast<__int128>(page) * NATIVE_HYDROLOGY_PAGE_EDGE +
                           static_cast<__int128>(rasterOffset) * NATIVE_HYDROLOGY_RASTER_SPACING;
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max()) {
        throw std::invalid_argument("native hydrology raster coordinate is out of range");
    }
    return static_cast<int64_t>(value);
}

struct Receiver {
    int32_t first = -1;
    int32_t second = -1;
    float secondWeight = 0.0F;
};

float receiverBranchWeight(const Receiver& receiver, int branch) noexcept {
    if (branch == 0) return receiver.second >= 0 ? 1.0F - receiver.secondWeight : 1.0F;
    return receiver.second >= 0 ? receiver.secondWeight : 0.0F;
}

bool receiverBranchActive(const Receiver& receiver, int branch) noexcept {
    const int target = branch == 0 ? receiver.first : receiver.second;
    return target >= 0 && receiverBranchWeight(receiver, branch) >= MINIMUM_CHANNEL_BRANCH_WEIGHT;
}

struct LakeStats {
    double areaSquareKilometers = 0.0;
    double volumeCubicMeters = 0.0;
    double runoffMmSquareKilometers = 0.0;
};

enum class PageEdge : uint8_t {
    WEST = 0,
    EAST = 1,
    NORTH = 2,
    SOUTH = 3,
};

struct EdgeSummarySample {
    float rawElevation = 0.0F;
    float waterSurface = 0.0F;
    float discharge = 0.0F;
    WaterBodyId waterBodyId = NO_WATER_BODY;
    uint8_t flags = 0;
};

struct DepressionSummary {
    WaterBodyId waterBodyId = NO_WATER_BODY;
    // This anchor and stage describe one local component. Opposing immutable
    // edge summaries form a deterministic tiled spill hierarchy at query
    // time. Statistics use the half-open owner core, never the overlapping
    // apron, so a reconciled component can add them without double counting.
    int64_t localAnchorX = 0;
    int64_t localAnchorZ = 0;
    float localStage = 0.0F;
    double coreAreaSquareKilometers = 0.0;
    double coreVolumeCubicMeters = 0.0;
    double coreRunoffMmSquareKilometers = 0.0;
    int64_t naturalOutletX = 0;
    int64_t naturalOutletZ = 0;
    float naturalOutletStage = 0.0F;
    uint8_t edgeMask = 0;
    BasinOutlet naturalOutlet = BasinOutlet::NONE;

    auto operator<=>(const DepressionSummary&) const = default;
};

struct ReceiverOwnedOutletIndexEntry {
    int32_t source = -1;
    float standingStage = 0.0F;
    WaterBodyId standingBody = NO_WATER_BODY;

    auto operator<=>(const ReceiverOwnedOutletIndexEntry&) const = default;
};

struct OrdinaryStageTile;
using OrdinaryStageTileKey = std::pair<int64_t, int64_t>;

struct OrdinaryStageTileCacheEntry {
    std::shared_ptr<const OrdinaryStageTile> tile;
    size_t bytes = 0;
    std::list<OrdinaryStageTileKey>::iterator recency;
};

struct OrdinaryStageTileFlight {
    std::condition_variable ready;
    std::shared_ptr<const OrdinaryStageTile> tile;
    std::exception_ptr failure;
    bool complete = false;
};

struct NativePage {
    PageKey key;
    int64_t originX = 0;
    int64_t originZ = 0;
    // The physical elevation and local runoff remain persisted because a
    // depression that is open on one page can close only after a bounded
    // minimax solve admits adjacent immutable pages. Reconstructing that
    // solve from game-height truncation or accumulated discharge would make
    // its shoreline and mass depend on which page was requested first.
    std::vector<float> routingElevationMeters;
    std::vector<float> localRunoffMmSquareKilometers;
    // Game-facing vertical fields use Rycraft world-block coordinates.
    std::vector<float> rawElevation;
    std::vector<float> waterSurface;
    std::vector<float> discharge;
    std::vector<float> baseflow;
    std::vector<float> seasonality;
    std::vector<float> groundwaterRecharge;
    std::vector<float> groundwaterHead;
    std::vector<float> flowX;
    std::vector<float> flowZ;
    std::vector<float> channelGradient;   // dimensionless physical grade
    std::vector<float> lakeShoreDistance; // world blocks
    std::vector<int32_t> receiverFirst;
    std::vector<int32_t> receiverSecond;
    std::vector<float> receiverSecondWeight;
    std::vector<WaterBodyId> waterBodyIds;
    std::vector<uint8_t> streamOrder;
    std::vector<uint8_t> flags;
    // Two frozen bits identify which active D-infinity receiver branches own
    // an explicit stage fall. Persisting this partition prevents stage
    // reconciliation from changing reach topology after a reload.
    std::vector<uint8_t> waterfallBranchMasks;
    // Derived from flags after building or loading. This is not persisted:
    // it is an immutable sampling index that avoids a 17-by-17 channel scan
    // for dry points that cannot reach a routed segment.
    std::vector<uint8_t> channelProximity;
    // Source cells are stored row-major so projectChannel can visit the same
    // ordered candidates without inspecting every dry cell in its 17-by-17
    // search neighborhood. Like channelProximity, this is derived only from
    // persisted flags and has no authority or on-disk representation.
    std::vector<uint32_t> channelRowOffsets;
    std::vector<uint16_t> channelSourceXs;
    // Ordinary reach IDs are rebuilt from persisted receiver edges. A routed
    // fall severs its outgoing edge so overlap blending cannot reconnect the
    // upstream and downstream stages of the same water body.
    std::vector<uint32_t> ordinaryReachIds;
    // Persisted depression outlets reduced to their exact routed source cell.
    // Sampling uses binary search instead of scanning every lake summary.
    std::vector<ReceiverOwnedOutletIndexEntry> receiverOwnedOutlets;
    // A 64-by-64 reduction of native water and channel evidence. It is
    // derived after page load as well as page construction, so it does not
    // alter persisted authority or the generator fingerprint.
    std::vector<uint8_t> topologyCells;
    std::array<std::vector<EdgeSummarySample>, 4> edgeSummaries;
    std::vector<DepressionSummary> depressionSummaries;
    std::unordered_map<WaterBodyId, LakeStats> lakeStats;
    WaterBodyId oceanId = NO_WATER_BODY;
    bool hasWetlandCandidates = false;
    bool hasCoastalResolutionCandidates = false;
    // A bit marks an edge whose cheapest page-local escape path rises above
    // at least one boundary port. A dry owner with this bit must inspect the
    // corresponding neighbor because that port can be the continuation of a
    // basin whose true spill lies beyond the local page.
    uint8_t openSpillReceivingEdgeMask = 0;
    mutable std::mutex ordinaryStageTilesMutex;
    mutable std::map<OrdinaryStageTileKey, OrdinaryStageTileCacheEntry> ordinaryStageTiles;
    mutable std::map<OrdinaryStageTileKey, std::shared_ptr<OrdinaryStageTileFlight>>
        ordinaryStageTileFlights;
    mutable std::list<OrdinaryStageTileKey> ordinaryStageTileRecency;
    mutable size_t ordinaryStageTileBytes = 0;
    mutable size_t ordinaryStageTilePeakBytes = 0;
    mutable uint64_t ordinaryStageTileHits = 0;
    mutable uint64_t ordinaryStageTileMisses = 0;
    mutable uint64_t ordinaryStageTileBuilds = 0;
    mutable uint64_t ordinaryStageTileFailures = 0;
    mutable uint64_t ordinaryStageTileBuildNanoseconds = 0;
    mutable uint64_t ordinaryStageTileExpandedBuilds = 0;
    mutable uint64_t ordinaryStageTileBuildWaits = 0;

    size_t byteSize() const {
        size_t result = 0;
        const auto add = [&](const auto& values) {
            result +=
                values.capacity() * sizeof(typename std::decay_t<decltype(values)>::value_type);
        };
        add(routingElevationMeters);
        add(localRunoffMmSquareKilometers);
        add(rawElevation);
        add(waterSurface);
        add(discharge);
        add(baseflow);
        add(seasonality);
        add(groundwaterRecharge);
        add(groundwaterHead);
        add(flowX);
        add(flowZ);
        add(channelGradient);
        add(lakeShoreDistance);
        add(receiverFirst);
        add(receiverSecond);
        add(receiverSecondWeight);
        add(waterBodyIds);
        add(streamOrder);
        add(flags);
        add(waterfallBranchMasks);
        add(channelProximity);
        add(channelRowOffsets);
        add(channelSourceXs);
        add(ordinaryReachIds);
        add(receiverOwnedOutlets);
        add(topologyCells);
        for (const auto& edge : edgeSummaries)
            add(edge);
        add(depressionSummaries);
        result += lakeStats.size() * (sizeof(WaterBodyId) + sizeof(LakeStats) + 32U);
        return result;
    }
};

void buildNativePageSamplingIndexes(NativePage& page);
void buildOrdinaryReachIndex(NativePage& page);
uint8_t computeWaterfallBranchMask(const NativePage& page, size_t sourceCell);
bool waterfallBranchMaskIsValid(const NativePage& page, size_t sourceCell);
void freezeWaterfallBranchMasks(NativePage& page);

void appendU16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8U));
}

void appendU32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void appendU64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void appendFloat(std::vector<uint8_t>& bytes, float value) {
    appendU32(bytes, std::bit_cast<uint32_t>(value));
}

void appendDouble(std::vector<uint8_t>& bytes, double value) {
    appendU64(bytes, std::bit_cast<uint64_t>(value));
}

bool readU16(std::span<const uint8_t> bytes, size_t& offset, uint16_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) return false;
    value = static_cast<uint16_t>(bytes[offset]) |
            static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8U);
    offset += sizeof(value);
    return true;
}

bool readU32(std::span<const uint8_t> bytes, size_t& offset, uint32_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) return false;
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    return true;
}

bool readU64(std::span<const uint8_t> bytes, size_t& offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    return true;
}

bool readFloat(std::span<const uint8_t> bytes, size_t& offset, float& value) {
    uint32_t encoded = 0;
    if (!readU32(bytes, offset, encoded)) return false;
    value = std::bit_cast<float>(encoded);
    return true;
}

bool readDouble(std::span<const uint8_t> bytes, size_t& offset, double& value) {
    uint64_t encoded = 0;
    if (!readU64(bytes, offset, encoded)) return false;
    value = std::bit_cast<double>(encoded);
    return true;
}

template <typename Value, typename Append>
void appendVector(std::vector<uint8_t>& bytes, const std::vector<Value>& values, Append&& append) {
    for (const Value value : values)
        append(bytes, value);
}

template <typename Value, typename Read>
bool readVector(std::span<const uint8_t> bytes, size_t& offset, size_t count,
                std::vector<Value>& values, Read&& read) {
    values.resize(count);
    for (Value& value : values) {
        if (!read(bytes, offset, value)) return false;
    }
    return true;
}

std::vector<uint8_t> serializePage(const NativePage& page) {
    constexpr uint16_t HEADER_BYTES = 52;
    std::vector<uint8_t> bytes;
    bytes.reserve(HEADER_BYTES + page.byteSize());
    bytes.insert(bytes.end(), {'N', 'H', '4', 'P'});
    appendU16(bytes, NATIVE_HYDROLOGY_PAYLOAD_SCHEMA_VERSION);
    appendU16(bytes, HEADER_BYTES);
    appendU64(bytes, static_cast<uint64_t>(page.key.x));
    appendU64(bytes, static_cast<uint64_t>(page.key.z));
    appendU16(bytes, NATIVE_HYDROLOGY_RASTER_SPACING);
    appendU16(bytes, NATIVE_HYDROLOGY_PAGE_EDGE);
    appendU16(bytes, RASTER_APRON);
    appendU16(bytes, RASTER_EDGE);
    appendU32(bytes, RASTER_CELLS);
    appendU32(bytes, static_cast<uint32_t>(CORE_EDGE_SAMPLES));
    appendU32(bytes, static_cast<uint32_t>(page.lakeStats.size()));
    appendU32(bytes, static_cast<uint32_t>(page.depressionSummaries.size()));
    appendU32(bytes, page.openSpillReceivingEdgeMask);
    if (bytes.size() != HEADER_BYTES)
        throw std::runtime_error("native hydrology payload header construction failed");

    const auto appendFloats = [&](const std::vector<float>& values) {
        appendVector(bytes, values, appendFloat);
    };
    appendFloats(page.routingElevationMeters);
    appendFloats(page.localRunoffMmSquareKilometers);
    appendFloats(page.rawElevation);
    appendFloats(page.waterSurface);
    appendFloats(page.discharge);
    appendFloats(page.baseflow);
    appendFloats(page.seasonality);
    appendFloats(page.groundwaterRecharge);
    appendFloats(page.groundwaterHead);
    appendFloats(page.flowX);
    appendFloats(page.flowZ);
    appendFloats(page.channelGradient);
    appendFloats(page.lakeShoreDistance);
    appendVector(bytes, page.receiverFirst, [](std::vector<uint8_t>& output, int32_t value) {
        appendU32(output, static_cast<uint32_t>(value));
    });
    appendVector(bytes, page.receiverSecond, [](std::vector<uint8_t>& output, int32_t value) {
        appendU32(output, static_cast<uint32_t>(value));
    });
    appendFloats(page.receiverSecondWeight);
    appendVector(bytes, page.waterBodyIds, appendU64);
    bytes.insert(bytes.end(), page.streamOrder.begin(), page.streamOrder.end());
    bytes.insert(bytes.end(), page.flags.begin(), page.flags.end());
    bytes.insert(bytes.end(), page.waterfallBranchMasks.begin(), page.waterfallBranchMasks.end());
    std::vector<std::pair<WaterBodyId, LakeStats>> lakeStats(page.lakeStats.begin(),
                                                             page.lakeStats.end());
    std::ranges::sort(lakeStats, {}, &decltype(lakeStats)::value_type::first);
    for (const auto& [body, stats] : lakeStats) {
        appendU64(bytes, body);
        appendDouble(bytes, stats.areaSquareKilometers);
        appendDouble(bytes, stats.volumeCubicMeters);
        appendDouble(bytes, stats.runoffMmSquareKilometers);
    }
    for (const DepressionSummary& summary : page.depressionSummaries) {
        appendU64(bytes, summary.waterBodyId);
        appendU64(bytes, static_cast<uint64_t>(summary.localAnchorX));
        appendU64(bytes, static_cast<uint64_t>(summary.localAnchorZ));
        appendFloat(bytes, summary.localStage);
        appendDouble(bytes, summary.coreAreaSquareKilometers);
        appendDouble(bytes, summary.coreVolumeCubicMeters);
        appendDouble(bytes, summary.coreRunoffMmSquareKilometers);
        appendU64(bytes, static_cast<uint64_t>(summary.naturalOutletX));
        appendU64(bytes, static_cast<uint64_t>(summary.naturalOutletZ));
        appendFloat(bytes, summary.naturalOutletStage);
        bytes.push_back(summary.edgeMask);
        bytes.push_back(static_cast<uint8_t>(summary.naturalOutlet));
        bytes.insert(bytes.end(), 2, 0);
    }
    for (const auto& edge : page.edgeSummaries) {
        for (const EdgeSummarySample& sample : edge) {
            appendFloat(bytes, sample.rawElevation);
            appendFloat(bytes, sample.waterSurface);
            appendFloat(bytes, sample.discharge);
            appendU64(bytes, sample.waterBodyId);
            bytes.push_back(sample.flags);
            bytes.insert(bytes.end(), 3, 0);
        }
    }
    return bytes;
}

std::shared_ptr<const NativePage> deserializePage(uint64_t seed, PageKey expectedKey,
                                                  std::span<const uint8_t> bytes) {
    constexpr uint16_t HEADER_BYTES = 52;
    if (bytes.size() < HEADER_BYTES || bytes[0] != 'N' || bytes[1] != 'H' || bytes[2] != '4' ||
        bytes[3] != 'P') {
        throw std::runtime_error("native hydrology payload header is invalid");
    }
    size_t offset = 4;
    uint16_t schema = 0;
    uint16_t headerBytes = 0;
    uint64_t pageX = 0;
    uint64_t pageZ = 0;
    uint16_t spacing = 0;
    uint16_t pageEdge = 0;
    uint16_t apron = 0;
    uint16_t rasterEdge = 0;
    uint32_t rasterCells = 0;
    uint32_t edgeSamples = 0;
    uint32_t lakeStatsCount = 0;
    uint32_t depressionCount = 0;
    uint32_t reserved = 0;
    if (!readU16(bytes, offset, schema) || !readU16(bytes, offset, headerBytes) ||
        !readU64(bytes, offset, pageX) || !readU64(bytes, offset, pageZ) ||
        !readU16(bytes, offset, spacing) || !readU16(bytes, offset, pageEdge) ||
        !readU16(bytes, offset, apron) || !readU16(bytes, offset, rasterEdge) ||
        !readU32(bytes, offset, rasterCells) || !readU32(bytes, offset, edgeSamples) ||
        !readU32(bytes, offset, lakeStatsCount) || !readU32(bytes, offset, depressionCount) ||
        !readU32(bytes, offset, reserved) || schema != NATIVE_HYDROLOGY_PAYLOAD_SCHEMA_VERSION ||
        headerBytes != HEADER_BYTES || offset != HEADER_BYTES ||
        PageKey{static_cast<int64_t>(pageX), static_cast<int64_t>(pageZ)} != expectedKey ||
        spacing != NATIVE_HYDROLOGY_RASTER_SPACING || pageEdge != NATIVE_HYDROLOGY_PAGE_EDGE ||
        apron != RASTER_APRON || rasterEdge != RASTER_EDGE || rasterCells != RASTER_CELLS ||
        edgeSamples != CORE_EDGE_SAMPLES || lakeStatsCount > RASTER_CELLS ||
        depressionCount > RASTER_CELLS || (reserved & ~0x0FU) != 0) {
        throw std::runtime_error("native hydrology payload schema is incompatible");
    }

    auto page = std::make_shared<NativePage>();
    page->key = expectedKey;
    page->originX = checkedCoordinate(expectedKey.x, 0);
    page->originZ = checkedCoordinate(expectedKey.z, 0);
    page->openSpillReceivingEdgeMask = static_cast<uint8_t>(reserved);
    const auto readFloats = [&](std::vector<float>& values) {
        return readVector(bytes, offset, RASTER_CELLS, values, readFloat);
    };
    if (!readFloats(page->routingElevationMeters) ||
        !readFloats(page->localRunoffMmSquareKilometers) || !readFloats(page->rawElevation) ||
        !readFloats(page->waterSurface) || !readFloats(page->discharge) ||
        !readFloats(page->baseflow) || !readFloats(page->seasonality) ||
        !readFloats(page->groundwaterRecharge) || !readFloats(page->groundwaterHead) ||
        !readFloats(page->flowX) || !readFloats(page->flowZ) ||
        !readFloats(page->channelGradient) || !readFloats(page->lakeShoreDistance) ||
        !readVector(bytes, offset, RASTER_CELLS, page->receiverFirst,
                    [](std::span<const uint8_t> input, size_t& position, int32_t& value) {
                        uint32_t encoded = 0;
                        if (!readU32(input, position, encoded)) return false;
                        value = static_cast<int32_t>(encoded);
                        return true;
                    }) ||
        !readVector(bytes, offset, RASTER_CELLS, page->receiverSecond,
                    [](std::span<const uint8_t> input, size_t& position, int32_t& value) {
                        uint32_t encoded = 0;
                        if (!readU32(input, position, encoded)) return false;
                        value = static_cast<int32_t>(encoded);
                        return true;
                    }) ||
        !readFloats(page->receiverSecondWeight) ||
        !readVector(bytes, offset, RASTER_CELLS, page->waterBodyIds, readU64)) {
        throw std::runtime_error("native hydrology payload fields are truncated");
    }
    if (offset > bytes.size() || bytes.size() - offset < 3U * RASTER_CELLS)
        throw std::runtime_error("native hydrology payload flags are truncated");
    page->streamOrder.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                             bytes.begin() + static_cast<std::ptrdiff_t>(offset + RASTER_CELLS));
    offset += RASTER_CELLS;
    page->flags.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                       bytes.begin() + static_cast<std::ptrdiff_t>(offset + RASTER_CELLS));
    offset += RASTER_CELLS;
    page->waterfallBranchMasks.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                      bytes.begin() +
                                          static_cast<std::ptrdiff_t>(offset + RASTER_CELLS));
    offset += RASTER_CELLS;
    for (uint32_t index = 0; index < lakeStatsCount; ++index) {
        WaterBodyId body = NO_WATER_BODY;
        LakeStats stats;
        if (!readU64(bytes, offset, body) ||
            !readDouble(bytes, offset, stats.areaSquareKilometers) ||
            !readDouble(bytes, offset, stats.volumeCubicMeters) ||
            !readDouble(bytes, offset, stats.runoffMmSquareKilometers) || body == NO_WATER_BODY ||
            !page->lakeStats.emplace(body, stats).second) {
            throw std::runtime_error("native hydrology lake summary is invalid");
        }
    }
    page->depressionSummaries.resize(depressionCount);
    for (DepressionSummary& summary : page->depressionSummaries) {
        uint64_t localAnchorX = 0;
        uint64_t localAnchorZ = 0;
        uint64_t naturalOutletX = 0;
        uint64_t naturalOutletZ = 0;
        if (!readU64(bytes, offset, summary.waterBodyId) || !readU64(bytes, offset, localAnchorX) ||
            !readU64(bytes, offset, localAnchorZ) ||
            !readFloat(bytes, offset, summary.localStage) ||
            !readDouble(bytes, offset, summary.coreAreaSquareKilometers) ||
            !readDouble(bytes, offset, summary.coreVolumeCubicMeters) ||
            !readDouble(bytes, offset, summary.coreRunoffMmSquareKilometers) ||
            !readU64(bytes, offset, naturalOutletX) || !readU64(bytes, offset, naturalOutletZ) ||
            !readFloat(bytes, offset, summary.naturalOutletStage) || offset > bytes.size() ||
            bytes.size() - offset < 4) {
            throw std::runtime_error("native hydrology depression summary is truncated");
        }
        summary.localAnchorX = static_cast<int64_t>(localAnchorX);
        summary.localAnchorZ = static_cast<int64_t>(localAnchorZ);
        summary.naturalOutletX = static_cast<int64_t>(naturalOutletX);
        summary.naturalOutletZ = static_cast<int64_t>(naturalOutletZ);
        summary.edgeMask = bytes[offset++];
        summary.naturalOutlet = static_cast<BasinOutlet>(bytes[offset++]);
        if (bytes[offset++] != 0 || bytes[offset++] != 0 || summary.waterBodyId == NO_WATER_BODY ||
            !std::isfinite(summary.localStage) ||
            !std::isfinite(summary.coreAreaSquareKilometers) ||
            !std::isfinite(summary.coreVolumeCubicMeters) ||
            !std::isfinite(summary.coreRunoffMmSquareKilometers) ||
            !std::isfinite(summary.naturalOutletStage) || summary.coreAreaSquareKilometers < 0.0 ||
            summary.coreVolumeCubicMeters < 0.0 || summary.coreRunoffMmSquareKilometers < 0.0 ||
            (summary.edgeMask & 0xF0U) != 0 ||
            (summary.naturalOutlet != BasinOutlet::NONE &&
             summary.naturalOutlet != BasinOutlet::OCEAN &&
             summary.naturalOutlet != BasinOutlet::SHARED_PORTAL)) {
            throw std::runtime_error("native hydrology depression summary is invalid");
        }
    }
    for (auto& edge : page->edgeSummaries) {
        edge.resize(CORE_EDGE_SAMPLES);
        for (EdgeSummarySample& sample : edge) {
            if (!readFloat(bytes, offset, sample.rawElevation) ||
                !readFloat(bytes, offset, sample.waterSurface) ||
                !readFloat(bytes, offset, sample.discharge) ||
                !readU64(bytes, offset, sample.waterBodyId) || offset > bytes.size() ||
                bytes.size() - offset < 4) {
                throw std::runtime_error("native hydrology edge summary is truncated");
            }
            sample.flags = bytes[offset++];
            if (bytes[offset++] != 0 || bytes[offset++] != 0 || bytes[offset++] != 0)
                throw std::runtime_error(
                    "native hydrology edge summary reserved bytes are invalid");
        }
    }
    if (offset != bytes.size())
        throw std::runtime_error("native hydrology payload has trailing bytes");

    const auto finite = [](const std::vector<float>& values) {
        return std::ranges::all_of(values, [](float value) { return std::isfinite(value); });
    };
    if (!finite(page->routingElevationMeters) || !finite(page->localRunoffMmSquareKilometers) ||
        !finite(page->rawElevation) || !finite(page->waterSurface) || !finite(page->discharge) ||
        !finite(page->baseflow) || !finite(page->seasonality) ||
        !finite(page->groundwaterRecharge) || !finite(page->groundwaterHead) ||
        !finite(page->flowX) || !finite(page->flowZ) || !finite(page->channelGradient) ||
        !finite(page->lakeShoreDistance) || !finite(page->receiverSecondWeight) ||
        page->byteSize() > NATIVE_HYDROLOGY_MAX_PAGE_BYTES) {
        throw std::runtime_error("native hydrology payload contains invalid fields");
    }
    if (!std::ranges::all_of(page->localRunoffMmSquareKilometers,
                             [](float value) { return value >= 0.0F; })) {
        throw std::runtime_error("native hydrology payload contains negative local runoff");
    }
    for (size_t index = 0; index < RASTER_CELLS; ++index) {
        const auto validReceiver = [](int32_t receiver) {
            return receiver >= -1 && receiver < RASTER_CELLS;
        };
        if (!validReceiver(page->receiverFirst[index]) ||
            !validReceiver(page->receiverSecond[index]) ||
            page->receiverSecondWeight[index] < 0.0F || page->receiverSecondWeight[index] > 1.0F ||
            page->waterfallBranchMasks[index] > 0x03U ||
            (page->flags[index] &
             ~(CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND | CELL_WATERFALL |
               CELL_WETLAND_CANDIDATE | CELL_ESTUARY | CELL_DELTA)) != 0) {
            throw std::runtime_error("native hydrology payload graph is invalid");
        }
        if (!waterfallBranchMaskIsValid(*page, index)) {
            throw std::runtime_error("native hydrology payload waterfall branches are invalid");
        }
        if ((page->flags[index] & CELL_WETLAND) != 0 &&
            (page->waterBodyIds[index] == NO_WATER_BODY ||
             !std::isfinite(page->waterSurface[index]))) {
            throw std::runtime_error("native hydrology payload wetland is unowned");
        }
        if (((page->flags[index] & CELL_ESTUARY) != 0 && (page->flags[index] & CELL_RIVER) == 0) ||
            ((page->flags[index] & CELL_DELTA) != 0 &&
             (page->flags[index] & (CELL_RIVER | CELL_ESTUARY)) != (CELL_RIVER | CELL_ESTUARY))) {
            throw std::runtime_error("native hydrology payload coastal identity is invalid");
        }
        if ((page->flags[index] & CELL_ESTUARY) != 0 && page->waterSurface[index] < SEA_LEVEL) {
            throw std::runtime_error("native hydrology payload estuary stage is below sea level");
        }
        if ((page->flags[index] & CELL_DELTA) != 0) {
            const int32_t first = page->receiverFirst[index];
            const int32_t second = page->receiverSecond[index];
            if (first < 0 || second < 0 ||
                page->receiverSecondWeight[index] < MINIMUM_CHANNEL_BRANCH_WEIGHT ||
                1.0F - page->receiverSecondWeight[index] < MINIMUM_CHANNEL_BRANCH_WEIGHT ||
                (page->flags[static_cast<size_t>(first)] & CELL_OCEAN) == 0 ||
                (page->flags[static_cast<size_t>(second)] & CELL_OCEAN) == 0) {
                throw std::runtime_error("native hydrology payload distributary is invalid");
            }
        }
    }
    page->oceanId = stableWaterBodyId(seed, 0, 0, 0x4F43'4541'4E00'0001ULL);
    buildNativePageSamplingIndexes(*page);
    if (page->byteSize() > NATIVE_HYDROLOGY_MAX_PAGE_BYTES)
        throw std::runtime_error("native hydrology payload sampling index exceeds memory bound");
    return page;
}

void squaredDistanceTransform1D(std::span<const float> source, std::span<float> destination,
                                std::vector<int>& sites, std::vector<double>& boundaries) {
    const auto firstFinite =
        std::ranges::find_if(source, [](float value) { return std::isfinite(value); });
    if (firstFinite == source.end()) {
        std::ranges::fill(destination, std::numeric_limits<float>::infinity());
        return;
    }
    sites.resize(source.size());
    boundaries.resize(sites.size() + 1);
    size_t envelope = 0;
    const int first = static_cast<int>(std::distance(source.begin(), firstFinite));
    sites[0] = first;
    boundaries[0] = -std::numeric_limits<double>::infinity();
    boundaries[1] = std::numeric_limits<double>::infinity();
    for (int candidate = first + 1; candidate < static_cast<int>(source.size()); ++candidate) {
        if (!std::isfinite(source[static_cast<size_t>(candidate)])) continue;
        double intersection = 0.0;
        while (true) {
            const int retained = sites[envelope];
            intersection = ((static_cast<double>(source[static_cast<size_t>(candidate)]) +
                             static_cast<double>(candidate) * candidate) -
                            (static_cast<double>(source[static_cast<size_t>(retained)]) +
                             static_cast<double>(retained) * retained)) /
                           (2.0 * (candidate - retained));
            if (intersection > boundaries[envelope] || envelope == 0) break;
            --envelope;
        }
        ++envelope;
        sites[envelope] = candidate;
        boundaries[envelope] = intersection;
        boundaries[envelope + 1] = std::numeric_limits<double>::infinity();
    }
    envelope = 0;
    for (size_t index = 0; index < source.size(); ++index) {
        while (boundaries[envelope + 1] < static_cast<double>(index))
            ++envelope;
        const int site = sites[envelope];
        const double delta = static_cast<double>(index) - site;
        destination[index] = static_cast<float>(delta * delta + source[static_cast<size_t>(site)]);
    }
}

std::vector<float> squaredDistanceTransform(std::span<const uint8_t> mask, uint8_t selected,
                                            int edge) {
    if (edge <= 0 || mask.size() != static_cast<size_t>(edge) * edge)
        throw std::invalid_argument("invalid distance transform mask");
    std::vector<float> temporary(mask.size());
    std::vector<float> result(mask.size());
    std::vector<float> source(static_cast<size_t>(edge));
    std::vector<float> destination(static_cast<size_t>(edge));
    std::vector<int> sites(static_cast<size_t>(edge));
    std::vector<double> boundaries(static_cast<size_t>(edge + 1));
    const auto dynamicIndex = [edge](int x, int z) { return z * edge + x; };
    for (int z = 0; z < edge; ++z) {
        for (int x = 0; x < edge; ++x) {
            source[static_cast<size_t>(x)] =
                mask[static_cast<size_t>(dynamicIndex(x, z))] == selected
                    ? 0.0F
                    : std::numeric_limits<float>::infinity();
        }
        squaredDistanceTransform1D(source, destination, sites, boundaries);
        for (int x = 0; x < edge; ++x)
            temporary[static_cast<size_t>(dynamicIndex(x, z))] =
                destination[static_cast<size_t>(x)];
    }
    for (int x = 0; x < edge; ++x) {
        for (int z = 0; z < edge; ++z)
            source[static_cast<size_t>(z)] = temporary[static_cast<size_t>(dynamicIndex(x, z))];
        squaredDistanceTransform1D(source, destination, sites, boundaries);
        for (int z = 0; z < edge; ++z)
            result[static_cast<size_t>(dynamicIndex(x, z))] = destination[static_cast<size_t>(z)];
    }
    return result;
}

std::vector<float> squaredDistanceTransformRectangle(std::span<const uint8_t> mask,
                                                     uint8_t selected, int width, int height) {
    if (width <= 0 || height <= 0 ||
        mask.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw std::invalid_argument("invalid rectangular distance transform mask");
    }
    std::vector<float> temporary(mask.size());
    std::vector<float> result(mask.size());
    std::vector<float> source(static_cast<size_t>(std::max(width, height)));
    std::vector<float> destination(source.size());
    std::vector<int> sites(source.size());
    std::vector<double> boundaries(source.size() + 1);
    const auto rectangularIndex = [width](int x, int z) {
        return static_cast<size_t>(z) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };
    for (int z = 0; z < height; ++z) {
        for (int x = 0; x < width; ++x) {
            source[static_cast<size_t>(x)] = mask[rectangularIndex(x, z)] == selected
                                                 ? 0.0F
                                                 : std::numeric_limits<float>::infinity();
        }
        squaredDistanceTransform1D(std::span(source).first(static_cast<size_t>(width)),
                                   std::span(destination).first(static_cast<size_t>(width)), sites,
                                   boundaries);
        for (int x = 0; x < width; ++x)
            temporary[rectangularIndex(x, z)] = destination[static_cast<size_t>(x)];
    }
    for (int x = 0; x < width; ++x) {
        for (int z = 0; z < height; ++z)
            source[static_cast<size_t>(z)] = temporary[rectangularIndex(x, z)];
        squaredDistanceTransform1D(std::span(source).first(static_cast<size_t>(height)),
                                   std::span(destination).first(static_cast<size_t>(height)), sites,
                                   boundaries);
        for (int z = 0; z < height; ++z)
            result[rectangularIndex(x, z)] = destination[static_cast<size_t>(z)];
    }
    return result;
}

float runoffFromClimate(const BasinHydroclimateSample& climate) {
    const double precipitation = std::clamp(climate.annualPrecipitationMm, 0.0, 20'000.0);
    if (precipitation <= 0.0) return 0.0F;
    const double aridity =
        std::clamp(climate.potentialEvapotranspirationMm / precipitation, 0.0, 100.0);
    constexpr double BUDYKO_OMEGA = 2.6;
    const double evapotranspirationFraction = std::clamp(
        1.0 + aridity - std::pow(1.0 + std::pow(aridity, BUDYKO_OMEGA), 1.0 / BUDYKO_OMEGA), 0.0,
        1.0);
    return static_cast<float>(precipitation * (1.0 - evapotranspirationFraction));
}

float baseflowFromClimate(float runoff, const BasinHydroclimateSample& climate) {
    const double variability = std::clamp(climate.precipitationCoefficientOfVariation, 0.0, 2.0);
    const double temperaturePenalty =
        std::clamp((climate.meanTemperatureC - 18.0) / 35.0, 0.0, 0.35);
    const double fraction = std::clamp(0.34 - variability * 0.12 - temperaturePenalty, 0.05, 0.42);
    return static_cast<float>(runoff * fraction);
}

struct FloodNode {
    float elevation = 0.0F;
    int32_t index = -1;
};

struct FloodNodeGreater {
    bool operator()(const FloodNode& first, const FloodNode& second) const noexcept {
        if (first.elevation != second.elevation) return first.elevation > second.elevation;
        return first.index > second.index;
    }
};

struct DisjointSets {
    explicit DisjointSets(size_t size) : parents(size, -1) {}

    int32_t find(int32_t value) {
        int32_t root = value;
        while (parents[static_cast<size_t>(root)] != root)
            root = parents[static_cast<size_t>(root)];
        while (parents[static_cast<size_t>(value)] != value) {
            const int32_t next = parents[static_cast<size_t>(value)];
            parents[static_cast<size_t>(value)] = root;
            value = next;
        }
        return root;
    }

    void activate(int32_t value) { parents[static_cast<size_t>(value)] = value; }

    bool active(int32_t value) const { return parents[static_cast<size_t>(value)] >= 0; }

    void unite(int32_t first, int32_t second) {
        first = find(first);
        second = find(second);
        if (first == second) return;
        const int32_t low = std::min(first, second);
        const int32_t high = std::max(first, second);
        parents[static_cast<size_t>(high)] = low;
    }

    std::vector<int32_t> parents;
};

WaterBodyId boundedRiverWaterBodyId(uint64_t seed, int64_t boundary, int64_t along, bool vertical) {
    constexpr int64_t HIERARCHY_REACH_BLOCKS = 3LL * NATIVE_HYDROLOGY_PAGE_EDGE;
    const int64_t reachAnchor = world_coord::floorMultiple(boundary, HIERARCHY_REACH_BLOCKS);
    const int64_t crossAnchor = world_coord::floorMultiple(along, int64_t{256});
    return vertical ? stableWaterBodyId(seed, reachAnchor, crossAnchor, 0x5256'4849'4552'0001ULL)
                    : stableWaterBodyId(seed, crossAnchor, reachAnchor, 0x5248'4849'4552'0001ULL);
}

std::shared_ptr<const NativePage> buildPage(uint64_t seed,
                                            const NativeHydrologyIdentityRegistry& identities,
                                            PageKey key, const NativeHydrologyInputFunction& input,
                                            size_t& peakBuildBytes) {
    if (!input) throw std::invalid_argument("native hydrology input callback is empty");
    auto page = std::make_shared<NativePage>();
    page->key = key;
    page->originX = checkedCoordinate(key.x, 0);
    page->originZ = checkedCoordinate(key.z, 0);

    std::vector<NativeHydrologyPosition> positions;
    positions.reserve(RASTER_CELLS);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        const int64_t worldZ = checkedCoordinate(key.z, z - RASTER_APRON);
        for (int x = 0; x < RASTER_EDGE; ++x) {
            positions.push_back({
                .x = checkedCoordinate(key.x, x - RASTER_APRON),
                .z = worldZ,
            });
        }
    }
    std::vector<NativeHydrologyInput> inputs(RASTER_CELLS);
    for (NativeHydrologyInput& value : inputs)
        value.elevationMeters = std::numeric_limits<double>::quiet_NaN();
    input(positions, inputs);

    // Keep the physical learned field through every topological decision. The
    // persisted page below is deliberately game-facing and uses block-height
    // coordinates, so it cannot become an accidental routing input.
    page->routingElevationMeters.resize(RASTER_CELLS);
    std::vector<float>& routingElevationMeters = page->routingElevationMeters;
    std::vector<float> groundwaterHeadMeters(RASTER_CELLS);
    page->localRunoffMmSquareKilometers.resize(RASTER_CELLS);
    std::vector<float>& localRunoff = page->localRunoffMmSquareKilometers;
    std::vector<float> localBaseflow(RASTER_CELLS);
    std::vector<float> localRecharge(RASTER_CELLS);
    page->seasonality.resize(RASTER_CELLS);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const NativeHydrologyInput& value = inputs[static_cast<size_t>(index)];
        if (!std::isfinite(value.elevationMeters) ||
            !std::isfinite(value.climate.annualPrecipitationMm) ||
            !std::isfinite(value.climate.potentialEvapotranspirationMm) ||
            !std::isfinite(value.climate.precipitationCoefficientOfVariation)) {
            throw std::runtime_error("native hydrology input contains a nonfinite value");
        }
        routingElevationMeters[static_cast<size_t>(index)] =
            static_cast<float>(value.elevationMeters);
        const float runoff = runoffFromClimate(value.climate);
        const float baseflow = baseflowFromClimate(runoff, value.climate);
        localRunoff[static_cast<size_t>(index)] =
            static_cast<float>(runoff * CELL_AREA_SQUARE_KILOMETERS);
        localBaseflow[static_cast<size_t>(index)] =
            static_cast<float>(baseflow * CELL_AREA_SQUARE_KILOMETERS);
        localRecharge[static_cast<size_t>(index)] = std::max(0.0F, baseflow * 0.72F);
        page->seasonality[static_cast<size_t>(index)] = static_cast<float>(
            std::clamp(value.climate.precipitationCoefficientOfVariation, 0.0, 2.0));
        const double waterTableDepth =
            std::clamp(2.0 + value.climate.potentialEvapotranspirationMm / 450.0 -
                           value.climate.annualPrecipitationMm / 1'200.0,
                       0.25, 12.0);
        groundwaterHeadMeters[static_cast<size_t>(index)] =
            static_cast<float>(value.elevationMeters - waterTableDepth);
    }

    // A page can be locally dry only because its apron is treated as an
    // outlet, while an edge-connected depression on the adjacent page would
    // continue into this page after the canonical multi-page solve. Record a
    // cheap conservative hint for that case. Each tested escape is a real
    // path to another core edge; if even the cheapest straight or edge path
    // rises by a lake-depth threshold, the facing neighbor must be inspected
    // before an interior dry sample can be published.
    std::array<float, CORE_EDGE_SAMPLES> westNorth{};
    std::array<float, CORE_EDGE_SAMPLES> westSouth{};
    std::array<float, CORE_EDGE_SAMPLES> eastNorth{};
    std::array<float, CORE_EDGE_SAMPLES> eastSouth{};
    std::array<float, CORE_EDGE_SAMPLES> northWest{};
    std::array<float, CORE_EDGE_SAMPLES> northEast{};
    std::array<float, CORE_EDGE_SAMPLES> southWest{};
    std::array<float, CORE_EDGE_SAMPLES> southEast{};
    std::array<float, CORE_EDGE_SAMPLES> rowPeak{};
    std::array<float, CORE_EDGE_SAMPLES> columnPeak{};
    for (int along = 0; along <= INTERIOR_INTERVALS; ++along) {
        const int raster = CORE_BEGIN + along;
        float horizontalPeak = -std::numeric_limits<float>::infinity();
        float verticalPeak = -std::numeric_limits<float>::infinity();
        for (int cross = CORE_BEGIN; cross <= CORE_END; ++cross) {
            horizontalPeak =
                std::max(horizontalPeak,
                         routingElevationMeters[static_cast<size_t>(indexOf(cross, raster))]);
            verticalPeak = std::max(
                verticalPeak, routingElevationMeters[static_cast<size_t>(indexOf(raster, cross))]);
        }
        rowPeak[static_cast<size_t>(along)] = horizontalPeak;
        columnPeak[static_cast<size_t>(along)] = verticalPeak;
    }
    for (int along = 0; along <= INTERIOR_INTERVALS; ++along) {
        const int raster = CORE_BEGIN + along;
        const size_t index = static_cast<size_t>(along);
        const float west = routingElevationMeters[static_cast<size_t>(indexOf(CORE_BEGIN, raster))];
        const float east = routingElevationMeters[static_cast<size_t>(indexOf(CORE_END, raster))];
        const float north =
            routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_BEGIN))];
        const float south = routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_END))];
        westNorth[index] = along == 0 ? west : std::max(westNorth[index - 1], west);
        eastNorth[index] = along == 0 ? east : std::max(eastNorth[index - 1], east);
        northWest[index] = along == 0 ? north : std::max(northWest[index - 1], north);
        southWest[index] = along == 0 ? south : std::max(southWest[index - 1], south);
    }
    for (int along = INTERIOR_INTERVALS; along >= 0; --along) {
        const int raster = CORE_BEGIN + along;
        const size_t index = static_cast<size_t>(along);
        const float west = routingElevationMeters[static_cast<size_t>(indexOf(CORE_BEGIN, raster))];
        const float east = routingElevationMeters[static_cast<size_t>(indexOf(CORE_END, raster))];
        const float north =
            routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_BEGIN))];
        const float south = routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_END))];
        const bool last = along == INTERIOR_INTERVALS;
        westSouth[index] = last ? west : std::max(westSouth[index + 1], west);
        eastSouth[index] = last ? east : std::max(eastSouth[index + 1], east);
        northEast[index] = last ? north : std::max(northEast[index + 1], north);
        southEast[index] = last ? south : std::max(southEast[index + 1], south);
    }
    const auto receiving = [](float port, float firstEdgeEscape, float secondEdgeEscape,
                              float oppositeEscape) {
        if (port < NATIVE_HYDROLOGY_SEA_LEVEL_METERS) return false;
        const float cheapestEscape = std::min({firstEdgeEscape, secondEdgeEscape, oppositeEscape});
        return cheapestEscape - port > LAKE_DEPTH_THRESHOLD_METERS;
    };
    for (int along = 0; along <= INTERIOR_INTERVALS; ++along) {
        const int raster = CORE_BEGIN + along;
        const size_t index = static_cast<size_t>(along);
        const float west = routingElevationMeters[static_cast<size_t>(indexOf(CORE_BEGIN, raster))];
        const float east = routingElevationMeters[static_cast<size_t>(indexOf(CORE_END, raster))];
        const float north =
            routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_BEGIN))];
        const float south = routingElevationMeters[static_cast<size_t>(indexOf(raster, CORE_END))];
        if (receiving(west, westNorth[index], westSouth[index], rowPeak[index]))
            page->openSpillReceivingEdgeMask |= 1U << static_cast<uint8_t>(PageEdge::WEST);
        if (receiving(east, eastNorth[index], eastSouth[index], rowPeak[index]))
            page->openSpillReceivingEdgeMask |= 1U << static_cast<uint8_t>(PageEdge::EAST);
        if (receiving(north, northWest[index], northEast[index], columnPeak[index]))
            page->openSpillReceivingEdgeMask |= 1U << static_cast<uint8_t>(PageEdge::NORTH);
        if (receiving(south, southWest[index], southEast[index], columnPeak[index]))
            page->openSpillReceivingEdgeMask |= 1U << static_cast<uint8_t>(PageEdge::SOUTH);
    }

    std::vector<float> filledMeters(RASTER_CELLS, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> visited(RASTER_CELLS, 0);
    std::vector<int32_t> floodParent(RASTER_CELLS, -1);
    std::vector<int32_t> floodRank(RASTER_CELLS, -1);
    std::vector<int32_t> floodOrder;
    floodOrder.reserve(RASTER_CELLS);
    std::priority_queue<FloodNode, std::vector<FloodNode>, FloodNodeGreater> frontier;
    const auto addTerminal = [&](int index) {
        if (visited[static_cast<size_t>(index)] != 0) return;
        visited[static_cast<size_t>(index)] = 1;
        filledMeters[static_cast<size_t>(index)] =
            routingElevationMeters[static_cast<size_t>(index)];
        frontier.push({filledMeters[static_cast<size_t>(index)], index});
    };
    for (int z = 0; z < RASTER_EDGE; ++z) {
        addTerminal(indexOf(0, z));
        addTerminal(indexOf(RASTER_EDGE - 1, z));
    }
    for (int x = 1; x < RASTER_EDGE - 1; ++x) {
        addTerminal(indexOf(x, 0));
        addTerminal(indexOf(x, RASTER_EDGE - 1));
    }
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (routingElevationMeters[static_cast<size_t>(index)] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS)
            addTerminal(index);
    }
    while (!frontier.empty()) {
        const FloodNode current = frontier.top();
        frontier.pop();
        if (floodRank[static_cast<size_t>(current.index)] >= 0) continue;
        floodRank[static_cast<size_t>(current.index)] = static_cast<int32_t>(floodOrder.size());
        floodOrder.push_back(current.index);
        const int x = current.index % RASTER_EDGE;
        const int z = current.index / RASTER_EDGE;
        for (size_t direction = 0; direction < DX.size(); ++direction) {
            const int neighborX = x + DX[direction];
            const int neighborZ = z + DZ[direction];
            if (!inside(neighborX, neighborZ)) continue;
            const int neighbor = indexOf(neighborX, neighborZ);
            if (visited[static_cast<size_t>(neighbor)] != 0) continue;
            visited[static_cast<size_t>(neighbor)] = 1;
            floodParent[static_cast<size_t>(neighbor)] = current.index;
            filledMeters[static_cast<size_t>(neighbor)] =
                std::max(routingElevationMeters[static_cast<size_t>(neighbor)], current.elevation);
            frontier.push({filledMeters[static_cast<size_t>(neighbor)], neighbor});
        }
    }
    if (floodOrder.size() != RASTER_CELLS)
        throw std::runtime_error("native hydrology priority flood is incomplete");

    std::vector<Receiver> receivers(RASTER_CELLS);
    page->flowX.assign(RASTER_CELLS, 1.0F);
    page->flowZ.assign(RASTER_CELLS, 0.0F);
    for (int z = 1; z < RASTER_EDGE - 1; ++z) {
        for (int x = 1; x < RASTER_EDGE - 1; ++x) {
            const int index = indexOf(x, z);
            if (routingElevationMeters[static_cast<size_t>(index)] <
                NATIVE_HYDROLOGY_SEA_LEVEL_METERS)
                continue;
            const double gradientX = (filledMeters[static_cast<size_t>(indexOf(x + 1, z))] -
                                      filledMeters[static_cast<size_t>(indexOf(x - 1, z))]) /
                                     (2.0 * NATIVE_HYDROLOGY_CELL_EDGE_METERS);
            const double gradientZ = (filledMeters[static_cast<size_t>(indexOf(x, z + 1))] -
                                      filledMeters[static_cast<size_t>(indexOf(x, z - 1))]) /
                                     (2.0 * NATIVE_HYDROLOGY_CELL_EDGE_METERS);
            double angle = std::atan2(-gradientZ, -gradientX);
            if (angle < 0.0) angle += 2.0 * std::numbers::pi;
            const double sector = angle / (std::numbers::pi / 4.0);
            const int firstDirection = static_cast<int>(std::floor(sector)) & 7;
            const int secondDirection = (firstDirection + 1) & 7;
            const auto eligible = [&](int direction) {
                const int candidate = indexOf(x + DX[static_cast<size_t>(direction)],
                                              z + DZ[static_cast<size_t>(direction)]);
                const float candidateHeight = filledMeters[static_cast<size_t>(candidate)];
                const float currentHeight = filledMeters[static_cast<size_t>(index)];
                return candidateHeight < currentHeight - ROUTING_ELEVATION_EPSILON_METERS ||
                       (std::abs(candidateHeight - currentHeight) <=
                            ROUTING_ELEVATION_EPSILON_METERS &&
                        floodRank[static_cast<size_t>(candidate)] <
                            floodRank[static_cast<size_t>(index)]);
            };
            Receiver receiver;
            const bool firstEligible = eligible(firstDirection);
            const bool secondEligible = eligible(secondDirection);
            if (firstEligible)
                receiver.first = indexOf(x + DX[static_cast<size_t>(firstDirection)],
                                         z + DZ[static_cast<size_t>(firstDirection)]);
            if (secondEligible)
                receiver.second = indexOf(x + DX[static_cast<size_t>(secondDirection)],
                                          z + DZ[static_cast<size_t>(secondDirection)]);
            receiver.secondWeight = static_cast<float>(sector - std::floor(sector));
            if (receiver.first < 0 && receiver.second >= 0) {
                receiver.first = receiver.second;
                receiver.second = -1;
                receiver.secondWeight = 0.0F;
            } else if (receiver.first >= 0 && receiver.second < 0) {
                receiver.secondWeight = 0.0F;
            } else if (receiver.first < 0) {
                receiver.first = floodParent[static_cast<size_t>(index)];
                receiver.secondWeight = 0.0F;
            }
            if (receiver.first >= 0) {
                const int firstX = receiver.first % RASTER_EDGE - x;
                const int firstZ = receiver.first / RASTER_EDGE - z;
                double flowX = firstX;
                double flowZ = firstZ;
                if (receiver.second >= 0) {
                    const double secondWeight = receiver.secondWeight;
                    flowX = firstX * (1.0 - secondWeight) +
                            (receiver.second % RASTER_EDGE - x) * secondWeight;
                    flowZ = firstZ * (1.0 - secondWeight) +
                            (receiver.second / RASTER_EDGE - z) * secondWeight;
                }
                const double length = std::hypot(flowX, flowZ);
                if (length > 1.0e-9) {
                    page->flowX[static_cast<size_t>(index)] = static_cast<float>(flowX / length);
                    page->flowZ[static_cast<size_t>(index)] = static_cast<float>(flowZ / length);
                }
            }
            receivers[static_cast<size_t>(index)] = receiver;
        }
    }

    struct IncomingPort {
        int index = -1;
        WaterBodyId identity = NO_WATER_BODY;
    };
    std::vector<IncomingPort> incomingPorts;
    const auto addIncomingPort = [&](int x, int z, PageEdge edge) {
        const int index = indexOf(x, z);
        const Receiver receiver = receivers[static_cast<size_t>(index)];
        if (receiver.first < 0 ||
            routingElevationMeters[static_cast<size_t>(index)] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS)
            return;
        const int targetX = receiver.first % RASTER_EDGE;
        const int targetZ = receiver.first / RASTER_EDGE;
        const bool inward =
            (edge == PageEdge::WEST && targetX > x) || (edge == PageEdge::EAST && targetX < x) ||
            (edge == PageEdge::NORTH && targetZ > z) || (edge == PageEdge::SOUTH && targetZ < z);
        if (!inward) return;
        constexpr int VALLEY_RADIUS = 6;
        const float elevation = filledMeters[static_cast<size_t>(index)];
        for (int offset = -VALLEY_RADIUS; offset <= VALLEY_RADIUS; ++offset) {
            if (offset == 0) continue;
            const int compareX =
                (edge == PageEdge::WEST || edge == PageEdge::EAST) ? x : x + offset;
            const int compareZ =
                (edge == PageEdge::WEST || edge == PageEdge::EAST) ? z + offset : z;
            if (compareX < CORE_BEGIN || compareX > CORE_END || compareZ < CORE_BEGIN ||
                compareZ > CORE_END)
                continue;
            const float comparison = filledMeters[static_cast<size_t>(indexOf(compareX, compareZ))];
            if (comparison < elevation - ROUTING_ELEVATION_EPSILON_METERS ||
                (std::abs(comparison - elevation) <= ROUTING_ELEVATION_EPSILON_METERS &&
                 offset < 0)) {
                return;
            }
        }
        int64_t boundary = 0;
        int64_t along = 0;
        bool vertical = false;
        if (edge == PageEdge::WEST || edge == PageEdge::EAST) {
            vertical = true;
            boundary = checkedCoordinate(key.x, edge == PageEdge::WEST ? 0 : INTERIOR_INTERVALS);
            along = checkedCoordinate(key.z, z - RASTER_APRON);
        } else {
            boundary = checkedCoordinate(key.z, edge == PageEdge::NORTH ? 0 : INTERIOR_INTERVALS);
            along = checkedCoordinate(key.x, x - RASTER_APRON);
        }
        const WaterBodyId identity = boundedRiverWaterBodyId(seed, boundary, along, vertical);
        const auto found = std::ranges::find(incomingPorts, index, &IncomingPort::index);
        if (found == incomingPorts.end())
            incomingPorts.push_back({.index = index, .identity = identity});
        else
            found->identity = std::min(found->identity, identity);
    };
    for (int along = CORE_BEGIN; along <= CORE_END; ++along) {
        addIncomingPort(CORE_BEGIN, along, PageEdge::WEST);
        addIncomingPort(CORE_END, along, PageEdge::EAST);
        addIncomingPort(along, CORE_BEGIN, PageEdge::NORTH);
        addIncomingPort(along, CORE_END, PageEdge::SOUTH);
    }

    page->discharge = localRunoff;
    page->baseflow = localBaseflow;
    std::vector<WaterBodyId> reachLabels(RASTER_CELLS, NO_WATER_BODY);
    double pageRunoff = 0.0;
    double pageBaseflow = 0.0;
    for (int z = CORE_BEGIN; z < CORE_END; ++z) {
        for (int x = CORE_BEGIN; x < CORE_END; ++x) {
            const size_t cell = static_cast<size_t>(indexOf(x, z));
            pageRunoff += localRunoff[cell];
            pageBaseflow += localBaseflow[cell];
        }
    }
    if (!incomingPorts.empty()) {
        const float importedRunoff = static_cast<float>(std::max<double>(
            MINIMUM_CHANNEL_DISCHARGE * 2.0, pageRunoff * 0.18 / incomingPorts.size()));
        const float importedBaseflow = static_cast<float>(std::max<double>(
            MINIMUM_CHANNEL_DISCHARGE * 0.25, pageBaseflow * 0.18 / incomingPorts.size()));
        for (const IncomingPort port : incomingPorts) {
            page->discharge[static_cast<size_t>(port.index)] += importedRunoff;
            page->baseflow[static_cast<size_t>(port.index)] += importedBaseflow;
            reachLabels[static_cast<size_t>(port.index)] = port.identity;
        }
    }
    for (auto order = floodOrder.rbegin(); order != floodOrder.rend(); ++order) {
        const int index = *order;
        const Receiver receiver = receivers[static_cast<size_t>(index)];
        if (receiver.first < 0) continue;
        const float secondWeight = receiver.second >= 0 ? receiver.secondWeight : 0.0F;
        const float firstWeight = 1.0F - secondWeight;
        page->discharge[static_cast<size_t>(receiver.first)] +=
            page->discharge[static_cast<size_t>(index)] * firstWeight;
        page->baseflow[static_cast<size_t>(receiver.first)] +=
            page->baseflow[static_cast<size_t>(index)] * firstWeight;
        if (reachLabels[static_cast<size_t>(index)] != NO_WATER_BODY) {
            WaterBodyId& firstLabel = reachLabels[static_cast<size_t>(receiver.first)];
            firstLabel = firstLabel == NO_WATER_BODY
                             ? reachLabels[static_cast<size_t>(index)]
                             : std::min(firstLabel, reachLabels[static_cast<size_t>(index)]);
        }
        if (receiver.second >= 0) {
            page->discharge[static_cast<size_t>(receiver.second)] +=
                page->discharge[static_cast<size_t>(index)] * secondWeight;
            page->baseflow[static_cast<size_t>(receiver.second)] +=
                page->baseflow[static_cast<size_t>(index)] * secondWeight;
            if (reachLabels[static_cast<size_t>(index)] != NO_WATER_BODY) {
                WaterBodyId& secondLabel = reachLabels[static_cast<size_t>(receiver.second)];
                secondLabel = secondLabel == NO_WATER_BODY
                                  ? reachLabels[static_cast<size_t>(index)]
                                  : std::min(secondLabel, reachLabels[static_cast<size_t>(index)]);
            }
        }
    }
    page->receiverFirst.resize(RASTER_CELLS);
    page->receiverSecond.resize(RASTER_CELLS);
    page->receiverSecondWeight.resize(RASTER_CELLS);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        page->receiverFirst[static_cast<size_t>(index)] =
            receivers[static_cast<size_t>(index)].first;
        page->receiverSecond[static_cast<size_t>(index)] =
            receivers[static_cast<size_t>(index)].second;
        page->receiverSecondWeight[static_cast<size_t>(index)] =
            receivers[static_cast<size_t>(index)].secondWeight;
    }

    DisjointSets lakeSets(RASTER_CELLS);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (routingElevationMeters[static_cast<size_t>(index)] >=
                NATIVE_HYDROLOGY_SEA_LEVEL_METERS &&
            filledMeters[static_cast<size_t>(index)] -
                    routingElevationMeters[static_cast<size_t>(index)] >
                LAKE_DEPTH_THRESHOLD_METERS) {
            lakeSets.activate(index);
        }
    }
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const int index = indexOf(x, z);
            if (!lakeSets.active(index)) continue;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (!inside(x + offsetX, z + offsetZ)) continue;
                const int neighbor = indexOf(x + offsetX, z + offsetZ);
                if (lakeSets.active(neighbor) &&
                    std::abs(filledMeters[static_cast<size_t>(index)] -
                             filledMeters[static_cast<size_t>(neighbor)]) <=
                        LAKE_COMPONENT_STAGE_EPSILON_METERS) {
                    lakeSets.unite(index, neighbor);
                }
            }
        }
    }
    struct LakeAnchor {
        float elevation = std::numeric_limits<float>::infinity();
        std::pair<int64_t, int64_t> coordinate;
    };
    std::unordered_map<int32_t, LakeAnchor> lakeAnchors;
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if (!lakeSets.active(index)) continue;
        const int32_t root = lakeSets.find(index);
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        const std::pair coordinate{
            checkedCoordinate(key.x, x - RASTER_APRON),
            checkedCoordinate(key.z, z - RASTER_APRON),
        };
        const float elevation = routingElevationMeters[static_cast<size_t>(index)];
        auto [entry, inserted] = lakeAnchors.try_emplace(
            root, LakeAnchor{.elevation = elevation, .coordinate = coordinate});
        if (!inserted &&
            (elevation < entry->second.elevation - ROUTING_ELEVATION_EPSILON_METERS ||
             (std::abs(elevation - entry->second.elevation) <= ROUTING_ELEVATION_EPSILON_METERS &&
              coordinate < entry->second.coordinate))) {
            entry->second = {.elevation = elevation, .coordinate = coordinate};
        }
    }

    // A coarse 256-block anchor keeps the usual one-component preview/final
    // refinement stable. It is not enough to distinguish two independent
    // depressions that happen to fall in the same anchor, so only those
    // collisions receive an exact local-component ID. Proven opposing edge
    // portals reconcile those local IDs through the tiled spill hierarchy below.
    std::map<std::pair<int64_t, int64_t>, std::vector<int32_t>> rootsByCoarseAnchor;
    for (const auto& [root, anchor] : lakeAnchors) {
        rootsByCoarseAnchor[{world_coord::floorMultiple(anchor.coordinate.first,
                                                        LOCAL_LAKE_COARSE_ANCHOR_BLOCKS),
                             world_coord::floorMultiple(anchor.coordinate.second,
                                                        LOCAL_LAKE_COARSE_ANCHOR_BLOCKS)}]
            .push_back(root);
    }
    std::vector<WaterBodyId> lakeComponentIds;
    if (!lakeAnchors.empty()) lakeComponentIds.resize(RASTER_CELLS, NO_WATER_BODY);
    for (auto& [coarseAnchor, roots] : rootsByCoarseAnchor) {
        static_cast<void>(coarseAnchor);
        std::ranges::sort(roots, [&lakeAnchors](int32_t first, int32_t second) {
            return lakeAnchors.at(first).coordinate < lakeAnchors.at(second).coordinate;
        });
        const bool collision = roots.size() > 1;
        for (const int32_t root : roots) {
            const auto localAnchor = lakeAnchors.at(root).coordinate;
            const WaterBodyId body =
                collision
                    ? identities.disambiguatedLocalLakeBodyId(localAnchor.first, localAnchor.second)
                    : identities.localLakeBodyId(localAnchor.first, localAnchor.second);
            lakeComponentIds[static_cast<size_t>(root)] = body;
        }
    }

    // From this point forward, the page is game-facing authority. Preserve
    // raw terrain, standing stages, and hydraulic head in world-block
    // coordinates while the routing vectors above retain native meters.
    page->rawElevation.resize(RASTER_CELLS);
    page->groundwaterHead.resize(RASTER_CELLS);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const size_t cell = static_cast<size_t>(index);
        page->rawElevation[cell] = worldHeightFromNativeMeters(routingElevationMeters[cell]);
        // A hydraulic head is continuous even though raw terrain is emitted
        // through the shared terrain-height conversion. Keep its physical
        // meter depth when expressing it in the same world-block coordinate.
        page->groundwaterHead[cell] = page->rawElevation[cell] -
                                      (routingElevationMeters[cell] - groundwaterHeadMeters[cell]) /
                                          static_cast<float>(learned::WORLD_METERS_PER_BLOCK);
    }

    page->waterSurface.assign(RASTER_CELLS, 0.0F);
    page->groundwaterRecharge = localRecharge;
    page->channelGradient.assign(RASTER_CELLS, 0.0F);
    page->lakeShoreDistance.assign(RASTER_CELLS, -1.0e9F);
    page->waterBodyIds.assign(RASTER_CELLS, NO_WATER_BODY);
    page->streamOrder.assign(RASTER_CELLS, 0);
    page->flags.assign(RASTER_CELLS, 0);
    page->waterfallBranchMasks.assign(RASTER_CELLS, 0);
    std::vector<uint16_t> estuaryDistanceCells(RASTER_CELLS, std::numeric_limits<uint16_t>::max());
    const WaterBodyId oceanId = stableWaterBodyId(seed, 0, 0, 0x4F43'4541'4E00'0001ULL);
    page->oceanId = oceanId;
    for (const int index : floodOrder) {
        const size_t cell = static_cast<size_t>(index);
        const int x = index % RASTER_EDGE;
        const int z = index / RASTER_EDGE;
        if (routingElevationMeters[cell] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS) {
            page->flags[cell] |= CELL_OCEAN;
            page->waterSurface[cell] = SEA_LEVEL;
            page->waterBodyIds[cell] = oceanId;
            continue;
        }
        if (lakeSets.active(index)) {
            const int32_t root = lakeSets.find(index);
            const WaterBodyId body = lakeComponentIds[static_cast<size_t>(root)];
            if (body == NO_WATER_BODY)
                throw std::runtime_error("native hydrology lake component identity is missing");
            page->flags[cell] |= CELL_LAKE;
            page->waterSurface[cell] = worldHeightFromNativeMeters(filledMeters[cell]);
            page->waterBodyIds[cell] = body;
            LakeStats& stats = page->lakeStats[body];
            stats.areaSquareKilometers += CELL_AREA_SQUARE_KILOMETERS;
            stats.volumeCubicMeters += (filledMeters[cell] - routingElevationMeters[cell]) *
                                       NATIVE_HYDROLOGY_CELL_EDGE_METERS *
                                       NATIVE_HYDROLOGY_CELL_EDGE_METERS;
            stats.runoffMmSquareKilometers += localRunoff[cell];
            continue;
        }
        if (page->discharge[cell] < MINIMUM_CHANNEL_DISCHARGE) {
            const float headDistanceMeters =
                routingElevationMeters[cell] - groundwaterHeadMeters[cell];
            if (headDistanceMeters <= WETLAND_MAX_GROUNDWATER_DEPTH_METERS &&
                page->seasonality[cell] < WETLAND_MAX_SEASONALITY)
                page->flags[cell] |= CELL_WETLAND_CANDIDATE;
            continue;
        }

        const Receiver receiver = receivers[cell];
        const float naturalStage =
            std::max(static_cast<float>(SEA_LEVEL),
                     page->rawElevation[cell] - RIVER_NATURAL_STAGE_BELOW_TERRAIN_BLOCKS);
        float downstreamStage = naturalStage;
        float highestReceiverStage = -std::numeric_limits<float>::infinity();
        float maximumOrdinaryStage = std::numeric_limits<float>::infinity();
        float maximumDropMeters = 0.0F;
        WaterBodyId inheritedRiverBody = NO_WATER_BODY;
        uint16_t downstreamEstuaryDistance = std::numeric_limits<uint16_t>::max();
        const std::array targets{receiver.first, receiver.second};
        for (int branch = 0; branch < static_cast<int>(targets.size()); ++branch) {
            if (!receiverBranchActive(receiver, branch)) continue;
            const int target = targets[static_cast<size_t>(branch)];
            const size_t targetCell = static_cast<size_t>(target);
            const float targetStage = page->waterSurface[targetCell];
            if (std::isfinite(targetStage) && targetStage > 0.0F) {
                downstreamStage = std::max(downstreamStage, targetStage);
                highestReceiverStage = std::max(highestReceiverStage, targetStage);
                const int targetX = target % RASTER_EDGE;
                const int targetZ = target / RASTER_EDGE;
                const float runBlocks = static_cast<float>(std::hypot(targetX - x, targetZ - z) *
                                                           NATIVE_HYDROLOGY_RASTER_SPACING);
                maximumOrdinaryStage =
                    std::min(maximumOrdinaryStage,
                             targetStage + runBlocks * RIVER_MAXIMUM_ORDINARY_STAGE_DROP_PER_BLOCK);
                maximumDropMeters =
                    std::max(maximumDropMeters,
                             routingElevationMeters[cell] - routingElevationMeters[targetCell]);
            }
            if ((page->flags[targetCell] & CELL_RIVER) != 0 &&
                page->waterBodyIds[targetCell] != NO_WATER_BODY) {
                inheritedRiverBody =
                    inheritedRiverBody == NO_WATER_BODY
                        ? page->waterBodyIds[targetCell]
                        : std::min(inheritedRiverBody, page->waterBodyIds[targetCell]);
            }
            if ((page->flags[targetCell] & CELL_OCEAN) != 0) {
                downstreamEstuaryDistance = 0;
            } else if ((page->flags[targetCell] & CELL_ESTUARY) != 0 &&
                       estuaryDistanceCells[targetCell] != std::numeric_limits<uint16_t>::max()) {
                downstreamEstuaryDistance =
                    std::min(downstreamEstuaryDistance, estuaryDistanceCells[targetCell]);
            }
        }
        // Stage remains monotone toward every compatible finished receiver.
        // If learned height quantization leaves too little substrate beneath
        // that stage, exact and far sampling lower the canonical bed. Clamping
        // stage below the raw surface here would instead create an uphill
        // water step, most visibly where a sea-backed reach leaves Y=64.
        float stage = std::max(naturalStage, downstreamStage);
        const bool incompatibleBranchStages = std::isfinite(maximumOrdinaryStage) &&
                                              highestReceiverStage > maximumOrdinaryStage + 1.0e-5F;
        const bool waterfall =
            maximumDropMeters > WATERFALL_MINIMUM_DROP_METERS || incompatibleBranchStages;
        if (!waterfall && std::isfinite(maximumOrdinaryStage))
            stage = std::min(stage, maximumOrdinaryStage);
        const uint8_t order = static_cast<uint8_t>(
            std::clamp(1 + static_cast<int>(std::floor(std::log2(
                               std::max(1.0F, page->discharge[cell] / MINIMUM_CHANNEL_DISCHARGE)))),
                       1, 12));
        page->flags[cell] |= CELL_RIVER;
        page->waterSurface[cell] = stage;
        page->streamOrder[cell] = order;
        if (receiver.first >= 0) {
            const int receiverX = receiver.first % RASTER_EDGE;
            const int receiverZ = receiver.first / RASTER_EDGE;
            const double run =
                std::hypot(receiverX - x, receiverZ - z) * NATIVE_HYDROLOGY_CELL_EDGE_METERS;
            const float dropMeters = routingElevationMeters[cell] -
                                     routingElevationMeters[static_cast<size_t>(receiver.first)];
            page->channelGradient[cell] = static_cast<float>(std::max(0.0, dropMeters / run));
            page->waterBodyIds[cell] =
                inheritedRiverBody != NO_WATER_BODY
                    ? inheritedRiverBody
                    : stableWaterBodyId(seed, checkedCoordinate(key.x, x - RASTER_APRON),
                                        checkedCoordinate(key.z, z - RASTER_APRON),
                                        0x5249'5645'5200'0001ULL);
            if (waterfall) page->flags[cell] |= CELL_WATERFALL;
        } else {
            page->waterBodyIds[cell] = stableWaterBodyId(
                seed, checkedCoordinate(key.x, x - RASTER_APRON),
                checkedCoordinate(key.z, z - RASTER_APRON), 0x5249'5645'5200'0001ULL);
        }

        if (!waterfall && page->channelGradient[cell] <= ESTUARY_MAXIMUM_CHANNEL_GRADIENT &&
            downstreamEstuaryDistance < ESTUARY_MAXIMUM_BACKWATER_NATIVE_CELLS) {
            page->flags[cell] |= CELL_ESTUARY;
            estuaryDistanceCells[cell] = static_cast<uint16_t>(downstreamEstuaryDistance + 1U);
            // The unmodified learned coast can quantize to the sea-level
            // block even though its physical elevation remains positive.
            // Preserve sea backwater as stage authority and let sampling cut
            // the supported bed downward. Constraining stage below that raw
            // surface would create an uphill final step into the ocean.
            page->waterSurface[cell] =
                std::max(page->waterSurface[cell], static_cast<float>(SEA_LEVEL));
        }

        Receiver& mutableReceiver = receivers[cell];
        const auto oceanReceiverCount = [&]() {
            int count = 0;
            if (receiverBranchActive(mutableReceiver, 0) &&
                (page->flags[static_cast<size_t>(mutableReceiver.first)] & CELL_OCEAN) != 0) {
                ++count;
            }
            if (receiverBranchActive(mutableReceiver, 1) &&
                (page->flags[static_cast<size_t>(mutableReceiver.second)] & CELL_OCEAN) != 0) {
                ++count;
            }
            return count;
        };
        if ((page->flags[cell] & CELL_ESTUARY) != 0 && downstreamEstuaryDistance == 0 &&
            !receiverBranchActive(mutableReceiver, 1) && oceanReceiverCount() == 1) {
            const double sourceFlowX = page->flowX[cell];
            const double sourceFlowZ = page->flowZ[cell];
            const uint64_t phase = mix64(
                seed ^ static_cast<uint64_t>(checkedCoordinate(key.x, x - RASTER_APRON)) ^
                std::rotl(static_cast<uint64_t>(checkedCoordinate(key.z, z - RASTER_APRON)), 23));
            const double preferredSide = (phase & 1U) != 0 ? 1.0 : -1.0;
            std::optional<std::tuple<double, int, double, double, int64_t, int64_t, int>> branch;
            for (int offsetZ = -DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS;
                 offsetZ <= DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS; ++offsetZ) {
                for (int offsetX = -DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS;
                     offsetX <= DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS; ++offsetX) {
                    if (offsetX == 0 && offsetZ == 0) continue;
                    const int candidateX = x + offsetX;
                    const int candidateZ = z + offsetZ;
                    if (!inside(candidateX, candidateZ)) continue;
                    const int candidate = indexOf(candidateX, candidateZ);
                    if (candidate == mutableReceiver.first ||
                        (page->flags[static_cast<size_t>(candidate)] & CELL_OCEAN) == 0) {
                        continue;
                    }
                    const double length = std::hypot(offsetX, offsetZ);
                    if (length > DELTA_DISTRIBUTARY_TARGET_RADIUS_NATIVE_CELLS + 1.0e-9) continue;
                    const double alignment =
                        (offsetX * sourceFlowX + offsetZ * sourceFlowZ) / length;
                    if (alignment < 0.25) continue;
                    const double lateral = std::abs(offsetX * sourceFlowZ - offsetZ * sourceFlowX);
                    if (lateral < 0.75) continue;
                    const double signedLateral = -offsetX * sourceFlowZ + offsetZ * sourceFlowX;
                    const int sideRank = signedLateral * preferredSide > 0.0 ? 0 : 1;
                    const int64_t candidateWorldX =
                        checkedCoordinate(key.x, candidateX - RASTER_APRON);
                    const int64_t candidateWorldZ =
                        checkedCoordinate(key.z, candidateZ - RASTER_APRON);
                    const auto ranked =
                        std::tuple{-lateral,        sideRank,        -alignment, -length,
                                   candidateWorldX, candidateWorldZ, candidate};
                    if (!branch || ranked < *branch) branch = ranked;
                }
            }
            if (branch) {
                mutableReceiver.second = std::get<6>(*branch);
                mutableReceiver.secondWeight =
                    0.32F + static_cast<float>(phase & 0xFFFFU) / 65'535.0F * 0.10F;
            }
        }
        if ((page->flags[cell] & CELL_ESTUARY) != 0 && downstreamEstuaryDistance == 0 &&
            oceanReceiverCount() >= 2) {
            page->flags[cell] |= CELL_DELTA;
        }
        page->receiverFirst[cell] = mutableReceiver.first;
        page->receiverSecond[cell] = mutableReceiver.second;
        page->receiverSecondWeight[cell] = mutableReceiver.secondWeight;
    }
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const size_t cell = static_cast<size_t>(index);
        if ((page->flags[cell] & CELL_RIVER) != 0 && reachLabels[cell] != NO_WATER_BODY) {
            page->waterBodyIds[cell] = reachLabels[cell];
        }
    }
    // Resolve the complete downstream candidate graph in the same immutable
    // flood order as rivers. A candidate can inherit through another finished
    // wetland, while unresolved page-edge paths retain their candidate bit for
    // the bounded cross-page resolver at query time.
    for (const int index : floodOrder) {
        const size_t cell = static_cast<size_t>(index);
        if ((page->flags[cell] & CELL_WETLAND_CANDIDATE) == 0) continue;

        int parent = -1;
        float parentStage = 0.0F;
        WaterBodyId parentBody = NO_WATER_BODY;
        const auto considerParent = [&](int receiver) {
            if (receiver < 0) return;
            const size_t downstream = static_cast<size_t>(receiver);
            const uint8_t downstreamFlags = page->flags[downstream];
            if ((downstreamFlags & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) == 0 ||
                (downstreamFlags & CELL_WATERFALL) != 0 ||
                page->waterBodyIds[downstream] == NO_WATER_BODY) {
                return;
            }
            const float stage = page->waterSurface[downstream];
            if (!std::isfinite(stage) || std::abs(page->rawElevation[cell] - stage) >
                                             WETLAND_MAX_PARENT_STAGE_OFFSET_BLOCKS) {
                return;
            }
            const WaterBodyId body = page->waterBodyIds[downstream];
            if (parent < 0 || stage < parentStage - WETLAND_PARENT_STAGE_EPSILON_BLOCKS ||
                (std::abs(stage - parentStage) <= WETLAND_PARENT_STAGE_EPSILON_BLOCKS &&
                 std::pair{body, receiver} < std::pair{parentBody, parent})) {
                parent = receiver;
                parentStage = stage;
                parentBody = body;
            }
        };
        const Receiver receiver = receivers[cell];
        considerParent(receiver.first);
        considerParent(receiver.second);
        if (parent < 0) continue;
        page->flags[cell] |= CELL_WETLAND;
        page->waterSurface[cell] = parentStage;
        page->waterBodyIds[cell] = parentBody;
    }
    freezeWaterfallBranchMasks(*page);

    std::vector<uint8_t> lakeMask(RASTER_CELLS, 0);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        if ((page->flags[static_cast<size_t>(index)] & CELL_LAKE) != 0)
            lakeMask[static_cast<size_t>(index)] = 1;
    }
    const std::vector<float> distanceToDry = squaredDistanceTransform(lakeMask, 0, RASTER_EDGE);
    const std::vector<float> distanceToLake = squaredDistanceTransform(lakeMask, 1, RASTER_EDGE);
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const bool lake = lakeMask[static_cast<size_t>(index)] != 0;
        const float squared = lake ? distanceToDry[static_cast<size_t>(index)]
                                   : distanceToLake[static_cast<size_t>(index)];
        const float distance = std::isfinite(squared)
                                   ? (std::sqrt(squared) - 0.5F) * NATIVE_HYDROLOGY_RASTER_SPACING
                                   : 1.0e9F;
        page->lakeShoreDistance[static_cast<size_t>(index)] = lake ? distance : -distance;
    }

    const auto summarizeEdge = [&](PageEdge edge) {
        std::vector<EdgeSummarySample>& summary = page->edgeSummaries[static_cast<size_t>(edge)];
        summary.reserve(CORE_EDGE_SAMPLES);
        for (int along = CORE_BEGIN; along <= CORE_END; ++along) {
            int rasterX = along;
            int rasterZ = along;
            switch (edge) {
                case PageEdge::WEST:
                    rasterX = CORE_BEGIN;
                    break;
                case PageEdge::EAST:
                    rasterX = CORE_END;
                    break;
                case PageEdge::NORTH:
                    rasterZ = CORE_BEGIN;
                    break;
                case PageEdge::SOUTH:
                    rasterZ = CORE_END;
                    break;
            }
            const size_t cell = static_cast<size_t>(indexOf(rasterX, rasterZ));
            summary.push_back({
                .rawElevation = page->rawElevation[cell],
                .waterSurface = page->waterSurface[cell],
                .discharge = page->discharge[cell],
                .waterBodyId = page->waterBodyIds[cell],
                .flags = page->flags[cell],
            });
        }
    };
    summarizeEdge(PageEdge::WEST);
    summarizeEdge(PageEdge::EAST);
    summarizeEdge(PageEdge::NORTH);
    summarizeEdge(PageEdge::SOUTH);

    std::map<WaterBodyId, DepressionSummary> depressions;
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const size_t cell = static_cast<size_t>(indexOf(x, z));
            if ((page->flags[cell] & CELL_LAKE) == 0) continue;
            const WaterBodyId body = page->waterBodyIds[cell];
            const int32_t root = lakeSets.find(indexOf(x, z));
            const WaterBodyId componentBody = lakeComponentIds[static_cast<size_t>(root)];
            if (componentBody != body)
                throw std::runtime_error("native hydrology lake component summary is inconsistent");
            const auto localAnchor = lakeAnchors.at(root).coordinate;
            auto [found, inserted] =
                depressions.try_emplace(body, DepressionSummary{
                                                  .waterBodyId = body,
                                                  .localAnchorX = localAnchor.first,
                                                  .localAnchorZ = localAnchor.second,
                                                  .localStage = page->waterSurface[cell],
                                              });
            DepressionSummary& summary = found->second;
            static_cast<void>(inserted);
            summary.localStage = std::min(summary.localStage, page->waterSurface[cell]);
            if (x >= CORE_BEGIN && x < CORE_END && z >= CORE_BEGIN && z < CORE_END) {
                summary.coreAreaSquareKilometers += CELL_AREA_SQUARE_KILOMETERS;
                summary.coreVolumeCubicMeters +=
                    (filledMeters[cell] - routingElevationMeters[cell]) *
                    NATIVE_HYDROLOGY_CELL_EDGE_METERS * NATIVE_HYDROLOGY_CELL_EDGE_METERS;
                summary.coreRunoffMmSquareKilometers += localRunoff[cell];
            }
            if (x == CORE_BEGIN) summary.edgeMask |= 1U << static_cast<uint8_t>(PageEdge::WEST);
            if (x == CORE_END) summary.edgeMask |= 1U << static_cast<uint8_t>(PageEdge::EAST);
            if (z == CORE_BEGIN) summary.edgeMask |= 1U << static_cast<uint8_t>(PageEdge::NORTH);
            if (z == CORE_END) summary.edgeMask |= 1U << static_cast<uint8_t>(PageEdge::SOUTH);
        }
    }
    // Record only an outlet proven strictly inside the half-open owner core.
    // A route into the apron may be an artificial local-page terminal and is
    // therefore connectivity evidence, not a natural basin outlet. The tiled
    // hierarchy selects one compatible outlet after it closes the component.
    for (int z = CORE_BEGIN; z < CORE_END; ++z) {
        for (int x = CORE_BEGIN; x < CORE_END; ++x) {
            const int index = indexOf(x, z);
            const size_t cell = static_cast<size_t>(index);
            if ((page->flags[cell] & CELL_LAKE) == 0) continue;
            const int32_t root = lakeSets.find(index);
            const WaterBodyId body = lakeComponentIds[static_cast<size_t>(root)];
            DepressionSummary& summary = depressions.at(body);
            const Receiver receiver = receivers[cell];
            const std::array targets{receiver.first, receiver.second};
            for (int branch = 0; branch < static_cast<int>(targets.size()); ++branch) {
                if (!receiverBranchActive(receiver, branch)) continue;
                const int target = targets[static_cast<size_t>(branch)];
                if (target < 0) continue;
                const int targetX = target % RASTER_EDGE;
                const int targetZ = target / RASTER_EDGE;
                if (targetX < CORE_BEGIN || targetX >= CORE_END || targetZ < CORE_BEGIN ||
                    targetZ >= CORE_END) {
                    continue;
                }
                const bool sameLake = lakeSets.active(target) && lakeSets.find(target) == root;
                if (sameLake) continue;

                const float outletStage = page->waterSurface[cell];
                const int64_t outletX = checkedCoordinate(key.x, targetX - RASTER_APRON);
                const int64_t outletZ = checkedCoordinate(key.z, targetZ - RASTER_APRON);
                const BasinOutlet outlet =
                    (page->flags[static_cast<size_t>(target)] & CELL_OCEAN) != 0
                        ? BasinOutlet::OCEAN
                        : BasinOutlet::SHARED_PORTAL;
                const bool replace =
                    summary.naturalOutlet == BasinOutlet::NONE ||
                    outletStage < summary.naturalOutletStage - ROUTING_ELEVATION_EPSILON_METERS ||
                    (std::abs(outletStage - summary.naturalOutletStage) <=
                         ROUTING_ELEVATION_EPSILON_METERS &&
                     std::tuple{outletX, outletZ, outlet} < std::tuple{summary.naturalOutletX,
                                                                       summary.naturalOutletZ,
                                                                       summary.naturalOutlet});
                if (replace) {
                    summary.naturalOutlet = outlet;
                    summary.naturalOutletX = outletX;
                    summary.naturalOutletZ = outletZ;
                    summary.naturalOutletStage = outletStage;
                }
            }
        }
    }
    page->depressionSummaries.reserve(depressions.size());
    for (const auto& [body, summary] : depressions) {
        static_cast<void>(body);
        page->depressionSummaries.push_back(summary);
    }
    buildNativePageSamplingIndexes(*page);

    size_t lakeIdentityBytes = lakeComponentIds.capacity() * sizeof(WaterBodyId);
    for (const auto& [coarseAnchor, roots] : rootsByCoarseAnchor) {
        static_cast<void>(coarseAnchor);
        lakeIdentityBytes += roots.capacity() * sizeof(int32_t);
    }
    const size_t scratchBytes =
        positions.capacity() * sizeof(NativeHydrologyPosition) +
        inputs.capacity() * sizeof(NativeHydrologyInput) +
        // Physical elevation and local runoff are already counted in the
        // immutable page payload above; only build-local fields belong here.
        groundwaterHeadMeters.capacity() * sizeof(float) + filledMeters.capacity() * sizeof(float) +
        visited.capacity() * sizeof(uint8_t) + floodParent.capacity() * sizeof(int32_t) +
        floodRank.capacity() * sizeof(int32_t) + floodOrder.capacity() * sizeof(int32_t) +
        receivers.capacity() * sizeof(Receiver) + localBaseflow.capacity() * sizeof(float) +
        localRecharge.capacity() * sizeof(float) + lakeSets.parents.capacity() * sizeof(int32_t) +
        lakeMask.capacity() * sizeof(uint8_t) + distanceToDry.capacity() * sizeof(float) +
        distanceToLake.capacity() * sizeof(float) +
        estuaryDistanceCells.capacity() * sizeof(uint16_t) + lakeIdentityBytes +
        static_cast<size_t>(RASTER_CELLS) * (sizeof(int32_t) + sizeof(uint8_t));
    peakBuildBytes = page->byteSize() + scratchBytes;
    if (page->byteSize() > NATIVE_HYDROLOGY_MAX_PAGE_BYTES ||
        peakBuildBytes > NATIVE_HYDROLOGY_MAX_BUILD_BYTES) {
        throw std::runtime_error("native hydrology page exceeded its memory bound");
    }
    return page;
}

uint8_t computeWaterfallBranchMask(const NativePage& page, size_t sourceCell) {
    if ((page.flags[sourceCell] & (CELL_RIVER | CELL_WATERFALL)) != (CELL_RIVER | CELL_WATERFALL)) {
        return 0;
    }
    const std::array<int32_t, 2> targets{page.receiverFirst[sourceCell],
                                         page.receiverSecond[sourceCell]};
    const std::array<float, 2> weights{
        targets[1] >= 0 ? 1.0F - page.receiverSecondWeight[sourceCell] : 1.0F,
        targets[1] >= 0 ? page.receiverSecondWeight[sourceCell] : 0.0F,
    };
    uint8_t mask = 0;
    for (size_t branch = 0; branch < targets.size(); ++branch) {
        if (targets[branch] < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT) continue;
        const size_t targetCell = static_cast<size_t>(targets[branch]);
        if ((page.flags[targetCell] & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) == 0 ||
            !std::isfinite(page.waterSurface[sourceCell]) ||
            !std::isfinite(page.waterSurface[targetCell]) ||
            page.waterSurface[sourceCell] - page.waterSurface[targetCell] <
                MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS) {
            continue;
        }
        mask |= static_cast<uint8_t>(1U << branch);
    }
    return mask;
}

bool waterfallBranchMaskIsValid(const NativePage& page, size_t sourceCell) {
    const uint8_t mask = page.waterfallBranchMasks[sourceCell];
    if (mask != computeWaterfallBranchMask(page, sourceCell)) return false;
    if (((page.flags[sourceCell] & CELL_WATERFALL) != 0) != (mask != 0)) return false;
    const std::array<int32_t, 2> targets{page.receiverFirst[sourceCell],
                                         page.receiverSecond[sourceCell]};
    const std::array<float, 2> weights{
        targets[1] >= 0 ? 1.0F - page.receiverSecondWeight[sourceCell] : 1.0F,
        targets[1] >= 0 ? page.receiverSecondWeight[sourceCell] : 0.0F,
    };
    for (size_t branch = 0; branch < targets.size(); ++branch) {
        if ((mask & static_cast<uint8_t>(1U << branch)) == 0) continue;
        if (targets[branch] < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT) return false;
        const size_t targetCell = static_cast<size_t>(targets[branch]);
        if ((page.flags[targetCell] & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) == 0 ||
            !std::isfinite(page.waterSurface[sourceCell]) ||
            !std::isfinite(page.waterSurface[targetCell]) ||
            page.waterSurface[sourceCell] - page.waterSurface[targetCell] <
                MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS) {
            return false;
        }
    }
    return true;
}

void freezeWaterfallBranchMasks(NativePage& page) {
    page.waterfallBranchMasks.resize(RASTER_CELLS);
    for (size_t sourceCell = 0; sourceCell < RASTER_CELLS; ++sourceCell) {
        const uint8_t mask = computeWaterfallBranchMask(page, sourceCell);
        page.waterfallBranchMasks[sourceCell] = mask;
        if (mask == 0)
            page.flags[sourceCell] &= static_cast<uint8_t>(~CELL_WATERFALL);
        else
            page.flags[sourceCell] |= CELL_WATERFALL;
    }
}

bool receiverBranchHasExplicitFall(const NativePage& page, size_t sourceCell, size_t branch) {
    return branch < 2 && sourceCell < page.waterfallBranchMasks.size() &&
           (page.waterfallBranchMasks[sourceCell] & static_cast<uint8_t>(1U << branch)) != 0;
}

void buildOrdinaryReachIndex(NativePage& page) {
    std::vector<int32_t> reachParents(RASTER_CELLS, -1);
    for (int cell = 0; cell < RASTER_CELLS; ++cell) {
        if ((page.flags[static_cast<size_t>(cell)] & CELL_RIVER) != 0)
            reachParents[static_cast<size_t>(cell)] = cell;
    }
    const auto findReach = [&reachParents](int cell) {
        int root = cell;
        while (reachParents[static_cast<size_t>(root)] != root)
            root = reachParents[static_cast<size_t>(root)];
        while (cell != root) {
            const int parent = reachParents[static_cast<size_t>(cell)];
            reachParents[static_cast<size_t>(cell)] = root;
            cell = parent;
        }
        return root;
    };
    const auto uniteReach = [&reachParents, &findReach](int first, int second) {
        const int firstRoot = findReach(first);
        const int secondRoot = findReach(second);
        if (firstRoot == secondRoot) return;
        const int low = std::min(firstRoot, secondRoot);
        const int high = std::max(firstRoot, secondRoot);
        reachParents[static_cast<size_t>(high)] = low;
    };
    for (int source = 0; source < RASTER_CELLS; ++source) {
        const size_t sourceCell = static_cast<size_t>(source);
        if ((page.flags[sourceCell] & CELL_RIVER) == 0) continue;
        const std::array<int32_t, 2> targets{page.receiverFirst[sourceCell],
                                             page.receiverSecond[sourceCell]};
        const std::array<float, 2> weights{
            targets[1] >= 0 ? 1.0F - page.receiverSecondWeight[sourceCell] : 1.0F,
            targets[1] >= 0 ? page.receiverSecondWeight[sourceCell] : 0.0F,
        };
        for (size_t branch = 0; branch < targets.size(); ++branch) {
            const int target = targets[branch];
            if (target < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT ||
                receiverBranchHasExplicitFall(page, sourceCell, branch) ||
                (page.flags[static_cast<size_t>(target)] & CELL_RIVER) == 0) {
                continue;
            }
            uniteReach(source, target);
        }
    }
    page.ordinaryReachIds.assign(RASTER_CELLS, 0);
    for (int cell = 0; cell < RASTER_CELLS; ++cell) {
        if (reachParents[static_cast<size_t>(cell)] < 0) continue;
        page.ordinaryReachIds[static_cast<size_t>(cell)] =
            static_cast<uint32_t>(findReach(cell) + 1);
    }
}

void buildNativePageSamplingIndexes(NativePage& page) {
    // The coarse horizon has thousands of mesh points but only a small
    // fraction lie near a river. A separable square dilation exactly matches
    // projectChannel's candidate radius while making dry samples constant
    // time. Its data is derived from persisted flags, so it cannot affect
    // generator identity or on-disk compatibility.
    page.channelRowOffsets.assign(static_cast<size_t>(RASTER_EDGE + 1), 0);
    page.channelSourceXs.clear();
    page.hasWetlandCandidates = false;
    page.hasCoastalResolutionCandidates = false;
    for (int z = 0; z < RASTER_EDGE; ++z) {
        page.channelRowOffsets[static_cast<size_t>(z)] =
            static_cast<uint32_t>(page.channelSourceXs.size());
        for (int x = 0; x < RASTER_EDGE; ++x) {
            if ((page.flags[static_cast<size_t>(indexOf(x, z))] & CELL_RIVER) != 0) {
                page.channelSourceXs.push_back(static_cast<uint16_t>(x));
            }
            const size_t cell = static_cast<size_t>(indexOf(x, z));
            const uint8_t flags = page.flags[cell];
            if ((flags & CELL_WETLAND_CANDIDATE) != 0) page.hasWetlandCandidates = true;
            if ((flags & CELL_RIVER) != 0 && (flags & CELL_ESTUARY) == 0 &&
                page.channelGradient[cell] <= ESTUARY_MAXIMUM_CHANNEL_GRADIENT &&
                (flags & CELL_WATERFALL) == 0) {
                page.hasCoastalResolutionCandidates = true;
            }
        }
    }
    page.channelRowOffsets[static_cast<size_t>(RASTER_EDGE)] =
        static_cast<uint32_t>(page.channelSourceXs.size());

    buildOrdinaryReachIndex(page);

    page.receiverOwnedOutlets.clear();
    page.receiverOwnedOutlets.reserve(page.depressionSummaries.size());
    for (const DepressionSummary& summary : page.depressionSummaries) {
        // An edge-open component can acquire a lower resolved stage from the
        // bounded cross-page spill hierarchy. Its page-local stage is not
        // final standing-water authority and cannot own a curtain or
        // backwater blend. Closed summaries retain an immutable local stage.
        if (summary.edgeMask != 0 || summary.naturalOutlet == BasinOutlet::NONE ||
            summary.naturalOutlet == BasinOutlet::ENDORHEIC) {
            continue;
        }
        const int64_t localX = summary.naturalOutletX - page.originX;
        const int64_t localZ = summary.naturalOutletZ - page.originZ;
        if (localX % NATIVE_HYDROLOGY_RASTER_SPACING != 0 ||
            localZ % NATIVE_HYDROLOGY_RASTER_SPACING != 0) {
            continue;
        }
        const int rasterX =
            static_cast<int>(localX / NATIVE_HYDROLOGY_RASTER_SPACING) + RASTER_APRON;
        const int rasterZ =
            static_cast<int>(localZ / NATIVE_HYDROLOGY_RASTER_SPACING) + RASTER_APRON;
        if (!inside(rasterX, rasterZ)) continue;
        const int source = indexOf(rasterX, rasterZ);
        const size_t sourceCell = static_cast<size_t>(source);
        if ((page.flags[sourceCell] & CELL_RIVER) == 0 ||
            page.waterBodyIds[sourceCell] == NO_WATER_BODY) {
            continue;
        }
        page.receiverOwnedOutlets.push_back({.source = source,
                                             .standingStage = summary.localStage,
                                             .standingBody = summary.waterBodyId});
    }
    std::ranges::sort(page.receiverOwnedOutlets);
    page.receiverOwnedOutlets.erase(std::unique(page.receiverOwnedOutlets.begin(),
                                                page.receiverOwnedOutlets.end(),
                                                [](const ReceiverOwnedOutletIndexEntry& first,
                                                   const ReceiverOwnedOutletIndexEntry& second) {
                                                    return first.source == second.source;
                                                }),
                                    page.receiverOwnedOutlets.end());

    std::vector<uint8_t> horizontal(RASTER_CELLS, 0);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        int riverCount = 0;
        for (int x = -CHANNEL_SEARCH_RADIUS; x < RASTER_EDGE; ++x) {
            const int entering = x + CHANNEL_SEARCH_RADIUS;
            const int leaving = x - CHANNEL_SEARCH_RADIUS - 1;
            if (entering < RASTER_EDGE &&
                (page.flags[static_cast<size_t>(indexOf(entering, z))] & CELL_RIVER) != 0) {
                ++riverCount;
            }
            if (leaving >= 0 &&
                (page.flags[static_cast<size_t>(indexOf(leaving, z))] & CELL_RIVER) != 0) {
                --riverCount;
            }
            if (x >= 0) horizontal[static_cast<size_t>(indexOf(x, z))] = riverCount > 0 ? 1U : 0U;
        }
    }
    page.channelProximity.assign(RASTER_CELLS, 0);
    for (int x = 0; x < RASTER_EDGE; ++x) {
        int riverCount = 0;
        for (int z = -CHANNEL_SEARCH_RADIUS; z < RASTER_EDGE; ++z) {
            const int entering = z + CHANNEL_SEARCH_RADIUS;
            const int leaving = z - CHANNEL_SEARCH_RADIUS - 1;
            if (entering < RASTER_EDGE &&
                horizontal[static_cast<size_t>(indexOf(x, entering))] != 0) {
                ++riverCount;
            }
            if (leaving >= 0 && horizontal[static_cast<size_t>(indexOf(x, leaving))] != 0) {
                --riverCount;
            }
            if (z >= 0)
                page.channelProximity[static_cast<size_t>(indexOf(x, z))] =
                    riverCount > 0 ? 1U : 0U;
        }
    }
    // Step-32 far parents must not infer water solely from their corners and
    // center. Reduce the already solved four-block authority to one compact
    // bit field per 32-block half-open cell instead. A uniformly wet cell
    // with one standing authority intentionally has no topology bit,
    // preserving the one-quad ocean and lake fast path. A narrow lake, inlet,
    // island, channel, or standing-authority boundary requests the canonical
    // raster exactly where it is needed.
    page.topologyCells.assign(TOPOLOGY_CELLS, 0);
    for (int topologyZ = 0; topologyZ < TOPOLOGY_EDGE; ++topologyZ) {
        for (int topologyX = 0; topologyX < TOPOLOGY_EDGE; ++topologyX) {
            uint8_t topology = 0;
            std::optional<std::tuple<uint8_t, WaterBodyId, float>> standingAuthority;
            const int firstRasterX = CORE_BEGIN + topologyX * TOPOLOGY_CELL_RASTER_EDGE;
            const int firstRasterZ = CORE_BEGIN + topologyZ * TOPOLOGY_CELL_RASTER_EDGE;
            for (int rasterZ = firstRasterZ; rasterZ < firstRasterZ + TOPOLOGY_CELL_RASTER_EDGE;
                 ++rasterZ) {
                for (int rasterX = firstRasterX; rasterX < firstRasterX + TOPOLOGY_CELL_RASTER_EDGE;
                     ++rasterX) {
                    const size_t cell = static_cast<size_t>(indexOf(rasterX, rasterZ));
                    const uint8_t flags = page.flags[cell];
                    const bool directlyWet =
                        (flags & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) != 0;
                    topology |= directlyWet ? TOPOLOGY_WET : TOPOLOGY_DRY;
                    if ((flags & CELL_WETLAND_CANDIDATE) != 0)
                        topology |= TOPOLOGY_WET | TOPOLOGY_DRY;
                    if (page.channelProximity[cell] != 0) topology |= TOPOLOGY_CHANNEL;
                    if ((flags & CELL_WATERFALL) != 0) topology |= TOPOLOGY_WATERFALL;
                    if (directlyWet && (flags & CELL_RIVER) == 0) {
                        const uint8_t kind = flags & (CELL_OCEAN | CELL_LAKE | CELL_WETLAND);
                        const auto authority =
                            std::tuple{kind, page.waterBodyIds[cell], page.waterSurface[cell]};
                        if (!standingAuthority) {
                            standingAuthority = authority;
                        } else if (*standingAuthority != authority) {
                            // A wholly wet 32-block cell can still contain a
                            // lake-to-ocean, lake-to-wetland, or distinct-lake
                            // boundary. Preserve that categorical transition
                            // in the same compact topology signal used for a
                            // hidden shoreline or channel.
                            topology |= TOPOLOGY_AUTHORITY_DISCONTINUITY;
                        }
                    }
                }
            }
            page.topologyCells[static_cast<size_t>(topologyZ * TOPOLOGY_EDGE + topologyX)] =
                topology;
        }
    }
    // A curved fall ribbon can cross a 32-block topology boundary even when
    // its persisted CELL_WATERFALL source remains in the neighboring cell.
    // Its receiver interval, maximum width, and bend are all shorter than one
    // topology cell, so a one-cell dilation around both persisted endpoints is
    // a complete conservative projection footprint.
    const auto markProjectedFallTopology = [&page](int endpoint) {
        if (endpoint < 0) return;
        const int rasterX = endpoint % RASTER_EDGE;
        const int rasterZ = endpoint / RASTER_EDGE;
        const int topologyX = world_coord::floorDiv(
            static_cast<int64_t>(rasterX - CORE_BEGIN) * NATIVE_HYDROLOGY_RASTER_SPACING,
            static_cast<int64_t>(NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE));
        const int topologyZ = world_coord::floorDiv(
            static_cast<int64_t>(rasterZ - CORE_BEGIN) * NATIVE_HYDROLOGY_RASTER_SPACING,
            static_cast<int64_t>(NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE));
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const int markedX = topologyX + offsetX;
                const int markedZ = topologyZ + offsetZ;
                if (markedX < 0 || markedX >= TOPOLOGY_EDGE || markedZ < 0 ||
                    markedZ >= TOPOLOGY_EDGE) {
                    continue;
                }
                page.topologyCells[static_cast<size_t>(markedZ * TOPOLOGY_EDGE + markedX)] |=
                    TOPOLOGY_CHANNEL | TOPOLOGY_WATERFALL;
            }
        }
    };
    for (int source = 0; source < RASTER_CELLS; ++source) {
        const size_t sourceCell = static_cast<size_t>(source);
        if ((page.flags[sourceCell] & CELL_WATERFALL) == 0) continue;
        const std::array<int32_t, 2> targets{page.receiverFirst[sourceCell],
                                             page.receiverSecond[sourceCell]};
        for (size_t branch = 0; branch < targets.size(); ++branch) {
            if (!receiverBranchHasExplicitFall(page, sourceCell, branch)) continue;
            markProjectedFallTopology(source);
            markProjectedFallTopology(targets[branch]);
        }
    }
    for (const ReceiverOwnedOutletIndexEntry& outlet : page.receiverOwnedOutlets) {
        if (outlet.source < 0) continue;
        const size_t sourceCell = static_cast<size_t>(outlet.source);
        const std::array<int32_t, 2> targets{page.receiverFirst[sourceCell],
                                             page.receiverSecond[sourceCell]};
        const std::array<float, 2> weights{
            targets[1] >= 0 ? 1.0F - page.receiverSecondWeight[sourceCell] : 1.0F,
            targets[1] >= 0 ? page.receiverSecondWeight[sourceCell] : 0.0F,
        };
        for (size_t branch = 0; branch < targets.size(); ++branch) {
            const int target = targets[branch];
            if (target < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT ||
                outlet.standingStage - page.waterSurface[static_cast<size_t>(target)] <
                    MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS) {
                continue;
            }
            markProjectedFallTopology(outlet.source);
            markProjectedFallTopology(target);
        }
    }
}

float reconstruct(const std::vector<float>& field, double gridX, double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double amountX = std::clamp(gridX - x0, 0.0, 1.0);
    const double amountZ = std::clamp(gridZ - z0, 0.0, 1.0);
    const double north = std::lerp(field[static_cast<size_t>(indexOf(x0, z0))],
                                   field[static_cast<size_t>(indexOf(x0 + 1, z0))], amountX);
    const double south = std::lerp(field[static_cast<size_t>(indexOf(x0, z0 + 1))],
                                   field[static_cast<size_t>(indexOf(x0 + 1, z0 + 1))], amountX);
    return static_cast<float>(std::lerp(north, south, amountZ));
}

// The native page is a four-block piecewise-bilinear authority. Derive its
// physical gradient from the same cell reconstruction instead of sampling an
// unrelated eight-block macro control field. This deliberately returns the
// one-sided derivative at a native-cell boundary, matching `reconstruct`'s
// half-open ownership rule and keeping point and grid queries identical.
std::pair<double, double> reconstructGradient(const std::vector<float>& field, double gridX,
                                              double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const double amountX = std::clamp(gridX - x0, 0.0, 1.0);
    const double amountZ = std::clamp(gridZ - z0, 0.0, 1.0);
    const double west = std::lerp(field[static_cast<size_t>(indexOf(x0, z0))],
                                  field[static_cast<size_t>(indexOf(x0, z0 + 1))], amountZ);
    const double east = std::lerp(field[static_cast<size_t>(indexOf(x0 + 1, z0))],
                                  field[static_cast<size_t>(indexOf(x0 + 1, z0 + 1))], amountZ);
    const double north = std::lerp(field[static_cast<size_t>(indexOf(x0, z0))],
                                   field[static_cast<size_t>(indexOf(x0 + 1, z0))], amountX);
    const double south = std::lerp(field[static_cast<size_t>(indexOf(x0, z0 + 1))],
                                   field[static_cast<size_t>(indexOf(x0 + 1, z0 + 1))], amountX);
    constexpr double INV_NATIVE_SPACING = 1.0 / NATIVE_HYDROLOGY_RASTER_SPACING;
    return {(east - west) * INV_NATIVE_SPACING, (south - north) * INV_NATIVE_SPACING};
}

double wetlandHydroperiod(double rawElevation, double hydraulicHead, double seasonality) {
    const double headDistance = std::max(0.0, rawElevation - hydraulicHead);
    const double shallowHead =
        1.0 - std::clamp(headDistance / static_cast<double>(WETLAND_MAX_GROUNDWATER_DEPTH_BLOCKS),
                         0.0, 1.0);
    const double stableRain =
        1.0 - std::clamp(seasonality / static_cast<double>(WETLAND_MAX_SEASONALITY), 0.0, 1.0);
    return std::clamp(0.55 + shallowHead * 0.30 + stableRain * 0.15, 0.55, 1.0);
}

struct ChannelProjection {
    double distance = std::numeric_limits<double>::infinity();
    double width = 0.0;
    double depth = 0.0;
    double stage = 0.0;
    // Stage owned by the selected persisted receiver edge before spatial
    // overlap blending. Strict route-grade validation uses this authority;
    // the published stage can additionally incorporate neighboring ribbons
    // and is validated after the graph minorant is solved.
    double routeStage = 0.0;
    double discharge = 0.0;
    double gradient = 0.0;
    double waterfallTop = 0.0;
    double waterfallBottom = 0.0;
    double waterfallAnchorX = 0.0;
    double waterfallAnchorZ = 0.0;
    double distanceAlong = 0.0;
    double segmentLength = 0.0;
    double flowX = 1.0;
    double flowZ = 0.0;
    WaterBodyId body = NO_WATER_BODY;
    // This is the canonical native raster source for the selected channel
    // segment.  It lets a routed waterfall carry an identity that remains
    // stable when point and native-lattice sampling choose the same segment.
    int source = -1;
    int target = -1;
    uint32_t ordinaryReach = 0;
    uint8_t streamOrder = 0;
    uint8_t distributaryCount = 0;
    bool waterfall = false;
    bool routedFallEdge = false;
    bool routedFallUpstream = false;
    bool waterfallAnchor = false;
    bool conflictingOrdinaryOverlap = false;
    bool delta = false;
    bool estuary = false;
};

struct ChannelStageCandidate {
    double stage = 0.0;
    double weight = 0.0;
    WaterBodyId body = NO_WATER_BODY;
    int source = -1;
    int target = -1;
};

constexpr size_t MAXIMUM_CHANNEL_STAGE_CANDIDATES =
    static_cast<size_t>((CHANNEL_SEARCH_RADIUS * 2 + 1) * (CHANNEL_SEARCH_RADIUS * 2 + 1) * 2);

bool ordinaryRibbonsShareStageAuthority(const ChannelStageCandidate& candidate,
                                        const ChannelProjection& selected) {
    const bool sharePersistedEndpoint =
        candidate.source == selected.source || candidate.source == selected.target ||
        candidate.target == selected.source || candidate.target == selected.target;
    // A reach ID is graph-wide. Two distant portions of a hairpin can share
    // one reach while their wide raster ribbons overlap in world space. Only
    // the selected edge and edges meeting it at a persisted endpoint own its
    // local stage blend.
    if (candidate.body == selected.body) return sharePersistedEndpoint;
    if (sharePersistedEndpoint) return true;
    return std::abs(candidate.stage - selected.stage) <= MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS;
}

const ReceiverOwnedOutletIndexEntry* receiverOwnedOutlet(const NativePage& page, int source) {
    const auto outlet = std::ranges::lower_bound(page.receiverOwnedOutlets, source, {},
                                                 &ReceiverOwnedOutletIndexEntry::source);
    return outlet != page.receiverOwnedOutlets.end() && outlet->source == source
               ? std::to_address(outlet)
               : nullptr;
}

std::pair<int64_t, int64_t> canonicalOutletFallAnchor(const NativePage& page, int sourceX,
                                                      int sourceZ,
                                                      const std::array<int32_t, 2>& targets,
                                                      const std::array<double, 2>& branchWeights,
                                                      size_t selectedBranch) {
    struct CandidateSet {
        std::array<std::pair<int, int>, 2> offsets{};
        size_t count = 0;
    };
    const auto candidatesFor = [&](size_t branch) {
        CandidateSet result;
        if (branch >= targets.size() || targets[branch] < 0 ||
            branchWeights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT) {
            return result;
        }
        const int targetX = targets[branch] % RASTER_EDGE;
        const int targetZ = targets[branch] / RASTER_EDGE;
        const int deltaX = targetX - sourceX;
        const int deltaZ = targetZ - sourceZ;
        if (std::abs(deltaX) > 1 || std::abs(deltaZ) > 1 || (deltaX == 0 && deltaZ == 0)) {
            result.offsets[0] = {deltaX * NATIVE_HYDROLOGY_RASTER_SPACING,
                                 deltaZ * NATIVE_HYDROLOGY_RASTER_SPACING};
            result.count = 1;
            return result;
        }
        if (deltaX != 0 && deltaZ != 0) {
            result.offsets[0] = {
                deltaX > 0 ? NATIVE_HYDROLOGY_RASTER_SPACING : 0,
                deltaZ > 0 ? NATIVE_HYDROLOGY_RASTER_SPACING : 0,
            };
            result.count = 1;
            return result;
        }
        if (deltaX != 0) {
            const int midpointX = deltaX > 0 ? NATIVE_HYDROLOGY_RASTER_SPACING : 0;
            result.offsets = {{{midpointX, 0}, {midpointX, NATIVE_HYDROLOGY_RASTER_SPACING}}};
        } else {
            const int midpointZ = deltaZ > 0 ? NATIVE_HYDROLOGY_RASTER_SPACING : 0;
            result.offsets = {{{0, midpointZ}, {NATIVE_HYDROLOGY_RASTER_SPACING, midpointZ}}};
        }
        result.count = 2;
        return result;
    };

    const std::array<CandidateSet, 2> candidates{candidatesFor(0), candidatesFor(1)};
    std::array<std::pair<int, int>, 2> selected{};
    bool found = false;
    std::tuple<size_t, int, int, int, int> selectedRank{};
    const size_t firstCount = std::max<size_t>(1, candidates[0].count);
    const size_t secondCount = std::max<size_t>(1, candidates[1].count);
    for (size_t first = 0; first < firstCount; ++first) {
        for (size_t second = 0; second < secondCount; ++second) {
            const std::pair<int, int> firstOffset =
                candidates[0].count != 0 ? candidates[0].offsets[first] : std::pair{0, 0};
            const std::pair<int, int> secondOffset =
                candidates[1].count != 0 ? candidates[1].offsets[second] : std::pair{0, 0};
            if (candidates[0].count != 0 && candidates[1].count != 0 &&
                firstOffset == secondOffset) {
                continue;
            }
            const auto rank = std::tuple{first + second, firstOffset.first, firstOffset.second,
                                         secondOffset.first, secondOffset.second};
            if (!found || rank < selectedRank) {
                found = true;
                selectedRank = rank;
                selected = {firstOffset, secondOffset};
            }
        }
    }
    if (!found || selectedBranch >= selected.size() || candidates[selectedBranch].count == 0)
        return {page.originX, page.originZ};

    const int64_t sourceWorldX = page.originX + static_cast<int64_t>(sourceX - RASTER_APRON) *
                                                    NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t sourceWorldZ = page.originZ + static_cast<int64_t>(sourceZ - RASTER_APRON) *
                                                    NATIVE_HYDROLOGY_RASTER_SPACING;
    return {sourceWorldX + selected[selectedBranch].first,
            sourceWorldZ + selected[selectedBranch].second};
}

std::optional<double> denseOrdinaryChannelStage(const NativePage& page, double gridX,
                                                double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    double minimumStage = std::numeric_limits<double>::infinity();
    double maximumStage = -std::numeric_limits<double>::infinity();
    for (const int z : {z0, z0 + 1}) {
        for (const int x : {x0, x0 + 1}) {
            const size_t cell = static_cast<size_t>(indexOf(x, z));
            if ((page.flags[cell] & CELL_RIVER) == 0 || (page.flags[cell] & CELL_WATERFALL) != 0 ||
                !std::isfinite(page.waterSurface[cell])) {
                return std::nullopt;
            }
            minimumStage = std::min(minimumStage, static_cast<double>(page.waterSurface[cell]));
            maximumStage = std::max(maximumStage, static_cast<double>(page.waterSurface[cell]));
        }
    }
    // A reach ID describes graph connectivity, not spatial adjacency. At a
    // hairpin, four neighboring raster cells can belong to distant portions
    // of that same reach and carry legitimately different routed stages.
    // Bilinear reconstruction across such a neighborhood can then override
    // one selected edge with an explicit-fall-sized one-column shelf. Leave
    // the edge projection and weighted ribbon candidates authoritative unless
    // the complete dense neighborhood is locally below the fall threshold.
    if (maximumStage - minimumStage >= MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS) {
        return std::nullopt;
    }
    return reconstruct(page.waterSurface, gridX, gridZ);
}

constexpr double CHANNEL_PROJECTION_TIE_EPSILON = 1.0e-9;

double distanceToBounds(double x, double z, double minimumX, double maximumX, double minimumZ,
                        double maximumZ) {
    const double deltaX = x < minimumX ? minimumX - x : x > maximumX ? x - maximumX : 0.0;
    const double deltaZ = z < minimumZ ? minimumZ - z : z > maximumZ ? z - maximumZ : 0.0;
    return std::hypot(deltaX, deltaZ);
}

bool channelCurveCanImprove(double worldX, double worldZ, double startX, double startZ, double endX,
                            double endZ, double run, double width, double bestNormalizedDistance) {
    if (!std::isfinite(bestNormalizedDistance)) return true;

    // A channel's six line segments are sampled from a quadratic Bezier.
    // Its control point is the midpoint moved perpendicular to the direct
    // receiver segment by no more than this amount. Every sampled segment is
    // therefore inside the direct-segment bounds expanded by maximumBend.
    // The distance to that expanded AABB is a lower bound for the candidate's
    // eventual distance. If it is already outside the incumbent's normalized
    // distance, this candidate cannot affect stage, body selection, or the
    // one-nanounit tie break below.
    const double maximumBend = std::min(1.6, run * 0.28);
    const double coordinateMagnitude =
        std::max({1.0, std::abs(worldX), std::abs(worldZ), std::abs(startX), std::abs(startZ),
                  std::abs(endX), std::abs(endZ)});
    // Keep the bound conservative even at the largest representable world
    // coordinates, where the later Bezier arithmetic can round a few ulps
    // beyond its mathematical convex hull.
    const double roundingSlack =
        std::numeric_limits<double>::epsilon() * coordinateMagnitude * 64.0;
    const double lowerDistance =
        distanceToBounds(worldX, worldZ, std::min(startX, endX) - maximumBend - roundingSlack,
                         std::max(startX, endX) + maximumBend + roundingSlack,
                         std::min(startZ, endZ) - maximumBend - roundingSlack,
                         std::max(startZ, endZ) + maximumBend + roundingSlack);
    const double lowerNormalizedDistance = lowerDistance / std::max(0.25, width * 0.5);
    return lowerNormalizedDistance <= bestNormalizedDistance + CHANNEL_PROJECTION_TIE_EPSILON;
}

double distanceToSegment(double x, double z, double startX, double startZ, double endX, double endZ,
                         double& along) {
    const double directionX = endX - startX;
    const double directionZ = endZ - startZ;
    const double lengthSquared = directionX * directionX + directionZ * directionZ;
    along =
        lengthSquared > 1.0e-12
            ? std::clamp(((x - startX) * directionX + (z - startZ) * directionZ) / lengthSquared,
                         0.0, 1.0)
            : 0.0;
    return std::hypot(x - std::lerp(startX, endX, along), z - std::lerp(startZ, endZ, along));
}

double channelWidth(const NativePage& page, size_t cell, double branchWeight = 1.0) {
    const double relativeDischarge =
        std::max(1.0, static_cast<double>(page.discharge[cell]) / MINIMUM_CHANNEL_DISCHARGE);
    const double base = std::clamp(
        2.5 + std::sqrt(relativeDischarge) * 0.32 + page.streamOrder[cell] * 0.6, 4.5, 36.0);
    return base * std::sqrt(std::clamp(branchWeight, 0.18, 1.0));
}

ChannelProjection projectChannelCore(const NativePage& page, double worldX, double worldZ,
                                     double gridX, double gridZ, bool determineAnchor = true) {
    ChannelProjection best;
    constexpr int CURVE_SEGMENTS = 6;
    const int centerX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int centerZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    double bestNormalizedDistance = std::numeric_limits<double>::infinity();
    std::array<ChannelStageCandidate, MAXIMUM_CHANNEL_STAGE_CANDIDATES> stageCandidates{};
    size_t stageCandidateCount = 0;
    for (int sourceZ = std::max(0, centerZ - CHANNEL_SEARCH_RADIUS);
         sourceZ <= std::min(RASTER_EDGE - 1, centerZ + CHANNEL_SEARCH_RADIUS); ++sourceZ) {
        const uint32_t rowBegin = page.channelRowOffsets[static_cast<size_t>(sourceZ)];
        const uint32_t rowEnd = page.channelRowOffsets[static_cast<size_t>(sourceZ + 1)];
        const auto first = std::lower_bound(
            page.channelSourceXs.begin() + rowBegin, page.channelSourceXs.begin() + rowEnd,
            static_cast<uint16_t>(std::max(0, centerX - CHANNEL_SEARCH_RADIUS)));
        for (auto candidate = first;
             candidate != page.channelSourceXs.begin() + rowEnd &&
             *candidate <= std::min(RASTER_EDGE - 1, centerX + CHANNEL_SEARCH_RADIUS);
             ++candidate) {
            const int sourceX = *candidate;
            const int source = indexOf(sourceX, sourceZ);
            const size_t sourceCell = static_cast<size_t>(source);
            const ReceiverOwnedOutletIndexEntry* persistedOutlet =
                receiverOwnedOutlet(page, source);
            const std::array<int32_t, 2> targets{
                page.receiverFirst[sourceCell],
                page.receiverSecond[sourceCell],
            };
            const std::array<double, 2> branchWeights{
                targets[1] >= 0 ? 1.0 - page.receiverSecondWeight[sourceCell] : 1.0,
                targets[1] >= 0 ? page.receiverSecondWeight[sourceCell] : 0.0,
            };
            uint8_t distributaryCount = 0;
            for (size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex) {
                if (targets[targetIndex] >= 0 &&
                    branchWeights[targetIndex] >= MINIMUM_CHANNEL_BRANCH_WEIGHT &&
                    (page.flags[static_cast<size_t>(targets[targetIndex])] & CELL_OCEAN) != 0) {
                    ++distributaryCount;
                }
            }
            for (size_t branch = 0; branch < targets.size(); ++branch) {
                const int target = targets[branch];
                if (target < 0 || branchWeights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT) continue;
                const size_t targetCell = static_cast<size_t>(target);
                const uint8_t targetFlags = page.flags[targetCell];
                const bool targetSupportsChannel =
                    (targetFlags & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) != 0 &&
                    page.waterBodyIds[targetCell] != NO_WATER_BODY &&
                    std::isfinite(page.waterSurface[targetCell]);
                if (!targetSupportsChannel) continue;
                const int targetX = target % RASTER_EDGE;
                const int targetZ = target / RASTER_EDGE;
                const double sourceStage = page.waterSurface[sourceCell];
                const double targetStage = page.waterSurface[targetCell];
                const bool branchHasFall = receiverBranchHasExplicitFall(page, sourceCell, branch);
                const bool outletBranchHasFall =
                    persistedOutlet != nullptr && persistedOutlet->standingStage - targetStage >=
                                                      MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS;
                const bool branchOwnsFall = branchHasFall || outletBranchHasFall;
                const double fallTopStage =
                    outletBranchHasFall
                        ? std::max(sourceStage, static_cast<double>(persistedOutlet->standingStage))
                        : sourceStage;
                const double width = channelWidth(page, sourceCell, branchWeights[branch]);
                const double startX = static_cast<double>(page.originX) +
                                      (sourceX - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                      NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
                const double startZ = static_cast<double>(page.originZ) +
                                      (sourceZ - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                      NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
                const double endX = static_cast<double>(page.originX) +
                                    (targetX - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                    NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
                const double endZ = static_cast<double>(page.originZ) +
                                    (targetZ - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                    NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
                const double run = std::hypot(endX - startX, endZ - startZ);
                if (run <= 1.0e-9) continue;
                // A nearest point on the bowed centerline is not a stable
                // hydraulic coordinate outside the curve's radius: adjacent
                // columns can lie on opposite sides of its medial axis and
                // snap from near the receiver to near the source. The control
                // point is displaced only perpendicular to the receiver edge,
                // so projection on that edge recovers the Bezier parameter and
                // remains monotone across the complete wide ribbon.
                const double receiverAxisAmount =
                    ((worldX - startX) * (endX - startX) + (worldZ - startZ) * (endZ - startZ)) /
                    (run * run);
                const double hydraulicAmount = native_hydrology_detail::receiverAxisProgress(
                    worldX, worldZ, startX, startZ, endX, endZ);
                const double maximumProjectionWidth = width + (branchOwnsFall ? 2.0 : 0.0);
                if (!channelCurveCanImprove(worldX, worldZ, startX, startZ, endX, endZ, run,
                                            maximumProjectionWidth,
                                            std::max(bestNormalizedDistance, 1.0))) {
                    continue;
                }
                const double perpendicularX = -(endZ - startZ) / run;
                const double perpendicularZ = (endX - startX) / run;
                const double bodyPhase =
                    static_cast<double>(mix64(page.waterBodyIds[sourceCell]) & 0xFFFFU) / 65'536.0 *
                    2.0 * std::numbers::pi;
                const double bend = std::sin(startX * 0.0131 + startZ * 0.0173 + bodyPhase) *
                                    std::min(1.6, run * 0.28);
                const double controlX = (startX + endX) * 0.5 + perpendicularX * bend;
                const double controlZ = (startZ + endZ) * 0.5 + perpendicularZ * bend;
                const auto curvePoint = [&](double amount) {
                    const double inverse = 1.0 - amount;
                    return std::pair{
                        inverse * inverse * startX + 2.0 * inverse * amount * controlX +
                            amount * amount * endX,
                        inverse * inverse * startZ + 2.0 * inverse * amount * controlZ +
                            amount * amount * endZ,
                    };
                };
                std::array<std::pair<double, double>, CURVE_SEGMENTS + 1> curvePoints{};
                std::array<double, CURVE_SEGMENTS> curveSegmentLengths{};
                for (int point = 0; point <= CURVE_SEGMENTS; ++point)
                    curvePoints[static_cast<size_t>(point)] =
                        curvePoint(static_cast<double>(point) / CURVE_SEGMENTS);
                double curveLength = 0.0;
                for (int segment = 0; segment < CURVE_SEGMENTS; ++segment) {
                    const auto start = curvePoints[static_cast<size_t>(segment)];
                    const auto end = curvePoints[static_cast<size_t>(segment + 1)];
                    const double length =
                        std::hypot(end.first - start.first, end.second - start.second);
                    curveSegmentLengths[static_cast<size_t>(segment)] = length;
                    curveLength += length;
                }
                double fallEndAlong = curveLength;
                double fallAnchorAmount = curveLength > 1.0e-9 ? 0.5 / curveLength : 0.0;
                if (branchHasFall && !outletBranchHasFall &&
                    (targetFlags & (CELL_OCEAN | CELL_LAKE)) != 0) {
                    const auto insideReceivingBody = [&](double amount) {
                        const auto point = curvePoint(amount);
                        const double pointGridX =
                            (point.first - static_cast<double>(page.originX)) /
                                NATIVE_HYDROLOGY_RASTER_SPACING +
                            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
                        const double pointGridZ =
                            (point.second - static_cast<double>(page.originZ)) /
                                NATIVE_HYDROLOGY_RASTER_SPACING +
                            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
                        const double terrain =
                            reconstruct(page.rawElevation, pointGridX, pointGridZ);
                        if ((targetFlags & CELL_OCEAN) != 0) return terrain < SEA_LEVEL;
                        return terrain >= SEA_LEVEL &&
                               reconstruct(page.lakeShoreDistance, pointGridX, pointGridZ) > 0.0;
                    };
                    if (insideReceivingBody(1.0)) {
                        double dryAmount = 0.0;
                        double wetAmount = 1.0;
                        for (int iteration = 0; iteration < 32; ++iteration) {
                            const double middle = (dryAmount + wetAmount) * 0.5;
                            if (insideReceivingBody(middle))
                                wetAmount = middle;
                            else
                                dryAmount = middle;
                        }
                        const double scaled = wetAmount * CURVE_SEGMENTS;
                        const int boundarySegment =
                            std::clamp(static_cast<int>(std::floor(scaled)), 0, CURVE_SEGMENTS - 1);
                        fallEndAlong = 0.0;
                        for (int segment = 0; segment < boundarySegment; ++segment)
                            fallEndAlong += curveSegmentLengths[static_cast<size_t>(segment)];
                        fallEndAlong += curveSegmentLengths[static_cast<size_t>(boundarySegment)] *
                                        (scaled - boundarySegment);
                        fallAnchorAmount =
                            std::clamp(wetAmount - 1.0 / std::max(1.0, curveLength), 0.0, 1.0);
                    }
                }
                double curveDistance = std::numeric_limits<double>::infinity();
                double curveAlong = 0.0;
                double curveFlowX = endX - startX;
                double curveFlowZ = endZ - startZ;
                double lengthBeforeSegment = 0.0;
                for (int segment = 0; segment < CURVE_SEGMENTS; ++segment) {
                    const auto previous = curvePoints[static_cast<size_t>(segment)];
                    const auto current = curvePoints[static_cast<size_t>(segment + 1)];
                    double segmentAlong = 0.0;
                    const double distance =
                        distanceToSegment(worldX, worldZ, previous.first, previous.second,
                                          current.first, current.second, segmentAlong);
                    if (distance < curveDistance) {
                        curveDistance = distance;
                        curveAlong = curveLength > 1.0e-9
                                         ? (lengthBeforeSegment +
                                            segmentAlong *
                                                curveSegmentLengths[static_cast<size_t>(segment)]) /
                                               curveLength
                                         : 0.0;
                        curveFlowX = current.first - previous.first;
                        curveFlowZ = current.second - previous.second;
                    }
                    lengthBeforeSegment += curveSegmentLengths[static_cast<size_t>(segment)];
                }
                // A legal fall is a compact, half-open curtain ending at the
                // reconstructed receiving contour. The upstream and
                // receiving reaches stay flat on their own stages instead of
                // interpolating a steep drop into horizontal water shelves.
                const double queryAlong = curveAlong * curveLength;
                // Each explicit fall branch owns its complete persisted receiver edge.
                // Restricting the curtain to an arbitrary one- or two-block
                // tail leaves ordinary overlap caps free to expose the full
                // top-to-bottom drop without transition ownership. The
                // centerline projection clamps to its first segment, though,
                // so a wide ribbon otherwise turns the source's round cap
                // into a broad upstream curtain. Clip that cap at the plane
                // normal to the routed receiver axis while retaining the
                // complete lateral width on and downstream of the plane.
                const double fallBeginAlong = branchOwnsFall ? 0.0 : fallEndAlong;
                const bool candidateWaterfall =
                    branchOwnsFall && receiverAxisAmount >= -CHANNEL_PROJECTION_TIE_EPSILON &&
                    queryAlong >= fallBeginAlong - CHANNEL_PROJECTION_TIE_EPSILON &&
                    queryAlong < fallEndAlong - CHANNEL_PROJECTION_TIE_EPSILON;
                const double projectionWidth = width + (candidateWaterfall ? 2.0 : 0.0);
                const double normalizedDistance =
                    curveDistance / std::max(0.25, projectionWidth * 0.5);
                auto waterfallAnchorPoint = curvePoint(fallAnchorAmount);
                double anchorNormalX = -(endZ - startZ) / run;
                double anchorNormalZ = (endX - startX) / run;
                if (std::pair{anchorNormalX, anchorNormalZ} > std::pair{0.0, 0.0}) {
                    anchorNormalX = -anchorNormalX;
                    anchorNormalZ = -anchorNormalZ;
                }
                const double anchorEdgeOffset = std::max(0.0, width * 0.5 - 0.5);
                waterfallAnchorPoint.first += anchorNormalX * anchorEdgeOffset;
                waterfallAnchorPoint.second += anchorNormalZ * anchorEdgeOffset;
                int64_t waterfallAnchorX =
                    static_cast<int64_t>(std::floor(waterfallAnchorPoint.first + 0.5));
                int64_t waterfallAnchorZ =
                    static_cast<int64_t>(std::floor(waterfallAnchorPoint.second + 0.5));
                if (outletBranchHasFall) {
                    // A persisted standing-water outlet owns one fall per
                    // active D-infinity receiver branch. A canonical midpoint
                    // corner lies on the global four-block lattice probed by
                    // every far-water tier. Two active branches select distinct
                    // corners, and the widened fall ribbon includes each corner
                    // even though learned native samples represent block centers.
                    const auto anchor = canonicalOutletFallAnchor(page, sourceX, sourceZ, targets,
                                                                  branchWeights, branch);
                    waterfallAnchorX = anchor.first;
                    waterfallAnchorZ = anchor.second;
                } else if (branchHasFall) {
                    bool foundAnchor = false;
                    double anchorRank = std::numeric_limits<double>::infinity();
                    const int64_t baseX =
                        static_cast<int64_t>(std::floor(waterfallAnchorPoint.first));
                    const int64_t baseZ =
                        static_cast<int64_t>(std::floor(waterfallAnchorPoint.second));
                    for (int offsetZ = -1; offsetZ <= 2; ++offsetZ) {
                        for (int offsetX = -1; offsetX <= 2; ++offsetX) {
                            const int64_t candidateX = baseX + offsetX;
                            const int64_t candidateZ = baseZ + offsetZ;
                            double candidateDistance = std::numeric_limits<double>::infinity();
                            double candidateAlong = 0.0;
                            double distanceBeforeSegment = 0.0;
                            for (int segment = 0; segment < CURVE_SEGMENTS; ++segment) {
                                double segmentAlong = 0.0;
                                const double distance = distanceToSegment(
                                    static_cast<double>(candidateX),
                                    static_cast<double>(candidateZ),
                                    curvePoints[static_cast<size_t>(segment)].first,
                                    curvePoints[static_cast<size_t>(segment)].second,
                                    curvePoints[static_cast<size_t>(segment + 1)].first,
                                    curvePoints[static_cast<size_t>(segment + 1)].second,
                                    segmentAlong);
                                if (distance < candidateDistance) {
                                    candidateDistance = distance;
                                    candidateAlong =
                                        distanceBeforeSegment +
                                        segmentAlong *
                                            curveSegmentLengths[static_cast<size_t>(segment)];
                                }
                                distanceBeforeSegment +=
                                    curveSegmentLengths[static_cast<size_t>(segment)];
                            }
                            const double candidateReceiverAxisAmount =
                                ((static_cast<double>(candidateX) - startX) * (endX - startX) +
                                 (static_cast<double>(candidateZ) - startZ) * (endZ - startZ)) /
                                (run * run);
                            if (candidateDistance > width * 0.5 ||
                                candidateReceiverAxisAmount < -CHANNEL_PROJECTION_TIE_EPSILON ||
                                candidateAlong < fallBeginAlong - CHANNEL_PROJECTION_TIE_EPSILON ||
                                candidateAlong >= fallEndAlong - CHANNEL_PROJECTION_TIE_EPSILON) {
                                continue;
                            }
                            const double rank = std::hypot(
                                static_cast<double>(candidateX) - waterfallAnchorPoint.first,
                                static_cast<double>(candidateZ) - waterfallAnchorPoint.second);
                            if (!foundAnchor ||
                                rank < anchorRank - CHANNEL_PROJECTION_TIE_EPSILON ||
                                (std::abs(rank - anchorRank) <= CHANNEL_PROJECTION_TIE_EPSILON &&
                                 std::pair{candidateX, candidateZ} <
                                     std::pair{waterfallAnchorX, waterfallAnchorZ})) {
                                foundAnchor = true;
                                anchorRank = rank;
                                waterfallAnchorX = candidateX;
                                waterfallAnchorZ = candidateZ;
                            }
                        }
                    }
                }
                const double candidateStage =
                    branchOwnsFall
                        ? (receiverAxisAmount < -CHANNEL_PROJECTION_TIE_EPSILON ? fallTopStage
                                                                                : targetStage)
                        : std::lerp(sourceStage, targetStage, hydraulicAmount);
                const bool candidateInside =
                    normalizedDistance <= 1.0 + CHANNEL_PROJECTION_TIE_EPSILON;
                const bool bestInside =
                    best.body != NO_WATER_BODY &&
                    bestNormalizedDistance <= 1.0 + CHANNEL_PROJECTION_TIE_EPSILON;
                const double flowLength = std::hypot(curveFlowX, curveFlowZ);
                if (candidateInside && !branchOwnsFall) {
                    const double interior = std::clamp(1.0 - normalizedDistance, 0.0, 1.0);
                    const double weight = interior * interior * (3.0 - 2.0 * interior);
                    if (stageCandidateCount >= stageCandidates.size())
                        throw std::runtime_error("native channel stage candidate overflow");
                    stageCandidates[stageCandidateCount++] = {
                        .stage = candidateStage,
                        .weight = std::max(weight, CHANNEL_PROJECTION_TIE_EPSILON),
                        .body = page.waterBodyIds[sourceCell],
                        .source = source,
                        .target = target,
                    };
                }
                bool replace = false;
                if (candidateInside != bestInside) {
                    replace = candidateInside;
                } else if (candidateInside && candidateWaterfall != best.waterfall) {
                    // A falling curtain owns its transition footprint even
                    // where the circular cap of an adjacent ordinary reach
                    // overlaps the same block column.
                    replace = candidateWaterfall;
                } else if (candidateInside) {
                    // Channel ribbons are a geometric union. At a split,
                    // junction, or tight bend, nearest-centerline selection
                    // can switch between two overlapping, individually
                    // monotone segments and expose a discontinuous water
                    // shelf. The upper continuous surface is the canonical
                    // surface throughout their overlap.
                    const double candidateRankStage =
                        candidateWaterfall ? fallTopStage : candidateStage;
                    const double bestRankStage = best.waterfall ? best.waterfallTop : best.stage;
                    replace = candidateRankStage > bestRankStage + CHANNEL_PROJECTION_TIE_EPSILON ||
                              (std::abs(candidateRankStage - bestRankStage) <=
                                   CHANNEL_PROJECTION_TIE_EPSILON &&
                               (normalizedDistance <
                                    bestNormalizedDistance - CHANNEL_PROJECTION_TIE_EPSILON ||
                                (std::abs(normalizedDistance - bestNormalizedDistance) <=
                                     CHANNEL_PROJECTION_TIE_EPSILON &&
                                 page.waterBodyIds[sourceCell] < best.body)));
                } else {
                    replace = normalizedDistance <
                                  bestNormalizedDistance - CHANNEL_PROJECTION_TIE_EPSILON ||
                              (std::abs(normalizedDistance - bestNormalizedDistance) <=
                                   CHANNEL_PROJECTION_TIE_EPSILON &&
                               page.waterBodyIds[sourceCell] < best.body);
                }
                if (!replace) continue;
                bestNormalizedDistance = normalizedDistance;
                best.distance = curveDistance;
                best.width = projectionWidth;
                const double relativeDischarge =
                    std::max(1.0, static_cast<double>(page.discharge[sourceCell]) /
                                      MINIMUM_CHANNEL_DISCHARGE);
                best.depth = std::clamp(0.9 + std::sqrt(relativeDischarge) * 0.08 +
                                            page.streamOrder[sourceCell] * 0.18,
                                        1.0, 12.0);
                best.stage = candidateStage;
                best.routeStage = candidateStage;
                best.discharge =
                    page.discharge[sourceCell] *
                    ((page.flags[sourceCell] & CELL_DELTA) != 0 ? branchWeights[branch] : 1.0);
                best.gradient =
                    branchOwnsFall ? std::max(static_cast<double>(page.channelGradient[sourceCell]),
                                              (fallTopStage - targetStage) / run)
                                   : page.channelGradient[sourceCell];
                best.waterfallTop = candidateWaterfall ? fallTopStage : 0.0;
                best.waterfallBottom = candidateWaterfall ? targetStage : 0.0;
                best.waterfallAnchorX = static_cast<double>(waterfallAnchorX);
                best.waterfallAnchorZ = static_cast<double>(waterfallAnchorZ);
                best.distanceAlong = hydraulicAmount * curveLength;
                best.segmentLength = curveLength;
                best.flowX = flowLength > 1.0e-9 ? curveFlowX / flowLength : 1.0;
                best.flowZ = flowLength > 1.0e-9 ? curveFlowZ / flowLength : 0.0;
                best.body = page.waterBodyIds[sourceCell];
                best.source = source;
                best.target = target;
                best.ordinaryReach = page.ordinaryReachIds[sourceCell];
                best.streamOrder = page.streamOrder[sourceCell];
                best.distributaryCount =
                    (page.flags[sourceCell] & CELL_DELTA) != 0 ? distributaryCount : 0;
                best.waterfall = candidateWaterfall;
                best.routedFallEdge = branchOwnsFall;
                best.routedFallUpstream =
                    branchOwnsFall && receiverAxisAmount < -CHANNEL_PROJECTION_TIE_EPSILON;
                best.waterfallAnchor = false;
                best.delta = (page.flags[sourceCell] & CELL_DELTA) != 0;
                best.estuary = (page.flags[sourceCell] & CELL_ESTUARY) != 0;
            }
        }
    }

    if (best.body != NO_WATER_BODY && !best.waterfall &&
        bestNormalizedDistance <= 1.0 + CHANNEL_PROJECTION_TIE_EPSILON) {
        double weightedStage = 0.0;
        double totalWeight = 0.0;
        for (size_t index = 0; index < stageCandidateCount; ++index) {
            const ChannelStageCandidate& candidate = stageCandidates[index];
            if (!ordinaryRibbonsShareStageAuthority(candidate, best)) continue;
            weightedStage += candidate.stage * candidate.weight;
            totalWeight += candidate.weight;
        }
        if (totalWeight > 0.0) best.stage = weightedStage / totalWeight;
        if (const std::optional<double> denseStage =
                denseOrdinaryChannelStage(page, gridX, gridZ)) {
            best.stage = *denseStage;
        }
        for (size_t index = 0; index < stageCandidateCount; ++index) {
            const ChannelStageCandidate& candidate = stageCandidates[index];
            if (candidate.body == best.body &&
                std::abs(candidate.stage - best.stage) >
                    MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS * 0.75) {
                best.conflictingOrdinaryOverlap = true;
            }
        }
        if (best.target >= 0) {
            const int targetX = best.target % RASTER_EDGE;
            const int targetZ = best.target / RASTER_EDGE;
            const double targetWorldX = static_cast<double>(page.originX) +
                                        (targetX - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                        NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
            const double targetWorldZ = static_cast<double>(page.originZ) +
                                        (targetZ - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING +
                                        NATIVE_SAMPLE_WORLD_CENTER_OFFSET;
            const double targetConeStage =
                page.waterSurface[static_cast<size_t>(best.target)] +
                std::hypot(worldX - targetWorldX, worldZ - targetWorldZ) *
                    RIVER_MAXIMUM_ORDINARY_STAGE_DROP_PER_BLOCK;
            const double alongAmount =
                std::clamp(best.distanceAlong / std::max(1.0, best.segmentLength), 0.0, 1.0);
            const double smoothAlong = alongAmount * alongAmount * (3.0 - 2.0 * alongAmount);
            best.stage = std::lerp(best.stage, std::min(best.stage, targetConeStage), smoothAlong);
            best.routeStage =
                std::lerp(best.routeStage, std::min(best.routeStage, targetConeStage), smoothAlong);
        }
    }

    if (determineAnchor && best.waterfall && worldX == std::floor(worldX) &&
        worldZ == std::floor(worldZ)) {
        const double rank =
            std::hypot(worldX - best.waterfallAnchorX, worldZ - best.waterfallAnchorZ);
        best.waterfallAnchor = true;
        for (int offsetZ = -1; offsetZ <= 1 && best.waterfallAnchor; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                if (offsetX == 0 && offsetZ == 0) continue;
                const ChannelProjection neighbor = projectChannelCore(
                    page, worldX + offsetX, worldZ + offsetZ,
                    gridX + static_cast<double>(offsetX) / NATIVE_HYDROLOGY_RASTER_SPACING,
                    gridZ + static_cast<double>(offsetZ) / NATIVE_HYDROLOGY_RASTER_SPACING, false);
                if (!neighbor.waterfall || neighbor.source != best.source ||
                    neighbor.target != best.target) {
                    continue;
                }
                const double neighborRank = std::hypot(worldX + offsetX - best.waterfallAnchorX,
                                                       worldZ + offsetZ - best.waterfallAnchorZ);
                if (neighborRank < rank - CHANNEL_PROJECTION_TIE_EPSILON ||
                    (std::abs(neighborRank - rank) <= CHANNEL_PROJECTION_TIE_EPSILON &&
                     std::pair{worldX + offsetX, worldZ + offsetZ} < std::pair{worldX, worldZ})) {
                    best.waterfallAnchor = false;
                    break;
                }
            }
        }
    }

    return best;
}

ChannelProjection projectChannel(const NativePage& page, double worldX, double worldZ, double gridX,
                                 double gridZ) {
    return projectChannelCore(page, worldX, worldZ, gridX, gridZ);
}

uint64_t nativeFallTransitionId(const NativePage& page, const ChannelProjection& channel) {
    if (channel.source < 0 || channel.target < 0 || channel.body == NO_WATER_BODY) return 0;
    const int sourceX = channel.source % RASTER_EDGE;
    const int sourceZ = channel.source / RASTER_EDGE;
    const int targetX = channel.target % RASTER_EDGE;
    const int targetZ = channel.target / RASTER_EDGE;
    const int64_t worldX =
        static_cast<int64_t>(page.originX) +
        static_cast<int64_t>(sourceX - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t worldZ =
        static_cast<int64_t>(page.originZ) +
        static_cast<int64_t>(sourceZ - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t targetWorldX =
        static_cast<int64_t>(page.originX) +
        static_cast<int64_t>(targetX - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING;
    const int64_t targetWorldZ =
        static_cast<int64_t>(page.originZ) +
        static_cast<int64_t>(targetZ - RASTER_APRON) * NATIVE_HYDROLOGY_RASTER_SPACING;
    uint64_t identity = mix64(0x4E41'5449'5645'464CULL ^ channel.body);
    identity = mix64(identity ^ static_cast<uint64_t>(worldX));
    identity = mix64(identity ^ static_cast<uint64_t>(worldZ));
    identity = mix64(identity ^ static_cast<uint64_t>(targetWorldX));
    identity = mix64(identity ^ static_cast<uint64_t>(targetWorldZ));
    return identity == 0 ? 1 : identity;
}

std::optional<size_t> supportingLakeCell(const NativePage& page, double gridX, double gridZ) {
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    std::optional<size_t> selected;
    double selectedDistance = std::numeric_limits<double>::infinity();
    for (int z = z0; z <= z0 + 1; ++z) {
        for (int x = x0; x <= x0 + 1; ++x) {
            const size_t cell = static_cast<size_t>(indexOf(x, z));
            if ((page.flags[cell] & CELL_LAKE) == 0) continue;
            const double distance = std::hypot(gridX - x, gridZ - z);
            if (!selected || distance < selectedDistance - CHANNEL_PROJECTION_TIE_EPSILON ||
                (std::abs(distance - selectedDistance) <= CHANNEL_PROJECTION_TIE_EPSILON &&
                 std::tuple{page.waterBodyIds[cell], cell} <
                     std::tuple{page.waterBodyIds[*selected], *selected})) {
                selected = cell;
                selectedDistance = distance;
            }
        }
    }
    return selected;
}

struct ReceiverOwnedStandingConnection {
    double stage = 0.0;
    double distance = 0.0;
    WaterBodyId body = NO_WATER_BODY;
    bool outlet = false;
};

std::optional<ReceiverOwnedStandingConnection>
receiverOwnedStandingConnection(const NativePage& page, const ChannelProjection& channel) {
    if (channel.source < 0 || channel.target < 0 || channel.routedFallEdge) return std::nullopt;
    const size_t targetCell = static_cast<size_t>(channel.target);
    const uint8_t targetFlags = page.flags[targetCell];
    if ((targetFlags & (CELL_OCEAN | CELL_LAKE | CELL_WETLAND)) != 0 &&
        page.waterBodyIds[targetCell] != NO_WATER_BODY &&
        std::isfinite(page.waterSurface[targetCell])) {
        return ReceiverOwnedStandingConnection{
            .stage = page.waterSurface[targetCell],
            .distance = std::max(0.0, channel.segmentLength - channel.distanceAlong),
            .body = page.waterBodyIds[targetCell],
            .outlet = false,
        };
    }

    if (const ReceiverOwnedOutletIndexEntry* outlet = receiverOwnedOutlet(page, channel.source)) {
        return ReceiverOwnedStandingConnection{
            .stage = outlet->standingStage,
            .distance = channel.distanceAlong,
            .body = outlet->standingBody,
            .outlet = true,
        };
    }
    return std::nullopt;
}

void applyReceiverOwnedStandingBackwater(const NativePage& page, bool lake,
                                         ChannelProjection& channel) {
    constexpr double STANDING_BACKWATER_REACH_BLOCKS =
        MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS / RIVER_MAXIMUM_ORDINARY_STAGE_DROP_PER_BLOCK;
    const bool channelInside =
        channel.body != NO_WATER_BODY && channel.distance <= channel.width * 0.5;
    if (lake || !channelInside) return;
    const std::optional<ReceiverOwnedStandingConnection> connection =
        receiverOwnedStandingConnection(page, channel);
    const double effectiveReach = std::min(STANDING_BACKWATER_REACH_BLOCKS, channel.segmentLength);
    if (!connection || effectiveReach <= 1.0e-9 || connection->distance > effectiveReach) return;

    // A proven receiver edge approaches its standing stage with zero slope at
    // both ends of the bounded blend. Outlet reaches descend away from the
    // lake; inflows rise away from it. Unrelated spatially nearby bodies never
    // enter this calculation.
    const double distanceAmount = std::clamp(connection->distance / effectiveReach, 0.0, 1.0);
    const double smoothDistance = distanceAmount * distanceAmount * (3.0 - 2.0 * distanceAmount);
    const double signedOrdinaryDrop =
        connection->outlet ? -MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS * smoothDistance
                           : MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS * smoothDistance;
    const double targetStage = connection->stage + signedOrdinaryDrop;
    channel.stage = std::lerp(channel.stage, targetStage, 1.0 - smoothDistance);
    channel.routeStage = std::lerp(channel.routeStage, targetStage, 1.0 - smoothDistance);
}

double channelBedDepthAt(const ChannelProjection& channel) {
    if (channel.body == NO_WATER_BODY || channel.width <= 0.0 || !std::isfinite(channel.distance)) {
        return 0.0;
    }
    const double interior =
        std::clamp(1.0 - channel.distance / std::max(0.25, channel.width * 0.5), 0.0, 1.0);
    const double smoothInterior = interior * interior * (3.0 - 2.0 * interior);
    return std::lerp(static_cast<double>(RIVER_MINIMUM_BED_DEPTH_BLOCKS), channel.depth,
                     smoothInterior);
}

bool channelRibbonSupported(const ChannelProjection& channel, double undisturbed) {
    if (channel.body == NO_WATER_BODY || channel.distance > channel.width * 0.5) return false;
    if (channel.waterfall) return true;
    constexpr double CONFLICTING_REACH_CENTERLINE_SUPPORT_BLOCKS = 1.0;
    if (channel.conflictingOrdinaryOverlap &&
        channel.distance >
            std::min(channel.width * 0.5, CONFLICTING_REACH_CENTERLINE_SUPPORT_BLOCKS)) {
        return false;
    }
    constexpr double ROUTED_CENTERLINE_SUPPORT_BLOCKS = NATIVE_HYDROLOGY_RASTER_SPACING * 0.75;
    if (channel.distance <= std::min(channel.width * 0.5, ROUTED_CENTERLINE_SUPPORT_BLOCKS)) {
        return true;
    }
    return undisturbed <= channel.stage + channelBedDepthAt(channel) + 1.0e-6;
}

constexpr int ORDINARY_STAGE_TILE_EDGE = 32;
constexpr int ORDINARY_STAGE_PRIMARY_HALO = 4;
constexpr int ORDINARY_STAGE_CERTIFICATE_HALO = 8;
constexpr int ORDINARY_STAGE_MAXIMUM_HALO = 16;
constexpr int32_t ORDINARY_STAGE_MAXIMUM_RAW_ROUTE_DROP_EIGHTHS =
    static_cast<int32_t>(MINIMUM_EXPLICIT_CHANNEL_FALL_DROP_BLOCKS * 8.0) - 1;
constexpr int32_t ORDINARY_STAGE_MAXIMUM_VISIBLE_DROP_EIGHTHS = 1;
static_assert(ORDINARY_STAGE_MAXIMUM_RAW_ROUTE_DROP_EIGHTHS == 3);

constexpr bool unresolvedRawOrdinaryStageContact(int32_t routeDrop, int32_t publishedDrop) {
    return routeDrop > ORDINARY_STAGE_MAXIMUM_RAW_ROUTE_DROP_EIGHTHS &&
           publishedDrop > ORDINARY_STAGE_MAXIMUM_VISIBLE_DROP_EIGHTHS;
}

static_assert(unresolvedRawOrdinaryStageContact(4, 2));
static_assert(!unresolvedRawOrdinaryStageContact(4, 1));
static_assert(!unresolvedRawOrdinaryStageContact(3, 2));

struct OrdinaryStageTileEntry {
    uint16_t index = 0;
    int32_t stageEighths = 0;
    WaterBodyId body = NO_WATER_BODY;
    uint32_t ordinaryReach = 0;
};

struct OrdinaryStageTile {
    int64_t originX = 0;
    int64_t originZ = 0;
    uint8_t certifiedHalo = 0;
    std::vector<OrdinaryStageTileEntry> entries;
};

struct RawOrdinaryStagePixel {
    int32_t stageEighths = 0;
    int32_t routeStageEighths = 0;
    WaterBodyId body = NO_WATER_BODY;
    PageKey owner;
    uint32_t ordinaryReach = 0;
    int64_t topologyStartX = 0;
    int64_t topologyStartZ = 0;
    int64_t topologyEndX = 0;
    int64_t topologyEndZ = 0;
    double topologyDistanceAlong = 0.0;
    double topologyLength = 0.0;
    bool hasTopologyEdge = false;
    bool wet = false;
};

using NativeStagePageResolver = std::function<std::shared_ptr<const NativePage>(PageKey)>;

uint32_t effectiveOrdinaryReach(const NativePage& page, const ChannelProjection& channel) {
    uint32_t reach = channel.ordinaryReach;
    if (channel.routedFallEdge && !channel.routedFallUpstream && channel.target >= 0) {
        const uint32_t receivingReach = page.ordinaryReachIds[static_cast<size_t>(channel.target)];
        if (receivingReach != 0) reach = receivingReach;
    }
    return reach;
}

RawOrdinaryStagePixel rawOrdinaryStagePixel(const NativePage& page, int64_t worldX,
                                            int64_t worldZ) {
    const double gridX = (static_cast<double>(worldX) - static_cast<double>(page.originX)) /
                             NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const double gridZ = (static_cast<double>(worldZ) - static_cast<double>(page.originZ)) /
                             NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const double undisturbed = reconstruct(page.rawElevation, gridX, gridZ);
    const bool ocean = undisturbed < SEA_LEVEL;
    const bool lake = !ocean && reconstruct(page.lakeShoreDistance, gridX, gridZ) > 0.0;
    if (ocean || lake) return {};

    const int nearestX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int nearestZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    const size_t nearestCell = static_cast<size_t>(indexOf(nearestX, nearestZ));
    if (page.channelProximity[nearestCell] == 0) return {};

    ChannelProjection channel = projectChannelCore(
        page, static_cast<double>(worldX), static_cast<double>(worldZ), gridX, gridZ, false);
    applyReceiverOwnedStandingBackwater(page, false, channel);
    if (channel.waterfall || !channelRibbonSupported(channel, undisturbed) ||
        channel.body == NO_WATER_BODY || !std::isfinite(channel.stage)) {
        return {};
    }
    const uint32_t reach = effectiveOrdinaryReach(page, channel);
    if (reach == 0) return {};
    if (channel.source < 0 || channel.target < 0) return {};
    const int sourceX = channel.source % RASTER_EDGE;
    const int sourceZ = channel.source / RASTER_EDGE;
    const int targetX = channel.target % RASTER_EDGE;
    const int targetZ = channel.target / RASTER_EDGE;
    int64_t topologyStartX = checkedCoordinate(page.key.x, sourceX - RASTER_APRON);
    int64_t topologyStartZ = checkedCoordinate(page.key.z, sourceZ - RASTER_APRON);
    int64_t topologyEndX = checkedCoordinate(page.key.x, targetX - RASTER_APRON);
    int64_t topologyEndZ = checkedCoordinate(page.key.z, targetZ - RASTER_APRON);
    if (channel.routedFallEdge) {
        // The source-plane gate leaves ordinary round-cap samples on both
        // sides of the explicit curtain. Attach each only to its own endpoint
        // so ordinary reconciliation cannot reconnect the two reaches around
        // the persisted fall or replace the upstream top stage with the
        // receiving stage.
        if (channel.routedFallUpstream) {
            topologyEndX = topologyStartX;
            topologyEndZ = topologyStartZ;
        } else {
            topologyStartX = topologyEndX;
            topologyStartZ = topologyEndZ;
        }
    }
    return {
        .stageEighths = static_cast<int32_t>(std::llround(channel.stage * 8.0)),
        .routeStageEighths = static_cast<int32_t>(std::llround(channel.routeStage * 8.0)),
        .body = channel.body,
        .owner = page.key,
        .ordinaryReach = reach,
        .topologyStartX = topologyStartX,
        .topologyStartZ = topologyStartZ,
        .topologyEndX = topologyEndX,
        .topologyEndZ = topologyEndZ,
        .topologyDistanceAlong = channel.routedFallEdge ? 0.0 : channel.distanceAlong,
        .topologyLength = channel.routedFallEdge ? 0.0 : channel.segmentLength,
        .hasTopologyEdge = true,
        .wet = true,
    };
}

bool ordinaryStagePixelsSharePersistedJunction(const RawOrdinaryStagePixel& first,
                                               const RawOrdinaryStagePixel& second) {
    if (!first.wet || !second.wet || first.body != second.body) return false;
    if (!first.hasTopologyEdge || !second.hasTopologyEdge) return false;
    const std::array<std::pair<int64_t, int64_t>, 2> firstEndpoints{{
        {first.topologyStartX, first.topologyStartZ},
        {first.topologyEndX, first.topologyEndZ},
    }};
    const std::array<std::pair<int64_t, int64_t>, 2> secondEndpoints{{
        {second.topologyStartX, second.topologyStartZ},
        {second.topologyEndX, second.topologyEndZ},
    }};
    return std::ranges::any_of(firstEndpoints, [&](const auto& firstEndpoint) {
        return std::ranges::find(secondEndpoints, firstEndpoint) != secondEndpoints.end();
    });
}

std::optional<double> ordinaryStageProjectedRouteDistance(const RawOrdinaryStagePixel& first,
                                                          const RawOrdinaryStagePixel& second) {
    if (!ordinaryStagePixelsSharePersistedJunction(first, second)) return std::nullopt;
    const std::pair firstStart{first.topologyStartX, first.topologyStartZ};
    const std::pair firstEnd{first.topologyEndX, first.topologyEndZ};
    const std::pair secondStart{second.topologyStartX, second.topologyStartZ};
    const std::pair secondEnd{second.topologyEndX, second.topologyEndZ};
    const double firstAlong = std::clamp(first.topologyDistanceAlong, 0.0, first.topologyLength);
    const double secondAlong = std::clamp(second.topologyDistanceAlong, 0.0, second.topologyLength);
    if (firstStart == secondStart && firstEnd == secondEnd)
        return std::abs(firstAlong - secondAlong);
    if (firstStart == secondEnd && firstEnd == secondStart)
        return std::abs(firstAlong - (second.topologyLength - secondAlong));

    const auto distanceToEndpoint = [](const RawOrdinaryStagePixel& pixel,
                                       const std::pair<int64_t, int64_t>& endpoint) {
        if (pixel.topologyStartX == pixel.topologyEndX &&
            pixel.topologyStartZ == pixel.topologyEndZ) {
            return 0.0;
        }
        const double along = std::clamp(pixel.topologyDistanceAlong, 0.0, pixel.topologyLength);
        if (endpoint == std::pair{pixel.topologyStartX, pixel.topologyStartZ}) return along;
        return std::max(0.0, pixel.topologyLength - along);
    };
    double distance = std::numeric_limits<double>::infinity();
    for (const auto& firstEndpoint : {firstStart, firstEnd}) {
        for (const auto& secondEndpoint : {secondStart, secondEnd}) {
            if (firstEndpoint != secondEndpoint) continue;
            distance = std::min(distance, distanceToEndpoint(first, firstEndpoint) +
                                              distanceToEndpoint(second, secondEndpoint));
        }
    }
    return std::isfinite(distance) ? std::optional{distance} : std::nullopt;
}

bool ordinaryStagePixelsConnected(const RawOrdinaryStagePixel& first,
                                  const RawOrdinaryStagePixel& second) {
    if (!first.wet || !second.wet || first.body != second.body) return false;
    // Two broad portions of one ordinary reach can touch spatially at a
    // hairpin without their selected raster edges being consecutive. They
    // still need one visually continuous ribbon, so the graph minorant joins
    // them. Raw route-grade validation below is intentionally narrower and
    // applies only to an actual persisted edge/junction contact.
    if (first.owner == second.owner && first.ordinaryReach != 0 &&
        first.ordinaryReach == second.ordinaryReach) {
        return true;
    }
    return ordinaryStagePixelsSharePersistedJunction(first, second);
}

std::vector<int32_t> solveOrdinaryStageInfimum(std::span<const RawOrdinaryStagePixel> pixels,
                                               int edge) {
    constexpr int32_t INFINITY_STAGE = std::numeric_limits<int32_t>::max();
    std::vector<int32_t> stages(pixels.size(), INFINITY_STAGE);
    using QueueEntry = std::pair<int32_t, int>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> queue;
    for (int z = 0; z < edge; ++z) {
        for (int x = 0; x < edge; ++x) {
            const int index = z * edge + x;
            const RawOrdinaryStagePixel& pixel = pixels[static_cast<size_t>(index)];
            if (!pixel.wet) continue;
            stages[static_cast<size_t>(index)] = pixel.stageEighths;
            queue.emplace(pixel.stageEighths, index);
        }
    }

    constexpr std::array<std::pair<int, int>, 4> CARDINAL_OFFSETS{
        std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1}, std::pair{0, 1}};
    while (!queue.empty()) {
        const auto [stage, index] = queue.top();
        queue.pop();
        if (stages[static_cast<size_t>(index)] != stage) continue;
        const int x = index % edge;
        const int z = index / edge;
        const RawOrdinaryStagePixel& pixel = pixels[static_cast<size_t>(index)];
        for (const auto [offsetX, offsetZ] : CARDINAL_OFFSETS) {
            const int neighborX = x + offsetX;
            const int neighborZ = z + offsetZ;
            if (neighborX < 0 || neighborX >= edge || neighborZ < 0 || neighborZ >= edge) {
                continue;
            }
            const int neighborIndex = neighborZ * edge + neighborX;
            const RawOrdinaryStagePixel& neighbor = pixels[static_cast<size_t>(neighborIndex)];
            if (!ordinaryStagePixelsConnected(pixel, neighbor)) {
                continue;
            }
            const int32_t candidate = stage + 1;
            if (candidate >= stages[static_cast<size_t>(neighborIndex)]) continue;
            stages[static_cast<size_t>(neighborIndex)] = candidate;
            queue.emplace(candidate, neighborIndex);
        }
    }
    return stages;
}

struct OrdinaryStageDomain {
    int64_t originX = 0;
    int64_t originZ = 0;
    int halo = 0;
    int edge = 0;
    std::vector<RawOrdinaryStagePixel> pixels;
    std::vector<int32_t> stages;
};

void validateRawOrdinaryStageContacts(const OrdinaryStageDomain& domain) {
    // A cardinal world-space contact can project at most one block along a
    // single receiver edge. Permit a second block for a true two-edge
    // junction contact and numerical curve segmentation. Wider round caps can
    // make consecutive edges touch many routed blocks away from their shared
    // endpoint; those are spatial-overlap contacts, not raw route contacts,
    // and the graph minorant below owns their visible continuity.
    constexpr double MAXIMUM_LOCAL_PROJECTED_ROUTE_DISTANCE_BLOCKS = 2.0;
    for (int z = 0; z < domain.edge; ++z) {
        for (int x = 0; x < domain.edge; ++x) {
            const RawOrdinaryStagePixel& pixel =
                domain.pixels[static_cast<size_t>(z * domain.edge + x)];
            if (!pixel.wet) continue;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= domain.edge || z + offsetZ >= domain.edge) continue;
                const RawOrdinaryStagePixel& adjacent =
                    domain.pixels[static_cast<size_t>((z + offsetZ) * domain.edge + x + offsetX)];
                if (!ordinaryStagePixelsSharePersistedJunction(pixel, adjacent)) continue;
                const std::optional<double> routeDistance =
                    ordinaryStageProjectedRouteDistance(pixel, adjacent);
                if (!routeDistance)
                    throw std::logic_error("persisted ordinary junction has no route distance");
                if (*routeDistance > MAXIMUM_LOCAL_PROJECTED_ROUTE_DISTANCE_BLOCKS +
                                         CHANNEL_PROJECTION_TIE_EPSILON) {
                    continue;
                }
                const int32_t routeDrop =
                    std::abs(adjacent.routeStageEighths - pixel.routeStageEighths);
                if (routeDrop <= ORDINARY_STAGE_MAXIMUM_RAW_ROUTE_DROP_EIGHTHS) continue;
                // A selected edge at a collapsed explicit-fall cap can retain
                // its top-stage provenance while the overlapping ordinary
                // ribbon has already published one continuous surface. That
                // is not an unresolved raw discontinuity: the stricter
                // one-eighth visible contract is already satisfied, and the
                // solved-domain validator below proves it remains satisfied.
                // Keep rejecting a bad local route grade whenever the
                // published ordinary contact has not already reconciled it.
                const int32_t publishedDrop = std::abs(adjacent.stageEighths - pixel.stageEighths);
                if (!unresolvedRawOrdinaryStageContact(routeDrop, publishedDrop)) continue;
                const int64_t worldX = domain.originX + x;
                const int64_t worldZ = domain.originZ + z;
                throw std::runtime_error(
                    "raw ordinary stage contact exceeds the explicit-fall bound at (" +
                    std::to_string(worldX) + ", " + std::to_string(worldZ) + ") -> (" +
                    std::to_string(worldX + offsetX) + ", " + std::to_string(worldZ + offsetZ) +
                    "), stages " + std::to_string(pixel.routeStageEighths) + " -> " +
                    std::to_string(adjacent.routeStageEighths) + ", published " +
                    std::to_string(pixel.stageEighths) + " -> " +
                    std::to_string(adjacent.stageEighths) + ", route distance " +
                    std::to_string(*routeDistance) + ", body " + std::to_string(pixel.body) +
                    ", reaches " + std::to_string(pixel.ordinaryReach) + " -> " +
                    std::to_string(adjacent.ordinaryReach) + ", edges (" +
                    std::to_string(pixel.topologyStartX) + ", " +
                    std::to_string(pixel.topologyStartZ) + ") -> (" +
                    std::to_string(pixel.topologyEndX) + ", " + std::to_string(pixel.topologyEndZ) +
                    ") and (" + std::to_string(adjacent.topologyStartX) + ", " +
                    std::to_string(adjacent.topologyStartZ) + ") -> (" +
                    std::to_string(adjacent.topologyEndX) + ", " +
                    std::to_string(adjacent.topologyEndZ) + ")");
            }
        }
    }
}

void validateSolvedOrdinaryStageContacts(const OrdinaryStageDomain& domain) {
    constexpr int32_t INFINITY_STAGE = std::numeric_limits<int32_t>::max();
    if (domain.stages.size() != domain.pixels.size())
        throw std::logic_error("ordinary stage solution has the wrong extent");
    for (size_t index = 0; index < domain.pixels.size(); ++index) {
        const RawOrdinaryStagePixel& pixel = domain.pixels[index];
        const int32_t stage = domain.stages[index];
        if (!pixel.wet) {
            if (stage != INFINITY_STAGE)
                throw std::runtime_error("ordinary stage solution raised a dry pixel");
            continue;
        }
        if (stage == INFINITY_STAGE)
            throw std::runtime_error("ordinary stage solution deleted a wet pixel");
        if (stage > pixel.stageEighths)
            throw std::runtime_error("ordinary stage solution raised raw water authority");
    }
    for (int z = 0; z < domain.edge; ++z) {
        for (int x = 0; x < domain.edge; ++x) {
            const int index = z * domain.edge + x;
            const RawOrdinaryStagePixel& pixel = domain.pixels[static_cast<size_t>(index)];
            if (!pixel.wet) continue;
            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= domain.edge || z + offsetZ >= domain.edge) continue;
                const int adjacentIndex = (z + offsetZ) * domain.edge + x + offsetX;
                const RawOrdinaryStagePixel& adjacent =
                    domain.pixels[static_cast<size_t>(adjacentIndex)];
                if (!ordinaryStagePixelsConnected(pixel, adjacent)) continue;
                if (std::abs(domain.stages[static_cast<size_t>(index)] -
                             domain.stages[static_cast<size_t>(adjacentIndex)]) >
                    ORDINARY_STAGE_MAXIMUM_VISIBLE_DROP_EIGHTHS) {
                    throw std::runtime_error(
                        "solved ordinary stage contact exceeds one visible eighth");
                }
            }
        }
    }
}

OrdinaryStageDomain sampleOrdinaryStageDomain(const NativePage& page, int64_t tileOriginX,
                                              int64_t tileOriginZ, int halo,
                                              const NativeStagePageResolver* resolvePage) {
    OrdinaryStageDomain domain;
    domain.originX = tileOriginX - halo;
    domain.originZ = tileOriginZ - halo;
    domain.halo = halo;
    domain.edge = ORDINARY_STAGE_TILE_EDGE + halo * 2;
    domain.pixels.resize(static_cast<size_t>(domain.edge * domain.edge));
    std::map<PageKey, std::shared_ptr<const NativePage>> resolvedPages;
    const auto pageAt = [&](int64_t worldX, int64_t worldZ) -> const NativePage& {
        const PageKey owner = pageKeyAt(static_cast<double>(worldX), static_cast<double>(worldZ));
        if (owner == page.key) return page;
        if (resolvePage == nullptr)
            throw std::runtime_error("ordinary stage halo requires its half-open owner page");
        if (const auto found = resolvedPages.find(owner); found != resolvedPages.end())
            return *found->second;
        std::shared_ptr<const NativePage> resolved = (*resolvePage)(owner);
        if (!resolved || resolved->key != owner)
            throw std::runtime_error("ordinary stage halo resolved the wrong owner page");
        return *resolvedPages.emplace(owner, std::move(resolved)).first->second;
    };
    for (int z = 0; z < domain.edge; ++z) {
        for (int x = 0; x < domain.edge; ++x) {
            const int64_t worldX = tileOriginX + x - halo;
            const int64_t worldZ = tileOriginZ + z - halo;
            domain.pixels[static_cast<size_t>(z * domain.edge + x)] =
                rawOrdinaryStagePixel(pageAt(worldX, worldZ), worldX, worldZ);
        }
    }
    validateRawOrdinaryStageContacts(domain);
    domain.stages = solveOrdinaryStageInfimum(domain.pixels, domain.edge);
    validateSolvedOrdinaryStageContacts(domain);
    return domain;
}

OrdinaryStageDomain cropOrdinaryStageDomain(const OrdinaryStageDomain& source, int halo) {
    if (halo < 0 || halo > source.halo)
        throw std::invalid_argument("invalid ordinary stage domain crop");
    const int offset = source.halo - halo;
    OrdinaryStageDomain result;
    result.originX = source.originX + offset;
    result.originZ = source.originZ + offset;
    result.halo = halo;
    result.edge = ORDINARY_STAGE_TILE_EDGE + halo * 2;
    result.pixels.resize(static_cast<size_t>(result.edge * result.edge));
    for (int z = 0; z < result.edge; ++z) {
        const auto begin = source.pixels.begin() +
                           static_cast<std::ptrdiff_t>((z + offset) * source.edge + offset);
        std::copy_n(begin, result.edge,
                    result.pixels.begin() + static_cast<std::ptrdiff_t>(z * result.edge));
    }
    validateRawOrdinaryStageContacts(result);
    result.stages = solveOrdinaryStageInfimum(result.pixels, result.edge);
    validateSolvedOrdinaryStageContacts(result);
    return result;
}

bool ordinaryStageCoresEqual(const OrdinaryStageDomain& first, const OrdinaryStageDomain& second) {
    for (int z = 0; z < ORDINARY_STAGE_TILE_EDGE; ++z) {
        for (int x = 0; x < ORDINARY_STAGE_TILE_EDGE; ++x) {
            const int firstIndex = (z + first.halo) * first.edge + x + first.halo;
            const int secondIndex = (z + second.halo) * second.edge + x + second.halo;
            if (first.stages[static_cast<size_t>(firstIndex)] !=
                second.stages[static_cast<size_t>(secondIndex)]) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<const OrdinaryStageTile>
finishOrdinaryStageTile(int64_t tileOriginX, int64_t tileOriginZ,
                        const OrdinaryStageDomain& authority) {
    auto tile = std::make_shared<OrdinaryStageTile>();
    tile->originX = tileOriginX;
    tile->originZ = tileOriginZ;
    tile->certifiedHalo = static_cast<uint8_t>(authority.halo);
    for (int z = 0; z < ORDINARY_STAGE_TILE_EDGE; ++z) {
        for (int x = 0; x < ORDINARY_STAGE_TILE_EDGE; ++x) {
            const int domainIndex = (z + authority.halo) * authority.edge + x + authority.halo;
            const RawOrdinaryStagePixel& pixel = authority.pixels[static_cast<size_t>(domainIndex)];
            if (!pixel.wet) continue;
            tile->entries.push_back({
                .index = static_cast<uint16_t>(z * ORDINARY_STAGE_TILE_EDGE + x),
                .stageEighths = authority.stages[static_cast<size_t>(domainIndex)],
                .body = pixel.body,
                .ordinaryReach = pixel.ordinaryReach,
            });
        }
    }
    return tile;
}

std::shared_ptr<const OrdinaryStageTile>
buildOrdinaryStageTile(const NativePage& page, int64_t tileOriginX, int64_t tileOriginZ,
                       const NativeStagePageResolver* resolvePage) {
    OrdinaryStageDomain certificate = sampleOrdinaryStageDomain(
        page, tileOriginX, tileOriginZ, ORDINARY_STAGE_CERTIFICATE_HALO, resolvePage);
    OrdinaryStageDomain primary = cropOrdinaryStageDomain(certificate, ORDINARY_STAGE_PRIMARY_HALO);
    if (ordinaryStageCoresEqual(primary, certificate))
        return finishOrdinaryStageTile(tileOriginX, tileOriginZ, primary);

    OrdinaryStageDomain maximum = sampleOrdinaryStageDomain(
        page, tileOriginX, tileOriginZ, ORDINARY_STAGE_MAXIMUM_HALO, resolvePage);
    if (!ordinaryStageCoresEqual(certificate, maximum))
        throw std::runtime_error("ordinary stage tile exceeds its certified influence bound");
    return finishOrdinaryStageTile(tileOriginX, tileOriginZ, certificate);
}

std::shared_ptr<const OrdinaryStageTile>
ordinaryStageTile(const NativePage& page, int64_t tileOriginX, int64_t tileOriginZ,
                  const NativeStagePageResolver* resolvePage) {
    const OrdinaryStageTileKey key{tileOriginX, tileOriginZ};
    std::shared_ptr<OrdinaryStageTileFlight> flight;
    bool builder = false;
    {
        std::unique_lock lock(page.ordinaryStageTilesMutex);
        if (const auto found = page.ordinaryStageTiles.find(key);
            found != page.ordinaryStageTiles.end()) {
            page.ordinaryStageTileRecency.splice(page.ordinaryStageTileRecency.begin(),
                                                 page.ordinaryStageTileRecency,
                                                 found->second.recency);
            ++page.ordinaryStageTileHits;
            return found->second.tile;
        }
        ++page.ordinaryStageTileMisses;
        const auto [found, inserted] = page.ordinaryStageTileFlights.try_emplace(
            key, std::make_shared<OrdinaryStageTileFlight>());
        flight = found->second;
        builder = inserted;
        if (!builder) {
            ++page.ordinaryStageTileBuildWaits;
            flight->ready.wait(lock, [&] { return flight->complete; });
            if (flight->failure) std::rethrow_exception(flight->failure);
            if (!flight->tile)
                throw std::logic_error("ordinary stage tile flight completed without a tile");
            return flight->tile;
        }
    }
    const auto buildStart = std::chrono::steady_clock::now();
    try {
        const std::shared_ptr<const OrdinaryStageTile> built =
            buildOrdinaryStageTile(page, tileOriginX, tileOriginZ, resolvePage);
        const uint64_t buildNanoseconds =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - buildStart)
                                      .count());
        constexpr size_t CACHE_ENTRY_OVERHEAD_BYTES = 128;
        const size_t builtBytes = sizeof(OrdinaryStageTile) +
                                  built->entries.capacity() * sizeof(OrdinaryStageTileEntry) +
                                  CACHE_ENTRY_OVERHEAD_BYTES;
        std::unique_lock lock(page.ordinaryStageTilesMutex);
        while (page.ordinaryStageTileBytes + builtBytes >
                   NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET &&
               !page.ordinaryStageTileRecency.empty()) {
            const OrdinaryStageTileKey evicted = page.ordinaryStageTileRecency.back();
            page.ordinaryStageTileRecency.pop_back();
            const auto found = page.ordinaryStageTiles.find(evicted);
            if (found == page.ordinaryStageTiles.end())
                throw std::logic_error("ordinary stage tile LRU is inconsistent");
            page.ordinaryStageTileBytes -= found->second.bytes;
            page.ordinaryStageTiles.erase(found);
        }
        if (builtBytes > NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET)
            throw std::runtime_error("ordinary stage tile exceeds its cache budget");
        page.ordinaryStageTileRecency.push_front(key);
        const auto inserted = page.ordinaryStageTiles.emplace(
            key, OrdinaryStageTileCacheEntry{
                     .tile = built,
                     .bytes = builtBytes,
                     .recency = page.ordinaryStageTileRecency.begin(),
                 });
        if (!inserted.second) throw std::logic_error("ordinary stage tile cache insertion failed");
        page.ordinaryStageTileBytes += builtBytes;
        page.ordinaryStageTilePeakBytes =
            std::max(page.ordinaryStageTilePeakBytes, page.ordinaryStageTileBytes);
        ++page.ordinaryStageTileBuilds;
        page.ordinaryStageTileBuildNanoseconds += buildNanoseconds;
        page.ordinaryStageTileExpandedBuilds +=
            built->certifiedHalo > ORDINARY_STAGE_PRIMARY_HALO ? 1U : 0U;
        flight->tile = built;
        flight->complete = true;
        page.ordinaryStageTileFlights.erase(key);
        lock.unlock();
        flight->ready.notify_all();
        return built;
    } catch (...) {
        std::unique_lock lock(page.ordinaryStageTilesMutex);
        ++page.ordinaryStageTileFailures;
        flight->failure = std::current_exception();
        flight->complete = true;
        page.ordinaryStageTileFlights.erase(key);
        lock.unlock();
        flight->ready.notify_all();
        throw;
    }
}

void applyOrdinaryStageAuthority(const NativePage& page, double worldX, double worldZ,
                                 ChannelProjection& channel,
                                 const NativeStagePageResolver* resolvePage) {
    if (channel.body == NO_WATER_BODY || channel.waterfall || worldX != std::floor(worldX) ||
        worldZ != std::floor(worldZ)) {
        return;
    }
    const int64_t blockX = static_cast<int64_t>(worldX);
    const int64_t blockZ = static_cast<int64_t>(worldZ);
    const int64_t tileOriginX = world_coord::floorMultiple(blockX, ORDINARY_STAGE_TILE_EDGE);
    const int64_t tileOriginZ = world_coord::floorMultiple(blockZ, ORDINARY_STAGE_TILE_EDGE);
    const std::shared_ptr<const OrdinaryStageTile> tile =
        ordinaryStageTile(page, tileOriginX, tileOriginZ, resolvePage);
    const uint16_t index = static_cast<uint16_t>((blockZ - tileOriginZ) * ORDINARY_STAGE_TILE_EDGE +
                                                 blockX - tileOriginX);
    const auto found =
        std::ranges::lower_bound(tile->entries, index, {}, &OrdinaryStageTileEntry::index);
    const uint32_t reach = effectiveOrdinaryReach(page, channel);
    if (found == tile->entries.end() || found->index != index || found->body != channel.body ||
        reach == 0 || found->ordinaryReach != reach) {
        return;
    }
    channel.stage = static_cast<double>(found->stageEighths) / 8.0;
}

BasinSample samplePage(const NativePage& page, double x, double z,
                       const NativeStagePageResolver* resolvePage = nullptr,
                       bool reconcileOrdinaryStage = true) {
    const double gridX = (x - static_cast<double>(page.originX)) / NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const double gridZ = (z - static_cast<double>(page.originZ)) / NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const int nearestX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int nearestZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    const int nearest = indexOf(nearestX, nearestZ);
    const size_t cell = static_cast<size_t>(nearest);
    const uint8_t flags = page.flags[cell];
    const double undisturbed = reconstruct(page.rawElevation, gridX, gridZ);
    const auto [slopeX, slopeZ] = reconstructGradient(page.rawElevation, gridX, gridZ);
    const double signedLakeDistance = reconstruct(page.lakeShoreDistance, gridX, gridZ);
    const bool ocean = undisturbed < SEA_LEVEL;
    const bool lake = !ocean && signedLakeDistance > 0.0;
    const std::optional<size_t> nearbyLakeCell =
        lake ? supportingLakeCell(page, gridX, gridZ) : std::nullopt;
    const size_t lakeCell = lake && nearbyLakeCell ? *nearbyLakeCell : cell;
    const double lakeStage = lake ? page.waterSurface[lakeCell] : 0.0;
    ChannelProjection channel = page.channelProximity[cell] != 0
                                    ? projectChannel(page, x, z, gridX, gridZ)
                                    : ChannelProjection{};
    applyReceiverOwnedStandingBackwater(page, lake, channel);
    const bool channelInside = channelRibbonSupported(channel, undisturbed);
    const bool river = !ocean && !lake && channelInside;
    const bool waterfall = channelInside && channel.waterfall;
    if (river && !waterfall && reconcileOrdinaryStage)
        applyOrdinaryStageAuthority(page, x, z, channel, resolvePage);
    const bool wetland = !ocean && !lake && !river && (flags & CELL_WETLAND) != 0 &&
                         page.waterBodyIds[cell] != NO_WATER_BODY &&
                         std::isfinite(page.waterSurface[cell]);

    BasinSample result;
    result.flowX = river || waterfall ? channel.flowX : reconstruct(page.flowX, gridX, gridZ);
    result.flowZ = river || waterfall ? channel.flowZ : reconstruct(page.flowZ, gridX, gridZ);
    const double flowLength = std::hypot(result.flowX, result.flowZ);
    if (flowLength > 1.0e-9) {
        result.flowX /= flowLength;
        result.flowZ /= flowLength;
    } else {
        result.flowX = 1.0;
        result.flowZ = 0.0;
    }
    result.surfaceElevation = undisturbed;
    result.terrainSlope = std::hypot(slopeX, slopeZ);
    // The explicit fall owns its full curtain footprint, including columns
    // where the widened ribbon overlaps the upstream lake or downstream
    // ocean classification. Publish the receiving stage there; retaining a
    // standing body's top stage turns the curtain into a horizontal shelf.
    const double ordinaryWaterSurface = ocean     ? SEA_LEVEL
                                        : lake    ? lakeStage
                                        : river   ? channel.stage
                                        : wetland ? page.waterSurface[cell]
                                                  : 0.0;
    result.waterSurface = native_hydrology_detail::explicitFallPublishedWaterSurface(
        waterfall, channel.waterfallBottom, ordinaryWaterSurface);
    if (river || wetland) {
        const double bedDepth =
            river ? channelBedDepthAt(channel) : static_cast<double>(WETLAND_BED_DEPTH_BLOCKS);
        result.surfaceElevation = std::min(result.surfaceElevation, result.waterSurface - bedDepth);
    }
    result.discharge =
        river || waterfall ? channel.discharge : reconstruct(page.discharge, gridX, gridZ);
    result.baseflow = reconstruct(page.baseflow, gridX, gridZ);
    result.precipitationSeasonality = reconstruct(page.seasonality, gridX, gridZ);
    result.groundwaterRechargeMm = reconstruct(page.groundwaterRecharge, gridX, gridZ);
    const double reconstructedGroundwaterHead = reconstruct(page.groundwaterHead, gridX, gridZ);
    const double hydraulicHead = wetland
                                     ? std::max(reconstructedGroundwaterHead, result.waterSurface)
                                     : reconstructedGroundwaterHead;
    result.groundwaterHead =
        wetland ? hydraulicHead : std::min(result.surfaceElevation, hydraulicHead);
    result.hydroperiod =
        lake      ? 1.0
        : wetland ? wetlandHydroperiod(undisturbed, hydraulicHead, result.precipitationSeasonality)
                  : 0.0;
    result.channelGradient =
        river || waterfall ? channel.gradient : reconstruct(page.channelGradient, gridX, gridZ);
    result.erosionDepth = std::max(0.0, undisturbed - result.surfaceElevation);
    result.lakeDepth = lake ? std::max(0.0, result.waterSurface - result.surfaceElevation) : 0.0;
    result.lakeShoreDistance = signedLakeDistance;
    result.shoreWaterSurface = lake ? result.waterSurface : 0.0;
    result.channelDistance = std::isfinite(channel.distance) ? channel.distance : 1.0e9;
    result.streamOrder =
        river || waterfall || (channelInside && channel.delta) ? channel.streamOrder : 0;
    result.distributaryCount = channelInside && channel.delta ? channel.distributaryCount : 0;
    result.channelWidth = channel.width;
    result.channelDepth =
        river ? std::max(0.125, result.waterSurface - result.surfaceElevation) : 0.0;
    result.sediment =
        river ? result.discharge * std::clamp(result.channelGradient, 0.002, 0.2) : 0.0;
    result.waterBodyId = ocean     ? page.oceanId
                         : lake    ? page.waterBodyIds[lakeCell]
                         : river   ? channel.body
                         : wetland ? page.waterBodyIds[cell]
                                   : NO_WATER_BODY;
    result.ocean = ocean;
    result.lake = lake;
    result.river = river;
    result.wetland = wetland;
    result.delta = channelInside && channel.delta;
    result.estuary = channelInside && channel.estuary;
    result.brackish = result.estuary;
    result.waterfall = waterfall;
    result.waterfallAnchor = result.waterfall && channel.waterfallAnchor;
    result.waterfallTop = result.waterfall ? channel.waterfallTop : 0.0;
    result.waterfallBottom = result.waterfall ? channel.waterfallBottom : 0.0;
    result.waterfallWidth = result.waterfall ? result.channelWidth : 0.0;
    if (result.waterfall) {
        result.transitionOwnerKind = WaterTransitionKind::EXPLICIT_FALL;
        result.transitionOwnerId = nativeFallTransitionId(page, channel);
        result.generatedFluidLevel = 7;
    }
    const double baseflowFraction = result.baseflow / std::max(1.0e-9, result.discharge);
    result.perennial = river && baseflowFraction >= 0.12;
    result.ephemeral = river && !result.perennial;
    result.outlet = ocean ? BasinOutlet::OCEAN : BasinOutlet::NONE;
    if (lake) {
        if (const auto found = page.lakeStats.find(result.waterBodyId);
            found != page.lakeStats.end()) {
            result.lakeAreaSquareKilometers = found->second.areaSquareKilometers;
            result.lakeVolumeCubicMeters = found->second.volumeCubicMeters;
            result.lakeRunoffMmSquareKilometers = found->second.runoffMmSquareKilometers;
        }
        result.lakeSpillSurface = result.waterSurface;
    }
    result.valid = true;
    return result;
}

// A regular native-spacing grid always reaches the same align-corners=false
// block-center phase. Keep the public point sampler on the general path and
// use this equivalent cell sampler only after sampleInteriorGrid has proved
// page ownership and native-grid alignment. Every reconstructed field retains
// the shared fractional phase used by exact terrain.
BasinSample sampleNativeLatticeCell(const NativePage& page, double x, double z, int rasterX,
                                    int rasterZ,
                                    const NativeStagePageResolver* resolvePage = nullptr,
                                    bool reconcileOrdinaryStage = true) {
    const double gridX = static_cast<double>(rasterX) + WORLD_SAMPLE_NATIVE_OFFSET;
    const double gridZ = static_cast<double>(rasterZ) + WORLD_SAMPLE_NATIVE_OFFSET;
    const int nearest = indexOf(rasterX, rasterZ);
    const size_t cell = static_cast<size_t>(nearest);
    const uint8_t flags = page.flags[cell];
    const double undisturbed = reconstruct(page.rawElevation, gridX, gridZ);
    const auto [slopeX, slopeZ] = reconstructGradient(page.rawElevation, gridX, gridZ);
    const double signedLakeDistance = reconstruct(page.lakeShoreDistance, gridX, gridZ);
    const bool ocean = undisturbed < SEA_LEVEL;
    const bool lake = !ocean && signedLakeDistance > 0.0;
    const std::optional<size_t> nearbyLakeCell =
        lake ? supportingLakeCell(page, gridX, gridZ) : std::nullopt;
    const size_t lakeCell = lake && nearbyLakeCell ? *nearbyLakeCell : cell;
    const double lakeStage = lake ? page.waterSurface[lakeCell] : 0.0;
    ChannelProjection channel = page.channelProximity[cell] != 0
                                    ? projectChannel(page, x, z, gridX, gridZ)
                                    : ChannelProjection{};
    applyReceiverOwnedStandingBackwater(page, lake, channel);
    const bool channelInside = channelRibbonSupported(channel, undisturbed);
    const bool river = !ocean && !lake && channelInside;
    const bool waterfall = channelInside && channel.waterfall;
    if (river && !waterfall && reconcileOrdinaryStage)
        applyOrdinaryStageAuthority(page, x, z, channel, resolvePage);
    const bool wetland = !ocean && !lake && !river && (flags & CELL_WETLAND) != 0 &&
                         page.waterBodyIds[cell] != NO_WATER_BODY &&
                         std::isfinite(page.waterSurface[cell]);

    BasinSample result;
    result.flowX = river || waterfall ? channel.flowX : reconstruct(page.flowX, gridX, gridZ);
    result.flowZ = river || waterfall ? channel.flowZ : reconstruct(page.flowZ, gridX, gridZ);
    const double flowLength = std::hypot(result.flowX, result.flowZ);
    if (flowLength > 1.0e-9) {
        result.flowX /= flowLength;
        result.flowZ /= flowLength;
    } else {
        result.flowX = 1.0;
        result.flowZ = 0.0;
    }
    result.surfaceElevation = undisturbed;
    result.terrainSlope = std::hypot(slopeX, slopeZ);
    const double ordinaryWaterSurface = ocean     ? SEA_LEVEL
                                        : lake    ? lakeStage
                                        : river   ? channel.stage
                                        : wetland ? page.waterSurface[cell]
                                                  : 0.0;
    result.waterSurface = native_hydrology_detail::explicitFallPublishedWaterSurface(
        waterfall, channel.waterfallBottom, ordinaryWaterSurface);
    if (river || wetland) {
        const double bedDepth =
            river ? channelBedDepthAt(channel) : static_cast<double>(WETLAND_BED_DEPTH_BLOCKS);
        result.surfaceElevation = std::min(result.surfaceElevation, result.waterSurface - bedDepth);
    }
    result.discharge =
        river || waterfall ? channel.discharge : reconstruct(page.discharge, gridX, gridZ);
    result.baseflow = reconstruct(page.baseflow, gridX, gridZ);
    result.precipitationSeasonality = reconstruct(page.seasonality, gridX, gridZ);
    result.groundwaterRechargeMm = reconstruct(page.groundwaterRecharge, gridX, gridZ);
    const double groundwaterHead = reconstruct(page.groundwaterHead, gridX, gridZ);
    const double hydraulicHead =
        wetland ? std::max(groundwaterHead, result.waterSurface) : groundwaterHead;
    result.groundwaterHead =
        wetland ? hydraulicHead : std::min(result.surfaceElevation, hydraulicHead);
    result.hydroperiod =
        lake      ? 1.0
        : wetland ? wetlandHydroperiod(undisturbed, hydraulicHead, result.precipitationSeasonality)
                  : 0.0;
    result.channelGradient =
        river || waterfall ? channel.gradient : reconstruct(page.channelGradient, gridX, gridZ);
    result.erosionDepth = std::max(0.0, undisturbed - result.surfaceElevation);
    result.lakeDepth = lake ? std::max(0.0, result.waterSurface - result.surfaceElevation) : 0.0;
    result.lakeShoreDistance = signedLakeDistance;
    result.shoreWaterSurface = lake ? result.waterSurface : 0.0;
    result.channelDistance = std::isfinite(channel.distance) ? channel.distance : 1.0e9;
    result.streamOrder =
        river || waterfall || (channelInside && channel.delta) ? channel.streamOrder : 0;
    result.distributaryCount = channelInside && channel.delta ? channel.distributaryCount : 0;
    result.channelWidth = channel.width;
    result.channelDepth =
        river ? std::max(0.125, result.waterSurface - result.surfaceElevation) : 0.0;
    result.sediment =
        river ? result.discharge * std::clamp(result.channelGradient, 0.002, 0.2) : 0.0;
    result.waterBodyId = ocean     ? page.oceanId
                         : lake    ? page.waterBodyIds[lakeCell]
                         : river   ? channel.body
                         : wetland ? page.waterBodyIds[cell]
                                   : NO_WATER_BODY;
    result.ocean = ocean;
    result.lake = lake;
    result.river = river;
    result.wetland = wetland;
    result.delta = channelInside && channel.delta;
    result.estuary = channelInside && channel.estuary;
    result.brackish = result.estuary;
    result.waterfall = waterfall;
    result.waterfallAnchor = result.waterfall && channel.waterfallAnchor;
    result.waterfallTop = result.waterfall ? channel.waterfallTop : 0.0;
    result.waterfallBottom = result.waterfall ? channel.waterfallBottom : 0.0;
    result.waterfallWidth = result.waterfall ? result.channelWidth : 0.0;
    if (result.waterfall) {
        result.transitionOwnerKind = WaterTransitionKind::EXPLICIT_FALL;
        result.transitionOwnerId = nativeFallTransitionId(page, channel);
        result.generatedFluidLevel = 7;
    }
    const double baseflowFraction = result.baseflow / std::max(1.0e-9, result.discharge);
    result.perennial = river && baseflowFraction >= 0.12;
    result.ephemeral = river && !result.perennial;
    result.outlet = ocean ? BasinOutlet::OCEAN : BasinOutlet::NONE;
    if (lake) {
        if (const auto found = page.lakeStats.find(result.waterBodyId);
            found != page.lakeStats.end()) {
            result.lakeAreaSquareKilometers = found->second.areaSquareKilometers;
            result.lakeVolumeCubicMeters = found->second.volumeCubicMeters;
            result.lakeRunoffMmSquareKilometers = found->second.runoffMmSquareKilometers;
        }
        result.lakeSpillSurface = result.waterSurface;
    }
    result.valid = true;
    return result;
}

struct NativeDryLocalityProof {
    std::vector<float> permanentOceanEscapeElevation;
    std::vector<uint8_t> boundaryDownstream;
    std::vector<uint8_t> boundarySourceProximity;
    bool complete = false;

    [[nodiscard]] size_t byteSize() const noexcept {
        return permanentOceanEscapeElevation.capacity() * sizeof(float) +
               boundaryDownstream.capacity() * sizeof(uint8_t) +
               boundarySourceProximity.capacity() * sizeof(uint8_t);
    }
};

struct CertifiedDrySample {
    NativeHydrologyPosition position;
    BasinSample sample;
};

bool certifiedDrySampleLess(const CertifiedDrySample& left,
                            const NativeHydrologyPosition& right) noexcept {
    return left.position < right;
}

std::optional<NativeHydrologyPosition>
exactIntegerHydrologyPosition(BasinSamplePosition position) noexcept {
    constexpr long double MINIMUM = static_cast<long double>(std::numeric_limits<int64_t>::min());
    constexpr long double MAXIMUM_EXCLUSIVE =
        static_cast<long double>(std::numeric_limits<int64_t>::max()) + 1.0L;
    if (!std::isfinite(position.x) || !std::isfinite(position.z) ||
        std::trunc(position.x) != position.x || std::trunc(position.z) != position.z ||
        static_cast<long double>(position.x) < MINIMUM ||
        static_cast<long double>(position.x) >= MAXIMUM_EXCLUSIVE ||
        static_cast<long double>(position.z) < MINIMUM ||
        static_cast<long double>(position.z) >= MAXIMUM_EXCLUSIVE) {
        return std::nullopt;
    }
    return NativeHydrologyPosition{.x = static_cast<int64_t>(position.x),
                                   .z = static_cast<int64_t>(position.z)};
}

bool nativeDrySampleIsInstallable(const BasinSample& sample) noexcept {
    constexpr double WATER_DEPTH_EPSILON = 0.01;
    return sample.valid && std::isfinite(sample.surfaceElevation) &&
           std::isfinite(sample.waterSurface) && !sample.ocean && !sample.lake && !sample.river &&
           !sample.wetland && !sample.waterfall && !sample.delta && !sample.estuary &&
           !sample.brackish && sample.generatedFluidLevel == 0 &&
           sample.transitionOwnerKind == WaterTransitionKind::NONE &&
           sample.transitionOwnerId == 0 && sample.waterBodyId == NO_WATER_BODY &&
           sample.waterSurface <= sample.surfaceElevation + WATER_DEPTH_EPSILON;
}

bool nativeDryCertificatePageLayoutValid(const NativePage& page) {
    const auto complete = [](const auto& values) { return values.size() == RASTER_CELLS; };
    return complete(page.routingElevationMeters) && complete(page.rawElevation) &&
           complete(page.waterSurface) && complete(page.discharge) && complete(page.baseflow) &&
           complete(page.seasonality) && complete(page.groundwaterRecharge) &&
           complete(page.groundwaterHead) && complete(page.flowX) && complete(page.flowZ) &&
           complete(page.channelGradient) && complete(page.lakeShoreDistance) &&
           complete(page.receiverFirst) && complete(page.receiverSecond) &&
           complete(page.receiverSecondWeight) && complete(page.waterBodyIds) &&
           complete(page.streamOrder) && complete(page.flags) && complete(page.channelProximity) &&
           page.channelRowOffsets.size() == static_cast<size_t>(RASTER_EDGE + 1);
}

NativeDryLocalityProof buildNativeDryLocalityProof(const NativePage& page) {
    NativeDryLocalityProof proof;
    if (!nativeDryCertificatePageLayoutValid(page)) return proof;

    proof.permanentOceanEscapeElevation.assign(RASTER_CELLS,
                                               std::numeric_limits<float>::infinity());
    std::priority_queue<FloodNode, std::vector<FloodNode>, FloodNodeGreater> frontier;
    bool hasPermanentOcean = false;
    for (int index = 0; index < RASTER_CELLS; ++index) {
        const size_t cell = static_cast<size_t>(index);
        const float elevation = page.routingElevationMeters[static_cast<size_t>(index)];
        const int32_t first = page.receiverFirst[cell];
        const int32_t second = page.receiverSecond[cell];
        const float secondWeight = page.receiverSecondWeight[cell];
        if (!std::isfinite(elevation) || !std::isfinite(secondWeight) || secondWeight < 0.0F ||
            secondWeight > 1.0F || first < -1 || first >= RASTER_CELLS || second < -1 ||
            second >= RASTER_CELLS || (first < 0 && second >= 0)) {
            return {};
        }
        if (elevation >= NATIVE_HYDROLOGY_SEA_LEVEL_METERS) continue;
        hasPermanentOcean = true;
        proof.permanentOceanEscapeElevation[cell] = elevation;
        frontier.push({.elevation = elevation, .index = index});
    }
    if (!hasPermanentOcean) return {};

    // Unlike the page-local Priority-Flood, only immutable negative learned
    // elevation is a terminal here. A page edge is not evidence of an outlet
    // because its continuation is deliberately absent from this proof.
    while (!frontier.empty()) {
        const FloodNode current = frontier.top();
        frontier.pop();
        if (current.index < 0 || current.index >= RASTER_CELLS) return {};
        if (current.elevation !=
            proof.permanentOceanEscapeElevation[static_cast<size_t>(current.index)]) {
            continue;
        }
        const int x = current.index % RASTER_EDGE;
        const int z = current.index / RASTER_EDGE;
        for (size_t direction = 0; direction < DX.size(); ++direction) {
            const int neighborX = x + DX[direction];
            const int neighborZ = z + DZ[direction];
            if (!inside(neighborX, neighborZ)) continue;
            const int neighbor = indexOf(neighborX, neighborZ);
            const float candidate = std::max(
                current.elevation, page.routingElevationMeters[static_cast<size_t>(neighbor)]);
            float& selected = proof.permanentOceanEscapeElevation[static_cast<size_t>(neighbor)];
            if (candidate >= selected) continue;
            selected = candidate;
            frontier.push({.elevation = candidate, .index = neighbor});
        }
    }

    proof.boundaryDownstream.assign(RASTER_CELLS, 0);
    std::queue<int32_t> downstream;
    const auto admitBoundary = [&](int x, int z) {
        const int32_t index = indexOf(x, z);
        uint8_t& marked = proof.boundaryDownstream[static_cast<size_t>(index)];
        if (marked != 0) return;
        marked = 1;
        downstream.push(index);
    };
    for (int along = CORE_BEGIN; along <= CORE_END; ++along) {
        admitBoundary(CORE_BEGIN, along);
        admitBoundary(CORE_END, along);
        admitBoundary(along, CORE_BEGIN);
        admitBoundary(along, CORE_END);
    }
    while (!downstream.empty()) {
        const int32_t source = downstream.front();
        downstream.pop();
        const size_t cell = static_cast<size_t>(source);
        const Receiver receiver{
            .first = page.receiverFirst[cell],
            .second = page.receiverSecond[cell],
            .secondWeight = page.receiverSecondWeight[cell],
        };
        for (int branch = 0; branch < 2; ++branch) {
            if (!receiverBranchActive(receiver, branch)) continue;
            const int32_t target = branch == 0 ? receiver.first : receiver.second;
            if (target < 0 || target >= RASTER_CELLS) return {};
            uint8_t& marked = proof.boundaryDownstream[static_cast<size_t>(target)];
            if (marked != 0) continue;
            marked = 1;
            downstream.push(target);
        }
    }

    // Match projectChannel's complete 17 by 17 source search. Every cell in
    // that source footprint must be independent of unresolved boundary
    // runoff. Limiting this proof to currently flagged river cells would
    // assume the very cross-page discharge classification being avoided.
    std::vector<uint8_t> horizontal(RASTER_CELLS, 0);
    for (int z = 0; z < RASTER_EDGE; ++z) {
        int count = 0;
        for (int x = -CHANNEL_SEARCH_RADIUS; x < RASTER_EDGE; ++x) {
            const int entering = x + CHANNEL_SEARCH_RADIUS;
            const int leaving = x - CHANNEL_SEARCH_RADIUS - 1;
            if (entering < RASTER_EDGE) {
                const size_t cell = static_cast<size_t>(indexOf(entering, z));
                if (proof.boundaryDownstream[cell] != 0) ++count;
            }
            if (leaving >= 0) {
                const size_t cell = static_cast<size_t>(indexOf(leaving, z));
                if (proof.boundaryDownstream[cell] != 0) --count;
            }
            if (x >= 0) horizontal[static_cast<size_t>(indexOf(x, z))] = count > 0 ? 1U : 0U;
        }
    }
    proof.boundarySourceProximity.assign(RASTER_CELLS, 0);
    for (int x = 0; x < RASTER_EDGE; ++x) {
        int count = 0;
        for (int z = -CHANNEL_SEARCH_RADIUS; z < RASTER_EDGE; ++z) {
            const int entering = z + CHANNEL_SEARCH_RADIUS;
            const int leaving = z - CHANNEL_SEARCH_RADIUS - 1;
            if (entering < RASTER_EDGE &&
                horizontal[static_cast<size_t>(indexOf(x, entering))] != 0) {
                ++count;
            }
            if (leaving >= 0 && horizontal[static_cast<size_t>(indexOf(x, leaving))] != 0) --count;
            if (z >= 0)
                proof.boundarySourceProximity[static_cast<size_t>(indexOf(x, z))] =
                    count > 0 ? 1U : 0U;
        }
    }
    proof.complete = true;
    return proof;
}

std::optional<BasinSample> certifyNativeDryPoint(const NativePage& page,
                                                 const NativeDryLocalityProof& proof, double x,
                                                 double z) {
    if (!proof.complete) return std::nullopt;
    const double gridX = (x - static_cast<double>(page.originX)) / NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const double gridZ = (z - static_cast<double>(page.originZ)) / NATIVE_HYDROLOGY_RASTER_SPACING +
                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, RASTER_EDGE - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, RASTER_EDGE - 2);
    const int nearestX = std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
    const int nearestZ = std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
    const size_t nearest = static_cast<size_t>(indexOf(nearestX, nearestZ));

    constexpr uint8_t UNCERTAIN_FLAGS = CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND |
                                        CELL_WATERFALL | CELL_WETLAND_CANDIDATE | CELL_ESTUARY |
                                        CELL_DELTA;
    const std::array<size_t, 4> supports{
        static_cast<size_t>(indexOf(x0, z0)),
        static_cast<size_t>(indexOf(x0 + 1, z0)),
        static_cast<size_t>(indexOf(x0, z0 + 1)),
        static_cast<size_t>(indexOf(x0 + 1, z0 + 1)),
    };
    for (const size_t support : supports) {
        if ((page.flags[support] & UNCERTAIN_FLAGS) != 0 ||
            proof.boundaryDownstream[support] != 0) {
            return std::nullopt;
        }
        const float escape = proof.permanentOceanEscapeElevation[support];
        const float elevation = page.routingElevationMeters[support];
        if (!std::isfinite(escape) || !std::isfinite(elevation) ||
            escape - elevation > LAKE_DEPTH_THRESHOLD_METERS + ROUTING_ELEVATION_EPSILON_METERS) {
            return std::nullopt;
        }
    }
    if ((page.flags[nearest] & CELL_WETLAND_CANDIDATE) != 0 ||
        proof.boundarySourceProximity[nearest] != 0) {
        return std::nullopt;
    }

    BasinSample sample = samplePage(page, x, z);
    if (!nativeDrySampleIsInstallable(sample)) return std::nullopt;
    return sample;
}

struct HandoffCandidate {
    std::shared_ptr<const NativePage> page;
    BasinSample sample;
    // Lake reconciliation can admit a cardinal neighbor because the owner's
    // depression reaches their shared edge even when the query itself is far
    // outside that neighbor's two-cell raster apron. Such a page contributes
    // immutable edge summaries only. Its clamped point reconstruction must
    // never compete with the half-open owner at the query position.
    bool containsQuery = true;
};

const DepressionSummary* depressionSummary(const NativePage& page, WaterBodyId body) {
    const auto found =
        std::ranges::find(page.depressionSummaries, body, &DepressionSummary::waterBodyId);
    return found == page.depressionSummaries.end() ? nullptr : &*found;
}

bool pageRasterContains(const NativePage& page, double x, double z) {
    constexpr long double APRON_BLOCKS = RASTER_APRON * NATIVE_HYDROLOGY_RASTER_SPACING;
    const long double queryX = x;
    const long double queryZ = z;
    const long double originX = page.originX;
    const long double originZ = page.originZ;
    return queryX >= originX - APRON_BLOCKS &&
           queryX <= originX + NATIVE_HYDROLOGY_PAGE_EDGE + APRON_BLOCKS &&
           queryZ >= originZ - APRON_BLOCKS &&
           queryZ <= originZ + NATIVE_HYDROLOGY_PAGE_EDGE + APRON_BLOCKS;
}

struct LakeNode {
    PageKey page;
    WaterBodyId body = NO_WATER_BODY;

    auto operator<=>(const LakeNode&) const = default;
};

struct LakePortal {
    LakeNode first;
    LakeNode second;
    float minimumWetStage = 0.0F;
    float compatibleStage = 0.0F;
    int64_t x = 0;
    int64_t z = 0;
};

struct LakePortalConnection {
    LakeNode first;
    LakeNode second;
    float minimumWetStage = 0.0F;
    float compatibleStage = 0.0F;
    int64_t x = 0;
    int64_t z = 0;
};

struct LakePairKey {
    PageKey first;
    PageKey second;

    auto operator<=>(const LakePairKey&) const = default;
};

constexpr size_t TILED_LAKE_PAIR_CACHE_ENTRIES = 512;
constexpr size_t OPEN_LAKE_REGION_CACHE_BYTE_BUDGET = 512U * 1024U * 1024U;
constexpr size_t OPEN_LAKE_REGION_CACHE_ENTRIES = 128;

// Compact immutable summaries form a page-tiled spill graph. The traversal
// that populates this object is bounded independently from the native-page
// cache and processes every frontier in lexicographic order. It never stores
// a raster, raises terrain, or infers connectivity through a dry edge.
class TiledLakeSpillHierarchy {
public:
    bool add(LakeNode node, const DepressionSummary& summary) {
        return nodes_.try_emplace(node, summary).second;
    }

    void connect(const LakePortalConnection& connection) {
        portals_.push_back({
            .first = connection.first,
            .second = connection.second,
            .minimumWetStage = connection.minimumWetStage,
            .compatibleStage = connection.compatibleStage,
            .x = connection.x,
            .z = connection.z,
        });
    }

    const std::map<LakeNode, DepressionSummary>& nodes() const noexcept { return nodes_; }
    const std::vector<LakePortal>& portals() const noexcept { return portals_; }

private:
    std::map<LakeNode, DepressionSummary> nodes_;
    std::vector<LakePortal> portals_;
};

struct TiledLakeResolution {
    WaterBodyId canonicalBody = NO_WATER_BODY;
    float stage = 0.0F;
    double areaSquareKilometers = 0.0;
    double volumeCubicMeters = 0.0;
    double runoffMmSquareKilometers = 0.0;
    BasinOutlet outlet = BasinOutlet::NONE;
    int64_t outletX = 0;
    int64_t outletZ = 0;
    size_t pageCount = 0;
};

bool adjacentPagePair(const NativePage& first, const NativePage& second, bool& vertical,
                      const NativePage*& low, const NativePage*& high) {
    const __int128 deltaX = static_cast<__int128>(second.key.x) - first.key.x;
    const __int128 deltaZ = static_cast<__int128>(second.key.z) - first.key.z;
    if (deltaZ == 0 && (deltaX == 1 || deltaX == -1)) {
        vertical = true;
        low = deltaX > 0 ? &first : &second;
        high = deltaX > 0 ? &second : &first;
        return true;
    }
    if (deltaX == 0 && (deltaZ == 1 || deltaZ == -1)) {
        vertical = false;
        low = deltaZ > 0 ? &first : &second;
        high = deltaZ > 0 ? &second : &first;
        return true;
    }
    return false;
}

std::vector<LakePortalConnection> compatibleLakePortals(const NativePage& first,
                                                        const NativePage& second) {
    bool vertical = false;
    const NativePage* low = nullptr;
    const NativePage* high = nullptr;
    if (!adjacentPagePair(first, second, vertical, low, high)) return {};

    const PageEdge lowEdge = vertical ? PageEdge::EAST : PageEdge::SOUTH;
    const PageEdge highEdge = vertical ? PageEdge::WEST : PageEdge::NORTH;
    const auto& lowSamples = low->edgeSummaries[static_cast<size_t>(lowEdge)];
    const auto& highSamples = high->edgeSummaries[static_cast<size_t>(highEdge)];
    if (lowSamples.size() != CORE_EDGE_SAMPLES || highSamples.size() != CORE_EDGE_SAMPLES)
        throw std::runtime_error("native hydrology lake edge summary is incomplete");

    std::map<std::pair<LakeNode, LakeNode>, LakePortalConnection> connections;
    for (size_t along = 0; along < CORE_EDGE_SAMPLES; ++along) {
        const EdgeSummarySample& lowSample = lowSamples[along];
        const EdgeSummarySample& highSample = highSamples[along];
        if ((lowSample.flags & CELL_LAKE) == 0 || (highSample.flags & CELL_LAKE) == 0 ||
            lowSample.waterBodyId == NO_WATER_BODY || highSample.waterBodyId == NO_WATER_BODY ||
            std::abs(lowSample.rawElevation - highSample.rawElevation) > 1.0e-5F) {
            continue;
        }
        const DepressionSummary* lowDepression = depressionSummary(*low, lowSample.waterBodyId);
        const DepressionSummary* highDepression = depressionSummary(*high, highSample.waterBodyId);
        const uint8_t lowBit = 1U << static_cast<uint8_t>(lowEdge);
        const uint8_t highBit = 1U << static_cast<uint8_t>(highEdge);
        if (lowDepression == nullptr || highDepression == nullptr ||
            (lowDepression->edgeMask & lowBit) == 0 || (highDepression->edgeMask & highBit) == 0) {
            continue;
        }

        // Lower a conflicting pair to the lower locally proven spill stage.
        // Both pages must still contain water above the identical substrate at
        // that stage. This is the tiled Fill-Spill-Merge compatibility test.
        const float stage = std::min(lowDepression->localStage, highDepression->localStage);
        const float minimumWetStage =
            std::max(lowSample.rawElevation, highSample.rawElevation) + 1.0e-5F;
        if (!std::isfinite(stage) || !std::isfinite(minimumWetStage) || stage <= minimumWetStage) {
            continue;
        }
        const int64_t coordinate =
            checkedCoordinate(vertical ? low->key.z : low->key.x, static_cast<int>(along));
        const LakeNode lowNode{.page = low->key, .body = lowSample.waterBodyId};
        const LakeNode highNode{.page = high->key, .body = highSample.waterBodyId};
        const int64_t portalX = vertical ? high->originX : coordinate;
        const int64_t portalZ = vertical ? coordinate : high->originZ;
        auto [found, inserted] = connections.try_emplace(std::pair{lowNode, highNode},
                                                         LakePortalConnection{
                                                             .first = lowNode,
                                                             .second = highNode,
                                                             .minimumWetStage = minimumWetStage,
                                                             .compatibleStage = stage,
                                                             .x = portalX,
                                                             .z = portalZ,
                                                         });
        if (!inserted) {
            found->second.compatibleStage = std::min(found->second.compatibleStage, stage);
            if (minimumWetStage < found->second.minimumWetStage ||
                (minimumWetStage == found->second.minimumWetStage &&
                 std::pair{portalX, portalZ} < std::pair{found->second.x, found->second.z})) {
                found->second.minimumWetStage = minimumWetStage;
                found->second.x = portalX;
                found->second.z = portalZ;
            }
        }
    }
    std::vector<LakePortalConnection> result;
    result.reserve(connections.size());
    for (const auto& [nodes, connection] : connections) {
        static_cast<void>(nodes);
        result.push_back(connection);
    }
    return result;
}

std::optional<TiledLakeResolution>
resolveTiledLakeSpillHierarchy(const TiledLakeSpillHierarchy& hierarchy, LakeNode source) {
    if (!hierarchy.nodes().contains(source) || hierarchy.portals().empty()) return std::nullopt;

    // A connected portal can become dry after another tile contributes a
    // lower natural spill stage. Remove such a portal monotonically, then
    // recompute the source component. This fixed point cannot oscillate and
    // cannot preserve a route whose substrate is above the final flat stage.
    std::vector<uint8_t> activePortals(hierarchy.portals().size(), 1);
    std::set<LakeNode> connected;
    float stage = std::numeric_limits<float>::infinity();
    while (true) {
        connected.clear();
        connected.insert(source);
        bool expanded = true;
        while (expanded) {
            expanded = false;
            for (size_t index = 0; index < hierarchy.portals().size(); ++index) {
                if (activePortals[index] == 0) continue;
                const LakePortal& portal = hierarchy.portals()[index];
                const bool hasFirst = connected.contains(portal.first);
                const bool hasSecond = connected.contains(portal.second);
                if (hasFirst == hasSecond) continue;
                connected.insert(hasFirst ? portal.second : portal.first);
                expanded = true;
            }
        }

        stage = std::numeric_limits<float>::infinity();
        for (const LakeNode node : connected) {
            const DepressionSummary& summary = hierarchy.nodes().at(node);
            stage = std::min(stage, summary.localStage);
            if (summary.naturalOutlet != BasinOutlet::NONE)
                stage = std::min(stage, summary.naturalOutletStage);
        }
        if (!std::isfinite(stage))
            throw std::runtime_error("native hydrology spill hierarchy has no finite stage");

        bool removed = false;
        for (size_t index = 0; index < hierarchy.portals().size(); ++index) {
            if (activePortals[index] == 0) continue;
            const LakePortal& portal = hierarchy.portals()[index];
            if (!connected.contains(portal.first) || !connected.contains(portal.second)) continue;
            if (stage <= portal.minimumWetStage ||
                stage > portal.compatibleStage + ROUTING_ELEVATION_EPSILON_METERS) {
                activePortals[index] = 0;
                removed = true;
            }
        }
        if (!removed) break;
    }

    std::set<PageKey> pages;
    TiledLakeResolution resolution;
    std::optional<std::tuple<int64_t, int64_t, WaterBodyId>> canonicalRoot;
    std::optional<std::tuple<float, int64_t, int64_t, BasinOutlet>> outlet;
    for (const LakeNode node : connected) {
        const DepressionSummary& summary = hierarchy.nodes().at(node);
        pages.insert(node.page);
        resolution.areaSquareKilometers += summary.coreAreaSquareKilometers;
        resolution.volumeCubicMeters += summary.coreVolumeCubicMeters;
        resolution.runoffMmSquareKilometers += summary.coreRunoffMmSquareKilometers;
        const auto root = std::tuple{summary.localAnchorX, summary.localAnchorZ, node.body};
        if (!canonicalRoot || root < *canonicalRoot) {
            canonicalRoot = root;
            resolution.canonicalBody = node.body;
        }
        if (summary.naturalOutlet != BasinOutlet::NONE &&
            summary.naturalOutletStage <= stage + ROUTING_ELEVATION_EPSILON_METERS) {
            const auto candidate = std::tuple{summary.naturalOutletStage, summary.naturalOutletX,
                                              summary.naturalOutletZ, summary.naturalOutlet};
            if (!outlet || candidate < *outlet) outlet = candidate;
        }
    }
    resolution.pageCount = pages.size();
    resolution.stage = stage;
    if (resolution.canonicalBody == NO_WATER_BODY || resolution.pageCount < 2) return std::nullopt;
    if (outlet) {
        resolution.outlet = std::get<3>(*outlet);
        resolution.outletX = std::get<1>(*outlet);
        resolution.outletZ = std::get<2>(*outlet);
    } else {
        resolution.outlet = BasinOutlet::ENDORHEIC;
    }
    return resolution;
}

BasinSample applyTiledLakeResolution(BasinSample result, const TiledLakeResolution& resolution) {
    const double previousSurface = result.surfaceElevation;
    result.waterSurface = resolution.stage;
    // A lower reconciled stage can expose a locally shallow fringe. Keep the
    // proven connected route by lowering only its bed. Hydrology never raises
    // dry terrain, and the summary overlay never marks an edge wet unless both
    // persisted pages already did so at the same native sample.
    result.surfaceElevation = std::min(result.surfaceElevation, result.waterSurface - 0.125);
    result.erosionDepth = std::max(result.erosionDepth, previousSurface - result.surfaceElevation);
    result.lakeDepth = std::max(0.0, result.waterSurface - result.surfaceElevation);
    result.shoreWaterSurface = result.waterSurface;
    result.lakeSpillSurface = result.waterSurface;
    result.groundwaterHead = std::min(result.groundwaterHead, result.surfaceElevation);
    result.hydroperiod = 1.0;
    result.waterBodyId = resolution.canonicalBody;
    result.lakeAreaSquareKilometers = resolution.areaSquareKilometers;
    result.lakeVolumeCubicMeters = resolution.volumeCubicMeters;
    result.lakeRunoffMmSquareKilometers = resolution.runoffMmSquareKilometers;
    result.lakeOverflowMmSquareKilometers =
        resolution.outlet == BasinOutlet::ENDORHEIC ? 0.0 : resolution.runoffMmSquareKilometers;
    result.outlet = resolution.outlet;
    result.outletX = static_cast<double>(resolution.outletX);
    result.outletZ = static_cast<double>(resolution.outletZ);
    result.ocean = false;
    result.river = false;
    result.lake = true;
    result.wetland = false;
    result.delta = false;
    result.estuary = false;
    result.brackish = false;
    result.distributaryCount = 0;
    result.endorheic = resolution.outlet == BasinOutlet::ENDORHEIC;
    result.waterfall = false;
    result.waterfallAnchor = false;
    result.waterfallTop = 0.0;
    result.waterfallBottom = 0.0;
    result.waterfallWidth = 0.0;
    result.generatedFluidLevel = 0;
    result.transitionOwnerKind = WaterTransitionKind::NONE;
    result.transitionOwnerId = 0;
    result.valid = true;
    return result;
}

struct ResolvedOpenLakeRegion {
    PageKey minimumPage;
    PageKey maximumPage;
    int64_t originX = 0;
    int64_t originZ = 0;
    int width = 0;
    int height = 0;
    std::vector<float> signedShoreDistance;
    WaterBodyId canonicalBody = NO_WATER_BODY;
    float stageMeters = 0.0F;
    float stageWorld = 0.0F;
    double areaSquareKilometers = 0.0;
    double volumeCubicMeters = 0.0;
    double runoffMmSquareKilometers = 0.0;
    BasinOutlet outlet = BasinOutlet::NONE;
    int64_t outletX = 0;
    int64_t outletZ = 0;
    size_t pageCount = 0;

    size_t byteSize() const noexcept {
        return sizeof(*this) + signedShoreDistance.capacity() * sizeof(float);
    }
};

float reconstructRectangle(std::span<const float> field, int width, int height, double gridX,
                           double gridZ) {
    if (width < 2 || height < 2 ||
        field.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw std::invalid_argument("invalid native hydrology rectangle reconstruction");
    }
    const int x0 = std::clamp(static_cast<int>(std::floor(gridX)), 0, width - 2);
    const int z0 = std::clamp(static_cast<int>(std::floor(gridZ)), 0, height - 2);
    const double amountX = std::clamp(gridX - x0, 0.0, 1.0);
    const double amountZ = std::clamp(gridZ - z0, 0.0, 1.0);
    const auto cell = [width](int x, int z) {
        return static_cast<size_t>(z) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };
    const double north = std::lerp(field[cell(x0, z0)], field[cell(x0 + 1, z0)], amountX);
    const double south = std::lerp(field[cell(x0, z0 + 1)], field[cell(x0 + 1, z0 + 1)], amountX);
    return static_cast<float>(std::lerp(north, south, amountZ));
}

std::optional<BasinSample> sampleResolvedOpenLake(BasinSample result,
                                                  const ResolvedOpenLakeRegion& region, double x,
                                                  double z) {
    const double gridX =
        (x - static_cast<double>(region.originX)) / NATIVE_HYDROLOGY_RASTER_SPACING +
        WORLD_SAMPLE_NATIVE_OFFSET;
    const double gridZ =
        (z - static_cast<double>(region.originZ)) / NATIVE_HYDROLOGY_RASTER_SPACING +
        WORLD_SAMPLE_NATIVE_OFFSET;
    if (gridX < 0.0 || gridZ < 0.0 || gridX > static_cast<double>(region.width - 1) ||
        gridZ > static_cast<double>(region.height - 1)) {
        return std::nullopt;
    }
    const double signedDistance =
        reconstructRectangle(region.signedShoreDistance, region.width, region.height, gridX, gridZ);
    if (!std::isfinite(signedDistance)) return std::nullopt;
    result.lakeShoreDistance = signedDistance;
    result.shoreWaterSurface = region.stageWorld;
    if (signedDistance <= 0.0 || result.ocean) return result;

    const double previousSurface = result.surfaceElevation;
    result.waterSurface = region.stageWorld;
    result.surfaceElevation = std::min(result.surfaceElevation, result.waterSurface - 0.125);
    result.erosionDepth = std::max(result.erosionDepth, previousSurface - result.surfaceElevation);
    result.lakeDepth = std::max(0.0, result.waterSurface - result.surfaceElevation);
    result.shoreWaterSurface = result.waterSurface;
    result.lakeSpillSurface = result.waterSurface;
    result.groundwaterHead = std::min(result.groundwaterHead, result.surfaceElevation);
    result.hydroperiod = 1.0;
    result.waterBodyId = region.canonicalBody;
    result.lakeAreaSquareKilometers = region.areaSquareKilometers;
    result.lakeVolumeCubicMeters = region.volumeCubicMeters;
    result.lakeRunoffMmSquareKilometers = region.runoffMmSquareKilometers;
    result.lakeOverflowMmSquareKilometers =
        region.outlet == BasinOutlet::ENDORHEIC ? 0.0 : region.runoffMmSquareKilometers;
    result.outlet = region.outlet;
    result.outletX = static_cast<double>(region.outletX);
    result.outletZ = static_cast<double>(region.outletZ);
    result.ocean = false;
    result.river = false;
    result.lake = true;
    result.wetland = false;
    result.delta = false;
    result.estuary = false;
    result.brackish = false;
    result.distributaryCount = 0;
    result.endorheic = region.outlet == BasinOutlet::ENDORHEIC;
    result.waterfall = false;
    result.waterfallAnchor = false;
    result.waterfallTop = 0.0;
    result.waterfallBottom = 0.0;
    result.waterfallWidth = 0.0;
    result.channelWidth = 0.0;
    result.channelDepth = 0.0;
    result.streamOrder = 0;
    result.generatedFluidLevel = 0;
    result.transitionOwnerKind = WaterTransitionKind::NONE;
    result.transitionOwnerId = 0;
    result.valid = true;
    return result;
}

struct OpenLakeRasterSolve {
    std::shared_ptr<const ResolvedOpenLakeRegion> region;
    bool expandWest = false;
    bool expandEast = false;
    bool expandNorth = false;
    bool expandSouth = false;
};

struct WetlandNode {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const WetlandNode&) const = default;
};

struct WetlandNodeHash {
    size_t operator()(WetlandNode node) const noexcept {
        return static_cast<size_t>(mix64(static_cast<uint64_t>(node.x)) ^
                                   std::rotl(mix64(static_cast<uint64_t>(node.z)), 31));
    }
};

struct WetlandResolution {
    WaterBodyId body = NO_WATER_BODY;
    float stage = 0.0F;

    auto operator<=>(const WetlandResolution&) const = default;
};

struct EstuaryResolution {
    WaterBodyId oceanBody = NO_WATER_BODY;
    uint16_t distanceToOcean = 0;

    auto operator<=>(const EstuaryResolution&) const = default;
};

BasinSample applyWetlandResolution(BasinSample result, const WetlandResolution& resolution) {
    if (resolution.body == NO_WATER_BODY || !std::isfinite(resolution.stage))
        throw std::invalid_argument("invalid connected wetland resolution");
    const double undisturbed = result.surfaceElevation;
    result.waterSurface = resolution.stage;
    result.surfaceElevation =
        std::min(result.surfaceElevation, result.waterSurface - WETLAND_BED_DEPTH_BLOCKS);
    result.erosionDepth = std::max(result.erosionDepth, undisturbed - result.surfaceElevation);
    result.groundwaterHead = std::max(result.groundwaterHead, result.waterSurface);
    result.hydroperiod =
        wetlandHydroperiod(undisturbed, result.groundwaterHead, result.precipitationSeasonality);
    result.waterBodyId = resolution.body;
    result.ocean = false;
    result.river = false;
    result.lake = false;
    result.wetland = true;
    result.delta = false;
    result.estuary = false;
    result.brackish = false;
    result.waterfall = false;
    result.waterfallAnchor = false;
    result.waterfallTop = 0.0;
    result.waterfallBottom = 0.0;
    result.waterfallWidth = 0.0;
    result.distributaryCount = 0;
    result.generatedFluidLevel = 0;
    result.transitionOwnerKind = WaterTransitionKind::NONE;
    result.transitionOwnerId = 0;
    result.valid = true;
    return result;
}

BasinSample applyEstuaryResolution(BasinSample result, const EstuaryResolution& resolution) {
    if (!result.river || resolution.oceanBody == NO_WATER_BODY ||
        resolution.distanceToOcean > ESTUARY_MAXIMUM_BACKWATER_NATIVE_CELLS)
        throw std::invalid_argument("invalid estuary resolution");
    const double priorSurface = result.surfaceElevation;
    result.waterSurface = std::max(result.waterSurface, static_cast<double>(SEA_LEVEL));
    result.surfaceElevation =
        std::min(result.surfaceElevation,
                 result.waterSurface - std::max(static_cast<double>(RIVER_MINIMUM_BED_DEPTH_BLOCKS),
                                                result.channelDepth));
    result.erosionDepth = std::max(result.erosionDepth, priorSurface - result.surfaceElevation);
    result.groundwaterHead = std::max(result.groundwaterHead, result.waterSurface);
    result.channelDepth = std::max(static_cast<double>(RIVER_MINIMUM_BED_DEPTH_BLOCKS),
                                   result.waterSurface - result.surfaceElevation);
    result.estuary = true;
    result.brackish = true;
    result.valid = true;
    return result;
}

template <typename PageResolver>
OpenLakeRasterSolve solveOpenLakeRectangle(PageKey minimumPage, PageKey maximumPage,
                                           LakeNode source, PageResolver& resolvePage) {
    const __int128 pagesWide128 = static_cast<__int128>(maximumPage.x) - minimumPage.x + 1;
    const __int128 pagesHigh128 = static_cast<__int128>(maximumPage.z) - minimumPage.z + 1;
    if (pagesWide128 <= 0 || pagesHigh128 <= 0 ||
        pagesWide128 * pagesHigh128 > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES) {
        throw std::runtime_error("native hydrology open-depression page bound exceeded");
    }
    const int pagesWide = static_cast<int>(pagesWide128);
    const int pagesHigh = static_cast<int>(pagesHigh128);
    const int width = pagesWide * INTERIOR_INTERVALS + 1;
    const int height = pagesHigh * INTERIOR_INTERVALS + 1;
    const size_t cells = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int64_t originX = checkedCoordinate(minimumPage.x, 0);
    const int64_t originZ = checkedCoordinate(minimumPage.z, 0);
    const auto cell = [width](int x, int z) {
        return static_cast<size_t>(z) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    std::map<PageKey, std::shared_ptr<const NativePage>> pages;
    for (int64_t pageZ = minimumPage.z;; ++pageZ) {
        for (int64_t pageX = minimumPage.x;; ++pageX) {
            const PageKey key{.x = pageX, .z = pageZ};
            pages.emplace(key, resolvePage(key));
            if (pageX == maximumPage.x) break;
        }
        if (pageZ == maximumPage.z) break;
    }

    std::vector<float> elevationMeters(cells);
    std::vector<float> localRunoff(cells);
    for (const auto& [key, page] : pages) {
        const int destinationX = static_cast<int>(key.x - minimumPage.x) * INTERIOR_INTERVALS;
        const int destinationZ = static_cast<int>(key.z - minimumPage.z) * INTERIOR_INTERVALS;
        const int copyWidth = key.x == maximumPage.x ? INTERIOR_INTERVALS + 1 : INTERIOR_INTERVALS;
        const int copyHeight = key.z == maximumPage.z ? INTERIOR_INTERVALS + 1 : INTERIOR_INTERVALS;
        for (int z = 0; z < copyHeight; ++z) {
            for (int x = 0; x < copyWidth; ++x) {
                const size_t sourceCell =
                    static_cast<size_t>(indexOf(CORE_BEGIN + x, CORE_BEGIN + z));
                const size_t destination = cell(destinationX + x, destinationZ + z);
                elevationMeters[destination] = page->routingElevationMeters[sourceCell];
                localRunoff[destination] = page->localRunoffMmSquareKilometers[sourceCell];
            }
        }
    }

    std::vector<float> filledMeters(cells, std::numeric_limits<float>::infinity());
    std::vector<uint8_t> visited(cells, 0);
    std::priority_queue<FloodNode, std::vector<FloodNode>, FloodNodeGreater> frontier;
    const auto addTerminal = [&](size_t index) {
        if (visited[index] != 0) return;
        if (index > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("native hydrology open-depression raster is too large");
        visited[index] = 1;
        filledMeters[index] = elevationMeters[index];
        frontier.push({.elevation = elevationMeters[index], .index = static_cast<int32_t>(index)});
    };
    for (int z = 0; z < height; ++z) {
        addTerminal(cell(0, z));
        addTerminal(cell(width - 1, z));
    }
    for (int x = 1; x < width - 1; ++x) {
        addTerminal(cell(x, 0));
        addTerminal(cell(x, height - 1));
    }
    for (size_t index = 0; index < cells; ++index) {
        if (elevationMeters[index] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS) addTerminal(index);
    }
    while (!frontier.empty()) {
        const FloodNode current = frontier.top();
        frontier.pop();
        const int x = current.index % width;
        const int z = current.index / width;
        for (size_t direction = 0; direction < DX.size(); ++direction) {
            const int neighborX = x + DX[direction];
            const int neighborZ = z + DZ[direction];
            if (neighborX < 0 || neighborX >= width || neighborZ < 0 || neighborZ >= height)
                continue;
            const size_t neighbor = cell(neighborX, neighborZ);
            if (visited[neighbor] != 0) continue;
            visited[neighbor] = 1;
            filledMeters[neighbor] = std::max(elevationMeters[neighbor], current.elevation);
            frontier.push(
                {.elevation = filledMeters[neighbor], .index = static_cast<int32_t>(neighbor)});
        }
    }

    const auto sourcePageFound = pages.find(source.page);
    if (sourcePageFound == pages.end())
        throw std::runtime_error("native hydrology open-depression source page is missing");
    const NativePage& sourcePage = *sourcePageFound->second;
    std::optional<size_t> sourceCell;
    for (int z = 0; z < RASTER_EDGE; ++z) {
        for (int x = 0; x < RASTER_EDGE; ++x) {
            const size_t localCell = static_cast<size_t>(indexOf(x, z));
            if ((sourcePage.flags[localCell] & CELL_LAKE) == 0 ||
                sourcePage.waterBodyIds[localCell] != source.body) {
                continue;
            }
            const int64_t worldX = checkedCoordinate(source.page.x, x - RASTER_APRON);
            const int64_t worldZ = checkedCoordinate(source.page.z, z - RASTER_APRON);
            const __int128 gridX128 =
                (static_cast<__int128>(worldX) - originX) / NATIVE_HYDROLOGY_RASTER_SPACING;
            const __int128 gridZ128 =
                (static_cast<__int128>(worldZ) - originZ) / NATIVE_HYDROLOGY_RASTER_SPACING;
            if (gridX128 < 0 || gridX128 >= width || gridZ128 < 0 || gridZ128 >= height) continue;
            const size_t candidate = cell(static_cast<int>(gridX128), static_cast<int>(gridZ128));
            if (!sourceCell ||
                elevationMeters[candidate] <
                    elevationMeters[*sourceCell] - ROUTING_ELEVATION_EPSILON_METERS ||
                (std::abs(elevationMeters[candidate] - elevationMeters[*sourceCell]) <=
                     ROUTING_ELEVATION_EPSILON_METERS &&
                 candidate < *sourceCell)) {
                sourceCell = candidate;
            }
        }
    }
    if (!sourceCell ||
        filledMeters[*sourceCell] - elevationMeters[*sourceCell] <= LAKE_DEPTH_THRESHOLD_METERS) {
        return {};
    }

    const float stageMeters = filledMeters[*sourceCell];
    std::vector<uint8_t> component(cells, 0);
    std::vector<int32_t> queue;
    queue.reserve(cells / 16U);
    component[*sourceCell] = 1;
    queue.push_back(static_cast<int32_t>(*sourceCell));
    for (size_t head = 0; head < queue.size(); ++head) {
        const int current = queue[head];
        const int x = current % width;
        const int z = current / width;
        constexpr std::array<std::pair<int, int>, 4> CARDINAL{{
            {-1, 0},
            {1, 0},
            {0, -1},
            {0, 1},
        }};
        for (const auto [deltaX, deltaZ] : CARDINAL) {
            const int neighborX = x + deltaX;
            const int neighborZ = z + deltaZ;
            if (neighborX < 0 || neighborX >= width || neighborZ < 0 || neighborZ >= height)
                continue;
            const size_t neighbor = cell(neighborX, neighborZ);
            if (component[neighbor] != 0 ||
                elevationMeters[neighbor] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS ||
                filledMeters[neighbor] - elevationMeters[neighbor] <= LAKE_DEPTH_THRESHOLD_METERS ||
                std::abs(filledMeters[neighbor] - stageMeters) >
                    LAKE_COMPONENT_STAGE_EPSILON_METERS) {
                continue;
            }
            component[neighbor] = 1;
            queue.push_back(static_cast<int32_t>(neighbor));
        }
    }

    OpenLakeRasterSolve solve;
    const auto boundaryCanContinue = [&](int x, int z) {
        const float boundary = elevationMeters[cell(x, z)];
        return boundary >= NATIVE_HYDROLOGY_SEA_LEVEL_METERS &&
               boundary <= stageMeters + LAKE_COMPONENT_STAGE_EPSILON_METERS;
    };
    for (const int32_t index : queue) {
        const int x = index % width;
        const int z = index / width;
        if (x == 1 && boundaryCanContinue(0, z)) solve.expandWest = true;
        if (x == width - 2 && boundaryCanContinue(width - 1, z)) solve.expandEast = true;
        if (z == 1 && boundaryCanContinue(x, 0)) solve.expandNorth = true;
        if (z == height - 2 && boundaryCanContinue(x, height - 1)) solve.expandSouth = true;
    }
    if (solve.expandWest || solve.expandEast || solve.expandNorth || solve.expandSouth)
        return solve;

    const std::vector<float> distanceToDry =
        squaredDistanceTransformRectangle(component, 0, width, height);
    const std::vector<float> distanceToLake =
        squaredDistanceTransformRectangle(component, 1, width, height);
    // Refine the EDT on a one-block subgrid around the complete component.
    // Bilinear mask reconstruction is the same continuous field sampled by
    // exact columns, so its half-cell contour removes native four-block
    // staircase bias without changing the canonical native wet mask.
    constexpr int SHORE_REFINEMENT = 2 * NATIVE_HYDROLOGY_RASTER_SPACING;
    constexpr int SHORE_HALO_NATIVE_CELLS = 64;
    constexpr size_t MAXIMUM_REFINED_SHORE_CELLS = 32U * 1024U * 1024U;
    int componentMinimumX = width - 1;
    int componentMaximumX = 0;
    int componentMinimumZ = height - 1;
    int componentMaximumZ = 0;
    for (const int32_t index : queue) {
        const int x = index % width;
        const int z = index / width;
        componentMinimumX = std::min(componentMinimumX, x);
        componentMaximumX = std::max(componentMaximumX, x);
        componentMinimumZ = std::min(componentMinimumZ, z);
        componentMaximumZ = std::max(componentMaximumZ, z);
    }
    const int shoreMinimumX = std::max(0, componentMinimumX - SHORE_HALO_NATIVE_CELLS);
    const int shoreMaximumX = std::min(width - 1, componentMaximumX + SHORE_HALO_NATIVE_CELLS);
    const int shoreMinimumZ = std::max(0, componentMinimumZ - SHORE_HALO_NATIVE_CELLS);
    const int shoreMaximumZ = std::min(height - 1, componentMaximumZ + SHORE_HALO_NATIVE_CELLS);
    const int refinedWidth = (shoreMaximumX - shoreMinimumX) * SHORE_REFINEMENT + 1;
    const int refinedHeight = (shoreMaximumZ - shoreMinimumZ) * SHORE_REFINEMENT + 1;
    const size_t refinedCells =
        static_cast<size_t>(refinedWidth) * static_cast<size_t>(refinedHeight);
    std::vector<float> refinedDistanceToDry;
    std::vector<float> refinedDistanceToLake;
    if (refinedCells <= MAXIMUM_REFINED_SHORE_CELLS) {
        std::vector<uint8_t> refinedMask(refinedCells, 0);
        const auto refinedCell = [refinedWidth](int x, int z) {
            return static_cast<size_t>(z) * static_cast<size_t>(refinedWidth) +
                   static_cast<size_t>(x);
        };
        for (int refinedZ = 0; refinedZ < refinedHeight; ++refinedZ) {
            const int nativeZ = shoreMinimumZ + refinedZ / SHORE_REFINEMENT;
            const int nextZ = std::min(nativeZ + 1, height - 1);
            const double amountZ =
                static_cast<double>(refinedZ % SHORE_REFINEMENT) / SHORE_REFINEMENT;
            for (int refinedX = 0; refinedX < refinedWidth; ++refinedX) {
                const int nativeX = shoreMinimumX + refinedX / SHORE_REFINEMENT;
                const int nextX = std::min(nativeX + 1, width - 1);
                const double amountX =
                    static_cast<double>(refinedX % SHORE_REFINEMENT) / SHORE_REFINEMENT;
                const std::array<size_t, 4> corners{cell(nativeX, nativeZ), cell(nextX, nativeZ),
                                                    cell(nativeX, nextZ), cell(nextX, nextZ)};
                const bool componentSupport = std::ranges::any_of(
                    corners, [&component](size_t index) { return component[index] != 0; });
                const double north =
                    std::lerp(static_cast<double>(elevationMeters[corners[0]]),
                              static_cast<double>(elevationMeters[corners[1]]), amountX);
                const double south =
                    std::lerp(static_cast<double>(elevationMeters[corners[2]]),
                              static_cast<double>(elevationMeters[corners[3]]), amountX);
                const double elevation = std::lerp(north, south, amountZ);
                refinedMask[refinedCell(refinedX, refinedZ)] =
                    componentSupport && elevation < stageMeters - LAKE_DEPTH_THRESHOLD_METERS ? 1U
                                                                                              : 0U;
            }
        }
        refinedDistanceToDry =
            squaredDistanceTransformRectangle(refinedMask, 0, refinedWidth, refinedHeight);
        refinedDistanceToLake =
            squaredDistanceTransformRectangle(refinedMask, 1, refinedWidth, refinedHeight);
    }

    struct ShoreSegment {
        double firstX = 0.0;
        double firstZ = 0.0;
        double secondX = 0.0;
        double secondZ = 0.0;
    };
    struct ShorePoint {
        double x = 0.0;
        double z = 0.0;
        int edge = 0;
    };
    std::vector<ShoreSegment> shoreSegments;
    const double shoreLevelMeters = stageMeters - LAKE_DEPTH_THRESHOLD_METERS;
    const auto signedContourValue = [&](int x, int z) {
        const size_t index = cell(x, z);
        const double magnitude =
            std::max(1.0e-6, std::abs(shoreLevelMeters - elevationMeters[index]));
        return component[index] != 0 ? magnitude : -magnitude;
    };
    const auto intersection = [](double firstX, double firstZ, double firstValue, double secondX,
                                 double secondZ, double secondValue, int edge) {
        const double amount = std::clamp(firstValue / (firstValue - secondValue), 0.0, 1.0);
        return ShorePoint{.x = std::lerp(firstX, secondX, amount),
                          .z = std::lerp(firstZ, secondZ, amount),
                          .edge = edge};
    };
    for (int z = shoreMinimumZ; z < shoreMaximumZ; ++z) {
        for (int x = shoreMinimumX; x < shoreMaximumX; ++x) {
            const std::array<double, 4> values{
                signedContourValue(x, z), signedContourValue(x + 1, z),
                signedContourValue(x + 1, z + 1), signedContourValue(x, z + 1)};
            std::array<ShorePoint, 4> crossings{};
            size_t crossingCount = 0;
            const auto addCrossing = [&](int first, int second, double firstX, double firstZ,
                                         double secondX, double secondZ, int edge) {
                if ((values[static_cast<size_t>(first)] > 0.0) ==
                    (values[static_cast<size_t>(second)] > 0.0)) {
                    return;
                }
                crossings[crossingCount++] =
                    intersection(firstX, firstZ, values[static_cast<size_t>(first)], secondX,
                                 secondZ, values[static_cast<size_t>(second)], edge);
            };
            addCrossing(0, 1, x, z, x + 1.0, z, 0);
            addCrossing(1, 2, x + 1.0, z, x + 1.0, z + 1.0, 1);
            addCrossing(2, 3, x + 1.0, z + 1.0, x, z + 1.0, 2);
            addCrossing(3, 0, x, z + 1.0, x, z, 3);
            const auto appendSegment = [&](const ShorePoint& first, const ShorePoint& second) {
                shoreSegments.push_back({.firstX = first.x,
                                         .firstZ = first.z,
                                         .secondX = second.x,
                                         .secondZ = second.z});
            };
            if (crossingCount == 2) {
                appendSegment(crossings[0], crossings[1]);
            } else if (crossingCount == 4) {
                const bool northWestInside = values[0] > 0.0;
                const bool centerInside = std::accumulate(values.begin(), values.end(), 0.0) > 0.0;
                if (northWestInside == centerInside) {
                    appendSegment(crossings[0], crossings[1]);
                    appendSegment(crossings[2], crossings[3]);
                } else {
                    appendSegment(crossings[0], crossings[3]);
                    appendSegment(crossings[1], crossings[2]);
                }
            }
        }
    }
    const size_t shoreNativeCells = static_cast<size_t>(shoreMaximumX - shoreMinimumX + 1) *
                                    static_cast<size_t>(shoreMaximumZ - shoreMinimumZ + 1);
    constexpr size_t MAXIMUM_EXACT_SHORE_DISTANCE_WORK = 200U * 1024U * 1024U;
    const bool exactShoreDistance =
        !shoreSegments.empty() && shoreSegments.size() <= MAXIMUM_EXACT_SHORE_DISTANCE_WORK /
                                                              std::max<size_t>(1, shoreNativeCells);
    const auto exactDistanceToShore = [&](int x, int z) {
        double bestSquared = std::numeric_limits<double>::infinity();
        for (const ShoreSegment& segment : shoreSegments) {
            const double deltaX = segment.secondX - segment.firstX;
            const double deltaZ = segment.secondZ - segment.firstZ;
            const double lengthSquared = deltaX * deltaX + deltaZ * deltaZ;
            const double amount =
                lengthSquared > 1.0e-12
                    ? std::clamp(((x - segment.firstX) * deltaX + (z - segment.firstZ) * deltaZ) /
                                     lengthSquared,
                                 0.0, 1.0)
                    : 0.0;
            const double nearestX = segment.firstX + deltaX * amount;
            const double nearestZ = segment.firstZ + deltaZ * amount;
            const double distanceX = x - nearestX;
            const double distanceZ = z - nearestZ;
            bestSquared = std::min(bestSquared, distanceX * distanceX + distanceZ * distanceZ);
        }
        return static_cast<float>(std::sqrt(bestSquared) * NATIVE_HYDROLOGY_RASTER_SPACING);
    };
    auto region = std::make_shared<ResolvedOpenLakeRegion>();
    region->minimumPage = minimumPage;
    region->maximumPage = maximumPage;
    region->originX = originX;
    region->originZ = originZ;
    region->width = width;
    region->height = height;
    region->signedShoreDistance.resize(cells);
    region->stageMeters = stageMeters;
    region->stageWorld = worldHeightFromNativeMeters(stageMeters);
    std::optional<std::tuple<int64_t, int64_t, WaterBodyId>> canonicalRoot;
    std::optional<std::tuple<float, int64_t, int64_t, BasinOutlet>> outlet;
    std::set<PageKey> componentPages;

    for (size_t index = 0; index < cells; ++index) {
        const bool lake = component[index] != 0;
        float squared = lake ? distanceToDry[index] : distanceToLake[index];
        float cellSpacing = NATIVE_HYDROLOGY_RASTER_SPACING;
        const int x = static_cast<int>(index % static_cast<size_t>(width));
        const int z = static_cast<int>(index / static_cast<size_t>(width));
        if (!refinedDistanceToDry.empty() && x >= shoreMinimumX && x <= shoreMaximumX &&
            z >= shoreMinimumZ && z <= shoreMaximumZ) {
            const int refinedX = (x - shoreMinimumX) * SHORE_REFINEMENT;
            const int refinedZ = (z - shoreMinimumZ) * SHORE_REFINEMENT;
            const size_t refinedIndex =
                static_cast<size_t>(refinedZ) * static_cast<size_t>(refinedWidth) +
                static_cast<size_t>(refinedX);
            squared =
                lake ? refinedDistanceToDry[refinedIndex] : refinedDistanceToLake[refinedIndex];
            cellSpacing = static_cast<float>(NATIVE_HYDROLOGY_RASTER_SPACING) / SHORE_REFINEMENT;
        }
        float distance =
            std::isfinite(squared) ? (std::sqrt(squared) - 0.5F) * cellSpacing : 1.0e9F;
        if (exactShoreDistance && x >= shoreMinimumX && x <= shoreMaximumX && z >= shoreMinimumZ &&
            z <= shoreMaximumZ) {
            distance = exactDistanceToShore(x, z);
        }
        region->signedShoreDistance[index] = lake ? distance : -distance;
        if (!lake) continue;
        const int64_t pageX = minimumPage.x + x / INTERIOR_INTERVALS;
        const int64_t pageZ = minimumPage.z + z / INTERIOR_INTERVALS;
        componentPages.insert(
            {.x = std::min(pageX, maximumPage.x), .z = std::min(pageZ, maximumPage.z)});
        if (x + 1 < width && z + 1 < height) {
            region->areaSquareKilometers += CELL_AREA_SQUARE_KILOMETERS;
            region->volumeCubicMeters += std::max(0.0F, stageMeters - elevationMeters[index]) *
                                         NATIVE_HYDROLOGY_CELL_EDGE_METERS *
                                         NATIVE_HYDROLOGY_CELL_EDGE_METERS;
            region->runoffMmSquareKilometers += localRunoff[index];
        }
    }
    region->pageCount = componentPages.size();

    for (const auto& [key, page] : pages) {
        for (const DepressionSummary& summary : page->depressionSummaries) {
            bool joined = false;
            for (int z = 0; z < RASTER_EDGE && !joined; ++z) {
                for (int x = 0; x < RASTER_EDGE; ++x) {
                    const size_t localCell = static_cast<size_t>(indexOf(x, z));
                    if ((page->flags[localCell] & CELL_LAKE) == 0 ||
                        page->waterBodyIds[localCell] != summary.waterBodyId) {
                        continue;
                    }
                    const int64_t worldX = checkedCoordinate(key.x, x - RASTER_APRON);
                    const int64_t worldZ = checkedCoordinate(key.z, z - RASTER_APRON);
                    const __int128 gridX128 =
                        (static_cast<__int128>(worldX) - originX) / NATIVE_HYDROLOGY_RASTER_SPACING;
                    const __int128 gridZ128 =
                        (static_cast<__int128>(worldZ) - originZ) / NATIVE_HYDROLOGY_RASTER_SPACING;
                    if (gridX128 < 0 || gridX128 >= width || gridZ128 < 0 || gridZ128 >= height) {
                        continue;
                    }
                    joined =
                        component[cell(static_cast<int>(gridX128), static_cast<int>(gridZ128))] !=
                        0;
                    if (joined) break;
                }
            }
            if (!joined) continue;
            const auto root =
                std::tuple{summary.localAnchorX, summary.localAnchorZ, summary.waterBodyId};
            if (!canonicalRoot || root < *canonicalRoot) canonicalRoot = root;
            if (summary.naturalOutlet == BasinOutlet::NONE ||
                summary.naturalOutlet == BasinOutlet::ENDORHEIC ||
                std::abs(summary.naturalOutletStage - region->stageWorld) > 1.0e-5F) {
                continue;
            }
            const __int128 outletGridX = (static_cast<__int128>(summary.naturalOutletX) - originX) /
                                         NATIVE_HYDROLOGY_RASTER_SPACING;
            const __int128 outletGridZ = (static_cast<__int128>(summary.naturalOutletZ) - originZ) /
                                         NATIVE_HYDROLOGY_RASTER_SPACING;
            if (outletGridX < 0 || outletGridX >= width || outletGridZ < 0 ||
                outletGridZ >= height ||
                component[cell(static_cast<int>(outletGridX), static_cast<int>(outletGridZ))] !=
                    0) {
                continue;
            }
            const auto candidate = std::tuple{summary.naturalOutletStage, summary.naturalOutletX,
                                              summary.naturalOutletZ, summary.naturalOutlet};
            if (!outlet || candidate < *outlet) outlet = candidate;
        }
    }
    region->canonicalBody = canonicalRoot ? std::get<2>(*canonicalRoot) : source.body;

    for (const int32_t index : queue) {
        const int x = index % width;
        const int z = index / width;
        for (size_t direction = 0; direction < DX.size(); ++direction) {
            const int neighborX = x + DX[direction];
            const int neighborZ = z + DZ[direction];
            if (neighborX < 0 || neighborX >= width || neighborZ < 0 || neighborZ >= height)
                continue;
            const size_t neighbor = cell(neighborX, neighborZ);
            if (component[neighbor] != 0) continue;
            const float saddle =
                std::max(elevationMeters[static_cast<size_t>(index)], elevationMeters[neighbor]);
            if (saddle < stageMeters - LAKE_DEPTH_THRESHOLD_METERS ||
                saddle > stageMeters + LAKE_COMPONENT_STAGE_EPSILON_METERS) {
                continue;
            }
            const int64_t outletX =
                originX + static_cast<int64_t>(neighborX) * NATIVE_HYDROLOGY_RASTER_SPACING;
            const int64_t outletZ =
                originZ + static_cast<int64_t>(neighborZ) * NATIVE_HYDROLOGY_RASTER_SPACING;
            const BasinOutlet kind = elevationMeters[neighbor] < NATIVE_HYDROLOGY_SEA_LEVEL_METERS
                                         ? BasinOutlet::OCEAN
                                         : BasinOutlet::SHARED_PORTAL;
            // A geometric saddle is not by itself proof that the equilibrium
            // lake overflows. Only an ocean contact or a persisted routed
            // outlet may turn a closed basin into an open one.
            if (kind == BasinOutlet::SHARED_PORTAL) continue;
            const auto candidate =
                std::tuple{worldHeightFromNativeMeters(saddle), outletX, outletZ, kind};
            if (!outlet || candidate < *outlet) outlet = candidate;
        }
    }
    if (outlet) {
        region->outlet = std::get<3>(*outlet);
        region->outletX = std::get<1>(*outlet);
        region->outletZ = std::get<2>(*outlet);
    } else {
        region->outlet = BasinOutlet::ENDORHEIC;
    }
    solve.region = std::move(region);
    return solve;
}

template <typename PageResolver>
std::shared_ptr<const ResolvedOpenLakeRegion>
buildResolvedOpenLakeRegion(const std::shared_ptr<const NativePage>& sourcePage,
                            WaterBodyId sourceBody, PageResolver& resolvePage) {
    const DepressionSummary* summary = depressionSummary(*sourcePage, sourceBody);
    if (summary == nullptr || summary->edgeMask == 0) return nullptr;
    PageKey minimumPage = sourcePage->key;
    PageKey maximumPage = sourcePage->key;
    const auto includeNeighbor = [&](PageEdge edge, int64_t deltaX, int64_t deltaZ) {
        if ((summary->edgeMask & (1U << static_cast<uint8_t>(edge))) == 0) return;
        const __int128 x = static_cast<__int128>(sourcePage->key.x) + deltaX;
        const __int128 z = static_cast<__int128>(sourcePage->key.z) + deltaZ;
        if (x < std::numeric_limits<int64_t>::min() || x > std::numeric_limits<int64_t>::max() ||
            z < std::numeric_limits<int64_t>::min() || z > std::numeric_limits<int64_t>::max()) {
            throw std::runtime_error("native hydrology open-depression neighbor is out of range");
        }
        minimumPage.x = std::min(minimumPage.x, static_cast<int64_t>(x));
        minimumPage.z = std::min(minimumPage.z, static_cast<int64_t>(z));
        maximumPage.x = std::max(maximumPage.x, static_cast<int64_t>(x));
        maximumPage.z = std::max(maximumPage.z, static_cast<int64_t>(z));
    };
    includeNeighbor(PageEdge::WEST, -1, 0);
    includeNeighbor(PageEdge::EAST, 1, 0);
    includeNeighbor(PageEdge::NORTH, 0, -1);
    includeNeighbor(PageEdge::SOUTH, 0, 1);

    const LakeNode source{.page = sourcePage->key, .body = sourceBody};
    for (;;) {
        OpenLakeRasterSolve solve =
            solveOpenLakeRectangle(minimumPage, maximumPage, source, resolvePage);
        if (solve.region) return solve.region;
        if (!solve.expandWest && !solve.expandEast && !solve.expandNorth && !solve.expandSouth)
            return nullptr;
        const auto expand = [](int64_t value, int direction) {
            const __int128 result = static_cast<__int128>(value) + direction;
            if (result < std::numeric_limits<int64_t>::min() ||
                result > std::numeric_limits<int64_t>::max()) {
                throw std::runtime_error("native hydrology open-depression bound is out of range");
            }
            return static_cast<int64_t>(result);
        };
        if (solve.expandWest) minimumPage.x = expand(minimumPage.x, -1);
        if (solve.expandEast) maximumPage.x = expand(maximumPage.x, 1);
        if (solve.expandNorth) minimumPage.z = expand(minimumPage.z, -1);
        if (solve.expandSouth) maximumPage.z = expand(maximumPage.z, 1);
        const __int128 pageCount = (static_cast<__int128>(maximumPage.x) - minimumPage.x + 1) *
                                   (static_cast<__int128>(maximumPage.z) - minimumPage.z + 1);
        if (pageCount > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES) {
            throw std::runtime_error(
                "native hydrology open-depression closure exceeded its deterministic bound");
        }
    }
}

const EdgeSummarySample* boundarySummary(const NativePage& page, bool vertical, int64_t boundary,
                                         double along) {
    PageEdge edge = PageEdge::WEST;
    double edgeOrigin = 0.0;
    if (vertical) {
        if (boundary == page.originX)
            edge = PageEdge::WEST;
        else if (boundary == page.originX + NATIVE_HYDROLOGY_PAGE_EDGE)
            edge = PageEdge::EAST;
        else
            return nullptr;
        edgeOrigin = static_cast<double>(page.originZ);
    } else {
        if (boundary == page.originZ)
            edge = PageEdge::NORTH;
        else if (boundary == page.originZ + NATIVE_HYDROLOGY_PAGE_EDGE)
            edge = PageEdge::SOUTH;
        else
            return nullptr;
        edgeOrigin = static_cast<double>(page.originX);
    }
    const int index = std::clamp(
        static_cast<int>(std::llround((along - edgeOrigin) / NATIVE_HYDROLOGY_RASTER_SPACING)), 0,
        INTERIOR_INTERVALS);
    return &page.edgeSummaries[static_cast<size_t>(edge)][static_cast<size_t>(index)];
}

WaterBodyId canonicalRiverPortId(uint64_t seed, double x, double z,
                                 std::optional<int64_t> verticalBoundary,
                                 std::optional<int64_t> horizontalBoundary,
                                 const BasinSample& routed) {
    bool vertical = verticalBoundary.has_value();
    if (verticalBoundary && horizontalBoundary)
        vertical = std::abs(routed.flowX) >= std::abs(routed.flowZ);
    if (vertical) {
        const int64_t boundary = *verticalBoundary;
        double crossing = z;
        if (std::abs(routed.flowX) > 1.0e-6)
            crossing += (static_cast<double>(boundary) - x) * routed.flowZ / routed.flowX;
        const int64_t along =
            static_cast<int64_t>(std::llround(crossing / NATIVE_HYDROLOGY_RASTER_SPACING)) *
            NATIVE_HYDROLOGY_RASTER_SPACING;
        return boundedRiverWaterBodyId(seed, boundary, along, true);
    }
    const int64_t boundary = *horizontalBoundary;
    double crossing = x;
    if (std::abs(routed.flowZ) > 1.0e-6)
        crossing += (static_cast<double>(boundary) - z) * routed.flowX / routed.flowZ;
    const int64_t along =
        static_cast<int64_t>(std::llround(crossing / NATIVE_HYDROLOGY_RASTER_SPACING)) *
        NATIVE_HYDROLOGY_RASTER_SPACING;
    return boundedRiverWaterBodyId(seed, boundary, along, false);
}

// A channel's wet ribbon can be narrower than its ecological bank. At a
// native page edge, the half-open owner may therefore be dry while its
// overlapping neighbor still resolves the same nearby routed segment. Blend
// the dry-bank projection over the existing handoff band instead of allowing
// the nearest locally routed segment to switch abruptly at ownership. This
// does not transfer a water stage, body ID, or terrain elevation across the
// handoff.
const HandoffCandidate* nearestHandoffChannel(std::span<const HandoffCandidate> candidates) {
    const HandoffCandidate* selected = nullptr;
    double selectedNormalizedDistance = std::numeric_limits<double>::infinity();
    for (const HandoffCandidate& candidate : candidates) {
        if (!candidate.containsQuery) continue;
        const BasinSample& sample = candidate.sample;
        if (sample.channelWidth <= 0.0 || !std::isfinite(sample.channelDistance)) continue;
        const double normalizedDistance =
            sample.channelDistance / std::max(0.25, sample.channelWidth * 0.5);
        if (selected != nullptr &&
            (normalizedDistance > selectedNormalizedDistance + CHANNEL_PROJECTION_TIE_EPSILON ||
             (std::abs(normalizedDistance - selectedNormalizedDistance) <=
                  CHANNEL_PROJECTION_TIE_EPSILON &&
              candidate.page->key >= selected->page->key))) {
            continue;
        }
        selected = &candidate;
        selectedNormalizedDistance = normalizedDistance;
    }
    return selected;
}

bool hasHandoffChannelProjection(const HandoffCandidate& candidate) {
    return candidate.containsQuery && candidate.sample.channelWidth > 0.0 &&
           std::isfinite(candidate.sample.channelDistance);
}

double handoffAxisWeight(int64_t pageOrigin, int64_t boundary, double coordinate) {
    const double eastFraction =
        std::clamp(0.5 + (coordinate - static_cast<double>(boundary)) /
                             (2.0 * static_cast<double>(NATIVE_HYDROLOGY_HANDOFF_BLOCKS)),
                   0.0, 1.0);
    return pageOrigin < boundary ? 1.0 - eastFraction : eastFraction;
}

double handoffProjectionWeight(const HandoffCandidate& candidate, double x, double z,
                               std::optional<int64_t> verticalBoundary,
                               std::optional<int64_t> horizontalBoundary) {
    double weight = 1.0;
    if (verticalBoundary)
        weight *= handoffAxisWeight(candidate.page->originX, *verticalBoundary, x);
    if (horizontalBoundary)
        weight *= handoffAxisWeight(candidate.page->originZ, *horizontalBoundary, z);
    return weight;
}

void carryHandoffChannelBank(BasinSample& result, std::span<const HandoffCandidate> candidates,
                             double x, double z, std::optional<int64_t> verticalBoundary,
                             std::optional<int64_t> horizontalBoundary) {
    // Wet topology already has a stage-owning authority.  A dry bank is the
    // only case where a nearby channel projection is descriptive rather than
    // competing with an established water body.
    if (result.ocean || result.lake || result.river || result.wetland) return;

    // Page construction is independently single-flight, so source order
    // must not select the bank geometry. Accumulate in lexical page order;
    // the per-axis weights form a partition of unity through an edge and a
    // bilinear partition through a corner.
    std::array<const HandoffCandidate*, 4> projections{};
    size_t projectionCount = 0;
    for (const HandoffCandidate& candidate : candidates) {
        if (hasHandoffChannelProjection(candidate)) projections[projectionCount++] = &candidate;
    }
    if (projectionCount == 0) return;
    std::sort(projections.begin(),
              projections.begin() + static_cast<std::ptrdiff_t>(projectionCount),
              [](const HandoffCandidate* first, const HandoffCandidate* second) {
                  return first->page->key < second->page->key;
              });

    double totalWeight = 0.0;
    double distance = 0.0;
    double width = 0.0;
    double depth = 0.0;
    double gradient = 0.0;
    for (size_t index = 0; index < projectionCount; ++index) {
        const HandoffCandidate& candidate = *projections[index];
        const double weight =
            handoffProjectionWeight(candidate, x, z, verticalBoundary, horizontalBoundary);
        if (weight <= 0.0) continue;
        totalWeight += weight;
        distance += weight * candidate.sample.channelDistance;
        width += weight * candidate.sample.channelWidth;
        depth += weight * candidate.sample.channelDepth;
        gradient += weight * candidate.sample.channelGradient;
    }

    if (totalWeight <= std::numeric_limits<double>::epsilon()) {
        // A projection can have zero cross-fade weight only at the outside
        // edge of the handoff. Keep the prior deterministic nearest fallback
        // there rather than dropping a valid dry-bank descriptor.
        const HandoffCandidate* channel = nearestHandoffChannel(candidates);
        if (channel == nullptr) return;
        result.channelDistance = channel->sample.channelDistance;
        result.channelWidth = channel->sample.channelWidth;
        result.channelDepth = channel->sample.channelDepth;
        result.channelGradient = channel->sample.channelGradient;
        return;
    }

    result.channelDistance = distance / totalWeight;
    result.channelWidth = width / totalWeight;
    result.channelDepth = depth / totalWeight;
    result.channelGradient = gradient / totalWeight;
}

BasinSample reconcileHandoff(uint64_t seed, std::span<const HandoffCandidate> candidates, double x,
                             double z, std::optional<int64_t> verticalBoundary,
                             std::optional<int64_t> horizontalBoundary) {
    if (candidates.empty()) throw std::logic_error("native hydrology handoff has no page");
    BasinSample result = candidates.front().sample;
    if (candidates.size() == 1) return result;

    const HandoffCandidate* bestRiver = nullptr;
    for (const HandoffCandidate& candidate : candidates) {
        if (!candidate.containsQuery || !candidate.sample.river) continue;
        if (bestRiver == nullptr || candidate.sample.discharge > bestRiver->sample.discharge ||
            (candidate.sample.discharge == bestRiver->sample.discharge &&
             candidate.page->key < bestRiver->page->key)) {
            bestRiver = &candidate;
        }
    }
    // Tiled lake reconciliation is completed from the half-open owner's
    // spill graph before ordinary apron handoff reaches this function.
    if (result.lake) return result;
    const bool anyOcean = std::ranges::any_of(candidates, [](const auto& candidate) {
        return candidate.containsQuery && candidate.sample.ocean;
    });
    if (anyOcean) {
        const auto ocean = std::ranges::find_if(candidates, [](const auto& candidate) {
            return candidate.containsQuery && candidate.sample.ocean;
        });
        return ocean->sample;
    }

    // A neighboring apron lake cannot overwrite dry half-open ownership.
    if (std::ranges::any_of(candidates, [](const auto& candidate) {
            return candidate.containsQuery && candidate.sample.lake;
        })) {
        return result;
    }

    if (!verticalBoundary && !horizontalBoundary) return result;

    if (bestRiver != nullptr) {
        result = bestRiver->sample;
        double boundaryStage = std::numeric_limits<double>::infinity();
        double boundaryDischarge = 0.0;
        const auto includeBoundary = [&](bool vertical, int64_t boundary) {
            for (const HandoffCandidate& candidate : candidates) {
                if (!candidate.containsQuery) continue;
                const EdgeSummarySample* summary =
                    boundarySummary(*candidate.page, vertical, boundary, vertical ? z : x);
                if (!summary || (summary->flags & CELL_RIVER) == 0) continue;
                boundaryStage = std::min(boundaryStage, static_cast<double>(summary->waterSurface));
                boundaryDischarge =
                    std::max(boundaryDischarge, static_cast<double>(summary->discharge));
            }
        };
        if (verticalBoundary) includeBoundary(true, *verticalBoundary);
        if (horizontalBoundary) includeBoundary(false, *horizontalBoundary);
        if (!std::isfinite(boundaryStage)) {
            boundaryStage = result.waterSurface;
            for (const HandoffCandidate& candidate : candidates) {
                if (candidate.containsQuery && candidate.sample.river)
                    boundaryStage = std::min(boundaryStage, candidate.sample.waterSurface);
            }
        }
        double boundaryDistance = NATIVE_HYDROLOGY_HANDOFF_BLOCKS;
        if (verticalBoundary)
            boundaryDistance =
                std::min(boundaryDistance, std::abs(x - static_cast<double>(*verticalBoundary)));
        if (horizontalBoundary)
            boundaryDistance =
                std::min(boundaryDistance, std::abs(z - static_cast<double>(*horizontalBoundary)));
        const double localAmount =
            std::clamp(boundaryDistance / NATIVE_HYDROLOGY_HANDOFF_BLOCKS, 0.0, 1.0);
        // An explicit fall already carries its immutable receiving stage.
        // Blending that stage without moving its lip metadata would produce a
        // different vertical interval on opposite sides of a page handoff.
        if (!result.waterfall)
            result.waterSurface = std::lerp(boundaryStage, result.waterSurface, localAmount);
        result.discharge = std::max(result.discharge, boundaryDischarge);
        result.waterBodyId =
            canonicalRiverPortId(seed, x, z, verticalBoundary, horizontalBoundary, result);
        result.surfaceElevation = std::min(
            result.surfaceElevation, result.waterSurface - std::max(0.125, result.channelDepth));
        result.erosionDepth = std::max(result.erosionDepth, bestRiver->sample.surfaceElevation -
                                                                result.surfaceElevation);
        result.channelDistance =
            std::isfinite(result.channelDistance) ? result.channelDistance : 1.0e9;
        result.valid = true;
    }
    carryHandoffChannelBank(result, candidates, x, z, verticalBoundary, horizontalBoundary);
    return result;
}

} // namespace

std::optional<NativeHydrologySpillResolution>
resolveNativeHydrologySpillSummaries(int64_t sourcePageX, int64_t sourcePageZ,
                                     WaterBodyId sourceLocalBodyId,
                                     std::span<const NativeHydrologySpillNodeSummary> nodes,
                                     std::span<const NativeHydrologySpillPortalSummary> portals) {
    if (sourceLocalBodyId == NO_WATER_BODY ||
        nodes.size() > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_NODES) {
        throw std::invalid_argument("invalid native hydrology spill-summary source or node count");
    }

    TiledLakeSpillHierarchy hierarchy;
    std::set<PageKey> pages;
    for (const NativeHydrologySpillNodeSummary& node : nodes) {
        if (node.localBodyId == NO_WATER_BODY || !std::isfinite(node.localStage) ||
            !std::isfinite(node.coreAreaSquareKilometers) ||
            !std::isfinite(node.coreVolumeCubicMeters) ||
            !std::isfinite(node.coreRunoffMmSquareKilometers) ||
            !std::isfinite(node.naturalOutletStage) || node.coreAreaSquareKilometers < 0.0 ||
            node.coreVolumeCubicMeters < 0.0 || node.coreRunoffMmSquareKilometers < 0.0 ||
            (node.naturalOutlet != BasinOutlet::NONE && node.naturalOutlet != BasinOutlet::OCEAN &&
             node.naturalOutlet != BasinOutlet::SHARED_PORTAL)) {
            throw std::invalid_argument("invalid native hydrology spill node");
        }
        pages.insert({.x = node.pageX, .z = node.pageZ});
        if (pages.size() > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES)
            throw std::invalid_argument("native hydrology spill-summary page bound exceeded");
        const float localStage = static_cast<float>(node.localStage);
        const float naturalOutletStage = static_cast<float>(node.naturalOutletStage);
        if (!std::isfinite(localStage) || !std::isfinite(naturalOutletStage))
            throw std::invalid_argument("native hydrology spill node exceeds float range");
        DepressionSummary summary{
            .waterBodyId = node.localBodyId,
            .localAnchorX = node.localAnchorX,
            .localAnchorZ = node.localAnchorZ,
            .localStage = localStage,
            .coreAreaSquareKilometers = node.coreAreaSquareKilometers,
            .coreVolumeCubicMeters = node.coreVolumeCubicMeters,
            .coreRunoffMmSquareKilometers = node.coreRunoffMmSquareKilometers,
            .naturalOutletX = node.naturalOutletX,
            .naturalOutletZ = node.naturalOutletZ,
            .naturalOutletStage = naturalOutletStage,
            .naturalOutlet = node.naturalOutlet,
        };
        const LakeNode key{.page = {.x = node.pageX, .z = node.pageZ}, .body = node.localBodyId};
        if (!hierarchy.add(key, summary))
            throw std::invalid_argument("duplicate native hydrology spill node");
    }
    for (const NativeHydrologySpillPortalSummary& portal : portals) {
        const __int128 deltaX = static_cast<__int128>(portal.secondPageX) - portal.firstPageX;
        const __int128 deltaZ = static_cast<__int128>(portal.secondPageZ) - portal.firstPageZ;
        const bool cardinal = (deltaZ == 0 && (deltaX == -1 || deltaX == 1)) ||
                              (deltaX == 0 && (deltaZ == -1 || deltaZ == 1));
        if (portal.firstLocalBodyId == NO_WATER_BODY || portal.secondLocalBodyId == NO_WATER_BODY ||
            !std::isfinite(portal.minimumWetStage) || !std::isfinite(portal.compatibleStage) ||
            portal.minimumWetStage >= portal.compatibleStage || !cardinal) {
            throw std::invalid_argument("invalid native hydrology spill portal");
        }
        const float minimumWetStage = static_cast<float>(portal.minimumWetStage);
        const float compatibleStage = static_cast<float>(portal.compatibleStage);
        if (!std::isfinite(minimumWetStage) || !std::isfinite(compatibleStage) ||
            minimumWetStage >= compatibleStage) {
            throw std::invalid_argument("native hydrology spill portal exceeds float range");
        }
        LakePortalConnection connection{
            .first = {.page = {.x = portal.firstPageX, .z = portal.firstPageZ},
                      .body = portal.firstLocalBodyId},
            .second = {.page = {.x = portal.secondPageX, .z = portal.secondPageZ},
                       .body = portal.secondLocalBodyId},
            .minimumWetStage = minimumWetStage,
            .compatibleStage = compatibleStage,
            .x = portal.x,
            .z = portal.z,
        };
        if (!hierarchy.nodes().contains(connection.first) ||
            !hierarchy.nodes().contains(connection.second)) {
            throw std::invalid_argument("native hydrology spill portal names an unknown node");
        }
        hierarchy.connect(connection);
    }

    const auto resolved = resolveTiledLakeSpillHierarchy(
        hierarchy, {.page = {.x = sourcePageX, .z = sourcePageZ}, .body = sourceLocalBodyId});
    if (!resolved) return std::nullopt;
    return NativeHydrologySpillResolution{
        .canonicalBodyId = resolved->canonicalBody,
        .stage = resolved->stage,
        .areaSquareKilometers = resolved->areaSquareKilometers,
        .volumeCubicMeters = resolved->volumeCubicMeters,
        .runoffMmSquareKilometers = resolved->runoffMmSquareKilometers,
        .outlet = resolved->outlet,
        .outletX = resolved->outletX,
        .outletZ = resolved->outletZ,
        .pageCount = resolved->pageCount,
    };
}

WaterBodyId nativeOceanWaterBodyId(uint64_t worldSeed) noexcept {
    return stableWaterBodyId(worldSeed, 0, 0, 0x4F43'4541'4E00'0001ULL);
}

NativeHydrologyIdentityRegistry::NativeHydrologyIdentityRegistry(uint64_t worldSeed) noexcept
    : seed_(worldSeed) {}

WaterBodyId NativeHydrologyIdentityRegistry::localLakeBodyId(int64_t rootX,
                                                             int64_t rootZ) const noexcept {
    return stableWaterBodyId(seed_,
                             world_coord::floorMultiple(rootX, LOCAL_LAKE_COARSE_ANCHOR_BLOCKS),
                             world_coord::floorMultiple(rootZ, LOCAL_LAKE_COARSE_ANCHOR_BLOCKS),
                             0x4C41'4B45'0000'0002ULL);
}

WaterBodyId
NativeHydrologyIdentityRegistry::disambiguatedLocalLakeBodyId(int64_t rootX,
                                                              int64_t rootZ) const noexcept {
    return stableWaterBodyId(seed_, rootX, rootZ, 0x4C41'4B45'0000'0003ULL);
}

uint64_t NativeHydrologyIdentityRegistry::seed() const noexcept {
    return seed_;
}

learned::NativeRect nativeHydrologyFinalTerrainRegion(int64_t ownerPageX, int64_t ownerPageZ) {
    const auto checkedNativeCoordinate = [](int64_t owner, int offset) {
        const __int128 coordinate = static_cast<__int128>(owner) * INTERIOR_INTERVALS + offset;
        if (coordinate < std::numeric_limits<int64_t>::min() ||
            coordinate > std::numeric_limits<int64_t>::max()) {
            throw std::out_of_range("native hydrology final terrain region exceeds int64 range");
        }
        return static_cast<int64_t>(coordinate);
    };
    const int64_t rowBegin = checkedNativeCoordinate(ownerPageZ, -RASTER_APRON);
    const int64_t columnBegin = checkedNativeCoordinate(ownerPageX, -RASTER_APRON);
    const int64_t rowEnd =
        checkedNativeCoordinate(ownerPageZ, INTERIOR_INTERVALS + RASTER_APRON + 1);
    const int64_t columnEnd =
        checkedNativeCoordinate(ownerPageX, INTERIOR_INTERVALS + RASTER_APRON + 1);
    const learned::NativeRect region{
        .rowBegin = rowBegin, .columnBegin = columnBegin, .rowEnd = rowEnd, .columnEnd = columnEnd};
    if (region.height() != RASTER_EDGE || region.width() != RASTER_EDGE)
        throw std::logic_error("native hydrology final terrain region has an invalid extent");
    return region;
}

std::vector<learned::TerrainPageCoordinate>
nativeHydrologyRequiredAuthorityPages(int64_t ownerPageX, int64_t ownerPageZ) {
    const int64_t minimumX = checkedCoordinate(ownerPageX, -RASTER_APRON);
    const int64_t maximumX = checkedCoordinate(ownerPageX, INTERIOR_INTERVALS + RASTER_APRON);
    const int64_t minimumZ = checkedCoordinate(ownerPageZ, -RASTER_APRON);
    const int64_t maximumZ = checkedCoordinate(ownerPageZ, INTERIOR_INTERVALS + RASTER_APRON);
    const int64_t firstColumn =
        world_coord::floorDiv(minimumX, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastColumn =
        world_coord::floorDiv(maximumX, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t firstRow =
        world_coord::floorDiv(minimumZ, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastRow =
        world_coord::floorDiv(maximumZ, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    std::vector<learned::TerrainPageCoordinate> result;
    result.reserve(static_cast<size_t>(lastRow - firstRow + 1) *
                   static_cast<size_t>(lastColumn - firstColumn + 1));
    for (int64_t row = firstRow; row <= lastRow; ++row) {
        for (int64_t column = firstColumn; column <= lastColumn; ++column)
            result.push_back({.row = row, .column = column});
    }
    return result;
}

NativeHydrologyAuthorityRequirements nativeHydrologyAuthorityRequirementsForWorldRect(
    int64_t minimumX, int64_t minimumZ, int64_t maximumXExclusive, int64_t maximumZExclusive) {
    NativeHydrologyAuthorityRequirements result;
    if (minimumX >= maximumXExclusive || minimumZ >= maximumZExclusive) return result;

    const int64_t firstOwnerX =
        world_coord::floorDiv(minimumX, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t lastOwnerX = world_coord::floorDiv(
        maximumXExclusive - 1, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t firstOwnerZ =
        world_coord::floorDiv(minimumZ, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
    const int64_t lastOwnerZ = world_coord::floorDiv(
        maximumZExclusive - 1, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE));
    std::set<learned::TerrainPageCoordinate> topologyPages;
    for (int64_t ownerZ = firstOwnerZ;; ++ownerZ) {
        for (int64_t ownerX = firstOwnerX;; ++ownerX) {
            const std::vector<learned::TerrainPageCoordinate> ownerPages =
                nativeHydrologyRequiredAuthorityPages(ownerX, ownerZ);
            topologyPages.insert(ownerPages.begin(), ownerPages.end());
            if (ownerX == lastOwnerX) break;
        }
        if (ownerZ == lastOwnerZ) break;
    }
    result.finalTopologyPages.assign(topologyPages.begin(), topologyPages.end());

    const int64_t firstFinalColumn =
        world_coord::floorDiv(minimumX, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastFinalColumn = world_coord::floorDiv(
        maximumXExclusive - 1, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t firstFinalRow =
        world_coord::floorDiv(minimumZ, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastFinalRow = world_coord::floorDiv(
        maximumZExclusive - 1, static_cast<int64_t>(learned::AUTHORITY_PAGE_BLOCK_EDGE));
    for (int64_t row = firstFinalRow;; ++row) {
        for (int64_t column = firstFinalColumn;; ++column) {
            result.finalRefinementPages.push_back({.row = row, .column = column});
            if (column == lastFinalColumn) break;
        }
        if (row == lastFinalRow) break;
    }
    return result;
}

class NativeHydrologyRouter::Impl {
    struct LakePairEntry {
        std::shared_ptr<const std::vector<LakePortalConnection>> connections;
        size_t bytes = 0;
        std::list<LakePairKey>::iterator recency;
    };

    struct OpenLakeRegionEntry {
        std::shared_ptr<const ResolvedOpenLakeRegion> region;
        size_t bytes = 0;
        std::list<LakeNode>::iterator recency;
    };

    struct OpenLakeRegionFlight {
        std::condition_variable ready;
        bool complete = false;
        std::shared_ptr<const ResolvedOpenLakeRegion> region;
        std::exception_ptr failure;
    };

public:
    Impl(uint64_t requestedSeed, size_t requestedBudget,
         std::shared_ptr<hydrology::HydrologyAuthorityStore> requestedStore,
         std::shared_ptr<const NativeHydrologyIdentityRegistry> requestedIdentityRegistry)
        : seed(requestedSeed)
        , cacheByteBudget(requestedBudget)
        , authorityStore(std::move(requestedStore))
        , identityRegistry(requestedIdentityRegistry
                               ? std::move(requestedIdentityRegistry)
                               : std::make_shared<NativeHydrologyIdentityRegistry>(requestedSeed)) {
        if (identityRegistry->seed() != seed)
            throw std::invalid_argument("native hydrology identity registry seed mismatch");
    }

    static void promotePriority(std::atomic<learned::AuthorityRequestPriority>& destination,
                                learned::AuthorityRequestPriority priority) noexcept {
        learned::AuthorityRequestPriority current = destination.load(std::memory_order_relaxed);
        while (priority < current &&
               !destination.compare_exchange_weak(current, priority, std::memory_order_relaxed)) {
        }
    }

    bool evictPageFor(learned::AuthorityRequestPriority incomingPriority) const {
        auto victim = entries.end();
        for (auto key = recency.rbegin(); key != recency.rend(); ++key) {
            const auto found = entries.find(*key);
            if (found == entries.end()) std::terminate();
            if (found->second.priority < incomingPriority) continue;
            if (victim == entries.end() || found->second.priority > victim->second.priority)
                victim = found;
        }
        if (victim == entries.end()) return false;
        metrics.bytes -= victim->second.bytes;
        recency.erase(victim->second.recency);
        entries.erase(victim);
        return true;
    }

    std::shared_ptr<const NativePage>
    getOrCreate(PageKey key, const NativeHydrologyInputFunction& input,
                learned::AuthorityRequestPriority priority) const {
        std::shared_ptr<Flight> flight;
        bool builder = false;
        uint64_t buildEpoch = 0;
        {
            std::unique_lock lock(mutex);
            if (auto found = entries.find(key); found != entries.end()) {
                recency.splice(recency.begin(), recency, found->second.recency);
                found->second.priority = std::min(found->second.priority, priority);
                ++metrics.hits;
                return found->second.page;
            }
            if (auto active = flights.find(key); active != flights.end()) {
                flight = active->second;
                promotePriority(flight->priority, priority);
                ++metrics.hits;
            } else {
                flight = std::make_shared<Flight>();
                flight->priority.store(priority, std::memory_order_relaxed);
                flights.emplace(key, flight);
                builder = true;
                buildEpoch = epoch;
                ++metrics.misses;
            }
            if (!builder) {
                flight->ready.wait(lock, [&] { return flight->complete; });
                if (flight->failure) std::rethrow_exception(flight->failure);
                return flight->page;
            }
        }

        std::shared_ptr<const NativePage> page;
        std::exception_ptr failure;
        size_t peakBuildBytes = 0;
        bool loadedPersisted = false;
        bool wrotePersisted = false;
        bool repairedPersisted = false;
        size_t persistedPayloadBytes = 0;
        uint64_t warmLoadNanoseconds = 0;
        bool admissionWaited = false;
        bool deferredBuild = false;
        NativeHydrologyBuildGate::Lease buildLease =
            sharedNativeHydrologyBuildGate().acquire(flight->priority, admissionWaited);
        {
            std::lock_guard lock(mutex);
            ++metrics.activeBuilds;
            metrics.peakConcurrentBuilds =
                std::max(metrics.peakConcurrentBuilds, metrics.activeBuilds);
            if (admissionWaited) ++metrics.buildAdmissionWaits;
        }
        try {
            bool loaded = false;
            std::optional<std::vector<uint8_t>> corruptPersistedPayload;
            if (authorityStore) {
                const auto loadStart = std::chrono::steady_clock::now();
                auto persisted = authorityStore->load({.x = key.x, .z = key.z});
                if (persisted.isReady()) {
                    try {
                        page = deserializePage(seed, key, *persisted.value());
                    } catch (const std::exception&) {
                        // The RYHY envelope is valid, but its opaque native
                        // payload is not. Keep the exact bytes so publication
                        // can replace only this proven-corrupt version.
                        corruptPersistedPayload = *persisted.value();
                        repairedPersisted = true;
                    }
                    if (!corruptPersistedPayload) {
                        loaded = true;
                        loadedPersisted = true;
                        persistedPayloadBytes = persisted.value()->size();
                        warmLoadNanoseconds = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - loadStart)
                                .count());
                        peakBuildBytes = page->byteSize() + persisted.value()->capacity();
                    }
                } else if (persisted.status() == learned::AuthorityStatus::FAILED) {
                    if (persisted.failure() &&
                        persisted.failure()->code == learned::GenerationFailureCode::CORRUPT_PAGE) {
                        repairedPersisted = true;
                    } else {
                        throw learned::GenerationFailureException(
                            persisted.status(),
                            persisted.failure()
                                ? *persisted.failure()
                                : learned::GenerationFailure{
                                      .code = learned::GenerationFailureCode::IO_ERROR,
                                      .message =
                                          "Native hydrology persistence failed without a reason",
                                      .retriable = true,
                                  });
                    }
                }
            }
            if (!loaded) {
                page = buildPage(seed, *identityRegistry, key, input, peakBuildBytes);
                if (authorityStore) {
                    const std::vector<uint8_t> payload = serializePage(*page);
                    auto written =
                        corruptPersistedPayload
                            ? authorityStore->replaceCorruptPayload(
                                  {.x = key.x, .z = key.z}, *corruptPersistedPayload, payload)
                            : authorityStore->write({.x = key.x, .z = key.z}, payload);
                    if (!written.isReady())
                        throw learned::GenerationFailureException(written.status(),
                                                                  *written.failure());
                    wrotePersisted = true;
                    persistedPayloadBytes = payload.size();
                    peakBuildBytes =
                        std::max(peakBuildBytes, page->byteSize() + payload.capacity());
                }
            }
        } catch (const learned::GenerationFailureException& exception) {
            deferredBuild = exception.status() == learned::AuthorityStatus::DEFERRED;
            failure = std::current_exception();
        } catch (...) {
            failure = std::current_exception();
        }

        {
            std::lock_guard lock(mutex);
            if (metrics.activeBuilds == 0) std::terminate();
            --metrics.activeBuilds;
            metrics.peakBuildBytes = std::max(metrics.peakBuildBytes, peakBuildBytes);
            if (failure) {
                if (deferredBuild)
                    ++metrics.deferredBuilds;
                else
                    ++metrics.failures;
            } else {
                ++metrics.builds;
                if (loadedPersisted) {
                    ++metrics.persistedLoads;
                    metrics.lastWarmLoadNanoseconds = warmLoadNanoseconds;
                }
                if (wrotePersisted) ++metrics.persistedWrites;
                if (repairedPersisted) ++metrics.persistedRepairs;
                if (persistedPayloadBytes != 0)
                    metrics.lastPersistedPayloadBytes = persistedPayloadBytes;
                if (buildEpoch == epoch && page->byteSize() <= cacheByteBudget) {
                    const learned::AuthorityRequestPriority retainedPriority =
                        flight->priority.load(std::memory_order_relaxed);
                    recency.push_front(key);
                    entries.emplace(key, Entry{.page = page,
                                               .bytes = page->byteSize(),
                                               .priority = retainedPriority,
                                               .recency = recency.begin()});
                    metrics.bytes += page->byteSize();
                    while (metrics.bytes > cacheByteBudget && !recency.empty()) {
                        if (!evictPageFor(retainedPriority)) break;
                    }
                    metrics.entries = entries.size();
                }
            }
            flight->page = page;
            flight->failure = failure;
            flight->complete = true;
            flights.erase(key);
        }
        flight->ready.notify_all();
        if (failure) std::rethrow_exception(failure);
        return page;
    }

    std::shared_ptr<const std::vector<LakePortalConnection>>
    lakePortalConnections(const NativePage& first, const NativePage& second) const {
        bool vertical = false;
        const NativePage* low = nullptr;
        const NativePage* high = nullptr;
        if (!adjacentPagePair(first, second, vertical, low, high)) {
            static const auto EMPTY = std::make_shared<const std::vector<LakePortalConnection>>();
            return EMPTY;
        }
        static_cast<void>(vertical);
        const LakePairKey key{.first = low->key, .second = high->key};
        {
            std::lock_guard lock(mutex);
            if (auto found = lakePairEntries.find(key); found != lakePairEntries.end()) {
                lakePairRecency.splice(lakePairRecency.begin(), lakePairRecency,
                                       found->second.recency);
                ++metrics.reconciliationHits;
                return found->second.connections;
            }
            ++metrics.reconciliationMisses;
        }

        auto built = std::make_shared<const std::vector<LakePortalConnection>>(
            compatibleLakePortals(*low, *high));
        const size_t bytes = built->capacity() * sizeof(LakePortalConnection);
        std::lock_guard lock(mutex);
        if (auto found = lakePairEntries.find(key); found != lakePairEntries.end()) {
            lakePairRecency.splice(lakePairRecency.begin(), lakePairRecency, found->second.recency);
            ++metrics.reconciliationHits;
            return found->second.connections;
        }
        lakePairRecency.push_front(key);
        lakePairEntries.emplace(key, LakePairEntry{.connections = built,
                                                   .bytes = bytes,
                                                   .recency = lakePairRecency.begin()});
        metrics.reconciliationBytes += bytes;
        while (lakePairEntries.size() > TILED_LAKE_PAIR_CACHE_ENTRIES) {
            const LakePairKey evicted = lakePairRecency.back();
            const auto found = lakePairEntries.find(evicted);
            if (found != lakePairEntries.end()) {
                metrics.reconciliationBytes -= found->second.bytes;
                lakePairEntries.erase(found);
            }
            lakePairRecency.pop_back();
        }
        metrics.reconciliationEntries = lakePairEntries.size();
        return built;
    }

    template <typename PageResolver>
    std::shared_ptr<const ResolvedOpenLakeRegion>
    openLakeRegion(const std::shared_ptr<const NativePage>& sourcePage, WaterBodyId sourceBody,
                   PageResolver& resolvePage) const {
        const LakeNode key{.page = sourcePage->key, .body = sourceBody};
        std::shared_ptr<OpenLakeRegionFlight> flight;
        bool builder = false;
        uint64_t buildEpoch = 0;
        {
            std::unique_lock lock(mutex);
            if (auto found = openLakeRegionEntries.find(key);
                found != openLakeRegionEntries.end()) {
                openLakeRegionRecency.splice(openLakeRegionRecency.begin(), openLakeRegionRecency,
                                             found->second.recency);
                ++metrics.openDepressionHits;
                return found->second.region;
            }
            if (auto active = openLakeRegionFlights.find(key);
                active != openLakeRegionFlights.end()) {
                flight = active->second;
                ++metrics.openDepressionHits;
            } else {
                flight = std::make_shared<OpenLakeRegionFlight>();
                openLakeRegionFlights.emplace(key, flight);
                builder = true;
                buildEpoch = epoch;
                ++metrics.openDepressionMisses;
            }
            if (!builder) {
                flight->ready.wait(lock, [&] { return flight->complete; });
                if (flight->failure) std::rethrow_exception(flight->failure);
                return flight->region;
            }
        }

        std::shared_ptr<const ResolvedOpenLakeRegion> region;
        std::exception_ptr failure;
        try {
            region = buildResolvedOpenLakeRegion(sourcePage, sourceBody, resolvePage);
        } catch (...) {
            failure = std::current_exception();
        }

        {
            std::lock_guard lock(mutex);
            if (!failure && buildEpoch == epoch) {
                const size_t bytes = region ? region->byteSize() : 0;
                openLakeRegionRecency.push_front(key);
                openLakeRegionEntries.emplace(
                    key, OpenLakeRegionEntry{.region = region,
                                             .bytes = bytes,
                                             .recency = openLakeRegionRecency.begin()});
                openLakeRegionBytes += bytes;
                if (region) {
                    for (int64_t pageZ = region->minimumPage.z;; ++pageZ) {
                        for (int64_t pageX = region->minimumPage.x;; ++pageX) {
                            auto& associated = openLakeRegionsByPage[{.x = pageX, .z = pageZ}];
                            const bool alreadyAssociated =
                                std::ranges::any_of(associated, [&region](const auto& weak) {
                                    return weak.lock() == region;
                                });
                            if (!alreadyAssociated) associated.push_back(region);
                            if (pageX == region->maximumPage.x) break;
                        }
                        if (pageZ == region->maximumPage.z) break;
                    }
                }
                while ((!openLakeRegionRecency.empty() &&
                        openLakeRegionBytes > OPEN_LAKE_REGION_CACHE_BYTE_BUDGET) ||
                       openLakeRegionEntries.size() > OPEN_LAKE_REGION_CACHE_ENTRIES) {
                    const LakeNode evicted = openLakeRegionRecency.back();
                    const auto found = openLakeRegionEntries.find(evicted);
                    if (found != openLakeRegionEntries.end()) {
                        openLakeRegionBytes -= found->second.bytes;
                        openLakeRegionEntries.erase(found);
                    }
                    openLakeRegionRecency.pop_back();
                }
                ++metrics.openDepressionBuilds;
                metrics.openDepressionEntries = openLakeRegionEntries.size();
                metrics.openDepressionBytes = openLakeRegionBytes;
            } else if (failure) {
                ++metrics.openDepressionFailures;
            }
            flight->region = region;
            flight->failure = failure;
            flight->complete = true;
            openLakeRegionFlights.erase(key);
        }
        flight->ready.notify_all();
        if (failure) std::rethrow_exception(failure);
        return region;
    }

    template <typename PageResolver>
    std::optional<TiledLakeResolution>
    resolveLakeSpillHierarchy(const std::shared_ptr<const NativePage>& ownerPage,
                              WaterBodyId ownerBody, PageResolver& resolvePage) const {
        const LakeNode source{.page = ownerPage->key, .body = ownerBody};
        const DepressionSummary* sourceSummary = depressionSummary(*ownerPage, ownerBody);
        if (sourceSummary == nullptr)
            throw std::runtime_error("native hydrology lake has no depression summary");
        if (sourceSummary->edgeMask == 0) return std::nullopt;

        TiledLakeSpillHierarchy hierarchy;
        std::set<LakeNode> frontier{source};
        std::set<LakeNode> discovered{source};
        std::set<PageKey> connectedPages{source.page};
        std::set<PageKey> inspectedPages;
        constexpr size_t MAXIMUM_INSPECTED_PAGES = NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES * 5U;
        while (!frontier.empty()) {
            const LakeNode node = *frontier.begin();
            frontier.erase(frontier.begin());
            const std::shared_ptr<const NativePage> page =
                node.page == ownerPage->key ? ownerPage : resolvePage(node.page);
            inspectedPages.insert(node.page);
            const DepressionSummary* summary = depressionSummary(*page, node.body);
            if (summary == nullptr)
                throw std::runtime_error("native hydrology spill hierarchy lost a component");
            hierarchy.add(node, *summary);

            constexpr std::array<std::tuple<PageEdge, int64_t, int64_t>, 4> NEIGHBORS{{
                {PageEdge::WEST, -1, 0},
                {PageEdge::EAST, 1, 0},
                {PageEdge::NORTH, 0, -1},
                {PageEdge::SOUTH, 0, 1},
            }};
            for (const auto [edge, deltaX, deltaZ] : NEIGHBORS) {
                if ((summary->edgeMask & (1U << static_cast<uint8_t>(edge))) == 0) continue;
                const __int128 neighborX = static_cast<__int128>(node.page.x) + deltaX;
                const __int128 neighborZ = static_cast<__int128>(node.page.z) + deltaZ;
                if (neighborX < std::numeric_limits<int64_t>::min() ||
                    neighborX > std::numeric_limits<int64_t>::max() ||
                    neighborZ < std::numeric_limits<int64_t>::min() ||
                    neighborZ > std::numeric_limits<int64_t>::max()) {
                    continue;
                }
                const PageKey neighborKey{.x = static_cast<int64_t>(neighborX),
                                          .z = static_cast<int64_t>(neighborZ)};
                const std::shared_ptr<const NativePage> neighbor = resolvePage(neighborKey);
                inspectedPages.insert(neighborKey);
                if (inspectedPages.size() > MAXIMUM_INSPECTED_PAGES) {
                    throw std::runtime_error(
                        "native hydrology spill hierarchy exceeded its inspection bound");
                }
                const std::shared_ptr<const std::vector<LakePortalConnection>> connections =
                    lakePortalConnections(*page, *neighbor);
                for (const LakePortalConnection& connection : *connections) {
                    LakeNode next;
                    if (connection.first == node)
                        next = connection.second;
                    else if (connection.second == node)
                        next = connection.first;
                    else
                        continue;
                    hierarchy.connect(connection);
                    if (!discovered.insert(next).second) continue;
                    connectedPages.insert(next.page);
                    if (connectedPages.size() > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES ||
                        discovered.size() > NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_NODES) {
                        throw std::runtime_error(
                            "native hydrology spill hierarchy exceeded its deterministic bound");
                    }
                    frontier.insert(next);
                }
            }
        }
        return resolveTiledLakeSpillHierarchy(hierarchy, source);
    }

    template <typename PageResolver>
    std::optional<WetlandResolution>
    resolveConnectedWetland(const std::shared_ptr<const NativePage>& ownerPage, double x, double z,
                            PageResolver& resolvePage) const {
        const double ownerGridX =
            (x - static_cast<double>(ownerPage->originX)) / NATIVE_HYDROLOGY_RASTER_SPACING +
            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
        const double ownerGridZ =
            (z - static_cast<double>(ownerPage->originZ)) / NATIVE_HYDROLOGY_RASTER_SPACING +
            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
        const int nearestX =
            std::clamp(static_cast<int>(std::floor(ownerGridX + 0.5)), 0, RASTER_EDGE - 1);
        const int nearestZ =
            std::clamp(static_cast<int>(std::floor(ownerGridZ + 0.5)), 0, RASTER_EDGE - 1);
        const WetlandNode source{
            .x = checkedCoordinate(ownerPage->key.x, nearestX - RASTER_APRON),
            .z = checkedCoordinate(ownerPage->key.z, nearestZ - RASTER_APRON),
        };
        const size_t sourceCell = static_cast<size_t>(indexOf(nearestX, nearestZ));
        if ((ownerPage->flags[sourceCell] & CELL_WETLAND_CANDIDATE) == 0) return std::nullopt;
        {
            std::lock_guard lock(mutex);
            if (const auto found = connectedWetlandResolutions.find(source);
                found != connectedWetlandResolutions.end()) {
                return found->second;
            }
        }

        struct GraphNode {
            WetlandNode key;
            float rawElevation = 0.0F;
            std::vector<size_t> downstream;
            std::vector<WetlandResolution> directParents;
        };
        std::vector<GraphNode> graph;
        std::unordered_map<WetlandNode, size_t, WetlandNodeHash> indices;
        std::vector<size_t> frontier;
        std::set<PageKey> pages;
        const auto addNode = [&](WetlandNode key) {
            const auto [found, inserted] = indices.emplace(key, graph.size());
            if (inserted) {
                if (graph.size() >= MAXIMUM_CONNECTED_WETLAND_CELLS) {
                    throw std::runtime_error(
                        "native hydrology connected wetland exceeded its cell bound");
                }
                graph.push_back({.key = key});
                frontier.push_back(found->second);
            }
            return found->second;
        };
        const size_t sourceIndex = addNode(source);

        struct LocatedCell {
            std::shared_ptr<const NativePage> page;
            size_t cell = 0;
            int rasterX = 0;
            int rasterZ = 0;
        };
        const auto locate = [&](WetlandNode node) {
            const PageKey owner{
                .x =
                    world_coord::floorDiv(node.x, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
                .z =
                    world_coord::floorDiv(node.z, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
            };
            if (pages.insert(owner).second && pages.size() > MAXIMUM_CONNECTED_WETLAND_PAGES) {
                throw std::runtime_error(
                    "native hydrology connected wetland exceeded its page bound");
            }
            std::shared_ptr<const NativePage> page =
                owner == ownerPage->key ? ownerPage : resolvePage(owner);
            const int64_t localX = node.x - page->originX;
            const int64_t localZ = node.z - page->originZ;
            if (localX < 0 || localX >= NATIVE_HYDROLOGY_PAGE_EDGE || localZ < 0 ||
                localZ >= NATIVE_HYDROLOGY_PAGE_EDGE ||
                localX % NATIVE_HYDROLOGY_RASTER_SPACING != 0 ||
                localZ % NATIVE_HYDROLOGY_RASTER_SPACING != 0) {
                throw std::runtime_error("native hydrology connected wetland cell is misaligned");
            }
            const int rasterX =
                CORE_BEGIN + static_cast<int>(localX / NATIVE_HYDROLOGY_RASTER_SPACING);
            const int rasterZ =
                CORE_BEGIN + static_cast<int>(localZ / NATIVE_HYDROLOGY_RASTER_SPACING);
            return LocatedCell{.page = std::move(page),
                               .cell = static_cast<size_t>(indexOf(rasterX, rasterZ)),
                               .rasterX = rasterX,
                               .rasterZ = rasterZ};
        };

        for (size_t head = 0; head < frontier.size(); ++head) {
            const size_t nodeIndex = frontier[head];
            const LocatedCell located = locate(graph[nodeIndex].key);
            const uint8_t flags = located.page->flags[located.cell];
            if ((flags & CELL_WETLAND_CANDIDATE) == 0) continue;
            graph[nodeIndex].rawElevation = located.page->rawElevation[located.cell];
            if ((flags & CELL_WETLAND) != 0 &&
                located.page->waterBodyIds[located.cell] != NO_WATER_BODY &&
                std::isfinite(located.page->waterSurface[located.cell])) {
                graph[nodeIndex].directParents.push_back(
                    {.body = located.page->waterBodyIds[located.cell],
                     .stage = located.page->waterSurface[located.cell]});
            }
            const std::array<int32_t, 2> targets{
                located.page->receiverFirst[located.cell],
                located.page->receiverSecond[located.cell],
            };
            const std::array<float, 2> weights{
                targets[1] >= 0 ? 1.0F - located.page->receiverSecondWeight[located.cell] : 1.0F,
                targets[1] >= 0 ? located.page->receiverSecondWeight[located.cell] : 0.0F,
            };
            for (size_t branch = 0; branch < targets.size(); ++branch) {
                const int target = targets[branch];
                if (target < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT) continue;
                const int targetRasterX = target % RASTER_EDGE;
                const int targetRasterZ = target / RASTER_EDGE;
                const WetlandNode targetKey{
                    .x = checkedCoordinate(located.page->key.x, targetRasterX - RASTER_APRON),
                    .z = checkedCoordinate(located.page->key.z, targetRasterZ - RASTER_APRON),
                };
                const LocatedCell targetCell = locate(targetKey);
                const uint8_t targetFlags = targetCell.page->flags[targetCell.cell];
                const bool finishedWater =
                    (targetFlags & (CELL_OCEAN | CELL_LAKE | CELL_RIVER | CELL_WETLAND)) != 0 &&
                    (targetFlags & CELL_WATERFALL) == 0 &&
                    targetCell.page->waterBodyIds[targetCell.cell] != NO_WATER_BODY &&
                    std::isfinite(targetCell.page->waterSurface[targetCell.cell]);
                if (finishedWater) {
                    graph[nodeIndex].directParents.push_back(
                        {.body = targetCell.page->waterBodyIds[targetCell.cell],
                         .stage = targetCell.page->waterSurface[targetCell.cell]});
                    continue;
                }
                if ((targetFlags & CELL_WETLAND_CANDIDATE) == 0) continue;
                const size_t downstream = addNode(targetKey);
                graph[nodeIndex].downstream.push_back(downstream);
            }
            std::ranges::sort(graph[nodeIndex].downstream);
            graph[nodeIndex].downstream.erase(
                std::unique(graph[nodeIndex].downstream.begin(), graph[nodeIndex].downstream.end()),
                graph[nodeIndex].downstream.end());
        }

        std::vector<std::vector<size_t>> upstream(graph.size());
        for (size_t node = 0; node < graph.size(); ++node) {
            for (const size_t downstream : graph[node].downstream)
                upstream[downstream].push_back(node);
        }
        struct RankedResolution {
            WetlandResolution resolution;
            size_t node = 0;
        };
        const auto rankedGreater = [&graph](const RankedResolution& first,
                                            const RankedResolution& second) {
            return std::tuple{first.resolution.stage, first.resolution.body,
                              graph[first.node].key.x, graph[first.node].key.z} >
                   std::tuple{second.resolution.stage, second.resolution.body,
                              graph[second.node].key.x, graph[second.node].key.z};
        };
        std::priority_queue<RankedResolution, std::vector<RankedResolution>,
                            decltype(rankedGreater)>
            ready(rankedGreater);
        std::vector<std::optional<WetlandResolution>> resolved(graph.size());
        const auto compatible = [&graph](size_t node, const WetlandResolution& resolution) {
            return std::abs(graph[node].rawElevation - resolution.stage) <=
                   WETLAND_MAX_PARENT_STAGE_OFFSET_BLOCKS;
        };
        const auto consider = [&](size_t node, WetlandResolution candidate) {
            if (!compatible(node, candidate)) return;
            if (!resolved[node] || std::tuple{candidate.stage, candidate.body} <
                                       std::tuple{resolved[node]->stage, resolved[node]->body}) {
                resolved[node] = candidate;
                ready.push({.resolution = candidate, .node = node});
            }
        };
        for (size_t node = 0; node < graph.size(); ++node) {
            for (const WetlandResolution parent : graph[node].directParents)
                consider(node, parent);
        }
        while (!ready.empty()) {
            const RankedResolution current = ready.top();
            ready.pop();
            if (!resolved[current.node] || *resolved[current.node] != current.resolution) continue;
            for (const size_t parent : upstream[current.node])
                consider(parent, current.resolution);
        }

        {
            std::lock_guard lock(mutex);
            if (connectedWetlandResolutions.size() + graph.size() >
                MAXIMUM_CONNECTED_WETLAND_CELLS) {
                connectedWetlandResolutions.clear();
            }
            for (size_t node = 0; node < graph.size(); ++node)
                connectedWetlandResolutions.insert_or_assign(graph[node].key, resolved[node]);
        }
        return resolved[sourceIndex];
    }

    template <typename PageResolver>
    std::optional<EstuaryResolution>
    resolveSeaBackwater(const std::shared_ptr<const NativePage>& ownerPage, double x, double z,
                        PageResolver& resolvePage) const {
        if (!ownerPage->hasCoastalResolutionCandidates) return std::nullopt;
        const double gridX =
            (x - static_cast<double>(ownerPage->originX)) / NATIVE_HYDROLOGY_RASTER_SPACING +
            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
        const double gridZ =
            (z - static_cast<double>(ownerPage->originZ)) / NATIVE_HYDROLOGY_RASTER_SPACING +
            RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET;
        const int nearestX =
            std::clamp(static_cast<int>(std::floor(gridX + 0.5)), 0, RASTER_EDGE - 1);
        const int nearestZ =
            std::clamp(static_cast<int>(std::floor(gridZ + 0.5)), 0, RASTER_EDGE - 1);
        const size_t nearestCell = static_cast<size_t>(indexOf(nearestX, nearestZ));
        if (ownerPage->channelProximity[nearestCell] == 0) return std::nullopt;
        const ChannelProjection channel = projectChannel(*ownerPage, x, z, gridX, gridZ);
        if (channel.source < 0 || channel.body == NO_WATER_BODY ||
            channel.distance > channel.width * 0.5 || channel.waterfall ||
            channel.gradient > ESTUARY_MAXIMUM_CHANNEL_GRADIENT) {
            return std::nullopt;
        }
        int sourceRasterX = channel.source % RASTER_EDGE;
        int sourceRasterZ = channel.source / RASTER_EDGE;
        const uint8_t nearestFlags = ownerPage->flags[nearestCell];
        if ((nearestFlags & CELL_RIVER) != 0 && (nearestFlags & CELL_WATERFALL) == 0 &&
            ownerPage->channelGradient[nearestCell] <= ESTUARY_MAXIMUM_CHANNEL_GRADIENT) {
            sourceRasterX = nearestX;
            sourceRasterZ = nearestZ;
        }
        const WetlandNode source{
            .x = checkedCoordinate(ownerPage->key.x, sourceRasterX - RASTER_APRON),
            .z = checkedCoordinate(ownerPage->key.z, sourceRasterZ - RASTER_APRON),
        };
        {
            std::lock_guard lock(mutex);
            if (const auto found = seaBackwaterResolutions.find(source);
                found != seaBackwaterResolutions.end()) {
                return found->second;
            }
        }

        struct LocatedCell {
            std::shared_ptr<const NativePage> page;
            size_t cell = 0;
        };
        std::set<PageKey> pages;
        const auto locate = [&](WetlandNode node) {
            const PageKey owner{
                .x =
                    world_coord::floorDiv(node.x, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
                .z =
                    world_coord::floorDiv(node.z, static_cast<int64_t>(NATIVE_HYDROLOGY_PAGE_EDGE)),
            };
            if (pages.insert(owner).second && pages.size() > MAXIMUM_CONNECTED_WETLAND_PAGES) {
                throw std::runtime_error("native hydrology estuary search exceeded its page bound");
            }
            std::shared_ptr<const NativePage> page =
                owner == ownerPage->key ? ownerPage : resolvePage(owner);
            const int64_t localX = node.x - page->originX;
            const int64_t localZ = node.z - page->originZ;
            if (localX < 0 || localX >= NATIVE_HYDROLOGY_PAGE_EDGE || localZ < 0 ||
                localZ >= NATIVE_HYDROLOGY_PAGE_EDGE ||
                localX % NATIVE_HYDROLOGY_RASTER_SPACING != 0 ||
                localZ % NATIVE_HYDROLOGY_RASTER_SPACING != 0) {
                throw std::runtime_error("native hydrology estuary cell is misaligned");
            }
            const int rasterX =
                CORE_BEGIN + static_cast<int>(localX / NATIVE_HYDROLOGY_RASTER_SPACING);
            const int rasterZ =
                CORE_BEGIN + static_cast<int>(localZ / NATIVE_HYDROLOGY_RASTER_SPACING);
            return LocatedCell{.page = std::move(page),
                               .cell = static_cast<size_t>(indexOf(rasterX, rasterZ))};
        };

        std::set<std::tuple<uint16_t, int64_t, int64_t>> frontier;
        std::unordered_map<WetlandNode, uint16_t, WetlandNodeHash> distances;
        frontier.emplace(0, source.x, source.z);
        distances.emplace(source, 0);
        std::optional<EstuaryResolution> resolution;
        while (!frontier.empty()) {
            const auto [distance, cellX, cellZ] = *frontier.begin();
            frontier.erase(frontier.begin());
            const WetlandNode node{.x = cellX, .z = cellZ};
            const auto current = distances.find(node);
            if (current == distances.end() || current->second != distance) continue;
            const LocatedCell located = locate(node);
            const uint8_t flags = located.page->flags[located.cell];
            if ((flags & CELL_OCEAN) != 0) {
                resolution = EstuaryResolution{.oceanBody = located.page->oceanId,
                                               .distanceToOcean = distance};
                break;
            }
            if (distance >= ESTUARY_MAXIMUM_BACKWATER_NATIVE_CELLS || (flags & CELL_RIVER) == 0 ||
                (flags & CELL_WATERFALL) != 0 ||
                located.page->channelGradient[located.cell] > ESTUARY_MAXIMUM_CHANNEL_GRADIENT) {
                continue;
            }
            const std::array<int32_t, 2> targets{
                located.page->receiverFirst[located.cell],
                located.page->receiverSecond[located.cell],
            };
            const std::array<float, 2> weights{
                targets[1] >= 0 ? 1.0F - located.page->receiverSecondWeight[located.cell] : 1.0F,
                targets[1] >= 0 ? located.page->receiverSecondWeight[located.cell] : 0.0F,
            };
            for (size_t branch = 0; branch < targets.size(); ++branch) {
                if (targets[branch] < 0 || weights[branch] < MINIMUM_CHANNEL_BRANCH_WEIGHT)
                    continue;
                const int targetRasterX = targets[branch] % RASTER_EDGE;
                const int targetRasterZ = targets[branch] / RASTER_EDGE;
                const WetlandNode target{
                    .x = checkedCoordinate(located.page->key.x, targetRasterX - RASTER_APRON),
                    .z = checkedCoordinate(located.page->key.z, targetRasterZ - RASTER_APRON),
                };
                const uint16_t targetDistance = static_cast<uint16_t>(distance + 1U);
                const auto [found, inserted] = distances.emplace(target, targetDistance);
                if (!inserted && found->second <= targetDistance) continue;
                found->second = targetDistance;
                frontier.emplace(targetDistance, target.x, target.z);
            }
        }

        {
            std::lock_guard lock(mutex);
            if (seaBackwaterResolutions.size() >= MAXIMUM_CONNECTED_WETLAND_CELLS)
                seaBackwaterResolutions.clear();
            seaBackwaterResolutions.insert_or_assign(source, resolution);
        }
        return resolution;
    }

    template <typename PageResolver>
    std::optional<BasinSample>
    resolveOpenLakeAtPoint(const std::shared_ptr<const NativePage>& ownerPage,
                           const BasinSample& ownerSample, double x, double z,
                           PageResolver& resolvePage) const {
        std::vector<std::shared_ptr<const ResolvedOpenLakeRegion>> associatedRegions;
        {
            std::lock_guard lock(mutex);
            if (auto found = openLakeRegionsByPage.find(ownerPage->key);
                found != openLakeRegionsByPage.end()) {
                for (auto iterator = found->second.begin(); iterator != found->second.end();) {
                    if (std::shared_ptr<const ResolvedOpenLakeRegion> region = iterator->lock()) {
                        associatedRegions.push_back(std::move(region));
                        ++metrics.openDepressionHits;
                        ++iterator;
                    } else {
                        iterator = found->second.erase(iterator);
                    }
                }
                if (found->second.empty()) openLakeRegionsByPage.erase(found);
            }
        }
        std::optional<BasinSample> selected;
        const auto considerRegion = [&](const ResolvedOpenLakeRegion& region) {
            const std::optional<BasinSample> candidate =
                sampleResolvedOpenLake(ownerSample, region, x, z);
            if (!candidate) return;
            if (!selected || candidate->lakeShoreDistance > selected->lakeShoreDistance + 1.0e-6 ||
                (std::abs(candidate->lakeShoreDistance - selected->lakeShoreDistance) <= 1.0e-6 &&
                 candidate->waterBodyId < selected->waterBodyId)) {
                selected = candidate;
            }
        };
        for (const std::shared_ptr<const ResolvedOpenLakeRegion>& region : associatedRegions)
            considerRegion(*region);
        if (selected) return selected;

        if (ownerSample.lake && ownerSample.waterBodyId != NO_WATER_BODY) {
            const DepressionSummary* local = depressionSummary(*ownerPage, ownerSample.waterBodyId);
            if (local == nullptr || local->edgeMask == 0) return std::nullopt;
        } else {
            const double localX = x - static_cast<double>(ownerPage->originX);
            const double localZ = z - static_cast<double>(ownerPage->originZ);
            const bool nearBoundary =
                localX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                NATIVE_HYDROLOGY_PAGE_EDGE - localX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                localZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                NATIVE_HYDROLOGY_PAGE_EDGE - localZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS;
            if (!nearBoundary &&
                std::ranges::none_of(
                    ownerPage->depressionSummaries,
                    [](const DepressionSummary& summary) { return summary.edgeMask != 0; }) &&
                ownerPage->openSpillReceivingEdgeMask == 0) {
                return std::nullopt;
            }
        }

        std::map<LakeNode, std::shared_ptr<const NativePage>> sources;
        for (const DepressionSummary& summary : ownerPage->depressionSummaries) {
            if (summary.edgeMask != 0)
                sources.emplace(LakeNode{.page = ownerPage->key, .body = summary.waterBodyId},
                                ownerPage);
        }

        constexpr std::array<std::tuple<int64_t, int64_t, PageEdge, PageEdge>, 4> NEIGHBORS{{
            {-1, 0, PageEdge::WEST, PageEdge::EAST},
            {1, 0, PageEdge::EAST, PageEdge::WEST},
            {0, -1, PageEdge::NORTH, PageEdge::SOUTH},
            {0, 1, PageEdge::SOUTH, PageEdge::NORTH},
        }};
        const double ownerLocalX = x - static_cast<double>(ownerPage->originX);
        const double ownerLocalZ = z - static_cast<double>(ownerPage->originZ);
        const bool ownerHasSource = !sources.empty();
        for (const auto [deltaX, deltaZ, ownerEdge, facingEdge] : NEIGHBORS) {
            const bool receivingEdge = (ownerPage->openSpillReceivingEdgeMask &
                                        (1U << static_cast<uint8_t>(ownerEdge))) != 0;
            const bool queryNearEdge =
                (deltaX < 0 && ownerLocalX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS) ||
                (deltaX > 0 &&
                 NATIVE_HYDROLOGY_PAGE_EDGE - ownerLocalX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS) ||
                (deltaZ < 0 && ownerLocalZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS) ||
                (deltaZ > 0 &&
                 NATIVE_HYDROLOGY_PAGE_EDGE - ownerLocalZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS);
            const bool relevantBoundary = receivingEdge || (!ownerHasSource && queryNearEdge);
            if (!relevantBoundary) continue;
            const __int128 neighborX = static_cast<__int128>(ownerPage->key.x) + deltaX;
            const __int128 neighborZ = static_cast<__int128>(ownerPage->key.z) + deltaZ;
            if (neighborX < std::numeric_limits<int64_t>::min() ||
                neighborX > std::numeric_limits<int64_t>::max() ||
                neighborZ < std::numeric_limits<int64_t>::min() ||
                neighborZ > std::numeric_limits<int64_t>::max()) {
                continue;
            }
            const std::shared_ptr<const NativePage> neighbor = resolvePage(
                {.x = static_cast<int64_t>(neighborX), .z = static_cast<int64_t>(neighborZ)});
            const uint8_t facingBit = 1U << static_cast<uint8_t>(facingEdge);
            for (const DepressionSummary& summary : neighbor->depressionSummaries) {
                if ((summary.edgeMask & facingBit) != 0) {
                    sources.emplace(LakeNode{.page = neighbor->key, .body = summary.waterBodyId},
                                    neighbor);
                }
            }
        }
        if (sources.empty()) return std::nullopt;

        std::vector<std::shared_ptr<const ResolvedOpenLakeRegion>> resolvedRegions;
        for (const auto& [node, page] : sources) {
            const DepressionSummary* sourceSummary = depressionSummary(*page, node.body);
            if (sourceSummary == nullptr)
                throw std::runtime_error("native hydrology open-depression source disappeared");
            const bool alreadyResolved =
                std::ranges::any_of(resolvedRegions, [sourceSummary](const auto& region) {
                    const std::optional<BasinSample> anchor = sampleResolvedOpenLake(
                        {}, *region, static_cast<double>(sourceSummary->localAnchorX),
                        static_cast<double>(sourceSummary->localAnchorZ));
                    return anchor && anchor->lake;
                });
            if (alreadyResolved) continue;
            const std::shared_ptr<const ResolvedOpenLakeRegion> region =
                openLakeRegion(page, node.body, resolvePage);
            if (!region) continue;
            considerRegion(*region);
            resolvedRegions.push_back(region);
        }
        return selected;
    }

    template <typename PageResolver>
    BasinSample sampleWithPages(double x, double z, PageResolver&& resolvePage,
                                bool reconcileOrdinaryStage = true) const {
        const PageKey owner = pageKeyAt(x, z);
        const int64_t originX = checkedCoordinate(owner.x, 0);
        const int64_t originZ = checkedCoordinate(owner.z, 0);
        std::optional<int64_t> adjacentX;
        std::optional<int64_t> adjacentZ;
        std::optional<int64_t> verticalBoundary;
        std::optional<int64_t> horizontalBoundary;
        if (x - static_cast<double>(originX) <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS &&
            owner.x > std::numeric_limits<int64_t>::min()) {
            adjacentX = owner.x - 1;
            verticalBoundary = originX;
        } else if (static_cast<double>(originX) + NATIVE_HYDROLOGY_PAGE_EDGE - x <=
                       NATIVE_HYDROLOGY_HANDOFF_BLOCKS &&
                   owner.x < std::numeric_limits<int64_t>::max()) {
            adjacentX = owner.x + 1;
            verticalBoundary = checkedCoordinate(owner.x + 1, 0);
        }
        if (z - static_cast<double>(originZ) <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS &&
            owner.z > std::numeric_limits<int64_t>::min()) {
            adjacentZ = owner.z - 1;
            horizontalBoundary = originZ;
        } else if (static_cast<double>(originZ) + NATIVE_HYDROLOGY_PAGE_EDGE - z <=
                       NATIVE_HYDROLOGY_HANDOFF_BLOCKS &&
                   owner.z < std::numeric_limits<int64_t>::max()) {
            adjacentZ = owner.z + 1;
            horizontalBoundary = checkedCoordinate(owner.z + 1, 0);
        }

        const std::shared_ptr<const NativePage> ownerPage = resolvePage(owner);
        const NativeStagePageResolver stagePageResolver = [&](PageKey key) {
            return resolvePage(key);
        };
        const BasinSample ownerSample =
            samplePage(*ownerPage, x, z, &stagePageResolver, reconcileOrdinaryStage);
        BasinSample resolvedOwner = ownerSample;
        if (const std::optional<BasinSample> openLake =
                resolveOpenLakeAtPoint(ownerPage, ownerSample, x, z, resolvePage)) {
            if (openLake->lake) return *openLake;
            if (!ownerSample.lake) resolvedOwner = *openLake;
        }
        if (resolvedOwner.lake && resolvedOwner.waterBodyId != NO_WATER_BODY) {
            if (const std::optional<TiledLakeResolution> resolution =
                    resolveLakeSpillHierarchy(ownerPage, resolvedOwner.waterBodyId, resolvePage)) {
                return applyTiledLakeResolution(resolvedOwner, *resolution);
            }
            // The half-open owner remains authoritative when no opposing wet
            // portal proves a larger component. An apron reconstruction from
            // another page cannot turn that lake into ocean or dry terrain.
            return resolvedOwner;
        }
        if (resolvedOwner.river && !resolvedOwner.estuary) {
            if (const std::optional<EstuaryResolution> estuary =
                    resolveSeaBackwater(ownerPage, x, z, resolvePage)) {
                resolvedOwner = applyEstuaryResolution(resolvedOwner, *estuary);
            }
        }
        if (!resolvedOwner.ocean && !resolvedOwner.lake && !resolvedOwner.river &&
            !resolvedOwner.wetland && !resolvedOwner.waterfall) {
            if (const std::optional<WetlandResolution> wetland =
                    resolveConnectedWetland(ownerPage, x, z, resolvePage)) {
                return applyWetlandResolution(resolvedOwner, *wetland);
            }
        }
        std::set<PageKey> closure;
        closure.insert(owner);
        if (adjacentX) closure.insert({*adjacentX, owner.z});
        if (adjacentZ) closure.insert({owner.x, *adjacentZ});
        if (adjacentX && adjacentZ) closure.insert({*adjacentX, *adjacentZ});
        if (closure.size() == 1) return resolvedOwner;
        if (closure.size() > NATIVE_HYDROLOGY_MAX_HANDOFF_PAGES)
            throw std::runtime_error("native hydrology lake closure exceeded its page bound");

        std::vector<HandoffCandidate> candidates;
        candidates.reserve(closure.size());
        candidates.push_back({.page = ownerPage, .sample = resolvedOwner, .containsQuery = true});
        for (const PageKey key : closure) {
            if (key == owner) continue;
            std::shared_ptr<const NativePage> page = resolvePage(key);
            const bool containsQuery = pageRasterContains(*page, x, z);
            candidates.push_back({
                .page = page,
                .sample = containsQuery
                              ? samplePage(*page, x, z, &stagePageResolver, reconcileOrdinaryStage)
                              : BasinSample{},
                .containsQuery = containsQuery,
            });
        }
        BasinSample result =
            reconcileHandoff(seed, candidates, x, z, verticalBoundary, horizontalBoundary);
        if (result.river && !result.estuary) {
            if (const std::optional<EstuaryResolution> estuary =
                    resolveSeaBackwater(ownerPage, x, z, resolvePage)) {
                result = applyEstuaryResolution(result, *estuary);
            }
        }
        return result;
    }

    BasinSample sample(double x, double z, const NativeHydrologyInputFunction& input,
                       bool* certifiedDryHit, learned::AuthorityRequestPriority priority) const {
        const auto footprint = certifiedDrySnapshot();
        if (const BasinSample* certified =
                findCertifiedDrySample(footprint, BasinSamplePosition{.x = x, .z = z})) {
            if (certifiedDryHit) *certifiedDryHit = true;
            return *certified;
        }
        if (certifiedDryHit) *certifiedDryHit = false;
        return sampleWithPages(x, z,
                               [&](PageKey key) { return getOrCreate(key, input, priority); });
    }

    void prepareOwner(int64_t ownerPageX, int64_t ownerPageZ,
                      const NativeHydrologyInputFunction& input,
                      learned::AuthorityRequestPriority priority) const {
        const PageKey owner{.x = ownerPageX, .z = ownerPageZ};
        // Validate both axes before entering the page flight. getOrCreate then
        // remains the only operation, so direct readiness cannot accidentally
        // construct an open-depression neighbor or other semantic closure.
        static_cast<void>(checkedCoordinate(owner.x, 0));
        static_cast<void>(checkedCoordinate(owner.z, 0));
        static_cast<void>(getOrCreate(owner, input, priority));
    }

    std::shared_ptr<const NativeDryLocalityProof>
    dryLocalityProof(const std::shared_ptr<const NativePage>& page) const {
        const auto cached = [&]() -> std::shared_ptr<const NativeDryLocalityProof> {
            std::lock_guard lock(mutex);
            const auto found = entries.find(page->key);
            if (found == entries.end() || found->second.page != page) return nullptr;
            return found->second.dryProof;
        };
        if (std::shared_ptr<const NativeDryLocalityProof> proof = cached()) return proof;

        // Candidate screens for one owner share this immutable derived proof.
        // Build outside the page cache lock so unrelated sampling continues.
        std::lock_guard proofBuildLock(dryProofBuildMutex);
        if (std::shared_ptr<const NativeDryLocalityProof> proof = cached()) return proof;
        auto built =
            std::make_shared<const NativeDryLocalityProof>(buildNativeDryLocalityProof(*page));

        std::lock_guard lock(mutex);
        const auto found = entries.find(page->key);
        if (found == entries.end() || found->second.page != page) return built;
        if (found->second.dryProof) return found->second.dryProof;
        found->second.dryProof = built;
        found->second.bytes += built->byteSize();
        metrics.bytes += built->byteSize();
        const learned::AuthorityRequestPriority retainedPriority = found->second.priority;
        while (metrics.bytes > cacheByteBudget && !recency.empty()) {
            if (!evictPageFor(retainedPriority)) break;
        }
        metrics.entries = entries.size();
        return built;
    }

    void certifyDryPoints(std::span<const BasinSamplePosition> positions,
                          const NativeHydrologyInputFunction& input, std::span<BasinSample> output,
                          std::span<uint8_t> certified,
                          learned::AuthorityRequestPriority priority) const {
        if (positions.size() != output.size() || positions.size() != certified.size() ||
            positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
            throw std::invalid_argument("invalid native hydrology dry certificate batch");
        }
        std::fill(output.begin(), output.end(), BasinSample{});
        std::fill(certified.begin(), certified.end(), uint8_t{0});
        if (positions.empty()) return;

        const PageKey owner = pageKeyAt(positions.front().x, positions.front().z);
        const long double ownerOriginX = checkedCoordinate(owner.x, 0);
        const long double ownerOriginZ = checkedCoordinate(owner.z, 0);
        const long double ownerEndX = ownerOriginX + NATIVE_HYDROLOGY_PAGE_EDGE;
        const long double ownerEndZ = ownerOriginZ + NATIVE_HYDROLOGY_PAGE_EDGE;
        for (const BasinSamplePosition position : positions) {
            if (pageKeyAt(position.x, position.z) != owner) return;
            const long double x = position.x;
            const long double z = position.z;
            if (x - ownerOriginX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                ownerEndX - x <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                z - ownerOriginZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
                ownerEndZ - z <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS) {
                return;
            }
        }
        {
            std::lock_guard lock(mutex);
            const uint64_t sampleCount = static_cast<uint64_t>(positions.size());
            if (std::numeric_limits<uint64_t>::max() - metrics.dryCertificateSamples <
                sampleCount) {
                metrics.dryCertificateSamples = std::numeric_limits<uint64_t>::max();
            } else {
                metrics.dryCertificateSamples += sampleCount;
            }
        }

        const std::shared_ptr<const NativePage> page = getOrCreate(owner, input, priority);
        const std::shared_ptr<const NativeDryLocalityProof> proof = dryLocalityProof(page);
        if (!proof->complete) return;
        for (size_t index = 0; index < positions.size(); ++index) {
            const std::optional<BasinSample> sample =
                certifyNativeDryPoint(*page, *proof, positions[index].x, positions[index].z);
            if (!sample) continue;
            output[index] = *sample;
            certified[index] = 1;
        }
    }

    bool certifyDryFootprint(std::span<const BasinSamplePosition> positions,
                             const NativeHydrologyInputFunction& input,
                             std::vector<NativeHydrologyPosition>& integerPositions,
                             std::vector<BasinSample>& samples,
                             learned::AuthorityRequestPriority priority) const {
        integerPositions.clear();
        samples.clear();
        if (positions.empty() || positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
            return false;
        }
        integerPositions.reserve(positions.size());
        for (const BasinSamplePosition position : positions) {
            const std::optional<NativeHydrologyPosition> integer =
                exactIntegerHydrologyPosition(position);
            if (!integer) {
                integerPositions.clear();
                return false;
            }
            integerPositions.push_back(*integer);
        }
        std::vector<NativeHydrologyPosition> uniquePositions = integerPositions;
        std::ranges::sort(uniquePositions);
        if (std::ranges::adjacent_find(uniquePositions) != uniquePositions.end()) {
            integerPositions.clear();
            return false;
        }

        samples.resize(positions.size());
        std::vector<uint8_t> certified(positions.size());
        certifyDryPoints(positions, input, samples, certified, priority);
        if (!std::ranges::all_of(certified, [](uint8_t value) { return value != 0; })) {
            integerPositions.clear();
            samples.clear();
            return false;
        }
        return true;
    }

    bool replaceCertifiedDryFootprint(std::span<const NativeHydrologyPosition> positions,
                                      std::span<const BasinSample> samples) const {
        if (positions.empty() || positions.size() != samples.size() ||
            positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
            return false;
        }
        auto replacement = std::make_shared<std::vector<CertifiedDrySample>>();
        replacement->reserve(positions.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            if (!nativeDrySampleIsInstallable(samples[index])) return false;
            replacement->push_back({.position = positions[index], .sample = samples[index]});
        }
        std::ranges::sort(*replacement, {}, &CertifiedDrySample::position);
        if (std::ranges::adjacent_find(*replacement, {}, &CertifiedDrySample::position) !=
            replacement->end()) {
            return false;
        }

        std::lock_guard lock(mutex);
        certifiedDryFootprint = std::move(replacement);
        return true;
    }

    std::shared_ptr<const std::vector<CertifiedDrySample>> certifiedDrySnapshot() const {
        std::lock_guard lock(mutex);
        return certifiedDryFootprint;
    }

    bool hasCertifiedDryFootprint() const {
        std::lock_guard lock(mutex);
        return certifiedDryFootprint && !certifiedDryFootprint->empty();
    }

    static const BasinSample*
    findCertifiedDrySample(const std::shared_ptr<const std::vector<CertifiedDrySample>>& footprint,
                           BasinSamplePosition position) {
        if (!footprint) return nullptr;
        const std::optional<NativeHydrologyPosition> integer =
            exactIntegerHydrologyPosition(position);
        if (!integer) return nullptr;
        const auto found = std::lower_bound(footprint->begin(), footprint->end(), *integer,
                                            certifiedDrySampleLess);
        return found != footprint->end() && found->position == *integer ? &found->sample : nullptr;
    }

    bool certifiedDryFootprintContains(std::span<const BasinSamplePosition> positions) const {
        if (positions.empty() || positions.size() > NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES) {
            return false;
        }
        const auto footprint = certifiedDrySnapshot();
        return std::ranges::all_of(positions, [&](BasinSamplePosition position) {
            return findCertifiedDrySample(footprint, position) != nullptr;
        });
    }

    void clearCertifiedDryFootprint() const {
        std::lock_guard lock(mutex);
        certifiedDryFootprint.reset();
    }

    // A far-terrain water raster normally lies entirely inside one immutable
    // hydrology owner. Its regular points still used to enter samplePoints,
    // allocating a position vector and resolving the same owner through a
    // hash map for every point. Keep the generic path at every handoff, but
    // take the direct owner path when the complete rectangle is safely inside
    // one page. samplePage remains the single source of authority, so this
    // only removes lookup work.
    bool sampleInteriorGrid(int64_t originX, int64_t originZ, int64_t lastX, int64_t lastZ,
                            int spacingX, int spacingZ, int sampleWidth, int sampleHeight,
                            const NativeHydrologyInputFunction& input,
                            std::span<BasinSample> output,
                            learned::AuthorityRequestPriority priority) const {
        const double firstX = static_cast<double>(originX);
        const double firstZ = static_cast<double>(originZ);
        const double finalX = static_cast<double>(lastX);
        const double finalZ = static_cast<double>(lastZ);
        const PageKey owner = pageKeyAt(firstX, firstZ);
        if (pageKeyAt(finalX, finalZ) != owner) return false;

        const double ownerOriginX = static_cast<double>(checkedCoordinate(owner.x, 0));
        const double ownerOriginZ = static_cast<double>(checkedCoordinate(owner.z, 0));
        const double ownerEndX = ownerOriginX + NATIVE_HYDROLOGY_PAGE_EDGE;
        const double ownerEndZ = ownerOriginZ + NATIVE_HYDROLOGY_PAGE_EDGE;
        if (firstX - ownerOriginX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
            ownerEndX - finalX <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
            firstZ - ownerOriginZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS ||
            ownerEndZ - finalZ <= NATIVE_HYDROLOGY_HANDOFF_BLOCKS) {
            return false;
        }

        const std::shared_ptr<const NativePage> page = getOrCreate(owner, input, priority);
        bool openLakePossible = std::ranges::any_of(page->depressionSummaries,
                                                    [](const DepressionSummary& summary) {
                                                        return summary.edgeMask != 0;
                                                    }) ||
                                page->openSpillReceivingEdgeMask != 0;
        if (!openLakePossible) {
            std::lock_guard lock(mutex);
            if (const auto found = openLakeRegionsByPage.find(owner);
                found != openLakeRegionsByPage.end()) {
                openLakePossible = std::ranges::any_of(
                    found->second, [](const auto& weak) { return !weak.expired(); });
            }
        }
        if (openLakePossible) {
            // Any sample in an edge-connected local depression needs the same
            // tiled minimax closure as a scalar query. This includes a dry
            // owner that receives a one-sided depression from its neighbor.
            // The generic batch path retains those pages for the whole mesh
            // request.
            return false;
        }
        const bool nativeLattice =
            spacingX % NATIVE_HYDROLOGY_RASTER_SPACING == 0 &&
            spacingZ % NATIVE_HYDROLOGY_RASTER_SPACING == 0 &&
            (originX - page->originX) % NATIVE_HYDROLOGY_RASTER_SPACING == 0 &&
            (originZ - page->originZ) % NATIVE_HYDROLOGY_RASTER_SPACING == 0;
        const int latticeOriginX =
            nativeLattice ? RASTER_APRON + static_cast<int>((originX - page->originX) /
                                                            NATIVE_HYDROLOGY_RASTER_SPACING)
                          : 0;
        const int latticeOriginZ =
            nativeLattice ? RASTER_APRON + static_cast<int>((originZ - page->originZ) /
                                                            NATIVE_HYDROLOGY_RASTER_SPACING)
                          : 0;
        const int latticeStepX = nativeLattice ? spacingX / NATIVE_HYDROLOGY_RASTER_SPACING : 0;
        const int latticeStepZ = nativeLattice ? spacingZ / NATIVE_HYDROLOGY_RASTER_SPACING : 0;
        const auto resolvePage = [&](PageKey key) { return getOrCreate(key, input, priority); };
        const NativeStagePageResolver stagePageResolver = [&](PageKey key) {
            return resolvePage(key);
        };
        // The graph-minorant stage tile is block-column authority. Coarser
        // terrain/water grids consume native topology and raw routed stages;
        // building thousands of one-block halo samples for a sparse parent
        // would make LOD work scale like exact chunk generation.
        const bool reconcileOrdinaryStage = spacingX == 1 && spacingZ == 1;
        for (int sampleZ = 0; sampleZ < sampleHeight; ++sampleZ) {
            const double worldZ = static_cast<double>(static_cast<int64_t>(
                static_cast<__int128>(originZ) + static_cast<__int128>(sampleZ) * spacingZ));
            for (int sampleX = 0; sampleX < sampleWidth; ++sampleX) {
                const double worldX = static_cast<double>(static_cast<int64_t>(
                    static_cast<__int128>(originX) + static_cast<__int128>(sampleX) * spacingX));
                BasinSample sample =
                    nativeLattice
                        ? sampleNativeLatticeCell(*page, worldX, worldZ,
                                                  latticeOriginX + sampleX * latticeStepX,
                                                  latticeOriginZ + sampleZ * latticeStepZ,
                                                  &stagePageResolver, reconcileOrdinaryStage)
                        : samplePage(*page, worldX, worldZ, &stagePageResolver,
                                     reconcileOrdinaryStage);
                const int sampleRasterX =
                    nativeLattice
                        ? latticeOriginX + sampleX * latticeStepX
                        : std::clamp(static_cast<int>(std::floor(
                                         (worldX - static_cast<double>(page->originX)) /
                                             NATIVE_HYDROLOGY_RASTER_SPACING +
                                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET + 0.5)),
                                     0, RASTER_EDGE - 1);
                const int sampleRasterZ =
                    nativeLattice
                        ? latticeOriginZ + sampleZ * latticeStepZ
                        : std::clamp(static_cast<int>(std::floor(
                                         (worldZ - static_cast<double>(page->originZ)) /
                                             NATIVE_HYDROLOGY_RASTER_SPACING +
                                         RASTER_APRON + WORLD_SAMPLE_NATIVE_OFFSET + 0.5)),
                                     0, RASTER_EDGE - 1);
                const uint8_t flags =
                    page->flags[static_cast<size_t>(indexOf(sampleRasterX, sampleRasterZ))];
                if (page->hasWetlandCandidates && (flags & CELL_WETLAND_CANDIDATE) != 0 &&
                    !sample.ocean && !sample.lake && !sample.river && !sample.wetland &&
                    !sample.waterfall) {
                    if (const std::optional<WetlandResolution> wetland =
                            resolveConnectedWetland(page, worldX, worldZ, resolvePage)) {
                        sample = applyWetlandResolution(sample, *wetland);
                    }
                }
                if (page->hasCoastalResolutionCandidates && sample.river && !sample.estuary) {
                    if (const std::optional<EstuaryResolution> estuary =
                            resolveSeaBackwater(page, worldX, worldZ, resolvePage)) {
                        sample = applyEstuaryResolution(sample, *estuary);
                    }
                }
                output[static_cast<size_t>(sampleZ) * static_cast<size_t>(sampleWidth) +
                       static_cast<size_t>(sampleX)] = std::move(sample);
            }
        }
        if (output.size() > 1) {
            std::lock_guard lock(mutex);
            metrics.batchPageReuses += output.size() - 1;
        }
        return true;
    }

    void sampleTopologyGrid(int64_t originX, int64_t originZ, int cellWidth, int cellHeight,
                            const NativeHydrologyInputFunction& input,
                            std::span<NativeHydrologyTopologyCell> output,
                            std::span<uint8_t> certifiedDryHits,
                            learned::AuthorityRequestPriority priority) const {
        if (cellWidth <= 0 || cellHeight <= 0 ||
            output.size() != static_cast<size_t>(cellWidth) * static_cast<size_t>(cellHeight) ||
            (!certifiedDryHits.empty() && certifiedDryHits.size() != output.size()) ||
            originX % NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE != 0 ||
            originZ % NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE != 0) {
            throw std::invalid_argument("invalid native hydrology topology grid");
        }
        if (!certifiedDryHits.empty())
            std::fill(certifiedDryHits.begin(), certifiedDryHits.end(), uint8_t{0});
        const auto checkedCoordinate = [](int64_t origin, int index) {
            const __int128 value =
                static_cast<__int128>(origin) +
                static_cast<__int128>(index) * NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
            if (value < std::numeric_limits<int64_t>::min() ||
                value > std::numeric_limits<int64_t>::max()) {
                throw std::invalid_argument("native hydrology topology coordinate is out of range");
            }
            return static_cast<int64_t>(value);
        };
        // Validate each half-open maximum as well as the first sample. This
        // prevents a successful final start coordinate from wrapping the
        // parent cell's exclusive edge at the int64 limit.
        static_cast<void>(checkedCoordinate(originX, cellWidth));
        static_cast<void>(checkedCoordinate(originZ, cellHeight));

        std::unordered_map<PageKey, std::shared_ptr<const NativePage>, PageKeyHash> pages;
        pages.reserve(std::min<size_t>(static_cast<size_t>(cellWidth) * cellHeight, 4));
        std::unordered_map<PageKey, std::vector<std::shared_ptr<const ResolvedOpenLakeRegion>>,
                           PageKeyHash>
            openRegions;
        const auto certifiedFootprint = certifiedDrySnapshot();
        const auto cellIsCertifiedDry = [&](int64_t worldX, int64_t worldZ) {
            if (!certifiedFootprint) return false;
            for (int rasterZ = 0; rasterZ < TOPOLOGY_CELL_RASTER_EDGE; ++rasterZ) {
                for (int rasterX = 0; rasterX < TOPOLOGY_CELL_RASTER_EDGE; ++rasterX) {
                    const BasinSamplePosition position{
                        .x =
                            static_cast<double>(worldX + rasterX * NATIVE_HYDROLOGY_RASTER_SPACING),
                        .z =
                            static_cast<double>(worldZ + rasterZ * NATIVE_HYDROLOGY_RASTER_SPACING),
                    };
                    if (!findCertifiedDrySample(certifiedFootprint, position)) return false;
                }
            }
            return true;
        };
        uint64_t localReuses = 0;
        const auto resolvePage = [&](PageKey key) -> std::shared_ptr<const NativePage> {
            if (const auto found = pages.find(key); found != pages.end()) {
                ++localReuses;
                return found->second;
            }
            const std::shared_ptr<const NativePage> page = getOrCreate(key, input, priority);
            return pages.emplace(key, page).first->second;
        };
        const auto resolveOpenRegions = [&](const std::shared_ptr<const NativePage>& page)
            -> const std::vector<std::shared_ptr<const ResolvedOpenLakeRegion>>& {
            if (const auto found = openRegions.find(page->key); found != openRegions.end())
                return found->second;
            const bool possible = page->openSpillReceivingEdgeMask != 0 ||
                                  std::ranges::any_of(page->depressionSummaries,
                                                      [](const DepressionSummary& summary) {
                                                          return summary.edgeMask != 0;
                                                      });
            if (possible) {
                // Build or reuse the same bounded closure used by exact
                // sampling. The center is only a trigger; every topology
                // cell below reads the associated immutable region directly.
                const double centerX =
                    static_cast<double>(page->originX) + NATIVE_HYDROLOGY_PAGE_EDGE * 0.5;
                const double centerZ =
                    static_cast<double>(page->originZ) + NATIVE_HYDROLOGY_PAGE_EDGE * 0.5;
                static_cast<void>(sampleWithPages(centerX, centerZ, resolvePage));
            }
            std::vector<std::shared_ptr<const ResolvedOpenLakeRegion>> associated;
            {
                std::lock_guard lock(mutex);
                if (auto found = openLakeRegionsByPage.find(page->key);
                    found != openLakeRegionsByPage.end()) {
                    for (auto iterator = found->second.begin(); iterator != found->second.end();) {
                        if (std::shared_ptr<const ResolvedOpenLakeRegion> region =
                                iterator->lock()) {
                            associated.push_back(std::move(region));
                            ++metrics.openDepressionHits;
                            ++iterator;
                        } else {
                            iterator = found->second.erase(iterator);
                        }
                    }
                    if (found->second.empty()) openLakeRegionsByPage.erase(found);
                }
            }
            return openRegions.emplace(page->key, std::move(associated)).first->second;
        };

        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            const int64_t worldZ = checkedCoordinate(originZ, cellZ);
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t worldX = checkedCoordinate(originX, cellX);
                const size_t outputIndex =
                    static_cast<size_t>(cellZ) * static_cast<size_t>(cellWidth) +
                    static_cast<size_t>(cellX);
                // The installed startup certificate is opaque evidence that
                // every canonical four-block sample in this half-open cell is
                // dry and has no legal fall. Do not let an unrelated open
                // depression elsewhere in the owner expand this local proof
                // into a cardinal-page closure.
                if (cellIsCertifiedDry(worldX, worldZ)) {
                    output[outputIndex] = {};
                    if (!certifiedDryHits.empty()) certifiedDryHits[outputIndex] = 1;
                    continue;
                }
                const PageKey key =
                    pageKeyAt(static_cast<double>(worldX), static_cast<double>(worldZ));
                const std::shared_ptr<const NativePage> page = resolvePage(key);
                const int64_t localX = worldX - page->originX;
                const int64_t localZ = worldZ - page->originZ;
                if (localX < 0 || localZ < 0 || localX % NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE != 0 ||
                    localZ % NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE != 0) {
                    throw std::runtime_error("native hydrology topology owner is misaligned");
                }
                const int topologyX =
                    static_cast<int>(localX / NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
                const int topologyZ =
                    static_cast<int>(localZ / NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
                if (topologyX < 0 || topologyX >= TOPOLOGY_EDGE || topologyZ < 0 ||
                    topologyZ >= TOPOLOGY_EDGE ||
                    page->topologyCells.size() != static_cast<size_t>(TOPOLOGY_CELLS)) {
                    throw std::runtime_error("native hydrology topology page is invalid");
                }
                uint8_t topology =
                    page->topologyCells[static_cast<size_t>(topologyZ * TOPOLOGY_EDGE + topologyX)];
                for (const std::shared_ptr<const ResolvedOpenLakeRegion>& region :
                     resolveOpenRegions(page)) {
                    for (int rasterZ = 0; rasterZ < TOPOLOGY_CELL_RASTER_EDGE; ++rasterZ) {
                        const double sampleZ =
                            static_cast<double>(worldZ + rasterZ * NATIVE_HYDROLOGY_RASTER_SPACING);
                        for (int rasterX = 0; rasterX < TOPOLOGY_CELL_RASTER_EDGE; ++rasterX) {
                            const double sampleX = static_cast<double>(
                                worldX + rasterX * NATIVE_HYDROLOGY_RASTER_SPACING);
                            const std::optional<BasinSample> sample =
                                sampleResolvedOpenLake({}, *region, sampleX, sampleZ);
                            if (!sample) continue;
                            topology |= sample->lake ? TOPOLOGY_WET : TOPOLOGY_DRY;
                            // A reconciled open basin can replace a local
                            // body's stage or identity without introducing a
                            // dry native sample. Keep that parent on the
                            // canonical raster path instead of treating it as
                            // proven uniform standing water.
                            topology |= TOPOLOGY_AUTHORITY_DISCONTINUITY;
                        }
                    }
                }
                output[outputIndex] = {
                    .waterTopologyPossible =
                        ((topology & (TOPOLOGY_WET | TOPOLOGY_DRY)) ==
                         (TOPOLOGY_WET | TOPOLOGY_DRY)) ||
                        (topology & (TOPOLOGY_CHANNEL | TOPOLOGY_AUTHORITY_DISCONTINUITY)) != 0,
                    .waterfallPossible = (topology & TOPOLOGY_WATERFALL) != 0,
                };
            }
        }
        if (localReuses != 0) {
            std::lock_guard lock(mutex);
            metrics.batchPageReuses += localReuses;
        }
    }

    void samplePoints(std::span<const BasinSamplePosition> positions,
                      const NativeHydrologyInputFunction& input, std::span<BasinSample> output,
                      std::span<uint8_t> certifiedDryHits, bool reconcileOrdinaryStage,
                      learned::AuthorityRequestPriority priority) const {
        // Terrain and far-water meshes ask for dense grids. Resolving each
        // sample through the global LRU turned one page into hundreds of
        // cache locks and, under horizon pressure, repeated disk loads. Keep
        // the immutable pages alive for the duration of this bounded query.
        std::unordered_map<PageKey, std::shared_ptr<const NativePage>, PageKeyHash> pages;
        pages.reserve(std::min<size_t>(positions.size(), 16));
        const auto footprint = certifiedDrySnapshot();
        if (!certifiedDryHits.empty())
            std::fill(certifiedDryHits.begin(), certifiedDryHits.end(), uint8_t{0});
        uint64_t localReuses = 0;
        const auto resolvePage = [&](PageKey key) -> std::shared_ptr<const NativePage> {
            if (const auto found = pages.find(key); found != pages.end()) {
                ++localReuses;
                return found->second;
            }
            const std::shared_ptr<const NativePage> page = getOrCreate(key, input, priority);
            pages.emplace(key, page);
            return page;
        };
        for (size_t index = 0; index < positions.size(); ++index) {
            if (const BasinSample* certified =
                    findCertifiedDrySample(footprint, positions[index])) {
                output[index] = *certified;
                if (!certifiedDryHits.empty()) certifiedDryHits[index] = 1;
                continue;
            }
            output[index] = sampleWithPages(positions[index].x, positions[index].z, resolvePage,
                                            reconcileOrdinaryStage);
        }
        if (localReuses != 0) {
            std::lock_guard lock(mutex);
            metrics.batchPageReuses += localReuses;
        }
    }

    void clear() const {
        std::lock_guard lock(mutex);
        certifiedDryFootprint.reset();
        entries.clear();
        recency.clear();
        lakePairEntries.clear();
        lakePairRecency.clear();
        openLakeRegionEntries.clear();
        openLakeRegionRecency.clear();
        openLakeRegionsByPage.clear();
        connectedWetlandResolutions.clear();
        seaBackwaterResolutions.clear();
        openLakeRegionBytes = 0;
        metrics.entries = 0;
        metrics.bytes = 0;
        metrics.reconciliationEntries = 0;
        metrics.reconciliationBytes = 0;
        metrics.openDepressionEntries = 0;
        metrics.openDepressionBytes = 0;
        ++epoch;
    }

    NativeHydrologyCacheMetrics cacheMetrics() const {
        std::lock_guard lock(mutex);
        NativeHydrologyCacheMetrics result = metrics;
        result.connectedWetlandEntries = connectedWetlandResolutions.size();
        result.seaBackwaterEntries = seaBackwaterResolutions.size();
        for (const auto& [key, entry] : entries) {
            static_cast<void>(key);
            std::scoped_lock tileLock(entry.page->ordinaryStageTilesMutex);
            result.ordinaryStageTileEntries += entry.page->ordinaryStageTiles.size();
            result.ordinaryStageTileBytes += entry.page->ordinaryStageTileBytes;
            result.ordinaryStageTilePeakPageBytes = std::max(
                result.ordinaryStageTilePeakPageBytes, entry.page->ordinaryStageTilePeakBytes);
            result.ordinaryStageTileHits += entry.page->ordinaryStageTileHits;
            result.ordinaryStageTileMisses += entry.page->ordinaryStageTileMisses;
            result.ordinaryStageTileBuilds += entry.page->ordinaryStageTileBuilds;
            result.ordinaryStageTileFailures += entry.page->ordinaryStageTileFailures;
            result.ordinaryStageTileBuildNanoseconds +=
                entry.page->ordinaryStageTileBuildNanoseconds;
            result.ordinaryStageTileExpandedBuilds += entry.page->ordinaryStageTileExpandedBuilds;
            result.ordinaryStageTileBuildWaits += entry.page->ordinaryStageTileBuildWaits;
        }
        return result;
    }

    void recordOrdinaryStageCoarseGridSamples(size_t sampleCount) const {
        std::lock_guard lock(mutex);
        const uint64_t count = static_cast<uint64_t>(sampleCount);
        if (std::numeric_limits<uint64_t>::max() - metrics.ordinaryStageCoarseGridSamples < count) {
            metrics.ordinaryStageCoarseGridSamples = std::numeric_limits<uint64_t>::max();
        } else {
            metrics.ordinaryStageCoarseGridSamples += count;
        }
    }

private:
    struct Entry {
        std::shared_ptr<const NativePage> page;
        std::shared_ptr<const NativeDryLocalityProof> dryProof;
        size_t bytes = 0;
        learned::AuthorityRequestPriority priority =
            learned::AuthorityRequestPriority::EXPLORATION_EXACT;
        std::list<PageKey>::iterator recency;
    };

    struct Flight {
        std::condition_variable ready;
        std::atomic<learned::AuthorityRequestPriority> priority{
            learned::AuthorityRequestPriority::EXPLORATION_EXACT};
        bool complete = false;
        std::shared_ptr<const NativePage> page;
        std::exception_ptr failure;
    };

    uint64_t seed = 0;
    size_t cacheByteBudget = 0;
    std::shared_ptr<hydrology::HydrologyAuthorityStore> authorityStore;
    std::shared_ptr<const NativeHydrologyIdentityRegistry> identityRegistry;
    mutable std::mutex dryProofBuildMutex;
    mutable std::mutex mutex;
    mutable std::shared_ptr<const std::vector<CertifiedDrySample>> certifiedDryFootprint;
    mutable std::unordered_map<PageKey, Entry, PageKeyHash> entries;
    mutable std::unordered_map<PageKey, std::shared_ptr<Flight>, PageKeyHash> flights;
    mutable std::list<PageKey> recency;
    mutable std::map<LakePairKey, LakePairEntry> lakePairEntries;
    mutable std::list<LakePairKey> lakePairRecency;
    mutable std::map<LakeNode, OpenLakeRegionEntry> openLakeRegionEntries;
    mutable std::map<LakeNode, std::shared_ptr<OpenLakeRegionFlight>> openLakeRegionFlights;
    mutable std::list<LakeNode> openLakeRegionRecency;
    mutable std::map<PageKey, std::vector<std::weak_ptr<const ResolvedOpenLakeRegion>>>
        openLakeRegionsByPage;
    mutable std::unordered_map<WetlandNode, std::optional<WetlandResolution>, WetlandNodeHash>
        connectedWetlandResolutions;
    mutable std::unordered_map<WetlandNode, std::optional<EstuaryResolution>, WetlandNodeHash>
        seaBackwaterResolutions;
    mutable size_t openLakeRegionBytes = 0;
    mutable NativeHydrologyCacheMetrics metrics;
    mutable uint64_t epoch = 0;
};

NativeHydrologyRouter::NativeHydrologyRouter(uint64_t worldSeed, size_t cacheByteBudget)
    : NativeHydrologyRouter(worldSeed, nullptr,
                            std::make_shared<NativeHydrologyIdentityRegistry>(worldSeed),
                            cacheByteBudget) {}

NativeHydrologyRouter::NativeHydrologyRouter(
    uint64_t worldSeed, std::shared_ptr<hydrology::HydrologyAuthorityStore> authorityStore,
    size_t cacheByteBudget)
    : NativeHydrologyRouter(worldSeed, std::move(authorityStore),
                            std::make_shared<NativeHydrologyIdentityRegistry>(worldSeed),
                            cacheByteBudget) {}

NativeHydrologyRouter::NativeHydrologyRouter(
    uint64_t worldSeed, std::shared_ptr<hydrology::HydrologyAuthorityStore> authorityStore,
    std::shared_ptr<const NativeHydrologyIdentityRegistry> identityRegistry, size_t cacheByteBudget)
    : impl_(std::make_unique<Impl>(worldSeed, cacheByteBudget, std::move(authorityStore),
                                   std::move(identityRegistry))) {}

NativeHydrologyRouter::~NativeHydrologyRouter() = default;
NativeHydrologyRouter::NativeHydrologyRouter(NativeHydrologyRouter&&) noexcept = default;
NativeHydrologyRouter& NativeHydrologyRouter::operator=(NativeHydrologyRouter&&) noexcept = default;

BasinSample NativeHydrologyRouter::sample(double x, double z,
                                          const NativeHydrologyInputFunction& input,
                                          bool* certifiedDryHit,
                                          learned::AuthorityRequestPriority priority) const {
    return impl_->sample(x, z, input, certifiedDryHit, priority);
}

void NativeHydrologyRouter::prepareOwner(int64_t ownerPageX, int64_t ownerPageZ,
                                         const NativeHydrologyInputFunction& input,
                                         learned::AuthorityRequestPriority priority) const {
    impl_->prepareOwner(ownerPageX, ownerPageZ, input, priority);
}

void NativeHydrologyRouter::certifyDryPoints(std::span<const BasinSamplePosition> positions,
                                             const NativeHydrologyInputFunction& input,
                                             std::span<BasinSample> output,
                                             std::span<uint8_t> certified,
                                             learned::AuthorityRequestPriority priority) const {
    impl_->certifyDryPoints(positions, input, output, certified, priority);
}

std::optional<NativeHydrologyDryFootprintCertificate>
NativeHydrologyRouter::certifyDryFootprint(std::span<const BasinSamplePosition> positions,
                                           const NativeHydrologyInputFunction& input,
                                           learned::AuthorityRequestPriority priority) const {
    NativeHydrologyDryFootprintCertificate certificate;
    if (!impl_->certifyDryFootprint(positions, input, certificate.positions_, certificate.samples_,
                                    priority)) {
        return std::nullopt;
    }
    certificate.issuer_ = impl_.get();
    return certificate;
}

bool NativeHydrologyRouter::replaceCertifiedDryFootprint(
    const NativeHydrologyDryFootprintCertificate& certificate) const {
    return certificate.issuer_ == impl_.get() &&
           impl_->replaceCertifiedDryFootprint(certificate.positions_, certificate.samples_);
}

bool NativeHydrologyRouter::certifiedDryFootprintContains(
    std::span<const BasinSamplePosition> positions) const {
    return impl_->certifiedDryFootprintContains(positions);
}

void NativeHydrologyRouter::clearCertifiedDryFootprint() const {
    impl_->clearCertifiedDryFootprint();
}

void NativeHydrologyRouter::sampleGrid(int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                       int sampleWidth, int sampleHeight,
                                       const NativeHydrologyInputFunction& input,
                                       std::span<BasinSample> output,
                                       std::span<uint8_t> certifiedDryHits,
                                       learned::AuthorityRequestPriority priority) const {
    if (spacingX <= 0 || spacingZ <= 0 || sampleWidth <= 0 || sampleHeight <= 0 ||
        output.size() != static_cast<size_t>(sampleWidth) * sampleHeight ||
        (!certifiedDryHits.empty() && certifiedDryHits.size() != output.size())) {
        throw std::invalid_argument("invalid native hydrology grid");
    }
    const auto checkedLastCoordinate = [](int64_t origin, int spacing, int sampleCount) {
        const __int128 value =
            static_cast<__int128>(origin) + static_cast<__int128>(sampleCount - 1) * spacing;
        if (value < std::numeric_limits<int64_t>::min() ||
            value > std::numeric_limits<int64_t>::max()) {
            throw std::invalid_argument("native hydrology grid coordinate is out of range");
        }
        return static_cast<int64_t>(value);
    };
    const int64_t lastX = checkedLastCoordinate(originX, spacingX, sampleWidth);
    const int64_t lastZ = checkedLastCoordinate(originZ, spacingZ, sampleHeight);
    if (spacingX != 1 || spacingZ != 1) impl_->recordOrdinaryStageCoarseGridSamples(output.size());
    if (!impl_->hasCertifiedDryFootprint() &&
        impl_->sampleInteriorGrid(originX, originZ, lastX, lastZ, spacingX, spacingZ, sampleWidth,
                                  sampleHeight, input, output, priority)) {
        if (!certifiedDryHits.empty())
            std::fill(certifiedDryHits.begin(), certifiedDryHits.end(), uint8_t{0});
        return;
    }

    std::vector<BasinSamplePosition> positions;
    positions.reserve(output.size());
    for (int z = 0; z < sampleHeight; ++z) {
        for (int x = 0; x < sampleWidth; ++x) {
            const int64_t worldX = static_cast<int64_t>(static_cast<__int128>(originX) +
                                                        static_cast<__int128>(x) * spacingX);
            const int64_t worldZ = static_cast<int64_t>(static_cast<__int128>(originZ) +
                                                        static_cast<__int128>(z) * spacingZ);
            positions.push_back({static_cast<double>(worldX), static_cast<double>(worldZ)});
        }
    }
    impl_->samplePoints(positions, input, output, certifiedDryHits, spacingX == 1 && spacingZ == 1,
                        priority);
}

void NativeHydrologyRouter::samplePoints(std::span<const BasinSamplePosition> positions,
                                         const NativeHydrologyInputFunction& input,
                                         std::span<BasinSample> output,
                                         std::span<uint8_t> certifiedDryHits,
                                         learned::AuthorityRequestPriority priority) const {
    if (positions.size() != output.size() ||
        (!certifiedDryHits.empty() && certifiedDryHits.size() != positions.size())) {
        throw std::invalid_argument("invalid native hydrology point batch");
    }
    impl_->samplePoints(positions, input, output, certifiedDryHits, true, priority);
}

void NativeHydrologyRouter::sampleCoarsePoints(std::span<const BasinSamplePosition> positions,
                                               const NativeHydrologyInputFunction& input,
                                               std::span<BasinSample> output,
                                               std::span<uint8_t> certifiedDryHits,
                                               learned::AuthorityRequestPriority priority) const {
    if (positions.size() != output.size() ||
        (!certifiedDryHits.empty() && certifiedDryHits.size() != positions.size())) {
        throw std::invalid_argument("invalid coarse native hydrology point batch");
    }
    impl_->recordOrdinaryStageCoarseGridSamples(positions.size());
    impl_->samplePoints(positions, input, output, certifiedDryHits, false, priority);
}

void NativeHydrologyRouter::sampleTopologyGrid(int64_t originX, int64_t originZ, int cellWidth,
                                               int cellHeight,
                                               const NativeHydrologyInputFunction& input,
                                               std::span<NativeHydrologyTopologyCell> output,
                                               std::span<uint8_t> certifiedDryHits,
                                               learned::AuthorityRequestPriority priority) const {
    impl_->sampleTopologyGrid(originX, originZ, cellWidth, cellHeight, input, output,
                              certifiedDryHits, priority);
}

NativeHydrologyCacheMetrics NativeHydrologyRouter::cacheMetrics() const {
    return impl_->cacheMetrics();
}

void NativeHydrologyRouter::clear() const {
    impl_->clear();
}

} // namespace worldgen
