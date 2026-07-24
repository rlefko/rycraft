#include "common/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <tuple>

// ---------------------------------------------------------------------------
// common/trace implementation. See common/trace.hpp for the contract.
//
// The buffer is one std::vector<Event> sized once inside enable(). Writers
// claim a unique slot with a relaxed fetch_add; the only hot-path
// synchronization is that single atomic. disable()/reset() never free the
// storage, so an in-flight recordEvent that already passed enabled() still
// writes into a valid slot.
// ---------------------------------------------------------------------------

namespace trace {
namespace {

std::vector<Event> g_events;
std::atomic<uint64_t> g_cursor{0};
std::atomic<size_t> g_capacity{0};
std::atomic<bool> g_ring{false};
std::atomic<uint16_t> g_route{static_cast<uint16_t>(RouteTag::Unknown)};
std::atomic<uint64_t> g_fingerprintLow{0};
std::string g_basePath;
std::atomic<int32_t> g_nextThreadTag{0};

std::chrono::steady_clock::time_point traceEpoch() {
    static const std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
    return epoch;
}

constexpr const char* kNames[] = {
    "bootstrap.download",
    "bootstrap.verify",
    "bootstrap.extract",
    "bootstrap.compile",
    "bootstrap.loadQualify",
    "model.coarse",
    "model.base",
    "model.decoder",
    "authority.enqueue",
    "authority.pageBuild",
    "authority.cacheHit",
    "authority.cacheMiss",
    "authority.cacheEvict",
    "authority.coarseSpawn",
    "authority.transientGrid",
    "hydrology.build",
    "hydrology.hit",
    "hydrology.deferred",
    "exact.chunkGenerate",
    "far.tileBuild",
    "far.canopyBuild",
    "upload.exactMesh",
    "upload.farMesh",
    "gpu.resident",
    "gpu.evict",
    "screenError.worst",
    "omitted.water",
    "omitted.flora",
};
static_assert(sizeof(kNames) / sizeof(kNames[0]) == static_cast<size_t>(Name::Count),
              "kNames must cover every trace::Name");

constexpr const char* kTracks[] = {
    "bootstrap", "modelWindow", "learnedAuthority", "hydrology",       "exactGeneration",
    "far",       "upload",      "gpuResidency",     "screenErrorDebt",
};
static_assert(sizeof(kTracks) / sizeof(kTracks[0]) == static_cast<size_t>(Track::Count),
              "kTracks must cover every trace::Track");

constexpr const char* kRoutes[] = {
    "unknown", "cold", "warm", "move", "hover", "reversal", "lod", "settle",
};
static_assert(sizeof(kRoutes) / sizeof(kRoutes[0]) == static_cast<size_t>(RouteTag::Count),
              "kRoutes must cover every trace::RouteTag");

struct BinaryHeader {
    char magic[4]; // "RYTR"
    uint32_t version;
    uint64_t eventCount;
};
constexpr uint32_t kBinaryVersion = 1;

uint64_t percentile(const std::vector<uint64_t>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0;
    }
    const size_t maxIndex = sorted.size() - 1;
    size_t index = static_cast<size_t>(fraction * static_cast<double>(maxIndex) + 0.5);
    if (index > maxIndex) {
        index = maxIndex;
    }
    return sorted[index];
}

} // namespace

const char* nameString(uint32_t nameId) noexcept {
    if (nameId >= static_cast<uint32_t>(Name::Count)) {
        return "unknown";
    }
    return kNames[nameId];
}

const char* trackString(Track track) noexcept {
    if (static_cast<size_t>(track) >= static_cast<size_t>(Track::Count)) {
        return "unknown";
    }
    return kTracks[static_cast<size_t>(track)];
}

const char* routeString(RouteTag route) noexcept {
    if (static_cast<size_t>(route) >= static_cast<size_t>(RouteTag::Count)) {
        return "unknown";
    }
    return kRoutes[static_cast<size_t>(route)];
}

RouteTag routeFromName(std::string_view name) noexcept {
    for (size_t index = 0; index < static_cast<size_t>(RouteTag::Count); ++index) {
        if (name == kRoutes[index]) {
            return static_cast<RouteTag>(index);
        }
    }
    return RouteTag::Unknown;
}

uint64_t nowNs() noexcept {
    const auto delta = std::chrono::steady_clock::now() - traceEpoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count());
}

int32_t threadTag() noexcept {
    static thread_local int32_t tag = g_nextThreadTag.fetch_add(1, std::memory_order_relaxed);
    return tag;
}

void recordEvent(const Event& e) noexcept {
    const size_t cap = g_capacity.load(std::memory_order_relaxed);
    if (cap == 0) {
        return;
    }
    const uint64_t index = g_cursor.fetch_add(1, std::memory_order_relaxed);
    if (g_ring.load(std::memory_order_relaxed)) {
        g_events[index % cap] = e;
    } else if (index < cap) {
        g_events[index] = e;
    }
    // Dropped records are derived from the cursor in dropped(); ring mode
    // overwrites the oldest, linear mode discards past capacity.
}

void enable(std::string basePath, size_t capacityEvents, bool ring) {
    (void)traceEpoch(); // stamp the epoch before the first event
    g_basePath = std::move(basePath);
    if (capacityEvents == 0) {
        capacityEvents = kDefaultCapacityEvents;
    }
    g_events.assign(capacityEvents, Event{});
    g_capacity.store(capacityEvents, std::memory_order_relaxed);
    g_ring.store(ring, std::memory_order_relaxed);
    g_cursor.store(0, std::memory_order_relaxed);
    g_enabled.store(true, std::memory_order_release);
}

void enableFromEnvironment() {
    const char* path = std::getenv("RYCRAFT_TRACE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    size_t capacity = kDefaultCapacityEvents;
    if (const char* cap = std::getenv("RYCRAFT_TRACE_CAPACITY")) {
        const long long parsed = std::atoll(cap);
        if (parsed > 0) {
            capacity = static_cast<size_t>(parsed);
        }
    }
    const char* ring = std::getenv("RYCRAFT_TRACE_RING");
    const bool ringMode = ring != nullptr && ring[0] == '1';
    if (const char* routeName = std::getenv("RYCRAFT_TRACE_ROUTE")) {
        setRoute(routeFromName(routeName));
    }
    enable(path, capacity, ringMode);
}

void disable() noexcept {
    g_enabled.store(false, std::memory_order_release);
}

void reset() noexcept {
    g_cursor.store(0, std::memory_order_relaxed);
}

void setRoute(RouteTag route) noexcept {
    g_route.store(static_cast<uint16_t>(route), std::memory_order_relaxed);
}

RouteTag route() noexcept {
    return static_cast<RouteTag>(g_route.load(std::memory_order_relaxed));
}

void setFingerprintLow(uint64_t value) noexcept {
    g_fingerprintLow.store(value, std::memory_order_relaxed);
}

uint64_t fingerprintLow() noexcept {
    return g_fingerprintLow.load(std::memory_order_relaxed);
}

size_t capacity() noexcept {
    return g_capacity.load(std::memory_order_relaxed);
}

size_t recorded() noexcept {
    const size_t cap = g_capacity.load(std::memory_order_relaxed);
    const uint64_t cursor = g_cursor.load(std::memory_order_relaxed);
    return static_cast<size_t>(std::min<uint64_t>(cursor, cap));
}

size_t dropped() noexcept {
    const size_t cap = g_capacity.load(std::memory_order_relaxed);
    const uint64_t cursor = g_cursor.load(std::memory_order_relaxed);
    return cursor > cap ? static_cast<size_t>(cursor - cap) : 0;
}

std::span<const Event> events() noexcept {
    return std::span<const Event>(g_events.data(), recorded());
}

Summary summarize(std::span<const Event> events) {
    Summary summary;
    summary.eventCount = events.size();
    // summarize is pure over its input span; the live buffer's drop count is
    // reported separately through trace::dropped() at flush time.
    summary.droppedEvents = 0;
    if (!events.empty()) {
        summary.route = static_cast<RouteTag>(events.front().routeTag);
    }

    std::vector<uint64_t> durations[static_cast<size_t>(Track::Count)];
    std::vector<std::pair<uint64_t, int>> depthEvents[static_cast<size_t>(Track::Count)];

    // Pair enqueue markers with executing spans by (track, spatialKey, epoch).
    std::map<std::tuple<uint8_t, uint64_t, uint32_t>, uint64_t> enqueueTimes;
    // Detect duplicate model-window execution within the run.
    std::map<std::pair<uint32_t, uint64_t>, uint64_t> modelWindowExecutions;
    std::vector<std::pair<uint64_t, uint64_t>> protectedWaits; // [begin, end)

    for (const Event& e : events) {
        const size_t track = static_cast<size_t>(e.track);
        if (track >= static_cast<size_t>(Track::Count)) {
            continue;
        }
        const auto pairKey =
            std::make_tuple(static_cast<uint8_t>(e.track), e.spatialKey, e.cameraEpoch);

        if (e.kind == EventKind::Begin) {
            enqueueTimes[pairKey] = e.timestampNs;
            depthEvents[track].emplace_back(e.timestampNs, +1);
            continue;
        }

        // insertWindow emits one instant per actual window computation, keyed by
        // the window's lattice index and stage. A window that reappears here
        // after eviction is a recompute, which the cross-check test confirms is
        // the real per-window record (model-call Complete spans carry no window
        // key and are excluded).
        if (e.track == Track::ModelWindow && e.kind == EventKind::Instant) {
            const auto key = std::make_pair(e.nameId, e.spatialKey);
            if (modelWindowExecutions[key]++ > 0) {
                ++summary.duplicateModelWindows;
            }
        }

        if (e.kind == EventKind::Complete) {
            durations[track].push_back(e.durationNs);
            summary.tracks[track].totalNs += e.durationNs;
            depthEvents[track].emplace_back(e.timestampNs, -1);

            auto it = enqueueTimes.find(pairKey);
            if (it != enqueueTimes.end()) {
                const uint64_t start = e.timestampNs;
                if (start > it->second &&
                    e.priority <= 2) { // SPAWN/EXPLORATION_EXACT/PROTECTED_HANDOFF
                    protectedWaits.emplace_back(it->second, start);
                    summary.protectedWaitNs += start - it->second;
                }
                enqueueTimes.erase(it);
            }
        }

        if (e.cancellation != Cancellation::None) {
            const size_t code = static_cast<size_t>(e.cancellation);
            if (code < static_cast<size_t>(Cancellation::Count)) {
                ++summary.cancellation[code];
            }
            if (e.cancellation != Cancellation::Completed) {
                ++summary.tracks[track].canceled;
            }
        }

        switch (static_cast<Name>(e.nameId)) {
            case Name::AuthorityCacheHit:
            case Name::HydrologyHit:
                ++summary.cacheHits;
                break;
            case Name::AuthorityCacheMiss:
                ++summary.cacheMisses;
                break;
            case Name::AuthorityCacheEvict:
            case Name::GpuEvict:
                ++summary.cacheEvictions;
                break;
            case Name::OmittedWater:
                summary.omittedWater += e.bytesRetained;
                break;
            case Name::OmittedFlora:
                summary.omittedFlora += e.bytesRetained;
                break;
            default:
                break;
        }
    }

    for (size_t track = 0; track < static_cast<size_t>(Track::Count); ++track) {
        TrackSummary& ts = summary.tracks[track];
        ts.track = static_cast<Track>(track);
        std::vector<uint64_t>& d = durations[track];
        ts.count = d.size();
        if (!d.empty()) {
            std::sort(d.begin(), d.end());
            ts.p50Ns = percentile(d, 0.50);
            ts.p95Ns = percentile(d, 0.95);
            ts.maxNs = d.back();
        }

        auto& depth = depthEvents[track];
        std::sort(depth.begin(), depth.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second > b.second; // count enqueues before starts at equal time
        });
        int64_t running = 0;
        int64_t peak = 0;
        for (const auto& [time, delta] : depth) {
            (void)time;
            running += delta;
            peak = std::max(peak, running);
        }
        ts.maxQueueDepth = static_cast<uint64_t>(std::max<int64_t>(peak, 0));
    }

    // Merge protected-wait intervals, then attribute optional execution overlap.
    std::sort(protectedWaits.begin(), protectedWaits.end());
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (const auto& interval : protectedWaits) {
        if (!merged.empty() && interval.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, interval.second);
        } else {
            merged.push_back(interval);
        }
    }
    for (const Event& e : events) {
        if (e.kind != EventKind::Complete ||
            e.priority < 4) { // COARSE_PREVIEW/SPECULATIVE_PREFETCH
            continue;
        }
        const uint64_t start = e.timestampNs;
        const uint64_t end = start + e.durationNs;
        auto it = std::upper_bound(merged.begin(), merged.end(), std::make_pair(end, end));
        for (auto scan = merged.begin(); scan != it; ++scan) {
            const uint64_t lo = std::max(start, scan->first);
            const uint64_t hi = std::min(end, scan->second);
            if (lo < hi) {
                summary.optionalOwnedProtectedWaitNs += hi - lo;
            }
        }
    }

    // Which track owned the most protected wait (its enqueue-to-start time).
    uint64_t bestOwner = 0;
    for (size_t track = 0; track < static_cast<size_t>(Track::Count); ++track) {
        if (summary.tracks[track].totalNs > bestOwner && summary.protectedWaitNs > 0) {
            bestOwner = summary.tracks[track].totalNs;
            summary.topProtectedWaitOwner = static_cast<Track>(track);
            summary.topProtectedWaitOwnerNs = summary.tracks[track].totalNs;
        }
    }
    return summary;
}

bool writeBinary(const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    const std::span<const Event> span = events();
    BinaryHeader header{{'R', 'Y', 'T', 'R'}, kBinaryVersion, span.size()};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(span.data()),
              static_cast<std::streamsize>(span.size() * sizeof(Event)));
    return static_cast<bool>(out);
}

bool readBinary(const std::string& path, std::vector<Event>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    BinaryHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "RYTR", 4) != 0 || header.version != kBinaryVersion) {
        return false;
    }
    out.resize(header.eventCount);
    if (header.eventCount > 0) {
        in.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(header.eventCount * sizeof(Event)));
    }
    return static_cast<bool>(in);
}

bool writeChromeTrace(const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return false;
    }
    out << "{\"displayTimeUnit\":\"ms\",\"otherData\":{\"fingerprintLow\":\""
        << g_fingerprintLow.load(std::memory_order_relaxed) << "\",\"route\":\""
        << routeString(route()) << "\"},\"traceEvents\":[";
    bool first = true;
    auto comma = [&]() {
        if (!first) {
            out << ',';
        }
        first = false;
    };
    // Name each subsystem lane once.
    for (size_t track = 0; track < static_cast<size_t>(Track::Count); ++track) {
        comma();
        out << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << track
            << ",\"args\":{\"name\":\"" << kTracks[track] << "\"}}";
    }
    const std::span<const Event> span = events();
    for (const Event& e : span) {
        const double ts = static_cast<double>(e.timestampNs) / 1000.0;
        const char* ph = e.kind == EventKind::Complete ? "X" : "i";
        comma();
        out << "{\"name\":\"" << nameString(e.nameId) << "\",\"ph\":\"" << ph
            << "\",\"pid\":1,\"tid\":" << static_cast<int>(e.track) << ",\"ts\":" << ts;
        if (e.kind == EventKind::Complete) {
            out << ",\"dur\":" << static_cast<double>(e.durationNs) / 1000.0;
        } else {
            out << ",\"s\":\"g\"";
        }
        out << ",\"args\":{\"spatialKey\":" << e.spatialKey << ",\"cameraEpoch\":" << e.cameraEpoch
            << ",\"quality\":" << static_cast<int>(e.quality)
            << ",\"priority\":" << static_cast<int>(e.priority)
            << ",\"dependency\":" << static_cast<int>(e.dependency)
            << ",\"cancellation\":" << static_cast<int>(e.cancellation)
            << ",\"bytesRetained\":" << e.bytesRetained << "}}";
    }
    out << "]}";
    return static_cast<bool>(out);
}

bool flush() {
    if (g_basePath.empty()) {
        return false;
    }
    const bool jsonOk = writeChromeTrace(g_basePath);
    const bool binaryOk = writeBinary(g_basePath + ".rytrace");
    return jsonOk && binaryOk;
}

} // namespace trace
