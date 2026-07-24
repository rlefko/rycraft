#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// common/trace: one machine-readable critical-path trace for generator v4.
//
// This is the single source of truth for the trace record layout, the span
// enable switch, and the summary computation. Every subsystem (bootstrap,
// learned authority, hydrology, exact and far generation, upload, GPU
// residency, and screen-error debt) emits through this header, and the
// rycraft_trace_summary tool and tests/test_trace.mm read the same layout.
//
// The subsystem is observability only. It must never change authority,
// hydrology, selection, or rendering output. When disabled (the default) an
// emit site costs one relaxed atomic load plus a predicted-not-taken branch,
// performs no heap allocation, and the buffer holds zero bytes. When enabled
// the buffer is one up-front allocation that never grows: linear append drops
// past capacity, ring mode overwrites the oldest. Export happens only at
// quiesce, on one thread, after workers stop emitting.
// ---------------------------------------------------------------------------

namespace trace {

// One Chrome-trace lane per subsystem (used as the Chrome "tid").
enum class Track : uint8_t {
    Bootstrap = 0,
    ModelWindow,
    LearnedAuthority,
    Hydrology,
    ExactGeneration,
    FarGeneration,
    Upload,
    GpuResidency,
    ScreenErrorDebt,
    Count,
};

// Chrome phase: Begin/End are queue markers, Instant is a point, Complete is a
// self-contained span carrying its own duration.
enum class EventKind : uint8_t { Begin = 0, End, Instant, Complete };

// Why a work item was waiting on a dependency (diagnostic only).
enum class Dependency : uint8_t {
    None = 0,
    WaitingAuthority,
    WaitingHydrology,
    WaitingParent,
    WaitingUpload,
    WaitingInference,
};

// Terminal disposition of a work item (diagnostic only).
enum class Cancellation : uint8_t {
    None = 0,
    Completed,
    Deferred,
    Failed,
    CanceledEpoch,
    CanceledPriority,
    Count,
};

// The whole run is one route; the summarizer partitions by it.
enum class RouteTag : uint16_t {
    Unknown = 0,
    ColdEntry,
    WarmEntry,
    Movement,
    Hover,
    Reversal,
    LodFlight,
    Settlement,
    Count,
};

// Stable interned span names. Ids are compile-time so the hot path never
// touches a string; the exporter and summarizer map id -> name.
enum class Name : uint32_t {
    BootstrapDownload = 0,
    BootstrapVerify,
    BootstrapExtract,
    BootstrapCompile,
    BootstrapLoadQualify,
    ModelCoarse,
    ModelBase,
    ModelDecoder,
    AuthorityEnqueue,
    AuthorityPageBuild,
    AuthorityCacheHit,
    AuthorityCacheMiss,
    AuthorityCacheEvict,
    AuthorityCoarseSpawn,
    AuthorityTransientGrid,
    HydrologyBuild,
    HydrologyHit,
    HydrologyDeferred,
    ExactChunkGenerate,
    FarTileBuild,
    FarCanopyBuild,
    ExactMeshUpload,
    FarMeshUpload,
    GpuResident,
    GpuEvict,
    ScreenErrorWorst,
    OmittedWater,
    OmittedFlora,
    Count,
};

const char* nameString(uint32_t nameId) noexcept;
const char* trackString(Track track) noexcept;
const char* routeString(RouteTag route) noexcept;
// One mapping from a route token (cold/warm/move/hover/reversal/lod/settle) to
// its tag, shared by the enable switch and the summarizer tool.
RouteTag routeFromName(std::string_view name) noexcept;

// Fixed 64-byte trivially copyable record. Every field is a scalar already in
// scope at the emit site; nothing owns heap.
struct Event {
    uint64_t timestampNs = 0;    // steady time since the trace epoch
    uint64_t durationNs = 0;     // execution time (Complete); 0 otherwise
    uint64_t spatialKey = 0;     // packed immutable spatial key (see packers)
    uint64_t fingerprintLow = 0; // low 64 bits of the generation fingerprint
    uint64_t bytesRetained = 0;  // page/mesh/buffer bytes, or an omitted count
    uint32_t cameraEpoch = 0;    // protected-handoff / view epoch
    uint32_t nameId = 0;         // trace::Name
    int32_t threadTag = 0;       // interned emitting-thread id
    uint16_t routeTag = 0;       // trace::RouteTag
    Track track = Track::Bootstrap;
    EventKind kind = EventKind::Instant;
    uint8_t quality = 0;  // AuthorityQuality / far quality
    uint8_t priority = 0; // AuthorityRequestPriority / far step
    Dependency dependency = Dependency::None;
    Cancellation cancellation = Cancellation::None;
    uint32_t reserved = 0;
};

static_assert(sizeof(Event) == 64, "trace::Event must stay 64 bytes");
static_assert(std::is_trivially_copyable_v<Event>, "trace::Event must be trivially copyable");

// Optional per-emit fields. Lives on the caller stack; never heap.
struct Context {
    uint64_t spatialKey = 0;
    uint64_t bytesRetained = 0;
    uint64_t fingerprintLow = 0;
    uint32_t cameraEpoch = 0;
    uint8_t quality = 0;
    uint8_t priority = 0;
    Dependency dependency = Dependency::None;
    Cancellation cancellation = Cancellation::None;
};

// -- Enable switch (disabled by default) ------------------------------------
// One process-wide inline atomic so enabled() inlines into every C++ and
// ObjC++ translation unit as a single relaxed load.
inline std::atomic<bool> g_enabled{false};

[[nodiscard]] inline bool enabled() noexcept {
    return g_enabled.load(std::memory_order_relaxed);
}

constexpr size_t kDefaultCapacityEvents = 1u << 20; // 64 MiB of records

// Reserve the buffer once and turn emission on. basePath is the trace output
// stem: flush() writes basePath (Chrome JSON) and basePath + ".rytrace"
// (binary). Safe to call once before any worker emits.
void enable(std::string basePath, size_t capacityEvents = kDefaultCapacityEvents,
            bool ring = false);
// Read RYCRAFT_TRACE / RYCRAFT_TRACE_CAPACITY / RYCRAFT_TRACE_RING /
// RYCRAFT_TRACE_ROUTE and enable when RYCRAFT_TRACE is set. No-op otherwise.
void enableFromEnvironment();
void disable() noexcept; // stop emission; retained for tests
void reset() noexcept;   // clear the buffer without disabling; for tests

void setRoute(RouteTag route) noexcept;
RouteTag route() noexcept;
void setFingerprintLow(uint64_t fingerprintLow) noexcept;
[[nodiscard]] uint64_t fingerprintLow() noexcept;

size_t capacity() noexcept;
size_t recorded() noexcept;               // events retained (min(cursor, capacity))
size_t dropped() noexcept;                // events dropped past capacity
std::span<const Event> events() noexcept; // valid only while enabled

uint64_t nowNs() noexcept;                 // steady time since the trace epoch
int32_t threadTag() noexcept;              // interned id for the calling thread
void recordEvent(const Event& e) noexcept; // out of line; only touches heap-free storage

// -- Emit API ---------------------------------------------------------------
inline Event makeEvent(Track track, EventKind kind, Name name, const Context& c) noexcept {
    Event e;
    e.timestampNs = nowNs();
    e.spatialKey = c.spatialKey;
    e.fingerprintLow = c.fingerprintLow != 0 ? c.fingerprintLow : fingerprintLow();
    e.bytesRetained = c.bytesRetained;
    e.cameraEpoch = c.cameraEpoch;
    e.nameId = static_cast<uint32_t>(name);
    e.threadTag = threadTag();
    e.routeTag = static_cast<uint16_t>(route());
    e.track = track;
    e.kind = kind;
    e.quality = c.quality;
    e.priority = c.priority;
    e.dependency = c.dependency;
    e.cancellation = c.cancellation;
    return e;
}

inline void instant(Track track, Name name, const Context& c = {}) noexcept {
    if (!enabled()) {
        return;
    }
    recordEvent(makeEvent(track, EventKind::Instant, name, c));
}

// Queue marker at enqueue time; paired with the executing Complete span by
// (track, spatialKey, cameraEpoch) in the summarizer to derive queue wait.
inline void enqueued(Track track, Name name, const Context& c = {}) noexcept {
    if (!enabled()) {
        return;
    }
    recordEvent(makeEvent(track, EventKind::Begin, name, c));
}

// RAII span that emits one Complete on destruction. Mirrors the existing
// InferencePhaseScope idiom. Disabled cost is one bool store plus one branch.
class Scope {
public:
    Scope(Track track, Name name, const Context& c = {}) noexcept
        : enabled_(enabled())
        , track_(track)
        , name_(name) {
        if (enabled_) {
            beginNs_ = nowNs();
            ctx_ = c;
        }
    }
    ~Scope() {
        if (!enabled_) {
            return;
        }
        Event e = makeEvent(track_, EventKind::Complete, name_, ctx_);
        e.timestampNs = beginNs_;
        e.durationNs = nowNs() - beginNs_;
        recordEvent(e);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    // Refine the context after construction without changing behavior when
    // disabled (e.g. attach bytesRetained or a cancellation known at the end).
    void setBytesRetained(uint64_t bytes) noexcept {
        if (enabled_) {
            ctx_.bytesRetained = bytes;
        }
    }
    void setCancellation(Cancellation c) noexcept {
        if (enabled_) {
            ctx_.cancellation = c;
        }
    }

private:
    bool enabled_;
    Track track_;
    Name name_;
    uint64_t beginNs_ = 0;
    Context ctx_{};
};

// -- Spatial key packers (dependency-free; callers pack their own keys) ------
// Half-open signed coordinates fit two 26-bit fields; the low byte carries a
// step/quality/edge discriminator. One definition so every track packs alike.
constexpr uint64_t packCoord(int64_t a, int64_t b, uint8_t discriminator = 0) noexcept {
    const uint64_t ua = static_cast<uint64_t>(a) & 0x3FF'FFFFull;
    const uint64_t ub = static_cast<uint64_t>(b) & 0x3FF'FFFFull;
    return (ua << 34) | (ub << 8) | discriminator;
}

// -- Summary ----------------------------------------------------------------
struct TrackSummary {
    Track track = Track::Bootstrap;
    uint64_t count = 0;
    uint64_t p50Ns = 0;
    uint64_t p95Ns = 0;
    uint64_t maxNs = 0;
    uint64_t totalNs = 0;
    uint64_t maxQueueDepth = 0;
    uint64_t canceled = 0; // events with a non-None, non-Completed disposition
};

struct Summary {
    RouteTag route = RouteTag::Unknown;
    uint64_t eventCount = 0;
    uint64_t droppedEvents = 0;
    TrackSummary tracks[static_cast<size_t>(Track::Count)]{};
    uint64_t cacheHits = 0;
    uint64_t cacheMisses = 0;
    uint64_t cacheEvictions = 0;
    uint64_t cancellation[static_cast<size_t>(Cancellation::Count)]{};
    uint64_t omittedWater = 0;
    uint64_t omittedFlora = 0;
    uint64_t duplicateModelWindows = 0;         // ModelWindow spans reusing an already-executed key
    uint64_t protectedWaitNs = 0;               // total protected enqueue-to-start wait
    uint64_t optionalOwnedProtectedWaitNs = 0;  // optional work executing during protected wait
    Track topProtectedWaitOwner = Track::Count; // which track owned the most protected wait
    uint64_t topProtectedWaitOwnerNs = 0;
};

// Pure, deterministic summary over an event span. Shared by the tool and
// tests so both compute identically.
Summary summarize(std::span<const Event> events);

// Write the current buffer to the configured sinks. Call at quiesce only.
bool flush();
bool writeBinary(const std::string& path);
bool writeChromeTrace(const std::string& path);

// Read a binary trace written by writeBinary. Returns false on a bad header.
bool readBinary(const std::string& path, std::vector<Event>& out);

} // namespace trace
