#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <common/trace.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <new>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Program 0 trace tests. All headless, fake, and network-free. They prove the
// disabled path allocates nothing, the enabled buffer is bounded and drops
// past capacity, concurrent emit is self-consistent, and trace::summarize is
// deterministic. The real five-route magnitudes need the model pack on the
// M4 Max and are recorded through the playtest workflow, not here.
// ---------------------------------------------------------------------------

namespace {

std::atomic<uint64_t> g_allocCount{0};
thread_local bool g_countAllocs = false;

void beginAllocCount() {
    g_countAllocs = true;
    g_allocCount.store(0, std::memory_order_relaxed);
}

uint64_t endAllocCount() {
    g_countAllocs = false;
    return g_allocCount.load(std::memory_order_relaxed);
}

} // namespace

// Replaceable global allocation functions so a test can prove the disabled and
// enabled emit paths perform zero heap allocation. Every call routes through
// malloc so the matching deletes below stay compatible.
void* operator new(std::size_t n) {
    if (g_countAllocs) {
        g_allocCount.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(n != 0 ? n : 1);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}
void* operator new[](std::size_t n) {
    return operator new(n);
}
void* operator new(std::size_t n, std::align_val_t a) {
    if (g_countAllocs) {
        g_allocCount.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = nullptr;
    const std::size_t alignment = std::max(sizeof(void*), static_cast<std::size_t>(a));
    if (::posix_memalign(&p, alignment, n != 0 ? n : 1) != 0) {
        throw std::bad_alloc();
    }
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return operator new(n, a);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

TEST_CASE("trace Event is a fixed 64-byte trivially copyable record", "[trace]") {
    CHECK(sizeof(trace::Event) == 64);
    CHECK(std::is_trivially_copyable_v<trace::Event>);
}

TEST_CASE("Disabled tracing performs no heap allocation", "[trace]") {
    trace::disable();
    beginAllocCount();
    for (int i = 0; i < 100000; ++i) {
        trace::instant(trace::Track::LearnedAuthority, trace::Name::AuthorityPageBuild,
                       {.spatialKey = static_cast<uint64_t>(i)});
        trace::Scope scope(trace::Track::Hydrology, trace::Name::HydrologyBuild,
                           {.bytesRetained = 64});
        (void)scope;
    }
    const uint64_t allocations = endAllocCount();
    CHECK(allocations == 0);
    CHECK(trace::enabled() == false);
}

TEST_CASE("Enabled tracing is bounded and drops past capacity", "[trace]") {
    trace::enable("", 1000, /*ring=*/false);
    const int total = 2500;
    for (int i = 0; i < total; ++i) {
        trace::instant(trace::Track::FarGeneration, trace::Name::FarTileBuild,
                       {.spatialKey = static_cast<uint64_t>(i)});
    }
    CHECK(trace::capacity() == 1000);
    CHECK(trace::recorded() == 1000);
    CHECK(trace::dropped() == 1500);
    trace::disable();
}

TEST_CASE("Enabled emit allocates nothing after enable", "[trace]") {
    trace::enable("", 4096, /*ring=*/false);
    trace::instant(trace::Track::Upload, trace::Name::FarMeshUpload); // warm the thread tag
    beginAllocCount();
    for (int i = 0; i < 2000; ++i) {
        trace::instant(trace::Track::Upload, trace::Name::FarMeshUpload,
                       {.spatialKey = static_cast<uint64_t>(i)});
    }
    const uint64_t allocations = endAllocCount();
    CHECK(allocations == 0);
    trace::disable();
}

TEST_CASE("Ring mode overwrites the oldest and stays bounded", "[trace]") {
    trace::enable("", 8, /*ring=*/true);
    for (int i = 0; i < 20; ++i) {
        trace::instant(trace::Track::ExactGeneration, trace::Name::ExactChunkGenerate,
                       {.spatialKey = static_cast<uint64_t>(i)});
    }
    CHECK(trace::capacity() == 8);
    CHECK(trace::recorded() == 8);
    CHECK(trace::dropped() == 12);
    trace::disable();
}

TEST_CASE("Concurrent emit is bounded and self-consistent", "[trace]") {
    trace::enable("", 100000, /*ring=*/false);
    const int threadCount = 8;
    const int perThread = 20000;
    std::vector<std::thread> pool;
    for (int t = 0; t < threadCount; ++t) {
        pool.emplace_back([&]() {
            for (int i = 0; i < perThread; ++i) {
                trace::instant(trace::Track::Upload, trace::Name::FarMeshUpload,
                               {.spatialKey = static_cast<uint64_t>(i)});
            }
        });
    }
    for (std::thread& worker : pool) {
        worker.join();
    }
    CHECK(trace::recorded() == 100000);
    CHECK(trace::dropped() == static_cast<size_t>(threadCount * perThread) - 100000);
    bool everyRecordValid = true;
    for (const trace::Event& e : trace::events()) {
        if (e.track != trace::Track::Upload ||
            e.nameId != static_cast<uint32_t>(trace::Name::FarMeshUpload)) {
            everyRecordValid = false;
            break;
        }
    }
    CHECK(everyRecordValid);
    trace::disable();
}

namespace {

trace::Event makeSpan(trace::Track track, trace::Name name, uint64_t ts, uint64_t dur,
                      uint8_t priority = 3, uint64_t spatialKey = 0, uint32_t epoch = 0) {
    trace::Event e;
    e.track = track;
    e.kind = trace::EventKind::Complete;
    e.nameId = static_cast<uint32_t>(name);
    e.timestampNs = ts;
    e.durationNs = dur;
    e.priority = priority;
    e.spatialKey = spatialKey;
    e.cameraEpoch = epoch;
    return e;
}

trace::Event makeInstant(trace::Track track, trace::Name name, uint64_t bytes = 0,
                         uint64_t spatialKey = 0) {
    trace::Event e;
    e.track = track;
    e.kind = trace::EventKind::Instant;
    e.nameId = static_cast<uint32_t>(name);
    e.bytesRetained = bytes;
    e.spatialKey = spatialKey;
    return e;
}

} // namespace

TEST_CASE("summarize computes deterministic critical-path statistics", "[trace]") {
    std::vector<trace::Event> events;
    // Five learned-authority spans with known durations.
    for (uint64_t d : {50ull, 10ull, 40ull, 20ull, 30ull}) {
        events.push_back(makeSpan(trace::Track::LearnedAuthority, trace::Name::AuthorityPageBuild,
                                  /*ts=*/1000 + d, d));
    }
    // A protected enqueue-to-start wait of 150 ns.
    trace::Event begin;
    begin.track = trace::Track::LearnedAuthority;
    begin.kind = trace::EventKind::Begin;
    begin.nameId = static_cast<uint32_t>(trace::Name::AuthorityEnqueue);
    begin.timestampNs = 100;
    begin.spatialKey = 7;
    begin.cameraEpoch = 1;
    begin.priority = 2;
    events.push_back(begin);
    events.push_back(makeSpan(trace::Track::LearnedAuthority, trace::Name::AuthorityEnqueue,
                              /*ts=*/250, /*dur=*/5, /*priority=*/2, /*spatialKey=*/7,
                              /*epoch=*/1));
    // Duplicate model window: the same window (name + lattice key) computed
    // twice is recorded by two insertWindow instants, not the model-call spans.
    events.push_back(makeInstant(trace::Track::ModelWindow, trace::Name::ModelDecoder, 0, 99));
    events.push_back(makeInstant(trace::Track::ModelWindow, trace::Name::ModelDecoder, 0, 99));
    // Cache and omission instants.
    events.push_back(makeInstant(trace::Track::LearnedAuthority, trace::Name::AuthorityCacheHit));
    events.push_back(makeInstant(trace::Track::LearnedAuthority, trace::Name::AuthorityCacheHit));
    events.push_back(makeInstant(trace::Track::Hydrology, trace::Name::HydrologyHit));
    events.push_back(makeInstant(trace::Track::LearnedAuthority, trace::Name::AuthorityCacheMiss));
    events.push_back(makeInstant(trace::Track::LearnedAuthority, trace::Name::AuthorityCacheEvict));
    events.push_back(makeInstant(trace::Track::ScreenErrorDebt, trace::Name::OmittedWater, 5));
    events.push_back(makeInstant(trace::Track::ScreenErrorDebt, trace::Name::OmittedFlora, 7));
    // A failed cancellation.
    trace::Event failed = makeSpan(trace::Track::FarGeneration, trace::Name::FarTileBuild, 3, 1);
    failed.cancellation = trace::Cancellation::Failed;
    events.push_back(failed);

    const trace::Summary a = trace::summarize(events);
    const trace::Summary b = trace::summarize(events);

    const trace::TrackSummary& authority =
        a.tracks[static_cast<size_t>(trace::Track::LearnedAuthority)];
    CHECK(authority.count == 6); // five spans plus the paired protected span
    // Percentiles over {10,20,30,40,50,5} sorted {5,10,20,30,40,50}: p50 index 3, p95 index 5.
    CHECK(authority.p50Ns == 30);
    CHECK(authority.p95Ns == 50);
    CHECK(authority.maxNs == 50);

    CHECK(a.protectedWaitNs == 150);
    CHECK(a.duplicateModelWindows == 1);
    CHECK(a.cacheHits == 3);
    CHECK(a.cacheMisses == 1);
    CHECK(a.cacheEvictions == 1);
    CHECK(a.omittedWater == 5);
    CHECK(a.omittedFlora == 7);
    CHECK(a.cancellation[static_cast<size_t>(trace::Cancellation::Failed)] == 1);

    // Determinism: identical output for identical input.
    CHECK(a.protectedWaitNs == b.protectedWaitNs);
    CHECK(a.duplicateModelWindows == b.duplicateModelWindows);
    CHECK(authority.p50Ns == b.tracks[static_cast<size_t>(trace::Track::LearnedAuthority)].p50Ns);
}

TEST_CASE("binary trace round-trips through readBinary", "[trace]") {
    TempDir dir("trace");
    std::filesystem::create_directories(dir.path());
    const std::string base = dir.path() + "/route";

    trace::enable(base, 256, /*ring=*/false);
    for (int i = 0; i < 32; ++i) {
        trace::instant(trace::Track::Hydrology, trace::Name::HydrologyBuild,
                       {.spatialKey = static_cast<uint64_t>(i), .bytesRetained = 128});
    }
    std::vector<trace::Event> live(trace::events().begin(), trace::events().end());
    CHECK(trace::flush());
    trace::disable();

    std::vector<trace::Event> loaded;
    REQUIRE(trace::readBinary(base + ".rytrace", loaded));
    REQUIRE(loaded.size() == live.size());
    bool identical = true;
    for (size_t i = 0; i < loaded.size(); ++i) {
        if (loaded[i].spatialKey != live[i].spatialKey || loaded[i].nameId != live[i].nameId) {
            identical = false;
            break;
        }
    }
    CHECK(identical);

    // The Chrome JSON export is present and well-formed at the outer braces.
    std::ifstream json(base);
    REQUIRE(json.good());
    std::string first;
    std::getline(json, first);
    CHECK(!first.empty());
    CHECK(first.front() == '{');
}
