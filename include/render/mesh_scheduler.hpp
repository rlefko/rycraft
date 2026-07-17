#pragma once

#include "common/ema.hpp"
#include "render/lod_mesher.hpp"
#include "world/chunk_pos.hpp"
#include "world/mesh_snapshot.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

class World;

// One finished (or failed) mesh build, drained by the render thread.
struct MeshResult {
    ChunkPos pos;
    uint32_t requestedVersion = 0; // chunk revision requested by the renderer
    uint32_t builtVersion = 0;     // chunk revision the snapshot captured
    bool snapshotOk = false;       // false when snapshot prerequisites were unavailable
    MeshOutput mesh;
};

// Coalescing must preserve the completion for the newest renderer request,
// even when that snapshot failed. Keeping an older successful result instead
// would make the renderer reject it while the newer request remained marked
// in flight forever. Within one request, prefer a successful and newer build.
constexpr bool meshResultSupersedes(const MeshResult& existing, const MeshResult& incoming) {
    const int32_t requestOrder =
        static_cast<int32_t>(incoming.requestedVersion - existing.requestedVersion);
    if (requestOrder != 0) return requestOrder > 0;
    if (existing.snapshotOk != incoming.snapshotOk) return incoming.snapshotOk;
    if (!incoming.snapshotOk) return true;
    return static_cast<int32_t>(incoming.builtVersion - existing.builtVersion) >= 0;
}

struct MeshSchedulerStats {
    size_t schedulerOwned = 0; // queued + building + completed
    size_t consumerPending = 0;
    size_t completed = 0;
    size_t highWater = 0;
    uint64_t coalesced = 0;
    uint64_t droppedStale = 0;
};

enum class MeshPriorityLane : uint8_t {
    BROAD_SURFACE = 0,
    CAMERA_BAND = 1,
    CAMERA_COLUMN = 2,
};

inline constexpr size_t EXACT_MESH_CAMERA_RESERVED_SLOTS = 32;
inline constexpr size_t EXACT_MESH_MAX_INFLIGHT = 64;

inline constexpr bool meshJobRanksBefore(MeshPriorityLane leftLane, uint64_t leftDistanceSquared,
                                         uint64_t leftSequence, MeshPriorityLane rightLane,
                                         uint64_t rightDistanceSquared, uint64_t rightSequence) {
    if (leftLane != rightLane)
        return static_cast<uint8_t>(leftLane) > static_cast<uint8_t>(rightLane);
    if (leftDistanceSquared != rightDistanceSquared)
        return leftDistanceSquared < rightDistanceSquared;
    return leftSequence < rightSequence;
}

inline constexpr bool meshLaneCanReserve(size_t schedulerOwned, size_t consumerPending,
                                         MeshPriorityLane lane) {
    const size_t total = schedulerOwned + consumerPending;
    if (total >= EXACT_MESH_MAX_INFLIGHT) return false;
    return lane != MeshPriorityLane::BROAD_SURFACE ||
           total < EXACT_MESH_MAX_INFLIGHT - EXACT_MESH_CAMERA_RESERVED_SLOTS;
}

// ---------------------------------------------------------------------------
// MeshScheduler — chunk meshing off the render thread.
//
// A full chunk build costs ~0.5-2 ms of pure CPU; a streaming burst of 16
// used to consume the whole frame budget on the render thread. Workers pull
// jobs, read immutable column plans, copy one bounded cube halo, mesh
// lock-free, and push results; the render thread's remaining cost per mesh is
// a memcpy into the MegaBuffer.
//
// Owned by RenderPipeline. Not a ThreadPool: it needs a clearable queue and
// version-stamped results. shutdown() MUST run before the World dies — the
// workers hold a reference to it (the engine's quit path does this).
// ---------------------------------------------------------------------------
class MeshScheduler {
public:
    // One frame of bounded work is enough to keep both workers saturated
    // during a 32-chunk streaming burst while the next frame can still
    // reprioritize everything that has not been submitted.
    static constexpr size_t MAX_INFLIGHT_MESH = EXACT_MESH_MAX_INFLIGHT;

    MeshScheduler(const World& world, size_t workerCount);
    ~MeshScheduler();

    // Stop workers and join. Idempotent; called explicitly before the World
    // is destroyed and defensively from the destructor.
    void shutdown();

    // Queue one chunk. Returns false when the in-flight cap is reached or
    // the scheduler is stopping.
    bool enqueue(ChunkPos pos, uint32_t requestedVersion = 0,
                 MeshPriorityLane lane = MeshPriorityLane::BROAD_SURFACE,
                 uint64_t distanceSquared = 0);

    // Move finished results into the consumer's bounded pending vector. The
    // consumer calls this once per frame, including when no new result is
    // expected, so slots freed by uploads become available to enqueue().
    // Results for the same cube coalesce to the newest captured revision.
    void drainCompleted(std::vector<MeshResult>& out);

    // Report the consumer vector after uploads erase processed results. This
    // releases their shared-budget slots before the next enqueue burst.
    void acknowledgeConsumerPending(size_t count);

    size_t inFlight() const { return inFlight_.load(std::memory_order_relaxed); }
    MeshSchedulerStats stats() const;
    float meshMsAvg() const;

private:
    struct MeshJob {
        ChunkPos pos;
        uint32_t requestedVersion = 0;
        MeshPriorityLane lane = MeshPriorityLane::BROAD_SURFACE;
        uint64_t distanceSquared = 0;
        uint64_t sequence = 0;
    };

    const World& world_;
    std::vector<std::thread> workers_;

    std::deque<MeshJob> jobs_;  // guarded by jobMutex_ (leaf lock)
    uint64_t nextSequence_ = 0; // guarded by jobMutex_
    mutable std::mutex jobMutex_;
    std::condition_variable jobCv_;

    std::vector<MeshResult> completed_; // guarded by completedMutex_ (leaf lock)
    mutable std::mutex completedMutex_;

    std::atomic<bool> running_{true};
    // inFlight_ owns every scheduler-side slot, including completed results.
    // consumerPending_ mirrors the vector last passed to drainCompleted().
    // Their sum never exceeds MAX_INFLIGHT_MESH during normal API use.
    std::atomic<size_t> inFlight_{0};
    std::atomic<size_t> consumerPending_{0};
    std::atomic<size_t> highWater_{0};
    std::atomic<uint64_t> coalesced_{0};
    std::atomic<uint64_t> droppedStale_{0};
    AtomicEmaMs meshMs_;

    bool reserveSlot(MeshPriorityLane lane);
    void publishCompleted(MeshResult result);
    void workerLoop();
};
