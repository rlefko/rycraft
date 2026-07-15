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
    uint32_t builtVersion = 0; // chunk revision the snapshot captured
    bool snapshotOk = false;   // false: a neighbor wasn't generated yet
    MeshOutput mesh;
};

struct MeshSchedulerStats {
    size_t schedulerOwned = 0; // queued + building + completed
    size_t consumerPending = 0;
    size_t completed = 0;
    size_t highWater = 0;
    uint64_t coalesced = 0;
    uint64_t droppedStale = 0;
};

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
    static constexpr size_t MAX_INFLIGHT_MESH = 64;

    MeshScheduler(const World& world, size_t workerCount);
    ~MeshScheduler();

    // Stop workers and join. Idempotent; called explicitly before the World
    // is destroyed and defensively from the destructor.
    void shutdown();

    // Queue one chunk. Returns false when the in-flight cap is reached or
    // the scheduler is stopping.
    bool enqueue(ChunkPos pos);

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
    const World& world_;
    std::vector<std::thread> workers_;

    std::deque<ChunkPos> jobs_; // guarded by jobMutex_ (leaf lock)
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

    bool reserveSlot();
    void publishCompleted(MeshResult result);
    void workerLoop();
};
