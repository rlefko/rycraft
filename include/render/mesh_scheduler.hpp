#pragma once

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

// ---------------------------------------------------------------------------
// MeshScheduler — chunk meshing off the render thread.
//
// A full chunk build costs ~0.5-2 ms of pure CPU; a streaming burst of 16
// used to consume the whole frame budget on the render thread. Workers pull
// jobs, snapshot through World::snapshotForMeshing (the only lock they
// touch), mesh lock-free, and push results; the render thread's remaining
// cost per mesh is a memcpy into the MegaBuffer.
//
// Owned by RenderPipeline. Not a ThreadPool: it needs a clearable queue and
// version-stamped results. shutdown() MUST run before the World dies — the
// workers hold a reference to it (the engine's quit path does this).
// ---------------------------------------------------------------------------
class MeshScheduler {
public:
    // Bounded in-flight jobs: tiny on purpose, so per-frame re-prioritization
    // by camera distance needs no queue surgery.
    static constexpr size_t MAX_INFLIGHT_MESH = 8;

    MeshScheduler(const World& world, size_t workerCount);
    ~MeshScheduler();

    // Stop workers and join. Idempotent; called explicitly before the World
    // is destroyed and defensively from the destructor.
    void shutdown();

    // Queue one chunk. Returns false when the in-flight cap is reached or
    // the scheduler is stopping.
    bool enqueue(ChunkPos pos);

    // Swap out all finished results (no allocation on the steady path).
    void drainCompleted(std::vector<MeshResult>& out);

    size_t inFlight() const { return inFlight_.load(std::memory_order_relaxed); }
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
    std::atomic<size_t> inFlight_{0}; // queued + building
    std::atomic<uint32_t> meshMsEmaBits_{0};

    void workerLoop();
    void recordMeshMs(float ms);
};
