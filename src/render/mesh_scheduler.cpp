#include "render/mesh_scheduler.hpp"

#include "world/world.hpp"

#include <bit>
#include <chrono>

MeshScheduler::MeshScheduler(const World& world, size_t workerCount) : world_(world) {
    workers_.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

MeshScheduler::~MeshScheduler() {
    shutdown();
}

void MeshScheduler::shutdown() {
    bool wasRunning = running_.exchange(false);
    if (!wasRunning) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        inFlight_.fetch_sub(jobs_.size(), std::memory_order_relaxed);
        jobs_.clear();
    }
    jobCv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

bool MeshScheduler::enqueue(ChunkPos pos) {
    if (!running_.load(std::memory_order_relaxed) ||
        inFlight_.load(std::memory_order_relaxed) >= MAX_INFLIGHT_MESH) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        jobs_.push_back(pos);
    }
    inFlight_.fetch_add(1, std::memory_order_relaxed);
    jobCv_.notify_one();
    return true;
}

void MeshScheduler::drainCompleted(std::vector<MeshResult>& out) {
    std::lock_guard<std::mutex> lock(completedMutex_);
    if (out.empty()) {
        out.swap(completed_);
    } else {
        // Rare path: the caller still holds results (upload budget ran out)
        for (MeshResult& result : completed_) {
            out.push_back(std::move(result));
        }
        completed_.clear();
    }
}

void MeshScheduler::workerLoop() {
    // Per-worker buffers keep their capacity across builds
    thread_local MeshSnapshot snapshot;
    thread_local MeshScratch scratch;

    while (true) {
        ChunkPos pos;
        {
            std::unique_lock<std::mutex> lock(jobMutex_);
            jobCv_.wait(lock, [this] { return !jobs_.empty() || !running_.load(); });
            if (!running_.load()) {
                return;
            }
            pos = jobs_.front();
            jobs_.pop_front();
        }

        MeshResult result;
        result.pos = pos;
        // snapshotForMeshing takes chunksMutex_ for one bounded copy — the
        // only lock a mesh worker ever holds, and never together with the
        // scheduler's own (leaf) locks
        if (world_.snapshotForMeshing(pos, snapshot)) {
            result.snapshotOk = true;
            result.builtVersion = snapshot.version;
            auto start = std::chrono::steady_clock::now();
            result.mesh = LODMesher::buildMesh(snapshot, scratch);
            recordMeshMs(
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
        }

        {
            std::lock_guard<std::mutex> lock(completedMutex_);
            completed_.push_back(std::move(result));
        }
        inFlight_.fetch_sub(1, std::memory_order_relaxed);
    }
}

void MeshScheduler::recordMeshMs(float ms) {
    uint32_t oldBits = meshMsEmaBits_.load(std::memory_order_relaxed);
    for (;;) {
        float oldEma = std::bit_cast<float>(oldBits);
        float newEma = oldEma == 0.f ? ms : oldEma * 0.9f + ms * 0.1f;
        if (meshMsEmaBits_.compare_exchange_weak(oldBits, std::bit_cast<uint32_t>(newEma),
                                                 std::memory_order_relaxed)) {
            return;
        }
    }
}

float MeshScheduler::meshMsAvg() const {
    return std::bit_cast<float>(meshMsEmaBits_.load(std::memory_order_relaxed));
}
