#include "render/mesh_scheduler.hpp"

#include "common/thread_priority.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <bit>
#include <chrono>

namespace {

auto findResult(std::vector<MeshResult>& results, ChunkPos pos) {
    return std::find_if(results.begin(), results.end(),
                        [pos](const MeshResult& result) { return result.pos == pos; });
}

} // namespace

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

    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        inFlight_.fetch_sub(completed_.size(), std::memory_order_relaxed);
        completed_.clear();
    }
    consumerPending_.store(0, std::memory_order_relaxed);
}

bool MeshScheduler::reserveSlot(MeshPriorityLane lane) {
    size_t owned = inFlight_.load(std::memory_order_relaxed);
    for (;;) {
        const size_t consumer = consumerPending_.load(std::memory_order_relaxed);
        if (!meshLaneCanReserve(owned, consumer, lane)) return false;
        if (inFlight_.compare_exchange_weak(owned, owned + 1, std::memory_order_relaxed)) {
            const size_t total = owned + 1 + consumer;
            size_t high = highWater_.load(std::memory_order_relaxed);
            while (high < total &&
                   !highWater_.compare_exchange_weak(high, total, std::memory_order_relaxed)) {
            }
            return true;
        }
    }
}

bool MeshScheduler::enqueue(ChunkPos pos, uint32_t requestedVersion, MeshPriorityLane lane,
                            uint64_t distanceSquared,
                            std::optional<MeshCanceledRequest>* displaced) {
    if (displaced) displaced->reset();
    if (!running_.load(std::memory_order_relaxed)) return false;

    bool ownsNewSlot = reserveSlot(lane);
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        if (!running_.load(std::memory_order_relaxed)) {
            if (ownsNewSlot) inFlight_.fetch_sub(1, std::memory_order_relaxed);
            return false;
        }

        MeshJob job{pos, requestedVersion, lane, distanceSquared, nextSequence_};
        if (!ownsNewSlot) {
            // At the shared cap, transfer the worst queued slot to a request
            // that ranks ahead of it. This is a true displacement: completed
            // and running work remain untouched, and the bounded ownership
            // count does not change.
            if (jobs_.empty()) return false;
            const MeshJob& worst = jobs_.back();
            if (!meshJobRanksBefore(job.lane, job.distanceSquared, job.sequence, worst.lane,
                                    worst.distanceSquared, worst.sequence)) {
                return false;
            }
            if (displaced) {
                *displaced = MeshCanceledRequest{worst.pos, worst.requestedVersion};
            }
            jobs_.pop_back();
            displaced_.fetch_add(1, std::memory_order_relaxed);
        }

        ++nextSequence_;
        const auto insertion = std::find_if(jobs_.begin(), jobs_.end(), [&](const MeshJob& queued) {
            return meshJobRanksBefore(job.lane, job.distanceSquared, job.sequence, queued.lane,
                                      queued.distanceSquared, queued.sequence);
        });
        jobs_.insert(insertion, job);
    }
    jobCv_.notify_one();
    return true;
}

std::vector<MeshCanceledRequest>
MeshScheduler::cancelQueuedOutside(const std::unordered_set<ChunkPos>& candidates) {
    std::vector<MeshCanceledRequest> canceled;
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        for (auto iterator = jobs_.begin(); iterator != jobs_.end();) {
            if (candidates.contains(iterator->pos)) {
                ++iterator;
                continue;
            }
            canceled.push_back({iterator->pos, iterator->requestedVersion});
            iterator = jobs_.erase(iterator);
        }
        if (!canceled.empty()) {
            inFlight_.fetch_sub(canceled.size(), std::memory_order_relaxed);
            canceledQueued_.fetch_add(canceled.size(), std::memory_order_relaxed);
        }
    }
    if (!canceled.empty()) jobCv_.notify_all();
    return canceled;
}

std::optional<MeshCanceledRequest> MeshScheduler::cancelQueued(ChunkPos position) {
    std::optional<MeshCanceledRequest> canceled;
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        const auto iterator = std::find_if(jobs_.begin(), jobs_.end(), [&](const MeshJob& job) {
            return job.pos == position;
        });
        if (iterator == jobs_.end()) return std::nullopt;
        canceled = MeshCanceledRequest{iterator->pos, iterator->requestedVersion};
        jobs_.erase(iterator);
        inFlight_.fetch_sub(1, std::memory_order_relaxed);
        canceledQueued_.fetch_add(1, std::memory_order_relaxed);
    }
    jobCv_.notify_all();
    return canceled;
}

size_t MeshScheduler::reprioritizeQueued(
    const std::function<MeshRequestPriority(ChunkPos)>& priorityFor) {
    if (!priorityFor) return 0;
    size_t changed = 0;
    {
        std::lock_guard<std::mutex> lock(jobMutex_);
        for (MeshJob& job : jobs_) {
            const MeshRequestPriority priority = priorityFor(job.pos);
            if (job.lane == priority.lane && job.distanceSquared == priority.distanceSquared)
                continue;
            job.lane = priority.lane;
            job.distanceSquared = priority.distanceSquared;
            ++changed;
        }
        std::stable_sort(jobs_.begin(), jobs_.end(), [](const MeshJob& left, const MeshJob& right) {
            return meshJobRanksBefore(left.lane, left.distanceSquared, left.sequence, right.lane,
                                      right.distanceSquared, right.sequence);
        });
    }
    if (changed != 0) jobCv_.notify_all();
    return changed;
}

void MeshScheduler::drainCompleted(std::vector<MeshResult>& out) {
    std::lock_guard<std::mutex> lock(completedMutex_);

    // Observe results the renderer consumed since the previous drain before
    // admitting more work. Keeping the count one frame conservative is what
    // closes the old unbounded scheduler-to-renderer handoff.
    consumerPending_.store(std::min(out.size(), MAX_INFLIGHT_MESH), std::memory_order_relaxed);

    size_t released = 0;
    size_t retained = 0;
    for (size_t index = 0; index < completed_.size(); ++index) {
        MeshResult& result = completed_[index];
        auto existing = findResult(out, result.pos);
        if (existing != out.end()) {
            if (meshResultSupersedes(*existing, result)) {
                *existing = std::move(result);
                coalesced_.fetch_add(1, std::memory_order_relaxed);
            } else {
                droppedStale_.fetch_add(1, std::memory_order_relaxed);
            }
            ++released;
            continue;
        }
        if (out.size() < MAX_INFLIGHT_MESH) {
            out.push_back(std::move(result));
            ++released;
            continue;
        }
        if (retained != index) completed_[retained] = std::move(result);
        ++retained;
    }
    completed_.resize(retained);

    // Count the consumer side before releasing scheduler slots. Concurrent
    // enqueue attempts can be overly conservative during this short window,
    // but can never oversubscribe the shared 64-result contract.
    consumerPending_.store(std::min(out.size(), MAX_INFLIGHT_MESH), std::memory_order_relaxed);
    if (released != 0) inFlight_.fetch_sub(released, std::memory_order_relaxed);
}

void MeshScheduler::acknowledgeConsumerPending(size_t count) {
    consumerPending_.store(std::min(count, MAX_INFLIGHT_MESH), std::memory_order_relaxed);
}

void MeshScheduler::publishCompleted(MeshResult result) {
    std::lock_guard<std::mutex> lock(completedMutex_);
    auto existing = findResult(completed_, result.pos);
    if (existing == completed_.end()) {
        completed_.push_back(std::move(result));
        return;
    }
    if (meshResultSupersedes(*existing, result)) {
        *existing = std::move(result);
        coalesced_.fetch_add(1, std::memory_order_relaxed);
    } else {
        droppedStale_.fetch_add(1, std::memory_order_relaxed);
    }
    inFlight_.fetch_sub(1, std::memory_order_relaxed);
}

void MeshScheduler::workerLoop() {
    setCurrentThreadPriority(ThreadPriority::USER_INITIATED);
    // Per-worker buffers keep their capacity across builds
    thread_local MeshSnapshot snapshot;
    thread_local MeshScratch scratch;

    while (true) {
        MeshJob job;
        {
            std::unique_lock<std::mutex> lock(jobMutex_);
            jobCv_.wait(lock, [this] { return !jobs_.empty() || !running_.load(); });
            if (!running_.load()) {
                return;
            }
            job = jobs_.front();
            jobs_.pop_front();
        }

        MeshResult result;
        result.pos = job.pos;
        result.requestedVersion = job.requestedVersion;
        // snapshotForMeshing reads immutable column metadata, then takes
        // chunksMutex_ for one bounded copy. It never nests that lock with
        // the scheduler's own leaf locks.
        if (world_.snapshotForMeshing(job.pos, snapshot)) {
            result.snapshotOk = true;
            result.builtVersion = snapshot.version;
            auto start = std::chrono::steady_clock::now();
            result.mesh = LODMesher::buildMesh(snapshot, scratch);
            meshMs_.record(
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
        }

        publishCompleted(std::move(result));
    }
}

MeshSchedulerStats MeshScheduler::stats() const {
    MeshSchedulerStats result;
    result.schedulerOwned = inFlight_.load(std::memory_order_relaxed);
    result.consumerPending = consumerPending_.load(std::memory_order_relaxed);
    result.highWater = highWater_.load(std::memory_order_relaxed);
    result.coalesced = coalesced_.load(std::memory_order_relaxed);
    result.droppedStale = droppedStale_.load(std::memory_order_relaxed);
    result.displaced = displaced_.load(std::memory_order_relaxed);
    result.canceledQueued = canceledQueued_.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(completedMutex_);
        result.completed = completed_.size();
    }
    return result;
}

float MeshScheduler::meshMsAvg() const {
    return meshMs_.value();
}
