#include <common/error.hpp>
#include <common/thread_pool.hpp>

#include <chrono>
#include <limits>
#include <stdexcept>

namespace {

thread_local const void* activeThreadPoolState = nullptr;

class ActiveThreadPoolStateScope {
public:
    explicit ActiveThreadPoolStateScope(const void* state)
        : previous_(activeThreadPoolState) {
        activeThreadPoolState = state;
    }

    ~ActiveThreadPoolStateScope() { activeThreadPoolState = previous_; }

private:
    const void* previous_ = nullptr;
};

} // namespace

// ---------------------------------------------------------------------------
// ThreadPool implementation
// ---------------------------------------------------------------------------
ThreadPool::ThreadPool(size_t numWorkers, ThreadPriority priority, size_t latencySensitiveWorkers)
    : state_(std::make_shared<State>()) {
    if (latencySensitiveWorkers > numWorkers) {
        throw std::invalid_argument("latency-sensitive worker count exceeds pool size");
    }
    workers_.reserve(numWorkers);
    try {
        for (size_t i = 0; i < numWorkers; ++i) {
            const std::shared_ptr<State> state = state_;
            const ThreadPriority workerPriority =
                i < latencySensitiveWorkers ? ThreadPriority::USER_INITIATED : priority;
            {
                std::lock_guard lock(state->queueMutex);
                ++state->activeWorkers;
            }
            try {
                workers_.emplace_back([state, workerPriority]() {
                    ActiveThreadPoolStateScope activeState(state.get());
                    struct WorkerExit {
                        std::shared_ptr<State> state;
                        ~WorkerExit() {
                            {
                                std::lock_guard lock(state->queueMutex);
                                if (state->activeWorkers > 0) --state->activeWorkers;
                            }
                            state->condition.notify_all();
                        }
                    } workerExit{state};

                    setCurrentThreadPriority(workerPriority);
                    while (true) {
                        std::function<void()> task;

                        {
                            std::unique_lock lock(state->queueMutex);
                            state->condition.wait_for(
                                lock, std::chrono::milliseconds(50),
                                [&state]() { return state->stop || !state->tasks.empty(); });

                            // Exit only when stop is requested AND the queue is drained.
                            if (state->stop && state->tasks.empty()) return;
                            if (state->tasks.empty()) continue;

                            task = state->tasks.top().function;
                            state->tasks.pop();
                        }

                        try {
                            task();
                        } catch (const std::exception& e) {
                            RY_LOG_ERROR(std::string("ThreadPool worker caught: ") + e.what());
                            throw;
                        } catch (...) {
                            RY_LOG_ERROR("ThreadPool worker caught unknown exception");
                            throw;
                        }
                    }
                });
            } catch (...) {
                std::lock_guard lock(state->queueMutex);
                --state->activeWorkers;
                throw;
            }
        }
    } catch (...) {
        {
            std::lock_guard lock(state_->queueMutex);
            state_->stop = true;
        }
        state_->condition.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        throw;
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

bool ThreadPool::reprioritize(TaskHandle handle, int64_t priority) {
    const std::shared_ptr<State> state = state_;
    if (!handle || handle.owner != state.get()) return false;

    bool found = false;
    std::vector<QueuedTask> queued;
    {
        std::lock_guard lock(state->queueMutex);
        queued.reserve(state->tasks.size());
        while (!state->tasks.empty()) {
            QueuedTask task = state->tasks.top();
            state->tasks.pop();
            if (task.id == handle.id) {
                task.priority = priority;
                found = true;
            }
            queued.push_back(std::move(task));
        }
        for (QueuedTask& task : queued)
            state->tasks.push(std::move(task));
    }
    if (found) state->condition.notify_all();
    return found;
}

bool ThreadPool::cancelQueued(TaskHandle handle) {
    const std::shared_ptr<State> state = state_;
    if (!handle || handle.owner != state.get()) return false;

    bool found = false;
    std::vector<QueuedTask> queued;
    {
        std::lock_guard lock(state->queueMutex);
        queued.reserve(state->tasks.size());
        while (!state->tasks.empty()) {
            QueuedTask task = state->tasks.top();
            state->tasks.pop();
            if (task.id == handle.id) {
                handle.cancellation->store(true, std::memory_order_release);
                task.priority = std::numeric_limits<int64_t>::max();
                found = true;
            }
            queued.push_back(std::move(task));
        }
        for (QueuedTask& task : queued)
            state->tasks.push(std::move(task));
    }
    if (found) state->condition.notify_all();
    return found;
}

void ThreadPool::shutdown() {
    const std::shared_ptr<State> state = state_;
    const bool callerIsWorker = activeThreadPoolState == state.get();
    bool ownsShutdown = false;
    {
        std::unique_lock lock(shutdownMutex_);
        if (!shutdownStarted_) {
            shutdownStarted_ = true;
            ownsShutdown = true;
            {
                std::lock_guard stateLock(state->queueMutex);
                state->stop = true;
            }
        } else if (!shutdownComplete_) {
            // A worker must return so the thread performing shutdown can join
            // it. External callers can wait for thread-object disposition.
            if (callerIsWorker) return;
            shutdownCondition_.wait(lock, [this] { return shutdownComplete_; });
        }
    }

    if (ownsShutdown) {
        state->condition.notify_all();
        const std::thread::id caller = std::this_thread::get_id();
        for (std::thread& worker : workers_) {
            if (!worker.joinable()) continue;
            if (worker.get_id() == caller) {
                worker.detach();
            } else {
                worker.join();
            }
        }
        {
            std::lock_guard lock(shutdownMutex_);
            shutdownComplete_ = true;
        }
        shutdownCondition_.notify_all();
    }

    // A detached initiating worker still owns State and drains the queue after
    // returning from this method. External callers retain the ordinary drain
    // guarantee by waiting for that final worker to leave.
    if (!callerIsWorker) {
        std::unique_lock lock(state->queueMutex);
        state->condition.wait(lock, [&state] { return state->activeWorkers == 0; });
    }
}
