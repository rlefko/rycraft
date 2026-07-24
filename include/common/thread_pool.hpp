#pragma once

#include "common/thread_priority.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// ThreadPool - Fixed-size thread pool using std::thread
//
// Non-copyable, non-movable. Tasks are submitted via submit() which returns
// a std::future. Worker threads catch exceptions and propagate them to the
// caller's future via std::rethrow_exception.
//
// Workers retain the synchronized queue state independently of this owner. An
// external shutdown drains accepted work and joins every worker. A worker that
// initiates shutdown detaches only itself, joins its peers, then keeps the
// shared state alive while it drains any remaining accepted work and exits.
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    class TaskCanceled final : public std::exception {
    public:
        const char* what() const noexcept override { return "ThreadPool task canceled"; }
    };

    struct TaskHandle {
        uint64_t id = 0;
        const void* owner = nullptr;
        std::shared_ptr<std::atomic<bool>> cancellation;

        [[nodiscard]] explicit operator bool() const noexcept {
            return id != 0 && owner && cancellation;
        }
    };

    explicit ThreadPool(size_t numWorkers, ThreadPriority priority = ThreadPriority::UTILITY,
                        size_t latencySensitiveWorkers = 0);
    ~ThreadPool();

    // Delete copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Idempotent explicit lifecycle boundary. New submissions are rejected and
    // accepted work drains. An external caller waits for every worker to exit;
    // a worker caller cannot wait for itself and returns after detaching only
    // its own thread object and joining every peer.
    void shutdown();

    // Submit a callable, returns future for result retrieval
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        return submitWithPriority(0, std::forward<F>(f), std::forward<Args>(args)...);
    }

    // Higher values start first among work that has not begun. Equal-priority
    // submissions retain FIFO order. Running work is never interrupted.
    template <typename F, typename... Args>
    auto submitWithPriority(int64_t priority, F&& f,
                            Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        return submitWithPriorityImpl(priority, nullptr, std::forward<F>(f),
                                      std::forward<Args>(args)...);
    }

    // Return a stable handle for work that may become more important while it
    // is still queued. Reprioritization never interrupts a running callable.
    template <typename F, typename... Args>
    auto submitTrackedWithPriority(int64_t priority, TaskHandle& handle, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return submitWithPriorityImpl(priority, &handle, std::forward<F>(f),
                                      std::forward<Args>(args)...);
    }

    // Raises or lowers one accepted task if it has not begun. Returns false
    // after a worker has already removed the task from the queue.
    bool reprioritize(TaskHandle handle, int64_t priority);

    // Marks one queued task canceled and moves its lightweight completion to
    // the front. The callable never runs, but its future becomes ready with
    // TaskCanceled so owners can release admission state immediately.
    bool cancelQueued(TaskHandle handle);

    // Worker count
    [[nodiscard]] size_t size() const { return workers_.size(); }

private:
    template <typename F, typename... Args>
    auto submitWithPriorityImpl(int64_t priority, TaskHandle* handle, F&& f,
                                Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        auto cancellation = std::make_shared<std::atomic<bool>>(false);
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [cancellation, f = std::forward<F>(f),
             t = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                if (cancellation->load(std::memory_order_acquire)) throw TaskCanceled{};
                try {
                    return std::apply(std::move(f), std::move(t));
                } catch (...) {
                    std::cerr << "[ThreadPool] Task threw an exception" << std::endl;
                    throw;
                }
            });

        std::future<ReturnType> result = task->get_future();

        const std::shared_ptr<State> state = state_;
        {
            std::lock_guard<std::mutex> lock(state->queueMutex);
            if (state->stop) {
                throw std::runtime_error("ThreadPool is shutting down");
            }
            const uint64_t taskId = state->nextTaskId++;
            state->tasks.push({priority, state->nextSequence++, taskId, [task]() { (*task)(); }});
            if (handle) {
                handle->id = taskId;
                handle->owner = state.get();
                handle->cancellation = cancellation;
            }
        }

        state->condition.notify_one();
        return result;
    }
    struct QueuedTask {
        int64_t priority = 0;
        uint64_t sequence = 0;
        uint64_t id = 0;
        std::function<void()> function;
    };
    struct QueuedTaskLater {
        bool operator()(const QueuedTask& left, const QueuedTask& right) const {
            if (left.priority != right.priority) return left.priority < right.priority;
            return left.sequence > right.sequence;
        }
    };

    struct State {
        std::priority_queue<QueuedTask, std::vector<QueuedTask>, QueuedTaskLater> tasks;
        uint64_t nextSequence = 0; // guarded by queueMutex
        uint64_t nextTaskId = 1;   // guarded by queueMutex; zero is an invalid handle
        std::mutex queueMutex;
        std::condition_variable condition;
        bool stop = false;        // guarded by queueMutex
        size_t activeWorkers = 0; // guarded by queueMutex
    };

    std::shared_ptr<State> state_;
    std::vector<std::thread> workers_;
    std::mutex shutdownMutex_;
    std::condition_variable shutdownCondition_;
    bool shutdownStarted_ = false;  // guarded by shutdownMutex_
    bool shutdownComplete_ = false; // guarded by shutdownMutex_
};
