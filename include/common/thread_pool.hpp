#pragma once

#include "common/thread_priority.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// ThreadPool — Fixed-size thread pool using C++23 std::jthread
//
// Non-copyable, non-movable. Tasks are submitted via submit() which returns
// a std::future. Worker threads catch exceptions and propagate them to the
// caller's future via std::rethrow_exception.
//
// Destructor signals stop_token to all workers for graceful shutdown.
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(size_t numWorkers, ThreadPriority priority = ThreadPriority::UTILITY,
                        size_t latencySensitiveWorkers = 0);
    ~ThreadPool();

    // Delete copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Submit a callable, returns future for result retrieval
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        return submitWithPriority(0, std::forward<F>(f), std::forward<Args>(args)...);
    }

    // Higher values start first among work that has not begun. Equal-priority
    // submissions retain FIFO order. Running work is never interrupted.
    template <typename F, typename... Args>
    auto submitWithPriority(int64_t priority, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [f = std::forward<F>(f), t = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                try {
                    return std::apply(std::move(f), std::move(t));
                } catch (...) {
                    std::cerr << "[ThreadPool] Task threw an exception" << std::endl;
                    throw;
                }
            });

        std::future<ReturnType> result = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (stop_.load()) {
                throw std::runtime_error("ThreadPool is shutting down");
            }
            tasks_.push({priority, nextSequence_++, [task]() { (*task)(); }});
        }

        condition_.notify_one();
        return result;
    }

    // Worker count
    [[nodiscard]] size_t size() const { return workers_.size(); }

private:
    struct QueuedTask {
        int64_t priority = 0;
        uint64_t sequence = 0;
        std::function<void()> function;
    };
    struct QueuedTaskLater {
        bool operator()(const QueuedTask& left, const QueuedTask& right) const {
            if (left.priority != right.priority) return left.priority < right.priority;
            return left.sequence > right.sequence;
        }
    };

    ThreadPriority priority_;
    size_t latencySensitiveWorkers_ = 0;
    std::vector<std::thread> workers_;
    std::priority_queue<QueuedTask, std::vector<QueuedTask>, QueuedTaskLater> tasks_;
    uint64_t nextSequence_ = 0; // guarded by queueMutex_
    std::mutex queueMutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
};
