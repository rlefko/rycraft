#pragma once

#include <atomic>
#include <condition_variable>
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
    explicit ThreadPool(size_t numWorkers);
    ~ThreadPool();

    // Delete copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Submit a callable, returns future for result retrieval
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
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
            tasks_.emplace([task]() { (*task)(); });
        }

        condition_.notify_one();
        return result;
    }

    // Worker count
    [[nodiscard]] size_t size() const { return workers_.size(); }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queueMutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
};
