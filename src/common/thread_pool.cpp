#include <common/thread_pool.hpp>
#include <common/error.hpp>

#include <chrono>
#include <stdexcept>

// ---------------------------------------------------------------------------
// ThreadPool implementation
// ---------------------------------------------------------------------------
ThreadPool::ThreadPool(size_t numWorkers) {
  for (size_t i = 0; i < numWorkers; ++i) {
    workers_.emplace_back([this]() {
      while (true) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(queueMutex_);
          condition_.wait_for(lock, std::chrono::milliseconds(50), [this]() {
            return stop_.load() || !tasks_.empty();
          });

          // Exit only when stop is requested AND queue is drained
          if (stop_.load() && tasks_.empty()) {
            return;
          }

          if (tasks_.empty()) {
            continue;
          }

          task = std::move(tasks_.front());
          tasks_.pop();
        }
        // lock released here, task runs outside the lock

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
  }
}

ThreadPool::~ThreadPool() {
  // Signal shutdown
  stop_.store(true);
  condition_.notify_all();

  // Explicitly join all worker threads
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}
