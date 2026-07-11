#include <common/thread_pool.hpp>
#include <common/error.hpp>

#include <chrono>
#include <stdexcept>

// ---------------------------------------------------------------------------
// ThreadPool implementation
// ---------------------------------------------------------------------------
ThreadPool::ThreadPool(size_t numWorkers) {
  for (size_t i = 0; i < numWorkers; ++i) {
    workers_.emplace_back([this](std::stop_token stopToken) {
      while (true) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(queueMutex_);
          condition_.wait(lock, [this, &stopToken]() {
            return stop_ || !tasks_.empty() ||
                   stopToken.stop_requested();
          });

          // Exit if shutdown requested and no work remaining
          if (tasks_.empty() && (stop_ || stopToken.stop_requested())) {
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
  stop_ = true;
  condition_.notify_all();
  // jthread destructor requests stop and joins automatically.
  // Workers exit their loop after seeing stop_ and return from emplace_back's lambda.
}
