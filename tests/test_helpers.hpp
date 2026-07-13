#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>

#include <unistd.h>

// RAII temp directory for tests that touch the filesystem. Unique per
// process and per instance, cleaned up on destruction — no fixed /tmp paths,
// no std::system("rm -rf"), and concurrent test runs never collide.
class TempDir {
public:
    explicit TempDir(const std::string& name) {
        static std::atomic<uint64_t> counter{0};
        path_ = (std::filesystem::temp_directory_path() /
                 ("rycraft_test_" + name + "_" + std::to_string(::getpid()) + "_" +
                  std::to_string(counter++)))
                    .string();
        std::filesystem::remove_all(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};
