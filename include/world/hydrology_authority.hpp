#pragma once

#include "world/learned_terrain.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace worldgen::hydrology {

inline constexpr int HYDROLOGY_AUTHORITY_PAGE_BLOCK_EDGE = 2'048;
inline constexpr uint16_t HYDROLOGY_AUTHORITY_SCHEMA_VERSION = 2;
inline constexpr size_t HYDROLOGY_AUTHORITY_MAX_PAYLOAD_BYTES = 256U * 1024U * 1024U;

struct HydrologyPageCoordinate {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const HydrologyPageCoordinate&) const = default;
};

// Persists one immutable opaque hydrology summary per signed 2,048-block
// page. Authority quality, generation fingerprint, and hydrology revision are
// checked before a payload can enter the caller's authority graph.
class HydrologyAuthorityStore {
public:
    struct TestHooks {
        // Test-only seam used to make cross-process-style publication races
        // deterministic. It runs after staging is durable and before the
        // per-page exclusive publication lock is acquired.
        std::function<void()> beforeExclusivePublish;
    };

    HydrologyAuthorityStore(std::filesystem::path root, learned::GenerationIdentity identity,
                            learned::AuthorityQuality quality,
                            std::shared_ptr<const TestHooks> testHooks = nullptr);

    [[nodiscard]] std::filesystem::path pagePath(HydrologyPageCoordinate coordinate) const;
    [[nodiscard]] learned::AuthorityResult<std::vector<uint8_t>>
    load(HydrologyPageCoordinate coordinate) const;
    [[nodiscard]] learned::AuthorityResult<bool> write(HydrologyPageCoordinate coordinate,
                                                       std::span<const uint8_t> payload) const;
    // Replaces an opaque payload only after its semantic consumer has proved
    // that exact persisted bytes are corrupt. A concurrent valid replacement
    // remains immutable and is never overwritten.
    [[nodiscard]] learned::AuthorityResult<bool>
    replaceCorruptPayload(HydrologyPageCoordinate coordinate,
                          std::span<const uint8_t> expectedCorruptPayload,
                          std::span<const uint8_t> replacementPayload) const;

private:
    [[nodiscard]] learned::AuthorityResult<bool>
    writeImpl(HydrologyPageCoordinate coordinate, std::span<const uint8_t> payload,
              std::optional<std::span<const uint8_t>> expectedCorruptPayload) const;

    std::filesystem::path root_;
    learned::GenerationIdentity identity_;
    learned::AuthorityQuality quality_;
    std::shared_ptr<const TestHooks> testHooks_;
};

} // namespace worldgen::hydrology
