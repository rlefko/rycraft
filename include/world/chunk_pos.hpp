#pragma once

#include "common/random.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>

// Horizontal coordinates use 64 bits so procedural sampling is not tied to
// renderer precision. Vertical coordinates stay bounded and fit in 32 bits.
struct ColumnPos {
    int64_t x = 0;
    int64_t z = 0;

    constexpr ColumnPos() = default;
    constexpr ColumnPos(int64_t xValue, int64_t zValue) : x(xValue), z(zValue) {}

    constexpr bool operator==(const ColumnPos&) const = default;
};

struct ChunkPos {
    int64_t x = 0;
    int32_t y = 0;
    int64_t z = 0;

    constexpr ChunkPos() = default;
    constexpr ChunkPos(int64_t xValue, int64_t zValue) : x(xValue), y(0), z(zValue) {}
    constexpr ChunkPos(int64_t xValue, int32_t yValue, int64_t zValue)
        : x(xValue)
        , y(yValue)
        , z(zValue) {}

    constexpr bool operator==(const ChunkPos&) const = default;
};

struct BlockPos {
    int64_t x = 0;
    int32_t y = 0;
    int64_t z = 0;

    constexpr BlockPos() = default;
    constexpr BlockPos(int64_t xValue, int32_t yValue, int64_t zValue)
        : x(xValue)
        , y(yValue)
        , z(zValue) {}

    constexpr bool operator==(const BlockPos&) const = default;
};

namespace world_coord {

inline int64_t floorToNeighborSafeInt64(double value) noexcept {
    constexpr double INT64_LOWER_EXCLUSIVE = -0x1p63;
    constexpr double INT64_UPPER_EXCLUSIVE = 0x1p63;
    constexpr int64_t MINIMUM = std::numeric_limits<int64_t>::min() + 1;
    constexpr int64_t MAXIMUM = std::numeric_limits<int64_t>::max() - 1;
    if (std::isnan(value)) return 0;
    if (value <= INT64_LOWER_EXCLUSIVE) return MINIMUM;
    if (value >= INT64_UPPER_EXCLUSIVE) return MAXIMUM;
    const int64_t floored = static_cast<int64_t>(std::floor(value));
    if (floored < MINIMUM) return MINIMUM;
    if (floored > MAXIMUM) return MAXIMUM;
    return floored;
}

constexpr int64_t floorDiv(int64_t value, int64_t divisor) {
    const int64_t quotient = value / divisor;
    const int64_t remainder = value % divisor;
    return quotient - ((remainder != 0 && ((remainder < 0) != (divisor < 0))) ? 1 : 0);
}

constexpr int32_t floorDiv(int32_t value, int32_t divisor) {
    const int32_t quotient = value / divisor;
    const int32_t remainder = value % divisor;
    return quotient - ((remainder != 0 && ((remainder < 0) != (divisor < 0))) ? 1 : 0);
}

constexpr int64_t floorMultiple(int64_t value, int64_t divisor) {
    return floorDiv(value, divisor) * divisor;
}

constexpr int32_t floorMod(int64_t value, int32_t divisor) {
    const int64_t remainder = value % divisor;
    return static_cast<int32_t>(remainder < 0 ? remainder + divisor : remainder);
}

constexpr int32_t floorMod(int32_t value, int32_t divisor) {
    const int32_t remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

inline size_t mix(size_t seed, uint64_t value) noexcept {
    const uint64_t mixed = hash64(value + 0x9e3779b97f4a7c15ULL);
    return seed ^ (static_cast<size_t>(mixed) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U));
}

} // namespace world_coord

template <>
struct std::hash<ColumnPos> {
    size_t operator()(const ColumnPos& pos) const noexcept {
        size_t seed = world_coord::mix(0, static_cast<uint64_t>(pos.x));
        return world_coord::mix(seed, static_cast<uint64_t>(pos.z));
    }
};

template <>
struct std::hash<ChunkPos> {
    size_t operator()(const ChunkPos& pos) const noexcept {
        size_t seed = world_coord::mix(0, static_cast<uint64_t>(pos.x));
        seed = world_coord::mix(seed, static_cast<uint32_t>(pos.y));
        return world_coord::mix(seed, static_cast<uint64_t>(pos.z));
    }
};

template <>
struct std::hash<BlockPos> {
    size_t operator()(const BlockPos& pos) const noexcept {
        size_t seed = world_coord::mix(0, static_cast<uint64_t>(pos.x));
        seed = world_coord::mix(seed, static_cast<uint32_t>(pos.y));
        return world_coord::mix(seed, static_cast<uint64_t>(pos.z));
    }
};
