#pragma once

#include "common/random.hpp"

#include <cstdint>
#include <functional>

// ---------------------------------------------------------------------------
// ChunkPos — THE chunk grid coordinate/key type.
//
// World storage, the renderer's mesh cache, and anything else addressing
// chunks keys on this. (Three ad-hoc schemes used to coexist: World built
// heap-allocating "x_z" strings on every block access, the renderer packed
// its own uint64, and the spatial hash packed an int64.)
// ---------------------------------------------------------------------------
struct ChunkPos {
    int32_t x = 0;
    int32_t z = 0;

    constexpr bool operator==(const ChunkPos&) const = default;

    // Stable 64-bit packing (x in the high half, z in the low half).
    constexpr uint64_t packed() const {
        return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(z));
    }
};

template <>
struct std::hash<ChunkPos> {
    size_t operator()(const ChunkPos& p) const noexcept {
        return static_cast<size_t>(hash64(p.packed()));
    }
};
