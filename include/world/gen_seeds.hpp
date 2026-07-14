#pragma once

#include "common/random.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Worldgen seed-offset table — the single place sub-seeds come from.
//
// Every noise field and placement hash derives from the world seed through
// subSeed(worldSeed, key). Keys are grouped by subsystem (0x1xx climate,
// 0x2xx terrain detail, 0x3xx caves, 0x4xx surface, 0x5xx ores, 0x6xx
// features, 0x7xx structures) so a new field can be added without
// decorrelating existing worlds by accident.
// ---------------------------------------------------------------------------

namespace genseed {

inline constexpr uint64_t CONTINENTS = 0x101;
inline constexpr uint64_t EROSION = 0x102;
inline constexpr uint64_t RIDGES = 0x103;
inline constexpr uint64_t TEMPERATURE = 0x104;
inline constexpr uint64_t HUMIDITY = 0x105;
inline constexpr uint64_t ENTRANCE = 0x106;
inline constexpr uint64_t RAVINE = 0x107;

inline constexpr uint64_t DETAIL_3D = 0x201;

inline constexpr uint64_t CHEESE = 0x301;
inline constexpr uint64_t SPAGHETTI_1 = 0x302;
inline constexpr uint64_t SPAGHETTI_2 = 0x303;
inline constexpr uint64_t NOODLE_1 = 0x304;
inline constexpr uint64_t NOODLE_2 = 0x305;

inline constexpr uint64_t BEDROCK = 0x401;
inline constexpr uint64_t SURFACE = 0x402;

inline constexpr uint64_t ORES = 0x500; // + ore index

inline constexpr uint64_t TREES = 0x601;
inline constexpr uint64_t FLORA = 0x602;

inline constexpr uint64_t STRUCTURES = 0x701;

// Derive a subsystem seed from the world seed. The key sits in the high bits
// so consecutive keys produce fully decorrelated hashes.
constexpr uint32_t subSeed(uint32_t worldSeed, uint64_t key) {
    return static_cast<uint32_t>(hash64(static_cast<uint64_t>(worldSeed) ^ (key << 32)));
}

} // namespace genseed
