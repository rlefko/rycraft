#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Deterministic hashing and serial visual-effect randomness.
//
// Coordinate-addressed generation and wild spawning use CounterRng instead,
// so query order and worker scheduling cannot advance mutable state. Hashing,
// seed derivation, and serial particle effects use the helpers below.
// ---------------------------------------------------------------------------

// splitmix64 finalizer — fast 64-bit hash with full avalanche.
constexpr uint64_t hash64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Small deterministic PRNG (splitmix64 stream). Cheap to construct, cheap to
// copy, and reproducible from its seed — unlike std::mt19937 there is no
// hidden global state to drift.
class SeededRng {
public:
    explicit constexpr SeededRng(uint64_t seed) : state_(seed) {}

    constexpr uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Uniform float in [0, 1)
    constexpr float nextFloat() {
        return static_cast<float>(next() >> 40) / static_cast<float>(1u << 24);
    }

    // Uniform int in [min, max] (inclusive)
    constexpr int nextInt(int min, int max) {
        if (max <= min) return min;
        uint64_t range = static_cast<uint64_t>(max - min) + 1;
        return min + static_cast<int>(next() % range);
    }

private:
    uint64_t state_;
};
