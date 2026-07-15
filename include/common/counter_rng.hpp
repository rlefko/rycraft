#pragma once

#include <array>
#include <cstdint>

// Counter-based random access to deterministic values. Unlike a stateful
// generator, each result is addressed by its stream, coordinates, and index.
// Generation order and worker scheduling therefore cannot change a result.
class CounterRng {
public:
    explicit constexpr CounterRng(uint64_t seed) : seed_(seed) {}

    using Block = std::array<uint32_t, 4>;

    constexpr Block block(uint64_t stream, int64_t x, int32_t y, int64_t z,
                          uint32_t index = 0) const {
        Block counter = {
            static_cast<uint32_t>(static_cast<uint64_t>(x)),
            static_cast<uint32_t>(static_cast<uint64_t>(x) >> 32),
            static_cast<uint32_t>(static_cast<uint64_t>(z)),
            static_cast<uint32_t>(static_cast<uint64_t>(z) >> 32),
        };

        uint64_t localAddress = (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 32) | index;
        uint64_t address = mix64(mix64(stream) ^ rotateLeft(mix64(localAddress), 29));
        std::array<uint32_t, 2> key = {
            static_cast<uint32_t>(seed_ ^ address),
            static_cast<uint32_t>((seed_ >> 32) ^ (address >> 32)),
        };

        for (int round = 0; round < 10; ++round) {
            counter = philoxRound(counter, key);
            key[0] += PHILOX_WEYL_0;
            key[1] += PHILOX_WEYL_1;
        }
        return counter;
    }

    constexpr uint32_t u32(uint64_t stream, int64_t x, int32_t y, int64_t z, uint32_t index = 0,
                           uint32_t lane = 0) const {
        return block(stream, x, y, z, index)[lane & 3U];
    }

    constexpr uint64_t u64(uint64_t stream, int64_t x, int32_t y, int64_t z,
                           uint32_t index = 0) const {
        Block value = block(stream, x, y, z, index);
        return static_cast<uint64_t>(value[0]) | (static_cast<uint64_t>(value[1]) << 32);
    }

    // Uniform double in [0, 1) using 53 random mantissa bits.
    constexpr double uniform01(uint64_t stream, int64_t x, int32_t y, int64_t z,
                               uint32_t index = 0) const {
        return static_cast<double>(u64(stream, x, y, z, index) >> 11) * (1.0 / 9007199254740992.0);
    }

    constexpr double signedUnit(uint64_t stream, int64_t x, int32_t y, int64_t z,
                                uint32_t index = 0) const {
        return uniform01(stream, x, y, z, index) * 2.0 - 1.0;
    }

    constexpr int32_t uniformInt(uint64_t stream, int64_t x, int32_t y, int64_t z, uint32_t index,
                                 int32_t min, int32_t max) const {
        if (max <= min) return min;
        uint64_t range = static_cast<uint64_t>(static_cast<int64_t>(max) - min + 1);
        uint64_t scaled = (static_cast<uint64_t>(u32(stream, x, y, z, index)) * range) >> 32;
        return static_cast<int32_t>(static_cast<int64_t>(min) + static_cast<int64_t>(scaled));
    }

private:
    static constexpr uint32_t PHILOX_MULTIPLIER_0 = 0xD2511F53U;
    static constexpr uint32_t PHILOX_MULTIPLIER_1 = 0xCD9E8D57U;
    static constexpr uint32_t PHILOX_WEYL_0 = 0x9E3779B9U;
    static constexpr uint32_t PHILOX_WEYL_1 = 0xBB67AE85U;

    static constexpr uint64_t mix64(uint64_t value) {
        value ^= value >> 30;
        value *= 0xBF58476D1CE4E5B9ULL;
        value ^= value >> 27;
        value *= 0x94D049BB133111EBULL;
        return value ^ (value >> 31);
    }

    static constexpr uint64_t rotateLeft(uint64_t value, unsigned distance) {
        return (value << distance) | (value >> (64U - distance));
    }

    static constexpr Block philoxRound(const Block& counter, const std::array<uint32_t, 2>& key) {
        uint64_t product0 = static_cast<uint64_t>(PHILOX_MULTIPLIER_0) * counter[0];
        uint64_t product1 = static_cast<uint64_t>(PHILOX_MULTIPLIER_1) * counter[2];
        return {
            static_cast<uint32_t>(product1 >> 32) ^ counter[1] ^ key[0],
            static_cast<uint32_t>(product1),
            static_cast<uint32_t>(product0 >> 32) ^ counter[3] ^ key[1],
            static_cast<uint32_t>(product0),
        };
    }

    uint64_t seed_;
};
