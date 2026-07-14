#pragma once

#include <atomic>
#include <bit>
#include <cstdint>

// ---------------------------------------------------------------------------
// AtomicEmaMs — lock-free exponential moving average for timing counters.
//
// Workers CAS-update, the HUD reads. Float bits ride an atomic uint32
// because std::atomic<float> has no portable read-modify-write for the
// blend. One definition: the gen pool and the mesh workers both report
// through this (they used to carry identical copies).
// ---------------------------------------------------------------------------
class AtomicEmaMs {
public:
    void record(float ms) {
        uint32_t oldBits = bits_.load(std::memory_order_relaxed);
        for (;;) {
            float oldEma = std::bit_cast<float>(oldBits);
            float newEma = oldEma == 0.f ? ms : oldEma * 0.9f + ms * 0.1f;
            if (bits_.compare_exchange_weak(oldBits, std::bit_cast<uint32_t>(newEma),
                                            std::memory_order_relaxed)) {
                return;
            }
        }
    }

    float value() const { return std::bit_cast<float>(bits_.load(std::memory_order_relaxed)); }

private:
    std::atomic<uint32_t> bits_{0};
};
