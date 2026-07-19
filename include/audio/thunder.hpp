#pragma once

#include "world/weather.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

struct ScheduledThunder {
    uint64_t eventId = 0;
    uint64_t eventTick = 0;
    double dueTimeSeconds = 0.0;
    float gain = 0.0F;
    float distanceBlocks = 0.0F;
};

// A bounded main-thread timeline for delayed thunder. beginTimeline() is
// called after a world load or forced-time change so old deterministic strike
// buckets cannot replay. Remembered IDs survive playback and suppress repeated
// weather queries for the same event.
class ThunderScheduler {
public:
    static constexpr size_t MAX_PENDING = 16;
    static constexpr size_t MAX_REMEMBERED_IDS = 64;

    void beginTimeline(uint64_t currentWorldTick);
    bool schedule(const LightningEvent& event, double listenerX, double listenerY, double listenerZ,
                  double nowSeconds);
    std::vector<ScheduledThunder> popDue(double nowSeconds);

    size_t pendingCount() const noexcept { return pending_.size(); }
    uint64_t timelineStartTick() const noexcept { return timelineStartTick_; }

private:
    bool remembers(uint64_t eventId) const noexcept;
    void remember(uint64_t eventId);

    uint64_t timelineStartTick_ = 0;
    bool timelineInitialized_ = false;
    std::vector<ScheduledThunder> pending_;
    std::deque<uint64_t> rememberedIds_;
};
