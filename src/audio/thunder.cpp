#include "audio/thunder.hpp"

#include <algorithm>
#include <cmath>

namespace {

bool hasHigherPlaybackPriority(const ScheduledThunder& left,
                               const ScheduledThunder& right) noexcept {
    if (left.dueTimeSeconds != right.dueTimeSeconds) {
        return left.dueTimeSeconds < right.dueTimeSeconds;
    }
    if (left.distanceBlocks != right.distanceBlocks) {
        return left.distanceBlocks < right.distanceBlocks;
    }
    if (left.eventTick != right.eventTick) {
        return left.eventTick > right.eventTick;
    }
    return left.eventId < right.eventId;
}

} // namespace

void ThunderScheduler::beginTimeline(uint64_t currentWorldTick) {
    timelineStartTick_ = currentWorldTick;
    timelineInitialized_ = true;
    pending_.clear();
    rememberedIds_.clear();
}

bool ThunderScheduler::remembers(uint64_t eventId) const noexcept {
    return std::find(rememberedIds_.begin(), rememberedIds_.end(), eventId) != rememberedIds_.end();
}

void ThunderScheduler::remember(uint64_t eventId) {
    rememberedIds_.push_back(eventId);
    if (rememberedIds_.size() > MAX_REMEMBERED_IDS) {
        rememberedIds_.pop_front();
    }
}

bool ThunderScheduler::schedule(const LightningEvent& event, double listenerX, double listenerY,
                                double listenerZ, double nowSeconds) {
    if (!timelineInitialized_) {
        // Accept the first live event for callers that do not need a load
        // boundary. Engine integration should still begin the timeline at the
        // saved world tick before querying deterministic weather events.
        timelineStartTick_ = event.tick == 0 ? 0 : event.tick - 1;
        timelineInitialized_ = true;
    }
    if (event.tick <= timelineStartTick_ || remembers(event.id) || !std::isfinite(nowSeconds) ||
        !std::isfinite(event.intensity)) {
        return false;
    }

    const double delay = thunderDelaySeconds(event, listenerX, listenerY, listenerZ);
    if (!std::isfinite(delay)) {
        return false;
    }
    const double deltaX = event.x - listenerX;
    const double deltaY = static_cast<double>(event.y) - listenerY;
    const double deltaZ = event.z - listenerZ;
    const float distance =
        static_cast<float>(std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ));
    const float distanceGain = 1.0F / std::sqrt(1.0F + distance / 384.0F);
    const ScheduledThunder candidate{event.id, event.tick, nowSeconds + delay,
                                     std::clamp(event.intensity, 0.0F, 1.25F) * distanceGain,
                                     distance};
    if (!std::isfinite(candidate.dueTimeSeconds) || !std::isfinite(candidate.distanceBlocks)) {
        return false;
    }

    // Every valid strike is consumed exactly once, including a strike dropped
    // under queue pressure. Otherwise a repeated weather query could enqueue
    // it later with a newly computed delay and replay old storm backlog.
    remember(event.id);

    if (pending_.size() >= MAX_PENDING) {
        const auto lowestPriority =
            std::max_element(pending_.begin(), pending_.end(), hasHigherPlaybackPriority);
        if (lowestPriority == pending_.end() ||
            !hasHigherPlaybackPriority(candidate, *lowestPriority)) {
            return false;
        }
        *lowestPriority = candidate;
    } else {
        pending_.push_back(candidate);
    }
    std::sort(pending_.begin(), pending_.end(), hasHigherPlaybackPriority);
    return true;
}

std::vector<ScheduledThunder> ThunderScheduler::popDue(double nowSeconds) {
    std::vector<ScheduledThunder> due;
    if (!std::isfinite(nowSeconds)) {
        return due;
    }
    auto firstFuture = std::upper_bound(
        pending_.begin(), pending_.end(), nowSeconds,
        [](double time, const ScheduledThunder& thunder) { return time < thunder.dueTimeSeconds; });
    due.assign(pending_.begin(), firstFuture);
    pending_.erase(pending_.begin(), firstFuture);
    return due;
}
