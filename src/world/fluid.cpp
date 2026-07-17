#include "world/fluid.hpp"

#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr FluidPos offsetPosition(FluidPos position, FluidDirection direction) {
    switch (direction) {
        case FluidDirection::CENTER:
            return position;
        case FluidDirection::DOWN:
            --position.y;
            return position;
        case FluidDirection::UP:
            ++position.y;
            return position;
        case FluidDirection::WEST:
            --position.x;
            return position;
        case FluidDirection::EAST:
            ++position.x;
            return position;
        case FluidDirection::NORTH:
            --position.z;
            return position;
        case FluidDirection::SOUTH:
            ++position.z;
            return position;
    }
    return position;
}

constexpr std::array<FluidDirection, 6> FACE_DIRECTIONS = {
    FluidDirection::DOWN, FluidDirection::UP,    FluidDirection::WEST,
    FluidDirection::EAST, FluidDirection::NORTH, FluidDirection::SOUTH,
};

constexpr std::array<FluidDirection, 4> HORIZONTAL_DIRECTIONS = {
    FluidDirection::WEST,
    FluidDirection::EAST,
    FluidDirection::NORTH,
    FluidDirection::SOUTH,
};

const FluidCell& cellInDirection(const FluidNeighborhood& cells, FluidDirection direction) {
    switch (direction) {
        case FluidDirection::CENTER:
            return cells.center;
        case FluidDirection::DOWN:
            return cells.down;
        case FluidDirection::UP:
            return cells.up;
        case FluidDirection::WEST:
            return cells.west;
        case FluidDirection::EAST:
            return cells.east;
        case FluidDirection::NORTH:
            return cells.north;
        case FluidDirection::SOUTH:
            return cells.south;
    }
    return cells.center;
}

void appendDeferred(FluidRuleResult& result, FluidDirection direction) {
    for (uint8_t i = 0; i < result.deferredCount; ++i) {
        if (result.deferred[i] == direction) {
            return;
        }
    }
    result.deferred[result.deferredCount++] = direction;
}

void appendSet(FluidRuleResult& result, FluidDirection direction, FluidState state) {
    result.mutations[result.mutationCount++] = {
        .direction = direction,
        .type = FluidMutationType::SET_WATER,
        .state = state,
    };
}

void appendRemove(FluidRuleResult& result, FluidDirection direction) {
    result.mutations[result.mutationCount++] = {
        .direction = direction,
        .type = FluidMutationType::REMOVE_WATER,
        .state = FluidState::source(),
    };
}

bool supportsSource(const FluidCell& cell) {
    return cell.loaded && (isSolid(cell.block) || (cell.isWater() && cell.state.isSource()));
}

bool acceptsDownwardWater(const FluidCell& cell) {
    return isWaterReplaceable(cell.block) || (cell.isWater() && !cell.state.isSource());
}

bool acceptsHorizontalWater(const FluidCell& cell, FluidState candidate) {
    if (isWaterReplaceable(cell.block)) {
        return true;
    }
    if (!cell.isWater() || cell.state.isSource()) {
        return false;
    }
    // A supported falling column already has a stronger vertical feed than
    // horizontal runoff. Replacing it with a side-flow level makes adjacent
    // waterfall lanes repeatedly rewrite one another at the receiving pool.
    if (cell.state.isFalling()) return false;
    return cell.state.level() > candidate.level();
}

bool positionLess(const FluidPos& left, const FluidPos& right) {
    if (left.x != right.x) {
        return left.x < right.x;
    }
    if (left.y != right.y) {
        return left.y < right.y;
    }
    return left.z < right.z;
}

struct HorizontalWaterStats {
    unsigned sources = 0;
    uint8_t minimumLevel = std::numeric_limits<uint8_t>::max();
};

HorizontalWaterStats collectHorizontalWater(const FluidNeighborhood& cells,
                                            FluidRuleResult& result) {
    HorizontalWaterStats stats;
    for (FluidDirection direction : HORIZONTAL_DIRECTIONS) {
        const FluidCell& neighbor = cellInDirection(cells, direction);
        if (!neighbor.loaded) {
            appendDeferred(result, direction);
            continue;
        }
        if (!neighbor.isWater()) {
            continue;
        }
        if (neighbor.state.isSource()) {
            ++stats.sources;
        }
        stats.minimumLevel = std::min(stats.minimumLevel, neighbor.state.level());
    }
    return stats;
}

struct EffectiveWater {
    bool present = false;
    bool originalSource = false;
    FluidState state = FluidState::source();
};

EffectiveWater resolveEffectiveWater(const FluidNeighborhood& cells, FluidRuleResult& result) {
    const FluidCell& center = cells.center;
    if (center.isWater() && center.state.isSource()) {
        return {.present = true, .originalSource = true, .state = FluidState::source()};
    }

    HorizontalWaterStats horizontal = collectHorizontalWater(cells, result);
    if (!cells.up.loaded) {
        appendDeferred(result, FluidDirection::UP);
    }
    if (!cells.down.loaded) {
        appendDeferred(result, FluidDirection::DOWN);
    }

    EffectiveWater effective;
    if (horizontal.sources >= 2 && supportsSource(cells.down)) {
        effective = {.present = true, .state = FluidState::source()};
    } else if (cells.up.isWater()) {
        effective = {
            .present = true,
            .state = FluidState::falling(std::max<uint8_t>(1, cells.up.state.level())),
        };
    } else if (horizontal.minimumLevel < 7) {
        effective = {
            .present = true,
            .state = FluidState::flowing(static_cast<uint8_t>(horizontal.minimumLevel + 1)),
        };
    }

    if (effective.present && (!center.isWater() || center.state != effective.state)) {
        appendSet(result, FluidDirection::CENTER, effective.state);
    } else if (!effective.present && center.isWater()) {
        appendRemove(result, FluidDirection::CENTER);
    }
    return effective;
}

void appendSourceDeferrals(const FluidNeighborhood& cells, FluidRuleResult& result) {
    for (FluidDirection direction : HORIZONTAL_DIRECTIONS) {
        if (!cellInDirection(cells, direction).loaded) {
            appendDeferred(result, direction);
        }
    }
}

void appendHorizontalFlow(const FluidNeighborhood& cells, FluidState effectiveState,
                          FluidRuleResult& result) {
    uint8_t horizontalLevel =
        effectiveState.isFalling() ? 1 : static_cast<uint8_t>(effectiveState.level() + 1);
    if (horizontalLevel > 7) {
        return;
    }

    FluidState horizontal = FluidState::flowing(horizontalLevel);
    for (FluidDirection direction : HORIZONTAL_DIRECTIONS) {
        const FluidCell& neighbor = cellInDirection(cells, direction);
        if (!neighbor.loaded) {
            appendDeferred(result, direction);
            continue;
        }
        if (!acceptsHorizontalWater(neighbor, horizontal)) {
            continue;
        }
        if (!neighbor.isWater() || neighbor.state != horizontal) {
            appendSet(result, direction, horizontal);
        }
    }
}

struct FluidFrontierLess {
    bool operator()(const FluidBoundaryFrontier& left,
                    const FluidBoundaryFrontier& right) const noexcept {
        if (left.unavailable != right.unavailable) {
            return positionLess(left.unavailable, right.unavailable);
        }
        return positionLess(left.available, right.available);
    }
};

ChunkPos containingChunk(FluidPos position) {
    return {
        Chunk::worldToChunk(position.x),
        Chunk::worldToChunkY(position.y),
        Chunk::worldToChunk(position.z),
    };
}

struct ScheduledFluidUpdate {
    uint64_t dueTick = 0;
    FluidPos position;
};

struct ScheduledFluidUpdateLater {
    bool operator()(const ScheduledFluidUpdate& left, const ScheduledFluidUpdate& right) const {
        if (left.dueTick != right.dueTick) {
            return left.dueTick > right.dueTick;
        }
        return positionLess(right.position, left.position);
    }
};

} // namespace

size_t FluidPosHash::operator()(const FluidPos& position) const noexcept {
    return std::hash<BlockPos>{}(position);
}

FluidRuleResult evaluateWaterRules(const FluidNeighborhood& cells) noexcept {
    FluidRuleResult result;
    const FluidCell& center = cells.center;
    if (!center.loaded) {
        return result;
    }
    if (!center.isWater() && !isWaterReplaceable(center.block)) {
        return result;
    }

    // The lower cell decides whether water falls, forms a supported source,
    // or may spread horizontally. Preserve only that dependency until its
    // cube is available so no provisional center or side mutation leaks out.
    if (!cells.down.loaded) {
        appendDeferred(result, FluidDirection::DOWN);
        return result;
    }

    // Flowing water is supported by the cell above or by its four horizontal
    // neighbors. Treat any unavailable support dependency as unknown instead
    // of dry, otherwise a boundary update can provisionally weaken or remove
    // water before the neighboring cube arrives. Sources are stable without
    // those dependencies and may still spread into every resident direction.
    const bool sourceCenter = center.isWater() && center.state.isSource();
    if (!sourceCenter) {
        bool missingSupport = false;
        if (!cells.up.loaded) {
            appendDeferred(result, FluidDirection::UP);
            missingSupport = true;
        }
        for (FluidDirection direction : HORIZONTAL_DIRECTIONS) {
            if (!cellInDirection(cells, direction).loaded) {
                appendDeferred(result, direction);
                missingSupport = true;
            }
        }
        if (missingSupport) {
            return result;
        }
    }

    EffectiveWater effective = resolveEffectiveWater(cells, result);
    if (!effective.present) {
        return result;
    }

    // A source does not depend on water above, but missing horizontal cells
    // still need a frontier because the source may enter them.
    if (effective.originalSource) {
        appendSourceDeferrals(cells, result);
    }

    if (acceptsDownwardWater(cells.down)) {
        FluidState downward = FluidState::falling(std::max<uint8_t>(1, effective.state.level()));
        if (!cells.down.isWater() || cells.down.state != downward) {
            appendSet(result, FluidDirection::DOWN, downward);
        }
        return result;
    }

    // A falling column remains vertical until it reaches solid ground. A
    // source receiver already supplies the landing pool, so spreading from
    // the falling cell itself would create a second, artificial horizontal
    // origin around every generated waterfall.
    if (effective.state.isFalling() && cells.down.isWater()) {
        return result;
    }

    appendHorizontalFlow(cells, effective.state, result);
    return result;
}

class FluidScheduler::Impl {
public:
    using DeferredBucket = std::set<FluidBoundaryFrontier, FluidFrontierLess>;

    struct ResumeCursor {
        DeferredBucket::const_iterator current;
        DeferredBucket::const_iterator end;
    };

    struct CursorLater {
        const std::vector<ResumeCursor>* cursors = nullptr;

        bool operator()(size_t left, size_t right) const {
            return FluidFrontierLess{}(*(*cursors)[right].current, *(*cursors)[left].current);
        }
    };

    explicit Impl(FluidSchedulerLimits limits) : limits_(limits) {
        limits_.updatesPerTick = std::max<size_t>(1, limits_.updatesPerTick);
        limits_.pendingUpdates = std::max<size_t>(7, limits_.pendingUpdates);
        limits_.deferredFrontiers = std::max<size_t>(1, limits_.deferredFrontiers);
        limits_.catchUpTicks = std::max<uint32_t>(1, limits_.catchUpTicks);
    }

    bool enqueue(FluidPos position) {
        uint64_t dueTick = currentTick_ + WATER_UPDATE_DELAY_TICKS;
        auto existing = pending_.find(position);
        if (existing != pending_.end()) {
            if (dueTick >= existing->second) {
                return true;
            }
            existing->second = dueTick;
            queue_.push({dueTick, position});
            return true;
        }
        if (pending_.size() >= limits_.pendingUpdates) {
            ++droppedUpdates_;
            return false;
        }
        pending_.emplace(position, dueTick);
        queue_.push({dueTick, position});
        return true;
    }

    size_t enqueueAround(FluidPos position) {
        size_t accepted = enqueue(position) ? 1 : 0;
        for (FluidDirection direction : FACE_DIRECTIONS) {
            accepted += enqueue(offsetPosition(position, direction)) ? 1 : 0;
        }
        return accepted;
    }

    bool defer(FluidPos available, FluidPos unavailable) {
        FluidBoundaryFrontier frontier{.available = available, .unavailable = unavailable};
        const ChunkPos unavailableChunk = containingChunk(unavailable);
        auto existingBucket = deferredByChunk_.find(unavailableChunk);
        if (existingBucket != deferredByChunk_.end() && existingBucket->second.contains(frontier)) {
            return true;
        }
        if (deferredCount_ >= limits_.deferredFrontiers) {
            ++droppedFrontiers_;
            return false;
        }
        auto [bucket, inserted] = deferredByChunk_.try_emplace(unavailableChunk);
        (void)inserted;
        bucket->second.insert(frontier);
        ++deferredCount_;
        return true;
    }

    void eraseDeferred(const FluidBoundaryFrontier& frontier) {
        const ChunkPos unavailableChunk = containingChunk(frontier.unavailable);
        auto bucket = deferredByChunk_.find(unavailableChunk);
        if (bucket == deferredByChunk_.end()) {
            return;
        }
        const size_t erased = bucket->second.erase(frontier);
        deferredCount_ -= erased;
        if (bucket->second.empty()) {
            deferredByChunk_.erase(bucket);
        }
    }

    FluidNeighborhood sample(FluidPos position, const FluidWorldAccess& world) const {
        return {
            .center = world.readFluidCell(position),
            .down = world.readFluidCell(offsetPosition(position, FluidDirection::DOWN)),
            .up = world.readFluidCell(offsetPosition(position, FluidDirection::UP)),
            .west = world.readFluidCell(offsetPosition(position, FluidDirection::WEST)),
            .east = world.readFluidCell(offsetPosition(position, FluidDirection::EAST)),
            .north = world.readFluidCell(offsetPosition(position, FluidDirection::NORTH)),
            .south = world.readFluidCell(offsetPosition(position, FluidDirection::SOUTH)),
        };
    }

    void process(FluidPos position, FluidWorldAccess& world) {
        FluidNeighborhood cells = sample(position, world);
        if (!cells.center.loaded) {
            return;
        }

        FluidRuleResult result = evaluateWaterRules(cells);
        for (uint8_t i = 0; i < result.deferredCount; ++i) {
            FluidPos unavailable = offsetPosition(position, result.deferred[i]);
            defer(position, unavailable);
        }

        for (uint8_t i = 0; i < result.mutationCount; ++i) {
            const FluidMutation& mutation = result.mutations[i];
            FluidPos target = offsetPosition(position, mutation.direction);
            if (mutation.type == FluidMutationType::SET_WATER) {
                world.writeWater(target, mutation.state);
            } else {
                world.removeWater(target);
            }
            enqueueAround(target);
        }
    }

    size_t tick(FluidWorldAccess& world) {
        ++currentTick_;
        size_t processed = 0;
        while (processed < limits_.updatesPerTick && !queue_.empty() &&
               queue_.top().dueTick <= currentTick_) {
            ScheduledFluidUpdate update = queue_.top();
            queue_.pop();

            auto pending = pending_.find(update.position);
            if (pending == pending_.end() || pending->second != update.dueTick) {
                continue;
            }
            pending_.erase(pending);
            process(update.position, world);
            ++processed;
        }
        return processed;
    }

    size_t advance(double elapsedSeconds, FluidWorldAccess& world) {
        if (!std::isfinite(elapsedSeconds) || elapsedSeconds <= 0.0) {
            return 0;
        }
        constexpr double TICK_SECONDS = 1.0 / static_cast<double>(FLUID_TICKS_PER_SECOND);
        double maximumDebt = static_cast<double>(limits_.catchUpTicks) * TICK_SECONDS;
        accumulator_ = std::min(maximumDebt, accumulator_ + elapsedSeconds);
        uint32_t ticks = static_cast<uint32_t>(accumulator_ / TICK_SECONDS);
        ticks = std::min(ticks, limits_.catchUpTicks);

        size_t processed = 0;
        for (uint32_t i = 0; i < ticks; ++i) {
            processed += tick(world);
        }
        accumulator_ -= static_cast<double>(ticks) * TICK_SECONDS;
        return processed;
    }

    std::vector<const DeferredBucket*> deferredBucketsIn(const FluidBounds& bounds) const {
        std::vector<const DeferredBucket*> buckets;
        const ChunkPos minimumChunk = containingChunk({bounds.minX, bounds.minY, bounds.minZ});
        const ChunkPos maximumChunk = containingChunk({bounds.maxX, bounds.maxY, bounds.maxZ});
        if (minimumChunk == maximumChunk) {
            auto bucket = deferredByChunk_.find(minimumChunk);
            if (bucket != deferredByChunk_.end()) {
                buckets.push_back(&bucket->second);
            }
            return buckets;
        }

        // Wide diagnostic or persistence queries are rare. Inspecting the
        // compact chunk index avoids walking a potentially enormous empty
        // coordinate range while still never scanning unrelated entries.
        for (const auto& [chunk, bucket] : deferredByChunk_) {
            if (chunk.x >= minimumChunk.x && chunk.x <= maximumChunk.x &&
                chunk.y >= minimumChunk.y && chunk.y <= maximumChunk.y &&
                chunk.z >= minimumChunk.z && chunk.z <= maximumChunk.z) {
                buckets.push_back(&bucket);
            }
        }
        return buckets;
    }

    static void skipOutside(ResumeCursor& cursor, const FluidBounds& bounds) {
        while (cursor.current != cursor.end && !bounds.contains(cursor.current->unavailable)) {
            ++cursor.current;
        }
    }

    std::vector<FluidBoundaryFrontier> matchingFrontiers(const FluidBounds& bounds,
                                                         size_t maximumFrontiers) const {
        const std::vector<const DeferredBucket*> buckets = deferredBucketsIn(bounds);
        std::vector<ResumeCursor> cursors;
        cursors.reserve(buckets.size());
        size_t matchingCapacity = 0;
        for (const DeferredBucket* bucket : buckets) {
            ResumeCursor cursor{.current = bucket->cbegin(), .end = bucket->cend()};
            skipOutside(cursor, bounds);
            if (cursor.current != cursor.end) {
                cursors.push_back(cursor);
                matchingCapacity += std::min(bucket->size(), maximumFrontiers - matchingCapacity);
            }
        }

        std::priority_queue<size_t, std::vector<size_t>, CursorLater> ready(
            CursorLater{.cursors = &cursors});
        for (size_t i = 0; i < cursors.size(); ++i) {
            ready.push(i);
        }

        std::vector<FluidBoundaryFrontier> matching;
        matching.reserve(matchingCapacity);
        while (!ready.empty() && matching.size() < maximumFrontiers) {
            const size_t cursorIndex = ready.top();
            ready.pop();
            ResumeCursor& cursor = cursors[cursorIndex];
            matching.push_back(*cursor.current);
            ++cursor.current;
            skipOutside(cursor, bounds);
            if (cursor.current != cursor.end) {
                ready.push(cursorIndex);
            }
        }
        return matching;
    }

    size_t resumeDeferredIn(const FluidBounds& bounds, size_t maximumFrontiers) {
        if (maximumFrontiers == 0 || deferredCount_ == 0 || bounds.minX > bounds.maxX ||
            bounds.minY > bounds.maxY || bounds.minZ > bounds.maxZ) {
            return 0;
        }

        size_t resumed = 0;
        for (const FluidBoundaryFrontier& frontier : matchingFrontiers(bounds, maximumFrontiers)) {
            bool acceptedAvailable = enqueue(frontier.available);
            bool acceptedUnavailable = enqueue(frontier.unavailable);
            if (!acceptedAvailable || !acceptedUnavailable) {
                continue;
            }
            eraseDeferred(frontier);
            ++resumed;
        }
        return resumed;
    }

    std::vector<FluidBoundaryFrontier> deferredFrontiers() const {
        std::vector<FluidBoundaryFrontier> frontiers;
        frontiers.reserve(deferredCount_);
        for (const auto& [chunk, bucket] : deferredByChunk_) {
            (void)chunk;
            frontiers.insert(frontiers.end(), bucket.cbegin(), bucket.cend());
        }
        std::ranges::sort(frontiers, FluidFrontierLess{});
        return frontiers;
    }

    size_t resumeDeferredIn(const FluidBounds& bounds) {
        return resumeDeferredIn(bounds, std::numeric_limits<size_t>::max());
    }

    size_t deferredCountIn(const FluidBounds& bounds) const {
        size_t count = 0;
        for (const DeferredBucket* bucket : deferredBucketsIn(bounds)) {
            count += static_cast<size_t>(std::count_if(
                bucket->cbegin(), bucket->cend(), [&bounds](const FluidBoundaryFrontier& frontier) {
                    return bounds.contains(frontier.unavailable);
                }));
        }
        return count;
    }

    size_t deferredCount() const { return deferredCount_; }

    void clear() {
        queue_ = {};
        pending_.clear();
        deferredByChunk_.clear();
        deferredCount_ = 0;
        currentTick_ = 0;
        accumulator_ = 0.0;
        droppedUpdates_ = 0;
        droppedFrontiers_ = 0;
    }

    FluidSchedulerLimits limits_;
    uint64_t currentTick_ = 0;
    double accumulator_ = 0.0;
    uint64_t droppedUpdates_ = 0;
    uint64_t droppedFrontiers_ = 0;
    std::priority_queue<ScheduledFluidUpdate, std::vector<ScheduledFluidUpdate>,
                        ScheduledFluidUpdateLater>
        queue_;
    std::unordered_map<FluidPos, uint64_t, FluidPosHash> pending_;
    std::unordered_map<ChunkPos, DeferredBucket> deferredByChunk_;
    size_t deferredCount_ = 0;
    mutable std::mutex mutex_;
};

FluidScheduler::FluidScheduler(FluidSchedulerLimits limits)
    : impl_(std::make_unique<Impl>(limits)) {}

FluidScheduler::~FluidScheduler() = default;
FluidScheduler::FluidScheduler(FluidScheduler&&) noexcept = default;
FluidScheduler& FluidScheduler::operator=(FluidScheduler&&) noexcept = default;

size_t FluidScheduler::activateBlockChange(FluidPos position) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->enqueueAround(position);
}

size_t FluidScheduler::tick(FluidWorldAccess& world) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->tick(world);
}

size_t FluidScheduler::advance(double elapsedSeconds, FluidWorldAccess& world) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->advance(elapsedSeconds, world);
}

size_t FluidScheduler::resumeDeferredIn(const FluidBounds& loadedBounds) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->resumeDeferredIn(loadedBounds);
}

size_t FluidScheduler::resumeDeferredIn(const FluidBounds& loadedBounds, size_t maximumFrontiers) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->resumeDeferredIn(loadedBounds, maximumFrontiers);
}

size_t FluidScheduler::deferredCountIn(const FluidBounds& loadedBounds) const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->deferredCountIn(loadedBounds);
}

std::vector<FluidBoundaryFrontier> FluidScheduler::deferredFrontiers() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->deferredFrontiers();
}

bool FluidScheduler::restoreDeferredFrontier(FluidBoundaryFrontier frontier) {
    std::lock_guard lock(impl_->mutex_);
    return impl_->defer(frontier.available, frontier.unavailable);
}

void FluidScheduler::clear() {
    std::lock_guard lock(impl_->mutex_);
    impl_->clear();
}

uint64_t FluidScheduler::currentTick() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->currentTick_;
}

size_t FluidScheduler::pendingCount() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->pending_.size();
}

size_t FluidScheduler::deferredCount() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->deferredCount();
}

uint64_t FluidScheduler::droppedUpdateCount() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->droppedUpdates_;
}

uint64_t FluidScheduler::droppedFrontierCount() const {
    std::lock_guard lock(impl_->mutex_);
    return impl_->droppedFrontiers_;
}
