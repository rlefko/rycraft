#pragma once

#include "world/block_properties.hpp"
#include "world/chunk_pos.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using FluidPos = BlockPos;

struct FluidPosHash {
    size_t operator()(const FluidPos& position) const noexcept;
};

// One byte is persisted only for water cells whose runtime state differs
// from generated source water. Bits 0 through 2 store level 0 through 7 and
// bit 3 marks downward flow. The remaining bits are reserved.
class FluidState {
public:
    static constexpr uint8_t LEVEL_MASK = 0x07;
    static constexpr uint8_t FALLING_MASK = 0x08;
    static constexpr uint8_t PACKED_MASK = LEVEL_MASK | FALLING_MASK;

    constexpr FluidState() = default;

    static constexpr FluidState source() { return FluidState(0); }

    static constexpr FluidState flowing(uint8_t level) { return FluidState(boundedLevel(level)); }

    static constexpr FluidState falling(uint8_t level = 1) {
        return FluidState(static_cast<uint8_t>(boundedLevel(level) | FALLING_MASK));
    }

    static constexpr FluidState fromPacked(uint8_t packed) {
        return FluidState(static_cast<uint8_t>(packed & PACKED_MASK));
    }

    static constexpr bool isValidPacked(uint8_t packed) {
        return (packed & static_cast<uint8_t>(~PACKED_MASK)) == 0 &&
               (packed & PACKED_MASK) != FALLING_MASK;
    }

    constexpr uint8_t packed() const { return packed_; }
    constexpr uint8_t level() const { return packed_ & LEVEL_MASK; }
    constexpr bool isFalling() const { return (packed_ & FALLING_MASK) != 0; }
    constexpr bool isSource() const { return level() == 0 && !isFalling(); }

    constexpr bool operator==(const FluidState&) const = default;

private:
    static constexpr uint8_t boundedLevel(uint8_t level) {
        if (level < 1) {
            return 1;
        }
        if (level > 7) {
            return 7;
        }
        return level;
    }

    explicit constexpr FluidState(uint8_t packed) : packed_(packed) {}

    uint8_t packed_ = 0;
};

static_assert(sizeof(FluidState) == 1);

constexpr float fluidSurfaceHeight(FluidState state) {
    if (state.isFalling()) return 1.0f;
    if (state.isSource()) return 0.875f;
    return static_cast<float>(8 - state.level()) * 0.125f;
}

// An unavailable cell means its cube is not resident. Implementations must
// not load or generate terrain while answering this query.
struct FluidCell {
    bool loaded = false;
    BlockType block = BlockType::AIR;
    FluidState state = FluidState::source();

    constexpr bool isWater() const { return loaded && block == BlockType::WATER; }
};

// Runtime writes bypass the player-edit activation hook. The scheduler
// handles follow-up activation itself, which prevents recursive scheduling.
class FluidWorldAccess {
public:
    virtual ~FluidWorldAccess() = default;

    virtual FluidCell readFluidCell(FluidPos position) const = 0;
    virtual void writeWater(FluidPos position, FluidState state) = 0;
    virtual void removeWater(FluidPos position) = 0;
};

enum class FluidDirection : uint8_t {
    CENTER = 0,
    DOWN,
    UP,
    WEST,
    EAST,
    NORTH,
    SOUTH,
};

enum class FluidMutationType : uint8_t {
    SET_WATER = 0,
    REMOVE_WATER,
};

struct FluidMutation {
    FluidDirection direction = FluidDirection::CENTER;
    FluidMutationType type = FluidMutationType::SET_WATER;
    FluidState state = FluidState::source();
};

struct FluidNeighborhood {
    FluidCell center;
    FluidCell down;
    FluidCell up;
    FluidCell west;
    FluidCell east;
    FluidCell north;
    FluidCell south;
};

// A cell update can change the center plus four horizontal neighbors, or the
// center plus the cell below. Missing neighbors are returned separately so
// the scheduler can retain only already-activated boundary work.
struct FluidRuleResult {
    std::array<FluidMutation, 6> mutations{};
    uint8_t mutationCount = 0;
    std::array<FluidDirection, 6> deferred{};
    uint8_t deferredCount = 0;
};

constexpr bool isWaterReplaceable(BlockType block) {
    return block == BlockType::AIR || isFlora(block);
}

// Pure Java-style source and level rules. Mutations are ordered with the
// center first, then downward flow, then horizontal flow.
FluidRuleResult evaluateWaterRules(const FluidNeighborhood& cells) noexcept;

struct FluidBounds {
    int64_t minX = 0;
    int32_t minY = 0;
    int64_t minZ = 0;
    int64_t maxX = 0;
    int32_t maxY = 0;
    int64_t maxZ = 0;

    constexpr bool contains(FluidPos position) const {
        return position.x >= minX && position.x <= maxX && position.y >= minY &&
               position.y <= maxY && position.z >= minZ && position.z <= maxZ;
    }
};

struct FluidBoundaryFrontier {
    FluidPos available;
    FluidPos unavailable;

    constexpr bool operator==(const FluidBoundaryFrontier&) const = default;
};

inline constexpr uint32_t FLUID_TICKS_PER_SECOND = 20;
inline constexpr uint32_t WATER_UPDATE_DELAY_TICKS = 5;
inline constexpr size_t MAX_FLUID_UPDATES_PER_TICK = 1024;
inline constexpr size_t MAX_PENDING_FLUID_UPDATES = 65'536;
inline constexpr size_t MAX_DEFERRED_FLUID_FRONTIERS = 65'536;
inline constexpr uint32_t MAX_FLUID_CATCH_UP_TICKS = 8;

struct FluidSchedulerLimits {
    size_t updatesPerTick = MAX_FLUID_UPDATES_PER_TICK;
    size_t pendingUpdates = MAX_PENDING_FLUID_UPDATES;
    size_t deferredFrontiers = MAX_DEFERRED_FLUID_FRONTIERS;
    uint32_t catchUpTicks = MAX_FLUID_CATCH_UP_TICKS;
};

// Main-thread fixed-step scheduler. Construction, terrain generation, and
// ordinary cube loading never enqueue water. Only activateBlockChange and a
// previously deferred boundary resume introduce work.
class FluidScheduler {
public:
    explicit FluidScheduler(FluidSchedulerLimits limits = {});
    ~FluidScheduler();

    FluidScheduler(const FluidScheduler&) = delete;
    FluidScheduler& operator=(const FluidScheduler&) = delete;
    FluidScheduler(FluidScheduler&&) noexcept;
    FluidScheduler& operator=(FluidScheduler&&) noexcept;

    // Called after a player or gameplay edit. The changed cell and its six
    // face neighbors become due after the fixed five-tick water delay.
    size_t activateBlockChange(FluidPos position);

    // Advances exactly one 20 Hz fluid tick and processes at most the update
    // budget. Useful when the engine already owns a fixed-step game clock.
    size_t tick(FluidWorldAccess& world);

    // Convenience wall-time accumulator. Long frame gaps are clamped to the
    // configured catch-up count so water cannot monopolize a recovery frame.
    size_t advance(double elapsedSeconds, FluidWorldAccess& world);

    // Resume only frontiers created by active water before this region was
    // available. Calling this for a normal generated load with no matching
    // frontier schedules nothing.
    size_t resumeDeferredIn(const FluidBounds& loadedBounds);

    // Bounded variant for streaming paths. Frontiers are visited in stable
    // unavailable-position order, and matching work beyond the limit remains
    // deferred for a later load or tick budget.
    size_t resumeDeferredIn(const FluidBounds& loadedBounds, size_t maximumFrontiers);

    // Indexed count used by streaming to retain a loaded-cube resume request
    // when a per-frame budget or pending-update cap leaves work behind.
    size_t deferredCountIn(const FluidBounds& loadedBounds) const;

    // Persistence can snapshot and restore activated frontiers without
    // scheduling ordinary generated water. The returned order is stable.
    std::vector<FluidBoundaryFrontier> deferredFrontiers() const;
    bool restoreDeferredFrontier(FluidBoundaryFrontier frontier);

    void clear();

    uint64_t currentTick() const;
    size_t pendingCount() const;
    size_t deferredCount() const;
    uint64_t droppedUpdateCount() const;
    uint64_t droppedFrontierCount() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
