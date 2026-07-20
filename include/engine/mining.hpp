#pragma once

#include "world/item.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Mining - held-left-click block breaking over time. Pure C++ so the timing
// and target-tracking rules are unit-testable. The break-time formula and
// tool multipliers live in world/item.hpp (blockBreakTicks); this module only
// accumulates progress against a stable ray target.
// ---------------------------------------------------------------------------

struct MiningState {
    bool active = false;
    int64_t x = 0;
    int32_t y = 0;
    int64_t z = 0;
    BlockType block = BlockType::AIR;
    ItemType tool = ItemType::NONE; // held item the break time was computed for
    int ticksElapsed = 0;
    int ticksNeeded = 0;
    float progress = 0.f; // 0..1 for the HUD

    void reset() { *this = MiningState{}; }
};

// Advance mining one 20 Hz tick. Progress accumulates only while the button
// is held and the ray keeps hitting the same block cell of the same type;
// changing target, releasing, or an unbreakable block resets it. Returns
// true on the tick the block finishes breaking (and resets the state).
bool tickMining(MiningState& state, bool leftHeld, bool hasBlockTarget, int64_t targetX,
                int32_t targetY, int64_t targetZ, BlockType targetBlock, ItemType heldItem);
