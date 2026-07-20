#include "engine/mining.hpp"

bool tickMining(MiningState& state, bool leftHeld, bool hasBlockTarget, int64_t targetX,
                int32_t targetY, int64_t targetZ, BlockType targetBlock, ItemType heldItem) {
    if (!leftHeld || !hasBlockTarget) {
        state.reset();
        return false;
    }

    // Switching tools mid-mine restarts progress so the break time always
    // matches the item actually held.
    const bool sameTarget = state.active && state.x == targetX && state.y == targetY &&
                            state.z == targetZ && state.block == targetBlock &&
                            state.tool == heldItem;
    if (!sameTarget) {
        const int needed = blockBreakTicks(targetBlock, heldItem);
        state.active = true;
        state.x = targetX;
        state.y = targetY;
        state.z = targetZ;
        state.block = targetBlock;
        state.tool = heldItem;
        state.ticksElapsed = 0;
        state.ticksNeeded = needed;
        state.progress = 0.f;
    }

    // Unbreakable blocks (bedrock) never accumulate progress.
    if (state.ticksNeeded >= UNBREAKABLE_BREAK_TICKS) {
        state.progress = 0.f;
        return false;
    }

    // Hardness-zero blocks break the first tick the ray settles on them.
    if (state.ticksNeeded <= 0) {
        state.progress = 1.f;
        state.reset();
        return true;
    }

    ++state.ticksElapsed;
    state.progress = static_cast<float>(state.ticksElapsed) / static_cast<float>(state.ticksNeeded);
    if (state.ticksElapsed >= state.ticksNeeded) {
        state.reset();
        return true;
    }
    return false;
}
