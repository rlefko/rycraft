#pragma once

#include "world/chunk_pos.hpp"
#include "world/item.hpp"

#include <unordered_map>

// ---------------------------------------------------------------------------
// Furnace - the single source of truth for smelting state and its 20 Hz step.
//
// Furnaces are the only stateful blocks: the engine owns a FurnaceMap keyed
// by block position, ticks every entry each gameTick, and swaps the world
// block between FURNACE and FURNACE_LIT when lit() changes. The map persists
// through SaveManager's block-entities sidecar, never the cube format.
// ---------------------------------------------------------------------------

struct FurnaceState {
    ItemStack input;
    ItemStack fuel;
    ItemStack output;
    uint16_t burnTicksRemaining = 0;
    uint16_t burnTicksTotal = 0; // denominator for the flame gauge
    uint16_t cookTicks = 0;      // 0..FURNACE_COOK_TICKS

    bool lit() const { return burnTicksRemaining > 0; }
    float cookFraction() const;
    float fuelFraction() const;
};

using FurnaceMap = std::unordered_map<BlockPos, FurnaceState>;

// One 20 Hz step. Pure item logic: ignites by consuming one fuel unit when
// the input can smelt into the output, advances cooking while lit, and moves
// one result per FURNACE_COOK_TICKS. Unlit progress decays instead of
// holding forever. Returns true when lit() changed so the caller updates the
// world block.
bool furnaceTick(FurnaceState& furnace);
