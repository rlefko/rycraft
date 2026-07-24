#pragma once

#include "world/chunk_pos.hpp"
#include "world/item.hpp"

#include <array>
#include <optional>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Chest - a placed 27-slot storage block. Like the furnace, the engine owns a
// ChestMap keyed by block position and persists it through SaveManager's
// block-entities sidecar, never the cube format. Unlike the furnace it holds
// no timers, so there is no per-tick step: it is pure storage.
// ---------------------------------------------------------------------------

struct ChestState {
    static constexpr int SLOT_COUNT = 27; // three rows of nine, exactly as Minecraft

    std::array<ItemStack, SLOT_COUNT> slots{};

    bool empty() const {
        for (const ItemStack& slot : slots) {
            if (!slot.empty()) return false;
        }
        return true;
    }
};

using ChestMap = std::unordered_map<BlockPos, ChestState>;

// Missing means the cube is not resident and cannot invalidate persistence.
// A resident non-chest cell proves the sidecar is stale.
constexpr bool chestSidecarMatchesLoadedBlock(std::optional<BlockType> loadedBlock) noexcept {
    return !loadedBlock || *loadedBlock == BlockType::CHEST;
}
