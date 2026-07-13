#pragma once

#include <world/chunk.hpp>

// ---------------------------------------------------------------------------
// Hotbar — 9-slot inventory for block type selection
//
// Slots 1-9 mapped to number keys. Scroll wheel cycles forward/backward.
// Selected slot is highlighted in the HUD overlay.
// ---------------------------------------------------------------------------
class Hotbar {
public:
    static constexpr int SLOTS = 9;

    // Get block type for currently selected slot
    BlockType getSelectedBlockType() const;

    // Select slot by index (0-8). Clamps to valid range.
    void selectSlot(int index);

    // Cycle selection forward (wraps 8 → 0)
    void selectNext();

    // Cycle selection backward (wraps 0 → 8)
    void selectPrev();

    // Set block type in specific slot (0-8). Clamps to valid range.
    void setSlot(int index, BlockType type);

    // Get block type in specific slot (0-8). Returns AIR for out-of-range.
    BlockType getSlot(int index) const;

    // Get current selected index (0-8)
    int getSelectedIndex() const;

private:
    BlockType _slots[SLOTS] = {BlockType::STONE,     BlockType::DIRT,   BlockType::GRASS,
                               BlockType::LOG,       BlockType::PLANKS, BlockType::SAND,
                               BlockType::SANDSTONE, BlockType::GLASS,  BlockType::FLOWER_RED};
    int _selectedIndex = 0;
};
