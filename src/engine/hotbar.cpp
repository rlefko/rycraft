#include "engine/hotbar.hpp"

#include <algorithm>

// ---------------------------------------------------------------------------
// getSelectedBlockType — Return block type in currently selected slot
// ---------------------------------------------------------------------------
BlockType Hotbar::getSelectedBlockType() const {
    return _slots[_selectedIndex];
}

// ---------------------------------------------------------------------------
// selectSlot — Clamp index to valid range [0, SLOTS-1]
// ---------------------------------------------------------------------------
void Hotbar::selectSlot(int index) {
    if (index < 0) index = 0;
    if (index >= SLOTS) index = SLOTS - 1;
    _selectedIndex = index;
}

// ---------------------------------------------------------------------------
// selectNext — Cycle forward with wrap-around (8 → 0)
// ---------------------------------------------------------------------------
void Hotbar::selectNext() {
    _selectedIndex = (_selectedIndex + 1) % SLOTS;
}

// ---------------------------------------------------------------------------
// selectPrev — Cycle backward with wrap-around (0 → 8)
// ---------------------------------------------------------------------------
void Hotbar::selectPrev() {
    _selectedIndex = (_selectedIndex - 1 + SLOTS) % SLOTS;
}

// ---------------------------------------------------------------------------
// setSlot — Set block type in specific slot, clamped to valid range
// ---------------------------------------------------------------------------
void Hotbar::setSlot(int index, BlockType type) {
    if (index < 0 || index >= SLOTS) return;
    _slots[index] = type;
}

// ---------------------------------------------------------------------------
// getSlot — Get block type in specific slot, returns AIR for out-of-range
// ---------------------------------------------------------------------------
BlockType Hotbar::getSlot(int index) const {
    if (index < 0 || index >= SLOTS) return BlockType::AIR;
    return _slots[index];
}

// ---------------------------------------------------------------------------
// getSelectedIndex — Return current selection (0-8)
// ---------------------------------------------------------------------------
int Hotbar::getSelectedIndex() const {
    return _selectedIndex;
}
