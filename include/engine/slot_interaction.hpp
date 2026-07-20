#pragma once

#include "render/ui_menu.hpp"
#include "world/item.hpp"

#include <span>
#include <vector>

// ---------------------------------------------------------------------------
// Slot interaction - the single home for what mouse clicks do to stacks on
// the container screens: pick up, place, split, quick-move, and craft-output
// rules. Pure C++ over raw stack arrays so every rule is unit-testable; the
// engine owns the stacks and the cursor and calls in from its click handler.
// ---------------------------------------------------------------------------

enum class SlotClickKind : uint8_t { LEFT, RIGHT, SHIFT_LEFT };

// Mutable access to everything the open screen exposes. Null = absent.
struct SlotAccess {
    ItemStack* inventory = nullptr; // 36 slots: 0-8 hotbar, 9-35 main
    ItemStack* craftGrid = nullptr;
    int craftGridSize = 0;  // 4 (2x2) or 9 (3x3)
    int craftGridWidth = 0; // 2 or 3
    ItemStack* craftResult = nullptr;
    ItemStack* furnaceInput = nullptr;
    ItemStack* furnaceFuel = nullptr;
    ItemStack* furnaceOutput = nullptr;
    ItemStack* chest = nullptr; // 27 storage slots when a chest is open
    int chestSize = 0;
    const ItemType* palette = nullptr; // creative palette entries
    int paletteSize = 0;
};

struct SlotClickOutcome {
    bool changed = false; // engine re-evaluates the craft result on true
    bool crafted = false; // a craft or smelt output was taken
};

// Apply one click. Craft-output slots are take-only and consume the grid;
// the module recomputes the result through world/recipes.hpp so shift-craft
// can loop. Palette slots never mutate.
SlotClickOutcome applySlotClick(const SlotAccess& access, ItemStack& cursor, SlotRef slot,
                                SlotClickKind kind);

// Distribute the held cursor stack across the slots painted while a mouse
// button stayed down (Minecraft's drag/quick-craft). LEFT splits the held
// count evenly among the accepting slots; RIGHT drops exactly one into each.
// Output and palette slots are ignored and the cursor keeps the remainder.
SlotClickOutcome applySlotDrag(const SlotAccess& access, ItemStack& cursor,
                               std::span<const SlotRef> slots, SlotClickKind kind);

// Double-click gather: pull every matching loose stack across the open
// surfaces into the held cursor, up to a full stack, consolidating partial
// stacks first exactly like Minecraft.
SlotClickOutcome applyDoubleClick(const SlotAccess& access, ItemStack& cursor);

// Clicking outside the panel: LEFT drops the whole held stack, RIGHT one.
ItemStack takeOutsideDrop(ItemStack& cursor, SlotClickKind kind);

// Closing a container returns the craft grid and cursor to the inventory;
// whatever cannot fit comes back for the engine to drop at the feet.
std::vector<ItemStack> collectOnClose(const SlotAccess& access, ItemStack& cursor);
