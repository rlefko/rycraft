#pragma once

#include <world/item.hpp>

#include <array>

// ---------------------------------------------------------------------------
// Inventory - the player's 36 item stacks and hotbar selection.
//
// Slots 0-8 are the hotbar (number keys and scroll selection); slots 9-35
// are the main grid the inventory screen shows. Pickups fill the hotbar
// first. This is the single home for slot arithmetic; screens address slots
// by these indices.
// ---------------------------------------------------------------------------
class Inventory {
public:
    static constexpr int HOTBAR_SLOTS = 9;
    static constexpr int SLOTS = 36;

    // Block form of the selected stack for placement; AIR when the slot is
    // empty or holds a non-block item.
    BlockType getSelectedBlockType() const;
    ItemStack getSelectedStack() const;

    // Selection mirrors the original hotbar: clamp on direct select, wrap on
    // scroll cycling.
    void selectSlot(int index);
    void selectNext();
    void selectPrev();
    int getSelectedIndex() const;

    // Empty-stack result for out-of-range reads; out-of-range writes drop.
    ItemStack getSlot(int index) const;
    void setSlot(int index, const ItemStack& stack);

    // Absorb as much of the stack as fits (merge first, then empty slots,
    // hotbar before main). Returns how many items were absorbed.
    int add(const ItemStack& stack);

    // Remove items from the selected stack (placing, dropping, eating).
    void consumeSelected(int amount = 1);

    // One point of tool wear; a tool at zero durability breaks and leaves the
    // slot empty. Non-tools never wear. Returns true when the tool broke.
    bool damageSelectedTool();

    std::array<ItemStack, SLOTS>& slots() { return slots_; }
    const std::array<ItemStack, SLOTS>& slots() const { return slots_; }
    void clear();

private:
    std::array<ItemStack, SLOTS> slots_{};
    int selectedIndex_ = 0;
};
