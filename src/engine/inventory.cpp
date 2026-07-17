#include "engine/inventory.hpp"

#include <algorithm>

BlockType Inventory::getSelectedBlockType() const {
    const ItemStack& stack = slots_[static_cast<size_t>(selectedIndex_)];
    return stack.empty() ? BlockType::AIR : blockFromItem(stack.type);
}

ItemStack Inventory::getSelectedStack() const {
    return slots_[static_cast<size_t>(selectedIndex_)];
}

void Inventory::selectSlot(int index) {
    selectedIndex_ = std::clamp(index, 0, HOTBAR_SLOTS - 1);
}

void Inventory::selectNext() {
    selectedIndex_ = (selectedIndex_ + 1) % HOTBAR_SLOTS;
}

void Inventory::selectPrev() {
    selectedIndex_ = (selectedIndex_ + HOTBAR_SLOTS - 1) % HOTBAR_SLOTS;
}

int Inventory::getSelectedIndex() const {
    return selectedIndex_;
}

ItemStack Inventory::getSlot(int index) const {
    if (index < 0 || index >= SLOTS) return ItemStack{};
    return slots_[static_cast<size_t>(index)];
}

void Inventory::setSlot(int index, const ItemStack& stack) {
    if (index < 0 || index >= SLOTS) return;
    slots_[static_cast<size_t>(index)] = stack;
}

int Inventory::add(const ItemStack& stack) {
    if (stack.empty()) return 0;
    int remaining = stack.count;

    // Merge into existing stacks of the same item first, hotbar order.
    const int limit = maxStackSize(stack.type);
    for (ItemStack& slot : slots_) {
        if (remaining == 0) break;
        if (slot.empty() || slot.type != stack.type) continue;
        const int room = limit - slot.count;
        if (room <= 0) continue;
        const int moved = std::min(room, remaining);
        slot.count = static_cast<uint8_t>(slot.count + moved);
        remaining -= moved;
    }

    // Then fill empty slots.
    for (ItemStack& slot : slots_) {
        if (remaining == 0) break;
        if (!slot.empty()) continue;
        const int moved = std::min(limit, remaining);
        slot = ItemStack{stack.type, static_cast<uint8_t>(moved), stack.durability};
        remaining -= moved;
    }

    return stack.count - remaining;
}

void Inventory::consumeSelected(int amount) {
    ItemStack& stack = slots_[static_cast<size_t>(selectedIndex_)];
    if (stack.empty()) return;
    if (stack.count <= amount) {
        stack.clear();
    } else {
        stack.count = static_cast<uint8_t>(stack.count - amount);
    }
}

bool Inventory::damageSelectedTool() {
    ItemStack& stack = slots_[static_cast<size_t>(selectedIndex_)];
    if (stack.empty() || !isTool(stack.type)) return false;
    if (stack.durability > 1) {
        --stack.durability;
        return false;
    }
    stack.clear();
    return true;
}

void Inventory::clear() {
    slots_.fill(ItemStack{});
    selectedIndex_ = 0;
}
