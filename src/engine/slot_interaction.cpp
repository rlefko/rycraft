#include "engine/slot_interaction.hpp"

#include "world/recipes.hpp"

#include <algorithm>
#include <span>

namespace {

// How many of `stack` would fit into the slot range without mutating it.
int roomFor(const ItemStack* slots, int count, const ItemStack& stack) {
    if (!slots || stack.empty()) return 0;
    const int limit = maxStackSize(stack.type);
    int room = 0;
    for (int index = 0; index < count; ++index) {
        if (slots[index].empty()) {
            room += limit;
        } else if (slots[index].type == stack.type) {
            room += std::max(0, limit - slots[index].count);
        }
    }
    return std::min(room, static_cast<int>(stack.count));
}

// Merge-then-fill absorption into a raw slot range; returns items absorbed.
int addToSlots(ItemStack* slots, int count, const ItemStack& stack) {
    if (!slots || stack.empty()) return 0;
    int remaining = stack.count;
    const int limit = maxStackSize(stack.type);

    for (int index = 0; index < count && remaining > 0; ++index) {
        ItemStack& slot = slots[index];
        if (slot.empty() || slot.type != stack.type) continue;
        const int moved = std::min(limit - slot.count, remaining);
        if (moved <= 0) continue;
        slot.count = static_cast<uint8_t>(slot.count + moved);
        remaining -= moved;
    }
    for (int index = 0; index < count && remaining > 0; ++index) {
        ItemStack& slot = slots[index];
        if (!slot.empty()) continue;
        const int moved = std::min(limit, remaining);
        slot = ItemStack{stack.type, static_cast<uint8_t>(moved), stack.durability};
        remaining -= moved;
    }
    return stack.count - remaining;
}

ItemStack* resolveSlot(const SlotAccess& access, SlotRef slot) {
    switch (slot.domain) {
        case SlotDomain::INVENTORY:
            if (access.inventory && slot.index >= 0 && slot.index < 36) {
                return &access.inventory[slot.index];
            }
            return nullptr;
        case SlotDomain::CRAFT_IN:
            if (access.craftGrid && slot.index >= 0 && slot.index < access.craftGridSize) {
                return &access.craftGrid[slot.index];
            }
            return nullptr;
        case SlotDomain::FURNACE_INPUT:
            return access.furnaceInput;
        case SlotDomain::FURNACE_FUEL:
            return access.furnaceFuel;
        case SlotDomain::FURNACE_OUTPUT:
            return access.furnaceOutput;
        default:
            return nullptr;
    }
}

void refreshCraftResult(const SlotAccess& access) {
    if (!access.craftGrid || !access.craftResult) return;
    const auto result = matchCraftingRecipe(
        std::span<const ItemStack>(access.craftGrid, static_cast<size_t>(access.craftGridSize)),
        access.craftGridWidth);
    *access.craftResult = result.value_or(ItemStack{});
}

// Take-only output slots (craft result, furnace output): LEFT takes one
// batch, SHIFT_LEFT drains into the inventory.
SlotClickOutcome takeFromOutput(const SlotAccess& access, ItemStack& cursor, bool isCraft,
                                SlotClickKind kind) {
    SlotClickOutcome outcome;
    ItemStack* output = isCraft ? access.craftResult : access.furnaceOutput;
    if (!output || output->empty()) return outcome;

    if (kind == SlotClickKind::SHIFT_LEFT) {
        // Craft or drain the output into the inventory repeatedly. Hard-capped
        // so a self-sustaining recipe cannot spin forever.
        for (int iteration = 0; iteration < 64; ++iteration) {
            if (output->empty()) break;
            const ItemStack batch = *output;
            if (isCraft) {
                // A craft batch is atomic: the virtual result exists only if a
                // craft happens, so take it only when it fully fits. Never
                // deposit a partial batch (that would create items).
                if (roomFor(access.inventory, 36, batch) < batch.count) return outcome;
                addToSlots(access.inventory, 36, batch);
                consumeOneCraft(std::span<ItemStack>(access.craftGrid,
                                                     static_cast<size_t>(access.craftGridSize)));
                refreshCraftResult(access);
                outcome.changed = true;
                outcome.crafted = true;
            } else {
                // The furnace output is a real stack, so a partial take is
                // fine: shrink it by whatever the inventory absorbed.
                const int absorbed = addToSlots(access.inventory, 36, batch);
                if (absorbed > 0) {
                    outcome.changed = true;
                    outcome.crafted = true;
                }
                if (absorbed < batch.count) {
                    output->count = static_cast<uint8_t>(batch.count - absorbed);
                    if (output->count == 0) output->clear();
                    return outcome;
                }
                output->clear();
            }
        }
        return outcome;
    }

    if (kind != SlotClickKind::LEFT) return outcome;
    const bool cursorAccepts =
        cursor.empty() ||
        (cursor.type == output->type && cursor.count + output->count <= maxStackSize(cursor.type));
    if (!cursorAccepts) return outcome;
    if (cursor.empty()) {
        cursor = *output;
    } else {
        cursor.count = static_cast<uint8_t>(cursor.count + output->count);
    }
    outcome.changed = true;
    outcome.crafted = true;
    if (isCraft) {
        consumeOneCraft(
            std::span<ItemStack>(access.craftGrid, static_cast<size_t>(access.craftGridSize)));
        refreshCraftResult(access);
    } else {
        output->clear();
    }
    return outcome;
}

// Quick move between regions: containers empty into the inventory; the
// inventory feeds an open furnace by item kind or swaps hotbar and main.
SlotClickOutcome quickMove(const SlotAccess& access, SlotRef slot) {
    SlotClickOutcome outcome;
    ItemStack* source = resolveSlot(access, slot);
    if (!source || source->empty()) return outcome;

    const bool furnaceOpen = access.furnaceInput != nullptr;
    if (slot.domain == SlotDomain::INVENTORY && furnaceOpen) {
        ItemStack* target = isSmeltable(source->type)     ? access.furnaceInput
                            : isFurnaceFuel(source->type) ? access.furnaceFuel
                                                          : nullptr;
        if (!target) return outcome;
        const int absorbed = addToSlots(target, 1, *source);
        if (absorbed > 0) {
            source->count = static_cast<uint8_t>(source->count - absorbed);
            if (source->count == 0) source->clear();
            outcome.changed = true;
        }
        return outcome;
    }

    ItemStack* target = access.inventory;
    int targetOffset = 0;
    int targetCount = 36;
    if (slot.domain == SlotDomain::INVENTORY) {
        // Hotbar to main and back.
        targetOffset = slot.index < 9 ? 9 : 0;
        targetCount = slot.index < 9 ? 27 : 9;
    }
    const int absorbed = addToSlots(target + targetOffset, targetCount, *source);
    if (absorbed > 0) {
        source->count = static_cast<uint8_t>(source->count - absorbed);
        if (source->count == 0) source->clear();
        outcome.changed = true;
    }
    return outcome;
}

} // namespace

SlotClickOutcome applySlotClick(const SlotAccess& access, ItemStack& cursor, SlotRef slot,
                                SlotClickKind kind) {
    SlotClickOutcome outcome;

    if (slot.domain == SlotDomain::CRAFT_OUT || slot.domain == SlotDomain::FURNACE_OUTPUT) {
        return takeFromOutput(access, cursor, slot.domain == SlotDomain::CRAFT_OUT, kind);
    }

    if (slot.domain == SlotDomain::CREATIVE_PALETTE) {
        if (!access.palette || slot.index < 0 || slot.index >= access.paletteSize) return outcome;
        const ItemType type = access.palette[slot.index];
        if (kind == SlotClickKind::SHIFT_LEFT) {
            const ItemStack full = makeItemStack(type, static_cast<uint8_t>(maxStackSize(type)));
            outcome.changed = addToSlots(access.inventory, 36, full) > 0;
            return outcome;
        }
        if (kind == SlotClickKind::LEFT) {
            // Holding anything, the palette doubles as a trash can.
            if (!cursor.empty()) {
                cursor.clear();
            } else {
                cursor = makeItemStack(type, static_cast<uint8_t>(maxStackSize(type)));
            }
            outcome.changed = true;
            return outcome;
        }
        // RIGHT: one more of the hovered type onto a matching cursor.
        if (cursor.empty()) {
            cursor = makeItemStack(type, 1);
            outcome.changed = true;
        } else if (cursor.type == type && cursor.count < maxStackSize(type)) {
            ++cursor.count;
            outcome.changed = true;
        }
        return outcome;
    }

    if (kind == SlotClickKind::SHIFT_LEFT) {
        return quickMove(access, slot);
    }

    ItemStack* target = resolveSlot(access, slot);
    if (!target) return outcome;

    if (kind == SlotClickKind::LEFT) {
        if (cursor.empty()) {
            if (target->empty()) return outcome;
            cursor = *target;
            target->clear();
        } else if (!target->empty() && target->type == cursor.type) {
            const int moved =
                std::min<int>(maxStackSize(target->type) - target->count, cursor.count);
            if (moved == 0) return outcome;
            target->count = static_cast<uint8_t>(target->count + moved);
            cursor.count = static_cast<uint8_t>(cursor.count - moved);
            if (cursor.count == 0) cursor.clear();
        } else {
            std::swap(cursor, *target);
        }
        outcome.changed = true;
        return outcome;
    }

    // RIGHT: take half of a stack or place exactly one item.
    if (cursor.empty()) {
        if (target->empty()) return outcome;
        const auto taken = static_cast<uint8_t>((target->count + 1) / 2);
        cursor = ItemStack{target->type, taken, target->durability};
        target->count = static_cast<uint8_t>(target->count - taken);
        if (target->count == 0) target->clear();
        outcome.changed = true;
        return outcome;
    }
    if (target->empty()) {
        *target = ItemStack{cursor.type, 1, cursor.durability};
    } else if (target->type == cursor.type && target->count < maxStackSize(target->type)) {
        ++target->count;
    } else {
        return outcome;
    }
    --cursor.count;
    if (cursor.count == 0) cursor.clear();
    outcome.changed = true;
    return outcome;
}

ItemStack takeOutsideDrop(ItemStack& cursor, SlotClickKind kind) {
    if (cursor.empty()) return ItemStack{};
    if (kind == SlotClickKind::RIGHT) {
        ItemStack one{cursor.type, 1, cursor.durability};
        --cursor.count;
        if (cursor.count == 0) cursor.clear();
        return one;
    }
    ItemStack all = cursor;
    cursor.clear();
    return all;
}

std::vector<ItemStack> collectOnClose(const SlotAccess& access, ItemStack& cursor) {
    std::vector<ItemStack> overflow;
    auto returnStack = [&](ItemStack& stack) {
        if (stack.empty()) return;
        const int absorbed = addToSlots(access.inventory, 36, stack);
        stack.count = static_cast<uint8_t>(stack.count - absorbed);
        if (stack.count > 0) {
            overflow.push_back(stack);
        }
        stack.clear();
    };
    for (int index = 0; index < access.craftGridSize; ++index) {
        returnStack(access.craftGrid[index]);
    }
    returnStack(cursor);
    if (access.craftResult) access.craftResult->clear();
    return overflow;
}
