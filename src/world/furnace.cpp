#include "world/furnace.hpp"

#include "world/recipes.hpp"

namespace {

// Smelting proceeds only when the input has a result the output slot can
// still absorb.
bool canSmelt(const FurnaceState& furnace) {
    if (furnace.input.empty()) return false;
    const auto result = smeltingResult(furnace.input.type);
    if (!result.has_value()) return false;
    if (furnace.output.empty()) return true;
    return furnace.output.type == *result &&
           furnace.output.count < maxStackSize(furnace.output.type);
}

} // namespace

float FurnaceState::cookFraction() const {
    return static_cast<float>(cookTicks) / static_cast<float>(FURNACE_COOK_TICKS);
}

float FurnaceState::fuelFraction() const {
    if (burnTicksTotal == 0) return 0.0f;
    return static_cast<float>(burnTicksRemaining) / static_cast<float>(burnTicksTotal);
}

bool furnaceTick(FurnaceState& furnace) {
    const bool wasLit = furnace.lit();
    const bool smeltable = canSmelt(furnace);

    if (!furnace.lit() && smeltable && !furnace.fuel.empty() && isFurnaceFuel(furnace.fuel.type)) {
        const int burn = fuelBurnTicks(furnace.fuel.type);
        furnace.burnTicksRemaining = static_cast<uint16_t>(burn);
        furnace.burnTicksTotal = static_cast<uint16_t>(burn);
        if (--furnace.fuel.count == 0) furnace.fuel.clear();
    }

    if (furnace.lit()) {
        --furnace.burnTicksRemaining;
        if (smeltable) {
            if (++furnace.cookTicks >= FURNACE_COOK_TICKS) {
                furnace.cookTicks = 0;
                const ItemType result = *smeltingResult(furnace.input.type);
                if (furnace.output.empty()) {
                    furnace.output = ItemStack{result, 1, 0};
                } else {
                    ++furnace.output.count;
                }
                if (--furnace.input.count == 0) furnace.input.clear();
            }
        } else {
            furnace.cookTicks = 0;
        }
    } else if (furnace.cookTicks > 0) {
        // Fire ran out mid-item: progress cools off instead of freezing.
        furnace.cookTicks =
            furnace.cookTicks >= 2 ? static_cast<uint16_t>(furnace.cookTicks - 2) : uint16_t{0};
    }

    return furnace.lit() != wasLit;
}
