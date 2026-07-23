#pragma once

#include "common/math.hpp"
#include "world/block_properties.hpp"

#include <cmath>
#include <cstdint>
#include <optional>

// ---------------------------------------------------------------------------
// Survival stats - food, saturation, exhaustion, air, and the regeneration,
// starvation, and drowning timers. Pure C++ so the whole loop is unit-
// testable without the engine. Health lives on the Player (already persisted);
// this module returns the per-tick health delta the engine applies.
//
// Values are borrowed from Minecraft as placeholders: food 0-20, exhaustion
// spends saturation then food, high food regenerates, empty food starves to a
// floor, and depleted air drowns.
// ---------------------------------------------------------------------------

struct SurvivalStats {
    int food = 20;
    float saturation = 5.f;   // spent before food; refilled by eating
    float exhaustion = 0.f;   // accumulates from activity, costs saturation/food
    int air = 300;            // ticks of breath remaining
    int foodTimer = 0;        // shared regen/starve timer, exactly like Minecraft
    float healResidual = 0.f; // fractional saturation healing awaiting a whole hp
    int drownCounter = 0;

    static constexpr int MAX_HEALTH = 20;
    static constexpr int MAX_FOOD = 20;
    static constexpr int MAX_AIR = 300;              // 15 s
    static constexpr int AIR_REFILL_PER_TICK = 8;    // full refill in ~2 s
    static constexpr int DROWN_DAMAGE_INTERVAL = 20; // 2 dmg/s once out of air
    static constexpr int DROWN_DAMAGE = 2;
    // Saturation heals fast (min(saturation, cap)/cap hp every interval, costing
    // that much exhaustion), so a full food bar with leftover saturation
    // regenerates to full within seconds, just as in Minecraft.
    static constexpr int FAST_REGEN_INTERVAL = 10;
    static constexpr float FAST_REGEN_SATURATION_CAP = 6.0f;
    static constexpr int SLOW_REGEN_INTERVAL = 80; // +1 hp / 4 s once food is high
    static constexpr int REGEN_FOOD_MIN = 18;      // food needed for slow regen
    static constexpr float SLOW_REGEN_EXHAUSTION = 6.0f;
    static constexpr int STARVE_INTERVAL = 80;    // -1 hp / 4 s at food 0
    static constexpr int STARVE_HEALTH_FLOOR = 1; // starvation never kills outright
    static constexpr int SPRINT_DISABLE_FOOD = 6;
    static constexpr float EXHAUSTION_THRESHOLD = 4.0f;

    // Exhaustion sources (per event/tick).
    static constexpr float EXHAUST_SPRINT_TICK = 0.006f;
    static constexpr float EXHAUST_SWIM_TICK = 0.004f;
    static constexpr float EXHAUST_JUMP = 0.05f;
    static constexpr float EXHAUST_MINE_BLOCK = 0.005f;
    static constexpr float EXHAUST_ATTACK = 0.1f;
};

enum class BedSpawnValidation : uint8_t {
    DEFERRED,
    VALID,
    INVALID,
};

// A bed respawn is valid only after all three cells are resident. The bed must
// still exist and both player cells must remain breathable. Missing cells stay
// deferred so a distant valid bed is not discarded before streaming reaches it.
constexpr BedSpawnValidation validateBedSpawnCells(std::optional<BlockType> bed,
                                                   std::optional<BlockType> feet,
                                                   std::optional<BlockType> head) noexcept {
    if (!bed || !feet || !head) return BedSpawnValidation::DEFERRED;
    if (*bed != BlockType::BED) return BedSpawnValidation::INVALID;
    const auto breathable = [](BlockType block) {
        return !isSolid(block) && block != BlockType::WATER && block != BlockType::LAVA;
    };
    return breathable(*feet) && breathable(*head) ? BedSpawnValidation::VALID
                                                  : BedSpawnValidation::INVALID;
}

inline bool bedSpawnAnchoredToBlock(Vec3 spawn, int64_t blockX, int32_t blockY,
                                    int64_t blockZ) noexcept {
    return static_cast<int64_t>(std::floor(spawn.x)) == blockX &&
           static_cast<int32_t>(std::floor(spawn.y)) - 1 == blockY &&
           static_cast<int64_t>(std::floor(spawn.z)) == blockZ;
}

struct SurvivalTickInputs {
    bool sprinting = false;
    bool swimming = false;
    bool jumped = false;
    bool eyesUnderwater = false;
    bool minedBlock = false;
    bool attacked = false;
};

// Advance food/saturation/exhaustion/air one 20 Hz tick and return the health
// delta (negative for drowning/starvation, positive for regeneration), already
// clamped against currentHealth and the starvation floor.
int tickSurvivalStats(SurvivalStats& stats, const SurvivalTickInputs& inputs, int currentHealth);

// Timed eating: hold right-click for EAT_TICKS on a food stack. Returns true
// the tick the item finishes (the engine then consumes one and applies the
// food value).
struct EatingState {
    bool active = false;
    int ticks = 0;
    int slot = -1;
    static constexpr int EAT_TICKS = 32; // ~1.6 s

    void reset() { *this = EatingState{}; }
};

bool tickEating(EatingState& eating, bool rightHeld, int selectedSlot, bool holdingFood, int food);
