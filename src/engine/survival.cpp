#include "engine/survival.hpp"

#include <algorithm>

namespace {

// Spend one unit of exhaustion: saturation first, then food.
void applyExhaustion(SurvivalStats& stats) {
    while (stats.exhaustion >= SurvivalStats::EXHAUSTION_THRESHOLD) {
        stats.exhaustion -= SurvivalStats::EXHAUSTION_THRESHOLD;
        if (stats.saturation > 0.f) {
            stats.saturation = std::max(0.f, stats.saturation - 1.f);
        } else if (stats.food > 0) {
            --stats.food;
        }
    }
}

} // namespace

int tickSurvivalStats(SurvivalStats& stats, const SurvivalTickInputs& inputs, int currentHealth) {
    // Activity raises exhaustion.
    if (inputs.sprinting) stats.exhaustion += SurvivalStats::EXHAUST_SPRINT_TICK;
    if (inputs.swimming) stats.exhaustion += SurvivalStats::EXHAUST_SWIM_TICK;
    if (inputs.jumped) stats.exhaustion += SurvivalStats::EXHAUST_JUMP;
    if (inputs.minedBlock) stats.exhaustion += SurvivalStats::EXHAUST_MINE_BLOCK;
    if (inputs.attacked) stats.exhaustion += SurvivalStats::EXHAUST_ATTACK;
    applyExhaustion(stats);

    int healthDelta = 0;

    // Air: drain underwater, refill above it. Drowning damage once empty.
    if (inputs.eyesUnderwater) {
        stats.air = std::max(0, stats.air - 1);
        if (stats.air == 0) {
            if (++stats.drownCounter >= SurvivalStats::DROWN_DAMAGE_INTERVAL) {
                stats.drownCounter = 0;
                healthDelta -= SurvivalStats::DROWN_DAMAGE;
            }
        }
    } else {
        stats.air =
            std::min(SurvivalStats::MAX_AIR, stats.air + SurvivalStats::AIR_REFILL_PER_TICK);
        stats.drownCounter = 0;
    }

    // Regeneration at high food, starvation at empty food.
    if (stats.food >= SurvivalStats::REGEN_FOOD_MIN && currentHealth + healthDelta < 20) {
        if (++stats.regenCounter >= SurvivalStats::REGEN_INTERVAL) {
            stats.regenCounter = 0;
            ++healthDelta;
            stats.exhaustion += SurvivalStats::REGEN_EXHAUSTION;
            applyExhaustion(stats);
        }
    } else {
        stats.regenCounter = 0;
    }

    if (stats.food == 0) {
        if (++stats.starveCounter >= SurvivalStats::STARVE_INTERVAL) {
            stats.starveCounter = 0;
            if (currentHealth + healthDelta > SurvivalStats::STARVE_HEALTH_FLOOR) {
                --healthDelta;
            }
        }
    } else {
        stats.starveCounter = 0;
    }

    // Clamp so the caller never overshoots full health or the starve floor.
    healthDelta = std::clamp(healthDelta, -currentHealth, 20 - currentHealth);
    return healthDelta;
}

bool tickEating(EatingState& eating, bool rightHeld, int selectedSlot, bool holdingFood, int food) {
    if (!rightHeld || !holdingFood || food >= SurvivalStats::MAX_FOOD) {
        eating.reset();
        return false;
    }
    if (!eating.active || eating.slot != selectedSlot) {
        eating.active = true;
        eating.slot = selectedSlot;
        eating.ticks = 0;
    }
    if (++eating.ticks >= EatingState::EAT_TICKS) {
        eating.reset();
        return true;
    }
    return false;
}
