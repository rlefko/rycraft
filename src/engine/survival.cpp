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

    // Regeneration and starvation share one food timer, evaluated as the same
    // mutually exclusive chain Minecraft uses: a full food bar with leftover
    // saturation heals fast, a merely high food bar heals slowly, and an empty
    // food bar starves toward the floor.
    const bool hurt = currentHealth + healthDelta < SurvivalStats::MAX_HEALTH;
    if (stats.food >= SurvivalStats::MAX_FOOD && stats.saturation > 0.f && hurt) {
        if (++stats.foodTimer >= SurvivalStats::FAST_REGEN_INTERVAL) {
            stats.foodTimer = 0;
            const float healAmount =
                std::min(stats.saturation, SurvivalStats::FAST_REGEN_SATURATION_CAP);
            // Heal healAmount/cap hp, carrying the fraction so integer health
            // still tracks Minecraft's sub-heart regeneration over time.
            stats.healResidual += healAmount / SurvivalStats::FAST_REGEN_SATURATION_CAP;
            const int wholeHp = static_cast<int>(stats.healResidual);
            stats.healResidual -= static_cast<float>(wholeHp);
            healthDelta += wholeHp;
            stats.exhaustion += healAmount;
            applyExhaustion(stats);
        }
    } else if (stats.food >= SurvivalStats::REGEN_FOOD_MIN && hurt) {
        if (++stats.foodTimer >= SurvivalStats::SLOW_REGEN_INTERVAL) {
            stats.foodTimer = 0;
            ++healthDelta;
            stats.exhaustion += SurvivalStats::SLOW_REGEN_EXHAUSTION;
            applyExhaustion(stats);
        }
    } else if (stats.food == 0) {
        if (++stats.foodTimer >= SurvivalStats::STARVE_INTERVAL) {
            stats.foodTimer = 0;
            if (currentHealth + healthDelta > SurvivalStats::STARVE_HEALTH_FLOOR) {
                --healthDelta;
            }
        }
    } else {
        stats.foodTimer = 0;
    }

    // Clamp so the caller never overshoots full health or the starve floor.
    healthDelta =
        std::clamp(healthDelta, -currentHealth, SurvivalStats::MAX_HEALTH - currentHealth);
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
