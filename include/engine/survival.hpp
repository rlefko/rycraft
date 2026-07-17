#pragma once

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
    float saturation = 5.f; // spent before food; refilled by eating
    float exhaustion = 0.f; // accumulates from activity, costs saturation/food
    int air = 300;          // ticks of breath remaining
    int regenCounter = 0;
    int starveCounter = 0;
    int drownCounter = 0;

    static constexpr int MAX_FOOD = 20;
    static constexpr int MAX_AIR = 300;              // 15 s
    static constexpr int AIR_REFILL_PER_TICK = 8;    // full refill in ~2 s
    static constexpr int DROWN_DAMAGE_INTERVAL = 20; // 2 dmg/s once out of air
    static constexpr int DROWN_DAMAGE = 2;
    static constexpr int REGEN_INTERVAL = 80; // +1 hp / 4 s at high food
    static constexpr int REGEN_FOOD_MIN = 18; // food needed to regenerate
    static constexpr float REGEN_EXHAUSTION = 3.0f;
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
