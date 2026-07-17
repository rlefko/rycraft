#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Per-world configuration — the single home for the game-mode rules and the
// generation toggles a world is created with. Both persist in metadata.json
// and never change the deterministic output of an unchanged toggle set.
// ---------------------------------------------------------------------------

enum class GameMode : uint8_t { SURVIVAL = 0, CREATIVE = 1 };

constexpr const char* gameModeName(GameMode mode) {
    return mode == GameMode::CREATIVE ? "Creative" : "Survival";
}

// The semantics matrix: every mode difference reads from these predicates so
// a rule never forks across call sites.
constexpr bool modeAllowsFlight(GameMode mode) {
    return mode == GameMode::CREATIVE;
}
constexpr bool modeInstantBreak(GameMode mode) {
    return mode == GameMode::CREATIVE;
}
constexpr bool modeTakesDamage(GameMode mode) {
    return mode == GameMode::SURVIVAL;
}
constexpr bool modeDrainsHunger(GameMode mode) {
    return mode == GameMode::SURVIVAL;
}
constexpr bool modeConsumesItems(GameMode mode) {
    return mode == GameMode::SURVIVAL;
}
constexpr bool modeBlockDrops(GameMode mode) {
    return mode == GameMode::SURVIVAL;
}

// World-creation toggles. Structures feeds the generator; fauna, weather,
// and the day cycle are engine-side gates. Defaults reproduce legacy worlds
// byte for byte.
struct GenerationSettings {
    bool structures = true;
    bool fauna = true;
    bool weather = true;
    bool dayCycle = true;

    constexpr bool operator==(const GenerationSettings&) const = default;
};
