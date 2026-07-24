#pragma once

#include "world/generator_v4.hpp"
#include "world/save_manager.hpp"

#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// World list - the single home for enumerating, creating, and deleting the
// named worlds under <root>/saves, plus the legacy <root>/rycraft_world
// save, which is adopted in place (never moved: a half-finished migration of
// a live save is worse than two roots). The game passes the default root
// "."; tests pass a TempDir.
// ---------------------------------------------------------------------------

inline constexpr const char* SAVES_ROOT = "saves";
inline constexpr const char* LEGACY_WORLD_DIRECTORY = "rycraft_world";
inline constexpr const char* GENERATOR_V4_WORLD_DIRECTORY = worldgen::v4_profile::WORLD_DIRECTORY;
inline constexpr size_t MAX_WORLD_NAME_LENGTH = 24;

// The world-name charset. Bounded so the minimal JSON writer never needs
// escapes and the bitmap font can prove coverage of every typed name.
constexpr bool isWorldNameChar(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == ' ' ||
           c == '.' || c == '_' || c == '-';
}

struct WorldSummary {
    std::string directory; // path a SaveManager opens
    SaveManager::WorldMetadata metadata;

    [[nodiscard]] bool requiresGeneratorV4Successor() const noexcept {
        return metadata.generatorVersion != SaveManager::GENERATOR_V4_VERSION;
    }
};

// Every world under saves/ plus the legacy directory when present, most
// recently played first. A world with unreadable metadata still lists (name
// falls back to the directory) instead of vanishing from the screen.
std::vector<WorldSummary> listWorlds(const std::string& root = ".");

// Generator v4 profiles live directly under the root so their immutable
// authority and hydrology stores never overlap legacy saves. This includes
// the default profile and stable seed-and-fingerprint sibling profiles.
std::vector<WorldSummary> listGeneratorV4Worlds(const std::string& root = ".");

// The generator v4 Worlds screen also lists legacy profiles as read-only
// sources for the explicit successor action. Starting that action creates a
// separate v4 profile and never writes into, moves, or deletes the source.
std::vector<WorldSummary> listWorldsForGeneratorV4(const std::string& root = ".");

// Display name -> collision-free directory name under <root>/saves using
// [A-Za-z0-9_-]; empty input becomes "world".
std::string sanitizeWorldDirectory(const std::string& name, const std::string& root = ".");

// Creates <root>/saves/<sanitized>/ with initial metadata and returns the
// new directory, or nullopt (with a log) on I/O failure. Survival worlds
// start with an empty inventory; creative worlds keep the classic palette.
std::optional<std::string> createWorld(const std::string& name, uint32_t seed, GameMode mode,
                                       const GenerationSettings& generation,
                                       const std::string& root = ".");

// Deletes a world directory. Guarded: only the legacy directory or a direct
// child of <root>/saves that actually looks like a world (metadata or
// regions) can be removed.
bool deleteWorld(const std::string& directory, const std::string& root = ".");
