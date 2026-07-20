#include "world/world_list.hpp"

#include "common/error.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

bool looksLikeWorld(const fs::path& directory) {
    std::error_code ec;
    return fs::exists(directory / "metadata.json", ec) ||
           fs::exists(directory / SaveManager::CURRENT_REGIONS_DIRECTORY, ec);
}

WorldSummary summarize(const fs::path& directory) {
    WorldSummary summary;
    summary.directory = directory.string();
    if (auto metadata = SaveManager::readMetadataFile((directory / "metadata.json").string())) {
        summary.metadata = *metadata;
    }
    if (summary.metadata.name.empty()) {
        summary.metadata.name = directory.filename().string();
    }
    return summary;
}

} // namespace

std::vector<WorldSummary> listWorlds(const std::string& root) {
    std::vector<WorldSummary> worlds;
    std::error_code ec;

    const fs::path legacy = fs::path(root) / LEGACY_WORLD_DIRECTORY;
    if (fs::is_directory(legacy, ec) && looksLikeWorld(legacy)) {
        worlds.push_back(summarize(legacy));
    }

    const fs::path savesRoot = fs::path(root) / SAVES_ROOT;
    if (fs::is_directory(savesRoot, ec)) {
        for (const auto& entry : fs::directory_iterator(savesRoot, ec)) {
            if (!entry.is_directory() || !looksLikeWorld(entry.path())) continue;
            worlds.push_back(summarize(entry.path()));
        }
    }

    std::stable_sort(worlds.begin(), worlds.end(), [](const auto& left, const auto& right) {
        return left.metadata.lastPlayedMs > right.metadata.lastPlayedMs;
    });
    return worlds;
}

std::string sanitizeWorldDirectory(const std::string& name, const std::string& root) {
    std::string base;
    for (char c : name) {
        if (base.size() >= MAX_WORLD_NAME_LENGTH) break;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
            c == '_' || c == '-') {
            base.push_back(c);
        } else if (isWorldNameChar(c)) {
            base.push_back('_');
        }
    }
    if (base.empty()) base = "world";

    std::string candidate = base;
    std::error_code ec;
    for (int suffix = 2; fs::exists(fs::path(root) / SAVES_ROOT / candidate, ec); ++suffix) {
        candidate = base + "_" + std::to_string(suffix);
    }
    return candidate;
}

std::optional<std::string> createWorld(const std::string& name, uint32_t seed, GameMode mode,
                                       const GenerationSettings& generation,
                                       const std::string& root) {
    const fs::path directory = fs::path(root) / SAVES_ROOT / sanitizeWorldDirectory(name, root);
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        RY_LOG_ERROR("Failed to create the world directory");
        return std::nullopt;
    }

    SaveManager::WorldMetadata metadata;
    metadata.seed = seed;
    metadata.spawnPos = Vec3{0.f, 100.f, 0.f};
    metadata.playerPos = metadata.spawnPos;
    metadata.name = name.empty() ? directory.filename().string() : name;
    metadata.gameMode = mode;
    metadata.generation = generation;
    metadata.createdMs =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());
    if (mode == GameMode::SURVIVAL) {
        metadata.player.inventory.fill(ItemStack{});
    }
    if (!generation.dayCycle) {
        metadata.worldTime = 6000; // frozen clocks read noon, not midnight
    }

    SaveManager saves(directory.string());
    if (!saves.saveMetadata(metadata)) {
        RY_LOG_ERROR("Failed to write initial world metadata");
        return std::nullopt;
    }
    return directory.string();
}

bool deleteWorld(const std::string& directory, const std::string& root) {
    const fs::path path(directory);
    std::error_code ec;

    // Containment guard: remove_all must never walk outside the two world
    // roots. The path must be exactly the legacy directory or a direct child
    // of <root>/saves whose name is free of traversal segments.
    const std::string leaf = path.filename().string();
    const bool isLegacy = path == fs::path(root) / LEGACY_WORLD_DIRECTORY;
    const bool underSaves = path.parent_path() == fs::path(root) / SAVES_ROOT && !leaf.empty() &&
                            leaf != "." && leaf != ".." && leaf.find('/') == std::string::npos;
    if (!isLegacy && !underSaves) {
        RY_LOG_ERROR("Refused to delete a path outside the world roots");
        return false;
    }
    if (!fs::is_directory(path, ec) || !looksLikeWorld(path)) {
        RY_LOG_ERROR("Refused to delete a directory that is not a world");
        return false;
    }
    fs::remove_all(path, ec);
    if (ec) {
        RY_LOG_ERROR("World deletion failed");
        return false;
    }
    return true;
}
