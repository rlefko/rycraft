#include "world/save_manager.hpp"

#include "common/error.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <lz4.h>
#include <sstream>

namespace fs = std::filesystem;

namespace {

template <typename Number, typename Parser>
bool parseNumber(const std::string& content, const char* field, Number& value, Parser parser) {
    size_t position = content.find(field);
    if (position == std::string::npos) return false;
    position += std::strlen(field);
    while (position < content.size() && (content[position] == ' ' || content[position] == '\n')) {
        ++position;
    }
    try {
        value = static_cast<Number>(parser(content.substr(position)));
        return true;
    } catch (...) {
        return false;
    }
}

// Reads `count` unsigned integers after `field`, skipping brackets, commas,
// and whitespace. The minimal-JSON metadata format never nests arrays of
// anything but numbers, so a flat scan is sufficient and stays tolerant.
bool parseNumberList(const std::string& content, const char* field, unsigned long* values,
                     size_t count) {
    size_t position = content.find(field);
    if (position == std::string::npos) return false;
    position = content.find('[', position);
    if (position == std::string::npos) return false;
    ++position;
    for (size_t index = 0; index < count; ++index) {
        while (position < content.size() &&
               (content[position] == ' ' || content[position] == '\n' || content[position] == ',' ||
                content[position] == '[' || content[position] == ']')) {
            ++position;
        }
        if (position >= content.size()) return false;
        size_t consumed = 0;
        try {
            values[index] = std::stoul(content.substr(position), &consumed);
        } catch (...) {
            return false;
        }
        if (consumed == 0) return false;
        position += consumed;
    }
    return true;
}

// Legacy nine-block hotbar array ("inventory"); each entry becomes a
// count-one stack.
bool parseLegacyInventory(const std::string& content,
                          std::array<ItemStack, SaveManager::PLAYER_INVENTORY_SLOTS>& inventory) {
    std::array<unsigned long, SaveManager::PLAYER_HOTBAR_SLOTS> values{};
    if (!parseNumberList(content, "\"inventory\"", values.data(), values.size())) return false;
    for (unsigned long value : values) {
        if (value >= static_cast<unsigned long>(BlockType::COUNT)) return false;
    }
    for (size_t slot = 0; slot < values.size(); ++slot) {
        const auto block = static_cast<BlockType>(values[slot]);
        inventory[slot] =
            block == BlockType::AIR ? ItemStack{} : ItemStack{itemFromBlock(block), 1, 0};
    }
    return true;
}

// Current stack arrays use [[type,count,durability],...]. Out-of-range item
// ids and zero counts read as empty slots so newer saves stay loadable.
template <size_t SlotCount>
bool parseStackSlots(const std::string& content, const char* field,
                     std::array<ItemStack, SlotCount>& slots) {
    std::array<unsigned long, SlotCount * 3> values{};
    if (!parseNumberList(content, field, values.data(), values.size())) return false;
    for (size_t slot = 0; slot < slots.size(); ++slot) {
        const unsigned long type = values[slot * 3];
        const unsigned long count = values[slot * 3 + 1];
        const unsigned long durability = values[slot * 3 + 2];
        if (type == 0 || type > UINT16_MAX || !isValidItemId(static_cast<uint16_t>(type)) ||
            count == 0) {
            slots[slot] = ItemStack{};
            continue;
        }
        const auto item = static_cast<ItemType>(type);
        const auto capped =
            static_cast<uint8_t>(std::min(count, static_cast<unsigned long>(maxStackSize(item))));
        slots[slot] = ItemStack{item, capped, static_cast<uint16_t>(durability)};
    }
    return true;
}

bool parseInventorySlots(const std::string& content,
                         std::array<ItemStack, SaveManager::PLAYER_INVENTORY_SLOTS>& inventory) {
    return parseStackSlots(content, "\"inventorySlots\"", inventory);
}

bool parseCarriedStacks(const std::string& content,
                        std::array<ItemStack, SaveManager::PLAYER_CARRIED_SLOTS>& carriedStacks) {
    return parseStackSlots(content, "\"carriedStacks\"", carriedStacks);
}

// One quoted string value; the world-name charset is restricted at input so
// escapes never occur in a file this game wrote.
bool parseString(const std::string& content, const char* field, std::string& value) {
    size_t position = content.find(field);
    if (position == std::string::npos) return false;
    position += std::strlen(field);
    while (position < content.size() && (content[position] == ' ' || content[position] == '\n')) {
        ++position;
    }
    if (position >= content.size() || content[position] != '"') return false;
    const size_t end = content.find('"', position + 1);
    if (end == std::string::npos) return false;
    value = content.substr(position + 1, end - position - 1);
    return true;
}

uint64_t wallClockMs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
}

bool parseBoolean(const std::string& content, const char* field, bool& value) {
    size_t position = content.find(field);
    if (position == std::string::npos) return false;
    position += std::strlen(field);
    while (position < content.size() && (content[position] == ' ' || content[position] == '\n')) {
        ++position;
    }
    if (content.compare(position, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (content.compare(position, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

enum class Vec3FieldStatus : uint8_t {
    Missing,
    Null,
    Value,
    Invalid,
};

bool finiteVec3(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

// The largest float below 2^63 still leaves more than 500 billion blocks
// before either signed integer limit. Keeping metadata strictly within this
// boundary makes floor-to-int64 conversion and nearby startup probes safe.
constexpr float MAX_METADATA_HORIZONTAL_COORDINATE = 0x1.fffffcp+62F;

bool validMetadataPosition(const Vec3& value) {
    return finiteVec3(value) && value.y >= static_cast<float>(WORLD_MIN_Y) &&
           value.y <= static_cast<float>(WORLD_MAX_Y) &&
           value.x >= -MAX_METADATA_HORIZONTAL_COORDINATE &&
           value.x <= MAX_METADATA_HORIZONTAL_COORDINATE &&
           value.z >= -MAX_METADATA_HORIZONTAL_COORDINATE &&
           value.z <= MAX_METADATA_HORIZONTAL_COORDINATE;
}

Vec3FieldStatus parseVec3Field(const std::string& content, const char* field, Vec3& value) {
    size_t position = content.find(field);
    if (position == std::string::npos) return Vec3FieldStatus::Missing;
    position += std::strlen(field);
    while (position < content.size() && (content[position] == ' ' || content[position] == '\n' ||
                                         content[position] == '\r' || content[position] == '\t')) {
        ++position;
    }
    if (content.compare(position, 4, "null") == 0) return Vec3FieldStatus::Null;
    if (position >= content.size() || content[position] != '{') return Vec3FieldStatus::Invalid;
    const size_t end = content.find('}', position + 1);
    if (end == std::string::npos) return Vec3FieldStatus::Invalid;

    const std::string object = content.substr(position, end - position + 1);
    Vec3 parsed;
    if (!parseNumber(object, "\"x\":", parsed.x,
                     [](const std::string& text) { return std::stof(text); }) ||
        !parseNumber(object, "\"y\":", parsed.y,
                     [](const std::string& text) { return std::stof(text); }) ||
        !parseNumber(object, "\"z\":", parsed.z,
                     [](const std::string& text) { return std::stof(text); }) ||
        !finiteVec3(parsed)) {
        return Vec3FieldStatus::Invalid;
    }
    value = parsed;
    return Vec3FieldStatus::Value;
}

bool validGenerationFingerprint(std::string_view fingerprint) {
    return fingerprint.size() == 64 && std::ranges::all_of(fingerprint, [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}

bool validPersistedStack(const ItemStack& stack) {
    const uint16_t type = static_cast<uint16_t>(stack.type);
    if (type == static_cast<uint16_t>(ItemType::NONE))
        return stack.count == 0 && stack.durability == 0;
    return isValidItemId(type) && stack.count > 0 && stack.count <= maxStackSize(stack.type);
}

template <size_t SlotCount>
bool validPersistedStacks(const std::array<ItemStack, SlotCount>& stacks) {
    return std::ranges::all_of(stacks, validPersistedStack);
}

bool blockPositionLess(const BlockPos& left, const BlockPos& right) {
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    return left.z < right.z;
}

bool frontierLess(const FluidBoundaryFrontier& left, const FluidBoundaryFrontier& right) {
    if (left.available != right.available) {
        return blockPositionLess(left.available, right.available);
    }
    return blockPositionLess(left.unavailable, right.unavailable);
}

constexpr bool validSectionY(int32_t y) {
    return y >= WORLD_MIN_CHUNK_Y && y <= WORLD_MAX_CHUNK_Y;
}

bool validFrontier(const FluidBoundaryFrontier& frontier) {
    const BlockPos& available = frontier.available;
    const BlockPos& unavailable = frontier.unavailable;
    if (available.y < WORLD_MIN_Y || available.y > WORLD_MAX_Y || unavailable.y < WORLD_MIN_Y ||
        unavailable.y > WORLD_MAX_Y) {
        return false;
    }
    const bool adjacentX = available.y == unavailable.y && available.z == unavailable.z &&
                           ((available.x != INT64_MAX && unavailable.x == available.x + 1) ||
                            (available.x != INT64_MIN && unavailable.x == available.x - 1));
    const bool adjacentY = available.x == unavailable.x && available.z == unavailable.z &&
                           (unavailable.y == available.y + 1 || unavailable.y == available.y - 1);
    const bool adjacentZ = available.x == unavailable.x && available.y == unavailable.y &&
                           ((available.z != INT64_MAX && unavailable.z == available.z + 1) ||
                            (available.z != INT64_MIN && unavailable.z == available.z - 1));
    if (!adjacentX && !adjacentY && !adjacentZ) return false;
    const ChunkPos availableCube{Chunk::worldToChunk(available.x),
                                 Chunk::worldToChunkY(available.y),
                                 Chunk::worldToChunk(available.z)};
    const ChunkPos unavailableCube{Chunk::worldToChunk(unavailable.x),
                                   Chunk::worldToChunkY(unavailable.y),
                                   Chunk::worldToChunk(unavailable.z)};
    return availableCube != unavailableCube;
}

bool parseManifestName(const std::string& name, ColumnPos& column) {
    long long x = 0;
    long long z = 0;
    int consumed = 0;
    if (std::sscanf(name.c_str(), "m.%lld.%lld.manifest%n", &x, &z, &consumed) != 2 ||
        consumed != static_cast<int>(name.size())) {
        return false;
    }
    column = {static_cast<int64_t>(x), static_cast<int64_t>(z)};
    return true;
}

bool parseCubeName(const std::string& name, ChunkPos& position) {
    long long x = 0;
    long long z = 0;
    int y = 0;
    int consumed = 0;
    if (std::sscanf(name.c_str(), "c.%lld.%d.%lld.dat%n", &x, &y, &z, &consumed) != 3 ||
        consumed != static_cast<int>(name.size()) || !validSectionY(y)) {
        return false;
    }
    position = {static_cast<int64_t>(x), static_cast<int32_t>(y), static_cast<int64_t>(z)};
    return true;
}

} // namespace

SaveManager::SaveManager(const std::string& worldPath, std::shared_ptr<TestHooks> testHooks)
    : SaveManager(worldPath, Profile::LegacyV3, std::move(testHooks)) {}

SaveManager::SaveManager(const std::string& worldPath, Profile profile,
                         std::shared_ptr<TestHooks> testHooks)
    : profile_(profile)
    , generatorVersion_(profile == Profile::GeneratorV4 ? GENERATOR_V4_VERSION
                                                        : CURRENT_GENERATOR_VERSION)
    , worldPath_(worldPath)
    , regionsDirectory_(profile == Profile::GeneratorV4 ? V4_REGIONS_DIRECTORY
                                                        : CURRENT_REGIONS_DIRECTORY)
    , regionsPath_(worldPath + "/" + regionsDirectory_)
    , terrainAuthorityPath_(profile == Profile::GeneratorV4
                                ? worldPath + "/" + V4_TERRAIN_AUTHORITY_DIRECTORY
                                : std::string{})
    , hydrologyAuthorityPath_(profile == Profile::GeneratorV4
                                  ? worldPath + "/" + V4_HYDROLOGY_AUTHORITY_DIRECTORY
                                  : std::string{})
    , metadataPath_(worldPath + "/metadata.json")
    , blockEntitiesPath_(worldPath + "/block_entities.dat")
    , testHooks_(std::move(testHooks)) {
    ensureDirectory(worldPath_);
    ensureDirectory(regionsPath_);
    if (profile_ == Profile::GeneratorV4) {
        ensureDirectory(terrainAuthorityPath_);
        ensureDirectory(hydrologyAuthorityPath_);
    }
    loadManifestIndex();
    recoverOrphanedCubes();
    saveThread_ = std::thread(&SaveManager::saveLoop, this);
}

SaveManager::~SaveManager() {
    running_.store(false);
    if (testHooks_) {
        testHooks_->pauseWrites.store(false, std::memory_order_release);
        testHooks_->pauseWrites.notify_all();
    }
    saveCondition_.notify_one();
    if (saveThread_.joinable()) saveThread_.join();
}

void SaveManager::saveChunk(const Chunk& chunk) {
    if (!validSectionY(chunk.chunkY)) return;
    enqueueChunkSnapshot(std::make_shared<const Chunk>(chunk));
}

void SaveManager::saveChunkAsync(std::shared_ptr<const Chunk> chunk) {
    if (!chunk || !validSectionY(chunk->chunkY)) return;
    enqueueChunkSnapshot(std::make_shared<const Chunk>(*chunk));
}

void SaveManager::enqueueChunkSnapshot(std::shared_ptr<const Chunk> chunk) {
    const ChunkPos pos = chunk->pos();
    {
        std::unique_lock<std::mutex> lock(saveMutex_);
        if (queuedChunks_.contains(pos)) {
            pendingChunks_[pos] = std::move(chunk);
            coalescedSaves_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        saveCondition_.wait(lock, [this] {
            return pendingWrites_.load(std::memory_order_relaxed) < MAX_PENDING_SAVE_JOBS ||
                   !running_.load();
        });
        if (!running_.load()) return;
        pendingChunks_[pos] = chunk;
        saveQueue_.push_back(pos);
        queuedChunks_.insert(pos);
        pendingWrites_.fetch_add(1, std::memory_order_release);
    }
    saveCondition_.notify_one();
}

std::optional<Chunk> SaveManager::loadChunk(ChunkPos pos) const {
    if (!validSectionY(pos.y)) return std::nullopt;
    {
        std::lock_guard<std::mutex> lock(saveMutex_);
        auto pending = pendingChunks_.find(pos);
        if (pending != pendingChunks_.end()) return Chunk(*pending->second);
    }

    const std::string path = getChunkPath(pos);
    auto fileData = readFile(path);
    if (!fileData) {
        std::error_code error;
        if (fs::exists(path, error) && !error) reportLoadFailureOnce(pos, "file could not be read");
        return std::nullopt;
    }
    constexpr size_t MAX_CUBE_BYTES = HEADER_SIZE + CHUNK_VOLUME * 2;
    auto decompressed = decompress(*fileData, MAX_CUBE_BYTES);
    if (decompressed.empty()) {
        reportLoadFailureOnce(pos, "LZ4 payload is corrupt");
        return std::nullopt;
    }
    const ChunkPayloadValidation validation = ChunkSerializer::validatePayload(decompressed);
    if (validation == ChunkPayloadValidation::CHECKSUM_MISMATCH) {
        reportLoadFailureOnce(pos, "payload checksum does not match");
        return std::nullopt;
    }
    if (validation == ChunkPayloadValidation::INCOMPATIBLE) {
        reportLoadFailureOnce(pos, "header or payload is incompatible");
        return std::nullopt;
    }
    auto chunk = ChunkSerializer::deserialize(decompressed);
    if (!chunk || chunk->pos() != pos) {
        reportLoadFailureOnce(pos, "header or payload is incompatible");
        return std::nullopt;
    }
    return chunk;
}

void SaveManager::reportLoadFailureOnce(ChunkPos pos, const char* reason) const {
    {
        std::lock_guard<std::mutex> lock(saveMutex_);
        if (!reportedLoadFailures_.insert(pos).second) return;
    }
    if (testHooks_) testHooks_->loadFailuresReported.fetch_add(1, std::memory_order_relaxed);
    RY_LOG_ERROR((std::string("Cube save rejected at ") + std::to_string(pos.x) + "," +
                  std::to_string(pos.y) + "," + std::to_string(pos.z) + ": " + reason)
                     .c_str());
}

std::vector<int32_t> SaveManager::savedSections(ColumnPos pos) const {
    std::lock_guard<std::mutex> lock(manifestMutex_);
    auto manifest = manifests_.find(pos);
    return manifest == manifests_.end() ? std::vector<int32_t>{} : manifest->second.editedSections;
}

std::unordered_map<ColumnPos, std::vector<int32_t>>
SaveManager::savedSectionsForColumns(std::span<const ColumnPos> columns) const {
    std::unordered_map<ColumnPos, std::vector<int32_t>> result;
    result.reserve(columns.size());
    std::lock_guard<std::mutex> lock(manifestMutex_);
    for (ColumnPos column : columns) {
        const auto manifest = manifests_.find(column);
        if (manifest != manifests_.end() && !manifest->second.editedSections.empty()) {
            result.emplace(column, manifest->second.editedSections);
        }
    }
    return result;
}

bool SaveManager::saveDeferredFluidFrontiers(const std::vector<FluidBoundaryFrontier>& frontiers) {
    if (!std::all_of(frontiers.begin(), frontiers.end(), validFrontier)) return false;

    std::unordered_map<ColumnPos, std::vector<FluidBoundaryFrontier>> grouped;
    for (const FluidBoundaryFrontier& frontier : frontiers) {
        ColumnPos column{Chunk::worldToChunk(frontier.available.x),
                         Chunk::worldToChunk(frontier.available.z)};
        grouped[column].push_back(frontier);
    }

    std::lock_guard<std::mutex> writerLock(manifestWriteMutex_);
    std::unordered_map<ColumnPos, ColumnManifest> current;
    {
        std::lock_guard<std::mutex> lock(manifestMutex_);
        current = manifests_;
    }
    std::vector<ColumnPos> columns;
    columns.reserve(current.size() + grouped.size());
    for (const auto& [column, manifest] : current)
        columns.push_back(column);
    for (const auto& [column, frontiersForColumn] : grouped)
        columns.push_back(column);
    std::ranges::sort(columns, [](ColumnPos left, ColumnPos right) {
        return left.x != right.x ? left.x < right.x : left.z < right.z;
    });
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());

    bool saved = true;
    for (ColumnPos column : columns) {
        ColumnManifest nextManifest;
        auto existing = current.find(column);
        if (existing != current.end()) nextManifest = existing->second;
        auto replacement = grouped.find(column);
        std::vector<FluidBoundaryFrontier> next = replacement == grouped.end()
                                                      ? std::vector<FluidBoundaryFrontier>{}
                                                      : std::move(replacement->second);
        std::sort(next.begin(), next.end(), frontierLess);
        next.erase(std::unique(next.begin(), next.end()), next.end());
        if (nextManifest.fluidFrontiers == next) continue;
        nextManifest.fluidFrontiers = std::move(next);
        if (!writeManifest(column, nextManifest)) {
            saved = false;
            continue;
        }
        std::lock_guard<std::mutex> lock(manifestMutex_);
        manifests_[column] = std::move(nextManifest);
    }
    return saved;
}

std::vector<FluidBoundaryFrontier> SaveManager::loadDeferredFluidFrontiers() const {
    std::lock_guard<std::mutex> lock(manifestMutex_);
    std::vector<FluidBoundaryFrontier> result;
    for (const auto& [column, manifest] : manifests_) {
        result.insert(result.end(), manifest.fluidFrontiers.begin(), manifest.fluidFrontiers.end());
    }
    std::sort(result.begin(), result.end(), frontierLess);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool SaveManager::saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime) {
    return saveMetadata(seed, spawnPos, worldTime, PlayerMetadata{});
}

bool SaveManager::saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime,
                               const PlayerMetadata& player) {
    if (profile_ != Profile::LegacyV3) return false;
    WorldMetadata metadata;
    metadata.seed = seed;
    metadata.spawnPos = spawnPos;
    metadata.playerPos = spawnPos;
    metadata.worldTime = worldTime;
    metadata.player = player;
    return saveMetadata(metadata);
}

bool SaveManager::saveV4Metadata(uint64_t seed, std::string_view generationFingerprint,
                                 Vec3 playerPos, std::optional<Vec3> safeSpawnPos,
                                 uint64_t worldTime, bool spawnFinalized,
                                 uint32_t spawnSafetyRevision) {
    return saveV4Metadata(seed, generationFingerprint, playerPos, std::move(safeSpawnPos),
                          worldTime, PlayerMetadata{}, spawnFinalized, spawnSafetyRevision);
}

bool SaveManager::saveV4Metadata(uint64_t seed, std::string_view generationFingerprint,
                                 Vec3 playerPos, std::optional<Vec3> safeSpawnPos,
                                 uint64_t worldTime, const PlayerMetadata& player,
                                 bool spawnFinalized, uint32_t spawnSafetyRevision) {
    WorldMetadata metadata;
    metadata.seed = seed;
    metadata.generationFingerprint = generationFingerprint;
    metadata.spawnFinalized = spawnFinalized;
    metadata.spawnSafetyRevision = spawnSafetyRevision;
    metadata.playerPos = playerPos;
    metadata.safeSpawnPos = std::move(safeSpawnPos);
    metadata.spawnPos = metadata.safeSpawnPos.value_or(playerPos);
    metadata.worldTime = worldTime;
    metadata.generatorVersion = GENERATOR_V4_VERSION;
    metadata.player = player;
    return saveMetadata(metadata);
}

bool SaveManager::saveMetadata(const WorldMetadata& metadata) {
    if (!validMetadataPosition(metadata.spawnPos) || !validMetadataPosition(metadata.playerPos) ||
        (metadata.safeSpawnPos && !validMetadataPosition(*metadata.safeSpawnPos)) ||
        !validPersistedStacks(metadata.player.inventory) ||
        !validPersistedStacks(metadata.player.carriedStacks)) {
        return false;
    }
    if (profile_ == Profile::GeneratorV4) {
        if (!validGenerationFingerprint(metadata.generationFingerprint) ||
            metadata.spawnFinalized != metadata.safeSpawnPos.has_value()) {
            return false;
        }
        std::error_code error;
        if (fs::exists(metadataPath_, error)) {
            const std::optional<WorldMetadata> existing = readMetadataFile(metadataPath_);
            if (!existing || existing->generatorVersion != GENERATOR_V4_VERSION ||
                existing->seed != metadata.seed ||
                existing->generationFingerprint != metadata.generationFingerprint ||
                existing->generation.structures != metadata.generation.structures) {
                return false;
            }
        }
        if (error) return false;
    } else if (metadata.seed > UINT32_MAX) {
        return false;
    }
    return writeMetadata(metadata);
}

bool SaveManager::writeMetadata(const WorldMetadata& metadata) {
    Vec3 playerPos = metadata.playerPos;
    if (profile_ == Profile::LegacyV3 && playerPos == Vec3{}) playerPos = metadata.spawnPos;

    std::ostringstream json;
    json << "{\n"
         << "  \"name\": \"" << metadata.name << "\",\n"
         << "  \"seed\": " << metadata.seed << ",\n"
         << "  \"gameMode\": " << static_cast<unsigned>(metadata.gameMode) << ",\n"
         << "  \"generation\": {\n"
         << "    \"structures\": " << (metadata.generation.structures ? 1 : 0) << ",\n"
         << "    \"fauna\": " << (metadata.generation.fauna ? 1 : 0) << ",\n"
         << "    \"weather\": " << (metadata.generation.weather ? 1 : 0) << ",\n"
         << "    \"dayCycle\": " << (metadata.generation.dayCycle ? 1 : 0) << "\n"
         << "  },\n"
         << "  \"createdMs\": " << metadata.createdMs << ",\n"
         << "  \"lastPlayedMs\": " << wallClockMs() << ",\n";
    if (profile_ == Profile::GeneratorV4) {
        json << "  \"generationFingerprint\": \"" << metadata.generationFingerprint << "\",\n"
             << "  \"spawnFinalized\": " << (metadata.spawnFinalized ? "true" : "false") << ",\n"
             << "  \"spawnSafetyRevision\": " << metadata.spawnSafetyRevision << ",\n"
             << "  \"safeSpawnPos\": ";
        if (metadata.safeSpawnPos) {
            json << "{\n"
                 << "    \"x\": " << metadata.safeSpawnPos->x << ",\n"
                 << "    \"y\": " << metadata.safeSpawnPos->y << ",\n"
                 << "    \"z\": " << metadata.safeSpawnPos->z << "\n"
                 << "  },\n";
        } else {
            json << "null,\n";
        }
    }
    json << "  \"spawnPos\": {\n"
         << "    \"x\": " << metadata.spawnPos.x << ",\n"
         << "    \"y\": " << metadata.spawnPos.y << ",\n"
         << "    \"z\": " << metadata.spawnPos.z << "\n"
         << "  },\n"
         << "  \"bedSpawnSet\": " << (metadata.bedSpawnSet ? "true" : "false") << ",\n"
         << "  \"playerPos\": {\n"
         << "    \"x\": " << playerPos.x << ",\n"
         << "    \"y\": " << playerPos.y << ",\n"
         << "    \"z\": " << playerPos.z << "\n"
         << "  },\n"
         << "  \"worldTime\": " << metadata.worldTime << ",\n"
         << "  \"player\": {\n"
         << "    \"yaw\": " << metadata.player.yaw << ",\n"
         << "    \"pitch\": " << metadata.player.pitch << ",\n"
         << "    \"health\": " << metadata.player.health << ",\n"
         << "    \"hunger\": " << metadata.player.hunger << ",\n"
         << "    \"selectedSlot\": " << metadata.player.selectedSlot << ",\n"
         << "    \"inventorySlots\": [";
    for (size_t slot = 0; slot < metadata.player.inventory.size(); ++slot) {
        const ItemStack& stack = metadata.player.inventory[slot];
        if (slot != 0) json << ", ";
        json << '[' << static_cast<unsigned>(stack.type) << ", "
             << static_cast<unsigned>(stack.count) << ", " << stack.durability << ']';
    }
    json << "],\n"
         << "    \"carriedStacks\": [";
    for (size_t slot = 0; slot < metadata.player.carriedStacks.size(); ++slot) {
        const ItemStack& stack = metadata.player.carriedStacks[slot];
        if (slot != 0) json << ", ";
        json << '[' << static_cast<unsigned>(stack.type) << ", "
             << static_cast<unsigned>(stack.count) << ", " << stack.durability << ']';
    }
    json << "]\n"
         << "  },\n"
         << "  \"chunkFormatVersion\": " << CHUNK_VERSION << ",\n"
         << "  \"generatorVersion\": " << generatorVersion_ << "\n"
         << "}\n";
    const std::string text = json.str();
    return writeFileWithRetries(metadataPath_, std::vector<uint8_t>(text.begin(), text.end()));
}

std::optional<SaveManager::WorldMetadata> SaveManager::loadMetadata() const {
    return inspectMetadata(worldPath_, profile_);
}

std::optional<SaveManager::WorldMetadata> SaveManager::readMetadataFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return std::nullopt;
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    WorldMetadata metadata;
    if (!parseNumber(content, "\"seed\":", metadata.seed,
                     [](const std::string& value) { return std::stoull(value); }) ||
        !parseNumber(content, "\"worldTime\":", metadata.worldTime,
                     [](const std::string& value) { return std::stoull(value); })) {
        return std::nullopt;
    }
    parseNumber(content, "\"chunkFormatVersion\":", metadata.chunkFormatVersion,
                [](const std::string& value) { return std::stoul(value); });
    parseNumber(content, "\"generatorVersion\":", metadata.generatorVersion,
                [](const std::string& value) { return std::stoul(value); });
    parseString(content, "\"name\":", metadata.name);
    unsigned long mode = static_cast<unsigned long>(GameMode::CREATIVE);
    parseNumber(content, "\"gameMode\":", mode,
                [](const std::string& value) { return std::stoul(value); });
    metadata.gameMode = mode == static_cast<unsigned long>(GameMode::SURVIVAL) ? GameMode::SURVIVAL
                                                                               : GameMode::CREATIVE;
    auto parseToggle = [&content](const char* field, bool& toggle) {
        int value = toggle ? 1 : 0;
        parseNumber(content, field, value, [](const std::string& text) { return std::stoi(text); });
        toggle = value != 0;
    };
    parseToggle("\"structures\":", metadata.generation.structures);
    parseToggle("\"fauna\":", metadata.generation.fauna);
    parseToggle("\"weather\":", metadata.generation.weather);
    parseToggle("\"dayCycle\":", metadata.generation.dayCycle);
    parseNumber(content, "\"createdMs\":", metadata.createdMs,
                [](const std::string& value) { return std::stoull(value); });
    parseNumber(content, "\"lastPlayedMs\":", metadata.lastPlayedMs,
                [](const std::string& value) { return std::stoull(value); });

    const Vec3FieldStatus spawnStatus = parseVec3Field(content, "\"spawnPos\":", metadata.spawnPos);
    Vec3FieldStatus playerStatus = parseVec3Field(content, "\"playerPos\":", metadata.playerPos);
    if (playerStatus != Vec3FieldStatus::Value || !validMetadataPosition(metadata.playerPos)) {
        Vec3 legacyPlayer{};
        const bool parsedLegacy =
            parseNumber(content, "\"px\":", legacyPlayer.x,
                        [](const std::string& value) { return std::stof(value); }) &&
            parseNumber(content, "\"py\":", legacyPlayer.y,
                        [](const std::string& value) { return std::stof(value); }) &&
            parseNumber(content, "\"pz\":", legacyPlayer.z,
                        [](const std::string& value) { return std::stof(value); }) &&
            validMetadataPosition(legacyPlayer);
        if (parsedLegacy) {
            metadata.playerPos = legacyPlayer;
            playerStatus = Vec3FieldStatus::Value;
        }
    }
    parseBoolean(content, "\"bedSpawnSet\":", metadata.bedSpawnSet);

    bool spawnValid =
        spawnStatus == Vec3FieldStatus::Value && validMetadataPosition(metadata.spawnPos);
    bool playerValid =
        playerStatus == Vec3FieldStatus::Value && validMetadataPosition(metadata.playerPos);

    if (metadata.generatorVersion == GENERATOR_V4_VERSION) {
        if (!parseString(content, "\"generationFingerprint\":", metadata.generationFingerprint) ||
            !validGenerationFingerprint(metadata.generationFingerprint) ||
            !parseBoolean(content, "\"spawnFinalized\":", metadata.spawnFinalized)) {
            return std::nullopt;
        }
        parseNumber(content, "\"spawnSafetyRevision\":", metadata.spawnSafetyRevision,
                    [](const std::string& value) { return std::stoul(value); });
        Vec3 safeSpawn;
        const Vec3FieldStatus safeSpawnStatus =
            parseVec3Field(content, "\"safeSpawnPos\":", safeSpawn);
        const bool safeSpawnValid =
            safeSpawnStatus == Vec3FieldStatus::Value && validMetadataPosition(safeSpawn);
        if (safeSpawnStatus == Vec3FieldStatus::Invalid ||
            (safeSpawnStatus == Vec3FieldStatus::Value && !safeSpawnValid)) {
            metadata.spawnFinalized = false;
            metadata.spawnSafetyRevision = 0;
        }
        // A provisional record is never allowed to supply a recovery anchor.
        // Current writers reject this combination, but treating an interrupted
        // or hand-edited older record as unvalidated is safer than allowing a
        // stale coordinate to steer dry-land recovery.
        if (safeSpawnValid && metadata.spawnFinalized) metadata.safeSpawnPos = safeSpawn;
        if (!spawnValid) {
            if (metadata.safeSpawnPos) {
                metadata.spawnPos = *metadata.safeSpawnPos;
                spawnValid = true;
            } else if (playerValid) {
                metadata.spawnPos = metadata.playerPos;
                spawnValid = true;
            } else {
                return std::nullopt;
            }
            // A repaired respawn location cannot retain bed provenance.
            metadata.bedSpawnSet = false;
        }
        if (!playerValid) {
            metadata.playerPos = metadata.safeSpawnPos.value_or(metadata.spawnPos);
            playerValid = true;
        }
    } else {
        if (!spawnValid) return std::nullopt;
        if (!playerValid) {
            metadata.playerPos = metadata.spawnPos;
            playerValid = true;
        }
    }
    if (!playerValid) return std::nullopt;
    parseNumber(content, "\"yaw\":", metadata.player.yaw,
                [](const std::string& value) { return std::stof(value); });
    parseNumber(content, "\"pitch\":", metadata.player.pitch,
                [](const std::string& value) { return std::stof(value); });
    parseNumber(content, "\"health\":", metadata.player.health,
                [](const std::string& value) { return std::stoi(value); });
    parseNumber(content, "\"hunger\":", metadata.player.hunger,
                [](const std::string& value) { return std::stoi(value); });
    parseNumber(content, "\"selectedSlot\":", metadata.player.selectedSlot,
                [](const std::string& value) { return std::stoi(value); });
    metadata.player.health = std::clamp(metadata.player.health, 0, 20);
    metadata.player.hunger = std::clamp(metadata.player.hunger, 0, 20);
    metadata.player.selectedSlot =
        std::clamp(metadata.player.selectedSlot, 0, static_cast<int>(PLAYER_HOTBAR_SLOTS) - 1);
    if (!parseInventorySlots(content, metadata.player.inventory)) {
        parseLegacyInventory(content, metadata.player.inventory);
    }
    parseCarriedStacks(content, metadata.player.carriedStacks);
    return metadata;
}

std::optional<SaveManager::WorldMetadata> SaveManager::inspectMetadata(const std::string& worldPath,
                                                                       Profile profile) {
    const std::string path = worldPath + "/metadata.json";
    std::optional<WorldMetadata> metadata = readMetadataFile(path);
    if (!metadata) return std::nullopt;
    if (profile == Profile::GeneratorV4 && metadata->generatorVersion != GENERATOR_V4_VERSION) {
        return std::nullopt;
    }
    if (profile == Profile::LegacyV3 && metadata->generatorVersion == GENERATOR_V4_VERSION) {
        return std::nullopt;
    }
    return metadata;
}

bool SaveManager::saveBlockEntities(const FurnaceMap& furnaces, const ChestMap& chests) {
    std::ostringstream out;
    out << "RYBE 1\n";
    auto stack = [&out](const ItemStack& item) {
        out << ' ' << static_cast<unsigned>(item.type) << ' ' << static_cast<unsigned>(item.count)
            << ' ' << item.durability;
    };

    // Deterministic order keeps the file diffable and the tests stable.
    std::vector<const FurnaceMap::value_type*> furnaceEntries;
    furnaceEntries.reserve(furnaces.size());
    for (const auto& entry : furnaces) {
        if (entry.first.y >= WORLD_MIN_Y && entry.first.y <= WORLD_MAX_Y) {
            furnaceEntries.push_back(&entry);
        }
    }
    std::sort(furnaceEntries.begin(), furnaceEntries.end(),
              [](const auto* left, const auto* right) {
                  return blockPositionLess(left->first, right->first);
              });
    for (const auto* entry : furnaceEntries) {
        const BlockPos& pos = entry->first;
        const FurnaceState& furnace = entry->second;
        out << "furnace " << pos.x << ' ' << pos.y << ' ' << pos.z << ' '
            << furnace.burnTicksRemaining << ' ' << furnace.burnTicksTotal << ' '
            << furnace.cookTicks;
        stack(furnace.input);
        stack(furnace.fuel);
        stack(furnace.output);
        out << '\n';
    }

    std::vector<const ChestMap::value_type*> chestEntries;
    chestEntries.reserve(chests.size());
    for (const auto& entry : chests) {
        if (entry.first.y >= WORLD_MIN_Y && entry.first.y <= WORLD_MAX_Y) {
            chestEntries.push_back(&entry);
        }
    }
    std::sort(chestEntries.begin(), chestEntries.end(), [](const auto* left, const auto* right) {
        return blockPositionLess(left->first, right->first);
    });
    for (const auto* entry : chestEntries) {
        const BlockPos& pos = entry->first;
        out << "chest " << pos.x << ' ' << pos.y << ' ' << pos.z;
        for (const ItemStack& slot : entry->second.slots) {
            stack(slot);
        }
        out << '\n';
    }

    const std::string text = out.str();
    return writeFileWithRetries(blockEntitiesPath_, std::vector<uint8_t>(text.begin(), text.end()));
}

SaveManager::BlockEntities SaveManager::loadBlockEntities() const {
    BlockEntities entities;
    std::ifstream file(blockEntitiesPath_);
    if (!file.is_open()) return entities;

    std::string header;
    if (!std::getline(file, header) || header.rfind("RYBE ", 0) != 0) {
        RY_LOG_ERROR("Block entities file has an unrecognized header; ignoring it");
        return entities;
    }

    // A slot triple is valid only when its type is a defined item id.
    auto decodeStack = [](unsigned type, unsigned count, unsigned durability) {
        if (type == 0 || !isValidItemId(static_cast<uint16_t>(type)) || count == 0) {
            return ItemStack{};
        }
        const auto item = static_cast<ItemType>(type);
        const auto capped = static_cast<uint8_t>(std::min<unsigned>(count, maxStackSize(item)));
        return ItemStack{item, capped, static_cast<uint16_t>(durability)};
    };

    std::string line;
    bool reportedMalformed = false;
    auto reportMalformed = [&reportedMalformed] {
        if (!reportedMalformed) {
            RY_LOG_ERROR("Dropped a malformed block entity line");
            reportedMalformed = true;
        }
    };
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream in(line);
        std::string record;
        in >> record;

        if (record == "furnace") {
            long long x = 0;
            long long z = 0;
            int y = 0;
            unsigned burnRemaining = 0;
            unsigned burnTotal = 0;
            unsigned cook = 0;
            std::array<unsigned, 9> stackValues{};
            in >> x >> y >> z >> burnRemaining >> burnTotal >> cook;
            for (unsigned& value : stackValues) {
                in >> value;
            }
            if (in.fail() || y < WORLD_MIN_Y || y > WORLD_MAX_Y) {
                reportMalformed();
                continue;
            }
            FurnaceState furnace;
            furnace.burnTicksRemaining = static_cast<uint16_t>(burnRemaining);
            furnace.burnTicksTotal = static_cast<uint16_t>(burnTotal);
            furnace.cookTicks = static_cast<uint16_t>(cook);
            furnace.input = decodeStack(stackValues[0], stackValues[1], stackValues[2]);
            furnace.fuel = decodeStack(stackValues[3], stackValues[4], stackValues[5]);
            furnace.output = decodeStack(stackValues[6], stackValues[7], stackValues[8]);
            entities.furnaces[BlockPos{static_cast<int64_t>(x), y, static_cast<int64_t>(z)}] =
                furnace;
        } else if (record == "chest") {
            long long x = 0;
            long long z = 0;
            int y = 0;
            in >> x >> y >> z;
            ChestState chest;
            bool malformed = false;
            for (ItemStack& slot : chest.slots) {
                unsigned type = 0;
                unsigned count = 0;
                unsigned durability = 0;
                in >> type >> count >> durability;
                if (in.fail()) {
                    malformed = true;
                    break;
                }
                slot = decodeStack(type, count, durability);
            }
            if (malformed || y < WORLD_MIN_Y || y > WORLD_MAX_Y) {
                reportMalformed();
                continue;
            }
            entities.chests[BlockPos{static_cast<int64_t>(x), y, static_cast<int64_t>(z)}] = chest;
        }
        // Any other record type loads as unknown for forward compatibility.
    }
    return entities;
}

bool SaveManager::flush() {
    std::unique_lock<std::mutex> lock(saveMutex_);
    saveCondition_.wait(lock, [this] {
        return saveQueue_.empty() && pendingWrites_.load(std::memory_order_acquire) == 0;
    });
    return failedChunks_.empty();
}

size_t SaveManager::pendingSaveCount() const {
    std::lock_guard<std::mutex> lock(saveMutex_);
    return pendingWrites_.load(std::memory_order_relaxed);
}

void SaveManager::saveLoop() {
    while (running_.load() || pendingWrites_.load(std::memory_order_acquire) > 0) {
        ChunkPos pos;
        std::shared_ptr<const Chunk> chunk;
        {
            std::unique_lock<std::mutex> lock(saveMutex_);
            saveCondition_.wait(lock, [this] { return !saveQueue_.empty() || !running_.load(); });
            if (saveQueue_.empty()) {
                if (!running_.load()) break;
                continue;
            }
            pos = saveQueue_.front();
            saveQueue_.pop_front();
            queuedChunks_.erase(pos);
            const auto pending = pendingChunks_.find(pos);
            if (pending != pendingChunks_.end()) chunk = pending->second;
        }
        saveCondition_.notify_all();

        if (testHooks_) {
            while (testHooks_->pauseWrites.load(std::memory_order_acquire) && running_.load()) {
                testHooks_->pauseWrites.wait(true, std::memory_order_relaxed);
            }
        }

        const auto compressed =
            chunk ? compress(ChunkSerializer::serialize(*chunk)) : std::vector<uint8_t>{};
        const bool saved = !compressed.empty() && ensureDirectory(getChunkDir({pos.x, pos.z})) &&
                           writeFileWithRetries(getChunkPath(pos), compressed) &&
                           updateManifest(pos);

        {
            std::lock_guard<std::mutex> lock(saveMutex_);
            if (saved) {
                failedChunks_.erase(pos);
                auto pending = pendingChunks_.find(pos);
                if (pending != pendingChunks_.end() && pending->second == chunk) {
                    pendingChunks_.erase(pending);
                }
            } else {
                failedChunks_.insert(pos);
            }
            pendingWrites_.fetch_sub(1, std::memory_order_release);
        }
        if (!saved) {
            RY_LOG_ERROR((std::string("Cube save failed after retries at ") +
                          std::to_string(pos.x) + "," + std::to_string(pos.y) + "," +
                          std::to_string(pos.z))
                             .c_str());
        }
        saveCondition_.notify_all();
    }
}

std::vector<uint8_t> SaveManager::compress(const std::vector<uint8_t>& data) const {
    const int maximum = LZ4_COMPRESSBOUND(static_cast<int>(data.size()));
    std::vector<uint8_t> compressed(sizeof(uint32_t) + static_cast<size_t>(maximum));
    const uint32_t originalSize = static_cast<uint32_t>(data.size());
    std::memcpy(compressed.data(), &originalSize, sizeof(originalSize));
    const int written =
        LZ4_compress_default(reinterpret_cast<const char*>(data.data()),
                             reinterpret_cast<char*>(compressed.data() + sizeof(originalSize)),
                             static_cast<int>(data.size()), maximum);
    if (written <= 0) return {};
    compressed.resize(sizeof(originalSize) + static_cast<size_t>(written));
    return compressed;
}

std::vector<uint8_t> SaveManager::decompress(const std::vector<uint8_t>& data,
                                             size_t maxDecompressedSize) const {
    if (data.size() <= sizeof(uint32_t)) return {};
    uint32_t originalSize = 0;
    std::memcpy(&originalSize, data.data(), sizeof(originalSize));
    if (originalSize == 0 || originalSize > maxDecompressedSize) return {};
    std::vector<uint8_t> decompressed(originalSize);
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(data.data() + sizeof(originalSize)),
        reinterpret_cast<char*>(decompressed.data()),
        static_cast<int>(data.size() - sizeof(originalSize)), static_cast<int>(originalSize));
    if (written != static_cast<int>(originalSize)) return {};
    return decompressed;
}

std::string SaveManager::getChunkDir(ColumnPos pos) const {
    std::ostringstream path;
    path << regionsPath_ << "/r." << getRegionCoord(pos.x) << "." << getRegionCoord(pos.z);
    return path.str();
}

std::string SaveManager::getChunkPath(ChunkPos pos) const {
    std::ostringstream path;
    path << getChunkDir({pos.x, pos.z}) << "/c." << pos.x << "." << pos.y << "." << pos.z << ".dat";
    return path.str();
}

std::string SaveManager::getManifestPath(ColumnPos pos) const {
    std::ostringstream path;
    path << getChunkDir(pos) << "/m." << pos.x << "." << pos.z << ".manifest";
    return path.str();
}

int64_t SaveManager::getRegionCoord(int64_t chunkCoord) {
    return world_coord::floorDiv(chunkCoord, int64_t{32});
}

bool SaveManager::updateManifest(ChunkPos pos) const {
    if (!validSectionY(pos.y)) return false;
    std::lock_guard<std::mutex> writerLock(manifestWriteMutex_);
    const ColumnPos column{pos.x, pos.z};
    ColumnManifest manifest;
    {
        std::lock_guard<std::mutex> lock(manifestMutex_);
        auto existing = manifests_.find(column);
        if (existing != manifests_.end()) manifest = existing->second;
    }
    manifest.editedSections.push_back(pos.y);
    std::sort(manifest.editedSections.begin(), manifest.editedSections.end());
    manifest.editedSections.erase(
        std::unique(manifest.editedSections.begin(), manifest.editedSections.end()),
        manifest.editedSections.end());
    if (!writeManifest(column, manifest)) return false;
    {
        std::lock_guard<std::mutex> lock(manifestMutex_);
        manifests_[column] = std::move(manifest);
    }
    return true;
}

void SaveManager::loadManifestIndex() {
    std::error_code error;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(regionsPath_, error)) {
        if (error) break;
        if (!entry.is_regular_file() || entry.path().extension() != ".manifest") continue;

        ColumnPos column;
        if (!parseManifestName(entry.path().filename().string(), column)) continue;

        ColumnManifest manifest;
        std::ifstream input(entry.path());
        std::string token = {};
        unsigned version = 0;
        int64_t claimedX = 0;
        int64_t claimedZ = 0;
        bool valid = static_cast<bool>(input >> token >> version) && token == "RYCM" &&
                     version == 1 && static_cast<bool>(input >> token >> claimedX >> claimedZ) &&
                     token == "column" && claimedX == column.x && claimedZ == column.z;
        while (input >> token) {
            if (!valid) break;
            if (token == "section") {
                int32_t section = 0;
                valid = static_cast<bool>(input >> section) && validSectionY(section);
                if (valid) manifest.editedSections.push_back(section);
            } else if (token == "frontier") {
                FluidBoundaryFrontier frontier;
                valid = static_cast<bool>(input >> frontier.available.x >> frontier.available.y >>
                                          frontier.available.z >> frontier.unavailable.x >>
                                          frontier.unavailable.y >> frontier.unavailable.z) &&
                        validFrontier(frontier) &&
                        ColumnPos{Chunk::worldToChunk(frontier.available.x),
                                  Chunk::worldToChunk(frontier.available.z)} == column;
                if (valid) manifest.fluidFrontiers.push_back(frontier);
            } else {
                valid = false;
            }
        }
        if (!valid) {
            RY_LOG_ERROR(
                (std::string("Column manifest rejected: ") + entry.path().string()).c_str());
            continue;
        }
        std::sort(manifest.editedSections.begin(), manifest.editedSections.end());
        manifest.editedSections.erase(
            std::unique(manifest.editedSections.begin(), manifest.editedSections.end()),
            manifest.editedSections.end());
        std::sort(manifest.fluidFrontiers.begin(), manifest.fluidFrontiers.end(), frontierLess);
        manifest.fluidFrontiers.erase(
            std::unique(manifest.fluidFrontiers.begin(), manifest.fluidFrontiers.end()),
            manifest.fluidFrontiers.end());
        manifests_[column] = std::move(manifest);
    }
}

void SaveManager::recoverOrphanedCubes() {
    std::unordered_map<ColumnPos, std::vector<int32_t>> recovered;
    std::error_code error;
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(regionsPath_, error)) {
        if (error) break;
        if (!entry.is_regular_file() || entry.path().extension() != ".dat") continue;

        ChunkPos position;
        if (!parseCubeName(entry.path().filename().string(), position) ||
            fs::path(getChunkPath(position)) != entry.path()) {
            continue;
        }
        const ColumnPos column{position.x, position.z};
        auto existing = manifests_.find(column);
        if (existing != manifests_.end() &&
            std::find(existing->second.editedSections.begin(),
                      existing->second.editedSections.end(),
                      position.y) != existing->second.editedSections.end()) {
            continue;
        }
        if (loadChunk(position)) recovered[column].push_back(position.y);
    }

    for (auto& [column, sections] : recovered) {
        ColumnManifest manifest = manifests_[column];
        manifest.editedSections.insert(manifest.editedSections.end(), sections.begin(),
                                       sections.end());
        std::sort(manifest.editedSections.begin(), manifest.editedSections.end());
        manifest.editedSections.erase(
            std::unique(manifest.editedSections.begin(), manifest.editedSections.end()),
            manifest.editedSections.end());
        manifests_[column] = manifest;
        if (!writeManifest(column, manifest)) {
            RY_LOG_ERROR((std::string("Recovered cube manifest could not be rewritten at ") +
                          std::to_string(column.x) + "," + std::to_string(column.z))
                             .c_str());
        }
    }
}

bool SaveManager::writeManifest(ColumnPos pos, const ColumnManifest& manifest) const {
    std::ostringstream text;
    text << "RYCM 1\n";
    text << "column " << pos.x << ' ' << pos.z << '\n';
    for (int32_t section : manifest.editedSections)
        text << "section " << section << '\n';
    for (const FluidBoundaryFrontier& frontier : manifest.fluidFrontiers) {
        text << "frontier " << frontier.available.x << ' ' << frontier.available.y << ' '
             << frontier.available.z << ' ' << frontier.unavailable.x << ' '
             << frontier.unavailable.y << ' ' << frontier.unavailable.z << '\n';
    }
    const std::string contents = text.str();
    return ensureDirectory(getChunkDir(pos)) &&
           writeFileWithRetries(getManifestPath(pos),
                                std::vector<uint8_t>(contents.begin(), contents.end()));
}

bool SaveManager::ensureDirectory(const std::string& path) const {
    std::error_code error;
    fs::create_directories(path, error);
    return !error;
}

bool SaveManager::writeFile(const std::string& path, const std::vector<uint8_t>& data) const {
    if (testHooks_) {
        size_t remaining = testHooks_->writeFailuresRemaining.load(std::memory_order_acquire);
        while (remaining != 0 && !testHooks_->writeFailuresRemaining.compare_exchange_weak(
                                     remaining, remaining - 1, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
        }
        if (remaining != 0) return false;
    }

    const std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) return false;
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size()));
        if (!file.good()) {
            file.close();
            std::error_code removeError;
            fs::remove(temporary, removeError);
            return false;
        }
    }
    if (std::rename(temporary.c_str(), path.c_str()) == 0) return true;
    std::error_code removeError;
    fs::remove(temporary, removeError);
    return false;
}

bool SaveManager::writeFileWithRetries(const std::string& path,
                                       const std::vector<uint8_t>& data) const {
    constexpr size_t MAX_WRITE_ATTEMPTS = 3;
    for (size_t attempt = 0; attempt < MAX_WRITE_ATTEMPTS; ++attempt) {
        if (writeFile(path, data)) return true;
    }
    return false;
}

std::optional<std::vector<uint8_t>> SaveManager::readFile(const std::string& path) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return std::nullopt;
    const auto size = file.tellg();
    if (size <= 0) return std::nullopt;
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return file.good() ? std::optional<std::vector<uint8_t>>(std::move(data)) : std::nullopt;
}
