#include "world/save_manager.hpp"

#include "common/error.hpp"

#include <algorithm>
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

bool parseInventory(const std::string& content,
                    std::array<BlockType, SaveManager::PLAYER_INVENTORY_SLOTS>& inventory) {
    size_t position = content.find("\"inventory\"");
    if (position == std::string::npos) return false;
    position = content.find('[', position);
    if (position == std::string::npos) return false;
    ++position;
    auto parsed = inventory;
    for (size_t slot = 0; slot < parsed.size(); ++slot) {
        while (
            position < content.size() &&
            (content[position] == ' ' || content[position] == '\n' || content[position] == ',')) {
            ++position;
        }
        if (position >= content.size()) return false;
        size_t consumed = 0;
        unsigned long value = 0;
        try {
            value = std::stoul(content.substr(position), &consumed);
        } catch (...) {
            return false;
        }
        if (consumed == 0 || value >= static_cast<unsigned long>(BlockType::COUNT)) return false;
        parsed[slot] = static_cast<BlockType>(value);
        position += consumed;
    }
    while (position < content.size() && (content[position] == ' ' || content[position] == '\n')) {
        ++position;
    }
    if (position >= content.size() || content[position] != ']') return false;
    inventory = parsed;
    return true;
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
    : worldPath_(worldPath)
    , regionsPath_(worldPath + "/" + CURRENT_REGIONS_DIRECTORY)
    , metadataPath_(worldPath + "/metadata.json")
    , testHooks_(std::move(testHooks)) {
    ensureDirectory(worldPath_);
    ensureDirectory(regionsPath_);
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
    std::ostringstream json;
    json << "{\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"spawnPos\": {\n"
         << "    \"x\": " << spawnPos.x << ",\n"
         << "    \"y\": " << spawnPos.y << ",\n"
         << "    \"z\": " << spawnPos.z << "\n"
         << "  },\n"
         << "  \"worldTime\": " << worldTime << ",\n"
         << "  \"player\": {\n"
         << "    \"yaw\": " << player.yaw << ",\n"
         << "    \"pitch\": " << player.pitch << ",\n"
         << "    \"health\": " << player.health << ",\n"
         << "    \"selectedSlot\": " << player.selectedSlot << ",\n"
         << "    \"inventory\": [";
    for (size_t slot = 0; slot < player.inventory.size(); ++slot) {
        if (slot != 0) json << ", ";
        json << static_cast<unsigned>(player.inventory[slot]);
    }
    json << "]\n"
         << "  },\n"
         << "  \"chunkFormatVersion\": " << CHUNK_VERSION << ",\n"
         << "  \"generatorVersion\": " << CURRENT_GENERATOR_VERSION << "\n"
         << "}\n";
    const std::string text = json.str();
    return writeFileWithRetries(metadataPath_, std::vector<uint8_t>(text.begin(), text.end()));
}

std::optional<SaveManager::WorldMetadata> SaveManager::loadMetadata() const {
    std::ifstream file(metadataPath_);
    if (!file.is_open()) return std::nullopt;
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    WorldMetadata metadata;
    if (!parseNumber(content, "\"seed\":", metadata.seed,
                     [](const std::string& value) { return std::stoul(value); }) ||
        !parseNumber(content, "\"x\":", metadata.spawnPos.x,
                     [](const std::string& value) { return std::stof(value); }) ||
        !parseNumber(content, "\"y\":", metadata.spawnPos.y,
                     [](const std::string& value) { return std::stof(value); }) ||
        !parseNumber(content, "\"z\":", metadata.spawnPos.z,
                     [](const std::string& value) { return std::stof(value); }) ||
        !parseNumber(content, "\"worldTime\":", metadata.worldTime,
                     [](const std::string& value) { return std::stoull(value); })) {
        return std::nullopt;
    }
    parseNumber(content, "\"chunkFormatVersion\":", metadata.chunkFormatVersion,
                [](const std::string& value) { return std::stoul(value); });
    parseNumber(content, "\"generatorVersion\":", metadata.generatorVersion,
                [](const std::string& value) { return std::stoul(value); });
    parseNumber(content, "\"yaw\":", metadata.player.yaw,
                [](const std::string& value) { return std::stof(value); });
    parseNumber(content, "\"pitch\":", metadata.player.pitch,
                [](const std::string& value) { return std::stof(value); });
    parseNumber(content, "\"health\":", metadata.player.health,
                [](const std::string& value) { return std::stoi(value); });
    parseNumber(content, "\"selectedSlot\":", metadata.player.selectedSlot,
                [](const std::string& value) { return std::stoi(value); });
    metadata.player.health = std::clamp(metadata.player.health, 0, 20);
    metadata.player.selectedSlot =
        std::clamp(metadata.player.selectedSlot, 0, static_cast<int>(PLAYER_INVENTORY_SLOTS) - 1);
    parseInventory(content, metadata.player.inventory);
    return metadata;
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
