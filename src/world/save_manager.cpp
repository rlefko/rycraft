#include "world/save_manager.hpp"

#include "world/chunk_pos.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <lz4.h>

namespace fs = std::filesystem;

SaveManager::SaveManager(const std::string& worldPath)
    : worldPath_(worldPath)
    , regionsPath_(worldPath + "/regions")
    , metadataPath_(worldPath + "/metadata.json") {
    // Ensure directories exist
    ensureDirectory(worldPath_);
    ensureDirectory(regionsPath_);

    // Start background save thread
    saveThread_ = std::thread(&SaveManager::saveLoop, this);
}

SaveManager::~SaveManager() {
    running_.store(false);
    saveCondition_.notify_one();
    if (saveThread_.joinable()) {
        saveThread_.join();
    }
}

void SaveManager::saveChunk(const Chunk& chunk) {
    saveChunkAsync(std::make_shared<const Chunk>(chunk));
}

void SaveManager::saveChunkAsync(std::shared_ptr<const Chunk> chunk) {
    uint64_t key = packChunkKey(chunk->chunkX, chunk->chunkZ);
    {
        std::lock_guard<std::mutex> lock(saveMutex_);
        // A newer snapshot of the same chunk simply replaces the shield
        // entry; both queued jobs still write (last write wins on disk)
        pendingChunks_[key] = chunk;
        saveQueue_.push(SaveJob{std::move(chunk)});
    }

    pendingWrites_.fetch_add(1, std::memory_order_acq_rel);
    saveCondition_.notify_one();
}

std::optional<Chunk> SaveManager::loadChunk(int chunkX, int chunkZ) const {
    // Load shield: a chunk that was just unloaded may still sit in the save
    // queue — walking back toward it must see the queued edits, not the
    // stale file on disk.
    {
        std::lock_guard<std::mutex> lock(saveMutex_);
        auto it = pendingChunks_.find(packChunkKey(chunkX, chunkZ));
        if (it != pendingChunks_.end()) {
            return Chunk(*it->second);
        }
    }

    // Read file
    auto fileData = readFile(getChunkPath(chunkX, chunkZ));
    if (!fileData.has_value()) {
        return std::nullopt;
    }

    // Decompress
    // Max decompressed size: serialize one chunk
    size_t maxDecompressed = ChunkSerializer::serializedSize(Chunk{0, 0});
    std::vector<uint8_t> decompressed = decompress(fileData.value(), maxDecompressed);

    if (decompressed.empty()) {
        return std::nullopt;
    }

    // Deserialize
    auto chunk = ChunkSerializer::deserialize(decompressed);
    if (!chunk.has_value()) {
        return std::nullopt;
    }

    return chunk;
}

void SaveManager::saveMetadata(uint32_t seed, Vec3 spawnPos, uint64_t worldTime) {
    // Simple JSON-like format
    std::ostringstream json;
    json << "{\n"
         << "  \"seed\": " << seed << ",\n"
         << "  \"spawnPos\": {\n"
         << "    \"x\": " << spawnPos.x << ",\n"
         << "    \"y\": " << spawnPos.y << ",\n"
         << "    \"z\": " << spawnPos.z << "\n"
         << "  },\n"
         << "  \"worldTime\": " << worldTime << "\n"
         << "}\n";

    std::ofstream file(metadataPath_);
    if (!file.is_open()) {
        return;
    }

    file << json.str();
    file.close();
}

std::optional<SaveManager::WorldMetadata> SaveManager::loadMetadata() const {
    std::ifstream file(metadataPath_);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing (no external dependency)
    WorldMetadata metadata;

    // Parse seed
    auto seedPos = content.find("\"seed\":");
    if (seedPos == std::string::npos) {
        return std::nullopt;
    }
    seedPos += 7;
    while (seedPos < content.size() && (content[seedPos] == ' ' || content[seedPos] == '\n')) {
        ++seedPos;
    }
    try {
        metadata.seed = static_cast<uint32_t>(std::stoul(content.substr(seedPos)));
    } catch (...) {
        return std::nullopt;
    }

    // Parse spawnPos.x
    auto spawnXPos = content.find("\"x\":");
    if (spawnXPos == std::string::npos) {
        return std::nullopt;
    }
    spawnXPos += 4;
    while (spawnXPos < content.size() &&
           (content[spawnXPos] == ' ' || content[spawnXPos] == '\n')) {
        ++spawnXPos;
    }
    try {
        metadata.spawnPos.x = std::stof(content.substr(spawnXPos));
    } catch (...) {
        return std::nullopt;
    }

    // Parse spawnPos.y
    auto spawnYPos = content.find("\"y\":");
    if (spawnYPos == std::string::npos) {
        return std::nullopt;
    }
    spawnYPos += 4;
    while (spawnYPos < content.size() &&
           (content[spawnYPos] == ' ' || content[spawnYPos] == '\n')) {
        ++spawnYPos;
    }
    try {
        metadata.spawnPos.y = std::stof(content.substr(spawnYPos));
    } catch (...) {
        return std::nullopt;
    }

    // Parse spawnPos.z
    auto spawnZPos = content.find("\"z\":");
    if (spawnZPos == std::string::npos) {
        return std::nullopt;
    }
    spawnZPos += 4;
    while (spawnZPos < content.size() &&
           (content[spawnZPos] == ' ' || content[spawnZPos] == '\n')) {
        ++spawnZPos;
    }
    try {
        metadata.spawnPos.z = std::stof(content.substr(spawnZPos));
    } catch (...) {
        return std::nullopt;
    }

    // Parse worldTime
    auto timePos = content.find("\"worldTime\":");
    if (timePos == std::string::npos) {
        return std::nullopt;
    }
    timePos += 12;
    while (timePos < content.size() && (content[timePos] == ' ' || content[timePos] == '\n')) {
        ++timePos;
    }
    try {
        metadata.worldTime = static_cast<uint64_t>(std::stoull(content.substr(timePos)));
    } catch (...) {
        return std::nullopt;
    }

    return metadata;
}

void SaveManager::flush() {
    // Wait until all queued saves are written to disk.
    // Two conditions: queue is empty (no pending jobs) AND
    // pendingWrites is 0 (all in-flight writes completed).
    std::unique_lock<std::mutex> lock(saveMutex_);
    saveCondition_.wait(lock, [this]() {
        return saveQueue_.empty() && pendingWrites_.load(std::memory_order_acquire) == 0;
    });
}

const std::string& SaveManager::getWorldPath() const {
    return worldPath_;
}

void SaveManager::saveLoop() {
    while (running_.load()) {
        SaveJob job;
        {
            std::unique_lock<std::mutex> lock(saveMutex_);
            saveCondition_.wait(lock, [this]() { return !saveQueue_.empty() || !running_.load(); });

            if (!running_.load() && saveQueue_.empty()) {
                break;
            }

            if (saveQueue_.empty()) {
                continue;
            }

            job = std::move(saveQueue_.front());
            saveQueue_.pop();
        }

        // Serialize + compress here, off the game threads
        std::vector<uint8_t> compressed = compress(ChunkSerializer::serialize(*job.chunk));
        std::string path = getChunkPath(job.chunk->chunkX, job.chunk->chunkZ);
        ensureDirectory(getChunkDir(job.chunk->chunkX, job.chunk->chunkZ));
        writeFile(path, compressed);

        // Drop the load shield only if no newer snapshot replaced ours
        {
            std::lock_guard<std::mutex> lock(saveMutex_);
            uint64_t key = packChunkKey(job.chunk->chunkX, job.chunk->chunkZ);
            auto it = pendingChunks_.find(key);
            if (it != pendingChunks_.end() && it->second == job.chunk) {
                pendingChunks_.erase(it);
            }
        }

        // Mark write as complete
        pendingWrites_.fetch_sub(1, std::memory_order_release);

        // Notify flush waiters
        saveCondition_.notify_all();
    }
}

std::vector<uint8_t> SaveManager::compress(const std::vector<uint8_t>& data) const {
    // Calculate max compressed size
    int maxCompressedSize = LZ4_COMPRESSBOUND(static_cast<int>(data.size()));
    std::vector<uint8_t> compressed(maxCompressedSize);

    int compressedSize = LZ4_compress_default(reinterpret_cast<const char*>(data.data()),
                                              reinterpret_cast<char*>(compressed.data()),
                                              static_cast<int>(data.size()), maxCompressedSize);

    if (compressedSize <= 0) {
        // Compression failed, return uncompressed with size prefix
        std::vector<uint8_t> fallback(data.size() + sizeof(int32_t));
        int32_t dataSize = static_cast<int32_t>(data.size());
        std::memcpy(fallback.data(), &dataSize, sizeof(int32_t));
        std::memcpy(fallback.data() + sizeof(int32_t), data.data(), data.size());
        return fallback;
    }

    compressed.resize(static_cast<size_t>(compressedSize));
    return compressed;
}

std::vector<uint8_t> SaveManager::decompress(const std::vector<uint8_t>& data,
                                             size_t maxDecompressedSize) const {
    if (data.empty()) {
        return {};
    }

    std::vector<uint8_t> decompressed(maxDecompressedSize);

    int decompressedSize = LZ4_decompress_safe(
        reinterpret_cast<const char*>(data.data()), reinterpret_cast<char*>(decompressed.data()),
        static_cast<int>(data.size()), static_cast<int>(maxDecompressedSize));

    if (decompressedSize <= 0) {
        // Decompression failed, return empty
        return {};
    }

    decompressed.resize(static_cast<size_t>(decompressedSize));
    return decompressed;
}

std::string SaveManager::getChunkDir(int chunkX, int chunkZ) const {
    std::ostringstream oss;
    oss << regionsPath_ << "/r." << getRegionCoord(chunkX) << "." << getRegionCoord(chunkZ);
    return oss.str();
}

std::string SaveManager::getChunkPath(int chunkX, int chunkZ) const {
    std::ostringstream oss;
    oss << getChunkDir(chunkX, chunkZ) << "/c." << chunkX << "." << chunkZ << ".dat";
    return oss.str();
}

int SaveManager::getRegionCoord(int chunkCoord) {
    // Region is 32x32 chunks (Minecraft convention)
    // Using floor division for negative coordinates
    return (chunkCoord >= 0) ? (chunkCoord / 32) : ((chunkCoord - 31) / 32);
}

uint64_t SaveManager::packChunkKey(int chunkX, int chunkZ) {
    return ChunkPos{chunkX, chunkZ}.packed(); // THE xz key bit layout
}

bool SaveManager::ensureDirectory(const std::string& path) const {
    try {
        fs::create_directories(path);
        return true;
    } catch (const fs::filesystem_error&) {
        return false;
    }
}

bool SaveManager::writeFile(const std::string& path, const std::vector<uint8_t>& data) const {
    // Write-temp-then-rename: a crash mid-write can never leave a truncated
    // chunk file behind (rename is atomic on APFS)
    std::string tempPath = path + ".tmp";
    {
        std::ofstream file(tempPath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size()));
        if (!file.good()) {
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tempPath, path, ec);
    return !ec;
}

std::optional<std::vector<uint8_t>> SaveManager::readFile(const std::string& path) const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return std::nullopt;
    }

    auto fileSize = file.tellg();
    if (fileSize <= 0) {
        return std::nullopt;
    }

    std::vector<uint8_t> data(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    return data;
}
