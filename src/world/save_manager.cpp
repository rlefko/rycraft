#include "world/save_manager.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

SaveManager::SaveManager(const std::string& worldPath)
    : worldPath_(worldPath)
    , regionsPath_(worldPath + "/regions")
    , metadataPath_(worldPath + "/metadata.json")
{
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
    // Serialize chunk
    std::vector<uint8_t> serialized = ChunkSerializer::serialize(chunk);

    // Compress with LZ4
    std::vector<uint8_t> compressed = compress(serialized);

    // Determine region file path
    std::string regionPath = getRegionPath(chunk.chunkX, chunk.chunkZ);

    // Create save job
    SaveJob job;
    job.regionFile = regionPath;
    job.compressedData = std::move(compressed);
    job.chunkIndex = getChunkIndexInRegion(chunk.chunkX, chunk.chunkZ);

    // Queue the job
    {
        std::lock_guard<std::mutex> lock(saveMutex_);
        saveQueue_.push(std::move(job));
    }

    saveCondition_.notify_one();
}

std::optional<Chunk> SaveManager::loadChunk(int chunkX, int chunkZ) const {
    std::string regionPath = getRegionPath(chunkX, chunkZ);

    // Read file
    auto fileData = readFile(regionPath);
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

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
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
    while (spawnXPos < content.size() && (content[spawnXPos] == ' ' || content[spawnXPos] == '\n')) {
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
    while (spawnYPos < content.size() && (content[spawnYPos] == ' ' || content[spawnYPos] == '\n')) {
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
    while (spawnZPos < content.size() && (content[spawnZPos] == ' ' || content[spawnZPos] == '\n')) {
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
    // Wait until save queue is empty
    std::unique_lock<std::mutex> lock(saveMutex_);
    saveCondition_.wait(lock, [this]() {
        return saveQueue_.empty();
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
            saveCondition_.wait(lock, [this]() {
                return !saveQueue_.empty() || !running_.load();
            });

            if (!running_.load() && saveQueue_.empty()) {
                break;
            }

            if (saveQueue_.empty()) {
                continue;
            }

            job = std::move(saveQueue_.front());
            saveQueue_.pop();
        }

        // Write to file
        writeFile(job.regionFile, job.compressedData);

        // Notify flush waiters
        saveCondition_.notify_all();
    }
}

std::vector<uint8_t> SaveManager::compress(const std::vector<uint8_t>& data) const {
    // Calculate max compressed size
    int maxCompressedSize = LZ4_COMPRESSBOUND(static_cast<int>(data.size()));
    std::vector<uint8_t> compressed(maxCompressedSize);

    int compressedSize = LZ4_compress_default(
        reinterpret_cast<const char*>(data.data()),
        reinterpret_cast<char*>(compressed.data()),
        static_cast<int>(data.size()),
        maxCompressedSize
    );

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

std::vector<uint8_t> SaveManager::decompress(const std::vector<uint8_t>& data, size_t maxDecompressedSize) const {
    if (data.empty()) {
        return {};
    }

    std::vector<uint8_t> decompressed(maxDecompressedSize);

    int decompressedSize = LZ4_decompress_safe(
        reinterpret_cast<const char*>(data.data()),
        reinterpret_cast<char*>(decompressed.data()),
        static_cast<int>(data.size()),
        static_cast<int>(maxDecompressedSize)
    );

    if (decompressedSize <= 0) {
        // Decompression failed, return empty
        return {};
    }

    decompressed.resize(static_cast<size_t>(decompressedSize));
    return decompressed;
}

std::string SaveManager::getRegionPath(int chunkX, int chunkZ) const {
    int regionX = getRegionCoord(chunkX);
    int regionZ = getRegionCoord(chunkZ);

    std::ostringstream oss;
    oss << regionsPath_ << "/r." << regionX << "." << regionZ << ".dat";
    return oss.str();
}

int SaveManager::getRegionCoord(int chunkCoord) {
    // Region is 32x32 chunks (Minecraft convention)
    // Using floor division for negative coordinates
    return (chunkCoord >= 0) ? (chunkCoord / 32) : ((chunkCoord - 31) / 32);
}

int SaveManager::getChunkIndexInRegion(int chunkX, int chunkZ) {
    int localX = (chunkX % 32 + 32) % 32;
    int localZ = (chunkZ % 32 + 32) % 32;
    return localZ * 32 + localX;
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
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    file.close();
    return true;
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
