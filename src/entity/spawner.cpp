#include "entity/spawner.hpp"
#include "entity/ai.hpp"
#include "entity/entity.hpp"
#include "entity/physics.hpp"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Spawner constructor
// ---------------------------------------------------------------------------
Spawner::Spawner(World& world) : world_(world), rng_(static_cast<uint64_t>(world.getSeed())) {}

// ---------------------------------------------------------------------------
// getSpawnRule — Biome → entity spawn counts
// ---------------------------------------------------------------------------
BiomeSpawnRule Spawner::getSpawnRule(Biome biome) {
    switch (biome) {
        case Biome::PLAINS:
            return {8, 4, 2, 0};
        case Biome::FOREST:
            return {4, 4, 4, 0};
        case Biome::DESERT:
            return {0, 0, 0, 0};
        case Biome::OCEAN:
        case Biome::DEEP_OCEAN:
            return {0, 0, 0, 0};
        case Biome::TAIGA:
            return {6, 2, 0, 0};
        case Biome::SWAMP:
            return {0, 2, 2, 0};
        case Biome::EXTREME_HILLS:
            return {2, 2, 1, 0};
        case Biome::MUSHROOM_ISLAND:
            return {4, 0, 0, 0};
        case Biome::ICE_SPIKES:
            return {2, 1, 0, 0};
        default:
            return {0, 0, 0, 0};
    }
}

// ---------------------------------------------------------------------------
// findSpawnHeight — Find surface Y at world coordinates
// ---------------------------------------------------------------------------
std::optional<int> Spawner::findSpawnHeight(int x, int z) {
    // Scan from top to bottom for the first solid block
    for (int y = 255; y >= 0; --y) {
        BlockType block = world_.getBlock(x, y, z);
        if (isSolid(block)) {
            return y + 1; // Spawn on top of solid block
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// isSpawnValid — Check spawn position is safe
// ---------------------------------------------------------------------------
bool Spawner::isSpawnValid(int x, int y, int z) {
    // Block at spawn Y must be AIR (entity can stand here)
    // Block at spawn Y+1 must be AIR (headroom)
    // Block at spawn Y-1 must be solid (ground)
    BlockType atY = world_.getBlock(x, y, z);
    BlockType aboveY = world_.getBlock(x, y + 1, z);
    BlockType belowY = world_.getBlock(x, y - 1, z);

    if (atY != BlockType::AIR) return false;
    if (aboveY != BlockType::AIR) return false;
    if (belowY == BlockType::AIR) return false;

    return true;
}

// ---------------------------------------------------------------------------
// randomInt — Random number in [min, max]
// ---------------------------------------------------------------------------
int Spawner::randomInt(int min, int max) {
    if (min > max) std::swap(min, max);
    return rng_.nextInt(min, max);
}

// ---------------------------------------------------------------------------
// spawnEntity — Create and register a new entity
// ---------------------------------------------------------------------------
std::shared_ptr<Entity> Spawner::spawnEntity(EntityType type, const Vec3& position) {
    auto entity = std::make_shared<Entity>(Entity::nextId(), type, position);
    entities_.push_back(entity);
    spatialHash_.insert(entity->id, entity->position);
    return entity;
}

// ---------------------------------------------------------------------------
// spawnBaby — Create a baby entity
// ---------------------------------------------------------------------------
std::shared_ptr<Entity> Spawner::spawnBaby(EntityType type, const Vec3& position,
                                           uint64_t parentId) {
    auto baby = std::make_shared<Entity>(Entity::nextId(), type, position);
    baby->isBaby = true;
    baby->babyTimer = 600; // 600 ticks = 30 seconds
    baby->parentId = parentId;
    entities_.push_back(baby);
    spatialHash_.insert(baby->id, baby->position);
    return baby;
}

// ---------------------------------------------------------------------------
// spawnForChunk — Spawn entities for a single chunk
// ---------------------------------------------------------------------------
void Spawner::spawnForChunk(int chunkX, int chunkZ) {
    // Get biome for this chunk center
    int centerX = chunkX * CHUNK_WIDTH + 8;
    int centerZ = chunkZ * CHUNK_DEPTH + 8;
    Biome biome = world_.getBiome(centerX, centerZ);

    BiomeSpawnRule rule = getSpawnRule(biome);

    // Try to spawn each type
    auto trySpawn = [&](EntityType type, int count) {
        for (int i = 0; i < count; ++i) {
            // Random position within chunk
            int localX = randomInt(2, CHUNK_WIDTH - 3);
            int localZ = randomInt(2, CHUNK_DEPTH - 3);
            int worldX = chunkX * CHUNK_WIDTH + localX;
            int worldZ = chunkZ * CHUNK_DEPTH + localZ;

            // Find surface height
            auto height = findSpawnHeight(worldX, worldZ);
            if (!height.has_value()) continue;

            // Validate spawn
            if (!isSpawnValid(worldX, height.value(), worldZ)) continue;

            // Spawn entity
            Vec3 spawnPos{static_cast<float>(worldX) + 0.5f, static_cast<float>(height.value()),
                          static_cast<float>(worldZ) + 0.5f};
            spawnEntity(type, spawnPos);
        }
    };

    trySpawn(EntityType::SHEEP, rule.sheepCount);
    trySpawn(EntityType::COW, rule.cowCount);
    trySpawn(EntityType::PIG, rule.pigCount);
    trySpawn(EntityType::CHICKEN, rule.chickenCount);
}

// ---------------------------------------------------------------------------
// spawnInitialPopulation — Spawn entities for all loaded chunks
// ---------------------------------------------------------------------------
void Spawner::spawnInitialPopulation() {
    auto chunks = world_.getLoadedChunks();
    for (const auto& chunk : chunks) {
        spawnForChunk(chunk->chunkX, chunk->chunkZ);
    }
}

// ---------------------------------------------------------------------------
// getEntities — Access managed entities
// ---------------------------------------------------------------------------
std::vector<std::shared_ptr<Entity>>& Spawner::getEntities() {
    return entities_;
}

// ---------------------------------------------------------------------------
// getEntity — Lookup entity by ID
// ---------------------------------------------------------------------------
Entity* Spawner::getEntity(uint64_t entityId) {
    for (auto& e : entities_) {
        if (e->id == entityId) return e.get();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// getEntityPositions — Build position map for spatial hash queries
// ---------------------------------------------------------------------------
std::unordered_map<uint64_t, Vec3> Spawner::getEntityPositions() const {
    std::unordered_map<uint64_t, Vec3> result;
    for (const auto& e : entities_) {
        result[e->id] = e->position;
    }
    return result;
}

// ---------------------------------------------------------------------------
// removeEntity — Remove dead entity
// ---------------------------------------------------------------------------
void Spawner::removeEntity(uint64_t entityId) {
    spatialHash_.remove(entityId);
    entities_.erase(std::remove_if(entities_.begin(), entities_.end(),
                                   [entityId](const std::shared_ptr<Entity>& e) {
                                       return e->id == entityId && !e->alive;
                                   }),
                    entities_.end());
}

// ---------------------------------------------------------------------------
// getSpatialHash — Access spatial hash
// ---------------------------------------------------------------------------
SpatialHash& Spawner::getSpatialHash() {
    return spatialHash_;
}
