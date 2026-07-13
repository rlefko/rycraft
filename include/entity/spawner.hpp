#pragma once

#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/spatial_hash.hpp>
#include <world/biome.hpp>
#include <world/world.hpp>

#include "common/random.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// BiomeSpawnRule — Entity spawn counts per biome
// ---------------------------------------------------------------------------
struct BiomeSpawnRule {
    int sheepCount = 0;
    int cowCount = 0;
    int pigCount = 0;
    int chickenCount = 0;
};

// ---------------------------------------------------------------------------
// Spawner — Manages entity lifecycle and population
//
// On world load, scans chunks and spawns entities based on biome density.
// Handles surface-level spawn validation and entity creation.
// ---------------------------------------------------------------------------
class Spawner {
public:
    explicit Spawner(World& world);

    // Spawn entities for a single chunk based on its biome
    void spawnForChunk(int chunkX, int chunkZ);

    // Spawn all entities for loaded chunks (initial population)
    void spawnInitialPopulation();

    // Get all managed entities
    std::vector<std::shared_ptr<Entity>>& getEntities();

    // Get entity by ID (returns nullptr if not found)
    Entity* getEntity(uint64_t entityId);

    // Get entity positions map for spatial hash queries
    std::unordered_map<uint64_t, Vec3> getEntityPositions() const;

    // Remove a dead entity by ID
    void removeEntity(uint64_t entityId);

    // Spawn a baby entity (called from breed behavior)
    std::shared_ptr<Entity> spawnBaby(EntityType type, const Vec3& position, uint64_t parentId);

    // Get spatial hash for neighbor queries
    SpatialHash& getSpatialHash();

    // Get spawn density for a biome (exposed for testing)
    static BiomeSpawnRule getSpawnRule(Biome biome);

    // Find a valid spawn Y at world coordinates (surface level) (exposed for testing)
    std::optional<int> findSpawnHeight(int x, int z);

    // Check if spawn position is valid (no solid block at spawn+1) (exposed for testing)
    bool isSpawnValid(int x, int y, int z);

    // Spawn a single entity at a position (exposed for testing)
    std::shared_ptr<Entity> spawnEntity(EntityType type, const Vec3& position);

private:
    World& world_;
    std::vector<std::shared_ptr<Entity>> entities_;
    SpatialHash spatialHash_;
    SeededRng rng_;

    // Random number in [min, max]
    int randomInt(int min, int max);
};
