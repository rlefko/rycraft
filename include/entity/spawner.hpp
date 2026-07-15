#pragma once

#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/spatial_hash.hpp>
#include <world/chunk.hpp>
#include <world/world.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// BiomeSpawnRule — Entity spawn counts per biome
// ---------------------------------------------------------------------------
struct BiomeSpawnRule {
    int sheepCount = 0;
    int cowCount = 0;
    int pigCount = 0;
    int chickenCount = 0;
    int deerCount = 0;
    int goatCount = 0;
    int rabbitCount = 0;
    int frogCount = 0;
    int fishCount = 0;
};

// ---------------------------------------------------------------------------
// HabitatSample - Continuous current-world inputs for ecological carrying
// capacity. Biome is retained for diagnostics and compatibility only. Wild
// species selection does not branch or weight by biome.
// ---------------------------------------------------------------------------
struct HabitatSample {
    Biome biome = Biome::PLAINS;
    float temperature = 0.5f;
    float moisture = 0.5f;
    float food = 0.5f;
    float cover = 0.0f;
    float fertility = 0.5f;
    float slope = 0.0f;
    float riverSize = 0.0f;
    float waterAccess = 0.0f;
    int surfaceY = SEA_LEVEL;
    int waterDepth = 0;
    bool nearWater = false;
    // Hand-authored samples historically represented natural water. Runtime
    // sampling always overwrites these flags from generated hydrology.
    bool generatedWaterBody = true;
    bool generatedRiver = false;
    bool generatedLake = false;
    bool generatedOcean = false;
};

// ---------------------------------------------------------------------------
// Spawner — Manages entity lifecycle and population
//
// On world load, scans chunks and spawns entities based on biome density.
// Handles surface-level spawn validation and entity creation.
// ---------------------------------------------------------------------------
class Spawner {
public:
    static constexpr size_t MAX_ANIMALS = 64;
    static constexpr float ACTIVE_RADIUS = 96.0f;
    static constexpr float DESPAWN_RADIUS = 112.0f;
    static constexpr int TERRITORY_SIZE = 64;
    static constexpr int POPULATION_UPDATE_TICKS = 20;

    explicit Spawner(World& world);

    // Spawn entities for a single chunk based on its biome
    void spawnForChunk(int64_t chunkX, int64_t chunkZ);

    // Spawn all entities for loaded chunks (initial population)
    void spawnInitialPopulation();

    // Reevaluate deterministic habitat territories around the player. Safe
    // to call every simulation tick; expensive work runs once per second or
    // after the player moves at least half a territory.
    void updatePopulation(const Vec3& playerPosition);

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
    std::optional<int> findSpawnHeight(int64_t x, int64_t z);

    // Check if spawn position is valid (no solid block at spawn+1) (exposed for testing)
    bool isSpawnValid(int64_t x, int y, int64_t z);

    // Spawn a single entity at a position (exposed for testing)
    std::shared_ptr<Entity> spawnEntity(EntityType type, const Vec3& position);

    // Current living population. Babies and adults both count toward the
    // same hard limit.
    size_t livingCount() const;

    // Pure ecology helpers exposed for deterministic tests and diagnostics.
    static float getHabitatScore(EntityType type, const HabitatSample& habitat);
    static int getCarryingCapacity(EntityType type, const HabitatSample& habitat);
    static Vec2 getTerritoryAnchor(uint32_t worldSeed, int64_t territoryX, int64_t territoryZ);
    static uint64_t makeTerritoryEntityId(uint32_t worldSeed, int64_t territoryX,
                                          int64_t territoryZ, EntityType type, int memberIndex);
    static bool shouldDespawn(const Vec3& entityPosition, const Vec3& playerPosition);

    // Return habitat only when its immutable column plan is already cached.
    // This never constructs a basin or column plan on the simulation thread.
    std::optional<HabitatSample> findHabitatSample(int64_t x, int64_t z);

    // Exposed for runtime diagnostics and territory retry regression tests.
    size_t populatedTerritoryCount() const { return populatedTerritories_.size(); }

    // Find a submerged spawn cell for aquatic entities.
    std::optional<int> findWaterSpawnHeight(int64_t x, int64_t z);

private:
    struct TerritoryPos {
        int64_t x;
        int64_t z;

        bool operator==(const TerritoryPos&) const = default;
    };

    struct TerritoryPosHash {
        size_t operator()(const TerritoryPos& pos) const;
    };

    World& world_;
    std::vector<std::shared_ptr<Entity>> entities_;
    SpatialHash spatialHash_;
    std::unordered_set<TerritoryPos, TerritoryPosHash> populatedTerritories_;
    uint32_t populationTick_ = 0;
    Vec3 lastPopulationCenter_ = Vec3::zero();
    bool hasPopulationCenter_ = false;

    static int64_t floorDiv(int64_t value, int64_t divisor);
    static uint64_t territoryHash(uint32_t worldSeed, int64_t territoryX, int64_t territoryZ,
                                  uint64_t stream);

    std::optional<EntityType> selectSpecies(const HabitatSample& habitat, uint64_t roll) const;
    void spawnTerritory(TerritoryPos territory);
    std::shared_ptr<Entity> spawnWildEntity(EntityType type, const Vec3& position,
                                            TerritoryPos territory, int memberIndex);
    void despawnDistantWildlife(const Vec3& playerPosition);
    void pruneTerritories(const Vec3& playerPosition);
};
