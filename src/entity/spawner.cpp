#include "entity/spawner.hpp"

#include "common/random.hpp"
#include "entity/entity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>

namespace {

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float preference(float value, float optimum, float tolerance) {
    return clamp01(1.0f - std::abs(value - optimum) / tolerance);
}

float waterAccess(const HabitatSample& habitat) {
    return habitat.waterAccess > 0.0f ? habitat.waterAccess : (habitat.nearWater ? 1.0f : 0.0f);
}

bool generatedWaterBody(const worldgen::SurfaceSample& surface) {
    return surface.hydrology.river || surface.hydrology.lake || surface.hydrology.ocean;
}

bool suitableGround(EntityType type, BlockType ground) {
    if (!isSolid(ground) || isLeafBlock(ground) || ground == BlockType::GLASS ||
        ground == BlockType::CACTUS) {
        return false;
    }

    switch (type) {
        case EntityType::GOAT:
            return ground == BlockType::STONE || ground == BlockType::GRAVEL ||
                   ground == BlockType::SNOW || ground == BlockType::GRASS;
        case EntityType::RABBIT:
            return ground == BlockType::GRASS || ground == BlockType::DIRT ||
                   ground == BlockType::SAND || ground == BlockType::SNOW;
        case EntityType::FROG:
            return ground == BlockType::GRASS || ground == BlockType::DIRT ||
                   ground == BlockType::SAND;
        case EntityType::FISH:
            return false;
        case EntityType::SHEEP:
        case EntityType::COW:
        case EntityType::PIG:
        case EntityType::CHICKEN:
        case EntityType::DEER:
            return ground == BlockType::GRASS || ground == BlockType::DIRT ||
                   ground == BlockType::SNOW || ground == BlockType::SAND;
        case EntityType::COUNT:
            return false;
    }
}

} // namespace

Spawner::Spawner(World& world) : world_(world) {}

size_t Spawner::TerritoryPosHash::operator()(const TerritoryPos& pos) const {
    uint64_t xHash = hash64(static_cast<uint64_t>(pos.x));
    uint64_t zHash = hash64(static_cast<uint64_t>(pos.z));
    return static_cast<size_t>(hash64(xHash ^ (zHash + 0x9E3779B97F4A7C15ULL)));
}

int64_t Spawner::floorDiv(int64_t value, int64_t divisor) {
    return world_coord::floorDiv(value, divisor);
}

uint64_t Spawner::territoryHash(uint64_t worldSeed, int64_t territoryX, int64_t territoryZ,
                                uint64_t stream) {
    uint64_t xHash = hash64(static_cast<uint64_t>(territoryX));
    uint64_t zHash = hash64(static_cast<uint64_t>(territoryZ));
    return hash64(xHash ^ (zHash + 0x9E3779B97F4A7C15ULL) ^
                  hash64(static_cast<uint64_t>(worldSeed)) ^ hash64(stream));
}

Vec2 Spawner::getTerritoryAnchor(uint64_t worldSeed, int64_t territoryX, int64_t territoryZ) {
    uint64_t hash = territoryHash(worldSeed, territoryX, territoryZ, 0x414E43484F52ULL);
    int offsetX = 8 + static_cast<int>(hash % 48ULL);
    int offsetZ = 8 + static_cast<int>((hash >> 16) % 48ULL);
    return {static_cast<float>(territoryX * TERRITORY_SIZE + offsetX),
            static_cast<float>(territoryZ * TERRITORY_SIZE + offsetZ)};
}

uint64_t Spawner::makeTerritoryEntityId(uint64_t worldSeed, int64_t territoryX, int64_t territoryZ,
                                        EntityType type, int memberIndex) {
    uint64_t stream = 0x57494C444C494645ULL;
    stream ^= static_cast<uint64_t>(static_cast<uint8_t>(type)) << 32;
    stream ^= static_cast<uint32_t>(memberIndex);
    return territoryHash(worldSeed, territoryX, territoryZ, stream) | (1ULL << 63);
}

BiomeSpawnRule Spawner::getSpawnRule(Biome biome) {
    switch (biome) {
        case Biome::PLAINS:
            return {8, 4, 2, 0, 1, 0, 6, 0, 0};
        case Biome::FOREST:
            return {4, 4, 4, 2, 7, 0, 2, 0, 0};
        case Biome::DESERT:
            return {0, 0, 0, 0, 0, 0, 4, 0, 0};
        case Biome::OCEAN:
            return {0, 0, 0, 0, 0, 0, 0, 0, 6};
        case Biome::DEEP_OCEAN:
            return {0, 0, 0, 0, 0, 0, 0, 0, 8};
        case Biome::TAIGA:
            return {6, 2, 0, 1, 5, 1, 2, 0, 0};
        case Biome::SWAMP:
            return {0, 2, 2, 1, 1, 0, 1, 8, 2};
        case Biome::EXTREME_HILLS:
            return {2, 2, 1, 0, 2, 8, 2, 0, 0};
        case Biome::MUSHROOM_ISLAND:
            return {4, 0, 0, 0, 1, 0, 2, 0, 0};
        case Biome::ICE_SPIKES:
            return {2, 1, 0, 0, 1, 5, 2, 0, 0};
        case Biome::BEACH:
            return {0, 0, 0, 1, 0, 0, 4, 1, 2};
        case Biome::RIVER:
            return {1, 1, 1, 1, 2, 0, 2, 7, 7};
        case Biome::BIRCH_FOREST:
            return {4, 4, 4, 2, 8, 0, 3, 0, 0};
        case Biome::FLOWER_FIELD:
            return {8, 4, 2, 3, 2, 0, 8, 1, 0};
        case Biome::SAVANNA:
            return {5, 4, 1, 4, 2, 1, 5, 0, 0};
        case Biome::TROPICAL_RAINFOREST:
            return {0, 1, 6, 5, 4, 0, 2, 5, 0};
        case Biome::TEMPERATE_RAINFOREST:
            return {3, 4, 5, 3, 8, 0, 3, 4, 0};
        case Biome::SHRUBLAND:
            return {3, 2, 1, 2, 4, 2, 8, 0, 0};
        case Biome::STEPPE:
            return {7, 4, 1, 1, 3, 2, 8, 0, 0};
        case Biome::COLD_DESERT:
            return {1, 0, 0, 0, 1, 3, 5, 0, 0};
        case Biome::BADLANDS:
            return {0, 0, 0, 0, 1, 5, 4, 0, 0};
        case Biome::TUNDRA:
            return {4, 1, 0, 0, 2, 5, 5, 0, 0};
        case Biome::ALPINE:
            return {1, 0, 0, 0, 2, 8, 3, 0, 0};
        case Biome::MANGROVE:
            return {0, 1, 4, 3, 3, 0, 1, 8, 5};
        case Biome::FROZEN_OCEAN:
            return {0, 0, 0, 0, 0, 0, 0, 0, 3};
        case Biome::VOLCANIC_BARREN:
            return {0, 0, 0, 0, 0, 2, 0, 0, 0};
        case Biome::GLACIER:
            return {0, 0, 0, 0, 0, 1, 0, 0, 0};
        case Biome::MONTANE_GRASSLAND:
            return {4, 1, 0, 0, 3, 7, 4, 0, 0};
        case Biome::FLOODED_GRASSLAND:
            return {2, 4, 3, 2, 3, 0, 2, 7, 2};
        case Biome::MEDITERRANEAN_WOODLAND:
            return {3, 2, 2, 2, 6, 2, 6, 0, 0};
        case Biome::TEMPERATE_CONIFER_FOREST:
            return {4, 2, 2, 1, 7, 2, 2, 0, 0};
        case Biome::TROPICAL_CONIFER_FOREST:
            return {1, 2, 4, 4, 5, 2, 2, 2, 0};
        case Biome::TROPICAL_DRY_FOREST:
            return {2, 3, 2, 4, 5, 1, 4, 1, 0};
        case Biome::COUNT:
            return {};
    }
}

float Spawner::getHabitatScore(EntityType type, const HabitatSample& habitat) {
    const float water = waterAccess(habitat);
    const float lowSlope = 1.0f - clamp01(habitat.slope);
    const float openGround = 1.0f - clamp01(habitat.cover);
    const float temperate = preference(habitat.temperature, 0.55f, 0.55f);
    const float mildMoisture = preference(habitat.moisture, 0.50f, 0.55f);
    float score = 0.0f;

    switch (type) {
        case EntityType::SHEEP:
            score = habitat.food * 0.26f + habitat.fertility * 0.14f + openGround * 0.18f +
                    temperate * 0.15f + mildMoisture * 0.12f + lowSlope * 0.10f + water * 0.05f;
            break;
        case EntityType::COW:
            score = habitat.food * 0.25f + habitat.fertility * 0.18f + mildMoisture * 0.12f +
                    temperate * 0.12f + lowSlope * 0.12f + water * 0.10f +
                    preference(habitat.cover, 0.25f, 0.75f) * 0.11f;
            break;
        case EntityType::PIG:
            score = habitat.food * 0.18f + habitat.fertility * 0.16f + habitat.moisture * 0.20f +
                    habitat.cover * 0.14f + preference(habitat.temperature, 0.65f, 0.65f) * 0.12f +
                    lowSlope * 0.10f + water * 0.10f;
            break;
        case EntityType::CHICKEN:
            score = habitat.food * 0.25f + habitat.fertility * 0.15f +
                    preference(habitat.cover, 0.30f, 0.70f) * 0.15f + temperate * 0.15f +
                    mildMoisture * 0.10f + lowSlope * 0.15f + water * 0.05f;
            break;
        case EntityType::DEER:
            score = habitat.food * 0.16f + habitat.fertility * 0.11f +
                    preference(habitat.cover, 0.55f, 0.55f) * 0.25f + temperate * 0.14f +
                    habitat.moisture * 0.10f + lowSlope * 0.10f + water * 0.10f +
                    habitat.riverSize * 0.04f;
            break;
        case EntityType::GOAT:
            score = habitat.slope * 0.38f +
                    clamp01((static_cast<float>(habitat.surfaceY) - 72.0f) / 80.0f) * 0.20f +
                    openGround * 0.12f + preference(habitat.temperature, 0.35f, 0.55f) * 0.12f +
                    habitat.food * 0.07f + habitat.fertility * 0.03f +
                    preference(habitat.moisture, 0.35f, 0.65f) * 0.04f + water * 0.04f;
            break;
        case EntityType::RABBIT:
            score = habitat.food * 0.24f + habitat.fertility * 0.16f + openGround * 0.20f +
                    lowSlope * 0.15f + preference(habitat.moisture, 0.35f, 0.55f) * 0.10f +
                    temperate * 0.10f + water * 0.05f;
            break;
        case EntityType::FROG:
            if (water < 0.35f || habitat.moisture < 0.35f) return 0.0f;
            score = water * 0.25f + habitat.moisture * 0.20f +
                    preference(habitat.temperature, 0.75f, 0.55f) * 0.18f +
                    preference(habitat.cover, 0.55f, 0.65f) * 0.12f + habitat.fertility * 0.08f +
                    habitat.food * 0.07f + habitat.riverSize * 0.10f;
            break;
        case EntityType::FISH:
            if (!habitat.generatedWaterBody || habitat.waterDepth < 2) return 0.0f;
            score = clamp01((static_cast<float>(habitat.waterDepth) - 1.0f) / 7.0f) * 0.35f +
                    water * 0.20f + preference(habitat.temperature, 0.65f, 0.65f) * 0.15f +
                    habitat.moisture * 0.10f + habitat.fertility * 0.10f +
                    habitat.riverSize * 0.10f;
            break;
        case EntityType::COUNT:
            return 0.0f;
    }

    return clamp01(score);
}

int Spawner::getCarryingCapacity(EntityType type, const HabitatSample& habitat) {
    float score = getHabitatScore(type, habitat);
    if (score < 0.25f) return 0;
    if (type == EntityType::FISH) {
        return std::clamp(2 + static_cast<int>(std::lround(score * 4.0f)), 2, 6);
    }
    return std::clamp(1 + static_cast<int>(std::lround(score * 3.0f)), 1, 4);
}

std::optional<int> Spawner::findSpawnHeight(int64_t x, int64_t z) {
    int64_t chunkX = Chunk::worldToChunk(x);
    int64_t chunkZ = Chunk::worldToChunk(z);
    for (int chunkY = WORLD_MAX_CHUNK_Y; chunkY >= WORLD_MIN_CHUNK_Y; --chunkY) {
        if (!world_.isChunkLoaded({chunkX, chunkY, chunkZ})) continue;
        for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
            int y = chunkY * CHUNK_EDGE + localY;
            if (isSolid(world_.getBlockIfLoaded(x, y, z))) return y + 1;
        }
    }
    return std::nullopt;
}

std::optional<int> Spawner::findWaterSpawnHeight(int64_t x, int64_t z) {
    int topWater = -1;
    int depth = 0;
    int64_t chunkX = Chunk::worldToChunk(x);
    int64_t chunkZ = Chunk::worldToChunk(z);
    bool finished = false;
    for (int chunkY = WORLD_MAX_CHUNK_Y; chunkY >= WORLD_MIN_CHUNK_Y && !finished; --chunkY) {
        if (!world_.isChunkLoaded({chunkX, chunkY, chunkZ})) {
            if (topWater >= 0) break;
            continue;
        }
        for (int localY = CHUNK_EDGE - 1; localY >= 0; --localY) {
            int y = chunkY * CHUNK_EDGE + localY;
            BlockType block = world_.getBlockIfLoaded(x, y, z);
            if (topWater < 0) {
                if (block == BlockType::WATER) {
                    topWater = y;
                    depth = 1;
                }
            } else if (block == BlockType::WATER) {
                ++depth;
            } else {
                finished = true;
                break;
            }
        }
    }
    if (depth < 2) return std::nullopt;
    return topWater - std::min(depth - 1, 2);
}

bool Spawner::isSpawnValid(int64_t x, int y, int64_t z) {
    BlockType atY = world_.getBlockIfLoaded(x, y, z);
    BlockType aboveY = world_.getBlockIfLoaded(x, y + 1, z);
    BlockType belowY = world_.getBlockIfLoaded(x, y - 1, z);
    return atY == BlockType::AIR && aboveY == BlockType::AIR && isSolid(belowY);
}

size_t Spawner::livingCount() const {
    return static_cast<size_t>(std::count_if(
        entities_.begin(), entities_.end(),
        [](const std::shared_ptr<Entity>& entity) { return entity && entity->alive; }));
}

std::shared_ptr<Entity> Spawner::spawnEntity(EntityType type, const Vec3& position) {
    if (type == EntityType::COUNT || livingCount() >= MAX_ANIMALS) return nullptr;
    auto entity = std::make_shared<Entity>(Entity::nextId(), type, position);
    entities_.push_back(entity);
    spatialHash_.insert(entity->id, entity->position);
    return entity;
}

std::shared_ptr<Entity> Spawner::spawnWildEntity(EntityType type, const Vec3& position,
                                                 TerritoryPos territory, int memberIndex) {
    if (type == EntityType::COUNT || livingCount() >= MAX_ANIMALS) return nullptr;
    uint64_t id =
        makeTerritoryEntityId(world_.getSeed(), territory.x, territory.z, type, memberIndex);
    if (getEntity(id) != nullptr) return nullptr;

    auto entity = std::make_shared<Entity>(id, type, position);
    entity->isWild = true;
    entity->territoryX = territory.x;
    entity->territoryZ = territory.z;
    entity->homePosition = position;
    entities_.push_back(entity);
    spatialHash_.insert(entity->id, entity->position);
    return entity;
}

std::shared_ptr<Entity> Spawner::spawnBaby(EntityType type, const Vec3& position,
                                           uint64_t parentId) {
    if (type == EntityType::COUNT || livingCount() >= MAX_ANIMALS) return nullptr;
    auto baby = std::make_shared<Entity>(Entity::nextId(), type, position);
    baby->isBaby = true;
    baby->babyTimer = 600;
    baby->parentId = parentId;

    if (Entity* parent = getEntity(parentId); parent != nullptr && parent->isWild) {
        baby->isWild = true;
        baby->territoryX = parent->territoryX;
        baby->territoryZ = parent->territoryZ;
        baby->homePosition = parent->homePosition;
    }

    entities_.push_back(baby);
    spatialHash_.insert(baby->id, baby->position);
    return baby;
}

std::optional<HabitatSample> Spawner::findHabitatSample(int64_t x, int64_t z) {
    const std::optional<worldgen::SurfaceSample> foundSurface = world_.findSurfaceSample(x, z);
    if (!foundSurface) return std::nullopt;

    const worldgen::SurfaceSample& surface = *foundSurface;
    HabitatSample sample;
    sample.biome = surface.biome.primary;
    sample.surfaceY = static_cast<int>(std::lround(surface.terrainHeight));
    sample.temperature = clamp01(static_cast<float>((surface.climate.temperatureC + 20.0) / 55.0));
    sample.moisture = clamp01(static_cast<float>(surface.soil.moisture));
    sample.fertility = clamp01(static_cast<float>(surface.soil.fertility));
    sample.slope = clamp01(static_cast<float>(surface.slope / 1.6));
    sample.riverSize = clamp01(static_cast<float>(surface.hydrology.channelWidth / 24.0 +
                                                  std::log1p(surface.hydrology.discharge) / 18.0));
    sample.generatedRiver = surface.hydrology.river;
    sample.generatedLake = surface.hydrology.lake;
    sample.generatedOcean = surface.hydrology.ocean;
    sample.generatedWaterBody = generatedWaterBody(surface);
    const auto biomeScore = [&](Biome biome) {
        return static_cast<float>(surface.suitability.scores[static_cast<size_t>(biome)]);
    };
    sample.cover = clamp01(std::max(
        {biomeScore(Biome::FOREST), biomeScore(Biome::BIRCH_FOREST), biomeScore(Biome::TAIGA),
         biomeScore(Biome::TROPICAL_RAINFOREST), biomeScore(Biome::TEMPERATE_RAINFOREST),
         biomeScore(Biome::MANGROVE), biomeScore(Biome::MEDITERRANEAN_WOODLAND),
         biomeScore(Biome::TEMPERATE_CONIFER_FOREST), biomeScore(Biome::TROPICAL_CONIFER_FOREST),
         biomeScore(Biome::TROPICAL_DRY_FOREST)}));
    sample.food = clamp01(sample.fertility * 0.55f + sample.moisture * 0.25f +
                          static_cast<float>(surface.climate.annualPrecipitationMm / 6000.0));
    sample.waterDepth = static_cast<int>(std::lround(
        std::max({0.0, surface.hydrology.lakeDepth, surface.hydrology.channelDepth,
                  surface.hydrology.ocean ? surface.waterSurface - surface.terrainHeight : 0.0})));

    const bool waterMargin =
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::RIVERBANK) ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::FLOODPLAIN) ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::LAKESHORE) ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::COAST);
    sample.waterAccess = sample.generatedOcean || sample.generatedLake
                             ? 1.0f
                             : (sample.generatedRiver ? 0.55f + sample.riverSize * 0.45f
                                                      : (waterMargin ? 0.65f : 0.0f));
    sample.nearWater = sample.waterDepth > 0 || sample.generatedWaterBody || waterMargin;
    return sample;
}

std::optional<EntityType> Spawner::selectSpecies(const HabitatSample& habitat,
                                                 uint64_t roll) const {
    std::array<float, ENTITY_TYPE_COUNT> weights{};
    float total = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) {
        EntityType type = static_cast<EntityType>(i);
        weights[i] = getHabitatScore(type, habitat);
        if (weights[i] >= 0.25f) total += weights[i];
    }
    if (total <= 0.0f) return std::nullopt;

    float value = static_cast<float>(roll >> 40) / static_cast<float>(1ULL << 24) * total;
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < 0.25f) continue;
        if (value < weights[i]) return static_cast<EntityType>(i);
        value -= weights[i];
    }
    return static_cast<EntityType>(weights.size() - 1);
}

void Spawner::spawnTerritory(TerritoryPos territory) {
    if (populatedTerritories_.contains(territory) || livingCount() >= MAX_ANIMALS) return;

    Vec2 anchor = getTerritoryAnchor(world_.getSeed(), territory.x, territory.z);
    const auto foundHabitat = findHabitatSample(static_cast<int64_t>(std::floor(anchor.x)),
                                                static_cast<int64_t>(std::floor(anchor.y)));
    if (!foundHabitat) return;
    const HabitatSample& habitat = *foundHabitat;
    uint64_t selectionHash =
        territoryHash(world_.getSeed(), territory.x, territory.z, 0x53504543ULL);
    std::optional<EntityType> selected = selectSpecies(habitat, selectionHash);
    if (!selected.has_value()) return;

    EntityType type = selected.value();
    int capacity = getCarryingCapacity(type, habitat);
    if (capacity <= 0) return;
    int64_t territoryBaseX = territory.x * TERRITORY_SIZE;
    int64_t territoryBaseZ = territory.z * TERRITORY_SIZE;
    bool territoryResolved = true;

    int member = 0;
    for (; member < capacity && livingCount() < MAX_ANIMALS; ++member) {
        const uint64_t memberId =
            makeTerritoryEntityId(world_.getSeed(), territory.x, territory.z, type, member);
        if (getEntity(memberId) != nullptr) continue;

        bool spawned = false;
        bool memberResolved = true;
        for (int attempt = 0; attempt < 12 && !spawned; ++attempt) {
            uint64_t placement = territoryHash(world_.getSeed(), territory.x, territory.z,
                                               0x504C414345ULL + member * 31ULL + attempt);
            int64_t x = territoryBaseX + 4 + static_cast<int>(placement % 56ULL);
            int64_t z = territoryBaseZ + 4 + static_cast<int>((placement >> 20) % 56ULL);

            const auto candidateSurface = world_.findSurfaceSample(x, z);
            if (!candidateSurface) {
                memberResolved = false;
                continue;
            }

            if (type == EntityType::FISH) {
                if (!generatedWaterBody(*candidateSurface) ||
                    std::max({candidateSurface->hydrology.lakeDepth,
                              candidateSurface->hydrology.channelDepth,
                              candidateSurface->hydrology.ocean
                                  ? candidateSurface->waterSurface - candidateSurface->terrainHeight
                                  : 0.0}) < 2.0) {
                    continue;
                }
                const int waterSurfaceY =
                    static_cast<int>(std::floor(candidateSurface->waterSurface));
                const int64_t chunkX = Chunk::worldToChunk(x);
                const int64_t chunkZ = Chunk::worldToChunk(z);
                if (!world_.isChunkLoaded({chunkX, Chunk::worldToChunkY(waterSurfaceY), chunkZ}) ||
                    !world_.isChunkLoaded(
                        {chunkX, Chunk::worldToChunkY(waterSurfaceY - 2), chunkZ})) {
                    memberResolved = false;
                    continue;
                }
                auto waterY = findWaterSpawnHeight(x, z);
                if (!waterY.has_value()) continue;
                Vec3 position{static_cast<float>(x) + 0.5f, static_cast<float>(waterY.value()),
                              static_cast<float>(z) + 0.5f};
                spawned = spawnWildEntity(type, position, territory, member) != nullptr;
                continue;
            }

            const std::optional<int> loadedSurface = world_.surfaceHeightIfLoaded(x, z);
            if (!loadedSurface) {
                memberResolved = false;
                continue;
            }
            const int surfaceY = *loadedSurface;
            const int64_t chunkX = Chunk::worldToChunk(x);
            const int64_t chunkZ = Chunk::worldToChunk(z);
            if (!world_.isChunkLoaded({chunkX, Chunk::worldToChunkY(surfaceY), chunkZ}) ||
                !world_.isChunkLoaded({chunkX, Chunk::worldToChunkY(surfaceY + 2), chunkZ})) {
                memberResolved = false;
                continue;
            }

            auto height = findSpawnHeight(x, z);
            if (!height.has_value() || !isSpawnValid(x, height.value(), z)) continue;
            BlockType ground = world_.getBlockIfLoaded(x, height.value() - 1, z);
            if (!suitableGround(type, ground)) continue;

            Vec3 position{static_cast<float>(x) + 0.5f, static_cast<float>(height.value()),
                          static_cast<float>(z) + 0.5f};
            spawned = spawnWildEntity(type, position, territory, member) != nullptr;
        }
        if (!spawned && !memberResolved) territoryResolved = false;
    }
    if (member < capacity) territoryResolved = false;
    if (territoryResolved) populatedTerritories_.insert(territory);
}

void Spawner::spawnForChunk(int64_t chunkX, int64_t chunkZ) {
    int64_t centerX = chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2;
    int64_t centerZ = chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2;
    spawnTerritory({floorDiv(centerX, TERRITORY_SIZE), floorDiv(centerZ, TERRITORY_SIZE)});
}

void Spawner::spawnInitialPopulation() {
    const auto chunks = world_.getLoadedChunks();
    std::vector<TerritoryPos> territories;
    territories.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        const int64_t centerX = chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2;
        const int64_t centerZ = chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2;
        territories.push_back(
            {floorDiv(centerX, TERRITORY_SIZE), floorDiv(centerZ, TERRITORY_SIZE)});
    }
    std::sort(territories.begin(), territories.end(),
              [](const TerritoryPos& lhs, const TerritoryPos& rhs) {
                  return std::tie(lhs.x, lhs.z) < std::tie(rhs.x, rhs.z);
              });
    territories.erase(std::unique(territories.begin(), territories.end()), territories.end());

    for (const TerritoryPos territory : territories) {
        if (livingCount() >= MAX_ANIMALS) break;
        spawnTerritory(territory);
    }
}

bool Spawner::shouldDespawn(const Vec3& entityPosition, const Vec3& playerPosition) {
    Vec3 offset = entityPosition - playerPosition;
    return offset.lengthSq() > DESPAWN_RADIUS * DESPAWN_RADIUS;
}

void Spawner::despawnDistantWildlife(const Vec3& playerPosition) {
    entities_.erase(std::remove_if(entities_.begin(), entities_.end(),
                                   [&](const std::shared_ptr<Entity>& entity) {
                                       if (!entity) return true;
                                       bool remove =
                                           !entity->alive ||
                                           (entity->isWild &&
                                            shouldDespawn(entity->position, playerPosition));
                                       if (remove) spatialHash_.remove(entity->id);
                                       return remove;
                                   }),
                    entities_.end());
}

void Spawner::pruneTerritories(const Vec3& playerPosition) {
    for (auto it = populatedTerritories_.begin(); it != populatedTerritories_.end();) {
        Vec2 anchor = getTerritoryAnchor(world_.getSeed(), it->x, it->z);
        float dx = anchor.x - playerPosition.x;
        float dz = anchor.y - playerPosition.z;
        if (dx * dx + dz * dz > DESPAWN_RADIUS * DESPAWN_RADIUS) {
            it = populatedTerritories_.erase(it);
        } else {
            ++it;
        }
    }
}

void Spawner::updatePopulation(const Vec3& playerPosition) {
    ++populationTick_;
    Vec3 moved = playerPosition - lastPopulationCenter_;
    float movedHorizontalSq = moved.x * moved.x + moved.z * moved.z;
    bool movedFar = movedHorizontalSq >= (TERRITORY_SIZE / 2) * (TERRITORY_SIZE / 2);
    bool periodic = populationTick_ % POPULATION_UPDATE_TICKS == 0;
    if (hasPopulationCenter_ && !movedFar && !periodic) return;

    despawnDistantWildlife(playerPosition);
    pruneTerritories(playerPosition);

    int64_t playerX = static_cast<int64_t>(std::floor(playerPosition.x));
    int64_t playerZ = static_cast<int64_t>(std::floor(playerPosition.z));
    int64_t centerTerritoryX = floorDiv(playerX, TERRITORY_SIZE);
    int64_t centerTerritoryZ = floorDiv(playerZ, TERRITORY_SIZE);
    constexpr int TERRITORY_RADIUS = 2;

    for (int dz = -TERRITORY_RADIUS; dz <= TERRITORY_RADIUS; ++dz) {
        for (int dx = -TERRITORY_RADIUS; dx <= TERRITORY_RADIUS; ++dx) {
            TerritoryPos territory{centerTerritoryX + dx, centerTerritoryZ + dz};
            Vec2 anchor = getTerritoryAnchor(world_.getSeed(), territory.x, territory.z);
            float offsetX = anchor.x - playerPosition.x;
            float offsetZ = anchor.y - playerPosition.z;
            if (offsetX * offsetX + offsetZ * offsetZ <= ACTIVE_RADIUS * ACTIVE_RADIUS) {
                spawnTerritory(territory);
            }
        }
    }

    lastPopulationCenter_ = playerPosition;
    hasPopulationCenter_ = true;
}

std::vector<std::shared_ptr<Entity>>& Spawner::getEntities() {
    return entities_;
}

Entity* Spawner::getEntity(uint64_t entityId) {
    for (auto& entity : entities_) {
        if (entity && entity->id == entityId) return entity.get();
    }
    return nullptr;
}

std::unordered_map<uint64_t, Vec3> Spawner::getEntityPositions() const {
    std::unordered_map<uint64_t, Vec3> result;
    result.reserve(entities_.size());
    for (const auto& entity : entities_) {
        if (entity && entity->alive) result[entity->id] = entity->position;
    }
    return result;
}

void Spawner::removeEntity(uint64_t entityId) {
    spatialHash_.remove(entityId);
    entities_.erase(std::remove_if(entities_.begin(), entities_.end(),
                                   [entityId](const std::shared_ptr<Entity>& entity) {
                                       return entity && entity->id == entityId && !entity->alive;
                                   }),
                    entities_.end());
}

SpatialHash& Spawner::getSpatialHash() {
    return spatialHash_;
}
