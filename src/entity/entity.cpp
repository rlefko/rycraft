#include "entity/entity.hpp"
#include "entity/physics.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <utility>

namespace {

constexpr std::array<EntityConfig, ENTITY_TYPE_COUNT> ENTITY_CONFIGS = {{
    {0.6f, 0.9f, 0.025f, Vec3{0.9f, 0.9f, 0.9f}, MovementMode::WALK, 0.0f, 1.0f, 8,
     ItemType::RAW_MUTTON, 2},
    {0.9f, 1.4f, 0.020f, Vec3{0.55f, 0.27f, 0.07f}, MovementMode::WALK, 0.0f, 1.0f, 10,
     ItemType::RAW_BEEF, 2},
    {0.7f, 0.9f, 0.025f, Vec3{0.9f, 0.65f, 0.75f}, MovementMode::WALK, 0.0f, 1.0f, 10,
     ItemType::RAW_PORKCHOP, 2},
    {0.4f, 0.7f, 0.030f, Vec3{0.95f, 0.95f, 0.9f}, MovementMode::WALK, 0.0f, 1.0f, 4,
     ItemType::RAW_CHICKEN, 1},
    {0.7f, 1.5f, 0.040f, Vec3{0.55f, 0.32f, 0.16f}, MovementMode::WALK, 0.0f, 1.0f, 10,
     ItemType::RAW_BEEF, 2},
    {0.7f, 1.2f, 0.038f, Vec3{0.72f, 0.68f, 0.58f}, MovementMode::CLIMB, 0.22f, 1.25f, 10,
     ItemType::RAW_MUTTON, 1},
    {0.4f, 0.55f, 0.045f, Vec3{0.67f, 0.58f, 0.46f}, MovementMode::HOP, 0.24f, 0.6f, 3,
     ItemType::RAW_CHICKEN, 1},
    {0.45f, 0.35f, 0.035f, Vec3{0.25f, 0.62f, 0.23f}, MovementMode::AMPHIBIOUS_HOP, 0.28f, 0.6f, 5,
     ItemType::NONE, 0},
    {0.55f, 0.45f, 0.035f, Vec3{0.28f, 0.58f, 0.78f}, MovementMode::SWIM, 0.0f, 0.0f, 3,
     ItemType::RAW_FISH, 1},
}};

static_assert(ENTITY_CONFIGS.size() == ENTITY_TYPE_COUNT);
static_assert([] {
    for (const EntityConfig& config : ENTITY_CONFIGS) {
        if (config.maxHealth <= 0) return false;
        if ((config.drop == ItemType::NONE) != (config.dropCount == 0)) return false;
        if (config.drop != ItemType::NONE && !isFood(config.drop)) return false;
    }
    return true;
}());

bool isWaterAt(World& world, const Vec3& position, float height) {
    const int x = static_cast<int>(std::floor(position.x));
    const float sampleY = position.y + height * 0.5f;
    const int y = static_cast<int>(std::floor(sampleY));
    const int z = static_cast<int>(std::floor(position.z));
    return world.getBlockIfLoaded(x, y, z) == BlockType::WATER &&
           sampleY < static_cast<float>(y) + world.getFluidHeightIfLoaded(x, y, z);
}

} // namespace

// ---------------------------------------------------------------------------
// nextId — Monotonically increasing entity UUID
// ---------------------------------------------------------------------------
static std::atomic<uint64_t> g_entityCounter{1};

uint64_t Entity::nextId() {
    return g_entityCounter.fetch_add(1);
}

// ---------------------------------------------------------------------------
// getConfig — Per-type entity configuration
// ---------------------------------------------------------------------------
EntityConfig Entity::getConfig(EntityType type) {
    const auto index = static_cast<size_t>(type);
    assert(index < ENTITY_CONFIGS.size());
    return ENTITY_CONFIGS[index];
}

// ---------------------------------------------------------------------------
// Entity constructor
// ---------------------------------------------------------------------------
Entity::Entity(uint64_t entityId, EntityType entityType, const Vec3& spawnPos)
    : id(entityId)
    , type(entityType)
    , position(spawnPos)
    , velocity(Vec3::zero())
    , health(getConfig(entityType).maxHealth)
    , homePosition(spawnPos) {
    aabb = computeAABB();
}

// ---------------------------------------------------------------------------
// computeAABB — Build bounding box from position and type config
// ---------------------------------------------------------------------------
AABB Entity::computeAABB() const {
    EntityConfig cfg = getConfig(type);
    float effectiveWidth = isBaby ? cfg.width * 0.5f : cfg.width;
    float effectiveHeight = isBaby ? cfg.height * 0.5f : cfg.height;
    float halfWidth = effectiveWidth * 0.5f;

    return AABB{Vec3{position.x - halfWidth, position.y, position.z - halfWidth},
                Vec3{position.x + halfWidth, position.y + effectiveHeight, position.z + halfWidth}};
}

// ---------------------------------------------------------------------------
// applyForces — Gravity and drag (pure physics, no collision)
// ---------------------------------------------------------------------------
void Entity::applyForces(bool inWater) {
    // Gravity
    float effectiveGravity = GRAVITY;
    if (inWater) {
        effectiveGravity *= WATER_GRAVITY_MULTIPLIER;
    }
    velocity.y += effectiveGravity;

    // Drag
    float horizontalDrag = HORIZONTAL_DRAG_AIR;
    if (inWater) {
        horizontalDrag = HORIZONTAL_DRAG_WATER;
    }
    velocity.x *= horizontalDrag;
    velocity.z *= horizontalDrag;
    velocity.y *= VERTICAL_DRAG;

    // Buoyancy
    if (inWater && velocity.y < 0.f) {
        velocity.y += WATER_BUOYANCY_FORCE;
    }

    // Terminal velocity
    if (velocity.y < TERMINAL_VELOCITY) {
        velocity.y = TERMINAL_VELOCITY;
    }
}

// ---------------------------------------------------------------------------
// resolveCollision — Per-axis sweep via PhysicsEngine
// ---------------------------------------------------------------------------
void Entity::resolveCollision(World& world, Vec3& movement) {
    aabb = computeAABB();
    PhysicsEngine physics;
    movement = physics.sweepCollision(aabb, movement, world);
}

// ---------------------------------------------------------------------------
// tryStepAssist — Climb up to 1 block if gap is small enough
// ---------------------------------------------------------------------------
void Entity::tryStepAssist(World& world, float deltaY) {
    // Only step up when falling and the fall distance is small
    if (deltaY >= 0.f) return;
    const float stepHeight = std::max(STEP_ASSIST_HEIGHT, getConfig(type).stepHeight);
    if (std::abs(deltaY) > stepHeight) return;

    int stepX = static_cast<int>(std::floor(position.x));
    int stepZ = static_cast<int>(std::floor(position.z));
    int groundY = static_cast<int>(std::floor(position.y));

    // Check if stepping up by 1 block is possible
    // Block above must be AIR, and block at new ground must be solid
    if (!PhysicsEngine::isSolid(world, stepX, groundY + 1, stepZ)) {
        if (PhysicsEngine::isSolid(world, stepX, groundY + 2, stepZ)) {
            // Step up
            position.y += std::min(stepHeight, 1.0f);
            velocity.y = 0.f;
            onGround = true;
        }
    }
}

// ---------------------------------------------------------------------------
// tick — Full physics tick for the entity
// ---------------------------------------------------------------------------
void Entity::tickLifecycle() {
    if (isBaby && babyTimer > 0) {
        babyTimer--;
        if (babyTimer <= 0) {
            isBaby = false;
            aabb = computeAABB();
        }
    }

    if (eatAnimationTimer > 0) {
        eatAnimationTimer--;
    }

    if (hungerTimer < 600) {
        hungerTimer++;
    }
}

void Entity::tickAquatic(World& world) {
    const EntityConfig config = getConfig(type);
    aabb = computeAABB();

    if (!isWaterAt(world, position, config.height)) {
        applyForces(false);
        Vec3 movement = velocity;
        resolveCollision(world, movement);
        position += movement;
        onGround = movement.y != velocity.y && velocity.y < 0.0f;
        if (onGround) velocity.y = 0.0f;
        aabb = computeAABB();
        tickLifecycle();
        return;
    }

    // Neutral buoyancy with strong damping keeps schools responsive without
    // allowing accumulated steering to launch fish through the water surface.
    velocity *= 0.86f;
    Vec3 movement = velocity;
    resolveCollision(world, movement);

    Vec3 candidate = position + movement;
    if (!isWaterAt(world, candidate, config.height)) {
        Vec3 axisCandidate = position;
        axisCandidate.x += movement.x;
        if (!isWaterAt(world, axisCandidate, config.height)) {
            movement.x = 0.0f;
            velocity.x = 0.0f;
        }
        axisCandidate = position;
        axisCandidate.y += movement.y;
        if (!isWaterAt(world, axisCandidate, config.height)) {
            movement.y = 0.0f;
            velocity.y = 0.0f;
        }
        axisCandidate = position;
        axisCandidate.z += movement.z;
        if (!isWaterAt(world, axisCandidate, config.height)) {
            movement.z = 0.0f;
            velocity.z = 0.0f;
        }
    }

    position += movement;
    onGround = false;
    aabb = computeAABB();
    tickLifecycle();
}

void Entity::tick(World& world) {
    // Guard: dead entities don't tick
    if (!alive) return;

    if (getConfig(type).movementMode == MovementMode::SWIM) {
        tickAquatic(world);
        return;
    }

    // 0. Check water
    aabb = computeAABB();
    bool inWater = PhysicsEngine::isInWater(world, aabb);

    // 1. Apply forces (gravity, drag, buoyancy)
    applyForces(inWater);

    // 2. Resolve collision
    Vec3 movement = velocity;
    resolveCollision(world, movement);

    // 3. Update position
    position.x += movement.x;
    position.y += movement.y;
    position.z += movement.z;

    // 4. Step assist (only for negative Y movement)
    if (movement.y < 0.f) {
        tryStepAssist(world, movement.y);
    }

    // 5. Update onGround flag: check if block below feet is solid
    int footX = static_cast<int>(std::floor(position.x));
    int footY = static_cast<int>(std::floor(position.y)) - 1;
    int footZ = static_cast<int>(std::floor(position.z));
    onGround = PhysicsEngine::isSolid(world, footX, footY, footZ);

    // Zero downward velocity while grounded so it does not saturate toward
    // terminal velocity between ticks — the same missing reset that made the
    // player fall a whole block in a single tick after standing still.
    if (onGround && velocity.y < 0.f) {
        velocity.y = 0.f;
    }

    // 6. Update AABB
    aabb = computeAABB();

    // 7. Tick lifecycle state
    tickLifecycle();
}

// ---------------------------------------------------------------------------
// getVoxelModel — Box-based voxel model per entity type
// ---------------------------------------------------------------------------
std::vector<VoxelBlock> Entity::getVoxelModel(EntityType type, bool isBaby) {
    std::vector<VoxelBlock> blocks;
    float scale = isBaby ? 0.5f : 1.0f;

    auto scaled = [&](Vec3 v) { return v * scale; };

    switch (type) {
        case EntityType::SHEEP: {
            Vec3 woolColor = Vec3{0.9f, 0.9f, 0.9f};
            Vec3 legColor = Vec3{0.3f, 0.2f, 0.15f};

            // Body (3-block wide, centered)
            blocks.push_back({{0.f, 0.5f * scale, 0.f}, scaled({0.6f, 0.5f, 0.6f}), woolColor});
            // Head
            blocks.push_back(
                {{0.f, 0.8f * scale, -0.35f * scale}, scaled({0.35f, 0.35f, 0.35f}), woolColor});
            // Legs
            for (int lx = -1; lx <= 1; lx += 2) {
                for (int lz = -1; lz <= 1; lz += 2) {
                    blocks.push_back({{lx * 0.15f * scale, 0.f, lz * 0.15f * scale},
                                      scaled({0.1f, 0.3f, 0.1f}),
                                      legColor});
                }
            }
            break;
        }
        case EntityType::COW: {
            Vec3 bodyColor = Vec3{0.55f, 0.27f, 0.07f};
            Vec3 whitePatch = Vec3{0.9f, 0.9f, 0.9f};
            Vec3 legColor = Vec3{0.4f, 0.2f, 0.1f};

            // Body (larger)
            blocks.push_back({{0.f, 0.6f * scale, 0.f}, scaled({0.9f, 0.6f, 0.8f}), bodyColor});
            // White patch on body
            blocks.push_back(
                {{0.1f * scale, 0.7f * scale, 0.f}, scaled({0.3f, 0.3f, 0.4f}), whitePatch});
            // Head
            blocks.push_back(
                {{0.f, 1.0f * scale, -0.5f * scale}, scaled({0.4f, 0.4f, 0.4f}), bodyColor});
            // Legs
            for (int lx = -1; lx <= 1; lx += 2) {
                for (int lz = -1; lz <= 1; lz += 2) {
                    blocks.push_back({{lx * 0.25f * scale, 0.f, lz * 0.25f * scale},
                                      scaled({0.12f, 0.4f, 0.12f}),
                                      legColor});
                }
            }
            // Udder
            blocks.push_back({{0.f, 0.2f * scale, 0.35f * scale},
                              scaled({0.2f, 0.15f, 0.15f}),
                              Vec3{0.9f, 0.7f, 0.8f}});
            break;
        }
        case EntityType::PIG: {
            Vec3 bodyColor = Vec3{0.9f, 0.65f, 0.75f};
            Vec3 legColor = Vec3{0.8f, 0.55f, 0.65f};

            // Body (compact)
            blocks.push_back({{0.f, 0.45f * scale, 0.f}, scaled({0.7f, 0.5f, 0.6f}), bodyColor});
            // Head
            blocks.push_back(
                {{0.f, 0.7f * scale, -0.35f * scale}, scaled({0.3f, 0.3f, 0.3f}), bodyColor});
            // Snout
            blocks.push_back({{0.f, 0.65f * scale, -0.5f * scale},
                              scaled({0.2f, 0.15f, 0.15f}),
                              Vec3{0.8f, 0.4f, 0.5f}});
            // Legs
            for (int lx = -1; lx <= 1; lx += 2) {
                for (int lz = -1; lz <= 1; lz += 2) {
                    blocks.push_back({{lx * 0.2f * scale, 0.f, lz * 0.15f * scale},
                                      scaled({0.1f, 0.25f, 0.1f}),
                                      legColor});
                }
            }
            break;
        }
        case EntityType::CHICKEN: {
            Vec3 bodyColor = Vec3{0.95f, 0.95f, 0.9f};
            Vec3 beakColor = Vec3{0.9f, 0.7f, 0.1f};
            Vec3 combColor = Vec3{0.9f, 0.1f, 0.1f};

            // Body (small)
            blocks.push_back({{0.f, 0.3f * scale, 0.f}, scaled({0.4f, 0.35f, 0.4f}), bodyColor});
            // Head
            blocks.push_back(
                {{0.f, 0.55f * scale, -0.15f * scale}, scaled({0.25f, 0.25f, 0.25f}), bodyColor});
            // Beak
            blocks.push_back(
                {{0.f, 0.5f * scale, -0.3f * scale}, scaled({0.1f, 0.08f, 0.1f}), beakColor});
            // Comb
            blocks.push_back(
                {{0.f, 0.7f * scale, -0.1f * scale}, scaled({0.1f, 0.1f, 0.1f}), combColor});
            // Legs (2)
            blocks.push_back(
                {{-0.08f * scale, 0.f, 0.f}, scaled({0.06f, 0.15f, 0.06f}), beakColor});
            blocks.push_back({{0.08f * scale, 0.f, 0.f}, scaled({0.06f, 0.15f, 0.06f}), beakColor});
            // Wings
            blocks.push_back(
                {{-0.25f * scale, 0.35f * scale, 0.f}, scaled({0.08f, 0.2f, 0.1f}), bodyColor});
            blocks.push_back(
                {{0.25f * scale, 0.35f * scale, 0.f}, scaled({0.08f, 0.2f, 0.1f}), bodyColor});
            break;
        }
        case EntityType::DEER: {
            const Vec3 coat{0.55f, 0.32f, 0.16f};
            const Vec3 darkCoat{0.32f, 0.18f, 0.09f};
            const Vec3 antler{0.72f, 0.61f, 0.43f};
            blocks.push_back({{0.f, 0.72f * scale, 0.f}, scaled({0.65f, 0.62f, 0.9f}), coat});
            blocks.push_back(
                {{0.f, 1.1f * scale, -0.55f * scale}, scaled({0.38f, 0.4f, 0.4f}), coat});
            blocks.push_back(
                {{0.f, 0.85f * scale, -0.42f * scale}, scaled({0.18f, 0.45f, 0.18f}), coat});
            for (int lx : {-1, 1}) {
                for (int lz : {-1, 1}) {
                    blocks.push_back({{lx * 0.2f * scale, 0.f, lz * 0.28f * scale},
                                      scaled({0.09f, 0.65f, 0.09f}),
                                      darkCoat});
                }
            }
            if (!isBaby) {
                for (int side : {-1, 1}) {
                    blocks.push_back(
                        {{side * 0.13f, 1.43f, -0.55f}, {0.05f, 0.34f, 0.05f}, antler});
                    blocks.push_back({{side * 0.2f, 1.64f, -0.55f}, {0.18f, 0.05f, 0.05f}, antler});
                }
            }
            break;
        }
        case EntityType::GOAT: {
            const Vec3 coat{0.72f, 0.68f, 0.58f};
            const Vec3 leg{0.38f, 0.34f, 0.29f};
            const Vec3 horn{0.48f, 0.43f, 0.33f};
            blocks.push_back({{0.f, 0.55f * scale, 0.f}, scaled({0.68f, 0.55f, 0.75f}), coat});
            blocks.push_back(
                {{0.f, 0.88f * scale, -0.45f * scale}, scaled({0.4f, 0.4f, 0.38f}), coat});
            for (int lx : {-1, 1}) {
                for (int lz : {-1, 1}) {
                    blocks.push_back({{lx * 0.19f * scale, 0.f, lz * 0.22f * scale},
                                      scaled({0.1f, 0.42f, 0.1f}),
                                      leg});
                }
            }
            blocks.push_back(
                {{0.f, 0.78f * scale, -0.68f * scale}, scaled({0.15f, 0.18f, 0.12f}), leg});
            if (!isBaby) {
                blocks.push_back({{-0.12f, 1.17f, -0.45f}, {0.07f, 0.28f, 0.07f}, horn});
                blocks.push_back({{0.12f, 1.17f, -0.45f}, {0.07f, 0.28f, 0.07f}, horn});
            }
            break;
        }
        case EntityType::RABBIT: {
            const Vec3 fur{0.67f, 0.58f, 0.46f};
            const Vec3 innerEar{0.82f, 0.56f, 0.58f};
            blocks.push_back(
                {{0.f, 0.18f * scale, 0.08f * scale}, scaled({0.4f, 0.3f, 0.5f}), fur});
            blocks.push_back(
                {{0.f, 0.32f * scale, -0.26f * scale}, scaled({0.3f, 0.3f, 0.28f}), fur});
            blocks.push_back({{-0.09f * scale, 0.56f * scale, -0.25f * scale},
                              scaled({0.08f, 0.3f, 0.08f}),
                              innerEar});
            blocks.push_back({{0.09f * scale, 0.56f * scale, -0.25f * scale},
                              scaled({0.08f, 0.3f, 0.08f}),
                              innerEar});
            blocks.push_back({{0.f, 0.25f * scale, 0.37f * scale},
                              scaled({0.16f, 0.16f, 0.16f}),
                              Vec3{0.88f, 0.85f, 0.78f}});
            blocks.push_back(
                {{-0.14f * scale, 0.f, 0.16f * scale}, scaled({0.12f, 0.16f, 0.24f}), fur});
            blocks.push_back(
                {{0.14f * scale, 0.f, 0.16f * scale}, scaled({0.12f, 0.16f, 0.24f}), fur});
            break;
        }
        case EntityType::FROG: {
            const Vec3 green{0.25f, 0.62f, 0.23f};
            const Vec3 lightGreen{0.48f, 0.75f, 0.34f};
            blocks.push_back({{0.f, 0.1f * scale, 0.f}, scaled({0.45f, 0.22f, 0.38f}), green});
            blocks.push_back(
                {{0.f, 0.22f * scale, -0.2f * scale}, scaled({0.35f, 0.2f, 0.25f}), green});
            blocks.push_back({{-0.12f * scale, 0.36f * scale, -0.22f * scale},
                              scaled({0.11f, 0.11f, 0.11f}),
                              lightGreen});
            blocks.push_back({{0.12f * scale, 0.36f * scale, -0.22f * scale},
                              scaled({0.11f, 0.11f, 0.11f}),
                              lightGreen});
            for (int side : {-1, 1}) {
                blocks.push_back({{side * 0.24f * scale, 0.f, 0.1f * scale},
                                  scaled({0.18f, 0.08f, 0.25f}),
                                  green});
            }
            break;
        }
        case EntityType::FISH: {
            const Vec3 body{0.28f, 0.58f, 0.78f};
            const Vec3 fin{0.17f, 0.39f, 0.62f};
            blocks.push_back({{0.f, 0.08f * scale, 0.f}, scaled({0.42f, 0.3f, 0.68f}), body});
            blocks.push_back(
                {{0.f, 0.12f * scale, -0.39f * scale}, scaled({0.3f, 0.25f, 0.2f}), body});
            blocks.push_back(
                {{0.f, 0.08f * scale, 0.44f * scale}, scaled({0.06f, 0.38f, 0.34f}), fin});
            blocks.push_back(
                {{-0.27f * scale, 0.02f * scale, 0.f}, scaled({0.16f, 0.05f, 0.22f}), fin});
            blocks.push_back(
                {{0.27f * scale, 0.02f * scale, 0.f}, scaled({0.16f, 0.05f, 0.22f}), fin});
            break;
        }
        case EntityType::COUNT:
            std::unreachable();
    }

    return blocks;
}
