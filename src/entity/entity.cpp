#include "entity/entity.hpp"
#include "entity/physics.hpp"

#include <atomic>
#include <cmath>

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
    switch (type) {
    case EntityType::SHEEP:
        return {0.6f, 0.9f, 0.025f, Vec3{0.9f, 0.9f, 0.9f}};
    case EntityType::COW:
        return {0.9f, 1.4f, 0.02f, Vec3{0.55f, 0.27f, 0.07f}};
    case EntityType::PIG:
        return {0.7f, 0.9f, 0.025f, Vec3{0.9f, 0.65f, 0.75f}};
    case EntityType::CHICKEN:
        return {0.4f, 0.7f, 0.03f, Vec3{0.95f, 0.95f, 0.9f}};
    }
    // Fail loud: unreachable
    return {0.5f, 0.5f, 0.02f, Vec3{1.f, 0.f, 1.f}};
}

// ---------------------------------------------------------------------------
// Entity constructor
// ---------------------------------------------------------------------------
Entity::Entity(uint64_t entityId, EntityType entityType, const Vec3& spawnPos)
    : id(entityId), type(entityType), position(spawnPos), velocity(Vec3::zero()) {
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

    return AABB{
        Vec3{position.x - halfWidth, position.y, position.z - halfWidth},
        Vec3{position.x + halfWidth, position.y + effectiveHeight, position.z + halfWidth}
    };
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
    if (std::abs(deltaY) > STEP_ASSIST_HEIGHT) return;

    int stepX = static_cast<int>(std::floor(position.x));
    int stepZ = static_cast<int>(std::floor(position.z));
    int groundY = static_cast<int>(std::floor(position.y));

    // Check if stepping up by 1 block is possible
    // Block above must be AIR, and block at new ground must be solid
    if (!PhysicsEngine::isSolid(world, stepX, groundY + 1, stepZ)) {
        if (PhysicsEngine::isSolid(world, stepX, groundY + 2, stepZ)) {
            // Step up
            position.y += STEP_ASSIST_HEIGHT;
            velocity.y = 0.f;
            onGround = true;
        }
    }
}

// ---------------------------------------------------------------------------
// tick — Full physics tick for the entity
// ---------------------------------------------------------------------------
void Entity::tick(World& world) {
    // Guard: dead entities don't tick
    if (!alive) return;

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

    // 5. Update onGround flag
    onGround = (std::abs(movement.y) < 1e-6f && velocity.y < 0.f);

    // 6. Update AABB
    aabb = computeAABB();

    // 7. Tick baby timer
    if (isBaby && babyTimer > 0) {
        babyTimer--;
        if (babyTimer <= 0) {
            isBaby = false;
            aabb = computeAABB();
        }
    }

    // 8. Tick eat animation
    if (eatAnimationTimer > 0) {
        eatAnimationTimer--;
    }

    // 9. Tick hunger
    if (hungerTimer < 600) {
        hungerTimer++;
    }
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
        blocks.push_back({{0.f, 0.8f * scale, -0.35f * scale}, scaled({0.35f, 0.35f, 0.35f}), woolColor});
        // Legs
        for (int lx = -1; lx <= 1; lx += 2) {
            for (int lz = -1; lz <= 1; lz += 2) {
                blocks.push_back({{lx * 0.15f * scale, 0.f, lz * 0.15f * scale},
                                  scaled({0.1f, 0.3f, 0.1f}), legColor});
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
        blocks.push_back({{0.1f * scale, 0.7f * scale, 0.f}, scaled({0.3f, 0.3f, 0.4f}), whitePatch});
        // Head
        blocks.push_back({{0.f, 1.0f * scale, -0.5f * scale}, scaled({0.4f, 0.4f, 0.4f}), bodyColor});
        // Legs
        for (int lx = -1; lx <= 1; lx += 2) {
            for (int lz = -1; lz <= 1; lz += 2) {
                blocks.push_back({{lx * 0.25f * scale, 0.f, lz * 0.25f * scale},
                                  scaled({0.12f, 0.4f, 0.12f}), legColor});
            }
        }
        // Udder
        blocks.push_back({{0.f, 0.2f * scale, 0.35f * scale}, scaled({0.2f, 0.15f, 0.15f}), Vec3{0.9f, 0.7f, 0.8f}});
        break;
    }
    case EntityType::PIG: {
        Vec3 bodyColor = Vec3{0.9f, 0.65f, 0.75f};
        Vec3 legColor = Vec3{0.8f, 0.55f, 0.65f};

        // Body (compact)
        blocks.push_back({{0.f, 0.45f * scale, 0.f}, scaled({0.7f, 0.5f, 0.6f}), bodyColor});
        // Head
        blocks.push_back({{0.f, 0.7f * scale, -0.35f * scale}, scaled({0.3f, 0.3f, 0.3f}), bodyColor});
        // Snout
        blocks.push_back({{0.f, 0.65f * scale, -0.5f * scale}, scaled({0.2f, 0.15f, 0.15f}), Vec3{0.8f, 0.4f, 0.5f}});
        // Legs
        for (int lx = -1; lx <= 1; lx += 2) {
            for (int lz = -1; lz <= 1; lz += 2) {
                blocks.push_back({{lx * 0.2f * scale, 0.f, lz * 0.15f * scale},
                                  scaled({0.1f, 0.25f, 0.1f}), legColor});
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
        blocks.push_back({{0.f, 0.55f * scale, -0.15f * scale}, scaled({0.25f, 0.25f, 0.25f}), bodyColor});
        // Beak
        blocks.push_back({{0.f, 0.5f * scale, -0.3f * scale}, scaled({0.1f, 0.08f, 0.1f}), beakColor});
        // Comb
        blocks.push_back({{0.f, 0.7f * scale, -0.1f * scale}, scaled({0.1f, 0.1f, 0.1f}), combColor});
        // Legs (2)
        blocks.push_back({{-0.08f * scale, 0.f, 0.f}, scaled({0.06f, 0.15f, 0.06f}), beakColor});
        blocks.push_back({{0.08f * scale, 0.f, 0.f}, scaled({0.06f, 0.15f, 0.06f}), beakColor});
        // Wings
        blocks.push_back({{-0.25f * scale, 0.35f * scale, 0.f}, scaled({0.08f, 0.2f, 0.1f}), bodyColor});
        blocks.push_back({{0.25f * scale, 0.35f * scale, 0.f}, scaled({0.08f, 0.2f, 0.1f}), bodyColor});
        break;
    }
    }

    return blocks;
}
