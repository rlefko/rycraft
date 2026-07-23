#pragma once

#include <common/math.hpp>
#include <world/chunk.hpp>
#include <world/item.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Forward declaration
class World;

// ---------------------------------------------------------------------------
// EntityType — Animal kinds the game supports
// ---------------------------------------------------------------------------
enum class EntityType : uint8_t {
    SHEEP,
    COW,
    PIG,
    CHICKEN,
    DEER,
    GOAT,
    RABBIT,
    FROG,
    FISH,
    COUNT,
};

inline constexpr size_t ENTITY_TYPE_COUNT = static_cast<size_t>(EntityType::COUNT);
static_assert(ENTITY_TYPE_COUNT == 9, "Entity tables must cover every entity type");

// ---------------------------------------------------------------------------
// MovementMode - Shared movement family used by physics and animal AI
// ---------------------------------------------------------------------------
enum class MovementMode : uint8_t {
    WALK,
    CLIMB,
    HOP,
    AMPHIBIOUS_HOP,
    SWIM,
};

// ---------------------------------------------------------------------------
// EntityConfig — Per-type dimensions, speed, and voxel color
// ---------------------------------------------------------------------------
struct EntityConfig {
    float width;
    float height;
    float speed;
    Vec3 color;
    MovementMode movementMode;
    float jumpImpulse;
    float stepHeight;
    int maxHealth; // half-hearts an attack must remove
    ItemType drop; // survival loot on death; NONE drops nothing
    uint8_t dropCount;
};

// ---------------------------------------------------------------------------
// VoxelBlock — A single colored box for entity voxel rendering
// ---------------------------------------------------------------------------
struct VoxelBlock {
    Vec3 offset; // relative to entity feet center
    Vec3 size;   // block dimensions
    Vec3 color;  // RGB color (0-1)
};

// ---------------------------------------------------------------------------
// Entity — Base class for all living entities (animals, mobs)
//
// Physics mirrors the player: gravity, drag, terminal velocity, per-axis
// sweep collision via PhysicsEngine. Entities are identified by a unique
// uint64_t UUID assigned at construction.
// ---------------------------------------------------------------------------
class Entity {
public:
    // Identifiers
    uint64_t id;
    EntityType type;

    // Position and motion
    Vec3 position;
    Vec3 velocity;

    // Bounding box (computed from position + config)
    AABB aabb;

    // State flags
    bool onGround = false;
    bool alive = true;
    int health = 0; // set from EntityConfig::maxHealth at construction

    // AI timers
    int hungerTimer = 0;   // ticks since last eat
    bool isFed = false;    // true if recently fed (for breeding)
    bool isBaby = false;   // baby entities are smaller
    int babyTimer = 0;     // ticks remaining as baby (0 = adult)
    uint64_t parentId = 0; // parent entity ID (for babies)

    // Wild animals stay near a deterministic territory. Manually spawned
    // animals retain their spawn position as home but are not distance culled.
    Vec3 homePosition;
    bool isWild = false;
    int64_t territoryX = 0;
    int64_t territoryZ = 0;

    // Animation
    int eatAnimationTimer = 0; // ticks remaining for eat bob animation

    // Fixed-tick packed sky/block light sampled at the model center. The
    // render thread consumes this cache without taking World locks.
    uint8_t renderPackedLight = FULL_SKY_PACKED_LIGHT;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------
    Entity(uint64_t entityId, EntityType entityType, const Vec3& spawnPos);

    // Delete copy (entities are unique)
    Entity(const Entity&) = delete;
    Entity& operator=(const Entity&) = delete;

    // -----------------------------------------------------------------------
    // Physics
    // -----------------------------------------------------------------------
    // Full physics tick: gravity, drag, collision, step assist, buoyancy
    void tick(World& world);

    // Compute AABB from current position and type config
    AABB computeAABB() const;

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------
    // Get config for an entity type
    static EntityConfig getConfig(EntityType type);

    // Get voxel model blocks for an entity type
    static std::vector<VoxelBlock> getVoxelModel(EntityType type, bool isBaby);

    // -----------------------------------------------------------------------
    // UUID generation
    // -----------------------------------------------------------------------
    static uint64_t nextId();

private:
    // Physics constants (shared with player)
    static constexpr float GRAVITY = -0.08f;
    static constexpr float HORIZONTAL_DRAG_AIR = 0.91f;
    static constexpr float HORIZONTAL_DRAG_WATER = 0.7f;
    static constexpr float VERTICAL_DRAG = 0.98f;
    static constexpr float TERMINAL_VELOCITY = -3.92f;
    static constexpr float WATER_GRAVITY_MULTIPLIER = 0.3f;
    static constexpr float WATER_BUOYANCY_FORCE = 0.02f;
    static constexpr float STEP_ASSIST_HEIGHT = 1.0f;

    // Apply gravity and drag modifiers
    void applyForces(bool inWater);

    // Fish use neutral-buoyancy movement while submerged and regular gravity
    // if stranded on land.
    void tickAquatic(World& world);

    // Update timers shared by terrestrial and aquatic physics paths.
    void tickLifecycle();

    // Attempt step assist: climb up to STEP_ASSIST_HEIGHT blocks
    void tryStepAssist(World& world, float deltaY);

    // Resolve per-axis collision using PhysicsEngine
    void resolveCollision(World& world, Vec3& movement);
};
