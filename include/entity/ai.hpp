#pragma once

#include <entity/entity.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

// Forward declarations
class SpatialHash;
class Spawner;

// ---------------------------------------------------------------------------
// AnimalState — AI behavior states for entity state machine
// ---------------------------------------------------------------------------
enum class AnimalState : uint8_t {
    IDLE,
    WANDER,
    FLEE,
    EAT,
    BREED,
    FOLLOW_PLAYER,
};

// ---------------------------------------------------------------------------
// StateMachine — Finite state machine for animal AI
//
// Each state has onEnter, update, and onExit callbacks. Transitions are
// driven by conditions checked in the update loop. The state machine
// maintains per-state timers and target data.
// ---------------------------------------------------------------------------
class StateMachine {
public:
    StateMachine();

    // Current state
    AnimalState currentState = AnimalState::IDLE;

    // Per-state data
    Vec3 targetPosition = Vec3::zero();
    std::optional<uint64_t> targetEntityId = std::nullopt;
    int stateTimer = 0; // ticks in current state
    int idleTimer = 0;  // ticks since last state change (for idle→wander)
    int fleeTimer = 0;  // ticks in FLEE state

    // -----------------------------------------------------------------------
    // State transitions
    // -----------------------------------------------------------------------
    // Evaluate conditions and update state.
    // Returns the velocity offset to apply (AI-driven movement).
    Vec3 update(Entity& entity, World& world, const Vec3& playerPosition, bool playerMovingToward,
                bool playerHoldingFood, Spawner& spawner);

    // Transition to a new state
    void transitionTo(AnimalState newState);

    // -----------------------------------------------------------------------
    // State callbacks
    // -----------------------------------------------------------------------
    void onEnterIdle();
    void onEnterWander();
    void onEnterFlee();
    void onEnterEat();
    void onEnterBreed();
    void onEnterFollowPlayer();

    void onExitIdle();
    void onExitWander();
    void onExitFlee();
    void onExitEat();
    void onExitBreed();
    void onExitFollowPlayer();

    // -----------------------------------------------------------------------
    // Condition checks (pure functions)
    // -----------------------------------------------------------------------
    static bool shouldFlee(const Vec3& entityPos, const Vec3& playerPos, bool playerMovingToward,
                           float fleeDistance = 6.0f);
    static bool shouldStopFleeing(const Vec3& entityPos, const Vec3& playerPos, int fleeTicks,
                                  float safeDistance = 10.0f, int maxFleeTicks = 300);
    static bool shouldEat(int hungerTicks, int eatThreshold = 300);
    static bool shouldWander(int idleTicks, int minIdle = 30, int maxIdle = 120);
    static bool shouldStopWandering(int wanderTicks, int maxWander = 200);
    static bool shouldFollowPlayer(const Vec3& entityPos, const Vec3& playerPos,
                                   bool playerHoldingFood, float followDistance = 10.0f);
    static bool shouldStopFollowing(const Vec3& entityPos, const Vec3& playerPos,
                                    bool playerHoldingFood, float stopDistance = 10.0f);
};

// ---------------------------------------------------------------------------
// FlockingController — Boid-style flocking behavior
//
// Implements separation, alignment, and cohesion rules. Only active during
// WANDER state. Queries SpatialHash for nearby neighbors.
// ---------------------------------------------------------------------------
class FlockingController {
public:
    // Weights per rule
    static constexpr float SEPARATION_WEIGHT = 2.0f;
    static constexpr float ALIGNMENT_WEIGHT = 1.0f;
    static constexpr float COHESION_WEIGHT = 1.0f;

    // Radii
    static constexpr float SEPARATION_RADIUS = 2.0f;
    static constexpr float ALIGNMENT_RADIUS = 5.0f;
    static constexpr float COHESION_RADIUS = 5.0f;

    // Maximum flocking force per tick
    static constexpr float MAX_FLOCKING_FORCE = 0.05f;

    // Compute flocking steering force for an entity
    static Vec3 computeSteering(Entity& entity, Spawner& spawner);

    // Clamp force to maximum (exposed for testing)
    static Vec3 clampForce(const Vec3& force, float maxForce);

private:
    // Pure functions for each flocking rule
    static Vec3 computeSeparation(Entity& entity, const std::vector<uint64_t>& neighborIds,
                                  Spawner& spawner);
    static Vec3 computeAlignment(Entity& entity, const std::vector<uint64_t>& neighborIds,
                                 Spawner& spawner);
    static Vec3 computeCohesion(Entity& entity, const std::vector<uint64_t>& neighborIds,
                                Spawner& spawner);
};

// ---------------------------------------------------------------------------
// EdgeDetector — Terrain edge awareness for AI movement
//
// Before moving, checks blocks ahead to detect cliffs and water.
// Prevents entities from walking off edges.
// ---------------------------------------------------------------------------
class EdgeDetector {
public:
    // Check if movement toward `direction` is safe.
    // Returns false if there's a cliff or (for non-pigs) water ahead.
    static bool isSafeToMove(const Vec3& entityPos, const Vec3& direction, EntityType entityType,
                             World& world);

    // Check for cliff: block ahead is AIR and block below that is also AIR
    static bool isCliffAhead(const Vec3& entityPos, const Vec3& direction, World& world);

    // Check for water ahead
    static bool isWaterAhead(const Vec3& entityPos, const Vec3& direction, World& world);
};

// ---------------------------------------------------------------------------
// BehaviorController — Specific AI behaviors (eat, breed, follow)
// ---------------------------------------------------------------------------
class BehaviorController {
public:
    // Execute eat behavior: bob animation, reset hunger
    static void doEat(Entity& entity, World& world);

    // Execute breed behavior: find mate, spawn baby
    static std::optional<uint64_t> doBreed(Entity& entity, Spawner& spawner);

    // Execute follow-player behavior: steer toward player
    static Vec3 computeFollowSteering(const Vec3& entityPos, const Vec3& playerPos,
                                      float minDistance = 3.0f, float maxDistance = 6.0f);

    // Check if entity is standing on grass
    static bool isOnGrass(Entity& entity, World& world);

    // Check if entity can breed: mate nearby and both fed
    static bool canBreed(Entity& entity, Spawner& spawner);
};
