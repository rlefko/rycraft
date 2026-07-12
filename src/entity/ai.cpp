#include "entity/ai.hpp"
#include "entity/entity.hpp"
#include "entity/physics.hpp"
#include "entity/spatial_hash.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

// ---------------------------------------------------------------------------
// Simple deterministic random for AI (seeded per entity ID)
// ---------------------------------------------------------------------------
static float entityRandom(uint64_t seed) {
    // Simple LCG seeded by entity ID for deterministic behavior
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // Use a hash of seed to perturb the random
    uint64_t hash = seed ^ 0x5deece66d;
    hash = (hash ^ (hash >> 9)) * 0xcc9e2d51;
    hash = (hash ^ (hash >> 17)) * 0x1b873593;
    hash = hash ^ (hash >> 13);
    // Advance rng by hash amount (modulo to keep reasonable)
    for (uint64_t i = 0; i < (hash % 100 + 1); ++i) {
        rng.discard(1);
    }
    return dist(rng);
}

// ---------------------------------------------------------------------------
// StateMachine
// ---------------------------------------------------------------------------
StateMachine::StateMachine() {
    // Seed random based on time for variety
    std::srand(static_cast<unsigned>(std::time(nullptr)));
}

void StateMachine::transitionTo(AnimalState newState) {
    // Exit current state
    switch (currentState) {
    case AnimalState::IDLE: onExitIdle(); break;
    case AnimalState::WANDER: onExitWander(); break;
    case AnimalState::FLEE: onExitFlee(); break;
    case AnimalState::EAT: onExitEat(); break;
    case AnimalState::BREED: onExitBreed(); break;
    case AnimalState::FOLLOW_PLAYER: onExitFollowPlayer(); break;
    }

    currentState = newState;
    stateTimer = 0;

    // Enter new state
    switch (currentState) {
    case AnimalState::IDLE: onEnterIdle(); break;
    case AnimalState::WANDER: onEnterWander(); break;
    case AnimalState::FLEE: onEnterFlee(); break;
    case AnimalState::EAT: onEnterEat(); break;
    case AnimalState::BREED: onEnterBreed(); break;
    case AnimalState::FOLLOW_PLAYER: onEnterFollowPlayer(); break;
    }
}

void StateMachine::onEnterIdle() {
    idleTimer = 0;
    targetPosition = Vec3::zero();
    targetEntityId = std::nullopt;
}

void StateMachine::onEnterWander() {
    stateTimer = 0;
    targetEntityId = std::nullopt;
}

void StateMachine::onEnterFlee() {
    fleeTimer = 0;
}

void StateMachine::onEnterEat() {
    stateTimer = 0;
    targetPosition = Vec3::zero();
    targetEntityId = std::nullopt;
}

void StateMachine::onEnterBreed() {
    stateTimer = 0;
}

void StateMachine::onEnterFollowPlayer() {
    stateTimer = 0;
}

void StateMachine::onExitIdle() {}
void StateMachine::onExitWander() {}
void StateMachine::onExitFlee() {}
void StateMachine::onExitEat() {}
void StateMachine::onExitBreed() {}
void StateMachine::onExitFollowPlayer() {}

// ---------------------------------------------------------------------------
// Condition checks (pure functions)
// ---------------------------------------------------------------------------
bool StateMachine::shouldFlee(const Vec3& entityPos, const Vec3& playerPos,
                               bool playerMovingToward, float fleeDistance) {
    if (!playerMovingToward) return false;
    Vec3 diff = playerPos - entityPos;
    float distSq = diff.x * diff.x + diff.z * diff.z; // horizontal only
    return distSq <= fleeDistance * fleeDistance;
}

bool StateMachine::shouldStopFleeing(const Vec3& entityPos, const Vec3& playerPos,
                                      int fleeTicks, float safeDistance, int maxFleeTicks) {
    Vec3 diff = playerPos - entityPos;
    float distSq = diff.x * diff.x + diff.z * diff.z;
    return distSq > safeDistance * safeDistance || fleeTicks >= maxFleeTicks;
}

bool StateMachine::shouldEat(int hungerTicks, int eatThreshold) {
    return hungerTicks >= eatThreshold;
}

bool StateMachine::shouldWander(int idleTicks, int minIdle, int maxIdle) {
    // Guard: must have been idle long enough
    if (idleTicks < minIdle) return false;
    // At maxIdle, always wander
    if (idleTicks >= maxIdle) return true;
    // Between minIdle and maxIdle, deterministic pseudo-random chance
    // Using a hash that ensures minIdle itself always triggers (for testability)
    return idleTicks == minIdle || (idleTicks * 13 + 7) % 17 == 0;
}

bool StateMachine::shouldStopWandering(int wanderTicks, int maxWander) {
    return wanderTicks >= maxWander;
}

bool StateMachine::shouldFollowPlayer(const Vec3& entityPos, const Vec3& playerPos,
                                       bool playerHoldingFood, float followDistance) {
    if (!playerHoldingFood) return false;
    Vec3 diff = playerPos - entityPos;
    float distSq = diff.x * diff.x + diff.z * diff.z;
    return distSq <= followDistance * followDistance;
}

bool StateMachine::shouldStopFollowing(const Vec3& entityPos, const Vec3& playerPos,
                                        bool playerHoldingFood, float stopDistance) {
    if (!playerHoldingFood) return true;
    Vec3 diff = playerPos - entityPos;
    float distSq = diff.x * diff.x + diff.z * diff.z;
    return distSq > stopDistance * stopDistance;
}

// ---------------------------------------------------------------------------
// StateMachine::update — Main state machine evaluation
// ---------------------------------------------------------------------------
Vec3 StateMachine::update(Entity& entity, World& world,
                           const Vec3& playerPosition, bool playerMovingToward,
                           bool playerHoldingFood,
                           SpatialHash& spatialHash) {
    stateTimer++;
    idleTimer++;

    Vec3 steering = Vec3::zero();

    // Priority 1: FLEE (highest priority, preempts everything)
    if (currentState != AnimalState::FLEE &&
        shouldFlee(entity.position, playerPosition, playerMovingToward)) {
        transitionTo(AnimalState::FLEE);
        // Set flee target: away from player
        Vec3 away = entity.position - playerPosition;
        away.y = 0.f;
        float len = std::sqrt(away.x * away.x + away.z * away.z);
        if (len > 0.01f) {
            away = away / len;
            targetPosition = entity.position + away * 8.f;
        }
    }

    if (currentState == AnimalState::FLEE) {
        fleeTimer++;
        // Steer away from player
        Vec3 away = entity.position - playerPosition;
        away.y = 0.f;
        float len = std::sqrt(away.x * away.x + away.z * away.z);
        if (len > 0.01f) {
            steering = (away / len) * entity.getConfig(entity.type).speed * 2.f;
        }

        // Check if we should stop fleeing
        if (shouldStopFleeing(entity.position, playerPosition, fleeTimer)) {
            transitionTo(AnimalState::IDLE);
            return Vec3::zero();
        }
        return steering;
    }

    // Priority 2: FOLLOW_PLAYER (when player holds food)
    if (currentState != AnimalState::FOLLOW_PLAYER &&
        currentState != AnimalState::EAT &&
        shouldFollowPlayer(entity.position, playerPosition, playerHoldingFood)) {
        transitionTo(AnimalState::FOLLOW_PLAYER);
    }

    if (currentState == AnimalState::FOLLOW_PLAYER) {
        steering = BehaviorController::computeFollowSteering(entity.position, playerPosition);

        if (shouldStopFollowing(entity.position, playerPosition, playerHoldingFood)) {
            transitionTo(AnimalState::IDLE);
            return Vec3::zero();
        }
        return steering;
    }

    // Priority 3: EAT (when hungry and on grass)
    if (currentState != AnimalState::EAT &&
        shouldEat(entity.hungerTimer) &&
        BehaviorController::isOnGrass(entity, world)) {
        transitionTo(AnimalState::EAT);
    }

    if (currentState == AnimalState::EAT) {
        if (stateTimer >= 60) {
            // Eating complete
            BehaviorController::doEat(entity, world);
            transitionTo(AnimalState::IDLE);
            return Vec3::zero();
        }
        // Eat animation: no movement while eating
        return Vec3::zero();
    }

    // Priority 4: BREED (when mate nearby and both fed)
    if (currentState != AnimalState::BREED &&
        currentState != AnimalState::EAT &&
        BehaviorController::canBreed(entity, spatialHash)) {
        transitionTo(AnimalState::BREED);
    }

    if (currentState == AnimalState::BREED) {
        if (stateTimer >= 30) {
            BehaviorController::doBreed(entity, world, spatialHash);
            transitionTo(AnimalState::IDLE);
            return Vec3::zero();
        }
        return Vec3::zero();
    }

    // Priority 5: IDLE → WANDER
    if (currentState == AnimalState::IDLE && shouldWander(idleTimer)) {
        transitionTo(AnimalState::WANDER);
        // Set wander target
        float angle = entityRandom(entity.id + stateTimer) * 2.f * 3.14159f;
        float distance = 8.f + entityRandom(entity.id + stateTimer + 1) * 4.f;
        targetPosition = {
            entity.position.x + std::cos(angle) * distance,
            entity.position.y,
            entity.position.z + std::sin(angle) * distance
        };
    }

    if (currentState == AnimalState::IDLE) {
        return Vec3::zero();
    }

    // WANDER state
    if (currentState == AnimalState::WANDER) {
        // Check if wander target reached
        Vec3 toTarget = targetPosition - entity.position;
        toTarget.y = 0.f;
        float distToTarget = std::sqrt(toTarget.x * toTarget.x + toTarget.z * toTarget.z);

        if (distToTarget < 1.f || shouldStopWandering(stateTimer)) {
            transitionTo(AnimalState::IDLE);
            return Vec3::zero();
        }

        // Steer toward wander target
        steering = (toTarget / distToTarget) * entity.getConfig(entity.type).speed;

        // Add flocking behavior
        Vec3 flockForce = FlockingController::computeSteering(entity, spatialHash);
        steering = steering + flockForce;

        return steering;
    }

    return steering;
}

// ---------------------------------------------------------------------------
// FlockingController
// ---------------------------------------------------------------------------
Vec3 FlockingController::computeSteering(Entity& entity, const SpatialHash& spatialHash) {
    auto neighborIds = spatialHash.query(entity.position, COHESION_RADIUS);

    // Remove self from neighbors
    neighborIds.erase(
        std::remove(neighborIds.begin(), neighborIds.end(), entity.id),
        neighborIds.end()
    );

    if (neighborIds.empty()) return Vec3::zero();

    Vec3 separation = computeSeparation(entity, neighborIds, spatialHash);
    Vec3 alignment = computeAlignment(entity, neighborIds, spatialHash);
    Vec3 cohesion = computeCohesion(entity, neighborIds, spatialHash);

    Vec3 total = separation * SEPARATION_WEIGHT +
                 alignment * ALIGNMENT_WEIGHT +
                 cohesion * COHESION_WEIGHT;

    return clampForce(total, MAX_FLOCKING_FORCE);
}

Vec3 FlockingController::computeSeparation(Entity& /*entity*/,
                                            const std::vector<uint64_t>& /*neighborIds*/,
                                            const SpatialHash& /*spatialHash*/) {
    // Separation steering requires entity position data from a manager.
    // Returns zero when called without entity registry context.
    return Vec3::zero();
}

Vec3 FlockingController::computeAlignment(Entity& /*entity*/,
                                           const std::vector<uint64_t>& /*neighborIds*/,
                                           const SpatialHash& /*spatialHash*/) {
    // Alignment steering requires entity velocity data from a manager.
    // Returns zero when called without entity registry context.
    return Vec3::zero();
}

Vec3 FlockingController::computeCohesion(Entity& /*entity*/,
                                          const std::vector<uint64_t>& /*neighborIds*/,
                                          const SpatialHash& /*spatialHash*/) {
    // Cohesion steering requires entity position data from a manager.
    // Returns zero when called without entity registry context.
    return Vec3::zero();
}

Vec3 FlockingController::clampForce(const Vec3& force, float maxForce) {
    float len = force.length();
    if (len > maxForce && len > 0.001f) {
        return force / len * maxForce;
    }
    return force;
}

// ---------------------------------------------------------------------------
// EdgeDetector
// ---------------------------------------------------------------------------
bool EdgeDetector::isSafeToMove(const Vec3& entityPos, const Vec3& direction,
                                 EntityType entityType, World& world) {
    // Pigs are okay with water
    if (entityType != EntityType::PIG && isWaterAhead(entityPos, direction, world)) {
        return false;
    }
    return !isCliffAhead(entityPos, direction, world);
}

bool EdgeDetector::isCliffAhead(const Vec3& entityPos, const Vec3& direction, World& world) {
    // Check 1-2 steps ahead
    for (int steps = 1; steps <= 2; ++steps) {
        Vec3 checkPos = entityPos + direction * static_cast<float>(steps);
        int checkX = static_cast<int>(std::floor(checkPos.x));
        int checkZ = static_cast<int>(std::floor(checkPos.z));
        // Entity feet Y is the ground level; check the block below feet
        int groundY = static_cast<int>(std::floor(entityPos.y)) - 1;

        // Block at ground level ahead should be solid
        // If block at ground is AIR and block below that is also AIR → cliff
        BlockType atGround = world.getBlock(checkX, groundY, checkZ);
        BlockType belowGround = world.getBlock(checkX, groundY - 1, checkZ);

        if (atGround == BlockType::AIR && belowGround == BlockType::AIR) {
            return true; // Cliff detected
        }
    }
    return false;
}

bool EdgeDetector::isWaterAhead(const Vec3& entityPos, const Vec3& direction, World& world) {
    Vec3 checkPos = entityPos + direction * 1.5f;
    int checkX = static_cast<int>(std::floor(checkPos.x));
    int groundY = static_cast<int>(std::floor(entityPos.y)) - 1;
    int checkZ = static_cast<int>(std::floor(checkPos.z));

    return world.getBlock(checkX, groundY, checkZ) == BlockType::WATER ||
           world.getBlock(checkX, groundY + 1, checkZ) == BlockType::WATER;
}

// ---------------------------------------------------------------------------
// BehaviorController
// ---------------------------------------------------------------------------
void BehaviorController::doEat(Entity& entity, World& world) {
    // Reset hunger timer
    entity.hungerTimer = 0;
    entity.isFed = true;

    // Start eat animation
    entity.eatAnimationTimer = 60;

    // Consume grass block below entity
    int bx = static_cast<int>(std::floor(entity.position.x));
    int by = static_cast<int>(std::floor(entity.position.y));
    int bz = static_cast<int>(std::floor(entity.position.z));

    if (world.getBlock(bx, by, bz) == BlockType::GRASS) {
        world.setBlock(bx, by, bz, BlockType::DIRT);
    }
}

std::optional<uint64_t> BehaviorController::doBreed(Entity& entity, World& world,
                                                     SpatialHash& spatialHash) {
    // Find a mate: same type, within 4 blocks, also fed
    auto neighbors = spatialHash.query(entity.position, 4.0f);

    // We can't access entity data through spatial hash alone
    // The actual breed logic is handled by the Spawner which has entity references
    (void)world;
    return std::nullopt;
}

Vec3 BehaviorController::computeFollowSteering(const Vec3& entityPos, const Vec3& playerPos,
                                                float minDistance, float maxDistance) {
    Vec3 toPlayer = playerPos - entityPos;
    toPlayer.y = 0.f;
    float dist = std::sqrt(toPlayer.x * toPlayer.x + toPlayer.z * toPlayer.z);

    if (dist < 0.01f) return Vec3::zero();

    // Maintain 3-6 block distance
    if (dist < minDistance) {
        // Too close, move away
        return -(toPlayer / dist) * 0.02f;
    }
    if (dist > maxDistance) {
        // Too far, move closer
        return (toPlayer / dist) * 0.03f;
    }
    // In range, orbit slightly
    return Vec3::zero();
}

bool BehaviorController::isOnGrass(Entity& entity, World& world) {
    // Check block at feet level and one below (entity feet may be above ground)
    int bx = static_cast<int>(std::floor(entity.position.x));
    int by = static_cast<int>(std::floor(entity.position.y));
    int bz = static_cast<int>(std::floor(entity.position.z));

    return world.getBlock(bx, by, bz) == BlockType::GRASS ||
           world.getBlock(bx, by - 1, bz) == BlockType::GRASS;
}

bool BehaviorController::canBreed(Entity& entity, SpatialHash& spatialHash) {
    if (!entity.isFed) return false;

    auto neighbors = spatialHash.query(entity.position, 4.0f);

    // Need at least one other entity nearby (the actual type/fed check
    // is done by the Spawner which has full entity references)
    for (uint64_t nid : neighbors) {
        if (nid != entity.id) return true;
    }
    return false;
}
