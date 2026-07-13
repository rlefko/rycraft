#include "entity/player.hpp"
#include "entity/physics.hpp"

#include <cmath>

// Water physics modifiers (Task 6.7-6.8)
static constexpr float WATER_GRAVITY_MULTIPLIER = 0.3f;
static constexpr float WATER_BUOYANCY_FORCE = 0.02f;

// ---------------------------------------------------------------------------
// getAABB — Compute player's axis-aligned bounding box
// ---------------------------------------------------------------------------
AABB Player::getAABB() const {
    float halfWidth = WIDTH * 0.5f;
    return AABB{Vec3{position.x - halfWidth, position.y, position.z - halfWidth},
                Vec3{position.x + halfWidth, position.y + HEIGHT, position.z + halfWidth}};
}

// ---------------------------------------------------------------------------
// tick — Full physics tick for the player
// ---------------------------------------------------------------------------
void Player::tick(World& world, const InputState& input, bool sprinting) {
    // 0. Check if player is in water
    AABB playerAABB = getAABB();
    bool inWater = PhysicsEngine::isInWater(world, playerAABB);

    // 1. Apply input: WASD → horizontal velocity based on yaw.
    // Water halves the pace (horizontal drag used to approximate this).
    float speed = inWater ? WALK_SPEED * 0.5f : WALK_SPEED;
    if (sprinting) {
        speed *= SPRINT_MULTIPLIER;
    }

    float moveX = 0.f;
    float moveZ = 0.f;

    // Camera basis (see Camera::updateFront/right): forward is
    // (+sin yaw, +cos yaw) and right is (-cos yaw, +sin yaw). These signs
    // were all inverted before, so W walked backwards and D strafed left.
    if (input.isDown(Key::W)) {
        moveX += std::sin(yaw) * speed;
        moveZ += std::cos(yaw) * speed;
    }
    if (input.isDown(Key::S)) {
        moveX -= std::sin(yaw) * speed;
        moveZ -= std::cos(yaw) * speed;
    }
    if (input.isDown(Key::A)) {
        moveX += std::cos(yaw) * speed;
        moveZ -= std::sin(yaw) * speed;
    }
    if (input.isDown(Key::D)) {
        moveX -= std::cos(yaw) * speed;
        moveZ += std::sin(yaw) * speed;
    }

    velocity.x = moveX;
    velocity.z = moveZ;

    // 2. Move FIRST using the current velocity, then integrate gravity/drag
    // for the NEXT tick (vanilla-Minecraft semi-implicit order). Applying
    // gravity+drag before the move used to spend the fresh 0.42 jump impulse
    // on decay instead of displacement, capping the jump apex at ~0.83 blocks
    // — below a 1-block step, so the player could never climb onto one. Moving
    // first restores the ~1.25-block apex.
    PhysicsEngine physics;
    Vec3 resolvedMovement = physics.sweepCollision(playerAABB, velocity, world);

    position.x += resolvedMovement.x;
    position.y += resolvedMovement.y;
    position.z += resolvedMovement.z;

    // 3. Vertical collision response: when the sweep clips the intended Y move
    // (floor or ceiling), zero velocity.y. Without this reset velocity.y kept
    // integrating downward while merely standing, saturating toward terminal
    // velocity; the instant the player stepped off a ledge that stored speed
    // dropped them a whole block in a single tick ("instantaneous fall").
    bool yBlocked = std::abs(resolvedMovement.y - velocity.y) > 1e-6f;
    onGround = yBlocked && velocity.y < 0.f;
    if (yBlocked) {
        velocity.y = 0.f;
    }

    // 4. Track fall distance in BLOCKS actually descended (the old ceil(·×20)
    // summed velocities, so a 1-block fall registered as ~17 blocks of damage).
    if (resolvedMovement.y < 0.f) {
        fallDistance += -resolvedMovement.y;
    }

    // If on ground, apply fall damage and reset
    if (onGround) {
        if (fallDistance > 3.f) {
            applyFallDamage();
        }
        resetFallDistance();
    }

    // 5. Integrate gravity, drag, buoyancy, and terminal velocity for the next
    // tick (reduced gravity + buoyancy in water)
    float effectiveGravity = GRAVITY;
    if (inWater) {
        effectiveGravity *= WATER_GRAVITY_MULTIPLIER;
    }
    velocity.y += effectiveGravity;
    velocity.y *= VERTICAL_DRAG;
    if (inWater && velocity.y < 0.f) {
        velocity.y += WATER_BUOYANCY_FORCE;
    }
    if (velocity.y < TERMINAL_VELOCITY) {
        velocity.y = TERMINAL_VELOCITY;
    }

    // 6. Reset fall distance every 100 ticks to avoid FP drift
    fallResetTimer++;
    if (fallResetTimer >= 100) {
        resetFallDistance();
        fallResetTimer = 0;
    }

    // 7. Decrement jump cooldown
    if (jumpCooldown > 0) {
        jumpCooldown--;
    }
}

// ---------------------------------------------------------------------------
// jump — Apply jump velocity when on ground with cooldown
// ---------------------------------------------------------------------------
void Player::jump() {
    // Guard: cannot jump if cooldown active or not on ground
    if (jumpCooldown > 0) return;
    if (!onGround) return;

    velocity.y = JUMP_VELOCITY;

    // Reduced jump velocity in water
    // (checked in tick, but we store the intent here)
    jumpCooldown = JUMP_COOLDOWN_TICKS;
}

// ---------------------------------------------------------------------------
// applyFallDamage — Deal damage based on fall distance
// ---------------------------------------------------------------------------
void Player::applyFallDamage() {
    if (fallDistance <= 3.f) return;

    int damage = static_cast<int>(std::ceil(fallDistance - 3.f));
    health -= damage;

    if (health < 0) {
        health = 0;
    }
}

// ---------------------------------------------------------------------------
// resetFallDistance — Clear fall distance tracking
// ---------------------------------------------------------------------------
void Player::resetFallDistance() {
    fallDistance = 0.f;
}
