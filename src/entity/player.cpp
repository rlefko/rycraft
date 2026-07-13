#include "entity/player.hpp"
#include "entity/physics.hpp"

#include <cmath>

// Water physics modifiers (Task 6.7-6.8)
static constexpr float WATER_GRAVITY_MULTIPLIER = 0.3f;
static constexpr float WATER_HORIZONTAL_DRAG = 0.7f;
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

    // 1. Apply input: WASD → horizontal velocity based on yaw
    float speed = WALK_SPEED;
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

    // 2. Apply gravity (reduced in water)
    float effectiveGravity = GRAVITY;
    if (inWater) {
        effectiveGravity *= WATER_GRAVITY_MULTIPLIER;
    }
    velocity.y += effectiveGravity;

    // 3. Apply drag (increased in water)
    float horizontalDrag = onGround ? HORIZONTAL_DRAG_GROUND : HORIZONTAL_DRAG_AIR;
    if (inWater) {
        horizontalDrag = WATER_HORIZONTAL_DRAG;
    }
    velocity.x *= horizontalDrag;
    velocity.z *= horizontalDrag;
    velocity.y *= VERTICAL_DRAG;

    // 4. Apply buoyancy force when in water
    if (inWater && velocity.y < 0.f) {
        velocity.y += WATER_BUOYANCY_FORCE;
    }

    // 5. Clamp velocity to terminal velocity
    if (velocity.y < TERMINAL_VELOCITY) {
        velocity.y = TERMINAL_VELOCITY;
    }

    // 6. Resolve collisions via physics engine
    PhysicsEngine physics;
    Vec3 resolvedMovement = physics.sweepCollision(playerAABB, velocity, world);

    // 7. Update position by resolved movement
    position.x += resolvedMovement.x;
    position.y += resolvedMovement.y;
    position.z += resolvedMovement.z;

    // 8. Track fall distance: accumulate negative Y displacement
    if (resolvedMovement.y < 0.f) {
        fallDistance += static_cast<int>(std::ceil(-resolvedMovement.y * 20.f));
    }

    // 9. Check if on ground (zero vertical movement while velocity was negative)
    onGround = (std::abs(resolvedMovement.y) < 1e-6f && velocity.y < 0.f);

    // If on ground, apply fall damage and reset
    if (onGround) {
        if (fallDistance > 3) {
            applyFallDamage();
        }
        resetFallDistance();
    }

    // 10. Reset fall distance every 100 ticks to avoid FP drift
    fallResetTimer++;
    if (fallResetTimer >= 100) {
        resetFallDistance();
        fallResetTimer = 0;
    }

    // 11. Decrement jump cooldown
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
    if (fallDistance <= 3) return;

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
    fallDistance = 0;
}
