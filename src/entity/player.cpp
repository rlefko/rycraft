#include "entity/player.hpp"
#include "entity/physics.hpp"

#include <algorithm>
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
void Player::tick(World& world, const PlayerInput& in) {
    // 0. Check if player is in water
    AABB playerAABB = getAABB();
    bool inWater = PhysicsEngine::isInWater(world, playerAABB);

    // 1. Movement modes. Fly toggles on a jump double-tap; the sprint latch
    // sets on a forward double-tap or the held sprint key, and clears the
    // tick forward is released (releasing the sprint key alone keeps it —
    // it is an initiator, not a maintainer).
    if (in.doubleTapJump) {
        flying = !flying;
        if (flying) {
            velocity.y = 0.f; // hover immediately — cancels the current fall
            resetFallDistance();
        }
    }
    if (!in.forward) {
        sprinting = false;
    } else if (in.doubleTapForward || in.sprintHeld) {
        sprinting = true;
    }
    swimming = sprinting && inWater && !flying;

    // 2. Apply input: WASD → velocity from yaw (swimming steers by the full
    // look direction, vertical included). Water halves the pace, sprint
    // multiplies it, fly replaces it (and ignores the water penalty — fly
    // overrides water physics entirely).
    float speed =
        flying ? WALK_SPEED * FLY_SPEED_MULTIPLIER : (inWater ? WALK_SPEED * 0.5f : WALK_SPEED);
    if (sprinting) {
        speed *= SPRINT_MULTIPLIER;
    }

    float moveX = 0.f;
    float moveZ = 0.f;

    // Camera basis (directionFromYawPitch at pitch 0): forward is
    // (+sin yaw, +cos yaw) and right is its 90° XZ rotation (-cos yaw,
    // +sin yaw). A hand-copied version of these signs was once inverted, so
    // W walked backwards and D strafed left.
    const Vec3 fwd = directionFromYawPitch(yaw, 0.f);
    const Vec3 rightDir{-fwd.z, 0.f, fwd.x};

    if (in.forward) {
        if (swimming) {
            // Swim along the full look direction: look down to dive, look up
            // to surface. Vertical velocity is overwritten, not added, so the
            // first non-swimming tick hands a clean value back to buoyancy.
            Vec3 look = directionFromYawPitch(yaw, pitch);
            moveX += look.x * SWIM_SPEED;
            moveZ += look.z * SWIM_SPEED;
            velocity.y = look.y * SWIM_SPEED;
        } else {
            moveX += fwd.x * speed;
            moveZ += fwd.z * speed;
        }
    }
    if (in.backward) {
        moveX -= fwd.x * speed;
        moveZ -= fwd.z * speed;
    }
    if (in.left) {
        moveX -= rightDir.x * speed;
        moveZ -= rightDir.z * speed;
    }
    if (in.right) {
        moveX += rightDir.x * speed;
        moveZ += rightDir.z * speed;
    }

    velocity.x = moveX;
    velocity.z = moveZ;

    // 2b. Vertical intent. Fly sets vertical velocity directly; in water a
    // held jump key floats the player toward a capped ascent (independent of
    // jump()'s onGround gate — this is what makes swimming up possible).
    // While swimming the look direction already owns velocity.y, so the
    // float-up stands down (it would drag a dive back toward the surface).
    if (flying) {
        velocity.y =
            in.jumpHeld ? FLY_VERTICAL_SPEED : (in.descendHeld ? -FLY_VERTICAL_SPEED : 0.f);
    } else if (inWater) {
        if (in.jumpHeld && !swimming && velocity.y < SWIM_UP_MAX_SPEED) {
            velocity.y = std::min(velocity.y + SWIM_UP_ACCELERATION, SWIM_UP_MAX_SPEED);
        }
        // Paddling against a bank pops the player upward: jump() is useless
        // here (it needs solid ground), so a swimmer could otherwise face a
        // 1-block shore forever. Uses last tick's wall contact, after the
        // swim overwrite above so the hop survives it.
        if (blockedHorizontally && (swimming || in.jumpHeld)) {
            velocity.y = std::max(velocity.y, WATER_EXIT_HOP);
        }
    }

    // 2c. Jump. A press always tries (standing on a lake bed included); a
    // HELD key auto-jumps on land only — in water it means "float up", and
    // bouncing off the bottom while wading felt wrong. Runs before the sweep
    // so the fresh impulse displaces this tick undecayed, and after the
    // water block so a lake-bed jump overrides the gentler float-up. The
    // fly-toggle tap can't double as a jump: toggling on just set `flying`,
    // and toggling off is only reachable airborne (flight never ends a tick
    // grounded), where jump()'s ground gate no-ops.
    if (!flying && (in.jumpPressed || (in.jumpHeld && !inWater))) {
        jump();
    }

    // 3. Move FIRST using the current velocity, then integrate gravity/drag
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

    // 4. Vertical collision response: when the sweep clips the intended Y move
    // (floor or ceiling), zero velocity.y. Without this reset velocity.y kept
    // integrating downward while merely standing, saturating toward terminal
    // velocity; the instant the player stepped off a ledge that stored speed
    // dropped them a whole block in a single tick ("instantaneous fall").
    bool yBlocked = std::abs(resolvedMovement.y - velocity.y) > 1e-6f;
    onGround = yBlocked && velocity.y < 0.f;
    if (yBlocked) {
        velocity.y = 0.f;
    }
    blockedHorizontally = std::abs(resolvedMovement.x - velocity.x) > 1e-6f ||
                          std::abs(resolvedMovement.z - velocity.z) > 1e-6f;

    // Landing while descending ends flight: onGround needs a clipped
    // downward move, and the only downward move in fly mode is the descend
    // key — so this fires exactly on "touched ground while descending".
    if (flying && onGround) {
        flying = false;
    }

    // 5. Track fall distance in BLOCKS actually descended (the old ceil(·×20)
    // summed velocities, so a 1-block fall registered as ~17 blocks of damage).
    // Flight has no falls and water breaks them — sinking used to bank
    // distance, hurting deep divers on touchdown depending on the 100-tick
    // reset's phase.
    if (flying || inWater) {
        resetFallDistance();
    } else if (resolvedMovement.y < 0.f) {
        fallDistance += -resolvedMovement.y;
    }

    // If on ground, apply fall damage and reset
    if (onGround) {
        if (fallDistance > 3.f) {
            applyFallDamage();
        }
        resetFallDistance();
    }

    // 6. Integrate gravity, drag, buoyancy, and terminal velocity for the next
    // tick (reduced gravity + buoyancy in water). Fly mode skips it all —
    // vertical velocity there is a direct intent, not an integrated state.
    if (!flying) {
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
    }

    // 7. Reset fall distance every 100 ticks to avoid FP drift
    fallResetTimer++;
    if (fallResetTimer >= 100) {
        resetFallDistance();
        fallResetTimer = 0;
    }

    // 8. Decrement jump cooldown
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
