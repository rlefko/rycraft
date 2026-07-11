#pragma once

#include <common/math.hpp>
#include <engine/input.hpp>

#include <world/world.hpp>

// Forward declaration
class PhysicsEngine;

// ---------------------------------------------------------------------------
// Player — First-person character controller with Minecraft-tuned physics
//
// Dimensions: 0.6 wide × 1.8 tall (vanilla Minecraft player hitbox).
// Physics values tuned to match vanilla Minecraft tick rate (20 ticks/sec).
// ---------------------------------------------------------------------------
class Player {
public:
    Vec3 position;
    Vec3 velocity;
    float yaw = 0.f;
    float pitch = 0.f;
    bool onGround = false;
    int health = 20;
    int fallDistance = 0;
    int jumpCooldown = 0;
    int fallResetTimer = 0;

    // Player dimensions: 0.6 wide, 1.8 tall
    static constexpr float WIDTH = 0.6f;
    static constexpr float HEIGHT = 1.8f;

    // Physics constants (Minecraft-tuned)
    static constexpr float GRAVITY = -0.08f;
    static constexpr float HORIZONTAL_DRAG_AIR = 0.91f;
    static constexpr float HORIZONTAL_DRAG_GROUND = 0.546f;
    static constexpr float VERTICAL_DRAG = 0.98f;
    static constexpr float TERMINAL_VELOCITY = -3.92f;
    static constexpr float JUMP_VELOCITY = 0.42f;
    static constexpr float SPRINT_MULTIPLIER = 1.3f;
    static constexpr float WALK_SPEED = 0.05f;
    static constexpr int JUMP_COOLDOWN_TICKS = 10;

    // Get player's AABB based on current position
    AABB getAABB() const;

    // Physics tick: apply input, gravity, drag, collision
    void tick(World& world, const InputState& input, bool sprinting);

    // Jump: apply +0.42 velocity when on ground, with cooldown
    void jump();

    // Apply fall damage: ceil(fallDistance - 3) hearts
    void applyFallDamage();

    // Reset fall distance tracking
    void resetFallDistance();
};
