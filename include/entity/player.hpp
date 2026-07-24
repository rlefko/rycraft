#pragma once

#include <common/math.hpp>

#include <world/world.hpp>

// Forward declaration
class PhysicsEngine;

// ---------------------------------------------------------------------------
// PlayerInput — Per-tick movement intents, decoded from bindings by the
// engine. A plain snapshot (not InputState) keeps Player free of the input
// layer and makes every movement-mode transition unit-testable as pure data.
// ---------------------------------------------------------------------------
struct PlayerInput {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool jumpHeld = false;         // jump key held (auto-jump, swim up, fly ascend)
    bool jumpPressed = false;      // jump key edge this tick
    bool sprintHeld = false;       // sprint key held (Ctrl)
    bool descendHeld = false;      // sneak key held (Shift) — fly descend
    bool doubleTapForward = false; // forward double-tapped this tick → sprint
    bool doubleTapJump = false;    // jump double-tapped this tick → fly toggle
    // Engine-decoded game-mode gates, so the entity layer never sees
    // GameMode. Defaults preserve the standalone physics contract the tests
    // pin; the engine narrows them per mode each tick.
    bool allowFlight = true;     // creative only; false also clears active flight
    bool takesFallDamage = true; // survival only
};

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
    // Horizontal sweep clipped last tick (wall contact) — drives the
    // water-exit hop, mirroring how onGround is last tick's Y-clip.
    bool blockedHorizontally = false;
    int health = 20;
    float fallDistance = 0.f; // blocks descended since last landing/reset
    int jumpCooldown = 0;
    int fallResetTimer = 0;

    // Movement modes. sprinting is a latch: set by a forward double-tap or
    // the held sprint key, cleared the tick forward is released. swimming is
    // DERIVED each tick (sprinting && in water && !flying) so every water
    // boundary case — sprinting into a pond, swimming out onto shore —
    // resolves from one rule. flying toggles on a jump double-tap and
    // survives pauses/focus loss (Player state is never cleared by the
    // input layer).
    bool flying = false;
    bool sprinting = false;
    bool swimming = false;

    // Player dimensions: 0.6 wide, 1.8 tall
    static constexpr float WIDTH = 0.6f;
    static constexpr float HEIGHT = 1.8f;
    static constexpr float MAX_FEET_Y = static_cast<float>(WORLD_MAX_Y + 1) - HEIGHT;

    // Physics constants — authentic vanilla Minecraft: gravity -0.08 blocks/tick²,
    // vertical drag 0.98, terminal velocity -3.92 blocks/tick (= 49 × gravity),
    // jump velocity 0.42 (giving a ~1.25-block apex that clears a 1-block step).
    static constexpr float GRAVITY = -0.08f;
    static constexpr float HORIZONTAL_DRAG_AIR = 0.91f;
    static constexpr float HORIZONTAL_DRAG_GROUND = 0.546f;
    static constexpr float VERTICAL_DRAG = 0.98f;
    static constexpr float TERMINAL_VELOCITY = -3.92f;
    static constexpr float JUMP_VELOCITY = 0.42f;
    static constexpr float SPRINT_MULTIPLIER = 1.3f;
    // Blocks per tick (0.216 × 20 Hz ≈ 4.3 blocks/s, classic voxel-game
    // walking pace). This was 0.05 with ground drag shrinking it further to
    // ~0.55 blocks/s — slow enough to read as "the player can't move".
    static constexpr float WALK_SPEED = 0.216f;

    // Movement-mode tuning; speeds in blocks/tick at 20 Hz unless noted.
    static constexpr float FLY_SPEED_MULTIPLIER = 2.5f;  // of WALK_SPEED ≈10.8 blocks/s
    static constexpr float FLY_VERTICAL_SPEED = 0.375f;  // ≈7.5 blocks/s ascend/descend
    static constexpr float SWIM_SPEED = 0.28f;           // look-direction swim ≈5.6 blocks/s
    static constexpr float SWIM_UP_ACCELERATION = 0.04f; // hold-jump float-up, per tick
    static constexpr float SWIM_UP_MAX_SPEED = 0.12f;    // ascent cap ≈2.4 blocks/s
    // Upward pop when paddling against a bank — without it a swimmer can
    // face a 1-block shore forever, since jump() needs solid ground. Water
    // fills whole cells here, so a shore sits a full block above the
    // surface: jump strength clears it where Minecraft's 0.3 fell short.
    static constexpr float WATER_EXIT_HOP = JUMP_VELOCITY;

    // Camera height above the feet. position is the AABB bottom; rendering
    // from it puts the horizon at ankle level and reads as being sunk
    // waist-deep into the ground.
    static constexpr float EYE_HEIGHT = 1.62f;
    static constexpr int JUMP_COOLDOWN_TICKS = 10;

    // Get player's AABB based on current position
    AABB getAABB() const;

    // Physics tick: update movement modes, apply input, gravity, drag,
    // collision
    void tick(World& world, const PlayerInput& in);

    // Jump: apply +0.42 velocity when on ground, with cooldown
    void jump();

    // Apply fall damage: ceil(fallDistance - 3) hearts
    void applyFallDamage();

    // Reset fall distance tracking
    void resetFallDistance();
};
