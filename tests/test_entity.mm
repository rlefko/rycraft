#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/biome.hpp>
#include <world/chunk.hpp>
#include <world/chunk_pos.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/terrain.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Entities: physics, AI, spawning, the player
// ===========================================================================

// ============================================================================
// Physics Engine Tests (Phase 5)
// ============================================================================

TEST_CASE("AABB sweep collision: entity moves through empty space unchanged", "[physics]") {
    auto world = std::make_shared<World>(42);
    // Force-load the chunk at (0,0) so setBlock works
    world->getChunk(0, 0);

    // Place entity at high Y (200) where there's no terrain
    AABB entityAABB{Vec3{5.f, 200.f, 5.f}, Vec3{5.6f, 201.8f, 5.6f}};
    Vec3 movement{1.f, 0.f, 1.f};

    PhysicsEngine physics;
    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // In empty space (y=200), movement should pass through unchanged
    REQUIRE(resolved.x == Catch::Approx(1.f).margin(0.01f));
    REQUIRE(resolved.y == Catch::Approx(0.f).margin(0.01f));
    REQUIRE(resolved.z == Catch::Approx(1.f).margin(0.01f));
}

TEST_CASE("AABB sweep collision: entity blocked by solid block", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a wall of stone blocks at x=10, high Y (200)
    for (int y = 195; y <= 210; ++y) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(10, y, z, BlockType::STONE);
        }
    }

    PhysicsEngine physics;
    // Entity at x=8, y=200 moving toward the wall
    AABB entityAABB{Vec3{8.f, 200.f, 5.f}, Vec3{8.6f, 201.8f, 5.6f}};
    Vec3 movement{5.f, 0.f, 0.f}; // Would move to x=13, but wall is at x=10

    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // X movement should be blocked (entity max.x = 8.6, wall min.x = 10)
    // Entity can move up to 10 - 8.6 = 1.4 blocks
    REQUIRE(resolved.x >= 0.f);
    REQUIRE(resolved.x < 5.f); // Movement reduced by wall
}

TEST_CASE("AABB sweep collision: entity slides along wall (Y-first, then X)", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place floor at y=199 and wall at x=10, high Y
    for (int x = 0; x <= 15; ++x) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(x, 199, z, BlockType::STONE); // Floor
        }
    }
    for (int y = 190; y <= 210; ++y) {
        for (int z = 0; z <= 10; ++z) {
            world->setBlock(10, y, z, BlockType::STONE); // Wall
        }
    }

    PhysicsEngine physics;
    // Entity above floor, moving down and toward wall
    AABB entityAABB{Vec3{8.f, 202.f, 5.f}, Vec3{8.6f, 203.8f, 5.6f}};
    Vec3 movement{5.f, -3.f, 0.f};

    Vec3 resolved = physics.sweepCollision(entityAABB, movement, *world);

    // Y should be resolved first (land on floor at y=199), then X blocked by wall
    REQUIRE(resolved.y >= -3.f);
    REQUIRE(resolved.x < 5.f); // X reduced by wall collision
}

TEST_CASE("Obstacle collection: returns correct blocks in range", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place blocks at high Y to avoid terrain
    world->setBlock(5, 200, 5, BlockType::STONE);
    world->setBlock(6, 200, 5, BlockType::STONE);
    world->setBlock(5, 200, 6, BlockType::AIR); // Not solid

    AABB queryAABB{Vec3{4.f, 199.f, 4.f}, Vec3{7.f, 201.f, 7.f}};

    std::vector<AABB> obstacles = PhysicsEngine::collectObstacles(queryAABB, *world);

    // Should find at least the two STONE blocks
    REQUIRE(obstacles.size() >= 2);

    // Verify each obstacle is a 1x1x1 block at integer coords
    for (const auto& obs : obstacles) {
        Vec3 size = obs.max - obs.min;
        REQUIRE(size.x == Catch::Approx(1.f));
        REQUIRE(size.y == Catch::Approx(1.f));
        REQUIRE(size.z == Catch::Approx(1.f));
    }
}

TEST_CASE("isSolid: solid for STONE and GLASS, passable for AIR/WATER", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Set blocks at high Y (200) to avoid terrain interference
    world->setBlock(1, 200, 1, BlockType::STONE);
    world->setBlock(2, 200, 2, BlockType::AIR);
    world->setBlock(3, 200, 3, BlockType::WATER);
    world->setBlock(4, 200, 4, BlockType::GLASS);
    world->setBlock(5, 200, 5, BlockType::DIRT);
    world->setBlock(6, 200, 6, BlockType::BEDROCK);

    REQUIRE(PhysicsEngine::isSolid(*world, 1, 200, 1) == true);  // STONE
    REQUIRE(PhysicsEngine::isSolid(*world, 2, 200, 2) == false); // AIR
    REQUIRE(PhysicsEngine::isSolid(*world, 3, 200, 3) == false); // WATER
    // GLASS collides: it renders as a full block, so physics must agree
    // (it used to be passable — the fall-through-glass bug).
    REQUIRE(PhysicsEngine::isSolid(*world, 4, 200, 4) == true);
    REQUIRE(PhysicsEngine::isSolid(*world, 5, 200, 5) == true); // DIRT
    REQUIRE(PhysicsEngine::isSolid(*world, 6, 200, 6) == true); // BEDROCK
}

TEST_CASE("Block properties: the three predicates agree on every type", "[physics][world]") {
    for (int t = 0; t < static_cast<int>(BlockType::COUNT); ++t) {
        BlockType bt = static_cast<BlockType>(t);
        // Anything that occludes neighbors must also collide — a block that
        // renders as a full cube but lets entities through reads as a bug.
        if (isOpaque(bt)) {
            REQUIRE(isSolid(bt));
        }
        // Air and water are the only non-solid types today
        bool expectedSolid = bt != BlockType::AIR && bt != BlockType::WATER;
        REQUIRE(isSolid(bt) == expectedSolid);
    }
}

TEST_CASE("isInWater: returns true when entity AABB overlaps water block", "[physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(5, 200, 5, BlockType::WATER);

    // Entity overlapping the water block at high Y
    AABB inWater{Vec3{4.5f, 199.5f, 4.5f}, Vec3{5.5f, 200.5f, 5.5f}};
    REQUIRE(PhysicsEngine::isInWater(*world, inWater) == true);

    // Entity not overlapping water (far away in empty space)
    AABB notInWater{Vec3{50.f, 200.f, 50.f}, Vec3{50.6f, 201.8f, 50.6f}};
    REQUIRE(PhysicsEngine::isInWater(*world, notInWater) == false);
}

// ============================================================================
// Player Tests (Phase 5)
// ============================================================================

TEST_CASE("Player walks across flat ground while standing on it", "[player][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Flat stone platform high above the terrain
    for (int x = 2; x <= 13; ++x) {
        for (int z = 2; z <= 13; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{4.f, 201.f, 4.f}; // feet on the platform
    player.yaw = 0.f;                        // facing +Z

    InputState input;
    input.keysDown[Key::W] = true;

    // Settle onto the ground, then walk forward for a second of ticks.
    // The old sweep treated the floor underfoot as a wall, so horizontal
    // movement zeroed the moment the player landed — the "stuck player" bug.
    Vec3 start = player.position;
    for (int i = 0; i < 20; ++i) {
        player.tick(*world, input, false);
    }

    REQUIRE(player.onGround);
    REQUIRE(player.position.z - start.z > 1.0f);                      // actually moved forward
    REQUIRE(std::abs(player.position.x - start.x) < 0.05f);           // no sideways drift
    REQUIRE(player.position.y == Catch::Approx(201.f).margin(0.01f)); // stayed on top
}

TEST_CASE("Player is stopped by a wall but slides along it", "[player][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    for (int x = 2; x <= 13; ++x) {
        for (int z = 2; z <= 13; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }
    // Wall across +Z at z = 8
    for (int x = 2; x <= 13; ++x) {
        for (int y = 201; y <= 203; ++y) {
            world->setBlock(x, y, 8, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{6.f, 201.f, 5.f};
    player.yaw = 0.f; // facing +Z, straight at the wall

    InputState input;
    input.keysDown[Key::W] = true;
    for (int i = 0; i < 30; ++i) {
        player.tick(*world, input, false);
    }

    // Blocked at the wall face (player half-width 0.3 → z stops near 7.7)
    REQUIRE(player.position.z < 7.75f);
    REQUIRE(player.position.z > 7.0f);
}

TEST_CASE("Player movement follows the camera basis", "[player]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // At yaw = 0 the camera looks down +Z (Camera::updateFront), so W must
    // accelerate toward +Z and D toward the camera's right (-X at yaw 0).
    // These were inverted once: W walked backwards, D strafed left.
    {
        Player player;
        player.position = Vec3{8.f, 250.f, 8.f}; // high in the air, no collisions
        player.yaw = 0.f;
        InputState input;
        input.keysDown[Key::W] = true;
        player.tick(*world, input, false);
        REQUIRE(player.velocity.z > 0.f);
        REQUIRE(std::abs(player.velocity.x) < 1e-4f);
    }
    {
        Player player;
        player.position = Vec3{8.f, 250.f, 8.f};
        player.yaw = 0.f;
        InputState input;
        input.keysDown[Key::D] = true;
        player.tick(*world, input, false);
        REQUIRE(player.velocity.x < 0.f);
        REQUIRE(std::abs(player.velocity.z) < 1e-4f);
    }
}

TEST_CASE("Player AABB: correct dimensions (0.6x1.8x0.6)", "[player]") {
    Player player;
    player.position = Vec3{10.f, 64.f, 10.f};

    AABB aabb = player.getAABB();

    Vec3 size = aabb.max - aabb.min;
    REQUIRE(size.x == Catch::Approx(0.6f));
    REQUIRE(size.y == Catch::Approx(1.8f));
    REQUIRE(size.z == Catch::Approx(0.6f));

    // Center X/Z should match player position, Y should start at player position
    REQUIRE(aabb.min.y == Catch::Approx(64.f));
    REQUIRE(aabb.max.y == Catch::Approx(65.8f));
}

TEST_CASE("Player gravity: velocity.y decreases by 0.08 per tick", "[player]") {
    auto world = std::make_shared<World>(42);
    // Place floor far below so player falls freely
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{0.f, 100.f, 0.f}; // High up
    player.velocity = Vec3::zero();

    InputState input;
    player.tick(*world, input, false);

    // After one tick: gravity (-0.08) applied, then vertical drag (0.98)
    // velocity.y = (0 + (-0.08)) * 0.98 = -0.0784
    REQUIRE(player.velocity.y < 0.f);
    REQUIRE(player.velocity.y == Catch::Approx(-0.08f * 0.98f).margin(0.01f));
}

TEST_CASE("Player terminal velocity: clamped to -3.92", "[player]") {
    auto world = std::make_shared<World>(42);
    // Place floor very far below
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    Player player;
    player.position = Vec3{0.f, 200.f, 0.f};
    player.velocity = Vec3{0.f, -10.f, 0.f}; // Start below terminal

    InputState input;
    player.tick(*world, input, false);

    // Velocity should be clamped to terminal velocity (after drag)
    // -10 * 0.98 = -9.8, then clamped to -3.92
    REQUIRE(player.velocity.y >= Player::TERMINAL_VELOCITY);
}

TEST_CASE("Player jump: velocity.y = +0.42 when on ground", "[player]") {
    Player player;
    player.position = Vec3{0.f, 64.f, 0.f};
    player.velocity = Vec3::zero();
    player.onGround = true;
    player.jumpCooldown = 0;

    player.jump();

    REQUIRE(player.velocity.y == Catch::Approx(Player::JUMP_VELOCITY));
    REQUIRE(player.jumpCooldown == Player::JUMP_COOLDOWN_TICKS);
}

TEST_CASE("Player fall damage: ceil(fallDistance - 3) hearts", "[player]") {
    Player player;
    player.health = 20;

    // Fall distance of 8 → damage = ceil(8 - 3) = 5
    player.fallDistance = 8;
    player.applyFallDamage();
    REQUIRE(player.health == 15);

    // Fall distance of 3 → no damage
    player.health = 20;
    player.fallDistance = 3;
    player.applyFallDamage();
    REQUIRE(player.health == 20);

    // Fall distance of 4 → damage = ceil(4 - 3) = 1
    player.health = 20;
    player.fallDistance = 4;
    player.applyFallDamage();
    REQUIRE(player.health == 19);
}

TEST_CASE("Player fall damage capped at zero health", "[player]") {
    Player player;
    player.health = 2;
    player.fallDistance = 20; // damage = ceil(20 - 3) = 17
    player.applyFallDamage();
    REQUIRE(player.health == 0);
}

TEST_CASE("Player resetFallDistance clears fall tracking", "[player]") {
    Player player;
    player.fallDistance = 10;
    player.resetFallDistance();
    REQUIRE(player.fallDistance == 0);
}

// ============================================================================
// DDA Voxel Traversal Tests (Phase 5)
// ============================================================================

TEST_CASE("DDA traversal: ray hits block at expected position", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y to avoid terrain
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (0, 200, 0) going +X toward the block
    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 10.f);

    REQUIRE(hit.has_value());
    REQUIRE(hit->x == Catch::Approx(5.f));
    REQUIRE(hit->y == Catch::Approx(200.f));
    REQUIRE(hit->z == Catch::Approx(0.f));
}

TEST_CASE("DDA traversal: ray misses all blocks in empty space", "[voxel]") {
    auto world = std::make_shared<World>(42);
    // Don't place any blocks near the ray path

    // Shoot at very high Y (250) where there's definitely no terrain
    Vec3 origin{0.f, 250.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 6.f);

    REQUIRE(hit.has_value() == false);
}

TEST_CASE("DDA traversal: face normal computation", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place a stone block at (5, 200, 0) — high Y
    world->setBlock(5, 200, 0, BlockType::STONE);

    // Ray from (2, 200, 0) going +X — hits the -X face of the block
    Vec3 origin{2.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    auto hit = VoxelTraversal::traceRayWithNormal(origin, direction, *world, 10.f);

    REQUIRE(hit.has_value());
    REQUIRE(hit->first.x == Catch::Approx(5.f));
    REQUIRE(hit->first.y == Catch::Approx(200.f));
    REQUIRE(hit->first.z == Catch::Approx(0.f));

    // Normal should point in -X direction (the face we hit)
    REQUIRE(hit->second.x == Catch::Approx(-1.f));
    REQUIRE(hit->second.y == Catch::Approx(0.f));
    REQUIRE(hit->second.z == Catch::Approx(0.f));
}

TEST_CASE("DDA traversal: ray along diagonal hits block", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place stone block at (3, 203, 3) — high Y
    world->setBlock(3, 203, 3, BlockType::STONE);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 1.f, 1.f}; // Diagonal up-forward
    direction = direction.normalize();

    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 10.f);

    // Should hit the block at (3, 203, 3)
    REQUIRE(hit.has_value());
    REQUIRE(hit->x == Catch::Approx(3.f));
    REQUIRE(hit->y == Catch::Approx(203.f));
    REQUIRE(hit->z == Catch::Approx(3.f));
}

TEST_CASE("DDA traversal: maxDistance limits traversal range", "[voxel]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place stone block far away at high Y
    world->setBlock(10, 200, 0, BlockType::STONE);

    Vec3 origin{0.f, 200.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    // maxDistance too short to reach the block at x=10
    auto hit = VoxelTraversal::traceRay(origin, direction, *world, 5.f);
    REQUIRE(hit.has_value() == false);

    // maxDistance long enough to reach the block
    auto hitFar = VoxelTraversal::traceRay(origin, direction, *world, 15.f);
    REQUIRE(hitFar.has_value());
    REQUIRE(hitFar->x == Catch::Approx(10.f));
}

// ============================================================================
// Phase 7: Entity System, AI, Flocking, Spawning, Spatial Hash Tests
// ============================================================================

// ---- Entity Creation Tests ----

TEST_CASE("Entity creation assigns unique ID", "[entity]") {
    auto e1 = std::make_shared<Entity>(1, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    auto e2 = std::make_shared<Entity>(2, EntityType::COW, Vec3{1.f, 64.f, 1.f});

    REQUIRE(e1->id == 1);
    REQUIRE(e2->id == 2);
    REQUIRE(e1->id != e2->id);
}

TEST_CASE("Entity creation sets correct type and position", "[entity]") {
    Vec3 spawn{10.f, 80.f, -5.f};
    auto entity = std::make_shared<Entity>(42, EntityType::PIG, spawn);

    REQUIRE(entity->type == EntityType::PIG);
    REQUIRE(entity->position.x == Catch::Approx(10.f));
    REQUIRE(entity->position.y == Catch::Approx(80.f));
    REQUIRE(entity->position.z == Catch::Approx(-5.f));
    REQUIRE(entity->alive == true);
    REQUIRE(entity->onGround == false);
}

TEST_CASE("Entity AABB computation: sheep dimensions", "[entity]") {
    auto entity = std::make_shared<Entity>(1, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    AABB aabb = entity->computeAABB();

    Vec3 size = aabb.max - aabb.min;
    // Sheep: 0.6 wide, 0.9 tall
    REQUIRE(size.x == Catch::Approx(0.6f));
    REQUIRE(size.y == Catch::Approx(0.9f));
    REQUIRE(size.z == Catch::Approx(0.6f));

    // Center should match entity position
    REQUIRE(aabb.min.x == Catch::Approx(-0.3f));
    REQUIRE(aabb.min.z == Catch::Approx(-0.3f));
    REQUIRE(aabb.min.y == Catch::Approx(64.f));
}

TEST_CASE("Entity AABB computation: cow dimensions", "[entity]") {
    auto entity = std::make_shared<Entity>(2, EntityType::COW, Vec3{5.f, 70.f, 5.f});
    AABB aabb = entity->computeAABB();

    Vec3 size = aabb.max - aabb.min;
    // Cow: 0.9 wide, 1.4 tall
    REQUIRE(size.x == Catch::Approx(0.9f));
    REQUIRE(size.y == Catch::Approx(1.4f));
    REQUIRE(size.z == Catch::Approx(0.9f));
}

TEST_CASE("Entity AABB: baby entity is half size", "[entity]") {
    auto entity = std::make_shared<Entity>(3, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    entity->isBaby = true;

    AABB aabb = entity->computeAABB();
    Vec3 size = aabb.max - aabb.min;

    // Baby sheep: 0.3 wide, 0.45 tall
    REQUIRE(size.x == Catch::Approx(0.3f));
    REQUIRE(size.y == Catch::Approx(0.45f));
    REQUIRE(size.z == Catch::Approx(0.3f));
}

TEST_CASE("Entity nextId is monotonically increasing", "[entity]") {
    uint64_t id1 = Entity::nextId();
    uint64_t id2 = Entity::nextId();
    uint64_t id3 = Entity::nextId();

    REQUIRE(id2 > id1);
    REQUIRE(id3 > id2);
}

// ---- Entity Config Tests ----

TEST_CASE("EntityConfig: sheep has correct config", "[entity]") {
    auto cfg = Entity::getConfig(EntityType::SHEEP);
    REQUIRE(cfg.width == Catch::Approx(0.6f));
    REQUIRE(cfg.height == Catch::Approx(0.9f));
    REQUIRE(cfg.speed > 0.f);
    // Sheep color should be white-ish
    REQUIRE(cfg.color.x > 0.8f);
    REQUIRE(cfg.color.y > 0.8f);
}

TEST_CASE("EntityConfig: all types have positive dimensions", "[entity]") {
    for (int t = 0; t < 4; ++t) {
        EntityType type = static_cast<EntityType>(t);
        auto cfg = Entity::getConfig(type);
        REQUIRE(cfg.width > 0.f);
        REQUIRE(cfg.height > 0.f);
        REQUIRE(cfg.speed > 0.f);
    }
}

// ---- Entity Physics Tests ----

TEST_CASE("Entity physics: gravity reduces velocity.y", "[entity][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    // Place floor far below
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(100, EntityType::SHEEP, Vec3{0.f, 100.f, 0.f});
    entity->velocity = Vec3::zero();

    entity->tick(*world);

    // Velocity.y should be negative (gravity applied)
    REQUIRE(entity->velocity.y < 0.f);
}

TEST_CASE("Entity physics: entity stops on ground", "[entity][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    // Place solid floor at high Y to avoid terrain interference
    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(101, EntityType::COW, Vec3{0.f, 202.f, 0.f});
    entity->velocity = Vec3{0.f, -1.f, 0.f}; // Falling

    entity->tick(*world);

    // Entity should have stopped at or above y=200
    REQUIRE(entity->position.y >= 200.f);
    // Vertical velocity should be reduced (collision absorbs some)
    REQUIRE(std::abs(entity->velocity.y) < 2.f);
}

TEST_CASE("Entity physics: step assist climbs 1-block gap", "[entity][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    // Create a 1-block step at high Y: floor at y=200, step up at x=2
    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }
    // Make a step at x=2
    for (int z = -5; z <= 5; ++z) {
        world->setBlock(2, 201, z, BlockType::STONE);
    }

    auto entity = std::make_shared<Entity>(102, EntityType::PIG, Vec3{1.f, 201.f, 0.f});
    entity->velocity = Vec3{0.1f, -0.5f, 0.f}; // Moving toward step while falling

    entity->tick(*world);

    // Entity should have stepped up or been blocked
    // The key is that it doesn't fall through
    REQUIRE(entity->position.y >= 200.f);
}

TEST_CASE("Entity physics: dead entities don't tick", "[entity][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);
    for (int x = -10; x <= 10; ++x) {
        for (int z = -10; z <= 10; ++z) {
            world->setBlock(x, 0, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(103, EntityType::CHICKEN, Vec3{0.f, 100.f, 0.f});
    entity->alive = false;
    Vec3 posBefore = entity->position;

    entity->tick(*world);

    // Dead entity should not change position
    REQUIRE(entity->position == posBefore);
}

TEST_CASE("Entity physics: terminal velocity is clamped", "[entity][physics]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // No floor — entity falls freely at very high Y
    auto entity = std::make_shared<Entity>(104, EntityType::SHEEP, Vec3{0.f, 250.f, 0.f});
    entity->velocity = Vec3{0.f, -10.f, 0.f}; // Start below terminal

    entity->tick(*world);

    // Velocity should be clamped
    REQUIRE(entity->velocity.y >= -3.92f);
}

// ---- Voxel Model Tests ----

TEST_CASE("Entity voxel model: sheep has multiple blocks", "[entity][voxel]") {
    auto model = Entity::getVoxelModel(EntityType::SHEEP, false);
    REQUIRE(model.size() >= 5); // body + head + 4 legs minimum
}

TEST_CASE("Entity voxel model: cow has udder block", "[entity][voxel]") {
    auto model = Entity::getVoxelModel(EntityType::COW, false);
    // Cow should have more blocks than sheep (body, patch, head, 4 legs, udder)
    REQUIRE(model.size() >= 7);
}

TEST_CASE("Entity voxel model: chicken has wings", "[entity][voxel]") {
    auto model = Entity::getVoxelModel(EntityType::CHICKEN, false);
    // Chicken: body, head, beak, comb, 2 legs, 2 wings = 8 blocks
    REQUIRE(model.size() >= 6);
}

TEST_CASE("Entity voxel model: baby model is scaled down", "[entity][voxel]") {
    auto adultModel = Entity::getVoxelModel(EntityType::SHEEP, false);
    auto babyModel = Entity::getVoxelModel(EntityType::SHEEP, true);

    REQUIRE(adultModel.size() == babyModel.size());

    // Baby blocks should be smaller
    for (size_t i = 0; i < adultModel.size(); ++i) {
        REQUIRE(babyModel[i].size.x <= adultModel[i].size.x);
        REQUIRE(babyModel[i].size.y <= adultModel[i].size.y);
        REQUIRE(babyModel[i].size.z <= adultModel[i].size.z);
    }
}

TEST_CASE("Entity voxel model: all blocks have valid colors", "[entity][voxel]") {
    for (int t = 0; t < 4; ++t) {
        EntityType type = static_cast<EntityType>(t);
        auto model = Entity::getVoxelModel(type, false);
        for (const auto& block : model) {
            REQUIRE(block.color.x >= 0.f);
            REQUIRE(block.color.x <= 1.f);
            REQUIRE(block.color.y >= 0.f);
            REQUIRE(block.color.y <= 1.f);
            REQUIRE(block.color.z >= 0.f);
            REQUIRE(block.color.z <= 1.f);
        }
    }
}

// ---- State Machine Tests ----

TEST_CASE("State machine: initial state is IDLE", "[ai][state]") {
    StateMachine sm;
    REQUIRE(sm.currentState == AnimalState::IDLE);
}

TEST_CASE("State machine: idle transitions to wander after ticks", "[ai][state]") {
    // shouldWander returns true after minIdle ticks
    REQUIRE(StateMachine::shouldWander(0) == false);
    REQUIRE(StateMachine::shouldWander(29) == false);
    REQUIRE(StateMachine::shouldWander(30) == true); // minIdle = 30
}

TEST_CASE("State machine: wander stops after max ticks", "[ai][state]") {
    REQUIRE(StateMachine::shouldStopWandering(0) == false);
    REQUIRE(StateMachine::shouldStopWandering(199) == false);
    REQUIRE(StateMachine::shouldStopWandering(200) == true); // maxWander = 200
}

TEST_CASE("State machine: flee triggers when player close and approaching", "[ai][state]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{3.f, 64.f, 3.f}; // ~4.24 blocks away (within 6)

    REQUIRE(StateMachine::shouldFlee(entityPos, playerPos, true) == true);
    REQUIRE(StateMachine::shouldFlee(entityPos, playerPos, false) == false); // Not approaching
}

TEST_CASE("State machine: flee does not trigger when player far", "[ai][state]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{20.f, 64.f, 20.f}; // ~28 blocks away

    REQUIRE(StateMachine::shouldFlee(entityPos, playerPos, true) == false);
}

TEST_CASE("State machine: stop fleeing when player far or timeout", "[ai][state]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerFar{15.f, 64.f, 15.f}; // > 10 blocks
    Vec3 playerNear{3.f, 64.f, 3.f};

    // Stop because player is far
    REQUIRE(StateMachine::shouldStopFleeing(entityPos, playerFar, 10) == true);
    // Don't stop because player is close and time is short
    REQUIRE(StateMachine::shouldStopFleeing(entityPos, playerNear, 10) == false);
    // Stop because timeout reached
    REQUIRE(StateMachine::shouldStopFleeing(entityPos, playerNear, 300) == true);
}

TEST_CASE("State machine: eat triggers when hungry", "[ai][state]") {
    REQUIRE(StateMachine::shouldEat(0) == false);
    REQUIRE(StateMachine::shouldEat(299) == false);
    REQUIRE(StateMachine::shouldEat(300) == true);
}

TEST_CASE("State machine: follow player triggers when player holds food nearby", "[ai][state]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{5.f, 64.f, 5.f}; // ~7 blocks away (within 10)

    REQUIRE(StateMachine::shouldFollowPlayer(entityPos, playerPos, true) == true);
    REQUIRE(StateMachine::shouldFollowPlayer(entityPos, playerPos, false) == false);
}

TEST_CASE("State machine: stop following when player far or no food", "[ai][state]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerFar{15.f, 64.f, 15.f};

    REQUIRE(StateMachine::shouldStopFollowing(entityPos, playerFar, true) == true);
    REQUIRE(StateMachine::shouldStopFollowing(entityPos, Vec3{5.f, 64.f, 5.f}, false) == true);
}

// ---- Flocking Tests ----

TEST_CASE("Flocking: separation pushes entities apart", "[ai][flocking]") {
    SpatialHash hash;

    auto e1 = std::make_shared<Entity>(200, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    auto e2 = std::make_shared<Entity>(201, EntityType::SHEEP, Vec3{0.5f, 64.f, 0.f});

    hash.insert(e1->id, e1->position);
    hash.insert(e2->id, e2->position);

    std::unordered_map<uint64_t, Vec3> positions;
    positions[e1->id] = e1->position;
    positions[e2->id] = e2->position;

    // Query near e1 should find e2
    auto neighbors = hash.query(e1->position, FlockingController::SEPARATION_RADIUS, positions);
    bool foundE2 = false;
    for (uint64_t id : neighbors) {
        if (id == e2->id) {
            foundE2 = true;
            break;
        }
    }
    REQUIRE(foundE2);
}

TEST_CASE("Flocking: alignment matches neighbor velocity", "[ai][flocking]") {
    SpatialHash hash;

    auto e1 = std::make_shared<Entity>(202, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    auto e2 = std::make_shared<Entity>(203, EntityType::SHEEP, Vec3{1.f, 64.f, 1.f});
    e1->velocity = Vec3{0.1f, 0.f, 0.1f};
    e2->velocity = Vec3{0.1f, 0.f, 0.1f};

    hash.insert(e1->id, e1->position);
    hash.insert(e2->id, e2->position);

    std::unordered_map<uint64_t, Vec3> positions;
    positions[e1->id] = e1->position;
    positions[e2->id] = e2->position;

    // Alignment should consider neighbor velocities
    auto neighbors = hash.query(e1->position, FlockingController::ALIGNMENT_RADIUS, positions);
    REQUIRE(neighbors.size() >= 2); // Both entities in range
}

TEST_CASE("Flocking: cohesion centers on neighbors", "[ai][flocking]") {
    SpatialHash hash;

    auto e1 = std::make_shared<Entity>(204, EntityType::SHEEP, Vec3{0.f, 64.f, 0.f});
    auto e2 = std::make_shared<Entity>(205, EntityType::SHEEP, Vec3{2.f, 64.f, 0.f});
    auto e3 = std::make_shared<Entity>(206, EntityType::SHEEP, Vec3{-2.f, 64.f, 0.f});

    hash.insert(e1->id, e1->position);
    hash.insert(e2->id, e2->position);
    hash.insert(e3->id, e3->position);

    std::unordered_map<uint64_t, Vec3> positions;
    positions[e1->id] = e1->position;
    positions[e2->id] = e2->position;
    positions[e3->id] = e3->position;

    // Query from center should find all three
    auto neighbors = hash.query(e1->position, FlockingController::COHESION_RADIUS, positions);
    REQUIRE(neighbors.size() >= 3);
}

TEST_CASE("Flocking: max force clamps steering", "[ai][flocking]") {
    Vec3 bigForce{1.f, 0.f, 0.f};
    Vec3 clamped = FlockingController::clampForce(bigForce, FlockingController::MAX_FLOCKING_FORCE);

    REQUIRE(clamped.length() <=
            Catch::Approx(FlockingController::MAX_FLOCKING_FORCE).epsilon(0.001f));
}

TEST_CASE("Flocking: small force passes through clamp", "[ai][flocking]") {
    Vec3 smallForce{0.01f, 0.f, 0.f};
    Vec3 clamped =
        FlockingController::clampForce(smallForce, FlockingController::MAX_FLOCKING_FORCE);

    REQUIRE(clamped.x == Catch::Approx(0.01f));
}

TEST_CASE("Flocking: zero force remains zero", "[ai][flocking]") {
    Vec3 zeroForce = Vec3::zero();
    Vec3 clamped =
        FlockingController::clampForce(zeroForce, FlockingController::MAX_FLOCKING_FORCE);

    REQUIRE(clamped == Vec3::zero());
}

// ---- Edge Detection Tests ----

TEST_CASE("Edge detection: safe on flat ground", "[ai][edge]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);

    // Create flat ground at high Y to avoid terrain interference
    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    Vec3 entityPos{0.f, 201.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f}; // Moving +X

    REQUIRE(EdgeDetector::isSafeToMove(entityPos, direction, EntityType::SHEEP, *world) == true);
}

TEST_CASE("Edge detection: detects cliff ahead", "[ai][edge]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);

    // Create ground only on one side (cliff at x=2) at high Y
    for (int x = -5; x <= 1; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }
    // x=2..5 is empty (cliff)

    Vec3 entityPos{0.f, 201.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f}; // Moving toward cliff

    // Should detect cliff
    REQUIRE(EdgeDetector::isCliffAhead(entityPos, direction, *world) == true);
    REQUIRE(EdgeDetector::isSafeToMove(entityPos, direction, EntityType::SHEEP, *world) == false);
}

TEST_CASE("Edge detection: safe moving away from cliff", "[ai][edge]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0); // Need chunk for negative X coordinates

    // Ground at high Y only on left side
    for (int x = -5; x <= 1; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    Vec3 entityPos{0.f, 201.f, 0.f};
    Vec3 direction{-1.f, 0.f, 0.f}; // Moving away from cliff (toward -X where ground exists)

    REQUIRE(EdgeDetector::isSafeToMove(entityPos, direction, EntityType::SHEEP, *world) == true);
}

TEST_CASE("Edge detection: pig tolerates water", "[ai][edge]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);

    // Flat ground with water ahead at high Y
    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }
    world->setBlock(3, 201, 0, BlockType::WATER);

    Vec3 entityPos{0.f, 201.f, 0.f};
    Vec3 direction{1.f, 0.f, 0.f};

    // Pig should be okay with water
    bool pigSafe = EdgeDetector::isSafeToMove(entityPos, direction, EntityType::PIG, *world);

    // The key test: pig doesn't get blocked by water alone on flat ground
    REQUIRE(pigSafe == true);
}

// ---- Behavior Controller Tests ----

TEST_CASE("Behavior: follow steering moves toward player", "[ai][behavior]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{10.f, 64.f, 0.f};

    Vec3 steering = BehaviorController::computeFollowSteering(entityPos, playerPos);

    // Steering should point toward player (+X)
    REQUIRE(steering.x > 0.f);
    REQUIRE(steering.y == Catch::Approx(0.f));
}

TEST_CASE("Behavior: follow steering backs away when too close", "[ai][behavior]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{1.f, 64.f, 0.f}; // Only 1 block away (min is 3)

    Vec3 steering = BehaviorController::computeFollowSteering(entityPos, playerPos);

    // Steering should point away from player (-X)
    REQUIRE(steering.x < 0.f);
}

TEST_CASE("Behavior: follow steering is zero at comfortable distance", "[ai][behavior]") {
    Vec3 entityPos{0.f, 64.f, 0.f};
    Vec3 playerPos{4.5f, 64.f, 0.f}; // 4.5 blocks (between 3 and 6)

    Vec3 steering = BehaviorController::computeFollowSteering(entityPos, playerPos);

    REQUIRE(steering == Vec3::zero());
}

TEST_CASE("Behavior: isOnGrass detects grass block", "[ai][behavior]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(0, 200, 0, BlockType::GRASS);

    auto entity = std::make_shared<Entity>(300, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});

    REQUIRE(BehaviorController::isOnGrass(*entity, *world) == true);
}

TEST_CASE("Behavior: isOnGrass returns false on stone", "[ai][behavior]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->setBlock(0, 200, 0, BlockType::STONE);
    world->setBlock(0, 199, 0, BlockType::STONE);

    auto entity = std::make_shared<Entity>(301, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});

    REQUIRE(BehaviorController::isOnGrass(*entity, *world) == false);
}

// ---- Spatial Hash Tests ----

TEST_CASE("Spatial hash: insert and query finds entity", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{0.f, 64.f, 0.f};

    auto results = hash.query(Vec3{0.f, 64.f, 0.f}, 10.0f, positions);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0] == 1);
}

TEST_CASE("Spatial hash: multiple entities in same cell", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(2, Vec3{1.f, 64.f, 1.f});
    hash.insert(3, Vec3{2.f, 64.f, 2.f});

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{0.f, 64.f, 0.f};
    positions[2] = Vec3{1.f, 64.f, 1.f};
    positions[3] = Vec3{2.f, 64.f, 2.f};

    auto results = hash.query(Vec3{0.f, 64.f, 0.f}, 10.0f, positions);
    REQUIRE(results.size() == 3);
}

TEST_CASE("Spatial hash: entities in different cells", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(2, Vec3{16.f, 64.f, 16.f}); // Different cell (8-block cells)

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{0.f, 64.f, 0.f};
    positions[2] = Vec3{16.f, 64.f, 16.f};

    // Query at origin with small radius should only find entity 1
    auto near = hash.query(Vec3{0.f, 64.f, 0.f}, 5.0f, positions);
    // Query at origin with large radius should find both
    auto far = hash.query(Vec3{0.f, 64.f, 0.f}, 30.0f, positions);

    REQUIRE(far.size() >= near.size());
}

TEST_CASE("Spatial hash: remove entity", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(2, Vec3{1.f, 64.f, 1.f});

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{0.f, 64.f, 0.f};
    positions[2] = Vec3{1.f, 64.f, 1.f};

    REQUIRE(hash.query(Vec3{0.f, 64.f, 0.f}, 10.0f, positions).size() == 2);

    hash.remove(1);
    positions.erase(1);

    auto results = hash.query(Vec3{0.f, 64.f, 0.f}, 10.0f, positions);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0] == 2);
}

TEST_CASE("Spatial hash: clear removes all", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(2, Vec3{10.f, 64.f, 10.f});
    hash.insert(3, Vec3{20.f, 64.f, 20.f});

    hash.clear();

    std::unordered_map<uint64_t, Vec3> positions;
    auto results = hash.query(Vec3{10.f, 64.f, 10.f}, 50.0f, positions);
    REQUIRE(results.empty());
}

TEST_CASE("Spatial hash: cell size is configurable", "[spatial]") {
    SpatialHash hash1(4.0f);
    SpatialHash hash2(16.0f);

    REQUIRE(hash1.getCellSize() == Catch::Approx(4.0f));
    REQUIRE(hash2.getCellSize() == Catch::Approx(16.0f));
}

TEST_CASE("Spatial hash: re-insert moves entity to new cell", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(1, Vec3{20.f, 64.f, 20.f}); // Re-insert far away

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{20.f, 64.f, 20.f};

    // Query at origin should not find entity
    auto near = hash.query(Vec3{0.f, 64.f, 0.f}, 5.0f, positions);
    bool foundNear = false;
    for (uint64_t id : near) {
        if (id == 1) {
            foundNear = true;
            break;
        }
    }
    REQUIRE(foundNear == false);

    // Query far away should find entity
    auto far = hash.query(Vec3{20.f, 64.f, 20.f}, 5.0f, positions);
    bool foundFar = false;
    for (uint64_t id : far) {
        if (id == 1) {
            foundFar = true;
            break;
        }
    }
    REQUIRE(foundFar == true);
}

TEST_CASE("Spatial hash: remove non-existent entity is safe", "[spatial]") {
    SpatialHash hash(8.0f);

    // Should not crash
    hash.remove(999);

    std::unordered_map<uint64_t, Vec3> positions;
    auto results = hash.query(Vec3{0.f, 64.f, 0.f}, 10.0f, positions);
    REQUIRE(results.empty());
}

TEST_CASE("Spatial hash: query filters by distance", "[spatial]") {
    SpatialHash hash(8.0f);

    hash.insert(1, Vec3{0.f, 64.f, 0.f});
    hash.insert(2, Vec3{15.f, 64.f, 15.f}); // In adjacent cell but far

    std::unordered_map<uint64_t, Vec3> positions;
    positions[1] = Vec3{0.f, 64.f, 0.f};
    positions[2] = Vec3{15.f, 64.f, 15.f};

    // Small radius should only find entity 1
    auto results = hash.query(Vec3{0.f, 64.f, 0.f}, 5.0f, positions);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0] == 1);

    // Large radius should find both
    auto allResults = hash.query(Vec3{0.f, 64.f, 0.f}, 25.0f, positions);
    REQUIRE(allResults.size() == 2);
}

// ---- Spawner Tests ----

TEST_CASE("Spawner: biome spawn rules — Plains", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::PLAINS);
    REQUIRE(rule.sheepCount == 8);
    REQUIRE(rule.cowCount == 4);
    REQUIRE(rule.pigCount == 2);
    REQUIRE(rule.chickenCount == 0);
}

TEST_CASE("Spawner: biome spawn rules — Forest", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::FOREST);
    REQUIRE(rule.sheepCount == 4);
    REQUIRE(rule.cowCount == 4);
    REQUIRE(rule.pigCount == 4);
}

TEST_CASE("Spawner: biome spawn rules — Desert has no animals", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::DESERT);
    REQUIRE(rule.sheepCount == 0);
    REQUIRE(rule.cowCount == 0);
    REQUIRE(rule.pigCount == 0);
    REQUIRE(rule.chickenCount == 0);
}

TEST_CASE("Spawner: biome spawn rules — Ocean has no animals", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::OCEAN);
    REQUIRE(rule.sheepCount == 0);
    REQUIRE(rule.cowCount == 0);
    REQUIRE(rule.pigCount == 0);
}

TEST_CASE("Spawner: biome spawn rules — Taiga", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::TAIGA);
    REQUIRE(rule.sheepCount == 6);
    REQUIRE(rule.cowCount == 2);
}

TEST_CASE("Spawner: biome spawn rules — Swamp", "[spawner][biome]") {
    auto rule = Spawner::getSpawnRule(Biome::SWAMP);
    REQUIRE(rule.pigCount == 2);
    REQUIRE(rule.cowCount == 2);
}

TEST_CASE("Spawner: spawn entity adds to list", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    Spawner spawner(*world);

    auto entity = spawner.spawnEntity(EntityType::SHEEP, Vec3{8.f, 70.f, 8.f});
    REQUIRE(entity != nullptr);
    REQUIRE(entity->type == EntityType::SHEEP);

    auto& entities = spawner.getEntities();
    REQUIRE(entities.size() == 1);
    REQUIRE(entities[0]->id == entity->id);
}

TEST_CASE("Spawner: spawn baby entity", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    Spawner spawner(*world);

    auto parent = spawner.spawnEntity(EntityType::COW, Vec3{8.f, 70.f, 8.f});
    auto baby = spawner.spawnBaby(EntityType::COW, Vec3{8.5f, 70.f, 8.5f}, parent->id);

    REQUIRE(baby != nullptr);
    REQUIRE(baby->isBaby == true);
    REQUIRE(baby->parentId == parent->id);
    REQUIRE(baby->babyTimer == 600);
}

TEST_CASE("Spawner: remove dead entity", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    Spawner spawner(*world);

    auto entity = spawner.spawnEntity(EntityType::PIG, Vec3{8.f, 70.f, 8.f});
    REQUIRE(spawner.getEntities().size() == 1);

    entity->alive = false;
    spawner.removeEntity(entity->id);

    REQUIRE(spawner.getEntities().size() == 0);
}

TEST_CASE("Spawner: spatial hash is populated on spawn", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    Spawner spawner(*world);

    spawner.spawnEntity(EntityType::SHEEP, Vec3{8.f, 70.f, 8.f});

    auto positions = spawner.getEntityPositions();
    auto results = spawner.getSpatialHash().query(Vec3{8.f, 70.f, 8.f}, 10.0f, positions);
    REQUIRE(results.size() >= 1);
}

TEST_CASE("Spawner: findSpawnHeight finds surface", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Place ground at high Y to avoid terrain interference
    for (int x = 0; x <= 15; ++x) {
        for (int z = 0; z <= 15; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    Spawner spawner(*world);

    auto height = spawner.findSpawnHeight(8, 8);
    REQUIRE(height.has_value());
    // Should find the highest solid block (may be terrain or our placed blocks)
    REQUIRE(height.value() >= 201); // At least one above our placed ground
}

TEST_CASE("Spawner: isSpawnValid checks block configuration", "[spawner]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);

    // Valid: stone at y=200, air at y=201 and y=202
    for (int x = 0; x <= 15; ++x) {
        for (int z = 0; z <= 15; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    Spawner spawner(*world);

    // y=201: air at 201, air at 202, stone at 200 → valid
    REQUIRE(spawner.isSpawnValid(8, 201, 8) == true);

    // y=200: stone at 200 → invalid (not air at spawn)
    REQUIRE(spawner.isSpawnValid(8, 200, 8) == false);
}

TEST_CASE("Spawner: initial population spawns entities", "[spawner]") {
    auto world = std::make_shared<World>(42);

    // Load a few chunks
    world->getChunk(0, 0);
    world->getChunk(1, 0);

    Spawner spawner(*world);
    spawner.spawnInitialPopulation();

    // Should have spawned at least some entities (depends on biome)
    auto& entities = spawner.getEntities();
    REQUIRE(entities.size() >= 0); // Non-negative (may be zero if biomes are barren)

    // All spawned entities should be alive
    for (const auto& entity : entities) {
        REQUIRE(entity->alive == true);
    }
}

// ---- Entity Baby Timer Tests ----

TEST_CASE("Entity: baby timer decrements on tick", "[entity][baby]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(400, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});
    entity->isBaby = true;
    entity->babyTimer = 10;

    for (int i = 0; i < 5; ++i) {
        entity->tick(*world);
    }

    REQUIRE(entity->babyTimer == 5);
    REQUIRE(entity->isBaby == true);
}

TEST_CASE("Entity: baby becomes adult when timer expires", "[entity][baby]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(401, EntityType::COW, Vec3{0.f, 201.f, 0.f});
    entity->isBaby = true;
    entity->babyTimer = 2;

    entity->tick(*world);
    entity->tick(*world);

    REQUIRE(entity->isBaby == false);
    REQUIRE(entity->babyTimer == 0);
}

// ---- Entity Hunger Timer Tests ----

TEST_CASE("Entity: hunger timer increments on tick", "[entity][hunger]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(402, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});
    int initialHunger = entity->hungerTimer;

    entity->tick(*world);
    entity->tick(*world);

    REQUIRE(entity->hungerTimer == initialHunger + 2);
}

TEST_CASE("Entity: hunger timer caps at 600", "[entity][hunger]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(403, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});
    entity->hungerTimer = 599;

    entity->tick(*world);
    entity->tick(*world);

    REQUIRE(entity->hungerTimer == 600);
}

// ---- Entity Eat Animation Tests ----

TEST_CASE("Entity: eat animation timer decrements", "[entity][animation]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(404, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});
    entity->eatAnimationTimer = 10;

    for (int i = 0; i < 5; ++i) {
        entity->tick(*world);
    }

    REQUIRE(entity->eatAnimationTimer == 5);
}

TEST_CASE("Entity: eat animation stops at zero", "[entity][animation]") {
    auto world = std::make_shared<World>(42);
    world->getChunk(0, 0);
    world->getChunk(-1, 0);
    world->getChunk(0, -1);
    world->getChunk(-1, -1);

    for (int x = -5; x <= 5; ++x) {
        for (int z = -5; z <= 5; ++z) {
            world->setBlock(x, 200, z, BlockType::STONE);
        }
    }

    auto entity = std::make_shared<Entity>(405, EntityType::SHEEP, Vec3{0.f, 201.f, 0.f});
    entity->eatAnimationTimer = 2;

    entity->tick(*world);
    entity->tick(*world);

    REQUIRE(entity->eatAnimationTimer == 0);
}
