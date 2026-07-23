#pragma once

#include "world/chunk.hpp"
#include "world/item.hpp"

#include <common/math.hpp>

#include <vector>

class World;

// ---------------------------------------------------------------------------
// Dropped items - the collectible entities a broken block or a Q-drop leaves
// in the world. Deliberately NOT an Entity: dropped items carry no AI,
// territory, or breeding, and must never spend the 64-animal budget. A
// lightweight value struct plus a bounded manager, owned by the engine.
//
// Physics mirrors the animal path (gravity, drag, per-axis sweep) but with a
// resting early-out so a pile of grounded items costs almost nothing. Items
// are never persisted: they despawn on quit and on a world switch.
// ---------------------------------------------------------------------------

struct ItemEntity {
    ItemStack stack;
    Vec3 position; // AABB bottom center
    Vec3 velocity;
    int ageTicks = 0;
    int pickupDelay = 10; // ticks before pickup and merge eligibility
    bool onGround = false;
    uint8_t renderPackedLight = FULL_SKY_PACKED_LIGHT; // fixed-tick sky/block lighting probe

    static constexpr float SIZE = 0.25f;
    static constexpr int DESPAWN_TICKS = 6000; // 5 minutes at 20 Hz

    AABB getAABB() const;
    // Gravity, drag, water buoyancy, and a per-axis sweep. Resting items skip
    // the sweep entirely. Never force-loads: missing cubes are solid, so a
    // scattered item freezes at an unloaded boundary instead of falling
    // through the world.
    void tick(World& world);
};

class ItemEntityManager {
public:
    static constexpr size_t MAX_ITEMS = 128;
    static constexpr float ACTIVE_RADIUS = 96.f; // matches the animal sim radius
    static constexpr float MERGE_RADIUS = 0.75f;
    static constexpr float PICKUP_XZ = 0.5f; // player AABB inflation for pickup
    static constexpr float PICKUP_Y = 0.5f;
    static constexpr int MERGE_SCAN_INTERVAL = 20; // one O(n^2) merge pass per second

    // At MAX_ITEMS the oldest grounded item is evicted (falling back to the
    // oldest of any), never the fresh spawn.
    void spawn(const ItemStack& stack, const Vec3& position, const Vec3& velocity,
               int pickupDelay = 10);

    // Physics, merge, aging, and despawn. Items beyond ACTIVE_RADIUS of the
    // player freeze completely (no physics, no aging), like distant animals.
    void tick(World& world, const Vec3& playerPosition);

    // Renderer reads this; the engine's pickup pass mutates it and calls
    // compact() to drop emptied entries.
    const std::vector<ItemEntity>& items() const { return items_; }
    std::vector<ItemEntity>& items() { return items_; }
    void compact();
    void clear();

private:
    std::vector<ItemEntity> items_;
    int mergeCountdown_ = MERGE_SCAN_INTERVAL;
};
