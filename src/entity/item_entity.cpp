#include "entity/item_entity.hpp"

#include "entity/physics.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float GRAVITY = -0.04f; // lighter than the player: items float down
constexpr float VERTICAL_DRAG = 0.98f;
constexpr float HORIZONTAL_DRAG_AIR = 0.92f;
constexpr float HORIZONTAL_DRAG_GROUND = 0.6f;
constexpr float TERMINAL_VELOCITY = -3.0f;
constexpr float WATER_BUOYANCY = 0.02f;
constexpr float REST_EPSILON = 0.002f;

} // namespace

AABB ItemEntity::getAABB() const {
    const float half = SIZE * 0.5f;
    return AABB{Vec3{position.x - half, position.y, position.z - half},
                Vec3{position.x + half, position.y + SIZE, position.z + half}};
}

void ItemEntity::tick(World& world) {
    ++ageTicks;

    // Resting early-out: a grounded, still item costs nothing per tick.
    if (onGround && std::abs(velocity.x) < REST_EPSILON && std::abs(velocity.z) < REST_EPSILON &&
        velocity.y <= 0.f) {
        velocity = Vec3{0.f, 0.f, 0.f};
        return;
    }

    const AABB box = getAABB();
    const bool inWater = PhysicsEngine::isInWater(world, box);

    velocity.y += inWater ? GRAVITY * 0.4f : GRAVITY;
    if (inWater && velocity.y < 0.f) {
        velocity.y += WATER_BUOYANCY;
    }
    velocity.y = std::max(velocity.y, TERMINAL_VELOCITY);
    velocity.y *= VERTICAL_DRAG;

    const float horizontalDrag = onGround ? HORIZONTAL_DRAG_GROUND : HORIZONTAL_DRAG_AIR;
    velocity.x *= horizontalDrag;
    velocity.z *= horizontalDrag;

    PhysicsEngine physics;
    Vec3 movement{velocity.x, velocity.y, velocity.z};
    const Vec3 resolved = physics.sweepCollision(box, movement, world);

    position.x += resolved.x;
    position.y += resolved.y;
    position.z += resolved.z;

    onGround = movement.y < 0.f && resolved.y > movement.y;
    if (onGround) {
        velocity.y = 0.f;
    }
    // Stopped horizontally against a wall: shed the blocked component.
    if (resolved.x != movement.x) velocity.x = 0.f;
    if (resolved.z != movement.z) velocity.z = 0.f;
}

void ItemEntityManager::spawn(const ItemStack& stack, const Vec3& position, const Vec3& velocity,
                              int pickupDelay) {
    if (stack.empty()) return;

    if (items_.size() >= MAX_ITEMS) {
        // Evict the oldest grounded item, else the oldest of any; never the
        // fresh spawn (it has not been inserted yet).
        auto victim = items_.end();
        for (auto it = items_.begin(); it != items_.end(); ++it) {
            if (victim == items_.end() || (it->onGround && !victim->onGround) ||
                (it->onGround == victim->onGround && it->ageTicks > victim->ageTicks)) {
                victim = it;
            }
        }
        if (victim != items_.end()) {
            *victim = items_.back();
            items_.pop_back();
        }
    }

    ItemEntity item;
    item.stack = stack;
    item.position = position;
    item.velocity = velocity;
    item.pickupDelay = pickupDelay;
    items_.push_back(item);
}

void ItemEntityManager::tick(World& world, const Vec3& playerPosition) {
    const float activeSq = ACTIVE_RADIUS * ACTIVE_RADIUS;
    for (ItemEntity& item : items_) {
        const float dx = item.position.x - playerPosition.x;
        const float dz = item.position.z - playerPosition.z;
        if (dx * dx + dz * dz > activeSq) {
            continue; // frozen far from the player, like distant animals
        }
        if (item.pickupDelay > 0) --item.pickupDelay;
        item.tick(world);
    }

    if (--mergeCountdown_ <= 0) {
        mergeCountdown_ = MERGE_SCAN_INTERVAL;
        const float mergeSq = MERGE_RADIUS * MERGE_RADIUS;
        for (size_t a = 0; a < items_.size(); ++a) {
            for (size_t b = a + 1; b < items_.size(); ++b) {
                ItemEntity& older = items_[a];
                ItemEntity& younger = items_[b];
                if (older.stack.empty() || younger.stack.empty()) continue;
                if (older.stack.type != younger.stack.type) continue;
                if (older.pickupDelay > 0 || younger.pickupDelay > 0) continue;
                if (older.stack.count + younger.stack.count > maxStackSize(older.stack.type)) {
                    continue;
                }
                const float ddx = older.position.x - younger.position.x;
                const float ddy = older.position.y - younger.position.y;
                const float ddz = older.position.z - younger.position.z;
                if (ddx * ddx + ddy * ddy + ddz * ddz > mergeSq) continue;
                older.stack.count = static_cast<uint8_t>(older.stack.count + younger.stack.count);
                younger.stack.clear();
            }
        }
    }

    // Despawn aged-out and merged-away items with a swap-remove.
    for (size_t i = 0; i < items_.size();) {
        if (items_[i].stack.empty() || items_[i].ageTicks >= ItemEntity::DESPAWN_TICKS) {
            items_[i] = items_.back();
            items_.pop_back();
        } else {
            ++i;
        }
    }
}

void ItemEntityManager::compact() {
    for (size_t i = 0; i < items_.size();) {
        if (items_[i].stack.empty()) {
            items_[i] = items_.back();
            items_.pop_back();
        } else {
            ++i;
        }
    }
}

void ItemEntityManager::clear() {
    items_.clear();
    mergeCountdown_ = MERGE_SCAN_INTERVAL;
}
