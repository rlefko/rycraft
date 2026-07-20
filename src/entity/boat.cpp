#include "entity/boat.hpp"

#include "entity/physics.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float GRAVITY = -0.06f;      // pulls a beached boat down to the ground
constexpr float BUOYANCY = 0.06f;      // lifts a submerged hull toward the surface
constexpr float VERTICAL_DAMP = 0.55f; // settles the bob so the boat rides level
constexpr float WATER_DRAG = 0.90f;    // gliding resistance on water
constexpr float GROUND_DRAG = 0.70f;   // a grounded boat barely slides

// Ray-versus-AABB slab entry distance, or -1 when the ray misses.
float rayBoxEntry(const Vec3& origin, const Vec3& dir, const AABB& box, float maxDistance) {
    float tMin = 0.f;
    float tMax = maxDistance;
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {dir.x, dir.y, dir.z};
    const float lo[3] = {box.min.x, box.min.y, box.min.z};
    const float hi[3] = {box.max.x, box.max.y, box.max.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(d[axis]) < 1e-6f) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) return -1.f;
            continue;
        }
        const float inv = 1.f / d[axis];
        float t1 = (lo[axis] - o[axis]) * inv;
        float t2 = (hi[axis] - o[axis]) * inv;
        if (t1 > t2) std::swap(t1, t2);
        tMin = std::max(tMin, t1);
        tMax = std::min(tMax, t2);
        if (tMin > tMax) return -1.f;
    }
    return tMin;
}

} // namespace

AABB Boat::getAABB() const {
    const float halfW = WIDTH * 0.5f;
    const float halfL = LENGTH * 0.5f;
    return AABB{Vec3{position.x - halfW, position.y, position.z - halfL},
                Vec3{position.x + halfW, position.y + HEIGHT, position.z + halfL}};
}

size_t BoatManager::spawn(const Vec3& position, float yaw) {
    if (boats_.size() >= MAX_BOATS) {
        boats_.erase(boats_.begin()); // evict the oldest at the cap
    }
    boats_.push_back(Boat{position, Vec3{0.f, 0.f, 0.f}, yaw, false});
    return boats_.size() - 1;
}

void BoatManager::remove(size_t index) {
    if (index < boats_.size()) {
        boats_.erase(boats_.begin() + static_cast<std::ptrdiff_t>(index));
    }
}

void BoatManager::tick(World& world, const Vec3& playerPosition, int riddenIndex,
                       const Vec3& riderAccel) {
    for (size_t index = 0; index < boats_.size(); ++index) {
        Boat& boat = boats_[index];
        const bool ridden = static_cast<int>(index) == riddenIndex;
        // Distant unmanned boats freeze completely, like the animal sim.
        if (!ridden && (boat.position - playerPosition).length() > ACTIVE_RADIUS) {
            continue;
        }

        const AABB box = boat.getAABB();
        const bool inWater = PhysicsEngine::isInWater(world, box);

        // Vertical: buoyancy floats the hull toward the surface, otherwise it
        // sinks under gravity until it grounds.
        boat.velocity.y += inWater ? BUOYANCY : GRAVITY;
        if (inWater) boat.velocity.y *= VERTICAL_DAMP;

        // Horizontal: the rider's push (ridden boat only) plus water/ground drag.
        if (ridden) {
            boat.velocity.x += riderAccel.x;
            boat.velocity.z += riderAccel.z;
        }
        const float drag = boat.onGround ? GROUND_DRAG : (inWater ? WATER_DRAG : 0.98f);
        boat.velocity.x *= drag;
        boat.velocity.z *= drag;

        // Clamp the planar speed so a long push cannot rocket the boat.
        const float speed =
            std::sqrt(boat.velocity.x * boat.velocity.x + boat.velocity.z * boat.velocity.z);
        if (speed > MAX_SPEED) {
            boat.velocity.x *= MAX_SPEED / speed;
            boat.velocity.z *= MAX_SPEED / speed;
        }

        PhysicsEngine physics;
        const Vec3 movement = boat.velocity;
        const Vec3 resolved = physics.sweepCollision(box, movement, world);
        boat.position += resolved;

        // Zero the velocity on any axis the sweep blocked so it does not build.
        boat.onGround = movement.y < 0.f && resolved.y > movement.y;
        if (std::abs(resolved.y - movement.y) > 1e-5f) boat.velocity.y = 0.f;
        if (std::abs(resolved.x - movement.x) > 1e-5f) boat.velocity.x = 0.f;
        if (std::abs(resolved.z - movement.z) > 1e-5f) boat.velocity.z = 0.f;
    }
}

int BoatManager::pick(const Vec3& origin, const Vec3& direction, float maxDistance) const {
    const float length = direction.length();
    if (length < 1e-6f) return -1;
    const Vec3 unit{direction.x / length, direction.y / length, direction.z / length};

    int nearest = -1;
    float nearestT = maxDistance;
    for (size_t index = 0; index < boats_.size(); ++index) {
        const float t = rayBoxEntry(origin, unit, boats_[index].getAABB(), maxDistance);
        if (t < 0.f) continue;
        if (nearest < 0 || t < nearestT) {
            nearest = static_cast<int>(index);
            nearestT = t;
        }
    }
    return nearest;
}
