#include "entity/entity_picking.hpp"

#include <cmath>

namespace {

// Slab test: the ray parameter t at which `origin + t*dir` first enters the
// box, or -1 if it never does within [0, maxDistance]. `dir` is normalized by
// the caller-supplied length so t is in world units.
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

std::optional<EntityHit> pickEntity(const Vec3& origin, const Vec3& dir, float maxDistance,
                                    const std::vector<std::shared_ptr<Entity>>& entities) {
    const float length = dir.length();
    if (length < 1e-6f) return std::nullopt;
    const Vec3 unit{dir.x / length, dir.y / length, dir.z / length};

    std::optional<EntityHit> nearest;
    for (const auto& entity : entities) {
        if (!entity || !entity->alive) continue;
        const float t = rayBoxEntry(origin, unit, entity->aabb, maxDistance);
        if (t < 0.f) continue;
        if (!nearest || t < nearest->distance) {
            nearest = EntityHit{entity->id, t};
        }
    }
    return nearest;
}
