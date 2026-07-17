#pragma once

#include "entity/entity.hpp"

#include <common/math.hpp>

#include <memory>
#include <optional>
#include <vector>

// ---------------------------------------------------------------------------
// Entity picking — ray versus living-entity AABBs for melee targeting. A
// linear slab test over the spawner's entity vector: at the 64-animal cap a
// scan is cheaper than materializing spatial-hash queries, and it runs only
// on attack click edges.
// ---------------------------------------------------------------------------

struct EntityHit {
    uint64_t entityId = 0;
    float distance = 0.f;
};

// Nearest living entity whose AABB the ray crosses within maxDistance, or
// nullopt. `dir` need not be normalized.
std::optional<EntityHit> pickEntity(const Vec3& origin, const Vec3& dir, float maxDistance,
                                    const std::vector<std::shared_ptr<Entity>>& entities);
