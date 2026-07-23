#pragma once

#include <common/math.hpp>
#include <world/chunk.hpp>
#include <world/fluid.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

class World;

// Resolve an aimed boat placement against the exposed top of one resident
// water cell. Canonical partial-height water remains placeable, while falling
// columns, covered cells, and rays that approach from below or the side do not.
std::optional<float> boatPlacementSurfaceY(int64_t blockX, int32_t blockY, int64_t blockZ,
                                           FluidCell waterCell, std::optional<BlockType> blockAbove,
                                           const Vec3& rayOrigin, const Vec3& rayDirection,
                                           float maxDistance);

// A boat receives buoyancy only from a resident, nonfalling water cell that
// vertically overlaps its hull at the cell's actual partial-fluid surface.
// Lava and explicit waterfall columns deliberately remain unsupported.
bool boatFluidCellProvidesBuoyancy(FluidCell cell, int32_t blockY, const AABB& hull);

// ---------------------------------------------------------------------------
// Boats - rideable craft that float on water. Like dropped items, a boat is a
// lightweight value type owned by a bounded manager on the engine, not an
// Entity: it carries no AI and must never spend the animal budget. Boats are
// not persisted; they despawn on quit and on a world switch.
//
// Physics mirrors the dropped-item path (gravity, drag, a per-axis terrain
// sweep) with buoyancy from nonfalling water that settles the hull at the
// actual fluid surface. A ridden boat additionally takes its rider's steering
// acceleration.
// ---------------------------------------------------------------------------

struct Boat {
    Vec3 position;   // AABB bottom center
    Vec3 velocity;   // blocks per tick
    float yaw = 0.f; // heading in radians, faces the rider's view
    bool onGround = false;
    uint8_t renderPackedLight = FULL_SKY_PACKED_LIGHT; // fixed-tick sky/block lighting probe

    static constexpr float WIDTH = 1.4f;
    static constexpr float HEIGHT = 0.55f;
    static constexpr float LENGTH = 1.4f;

    AABB getAABB() const;
};

struct BoatSpawnResult {
    size_t index = 0;
    bool evictedOldest = false;
};

class BoatManager {
public:
    static constexpr size_t MAX_BOATS = 40;
    static constexpr float ACTIVE_RADIUS = 128.f; // ridden boats always tick
    static constexpr float MAX_SPEED = 0.35f;     // blocks per tick on water

    // Spawn a boat and report both its index and any index-shifting eviction.
    BoatSpawnResult spawn(const Vec3& position, float yaw);
    void remove(size_t index);
    void clear() { boats_.clear(); }
    bool empty() const { return boats_.empty(); }

    // One physics step for every boat within range of the player (the ridden
    // boat always steps). `riddenIndex` receives `riderAccel`; the rest drift.
    void tick(World& world, const Vec3& playerPosition, int riddenIndex, const Vec3& riderAccel);

    std::vector<Boat>& boats() { return boats_; }
    const std::vector<Boat>& boats() const { return boats_; }

    // Nearest boat whose AABB the ray crosses within maxDistance, or -1.
    int pick(const Vec3& origin, const Vec3& direction, float maxDistance) const;

private:
    std::vector<Boat> boats_;
};
