#pragma once

#include <common/math.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// SpatialHash — Grid-based broad-phase spatial partition for entities
//
// Cell size is 8 blocks. Used for flocking neighbor queries and breed
// mate finding. Each cell maps to a list of entity IDs.
// ---------------------------------------------------------------------------
class SpatialHash {
public:
    explicit SpatialHash(float cellSize = 8.0f);

    // Insert an entity at a position
    void insert(uint64_t entityId, const Vec3& position);

    // Remove an entity (called on death or re-insert)
    void remove(uint64_t entityId);

    // Query entities within a radius of a position
    std::vector<uint64_t> query(const Vec3& position, float radius,
                                const std::unordered_map<uint64_t, Vec3>& entityPositions) const;

    // Query entities within a radius (no distance filter, cell-based only)
    std::vector<uint64_t> queryCells(const Vec3& position, float radius) const;

    // Clear all entries
    void clear();

    // Get cell size
    float getCellSize() const { return cellSize_; }

private:
    float cellSize_;
    float invCellSize_;

    // Grid: cell key → list of entity IDs
    std::unordered_map<int64_t, std::vector<uint64_t>> grid_;

    // Reverse map: entity ID → cell key (for fast removal)
    std::unordered_map<uint64_t, int64_t> entityCell_;

    // Convert world position to cell key
    int64_t positionToCell(const Vec3& position) const;

    // Convert cell coordinates to packed key
    static int64_t cellKey(int cx, int cy, int cz);

    // Convert packed key to cell coordinates
    static void cellToCoords(int64_t key, int& cx, int& cy, int& cz);
};
