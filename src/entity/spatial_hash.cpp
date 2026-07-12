#include "entity/spatial_hash.hpp"

#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// SpatialHash constructor
// ---------------------------------------------------------------------------
SpatialHash::SpatialHash(float cellSize)
    : cellSize_(cellSize), invCellSize_(1.f / cellSize) {
    // Guard: cell size must be positive
    if (cellSize_ <= 0.f) {
        cellSize_ = 8.0f;
        invCellSize_ = 1.f / cellSize_;
    }
}

// ---------------------------------------------------------------------------
// positionToCell — World position → cell coordinates
// ---------------------------------------------------------------------------
int64_t SpatialHash::positionToCell(const Vec3& position) const {
    int cx = static_cast<int>(std::floor(position.x * invCellSize_));
    int cy = static_cast<int>(std::floor(position.y * invCellSize_));
    int cz = static_cast<int>(std::floor(position.z * invCellSize_));
    return cellKey(cx, cy, cz);
}

// ---------------------------------------------------------------------------
// cellKey — Pack 3 cell coordinates into a single int64_t
//
// Uses 21 bits per axis (±1 million blocks), total 63 bits + sign.
// ---------------------------------------------------------------------------
int64_t SpatialHash::cellKey(int cx, int cy, int cz) {
    // Offset to make all coordinates positive (21-bit range: 0..2097151)
    constexpr int OFFSET = 1 << 20; // 1048576
    uint32_t ux = static_cast<uint32_t>(cx + OFFSET) & 0x1FFFFF;
    uint32_t uy = static_cast<uint32_t>(cy + OFFSET) & 0x1FFFFF;
    uint32_t uz = static_cast<uint32_t>(cz + OFFSET) & 0x1FFFFF;

    return static_cast<int64_t>(
        (static_cast<int64_t>(ux) << 42) |
        (static_cast<int64_t>(uy) << 21) |
        static_cast<int64_t>(uz)
    );
}

void SpatialHash::cellToCoords(int64_t key, int& cx, int& cy, int& cz) {
    constexpr int OFFSET = 1 << 20;
    uint32_t ux = static_cast<uint32_t>((key >> 42) & 0x1FFFFF);
    uint32_t uy = static_cast<uint32_t>((key >> 21) & 0x1FFFFF);
    uint32_t uz = static_cast<uint32_t>(key & 0x1FFFFF);
    cx = static_cast<int>(ux) - OFFSET;
    cy = static_cast<int>(uy) - OFFSET;
    cz = static_cast<int>(uz) - OFFSET;
}

// ---------------------------------------------------------------------------
// insert — Add entity to grid
// ---------------------------------------------------------------------------
void SpatialHash::insert(uint64_t entityId, const Vec3& position) {
    // Remove from old cell first (handles re-insertion after movement)
    auto it = entityCell_.find(entityId);
    if (it != entityCell_.end()) {
        int64_t oldCell = it->second;
        auto& cellList = grid_[oldCell];
        cellList.erase(
            std::remove(cellList.begin(), cellList.end(), entityId),
            cellList.end()
        );
        if (cellList.empty()) {
            grid_.erase(oldCell);
        }
    }

    // Insert into new cell
    int64_t newCell = positionToCell(position);
    grid_[newCell].push_back(entityId);
    entityCell_[entityId] = newCell;
}

// ---------------------------------------------------------------------------
// remove — Remove entity from grid
// ---------------------------------------------------------------------------
void SpatialHash::remove(uint64_t entityId) {
    auto it = entityCell_.find(entityId);
    if (it == entityCell_.end()) return;

    int64_t cell = it->second;
    entityCell_.erase(it);

    auto& cellList = grid_[cell];
    cellList.erase(
        std::remove(cellList.begin(), cellList.end(), entityId),
        cellList.end()
    );
    if (cellList.empty()) {
        grid_.erase(cell);
    }
}

// ---------------------------------------------------------------------------
// query — Find all entities within radius of position
// ---------------------------------------------------------------------------
std::vector<uint64_t> SpatialHash::query(const Vec3& position, float radius) const {
    std::vector<uint64_t> results;

    // Calculate cell range
    int centerCX = static_cast<int>(std::floor(position.x * invCellSize_));
    int centerCY = static_cast<int>(std::floor(position.y * invCellSize_));
    int centerCZ = static_cast<int>(std::floor(position.z * invCellSize_));

    int radiusCells = static_cast<int>(std::ceil(radius / cellSize_));

    // Collect entity IDs from neighboring cells
    std::vector<uint64_t> candidates;

    for (int cx = centerCX - radiusCells; cx <= centerCX + radiusCells; ++cx) {
        for (int cy = centerCY - radiusCells; cy <= centerCY + radiusCells; ++cy) {
            for (int cz = centerCZ - radiusCells; cz <= centerCZ + radiusCells; ++cz) {
                int64_t key = cellKey(cx, cy, cz);
                auto it = grid_.find(key);
                if (it != grid_.end()) {
                    candidates.insert(candidates.end(), it->second.begin(), it->second.end());
                }
            }
        }
    }

    // Return all candidates from neighboring cells.
    // The caller should filter by actual entity position if needed.
    for (uint64_t id : candidates) {
        results.push_back(id);
    }

    return results;
}

// ---------------------------------------------------------------------------
// clear — Remove all entries
// ---------------------------------------------------------------------------
void SpatialHash::clear() {
    grid_.clear();
    entityCell_.clear();
}
