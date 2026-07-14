#pragma once

#include "world/chunk.hpp"

#include <array>

// ---------------------------------------------------------------------------
// LightEngine — derives per-chunk block light (the orange glow lava casts into
// caves) by flooding outward from emitters, losing one level per block through
// transparent cells. Light is DERIVED state: never serialized (RYCH untouched),
// recomputed on generation/load and after edits.
//
// floodChunk recomputes a whole chunk from scratch each call (self emitters +
// the border light of its neighbors), so it needs no separate add/remove
// passes — placing or breaking a light source is just a re-flood. The flood is
// the unique max-light fixed point for a fixed set of blocks and neighbor
// borders, so repeated reconciliation converges to the same global result
// regardless of the order chunks stream in (World::reconcileLight drives it).
// ---------------------------------------------------------------------------
namespace LightEngine {

// The four horizontal face neighbors whose border light spills into a chunk,
// in -X, +X, -Z, +Z order (vertical stays inside a full-height chunk). Null
// entries are treated as dark — an un-loaded neighbor pulls its light in later
// when it loads and re-queues this chunk.
using FaceNeighbors = std::array<const Chunk*, 4>;

// Recompute chunk.blockLight from its own emitters plus the neighbors' border
// light. Returns true if the light changed, so the caller can re-mesh the
// chunk and re-reconcile its neighbors.
bool floodChunk(Chunk& chunk, const FaceNeighbors& neighbors);

// floodChunk with no neighbors — self emitters only. Runs on the generation
// worker before the chunk is shared, so the first mesh already shows lava glow;
// cross-chunk spill is layered on later by reconcileLight.
inline bool computeSelfLight(Chunk& chunk) {
    return floodChunk(chunk, FaceNeighbors{nullptr, nullptr, nullptr, nullptr});
}

} // namespace LightEngine
