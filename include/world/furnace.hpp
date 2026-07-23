#pragma once

#include "world/chunk_pos.hpp"
#include "world/item.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

class Chunk;

// ---------------------------------------------------------------------------
// Furnace - the single source of truth for smelting state and its 20 Hz step.
//
// Furnaces are the only stateful blocks: the engine owns a FurnaceMap keyed
// by block position, ticks every entry each gameTick, and swaps the world
// block between FURNACE and FURNACE_LIT when lit() changes. The map persists
// through SaveManager's block-entities sidecar, never the cube format.
// ---------------------------------------------------------------------------

struct FurnaceState {
    ItemStack input;
    ItemStack fuel;
    ItemStack output;
    uint16_t burnTicksRemaining = 0;
    uint16_t burnTicksTotal = 0; // denominator for the flame gauge
    uint16_t cookTicks = 0;      // 0..FURNACE_COOK_TICKS

    bool lit() const { return burnTicksRemaining > 0; }
    float cookFraction() const;
    float fuelFraction() const;
};

using FurnaceMap = std::unordered_map<BlockPos, FurnaceState>;

// A missing value means the cube is not resident yet, so its sidecar must be
// retained. A resident non-furnace cell proves the sidecar is stale.
constexpr bool furnaceSidecarMatchesLoadedBlock(std::optional<BlockType> loadedBlock) noexcept {
    return !loadedBlock || *loadedBlock == BlockType::FURNACE ||
           *loadedBlock == BlockType::FURNACE_LIT;
}

constexpr BlockType furnaceBlockForState(const FurnaceState& furnace) noexcept {
    return furnace.lit() ? BlockType::FURNACE_LIT : BlockType::FURNACE;
}

// Lit furnace blocks are a derived projection of the sidecar burn state. A
// saved cube enters the world at the inactive baseline, then the engine
// reasserts FURNACE_LIT only for a matching live sidecar.
bool normalizePersistedFurnaceVisuals(Chunk& chunk);

// Engine-owned render projection for saved furnace blocks. The furnace map
// remains the gameplay authority. This object publishes only the desired
// inactive or active block appearance in immutable ChunkPos-grouped shards,
// so generation workers can rehydrate a saved cube without reading mutable
// engine state. Writers copy one shard only when a furnace is placed,
// removed, or changes lit state. Readers perform one atomic snapshot load and
// one chunk lookup.
class FurnaceVisualAuthority {
public:
    FurnaceVisualAuthority();

    void replace(const FurnaceMap& furnaces);
    void set(BlockPos position, BlockType block);
    void erase(BlockPos position);

    // Normalizes orphan FURNACE_LIT cells and reasserts sidecar-backed active
    // cells in one pre-publication transform. Returns the immutable snapshot
    // revision that was applied.
    uint64_t projectSavedChunk(Chunk& chunk) const;
    uint64_t revision() const noexcept;

private:
    struct VisualEntry {
        uint16_t localIndex = 0;
        BlockType block = BlockType::FURNACE;
    };
    using ChunkVisuals = std::vector<VisualEntry>;
    using Shard = std::unordered_map<ChunkPos, ChunkVisuals>;
    static constexpr size_t SHARD_COUNT = 64;
    static_assert((SHARD_COUNT & (SHARD_COUNT - 1)) == 0);

    struct Snapshot {
        uint64_t revision = 1;
        std::array<std::shared_ptr<const Shard>, SHARD_COUNT> shards;
    };

    static ChunkPos chunkPosition(BlockPos position) noexcept;
    static uint16_t localIndex(BlockPos position) noexcept;
    static size_t shardIndex(ChunkPos position) noexcept;
    static uint64_t nextRevision(uint64_t revision) noexcept;

    mutable std::mutex writeMutex_;
    std::shared_ptr<const Snapshot> snapshot_;
    std::atomic<uint64_t> publishedRevision_{0};
};

// One 20 Hz step. Pure item logic: ignites by consuming one fuel unit when
// the input can smelt into the output, advances cooking while lit, and moves
// one result per FURNACE_COOK_TICKS. Unlit progress decays instead of
// holding forever. Returns true when lit() changed so the caller updates the
// world block.
bool furnaceTick(FurnaceState& furnace);
