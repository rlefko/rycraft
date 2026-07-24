#include "world/furnace.hpp"

#include "world/chunk.hpp"
#include "world/recipes.hpp"

#include <algorithm>
#include <atomic>

namespace {

// Smelting proceeds only when the input has a result the output slot can
// still absorb.
bool canSmelt(const FurnaceState& furnace) {
    if (furnace.input.empty()) return false;
    const auto result = smeltingResult(furnace.input.type);
    if (!result.has_value()) return false;
    if (furnace.output.empty()) return true;
    return furnace.output.type == *result &&
           furnace.output.count < maxStackSize(furnace.output.type);
}

} // namespace

bool normalizePersistedFurnaceVisuals(Chunk& chunk) {
    if (chunk.isUniform() && chunk.uniformBlock() != BlockType::FURNACE_LIT) return false;
    if (!chunk.isUniform() && std::ranges::find(chunk.denseBlocks(), BlockType::FURNACE_LIT) ==
                                  chunk.denseBlocks().end()) {
        return false;
    }

    std::vector<BlockType> blocks = chunk.copyBlocks();
    std::ranges::replace(blocks, BlockType::FURNACE_LIT, BlockType::FURNACE);
    chunk.replaceBlocks(std::move(blocks));
    chunk.clearDerivedLight();
    chunk.modifiedSinceSave = true;
    chunk.markDirty();
    chunk.version.fetch_add(1, std::memory_order_relaxed);
    return true;
}

FurnaceVisualAuthority::FurnaceVisualAuthority() {
    auto snapshot = std::make_shared<Snapshot>();
    const auto emptyShard = std::make_shared<const Shard>();
    snapshot->shards.fill(emptyShard);
    std::atomic_store_explicit(&snapshot_, std::shared_ptr<const Snapshot>(std::move(snapshot)),
                               std::memory_order_release);
    publishedRevision_.store(1, std::memory_order_release);
}

ChunkPos FurnaceVisualAuthority::chunkPosition(BlockPos position) noexcept {
    return {Chunk::worldToChunk(position.x), Chunk::worldToChunkY(position.y),
            Chunk::worldToChunk(position.z)};
}

uint16_t FurnaceVisualAuthority::localIndex(BlockPos position) noexcept {
    return static_cast<uint16_t>(Chunk::index(Chunk::worldToLocal(position.x),
                                              Chunk::worldToLocalY(position.y),
                                              Chunk::worldToLocal(position.z)));
}

size_t FurnaceVisualAuthority::shardIndex(ChunkPos position) noexcept {
    return std::hash<ChunkPos>{}(position) & (SHARD_COUNT - 1);
}

uint64_t FurnaceVisualAuthority::nextRevision(uint64_t revision) noexcept {
    ++revision;
    return revision == 0 ? 1 : revision;
}

void FurnaceVisualAuthority::replace(const FurnaceMap& furnaces) {
    std::lock_guard lock(writeMutex_);
    const auto current = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    auto next = std::make_shared<Snapshot>();
    next->revision = nextRevision(current ? current->revision : 0);

    std::array<std::shared_ptr<Shard>, SHARD_COUNT> mutableShards;
    for (auto& shard : mutableShards)
        shard = std::make_shared<Shard>();
    for (const auto& [position, furnace] : furnaces) {
        const ChunkPos chunk = chunkPosition(position);
        (*mutableShards[shardIndex(chunk)])[chunk].push_back(
            {localIndex(position), furnaceBlockForState(furnace)});
    }
    for (size_t shardIndex = 0; shardIndex < SHARD_COUNT; ++shardIndex) {
        for (auto& [chunk, entries] : *mutableShards[shardIndex]) {
            (void)chunk;
            std::ranges::sort(entries, {}, &VisualEntry::localIndex);
        }
        next->shards[shardIndex] = std::move(mutableShards[shardIndex]);
    }
    const uint64_t revision = next->revision;
    std::atomic_store_explicit(&snapshot_, std::shared_ptr<const Snapshot>(std::move(next)),
                               std::memory_order_release);
    publishedRevision_.store(revision, std::memory_order_release);
}

void FurnaceVisualAuthority::set(BlockPos position, BlockType block) {
    if (block != BlockType::FURNACE && block != BlockType::FURNACE_LIT) return;
    std::lock_guard lock(writeMutex_);
    const auto current = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    auto next = std::make_shared<Snapshot>(*current);
    next->revision = nextRevision(current->revision);

    const ChunkPos chunk = chunkPosition(position);
    const size_t targetShard = shardIndex(chunk);
    auto shard = std::make_shared<Shard>(*current->shards[targetShard]);
    ChunkVisuals& entries = (*shard)[chunk];
    const uint16_t index = localIndex(position);
    const auto entry = std::ranges::lower_bound(entries, index, {}, &VisualEntry::localIndex);
    if (entry != entries.end() && entry->localIndex == index) {
        entry->block = block;
    } else {
        entries.insert(entry, VisualEntry{index, block});
    }
    next->shards[targetShard] = std::move(shard);
    const uint64_t revision = next->revision;
    std::atomic_store_explicit(&snapshot_, std::shared_ptr<const Snapshot>(std::move(next)),
                               std::memory_order_release);
    publishedRevision_.store(revision, std::memory_order_release);
}

void FurnaceVisualAuthority::erase(BlockPos position) {
    std::lock_guard lock(writeMutex_);
    const auto current = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    const ChunkPos chunk = chunkPosition(position);
    const size_t targetShard = shardIndex(chunk);
    const auto currentChunk = current->shards[targetShard]->find(chunk);
    if (currentChunk == current->shards[targetShard]->end()) return;

    const uint16_t index = localIndex(position);
    const auto currentEntry =
        std::ranges::lower_bound(currentChunk->second, index, {}, &VisualEntry::localIndex);
    if (currentEntry == currentChunk->second.end() || currentEntry->localIndex != index) return;

    auto next = std::make_shared<Snapshot>(*current);
    next->revision = nextRevision(current->revision);
    auto shard = std::make_shared<Shard>(*current->shards[targetShard]);
    ChunkVisuals& entries = shard->at(chunk);
    const auto entry = std::ranges::lower_bound(entries, index, {}, &VisualEntry::localIndex);
    entries.erase(entry);
    if (entries.empty()) shard->erase(chunk);
    next->shards[targetShard] = std::move(shard);
    const uint64_t revision = next->revision;
    std::atomic_store_explicit(&snapshot_, std::shared_ptr<const Snapshot>(std::move(next)),
                               std::memory_order_release);
    publishedRevision_.store(revision, std::memory_order_release);
}

uint64_t FurnaceVisualAuthority::projectSavedChunk(Chunk& chunk) const {
    const auto snapshot = std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
    const auto& shard = *snapshot->shards[shardIndex(chunk.pos())];
    const auto projected = shard.find(chunk.pos());
    const bool hasPersistedActive =
        chunk.isUniform() ? chunk.uniformBlock() == BlockType::FURNACE_LIT
                          : std::ranges::find(chunk.denseBlocks(), BlockType::FURNACE_LIT) !=
                                chunk.denseBlocks().end();
    if (!hasPersistedActive && projected == shard.end()) return snapshot->revision;

    std::vector<BlockType> blocks = chunk.copyBlocks();
    const std::vector<BlockType> original = blocks;
    std::ranges::replace(blocks, BlockType::FURNACE_LIT, BlockType::FURNACE);
    if (projected != shard.end()) {
        for (const VisualEntry& entry : projected->second) {
            BlockType& cell = blocks[entry.localIndex];
            if (cell == BlockType::FURNACE || cell == BlockType::FURNACE_LIT) {
                cell = entry.block;
            }
        }
    }
    if (blocks != original) {
        chunk.replaceBlocks(std::move(blocks));
        chunk.clearDerivedLight();
        chunk.modifiedSinceSave = true;
        chunk.markDirty();
        chunk.version.fetch_add(1, std::memory_order_relaxed);
    }
    return snapshot->revision;
}

uint64_t FurnaceVisualAuthority::revision() const noexcept {
    return publishedRevision_.load(std::memory_order_acquire);
}

float FurnaceState::cookFraction() const {
    return static_cast<float>(cookTicks) / static_cast<float>(FURNACE_COOK_TICKS);
}

float FurnaceState::fuelFraction() const {
    if (burnTicksTotal == 0) return 0.0f;
    return static_cast<float>(burnTicksRemaining) / static_cast<float>(burnTicksTotal);
}

bool furnaceTick(FurnaceState& furnace) {
    const bool wasLit = furnace.lit();
    const bool smeltable = canSmelt(furnace);

    if (!furnace.lit() && smeltable && !furnace.fuel.empty() && isFurnaceFuel(furnace.fuel.type)) {
        const int burn = fuelBurnTicks(furnace.fuel.type);
        furnace.burnTicksRemaining = static_cast<uint16_t>(burn);
        furnace.burnTicksTotal = static_cast<uint16_t>(burn);
        if (--furnace.fuel.count == 0) furnace.fuel.clear();
    }

    if (furnace.lit()) {
        --furnace.burnTicksRemaining;
        if (smeltable) {
            if (++furnace.cookTicks >= FURNACE_COOK_TICKS) {
                furnace.cookTicks = 0;
                const ItemType result = *smeltingResult(furnace.input.type);
                if (furnace.output.empty()) {
                    furnace.output = ItemStack{result, 1, 0};
                } else {
                    ++furnace.output.count;
                }
                if (--furnace.input.count == 0) furnace.input.clear();
            }
        } else {
            furnace.cookTicks = 0;
        }
    } else if (furnace.cookTicks > 0) {
        // Fire ran out mid-item: progress cools off instead of freezing.
        furnace.cookTicks =
            furnace.cookTicks >= 2 ? static_cast<uint16_t>(furnace.cookTicks - 2) : uint16_t{0};
    }

    return furnace.lit() != wasLit;
}
