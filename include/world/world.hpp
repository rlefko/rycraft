#pragma once
#include "common/ema.hpp"
#include "common/thread_pool.hpp"
#include "world/chunk.hpp"
#include "world/chunk_generator.hpp"
#include "world/chunk_pos.hpp"
#include "world/mesh_snapshot.hpp"

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class SaveManager;

// Sorts farthest-first so pop_back() consumes the nearest chunk next
// (free function so priority ordering is unit-testable).
void sortChunksByDistance(std::vector<ChunkPos>& chunks, int centerChunkX, int centerChunkZ);

// Generation submission window: nearest chunks stream through a small
// in-flight set so a boundary cross re-prioritizes everything still queued
// instead of fighting hundreds of already-submitted stale-priority tasks.
inline constexpr size_t MAX_INFLIGHT_GEN = 32;

class World {
public:
    explicit World(uint32_t seed, int viewDistance = 32);
    ~World();

    // Delete copy/move (contains non-copyable ThreadPool and mutex)
    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) = delete;
    World& operator=(World&&) = delete;

    // Get or generate a chunk (thread-safe)
    std::shared_ptr<Chunk> getChunk(int chunkX, int chunkZ);

    // Get block at world position (thread-safe; generates the chunk on a
    // miss — never call from the render path)
    BlockType getBlock(int x, int y, int z);

    // Non-generating read: a missing chunk reads as air. For per-frame
    // queries (e.g. is-the-camera-underwater) that must never stall on
    // generation or disk.
    BlockType getBlockIfLoaded(int x, int y, int z) const;

    // Non-generating surface lookup (the heightMap top of the column), or
    // nullopt while the chunk isn't loaded. Weather uses it to keep rain
    // from spawning inside caves and under overhangs.
    std::optional<int> surfaceHeightIfLoaded(int x, int z) const;

    // Set block at world position (thread-safe, marks chunk dirty)
    void setBlock(int x, int y, int z, BlockType type);

    // Get terrain height at world position
    double getTerrainHeight(int x, int z) const;

    // Get biome at world position
    Biome getBiome(int x, int z) const;

    // Get all loaded chunks
    std::vector<std::shared_ptr<Chunk>> getLoadedChunks() const;

    // Copy a chunk + one-block neighbor walls for lock-free meshing.
    // Returns false until the chunk and all four face neighbors are
    // generated (the caller simply retries next frame).
    bool snapshotForMeshing(ChunkPos pos, MeshSnapshot& out) const;

    // Get chunks that need mesh updates
    std::vector<std::shared_ptr<Chunk>> getDirtyChunks();

    // Mark chunk as meshed
    void markChunkMeshed(int chunkX, int chunkZ);

    // Get seed
    uint32_t getSeed() const;

    // View distance
    int getViewDistance() const;
    void setViewDistance(int distance);

    // Update chunks around player position (synchronous)
    void updatePlayerPosition(int playerX, int playerZ);

    // Drain the block-light reconcile queue: pull lava light across chunk
    // borders until quiescent, bounded to budgetChunks re-floods per call so
    // the tick never stalls. Called every tick; re-meshes chunks whose light
    // changed. See LightEngine for the flood/fixed-point rationale.
    void reconcileLight(int budgetChunks);

    // Unload chunks outside the view distance (called by updatePlayerPosition).
    // Edited chunks queue for saving as they leave the map.
    void unloadDistantChunks();

    // Queue every still-loaded edited chunk for saving (quit path — callers
    // must not mutate blocks afterward until SaveManager::flush returns)
    void saveModifiedChunks();

    // Rebuild the generation backlog (nearest-first) and start pumping
    void generateAroundPlayer(int playerX, int playerZ);

    // Submit backlog chunks up to the in-flight window. Called every tick
    // and by finishing workers, so the pipeline sustains itself.
    void pumpGeneration();

    // Get generation queue status
    size_t getPendingChunkCount() const;

    // EMA of per-chunk generation time in ms (updated by the gen workers,
    // read lock-free by the HUD)
    float averageGenMs() const;

    // Attach the save manager (non-owning). Once set, getChunk and the async
    // generation path try loading a chunk from disk before generating it —
    // without this, saved block edits were written but never read back.
    void setSaveManager(SaveManager* saveManager);

private:
    uint32_t seed_;
    int viewDistance_;

    // Chunk storage keyed by chunk grid coordinate
    std::unordered_map<ChunkPos, std::shared_ptr<Chunk>> chunks_;
    mutable std::mutex chunksMutex_;

    // Persistence (non-owning; the engine owns the SaveManager)
    SaveManager* saveManager_ = nullptr;

    // World generation (climate + density + surface + features)
    ChunkGenerator generator_;

    // Player position for chunk loading. hasPlayerChunk_ stays false until
    // the first updatePlayerPosition call, so the spawn area streams in even
    // when the player starts in chunk (0, 0).
    int playerChunkX_ = 0;
    int playerChunkZ_ = 0;
    bool hasPlayerChunk_ = false;

    // Async generation state. genBacklog_ holds not-yet-submitted chunks
    // sorted farthest-first (pop_back = nearest); both containers are
    // guarded by pendingMutex_ (lock order: pendingMutex_ → chunksMutex_).
    std::shared_ptr<ThreadPool> genPool_; // lazily initialized
    std::unordered_map<ChunkPos, std::future<void>> pendingGenerations_;
    std::vector<ChunkPos> genBacklog_;
    std::atomic<bool> shuttingDown_{false};
    mutable std::mutex pendingMutex_;

    // Generation-time EMA: workers record, the HUD reads
    AtomicEmaMs genMs_;

    // Block-light reconcile queue: chunks whose stored light may be stale
    // because a neighbor loaded or an edit landed. Drained by reconcileLight
    // on the tick thread. lightQueued_ dedups. lightMutex_ is a leaf below
    // chunksMutex_ (only ever taken innermost; never held while acquiring
    // another lock).
    std::vector<ChunkPos> lightQueue_;
    std::unordered_set<ChunkPos> lightQueued_;
    mutable std::mutex lightMutex_;

    // Enqueue a chunk (and its four face neighbors) for light reconciliation.
    void queueLightReconcile(ChunkPos pos);
    void queueFaceNeighbors(ChunkPos pos);
    void queueLightReconcileWithNeighbors(ChunkPos pos);

    // Generate a single chunk (synchronous)
    void generateChunk(std::shared_ptr<Chunk> chunk);

    // Generate chunk asynchronously
    void generateChunkAsync(int chunkX, int chunkZ);

    // Load the chunk from disk when possible, otherwise generate it
    // (with the flat-fallback policy on generation failure).
    std::shared_ptr<Chunk> loadOrGenerateChunk(int chunkX, int chunkZ);
};
