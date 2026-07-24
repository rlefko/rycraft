#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/boat_renderer.hpp>
#include <render/celestial.hpp>
#include <render/cloud_renderer.hpp>
#include <render/dynamic_object_lighting.hpp>
#include <render/entity_renderer.hpp>
#include <render/far_terrain.hpp>
#include <render/item_entity_renderer.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/mesh_scheduler.hpp>
#include <render/metal_ownership.hpp>
#include <render/particles.hpp>
#include <render/pixel_formats.hpp>
#include <render/post_stack.hpp>
#include <render/render_pipeline.hpp>
#include <render/screen_space_lighting.hpp>
#include <render/shader_types.hpp>
#include <render/shadow_map.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/learned_terrain.hpp>
#include <world/light_engine.hpp>
#include <world/native_hydrology.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/weather.hpp>
#include <world/world.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <numbers>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Rendering: meshing, textures, shared GPU layouts
// ===========================================================================

TEST_CASE("LOD overlay distinguishes exact ownership from every far tier",
          "[render][far][overlay]") {
    const auto exact = terrainLodOverlayColor(std::nullopt);
    const std::array<FarTerrainStep, 6> steps = {
        FarTerrainStep::ONE,   FarTerrainStep::TWO,     FarTerrainStep::FOUR,
        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    std::array<std::array<float, 4>, 6> farColors{};
    for (size_t index = 0; index < steps.size(); ++index) {
        farColors[index] = terrainLodOverlayColor(steps[index]);
        REQUIRE(farColors[index] != exact);
        REQUIRE(farColors[index][3] == exact[3]);
        for (size_t previous = 0; previous < index; ++previous) {
            REQUIRE(farColors[index] != farColors[previous]);
        }
    }
}

namespace {
bool metalOwnershipProbeDeallocated = false;
}

@interface MetalOwnershipProbe : NSObject
@end

@implementation MetalOwnershipProbe
- (void)dealloc {
    metalOwnershipProbeDeallocated = true;
#if !__has_feature(objc_arc)
    [super dealloc];
#endif
}
@end

// ============================================================================
// Vertex Format Tests
// ============================================================================

TEST_CASE("Vertex size is 16 bytes", "[render][vertex]") {
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Vertex alignment is 16 bytes", "[render][vertex]") {
    REQUIRE(alignof(Vertex) == 16);
}

TEST_CASE("Vertex fields have expected sizes", "[render][vertex]") {
    REQUIRE(sizeof(float16_t) == 2);
    REQUIRE(sizeof(uint8_t) == 1);
    REQUIRE(sizeof(uint32_t) == 4);
}

namespace {

worldgen::surface_material::SurfaceMaterialPalette testMaterialPalette(BlockType material) {
    worldgen::surface_material::SurfaceMaterialPalette palette;
    palette.count = 1;
    palette.entries[0] = {.material = material, .weight = 255};
    return palette;
}

bool sameMaterialPalette(const worldgen::surface_material::SurfaceMaterialPalette& first,
                         const worldgen::surface_material::SurfaceMaterialPalette& second) {
    if (first.count != second.count)
        return false;
    return std::equal(first.entries.begin(), first.entries.begin() + first.count,
                      second.entries.begin(), [](const auto& lhs, const auto& rhs) {
                          return lhs.material == rhs.material && lhs.weight == rhs.weight;
                      });
}

using TestFarGeometryFunction =
    std::function<FarTerrainGeometrySample(int64_t worldX, int64_t worldZ)>;
using TestFarMaterialFunction = std::function<BlockType(int64_t worldX, int64_t worldZ,
                                                        const FarTerrainGeometrySample& geometry)>;

FarTerrainSource testFarTerrainSource(TestFarGeometryFunction geometry,
                                      TestFarMaterialFunction material) {
    FarTerrainSource source;
    source.sample = [geometry = std::move(geometry), material = std::move(material)](
                        int64_t x, int64_t z, worldgen::SurfaceFootprint) {
        FarTerrainGeometrySample surface = geometry(x, z);
        if (surface.lake && surface.waterBodyId == worldgen::NO_WATER_BODY) {
            surface.waterBodyId = 0x5445'5354'4C41'4B45ULL;
        }
        return FarSurfaceSample{
            .geometry = surface,
            .footprintMinimumTerrainHeight = surface.terrainHeight,
            .footprintMaximumTerrainHeight = surface.terrainHeight,
            .materialPalette = testMaterialPalette(material(x, z, surface)),
        };
    };
    return source;
}

FarTerrainGeometrySample
testFarGeometry(const FarTerrainSource& source, int64_t x, int64_t z,
                worldgen::SurfaceFootprint footprint = worldgen::SurfaceFootprint::BLOCK_1) {
    return source.sample(x, z, footprint).geometry;
}

FarTerrainSource farTerrainTestSource() {
    return testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            const int64_t variation = world_coord::floorMod(x + z * 3, 29);
            sample.terrainHeight = 72.0 + static_cast<double>(variation) * 0.25;
            sample.waterSurface = SEA_LEVEL;
            return sample;
        },
        [](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
            return world_coord::floorMod(x / 64 + z / 64, 2) == 0 ? BlockType::GRASS
                                                                  : BlockType::STONE;
        });
}

struct FarTerrainTestGateRelease {
    std::mutex& mutex;
    std::condition_variable& condition;
    bool& released;

    ~FarTerrainTestGateRelease() {
        {
            std::lock_guard lock(mutex);
            released = true;
        }
        condition.notify_all();
    }
};

worldgen::learned::GenerationIdentity coarsePrefetchTestIdentity() {
    worldgen::learned::GenerationIdentity identity;
    identity.seed = 0xC0A5'EA17'0000'0042ULL;
    identity.modelPackHash[0] = 0x42;
    identity.runtimeHash[0] = 0x24;
    return identity;
}

worldgen::learned::GenerationIdentity nativeTopologyTestIdentity(uint64_t seed) {
    worldgen::learned::GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash.fill(0x6AU);
    identity.runtimeHash.fill(0xB4U);
    return identity;
}

template <typename Operation> decltype(auto) awaitV4Authority(Operation&& operation) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        try {
            return operation();
        } catch (const worldgen::learned::GenerationFailureException& failure) {
            if (failure.status() != worldgen::learned::AuthorityStatus::DEFERRED ||
                std::chrono::steady_clock::now() >= deadline) {
                throw;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

class CoarsePrefetchFailingBackend final : public worldgen::learned::TerrainInferenceBackend {
public:
    worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>
    inferPage(const worldgen::learned::GenerationIdentity&,
              worldgen::learned::TerrainPageKey) override {
        calls_.fetch_add(1, std::memory_order_release);
        return worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>::failed(
            {.code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
             .message = "Synthetic coarse prefetch failure",
             .retriable = true});
    }

    [[nodiscard]] uint64_t callCount() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> calls_{0};
};

class AuthorityQualityRecordingBackend final : public worldgen::learned::TerrainInferenceBackend {
public:
    worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>
    inferPage(const worldgen::learned::GenerationIdentity& identity,
              worldgen::learned::TerrainPageKey key) override {
        calls_[static_cast<size_t>(key.quality)].fetch_add(1, std::memory_order_release);
        return delegate_.inferPage(identity, key);
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    inferFinalNativeGrid(const worldgen::learned::GenerationIdentity& identity,
                         worldgen::learned::NativeRect region) override {
        finalRectangleCalls_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard lock(finalRectangleMutex_);
            ++finalRectangleCallCounts_[region];
        }
        return delegate_.inferFinalNativeGrid(identity, region);
    }

    worldgen::learned::AuthorityResult<worldgen::learned::CoarseSpawnGrid>
    inferCoarseSpawnGrid(const worldgen::learned::GenerationIdentity& identity,
                         worldgen::learned::CoarseSpawnRegion region) override {
        return delegate_.inferCoarseSpawnGrid(identity, region);
    }

    [[nodiscard]] uint64_t pageCalls(worldgen::learned::AuthorityQuality quality) const noexcept {
        return calls_[static_cast<size_t>(quality)].load(std::memory_order_acquire);
    }

    [[nodiscard]] uint64_t finalRectangleCalls() const noexcept {
        return finalRectangleCalls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::map<worldgen::learned::NativeRect, uint64_t>
    finalRectangleCallCounts() const {
        std::lock_guard lock(finalRectangleMutex_);
        return finalRectangleCallCounts_;
    }

private:
    worldgen::learned::DeterministicFakeTerrainBackend delegate_;
    std::array<std::atomic<uint64_t>, 2> calls_{};
    std::atomic<uint64_t> finalRectangleCalls_{0};
    mutable std::mutex finalRectangleMutex_;
    std::map<worldgen::learned::NativeRect, uint64_t> finalRectangleCallCounts_;
};

class BlockingTransientBackend final : public worldgen::learned::TerrainInferenceBackend {
public:
    worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>
    inferPage(const worldgen::learned::GenerationIdentity& identity,
              worldgen::learned::TerrainPageKey key) override {
        return delegate_.inferPage(identity, key);
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    inferFinalNativeGrid(const worldgen::learned::GenerationIdentity&,
                         worldgen::learned::NativeRect region) override {
        calls_.fetch_add(1, std::memory_order_release);
        {
            std::unique_lock lock(mutex_);
            entered_ = true;
            ready_.notify_all();
            if (!ready_.wait_for(lock, std::chrono::seconds(5), [&] { return released_; })) {
                return worldgen::learned::AuthorityResult<
                    worldgen::learned::PhysicalTerrainGrid>::failed({
                    .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                    .message = "Synthetic transient terrain gate timed out",
                    .retriable = true,
                });
            }
        }
        worldgen::learned::PhysicalTerrainGrid grid;
        grid.region = region;
        grid.samples.resize(static_cast<size_t>(region.height() * region.width()));
        return worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>::ready(
            std::move(grid));
    }

    bool waitUntilEntered(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return ready_.wait_for(lock, timeout, [&] { return entered_; });
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        ready_.notify_all();
    }

    [[nodiscard]] uint64_t callCount() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    worldgen::learned::DeterministicFakeTerrainBackend delegate_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    bool entered_ = false;
    bool released_ = false;
    std::atomic<uint64_t> calls_{0};
};

class GateablePreviewAuthority final : public worldgen::learned::TerrainAuthority {
public:
    explicit GateablePreviewAuthority(worldgen::learned::GenerationIdentity identity)
        : identity_(std::move(identity)) {}

    const worldgen::learned::GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey key,
                worldgen::learned::AuthorityRequestPriority priority) override {
        prepareCalls_.fetch_add(1, std::memory_order_relaxed);
        qualityPrepareCalls_[static_cast<size_t>(key.quality)].fetch_add(1,
                                                                         std::memory_order_relaxed);
        priorityPrepareCalls_[static_cast<size_t>(priority)].fetch_add(1,
                                                                       std::memory_order_relaxed);
        if (!ready_.load(std::memory_order_acquire)) {
            return worldgen::learned::AuthorityResult<
                std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::deferred({
                .code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                .message = "Synthetic cold preview authority page",
                .retriable = true,
            });
        }
        return worldgen::learned::
            AuthorityResult<std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::ready(
                std::make_shared<worldgen::learned::TerrainAuthorityPage>());
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey key,
                worldgen::learned::AuthorityRequestPriority priority,
                worldgen::learned::ProtectedHandoffEpoch epoch) override {
        if (epoch.valid() &&
            (key.quality != worldgen::learned::AuthorityQuality::FINAL ||
             priority != worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF)) {
            return worldgen::learned::AuthorityResult<
                std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::failed({
                .code = worldgen::learned::GenerationFailureCode::INVALID_REQUEST,
                .message = "Synthetic authority rejected an invalid protected handoff",
                .retriable = false,
            });
        }
        if (epoch.valid())
            latestProtectedHandoffEpoch_.store(epoch.value, std::memory_order_release);
        return preparePage(key, priority);
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    queryNative(worldgen::learned::NativeRect, worldgen::learned::AuthorityQuality,
                worldgen::learned::AuthorityRequestPriority) override {
        return unexpected<worldgen::learned::PhysicalTerrainGrid>();
    }

    worldgen::learned::AuthorityResult<std::vector<worldgen::learned::PhysicalTerrainSample>>
    queryNativePoints(std::span<const worldgen::learned::NativePoint>,
                      worldgen::learned::AuthorityQuality,
                      worldgen::learned::AuthorityRequestPriority) override {
        return unexpected<std::vector<worldgen::learned::PhysicalTerrainSample>>();
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(worldgen::learned::NativeRect region,
                                  worldgen::learned::AuthorityRequestPriority) override {
        transientCalls_.fetch_add(1, std::memory_order_relaxed);
        if (readyAfterFirstTransientDeferral_.exchange(false, std::memory_order_acq_rel)) {
            transientReady_.store(true, std::memory_order_release);
            return worldgen::learned::AuthorityResult<
                std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>::deferred({
                .code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                .message = "Synthetic transient terrain became ready before parking",
                .retriable = true,
            });
        }
        if (!transientReady_.load(std::memory_order_acquire)) {
            return worldgen::learned::AuthorityResult<
                std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>::deferred({
                .code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                .message = "Synthetic cold transient terrain",
                .retriable = true,
            });
        }
        std::lock_guard lock(transientMutex_);
        if (!transientGrid_ || transientGrid_->region != region) {
            auto grid = std::make_shared<worldgen::learned::PhysicalTerrainGrid>();
            grid->region = region;
            grid->samples.resize(static_cast<size_t>(region.height() * region.width()));
            transientGrid_ = std::move(grid);
        }
        return worldgen::learned::AuthorityResult<
            std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>::ready(transientGrid_);
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(worldgen::learned::NativeRect region,
                                  worldgen::learned::AuthorityRequestPriority priority,
                                  worldgen::learned::ProtectedHandoffEpoch epoch) override {
        if (epoch.valid())
            latestProtectedHandoffEpoch_.store(epoch.value, std::memory_order_release);
        return queryTransientFinalNativeGrid(region, priority);
    }

    worldgen::learned::TerrainAuthorityCacheMetrics cacheMetrics() const override { return {}; }

    void setReady(bool ready = true) noexcept { ready_.store(ready, std::memory_order_release); }
    void setTransientReady(bool ready = true) noexcept {
        transientReady_.store(ready, std::memory_order_release);
    }
    void setTransientReadyAfterFirstDeferral() noexcept {
        readyAfterFirstTransientDeferral_.store(true, std::memory_order_release);
    }
    [[nodiscard]] uint64_t prepareCalls() const noexcept {
        return prepareCalls_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t
    prepareCalls(worldgen::learned::AuthorityQuality quality) const noexcept {
        return qualityPrepareCalls_[static_cast<size_t>(quality)].load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t
    prepareCalls(worldgen::learned::AuthorityRequestPriority priority) const noexcept {
        return priorityPrepareCalls_[static_cast<size_t>(priority)].load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t transientCalls() const noexcept {
        return transientCalls_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t latestProtectedHandoffEpoch() const noexcept {
        return latestProtectedHandoffEpoch_.load(std::memory_order_acquire);
    }

private:
    template <typename Value> static worldgen::learned::AuthorityResult<Value> unexpected() {
        return worldgen::learned::AuthorityResult<Value>::failed({
            .code = worldgen::learned::GenerationFailureCode::INVALID_REQUEST,
            .message = "Unexpected terrain authority query",
            .retriable = false,
        });
    }

    worldgen::learned::GenerationIdentity identity_;
    std::atomic<bool> ready_{false};
    std::atomic<uint64_t> prepareCalls_{0};
    std::array<std::atomic<uint64_t>, 2> qualityPrepareCalls_{};
    std::array<std::atomic<uint64_t>, 6> priorityPrepareCalls_{};
    std::atomic<bool> transientReady_{false};
    std::atomic<bool> readyAfterFirstTransientDeferral_{false};
    std::atomic<uint64_t> latestProtectedHandoffEpoch_{0};
    std::atomic<uint64_t> transientCalls_{0};
    std::mutex transientMutex_;
    std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid> transientGrid_;
};

std::map<int, float> farTerrainEdge(const FarTerrainMesh& mesh, bool eastEdge) {
    std::map<int, float> result;
    const float edgeX = eastEdge ? static_cast<float>(FAR_TERRAIN_TILE_EDGE) : 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
            unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::WATER) ||
            static_cast<float>(vertex.px) != edgeX) {
            continue;
        }
        result[static_cast<int>(static_cast<float>(vertex.pz))] = static_cast<float>(vertex.py);
    }
    return result;
}

std::map<int, float> farTerrainBoundary(const FarTerrainMesh& mesh, FaceNormal edge) {
    std::map<int, float> result;
    const bool xEdge = edge == FaceNormal::PLUS_X || edge == FaceNormal::MINUS_X;
    const float fixed = edge == FaceNormal::PLUS_X || edge == FaceNormal::PLUS_Z
                            ? static_cast<float>(FAR_TERRAIN_TILE_EDGE)
                            : 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
            (vertex.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) == 0U) {
            continue;
        }
        const float vertexFixed =
            xEdge ? static_cast<float>(vertex.px) : static_cast<float>(vertex.pz);
        if (vertexFixed != fixed)
            continue;
        const int along =
            static_cast<int>(xEdge ? static_cast<float>(vertex.pz) : static_cast<float>(vertex.px));
        const auto [entry, inserted] = result.emplace(along, static_cast<float>(vertex.py));
        if (!inserted)
            REQUIRE(entry->second == static_cast<float>(vertex.py));
    }
    return result;
}

bool farTerrainTopsAreVoxelFlat(const FarTerrainMesh& mesh) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        const float height = static_cast<float>(first.py);
        if (height != std::round(height))
            return false;
        for (const uint32_t corner : {offset + 1, offset + 2, offset + 5}) {
            if (static_cast<float>(mesh.vertices[mesh.indices[corner]].py) != height)
                return false;
        }
    }
    return true;
}

bool farTerrainUsesVoxelFaces(const FarTerrainMesh& mesh, int step) {
    for (uint32_t offset = 0; offset + 5 < mesh.opaqueIndexCount; offset += 6) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if ((first.faceAttr &
             (FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK | FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK)) != 0U) {
            continue;
        }
        const FaceNormal face = unpackFace(first.faceAttr);
        std::array<float, 4> xs{};
        std::array<float, 4> ys{};
        std::array<float, 4> zs{};
        constexpr std::array<uint32_t, 4> QUAD_CORNERS = {0, 1, 2, 5};
        for (size_t corner = 0; corner < QUAD_CORNERS.size(); ++corner) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset + QUAD_CORNERS[corner]]];
            xs[corner] = static_cast<float>(vertex.px);
            ys[corner] = static_cast<float>(vertex.py);
            zs[corner] = static_cast<float>(vertex.pz);
            if (ys[corner] != std::round(ys[corner]))
                return false;
        }
        const auto allEqual = [](const auto& values) {
            return std::all_of(values.begin() + 1, values.end(),
                               [&](float value) { return value == values.front(); });
        };
        if (face == FaceNormal::PLUS_Y) {
            if (!allEqual(ys))
                return false;
            for (float x : xs) {
                if (world_coord::floorMod(static_cast<int64_t>(std::llround(x)),
                                          static_cast<int64_t>(step)) != 0) {
                    return false;
                }
            }
            for (float z : zs) {
                if (world_coord::floorMod(static_cast<int64_t>(std::llround(z)),
                                          static_cast<int64_t>(step)) != 0) {
                    return false;
                }
            }
        } else if (face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X) {
            if (!allEqual(xs))
                return false;
        } else if (face == FaceNormal::PLUS_Z || face == FaceNormal::MINUS_Z) {
            if (!allEqual(zs))
                return false;
        } else {
            return false;
        }
    }
    return true;
}

float expectedVoxelCellHeight(const FarTerrainSource& source, int64_t worldX, int64_t worldZ,
                              FarTerrainStep step) {
    const int width = farTerrainStepSize(step);
    const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(step);
    if (step == FarTerrainStep::ONE) {
        const FarSurfaceSample sample = source.sample(worldX, worldZ, footprint);
        return static_cast<float>(std::floor(sample.geometry.terrainHeight + 0.5));
    }
    double height = 0.0;
    for (const auto [dx, dz] :
         std::array<std::pair<int, int>, 4>{{{0, 0}, {width, 0}, {width, width}, {0, width}}}) {
        const FarSurfaceSample sample = source.sample(worldX + dx, worldZ + dz, footprint);
        height += sample.geometry.terrainHeight;
    }
    return static_cast<float>(std::ceil(height / 4.0));
}

std::optional<float> farTerrainHeightAt(const FarTerrainMesh& mesh, float x, float z) {
    std::optional<float> result;
    for (uint32_t offset = 0; offset + 2 < mesh.opaqueIndexCount; offset += 3) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U) {
            continue;
        }
        const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
        const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
        const double ax = static_cast<float>(first.px);
        const double az = static_cast<float>(first.pz);
        const double bx = static_cast<float>(second.px);
        const double bz = static_cast<float>(second.pz);
        const double cx = static_cast<float>(third.px);
        const double cz = static_cast<float>(third.pz);
        const double denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
        if (std::abs(denominator) <= 1.0e-8)
            continue;
        const double firstWeight = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator;
        const double secondWeight = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator;
        const double thirdWeight = 1.0 - firstWeight - secondWeight;
        constexpr double EPSILON = 1.0e-6;
        if (firstWeight < -EPSILON || secondWeight < -EPSILON || thirdWeight < -EPSILON)
            continue;
        const float height = static_cast<float>(firstWeight * static_cast<float>(first.py) +
                                                secondWeight * static_cast<float>(second.py) +
                                                thirdWeight * static_cast<float>(third.py));
        result = result ? std::max(*result, height) : height;
    }
    return result;
}

std::vector<float> farTerrainHeightRaster(const FarTerrainMesh& mesh, int spacing) {
    const int edge = FAR_TERRAIN_TILE_EDGE / spacing;
    std::vector<float> result(static_cast<size_t>(edge * edge),
                              std::numeric_limits<float>::quiet_NaN());
    for (uint32_t offset = 0; offset + 2 < mesh.opaqueIndexCount; offset += 3) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            (first.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U) {
            continue;
        }
        const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
        const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
        const std::array<float, 3> xs = {static_cast<float>(first.px),
                                         static_cast<float>(second.px),
                                         static_cast<float>(third.px)};
        const std::array<float, 3> zs = {static_cast<float>(first.pz),
                                         static_cast<float>(second.pz),
                                         static_cast<float>(third.pz)};
        const auto [minimumX, maximumX] = std::minmax_element(xs.begin(), xs.end());
        const auto [minimumZ, maximumZ] = std::minmax_element(zs.begin(), zs.end());
        const int firstX = std::clamp(static_cast<int>(std::floor(*minimumX / spacing)), 0, edge);
        const int lastX = std::clamp(static_cast<int>(std::ceil(*maximumX / spacing)), 0, edge);
        const int firstZ = std::clamp(static_cast<int>(std::floor(*minimumZ / spacing)), 0, edge);
        const int lastZ = std::clamp(static_cast<int>(std::ceil(*maximumZ / spacing)), 0, edge);
        const double denominator =
            (zs[1] - zs[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (zs[0] - zs[2]);
        if (std::abs(denominator) <= 1.0e-8)
            continue;
        for (int z = firstZ; z < lastZ; ++z) {
            for (int x = firstX; x < lastX; ++x) {
                const double sampleX = static_cast<double>(x * spacing) + spacing * 0.5;
                const double sampleZ = static_cast<double>(z * spacing) + spacing * 0.5;
                const double firstWeight =
                    ((zs[1] - zs[2]) * (sampleX - xs[2]) + (xs[2] - xs[1]) * (sampleZ - zs[2])) /
                    denominator;
                const double secondWeight =
                    ((zs[2] - zs[0]) * (sampleX - xs[2]) + (xs[0] - xs[2]) * (sampleZ - zs[2])) /
                    denominator;
                const double thirdWeight = 1.0 - firstWeight - secondWeight;
                constexpr double EPSILON = 1.0e-6;
                if (firstWeight < -EPSILON || secondWeight < -EPSILON || thirdWeight < -EPSILON) {
                    continue;
                }
                const float height =
                    static_cast<float>(firstWeight * static_cast<float>(first.py) +
                                       secondWeight * static_cast<float>(second.py) +
                                       thirdWeight * static_cast<float>(third.py));
                float& retained = result[static_cast<size_t>(z * edge + x)];
                retained = std::isfinite(retained) ? std::max(retained, height) : height;
            }
        }
    }
    return result;
}

std::optional<float> farWaterTopHeightAt(const FarTerrainMesh& mesh, float x, float z) {
    const auto signedArea = [](float ax, float az, float bx, float bz, float px, float pz) {
        return (px - bx) * (az - bz) - (ax - bx) * (pz - bz);
    };
    for (size_t offset = mesh.opaqueIndexCount; offset + 2 < mesh.indices.size(); offset += 3) {
        const Vertex& first = mesh.vertices[mesh.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
        const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
        const float firstSign =
            signedArea(static_cast<float>(first.px), static_cast<float>(first.pz),
                       static_cast<float>(second.px), static_cast<float>(second.pz), x, z);
        const float secondSign =
            signedArea(static_cast<float>(second.px), static_cast<float>(second.pz),
                       static_cast<float>(third.px), static_cast<float>(third.pz), x, z);
        const float thirdSign =
            signedArea(static_cast<float>(third.px), static_cast<float>(third.pz),
                       static_cast<float>(first.px), static_cast<float>(first.pz), x, z);
        constexpr float EPSILON = 0.001F;
        const bool hasNegative =
            firstSign < -EPSILON || secondSign < -EPSILON || thirdSign < -EPSILON;
        const bool hasPositive = firstSign > EPSILON || secondSign > EPSILON || thirdSign > EPSILON;
        if (!(hasNegative && hasPositive))
            return static_cast<float>(first.py);
    }
    return std::nullopt;
}

bool farWaterTopCovers(const FarTerrainMesh& mesh, float x, float z) {
    return farWaterTopHeightAt(mesh, x, z).has_value();
}

} // namespace

TEST_CASE("Far terrain chooses globally specified LOD rings", "[render][far-terrain]") {
    REQUIRE_FALSE(farTerrainStepForChunkDistance(31.999).has_value());
    REQUIRE(farTerrainStepForChunkDistance(32.0) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(63.999) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForChunkDistance(64.0) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(127.999) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForChunkDistance(128.0) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(255.999) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForChunkDistance(256.0) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForChunkDistance(511.999) == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(farTerrainStepForChunkDistance(512.0).has_value());
    REQUIRE(FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS == FAR_TERRAIN_NEAR_CHUNK_RADIUS);
    REQUIRE(FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS - FAR_TERRAIN_STEP_ONE_LIMIT_CHUNKS == 32.0);
    REQUIRE(FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS - FAR_TERRAIN_STEP_TWO_LIMIT_CHUNKS == 64.0);
    REQUIRE(FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS - FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS == 128.0);
    REQUIRE(FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS - FAR_TERRAIN_STEP_EIGHT_LIMIT_CHUNKS == 256.0);
    REQUIRE(FAR_TERRAIN_MAX_CHUNK_RADIUS == FAR_TERRAIN_STEP_SIXTEEN_LIMIT_CHUNKS);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_CHUNK_RADIUS == MAX_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Far terrain absolute rings are bounded across negative tile seams",
          "[render][far-terrain][selection][lod][negative][performance][regression]") {
    constexpr std::array cameras = {
        std::pair{0.0, 0.0},       std::pair{127.5, -127.5}, std::pair{255.999, 255.999},
        std::pair{-0.001, -0.001}, std::pair{-256.0, 256.0}, std::pair{-65'536.25, -32'768.75},
    };
    size_t maximumStepTwoTiles = 0;
    for (const auto [cameraX, cameraZ] : cameras) {
        std::vector<FarTerrainViewTile> selected;
        selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
        std::map<std::pair<int64_t, int64_t>, int> stepByTile;
        size_t stepTwoTiles = 0;
        for (const FarTerrainViewTile& tile : selected) {
            REQUIRE(tile.key.step != FarTerrainStep::ONE);
            REQUIRE(tile.key.step != FarTerrainStep::THIRTY_TWO);
            stepTwoTiles += tile.key.step == FarTerrainStep::TWO ? 1 : 0;
            stepByTile.emplace(std::pair{tile.key.tileX, tile.key.tileZ},
                               farTerrainStepSize(tile.key.step));
        }
        maximumStepTwoTiles = std::max(maximumStepTwoTiles, stepTwoTiles);

        constexpr std::array neighborOffsets = {
            std::pair{int64_t{1}, int64_t{0}},
            std::pair{int64_t{0}, int64_t{1}},
        };
        for (const auto& [coordinate, step] : stepByTile) {
            for (const auto [dx, dz] : neighborOffsets) {
                const auto neighbor =
                    stepByTile.find({coordinate.first + dx, coordinate.second + dz});
                if (neighbor == stepByTile.end())
                    continue;
                REQUIRE(std::max(step, neighbor->second) <= 2 * std::min(step, neighbor->second));
            }
        }
    }
    CAPTURE(maximumStepTwoTiles);
    REQUIRE(maximumStepTwoTiles > 0);
}

TEST_CASE("A diagnostic step one tile fits the far terrain upload contract",
          "[render][far-terrain][lod][memory][upload][performance][regression]") {
    const auto mesh =
        FarTerrainMesher::build({-1, -1, FarTerrainStep::ONE}, farTerrainTestSource());
    const size_t uploadBytes =
        mesh->vertices.size() * sizeof(Vertex) + mesh->indices.size() * sizeof(uint32_t);
    const FarTerrainSchedulerLimits limits;
    CAPTURE(mesh->vertices.size(), mesh->indices.size(), uploadBytes, limits.maxCacheBytes);
    REQUIRE(uploadBytes <= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME);
    REQUIRE(uploadBytes <= limits.maxCacheBytes);
}

TEST_CASE("Step one topology sentinels own every integer water column",
          "[render][far-terrain][lod][water][topology][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = SEA_LEVEL;
            sample.ocean = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t x = originX + static_cast<int64_t>(cellX * step);
                const int64_t z = originZ + static_cast<int64_t>(cellZ * step);
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterTopologyPossible = x == 1 && z == 1,
                };
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::ONE}, source);
    REQUIRE(mesh->waterQuadCount > 0);
    REQUIRE(farWaterTopCovers(*mesh, 1.5F, 1.5F));
}

TEST_CASE("An oversized far terrain mesh gets one otherwise empty upload frame",
          "[render][far-terrain][upload][lod][regression]") {
    constexpr size_t BUDGET = FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME;
    REQUIRE(farTerrainUploadFitsFrameBudget(0, BUDGET, BUDGET));
    REQUIRE(farTerrainUploadFitsFrameBudget(0, BUDGET + 1, BUDGET));
    REQUIRE_FALSE(farTerrainUploadFitsFrameBudget(1, BUDGET, BUDGET));
    REQUIRE_FALSE(farTerrainUploadFitsFrameBudget(BUDGET + 1, 1, BUDGET));
    REQUIRE_FALSE(farTerrainUploadFitsFrameBudget(std::numeric_limits<size_t>::max(),
                                                  std::numeric_limits<size_t>::max(), BUDGET));
}

TEST_CASE("Distant uploads preserve a nearby refinement opportunity",
          "[render][far-terrain][upload][lod][priority][regression]") {
    constexpr size_t BUDGET = FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME;
    constexpr size_t RESERVE = FAR_TERRAIN_NEAR_REFINEMENT_UPLOAD_RESERVE_BYTES;
    constexpr size_t ONE_MIB = 1024 * 1024;

    REQUIRE(
        farTerrainUploadFitsPrioritizedFrameBudget(0, BUDGET - RESERVE, BUDGET, RESERVE, false));
    REQUIRE_FALSE(farTerrainUploadFitsPrioritizedFrameBudget(0, BUDGET, BUDGET, RESERVE, false));
    REQUIRE_FALSE(farTerrainUploadFitsPrioritizedFrameBudget(BUDGET - RESERVE, ONE_MIB, BUDGET,
                                                             RESERVE, false));
    REQUIRE(farTerrainUploadFitsPrioritizedFrameBudget(BUDGET - RESERVE, RESERVE, BUDGET, RESERVE,
                                                       true));
    REQUIRE_FALSE(farTerrainUploadFitsPrioritizedFrameBudget(BUDGET - RESERVE, RESERVE + 1, BUDGET,
                                                             RESERVE, true));
    // A single dense local tile retains the ordinary oversized-frame escape.
    REQUIRE(farTerrainUploadFitsPrioritizedFrameBudget(0, BUDGET + 1, BUDGET, RESERVE, true));
}

TEST_CASE("Far terrain arena admission preserves parent coverage and nearby flora",
          "[render][far-terrain][upload][arena][capacity][flora][regression]") {
    constexpr uint64_t VERTEX_CAPACITY = 2ull * 1024 * 1024 * 1024;
    constexpr uint64_t INDEX_CAPACITY = 1ull * 1024 * 1024 * 1024;
    constexpr uint64_t SMALL_ALLOCATION = 1024;
    const uint64_t nearRefinementVertexLimit =
        VERTEX_CAPACITY - FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES;
    const uint64_t nearRefinementIndexLimit =
        INDEX_CAPACITY - FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES;
    const uint64_t floraVertexLimit =
        nearRefinementVertexLimit - FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES;
    const uint64_t floraIndexLimit =
        nearRefinementIndexLimit - FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES;
    const uint64_t refinementVertexLimit =
        floraVertexLimit - FAR_TERRAIN_GPU_VERTEX_FLORA_RESERVE_BYTES;
    const uint64_t refinementIndexLimit =
        floraIndexLimit - FAR_TERRAIN_GPU_INDEX_FLORA_RESERVE_BYTES;

    REQUIRE(farTerrainGpuUploadFitsArena(refinementVertexLimit - SMALL_ALLOCATION,
                                         refinementIndexLimit - SMALL_ALLOCATION, VERTEX_CAPACITY,
                                         INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                         FarTerrainGpuArenaClass::REFINEMENT));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(
        refinementVertexLimit, refinementIndexLimit, VERTEX_CAPACITY, INDEX_CAPACITY,
        SMALL_ALLOCATION, SMALL_ALLOCATION, FarTerrainGpuArenaClass::REFINEMENT));
    REQUIRE(farTerrainGpuUploadFitsArena(refinementVertexLimit, refinementIndexLimit,
                                         VERTEX_CAPACITY, INDEX_CAPACITY, SMALL_ALLOCATION,
                                         SMALL_ALLOCATION, FarTerrainGpuArenaClass::FLORA));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(floraVertexLimit, floraIndexLimit, VERTEX_CAPACITY,
                                               INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                               FarTerrainGpuArenaClass::FLORA));
    REQUIRE(farTerrainGpuUploadFitsArena(floraVertexLimit, floraIndexLimit, VERTEX_CAPACITY,
                                         INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                         FarTerrainGpuArenaClass::NEAR_REFINEMENT));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(
        nearRefinementVertexLimit, nearRefinementIndexLimit, VERTEX_CAPACITY, INDEX_CAPACITY,
        SMALL_ALLOCATION, SMALL_ALLOCATION, FarTerrainGpuArenaClass::NEAR_REFINEMENT));
    const uint64_t coverageVertexLimit =
        VERTEX_CAPACITY - FAR_TERRAIN_GPU_VERTEX_NEAR_REFINEMENT_RESERVE_BYTES;
    const uint64_t coverageIndexLimit =
        INDEX_CAPACITY - FAR_TERRAIN_GPU_INDEX_NEAR_REFINEMENT_RESERVE_BYTES;
    REQUIRE(farTerrainGpuUploadFitsArena(coverageVertexLimit - SMALL_ALLOCATION,
                                         coverageIndexLimit - SMALL_ALLOCATION, VERTEX_CAPACITY,
                                         INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                         FarTerrainGpuArenaClass::COVERAGE));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(
        coverageVertexLimit, coverageIndexLimit, VERTEX_CAPACITY, INDEX_CAPACITY, SMALL_ALLOCATION,
        SMALL_ALLOCATION, FarTerrainGpuArenaClass::COVERAGE));
    REQUIRE(farTerrainGpuUploadFitsArena(
        nearRefinementVertexLimit, nearRefinementIndexLimit, VERTEX_CAPACITY, INDEX_CAPACITY,
        SMALL_ALLOCATION, SMALL_ALLOCATION, FarTerrainGpuArenaClass::CRITICAL_COVERAGE));
    REQUIRE(farTerrainGpuUploadFitsArena(
        nearRefinementVertexLimit, nearRefinementIndexLimit, VERTEX_CAPACITY, INDEX_CAPACITY,
        SMALL_ALLOCATION, SMALL_ALLOCATION, FarTerrainGpuArenaClass::CRITICAL_REFINEMENT));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(VERTEX_CAPACITY, INDEX_CAPACITY, VERTEX_CAPACITY,
                                               INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                               FarTerrainGpuArenaClass::CRITICAL_COVERAGE));
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(VERTEX_CAPACITY, INDEX_CAPACITY, VERTEX_CAPACITY,
                                               INDEX_CAPACITY, SMALL_ALLOCATION, SMALL_ALLOCATION,
                                               FarTerrainGpuArenaClass::CRITICAL_REFINEMENT));

    // Every allocation is charged at the arena's 256-byte granularity.
    REQUIRE_FALSE(farTerrainGpuUploadFitsArena(
        refinementVertexLimit - 255, refinementIndexLimit - 255, VERTEX_CAPACITY, INDEX_CAPACITY, 1,
        1, FarTerrainGpuArenaClass::REFINEMENT));
}

TEST_CASE("Nearby GPU reclamation preserves every coverage and transition owner",
          "[render][far-terrain][upload][arena][eviction][priority][regression]") {
    REQUIRE(farTerrainGpuMayEvictForNear(false, false, false, false, false, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(true, false, false, false, false, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, true, false, false, false, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, false, true, false, false, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, false, false, true, false, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, false, false, false, true, false, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, false, false, false, false, true, false));
    REQUIRE_FALSE(farTerrainGpuMayEvictForNear(false, false, false, false, false, false, true));

    constexpr std::array<std::optional<FarTerrainStep>, 4> LEGAL_DEMOTION_NEIGHBORS{
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO, FarTerrainStep::SIXTEEN, std::nullopt};
    STATIC_REQUIRE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, false, false, false, false,
        false));
    constexpr std::array<std::optional<FarTerrainStep>, 4> TOO_FINE_NEIGHBORS{
        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO, std::nullopt};
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, TOO_FINE_NEIGHBORS, true, false, false, false, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::EIGHT, LEGAL_DEMOTION_NEIGHBORS, true, false, false, false, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, false, false, false, false, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, true, false, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, false, false, true, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, true, false, false, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, false, true, false, false,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, false, false, false, true,
        false));
    STATIC_REQUIRE_FALSE(farTerrainDisplayedRefinementMayYieldToParentForNear(
        FarTerrainStep::SIXTEEN, LEGAL_DEMOTION_NEIGHBORS, true, false, false, false, false, false,
        true));
}

TEST_CASE("Far terrain planner timing keeps heap-free p95 and maximum counters",
          "[render][far-terrain][performance][planner]") {
    FarTerrainPlannerTimingHistogram timing;
    for (size_t sample = 0; sample < 95; ++sample)
        timing.record(0.05);
    for (size_t sample = 0; sample < 5; ++sample)
        timing.record(3.25);
    timing.record(-1.0);
    timing.record(std::numeric_limits<double>::quiet_NaN());

    REQUIRE(timing.sampleCount() == 100);
    REQUIRE(timing.percentile95Milliseconds() == Catch::Approx(0.1F));
    REQUIRE(timing.maximumMilliseconds() == Catch::Approx(3.25F));
    timing.clear();
    REQUIRE(timing.sampleCount() == 0);
    REQUIRE(timing.percentile95Milliseconds() == 0.0F);
    REQUIRE(timing.maximumMilliseconds() == 0.0F);
}

TEST_CASE("Far terrain selection refreshes only after meaningful camera or view changes",
          "[render][far-terrain][selection][performance][regression]") {
    constexpr std::optional<std::pair<double, double>> ORIGIN = std::pair{8.0, -8.0};

    REQUIRE(farTerrainCameraMovementRequiresRefresh(std::nullopt, 8.0, -8.0, 4.0));
    REQUIRE_FALSE(farTerrainCameraMovementRequiresRefresh(ORIGIN, 11.999, -8.0, 4.0));
    REQUIRE(farTerrainCameraMovementRequiresRefresh(ORIGIN, 12.0, -8.0, 4.0));
    REQUIRE(farTerrainSelectionRequiresRefresh(std::nullopt, 8.0, -8.0, 512, 512));
    REQUIRE_FALSE(farTerrainSelectionRequiresRefresh(ORIGIN, 23.999, -8.0, 512, 512));
    REQUIRE(farTerrainSelectionRequiresRefresh(ORIGIN, 24.0, -8.0, 512, 512));
    REQUIRE(farTerrainSelectionRequiresRefresh(ORIGIN, 20.0, 4.0, 512, 512));
    REQUIRE(farTerrainSelectionRequiresRefresh(ORIGIN, 8.0, -8.0, 256, 512));
}

TEST_CASE("Stable chunk motion preserves far horizon residency membership",
          "[render][far-terrain][selection][residency][performance][movement][regression]") {
    std::vector<FarTerrainViewTile> initial(3);
    initial[0].key = {-1, 0, FarTerrainStep::FOUR};
    initial[1].key = {0, 0, FarTerrainStep::TWO};
    initial[2].key = {1, 0, FarTerrainStep::FOUR};
    for (FarTerrainViewTile& tile : initial)
        tile.distanceChunks = 64.0;
    std::vector<FarTerrainKey> order;
    buildFarTerrainResidencyOrder(initial, order);
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());

    // A one-chunk camera move may reorder equidistant coordinates without
    // changing any desired tier or immutable key. That refresh must not be
    // interpreted as a global residency revision.
    std::vector<FarTerrainViewTile> moved{initial[2], initial[1], initial[0]};
    REQUIRE(farTerrainSelectionRequiresRefresh(std::pair{0.0, 0.0}, CHUNK_EDGE, 0.0,
                                               FAR_TERRAIN_MAX_CHUNK_RADIUS,
                                               FAR_TERRAIN_MAX_CHUNK_RADIUS));
    REQUIRE_FALSE(farTerrainResidencyOrderMatches(moved, order));
    REQUIRE(farTerrainResidencyMembershipMatches(moved, wanted));
}

TEST_CASE("Far terrain desired metrics remain cached until an input changes",
          "[render][far-terrain][selection][screen-error][performance][regression]") {
    constexpr uint32_t VIEWPORT_HEIGHT = 1536;
    constexpr double VERTICAL_FOV = 70.0 * std::numbers::pi / 180.0;

    REQUIRE_FALSE(farTerrainDesiredMetricsRequireRefresh(
        false, false, VIEWPORT_HEIGHT, VIEWPORT_HEIGHT, VERTICAL_FOV, VERTICAL_FOV, true, true));
    REQUIRE(farTerrainDesiredMetricsRequireRefresh(true, false, VIEWPORT_HEIGHT, VIEWPORT_HEIGHT,
                                                   VERTICAL_FOV, VERTICAL_FOV, true, true));
    REQUIRE(farTerrainDesiredMetricsRequireRefresh(false, true, VIEWPORT_HEIGHT, VIEWPORT_HEIGHT,
                                                   VERTICAL_FOV, VERTICAL_FOV, true, true));
    REQUIRE(farTerrainDesiredMetricsRequireRefresh(false, false, VIEWPORT_HEIGHT,
                                                   VIEWPORT_HEIGHT + 1, VERTICAL_FOV, VERTICAL_FOV,
                                                   true, true));
    REQUIRE(farTerrainDesiredMetricsRequireRefresh(false, false, VIEWPORT_HEIGHT, VIEWPORT_HEIGHT,
                                                   VERTICAL_FOV, VERTICAL_FOV + 0.01, true, true));
    REQUIRE(farTerrainDesiredMetricsRequireRefresh(false, false, VIEWPORT_HEIGHT, VIEWPORT_HEIGHT,
                                                   VERTICAL_FOV, VERTICAL_FOV, true, false));
}

TEST_CASE("Far terrain tiers map explicitly to surface footprints",
          "[render][far-terrain][lod][sampling][contract]") {
    constexpr std::array steps = {
        FarTerrainStep::ONE,   FarTerrainStep::TWO,     FarTerrainStep::FOUR,
        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    constexpr std::array footprints = {
        worldgen::SurfaceFootprint::BLOCK_1,  worldgen::SurfaceFootprint::BLOCK_2,
        worldgen::SurfaceFootprint::BLOCK_4,  worldgen::SurfaceFootprint::BLOCK_8,
        worldgen::SurfaceFootprint::BLOCK_16, worldgen::SurfaceFootprint::BLOCK_32,
    };
    for (size_t index = 0; index < steps.size(); ++index) {
        CAPTURE(index);
        REQUIRE(farTerrainSurfaceFootprint(steps[index]) == footprints[index]);
        REQUIRE(farTerrainStepForSize(worldgen::surfaceFootprintWidth(footprints[index])) ==
                steps[index]);
    }
    REQUIRE_FALSE(farTerrainStepForSize(3).has_value());
}

TEST_CASE("Far terrain transition cells cover every edge with positive winding",
          "[render][far-terrain][lod][transition][topology][regression]") {
    const auto edgeBit = [](FaceNormal edge) { return 1U << static_cast<uint8_t>(edge); };
    constexpr std::array EDGES = {FaceNormal::PLUS_X, FaceNormal::MINUS_X, FaceNormal::PLUS_Z,
                                  FaceNormal::MINUS_Z};
    const std::array<uint32_t, 8> masks = {
        edgeBit(FaceNormal::PLUS_X),
        edgeBit(FaceNormal::MINUS_X),
        edgeBit(FaceNormal::PLUS_Z),
        edgeBit(FaceNormal::MINUS_Z),
        edgeBit(FaceNormal::MINUS_X) | edgeBit(FaceNormal::MINUS_Z),
        edgeBit(FaceNormal::PLUS_X) | edgeBit(FaceNormal::MINUS_Z),
        edgeBit(FaceNormal::MINUS_X) | edgeBit(FaceNormal::PLUS_Z),
        edgeBit(FaceNormal::PLUS_X) | edgeBit(FaceNormal::PLUS_Z),
    };
    for (const int coarseStep : {2, 4, 8, 16, 32}) {
        for (const uint32_t mask : masks) {
            CAPTURE(coarseStep, mask);
            const FarTerrainTransitionTopology topology = farTerrainTransitionCellTopology(
                coarseStep, FAR_TERRAIN_TRANSITION_SAMPLE_STEP, mask);
            REQUIRE(topology.vertexCount >= 5);
            REQUIRE(topology.indexCount % 3 == 0);
            std::set<std::array<uint8_t, 3>> uniqueTriangles;
            double area = 0.0;
            for (size_t offset = 0; offset < topology.indexCount; offset += 3) {
                const uint8_t firstIndex = topology.indices[offset];
                const uint8_t secondIndex = topology.indices[offset + 1];
                const uint8_t thirdIndex = topology.indices[offset + 2];
                REQUIRE(firstIndex < topology.vertexCount);
                REQUIRE(secondIndex < topology.vertexCount);
                REQUIRE(thirdIndex < topology.vertexCount);
                std::array<uint8_t, 3> identity = {firstIndex, secondIndex, thirdIndex};
                std::ranges::sort(identity);
                REQUIRE(uniqueTriangles.insert(identity).second);
                const FarTerrainTransitionVertex& first = topology.vertices[firstIndex];
                const FarTerrainTransitionVertex& second = topology.vertices[secondIndex];
                const FarTerrainTransitionVertex& third = topology.vertices[thirdIndex];
                const int twiceArea = (second.z - first.z) * (third.x - first.x) -
                                      (second.x - first.x) * (third.z - first.z);
                REQUIRE(twiceArea > 0);
                area += static_cast<double>(twiceArea) * 0.5;
            }
            REQUIRE(area == Catch::Approx(static_cast<double>(coarseStep * coarseStep)));

            for (const FaceNormal edge : EDGES) {
                const uint32_t bit = edgeBit(edge);
                if ((mask & bit) == 0)
                    continue;
                std::set<int> coordinates;
                for (const FarTerrainTransitionVertex& vertex :
                     std::span(topology.vertices).first(topology.vertexCount)) {
                    if ((vertex.boundaryEdgeMask & bit) == 0)
                        continue;
                    const bool xEdge = edge == FaceNormal::PLUS_X || edge == FaceNormal::MINUS_X;
                    coordinates.insert(xEdge ? vertex.z : vertex.x);
                }
                REQUIRE(coordinates.size() ==
                        static_cast<size_t>(coarseStep / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1));
                int expected = 0;
                for (const int coordinate : coordinates) {
                    REQUIRE(coordinate == expected);
                    expected += FAR_TERRAIN_TRANSITION_SAMPLE_STEP;
                }
            }
        }
    }
    REQUIRE_THROWS_AS(farTerrainTransitionCellTopology(
                          16, 8, edgeBit(FaceNormal::PLUS_X) | edgeBit(FaceNormal::MINUS_X)),
                      std::invalid_argument);
}

TEST_CASE("Far terrain neighbor compatibility permits only one displayed tier",
          "[render][far-terrain][lod][transition][neighbor][regression]") {
    std::array<std::optional<FarTerrainStep>, 4> neighbors{};
    neighbors[0] = FarTerrainStep::FOUR;
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::TWO, neighbors));
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::EIGHT, neighbors));
    REQUIRE_FALSE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::SIXTEEN, neighbors));
    neighbors[1] = FarTerrainStep::TWO;
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::FOUR, neighbors));
    REQUIRE_FALSE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::EIGHT, neighbors));
    neighbors = {};
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::THIRTY_TWO, neighbors));
}

TEST_CASE("Far terrain transition topology stays within its fixed mesh budget",
          "[render][far-terrain][lod][transition][topology][performance]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 80.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    constexpr std::array STEPS = {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
                                  FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO};
    constexpr std::array<uint32_t, STEPS.size()> EXPECTED_TRIANGLES = {2032, 1264, 880, 688, 592};
    constexpr std::array<size_t, STEPS.size()> EXPECTED_VERTICES = {2540, 1516, 1004, 748, 620};
    for (size_t index = 0; index < STEPS.size(); ++index) {
        const auto mesh = FarTerrainMesher::build({-7, -11, STEPS[index]}, source);
        const size_t transitionVertices =
            std::ranges::count_if(mesh->vertices, [](const Vertex& vertex) {
                return (vertex.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) != 0U;
            });
        const size_t transitionBytes = transitionVertices * sizeof(Vertex) +
                                       mesh->transitionTriangleCount * 3 * sizeof(uint32_t);
        CAPTURE(farTerrainStepSize(STEPS[index]), transitionVertices, transitionBytes);
        REQUIRE(mesh->transitionTriangleCount == EXPECTED_TRIANGLES[index]);
        REQUIRE(transitionVertices == EXPECTED_VERTICES[index]);
        REQUIRE(transitionBytes <= 65'024);
        REQUIRE(std::ranges::none_of(mesh->vertices, [](const Vertex& vertex) {
            return (vertex.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) != 0U &&
                   unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y;
        }));
    }
}

TEST_CASE("Far transition wedges close only internal source terrain discontinuities",
          "[render][far-terrain][lod][transition][topology][seam][regression]") {
    constexpr int64_t SPLIT_X = 80;
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < SPLIT_X ? 100.0 : 40.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t, int step, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const double height =
                    originX + static_cast<int64_t>(x * step) < SPLIT_X ? 100.0 : 40.0;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = height,
                    .minimumTerrainHeight = height,
                    .maximumTerrainHeight = height,
                };
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);
    size_t discontinuityTriangles = 0;
    for (size_t offset = 0; offset + 2 < mesh->opaqueIndexCount; offset += 3) {
        const Vertex& first = mesh->vertices[mesh->indices[offset]];
        if ((first.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) == 0U ||
            unpackFace(first.faceAttr) == FaceNormal::PLUS_Y) {
            continue;
        }
        const Vertex& second = mesh->vertices[mesh->indices[offset + 1]];
        const Vertex& third = mesh->vertices[mesh->indices[offset + 2]];
        CAPTURE(offset, unpackFace(first.faceAttr));
        REQUIRE((unpackFace(first.faceAttr) == FaceNormal::PLUS_X ||
                 unpackFace(first.faceAttr) == FaceNormal::MINUS_X));
        REQUIRE(static_cast<float>(first.px) == static_cast<float>(SPLIT_X));
        REQUIRE(static_cast<float>(second.px) == static_cast<float>(SPLIT_X));
        REQUIRE(static_cast<float>(third.px) == static_cast<float>(SPLIT_X));
        REQUIRE(static_cast<float>(first.px) > 0.0F);
        REQUIRE(static_cast<float>(first.px) < FAR_TERRAIN_TILE_EDGE);
        const Vec3 firstPosition{static_cast<float>(first.px), static_cast<float>(first.py),
                                 static_cast<float>(first.pz)};
        const Vec3 secondPosition{static_cast<float>(second.px), static_cast<float>(second.py),
                                  static_cast<float>(second.pz)};
        const Vec3 thirdPosition{static_cast<float>(third.px), static_cast<float>(third.py),
                                 static_cast<float>(third.pz)};
        REQUIRE((secondPosition - firstPosition).cross(thirdPosition - firstPosition).lengthSq() >
                0.0F);
        const int64_t probeZ =
            std::clamp<int64_t>(static_cast<int64_t>(std::llround((static_cast<float>(first.pz) +
                                                                   static_cast<float>(second.pz) +
                                                                   static_cast<float>(third.pz)) /
                                                                  3.0F)),
                                0, FAR_TERRAIN_TILE_EDGE - 1);
        const double westHeight = testFarGeometry(source, SPLIT_X - 1, probeZ).terrainHeight;
        const double eastHeight = testFarGeometry(source, SPLIT_X, probeZ).terrainHeight;
        REQUIRE(westHeight != eastHeight);
        ++discontinuityTriangles;
    }
    REQUIRE(discontinuityTriangles == 4);
}

TEST_CASE("Far terrain samples one material palette per active LOD cell",
          "[render][far-terrain][material][lod][seam]") {
    std::array<std::set<std::pair<int64_t, int64_t>>, 6> sampledFootprints;
    FarTerrainSource source;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        sampledFootprints[static_cast<size_t>(std::countr_zero(static_cast<unsigned int>(
                              worldgen::surfaceFootprintWidth(footprint))))]
            .emplace(x, z);
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 72.0;
        const int64_t cellX = world_coord::floorDiv(
            x, static_cast<int64_t>(worldgen::surfaceFootprintWidth(footprint)));
        const int64_t cellZ = world_coord::floorDiv(
            z, static_cast<int64_t>(worldgen::surfaceFootprintWidth(footprint)));
        const BlockType material = world_coord::floorMod(cellX + cellZ, int64_t{2}) == 0
                                       ? BlockType::LIMESTONE
                                       : BlockType::ANDESITE;
        return FarSurfaceSample{
            .geometry = sample,
            .footprintMinimumTerrainHeight = sample.terrainHeight,
            .footprintMaximumTerrainHeight = sample.terrainHeight,
            .materialPalette = testMaterialPalette(material),
        };
    };

    const auto mesh = FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::TWO}, source);
    const auto& stepTwoSamples = sampledFootprints[1];
    // Boundary risers inspect one neighboring control ring, but the active
    // cell grid must still contain every footprint sample exactly once.
    REQUIRE(stepTwoSamples.size() >= 129 * 129);
    for (int z = 0; z <= 128; ++z) {
        for (int x = 0; x <= 128; ++x)
            REQUIRE(stepTwoSamples.contains({-256 + x * 2, -256 + z * 2}));
    }
    REQUIRE(std::any_of(stepTwoSamples.begin(), stepTwoSamples.end(), [](const auto& coordinate) {
        return world_coord::floorMod(coordinate.first, int64_t{32}) != 0 ||
               world_coord::floorMod(coordinate.second, int64_t{32}) != 0;
    }));
    bool sawLimestone = false;
    bool sawAndesite = false;
    for (const Vertex& vertex : mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        const uint8_t texture = unpackTextureLayer(vertex.faceAttr);
        sawLimestone = sawLimestone || texture == static_cast<uint8_t>(BlockType::LIMESTONE);
        sawAndesite = sawAndesite || texture == static_cast<uint8_t>(BlockType::ANDESITE);
    }
    REQUIRE(sawLimestone);
    REQUIRE(sawAndesite);

    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::FOUR}, source);
    REQUIRE(sampledFootprints[2].size() >= 65 * 65);

    FarTerrainMesher::build(FarTerrainKey{-1, -1, FarTerrainStep::SIXTEEN}, source);
    REQUIRE(sampledFootprints[4].size() >= 17 * 17);
}

TEST_CASE("Fine scheduler terrain follows its filtered footprint material palette",
          "[render][far-terrain][scheduler][material][water][lod][seam][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    // This is a scheduler and filtered-material contract test, not a legacy
    // BasinSolver throughput benchmark. The legacy diagnostic generator can
    // spend seconds materializing an unrelated cold basin before this tiny
    // palette fixture reaches the worker. Keep the requested negative tile,
    // water threshold, and footprint disagreement explicit and local.
    constexpr worldgen::WaterBodyId LAKE_ID = 0x5445'5354'4C41'4B45ULL;
    FarTerrainSource source;
    source.sample = [](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        const int width = worldgen::surfaceFootprintWidth(footprint);
        const int64_t coarseX = world_coord::floorDiv(x, int64_t{4});
        const int64_t coarseZ = world_coord::floorDiv(z, int64_t{4});
        const bool lake = world_coord::floorMod(coarseX + coarseZ * 3, int64_t{5}) == 0;
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = lake && width >= 4 ? 61.0 : 72.0;
        // Block-resolution terrain remains just above the same lake stage,
        // which is the explicit filtered-submersion case the mesh must carry.
        if (lake && width == 1)
            geometry.terrainHeight = 65.0;
        geometry.waterSurface = SEA_LEVEL;
        geometry.waterBodyId = lake ? LAKE_ID : worldgen::NO_WATER_BODY;
        geometry.lake = lake;
        const BlockType material = world_coord::floorMod(coarseX + coarseZ, int64_t{2}) == 0
                                       ? BlockType::GRASS
                                       : BlockType::STONE;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = geometry.terrainHeight,
            .footprintMaximumTerrainHeight = geometry.terrainHeight,
            .materialPalette = testMaterialPalette(material),
        };
    };
    source.materialRank = [](int64_t, int64_t) { return 0.0; };
    const FarTerrainSource expectedSource = source;
    FarTerrainScheduler scheduler(std::move(source), limits);
    constexpr FarTerrainKey KEY{-54, 3, FarTerrainStep::FOUR};
    REQUIRE(scheduler.enqueue(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const FarTerrainSchedulerStats timeoutStats = scheduler.stats();
    scheduler.shutdown();
    CAPTURE(timeoutStats.inFlight, timeoutStats.activeWorkers, timeoutStats.queued,
            timeoutStats.completed, timeoutStats.built, timeoutStats.canceled, timeoutStats.failed,
            timeoutStats.deferred);
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);

    // This patch is a lake threshold where block-resolution density and the
    // step-four filtered surface disagree about submersion. The far sample's
    // footprint owns geometry and its palette together, so derive expected
    // materials from that contract instead of pinning a former palette entry.
    std::set<BlockType> expectedMaterials;
    bool sawSubmerged = false;
    bool sawDry = false;
    bool sawFilteredSubmersionDifference = false;
    constexpr int64_t ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    constexpr int STEP = static_cast<int>(FarTerrainStep::FOUR);
    for (int localZ = 64; localZ < 128; localZ += STEP) {
        for (int localX = 64; localX < 128; localX += STEP) {
            const int64_t worldX = ORIGIN_X + localX;
            const int64_t worldZ = ORIGIN_Z + localZ;
            const FarSurfaceSample canonical =
                expectedSource.sample(worldX, worldZ, worldgen::SurfaceFootprint::BLOCK_1);
            const FarSurfaceSample filtered =
                expectedSource.sample(worldX, worldZ, worldgen::SurfaceFootprint::BLOCK_4);
            const auto submerged = [](const FarTerrainGeometrySample& geometry) {
                return geometry.lake && geometry.waterSurface > geometry.terrainHeight + 0.01;
            };
            const bool filteredSubmerged = submerged(filtered.geometry);
            sawSubmerged = sawSubmerged || filteredSubmerged;
            sawDry = sawDry || !filteredSubmerged;
            sawFilteredSubmersionDifference = sawFilteredSubmersionDifference ||
                                              filteredSubmerged != submerged(canonical.geometry);
            if (!filteredSubmerged) {
                const BlockType expectedMaterial = worldgen::surface_material::selectMaterial(
                    filtered.materialPalette,
                    expectedSource.materialRank(worldX + STEP / 2, worldZ + STEP / 2));
                expectedMaterials.insert(expectedMaterial);
            }
        }
    }

    std::set<BlockType> observedMaterials;
    for (const Vertex& vertex : completed.front().mesh->vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y || vertex.px <= 64 ||
            vertex.px >= 128 || vertex.pz <= 64 || vertex.pz >= 128) {
            continue;
        }
        const BlockType material = static_cast<BlockType>(unpackTextureLayer(vertex.faceAttr));
        if (material != BlockType::WATER)
            observedMaterials.insert(material);
    }
    REQUIRE(sawSubmerged);
    REQUIRE(sawDry);
    REQUIRE(sawFilteredSubmersionDifference);
    REQUIRE(expectedMaterials.size() >= 2);
    // Canopy top faces can share this horizontal window, so the mesh may
    // contain additional leaf materials alongside every terrain material.
    REQUIRE(std::ranges::includes(observedMaterials, expectedMaterials));
}

TEST_CASE("Coverage parents separate filtered voxel tops from conservative bounds",
          "[render][far-terrain][coverage][lod][bounds]") {
    FarTerrainSource source;
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint footprint) {
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = 80.0;
        const bool coverageParent = worldgen::surfaceFootprintWidth(footprint) >= 16;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = coverageParent ? 72.0 : 80.0,
            .footprintMaximumTerrainHeight = coverageParent ? 91.0 : 80.0,
            .materialPalette = testMaterialPalette(BlockType::STONE),
        };
    };

    const auto exact = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const auto coverage = FarTerrainMesher::build({0, 0, FAR_TERRAIN_BASE_STEP}, source);
    const auto terrainHeights = [](const FarTerrainMesh& mesh) {
        std::set<float> result;
        for (const Vertex& vertex : mesh.vertices) {
            if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::WATER)) {
                result.insert(static_cast<float>(vertex.py));
            }
        }
        return result;
    };
    REQUIRE(terrainHeights(*exact) == std::set<float>{80.0F});
    REQUIRE(terrainHeights(*coverage) == std::set<float>{80.0F});
    REQUIRE(coverage->surfaceBounds.minY == 72.0F);
    REQUIRE(exact->surfaceBounds.maxY == 80.0F);
    REQUIRE(coverage->surfaceBounds.maxY == 91.0F);
}

TEST_CASE("Batched far cell bounds cover subcell relief and shorelines at every LOD",
          "[render][far-terrain][coverage][lod][bounds][water][seam][regression]") {
    constexpr std::array STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    for (const FarTerrainStep terrainStep : STEPS) {
        const int step = farTerrainStepSize(terrainStep);
        const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
        constexpr FarTerrainKey BASE_KEY{-2, -3, FarTerrainStep::TWO};
        const FarTerrainKey key{BASE_KEY.tileX, BASE_KEY.tileZ, terrainStep};
        const int64_t originX = key.tileX * FAR_TERRAIN_TILE_EDGE;
        const int64_t originZ = key.tileZ * FAR_TERRAIN_TILE_EDGE;
        const int64_t shorelineX = originX + FAR_TERRAIN_TILE_EDGE / 2 + step / 2;
        const int peakCellX = cellEdge * 3 / 4;
        const int peakCellZ = cellEdge * 3 / 4;
        const int trenchCellX = 0;
        const int trenchCellZ = cellEdge / 4;
        const int64_t peakWorldX = originX + static_cast<int64_t>(peakCellX * step);
        const int64_t peakWorldZ = originZ + static_cast<int64_t>(peakCellZ * step);
        const int64_t trenchWorldX = originX;
        const int64_t trenchWorldZ = originZ + static_cast<int64_t>(trenchCellZ * step);
        int callbackCalls = 0;
        int64_t callbackOriginX = 0;
        int64_t callbackOriginZ = 0;
        int callbackWidth = 0;
        int callbackHeight = 0;
        worldgen::SurfaceFootprint callbackFootprint = worldgen::SurfaceFootprint::BLOCK_1;
        FarTerrainSource source = testFarTerrainSource(
            [shorelineX](int64_t x, int64_t) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = x < shorelineX ? 40.0 : 96.0;
                sample.waterSurface = SEA_LEVEL;
                sample.lake = x < shorelineX;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample& sample) {
                return sample.lake ? BlockType::CLAY : BlockType::STONE;
            });
        source.cellBoundsGrid = [&](int64_t gridOriginX, int64_t gridOriginZ, int gridStep,
                                    int cellWidth, int cellHeight,
                                    worldgen::SurfaceFootprint footprint,
                                    std::span<FarTerrainCellBounds> output) {
            ++callbackCalls;
            callbackOriginX = gridOriginX;
            callbackOriginZ = gridOriginZ;
            callbackWidth = cellWidth;
            callbackHeight = cellHeight;
            callbackFootprint = footprint;
            if (gridStep != step || output.size() != static_cast<size_t>(cellWidth * cellHeight)) {
                throw std::invalid_argument("unexpected test cell bounds grid");
            }
            for (int z = 0; z < cellHeight; ++z) {
                for (int x = 0; x < cellWidth; ++x) {
                    const int64_t worldX = gridOriginX + static_cast<int64_t>(x * gridStep);
                    const int64_t worldZ = gridOriginZ + static_cast<int64_t>(z * gridStep);
                    const int64_t maximumX = worldX + gridStep;
                    double minimum = maximumX <= shorelineX ? 40.0 : 96.0;
                    double maximum = minimum;
                    if (worldX < shorelineX && maximumX > shorelineX) {
                        minimum = 40.0;
                        maximum = 96.0;
                    }
                    if (worldX == peakWorldX && worldZ == peakWorldZ) {
                        minimum = 88.9;
                        maximum = 221.25;
                    }
                    if (worldX == trenchWorldX && worldZ == trenchWorldZ) {
                        minimum = -33.2;
                        maximum = 102.1;
                    }
                    output[static_cast<size_t>(z * cellWidth + x)] = {
                        .terrainHeight = minimum,
                        .minimumTerrainHeight = minimum,
                        .maximumTerrainHeight = maximum,
                    };
                }
            }
        };

        const auto mesh = FarTerrainMesher::build(key, source);
        CAPTURE(step);
        REQUIRE(callbackCalls == 1);
        REQUIRE(callbackOriginX == originX - step);
        REQUIRE(callbackOriginZ == originZ - step);
        REQUIRE(callbackWidth == cellEdge + 2);
        REQUIRE(callbackHeight == cellEdge + 2);
        REQUIRE(worldgen::surfaceFootprintWidth(callbackFootprint) == step);
        REQUIRE(farTerrainHeightAt(*mesh, static_cast<float>(peakCellX * step + step / 2),
                                   static_cast<float>(peakCellZ * step + step / 2)) == 89.0F);
        REQUIRE(farTerrainHeightAt(*mesh, static_cast<float>(trenchCellX * step + step / 2),
                                   static_cast<float>(trenchCellZ * step + step / 2)) == -33.0F);
        REQUIRE(mesh->surfaceBounds.minY == -34.0F);
        REQUIRE(mesh->surfaceBounds.maxY == 222.0F);
        REQUIRE(mesh->bounds.minY == -33.0F);
        REQUIRE(mesh->bounds.maxY < mesh->surfaceBounds.maxY);
        REQUIRE(
            farWaterTopCovers(*mesh, FAR_TERRAIN_TILE_EDGE * 0.25F, FAR_TERRAIN_TILE_EDGE * 0.5F));
        REQUIRE_FALSE(farWaterTopCovers(*mesh, static_cast<float>(peakCellX * step + step / 2),
                                        static_cast<float>(peakCellZ * step + step / 2)));

        constexpr int PATCHES_PER_EDGE = FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int peakPatchX = peakCellX * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int peakPatchZ = peakCellZ * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const FarTerrainBounds& peakPatch =
            mesh->occluderPatches[static_cast<size_t>(peakPatchZ * PATCHES_PER_EDGE + peakPatchX)];
        REQUIRE(peakPatch.maxY == 222.0F);
        const int trenchPatchX = trenchCellX * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const int trenchPatchZ = trenchCellZ * step / FAR_TERRAIN_OCCLUDER_PATCH_EDGE;
        const FarTerrainBounds& trenchPatch = mesh->occluderPatches[static_cast<size_t>(
            trenchPatchZ * PATCHES_PER_EDGE + trenchPatchX)];
        REQUIRE(trenchPatch.minY == -34.0F);
    }
}

TEST_CASE("Far cell bounds stitch negative tile faces independent of build order",
          "[render][far-terrain][coverage][lod][bounds][seam][determinism][regression]") {
    constexpr std::array STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    for (const FarTerrainStep terrainStep : STEPS) {
        const int step = farTerrainStepSize(terrainStep);
        size_t boundsCalls = 0;
        FarTerrainSource source = testFarTerrainSource(
            [](int64_t, int64_t) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = 90.0;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::ANDESITE; });
        source.cellBoundsGrid = [&](int64_t originX, int64_t originZ, int gridStep, int cellWidth,
                                    int cellHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainCellBounds> output) {
            ++boundsCalls;
            for (int z = 0; z < cellHeight; ++z) {
                for (int x = 0; x < cellWidth; ++x) {
                    const int64_t worldX = originX + static_cast<int64_t>(x * gridStep);
                    const int64_t worldZ = originZ + static_cast<int64_t>(z * gridStep);
                    const int64_t cellX = world_coord::floorDiv(worldX, int64_t{gridStep});
                    const int64_t cellZ = world_coord::floorDiv(worldZ, int64_t{gridStep});
                    const double minimum =
                        70.0 + world_coord::floorMod(cellX + cellZ * 3, int64_t{9});
                    output[static_cast<size_t>(z * cellWidth + x)] = {
                        .terrainHeight = minimum,
                        .minimumTerrainHeight = minimum,
                        .maximumTerrainHeight = minimum + 0.25,
                    };
                }
            }
        };
        const FarTerrainKey leftKey{-1, -1, terrainStep};
        const FarTerrainKey rightKey{0, -1, terrainStep};
        const auto rightFirst = FarTerrainMesher::build(rightKey, source);
        const auto leftSecond = FarTerrainMesher::build(leftKey, source);
        const auto leftFirst = FarTerrainMesher::build(leftKey, source);
        const auto rightSecond = FarTerrainMesher::build(rightKey, source);
        CAPTURE(step);
        REQUIRE(boundsCalls == 4);
        REQUIRE(leftFirst->deterministicHash == leftSecond->deterministicHash);
        REQUIRE(rightFirst->deterministicHash == rightSecond->deterministicHash);
        REQUIRE(leftFirst->surfaceBounds.minY == leftSecond->surfaceBounds.minY);
        REQUIRE(leftFirst->surfaceBounds.maxY == leftSecond->surfaceBounds.maxY);
        REQUIRE(rightFirst->surfaceBounds.minY == rightSecond->surfaceBounds.minY);
        REQUIRE(rightFirst->surfaceBounds.maxY == rightSecond->surfaceBounds.maxY);

        REQUIRE(farTerrainBoundary(*leftFirst, FaceNormal::PLUS_X) ==
                farTerrainBoundary(*rightFirst, FaceNormal::MINUS_X));
        REQUIRE(
            farTerrainBoundary(*leftFirst, FaceNormal::PLUS_X).size() ==
            static_cast<size_t>(FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1));
    }
}

TEST_CASE("Far cell bounds remain stable after scheduler cache eviction",
          "[render][far-terrain][coverage][bounds][scheduler][cache][determinism][regression]") {
    auto boundsCalls = std::make_shared<std::atomic<size_t>>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 80.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::BASALT; });
    source.cellBoundsGrid = [boundsCalls](int64_t originX, int64_t originZ, int step, int cellWidth,
                                          int cellHeight, worldgen::SurfaceFootprint,
                                          std::span<FarTerrainCellBounds> output) {
        boundsCalls->fetch_add(1, std::memory_order_relaxed);
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t worldX = originX + static_cast<int64_t>(x * step);
                const int64_t worldZ = originZ + static_cast<int64_t>(z * step);
                const int64_t rank =
                    world_coord::floorMod(world_coord::floorDiv(worldX, int64_t{step}) * 5 +
                                              world_coord::floorDiv(worldZ, int64_t{step}) * 7,
                                          int64_t{13});
                const double minimum = 64.0 + rank;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = minimum,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = minimum + 8.25,
                };
            }
        }
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    constexpr FarTerrainKey LEFT{-1, -1, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey RIGHT{0, -1, FarTerrainStep::EIGHT};
    const auto buildPass = [&](std::array<FarTerrainKey, 2> keys) {
        for (const FarTerrainKey key : keys)
            REQUIRE(scheduler.enqueue(key));
        std::vector<FarTerrainResult> results;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (results.size() < keys.size() && std::chrono::steady_clock::now() < deadline) {
            scheduler.drainCompleted(results);
            if (results.size() < keys.size())
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        REQUIRE(results.size() == keys.size());
        std::unordered_map<FarTerrainKey, std::shared_ptr<const FarTerrainMesh>, FarTerrainKeyHash>
            meshes;
        for (const FarTerrainResult& result : results) {
            REQUIRE_FALSE(result.failed);
            REQUIRE(result.mesh);
            meshes.emplace(result.key, result.mesh);
        }
        return meshes;
    };

    const auto first = buildPass({LEFT, RIGHT});
    REQUIRE(scheduler.findCached(LEFT));
    REQUIRE(scheduler.findCached(RIGHT));
    scheduler.clearCache();
    REQUIRE_FALSE(scheduler.findCached(LEFT));
    REQUIRE_FALSE(scheduler.findCached(RIGHT));
    const auto second = buildPass({RIGHT, LEFT});
    scheduler.shutdown();
    for (const FarTerrainKey key : {LEFT, RIGHT}) {
        REQUIRE(first.at(key)->deterministicHash == second.at(key)->deterministicHash);
        REQUIRE(first.at(key)->surfaceBounds.minY == second.at(key)->surfaceBounds.minY);
        REQUIRE(first.at(key)->surfaceBounds.maxY == second.at(key)->surfaceBounds.maxY);
        REQUIRE(first.at(key)->bounds.minY == second.at(key)->bounds.minY);
        REQUIRE(first.at(key)->indices == second.at(key)->indices);
    }
    REQUIRE(boundsCalls->load(std::memory_order_relaxed) == 4);
}

TEST_CASE("Production far terrain exposes batched conservative cell bounds",
          "[render][far-terrain][coverage][bounds][production][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(static_cast<bool>(source.cellBoundsGrid));
    REQUIRE(static_cast<bool>(source.sampleGrid));
    std::array<FarSurfaceSample, 9> samples;
    source.sampleGrid(-32, -32, 32, 3, worldgen::SurfaceFootprint::BLOCK_32, samples);
    for (const FarSurfaceSample& sample : samples) {
        REQUIRE(std::isfinite(sample.footprintMinimumTerrainHeight));
        REQUIRE(std::isfinite(sample.footprintMaximumTerrainHeight));
        REQUIRE(sample.footprintMinimumTerrainHeight <= sample.geometry.terrainHeight);
        REQUIRE(sample.footprintMaximumTerrainHeight >= sample.geometry.terrainHeight);
    }
    std::array<FarTerrainCellBounds, 16> bounds;
    source.cellBoundsGrid(-64, -64, 32, 4, 4, worldgen::SurfaceFootprint::BLOCK_32, bounds);
    for (const FarTerrainCellBounds& cell : bounds) {
        REQUIRE(std::isfinite(cell.terrainHeight));
        REQUIRE(std::isfinite(cell.minimumTerrainHeight));
        REQUIRE(std::isfinite(cell.maximumTerrainHeight));
        REQUIRE(cell.minimumTerrainHeight <= cell.maximumTerrainHeight);
        REQUIRE(cell.minimumTerrainHeight <= cell.terrainHeight);
        REQUIRE(cell.terrainHeight <= cell.maximumTerrainHeight);
    }
}

TEST_CASE("Production cell bounds enclose interior emitted terrain and water floors",
          "[render][far-terrain][coverage][bounds][worldgen][hydrology][regression]") {
    struct Fixture {
        uint32_t seed;
        int64_t x;
        int64_t z;
        const char* name;
    };
    constexpr std::array FIXTURES = {
        Fixture{42, -12'289, 2'649, "negative river boundary"},
        Fixture{42, -8'240, 3'088, "waterfall receiver"},
        Fixture{764891, 23'029, -111'486, "caldera interior"},
    };
    constexpr int STEP = 16;
    for (const Fixture& fixture : FIXTURES) {
        auto generator = std::make_shared<ChunkGenerator>(fixture.seed);
        const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        const int64_t originX = world_coord::floorDiv(fixture.x, int64_t{STEP}) * STEP;
        const int64_t originZ = world_coord::floorDiv(fixture.z, int64_t{STEP}) * STEP;
        std::array<FarTerrainCellBounds, 1> bounds{};
        source.cellBoundsGrid(originX, originZ, STEP, 1, 1, worldgen::SurfaceFootprint::BLOCK_16,
                              bounds);
        double exactMinimum = std::numeric_limits<double>::max();
        double exactMaximum = std::numeric_limits<double>::lowest();
        bool sawGeneratedWater = false;
        bool sawWaterfall = false;
        for (int z = 0; z < STEP; ++z) {
            for (int x = 0; x < STEP; ++x) {
                const worldgen::SurfaceSample exact =
                    generator->sampleExactSurface(originX + x, originZ + z);
                exactMinimum = std::min(exactMinimum, exact.terrainHeight);
                exactMaximum = std::max(exactMaximum, exact.terrainHeight);
                sawGeneratedWater = sawGeneratedWater || exact.hydrology.ocean ||
                                    exact.hydrology.river || exact.hydrology.lake ||
                                    exact.hydrology.wetland;
                sawWaterfall = sawWaterfall || exact.hydrology.waterfall;
            }
        }
        CAPTURE(fixture.name, originX, originZ, exactMinimum, exactMaximum,
                bounds.front().minimumTerrainHeight, bounds.front().maximumTerrainHeight);
        REQUIRE(bounds.front().minimumTerrainHeight <= exactMinimum);
        REQUIRE(bounds.front().maximumTerrainHeight >= exactMaximum);
        if (std::string_view(fixture.name) == "waterfall receiver") {
            REQUIRE(sawGeneratedWater);
            REQUIRE(sawWaterfall);
        }
    }
}

TEST_CASE("Production bounds retain negative step thirty-two parents after cache eviction",
          "[render][far-terrain][coverage][bounds][negative][determinism][cache]") {
    constexpr int64_t ORIGIN_X = -66;
    constexpr int64_t ORIGIN_Z = -34;
    constexpr int CELL_EDGE = 4;
    const auto sampleFineBounds = [](const std::shared_ptr<ChunkGenerator>& generator) {
        const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        std::array<FarTerrainCellBounds, CELL_EDGE * CELL_EDGE> bounds{};
        source.cellBoundsGrid(ORIGIN_X, ORIGIN_Z, 2, CELL_EDGE, CELL_EDGE,
                              worldgen::SurfaceFootprint::BLOCK_2, bounds);
        return bounds;
    };
    auto firstGenerator = std::make_shared<ChunkGenerator>(42);
    const auto first = sampleFineBounds(firstGenerator);
    firstGenerator->clearMacroCaches();
    auto rebuiltGenerator = std::make_shared<ChunkGenerator>(42);
    const auto rebuilt = sampleFineBounds(rebuiltGenerator);
    for (size_t index = 0; index < first.size(); ++index) {
        CAPTURE(index);
        REQUIRE(first[index].terrainHeight == rebuilt[index].terrainHeight);
        REQUIRE(first[index].minimumTerrainHeight == rebuilt[index].minimumTerrainHeight);
        REQUIRE(first[index].maximumTerrainHeight == rebuilt[index].maximumTerrainHeight);
        REQUIRE(first[index].minimumTerrainHeight <= first[index].terrainHeight);
        REQUIRE(first[index].terrainHeight <= first[index].maximumTerrainHeight);
    }
}

TEST_CASE("Production cell top authority stitches overlapping negative query aprons",
          "[render][far-terrain][coverage][bounds][negative][seam][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int STEP = 2;
    constexpr int EDGE = 4;
    // A three-by-three surface request expands to the same -4,-4 origin and
    // four-by-four cell metadata as the first bounds request. It must remain
    // scratch storage rather than masquerading as a prepared bounds batch.
    std::array<FarSurfaceSample, 9> surfaceScratch{};
    source.sampleGrid(-2, -2, STEP, 3, worldgen::SurfaceFootprint::BLOCK_2, surfaceScratch);
    std::array<FarTerrainCellBounds, EDGE * EDGE> crossing{};
    std::array<FarTerrainCellBounds, EDGE * EDGE> nonnegative{};
    source.cellBoundsGrid(-4, -4, STEP, EDGE, EDGE, worldgen::SurfaceFootprint::BLOCK_2, crossing);
    source.cellBoundsGrid(0, -4, STEP, EDGE, EDGE, worldgen::SurfaceFootprint::BLOCK_2,
                          nonnegative);
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < 2; ++x) {
            const FarTerrainCellBounds& left = crossing[static_cast<size_t>(z * EDGE + x + 2)];
            const FarTerrainCellBounds& right = nonnegative[static_cast<size_t>(z * EDGE + x)];
            CAPTURE(x, z);
            REQUIRE(left.terrainHeight == right.terrainHeight);
            REQUIRE(left.minimumTerrainHeight == right.minimumTerrainHeight);
            REQUIRE(left.maximumTerrainHeight == right.maximumTerrainHeight);
        }
    }
}

TEST_CASE("Partially faded coverage and LOD parents never establish an occluder",
          "[render][far-terrain][coverage][occlusion][lod][transition][regression]") {
    FarTerrainCoverageFrontier frontier;
    frontier.complete = false;
    frontier.distanceBlocks = 1024.0F;
    frontier.distanceSquaredBlocks = 1024.0 * 1024.0;
    frontier.missingBaseTiles = 1;
    const TerrainHorizonViewpoint viewpoint{};
    constexpr double FADE_BLOCKS = 256.0;
    const FarTerrainBounds opaquePatch{600, 700, -16, 16, 40.0F, 80.0F};
    const FarTerrainBounds fadingPatch{700, 800, -16, 16, 40.0F, 80.0F};
    REQUIRE(
        farTerrainCoveragePatchMayOcclude(opaquePatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(opaquePatch, viewpoint, frontier, FADE_BLOCKS, true));
    frontier.complete = true;
    REQUIRE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, false));
    REQUIRE_FALSE(
        farTerrainCoveragePatchMayOcclude(fadingPatch, viewpoint, frontier, FADE_BLOCKS, true));
}

TEST_CASE("Cross-tile canopy crowns expand horizontal surface bounds",
          "[render][far-terrain][canopy][bounds][frustum][seam][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    constexpr FarTerrainKey KEY{1, -2, FarTerrainStep::FOUR};
    constexpr int64_t ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    source.canopies = [](int64_t minimumX, int64_t minimumZ, int64_t, int64_t, FarTerrainStep) {
        FarCanopy canopy;
        canopy.x = minimumX + 1;
        canopy.z = minimumZ + FAR_TERRAIN_TILE_EDGE / 2;
        canopy.baseY = 64;
        canopy.topY = 75;
        canopy.canopyMinimumY = 67;
        canopy.canopyMaximumY = 75;
        canopy.canopyRadius = 8;
        canopy.logBlock = BlockType::LOG;
        canopy.leafBlock = BlockType::LEAVES;
        canopy.anchorId = 17;
        return std::vector<FarCanopy>{canopy};
    };

    const auto mesh = FarTerrainMesher::build(KEY, source);
    const auto attachment = FarTerrainMesher::buildCanopyAttachment(KEY, source);
    REQUIRE(attachment->canopyAnchorCount == 1);
    REQUIRE_FALSE(attachment->vertices.empty());
    REQUIRE(std::ranges::none_of(mesh->vertices, [](const Vertex& vertex) {
        return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U;
    }));
    REQUIRE(std::ranges::all_of(attachment->vertices, [](const Vertex& vertex) {
        return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U;
    }));
    REQUIRE(attachment->bounds.minX < ORIGIN_X);
    REQUIRE(mesh->surfaceBounds.minX == ORIGIN_X);
    REQUIRE(mesh->surfaceBounds.maxX == ORIGIN_X + FAR_TERRAIN_TILE_EDGE);
    REQUIRE(mesh->surfaceBounds.minZ == ORIGIN_Z);
    REQUIRE(mesh->surfaceBounds.maxZ == ORIGIN_Z + FAR_TERRAIN_TILE_EDGE);
    // Tile-local vertices still draw from the canonical tile origin. Only the
    // conservative frustum bounds expand around the crossing crown.
    REQUIRE(mesh->bounds.minX == ORIGIN_X);
    REQUIRE(mesh->bounds.maxX == ORIGIN_X + FAR_TERRAIN_TILE_EDGE);
}

TEST_CASE("Final ecology anchors ground against preview terrain and remain stable on promotion",
          "[render][far-terrain][canopy][authority][preview][promotion][regression]") {
    const auto flatSource = [](double height) {
        return testFarTerrainSource(
            [height](int64_t, int64_t) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = height;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    };
    FarTerrainSource ecology = flatSource(96.0);
    ecology.canopies = [](int64_t minimumX, int64_t minimumZ, int64_t, int64_t, FarTerrainStep) {
        FarCanopy canopy;
        canopy.x = minimumX + 32;
        canopy.z = minimumZ + 48;
        canopy.baseY = 96;
        canopy.topY = 106;
        canopy.canopyMinimumY = 101;
        canopy.canopyMaximumY = 106;
        canopy.canopyRadius = 3;
        canopy.logBlock = BlockType::LOG;
        canopy.leafBlock = BlockType::LEAVES;
        canopy.anchorId = 0xCA90'0000'0000'0042ULL;
        canopy.species = feature_generation::TreeSpecies::OAK;
        return std::vector<FarCanopy>{canopy};
    };
    const FarTerrainSource previewGround = flatSource(72.0);
    constexpr FarTerrainKey KEY{2, -3, FarTerrainStep::FOUR};

    const auto preview = FarTerrainMesher::buildCanopyAttachment(
        KEY, ecology, previewGround, FarTerrainAuthorityQuality::PREVIEW);
    const auto provisionalPreview = FarTerrainMesher::buildCanopyAttachment(
        KEY, ecology, previewGround, FarTerrainAuthorityQuality::PREVIEW,
        FarTerrainAuthorityQuality::PREVIEW);
    const auto provisionalFinalGround = FarTerrainMesher::buildCanopyAttachment(
        KEY, ecology, ecology, FarTerrainAuthorityQuality::FINAL,
        FarTerrainAuthorityQuality::PREVIEW);
    const auto final = FarTerrainMesher::buildCanopyAttachment(KEY, ecology, ecology,
                                                               FarTerrainAuthorityQuality::FINAL);
    FarTerrainSource changedEcology = ecology;
    const auto stableCanopies = ecology.canopies;
    changedEcology.canopies = [stableCanopies](int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                               int64_t maximumZ, FarTerrainStep step) {
        std::vector<FarCanopy> changed =
            stableCanopies(minimumX, minimumZ, maximumX, maximumZ, step);
        changed.front().canopyOffsetX += 1;
        return changed;
    };
    const auto changedShape = FarTerrainMesher::buildCanopyAttachment(
        KEY, changedEcology, ecology, FarTerrainAuthorityQuality::FINAL);

    REQUIRE(preview->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(provisionalPreview->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(provisionalFinalGround->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(final->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(preview->groundingQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(final->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(preview->canopyAnchorCount == 1);
    REQUIRE(final->canopyAnchorCount == preview->canopyAnchorCount);
    REQUIRE(final->anchorIdentityHash == preview->anchorIdentityHash);
    REQUIRE(changedShape->anchorIdentityHash != final->anchorIdentityHash);
    REQUIRE_FALSE(farCanopyAnchorIdentityCompatible(
        final->authorityQuality, final->anchorIdentityHash, changedShape->authorityQuality,
        changedShape->anchorIdentityHash));
    REQUIRE(final->deterministicHash != preview->deterministicHash);
    REQUIRE(preview->bounds.minY == Catch::Approx(72.0F));
    REQUIRE(final->bounds.minY == Catch::Approx(96.0F));
    REQUIRE(farCanopyMatchesSurface(preview->authorityQuality, preview->groundingQuality,
                                    FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(farCanopyMatchesSurface(provisionalPreview->authorityQuality,
                                    provisionalPreview->groundingQuality,
                                    FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(farCanopyMatchesSurface(provisionalFinalGround->authorityQuality,
                                    provisionalFinalGround->groundingQuality,
                                    FarTerrainAuthorityQuality::FINAL));
    REQUIRE_FALSE(farCanopyMatchesSurface(preview->authorityQuality, preview->groundingQuality,
                                          FarTerrainAuthorityQuality::FINAL));
    REQUIRE(farCanopyAnchorIdentityCompatible(preview->authorityQuality,
                                              preview->anchorIdentityHash, final->authorityQuality,
                                              final->anchorIdentityHash));
    REQUIRE_FALSE(
        farCanopyAnchorIdentityCompatible(preview->authorityQuality, preview->anchorIdentityHash,
                                          final->authorityQuality, final->anchorIdentityHash ^ 1U));
    REQUIRE(farCanopyMayReplace(
        FarTerrainAuthorityQuality::FINAL, FarTerrainAuthorityQuality::PREVIEW,
        FarTerrainAuthorityQuality::PREVIEW, FarTerrainAuthorityQuality::FINAL));
    REQUIRE_FALSE(farCanopyMayReplace(
        FarTerrainAuthorityQuality::PREVIEW, FarTerrainAuthorityQuality::FINAL,
        FarTerrainAuthorityQuality::FINAL, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(
        farCanopyMayReplace(FarTerrainAuthorityQuality::PREVIEW, FarTerrainAuthorityQuality::FINAL,
                            FarTerrainAuthorityQuality::FINAL, FarTerrainAuthorityQuality::FINAL));

    std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash> residents;
    FarTerrainMeshState surface{};
    surface.uploaded = true;
    surface.authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    residents.emplace(KEY, surface);
    std::unordered_map<FarTerrainKey, FarCanopyMeshState, FarTerrainKeyHash> attachments;
    FarCanopyMeshState canopy{};
    canopy.authorityQuality = preview->authorityQuality;
    canopy.groundingQuality = preview->groundingQuality;
    canopy.deterministicHash = preview->deterministicHash;
    canopy.anchorIdentityHash = preview->anchorIdentityHash;
    attachments.emplace(KEY, canopy);

    const std::unordered_map<ColumnPos, FarTerrainKey> displayed = {
        {ColumnPos{KEY.tileX, KEY.tileZ}, KEY},
    };
    const std::unordered_map<ColumnPos, FarTerrainLodTransition> transitions;
    std::vector<FarTerrainCanopyRefreshRequest> requests;
    attachments.clear();
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().groundingQuality == FarTerrainAuthorityQuality::PREVIEW);

    attachments.emplace(KEY, canopy);
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.empty());

    // PREVIEW ecology is already drawable, but remains in the refresh batch
    // so the parked FINAL replacement cannot lose liveness.
    attachments.at(KEY).authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().groundingQuality == FarTerrainAuthorityQuality::PREVIEW);

    // Promotion keeps the preview-grounded attachment resident and requests
    // a grounded replacement. A provisional attachment built against FINAL
    // terrain can publish with that surface immediately while FINAL ecology
    // continues in the background, so neither side of the exchange is bare.
    residents.at(KEY).authorityQuality = FarTerrainAuthorityQuality::FINAL;
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().key == KEY);
    REQUIRE(requests.front().viewPriority ==
            static_cast<uint32_t>(
                std::hypot(2.0 * FAR_TERRAIN_TILE_EDGE, 2.0 * FAR_TERRAIN_TILE_EDGE)));
    REQUIRE(attachments.contains(KEY));
    REQUIRE(attachments.at(KEY).deterministicHash == preview->deterministicHash);

    attachments.at(KEY).groundingQuality = FarTerrainAuthorityQuality::FINAL;
    attachments.at(KEY).deterministicHash = provisionalFinalGround->deterministicHash;
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.size() == 1);
    REQUIRE(farCanopyMatchesSurface(attachments.at(KEY).authorityQuality,
                                    attachments.at(KEY).groundingQuality,
                                    residents.at(KEY).authorityQuality));

    attachments.at(KEY).authorityQuality = FarTerrainAuthorityQuality::FINAL;
    attachments.at(KEY).deterministicHash = final->deterministicHash;
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 0.0, 0.0, 1,
                                      requests);
    REQUIRE(requests.empty());
}

TEST_CASE("A running preview-grounded canopy automatically follows final promotion",
          "[render][far-terrain][scheduler][canopy][authority][promotion][regression]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool firstEntered = false;
    bool releaseFirst = false;
    std::atomic<uint32_t> builds{0};
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        if (builds.fetch_add(1, std::memory_order_relaxed) == 0) {
            std::unique_lock lock(gateMutex);
            firstEntered = true;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return releaseFirst; });
        }
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 1;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), limits);
    constexpr FarTerrainKey KEY{6, -4, FarTerrainStep::EIGHT};
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::PREVIEW));

    bool entered = false;
    {
        std::unique_lock lock(gateMutex);
        entered = gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return firstEntered; });
    }
    const bool promotionCoalesced =
        entered && !scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL);
    {
        std::lock_guard lock(gateMutex);
        releaseFirst = true;
    }
    gateCv.notify_all();
    REQUIRE(entered);
    REQUIRE(promotionCoalesced);

    std::vector<FarCanopyResult> canopies;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (canopies.size() < 2 && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCanopyCompleted(canopies);
        if (canopies.size() < 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(canopies.size() == 2);
    REQUIRE(canopies[0].attachment);
    REQUIRE(canopies[1].attachment);
    REQUIRE(canopies[0].attachment->groundingQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(canopies[1].attachment->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(canopies[0].attachment->anchorIdentityHash ==
            canopies[1].attachment->anchorIdentityHash);
    REQUIRE(builds.load(std::memory_order_relaxed) == 2);
    REQUIRE(scheduler.stats().canopySubmitted == 2);
    REQUIRE(scheduler.stats().canopyBuilt == 2);
    REQUIRE(scheduler.stats().canopyInFlight == 0);
    REQUIRE(scheduler.findCachedCanopy(KEY)->groundingQuality == FarTerrainAuthorityQuality::FINAL);
}

TEST_CASE("A stale canopy completion preserves the next epoch's final followup",
          "[render][far-terrain][scheduler][canopy][authority][epoch][promotion][regression]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool oldEntered = false;
    bool newEntered = false;
    bool releaseOld = false;
    bool releaseNew = false;
    std::atomic<uint32_t> builds{0};
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        const uint32_t build = builds.fetch_add(1, std::memory_order_relaxed);
        if (build < 2) {
            std::unique_lock lock(gateMutex);
            bool& entered = build == 0 ? oldEntered : newEntered;
            bool& released = build == 0 ? releaseOld : releaseNew;
            entered = true;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return released; });
        }
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 3;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), limits);
    constexpr FarTerrainKey KEY{-3, 7, FarTerrainStep::FOUR};
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::PREVIEW));

    bool sawOld = false;
    {
        std::unique_lock lock(gateMutex);
        sawOld = gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return oldEntered; });
    }
    if (sawOld) {
        scheduler.advanceEpoch();
    }
    const bool enqueuedNew =
        sawOld && scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::PREVIEW);
    bool sawNew = false;
    if (enqueuedNew) {
        std::unique_lock lock(gateMutex);
        sawNew = gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return newEntered; });
    }
    const bool finalFollowup =
        sawNew && !scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL);
    {
        std::lock_guard lock(gateMutex);
        releaseOld = true;
    }
    gateCv.notify_all();
    const auto staleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().canopyCanceled == 0 &&
           std::chrono::steady_clock::now() < staleDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool staleFinished = scheduler.stats().canopyCanceled == 1;
    {
        std::lock_guard lock(gateMutex);
        releaseNew = true;
    }
    gateCv.notify_all();
    REQUIRE(sawOld);
    REQUIRE(enqueuedNew);
    REQUIRE(sawNew);
    REQUIRE(finalFollowup);
    REQUIRE(staleFinished);

    std::vector<FarCanopyResult> canopies;
    const auto completionDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (canopies.size() < 2 && std::chrono::steady_clock::now() < completionDeadline) {
        scheduler.drainCanopyCompleted(canopies);
        if (canopies.size() < 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(canopies.size() == 2);
    REQUIRE(canopies[0].attachment->groundingQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(canopies[1].attachment->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(scheduler.stats().canopySubmitted == 3);
    REQUIRE(scheduler.stats().canopyBuilt == 2);
    REQUIRE(scheduler.stats().canopyCanceled == 1);
    REQUIRE(scheduler.stats().canopyInFlight == 0);
    REQUIRE(scheduler.stats().activeCanopyWorkers == 0);
}

TEST_CASE("Final-grounded canopy promotion retries after optional queue saturation",
          "[render][far-terrain][scheduler][canopy][authority][promotion][capacity][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 1;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey BLOCKER{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey PROMOTION{1, 0, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueueCanopy(BLOCKER, 0, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE_FALSE(scheduler.enqueueCanopy(PROMOTION, 0, FarTerrainAuthorityQuality::FINAL));

    scheduler.setCanopyWorkerBudget(1);
    std::vector<FarCanopyResult> canopies;
    const auto blockerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().canopyInFlight != 0 &&
           std::chrono::steady_clock::now() < blockerDeadline) {
        scheduler.drainCanopyCompleted(canopies);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().canopyInFlight == 0);
    REQUIRE(scheduler.enqueueCanopy(PROMOTION, 0, FarTerrainAuthorityQuality::FINAL));

    const auto promotionDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!scheduler.findCachedCanopy(PROMOTION) &&
           std::chrono::steady_clock::now() < promotionDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    const std::shared_ptr<const FarCanopyAttachment> promoted =
        scheduler.findCachedCanopy(PROMOTION);
    REQUIRE(promoted);
    REQUIRE(promoted->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(scheduler.stats().canopySubmitted == 2);
    REQUIRE(scheduler.stats().canopyBuilt == 2);
}

TEST_CASE("Canopy roots use the displayed owning-cell top at every far LOD",
          "[render][far-terrain][canopy][grounding][lod][slope][regression]") {
    const auto canopies = [](int64_t minimumX, int64_t minimumZ, int64_t maximumX, int64_t maximumZ,
                             FarTerrainStep farStep) {
        (void)farStep;
        FarCanopy canopy;
        canopy.x = minimumX;
        canopy.z = minimumZ;
        canopy.baseY = 100;
        canopy.topY = 110;
        canopy.canopyMinimumY = 105;
        canopy.canopyMaximumY = 110;
        canopy.canopyRadius = 2;
        canopy.logBlock = BlockType::LOG;
        canopy.leafBlock = BlockType::LEAVES;
        canopy.anchorId = 0xCE11'7000'0000'0001ULL;
        FarCanopy west = canopy;
        west.z = minimumZ + FAR_TERRAIN_TILE_EDGE / 2;
        west.anchorId += 1;
        FarCanopy north = canopy;
        north.x = minimumX + FAR_TERRAIN_TILE_EDGE / 2;
        north.anchorId += 2;
        FarCanopy opposite = canopy;
        opposite.x = maximumX - 1;
        opposite.z = maximumZ - 1;
        opposite.anchorId += 3;
        return std::vector{canopy, west, north, opposite};
    };
    std::atomic<size_t> authoritativeCellsSampled{0};
    FarTerrainSource authoritative = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight =
                300.0 + static_cast<double>(x) * 0.5 + static_cast<double>(z) * 0.25;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    authoritative.canopies = canopies;
    authoritative.cellBoundsGrid = [&](int64_t originX, int64_t originZ, int step, int width,
                                       int height, worldgen::SurfaceFootprint,
                                       std::span<FarTerrainCellBounds> output) {
        authoritativeCellsSampled.fetch_add(output.size(), std::memory_order_relaxed);
        for (int cellZ = 0; cellZ < height; ++cellZ) {
            for (int cellX = 0; cellX < width; ++cellX) {
                const int64_t worldX = originX + static_cast<int64_t>(cellX) * step;
                const int64_t worldZ = originZ + static_cast<int64_t>(cellZ) * step;
                const double top = 100.0 + static_cast<double>(worldX) * 0.25 +
                                   static_cast<double>(worldZ) * 0.125;
                output[static_cast<size_t>(cellZ * width + cellX)] = {
                    .terrainHeight = top,
                    .minimumTerrainHeight = top,
                    .maximumTerrainHeight = top,
                };
            }
        }
    };

    FarTerrainSource fallback = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight =
                80.0 + static_cast<double>(x) * 0.25 + static_cast<double>(z) * 0.125;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    fallback.canopies = canopies;

    const auto rootMinimumY = [](const FarCanopyAttachment& attachment, const FarCanopy& canopy) {
        const float minimumX = static_cast<float>(canopy.x - attachment.originX);
        const float minimumZ = static_cast<float>(canopy.z - attachment.originZ);
        float minimumY = std::numeric_limits<float>::max();
        for (const Vertex& vertex : attachment.vertices) {
            const float x = static_cast<float>(vertex.px);
            const float z = static_cast<float>(vertex.pz);
            if (x >= minimumX && x <= minimumX + 1.0F && z >= minimumZ && z <= minimumZ + 1.0F) {
                minimumY = std::min(minimumY, static_cast<float>(vertex.py));
            }
        }
        return minimumY;
    };
    for (const ColumnPos tile : {ColumnPos{0, 0}, ColumnPos{-1, -2}}) {
        for (const FarTerrainStep step :
             {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            CAPTURE(tile.x, tile.z, farTerrainStepSize(step));
            const FarTerrainKey key{tile.x, tile.z, step};
            const int64_t originX = tile.x * FAR_TERRAIN_TILE_EDGE;
            const int64_t originZ = tile.z * FAR_TERRAIN_TILE_EDGE;
            const std::vector<FarCanopy> anchors =
                canopies(originX, originZ, originX + FAR_TERRAIN_TILE_EDGE,
                         originZ + FAR_TERRAIN_TILE_EDGE, step);

            authoritativeCellsSampled.store(0, std::memory_order_relaxed);
            const auto authoritativeCanopy =
                FarTerrainMesher::buildCanopyAttachment(key, authoritative);
            REQUIRE(authoritativeCellsSampled.load(std::memory_order_relaxed) ==
                    (step == FarTerrainStep::ONE ? 0 : anchors.size()));
            const auto authoritativeTerrain = FarTerrainMesher::build(key, authoritative);
            for (const FarCanopy& anchor : anchors) {
                const float localX = static_cast<float>(anchor.x - originX) + 0.5F;
                const float localZ = static_cast<float>(anchor.z - originZ) + 0.5F;
                const std::optional<float> displayed =
                    farTerrainHeightAt(*authoritativeTerrain, localX, localZ);
                REQUIRE(displayed);
                const float representableGround =
                    static_cast<float>(static_cast<float16_t>(*displayed));
                REQUIRE(rootMinimumY(*authoritativeCanopy, anchor) ==
                        Catch::Approx(representableGround));
            }

            const auto fallbackCanopy = FarTerrainMesher::buildCanopyAttachment(key, fallback);
            const auto fallbackTerrain = FarTerrainMesher::build(key, fallback);
            for (const FarCanopy& anchor : anchors) {
                const float localX = static_cast<float>(anchor.x - originX) + 0.5F;
                const float localZ = static_cast<float>(anchor.z - originZ) + 0.5F;
                const std::optional<float> displayed =
                    farTerrainHeightAt(*fallbackTerrain, localX, localZ);
                REQUIRE(displayed);
                const float representableGround =
                    static_cast<float>(static_cast<float16_t>(*displayed));
                REQUIRE(rootMinimumY(*fallbackCanopy, anchor) ==
                        Catch::Approx(representableGround));
            }
        }
    }
}

TEST_CASE("Resident far canopies join caster bounds and shadow revisions on arrival",
          "[render][far-terrain][canopy][shadow][regression]") {
    const FarTerrainBounds surface{0, 256, 0, 256, 50.0F, 80.0F};
    const FarTerrainBounds canopy{-3, 259, -4, 260, 60.0F, 100.0F};
    REQUIRE_FALSE(farCanopyCastsShadow(false, true, 24));
    REQUIRE_FALSE(farCanopyCastsShadow(true, false, 24));
    REQUIRE_FALSE(farCanopyCastsShadow(true, true, 0));
    REQUIRE(farCanopyCastsShadow(true, true, 24));

    const FarTerrainBounds combined = farShadowCasterBounds(surface, canopy);
    REQUIRE(combined.minX == canopy.minX);
    REQUIRE(combined.maxX == canopy.maxX);
    REQUIRE(combined.minZ == canopy.minZ);
    REQUIRE(combined.maxZ == canopy.maxZ);
    REQUIRE(combined.minY == surface.minY);
    REQUIRE(combined.maxY == canopy.maxY);

    constexpr uint64_t BASE = 0x1234;
    const uint64_t absent =
        farCanopyShadowRevision(BASE, false, FarTerrainAuthorityQuality::FINAL, 0);
    const uint64_t arrived =
        farCanopyShadowRevision(BASE, true, FarTerrainAuthorityQuality::FINAL, 0xCAFE);
    REQUIRE(absent != arrived);
}

TEST_CASE("Far terrain boundaries stop at neighboring terrain without downward skirts",
          "[render][far-terrain][coverage][lod][bounds][skirt][seam][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < 0 ? 20.0 : 100.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t, int step, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t worldX = originX + static_cast<int64_t>(x * step);
                const double minimum = worldX < 0 ? 20.0 : 100.0;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = minimum,
                    .minimumTerrainHeight = minimum,
                    .maximumTerrainHeight = minimum,
                };
            }
        }
    };

    const auto parent = FarTerrainMesher::build({-1, 0, FarTerrainStep::THIRTY_TWO}, source);
    const auto fine = FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    REQUIRE(parent->surfaceBounds.minY == 20.0F);
    REQUIRE(fine->surfaceBounds.maxY == 100.0F);
    REQUIRE(fine->bounds.minY == 100.0F);
    REQUIRE(farTerrainEdge(*parent, true) == farTerrainEdge(*fine, false));
    REQUIRE(farTerrainEdge(*fine, false).size() ==
            static_cast<size_t>(FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1));
    constexpr uint32_t RESERVED_PANEL_ATTRIBUTE = 1U << 29U;
    REQUIRE(std::ranges::none_of(fine->vertices, [](const Vertex& vertex) {
        return (vertex.faceAttr & RESERVED_PANEL_ATTRIBUTE) != 0U;
    }));
}

TEST_CASE("Far terrain LOD uses absolute bands with outward-only hysteresis",
          "[render][far-terrain][selection]") {
    REQUIRE(farTerrainStepForMetrics(100.0) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(50.0) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(300.0) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(450.0) == FarTerrainStep::SIXTEEN);

    REQUIRE(farTerrainStepForMetrics(35.0, FarTerrainStep::ONE) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(68.0, FarTerrainStep::TWO) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(77.0, FarTerrainStep::TWO) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(60.0, FarTerrainStep::FOUR) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(120.0, FarTerrainStep::EIGHT) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(220.0, FarTerrainStep::SIXTEEN) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(400.0, FarTerrainStep::THIRTY_TWO) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainStepForMetrics(140.0, FarTerrainStep::FOUR) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForMetrics(145.0, FarTerrainStep::FOUR) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(270.0, FarTerrainStep::EIGHT) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForMetrics(281.0, FarTerrainStep::EIGHT) == FarTerrainStep::SIXTEEN);
}

TEST_CASE("Far terrain screen error retains perceptible relief without breaking distance caps",
          "[render][far-terrain][selection][screen-error][lod][regression]") {
    constexpr double FOV_70 = 70.0 * std::numbers::pi / 180.0;
    FarTerrainScreenErrorMetrics metrics{
        .distanceBlocks = 300.0 * CHUNK_EDGE,
        .viewportHeightPixels = 1536.0,
        .verticalFovRadians = FOV_70,
        .tileReliefBlocks = 0.0,
    };
    REQUIRE(farTerrainProjectedBlockPixels(metrics) == Catch::Approx(0.2285).margin(0.001));
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics) == FarTerrainStep::FOUR);

    metrics.tileReliefBlocks = 200.0;
    REQUIRE(farTerrainProjectedGeometricErrorPixels(FarTerrainStep::EIGHT, metrics) >
            FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS);
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics) == FarTerrainStep::TWO);
    const double finalDisplayError = farTerrainProjectedDisplayErrorPixels(
        FarTerrainStep::TWO, FarTerrainAuthorityQuality::FINAL, metrics);
    const double previewDisplayError = farTerrainProjectedDisplayErrorPixels(
        FarTerrainStep::TWO, FarTerrainAuthorityQuality::PREVIEW, metrics);
    REQUIRE(finalDisplayError ==
            Catch::Approx(farTerrainProjectedGeometricErrorPixels(FarTerrainStep::TWO, metrics)));
    REQUIRE(previewDisplayError ==
            Catch::Approx(finalDisplayError + farTerrainProjectedBlockPixels(metrics) *
                                                  FAR_TERRAIN_PREVIEW_RESIDUAL_MAX_BLOCKS));
    REQUIRE(previewDisplayError > finalDisplayError);

    metrics.distanceBlocks = 180.0 * CHUNK_EDGE;
    REQUIRE(farTerrainStepForScreenMetrics(180.0, metrics) == FarTerrainStep::TWO);

    // Screen-space selection reaches the finest far tier when step 2 would
    // leave a visible grid at the exact handoff. The physical one-block grid
    // is the irreducible floor even when its projected footprint exceeds the
    // target at very close range.
    metrics.distanceBlocks = 32.0 * CHUNK_EDGE;
    metrics.viewportHeightPixels = 2048.0;
    metrics.tileReliefBlocks = 200.0;
    REQUIRE(farTerrainProjectedGeometricErrorPixels(FarTerrainStep::TWO, metrics) >
            FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS);
    REQUIRE(farTerrainStepForScreenMetrics(32.0, metrics) == FarTerrainStep::ONE);

    // Absolute bands remain hard maximum-coarseness limits even when a low
    // resolution viewport would otherwise hide the missing detail.
    metrics.distanceBlocks = 50.0 * CHUNK_EDGE;
    metrics.viewportHeightPixels = 320.0;
    REQUIRE(farTerrainStepForScreenMetrics(50.0, metrics) == FarTerrainStep::TWO);

    // A narrower FOV magnifies the same terrain and must retain more detail.
    metrics.distanceBlocks = 300.0 * CHUNK_EDGE;
    metrics.viewportHeightPixels = 1536.0;
    metrics.verticalFovRadians = 45.0 * std::numbers::pi / 180.0;
    metrics.tileReliefBlocks = 0.0;
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics) == FarTerrainStep::TWO);

    // Invalid projection data falls back to the deterministic absolute tier.
    metrics.viewportHeightPixels = 0.0;
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics) == FarTerrainStep::SIXTEEN);
}

TEST_CASE("Far terrain screen error uses outward hysteresis without retaining coarse terrain",
          "[render][far-terrain][selection][screen-error][hysteresis][regression]") {
    constexpr double FOV_70 = 70.0 * std::numbers::pi / 180.0;
    FarTerrainScreenErrorMetrics metrics{
        .distanceBlocks = 400.0 * CHUNK_EDGE,
        .viewportHeightPixels = 1536.0,
        .verticalFovRadians = FOV_70,
        .tileReliefBlocks = 200.0,
    };
    REQUIRE(farTerrainStepForScreenMetrics(400.0, metrics) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForScreenMetrics(400.0, metrics, FarTerrainStep::FOUR) ==
            FarTerrainStep::FOUR);

    metrics.distanceBlocks = 480.0 * CHUNK_EDGE;
    metrics.tileReliefBlocks = 0.0;
    REQUIRE(farTerrainStepForScreenMetrics(480.0, metrics, FarTerrainStep::FOUR) ==
            FarTerrainStep::FOUR);

    metrics.distanceBlocks = 300.0 * CHUNK_EDGE;
    metrics.tileReliefBlocks = 200.0;
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics, FarTerrainStep::EIGHT) ==
            FarTerrainStep::TWO);

    // A large outward move validates only one adjacent coarsening tier per
    // evaluation. It cannot jump directly from step 2 to step 16 merely
    // because every farther tier happens to satisfy the target threshold.
    metrics.projectionScalePixels = 0.10 * metrics.distanceBlocks;
    metrics.tileReliefBlocks = 0.0;
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics, FarTerrainStep::TWO) ==
            FarTerrainStep::FOUR);
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics, FarTerrainStep::FOUR) ==
            FarTerrainStep::EIGHT);
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics, FarTerrainStep::EIGHT) ==
            FarTerrainStep::EIGHT);

    metrics.projectionScalePixels = 0.12 * metrics.distanceBlocks;
    REQUIRE(farTerrainStepForScreenMetrics(300.0, metrics, FarTerrainStep::EIGHT) ==
            FarTerrainStep::EIGHT);
}

TEST_CASE("Settled screen-error selection uses every middle tier without emergency LODs",
          "[render][far-terrain][selection][screen-error][settling][negative][regression]") {
    const auto tierCounts = [](double cameraX, double cameraZ, double tileReliefBlocks) {
        std::vector<FarTerrainViewTile> selected;
        selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
        std::array<size_t, 33> counts{};
        for (FarTerrainViewTile& tile : selected) {
            const FarTerrainScreenErrorMetrics metrics{
                .distanceBlocks = std::max(1.0, tile.distanceChunks * CHUNK_EDGE),
                .viewportHeightPixels = 1536.0,
                .verticalFovRadians = 70.0 * std::numbers::pi / 180.0,
                .tileReliefBlocks = tileReliefBlocks,
            };
            const auto settled = farTerrainStepForScreenMetrics(tile.distanceChunks, metrics);
            REQUIRE(settled.has_value());
            REQUIRE(farTerrainStepSize(*settled) <= farTerrainStepSize(tile.key.step));
            REQUIRE(farTerrainStepForScreenMetrics(tile.distanceChunks, metrics, *settled) ==
                    settled);
            ++counts[static_cast<size_t>(farTerrainStepSize(*settled))];
            tile.key.step = *settled;
        }
        std::vector<FarTerrainKey> residency;
        buildFarTerrainResidencyOrder(selected, residency);
        return std::pair{counts, residency.size()};
    };

    // A flat horizon exercises every settled middle tier. High-relief tiles
    // intentionally retain step 4 through the 512-chunk edge at this
    // projection, as covered by the focused relief test above.
    const auto [negative, wantedCount] = tierCounts(-257.25, 513.75, 0.0);
    REQUIRE(negative[1] > 0);
    REQUIRE(negative[2] > 0);
    REQUIRE(negative[4] > 0);
    REQUIRE(negative[8] > 0);
    REQUIRE(negative[16] == 0);
    REQUIRE(negative[32] == 0);
    REQUIRE(wantedCount <= FarTerrainSchedulerLimits{}.maxCacheEntries);

    // Translating by whole tile widths preserves the exact deterministic LOD
    // distribution on both sides of the global coordinate origin.
    const auto translated = tierCounts(-257.25 + FAR_TERRAIN_TILE_EDGE * 8.0,
                                       513.75 - FAR_TERRAIN_TILE_EDGE * 12.0, 0.0);
    REQUIRE(translated == std::pair{negative, wantedCount});
}

TEST_CASE("A warm proxy chain never leaves a perceptible coverage parent nearby",
          "[render][far-terrain][selection][screen-error][preview][warm][regression]") {
    constexpr double FOV_70 = 70.0 * std::numbers::pi / 180.0;
    for (double distanceChunks = FAR_TERRAIN_NEAR_CHUNK_RADIUS;
         distanceChunks < FAR_TERRAIN_STEP_FOUR_LIMIT_CHUNKS; distanceChunks += 4.0) {
        FarTerrainScreenErrorMetrics metrics{
            .distanceBlocks = distanceChunks * CHUNK_EDGE,
            .viewportHeightPixels = 1536.0,
            .verticalFovRadians = FOV_70,
            .tileReliefBlocks = 160.0,
        };
        const std::optional<FarTerrainStep> desired =
            farTerrainStepForScreenMetrics(distanceChunks, metrics);
        REQUIRE(desired.has_value());
        FarTerrainStepMask resident = farTerrainStepMask(FAR_TERRAIN_BASE_STEP);
        const FarTerrainRefinementOrder chain = farTerrainRefinementOrder(*desired);
        for (const FarTerrainStep step : std::span(chain.steps).first(chain.count))
            resident |= farTerrainStepMask(step);
        const std::optional<FarTerrainStep> displayed = farTerrainInitialDisplayedStep(resident);
        REQUIRE(displayed.has_value());
        REQUIRE(*displayed != FAR_TERRAIN_BASE_STEP);
        REQUIRE(farTerrainStepSize(*displayed) <= farTerrainStepSize(*desired));
        if (*desired != FarTerrainStep::ONE) {
            REQUIRE(farTerrainProjectedGeometricErrorPixels(*desired, metrics) <=
                    FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS);
        }
    }
}

TEST_CASE("Far terrain topology swaps atomically beneath a narrow fog pulse",
          "[render][far-terrain][transition]") {
    const auto start = sampleFarTerrainTransition(0.0F);
    const auto quarter = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.25F);
    const auto midpoint = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.5F);
    const auto threeQuarter =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.75F);
    const auto complete = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS);

    REQUIRE_FALSE(start.drawTarget);
    REQUIRE(start.fogBlend == 0.0F);
    REQUIRE(start.progress == 0.0F);
    REQUIRE_FALSE(quarter.drawTarget);
    REQUIRE(quarter.fogBlend == 0.0F);
    REQUIRE(quarter.progress == Catch::Approx(0.15625F));
    REQUIRE(midpoint.drawTarget);
    REQUIRE(midpoint.fogBlend == 0.0F);
    REQUIRE(midpoint.progress == Catch::Approx(0.5F));
    REQUIRE(threeQuarter.drawTarget);
    REQUIRE(threeQuarter.fogBlend == 0.0F);
    REQUIRE(threeQuarter.progress == Catch::Approx(0.84375F));
    REQUIRE(complete.drawTarget);
    REQUIRE(complete.complete);
    REQUIRE(complete.fogBlend == 0.0F);
    REQUIRE(complete.progress == 1.0F);

    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    REQUIRE(farTerrainLodTerrainVisible(0.0F, SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(0.0F, TARGET));
    REQUIRE(farTerrainLodTerrainVisible(std::nextafter(0.5F, 0.0F), SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(std::nextafter(0.5F, 0.0F), TARGET));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(0.5F, SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(0.5F, TARGET));
    REQUIRE(farTerrainLodTerrainFog(0.42F, SOURCE) == Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodTerrainFog(0.5F, SOURCE) == Catch::Approx(1.0F));
    REQUIRE(farTerrainLodTerrainFog(0.58F, TARGET) == Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodConnectedGeometryVisible(0.0F, SOURCE));
    REQUIRE_FALSE(farTerrainLodConnectedGeometryVisible(0.0F, TARGET));
    REQUIRE_FALSE(farTerrainLodConnectedGeometryVisible(0.5F, SOURCE));
    REQUIRE(farTerrainLodConnectedGeometryVisible(0.5F, TARGET));

    constexpr unsigned int EMERGENCY_SOURCE = SOURCE | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    constexpr unsigned int EMERGENCY_TARGET = TARGET | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    const FarTerrainTransitionSample beforeEmergencySwap =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS - 0.001F);
    const FarTerrainTransitionSample afterEmergencySwap =
        sampleFarTerrainTransition(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS + 0.001F);
    REQUIRE(farTerrainLodTerrainVisible(beforeEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(beforeEmergencySwap.progress, EMERGENCY_TARGET));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(afterEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(afterEmergencySwap.progress, EMERGENCY_TARGET));
    REQUIRE(farTerrainLodConnectedGeometryVisible(beforeEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE_FALSE(
        farTerrainLodConnectedGeometryVisible(beforeEmergencySwap.progress, EMERGENCY_TARGET));
    REQUIRE_FALSE(
        farTerrainLodConnectedGeometryVisible(afterEmergencySwap.progress, EMERGENCY_SOURCE));
    REQUIRE(farTerrainLodConnectedGeometryVisible(afterEmergencySwap.progress, EMERGENCY_TARGET));
    const float emergencySwapProgress =
        farTerrainLodTransitionProgressAtSeconds(FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS);
    REQUIRE(farTerrainLodTerrainSwapProgress(EMERGENCY_SOURCE) ==
            Catch::Approx(emergencySwapProgress));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress, EMERGENCY_SOURCE) ==
            Catch::Approx(1.0F));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress - 0.030F, EMERGENCY_SOURCE) ==
            Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE(farTerrainLodTerrainFog(emergencySwapProgress + 0.030F, EMERGENCY_TARGET) ==
            Catch::Approx(0.0F).margin(1.0e-6F));
    REQUIRE_FALSE(farTerrainLodTerrainVisible(1.0F, SOURCE));
    REQUIRE(farTerrainLodTerrainVisible(1.0F, TARGET));
    REQUIRE(sizeof(Vertex) == 16);
}

TEST_CASE("Water keeps one owner through far LOD and exact handoffs",
          "[render][far-terrain][water][transition][ownership][flicker][regression]") {
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    STATIC_REQUIRE(sizeof(Vertex) == 16);

    const auto requireSingleOwner = [=](bool exactOwned, float progress) {
        const bool sourceFarWater =
            !exactOwned && farTerrainLodConnectedGeometryVisible(progress, SOURCE);
        const bool targetFarWater =
            !exactOwned && farTerrainLodConnectedGeometryVisible(progress, TARGET);
        const unsigned int ownerCount = static_cast<unsigned int>(exactOwned) +
                                        static_cast<unsigned int>(sourceFarWater) +
                                        static_cast<unsigned int>(targetFarWater);
        CAPTURE(exactOwned, progress, sourceFarWater, targetFarWater);
        REQUIRE(ownerCount == 1U);
        if (!exactOwned) {
            REQUIRE(sourceFarWater == (progress < 0.5F));
            REQUIRE(targetFarWater == (progress >= 0.5F));
        }
    };

    for (int sample = 0; sample <= 64; ++sample) {
        const float progress = static_cast<float>(sample) / 64.0F;
        requireSingleOwner(false, progress);
    }

    bool exactOwned = false;
    for (const auto [builtRevision, currentRevision] :
         {std::pair{4U, 5U}, std::pair{5U, 5U}, std::pair{5U, 6U}, std::pair{6U, 6U}}) {
        exactOwned = farTerrainExactSectionOwnsSurface(exactOwned, builtRevision, currentRevision);
        for (const float progress : {0.0F, 0.25F, 0.5F, 0.75F, 1.0F}) {
            CAPTURE(builtRevision, currentRevision);
            requireSingleOwner(exactOwned, progress);
        }
    }
    REQUIRE(exactOwned);

    // Once the replacement completes, the scheduler submits only the new
    // regular draw. The transition helper must admit that sole owner.
    REQUIRE(farTerrainLodConnectedGeometryVisible(0.0F, FAR_TERRAIN_DRAW_FLAG));
}

TEST_CASE("Preview parents are replaced atomically before final LODs display",
          "[render][far-terrain][authority][lod][upload][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    STATIC_REQUIRE(farTerrainAuthoritySatisfies(Quality::FINAL, Quality::PREVIEW));
    STATIC_REQUIRE(farTerrainAuthoritySatisfies(Quality::FINAL, Quality::FINAL));
    STATIC_REQUIRE_FALSE(farTerrainAuthoritySatisfies(Quality::PREVIEW, Quality::FINAL));
    STATIC_REQUIRE(farTerrainAuthorityMayReplace(Quality::PREVIEW, Quality::FINAL));
    STATIC_REQUIRE_FALSE(farTerrainAuthorityMayReplace(Quality::FINAL, Quality::PREVIEW));

    STATIC_REQUIRE(farTerrainAuthorityAllowsDisplayedStep(Quality::PREVIEW, Quality::PREVIEW,
                                                          FarTerrainStep::THIRTY_TWO));
    STATIC_REQUIRE(farTerrainAuthorityAllowsDisplayedStep(Quality::PREVIEW, Quality::PREVIEW,
                                                          FarTerrainStep::SIXTEEN));
    STATIC_REQUIRE_FALSE(farTerrainAuthorityAllowsDisplayedStep(Quality::PREVIEW, Quality::FINAL,
                                                                FarTerrainStep::SIXTEEN));
    STATIC_REQUIRE_FALSE(farTerrainAuthorityAllowsDisplayedStep(Quality::FINAL, Quality::PREVIEW,
                                                                FarTerrainStep::TWO));
    STATIC_REQUIRE(farTerrainAuthorityAllowsDisplayedStep(Quality::FINAL, Quality::FINAL,
                                                          FarTerrainStep::TWO));
    STATIC_REQUIRE(farTerrainAuthorityAllowsDisplayedStepDuringParentPromotion(
        Quality::FINAL, Quality::PREVIEW, Quality::PREVIEW, FarTerrainStep::SIXTEEN));
    STATIC_REQUIRE(farTerrainAuthorityAllowsDisplayedStepDuringParentPromotion(
        Quality::FINAL, Quality::PREVIEW, Quality::FINAL, FarTerrainStep::SIXTEEN));
    STATIC_REQUIRE_FALSE(farTerrainAuthorityAllowsDisplayedStepDuringParentPromotion(
        Quality::FINAL, std::nullopt, Quality::PREVIEW, FarTerrainStep::SIXTEEN));
    STATIC_REQUIRE_FALSE(
        farTerrainRefinementRequiresFinalAuthority(Quality::FINAL, Quality::PREVIEW, false));
    STATIC_REQUIRE(farTerrainRefinementRequiresFinalAuthority(Quality::FINAL, std::nullopt, false));
    STATIC_REQUIRE(
        farTerrainRefinementRequiresFinalAuthority(Quality::FINAL, Quality::PREVIEW, true));

    constexpr FarTerrainKey parent{7, -9, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey previewChild{7, -9, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey neighboringPreviewParent{8, -9, FarTerrainStep::THIRTY_TWO};
    STATIC_REQUIRE_FALSE(
        farTerrainPreviewChildDependsOnParentSource(parent, parent, Quality::PREVIEW));
    STATIC_REQUIRE(
        farTerrainPreviewChildDependsOnParentSource(parent, previewChild, Quality::PREVIEW));
    STATIC_REQUIRE_FALSE(
        farTerrainPreviewChildDependsOnParentSource(parent, previewChild, Quality::FINAL));
    STATIC_REQUIRE_FALSE(farTerrainPreviewChildDependsOnParentSource(
        parent, neighboringPreviewParent, Quality::PREVIEW));

    constexpr FarTerrainUploadAction INSERT =
        farTerrainUploadAction(std::nullopt, Quality::PREVIEW);
    constexpr FarTerrainUploadAction REPLACE =
        farTerrainUploadAction(Quality::PREVIEW, Quality::FINAL);
    STATIC_REQUIRE(INSERT == FarTerrainUploadAction::INSERT_AFTER_UPLOAD);
    STATIC_REQUIRE(REPLACE == FarTerrainUploadAction::REPLACE_AFTER_UPLOAD);
    STATIC_REQUIRE(farTerrainUploadAction(Quality::FINAL, Quality::PREVIEW) ==
                   FarTerrainUploadAction::REJECT);
    STATIC_REQUIRE(farTerrainUploadAction(Quality::FINAL, Quality::FINAL) ==
                   FarTerrainUploadAction::REJECT);
    STATIC_REQUIRE_FALSE(farTerrainUploadCommitAllowed(REPLACE, false));
    STATIC_REQUIRE(farTerrainUploadCommitAllowed(REPLACE, true));

    constexpr FarTerrainKey KEY{-3, 7, FAR_TERRAIN_BASE_STEP};
    const FarTerrainSource source = farTerrainTestSource();
    const auto preview = FarTerrainMesher::build(KEY, source, Quality::PREVIEW);
    const auto final = FarTerrainMesher::build(KEY, source, Quality::FINAL);
    REQUIRE(preview->authorityQuality == Quality::PREVIEW);
    REQUIRE(final->authorityQuality == Quality::FINAL);
    REQUIRE(preview->deterministicHash != final->deterministicHash);
    REQUIRE(preview->vertices.size() == final->vertices.size());
    REQUIRE(preview->indices == final->indices);
    REQUIRE(
        farTerrainAuthorityPromotionPreservesWater(preview->waterTopology, final->waterTopology));
    REQUIRE(farTerrainWaterPromotionAction(preview->waterTopology, final->waterTopology) ==
            FarTerrainWaterPromotionAction::MATCHED_TOPOLOGY_TRANSITION);

    FarTerrainWaterTopologySignature disconnected = final->waterTopology;
    ++disconnected.bodyIdentityCount;
    REQUIRE_FALSE(farTerrainAuthorityPromotionPreservesWater(preview->waterTopology, disconnected));
    REQUIRE(farTerrainWaterPromotionAction(preview->waterTopology, disconnected) ==
            FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP);

    FarTerrainWaterTopologySignature dry{};
    FarTerrainWaterTopologySignature wet{};
    wet.bodyIdentityCount = 1;
    wet.bodyIdentityHash = 0x42U;
    REQUIRE(farTerrainWaterPromotionAction(dry, wet) ==
            FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP);
    REQUIRE(farTerrainWaterPromotionAction(wet, dry) ==
            FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP);
    disconnected = final->waterTopology;
    disconnected.connectivityHash ^= 1U;
    REQUIRE_FALSE(farTerrainAuthorityPromotionPreservesWater(preview->waterTopology, disconnected));
    REQUIRE(farTerrainWaterPromotionAction(preview->waterTopology, disconnected) ==
            FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP);

    // Both authority surfaces remain submitted during promotion. Their water
    // topology changes ownership with terrain at the fog-covered midpoint.
    for (const float elapsed :
         {0.0F, FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.5F, FAR_TERRAIN_LOD_TRANSITION_SECONDS}) {
        const FarTerrainTransitionSample sample = sampleFarTerrainTransition(elapsed);
        const uint32_t sourceFlags = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
        const uint32_t targetFlags = sourceFlags | FAR_TERRAIN_LOD_TARGET_FLAG;
        CAPTURE(sample.progress);
        const bool sourceWater =
            farTerrainLodConnectedGeometryVisible(sample.progress, sourceFlags);
        const bool targetWater =
            farTerrainLodConnectedGeometryVisible(sample.progress, targetFlags);
        REQUIRE(sourceWater != targetWater);
        REQUIRE(sourceWater == (sample.progress < 0.5F));
        REQUIRE(targetWater == (sample.progress >= 0.5F));
    }
}

TEST_CASE("Exact collision ownership survives PREVIEW to FINAL promotion",
          "[render][far-terrain][exact][collision][preview][promotion][regression]") {
    using Quality = FarTerrainAuthorityQuality;

    // Collision already consumes loaded exact cubes. A complete exact column
    // therefore draws and clips its far parent even while that parent remains
    // PREVIEW or participates in the atomic FINAL promotion. Far authority
    // changes may refine the remaining surface, but cannot conceal live exact
    // collision for a frame.
    constexpr FarTerrainExactVisualOwnership READY_EXACT =
        farTerrainExactVisualOwnership(true, true, true, true);
    STATIC_REQUIRE(READY_EXACT.drawExact);
    STATIC_REQUIRE(READY_EXACT.clipFar);
    for (const Quality displayedQuality : {Quality::PREVIEW, Quality::FINAL}) {
        for (const bool promotionActive : {false, true}) {
            CAPTURE(displayedQuality, promotionActive);
            REQUIRE(farTerrainExactVisualOwnership(true, true, true, true) == READY_EXACT);
        }
    }

    constexpr FarTerrainExactVisualOwnership PARTIAL_COLUMN =
        farTerrainExactVisualOwnership(true, false, true, true);
    STATIC_REQUIRE_FALSE(PARTIAL_COLUMN.drawExact);
    STATIC_REQUIRE_FALSE(PARTIAL_COLUMN.clipFar);
    constexpr FarTerrainExactVisualOwnership MISSING_PARENT =
        farTerrainExactVisualOwnership(true, false, false, true);
    STATIC_REQUIRE(MISSING_PARENT.drawExact);
    STATIC_REQUIRE_FALSE(MISSING_PARENT.clipFar);

    // A revision-ready empty exact section has no geometry to submit, but it
    // still participates in the complete column publication. The far parent
    // yields to the exact column's lower visible sections or intentional
    // empty space.
    constexpr FarTerrainExactVisualOwnership EMPTY_EXACT_SECTION =
        farTerrainExactVisualOwnership(true, true, true, false);
    STATIC_REQUIRE_FALSE(EMPTY_EXACT_SECTION.drawExact);
    STATIC_REQUIRE(EMPTY_EXACT_SECTION.clipFar);

    STATIC_REQUIRE(farTerrainExactCollisionOwnsSection(true, true, true, true));
    STATIC_REQUIRE_FALSE(farTerrainExactCollisionOwnsSection(true, false, true, true));
    STATIC_REQUIRE(farTerrainExactCollisionOwnsSection(true, false, false, true));
    STATIC_REQUIRE(farTerrainExactCollisionOwnsSection(false, false, true, true));
    STATIC_REQUIRE_FALSE(farTerrainExactCollisionOwnsSection(false, false, true, false));
}

TEST_CASE("Production far terrain has no crack-hiding panel vertex class",
          "[render][far-terrain][shader][transition][skirt][regression]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 80.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto mesh = FarTerrainMesher::build({-98, -39, FarTerrainStep::TWO}, source);
    constexpr uint32_t RESERVED_PANEL_ATTRIBUTE = 1U << 29U;
    const auto markedVertices =
        std::count_if(mesh->vertices.begin(), mesh->vertices.end(), [](const Vertex& vertex) {
            return (vertex.faceAttr & RESERVED_PANEL_ATTRIBUTE) != 0U;
        });
    REQUIRE(markedVertices == 0);
}

TEST_CASE("Pinned seed forty-two handoff emits no downward tile skirt at any LOD",
          "[render][far-terrain][skirt][ownership][exact][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr std::array STEPS = {
        FarTerrainStep::TWO,     FarTerrainStep::FOUR,       FarTerrainStep::EIGHT,
        FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    for (const FarTerrainStep step : STEPS) {
        const auto mesh = FarTerrainMesher::build(FarTerrainKey{2, -6, step}, source);
        CAPTURE(farTerrainStepSize(step), mesh->vertices.size(), mesh->indices.size());
        constexpr uint32_t RESERVED_PANEL_ATTRIBUTE = 1U << 29U;
        REQUIRE(std::ranges::none_of(mesh->vertices, [](const Vertex& vertex) {
            return (vertex.faceAttr & RESERVED_PANEL_ATTRIBUTE) != 0U;
        }));
    }
}

TEST_CASE("Far terrain view selection is circular ordered and negative-coordinate safe",
          "[render][far-terrain][selection]") {
    constexpr double cameraX = -320.5;
    constexpr double cameraZ = -511.25;
    constexpr int exactRadius = 32;
    constexpr int visibleRadius = 512;
    const double exactSquared = std::pow(exactRadius * CHUNK_EDGE, 2.0);
    const double visibleSquared = std::pow(visibleRadius * CHUNK_EDGE, 2.0);

    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(cameraX, cameraZ, visibleRadius, selected);
    REQUIRE_FALSE(selected.empty());

    std::array<bool, 4> reachedStep{};
    bool sawNegativeTile = false;
    bool sawExactBoundaryOverlap = false;
    bool sawTileWhollyInsideExactDisk = false;
    double previousDistance = -1.0;
    for (const FarTerrainViewTile& tile : selected) {
        REQUIRE(tile.distanceSquared >= previousDistance);
        previousDistance = tile.distanceSquared;
        sawNegativeTile = sawNegativeTile || tile.key.tileX < 0 || tile.key.tileZ < 0;
        reachedStep[tile.key.step == FarTerrainStep::TWO     ? 0
                    : tile.key.step == FarTerrainStep::FOUR  ? 1
                    : tile.key.step == FarTerrainStep::EIGHT ? 2
                                                             : 3] = true;

        double nearestSquared = 0.0;
        if (cameraX < tile.bounds.minX)
            nearestSquared += std::pow(tile.bounds.minX - cameraX, 2.0);
        if (cameraX > tile.bounds.maxX)
            nearestSquared += std::pow(cameraX - tile.bounds.maxX, 2.0);
        if (cameraZ < tile.bounds.minZ)
            nearestSquared += std::pow(tile.bounds.minZ - cameraZ, 2.0);
        if (cameraZ > tile.bounds.maxZ)
            nearestSquared += std::pow(cameraZ - tile.bounds.maxZ, 2.0);
        REQUIRE(nearestSquared < visibleSquared);
        sawExactBoundaryOverlap = sawExactBoundaryOverlap || nearestSquared < exactSquared;

        double farthestSquared = 0.0;
        for (const int64_t x : {tile.bounds.minX, tile.bounds.maxX}) {
            for (const int64_t z : {tile.bounds.minZ, tile.bounds.maxZ}) {
                farthestSquared =
                    std::max(farthestSquared, std::pow(static_cast<double>(x) - cameraX, 2.0) +
                                                  std::pow(static_cast<double>(z) - cameraZ, 2.0));
            }
        }
        sawTileWhollyInsideExactDisk =
            sawTileWhollyInsideExactDisk || farthestSquared <= exactSquared;
    }

    REQUIRE(sawNegativeTile);
    REQUIRE(sawExactBoundaryOverlap);
    REQUIRE(sawTileWhollyInsideExactDisk);
    REQUIRE(
        std::all_of(reachedStep.begin(), reachedStep.end(), [](bool reached) { return reached; }));
}

TEST_CASE("Far terrain coverage stops before the nearest absent base",
          "[render][far-terrain][coverage][residency]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(128.0, 128.0, 64, selected);
    REQUIRE(selected.size() > 8);

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    for (const FarTerrainViewTile& tile : selected) {
        resident.insert({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    const auto isResident = [&](const FarTerrainKey& key) { return resident.contains(key); };
    const FarTerrainCoverageFrontier complete = farTerrainCoverageFrontier(selected, isResident);
    REQUIRE(complete.complete);
    REQUIRE(complete.missingBaseTiles == 0);
    REQUIRE(complete.distanceBlocks == 0.0F);

    const FarTerrainViewTile& missing = selected[selected.size() / 3];
    resident.erase({missing.key.tileX, missing.key.tileZ, FAR_TERRAIN_BASE_STEP});
    const FarTerrainCoverageFrontier incomplete = farTerrainCoverageFrontier(selected, isResident);
    REQUIRE_FALSE(incomplete.complete);
    REQUIRE(incomplete.missingBaseTiles == 1);
    REQUIRE(incomplete.distanceBlocks == Catch::Approx(std::sqrt(missing.distanceSquared)));
    REQUIRE(incomplete.distanceSquaredBlocks == missing.distanceSquared);
    REQUIRE(farTerrainCoverageFog(incomplete.distanceBlocks, incomplete.distanceBlocks) == 1.0F);
    REQUIRE_FALSE(farTerrainCoverageVisible(incomplete.distanceBlocks, incomplete.distanceBlocks));
    REQUIRE(farTerrainCoverageVisible(incomplete.distanceBlocks - 1.0F, incomplete.distanceBlocks));

    for (const FarTerrainViewTile& tile : selected) {
        CAPTURE(tile.key.tileX, tile.key.tileZ, tile.distanceSquared,
                incomplete.distanceSquaredBlocks);
        REQUIRE(farTerrainCoverageDrawEligible(tile.distanceSquared, incomplete) ==
                (tile.distanceSquared < missing.distanceSquared));
    }
}

TEST_CASE("A nearby cold coverage frontier preserves a clear camera neighborhood",
          "[render][far-terrain][coverage][fog][cold-start][regression]") {
    constexpr float NEAR_FRONTIER = 64.0F;
    STATIC_REQUIRE(FAR_TERRAIN_COVERAGE_MIN_FADE_BLOCKS == 16.0F);
    STATIC_REQUIRE(FAR_TERRAIN_COVERAGE_FADE_FRACTION == 0.125F);
    REQUIRE(farTerrainCoverageFadeBlocks(NEAR_FRONTIER) == 16.0F);
    REQUIRE(farTerrainCoverageFog(0.0F, NEAR_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(48.0F, NEAR_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(56.0F, NEAR_FRONTIER) == Catch::Approx(0.5F));
    REQUIRE(farTerrainCoverageFog(NEAR_FRONTIER, NEAR_FRONTIER) == 1.0F);

    // Once the connected prefix is at least eight tiles deep, retain the full
    // 256-block horizon taper used by settled long-distance coverage.
    constexpr float DISTANT_FRONTIER = 2048.0F;
    REQUIRE(farTerrainCoverageFadeBlocks(DISTANT_FRONTIER) == FAR_TERRAIN_COVERAGE_FADE_BLOCKS);
    REQUIRE(farTerrainCoverageFog(1792.0F, DISTANT_FRONTIER) == 0.0F);
    REQUIRE(farTerrainCoverageFog(1920.0F, DISTANT_FRONTIER) == Catch::Approx(0.5F));
}

TEST_CASE("Connected parents refine every distance tier before full horizon coverage",
          "[render][far-terrain][coverage][lod][priority][cold-start][camera-jump][regression]") {
    FarTerrainViewTile near;
    near.key = {0, 0, FarTerrainStep::TWO};
    near.distanceSquared = 560.0 * 560.0;
    near.distanceChunks = 35.0;

    FarTerrainCoverageFrontier incomplete;
    incomplete.complete = false;
    incomplete.missingBaseTiles = 12;
    incomplete.distanceBlocks = 900.0F;
    incomplete.distanceSquaredBlocks = 900.0 * 900.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, false));

    // The camera target may build alongside its missing parent. It remains
    // undisplayable until the parent is resident, so this reduces cold
    // latency without exposing an isolated refinement.
    near.distanceSquared = 0.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, false, true));
    REQUIRE_FALSE(farTerrainInitialDisplayedStep(farTerrainStepMask(FarTerrainStep::TWO)));

    // A camera jump can contract the exact handoff to zero. Every connected
    // parent remains eligible, independent of that exact-residency radius.
    near.distanceSquared = 500.0 * 500.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, true));
    near.distanceSquared = 513.0 * 513.0;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 0.0F, incomplete, true));

    // No target may appear at or beyond the nearest missing base, even when
    // the tile is inside the urgent distance band and its own parent exists.
    near.distanceSquared = incomplete.distanceSquaredBlocks;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.distanceSquared = std::nextafter(incomplete.distanceSquaredBlocks, 0.0);
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));

    near.key.step = FarTerrainStep::ONE;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::FOUR;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::EIGHT;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::SIXTEEN;
    REQUIRE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::THIRTY_TWO;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, 512.0F, incomplete, true));
    near.key.step = FarTerrainStep::TWO;
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(
        near, std::numeric_limits<float>::infinity(), incomplete, true));
    REQUIRE_FALSE(farTerrainConnectedRefinementEligible(near, -1.0F, incomplete, true));

    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT == 12);
    STATIC_REQUIRE(FAR_TERRAIN_MIN_BASE_WORKERS_DURING_COVERAGE == 4);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME == 12);
    STATIC_REQUIRE(FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(1, true) == 1);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(2, true) == 2);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(4, true) == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(8, true) == 4);
    STATIC_REQUIRE(farTerrainBaseWorkerReservation(4, false) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(1, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(2, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(4, true) == 0);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(5, true) == 1);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(8, true) == 4);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(16, true) == 12);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(4, false) == 4);
}

TEST_CASE("V4 entry stages connected parents before gameplay refinement",
          "[render][far-terrain][selection][startup][regression]") {
    STATIC_REQUIRE_FALSE(farTerrainFinalStreamingWorkEnabled(false));
    STATIC_REQUIRE(farTerrainFinalStreamingWorkEnabled(true));
    STATIC_REQUIRE(farTerrainFinalStreamingWorkEnabled(false, true));
    STATIC_REQUIRE_FALSE(farTerrainOptionalStreamingWorkEnabled(false, false));
    STATIC_REQUIRE_FALSE(farTerrainOptionalStreamingWorkEnabled(false, true));
    STATIC_REQUIRE_FALSE(farTerrainOptionalStreamingWorkEnabled(true, false));
    STATIC_REQUIRE(farTerrainOptionalStreamingWorkEnabled(true, true));

    REQUIRE(farTerrainHorizonRadiusValid(FAR_TERRAIN_NEAR_CHUNK_RADIUS));
    REQUIRE(farTerrainHorizonRadiusValid(FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS));
    REQUIRE(farTerrainHorizonRadiusValid(FAR_TERRAIN_MAX_CHUNK_RADIUS));
    REQUIRE_FALSE(farTerrainHorizonRadiusValid(FAR_TERRAIN_NEAR_CHUNK_RADIUS - 1));
    REQUIRE_FALSE(farTerrainHorizonRadiusValid(FAR_TERRAIN_MAX_CHUNK_RADIUS + 1));
    REQUIRE(farTerrainEntryHorizonViewDistance(MAX_RENDER_DISTANCE_CHUNKS) ==
            MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(FAR_TERRAIN_NEAR_CHUNK_RADIUS) ==
            FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(MIN_RENDER_DISTANCE_CHUNKS) ==
            FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(64) == FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS) ==
            FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(MAX_RENDER_DISTANCE_CHUNKS + 1) ==
            MAX_RENDER_DISTANCE_CHUNKS);
    REQUIRE(farTerrainEntryHorizonViewDistance(-1) == 0);

    for (const int configured :
         {MIN_RENDER_DISTANCE_CHUNKS, 32, 64, FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS}) {
        std::vector<FarTerrainViewTile> minimumSelection;
        selectFarTerrainView(127.0, 127.0, farTerrainEntryHorizonViewDistance(configured),
                             minimumSelection);
        std::vector<FarTerrainKey> protectedTargets;
        buildFarTerrainProtectedNearTargets(farTerrainProtectedNearAnchor(127, 127),
                                            minimumSelection, protectedTargets);
        CAPTURE(configured, minimumSelection.size(), protectedTargets.size());
        REQUIRE(protectedTargets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    }

    constexpr double CAMERA_X = -257.25;
    constexpr double CAMERA_Z = 513.75;
    std::vector<FarTerrainViewTile> entry;
    std::vector<FarTerrainViewTile> full;
    selectFarTerrainView(CAMERA_X, CAMERA_Z,
                         farTerrainEntryHorizonViewDistance(MAX_RENDER_DISTANCE_CHUNKS), entry);
    selectFarTerrainView(CAMERA_X, CAMERA_Z, MAX_RENDER_DISTANCE_CHUNKS, full);

    REQUIRE_FALSE(entry.empty());
    REQUIRE(entry.size() == full.size());
    for (size_t index = 0; index < entry.size(); ++index) {
        CAPTURE(index);
        REQUIRE(entry[index].key == full[index].key);
        REQUIRE(entry[index].bounds.minX == full[index].bounds.minX);
        REQUIRE(entry[index].bounds.minZ == full[index].bounds.minZ);
        REQUIRE(entry[index].distanceSquared == full[index].distanceSquared);
    }
    const auto isEntryParentResident = [&](const FarTerrainKey& key) {
        return key.step == FAR_TERRAIN_BASE_STEP &&
               std::ranges::any_of(entry, [&](const FarTerrainViewTile& tile) {
                   return tile.key.tileX == key.tileX && tile.key.tileZ == key.tileZ;
               });
    };

    const FarTerrainCoverageFrontier entryCoverage =
        farTerrainCoverageFrontier(entry, isEntryParentResident);
    REQUIRE(entryCoverage.complete);
    REQUIRE(entryCoverage.missingBaseTiles == 0);
    REQUIRE(farTerrainConnectedRefinementLaneOpen(entryCoverage));

    const auto entryPages = farTerrainCoarseAuthorityPages(entry, CAMERA_X, CAMERA_Z);
    const auto fullPages = farTerrainCoarseAuthorityPages(full, CAMERA_X, CAMERA_Z);
    REQUIRE_FALSE(entryPages.empty());
    REQUIRE(entryPages == fullPages);

    FarTerrainCoverageFrontier premature;
    premature.complete = false;
    premature.missingBaseTiles = 1;
    premature.distanceBlocks =
        static_cast<float>(FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS * CHUNK_EDGE - 1);
    REQUIRE_FALSE(farTerrainConnectedRefinementLaneOpen(premature));
}

TEST_CASE("Cold zero-radius entry leaves final parents to protected publication",
          "[render][far-terrain][authority][exact][startup][priority][regression]") {
    constexpr double CAMERA_X = -257.25;
    constexpr double CAMERA_Z = 513.75;
    constexpr float NOMINAL_EXACT_BLOCKS = COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS * CHUNK_EDGE;
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(CAMERA_X, CAMERA_Z,
                         farTerrainEntryHorizonViewDistance(MAX_RENDER_DISTANCE_CHUNKS), selected);
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(CAMERA_X, CAMERA_Z, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS, {}, {},
                               [](ChunkPos) { return false; });

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> finalParents;
    const auto isFinal = [&](const FarTerrainKey& key) { return finalParents.contains(key); };
    std::vector<FarTerrainKey> upgrades;
    const uint32_t required = buildFarTerrainFinalParentUpgradeOrder(
        selected, CAMERA_X, CAMERA_Z, NOMINAL_EXACT_BLOCKS, handoff, isFinal, upgrades);
    REQUIRE(required == 0);
    REQUIRE(upgrades.size() == selected.size());
    REQUIRE(std::ranges::all_of(upgrades, [&](const FarTerrainKey& key) {
        return !farTerrainRequiresCoverageParent(CAMERA_X, CAMERA_Z, {key.tileX, key.tileZ},
                                                 NOMINAL_EXACT_BLOCKS, handoff);
    }));
}

TEST_CASE("Full horizon residency orders every coarse parent before refinements",
          "[render][far-terrain][coverage][residency][priority]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(-257.25, 513.75, 512, selected);
    REQUIRE(selected.size() > 3'000);

    std::vector<FarTerrainKey> order;
    buildFarTerrainResidencyOrder(selected, order);
    REQUIRE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE(order.size() >= selected.size());
    for (size_t index = 0; index < selected.size(); ++index) {
        INFO("base index " << index);
        REQUIRE(order[index].tileX == selected[index].key.tileX);
        REQUIRE(order[index].tileZ == selected[index].key.tileZ);
        REQUIRE(farTerrainIsBaseStep(order[index].step));
        if (index > 0) {
            REQUIRE(selected[index - 1].distanceSquared <= selected[index].distanceSquared);
        }
    }
    for (size_t index = selected.size(); index < order.size(); ++index) {
        INFO("refinement index " << index);
        REQUIRE_FALSE(farTerrainIsBaseStep(order[index].step));
    }

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> unique(order.begin(), order.end());
    REQUIRE(unique.size() == order.size());
    REQUIRE(farTerrainResidencyMembershipMatches(selected, unique));
    size_t refinementIndex = selected.size();
    for (const FarTerrainStep step : FAR_TERRAIN_REFINEMENT_STEPS) {
        for (size_t index = 0; index < selected.size(); ++index) {
            CAPTURE(index, selected[index].key.tileX, selected[index].key.tileZ,
                    farTerrainStepSize(selected[index].key.step), farTerrainStepSize(step));
            const FarTerrainStep target = farTerrainResidencyTarget(selected[index]);
            if (farTerrainStepSize(step) < farTerrainStepSize(target))
                continue;
            REQUIRE(order[refinementIndex++] ==
                    FarTerrainKey{selected[index].key.tileX, selected[index].key.tileZ, step});
        }
    }
    REQUIRE(refinementIndex == order.size());
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO) ==
            FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO) ==
            FarTerrainStep::EIGHT);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::EIGHT, FarTerrainStep::TWO) ==
            FarTerrainStep::FOUR);
    REQUIRE(farTerrainNextDisplayedStep(FarTerrainStep::FOUR, FarTerrainStep::TWO) ==
            FarTerrainStep::TWO);

    std::reverse(selected.begin(), selected.end());
    REQUIRE_FALSE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE(farTerrainResidencyMembershipMatches(selected, unique));

    selected.front().key.step = selected.front().key.step == FarTerrainStep::TWO
                                    ? FarTerrainStep::FOUR
                                    : FarTerrainStep::TWO;
    REQUIRE_FALSE(farTerrainResidencyOrderMatches(selected, order));
    REQUIRE_FALSE(farTerrainResidencyMembershipMatches(selected, unique));
}

TEST_CASE("Residency cache priority advances one global displayable wavefront at a time",
          "[render][far-terrain][coverage][residency][lod][priority][regression]") {
    std::vector<FarTerrainViewTile> selected(3);
    selected[0].key = {0, 0, FarTerrainStep::ONE};
    selected[1].key = {1, 0, FarTerrainStep::TWO};
    selected[2].key = {2, 0, FarTerrainStep::SIXTEEN};
    for (FarTerrainViewTile& tile : selected)
        tile.distanceChunks = 64.0;
    std::vector<FarTerrainKey> order;
    buildFarTerrainResidencyOrder(selected, order);

    const std::vector<FarTerrainKey> expected{
        {0, 0, FarTerrainStep::THIRTY_TWO}, {1, 0, FarTerrainStep::THIRTY_TWO},
        {2, 0, FarTerrainStep::THIRTY_TWO}, {0, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::SIXTEEN},    {2, 0, FarTerrainStep::SIXTEEN},
        {0, 0, FarTerrainStep::EIGHT},      {1, 0, FarTerrainStep::EIGHT},
        {0, 0, FarTerrainStep::FOUR},       {1, 0, FarTerrainStep::FOUR},
        {0, 0, FarTerrainStep::TWO},        {1, 0, FarTerrainStep::TWO},
        {0, 0, FarTerrainStep::ONE},
    };
    REQUIRE(order == expected);
    REQUIRE(farTerrainResidencyOrderMatches(selected, order));

    constexpr std::array critical{ColumnPos{0, 0}};
    buildFarTerrainResidencyOrder(selected, order, critical);
    const std::vector<FarTerrainKey> criticalExpected{
        {0, 0, FarTerrainStep::THIRTY_TWO}, {1, 0, FarTerrainStep::THIRTY_TWO},
        {2, 0, FarTerrainStep::THIRTY_TWO}, {0, 0, FarTerrainStep::SIXTEEN},
        {0, 0, FarTerrainStep::EIGHT},      {0, 0, FarTerrainStep::FOUR},
        {0, 0, FarTerrainStep::TWO},        {0, 0, FarTerrainStep::ONE},
        {1, 0, FarTerrainStep::SIXTEEN},    {2, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::EIGHT},      {1, 0, FarTerrainStep::FOUR},
        {1, 0, FarTerrainStep::TWO},
    };
    REQUIRE(order == criticalExpected);
    REQUIRE(farTerrainResidencyOrderMatches(selected, order, critical));
}

TEST_CASE("Critical residency admits every required surface before bridge extras",
          "[render][far-terrain][residency][cache][critical][priority][regression]") {
    constexpr std::array targets{
        FarTerrainKey{0, 0, FarTerrainStep::ONE},  FarTerrainKey{1, 0, FarTerrainStep::ONE},
        FarTerrainKey{0, 1, FarTerrainStep::ONE},  FarTerrainKey{1, 1, FarTerrainStep::ONE},
        FarTerrainKey{-1, 0, FarTerrainStep::TWO},
    };
    std::vector<FarTerrainKey> order;
    buildFarTerrainCriticalResidencyOrder(targets, order);
    REQUIRE(order.size() > targets.size() * 2);
    REQUIRE(std::ranges::equal(std::span(order).first(targets.size()), targets));
    for (size_t index = 0; index < targets.size(); ++index) {
        CAPTURE(index);
        REQUIRE(order[targets.size() + index] ==
                FarTerrainKey{targets[index].tileX, targets[index].tileZ, FAR_TERRAIN_BASE_STEP});
    }
    REQUIRE(std::ranges::none_of(std::span(order).first(targets.size() * 2), [](FarTerrainKey key) {
        return key.step == FarTerrainStep::FOUR || key.step == FarTerrainStep::EIGHT ||
               key.step == FarTerrainStep::SIXTEEN;
    }));
}

TEST_CASE("Current protected residency strictly precedes directional prediction",
          "[render][far-terrain][residency][priority][prediction][regression]") {
    const std::array current{FarTerrainKey{0, 0, FarTerrainStep::ONE},
                             FarTerrainKey{1, 0, FarTerrainStep::TWO}};
    const std::array predicted{FarTerrainKey{1, 0, FarTerrainStep::TWO},
                               FarTerrainKey{2, 0, FarTerrainStep::FOUR}};
    std::vector<FarTerrainKey> currentOrder;
    std::vector<FarTerrainKey> predictedOrder;
    std::vector<FarTerrainKey> tiered;
    buildFarTerrainCriticalResidencyOrder(current, currentOrder);
    buildFarTerrainCriticalResidencyOrder(predicted, predictedOrder);
    buildFarTerrainTieredCriticalResidencyOrder(current, predicted, tiered);

    REQUIRE(tiered.size() >= currentOrder.size());
    REQUIRE(std::ranges::equal(currentOrder, std::span(tiered).first(currentOrder.size())));
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> unique(tiered.begin(), tiered.end());
    REQUIRE(unique.size() == tiered.size());
    for (const FarTerrainKey key : predictedOrder) {
        CAPTURE(key.tileX, key.tileZ, farTerrainStepSize(key.step));
        REQUIRE(unique.contains(key));
        if (!std::ranges::contains(currentOrder, key)) {
            REQUIRE(static_cast<size_t>(std::ranges::find(tiered, key) - tiered.begin()) >=
                    currentOrder.size());
        }
    }

    std::vector<FarTerrainViewTile> selected(1);
    selected.front().key = {0, 0, FarTerrainStep::SIXTEEN};
    selected.front().distanceChunks = 64.0;
    std::vector<FarTerrainKey> broad;
    buildFarTerrainResidencyOrder(selected, broad);
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(broad.begin(), broad.end());
    wanted.insert(predictedOrder.begin(), predictedOrder.end());
    REQUIRE(farTerrainResidencyMembershipMatches(selected, wanted, predictedOrder));
    wanted.erase(FarTerrainKey{2, 0, FAR_TERRAIN_BASE_STEP});
    REQUIRE_FALSE(farTerrainResidencyMembershipMatches(selected, wanted, predictedOrder));
    wanted.insert(FarTerrainKey{2, 0, FAR_TERRAIN_BASE_STEP});
    wanted.insert(FarTerrainKey{99, 99, FarTerrainStep::SIXTEEN});
    REQUIRE_FALSE(farTerrainResidencyMembershipMatches(selected, wanted, predictedOrder));

    STATIC_REQUIRE(farTerrainProtectedFinalSubmissionFloor(
                       FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME, true) == 8);
    STATIC_REQUIRE(farTerrainProtectedFinalSubmissionFloor(
                       FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME, false) == 12);
}

TEST_CASE("Coarse horizon authority prefetch deduplicates native page closures",
          "[render][far-terrain][coverage][authority][startup][regression]") {
    FarTerrainViewTile interior;
    interior.bounds = {.minX = 256, .maxX = 512, .minZ = 256, .maxZ = 512};
    const std::array oneInterior{interior};
    const auto onePlan = farTerrainCoarseAuthorityPages(oneInterior, 384.0, 384.0);
    std::set<worldgen::learned::TerrainPageCoordinate> oneExpected;
    // V4 base geometry consumes one 32-block sample and topology apron, not
    // the retired 384-block far-climate control halo. This tile remains
    // entirely in native hydrology owner (0, 0).
    const auto ownerPages = worldgen::nativeHydrologyRequiredAuthorityPages(0, 0);
    oneExpected.insert(ownerPages.begin(), ownerPages.end());
    REQUIRE(std::set(onePlan.begin(), onePlan.end()) == oneExpected);

    // A base parent samples one 32-block geometry and topology apron. This
    // negative-coordinate parent therefore reaches native owner (-3, 2),
    // including that owner's two-raster-cell learned-authority apron.
    std::vector<FarTerrainViewTile> reportedOuterTiles;
    selectFarTerrainView(512.0, 512.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, reportedOuterTiles);
    REQUIRE(std::ranges::any_of(reportedOuterTiles, [](const FarTerrainViewTile& tile) {
        return tile.key.tileX == -23 && tile.key.tileZ == 22;
    }));
    const auto reportedOuterPlan = farTerrainCoarseAuthorityPages(reportedOuterTiles, 512.0, 512.0);
    const auto reportedOuterOwner = worldgen::nativeHydrologyRequiredAuthorityPages(-3, 2);
    for (const worldgen::learned::TerrainPageCoordinate coordinate : reportedOuterOwner) {
        CAPTURE(coordinate.row, coordinate.column);
        REQUIRE(std::ranges::find(reportedOuterPlan, coordinate) != reportedOuterPlan.end());
    }
    REQUIRE(std::ranges::find(reportedOuterPlan,
                              worldgen::learned::TerrainPageCoordinate{.row = 5, .column = -7}) !=
            reportedOuterPlan.end());

    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(0.0, 0.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    const auto pages = farTerrainCoarseAuthorityPages(selected, 0.0, 0.0);
    REQUIRE(selected.size() > 3'000);
    // A 1,024-block authority page serves many 256-block parents. The plan
    // is therefore page-bounded rather than one inference request per tile.
    REQUIRE(pages.size() < selected.size() / 8);
    REQUIRE(std::set(pages.begin(), pages.end()).size() == pages.size());
    REQUIRE(farTerrainCoarseAuthorityPages(selected, 0.0, 0.0) == pages);
    const auto pageDistanceSquared = [](worldgen::learned::TerrainPageCoordinate page) {
        const long double centerX =
            static_cast<long double>(page.column) * worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE +
            worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE / 2.0L;
        const long double centerZ =
            static_cast<long double>(page.row) * worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE +
            worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE / 2.0L;
        return centerX * centerX + centerZ * centerZ;
    };
    for (size_t index = 1; index < pages.size(); ++index)
        REQUIRE(pageDistanceSquared(pages[index - 1]) <= pageDistanceSquared(pages[index]));
}

TEST_CASE("Final base authority plans only exact geometry and transient owner inputs",
          "[render][far-terrain][authority][final][hydrology][signed][regression]") {
    using worldgen::learned::AUTHORITY_PAGE_NATIVE_EDGE;
    using worldgen::learned::MODEL_BLOCK_SCALE;
    using worldgen::learned::TerrainPageCoordinate;

    const auto verify = [&](FarTerrainKey key) {
        const FarTerrainFinalBaseAuthorityDependencies plan =
            farTerrainFinalBaseAuthorityDependencies(key);
        const int64_t originX = key.tileX * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        const int64_t originZ = key.tileZ * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        const auto contains = [&](int64_t x, int64_t z) {
            return x >= plan.minimumWorldX && x < plan.maximumWorldXExclusive &&
                   z >= plan.minimumWorldZ && z < plan.maximumWorldZExclusive;
        };
        const auto containsRect = [&](int64_t minimumX, int64_t minimumZ, int64_t maximumXExclusive,
                                      int64_t maximumZExclusive) {
            return minimumX >= plan.minimumWorldX && minimumZ >= plan.minimumWorldZ &&
                   maximumXExclusive <= plan.maximumWorldXExclusive &&
                   maximumZExclusive <= plan.maximumWorldZExclusive;
        };

        // These are the exact production callback domains for a step-32
        // parent. The one-cell sample and bounds apron is widest. Center,
        // topology, native-water, waterfall, and volcanic recovery probes
        // are all contained by the same half-open support rectangle.
        REQUIRE(contains(originX - 32, originZ - 32));
        REQUIRE(
            contains(originX + FAR_TERRAIN_TILE_EDGE + 32, originZ + FAR_TERRAIN_TILE_EDGE + 32));
        REQUIRE(containsRect(originX - 32, originZ - 32, originX + FAR_TERRAIN_TILE_EDGE + 32,
                             originZ + FAR_TERRAIN_TILE_EDGE + 32));
        REQUIRE(containsRect(originX - 16, originZ - 16, originX + FAR_TERRAIN_TILE_EDGE + 16,
                             originZ + FAR_TERRAIN_TILE_EDGE + 16));
        REQUIRE(containsRect(originX - 4, originZ - 4, originX + FAR_TERRAIN_TILE_EDGE + 4,
                             originZ + FAR_TERRAIN_TILE_EDGE + 4));

        std::set<TerrainPageCoordinate> expectedGeometryPages;
        const auto retainWorldPointPages = [&](int64_t worldX, int64_t worldZ) {
            const auto nativeAxis = [](int64_t worldCoordinate) {
                const int64_t containing =
                    worldgen::learned::floorDivide(worldCoordinate, MODEL_BLOCK_SCALE);
                const int64_t remainder =
                    worldCoordinate - containing * static_cast<int64_t>(MODEL_BLOCK_SCALE);
                const int64_t lower =
                    remainder < MODEL_BLOCK_SCALE / 2 ? containing - 1 : containing;
                return std::array{lower, lower + 1};
            };
            const std::array rows = nativeAxis(worldZ);
            const std::array columns = nativeAxis(worldX);
            for (const int64_t row : rows) {
                for (const int64_t column : columns) {
                    expectedGeometryPages.insert({
                        .row = worldgen::learned::floorDivide(row, AUTHORITY_PAGE_NATIVE_EDGE),
                        .column =
                            worldgen::learned::floorDivide(column, AUTHORITY_PAGE_NATIVE_EDGE),
                    });
                }
            }
        };
        // Expanded terrain vertices and cell-bound corners.
        for (int localZ = -32; localZ <= FAR_TERRAIN_TILE_EDGE + 32; localZ += 32) {
            for (int localX = -32; localX <= FAR_TERRAIN_TILE_EDGE + 32; localX += 32)
                retainWorldPointPages(originX + localX, originZ + localZ);
        }
        // Conservative cell-center probes.
        for (int localZ = -16; localZ <= FAR_TERRAIN_TILE_EDGE + 16; localZ += 32) {
            for (int localX = -16; localX <= FAR_TERRAIN_TILE_EDGE + 16; localX += 32)
                retainWorldPointPages(originX + localX, originZ + localZ);
        }
        // Complete canonical four-block water raster and refinement points.
        for (int localZ = -4; localZ <= FAR_TERRAIN_TILE_EDGE; localZ += 4) {
            for (int localX = -4; localX <= FAR_TERRAIN_TILE_EDGE; localX += 4)
                retainWorldPointPages(originX + localX, originZ + localZ);
        }
        const std::set<TerrainPageCoordinate> plannedGeometryPages(plan.geometryPages.begin(),
                                                                   plan.geometryPages.end());
        REQUIRE(plannedGeometryPages == expectedGeometryPages);
        REQUIRE(std::ranges::is_sorted(plan.geometryPages));

        std::set<std::pair<int64_t, int64_t>> expectedOwners;
        for (int64_t worldZ = plan.minimumWorldZ; worldZ < plan.maximumWorldZExclusive;) {
            for (int64_t worldX = plan.minimumWorldX; worldX < plan.maximumWorldXExclusive;) {
                expectedOwners.emplace(
                    worldgen::learned::floorDivide(worldX, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE),
                    worldgen::learned::floorDivide(worldZ, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
                const int64_t nextOwnerX =
                    (worldgen::learned::floorDivide(worldX, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE) +
                     1) *
                    worldgen::NATIVE_HYDROLOGY_PAGE_EDGE;
                worldX = std::min(nextOwnerX, plan.maximumWorldXExclusive);
            }
            const int64_t nextOwnerZ =
                (worldgen::learned::floorDivide(worldZ, worldgen::NATIVE_HYDROLOGY_PAGE_EDGE) + 1) *
                worldgen::NATIVE_HYDROLOGY_PAGE_EDGE;
            worldZ = std::min(nextOwnerZ, plan.maximumWorldZExclusive);
        }
        std::set<std::pair<int64_t, int64_t>> plannedOwners;
        for (const FarTerrainNativeHydrologyDependency& dependency : plan.nativeHydrology) {
            plannedOwners.emplace(dependency.ownerPageX, dependency.ownerPageZ);
            REQUIRE(dependency.finalTerrainRegion ==
                    worldgen::nativeHydrologyFinalTerrainRegion(dependency.ownerPageX,
                                                                dependency.ownerPageZ));
            REQUIRE(dependency.finalTerrainRegion.height() == 517);
            REQUIRE(dependency.finalTerrainRegion.width() == 517);
        }
        REQUIRE(plannedOwners == expectedOwners);
        REQUIRE(std::ranges::is_sorted(plan.nativeHydrology));
        return plan;
    };

    constexpr FarTerrainKey POSITIVE{1, 1, FAR_TERRAIN_BASE_STEP};
    const FarTerrainFinalBaseAuthorityDependencies positive = verify(POSITIVE);
    REQUIRE(positive.transientGeometryRegion);
    REQUIRE(positive.transientGeometryRegion->valid());
    FarTerrainViewTile tile;
    tile.key = POSITIVE;
    tile.bounds = {.minX = 256, .maxX = 512, .minZ = 256, .maxZ = 512};
    const std::array selected{tile};
    const auto topologyClosure = farTerrainCoarseAuthorityPages(selected, 384.0, 384.0);
    REQUIRE(topologyClosure.size() > positive.geometryPages.size());
    for (const TerrainPageCoordinate page : positive.geometryPages)
        REQUIRE(std::ranges::find(topologyClosure, page) != topologyClosure.end());

    const FarTerrainFinalBaseAuthorityDependencies negative =
        verify({-2, -2, FAR_TERRAIN_BASE_STEP});
    REQUIRE(negative.transientGeometryRegion);
    REQUIRE(negative.minimumWorldX == -544);
    REQUIRE(negative.minimumWorldZ == -544);
    REQUIRE(negative.maximumWorldXExclusive == -223);
    REQUIRE(negative.maximumWorldZExclusive == -223);

    REQUIRE_THROWS_AS(farTerrainFinalBaseAuthorityDependencies(
                          {std::numeric_limits<int64_t>::max(), 0, FAR_TERRAIN_BASE_STEP}),
                      std::out_of_range);
    REQUIRE_THROWS_AS(farTerrainFinalBaseAuthorityDependencies({0, 0, FarTerrainStep::SIXTEEN}),
                      std::invalid_argument);
}

TEST_CASE("Movement prefetch selects a deterministic bounded row beyond visible authority",
          "[render][far-terrain][authority][prefetch][priority][regression]") {
    using worldgen::learned::TerrainPageCoordinate;
    std::vector<TerrainPageCoordinate> visible;
    for (int64_t row = -1; row <= 1; ++row) {
        for (int64_t column = -1; column <= 1; ++column)
            visible.push_back({.row = row, .column = column});
    }
    const std::set<TerrainPageCoordinate> visibleSet(visible.begin(), visible.end());

    const auto east = farTerrainSpeculativeAuthorityPages(visible, 496.0, 512.0, 512.0, 512.0);
    REQUIRE_FALSE(east.empty());
    REQUIRE(east.size() <= FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES);
    REQUIRE((east.front() == TerrainPageCoordinate{.row = 0, .column = 2}));
    REQUIRE(std::ranges::none_of(
        east, [&](TerrainPageCoordinate coordinate) { return visibleSet.contains(coordinate); }));
    REQUIRE(std::ranges::all_of(
        east, [](TerrainPageCoordinate coordinate) { return coordinate.column == 2; }));
    REQUIRE(farTerrainSpeculativeAuthorityPages(visible, 496.0, 512.0, 512.0, 512.0) == east);

    const auto west = farTerrainSpeculativeAuthorityPages(visible, 528.0, 512.0, 512.0, 512.0);
    REQUIRE_FALSE(west.empty());
    REQUIRE(west.size() <= FAR_TERRAIN_MAX_SPECULATIVE_AUTHORITY_PAGES);
    REQUIRE((west.front() == TerrainPageCoordinate{.row = 0, .column = -2}));
    REQUIRE(std::ranges::all_of(
        west, [](TerrainPageCoordinate coordinate) { return coordinate.column == -2; }));
    REQUIRE(farTerrainSpeculativeAuthorityPages(visible, 512.0, 512.0, 512.0, 512.0).empty());
    REQUIRE(farTerrainSpeculativeAuthorityPages(visible, std::numeric_limits<double>::quiet_NaN(),
                                                0.0, 1.0, 0.0)
                .empty());
}

TEST_CASE("Production far authority uses protected handoff and deferred movement lanes",
          "[render][far-terrain][scheduler][authority][prefetch][priority][regression]") {
    using worldgen::learned::AuthorityRequestPriority;
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey HANDOFF{0, 0, FAR_TERRAIN_BASE_STEP};
    const uint64_t handoffEpoch = scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueFinalBase(HANDOFF, 0, true));
    REQUIRE(authority->prepareCalls(AuthorityRequestPriority::PROTECTED_HANDOFF) > 0);
    REQUIRE(authority->prepareCalls(AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT) == 0);
    REQUIRE(authority->latestProtectedHandoffEpoch() == handoffEpoch);

    scheduler.setCoarseAuthorityPrefetchPages({{.row = 20, .column = 20}});
    scheduler.setSpeculativeAuthorityPrefetchPages({{.row = 21, .column = 20}});
    scheduler.pumpCoarseAuthorityPrefetch();
    scheduler.pumpSpeculativeAuthorityPrefetch();
    REQUIRE(authority->prepareCalls(AuthorityRequestPriority::COARSE_PREVIEW) > 0);
    REQUIRE(authority->prepareCalls(AuthorityRequestPriority::SPECULATIVE_PREFETCH) == 0);

    authority->setReady();
    scheduler.pumpCoarseAuthorityPrefetch();
    scheduler.pumpSpeculativeAuthorityPrefetch();
    REQUIRE(authority->prepareCalls(AuthorityRequestPriority::SPECULATIVE_PREFETCH) > 0);
    scheduler.shutdown();
}

TEST_CASE("Current protected handoff jobs pass stale camera epochs in the far queue",
          "[render][far-terrain][scheduler][authority][priority][handoff-epoch][movement]"
          "[regression]") {
    constexpr FarTerrainKey BLOCKER{700, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey STALE{1, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey CURRENT{2, 0, FAR_TERRAIN_BASE_STEP};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    std::vector<int64_t> started;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            std::unique_lock lock(gateMutex);
            if ((tileX == BLOCKER.tileX || tileX == STALE.tileX || tileX == CURRENT.tileX) &&
                std::ranges::find(started, tileX) == started.end()) {
                started.push_back(tileX);
            }
            if (tileX == BLOCKER.tileX && !blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };

    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::vector order{BLOCKER, STALE, CURRENT};
    REQUIRE(scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order));
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }

    const uint64_t staleEpoch = scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueFinalBase(STALE, 0, true));
    const uint64_t currentEpoch = scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(currentEpoch > staleEpoch);
    REQUIRE(scheduler.enqueueFinalBase(CURRENT, 100, true));
    REQUIRE(authority->latestProtectedHandoffEpoch() == currentEpoch);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    {
        std::lock_guard lock(gateMutex);
        REQUIRE(started.size() >= 2);
        REQUIRE(started[1] == CURRENT.tileX);
    }
    scheduler.shutdown();
}

TEST_CASE("Coarse authority prefetch observes failed flights after a horizon replacement",
          "[render][far-terrain][coverage][authority][startup][failure][regression]") {
    TempDir directory("coarse_authority_prefetch_failure");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto backend = std::make_shared<CoarsePrefetchFailingBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(), backend);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainScheduler scheduler(identity.seed, context);

    std::vector<worldgen::learned::TerrainPageCoordinate> firstHorizon;
    firstHorizon.reserve(worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS);
    for (int64_t row = 0;
         row < static_cast<int64_t>(worldgen::learned::MAXIMUM_AUTHORITY_QUEUED_REQUESTS); ++row) {
        firstHorizon.push_back({.row = row, .column = 0});
    }
    scheduler.setCoarseAuthorityPrefetchPages(std::move(firstHorizon));
    scheduler.pumpCoarseAuthorityPrefetch();

    const auto backendDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (backend->callCount() == 0 && std::chrono::steady_clock::now() < backendDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(backend->callCount() != 0);

    // This is the failure mode from the cold-horizon stall: the old horizon
    // has filled the authority queue, then a camera update asks for a page
    // outside it. The replacement cannot be admitted until the scheduler
    // polls the terminal flights it already submitted.
    scheduler.setCoarseAuthorityPrefetchPages({{.row = 64, .column = 0}});
    const auto failureDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!context->failure() && std::chrono::steady_clock::now() < failureDeadline) {
        scheduler.pumpCoarseAuthorityPrefetch();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const auto failure = context->failure();
    REQUIRE(failure.has_value());
    REQUIRE(failure->code == worldgen::learned::GenerationFailureCode::INFERENCE_FAILED);
}

TEST_CASE("Cold protected refinement queues adjacent preview bridges before final targets",
          "[render][far-terrain][coverage][lod][priority][cold-start][regression]") {
    constexpr ColumnPos CAMERA{0, 0};
    constexpr ColumnPos NEAR_A{1, 0};
    constexpr ColumnPos NEAR_B{0, 1};
    constexpr ColumnPos DISTANT{8, 0};
    constexpr ColumnPos TRANSITIONING{2, 0};
    std::array<FarTerrainRefinementCacheRequest, 5> requests{{
        {CAMERA, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {NEAR_A, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {NEAR_B, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO},
        {DISTANT, FarTerrainStep::THIRTY_TWO, FarTerrainStep::SIXTEEN},
        {TRANSITIONING, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, 0, true},
    }};
    for (size_t index = 0; index < 3; ++index)
        requests[index].protectedNearTarget = true;
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    const std::vector<FarTerrainKey> expected = {
        {CAMERA.x, CAMERA.z, FarTerrainStep::SIXTEEN},
        {NEAR_B.x, NEAR_B.z, FarTerrainStep::SIXTEEN},
        {NEAR_A.x, NEAR_A.z, FarTerrainStep::SIXTEEN},
        {DISTANT.x, DISTANT.z, FarTerrainStep::SIXTEEN},
    };
    REQUIRE(order == expected);
    REQUIRE(std::ranges::none_of(order, [](FarTerrainKey key) {
        return key.tileX == TRANSITIONING.x && key.tileZ == TRANSITIONING.z;
    }));

    auto oneSlot = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                oneSlot, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS - 1) == 1);
    for (size_t index = 0; index < 4; ++index)
        REQUIRE_FALSE(oneSlot[index].deferIntermediate);

    auto noSlots = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                noSlots, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS) == 0);
    for (size_t index = 0; index < 3; ++index)
        REQUIRE_FALSE(noSlots[index].deferIntermediate);
    REQUIRE(noSlots[3].deferIntermediate);

    std::array<FarTerrainRefinementCacheRequest, 3> authoritySaturated{{
        {.coordinate = CAMERA,
         .displayed = FarTerrainStep::THIRTY_TWO,
         .desired = FarTerrainStep::TWO,
         .requiresFineFallback = true,
         .cameraTile = true},
        {.coordinate = NEAR_A,
         .displayed = FarTerrainStep::THIRTY_TWO,
         .desired = FarTerrainStep::TWO,
         .protectedNearTarget = true},
        {.coordinate = DISTANT,
         .displayed = FarTerrainStep::THIRTY_TWO,
         .desired = FarTerrainStep::SIXTEEN},
    }};
    constexpr size_t ACTIVE_LOD_TRANSITIONS = FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS - 3;
    constexpr size_t ACTIVE_AUTHORITY_TRANSITIONS = 3;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                authoritySaturated, ACTIVE_LOD_TRANSITIONS + ACTIVE_AUTHORITY_TRANSITIONS) == 0);
    REQUIRE_FALSE(authoritySaturated[0].deferIntermediate);
    REQUIRE_FALSE(authoritySaturated[1].deferIntermediate);
    REQUIRE(authoritySaturated[2].deferIntermediate);
}

TEST_CASE("Refinement priority is deterministic across screen error and negative coordinates",
          "[render][far-terrain][lod][priority][screen-error][negative][regression]") {
    const std::array requests{
        FarTerrainRefinementCacheRequest{
            .coordinate = {-2, -7},
            .displayed = FarTerrainStep::THIRTY_TWO,
            .desired = FarTerrainStep::ONE,
            .projectedErrorPixels = 2.0,
            .distanceSquaredBlocks = 100.0,
        },
        FarTerrainRefinementCacheRequest{
            .coordinate = {-8, 3},
            .displayed = FarTerrainStep::THIRTY_TWO,
            .desired = FarTerrainStep::ONE,
            .projectedErrorPixels = 8.0,
            .distanceSquaredBlocks = 400.0,
        },
        FarTerrainRefinementCacheRequest{
            .coordinate = {4, -5},
            .displayed = FarTerrainStep::THIRTY_TWO,
            .desired = FarTerrainStep::ONE,
            .requiresFineFallback = true,
            .requiresBlockScaleFallback = true,
            .projectedErrorPixels = 1.0,
            .distanceSquaredBlocks = 900.0,
        },
        FarTerrainRefinementCacheRequest{
            .coordinate = {-11, -13},
            .displayed = FarTerrainStep::THIRTY_TWO,
            .desired = FarTerrainStep::ONE,
            .cameraTile = true,
            .projectedErrorPixels = 0.5,
            .distanceSquaredBlocks = 0.0,
        },
    };
    auto protectedRequests = requests;
    for (auto& request : protectedRequests)
        request.protectedNearTarget = true;
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(protectedRequests, order);
    const std::vector expected{
        FarTerrainKey{-11, -13, FarTerrainStep::SIXTEEN},
        FarTerrainKey{4, -5, FarTerrainStep::SIXTEEN},
        FarTerrainKey{-2, -7, FarTerrainStep::SIXTEEN},
        FarTerrainKey{-8, 3, FarTerrainStep::SIXTEEN},
    };
    REQUIRE(order == expected);

    for (int frame = 0; frame < 64; ++frame) {
        std::vector<FarTerrainKey> repeated;
        buildFarTerrainProgressiveSubmissionOrder(protectedRequests, repeated);
        REQUIRE(repeated == expected);
    }

    auto visibleRequests = requests;
    for (FarTerrainRefinementCacheRequest& request : visibleRequests) {
        request.cameraTile = false;
        request.protectedNearTarget = false;
        request.requiresFineFallback = false;
        request.requiresBlockScaleFallback = false;
    }
    visibleRequests[0].visible = true;
    visibleRequests[0].projectedErrorPixels = 1.0;
    visibleRequests[1].projectedErrorPixels = 100.0;
    buildFarTerrainProgressiveSubmissionOrder(visibleRequests, order, 2);
    REQUIRE((order == std::vector<FarTerrainKey>{
                          {-11, -13, FarTerrainStep::SIXTEEN},
                          {-2, -7, FarTerrainStep::SIXTEEN},
                      }));

    visibleRequests[0].visible = false;
    visibleRequests[1].displayableWavefront = false;
    buildFarTerrainProgressiveSubmissionOrder(visibleRequests, order, 2);
    REQUIRE((order == std::vector<FarTerrainKey>{
                          {-11, -13, FarTerrainStep::SIXTEEN},
                          {-2, -7, FarTerrainStep::SIXTEEN},
                      }));

    auto oneOrdinarySlot = requests;
    oneOrdinarySlot[2].requiresFineFallback = false;
    oneOrdinarySlot[2].requiresBlockScaleFallback = false;
    oneOrdinarySlot[3].cameraTile = false;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                oneOrdinarySlot, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS - 1) == 1);
    REQUIRE(oneOrdinarySlot[0].deferIntermediate);
    REQUIRE(oneOrdinarySlot[1].deferIntermediate);
    REQUIRE(oneOrdinarySlot[2].deferIntermediate);
    REQUIRE_FALSE(oneOrdinarySlot[3].deferIntermediate);
}

TEST_CASE("Full horizon refinement ranking is bounded without output growth",
          "[render][far-terrain][lod][priority][performance][regression]") {
    constexpr size_t HORIZON_CANDIDATES = 3336;
    std::vector<FarTerrainRefinementCacheRequest> requests(HORIZON_CANDIDATES);
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index] = {
            .coordinate = {static_cast<int64_t>(index), -7},
            .displayed = FarTerrainStep::THIRTY_TWO,
            .desired = FarTerrainStep::ONE,
            .projectedErrorPixels = static_cast<double>(index),
            .distanceSquaredBlocks = static_cast<double>(index * index),
        };
    }

    std::vector<FarTerrainKey> order;
    order.reserve(FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
    const size_t reservedCapacity = order.capacity();
    for (size_t frame = 0; frame < 32; ++frame) {
        buildFarTerrainProgressiveSubmissionOrder(requests, order,
                                                  FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
        REQUIRE(order.size() == FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
        REQUIRE(order.capacity() == reservedCapacity);
        REQUIRE(order.front().tileX == 0);
        REQUIRE(order.back().tileX ==
                static_cast<int64_t>(FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS - 1));
        REQUIRE(std::ranges::all_of(
            order, [](FarTerrainKey key) { return key.step == FarTerrainStep::SIXTEEN; }));
    }

    const size_t reserved = reserveFarTerrainIntermediateTransitionSlots(requests, 0);
    REQUIRE(reserved == FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS);
    for (size_t index = 0; index < requests.size(); ++index) {
        const bool selected = index < FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS;
        CAPTURE(index);
        REQUIRE(requests[index].deferIntermediate != selected);
    }
}

TEST_CASE("Far terrain parent lane outranks every refinement priority",
          "[render][far-terrain][coverage][scheduler][priority]") {
    const FarTerrainKey parent{12, -7, FAR_TERRAIN_BASE_STEP};
    const FarTerrainKey nearTarget{0, 0, FarTerrainStep::TWO};
    const FarTerrainKey fartherTarget{3, 4, FarTerrainStep::EIGHT};

    REQUIRE(farTerrainSubmissionBefore(parent, 10'000, nearTarget, 0));
    REQUIRE_FALSE(farTerrainSubmissionBefore(nearTarget, 0, parent, 10'000));
    REQUIRE(farTerrainSubmissionBefore(nearTarget, 4, fartherTarget, 8));

    FarTerrainCoverageFrontier frontier;
    frontier.complete = false;
    frontier.missingBaseTiles = 1;
    REQUIRE_FALSE(farTerrainRefinementLaneOpen(frontier, true));
    frontier.complete = true;
    REQUIRE_FALSE(farTerrainRefinementLaneOpen(frontier, true));
    frontier.missingBaseTiles = 0;
    REQUIRE_FALSE(farTerrainRefinementLaneOpen(frontier, false));
    REQUIRE(farTerrainRefinementLaneOpen(frontier, true));
}

TEST_CASE("Urgent nearby refinement honors parent reservation within an eight-worker budget",
          "[render][far-terrain][scheduler][priority][cold-start][camera-jump][performance]"
          "[regression]") {
    constexpr std::array<FarTerrainKey, 10> BASES{{
        {100, 0, FarTerrainStep::THIRTY_TWO},
        {200, 0, FarTerrainStep::THIRTY_TWO},
        {300, 0, FarTerrainStep::THIRTY_TWO},
        {400, 0, FarTerrainStep::THIRTY_TWO},
        {500, 0, FarTerrainStep::THIRTY_TWO},
        {600, 0, FarTerrainStep::THIRTY_TWO},
        {700, 0, FarTerrainStep::THIRTY_TWO},
        {800, 0, FarTerrainStep::THIRTY_TWO},
        {900, 0, FarTerrainStep::THIRTY_TWO},
        {1'000, 0, FarTerrainStep::THIRTY_TWO},
    }};
    constexpr std::array<FarTerrainKey, 5> URGENT{{
        {1'100, 0, FarTerrainStep::TWO},
        {1'200, 0, FarTerrainStep::FOUR},
        {1'300, 0, FarTerrainStep::EIGHT},
        {1'400, 0, FarTerrainStep::SIXTEEN},
        {1'500, 0, FarTerrainStep::TWO},
    }};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    std::unordered_set<int64_t> started;
    bool releaseInitialBases = false;
    bool releaseUrgent = false;

    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            const bool known =
                std::ranges::any_of(BASES, [&](FarTerrainKey key) { return key.tileX == tileX; }) ||
                std::ranges::any_of(URGENT, [&](FarTerrainKey key) { return key.tileX == tileX; });
            if (known) {
                std::unique_lock lock(gateMutex);
                if (started.insert(tileX).second) {
                    gateCv.notify_all();
                    if (std::find_if(BASES.begin(), BASES.begin() + 8, [&](FarTerrainKey key) {
                            return key.tileX == tileX;
                        }) != BASES.begin() + 8) {
                        gateCv.wait(lock, [&] { return releaseInitialBases; });
                    } else if (std::find_if(URGENT.begin(), URGENT.begin() + 4,
                                            [&](FarTerrainKey key) {
                                                return key.tileX == tileX;
                                            }) != URGENT.begin() + 4) {
                        gateCv.wait(lock, [&] { return releaseUrgent; });
                    }
                }
            }
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 16;
    limits.maxCompleted = 16;
    limits.maxCacheEntries = 16;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct GateRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& releaseInitial;
        bool& releaseUrgent;
        ~GateRelease() {
            {
                std::lock_guard lock(mutex);
                releaseInitial = true;
                releaseUrgent = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseInitialBases, releaseUrgent};
    constexpr size_t TEST_WORKER_BUDGET = 8;
    scheduler.setWorkerBudget(TEST_WORKER_BUDGET);
    for (const FarTerrainKey base : BASES)
        REQUIRE(scheduler.enqueue(base));

    bool initialBasesStarted = false;
    {
        std::unique_lock lock(gateMutex);
        initialBasesStarted = gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            return std::all_of(BASES.begin(), BASES.begin() + 8,
                               [&](FarTerrainKey key) { return started.contains(key.tileX); });
        });
    }
    if (!initialBasesStarted) {
        {
            std::lock_guard lock(gateMutex);
            releaseInitialBases = true;
            releaseUrgent = true;
        }
        gateCv.notify_all();
        scheduler.shutdown();
    }
    REQUIRE(initialBasesStarted);

    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[0], 0));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[1], 1));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[2], 2));
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[3], 3));
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(URGENT[4], 4));
    REQUIRE_FALSE(scheduler.enqueueUrgentRefinement(BASES[0], 0));
    {
        const FarTerrainSchedulerStats queued = scheduler.stats();
        REQUIRE(queued.urgentRefinementInFlight == URGENT.size());
        REQUIRE(queued.queuedUrgentRefinement == URGENT.size());
        REQUIRE(queued.queuedBase >= 2);
        REQUIRE(queued.activeBaseWorkers == 8);
        REQUIRE(queued.reservedBaseWorkers == 4);
        REQUIRE(queued.activeUrgentRefinement == 0);
        REQUIRE(queued.workerBudget == TEST_WORKER_BUDGET);
    }

    {
        std::lock_guard lock(gateMutex);
        releaseInitialBases = true;
    }
    gateCv.notify_all();

    bool nearbyAndParentAdvancedTogether = false;
    {
        std::unique_lock lock(gateMutex);
        nearbyAndParentAdvancedTogether = gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            const bool urgentStarted =
                std::all_of(URGENT.begin(), URGENT.begin() + 4,
                            [&](FarTerrainKey key) { return started.contains(key.tileX); });
            const bool nextBaseStarted =
                started.contains(BASES[8].tileX) || started.contains(BASES[9].tileX);
            return urgentStarted && nextBaseStarted;
        });
        releaseUrgent = true;
    }
    gateCv.notify_all();
    REQUIRE(nearbyAndParentAdvancedTogether);

    for (int attempt = 0; attempt < 500 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const FarTerrainSchedulerStats finished = scheduler.stats();
    scheduler.shutdown();
    REQUIRE(finished.inFlight == 0);
    REQUIRE(finished.urgentRefinementInFlight == 0);
    REQUIRE(finished.activeBaseWorkers == 0);
    REQUIRE(finished.reservedBaseWorkers == 0);
    REQUIRE(finished.activeUrgentRefinement == 0);
    REQUIRE(std::all_of(URGENT.begin(), URGENT.begin() + 4,
                        [&](FarTerrainKey key) { return started.contains(key.tileX); }));
    REQUIRE(started.contains(URGENT[4].tileX));
    REQUIRE(started.contains(BASES[8].tileX));
    REQUIRE(started.contains(BASES[9].tileX));
}

TEST_CASE("Gameplay near-first mode drains desired LODs before queued horizon parents",
          "[render][far-terrain][scheduler][lod][priority][gameplay][regression]") {
    constexpr std::array<FarTerrainKey, 6> BASES{{
        {100, 0, FarTerrainStep::THIRTY_TWO},
        {200, 0, FarTerrainStep::THIRTY_TWO},
        {300, 0, FarTerrainStep::THIRTY_TWO},
        {400, 0, FarTerrainStep::THIRTY_TWO},
        {500, 0, FarTerrainStep::THIRTY_TWO},
        {600, 0, FarTerrainStep::THIRTY_TWO},
    }};
    constexpr std::array<FarTerrainKey, 4> NEAR{{
        {1, 0, FarTerrainStep::SIXTEEN},
        {2, 0, FarTerrainStep::SIXTEEN},
        {3, 0, FarTerrainStep::SIXTEEN},
        {4, 0, FarTerrainStep::SIXTEEN},
    }};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    std::unordered_set<int64_t> started;
    bool releaseInitial = false;
    bool releaseNear = false;

    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            const bool initial =
                std::find_if(BASES.begin(), BASES.begin() + 4, [&](FarTerrainKey key) {
                    return key.tileX == tileX;
                }) != BASES.begin() + 4;
            const bool base =
                std::ranges::any_of(BASES, [&](FarTerrainKey key) { return key.tileX == tileX; });
            const bool near =
                std::ranges::any_of(NEAR, [&](FarTerrainKey key) { return key.tileX == tileX; });
            if (base || near) {
                std::unique_lock lock(gateMutex);
                if (started.insert(tileX).second) {
                    gateCv.notify_all();
                    if (initial)
                        gateCv.wait(lock, [&] { return releaseInitial; });
                    else
                        gateCv.wait(lock, [&] { return releaseNear; });
                }
            }
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 12;
    limits.maxCompleted = 12;
    limits.maxCacheEntries = 12;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct GateRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& initial;
        bool& near;
        ~GateRelease() {
            {
                std::lock_guard lock(mutex);
                initial = true;
                near = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseInitial, releaseNear};

    scheduler.setWorkerBudget(4);
    for (const FarTerrainKey base : BASES)
        REQUIRE(scheduler.enqueue(base));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            return std::all_of(BASES.begin(), BASES.begin() + 4,
                               [&](FarTerrainKey key) { return started.contains(key.tileX); });
        }));
    }

    scheduler.setNearFirstWorkEnabled(true);
    for (size_t index = 0; index < NEAR.size(); ++index)
        REQUIRE(scheduler.enqueueUrgentRefinement(NEAR[index], static_cast<uint32_t>(index)));
    {
        std::lock_guard lock(gateMutex);
        releaseInitial = true;
    }
    gateCv.notify_all();

    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            return std::ranges::all_of(
                NEAR, [&](FarTerrainKey key) { return started.contains(key.tileX); });
        }));
        CHECK_FALSE(started.contains(BASES[4].tileX));
        CHECK_FALSE(started.contains(BASES[5].tileX));
        releaseNear = true;
    }
    gateCv.notify_all();
    for (int attempt = 0; attempt < 500 && scheduler.stats().inFlight != 0; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const FarTerrainSchedulerStats finished = scheduler.stats();
    scheduler.shutdown();
    REQUIRE(finished.inFlight == 0);
    REQUIRE(started.contains(BASES[4].tileX));
    REQUIRE(started.contains(BASES[5].tileX));
}

TEST_CASE("Protected cold step one schedules adjacent preview bridges first",
          "[render][far-terrain][scheduler][lod][cold-start][priority][regression]") {
    constexpr ColumnPos CAMERA{0, 0};
    FarTerrainRefinementCacheRequest request{
        CAMERA,
        FarTerrainStep::THIRTY_TWO,
        FarTerrainStep::ONE,
        farTerrainStepMask(FarTerrainStep::THIRTY_TWO),
    };
    request.protectedNearTarget = true;
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(std::span(&request, 1), order);
    REQUIRE((order == std::vector<FarTerrainKey>{
                          {CAMERA.x, CAMERA.z, FarTerrainStep::SIXTEEN},
                      }));
    request.residentSteps |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    request.displayed = FarTerrainStep::SIXTEEN;
    buildFarTerrainProgressiveSubmissionOrder(std::span(&request, 1), order);
    REQUIRE((order == std::vector<FarTerrainKey>{
                          {CAMERA.x, CAMERA.z, FarTerrainStep::EIGHT},
                      }));

    request.displayed = FarTerrainStep::THIRTY_TWO;
    request.residentSteps = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    request.transitionActive = true;
    buildFarTerrainProgressiveSubmissionOrder(std::span(&request, 1), order);
    REQUIRE(order.empty());
}

TEST_CASE("Bounded refinement capacity admits every nearer bridge before final targets",
          "[render][far-terrain][scheduler][lod][capacity][priority][regression]") {
    constexpr size_t CANDIDATE_COUNT = 96;
    constexpr size_t QUEUE_CAPACITY = 64;
    std::array<FarTerrainRefinementCacheRequest, CANDIDATE_COUNT> requests{};
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index] = {
            {static_cast<int64_t>(index), 0},
            FarTerrainStep::THIRTY_TWO,
            FarTerrainStep::ONE,
            farTerrainStepMask(FarTerrainStep::THIRTY_TWO),
        };
    }

    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    REQUIRE(order.size() == CANDIDATE_COUNT);
    for (size_t index = 0; index < QUEUE_CAPACITY; ++index) {
        CAPTURE(index, order[index].tileX, static_cast<int>(order[index].step));
        REQUIRE((order[index] ==
                 FarTerrainKey{static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN}));
    }

    requests.front().residentSteps |= farTerrainStepMask(FarTerrainStep::TWO);
    requests.front().residentSteps |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    requests.front().displayed = FarTerrainStep::SIXTEEN;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    REQUIRE((order.front() == FarTerrainKey{0, 0, FarTerrainStep::EIGHT}));
    REQUIRE((order[1] == FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}));
}

TEST_CASE("Camera-near refinement displaces distant queued work at the pending cap",
          "[render][far-terrain][scheduler][lod][capacity][priority][movement][regression]") {
    constexpr FarTerrainKey BLOCKER{800, 0, FAR_TERRAIN_BASE_STEP};
    constexpr std::array DISTANT{
        FarTerrainKey{1'000, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{1'200, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{1'400, 0, FarTerrainStep::SIXTEEN},
    };
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    std::vector<int64_t> started;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            std::unique_lock lock(gateMutex);
            const bool tracked =
                tileX == BLOCKER.tileX || tileX == NEAR.tileX ||
                std::ranges::any_of(DISTANT, [&](FarTerrainKey key) { return key.tileX == tileX; });
            if (tracked && std::ranges::find(started, tileX) == started.end())
                started.push_back(tileX);
            if (tileX == BLOCKER.tileX && !blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::vector<FarTerrainKey> order{BLOCKER, DISTANT[0], DISTANT[1], DISTANT[2], NEAR};
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    for (size_t index = 0; index < DISTANT.size(); ++index)
        REQUIRE(scheduler.enqueue(DISTANT[index], static_cast<uint32_t>(100 + index)));
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR, 0, true));
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.stats().canceled == 1);
    REQUIRE(scheduler.stats().criticalDisplacements == 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE_FALSE(scheduler.findCached(DISTANT.back()));
    {
        std::lock_guard lock(gateMutex);
        REQUIRE(started.size() >= 2);
        REQUIRE(started[1] == NEAR.tileX);
    }
    scheduler.shutdown();
}

TEST_CASE("Protected final parent displaces distant queued refinement at the pending cap",
          "[render][far-terrain][scheduler][authority][capacity][priority][regression]") {
    constexpr FarTerrainKey BLOCKER{810, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey DISTANT{1'800, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey PROTECTED{0, 0, FAR_TERRAIN_BASE_STEP};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::vector order{BLOCKER, PROTECTED, DISTANT};
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    REQUIRE(scheduler.enqueueUrgentRefinement(DISTANT, 100));
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueFinalBase(PROTECTED, 0, true));
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.stats().canceled == 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    scheduler.shutdown();
    REQUIRE_FALSE(scheduler.findCached(DISTANT));
}

TEST_CASE("Camera movement replaces a saturated stale protected closure",
          "[render][far-terrain][scheduler][authority][handoff-epoch][capacity][movement]"
          "[priority][regression]") {
    constexpr FarTerrainKey BLOCKER{820, 0, FAR_TERRAIN_BASE_STEP};
    constexpr auto OLD_CLOSURE = [] {
        std::array<FarTerrainKey, FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT> keys{};
        for (size_t index = 0; index < keys.size(); ++index)
            keys[index] = {static_cast<int64_t>(1'000 + index), 0, FarTerrainStep::SIXTEEN};
        return keys;
    }();
    constexpr auto CURRENT_CLOSURE = [] {
        std::array<FarTerrainKey, FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT> keys{};
        for (size_t index = 0; index < keys.size(); ++index)
            keys[index] = {static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN};
        return keys;
    }();
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT + 1;
    limits.maxCompleted = 32;
    limits.maxCacheEntries = 32;
    limits.maxCacheBytes = 128 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    std::vector<FarTerrainKey> order{BLOCKER};
    order.insert(order.end(), CURRENT_CLOSURE.begin(), CURRENT_CLOSURE.end());
    order.insert(order.end(), OLD_CLOSURE.begin(), OLD_CLOSURE.end());
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    scheduler.advanceProtectedHandoffEpoch();
    for (size_t index = 0; index < OLD_CLOSURE.size(); ++index) {
        REQUIRE(scheduler.enqueueUrgentFinalRefinement(OLD_CLOSURE[index],
                                                       static_cast<uint32_t>(100 + index), true));
    }
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == OLD_CLOSURE.size());

    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.stats().inFlight == 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.stats().canceled == OLD_CLOSURE.size());
    REQUIRE(scheduler.stats().criticalDisplacements == OLD_CLOSURE.size());
    for (size_t index = 0; index < CURRENT_CLOSURE.size(); ++index) {
        REQUIRE(scheduler.enqueueUrgentFinalRefinement(CURRENT_CLOSURE[index],
                                                       static_cast<uint32_t>(index), true));
    }
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == CURRENT_CLOSURE.size());
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    for (const FarTerrainKey key : CURRENT_CLOSURE)
        REQUIRE(scheduler.findCached(key));
    for (const FarTerrainKey key : OLD_CLOSURE)
        REQUIRE_FALSE(scheduler.findCached(key));
    scheduler.shutdown();
}

TEST_CASE("Camera movement reuses protected work shared by consecutive closures",
          "[render][far-terrain][scheduler][authority][handoff-epoch][movement][overlap]"
          "[priority][regression]") {
    constexpr FarTerrainKey BLOCKER{825, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey OVERLAP{1, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey OLD_ONLY{-6, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::vector order{BLOCKER, OVERLAP, OLD_ONLY};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    scheduler.retainWanted(wanted, order);
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }

    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(OVERLAP, 0, true));
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(OLD_ONLY, 1, true));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 2);

    const std::array currentCritical{OVERLAP};
    REQUIRE(scheduler.retainWanted(wanted, order, currentCritical));
    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.stats().canceled == 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);

    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(OVERLAP));
    REQUIRE_FALSE(scheduler.findCached(OLD_ONLY));
    scheduler.shutdown();
}

TEST_CASE("Directional prediction cancels on reversal and reuses on handoff",
          "[render][far-terrain][scheduler][authority][handoff-epoch][movement][prediction]"
          "[priority][regression]") {
    constexpr FarTerrainKey CURRENT{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey EAST_PREDICTED{1, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey WEST_PREDICTED{-1, 0, FarTerrainStep::SIXTEEN};
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setWorkerBudget(0);

    std::vector<FarTerrainKey> order{CURRENT, EAST_PREDICTED};
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    REQUIRE(scheduler.retainWanted(wanted, order, order));
    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(CURRENT, 0, true));
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(EAST_PREDICTED, 1, true));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 2);

    const size_t canceledBeforeReversal = scheduler.stats().canceled;
    order = {CURRENT, WEST_PREDICTED};
    wanted = std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end());
    REQUIRE(scheduler.retainWanted(wanted, order, order));
    REQUIRE(scheduler.stats().canceled == canceledBeforeReversal + 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(WEST_PREDICTED, 1, true));

    order = {WEST_PREDICTED};
    wanted = {WEST_PREDICTED};
    REQUIRE(scheduler.retainWanted(wanted, order, order));
    const size_t canceledBeforeEpochAdvance = scheduler.stats().canceled;
    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.stats().canceled == canceledBeforeEpochAdvance);

    scheduler.setWorkerBudget(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(WEST_PREDICTED));
    REQUIRE_FALSE(scheduler.findCached(EAST_PREDICTED));
    scheduler.shutdown();
}

TEST_CASE("Camera-near refinement replaces the farthest queued urgent request",
          "[render][far-terrain][scheduler][lod][capacity][urgent][priority][regression]") {
    constexpr FarTerrainKey BLOCKER{900, 0, FAR_TERRAIN_BASE_STEP};
    constexpr auto DISTANT = [] {
        std::array<FarTerrainKey, FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT> keys{};
        for (size_t index = 0; index < keys.size(); ++index) {
            keys[index] = {static_cast<int64_t>(1'000 + index), 0, FarTerrainStep::SIXTEEN};
        }
        return keys;
    }();
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = DISTANT.size() + 1;
    limits.maxCompleted = 32;
    limits.maxCacheEntries = 32;
    limits.maxCacheBytes = 128 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    std::vector<FarTerrainKey> order{BLOCKER};
    order.insert(order.end(), DISTANT.begin(), DISTANT.end());
    order.push_back(NEAR);
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    for (size_t index = 0; index < DISTANT.size(); ++index) {
        REQUIRE(
            scheduler.enqueueUrgentRefinement(DISTANT[index], static_cast<uint32_t>(100 + index)));
    }
    REQUIRE(scheduler.stats().urgentRefinementInFlight == DISTANT.size());
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR, 0, true));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == DISTANT.size());
    REQUIRE(scheduler.stats().canceled == 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE_FALSE(scheduler.findCached(DISTANT.back()));
    scheduler.shutdown();
}

TEST_CASE("Camera-near refinement displaces a distant parked final parent",
          "[render][far-terrain][scheduler][lod][capacity][deferred][priority][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey FAR_PARENT{2'000, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::SIXTEEN};
    const std::vector order{NEAR, FAR_PARENT};
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);
    REQUIRE(scheduler.enqueueFinalBase(FAR_PARENT, 100, true));
    REQUIRE(scheduler.stats().parkedBase == 1);
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR, 0, true));
    REQUIRE(scheduler.stats().parkedBase == 0);
    REQUIRE(scheduler.stats().canceled == 1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE_FALSE(scheduler.findCached(FAR_PARENT));
    scheduler.shutdown();
}

TEST_CASE("Retained terrain work follows the latest camera order",
          "[render][far-terrain][scheduler][lod][priority][movement][regression]") {
    constexpr FarTerrainKey BLOCKER{1'200, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FIRST{1'400, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey SECOND{1'600, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    std::vector<int64_t> started;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            std::unique_lock lock(gateMutex);
            const bool tracked =
                tileX == BLOCKER.tileX || tileX == FIRST.tileX || tileX == SECOND.tileX;
            if (tracked && std::ranges::find(started, tileX) == started.end())
                started.push_back(tileX);
            if (tileX == BLOCKER.tileX && !blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted{BLOCKER, FIRST, SECOND};
    REQUIRE(scheduler.retainWanted(wanted, {BLOCKER, FIRST, SECOND}));
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    REQUIRE(scheduler.enqueueUrgentRefinement(FIRST));
    REQUIRE(scheduler.enqueueUrgentRefinement(SECOND));
    REQUIRE(scheduler.retainWanted(wanted, {BLOCKER, SECOND, FIRST}));
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    {
        std::lock_guard lock(gateMutex);
        REQUIRE(started.size() >= 3);
        REQUIRE(started[1] == SECOND.tileX);
        REQUIRE(started[2] == FIRST.tileX);
    }
    scheduler.shutdown();
}

TEST_CASE("Camera-near cache insertion evicts the farthest optional refinement",
          "[render][far-terrain][scheduler][lod][cache][eviction][priority][regression]") {
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey FAR_FIRST{100, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey FAR_LAST{200, 0, FarTerrainStep::SIXTEEN};
    const std::vector order{NEAR, FAR_FIRST, FAR_LAST};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    REQUIRE(scheduler.retainWanted(wanted, order));
    const auto maintenanceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().maintenancePending != 0 &&
           std::chrono::steady_clock::now() < maintenanceDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().maintenancePending == 0);
    REQUIRE(scheduler.enqueue(FAR_FIRST, 100));
    REQUIRE(scheduler.enqueue(FAR_LAST, 200));
    const auto farDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < farDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().cacheEntries == 2);
    REQUIRE(scheduler.findCached(FAR_LAST));

    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR, 0, true));
    const auto nearDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < nearDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE(scheduler.findCached(FAR_FIRST));
    REQUIRE_FALSE(scheduler.findCached(FAR_LAST));
    scheduler.shutdown();
}

TEST_CASE("Camera-critical refinement uses capacity before a distant parent reservation",
          "[render][far-terrain][scheduler][lod][priority][near-player][regression]") {
    constexpr FarTerrainKey DISTANT_PARENT{100, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey NEAR_REFINEMENT{0, 0, FarTerrainStep::SIXTEEN};
    std::mutex startedMutex;
    std::vector<int64_t> startedTiles;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        const int64_t tile = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
        if (z == 0 && (tile == DISTANT_PARENT.tileX || tile == NEAR_REFINEMENT.tileX)) {
            std::lock_guard lock(startedMutex);
            if (std::ranges::find(startedTiles, tile) == startedTiles.end())
                startedTiles.push_back(tile);
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    scheduler.setWorkerBudget(0);
    scheduler.setCanopyWorkerBudget(0);
    REQUIRE(scheduler.enqueue(DISTANT_PARENT, 100));
    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR_REFINEMENT, 0, true));
    scheduler.setWorkerBudget(1);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    {
        std::lock_guard lock(startedMutex);
        REQUIRE(startedTiles.size() == 2);
        REQUIRE(startedTiles.front() == NEAR_REFINEMENT.tileX);
    }
    scheduler.shutdown();
}

TEST_CASE("Horizontal proximity outranks a coarse global cache wavefront",
          "[render][far-terrain][scheduler][lod][cache][eviction][priority][near-player]"
          "[regression]") {
    constexpr FarTerrainKey NEAR_PARENT{0, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR_PARENT{200, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR_COARSE{200, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey NEAR_FINE{0, 0, FarTerrainStep::TWO};
    // This is the normal broad residency shape: every parent first, followed
    // by a coarse-to-fine global wave. The far step-16 key therefore has a
    // lower exact-key rank than the near step-2 key, while the parent prefix
    // still records the near tile as the more important horizontal location.
    const std::vector order{NEAR_PARENT, FAR_PARENT, FAR_COARSE, NEAR_FINE};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    scheduler.setWorkerBudget(1);
    REQUIRE(scheduler.retainWanted(wanted, order));
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueueUrgentRefinement(FAR_COARSE, 100));
    waitForIdle();
    REQUIRE(scheduler.findCached(FAR_COARSE));

    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR_FINE, 0));
    waitForIdle();
    REQUIRE(scheduler.findCached(NEAR_FINE));
    REQUIRE_FALSE(scheduler.findCached(FAR_COARSE));

    // A later distant completion cannot reverse the decision simply because
    // it is newer or belongs to the earlier coarse wavefront.
    REQUIRE(scheduler.enqueueUrgentRefinement(FAR_COARSE, 100));
    waitForIdle();
    REQUIRE(scheduler.findCached(NEAR_FINE));
    REQUIRE_FALSE(scheduler.findCached(FAR_COARSE));
    scheduler.shutdown();
}

TEST_CASE("Bounded completion retention keeps near detail ahead of later coarse work",
          "[render][far-terrain][scheduler][lod][completion][priority][near-player]"
          "[regression]") {
    constexpr FarTerrainKey NEAR_PARENT{0, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR_PARENT{200, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR_COARSE{200, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey NEAR_FINE{0, 0, FarTerrainStep::TWO};
    const std::vector order{NEAR_PARENT, FAR_PARENT, FAR_COARSE, NEAR_FINE};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    scheduler.setWorkerBudget(1);
    REQUIRE(scheduler.retainWanted(wanted, order));
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueueUrgentRefinement(NEAR_FINE, 0));
    waitForIdle();
    REQUIRE(scheduler.stats().completed == 1);
    REQUIRE(scheduler.enqueueUrgentRefinement(FAR_COARSE, 100));
    waitForIdle();
    REQUIRE(scheduler.stats().completed == 1);

    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().key == NEAR_FINE);
    REQUIRE(completed.front().mesh);
    scheduler.shutdown();
}

TEST_CASE("A distant coverage parent cannot evict a nearer coverage parent",
          "[render][far-terrain][scheduler][coverage][cache][eviction][priority][near-player]"
          "[regression]") {
    constexpr FarTerrainKey NEAR{0, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR{200, 0, FAR_TERRAIN_BASE_STEP};
    const std::vector order{NEAR, FAR};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    scheduler.setWorkerBudget(1);
    REQUIRE(scheduler.retainWanted(wanted, order));
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueue(NEAR, 0));
    waitForIdle();
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE(scheduler.enqueue(FAR, 100));
    waitForIdle();
    REQUIRE(scheduler.findCached(NEAR));
    REQUIRE_FALSE(scheduler.findCached(FAR));
    scheduler.shutdown();
}

TEST_CASE("A distant base cannot evict a camera-critical cached refinement",
          "[render][far-terrain][scheduler][cache][critical][capacity][eviction][regression]") {
    constexpr FarTerrainKey CRITICAL{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey DISTANT_BASE{200, 0, FAR_TERRAIN_BASE_STEP};
    const std::vector order{CRITICAL, DISTANT_BASE};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    constexpr std::array criticalKeys{CRITICAL};
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    REQUIRE(scheduler.retainWanted(wanted, order, criticalKeys));

    REQUIRE(scheduler.enqueueUrgentRefinement(CRITICAL, 0, true));
    const auto criticalDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < criticalDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.findCached(CRITICAL));

    REQUIRE(scheduler.enqueue(DISTANT_BASE, 100));
    const auto baseDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < baseDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(CRITICAL));
    REQUIRE_FALSE(scheduler.findCached(DISTANT_BASE));
    scheduler.shutdown();
}

TEST_CASE("A camera-critical cache insert can evict a distant base at the byte cap",
          "[render][far-terrain][scheduler][cache][critical][capacity][eviction][regression]") {
    constexpr FarTerrainKey DISTANT_BASE{220, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey CRITICAL{0, 0, FarTerrainStep::SIXTEEN};
    const FarTerrainSource source = farTerrainTestSource();
    const auto baseProbe = FarTerrainMesher::build(DISTANT_BASE, source);
    const auto criticalProbe = FarTerrainMesher::build(CRITICAL, source);
    REQUIRE(baseProbe);
    REQUIRE(criticalProbe);

    const std::vector order{CRITICAL, DISTANT_BASE};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    constexpr std::array criticalKeys{CRITICAL};
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = std::max(baseProbe->byteSize(), criticalProbe->byteSize());
    FarTerrainScheduler scheduler(source, limits);
    REQUIRE(scheduler.retainWanted(wanted, order, criticalKeys));

    REQUIRE(scheduler.enqueue(DISTANT_BASE, 100));
    const auto baseDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < baseDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.findCached(DISTANT_BASE));

    REQUIRE(scheduler.enqueueUrgentRefinement(CRITICAL, 0, true));
    const auto criticalDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < criticalDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.findCached(CRITICAL));
    REQUIRE_FALSE(scheduler.findCached(DISTANT_BASE));
    scheduler.shutdown();
}

TEST_CASE("A higher-priority critical mesh displaces only a lower-priority critical mesh",
          "[render][far-terrain][scheduler][cache][critical][priority][eviction][regression]") {
    constexpr FarTerrainKey HIGH{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey LOW{1, 0, FarTerrainStep::SIXTEEN};
    const std::vector broadOrder{LOW, HIGH};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(broadOrder.begin(),
                                                                      broadOrder.end());
    constexpr std::array criticalOrder{HIGH, LOW};
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    REQUIRE(scheduler.retainWanted(wanted, broadOrder, criticalOrder));
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueueUrgentRefinement(LOW, 0, true));
    waitForIdle();
    REQUIRE(scheduler.findCached(LOW));

    REQUIRE(scheduler.enqueueUrgentRefinement(HIGH, 100, true));
    waitForIdle();
    REQUIRE(scheduler.findCached(HIGH));
    REQUIRE_FALSE(scheduler.findCached(LOW));

    REQUIRE(scheduler.enqueueUrgentRefinement(LOW, 0, true));
    waitForIdle();
    REQUIRE(scheduler.findCached(HIGH));
    REQUIRE_FALSE(scheduler.findCached(LOW));
    REQUIRE(scheduler.stats().cacheEntries == 1);

    constexpr std::array reversedCriticalOrder{LOW, HIGH};
    REQUIRE(scheduler.retainWanted(wanted, broadOrder, reversedCriticalOrder));
    REQUIRE_FALSE(scheduler.retainWanted(wanted, broadOrder, reversedCriticalOrder));
    REQUIRE(scheduler.enqueueUrgentRefinement(LOW, 0, true));
    waitForIdle();
    REQUIRE(scheduler.findCached(LOW));
    REQUIRE_FALSE(scheduler.findCached(HIGH));
    scheduler.shutdown();
}

TEST_CASE("A rejected larger FINAL cache replacement retains its PREVIEW payload",
          "[render][far-terrain][scheduler][authority][cache][critical][capacity][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    constexpr FarTerrainKey BLOCKER{0, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey REPLACEMENT{1, 0, FarTerrainStep::EIGHT};
    const auto detailed = std::make_shared<std::atomic<bool>>(false);
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 72.0;
            sample.waterSurface = SEA_LEVEL;
            return sample;
        },
        [detailed](int64_t x, int64_t z, const FarTerrainGeometrySample&) {
            if (!detailed->load(std::memory_order_relaxed))
                return BlockType::GRASS;
            return world_coord::floorMod(x / 8 + z / 8, 2) == 0 ? BlockType::GRASS
                                                                : BlockType::STONE;
        });
    const auto blockerProbe = FarTerrainMesher::build(BLOCKER, source, Quality::PREVIEW);
    const auto previewProbe = FarTerrainMesher::build(REPLACEMENT, source, Quality::PREVIEW);
    detailed->store(true, std::memory_order_relaxed);
    const auto finalProbe = FarTerrainMesher::build(REPLACEMENT, source, Quality::FINAL);
    REQUIRE(blockerProbe);
    REQUIRE(previewProbe);
    REQUIRE(finalProbe);
    REQUIRE(finalProbe->byteSize() > previewProbe->byteSize());
    detailed->store(false, std::memory_order_relaxed);

    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    const std::vector broadOrder{BLOCKER, REPLACEMENT};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(broadOrder.begin(),
                                                                      broadOrder.end());
    constexpr std::array criticalOrder{BLOCKER, REPLACEMENT};
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = blockerProbe->byteSize() + previewProbe->byteSize();
    FarTerrainScheduler scheduler(source, context, limits);
    scheduler.setCanopyWorkerBudget(0);
    REQUIRE(scheduler.retainWanted(wanted, broadOrder, criticalOrder));
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueue(BLOCKER));
    REQUIRE(scheduler.enqueue(REPLACEMENT));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> blocker = scheduler.findCached(BLOCKER);
    const std::shared_ptr<const FarTerrainMesh> preview = scheduler.findCached(REPLACEMENT);
    REQUIRE(blocker);
    REQUIRE(preview);
    REQUIRE(preview->authorityQuality == Quality::PREVIEW);
    REQUIRE(scheduler.stats().cacheEntries == 2);
    const size_t previewCacheBytes = scheduler.stats().cacheBytes;

    detailed->store(true, std::memory_order_relaxed);
    REQUIRE(scheduler.enqueueFinalRefinement(REPLACEMENT, 1, true));
    waitForIdle();
    REQUIRE(scheduler.findCached(BLOCKER) == blocker);
    REQUIRE(scheduler.findCached(REPLACEMENT) == preview);
    REQUIRE(scheduler.findCached(REPLACEMENT)->authorityQuality == Quality::PREVIEW);
    REQUIRE(scheduler.stats().cacheEntries == 2);
    REQUIRE(scheduler.stats().cacheBytes == previewCacheBytes);
    scheduler.shutdown();
}

TEST_CASE("Bounded critical refresh preserves camera-critical queued ordering",
          "[render][far-terrain][scheduler][critical][priority][movement][regression]") {
    constexpr FarTerrainKey BLOCKER{1'250, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey ORDINARY{1, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey CRITICAL{2, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    std::vector<int64_t> started;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (z == 0 && world_coord::floorMod(x, int64_t{FAR_TERRAIN_TILE_EDGE}) == 0) {
            const int64_t tileX = world_coord::floorDiv(x, int64_t{FAR_TERRAIN_TILE_EDGE});
            std::unique_lock lock(gateMutex);
            if ((tileX == BLOCKER.tileX || tileX == ORDINARY.tileX || tileX == CRITICAL.tileX) &&
                std::ranges::find(started, tileX) == started.end()) {
                started.push_back(tileX);
            }
            if (tileX == BLOCKER.tileX && !blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setWorkerBudget(1);
    const std::vector order{BLOCKER, ORDINARY, CRITICAL};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    REQUIRE(scheduler.retainWanted(wanted, order));
    REQUIRE(scheduler.enqueue(BLOCKER));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    REQUIRE(scheduler.enqueueUrgentRefinement(ORDINARY, 0));
    REQUIRE(scheduler.enqueueUrgentRefinement(CRITICAL, 100));

    constexpr std::array criticalKeys{CRITICAL};
    const FarTerrainSchedulerStats beforeRefresh = scheduler.stats();
    REQUIRE(scheduler.refreshCriticalPriorities(criticalKeys));
    const FarTerrainSchedulerStats afterRefresh = scheduler.stats();
    REQUIRE(afterRefresh.wantedUpdates == beforeRefresh.wantedUpdates);
    REQUIRE(afterRefresh.criticalPriorityUpdates == beforeRefresh.criticalPriorityUpdates + 1);
    REQUIRE_FALSE(scheduler.refreshCriticalPriorities(criticalKeys));
    REQUIRE(scheduler.stats().criticalPriorityNoops == afterRefresh.criticalPriorityNoops + 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    {
        std::lock_guard lock(gateMutex);
        REQUIRE(started.size() >= 3);
        REQUIRE(started[1] == CRITICAL.tileX);
        REQUIRE(started[2] == ORDINARY.tileX);
    }
    scheduler.shutdown();
}

TEST_CASE("Camera jumps reset obsolete urgent refinement quota",
          "[render][far-terrain][scheduler][priority][camera-jump][cancellation][regression]") {
    constexpr FarTerrainKey BLOCKING_BASE{900, 0, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey OLD_FIRST{901, 0, FarTerrainStep::TWO};
    constexpr FarTerrainKey OLD_SECOND{902, 0, FarTerrainStep::FOUR};
    constexpr FarTerrainKey CURRENT{903, 0, FarTerrainStep::TWO};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool baseStarted = false;
    bool releaseBase = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKING_BASE.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!baseStarted) {
                baseStarted = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBase; });
            }
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct BaseRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& release;
        ~BaseRelease() {
            {
                std::lock_guard lock(mutex);
                release = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseBase};
    scheduler.setWorkerBudget(1);
    REQUIRE(scheduler.enqueue(BLOCKING_BASE));
    {
        std::unique_lock lock(gateMutex);
        const bool started =
            gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return baseStarted; });
        if (!started)
            releaseBase = true;
        REQUIRE(started);
    }
    REQUIRE(scheduler.enqueueUrgentRefinement(OLD_FIRST));
    REQUIRE(scheduler.enqueueUrgentRefinement(OLD_SECOND));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 2);

    const uint64_t currentEpoch = scheduler.advanceEpoch();
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    REQUIRE(scheduler.enqueueUrgentRefinement(CURRENT));
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    {
        std::lock_guard lock(gateMutex);
        releaseBase = true;
    }
    gateCv.notify_all();

    for (int attempt = 0; attempt < 500 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    const FarTerrainSchedulerStats finished = scheduler.stats();
    scheduler.shutdown();
    REQUIRE(finished.inFlight == 0);
    REQUIRE(finished.urgentRefinementInFlight == 0);
    REQUIRE(finished.canceled >= 3);
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().key == CURRENT);
    REQUIRE(completed.front().epoch == currentEpoch);
}

TEST_CASE("Full horizon wanted keys fit the CPU cache across tile offsets",
          "[render][far-terrain][coverage][residency][cache][capacity][regression]") {
    std::vector<double> offsets{0.0, std::nextafter(0.0, 1.0), 1.0};
    for (int offset = 8; offset < FAR_TERRAIN_TILE_EDGE; offset += 8)
        offsets.push_back(static_cast<double>(offset));
    offsets.push_back(static_cast<double>(FAR_TERRAIN_TILE_EDGE - 1));
    offsets.push_back(std::nextafter(static_cast<double>(FAR_TERRAIN_TILE_EDGE), 0.0));

    const size_t cacheCapacity = FarTerrainSchedulerLimits{}.maxCacheEntries;
    size_t maximumWanted = 0;
    double maximumCameraX = 0.0;
    double maximumCameraZ = 0.0;
    std::vector<FarTerrainViewTile> selected;
    std::vector<FarTerrainKey> wanted;
    for (const double cameraX : offsets) {
        for (const double cameraZ : offsets) {
            selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
            buildFarTerrainResidencyOrder(selected, wanted);
            CAPTURE(cameraX, cameraZ, selected.size(), wanted.size(), cacheCapacity);
            REQUIRE(farTerrainResidencyOrderMatches(selected, wanted));
            if (wanted.size() > maximumWanted) {
                maximumWanted = wanted.size();
                maximumCameraX = cameraX;
                maximumCameraZ = cameraZ;
            }
        }
    }
    CAPTURE(maximumCameraX, maximumCameraZ, maximumWanted, cacheCapacity);
    // The pure settled selector no longer requests a broad step-1 disk. Its
    // 7,500-plus ordinary parent and refinement keys still leave room for the
    // separately bounded 60-target protected closure and replacement overlap.
    constexpr size_t MINIMUM_ENTRY_MARGIN = 32;
    REQUIRE(maximumWanted + MINIMUM_ENTRY_MARGIN <= cacheCapacity);
    REQUIRE(maximumWanted > 7'500);
}

TEST_CASE("Cold parent uploads fit the startup envelope without consuming refinement budget",
          "[render][far-terrain][coverage][upload][budget]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(0.0, 0.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    const size_t referenceColdBaseCount = selected.size();
    constexpr size_t SIXTY_FPS_FRAMES_IN_TWO_SECONDS = 120;
    const size_t parentFrames =
        (referenceColdBaseCount + FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - 1) /
        FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME;
    REQUIRE(referenceColdBaseCount > 3'000);
    REQUIRE(parentFrames < SIXTY_FPS_FRAMES_IN_TWO_SECONDS);
    REQUIRE(FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME == 12);
    REQUIRE(FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME == 32 * 1024 * 1024);
}

TEST_CASE("Exact handoff handles large snapshots empty meshes and stale revisions",
          "[render][far-terrain][coverage][exact][revision]") {
    std::vector<ChunkPos> required;
    for (int32_t y = -8; y <= -4; ++y) {
        for (int64_t z = -32; z <= 32; ++z) {
            for (int64_t x = -32; x <= 32; ++x) {
                required.push_back({x, y, z});
            }
        }
    }
    REQUIRE(required.size() > 16'384);

    std::unordered_set<ChunkPos> ready(required.begin(), required.end());
    const auto isReady = [&](ChunkPos position) { return ready.contains(position); };
    const FarTerrainExactHandoff complete =
        farTerrainExactHandoff(0.0, 0.0, 32, required, {}, isReady);
    REQUIRE(complete.requiredSections == required.size());
    REQUIRE(complete.readySections == required.size());
    REQUIRE(complete.unresolvedColumns == 0);
    REQUIRE(complete.distanceBlocks == 32 * CHUNK_EDGE);

    // Empty completed meshes own no GPU allocation, but their matching
    // revision still closes the exact coverage requirement.
    REQUIRE(farTerrainExactSectionReady(7, 7));
    REQUIRE_FALSE(farTerrainExactSectionReady(6, 7));

    constexpr ChunkPos STALE{4, -8, 0};
    ready.erase(STALE);
    const FarTerrainExactHandoff stale =
        farTerrainExactHandoff(0.0, 0.0, 32, required, {}, isReady);
    REQUIRE(stale.readySections + 1 == stale.requiredSections);
    REQUIRE(stale.distanceBlocks == 4 * CHUNK_EDGE);

    const std::array unresolved{ColumnPos{-3, 0}};
    const FarTerrainExactHandoff unresolvedResult =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved, isReady);
    REQUIRE(unresolvedResult.unresolvedColumns == 1);
    REQUIRE(unresolvedResult.distanceBlocks == 2 * CHUNK_EDGE);

    size_t readinessProbes = 0;
    FarTerrainExactCoverageCache cache;
    cache.rebuild(73, 32, required, {}, [&](ChunkPos position) {
        ++readinessProbes;
        return position != STALE;
    });
    REQUIRE(cache.matches(73, 32));
    REQUIRE_FALSE(cache.matches(74, 32));
    REQUIRE(readinessProbes == required.size());
    REQUIRE(cache.sample(0.0, 0.0).distanceBlocks == 4 * CHUNK_EDGE);
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(cache.sample(80.0, 0.0).distanceBlocks == 0.0F);
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(readinessProbes == required.size());

    REQUIRE(cache.setSectionReady(STALE, true));
    REQUIRE_FALSE(cache.setSectionReady(STALE, true));
    REQUIRE(cache.sample(0.0, 0.0).readySections == required.size());
    REQUIRE(cache.lastSampleColumnVisits() == 0);
    REQUIRE(cache.sample(0.0, 0.0).distanceBlocks == 32 * CHUNK_EDGE);
    REQUIRE(cache.setSectionReady(STALE, false));
    REQUIRE(cache.sample(0.0, 0.0).readySections + 1 == required.size());
    REQUIRE(cache.lastSampleColumnVisits() == 1);
    REQUIRE(readinessProbes == required.size());
}

TEST_CASE("Far refinement stays bounded until all exact streaming lanes drain",
          "[render][far-terrain][coverage][exact][priority][regression]") {
    REQUIRE_FALSE(farTerrainExactStreamingBusy(0, 0, 0, 24, 24, 0));

    REQUIRE(farTerrainExactStreamingBusy(1, 0, 0, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 1, 0, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 1, 24, 24, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 0, 24, 23, 0));
    REQUIRE(farTerrainExactStreamingBusy(0, 0, 0, 24, 24, 1));

    // A duplicate publication can make an observational ready count exceed
    // its requirement, but cannot manufacture pending exact work.
    REQUIRE_FALSE(farTerrainExactStreamingBusy(0, 0, 0, 24, 25, 0));
}

TEST_CASE("Exact surface handoff does not retire far canopy before crown sections",
          "[render][far-terrain][coverage][exact][canopy][flora][regression]") {
    constexpr ChunkPos SURFACE{0, 4, 0};
    constexpr ChunkPos CROWN{0, 5, 0};
    constexpr ColumnPos COLUMN{0, 0};
    constexpr std::array surfaceRequirements{SURFACE};
    constexpr std::array floraRequirements{SURFACE, CROWN};

    FarTerrainExactCoverageCache surfaceCoverage;
    FarTerrainExactCoverageCache floraCoverage;
    const auto initiallyReady = [SURFACE](ChunkPos position) { return position == SURFACE; };
    surfaceCoverage.rebuild(91, 32, surfaceRequirements, {}, initiallyReady);
    floraCoverage.rebuild(91, 32, floraRequirements, {}, initiallyReady);

    const FarTerrainExactHandoff& surface = surfaceCoverage.sample(8.0, 8.0);
    const FarTerrainExactHandoff& flora = floraCoverage.sample(8.0, 8.0);
    REQUIRE(surface.columnFullyReady(COLUMN));
    REQUIRE_FALSE(flora.columnFullyReady(COLUMN));
    REQUIRE(surface.readySections == 1);
    REQUIRE(flora.readySections == 1);
    REQUIRE(flora.requiredSections == 2);

    REQUIRE(floraCoverage.setSectionReady(CROWN, true));
    REQUIRE(floraCoverage.sample(8.0, 8.0).columnFullyReady(COLUMN));

    STATIC_REQUIRE(exactFloraSectionMayPublish(true, true, false));
    STATIC_REQUIRE(exactFloraSectionMayPublish(false, false, false));
    STATIC_REQUIRE_FALSE(exactFloraSectionMayPublish(false, true, false));
    STATIC_REQUIRE(exactFloraSectionMayPublish(false, true, true));
}

TEST_CASE("Published exact sections retain ownership while replacement meshes build",
          "[render][far-terrain][coverage][exact][ownership][latch][regression]") {
    REQUIRE(farTerrainExactSectionOwnsSurface(false, 12, 12));
    REQUIRE_FALSE(farTerrainExactSectionOwnsSurface(false, 12, 13));
    REQUIRE(farTerrainExactSectionOwnsSurface(true, 12, 13));
}

TEST_CASE("Partial exact columns never draw beneath a resident coverage parent",
          "[render][far-terrain][coverage][exact][ownership][overlap][regression]") {
    constexpr ChunkPos READY_TOP{4, 6, -3};
    constexpr ChunkPos MISSING_WALL{4, 5, -3};
    constexpr ColumnPos COLUMN{READY_TOP.x, READY_TOP.z};
    constexpr std::array required{READY_TOP, MISSING_WALL};

    FarTerrainExactCoverageCache coverage;
    coverage.rebuild(81, 32, required, {},
                     [READY_TOP](ChunkPos position) { return position == READY_TOP; });
    const FarTerrainExactHandoff& partial = coverage.sample(72.0, -40.0);
    REQUIRE(coverage.sectionRequired(READY_TOP));
    REQUIRE(coverage.sectionRequired(MISSING_WALL));
    REQUIRE_FALSE(coverage.sectionRequired({COLUMN.x, 7, COLUMN.z}));
    REQUIRE_FALSE(partial.columnFullyReady(COLUMN));

    // A cold outer horizon must not disable the atomic handoff for an inner
    // parent that is already inside the connected drawable prefix.
    const std::vector<FarTerrainViewTile> selected = {
        {{0, 0, FarTerrainStep::ONE}, {}, 0.0, 32.0, {}},
        {{4, 0, FarTerrainStep::ONE}, {}, 4096.0, 64.0, {}},
    };
    const FarTerrainCoverageFrontier frontier = farTerrainCoverageFrontier(
        selected, [](const FarTerrainKey& key) { return key.tileX == 0; });
    REQUIRE_FALSE(frontier.complete);
    REQUIRE(frontier.missingBaseTiles == 1);
    const bool nearParentDrawable =
        farTerrainCoverageDrawEligible(selected.front().distanceSquared, frontier);
    REQUIRE(nearParentDrawable);
    REQUIRE_FALSE(farTerrainCoverageDrawEligible(selected.back().distanceSquared, frontier));

    // The old draw path submitted READY_TOP immediately while the same
    // column's coarse parent remained visible for MISSING_WALL. The result was
    // the large grass and stone slabs layered over otherwise complete exact
    // terrain in the reported v4 scene.
    REQUIRE_FALSE(farTerrainExactSectionDrawAllowed(
        coverage.sectionRequired(READY_TOP), partial.columnFullyReady(COLUMN), nearParentDrawable));
    REQUIRE(farTerrainExactSectionDrawAllowed(coverage.sectionRequired(READY_TOP),
                                              partial.columnFullyReady(COLUMN), false));
    REQUIRE(farTerrainExactSectionDrawAllowed(false, partial.columnFullyReady(COLUMN), true));

    REQUIRE(coverage.setSectionReady(MISSING_WALL, true));
    const FarTerrainExactHandoff& complete = coverage.sample(72.0, -40.0);
    REQUIRE(complete.columnFullyReady(COLUMN));
    REQUIRE(farTerrainExactSectionDrawAllowed(coverage.sectionRequired(READY_TOP),
                                              complete.columnFullyReady(COLUMN), true));
    REQUIRE(farTerrainExactSectionDrawAllowed(coverage.sectionRequired(MISSING_WALL),
                                              complete.columnFullyReady(COLUMN), true));
}

TEST_CASE("Exact mesh registry waits when every capacity slot owns terrain",
          "[render][coverage][exact][ownership][capacity][regression]") {
    REQUIRE(chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES - 1, MAX_MESH_RESIDENT_CUBES, false,
                                      false));
    REQUIRE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, true, false));
    REQUIRE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, false, true));
    REQUIRE_FALSE(
        chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES, MAX_MESH_RESIDENT_CUBES, false, false));
    REQUIRE_FALSE(chunkMeshRegistryCanAdmit(MAX_MESH_RESIDENT_CUBES + 1, MAX_MESH_RESIDENT_CUBES,
                                            false, true));
}

TEST_CASE("Exact registry pressure can only replace lower-priority distant work",
          "[render][coverage][exact][capacity][eviction][priority][regression]") {
    constexpr ChunkPos CAMERA{0, 4, 0};
    constexpr ExactMeshUploadPriority NEAR =
        exactMeshUploadPriority({1, 4, 0}, CAMERA, EXPLORATION_RADIUS_CHUNKS, true, false);
    constexpr ExactMeshUploadPriority FAR =
        exactMeshUploadPriority({28, 4, 0}, CAMERA, EXPLORATION_RADIUS_CHUNKS, true, false);
    constexpr ExactMeshUploadPriority FARTHER =
        exactMeshUploadPriority({31, 4, 0}, CAMERA, EXPLORATION_RADIUS_CHUNKS, true, false);

    STATIC_REQUIRE(exactMeshRegistryMayReplace(NEAR, FAR));
    STATIC_REQUIRE(exactMeshRegistryMayReplace(FAR, FARTHER));
    STATIC_REQUIRE_FALSE(exactMeshRegistryMayReplace(FAR, NEAR));
    STATIC_REQUIRE_FALSE(exactMeshRegistryMayReplace(FAR, FAR));
}

TEST_CASE("Current-drain exact uploads survive cap pressure during vertical flight",
          "[render][coverage][exact][ownership][capacity][flight][priority][regression]") {
    constexpr ChunkPos CAMERA{0, 80, 0};
    constexpr ChunkPos FAR_PENDING{24, 80, 0};
    constexpr ChunkPos MEDIUM_PENDING{12, 80, 0};
    constexpr ChunkPos NEAR_SURFACE{1, 4, 0};
    constexpr ChunkPos SECOND_NEAR_SURFACE{0, 3, 1};

    // A 3D-only capacity sweep considers the terrain directly below a flying
    // camera farther away than the genuinely distant horizontal request.
    constexpr int64_t nearDx = NEAR_SURFACE.x - CAMERA.x;
    constexpr int64_t nearDy = NEAR_SURFACE.y - CAMERA.y;
    constexpr int64_t nearDz = NEAR_SURFACE.z - CAMERA.z;
    constexpr int64_t farDx = FAR_PENDING.x - CAMERA.x;
    constexpr int64_t farDy = FAR_PENDING.y - CAMERA.y;
    constexpr int64_t farDz = FAR_PENDING.z - CAMERA.z;
    STATIC_REQUIRE(nearDx * nearDx + nearDy * nearDy + nearDz * nearDz >
                   farDx * farDx + farDy * farDy + farDz * farDz);
    constexpr ExactMeshUploadPriority NEAR_PRIORITY =
        exactMeshUploadPriority(NEAR_SURFACE, CAMERA, EXPLORATION_RADIUS_CHUNKS, true, false);
    constexpr ExactMeshUploadPriority FAR_PRIORITY =
        exactMeshUploadPriority(FAR_PENDING, CAMERA, EXPLORATION_RADIUS_CHUNKS, false, false);
    STATIC_REQUIRE(exactMeshUploadRanksBefore(NEAR_PRIORITY, FAR_PRIORITY));
    STATIC_REQUIRE(exactMeshEvictionRanksBefore(FAR_PRIORITY, NEAR_PRIORITY));

    struct Slot {
        ChunkPos position;
        bool surfaceRequired = false;
        bool exactOwned = false;
        bool committedThisDrain = false;
    };
    std::array slots{
        Slot{FAR_PENDING},
        Slot{MEDIUM_PENDING},
    };
    const auto selectVictim = [&]() -> std::optional<size_t> {
        std::optional<size_t> victim;
        std::optional<ExactMeshUploadPriority> victimPriority;
        for (size_t index = 0; index < slots.size(); ++index) {
            const Slot& slot = slots[index];
            if (!exactMeshRegistryVictimEligible(slot.exactOwned, slot.committedThisDrain))
                continue;
            const ExactMeshUploadPriority priority = exactMeshUploadPriority(
                slot.position, CAMERA, EXPLORATION_RADIUS_CHUNKS, slot.surfaceRequired, false);
            if (!victimPriority || exactMeshEvictionRanksBefore(priority, *victimPriority)) {
                victim = index;
                victimPriority = priority;
            }
        }
        return victim;
    };

    // The first completed near surface replaces the least important distant
    // placeholder. It has committed GPU storage but intentionally remains
    // outside exact column ownership until the post-drain atomic handoff.
    const std::optional<size_t> firstVictim = selectVictim();
    REQUIRE(firstVictim == 0);
    slots[*firstVictim] = Slot{NEAR_SURFACE, true, false, true};
    REQUIRE_FALSE(exactMeshRegistryVictimEligible(false, true));

    // A second completion in the same frame must evict the remaining distant
    // placeholder, never the vertically distant surface that just committed.
    const std::optional<size_t> secondVictim = selectVictim();
    REQUIRE(secondVictim == 1);
    slots[*secondVictim] = Slot{SECOND_NEAR_SURFACE, true, false, true};
    REQUIRE_FALSE(selectVictim().has_value());
    REQUIRE_FALSE(exactMeshRegistryVictimEligible(true, false));
}

TEST_CASE("One unresolved exact section preserves refinement in every ready column",
          "[render][far-terrain][coverage][exact][ownership][column][regression]") {
    std::vector<ChunkPos> required;
    required.reserve(16 * 16);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x)
            required.push_back({x, 4, z});
    }
    constexpr ChunkPos UNRESOLVED{7, 4, 9};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(128.0, 128.0, 32, required, {},
                               [UNRESOLVED](ChunkPos section) { return section != UNRESOLVED; });

    REQUIRE_FALSE(handoff.tileFullyReady({0, 0}));
    size_t exactOwnedColumns = 0;
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            const bool ready = handoff.columnFullyReady({x, z});
            CAPTURE(x, z);
            REQUIRE(ready == (ChunkPos{x, 4, z} != UNRESOLVED));
            exactOwnedColumns += ready ? 1 : 0;
        }
    }
    REQUIRE(exactOwnedColumns == 255);
}

TEST_CASE("Active far LOD transitions complete before selecting another target",
          "[render][far-terrain][lod][transition][monotonic][regression]") {
    const FarTerrainLodAdvance started =
        advanceFarTerrainLod(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO);
    REQUIRE(started.displayed == FarTerrainStep::THIRTY_TWO);
    REQUIRE(started.transitionTarget == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(started.completedTransition);

    const FarTerrainLodAdvance redirected =
        advanceFarTerrainLod(started.displayed, FarTerrainStep::FOUR, started.transitionTarget,
                             FAR_TERRAIN_LOD_TRANSITION_SECONDS * 0.75F);
    REQUIRE(redirected.displayed == FarTerrainStep::THIRTY_TWO);
    REQUIRE(redirected.transitionTarget == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(redirected.completedTransition);

    const FarTerrainLodAdvance completed =
        advanceFarTerrainLod(redirected.displayed, FarTerrainStep::FOUR,
                             redirected.transitionTarget, FAR_TERRAIN_LOD_TRANSITION_SECONDS);
    REQUIRE(completed.displayed == FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(completed.transitionTarget.has_value());
    REQUIRE(completed.completedTransition);

    const FarTerrainLodAdvance next =
        advanceFarTerrainLod(completed.displayed, FarTerrainStep::FOUR);
    REQUIRE(next.displayed == FarTerrainStep::SIXTEEN);
    REQUIRE(next.transitionTarget == FarTerrainStep::EIGHT);
}

TEST_CASE("Nearby far fallback chooses its finest ready tier without regressing",
          "[render][far-terrain][lod][residency][exact][priority][regression]") {
    FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::THIRTY_TWO);

    ready |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::SIXTEEN);
    REQUIRE_FALSE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  ready, true));
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::SIXTEEN);
    ready |= farTerrainStepMask(FarTerrainStep::FOUR);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::FOUR);
    ready |= farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::FOUR, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::TWO);

    // A step-16 replacement can finish while finer CPU work completes. Cold
    // work advances through adjacent powers of two so every completed mesh
    // can satisfy the renderer's neighbor compatibility gate.
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE_FALSE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  ready, true));
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::EIGHT);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::EIGHT, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::FOUR, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::TWO);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready,
                                            false) == FarTerrainStep::SIXTEEN);

    // A transient cache observation cannot make an already displayed nearby
    // refinement fall back to an emergency parent.
    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
            farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::TWO, ready) ==
            FarTerrainStep::TWO);

    // Distance-selected coarsening remains intentional and requires the
    // requested replacement to be resident.
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::EIGHT, ready) ==
            FarTerrainStep::TWO);
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE(farTerrainFinestReadyStep(FarTerrainStep::TWO, FarTerrainStep::EIGHT, ready) ==
            FarTerrainStep::EIGHT);
}

TEST_CASE("Near selected bands cap temporary fallback without shrinking the horizon",
          "[render][far-terrain][lod][residency][near-camera][regression]") {
    for (const FarTerrainStep desired : {FarTerrainStep::ONE, FarTerrainStep::TWO}) {
        CAPTURE(desired);
        REQUIRE(farTerrainCoarsestDrawableFallback(desired, false, false) == FarTerrainStep::TWO);
        REQUIRE(farTerrainCoarsestDrawableFallback(desired, true, false) == FarTerrainStep::TWO);
    }
    REQUIRE(farTerrainCoarsestDrawableFallback(FarTerrainStep::FOUR, true, false) ==
            FarTerrainStep::EIGHT);
    REQUIRE(farTerrainCoarsestDrawableFallback(FarTerrainStep::EIGHT, false, true) ==
            FarTerrainStep::TWO);
    REQUIRE(farTerrainCoarsestDrawableFallback(FarTerrainStep::EIGHT, false, false) ==
            FarTerrainStep::THIRTY_TWO);

    REQUIRE(farTerrainProtectedIntermediateMayDisplay(true, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE_FALSE(
        farTerrainProtectedIntermediateMayDisplay(true, FarTerrainAuthorityQuality::FINAL));
    REQUIRE(farTerrainProtectedIntermediateMayDisplay(false, FarTerrainAuthorityQuality::FINAL));
}

TEST_CASE("A flying camera reuses a resident near target without replaying coarse tiers",
          "[render][far-terrain][lod][residency][flying][flicker][regression]") {
    const FarTerrainStepMask ready =
        farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
        farTerrainStepMask(FarTerrainStep::SIXTEEN) | farTerrainStepMask(FarTerrainStep::EIGHT) |
        farTerrainStepMask(FarTerrainStep::TWO) | farTerrainStepMask(FarTerrainStep::ONE);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::ONE);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::ONE, ready,
                                            false) == FarTerrainStep::SIXTEEN);

    const std::array<std::optional<FarTerrainStep>, 4> compatibleNeighbors{
        FarTerrainStep::TWO, FarTerrainStep::TWO, FarTerrainStep::ONE, FarTerrainStep::TWO};
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::ONE, compatibleNeighbors));
    auto incompatibleNeighbors = compatibleNeighbors;
    incompatibleNeighbors[0] = FarTerrainStep::EIGHT;
    REQUIRE_FALSE(
        farTerrainStepCompatibleWithNeighbors(FarTerrainStep::ONE, incompatibleNeighbors));

    for (const bool exactReady : {false, true, false, true}) {
        CAPTURE(exactReady);
        REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::ONE);
        REQUIRE(farTerrainExactSectionDrawAllowed(true, exactReady, true) == exactReady);
    }
}

TEST_CASE("Connected near patch waits for every fine tile and its legal shell",
          "[render][far-terrain][lod][residency][patch][atomic][regression]") {
    std::vector<FarTerrainViewTile> selected;
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    constexpr ColumnPos MISSING_FINE{0, 0};
    for (int64_t z = -2; z <= 2; ++z) {
        for (int64_t x = -2; x <= 2; ++x) {
            const bool fine = std::abs(x) <= 1 && std::abs(z) <= 1;
            selected.push_back({{x, z, fine ? FarTerrainStep::ONE : FarTerrainStep::TWO},
                                {},
                                0.0,
                                fine ? 48.0 : 64.0,
                                std::nullopt});
            resident.insert({x, z, FAR_TERRAIN_BASE_STEP});
            if (fine && ColumnPos{x, z} != MISSING_FINE)
                resident.insert({x, z, FarTerrainStep::ONE});
            if (!fine)
                resident.insert({x, z, FarTerrainStep::TWO});
        }
    }
    const auto isResident = [&](const FarTerrainKey& key) { return resident.contains(key); };
    std::vector<FarTerrainKey> targets;
    REQUIRE_FALSE(buildFarTerrainConnectedNearPatchHandoff(selected, isResident, targets));
    REQUIRE(targets.empty());

    resident.insert({MISSING_FINE.x, MISSING_FINE.z, FarTerrainStep::ONE});
    REQUIRE(buildFarTerrainConnectedNearPatchHandoff(selected, isResident, targets));
    REQUIRE(targets.size() == 21);
    REQUIRE(std::ranges::count_if(
                targets, [](FarTerrainKey key) { return key.step == FarTerrainStep::ONE; }) == 9);
    REQUIRE(std::ranges::count_if(
                targets, [](FarTerrainKey key) { return key.step == FarTerrainStep::TWO; }) == 12);
    for (const FarTerrainKey target : targets) {
        REQUIRE(resident.contains(target));
        REQUIRE(resident.contains({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP}));
    }

    const std::vector<FarTerrainKey> atomicTargets = targets;
    for (int frame = 0; frame < 64; ++frame) {
        CAPTURE(frame);
        REQUIRE(buildFarTerrainConnectedNearPatchHandoff(selected, isResident, targets));
        REQUIRE(targets == atomicTargets);
    }

    resident.erase({2, 0, FarTerrainStep::TWO});
    REQUIRE_FALSE(buildFarTerrainConnectedNearPatchHandoff(selected, isResident, targets));
    REQUIRE(targets.empty());
}

TEST_CASE("Protected near anchor follows half-tile boundaries across the origin",
          "[render][far-terrain][lod][residency][entry][anchor][negative][regression]") {
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(0, 0) == ColumnPos{-1, -1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(127, 127) == ColumnPos{-1, -1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(128, 128) == ColumnPos{0, 0});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(255, 255) == ColumnPos{0, 0});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(256, 256) == ColumnPos{0, 0});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(383, 383) == ColumnPos{0, 0});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(384, 384) == ColumnPos{1, 1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(127, 128) == ColumnPos{-1, 0});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(128, 127) == ColumnPos{0, -1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(-1, -1) == ColumnPos{-1, -1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(-128, -128) == ColumnPos{-1, -1});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(-129, -129) == ColumnPos{-2, -2});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(-256, -256) == ColumnPos{-2, -2});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(-257, -257) == ColumnPos{-2, -2});
    STATIC_REQUIRE(farTerrainProtectedNearAnchor(std::numeric_limits<int64_t>::min(),
                                                 std::numeric_limits<int64_t>::max()) ==
                   ColumnPos{-36'028'797'018'963'969LL, 36'028'797'018'963'967LL});

    for (int64_t tile = -3; tile <= 3; ++tile) {
        const int64_t origin = tile * FAR_TERRAIN_TILE_EDGE;
        CAPTURE(tile, origin);
        REQUIRE(farTerrainProtectedNearAnchor(origin + 127, origin + 127) ==
                ColumnPos{tile - 1, tile - 1});
        REQUIRE(farTerrainProtectedNearAnchor(origin + 128, origin + 128) == ColumnPos{tile, tile});
        REQUIRE(farTerrainProtectedNearAnchor(origin + 255, origin + 255) == ColumnPos{tile, tile});
        REQUIRE(farTerrainProtectedNearAnchor(origin + 256, origin + 256) == ColumnPos{tile, tile});
    }
}

TEST_CASE("Protected near prediction is bounded directional and half-open",
          "[render][far-terrain][lod][residency][prediction][movement][negative][regression]") {
    STATIC_REQUIRE((farTerrainPredictedProtectedNearAnchor(256, 256, 1, 0) ==
                    std::optional<ColumnPos>{ColumnPos{1, 0}}));
    STATIC_REQUIRE((farTerrainPredictedProtectedNearAnchor(255, 256, -1, 0) ==
                    std::optional<ColumnPos>{ColumnPos{-1, 0}}));
    STATIC_REQUIRE((farTerrainPredictedProtectedNearAnchor(320, 320, 1, 1) ==
                    std::optional<ColumnPos>{ColumnPos{1, 1}}));
    STATIC_REQUIRE((farTerrainPredictedProtectedNearAnchor(-256, -256, 1, 0) ==
                    std::optional<ColumnPos>{ColumnPos{-1, -2}}));
    STATIC_REQUIRE((farTerrainPredictedProtectedNearAnchor(-257, -256, -1, 0) ==
                    std::optional<ColumnPos>{ColumnPos{-3, -2}}));

    STATIC_REQUIRE_FALSE(farTerrainPredictedProtectedNearAnchor(320, 320, 0, 0).has_value());
    STATIC_REQUIRE_FALSE(farTerrainPredictedProtectedNearAnchor(255, 256, 1, 0).has_value());
    STATIC_REQUIRE_FALSE(farTerrainPredictedProtectedNearAnchor(256, 256, -1, 0).has_value());
    STATIC_REQUIRE_FALSE(farTerrainPredictedProtectedNearAnchor(320, 320, -1, -1).has_value());

    const auto east = farTerrainPredictedProtectedNearAnchor(320, 320, 1, 0);
    const auto reversed = farTerrainPredictedProtectedNearAnchor(320, 320, -1, 0);
    REQUIRE(east == std::optional<ColumnPos>{ColumnPos{1, 0}});
    REQUIRE_FALSE(reversed.has_value());
    REQUIRE(320 >= east->x * FAR_TERRAIN_TILE_EDGE);
    REQUIRE(320 < (east->x + FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES) * FAR_TERRAIN_TILE_EDGE);
    REQUIRE(320 >= east->z * FAR_TERRAIN_TILE_EDGE);
    REQUIRE(320 < (east->z + FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES) * FAR_TERRAIN_TILE_EDGE);
}

TEST_CASE("Only exact requested final targets receive critical refinement admission",
          "[render][far-terrain][lod][residency][upload][arena][critical][regression]") {
    constexpr ColumnPos ANCHOR{0, 0};
    const std::optional<ColumnPos> requested = ANCHOR;
    constexpr FarTerrainKey CORE_TARGET{0, 0, FarTerrainStep::ONE};
    constexpr FarTerrainKey CORE_ALTERNATE{0, 0, FarTerrainStep::TWO};
    constexpr FarTerrainKey RING_TARGET{-1, 0, FarTerrainStep::TWO};
    constexpr FarTerrainKey PARENT{0, 0, FAR_TERRAIN_BASE_STEP};

    REQUIRE(farTerrainProtectedNearTargetKey(requested, CORE_TARGET));
    REQUIRE(farTerrainProtectedNearTargetKey(requested, RING_TARGET));
    REQUIRE_FALSE(farTerrainProtectedNearTargetKey(requested, CORE_ALTERNATE));
    REQUIRE_FALSE(farTerrainProtectedNearTargetKey(requested, PARENT));
    REQUIRE(farTerrainCriticalProtectedRefinement(requested, CORE_TARGET,
                                                  FarTerrainAuthorityQuality::FINAL));
    REQUIRE_FALSE(farTerrainCriticalProtectedRefinement(requested, CORE_TARGET,
                                                        FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE_FALSE(farTerrainCriticalProtectedRefinement(requested, CORE_ALTERNATE,
                                                        FarTerrainAuthorityQuality::FINAL));
    REQUIRE_FALSE(farTerrainCriticalProtectedRefinement(std::nullopt, CORE_TARGET,
                                                        FarTerrainAuthorityQuality::FINAL));
    REQUIRE(farTerrainGpuMayEvictForNear(
        false, false, false, false, farTerrainProtectedNearTargetKey(requested, CORE_ALTERNATE),
        false, false));

    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(320.0, 320.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    std::vector<FarTerrainKey> targets;
    buildFarTerrainProtectedNearTargets(ANCHOR, selected, targets);
    REQUIRE(targets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    for (const FarTerrainKey target : targets) {
        CAPTURE(target.tileX, target.tileZ, farTerrainStepSize(target.step));
        REQUIRE(farTerrainCriticalProtectedRefinement(requested, target,
                                                      FarTerrainAuthorityQuality::FINAL));
        for (const FarTerrainStep alternate :
             {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            if (alternate == target.step)
                continue;
            REQUIRE_FALSE(farTerrainCriticalProtectedRefinement(
                requested, {target.tileX, target.tileZ, alternate},
                FarTerrainAuthorityQuality::FINAL));
        }
    }
}

TEST_CASE("Protected near Manhattan topology has exact counts and legal exterior ratios",
          "[render][far-terrain][lod][residency][entry][topology][negative][regression]") {
    constexpr ColumnPos ANCHOR{-9, 7};
    const auto stepForRole = [](FarTerrainProtectedNearRole role) {
        switch (role) {
            case FarTerrainProtectedNearRole::STEP_ONE_CORE:
                return FarTerrainStep::ONE;
            case FarTerrainProtectedNearRole::STEP_TWO_RING:
                return FarTerrainStep::TWO;
            case FarTerrainProtectedNearRole::STEP_FOUR_RING:
                return FarTerrainStep::FOUR;
            case FarTerrainProtectedNearRole::STEP_EIGHT_RING:
                return FarTerrainStep::EIGHT;
            case FarTerrainProtectedNearRole::STEP_SIXTEEN_RING:
                return FarTerrainStep::SIXTEEN;
            case FarTerrainProtectedNearRole::NONE:
                return FarTerrainStep::THIRTY_TWO;
        }
        return FarTerrainStep::THIRTY_TWO;
    };
    std::array<size_t, 5> counts{};
    std::vector<ColumnPos> coordinates;
    for (int64_t z = ANCHOR.z - 5; z <= ANCHOR.z + 6; ++z) {
        for (int64_t x = ANCHOR.x - 5; x <= ANCHOR.x + 6; ++x) {
            const FarTerrainProtectedNearRole role = farTerrainProtectedNearRole(ANCHOR, {x, z});
            if (role == FarTerrainProtectedNearRole::NONE)
                continue;
            coordinates.push_back({x, z});
            ++counts[static_cast<size_t>(role) - 1];
        }
    }
    REQUIRE(counts[0] == FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_TILE_COUNT);
    REQUIRE(counts[1] == FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_TILE_COUNT);
    REQUIRE(counts[2] == FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_TILE_COUNT);
    REQUIRE(counts[3] == FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_TILE_COUNT);
    REQUIRE(counts[4] == FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_TILE_COUNT);
    REQUIRE(coordinates.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    for (const ColumnPos core :
         {ANCHOR, ColumnPos{ANCHOR.x + 1, ANCHOR.z}, ColumnPos{ANCHOR.x, ANCHOR.z + 1},
          ColumnPos{ANCHOR.x + 1, ANCHOR.z + 1}}) {
        REQUIRE(farTerrainProtectedNearRole(ANCHOR, core) ==
                FarTerrainProtectedNearRole::STEP_ONE_CORE);
    }

    constexpr std::array CARDINALS{ColumnPos{1, 0}, ColumnPos{-1, 0}, ColumnPos{0, 1},
                                   ColumnPos{0, -1}};
    bool sawStep32Exterior = false;
    for (const ColumnPos coordinate : coordinates) {
        const FarTerrainStep step = stepForRole(farTerrainProtectedNearRole(ANCHOR, coordinate));
        for (const ColumnPos offset : CARDINALS) {
            const ColumnPos neighbor{coordinate.x + offset.x, coordinate.z + offset.z};
            const FarTerrainStep neighborStep =
                stepForRole(farTerrainProtectedNearRole(ANCHOR, neighbor));
            const int smaller =
                std::min(farTerrainStepSize(step), farTerrainStepSize(neighborStep));
            const int larger = std::max(farTerrainStepSize(step), farTerrainStepSize(neighborStep));
            CAPTURE(coordinate.x, coordinate.z, neighbor.x, neighbor.z, smaller, larger);
            REQUIRE(larger <= smaller * 2);
            sawStep32Exterior = sawStep32Exterior || neighborStep == FarTerrainStep::THIRTY_TWO;
        }
    }
    REQUIRE(sawStep32Exterior);
}

TEST_CASE("Protected near terrain publishes complete entry and movement handoffs",
          "[render][far-terrain][lod][residency][entry][atomic][flying][regression]") {
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(0.0, 0.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);

    constexpr ColumnPos INITIAL_ANCHOR = farTerrainProtectedNearAnchor(128, 128);
    constexpr ColumnPos MOVED_ANCHOR = farTerrainProtectedNearAnchor(384, 128);
    static_assert(INITIAL_ANCHOR == ColumnPos{0, 0});
    static_assert(MOVED_ANCHOR == ColumnPos{1, 0});
    constexpr ColumnPos OLD_ONLY{-4, 0};
    constexpr ColumnPos NEW_ONLY{6, 0};
    std::vector<FarTerrainKey> initialTargets;
    buildFarTerrainProtectedNearTargets(INITIAL_ANCHOR, selected, initialTargets);
    REQUIRE(initialTargets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(std::ranges::count_if(initialTargets, [](FarTerrainKey key) {
                return key.step == FarTerrainStep::ONE;
            }) == FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_TILE_COUNT);
    REQUIRE(std::ranges::count_if(initialTargets, [](FarTerrainKey key) {
                return key.step == FarTerrainStep::TWO;
            }) == FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_TILE_COUNT);
    REQUIRE(std::ranges::count_if(initialTargets, [](FarTerrainKey key) {
                return key.step == FarTerrainStep::FOUR;
            }) == FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_TILE_COUNT);
    REQUIRE(std::ranges::count_if(initialTargets, [](FarTerrainKey key) {
                return key.step == FarTerrainStep::EIGHT;
            }) == FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_TILE_COUNT);
    REQUIRE(std::ranges::count_if(initialTargets, [](FarTerrainKey key) {
                return key.step == FarTerrainStep::SIXTEEN;
            }) == FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_TILE_COUNT);

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    for (const FarTerrainKey target : initialTargets) {
        resident.insert({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
    }
    const auto isResident = [&](const FarTerrainKey& key) { return resident.contains(key); };
    REQUIRE_FALSE(farTerrainProtectedNearTargetsReady(initialTargets, isResident));

    FarTerrainProtectedNearHandoff handoff;
    REQUIRE(handoff.request(INITIAL_ANCHOR));
    REQUIRE(farTerrainProtectedNearRequiredStep(handoff, INITIAL_ANCHOR) == FarTerrainStep::ONE);
    REQUIRE(farTerrainProtectedNearRequiredStep(handoff, OLD_ONLY) == FarTerrainStep::SIXTEEN);
    for (const FarTerrainKey target : initialTargets)
        resident.insert(target);
    REQUIRE(farTerrainProtectedNearTargetsReady(initialTargets, isResident));
    REQUIRE(handoff.commitRequested(true));
    REQUIRE(handoff.activeCenter() == INITIAL_ANCHOR);
    REQUIRE_FALSE(handoff.requestedCenter().has_value());

    REQUIRE(handoff.request(MOVED_ANCHOR));
    REQUIRE(handoff.activeCenter() == INITIAL_ANCHOR);
    REQUIRE(handoff.requestedCenter() == MOVED_ANCHOR);
    REQUIRE(farTerrainProtectedNearRequiredStep(handoff, OLD_ONLY) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainProtectedNearRequiredStep(handoff, NEW_ONLY) == FarTerrainStep::SIXTEEN);

    std::vector<FarTerrainKey> movedTargets;
    buildFarTerrainProtectedNearTargets(MOVED_ANCHOR, selected, movedTargets);
    REQUIRE(movedTargets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    for (const FarTerrainKey target : movedTargets) {
        resident.insert({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
        if (ColumnPos{target.tileX, target.tileZ} != NEW_ONLY)
            resident.insert(target);
    }
    REQUIRE_FALSE(farTerrainProtectedNearTargetsReady(movedTargets, isResident));
    REQUIRE_FALSE(handoff.commitRequested(false));
    REQUIRE(handoff.activeCenter() == INITIAL_ANCHOR);

    resident.insert({NEW_ONLY.x, NEW_ONLY.z, FarTerrainStep::SIXTEEN});
    REQUIRE(farTerrainProtectedNearTargetsReady(movedTargets, isResident));
    REQUIRE(handoff.commitRequested(true));
    REQUIRE(handoff.activeCenter() == MOVED_ANCHOR);
    REQUIRE_FALSE(handoff.requestedCenter().has_value());
    REQUIRE_FALSE(farTerrainProtectedNearRequiredStep(handoff, OLD_ONLY).has_value());
}

TEST_CASE("Protected hidden FINAL targets remain GPU-resident until handoff resolution",
          "[render][far-terrain][lod][residency][gpu][atomic][flying][regression]") {
    constexpr ColumnPos ACTIVE{0, 0};
    constexpr ColumnPos REQUESTED{1, 0};
    constexpr ColumnPos ACTIVE_ONLY{-4, 0};
    constexpr ColumnPos REQUESTED_ONLY{6, 0};
    constexpr ColumnPos OVERLAP_CHANGED_ROLE{0, 0};

    // This is the production failure mode: screen-space selection and the
    // displayed proxy have already advanced beyond the hidden outer target.
    // Progressive cleanup cannot retain it, but atomic closure ownership must.
    REQUIRE_FALSE(farTerrainRetainsProgressiveStep(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO,
                                                   FarTerrainStep::ONE));
    REQUIRE(farTerrainProtectedGpuResidencyRequired(
        {REQUESTED_ONLY.x, REQUESTED_ONLY.z, FarTerrainStep::SIXTEEN}, ACTIVE, REQUESTED));
    REQUIRE(farTerrainProtectedGpuResidencyRequired(
        {REQUESTED_ONLY.x, REQUESTED_ONLY.z, FAR_TERRAIN_BASE_STEP}, ACTIVE, REQUESTED));

    // The old drawable closure and the replacement closure coexist. A key at
    // an overlapping coordinate remains retained for either anchor's role.
    REQUIRE(farTerrainProtectedGpuResidencyRequired(
        {ACTIVE_ONLY.x, ACTIVE_ONLY.z, FarTerrainStep::SIXTEEN}, ACTIVE, REQUESTED));
    REQUIRE(farTerrainProtectedGpuResidencyRequired(
        {OVERLAP_CHANGED_ROLE.x, OVERLAP_CHANGED_ROLE.z, FarTerrainStep::ONE}, ACTIVE, REQUESTED));
    REQUIRE(farTerrainProtectedGpuResidencyRequired(
        {OVERLAP_CHANGED_ROLE.x, OVERLAP_CHANGED_ROLE.z, FarTerrainStep::TWO}, ACTIVE, REQUESTED));
    REQUIRE_FALSE(farTerrainProtectedGpuResidencyRequired(
        {REQUESTED_ONLY.x, REQUESTED_ONLY.z, FarTerrainStep::EIGHT}, ACTIVE, REQUESTED));

    // Commit abandons old-only keys. Clearing both anchors abandons every
    // protected allocation and returns ownership to ordinary cleanup.
    REQUIRE_FALSE(farTerrainProtectedGpuResidencyRequired(
        {ACTIVE_ONLY.x, ACTIVE_ONLY.z, FarTerrainStep::SIXTEEN}, REQUESTED, std::nullopt));
    REQUIRE_FALSE(farTerrainProtectedGpuResidencyRequired(
        {REQUESTED_ONLY.x, REQUESTED_ONLY.z, FarTerrainStep::SIXTEEN}, std::nullopt, std::nullopt));
}

TEST_CASE("Protected GPU retention exactly covers both complete handoff lineages",
          "[render][far-terrain][lod][residency][gpu][atomic][flying][exhaustive][regression]") {
    constexpr ColumnPos ACTIVE{-3, 4};
    constexpr ColumnPos REQUESTED{-2, 5};
    constexpr std::array STEPS{
        FarTerrainStep::ONE,   FarTerrainStep::TWO,     FarTerrainStep::FOUR,
        FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
    };
    const auto targetStep = [](FarTerrainProtectedNearRole role) -> std::optional<FarTerrainStep> {
        switch (role) {
            case FarTerrainProtectedNearRole::STEP_ONE_CORE:
                return FarTerrainStep::ONE;
            case FarTerrainProtectedNearRole::STEP_TWO_RING:
                return FarTerrainStep::TWO;
            case FarTerrainProtectedNearRole::STEP_FOUR_RING:
                return FarTerrainStep::FOUR;
            case FarTerrainProtectedNearRole::STEP_EIGHT_RING:
                return FarTerrainStep::EIGHT;
            case FarTerrainProtectedNearRole::STEP_SIXTEEN_RING:
                return FarTerrainStep::SIXTEEN;
            case FarTerrainProtectedNearRole::NONE:
                return std::nullopt;
        }
        return std::nullopt;
    };
    const auto requiredBy = [&](ColumnPos anchor, FarTerrainKey key) {
        const std::optional<FarTerrainStep> required =
            targetStep(farTerrainProtectedNearRole(anchor, {key.tileX, key.tileZ}));
        return required && (key.step == *required || key.step == FAR_TERRAIN_BASE_STEP);
    };

    size_t protectedTargetCount = 0;
    size_t protectedParentCount = 0;
    size_t rejectedIntermediateCount = 0;
    for (int64_t tileZ = ACTIVE.z - 6; tileZ <= REQUESTED.z + 7; ++tileZ) {
        for (int64_t tileX = ACTIVE.x - 6; tileX <= REQUESTED.x + 7; ++tileX) {
            for (const FarTerrainStep step : STEPS) {
                const FarTerrainKey key{tileX, tileZ, step};
                const bool expected = requiredBy(ACTIVE, key) || requiredBy(REQUESTED, key);
                CAPTURE(tileX, tileZ, static_cast<int>(step), expected);
                REQUIRE(farTerrainProtectedGpuResidencyRequired(key, ACTIVE, REQUESTED) ==
                        expected);
                protectedParentCount += expected && step == FAR_TERRAIN_BASE_STEP ? 1U : 0U;
                protectedTargetCount += expected && step != FAR_TERRAIN_BASE_STEP ? 1U : 0U;
                rejectedIntermediateCount +=
                    !expected && (farTerrainProtectedNearRole(ACTIVE, {tileX, tileZ}) !=
                                      FarTerrainProtectedNearRole::NONE ||
                                  farTerrainProtectedNearRole(REQUESTED, {tileX, tileZ}) !=
                                      FarTerrainProtectedNearRole::NONE)
                        ? 1U
                        : 0U;
            }
        }
    }
    REQUIRE(protectedTargetCount >= FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(protectedParentCount >= FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(rejectedIntermediateCount > 0);
}

TEST_CASE("Protected near production geometry is one continuous FINAL publication",
          "[render][far-terrain][lod][authority][entry][geometry][patch][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    constexpr ColumnPos ANCHOR{-4, 3};
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(
        static_cast<double>(ANCHOR.x * FAR_TERRAIN_TILE_EDGE + FAR_TERRAIN_TILE_EDGE / 2),
        static_cast<double>(ANCHOR.z * FAR_TERRAIN_TILE_EDGE + FAR_TERRAIN_TILE_EDGE / 2),
        FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    std::vector<FarTerrainKey> targets;
    buildFarTerrainProtectedNearTargets(ANCHOR, selected, targets);
    REQUIRE(targets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);

    const FarTerrainSource final = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight =
                112.0 + static_cast<double>(world_coord::floorMod(x + 3 * z, 31)) * 0.125;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    std::vector<FarTerrainProtectedNearSurface> surfaces;
    surfaces.reserve(targets.size());
    size_t stepOneCount = 0;
    size_t stepTwoCount = 0;
    size_t stepFourCount = 0;
    size_t stepEightCount = 0;
    size_t stepSixteenCount = 0;
    constexpr uint32_t RESERVED_PANEL_ATTRIBUTE = 1U << 29U;
    for (const FarTerrainKey target : targets) {
        const std::shared_ptr<const FarTerrainMesh> mesh =
            FarTerrainMesher::build(target, final, Quality::FINAL);
        const std::shared_ptr<const FarTerrainMesh> parent = FarTerrainMesher::build(
            {target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP}, final, Quality::FINAL);
        CAPTURE(target.tileX, target.tileZ, farTerrainStepSize(target.step), mesh->vertices.size(),
                mesh->indices.size());
        REQUIRE(mesh->authorityQuality == Quality::FINAL);
        REQUIRE(mesh->exactAuthorityCompatible);
        REQUIRE(parent->authorityQuality == Quality::FINAL);
        REQUIRE(mesh->surfaceBoundary.valid);
        REQUIRE_FALSE(mesh->vertices.empty());
        REQUIRE_FALSE(mesh->indices.empty());
        REQUIRE(std::ranges::none_of(mesh->vertices, [](const Vertex& vertex) {
            return (vertex.faceAttr & RESERVED_PANEL_ATTRIBUTE) != 0U;
        }));
        if (target.step == FarTerrainStep::ONE)
            ++stepOneCount;
        else if (target.step == FarTerrainStep::TWO)
            ++stepTwoCount;
        else if (target.step == FarTerrainStep::FOUR)
            ++stepFourCount;
        else if (target.step == FarTerrainStep::EIGHT)
            ++stepEightCount;
        else if (target.step == FarTerrainStep::SIXTEEN)
            ++stepSixteenCount;
        surfaces.push_back({
            .key = target,
            .authorityQuality = mesh->authorityQuality,
            .parentAuthorityQuality = parent->authorityQuality,
            .exactAuthorityCompatible = mesh->exactAuthorityCompatible,
            .surfaceBoundary = mesh->surfaceBoundary,
        });
    }
    REQUIRE(stepOneCount == FAR_TERRAIN_PROTECTED_NEAR_STEP_ONE_TILE_COUNT);
    REQUIRE(stepTwoCount == FAR_TERRAIN_PROTECTED_NEAR_STEP_TWO_TILE_COUNT);
    REQUIRE(stepFourCount == FAR_TERRAIN_PROTECTED_NEAR_STEP_FOUR_TILE_COUNT);
    REQUIRE(stepEightCount == FAR_TERRAIN_PROTECTED_NEAR_STEP_EIGHT_TILE_COUNT);
    REQUIRE(stepSixteenCount == FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_TILE_COUNT);

    const FarTerrainProtectedNearGeometryStatus complete =
        farTerrainProtectedNearGeometryStatus(ANCHOR, targets, surfaces);
    REQUIRE(complete.ready());
    REQUIRE(complete.presentTargets == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(complete.finalTargets == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(complete.finalParents == complete.expectedFinalParents);
    REQUIRE(complete.exactCompatibleTargets == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    REQUIRE(complete.expectedSharedBoundaries == 100);
    REQUIRE(complete.matchingSharedBoundaries == complete.expectedSharedBoundaries);
    REQUIRE(complete.mismatchedSharedBoundaries == 0);
    REQUIRE(complete.incompatibleLodBoundaries == 0);

    surfaces.front().authorityQuality = Quality::PREVIEW;
    REQUIRE_FALSE(farTerrainProtectedNearGeometryStatus(ANCHOR, targets, surfaces).ready());
    surfaces.front().authorityQuality = Quality::FINAL;
    surfaces.front().surfaceBoundary.heightHashes[0] ^= 1U;
    const FarTerrainProtectedNearGeometryStatus mismatched =
        farTerrainProtectedNearGeometryStatus(ANCHOR, targets, surfaces);
    REQUIRE_FALSE(mismatched.ready());
    REQUIRE(mismatched.mismatchedSharedBoundaries != 0);
    surfaces.erase(surfaces.begin());
    REQUIRE_FALSE(farTerrainProtectedNearGeometryStatus(ANCHOR, targets, surfaces).ready());
}

TEST_CASE("Protected near authority coalesces adjacent hydrology inputs without changing owners",
          "[render][far-terrain][authority][hydrology][entry][performance][regression]") {
    constexpr ColumnPos ANCHOR{-4, 3};
    const double cameraX =
        static_cast<double>(ANCHOR.x * FAR_TERRAIN_TILE_EDGE + FAR_TERRAIN_TILE_EDGE / 2);
    const double cameraZ =
        static_cast<double>(ANCHOR.z * FAR_TERRAIN_TILE_EDGE + FAR_TERRAIN_TILE_EDGE / 2);
    std::vector<FarTerrainViewTile> selected;
    selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, selected);
    std::vector<FarTerrainKey> targets;
    buildFarTerrainProtectedNearTargets(ANCHOR, selected, targets);
    REQUIRE(targets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);

    std::set<std::pair<int64_t, int64_t>> owners;
    std::vector<worldgen::learned::NativeRect> ownerRegions;
    for (const FarTerrainKey target : targets) {
        const FarTerrainFinalBaseAuthorityDependencies dependencies =
            farTerrainFinalBaseAuthorityDependencies(
                {target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
        for (const FarTerrainNativeHydrologyDependency& dependency : dependencies.nativeHydrology) {
            if (owners.emplace(dependency.ownerPageZ, dependency.ownerPageX).second)
                ownerRegions.push_back(dependency.finalTerrainRegion);
        }
    }
    const std::vector<worldgen::learned::NativeRect> grouped =
        farTerrainProtectedFinalTerrainRegions(targets);
    REQUIRE_FALSE(grouped.empty());
    REQUIRE(grouped.size() < ownerRegions.size());
    const auto contains = [](worldgen::learned::NativeRect outer,
                             worldgen::learned::NativeRect inner) {
        return outer.rowBegin <= inner.rowBegin && outer.columnBegin <= inner.columnBegin &&
               outer.rowEnd >= inner.rowEnd && outer.columnEnd >= inner.columnEnd;
    };
    for (const worldgen::learned::NativeRect owner : ownerRegions) {
        CAPTURE(owner.rowBegin, owner.columnBegin, owner.rowEnd, owner.columnEnd);
        REQUIRE(std::ranges::any_of(grouped, [&](worldgen::learned::NativeRect region) {
            return contains(region, owner);
        }));
    }
    for (const worldgen::learned::NativeRect region : grouped) {
        CAPTURE(region.rowBegin, region.columnBegin, region.rowEnd, region.columnEnd);
        REQUIRE(region.valid());
        REQUIRE(region.height() * region.width() <=
                worldgen::learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES);
        REQUIRE(region.height() <= 1'029);
        REQUIRE(region.width() <= 1'029);
    }

    std::ranges::reverse(targets);
    REQUIRE(farTerrainProtectedFinalTerrainRegions(targets) == grouped);
}

TEST_CASE("Protected near terrain covers exact overlap from every camera half-tile",
          "[render][far-terrain][lod][residency][entry][corner][negative][regression]") {
    constexpr std::array CAMERA_BLOCKS{
        ColumnPos{0, 0},       ColumnPos{127, 127},   ColumnPos{128, 128},   ColumnPos{255, 255},
        ColumnPos{256, 256},   ColumnPos{383, 383},   ColumnPos{384, 384},   ColumnPos{-1, -1},
        ColumnPos{-128, -128}, ColumnPos{-129, -129}, ColumnPos{-256, -256}, ColumnPos{-257, -257},
    };
    constexpr long double COVERAGE_RADIUS_BLOCKS =
        FAR_TERRAIN_ENTRY_PARENT_RADIUS_CHUNKS * static_cast<long double>(CHUNK_EDGE);
    constexpr long double COVERAGE_RADIUS_SQUARED = COVERAGE_RADIUS_BLOCKS * COVERAGE_RADIUS_BLOCKS;

    for (const ColumnPos cameraBlock : CAMERA_BLOCKS) {
        const double cameraX = static_cast<double>(cameraBlock.x) + 0.5;
        const double cameraZ = static_cast<double>(cameraBlock.z) + 0.5;
        const ColumnPos anchor = farTerrainProtectedNearAnchor(cameraBlock.x, cameraBlock.z);
        CAPTURE(cameraBlock.x, cameraBlock.z, anchor.x, anchor.z);

        std::vector<FarTerrainViewTile> exactOverlap;
        selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_NEAR_CHUNK_RADIUS, exactOverlap);
        REQUIRE_FALSE(exactOverlap.empty());
        for (const FarTerrainViewTile& tile : exactOverlap) {
            const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
            CAPTURE(coordinate.x, coordinate.z);
            REQUIRE(farTerrainProtectedNearRole(anchor, coordinate) !=
                    FarTerrainProtectedNearRole::NONE);
        }

        std::vector<FarTerrainViewTile> fullHorizon;
        selectFarTerrainView(cameraX, cameraZ, FAR_TERRAIN_MAX_CHUNK_RADIUS, fullHorizon);
        std::vector<FarTerrainKey> targets;
        buildFarTerrainProtectedNearTargets(anchor, fullHorizon, targets);
        REQUIRE(targets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
        for (const FarTerrainKey target : targets) {
            const long double minimumX =
                static_cast<long double>(target.tileX) * FAR_TERRAIN_TILE_EDGE;
            const long double minimumZ =
                static_cast<long double>(target.tileZ) * FAR_TERRAIN_TILE_EDGE;
            const long double maximumX = minimumX + FAR_TERRAIN_TILE_EDGE;
            const long double maximumZ = minimumZ + FAR_TERRAIN_TILE_EDGE;
            const long double farthestX =
                std::max(std::abs(minimumX - cameraX), std::abs(maximumX - cameraX));
            const long double farthestZ =
                std::max(std::abs(minimumZ - cameraZ), std::abs(maximumZ - cameraZ));
            const long double farthestSquared = farthestX * farthestX + farthestZ * farthestZ;
            CAPTURE(target.tileX, target.tileZ, farthestX, farthestZ, farthestSquared);
            REQUIRE(farthestSquared <= COVERAGE_RADIUS_SQUARED);
        }
    }
}

TEST_CASE("Resident nearby refinement keeps its coarse parent as coverage fallback",
          "[render][far-terrain][lod][residency][camera-jump][flicker][regression]") {
    FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE_FALSE(farTerrainInitialDisplayedStep(ready));

    ready |= farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::TWO);

    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
            farTerrainStepMask(FarTerrainStep::SIXTEEN) | farTerrainStepMask(FarTerrainStep::FOUR);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::FOUR);

    ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::THIRTY_TWO);

    REQUIRE(
        farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::EIGHT, ready));
    ready |= farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT, ready));
    ready |= farTerrainStepMask(FarTerrainStep::EIGHT);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::EIGHT);
    REQUIRE_FALSE(
        farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::EIGHT, ready));
    REQUIRE_FALSE(
        farTerrainDisplayedStepAllowed(FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT, ready));
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::EIGHT, FarTerrainStep::EIGHT, ready));
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::EIGHT, FarTerrainStep::TWO, ready));
    REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::THIRTY_TWO,
                                           ready));

    ready |= farTerrainStepMask(FarTerrainStep::TWO);
    REQUIRE_FALSE(
        farTerrainDisplayedStepAllowed(FarTerrainStep::EIGHT, FarTerrainStep::TWO, ready));
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::TWO);

    ready |= farTerrainStepMask(FarTerrainStep::ONE);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::ONE);
}

TEST_CASE("Progressive bridge residency cannot fall back to step thirty two",
          "[render][far-terrain][lod][residency][flicker][regression]") {
    const FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
                                     farTerrainStepMask(FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainInitialDisplayedStep(ready) == FarTerrainStep::SIXTEEN);
    REQUIRE(farTerrainRetainsProgressiveStep(FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
                                             FarTerrainStep::ONE));
    REQUIRE(farTerrainRetainsProgressiveStep(FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN,
                                             FarTerrainStep::ONE));
    REQUIRE_FALSE(farTerrainRetainsProgressiveStep(FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::SIXTEEN, FarTerrainStep::ONE));
    REQUIRE_FALSE(farTerrainRetainsProgressiveStep(FarTerrainStep::SIXTEEN, FarTerrainStep::ONE,
                                                   FarTerrainStep::ONE));
}

TEST_CASE("A ready near-camera bridge cannot toggle back to step thirty two",
          "[render][far-terrain][lod][residency][near-camera][flicker][regression]") {
    const FarTerrainStepMask ready = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
                                     farTerrainStepMask(FarTerrainStep::SIXTEEN) |
                                     farTerrainStepMask(FarTerrainStep::EIGHT) |
                                     farTerrainStepMask(FarTerrainStep::TWO);
    for (int frame = 0; frame < 64; ++frame) {
        CAPTURE(frame);
        REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                ready, false) == FarTerrainStep::SIXTEEN);
        REQUIRE_FALSE(
            farTerrainDisplayedStepAllowed(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, ready));
        REQUIRE(farTerrainDisplayedStepAllowed(FarTerrainStep::TWO, FarTerrainStep::TWO, ready));
    }

    std::array<std::optional<FarTerrainStep>, 4> compatibleNeighbors{
        FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::TWO, FarTerrainStep::FOUR};
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::TWO, compatibleNeighbors));
    compatibleNeighbors[0] = FarTerrainStep::SIXTEEN;
    REQUIRE_FALSE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::TWO, compatibleNeighbors));
    const std::array<std::optional<FarTerrainStep>, 4> bridgeNeighbors{
        FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN,
        FarTerrainStep::EIGHT};
    REQUIRE(farTerrainStepCompatibleWithNeighbors(FarTerrainStep::EIGHT, bridgeNeighbors));
}

TEST_CASE("Sub-tile motion cannot downgrade a resident final refinement",
          "[render][far-terrain][lod][selection][residency][flicker][regression]") {
    constexpr ColumnPos TRACKED{0, 0};
    std::vector<FarTerrainViewTile> firstSelection;
    std::vector<FarTerrainViewTile> movedSelection;
    selectFarTerrainView(32.0, 32.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, firstSelection);
    selectFarTerrainView(224.0, 224.0, FAR_TERRAIN_MAX_CHUNK_RADIUS, movedSelection);
    const auto trackedStep = [](const std::vector<FarTerrainViewTile>& selection) {
        const auto found = std::ranges::find_if(selection, [](const FarTerrainViewTile& tile) {
            return tile.key.tileX == TRACKED.x && tile.key.tileZ == TRACKED.z;
        });
        REQUIRE(found != selection.end());
        return found->key.step;
    };
    REQUIRE(trackedStep(firstSelection) == FarTerrainStep::TWO);
    REQUIRE(trackedStep(movedSelection) == FarTerrainStep::TWO);
    REQUIRE(farTerrainStepForMetrics(50.0, FarTerrainStep::ONE) == FarTerrainStep::TWO);

    const FarTerrainStepMask allReady =
        farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
        farTerrainStepMask(FarTerrainStep::SIXTEEN) | farTerrainStepMask(FarTerrainStep::EIGHT) |
        farTerrainStepMask(FarTerrainStep::FOUR) | farTerrainStepMask(FarTerrainStep::TWO);
    // Even if display bookkeeping is reconstructed, the resident fine mesh
    // remains the initial surface. An intentional transition from an older
    // displayed tier still advances through one adjacent topology at a time.
    REQUIRE(farTerrainInitialDisplayedStep(allReady) == FarTerrainStep::TWO);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::EIGHT, FarTerrainStep::TWO, allReady,
                                            false) == FarTerrainStep::FOUR);
    REQUIRE(farTerrainReadyTransitionTarget(FarTerrainStep::FOUR, FarTerrainStep::TWO, allReady,
                                            false) == FarTerrainStep::TWO);

    std::vector<FarTerrainViewTile> trackedSelection(1);
    trackedSelection.front().key = {TRACKED.x, TRACKED.z, FarTerrainStep::ONE};
    std::vector<FarTerrainKey> residencyOrder;
    buildFarTerrainResidencyOrder(trackedSelection, residencyOrder);
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(residencyOrder.begin(),
                                                                      residencyOrder.end());
    REQUIRE(farTerrainResidencyMembershipMatches(trackedSelection, wanted));
    for (const FarTerrainStep step :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        REQUIRE(wanted.contains({TRACKED.x, TRACKED.z, step}));
    }
}

TEST_CASE("Every unresolved exact-loading tile requests a protected preview bridge first",
          "[render][far-terrain][coverage][lod][priority][exact][regression]") {
    constexpr size_t PROTECTED_TILE_COUNT = 24;
    constexpr size_t BLOCK_SCALE_TILE_COUNT = 4;
    std::array<FarTerrainRefinementCacheRequest, PROTECTED_TILE_COUNT> requests{};
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index] = {{static_cast<int64_t>(index), 0},
                           FarTerrainStep::THIRTY_TWO,
                           FarTerrainStep::TWO,
                           0,
                           false,
                           false,
                           true,
                           index < BLOCK_SCALE_TILE_COUNT};
        requests[index].protectedNearTarget = true;
    }
    std::vector<FarTerrainKey> order;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);

    REQUIRE(order.size() == PROTECTED_TILE_COUNT);
    for (size_t index = 0; index < PROTECTED_TILE_COUNT; ++index) {
        CAPTURE(index, static_cast<int>(order[index].step));
        REQUIRE(order[index].tileX == static_cast<int64_t>(index));
        REQUIRE(order[index].step == FarTerrainStep::SIXTEEN);
    }

    requests.front().residentSteps = farTerrainStepMask(FarTerrainStep::THIRTY_TWO) |
                                     farTerrainStepMask(FarTerrainStep::SIXTEEN);
    requests.front().displayed = FarTerrainStep::SIXTEEN;
    buildFarTerrainProgressiveSubmissionOrder(requests, order);
    REQUIRE((order.front() == FarTerrainKey{0, 0, FarTerrainStep::EIGHT}));

    auto transitionLimited = requests;
    REQUIRE(reserveFarTerrainIntermediateTransitionSlots(
                transitionLimited, FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS) == 0);
    REQUIRE(std::ranges::none_of(transitionLimited,
                                 [](const auto& request) { return request.deferIntermediate; }));

    std::vector<FarTerrainViewTile> selected{
        {{0, 0, FarTerrainStep::TWO}, {}, 0.0, 32.0, {}},
        {{1, 0, FarTerrainStep::TWO}, {}, 256.0 * 256.0, 32.0, {}},
        {{2, 0, FarTerrainStep::TWO}, {}, 512.0 * 512.0, 32.0, {}},
    };
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> resident;
    for (const FarTerrainViewTile& tile : selected) {
        resident.insert({tile.key.tileX, tile.key.tileZ, FarTerrainStep::THIRTY_TWO});
    }
    const auto drawable = [&](FarTerrainKey base) {
        if (!resident.contains(base))
            return false;
        if (base.tileX != 0)
            return true;
        return resident.contains({base.tileX, base.tileZ, FarTerrainStep::TWO});
    };
    FarTerrainCoverageFrontier frontier = farTerrainCoverageFrontier(selected, drawable);
    REQUIRE(frontier.missingBaseTiles == 1);
    REQUIRE_FALSE(farTerrainCoverageDrawEligible(selected[1].distanceSquared, frontier));
    resident.insert({0, 0, FarTerrainStep::TWO});
    frontier = farTerrainCoverageFrontier(selected, drawable);
    REQUIRE(frontier.complete);
}

TEST_CASE("Cold nearby parents coalesce refinements for a bounded interval",
          "[render][far-terrain][lod][residency][priority][cold-start][camera-jump][regression]") {
    REQUIRE(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS <= 0.12F);
    REQUIRE(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS < FAR_TERRAIN_LOD_TRANSITION_SECONDS);
    REQUIRE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                            FarTerrainStep::SIXTEEN, 0.0F));
    REQUIRE(farTerrainDeferNearIntermediate(
        FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, FarTerrainStep::FOUR,
        std::nextafter(FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS, 0.0F)));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  FarTerrainStep::SIXTEEN,
                                                  FAR_TERRAIN_NEAR_REFINEMENT_GRACE_SECONDS));

    // The final requested tier bypasses the grace, and a tile that has
    // already refined never regresses to a coarser placeholder after travel.
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO,
                                                  FarTerrainStep::TWO, 0.0F));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::SIXTEEN, FarTerrainStep::TWO,
                                                  FarTerrainStep::FOUR, 0.0F));
    REQUIRE_FALSE(farTerrainDeferNearIntermediate(FarTerrainStep::THIRTY_TWO, FarTerrainStep::FOUR,
                                                  FarTerrainStep::SIXTEEN, 0.0F));
}

TEST_CASE("Far trees exchange monotonically through exact and LOD transitions",
          "[render][far-terrain][canopy][lod][transition][ownership][flicker][regression]") {
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    for (const float threshold : {0.1F, 0.33F, 0.67F, 0.9F}) {
        bool sourcePreviouslyVisible = true;
        bool targetPreviouslyVisible = false;
        size_t sourceChanges = 0;
        size_t targetChanges = 0;
        for (int tick = 0; tick <= 100; ++tick) {
            const float elapsed =
                FAR_TERRAIN_LOD_TRANSITION_SECONDS * static_cast<float>(tick) / 100.0F;
            const float progress = sampleFarTerrainTransition(elapsed).progress;
            const bool sourceVisible = farTerrainLodCanopyVisible(progress, threshold, SOURCE);
            const bool targetVisible = farTerrainLodCanopyVisible(progress, threshold, TARGET);
            CAPTURE(threshold, tick, progress, sourceVisible, targetVisible);
            REQUIRE((sourceVisible || targetVisible));
            REQUIRE_FALSE((!sourcePreviouslyVisible && sourceVisible));
            REQUIRE_FALSE((targetPreviouslyVisible && !targetVisible));
            sourceChanges += sourceVisible != sourcePreviouslyVisible ? 1 : 0;
            targetChanges += targetVisible != targetPreviouslyVisible ? 1 : 0;
            sourcePreviouslyVisible = sourceVisible;
            targetPreviouslyVisible = targetVisible;
        }
        REQUIRE(sourceChanges == 1);
        REQUIRE(targetChanges == 1);
        REQUIRE_FALSE(farTerrainLodCanopyVisible(1.0F, threshold, SOURCE));
        REQUIRE(farTerrainLodCanopyVisible(1.0F, threshold, TARGET));
        REQUIRE(farTerrainLodCanopyVisible(0.5F, threshold, SOURCE));
        REQUIRE(farTerrainLodCanopyVisible(0.5F, threshold, TARGET));
    }

    // The nearby step-32 emergency parent swaps its terrain earlier, but its
    // canopy still uses the same overlap contract for the full transition.
    constexpr unsigned int EMERGENCY_SOURCE = SOURCE | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    constexpr unsigned int EMERGENCY_TARGET = TARGET | FAR_TERRAIN_LOD_EMERGENCY_FLAG;
    for (const float threshold : {0.05F, 0.25F, 0.5F, 0.75F, 0.95F}) {
        bool sourceVisibleLast = true;
        bool targetVisibleLast = false;
        for (int tick = 0; tick <= 100; ++tick) {
            const float progress = sampleFarTerrainTransition(FAR_TERRAIN_LOD_TRANSITION_SECONDS *
                                                              static_cast<float>(tick) / 100.0F)
                                       .progress;
            const bool sourceVisible =
                farTerrainLodCanopyVisible(progress, threshold, EMERGENCY_SOURCE);
            const bool targetVisible =
                farTerrainLodCanopyVisible(progress, threshold, EMERGENCY_TARGET);
            CAPTURE(threshold, tick, progress, sourceVisible, targetVisible);
            REQUIRE((sourceVisible || targetVisible));
            REQUIRE_FALSE((!sourceVisibleLast && sourceVisible));
            REQUIRE_FALSE((targetVisibleLast && !targetVisible));
            sourceVisibleLast = sourceVisible;
            targetVisibleLast = targetVisible;
        }
    }

    bool exactOwned = false;
    bool previouslyExactOwned = false;
    for (const auto [built, current] :
         {std::pair{4U, 5U}, std::pair{5U, 5U}, std::pair{5U, 6U}, std::pair{6U, 7U}}) {
        exactOwned = farTerrainExactSectionOwnsSurface(exactOwned, built, current);
        const bool farOwned = !exactOwned;
        CAPTURE(built, current, exactOwned, farOwned);
        REQUIRE(exactOwned != farOwned);
        REQUIRE(static_cast<unsigned int>(exactOwned) + static_cast<unsigned int>(farOwned) == 1U);
        REQUIRE_FALSE((previouslyExactOwned && !exactOwned));
        if (built == current)
            REQUIRE(exactOwned);
        previouslyExactOwned = exactOwned;
    }
    REQUIRE(exactOwned);
}

TEST_CASE("A late target canopy retains the prior LOD without delaying terrain",
          "[render][far-terrain][canopy][lod][transition][fallback][regression]") {
    STATIC_REQUIRE(farCanopyLodCompletionAction(false, true, false) ==
                   FarCanopyLodCompletionAction::ADOPT_SOURCE);
    STATIC_REQUIRE(farCanopyLodCompletionAction(true, false, false) ==
                   FarCanopyLodCompletionAction::RETAIN_FALLBACK);
    STATIC_REQUIRE(farCanopyLodCompletionAction(true, true, true) ==
                   FarCanopyLodCompletionAction::RETIRE_FALLBACK);
    STATIC_REQUIRE(farCanopyLodCompletionAction(false, false, true) ==
                   FarCanopyLodCompletionAction::NONE);
    STATIC_REQUIRE(farCanopyLodTargetUsesSourceFallback(true, false));
    STATIC_REQUIRE_FALSE(farCanopyLodTargetUsesSourceFallback(false, false));
    STATIC_REQUIRE_FALSE(farCanopyLodTargetUsesSourceFallback(true, true));

    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    for (const float threshold : {0.1F, 0.33F, 0.67F, 0.9F}) {
        for (int tick = 0; tick <= 100; ++tick) {
            const float elapsed =
                FAR_TERRAIN_LOD_TRANSITION_SECONDS * static_cast<float>(tick) / 100.0F;
            const float progress = sampleFarTerrainTransition(elapsed).progress;
            const bool sourceVisible = farTerrainLodCanopyVisible(progress, threshold, SOURCE);
            // While the real target is late, its draw slot reuses the source
            // allocation. The ordinary target-in/source-out dither therefore
            // remains nonempty without gating the terrain transition.
            const bool sourceFallbackVisible =
                farCanopyLodTargetUsesSourceFallback(true, false) &&
                farTerrainLodCanopyVisible(progress, threshold, TARGET);
            CAPTURE(threshold, tick, progress, sourceVisible, sourceFallbackVisible);
            REQUIRE((sourceVisible || sourceFallbackVisible));
        }
    }
}

TEST_CASE("Far ownership requires every exact surface section in a column",
          "[render][far-terrain][coverage][exact][ownership][revision]") {
    constexpr std::array required{
        ChunkPos{0, 3, 0},   ChunkPos{0, 4, 0},   ChunkPos{1, 3, 0},     ChunkPos{1, 4, 0},
        ChunkPos{15, 3, 15}, ChunkPos{-1, 3, -1}, ChunkPos{-16, 3, -16}, ChunkPos{-17, 3, -17},
    };
    std::unordered_set<ChunkPos> ready(required.begin(), required.end());
    ready.erase(ChunkPos{1, 4, 0});
    constexpr std::array unresolved{ColumnPos{15, 15}};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved,
                               [&](ChunkPos position) { return ready.contains(position); });

    REQUIRE(handoff.columnFullyReady({0, 0}));
    REQUIRE_FALSE(handoff.columnFullyReady({1, 0}));
    REQUIRE_FALSE(handoff.columnFullyReady({15, 15}));
    REQUIRE(handoff.columnFullyReady({-1, -1}));
    REQUIRE(handoff.columnFullyReady({-16, -16}));
    REQUIRE(handoff.columnFullyReady({-17, -17}));

    const FarTerrainExactHandoff::ColumnMask positive = handoff.readyColumnMask({0, 0});
    REQUIRE((positive[0] & 1U) != 0U);
    REQUIRE((positive[0] & (1U << 1U)) == 0U);
    REQUIRE((positive[7] & (1U << 31U)) == 0U);

    // Floor division maps negative boundaries to the same far tile and mask
    // bit used by generation, streaming, and the fragment lookup.
    const FarTerrainExactHandoff::ColumnMask negativeOne = handoff.readyColumnMask({-1, -1});
    REQUIRE((negativeOne[7] & (1U << 31U)) != 0U);
    REQUIRE((negativeOne[0] & 1U) != 0U);
    const FarTerrainExactHandoff::ColumnMask negativeTwo = handoff.readyColumnMask({-2, -2});
    REQUIRE((negativeTwo[7] & (1U << 31U)) != 0U);
}

TEST_CASE("Submerged exact columns retain their parent until floor and water are ready",
          "[render][far-terrain][coverage][exact][ownership][water][floor][regression]") {
    // Keep the fixture pinned to a genuinely deep generated-water column so
    // the test exercises independent floor and water readiness.
    constexpr int64_t WORLD_X = -8'348;
    constexpr int64_t WORLD_Z = 2'281;
    const ColumnPos column{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunk(WORLD_Z)};
    ChunkGenerator generator(42);
    const auto plan = generator.getColumnPlan(column);
    const int localX = Chunk::worldToLocal(WORLD_X);
    const int localZ = Chunk::worldToLocal(WORLD_Z);
    const int32_t floorSection = Chunk::worldToChunkY(plan->surfaceY(localX, localZ));
    const worldgen::SurfaceSample surface = generator.sampleSurface(WORLD_X, WORLD_Z);
    REQUIRE((surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake ||
             surface.hydrology.wetland));
    const int32_t waterSection =
        Chunk::worldToChunkY(static_cast<int>(std::ceil(surface.waterSurface)) - 1);
    REQUIRE(waterSection > floorSection);
    REQUIRE(std::ranges::find(plan->surfaceOwnershipSections(), floorSection) !=
            plan->surfaceOwnershipSections().end());
    REQUIRE(std::ranges::find(plan->surfaceOwnershipSections(), waterSection) !=
            plan->surfaceOwnershipSections().end());

    std::vector<ChunkPos> required;
    required.reserve(plan->surfaceOwnershipSections().size());
    for (const int32_t section : plan->surfaceOwnershipSections()) {
        required.push_back({column.x, section, column.z});
    }
    const auto handoffWithMissing = [&](int32_t missingSection) {
        return farTerrainExactHandoff(
            static_cast<double>(WORLD_X), static_cast<double>(WORLD_Z), 32, required, {},
            [&](ChunkPos position) { return position.y != missingSection; });
    };
    REQUIRE_FALSE(handoffWithMissing(floorSection).columnFullyReady(column));
    REQUIRE_FALSE(handoffWithMissing(waterSection).columnFullyReady(column));
    REQUIRE(farTerrainExactHandoff(static_cast<double>(WORLD_X), static_cast<double>(WORLD_Z), 32,
                                   required, {}, [](ChunkPos) { return true; })
                .columnFullyReady(column));
}

TEST_CASE("Optional exact support does not retain grass and water parents over a ready surface",
          "[render][far-terrain][coverage][exact][ownership][overlap][regression]") {
    ChunkGenerator generator(42);
    std::shared_ptr<const ColumnPlan> selected;
    ColumnPos selectedColumn{};
    int32_t optionalSection = 0;
    for (int64_t z = -4; z <= 4 && !selected; ++z) {
        for (int64_t x = -4; x <= 4 && !selected; ++x) {
            const std::shared_ptr<const ColumnPlan> plan = generator.getColumnPlan({x, z});
            REQUIRE_FALSE(plan->surfaceOwnershipSections().empty());
            REQUIRE(
                std::ranges::includes(plan->exposedSections(), plan->surfaceOwnershipSections()));
            const auto optional = std::ranges::find_if(plan->exposedSections(), [&](int32_t value) {
                return !std::ranges::binary_search(plan->surfaceOwnershipSections(), value);
            });
            if (optional != plan->exposedSections().end()) {
                selected = plan;
                selectedColumn = {x, z};
                optionalSection = *optional;
            }
        }
    }
    REQUIRE(selected);

    std::vector<ChunkPos> ownershipRequirements;
    for (const int32_t section : selected->surfaceOwnershipSections())
        ownershipRequirements.push_back({selectedColumn.x, section, selectedColumn.z});
    const FarTerrainExactHandoff ownership =
        farTerrainExactHandoff(selectedColumn.x * CHUNK_EDGE + CHUNK_EDGE / 2.0,
                               selectedColumn.z * CHUNK_EDGE + CHUNK_EDGE / 2.0, 32,
                               ownershipRequirements, {}, [](ChunkPos) { return true; });
    REQUIRE(ownership.columnFullyReady(selectedColumn));

    // The previous handoff contract included optional tree and generation
    // support. One absent support section kept both coarse terrain and water
    // drawable even though every real surface owner was already resident.
    std::vector<ChunkPos> legacyRequirements;
    for (const int32_t section : selected->exposedSections())
        legacyRequirements.push_back({selectedColumn.x, section, selectedColumn.z});
    const FarTerrainExactHandoff legacy = farTerrainExactHandoff(
        selectedColumn.x * CHUNK_EDGE + CHUNK_EDGE / 2.0,
        selectedColumn.z * CHUNK_EDGE + CHUNK_EDGE / 2.0, 32, legacyRequirements, {},
        [&](ChunkPos section) { return section.y != optionalSection; });
    REQUIRE_FALSE(legacy.columnFullyReady(selectedColumn));
}

TEST_CASE("Exact and far ownership select one surface per ready column",
          "[render][far-terrain][coverage][exact][ownership]") {
    std::vector<ChunkPos> required;
    required.reserve(16 * 16 * 2);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            required.push_back({x, 3, z});
            required.push_back({x, 4, z});
        }
    }
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(128.0, 128.0, 32, required, {}, [](ChunkPos) { return true; });
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            const bool exactOwner = handoff.columnFullyReady({x, z});
            const bool farOwner = !exactOwner;
            CAPTURE(x, z);
            REQUIRE(exactOwner);
            REQUIRE(exactOwner != farOwner);
        }
    }
}

TEST_CASE("The ready camera column masks every far LOD underfoot",
          "[render][far-terrain][coverage][exact][ownership][camera][lod][regression]") {
    constexpr double CAMERA_X = -198.692;
    constexpr double CAMERA_Z = 63.7348;
    constexpr ColumnPos CAMERA_COLUMN{-13, 3};
    constexpr ColumnPos CAMERA_TILE{-1, 0};
    constexpr std::array REQUIRED{
        ChunkPos{CAMERA_COLUMN.x, 4, CAMERA_COLUMN.z},
        ChunkPos{CAMERA_COLUMN.x, 5, CAMERA_COLUMN.z},
        ChunkPos{CAMERA_COLUMN.x + 1, 4, CAMERA_COLUMN.z},
    };
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(CAMERA_X, CAMERA_Z, 32, REQUIRED, {}, [](ChunkPos) { return true; });
    REQUIRE(handoff.columnFullyReady(CAMERA_COLUMN));

    const FarTerrainExactHandoff::ColumnMask mask = handoff.readyColumnMask(CAMERA_TILE);
    constexpr uint32_t LOCAL_X = 3;
    constexpr uint32_t LOCAL_Z = 3;
    constexpr uint32_t BIT = LOCAL_Z * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE + LOCAL_X;
    REQUIRE((mask[BIT / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD] &
             (1U << (BIT % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD))) != 0U);

    // Match the shader's half-open lookup at the camera's horizontal sample.
    // Every far tier shares this ownership mask, so step 32 cannot overwrite
    // a published exact surface in the chunk the player occupies.
    const float localX = static_cast<float>(CAMERA_X - CAMERA_TILE.x * FAR_TERRAIN_TILE_EDGE);
    const float localZ = static_cast<float>(CAMERA_Z - CAMERA_TILE.z * FAR_TERRAIN_TILE_EDGE);
    const uint32_t sampledColumnX = static_cast<uint32_t>(std::floor(localX / CHUNK_EDGE));
    const uint32_t sampledColumnZ = static_cast<uint32_t>(std::floor(localZ / CHUNK_EDGE));
    REQUIRE(sampledColumnX == LOCAL_X);
    REQUIRE(sampledColumnZ == LOCAL_Z);
    const bool farTierVisibleAtCamera = !handoff.columnFullyReady(CAMERA_COLUMN);
    REQUIRE_FALSE(farTierVisibleAtCamera);
}

TEST_CASE("Far terrain risers query exact ownership from their emitting column",
          "[render][far-terrain][coverage][exact][ownership][riser][shader-contract][regression]") {
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::PLUS_X) == FAR_TERRAIN_FACE_PLUS_X);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::MINUS_X) == FAR_TERRAIN_FACE_MINUS_X);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::PLUS_Z) == FAR_TERRAIN_FACE_PLUS_Z);
    STATIC_REQUIRE(static_cast<uint8_t>(FaceNormal::MINUS_Z) == FAR_TERRAIN_FACE_MINUS_Z);

    struct OwnershipCell {
        int tileX;
        int tileZ;
        int columnX;
        int columnZ;

        bool operator==(const OwnershipCell&) const = default;
    };
    const auto lookup = [](simd_float2 position) {
        const int tileX = static_cast<int>(
            std::floor(position.x / static_cast<float>(FAR_TERRAIN_TILE_EDGE_BLOCKS)));
        const int tileZ = static_cast<int>(
            std::floor(position.y / static_cast<float>(FAR_TERRAIN_TILE_EDGE_BLOCKS)));
        const float tileLocalX =
            position.x - static_cast<float>(tileX) * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        const float tileLocalZ =
            position.y - static_cast<float>(tileZ) * FAR_TERRAIN_TILE_EDGE_BLOCKS;
        return OwnershipCell{
            tileX,
            tileZ,
            std::clamp(
                static_cast<int>(std::floor(tileLocalX / FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS)), 0,
                FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1),
            std::clamp(
                static_cast<int>(std::floor(tileLocalZ / FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS)), 0,
                FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1),
        };
    };
    struct Fixture {
        FaceNormal face;
        simd_float2 position;
        OwnershipCell emittingCell;
        OwnershipCell halfOpenCell;
    };
    const std::array fixtures = {
        Fixture{FaceNormal::PLUS_X, simd_make_float2(16.0F, 8.0F), {0, 0, 0, 0}, {0, 0, 1, 0}},
        Fixture{FaceNormal::MINUS_X, simd_make_float2(16.0F, 8.0F), {0, 0, 1, 0}, {0, 0, 1, 0}},
        Fixture{FaceNormal::PLUS_X, simd_make_float2(256.0F, 8.0F), {0, 0, 15, 0}, {1, 0, 0, 0}},
        Fixture{FaceNormal::MINUS_X, simd_make_float2(256.0F, 8.0F), {1, 0, 0, 0}, {1, 0, 0, 0}},
        Fixture{FaceNormal::PLUS_Z, simd_make_float2(8.0F, 16.0F), {0, 0, 0, 0}, {0, 0, 0, 1}},
        Fixture{FaceNormal::MINUS_Z, simd_make_float2(8.0F, 16.0F), {0, 0, 0, 1}, {0, 0, 0, 1}},
        Fixture{FaceNormal::PLUS_Z, simd_make_float2(8.0F, 256.0F), {0, 0, 0, 15}, {0, 1, 0, 0}},
        Fixture{FaceNormal::MINUS_Z, simd_make_float2(8.0F, 256.0F), {0, 1, 0, 0}, {0, 1, 0, 0}},
    };

    for (const Fixture& fixture : fixtures) {
        const unsigned int face = static_cast<unsigned int>(fixture.face);
        CAPTURE(face, fixture.position.x, fixture.position.y);
        REQUIRE(farTerrainOpaqueRiserUsesEmittingColumn(face, false));
        const simd_float2 emittingSample =
            farTerrainExactOwnershipSamplePosition(fixture.position, face, true);
        REQUIRE(lookup(emittingSample) == fixture.emittingCell);

        // Tops and water pass false directly. Canopies are excluded by the
        // shared classifier, so all retain half-open destination ownership at
        // chunk and tile boundaries.
        const simd_float2 halfOpenSample =
            farTerrainExactOwnershipSamplePosition(fixture.position, face, false);
        REQUIRE(halfOpenSample.x == fixture.position.x);
        REQUIRE(halfOpenSample.y == fixture.position.y);
        REQUIRE(lookup(halfOpenSample) == fixture.halfOpenCell);
        REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(face, true));
    }

    constexpr unsigned int TOP_FACE = static_cast<unsigned int>(FaceNormal::PLUS_Y);
    REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(TOP_FACE, false));
    const simd_float2 topPosition = simd_make_float2(16.0F, 256.0F);
    const simd_float2 topSample = farTerrainExactOwnershipSamplePosition(
        topPosition, TOP_FACE, farTerrainOpaqueRiserUsesEmittingColumn(TOP_FACE, false));
    REQUIRE(topSample.x == topPosition.x);
    REQUIRE(topSample.y == topPosition.y);
}

TEST_CASE("Seed 764891 coarse protrusions are hidden by exact column ownership",
          "[render][far-terrain][coverage][exact][ownership][regression]") {
    constexpr FarTerrainKey KEY{91, -437, FarTerrainStep::SIXTEEN};
    constexpr int64_t TILE_ORIGIN_X = KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t TILE_ORIGIN_Z = KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    std::array<float, 17 * 17> heights{};
    for (int sampleZ = 0; sampleZ <= 16; ++sampleZ) {
        for (int sampleX = 0; sampleX <= 16; ++sampleX) {
            const FarSurfaceSample sample =
                source.sample(TILE_ORIGIN_X + sampleX * 16, TILE_ORIGIN_Z + sampleZ * 16,
                              worldgen::SurfaceFootprint::BLOCK_16);
            heights[static_cast<size_t>(sampleZ * 17 + sampleX)] =
                static_cast<float>(static_cast<float16_t>(sample.footprintMinimumTerrainHeight));
        }
    }

    std::vector<ChunkPos> required;
    required.reserve(16 * 16);
    for (int64_t z = 0; z < 16; ++z) {
        for (int64_t x = 0; x < 16; ++x) {
            required.push_back({KEY.tileX * 16 + x, 14, KEY.tileZ * 16 + z});
        }
    }
    const FarTerrainExactHandoff handoff = farTerrainExactHandoff(
        23'029.0, -111'726.0, 32, required, {}, [](ChunkPos) { return true; });

    size_t rawProtrusions = 0;
    size_t visibleProtrusions = 0;
    for (int cellZ = 0; cellZ < 16; ++cellZ) {
        for (int cellX = 0; cellX < 16; ++cellX) {
            const float northwest = heights[static_cast<size_t>(cellZ * 17 + cellX)];
            const float northeast = heights[static_cast<size_t>(cellZ * 17 + cellX + 1)];
            const float southwest = heights[static_cast<size_t>((cellZ + 1) * 17 + cellX)];
            const float southeast = heights[static_cast<size_t>((cellZ + 1) * 17 + cellX + 1)];
            const ColumnPos chunkColumn{KEY.tileX * 16 + cellX, KEY.tileZ * 16 + cellZ};
            for (int blockZ = 0; blockZ < 16; ++blockZ) {
                for (int blockX = 0; blockX < 16; ++blockX) {
                    const float u = (static_cast<float>(blockX) + 0.5F) / 16.0F;
                    const float v = (static_cast<float>(blockZ) + 0.5F) / 16.0F;
                    const float coarse =
                        u <= v
                            ? northwest + v * (southwest - northwest) + u * (southeast - southwest)
                            : northwest + u * (northeast - northwest) + v * (southeast - northeast);
                    const int64_t worldX = TILE_ORIGIN_X + cellX * 16 + blockX;
                    const int64_t worldZ = TILE_ORIGIN_Z + cellZ * 16 + blockZ;
                    const double exact =
                        generator->sampleExactSurface(worldX, worldZ).terrainHeight;
                    if (coarse <= exact + 1.0e-5)
                        continue;
                    ++rawProtrusions;
                    if (!handoff.columnFullyReady(chunkColumn))
                        ++visibleProtrusions;
                }
            }
        }
    }
    INFO("raw coarse protrusions " << rawProtrusions);
    REQUIRE(visibleProtrusions == 0);
}

TEST_CASE("Far terrain exact handoff uses unresolved column AABBs",
          "[render][far-terrain][coverage][exact]") {
    REQUIRE(farTerrainColumnDistanceSquared(8.0, 8.0, {0, 0}) == 0.0);
    REQUIRE(farTerrainColumnDistanceSquared(-1.0, -1.0, {0, 0}) == Catch::Approx(2.0));
    REQUIRE(farTerrainColumnDistanceSquared(40.0, 8.0, {1, 0}) == Catch::Approx(64.0));
    REQUIRE(farTerrainColumnDistanceSquared(-40.0, -8.0, {-2, -1}) == Catch::Approx(64.0));
}

TEST_CASE("Ready exact tiles retain ownership when another tile is stale",
          "[render][far-terrain][coverage][exact][flicker][regression]") {
    constexpr std::array required = {
        ChunkPos{0, 4, 0},  ChunkPos{1, 4, 0},  ChunkPos{-1, 4, -1},
        ChunkPos{16, 4, 0}, ChunkPos{17, 4, 0}, ChunkPos{32, 4, 0},
    };
    const std::unordered_set<ChunkPos> ready = {
        required[0], required[1], required[2], required[3], required[5],
    };
    constexpr std::array unresolved = {ColumnPos{48, 0}};
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(0.0, 0.0, 32, required, unresolved,
                               [&](ChunkPos position) { return ready.contains(position); });

    REQUIRE(handoff.tileFullyReady({0, 0}));
    REQUIRE(handoff.tileFullyReady({-1, -1}));
    REQUIRE_FALSE(handoff.tileFullyReady({1, 0}));
    REQUIRE(handoff.tileFullyReady({2, 0}));
    REQUIRE_FALSE(handoff.tileFullyReady({3, 0}));
    REQUIRE_FALSE(handoff.tileFullyReady({4, 0}));

    constexpr float NOMINAL = 32.0F * CHUNK_EDGE;
    REQUIRE(handoff.distanceBlocksForTile({0, 0}, NOMINAL) == NOMINAL);
    REQUIRE(handoff.distanceBlocksForTile({2, 0}, NOMINAL) == NOMINAL);
    REQUIRE(handoff.distanceBlocksForTile({1, 0}, NOMINAL) == handoff.distanceBlocks);
    REQUIRE(handoff.distanceBlocksForTile({3, 0}, NOMINAL) == 48.0F * CHUNK_EDGE);
    REQUIRE(handoff.distanceBlocksForTile({4, 0}, NOMINAL) == handoff.distanceBlocks);

    REQUIRE_FALSE(handoff.tileFullyOwned({0, 0}));
    REQUIRE_FALSE(handoff.tileFullyOwned({2, 0}));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {0, 0}, NOMINAL, handoff));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, NOMINAL, handoff));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {2, 0}, NOMINAL, handoff));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {4, 0}, NOMINAL, handoff));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, 0.0F, handoff));
}

TEST_CASE("Only complete exact tile ownership releases fine boundary fallback",
          "[render][far-terrain][coverage][exact][lod][flicker][regression]") {
    std::vector<ChunkPos> completeTile;
    completeTile.reserve(FAR_TERRAIN_EXACT_COLUMNS_PER_TILE * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
    for (int64_t z = 0; z < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++z) {
        for (int64_t x = 0; x < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++x) {
            completeTile.push_back({x, 4, z});
        }
    }
    const FarTerrainExactHandoff complete =
        farTerrainExactHandoff(0.0, 0.0, 32, completeTile, {}, [](ChunkPos) { return true; });
    REQUIRE(complete.tileFullyReady({0, 0}));
    REQUIRE(complete.tileFullyOwned({0, 0}));
    REQUIRE_FALSE(farTerrainRequiresCoverageParent(0.0, 0.0, {0, 0}, 32.0F * CHUNK_EDGE, complete));

    constexpr std::array partialBoundary{ChunkPos{31, 4, 0}};
    const FarTerrainExactHandoff partial =
        farTerrainExactHandoff(0.0, 0.0, 32, partialBoundary, {}, [](ChunkPos) { return true; });
    REQUIRE(partial.tileFullyReady({1, 0}));
    REQUIRE_FALSE(partial.tileFullyOwned({1, 0}));
    REQUIRE(farTerrainRequiresCoverageParent(0.0, 0.0, {1, 0}, 32.0F * CHUNK_EDGE, partial));
}

TEST_CASE("Far terrain scheduler defaults match the physical-core startup topology",
          "[render][far-terrain][scheduler][coverage]") {
    // Cold coverage-first startup can use every terrain worker for base
    // parents. Optional canopy work is a separate, lower-priority two-worker
    // lane and the renderer holds that budget at zero until coverage settles.
    STATIC_REQUIRE(FarTerrainScheduler::WORKER_COUNT == 16);
    STATIC_REQUIRE(FarTerrainScheduler::LATENCY_WORKER_COUNT == 8);
    STATIC_REQUIRE(FarTerrainScheduler::CANOPY_WORKER_COUNT == 2);
    const FarTerrainSchedulerLimits limits;
    REQUIRE(limits.maxPending == 64);
    REQUIRE(limits.maxCompleted == 32);
    REQUIRE(limits.maxCacheEntries == 24576);
    REQUIRE(limits.maxCacheBytes == 3ull * 1024 * 1024 * 1024);
    REQUIRE(limits.maxCanopyCacheEntries == 24576);
}

TEST_CASE("Far terrain scheduler exposes the finest useful cached refinement",
          "[render][far-terrain][scheduler][cache][lod][priority][regression]") {
    FarTerrainScheduler scheduler(farTerrainTestSource());
    constexpr ColumnPos COORDINATE{0, 0};
    for (FarTerrainStep step :
         {FarTerrainStep::SIXTEEN, FarTerrainStep::FOUR, FarTerrainStep::TWO}) {
        REQUIRE(scheduler.enqueue({COORDINATE.x, COORDINATE.z, step}));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    REQUIRE(scheduler.stats().inFlight == 0);

    constexpr FarTerrainStepMask BASE_READY = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    REQUIRE_FALSE(scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                             FarTerrainStep::TWO, BASE_READY, true));
    const auto finest = scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_READY);
    REQUIRE(finest);
    REQUIRE(finest->key.step == FarTerrainStep::SIXTEEN);

    const auto distanceTier = scheduler.findFinestCached(COORDINATE, FarTerrainStep::THIRTY_TWO,
                                                         FarTerrainStep::FOUR, BASE_READY);
    REQUIRE(distanceTier);
    REQUIRE(distanceTier->key.step == FarTerrainStep::SIXTEEN);

    REQUIRE_FALSE(scheduler.findFinestCached(COORDINATE, FarTerrainStep::TWO, FarTerrainStep::TWO,
                                             BASE_READY));
}

TEST_CASE("Production coverage minima stay below exact emitted surfaces",
          "[render][far-terrain][coverage][bounds][worldgen]") {
    struct Fixture {
        std::shared_ptr<ChunkGenerator> generator;
        int64_t centerX;
        int64_t centerZ;
        const char* name;
    };
    auto ordinary = std::make_shared<ChunkGenerator>(42);
    auto volcanic = std::make_shared<ChunkGenerator>(764891);
    const std::array fixtures{
        Fixture{ordinary, -513, -257, "negative dry terrain"},
        Fixture{ordinary, -8'352, 2'160, "negative lake shoreline"},
        Fixture{ordinary, -12'289, 2'653, "negative river"},
        Fixture{volcanic, 23'029, -111'486, "caldera lake"},
    };

    bool sawSurfaceWater = false;
    bool sawVolcanicTerrain = false;
    for (const Fixture& fixture : fixtures) {
        const FarTerrainSource source =
            FarTerrainMesher::generatorGeometrySource(fixture.generator);
        for (int64_t dz = -4; dz <= 4; dz += 2) {
            for (int64_t dx = -4; dx <= 4; dx += 2) {
                const int64_t x = fixture.centerX + dx;
                const int64_t z = fixture.centerZ + dz;
                const worldgen::SurfaceSample exact = fixture.generator->sampleExactSurface(x, z);
                const worldgen::SurfaceSample filtered =
                    fixture.generator->sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_16);
                const worldgen::SurfaceSample canonicalWater =
                    fixture.generator->sampleFarGeometrySurface(
                        x, z, worldgen::SurfaceFootprint::BLOCK_1);
                const FarSurfaceSample coverage =
                    source.sample(x, z, worldgen::SurfaceFootprint::BLOCK_16);
                CAPTURE(fixture.name, x, z, exact.terrainHeight, filtered.terrainHeight,
                        coverage.footprintMinimumTerrainHeight);
                double expectedMinimum = filtered.terrainHeight -
                                         FAR_TERRAIN_STEP16_RELIEF_ENVELOPE -
                                         ChunkGenerator::emittedSurfaceDetailAmplitude(
                                             filtered, FAR_TERRAIN_STEP16_RELIEF_SLOPE_ENVELOPE) -
                                         FAR_TERRAIN_EMITTED_SURFACE_ENVELOPE;
                const bool standingWater =
                    canonicalWater.hydrology.ocean || canonicalWater.hydrology.river ||
                    canonicalWater.hydrology.lake || canonicalWater.hydrology.wetland;
                if (standingWater) {
                    expectedMinimum =
                        std::min(expectedMinimum, canonicalWater.hydrology.surfaceElevation);
                    expectedMinimum = std::min(
                        expectedMinimum, std::ceil(canonicalWater.hydrology.waterSurface) - 1.0);
                }
                if (canonicalWater.hydrology.waterfall &&
                    canonicalWater.hydrology.waterfallTop >=
                        canonicalWater.hydrology.waterfallBottom + 0.5) {
                    expectedMinimum = std::min(
                        expectedMinimum, std::ceil(canonicalWater.hydrology.waterfallBottom) - 1.0);
                }
                expectedMinimum = std::min(expectedMinimum, exact.terrainHeight);
                REQUIRE(coverage.footprintMinimumTerrainHeight == Catch::Approx(expectedMinimum));
                REQUIRE(coverage.footprintMinimumTerrainHeight <= exact.terrainHeight + 1.0e-6);
                sawSurfaceWater = sawSurfaceWater || exact.hydrology.ocean ||
                                  exact.hydrology.river || exact.hydrology.lake ||
                                  exact.hydrology.wetland;
                sawVolcanicTerrain = sawVolcanicTerrain || exact.geology.volcanicActivity > 0.5;
            }
        }
    }
    REQUIRE(sawSurfaceWater);
    REQUIRE(sawVolcanicTerrain);
}

TEST_CASE("Production far terrain uses generator-owned material palettes and ranks",
          "[render][far-terrain][material][worldgen][determinism]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.materialRank);

    constexpr int64_t WORLD_X = 23'029;
    constexpr int64_t WORLD_Z = -111'486;
    const worldgen::SurfaceSample surface =
        generator->sampleFarSurface(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8);
    const FarSurfaceSample sampled =
        source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8);
    REQUIRE(sameMaterialPalette(sampled.materialPalette,
                                generator->farSurfaceMaterialPaletteAt(WORLD_X, WORLD_Z, surface)));
    REQUIRE(source.materialRank(WORLD_X, WORLD_Z) ==
            Catch::Approx(generator->farSurfaceMaterialRankAt(WORLD_X, WORLD_Z)));

    std::array<FarSurfaceSample, 9> grid{};
    source.sampleGrid(WORLD_X - 8, WORLD_Z - 8, 8, 3, worldgen::SurfaceFootprint::BLOCK_8, grid);
    for (int z = 0; z < 3; ++z) {
        for (int x = 0; x < 3; ++x) {
            const int64_t sampleX = WORLD_X - 8 + x * 8;
            const int64_t sampleZ = WORLD_Z - 8 + z * 8;
            const worldgen::SurfaceSample expectedSurface =
                generator->sampleFarSurface(sampleX, sampleZ, worldgen::SurfaceFootprint::BLOCK_8);
            CAPTURE(sampleX, sampleZ);
            REQUIRE(sameMaterialPalette(
                grid[static_cast<size_t>(z * 3 + x)].materialPalette,
                generator->farSurfaceMaterialPaletteAt(sampleX, sampleZ, expectedSurface)));
        }
    }
}

TEST_CASE("The first far LOD reduces exact terrain to flat voxel terraces",
          "[render][far-terrain][seam][exact][voxel]") {
    ChunkGenerator generator(42);
    constexpr int64_t WORLD_X = -81'792;
    constexpr int64_t WORLD_Z = 126'976;
    const worldgen::SurfaceSample planned = generator.sampleSurface(WORLD_X, WORLD_Z);
    const worldgen::SurfaceSample coarse =
        generator.sampleFarSurface(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(WORLD_X, WORLD_Z);
    // This fixture must resolve to a different emitted density voxel so the
    // test can distinguish the exact handoff callback from its macro parent.
    // The bounded density detail contract now caps that displacement well
    // below the former twenty-block fixture delta.
    REQUIRE(std::abs(coarse.terrainHeight - exact.terrainHeight) > 2.0);
    // The public two-coordinate wrapper is block-resolution authority and
    // therefore agrees with exact cube emission. Only the explicit far
    // sampler returns the filtered macro parent used beyond exact residency.
    REQUIRE(planned.terrainHeight == Catch::Approx(exact.terrainHeight).margin(1.0e-4));
    REQUIRE(exact.terrainHeight == generator.surfaceYAt(WORLD_X, WORLD_Z) + 1.0);

    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) {
            return generator.sampleFarSurface(x, z, worldgen::SurfaceFootprint::BLOCK_16);
        });
    REQUIRE(source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_1)
                .geometry.terrainHeight == exact.terrainHeight);
    REQUIRE(source.sample(WORLD_X, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_8)
                .geometry.terrainHeight == coarse.terrainHeight);
    constexpr FarTerrainKey KEY{world_coord::floorDiv(WORLD_X, int64_t{FAR_TERRAIN_TILE_EDGE}),
                                world_coord::floorDiv(WORLD_Z, int64_t{FAR_TERRAIN_TILE_EDGE}),
                                FarTerrainStep::ONE};
    const auto mesh = FarTerrainMesher::build(KEY, source);
    REQUIRE(farTerrainTopsAreVoxelFlat(*mesh));
    const int64_t localCellX = WORLD_X - KEY.tileX * FAR_TERRAIN_TILE_EDGE;
    const int64_t localCellZ = WORLD_Z - KEY.tileZ * FAR_TERRAIN_TILE_EDGE;
    const float expectedHeight =
        expectedVoxelCellHeight(source, WORLD_X, WORLD_Z, FarTerrainStep::ONE);
    const std::optional<float> emittedHeight = farTerrainHeightAt(
        *mesh, static_cast<float>(localCellX) + 0.5F, static_cast<float>(localCellZ) + 0.5F);
    REQUIRE(emittedHeight);
    REQUIRE(*emittedHeight == expectedHeight);
}

TEST_CASE("Step one dense emitted tops survive conservative macro bounds",
          "[render][far-terrain][step-one][exact][bounds][regression]") {
    constexpr double EMITTED_TOP = 73.0;
    constexpr double MACRO_TOP = 96.0;
    constexpr double MINIMUM_BOUND = 12.0;
    constexpr double MAXIMUM_BOUND = 128.0;
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = EMITTED_TOP;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.sampleGrid = [sample = source.sample](int64_t originX, int64_t originZ, int spacing,
                                                 int sampleEdge,
                                                 worldgen::SurfaceFootprint footprint,
                                                 std::span<FarSurfaceSample> output) {
        for (int z = 0; z < sampleEdge; ++z) {
            for (int x = 0; x < sampleEdge; ++x) {
                output[static_cast<size_t>(z * sampleEdge + x)] =
                    sample(originX + static_cast<int64_t>(x * spacing),
                           originZ + static_cast<int64_t>(z * spacing), footprint);
            }
        }
    };
    source.cellBoundsGrid = [](int64_t, int64_t, int, int, int, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        std::ranges::fill(output, FarTerrainCellBounds{
                                      .terrainHeight = MACRO_TOP,
                                      .minimumTerrainHeight = MINIMUM_BOUND,
                                      .maximumTerrainHeight = MAXIMUM_BOUND,
                                  });
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::ONE}, source);
    const std::optional<float> emittedHeight = farTerrainHeightAt(*mesh, 128.5F, 128.5F);
    REQUIRE(emittedHeight);
    REQUIRE(*emittedHeight == static_cast<float>(EMITTED_TOP));
    REQUIRE(*emittedHeight != static_cast<float>(MACRO_TOP));
    REQUIRE(mesh->surfaceBounds.minY == static_cast<float>(MINIMUM_BOUND));
    REQUIRE(mesh->surfaceBounds.maxY == static_cast<float>(MAXIMUM_BOUND));
}

TEST_CASE("Far LOD reduces horizontal voxel resolution without sloped faces",
          "[render][far-terrain][lod][voxel][regression]") {
    const FarTerrainSource source = farTerrainTestSource();
    for (const FarTerrainStep step :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const int width = farTerrainStepSize(step);
        const auto mesh = FarTerrainMesher::build({-1, 2, step}, source);
        CAPTURE(width, mesh->terrainQuadCount, mesh->mergedTerrainCellCount);
        REQUIRE(farTerrainUsesVoxelFaces(*mesh, width));
        REQUIRE(mesh->mergedTerrainCellCount ==
                static_cast<uint32_t>((FAR_TERRAIN_TILE_EDGE / width) *
                                      (FAR_TERRAIN_TILE_EDGE / width)));
    }
}

TEST_CASE("Fallback voxel tops remain independent of conservative footprint minima",
          "[render][far-terrain][lod][voxel][transition][bounds][regression]") {
    FarTerrainSource source;
    source.sample = [](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        const double height = 92.0 + std::sin(static_cast<double>(x) * 0.071) * 7.0 +
                              std::cos(static_cast<double>(z) * 0.053) * 5.0;
        const double support = static_cast<double>(worldgen::surfaceFootprintWidth(footprint));
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = height;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = height - support * 0.75,
            .footprintMaximumTerrainHeight = height + support * 0.75,
            .materialPalette = testMaterialPalette(BlockType::STONE),
        };
    };

    constexpr std::array TIERS = {FarTerrainStep::ONE,     FarTerrainStep::TWO,
                                  FarTerrainStep::FOUR,    FarTerrainStep::EIGHT,
                                  FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO};
    std::array<std::shared_ptr<const FarTerrainMesh>, TIERS.size()> meshes;
    for (size_t index = 0; index < TIERS.size(); ++index) {
        meshes[index] = FarTerrainMesher::build({0, 0, TIERS[index]}, source);
    }
    bool sawSeparatedCoverageMinimum = false;
    for (size_t tierIndex = 1; tierIndex < TIERS.size(); ++tierIndex) {
        const FarTerrainStep tier = TIERS[tierIndex];
        const int step = farTerrainStepSize(tier);
        for (int cellZ = 0; cellZ < FAR_TERRAIN_TILE_EDGE; cellZ += step) {
            for (int cellX = 0; cellX < FAR_TERRAIN_TILE_EDGE; cellX += step) {
                if (cellX == 0 || cellZ == 0 || cellX + step == FAR_TERRAIN_TILE_EDGE ||
                    cellZ + step == FAR_TERRAIN_TILE_EDGE) {
                    continue;
                }
                const float x = static_cast<float>(cellX) + 0.5F;
                const float z = static_cast<float>(cellZ) + 0.5F;
                const auto top = farTerrainHeightAt(*meshes[tierIndex], x, z);
                const float expected = expectedVoxelCellHeight(source, cellX, cellZ, tier);
                REQUIRE(top);
                CAPTURE(step, x, z, *top, expected);
                REQUIRE(*top == expected);
                const FarSurfaceSample sample =
                    source.sample(cellX, cellZ, farTerrainSurfaceFootprint(tier));
                sawSeparatedCoverageMinimum = sawSeparatedCoverageMinimum ||
                                              *top - sample.footprintMinimumTerrainHeight > 10.0;
            }
        }
    }
    REQUIRE(sawSeparatedCoverageMinimum);
}

TEST_CASE("Production terrain keeps filtered voxel tops through atomic LOD swaps",
          "[render][far-terrain][lod][voxel][transition][worldgen][regression]") {
    struct Fixture {
        uint64_t seed;
        int64_t tileX;
        int64_t tileZ;
        const char* name;
    };
    constexpr std::array fixtures{
        Fixture{42, -33, 8, "lake shoreline"},
        Fixture{42, 0, -6, "ocean river exact handoff"},
        Fixture{764891, 89, -436, "volcanic caldera"},
    };
    constexpr std::array TIERS = {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
                                  FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO};
    constexpr int RASTER_SPACING = 2;
    for (const Fixture& fixture : fixtures) {
        auto generator = std::make_shared<ChunkGenerator>(fixture.seed);
        FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
        source.canopies = {};
        std::array<std::vector<float>, TIERS.size()> heights;
        std::array<double, TIERS.size()> means{};
        for (size_t tier = 0; tier < TIERS.size(); ++tier) {
            const auto mesh =
                FarTerrainMesher::build({fixture.tileX, fixture.tileZ, TIERS[tier]}, source);
            heights[tier] = farTerrainHeightRaster(*mesh, RASTER_SPACING);
            REQUIRE(std::ranges::none_of(heights[tier],
                                         [](float value) { return !std::isfinite(value); }));
            means[tier] = std::accumulate(heights[tier].begin(), heights[tier].end(), 0.0) /
                          static_cast<double>(heights[tier].size());
        }
        for (size_t tier = 1; tier < TIERS.size(); ++tier) {
            CAPTURE(fixture.name, farTerrainStepSize(TIERS[tier]), means[0], means[tier]);
            REQUIRE(std::abs(means[tier] - means[0]) <= 2.0);
        }
    }
    constexpr unsigned int SOURCE = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
    constexpr unsigned int TARGET = SOURCE | FAR_TERRAIN_LOD_TARGET_FLAG;
    for (const float progress : {0.0F, 0.25F, 0.499F, 0.5F, 0.75F, 1.0F}) {
        CAPTURE(progress);
        REQUIRE(farTerrainLodTerrainVisible(progress, SOURCE) !=
                farTerrainLodTerrainVisible(progress, TARGET));
    }
}

TEST_CASE("Seed forty-two coverage parents retain their filtered lowland surface",
          "[render][far-terrain][coverage][bounds][lod][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int PARENT_STEP = 32;
    constexpr int CHILD_STEP = 2;
    constexpr int CHILD_SCALE = PARENT_STEP / CHILD_STEP;
    struct Fixture {
        int64_t x;
        int64_t z;
    };
    constexpr std::array FIXTURES = {Fixture{64, -1'632}, Fixture{-512, -2'048}};
    for (const Fixture fixture : FIXTURES) {
        std::array<FarTerrainCellBounds, 1> parents{};
        source.cellBoundsGrid(fixture.x, fixture.z, PARENT_STEP, 1, 1,
                              worldgen::SurfaceFootprint::BLOCK_32, parents);
        std::array<worldgen::SurfaceSample, CHILD_SCALE * CHILD_SCALE> children{};
        generator->sampleExactSurfaceGrid(fixture.x, fixture.z, CHILD_STEP, CHILD_SCALE, children);
        const FarTerrainCellBounds& parent = parents.front();
        double maximumAbsoluteError = 0.0;
        for (const worldgen::SurfaceSample& child : children) {
            REQUIRE(child.terrainHeight > SEA_LEVEL + 1.0);
            maximumAbsoluteError = std::max(maximumAbsoluteError,
                                            std::abs(child.terrainHeight - parent.terrainHeight));
        }
        CAPTURE(fixture.x, fixture.z, parent.terrainHeight, parent.minimumTerrainHeight,
                maximumAbsoluteError);
        REQUIRE(parent.minimumTerrainHeight <= SEA_LEVEL);
        REQUIRE(parent.terrainHeight > SEA_LEVEL);
        REQUIRE(maximumAbsoluteError <= 6.0);

        const int64_t tileX = world_coord::floorDiv(fixture.x, int64_t{FAR_TERRAIN_TILE_EDGE});
        const int64_t tileZ = world_coord::floorDiv(fixture.z, int64_t{FAR_TERRAIN_TILE_EDGE});
        const auto mesh =
            FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
        const float localX = static_cast<float>(fixture.x - mesh->originX + PARENT_STEP / 2);
        const float localZ = static_cast<float>(fixture.z - mesh->originZ + PARENT_STEP / 2);
        const std::optional<float> meshTop = farTerrainHeightAt(*mesh, localX, localZ);
        REQUIRE(meshTop);
        REQUIRE(*meshTop == static_cast<float>(std::ceil(parent.terrainHeight)));
    }
}

TEST_CASE("Far generated water uses the exact source-block surface plane",
          "[render][far-terrain][water][seam][exact]") {
    ChunkGenerator generator(42);
    constexpr int64_t LAKE_X = -8352;
    constexpr int64_t LAKE_Z = 2160;
    const worldgen::SurfaceSample exact = generator.sampleExactSurface(LAKE_X, LAKE_Z);
    REQUIRE(exact.hydrology.lake);
    const FarTerrainSource source = FarTerrainMesher::surfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); });
    const FarTerrainGeometrySample geometry = testFarGeometry(source, LAKE_X, LAKE_Z);
    REQUIRE(geometry.lake);
    REQUIRE(geometry.waterSurface == std::ceil(exact.waterSurface));
    REQUIRE(geometry.waterSurface ==
            std::ceil(exact.waterSurface) - 1.0 + fluidSurfaceHeight(FluidState::source()));
}

TEST_CASE("Seed 42 far ocean coverage has no eight-block grid gaps",
          "[render][far-terrain][water][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    constexpr int64_t CENTER_X = -557;
    constexpr int64_t CENTER_Z = 379;
    constexpr int RADIUS = 32;
    constexpr int SAMPLE_EDGE = RADIUS * 2 + 3;
    constexpr int64_t SAMPLE_ORIGIN_X = CENTER_X - RADIUS - 1;
    constexpr int64_t SAMPLE_ORIGIN_Z = CENTER_Z - RADIUS - 1;
    std::array<FarTerrainGeometrySample, SAMPLE_EDGE * SAMPLE_EDGE> exact{};
    for (int z = 0; z < SAMPLE_EDGE; ++z) {
        for (int x = 0; x < SAMPLE_EDGE; ++x) {
            exact[static_cast<size_t>(z * SAMPLE_EDGE + x)] =
                source
                    .sample(SAMPLE_ORIGIN_X + x, SAMPLE_ORIGIN_Z + z,
                            worldgen::SurfaceFootprint::BLOCK_1)
                    .geometry;
        }
    }
    const auto wetAt = [&](int x, int z) {
        const FarTerrainGeometrySample& sample =
            exact[static_cast<size_t>((z + RADIUS + 1) * SAMPLE_EDGE + x + RADIUS + 1)];
        return (sample.ocean || sample.river || sample.lake || sample.wetland) &&
               sample.waterSurface > sample.terrainHeight + 0.01;
    };

    constexpr FarTerrainKey TILE{-3, 1, FarTerrainStep::TWO};
    // Step 32 intentionally uses one exact authority representative per
    // aligned 8x8 coverage cell and has dedicated ownership tests below.
    for (const FarTerrainStep step :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
          FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({TILE.tileX, TILE.tileZ, step}, source);
        size_t expectedWet = 0;
        size_t missing = 0;
        std::array<size_t, 64> missingByEightBlockPhase{};
        for (int dz = -RADIUS; dz <= RADIUS; ++dz) {
            for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                bool broadWater = true;
                for (int neighborZ = -1; neighborZ <= 1; ++neighborZ) {
                    for (int neighborX = -1; neighborX <= 1; ++neighborX)
                        broadWater = broadWater && wetAt(dx + neighborX, dz + neighborZ);
                }
                if (!broadWater)
                    continue;
                ++expectedWet;
                const int64_t worldX = CENTER_X + dx;
                const int64_t worldZ = CENTER_Z + dz;
                const float localX = static_cast<float>(worldX - mesh->originX) + 0.5F;
                const float localZ = static_cast<float>(worldZ - mesh->originZ) + 0.5F;
                if (farWaterTopCovers(*mesh, localX, localZ))
                    continue;
                ++missing;
                const size_t phaseX =
                    static_cast<size_t>(world_coord::floorMod(worldX, int64_t{8}));
                const size_t phaseZ =
                    static_cast<size_t>(world_coord::floorMod(worldZ, int64_t{8}));
                ++missingByEightBlockPhase[phaseZ * 8 + phaseX];
            }
        }
        CAPTURE(farTerrainStepSize(step), expectedWet, missing, missingByEightBlockPhase);
        REQUIRE(expectedWet > 1'000);
        REQUIRE(missing == 0);
    }
}

TEST_CASE("Step 32 skips the native water raster for bounds-proven dry terrain",
          "[render][far-terrain][water][coverage][startup][regression]") {
    const auto nativeWaterGridCalls = std::make_shared<size_t>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 72.0;
            sample.waterSurface = SEA_LEVEL;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t, int64_t, int, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 72.0,
                    .minimumTerrainHeight = 72.0,
                    .maximumTerrainHeight = 72.0,
                };
            }
        }
    };
    source.canonicalWaterGrid = [nativeWaterGridCalls](int64_t, int64_t, int, int, int, int,
                                                       worldgen::SurfaceFootprint,
                                                       std::span<FarTerrainGeometrySample> output) {
        ++*nativeWaterGridCalls;
        for (FarTerrainGeometrySample& sample : output) {
            sample.terrainHeight = 72.0;
            sample.waterSurface = SEA_LEVEL;
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(*nativeWaterGridCalls == 0);
    REQUIRE(mesh->waterQuadCount == 0);
    REQUIRE(mesh->waterContourTriangleCount == 0);
}

TEST_CASE("Step 32 emits uniform standing water without a canonical dense raster",
          "[render][far-terrain][water][coverage][startup][regression]") {
    constexpr worldgen::WaterBodyId LAKE_ID = 0x554E'4946'4F52'4DULL;
    const auto canonicalWaterGridCalls = std::make_shared<size_t>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = 64.0;
            sample.waterBodyId = LAKE_ID;
            sample.lake = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });
    source.cellBoundsGrid = [](int64_t, int64_t, int, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                };
            }
        }
    };
    source.canonicalWaterGrid =
        [canonicalWaterGridCalls](int64_t, int64_t, int, int, int, int, worldgen::SurfaceFootprint,
                                  std::span<FarTerrainGeometrySample> output) {
            ++*canonicalWaterGridCalls;
            for (FarTerrainGeometrySample& sample : output) {
                sample.terrainHeight = 48.0;
                sample.waterSurface = 64.0;
                sample.waterBodyId = LAKE_ID;
                sample.lake = true;
            }
        };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(*canonicalWaterGridCalls == 0);
    REQUIRE(mesh->waterQuadCount == 1);
    REQUIRE(mesh->waterContourTriangleCount == 0);
    REQUIRE(farWaterTopHeightAt(*mesh, 0.5F, 0.5F) == 64.0F);
    REQUIRE(farWaterTopHeightAt(*mesh, 127.5F, 127.5F) == 64.0F);
    REQUIRE(farWaterTopHeightAt(*mesh, 255.5F, 255.5F) == 64.0F);
}

TEST_CASE("Step 32 topology-marked standing water requests canonical authority",
          "[render][far-terrain][water][topology][coverage][regression]") {
    constexpr int ISLAND_MIN = 16;
    constexpr int ISLAND_MAX = 24;
    const auto canonicalWaterGridCalls = std::make_shared<size_t>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = SEA_LEVEL;
            sample.ocean = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t minimumX = originX + static_cast<int64_t>(cellX * step);
                const int64_t minimumZ = originZ + static_cast<int64_t>(cellZ * step);
                const bool islandCrossing = minimumX < ISLAND_MAX && minimumX + step > ISLAND_MIN &&
                                            minimumZ < ISLAND_MAX && minimumZ + step > ISLAND_MIN;
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = islandCrossing ? 72.0 : 48.0,
                    .waterTopologyPossible = islandCrossing,
                };
            }
        }
    };
    source.canonicalWaterGrid = [canonicalWaterGridCalls](
                                    int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                    int sampleWidth, int sampleHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainGeometrySample> output) {
        ++*canonicalWaterGridCalls;
        REQUIRE(output.size() == static_cast<size_t>(sampleWidth * sampleHeight));
        for (int z = 0; z < sampleHeight; ++z) {
            for (int x = 0; x < sampleWidth; ++x) {
                const int64_t worldX = originX + static_cast<int64_t>(x * spacingX);
                const int64_t worldZ = originZ + static_cast<int64_t>(z * spacingZ);
                const bool island = worldX >= ISLAND_MIN && worldX < ISLAND_MAX &&
                                    worldZ >= ISLAND_MIN && worldZ < ISLAND_MAX;
                FarTerrainGeometrySample& sample = output[static_cast<size_t>(z * sampleWidth + x)];
                sample.terrainHeight = island ? 72.0 : 48.0;
                sample.waterSurface = SEA_LEVEL;
                sample.ocean = !island;
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(*canonicalWaterGridCalls > 0);
    REQUIRE(farWaterTopCovers(*mesh, 5.5F, 5.5F));
    REQUIRE_FALSE(farWaterTopCovers(*mesh, 18.5F, 18.5F));
    REQUIRE(farWaterTopCovers(*mesh, 28.5F, 28.5F));
}

TEST_CASE("Step 32 heterogeneous standing water requests canonical authority",
          "[render][far-terrain][water][coverage][regression]") {
    constexpr int LAKE_MAX_X = 16;
    constexpr worldgen::WaterBodyId LAKE_ID = 0x4845'5445'524F'4745ULL;
    const auto canonicalWaterGridCalls = std::make_shared<size_t>(0);
    const auto standingWaterAt = [](int64_t x) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 48.0;
        if (x < LAKE_MAX_X) {
            sample.waterSurface = 70.0;
            sample.waterBodyId = LAKE_ID;
            sample.lake = true;
        } else {
            sample.waterSurface = 64.0;
            sample.ocean = true;
        }
        return sample;
    };
    FarTerrainSource source = testFarTerrainSource(
        [standingWaterAt](int64_t x, int64_t) { return standingWaterAt(x); },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });
    source.cellBoundsGrid = [](int64_t, int64_t, int, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                };
            }
        }
    };
    source.canonicalWaterGrid = [canonicalWaterGridCalls, standingWaterAt](
                                    int64_t originX, int64_t, int spacingX, int, int sampleWidth,
                                    int sampleHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainGeometrySample> output) {
        ++*canonicalWaterGridCalls;
        REQUIRE(output.size() == static_cast<size_t>(sampleWidth * sampleHeight));
        for (int z = 0; z < sampleHeight; ++z) {
            for (int x = 0; x < sampleWidth; ++x) {
                output[static_cast<size_t>(z * sampleWidth + x)] =
                    standingWaterAt(originX + static_cast<int64_t>(x * spacingX));
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(*canonicalWaterGridCalls > 0);
    REQUIRE(farWaterTopHeightAt(*mesh, 8.5F, 96.5F) == 70.0F);
    REQUIRE(farWaterTopHeightAt(*mesh, 20.5F, 96.5F) == 64.0F);
}

TEST_CASE("Sparse step 32 water preserves dense standing authority boundaries",
          "[render][far-terrain][water][coverage][topology][sparse][regression]") {
    constexpr int64_t LAKE_MIN = 12;
    constexpr int64_t LAKE_MAX = 20;
    constexpr worldgen::WaterBodyId LAKE_ID = 0x5350'4152'5345'4C4BULL;
    const auto standingWaterAt = [](int64_t x, int64_t z) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 48.0;
        const bool lake = x >= LAKE_MIN && x < LAKE_MAX && z >= LAKE_MIN && z < LAKE_MAX;
        sample.waterSurface = lake ? 70.0 : 64.0;
        sample.waterBodyId = lake ? LAKE_ID : worldgen::NO_WATER_BODY;
        sample.lake = lake;
        sample.ocean = !lake;
        return sample;
    };
    FarTerrainSource source = testFarTerrainSource(
        standingWaterAt,
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t minimumX = originX + static_cast<int64_t>(cellX * step);
                const int64_t minimumZ = originZ + static_cast<int64_t>(cellZ * step);
                const bool lakeBoundary = minimumX < LAKE_MAX && minimumX + step > LAKE_MIN &&
                                          minimumZ < LAKE_MAX && minimumZ + step > LAKE_MIN &&
                                          !(minimumX >= LAKE_MIN && minimumX + step <= LAKE_MAX &&
                                            minimumZ >= LAKE_MIN && minimumZ + step <= LAKE_MAX);
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterTopologyPossible = lakeBoundary,
                };
            }
        }
    };
    const auto denseSamples = std::make_shared<size_t>(0);
    source.canonicalWaterGrid = [standingWaterAt, denseSamples](
                                    int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                    int sampleWidth, int sampleHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainGeometrySample> output) {
        *denseSamples += output.size();
        for (int z = 0; z < sampleHeight; ++z) {
            for (int x = 0; x < sampleWidth; ++x) {
                output[static_cast<size_t>(z * sampleWidth + x)] =
                    standingWaterAt(originX + static_cast<int64_t>(x * spacingX),
                                    originZ + static_cast<int64_t>(z * spacingZ));
            }
        }
    };
    const auto sparseSamples = std::make_shared<size_t>(0);
    source.canonicalWaterPoints = [standingWaterAt,
                                   sparseSamples](std::span<const ColumnPos> positions,
                                                  worldgen::SurfaceFootprint,
                                                  std::span<FarTerrainGeometrySample> output) {
        REQUIRE(positions.size() == output.size());
        *sparseSamples += positions.size();
        for (size_t index = 0; index < positions.size(); ++index)
            output[index] = standingWaterAt(positions[index].x, positions[index].z);
    };

    FarTerrainSource denseSource = source;
    denseSource.sparseStep32Water = false;
    const auto dense = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, denseSource);
    const size_t denseSampleCount = *denseSamples;
    FarTerrainSource sparseSource = source;
    sparseSource.sparseStep32Water = true;
    const auto sparse = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, sparseSource);

    CAPTURE(denseSampleCount, *sparseSamples, dense->deterministicHash, sparse->deterministicHash,
            dense->waterTopology.bodyIdentityCount, sparse->waterTopology.bodyIdentityCount,
            dense->step32WaterGridCallCount, dense->step32WaterGridSampleCount,
            dense->step32WaterDenseGridCallCount, sparse->step32WaterGridCallCount,
            sparse->step32WaterGridSampleCount, sparse->step32WaterDenseGridCallCount);
    REQUIRE(dense->deterministicHash == sparse->deterministicHash);
    REQUIRE(dense->waterTopology == sparse->waterTopology);
    REQUIRE(dense->waterQuadCount == sparse->waterQuadCount);
    REQUIRE(dense->waterContourTriangleCount == sparse->waterContourTriangleCount);
    REQUIRE(dense->waterfallQuadCount == sparse->waterfallQuadCount);
    REQUIRE(sparse->waterTopology.bodyIdentityCount == 2);
    REQUIRE(*sparseSamples > 0);
    REQUIRE(*sparseSamples < denseSampleCount);
    REQUIRE(dense->step32WaterDenseGridCallCount == 1);
    REQUIRE(dense->step32WaterGridSampleCount == 66 * 66);
    REQUIRE(sparse->step32WaterDenseGridCallCount == 0);
    REQUIRE(sparse->step32WaterGridCallCount > 0);
    REQUIRE(sparse->step32WaterGridSampleCount > 0);
    REQUIRE(sparse->step32WaterGridSampleCount < dense->step32WaterGridSampleCount);
}

TEST_CASE("Sparse step 32 water retains partial pages when every shared edge is refined",
          "[render][far-terrain][water][coverage][topology][sparse][startup][regression]") {
    constexpr int CROSS_MIN = 112;
    constexpr int CROSS_MAX = 144;
    const auto waterAt = [](int64_t x, int64_t z) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 48.0;
        sample.waterSurface = SEA_LEVEL;
        sample.ocean = (x >= CROSS_MIN && x < CROSS_MAX) || (z >= CROSS_MIN && z < CROSS_MAX);
        return sample;
    };
    FarTerrainSource source =
        testFarTerrainSource(waterAt, [](int64_t, int64_t, const FarTerrainGeometrySample&) {
            return BlockType::STONE;
        });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int cellZ = 0; cellZ < cellHeight; ++cellZ) {
            for (int cellX = 0; cellX < cellWidth; ++cellX) {
                const int64_t minimumX = originX + static_cast<int64_t>(cellX * step);
                const int64_t minimumZ = originZ + static_cast<int64_t>(cellZ * step);
                const int64_t maximumX = minimumX + step;
                const int64_t maximumZ = minimumZ + step;
                const bool crossesVertical = (minimumX < CROSS_MIN && maximumX > CROSS_MIN) ||
                                             (minimumX < CROSS_MAX && maximumX > CROSS_MAX);
                const bool crossesHorizontal = (minimumZ < CROSS_MIN && maximumZ > CROSS_MIN) ||
                                               (minimumZ < CROSS_MAX && maximumZ > CROSS_MAX);
                output[static_cast<size_t>(cellZ * cellWidth + cellX)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterTopologyPossible = crossesVertical || crossesHorizontal,
                };
            }
        }
    };
    source.canonicalWaterGrid = [waterAt](int64_t originX, int64_t originZ, int spacingX,
                                          int spacingZ, int sampleWidth, int sampleHeight,
                                          worldgen::SurfaceFootprint,
                                          std::span<FarTerrainGeometrySample> output) {
        for (int z = 0; z < sampleHeight; ++z) {
            for (int x = 0; x < sampleWidth; ++x) {
                output[static_cast<size_t>(z * sampleWidth + x)] =
                    waterAt(originX + static_cast<int64_t>(x * spacingX),
                            originZ + static_cast<int64_t>(z * spacingZ));
            }
        }
    };
    source.canonicalWaterPoints = [waterAt](std::span<const ColumnPos> positions,
                                            worldgen::SurfaceFootprint,
                                            std::span<FarTerrainGeometrySample> output) {
        REQUIRE(positions.size() == output.size());
        for (size_t index = 0; index < positions.size(); ++index)
            output[index] = waterAt(positions[index].x, positions[index].z);
    };

    FarTerrainSource denseSource = source;
    denseSource.sparseStep32Water = false;
    const auto dense = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, denseSource);
    FarTerrainSource sparseSource = source;
    sparseSource.sparseStep32Water = true;
    const auto sparse = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, sparseSource);

    CAPTURE(dense->deterministicHash, sparse->deterministicHash, dense->step32WaterGridCallCount,
            dense->step32WaterGridSampleCount, dense->step32WaterDenseGridCallCount,
            sparse->step32WaterGridCallCount, sparse->step32WaterGridSampleCount,
            sparse->step32WaterDenseGridCallCount);
    REQUIRE(dense->deterministicHash == sparse->deterministicHash);
    REQUIRE(dense->waterTopology == sparse->waterTopology);
    REQUIRE(dense->step32WaterDenseGridCallCount == 1);
    REQUIRE(sparse->step32WaterDenseGridCallCount == 0);
    REQUIRE(sparse->step32WaterGridCallCount > 1);
    REQUIRE(sparse->step32WaterGridSampleCount > dense->step32WaterGridSampleCount / 2);
    REQUIRE(sparse->step32WaterGridSampleCount < dense->step32WaterGridSampleCount);
}

TEST_CASE("Topology-marked dry-corner channels survive every coarse far tier",
          "[render][far-terrain][water][topology][lod][regression]") {
    constexpr int64_t CHANNEL_MIN_X = 13;
    constexpr int64_t CHANNEL_MAX_X = 14;
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = 64.0;
            sample.river = x >= CHANNEL_MIN_X && x <= CHANNEL_MAX_X;
            sample.transitionOwnerKind = worldgen::WaterTransitionKind::RASTER_CHANNEL;
            sample.transitionOwnerId = 0x544F'504F'4C4F'4759ULL;
            sample.generatedFluidLevel = 1;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t minimumX = originX + static_cast<int64_t>(x * step);
                const int64_t maximumX = minimumX + step;
                const bool channelCrossing = minimumX <= CHANNEL_MAX_X && maximumX > CHANNEL_MIN_X;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterTopologyPossible = channelCrossing,
                };
            }
        }
        (void)originZ;
    };

    for (const FarTerrainStep step :
         {FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const auto mesh = FarTerrainMesher::build({0, 0, step}, source);
        CAPTURE(farTerrainStepSize(step), mesh->waterQuadCount, mesh->waterContourTriangleCount);
        REQUIRE(farWaterTopCovers(*mesh, 13.5F, 96.5F));
        REQUIRE(farWaterTopCovers(*mesh, 14.5F, 160.5F));
        REQUIRE_FALSE(farWaterTopCovers(*mesh, 10.5F, 96.5F));
        REQUIRE_FALSE(farWaterTopCovers(*mesh, 18.5F, 160.5F));
    }
}

TEST_CASE("Topology-marked parent-owned wetlands survive every coarse far tier",
          "[render][far-terrain][water][wetland][topology][lod][step-32][determinism]") {
    constexpr int64_t WETLAND_MIN_X = 12;
    constexpr int64_t WETLAND_MAX_X = 16;
    constexpr worldgen::WaterBodyId PARENT_BODY = 0x5745'544C'414E'4404ULL;
    const auto wetlandAt = [](int64_t x, int64_t) {
        FarTerrainGeometrySample sample;
        sample.terrainHeight = 63.875;
        sample.waterSurface = 64.0;
        sample.waterBodyId = PARENT_BODY;
        sample.wetland = x >= WETLAND_MIN_X && x < WETLAND_MAX_X;
        return sample;
    };
    const auto canonicalGridCalls = std::make_shared<size_t>(0);
    FarTerrainSource source =
        testFarTerrainSource(wetlandAt, [](int64_t, int64_t, const FarTerrainGeometrySample&) {
            return BlockType::CLAY;
        });
    source.cellBoundsGrid = [](int64_t originX, int64_t, int step, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t minimumX = originX + static_cast<int64_t>(x * step);
                const bool wetlandCrossing =
                    minimumX < WETLAND_MAX_X && minimumX + step > WETLAND_MIN_X;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = 63.875,
                    .minimumTerrainHeight = 63.875,
                    .maximumTerrainHeight = 63.875,
                    .waterTopologyPossible = wetlandCrossing,
                };
            }
        }
    };
    source.canonicalWaterGrid = [canonicalGridCalls, wetlandAt](
                                    int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                    int sampleWidth, int sampleHeight, worldgen::SurfaceFootprint,
                                    std::span<FarTerrainGeometrySample> output) {
        ++*canonicalGridCalls;
        REQUIRE(output.size() == static_cast<size_t>(sampleWidth * sampleHeight));
        for (int z = 0; z < sampleHeight; ++z) {
            for (int x = 0; x < sampleWidth; ++x) {
                output[static_cast<size_t>(z * sampleWidth + x)] =
                    wetlandAt(originX + static_cast<int64_t>(x * spacingX),
                              originZ + static_cast<int64_t>(z * spacingZ));
            }
        }
    };

    for (const FarTerrainStep step :
         {FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const auto first = FarTerrainMesher::build({0, 0, step}, source);
        const auto second = FarTerrainMesher::build({0, 0, step}, source);
        CAPTURE(farTerrainStepSize(step), first->waterQuadCount, first->waterContourTriangleCount,
                first->deterministicHash);
        REQUIRE(first->deterministicHash == second->deterministicHash);
        REQUIRE(farWaterTopCovers(*first, 14.5F, 96.5F));
        REQUIRE(farWaterTopCovers(*first, 14.5F, 160.5F));
        REQUIRE_FALSE(farWaterTopCovers(*first, 10.5F, 96.5F));
        REQUIRE_FALSE(farWaterTopCovers(*first, 18.5F, 160.5F));
    }
    REQUIRE(*canonicalGridCalls > 0);
}

TEST_CASE("Step 32 topology preserves dry islands inside uniform ocean samples",
          "[render][far-terrain][water][topology][island][lod][regression]") {
    constexpr int ISLAND_MIN = 13;
    constexpr int ISLAND_MAX = 16;
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            const bool island =
                x >= ISLAND_MIN && x < ISLAND_MAX && z >= ISLAND_MIN && z < ISLAND_MAX;
            FarTerrainGeometrySample sample;
            sample.terrainHeight = island ? 72.0 : 48.0;
            sample.waterSurface = SEA_LEVEL;
            sample.ocean = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t minimumX = originX + static_cast<int64_t>(x * step);
                const int64_t minimumZ = originZ + static_cast<int64_t>(z * step);
                const bool islandCrossing = minimumX < ISLAND_MAX && minimumX + step > ISLAND_MIN &&
                                            minimumZ < ISLAND_MAX && minimumZ + step > ISLAND_MIN;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 72.0,
                    .waterTopologyPossible = islandCrossing,
                };
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(farWaterTopCovers(*mesh, 5.5F, 5.5F));
    REQUIRE_FALSE(farWaterTopCovers(*mesh, 14.5F, 14.5F));
    REQUIRE(farWaterTopCovers(*mesh, 20.5F, 20.5F));
}

TEST_CASE("Step 32 topology recovers a narrow ocean inlet missed by every sample",
          "[render][far-terrain][water][topology][ocean][lod][regression]") {
    constexpr int INLET_MIN_X = 13;
    constexpr int INLET_MAX_X = 16;
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = SEA_LEVEL;
            sample.ocean = x >= INLET_MIN_X && x < INLET_MAX_X;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.cellBoundsGrid = [](int64_t originX, int64_t, int step, int cellWidth, int cellHeight,
                               worldgen::SurfaceFootprint, std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t minimumX = originX + static_cast<int64_t>(x * step);
                const bool inletCrossing = minimumX < INLET_MAX_X && minimumX + step > INLET_MIN_X;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterTopologyPossible = inletCrossing,
                };
            }
        }
    };

    const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE_FALSE(farWaterTopCovers(*mesh, 10.5F, 96.5F));
    REQUIRE(farWaterTopCovers(*mesh, 14.5F, 96.5F));
    REQUIRE_FALSE(farWaterTopCovers(*mesh, 18.5F, 96.5F));
}

TEST_CASE("Production v4 coarse tiers never construct exact plans or ordinary stage tiles",
          "[render][far-terrain][v4][performance][stage][column-plan][regression]") {
    constexpr uint64_t SEED = 42;
    constexpr int64_t TILE_X = 30;
    constexpr int64_t TILE_Z = -5;
    TempDir directory("v4_plan_free_coarse_tiers");
    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.planFreeCoarseAuthority);

    const uint64_t plansBefore = generator->columnPlanConstructionRequestCount();
    const worldgen::NativeHydrologyCacheMetrics hydrologyBefore =
        generator->nativeHydrologyCacheMetrics();
    const auto started = std::chrono::steady_clock::now();
    for (const FarTerrainStep step :
         {FarTerrainStep::THIRTY_TWO, FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT,
          FarTerrainStep::FOUR, FarTerrainStep::TWO}) {
        std::shared_ptr<const FarTerrainMesh> mesh;
        awaitV4Authority([&] {
            mesh = FarTerrainMesher::build({TILE_X, TILE_Z, step}, source,
                                           FarTerrainAuthorityQuality::FINAL);
        });
        CAPTURE(static_cast<int>(step));
        REQUIRE(mesh);
        REQUIRE(generator->columnPlanConstructionRequestCount() == plansBefore);
    }
    const double elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const worldgen::NativeHydrologyCacheMetrics hydrologyAfter =
        generator->nativeHydrologyCacheMetrics();
    CAPTURE(elapsedSeconds, hydrologyBefore.ordinaryStageTileBuilds,
            hydrologyAfter.ordinaryStageTileBuilds, hydrologyBefore.ordinaryStageTileFailures,
            hydrologyAfter.ordinaryStageTileFailures,
            hydrologyBefore.ordinaryStageCoarseGridSamples,
            hydrologyAfter.ordinaryStageCoarseGridSamples);
    REQUIRE(generator->columnPlanConstructionRequestCount() == plansBefore);
    REQUIRE(hydrologyAfter.ordinaryStageTileBuilds == hydrologyBefore.ordinaryStageTileBuilds);
    REQUIRE(hydrologyAfter.ordinaryStageTileFailures == hydrologyBefore.ordinaryStageTileFailures);
    REQUIRE(hydrologyAfter.ordinaryStageCoarseGridSamples >
            hydrologyBefore.ordinaryStageCoarseGridSamples);
}

TEST_CASE("Production v4 step one fallback uses the dense plan-free exact grid",
          "[render][far-terrain][v4][step-one][performance][column-plan][regression]") {
    constexpr uint64_t SEED = 42;
    TempDir directory("v4_plan_free_step_one");
    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const uint64_t plansBefore = generator->columnPlanConstructionRequestCount();

    constexpr int SAMPLE_EDGE = 17;
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> dense{};
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> macro{};
    awaitV4Authority([&] { generator->sampleExactSurfaceGrid(0, 0, 1, SAMPLE_EDGE, dense); });
    awaitV4Authority([&] {
        generator->sampleFarGeometryGrid(0, 0, 1, 1, SAMPLE_EDGE, SAMPLE_EDGE,
                                         worldgen::SurfaceFootprint::BLOCK_1, macro);
    });
    REQUIRE(generator->columnPlanConstructionRequestCount() == plansBefore);
    for (const ColumnPos position : {ColumnPos{0, 0}, ColumnPos{8, 8}, ColumnPos{16, 16}}) {
        const worldgen::SurfaceSample scalar =
            awaitV4Authority([&] { return generator->sampleExactSurface(position.x, position.z); });
        const worldgen::SurfaceSample& batched =
            dense[static_cast<size_t>(position.z * SAMPLE_EDGE + position.x)];
        CAPTURE(position.x, position.z, scalar.terrainHeight, batched.terrainHeight);
        REQUIRE(batched.terrainHeight == scalar.terrainHeight);
        REQUIRE(batched.waterSurface == scalar.waterSurface);
        REQUIRE(batched.hydrology.waterBodyId == scalar.hydrology.waterBodyId);
        REQUIRE(worldgen::generatedFluidColumn(batched).wet ==
                worldgen::generatedFluidColumn(scalar).wet);
    }
    std::optional<ColumnPos> authorityDelta;
    for (int z = 0; z + 1 < SAMPLE_EDGE && !authorityDelta; ++z) {
        for (int x = 0; x + 1 < SAMPLE_EDGE; ++x) {
            const size_t index = static_cast<size_t>(z * SAMPLE_EDGE + x);
            const double exactTop = worldgen::geometryTerrainHeight(dense[index]);
            const double macroTop = std::ceil(worldgen::geometryTerrainHeight(macro[index]));
            if (exactTop != macroTop) {
                authorityDelta = ColumnPos{x, z};
                break;
            }
        }
    }
    REQUIRE(authorityDelta);
    const uint64_t plansBeforeMesh = generator->columnPlanConstructionRequestCount();

    std::shared_ptr<const FarTerrainMesh> mesh;
    const auto started = std::chrono::steady_clock::now();
    awaitV4Authority([&] {
        mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::ONE}, source,
                                       FarTerrainAuthorityQuality::FINAL);
    });
    const double elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    CAPTURE(elapsedSeconds, plansBeforeMesh, generator->columnPlanConstructionRequestCount());
    REQUIRE(mesh);
    REQUIRE(generator->columnPlanConstructionRequestCount() == plansBeforeMesh);
    const size_t deltaIndex =
        static_cast<size_t>(authorityDelta->z * SAMPLE_EDGE + authorityDelta->x);
    const float exactTop = static_cast<float>(worldgen::geometryTerrainHeight(dense[deltaIndex]));
    const std::optional<float> meshTop =
        farTerrainHeightAt(*mesh, static_cast<float>(authorityDelta->x) + 0.5F,
                           static_cast<float>(authorityDelta->z) + 0.5F);
    CAPTURE(authorityDelta->x, authorityDelta->z, exactTop, meshTop);
    REQUIRE(meshTop);
    REQUIRE(*meshTop == exactTop);
}

TEST_CASE("Production v4 topology admits a hidden native route at a signed tile edge",
          "[render][far-terrain][water][topology][v4][negative][edge][regression]") {
    constexpr uint64_t SEED = 0x544F'504F'4C4F'4759ULL;
    constexpr int64_t TOPOLOGY_ORIGIN = -2'048;
    constexpr int TOPOLOGY_EDGE = 64;
    TempDir directory("v4_hidden_native_route");
    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    REQUIRE(identity.valid());
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);

    std::array<worldgen::NativeHydrologyTopologyCell, TOPOLOGY_EDGE * TOPOLOGY_EDGE> topology{};
    awaitV4Authority([&] {
        generator->sampleNativeHydrologyTopologyGrid(TOPOLOGY_ORIGIN, TOPOLOGY_ORIGIN,
                                                     TOPOLOGY_EDGE, TOPOLOGY_EDGE, topology);
    });

    const auto wet = [](const worldgen::HydrologySample& sample) {
        return (sample.ocean || sample.river || sample.lake || sample.wetland) &&
               sample.waterSurface > sample.surfaceElevation + 0.01;
    };
    struct HiddenRoute {
        bool found = false;
        int64_t parentX = 0;
        int64_t parentZ = 0;
        int64_t waterX = 0;
        int64_t waterZ = 0;
    } route;
    size_t topologyCandidates = 0;

    // Keep the search away from the native-page apron, then require a parent
    // on an owned far-tile edge. The terrain samples and the water authority
    // both come from the production v4 ChunkGenerator path.
    for (int topologyZ = 1; topologyZ + 1 < TOPOLOGY_EDGE && !route.found; ++topologyZ) {
        for (int topologyX = 1; topologyX + 1 < TOPOLOGY_EDGE && !route.found; ++topologyX) {
            const auto topologyCell =
                topology[static_cast<size_t>(topologyZ * TOPOLOGY_EDGE + topologyX)];
            if (!topologyCell.waterTopologyPossible)
                continue;
            const int64_t parentX =
                TOPOLOGY_ORIGIN + topologyX * worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
            const int64_t parentZ =
                TOPOLOGY_ORIGIN + topologyZ * worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE;
            const int localParentX =
                static_cast<int>(world_coord::floorMod(parentX, int64_t{FAR_TERRAIN_TILE_EDGE}) /
                                 worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
            const int localParentZ =
                static_cast<int>(world_coord::floorMod(parentZ, int64_t{FAR_TERRAIN_TILE_EDGE}) /
                                 worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
            if (localParentX != 0 && localParentX != 7 && localParentZ != 0 && localParentZ != 7)
                continue;
            ++topologyCandidates;

            const std::array<ColumnPos, 5> coarsePositions{{
                {parentX, parentZ},
                {parentX + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE, parentZ},
                {parentX + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE,
                 parentZ + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE},
                {parentX, parentZ + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE},
                {parentX + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE / 2,
                 parentZ + worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE / 2},
            }};
            std::array<worldgen::HydrologySample, coarsePositions.size()> coarse{};
            awaitV4Authority(
                [&] { generator->sampleNativeHydrologyAuthorityPoints(coarsePositions, coarse); });
            if (std::ranges::any_of(coarse, wet))
                continue;

            constexpr int NATIVE_EDGE = worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE /
                                            worldgen::NATIVE_HYDROLOGY_RASTER_SPACING +
                                        1;
            std::array<worldgen::HydrologySample, NATIVE_EDGE * NATIVE_EDGE> native{};
            awaitV4Authority([&] {
                generator->sampleNativeHydrologyAuthorityGrid(
                    parentX, parentZ, worldgen::NATIVE_HYDROLOGY_RASTER_SPACING,
                    worldgen::NATIVE_HYDROLOGY_RASTER_SPACING, NATIVE_EDGE, NATIVE_EDGE, native);
            });
            for (int nativeZ = 0; nativeZ + 1 < NATIVE_EDGE && !route.found; ++nativeZ) {
                for (int nativeX = 0; nativeX + 1 < NATIVE_EDGE; ++nativeX) {
                    const auto& sample =
                        native[static_cast<size_t>(nativeZ * NATIVE_EDGE + nativeX)];
                    if (!wet(sample))
                        continue;
                    route = {
                        .found = true,
                        .parentX = parentX,
                        .parentZ = parentZ,
                        .waterX = parentX + nativeX * worldgen::NATIVE_HYDROLOGY_RASTER_SPACING,
                        .waterZ = parentZ + nativeZ * worldgen::NATIVE_HYDROLOGY_RASTER_SPACING,
                    };
                    break;
                }
            }
        }
    }
    CAPTURE(topologyCandidates);
    REQUIRE(route.found);

    const int64_t tileX = world_coord::floorDiv(route.parentX, int64_t{FAR_TERRAIN_TILE_EDGE});
    const int64_t tileZ = world_coord::floorDiv(route.parentZ, int64_t{FAR_TERRAIN_TILE_EDGE});
    const int64_t tileOriginX = tileX * FAR_TERRAIN_TILE_EDGE;
    const int64_t tileOriginZ = tileZ * FAR_TERRAIN_TILE_EDGE;
    const int parentX = static_cast<int>((route.parentX - tileOriginX) /
                                         worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
    const int parentZ = static_cast<int>((route.parentZ - tileOriginZ) /
                                         worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);
    REQUIRE(tileX < 0);
    REQUIRE(tileZ < 0);
    REQUIRE((parentX == 0 || parentX == 7 || parentZ == 0 || parentZ == 7));

    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    std::array<FarTerrainCellBounds, 100> bounds{};
    awaitV4Authority([&] {
        source.cellBoundsGrid(tileOriginX - 32, tileOriginZ - 32, 32, 10, 10,
                              worldgen::SurfaceFootprint::BLOCK_32, bounds);
    });
    REQUIRE(bounds[static_cast<size_t>((parentZ + 1) * 10 + parentX + 1)].waterTopologyPossible);

    for (const ColumnPos position : std::array<ColumnPos, 5>{{
             {route.parentX, route.parentZ},
             {route.parentX + 32, route.parentZ},
             {route.parentX + 32, route.parentZ + 32},
             {route.parentX, route.parentZ + 32},
             {route.parentX + 16, route.parentZ + 16},
         }}) {
        const FarTerrainGeometrySample geometry = awaitV4Authority([&] {
            return source.sample(position.x, position.z, worldgen::SurfaceFootprint::BLOCK_32)
                .geometry;
        });
        REQUIRE_FALSE(((geometry.ocean || geometry.river || geometry.lake || geometry.wetland) &&
                       geometry.waterSurface > geometry.terrainHeight + 0.01));
    }

    const auto sparsePointSamples = std::make_shared<size_t>(0);
    FarTerrainSource sparseSource = source;
    const auto canonicalWaterPoints = sparseSource.canonicalWaterPoints;
    sparseSource.canonicalWaterPoints =
        [canonicalWaterPoints, sparsePointSamples](std::span<const ColumnPos> positions,
                                                   worldgen::SurfaceFootprint footprint,
                                                   std::span<FarTerrainGeometrySample> output) {
            *sparsePointSamples += positions.size();
            canonicalWaterPoints(positions, footprint, output);
        };

    const auto denseGridSamples = std::make_shared<size_t>(0);
    FarTerrainSource denseSource = source;
    denseSource.sparseStep32Water = false;
    const auto canonicalWaterGrid = denseSource.canonicalWaterGrid;
    denseSource.canonicalWaterGrid =
        [canonicalWaterGrid, denseGridSamples](int64_t originX, int64_t originZ, int spacingX,
                                               int spacingZ, int sampleWidth, int sampleHeight,
                                               worldgen::SurfaceFootprint footprint,
                                               std::span<FarTerrainGeometrySample> output) {
            *denseGridSamples += output.size();
            canonicalWaterGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight,
                               footprint, output);
        };

    const auto sparseMesh = awaitV4Authority([&] {
        return FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, sparseSource);
    });
    const auto denseMesh = awaitV4Authority([&] {
        return FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, denseSource);
    });
    const float localWaterX = static_cast<float>(route.waterX - tileOriginX) + 0.5F;
    const float localWaterZ = static_cast<float>(route.waterZ - tileOriginZ) + 0.5F;
    CAPTURE(route.parentX, route.parentZ, route.waterX, route.waterZ, localWaterX, localWaterZ,
            sparseMesh->waterQuadCount, sparseMesh->waterContourTriangleCount, *sparsePointSamples,
            *denseGridSamples, sparseMesh->step32WaterGridCallCount,
            sparseMesh->step32WaterGridSampleCount, sparseMesh->step32WaterDenseGridCallCount);
    REQUIRE(farWaterTopCovers(*sparseMesh, localWaterX, localWaterZ));
    REQUIRE(sparseMesh->deterministicHash == denseMesh->deterministicHash);
    REQUIRE(sparseMesh->waterTopology == denseMesh->waterTopology);
    REQUIRE(sparseMesh->waterQuadCount == denseMesh->waterQuadCount);
    REQUIRE(sparseMesh->waterContourTriangleCount == denseMesh->waterContourTriangleCount);
    REQUIRE(sparseMesh->waterfallQuadCount == denseMesh->waterfallQuadCount);
    REQUIRE(*sparsePointSamples < *denseGridSamples);
    REQUIRE(sparseMesh->step32WaterGridCallCount > 0);
    REQUIRE(sparseMesh->step32WaterGridSampleCount <= denseMesh->step32WaterGridSampleCount);
}

TEST_CASE("Step 32 emits a native waterfall anchor between terrain samples",
          "[render][far-terrain][water][waterfall][lod][regression]") {
    constexpr int FALL_X = 12;
    constexpr int FALL_Z = 16;
    const auto authorityGridCalls = std::make_shared<size_t>(0);
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = SEA_LEVEL;
            sample.river = x == FALL_X && z == FALL_Z;
            sample.waterfall = sample.river;
            sample.waterfallAnchor = sample.river;
            sample.waterfallTop = sample.river ? 84.0 : 0.0;
            sample.waterfallBottom = sample.river ? SEA_LEVEL : 0.0;
            sample.waterfallWidth = sample.river ? 4.0 : 0.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto sample = source.sample;
    source.cellBoundsGrid = [](int64_t originX, int64_t originZ, int step, int cellWidth,
                               int cellHeight, worldgen::SurfaceFootprint,
                               std::span<FarTerrainCellBounds> output) {
        for (int z = 0; z < cellHeight; ++z) {
            for (int x = 0; x < cellWidth; ++x) {
                const int64_t minimumX = originX + static_cast<int64_t>(x * step);
                const int64_t minimumZ = originZ + static_cast<int64_t>(z * step);
                const bool ownsAnchor = minimumX <= FALL_X && minimumX + step > FALL_X &&
                                        minimumZ <= FALL_Z && minimumZ + step > FALL_Z;
                output[static_cast<size_t>(z * cellWidth + x)] = {
                    .terrainHeight = 48.0,
                    .minimumTerrainHeight = 48.0,
                    .maximumTerrainHeight = 48.0,
                    .waterfallPossible = ownsAnchor,
                };
            }
        }
    };
    source.waterAuthorityGrid =
        [sample, authorityGridCalls](int64_t originX, int64_t originZ, int spacingX, int spacingZ,
                                     int sampleWidth, int sampleHeight,
                                     worldgen::SurfaceFootprint footprint,
                                     std::span<FarTerrainGeometrySample> output) {
            ++*authorityGridCalls;
            REQUIRE(spacingX == spacingZ);
            REQUIRE(output.size() == static_cast<size_t>(sampleWidth * sampleHeight));
            for (int z = 0; z < sampleHeight; ++z) {
                for (int x = 0; x < sampleWidth; ++x) {
                    output[static_cast<size_t>(z * sampleWidth + x)] =
                        sample(originX + static_cast<int64_t>(x * spacingX),
                               originZ + static_cast<int64_t>(z * spacingZ), footprint)
                            .geometry;
                }
            }
        };

    const auto owner = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    const auto neighbor = FarTerrainMesher::build({1, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(owner->waterfallQuadCount == 5);
    REQUIRE(neighbor->waterfallQuadCount == 0);
    // The single 66-by-66 coverage page also finds the hidden native anchor.
    // A per-parent waterfall probe here would issue a second authority grid
    // for the owner before rebuilding the same coverage page.
    REQUIRE(*authorityGridCalls == 1);
}

TEST_CASE("Far geometry emits canonical parent-owned wetland water",
          "[render][far-terrain][wetland][regression]") {
    const FarTerrainSource source =
        FarTerrainMesher::surfaceGeometrySource([](int64_t, int64_t, worldgen::SurfaceFootprint) {
            worldgen::SurfaceSample sample;
            sample.terrainHeight = 63.875;
            sample.hydrology.surfaceElevation = 63.875;
            sample.hydrology.waterSurface = 64.0;
            sample.waterSurface = 64.0;
            sample.hydrology.waterBodyId = 0x5745'544C'414E'4404ULL;
            sample.hydrology.wetland = true;
            sample.hydrology.groundwaterHead = 64.0;
            sample.hydrology.hydroperiod = 0.80;
            return sample;
        });
    const FarTerrainGeometrySample geometry = testFarGeometry(source, 0, 0);
    REQUIRE(geometry.wetland);
    REQUIRE_FALSE(geometry.ocean);
    REQUIRE_FALSE(geometry.river);
    REQUIRE_FALSE(geometry.lake);
    REQUIRE(geometry.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(geometry.waterSurface > geometry.terrainHeight);

    const auto first = FarTerrainMesher::build({0, 0, FarTerrainStep::EIGHT}, source);
    const auto second = FarTerrainMesher::build({0, 0, FarTerrainStep::EIGHT}, source);
    REQUIRE(first->deterministicHash == second->deterministicHash);
    REQUIRE(first->waterQuadCount > 0);
    REQUIRE(farWaterTopHeightAt(*first, 128.5F, 128.5F) == 64.0F);
}

TEST_CASE("Seed 42 step 32 water respects exact ownership through a cold handoff",
          "[render][far-terrain][water][coverage][ownership][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    const auto geometryPointCount = std::make_shared<size_t>(0);
    const auto authorityPointCount = std::make_shared<size_t>(0);
    const auto authorityGridCount = std::make_shared<size_t>(0);
    const auto canonicalPointCount = std::make_shared<size_t>(0);
    const auto canonicalGridCount = std::make_shared<size_t>(0);
    const auto geometryPoints = source.geometryPoints;
    source.geometryPoints = [geometryPointCount,
                             geometryPoints](std::span<const ColumnPos> positions,
                                             worldgen::SurfaceFootprint footprint,
                                             std::span<FarTerrainGeometrySample> output) {
        if (footprint == worldgen::SurfaceFootprint::BLOCK_1)
            *geometryPointCount += positions.size();
        geometryPoints(positions, footprint, output);
    };
    const auto authorityPoints = source.waterAuthorityPoints;
    source.waterAuthorityPoints = [authorityPointCount,
                                   authorityPoints](std::span<const ColumnPos> positions,
                                                    worldgen::SurfaceFootprint footprint,
                                                    std::span<FarTerrainGeometrySample> output) {
        *authorityPointCount += positions.size();
        authorityPoints(positions, footprint, output);
    };
    const auto authorityGrid = source.waterAuthorityGrid;
    source.waterAuthorityGrid = [authorityGridCount,
                                 authorityGrid](int64_t originX, int64_t originZ, int spacingX,
                                                int spacingZ, int sampleWidth, int sampleHeight,
                                                worldgen::SurfaceFootprint footprint,
                                                std::span<FarTerrainGeometrySample> output) {
        *authorityGridCount += output.size();
        authorityGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight, footprint,
                      output);
    };
    const auto canonicalPoints = source.canonicalWaterPoints;
    source.canonicalWaterPoints = [canonicalPointCount,
                                   canonicalPoints](std::span<const ColumnPos> positions,
                                                    worldgen::SurfaceFootprint footprint,
                                                    std::span<FarTerrainGeometrySample> output) {
        *canonicalPointCount += positions.size();
        canonicalPoints(positions, footprint, output);
    };
    const auto canonicalGrid = source.canonicalWaterGrid;
    source.canonicalWaterGrid = [canonicalGridCount,
                                 canonicalGrid](int64_t originX, int64_t originZ, int spacingX,
                                                int spacingZ, int sampleWidth, int sampleHeight,
                                                worldgen::SurfaceFootprint footprint,
                                                std::span<FarTerrainGeometrySample> output) {
        *canonicalGridCount += output.size();
        canonicalGrid(originX, originZ, spacingX, spacingZ, sampleWidth, sampleHeight, footprint,
                      output);
    };

    constexpr FarTerrainKey LEFT{-1, -6, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey RIGHT{0, -6, FarTerrainStep::THIRTY_TWO};
    const auto left = FarTerrainMesher::build(LEFT, source);
    const auto right = FarTerrainMesher::build(RIGHT, source);

    constexpr int SAMPLE_STEP = 8;
    constexpr int SAMPLE_EDGE = FAR_TERRAIN_TILE_EDGE / SAMPLE_STEP + 2;
    constexpr int64_t SAMPLE_ORIGIN_X = -SAMPLE_STEP;
    constexpr int64_t SAMPLE_ORIGIN_Z = RIGHT.tileZ * FAR_TERRAIN_TILE_EDGE - SAMPLE_STEP;
    std::array<worldgen::SurfaceSample, SAMPLE_EDGE * SAMPLE_EDGE> exact{};
    generator->sampleExactSurfaceGrid(SAMPLE_ORIGIN_X, SAMPLE_ORIGIN_Z, SAMPLE_STEP, SAMPLE_EDGE,
                                      exact);
    const auto exactWet = [&](int x, int z) {
        const worldgen::SurfaceSample& sample = exact[static_cast<size_t>(z * SAMPLE_EDGE + x)];
        return (sample.hydrology.ocean || sample.hydrology.river || sample.hydrology.lake ||
                sample.hydrology.wetland) &&
               sample.hydrology.waterSurface > sample.terrainHeight + 0.01;
    };
    size_t broadDry = 0;
    size_t broadWet = 0;
    size_t falseWater = 0;
    size_t missingWater = 0;
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        for (int x = 1; x + 1 < SAMPLE_EDGE; ++x) {
            bool neighborhoodDry = true;
            bool neighborhoodWet = true;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dx = -1; dx <= 1; ++dx) {
                    neighborhoodDry = neighborhoodDry && !exactWet(x + dx, z + dz);
                    neighborhoodWet = neighborhoodWet && exactWet(x + dx, z + dz);
                }
            }
            const float localX = static_cast<float>((x - 1) * SAMPLE_STEP) + 0.5F;
            const float localZ = static_cast<float>((z - 1) * SAMPLE_STEP) + 0.5F;
            const bool meshWet = farWaterTopCovers(*right, localX, localZ);
            broadDry += neighborhoodDry;
            broadWet += neighborhoodWet;
            falseWater += neighborhoodDry && meshWet;
            missingWater += neighborhoodWet && !meshWet;
        }
    }
    CAPTURE(broadDry, broadWet, falseWater, missingWater, *geometryPointCount, *authorityPointCount,
            *authorityGridCount, *canonicalPointCount, *canonicalGridCount, right->waterQuadCount,
            right->waterContourTriangleCount);
    REQUIRE(broadDry > 100);
    REQUIRE(broadWet > 100);
    REQUIRE(falseWater == 0);
    REQUIRE(missingWater == 0);
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        for (int x = 1; x + 1 < SAMPLE_EDGE; ++x) {
            const float localX = static_cast<float>((x - 1) * SAMPLE_STEP) + 0.5F;
            const float localZ = static_cast<float>((z - 1) * SAMPLE_STEP) + 0.5F;
            CAPTURE(x, z, localX, localZ);
            REQUIRE(farWaterTopCovers(*right, localX, localZ) == exactWet(x, z));
        }
    }
    REQUIRE(left->waterContourTriangleCount < 1'024);
    REQUIRE(right->waterContourTriangleCount < 1'024);

    size_t sharedWet = 0;
    for (int z = 1; z + 1 < SAMPLE_EDGE; ++z) {
        bool neighborhoodDry = true;
        bool neighborhoodWet = true;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int x = 0; x <= 2; ++x) {
                neighborhoodDry = neighborhoodDry && !exactWet(x, z + dz);
                neighborhoodWet = neighborhoodWet && exactWet(x, z + dz);
            }
        }
        if (!neighborhoodDry && !neighborhoodWet)
            continue;
        const int localZ = (z - 1) * SAMPLE_STEP;
        const float sampleZ = static_cast<float>(localZ) + 0.5F;
        const bool leftWet =
            farWaterTopCovers(*left, static_cast<float>(FAR_TERRAIN_TILE_EDGE), sampleZ);
        const bool rightWet = farWaterTopCovers(*right, 0.0F, sampleZ);
        CAPTURE(localZ, neighborhoodDry, neighborhoodWet, leftWet, rightWet);
        REQUIRE((leftWet || rightWet) == neighborhoodWet);
        sharedWet += neighborhoodWet;
    }
    REQUIRE(sharedWet > 4);
    constexpr int NATIVE_WATER_EDGE =
        FAR_TERRAIN_TILE_EDGE / worldgen::NATIVE_HYDROLOGY_RASTER_SPACING + 2;
    constexpr int SHARED_FACE_SAMPLES = 4 * (FAR_TERRAIN_TILE_EDGE / 2 + 1) - 4;
    REQUIRE(*canonicalGridCount ==
            2 * (NATIVE_WATER_EDGE * NATIVE_WATER_EDGE + SHARED_FACE_SAMPLES));
    REQUIRE(*canonicalPointCount > 0);
    REQUIRE(*canonicalPointCount < 10'000);
    // Exact terrain contact is now reserved for bounded volcanic or handoff
    // exceptions. The distant parent must not regress to a density query for
    // every canonical water point.
    REQUIRE(*authorityGridCount == 0);
    REQUIRE(*authorityPointCount == 0);
    REQUIRE(*geometryPointCount + *authorityPointCount + *authorityGridCount < 70'000);
}

TEST_CASE("Seed 42 step 32 water cells refine topology across twenty five tiles",
          "[render][far-terrain][water][coverage][ownership][seam][voxel][regression][seed-42]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    constexpr int64_t FIRST_TILE_X = -1;
    constexpr int64_t FIRST_TILE_Z = -8;
    constexpr int TILE_EDGE = 5;
    constexpr int WATER_STEP = 8;
    constexpr int WATER_CELLS = FAR_TERRAIN_TILE_EDGE / WATER_STEP;
    constexpr int AUTHORITY_EDGE = TILE_EDGE * WATER_CELLS + 1;
    constexpr int64_t AUTHORITY_ORIGIN_X = FIRST_TILE_X * FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t AUTHORITY_ORIGIN_Z = FIRST_TILE_Z * FAR_TERRAIN_TILE_EDGE;
    std::vector<worldgen::HydrologySample> authority(
        static_cast<size_t>(AUTHORITY_EDGE * AUTHORITY_EDGE));
    generator->sampleGeneratedWaterAuthorityGrid(AUTHORITY_ORIGIN_X, AUTHORITY_ORIGIN_Z, WATER_STEP,
                                                 AUTHORITY_EDGE, authority);
    const auto authorityAt = [&](int tileX, int tileZ, int cellX, int cellZ) {
        const int sampleX = (tileX - FIRST_TILE_X) * WATER_CELLS + cellX;
        const int sampleZ = (tileZ - FIRST_TILE_Z) * WATER_CELLS + cellZ;
        return authority[static_cast<size_t>(sampleZ * AUTHORITY_EDGE + sampleX)];
    };
    size_t wetCells = 0;
    size_t dryCells = 0;
    for (int tileZ = FIRST_TILE_Z; tileZ < FIRST_TILE_Z + TILE_EDGE; ++tileZ) {
        for (int tileX = FIRST_TILE_X; tileX < FIRST_TILE_X + TILE_EDGE; ++tileX) {
            const auto mesh =
                FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
            CAPTURE(tileX, tileZ, mesh->waterQuadCount, mesh->waterContourTriangleCount);
            REQUIRE(mesh->waterContourTriangleCount < 2'048);
            for (int cellZ = 0; cellZ < WATER_CELLS; ++cellZ) {
                for (int cellX = 0; cellX < WATER_CELLS; ++cellX) {
                    const worldgen::HydrologySample hydrology =
                        authorityAt(tileX, tileZ, cellX, cellZ);
                    const bool expectedWet =
                        (hydrology.ocean || hydrology.river || hydrology.lake ||
                         hydrology.wetland) &&
                        hydrology.waterSurface > hydrology.surfaceElevation + 0.01;
                    const float localX = static_cast<float>(cellX * WATER_STEP) + 0.5F;
                    const float localZ = static_cast<float>(cellZ * WATER_STEP) + 0.5F;
                    CAPTURE(cellX, cellZ, localX, localZ, expectedWet, hydrology.waterBodyId,
                            hydrology.waterSurface, hydrology.surfaceElevation);
                    REQUIRE(farWaterTopCovers(*mesh, localX, localZ) == expectedWet);
                    wetCells += expectedWet;
                    dryCells += !expectedWet;
                }
            }
            for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size();
                 offset += 3) {
                const Vertex& first = mesh->vertices[mesh->indices[offset]];
                if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
                    unpackFluidFalling(first.faceAttr)) {
                    continue;
                }
                const Vertex& second = mesh->vertices[mesh->indices[offset + 1]];
                const Vertex& third = mesh->vertices[mesh->indices[offset + 2]];
                CAPTURE(tileX, tileZ, static_cast<float>(first.px), static_cast<float>(first.py),
                        static_cast<float>(first.pz));
                REQUIRE(static_cast<float>(first.py) == static_cast<float>(second.py));
                REQUIRE(static_cast<float>(first.py) == static_cast<float>(third.py));
                REQUIRE(std::fmod(static_cast<float>(first.py) * 8.0F, 1.0F) ==
                        Catch::Approx(0.0F).margin(1.0e-4F));
                const float signedArea =
                    (static_cast<float>(second.px) - static_cast<float>(first.px)) *
                        (static_cast<float>(third.pz) - static_cast<float>(first.pz)) -
                    (static_cast<float>(second.pz) - static_cast<float>(first.pz)) *
                        (static_cast<float>(third.px) - static_cast<float>(first.px));
                // Uniform parents may merge to a 32-block quad, while a
                // topology-marked parent emits exact row runs and clipped
                // shoreline triangles. Every top retains positive area rather
                // than falling back to a phase-zero 8x8 assumption.
                REQUIRE(std::abs(signedArea) > 0.0F);
                REQUIRE(std::abs(signedArea) <= 1'024.0F);
            }
        }
    }
    REQUIRE(wetCells > 1'000);
    REQUIRE(dryCells > 1'000);

    constexpr std::array REPORTED_COLUMNS = {
        ColumnPos{-8, -1'896},  ColumnPos{8, -1'896},  ColumnPos{368, -1'880},
        ColumnPos{360, -1'864}, ColumnPos{-8, -1'264}, ColumnPos{248, -1'608},
    };
    for (const ColumnPos position : REPORTED_COLUMNS) {
        const worldgen::HydrologySample direct =
            generator->sampleGeneratedWaterAuthority(position.x, position.z);
        const worldgen::SurfaceSample exact = generator->sampleExactSurface(position.x, position.z);
        CAPTURE(position.x, position.z, direct.waterBodyId, exact.hydrology.waterBodyId);
        REQUIRE(direct.waterBodyId == exact.hydrology.waterBodyId);
        REQUIRE(direct.ocean == exact.hydrology.ocean);
        REQUIRE(direct.river == exact.hydrology.river);
        REQUIRE(direct.lake == exact.hydrology.lake);
    }
}

TEST_CASE("Step 32 shared water risers have one positive-side owner",
          "[render][far-terrain][water][coverage][seam][riser][negative][regression]") {
    const auto boundaryRisers = [](const FarTerrainMesh& mesh, FaceNormal face, float localX) {
        size_t result = 0;
        for (size_t offset = mesh.opaqueIndexCount; offset + 5 < mesh.indices.size(); offset += 6) {
            const Vertex& first = mesh.vertices[mesh.indices[offset]];
            if (unpackFace(first.faceAttr) != face || static_cast<float>(first.px) != localX) {
                continue;
            }
            ++result;
        }
        return result;
    };
    const auto flowingSource = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = x < 0 ? 64.875 : 64.625;
            sample.river = true;
            sample.generatedFluidLevel = x < 0 ? 1 : 3;
            sample.transitionOwnerKind = worldgen::WaterTransitionKind::RASTER_CHANNEL;
            sample.transitionOwnerId = 0x5154'4147'4552'4953ULL;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto left = FarTerrainMesher::build({-1, -1, FarTerrainStep::THIRTY_TWO}, flowingSource);
    const auto right = FarTerrainMesher::build({0, -1, FarTerrainStep::THIRTY_TWO}, flowingSource);
    REQUIRE(left->waterContourTriangleCount == 0);
    REQUIRE(right->waterContourTriangleCount == 0);
    REQUIRE(boundaryRisers(*left, FaceNormal::PLUS_X, static_cast<float>(FAR_TERRAIN_TILE_EDGE)) ==
            0);
    // V4 carries canonical hydrology on a four-block raster, so a 256-block
    // shared edge owns 64 independently sampled legal flow transitions.
    REQUIRE(boundaryRisers(*right, FaceNormal::PLUS_X, 0.0F) == 64);
    REQUIRE(boundaryRisers(*right, FaceNormal::MINUS_X, 0.0F) == 0);

    const auto shorelineSource = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 48.0;
            sample.waterSurface = 64.0;
            sample.ocean = x < 0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
    const auto wet = FarTerrainMesher::build({-1, -1, FarTerrainStep::THIRTY_TWO}, shorelineSource);
    const auto dry = FarTerrainMesher::build({0, -1, FarTerrainStep::THIRTY_TWO}, shorelineSource);
    REQUIRE(boundaryRisers(*wet, FaceNormal::PLUS_X, static_cast<float>(FAR_TERRAIN_TILE_EDGE)) ==
            0);
    REQUIRE(boundaryRisers(*dry, FaceNormal::PLUS_X, 0.0F) == 0);
    REQUIRE(boundaryRisers(*dry, FaceNormal::MINUS_X, 0.0F) == 0);
}

TEST_CASE("Canonical lake contours agree through every far LOD",
          "[render][far-terrain][water][lake][seam][lod]") {
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });
    constexpr FarTerrainKey BASE_KEY{-32, 8, FarTerrainStep::TWO};
    constexpr int64_t WORLD_Z = 2'288;
    constexpr int64_t WET_X = -8'192;
    constexpr int64_t SCAN_END_X = -8'160;
    constexpr float LOCAL_Z = 240.0F;
    constexpr float MINIMUM_LOCAL_X = 0.0F;
    constexpr float MAXIMUM_LOCAL_X = 32.0F;
    constexpr int64_t TILE_ORIGIN_X = BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE;

    bool foundWetReference = false;
    int64_t firstDryReferenceX = std::numeric_limits<int64_t>::max();
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const FarTerrainGeometrySample sample =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_2);
        if (sample.lake) {
            foundWetReference = true;
        } else if (foundWetReference) {
            firstDryReferenceX = x;
            break;
        }
    }
    REQUIRE(foundWetReference);
    REQUIRE(firstDryReferenceX != std::numeric_limits<int64_t>::max());
    const float expectedEdgeX = static_cast<float>(firstDryReferenceX - TILE_ORIGIN_X) - 0.5F;
    const float expectedWaterY =
        static_cast<float>(testFarGeometry(source, firstDryReferenceX - 1, WORLD_Z,
                                           worldgen::SurfaceFootprint::BLOCK_2)
                               .waterSurface);

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const int spacing = farTerrainStepSize(step);
        REQUIRE(world_coord::floorMod(WET_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
        REQUIRE(world_coord::floorMod(SCAN_END_X - BASE_KEY.tileX * FAR_TERRAIN_TILE_EDGE,
                                      static_cast<int64_t>(spacing)) == 0);
    }
    for (int64_t x = WET_X; x <= SCAN_END_X; ++x) {
        const worldgen::SurfaceSample exactAuthority = generator.sampleExactSurface(x, WORLD_Z);
        const worldgen::SurfaceSample coarseAuthority =
            generator.sampleFarSurface(x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
        REQUIRE(exactAuthority.hydrology.lake == coarseAuthority.hydrology.lake);
        REQUIRE(exactAuthority.hydrology.waterBodyId == coarseAuthority.hydrology.waterBodyId);
        REQUIRE(exactAuthority.waterSurface ==
                Catch::Approx(coarseAuthority.waterSurface).margin(1.0e-5));
        const FarTerrainGeometrySample exact =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_2);
        const FarTerrainGeometrySample coarse =
            testFarGeometry(source, x, WORLD_Z, worldgen::SurfaceFootprint::BLOCK_16);
        REQUIRE(exact.lake == coarse.lake);
        REQUIRE(exact.waterSurface == Catch::Approx(coarse.waterSurface).margin(1.0e-5));
    }

    const auto rightmostWaterAtCut = [=](const FarTerrainMesh& mesh) {
        float rightmost = -std::numeric_limits<float>::infinity();
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<Vec3, 3> triangle{};
            bool waterTriangle = true;
            for (size_t corner = 0; corner < triangle.size(); ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                waterTriangle =
                    waterTriangle && unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                    unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::WATER) &&
                    static_cast<float>(vertex.py) == expectedWaterY;
                triangle[corner] = {static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                                    static_cast<float>(vertex.pz)};
            }
            if (!waterTriangle)
                continue;
            for (size_t edge = 0; edge < triangle.size(); ++edge) {
                const Vec3& first = triangle[edge];
                const Vec3& second = triangle[(edge + 1) % triangle.size()];
                if (first.z == LOCAL_Z && second.z == LOCAL_Z) {
                    for (const float x : {first.x, second.x}) {
                        if (x >= MINIMUM_LOCAL_X && x <= MAXIMUM_LOCAL_X) {
                            rightmost = std::max(rightmost, x);
                        }
                    }
                    continue;
                }
                if ((first.z < LOCAL_Z && second.z < LOCAL_Z) ||
                    (first.z > LOCAL_Z && second.z > LOCAL_Z) || first.z == second.z) {
                    continue;
                }
                const float amount = (LOCAL_Z - first.z) / (second.z - first.z);
                if (amount < 0.0F || amount > 1.0F)
                    continue;
                const float x = first.x + (second.x - first.x) * amount;
                if (x >= MINIMUM_LOCAL_X && x <= MAXIMUM_LOCAL_X) {
                    rightmost = std::max(rightmost, x);
                }
            }
        }
        return rightmost;
    };

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(rightmostWaterAtCut(*mesh) == Catch::Approx(expectedEdgeX).margin(0.51));
    }
}

TEST_CASE("Far lake outlets use narrow explicit falling prisms at every LOD",
          "[render][far-terrain][water][lake][waterfall][seam][lod]") {
    constexpr int64_t FALL_X = -8'240;
    constexpr int64_t FALL_Z = 3'088;
    constexpr FarTerrainKey BASE_KEY{-33, 12, FarTerrainStep::TWO};
    constexpr float LOCAL_FALL_X = 208.0F;
    constexpr float LOCAL_FALL_Z = 16.0F;
    ChunkGenerator generator(42);
    const FarTerrainSource source = FarTerrainMesher::tieredSurfaceGeometrySource(
        [&generator](int64_t x, int64_t z) { return generator.sampleExactSurface(x, z); },
        [&generator](int64_t x, int64_t z) { return generator.sampleFarSurface(x, z); });

    const FarTerrainGeometrySample near =
        testFarGeometry(source, FALL_X, FALL_Z, worldgen::SurfaceFootprint::BLOCK_2);
    const FarTerrainGeometrySample coarse =
        testFarGeometry(source, FALL_X, FALL_Z, worldgen::SurfaceFootprint::BLOCK_16);
    for (const FarTerrainGeometrySample* sample : {&near, &coarse}) {
        REQUIRE(sample->ocean);
        REQUIRE_FALSE(sample->river);
        REQUIRE(sample->waterfall);
        REQUIRE(sample->waterfallAnchor);
        REQUIRE(sample->waterSurface == Catch::Approx(std::ceil(sample->waterfallBottom) - 1.0 +
                                                      fluidSurfaceHeight(FluidState::source())));
        REQUIRE(sample->waterfallBottom == Catch::Approx(SEA_LEVEL).margin(1.0e-4));
        REQUIRE(sample->waterfallTop == Catch::Approx(81.14503479).margin(1.0e-4));
        REQUIRE(sample->waterfallWidth >= 4.0);
    }

    const double flowLength = std::hypot(coarse.flowX, coarse.flowZ);
    REQUIRE(flowLength > 0.0);
    const double flowX = coarse.flowX / flowLength;
    const double flowZ = coarse.flowZ / flowLength;
    const int outsideOffset = static_cast<int>(std::ceil(coarse.waterfallWidth + 4.0));
    const int64_t outsideX = FALL_X + static_cast<int64_t>(std::llround(-flowZ * outsideOffset));
    const int64_t outsideZ = FALL_Z + static_cast<int64_t>(std::llround(flowX * outsideOffset));
    const FarTerrainGeometrySample outside =
        testFarGeometry(source, outsideX, outsideZ, worldgen::SurfaceFootprint::BLOCK_16);
    REQUIRE_FALSE(outside.waterfall);
    REQUIRE(outside.ocean);
    REQUIRE_FALSE(outside.lake);
    REQUIRE_FALSE(outside.river);

    for (const FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR,
                                      FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(mesh->waterfallQuadCount >= 5);
        float minimumFallingY = std::numeric_limits<float>::max();
        float maximumFallingY = std::numeric_limits<float>::lowest();
        double minimumAlong = std::numeric_limits<double>::max();
        double maximumAlong = std::numeric_limits<double>::lowest();
        size_t pinnedVertices = 0;
        size_t verticalVertices = 0;
        for (const Vertex& vertex : mesh->vertices) {
            if (!unpackFluidFalling(vertex.faceAttr))
                continue;
            const float localX = static_cast<float>(vertex.px);
            const float localZ = static_cast<float>(vertex.pz);
            if (std::abs(localX - LOCAL_FALL_X) > 12.0F ||
                std::abs(localZ - LOCAL_FALL_Z) > 12.0F) {
                continue;
            }
            const double offsetX = localX - LOCAL_FALL_X;
            const double offsetZ = localZ - LOCAL_FALL_Z;
            const double along = offsetX * flowX + offsetZ * flowZ;
            const double cross = -offsetX * flowZ + offsetZ * flowX;
            minimumAlong = std::min(minimumAlong, along);
            maximumAlong = std::max(maximumAlong, along);
            REQUIRE(std::abs(cross) <= coarse.waterfallWidth * 0.5 + 0.75);
            minimumFallingY = std::min(minimumFallingY, static_cast<float>(vertex.py));
            maximumFallingY = std::max(maximumFallingY, static_cast<float>(vertex.py));
            if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
                ++verticalVertices;
            ++pinnedVertices;
        }
        REQUIRE(pinnedVertices == 20);
        REQUIRE(verticalVertices == 16);
        REQUIRE(maximumAlong - minimumAlong <= 3.1);
        REQUIRE(minimumFallingY <= near.waterSurface);
        REQUIRE(maximumFallingY == Catch::Approx(std::ceil(near.waterfallTop)));

        const auto rebuilt =
            FarTerrainMesher::build({BASE_KEY.tileX, BASE_KEY.tileZ, step}, source);
        REQUIRE(rebuilt->deterministicHash == mesh->deterministicHash);
    }
}

TEST_CASE("Far terrain retains deterministic forests through every LOD",
          "[render][far-terrain][canopy][lod]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        std::vector<FarCanopy> canopies;
        size_t retained = 2;
        switch (step) {
            case FarTerrainStep::ONE:
            case FarTerrainStep::TWO:
                retained = 6;
                break;
            case FarTerrainStep::FOUR:
                retained = 5;
                break;
            case FarTerrainStep::EIGHT:
                retained = 4;
                break;
            case FarTerrainStep::SIXTEEN:
                retained = 3;
                break;
            case FarTerrainStep::THIRTY_TWO:
                retained = 2;
                break;
        }
        for (uint64_t index = 0; index < retained; ++index) {
            FarCanopy canopy;
            canopy.x = 20 + static_cast<int64_t>(index) * 40;
            canopy.z = 96;
            canopy.baseY = 64;
            canopy.topY = 70;
            canopy.canopyMinimumY = 67;
            canopy.canopyMaximumY = 72;
            canopy.canopyRadius = 2;
            canopy.logBlock = BlockType::LOG;
            canopy.leafBlock = BlockType::LEAVES;
            canopy.anchorId = index;
            canopies.push_back(canopy);
        }
        FarCanopy neighboringCanopy = canopies.front();
        neighboringCanopy.x = -1;
        neighboringCanopy.anchorId = 5;
        canopies.push_back(neighboringCanopy);
        return canopies;
    };

    const auto two = FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source);
    const auto four = FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::FOUR}, source);
    const auto eight =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::EIGHT}, source);
    const auto sixteen =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto thirtyTwo =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(two->canopyAnchorCount == 6);
    REQUIRE(four->canopyAnchorCount == 5);
    REQUIRE(eight->canopyAnchorCount == 4);
    REQUIRE(sixteen->canopyAnchorCount == 3);
    REQUIRE(thirtyTwo->canopyAnchorCount == 2);

    for (const auto& attachment : {two, four, eight, sixteen, thirtyTwo}) {
        REQUIRE(attachment->canopyImpostorQuadCount == attachment->canopyAnchorCount * 19);
        const size_t canopyVertexCount =
            static_cast<size_t>(attachment->canopyImpostorQuadCount) * 4;
        REQUIRE(std::count_if(attachment->vertices.begin(), attachment->vertices.end(),
                              [](const Vertex& vertex) {
                                  return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) !=
                                         0U;
                              }) == static_cast<std::ptrdiff_t>(canopyVertexCount));
        REQUIRE(attachment->bounds.maxY == 73.0F);
    }
    REQUIRE(two->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source)
                ->deterministicHash);
}

TEST_CASE("Step-two exact-anchor canopies reconstruct their displayed voxel ground",
          "[render][far-terrain][canopy][lod][grounding][performance][regression]") {
    size_t blockTwoSamples = 0;
    FarTerrainSource source;
    source.sample = [&](int64_t x, int64_t, worldgen::SurfaceFootprint footprint) {
        if (footprint == worldgen::SurfaceFootprint::BLOCK_2)
            ++blockTwoSamples;
        const double terrainHeight = 64.0 + static_cast<double>(x) * 0.5;
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = terrainHeight;
        geometry.waterSurface = SEA_LEVEL;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = terrainHeight,
            .footprintMaximumTerrainHeight = terrainHeight,
            .materialPalette = testMaterialPalette(BlockType::GRASS),
        };
    };
    FarTerrainMesher::build({0, 0, FarTerrainStep::TWO}, source);
    const size_t baselineBlockTwoSamples = blockTwoSamples;
    REQUIRE(baselineBlockTwoSamples > 1);
    blockTwoSamples = 0;

    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{FarCanopy{
            .x = 5,
            .z = 5,
            .baseY = 65,
            .topY = 73,
            .canopyMinimumY = 69,
            .canopyMaximumY = 73,
            .canopyRadius = 2,
            .logBlock = BlockType::LOG,
            .leafBlock = BlockType::LEAVES,
            .anchorId = 1,
            .aggregate = false,
        }};
    };

    const auto attachment =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source);
    REQUIRE(blockTwoSamples == 4);
    REQUIRE(attachment->canopyAnchorCount == 1);

    float minimumCanopyY = std::numeric_limits<float>::max();
    for (const Vertex& vertex : attachment->vertices) {
        if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U)
            continue;
        minimumCanopyY = std::min(minimumCanopyY, static_cast<float>(vertex.py));
    }
    // The anchor lies in the [4, 6) voxel cell. Its four canonical corners
    // reproduce the displayed filtered top of 67 without sampling an
    // unrelated point beneath the trunk.
    REQUIRE(minimumCanopyY == 67.0F);
}

TEST_CASE("Far canopies keep one half-open owner across signed tile boundaries",
          "[render][far-terrain][canopy][ownership][seam][lod][flicker][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    const auto canopy = [](int64_t x, BlockType logBlock, BlockType leafBlock, uint64_t anchorId) {
        return FarCanopy{
            .x = x,
            .z = 96,
            .baseY = 65,
            .topY = 74,
            .canopyMinimumY = 69,
            .canopyMaximumY = 74,
            .canopyRadius = 3,
            .logBlock = logBlock,
            .leafBlock = leafBlock,
            .anchorId = anchorId,
            .aggregate = true,
        };
    };
    const std::array canopies = {
        canopy(-257, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES, 1),
        canopy(-256, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES, 2),
        canopy(-1, BlockType::PALM_LOG, BlockType::PALM_LEAVES, 3),
        canopy(0, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES, 4),
        canopy(255, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES, 5),
        canopy(256, BlockType::PALM_LOG, BlockType::PALM_LEAVES, 6),
    };
    source.canopies = [canopies](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector<FarCanopy>(canopies.begin(), canopies.end());
    };

    const auto canopyVertices = [](const FarCanopyAttachment& attachment) {
        return attachment.vertices;
    };
    const auto sameVertices = [](std::span<const Vertex> first, std::span<const Vertex> second) {
        return first.size() == second.size() &&
               std::equal(first.begin(), first.end(), second.begin(),
                          [](const Vertex& lhs, const Vertex& rhs) {
                              return lhs.faceAttr == rhs.faceAttr &&
                                     static_cast<float>(lhs.px) == static_cast<float>(rhs.px) &&
                                     static_cast<float>(lhs.py) == static_cast<float>(rhs.py) &&
                                     static_cast<float>(lhs.pz) == static_cast<float>(rhs.pz) &&
                                     static_cast<float>(lhs.u) == static_cast<float>(rhs.u) &&
                                     static_cast<float>(lhs.v) == static_cast<float>(rhs.v);
                          });
    };

    constexpr std::array TILE_X = {-2LL, -1LL, 0LL, 1LL};
    constexpr std::array EXPECTED_OWNERS = {1U, 2U, 2U, 1U};
    size_t fineOwners = 0;
    size_t coarseOwners = 0;
    for (size_t index = 0; index < TILE_X.size(); ++index) {
        const auto fine = FarTerrainMesher::buildCanopyAttachment(
            {TILE_X[index], 0, FarTerrainStep::TWO}, source);
        const auto coarse = FarTerrainMesher::buildCanopyAttachment(
            {TILE_X[index], 0, FarTerrainStep::THIRTY_TWO}, source);
        CAPTURE(TILE_X[index]);
        REQUIRE(fine->canopyAnchorCount == EXPECTED_OWNERS[index]);
        REQUIRE(coarse->canopyAnchorCount == EXPECTED_OWNERS[index]);
        fineOwners += fine->canopyAnchorCount;
        coarseOwners += coarse->canopyAnchorCount;

        // Each owner emits the complete species silhouette even when its crown
        // crosses the tile face. The adjacent loaded tile does not emit a
        // replacement, and changing far tiers does not resize or relocate it.
        const std::vector<Vertex> fineCanopyVertices = canopyVertices(*fine);
        const std::vector<Vertex> coarseCanopyVertices = canopyVertices(*coarse);
        REQUIRE_FALSE(fineCanopyVertices.empty());
        REQUIRE(sameVertices(fineCanopyVertices, coarseCanopyVertices));
        for (const Vertex& vertex : fineCanopyVertices) {
            const unsigned int face = static_cast<unsigned int>(unpackFace(vertex.faceAttr));
            REQUIRE_FALSE(farTerrainOpaqueRiserUsesEmittingColumn(face, true));
            const simd_float2 position =
                simd_make_float2(static_cast<float>(vertex.px), static_cast<float>(vertex.pz));
            const simd_float2 ownershipSample = farTerrainExactOwnershipSamplePosition(
                position, face, farTerrainOpaqueRiserUsesEmittingColumn(face, true));
            REQUIRE(ownershipSample.x == position.x);
            REQUIRE(ownershipSample.y == position.y);
        }
        const auto [minimumX, maximumX] =
            std::minmax_element(fineCanopyVertices.begin(), fineCanopyVertices.end(),
                                [](const Vertex& lhs, const Vertex& rhs) {
                                    return static_cast<float>(lhs.px) < static_cast<float>(rhs.px);
                                });
        REQUIRE((static_cast<float>(minimumX->px) < 0.0F ||
                 static_cast<float>(maximumX->px) > FAR_TERRAIN_TILE_EDGE_BLOCKS));
    }
    REQUIRE(fineOwners == canopies.size());
    REQUIRE(coarseOwners == canopies.size());
}

TEST_CASE("Far canopy leaf materials produce distinct voxel silhouettes",
          "[render][far-terrain][canopy][species][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        const auto canopy = [](int64_t x, BlockType logBlock, BlockType leafBlock,
                               feature_generation::TreeSpecies species, uint64_t anchorId) {
            return FarCanopy{
                .x = x,
                .z = 96,
                .baseY = 64,
                .topY = 73,
                .canopyMinimumY = 68,
                .canopyMaximumY = 73,
                .canopyRadius = 3,
                .logBlock = logBlock,
                .leafBlock = leafBlock,
                .anchorId = anchorId,
                .species = species,
            };
        };
        return std::vector{
            canopy(32, BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES,
                   feature_generation::TreeSpecies::SPRUCE, 1),
            canopy(96, BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES,
                   feature_generation::TreeSpecies::ACACIA, 5),
            canopy(160, BlockType::PALM_LOG, BlockType::PALM_LEAVES,
                   feature_generation::TreeSpecies::PALM, 9),
            // Species is authoritative even when a material differs. This
            // deliberately mismatched fixture must keep an oak silhouette
            // while using jungle leaves as its texture layer.
            canopy(224, BlockType::LOG, BlockType::JUNGLE_LEAVES,
                   feature_generation::TreeSpecies::OAK, 13),
        };
    };

    const auto attachment =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source);
    struct TopSpan {
        float width = 0.0F;
        float depth = 0.0F;
    };
    const auto topSpans = [&](BlockType leafBlock) {
        struct Extents {
            float minimumX = std::numeric_limits<float>::max();
            float maximumX = std::numeric_limits<float>::lowest();
            float minimumZ = std::numeric_limits<float>::max();
            float maximumZ = std::numeric_limits<float>::lowest();
        };
        std::map<float, Extents> byHeight;
        for (const Vertex& vertex : attachment->vertices) {
            if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U ||
                unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(leafBlock) ||
                unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y) {
                continue;
            }
            Extents& extents = byHeight[static_cast<float>(vertex.py)];
            extents.minimumX = std::min(extents.minimumX, static_cast<float>(vertex.px));
            extents.maximumX = std::max(extents.maximumX, static_cast<float>(vertex.px));
            extents.minimumZ = std::min(extents.minimumZ, static_cast<float>(vertex.pz));
            extents.maximumZ = std::max(extents.maximumZ, static_cast<float>(vertex.pz));
        }
        std::vector<TopSpan> result;
        for (const auto& [height, extents] : byHeight) {
            (void)height;
            result.push_back({.width = extents.maximumX - extents.minimumX,
                              .depth = extents.maximumZ - extents.minimumZ});
        }
        return result;
    };

    const std::vector<TopSpan> spruce = topSpans(BlockType::SPRUCE_LEAVES);
    REQUIRE(spruce.size() == 4);
    REQUIRE(spruce[0].width > spruce[1].width);
    REQUIRE(spruce[1].width > spruce[2].width);
    REQUIRE(spruce[2].width > spruce[3].width);
    REQUIRE(std::ranges::all_of(spruce, [](TopSpan span) { return span.width == span.depth; }));

    const std::vector<TopSpan> acacia = topSpans(BlockType::ACACIA_LEAVES);
    REQUIRE(acacia.size() == 2);
    REQUIRE(acacia[0].width > acacia[1].width);
    REQUIRE(acacia[0].width == acacia[0].depth);
    REQUIRE(acacia[1].width == acacia[1].depth);

    const std::vector<TopSpan> palm = topSpans(BlockType::PALM_LEAVES);
    REQUIRE(palm.size() == 3);
    REQUIRE(palm[0].width > palm[0].depth);
    REQUIRE(palm[1].width < palm[1].depth);
    REQUIRE(palm[2].width == palm[2].depth);
    REQUIRE(palm[2].width < palm[0].width);

    const std::vector<TopSpan> authoritativeOak = topSpans(BlockType::JUNGLE_LEAVES);
    REQUIRE(authoritativeOak.size() == 3);
    REQUIRE(authoritativeOak[0].width < authoritativeOak[1].width);
    REQUIRE(authoritativeOak[1].width > authoritativeOak[2].width);
    REQUIRE(authoritativeOak[0].width == authoritativeOak[2].width);
    REQUIRE(std::ranges::all_of(authoritativeOak,
                                [](TopSpan span) { return span.width == span.depth; }));
    REQUIRE(attachment->canopyImpostorQuadCount == 76);
    REQUIRE(attachment->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source)
                ->deterministicHash);
}

TEST_CASE("Step-two fallen logs retain their exact horizontal morphology",
          "[render][far-terrain][canopy][species][fallen-log][lod][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{FarCanopy{
            .x = 32,
            .z = 48,
            .baseY = 65,
            .topY = 65,
            .canopyMinimumY = 65,
            .canopyMaximumY = 65,
            .logBlock = BlockType::WILLOW_LOG,
            .anchorId = 17,
            .species = feature_generation::TreeSpecies::FALLEN_LOG,
            .formX = 1,
            .formExtent = 7,
        }};
    };

    const auto fine = FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source);
    const auto coarse =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(fine->canopyAnchorCount == 1);
    REQUIRE(fine->canopyImpostorQuadCount == 5);
    REQUIRE(fine->canopyImpostorQuadCount == coarse->canopyImpostorQuadCount);

    float minimumX = std::numeric_limits<float>::max();
    float maximumX = std::numeric_limits<float>::lowest();
    float minimumY = std::numeric_limits<float>::max();
    float maximumY = std::numeric_limits<float>::lowest();
    float minimumZ = std::numeric_limits<float>::max();
    float maximumZ = std::numeric_limits<float>::lowest();
    for (const Vertex& vertex : fine->vertices) {
        if ((vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) == 0U)
            continue;
        minimumX = std::min(minimumX, static_cast<float>(vertex.px));
        maximumX = std::max(maximumX, static_cast<float>(vertex.px));
        minimumY = std::min(minimumY, static_cast<float>(vertex.py));
        maximumY = std::max(maximumY, static_cast<float>(vertex.py));
        minimumZ = std::min(minimumZ, static_cast<float>(vertex.pz));
        maximumZ = std::max(maximumZ, static_cast<float>(vertex.pz));
    }
    REQUIRE(maximumX - minimumX == 7.0F);
    REQUIRE(maximumY - minimumY == 1.0F);
    REQUIRE(maximumZ - minimumZ == 1.0F);
    REQUIRE(fine->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source)
                ->deterministicHash);
}

TEST_CASE("Production step-two meshing uses shared exact tree roots without scalar basin work",
          "[render][far-terrain][canopy][worldgen][performance][handoff][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const uint64_t scalarCallsBefore = generator->basinCacheMetrics().scalarSampleCalls;
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const auto attachment =
        FarTerrainMesher::buildCanopyAttachment({-1, 0, FarTerrainStep::TWO}, source);
    const std::vector<FarCanopy> canopies = generator->collectFarCanopiesForLod(-256, 0, 0, 256, 2);
    const std::vector<FarCanopy> stepOne = generator->collectFarCanopiesForLod(-256, 0, 0, 256, 1);
    const size_t owned = std::ranges::count_if(canopies, [](const FarCanopy& canopy) {
        return canopy.x >= -256 && canopy.x < 0 && canopy.z >= 0 && canopy.z < 256;
    });
    REQUIRE(owned > 0);
    REQUIRE(stepOne == canopies);
    REQUIRE(attachment->canopyAnchorCount == owned);
    REQUIRE(generator->cachedColumnPlanCount() == 0);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    generator->clearMacroCaches();
    const auto rebuilt =
        FarTerrainMesher::buildCanopyAttachment({-1, 0, FarTerrainStep::TWO}, source);
    REQUIRE(rebuilt->deterministicHash == attachment->deterministicHash);
    REQUIRE(rebuilt->canopyAnchorCount == attachment->canopyAnchorCount);
}

TEST_CASE("Far flora diagnostics isolate collection grounding and geometry work",
          "[render][far-terrain][canopy][flora][diagnostics][performance]") {
    FarTerrainSource source;
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint) {
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = 64.0;
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = 64.0,
            .footprintMaximumTerrainHeight = 64.0,
            .materialPalette = testMaterialPalette(BlockType::GRASS),
        };
    };
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        FarCanopy inside{
            .x = 32,
            .z = 32,
            .baseY = 65,
            .topY = 72,
            .canopyMinimumY = 68,
            .canopyMaximumY = 72,
            .canopyRadius = 2,
            .logBlock = BlockType::LOG,
            .leafBlock = BlockType::LEAVES,
            .anchorId = 1,
        };
        FarCanopy outside = inside;
        outside.x = FAR_TERRAIN_TILE_EDGE;
        outside.anchorId = 2;
        return std::vector{inside, outside};
    };
    source.flora = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{
            FarFlora{.x = 40,
                     .z = 40,
                     .baseY = 65,
                     .block = BlockType::TALL_GRASS,
                     .height = 1,
                     .anchorId = 3},
            FarFlora{.x = FAR_TERRAIN_TILE_EDGE,
                     .z = 40,
                     .baseY = 65,
                     .block = BlockType::TALL_GRASS,
                     .height = 1,
                     .anchorId = 4},
        };
    };
    size_t sparseCells = 0;
    source.terrainCellTopPoints = [&](int64_t, int64_t, int, int, worldgen::SurfaceFootprint,
                                      std::span<const uint32_t> occupied, std::span<float> output) {
        sparseCells = occupied.size();
        REQUIRE(output.size() == occupied.size());
        std::fill(output.begin(), output.end(), 65.0F);
    };
    source.terrainCellTopGrid = [](int64_t, int64_t, int, int, worldgen::SurfaceFootprint,
                                   std::span<float>) {
        throw std::logic_error("dense grounding should not run for two occupied cells");
    };

    FarCanopyBuildDiagnostics diagnostics;
    const auto attachment = FarTerrainMesher::buildCanopyAttachment(
        {0, 0, FarTerrainStep::TWO}, source, source, FarTerrainAuthorityQuality::FINAL,
        FarTerrainAuthorityQuality::FINAL, &diagnostics);
    REQUIRE(diagnostics.canopyCandidateCount == 2);
    REQUIRE(diagnostics.floraCandidateCount == 2);
    REQUIRE(diagnostics.acceptedCanopyCount == 1);
    REQUIRE(diagnostics.acceptedFloraCount == 1);
    REQUIRE(diagnostics.occupiedGroundCellCount == 2);
    REQUIRE(diagnostics.sparseGroundCellCount == 2);
    REQUIRE(diagnostics.denseGroundGridSampleCount == 0);
    REQUIRE(sparseCells == 2);
    REQUIRE(attachment->canopyAnchorCount == 1);
    REQUIRE(attachment->floraAnchorCount == 1);
    REQUIRE(attachment->buildDiagnostics == diagnostics);
    REQUIRE(diagnostics.totalMicroseconds >=
            diagnostics.canopyCollectionMicroseconds + diagnostics.floraCollectionMicroseconds +
                diagnostics.groundingMicroseconds + diagnostics.geometryMicroseconds);
    REQUIRE(attachment->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::TWO}, source)
                ->deterministicHash);
}

TEST_CASE("Sparse production flora grounding matches the displayed dense cell tops",
          "[render][far-terrain][canopy][flora][grounding][batch][determinism][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.terrainCellTopGrid);
    REQUIRE(source.terrainCellTopPoints);
    constexpr int CELL_EDGE = 8;
    constexpr std::array<uint32_t, 5> OCCUPIED = {0, 5, 9, 36, 63};
    for (const FarTerrainStep lod :
         {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR}) {
        const int step = farTerrainStepSize(lod);
        const worldgen::SurfaceFootprint footprint = farTerrainSurfaceFootprint(lod);
        std::array<float, CELL_EDGE * CELL_EDGE> dense{};
        source.terrainCellTopGrid(-27'136, -16'896, step, CELL_EDGE, footprint, dense);
        std::array<float, OCCUPIED.size()> sparse{};
        source.terrainCellTopPoints(-27'136, -16'896, step, CELL_EDGE, footprint, OCCUPIED, sparse);
        for (size_t index = 0; index < OCCUPIED.size(); ++index) {
            CAPTURE(step, index, OCCUPIED[index], dense[OCCUPIED[index]], sparse[index]);
            REQUIRE(sparse[index] == dense[OCCUPIED[index]]);
        }
    }
}

TEST_CASE("Step-thirty-two coverage batches water and aggregate forest authority",
          "[render][far-terrain][coverage][water][canopy][batch][performance][determinism]") {
    auto generator = std::make_shared<ChunkGenerator>(42);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.geometryPoints);
    // The reported seed-42 camera tile contains a mixed coastal forest. Use
    // it to prove the coverage preview keeps both land canopies and canonical
    // water without invoking scalar basin sampling.
    constexpr FarTerrainKey KEY{-1, 0, FarTerrainStep::THIRTY_TWO};
    size_t expectedCanopyCount = 0;
    const auto collectCanopies = source.canopies;
    source.canopies = [&](int64_t minimumX, int64_t minimumZ, int64_t maximumX, int64_t maximumZ,
                          FarTerrainStep step) {
        std::vector<FarCanopy> canopies =
            collectCanopies(minimumX, minimumZ, maximumX, maximumZ, step);
        expectedCanopyCount = std::ranges::count_if(canopies, [&](const FarCanopy& canopy) {
            return canopy.x >= minimumX && canopy.x < maximumX && canopy.z >= minimumZ &&
                   canopy.z < maximumZ;
        });
        return canopies;
    };
    const uint64_t scalarCallsBefore = generator->basinCacheMetrics().scalarSampleCalls;
    const auto first = FarTerrainMesher::build(KEY, source);
    const auto firstCanopy = FarTerrainMesher::buildCanopyAttachment(KEY, source);
    REQUIRE(firstCanopy->canopyAnchorCount > 0);
    REQUIRE(firstCanopy->canopyAnchorCount == expectedCanopyCount);
    REQUIRE(firstCanopy->floraAnchorCount > 0);
    REQUIRE(firstCanopy->floraImpostorQuadCount == firstCanopy->floraAnchorCount * 2);
    REQUIRE(first->waterQuadCount + first->waterContourTriangleCount > 0);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);

    generator->clearMacroCaches();
    const auto rebuilt = FarTerrainMesher::build(KEY, source);
    const auto rebuiltCanopy = FarTerrainMesher::buildCanopyAttachment(KEY, source);
    REQUIRE(rebuilt->deterministicHash == first->deterministicHash);
    REQUIRE(rebuiltCanopy->deterministicHash == firstCanopy->deterministicHash);
    REQUIRE(rebuilt->vertices.size() == first->vertices.size());
    REQUIRE(std::equal(first->vertices.begin(), first->vertices.end(), rebuilt->vertices.begin(),
                       [](const Vertex& left, const Vertex& right) {
                           return left.faceAttr == right.faceAttr && left.px == right.px &&
                                  left.py == right.py && left.pz == right.pz && left.u == right.u &&
                                  left.v == right.v;
                       }));
    REQUIRE(rebuilt->indices == first->indices);
    REQUIRE(generator->basinCacheMetrics().scalarSampleCalls == scalarCallsBefore);
}

TEST_CASE("Step-eight and step-sixteen canonical water probes stay in bulk batches",
          "[render][far-terrain][water][batch][performance][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            const int64_t shoreline =
                128 + static_cast<int64_t>(std::lround(std::sin(z * 0.075) * 13.0));
            sample.lake = x < shoreline;
            sample.waterBodyId = sample.lake ? 0x4255'4C4B'5741'5445ULL : worldgen::NO_WATER_BODY;
            sample.terrainHeight = sample.lake ? 60.0 : 70.0;
            sample.waterSurface = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    const auto sample = source.sample;
    std::vector<size_t> pointBatchSizes;
    source.geometryPoints = [&](std::span<const ColumnPos> positions,
                                worldgen::SurfaceFootprint footprint,
                                std::span<FarTerrainGeometrySample> output) {
        REQUIRE(output.size() == positions.size());
        pointBatchSizes.push_back(positions.size());
        for (size_t index = 0; index < positions.size(); ++index) {
            output[index] = sample(positions[index].x, positions[index].z, footprint).geometry;
        }
    };

    for (const FarTerrainStep step : {FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN}) {
        const auto mesh = FarTerrainMesher::build({0, 0, step}, source);
        REQUIRE(mesh->waterQuadCount + mesh->waterContourTriangleCount > 0);
    }
    REQUIRE_FALSE(pointBatchSizes.empty());
    REQUIRE(std::ranges::none_of(pointBatchSizes, [](size_t size) { return size <= 1; }));
}

TEST_CASE("Far canopy layers stay inside exact bounds without LOD inflation",
          "[render][far-terrain][canopy][lod][bounds][regression]") {
    FarTerrainSource source;
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint footprint) {
        FarTerrainGeometrySample geometry;
        geometry.terrainHeight = 60.0 + worldgen::surfaceFootprintWidth(footprint);
        return FarSurfaceSample{
            .geometry = geometry,
            .footprintMinimumTerrainHeight = geometry.terrainHeight,
            .footprintMaximumTerrainHeight = geometry.terrainHeight,
            .materialPalette = testMaterialPalette(BlockType::GRASS),
        };
    };
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        return std::vector{FarCanopy{
            .x = 64,
            .z = 64,
            .baseY = 64,
            .topY = 72,
            .canopyMinimumY = 67,
            .canopyMaximumY = 72,
            .canopyRadius = 3,
            .logBlock = BlockType::LOG,
            .leafBlock = BlockType::LEAVES,
            .anchorId = 1,
            .aggregate = farTerrainStepSize(step) >= 8,
        }};
    };

    const auto leafVertices = [](const FarCanopyAttachment& attachment) {
        std::vector<Vertex> result;
        std::ranges::copy_if(
            attachment.vertices, std::back_inserter(result), [](const Vertex& vertex) {
                return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U &&
                       unpackTextureLayer(vertex.faceAttr) ==
                           static_cast<uint8_t>(BlockType::LEAVES);
            });
        return result;
    };
    const auto requireSameVertices = [](const std::vector<Vertex>& first,
                                        const std::vector<Vertex>& second) {
        REQUIRE(first.size() == second.size());
        const auto minimumY = [](const std::vector<Vertex>& vertices) {
            return std::ranges::min(
                vertices, {}, [](const Vertex& vertex) { return static_cast<float>(vertex.py); });
        };
        const float firstMinimumY = static_cast<float>(minimumY(first).py);
        const float secondMinimumY = static_cast<float>(minimumY(second).py);
        for (size_t index = 0; index < first.size(); ++index) {
            CAPTURE(index);
            REQUIRE(first[index].faceAttr == second[index].faceAttr);
            REQUIRE(static_cast<float>(first[index].px) == static_cast<float>(second[index].px));
            REQUIRE(static_cast<float>(first[index].py) - firstMinimumY ==
                    static_cast<float>(second[index].py) - secondMinimumY);
            REQUIRE(static_cast<float>(first[index].pz) == static_cast<float>(second[index].pz));
            REQUIRE(static_cast<float>(first[index].u) == static_cast<float>(second[index].u));
            REQUIRE(static_cast<float>(first[index].v) == static_cast<float>(second[index].v));
        }
    };

    std::vector<std::vector<Vertex>> verticesByLod;
    for (FarTerrainStep step : {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
                                FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
        const auto attachment = FarTerrainMesher::buildCanopyAttachment({0, 0, step}, source);
        const std::vector<Vertex> vertices = leafVertices(*attachment);
        REQUIRE(vertices.size() == 15 * 4);
        float minimumX = std::numeric_limits<float>::max();
        float maximumX = std::numeric_limits<float>::lowest();
        float minimumY = std::numeric_limits<float>::max();
        float maximumY = std::numeric_limits<float>::lowest();
        float minimumZ = std::numeric_limits<float>::max();
        float maximumZ = std::numeric_limits<float>::lowest();
        for (const Vertex& vertex : vertices) {
            minimumX = std::min(minimumX, static_cast<float>(vertex.px));
            maximumX = std::max(maximumX, static_cast<float>(vertex.px));
            minimumY = std::min(minimumY, static_cast<float>(vertex.py));
            maximumY = std::max(maximumY, static_cast<float>(vertex.py));
            minimumZ = std::min(minimumZ, static_cast<float>(vertex.pz));
            maximumZ = std::max(maximumZ, static_cast<float>(vertex.pz));
        }
        REQUIRE(maximumX - minimumX == 7.0F);
        REQUIRE(maximumY - minimumY == 6.0F);
        REQUIRE(maximumZ - minimumZ == 7.0F);

        for (size_t offset = 0; offset < vertices.size(); offset += 4) {
            float quadMinimumY = std::numeric_limits<float>::max();
            float quadMaximumY = std::numeric_limits<float>::lowest();
            float quadMinimumHorizontal = std::numeric_limits<float>::max();
            float quadMaximumHorizontal = std::numeric_limits<float>::lowest();
            const FaceNormal face = unpackFace(vertices[offset].faceAttr);
            for (size_t corner = 0; corner < 4; ++corner) {
                const Vertex& vertex = vertices[offset + corner];
                quadMinimumY = std::min(quadMinimumY, static_cast<float>(vertex.py));
                quadMaximumY = std::max(quadMaximumY, static_cast<float>(vertex.py));
                const float horizontal = face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X
                                             ? static_cast<float>(vertex.pz)
                                             : static_cast<float>(vertex.px);
                quadMinimumHorizontal = std::min(quadMinimumHorizontal, horizontal);
                quadMaximumHorizontal = std::max(quadMaximumHorizontal, horizontal);
            }
            const bool giantSide = quadMaximumY - quadMinimumY == 6.0F &&
                                   quadMaximumHorizontal - quadMinimumHorizontal == 7.0F;
            REQUIRE_FALSE(giantSide);
        }
        verticesByLod.push_back(vertices);
    }
    requireSameVertices(verticesByLod[0], verticesByLod[1]);
    requireSameVertices(verticesByLod[2], verticesByLod[3]);
    requireSameVertices(verticesByLod[3], verticesByLod[4]);
}

TEST_CASE("Coarse far forests retain hierarchical compact anchors",
          "[render][far-terrain][canopy][lod][worldgen][determinism][regression]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    ChunkGenerator generator(42);

    const std::vector<FarCanopy> nearAnchors =
        generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2);
    REQUIRE_FALSE(nearAnchors.empty());
    REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 2) ==
            nearAnchors);
    for (const FarCanopy& canopy : nearAnchors) {
        REQUIRE_FALSE(canopy.aggregate);
        REQUIRE(canopy.anchorId != 0);
        REQUIRE((canopy.logBlock != BlockType::AIR || canopy.leafBlock != BlockType::AIR));
    }

    // Step two preserves exact accepted roots for a stable near handoff.
    // Aggregate cover begins at step four, then each coarser tier retains a
    // strict subset of the same fixed forest-cell candidates.
    constexpr std::array LOD_STEPS = {4, 8, 16, 32};
    constexpr std::array CROWN_LIMITS = {5U, 4U, 3U, 2U};
    using Cell = std::pair<int64_t, int64_t>;
    std::array<std::vector<FarCanopy>, LOD_STEPS.size()> tiers;
    std::array<std::map<Cell, size_t>, LOD_STEPS.size()> counts;

    for (size_t tierIndex = 0; tierIndex < LOD_STEPS.size(); ++tierIndex) {
        const int step = LOD_STEPS[tierIndex];
        tiers[tierIndex] =
            generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, step);
        REQUIRE_FALSE(tiers[tierIndex].empty());
        REQUIRE(generator.collectFarCanopiesForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z,
                                                   step) == tiers[tierIndex]);
        for (const FarCanopy& canopy : tiers[tierIndex]) {
            REQUIRE(canopy.aggregate);
            REQUIRE(canopy.canopyRadius <= 3);
            REQUIRE(static_cast<int>(canopy.canopyRadius) * 2 + 1 <= 7);
            ++counts[tierIndex][{world_coord::floorDiv(canopy.x, int64_t{64}),
                                 world_coord::floorDiv(canopy.z, int64_t{64})}];
        }
        if (tierIndex == 0)
            continue;
        REQUIRE(tiers[tierIndex].size() <= tiers[tierIndex - 1].size());
        std::unordered_map<uint64_t, FarCanopy> nearer;
        for (const FarCanopy& canopy : tiers[tierIndex - 1]) {
            REQUIRE(nearer.emplace(canopy.anchorId, canopy).second);
        }
        for (const FarCanopy& canopy : tiers[tierIndex]) {
            const auto matching = nearer.find(canopy.anchorId);
            REQUIRE(matching != nearer.end());
            REQUIRE(matching->second == canopy);
        }
    }

    bool sawSeveralCrowns = false;
    for (const auto& [cell, finestCount] : counts.front()) {
        REQUIRE(finestCount <= CROWN_LIMITS.front());
        for (size_t tierIndex = 1; tierIndex < counts.size(); ++tierIndex) {
            REQUIRE(counts[tierIndex][cell] ==
                    std::min<size_t>(finestCount, CROWN_LIMITS[tierIndex]));
        }
        sawSeveralCrowns = sawSeveralCrowns || counts.back()[cell] >= 2;
    }
    REQUIRE(sawSeveralCrowns);
    REQUIRE(generator.cachedColumnPlanCount() == 0);
}

TEST_CASE("Far ground flora stays optional, deterministic, and half-open",
          "[render][far-terrain][canopy][flora][ownership][determinism][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 64.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    source.canopies = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector<FarCanopy>{};
    };
    source.flora = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector{FarFlora{.x = 0,
                                    .z = 8,
                                    .baseY = 65,
                                    .block = BlockType::TALL_GRASS,
                                    .height = 1,
                                    .anchorId = 1},
                           FarFlora{.x = 64,
                                    .z = 48,
                                    .baseY = 65,
                                    .block = BlockType::FLOWER_RED,
                                    .height = 1,
                                    .anchorId = 2},
                           FarFlora{.x = 255,
                                    .z = 128,
                                    .baseY = 65,
                                    .block = BlockType::REED,
                                    .height = 3,
                                    .anchorId = 3},
                           FarFlora{.x = 256,
                                    .z = 128,
                                    .baseY = 65,
                                    .block = BlockType::CATTAIL,
                                    .height = 2,
                                    .anchorId = 4}};
    };

    const auto west = FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::FOUR}, source);
    const auto east = FarTerrainMesher::buildCanopyAttachment({1, 0, FarTerrainStep::FOUR}, source);
    REQUIRE(west->canopyAnchorCount == 0);
    REQUIRE(west->floraAnchorCount == 3);
    REQUIRE(west->floraImpostorQuadCount == 18);
    REQUIRE(west->canopyImpostorQuadCount == 0);
    REQUIRE(east->floraAnchorCount == 1);
    REQUIRE(east->floraImpostorQuadCount == 6);
    REQUIRE(west->indices.size() == west->floraImpostorQuadCount * 12);
    REQUIRE(std::ranges::all_of(west->vertices, [](const Vertex& vertex) {
        return (vertex.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U &&
               unpackFace(vertex.faceAttr) == FaceNormal::CROSS;
    }));
    std::set<uint8_t> layers;
    for (const Vertex& vertex : west->vertices)
        layers.insert(unpackTextureLayer(vertex.faceAttr));
    REQUIRE(layers == std::set<uint8_t>{static_cast<uint8_t>(BlockType::TALL_GRASS),
                                        static_cast<uint8_t>(BlockType::FLOWER_RED),
                                        static_cast<uint8_t>(BlockType::REED)});
    REQUIRE(west->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::FOUR}, source)
                ->deterministicHash);

    const auto repeatedTextureWidth = [](const FarCanopyAttachment& attachment) {
        float width = 0.0F;
        for (size_t offset = 0; offset < attachment.vertices.size(); offset += 4) {
            float minimum = std::numeric_limits<float>::max();
            float maximum = std::numeric_limits<float>::lowest();
            for (size_t corner = 0; corner < 4; ++corner) {
                minimum =
                    std::min(minimum, static_cast<float>(attachment.vertices[offset + corner].u));
                maximum =
                    std::max(maximum, static_cast<float>(attachment.vertices[offset + corner].u));
            }
            width += maximum - minimum;
        }
        return width;
    };
    const auto stepEight =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::EIGHT}, source);
    const auto stepSixteen =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto stepThirtyTwo =
        FarTerrainMesher::buildCanopyAttachment({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(stepEight->floraImpostorQuadCount == west->floraAnchorCount * 4);
    REQUIRE(stepSixteen->floraImpostorQuadCount == west->floraAnchorCount * 2);
    REQUIRE(stepThirtyTwo->floraImpostorQuadCount == west->floraAnchorCount * 2);
    REQUIRE(stepEight->anchorIdentityHash == west->anchorIdentityHash);
    REQUIRE(stepSixteen->anchorIdentityHash == west->anchorIdentityHash);
    REQUIRE(stepThirtyTwo->anchorIdentityHash == west->anchorIdentityHash);
    REQUIRE(repeatedTextureWidth(*stepSixteen) >= repeatedTextureWidth(*west));
    REQUIRE(repeatedTextureWidth(*stepThirtyTwo) >= repeatedTextureWidth(*stepSixteen));

    FarTerrainSource withoutFlora = source;
    withoutFlora.flora = {};
    REQUIRE(FarTerrainMesher::build({0, 0, FarTerrainStep::FOUR}, source)->deterministicHash ==
            FarTerrainMesher::build({0, 0, FarTerrainStep::FOUR}, withoutFlora)->deterministicHash);
}

TEST_CASE("Production far flora retains deterministic LOD subsets without column plans",
          "[render][far-terrain][canopy][flora][lod][worldgen][performance][regression]") {
    constexpr int64_t MINIMUM_X = -27'136;
    constexpr int64_t MINIMUM_Z = -16'896;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 256;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 256;
    constexpr std::array LOD_STEPS = {2, 4, 8, 16, 32};
    ChunkGenerator generator(42);
    std::array<std::vector<FarFlora>, LOD_STEPS.size()> tiers;
    for (size_t tierIndex = 0; tierIndex < LOD_STEPS.size(); ++tierIndex) {
        const int step = LOD_STEPS[tierIndex];
        tiers[tierIndex] =
            generator.collectFarFloraForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, step);
        REQUIRE_FALSE(tiers[tierIndex].empty());
        REQUIRE(generator.collectFarFloraForLod(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, step) ==
                tiers[tierIndex]);
        REQUIRE(std::ranges::all_of(tiers[tierIndex], [&](const FarFlora& plant) {
            return plant.x >= MINIMUM_X && plant.x < MAXIMUM_X && plant.z >= MINIMUM_Z &&
                   plant.z < MAXIMUM_Z && rendersAsCross(plant.block) && plant.anchorId != 0;
        }));
        if (tierIndex == 0)
            continue;
        REQUIRE(tiers[tierIndex].size() <= tiers[tierIndex - 1].size());
        std::unordered_map<uint64_t, FarFlora> nearer;
        for (const FarFlora& plant : tiers[tierIndex - 1])
            REQUIRE(nearer.emplace(plant.anchorId, plant).second);
        for (const FarFlora& plant : tiers[tierIndex]) {
            const auto found = nearer.find(plant.anchorId);
            REQUIRE(found != nearer.end());
            REQUIRE(found->second == plant);
        }
    }
    REQUIRE(tiers.front().size() >= tiers.back().size() * 2);

    const int64_t splitX = MINIMUM_X + 128;
    const std::vector<FarFlora> west =
        generator.collectFarFloraForLod(MINIMUM_X, MINIMUM_Z, splitX, MAXIMUM_Z, 4);
    const std::vector<FarFlora> east =
        generator.collectFarFloraForLod(splitX, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z, 4);
    std::unordered_map<uint64_t, FarFlora> partitioned;
    for (const FarFlora& plant : west)
        REQUIRE(partitioned.emplace(plant.anchorId, plant).second);
    for (const FarFlora& plant : east)
        REQUIRE(partitioned.emplace(plant.anchorId, plant).second);
    REQUIRE(partitioned.size() == tiers[1].size());
    for (const FarFlora& plant : tiers[1]) {
        const auto found = partitioned.find(plant.anchorId);
        REQUIRE(found != partitioned.end());
        REQUIRE(found->second == plant);
    }
    REQUIRE(generator.cachedColumnPlanCount() == 0);
}

TEST_CASE("Valid empty far habitat publishes an empty flora attachment",
          "[render][far-terrain][canopy][flora][empty][regression]") {
    FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 63.0;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::STONE; });
    source.flora = [](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        return std::vector<FarFlora>{};
    };
    const auto attachment =
        FarTerrainMesher::buildCanopyAttachment({-4, 7, FarTerrainStep::THIRTY_TWO}, source);
    REQUIRE(attachment->floraAnchorCount == 0);
    REQUIRE(attachment->canopyAnchorCount == 0);
    REQUIRE(attachment->vertices.empty());
    REQUIRE(attachment->indices.empty());
    REQUIRE(attachment->deterministicHash ==
            FarTerrainMesher::buildCanopyAttachment({-4, 7, FarTerrainStep::THIRTY_TWO}, source)
                ->deterministicHash);
}

TEST_CASE("Far terrain greedily merges flat terrain and water", "[render][far-terrain]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = true;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
    const auto mesh = FarTerrainMesher::build(FarTerrainKey{-2, 3, FarTerrainStep::FOUR}, source);
    REQUIRE(sizeof(Vertex) == 16);
    REQUIRE(mesh->originX == -512);
    REQUIRE(mesh->originZ == 768);
    REQUIRE(mesh->terrainQuadCount == 1);
    REQUIRE(mesh->waterQuadCount == 1);
    REQUIRE(mesh->mergedTerrainCellCount == 4096);
    REQUIRE(mesh->transitionTriangleCount > 0);
    REQUIRE(mesh->opaqueIndexCount ==
            mesh->terrainQuadCount * 6 + mesh->transitionTriangleCount * 3);
    REQUIRE(mesh->surfaceBounds.minY == 40.0F);
    REQUIRE(mesh->surfaceBounds.maxY == 64.0F);
    REQUIRE(mesh->bounds.minY == 40.0F);
    REQUIRE(mesh->bounds.maxY == 64.0F);
    REQUIRE(mesh->bounds.minX == -512);
    REQUIRE(mesh->bounds.maxX == -256);
    for (const FarTerrainBounds& patch : mesh->occluderPatches) {
        REQUIRE(patch.maxX - patch.minX == FAR_TERRAIN_OCCLUDER_PATCH_EDGE);
        REQUIRE(patch.maxZ - patch.minZ == FAR_TERRAIN_OCCLUDER_PATCH_EDGE);
        REQUIRE(patch.minY == 40.0F);
        REQUIRE(patch.maxY == 40.0F);
    }

    std::array<int, 6> faceCounts{};
    for (uint32_t indexOffset = 0; indexOffset < mesh->opaqueIndexCount; indexOffset += 6) {
        const Vertex& a = mesh->vertices[mesh->indices[indexOffset]];
        if ((a.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) != 0U)
            continue;
        const Vertex& b = mesh->vertices[mesh->indices[indexOffset + 1]];
        const Vertex& c = mesh->vertices[mesh->indices[indexOffset + 2]];
        const float abX = static_cast<float>(b.px) - static_cast<float>(a.px);
        const float abY = static_cast<float>(b.py) - static_cast<float>(a.py);
        const float abZ = static_cast<float>(b.pz) - static_cast<float>(a.pz);
        const float acX = static_cast<float>(c.px) - static_cast<float>(a.px);
        const float acY = static_cast<float>(c.py) - static_cast<float>(a.py);
        const float acZ = static_cast<float>(c.pz) - static_cast<float>(a.pz);
        const Vec3 normal{abY * acZ - abZ * acY, abZ * acX - abX * acZ, abX * acY - abY * acX};
        const FaceNormal face = unpackFace(a.faceAttr);
        ++faceCounts[static_cast<size_t>(face)];
        switch (face) {
            case FaceNormal::PLUS_X:
                REQUIRE(normal.x > 0.0F);
                break;
            case FaceNormal::MINUS_X:
                REQUIRE(normal.x < 0.0F);
                break;
            case FaceNormal::PLUS_Z:
                REQUIRE(normal.z > 0.0F);
                break;
            case FaceNormal::MINUS_Z:
                REQUIRE(normal.z < 0.0F);
                break;
            case FaceNormal::PLUS_Y:
                REQUIRE(normal.y > 0.0F);
                break;
            default:
                FAIL("unexpected far terrain opaque face");
        }
    }
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_Y)] == 1);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_X)] == 0);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::MINUS_X)] == 0);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::PLUS_Z)] == 0);
    REQUIRE(faceCounts[static_cast<size_t>(FaceNormal::MINUS_Z)] == 0);
}

TEST_CASE("Far terrain water follows deterministic shoreline contours",
          "[render][far-terrain][water][seam]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = x < FAR_TERRAIN_TILE_EDGE / 2;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });

    const FarTerrainKey key{0, 0, FarTerrainStep::SIXTEEN};
    const auto first = FarTerrainMesher::build(key, source);
    const auto second = FarTerrainMesher::build(key, source);
    REQUIRE(first->deterministicHash == second->deterministicHash);
    REQUIRE(first->indices == second->indices);
    REQUIRE(first->waterContourTriangleCount > 0);
    REQUIRE(first->waterQuadCount > 0);
    REQUIRE(first->complexity == 1.0F);

    float easternmostWater = 0.0F;
    bool sawContourVertex = false;
    for (size_t offset = first->opaqueIndexCount; offset < first->indices.size(); offset += 3) {
        std::array<Vec3, 3> triangle{};
        for (size_t corner = 0; corner < triangle.size(); ++corner) {
            const Vertex& vertex = first->vertices[first->indices[offset + corner]];
            REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
            const float x = static_cast<float>(vertex.px);
            easternmostWater = std::max(easternmostWater, x);
            sawContourVertex =
                sawContourVertex || (x > FAR_TERRAIN_TILE_EDGE / 2 - farTerrainStepSize(key.step) &&
                                     x < FAR_TERRAIN_TILE_EDGE / 2);
            triangle[corner] =
                Vec3{x, static_cast<float>(vertex.py), static_cast<float>(vertex.pz)};
        }
        REQUIRE((triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).y > 0.0F);
    }
    REQUIRE(sawContourVertex);
    REQUIRE(easternmostWater < FAR_TERRAIN_TILE_EDGE / 2);
}

TEST_CASE("Coarse lake contours stop at the supported shoreline",
          "[render][far-terrain][water][lake][support]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < 100 ? 74.0 : 84.0;
            sample.waterSurface = x < 100 ? 80.0 : SEA_LEVEL;
            sample.lake = x < 100;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });

    const auto mesh = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}, source);
    float easternmostWater = 0.0F;
    for (size_t offset = mesh->opaqueIndexCount; offset < mesh->indices.size(); ++offset) {
        const Vertex& vertex = mesh->vertices[mesh->indices[offset]];
        easternmostWater = std::max(easternmostWater, static_cast<float>(vertex.px));
        REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
    }
    REQUIRE(easternmostWater >= 99.0F);
    REQUIRE(easternmostWater <= 99.5F);
}

TEST_CASE("Far water never triangulates between distinct standing bodies",
          "[render][far-terrain][water][authority][seam][lod][regression]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.lake = true;
            if (x + z < 400) {
                sample.waterBodyId = 0xA11CE;
                sample.waterSurface = 307.875;
            } else {
                sample.waterBodyId = 0xB0B;
                sample.waterSurface = 106.875;
            }
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });

    const auto coarse = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto fine = FarTerrainMesher::build({1, 0, FarTerrainStep::FOUR}, source);
    const auto requireNoWaterBridge = [](const FarTerrainMesh& mesh) {
        bool sawUpperBody = false;
        bool sawLowerBody = false;
        for (size_t offset = mesh.opaqueIndexCount; offset + 2 < mesh.indices.size(); offset += 3) {
            std::array<float, 3> heights{};
            for (size_t corner = 0; corner < heights.size(); ++corner) {
                heights[corner] =
                    static_cast<float>(mesh.vertices[mesh.indices[offset + corner]].py);
                sawUpperBody = sawUpperBody || heights[corner] > 300.0F;
                sawLowerBody = sawLowerBody || heights[corner] < 110.0F;
            }
            const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
            CAPTURE(mesh.key.tileX, mesh.key.tileZ, static_cast<int>(mesh.key.step), *minimum,
                    *maximum);
            REQUIRE(*maximum - *minimum <= 0.25F);
        }
        REQUIRE(sawUpperBody);
        REQUIRE(sawLowerBody);
    };
    requireNoWaterBridge(*coarse);
    requireNoWaterBridge(*fine);

    const auto seamVertices = [](const FarTerrainMesh& mesh, float localX) {
        std::set<std::pair<float, float>> result;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); ++offset) {
            const Vertex& vertex = mesh.vertices[mesh.indices[offset]];
            if (static_cast<float>(vertex.px) == localX) {
                result.emplace(static_cast<float>(vertex.pz), static_cast<float>(vertex.py));
            }
        }
        return result;
    };
    const auto coarseSeam = seamVertices(*coarse, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto fineSeam = seamVertices(*fine, 0.0F);
    REQUIRE_FALSE(coarseSeam.empty());
    REQUIRE_FALSE(fineSeam.empty());
    for (int z = 0; z <= FAR_TERRAIN_TILE_EDGE; z += 2) {
        const float expectedHeight =
            static_cast<float>(static_cast<float16_t>(256 + z < 400 ? 307.875F : 106.875F));
        CAPTURE(z, expectedHeight);
        REQUIRE(coarseSeam.contains({static_cast<float>(z), expectedHeight}));
        REQUIRE(fineSeam.contains({static_cast<float>(z), expectedHeight}));
    }
}

TEST_CASE("Seed 764891 caldera water has no coarse interpolation wall",
          "[render][far-terrain][water][authority][caldera][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const FarTerrainGeometrySample caldera =
        testFarGeometry(source, 23'029, -111'486, worldgen::SurfaceFootprint::BLOCK_16);
    REQUIRE(caldera.lake);
    REQUIRE(caldera.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(caldera.waterSurface > 250.0);

    const auto mesh = FarTerrainMesher::build({89, -436, FarTerrainStep::SIXTEEN}, source);
    bool sawCalderaSurface = false;
    for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size(); offset += 3) {
        std::array<float, 3> heights{};
        bool topSurface = true;
        for (size_t corner = 0; corner < heights.size(); ++corner) {
            const Vertex& vertex = mesh->vertices[mesh->indices[offset + corner]];
            topSurface = topSurface && unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
                         !unpackFluidFalling(vertex.faceAttr);
            heights[corner] = static_cast<float>(vertex.py);
        }
        if (!topSurface)
            continue;
        const auto [minimum, maximum] = std::minmax_element(heights.begin(), heights.end());
        sawCalderaSurface = sawCalderaSurface || *maximum > 250.0F;
        CAPTURE(*minimum, *maximum);
        REQUIRE(*maximum - *minimum <= 8.0F);
    }
    REQUIRE(sawCalderaSurface);
}

TEST_CASE("Step 32 keeps caldera water and volcanic island land on canonical cells",
          "[render][far-terrain][water][coverage][caldera][volcanic][ownership][regression]") {
    auto generator = std::make_shared<ChunkGenerator>(764891);
    FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    source.canopies = {};
    struct Fixture {
        ColumnPos position;
        bool wet;
        const char* name;
    };
    constexpr std::array FIXTURES = {
        Fixture{{23'024, -111'488}, true, "caldera lake"},
        Fixture{{17'576, -9'632}, false, "volcanic island"},
    };
    for (const Fixture& fixture : FIXTURES) {
        const int64_t tileX =
            world_coord::floorDiv(fixture.position.x, int64_t{FAR_TERRAIN_TILE_EDGE});
        const int64_t tileZ =
            world_coord::floorDiv(fixture.position.z, int64_t{FAR_TERRAIN_TILE_EDGE});
        const auto mesh =
            FarTerrainMesher::build({tileX, tileZ, FarTerrainStep::THIRTY_TWO}, source);
        const worldgen::SurfaceSample exact =
            generator->sampleExactSurface(fixture.position.x, fixture.position.z);
        const bool exactWet = (exact.hydrology.ocean || exact.hydrology.river ||
                               exact.hydrology.lake || exact.hydrology.wetland) &&
                              exact.waterSurface > exact.terrainHeight + 0.01;
        const float localX = static_cast<float>(fixture.position.x - mesh->originX) + 0.5F;
        const float localZ = static_cast<float>(fixture.position.z - mesh->originZ) + 0.5F;
        CAPTURE(fixture.name, tileX, tileZ, exactWet, exact.hydrology.waterBodyId, localX, localZ);
        REQUIRE(exactWet == fixture.wet);
        REQUIRE(farWaterTopCovers(*mesh, localX, localZ) == exactWet);
        REQUIRE(mesh->waterContourTriangleCount == 0);
        if (fixture.wet)
            REQUIRE(exact.hydrology.waterBodyId != worldgen::NO_WATER_BODY);
    }
}

TEST_CASE("Far terrain shoreline contours stitch across tile faces",
          "[render][far-terrain][water][seam]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.ocean = z < x - FAR_TERRAIN_TILE_EDGE / 2;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
    const auto left = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}, source);
    const auto right =
        FarTerrainMesher::build(FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}, source);

    const auto edgeCoverage = [](const FarTerrainMesh& mesh, float edgeX) {
        std::vector<std::pair<float, float>> intervals;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<float, 3> edgeZ{};
            size_t edgeVertexCount = 0;
            for (size_t corner = 0; corner < 3; ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                if (static_cast<float>(vertex.px) == edgeX) {
                    edgeZ[edgeVertexCount++] = static_cast<float>(vertex.pz);
                }
            }
            if (edgeVertexCount >= 2) {
                const auto [minimum, maximum] =
                    std::minmax_element(edgeZ.begin(), edgeZ.begin() + edgeVertexCount);
                if (*minimum < *maximum)
                    intervals.emplace_back(*minimum, *maximum);
            }
        }
        std::sort(intervals.begin(), intervals.end());
        std::vector<std::pair<float, float>> merged;
        for (const auto interval : intervals) {
            if (merged.empty() || interval.first > merged.back().second) {
                merged.push_back(interval);
            } else {
                merged.back().second = std::max(merged.back().second, interval.second);
            }
        }
        return merged;
    };
    const auto leftEdge = edgeCoverage(*left, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto rightEdge = edgeCoverage(*right, 0.0F);
    REQUIRE_FALSE(leftEdge.empty());
    REQUIRE(leftEdge == rightEdge);
}

TEST_CASE("Narrow rivers retain identical coverage across mixed LOD tile faces",
          "[render][far-terrain][water][seam][lod]") {
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 40.0;
            sample.waterSurface = 64.0;
            sample.river = z >= 5 && z <= 7;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });
    const auto fine = FarTerrainMesher::build({0, 0, FarTerrainStep::FOUR}, source);
    const auto coarse = FarTerrainMesher::build({1, 0, FarTerrainStep::SIXTEEN}, source);

    const auto edgeCoverage = [](const FarTerrainMesh& mesh, float edgeX) {
        std::vector<std::pair<float, float>> intervals;
        for (size_t offset = mesh.opaqueIndexCount; offset < mesh.indices.size(); offset += 3) {
            std::array<float, 3> edgeZ{};
            size_t count = 0;
            for (size_t corner = 0; corner < 3; ++corner) {
                const Vertex& vertex = mesh.vertices[mesh.indices[offset + corner]];
                if (static_cast<float>(vertex.px) == edgeX) {
                    edgeZ[count++] = static_cast<float>(vertex.pz);
                }
            }
            if (count < 2)
                continue;
            const auto [minimum, maximum] =
                std::minmax_element(edgeZ.begin(), edgeZ.begin() + count);
            if (*minimum < *maximum)
                intervals.emplace_back(*minimum, *maximum);
        }
        std::sort(intervals.begin(), intervals.end());
        return intervals;
    };

    const auto fineEdge = edgeCoverage(*fine, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto coarseEdge = edgeCoverage(*coarse, 0.0F);
    REQUIRE_FALSE(fineEdge.empty());
    REQUIRE(fineEdge == coarseEdge);
    REQUIRE(fine->complexity == 1.0F);
    REQUIRE(coarse->complexity == 1.0F);
}

TEST_CASE("Large standing water keeps canonical coverage across step 32 and step 16",
          "[render][far-terrain][water][seam][lod][step-32][regression]") {
    constexpr worldgen::WaterBodyId LAKE_ID = 0x4C4F'4457'4154'4552ULL;
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t, int64_t z) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = 44.0;
            const int64_t localZ = world_coord::floorMod(z, FAR_TERRAIN_TILE_EDGE);
            if (localZ >= 37 && localZ < 221) {
                sample.waterSurface = 71.0;
                sample.waterBodyId = LAKE_ID;
                sample.lake = true;
            }
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::CLAY; });

    const auto coarse = FarTerrainMesher::build({0, 0, FarTerrainStep::THIRTY_TWO}, source);
    const auto fine = FarTerrainMesher::build({1, 0, FarTerrainStep::SIXTEEN}, source);
    size_t wetSamples = 0;
    for (int z = 0; z < FAR_TERRAIN_TILE_EDGE; ++z) {
        const float sampleZ = static_cast<float>(z) + 0.5F;
        const bool coarseWet =
            farWaterTopCovers(*coarse, static_cast<float>(FAR_TERRAIN_TILE_EDGE), sampleZ);
        const bool fineWet = farWaterTopCovers(*fine, 0.0F, sampleZ);
        CAPTURE(z, coarseWet, fineWet);
        REQUIRE(coarseWet == fineWet);
        REQUIRE(coarseWet == (z >= 36 && z <= 220));
        wetSamples += coarseWet;
    }
    REQUIRE(wetSamples == 185);
    REQUIRE(coarse->complexity == 1.0F);
    REQUIRE(fine->complexity == 1.0F);
}

TEST_CASE("Orthogonal water boundary refinement owns each corner once",
          "[render][far-terrain][water][seam][ownership][regression]") {
    constexpr uint8_t WEST_EDGE = 1U << 0U;
    constexpr uint8_t EAST_EDGE = 1U << 1U;
    constexpr uint8_t NORTH_EDGE = 1U << 2U;
    constexpr uint8_t SOUTH_EDGE = 1U << 3U;
    for (uint8_t edgeMask = 0; edgeMask < 16; ++edgeMask) {
        CAPTURE(static_cast<unsigned>(edgeMask));
        const FarTerrainSource source = testFarTerrainSource(
            [edgeMask](int64_t x, int64_t z) {
                FarTerrainGeometrySample sample;
                sample.terrainHeight = 40.0;
                sample.waterSurface = 64.0;
                const auto active = [edgeMask](uint8_t edge) { return (edgeMask & edge) != 0; };
                bool wet = false;
                wet = wet || (active(WEST_EDGE) && x <= 16 && z >= 48 && z <= 64);
                wet = wet ||
                      (active(EAST_EDGE) && x >= FAR_TERRAIN_TILE_EDGE - 16 && z >= 48 && z <= 64);
                wet = wet || (active(NORTH_EDGE) && z <= 16 && x >= 48 && x <= 64);
                wet = wet ||
                      (active(SOUTH_EDGE) && z >= FAR_TERRAIN_TILE_EDGE - 16 && x >= 48 && x <= 64);
                wet = wet || (active(WEST_EDGE) && active(NORTH_EDGE) && x <= 32 && z <= 32);
                wet = wet || (active(EAST_EDGE) && active(NORTH_EDGE) &&
                              x >= FAR_TERRAIN_TILE_EDGE - 32 && z <= 32);
                wet = wet || (active(WEST_EDGE) && active(SOUTH_EDGE) && x <= 32 &&
                              z >= FAR_TERRAIN_TILE_EDGE - 32);
                wet = wet || (active(EAST_EDGE) && active(SOUTH_EDGE) &&
                              x >= FAR_TERRAIN_TILE_EDGE - 32 && z >= FAR_TERRAIN_TILE_EDGE - 32);
                sample.ocean = wet;
                return sample;
            },
            [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::SAND; });
        const auto mesh = FarTerrainMesher::build({0, 0, FarTerrainStep::SIXTEEN}, source);

        std::array<double, 4> cornerAreas{};
        for (size_t offset = mesh->opaqueIndexCount; offset + 2 < mesh->indices.size();
             offset += 3) {
            std::array<const Vertex*, 3> vertices{};
            for (size_t corner = 0; corner < vertices.size(); ++corner) {
                vertices[corner] = &mesh->vertices[mesh->indices[offset + corner]];
            }
            const double x0 = static_cast<float>(vertices[0]->px);
            const double z0 = static_cast<float>(vertices[0]->pz);
            const double x1 = static_cast<float>(vertices[1]->px);
            const double z1 = static_cast<float>(vertices[1]->pz);
            const double x2 = static_cast<float>(vertices[2]->px);
            const double z2 = static_cast<float>(vertices[2]->pz);
            const double area = std::abs((x0 * (z1 - z2) + x1 * (z2 - z0) + x2 * (z0 - z1)) * 0.5);
            constexpr float TILE_EDGE = static_cast<float>(FAR_TERRAIN_TILE_EDGE);
            constexpr std::array<std::array<float, 4>, 4> CORNERS = {{
                {{0.0F, 16.0F, 0.0F, 16.0F}},
                {{TILE_EDGE - 16.0F, TILE_EDGE, 0.0F, 16.0F}},
                {{0.0F, 16.0F, TILE_EDGE - 16.0F, TILE_EDGE}},
                {{TILE_EDGE - 16.0F, TILE_EDGE, TILE_EDGE - 16.0F, TILE_EDGE}},
            }};
            for (size_t corner = 0; corner < CORNERS.size(); ++corner) {
                const auto [minimumX, maximumX, minimumZ, maximumZ] = CORNERS[corner];
                const bool inside = std::ranges::all_of(vertices, [&](const Vertex* vertex) {
                    const float x = static_cast<float>(vertex->px);
                    const float z = static_cast<float>(vertex->pz);
                    return x >= minimumX && x <= maximumX && z >= minimumZ && z <= maximumZ;
                });
                if (inside)
                    cornerAreas[corner] += area;
            }
        }
        constexpr std::array<std::pair<uint8_t, uint8_t>, 4> INCIDENT_EDGES = {{
            {WEST_EDGE, NORTH_EDGE},
            {EAST_EDGE, NORTH_EDGE},
            {WEST_EDGE, SOUTH_EDGE},
            {EAST_EDGE, SOUTH_EDGE},
        }};
        for (size_t corner = 0; corner < cornerAreas.size(); ++corner) {
            const auto [first, second] = INCIDENT_EDGES[corner];
            const double expected =
                (edgeMask & first) != 0 && (edgeMask & second) != 0 ? 16.0 * 16.0 : 0.0;
            REQUIRE(cornerAreas[corner] == Catch::Approx(expected));
        }
    }
}

TEST_CASE("Step one closes unequal exact columns once at each tile edge",
          "[render][far-terrain][step-one][exact][riser][seam][regression]") {
    constexpr double WEST_HEIGHT = 40.0;
    constexpr double EAST_HEIGHT = 48.0;
    const FarTerrainSource source = testFarTerrainSource(
        [](int64_t x, int64_t) {
            FarTerrainGeometrySample sample;
            sample.terrainHeight = x < FAR_TERRAIN_TILE_EDGE ? WEST_HEIGHT : EAST_HEIGHT;
            return sample;
        },
        [](int64_t, int64_t, const FarTerrainGeometrySample&) { return BlockType::GRASS; });
    const auto west = FarTerrainMesher::build({0, 0, FarTerrainStep::ONE}, source);
    const auto east = FarTerrainMesher::build({1, 0, FarTerrainStep::ONE}, source);

    const auto boundaryRiser = [](const FarTerrainMesh& mesh, float localX) {
        double area = 0.0;
        size_t triangles = 0;
        for (size_t offset = 0; offset + 2 < mesh.opaqueIndexCount; offset += 3) {
            const Vertex& first = mesh.vertices[mesh.indices[offset]];
            const FaceNormal face = unpackFace(first.faceAttr);
            if (face != FaceNormal::PLUS_X && face != FaceNormal::MINUS_X)
                continue;
            const Vertex& second = mesh.vertices[mesh.indices[offset + 1]];
            const Vertex& third = mesh.vertices[mesh.indices[offset + 2]];
            if (static_cast<float>(first.px) != localX || static_cast<float>(second.px) != localX ||
                static_cast<float>(third.px) != localX) {
                continue;
            }
            const Vec3 firstPosition{static_cast<float>(first.px), static_cast<float>(first.py),
                                     static_cast<float>(first.pz)};
            const Vec3 secondPosition{static_cast<float>(second.px), static_cast<float>(second.py),
                                      static_cast<float>(second.pz)};
            const Vec3 thirdPosition{static_cast<float>(third.px), static_cast<float>(third.py),
                                     static_cast<float>(third.pz)};
            const Vec3 normal =
                (secondPosition - firstPosition).cross(thirdPosition - firstPosition);
            REQUIRE(normal.lengthSq() > 0.0F);
            REQUIRE(face == FaceNormal::MINUS_X);
            area += static_cast<double>(normal.length()) * 0.5;
            ++triangles;
        }
        return std::pair{area, triangles};
    };

    const auto [westArea, westTriangles] =
        boundaryRiser(*west, static_cast<float>(FAR_TERRAIN_TILE_EDGE));
    const auto [eastArea, eastTriangles] = boundaryRiser(*east, 0.0F);
    REQUIRE(westTriangles == static_cast<size_t>(FAR_TERRAIN_TILE_EDGE * 2));
    REQUIRE(westArea == Catch::Approx((EAST_HEIGHT - WEST_HEIGHT) * FAR_TERRAIN_TILE_EDGE));
    REQUIRE(eastTriangles == 0);
    REQUIRE(eastArea == 0.0);
}

TEST_CASE("Far terrain meshes are order independent and stitch tile edges",
          "[render][far-terrain][determinism]") {
    const FarTerrainSource source = farTerrainTestSource();
    const FarTerrainKey leftKey{-1, -2, FarTerrainStep::FOUR};
    const FarTerrainKey rightKey{0, -2, FarTerrainStep::FOUR};
    const auto leftFirst = FarTerrainMesher::build(leftKey, source);
    const auto rightSecond = FarTerrainMesher::build(rightKey, source);
    const auto rightFirst = FarTerrainMesher::build(rightKey, source);
    const auto leftSecond = FarTerrainMesher::build(leftKey, source);
    REQUIRE(leftFirst->deterministicHash == leftSecond->deterministicHash);
    REQUIRE(rightFirst->deterministicHash == rightSecond->deterministicHash);
    REQUIRE(leftFirst->vertices.size() == leftSecond->vertices.size());
    REQUIRE(leftFirst->indices == leftSecond->indices);

    const std::map<int, float> leftEdge = farTerrainEdge(*leftFirst, true);
    const std::map<int, float> rightEdge = farTerrainEdge(*rightFirst, false);
    REQUIRE_FALSE(leftEdge.empty());
    REQUIRE_FALSE(rightEdge.empty());
    REQUIRE(farTerrainTopsAreVoxelFlat(*leftFirst));
    REQUIRE(farTerrainTopsAreVoxelFlat(*rightFirst));
    REQUIRE(leftEdge == rightEdge);
    REQUIRE(leftEdge.size() ==
            static_cast<size_t>(FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1));
}

TEST_CASE("Far terrain LOD edges share aligned samples without downward skirts",
          "[render][far-terrain][seam]") {
    const FarTerrainSource source = farTerrainTestSource();
    const auto fine = FarTerrainMesher::build(FarTerrainKey{0, 0, FarTerrainStep::FOUR}, source);
    const auto coarse =
        FarTerrainMesher::build(FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}, source);
    const std::map<int, float> fineEdge = farTerrainEdge(*fine, true);
    const std::map<int, float> coarseEdge = farTerrainEdge(*coarse, false);
    REQUIRE_FALSE(fineEdge.empty());
    REQUIRE_FALSE(coarseEdge.empty());
    for (const auto& [z, height] : fineEdge) {
        CAPTURE(z);
        REQUIRE(height == std::round(height));
    }
    for (const auto& [z, height] : coarseEdge) {
        CAPTURE(z);
        REQUIRE(height == std::round(height));
    }
    REQUIRE(fineEdge == coarseEdge);
    REQUIRE(farTerrainTopsAreVoxelFlat(*fine));
    REQUIRE(farTerrainTopsAreVoxelFlat(*coarse));
    REQUIRE(fine->bounds.minY >= static_cast<float>(WORLD_MIN_Y));
    REQUIRE(fine->bounds.minY >= fine->surfaceBounds.minY);
}

TEST_CASE("Mixed far LODs share canonical topology on all negative tile edges",
          "[render][far-terrain][lod][transition][topology][seam][negative][regression]") {
    const FarTerrainSource source = farTerrainTestSource();
    constexpr FarTerrainKey CENTER{-3, -4, FarTerrainStep::SIXTEEN};
    const auto center = FarTerrainMesher::build(CENTER, source);
    const auto west =
        FarTerrainMesher::build({CENTER.tileX - 1, CENTER.tileZ, FarTerrainStep::EIGHT}, source);
    const auto east =
        FarTerrainMesher::build({CENTER.tileX + 1, CENTER.tileZ, FarTerrainStep::EIGHT}, source);
    const auto north =
        FarTerrainMesher::build({CENTER.tileX, CENTER.tileZ - 1, FarTerrainStep::EIGHT}, source);
    const auto south =
        FarTerrainMesher::build({CENTER.tileX, CENTER.tileZ + 1, FarTerrainStep::EIGHT}, source);

    const auto requireSharedEdge = [](const FarTerrainMesh& first, FaceNormal firstEdge,
                                      const FarTerrainMesh& second, FaceNormal secondEdge) {
        const std::map<int, float> firstBoundary = farTerrainBoundary(first, firstEdge);
        const std::map<int, float> secondBoundary = farTerrainBoundary(second, secondEdge);
        REQUIRE(firstBoundary == secondBoundary);
        REQUIRE(
            firstBoundary.size() ==
            static_cast<size_t>(FAR_TERRAIN_TILE_EDGE / FAR_TERRAIN_TRANSITION_SAMPLE_STEP + 1));
    };
    requireSharedEdge(*center, FaceNormal::MINUS_X, *west, FaceNormal::PLUS_X);
    requireSharedEdge(*center, FaceNormal::PLUS_X, *east, FaceNormal::MINUS_X);
    requireSharedEdge(*center, FaceNormal::MINUS_Z, *north, FaceNormal::PLUS_Z);
    requireSharedEdge(*center, FaceNormal::PLUS_Z, *south, FaceNormal::MINUS_Z);

    // Positive-area terrain belongs to exactly one half-open tile. Neighboring
    // meshes may share their zero-area boundary polyline, but never a top
    // triangle or a centroid on the positive tile edge.
    using WorldVertex = std::array<double, 3>;
    using WorldTriangle = std::array<WorldVertex, 3>;
    std::set<WorldTriangle> ownedTopTriangles;
    const std::array meshes = {center, west, east, north, south};
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : meshes) {
        for (size_t offset = 0; offset + 2 < mesh->opaqueIndexCount; offset += 3) {
            const Vertex& first = mesh->vertices[mesh->indices[offset]];
            if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
                (first.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0U) {
                continue;
            }
            const Vertex& second = mesh->vertices[mesh->indices[offset + 1]];
            const Vertex& third = mesh->vertices[mesh->indices[offset + 2]];
            WorldTriangle triangle = {
                WorldVertex{static_cast<double>(mesh->originX) + static_cast<float>(first.px),
                            static_cast<float>(first.py),
                            static_cast<double>(mesh->originZ) + static_cast<float>(first.pz)},
                WorldVertex{static_cast<double>(mesh->originX) + static_cast<float>(second.px),
                            static_cast<float>(second.py),
                            static_cast<double>(mesh->originZ) + static_cast<float>(second.pz)},
                WorldVertex{static_cast<double>(mesh->originX) + static_cast<float>(third.px),
                            static_cast<float>(third.py),
                            static_cast<double>(mesh->originZ) + static_cast<float>(third.pz)},
            };
            const double centerX = (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0;
            const double centerZ = (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0;
            REQUIRE(centerX >= static_cast<double>(mesh->originX));
            REQUIRE(centerX < static_cast<double>(mesh->originX + FAR_TERRAIN_TILE_EDGE));
            REQUIRE(centerZ >= static_cast<double>(mesh->originZ));
            REQUIRE(centerZ < static_cast<double>(mesh->originZ + FAR_TERRAIN_TILE_EDGE));
            std::ranges::sort(triangle);
            REQUIRE(ownedTopTriangles.insert(triangle).second);
        }
    }
    REQUIRE_FALSE(ownedTopTriangles.empty());

    const int step = farTerrainStepSize(CENTER.step);
    const int cellEdge = FAR_TERRAIN_TILE_EDGE / step;
    const double expectedTransitionArea =
        static_cast<double>((cellEdge * cellEdge - (cellEdge - 2) * (cellEdge - 2)) * step * step);
    double transitionArea = 0.0;
    size_t transitionTriangles = 0;
    for (size_t offset = 0; offset + 2 < center->opaqueIndexCount; offset += 3) {
        const Vertex& first = center->vertices[center->indices[offset]];
        if ((first.faceAttr & FAR_TERRAIN_TRANSITION_ATTRIBUTE_MASK) == 0U)
            continue;
        const Vertex& second = center->vertices[center->indices[offset + 1]];
        const Vertex& third = center->vertices[center->indices[offset + 2]];
        const FaceNormal face = unpackFace(first.faceAttr);
        if (face != FaceNormal::PLUS_Y) {
            const bool fixedOnTileEdge =
                ((face == FaceNormal::PLUS_X || face == FaceNormal::MINUS_X) &&
                 static_cast<float>(first.px) == static_cast<float>(second.px) &&
                 (static_cast<float>(first.px) == 0.0F ||
                  static_cast<float>(first.px) == FAR_TERRAIN_TILE_EDGE)) ||
                ((face == FaceNormal::PLUS_Z || face == FaceNormal::MINUS_Z) &&
                 static_cast<float>(first.pz) == static_cast<float>(second.pz) &&
                 (static_cast<float>(first.pz) == 0.0F ||
                  static_cast<float>(first.pz) == FAR_TERRAIN_TILE_EDGE));
            REQUIRE_FALSE(fixedOnTileEdge);
            continue;
        }
        const double twiceArea = (static_cast<float>(second.pz) - static_cast<float>(first.pz)) *
                                     (static_cast<float>(third.px) - static_cast<float>(first.px)) -
                                 (static_cast<float>(second.px) - static_cast<float>(first.px)) *
                                     (static_cast<float>(third.pz) - static_cast<float>(first.pz));
        REQUIRE(twiceArea > 0.0);
        transitionArea += twiceArea * 0.5;
        ++transitionTriangles;
    }
    REQUIRE(transitionTriangles == center->transitionTriangleCount);
    REQUIRE(transitionArea == Catch::Approx(expectedTransitionArea));
    constexpr uint32_t RESERVED_PANEL_ATTRIBUTE = 1U << 29U;
    REQUIRE(std::ranges::none_of(center->vertices, [](const Vertex& vertex) {
        return (vertex.faceAttr & RESERVED_PANEL_ATTRIBUTE) != 0U;
    }));
}

TEST_CASE("Terrain horizon culling is conservative", "[render][far-terrain][occlusion]") {
    TerrainHorizonCuller culler({0.0, 64.0, 0.0});
    const FarTerrainBounds uniformRidge{100, 200, -100, 100, 200.0F, 220.0F};
    const FarTerrainBounds hiddenLowland{400, 500, -50, 50, 20.0F, 80.0F};
    REQUIRE_FALSE(culler.testAndAdd(uniformRidge));
    REQUIRE(culler.isOccluded(hiddenLowland));

    const FarTerrainBounds tallPeak{400, 500, -50, 50, 20.0F, 320.0F};
    REQUIRE_FALSE(culler.isOccluded(tallPeak));

    culler.reset({0.0, 64.0, 0.0});
    const FarTerrainBounds peakWithLowValleys{100, 200, -100, 100, 0.0F, 300.0F};
    culler.addOccluder(peakWithLowValleys);
    REQUIRE_FALSE(culler.isOccluded(hiddenLowland));

    culler.reset({0.0, 64.0, 0.0});
    const FarTerrainBounds narrowRidge{100, 200, -5, 5, 200.0F, 220.0F};
    culler.addOccluder(narrowRidge);
    REQUIRE_FALSE(culler.isOccluded(hiddenLowland));
    REQUIRE_FALSE(culler.isOccluded(FarTerrainBounds{-10, 10, -10, 10, 0.0F, 500.0F}));

    SECTION("terrain below a high camera uses conservative distance extrema") {
        culler.reset({0.0, 480.0, 0.0});
        const FarTerrainBounds lowNearRidge{100, 500, -100, 100, 0.0F, 100.0F};
        const FarTerrainBounds lowFarCandidate{600, 700, -50, 50, -128.0F, -100.0F};
        culler.addOccluder(lowNearRidge);
        REQUIRE_FALSE(culler.isOccluded(lowFarCandidate));
    }

    SECTION("terrain below a deep-world camera remains visible without full coverage") {
        culler.reset({0.0, 480.0, 0.0});
        const FarTerrainBounds narrowLowRidge{100, 200, -4, 4, -100.0F, 0.0F};
        const FarTerrainBounds wideLowCandidate{400, 500, -100, 100, -120.0F, -20.0F};
        culler.addOccluder(narrowLowRidge);
        REQUIRE_FALSE(culler.isOccluded(wideLowCandidate));
    }

    SECTION("candidate fringe bins must also have a valid occluder") {
        culler.reset({0.0, 64.0, 0.0});
        const FarTerrainBounds binAlignedRidge{108, 208, -8, 8, 200.0F, 220.0F};
        const FarTerrainBounds partialFringeCandidate{400, 500, -32, 32, 20.0F, 80.0F};
        culler.addOccluder(binAlignedRidge);
        REQUIRE_FALSE(culler.isOccluded(partialFringeCandidate));
    }

    SECTION("a farther horizon never hides nearer terrain") {
        culler.reset({0.0, 64.0, 0.0});
        const FarTerrainBounds farRidge{400, 500, -100, 100, 260.0F, 300.0F};
        const FarTerrainBounds nearCandidate{100, 200, -50, 50, 20.0F, 80.0F};
        culler.addOccluder(farRidge);
        REQUIRE_FALSE(culler.isOccluded(nearCandidate));
    }
}

TEST_CASE("Cold coarse horizon uses every base-capable terrain worker off the caller",
          "[render][far-terrain][scheduler][coverage][startup]") {
    const std::thread::id caller = std::this_thread::get_id();
    std::mutex threadMutex;
    std::condition_variable threadCv;
    std::set<std::thread::id> workerThreads;
    bool workersReleased = false;
    bool workerGateTimedOut = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(threadMutex);
            workerThreads.insert(std::this_thread::get_id());
            if (workerThreads.size() == FarTerrainScheduler::WORKER_COUNT) {
                workersReleased = true;
                threadCv.notify_all();
            } else if (!workersReleased && !threadCv.wait_for(lock, std::chrono::seconds(2),
                                                              [&] { return workersReleased; })) {
                workerGateTimedOut = true;
                workersReleased = true;
                threadCv.notify_all();
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    constexpr size_t MAXIMUM_PENDING = FarTerrainScheduler::WORKER_COUNT * 2;
    constexpr int JOB_COUNT =
        static_cast<int>(farTerrainNonurgentBaseAdmissionLimit(MAXIMUM_PENDING));
    limits.maxPending = MAXIMUM_PENDING;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 8 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    for (int index = 0; index < JOB_COUNT; ++index) {
        REQUIRE(scheduler.enqueue({index, 0, FAR_TERRAIN_BASE_STEP}));
    }
    REQUIRE_FALSE(scheduler.enqueue({JOB_COUNT, 0, FAR_TERRAIN_BASE_STEP}));
    REQUIRE_FALSE(scheduler.enqueue({0, 0, FAR_TERRAIN_BASE_STEP}));
    for (int attempt = 0; attempt < 400 && (scheduler.stats().inFlight != 0 ||
                                            scheduler.stats().maintenancePending != 0);
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const FarTerrainSchedulerStats stats = scheduler.stats();
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.built == JOB_COUNT);
    REQUIRE(stats.completed == 2);
    REQUIRE(stats.cacheEntries <= 2);
    REQUIRE(stats.cacheBytes <= limits.maxCacheBytes);
    {
        std::lock_guard lock(threadMutex);
        REQUIRE_FALSE(workerGateTimedOut);
        REQUIRE(workerThreads.size() == FarTerrainScheduler::WORKER_COUNT);
        REQUIRE_FALSE(workerThreads.contains(caller));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 2);
    for (const FarTerrainResult& result : completed) {
        REQUIRE_FALSE(result.failed);
        REQUIRE(result.mesh);
        REQUIRE(result.epoch == scheduler.currentEpoch());
    }
}

TEST_CASE("Far scheduler reserves admission for urgent preview and protected FINAL work",
          "[render][far-terrain][scheduler][capacity][authority][priority][regression]") {
    STATIC_REQUIRE(farTerrainUrgentSchedulerReservation(64) == 16);
    STATIC_REQUIRE(farTerrainNonurgentBaseAdmissionLimit(64) == 48);
    STATIC_REQUIRE(farTerrainUrgentSchedulerReservation(8) == 2);
    STATIC_REQUIRE(farTerrainNonurgentBaseAdmissionLimit(1) == 1);

    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 64;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr size_t DISTANT_LIMIT = farTerrainNonurgentBaseAdmissionLimit(64);
    for (size_t index = 0; index < DISTANT_LIMIT; ++index) {
        CAPTURE(index, scheduler.stats().inFlight, scheduler.stats().parkedBase);
        REQUIRE(scheduler.enqueue({static_cast<int64_t>(index), 0, FAR_TERRAIN_BASE_STEP},
                                  static_cast<uint32_t>(index)));
    }
    REQUIRE_FALSE(scheduler.hasSubmissionCapacity());
    REQUIRE_FALSE(scheduler.enqueue({static_cast<int64_t>(DISTANT_LIMIT), 0, FAR_TERRAIN_BASE_STEP},
                                    static_cast<uint32_t>(DISTANT_LIMIT)));
    REQUIRE(scheduler.stats().inFlight == DISTANT_LIMIT);
    REQUIRE(scheduler.stats().parkedBase == DISTANT_LIMIT);

    constexpr FarTerrainKey PROTECTED_PREVIEW{999, -999, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueUrgentCoverage(PROTECTED_PREVIEW, 0));
    FarTerrainSchedulerStats admitted = scheduler.stats();
    REQUIRE(admitted.inFlight == DISTANT_LIMIT + 1);
    REQUIRE(admitted.parkedBase == DISTANT_LIMIT + 1);
    REQUIRE(admitted.urgentRefinementInFlight == 1);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW) >
            0);
    REQUIRE(authority->prepareCalls(
                worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF) == 0);
    REQUIRE(authority->latestProtectedHandoffEpoch() == 0);
    REQUIRE_FALSE(context->failure().has_value());

    constexpr FarTerrainKey PROTECTED_FINAL{1000, -1000, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueueFinalBase(PROTECTED_FINAL, 1, true));
    admitted = scheduler.stats();
    REQUIRE(admitted.inFlight == DISTANT_LIMIT + 2);
    REQUIRE(admitted.parkedBase == DISTANT_LIMIT + 2);
    REQUIRE(admitted.urgentRefinementInFlight == 2);
    REQUIRE(authority->prepareCalls(
                worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF) > 0);
    REQUIRE(authority->latestProtectedHandoffEpoch() > 0);
    REQUIRE_FALSE(context->failure().has_value());
    scheduler.shutdown();
}

TEST_CASE("Warm urgent preview coverage cannot poison the shared FINAL context",
          "[render][far-terrain][scheduler][authority][preview][warm][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setWorkerBudget(0);
    scheduler.setCanopyWorkerBudget(0);

    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueUrgentCoverage({19, -19, FAR_TERRAIN_BASE_STEP}, 0));

    const FarTerrainSchedulerStats admitted = scheduler.stats();
    REQUIRE(admitted.inFlight == 1);
    REQUIRE(admitted.queuedBase == 1);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW) >
            0);
    REQUIRE(authority->prepareCalls(
                worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF) == 0);
    REQUIRE(authority->latestProtectedHandoffEpoch() == 0);
    REQUIRE_FALSE(context->failure().has_value());
    scheduler.shutdown();
}

TEST_CASE("Critical preview coverage replaces a farther queued parent at the hard cap",
          "[render][far-terrain][scheduler][coverage][capacity][priority][movement]"
          "[regression]") {
    constexpr FarTerrainKey BLOCKER{2'400, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey OPTIONAL_URGENT{2'200, 0, FarTerrainStep::SIXTEEN};
    constexpr std::array DISTANT_PARENTS{
        FarTerrainKey{1'800, 0, FAR_TERRAIN_BASE_STEP},
        FarTerrainKey{2'000, 0, FAR_TERRAIN_BASE_STEP},
    };
    constexpr FarTerrainKey NEAR_PARENT{0, 0, FAR_TERRAIN_BASE_STEP};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockerEntered = false;
    bool releaseBlocker = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE && z == 0) {
            std::unique_lock lock(gateMutex);
            if (!blockerEntered) {
                blockerEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releaseBlocker; });
            }
        }
        return sample(x, z, footprint);
    };

    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 128 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
    scheduler.setCanopyWorkerBudget(0);
    scheduler.setWorkerBudget(1);
    const std::vector order{NEAR_PARENT, BLOCKER, OPTIONAL_URGENT, DISTANT_PARENTS[0],
                            DISTANT_PARENTS[1]};
    scheduler.retainWanted(
        std::unordered_set<FarTerrainKey, FarTerrainKeyHash>(order.begin(), order.end()), order);

    REQUIRE(scheduler.enqueueUrgentRefinement(BLOCKER, 100));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; }));
    }
    REQUIRE(scheduler.enqueue(DISTANT_PARENTS[0], 200));
    REQUIRE(scheduler.enqueue(DISTANT_PARENTS[1], 300));
    REQUIRE(scheduler.enqueueUrgentRefinement(OPTIONAL_URGENT, 400));
    REQUIRE(scheduler.stats().inFlight == limits.maxPending);

    scheduler.advanceProtectedHandoffEpoch();
    REQUIRE(scheduler.enqueueUrgentCoverage(NEAR_PARENT, 0));
    const FarTerrainSchedulerStats admitted = scheduler.stats();
    REQUIRE(admitted.inFlight == limits.maxPending);
    REQUIRE(admitted.canceled == 1);
    REQUIRE(admitted.criticalDisplacements == 1);
    REQUIRE(admitted.urgentRefinementInFlight == 3);
    {
        std::lock_guard lock(gateMutex);
        releaseBlocker = true;
    }
    gateCv.notify_all();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    const std::shared_ptr<const FarTerrainMesh> near = scheduler.findCached(NEAR_PARENT);
    REQUIRE(near);
    REQUIRE(near->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    const size_t retainedDistantParents =
        static_cast<size_t>(scheduler.findCached(DISTANT_PARENTS[0]) != nullptr) +
        static_cast<size_t>(scheduler.findCached(DISTANT_PARENTS[1]) != nullptr);
    REQUIRE(retainedDistantParents == 1);
    scheduler.shutdown();
}

TEST_CASE("Cold entry scheduler excludes FINAL work until gameplay",
          "[render][far-terrain][scheduler][startup][authority][priority][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey ENTRY_PARENT{0, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FINAL_PARENT{1, 0, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey PROTECTED_CHILD{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey VISIBLE_FINAL_CHILD{0, 0, FarTerrainStep::EIGHT};

    scheduler.setFinalStreamingWorkEnabled(false);
    REQUIRE(scheduler.enqueue(ENTRY_PARENT, 0));
    REQUIRE(scheduler.stats().parkedBase == 1);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityQuality::PREVIEW) > 0);
    const uint64_t finalCallsBefore =
        authority->prepareCalls(worldgen::learned::AuthorityQuality::FINAL);
    REQUIRE_FALSE(scheduler.enqueueFinalBase(FINAL_PARENT, 0, true));
    REQUIRE_FALSE(scheduler.enqueueUrgentFinalRefinement(PROTECTED_CHILD, 0));
    REQUIRE_FALSE(scheduler.enqueueFinalRefinement(VISIBLE_FINAL_CHILD, 0));
    scheduler.pumpFinalBaseAuthority();
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityQuality::FINAL) ==
            finalCallsBefore);

    // The first gameplay planning pass reopens the same bounded scheduler
    // synchronously. Protected FINAL work can enter without rebuilding or
    // canceling the connected PREVIEW parent already parked for entry.
    scheduler.setFinalStreamingWorkEnabled(true);
    REQUIRE(scheduler.enqueueFinalBase(FINAL_PARENT, 0, true));
    REQUIRE(scheduler.stats().parkedBase == 2);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityQuality::FINAL) > finalCallsBefore);
    scheduler.shutdown();
}

TEST_CASE("Queued and parked preview keys upgrade in their existing scheduler slot",
          "[render][far-terrain][scheduler][authority][promotion][priority][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 128 * 1024 * 1024;

    SECTION("parked preview becomes protected FINAL without another slot") {
        std::mutex gateMutex;
        std::condition_variable gateCv;
        bool entered = false;
        bool released = false;
        FarTerrainSource source = farTerrainTestSource();
        const auto sample = source.sample;
        source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
            {
                std::unique_lock lock(gateMutex);
                if (!entered) {
                    entered = true;
                    gateCv.notify_all();
                    gateCv.wait(lock, [&] { return released; });
                }
            }
            return sample(x, z, footprint);
        };
        FarTerrainScheduler scheduler(source, context, limits);
        FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, released};
        scheduler.setCanopyWorkerBudget(0);
        constexpr FarTerrainKey KEY{4, -9, FAR_TERRAIN_BASE_STEP};
        REQUIRE(scheduler.enqueue(KEY, 10));
        REQUIRE(scheduler.stats().parkedBase == 1);
        REQUIRE(scheduler.stats().inFlight == 1);
        REQUIRE(scheduler.enqueueFinalBase(KEY, 0, true));
        {
            std::unique_lock lock(gateMutex);
            const bool started =
                gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; });
            if (!started)
                released = true;
            REQUIRE(started);
        }
        REQUIRE(scheduler.stats().parkedBase == 0);
        REQUIRE(scheduler.stats().inFlight == 1);
        REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
        {
            std::lock_guard lock(gateMutex);
            released = true;
        }
        gateCv.notify_all();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
        const auto final = scheduler.findCached(KEY);
        REQUIRE(final);
        REQUIRE(final->authorityQuality == FarTerrainAuthorityQuality::FINAL);
        REQUIRE(scheduler.stats().submitted == 1);
        scheduler.shutdown();
    }

    SECTION("queued preview is reprioritized and promoted before dispatch") {
        authority->setReady();
        std::mutex gateMutex;
        std::condition_variable gateCv;
        bool blockerEntered = false;
        bool releaseBlocker = false;
        constexpr FarTerrainKey BLOCKER{-20, 5, FarTerrainStep::SIXTEEN};
        constexpr FarTerrainKey TARGET{-19, 5, FAR_TERRAIN_BASE_STEP};
        FarTerrainSource source = farTerrainTestSource();
        const auto sample = source.sample;
        source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
            if (x == BLOCKER.tileX * FAR_TERRAIN_TILE_EDGE &&
                z == BLOCKER.tileZ * FAR_TERRAIN_TILE_EDGE) {
                std::unique_lock lock(gateMutex);
                if (!blockerEntered) {
                    blockerEntered = true;
                    gateCv.notify_all();
                    gateCv.wait(lock, [&] { return releaseBlocker; });
                }
            }
            return sample(x, z, footprint);
        };
        FarTerrainScheduler scheduler(source, context, limits);
        FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBlocker};
        scheduler.setCanopyWorkerBudget(0);
        scheduler.setWorkerBudget(1);
        REQUIRE(scheduler.enqueue(BLOCKER, 100));
        {
            std::unique_lock lock(gateMutex);
            const bool started =
                gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return blockerEntered; });
            if (!started)
                releaseBlocker = true;
            REQUIRE(started);
        }
        REQUIRE(scheduler.enqueue(TARGET, 50));
        REQUIRE(scheduler.stats().queuedBase == 1);
        REQUIRE(scheduler.enqueueFinalBase(TARGET, 0, true));
        REQUIRE(scheduler.stats().queuedBase == 1);
        REQUIRE(scheduler.stats().queuedUrgentRefinement == 1);
        REQUIRE(scheduler.stats().inFlight == 2);
        {
            std::lock_guard lock(gateMutex);
            releaseBlocker = true;
        }
        gateCv.notify_all();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
        const auto final = scheduler.findCached(TARGET);
        REQUIRE(final);
        REQUIRE(final->authorityQuality == FarTerrainAuthorityQuality::FINAL);
        REQUIRE(scheduler.stats().submitted == 2);
        scheduler.shutdown();
    }
}

TEST_CASE("An active preview key retains one deterministic FINAL followup",
          "[render][far-terrain][scheduler][authority][promotion][followup][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool previewEntered = false;
    bool releasePreview = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (!previewEntered) {
                previewEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releasePreview; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releasePreview};
    scheduler.setCanopyWorkerBudget(0);
    scheduler.setWorkerBudget(1);
    constexpr FarTerrainKey KEY{31, -14, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueue(KEY, 20));
    {
        std::unique_lock lock(gateMutex);
        const bool started =
            gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return previewEntered; });
        if (!started)
            releasePreview = true;
        REQUIRE(started);
    }
    REQUIRE(scheduler.enqueueFinalBase(KEY, 0, true));
    REQUIRE_FALSE(scheduler.enqueueFinalBase(KEY, 0, true));
    REQUIRE(scheduler.stats().inFlight == 1);
    REQUIRE(scheduler.stats().terrainFollowups == 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    {
        std::lock_guard lock(gateMutex);
        releasePreview = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().terrainFollowups == 0);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.stats().submitted == 2);
    REQUIRE(scheduler.stats().built == 2);
    const auto final = scheduler.findCached(KEY);
    REQUIRE(final);
    REQUIRE(final->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    scheduler.shutdown();
}

TEST_CASE("An epoch change cancels an active FINAL followup without leaking its quota",
          "[render][far-terrain][scheduler][authority][followup][cancellation][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool previewEntered = false;
    bool releasePreview = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (!previewEntered) {
                previewEntered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return releasePreview; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, context, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releasePreview};
    scheduler.setCanopyWorkerBudget(0);
    scheduler.setWorkerBudget(1);
    constexpr FarTerrainKey KEY{-41, 12, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueue(KEY, 10));
    {
        std::unique_lock lock(gateMutex);
        const bool started =
            gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return previewEntered; });
        if (!started)
            releasePreview = true;
        REQUIRE(started);
    }
    REQUIRE(scheduler.enqueueFinalBase(KEY, 0, true));
    REQUIRE(scheduler.stats().terrainFollowups == 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    scheduler.advanceEpoch();
    REQUIRE(scheduler.stats().terrainFollowups == 0);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.stats().inFlight == 1);
    {
        std::lock_guard lock(gateMutex);
        releasePreview = true;
    }
    gateCv.notify_all();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE_FALSE(scheduler.findCached(KEY));
    REQUIRE(scheduler.stats().canceled == 1);
    scheduler.shutdown();
}

TEST_CASE("Far terrain releases deferred authority work for a later retry",
          "[render][far-terrain][scheduler][learned]") {
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    std::atomic<bool> deferFirstSample{true};
    source.sample = [sample, &deferFirstSample](int64_t x, int64_t z,
                                                worldgen::SurfaceFootprint footprint) {
        if (deferFirstSample.exchange(false)) {
            throw worldgen::learned::GenerationFailureException(
                worldgen::learned::AuthorityStatus::DEFERRED,
                {.code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                 .message = "Synthetic cold far authority page",
                 .retriable = true});
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 8 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), limits);
    constexpr FarTerrainKey KEY{12, -7, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueue(KEY));

    const auto deferredDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((scheduler.stats().deferred == 0 || scheduler.stats().inFlight != 0) &&
           std::chrono::steady_clock::now() < deferredDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().deferred == 1);
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().failed == 0);
    REQUIRE(scheduler.stats().built == 0);
    REQUIRE(scheduler.enqueue(KEY));

    std::vector<FarTerrainResult> completed;
    const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < readyDeadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(scheduler.stats().deferred == 1);
    REQUIRE(scheduler.stats().failed == 0);
    REQUIRE(scheduler.stats().built == 1);
}

TEST_CASE("A preview parent failure does not poison final generation authority",
          "[render][far-terrain][scheduler][authority][preview][failure][learned][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSource source = farTerrainTestSource();
    source.sample = [](int64_t, int64_t, worldgen::SurfaceFootprint) -> FarSurfaceSample {
        throw worldgen::learned::GenerationFailureException(
            worldgen::learned::AuthorityStatus::FAILED,
            {.code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
             .message = "Synthetic preview parent failure",
             .retriable = true});
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    constexpr FarTerrainKey KEY{3, -5, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueue(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().failed);
    REQUIRE_FALSE(completed.front().mesh);
    REQUIRE(scheduler.stats().failed == 1);
    REQUIRE_FALSE(context->failure().has_value());
}

TEST_CASE("A generic final parent failure latches repair state and retains its preview parent",
          "[render][far-terrain][scheduler][authority][failure][parent][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    std::atomic<bool> failFinal{false};
    source.sample = [sample, &failFinal](int64_t x, int64_t z,
                                         worldgen::SurfaceFootprint footprint) {
        if (failFinal.load(std::memory_order_acquire))
            throw std::runtime_error("Synthetic final parent mesh failure");
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{-4, 7, FAR_TERRAIN_BASE_STEP};

    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
    };
    REQUIRE(scheduler.enqueue(KEY));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> preview = scheduler.findCached(KEY);
    REQUIRE(preview);
    REQUIRE(preview->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 1);

    failFinal.store(true, std::memory_order_release);
    REQUIRE(scheduler.enqueueFinalBase(KEY, 0, true));
    waitForIdle();
    completed.clear();
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().failed);
    REQUIRE_FALSE(completed.front().mesh);
    REQUIRE(scheduler.findCached(KEY) == preview);
    const std::optional<worldgen::learned::GenerationFailure> failure = context->failure();
    REQUIRE(failure.has_value());
    REQUIRE(failure->code == worldgen::learned::GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(failure->retriable);
    REQUIRE(failure->message.find("Synthetic final parent mesh failure") != std::string::npos);
    scheduler.shutdown();
}

TEST_CASE("An unknown final refinement failure latches repair state",
          "[render][far-terrain][scheduler][authority][failure][refinement][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    std::atomic<bool> failFinal{true};
    source.sample = [sample, &failFinal](int64_t x, int64_t z,
                                         worldgen::SurfaceFootprint footprint) {
        if (failFinal.load(std::memory_order_acquire))
            throw 17;
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 2;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{3, -2, FarTerrainStep::EIGHT};

    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        REQUIRE(scheduler.stats().inFlight == 0);
    };
    REQUIRE(scheduler.enqueueUrgentFinalRefinement(KEY));
    waitForIdle();
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().failed);
    REQUIRE_FALSE(completed.front().mesh);
    REQUIRE_FALSE(scheduler.findCached(KEY));
    const std::optional<worldgen::learned::GenerationFailure> failure = context->failure();
    REQUIRE(failure.has_value());
    REQUIRE(failure->code == worldgen::learned::GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(failure->retriable);
    REQUIRE(failure->message.find("unknown exception") != std::string::npos);
    scheduler.shutdown();
}

TEST_CASE("Cold base authority parks one parent until its preview closure is ready",
          "[render][far-terrain][scheduler][authority][startup][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    std::atomic<uint64_t> meshSampleCalls{0};
    source.sample = [sample, &meshSampleCalls](int64_t x, int64_t z,
                                               worldgen::SurfaceFootprint footprint) {
        meshSampleCalls.fetch_add(1, std::memory_order_relaxed);
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    constexpr FarTerrainKey KEY{0, 0, FAR_TERRAIN_BASE_STEP};
    FarTerrainViewTile tile;
    tile.key = KEY;
    tile.bounds = {.minX = 0,
                   .maxX = FAR_TERRAIN_TILE_EDGE,
                   .minZ = 0,
                   .maxZ = FAR_TERRAIN_TILE_EDGE,
                   .minY = static_cast<float>(WORLD_MIN_Y),
                   .maxY = static_cast<float>(WORLD_MAX_Y + 1)};
    const std::array selected{tile};
    scheduler.setCoarseAuthorityPrefetchPages(farTerrainCoarseAuthorityPages(
        selected, FAR_TERRAIN_TILE_EDGE / 2.0, FAR_TERRAIN_TILE_EDGE / 2.0));
    scheduler.pumpCoarseAuthorityPrefetch();
    REQUIRE(scheduler.enqueue(KEY));

    const uint64_t callsAfterFirstSubmission = authority->prepareCalls();
    for (int frame = 0; frame < 64; ++frame) {
        // This mirrors stable render frames: the authority pump may observe
        // its own single flight, but the absent parent must not re-enter a
        // terrain worker or issue a duplicate page-closure query.
        REQUIRE_FALSE(scheduler.enqueue(KEY));
        scheduler.pumpCoarseAuthorityPrefetch();
    }
    const FarTerrainSchedulerStats parked = scheduler.stats();
    REQUIRE(parked.submitted == 1);
    REQUIRE(parked.inFlight == 1);
    REQUIRE(parked.parkedBase == 1);
    REQUIRE(parked.queuedBase == 0);
    REQUIRE(parked.activeBaseWorkers == 0);
    REQUIRE(parked.deferred == 0);
    REQUIRE(meshSampleCalls.load(std::memory_order_relaxed) == 0);
    REQUIRE(authority->prepareCalls() > callsAfterFirstSubmission);

    authority->setReady();
    std::vector<FarTerrainResult> completed;
    const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < readyDeadline) {
        scheduler.pumpCoarseAuthorityPrefetch();
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(meshSampleCalls.load(std::memory_order_relaxed) != 0);
    REQUIRE(scheduler.stats().parkedBase == 0);
    REQUIRE(scheduler.stats().deferred == 0);
    REQUIRE(scheduler.stats().built == 1);
}

TEST_CASE("A final parent cannot strand when transient terrain wins the parking race",
          "[render][far-terrain][scheduler][authority][final][race][hash][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    authority->setTransientReadyAfterFirstDeferral();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSource source = farTerrainTestSource();
    source.finalBaseAuthorityDependencies = [](FarTerrainKey key) {
        return farTerrainFinalBaseAuthorityDependencies(key);
    };
    constexpr FarTerrainKey KEY{1, 1, FAR_TERRAIN_BASE_STEP};
    const std::shared_ptr<const FarTerrainMesh> expected =
        FarTerrainMesher::build(KEY, source, FarTerrainAuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    REQUIRE(scheduler.enqueueFinalBase(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(authority->transientCalls() == 2);
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(completed.front().mesh->deterministicHash == expected->deterministicHash);
    REQUIRE(scheduler.stats().submitted == 1);
    REQUIRE(scheduler.stats().built == 1);
    REQUIRE(scheduler.stats().parkedBase == 0);
    REQUIRE(scheduler.stats().deferred == 0);
}

TEST_CASE("A parked final parent retains one transient inference flight",
          "[render][far-terrain][scheduler][authority][final][single-flight][regression]") {
    TempDir directory("far_final_parent_transient_single_flight");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto backend = std::make_shared<BlockingTransientBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(), backend);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    std::atomic<uint64_t> meshSampleCalls{0};
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [sample, &meshSampleCalls](int64_t x, int64_t z,
                                               worldgen::SurfaceFootprint footprint) {
        meshSampleCalls.fetch_add(1, std::memory_order_relaxed);
        return sample(x, z, footprint);
    };
    source.finalBaseAuthorityDependencies = [](FarTerrainKey key) {
        return farTerrainFinalBaseAuthorityDependencies(key);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{1, 1, FAR_TERRAIN_BASE_STEP};
    const bool submitted = scheduler.enqueueFinalBase(KEY);
    const bool entered = backend->waitUntilEntered(std::chrono::seconds(2));

    bool duplicateAccepted = false;
    for (int frame = 0; frame < 64; ++frame)
        duplicateAccepted = scheduler.enqueueFinalBase(KEY) || duplicateAccepted;
    const uint64_t callsBeforePumps = backend->callCount();
    for (int frame = 0; frame < 8; ++frame)
        scheduler.pumpFinalBaseAuthority();
    const uint64_t callsWhileParked = backend->callCount();
    const FarTerrainSchedulerStats parked = scheduler.stats();
    const uint64_t samplesWhileParked = meshSampleCalls.load(std::memory_order_relaxed);
    backend->release();

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpFinalBaseAuthority();
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(submitted);
    REQUIRE(entered);
    REQUIRE_FALSE(duplicateAccepted);
    REQUIRE(callsBeforePumps == 1);
    REQUIRE(callsWhileParked == 1);
    REQUIRE(parked.submitted == 1);
    REQUIRE(parked.inFlight == 1);
    REQUIRE(parked.parkedBase == 1);
    REQUIRE(parked.queuedBase == 0);
    REQUIRE(parked.activeBaseWorkers == 0);
    REQUIRE(samplesWhileParked == 0);
    REQUIRE(authority->cacheMetrics().singleFlightDeferrals > 0);
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(backend->callCount() == 1);
    REQUIRE(scheduler.stats().built == 1);
    REQUIRE(scheduler.stats().parkedBase == 0);
}

TEST_CASE("A quiescent final parent retries after hydrology defers without learned work",
          "[render][far-terrain][scheduler][authority][final][hydrology][liveness][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    authority->setTransientReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSource source = farTerrainTestSource();
    source.finalBaseAuthorityDependencies = [](FarTerrainKey key) {
        return farTerrainFinalBaseAuthorityDependencies(key);
    };
    const auto sample = source.sample;
    std::atomic<uint32_t> deferredCalls{0};
    source.sample = [sample, &deferredCalls](int64_t x, int64_t z,
                                             worldgen::SurfaceFootprint footprint) {
        if (deferredCalls.fetch_add(1, std::memory_order_relaxed) == 0) {
            throw worldgen::learned::GenerationFailureException(
                worldgen::learned::AuthorityStatus::DEFERRED,
                {.code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                 .message = "Synthetic hydrology reconciliation became observable while idle",
                 .retriable = true});
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{2, -3, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueueFinalBase(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpFinalBaseAuthority();
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    const FarTerrainSchedulerStats stats = scheduler.stats();
    CAPTURE(stats.inFlight, stats.parkedBase, stats.submitted, stats.built,
            stats.quiescentAuthorityResumes, context->failure());
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(completed.front().mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.parkedBase == 0);
    REQUIRE(stats.submitted == 1);
    REQUIRE(stats.built == 1);
    REQUIRE(stats.quiescentAuthorityResumes == 1);
    REQUIRE_FALSE(context->failure());
}

TEST_CASE("A full protected parent lane cannot strand on quiescent hydrology deferrals",
          "[render][far-terrain][scheduler][authority][final][hydrology][liveness][protected]["
          "regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    authority->setTransientReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSource source = farTerrainTestSource();
    source.canopies = {};
    source.finalBaseAuthorityDependencies = [](FarTerrainKey key) {
        return farTerrainFinalBaseAuthorityDependencies(key);
    };
    const auto sample = source.sample;
    std::atomic<uint32_t> remainingDeferrals{
        static_cast<uint32_t>(FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT)};
    source.sample = [sample, &remainingDeferrals](int64_t x, int64_t z,
                                                  worldgen::SurfaceFootprint footprint) {
        uint32_t remaining = remainingDeferrals.load(std::memory_order_relaxed);
        while (remaining != 0 && !remainingDeferrals.compare_exchange_weak(
                                     remaining, remaining - 1, std::memory_order_relaxed)) {
        }
        if (remaining != 0) {
            throw worldgen::learned::GenerationFailureException(
                worldgen::learned::AuthorityStatus::DEFERRED,
                {.code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                 .message = "Synthetic protected hydrology reconciliation became idle",
                 .retriable = true});
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT;
    limits.maxCompleted = FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT;
    limits.maxCacheEntries = FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT;
    limits.maxCacheBytes = 128 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    std::array<FarTerrainKey, FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT> keys{};
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = {static_cast<int64_t>(index % 4), static_cast<int64_t>(index / 4),
                       FAR_TERRAIN_BASE_STEP};
        REQUIRE(scheduler.enqueueFinalBase(keys[index], static_cast<uint32_t>(index), true));
    }

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (completed.size() != keys.size() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpFinalBaseAuthority();
        std::vector<FarTerrainResult> frame;
        scheduler.drainCompleted(frame);
        completed.insert(completed.end(), std::make_move_iterator(frame.begin()),
                         std::make_move_iterator(frame.end()));
        if (completed.size() != keys.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    const FarTerrainSchedulerStats stats = scheduler.stats();
    CAPTURE(completed.size(), stats.inFlight, stats.parkedBase, stats.submitted, stats.built,
            stats.quiescentAuthorityResumes, context->failure());
    REQUIRE(completed.size() == keys.size());
    REQUIRE(std::ranges::all_of(completed, [](const FarTerrainResult& result) {
        return !result.failed && result.mesh &&
               result.mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL;
    }));
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.parkedBase == 0);
    REQUIRE(stats.submitted == keys.size());
    REQUIRE(stats.built == keys.size());
    REQUIRE(stats.quiescentAuthorityResumes == keys.size());
    REQUIRE_FALSE(context->failure());
}

TEST_CASE("A cold production final parent runs only after its exact dependencies are ready",
          "[render][far-terrain][scheduler][authority][final][v4][regression]") {
    TempDir directory("far_final_parent_exact_dependencies");
    constexpr uint64_t SEED = 0xF1A1'BA5E'0000'0042ULL;
    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    auto backend = std::make_shared<AuthorityQualityRecordingBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(), backend);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(FarTerrainMesher::generatorGeometrySource(generator), context,
                                  limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{1, 1, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueueFinalBase(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpFinalBaseAuthority();
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    const FarTerrainSchedulerStats stats = scheduler.stats();
    const worldgen::learned::WorldGenerationMetrics generationMetrics = context->metrics();
    const worldgen::NativeHydrologyCacheMetrics hydrologyMetrics =
        context->nativeHydrologyRouter()->cacheMetrics();
    CAPTURE(stats.submitted, stats.built, stats.deferred, stats.failed,
            stats.authorityCompletionResumes, stats.inFlight, stats.parkedBase,
            generationMetrics.authorityCache.builds, generationMetrics.authorityCache.hits,
            generationMetrics.authorityCache.singleFlightDeferrals, hydrologyMetrics.builds,
            hydrologyMetrics.deferredBuilds, hydrologyMetrics.failures, context->failure());
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(completed.front().mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(stats.submitted == 1);
    REQUIRE(stats.built == 1);
    REQUIRE(stats.deferred == 0);
    REQUIRE(stats.parkedBase == 0);
    REQUIRE(backend->pageCalls(worldgen::learned::AuthorityQuality::FINAL) == 0);
    REQUIRE(backend->finalRectangleCalls() >= 1);
    REQUIRE(hydrologyMetrics.deferredBuilds > 0);
    REQUIRE(hydrologyMetrics.failures == 0);
    REQUIRE(stats.authorityCompletionResumes == hydrologyMetrics.deferredBuilds);
    REQUIRE(stats.authorityCompletionResumes <= worldgen::NATIVE_HYDROLOGY_MAX_SPILL_SUMMARY_PAGES);
}

TEST_CASE("A final parent crossing two native seams retains every owner input",
          "[render][far-terrain][scheduler][authority][final][hydrology][seam][regression]") {
    TempDir directory("far_final_parent_four_owner_seam");
    constexpr uint64_t SEED = 0xF1A1'5EAA'0000'0042ULL;
    constexpr FarTerrainKey KEY{7, 7, FAR_TERRAIN_BASE_STEP};
    const FarTerrainFinalBaseAuthorityDependencies dependencies =
        farTerrainFinalBaseAuthorityDependencies(KEY);
    REQUIRE(dependencies.nativeHydrology.size() == 4);
    REQUIRE_FALSE(dependencies.transientGeometryRegion);

    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    auto backend = std::make_shared<AuthorityQualityRecordingBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(), backend);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCompleted = 1;
    limits.maxCacheEntries = 1;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(FarTerrainMesher::generatorGeometrySource(generator), context,
                                  limits);
    scheduler.setCanopyWorkerBudget(0);
    REQUIRE(scheduler.enqueueFinalBase(KEY));

    std::vector<FarTerrainResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpFinalBaseAuthority();
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    const FarTerrainSchedulerStats stats = scheduler.stats();
    const std::map<worldgen::learned::NativeRect, uint64_t> calls =
        backend->finalRectangleCallCounts();
    CAPTURE(stats.submitted, stats.built, stats.deferred, stats.failed,
            stats.authorityCompletionResumes, calls.size(), backend->finalRectangleCalls(),
            context->failure());
    REQUIRE(completed.size() == 1);
    REQUIRE_FALSE(completed.front().failed);
    REQUIRE(completed.front().mesh);
    REQUIRE(stats.submitted == 1);
    REQUIRE(stats.built == 1);
    REQUIRE(stats.deferred == 0);
    REQUIRE(calls.size() >= dependencies.nativeHydrology.size());
    for (const FarTerrainNativeHydrologyDependency& dependency : dependencies.nativeHydrology) {
        CAPTURE(dependency.ownerPageX, dependency.ownerPageZ, dependency.finalTerrainRegion);
        const auto found = calls.find(dependency.finalTerrainRegion);
        REQUIRE(found != calls.end());
        REQUIRE(found->second == 1);
        REQUIRE(
            context->nativeHydrologyOwnerPrepared(dependency.ownerPageX, dependency.ownerPageZ));
    }
    for (const auto& [region, callCount] : calls) {
        CAPTURE(region, callCount);
        REQUIRE(callCount == 1);
    }
}

TEST_CASE("Final parent cache entries replace preview entries without downgrade",
          "[render][far-terrain][scheduler][authority][cache][lod][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{-2, 3, FAR_TERRAIN_BASE_STEP};
    const std::array keys{KEY};

    REQUIRE(scheduler.enqueue(KEY));
    const auto previewDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < previewDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    const auto preview = scheduler.findCached(KEY);
    REQUIRE(preview);
    REQUIRE(preview->authorityQuality == Quality::PREVIEW);

    std::vector<std::shared_ptr<const FarTerrainMesh>> finalBatch;
    scheduler.findCachedBatch(keys, 1, finalBatch, Quality::FINAL);
    REQUIRE(finalBatch.empty());

    REQUIRE(scheduler.enqueueFinalBase(KEY));
    const auto finalDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < finalDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    const auto final = scheduler.findCached(KEY);
    REQUIRE(final);
    REQUIRE(final->authorityQuality == Quality::FINAL);
    REQUIRE(final->deterministicHash != preview->deterministicHash);
    REQUIRE(scheduler.stats().cacheEntries == 1);
    REQUIRE(scheduler.stats().cacheBaseEntries == 1);

    scheduler.findCachedBatch(keys, 1, finalBatch, Quality::FINAL);
    REQUIRE(finalBatch.size() == 1);
    REQUIRE(finalBatch.front() == final);
    REQUIRE_FALSE(scheduler.enqueue(KEY));
    scheduler.shutdown();
}

TEST_CASE("Preview proxy tiers refine geometry before atomic same-key FINAL replacement",
          "[render][far-terrain][scheduler][authority][preview][final][lod][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 6;
    limits.maxCacheBytes = 512 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr ColumnPos COORDINATE{4, -6};
    constexpr FarTerrainKey BASE{COORDINATE.x, COORDINATE.z, FAR_TERRAIN_BASE_STEP};
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueue(BASE));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> previewBase = scheduler.findCached(BASE);
    REQUIRE(previewBase);
    REQUIRE(previewBase->authorityQuality == Quality::PREVIEW);
    REQUIRE(scheduler.stats().canopySubmitted == 0);

    FarTerrainStep displayed = FAR_TERRAIN_BASE_STEP;
    FarTerrainStepMask residentSteps = farTerrainStepMask(displayed);
    const std::array requests{FarTerrainRefinementCacheRequest{
        .coordinate = COORDINATE,
        .displayed = displayed,
        .desired = FarTerrainStep::ONE,
        .residentSteps = residentSteps,
    }};
    std::vector<FarTerrainKey> submissions;
    buildFarTerrainProgressiveSubmissionOrder(requests, submissions);
    REQUIRE((submissions ==
             std::vector<FarTerrainKey>{{COORDINATE.x, COORDINATE.z, FarTerrainStep::SIXTEEN}}));
    REQUIRE(scheduler.enqueueUrgentRefinement(submissions.front()));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> cached = scheduler.findCached(submissions.front());
    REQUIRE(cached);
    REQUIRE(cached->authorityQuality == Quality::PREVIEW);
    for (const FarTerrainStep step :
         {FarTerrainStep::EIGHT, FarTerrainStep::FOUR, FarTerrainStep::TWO}) {
        const FarTerrainKey key{COORDINATE.x, COORDINATE.z, step};
        REQUIRE(scheduler.enqueueUrgentRefinement(key));
        waitForIdle();
        const std::shared_ptr<const FarTerrainMesh> tier = scheduler.findCached(key);
        REQUIRE(tier);
        REQUIRE(tier->authorityQuality == Quality::PREVIEW);
    }

    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityQuality::PREVIEW) > 0);
    REQUIRE(authority->prepareCalls(worldgen::learned::AuthorityQuality::FINAL) == 0);
    constexpr FarTerrainKey PROMOTION{COORDINATE.x, COORDINATE.z, FarTerrainStep::EIGHT};
    REQUIRE(scheduler.enqueueFinalRefinement(PROMOTION));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> final = scheduler.findCached(PROMOTION);
    REQUIRE(final);
    REQUIRE(final->authorityQuality == Quality::FINAL);
    REQUIRE(final->exactAuthorityCompatible);
    const std::shared_ptr<const FarTerrainMesh> retainedPreview =
        scheduler.findCached({COORDINATE.x, COORDINATE.z, FarTerrainStep::FOUR});
    REQUIRE(retainedPreview);
    REQUIRE(retainedPreview->authorityQuality == Quality::PREVIEW);
    REQUIRE_FALSE(context->failure().has_value());
    scheduler.shutdown();
}

TEST_CASE("Final refinement cache entries cannot be downgraded",
          "[render][far-terrain][scheduler][authority][final][cache][lod][regression]") {
    using Quality = FarTerrainAuthorityQuality;
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    authority->setReady();
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr FarTerrainKey KEY{5, -2, FarTerrainStep::EIGHT};
    const auto waitForIdle = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(scheduler.stats().inFlight == 0);
    };

    REQUIRE(scheduler.enqueueFinalRefinement(KEY));
    waitForIdle();
    const std::shared_ptr<const FarTerrainMesh> final = scheduler.findCached(KEY);
    REQUIRE(final);
    REQUIRE(final->authorityQuality == Quality::FINAL);
    REQUIRE(scheduler.stats().cacheEntries == 1);
    REQUIRE(scheduler.stats().cacheBaseEntries == 0);

    REQUIRE_FALSE(scheduler.enqueueFinalRefinement(KEY));
    REQUIRE_FALSE(scheduler.enqueueUrgentRefinement(KEY));
    REQUIRE(scheduler.findCached(KEY) == final);
    scheduler.shutdown();
}

TEST_CASE("Parked final parents retain the urgent cap through resume and cancellation",
          "[render][far-terrain][scheduler][authority][priority][cancellation][regression]") {
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<GateablePreviewAuthority>(identity);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxPending = FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT + 1;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 4;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);
    constexpr auto KEYS = [] {
        std::array<FarTerrainKey, FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT + 1> keys{};
        for (size_t index = 0; index < keys.size(); ++index) {
            keys[index] = FarTerrainKey{static_cast<int64_t>(index), 0, FAR_TERRAIN_BASE_STEP};
        }
        return keys;
    }();
    for (size_t index = 0; index < FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT; ++index) {
        CAPTURE(index, FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT, scheduler.stats().inFlight,
                scheduler.stats().parkedBase, scheduler.stats().urgentRefinementInFlight,
                context->failure().has_value());
        REQUIRE(scheduler.enqueueFinalBase(KEYS[index], static_cast<uint32_t>(index)));
    }
    REQUIRE_FALSE(scheduler.enqueueFinalBase(
        KEYS.back(), static_cast<uint32_t>(FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT)));

    FarTerrainSchedulerStats parked = scheduler.stats();
    REQUIRE(parked.parkedBase == FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT);
    REQUIRE(parked.urgentRefinementInFlight == FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT);
    REQUIRE(parked.visibleFinalParentInFlight == FAR_TERRAIN_MAX_VISIBLE_FINAL_PARENTS_IN_FLIGHT);
    REQUIRE(parked.queuedUrgentRefinement == 0);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());
    constexpr FarTerrainKey PROXY{77, 0, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueueUrgentRefinement(PROXY));

    const std::vector<FarTerrainKey> retainedOrder{KEYS[0], KEYS[1]};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> retained(retainedOrder.begin(),
                                                                        retainedOrder.end());
    REQUIRE(scheduler.retainWanted(retained, retainedOrder));
    // A worker may have claimed PROXY before the membership revision. Running
    // terrain work is intentionally nonpreemptive, so wait for that stale job
    // to observe cancellation instead of sampling its urgent accounting in
    // the middle of release.
    const auto cancellationDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().urgentRefinementInFlight != 2 &&
           std::chrono::steady_clock::now() < cancellationDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    parked = scheduler.stats();
    REQUIRE(parked.parkedBase == 2);
    REQUIRE(parked.urgentRefinementInFlight == 2);
    REQUIRE(parked.visibleFinalParentInFlight == 2);
    REQUIRE(scheduler.hasUrgentRefinementCapacity());

    authority->setReady();
    const auto resumeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (scheduler.stats().inFlight != 0 && std::chrono::steady_clock::now() < resumeDeadline) {
        scheduler.pumpFinalBaseAuthority();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const FarTerrainSchedulerStats resumed = scheduler.stats();
    REQUIRE(resumed.inFlight == 0);
    REQUIRE(resumed.parkedBase == 0);
    REQUIRE(resumed.queuedUrgentRefinement == 0);
    REQUIRE(resumed.urgentRefinementInFlight == 0);
    REQUIRE(resumed.visibleFinalParentInFlight == 0);
    REQUIRE(resumed.built >= 2);

    authority->setReady(false);
    constexpr FarTerrainKey SHUTDOWN_KEY{5, 0, FAR_TERRAIN_BASE_STEP};
    const std::vector shutdownOrder{SHUTDOWN_KEY};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> shutdownWanted(shutdownOrder.begin(),
                                                                              shutdownOrder.end());
    REQUIRE(scheduler.retainWanted(shutdownWanted, shutdownOrder));
    REQUIRE(scheduler.enqueueFinalBase(SHUTDOWN_KEY));
    REQUIRE(scheduler.stats().parkedBase == 1);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 1);
    REQUIRE(scheduler.stats().visibleFinalParentInFlight == 1);
    scheduler.shutdown();
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().parkedBase == 0);
    REQUIRE(scheduler.stats().urgentRefinementInFlight == 0);
    REQUIRE(scheduler.stats().visibleFinalParentInFlight == 0);
}

TEST_CASE("Far terrain holds optional canopy workers until coarse coverage opens",
          "[render][far-terrain][scheduler][canopy][startup][coverage][regression]") {
    std::mutex canopyMutex;
    std::condition_variable canopyCv;
    size_t canopyEntered = 0;
    bool releaseCanopies = false;
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        std::unique_lock lock(canopyMutex);
        ++canopyEntered;
        canopyCv.notify_all();
        canopyCv.wait(lock, [&] { return releaseCanopies; });
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 4;
    limits.maxCanopyPending = 2;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey FIRST{7, -4, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey SECOND{8, -4, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueue(FIRST));
    REQUIRE(scheduler.enqueue(SECOND));
    std::vector<FarTerrainResult> terrainCompleted;
    const auto terrainDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (terrainCompleted.size() < 2 && std::chrono::steady_clock::now() < terrainDeadline) {
        scheduler.drainCompleted(terrainCompleted);
        if (terrainCompleted.size() < 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(terrainCompleted.size() == 2);
    REQUIRE(std::ranges::all_of(
        terrainCompleted, [](const FarTerrainResult& result) { return result.mesh != nullptr; }));
    REQUIRE(scheduler.stats().canopySubmitted == 0);
    REQUIRE(scheduler.enqueueCanopy(FIRST, 0, FarTerrainAuthorityQuality::FINAL));
    REQUIRE(scheduler.enqueueCanopy(SECOND, 1, FarTerrainAuthorityQuality::FINAL));
    {
        std::unique_lock lock(canopyMutex);
        REQUIRE_FALSE(canopyCv.wait_for(lock, std::chrono::milliseconds(100),
                                        [&] { return canopyEntered != 0; }));
    }
    REQUIRE(scheduler.stats().activeCanopyWorkers == 0);
    REQUIRE(scheduler.stats().queuedCanopy == 2);

    constexpr double CAMERA_X = -257.25;
    constexpr double CAMERA_Z = 513.75;
    std::vector<FarTerrainViewTile> coldEntry;
    std::vector<FarTerrainViewTile> expanded;
    selectFarTerrainView(CAMERA_X, CAMERA_Z, FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS,
                         coldEntry);
    selectFarTerrainView(CAMERA_X, CAMERA_Z, MAX_RENDER_DISTANCE_CHUNKS, expanded);
    const auto isColdParentResident = [&](const FarTerrainKey& key) {
        return key.step == FAR_TERRAIN_BASE_STEP &&
               std::ranges::any_of(coldEntry, [&](const FarTerrainViewTile& tile) {
                   return tile.key.tileX == key.tileX && tile.key.tileZ == key.tileZ;
               });
    };
    const FarTerrainCoverageFrontier coverage =
        farTerrainCoverageFrontier(expanded, isColdParentResident);
    REQUIRE_FALSE(coverage.complete);
    REQUIRE(farTerrainConnectedRefinementLaneOpen(coverage));

    const size_t debtBudget = farTerrainCanopyWorkerBudget(
        true, farTerrainConnectedRefinementLaneOpen(coverage), true, true);
    REQUIRE(debtBudget == 1);
    scheduler.setCanopyWorkerBudget(debtBudget);
    {
        std::unique_lock lock(canopyMutex);
        REQUIRE(
            canopyCv.wait_for(lock, std::chrono::seconds(2), [&] { return canopyEntered == 1; }));
        REQUIRE_FALSE(canopyCv.wait_for(lock, std::chrono::milliseconds(100),
                                        [&] { return canopyEntered == 2; }));
    }

    scheduler.setCanopyWorkerBudget(farTerrainCanopyWorkerBudget(
        true, farTerrainConnectedRefinementLaneOpen(coverage), false, false));
    bool bothCanopiesEntered = false;
    {
        std::unique_lock lock(canopyMutex);
        bothCanopiesEntered =
            canopyCv.wait_for(lock, std::chrono::seconds(2), [&] { return canopyEntered == 2; });
        releaseCanopies = true;
    }
    canopyCv.notify_all();
    scheduler.shutdown();
    REQUIRE(bothCanopiesEntered);
}

TEST_CASE("Coarse canopy attachments publish preview ecology before final promotion",
          "[render][far-terrain][scheduler][canopy][authority][preview][lod][regression]") {
    STATIC_REQUIRE(FAR_TERRAIN_CANOPY_AUTHORITY_PRIORITY ==
                   worldgen::learned::AuthorityRequestPriority::COARSE_PREVIEW);
    TempDir directory("coarse_canopy_final_ecology");
    constexpr uint64_t SEED = 0xEC01'0A74'CA10'0004ULL;
    const worldgen::learned::GenerationIdentity identity = nativeTopologyTestIdentity(SEED);
    auto backend = std::make_shared<AuthorityQualityRecordingBackend>();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(), backend);
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 1;
    limits.maxCanopyCompleted = 1;
    limits.maxCanopyCacheEntries = 1;
    limits.maxCanopyCacheBytes = 8 * 1024 * 1024;
    FarTerrainScheduler scheduler(SEED, context, limits);
    scheduler.setCanopyWorkerBudget(1);

    constexpr FarTerrainKey PARENT{0, 0, FAR_TERRAIN_BASE_STEP};
    REQUIRE(scheduler.enqueueCanopy(PARENT));
    std::vector<FarCanopyResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.empty() && std::chrono::steady_clock::now() < deadline) {
        scheduler.pumpCanopyAuthority();
        scheduler.drainCanopyCompleted(completed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto finalDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (backend->pageCalls(worldgen::learned::AuthorityQuality::FINAL) == 0 &&
           backend->finalRectangleCalls() == 0 &&
           std::chrono::steady_clock::now() < finalDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().attachment);
    REQUIRE(completed.front().attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(backend->pageCalls(worldgen::learned::AuthorityQuality::PREVIEW) > 0);
    REQUIRE(backend->pageCalls(worldgen::learned::AuthorityQuality::FINAL) +
                backend->finalRectangleCalls() >
            0);
}

TEST_CASE("Visible refined canopies replace queued coarse horizon attachments",
          "[render][far-terrain][scheduler][canopy][priority][regression]") {
    std::atomic<int> firstStep{0};
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        int unset = 0;
        firstStep.compare_exchange_strong(unset, farTerrainStepSize(step),
                                          std::memory_order_relaxed);
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCanopyPending = 2;
    limits.maxCompleted = 2;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey NEAR_PARENT{7, -4, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey FAR_PARENT{8, -4, FAR_TERRAIN_BASE_STEP};
    constexpr FarTerrainKey VISIBLE_REFINEMENT{7, -4, FarTerrainStep::TWO};
    REQUIRE(scheduler.enqueueCanopy(NEAR_PARENT, 0));
    REQUIRE(scheduler.enqueueCanopy(FAR_PARENT, 1));
    REQUIRE(scheduler.enqueueCanopy(VISIBLE_REFINEMENT, 0));
    const FarTerrainSchedulerStats prioritized = scheduler.stats();
    REQUIRE(prioritized.canopySubmitted == 3);
    REQUIRE(prioritized.canopyCanceled == 1);
    REQUIRE(prioritized.canopyInFlight == 2);
    REQUIRE(prioritized.queuedCanopy == 2);

    scheduler.setCanopyWorkerBudget(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (firstStep.load(std::memory_order_relaxed) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(firstStep.load(std::memory_order_relaxed) == farTerrainStepSize(FarTerrainStep::TWO));
    scheduler.shutdown();
}

TEST_CASE("Cold canopy streaming establishes bounded coarse coverage before step one",
          "[render][far-terrain][scheduler][canopy][priority][coverage][regression]") {
    TempDir directory("bounded_coarse_canopy_dispatch");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    std::mutex orderMutex;
    std::condition_variable orderCv;
    std::vector<int> buildOrder;
    std::unordered_set<int64_t> observedTiles;
    bool firstBuildEntered = false;
    bool releaseFirstBuild = false;
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t minimumX, int64_t, int64_t, int64_t, FarTerrainStep step) {
        std::unique_lock lock(orderMutex);
        const int64_t tileX =
            world_coord::floorDiv(minimumX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE));
        if (observedTiles.insert(tileX).second) {
            buildOrder.push_back(farTerrainStepSize(step));
            if (!firstBuildEntered) {
                firstBuildEntered = true;
                orderCv.notify_all();
                orderCv.wait(lock, [&] { return releaseFirstBuild; });
            }
        }
        return std::vector<FarCanopy>{};
    };
    source.flora = {};

    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 9;
    limits.maxCanopyCompleted = 24;
    limits.maxCanopyCacheEntries = 9;
    limits.maxCanopyCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr std::array<FarTerrainKey, 5> INITIAL_KEYS = {
        FarTerrainKey{0, 0, FarTerrainStep::ONE},     FarTerrainKey{1, 0, FarTerrainStep::TWO},
        FarTerrainKey{2, 0, FarTerrainStep::FOUR},    FarTerrainKey{3, 0, FarTerrainStep::EIGHT},
        FarTerrainKey{4, 0, FarTerrainStep::SIXTEEN},
    };
    for (size_t index = 0; index < INITIAL_KEYS.size(); ++index) {
        REQUIRE(scheduler.enqueueCanopy(INITIAL_KEYS[index], static_cast<uint32_t>(index * 100),
                                        FarTerrainAuthorityQuality::FINAL));
    }

    scheduler.setCanopyWorkerBudget(1);
    bool firstBuildObserved = false;
    {
        std::unique_lock lock(orderMutex);
        firstBuildObserved =
            orderCv.wait_for(lock, std::chrono::seconds(2), [&] { return firstBuildEntered; });
    }
    constexpr std::array<FarTerrainKey, 4> RENEWED_KEYS = {
        FarTerrainKey{5, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{6, 0, FarTerrainStep::EIGHT},
        FarTerrainKey{7, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{8, 0, FarTerrainStep::THIRTY_TWO},
    };
    std::array<bool, RENEWED_KEYS.size()> renewedAccepted{};
    if (firstBuildObserved) {
        for (size_t index = 0; index < RENEWED_KEYS.size(); ++index) {
            renewedAccepted[index] =
                scheduler.enqueueCanopy(RENEWED_KEYS[index], static_cast<uint32_t>(50 + index),
                                        FarTerrainAuthorityQuality::FINAL);
        }
    }
    {
        std::lock_guard lock(orderMutex);
        releaseFirstBuild = true;
    }
    orderCv.notify_all();
    REQUIRE(firstBuildObserved);
    REQUIRE(std::ranges::all_of(renewedAccepted, std::identity{}));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().canopyInFlight != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(scheduler.stats().canopyInFlight == 0);
    std::lock_guard lock(orderMutex);
    REQUIRE(buildOrder.size() == INITIAL_KEYS.size() + RENEWED_KEYS.size());
    REQUIRE(std::ranges::none_of(std::span<const int>{buildOrder.data(), 4},
                                 [](int step) { return step == 1; }));
    REQUIRE(buildOrder[4] == 1);
}

TEST_CASE("Final canopy promotions retain ordinary distance priority",
          "[render][far-terrain][scheduler][canopy][priority][final][regression]") {
    std::atomic<int> firstStep{0};
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep step) {
        int unset = 0;
        firstStep.compare_exchange_strong(unset, farTerrainStepSize(step),
                                          std::memory_order_relaxed);
        return std::vector<FarCanopy>{};
    };
    source.flora = {};

    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 2;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey FAR_STEP_ONE{0, 0, FarTerrainStep::ONE};
    constexpr FarTerrainKey NEAR_COARSE{1, 0, FarTerrainStep::EIGHT};
    REQUIRE(scheduler.enqueueCanopy(FAR_STEP_ONE, 100, FarTerrainAuthorityQuality::FINAL));
    REQUIRE(scheduler.enqueueCanopy(NEAR_COARSE, 0, FarTerrainAuthorityQuality::FINAL));
    scheduler.setCanopyWorkerBudget(1);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (firstStep.load(std::memory_order_relaxed) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(firstStep.load(std::memory_order_relaxed) == farTerrainStepSize(NEAR_COARSE.step));
}

TEST_CASE("Canopy priorities follow camera movement when wanted membership is unchanged",
          "[render][far-terrain][scheduler][canopy][priority][residency][regression]") {
    constexpr int64_t UNSET = std::numeric_limits<int64_t>::min();
    std::atomic<int64_t> firstOriginX{UNSET};
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t minimumX, int64_t, int64_t, int64_t, FarTerrainStep) {
        int64_t unset = UNSET;
        firstOriginX.compare_exchange_strong(unset, minimumX, std::memory_order_relaxed);
        return std::vector<FarCanopy>{};
    };
    source.flora = {};
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 2;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey FIRST{3, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey SECOND{4, 0, FarTerrainStep::EIGHT};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted{FIRST, SECOND};
    REQUIRE(scheduler.retainWanted(wanted, {FIRST, SECOND}));
    REQUIRE(scheduler.enqueueCanopy(FIRST, 0));
    REQUIRE(scheduler.enqueueCanopy(SECOND, 1));
    REQUIRE(scheduler.retainWanted(wanted, {SECOND, FIRST}));
    REQUIRE(scheduler.stats().wantedUpdates == 2);
    REQUIRE(scheduler.stats().queuedCanopy == 2);

    scheduler.setCanopyWorkerBudget(1);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (firstOriginX.load(std::memory_order_relaxed) == UNSET &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(firstOriginX.load(std::memory_order_relaxed) == SECOND.tileX * FAR_TERRAIN_TILE_EDGE);
}

TEST_CASE("A near canopy displaces distant parked work at the pending cap",
          "[render][far-terrain][scheduler][canopy][priority][deferred][capacity][regression]") {
    TempDir directory("near_canopy_displaces_parked");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [](int64_t, int64_t, int64_t, int64_t,
                         FarTerrainStep) -> std::vector<FarCanopy> {
        throw worldgen::learned::GenerationFailureException(
            worldgen::learned::AuthorityStatus::DEFERRED,
            {.code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
             .message = "Synthetic parked canopy authority",
             .retriable = true});
    };
    source.flora = {};
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 2;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), context, limits);

    constexpr FarTerrainKey FAR_FIRST{8, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey FAR_SECOND{9, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey NEAR{1, 0, FarTerrainStep::EIGHT};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted{FAR_FIRST, FAR_SECOND, NEAR};
    REQUIRE(scheduler.retainWanted(wanted, {FAR_FIRST, FAR_SECOND, NEAR}));
    REQUIRE(scheduler.enqueueCanopy(FAR_FIRST));
    REQUIRE(scheduler.enqueueCanopy(FAR_SECOND));

    const auto parkedDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().parkedCanopy != 2 &&
           std::chrono::steady_clock::now() < parkedDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().parkedCanopy == 2);
    REQUIRE(scheduler.stats().activeCanopyWorkers == 0);
    scheduler.setCanopyWorkerBudget(0);

    REQUIRE(scheduler.retainWanted(wanted, {NEAR, FAR_SECOND, FAR_FIRST}));
    REQUIRE(scheduler.enqueueCanopy(NEAR));
    const FarTerrainSchedulerStats replaced = scheduler.stats();
    REQUIRE(replaced.canopySubmitted == 3);
    REQUIRE(replaced.canopyCanceled == 1);
    REQUIRE(replaced.canopyInFlight == 2);
    REQUIRE(replaced.queuedCanopy == 1);
    REQUIRE(replaced.parkedCanopy == 1);

    // FAR_FIRST became the least-important parked job only after the view
    // reorder, so it must be the displaced key. Making it nearest now admits
    // it as new work; a stale parked priority would have left it active.
    REQUIRE(scheduler.retainWanted(wanted, {FAR_FIRST, FAR_SECOND, NEAR}));
    REQUIRE(scheduler.enqueueCanopy(FAR_FIRST));
    const FarTerrainSchedulerStats reprioritized = scheduler.stats();
    REQUIRE(reprioritized.canopySubmitted == 4);
    REQUIRE(reprioritized.canopyCanceled == 2);
    REQUIRE(reprioritized.canopyInFlight == 2);
    REQUIRE(reprioritized.queuedCanopy == 1);
    REQUIRE(reprioritized.parkedCanopy == 1);

    scheduler.advanceEpoch();
    const FarTerrainSchedulerStats cleared = scheduler.stats();
    REQUIRE(cleared.canopyInFlight == 0);
    REQUIRE(cleared.queuedCanopy == 0);
    REQUIRE(cleared.parkedCanopy == 0);
    REQUIRE(cleared.canopyCanceled == 4);
    scheduler.shutdown();
}

TEST_CASE("Far terrain publishes terrain and water before canopy collection finishes",
          "[render][far-terrain][scheduler][canopy][staging][regression]") {
    std::mutex canopyMutex;
    std::condition_variable canopyCv;
    size_t canopyEntered = 0;
    bool releaseCanopy = false;
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        std::unique_lock lock(canopyMutex);
        ++canopyEntered;
        canopyCv.notify_all();
        canopyCv.wait(lock, [&] { return releaseCanopy; });
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCanopyPending = 2;
    limits.maxCompleted = 4;
    limits.maxCacheEntries = 2;
    limits.maxCacheBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    struct CanopyRelease {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& release;
        ~CanopyRelease() {
            {
                std::lock_guard lock(mutex);
                release = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{canopyMutex, canopyCv, releaseCanopy};

    constexpr FarTerrainKey FIRST{7, -4, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey SECOND{8, -4, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueue(FIRST));
    std::vector<FarTerrainResult> completed;
    const auto surfaceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.empty() && std::chrono::steady_clock::now() < surfaceDeadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().mesh);
    const std::shared_ptr<const FarTerrainMesh> surface = scheduler.findCached(FIRST);
    REQUIRE(surface);
    REQUIRE_FALSE(scheduler.findCachedCanopy(FIRST));
    REQUIRE(scheduler.stats().canopySubmitted == 0);
    REQUIRE(scheduler.enqueueCanopy(FIRST, 0, FarTerrainAuthorityQuality::FINAL));
    {
        std::unique_lock lock(canopyMutex);
        REQUIRE(
            canopyCv.wait_for(lock, std::chrono::seconds(2), [&] { return canopyEntered >= 1; }));
        REQUIRE(scheduler.stats().inFlight == 0);
        REQUIRE(scheduler.stats().canopyInFlight == 1);
        REQUIRE(scheduler.hasSubmissionCapacity());
    }

    REQUIRE(scheduler.enqueue(SECOND));
    completed.clear();
    const auto secondSurfaceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.empty() && std::chrono::steady_clock::now() < secondSurfaceDeadline) {
        scheduler.drainCompleted(completed);
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(completed.size() == 1);
    REQUIRE(completed.front().key == SECOND);
    REQUIRE(completed.front().mesh);
    REQUIRE(scheduler.findCached(SECOND));
    REQUIRE_FALSE(scheduler.findCachedCanopy(SECOND));
    REQUIRE(scheduler.enqueueCanopy(SECOND, 1, FarTerrainAuthorityQuality::FINAL));
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.hasSubmissionCapacity());
    {
        std::unique_lock lock(canopyMutex);
        REQUIRE(
            canopyCv.wait_for(lock, std::chrono::seconds(2), [&] { return canopyEntered >= 2; }));
        releaseCanopy = true;
    }
    canopyCv.notify_all();

    std::vector<FarCanopyResult> canopyCompleted;
    const auto completeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (canopyCompleted.size() < 2 && std::chrono::steady_clock::now() < completeDeadline) {
        scheduler.drainCanopyCompleted(canopyCompleted);
        if (canopyCompleted.size() < 2)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    scheduler.shutdown();
    REQUIRE(canopyCompleted.size() == 2);
    REQUIRE(std::ranges::all_of(canopyCompleted, [](const FarCanopyResult& result) {
        return !result.failed && result.attachment && result.attachment->vertices.empty() &&
               result.attachment->indices.empty();
    }));
    REQUIRE(scheduler.findCachedCanopy(FIRST));
    REQUIRE(scheduler.findCachedCanopy(SECOND));
    REQUIRE(scheduler.stats().built == 2);
    REQUIRE(scheduler.stats().canopyBuilt == 2);
}

TEST_CASE("Far canopy cache records explicit empty completion without a flora source",
          "[render][far-terrain][scheduler][canopy][cache][empty][regression]") {
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = {};
    FarTerrainScheduler scheduler(std::move(source));
    constexpr FarTerrainKey KEY{-4, 11, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueue(KEY));

    const auto surfaceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!scheduler.findCached(KEY) && std::chrono::steady_clock::now() < surfaceDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(scheduler.findCached(KEY));
    REQUIRE(scheduler.stats().canopySubmitted == 0);
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL));

    std::shared_ptr<const FarCanopyAttachment> attachment;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!attachment && std::chrono::steady_clock::now() < deadline) {
        attachment = scheduler.findCachedCanopy(KEY);
        if (!attachment)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(attachment);
    REQUIRE(attachment->vertices.empty());
    REQUIRE(attachment->indices.empty());
    REQUIRE(scheduler.stats().canopyBuilt == 1);
    REQUIRE(scheduler.findCachedCanopy(KEY) == attachment);
    REQUIRE_FALSE(scheduler.enqueueCanopy(KEY));
    scheduler.shutdown();
}

TEST_CASE("Nearest preview vegetation publishes before final ecology promotion",
          "[render][far-terrain][scheduler][canopy][preview][priority][regression]") {
    TempDir directory("far_canopy_preview_first");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 3;
    limits.maxCanopyCompleted = 8;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey FAR{8, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey MIDDLE{4, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueueCanopy(FAR, 800, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(scheduler.enqueueCanopy(MIDDLE, 400, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(scheduler.enqueueCanopy(NEAR, 0, FarTerrainAuthorityQuality::PREVIEW));
    scheduler.setCanopyWorkerBudget(1);

    std::vector<FarCanopyResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.size() < 4 && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCanopyCompleted(completed);
        if (completed.size() < 4)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(completed.size() >= 4);
    REQUIRE(completed[0].key == NEAR);
    REQUIRE(completed[1].key == MIDDLE);
    REQUIRE(completed[2].key == FAR);
    REQUIRE(completed[0].attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[1].attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[2].attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[3].key == NEAR);
    REQUIRE(completed[3].attachment->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(scheduler.stats().canopySubmitted == 6);
    REQUIRE(scheduler.stats().canopyBuilt == 6);
    REQUIRE(scheduler.stats().canopyInFlight == 0);
}

TEST_CASE("Preview flora drains before final promotion and keeps one movement lane",
          "[render][far-terrain][scheduler][canopy][preview][phase][movement][regression]") {
    TempDir directory("far_canopy_preview_phase_barrier");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);

    constexpr FarTerrainKey FAST{0, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey SLOW{1, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey NEW_VISIBLE{2, 0, FarTerrainStep::SIXTEEN};
    std::mutex gateMutex;
    std::condition_variable gateCv;
    std::unordered_map<int64_t, uint32_t> calls;
    bool slowPreviewEntered = false;
    bool newPreviewEntered = false;
    bool releaseSlowPreview = false;
    bool releaseFinal = false;
    size_t finalEntered = 0;
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [&](int64_t minimumX, int64_t, int64_t, int64_t, FarTerrainStep) {
        const int64_t tileX = world_coord::floorDiv(minimumX, int64_t{FAR_TERRAIN_TILE_EDGE});
        std::unique_lock lock(gateMutex);
        const uint32_t ordinal = ++calls[tileX];
        if (tileX == SLOW.tileX && ordinal == 1) {
            slowPreviewEntered = true;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return releaseSlowPreview; });
        } else if (tileX == NEW_VISIBLE.tileX && ordinal == 1) {
            newPreviewEntered = true;
            gateCv.notify_all();
        } else if (ordinal >= 2) {
            ++finalEntered;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return releaseFinal; });
        }
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 3;
    limits.maxCanopyCompleted = 8;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    struct ReleaseOnExit {
        std::mutex& mutex;
        std::condition_variable& condition;
        bool& slow;
        bool& final;
        ~ReleaseOnExit() {
            {
                std::lock_guard lock(mutex);
                slow = true;
                final = true;
            }
            condition.notify_all();
        }
    } releaseOnExit{gateMutex, gateCv, releaseSlowPreview, releaseFinal};
    scheduler.setCanopyWorkerBudget(0);
    REQUIRE(scheduler.enqueueCanopy(FAST, 0, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE(scheduler.enqueueCanopy(SLOW, 1, FarTerrainAuthorityQuality::PREVIEW));
    scheduler.setCanopyWorkerBudget(FarTerrainScheduler::CANOPY_WORKER_COUNT);

    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return slowPreviewEntered; }));
        REQUIRE_FALSE(gateCv.wait_for(lock, std::chrono::milliseconds(100),
                                      [&] { return finalEntered != 0; }));
        releaseSlowPreview = true;
    }
    gateCv.notify_all();
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return finalEntered == 1; }));
    }

    REQUIRE(scheduler.enqueueCanopy(NEW_VISIBLE, 0, FarTerrainAuthorityQuality::PREVIEW));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return newPreviewEntered; }));
        REQUIRE(finalEntered == 1);
    }

    std::vector<FarCanopyResult> provisional;
    const auto previewDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (provisional.size() < 3 && std::chrono::steady_clock::now() < previewDeadline) {
        scheduler.drainCanopyCompleted(provisional);
        if (provisional.size() < 3)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(provisional.size() == 3);
    REQUIRE(std::ranges::all_of(provisional, [](const FarCanopyResult& result) {
        return result.attachment &&
               result.attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW;
    }));
    {
        std::lock_guard lock(gateMutex);
        releaseFinal = true;
    }
    gateCv.notify_all();

    const auto completionDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().canopyInFlight != 0 &&
           std::chrono::steady_clock::now() < completionDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(scheduler.stats().canopyInFlight == 0);
    REQUIRE(finalEntered == 3);
}

TEST_CASE("Surface promotion publishes its provisional flora before final ecology",
          "[render][far-terrain][scheduler][canopy][preview][grounding][promotion][regression]") {
    TempDir directory("far_canopy_preview_grounding_promotion");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSchedulerLimits limits;
    limits.maxCanopyPending = 1;
    limits.maxCanopyCompleted = 4;
    FarTerrainScheduler scheduler(farTerrainTestSource(), context, limits);
    scheduler.setCanopyWorkerBudget(0);

    constexpr FarTerrainKey KEY{2, -5, FarTerrainStep::EIGHT};
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::PREVIEW));
    REQUIRE_FALSE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL));
    scheduler.setCanopyWorkerBudget(1);

    std::vector<FarCanopyResult> completed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.size() < 3 && std::chrono::steady_clock::now() < deadline) {
        scheduler.drainCanopyCompleted(completed);
        if (completed.size() < 3)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();

    REQUIRE(completed.size() == 3);
    REQUIRE(completed[0].attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[0].attachment->groundingQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[1].attachment->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(completed[1].attachment->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(completed[2].attachment->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(completed[2].attachment->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(scheduler.stats().canopySubmitted == 3);
    REQUIRE(scheduler.stats().canopyBuilt == 3);
}

TEST_CASE("Preview vegetation publishes while final ecology remains deferred",
          "[render][far-terrain][scheduler][canopy][deferred][preview][promotion][regression]") {
    TempDir directory("far_canopy_authority_resume");
    const worldgen::learned::GenerationIdentity identity = coarsePrefetchTestIdentity();
    auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>());
    auto context = std::make_shared<worldgen::learned::WorldGenerationContext>(
        identity, authority, worldgen::learned::AuthorityQuality::FINAL);
    FarTerrainSource source = farTerrainTestSource();
    std::atomic<uint32_t> ecologyBuilds{0};
    source.canopies = [&](int64_t, int64_t, int64_t, int64_t, FarTerrainStep) {
        // PREVIEW publishes on the first build. The automatically queued FINAL
        // phase then defers until the synthetic authority completion below.
        if (ecologyBuilds.fetch_add(1, std::memory_order_relaxed) == 1) {
            throw worldgen::learned::GenerationFailureException(
                worldgen::learned::AuthorityStatus::DEFERRED,
                {.code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
                 .message = "Synthetic cold canopy authority page",
                 .retriable = true});
        }
        return std::vector<FarCanopy>{};
    };
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 1;
    limits.maxCanopyPending = 1;
    limits.maxCompleted = 2;
    limits.maxCanopyCompleted = 2;
    FarTerrainScheduler scheduler(std::move(source), context, limits);
    constexpr FarTerrainKey KEY{10, 3, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueueFinalRefinement(KEY));

    std::vector<FarTerrainResult> surfaces;
    const auto surfaceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (surfaces.empty() && std::chrono::steady_clock::now() < surfaceDeadline) {
        scheduler.drainCompleted(surfaces);
        if (surfaces.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(surfaces.size() == 1);
    REQUIRE(surfaces.front().mesh);
    const uint64_t surfaceHash = surfaces.front().mesh->deterministicHash;
    REQUIRE(scheduler.stats().canopySubmitted == 0);
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL));
    const auto deferredDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.stats().canopyDeferred == 0 &&
           std::chrono::steady_clock::now() < deferredDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(scheduler.stats().canopyDeferred == 1);
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().canopyInFlight == 1);
    REQUIRE(scheduler.stats().parkedCanopy == 1);

    std::vector<FarCanopyResult> provisional;
    const auto provisionalDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (provisional.empty() && std::chrono::steady_clock::now() < provisionalDeadline) {
        scheduler.drainCanopyCompleted(provisional);
        if (provisional.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(provisional.size() == 1);
    REQUIRE_FALSE(provisional.front().failed);
    REQUIRE(provisional.front().attachment);
    REQUIRE(provisional.front().attachment->authorityQuality ==
            FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(provisional.front().attachment->groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE(farCanopyMatchesSurface(provisional.front().attachment->authorityQuality,
                                    provisional.front().attachment->groundingQuality,
                                    FarTerrainAuthorityQuality::FINAL));
    REQUIRE(scheduler.findCachedCanopy(KEY)->authorityQuality ==
            FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(scheduler.stats().parkedCanopy == 1);

    constexpr worldgen::learned::TerrainPageCoordinate COMPLETION_PAGE{.row = 37, .column = -19};
    auto prepared = context->requestAuthorityPage(
        COMPLETION_PAGE, worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    const auto authorityDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (prepared.status() == worldgen::learned::AuthorityStatus::DEFERRED &&
           std::chrono::steady_clock::now() < authorityDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        prepared = context->requestAuthorityPage(
            COMPLETION_PAGE, worldgen::learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    }
    REQUIRE(prepared.isReady());

    std::vector<FarCanopyResult> finalCanopies;
    const auto canopyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (finalCanopies.empty() && std::chrono::steady_clock::now() < canopyDeadline) {
        scheduler.pumpCanopyAuthority();
        scheduler.drainCanopyCompleted(finalCanopies);
        if (finalCanopies.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(finalCanopies.size() == 1);
    REQUIRE_FALSE(finalCanopies.front().failed);
    REQUIRE(finalCanopies.front().attachment);
    REQUIRE(finalCanopies.front().attachment->authorityQuality ==
            FarTerrainAuthorityQuality::FINAL);
    REQUIRE(finalCanopies.front().attachment->groundingQuality ==
            FarTerrainAuthorityQuality::FINAL);
    REQUIRE(scheduler.stats().built == 1);
    REQUIRE(scheduler.stats().canopyBuilt == 2);
    REQUIRE(scheduler.stats().canopyAuthorityCompletionResumes == 1);
    REQUIRE(scheduler.stats().parkedCanopy == 0);
    REQUIRE(scheduler.findCached(KEY)->deterministicHash == surfaceHash);
    REQUIRE(scheduler.findCachedCanopy(KEY)->authorityQuality == FarTerrainAuthorityQuality::FINAL);
}

TEST_CASE("Far canopy failures report without invalidating resident terrain and water",
          "[render][far-terrain][scheduler][canopy][failure][regression]") {
    FarTerrainSource source = farTerrainTestSource();
    source.canopies = [](int64_t, int64_t, int64_t, int64_t,
                         FarTerrainStep) -> std::vector<FarCanopy> {
        throw std::runtime_error("Synthetic canopy failure");
    };
    FarTerrainScheduler scheduler(std::move(source));
    constexpr FarTerrainKey KEY{-9, 6, FarTerrainStep::SIXTEEN};
    REQUIRE(scheduler.enqueue(KEY));

    std::vector<FarTerrainResult> surfaces;
    const auto surfaceDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (surfaces.empty() && std::chrono::steady_clock::now() < surfaceDeadline) {
        scheduler.drainCompleted(surfaces);
        if (surfaces.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(surfaces.size() == 1);
    REQUIRE_FALSE(surfaces.front().failed);
    REQUIRE(surfaces.front().mesh);
    REQUIRE(scheduler.findCached(KEY));
    REQUIRE(scheduler.stats().canopySubmitted == 0);
    REQUIRE(scheduler.enqueueCanopy(KEY, 0, FarTerrainAuthorityQuality::FINAL));

    std::vector<FarCanopyResult> canopies;
    const auto canopyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (canopies.empty() && std::chrono::steady_clock::now() < canopyDeadline) {
        scheduler.drainCanopyCompleted(canopies);
        if (canopies.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    scheduler.shutdown();
    REQUIRE(canopies.size() == 1);
    REQUIRE(canopies.front().failed);
    REQUIRE_FALSE(canopies.front().attachment);
    REQUIRE(scheduler.stats().failed == 0);
    REQUIRE(scheduler.stats().canopyFailed == 1);
}

TEST_CASE("Far terrain canopy refresh follows displayed surfaces nearest first",
          "[render][far-terrain][canopy][staging][cache][priority][regression]") {
    constexpr FarTerrainKey NEAR{0, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey COMPLETE{1, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey FAR{2, 0, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey TRANSITION_SOURCE{3, 0, FarTerrainStep::SIXTEEN};
    constexpr FarTerrainKey TRANSITION_TARGET{3, 0, FarTerrainStep::EIGHT};
    constexpr FarTerrainKey HIDDEN_PARENT{-1, 0, FarTerrainStep::THIRTY_TWO};
    constexpr FarTerrainKey PREVIEW{0, 1, FarTerrainStep::EIGHT};
    std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash> residents;
    std::unordered_map<FarTerrainKey, FarCanopyMeshState, FarTerrainKeyHash> attachments;
    auto addFinalResident = [&](FarTerrainKey key) {
        FarTerrainMeshState state{};
        state.uploaded = true;
        state.authorityQuality = FarTerrainAuthorityQuality::FINAL;
        residents.emplace(key, state);
    };
    for (FarTerrainKey key :
         {NEAR, COMPLETE, FAR, TRANSITION_SOURCE, TRANSITION_TARGET, HIDDEN_PARENT}) {
        addFinalResident(key);
    }
    FarTerrainMeshState preview{};
    preview.uploaded = true;
    preview.authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    residents.emplace(PREVIEW, preview);

    // A null allocation is a valid empty FINAL attachment and must not be
    // mistaken for missing flora.
    attachments.emplace(COMPLETE, FarCanopyMeshState{});
    attachments.emplace(TRANSITION_SOURCE, FarCanopyMeshState{});

    const std::unordered_map<ColumnPos, FarTerrainKey> displayed = {
        {{NEAR.tileX, NEAR.tileZ}, NEAR},
        {{COMPLETE.tileX, COMPLETE.tileZ}, COMPLETE},
        {{FAR.tileX, FAR.tileZ}, FAR},
        {{TRANSITION_SOURCE.tileX, TRANSITION_SOURCE.tileZ}, TRANSITION_SOURCE},
        {{PREVIEW.tileX, PREVIEW.tileZ}, PREVIEW},
    };
    const std::unordered_map<ColumnPos, FarTerrainLodTransition> transitions = {
        {{TRANSITION_SOURCE.tileX, TRANSITION_SOURCE.tileZ},
         {TRANSITION_SOURCE, TRANSITION_TARGET, 1.0}},
    };

    std::vector<FarTerrainCanopyRefreshRequest> requests;
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 128.0, 128.0,
                                      8, requests);
    REQUIRE(requests.size() == 4);
    REQUIRE(requests[0].key == NEAR);
    REQUIRE(requests[0].viewPriority == 0);
    REQUIRE(requests[0].groundingQuality == FarTerrainAuthorityQuality::FINAL);
    REQUIRE_FALSE(requests[0].transitionTarget);
    // COMPLETE has a valid empty attachment, so it never enters the request
    // batch. Priorities remain absolute block distances instead of closing up.
    REQUIRE(requests[1].key == PREVIEW);
    REQUIRE(requests[1].viewPriority == 128);
    REQUIRE(requests[1].groundingQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(requests[2].key == FAR);
    REQUIRE(requests[2].viewPriority == 384);
    REQUIRE(requests[3].key == TRANSITION_TARGET);
    REQUIRE(requests[3].viewPriority == 640);
    REQUIRE(requests[3].transitionTarget);
    REQUIRE(std::ranges::none_of(requests, [&](const auto& request) {
        return request.key == HIDDEN_PARENT || request.key == TRANSITION_SOURCE;
    }));

    FarCanopyMeshState provisionalNear{};
    provisionalNear.authorityQuality = FarTerrainAuthorityQuality::PREVIEW;
    provisionalNear.groundingQuality = FarTerrainAuthorityQuality::FINAL;
    attachments.emplace(NEAR, provisionalNear);
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 128.0, 128.0,
                                      2, requests);
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].key == PREVIEW);
    REQUIRE(requests[1].key == FAR);
    REQUIRE(
        std::ranges::none_of(requests, [&](const auto& request) { return request.key == NEAR; }));
    attachments.erase(NEAR);

    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 128.0, 128.0,
                                      2, requests);
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].key == NEAR);
    REQUIRE(requests[1].key == PREVIEW);

    std::vector<ChunkPos> exactFloraSections;
    exactFloraSections.reserve(FAR_TERRAIN_EXACT_COLUMNS_PER_TILE *
                               FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
    for (int64_t z = 0; z < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++z) {
        for (int64_t x = 0; x < FAR_TERRAIN_EXACT_COLUMNS_PER_TILE; ++x)
            exactFloraSections.push_back({x, 4, z});
    }
    const FarTerrainExactHandoff exactFlora = farTerrainExactHandoff(
        128.0, 128.0, 32, exactFloraSections, {}, [](ChunkPos) { return true; });
    REQUIRE(exactFlora.tileFullyOwned({NEAR.tileX, NEAR.tileZ}));
    buildFarTerrainCanopyRefreshBatch(displayed, transitions, residents, attachments, 128.0, 128.0,
                                      2, requests, &exactFlora);
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].key == PREVIEW);
    REQUIRE(requests[1].key == FAR);
}

TEST_CASE("Far terrain worker budgets yield to exact and local publication debt",
          "[render][far-terrain][scheduler][canopy][priority][startup][regression]") {
    STATIC_REQUIRE(farTerrainWorkerBudget(false, false) == FarTerrainScheduler::WORKER_COUNT);
    STATIC_REQUIRE(farTerrainWorkerBudget(true, false) == 0);
    STATIC_REQUIRE(farTerrainWorkerBudget(false, true) == FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET);
    STATIC_REQUIRE(farTerrainWorkerBudget(true, true) == FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET);
    STATIC_REQUIRE(farTerrainOrdinaryCoverageWorkEnabled(false, true, true));
    STATIC_REQUIRE_FALSE(farTerrainOrdinaryCoverageWorkEnabled(true, true, false));
    STATIC_REQUIRE_FALSE(farTerrainOrdinaryCoverageWorkEnabled(true, false, true));
    STATIC_REQUIRE(farTerrainOrdinaryCoverageWorkEnabled(true, false, false));
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET < FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET);
    STATIC_REQUIRE(FAR_TERRAIN_LOCAL_DEBT_WORKER_BUDGET < FarTerrainScheduler::WORKER_COUNT);
    STATIC_REQUIRE(farTerrainUrgentWorkerLimit(FAR_TERRAIN_EXACT_DEBT_WORKER_BUDGET, true) >= 4);

    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(false, false, true, true) == 0);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(false, true, false, false) == 0);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, false, true, true) == 0);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, true, true, false) == 1);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, true, true, true) == 1);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, false, false, false) == 1);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, true, false, false) ==
                   FarTerrainScheduler::CANOPY_WORKER_COUNT);
    STATIC_REQUIRE(farTerrainCanopyWorkerBudget(true, true, false, true) == 1);

    constexpr float FLORA_RADIUS_BLOCKS = EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS * CHUNK_EDGE;
    STATIC_REQUIRE(farTerrainCanopyHasNearExactSurfaceDebt(
        true, 0.0F, EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE(farTerrainCanopyHasNearExactSurfaceDebt(
        true, FLORA_RADIUS_BLOCKS, EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE_FALSE(farTerrainCanopyHasNearExactSurfaceDebt(
        true, FLORA_RADIUS_BLOCKS + 1.0F, EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE(
        farTerrainCanopyHasNearExactSurfaceDebt(true, std::numeric_limits<float>::quiet_NaN(),
                                                EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE_FALSE(farTerrainCanopyHasNearExactSurfaceDebt(
        false, 0.0F, EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE(farTerrainCanopyHasNearExactPublicationDebt(
        true, FLORA_RADIUS_BLOCKS + 1.0F, FLORA_RADIUS_BLOCKS,
        EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
    STATIC_REQUIRE_FALSE(farTerrainCanopyHasNearExactPublicationDebt(
        true, FLORA_RADIUS_BLOCKS + 1.0F, FLORA_RADIUS_BLOCKS + 1.0F,
        EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS));
}

TEST_CASE("Protected far terrain steps cap coarseness without overriding finer screen error",
          "[render][far-terrain][lod][protected][screen-error][regression]") {
    STATIC_REQUIRE(farTerrainProtectedDesiredStep(FarTerrainStep::ONE, FarTerrainStep::EIGHT) ==
                   FarTerrainStep::ONE);
    STATIC_REQUIRE(farTerrainProtectedDesiredStep(FarTerrainStep::SIXTEEN, FarTerrainStep::FOUR) ==
                   FarTerrainStep::FOUR);
    STATIC_REQUIRE(farTerrainProtectedDesiredStep(FarTerrainStep::FOUR, FarTerrainStep::FOUR) ==
                   FarTerrainStep::FOUR);
    STATIC_REQUIRE(farTerrainProtectedDesiredStep(std::nullopt, FarTerrainStep::TWO) ==
                   FarTerrainStep::TWO);
    STATIC_REQUIRE(farTerrainProtectedDesiredStep(FarTerrainStep::TWO, std::nullopt) ==
                   FarTerrainStep::TWO);
}

TEST_CASE("Protected FINAL children cannot deadlock behind an incompatible preview bridge",
          "[render][far-terrain][lod][protected][authority][entry][regression]") {
    STATIC_REQUIRE_FALSE(farTerrainProtectedFinalTargetMaySubmit(false, false, false));
    STATIC_REQUIRE(farTerrainProtectedFinalTargetMaySubmit(false, true, false));
    STATIC_REQUIRE(farTerrainProtectedFinalTargetMaySubmit(false, false, true));
    STATIC_REQUIRE_FALSE(farTerrainProtectedFinalTargetMaySubmit(true, true, true));
}

TEST_CASE("Rejected preparation anchors cannot publish old far terrain",
          "[render][far-terrain][scheduler][cancellation][startup][regression]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool entered = false;
    bool released = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (!entered) {
                entered = true;
                gateCv.notify_all();
                gateCv.wait(lock, [&] { return released; });
            }
        }
        return sample(x, z, footprint);
    };
    FarTerrainScheduler scheduler(source);
    REQUIRE(scheduler.enqueue({0, 0, FarTerrainStep::SIXTEEN}));
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gateCv.wait_for(lock, std::chrono::seconds(2), [&] { return entered; }));
    }
    const uint64_t newEpoch = scheduler.cancelViewPreparation();
    {
        std::lock_guard lock(gateMutex);
        released = true;
    }
    gateCv.notify_all();
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::vector<FarTerrainResult> completed;
    scheduler.drainCompleted(completed);
    REQUIRE(completed.empty());
    REQUIRE(scheduler.currentEpoch() == newEpoch);
    REQUIRE(scheduler.stats().canceled >= 1);
}

TEST_CASE("Far terrain scheduler retains only the current view",
          "[render][far-terrain][scheduler][cancellation][cache]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    const std::array<FarTerrainKey, 4> keys{{
        {0, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::SIXTEEN},
        {2, 0, FarTerrainStep::SIXTEEN},
        {3, 0, FarTerrainStep::SIXTEEN},
    }};
    for (const FarTerrainKey& key : keys)
        REQUIRE(scheduler.enqueue(key));
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == keys.size());

    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted{keys[1], keys[3]};
    scheduler.retainWanted(wanted);
    for (int attempt = 0; attempt < 400; ++attempt) {
        const FarTerrainSchedulerStats current = scheduler.stats();
        if (current.maintenancePending == 0 && current.cacheEntries == wanted.size())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const FarTerrainSchedulerStats retained = scheduler.stats();
    REQUIRE(retained.maintenancePending == 0);
    REQUIRE(retained.cacheEntries == wanted.size());
    REQUIRE(retained.completed == wanted.size());
    REQUIRE(scheduler.findCached(keys[1]));
    REQUIRE(scheduler.findCached(keys[3]));
    REQUIRE_FALSE(scheduler.findCached(keys[0]));
    REQUIRE_FALSE(scheduler.findCached(keys[2]));
    REQUIRE_FALSE(scheduler.enqueue(keys[0]));
}

TEST_CASE("Far terrain scheduler reuses stable wanted state",
          "[render][far-terrain][scheduler][residency][performance]") {
    FarTerrainScheduler scheduler(farTerrainTestSource());
    const std::vector<FarTerrainKey> order{
        {0, 0, FarTerrainStep::SIXTEEN},
        {1, 0, FarTerrainStep::SIXTEEN},
        {0, 0, FarTerrainStep::FOUR},
    };
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());

    REQUIRE(scheduler.retainWanted(wanted, order));
    const FarTerrainSchedulerStats first = scheduler.stats();
    REQUIRE(first.wantedUpdates == 1);
    REQUIRE(first.wantedNoops == 0);

    REQUIRE_FALSE(scheduler.retainWanted(wanted, order));
    const FarTerrainSchedulerStats second = scheduler.stats();
    REQUIRE(second.wantedUpdates == first.wantedUpdates);
    REQUIRE(second.wantedNoops == first.wantedNoops + 1);

    std::vector<FarTerrainKey> reprioritized = order;
    std::swap(reprioritized[0], reprioritized[1]);
    REQUIRE(scheduler.retainWanted(wanted, reprioritized));
    const FarTerrainSchedulerStats reprioritizedStats = scheduler.stats();
    REQUIRE(reprioritizedStats.wantedUpdates == first.wantedUpdates + 1);
    REQUIRE(reprioritizedStats.wantedNoops == second.wantedNoops);
}

TEST_CASE("Far terrain cache residency retires obsolete meshes in bounded worker passes",
          "[render][far-terrain][scheduler][cache][performance][concurrency][regression]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    limits.maxMaintenanceEntries = 1;
    limits.maxMaintenanceBytes = 16 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    std::array<FarTerrainKey, 6> keys{};
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = {static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN};
        REQUIRE(scheduler.enqueue(keys[index]));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().cacheEntries == keys.size());

    const std::vector<FarTerrainKey> order{keys[1], keys[4]};
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(order.begin(), order.end());
    REQUIRE(scheduler.retainWanted(wanted, order));
    for (int attempt = 0; attempt < 1000; ++attempt) {
        const FarTerrainSchedulerStats current = scheduler.stats();
        if (current.maintenancePending == 0 && current.cacheEntries == wanted.size())
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const FarTerrainSchedulerStats maintained = scheduler.stats();
    REQUIRE(maintained.maintenancePending == 0);
    REQUIRE(maintained.cacheEntries == wanted.size());
    REQUIRE(maintained.maintenanceEvicted == keys.size() - wanted.size());
    REQUIRE(maintained.maintenanceScanned >= keys.size());
    REQUIRE(maintained.maintenancePasses >= keys.size());
    REQUIRE(maintained.maximumMaintenanceScanned <= limits.maxMaintenanceEntries);
    REQUIRE(maintained.maximumMaintenanceBytes <= limits.maxMaintenanceBytes);
    REQUIRE(scheduler.findCached(keys[1]));
    REQUIRE(scheduler.findCached(keys[4]));
}

TEST_CASE("Far terrain cache batches preserve nearest useful refinement selection",
          "[render][far-terrain][scheduler][cache][batch][performance]") {
    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(farTerrainTestSource(), limits);
    constexpr std::array keys{
        FarTerrainKey{0, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{0, 0, FarTerrainStep::EIGHT},
        FarTerrainKey{0, 0, FarTerrainStep::TWO},
        FarTerrainKey{1, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN},
        FarTerrainKey{1, 0, FarTerrainStep::FOUR},
    };
    for (const FarTerrainKey key : keys)
        REQUIRE(scheduler.enqueue(key));
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == keys.size());
    REQUIRE(scheduler.stats().cacheBaseEntries == 2);

    const std::array baseRequests{
        FarTerrainKey{-1, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{0, 0, FarTerrainStep::THIRTY_TWO},
        FarTerrainKey{1, 0, FarTerrainStep::THIRTY_TWO},
    };
    std::vector<std::shared_ptr<const FarTerrainMesh>> batch;
    scheduler.findCachedBatch(baseRequests, 1, batch);
    REQUIRE(batch.size() == 1);
    REQUIRE(batch.front()->key == baseRequests[1]);

    constexpr FarTerrainStepMask BASE_RESIDENT = farTerrainStepMask(FarTerrainStep::THIRTY_TWO);
    const std::array refinementRequests{
        FarTerrainRefinementCacheRequest{
            {0, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, false},
        FarTerrainRefinementCacheRequest{
            {1, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, false},
        FarTerrainRefinementCacheRequest{
            {2, 0}, FarTerrainStep::THIRTY_TWO, FarTerrainStep::TWO, BASE_RESIDENT, true},
    };
    auto deferredRequests = refinementRequests;
    deferredRequests[0].deferIntermediate = true;
    deferredRequests[1].deferIntermediate = true;
    scheduler.findFinestCachedBatch(deferredRequests, 2, batch);
    REQUIRE(batch.empty());

    scheduler.findFinestCachedBatch(refinementRequests, 2, batch);
    REQUIRE(batch.size() == 2);
    REQUIRE((batch[0]->key == FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}));
    REQUIRE((batch[1]->key == FarTerrainKey{1, 0, FarTerrainStep::SIXTEEN}));
    REQUIRE(batch[0] == scheduler.findFinestCached({0, 0}, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_RESIDENT));
    REQUIRE(batch[1] == scheduler.findFinestCached({1, 0}, FarTerrainStep::THIRTY_TWO,
                                                   FarTerrainStep::TWO, BASE_RESIDENT));

    // A fine target can survive a camera move in GPU residency while its
    // display state is reconstructed from a step-32 parent. The cache lookup
    // must still return the adjacent step-16 bridge instead of repeatedly
    // selecting the already resident but topologically unusable step-2 mesh.
    auto strandedFineTarget = refinementRequests[0];
    strandedFineTarget.residentSteps |= farTerrainStepMask(FarTerrainStep::TWO);
    scheduler.findFinestCachedBatch(std::span(&strandedFineTarget, 1), 1, batch);
    REQUIRE(batch.size() == 1);
    REQUIRE((batch.front()->key == FarTerrainKey{0, 0, FarTerrainStep::SIXTEEN}));
}

TEST_CASE("Far terrain submission scans pass cache hits and stop at capacity",
          "[render][far-terrain][scheduler][capacity][cache][performance]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    bool blockBuilds = false;
    bool releaseBuilds = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            if (blockBuilds)
                gateCv.wait(lock, [&] { return releaseBuilds; });
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 32;
    limits.maxCacheEntries = 32;
    limits.maxCacheBytes = 64 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    FarTerrainTestGateRelease releaseOnExit{gateMutex, gateCv, releaseBuilds};

    std::vector<FarTerrainKey> scanOrder;
    for (int64_t x = 0; x < 4; ++x) {
        scanOrder.push_back({x, 0, FarTerrainStep::SIXTEEN});
        REQUIRE(scheduler.enqueue(scanOrder.back(), static_cast<uint32_t>(x)));
    }
    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(scheduler.stats().inFlight == 0);
    REQUIRE(scheduler.stats().cacheEntries == 4);

    for (int64_t x = 4; x < 13; ++x)
        scanOrder.push_back({x, 0, FarTerrainStep::SIXTEEN});
    const std::unordered_set<FarTerrainKey, FarTerrainKeyHash> wanted(scanOrder.begin(),
                                                                      scanOrder.end());
    REQUIRE(scheduler.retainWanted(wanted, scanOrder));
    {
        std::lock_guard lock(gateMutex);
        blockBuilds = true;
    }

    size_t scanned = 0;
    size_t submitted = 0;
    for (size_t index = 0; index < scanOrder.size(); ++index) {
        if (!scheduler.hasSubmissionCapacity())
            break;
        ++scanned;
        if (scheduler.enqueue(scanOrder[index], static_cast<uint32_t>(index)))
            ++submitted;
    }
    REQUIRE(scanned == 4 + farTerrainNonurgentBaseAdmissionLimit(limits.maxPending));
    REQUIRE(submitted == farTerrainNonurgentBaseAdmissionLimit(limits.maxPending));
    REQUIRE_FALSE(scheduler.hasSubmissionCapacity());
    REQUIRE(scheduler.stats().inFlight == farTerrainNonurgentBaseAdmissionLimit(limits.maxPending));

    {
        std::lock_guard lock(gateMutex);
        releaseBuilds = true;
    }
    gateCv.notify_all();
    scheduler.shutdown();
}

TEST_CASE("Far terrain scheduler cancels obsolete view work",
          "[render][far-terrain][scheduler][cancellation][priority]") {
    std::mutex gateMutex;
    std::condition_variable gateCv;
    size_t enteredWorkers = 0;
    bool released = false;
    FarTerrainSource source = farTerrainTestSource();
    const auto sample = source.sample;
    source.sample = [&](int64_t x, int64_t z, worldgen::SurfaceFootprint footprint) {
        {
            std::unique_lock lock(gateMutex);
            ++enteredWorkers;
            gateCv.notify_all();
            gateCv.wait(lock, [&] { return released; });
        }
        return sample(x, z, footprint);
    };

    FarTerrainSchedulerLimits limits;
    limits.maxPending = 8;
    limits.maxCompleted = 8;
    limits.maxCacheEntries = 8;
    limits.maxCacheBytes = 32 * 1024 * 1024;
    FarTerrainScheduler scheduler(source, limits);
    std::array<FarTerrainKey, 8> keys{};
    for (size_t index = 0; index < keys.size(); ++index) {
        keys[index] = {static_cast<int64_t>(index), 0, FarTerrainStep::SIXTEEN};
        REQUIRE(scheduler.enqueue(keys[index]));
    }

    bool allWorkersEntered = false;
    {
        std::unique_lock lock(gateMutex);
        allWorkersEntered = gateCv.wait_for(lock, std::chrono::seconds(2), [&] {
            return enteredWorkers >= std::min(limits.maxPending, FarTerrainScheduler::WORKER_COUNT);
        });
    }
    scheduler.retainWanted({keys[0]});
    {
        std::lock_guard lock(gateMutex);
        released = true;
    }
    gateCv.notify_all();
    REQUIRE(allWorkersEntered);

    for (int attempt = 0; attempt < 400 && scheduler.stats().inFlight != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const FarTerrainSchedulerStats stats = scheduler.stats();
    REQUIRE(stats.inFlight == 0);
    REQUIRE(stats.built == 1);
    REQUIRE(stats.canceled == keys.size() - 1);
    REQUIRE(stats.cacheEntries == 1);
    REQUIRE(stats.completed == 1);
    REQUIRE(scheduler.findCached(keys[0]));
}

// ============================================================================
// Greedy Mesher Tests
// ============================================================================

TEST_CASE("Mesher: empty chunk produces no geometry", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // All AIR, no solid blocks

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    REQUIRE(output.vertices.empty());
    REQUIRE(output.indices.empty());
    REQUIRE(output.vertices.capacity() == 0);
    REQUIRE(output.indices.capacity() == 0);
}

TEST_CASE("Mesher: single block produces 6 faces", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 6 faces × 4 vertices = 24 vertices
    REQUIRE(output.vertices.size() == 24);
    // 6 faces × 2 triangles × 3 indices = 36 indices
    REQUIRE(output.indices.size() == 36);
}

TEST_CASE("Mesher: opaque faces use outward winding in all six directions",
          "[render][mesher][winding]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.opaqueIndexCount == 36);

    const auto vertexPosition = [](const Vertex& vertex) {
        return Vec3{static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                    static_cast<float>(vertex.pz)};
    };
    const auto expectedNormal = [](FaceNormal face) {
        switch (face) {
            case FaceNormal::PLUS_X:
                return Vec3{1.f, 0.f, 0.f};
            case FaceNormal::MINUS_X:
                return Vec3{-1.f, 0.f, 0.f};
            case FaceNormal::PLUS_Z:
                return Vec3{0.f, 0.f, 1.f};
            case FaceNormal::MINUS_Z:
                return Vec3{0.f, 0.f, -1.f};
            case FaceNormal::PLUS_Y:
                return Vec3{0.f, 1.f, 0.f};
            case FaceNormal::MINUS_Y:
                return Vec3{0.f, -1.f, 0.f};
            case FaceNormal::CROSS:
            case FaceNormal::TORCH_CROSS:
                return Vec3{};
        }
        return Vec3{};
    };

    std::array<bool, 6> found{};
    for (size_t offset = 0; offset < output.opaqueIndexCount; offset += 6) {
        const Vertex& first = output.vertices[output.indices[offset]];
        const Vertex& second = output.vertices[output.indices[offset + 1]];
        const Vertex& third = output.vertices[output.indices[offset + 2]];
        const FaceNormal face = unpackFace(first.faceAttr);
        REQUIRE(face != FaceNormal::CROSS);
        const Vec3 normal = (vertexPosition(second) - vertexPosition(first))
                                .cross(vertexPosition(third) - vertexPosition(first));
        REQUIRE(normal.dot(expectedNormal(face)) > 0.f);
        found[static_cast<size_t>(face)] = true;
    }
    for (bool faceFound : found)
        REQUIRE(faceFound);
}

TEST_CASE("Mesher: a solid cuboid greedily reduces every opaque direction",
          "[render][mesher][greedy]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    // Uniform derived light keeps smooth per-vertex shading from splitting
    // the underside, so this fixture isolates greedy face merging.
    snapshot.derivedSkyLightValid = true;
    for (int y = 3; y < 10; ++y) {
        for (int z = 4; z < 10; ++z) {
            for (int x = 2; x < 7; ++x) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
        }
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.opaqueIndexCount == 36);

    std::array<size_t, 6> verticesPerFace{};
    for (const Vertex& vertex : output.vertices) {
        const FaceNormal face = unpackFace(vertex.faceAttr);
        REQUIRE(face != FaceNormal::CROSS);
        ++verticesPerFace[static_cast<size_t>(face)];
    }
    for (size_t count : verticesPerFace)
        REQUIRE(count == 4);
}

TEST_CASE("Mesher: reused scratch produces byte-identical output",
          "[render][mesher][determinism]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    for (int z = 3; z < 12; ++z) {
        for (int x = 2; x < 11; ++x) {
            const int height = 4 + (x * 3 + z * 5) % 6;
            for (int y = 2; y < height; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] =
                    y + 1 == height ? BlockType::GRASS : BlockType::STONE;
            }
        }
    }
    snapshot.blocks[MeshSnapshot::index(1, 10, 1)] = BlockType::TALL_GRASS;
    snapshot.blocks[MeshSnapshot::index(13, 8, 13)] = BlockType::LILY_PAD;
    snapshot.blocks[MeshSnapshot::index(14, 3, 14)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(14, 3, 14)] = FluidState::flowing(5).packed();

    MeshScratch scratch;
    const MeshOutput first = LODMesher::buildMesh(snapshot, scratch);
    scratch.faceKeys.fill(0xFFFFU);
    scratch.skyHeight.fill(0xFFU);
    const MeshOutput second = LODMesher::buildMesh(snapshot, scratch);

    REQUIRE(first.opaqueIndexCount == second.opaqueIndexCount);
    REQUIRE(first.indices == second.indices);
    REQUIRE(first.vertices.size() == second.vertices.size());
    REQUIRE(std::memcmp(first.vertices.data(), second.vertices.data(),
                        first.vertices.size() * sizeof(Vertex)) == 0);
}

TEST_CASE("Mesher: 2x2 flat merges top face", "[render][mesher]") {
    // Uniform derived light keeps smooth per-vertex shading from splitting
    // the underside, so this fixture isolates greedy face merging.
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    // 2x2 square of STONE at local y=8
    snapshot.blocks[MeshSnapshot::index(0, 8, 0)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(1, 8, 0)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(0, 8, 1)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(1, 8, 1)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);

    // Without greedy merge: 4 top faces = 16 vertices
    // With greedy merge: 1 top face = 4 vertices
    // Total expected:
    //   Top (+Y): 1 merged quad = 4 vertices, 6 indices
    //   Bottom (-Y): 1 merged quad = 4 vertices, 6 indices
    //   +X face (right side of x=1 column): 2 quads (z=0 and z=1) = 8 vertices, 12 indices
    //   -X face (left side of x=0 column): 2 quads = 8 vertices, 12 indices
    //   +Z face (front of z=1 row): 2 quads = 8 vertices, 12 indices
    //   -Z face (back of z=0 row): 2 quads = 8 vertices, 12 indices
    // Total: 40 vertices, 60 indices
    //
    // But +X and -X faces can also merge vertically since all 4 blocks are at same Y
    // +X: blocks at (1,8,0) and (1,8,1) both have +X exposed with the same type
    //   They're adjacent in Z direction, so they merge into 1 quad: 4 vertices, 6 indices
    // Same for -X, +Z, -Z
    //
    // Total: 6 faces × 4 vertices = 24 vertices, 6 × 6 = 36 indices

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify the top face is a single quad (first 4 vertices)
    // All 4 top-face vertices should decode to FaceNormal::PLUS_Y
    bool foundTopQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        if (unpackFace(output.vertices[i].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 1].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 2].faceAttr) == FaceNormal::PLUS_Y &&
            unpackFace(output.vertices[i + 3].faceAttr) == FaceNormal::PLUS_Y) {
            foundTopQuad = true;
            break;
        }
    }
    REQUIRE(foundTopQuad);
}

TEST_CASE("Mesher: flora emits a contained cross of two quads", "[render][mesher][flora]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::GRASS);
    chunk.setBlock(8, 9, 8, BlockType::TALL_GRASS);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Grass cube: 6 quads. Flora shares four vertices per diagonal while
    // indexing both windings so it remains visible with back-face culling.
    REQUIRE(output.vertices.size() == 32);
    REQUIRE(output.indices.size() == 60);

    int crossVerts = 0;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::CROSS)
            continue;
        ++crossVerts;
        REQUIRE(unpackTextureLayer(v.faceAttr) == static_cast<uint8_t>(BlockType::TALL_GRASS));
        // The coordinate-hashed pose remains contained in its anchor cell and
        // spans the full cell height.
        float px = static_cast<float>(v.px);
        float py = static_cast<float>(v.py);
        float pz = static_cast<float>(v.pz);
        REQUIRE(px >= 8.0f);
        REQUIRE(px <= 9.0f);
        REQUIRE(pz >= 8.0f);
        REQUIRE(pz <= 9.0f);
        REQUIRE((py == 9.f || py == 10.f));
    }
    REQUIRE(crossVerts == 8);

    for (size_t offset : {size_t{36}, size_t{48}}) {
        REQUIRE(output.indices[offset + 6] == output.indices[offset]);
        REQUIRE(output.indices[offset + 7] == output.indices[offset + 2]);
        REQUIRE(output.indices[offset + 8] == output.indices[offset + 1]);
        REQUIRE(output.indices[offset + 9] == output.indices[offset + 3]);
        REQUIRE(output.indices[offset + 10] == output.indices[offset + 5]);
        REQUIRE(output.indices[offset + 11] == output.indices[offset + 4]);
    }
}

TEST_CASE("Mesher: dense flora poses vary deterministically across the world lattice",
          "[render][mesher][flora][determinism]") {
    // This is the cubic column containing the seed-42 dense-flora regression
    // scene. Populate every surface cell so repeated poses would form the
    // conspicuous rows seen in the aerial capture.
    Chunk chunk(ChunkPos{-1553, 4, -601});
    for (int z = 0; z < CHUNK_DEPTH; ++z) {
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            chunk.setBlock(x, 8, z, BlockType::GRASS);
            chunk.setBlock(x, 9, z, BlockType::TALL_GRASS);
        }
    }

    LODMesher mesher;
    const MeshOutput first = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    const MeshOutput second = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    REQUIRE(first.indices == second.indices);
    REQUIRE(first.vertices.size() == second.vertices.size());
    REQUIRE(std::memcmp(first.vertices.data(), second.vertices.data(),
                        first.vertices.size() * sizeof(Vertex)) == 0);

    std::vector<const Vertex*> floraVertices;
    for (const Vertex& vertex : first.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::CROSS) {
            floraVertices.push_back(&vertex);
        }
    }
    REQUIRE(floraVertices.size() == CHUNK_WIDTH * CHUNK_DEPTH * 8);

    std::set<std::array<int, 4>> poses;
    for (size_t plant = 0; plant < floraVertices.size() / 8; ++plant) {
        const Vertex& firstBottom = *floraVertices[plant * 8];
        const Vertex& secondBottom = *floraVertices[plant * 8 + 1];
        const int localX = static_cast<int>(plant % CHUNK_WIDTH);
        const int localZ = static_cast<int>(plant / CHUNK_WIDTH);
        const float centerX =
            (static_cast<float>(firstBottom.px) + static_cast<float>(secondBottom.px)) * 0.5F;
        const float centerZ =
            (static_cast<float>(firstBottom.pz) + static_cast<float>(secondBottom.pz)) * 0.5F;
        poses.insert({
            static_cast<int>(std::lround((centerX - static_cast<float>(localX) - 0.5F) * 32.0F)),
            static_cast<int>(std::lround((centerZ - static_cast<float>(localZ) - 0.5F) * 32.0F)),
            static_cast<int>(std::lround(
                (static_cast<float>(secondBottom.px) - static_cast<float>(firstBottom.px)) * 8.0F)),
            static_cast<int>(std::lround(
                (static_cast<float>(secondBottom.pz) - static_cast<float>(firstBottom.pz)) * 8.0F)),
        });
    }
    REQUIRE(poses.size() >= 12);
}

TEST_CASE("Mesher keeps floor torches centered and outside flora shading",
          "[render][mesher][torch][emissive][gameplay]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    chunk.setBlock(8, 9, 8, BlockType::TORCH);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    std::vector<const Vertex*> torchVertices;
    for (const Vertex& vertex : output.vertices) {
        if (unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::TORCH)) {
            continue;
        }
        torchVertices.push_back(&vertex);
        REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::TORCH_CROSS);
        REQUIRE(unpackSway(vertex.faceAttr) == 0);
        REQUIRE(unpackEmissive(vertex.faceAttr));
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
    }

    REQUIRE(torchVertices.size() == 8);
    const auto coordinate = [](const Vertex* vertex) {
        return std::array{static_cast<float>(vertex->px), static_cast<float>(vertex->py),
                          static_cast<float>(vertex->pz)};
    };
    const auto first = coordinate(torchVertices[0]);
    const auto second = coordinate(torchVertices[1]);
    REQUIRE((first[0] + second[0]) * 0.5F == Catch::Approx(8.5F));
    REQUIRE((first[2] + second[2]) * 0.5F == Catch::Approx(8.5F));
    for (const Vertex* vertex : torchVertices) {
        REQUIRE(static_cast<float>(vertex->px) >= 8.0F);
        REQUIRE(static_cast<float>(vertex->px) <= 9.0F);
        REQUIRE(static_cast<float>(vertex->pz) >= 8.0F);
        REQUIRE(static_cast<float>(vertex->pz) <= 9.0F);
        REQUIRE(
            (static_cast<float>(vertex->py) == 9.0F || static_cast<float>(vertex->py) == 10.0F));
    }
}

TEST_CASE("Mesher: flat flora emits explicit front and back winding",
          "[render][mesher][flora][winding]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LILY_PAD);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    REQUIRE(output.vertices.size() == 4);
    REQUIRE(output.opaqueIndexCount == 12);
    REQUIRE(output.indices == std::vector<uint32_t>{0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2});
}

TEST_CASE("Mesher: flora does not break greedy merging of the ground", "[render][mesher][flora]") {
    // 2x2 grass floor with one flower on top: the floor's +Y face must still
    // merge into a single quad. Uniform derived light keeps the shaded
    // underside from splitting so the fixture isolates flora interaction.
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    for (int z = 0; z < 2; ++z)
        for (int x = 0; x < 2; ++x)
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::GRASS;
    snapshot.blocks[MeshSnapshot::index(0, 9, 0)] = BlockType::FLOWER_RED;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);

    // 2x2 slab = 24 vertices (all faces merged) + 8 flora vertices
    REQUIRE(output.vertices.size() == 32);
}

TEST_CASE("Mesher: water surfaces land in the water section", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // Stone floor with one water block on top: the water's top face (under
    // air) and four sides are water-section; the floor's faces are opaque.
    chunk.setBlock(8, 4, 8, BlockType::STONE);
    chunk.setBlock(8, 5, 8, BlockType::WATER);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Opaque: stone cube 6 faces (water doesn't hide the +Y face) = 36 idx.
    // Water: top + 4 sides = 5 quads = 30 indices (bottom hidden by stone).
    REQUIRE(output.opaqueIndexCount == 36);
    REQUIRE(output.indices.size() == 66);

    // Implicit generated water is a full source block in every meshing path.
    // Animation belongs to the fragment normal, not vertex displacement, so
    // exact and far ownership can exchange this planar top without exposing
    // a triangle diagonal.
    size_t topIndexCount = 0;
    for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
        const Vertex& vertex = output.vertices[output.indices[offset]];
        if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        REQUIRE(static_cast<float>(vertex.py) == 6.0F);
        ++topIndexCount;
    }
    REQUIRE(topIndexCount == 6);
}

TEST_CASE("Mesher: interior water-water faces are culled", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 2x2x2 water cube on a stone slab
    for (int z = 4; z < 6; ++z)
        for (int x = 4; x < 6; ++x) {
            chunk.setBlock(x, 3, z, BlockType::STONE);
            chunk.setBlock(x, 4, z, BlockType::WATER);
            chunk.setBlock(x, 5, z, BlockType::WATER);
        }

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Water section: greedy-merged top (1 quad) + 4 merged side walls
    // (2 wide × 2 tall each → 1 quad per direction) = 5 quads = 30 indices
    uint32_t waterIndexCount =
        static_cast<uint32_t>(output.indices.size()) - output.opaqueIndexCount;
    REQUIRE(waterIndexCount == 30);
}

TEST_CASE("Snapshot mesher uses runtime water levels and falling metadata",
          "[render][mesher][water]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::flowing(7).packed();

    MeshScratch scratch;
    MeshOutput shallow = LODMesher::buildMesh(snapshot, scratch);
    bool foundShallowTop = false;
    for (const Vertex& vertex : shallow.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
            static_cast<float>(vertex.py) == 8.125f) {
            foundShallowTop = true;
        }
    }
    REQUIRE(foundShallowTop);

    STATIC_REQUIRE(fluidSurfaceHeight(FluidState::source()) == 1.0F);
    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::source().packed();
    MeshOutput source = LODMesher::buildMesh(snapshot, scratch);
    size_t sourceTopIndexCount = 0;
    for (size_t offset = source.opaqueIndexCount; offset + 2 < source.indices.size(); offset += 3) {
        const Vertex& first = source.vertices[source.indices[offset]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        std::array<Vec3, 3> triangle{};
        for (size_t corner = 0; corner < triangle.size(); ++corner) {
            const Vertex& vertex = source.vertices[source.indices[offset + corner]];
            REQUIRE(unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y);
            REQUIRE(static_cast<float>(vertex.py) == 9.0F);
            triangle[corner] = {static_cast<float>(vertex.px), static_cast<float>(vertex.py),
                                static_cast<float>(vertex.pz)};
            ++sourceTopIndexCount;
        }
        const Vec3 normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
        REQUIRE(std::abs(normal.x) <= 1.0e-6F);
        REQUIRE(std::abs(normal.z) <= 1.0e-6F);
        REQUIRE(std::abs(normal.y) > 0.0F);
    }
    REQUIRE(sourceTopIndexCount == 6);

    snapshot.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::falling(3).packed();
    MeshOutput falling = LODMesher::buildMesh(snapshot, scratch);
    bool foundFallingFace = false;
    for (const Vertex& vertex : falling.vertices) {
        if (unpackFluidFalling(vertex.faceAttr))
            foundFallingFace = true;
    }
    REQUIRE(foundFallingFace);
}

TEST_CASE("Snapshot water sides are exclusive to falling columns",
          "[render][mesher][water][shoreline]") {
    constexpr std::array<std::pair<int, int>, 4> horizontalEdges{{
        {0, 8},
        {CHUNK_EDGE - 1, 8},
        {8, 0},
        {8, CHUNK_EDGE - 1},
    }};
    MeshScratch scratch;
    for (const auto& [x, z] : horizontalEdges) {
        MeshSnapshot source;
        source.clear();
        source.blocks[MeshSnapshot::index(x, 7, z)] = BlockType::STONE;
        source.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::WATER;
        const MeshOutput output = LODMesher::buildMesh(source, scratch);
        for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
            REQUIRE(unpackFace(output.vertices[output.indices[offset]].faceAttr) ==
                    FaceNormal::PLUS_Y);
        }
    }

    MeshSnapshot waterfall;
    waterfall.clear();
    waterfall.blocks[MeshSnapshot::index(8, 7, 8)] = BlockType::STONE;
    waterfall.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::WATER;
    waterfall.fluidStates[MeshSnapshot::index(8, 8, 8)] = FluidState::falling(3).packed();
    const MeshOutput falling = LODMesher::buildMesh(waterfall, scratch);
    std::array<bool, 4> sideFound{};
    for (size_t offset = falling.opaqueIndexCount; offset < falling.indices.size(); ++offset) {
        const FaceNormal face = unpackFace(falling.vertices[falling.indices[offset]].faceAttr);
        if (face == FaceNormal::MINUS_X)
            sideFound[0] = true;
        if (face == FaceNormal::PLUS_X)
            sideFound[1] = true;
        if (face == FaceNormal::MINUS_Z)
            sideFound[2] = true;
        if (face == FaceNormal::PLUS_Z)
            sideFound[3] = true;
    }
    REQUIRE(std::all_of(sideFound.begin(), sideFound.end(), [](bool found) { return found; }));
}

TEST_CASE("Generated incised rivers mesh continuously across exact cube faces",
          "[render][mesher][water][river][seam][regression]") {
    ChunkGenerator generator(42);
    constexpr int64_t RIVER_X = -12'801;
    constexpr int64_t RIVER_Z = 2'759;
    const std::array<worldgen::SurfaceSample, 4> riverSamples = {
        generator.sampleExactSurface(RIVER_X, RIVER_Z),
        generator.sampleExactSurface(RIVER_X + 1, RIVER_Z),
        generator.sampleExactSurface(RIVER_X, RIVER_Z + 1),
        generator.sampleExactSurface(RIVER_X + 1, RIVER_Z + 1),
    };
    const int WATER_Y = static_cast<int>(std::ceil(riverSamples.front().waterSurface)) - 1;
    for (const worldgen::SurfaceSample& sample : riverSamples) {
        REQUIRE(sample.hydrology.river);
        REQUIRE_FALSE(sample.hydrology.lake);
        REQUIRE_FALSE(sample.hydrology.waterfall);
        REQUIRE(sample.waterSurface > sample.terrainHeight);
        REQUIRE(static_cast<int>(std::ceil(sample.waterSurface)) - 1 == WATER_Y);
    }
    const ChunkPos center{Chunk::worldToChunk(RIVER_X), Chunk::worldToChunkY(WATER_Y),
                          Chunk::worldToChunk(RIVER_Z)};
    std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
    auto cubeAt = [&](ChunkPos position) -> Chunk& {
        const auto key = std::tuple{position.x, position.y, position.z};
        auto found = cubes.find(key);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(position);
            generator.generateCube(*cube);
            found = cubes.emplace(key, std::move(cube)).first;
        }
        return *found->second;
    };

    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = center;
    for (int y = -1; y <= CHUNK_EDGE; ++y) {
        for (int z = -1; z <= CHUNK_EDGE; ++z) {
            for (int x = -1; x <= CHUNK_EDGE; ++x) {
                const int64_t worldX = center.x * CHUNK_EDGE + x;
                const int worldY = center.y * CHUNK_EDGE + y;
                const int64_t worldZ = center.z * CHUNK_EDGE + z;
                Chunk& cube = cubeAt({Chunk::worldToChunk(worldX), Chunk::worldToChunkY(worldY),
                                      Chunk::worldToChunk(worldZ)});
                const int index = MeshSnapshot::index(x, y, z);
                snapshot.blocks[index] =
                    cube.getBlock(Chunk::worldToLocal(worldX), Chunk::worldToLocalY(worldY),
                                  Chunk::worldToLocal(worldZ));
                snapshot.fluidStates[index] =
                    cube.getFluidState(Chunk::worldToLocal(worldX), Chunk::worldToLocalY(worldY),
                                       Chunk::worldToLocal(worldZ))
                        .packed();
            }
        }
    }

    const int WATER_LOCAL_Y = Chunk::worldToLocalY(WATER_Y);
    const int RIVER_LOCAL_Z = Chunk::worldToLocal(RIVER_Z);
    REQUIRE(snapshot.at(15, WATER_LOCAL_Y, RIVER_LOCAL_Z) == BlockType::WATER);
    REQUIRE(snapshot.at(16, WATER_LOCAL_Y, RIVER_LOCAL_Z) == BlockType::WATER);
    REQUIRE(snapshot.at(15, WATER_LOCAL_Y, RIVER_LOCAL_Z + 1) == BlockType::WATER);
    REQUIRE(snapshot.at(16, WATER_LOCAL_Y, RIVER_LOCAL_Z + 1) == BlockType::WATER);
    REQUIRE_FALSE(snapshot.fluidAt(15, WATER_LOCAL_Y, RIVER_LOCAL_Z).isFalling());
    REQUIRE_FALSE(snapshot.fluidAt(16, WATER_LOCAL_Y, RIVER_LOCAL_Z).isFalling());

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.indices.size() > output.opaqueIndexCount);
    bool foundSurfaceAtSharedFace = false;
    for (size_t offset = output.opaqueIndexCount; offset < output.indices.size(); ++offset) {
        const Vertex& vertex = output.vertices[output.indices[offset]];
        REQUIRE_FALSE(unpackFluidFalling(vertex.faceAttr));
        const FaceNormal face = unpackFace(vertex.faceAttr);
        if (face == FaceNormal::PLUS_Y && vertex.px >= 15 && vertex.pz >= RIVER_LOCAL_Z &&
            vertex.pz <= RIVER_LOCAL_Z + 1) {
            foundSurfaceAtSharedFace = true;
        }
        REQUIRE_FALSE((face == FaceNormal::PLUS_X && vertex.px == CHUNK_EDGE &&
                       vertex.pz >= RIVER_LOCAL_Z && vertex.pz <= RIVER_LOCAL_Z + 1));
    }
    REQUIRE(foundSurfaceAtSharedFace);
}

TEST_CASE("Mesher: lava renders as an opaque cube section", "[render][mesher][water]") {
    Chunk chunk(ChunkPos{0, 0, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // 6 faces, all in the opaque section; nothing in the water section
    REQUIRE(output.opaqueIndexCount == 36);
    REQUIRE(output.indices.size() == 36);
}

// ============================================================================
// Neighbor-aware (snapshot) meshing: chunk border correctness
// ============================================================================

TEST_CASE("Snapshot mesher: boundary faces follow real neighbor blocks",
          "[render][mesher][border]") {
    MeshSnapshot snapshot;
    snapshot.resize();
    // One stone block on the +X border of the chunk
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;

    // Case 1: neighbor cell across the border is AIR → the +X boundary face
    // must exist (the old mesher skipped the boundary layer: a hole)
    {
        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 24); // full cube
        bool plusXAt16 = false;
        for (const Vertex& v : output.vertices) {
            if (unpackFace(v.faceAttr) == FaceNormal::PLUS_X && static_cast<float>(v.px) == 16.f)
                plusXAt16 = true;
        }
        REQUIRE(plusXAt16);
    }

    // Case 2: neighbor cell solid → the boundary face is culled (the old
    // mesher's -X pass always emitted a hidden wall from the other side)
    {
        snapshot.blocks[MeshSnapshot::index(16, 8, 8)] = BlockType::STONE;
        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 20); // cube minus the shared face
        for (const Vertex& v : output.vertices) {
            REQUIRE(!(unpackFace(v.faceAttr) == FaceNormal::PLUS_X &&
                      static_cast<float>(v.px) == 16.f));
        }
    }
}

TEST_CASE("Snapshot mesher: -X border wall culled against a solid neighbor",
          "[render][mesher][border]") {
    MeshSnapshot snapshot;
    snapshot.resize();
    snapshot.blocks[MeshSnapshot::index(0, 8, 8)] = BlockType::STONE;
    // Solid neighbor wall behind it (x = -1)
    snapshot.blocks[MeshSnapshot::index(-1, 8, 8)] = BlockType::STONE;

    MeshScratch scratch;
    MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    REQUIRE(output.vertices.size() == 20);
    for (const Vertex& v : output.vertices) {
        REQUIRE(
            !(unpackFace(v.faceAttr) == FaceNormal::MINUS_X && static_cast<float>(v.px) == 0.f));
    }
}

TEST_CASE("Snapshot mesher emits exposed faces on all six cube boundaries",
          "[render][mesher][border]") {
    struct BoundaryCase {
        int x;
        int y;
        int z;
        FaceNormal exposedFace;
    };
    constexpr std::array<BoundaryCase, 6> cases{{
        {0, 8, 8, FaceNormal::MINUS_X},
        {15, 8, 8, FaceNormal::PLUS_X},
        {8, 0, 8, FaceNormal::MINUS_Y},
        {8, 15, 8, FaceNormal::PLUS_Y},
        {8, 8, 0, FaceNormal::MINUS_Z},
        {8, 8, 15, FaceNormal::PLUS_Z},
    }};

    MeshScratch scratch;
    for (const BoundaryCase& test : cases) {
        MeshSnapshot snapshot;
        snapshot.clear();
        snapshot.blocks[MeshSnapshot::index(test.x, test.y, test.z)] = BlockType::STONE;
        const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 24);
        REQUIRE(std::any_of(output.vertices.begin(), output.vertices.end(), [&](const Vertex& v) {
            return unpackFace(v.faceAttr) == test.exposedFace;
        }));
    }
}

TEST_CASE("Snapshot mesher culls shared faces on all six cube boundaries",
          "[render][mesher][border]") {
    struct BoundaryCase {
        int blockX;
        int blockY;
        int blockZ;
        int neighborX;
        int neighborY;
        int neighborZ;
        FaceNormal hiddenFace;
    };
    constexpr std::array<BoundaryCase, 6> cases{{
        {0, 8, 8, -1, 8, 8, FaceNormal::MINUS_X},
        {15, 8, 8, 16, 8, 8, FaceNormal::PLUS_X},
        {8, 0, 8, 8, -1, 8, FaceNormal::MINUS_Y},
        {8, 15, 8, 8, 16, 8, FaceNormal::PLUS_Y},
        {8, 8, 0, 8, 8, -1, FaceNormal::MINUS_Z},
        {8, 8, 15, 8, 8, 16, FaceNormal::PLUS_Z},
    }};

    MeshScratch scratch;
    for (const BoundaryCase& test : cases) {
        MeshSnapshot snapshot;
        snapshot.clear();
        snapshot.blocks[MeshSnapshot::index(test.blockX, test.blockY, test.blockZ)] =
            BlockType::STONE;
        snapshot.blocks[MeshSnapshot::index(test.neighborX, test.neighborY, test.neighborZ)] =
            BlockType::STONE;

        MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
        REQUIRE(output.vertices.size() == 20);
        for (const Vertex& vertex : output.vertices) {
            REQUIRE(unpackFace(vertex.faceAttr) != test.hiddenFace);
        }
    }
}

TEST_CASE("MegaBuffer free-list coalescing is bounds-safe and lossless", "[render][megabuffer]") {
    using Region = std::pair<uint64_t, uint64_t>;

    // Regression: a single-entry list made the old compaction write one
    // element past the vector's end, slow heap corruption that surfaced as
    // buzzing audio and malloc traps minutes into a session.
    std::vector<Region> single = {{256, 512}};
    MegaBuffer::coalesceFreeList(single);
    REQUIRE(single == std::vector<Region>{{256, 512}});

    // Adjacent regions merge (any input order)…
    std::vector<Region> adjacent = {{768, 256}, {256, 512}};
    MegaBuffer::coalesceFreeList(adjacent);
    REQUIRE(adjacent == std::vector<Region>{{256, 768}});

    // …gaps survive, and the LAST region is kept (the old code erased it)
    std::vector<Region> gapped = {{0, 256}, {512, 256}, {2048, 256}};
    MegaBuffer::coalesceFreeList(gapped);
    REQUIRE(gapped == std::vector<Region>{{0, 256}, {512, 256}, {2048, 256}});

    // Chain of three merges into one
    std::vector<Region> chain = {{512, 256}, {0, 512}, {768, 1024}};
    MegaBuffer::coalesceFreeList(chain);
    REQUIRE(chain == std::vector<Region>{{0, 1792}});

    std::vector<Region> empty;
    MegaBuffer::coalesceFreeList(empty);
    REQUIRE(empty.empty());
}

TEST_CASE("Segmented far arena grows lazily and routes allocations to their slab",
          "[render][megabuffer][far-terrain][residency]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    constexpr uint64_t SLAB_BYTES = 4 * 1024;
    SegmentedMegaBuffer arena(device, SLAB_BYTES * 2, SLAB_BYTES * 2, SLAB_BYTES, SLAB_BYTES);
    REQUIRE(arena.segmentCount() == 0);

    constexpr uint32_t VERTEX_COUNT = 200;
    constexpr uint32_t INDEX_COUNT = 400;
    std::vector<Vertex> vertices(VERTEX_COUNT);
    std::vector<uint32_t> indices(INDEX_COUNT);
    auto first = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 1);
    arena.uploadVertices(vertices.data(), vertices.size() * sizeof(Vertex), first);
    arena.uploadIndices(indices.data(), indices.size() * sizeof(uint32_t), first);

    auto second = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 2);
    REQUIRE(second.vertexBuffer != first.vertexBuffer);
    REQUIRE(second.indexBuffer != first.indexBuffer);
    const uint64_t usedBeforeDeferred = arena.vertexUsed() + arena.indexUsed();
    arena.deferFree(first, 7);
    arena.drainDeferredFrees(6);
    REQUIRE(arena.vertexUsed() + arena.indexUsed() == usedBeforeDeferred);
    arena.drainDeferredFrees(7);
    REQUIRE(arena.vertexUsed() + arena.indexUsed() < usedBeforeDeferred);

    auto reused = arena.allocate(VERTEX_COUNT, INDEX_COUNT);
    REQUIRE(arena.segmentCount() == 2);
    REQUIRE(reused.vertexBuffer != second.vertexBuffer);
    arena.free(reused);
    arena.free(second);
    REQUIRE(arena.vertexUsed() == 0);
    REQUIRE(arena.indexUsed() == 0);
}

TEST_CASE("Segmented exact arena preserves spawn meshes while adding capacity",
          "[render][megabuffer][streaming][lod][regression]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    constexpr uint64_t SLAB_BYTES = 4 * 1024;
    SegmentedMegaBuffer arena(device, SLAB_BYTES * 2, SLAB_BYTES * 2, SLAB_BYTES, SLAB_BYTES);
    std::vector<Vertex> vertices(200);
    std::vector<uint32_t> indices(400);

    auto spawnMesh = arena.allocate(static_cast<uint32_t>(vertices.size()),
                                    static_cast<uint32_t>(indices.size()));
    const id<MTLBuffer> spawnVertexBuffer = spawnMesh.vertexBuffer;
    const id<MTLBuffer> spawnIndexBuffer = spawnMesh.indexBuffer;
    auto expandedMesh = arena.allocate(static_cast<uint32_t>(vertices.size()),
                                       static_cast<uint32_t>(indices.size()));

    REQUIRE(arena.segmentCount() == 2);
    REQUIRE(spawnMesh.vertexBuffer == spawnVertexBuffer);
    REQUIRE(spawnMesh.indexBuffer == spawnIndexBuffer);
    REQUIRE(spawnMesh.vertexBuffer != expandedMesh.vertexBuffer);
    REQUIRE_NOTHROW(
        arena.uploadVertices(vertices.data(), vertices.size() * sizeof(Vertex), spawnMesh));
    REQUIRE_NOTHROW(
        arena.uploadIndices(indices.data(), indices.size() * sizeof(uint32_t), spawnMesh));
    arena.free(spawnMesh);
    arena.free(expandedMesh);
}

TEST_CASE("MeshScheduler: builds off-thread with version stamps", "[render][scheduler]") {
    World world(42, 2);
    constexpr ChunkPos center{0, 4, 0};
    world.getChunk(center);
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                if (offsetX == 0 && offsetY == 0 && offsetZ == 0)
                    continue;
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    // A mesh snapshot also needs the sparse generated occupancy above this
    // cube so its skylight cutoff is final. Proven-empty gaps stay absent.
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto plan =
                world.generator().getColumnPlan({center.x + offsetX, center.z + offsetZ});
            REQUIRE(plan);
            for (const int32_t section : plan->exposedSections()) {
                if (section >= center.y) {
                    REQUIRE(world.getChunk({center.x + offsetX, section, center.z + offsetZ}));
                }
            }
        }
    }
    for (int z = 7; z <= 9; ++z) {
        for (int y = 71; y <= 73; ++y) {
            for (int x = 7; x <= 9; ++x) {
                world.setBlock(x, y, z, BlockType::AIR);
            }
        }
    }
    world.setBlock(8, 72, 8, BlockType::STONE);
    for (int pass = 0;
         pass < 64 && world.getStreamingWorkStats().publicationLightDeferredQueue != 0; ++pass) {
        world.reconcileLight(1'024);
    }
    REQUIRE(world.getStreamingWorkStats().publicationLightDeferredQueue == 0);

    MeshScheduler scheduler(world, 1);
    REQUIRE(scheduler.enqueue(center));

    std::vector<MeshResult> results;
    for (int i = 0; i < 500 && results.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        scheduler.drainCompleted(results);
    }
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].pos == center);
    REQUIRE(results[0].snapshotOk);
    REQUIRE(results[0].builtVersion == world.getChunk(center)->version.load());
    REQUIRE(!results[0].mesh.vertices.empty());

    // A chunk without generated neighbors reports the failed snapshot
    // instead of blocking (the renderer retries once the frontier catches up)
    REQUIRE(scheduler.enqueue(ChunkPos{40, 4, 40}));
    results.clear();
    for (int i = 0; i < 500 && results.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        scheduler.drainCompleted(results);
    }
    REQUIRE(results.size() == 1);
    REQUIRE(!results[0].snapshotOk);

    // Shutdown is idempotent and refuses further work
    scheduler.shutdown();
    scheduler.shutdown();
    REQUIRE(!scheduler.enqueue(center));
}

TEST_CASE("Exact mesh scheduling reserves capacity and ordering for the camera band",
          "[render][scheduler][priority][cold-start][regression]") {
    STATIC_REQUIRE(EXACT_MESH_CAMERA_RESERVED_SLOTS == 32);
    STATIC_REQUIRE(EXACT_MESH_MAX_INFLIGHT == 64);

    REQUIRE(meshLaneCanReserve(31, 0, MeshPriorityLane::BROAD_SURFACE));
    REQUIRE_FALSE(meshLaneCanReserve(32, 0, MeshPriorityLane::BROAD_SURFACE));
    REQUIRE(meshLaneCanReserve(32, 0, MeshPriorityLane::CAMERA_BAND));
    REQUIRE(meshLaneCanReserve(32, 0, MeshPriorityLane::CAMERA_COLUMN));
    REQUIRE(meshLaneCanReserve(63, 0, MeshPriorityLane::CAMERA_BAND));
    REQUIRE_FALSE(meshLaneCanReserve(63, 1, MeshPriorityLane::CAMERA_BAND));

    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4'096, 8,
                               MeshPriorityLane::BROAD_SURFACE, 0, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_COLUMN, 4'096, 8,
                               MeshPriorityLane::CAMERA_BAND, 0, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4, 9, MeshPriorityLane::CAMERA_BAND,
                               64, 0));
    REQUIRE(meshJobRanksBefore(MeshPriorityLane::CAMERA_BAND, 4, 9, MeshPriorityLane::CAMERA_BAND,
                               4, 10));
}

TEST_CASE("Exact surface and flora meshing use reserved near-player sublanes",
          "[render][scheduler][priority][surface][flora][regression]") {
    constexpr ExactMeshCandidatePriority CAMERA =
        exactMeshCandidatePriority(0, 40, 0, false, false, true);
    constexpr ExactMeshCandidatePriority EXPLORATION =
        exactMeshCandidatePriority(EXPLORATION_RADIUS_CHUNKS, 40, 0, true, false, false);
    constexpr ExactMeshCandidatePriority EXPLORATION_FLORA =
        exactMeshCandidatePriority(EXPLORATION_RADIUS_CHUNKS, 40, 0, true, false, true);
    constexpr ExactMeshCandidatePriority MEDIUM_SURFACE = exactMeshCandidatePriority(
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS, 4, 0, false, true, false);
    constexpr ExactMeshCandidatePriority MEDIUM_SURFACE_WITH_FLORA = exactMeshCandidatePriority(
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS, 4, 0, false, true, true);
    constexpr ExactMeshCandidatePriority OUTER_EXACT_SURFACE = exactMeshCandidatePriority(
        EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS, 4, 0, false, true, false);
    constexpr ExactMeshCandidatePriority BEYOND_EXACT_SURFACE = exactMeshCandidatePriority(
        EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS + 1, 4, 0, false, true, false);
    constexpr ExactMeshCandidatePriority MEDIUM_FLORA = exactMeshCandidatePriority(
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS, 4, 0, false, false, true);
    constexpr ExactMeshCandidatePriority MEDIUM_TERRAIN = exactMeshCandidatePriority(
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS, 4, 0, false, false, false);
    constexpr ExactMeshCandidatePriority OUTER_FLORA = exactMeshCandidatePriority(
        EXACT_FLORA_MESH_PRIORITY_RADIUS_CHUNKS + 1, 0, 0, false, false, true);
    constexpr ExactMeshCandidatePriority DISTANT_TERRAIN =
        exactMeshCandidatePriority(24, 0, 0, false, false, false);
    constexpr ExactMeshCandidatePriority NEAR_TALL_SURFACE =
        exactMeshCandidatePriority(7, WORLD_VERTICAL_CHUNKS - 1, 0, false, true, false);
    constexpr ExactMeshCandidatePriority FAR_FLAT_SURFACE = exactMeshCandidatePriority(
        EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS, 0, 0, false, true, false);

    STATIC_REQUIRE(CAMERA.lane == MeshPriorityLane::CAMERA_COLUMN);
    STATIC_REQUIRE(EXPLORATION.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(EXPLORATION_FLORA == EXPLORATION);
    STATIC_REQUIRE(EXPLORATION.distanceSquared < EXACT_FLORA_MESH_SUBLANE_OFFSET);
    STATIC_REQUIRE(MEDIUM_SURFACE.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(MEDIUM_SURFACE_WITH_FLORA == MEDIUM_SURFACE);
    STATIC_REQUIRE(MEDIUM_SURFACE.distanceSquared >= EXACT_SURFACE_MESH_SUBLANE_OFFSET);
    STATIC_REQUIRE(MEDIUM_SURFACE.distanceSquared < EXACT_FLORA_MESH_SUBLANE_OFFSET);
    STATIC_REQUIRE(OUTER_EXACT_SURFACE.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(BEYOND_EXACT_SURFACE.lane == MeshPriorityLane::BROAD_SURFACE);
    STATIC_REQUIRE(NEAR_TALL_SURFACE.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(FAR_FLAT_SURFACE.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(exactMeshCandidateRanksBefore(NEAR_TALL_SURFACE, FAR_FLAT_SURFACE));
    STATIC_REQUIRE(MEDIUM_FLORA.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(MEDIUM_FLORA.distanceSquared >= EXACT_FLORA_MESH_SUBLANE_OFFSET);
    STATIC_REQUIRE(MEDIUM_TERRAIN.lane == MeshPriorityLane::BROAD_SURFACE);
    STATIC_REQUIRE(OUTER_FLORA.lane == MeshPriorityLane::BROAD_SURFACE);
    STATIC_REQUIRE(DISTANT_TERRAIN.lane == MeshPriorityLane::BROAD_SURFACE);

    STATIC_REQUIRE(meshJobRanksBefore(CAMERA.lane, CAMERA.distanceSquared, 0, EXPLORATION.lane,
                                      EXPLORATION.distanceSquared, 0));
    STATIC_REQUIRE(meshJobRanksBefore(EXPLORATION.lane, EXPLORATION.distanceSquared, 0,
                                      MEDIUM_SURFACE.lane, MEDIUM_SURFACE.distanceSquared, 0));
    STATIC_REQUIRE(meshJobRanksBefore(MEDIUM_SURFACE.lane, MEDIUM_SURFACE.distanceSquared, 0,
                                      MEDIUM_FLORA.lane, MEDIUM_FLORA.distanceSquared, 0));
    STATIC_REQUIRE(meshJobRanksBefore(MEDIUM_FLORA.lane, MEDIUM_FLORA.distanceSquared, 0,
                                      DISTANT_TERRAIN.lane, DISTANT_TERRAIN.distanceSquared, 0));
    STATIC_REQUIRE(meshLaneCanReserve(EXACT_MESH_CAMERA_RESERVED_SLOTS, 0, MEDIUM_FLORA.lane));
}

TEST_CASE("Completed exact meshes publish near the camera before distant results",
          "[render][scheduler][priority][upload][regression]") {
    constexpr ChunkPos CAMERA{100, 8, -200};
    constexpr ExactMeshUploadPriority CAMERA_COLUMN =
        exactMeshUploadPriority({100, 80, -200}, CAMERA, EXPLORATION_RADIUS_CHUNKS);
    constexpr ExactMeshUploadPriority NEAR_EXPLORATION =
        exactMeshUploadPriority({104, 8, -203}, CAMERA, EXPLORATION_RADIUS_CHUNKS);
    constexpr ExactMeshUploadPriority FAR_RESULT =
        exactMeshUploadPriority({124, 8, -200}, CAMERA, EXPLORATION_RADIUS_CHUNKS);

    STATIC_REQUIRE(CAMERA_COLUMN.candidate.lane == MeshPriorityLane::CAMERA_COLUMN);
    STATIC_REQUIRE(NEAR_EXPLORATION.candidate.lane == MeshPriorityLane::CAMERA_BAND);
    STATIC_REQUIRE(FAR_RESULT.candidate.lane == MeshPriorityLane::BROAD_SURFACE);
    STATIC_REQUIRE(exactMeshUploadRanksBefore(CAMERA_COLUMN, NEAR_EXPLORATION));
    STATIC_REQUIRE(exactMeshUploadRanksBefore(NEAR_EXPLORATION, FAR_RESULT));

    std::vector<ChunkPos> completed{
        {124, 8, -200}, {104, 8, -203}, {100, 80, -200}, {99, 8, -199}, {101, 8, -199}};
    std::stable_sort(completed.begin(), completed.end(), [&](ChunkPos left, ChunkPos right) {
        return exactMeshUploadRanksBefore(
            exactMeshUploadPriority(left, CAMERA, EXPLORATION_RADIUS_CHUNKS),
            exactMeshUploadPriority(right, CAMERA, EXPLORATION_RADIUS_CHUNKS));
    });

    REQUIRE(completed[0] == ChunkPos{100, 80, -200});
    REQUIRE(completed[1] == ChunkPos{99, 8, -199});
    REQUIRE(completed[2] == ChunkPos{101, 8, -199});
    REQUIRE(completed.back() == ChunkPos{124, 8, -200});
}

TEST_CASE("Completed exact mesh upload ordering follows a moving camera",
          "[render][scheduler][priority][upload][movement][regression]") {
    constexpr ChunkPos OLD_CAMERA{0, 4, 0};
    constexpr ChunkPos NEW_CAMERA{64, 4, 0};
    constexpr ChunkPos OLD_RESULT{0, 4, 0};
    constexpr ChunkPos NEW_RESULT{64, 4, 0};

    STATIC_REQUIRE(exactMeshUploadRanksBefore(
        exactMeshUploadPriority(OLD_RESULT, OLD_CAMERA, EXPLORATION_RADIUS_CHUNKS),
        exactMeshUploadPriority(NEW_RESULT, OLD_CAMERA, EXPLORATION_RADIUS_CHUNKS)));
    STATIC_REQUIRE(exactMeshUploadRanksBefore(
        exactMeshUploadPriority(NEW_RESULT, NEW_CAMERA, EXPLORATION_RADIUS_CHUNKS),
        exactMeshUploadPriority(OLD_RESULT, NEW_CAMERA, EXPLORATION_RADIUS_CHUNKS)));
}

TEST_CASE("World snapshotForMeshing seals missing neighbors until the real halo arrives",
          "[world][mesher][border][streaming]") {
    World world(4242, 2);
    MeshSnapshot snapshot;
    // Keep this border-ownership fixture above every generated occupancy
    // section. Surface-height fixtures separately exercise the sparse sky
    // closure; here the missing horizontal halo is intentionally legal.
    constexpr ChunkPos center{0, WORLD_MAX_CHUNK_Y - 1, 0};

    // Nothing generated yet
    REQUIRE(!world.snapshotForMeshing(center, snapshot));

    // Plans are immutable prerequisites. An absent cube follows its generated
    // terrain silhouette: conservatively solid below the surface and air
    // above it, instead of presenting a full dark face while streaming.
    world.getChunk(center);
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({center.x + offsetX, center.z + offsetZ}));
        }
    }
    REQUIRE(world.snapshotForMeshing(center, snapshot));
    REQUIRE(snapshot.missingNeighborFaces == 0x3FU);
    const int32_t probeWorldY = center.y * CHUNK_EDGE + 8;
    const int32_t generatedCutoff = snapshot.generatedSurfaceCutoffAt(CHUNK_EDGE, 8);
    REQUIRE(generatedCutoff != MeshSnapshot::SKY_CUTOFF_UNKNOWN);
    REQUIRE(snapshot.at(CHUNK_EDGE, 8, 8) ==
            (probeWorldY < generatedCutoff ? BlockType::BEDROCK : BlockType::AIR));
    world.markChunkMeshed(center);
    REQUIRE_FALSE(world.getChunk(center)->needsMeshUpdate);
    REQUIRE(world.getChunk({1, center.y, 0}));
    REQUIRE(world.getChunk(center)->needsMeshUpdate);

    // Edge and corner cells affect baked corner accessibility and fluid corner heights,
    // so a complete 3x3x3 halo replaces every conservative placeholder.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                world.getChunk({center.x + offsetX, center.y + offsetY, center.z + offsetZ});
            }
        }
    }
    auto plusX = world.getChunk({1, center.y, 0});
    plusX->setBlock(0, 7, 8, BlockType::WATER);
    plusX->setFluidState(0, 7, 8, FluidState::flowing(5));
    plusX->setBlockLight(0, 7, 8, 9);
    REQUIRE(world.snapshotForMeshing(center, snapshot));
    REQUIRE(snapshot.missingNeighborFaces == 0);

    // Every padded face carries its neighbor's real border cells.
    auto minusX = world.getChunk({-1, center.y, 0});
    auto plusY = world.getChunk({0, center.y + 1, 0});
    auto minusY = world.getChunk({0, center.y - 1, 0});
    auto plusZ = world.getChunk({0, center.y, 1});
    auto minusZ = world.getChunk({0, center.y, -1});
    for (int coordinate = 0; coordinate < CHUNK_EDGE; coordinate += 5) {
        REQUIRE(snapshot.at(CHUNK_EDGE, coordinate, 8) == plusX->getBlock(0, coordinate, 8));
        REQUIRE(snapshot.at(-1, coordinate, 8) == minusX->getBlock(CHUNK_EDGE - 1, coordinate, 8));
        REQUIRE(snapshot.at(coordinate, CHUNK_EDGE, 8) == plusY->getBlock(coordinate, 0, 8));
        REQUIRE(snapshot.at(coordinate, -1, 8) == minusY->getBlock(coordinate, CHUNK_EDGE - 1, 8));
        REQUIRE(snapshot.at(coordinate, 8, CHUNK_EDGE) == plusZ->getBlock(coordinate, 8, 0));
        REQUIRE(snapshot.at(coordinate, 8, -1) == minusZ->getBlock(coordinate, 8, CHUNK_EDGE - 1));
    }
    REQUIRE(snapshot.at(CHUNK_EDGE, 7, 8) == BlockType::WATER);
    REQUIRE(snapshot.fluidAt(CHUNK_EDGE, 7, 8) == FluidState::flowing(5));
    REQUIRE(snapshot.lightAt(CHUNK_EDGE, 7, 8) == 9);
    REQUIRE(snapshot.skyCutoffAt(8, 8) != MeshSnapshot::SKY_CUTOFF_UNKNOWN);
}

TEST_CASE("Missing surface halos stay lit while underground openings remain dark",
          "[world][mesher][border][streaming][surface][underground]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    // The loaded column ends at world Y=67 while the arriving uphill column
    // continues through Y=71. Its four-block exposed silhouette should use a
    // normally lit terrain material. A solid roof isolates the three air
    // blocks below it, which still represent a cave opening and must remain
    // sealed and dark.
    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = BlockType::GRASS;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            snapshot.blocks[MeshSnapshot::index(x, 3, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t sealedVertices = 0;
    size_t litSurfaceVertices = 0;
    float highestCapY = 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        highestCapY = std::max(highestCapY, static_cast<float>(vertex.py));
        if (unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::STONE)) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++sealedVertices;
        } else if (unpackTextureLayer(vertex.faceAttr) == TEXTURE_LAYER_GRASS_SIDE) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++litSurfaceVertices;
        }
    }
    REQUIRE(sealedVertices == 3 * CHUNK_EDGE * 4);
    REQUIRE(litSurfaceVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(highestCapY == 8.0F);
    REQUIRE(sealedVertices + litSurfaceVertices < CHUNK_EDGE * CHUNK_EDGE * 4);
}

TEST_CASE("Missing caps recognize outdoor air beneath a generated overhang",
          "[world][mesher][border][streaming][surface][overhang]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = BlockType::GRASS;
    }
    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            snapshot.blocks[MeshSnapshot::index(x, 3, z)] = BlockType::STONE;
        }
    }

    // This column's generated top describes an overhang at world Y=71. Its
    // undercut air joins the neighboring outdoor column inside the snapshot,
    // so the three cells beneath the roof and four above it all need outdoor
    // lighting even though the undercut lies below its column cutoff.
    constexpr int overhangZ = 8;
    snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, overhangZ)] = 72;
    snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, overhangZ)] = 76;
    snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, overhangZ)] =
        BlockType::LIMESTONE;
    snapshot.blocks[MeshSnapshot::index(CHUNK_EDGE - 1, 7, overhangZ)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t litOverhangVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE) ||
            unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::LIMESTONE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++litOverhangVertices;
    }
    REQUIRE(litOverhangVertices == 7 * 4);
}

TEST_CASE("Edited roofs keep enclosed missing-neighbor caps dark",
          "[world][mesher][border][streaming][surface][roof][underground]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 73;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 72;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t enclosedVertices = 0;
    size_t litVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        if (unpackSkyLight(vertex.faceAttr) == 0) {
            REQUIRE(unpackTextureLayer(vertex.faceAttr) == static_cast<uint8_t>(BlockType::STONE));
            REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
            ++enclosedVertices;
        } else {
            ++litVertices;
        }
    }
    REQUIRE(enclosedVertices == 4 * CHUNK_EDGE * 4);
    REQUIRE(litVertices == 0);
}

TEST_CASE("Top-of-world roofs remain distinct from incomplete sky paths",
          "[world][mesher][border][streaming][surface][roof][limit]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, WORLD_MAX_CHUNK_Y, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 500;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = WORLD_MAX_Y + 1;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = WORLD_MAX_Y + 1;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = WORLD_MAX_Y + 1;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
            snapshot.blocks[MeshSnapshot::index(x, CHUNK_EDGE - 1, z)] = BlockType::STONE;
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t enclosedVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++enclosedVertices;
    }
    REQUIRE(enclosedVertices == 11 * CHUNK_EDGE * 4);
}

TEST_CASE("Lowered exact sky cutoffs light opened missing-neighbor caps",
          "[world][mesher][border][streaming][surface][edit]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.missingNeighborFaces = MeshSnapshot::MISSING_PLUS_X;

    for (int z = -1; z <= CHUNK_EDGE; ++z) {
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 96;
        snapshot.generatedSurfaceCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 80;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE - 1, z)] = 68;
        snapshot.skyCutoffY[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] = 80;
        snapshot.generatedSurfaceMaterial[MeshSnapshot::skyIndex(CHUNK_EDGE, z)] =
            BlockType::LIMESTONE;
        for (int x = -1; x <= CHUNK_EDGE; ++x) {
            for (int y = -1; y <= 3; ++y) {
                snapshot.blocks[MeshSnapshot::index(x, y, z)] = BlockType::STONE;
            }
        }
    }

    MeshScratch scratch;
    const MeshOutput mesh = LODMesher::buildMesh(snapshot, scratch);
    size_t litVertices = 0;
    for (const Vertex& vertex : mesh.vertices) {
        if (unpackFace(vertex.faceAttr) != FaceNormal::MINUS_X ||
            static_cast<float>(vertex.px) != static_cast<float>(CHUNK_EDGE)) {
            continue;
        }
        REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        REQUIRE(unpackCornerAO(vertex.faceAttr) == 3);
        ++litVertices;
    }
    REQUIRE(litVertices == 12 * CHUNK_EDGE * 4);
}

TEST_CASE("World setBlock marks boundary neighbors for remeshing", "[world][mesher][border]") {
    World world(7, 2);
    constexpr int32_t sectionY = 6;
    world.getChunk({0, sectionY, 0});
    world.getChunk({-1, sectionY, 0});
    world.getChunk({0, sectionY, -1});
    auto self = world.getChunk({0, sectionY, 0});
    auto negX = world.getChunk({-1, sectionY, 0});
    auto negZ = world.getChunk({0, sectionY, -1});

    self->needsMeshUpdate = false;
    negX->needsMeshUpdate = false;
    negZ->needsMeshUpdate = false;

    // Publication already settled the initial packed light. This interior edit
    // changes neither a halo block nor boundary light, so only its owner needs
    // a new mesh.
    world.setBlock(8, 100, 8, BlockType::STONE);
    REQUIRE(self->needsMeshUpdate);
    REQUIRE(!negX->needsMeshUpdate);
    REQUIRE(!negZ->needsMeshUpdate);

    // Clear every observed lighting invalidation before isolating geometric
    // boundary ownership. A boundary edit at local x == 0 must independently
    // remesh the -X neighbor and must not dirty the unrelated -Z neighbor.
    self->needsMeshUpdate = false;
    negX->needsMeshUpdate = false;
    negZ->needsMeshUpdate = false;
    world.setBlock(0, 100, 8, BlockType::STONE);
    REQUIRE(self->needsMeshUpdate);
    REQUIRE(negX->needsMeshUpdate);
    REQUIRE(!negZ->needsMeshUpdate);
}

TEST_CASE("Mesher: flora is skipped at coarse LODs", "[render][mesher][flora]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    for (int z = 0; z < CHUNK_DEPTH; ++z)
        for (int x = 0; x < CHUNK_WIDTH; ++x) {
            chunk.setBlock(x, 8, z, BlockType::GRASS);
            chunk.setBlock(x, 9, z, BlockType::TALL_GRASS);
        }

    LODMesher mesher;
    MeshOutput medium = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::MEDIUM));
    for (const Vertex& v : medium.vertices) {
        REQUIRE(unpackFace(v.faceAttr) != FaceNormal::CROSS);
        REQUIRE(unpackTextureLayer(v.faceAttr) != static_cast<uint8_t>(BlockType::TALL_GRASS));
    }
}

TEST_CASE("Mesher: vertical column merges side faces", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    // 4-block tall column of STONE at local y=6 through y=9
    chunk.setBlock(8, 6, 8, BlockType::STONE);
    chunk.setBlock(8, 7, 8, BlockType::STONE);
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    chunk.setBlock(8, 9, 8, BlockType::STONE);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // Top and bottom each form one quad. Every side forms one quad spanning
    // local y=6 through y=10, for 24 vertices and 36 indices total.

    REQUIRE(output.vertices.size() == 24);
    REQUIRE(output.indices.size() == 36);

    // Verify side faces span the full column height
    // Each side quad should span four local blocks.
    bool foundSideQuad = false;
    for (size_t i = 0; i + 3 < output.vertices.size(); ++i) {
        uint8_t ni = static_cast<uint8_t>(unpackFace(output.vertices[i].faceAttr));
        // Check side faces (face indices 0-3)
        if (ni <= 3 && static_cast<uint8_t>(unpackFace(output.vertices[i + 1].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 2].faceAttr)) == ni &&
            static_cast<uint8_t>(unpackFace(output.vertices[i + 3].faceAttr)) == ni) {
            // Check that the quad spans 4 units in Y
            float minY = std::min({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            float maxY = std::max({static_cast<float>(output.vertices[i].py),
                                   static_cast<float>(output.vertices[i + 1].py),
                                   static_cast<float>(output.vertices[i + 2].py),
                                   static_cast<float>(output.vertices[i + 3].py)});
            if (maxY - minY >= 3.5f) { // height=4, account for float16 precision
                foundSideQuad = true;
                break;
            }
        }
    }
    REQUIRE(foundSideQuad);
}

TEST_CASE("Mesher: produces mesh without side effects", "[render][mesher]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::STONE);
    chunk.needsMeshUpdate = true;
    REQUIRE(chunk.needsMeshUpdate == true);

    LODMesher mesher;
    MeshOutput mesh = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    // buildMesh is pure: it does not modify the chunk
    REQUIRE(chunk.needsMeshUpdate == true);
    REQUIRE(mesh.vertices.size() > 0u);
    REQUIRE(mesh.indices.size() > 0u);

    // Caller is responsible for marking the chunk as meshed
    chunk.setMeshed(true);
    chunk.needsMeshUpdate = false;
    REQUIRE(chunk.meshed == true);
    REQUIRE(chunk.needsMeshUpdate == false);
}

// ============================================================================
// Block texture mapping tests (no Metal device required)
// ============================================================================

TEST_CASE("Block textures: every block type maps to a valid layer", "[render][textures]") {
    for (int t = 0; t < static_cast<int>(BlockType::COUNT); ++t) {
        for (int f = 0; f < 6; ++f) {
            uint8_t layer = textureLayerFor(static_cast<BlockType>(t), static_cast<FaceNormal>(f));
            REQUIRE(layer < TEXTURE_LAYER_COUNT);
        }
    }
}

TEST_CASE("Block textures: grass uses per-face layers", "[render][textures]") {
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_Y) ==
            static_cast<uint8_t>(BlockType::GRASS));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Y) ==
            static_cast<uint8_t>(BlockType::DIRT));
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::PLUS_X) == TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(textureLayerFor(BlockType::GRASS, FaceNormal::MINUS_Z) == TEXTURE_LAYER_GRASS_SIDE);
}

TEST_CASE("UI icon vertices share one layout between C++ and Metal", "[render][ui]") {
    REQUIRE(sizeof(UIIconVertex) == 48);
    REQUIRE(offsetof(UIIconVertex, position) == 0);
    REQUIRE(offsetof(UIIconVertex, uv) == 8);
    REQUIRE(offsetof(UIIconVertex, tint) == 16);
    REQUIRE(offsetof(UIIconVertex, layer) == 32);
}

TEST_CASE("Item icon layers map the non-block range past the block layers", "[render][textures]") {
    REQUIRE(TEXTURE_LAYER_ITEM_FIRST == TEXTURE_LAYER_COUNT);
    REQUIRE(itemIconLayer(ItemType::STICK) == TEXTURE_LAYER_ITEM_FIRST);
    REQUIRE(itemIconLayer(static_cast<ItemType>(static_cast<uint16_t>(ItemType::COUNT) - 1)) ==
            TEXTURE_LAYER_ITEM_FIRST + NON_BLOCK_ITEM_COUNT - 1);
    REQUIRE(TEXTURE_LAYER_TOTAL == TEXTURE_LAYER_ITEM_FIRST + ITEM_ICON_COUNT);
    REQUIRE(TEXTURE_LAYER_TOTAL <= 255);
}

TEST_CASE("Block textures: workshop blocks use per-face layers", "[render][textures]") {
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::PLUS_Y) ==
            TEXTURE_LAYER_CRAFTING_TABLE_TOP);
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::MINUS_Y) ==
            static_cast<uint8_t>(BlockType::PLANKS));
    REQUIRE(textureLayerFor(BlockType::CRAFTING_TABLE, FaceNormal::PLUS_X) ==
            static_cast<uint8_t>(BlockType::CRAFTING_TABLE));
    for (BlockType furnace : {BlockType::FURNACE, BlockType::FURNACE_LIT}) {
        REQUIRE(textureLayerFor(furnace, FaceNormal::PLUS_Y) == TEXTURE_LAYER_FURNACE_TOP);
        REQUIRE(textureLayerFor(furnace, FaceNormal::MINUS_Y) == TEXTURE_LAYER_FURNACE_TOP);
        REQUIRE(textureLayerFor(furnace, FaceNormal::MINUS_Z) == static_cast<uint8_t>(furnace));
        REQUIRE(textureLayerFor(furnace, FaceNormal::MINUS_X) == TEXTURE_LAYER_FURNACE_SIDE);
        REQUIRE(textureLayerFor(furnace, FaceNormal::PLUS_Z) == TEXTURE_LAYER_FURNACE_SIDE);
    }
    REQUIRE(textureLayerFor(BlockType::CHEST, FaceNormal::MINUS_Z) ==
            static_cast<uint8_t>(BlockType::CHEST));
    REQUIRE(textureLayerFor(BlockType::CHEST, FaceNormal::PLUS_Z) == TEXTURE_LAYER_CHEST_SIDE);
    REQUIRE(textureLayerFor(BlockType::CHEST, FaceNormal::MINUS_X) == TEXTURE_LAYER_CHEST_SIDE);
    REQUIRE(textureLayerFor(BlockType::CHEST, FaceNormal::PLUS_Y) == TEXTURE_LAYER_CHEST_TOP);
    REQUIRE(itemIconRightFaceFor(BlockType::FURNACE) == FaceNormal::MINUS_Z);
    REQUIRE(itemIconRightFaceFor(BlockType::FURNACE_LIT) == FaceNormal::MINUS_Z);
    REQUIRE(itemIconRightFaceFor(BlockType::CHEST) == FaceNormal::MINUS_Z);
    REQUIRE(itemIconRightFaceFor(BlockType::STONE) == FaceNormal::PLUS_Z);
    REQUIRE(textureLayerFor(BlockType::TORCH, FaceNormal::PLUS_X) ==
            static_cast<uint8_t>(BlockType::TORCH));
}

TEST_CASE("Block textures: face attr pack/unpack round-trips", "[render][textures]") {
    for (int f = 0; f < 6; ++f) {
        for (uint8_t layer :
             {uint8_t{0}, uint8_t{7}, TEXTURE_LAYER_GRASS_SIDE, TEXTURE_LAYER_WHITE}) {
            for (uint8_t light : {uint8_t{0}, uint8_t{4}, uint8_t{15}}) {
                for (uint8_t ao : {uint8_t{0}, uint8_t{1}, uint8_t{2}, uint8_t{3}}) {
                    for (uint8_t blockLight : {uint8_t{0}, uint8_t{9}, uint8_t{15}}) {
                        for (bool emissive : {false, true}) {
                            for (uint8_t sway : {uint8_t{0}, uint8_t{1}, uint8_t{2}}) {
                                uint32_t attr = packFaceAttr(static_cast<FaceNormal>(f), layer,
                                                             light, ao, blockLight, emissive, sway);
                                REQUIRE(unpackFace(attr) == static_cast<FaceNormal>(f));
                                REQUIRE(unpackTextureLayer(attr) == layer);
                                REQUIRE(unpackSkyLight(attr) == light);
                                REQUIRE(unpackCornerAO(attr) == ao);
                                REQUIRE(unpackBlockLight(attr) == blockLight);
                                REQUIRE(unpackEmissive(attr) == emissive);
                                REQUIRE(unpackSway(attr) == sway);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Block textures: fluid metadata does not overlap shared face attributes",
          "[render][textures][water]") {
    constexpr uint8_t skyLight = 7;
    constexpr uint8_t blockLight = 11;
    const uint32_t attr = packFluidFaceAttr(FaceNormal::PLUS_Z, skyLight, 5, true, blockLight);

    REQUIRE(unpackFace(attr) == FaceNormal::PLUS_Z);
    REQUIRE(unpackTextureLayer(attr) == static_cast<uint8_t>(BlockType::WATER));
    REQUIRE(unpackSkyLight(attr) == skyLight);
    REQUIRE(unpackCornerAO(attr) == 3);
    REQUIRE(unpackBlockLight(attr) == blockLight);
    REQUIRE_FALSE(unpackEmissive(attr));
    REQUIRE(unpackSway(attr) == 0);
    REQUIRE(unpackFluidDirection(attr) == 5);
    REQUIRE(unpackFluidFalling(attr));
    REQUIRE((attr & 0x00FFFFFFU) == packFaceAttr(FaceNormal::PLUS_Z,
                                                 static_cast<uint8_t>(BlockType::WATER), skyLight,
                                                 3, blockLight));
}

TEST_CASE("Block definitions expose exhaustive lighting and sway traits",
          "[world][blocks][light]") {
    for (size_t index = 0; index < BLOCK_TYPE_COUNT; ++index) {
        const BlockType type = static_cast<BlockType>(index);
        const BlockDefinition& definition = blockDefinition(type);
        REQUIRE(blockLightEmission(type) == definition.lightEmission);
        REQUIRE(isEmissive(type) == definition.emissive);
        REQUIRE(swayClass(type) == definition.sway);
        REQUIRE(definition.lightEmission <= 15);
        REQUIRE(definition.sway <= 2);
        REQUIRE(definition.emissive == (definition.lightEmission > 0));
    }
    REQUIRE(blockLightEmission(BlockType::LAVA) == 15);
    REQUIRE(isEmissive(BlockType::LAVA));
    REQUIRE(swayClass(BlockType::ACACIA_LEAVES) == 2);
    REQUIRE(swayClass(BlockType::FLOWER_BLUE) == 1);
    REQUIRE(swayClass(BlockType::SUCCULENT) == 0);
}

TEST_CASE("Mesher: tags sway class for flora and leaves", "[render][mesher][sway]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(4, 8, 4, BlockType::STONE);
    chunk.setBlock(4, 9, 4, BlockType::TALL_GRASS);
    chunk.setBlock(8, 8, 8, BlockType::LEAVES);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool sawFlora = false, sawLeaves = false, sawStatic = false;
    for (const Vertex& v : output.vertices) {
        uint8_t layer = unpackTextureLayer(v.faceAttr);
        if (unpackFace(v.faceAttr) == FaceNormal::CROSS) {
            REQUIRE(unpackSway(v.faceAttr) == 1); // flora bends from the root
            sawFlora = true;
        } else if (layer == static_cast<uint8_t>(BlockType::LEAVES)) {
            REQUIRE(unpackSway(v.faceAttr) == 2); // canopy drifts whole-block
            sawLeaves = true;
        } else if (layer == static_cast<uint8_t>(BlockType::STONE)) {
            REQUIRE(unpackSway(v.faceAttr) == 0); // terrain never sways
            sawStatic = true;
        }
    }
    REQUIRE(sawFlora);
    REQUIRE(sawLeaves);
    REQUIRE(sawStatic);
}

TEST_CASE("Mesher: bakes lava block light and the emissive flag", "[render][mesher][light]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::LAVA);   // light source
    chunk.setBlock(10, 8, 8, BlockType::STONE); // a wall two blocks away
    LightEngine::computeSelfLight(chunk);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool foundLitStoneFace = false;
    bool foundEmissiveLava = false;
    for (const Vertex& v : output.vertices) {
        FaceNormal face = unpackFace(v.faceAttr);
        float x = static_cast<float>(v.px);
        // The stone's -X face (plane x = 10) samples the lit air at x = 9.
        if (face == FaceNormal::MINUS_X && x > 9.9f && x < 10.1f) {
            REQUIRE(unpackBlockLight(v.faceAttr) > 0);
            foundLitStoneFace = true;
        }
        if (unpackEmissive(v.faceAttr)) {
            foundEmissiveLava = true; // only lava sets the emissive bit
        }
    }
    REQUIRE(foundLitStoneFace);
    REQUIRE(foundEmissiveLava);
}

TEST_CASE("LightEngine: block light spills through all six cubic faces", "[world][light][cubic]") {
    struct FaceCase {
        size_t neighborIndex;
        std::array<int, 3> neighborCell;
        std::array<int, 3> borderCell;
        std::array<int, 3> inwardCell;
    };
    constexpr std::array<FaceCase, 6> cases{{
        {0, {15, 8, 8}, {0, 8, 8}, {1, 8, 8}},
        {1, {0, 8, 8}, {15, 8, 8}, {14, 8, 8}},
        {2, {8, 8, 15}, {8, 8, 0}, {8, 8, 1}},
        {3, {8, 8, 0}, {8, 8, 15}, {8, 8, 14}},
        {4, {8, 15, 8}, {8, 0, 8}, {8, 1, 8}},
        {5, {8, 0, 8}, {8, 15, 8}, {8, 14, 8}},
    }};

    for (const FaceCase& test : cases) {
        Chunk self(ChunkPos{0, 0, 0});
        Chunk neighbor(ChunkPos{0, 0, 0});
        neighbor.setBlockLight(test.neighborCell[0], test.neighborCell[1], test.neighborCell[2],
                               10);
        LightEngine::FaceNeighbors neighbors{};
        neighbors[test.neighborIndex] = &neighbor;

        REQUIRE(LightEngine::floodChunk(self, neighbors));
        REQUIRE(self.getBlockLight(test.borderCell[0], test.borderCell[1], test.borderCell[2]) ==
                9);
        REQUIRE(self.getBlockLight(test.inwardCell[0], test.inwardCell[1], test.inwardCell[2]) ==
                8);
    }
}

TEST_CASE("Snapshot mesher samples block light across a cubic halo",
          "[render][mesher][light][border]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // Smooth lighting averages a corner over the outward-plane 3x3 patch.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = packDerivedLight(0, 12);

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundLitBoundary = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 12);
            foundLitBoundary = true;
        }
    }
    REQUIRE(foundLitBoundary);
}

TEST_CASE("Mesher: baked corner AO darkens enclosed voxel corners", "[render][mesher][ao]") {
    // An L-shaped nook: a floor with two walls meeting at a corner. The floor
    // vertex tucked into the inner corner sees occluders on both sides and the
    // diagonal, so its baked AO is the lowest; a vertex out on the open floor
    // stays fully open (AO 3).
    Chunk chunk(ChunkPos{0, 4, 0});
    for (int x = 4; x <= 9; ++x)
        for (int z = 4; z <= 9; ++z)
            chunk.setBlock(x, 4, z, BlockType::STONE); // floor slab
    for (int z = 4; z <= 9; ++z)
        chunk.setBlock(4, 5, z, BlockType::STONE); // wall along -X edge
    for (int x = 4; x <= 9; ++x)
        chunk.setBlock(x, 5, 4, BlockType::STONE); // wall along -Z edge

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    uint8_t innerCornerAO = 3;
    uint8_t maxFloorAO = 0;
    bool foundInner = false;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        float x = static_cast<float>(v.px);
        float y = static_cast<float>(v.py);
        float z = static_cast<float>(v.pz);
        if (y < 4.5f || y > 5.5f)
            continue; // only the floor top at y = 5
        maxFloorAO = std::max(maxFloorAO, unpackCornerAO(v.faceAttr));
        // The concave corner vertex sits where the two walls meet (5,5,5)
        if (x > 4.9f && x < 5.1f && z > 4.9f && z < 5.1f) {
            innerCornerAO = std::min(innerCornerAO, unpackCornerAO(v.faceAttr));
            foundInner = true;
        }
    }
    REQUIRE(foundInner);
    REQUIRE(maxFloorAO == 3);    // open floor away from the walls stays lit
    REQUIRE(innerCornerAO == 0); // two walls + diagonal bury the tucked corner
}

TEST_CASE("Mesher: corner AO follows physical vertices on all six faces",
          "[render][mesher][ao][winding]") {
    struct FaceBasis {
        FaceNormal face;
        std::array<int, 3> normal;
        int tangentA;
        int tangentB;
    };
    constexpr std::array<FaceBasis, 6> faces{{
        {FaceNormal::PLUS_X, {1, 0, 0}, 1, 2},
        {FaceNormal::MINUS_X, {-1, 0, 0}, 1, 2},
        {FaceNormal::PLUS_Y, {0, 1, 0}, 0, 2},
        {FaceNormal::MINUS_Y, {0, -1, 0}, 0, 2},
        {FaceNormal::PLUS_Z, {0, 0, 1}, 0, 1},
        {FaceNormal::MINUS_Z, {0, 0, -1}, 0, 1},
    }};
    constexpr std::array<std::array<int, 2>, 4> cornerSigns{{
        {-1, -1},
        {-1, 1},
        {1, 1},
        {1, -1},
    }};

    MeshScratch scratch;
    for (const FaceBasis& basis : faces) {
        for (const auto& signs : cornerSigns) {
            MeshSnapshot snapshot;
            snapshot.clear();
            constexpr std::array<int, 3> center{8, 8, 8};
            snapshot.blocks[MeshSnapshot::index(center[0], center[1], center[2])] =
                BlockType::STONE;

            std::array<int, 3> exposure = center;
            for (int axis = 0; axis < 3; ++axis)
                exposure[axis] += basis.normal[axis];
            std::array<int, 3> sideA = exposure;
            std::array<int, 3> sideB = exposure;
            sideA[basis.tangentA] += signs[0];
            sideB[basis.tangentB] += signs[1];
            snapshot.blocks[MeshSnapshot::index(sideA[0], sideA[1], sideA[2])] = BlockType::STONE;
            snapshot.blocks[MeshSnapshot::index(sideB[0], sideB[1], sideB[2])] = BlockType::STONE;

            std::array<float, 3> target{8.0F, 8.0F, 8.0F};
            for (int axis = 0; axis < 3; ++axis) {
                if (basis.normal[axis] > 0)
                    target[axis] = 9.0F;
            }
            target[basis.tangentA] = signs[0] > 0 ? 9.0F : 8.0F;
            target[basis.tangentB] = signs[1] > 0 ? 9.0F : 8.0F;

            const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
            bool foundCorner = false;
            for (const Vertex& vertex : output.vertices) {
                if (unpackFace(vertex.faceAttr) == basis.face &&
                    static_cast<float>(vertex.px) == target[0] &&
                    static_cast<float>(vertex.py) == target[1] &&
                    static_cast<float>(vertex.pz) == target[2]) {
                    REQUIRE(unpackCornerAO(vertex.faceAttr) == 0);
                    foundCorner = true;
                }
            }
            REQUIRE(foundCorner);
        }
    }
}

TEST_CASE("Mesher: asymmetric AO triangulates across the brighter diagonal",
          "[render][mesher][ao]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(8, 8, 8)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(7, 9, 8)] = BlockType::STONE;
    snapshot.blocks[MeshSnapshot::index(8, 9, 7)] = BlockType::STONE;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundFace = false;
    for (size_t offset = 0; offset + 5 < output.opaqueIndexCount; offset += 6) {
        const Vertex& first = output.vertices[output.indices[offset]];
        const Vertex& second = output.vertices[output.indices[offset + 1]];
        const Vertex& third = output.vertices[output.indices[offset + 2]];
        const Vertex& fourth = output.vertices[output.indices[offset + 5]];
        if (unpackFace(first.faceAttr) != FaceNormal::PLUS_Y ||
            static_cast<float>(first.py) != 9.0F || static_cast<float>(second.py) != 9.0F ||
            static_cast<float>(third.py) != 9.0F || static_cast<float>(fourth.py) != 9.0F) {
            continue;
        }
        const uint8_t chosen = unpackCornerAO(first.faceAttr) + unpackCornerAO(third.faceAttr);
        const uint8_t alternate = unpackCornerAO(second.faceAttr) + unpackCornerAO(fourth.faceAttr);
        REQUIRE(chosen > alternate);
        foundFace = true;
    }
    REQUIRE(foundFace);
}

TEST_CASE("Mesher: opaque cover reduces skylight; non-opaque leaves do not", "[render][mesher]") {
    // Only OPAQUE blocks block the sky. A stone slab overhead shades the
    // ground below; a leaf canopy does not (its real cast shadow handles that,
    // and a column skylight shadow would double up under every tree).
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(4, 4, 8, BlockType::STONE);  // ground under stone cover
    chunk.setBlock(4, 8, 8, BlockType::STONE);  // opaque cover
    chunk.setBlock(12, 4, 8, BlockType::STONE); // ground under a leaf canopy
    chunk.setBlock(12, 8, 8, BlockType::LEAVES);

    LODMesher mesher;
    MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));

    bool foundShadedUnderStone = false;
    bool foundLitUnderLeaves = false;
    for (const Vertex& v : output.vertices) {
        if (unpackFace(v.faceAttr) != FaceNormal::PLUS_Y)
            continue;
        float x = static_cast<float>(v.px);
        float y = static_cast<float>(v.py);
        if (y > 4.5f && y < 5.5f && x > 4.4f && x < 5.6f) {
            REQUIRE(unpackSkyLight(v.faceAttr) < 15); // under opaque stone → shaded
            foundShadedUnderStone = true;
        }
        if (y > 4.5f && y < 5.5f && x > 12.4f && x < 13.6f) {
            REQUIRE(unpackSkyLight(v.faceAttr) == 15); // under leaves → still open
            foundLitUnderLeaves = true;
        }
    }
    REQUIRE(foundShadedUnderStone);
    REQUIRE(foundLitUnderLeaves);
}

TEST_CASE("Snapshot mesher uses a global sky cutoff above the cubic halo",
          "[render][mesher][light][skylight]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.blocks[MeshSnapshot::index(8, 4, 8)] = BlockType::STONE;
    // Fill every column read by the smoothed top-face corners so the fixture
    // represents one uniformly covered receiver.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx)
            snapshot.skyCutoffY[MeshSnapshot::skyIndex(8 + dx, 8 + dz)] = 96;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundShadedTop = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
            static_cast<float>(vertex.py) == 5.0F && static_cast<float>(vertex.px) >= 8.0F &&
            static_cast<float>(vertex.px) <= 9.0F && static_cast<float>(vertex.pz) >= 8.0F &&
            static_cast<float>(vertex.pz) <= 9.0F) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 0);
            foundShadedTop = true;
        }
    }
    REQUIRE(foundShadedTop);
}

TEST_CASE("Underground first meshes wait only for sparse skylight occupancy authority",
          "[world][mesher][light][skylight][streaming][publication][regression]") {
    World world(42, 4);
    constexpr int64_t worldX = 8;
    constexpr int64_t worldZ = 8;
    const int surfaceY = world.generator().surfaceYAt(worldX, worldZ);
    const int32_t surfaceSection = Chunk::worldToChunkY(surfaceY);
    const int32_t targetSection = surfaceSection - 4;
    REQUIRE(targetSection >= WORLD_MIN_CHUNK_Y);
    const ChunkPos target{0, targetSection, 0};

    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                REQUIRE(
                    world.getChunk({target.x + offsetX, target.y + offsetY, target.z + offsetZ}));
            }
        }
    }
    REQUIRE(world.getChunk({0, surfaceSection, 0}));

    MeshSnapshot separated;
    REQUIRE_FALSE(world.snapshotForMeshing(target, separated));

    // Complete only the sparse generated occupancy for the target and its
    // one-block meshing halo. No render or fixed-tick caller waits for this
    // work, and proven-empty vertical gaps remain absent.
    std::optional<ChunkPos> provenEmptyGap;
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto plan = world.generator().getColumnPlan({offsetX, offsetZ});
            REQUIRE(plan);
            for (const int32_t section : plan->exposedSections()) {
                if (section >= targetSection) {
                    REQUIRE(world.getChunk({offsetX, section, offsetZ}));
                }
            }
            if (offsetX == 0 && offsetZ == 0) {
                for (int32_t section = targetSection + 2; section <= WORLD_MAX_CHUNK_Y; ++section) {
                    if (!plan->exposesSection(section) && !world.isChunkLoaded({0, section, 0})) {
                        provenEmptyGap = ChunkPos{0, section, 0};
                        break;
                    }
                }
            }
        }
    }
    REQUIRE(provenEmptyGap);
    REQUIRE_FALSE(world.isChunkLoaded(*provenEmptyGap));
    MeshSnapshot connected;
    REQUIRE(world.snapshotForMeshing(target, connected));
    REQUIRE(connected.skyCutoffAt(8, 8) <= WORLD_MAX_Y + 1);
    REQUIRE_FALSE(world.isChunkLoaded(*provenEmptyGap));

    const auto firstVisibleLight = connected.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(target, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);
}

TEST_CASE("Generated opaque features extend the exact density sky cutoff",
          "[render][mesher][light][skylight][feature]") {
    World world(42, 4);
    constexpr int64_t MINIMUM_X = -27'392;
    constexpr int64_t MINIMUM_Z = -17'152;
    constexpr int64_t MAXIMUM_X = MINIMUM_X + 512;
    constexpr int64_t MAXIMUM_Z = MINIMUM_Z + 512;
    const std::vector<FarCanopy> canopies =
        world.generator().collectFarCanopies(MINIMUM_X, MINIMUM_Z, MAXIMUM_X, MAXIMUM_Z);
    const auto selected = std::ranges::find_if(canopies, [](const FarCanopy& canopy) {
        return canopy.logBlock != BlockType::AIR &&
               canopy.species != feature_generation::TreeSpecies::FALLEN_LOG;
    });
    REQUIRE(selected != canopies.end());
    const int64_t worldX = selected->x;
    const int64_t worldZ = selected->z;
    const ChunkPos target{Chunk::worldToChunk(worldX), Chunk::worldToChunkY(selected->baseY - 1),
                          Chunk::worldToChunk(worldZ)};
    const int32_t maximumTreeSection = Chunk::worldToChunkY(selected->topY);
    for (int offsetZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
         offsetZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetZ) {
        for (int offsetX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             offsetX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan({target.x + offsetX, target.z + offsetZ}));
        }
    }
    for (int32_t chunkY = target.y - 1; chunkY <= std::max(target.y + 1, maximumTreeSection);
         ++chunkY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                REQUIRE(world.getChunk({target.x + offsetX, chunkY, target.z + offsetZ}));
            }
        }
    }
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto skyPlan =
                world.generator().getColumnPlan({target.x + offsetX, target.z + offsetZ});
            REQUIRE(skyPlan);
            for (const int32_t section : skyPlan->exposedSections()) {
                if (section >= target.y) {
                    REQUIRE(world.getChunk({target.x + offsetX, section, target.z + offsetZ}));
                }
            }
        }
    }
    for (int pass = 0;
         pass < 64 && world.getStreamingWorkStats().publicationLightDeferredQueue != 0; ++pass) {
        world.reconcileLight(1'024);
    }
    REQUIRE(world.getStreamingWorkStats().publicationLightDeferredQueue == 0);
    REQUIRE(world.getStreamingWorkStats().publicationLightMaxSyncFloods <= 32);

    const auto plan = world.generator().getColumnPlan({target.x, target.z});
    const int plannedSurface =
        plan->surfaceY(Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ));
    const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(worldX, worldZ);
    REQUIRE(loadedTop);
    REQUIRE(*loadedTop > plannedSurface);
    REQUIRE(world.getBlockIfLoaded(worldX, *loadedTop, worldZ) == selected->logBlock);

    MeshSnapshot snapshot;
    REQUIRE(world.snapshotForMeshing(target, snapshot));
    REQUIRE(snapshot.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(worldX), Chunk::worldToLocal(worldZ))] == *loadedTop + 1);
    const auto firstVisibleLight = snapshot.packedLight;
    for (int pass = 0; pass < 4; ++pass)
        world.reconcileLight(64);
    MeshSnapshot settled;
    REQUIRE(world.snapshotForMeshing(target, settled));
    REQUIRE(settled.packedLight == firstVisibleLight);
}

TEST_CASE("Mesh skylight cutoffs follow opaque edits above the cubic halo",
          "[render][mesher][light][skylight][edit]") {
    World world(42, 4);
    constexpr int64_t WORLD_X = 0;
    constexpr int64_t WORLD_Z = 8;
    const int surfaceY = world.generator().surfaceYAt(WORLD_X, WORLD_Z);
    const ChunkPos surfaceCube{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(surfaceY),
                               Chunk::worldToChunk(WORLD_Z)};
    const int roofY = std::min((surfaceCube.y + 2) * CHUNK_EDGE + CHUNK_EDGE / 2, WORLD_MAX_Y - 1);
    REQUIRE(Chunk::worldToChunkY(roofY) > surfaceCube.y + 1);
    for (int offsetZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
         offsetZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetZ) {
        for (int offsetX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
             offsetX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++offsetX) {
            REQUIRE(world.generator().getColumnPlan(
                {surfaceCube.x + offsetX, surfaceCube.z + offsetZ}));
        }
    }

    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const ChunkPos neighbor{surfaceCube.x + offsetX, surfaceCube.y + offsetY,
                                        surfaceCube.z + offsetZ};
                if (neighbor.y >= WORLD_MIN_CHUNK_Y && neighbor.y <= WORLD_MAX_CHUNK_Y) {
                    REQUIRE(world.getChunk(neighbor));
                }
            }
        }
    }
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto skyPlan =
                world.generator().getColumnPlan({surfaceCube.x + offsetX, surfaceCube.z + offsetZ});
            REQUIRE(skyPlan);
            for (const int32_t section : skyPlan->exposedSections()) {
                if (section >= surfaceCube.y) {
                    REQUIRE(world.getChunk(
                        {surfaceCube.x + offsetX, section, surfaceCube.z + offsetZ}));
                }
            }
        }
    }
    REQUIRE(world.getChunk(Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(roofY),
                           Chunk::worldToChunk(WORLD_Z)));
    for (int pass = 0;
         pass < 64 && world.getStreamingWorkStats().publicationLightDeferredQueue != 0; ++pass) {
        world.reconcileLight(1'024);
    }
    REQUIRE(world.getStreamingWorkStats().publicationLightDeferredQueue == 0);
    REQUIRE(world.getStreamingWorkStats().publicationLightMaxSyncFloods <= 32);

    const ChunkPos negativeXSurfaceCube{surfaceCube.x - 1, surfaceCube.y, surfaceCube.z};
    world.markChunkMeshed(surfaceCube);
    world.markChunkMeshed(negativeXSurfaceCube);
    world.setBlock(WORLD_X, roofY, WORLD_Z, BlockType::STONE);
    REQUIRE(world.getChunk(surfaceCube)->needsMeshUpdate);
    REQUIRE(world.getChunk(negativeXSurfaceCube)->needsMeshUpdate);
    MeshSnapshot covered;
    REQUIRE(world.snapshotForMeshing(surfaceCube, covered));
    REQUIRE(covered.skyCutoffY[MeshSnapshot::skyIndex(Chunk::worldToLocal(WORLD_X),
                                                      Chunk::worldToLocal(WORLD_Z))] == roofY + 1);

    world.markChunkMeshed(surfaceCube);
    world.markChunkMeshed(negativeXSurfaceCube);
    world.setBlock(WORLD_X, roofY, WORLD_Z, BlockType::AIR);
    REQUIRE(world.getChunk(surfaceCube)->needsMeshUpdate);
    REQUIRE(world.getChunk(negativeXSurfaceCube)->needsMeshUpdate);
    const std::optional<int> restoredTop = world.surfaceHeightIfLoaded(WORLD_X, WORLD_Z);
    REQUIRE(restoredTop);
    MeshSnapshot opened;
    REQUIRE(world.snapshotForMeshing(surfaceCube, opened));
    REQUIRE(opened.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z))] == *restoredTop + 1);
}

TEST_CASE("Saved deep edits do not replace an unloaded generated sky cutoff",
          "[render][mesher][light][skylight][save]") {
    TempDir directory("saved_skylight_load_order");
    SaveManager saves(directory.path());
    constexpr uint32_t SEED = 42;
    constexpr int64_t WORLD_X = 8;
    constexpr int64_t WORLD_Z = 8;
    ChunkGenerator generator(SEED);
    const int surfaceY = generator.surfaceYAt(WORLD_X, WORLD_Z);
    const ChunkPos surfaceCube{Chunk::worldToChunk(WORLD_X), Chunk::worldToChunkY(surfaceY),
                               Chunk::worldToChunk(WORLD_Z)};
    const ChunkPos deepCube{surfaceCube.x, surfaceCube.y - 3, surfaceCube.z};
    REQUIRE(deepCube.y >= WORLD_MIN_CHUNK_Y);

    Chunk saved(deepCube);
    generator.generateCube(saved);
    saved.setBlock(Chunk::worldToLocal(WORLD_X), CHUNK_EDGE / 2, Chunk::worldToLocal(WORLD_Z),
                   BlockType::DIAMOND_ORE);
    saved.generated = true;
    saves.saveChunk(saved);
    REQUIRE(saves.flush());

    World world(SEED, 4);
    world.setSaveManager(&saves);
    REQUIRE(world.getChunk(deepCube));
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const ChunkPos neighbor{surfaceCube.x + offsetX, surfaceCube.y + offsetY,
                                        surfaceCube.z + offsetZ};
                if (neighbor.y >= WORLD_MIN_CHUNK_Y && neighbor.y <= WORLD_MAX_CHUNK_Y) {
                    REQUIRE(world.getChunk(neighbor));
                }
            }
        }
    }
    // Loading the saved deep section cannot prove the sky path above the
    // surface. Complete the sparse generated-occupancy authority for the
    // meshing halo while leaving every proven-empty vertical gap unloaded.
    for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const auto plan =
                world.generator().getColumnPlan({surfaceCube.x + offsetX, surfaceCube.z + offsetZ});
            REQUIRE(plan);
            for (const int32_t section : plan->exposedSections()) {
                if (section >= surfaceCube.y) {
                    REQUIRE(world.getChunk(
                        {surfaceCube.x + offsetX, section, surfaceCube.z + offsetZ}));
                }
            }
        }
    }

    const std::optional<int> loadedTop = world.surfaceHeightIfLoaded(WORLD_X, WORLD_Z);
    REQUIRE(loadedTop);
    REQUIRE(*loadedTop > (deepCube.y + 1) * CHUNK_EDGE);
    MeshSnapshot snapshot;
    bool ready = world.snapshotForMeshing(surfaceCube, snapshot);
    for (int pass = 0; pass < 64 && !ready; ++pass) {
        world.reconcileLight(64);
        ready = world.snapshotForMeshing(surfaceCube, snapshot);
    }
    REQUIRE(ready);
    REQUIRE(snapshot.skyCutoffY[MeshSnapshot::skyIndex(
                Chunk::worldToLocal(WORLD_X), Chunk::worldToLocal(WORLD_Z))] == *loadedTop + 1);
}

namespace {

struct alignas(4) TexturePixel {
    uint8_t b;
    uint8_t g;
    uint8_t r;
    uint8_t a;
};

static_assert(sizeof(TexturePixel) == 4);

std::vector<TexturePixel> readBlockTextureMip(id<MTLTexture> texture, uint8_t layer,
                                              uint32_t mipLevel) {
    const uint32_t edge = BlockTextureArray::TILE_SIZE >> mipLevel;
    std::vector<TexturePixel> pixels(edge * edge);
    [texture getBytes:pixels.data()
          bytesPerRow:edge * sizeof(TexturePixel)
        bytesPerImage:edge * edge * sizeof(TexturePixel)
           fromRegion:MTLRegionMake2D(0, 0, edge, edge)
          mipmapLevel:mipLevel
                slice:layer];
    return pixels;
}

uint64_t blockTextureHash(const BlockTextureArray& textures) {
    constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;
    for (uint8_t layer = 0; layer < TEXTURE_LAYER_COUNT; ++layer) {
        for (uint32_t mipLevel = 0; mipLevel < BlockTextureArray::MIP_LEVEL_COUNT; ++mipLevel) {
            const std::vector<TexturePixel> pixels =
                readBlockTextureMip(textures.texture(), layer, mipLevel);
            for (const TexturePixel& pixel : pixels) {
                const auto* bytes = reinterpret_cast<const uint8_t*>(&pixel);
                for (uint32_t component = 0; component < sizeof(TexturePixel); ++component) {
                    hash ^= bytes[component];
                    hash *= FNV_PRIME;
                }
            }
        }
    }
    return hash;
}

uint32_t coveredTextureTexels(const std::vector<TexturePixel>& pixels) {
    return static_cast<uint32_t>(std::count_if(pixels.begin(), pixels.end(),
                                               [](TexturePixel pixel) { return pixel.a >= 128; }));
}

} // namespace

TEST_CASE("Block textures: extra layers extend past the block types", "[render][textures]") {
    REQUIRE(TEXTURE_LAYER_GRASS_SIDE == static_cast<uint8_t>(BlockType::COUNT));
    REQUIRE(TEXTURE_LAYER_COUNT > TEXTURE_LAYER_GRASS_SIDE);
    REQUIRE(BlockTextureArray::TILE_SIZE == 16);
    REQUIRE(BlockTextureArray::MIP_LEVEL_COUNT == 5);
}

TEST_CASE("Block textures upload a complete deterministic mip chain", "[render][textures][mip]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    BlockTextureArray first(device);
    BlockTextureArray second(device);
    REQUIRE(first.texture().mipmapLevelCount == BlockTextureArray::MIP_LEVEL_COUNT);
    REQUIRE(first.texture().arrayLength == TEXTURE_LAYER_TOTAL);

    const uint64_t firstHash = blockTextureHash(first);
    REQUIRE(blockTextureHash(second) == firstHash);
    REQUIRE(firstHash == 0x8675a64efe4445cbULL);
}

TEST_CASE("Block texture mips preserve alpha-tested flora coverage", "[render][textures][mip]") {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    BlockTextureArray textures(device);

    constexpr std::array CUTOUT_LAYERS{BlockType::TALL_GRASS, BlockType::LILY_PAD,
                                       BlockType::LEAVES};
    constexpr uint32_t BASE_TEXELS = BlockTextureArray::TILE_SIZE * BlockTextureArray::TILE_SIZE;
    for (BlockType block : CUTOUT_LAYERS) {
        const auto base = readBlockTextureMip(textures.texture(), static_cast<uint8_t>(block), 0);
        const uint32_t baseCovered = coveredTextureTexels(base);
        REQUIRE(baseCovered > 0);
        REQUIRE(baseCovered < BASE_TEXELS);

        for (uint32_t mipLevel = 1; mipLevel < BlockTextureArray::MIP_LEVEL_COUNT; ++mipLevel) {
            const auto mip =
                readBlockTextureMip(textures.texture(), static_cast<uint8_t>(block), mipLevel);
            uint32_t expectedCovered = static_cast<uint32_t>(
                (static_cast<uint64_t>(baseCovered) * mip.size() + BASE_TEXELS / 2) / BASE_TEXELS);
            expectedCovered = std::max(expectedCovered, 1U);
            REQUIRE(coveredTextureTexels(mip) == expectedCovered);
            for (TexturePixel pixel : mip) {
                if (pixel.a >= 128)
                    REQUIRE(pixel.g > 0);
            }
        }
    }
}

// ============================================================================
// MegaBuffer Constant Tests (no Metal device required)
// ============================================================================

TEST_CASE("MegaBuffer alignment is power of 2", "[render][megabuffer]") {
    uint64_t align = MegaBuffer::ALIGNMENT;
    REQUIRE(align > 0);
    REQUIRE((align & (align - 1)) == 0); // Power of 2 check
    REQUIRE(align == 256);
}

TEST_CASE("MegaBuffer alignUp rounds up to alignment", "[render][megabuffer]") {
    // Test alignment behavior with known values
    // alignUp(x) = (x + 255) & ~255
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    REQUIRE(alignUp(0) == 0);
    REQUIRE(alignUp(1) == 256);
    REQUIRE(alignUp(256) == 256);
    REQUIRE(alignUp(257) == 512);
    REQUIRE(alignUp(512) == 512);
    REQUIRE(alignUp(1000) == 1024);
    REQUIRE(alignUp(16 * sizeof(Vertex)) == 256); // 256 bytes, already aligned
    REQUIRE(alignUp(17 * sizeof(Vertex)) == 512); // 272 bytes, rounds to 512
}

TEST_CASE("MegaBuffer vertex allocation size calculation", "[render][megabuffer]") {
    auto alignUp = [](uint64_t value) -> uint64_t {
        return (value + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };

    // 100 vertices × 16 bytes = 1600 bytes → aligned to 1792 (7 × 256)
    uint64_t vertexBytes = alignUp(100 * sizeof(Vertex));
    REQUIRE(vertexBytes >= 1600);
    REQUIRE(vertexBytes % MegaBuffer::ALIGNMENT == 0);

    // 1000 indices × 4 bytes = 4000 bytes → aligned to 4096 (16 × 256)
    uint64_t indexBytes = alignUp(1000 * sizeof(uint32_t));
    REQUIRE(indexBytes >= 4000);
    REQUIRE(indexBytes % MegaBuffer::ALIGNMENT == 0);
}

// ---- Day/Night Cycle Tests (Task 6.4-6.5) ----

TEST_CASE("Day/night cycle: sun position at noon", "[phase6][daynight]") {
    // At noon: worldTime = 6000 (25% of 24000)
    // orbitalAngle = 0.25 * 2*PI = PI/2
    // sunDirection = (cos(PI/2), sin(PI/2), 0.3) = (0, 1, 0.3)
    uint64_t worldTime = 6000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    // At noon: cos(PI/2) ≈ 0, sin(PI/2) = 1
    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at sunset", "[phase6][daynight]") {
    // At sunset: worldTime = 12000 (50% of 24000)
    // orbitalAngle = 0.5 * 2*PI = PI
    // sunDirection = (cos(PI), sin(PI), 0.3) = (-1, 0, 0.3)
    uint64_t worldTime = 12000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(-1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at midnight", "[phase6][daynight]") {
    // At midnight: worldTime = 18000 (75% of 24000)
    // orbitalAngle = 0.75 * 2*PI = 3PI/2
    // sunDirection = (cos(3PI/2), sin(3PI/2), 0.3) = (0, -1, 0.3)
    uint64_t worldTime = 18000;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(0.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(-1.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun position at dawn", "[phase6][daynight]") {
    // At dawn: worldTime = 0 (or 24000)
    // orbitalAngle = 0
    // sunDirection = (cos(0), sin(0), 0.3) = (1, 0, 0.3)
    uint64_t worldTime = 0;
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    float sunX = std::cos(orbitalAngle);
    float sunY = std::sin(orbitalAngle);

    REQUIRE(sunX == Catch::Approx(1.f).margin(0.001f));
    REQUIRE(sunY == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: world time wraps at day boundary", "[phase6][daynight]") {
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // worldTime = 48000 (2 days) should wrap to same position as 0
    uint64_t worldTime = 48000;
    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    REQUIRE(dayFraction == Catch::Approx(0.f).margin(0.001f));
}

TEST_CASE("Day/night cycle: sun elevation drives ambient brightness", "[phase6][daynight]") {
    // Test that sun elevation at noon produces higher ambient than at midnight
    auto computeAmbient = [](uint64_t worldTime) -> float {
        static constexpr uint64_t TICKS_PER_DAY = 24000;
        float dayFraction =
            static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
        float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);
        float sunElevation = std::sin(orbitalAngle);

        float ambientDay = 0.35f;
        float ambientNight = 0.1f;
        float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
        return ambientNight + (ambientDay - ambientNight) * ambientT;
    };

    float ambientNoon = computeAmbient(6000);
    float ambientMidnight = computeAmbient(18000);

    // Noon ambient should be higher than midnight
    REQUIRE(ambientNoon > ambientMidnight);
    REQUIRE(ambientNoon == Catch::Approx(0.35f).margin(0.01f));
    REQUIRE(ambientMidnight == Catch::Approx(0.1f).margin(0.01f));
}

// ============================================================================
// Phase 8: Post-Processing, Audio, Performance HUD Tests
// ============================================================================

// ---- Bloom Tests ----

// Hable "Uncharted 2" filmic tonemap replicated from post.metal (the
// composite owns tonemapping; this replaced Uchimura, whose shoulder
// plateaued the HDR sun to flat white within a few stops). Pins the curve's
// contract: black stays black, mids survive the fixed 2x gain near identity,
// highlights keep compressing across many stops, and it never decreases.
static float hableFilmicToneMap(float x) {
    auto curve = [](float v) {
        const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
    };
    const float W = 11.2f;             // linear white point
    return curve(x * 2.0f) / curve(W); // 2x gain matches displayColor
}

TEST_CASE("Post: filmic tone mapping curve", "[hdr][post]") {
    // Black in, black out
    REQUIRE(hableFilmicToneMap(0.0f) == Catch::Approx(0.0f).margin(0.001f));

    // The raw curve sits about a stop under identity at the mids; the
    // exposure key (0.85 in encodeExposure) compensates in the live path
    float atMid = hableFilmicToneMap(0.5f);
    REQUIRE(atMid > 0.22f);
    REQUIRE(atMid < 0.55f);

    // HDR highlights compress below display max and keep separating up to
    // the white point (5.6 scene units after the 2x gain); the auto-exposure
    // stop-down keeps the HDR-8 sun disc below it in the live path
    REQUIRE(hableFilmicToneMap(4.0f) < 1.0f);
    REQUIRE(hableFilmicToneMap(4.0f) - hableFilmicToneMap(2.0f) > 0.05f);
    REQUIRE(hableFilmicToneMap(8.0f) >= 1.0f); // past white: display max by design

    // Monotonically increasing across the range
    REQUIRE(hableFilmicToneMap(0.2f) < hableFilmicToneMap(0.5f));
    REQUIRE(hableFilmicToneMap(0.5f) < hableFilmicToneMap(1.0f));
    REQUIRE(hableFilmicToneMap(1.0f) < hableFilmicToneMap(2.0f));
}

TEST_CASE("Post: vibrance boosts low-saturation colors more than saturated ones", "[hdr][post]") {
    auto luma = [](float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; };
    // Vibrance boost factor from post.metal: vibrance * (1 - saturation)
    auto satBoost = [](float mx, float mn, float vibrance) {
        return vibrance * (1.0f - std::clamp(mx - mn, 0.0f, 1.0f));
    };
    const float vibrance = 0.5f;
    // A near-gray pixel (low saturation) gets a larger boost than a vivid one
    float grayBoost = satBoost(0.55f, 0.45f, vibrance); // sat 0.1
    float vividBoost = satBoost(0.9f, 0.1f, vibrance);  // sat 0.8
    REQUIRE(grayBoost > vividBoost);
    // Fully saturated → no boost
    REQUIRE(satBoost(1.0f, 0.0f, vibrance) == Catch::Approx(0.0f));
    (void)luma;
}

TEST_CASE("Bloom extract threshold passes bright pixels and blocks dark pixels",
          "[phase8][bloom]") {
    auto softThreshold = [](float luminance, float threshold) -> float {
        float low = threshold - 0.5f;
        float high = threshold + 0.5f;
        if (luminance <= low)
            return 0.0f;
        if (luminance >= high)
            return 1.0f;
        return (luminance - low) / (high - low);
    };

    // Dark pixel (luminance 0.2) with threshold 1.0 → blocked
    REQUIRE(softThreshold(0.2f, 1.0f) == Catch::Approx(0.0f));

    // Bright pixel (luminance 1.5) with threshold 1.0 → passes
    REQUIRE(softThreshold(1.5f, 1.0f) == Catch::Approx(1.0f));

    // Edge pixel (luminance 1.0) with threshold 1.0 → 0.5
    REQUIRE(softThreshold(1.0f, 1.0f) == Catch::Approx(0.5f));

    // Very bright pixel (luminance 3.0) → passes fully
    REQUIRE(softThreshold(3.0f, 1.0f) == Catch::Approx(1.0f));
}

TEST_CASE("Bloom: blur kernel weights are positive and symmetric", "[phase8][bloom]") {
    // 8-tap Kawase blur weights (normalized in shader by dividing by sum)
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // All weights are positive
    for (int i = 0; i < 8; ++i) {
        REQUIRE(weights[i] > 0.0f);
    }

    // Symmetric: first and last match, inner pairs match
    REQUIRE(weights[0] == weights[7]);
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);

    // Sum is used for normalization in shader
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        sum += weights[i];
    }
    REQUIRE(sum > 0.0f); // Non-zero for valid normalization
}

TEST_CASE("Bloom: blur kernel is symmetric", "[phase8][bloom]") {
    float weights[8] = {0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f};

    // First and last should match
    REQUIRE(weights[0] == weights[7]);
    // Inner pairs should match
    REQUIRE(weights[1] == weights[6]);
    REQUIRE(weights[2] == weights[5]);
    REQUIRE(weights[3] == weights[4]);
}

// ---- Fog Tests ----

TEST_CASE("Fog: exponential fog factor at various distances", "[phase8][fog]") {
    float density = 0.0003f;

    auto fogFactor = [](float distance, float density) -> float {
        return 1.0f - std::exp(-density * distance);
    };

    // At distance 0: no fog
    REQUIRE(fogFactor(0.0f, density) == Catch::Approx(0.0f).epsilon(0.0001f));

    // At distance 100: slight fog
    float f100 = fogFactor(100.0f, density);
    REQUIRE(f100 > 0.0f);
    REQUIRE(f100 < 0.5f);

    // At distance 1000: significant fog
    float f1000 = fogFactor(1000.0f, density);
    REQUIRE(f1000 > 0.2f);
    REQUIRE(f1000 < 0.5f);

    // At distance 5000: very foggy
    float f5000 = fogFactor(5000.0f, density);
    REQUIRE(f5000 > 0.7f);

    // Fog factor increases monotonically with distance
    REQUIRE(fogFactor(100.0f, density) < fogFactor(500.0f, density));
    REQUIRE(fogFactor(500.0f, density) < fogFactor(1000.0f, density));
}

TEST_CASE("Fog: fog color mixing", "[phase8][fog]") {
    struct F3 {
        float x, y, z;
    };

    auto mixFog = [](float fogFactor, F3 fogColor, F3 litColor) -> F3 {
        // fogFactor: 0 = fully fogged, 1 = fully lit
        // mix(fogColor, litColor, fogFactor) = fogColor*(1-fogFactor) + litColor*fogFactor
        return {
            fogColor.x * (1.0f - fogFactor) + litColor.x * fogFactor,
            fogColor.y * (1.0f - fogFactor) + litColor.y * fogFactor,
            fogColor.z * (1.0f - fogFactor) + litColor.z * fogFactor,
        };
    };

    F3 fogColor{0.5f, 0.7f, 0.8f}; // Sky-like
    F3 litColor{0.3f, 0.3f, 0.3f}; // Dark stone

    // No fog (factor=1): fully lit
    auto noFog = mixFog(1.0f, fogColor, litColor);
    REQUIRE(noFog.x == Catch::Approx(litColor.x));

    // Full fog (factor=0): fully fogged
    auto fullFog = mixFog(0.0f, fogColor, litColor);
    REQUIRE(fullFog.x == Catch::Approx(fogColor.x));

    // Half fog: blend
    auto halfFog = mixFog(0.5f, fogColor, litColor);
    REQUIRE(halfFog.x == Catch::Approx((fogColor.x + litColor.x) * 0.5f));
}

// ---- Cloud Tests ----

TEST_CASE("Clouds: noise threshold for cloud generation", "[phase8][clouds]") {
    // Cloud threshold 0.4: noise values above this render as clouds
    float threshold = 0.4f;

    auto cloudMask = [](float noise, float threshold) -> float {
        float low = threshold - 0.1f;
        float high = threshold + 0.1f;
        if (noise <= low)
            return 0.0f;
        if (noise >= high)
            return 1.0f;
        return (noise - low) / (high - low);
    };

    // Low noise → no cloud
    REQUIRE(cloudMask(0.2f, threshold) == Catch::Approx(0.0f));

    // High noise → full cloud
    REQUIRE(cloudMask(0.6f, threshold) == Catch::Approx(1.0f));

    // At threshold → partial cloud
    REQUIRE(cloudMask(0.4f, threshold) == Catch::Approx(0.5f));
}

TEST_CASE("Clouds: wind offset calculation", "[phase8][clouds]") {
    // Wind speed: 0.02 blocks/tick
    float windSpeed = 0.02f;

    auto windOffset = [](uint64_t worldTime, float windSpeed) -> float {
        return static_cast<float>(worldTime) * windSpeed;
    };

    // At time 0: no offset
    REQUIRE(windOffset(0, windSpeed) == Catch::Approx(0.0f));

    // At time 1000: offset = 20
    REQUIRE(windOffset(1000, windSpeed) == Catch::Approx(20.0f));

    // At time 5000: offset = 100
    REQUIRE(windOffset(5000, windSpeed) == Catch::Approx(100.0f));

    // Monotonically increasing
    REQUIRE(windOffset(100, windSpeed) < windOffset(200, windSpeed));
}

TEST_CASE("Weather uses v4 bounds coordinates and cold biomes",
          "[render][weather][v4][regression]") {
    REQUIRE_FALSE(weatherParticleBelowWorld(static_cast<float>(WORLD_MIN_Y)));
    REQUIRE_FALSE(weatherParticleBelowWorld(-1.0F));
    REQUIRE(weatherParticleBelowWorld(static_cast<float>(WORLD_MIN_Y) - 0.25F));

    REQUIRE(weatherBlockCoordinate(-16.25F) == -17);
    REQUIRE(weatherBlockCoordinate(3'000'000'000.0F) == 3'000'000'000LL);
    REQUIRE(weatherBlockCoordinate(-3'000'000'000.0F) == -3'000'000'000LL);
}

// ---- Shared shader struct layout pins ----
// shader_types.hpp is compiled by BOTH clang++ and the Metal compiler; simd
// types have the same layout in each. These pins catch accidental drift
// (reordered fields, ad-hoc padding) that previously corrupted fog, camera
// position, sky colors, and particle data.

TEST_CASE("Shader types: Uniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(FoliageWindUniforms) == 16);
    REQUIRE(offsetof(FoliageWindUniforms, direction) == 0);
    REQUIRE(offsetof(FoliageWindUniforms, speedBlocksPerSecond) == 8);
    REQUIRE(offsetof(FoliageWindUniforms, strength) == 12);
    REQUIRE(sizeof(Uniforms) == 320);
    REQUIRE(offsetof(Uniforms, sunDirection) == 192);
    REQUIRE(offsetof(Uniforms, fogColor) == 240);
    REQUIRE(offsetof(Uniforms, fogDensity) == 256);
    REQUIRE(offsetof(Uniforms, cameraPosition) == 272);
    REQUIRE(offsetof(Uniforms, foliageWind) == 288);
    REQUIRE(offsetof(Uniforms, time) == 304);
    REQUIRE(offsetof(Uniforms, wetness) == 308);
    REQUIRE(alignof(Uniforms) == 16);
    REQUIRE(sizeof(ChunkOrigin) == 48);
    REQUIRE(offsetof(ChunkOrigin, farMetadata) == 32);
    REQUIRE(sizeof(FarTerrainOwnershipUniforms) == 576);
    REQUIRE(offsetof(FarTerrainOwnershipUniforms, readyColumnMasks) == 0);
    REQUIRE(offsetof(FarTerrainOwnershipUniforms, floraReadyColumnMasks) == 288);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_WORD_COUNT * FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD ==
                   FAR_TERRAIN_EXACT_COLUMNS_PER_TILE * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE *
                       FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR ==
                   FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
    STATIC_REQUIRE(FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE * FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE ==
                   FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT);
    STATIC_REQUIRE(sizeof(FarTerrainOwnershipUniforms) ==
                   2 * FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT *
                       FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE * sizeof(simd_uint4));
    STATIC_REQUIRE(FarTerrainExactHandoff::COLUMN_MASK_WORD_COUNT ==
                   FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
}

TEST_CASE("Shader types: ShadowUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ShadowPassUniforms) == 112);
    REQUIRE(offsetof(ShadowPassUniforms, projectionOrigin) == 64);
    REQUIRE(offsetof(ShadowPassUniforms, foliageWind) == 80);
    REQUIRE(offsetof(ShadowPassUniforms, time) == 96);
    REQUIRE(sizeof(ShadowCascadeUniforms) == 112);
    REQUIRE(offsetof(ShadowCascadeUniforms, projectionOrigin) == 64);
    REQUIRE(offsetof(ShadowCascadeUniforms, depthRange) == 80);
    REQUIRE(offsetof(ShadowCascadeUniforms, samplingParams) == 96);
    REQUIRE(sizeof(ShadowUniforms) == 592);
    REQUIRE(offsetof(ShadowUniforms, cameraPositionAndStrength) == 560);
    REQUIRE(offsetof(ShadowUniforms, cameraForwardAndPadding) == 576);
    REQUIRE(SHADOW_DETAILED_CASCADE_COUNT == 4);
    REQUIRE(SHADOW_CASCADE_COUNT == 5);
    REQUIRE(SHADOW_HORIZON_CASCADE_INDEX == 4);
}

TEST_CASE("Dynamic object models carry fixed-tick packed lighting",
          "[render][shader-types][entity][light]") {
    REQUIRE(sizeof(EntityModel) == 80);
    REQUIRE(offsetof(EntityModel, model) == 0);
    REQUIRE(offsetof(EntityModel, lighting) == 64);

    constexpr uint8_t PACKED_LIGHT = 0xB5;
    STATIC_REQUIRE(FULL_SKY_PACKED_LIGHT == 0xF0);
    STATIC_REQUIRE(derivedSkyLight(FULL_SKY_PACKED_LIGHT) == MAX_DERIVED_LIGHT_LEVEL);
    STATIC_REQUIRE(derivedBlockLight(FULL_SKY_PACKED_LIGHT) == 0);
    STATIC_REQUIRE(normalizedDerivedLight(MAX_DERIVED_LIGHT_LEVEL) == 1.0F);
    REQUIRE(Entity(1, EntityType::SHEEP, Vec3{}).renderPackedLight == FULL_SKY_PACKED_LIGHT);
    REQUIRE(Boat{}.renderPackedLight == FULL_SKY_PACKED_LIGHT);
    REQUIRE(ItemEntity{}.renderPackedLight == FULL_SKY_PACKED_LIGHT);
    REQUIRE(dynamicObjectSkyLight(PACKED_LIGHT) == Catch::Approx(11.0F / 15.0F));
    REQUIRE(dynamicObjectBlockLight(PACKED_LIGHT) == Catch::Approx(5.0F / 15.0F));
    const simd_float4 lighting = dynamicObjectLighting(PACKED_LIGHT);
    REQUIRE(lighting.x == Catch::Approx(11.0F / 15.0F));
    REQUIRE(lighting.y == Catch::Approx(5.0F / 15.0F));
    REQUIRE(lighting.z == 1.0F);
    REQUIRE(lighting.w == 0.0F);
}

TEST_CASE("Deep-night SSGI preserves emissive sources while gating sky-lit bounce",
          "[render][indirect][emissive][night][regression]") {
    constexpr float ACCESS = 9.0F / 15.0F;
    const float ordinaryAlpha = screenSpaceSurfaceDataAlpha(ACCESS, false);
    REQUIRE_FALSE(screenSpaceSurfaceHasNightPersistentLight(ordinaryAlpha));
    REQUIRE(screenSpaceSurfaceAmbientAccess(ordinaryAlpha) == Catch::Approx(ACCESS).margin(0.01F));
    REQUIRE(screenSpaceBounceSourceScale(false, 0.0F) == Catch::Approx(0.0F));
    REQUIRE(screenSpaceBounceSourceScale(false, 0.4F) == Catch::Approx(0.4F));

    for (const BlockType source : {BlockType::TORCH, BlockType::FURNACE_LIT, BlockType::LAVA}) {
        REQUIRE(isEmissive(source));
        REQUIRE(screenSpaceNightPersistentSource(1.0F, 0.0F));
        const float packed =
            screenSpaceSurfaceDataAlpha(ACCESS, screenSpaceNightPersistentSource(1.0F, 0.0F));
        REQUIRE(screenSpaceSurfaceHasNightPersistentLight(packed));
        REQUIRE(screenSpaceSurfaceAmbientAccess(packed) == Catch::Approx(ACCESS).margin(0.01F));
        REQUIRE(screenSpaceBounceSourceScale(true, 0.0F) == Catch::Approx(1.0F));
    }

    REQUIRE_FALSE(isEmissive(BlockType::STONE));
    REQUIRE(screenSpaceNightPersistentSource(0.0F, 4.0F / 15.0F));
    const float torchLitWall =
        screenSpaceSurfaceDataAlpha(ACCESS, screenSpaceNightPersistentSource(0.0F, 4.0F / 15.0F));
    REQUIRE(screenSpaceSurfaceHasNightPersistentLight(torchLitWall));
    REQUIRE(screenSpaceBounceSourceScale(screenSpaceSurfaceHasNightPersistentLight(torchLitWall),
                                         0.0F) == Catch::Approx(1.0F));
}

TEST_CASE("Mat4 orthographic maps near->0 and far->1 (Metal depth)", "[common][math]") {
    Mat4 ortho = Mat4::orthographic(-10.f, 10.f, -10.f, 10.f, 0.f, 100.f);
    // A point at view-space z = -near (0) maps to NDC z = 0
    Vec3 nearPt = ortho.transformVec3({0.f, 0.f, 0.f});
    REQUIRE(nearPt.z == Catch::Approx(0.f).margin(1e-5));
    // A point at view-space z = -far maps to NDC z = 1
    Vec3 farPt = ortho.transformVec3({0.f, 0.f, -100.f});
    REQUIRE(farPt.z == Catch::Approx(1.f).margin(1e-5));
    // x/y map the ortho extents to [-1, 1]
    REQUIRE(ortho.transformVec3({10.f, 0.f, -1.f}).x == Catch::Approx(1.f).margin(1e-5));
    REQUIRE(ortho.transformVec3({-10.f, 0.f, -1.f}).x == Catch::Approx(-1.f).margin(1e-5));
}

TEST_CASE("Shader types: SkyUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(SkyUniforms) == 176);
    REQUIRE(offsetof(SkyUniforms, sunDirection) == 48);
    REQUIRE(offsetof(SkyUniforms, moonDirection) == 64);
    REQUIRE(offsetof(SkyUniforms, moonColor) == 96);
    REQUIRE(offsetof(SkyUniforms, zenithColor) == 112);
    REQUIRE(offsetof(SkyUniforms, visibilityAndPhase) == 144);
    REQUIRE(offsetof(SkyUniforms, tanHalfFov) == 160);
}

TEST_CASE("Shader types: WaterUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(WaterUniforms) == 288);
    REQUIRE(offsetof(WaterUniforms, cameraRelativeViewProjection) == 64);
    REQUIRE(offsetof(WaterUniforms, zenithColor) == 128);
    REQUIRE(offsetof(WaterUniforms, resolution) == 224);
    REQUIRE(offsetof(WaterUniforms, fogDensity) == 232);
    REQUIRE(offsetof(WaterUniforms, time) == 236);
    REQUIRE(offsetof(WaterUniforms, cameraUnderwater) == 240);
    REQUIRE(offsetof(WaterUniforms, ssrStrength) == 244);
    REQUIRE(offsetof(WaterUniforms, skyExposure) == 248);
    REQUIRE(offsetof(WaterUniforms, waterSurfaceY) == 252);
    REQUIRE(offsetof(WaterUniforms, solarDirection) == 256);
    REQUIRE(offsetof(WaterUniforms, physicalSkyBlend) == 272);
    REQUIRE(offsetof(WaterUniforms, directSpecularFactor) == 276);
}

TEST_CASE("Water procedural bands fade before their phase aliases",
          "[render][water][shader-types][antialiasing]") {
    REQUIRE(waterBandVisibility(0.0F) == Catch::Approx(1.0F));
    REQUIRE(waterBandVisibility(0.45F) == Catch::Approx(1.0F));
    REQUIRE(waterBandVisibility(1.125F) == Catch::Approx(0.5F));
    REQUIRE(waterBandVisibility(1.8F) <= 1.0e-6F);
    REQUIRE(waterBandVisibility(4.0F) <= 1.0e-6F);

    float previous = waterBandVisibility(0.0F);
    for (int sample = 1; sample <= 64; ++sample) {
        const float current = waterBandVisibility(static_cast<float>(sample) / 16.0F);
        REQUIRE(current <= previous);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        previous = current;
    }
}

TEST_CASE("Camera-relative water depth stays continuous at large world coordinates",
          "[render][water][precision][seam]") {
    const Vec3 camera{23029.0F, 380.0F, -111486.0F};
    const Mat4 view =
        Mat4::lookAt(camera, Vec3{23050.0F, 307.0F, -111460.0F}, Vec3{0.0F, 1.0F, 0.0F});
    const Mat4 projection =
        Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F, 16.0F / 9.0F, 0.1F, 1000.0F);

    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    std::memcpy(&viewMatrix, view.data.data(), sizeof(viewMatrix));
    std::memcpy(&projectionMatrix, projection.data.data(), sizeof(projectionMatrix));
    viewMatrix.columns[3] = simd_make_float4(0.0F, 0.0F, 0.0F, 1.0F);
    const simd_float4x4 viewProjection = simd_mul(projectionMatrix, viewMatrix);
    const simd_float4x4 inverseViewProjection = simd_inverse(viewProjection);

    auto reconstruct = [&](simd_float3 relative) {
        const simd_float4 clip =
            simd_mul(viewProjection, simd_make_float4(relative.x, relative.y, relative.z, 1.0F));
        const simd_float3 ndc = clip.xyz / clip.w;
        const simd_float2 uv = simd_make_float2(ndc.x * 0.5F + 0.5F, 0.5F - ndc.y * 0.5F);
        const simd_float4 reconstructedClip =
            simd_make_float4(uv.x * 2.0F - 1.0F, 1.0F - uv.y * 2.0F, ndc.z, 1.0F);
        const simd_float4 reconstructed = simd_mul(inverseViewProjection, reconstructedClip);
        return reconstructed.xyz / reconstructed.w;
    };

    const simd_float3 ray = simd_normalize(simd_make_float3(21.0F, -73.0F, 26.0F));
    const simd_float3 waterSurface = ray * 75.0F;
    const simd_float3 lakeFloor = ray * 78.0F;
    const simd_float3 reconstructedSurface = reconstruct(waterSurface);
    const simd_float3 reconstructedFloor = reconstruct(lakeFloor);

    REQUIRE(simd_length(reconstructedSurface - waterSurface) < 1.0e-2F);
    REQUIRE(simd_length(reconstructedFloor - lakeFloor) < 1.0e-2F);
    REQUIRE(simd_length(reconstructedFloor - reconstructedSurface) ==
            Catch::Approx(3.0F).margin(1.0e-2F));

    // The same reconstruction remains stable on both sides of a cubic chunk
    // face even though the absolute Z coordinate is more than 100,000 blocks
    // from the origin.
    for (float absoluteX : {23039.999F, 23040.001F}) {
        const simd_float3 relative =
            simd_make_float3(absoluteX - camera.x, 307.875F - camera.y, -111470.0F - camera.z);
        REQUIRE(simd_length(reconstruct(relative) - relative) < 1.0e-2F);
    }
}

TEST_CASE("Shader types: GPUParticle layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(GPUParticle) == 48);
    REQUIRE(offsetof(GPUParticle, velocity) == 16);
    REQUIRE(offsetof(GPUParticle, lifetime) == 32);
    REQUIRE(offsetof(GPUParticle, type) == 36);
}

TEST_CASE("Shader types: ParticleUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(ParticleUniforms) == 160);
    REQUIRE(offsetof(ParticleUniforms, cameraPosition) == 128);
    REQUIRE(offsetof(ParticleUniforms, atmosphericExtinction) == 144);
    REQUIRE(offsetof(ParticleUniforms, metersPerBlock) == 148);
}

TEST_CASE("Particle attenuation uses meters without scaling block-space billboards",
          "[render][particles][physical-scale][regression]") {
    const float legacyDistanceMeters = particleOpticalDistanceMeters(
        750.0F, static_cast<float>(LEGACY_WORLD_PHYSICAL_SCALE.horizontalMetersPerBlock));
    const float v4DistanceMeters = particleOpticalDistanceMeters(
        100.0F, static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock));
    REQUIRE(legacyDistanceMeters == Catch::Approx(750.0F));
    REQUIRE(v4DistanceMeters == Catch::Approx(legacyDistanceMeters));
    REQUIRE(beerLambertTransmittance(0.001F, legacyDistanceMeters) ==
            Catch::Approx(beerLambertTransmittance(0.001F, v4DistanceMeters)));

    REQUIRE(particleBillboardPointSize(20.0F) == Catch::Approx(8.0F));
    REQUIRE(particleBillboardPointSize(0.0F) == Catch::Approx(12.0F));
    REQUIRE(particleBillboardPointSize(1'000.0F) == Catch::Approx(2.0F));
}

TEST_CASE("Shader types: BloomUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(BloomUniforms) == 32);
    REQUIRE(offsetof(BloomUniforms, texelSize) == 8);
    REQUIRE(offsetof(BloomUniforms, threshold) == 16);
    REQUIRE(offsetof(BloomUniforms, blurRadius) == 24);
}

TEST_CASE("Shader types: PostUniforms layout matches MSL", "[render][shader-types]") {
    REQUIRE(sizeof(PostUniforms) == 48);
    REQUIRE(offsetof(PostUniforms, resolution) == 0);
    REQUIRE(offsetof(PostUniforms, exposure) == 8);
    REQUIRE(offsetof(PostUniforms, bloomIntensity) == 12);
    REQUIRE(offsetof(PostUniforms, vibrance) == 16);
    REQUIRE(offsetof(PostUniforms, sharpening) == 20);
    REQUIRE(offsetof(PostUniforms, frameIndex) == 24);
    REQUIRE(offsetof(PostUniforms, flareStrength) == 28);
    REQUIRE(offsetof(PostUniforms, sunScreenUV) == 32);
    REQUIRE(offsetof(PostUniforms, flareCloudOpacityTexture) == 40);
    REQUIRE(sizeof(FlareState) == 4);
}

TEST_CASE("Shader types: ExposureState + ExposureParams layout match MSL",
          "[render][shader-types]") {
    REQUIRE(sizeof(ExposureState) == 8);
    REQUIRE(offsetof(ExposureState, smoothedLogLum) == 0);
    REQUIRE(offsetof(ExposureState, exposure) == 4);

    REQUIRE(sizeof(ExposureParams) == 48);
    REQUIRE(offsetof(ExposureParams, keyValue) == 0);
    REQUIRE(offsetof(ExposureParams, adaptationDownRate) == 4);
    REQUIRE(offsetof(ExposureParams, minLogLum) == 8);
    REQUIRE(offsetof(ExposureParams, maxLogLum) == 12);
    REQUIRE(offsetof(ExposureParams, sampleGrid) == 16);
    REQUIRE(offsetof(ExposureParams, minExposure) == 24);
    REQUIRE(offsetof(ExposureParams, maxExposure) == 28);
    REQUIRE(offsetof(ExposureParams, adaptationUpRate) == 32);
    REQUIRE(offsetof(ExposureParams, highlightGain) == 36);
    REQUIRE(offsetof(ExposureParams, highlightKnee) == 40);
    REQUIRE(offsetof(ExposureParams, highlightRange) == 44);
}

TEST_CASE("Emission masks isolate lava furnace and torch radiance",
          "[render][textures][emissive][gameplay]") {
    for (uint8_t y = 0; y < BlockTextureArray::TILE_SIZE; ++y) {
        for (uint8_t x = 0; x < BlockTextureArray::TILE_SIZE; ++x) {
            REQUIRE(emissionMaskForTexel(static_cast<uint8_t>(BlockType::LAVA), x, y) == 255);
            REQUIRE(emissionMaskForTexel(static_cast<uint8_t>(BlockType::FURNACE), x, y) == 0);
            REQUIRE(emissionMaskForTexel(TEXTURE_LAYER_FURNACE_TOP, x, y) == 0);
            REQUIRE(emissionMaskForTexel(TEXTURE_LAYER_FURNACE_SIDE, x, y) == 0);
            REQUIRE(emissionMaskForTexel(TEXTURE_LAYER_CHEST_SIDE, x, y) == 0);
            REQUIRE(emissionMaskForTexel(TEXTURE_LAYER_CHEST_TOP, x, y) == 0);
            REQUIRE(emissionMaskForTexel(static_cast<uint8_t>(BlockType::BED), x, y) == 0);
        }
    }

    const uint8_t litFurnace = static_cast<uint8_t>(BlockType::FURNACE_LIT);
    REQUIRE(emissionMaskForTexel(litFurnace, 3, 6) == 255);
    REQUIRE(emissionMaskForTexel(litFurnace, 12, 12) == 255);
    REQUIRE(emissionMaskForTexel(litFurnace, 2, 6) == 0);
    REQUIRE(emissionMaskForTexel(litFurnace, 12, 13) == 0);

    const uint8_t torch = static_cast<uint8_t>(BlockType::TORCH);
    REQUIRE(emissionMaskForTexel(torch, 6, 1) == 255);
    REQUIRE(emissionMaskForTexel(torch, 9, 4) == 255);
    REQUIRE(emissionMaskForTexel(torch, 7, 5) == 0);
    REQUIRE(emissionMaskForTexel(torch, 5, 2) == 0);

    REQUIRE(isEmissive(BlockType::FURNACE_LIT));
    REQUIRE(isEmissive(BlockType::TORCH));
    REQUIRE_FALSE(isEmissive(BlockType::BED));
    REQUIRE(textureLayerFor(BlockType::BED, FaceNormal::PLUS_Y) ==
            static_cast<uint8_t>(BlockType::BED));
    REQUIRE(textureLayerFor(BlockType::BED, FaceNormal::MINUS_X) ==
            static_cast<uint8_t>(BlockType::BED));
}

TEST_CASE("Lit furnace emission belongs to exactly one fixed front face",
          "[render][mesher][furnace][emissive][gameplay]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::FURNACE_LIT);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    size_t mouthFaces = 0;
    size_t shellFaces = 0;
    for (size_t offset = 0; offset < output.indices.size(); offset += 6) {
        const Vertex& face = output.vertices[output.indices[offset]];
        REQUIRE(unpackEmissive(face.faceAttr));
        const uint8_t layer = unpackTextureLayer(face.faceAttr);
        if (layer == static_cast<uint8_t>(BlockType::FURNACE_LIT)) {
            ++mouthFaces;
            REQUIRE(unpackFace(face.faceAttr) == FaceNormal::MINUS_Z);
            REQUIRE(emissionMaskForTexel(layer, 8, 8) == 255);
        } else {
            ++shellFaces;
            REQUIRE((layer == TEXTURE_LAYER_FURNACE_SIDE || layer == TEXTURE_LAYER_FURNACE_TOP));
            REQUIRE(emissionMaskForTexel(layer, 8, 8) == 0);
        }
    }
    REQUIRE(mouthFaces == 1);
    REQUIRE(shellFaces == 5);
}

TEST_CASE("Inactive furnace keeps the fixed shell without emissive output",
          "[render][mesher][furnace][emissive][gameplay][regression]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::FURNACE);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    size_t frontFaces = 0;
    size_t shellFaces = 0;
    for (size_t offset = 0; offset < output.indices.size(); offset += 6) {
        const Vertex& face = output.vertices[output.indices[offset]];
        REQUIRE_FALSE(unpackEmissive(face.faceAttr));
        const uint8_t layer = unpackTextureLayer(face.faceAttr);
        REQUIRE(emissionMaskForTexel(layer, 8, 8) == 0);
        if (layer == static_cast<uint8_t>(BlockType::FURNACE)) {
            ++frontFaces;
            REQUIRE(unpackFace(face.faceAttr) == FaceNormal::MINUS_Z);
        } else {
            ++shellFaces;
            REQUIRE((layer == TEXTURE_LAYER_FURNACE_SIDE || layer == TEXTURE_LAYER_FURNACE_TOP));
        }
    }
    REQUIRE(frontFaces == 1);
    REQUIRE(shellFaces == 5);
}

TEST_CASE("Low bed culls supported faces and remains in shadow geometry",
          "[render][mesher][bed][shadow][gameplay]") {
    Chunk chunk(ChunkPos{0, 4, 0});
    chunk.setBlock(8, 8, 8, BlockType::BED);
    chunk.setBlock(9, 8, 8, BlockType::STONE);
    chunk.setBlock(8, 7, 8, BlockType::STONE);

    LODMesher mesher;
    const MeshOutput output = mesher.buildMesh(chunk, static_cast<int>(ChunkLOD::FULL));
    size_t bedVertices = 0;
    bool sawOccludedTopCorner = false;
    float minimumBedY = std::numeric_limits<float>::max();
    float maximumBedY = std::numeric_limits<float>::lowest();
    for (const Vertex& vertex : output.vertices) {
        if (unpackTextureLayer(vertex.faceAttr) != static_cast<uint8_t>(BlockType::BED))
            continue;
        ++bedVertices;
        REQUIRE(unpackFace(vertex.faceAttr) != FaceNormal::PLUS_X);
        REQUIRE(unpackFace(vertex.faceAttr) != FaceNormal::MINUS_Y);
        REQUIRE_FALSE(unpackEmissive(vertex.faceAttr));
        minimumBedY = std::min(minimumBedY, static_cast<float>(vertex.py));
        maximumBedY = std::max(maximumBedY, static_cast<float>(vertex.py));
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y &&
            unpackCornerAO(vertex.faceAttr) < 3) {
            sawOccludedTopCorner = true;
        }
    }
    REQUIRE(bedVertices == 16);
    REQUIRE(minimumBedY == Catch::Approx(8.0F));
    REQUIRE(maximumBedY == Catch::Approx(8.0F + BED_COLLISION_HEIGHT));
    REQUIRE(sawOccludedTopCorner);
    REQUIRE(output.opaqueIndexCount == output.indices.size());
}

// Latest-main physical atmosphere, smooth-lighting, and five-cascade regressions.

TEST_CASE("Metal ownership reset releases under ARC and manual reference counting",
          "[render][metal][ownership]") {
    @autoreleasepool {
        metalOwnershipProbeDeallocated = false;
        MetalOwnershipProbe* probe = [[MetalOwnershipProbe alloc] init];
        REQUIRE(probe != nil);

        resetMetalObject(probe);

        REQUIRE(probe == nil);
        REQUIRE(metalOwnershipProbeDeallocated);
    }

    STATIC_REQUIRE(std::is_destructible_v<EntityRenderer>);
    STATIC_REQUIRE(std::is_destructible_v<ItemEntityRenderer>);
    STATIC_REQUIRE(std::is_destructible_v<BoatRenderer>);
    STATIC_REQUIRE(std::is_destructible_v<UIOverlay>);
}

TEST_CASE("Opaque scene pipelines share the HDR surface attachment contract",
          "[render][pipeline]") {
    @autoreleasepool {
        auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        PixelFormats::configureScenePassPipeline(descriptor);

        REQUIRE(descriptor.colorAttachments[0].pixelFormat == PixelFormats::SCENE_HDR);
        REQUIRE(descriptor.colorAttachments[1].pixelFormat == PixelFormats::SURFACE);
        REQUIRE(descriptor.colorAttachments[2].pixelFormat == PixelFormats::REACTIVE);
        REQUIRE(descriptor.colorAttachments[3].pixelFormat == PixelFormats::RESOLVE_DEPTH_KEY);
        REQUIRE(descriptor.depthAttachmentPixelFormat == PixelFormats::SCENE_DEPTH);
        REQUIRE(descriptor.rasterSampleCount == PixelFormats::SCENE_SAMPLE_COUNT);
        resetMetalObject(descriptor);
    }

    STATIC_REQUIRE(PixelFormats::SCENE_SAMPLE_COUNT == 4);
    REQUIRE(PixelFormats::sceneResolveUsesTileShader(PixelFormats::SCENE_SAMPLE_COUNT));
    REQUIRE(PixelFormats::sceneResolveCoverageMask(PixelFormats::SCENE_SAMPLE_COUNT) == 0xFU);
    constexpr std::array<float, 4> depths = {0.50000006F, 0.50000000F, 0.75F, 0.9F};
    REQUIRE(PixelFormats::sceneResolveNearestDepthIndex(depths) == 1);

    constexpr PixelFormats::SceneTargetMemoryFootprint footprint =
        PixelFormats::sceneTargetMemoryFootprint(3, 5);
    STATIC_REQUIRE(footprint.reactiveResolveBytes == 15);
    STATIC_REQUIRE(footprint.sceneColorCopyBytes == 144);
    STATIC_REQUIRE(footprint.persistentMultisampleBytes == 0);
    STATIC_REQUIRE(footprint.totalBytes() == 459);
}

TEST_CASE("Snapshot water keeps exterior reflection authority separate from skylight",
          "[render][mesher][water][lighting]") {
    constexpr int waterX = 8;
    constexpr int waterY = 8;
    constexpr int waterZ = 8;
    constexpr int32_t receiverWorldY = 4 * CHUNK_EDGE + waterY + 1;

    auto topHasExteriorSky = [](const MeshOutput& output) {
        bool foundTop = false;
        bool exteriorSky = false;
        for (const Vertex& vertex : output.vertices) {
            if (unpackFace(vertex.faceAttr) != FaceNormal::PLUS_Y ||
                static_cast<float>(vertex.py) != static_cast<float>(waterY + 1)) {
                continue;
            }
            foundTop = true;
            exteriorSky = exteriorSky || unpackFluidExteriorSky(vertex.faceAttr);
        }
        REQUIRE(foundTop);
        return exteriorSky;
    };

    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.pos = {0, 4, 0};
    snapshot.derivedSkyLightValid = true;
    snapshot.blocks[MeshSnapshot::index(waterX, waterY, waterZ)] = BlockType::WATER;
    snapshot.fluidStates[MeshSnapshot::index(waterX, waterY, waterZ)] =
        FluidState::source().packed();
    snapshot.skyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] =
        MeshSnapshot::SKY_CUTOFF_INCOMPLETE;
    snapshot.visualSkyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] = receiverWorldY;

    MeshScratch scratch;
    REQUIRE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));

    // An edited roof must still suppress reflection while the packed skylight
    // remains conservatively dark.
    snapshot.visualSkyCutoffY[MeshSnapshot::skyIndex(waterX, waterZ)] = receiverWorldY + 1;
    REQUIRE_FALSE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));

    // Actual propagated skylight reopens a cave mouth without relying on the
    // incomplete-column visual cutoff.
    snapshot.packedLight[MeshSnapshot::index(waterX, waterY + 1, waterZ)] = packDerivedLight(1, 0);
    REQUIRE(topHasExteriorSky(LODMesher::buildMesh(snapshot, scratch)));
}

TEST_CASE("Snapshot mesher decodes independent packed light channels",
          "[render][mesher][light][skylight]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // Uniform patch so each smoothed corner reads the same skylight and block
    // light nibble the two channels are asserted against.
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 0xB5;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool foundBoundary = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 11);
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 5);
            foundBoundary = true;
        }
    }
    REQUIRE(foundBoundary);
}

TEST_CASE("Snapshot mesher smooths block light per vertex across a face",
          "[render][mesher][light][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // Block light rising along +Z in the air the +X face looks into: 0 at z=7,
    // 8 at z=8, 15 at z=9. Smooth lighting must give the +Z-side corners more
    // block light than the -Z-side corners instead of one flat value.
    for (int dy = -1; dy <= 1; ++dy) {
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 7)] = 0x00;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8)] = 0x08;
        snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 9)] = 0x0F;
    }

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    uint8_t low = 255;
    uint8_t high = 0;
    int faceVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            const uint8_t value = unpackBlockLight(vertex.faceAttr);
            low = std::min(low, value);
            high = std::max(high, value);
            ++faceVertices;
        }
    }
    REQUIRE(faceVertices == 4);
    REQUIRE(low < high); // a gradient, not one flat per-face value
    REQUIRE(low <= 5);   // the -Z corners see the dark end
    REQUIRE(high >= 10); // the +Z corners see the bright end
}

TEST_CASE("Snapshot mesher merges a uniformly lit face into one quad",
          "[render][mesher][light][smooth][greedy]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.derivedSkyLightValid = true;
    // A 3x3 stone slab with uniform light in the air above it must still merge
    // its top face to one quad despite the widened per-corner key.
    for (int z = 6; z <= 8; ++z)
        for (int x = 6; x <= 8; ++x)
            snapshot.blocks[MeshSnapshot::index(x, 8, z)] = BlockType::STONE;
    for (int z = 5; z <= 9; ++z)
        for (int x = 5; x <= 9; ++x)
            snapshot.packedLight[MeshSnapshot::index(x, 9, z)] = 0xF7; // sky 15, block 7

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    int topVertices = 0;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_Y) {
            ++topVertices;
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 7);
            REQUIRE(unpackSkyLight(vertex.faceAttr) == 15);
        }
    }
    REQUIRE(topVertices == 4);
}

TEST_CASE("Snapshot mesher excludes opaque neighbors from smoothed corners",
          "[render][mesher][light][smooth]") {
    MeshSnapshot snapshot;
    snapshot.clear();
    snapshot.blocks[MeshSnapshot::index(15, 8, 8)] = BlockType::STONE;
    // A solid neighbor sits beside the lit +X face. Its cell stores no
    // propagated light, so a corner touching it must average only the lit cells
    // and stay at 8 rather than being pulled toward zero (which would read 6).
    snapshot.blocks[MeshSnapshot::index(16, 8, 9)] = BlockType::STONE;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz)
            snapshot.packedLight[MeshSnapshot::index(16, 8 + dy, 8 + dz)] = 0x08;

    MeshScratch scratch;
    const MeshOutput output = LODMesher::buildMesh(snapshot, scratch);
    bool found = false;
    for (const Vertex& vertex : output.vertices) {
        if (unpackFace(vertex.faceAttr) == FaceNormal::PLUS_X &&
            static_cast<float>(vertex.px) == 16.0F) {
            REQUIRE(unpackBlockLight(vertex.faceAttr) == 8);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE("Foliage wind preserves canonical weather direction and physical speed",
          "[render][weather][sway][shader-types]") {
    WeatherSample localWeather{};
    localWeather.windBlocksPerSecond = {3.0F, 4.0F};
    const FoliageWindUniforms wind = makeFoliageWindUniforms(
        localWeather.windBlocksPerSecond.x, localWeather.windBlocksPerSecond.y, true);

    REQUIRE(wind.direction.x == Catch::Approx(0.6F));
    REQUIRE(wind.direction.y == Catch::Approx(0.8F));
    REQUIRE(wind.speedBlocksPerSecond == Catch::Approx(5.0F));
    REQUIRE(wind.strength == Catch::Approx(1.0F));

    Uniforms scene{};
    ShadowPassUniforms shadow{};
    scene.foliageWind = wind;
    shadow.foliageWind = wind;
    REQUIRE(scene.foliageWind.direction.x == shadow.foliageWind.direction.x);
    REQUIRE(scene.foliageWind.direction.y == shadow.foliageWind.direction.y);
    REQUIRE(scene.foliageWind.speedBlocksPerSecond == shadow.foliageWind.speedBlocksPerSecond);
    REQUIRE(scene.foliageWind.strength == shadow.foliageWind.strength);

    const FoliageWindUniforms disabled = makeFoliageWindUniforms(
        localWeather.windBlocksPerSecond.x, localWeather.windBlocksPerSecond.y, false);
    REQUIRE(disabled.direction.x == Catch::Approx(wind.direction.x));
    REQUIRE(disabled.direction.y == Catch::Approx(wind.direction.y));
    REQUIRE(disabled.speedBlocksPerSecond == Catch::Approx(wind.speedBlocksPerSecond));
    REQUIRE(disabled.strength == Catch::Approx(0.0F));

    const FoliageWindUniforms bounded = makeFoliageWindUniforms(24.0F, 0.0F, true);
    REQUIRE(bounded.direction.x == Catch::Approx(1.0F));
    REQUIRE(bounded.direction.y == Catch::Approx(0.0F));
    REQUIRE(bounded.speedBlocksPerSecond == Catch::Approx(FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND));
}

TEST_CASE("Deferred shadow cascades hold foliage casters static", "[render][shadow][sway]") {
    FoliageWindUniforms wind{};
    wind.direction = simd_make_float2(0.6F, 0.8F);
    wind.speedBlocksPerSecond = 5.0F;
    wind.strength = 1.0F;

    for (uint32_t cascade = 0U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const FoliageWindUniforms casterWind = shadowFoliageWindForCascade(wind, cascade);
        REQUIRE(casterWind.direction.x == Catch::Approx(wind.direction.x));
        REQUIRE(casterWind.direction.y == Catch::Approx(wind.direction.y));
        REQUIRE(casterWind.speedBlocksPerSecond == Catch::Approx(wind.speedBlocksPerSecond));
        REQUIRE(casterWind.strength == Catch::Approx(cascade < 2U ? wind.strength : 0.0F));
        REQUIRE(shadowCascadeUsesAnimatedFoliage(cascade) == (cascade < 2U));
    }

    wind.strength = 0.0F;
    for (uint32_t cascade = 0U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        REQUIRE(shadowFoliageWindForCascade(wind, cascade).strength == Catch::Approx(0.0F));
    }
}

TEST_CASE("Shadow cascades: quality table pins splits targets and slices", "[render][shadow]") {
    const std::array<float, SHADOW_CASCADE_COUNT> highFar = {48.0f, 160.0f, 512.0f, 1536.0f,
                                                             8192.0f};
    const std::array<float, SHADOW_CASCADE_COUNT> mediumFar = {40.0f, 128.0f, 384.0f, 768.0f,
                                                               8192.0f};
    const std::array<uint32_t, SHADOW_CASCADE_COUNT> highResolution = {4096u, 4096u, 2048u, 2048u,
                                                                       2048u};
    const std::array<uint32_t, SHADOW_CASCADE_COUNT> mediumResolution = {2048u, 2048u, 1024u, 1024u,
                                                                         1024u};

    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration high = shadowCascadeConfiguration(2u, cascade);
        const ShadowCascadeConfiguration medium = shadowCascadeConfiguration(1u, cascade);
        REQUIRE(high.farDepth == Catch::Approx(highFar[cascade]));
        REQUIRE(medium.farDepth == Catch::Approx(mediumFar[cascade]));
        REQUIRE(high.resolution == highResolution[cascade]);
        REQUIRE(medium.resolution == mediumResolution[cascade]);
        if (cascade < 2u) {
            REQUIRE(high.textureGroup == ShadowTextureGroup::NEAR);
            REQUIRE(high.textureSlice == cascade);
        } else if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
            REQUIRE(high.textureGroup == ShadowTextureGroup::FAR);
            REQUIRE(high.textureSlice == cascade - 2u);
        } else {
            REQUIRE(high.textureGroup == ShadowTextureGroup::HORIZON);
            REQUIRE(high.textureSlice == 0u);
        }
    }
}

TEST_CASE("Shadow cascades: overlap selection uses camera-forward view depth", "[render][shadow]") {
    ShadowUniforms shadow{};
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration configuration = shadowCascadeConfiguration(2u, cascade);
        shadow.cascades[cascade].depthRange =
            simd_make_float4(configuration.nearDepth, configuration.farDepth,
                             shadowCascadeBlendStart(configuration), 1.0f);
    }

    const simd_float3 camera = simd_make_float3(10.0f, 2.0f, 3.0f);
    const simd_float3 forward = simd_make_float3(0.0f, 0.0f, -1.0f);
    REQUIRE(shadowViewDepth(simd_make_float3(10.0f, 2.0f, -7.0f), camera, forward) ==
            Catch::Approx(10.0f));
    REQUIRE(shadowViewDepth(simd_make_float3(10.0f, 2.0f, 13.0f), camera, forward) ==
            Catch::Approx(-10.0f));

    const float firstBlendStart = shadowCascadeBlendStart(shadowCascadeConfiguration(2u, 0u));
    ShadowCascadeSelection beforeBlend = shadowCascadeSelection(firstBlendStart - 0.01f, shadow);
    REQUIRE(beforeBlend.primary == 0u);
    REQUIRE(beforeBlend.secondary == 0u);
    REQUIRE(beforeBlend.secondaryWeight == Catch::Approx(0.0f));

    ShadowCascadeSelection insideBlend = shadowCascadeSelection(45.0f, shadow);
    REQUIRE(insideBlend.primary == 0u);
    REQUIRE(insideBlend.secondary == 1u);
    REQUIRE(insideBlend.secondaryWeight > 0.0f);
    REQUIRE(insideBlend.secondaryWeight < 1.0f);

    ShadowCascadeSelection horizonBlend = shadowCascadeSelection(1500.0f, shadow);
    REQUIRE(horizonBlend.primary == 3u);
    REQUIRE(horizonBlend.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizonBlend.secondaryWeight > 0.0f);
    REQUIRE(horizonBlend.secondaryWeight < 1.0f);

    ShadowCascadeSelection horizon = shadowCascadeSelection(2000.0f, shadow);
    REQUIRE(horizon.primary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizon.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(horizon.covered == 1u);

    const ShadowCascadeConfiguration horizonConfiguration =
        shadowCascadeConfiguration(2u, SHADOW_HORIZON_CASCADE_INDEX);
    const float terminalStart = shadowCascadeBlendStart(horizonConfiguration);
    ShadowCascadeSelection terminalFade =
        shadowCascadeSelection(0.5F * (terminalStart + horizonConfiguration.farDepth), shadow);
    REQUIRE(terminalFade.primary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(terminalFade.secondary == SHADOW_HORIZON_CASCADE_INDEX);
    REQUIRE(terminalFade.exteriorWeight > 0.0F);
    REQUIRE(terminalFade.exteriorWeight < 1.0F);
    REQUIRE(shadowCascadeSelection(horizonConfiguration.farDepth, shadow).exteriorWeight ==
            Catch::Approx(1.0F));
    REQUIRE(shadowCascadeSelection(9000.0f, shadow).covered == 0u);
    REQUIRE(shadowCascadeSelection(-1.0f, shadow).covered == 0u);
}

TEST_CASE("Shadow cascades: overlap and texel snap metadata are stable", "[render][shadow]") {
    for (uint32_t quality : {1u, 2u}) {
        for (uint32_t cascade = 0; cascade < SHADOW_HORIZON_CASCADE_INDEX; ++cascade) {
            const ShadowCascadeConfiguration configuration =
                shadowCascadeConfiguration(quality, cascade);
            REQUIRE(configuration.farDepth - shadowCascadeBlendStart(configuration) ==
                    Catch::Approx((configuration.farDepth - configuration.nearDepth) *
                                  SHADOW_CASCADE_BLEND_FRACTION));
        }
    }

    const ShadowCascadeConfiguration first = shadowCascadeConfiguration(2u, 0u);
    REQUIRE(shadowCascadeBlendStart(first) == Catch::Approx(42.0625f));

    const float tanHalfFov = std::tan(70.0F * static_cast<float>(M_PI) / 360.0F);
    const float analytic = shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 1.5F);
    REQUIRE(analytic == shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 1.5F));
    REQUIRE(std::fmod(analytic * 16.0F, 1.0F) == Catch::Approx(0.0F).margin(1.0e-6F));
    // Reconstructing absolute frustum corners at the acceptance coordinate
    // alternated between 66.3125 and 66.375 while moving. The analytic native
    // route radius has one exact authority independent of camera position.
    REQUIRE(shadowCascadeBoundingRadius(0.5F, 48.0F, tanHalfFov, 3456.0F / 2234.0F) ==
            Catch::Approx(66.3125F));

    const double texelWorldSize = 2.0 * analytic / first.resolution;
    const double lightCoordinate = -102'753.123456;
    const double snapped = shadowSnappedLightCoordinate(lightCoordinate, texelWorldSize);
    REQUIRE(snapped / texelWorldSize ==
            Catch::Approx(std::round(lightCoordinate / texelWorldSize)));
    REQUIRE(std::abs(snapped - lightCoordinate) <= texelWorldSize * 0.5);
}

TEST_CASE("Shadow cascades: refresh cadence bounds stale depth", "[render][shadow]") {
    REQUIRE(shadowCascadeMaximumRefreshInterval(0U) == 1U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(1U) == 1U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(2U) == 2U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(3U) == 4U);
    REQUIRE(shadowCascadeMaximumRefreshInterval(4U) == 8U);
}

TEST_CASE("Shadow cascades: deferred depth coverage refresh is snap-safe", "[render][shadow]") {
    const Mat4 rendered = Mat4::identity();
    const Vec3 origin{22'784.0F, 0.0F, -111'872.0F};
    Mat4 candidate = rendered;
    REQUIRE_FALSE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));

    candidate(0, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(1, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(2, 3) += 0.125F;
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    candidate = rendered;
    candidate(3, 3) += 0.125F;
    REQUIRE_FALSE(shadowCascadeProjectionChanged(candidate, origin, rendered, origin));
    REQUIRE(shadowCascadeProjectionChanged(candidate, origin, rendered,
                                           origin + Vec3{256.0F, 0.0F, 0.0F}));

    constexpr float radius = 10.0F;
    constexpr float casterMargin = 2.0F;
    constexpr float normalBias = 0.5F;
    constexpr uint32_t resolution = 1024U;
    const float guard =
        shadowCascadeReceiverDepthGuard(radius, casterMargin, resolution, normalBias);
    const float depthRange = shadowCascadeDepthRange(radius, casterMargin, guard);
    const float depthTexel =
        shadowCascadeDepthTexelWorldSize(radius, casterMargin, guard, resolution);
    REQUIRE(guard >= normalBias + depthTexel);

    const double depthCenter = static_cast<double>(depthTexel) * 1'024.0;
    REQUIRE(shadowSnappedLightCoordinate(depthCenter + 0.49 * depthTexel, depthTexel) ==
            Catch::Approx(shadowSnappedLightCoordinate(depthCenter, depthTexel)));
    REQUIRE(shadowSnappedLightCoordinate(depthCenter + 0.51 * depthTexel, depthTexel) !=
            Catch::Approx(shadowSnappedLightCoordinate(depthCenter, depthTexel)));

    const Vec3 receiverCenter{};
    const Vec3 light{0.0F, 0.0F, 1.0F};
    const Mat4 sampled =
        Mat4::orthographic(-radius, radius, -radius, radius, 0.0F, depthRange) *
        Mat4::lookAt(light * (radius + casterMargin + guard), receiverCenter, Vec3::up());
    REQUIRE(shadowCascadeReceiverDepthCovered(sampled, Vec3::zero(), receiverCenter, radius));
    REQUIRE_FALSE(shadowCascadeReceiverDepthCovered(
        sampled, Vec3::zero(), receiverCenter - light * (guard + 0.25F), radius));
}

TEST_CASE("Shadow cascades: projection anchors preserve large-coordinate precision",
          "[render][shadow][large-coordinate]") {
    const Vec3 camera{23'029.0F, 225.0F, -111'726.0F};
    const Vec3 origin = shadowProjectionOrigin(camera);
    REQUIRE(origin.x == Catch::Approx(22'784.0F));
    REQUIRE(origin.y == Catch::Approx(0.0F));
    REQUIRE(origin.z == Catch::Approx(-111'872.0F));

    const Vec3 local = camera - origin;
    REQUIRE(local.x == Catch::Approx(245.0F));
    REQUIRE(local.y == Catch::Approx(225.0F));
    REQUIRE(local.z == Catch::Approx(146.0F));
    REQUIRE(std::abs(local.x) < 256.0F);
    REQUIRE(std::abs(local.y) < 256.0F);
    REQUIRE(std::abs(local.z) < 256.0F);

    const Vec3 lightAxis = Vec3{0.37F, 0.81F, -0.45F}.normalize();
    const double expected = static_cast<double>(camera.x) * lightAxis.x +
                            static_cast<double>(camera.y) * lightAxis.y +
                            static_cast<double>(camera.z) * lightAxis.z;
    REQUIRE(shadowPreciseDot(camera, lightAxis) == Catch::Approx(expected));
}

TEST_CASE("Shadow cascades: nearby entities do not invalidate coarse high-sun slices",
          "[render][shadow][entity]") {
    const Vec3 camera{0.0F, 64.0F, 0.0F};
    const Vec3 forward{0.0F, 0.0F, 1.0F};
    const Vec3 highSun{0.0F, 1.0F, 0.0F};
    const AABB before{{-0.5F, 64.0F, 9.5F}, {0.5F, 66.0F, 10.5F}};
    const AABB after{{-0.5F, 64.0F, 10.5F}, {0.5F, 66.0F, 11.5F}};

    const ShadowCascadeConfiguration near = shadowCascadeConfiguration(2U, 0U);
    REQUIRE(shadowEntityCasterReachesDepthSlice(before, camera, forward, highSun, near.nearDepth,
                                                near.farDepth));
    REQUIRE(shadowEntityCasterReachesDepthSlice(after, camera, forward, highSun, near.nearDepth,
                                                near.farDepth));

    for (uint32_t cascade = 2U; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const ShadowCascadeConfiguration coarse = shadowCascadeConfiguration(2U, cascade);
        REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(before, camera, forward, highSun,
                                                          coarse.nearDepth, coarse.farDepth));
        REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(after, camera, forward, highSun,
                                                          coarse.nearDepth, coarse.farDepth));
    }
}

TEST_CASE("Shadow cascades: low-sun entity extrusion retains reachable coarse shadows",
          "[render][shadow][entity]") {
    const Vec3 camera{0.0F, 64.0F, 0.0F};
    const Vec3 forward{0.0F, 0.0F, 1.0F};
    const AABB entity{{-0.5F, 64.0F, 9.5F}, {0.5F, 66.0F, 10.5F}};

    // Light comes from behind the camera, so its shadow travels forward. At a
    // low elevation the ray can reach the horizon slice before the world
    // floor bounds it; at a steep elevation it cannot.
    const Vec3 lowSun = Vec3{0.0F, 0.10F, -0.995F}.normalize();
    const Vec3 highSun = Vec3{0.0F, 0.80F, -0.60F}.normalize();
    const ShadowCascadeConfiguration horizon =
        shadowCascadeConfiguration(2U, SHADOW_HORIZON_CASCADE_INDEX);
    const float horizonNear = shadowCascadeBlendStart(shadowCascadeConfiguration(2U, 3U));
    REQUIRE(shadowEntityCasterReachesDepthSlice(entity, camera, forward, lowSun, horizonNear,
                                                horizon.farDepth));
    REQUIRE_FALSE(shadowEntityCasterReachesDepthSlice(entity, camera, forward, highSun, horizonNear,
                                                      horizon.farDepth));

    // A caster beyond a receiver slice remains eligible when the light points
    // toward the camera and its shadow travels back into that slice.
    const AABB distant{{-0.5F, 300.0F, 1'599.5F}, {0.5F, 302.0F, 1'600.5F}};
    const Vec3 frontLight = Vec3{0.0F, 0.20F, 0.98F}.normalize();
    const ShadowCascadeConfiguration fourth = shadowCascadeConfiguration(2U, 3U);
    REQUIRE(shadowEntityCasterReachesDepthSlice(distant, camera, forward, frontLight,
                                                fourth.nearDepth, fourth.farDepth));
}

TEST_CASE("Shadow visibility follows the active celestial source strength",
          "[render][shadow][celestial]") {
    REQUIRE(shadowVisibilityWithStrength(0.0F, 0.0F) == Catch::Approx(1.0F));
    REQUIRE(shadowVisibilityWithStrength(0.0F, 0.14F) == Catch::Approx(0.86F));
    REQUIRE(shadowVisibilityWithStrength(0.0F, 1.0F) == Catch::Approx(0.0F));
    REQUIRE(shadowVisibilityWithStrength(0.35F, 0.5F) == Catch::Approx(0.675F));
    REQUIRE(shadowVisibilityWithStrength(-1.0F, 2.0F) == Catch::Approx(0.0F));
}

TEST_CASE("Celestial state forms physical full quarter and new Moon phases",
          "[render][celestial]") {
    const uint64_t fullTick = CELESTIAL_FULL_MOON_REFERENCE_TICK;
    const CelestialState full = computeCelestialState(fullTick);
    const CelestialState quarter =
        computeCelestialState(fullTick + CELESTIAL_SYNODIC_PERIOD_TICKS / 4U);
    const CelestialState fresh =
        computeCelestialState(fullTick + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U);

    REQUIRE(full.sunDirection.dot(full.moonDirection) == Catch::Approx(-1.0F).margin(1.0e-5F));
    REQUIRE(full.illuminatedFraction == Catch::Approx(1.0F).margin(1.0e-5F));
    REQUIRE(full.phaseEnergy == Catch::Approx(1.0F).margin(1.0e-5F));
    REQUIRE(quarter.illuminatedFraction == Catch::Approx(0.5F).margin(2.0e-5F));
    REQUIRE(quarter.phaseEnergy == Catch::Approx(1.0F / static_cast<float>(M_PI)).margin(2.0e-5F));
    REQUIRE(fresh.sunDirection.dot(fresh.moonDirection) == Catch::Approx(1.0F).margin(2.0e-5F));
    REQUIRE(fresh.illuminatedFraction <= 2.0e-5F);
    REQUIRE(fresh.phaseEnergy <= 2.0e-5F);
    REQUIRE(static_cast<double>(CELESTIAL_SYNODIC_PERIOD_TICKS) /
                static_cast<double>(CELESTIAL_TICKS_PER_DAY) ==
            Catch::Approx(29.530583).margin(1.0e-6));
}

TEST_CASE("Celestial state suppresses competing Moon light through twilight",
          "[render][celestial]") {
    const uint64_t fullMidnight = CELESTIAL_FULL_MOON_REFERENCE_TICK;
    const uint64_t fullDayStart = fullMidnight - 18'000U;
    const CelestialState justAfterSunset = computeCelestialState(fullDayStart + 12'020U);
    const CelestialState civilTwilight = computeCelestialState(fullDayStart + 12'400U);
    const CelestialState midnight = computeCelestialState(fullMidnight);

    REQUIRE(justAfterSunset.sunVisibility <= 1.0e-5F);
    REQUIRE(justAfterSunset.moonDirectVisibility <= 1.0e-5F);
    REQUIRE(justAfterSunset.directSource == CelestialLightSource::NONE);
    REQUIRE(civilTwilight.directLightRadiance.length() < 0.04F);
    REQUIRE(midnight.directSource == CelestialLightSource::MOON);
    // Full-moon direct light sits at the playable-night level: far below any
    // daylight value, bright enough that moonlit terrain reads in motion.
    REQUIRE(midnight.directLightRadiance.length() < 0.05F);
    // The disc peaks just past the bloom threshold for a slight glow while
    // staying an order of magnitude below the sun disc's 18x on-screen
    // radiance, so the Moon reads at a glance without becoming a second sun.
    REQUIRE(midnight.lunarDiscRadiance.length() < 3.60F);
    REQUIRE(midnight.shadowStrength < 0.08F);
    REQUIRE(midnight.directSpecularFactor == Catch::Approx(midnight.phaseEnergy));
}

TEST_CASE("Visible horizon Sun does not apply daytime irradiance to terrain",
          "[render][celestial][twilight]") {
    const CelestialState sunset = computeCelestialState(12'000U);
    REQUIRE(sunset.sunVisibility == Catch::Approx(0.5F).margin(5.0e-5F));
    REQUIRE(sunset.sunDirectVisibility <= 1.0e-6F);
    REQUIRE(sunset.directSource == CelestialLightSource::NONE);
    REQUIRE(sunset.directLightRadiance.length() <= 1.0e-6F);
    REQUIRE(sunset.shadowStrength <= 1.0e-6F);

    const auto closestMorningTick = [](float elevationDegrees) {
        const float target = std::sin(elevationDegrees * static_cast<float>(M_PI) / 180.0F);
        uint64_t closest = 0U;
        float error = std::numeric_limits<float>::max();
        for (uint64_t tick = 0U; tick <= 6'000U; ++tick) {
            const float candidate = computeCelestialState(tick).sunDirection.y;
            const float candidateError = std::abs(candidate - target);
            if (candidateError < error) {
                error = candidateError;
                closest = tick;
            }
        }
        return closest;
    };
    const CelestialState oneDegree = computeCelestialState(closestMorningTick(1.0F));
    const CelestialState fiveDegrees = computeCelestialState(closestMorningTick(5.0F));
    const CelestialState tenDegrees = computeCelestialState(closestMorningTick(10.0F));
    REQUIRE(oneDegree.sunVisibility > 0.99F);
    REQUIRE(oneDegree.sunDirectVisibility < 0.04F);
    REQUIRE(fiveDegrees.sunDirectVisibility > oneDegree.sunDirectVisibility);
    REQUIRE(fiveDegrees.sunDirectVisibility < 0.60F);
    REQUIRE(tenDegrees.sunDirectVisibility > 0.99F);
}

TEST_CASE("Night ambient is phase-aware and cannot resemble daylight",
          "[render][celestial][night]") {
    const CelestialState fullMoon = computeCelestialState(CELESTIAL_FULL_MOON_REFERENCE_TICK);
    const uint64_t exactNewMoon =
        CELESTIAL_FULL_MOON_REFERENCE_TICK + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U;
    const uint64_t newMoonMidnight = exactNewMoon + (18'000U + CELESTIAL_TICKS_PER_DAY -
                                                     exactNewMoon % CELESTIAL_TICKS_PER_DAY) %
                                                        CELESTIAL_TICKS_PER_DAY;
    const CelestialState newMoon = computeCelestialState(newMoonMidnight);
    const CelestialState noon = computeCelestialState(6'000U);

    REQUIRE(newMoon.phaseEnergy < 0.002F);
    REQUIRE(fullMoon.ambientRadiance.length() > newMoon.ambientRadiance.length());
    // Playable-night contract: a full moon sits about a tenth of daylight so
    // moonlit terrain stays legible, while a new moon keeps only the stellar
    // floor and daylight remains an order of magnitude above any night.
    REQUIRE(fullMoon.ambientRadiance.length() < 0.055F);
    REQUIRE(newMoon.ambientRadiance.length() < 0.025F);
    REQUIRE(noon.ambientRadiance.length() > fullMoon.ambientRadiance.length() * 10.0F);
}

TEST_CASE("Moon fades in only after civil twilight at sunset and sunrise",
          "[render][celestial][twilight]") {
    const auto closestTick = [](float elevationDegrees, bool beforeSunrise) {
        const float target = std::sin(elevationDegrees * static_cast<float>(M_PI) / 180.0F);
        const uint64_t begin = beforeSunrise ? 18'000U : 12'000U;
        const uint64_t end = beforeSunrise ? 24'000U : 18'000U;
        uint64_t closest = begin;
        float error = std::numeric_limits<float>::max();
        for (uint64_t tick = begin; tick <= end; ++tick) {
            const float candidate = computeCelestialState(tick).sunDirection.y;
            const float candidateError = std::abs(candidate - target);
            if (candidateError < error) {
                error = candidateError;
                closest = tick;
            }
        }
        return closest;
    };

    for (const bool beforeSunrise : {false, true}) {
        const CelestialState minusFive = computeCelestialState(closestTick(-5.0F, beforeSunrise));
        const CelestialState minusSix = computeCelestialState(closestTick(-6.0F, beforeSunrise));
        const CelestialState minusSeven = computeCelestialState(closestTick(-7.0F, beforeSunrise));
        const CelestialState minusTwelve =
            computeCelestialState(closestTick(-12.0F, beforeSunrise));

        REQUIRE(minusFive.moonDirectVisibility <= 1.0e-6F);
        REQUIRE(minusFive.directSource == CelestialLightSource::NONE);
        REQUIRE(minusSix.moonDirectVisibility < 0.001F);
        REQUIRE(minusSeven.moonDirectVisibility > minusSix.moonDirectVisibility);
        REQUIRE(minusSeven.moonDirectVisibility < 0.20F);
        REQUIRE(minusTwelve.moonDirectVisibility > 0.98F);
        REQUIRE(minusSeven.directLightRadiance.length() < 0.004F);
    }
}

TEST_CASE("Celestial source is exclusive continuous and phase-scaled", "[render][celestial]") {
    const uint64_t cycle = CELESTIAL_SYNODIC_PERIOD_TICKS;
    for (uint64_t tick = 0; tick < cycle; tick += 137U) {
        const CelestialState state = computeCelestialState(tick);
        REQUIRE(state.sunDirection.length() == Catch::Approx(1.0F).margin(1.0e-5F));
        REQUIRE(state.moonDirection.length() == Catch::Approx(1.0F).margin(1.0e-5F));
        REQUIRE(std::isfinite(state.directLightRadiance.length()));
        if (state.directSource == CelestialLightSource::SUN) {
            REQUIRE(state.sunDirectVisibility > 0.0F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(1.0F));
        } else if (state.directSource == CelestialLightSource::MOON) {
            REQUIRE(state.sunVisibility <= 0.0001F);
            REQUIRE(state.moonDirectVisibility > 0.0F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(state.phaseEnergy));
        } else {
            REQUIRE(state.directLightRadiance.length() <= 1.0e-6F);
            REQUIRE(state.directSpecularFactor == Catch::Approx(0.0F));
        }
    }

    const CelestialState before = computeCelestialState(cycle - 1U);
    const CelestialState after = computeCelestialState(cycle);
    REQUIRE((before.sunDirection - after.sunDirection).length() < 0.001F);
    REQUIRE((before.moonDirection - after.moonDirection).length() < 0.001F);
    REQUIRE(std::abs(before.phaseEnergy - after.phaseEnergy) < 0.001F);
    const float angularDiameterDegrees =
        2.0F * LUNAR_ANGULAR_RADIUS_RADIANS * 180.0F / static_cast<float>(M_PI);
    REQUIRE(angularDiameterDegrees == Catch::Approx(0.518F).margin(0.01F));
}

TEST_CASE("New Moon cannot drive lighting shadows or a water glint", "[render][celestial]") {
    const uint64_t newMoonMidnight =
        CELESTIAL_FULL_MOON_REFERENCE_TICK + CELESTIAL_SYNODIC_PERIOD_TICKS / 2U;
    const CelestialState state = computeCelestialState(newMoonMidnight);
    REQUIRE(state.phaseEnergy <= 1.0e-5F);
    REQUIRE(state.moonDirectVisibility <= 1.0e-5F);
    REQUIRE(state.directLightRadiance.length() <= 1.0e-6F);
    REQUIRE(state.shadowStrength <= 1.0e-6F);
    REQUIRE(state.directSpecularFactor <= 1.0e-6F);
}

TEST_CASE("Air precipitation does not leak into the underwater medium",
          "[render][weather][water]") {
    REQUIRE(weatherParticlesVisible(false));
    REQUIRE_FALSE(weatherParticlesVisible(true));
}

TEST_CASE("Weather particle sessions clear and deterministically reseed across worlds",
          "[render][weather][particles][session][regression]") {
    WeatherParticleSessionState session;
    REQUIRE(session.beginWorld(11, 764891));
    const float first = session.nextRandomFloat();
    session.particles().front().active = true;
    REQUIRE(session.activeCount() == 1);
    REQUIRE_FALSE(session.beginWorld(11, 764891));
    REQUIRE(session.activeCount() == 1);

    session.endWorld();
    REQUIRE_FALSE(session.bound());
    REQUIRE(session.activeCount() == 0);
    REQUIRE(session.beginWorld(12, 764891));
    REQUIRE(session.nextRandomFloat() == Catch::Approx(first));

    session.endWorld();
    REQUIRE(session.beginWorld(13, 764892));
    REQUIRE(session.nextRandomFloat() != Catch::Approx(first));
}

TEST_CASE("Cloud noise world binding is nonblocking and disabled quality allocates nothing",
          "[render][clouds][noise][async][session][performance][regression]") {
    constexpr uint64_t INSTANCE = 71;
    constexpr uint64_t SEED = 764891;
    std::mutex gateMutex;
    std::condition_variable gate;
    bool builderEntered = false;
    bool releaseBuilder = false;
    CloudNoisePublication publication(
        [&](uint64_t worldInstanceId, uint64_t seed,
            std::stop_token stopToken) -> std::optional<CloudNoisePayload> {
            {
                std::unique_lock lock(gateMutex);
                builderEntered = true;
                gate.notify_all();
                gate.wait(lock, [&] { return releaseBuilder || stopToken.stop_requested(); });
            }
            if (stopToken.stop_requested())
                return std::nullopt;
            return CloudNoisePayload{
                .worldInstanceId = worldInstanceId,
                .seed = seed,
                .base = {static_cast<uint8_t>(seed)},
                .erosion = {static_cast<uint8_t>(seed >> 8U)},
                .curl = {static_cast<int8_t>(seed >> 16U)},
            };
        });

    publication.beginWorld(INSTANCE, SEED, false);
    const CloudNoisePublicationStats disabled = publication.stats();
    REQUIRE_FALSE(disabled.workerStarted);
    REQUIRE_FALSE(disabled.buildActive);
    REQUIRE_FALSE(disabled.requestPending);
    REQUIRE(disabled.buildsStarted == 0);
    REQUIRE(disabled.retainedPayloadBytes == 0);

    std::atomic<bool> beginReturned = false;
    std::jthread caller([&] {
        publication.beginWorld(INSTANCE, SEED, true);
        beginReturned.store(true, std::memory_order_release);
    });
    {
        std::unique_lock lock(gateMutex);
        REQUIRE(gate.wait_for(lock, std::chrono::seconds(1), [&] { return builderEntered; }));
    }
    const auto returnDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (!beginReturned.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < returnDeadline) {
        std::this_thread::yield();
    }
    const bool returnedBeforeGeneration = beginReturned.load(std::memory_order_acquire);
    {
        std::lock_guard lock(gateMutex);
        releaseBuilder = true;
    }
    gate.notify_all();
    caller.join();
    REQUIRE(returnedBeforeGeneration);

    std::optional<CloudNoisePayload> firstEnabledDraw;
    const auto publicationDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!firstEnabledDraw && std::chrono::steady_clock::now() < publicationDeadline) {
        firstEnabledDraw = publication.takeReadyForDraw();
        if (!firstEnabledDraw)
            std::this_thread::yield();
    }
    REQUIRE(firstEnabledDraw);
    REQUIRE(firstEnabledDraw->worldInstanceId == INSTANCE);
    REQUIRE(firstEnabledDraw->seed == SEED);
    REQUIRE(firstEnabledDraw->base == std::vector<uint8_t>{static_cast<uint8_t>(SEED)});
    REQUIRE(publication.stats().retainedPayloadBytes == 0);
}

TEST_CASE("Cloud noise teardown cancels and joins an active builder",
          "[render][clouds][noise][thread][shutdown][regression]") {
    std::mutex gateMutex;
    std::condition_variable gate;
    bool builderEntered = false;
    auto publication = std::make_unique<CloudNoisePublication>(
        [&](uint64_t, uint64_t, std::stop_token stopToken) -> std::optional<CloudNoisePayload> {
            std::stop_callback wakeOnStop(stopToken, [&] { gate.notify_all(); });
            std::unique_lock lock(gateMutex);
            builderEntered = true;
            gate.notify_all();
            gate.wait(lock, [&] { return stopToken.stop_requested(); });
            return std::nullopt;
        });
    publication->beginWorld(83, 764891, true);
    bool entered = false;
    {
        std::unique_lock lock(gateMutex);
        entered = gate.wait_for(lock, std::chrono::seconds(1), [&] { return builderEntered; });
    }
    REQUIRE(entered);

    std::future<void> teardown = std::async(std::launch::async, [&] { publication.reset(); });
    REQUIRE(teardown.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    REQUIRE_NOTHROW(teardown.get());
}

TEST_CASE("First enabled cloud draw receives the production payload for its exact seed",
          "[render][clouds][noise][async][session][seed][regression]") {
    constexpr uint64_t INSTANCE = 77;
    constexpr uint64_t SEED = 764891;
    CloudNoisePublication publication;
    publication.beginWorld(INSTANCE, SEED, true);

    std::optional<CloudNoisePayload> firstEnabledDraw;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!firstEnabledDraw && std::chrono::steady_clock::now() < deadline) {
        firstEnabledDraw = publication.takeReadyForDraw();
        if (!firstEnabledDraw)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(firstEnabledDraw);
    REQUIRE(firstEnabledDraw->worldInstanceId == INSTANCE);
    REQUIRE(firstEnabledDraw->seed == SEED);
    REQUIRE(firstEnabledDraw->base.size() == 2'097'152);
    REQUIRE(firstEnabledDraw->erosion.size() == 32'768);
    REQUIRE(firstEnabledDraw->curl.size() == 32'768);

    constexpr std::array<std::array<int, 3>, 4> SAMPLES{{
        {0, 0, 0},
        {31, 63, 95},
        {87, 42, 113},
        {127, 127, 127},
    }};
    for (const auto& coordinate : SAMPLES) {
        const int x = coordinate[0];
        const int y = coordinate[1];
        const int z = coordinate[2];
        const size_t index =
            (static_cast<size_t>(z) * CLOUD_BASE_NOISE_EDGE + static_cast<size_t>(y)) *
                CLOUD_BASE_NOISE_EDGE +
            static_cast<size_t>(x);
        const uint8_t expected = static_cast<uint8_t>(
            cloudBaseNoise(x, y, z, CLOUD_BASE_NOISE_EDGE, SEED) * 255.0F + 0.5F);
        REQUIRE(firstEnabledDraw->base[index] == expected);
    }
}

TEST_CASE("Cloud noise publication discards completed work from an old world",
          "[render][clouds][noise][async][session][identity][regression]") {
    constexpr uint64_t OLD_INSTANCE = 81;
    constexpr uint64_t NEW_INSTANCE = 82;
    constexpr uint64_t OLD_SEED = 0x0102'0304ULL;
    constexpr uint64_t NEW_SEED = 0xA1A2'A3A4ULL;
    std::mutex buildMutex;
    std::condition_variable buildWake;
    bool oldBuildEntered = false;
    CloudNoisePublication publication(
        [&](uint64_t worldInstanceId, uint64_t seed,
            std::stop_token stopToken) -> std::optional<CloudNoisePayload> {
            if (worldInstanceId == OLD_INSTANCE) {
                {
                    std::lock_guard lock(buildMutex);
                    oldBuildEntered = true;
                }
                buildWake.notify_all();
                while (!stopToken.stop_requested())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return std::nullopt;
            }
            return CloudNoisePayload{
                .worldInstanceId = worldInstanceId,
                .seed = seed,
                .base = {static_cast<uint8_t>(seed)},
                .erosion = {static_cast<uint8_t>(seed >> 8U)},
                .curl = {static_cast<int8_t>(seed >> 16U)},
            };
        });

    publication.beginWorld(OLD_INSTANCE, OLD_SEED, true);
    {
        std::unique_lock lock(buildMutex);
        REQUIRE(buildWake.wait_for(lock, std::chrono::seconds(1), [&] { return oldBuildEntered; }));
    }
    publication.beginWorld(NEW_INSTANCE, NEW_SEED, true);

    std::optional<CloudNoisePayload> current;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!current && std::chrono::steady_clock::now() < deadline) {
        current = publication.takeReadyForDraw();
        if (!current)
            std::this_thread::yield();
    }
    REQUIRE(current);
    REQUIRE(current->worldInstanceId == NEW_INSTANCE);
    REQUIRE(current->seed == NEW_SEED);
    REQUIRE(publication.stats().buildsStarted == 2);
    REQUIRE(publication.stats().buildsCanceled == 1);
    REQUIRE(publication.stats().buildsFailed == 0);
    REQUIRE(publication.stats().lastFailureMessage.empty());
}

TEST_CASE("Cloud noise publication backs off after a throwing builder and later recovers",
          "[render][clouds][noise][async][failure][backoff][regression]") {
    constexpr uint64_t INSTANCE = 91;
    constexpr uint64_t SEED = 0xC10D'5001ULL;
    std::atomic<uint32_t> calls = 0;
    CloudNoisePublication publication([&](uint64_t worldInstanceId, uint64_t seed,
                                          std::stop_token) -> std::optional<CloudNoisePayload> {
        if (calls.fetch_add(1, std::memory_order_relaxed) == 0)
            throw std::runtime_error("deterministic cloud builder failure");
        return CloudNoisePayload{
            .worldInstanceId = worldInstanceId,
            .seed = seed,
            .base = {1},
            .erosion = {2},
            .curl = {3},
        };
    });

    publication.beginWorld(INSTANCE, SEED, true);
    const auto failureDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    CloudNoisePublicationStats failed;
    while (failed.buildsFailed == 0 && std::chrono::steady_clock::now() < failureDeadline) {
        failed = publication.stats();
        std::this_thread::yield();
    }
    REQUIRE(failed.buildsFailed == 1);
    REQUIRE(failed.buildsCanceled == 0);
    REQUIRE(failed.consecutiveFailures == 1);
    REQUIRE(failed.retryBackoffActive);
    REQUIRE_FALSE(failed.retryExhausted);
    REQUIRE(failed.retryDelayRemainingMilliseconds > 0);
    REQUIRE(failed.lastFailureBuildMilliseconds >= 0.0);
    REQUIRE(failed.lastFailureMessage == "deterministic cloud builder failure");

    std::optional<CloudNoisePayload> recovered;
    const auto recoveryDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!recovered && std::chrono::steady_clock::now() < recoveryDeadline) {
        publication.beginWorld(INSTANCE, SEED, true);
        recovered = publication.takeReadyForDraw();
        if (!recovered)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(recovered);
    REQUIRE(recovered->worldInstanceId == INSTANCE);
    REQUIRE(recovered->seed == SEED);
    REQUIRE(calls.load(std::memory_order_relaxed) == 2);
    const CloudNoisePublicationStats recoveredStats = publication.stats();
    REQUIRE(recoveredStats.buildsFailed == 1);
    REQUIRE(recoveredStats.consecutiveFailures == 0);
    REQUIRE_FALSE(recoveredStats.retryBackoffActive);
    REQUIRE_FALSE(recoveredStats.retryExhausted);
    REQUIRE(recoveredStats.lastFailureMessage == "deterministic cloud builder failure");
}

TEST_CASE("Cloud noise publication caps null retries until explicitly re-enabled",
          "[render][clouds][noise][async][failure][bounded][regression]") {
    constexpr uint64_t INSTANCE = 92;
    constexpr uint64_t SEED = 0xC10D'5002ULL;
    std::atomic<uint32_t> calls = 0;
    std::atomic<bool> recover = false;
    CloudNoisePublication publication([&](uint64_t worldInstanceId, uint64_t seed,
                                          std::stop_token) -> std::optional<CloudNoisePayload> {
        calls.fetch_add(1, std::memory_order_relaxed);
        if (!recover.load(std::memory_order_acquire))
            return std::nullopt;
        return CloudNoisePayload{
            .worldInstanceId = worldInstanceId,
            .seed = seed,
            .base = {4},
            .erosion = {5},
            .curl = {6},
        };
    });

    publication.beginWorld(INSTANCE, SEED, true);
    CloudNoisePublicationStats exhausted;
    const auto exhaustionDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!exhausted.retryExhausted && std::chrono::steady_clock::now() < exhaustionDeadline) {
        publication.beginWorld(INSTANCE, SEED, true);
        exhausted = publication.stats();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(exhausted.retryExhausted);
    REQUIRE(exhausted.buildsStarted == 3);
    REQUIRE(exhausted.buildsFailed == 3);
    REQUIRE(exhausted.buildsCanceled == 0);
    REQUIRE(exhausted.automaticAttemptsRemaining == 0);
    REQUIRE(exhausted.lastFailureMessage == "Cloud noise builder returned no payload");

    for (int poll = 0; poll < 128; ++poll)
        publication.beginWorld(INSTANCE, SEED, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    publication.beginWorld(INSTANCE, SEED, true);
    REQUIRE(calls.load(std::memory_order_relaxed) == 3);

    recover.store(true, std::memory_order_release);
    publication.setGenerationRequired(false);
    publication.setGenerationRequired(true);
    std::optional<CloudNoisePayload> recovered;
    const auto recoveryDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!recovered && std::chrono::steady_clock::now() < recoveryDeadline) {
        recovered = publication.takeReadyForDraw();
        if (!recovered)
            std::this_thread::yield();
    }
    REQUIRE(recovered);
    REQUIRE(recovered->worldInstanceId == INSTANCE);
    REQUIRE(recovered->seed == SEED);
    REQUIRE(calls.load(std::memory_order_relaxed) == 4);
    const CloudNoisePublicationStats recoveredStats = publication.stats();
    REQUIRE(recoveredStats.buildsFailed == 3);
    REQUIRE(recoveredStats.consecutiveFailures == 0);
    REQUIRE_FALSE(recoveredStats.retryExhausted);
}

TEST_CASE("Post-stack session histories restart from neutral exposure and hidden flare",
          "[render][post][session][history]") {
    constexpr ExposureState exposure = canonicalExposureHistory();
    constexpr FlareState flare = canonicalFlareHistory();
    STATIC_REQUIRE(exposure.smoothedLogLum == 0.0F);
    STATIC_REQUIRE(exposure.exposure == 1.0F);
    STATIC_REQUIRE(flare.visibility == 0.0F);
}

TEST_CASE("Underwater caustics reject wall, ceiling, and silhouette receivers",
          "[render][water][caustics]") {
    // UV-space derivative winding can produce a downward raw floor normal.
    // Facing it toward a camera above the floor restores its physical +Y side
    // before the strict receiver gate runs.
    const simd_float3 floorNormal = orientUnderwaterReceiverNormalTowardCamera(
        simd_make_float3(0.0F, -1.0F, 0.0F), simd_make_float3(0.0F, -8.0F, 0.0F));
    REQUIRE(floorNormal.y > 0.99F);
    REQUIRE(underwaterCausticSurfaceConfidence(floorNormal.y, 0.0F, 8.0F) == Catch::Approx(1.0F));

    // A ceiling viewed from below must orient downward and remain ineligible.
    const simd_float3 ceilingNormal = orientUnderwaterReceiverNormalTowardCamera(
        simd_make_float3(0.0F, 1.0F, 0.0F), simd_make_float3(0.0F, 8.0F, 0.0F));
    REQUIRE(ceilingNormal.y < -0.99F);
    REQUIRE(underwaterCausticSurfaceConfidence(ceilingNormal.y, 0.0F, 8.0F) <= 1.0e-6F);

    // Walls, opposite-oriented ceilings, and oblique normals must not turn
    // into false floors. In particular, this pins the absence of abs(normalY).
    REQUIRE(underwaterCausticSurfaceConfidence(0.0F, 0.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(-1.0F, 0.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(0.55F, 0.0F, 8.0F) <= 1.0e-6F);

    // A depth discontinuity invalidates even an otherwise up-facing estimate.
    REQUIRE(underwaterCausticSurfaceConfidence(1.0F, 4.0F, 8.0F) <= 1.0e-6F);
    REQUIRE(underwaterCausticSurfaceConfidence(std::numeric_limits<float>::quiet_NaN(), 0.0F,
                                               8.0F) <= 1.0e-6F);
}

TEST_CASE("Water SSR filters and retires only unstable grazing hits",
          "[render][water][ssr][antialiasing]") {
    // Near and non-grazing reflections preserve the full-resolution source
    // and the original narrow IGN stride range.
    REQUIRE(waterSsrReflectionMipLevel(0.0F, 0.0F) == Catch::Approx(0.0F));
    REQUIRE(waterSsrStabilityConfidence(0.0F, 1'000.0F) == Catch::Approx(1.0F));
    REQUIRE(waterSsrJitterAmplitude(0.0F) == Catch::Approx(0.24F));

    // Distant glancing rays are explicitly blurred then retire into the
    // analytic sky fallback before depth discontinuities form reflection
    // bands. Nearby and non-grazing geometry remains available to SSR.
    const float nearMip = waterSsrReflectionMipLevel(0.80F, 12.0F);
    const float farMip = waterSsrReflectionMipLevel(0.98F, 128.0F);
    REQUIRE(nearMip > 0.0F);
    REQUIRE(farMip > nearMip);
    REQUIRE(farMip <= 4.0F);
    REQUIRE(waterSsrStabilityConfidence(0.98F, 128.0F) <= 1.0e-5F);
    REQUIRE(waterSsrStabilityConfidence(0.55F, 96.0F) < 0.10F);
    REQUIRE(waterSsrStabilityConfidence(0.35F, 12.0F) > 0.90F);
    REQUIRE(waterSsrJitterAmplitude(0.98F) < waterSsrJitterAmplitude(0.50F));

    float previousMip = 0.0F;
    float previousConfidence = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float grazing = static_cast<float>(sample) / 64.0F;
        const float mip = waterSsrReflectionMipLevel(grazing, 96.0F);
        const float confidence = waterSsrStabilityConfidence(grazing, 96.0F);
        REQUIRE(mip >= previousMip);
        REQUIRE(confidence <= previousConfidence);
        REQUIRE(mip >= 0.0F);
        REQUIRE(mip <= 4.0F);
        REQUIRE(confidence >= 0.0F);
        REQUIRE(confidence <= 1.0F);
        previousMip = mip;
        previousConfidence = confidence;
    }
}

TEST_CASE("Water wave detail retires at a glancing view", "[render][water][antialiasing]") {
    REQUIRE(waterGrazingWaveDetail(1.0F) == Catch::Approx(1.0F));
    REQUIRE(waterGrazingWaveDetail(0.60F) > 0.90F);
    REQUIRE(waterGrazingWaveDetail(0.20F) < 0.05F);
    REQUIRE(waterGrazingWaveDetail(0.0F) <= 1.0e-6F);
    for (int sample = 0; sample <= 64; ++sample) {
        const float current = waterGrazingWaveDetail(static_cast<float>(sample) / 64.0F);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
    }
}

TEST_CASE("Water reflection normal filtering follows reflected-ray variation",
          "[render][water][reflection][antialiasing]") {
    // Ordinary camera projection changes only a few milliradians per pixel,
    // so resolved nearby waves retain their normal detail.
    REQUIRE(waterReflectionNormalVisibility(0.0F) == Catch::Approx(1.0F));
    REQUIRE(waterReflectionNormalVisibility(0.012F) == Catch::Approx(1.0F));
    REQUIRE(waterReflectionNormalVisibility(0.006F) > 0.99F);

    // When an analytic normal sends neighboring pixels to unrelated
    // reflection samples, retire that normal before it forms horizontal bands.
    REQUIRE(waterReflectionNormalVisibility(0.065F) <= 1.0e-6F);
    REQUIRE(waterReflectionNormalVisibility(0.25F) <= 1.0e-6F);
    REQUIRE(waterReflectionNormalVisibility(std::numeric_limits<float>::quiet_NaN()) <= 1.0e-6F);

    float previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float footprint = static_cast<float>(sample) / 256.0F;
        const float current = waterReflectionNormalVisibility(footprint);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }
}

TEST_CASE("Water refraction rejects unstable grazing receivers",
          "[render][water][refraction][antialiasing]") {
    // The close interface preserves a detailed underwater view at normal
    // incidence and on a short grazing path.
    REQUIRE(waterRefractionVisibility(0.95F, 2.0F, 32.0F, 8.0F, 0.01F, 0.25F, true) ==
            Catch::Approx(1.0F));
    REQUIRE(waterRefractionVisibility(0.24F, 2.0F, 2.0F, 8.0F, 0.01F, 0.25F, true) ==
            Catch::Approx(1.0F));

    // The far tail is a single screen-space source sample per water fragment.
    // It must fade before a long grazing path turns terrain or LOD edges into
    // a moving grid of refracted slabs.
    REQUIRE(waterRefractionVisibility(0.15F, 64.0F, 16.0F, 24.0F, 0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 16.0F, 24.0F, 0.01F, 12.0F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 64.0F, 0.01F, 0.25F, true) <= 1.0e-6F);
    // A distant top-down interface has the same one-sample opaque receiver
    // problem as a grazing interface. It must use reflection rather than
    // expose a coarse terrain tile through a small Fresnel transmission tail.
    REQUIRE(waterRefractionVisibility(0.95F, 96.0F, 32.0F, 512.0F, 0.25F, 32.0F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F, 0.2F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F, 0.01F, 0.25F, false) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(std::numeric_limits<float>::quiet_NaN(), 2.0F, 2.0F, 24.0F,
                                      0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, std::numeric_limits<float>::quiet_NaN(), 24.0F,
                                      0.01F, 0.25F, true) <= 1.0e-6F);
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 2.0F, 24.0F,
                                      std::numeric_limits<float>::quiet_NaN(), 0.25F,
                                      true) <= 1.0e-6F);

    // A shallow lake bed is long along a grazing refracted ray. Its vertical
    // depth, rather than its slant distance, must retire an under-sampled
    // coarse terrain receiver before individual cells become visible panes.
    REQUIRE(waterRefractionVisibility(0.15F, 20.0F, 2.0F, 8.0F, 0.2F, 0.25F, true) <= 1.0e-6F);

    // A receiver may remain smooth within one large terrain cell. Past the
    // nearby grazing region, retire its transmission fully rather than leave
    // a partially visible rectangular floor sample.
    REQUIRE(waterRefractionVisibility(0.15F, 2.0F, 24.0F, 32.0F, 0.01F, 0.25F, true) <= 1.0e-6F);

    // The remaining Fresnel tail must not retain the raw depth of an unstable
    // receiver, otherwise a dark voxel or LOD rectangle leaks through the
    // reflection-only fallback.
    REQUIRE(waterStabilizedOpticalDepth(24.0F, 1.0F) == Catch::Approx(24.0F));
    REQUIRE(waterStabilizedOpticalDepth(64.0F, 0.0F) == Catch::Approx(4.0F));
    REQUIRE(waterStabilizedOpticalDepth(128.0F, 1.0F) == Catch::Approx(64.0F));
    REQUIRE(waterStabilizedOpticalDepth(std::numeric_limits<float>::quiet_NaN(), 0.5F) ==
            Catch::Approx(4.0F));

    float previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float footprint = static_cast<float>(sample) * 0.25F;
        const float current =
            waterRefractionVisibility(0.16F, 20.0F, 16.0F, 24.0F, 0.01F, footprint, true);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }

    // A flat far-terrain cell has a small receiver derivative in its interior.
    // The interface-distance guard must still retire its shallow grazing
    // transmission before the cell's different opaque color reads as a pane.
    previous = 1.0F;
    for (int sample = 0; sample <= 64; ++sample) {
        const float distance = static_cast<float>(sample) * 8.0F;
        const float current =
            waterRefractionVisibility(0.16F, 2.0F, 2.0F, distance, 0.01F, 0.25F, true);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current <= previous);
        previous = current;
    }
}

TEST_CASE("Water exterior reflection gate ignores skylight nibble seams",
          "[render][water][lighting][seam]") {
    // Propagated skylight carries ambient accessibility. Open-water reflection
    // is either exterior or sealed, so harmless level differences between
    // exact and far geometry must not become a fractional reflection grid.
    REQUIRE(waterExteriorSkyVisibility(0.0F) <= 1.0e-6F);
    REQUIRE(waterExteriorSkyVisibility(1.0F / 15.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(14.0F / 15.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(1.0F) == Catch::Approx(1.0F));
    REQUIRE(waterExteriorSkyVisibility(std::numeric_limits<float>::quiet_NaN()) <= 1.0e-6F);

    float previous = 0.0F;
    for (int level = 0; level <= 15; ++level) {
        const float current = waterExteriorSkyVisibility(static_cast<float>(level) / 15.0F);
        REQUIRE(current >= 0.0F);
        REQUIRE(current <= 1.0F);
        REQUIRE(current >= previous);
        previous = current;
    }
}

TEST_CASE("Shader types: cloud layouts match MSL", "[render][shader-types]") {
    REQUIRE(sizeof(WeatherMapUniforms) == 32);
    REQUIRE(offsetof(WeatherMapUniforms, gridSize) == 16);
    REQUIRE(offsetof(WeatherMapUniforms, motionWrapBlocks) == 24);
    REQUIRE(sizeof(CloudRenderUniforms) == 352);
    REQUIRE(offsetof(CloudRenderUniforms, cameraForward) == 144);
    REQUIRE(offsetof(CloudRenderUniforms, weatherMap) == 288);
    REQUIRE(offsetof(CloudRenderUniforms, previousWeatherMap) == 320);
    REQUIRE(sizeof(CloudShadowUniforms) == 80);
    REQUIRE(offsetof(CloudShadowUniforms, weatherMap) == 48);
}

TEST_CASE("Shader types: atmospheric overhaul layouts match MSL", "[render][shader-types]") {
    REQUIRE(sizeof(AtmosphereUniforms) == 160);
    REQUIRE(offsetof(AtmosphereUniforms, cameraPositionKm) == 0);
    REQUIRE(offsetof(AtmosphereUniforms, sunDirection) == 16);
    REQUIRE(offsetof(AtmosphereUniforms, rayleighScatteringAndScaleHeight) == 64);
    REQUIRE(offsetof(AtmosphereUniforms, weatherOptics) == 128);
    REQUIRE(offsetof(AtmosphereUniforms, renderParams) == 144);

    REQUIRE(sizeof(IndirectLightingUniforms) == 336);
    REQUIRE(offsetof(IndirectLightingUniforms, projection) == 0);
    REQUIRE(offsetof(IndirectLightingUniforms, invViewProjection) == 128);
    REQUIRE(offsetof(IndirectLightingUniforms, previousViewProjection) == 192);
    REQUIRE(offsetof(IndirectLightingUniforms, resolutionAndQuality) == 256);
    REQUIRE(offsetof(IndirectLightingUniforms, traceParams) == 272);
    REQUIRE(offsetof(IndirectLightingUniforms, temporalParams) == 288);
    REQUIRE(offsetof(IndirectLightingUniforms, filterParams) == 304);
    REQUIRE(offsetof(IndirectLightingUniforms, ambientAndFrame) == 320);

    REQUIRE(sizeof(FroxelUniforms) == 384);
    REQUIRE(offsetof(FroxelUniforms, invViewProjection) == 0);
    REQUIRE(offsetof(FroxelUniforms, cameraPosition) == 192);
    REQUIRE(offsetof(FroxelUniforms, volumeDimensions) == 256);
    REQUIRE(offsetof(FroxelUniforms, depthParams) == 272);
    REQUIRE(offsetof(FroxelUniforms, renderParams) == 320);
    REQUIRE(offsetof(FroxelUniforms, physicalScale) == 336);
    REQUIRE(offsetof(FroxelUniforms, weatherMap) == 352);

    REQUIRE(sizeof(LightningUniforms) == 128);
    REQUIRE(offsetof(LightningUniforms, viewProjection) == 0);
    REQUIRE(offsetof(LightningUniforms, cameraPosition) == 64);
    REQUIRE(offsetof(LightningUniforms, strikePosition) == 80);
    REQUIRE(offsetof(LightningUniforms, colorAndIntensity) == 96);
    REQUIRE(offsetof(LightningUniforms, eventAndShape) == 112);
}

TEST_CASE("Froxel extinction uses physical meters for every generator scale",
          "[render][volumetrics][physical-scale][regression]") {
    constexpr float SCALE_HEIGHT_METERS = 800.0F;
    const float legacyAltitude = froxelAltitudeMeters(
        800.0F, static_cast<float>(LEGACY_WORLD_PHYSICAL_SCALE.positiveVerticalMetersPerBlock),
        static_cast<float>(LEGACY_WORLD_PHYSICAL_SCALE.altitudeDatumY));
    const float v4WorldY = static_cast<float>(
        GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY +
        SCALE_HEIGHT_METERS / GENERATOR_V4_PHYSICAL_SCALE.positiveVerticalMetersPerBlock);
    const float v4Altitude = froxelAltitudeMeters(
        v4WorldY, static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.positiveVerticalMetersPerBlock),
        static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY));
    REQUIRE(legacyAltitude == Catch::Approx(SCALE_HEIGHT_METERS));
    REQUIRE(v4Altitude == Catch::Approx(SCALE_HEIGHT_METERS).margin(0.001F));
    REQUIRE(froxelHeightDensity(legacyAltitude, SCALE_HEIGHT_METERS) ==
            Catch::Approx(std::exp(-1.0F)));
    REQUIRE(froxelHeightDensity(v4Altitude, SCALE_HEIGHT_METERS) == Catch::Approx(std::exp(-1.0F)));

    const float legacyDistance = froxelPhysicalDistance(
        750.0F, static_cast<float>(LEGACY_WORLD_PHYSICAL_SCALE.horizontalMetersPerBlock));
    const float v4Distance = froxelPhysicalDistance(
        100.0F, static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock));
    REQUIRE(legacyDistance == Catch::Approx(750.0F));
    REQUIRE(v4Distance == Catch::Approx(legacyDistance));
    REQUIRE(beerLambertTransmittance(0.001F, legacyDistance) ==
            Catch::Approx(beerLambertTransmittance(0.001F, v4Distance)));

    const float summitAltitude = froxelAltitudeMeters(
        1'407.0F, static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.positiveVerticalMetersPerBlock),
        static_cast<float>(GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY));
    REQUIRE(summitAltitude == Catch::Approx(10'072.5F));
    REQUIRE(froxelHeightDensity(summitAltitude, SCALE_HEIGHT_METERS) < 0.00001F);
}

TEST_CASE("Froxel media only composites onto finite receivers",
          "[render][volumetric][shader-contract]") {
    // The sky shader already integrates atmosphere to infinity. A clear sky
    // depth must preserve it rather than treating it as a far opaque surface.
    REQUIRE_FALSE(froxelHasFiniteReceiver(1.0F, 0.0F));
    REQUIRE_FALSE(froxelHasFiniteReceiver(0.999999F, 0.0F));

    REQUIRE(froxelHasFiniteReceiver(0.999F, 0.0F));
    REQUIRE(froxelHasFiniteReceiver(1.0F, 128.0F));
    REQUIRE_FALSE(froxelHasFiniteReceiver(1.0F, 65504.0F));
}

TEST_CASE("Froxel history and upscale use stable linear depth", "[render][volumetric][history]") {
    // Device depth has too little useful precision along a grazing cave floor.
    // The linear-depth threshold grows only enough to retain the same receiver.
    REQUIRE(froxelTemporalLinearDepthTolerance(2.0F) == Catch::Approx(0.05F));
    REQUIRE(froxelTemporalLinearDepthTolerance(12.0F) == Catch::Approx(0.096F));
    REQUIRE(froxelTemporalLinearDepthTolerance(96.0F) == Catch::Approx(0.768F));
    REQUIRE(froxelTemporalLinearDepthTolerance(0.0F) == 0.0F);
    REQUIRE(froxelTemporalLinearDepthTolerance(std::numeric_limits<float>::quiet_NaN()) == 0.0F);

    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.0F) == Catch::Approx(1.0F));
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.1F) > 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 12.1F) < 1.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(12.0F, 13.0F) == 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(96.0F, 97.0F) > 0.0F);
    REQUIRE(froxelBilateralLinearDepthWeight(std::numeric_limits<float>::quiet_NaN(), 12.0F) ==
            0.0F);

    // The injection sequence is deterministic for replayable captures, while
    // each dimension advances between frames to break the fixed cell grid.
    for (unsigned int dimension = 0; dimension < 3U; ++dimension) {
        const float first = froxelLowDiscrepancySample(0U, dimension);
        const float second = froxelLowDiscrepancySample(1U, dimension);
        REQUIRE(first >= 0.0F);
        REQUIRE(first < 1.0F);
        REQUIRE(second >= 0.0F);
        REQUIRE(second < 1.0F);
        REQUIRE(first != second);
        REQUIRE(froxelLowDiscrepancySample(17U, dimension) ==
                Catch::Approx(froxelLowDiscrepancySample(17U, dimension)));
    }

    // The engine's perspective matrix writes negative view Z to clip W, so
    // reprojection can compare one linear-depth authority without storing a
    // second device-depth channel beside every froxel history sample.
    const Mat4 view =
        Mat4::lookAt(Vec3{11.0F, 72.0F, -9.0F}, Vec3{20.0F, 68.0F, 14.0F}, Vec3{0.0F, 1.0F, 0.0F});
    const Mat4 projection =
        Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F, 16.0F / 9.0F, 0.1F, 1000.0F);
    const Vec4 worldPoint{18.0F, 66.0F, 20.0F, 1.0F};
    const Vec4 viewPoint = view.transformVec4(worldPoint);
    const Vec4 clipPoint = (projection * view).transformVec4(worldPoint);
    REQUIRE(std::abs(clipPoint.w) == Catch::Approx(std::abs(viewPoint.z)).margin(1.0e-5F));
}

TEST_CASE("Cloud bilateral upscale preserves transparent silhouette coverage",
          "[render][cloud][shader-contract]") {
    const simd_float2 clear = cloudCompositeTapWeights(0.25F, 0.0F, 100.0F);
    REQUIRE(clear.x == 0.0F);
    REQUIRE(clear.y == Catch::Approx(0.25F));

    const simd_float2 occluded = cloudCompositeTapWeights(0.25F, 120.0F, 100.0F);
    REQUIRE(occluded.x == 0.0F);
    REQUIRE(occluded.y == Catch::Approx(0.25F));

    const simd_float2 visible = cloudCompositeTapWeights(0.25F, 80.0F, 100.0F);
    REQUIRE(visible.x > 0.0F);
    REQUIRE(visible.x < 0.25F);
    REQUIRE(visible.y == Catch::Approx(visible.x));

    // One cloudy quarter-resolution tap beside three terrain-occluded taps
    // must retain only its bilinear coverage instead of being normalized back
    // to a fully opaque cloud pixel.
    float colorWeight = visible.x;
    float normalizationWeight = visible.y;
    for (int tap = 0; tap < 3; ++tap) {
        const simd_float2 hidden = cloudCompositeTapWeights(0.25F, 120.0F, 100.0F);
        colorWeight += hidden.x;
        normalizationWeight += hidden.y;
    }
    REQUIRE(colorWeight / normalizationWeight < 0.25F);
}

TEST_CASE("Screen-space lighting keeps a projection-invariant view-space trace radius",
          "[render][indirect][projection]") {
    auto projectionFor = [](float fovDegrees) {
        const Mat4 matrix = Mat4::perspective(fovDegrees * static_cast<float>(M_PI) / 180.0F,
                                              16.0F / 9.0F, 0.1F, 1000.0F);
        simd_float4x4 projection;
        std::memcpy(&projection, matrix.data.data(), sizeof(projection));
        return projection;
    };
    auto reconstructFromLinearDepth = [](simd_float2 uv, float linearDepth,
                                         simd_float4x4 projection) {
        const simd_float4x4 inverse = simd_inverse(projection);
        const simd_float4 farClip =
            simd_make_float4(uv.x * 2.0F - 1.0F, 1.0F - uv.y * 2.0F, 1.0F, 1.0F);
        const simd_float4 farView = simd_mul(inverse, farClip);
        const simd_float3 ray = farView.xyz / farView.w;
        return ray * (linearDepth / std::abs(ray.z));
    };

    constexpr float RADIUS = 8.0F;
    const simd_float3 direction = simd_normalize(simd_make_float3(1.0F, 0.25F, 0.0F));
    const simd_float2 resolution = simd_make_float2(3456.0F, 2234.0F);
    std::array<float, 2> nearPixels{};
    std::array<float, 2> farPixels{};
    const std::array<float, 2> fieldsOfView = {50.0F, 90.0F};

    for (size_t field = 0; field < fieldsOfView.size(); ++field) {
        const simd_float4x4 projection = projectionFor(fieldsOfView[field]);
        for (float depth : {32.0F, 96.0F}) {
            const simd_float3 origin = simd_make_float3(0.0F, 0.0F, -depth);
            const simd_float3 endpoint = screenSpaceTraceViewSample(origin, direction, RADIUS);
            const simd_float2 endpointUv = screenSpaceProjectViewPosition(endpoint, projection);
            const simd_float3 reconstructed =
                reconstructFromLinearDepth(endpointUv, std::abs(endpoint.z), projection);

            REQUIRE(simd_length(endpoint - origin) == Catch::Approx(RADIUS).margin(1.0e-5F));
            REQUIRE(simd_length(reconstructed - endpoint) < 1.0e-3F);
            const simd_float2 originUv = screenSpaceProjectViewPosition(origin, projection);
            const float projectedPixels = simd_length((endpointUv - originUv) * resolution);
            if (depth < 64.0F) {
                nearPixels[field] = projectedPixels;
            } else {
                farPixels[field] = projectedPixels;
            }
        }
    }

    // Projection changes only screen footprint. The shared helper above still
    // reconstructs the same eight-block ray at either distance or FOV.
    REQUIRE(nearPixels[0] > nearPixels[1]);
    REQUIRE(farPixels[0] > farPixels[1]);
    REQUIRE(nearPixels[0] > farPixels[0]);
    REQUIRE(nearPixels[1] > farPixels[1]);
    REQUIRE(INDIRECT_MEDIUM_RAY_COUNT == 2u);
    REQUIRE(INDIRECT_HIGH_RAY_COUNT == 4u);
    REQUIRE(INDIRECT_MEDIUM_HIZ_ITERATION_CAP == 16u);
    REQUIRE(INDIRECT_HIGH_HIZ_ITERATION_CAP == 24u);
    REQUIRE(INDIRECT_MEDIUM_ATROUS_ITERATIONS == 2u);
    REQUIRE(INDIRECT_HIGH_ATROUS_ITERATIONS == 3u);
}

TEST_CASE("Screen-space lighting bilateral upsample rejects voxel depth discontinuities",
          "[render][indirect][upsample]") {
    // Equal-depth history remains a full contribution, while nearby values
    // soften smoothly across one physical receiver.
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.1F) > 0.0F);
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 12.1F) < 1.0F);

    // A close voxel face is outside the hard bilateral interval, so a
    // lower-resolution history sample cannot darken or brighten its neighbor.
    REQUIRE(screenSpaceBilateralDepthWeight(12.0F, 13.0F) <= 1.0e-6F);
    REQUIRE(screenSpaceBilateralDepthWeight(128.0F, 140.0F) <= 1.0e-6F);
    REQUIRE(screenSpaceBilateralDepthWeight(std::numeric_limits<float>::quiet_NaN(), 12.0F) <=
            1.0e-6F);
}

TEST_CASE("Screen-space lighting fallback keeps a compatible voxel-face owner",
          "[render][indirect][upsample]") {
    const simd_float3 floorNormal = simd_make_float3(0.0F, 1.0F, 0.0F);
    const simd_float3 wallNormal = simd_make_float3(1.0F, 0.0F, 0.0F);

    // The regular footprint can contain only a perpendicular wall and reject
    // it completely. A nearby coplanar candidate is safe for the bounded
    // no-owner fallback, while a different depth remains rejected.
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, wallNormal) == 0.0F);
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 12.0F, floorNormal, floorNormal) > 0.9F);
    REQUIRE(screenSpaceJointBilateralUpsampleWeight(12.0F, 13.0F, floorNormal, floorNormal) <=
            1.0e-6F);
}

TEST_CASE("Screen-space history uses linear depth for grazing cave floors",
          "[render][indirect][history]") {
    // The bounded linear tolerance admits a continuous receiver at distance
    // while remaining much smaller than a different voxel face.
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(2.0F) == Catch::Approx(0.04F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(12.0F) == Catch::Approx(0.12F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(96.0F) == Catch::Approx(0.96F));
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(0.0F) == 0.0F);
    REQUIRE(screenSpaceTemporalLinearDepthTolerance(std::numeric_limits<float>::quiet_NaN()) ==
            0.0F);
}

TEST_CASE("Hi-Z traversal steps cells and classifies exact hits", "[render][indirect][hiz]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();

    // Axis crossing with the epsilon nudge landing in the next cell.
    {
        const simd_float2 position = simd_make_float2(4.3F, 7.9F);
        const simd_float2 direction = simd_make_float2(1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(exit == Catch::Approx(0.75F).margin(0.06F));
        REQUIRE(std::floor(position.x + direction.x * exit) == 5.0F);
    }
    // Diagonal ray exits through the nearer boundary.
    {
        const simd_float2 position = simd_make_float2(0.5F, 0.5F);
        const simd_float2 direction = simd_normalize(simd_make_float2(2.0F, 1.0F));
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(std::floor(position.x + direction.x * exit) == 1.0F);
        REQUIRE(std::floor(position.y + direction.y * exit) == 0.0F);
    }
    // Negative direction crosses the low boundary.
    {
        const simd_float2 position = simd_make_float2(4.3F, 7.9F);
        const simd_float2 direction = simd_make_float2(-1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 1.0F);
        REQUIRE(std::floor(position.x + direction.x * exit) == 3.0F);
    }
    // Coarser mip levels step whole cells at once.
    {
        const simd_float2 position = simd_make_float2(5.0F, 6.0F);
        const simd_float2 direction = simd_make_float2(1.0F, 0.0F);
        const float exit = screenSpaceHiZCellExit(position, direction, 4.0F);
        REQUIRE(exit == Catch::Approx(3.0F).margin(0.06F));
    }
    REQUIRE(screenSpaceHiZCellExit(simd_make_float2(1.0F, 1.0F), simd_make_float2(1.0F, 0.0F),
                                   0.0F) == Catch::Approx(0.05F));

    // Reciprocal depth interpolation matches a real perspective projection:
    // the point on the 3D segment at the helper's midpoint depth projects to
    // the screen-space midpoint of the segment's endpoints.
    {
        const Mat4 matrix = Mat4::perspective(70.0F * static_cast<float>(M_PI) / 180.0F,
                                              16.0F / 9.0F, 0.1F, 1000.0F);
        simd_float4x4 projection;
        std::memcpy(&projection, matrix.data.data(), sizeof(projection));
        const simd_float3 start = simd_make_float3(1.0F, 0.5F, -4.0F);
        const simd_float3 end = simd_make_float3(3.0F, -1.0F, -20.0F);
        const simd_float2 startUv = screenSpaceProjectViewPosition(start, projection);
        const simd_float2 endUv = screenSpaceProjectViewPosition(end, projection);
        const float midDepth = screenSpaceHiZRayDepth(0.5F, 4.0F, 20.0F);
        REQUIRE(midDepth == Catch::Approx(1.0F / ((0.25F + 0.05F) * 0.5F)));
        const float along = (midDepth - 4.0F) / 16.0F;
        const simd_float3 midPoint = start + (end - start) * along;
        const simd_float2 midUv = screenSpaceProjectViewPosition(midPoint, projection);
        REQUIRE(midUv.x == Catch::Approx((startUv.x + endUv.x) * 0.5F).margin(1.0e-3F));
        REQUIRE(midUv.y == Catch::Approx((startUv.y + endUv.y) * 0.5F).margin(1.0e-3F));
    }
    REQUIRE(screenSpaceHiZRayDepth(0.0F, 4.0F, 20.0F) == Catch::Approx(4.0F));
    REQUIRE(screenSpaceHiZRayDepth(1.0F, 4.0F, 20.0F) == Catch::Approx(20.0F));
    REQUIRE(screenSpaceHiZRayDepth(0.5F, NAN_VALUE, 20.0F) ==
            Catch::Approx(INDIRECT_SKY_LINEAR_DEPTH));

    // Empty-cell classification, including a ray moving toward the camera.
    REQUIRE(screenSpaceHiZAdvances(5.0F, 6.0F, 6.5F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(5.0F, 7.0F, 6.5F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(8.0F, 5.0F, 7.0F));
    REQUIRE_FALSE(screenSpaceHiZAdvances(NAN_VALUE, 6.0F, 6.5F));

    // Exact mip-zero receiver test.
    REQUIRE(screenSpaceHiZSurfaceHit(10.05F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(9.9F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(10.3F, 10.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(65000.0F, 65504.0F, 0.15F));
    REQUIRE_FALSE(screenSpaceHiZSurfaceHit(NAN_VALUE, 10.0F, 0.15F));
}

TEST_CASE("Cosine hemisphere rays stay above the receiver surface", "[render][indirect][rays]") {
    const simd_float3 normal = simd_normalize(simd_make_float3(0.3F, 0.9F, -0.2F));
    double cosineSum = 0.0;
    int sampleCount = 0;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            const simd_float2 xi = simd_make_float2((static_cast<float>(i) + 0.5F) / 16.0F,
                                                    (static_cast<float>(j) + 0.5F) / 16.0F);
            const simd_float3 direction = screenSpaceCosineHemisphereDirection(xi, normal);
            REQUIRE(simd_length(direction) == Catch::Approx(1.0F).margin(1.0e-4F));
            REQUIRE(simd_dot(direction, normal) > 0.0F);
            cosineSum += simd_dot(direction, normal);
            ++sampleCount;
        }
    }
    // The cosine-weighted density has an exact mean cosine of two thirds.
    REQUIRE(cosineSum / sampleCount == Catch::Approx(2.0 / 3.0).margin(0.02));

    const simd_float3 fallback = screenSpaceCosineHemisphereDirection(
        simd_make_float2(0.3F, 0.7F), simd_make_float3(0.0F, 0.0F, 0.0F));
    REQUIRE(fallback.z == Catch::Approx(1.0F));

    // R2 sequence samples stay in the unit square and move a meaningful
    // distance every frame so a pixel's rays never clump.
    const simd_float2 noise = simd_make_float2(0.42F, 0.17F);
    simd_float2 previous = screenSpaceRaySequenceSample(0, noise);
    for (uint32_t index = 1; index < 8; ++index) {
        const simd_float2 sample = screenSpaceRaySequenceSample(index, noise);
        REQUIRE(sample.x >= 0.0F);
        REQUIRE(sample.x < 1.0F);
        REQUIRE(sample.y >= 0.0F);
        REQUIRE(sample.y < 1.0F);
        REQUIRE(std::abs(sample.x - previous.x) + std::abs(sample.y - previous.y) > 0.05F);
        previous = sample;
    }
}

TEST_CASE("Temporal blend ramps with age and clamps fireflies", "[render][indirect][temporal]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(screenSpaceTemporalBlendWeight(0.0F, 0.90F) == 0.0F);
    REQUIRE(screenSpaceTemporalBlendWeight(1.0F, 0.90F) == Catch::Approx(0.5F));
    REQUIRE(screenSpaceTemporalBlendWeight(3.0F, 0.90F) == Catch::Approx(0.75F));
    REQUIRE(screenSpaceTemporalBlendWeight(9.0F, 0.90F) == Catch::Approx(0.90F));
    REQUIRE(screenSpaceTemporalBlendWeight(INDIRECT_HISTORY_MAX_AGE, 0.90F) ==
            Catch::Approx(0.90F));
    REQUIRE(screenSpaceTemporalBlendWeight(NAN_VALUE, 0.90F) == 0.0F);

    REQUIRE(screenSpaceLuminanceVariance(0.5F, 0.25F) == 0.0F);
    REQUIRE(screenSpaceLuminanceVariance(0.5F, 0.50F) == Catch::Approx(0.25F));
    REQUIRE(screenSpaceLuminanceVariance(NAN_VALUE, 1.0F) == 0.0F);

    REQUIRE(screenSpaceFireflyClampScale(2.0F, 4.0F) == 1.0F);
    REQUIRE(screenSpaceFireflyClampScale(16.0F, 4.0F) == Catch::Approx(0.25F));
    REQUIRE(screenSpaceFireflyClampScale(NAN_VALUE, 4.0F) == 0.0F);
}

TEST_CASE("Variance clamp collapses stale history over a converged neighborhood",
          "[render][indirect][history]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    // A converged region has near-zero deviation, so a stale bright ghost is
    // clamped to the floor within one frame.
    REQUIRE(screenSpaceVarianceClampHalfRange(0.0F, 2.0F, 0.001F) == Catch::Approx(0.001F));
    // A genuinely sparse bright source keeps a wide clamp because its
    // accumulated variance stays high.
    REQUIRE(screenSpaceVarianceClampHalfRange(0.5F, 2.0F, 0.001F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceVarianceClampHalfRange(NAN_VALUE, 2.0F, 0.001F) == Catch::Approx(0.001F));

    // Young pixels take the wider of the spatial and temporal estimates so
    // disocclusion opens the spatial filter instead of trusting two samples.
    REQUIRE(screenSpaceVarianceForAge(0.01F, 0.2F, 1.0F, 4.0F) == Catch::Approx(0.2F));
    REQUIRE(screenSpaceVarianceForAge(0.01F, 0.2F, 8.0F, 4.0F) == Catch::Approx(0.01F));
}

TEST_CASE("A-trous edge weight stops at voxel edges and follows variance",
          "[render][indirect][denoise]") {
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.0F, 1.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceAtrousEdgeWeight(2.0F, 0.5F, 1.0F, 0.0F, 1.0F) == 0.0F);
    REQUIRE(screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 0.0F, 0.0F, 1.0F) == 0.0F);
    const float tight = screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.5F, 0.25F);
    const float loose = screenSpaceAtrousEdgeWeight(0.0F, 0.5F, 1.0F, 0.5F, 1.0F);
    REQUIRE(loose > tight);
    REQUIRE(screenSpaceAtrousEdgeWeight(NAN_VALUE, 0.5F, 1.0F, 0.0F, 1.0F) == 0.0F);

    REQUIRE(screenSpaceOcclusionFalloff(0.0F, 8.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceOcclusionFalloff(8.0F, 8.0F) == 0.0F);
    REQUIRE(screenSpaceOcclusionFalloff(2.0F, 8.0F) > screenSpaceOcclusionFalloff(6.0F, 8.0F));

    REQUIRE(screenSpaceBounceSourceWeight(-0.5F, 4.0F, 24.0F) == 0.0F);
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 4.0F, 24.0F) == Catch::Approx(1.0F));
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 24.0F, 24.0F) == Catch::Approx(0.0F));
    REQUIRE(screenSpaceBounceSourceWeight(1.0F, 19.0F, 24.0F) >
            screenSpaceBounceSourceWeight(1.0F, 23.0F, 24.0F));
    REQUIRE(screenSpaceBounceSourceWeight(NAN_VALUE, 4.0F, 24.0F) == 0.0F);
}
