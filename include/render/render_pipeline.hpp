#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/math.hpp"
#include "render/block_texture_array.hpp"
#include "render/far_terrain.hpp"
#include "render/frame_ring.hpp"
#include "render/gpu_timer.hpp"
#include "render/graphics_settings.hpp"
#include "render/lod_mesher.hpp"
#include "render/mega_buffer.hpp"
#include "render/mesh_scheduler.hpp"
#include "render/particles.hpp"
#include "render/shader_types.hpp"
#include "render/ui_menu.hpp"
#include "render/vertex.hpp"
#include "world/mesh_snapshot.hpp"

// Forward declarations
class Entity;
struct ItemEntity;
struct Boat;
class EntityRenderer;
class ItemEntityRenderer;
class BoatRenderer;
class World;
class Camera;
class UIOverlay;
class Bloom;
class PostStack;
class ShadowMap;
class Volumetrics;
class ScreenSpaceLighting;
class AtmosphereRenderer;
class CloudRenderer;
class LightningRenderer;
class ParticleSystem;
class WeatherSnapshot;
struct IndirectHistoryState;
struct WeatherSample;

// GPU-side per-chunk mesh allocation tracking. opaqueIndexCount splits the
// one allocation into the opaque section and the water section (see
// MeshOutput). builtVersion/requestedVersion carry the staleness protocol:
// a chunk re-meshes whenever its content version differs from the mesh it
// is drawing, and a version already requested isn't requested twice.
struct ChunkMeshState {
    MegaBuffer::ChunkAllocation alloc;
    uint32_t opaqueIndexCount = 0;
    uint32_t builtVersion = 0;     // 0 = nothing built yet
    uint32_t requestedVersion = 0; // 0 = no build in flight
    bool uploaded = false;
};

// Worker results may complete out of order across renderer drains. Publish
// only a result for the currently loaded revision, and never let it replace a
// resident result from a later serial revision. Revision zero remains the
// "nothing built" sentinel used by ChunkMeshState.
constexpr bool chunkMeshAsyncResultCanReplace(uint32_t builtVersion, uint32_t liveVersion,
                                              uint32_t residentVersion) {
    return builtVersion == liveVersion &&
           (residentVersion == 0 || static_cast<int32_t>(builtVersion - residentVersion) > 0);
}

// A completion owns only the request revision captured when it was queued.
// An older completion must not clear a newer request for the same cube.
constexpr uint32_t chunkMeshRequestAfterCompletion(uint32_t pendingVersion,
                                                   uint32_t completedRequestVersion) {
    return pendingVersion == completedRequestVersion ? 0 : pendingVersion;
}

// A hard-cap admission guard shared by exact mesh upload and scheduling.
// Published exact owners are not eviction candidates. When every resident
// entry owns its section, new work waits instead of growing past the cap.
constexpr bool chunkMeshRegistryCanAdmit(size_t residentCount, size_t capacity,
                                         bool alreadyResident, bool hasUnownedVictim) {
    if (residentCount > capacity) return false;
    return alreadyResident || residentCount < capacity || hasUnownedVictim;
}

// GPU allocation for one immutable 256x256-block far-terrain tile. Far tiles
// use the same compact vertex format and shader as exact cubes, but live in a
// separate bounded arena so horizon streaming can never evict collision-zone
// meshes.
struct FarTerrainMeshState {
    MegaBuffer::ChunkAllocation alloc;
    FarTerrainBounds bounds;
    FarTerrainBounds surfaceBounds;
    std::array<FarTerrainBounds, FAR_TERRAIN_OCCLUDER_PATCH_COUNT> occluderPatches{};
    uint32_t opaqueIndexCount = 0;
    float complexity = 0.0F;
    uint64_t deterministicHash = 0;
    bool uploaded = false;
};

struct FarTerrainLodTransition {
    FarTerrainKey from;
    FarTerrainKey to;
    double startedAtSeconds = 0.0;
};

// One chunk's water draw, recorded during the opaque pass and encoded by
// the water pass after the scene resolves.
struct WaterDraw {
    simd_float4 origin;
    simd_float4 overlayColorAndStrength;
    simd_uint4 farMetadata;
    FarTerrainOwnershipUniforms farOwnership;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint64_t vertexOffset;
    uint64_t indexOffset;
    uint32_t indexCount;
    float distSq; // camera distance², for back-to-front ordering
};

// ---------------------------------------------------------------------------
// RenderPipeline owns the integrated Metal frame graph. Weather, atmosphere,
// cloud shadows, and terrain shadows are prepared before the native 4x MSAA
// sky and opaque pass. Screen-space lighting, clouds, lightning, water,
// froxels, and weather particles then composite into resolved HDR before
// exposure, flare, bloom, grading, and UI produce the drawable.
//   3. UI overlay pass onto the drawable
//
// Also owns frustum culling and the on-demand chunk mesh cache.
// ---------------------------------------------------------------------------
class RenderPipeline {
public:
    RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                   uint32_t height, uint64_t worldSeed = 0);

    ~RenderPipeline();

    // Render a single frame.
    // Handles empty world gracefully (sky-only output).
    void render(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const World& world, const Camera& camera,
                uint64_t worldTime = 0, double deltaSeconds = 0.0,
                std::optional<Vec3> highlightedBlock = std::nullopt,
                const UIFrameState& uiFrame = UIFrameState{},
                const std::vector<std::shared_ptr<Entity>>* entities = nullptr,
                const std::vector<ItemEntity>* itemEntities = nullptr,
                const std::vector<Boat>* boats = nullptr,
                std::shared_ptr<const WeatherSnapshot> weatherSnapshot = nullptr,
                const std::vector<LightningEvent>* lightningEvents = nullptr);

    // Menu-only frame when no world session is live: one single-sample pass
    // that clears the drawable to the backdrop color and draws the UI
    // overlay. No HDR, no depth, no world reads.
    void renderMenuOnly(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable,
                        const UIFrameState& uiFrame);

    // Detach every renderer structure that references or is keyed by the
    // current World: the mesh scheduler (it captures a const World& lazily
    // at first render), pending mesh results, resident cube meshes, exact
    // ownership, and the recorded far-terrain identity so the next session
    // rebuilds even under an equal seed. Must run before the World dies.
    void endWorldSession();

    // Reallocate MSAA and resolve textures for new viewport size.
    void resize(uint32_t width, uint32_t height);

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When zero, the bloom pass is skipped entirely (saves 13 render passes).
    void setBloomIntensity(float intensity);
    float getBloomIntensity() const { return _bloomIntensity; }

    // Push the current video settings; disabled effects skip their passes
    // wholesale on the next frame. Called at init and on every change.
    void setGraphicsSettings(const GraphicsSettings& gfx);

    // Update particle system physics (call each game tick).
    void tickParticles(float dt, const World& world, const Vec3& playerPosition,
                       const WeatherSample& weather);

    // Exponential fog density per block (settings menu).
    void setFogDensity(float density) { _fogDensity = density; }

    // Rain wetness 0-1 from the engine's weather state (darkens albedo and
    // adds a sun sheen in the chunk shader).
    void setWetness(float wetness) { _wetness = wetness; }

    // Write the next presented frame to `path` as a PNG (async, off the
    // render thread). Used by the playtest workflow for headless visual
    // verification, macOS screen-recording permissions don't apply.
    void requestFrameCapture(const std::string& path);

    // Chunk streaming counters for the F3 HUD (render thread only).
    struct ChunkRenderStats {
        float meshMsAvg = 0.f;
        uint32_t meshBuildsLastFrame = 0;
        float megaUsedMB = 0.f;
        float megaCapMB = 0.f;
        uint32_t meshCubeCount = 0;
        uint32_t meshPendingCount = 0;
        uint32_t meshQueueHighWater = 0;
        uint64_t meshCoalescedCount = 0;
        uint64_t meshDroppedStaleCount = 0;
        uint32_t exactSurfaceRequiredCount = 0;
        uint32_t exactSurfaceReadyCount = 0;
        uint32_t exactSurfaceUnresolvedColumnCount = 0;
        float exactSurfaceHandoffBlocks = 0.f;
        uint32_t farWantedTileCount = 0;
        uint32_t farResidentTileCount = 0;
        uint32_t farBaseWantedTileCount = 0;
        uint32_t farBaseResidentTileCount = 0;
        uint32_t farBaseDrawnTileCount = 0;
        uint32_t farBaseMissingTileCount = 0;
        uint32_t farRefinementWantedTileCount = 0;
        uint32_t farRefinementResidentTileCount = 0;
        uint32_t farRefinementDrawnTileCount = 0;
        uint32_t farDrawnTileCount = 0;
        uint32_t farFrustumCulledTileCount = 0;
        uint32_t farOcclusionCulledTileCount = 0;
        uint32_t farPendingTileCount = 0;
        uint32_t farUploadsLastFrame = 0;
        uint32_t farQueuedBaseTileCount = 0;
        uint32_t farQueuedRefinementTileCount = 0;
        uint32_t farActiveBaseWorkerCount = 0;
        uint32_t farReservedBaseWorkerCount = 0;
        uint32_t farActiveUrgentRefinementCount = 0;
        uint32_t farWorkerBudget = 0;
        uint32_t farCachedBaseTileCount = 0;
        float farCoverageFrontierBlocks = 0.f;
        float farCacheMB = 0.f;
        float farMegaUsedMB = 0.f;
    };
    ChunkRenderStats chunkRenderStats() const { return _chunkStats; }
    FarTerrainGenerationCacheStats farGenerationCacheStats() const;

    struct AtmosphericRenderStats {
        uint32_t shadowRefreshMask = 0;
        std::array<uint32_t, SHADOW_CASCADE_COUNT> shadowCasterCounts{};
        std::array<uint64_t, SHADOW_CASCADE_COUNT> shadowRefreshCounts{};
        uint32_t indirectHistoryResetMask = 0;
        bool indirectHistoryValid = false;
        bool cloudHistoryValid = false;
        bool froxelHistoryValid = false;
        uint64_t atmosphereSlowRefreshCount = 0;
        uint64_t atmosphereSkyRefreshCount = 0;
        uint64_t indirectPersistentBytes = 0;
        uint64_t cloudPersistentBytes = 0;
        uint64_t froxelPersistentBytes = 0;
        uint64_t integratedPersistentBytes = 0;
        uint64_t lightningEventId = 0;
        float lunarPhaseEnergy = 0.0F;
        float lunarPhaseCycle = 0.0F;
    };
    AtmosphericRenderStats atmosphericRenderStats() const;

    // Real GPU frame time (EMA over completed command buffers) for the F3
    // HUD and the 60-frame diagnostic log.
    float gpuFrameMs() const { return _gpuTimer->frameMsEma(); }

    // Per-pass GPU breakdown; empty unless RYCRAFT_GPU_COUNTERS is set.
    std::string gpuPassBreakdown() const { return _gpuTimer->passBreakdown(); }

    // Stop the mesh workers (they reference the World). The engine calls
    // this on the quit path BEFORE the world is destroyed; the scheduler's
    // destructor also calls it defensively.
    void shutdownMeshWorkers();

private:
    enum class WorldgenOverlayMode : uint8_t {
        NONE,
        GEOLOGY,
        HYDROLOGY,
        CLIMATE,
        BIOME,
    };

    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState{};
    id<MTLDepthStencilState> _depthState{};

    // Sky pipeline state (drawn first in the scene pass, behind everything)
    id<MTLRenderPipelineState> _skyPipelineState{};
    id<MTLDepthStencilState> _skyDepthState{};

    // Depth-tested but non-writing state (block highlight)
    id<MTLDepthStencilState> _noDepthWriteState{};

    // Block highlight pipeline state (wireframe lines)
    id<MTLRenderPipelineState> _highlightPipelineState{};
    id<MTLBuffer> _highlightVertexBuffer{};

    // Water reads the resolved opaque depth for refraction and rejection, then
    // hardware depth-tests and writes the nearest interface into media depth.
    id<MTLRenderPipelineState> _waterPipelineState{};
    id<MTLRenderPipelineState> _underwaterOverlayState{};
    std::vector<WaterDraw> _waterDraws; // reused each frame

    // MSAA render targets (memoryless, resolved or discarded at pass end)
    id<MTLTexture> _colorMSAA{};
    id<MTLTexture> _surfaceMSAA{};
    id<MTLTexture> _depthMSAA{};

    // Single-sample resolve target feeding bloom / the drawable blit
    id<MTLTexture> _colorResolve{};
    id<MTLTexture> _surfaceResolve{};

    // Water pass inputs: the opaque scene's resolved depth, a water-inclusive
    // media depth, and a copy of the resolved color the refraction samples.
    id<MTLTexture> _depthResolve{};
    id<MTLTexture> _mediaDepthResolve{};
    id<MTLTexture> _sceneColorCopy{};

    // Frames-in-flight gate + per-frame constants arena: every uniform block
    // the CPU rewrites per frame sub-allocates from the current slot.
    FrameRing _frameRing;

    // The frame's chunk Uniforms allocation, filled by renderChunks, also
    // bound by the entity renderer and the water pass vertex stage.
    FrameRing::Alloc _frameUniforms;

    // MegaBuffer for centralized GPU memory management.
    std::unique_ptr<MegaBuffer> _megaBuffer;

    // Array texture of procedural block textures.
    std::unique_ptr<BlockTextureArray> _blockTextures;

    // UI overlay for HUD rendering (crosshair, hotbar, menus).
    std::unique_ptr<UIOverlay> _uiOverlay;

    // Bloom post-processing (HDR extract + blur)
    std::unique_ptr<Bloom> _bloom;

    // Final composite: exposure, tonemap, grade, sharpen (always runs)
    std::unique_ptr<PostStack> _postStack;

    // Cascaded sun/moon shadow maps (skipped when shadowQuality is 0)
    std::unique_ptr<ShadowMap> _shadowMap;

    // Unified atmospheric froxel volume.
    std::unique_ptr<Volumetrics> _volumetrics;

    // Near-field GTAO and diffuse screen-space indirect lighting.
    std::unique_ptr<ScreenSpaceLighting> _screenSpaceLighting;

    // Physical atmosphere LUTs and true volumetric cloud layers.
    std::unique_ptr<AtmosphereRenderer> _atmosphere;
    std::unique_ptr<CloudRenderer> _clouds;
    std::unique_ptr<LightningRenderer> _lightning;

    // The shadow sampling block the scene pass binds each frame, the
    // computed cascades when shadows are on, or a zeroed (strength 0) block
    // when off/faded so the chunk fragment reads full sun without branching.
    ShadowUniforms _sceneShadowUniforms{};

    // Weather particle system (rain/snow)
    std::unique_ptr<ParticleSystem> _particles;

    // Animal voxel-model renderer
    std::unique_ptr<EntityRenderer> _entityRenderer;
    std::unique_ptr<ItemEntityRenderer> _itemEntityRenderer;
    std::unique_ptr<BoatRenderer> _boatRenderer;

    // GPU frame/pass timing (per-pass sampling only under RYCRAFT_GPU_COUNTERS)
    std::unique_ptr<GpuFrameTimer> _gpuTimer;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _bloomIntensity;

    // Video settings copy, pushed by the engine (render thread only)
    GraphicsSettings _gfx;

    // Exponential fog density per block
    float _fogDensity = 0.00015f;
    float _wetness = 0.0f;
    uint32_t _shadowRefreshMask = 0;
    std::array<uint32_t, SHADOW_CASCADE_COUNT> _shadowCasterCounts{};
    uint32_t _indirectHistoryResetMask = 0;
    uint64_t _lastLightningEventId = 0;
    float _lunarPhaseEnergy = 0.0F;
    float _lunarPhaseCycle = 0.0F;
    float _directSpecularFactor = 0.0F;
    uint8_t _activeCelestialSource = 0;
    // True while the camera is submerged this frame (set by render()). Gates
    // the rain-wetness sun sheen off: a gloss toward the sun on floors seen
    // through five blocks of water read as a white-out, not rain.
    bool _cameraUnderwater = false;
    // Eased sky exposure of the camera's water column (see render()); 0 in
    // sealed aquifers and under roofed water where sunlight cannot reach.
    float _uwSkyExposure = 1.0f;
    // World Y of the surface of the water body the camera is in (see render()).
    float _uwSurfaceY = 0.0f;
    simd_float4x4 _previousViewProjection = matrix_identity_float4x4;
    std::unique_ptr<IndirectHistoryState> _indirectHistoryState;
    uint64_t _previousWorldTime = 0;
    uint64_t _forcedStateRevision = 0;
    uint8_t _previousWeatherPreset = 0;
    bool _hasPreviousWorldTime = false;
    bool _weatherSnapshotWasPresent = false;

    // Frame animation clock driving water waves, caustics, and foliage sway in
    // the scene AND shadow passes, one value per frame so the two can never
    // sample different phases. It accumulates the real frame delta (NOT the
    // day-night worldTime), so animation keeps flowing when the time of day is
    // frozen (captures) or paused and never jumps at the daily rollover. Bounded
    // (wraps at 3600 s) so the float keeps sub-millisecond phase precision.
    float _animTime = 0.0f;
    double _animClock = 0.0;

    // Optional deterministic developer overlay selected by
    // RYCRAFT_WORLDGEN_OVERLAY.
    WorldgenOverlayMode _worldgenOverlayMode = WorldgenOverlayMode::NONE;

    // Drawable dimensions (the scene renders at native resolution)
    uint32_t _displayWidth;
    uint32_t _displayHeight;

    // Pending frame-capture destination (empty when no capture is queued)
    std::string _capturePath;

    // (Re)allocate the MSAA + resolve textures at the current drawable size.
    void allocateSceneTargets();
    void releaseSceneTargets();

    // Encode the drawable readback + async PNG write for requestFrameCapture.
    void encodeFrameCapture(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> frameTexture);

    // ---- Frustum culling ----
    void extractFrustumPlanes(const Mat4& vpMatrix);
    bool isChunkInFrustum(const AABB& chunkAABB) const;
    float _frustumPlanes[6][4];

    // ---- Chunk mesh cache ----
    // An entry with uploaded == false marks a cube whose mesh is empty
    // (all air) so it is not rebuilt every frame.
    std::unordered_map<ChunkPos, ChunkMeshState> _chunkMeshes;

    // Scratch map reused each frame to sweep unloaded meshes and validate
    // asynchronous results against the loaded chunk's current revision.
    std::unordered_map<ChunkPos, const Chunk*> _liveChunksByPosition;

    // HUD counters, written only by the render thread during renderChunks.
    ChunkRenderStats _chunkStats;

    // Reused meshing buffers (render thread only), snapshot + scratch keep
    // their capacity across builds instead of reallocating ~85 KB each time.
    // (These serve the synchronous edit fast path; workers have their own.)
    MeshSnapshot _meshSnapshot;
    MeshScratch _meshScratch;

    // Async meshing: workers build, renderChunks drains + uploads. Both
    // vectors are reused across frames.
    std::unique_ptr<MeshScheduler> _meshScheduler;
    std::vector<MeshResult> _pendingResults;
    std::vector<std::pair<float, const Chunk*>> _meshCandidates;
    // Once a revision-matched exact mesh is published, it retains ownership
    // while that immutable mesh remains resident. A later edit can enqueue a
    // replacement without briefly resurfacing far terrain underneath it.
    std::unordered_set<ChunkPos> _exactOwnedSections;
    FarTerrainExactCoverageCache _farTerrainExactCoverage;

    // ---- Far-terrain LOD annulus ----
    // Exact cubic terrain stops at radius 32. Immutable 256x256-block tiles
    // cover the remaining visible annulus with 2, 4, 8, and 16-block sampling.
    // CPU construction, CPU caching, and GPU residency all have independent
    // hard bounds so the 512-chunk horizon remains inside the 64 GB target.
    std::unique_ptr<FarTerrainScheduler> _farTerrainScheduler;
    std::unique_ptr<SegmentedMegaBuffer> _farMegaBuffer;
    std::optional<uint64_t> _farTerrainSeed;
    std::optional<ColumnPos> _farTerrainCenterTile;
    std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash> _farTerrainMeshes;
    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> _farTerrainWanted;
    std::vector<FarTerrainKey> _farTerrainPriorityOrder;
    std::unordered_set<ColumnPos> _farTerrainActiveTiles;
    std::unordered_map<ColumnPos, FarTerrainKey> _farTerrainDesiredByTile;
    std::unordered_map<ColumnPos, FarTerrainKey> _farTerrainDisplayedByTile;
    std::unordered_map<ColumnPos, float> _farTerrainComplexityByTile;
    std::unordered_map<ColumnPos, FarTerrainLodTransition> _farTerrainTransitions;
    struct FarShadowDrawPlan {
        ColumnPos coordinate;
        FarTerrainKey key;
        simd_uint4 farMetadata{};
    };
    // Exact copy of the prior color pass's eligible far draw plans. Shadows
    // encode earlier in the next frame, so retaining this authority keeps LOD
    // transitions and disconnected coverage from diverging.
    std::vector<FarShadowDrawPlan> _farShadowDrawPlans;
    // Preallocated render-thread storage avoids one node allocation whenever
    // a cold near tile begins its intermediate-refinement grace period.
    std::vector<std::pair<ColumnPos, double>> _farTerrainNearGraceStartedAt;
    std::vector<FarTerrainResult> _farTerrainResults;
    std::vector<FarTerrainViewTile> _farTerrainCandidates;
    std::vector<FarTerrainKey> _farTerrainCachedBaseRequests;
    std::vector<FarTerrainRefinementCacheRequest> _farTerrainUrgentRefinementRequests;
    std::vector<FarTerrainKey> _farTerrainUrgentRefinementKeys;
    std::vector<FarTerrainRefinementCacheRequest> _farTerrainCachedRefinementRequests;
    std::vector<std::shared_ptr<const FarTerrainMesh>> _farTerrainCachedMeshes;
    size_t _farTerrainResidentWantedCount = 0;
    size_t _farTerrainResidentRefinementCount = 0;

    void renderFarTerrain(id<MTLRenderCommandEncoder> encoder, const World& world,
                          const Vec3& cameraPosition, const float fogColor[3]);
    void resetFarTerrain(uint64_t worldSeed, GenerationSettings generation = {});
    void setExactSectionOwned(ChunkPos position, bool owned);
    void clearExactSectionOwnership();

    // ---- Day/Night Cycle ----
    // sunDirection/sunColor come out as the ACTIVE directional light, the sun
    // by day, the moon (dim cool light) by night, so terrain shading and
    // shadows share one light. The sky keeps the real sun/moon positions for
    // its discs. shadowStrength is the cascade term's weight (0 at the horizon
    // crossing so the sun→moon swap never pops).
    void computeDayNightUniforms(uint64_t worldTime, float sunDirection[3], float sunColor[3],
                                 float ambientColor[3], SkyUniforms& skyUniforms,
                                 float& shadowStrength);

    // ---- Scene pass stages (all encode into the single MSAA scene encoder) ----
    void renderSky(id<MTLRenderCommandEncoder> encoder, const FrameRing::Alloc& skyUniforms);

    void renderChunks(id<MTLRenderCommandEncoder> encoder, const World& world,
                      const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                      const Mat4& viewMatrix, const Mat4& projectionMatrix,
                      const Vec3& cameraPosition, const float sunDirection[3],
                      const float sunColor[3], const float ambientColor[3], const float fogColor[3],
                      const FoliageWindUniforms& foliageWind);

    // Encode the cascade depth passes before the scene pass (no-op when
    // shadowQuality is 0 or strength is 0). lightDirection is the active
    // directional light (sun by day, moon by night). Fills
    // _sceneShadowUniforms for the chunk fragment; reuses loadedChunks so the
    // locked chunk-list copy happens once per frame.
    void renderShadows(id<MTLCommandBuffer> commandBuffer,
                       const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                       const std::vector<std::shared_ptr<Entity>>* entities, const Camera& camera,
                       const float lightDirection[3], float strength,
                       const FoliageWindUniforms& foliageWind);

    void renderBlockHighlight(id<MTLRenderCommandEncoder> encoder, const Vec3& blockPos,
                              const Mat4& viewMatrix, const Mat4& projectionMatrix);

    // Water pass: own encoder after the scene pass resolves. Encodes the
    // recorded _waterDraws plus the underwater overlay when submerged.
    void renderWater(id<MTLCommandBuffer> commandBuffer, const Mat4& viewMatrix,
                     const Mat4& projectionMatrix, const Vec3& cameraPosition,
                     bool cameraUnderwater, const SkyUniforms& skyUniforms,
                     const float directLightDirection[3], const float directLightRadiance[3],
                     const float fogColor[3]);

    void renderUIOverlay(id<MTLRenderCommandEncoder> encoder, const UIFrameState& uiFrame);
};
