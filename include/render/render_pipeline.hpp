#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/math.hpp"
#include "engine/hotbar.hpp"
#include "render/block_texture_array.hpp"
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
class EntityRenderer;
class World;
class Camera;
class UIOverlay;
class Bloom;
class PostStack;
class ShadowMap;
class Volumetrics;
class Ssao;
class ParticleSystem;

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

// One chunk's water draw, recorded during the opaque pass and encoded by
// the water pass after the scene resolves.
struct WaterDraw {
    simd_float4 origin;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint64_t vertexOffset;
    uint64_t indexOffset;
    uint32_t indexCount;
    float distSq; // camera distance², for back-to-front ordering
};

// ---------------------------------------------------------------------------
// RenderPipeline — Metal renderer with a single 4x MSAA scene pass.
//
// Frame structure:
//   1. Scene pass (MSAA, native resolution): sky → chunks → block highlight
//      → weather particles → clouds, resolved into _colorResolve
//   2. Bloom: extract/blur/composite from _colorResolve into the drawable
//      (plain blit when bloom intensity is zero)
//   3. UI overlay pass onto the drawable
//
// Also owns frustum culling and the on-demand chunk mesh cache.
// ---------------------------------------------------------------------------
class RenderPipeline {
public:
    RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                   uint32_t height);

    ~RenderPipeline();

    // Render a single frame.
    // Handles empty world gracefully (sky-only output).
    void render(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const World& world, const Camera& camera,
                uint64_t worldTime = 0, std::optional<Vec3> highlightedBlock = std::nullopt,
                const Hotbar& hotbar = Hotbar(), const UIFrameState& uiFrame = UIFrameState{},
                const std::vector<std::shared_ptr<Entity>>* entities = nullptr);

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
    void tickParticles(float dt, const World& world, const Vec3& playerPosition, bool raining);

    // Exponential fog density per block (settings menu).
    void setFogDensity(float density) { _fogDensity = density; }

    // Rain wetness 0-1 from the engine's weather state (darkens albedo and
    // adds a sun sheen in the chunk shader).
    void setWetness(float wetness) { _wetness = wetness; }

    // Write the next presented frame to `path` as a PNG (async, off the
    // render thread). Used by the playtest workflow for headless visual
    // verification — macOS screen-recording permissions don't apply.
    void requestFrameCapture(const std::string& path);

    // Chunk streaming counters for the F3 HUD (render thread only).
    struct ChunkRenderStats {
        float meshMsAvg = 0.f;
        uint32_t meshBuildsLastFrame = 0;
        float megaUsedMB = 0.f;
        float megaCapMB = 0.f;
    };
    ChunkRenderStats chunkRenderStats() const { return _chunkStats; }

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
    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // Sky pipeline state (drawn first in the scene pass, behind everything)
    id<MTLRenderPipelineState> _skyPipelineState;
    id<MTLDepthStencilState> _skyDepthState;

    // Depth-tested but non-writing state (block highlight)
    id<MTLDepthStencilState> _noDepthWriteState;

    // Block highlight pipeline state (wireframe lines)
    id<MTLRenderPipelineState> _highlightPipelineState;
    id<MTLBuffer> _highlightVertexBuffer;

    // Cloud pipeline state (Phase 8)
    id<MTLRenderPipelineState> _cloudPipelineState;
    id<MTLDepthStencilState> _cloudDepthState;

    // Water pass (refraction/reflection/caustics) — no depth attachment;
    // the fragment shader depth-tests against the resolved scene depth
    id<MTLRenderPipelineState> _waterPipelineState;
    id<MTLRenderPipelineState> _underwaterOverlayState;
    std::vector<WaterDraw> _waterDraws; // reused each frame

    // MSAA render targets (memoryless — resolved or discarded at pass end)
    id<MTLTexture> _colorMSAA;
    id<MTLTexture> _depthMSAA;

    // Single-sample resolve target feeding bloom / the drawable blit
    id<MTLTexture> _colorResolve;

    // Water pass inputs: the opaque scene's resolved depth, and a copy of
    // the resolved color the refraction samples (a render target cannot
    // sample itself)
    id<MTLTexture> _depthResolve;
    id<MTLTexture> _sceneColorCopy;

    // Frames-in-flight gate + per-frame constants arena: every uniform block
    // the CPU rewrites per frame sub-allocates from the current slot.
    FrameRing _frameRing;

    // The frame's chunk Uniforms allocation — filled by renderChunks, also
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

    // Ray-marched volumetric light shafts (skipped when the setting is off)
    std::unique_ptr<Volumetrics> _volumetrics;

    // Screen-space ambient occlusion (skipped when the setting is off)
    std::unique_ptr<Ssao> _ssao;

    // The shadow sampling block the scene pass binds each frame — the
    // computed cascades when shadows are on, or a zeroed (strength 0) block
    // when off/faded so the chunk fragment reads full sun without branching.
    ShadowUniforms _sceneShadowUniforms{};

    // Weather particle system (rain/snow)
    std::unique_ptr<ParticleSystem> _particles;

    // Animal voxel-model renderer
    std::unique_ptr<EntityRenderer> _entityRenderer;

    // GPU frame/pass timing (per-pass sampling only under RYCRAFT_GPU_COUNTERS)
    std::unique_ptr<GpuFrameTimer> _gpuTimer;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _bloomIntensity;

    // Video settings copy, pushed by the engine (render thread only)
    GraphicsSettings _gfx;

    // Exponential fog density per block
    float _fogDensity = 0.0003f;
    float _wetness = 0.0f;

    // Frame animation clock (worldTime -> seconds, wraps daily) driving the
    // foliage sway in the scene AND shadow passes — one value per frame so
    // the two can never sample different phases.
    float _animTime = 0.0f;

    // Drawable dimensions (the scene renders at native resolution)
    uint32_t _displayWidth;
    uint32_t _displayHeight;

    // Pending frame-capture destination (empty when no capture is queued)
    std::string _capturePath;

    // (Re)allocate the MSAA + resolve textures at the current drawable size.
    void allocateSceneTargets();

    // Encode the drawable readback + async PNG write for requestFrameCapture.
    void encodeFrameCapture(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> frameTexture);

    // ---- Frustum culling ----
    void extractFrustumPlanes(const Mat4& vpMatrix);
    bool isChunkInFrustum(const AABB& chunkAABB) const;
    float _frustumPlanes[6][4];

    // ---- Chunk mesh cache ----
    // Key: packed int64 ((uint32_t)chunkX << 32 | (uint32_t)chunkZ).
    // An entry with uploaded == false marks a chunk whose mesh is empty
    // (all air) so it is not rebuilt every frame.
    std::unordered_map<uint64_t, ChunkMeshState> _chunkMeshes;

    // Scratch set reused each frame to sweep meshes of unloaded chunks
    // without per-frame allocation.
    std::unordered_set<uint64_t> _liveChunkKeys;

    // HUD counters, written only by the render thread during renderChunks.
    ChunkRenderStats _chunkStats;

    // Reused meshing buffers (render thread only) — snapshot + scratch keep
    // their capacity across builds instead of reallocating ~85 KB each time.
    // (These serve the synchronous edit fast path; workers have their own.)
    MeshSnapshot _meshSnapshot;
    MeshScratch _meshScratch;

    // Async meshing: workers build, renderChunks drains + uploads. Both
    // vectors are reused across frames.
    std::unique_ptr<MeshScheduler> _meshScheduler;
    std::vector<MeshResult> _pendingResults;
    std::vector<std::pair<float, const Chunk*>> _meshCandidates;

    // ---- Day/Night Cycle ----
    // sunDirection/sunColor come out as the ACTIVE directional light — the sun
    // by day, the moon (dim cool light) by night — so terrain shading and
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
                      const float sunColor[3], const float ambientColor[3],
                      const float fogColor[3]);

    // Encode the cascade depth passes before the scene pass (no-op when
    // shadowQuality is 0 or strength is 0). lightDirection is the active
    // directional light (sun by day, moon by night). Fills
    // _sceneShadowUniforms for the chunk fragment; reuses loadedChunks so the
    // locked chunk-list copy happens once per frame.
    void renderShadows(id<MTLCommandBuffer> commandBuffer,
                       const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                       const Camera& camera, const float lightDirection[3], float strength);

    void renderBlockHighlight(id<MTLRenderCommandEncoder> encoder, const Vec3& blockPos,
                              const Mat4& viewMatrix, const Mat4& projectionMatrix);

    // Water pass: own encoder after the scene pass resolves. Encodes the
    // recorded _waterDraws plus the underwater overlay when submerged.
    void renderWater(id<MTLCommandBuffer> commandBuffer, const Mat4& viewMatrix,
                     const Mat4& projectionMatrix, const Vec3& cameraPosition,
                     bool cameraUnderwater, const SkyUniforms& skyUniforms,
                     const float fogColor[3]);

    void renderUIOverlay(id<MTLRenderCommandEncoder> encoder, const Hotbar& hotbar,
                         const UIFrameState& uiFrame);

    void renderClouds(id<MTLRenderCommandEncoder> encoder, const Camera& camera, uint64_t worldTime,
                      const float sunDirection[3], float sunIntensity);
};
