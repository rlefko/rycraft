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
#include "render/ui_menu.hpp"
#include "render/mega_buffer.hpp"
#include "render/particles.hpp"
#include "render/shader_types.hpp"
#include "render/block_texture_array.hpp"
#include "render/vertex.hpp"

// Forward declarations
class Entity;
class EntityRenderer;
class World;
class Camera;
class UIOverlay;
class Bloom;
class ParticleSystem;

// GPU-side per-chunk mesh allocation tracking.
struct ChunkMeshState {
    MegaBuffer::ChunkAllocation alloc;
    bool uploaded = false;
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
    RenderPipeline(id<MTLDevice> device,
                    id<MTLLibrary> shaderLibrary,
                    uint32_t width,
                    uint32_t height);

    ~RenderPipeline();

    // Render a single frame.
    // Handles empty world gracefully (sky-only output).
    void render(id<MTLCommandQueue> queue,
                id<CAMetalDrawable> drawable,
                const Mat4& viewMatrix,
                const Mat4& projectionMatrix,
                const World& world,
                const Camera& camera,
                uint64_t worldTime = 0,
                std::optional<Vec3> highlightedBlock = std::nullopt,
                const Hotbar& hotbar = Hotbar(),
                const UIFrameState& uiFrame = UIFrameState{},
                const std::vector<std::shared_ptr<Entity>>* entities = nullptr);

    // Reallocate MSAA and resolve textures for new viewport size.
    void resize(uint32_t width, uint32_t height);

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When zero, the bloom pass is skipped entirely (saves 13 render passes).
    void setBloomIntensity(float intensity);
    float getBloomIntensity() const { return _bloomIntensity; }

    // Update particle system physics (call each game tick).
    void tickParticles(float dt, const World& world, const Vec3& playerPosition);

    // Exponential fog density per block (settings menu).
    void setFogDensity(float density) { _fogDensity = density; }

    // Write the next presented frame to `path` as a PNG (async, off the
    // render thread). Used by the playtest workflow for headless visual
    // verification — macOS screen-recording permissions don't apply.
    void requestFrameCapture(const std::string& path);

private:
    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // Sky pipeline state (drawn first in the scene pass, behind everything)
    id<MTLRenderPipelineState> _skyPipelineState;
    id<MTLDepthStencilState> _skyDepthState;
    id<MTLBuffer> _skyUniformsBuffer;

    // Depth-tested but non-writing state (block highlight)
    id<MTLDepthStencilState> _noDepthWriteState;

    // Block highlight pipeline state (wireframe lines)
    id<MTLRenderPipelineState> _highlightPipelineState;
    id<MTLBuffer> _highlightVertexBuffer;
    id<MTLBuffer> _highlightUniformsBuffer;

    // Cloud pipeline state (Phase 8)
    id<MTLRenderPipelineState> _cloudPipelineState;
    id<MTLDepthStencilState> _cloudDepthState;
    id<MTLBuffer> _cloudUniformsBuffer;

    // MSAA render targets (memoryless — resolved or discarded at pass end)
    id<MTLTexture> _colorMSAA;
    id<MTLTexture> _depthMSAA;

    // Single-sample resolve target feeding bloom / the drawable blit
    id<MTLTexture> _colorResolve;

    // Uniform buffer (512 bytes with fog + camera position).
    id<MTLBuffer> _uniformsBuffer;

    // MegaBuffer for centralized GPU memory management.
    MegaBuffer* _megaBuffer;

    // Array texture of procedural block textures.
    BlockTextureArray* _blockTextures;

    // UI overlay for HUD rendering (crosshair, hotbar).
    UIOverlay* _uiOverlay;

    // Bloom post-processing (Phase 8)
    Bloom* _bloom;

    // Weather particle system (rain/snow)
    ParticleSystem* _particles;

    // Animal voxel-model renderer
    EntityRenderer* _entityRenderer;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _bloomIntensity;

    // Exponential fog density per block
    float _fogDensity = 0.0003f;

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

    // ---- Day/Night Cycle (Task 6.4-6.5) ----
    void computeDayNightUniforms(uint64_t worldTime,
                                  float sunDirection[3],
                                  float sunColor[3],
                                  float ambientColor[3],
                                  SkyUniforms& skyUniforms);

    // ---- Scene pass stages (all encode into the single MSAA scene encoder) ----
    void renderSky(id<MTLRenderCommandEncoder> encoder);

    void renderChunks(id<MTLRenderCommandEncoder> encoder,
                      const World& world,
                      const Mat4& viewMatrix,
                      const Mat4& projectionMatrix,
                      const Vec3& cameraPosition,
                      const float sunDirection[3],
                      const float sunColor[3],
                      const float ambientColor[3],
                      const float fogColor[3]);

    void renderBlockHighlight(id<MTLRenderCommandEncoder> encoder,
                              const Vec3& blockPos,
                              const Mat4& viewMatrix,
                              const Mat4& projectionMatrix);

    void renderUIOverlay(id<MTLRenderCommandEncoder> encoder,
                         const Hotbar& hotbar,
                         const UIFrameState& uiFrame);

    void renderClouds(id<MTLRenderCommandEncoder> encoder,
                      const Camera& camera,
                      uint64_t worldTime,
                      const float sunDirection[3]);
};
