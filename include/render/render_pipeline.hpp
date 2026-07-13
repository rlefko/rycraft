#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/math.hpp"
#include "engine/hotbar.hpp"
#include "render/mega_buffer.hpp"
#include "render/particles.hpp"
#include "render/shader_types.hpp"
#include "render/texture_atlas.hpp"
#include "render/vertex.hpp"

// Forward declarations
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

// MetalFX Upscaler stub (Phase 8.4 — placeholder for Phase 9.4 optimization)
class MetalFXUpscaler {
public:
    MetalFXUpscaler(id<MTLDevice> /*device*/, uint32_t /*srcWidth*/, uint32_t /*srcHeight*/,
                    uint32_t /*dstWidth*/, uint32_t /*dstHeight*/);
    ~MetalFXUpscaler();

    // Upscale source texture to destination using bilinear sampling.
    // (Placeholder — full MetalFX temporal upscaling in Phase 9.4)
    void upscale(id<MTLCommandBuffer> commandBuffer,
                 id<MTLTexture> source,
                 id<MTLTexture> destination);
};

// ---------------------------------------------------------------------------
// RenderPipeline — Full Metal render pass with MSAA 4x and frustum culling.
//
// Responsibilities:
//   • Create and manage render/depth pipeline states
//   • Allocate MSAA + resolve textures
//   • Upload uniforms (model/view/projection/lighting) each frame
//   • Frustum-cull chunks before drawing
//   • Mesh dirty chunks on-demand and upload to GPU
//   • Render sky, water, block highlight, and UI overlay
//   • Post-processing: bloom, fog, clouds (Phase 8)
//   • Render target upscaling preparation (Phase 8.4)
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
                const Hotbar& hotbar = Hotbar());

    // Reallocate MSAA and resolve textures for new viewport size.
    void resize(uint32_t width, uint32_t height);

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    // When zero, the bloom pass is skipped entirely (saves 13 render passes).
    void setBloomIntensity(float intensity);
    float getBloomIntensity() const { return _bloomIntensity; }

    // Update particle system physics (call each game tick).
    void tickParticles(float dt, const World& world, const Vec3& playerPosition);

private:
    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // Sky pipeline state
    id<MTLRenderPipelineState> _skyPipelineState;
    id<MTLBuffer> _skyUniformsBuffer;

    // Water pipeline state (transparent pass)
    id<MTLRenderPipelineState> _waterPipelineState;
    id<MTLDepthStencilState> _waterDepthState;

    // Block highlight pipeline state (wireframe lines)
    id<MTLRenderPipelineState> _highlightPipelineState;
    id<MTLBuffer> _highlightVertexBuffer;
    id<MTLBuffer> _highlightUniformsBuffer;

    // Cloud pipeline state (Phase 8)
    id<MTLRenderPipelineState> _cloudPipelineState;
    id<MTLDepthStencilState> _cloudDepthState;
    id<MTLBuffer> _cloudUniformsBuffer;

    // MSAA textures (multisample, at render resolution).
    id<MTLTexture> _colorMSAA;
    id<MTLTexture> _depthMSAA;

    // Resolve textures (single-sample, at render resolution).
    // With upscaling: render at half-resolution, upscale to display.
    id<MTLTexture> _colorResolve;
    id<MTLTexture> _depthResolve;

    // Uniform buffer (512 bytes with fog + camera position).
    id<MTLBuffer> _uniformsBuffer;

    // MegaBuffer for centralized GPU memory management.
    MegaBuffer* _megaBuffer;

    // Texture atlas for procedural block textures.
    TextureAtlas* _textureAtlas;

    // UI overlay for HUD rendering (crosshair, hotbar).
    UIOverlay* _uiOverlay;

    // Bloom post-processing (Phase 8)
    Bloom* _bloom;

    // MetalFX upscaler (Phase 8.4)
    MetalFXUpscaler* _upscaler;

    // Weather particle system (rain/snow)
    ParticleSystem* _particles;

    // Bloom intensity multiplier (0.0 = disabled, 1.0 = full strength).
    float _bloomIntensity;

    // Render target dimensions (may differ from display for upscaling)
    uint32_t _renderWidth;
    uint32_t _renderHeight;
    uint32_t _displayWidth;
    uint32_t _displayHeight;

    // ---- Frustum culling ----
    void extractFrustumPlanes(const Mat4& vpMatrix);
    bool isChunkInFrustum(const AABB& chunkAABB) const;
    float _frustumPlanes[6][4];

    // ---- Chunk mesh cache (per-LOD) ----
    // Outer key: packed int64 ((uint32_t)chunkX << 32 | (uint32_t)chunkZ)
    // Inner key: LOD level (0-2), enables multiple mesh resolutions per chunk.
    std::unordered_map<uint64_t, std::unordered_map<int, ChunkMeshState>> _chunkMeshes;

    // ---- Day/Night Cycle (Task 6.4-6.5) ----
    void computeDayNightUniforms(uint64_t worldTime,
                                  float sunDirection[3],
                                  float sunColor[3],
                                  float ambientColor[3],
                                  SkyUniforms& skyUniforms);

    // ---- Render passes ----
    void renderSky(id<MTLCommandBuffer> commandBuffer,
                   id<CAMetalDrawable> drawable,
                   const SkyUniforms& skyUniforms);

    void renderChunks(id<MTLRenderCommandEncoder> encoder,
                      const World& world,
                      const Mat4& viewMatrix,
                      const Mat4& projectionMatrix,
                      const float sunDirection[3],
                      const float sunColor[3],
                      const float ambientColor[3],
                      const float fogColor[3]);

    void renderWater(id<MTLCommandBuffer> commandBuffer,
                      id<CAMetalDrawable> drawable,
                      const Mat4& viewMatrix,
                     const Mat4& projectionMatrix,
                     const float sunDirection[3],
                     const float sunColor[3],
                     const float ambientColor[3]);

    void renderBlockHighlight(id<MTLRenderCommandEncoder> encoder,
                              const Vec3& blockPos,
                              const Mat4& viewMatrix,
                              const Mat4& projectionMatrix);

    void renderUIOverlay(id<MTLRenderCommandEncoder> encoder,
                         const Hotbar& hotbar);

    // ---- Phase 8: Clouds ----
    void renderClouds(id<MTLCommandBuffer> commandBuffer,
                      id<CAMetalDrawable> drawable,
                      const Mat4& viewMatrix,
                      const Mat4& projectionMatrix,
                      const Camera& camera,
                      uint64_t worldTime,
                      const float sunDirection[3]);
};
