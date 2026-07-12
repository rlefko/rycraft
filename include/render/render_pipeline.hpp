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
#include "render/texture_atlas.hpp"
#include "render/vertex.hpp"

// Forward declarations
class World;
class Camera;
class UIOverlay;

// GPU-side per-chunk mesh allocation tracking.
struct ChunkMeshState {
    MegaBuffer::ChunkAllocation alloc;
    bool uploaded = false;
};

// Sky uniforms for day/night cycle
struct SkyUniforms {
    float zenithColor[3];
    float horizonColor[3];
    float sunDirection[3];
    float sunColor[3];
    float sunIntensity;
    float padding;
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

    // MSAA textures (4x) and single-sample resolve targets.
    id<MTLTexture> _colorMSAA;
    id<MTLTexture> _colorResolve;
    id<MTLTexture> _depthMSAA;
    id<MTLTexture> _depthResolve;

    // Uniform buffer (256 bytes — fits Uniforms struct with padding).
    id<MTLBuffer> _uniformsBuffer;

    // MegaBuffer for centralized GPU memory management.
    MegaBuffer* _megaBuffer;

    // Texture atlas for procedural block textures.
    TextureAtlas* _textureAtlas;

    // UI overlay for HUD rendering (crosshair, hotbar).
    UIOverlay* _uiOverlay;

    uint32_t _width;
    uint32_t _height;

    // ---- Frustum culling ----
    void extractFrustumPlanes(const Mat4& vpMatrix);
    bool isChunkInFrustum(const AABB& chunkAABB) const;
    float _frustumPlanes[6][4];

    // ---- Chunk mesh cache ----
    // Key: packed int64 ((uint32_t)chunkX << 32 | (uint32_t)chunkZ)
    std::unordered_map<uint64_t, ChunkMeshState> _chunkMeshes;

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
                      const float ambientColor[3]);

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
};
