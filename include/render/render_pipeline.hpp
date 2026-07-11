#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/math.hpp"
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

// ---------------------------------------------------------------------------
// RenderPipeline — Full Metal render pass with MSAA 4x and frustum culling.
//
// Responsibilities:
//   • Create and manage render/depth pipeline states
//   • Allocate MSAA + resolve textures
//   • Upload uniforms (model/view/projection/lighting) each frame
//   • Frustum-cull chunks before drawing
//   • Mesh dirty chunks on-demand and upload to GPU
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
                const Camera& camera);

    // Reallocate MSAA and resolve textures for new viewport size.
    void resize(uint32_t width, uint32_t height);

private:
    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // MSAA textures (4×) and single-sample resolve targets.
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
    // Key: "chunkX,chunkZ"  (matches World::chunkKey format)
    std::unordered_map<std::string, ChunkMeshState> _chunkMeshes;
};
