#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "render/shader_types.hpp"

// ---------------------------------------------------------------------------
// ShadowMap — cascaded sun/moon shadow maps.
//
// Owns the depth array texture (SHADOW_CASCADE_COUNT slices), the depth-only
// cutout-aware chunk pipeline, and the cascade math.
// Each frame computeCascades() fits a stable, texel-snapped ortho box around
// each slice of the camera frustum; the caller then encodes one depth pass per
// cascade, and reads shadowUniforms() into the scene pass for PCF sampling.
//
// RenderPipeline drives the geometry (it owns the mega-buffer); this class
// owns the targets, pipelines, and math so render_pipeline.mm stays smaller.
// ---------------------------------------------------------------------------
class ShadowMap {
public:
    // vertexDescriptor is the shared chunk vertex layout (owned by
    // RenderPipeline) so the shadow pass reads the same 16-byte vertices.
    ShadowMap(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
              MTLVertexDescriptor* vertexDescriptor);

    // (Re)size each cascade's depth slice. Called on a quality change.
    void setResolution(uint32_t resolution);
    uint32_t resolution() const { return _resolution; }

    // Fit the cascades to the current camera + light. lightDir points FROM the
    // scene TO the light (sun by day, moon by night). shadowDistance caps how
    // far cascades reach; strength scales the shadow term (0 = fully lit).
    void computeCascades(const Vec3& cameraPos, const Vec3& cameraForward, const Vec3& cameraRight,
                         const Vec3& cameraUp, float fovY, float aspect, const Vec3& lightDir,
                         float shadowDistance, float strength);

    // Per-cascade light view-projection (for the depth pass) and the packed
    // sampling block (for the scene pass).
    const Mat4& cascadeViewProj(int cascade) const { return _cascadeVP[cascade]; }
    const ShadowUniforms& shadowUniforms() const { return _shadowUniforms; }

    id<MTLTexture> depthTexture() const { return _depthTexture; }
    id<MTLDepthStencilState> depthState() const { return _depthState; }
    id<MTLRenderPipelineState> chunkPipeline() const { return _chunkPipeline; }
    id<MTLSamplerState> comparisonSampler() const { return _comparisonSampler; }

    // A render pass descriptor targeting one cascade slice (depth-only clear).
    MTLRenderPassDescriptor* passDescriptor(int cascade) const;

    // World-space AABB test against a cascade's ortho volume, extruded toward
    // the light so casters behind the frustum still draw. Reused for culling.
    bool cascadeContains(int cascade, const struct AABB& aabb) const;

private:
    id<MTLDevice> _device;
    id<MTLTexture> _depthTexture; // Depth32Float, 2D array [SHADOW_CASCADE_COUNT]
    id<MTLDepthStencilState> _depthState;
    id<MTLRenderPipelineState> _chunkPipeline;
    id<MTLSamplerState> _comparisonSampler;
    uint32_t _resolution = 2048;

    Mat4 _cascadeVP[SHADOW_CASCADE_COUNT];
    ShadowUniforms _shadowUniforms{};

    void allocateTexture();
};
