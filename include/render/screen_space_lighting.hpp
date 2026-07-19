#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "render/shader_types.hpp"

#include <cstdint>

class GpuFrameTimer;

enum IndirectHistoryReset : uint32_t {
    INDIRECT_HISTORY_STABLE = 0,
    INDIRECT_HISTORY_RESIZE = 1U << 0U,
    INDIRECT_HISTORY_TELEPORT = 1U << 1U,
    INDIRECT_HISTORY_WORLD_CHANGE = 1U << 2U,
    INDIRECT_HISTORY_FOV_CHANGE = 1U << 3U,
    INDIRECT_HISTORY_QUALITY_CHANGE = 1U << 4U,
    INDIRECT_HISTORY_FORCED_STATE = 1U << 5U,
    INDIRECT_HISTORY_INVALID_DEPTH = 1U << 6U,
    INDIRECT_HISTORY_LIGHT_SOURCE = 1U << 7U,
};

struct IndirectHistoryState {
    uint32_t width = 0;
    uint32_t height = 0;
    Vec3 cameraPosition{};
    float fovDegrees = 0.0F;
    uint64_t worldIdentity = 0;
    uint64_t forcedStateRevision = 0;
    int quality = 0;
    uint8_t directLightSource = 0;
    bool priorDepthValid = false;
};

uint32_t indirectHistoryResetMask(const IndirectHistoryState& previous,
                                  const IndirectHistoryState& current);

// Texture payload retained by the screen-space lighting pass. Driver page
// alignment is device-specific and is intentionally outside this budget.
struct ScreenSpaceLightingMemoryFootprint {
    uint32_t workWidth = 1;
    uint32_t workHeight = 1;
    uint64_t neutralBytes = 0;
    uint64_t linearDepthPyramidBytes = 0;
    // Full-resolution octahedral view normals guide the joint bilateral
    // reconstruction. Keeping this separate from surface data preserves that
    // attachment's albedo-plus-accessibility contract.
    uint64_t normalBytes = 0;
    uint64_t traceBytes = 0;
    uint64_t historyBytes = 0;
    uint64_t historyDepthBytes = 0;
    // Luminance moments, accumulation age, and variance ride in a ping-pong
    // pair beside the color history; the scratch target ping-pongs the
    // a-trous wavelet iterations without touching the temporal feedback.
    uint64_t momentsBytes = 0;
    uint64_t scratchBytes = 0;

    uint64_t totalBytes() const noexcept {
        return neutralBytes + linearDepthPyramidBytes + normalBytes + traceBytes + historyBytes +
               historyDepthBytes + momentsBytes + scratchBytes;
    }
};

ScreenSpaceLightingMemoryFootprint
screenSpaceLightingMemoryFootprint(uint32_t width, uint32_t height, int quality) noexcept;

// The reduced-resolution trace can reconstruct only from a candidate that
// describes the same visible receiver. Depth alone cannot distinguish two
// perpendicular voxel faces at the same view distance, which smears a bright
// cave-floor bounce onto the wall. This CPU mirror pins the normal gate used
// by the full-resolution joint bilateral apply shader.
inline float screenSpaceBilateralNormalWeight(const Vec3& receiverNormal,
                                              const Vec3& candidateNormal) noexcept {
    const float receiverLengthSquared = receiverNormal.lengthSq();
    const float candidateLengthSquared = candidateNormal.lengthSq();
    if (!std::isfinite(receiverLengthSquared) || !std::isfinite(candidateLengthSquared) ||
        receiverLengthSquared <= 1.0e-8F || candidateLengthSquared <= 1.0e-8F) {
        return 0.0F;
    }
    const float agreement =
        std::clamp(receiverNormal.dot(candidateNormal) /
                       std::sqrt(receiverLengthSquared * candidateLengthSquared),
                   -1.0F, 1.0F);
    // Preserve a gently changing receiver while fully rejecting a separate
    // voxel face. The cubic curve avoids a visible threshold on shallow
    // slopes as the camera moves.
    const float t = std::clamp((agreement - 0.65F) / 0.25F, 0.0F, 1.0F);
    return t * t * (3.0F - 2.0F * t);
}

inline float screenSpaceJointBilateralUpsampleWeight(const float receiverDepth,
                                                     const float candidateDepth,
                                                     const Vec3& receiverNormal,
                                                     const Vec3& candidateNormal) noexcept {
    return screenSpaceBilateralDepthWeight(receiverDepth, candidateDepth) *
           screenSpaceBilateralNormalWeight(receiverNormal, candidateNormal);
}

class ScreenSpaceLighting {
public:
    ScreenSpaceLighting(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                        uint32_t height);
    ~ScreenSpaceLighting();

    void resize(uint32_t width, uint32_t height);
    void setQuality(int quality);
    void resetHistory(uint32_t reasons);

    // The surface texture is resolved RGBA8: diffuse albedo in RGB and baked
    // ambient accessibility in A. Direct, block, and emissive radiance remain
    // in sceneHDR and are never multiplied by this pass.
    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> depthResolve, id<MTLTexture> surfaceResolve,
                const IndirectLightingUniforms& uniforms, GpuFrameTimer* timer = nullptr);

    bool historyValid() const { return _historyValid; }
    uint32_t lastResetReasons() const { return _lastResetReasons; }
    uint32_t workWidth() const { return _workWidth; }
    uint32_t workHeight() const { return _workHeight; }
    uint64_t persistentBytes() const { return _persistentBytes; }

private:
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _linearDepthPipeline;
    id<MTLComputePipelineState> _depthReducePipeline;
    id<MTLComputePipelineState> _normalPipeline;
    id<MTLComputePipelineState> _tracePipeline;
    id<MTLComputePipelineState> _temporalPipeline;
    id<MTLComputePipelineState> _atrousPipeline;
    id<MTLComputePipelineState> _historyDepthPipeline;
    id<MTLRenderPipelineState> _applyPipeline;
    id<MTLTexture> _linearDepthPyramid{};
    id<MTLTexture> _normalTexture{};
    id<MTLTexture> _traceTexture{};
    id<MTLTexture> _history[2]{};
    id<MTLTexture> _historyDepth[2]{};
    id<MTLTexture> _momentsAge[2]{};
    id<MTLTexture> _denoiseScratch{};
    id<MTLTexture> _neutralTexture;
    uint32_t _displayWidth;
    uint32_t _displayHeight;
    uint32_t _workWidth = 1;
    uint32_t _workHeight = 1;
    int _quality = 0;
    uint32_t _historyIndex = 0;
    bool _historyValid = false;
    uint32_t _lastResetReasons = INDIRECT_HISTORY_RESIZE;
    uint64_t _persistentBytes = 0;

    void allocateTargets();
    static void dispatch2D(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline, NSUInteger width,
                           NSUInteger height);
};
