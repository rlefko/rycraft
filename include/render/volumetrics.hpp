#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"

#include <algorithm>
#include <cstdint>

class GpuFrameTimer;

constexpr uint64_t volumetricMemoryBytes(uint32_t width, uint32_t height) noexcept {
    const uint32_t halfWidth = std::max(width / 2U, 1U);
    const uint32_t halfHeight = std::max(height / 2U, 1U);
    const uint64_t froxelBytes =
        static_cast<uint64_t>(FROXEL_WIDTH) * FROXEL_HEIGHT * FROXEL_DEPTH * 8U * 2U;
    // Current scattering, R32 linear view depth, and two history pairs. The
    // linear depth targets cost two extra bytes per half-resolution pixel over
    // device depth, but keep grazing cave history stable instead of exposing
    // the fixed froxel grid.
    const uint64_t halfPixelBytes =
        static_cast<uint64_t>(halfWidth) * halfHeight * (8U + 4U + 2U * (8U + 4U));
    // Neutral atmosphere, shadow, hit-depth, and two-slice weather texels.
    return froxelBytes + halfPixelBytes + 27U;
}

// Unified air-medium rendering. A fixed logarithmic froxel volume receives
// weather extinction plus directional light visibility, then integrates into
// a temporally filtered half-resolution scattering/transmittance image. The
// final blend is scattering + scene * transmittance. Water absorption remains
// in the water renderer, and submerged cameras explicitly disable this air
// volume so the two media cannot overlap.
class Volumetrics {
public:
    static constexpr uint32_t GRID_WIDTH = FROXEL_WIDTH;
    static constexpr uint32_t GRID_HEIGHT = FROXEL_HEIGHT;
    static constexpr uint32_t GRID_DEPTH = FROXEL_DEPTH;

    Volumetrics(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                uint32_t height);
    ~Volumetrics();

    void resize(uint32_t width, uint32_t height);
    void resetHistory();

    // cloudShadowTransmittance is the snapped, camera-centered cloud-shadow
    // texture. It may be nil, in which case a neutral transmittance is used.
    // When enableFroxels is false, the same call applies a low-cost analytic
    // aerial perspective without allocating or sampling the froxel history.
    // FroxelUniforms uses these component contracts:
    //   depthParams: near, far, cloud footprint, water surface Y (-65536 = none)
    //   mediumParams: base extinction, scattering albedo, anisotropy, height scale
    //   weatherParams: aerosol, humidity, precipitation, additional fog extinction
    //   renderParams: history weight, caller history validity, shadow strength,
    //                 submerged-camera flag
    // volumeDimensions.w carries the deterministic frame index.
    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> depthResolve, id<MTLTexture> nearShadowDepth,
                id<MTLTexture> farShadowDepth, id<MTLTexture> horizonShadowDepth,
                id<MTLTexture> atmosphereSkyView, id<MTLTexture> cloudShadowTransmittance,
                id<MTLTexture> cloudHitDepth, id<MTLTexture> weatherCloud,
                id<MTLTexture> weatherLayer, id<MTLSamplerState> shadowSampler,
                const FroxelUniforms& uniforms, const ShadowUniforms& shadowUniforms,
                const CloudShadowUniforms& cloudShadowUniforms, bool enableFroxels,
                GpuFrameTimer* timer = nullptr);

    bool historyValid() const { return _historyValid; }
    uint32_t halfWidth() const { return _halfWidth; }
    uint32_t halfHeight() const { return _halfHeight; }
    uint64_t persistentBytes() const { return _persistentBytes; }

private:
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _injectPipeline;
    id<MTLComputePipelineState> _integratePipeline;
    id<MTLRenderPipelineState> _resolvePipeline;
    id<MTLRenderPipelineState> _reprojectPipeline;
    id<MTLRenderPipelineState> _compositePipeline;
    id<MTLRenderPipelineState> _aerialPerspectivePipeline;
    id<MTLTexture> _froxelTexture{};
    id<MTLTexture> _integratedFroxelTexture{};
    id<MTLTexture> _integratedCurrent{};
    id<MTLTexture> _currentDepth{};
    id<MTLTexture> _history[2]{};
    id<MTLTexture> _historyDepth[2]{};
    id<MTLTexture> _neutralAtmosphere{};
    id<MTLTexture> _neutralCloudShadow{};
    id<MTLTexture> _neutralHitDepth{};
    id<MTLTexture> _neutralWeather{};
    uint32_t _halfWidth = 1;
    uint32_t _halfHeight = 1;
    uint32_t _historyIndex = 0;
    uint64_t _persistentBytes = 0;
    bool _historyValid = false;

    void allocateTargets();
    static void dispatchFroxels(id<MTLComputeCommandEncoder> encoder,
                                id<MTLComputePipelineState> pipeline);
    static void dispatchFroxelColumns(id<MTLComputeCommandEncoder> encoder,
                                      id<MTLComputePipelineState> pipeline);
};
