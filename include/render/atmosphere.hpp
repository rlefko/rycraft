#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"

#include <cstdint>

class GpuFrameTimer;

// Returns the Earth-like optical model used by captures and normal gameplay.
// World Y is converted to kilometers only for atmospheric altitude response;
// large world X/Z coordinates never enter the spherical LUT integration.
AtmosphereUniforms earthAtmosphereUniforms(float cameraWorldY, simd_float3 sunDirection,
                                           simd_float3 sunRadiance, float aerosolDensity,
                                           float humidity, uint32_t frameIndex);

bool atmosphereUniformsFinite(const AtmosphereUniforms& uniforms);

constexpr uint64_t atmosphereLutMemoryBytes() noexcept {
    constexpr uint64_t BYTES_PER_TEXEL = 8U;
    return (static_cast<uint64_t>(ATMOSPHERE_TRANSMITTANCE_WIDTH) *
                ATMOSPHERE_TRANSMITTANCE_HEIGHT +
            static_cast<uint64_t>(ATMOSPHERE_MULTISCATTER_WIDTH) * ATMOSPHERE_MULTISCATTER_HEIGHT +
            static_cast<uint64_t>(ATMOSPHERE_SKY_VIEW_WIDTH) * ATMOSPHERE_SKY_VIEW_HEIGHT) *
           BYTES_PER_TEXEL;
}

class AtmosphereRenderer {
public:
    AtmosphereRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~AtmosphereRenderer();

    // Refreshes transmittance and multiple scattering only when slow optical
    // parameters change. Sky view follows the light direction and camera
    // altitude, so it may update independently each frame.
    void encode(id<MTLCommandBuffer> commandBuffer, const AtmosphereUniforms& uniforms,
                bool forceRefresh = false, GpuFrameTimer* timer = nullptr);

    id<MTLTexture> transmittanceTexture() const { return _transmittance; }
    id<MTLTexture> multipleScatteringTexture() const { return _multipleScattering; }
    id<MTLTexture> skyViewTexture() const { return _skyView; }
    uint64_t slowRefreshCount() const { return _slowRefreshCount; }
    uint64_t skyRefreshCount() const { return _skyRefreshCount; }

private:
    id<MTLComputePipelineState> _transmittancePipeline;
    id<MTLComputePipelineState> _multipleScatteringPipeline;
    id<MTLComputePipelineState> _skyViewPipeline;
    id<MTLTexture> _transmittance;
    id<MTLTexture> _multipleScattering;
    id<MTLTexture> _skyView;
    AtmosphereUniforms _previous{};
    bool _hasPrevious = false;
    uint64_t _slowRefreshCount = 0;
    uint64_t _skyRefreshCount = 0;

    static void dispatch2D(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline, NSUInteger width,
                           NSUInteger height);
};
