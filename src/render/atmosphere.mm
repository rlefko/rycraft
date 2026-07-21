#import "render/atmosphere.hpp"

#include "common/error.hpp"
#include "render/gpu_timer.hpp"
#include "render/metal_ownership.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

id<MTLComputePipelineState> makeComputePipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
                                                NSString* functionName) {
    id<MTLFunction> function = [shaderLibrary newFunctionWithName:functionName];
    if (!function) {
        RY_LOG_FATAL("Failed to load atmosphere compute function");
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    resetMetalObject(function);
    if (!pipeline) {
        RY_LOG_FATAL("Failed to create atmosphere compute pipeline");
    }
    return pipeline;
}

id<MTLTexture> makeLut(id<MTLDevice> device, NSUInteger width, NSUInteger height, NSString* label) {
    auto descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:width
                                                          height:height
                                                       mipmapped:false];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    texture.label = label;
    if (!texture) {
        RY_LOG_FATAL("Failed to allocate atmosphere LUT");
    }
    return texture;
}

bool float4Changed(simd_float4 a, simd_float4 b, float epsilon) {
    return simd_length(a - b) > epsilon;
}

} // namespace

AtmosphereUniforms earthAtmosphereUniforms(float cameraWorldY, simd_float3 sunDirection,
                                           simd_float3 sunRadiance, float aerosolDensity,
                                           float humidity, uint32_t frameIndex) {
    AtmosphereUniforms result{};
    constexpr float GROUND_RADIUS_KM = 6360.0F;
    constexpr float TOP_RADIUS_KM = 6460.0F;
    constexpr float BLOCK_TO_KM = 0.001F;
    const float altitudeKm = std::max(cameraWorldY * BLOCK_TO_KM, 0.0F);
    result.cameraPositionKm = simd_make_float3(0.0F, GROUND_RADIUS_KM + altitudeKm, 0.0F);
    result.sunDirection = simd_normalize(sunDirection);
    result.sunRadiance = sunRadiance;
    result.groundAlbedo = simd_make_float3(0.18F, 0.20F, 0.16F);
    result.rayleighScatteringAndScaleHeight =
        simd_make_float4(0.005802F, 0.013558F, 0.033100F, 8.0F);
    const float mieMultiplier = std::clamp(aerosolDensity / 0.08F, 0.35F, 5.0F);
    const float mie = 0.003996F * mieMultiplier;
    result.mieScatteringAndScaleHeight = simd_make_float4(mie, mie, mie, 1.2F);
    result.ozoneAbsorptionAndCenter = simd_make_float4(0.000650F, 0.001881F, 0.000085F, 25.0F);
    result.atmosphereRadii =
        simd_make_float4(GROUND_RADIUS_KM, TOP_RADIUS_KM, 0.004675F, BLOCK_TO_KM);
    result.weatherOptics = simd_make_float4(std::clamp(aerosolDensity, 0.0F, 4.0F),
                                            std::clamp(humidity, 0.0F, 1.0F), 0.0F, 0.8F);
    result.renderParams = simd_make_float4(static_cast<float>(frameIndex), 1.0F, 0.0F, 0.0F);
    return result;
}

bool atmosphereUniformsFinite(const AtmosphereUniforms& uniforms) {
    const float* values = reinterpret_cast<const float*>(&uniforms);
    for (size_t i = 0; i < sizeof(uniforms) / sizeof(float); ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return uniforms.atmosphereRadii.x > 0.0F &&
           uniforms.atmosphereRadii.y > uniforms.atmosphereRadii.x;
}

AtmosphereRenderer::AtmosphereRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary) {
    _transmittancePipeline =
        makeComputePipeline(device, shaderLibrary, @"atmosphereTransmittanceKernel");
    _multipleScatteringPipeline =
        makeComputePipeline(device, shaderLibrary, @"atmosphereMultipleScatteringKernel");
    _skyViewPipeline = makeComputePipeline(device, shaderLibrary, @"atmosphereSkyViewKernel");
    _transmittance = makeLut(device, ATMOSPHERE_TRANSMITTANCE_WIDTH,
                             ATMOSPHERE_TRANSMITTANCE_HEIGHT, @"Atmosphere Transmittance");
    _multipleScattering =
        makeLut(device, ATMOSPHERE_MULTISCATTER_WIDTH, ATMOSPHERE_MULTISCATTER_HEIGHT,
                @"Atmosphere Multiple Scattering");
    _skyView = makeLut(device, ATMOSPHERE_SKY_VIEW_WIDTH, ATMOSPHERE_SKY_VIEW_HEIGHT,
                       @"Atmosphere Sky View");
}

AtmosphereRenderer::~AtmosphereRenderer() {
    resetMetalObject(_transmittancePipeline);
    resetMetalObject(_multipleScatteringPipeline);
    resetMetalObject(_skyViewPipeline);
    resetMetalObject(_transmittance);
    resetMetalObject(_multipleScattering);
    resetMetalObject(_skyView);
}

void AtmosphereRenderer::dispatch2D(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLComputePipelineState> pipeline, NSUInteger width,
                                    NSUInteger height) {
    const NSUInteger threadWidth = std::min<NSUInteger>(pipeline.threadExecutionWidth, 16);
    const NSUInteger threadHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(
               pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1), 16));
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
}

void AtmosphereRenderer::encode(id<MTLCommandBuffer> commandBuffer,
                                const AtmosphereUniforms& uniforms, bool forceRefresh,
                                GpuFrameTimer* timer) {
    if (!commandBuffer || !atmosphereUniformsFinite(uniforms)) {
        return;
    }

    const bool slowDirty = forceRefresh || !_hasPrevious ||
                           simd_length(uniforms.sunRadiance - _previous.sunRadiance) > 1.0e-4F ||
                           float4Changed(uniforms.rayleighScatteringAndScaleHeight,
                                         _previous.rayleighScatteringAndScaleHeight, 1.0e-6F) ||
                           float4Changed(uniforms.mieScatteringAndScaleHeight,
                                         _previous.mieScatteringAndScaleHeight, 1.0e-6F) ||
                           float4Changed(uniforms.ozoneAbsorptionAndCenter,
                                         _previous.ozoneAbsorptionAndCenter, 1.0e-6F) ||
                           float4Changed(uniforms.weatherOptics, _previous.weatherOptics, 0.01F);
    const bool skyDirty =
        slowDirty || !_hasPrevious ||
        simd_length(uniforms.sunDirection - _previous.sunDirection) > 1.0e-4F ||
        std::abs(uniforms.cameraPositionKm.y - _previous.cameraPositionKm.y) > 0.025F;
    if (!slowDirty && !skyDirty) {
        _previous.renderParams.x = uniforms.renderParams.x;
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Atmosphere LUTs";
    const uint32_t timerToken =
        timer ? timer->beginComputePass(encoder, "atmosphereLuts") : UINT32_MAX;
    if (slowDirty) {
        [encoder setComputePipelineState:_transmittancePipeline];
        [encoder setTexture:_transmittance atIndex:0];
        [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        dispatch2D(encoder, _transmittancePipeline, _transmittance.width, _transmittance.height);

        [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

        [encoder setComputePipelineState:_multipleScatteringPipeline];
        [encoder setTexture:_transmittance atIndex:0];
        [encoder setTexture:_multipleScattering atIndex:1];
        [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        dispatch2D(encoder, _multipleScatteringPipeline, _multipleScattering.width,
                   _multipleScattering.height);

        [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
        ++_slowRefreshCount;
    }
    if (skyDirty) {
        [encoder setComputePipelineState:_skyViewPipeline];
        [encoder setTexture:_transmittance atIndex:0];
        [encoder setTexture:_multipleScattering atIndex:1];
        [encoder setTexture:_skyView atIndex:2];
        [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        dispatch2D(encoder, _skyViewPipeline, _skyView.width, _skyView.height);
        ++_skyRefreshCount;
    }
    if (timer) {
        timer->endComputePass(encoder, timerToken);
    }
    [encoder endEncoding];

    _previous = uniforms;
    _hasPrevious = true;
}
