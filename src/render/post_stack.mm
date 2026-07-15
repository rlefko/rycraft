#import "render/post_stack.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"
#include "render/shader_types.hpp"

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
PostStack::PostStack(id<MTLDevice> device, id<MTLLibrary> shaderLibrary) : _device(device) {
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"postCompositeVertex"];
    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"postCompositeFragment"];
    if (!vertexFunc || !fragmentFunc) {
        RY_LOG_FATAL("Failed to load post-composite shader functions");
    }

    auto desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vertexFunc;
    desc.fragmentFunction = fragmentFunc;
    desc.colorAttachments[0].pixelFormat = PixelFormats::DRAWABLE;

    NSError* error = nil;
    _compositePipelineState = [_device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!_compositePipelineState) {
        RY_LOG_FATAL("Failed to create post-composite pipeline state");
    }

    // 4×4 black fallback: bound as the bloom input when bloom is disabled.
    // The composite forces bloomIntensity to 0 in that case, so the sampled
    // value is multiplied out — but a Private RG11B10 render target is
    // cleared to black once below so even a stray NaN can't survive x*0.
    auto texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::BLOOM
                                                                      width:4
                                                                     height:4
                                                                  mipmapped:false];
    texDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    texDesc.storageMode = MTLStorageModePrivate;
    _blackFallback = [_device newTextureWithDescriptor:texDesc];
    if (!_blackFallback) {
        RY_LOG_FATAL("Failed to allocate post-stack black fallback texture");
    }
    id<MTLCommandQueue> queue = [_device newCommandQueue];
    id<MTLCommandBuffer> clearCmd = [queue commandBuffer];
    auto clearPass = [[MTLRenderPassDescriptor alloc] init];
    clearPass.colorAttachments[0].texture = _blackFallback;
    clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    clearPass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    [[clearCmd renderCommandEncoderWithDescriptor:clearPass] endEncoding];
    [clearCmd commit];

    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    _linearSampler = [_device newSamplerStateWithDescriptor:samplerDesc];
    if (!_linearSampler) {
        RY_LOG_FATAL("Failed to create post-stack sampler");
    }

    // ---- Auto-exposure compute pipeline + persistent state ----
    id<MTLFunction> exposureFunc = [shaderLibrary newFunctionWithName:@"exposureReduce"];
    if (!exposureFunc) {
        RY_LOG_FATAL("Failed to load exposureReduce compute function");
    }
    _exposurePipelineState = [_device newComputePipelineStateWithFunction:exposureFunc
                                                                    error:&error];
    if (!_exposurePipelineState) {
        RY_LOG_FATAL("Failed to create exposure compute pipeline state");
    }
    // Seed the persistent state so the very first frames aren't black while
    // the EMA converges (mid-grey log-luminance, exposure 1).
    ExposureState seed{};
    seed.smoothedLogLum = 0.0f;
    seed.exposure = 1.0f;
    _exposureBuffer = [_device newBufferWithBytes:&seed
                                           length:sizeof(ExposureState)
                                          options:MTLResourceStorageModeShared];
    if (!_exposureBuffer) {
        RY_LOG_FATAL("Failed to allocate exposure state buffer");
    }

    // ---- Lens-flare occlusion probe + persistent visibility ----
    id<MTLFunction> flareFunc = [shaderLibrary newFunctionWithName:@"flareProbe"];
    if (!flareFunc) {
        RY_LOG_FATAL("Failed to load flareProbe compute function");
    }
    _flarePipelineState = [_device newComputePipelineStateWithFunction:flareFunc error:&error];
    if (!_flarePipelineState) {
        RY_LOG_FATAL("Failed to create flare probe pipeline state");
    }
    FlareState flareSeed{};
    _flareBuffer = [_device newBufferWithBytes:&flareSeed
                                        length:sizeof(FlareState)
                                       options:MTLResourceStorageModeShared];
    if (!_flareBuffer) {
        RY_LOG_FATAL("Failed to allocate flare state buffer");
    }
}

// ---------------------------------------------------------------------------
// encodeFlareProbe — one compute thread taps the depth around the sun and
// eases the persistent visibility (see post.metal).
// ---------------------------------------------------------------------------
void PostStack::encodeFlareProbe(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneDepth,
                                 simd_float2 sunScreenUV) {
    if (!commandBuffer || !sceneDepth)
        return;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder)
        return;
    PostUniforms uniforms{};
    uniforms.sunScreenUV = sunScreenUV;
    [encoder setComputePipelineState:_flarePipelineState];
    [encoder setTexture:sceneDepth atIndex:0];
    [encoder setBuffer:_flareBuffer offset:0 atIndex:0];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
}

// ---------------------------------------------------------------------------
// encodeExposure — one threadgroup reduces average scene luminance and eases
// the persistent exposure toward it.
// ---------------------------------------------------------------------------
void PostStack::encodeExposure(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR) {
    if (!commandBuffer || !sceneHDR)
        return;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder)
        return;

    ExposureParams params{};
    // keyValue ≈ a lit surface's average luminance, so daylight maps near
    // exposure 1.0 (a middle-grey 0.18 target over-darkened bright scenes).
    params.keyValue = 0.5f;
    params.adaptationDownRate = 0.10f; // ~0.15 s to stop down at 60 fps
    params.adaptationUpRate = 0.03f;   // ~0.5 s to recover when it darkens
    params.minLogLum = -8.0f;
    params.maxLogLum = 4.0f;
    params.sampleGrid = simd_make_uint2(16, 16); // 256 samples = 1 threadgroup
    // Floor 0.35 lets sun-facing frames actually stop down (0.7 pinned the
    // exposure before it could react); ceiling 4 still lifts caves and night
    // — with no highlights the weighted mean matches the plain mean, so dark
    // scenes are not crushed.
    params.minExposure = 0.35f;
    params.maxExposure = 4.0f;
    params.highlightGain = 2.0f;
    params.highlightKnee = 1.0f;  // log2: above ~2x a typical lit surface
    params.highlightRange = 3.0f; // full weight ~8 log2 luminance and up

    [encoder setComputePipelineState:_exposurePipelineState];
    [encoder setTexture:sceneHDR atIndex:0];
    [encoder setBuffer:_exposureBuffer offset:0 atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
}

// ---------------------------------------------------------------------------
// encodeComposite
// ---------------------------------------------------------------------------
void PostStack::encodeComposite(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                                id<MTLTexture> bloomTexture, id<MTLTexture> outputTexture,
                                const GraphicsSettings& gfx, uint32_t frameIndex,
                                float flareStrength, simd_float2 sunScreenUV) {
    if (!commandBuffer || !sceneHDR || !outputTexture)
        return;

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = outputTexture;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder)
        return;

    const bool bloomOn = bloomTexture != nil;
    PostUniforms uniforms{};
    uniforms.resolution = simd_make_float2(static_cast<float>(outputTexture.width),
                                           static_cast<float>(outputTexture.height));
    // Exposure is applied in-shader from the persistent state buffer; the
    // uniform field stays 1 so the two never double-apply.
    uniforms.exposure = 1.0f;
    uniforms.bloomIntensity = bloomOn ? gfx.bloomIntensity() : 0.0f;
    // vibrance 0-10 → 0..2 grade multiplier; 5 = 1.0 (stock look)
    uniforms.vibrance = static_cast<float>(gfx.vibrance) * 0.2f;
    uniforms.sharpening = static_cast<float>(gfx.sharpening) * 0.1f;
    uniforms.frameIndex = frameIndex;
    uniforms.flareStrength = flareStrength;
    uniforms.sunScreenUV = sunScreenUV;

    [encoder setRenderPipelineState:_compositePipelineState];
    [encoder setFragmentTexture:sceneHDR atIndex:0];
    [encoder setFragmentTexture:bloomOn ? bloomTexture : _blackFallback atIndex:1];
    [encoder setFragmentSamplerState:_linearSampler atIndex:0];
    [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [encoder setFragmentBuffer:_exposureBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_flareBuffer offset:0 atIndex:2];

    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
}
