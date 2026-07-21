#import "render/volumetrics.hpp"

#include "common/error.hpp"
#include "render/gpu_timer.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>

namespace {

id<MTLTexture> makeTexture2D(id<MTLDevice> device, MTLPixelFormat format, NSUInteger width,
                             NSUInteger height, NSString* label) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:width
                                                                        height:height
                                                                     mipmapped:false];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage =
        MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    texture.label = label;
    if (!texture) {
        RY_LOG_FATAL("Failed to allocate volumetric render target");
    }
    return texture;
}

id<MTLRenderPipelineState> makeRenderPipeline(id<MTLDevice> device, id<MTLFunction> vertexFunction,
                                              id<MTLFunction> fragmentFunction,
                                              MTLPixelFormat color0,
                                              MTLPixelFormat color1 = MTLPixelFormatInvalid) {
    auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = color0;
    if (color1 != MTLPixelFormatInvalid) {
        descriptor.colorAttachments[1].pixelFormat = color1;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                                 error:&error];
    resetMetalObject(descriptor);
    if (!pipeline) {
        RY_LOG_FATAL("Failed to create volumetric render pipeline state");
    }
    return pipeline;
}

void bindTarget(MTLRenderPassColorAttachmentDescriptor* attachment, id<MTLTexture> texture,
                MTLLoadAction loadAction) {
    attachment.texture = texture;
    attachment.loadAction = loadAction;
    attachment.storeAction = MTLStoreActionStore;
}

void releaseTexture(id<MTLTexture> __strong& texture) {
    resetMetalObject(texture);
}

} // namespace

Volumetrics::Volumetrics(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                         uint32_t height)
    : _device(device), _halfWidth(std::max(width / 2, 1U)), _halfHeight(std::max(height / 2, 1U)) {
    id<MTLFunction> inject = [shaderLibrary newFunctionWithName:@"froxelInjectKernel"];
    id<MTLFunction> integrate = [shaderLibrary newFunctionWithName:@"froxelIntegrateKernel"];
    id<MTLFunction> vertex = [shaderLibrary newFunctionWithName:@"froxelFullscreenVertex"];
    id<MTLFunction> resolve = [shaderLibrary newFunctionWithName:@"froxelResolveFragment"];
    id<MTLFunction> reproject = [shaderLibrary newFunctionWithName:@"froxelReprojectFragment"];
    id<MTLFunction> composite = [shaderLibrary newFunctionWithName:@"froxelCompositeFragment"];
    id<MTLFunction> aerial = [shaderLibrary newFunctionWithName:@"aerialPerspectiveFragment"];
    if (!inject || !integrate || !vertex || !resolve || !reproject || !composite || !aerial) {
        RY_LOG_FATAL("Failed to load froxel shader functions");
    }

    NSError* error = nil;
    _injectPipeline = [_device newComputePipelineStateWithFunction:inject error:&error];
    if (!_injectPipeline) {
        RY_LOG_FATAL("Failed to create froxel injection pipeline state");
    }
    _integratePipeline = [_device newComputePipelineStateWithFunction:integrate error:&error];
    if (!_integratePipeline) {
        RY_LOG_FATAL("Failed to create froxel integration pipeline state");
    }
    _resolvePipeline = makeRenderPipeline(_device, vertex, resolve, MTLPixelFormatRGBA16Float,
                                          MTLPixelFormatR32Float);
    _reprojectPipeline = makeRenderPipeline(_device, vertex, reproject, MTLPixelFormatRGBA16Float,
                                            MTLPixelFormatR32Float);

    auto makeComposite = [&](id<MTLFunction> fragment) {
        auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
        descriptor.vertexFunction = vertex;
        descriptor.fragmentFunction = fragment;
        auto color = descriptor.colorAttachments[0];
        color.pixelFormat = PixelFormats::SCENE_HDR;
        color.blendingEnabled = true;
        color.rgbBlendOperation = MTLBlendOperationAdd;
        color.alphaBlendOperation = MTLBlendOperationAdd;
        color.sourceRGBBlendFactor = MTLBlendFactorOne;
        color.destinationRGBBlendFactor = MTLBlendFactorSourceAlpha;
        color.sourceAlphaBlendFactor = MTLBlendFactorZero;
        color.destinationAlphaBlendFactor = MTLBlendFactorOne;
        NSError* pipelineError = nil;
        id<MTLRenderPipelineState> result =
            [_device newRenderPipelineStateWithDescriptor:descriptor error:&pipelineError];
        resetMetalObject(descriptor);
        if (!result) {
            RY_LOG_FATAL("Failed to create volumetric composite pipeline state");
        }
        return result;
    };
    _compositePipeline = makeComposite(composite);
    _aerialPerspectivePipeline = makeComposite(aerial);
    resetMetalObject(inject);
    resetMetalObject(integrate);
    resetMetalObject(vertex);
    resetMetalObject(resolve);
    resetMetalObject(reproject);
    resetMetalObject(composite);
    resetMetalObject(aerial);

    auto froxelDescriptor = [[MTLTextureDescriptor alloc] init];
    froxelDescriptor.textureType = MTLTextureType3D;
    froxelDescriptor.pixelFormat = MTLPixelFormatRGBA16Float;
    froxelDescriptor.width = GRID_WIDTH;
    froxelDescriptor.height = GRID_HEIGHT;
    froxelDescriptor.depth = GRID_DEPTH;
    froxelDescriptor.mipmapLevelCount = 1;
    froxelDescriptor.storageMode = MTLStorageModePrivate;
    froxelDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _froxelTexture = [_device newTextureWithDescriptor:froxelDescriptor];
    _froxelTexture.label = @"Air Froxel Volume";
    _integratedFroxelTexture = [_device newTextureWithDescriptor:froxelDescriptor];
    _integratedFroxelTexture.label = @"Integrated Air Froxel Volume";
    if (!_froxelTexture || !_integratedFroxelTexture) {
        RY_LOG_FATAL("Failed to allocate air froxel volume");
    }
    resetMetalObject(froxelDescriptor);

    auto neutralDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    neutralDescriptor.storageMode = MTLStorageModeShared;
    neutralDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralCloudShadow = [_device newTextureWithDescriptor:neutralDescriptor];
    _neutralCloudShadow.label = @"Neutral Cloud Transmittance";
    const uint8_t neutral = 255;
    [_neutralCloudShadow replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                           mipmapLevel:0
                             withBytes:&neutral
                           bytesPerRow:sizeof(neutral)];

    auto hitDepthDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    hitDepthDescriptor.storageMode = MTLStorageModeShared;
    hitDepthDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralHitDepth = [_device newTextureWithDescriptor:hitDepthDescriptor];
    _neutralHitDepth.label = @"Neutral Cloud Hit Depth";
    const uint16_t noHit = 0;
    [_neutralHitDepth replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                        mipmapLevel:0
                          withBytes:&noHit
                        bytesPerRow:sizeof(noHit)];

    auto atmosphereDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    atmosphereDescriptor.storageMode = MTLStorageModeShared;
    atmosphereDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralAtmosphere = [_device newTextureWithDescriptor:atmosphereDescriptor];
    _neutralAtmosphere.label = @"Neutral Atmosphere Irradiance";
    const uint16_t neutralAtmosphere[] = {0x3400, 0x3400, 0x3400, 0x3C00};
    [_neutralAtmosphere replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                          mipmapLevel:0
                            withBytes:neutralAtmosphere
                          bytesPerRow:sizeof(neutralAtmosphere)];

    auto weatherDescriptor = [[MTLTextureDescriptor alloc] init];
    weatherDescriptor.textureType = MTLTextureType2DArray;
    weatherDescriptor.pixelFormat = MTLPixelFormatRGBA16Float;
    weatherDescriptor.width = 1;
    weatherDescriptor.height = 1;
    weatherDescriptor.arrayLength = 2;
    weatherDescriptor.storageMode = MTLStorageModeShared;
    weatherDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralWeather = [_device newTextureWithDescriptor:weatherDescriptor];
    _neutralWeather.label = @"Neutral Regional Weather";
    const uint16_t neutralWeather[] = {0, 0, 0, 0};
    for (NSUInteger slice = 0; slice < 2; ++slice) {
        [_neutralWeather replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                           mipmapLevel:0
                                 slice:slice
                             withBytes:neutralWeather
                           bytesPerRow:sizeof(neutralWeather)
                         bytesPerImage:sizeof(neutralWeather)];
    }
    resetMetalObject(weatherDescriptor);

    allocateTargets();
}

Volumetrics::~Volumetrics() {
    releaseTexture(_froxelTexture);
    releaseTexture(_integratedFroxelTexture);
    releaseTexture(_integratedCurrent);
    releaseTexture(_currentDepth);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_history[index]);
        releaseTexture(_historyDepth[index]);
    }
    releaseTexture(_neutralAtmosphere);
    releaseTexture(_neutralCloudShadow);
    releaseTexture(_neutralHitDepth);
    releaseTexture(_neutralWeather);
    resetMetalObject(_injectPipeline);
    resetMetalObject(_integratePipeline);
    resetMetalObject(_resolvePipeline);
    resetMetalObject(_reprojectPipeline);
    resetMetalObject(_compositePipeline);
    resetMetalObject(_aerialPerspectivePipeline);
}

void Volumetrics::allocateTargets() {
    releaseTexture(_integratedCurrent);
    releaseTexture(_currentDepth);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_history[index]);
        releaseTexture(_historyDepth[index]);
    }
    _integratedCurrent = makeTexture2D(_device, MTLPixelFormatRGBA16Float, _halfWidth, _halfHeight,
                                       @"Current Froxel Integration");
    _currentDepth = makeTexture2D(_device, MTLPixelFormatR32Float, _halfWidth, _halfHeight,
                                  @"Current Froxel View Depth");
    for (uint32_t index = 0; index < 2; ++index) {
        _history[index] = makeTexture2D(_device, MTLPixelFormatRGBA16Float, _halfWidth, _halfHeight,
                                        index == 0 ? @"Froxel History A" : @"Froxel History B");
        _historyDepth[index] = makeTexture2D(
            _device, MTLPixelFormatR32Float, _halfWidth, _halfHeight,
            index == 0 ? @"Froxel History View Depth A" : @"Froxel History View Depth B");
    }
    _persistentBytes = volumetricMemoryBytes(_halfWidth * 2U, _halfHeight * 2U);
    _historyIndex = 0;
    _historyValid = false;
}

void Volumetrics::resize(uint32_t width, uint32_t height) {
    const uint32_t halfWidth = std::max(width / 2, 1U);
    const uint32_t halfHeight = std::max(height / 2, 1U);
    if (halfWidth == _halfWidth && halfHeight == _halfHeight) {
        return;
    }
    _halfWidth = halfWidth;
    _halfHeight = halfHeight;
    allocateTargets();
}

void Volumetrics::resetHistory() {
    _historyValid = false;
}

void Volumetrics::dispatchFroxels(id<MTLComputeCommandEncoder> encoder,
                                  id<MTLComputePipelineState> pipeline) {
    const NSUInteger threadWidth = std::min<NSUInteger>(8, pipeline.threadExecutionWidth);
    const NSUInteger threadHeight = 4;
    const NSUInteger availableDepth = pipeline.maxTotalThreadsPerThreadgroup /
                                      std::max<NSUInteger>(threadWidth * threadHeight, 1);
    const NSUInteger threadDepth = std::max<NSUInteger>(1, std::min<NSUInteger>(4, availableDepth));
    [encoder dispatchThreads:MTLSizeMake(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, threadDepth)];
}

void Volumetrics::dispatchFroxelColumns(id<MTLComputeCommandEncoder> encoder,
                                        id<MTLComputePipelineState> pipeline) {
    const NSUInteger threadWidth = std::min<NSUInteger>(pipeline.threadExecutionWidth, 16);
    const NSUInteger threadHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(
               pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1), 16));
    [encoder dispatchThreads:MTLSizeMake(GRID_WIDTH, GRID_HEIGHT, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
}

void Volumetrics::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                         id<MTLTexture> depthResolve, id<MTLTexture> nearShadowDepth,
                         id<MTLTexture> farShadowDepth, id<MTLTexture> horizonShadowDepth,
                         id<MTLTexture> atmosphereSkyView, id<MTLTexture> cloudShadowTransmittance,
                         id<MTLTexture> cloudHitDepth, id<MTLTexture> weatherCloud,
                         id<MTLTexture> weatherLayer, id<MTLSamplerState> shadowSampler,
                         const FroxelUniforms& uniforms, const ShadowUniforms& shadowUniforms,
                         const CloudShadowUniforms& cloudShadowUniforms, bool enableFroxels,
                         GpuFrameTimer* timer) {
    if (!commandBuffer || !sceneHDR || !depthResolve) {
        return;
    }

    FroxelUniforms frameUniforms = uniforms;
    frameUniforms.volumeDimensions.x = GRID_WIDTH;
    frameUniforms.volumeDimensions.y = GRID_HEIGHT;
    frameUniforms.volumeDimensions.z = GRID_DEPTH;
    const bool callerHistoryValid = frameUniforms.renderParams.y > 0.5F;
    frameUniforms.renderParams.y = (_historyValid && callerHistoryValid) ? 1.0F : 0.0F;

    const bool shadowsReady =
        nearShadowDepth && farShadowDepth && horizonShadowDepth && shadowSampler;
    enableFroxels = enableFroxels && shadowsReady && frameUniforms.renderParams.w < 0.5F;
    if (!enableFroxels) {
        _historyValid = false;
        auto pass = [[MTLRenderPassDescriptor alloc] init];
        bindTarget(pass.colorAttachments[0], sceneHDR, MTLLoadActionLoad);
        if (timer) {
            timer->attachPass(pass, "aerialPerspective");
        }
        id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:pass];
        if (!encoder) {
            resetMetalObject(pass);
            return;
        }
        encoder.label = @"Atmosphere Aerial Perspective";
        [encoder setRenderPipelineState:_aerialPerspectivePipeline];
        [encoder setFragmentTexture:depthResolve atIndex:0];
        [encoder
            setFragmentTexture:atmosphereSkyView != nil ? atmosphereSkyView : _neutralAtmosphere
                       atIndex:1];
        [encoder setFragmentTexture:cloudHitDepth != nil ? cloudHitDepth : _neutralHitDepth
                            atIndex:2];
        [encoder setFragmentBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        resetMetalObject(pass);
        return;
    }

    id<MTLComputeCommandEncoder> injection = [commandBuffer computeCommandEncoder];
    if (!injection) {
        return;
    }
    injection.label = @"Froxel Weather and Light Injection";
    const uint32_t computeTimerToken =
        timer ? timer->beginComputePass(injection, "froxelInjectIntegrate") : UINT32_MAX;
    [injection setComputePipelineState:_injectPipeline];
    [injection setTexture:_froxelTexture atIndex:0];
    [injection setTexture:nearShadowDepth atIndex:1];
    [injection setTexture:farShadowDepth atIndex:2];
    [injection setTexture:horizonShadowDepth atIndex:3];
    [injection
        setTexture:cloudShadowTransmittance != nil ? cloudShadowTransmittance : _neutralCloudShadow
           atIndex:4];
    [injection setTexture:atmosphereSkyView != nil ? atmosphereSkyView : _neutralAtmosphere
                  atIndex:5];
    [injection setTexture:weatherCloud != nil ? weatherCloud : _neutralWeather atIndex:6];
    [injection setTexture:weatherLayer != nil ? weatherLayer : _neutralWeather atIndex:7];
    [injection setSamplerState:shadowSampler atIndex:1];
    [injection setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    [injection setBytes:&shadowUniforms length:sizeof(shadowUniforms) atIndex:1];
    [injection setBytes:&cloudShadowUniforms length:sizeof(cloudShadowUniforms) atIndex:2];
    dispatchFroxels(injection, _injectPipeline);
    [injection memoryBarrierWithScope:MTLBarrierScopeTextures];
    [injection setComputePipelineState:_integratePipeline];
    [injection setTexture:_froxelTexture atIndex:0];
    [injection setTexture:_integratedFroxelTexture atIndex:1];
    [injection setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    dispatchFroxelColumns(injection, _integratePipeline);
    if (timer) {
        timer->endComputePass(injection, computeTimerToken);
    }
    [injection endEncoding];

    auto resolvePass = [[MTLRenderPassDescriptor alloc] init];
    bindTarget(resolvePass.colorAttachments[0], _integratedCurrent, MTLLoadActionDontCare);
    bindTarget(resolvePass.colorAttachments[1], _currentDepth, MTLLoadActionDontCare);
    if (timer) {
        timer->attachPass(resolvePass, "froxelResolve");
    }
    id<MTLRenderCommandEncoder> resolveEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:resolvePass];
    if (!resolveEncoder) {
        resetMetalObject(resolvePass);
        return;
    }
    resolveEncoder.label = @"Resolve Integrated Froxel Volume";
    [resolveEncoder setRenderPipelineState:_resolvePipeline];
    [resolveEncoder setFragmentTexture:depthResolve atIndex:0];
    [resolveEncoder setFragmentTexture:_integratedFroxelTexture atIndex:1];
    [resolveEncoder setFragmentTexture:cloudHitDepth != nil ? cloudHitDepth : _neutralHitDepth
                               atIndex:2];
    [resolveEncoder setFragmentBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    [resolveEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [resolveEncoder endEncoding];
    resetMetalObject(resolvePass);

    const uint32_t writeIndex = _historyIndex ^ 1U;
    auto reprojectPass = [[MTLRenderPassDescriptor alloc] init];
    bindTarget(reprojectPass.colorAttachments[0], _history[writeIndex], MTLLoadActionDontCare);
    bindTarget(reprojectPass.colorAttachments[1], _historyDepth[writeIndex], MTLLoadActionDontCare);
    if (timer) {
        timer->attachPass(reprojectPass, "froxelTemporal");
    }
    id<MTLRenderCommandEncoder> reprojectEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:reprojectPass];
    if (!reprojectEncoder) {
        resetMetalObject(reprojectPass);
        return;
    }
    reprojectEncoder.label = @"Temporal Froxel Reprojection";
    [reprojectEncoder setRenderPipelineState:_reprojectPipeline];
    [reprojectEncoder setFragmentTexture:_integratedCurrent atIndex:0];
    [reprojectEncoder setFragmentTexture:_currentDepth atIndex:1];
    [reprojectEncoder setFragmentTexture:_history[_historyIndex] atIndex:2];
    [reprojectEncoder setFragmentTexture:_historyDepth[_historyIndex] atIndex:3];
    [reprojectEncoder setFragmentTexture:depthResolve atIndex:4];
    [reprojectEncoder setFragmentTexture:cloudHitDepth != nil ? cloudHitDepth : _neutralHitDepth
                                 atIndex:5];
    [reprojectEncoder setFragmentBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    [reprojectEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [reprojectEncoder endEncoding];
    resetMetalObject(reprojectPass);

    auto compositePass = [[MTLRenderPassDescriptor alloc] init];
    bindTarget(compositePass.colorAttachments[0], sceneHDR, MTLLoadActionLoad);
    if (timer) {
        timer->attachPass(compositePass, "froxelComposite");
    }
    id<MTLRenderCommandEncoder> compositeEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:compositePass];
    if (!compositeEncoder) {
        resetMetalObject(compositePass);
        return;
    }
    compositeEncoder.label = @"Composite Froxel Scattering";
    [compositeEncoder setRenderPipelineState:_compositePipeline];
    [compositeEncoder setFragmentTexture:_history[writeIndex] atIndex:0];
    [compositeEncoder setFragmentTexture:_historyDepth[writeIndex] atIndex:1];
    [compositeEncoder setFragmentTexture:depthResolve atIndex:2];
    [compositeEncoder setFragmentTexture:cloudHitDepth != nil ? cloudHitDepth : _neutralHitDepth
                                 atIndex:3];
    [compositeEncoder setFragmentBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    [compositeEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [compositeEncoder endEncoding];
    resetMetalObject(compositePass);

    _historyIndex = writeIndex;
    _historyValid = true;
}
