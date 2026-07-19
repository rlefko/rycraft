#import "render/screen_space_lighting.hpp"

#include "common/error.hpp"
#include "render/gpu_timer.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>
#include <cmath>

namespace {

id<MTLComputePipelineState> makeComputePipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
                                                NSString* functionName) {
    id<MTLFunction> function = [shaderLibrary newFunctionWithName:functionName];
    if (!function) {
        RY_LOG_FATAL("Failed to load screen-space lighting compute function");
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    resetMetalObject(function);
    if (!pipeline) {
        RY_LOG_FATAL("Failed to create screen-space lighting compute pipeline");
    }
    return pipeline;
}

NSUInteger mipCount(NSUInteger width, NSUInteger height) {
    NSUInteger levels = 1;
    while (width > 1 || height > 1) {
        width = std::max<NSUInteger>(width / 2, 1);
        height = std::max<NSUInteger>(height / 2, 1);
        ++levels;
    }
    return levels;
}

uint64_t mipPayloadBytes(uint32_t width, uint32_t height, uint32_t bytesPerPixel) {
    uint64_t bytes = 0;
    while (true) {
        bytes += static_cast<uint64_t>(width) * height * bytesPerPixel;
        if (width == 1U && height == 1U) {
            return bytes;
        }
        width = std::max(width / 2U, 1U);
        height = std::max(height / 2U, 1U);
    }
}

id<MTLTexture> makeTexture2D(id<MTLDevice> device, MTLPixelFormat format, NSUInteger width,
                             NSUInteger height, bool mipmapped, NSString* label) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:width
                                                                        height:height
                                                                     mipmapped:mipmapped];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage =
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    if (mipmapped) {
        descriptor.mipmapLevelCount = mipCount(width, height);
    }
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    texture.label = label;
    if (!texture) {
        RY_LOG_FATAL("Failed to allocate screen-space lighting texture");
    }
    return texture;
}

void releaseTexture(id<MTLTexture> __strong& texture) {
    resetMetalObject(texture);
}

} // namespace

ScreenSpaceLightingMemoryFootprint
screenSpaceLightingMemoryFootprint(uint32_t width, uint32_t height, int quality) noexcept {
    ScreenSpaceLightingMemoryFootprint footprint;
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    quality = std::clamp(quality, 0, 2);
    footprint.neutralBytes = 8U;
    if (quality == 0) {
        return footprint;
    }

    const uint32_t divisor = quality >= 2 ? 2U : 4U;
    footprint.workWidth = std::max(width / divisor, 1U);
    footprint.workHeight = std::max(height / divisor, 1U);
    footprint.linearDepthPyramidBytes = mipPayloadBytes(width, height, 4U);
    // RG16F octahedral normals retain a full-resolution receiver guide for
    // the final joint bilateral reconstruction without changing surface RGB
    // albedo or its baked-accessibility alpha channel.
    footprint.normalBytes = static_cast<uint64_t>(width) * height * 4U;
    const uint64_t workPixels = static_cast<uint64_t>(footprint.workWidth) * footprint.workHeight;
    footprint.traceBytes = workPixels * 8U;
    footprint.historyBytes = workPixels * 8U * 2U;
    footprint.historyDepthBytes = workPixels * 4U * 2U;
    footprint.momentsBytes = workPixels * 8U * 2U;
    footprint.scratchBytes = workPixels * 8U;
    return footprint;
}

uint32_t indirectHistoryResetMask(const IndirectHistoryState& previous,
                                  const IndirectHistoryState& current) {
    uint32_t reasons = INDIRECT_HISTORY_STABLE;
    if (previous.width != current.width || previous.height != current.height) {
        reasons |= INDIRECT_HISTORY_RESIZE;
    }
    if ((current.cameraPosition - previous.cameraPosition).length() > 8.0F) {
        reasons |= INDIRECT_HISTORY_TELEPORT;
    }
    if (previous.worldIdentity != current.worldIdentity) {
        reasons |= INDIRECT_HISTORY_WORLD_CHANGE;
    }
    if (std::abs(previous.fovDegrees - current.fovDegrees) > 0.5F) {
        reasons |= INDIRECT_HISTORY_FOV_CHANGE;
    }
    if (previous.quality != current.quality) {
        reasons |= INDIRECT_HISTORY_QUALITY_CHANGE;
    }
    if (previous.forcedStateRevision != current.forcedStateRevision) {
        reasons |= INDIRECT_HISTORY_FORCED_STATE;
    }
    if (previous.directLightSource != current.directLightSource) {
        reasons |= INDIRECT_HISTORY_LIGHT_SOURCE;
    }
    if (!current.priorDepthValid) {
        reasons |= INDIRECT_HISTORY_INVALID_DEPTH;
    }
    return reasons;
}

ScreenSpaceLighting::ScreenSpaceLighting(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
                                         uint32_t width, uint32_t height)
    : _device(device), _displayWidth(std::max(width, 1U)), _displayHeight(std::max(height, 1U)) {
    _linearDepthPipeline =
        makeComputePipeline(device, shaderLibrary, @"screenSpaceLinearDepthKernel");
    _depthReducePipeline =
        makeComputePipeline(device, shaderLibrary, @"screenSpaceDepthReduceKernel");
    _normalPipeline = makeComputePipeline(device, shaderLibrary, @"screenSpaceNormalKernel");
    _tracePipeline = makeComputePipeline(device, shaderLibrary, @"screenSpaceTraceKernel");
    _temporalPipeline = makeComputePipeline(device, shaderLibrary, @"screenSpaceTemporalKernel");
    _atrousPipeline = makeComputePipeline(device, shaderLibrary, @"screenSpaceAtrousKernel");
    _historyDepthPipeline =
        makeComputePipeline(device, shaderLibrary, @"screenSpaceHistoryDepthKernel");

    id<MTLFunction> vertex = [shaderLibrary newFunctionWithName:@"screenSpaceApplyVertex"];
    id<MTLFunction> fragment = [shaderLibrary newFunctionWithName:@"screenSpaceApplyFragment"];
    if (!vertex || !fragment) {
        RY_LOG_FATAL("Failed to load screen-space lighting apply functions");
    }
    auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = vertex;
    descriptor.fragmentFunction = fragment;
    descriptor.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    descriptor.colorAttachments[0].blendingEnabled = true;
    descriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    descriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    descriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    descriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorZero;
    descriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
    descriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
    NSError* error = nil;
    _applyPipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    resetMetalObject(descriptor);
    resetMetalObject(vertex);
    resetMetalObject(fragment);
    if (!_applyPipeline) {
        RY_LOG_FATAL("Failed to create screen-space lighting apply pipeline");
    }

    auto neutralDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    neutralDescriptor.storageMode = MTLStorageModeShared;
    neutralDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralTexture = [_device newTextureWithDescriptor:neutralDescriptor];
    const uint16_t neutral[] = {0, 0, 0, 0x3C00};
    [_neutralTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                       mipmapLevel:0
                         withBytes:neutral
                       bytesPerRow:sizeof(neutral)];
    allocateTargets();
}

ScreenSpaceLighting::~ScreenSpaceLighting() {
    releaseTexture(_linearDepthPyramid);
    releaseTexture(_normalTexture);
    releaseTexture(_traceTexture);
    releaseTexture(_denoiseScratch);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_history[index]);
        releaseTexture(_historyDepth[index]);
        releaseTexture(_momentsAge[index]);
    }
    releaseTexture(_neutralTexture);
    resetMetalObject(_linearDepthPipeline);
    resetMetalObject(_depthReducePipeline);
    resetMetalObject(_normalPipeline);
    resetMetalObject(_tracePipeline);
    resetMetalObject(_temporalPipeline);
    resetMetalObject(_atrousPipeline);
    resetMetalObject(_historyDepthPipeline);
    resetMetalObject(_applyPipeline);
}

void ScreenSpaceLighting::dispatch2D(id<MTLComputeCommandEncoder> encoder,
                                     id<MTLComputePipelineState> pipeline, NSUInteger width,
                                     NSUInteger height) {
    const NSUInteger threadWidth = std::min<NSUInteger>(pipeline.threadExecutionWidth, 16);
    const NSUInteger threadHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(
               pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1), 16));
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
}

void ScreenSpaceLighting::allocateTargets() {
    const ScreenSpaceLightingMemoryFootprint footprint =
        screenSpaceLightingMemoryFootprint(_displayWidth, _displayHeight, _quality);
    _workWidth = footprint.workWidth;
    _workHeight = footprint.workHeight;
    _persistentBytes = footprint.totalBytes();
    releaseTexture(_linearDepthPyramid);
    releaseTexture(_normalTexture);
    releaseTexture(_traceTexture);
    releaseTexture(_denoiseScratch);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_history[index]);
        releaseTexture(_historyDepth[index]);
        releaseTexture(_momentsAge[index]);
    }
    _historyIndex = 0;
    _historyValid = false;
    if (_quality == 0) {
        return;
    }

    _linearDepthPyramid = makeTexture2D(_device, MTLPixelFormatR32Float, _displayWidth,
                                        _displayHeight, true, @"SSL Linear Depth Pyramid");
    _normalTexture = makeTexture2D(_device, MTLPixelFormatRG16Float, _displayWidth, _displayHeight,
                                   false, @"SSL Full-Resolution Normal Guide");
    _traceTexture = makeTexture2D(_device, MTLPixelFormatRGBA16Float, _workWidth, _workHeight,
                                  false, @"SSL Current Trace");
    _denoiseScratch = makeTexture2D(_device, MTLPixelFormatRGBA16Float, _workWidth, _workHeight,
                                    false, @"SSL Denoise Scratch");
    for (uint32_t index = 0; index < 2; ++index) {
        _history[index] = makeTexture2D(_device, MTLPixelFormatRGBA16Float, _workWidth, _workHeight,
                                        false, index == 0 ? @"SSL History A" : @"SSL History B");
        _historyDepth[index] =
            makeTexture2D(_device, MTLPixelFormatR32Float, _workWidth, _workHeight, false,
                          index == 0 ? @"SSL History Depth A" : @"SSL History Depth B");
        _momentsAge[index] =
            makeTexture2D(_device, MTLPixelFormatRGBA16Float, _workWidth, _workHeight, false,
                          index == 0 ? @"SSL Moments A" : @"SSL Moments B");
    }
}

void ScreenSpaceLighting::resize(uint32_t width, uint32_t height) {
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    if (width == _displayWidth && height == _displayHeight) {
        return;
    }
    _displayWidth = width;
    _displayHeight = height;
    allocateTargets();
    resetHistory(INDIRECT_HISTORY_RESIZE);
}

void ScreenSpaceLighting::setQuality(int quality) {
    quality = std::clamp(quality, 0, 2);
    if (_quality == quality) {
        return;
    }
    _quality = quality;
    allocateTargets();
    resetHistory(INDIRECT_HISTORY_QUALITY_CHANGE);
}

void ScreenSpaceLighting::resetHistory(uint32_t reasons) {
    _historyValid = false;
    _lastResetReasons = reasons;
}

void ScreenSpaceLighting::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                                 id<MTLTexture> depthResolve, id<MTLTexture> surfaceResolve,
                                 const IndirectLightingUniforms& uniforms, GpuFrameTimer* timer) {
    if (!commandBuffer || !sceneHDR || !depthResolve || !surfaceResolve) {
        return;
    }

    const uint32_t writeIndex = _historyIndex ^ 1U;
    IndirectLightingUniforms frameUniforms = uniforms;
    frameUniforms.resolutionAndQuality =
        simd_make_float4(static_cast<float>(_workWidth), static_cast<float>(_workHeight),
                         static_cast<float>(_quality), _historyValid ? 1.0F : 0.0F);

    id<MTLTexture> indirectTexture = _neutralTexture;
    if (_quality > 0) {
        id<MTLComputeCommandEncoder> compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Screen-Space Lighting";
        uint32_t computeTimerToken =
            timer ? timer->beginComputePass(compute, "indirectDepth") : UINT32_MAX;
        [compute setComputePipelineState:_linearDepthPipeline];
        [compute setTexture:depthResolve atIndex:0];
        [compute setTexture:_linearDepthPyramid atIndex:1];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        dispatch2D(compute, _linearDepthPipeline, _displayWidth, _displayHeight);
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Screen-Space Normal Guide";
        computeTimerToken =
            timer ? timer->beginComputePass(compute, "indirectNormals") : UINT32_MAX;
        [compute setComputePipelineState:_normalPipeline];
        [compute setTexture:_linearDepthPyramid atIndex:0];
        [compute setTexture:_normalTexture atIndex:1];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        dispatch2D(compute, _normalPipeline, _displayWidth, _displayHeight);
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Conservative Screen-Space Depth Pyramid";
        computeTimerToken =
            timer ? timer->beginComputePass(compute, "indirectDepthPyramid") : UINT32_MAX;
        [compute setComputePipelineState:_depthReducePipeline];
        [compute setTexture:_linearDepthPyramid atIndex:0];
        for (uint32_t level = 1; level < _linearDepthPyramid.mipmapLevelCount; ++level) {
            const uint32_t sourceLevel = level - 1U;
            [compute setBytes:&sourceLevel length:sizeof(sourceLevel) atIndex:0];
            dispatch2D(compute, _depthReducePipeline,
                       std::max<NSUInteger>(_displayWidth >> level, 1U),
                       std::max<NSUInteger>(_displayHeight >> level, 1U));
            [compute memoryBarrierWithScope:MTLBarrierScopeTextures];
        }
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Screen-Space Hi-Z Trace";
        computeTimerToken = timer ? timer->beginComputePass(compute, "indirectTrace") : UINT32_MAX;
        [compute setComputePipelineState:_tracePipeline];
        [compute setTexture:_linearDepthPyramid atIndex:0];
        // The trace reads the scene HDR source directly at the exact hit
        // texel; the apply pass has not run yet, so no bounce feeds back.
        [compute setTexture:sceneHDR atIndex:1];
        [compute setTexture:_normalTexture atIndex:2];
        [compute setTexture:_traceTexture atIndex:3];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        dispatch2D(compute, _tracePipeline, _workWidth, _workHeight);
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Screen-Space Temporal Filter";
        computeTimerToken =
            timer ? timer->beginComputePass(compute, "indirectTemporal") : UINT32_MAX;
        [compute setComputePipelineState:_temporalPipeline];
        [compute setTexture:_traceTexture atIndex:0];
        [compute setTexture:_linearDepthPyramid atIndex:1];
        [compute setTexture:_history[_historyIndex] atIndex:2];
        [compute setTexture:_historyDepth[_historyIndex] atIndex:3];
        [compute setTexture:_history[writeIndex] atIndex:4];
        [compute setTexture:depthResolve atIndex:5];
        [compute setTexture:_normalTexture atIndex:6];
        [compute setTexture:_momentsAge[_historyIndex] atIndex:7];
        [compute setTexture:_momentsAge[writeIndex] atIndex:8];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        dispatch2D(compute, _temporalPipeline, _workWidth, _workHeight);

        [compute setComputePipelineState:_historyDepthPipeline];
        [compute setTexture:_linearDepthPyramid atIndex:0];
        [compute setTexture:_historyDepth[writeIndex] atIndex:1];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        dispatch2D(compute, _historyDepthPipeline, _workWidth, _workHeight);
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        // The wavelet passes read the untouched temporal output, so history
        // feedback stays pre-blur: repeated filtering cannot compound into a
        // low-pass history that misreports variance and erodes voxel edges.
        compute = [commandBuffer computeCommandEncoder];
        compute.label = @"Screen-Space A-Trous Denoise";
        computeTimerToken = timer ? timer->beginComputePass(compute, "indirectAtrous") : UINT32_MAX;
        [compute setComputePipelineState:_atrousPipeline];
        [compute setTexture:_linearDepthPyramid atIndex:1];
        [compute setTexture:_normalTexture atIndex:2];
        [compute setTexture:_momentsAge[writeIndex] atIndex:3];
        [compute setBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
        const uint32_t iterations =
            _quality >= 2 ? INDIRECT_HIGH_ATROUS_ITERATIONS : INDIRECT_MEDIUM_ATROUS_ITERATIONS;
        id<MTLTexture> atrousSource = _history[writeIndex];
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            id<MTLTexture> atrousDestination =
                atrousSource == _traceTexture ? _denoiseScratch : _traceTexture;
            const uint32_t stepSize = 1U << iteration;
            [compute setTexture:atrousSource atIndex:0];
            [compute setTexture:atrousDestination atIndex:4];
            [compute setBytes:&stepSize length:sizeof(stepSize) atIndex:1];
            dispatch2D(compute, _atrousPipeline, _workWidth, _workHeight);
            [compute memoryBarrierWithScope:MTLBarrierScopeTextures];
            atrousSource = atrousDestination;
        }
        if (timer) {
            timer->endComputePass(compute, computeTimerToken);
        }
        [compute endEncoding];

        indirectTexture = atrousSource;
        _historyIndex = writeIndex;
        _historyValid = true;
        _lastResetReasons = INDIRECT_HISTORY_STABLE;
    }

    auto applyPass = [[MTLRenderPassDescriptor alloc] init];
    applyPass.colorAttachments[0].texture = sceneHDR;
    applyPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    applyPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (timer) {
        timer->attachPass(applyPass, "indirectApply");
    }
    id<MTLRenderCommandEncoder> apply =
        [commandBuffer renderCommandEncoderWithDescriptor:applyPass];
    if (!apply) {
        resetMetalObject(applyPass);
        return;
    }
    apply.label = @"Apply Indirect Lighting";
    [apply setRenderPipelineState:_applyPipeline];
    [apply setFragmentTexture:indirectTexture atIndex:0];
    [apply setFragmentTexture:surfaceResolve atIndex:1];
    [apply setFragmentTexture:depthResolve atIndex:2];
    [apply setFragmentTexture:_normalTexture != nil ? _normalTexture : _neutralTexture atIndex:3];
    [apply setFragmentBytes:&frameUniforms length:sizeof(frameUniforms) atIndex:0];
    [apply drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [apply endEncoding];
    resetMetalObject(applyPass);
}
