#import "render/lightning_renderer.hpp"

#include "common/error.hpp"
#include "render/gpu_timer.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace {

constexpr uint32_t MAIN_BOLT_SEGMENTS = 48;
constexpr uint32_t BRANCH_SEGMENTS = 8;
constexpr uint32_t MAX_RENDERED_EVENTS = 4;

uint32_t branchCount(uint64_t eventId) noexcept {
    const uint32_t low = static_cast<uint32_t>(eventId);
    const uint32_t high = static_cast<uint32_t>(eventId >> 32U);
    return 2U + ((low ^ high) % 3U);
}

void configureAdditiveColor(MTLRenderPipelineColorAttachmentDescriptor* attachment) {
    attachment.pixelFormat = PixelFormats::SCENE_HDR;
    attachment.blendingEnabled = true;
    attachment.rgbBlendOperation = MTLBlendOperationAdd;
    attachment.alphaBlendOperation = MTLBlendOperationAdd;
    attachment.sourceRGBBlendFactor = MTLBlendFactorOne;
    attachment.sourceAlphaBlendFactor = MTLBlendFactorZero;
    attachment.destinationRGBBlendFactor = MTLBlendFactorOne;
    attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
    attachment.writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue;
}

id<MTLRenderPipelineState> makePipeline(id<MTLDevice> device, id<MTLFunction> vertexFunction,
                                        id<MTLFunction> fragmentFunction, bool usesDepth,
                                        NSString* label) {
    auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.label = label;
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    configureAdditiveColor(descriptor.colorAttachments[0]);
    if (usesDepth) {
        descriptor.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
    }
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                                 error:&error];
    resetMetalObject(descriptor);
    if (!pipeline) {
        RY_LOG_FATAL("Failed to create lightning render pipeline");
    }
    return pipeline;
}

LightningUniforms uniformsForEvent(const LightningEvent& event, float intensity,
                                   const simd_float4x4& viewProjection, simd_float3 cameraPosition,
                                   uint32_t viewportWidth, uint32_t viewportHeight) {
    LightningUniforms uniforms{};
    uniforms.viewProjection = viewProjection;
    uniforms.cameraPosition = cameraPosition;
    uniforms.strikePosition =
        simd_make_float3(static_cast<float>(event.x), event.y, static_cast<float>(event.z));
    uniforms.colorAndIntensity = simd_make_float4(0.66F, 0.78F, 1.0F, intensity);
    const uint32_t packedViewport =
        std::min(viewportWidth, 65'535U) | (std::min(viewportHeight, 65'535U) << 16U);
    uniforms.eventAndShape =
        simd_make_uint4(static_cast<uint32_t>(event.id), static_cast<uint32_t>(event.id >> 32U),
                        std::bit_cast<uint32_t>(event.cloudY), packedViewport);
    return uniforms;
}

} // namespace

float lightningFlashIntensity(const LightningEvent& event, uint64_t currentWorldTick,
                              float ticksPerSecond) noexcept {
    if (currentWorldTick < event.tick || !std::isfinite(ticksPerSecond) ||
        !std::isfinite(event.intensity) || ticksPerSecond <= 0.0F) {
        return 0.0F;
    }
    const float age = static_cast<float>(currentWorldTick - event.tick) / ticksPerSecond;
    if (age > 0.55F) {
        return 0.0F;
    }
    const float primary = std::exp(-age * 22.0F);
    const float secondaryOffset = (age - 0.12F) / 0.032F;
    const float tertiaryOffset = (age - 0.24F) / 0.045F;
    const float secondary = 0.52F * std::exp(-(secondaryOffset * secondaryOffset));
    const float tertiary = 0.22F * std::exp(-(tertiaryOffset * tertiaryOffset));
    return std::clamp(event.intensity, 0.0F, 1.25F) *
           std::clamp(primary + secondary + tertiary, 0.0F, 1.25F);
}

uint32_t lightningBoltSegmentCount(uint64_t eventId) noexcept {
    return MAIN_BOLT_SEGMENTS + branchCount(eventId) * BRANCH_SEGMENTS;
}

LightningRenderer::LightningRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary) {
    id<MTLFunction> boltVertex = [shaderLibrary newFunctionWithName:@"lightningBoltVertex"];
    id<MTLFunction> boltFragment = [shaderLibrary newFunctionWithName:@"lightningBoltFragment"];
    id<MTLFunction> flashVertex = [shaderLibrary newFunctionWithName:@"lightningFullscreenVertex"];
    id<MTLFunction> flashFragment = [shaderLibrary newFunctionWithName:@"lightningFlashFragment"];
    if (!boltVertex || !boltFragment || !flashVertex || !flashFragment) {
        RY_LOG_FATAL("Failed to load lightning shader functions");
    }

    _boltPipeline = makePipeline(device, boltVertex, boltFragment, true, @"Lightning Bolts");
    _flashPipeline =
        makePipeline(device, flashVertex, flashFragment, false, @"Lightning Atmospheric Flash");
    resetMetalObject(boltVertex);
    resetMetalObject(boltFragment);
    resetMetalObject(flashVertex);
    resetMetalObject(flashFragment);

    auto depthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    depthDescriptor.depthCompareFunction = MTLCompareFunctionLessEqual;
    depthDescriptor.depthWriteEnabled = false;
    _depthState = [device newDepthStencilStateWithDescriptor:depthDescriptor];
    resetMetalObject(depthDescriptor);
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create lightning depth state");
    }

    auto neutralDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    neutralDescriptor.usage = MTLTextureUsageShaderRead;
    neutralDescriptor.storageMode = MTLStorageModeShared;
    _neutralCloudDepth = [device newTextureWithDescriptor:neutralDescriptor];
    if (!_neutralCloudDepth) {
        RY_LOG_FATAL("Failed to create neutral lightning cloud depth");
    }
    const uint16_t zero = 0;
    [_neutralCloudDepth replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                          mipmapLevel:0
                            withBytes:&zero
                          bytesPerRow:sizeof(zero)];
}

LightningRenderer::~LightningRenderer() {
    resetMetalObject(_boltPipeline);
    resetMetalObject(_flashPipeline);
    resetMetalObject(_depthState);
    resetMetalObject(_neutralCloudDepth);
}

void LightningRenderer::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                               id<MTLTexture> sceneDepth, id<MTLTexture> cloudHitDepth,
                               const simd_float4x4& viewProjection, simd_float3 cameraPosition,
                               std::span<const LightningEvent> events, uint64_t currentWorldTick,
                               float ticksPerSecond, GpuFrameTimer* timer) {
    _stats = {};
    if (!commandBuffer || !sceneHDR || !sceneDepth || events.empty()) {
        return;
    }

    uint32_t activeCount = 0;
    for (const LightningEvent& event : events) {
        if (lightningFlashIntensity(event, currentWorldTick, ticksPerSecond) > 0.001F &&
            event.cloudY > event.y) {
            ++activeCount;
            if (activeCount == MAX_RENDERED_EVENTS) {
                break;
            }
        }
    }
    if (activeCount == 0) {
        return;
    }

    id<MTLTexture> cloudDepth = cloudHitDepth != nil ? cloudHitDepth : _neutralCloudDepth;
    auto boltPass = [[MTLRenderPassDescriptor alloc] init];
    boltPass.colorAttachments[0].texture = sceneHDR;
    boltPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    boltPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    boltPass.depthAttachment.texture = sceneDepth;
    boltPass.depthAttachment.loadAction = MTLLoadActionLoad;
    boltPass.depthAttachment.storeAction = MTLStoreActionStore;
    if (timer) {
        timer->attachPass(boltPass, "lightningBolts");
    }
    id<MTLRenderCommandEncoder> boltEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:boltPass];
    if (boltEncoder == nil) {
        resetMetalObject(boltPass);
        return;
    }
    boltEncoder.label = @"Depth-Aware Lightning Bolts";
    [boltEncoder setRenderPipelineState:_boltPipeline];
    [boltEncoder setDepthStencilState:_depthState];
    [boltEncoder setFragmentTexture:cloudDepth atIndex:0];

    uint32_t rendered = 0;
    for (const LightningEvent& event : events) {
        const float intensity = lightningFlashIntensity(event, currentWorldTick, ticksPerSecond);
        if (intensity <= 0.001F || event.cloudY <= event.y) {
            continue;
        }
        const LightningUniforms uniforms = uniformsForEvent(
            event, intensity, viewProjection, cameraPosition, static_cast<uint32_t>(sceneHDR.width),
            static_cast<uint32_t>(sceneHDR.height));
        [boltEncoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [boltEncoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [boltEncoder drawPrimitives:MTLPrimitiveTypeLine
                        vertexStart:0
                        vertexCount:lightningBoltSegmentCount(event.id) * 2U];
        _stats.lastEventId = event.id;
        _stats.peakFlashIntensity = std::max(_stats.peakFlashIntensity, intensity);
        if (++rendered >= MAX_RENDERED_EVENTS) {
            break;
        }
    }
    [boltEncoder endEncoding];
    resetMetalObject(boltPass);

    auto flashPass = [[MTLRenderPassDescriptor alloc] init];
    flashPass.colorAttachments[0].texture = sceneHDR;
    flashPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    flashPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (timer) {
        timer->attachPass(flashPass, "lightningFlash");
    }
    id<MTLRenderCommandEncoder> flashEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:flashPass];
    if (flashEncoder == nil) {
        _stats.renderedEventCount = rendered;
        resetMetalObject(flashPass);
        return;
    }
    flashEncoder.label = @"Cloud-Aware Lightning Flash";
    [flashEncoder setRenderPipelineState:_flashPipeline];
    [flashEncoder setFragmentTexture:sceneDepth atIndex:0];
    [flashEncoder setFragmentTexture:cloudDepth atIndex:1];
    rendered = 0;
    for (const LightningEvent& event : events) {
        const float intensity = lightningFlashIntensity(event, currentWorldTick, ticksPerSecond);
        if (intensity <= 0.001F || event.cloudY <= event.y) {
            continue;
        }
        const LightningUniforms uniforms = uniformsForEvent(
            event, intensity, viewProjection, cameraPosition, static_cast<uint32_t>(sceneHDR.width),
            static_cast<uint32_t>(sceneHDR.height));
        [flashEncoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [flashEncoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [flashEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        if (++rendered >= MAX_RENDERED_EVENTS) {
            break;
        }
    }
    [flashEncoder endEncoding];
    resetMetalObject(flashPass);
    _stats.renderedEventCount = rendered;
}
