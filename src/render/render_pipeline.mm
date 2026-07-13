#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "render/bloom.hpp"
#include "render/entity_renderer.hpp"
#include "render/block_textures.hpp"
#include "render/lod_mesher.hpp"

#include "render/particles.hpp"
#include "render/ui_hud.hpp"
#include "render/ui_overlay.hpp"
#include "world/chunk.hpp"
#include "world/chunk_pos.hpp"
#include "world/world.hpp"
#include "engine/camera.hpp"
#include "engine/hotbar.hpp"

#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RenderPipeline::RenderPipeline(id<MTLDevice> device,
                                id<MTLLibrary> shaderLibrary,
                                uint32_t width,
                                uint32_t height)
    : _device(device)
    , _bloomIntensity(1.0f)
    , _displayWidth(width)
    , _displayHeight(height)
    , _frustumPlanes{}
{
    // ---- Load main chunk shader functions ----
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    if (!vertexFunc) {
        RY_LOG_FATAL("Failed to load vertex shader function 'vertexMain'");
    }

    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];
    if (!fragmentFunc) {
        RY_LOG_FATAL("Failed to load fragment shader function 'fragmentMain'");
    }

    // ---- Vertex descriptor (Metal 2.0 API) ----
    auto vertexDesc = [MTLVertexDescriptor vertexDescriptor];

    // Attribute 0: normalIdx (uint)
    vertexDesc.attributes[0].format = MTLVertexFormatUInt;
    vertexDesc.attributes[0].offset = 0;
    vertexDesc.attributes[0].bufferIndex = 0;

    // Attribute 1: position (float3 from Half3)
    vertexDesc.attributes[1].format = MTLVertexFormatHalf3;
    vertexDesc.attributes[1].offset = 4;
    vertexDesc.attributes[1].bufferIndex = 0;

    // Attribute 2: uv (float2 from Half2)
    vertexDesc.attributes[2].format = MTLVertexFormatHalf2;
    vertexDesc.attributes[2].offset = 10;
    vertexDesc.attributes[2].bufferIndex = 0;

    // Buffer layout
    vertexDesc.layouts[0].stride = 16;
    vertexDesc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDesc.layouts[0].stepRate = 1;

    // ---- Main render pipeline state ----
    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;
    pipelineDesc.vertexDescriptor = vertexDesc;

    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    pipelineDesc.rasterSampleCount = 4;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                            error:&error];
    if (!_pipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create render pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Depth stencil state (opaque) ----
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create depth stencil state");
    }

    // ---- Depth-tested, non-writing state (block highlight) ----
    auto noWriteDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    noWriteDepthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    noWriteDepthDesc.depthWriteEnabled = false;
    _noDepthWriteState = [_device newDepthStencilStateWithDescriptor:noWriteDepthDesc];
    if (!_noDepthWriteState) {
        RY_LOG_FATAL("Failed to create no-write depth stencil state");
    }

    // ---- Sky pipeline state ----
    id<MTLFunction> skyVertexFunc = [shaderLibrary newFunctionWithName:@"skyVertexMain"];
    if (!skyVertexFunc) {
        RY_LOG_FATAL("Failed to load sky vertex shader function 'skyVertexMain'");
    }

    id<MTLFunction> skyFragmentFunc = [shaderLibrary newFunctionWithName:@"skyFragmentMain"];
    if (!skyFragmentFunc) {
        RY_LOG_FATAL("Failed to load sky fragment shader function 'skyFragmentMain'");
    }

    auto skyPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    skyPipelineDesc.vertexFunction = skyVertexFunc;
    skyPipelineDesc.fragmentFunction = skyFragmentFunc;
    skyPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    skyPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    skyPipelineDesc.rasterSampleCount = 4;

    _skyPipelineState = [_device newRenderPipelineStateWithDescriptor:skyPipelineDesc
                                                               error:&error];
    if (!_skyPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create sky pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // Sky depth state: always pass, never write — the sky sits behind everything
    auto skyDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    skyDepthDesc.depthCompareFunction = MTLCompareFunctionAlways;
    skyDepthDesc.depthWriteEnabled = false;
    _skyDepthState = [_device newDepthStencilStateWithDescriptor:skyDepthDesc];
    if (!_skyDepthState) {
        RY_LOG_FATAL("Failed to create sky depth stencil state");
    }

    // Sky uniforms buffer
    _skyUniformsBuffer = [_device newBufferWithLength:sizeof(SkyUniforms)
                                                options:MTLResourceStorageModeShared];
    if (!_skyUniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate sky uniforms buffer");
    }

    // ---- Block highlight pipeline state (lines) ----
    id<MTLFunction> highlightVertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    id<MTLFunction> highlightFragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];

    auto highlightPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    highlightPipelineDesc.vertexFunction = highlightVertexFunc;
    highlightPipelineDesc.fragmentFunction = highlightFragmentFunc;
    highlightPipelineDesc.vertexDescriptor = vertexDesc;
    highlightPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    highlightPipelineDesc.colorAttachments[0].blendingEnabled = true;
    highlightPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    highlightPipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    highlightPipelineDesc.rasterSampleCount = 4;

    _highlightPipelineState = [_device newRenderPipelineStateWithDescriptor:highlightPipelineDesc
                                                                     error:&error];
    if (!_highlightPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create highlight pipeline state: %@",
                         error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // Highlight vertex buffer: 24 vertices for wireframe box (12 lines × 2 endpoints)
    // Each vertex: faceAttr(4) + position(6) + uv(4) = 16 bytes.
    // The white texture layer makes the wireframe take its color purely from
    // the uniforms (yellow sun color).
    struct alignas(16) HighlightVertex {
        uint32_t faceAttr;
        float16_t px, py, pz;
        float16_t u, v;
    };
    const uint32_t highlightAttr = packFaceAttr(FaceNormal::PlusY, TEXTURE_LAYER_WHITE);
    HighlightVertex highlightVerts[24];
    std::memset(highlightVerts, 0, sizeof(highlightVerts));

    // Wireframe box vertices: 8 corners × 3 lines each = 24 vertex draws
    // Lines: 12 edges of a unit cube at origin
    float corners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, // Bottom face
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}  // Top face
    };
    uint16_t edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical
    };

    for (int i = 0; i < 12; ++i) {
        int a = edges[i][0];
        int b = edges[i][1];
        highlightVerts[i * 2] = {highlightAttr,
                                 static_cast<float16_t>(corners[a][0] - 0.002f),
                                 static_cast<float16_t>(corners[a][1] - 0.002f),
                                 static_cast<float16_t>(corners[a][2] - 0.002f),
                                 0, 0};
        highlightVerts[i * 2 + 1] = {highlightAttr,
                                     static_cast<float16_t>(corners[b][0] - 0.002f),
                                     static_cast<float16_t>(corners[b][1] - 0.002f),
                                     static_cast<float16_t>(corners[b][2] - 0.002f),
                                     0, 0};
    }

    _highlightVertexBuffer = [_device newBufferWithBytes:highlightVerts
                                                  length:sizeof(highlightVerts)
                                                 options:MTLResourceStorageModeShared];
    if (!_highlightVertexBuffer) {
        RY_LOG_FATAL("Failed to allocate highlight vertex buffer");
    }

    // Highlight uniforms buffer (same layout as Uniforms)
    _highlightUniformsBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                                      options:MTLResourceStorageModeShared];
    if (!_highlightUniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate highlight uniforms buffer");
    }

    // ---- Scene render targets (native resolution) ----
    allocateSceneTargets();

    // ---- Uniforms buffer ----
    _uniformsBuffer = [_device newBufferWithLength:sizeof(Uniforms)
                                              options:MTLResourceStorageModeShared];
    if (!_uniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate uniforms buffer");
    }

    // ---- MegaBuffer (centralized GPU memory for chunk meshes) ----
    _megaBuffer = std::make_unique<MegaBuffer>(_device);

    // ---- Block textures (procedural, one array layer per face texture) ----
    _blockTextures = std::make_unique<BlockTextureArray>(_device);

    // ---- UIOverlay (screen-space HUD rendering) ----
    _uiOverlay = std::make_unique<UIOverlay>(_device, shaderLibrary, _displayWidth, _displayHeight);

    // ---- Cloud pipeline state (Phase 8) ----
    id<MTLFunction> cloudVertexFunc = [shaderLibrary newFunctionWithName:@"cloudVertexMain"];
    id<MTLFunction> cloudFragmentFunc = [shaderLibrary newFunctionWithName:@"cloudFragmentMain"];

    if (cloudVertexFunc && cloudFragmentFunc) {
        auto cloudPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
        cloudPipelineDesc.vertexFunction = cloudVertexFunc;
        cloudPipelineDesc.fragmentFunction = cloudFragmentFunc;
        cloudPipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        cloudPipelineDesc.colorAttachments[0].blendingEnabled = true;
        cloudPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        cloudPipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        cloudPipelineDesc.rasterSampleCount = 4;

        _cloudPipelineState = [_device newRenderPipelineStateWithDescriptor:cloudPipelineDesc
                                                                     error:&error];
    }

    // Cloud depth state (depth test, no write)
    auto cloudDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    cloudDepthDesc.depthCompareFunction = MTLCompareFunctionLess;
    cloudDepthDesc.depthWriteEnabled = false;
    _cloudDepthState = [_device newDepthStencilStateWithDescriptor:cloudDepthDesc];

    // Cloud uniforms buffer
    _cloudUniformsBuffer = [_device newBufferWithLength:sizeof(CloudUniforms)
                                                  options:MTLResourceStorageModeShared];

    // ---- Bloom post-processing (Phase 8) ----
    _bloom = std::make_unique<Bloom>(_device, shaderLibrary, _displayWidth, _displayHeight);
    _bloom->setIntensity(_bloomIntensity);

    // ---- Weather Particle System ----
    _particles = std::make_unique<ParticleSystem>(_device, shaderLibrary);

    // ---- Animal renderer ----
    _entityRenderer = std::make_unique<EntityRenderer>(_device, shaderLibrary);
}

// ---------------------------------------------------------------------------
// allocateSceneTargets — (re)create the MSAA + resolve textures at the
// current drawable size. MSAA targets are memoryless: their contents never
// leave tile memory (color is resolved, depth is discarded at pass end).
// ---------------------------------------------------------------------------
void RenderPipeline::allocateSceneTargets() {
    auto colorMSAADesc = [[MTLTextureDescriptor alloc] init];
    colorMSAADesc.textureType = MTLTextureType2DMultisample;
    colorMSAADesc.pixelFormat = MTLPixelFormatBGRA8Unorm;
    colorMSAADesc.width = _displayWidth;
    colorMSAADesc.height = _displayHeight;
    colorMSAADesc.sampleCount = 4;
    colorMSAADesc.usage = MTLTextureUsageRenderTarget;
    colorMSAADesc.storageMode = MTLStorageModeMemoryless;
    _colorMSAA = [_device newTextureWithDescriptor:colorMSAADesc];
    if (!_colorMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA color texture");
    }

    auto depthMSAADesc = [[MTLTextureDescriptor alloc] init];
    depthMSAADesc.textureType = MTLTextureType2DMultisample;
    depthMSAADesc.pixelFormat = MTLPixelFormatDepth32Float;
    depthMSAADesc.width = _displayWidth;
    depthMSAADesc.height = _displayHeight;
    depthMSAADesc.sampleCount = 4;
    depthMSAADesc.usage = MTLTextureUsageRenderTarget;
    depthMSAADesc.storageMode = MTLStorageModeMemoryless;
    _depthMSAA = [_device newTextureWithDescriptor:depthMSAADesc];
    if (!_depthMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA depth texture");
    }

    auto colorResolveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                   width:_displayWidth
                                                                                  height:_displayHeight
                                                                              mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to allocate color resolve texture");
    }
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------
void RenderPipeline::render(id<MTLCommandQueue> queue,
                            id<CAMetalDrawable> drawable,
                            const Mat4& viewMatrix,
                            const Mat4& projectionMatrix,
                            const World& world,
                            const Camera& camera,
                            uint64_t worldTime,
                            std::optional<Vec3> highlightedBlock,
                            const Hotbar& hotbar,
                            const UIFrameState& uiFrame,
                            const std::vector<std::shared_ptr<Entity>>* entities)
{
    if (!drawable || !queue) return;

    // Track the true drawable size (pixels, not view points — 2x on Retina)
    // so the scene targets always match the surface we resolve into.
    if (drawable.texture.width != _displayWidth || drawable.texture.height != _displayHeight) {
        resize(static_cast<uint32_t>(drawable.texture.width),
               static_cast<uint32_t>(drawable.texture.height));
    }

    // Compute VP matrix and extract frustum planes
    Mat4 vpMatrix = projectionMatrix * viewMatrix;
    extractFrustumPlanes(vpMatrix);

    // Compute day/night uniforms
    float sunDirection[3] = {0.5f, 0.8f, 0.3f};
    float sunColor[3] = {1.0f, 0.95f, 0.9f};
    float ambientColor[3] = {0.35f, 0.35f, 0.4f};
    SkyUniforms skyUniforms{};
    computeDayNightUniforms(worldTime, sunDirection, sunColor, ambientColor, skyUniforms);

    // Normalize sun direction
    float sunLen = std::sqrt(
        sunDirection[0] * sunDirection[0] +
        sunDirection[1] * sunDirection[1] +
        sunDirection[2] * sunDirection[2]);
    if (sunLen > 0.001f) {
        sunDirection[0] /= sunLen;
        sunDirection[1] /= sunLen;
        sunDirection[2] /= sunLen;
    }

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return;

    // Upload sky uniforms
    std::memcpy((void*)_skyUniformsBuffer.contents, &skyUniforms, sizeof(SkyUniforms));

    // ---- Scene pass: sky → chunks → highlight → particles → clouds ----
    // Everything renders into the 4x MSAA target and resolves once into
    // _colorResolve. The memoryless depth buffer is discarded at pass end.
    auto renderPassDesc = [[MTLRenderPassDescriptor alloc] init];
    renderPassDesc.colorAttachments[0].texture = _colorMSAA;
    renderPassDesc.colorAttachments[0].resolveTexture = _colorResolve;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(
        skyUniforms.horizonColor.x,
        skyUniforms.horizonColor.y,
        skyUniforms.horizonColor.z,
        1.0f
    );

    renderPassDesc.depthAttachment.texture = _depthMSAA;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionDontCare;
    renderPassDesc.depthAttachment.clearDepth = 1.0;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) return;

    renderSky(encoder);

    const float fogColor[3] = {skyUniforms.horizonColor.x,
                               skyUniforms.horizonColor.y,
                               skyUniforms.horizonColor.z};
    renderChunks(encoder, world, viewMatrix, projectionMatrix, camera.getPosition(),
                 sunDirection, sunColor, ambientColor, fogColor);

    if (entities && _entityRenderer) {
        _entityRenderer->render(encoder, _uniformsBuffer, *entities,
                                [this](const AABB& aabb) { return isChunkInFrustum(aabb); });
    }

    if (highlightedBlock.has_value()) {
        renderBlockHighlight(encoder, highlightedBlock.value(), viewMatrix, projectionMatrix);
    }

    if (_particles) {
        _particles->render(encoder, viewMatrix, projectionMatrix, camera.getPosition());
    }

    renderClouds(encoder, camera, worldTime, sunDirection);

    [encoder endEncoding];

    // ---- Bloom (extract/blur, then composite into the drawable) ----
    // With zero intensity the bloom pipeline is skipped and the resolved
    // scene is blitted to the drawable unchanged (same dimensions).
    if (_bloom && _bloomIntensity > 0.0f) {
        _bloom->renderBloom(commandBuffer, _colorResolve, drawable.texture);
    } else {
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit) {
            [blit copyFromTexture:_colorResolve
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(_colorResolve.width, _colorResolve.height, 1)
                        toTexture:drawable.texture
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
        }
    }

    // ---- UI Overlay Pass (screen-space HUD at display resolution) ----
    auto uiPassDesc = [[MTLRenderPassDescriptor alloc] init];
    uiPassDesc.colorAttachments[0].texture = drawable.texture;
    uiPassDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    uiPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> uiEncoder = [commandBuffer renderCommandEncoderWithDescriptor:uiPassDesc];
    if (uiEncoder) {
        renderUIOverlay(uiEncoder, hotbar, uiFrame);
        [uiEncoder endEncoding];
    }

    // ---- Optional frame capture (playtest verification) ----
    if (!_capturePath.empty()) {
        encodeFrameCapture(commandBuffer, drawable.texture);
    }

    // Present and commit
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

// ---------------------------------------------------------------------------
// Frame capture — copy the finished drawable into a shared texture and write
// it out as a PNG once the GPU finishes. Lets automated playtests inspect
// real frames without macOS screen-recording permissions.
// ---------------------------------------------------------------------------
void RenderPipeline::requestFrameCapture(const std::string& path) {
    _capturePath = path;
}

void RenderPipeline::encodeFrameCapture(id<MTLCommandBuffer> commandBuffer,
                                        id<MTLTexture> frameTexture) {
    NSString* path = [NSString stringWithUTF8String:_capturePath.c_str()];
    _capturePath.clear();

    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:frameTexture.pixelFormat
                                                                   width:frameTexture.width
                                                                  height:frameTexture.height
                                                               mipmapped:false];
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> capture = [_device newTextureWithDescriptor:desc];
    if (!capture) {
        RY_LOG_ERROR("Frame capture: failed to allocate readback texture");
        return;
    }

    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) return;
    [blit copyFromTexture:frameTexture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(frameTexture.width, frameTexture.height, 1)
                toTexture:capture
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];

    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        const size_t width = capture.width;
        const size_t height = capture.height;
        const size_t bytesPerRow = width * 4;
        std::vector<uint8_t> pixels(bytesPerRow * height);
        [capture getBytes:pixels.data()
              bytesPerRow:bytesPerRow
               fromRegion:MTLRegionMake2D(0, 0, width, height)
              mipmapLevel:0];

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        const uint32_t bitmapInfo =  // BGRA8
            static_cast<uint32_t>(kCGBitmapByteOrder32Little) |
            static_cast<uint32_t>(kCGImageAlphaNoneSkipFirst);
        CGContextRef context = CGBitmapContextCreate(
            pixels.data(), width, height, 8, bytesPerRow, colorSpace, bitmapInfo);
        CGImageRef image = context ? CGBitmapContextCreateImage(context) : nullptr;

        bool ok = false;
        if (image) {
            NSURL* url = [NSURL fileURLWithPath:path];
            CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
                (__bridge CFURLRef)url, (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr);
            if (dest) {
                CGImageDestinationAddImage(dest, image, nullptr);
                ok = CGImageDestinationFinalize(dest);
                CFRelease(dest);
            }
            CGImageRelease(image);
        }
        if (context) CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);

        const std::string pathUtf8 = [path UTF8String];
        if (ok) {
            RY_LOG_INFO(("Frame captured to " + pathUtf8).c_str());
        } else {
            RY_LOG_ERROR(("Frame capture FAILED for " + pathUtf8).c_str());
        }
    }];
}

// ---------------------------------------------------------------------------
// computeDayNightUniforms (Task 6.4-6.5)
// ---------------------------------------------------------------------------
void RenderPipeline::computeDayNightUniforms(uint64_t worldTime,
                                              float sunDirection[3],
                                              float sunColor[3],
                                              float ambientColor[3],
                                              SkyUniforms& skyUniforms)
{
    // Full day = 24000 ticks (20 minutes real time at 20Hz)
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // Orbital angle: 0 = dawn, PI/2 = noon, PI = dusk, 3PI/2 = midnight
    float dayFraction = static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    // Sun direction: rotates in XZ plane with slight Z offset for visual depth
    sunDirection[0] = std::cos(orbitalAngle);
    sunDirection[1] = std::sin(orbitalAngle);
    sunDirection[2] = 0.3f;

    // Sun elevation factor: 1 at noon (angle=PI/2), -1 at midnight (angle=3PI/2)
    float sunElevation = std::sin(orbitalAngle);

    // ---- Sun color: white at noon, orange at sunrise/sunset, dark at night ----
    if (sunElevation > 0.1f) {
        // Daytime: white to slightly warm
        float intensity = std::min(sunElevation, 1.0f);
        sunColor[0] = 1.0f;
        sunColor[1] = 0.95f + 0.05f * intensity;
        sunColor[2] = 0.9f + 0.1f * intensity;
    } else if (sunElevation > -0.1f) {
        // Sunrise/sunset: orange
        float t = (sunElevation + 0.1f) / 0.2f; // 0..1 across twilight
        sunColor[0] = 1.0f;
        sunColor[1] = 0.5f + 0.45f * t;
        sunColor[2] = 0.2f + 0.7f * t;
    } else {
        // Night: very dim
        sunColor[0] = 0.1f;
        sunColor[1] = 0.1f;
        sunColor[2] = 0.15f;
    }

    // ---- Ambient color: 0.35/0.35/0.4 at noon, 0.1/0.1/0.15 at night ----
    float ambientDay[3] = {0.35f, 0.35f, 0.4f};
    float ambientNight[3] = {0.1f, 0.1f, 0.15f};
    float ambientT = std::max(0.0f, std::min(1.0f, (sunElevation + 0.2f) / 0.6f));
    for (int i = 0; i < 3; ++i) {
        ambientColor[i] = ambientNight[i] + (ambientDay[i] - ambientNight[i]) * ambientT;
    }

    // ---- Sky colors ----
    // Daytime sky: blue zenith, lighter horizon
    float dayZenith[3] = {0.2f, 0.4f, 0.8f};
    float dayHorizon[3] = {0.53f, 0.81f, 0.92f};

    // Night sky: very dark blue
    float nightZenith[3] = {0.02f, 0.02f, 0.05f};
    float nightHorizon[3] = {0.05f, 0.05f, 0.1f};

    // Twilight: purple/orange
    float twilightZenith[3] = {0.15f, 0.1f, 0.3f};
    float twilightHorizon[3] = {0.6f, 0.3f, 0.2f};

    // Interpolate based on sun elevation
    float* zenithColor = dayZenith;
    float* horizonColor = dayHorizon;

    if (sunElevation > 0.05f) {
        // Full day
        zenithColor = dayZenith;
        horizonColor = dayHorizon;
    } else if (sunElevation > -0.15f) {
        // Twilight
        zenithColor = twilightZenith;
        horizonColor = twilightHorizon;
    } else {
        // Night
        zenithColor = nightZenith;
        horizonColor = nightHorizon;
    }

    skyUniforms.zenithColor = simd_make_float3(zenithColor[0], zenithColor[1], zenithColor[2]);
    skyUniforms.horizonColor = simd_make_float3(horizonColor[0], horizonColor[1], horizonColor[2]);
    skyUniforms.sunDirection = simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
    skyUniforms.sunColor = simd_make_float3(sunColor[0], sunColor[1], sunColor[2]);
    skyUniforms.sunIntensity = std::max(0.0f, sunElevation);
}

// ---------------------------------------------------------------------------
// renderSky — fullscreen gradient drawn first in the scene pass
// ---------------------------------------------------------------------------
void RenderPipeline::renderSky(id<MTLRenderCommandEncoder> encoder) {
    [encoder setRenderPipelineState:_skyPipelineState];
    [encoder setDepthStencilState:_skyDepthState];
    [encoder setVertexBuffer:_skyUniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_skyUniformsBuffer offset:0 atIndex:1];

    // Draw fullscreen quad (6 vertices, no index buffer)
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

// ---------------------------------------------------------------------------
// renderChunks (opaque pass)
// ---------------------------------------------------------------------------
void RenderPipeline::renderChunks(id<MTLRenderCommandEncoder> encoder,
                                     const World& world,
                                     const Mat4& viewMatrix,
                                     const Mat4& projectionMatrix,
                                     const Vec3& cameraPosition,
                                     const float sunDirection[3],
                                     const float sunColor[3],
                                     const float ambientColor[3],
                                     const float fogColor[3])
{
    // Bind pipeline state
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];

    const float camX = cameraPosition.x;
    const float camY = cameraPosition.y;
    const float camZ = cameraPosition.z;

    // Pack and upload uniforms
    Uniforms uniforms{};

    uniforms.modelMatrix = matrix_identity_float4x4;
    std::memcpy(&uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(&uniforms.projectionMatrix, projectionMatrix.data.data(),
                sizeof(uniforms.projectionMatrix));

    // Lighting
    uniforms.sunDirection = simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
    uniforms.sunColor = simd_make_float3(sunColor[0], sunColor[1], sunColor[2]);
    uniforms.ambientColor = simd_make_float3(ambientColor[0], ambientColor[1], ambientColor[2]);

    // Fog parameters
    uniforms.fogColor = simd_make_float3(fogColor[0], fogColor[1], fogColor[2]);
    uniforms.fogDensity = _fogDensity;

    // Camera position for fog distance calculation
    uniforms.cameraPosition = simd_make_float3(camX, camY, camZ);

    // Upload to GPU
    std::memcpy((void*)_uniformsBuffer.contents, &uniforms, sizeof(Uniforms));

    // Bind uniforms buffer at index 1
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];

    // Bind the shared atlas + uniforms once; every chunk draw reuses them
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

    // LOD mesher instance (stateless — safe to reuse). Everything renders at
    // full detail; see the LOD note in lod_mesher.hpp.
    LODMesher lodMesher;

    auto loadedChunks = world.getLoadedChunks();

    // Cap mesh builds per frame so a burst of freshly generated chunks
    // amortizes over a few frames instead of stalling one.
    constexpr int MAX_MESH_BUILDS_PER_FRAME = 16;
    int meshBuilds = 0;
    bool allocFailureLogged = false;

    _liveChunkKeys.clear();

    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated) continue;

        // Chunk key for mesh cache lookup (packed, no allocation)
        uint64_t key = ChunkPos{chunk->chunkX, chunk->chunkZ}.packed();
        _liveChunkKeys.insert(key);

        // Frustum culling
        AABB chunkAABB = chunk->getAABB();
        if (!isChunkInFrustum(chunkAABB)) continue;

        // (Re)build the mesh when the chunk is dirty or was never meshed
        auto cached = _chunkMeshes.find(key);
        const bool needsBuild = chunk->needsMeshUpdate || cached == _chunkMeshes.end();
        if (needsBuild && meshBuilds < MAX_MESH_BUILDS_PER_FRAME) {
            ++meshBuilds;

            if (cached != _chunkMeshes.end()) {
                if (cached->second.uploaded) {
                    _megaBuffer->free(cached->second.alloc);
                }
                _chunkMeshes.erase(cached);
            }

            MeshOutput mesh = lodMesher.buildMesh(*chunk, static_cast<int>(ChunkLOD::Full));
            chunk->setMeshed(true);
            chunk->needsMeshUpdate = false;

            ChunkMeshState state;  // uploaded == false marks an empty mesh
            if (!mesh.vertices.empty()) {
                try {
                    auto alloc = _megaBuffer->allocate(
                        static_cast<uint32_t>(mesh.vertices.size()),
                        static_cast<uint32_t>(mesh.indices.size()));
                    _megaBuffer->uploadVertices(mesh.vertices.data(),
                                                mesh.vertices.size() * sizeof(Vertex),
                                                alloc.vertexOffset);
                    _megaBuffer->uploadIndices(mesh.indices.data(),
                                               mesh.indices.size() * sizeof(uint32_t),
                                               alloc.indexOffset);
                    state.alloc = alloc;
                    state.uploaded = true;
                } catch (const std::exception& e) {
                    if (!allocFailureLogged) {
                        RY_LOG_ERROR((std::string("Chunk mesh upload failed: ") + e.what()).c_str());
                        allocFailureLogged = true;
                    }
                }
            }
            _chunkMeshes[key] = state;
        }

        cached = _chunkMeshes.find(key);
        if (cached == _chunkMeshes.end() || !cached->second.uploaded) continue;

        const auto& meshState = cached->second;
        if (meshState.alloc.indexCount == 0) continue;

        // Mesh vertices are chunk-local; this restores world space (and keeps
        // fp16 positions exact regardless of how far the chunk is from origin)
        ChunkOrigin origin{};
        origin.origin = simd_make_float4(static_cast<float>(chunk->chunkX * CHUNK_WIDTH), 0.0f,
                                         static_cast<float>(chunk->chunkZ * CHUNK_DEPTH), 0.0f);
        [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];

        // Bind vertex buffer from MegaBuffer allocation
        [encoder setVertexBuffer:meshState.alloc.vertexBuffer
                            offset:meshState.alloc.vertexOffset
                          atIndex:0];

        // Draw indexed primitives (triangles)
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                              indexCount:meshState.alloc.indexCount
                               indexType:MTLIndexTypeUInt32
                             indexBuffer:meshState.alloc.indexBuffer
                         indexBufferOffset:meshState.alloc.indexOffset];
    }

    // Sweep mesh allocations of chunks the world has since unloaded
    for (auto it = _chunkMeshes.begin(); it != _chunkMeshes.end();) {
        if (_liveChunkKeys.count(it->first) == 0) {
            if (it->second.uploaded) {
                _megaBuffer->free(it->second.alloc);
            }
            it = _chunkMeshes.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------
// renderBlockHighlight (Task 6.9)
// ---------------------------------------------------------------------------
void RenderPipeline::renderBlockHighlight(id<MTLRenderCommandEncoder> encoder,
                                           const Vec3& blockPos,
                                           const Mat4& viewMatrix,
                                           const Mat4& projectionMatrix)
{
    // Upload highlight-specific uniforms with translation to block position
    Uniforms uniforms{};

    // Translation matrix to move wireframe box to block position
    // With slight offset (0.002) to prevent z-fighting
    uniforms.modelMatrix = matrix_identity_float4x4;
    uniforms.modelMatrix.columns[3] =
        simd_make_float4(blockPos.x - 0.002f, blockPos.y - 0.002f, blockPos.z - 0.002f, 1.0f);

    std::memcpy(&uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(&uniforms.projectionMatrix, projectionMatrix.data.data(),
                sizeof(uniforms.projectionMatrix));

    // Yellow highlight color
    uniforms.sunDirection = simd_make_float3(1.0f, 1.0f, 0.0f);
    uniforms.sunColor = simd_make_float3(1.0f, 1.0f, 0.0f);
    uniforms.ambientColor = simd_make_float3(0.0f, 0.0f, 0.0f);

    std::memcpy((void*)_highlightUniformsBuffer.contents, &uniforms, sizeof(Uniforms));

    [encoder setRenderPipelineState:_highlightPipelineState];
    [encoder setDepthStencilState:_noDepthWriteState];

    [encoder setVertexBuffer:_highlightVertexBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:_highlightUniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_highlightUniformsBuffer offset:0 atIndex:1];

    // Highlight vertices carry their translation in the model matrix
    ChunkOrigin zeroOrigin{};
    [encoder setVertexBytes:&zeroOrigin length:sizeof(zeroOrigin) atIndex:2];
    // fragmentMain samples the atlas; bind it so the highlight never relies
    // on a texture left over from the chunk loop (e.g. when no chunks drew).
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

    // Draw 12 lines (24 vertices) for wireframe box
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:24];
}

// ---------------------------------------------------------------------------
// renderUIOverlay (Task 6.10 + hotbar)
// ---------------------------------------------------------------------------
void RenderPipeline::renderUIOverlay(id<MTLRenderCommandEncoder> encoder,
                                      const Hotbar& hotbar,
                                      const UIFrameState& uiFrame)
{
    _uiOverlay->beginFrame();
    drawGameHud(*_uiOverlay, hotbar, uiFrame, _displayWidth, _displayHeight);
    if (uiFrame.screen != GameScreen::Playing) {
        drawMenu(*_uiOverlay, uiFrame.menu, uiFrame.hoveredButton, _displayWidth, _displayHeight);
    }
    _uiOverlay->flush(encoder);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
RenderPipeline::~RenderPipeline() = default;

// ---------------------------------------------------------------------------
// resize()
// ---------------------------------------------------------------------------
void RenderPipeline::resize(uint32_t width, uint32_t height) {
    if (width == _displayWidth && height == _displayHeight) return;

    _displayWidth = width;
    _displayHeight = height;

    allocateSceneTargets();

    _uiOverlay->resize(_displayWidth, _displayHeight);

    if (_bloom) {
        _bloom->resize(_displayWidth, _displayHeight);
    }
}

// ---------------------------------------------------------------------------
// setBloomIntensity
// ---------------------------------------------------------------------------
void RenderPipeline::setBloomIntensity(float intensity) {
    _bloomIntensity = intensity;
    if (_bloom) {
        _bloom->setIntensity(intensity);
    }
}

// ---------------------------------------------------------------------------
// tickParticles — Update weather particle physics each game tick
// ---------------------------------------------------------------------------
void RenderPipeline::tickParticles(float dt, const World& world, const Vec3& playerPosition) {
    if (!_particles) return;
    _particles->tick(dt, world, playerPosition);
}

// ---------------------------------------------------------------------------
// Frustum culling
// ---------------------------------------------------------------------------
void RenderPipeline::extractFrustumPlanes(const Mat4& vpMatrix) {
    // Gribb-Hartmann plane extraction for a column-major, column-vector VP
    // matrix. Row i of the matrix is (data[i], data[4+i], data[8+i], data[12+i]).
    // A point is inside a plane when dot(plane, (p, 1)) >= 0.
    auto row = [&vpMatrix](int i, int component) {
        return vpMatrix.data[static_cast<size_t>(component * 4 + i)];
    };

    for (int c = 0; c < 4; ++c) {
        _frustumPlanes[0][c] = row(3, c) + row(0, c);  // left
        _frustumPlanes[1][c] = row(3, c) - row(0, c);  // right
        _frustumPlanes[2][c] = row(3, c) + row(1, c);  // bottom
        _frustumPlanes[3][c] = row(3, c) - row(1, c);  // top
        _frustumPlanes[4][c] = row(2, c);              // near (Metal: z' >= 0)
        _frustumPlanes[5][c] = row(3, c) - row(2, c);  // far
    }

    for (int i = 0; i < 6; ++i) {
        float len = std::sqrt(
            _frustumPlanes[i][0] * _frustumPlanes[i][0] +
            _frustumPlanes[i][1] * _frustumPlanes[i][1] +
            _frustumPlanes[i][2] * _frustumPlanes[i][2]
        );
        if (len > 0.0f) {
            _frustumPlanes[i][0] /= len;
            _frustumPlanes[i][1] /= len;
            _frustumPlanes[i][2] /= len;
            _frustumPlanes[i][3] /= len;
        }
    }
}

bool RenderPipeline::isChunkInFrustum(const AABB& chunkAABB) const {
    Vec3 center = chunkAABB.center();
    Vec3 extents = chunkAABB.size() * 0.5f;

    for (int i = 0; i < 6; ++i) {
        float A = _frustumPlanes[i][0];
        float B = _frustumPlanes[i][1];
        float C = _frustumPlanes[i][2];
        float D = _frustumPlanes[i][3];

        float dist = A * center.x + B * center.y + C * center.z + D;
        float extent = std::abs(A) * extents.x +
                       std::abs(B) * extents.y +
                       std::abs(C) * extents.z;

        if (dist + extent < 0.f) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// renderClouds — alpha-blended cloud layer, drawn last in the scene pass
// ============================================================================
void RenderPipeline::renderClouds(id<MTLRenderCommandEncoder> encoder,
                                   const Camera& camera,
                                   uint64_t worldTime,
                                   const float sunDirection[3])
{
    if (!_cloudPipelineState) return;

    // Cloud uniforms
    CloudUniforms cloudUniforms{};

    Vec3 camPos = camera.getPosition();
    Vec3 camFwd = camera.forward();
    Vec3 camRight = camera.right();
    Vec3 camUp = camera.up();
    cloudUniforms.cameraPosition = simd_make_float3(camPos.x, camPos.y, camPos.z);
    cloudUniforms.cameraForward = simd_make_float3(camFwd.x, camFwd.y, camFwd.z);
    cloudUniforms.cameraRight = simd_make_float3(camRight.x, camRight.y, camRight.z);
    cloudUniforms.cameraUp = simd_make_float3(camUp.x, camUp.y, camUp.z);
    cloudUniforms.sunDirection =
        simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);

    // Projection shape for per-pixel ray reconstruction
    cloudUniforms.tanHalfFov =
        std::tan(camera.FOV() * 0.5f * static_cast<float>(M_PI) / 180.0f);
    cloudUniforms.aspect =
        static_cast<float>(_displayWidth) / static_cast<float>(std::max(_displayHeight, 1u));

    // Wind offset: worldTime * windSpeed (0.02 blocks/tick)
    cloudUniforms.windOffset = static_cast<float>(worldTime) * 0.02f;

    // Cloud parameters
    cloudUniforms.cloudAltitude = 192.0f;
    cloudUniforms.noiseFrequency = 0.005f;
    cloudUniforms.cloudThreshold = 0.55f;

    std::memcpy((void*)_cloudUniformsBuffer.contents, &cloudUniforms, sizeof(cloudUniforms));

    [encoder setRenderPipelineState:_cloudPipelineState];
    [encoder setDepthStencilState:_cloudDepthState];
    [encoder setVertexBuffer:_cloudUniformsBuffer offset:0 atIndex:0];
    [encoder setFragmentBuffer:_cloudUniformsBuffer offset:0 atIndex:0];

    // Draw fullscreen quad (6 vertices)
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}
