#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "render/block_textures.hpp"
#include "render/bloom.hpp"
#include "render/entity_renderer.hpp"
#include "render/lod_mesher.hpp"

#include "engine/camera.hpp"
#include "engine/hotbar.hpp"
#include "render/particles.hpp"
#include "render/ui_hud.hpp"
#include "render/ui_overlay.hpp"
#include "world/chunk.hpp"
#include "world/chunk_pos.hpp"
#include "world/world.hpp"

#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RenderPipeline::RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                               uint32_t height)
    : _device(device), _bloomIntensity(1.0f), _displayWidth(width), _displayHeight(height),
      _frustumPlanes{} {
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
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
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

    _skyPipelineState = [_device newRenderPipelineStateWithDescriptor:skyPipelineDesc error:&error];
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
    highlightPipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
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
    const uint32_t highlightAttr = packFaceAttr(FaceNormal::PLUS_Y, TEXTURE_LAYER_WHITE);
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
                                 0,
                                 0};
        highlightVerts[i * 2 + 1] = {highlightAttr,
                                     static_cast<float16_t>(corners[b][0] - 0.002f),
                                     static_cast<float16_t>(corners[b][1] - 0.002f),
                                     static_cast<float16_t>(corners[b][2] - 0.002f),
                                     0,
                                     0};
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

    _waterUniformsBuffer = [_device newBufferWithLength:sizeof(WaterUniforms)
                                                options:MTLResourceStorageModeShared];
    if (!_waterUniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate water uniforms buffer");
    }

    // ---- Water pipeline states ----
    // Water composites its own pixels from the resolved scene: single
    // sample, color-only (manual depth test in the shader), no blending.
    {
        id<MTLFunction> waterVertexFunc = [shaderLibrary newFunctionWithName:@"waterVertexMain"];
        id<MTLFunction> waterFragmentFunc =
            [shaderLibrary newFunctionWithName:@"waterFragmentMain"];
        if (!waterVertexFunc || !waterFragmentFunc) {
            RY_LOG_FATAL("Failed to load water shader functions");
        }
        auto waterDesc = [[MTLRenderPipelineDescriptor alloc] init];
        waterDesc.vertexFunction = waterVertexFunc;
        waterDesc.fragmentFunction = waterFragmentFunc;
        waterDesc.vertexDescriptor = vertexDesc;
        waterDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        waterDesc.rasterSampleCount = 1;
        _waterPipelineState = [_device newRenderPipelineStateWithDescriptor:waterDesc error:&error];
        if (!_waterPipelineState) {
            NSString* msg = [NSString stringWithFormat:@"Failed to create water pipeline: %@",
                                                       error.localizedDescription];
            RY_LOG_FATAL(msg.UTF8String);
        }

        id<MTLFunction> overlayVertexFunc =
            [shaderLibrary newFunctionWithName:@"underwaterOverlayVertex"];
        id<MTLFunction> overlayFragmentFunc =
            [shaderLibrary newFunctionWithName:@"underwaterOverlayFragment"];
        if (!overlayVertexFunc || !overlayFragmentFunc) {
            RY_LOG_FATAL("Failed to load underwater overlay shader functions");
        }
        auto overlayDesc = [[MTLRenderPipelineDescriptor alloc] init];
        overlayDesc.vertexFunction = overlayVertexFunc;
        overlayDesc.fragmentFunction = overlayFragmentFunc;
        overlayDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        overlayDesc.colorAttachments[0].blendingEnabled = true;
        overlayDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        overlayDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        overlayDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        overlayDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        overlayDesc.colorAttachments[0].destinationRGBBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
        overlayDesc.colorAttachments[0].destinationAlphaBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
        overlayDesc.rasterSampleCount = 1;
        _underwaterOverlayState = [_device newRenderPipelineStateWithDescriptor:overlayDesc
                                                                          error:&error];
        if (!_underwaterOverlayState) {
            NSString* msg =
                [NSString stringWithFormat:@"Failed to create underwater overlay pipeline: %@",
                                           error.localizedDescription];
            RY_LOG_FATAL(msg.UTF8String);
        }
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
        cloudPipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
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

    // ---- GPU timing (per-pass sampling is a diagnostic opt-in) ----
    const char* counters = std::getenv("RYCRAFT_GPU_COUNTERS");
    _gpuTimer = std::make_unique<GpuFrameTimer>(_device, counters && *counters &&
                                                             std::strcmp(counters, "0") != 0);
}

// ---------------------------------------------------------------------------
// allocateSceneTargets — (re)create the MSAA + resolve textures at the
// current drawable size. MSAA targets are memoryless: their tile contents
// are resolved at pass end (color into _colorResolve, depth into
// _depthResolve for the water pass) and never loaded or stored.
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

    auto colorResolveDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:false];
    colorResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    _colorResolve = [_device newTextureWithDescriptor:colorResolveDesc];
    if (!_colorResolve) {
        RY_LOG_FATAL("Failed to allocate color resolve texture");
    }

    // ---- Water pass inputs ----
    // The scene depth resolves here (min filter: nearest sample wins) so the
    // water shader can depth-test and reconstruct the world behind pixels.
    auto depthResolveDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:false];
    depthResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    depthResolveDesc.storageMode = MTLStorageModePrivate;
    _depthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    if (!_depthResolve) {
        RY_LOG_FATAL("Failed to allocate depth resolve texture");
    }

    // Refraction samples a copy of the resolved color (a render target
    // cannot sample itself)
    auto sceneCopyDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:false];
    sceneCopyDesc.usage = MTLTextureUsageShaderRead;
    sceneCopyDesc.storageMode = MTLStorageModePrivate;
    _sceneColorCopy = [_device newTextureWithDescriptor:sceneCopyDesc];
    if (!_sceneColorCopy) {
        RY_LOG_FATAL("Failed to allocate scene copy texture");
    }
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------
void RenderPipeline::render(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable,
                            const Mat4& viewMatrix, const Mat4& projectionMatrix,
                            const World& world, const Camera& camera, uint64_t worldTime,
                            std::optional<Vec3> highlightedBlock, const Hotbar& hotbar,
                            const UIFrameState& uiFrame,
                            const std::vector<std::shared_ptr<Entity>>* entities) {
    if (!drawable || !queue)
        return;

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
    float sunLen = std::sqrt(sunDirection[0] * sunDirection[0] + sunDirection[1] * sunDirection[1] +
                             sunDirection[2] * sunDirection[2]);
    if (sunLen > 0.001f) {
        sunDirection[0] /= sunLen;
        sunDirection[1] /= sunLen;
        sunDirection[2] /= sunLen;
    }

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer)
        return;

    _gpuTimer->beginFrame();

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
        skyUniforms.horizonColor.x, skyUniforms.horizonColor.y, skyUniforms.horizonColor.z, 1.0f);

    // Depth resolves out of tile memory (min filter: nearest sample) so the
    // water pass can depth-test and reconstruct world positions.
    renderPassDesc.depthAttachment.texture = _depthMSAA;
    renderPassDesc.depthAttachment.resolveTexture = _depthResolve;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterMin;
    renderPassDesc.depthAttachment.clearDepth = 1.0;
    _gpuTimer->attachPass(renderPassDesc, "scene");

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder)
        return;

    renderSky(encoder);

    // Underwater the whole scene sinks into a dense blue veil (light
    // attenuation); the water pass adds the god-ray overlay on top.
    const bool cameraUnderwater = uiFrame.cameraUnderwater;
    const float fogColor[3] = {cameraUnderwater ? 0.05f : skyUniforms.horizonColor.x,
                               cameraUnderwater ? 0.15f : skyUniforms.horizonColor.y,
                               cameraUnderwater ? 0.32f : skyUniforms.horizonColor.z};
    const float savedFogDensity = _fogDensity;
    if (cameraUnderwater) {
        _fogDensity = std::max(_fogDensity, 0.035f);
    }
    renderChunks(encoder, world, viewMatrix, projectionMatrix, camera.getPosition(), sunDirection,
                 sunColor, ambientColor, fogColor);

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

    // ---- Water pass (refraction/reflection/caustics over the resolved scene) ----
    renderWater(commandBuffer, viewMatrix, projectionMatrix, camera.getPosition(), cameraUnderwater,
                skyUniforms, fogColor, worldTime);
    _fogDensity = savedFogDensity;

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
    _gpuTimer->attachPass(uiPassDesc, "ui");

    id<MTLRenderCommandEncoder> uiEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:uiPassDesc];
    if (uiEncoder) {
        renderUIOverlay(uiEncoder, hotbar, uiFrame);
        [uiEncoder endEncoding];
    }

    // ---- Optional frame capture (playtest verification) ----
    if (!_capturePath.empty()) {
        encodeFrameCapture(commandBuffer, drawable.texture);
    }

    _gpuTimer->endFrame(commandBuffer);

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
    if (!blit)
        return;
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
        const uint32_t bitmapInfo = // BGRA8
            static_cast<uint32_t>(kCGBitmapByteOrder32Little) |
            static_cast<uint32_t>(kCGImageAlphaNoneSkipFirst);
        CGContextRef context = CGBitmapContextCreate(pixels.data(), width, height, 8, bytesPerRow,
                                                     colorSpace, bitmapInfo);
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
        if (context)
            CGContextRelease(context);
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
void RenderPipeline::computeDayNightUniforms(uint64_t worldTime, float sunDirection[3],
                                             float sunColor[3], float ambientColor[3],
                                             SkyUniforms& skyUniforms) {
    // Full day = 24000 ticks (20 minutes real time at 20Hz)
    static constexpr uint64_t TICKS_PER_DAY = 24000;

    // Orbital angle: 0 = dawn, PI/2 = noon, PI = dusk, 3PI/2 = midnight
    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    // Sun direction: rotates in XZ plane with slight Z offset for visual depth
    sunDirection[0] = std::cos(orbitalAngle);
    sunDirection[1] = std::sin(orbitalAngle);
    sunDirection[2] = 0.3f;

    // Sun elevation factor: 1 at noon (angle=PI/2), -1 at midnight (angle=3PI/2)
    float sunElevation = std::sin(orbitalAngle);

    // ---- Sun color: white at noon, orange at sunrise/sunset, dim at night ----
    // Piecewise formulas blended continuously — visible color snapping at the
    // branch thresholds made dawn and dusk read as abrupt jumps.
    float sunColorDay[3] = {1.0f, 0.95f + 0.05f * std::min(std::max(sunElevation, 0.0f), 1.0f),
                            0.9f + 0.1f * std::min(std::max(sunElevation, 0.0f), 1.0f)};
    float sunColorSunset[3] = {1.0f, 0.5f, 0.2f};
    float sunColorNight[3] = {0.1f, 0.1f, 0.15f};

    // 0 → deep sunset colors, 1 → full day colors, over elevation 0..0.35
    float dayBlend = std::clamp(sunElevation / 0.35f, 0.0f, 1.0f);
    // 0 → sunset colors, 1 → night colors, over elevation -0.05..-0.35
    float nightBlend = std::clamp((-sunElevation - 0.05f) / 0.30f, 0.0f, 1.0f);

    auto blend3 = [](const float a[3], const float b[3], float t, float out[3]) {
        for (int i = 0; i < 3; ++i) {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
    };

    blend3(sunColorSunset, sunColorDay, dayBlend, sunColor);
    blend3(sunColor, sunColorNight, nightBlend, sunColor);

    // ---- Ambient: bright at noon, dim at night ----
    float ambientDay[3] = {0.35f, 0.35f, 0.4f};
    float ambientNight[3] = {0.1f, 0.1f, 0.15f};
    float ambientT = std::clamp((sunElevation + 0.2f) / 0.6f, 0.0f, 1.0f);
    blend3(ambientNight, ambientDay, ambientT, ambientColor);

    // ---- Sky palette: blend twilight → day above the horizon and
    // twilight → night below it, so dawn and dusk sweep smoothly ----
    float dayZenith[3] = {0.2f, 0.4f, 0.8f};
    float dayHorizon[3] = {0.53f, 0.81f, 0.92f};
    float nightZenith[3] = {0.02f, 0.02f, 0.05f};
    float nightHorizon[3] = {0.05f, 0.05f, 0.1f};
    float twilightZenith[3] = {0.15f, 0.1f, 0.3f};
    float twilightHorizon[3] = {0.6f, 0.3f, 0.2f};

    float zenith[3];
    float horizon[3];
    blend3(twilightZenith, dayZenith, dayBlend, zenith);
    blend3(zenith, nightZenith, nightBlend, zenith);
    blend3(twilightHorizon, dayHorizon, dayBlend, horizon);
    blend3(horizon, nightHorizon, nightBlend, horizon);
    const float* zenithColor = zenith;
    const float* horizonColor = horizon;

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
void RenderPipeline::renderChunks(id<MTLRenderCommandEncoder> encoder, const World& world,
                                  const Mat4& viewMatrix, const Mat4& projectionMatrix,
                                  const Vec3& cameraPosition, const float sunDirection[3],
                                  const float sunColor[3], const float ambientColor[3],
                                  const float fogColor[3]) {
    // Bind pipeline state
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];
    // Flora cross-quads are single-winding and depend on cull mode None
    // (Metal's default — pinned here so it survives future changes)
    [encoder setCullMode:MTLCullModeNone];

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

    // Bind the shared atlas + uniforms once; every chunk draw reuses them
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

    // Water draws recorded here render later, in the dedicated water pass
    _waterDraws.clear();

    // Builds only happen within the render radius: the generation radius is
    // one chunk wider, so every meshable chunk has generated neighbors for
    // its snapshot (frontier chunks simply wait their turn).
    const int camChunkX = Chunk::worldToChunk(static_cast<int>(std::floor(camX)));
    const int camChunkZ = Chunk::worldToChunk(static_cast<int>(std::floor(camZ)));
    const int renderRadius = world.getViewDistance();

    auto loadedChunks = world.getLoadedChunks();

    // Sweep mesh allocations of chunks the world has since unloaded BEFORE
    // building, so the freed mega-buffer space serves this frame's builds
    _liveChunkKeys.clear();
    for (const auto& chunk : loadedChunks) {
        if (chunk) {
            _liveChunkKeys.insert(ChunkPos{chunk->chunkX, chunk->chunkZ}.packed());
        }
    }
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

    // ---- Async meshing: workers build, the render thread only uploads ----
    if (!_meshScheduler) {
        _meshScheduler = std::make_unique<MeshScheduler>(world, 2);
    }

    // ---- MegaBuffer sized to the view distance ----
    // Full-detail chunks average ~100 KB of vertices; 128 KB × visible
    // chunks + 30% headroom keeps the free-list from thrashing. Growing
    // recreates the buffers and drops every mesh — a settings-screen event,
    // after which everything re-streams through the workers.
    {
        uint64_t visibleChunks = static_cast<uint64_t>(2 * renderRadius + 1) *
                                 static_cast<uint64_t>(2 * renderRadius + 1);
        uint64_t requiredVertexBytes =
            std::max<uint64_t>(128ull * 1024 * 1024, visibleChunks * 128ull * 1024 * 13 / 10);
        if (requiredVertexBytes > _megaBuffer->vertexCapacity()) {
            RY_LOG_INFO((std::string("Growing mega-buffer for view distance ") +
                         std::to_string(renderRadius) + ": " +
                         std::to_string(requiredVertexBytes / (1024 * 1024)) + " MB vertices")
                            .c_str());
            _megaBuffer =
                std::make_unique<MegaBuffer>(_device, requiredVertexBytes, requiredVertexBytes / 2);
            _chunkMeshes.clear(); // allocations died with the old buffers
        }
    }

    // Upload one finished mesh into the registry. Returns false on a
    // transient MegaBuffer-full failure (builtVersion stays 0, so the chunk
    // re-requests once space frees up).
    bool allocFailureLogged = false;
    auto applyMesh = [&](uint64_t key, const MeshOutput& mesh, uint32_t builtVersion) -> bool {
        ChunkMeshState& state = _chunkMeshes[key];
        if (state.uploaded) {
            _megaBuffer->free(state.alloc);
            state.uploaded = false;
        }
        state.requestedVersion = 0;
        state.opaqueIndexCount = mesh.opaqueIndexCount;
        if (mesh.vertices.empty()) {
            state.builtVersion = builtVersion; // all-air: nothing to draw
            return true;
        }
        try {
            auto alloc = _megaBuffer->allocate(static_cast<uint32_t>(mesh.vertices.size()),
                                               static_cast<uint32_t>(mesh.indices.size()));
            _megaBuffer->uploadVertices(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
                                        alloc.vertexOffset);
            _megaBuffer->uploadIndices(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
                                       alloc.indexOffset);
            state.alloc = alloc;
            state.uploaded = true;
            state.builtVersion = builtVersion;
            return true;
        } catch (const std::exception& e) {
            if (!allocFailureLogged) {
                RY_LOG_ERROR((std::string("Chunk mesh upload failed: ") + e.what()).c_str());
                allocFailureLogged = true;
            }
            state.builtVersion = 0;
            return false;
        }
    };

    // 1. Drain worker results and upload within the per-frame budget; the
    //    leftovers stay in _pendingResults for next frame.
    _meshScheduler->drainCompleted(_pendingResults);
    constexpr int MAX_MESH_UPLOADS_PER_FRAME = 24;
    constexpr size_t MAX_UPLOAD_BYTES_PER_FRAME = 8 * 1024 * 1024;
    int uploads = 0;
    size_t uploadBytes = 0;
    size_t resultsConsumed = 0;
    for (MeshResult& result : _pendingResults) {
        if (uploads >= MAX_MESH_UPLOADS_PER_FRAME || uploadBytes >= MAX_UPLOAD_BYTES_PER_FRAME) {
            break;
        }
        uint64_t key = result.pos.packed();
        if (_liveChunkKeys.count(key) == 0) {
            ++resultsConsumed; // chunk unloaded while meshing — drop
            continue;
        }
        if (!result.snapshotOk) {
            // A neighbor was missing: forget the request so the candidate
            // scan below retries once the frontier catches up
            auto it = _chunkMeshes.find(key);
            if (it != _chunkMeshes.end()) {
                it->second.requestedVersion = 0;
            }
            ++resultsConsumed;
            continue;
        }
        if (!applyMesh(key, result.mesh, result.builtVersion)) {
            break; // MegaBuffer full: retry this result next frame
        }
        ++uploads;
        uploadBytes += result.mesh.vertices.size() * sizeof(Vertex) +
                       result.mesh.indices.size() * sizeof(uint32_t);
        ++resultsConsumed;
    }
    _pendingResults.erase(_pendingResults.begin(),
                          _pendingResults.begin() + static_cast<long>(resultsConsumed));

    // 2. Edit fast path: chunks right next to the camera re-mesh
    //    synchronously so breaking a block never shows a stale frame.
    int syncBuilds = 0;
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated || syncBuilds >= 2)
            continue;
        if (std::abs(chunk->chunkX - camChunkX) > 2 || std::abs(chunk->chunkZ - camChunkZ) > 2)
            continue;
        uint64_t key = ChunkPos{chunk->chunkX, chunk->chunkZ}.packed();
        auto it = _chunkMeshes.find(key);
        uint32_t version = chunk->version.load(std::memory_order_relaxed);
        // Only REBUILDS take the sync path (builtVersion != 0): first-time
        // builds stream through the workers like everything else
        if (it == _chunkMeshes.end() || it->second.builtVersion == 0 ||
            it->second.builtVersion == version) {
            continue;
        }
        if (!world.snapshotForMeshing(ChunkPos{chunk->chunkX, chunk->chunkZ}, _meshSnapshot)) {
            continue;
        }
        MeshOutput mesh = LODMesher::buildMesh(_meshSnapshot, _meshScratch);
        applyMesh(key, mesh, _meshSnapshot.version);
        ++syncBuilds;
        ++uploads;
    }

    // 3. Candidate scan: every generated chunk in the render radius whose
    //    mesh is missing or stale, nearest first, until the in-flight cap.
    _meshCandidates.clear();
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated)
            continue;
        if (std::abs(chunk->chunkX - camChunkX) > renderRadius ||
            std::abs(chunk->chunkZ - camChunkZ) > renderRadius)
            continue; // frontier ring: generated but not rendered
        uint64_t key = ChunkPos{chunk->chunkX, chunk->chunkZ}.packed();
        uint32_t version = chunk->version.load(std::memory_order_relaxed);
        auto it = _chunkMeshes.find(key);
        if (it != _chunkMeshes.end() &&
            (it->second.builtVersion == version || it->second.requestedVersion == version)) {
            continue; // up to date, or a build is already on its way
        }
        float dx = static_cast<float>(chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2) - camX;
        float dz = static_cast<float>(chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2) - camZ;
        _meshCandidates.push_back({dx * dx + dz * dz, chunk.get()});
    }
    std::sort(_meshCandidates.begin(), _meshCandidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [distSq, chunkPtr] : _meshCandidates) {
        ChunkPos pos{chunkPtr->chunkX, chunkPtr->chunkZ};
        if (!_meshScheduler->enqueue(pos)) {
            break; // in-flight cap reached — re-prioritized next frame
        }
        _chunkMeshes[pos.packed()].requestedVersion =
            chunkPtr->version.load(std::memory_order_relaxed);
    }

    // ---- Draw everything the registry has uploaded ----
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated)
            continue;

        // Chunk key for mesh cache lookup (packed, no allocation)
        uint64_t key = ChunkPos{chunk->chunkX, chunk->chunkZ}.packed();

        // Frustum culling
        AABB chunkAABB = chunk->getAABB();
        if (!isChunkInFrustum(chunkAABB))
            continue;

        auto cached = _chunkMeshes.find(key);
        if (cached == _chunkMeshes.end() || !cached->second.uploaded)
            continue;

        const auto& meshState = cached->second;

        // Mesh vertices are chunk-local; this restores world space (and keeps
        // fp16 positions exact regardless of how far the chunk is from origin)
        ChunkOrigin origin{};
        origin.origin = simd_make_float4(static_cast<float>(chunk->chunkX * CHUNK_WIDTH), 0.0f,
                                         static_cast<float>(chunk->chunkZ * CHUNK_DEPTH), 0.0f);

        // The chunk's water section renders after the scene resolves
        uint32_t waterIndexCount = meshState.alloc.indexCount - meshState.opaqueIndexCount;
        if (waterIndexCount > 0) {
            float dx = static_cast<float>(chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2) - camX;
            float dz = static_cast<float>(chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2) - camZ;
            _waterDraws.push_back(WaterDraw{
                origin.origin, meshState.alloc.vertexBuffer, meshState.alloc.indexBuffer,
                meshState.alloc.vertexOffset,
                meshState.alloc.indexOffset + meshState.opaqueIndexCount * sizeof(uint32_t),
                waterIndexCount, dx * dx + dz * dz});
        }

        if (meshState.opaqueIndexCount == 0)
            continue;

        [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];

        // Bind vertex buffer from MegaBuffer allocation
        [encoder setVertexBuffer:meshState.alloc.vertexBuffer
                          offset:meshState.alloc.vertexOffset
                         atIndex:0];

        // Draw indexed primitives (triangles)
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:meshState.opaqueIndexCount
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:meshState.alloc.indexBuffer
                     indexBufferOffset:meshState.alloc.indexOffset];
    }

    // F3 HUD counters: uploads applied this frame + the workers' build EMA
    _chunkStats.meshBuildsLastFrame = static_cast<uint32_t>(uploads);
    _chunkStats.meshMsAvg = _meshScheduler->meshMsAvg();
    _chunkStats.megaUsedMB = static_cast<float>(_megaBuffer->vertexUsed()) / (1024.f * 1024.f);
    _chunkStats.megaCapMB = static_cast<float>(_megaBuffer->vertexCapacity()) / (1024.f * 1024.f);
}

void RenderPipeline::shutdownMeshWorkers() {
    if (_meshScheduler) {
        _meshScheduler->shutdown();
    }
}

// ---------------------------------------------------------------------------
// renderWater — the water surfaces recorded by renderChunks, drawn into the
// resolved scene color with their own compositing (refraction from a copy
// of the scene, manual depth test against the resolved depth). Nearest
// chunks draw last so the closest surface owns the pixel. Ends with the
// underwater veil + god rays when the camera is submerged.
// ---------------------------------------------------------------------------
void RenderPipeline::renderWater(id<MTLCommandBuffer> commandBuffer, const Mat4& viewMatrix,
                                 const Mat4& projectionMatrix, const Vec3& cameraPosition,
                                 bool cameraUnderwater, const SkyUniforms& skyUniforms,
                                 const float fogColor[3], uint64_t worldTime) {
    if (_waterDraws.empty() && !cameraUnderwater)
        return;

    // Refraction input: copy the resolved scene (a render target cannot
    // sample itself)
    if (!_waterDraws.empty()) {
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (blit) {
            [blit copyFromTexture:_colorResolve
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(_colorResolve.width, _colorResolve.height, 1)
                        toTexture:_sceneColorCopy
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
        }
    }

    WaterUniforms wu{};
    simd_float4x4 view, proj;
    std::memcpy(&view, viewMatrix.data.data(), sizeof(view));
    std::memcpy(&proj, projectionMatrix.data.data(), sizeof(proj));
    wu.invViewProjection = simd_inverse(simd_mul(proj, view));
    wu.zenithColor = skyUniforms.zenithColor;
    wu.horizonColor = skyUniforms.horizonColor;
    wu.sunDirection = skyUniforms.sunDirection;
    wu.sunColor = skyUniforms.sunColor;
    wu.cameraPosition = simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    wu.fogColor = simd_make_float3(fogColor[0], fogColor[1], fogColor[2]);
    wu.resolution =
        simd_make_float2(static_cast<float>(_displayWidth), static_cast<float>(_displayHeight));
    wu.fogDensity = _fogDensity;
    // 20 Hz world time → seconds (the same stepping the clouds animate with)
    wu.time = static_cast<float>(worldTime % 24000) * 0.05f;
    wu.cameraUnderwater = cameraUnderwater ? 1.f : 0.f;
    std::memcpy((void*)_waterUniformsBuffer.contents, &wu, sizeof(WaterUniforms));

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = _colorResolve;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    _gpuTimer->attachPass(passDesc, "water");

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder)
        return;

    if (!_waterDraws.empty()) {
        // Back-to-front: the nearest surface draws last and wins the pixel
        std::sort(_waterDraws.begin(), _waterDraws.end(),
                  [](const WaterDraw& a, const WaterDraw& b) { return a.distSq > b.distSq; });

        [encoder setRenderPipelineState:_waterPipelineState];
        [encoder setCullMode:MTLCullModeNone]; // surface visible from below
        [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
        [encoder setVertexBuffer:_waterUniformsBuffer offset:0 atIndex:3];
        [encoder setFragmentBuffer:_waterUniformsBuffer offset:0 atIndex:3];
        [encoder setFragmentTexture:_sceneColorCopy atIndex:0];
        [encoder setFragmentTexture:_depthResolve atIndex:1];

        for (const WaterDraw& draw : _waterDraws) {
            ChunkOrigin origin{draw.origin};
            [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
            [encoder setVertexBuffer:draw.vertexBuffer offset:draw.vertexOffset atIndex:0];
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:draw.indexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:draw.indexBuffer
                         indexBufferOffset:draw.indexOffset];
        }
    }

    if (cameraUnderwater) {
        [encoder setRenderPipelineState:_underwaterOverlayState];
        [encoder setFragmentBuffer:_waterUniformsBuffer offset:0 atIndex:3];
        [encoder setFragmentTexture:_depthResolve atIndex:1];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }

    [encoder endEncoding];
}

// ---------------------------------------------------------------------------
// renderBlockHighlight (Task 6.9)
// ---------------------------------------------------------------------------
void RenderPipeline::renderBlockHighlight(id<MTLRenderCommandEncoder> encoder, const Vec3& blockPos,
                                          const Mat4& viewMatrix, const Mat4& projectionMatrix) {
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
void RenderPipeline::renderUIOverlay(id<MTLRenderCommandEncoder> encoder, const Hotbar& hotbar,
                                     const UIFrameState& uiFrame) {
    _uiOverlay->beginFrame();
    drawGameHud(*_uiOverlay, hotbar, uiFrame, _displayWidth, _displayHeight);
    if (uiFrame.screen != GameScreen::PLAYING) {
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
    if (width == _displayWidth && height == _displayHeight)
        return;

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
    if (!_particles)
        return;
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
        _frustumPlanes[0][c] = row(3, c) + row(0, c); // left
        _frustumPlanes[1][c] = row(3, c) - row(0, c); // right
        _frustumPlanes[2][c] = row(3, c) + row(1, c); // bottom
        _frustumPlanes[3][c] = row(3, c) - row(1, c); // top
        _frustumPlanes[4][c] = row(2, c);             // near (Metal: z' >= 0)
        _frustumPlanes[5][c] = row(3, c) - row(2, c); // far
    }

    for (int i = 0; i < 6; ++i) {
        float len = std::sqrt(_frustumPlanes[i][0] * _frustumPlanes[i][0] +
                              _frustumPlanes[i][1] * _frustumPlanes[i][1] +
                              _frustumPlanes[i][2] * _frustumPlanes[i][2]);
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
        float extent = std::abs(A) * extents.x + std::abs(B) * extents.y + std::abs(C) * extents.z;

        if (dist + extent < 0.f) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// renderClouds — alpha-blended cloud layer, drawn last in the scene pass
// ============================================================================
void RenderPipeline::renderClouds(id<MTLRenderCommandEncoder> encoder, const Camera& camera,
                                  uint64_t worldTime, const float sunDirection[3]) {
    if (!_cloudPipelineState)
        return;

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
    cloudUniforms.tanHalfFov = std::tan(camera.FOV() * 0.5f * static_cast<float>(M_PI) / 180.0f);
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
