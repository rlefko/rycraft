#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "render/block_textures.hpp"
#include "render/bloom.hpp"
#include "render/entity_renderer.hpp"
#include "render/lod_mesher.hpp"
#include "render/pixel_formats.hpp"
#include "render/post_stack.hpp"
#include "render/shadow_map.hpp"
#include "render/ssao.hpp"
#include "render/volumetrics.hpp"

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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

// One in-game day in 20 Hz ticks — shared by the animation clock, the
// morning-fog window, and the day/night cycle so they can never drift.
constexpr uint64_t TICKS_PER_DAY = 24000;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
// One frame's constants (chunk/sky/water/highlight/cloud uniform blocks plus
// the particle instance array) sub-allocate from this ring slot; the particle
// array dominates at 192 KB.
static constexpr uint64_t FRAME_RING_SLOT_BYTES = 256 * 1024;
static constexpr uint64_t FAR_VERTEX_BUFFER_BYTES = 2ull * 1024 * 1024 * 1024;
static constexpr uint64_t FAR_INDEX_BUFFER_BYTES = 1ull * 1024 * 1024 * 1024;
static constexpr uint64_t FAR_VERTEX_BUFFER_SLAB_BYTES = 256ull * 1024 * 1024;
static constexpr uint64_t FAR_INDEX_BUFFER_SLAB_BYTES = 128ull * 1024 * 1024;
static_assert(FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR == 4);
static_assert(FarTerrainExactHandoff::COLUMN_MASK_WORD_COUNT == FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
static_assert(FAR_TERRAIN_TILE_EDGE / CHUNK_EDGE == FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);

static FarTerrainOwnershipUniforms
farTerrainOwnershipUniforms(ColumnPos centerTile, const FarTerrainExactHandoff& handoff) {
    FarTerrainOwnershipUniforms ownership{};
    for (int64_t neighborZ = -FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS;
         neighborZ <= FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS; ++neighborZ) {
        for (int64_t neighborX = -FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS;
             neighborX <= FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS; ++neighborX) {
            const size_t tileIndex =
                static_cast<size_t>((neighborZ + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS) *
                                        FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE +
                                    neighborX + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS);
            const FarTerrainExactHandoff::ColumnMask mask =
                handoff.readyColumnMask({centerTile.x + neighborX, centerTile.z + neighborZ});
            for (size_t word = 0; word < mask.size(); ++word) {
                ownership.readyColumnMasks[tileIndex * FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE +
                                           word / FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR]
                                          [word % FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR] =
                    mask[word];
            }
        }
    }
    return ownership;
}

static bool
farTerrainOccluderIntersectsExact(const FarTerrainBounds& patch, ColumnPos tile,
                                  const FarTerrainExactHandoff::ColumnMask& readyColumns) {
    const int64_t tileOriginX = tile.x * FAR_TERRAIN_TILE_EDGE;
    const int64_t tileOriginZ = tile.z * FAR_TERRAIN_TILE_EDGE;
    const int64_t maximumX = std::max(patch.minX, patch.maxX - 1);
    const int64_t maximumZ = std::max(patch.minZ, patch.maxZ - 1);
    const int minimumColumnX =
        std::clamp(static_cast<int>(world_coord::floorDiv(patch.minX - tileOriginX,
                                                          static_cast<int64_t>(CHUNK_EDGE))),
                   0, FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1);
    const int maximumColumnX =
        std::clamp(static_cast<int>(world_coord::floorDiv(maximumX - tileOriginX,
                                                          static_cast<int64_t>(CHUNK_EDGE))),
                   0, FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1);
    const int minimumColumnZ =
        std::clamp(static_cast<int>(world_coord::floorDiv(patch.minZ - tileOriginZ,
                                                          static_cast<int64_t>(CHUNK_EDGE))),
                   0, FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1);
    const int maximumColumnZ =
        std::clamp(static_cast<int>(world_coord::floorDiv(maximumZ - tileOriginZ,
                                                          static_cast<int64_t>(CHUNK_EDGE))),
                   0, FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1);
    for (int columnZ = minimumColumnZ; columnZ <= maximumColumnZ; ++columnZ) {
        for (int columnX = minimumColumnX; columnX <= maximumColumnX; ++columnX) {
            const uint32_t bit =
                static_cast<uint32_t>(columnZ * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE + columnX);
            if ((readyColumns[bit / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD] &
                 (1U << (bit % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD))) != 0U)
                return true;
        }
    }
    return false;
}

RenderPipeline::RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                               uint32_t height)
    : _device(device), _frameRing(device, FRAME_RING_SLOT_BYTES), _bloomIntensity(1.0f),
      _displayWidth(width), _displayHeight(height), _frustumPlanes{} {
    if (const char* overlay = std::getenv("RYCRAFT_WORLDGEN_OVERLAY")) {
        const std::string_view name{overlay};
        if (name == "geology") {
            _worldgenOverlayMode = WorldgenOverlayMode::GEOLOGY;
        } else if (name == "hydrology") {
            _worldgenOverlayMode = WorldgenOverlayMode::HYDROLOGY;
        } else if (name == "climate") {
            _worldgenOverlayMode = WorldgenOverlayMode::CLIMATE;
        } else if (name == "biome") {
            _worldgenOverlayMode = WorldgenOverlayMode::BIOME;
        } else if (!name.empty()) {
            RY_LOG_ERROR("RYCRAFT_WORLDGEN_OVERLAY must be geology, hydrology, climate, or biome");
        }
    }
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

    pipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    pipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
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
    skyPipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    skyPipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
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

    // ---- Block highlight pipeline state (lines) ----
    id<MTLFunction> highlightVertexFunc = [shaderLibrary newFunctionWithName:@"vertexMain"];
    id<MTLFunction> highlightFragmentFunc = [shaderLibrary newFunctionWithName:@"fragmentMain"];

    auto highlightPipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    highlightPipelineDesc.vertexFunction = highlightVertexFunc;
    highlightPipelineDesc.fragmentFunction = highlightFragmentFunc;
    highlightPipelineDesc.vertexDescriptor = vertexDesc;
    highlightPipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    highlightPipelineDesc.colorAttachments[0].blendingEnabled = true;
    highlightPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    highlightPipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
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

    // ---- Scene render targets (native resolution) ----
    allocateSceneTargets();

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
        waterDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
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
        overlayDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
        overlayDesc.colorAttachments[0].blendingEnabled = true;
        overlayDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        overlayDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        // Dual-source blending: result = inscatter + scene * transmit. The
        // fragment's color(0) index(0) is the inscattered light and index(1)
        // the per-channel Beer-Lambert transmittance — a single alpha cannot
        // express spectral absorption (red must die faster than blue).
        overlayDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        overlayDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        overlayDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorSource1Color;
        overlayDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
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
        cloudPipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
        cloudPipelineDesc.colorAttachments[0].blendingEnabled = true;
        cloudPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        cloudPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        cloudPipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
            MTLBlendFactorOneMinusSourceAlpha;
        cloudPipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
        cloudPipelineDesc.rasterSampleCount = 4;

        _cloudPipelineState = [_device newRenderPipelineStateWithDescriptor:cloudPipelineDesc
                                                                      error:&error];
    }

    // Cloud depth state (depth test, no write)
    auto cloudDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    cloudDepthDesc.depthCompareFunction = MTLCompareFunctionLess;
    cloudDepthDesc.depthWriteEnabled = false;
    _cloudDepthState = [_device newDepthStencilStateWithDescriptor:cloudDepthDesc];

    // ---- Bloom post-processing (HDR extract + blur) ----
    _bloom = std::make_unique<Bloom>(_device, shaderLibrary, _displayWidth, _displayHeight);
    _bloom->setIntensity(_bloomIntensity);

    // ---- Final composite (tonemap + grade + sharpen) ----
    _postStack = std::make_unique<PostStack>(_device, shaderLibrary);

    // ---- Cascaded shadow maps (share the chunk vertex layout) ----
    _shadowMap = std::make_unique<ShadowMap>(_device, shaderLibrary, vertexDesc);

    // ---- Volumetric light shafts ----
    _volumetrics =
        std::make_unique<Volumetrics>(_device, shaderLibrary, _displayWidth, _displayHeight);

    // ---- Screen-space ambient occlusion ----
    _ssao = std::make_unique<Ssao>(_device, shaderLibrary, _displayWidth, _displayHeight);

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
    colorMSAADesc.pixelFormat = PixelFormats::SCENE_HDR;
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
    depthMSAADesc.pixelFormat = PixelFormats::SCENE_DEPTH;
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
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SCENE_HDR
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
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SCENE_DEPTH
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
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SCENE_HDR
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
                            double deltaSeconds, std::optional<Vec3> highlightedBlock,
                            const Hotbar& hotbar, const UIFrameState& uiFrame,
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

    // Compute day/night uniforms. sunDirection/sunColor come back as the
    // active directional light (sun by day, moon by night); shadowStrength
    // is its cascade weight (0 at the horizon crossing).
    float sunDirection[3] = {0.5f, 0.8f, 0.3f};
    float sunColor[3] = {1.0f, 0.95f, 0.9f};
    float ambientColor[3] = {0.35f, 0.35f, 0.4f};
    float shadowStrength = 0.0f;
    SkyUniforms skyUniforms{};
    computeDayNightUniforms(worldTime, sunDirection, sunColor, ambientColor, skyUniforms,
                            shadowStrength);

    // Animation clock accumulates the real frame delta (the engine already
    // clamps it to <= 0.25 s past a hitch/pause), NOT the day-night worldTime,
    // so water/caustics/foliage keep flowing even when the time of day is frozen
    // (captures) or paused and never jump at the daily rollover. It wraps at
    // 3600 s so the float stays sub-millisecond precise.
    if (deltaSeconds > 0.0) {
        _animClock = std::fmod(_animClock + deltaSeconds, 3600.0);
    }
    _animTime = static_cast<float>(_animClock);

    // Normalize sun direction
    float sunLen = std::sqrt(sunDirection[0] * sunDirection[0] + sunDirection[1] * sunDirection[1] +
                             sunDirection[2] * sunDirection[2]);
    if (sunLen > 0.001f) {
        sunDirection[0] /= sunLen;
        sunDirection[1] /= sunLen;
        sunDirection[2] /= sunLen;
    }

    // Fill the sky's per-pixel-ray inputs (the sun/moon are true
    // direction-projected discs now, so the atmosphere needs the camera
    // basis — computeDayNightUniforms only knows the time of day).
    {
        Vec3 camFwd = camera.forward();
        Vec3 camRight = camera.right();
        Vec3 camUp = camera.up();
        skyUniforms.cameraForward = simd_make_float3(camFwd.x, camFwd.y, camFwd.z);
        skyUniforms.cameraRight = simd_make_float3(camRight.x, camRight.y, camRight.z);
        skyUniforms.cameraUp = simd_make_float3(camUp.x, camUp.y, camUp.z);
        skyUniforms.tanHalfFov = std::tan(camera.FOV() * 0.5f * static_cast<float>(M_PI) / 180.0f);
        skyUniforms.aspect =
            static_cast<float>(_displayWidth) / static_cast<float>(std::max(_displayHeight, 1u));
    }

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer)
        return;

    _gpuTimer->beginFrame();

    // Claim a frames-in-flight slot: every per-frame uniform block below
    // sub-allocates from it, so the CPU never rewrites data the GPU reads.
    _frameRing.waitAndBegin();
    FrameRing::Alloc skyAlloc = _frameRing.push(&skyUniforms, sizeof(SkyUniforms));

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

    // One immutable tick snapshot feeds shadows, exact terrain, and entities.
    // The render thread never copies or locks the cubic chunk map.
    const auto loadedSnapshot = world.getLoadedSnapshot();
    static const std::vector<std::shared_ptr<Chunk>> emptyChunks;
    const auto& loadedChunks = loadedSnapshot ? *loadedSnapshot : emptyChunks;

    // ---- Shadow cascades (depth-only passes before the scene pass) ----
    renderShadows(commandBuffer, loadedChunks, camera, sunDirection, shadowStrength);

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        _frameRing.cancelFrame(); // nothing encoded references the slot
        return;
    }

    renderSky(encoder, skyAlloc);

    // Underwater the whole scene sinks into a dense blue veil (light
    // attenuation) — owned entirely by the underwater overlay's depth-based
    // scattering, so the scene/water passes apply no fog of their own below the
    // surface (two fogs stacked over-darkened the near water).
    const bool cameraUnderwater = uiFrame.cameraUnderwater;
    _cameraUnderwater = cameraUnderwater;

    // Sky exposure of the camera's water column: 0 when solid ground seals it
    // (aquifers, roofed lakes — the same surface-height gate rain spawning
    // uses). Sunlight cannot reach covered water, so the underwater caustics,
    // sun-driven murk, and volumetric shafts all scale by this. Eased so
    // swimming under an overhang lip fades rather than pops.
    {
        float target = 1.0f;
        if (cameraUnderwater) {
            const Vec3 camPos = camera.getPosition();
            auto surface = world.surfaceHeightIfLoaded(static_cast<int64_t>(std::floor(camPos.x)),
                                                       static_cast<int64_t>(std::floor(camPos.z)));
            if (surface.has_value() && static_cast<double>(*surface) > camPos.y) {
                target = 0.0f;
            }
            // Scan up for the top of the water body the camera is in: upward
            // rays exit the water there, so murk and caustics must stop at
            // that height instead of fogging out to the opaque depth behind
            // the from-below surface. 0.875 is the rendered surface plane.
            const int64_t bx = static_cast<int64_t>(std::floor(camPos.x));
            const int64_t bz = static_cast<int64_t>(std::floor(camPos.z));
            int32_t top = static_cast<int32_t>(std::floor(camPos.y));
            while (world.getBlockIfLoaded(bx, top + 1, bz) == BlockType::WATER) {
                ++top;
            }
            _uwSurfaceY = static_cast<float>(top) + 0.875f;
        }
        _uwSkyExposure += (target - _uwSkyExposure) * 0.1f;
    }

    const float fogColor[3] = {skyUniforms.horizonColor.x, skyUniforms.horizonColor.y,
                               skyUniforms.horizonColor.z};
    const float savedFogDensity = _fogDensity;
    {
        // Morning fog: a dawn haze that thickens fog and burns off by
        // mid-morning (worldTime 0 is sunrise; peak just after, gone ~2500).
        float t = static_cast<float>(worldTime % TICKS_PER_DAY);
        float fromDawn = std::min(std::abs(t - 700.0f), 24000.0f - std::abs(t - 700.0f));
        float morning = std::max(0.0f, 1.0f - fromDawn / 1800.0f);
        _fogDensity *= 1.0f + 5.0f * morning * morning;
    }
    if (cameraUnderwater) {
        _fogDensity = 0.0f; // the overlay owns the underwater murk
    }
    renderChunks(encoder, world, loadedChunks, viewMatrix, projectionMatrix, camera.getPosition(),
                 sunDirection, sunColor, ambientColor, fogColor);

    if (entities && _entityRenderer) {
        _entityRenderer->render(encoder, _frameUniforms.buffer, _frameUniforms.offset, *entities,
                                [this](const AABB& aabb) { return isChunkInFrustum(aabb); });
    }

    if (highlightedBlock.has_value()) {
        renderBlockHighlight(encoder, highlightedBlock.value(), viewMatrix, projectionMatrix);
    }

    if (_particles) {
        _particles->render(encoder, _frameRing, viewMatrix, projectionMatrix, camera.getPosition());
    }

    if (_gfx.cloudMode != 0) {
        renderClouds(encoder, camera, worldTime, sunDirection, skyUniforms.sunIntensity);
    }

    [encoder endEncoding];

    // ---- Screen-space ambient occlusion (darkens creases/caves) ----
    // Applied to the resolved opaque scene *before* water and volumetrics so it
    // only darkens opaque ambient — never the translucent water surface or the
    // additive light shafts, which AO (an opaque-ambient term) must not touch.
    if (_gfx.ssao) {
        SsaoUniforms su{};
        std::memcpy(&su.projection, projectionMatrix.data.data(), sizeof(su.projection));
        su.invProjection = simd_inverse(su.projection);
        su.resolution = _ssao->resolution();
        su.radius = 0.5f;
        su.strength = 0.6f;
        su.bias = 0.06f; // grazing ground has a steep depth slope; a bigger
                         // bias keeps flat surfaces from self-occluding (streaks)
        su.frameIndex = static_cast<uint32_t>(_frameRing.frameIndex());
        _ssao->encode(commandBuffer, _colorResolve, _depthResolve, su);
    }

    // ---- Water pass (refraction/reflection/caustics over the resolved scene) ----
    renderWater(commandBuffer, viewMatrix, projectionMatrix, camera.getPosition(), cameraUnderwater,
                skyUniforms, fogColor);
    _fogDensity = savedFogDensity;

    // ---- Volumetric light shafts (over the resolved opaque + water) ----
    // Needs the shadow cascades: when shadows are off their matrices are zero
    // and the march would divide by zero, so gate on shadowQuality too.
    if (_gfx.volumetricLight && _gfx.shadowQuality > 0 && shadowStrength > 0.001f) {
        VolumetricUniforms vu{};
        std::memcpy(&vu.invViewProjection, &vpMatrix, sizeof(vu.invViewProjection));
        vu.invViewProjection = simd_inverse(vu.invViewProjection);
        Vec3 camPos = camera.getPosition();
        vu.cameraPosition = simd_make_float3(camPos.x, camPos.y, camPos.z);
        vu.sunDirection = simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
        vu.sunColor = simd_make_float3(sunColor[0], sunColor[1], sunColor[2]);
        vu.stepCount = 24.0f;
        // Covered water (aquifers, roofed lakes) receives no sunlight: the
        // cascades cannot occlude terrain hundreds of blocks up, so without
        // this gate sealed pockets grew impossible sun shafts.
        vu.density = 0.055f * (cameraUnderwater ? _uwSkyExposure : 1.0f);
        vu.anisotropy = 0.6f; // forward scatter → bright halo toward the light
        vu.maxDistance = 96.0f;
        vu.underwater = cameraUnderwater ? 1.0f : 0.0f;
        vu.frameIndex = static_cast<uint32_t>(_frameRing.frameIndex());
        _volumetrics->encode(commandBuffer, _colorResolve, _depthResolve,
                             _shadowMap->depthTexture(), _shadowMap->comparisonSampler(), vu,
                             _sceneShadowUniforms);
    }

    // ---- Auto-exposure (measure the finished HDR scene, ease adaptation) ----
    _postStack->encodeExposure(commandBuffer, _colorResolve);

    // ---- Bloom (HDR extract + blur; result feeds the composite) ----
    // At zero intensity renderBloom early-outs and the composite binds a
    // black fallback, so no bloom texture is produced.
    const bool bloomOn = _bloom && _bloomIntensity > 0.0f;
    if (bloomOn) {
        _bloom->renderBloom(commandBuffer, _colorResolve);
    }

    // ---- Lens flare: project the sun to screen and probe its occlusion ----
    float flareStrength = 0.0f;
    simd_float2 sunUV = simd_make_float2(0.5f, 0.5f);
    if (_gfx.lensFlare) {
        // A direction (w=0) projects to the point at infinity where the sun disc
        // sits; transformVec4 is the same column-major M*v the shaders use.
        Vec4 sunClip =
            vpMatrix.transformVec4({skyUniforms.sunDirection.x, skyUniforms.sunDirection.y,
                                    skyUniforms.sunDirection.z, 0.0f});
        if (sunClip.w > 1e-4f) { // sun in front of the camera
            sunUV = simd_make_float2((sunClip.x / sunClip.w) * 0.5f + 0.5f,
                                     0.5f - (sunClip.y / sunClip.w) * 0.5f);
            // Fade out as the sun crosses the screen edge — the probe has no
            // depth data outside the frame, and a hard cut would pop.
            float edge =
                std::min(std::min(sunUV.x, 1.0f - sunUV.x), std::min(sunUV.y, 1.0f - sunUV.y));
            float fade = std::clamp((edge + 0.1f) / 0.2f, 0.0f, 1.0f);
            flareStrength = skyUniforms.sunIntensity * fade;
        }
    }
    if (flareStrength > 0.0f) {
        _postStack->encodeFlareProbe(commandBuffer, _depthResolve, sunUV);
    }

    // ---- Final composite (always runs: tonemap + grade + sharpen) ----
    // This is the one linear-HDR → display conversion; the pre-HDR pipeline
    // blitted raw when bloom was off, so that path was never tonemapped.
    _postStack->encodeComposite(
        commandBuffer, _colorResolve, bloomOn ? _bloom->bloomTexture() : nil, drawable.texture,
        _gfx, static_cast<uint32_t>(_frameRing.frameIndex()), flareStrength, sunUV);

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
    _frameRing.signalOnCompletion(commandBuffer);

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
                                             SkyUniforms& skyUniforms, float& shadowStrength) {
    // Full day = 24000 ticks (20 minutes real time at 20Hz)

    // Orbital angle: 0 = dawn, PI/2 = noon, PI = dusk, 3PI/2 = midnight
    float dayFraction =
        static_cast<float>(worldTime % TICKS_PER_DAY) / static_cast<float>(TICKS_PER_DAY);
    float orbitalAngle = dayFraction * 2.0f * static_cast<float>(M_PI);

    // Real sun position (slight Z tilt for depth); the sky's sun disc uses
    // this. sunElevation: 1 at noon, -1 at midnight. The moon rides opposite.
    float realSun[3] = {std::cos(orbitalAngle), std::sin(orbitalAngle), 0.3f};
    float sunElevation = std::sin(orbitalAngle);
    float realMoon[3] = {-realSun[0], -realSun[1], 0.3f};

    auto blend3 = [](const float a[3], const float b[3], float t, float out[3]) {
        for (int i = 0; i < 3; ++i) {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
    };

    // ---- Sun lit color: white at noon, orange at sunrise/sunset ----
    float sunColorDay[3] = {1.0f, 0.95f + 0.05f * std::clamp(sunElevation, 0.0f, 1.0f),
                            0.9f + 0.1f * std::clamp(sunElevation, 0.0f, 1.0f)};
    float sunColorSunset[3] = {1.0f, 0.5f, 0.2f};
    float dayBlend = std::clamp(sunElevation / 0.35f, 0.0f, 1.0f);
    float nightBlend = std::clamp((-sunElevation - 0.05f) / 0.30f, 0.0f, 1.0f);
    float sunLit[3];
    blend3(sunColorSunset, sunColorDay, dayBlend, sunLit);

    // ---- Ambient: bright at noon, dim (never black) at night so caves and
    // night surfaces keep a floor of sky/moon bounce ----
    float ambientDay[3] = {0.35f, 0.35f, 0.4f};
    float ambientNight[3] = {0.1f, 0.1f, 0.15f};
    float ambientT = std::clamp((sunElevation + 0.2f) / 0.6f, 0.0f, 1.0f);
    blend3(ambientNight, ambientDay, ambientT, ambientColor);

    // ---- Sky palette (also the fog color) ----
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

    skyUniforms.zenithColor = simd_make_float3(zenith[0], zenith[1], zenith[2]);
    skyUniforms.horizonColor = simd_make_float3(horizon[0], horizon[1], horizon[2]);
    skyUniforms.sunDirection = simd_make_float3(realSun[0], realSun[1], realSun[2]);
    skyUniforms.moonDirection = simd_make_float3(realMoon[0], realMoon[1], realMoon[2]);
    skyUniforms.sunColor = simd_make_float3(sunLit[0], sunLit[1], sunLit[2]);
    skyUniforms.sunIntensity = std::max(0.0f, sunElevation);
    skyUniforms.starStrength = std::clamp(-sunElevation / 0.2f, 0.0f, 1.0f);

    // ---- Active directional light for terrain + shadows ----
    // Sun while above the horizon, moon below it. Each fades to nothing at the
    // horizon crossing (grazing light is near-zero there anyway), so the
    // discrete direction swap lands where the term is invisible — no pop. The
    // moon is a dim cool light so nights stay navigable, not pitch black.
    float sunFade = std::clamp(sunElevation / 0.10f, 0.0f, 1.0f);
    float moonFade = std::clamp(-sunElevation / 0.15f, 0.0f, 1.0f);
    const float moonlight[3] = {0.16f, 0.20f, 0.38f};

    if (sunElevation >= 0.0f) {
        for (int i = 0; i < 3; ++i) {
            sunDirection[i] = realSun[i];
            sunColor[i] = sunLit[i] * sunFade;
        }
        shadowStrength = 0.85f * sunFade;
    } else {
        for (int i = 0; i < 3; ++i) {
            sunDirection[i] = realMoon[i];
            sunColor[i] = moonlight[i] * moonFade;
        }
        shadowStrength = 0.30f * moonFade; // moon shadows are soft and faint
    }
}

// ---------------------------------------------------------------------------
// renderSky — fullscreen gradient drawn first in the scene pass
// ---------------------------------------------------------------------------
void RenderPipeline::renderSky(id<MTLRenderCommandEncoder> encoder,
                               const FrameRing::Alloc& skyUniforms) {
    [encoder setRenderPipelineState:_skyPipelineState];
    [encoder setDepthStencilState:_skyDepthState];
    [encoder setVertexBuffer:skyUniforms.buffer offset:skyUniforms.offset atIndex:1];
    [encoder setFragmentBuffer:skyUniforms.buffer offset:skyUniforms.offset atIndex:1];

    // Draw fullscreen quad (6 vertices, no index buffer)
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

// ---------------------------------------------------------------------------
// renderShadows — depth-only cascade passes. Uses last frame's uploaded mesh
// registry (a streaming chunk casting a shadow one frame late is invisible),
// so it can run before renderChunks rebuilds it. A no-op at shadowQuality 0 or
// zero strength (the horizon crossing), where _sceneShadowUniforms carries
// strength 0 and the chunk fragment falls back to the baked skylight. The
// active light (sun by day, moon by night) and its strength come from
// computeDayNightUniforms. (Entities don't cast yet — small animals.)
// ---------------------------------------------------------------------------
void RenderPipeline::renderShadows(id<MTLCommandBuffer> commandBuffer,
                                   const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                                   const Camera& camera, const float lightDirection[3],
                                   float strength) {
    if (_gfx.shadowQuality == 0 || strength <= 0.001f) {
        _sceneShadowUniforms = ShadowUniforms{}; // strength 0 → chunk reads full sun
        return;
    }

    _shadowMap->setResolution(_gfx.shadowQuality >= 2 ? 2048 : 1536);

    Vec3 lightDir{lightDirection[0], lightDirection[1], lightDirection[2]};
    constexpr float SHADOW_DISTANCE = 160.0f;
    _shadowMap->computeCascades(camera.getPosition(), camera.forward(), camera.right(), camera.up(),
                                camera.FOV() * static_cast<float>(M_PI) / 180.0f,
                                static_cast<float>(_displayWidth) /
                                    static_cast<float>(std::max(_displayHeight, 1u)),
                                lightDir, SHADOW_DISTANCE, strength);
    _sceneShadowUniforms = _shadowMap->shadowUniforms();

    for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        MTLRenderPassDescriptor* passDesc = _shadowMap->passDescriptor(cascade);
        _gpuTimer->attachPass(passDesc, "shadow");
        id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
        if (!encoder)
            continue;

        [encoder setRenderPipelineState:_shadowMap->chunkPipeline()];
        [encoder setDepthStencilState:_shadowMap->depthState()];
        [encoder setCullMode:MTLCullModeNone]; // greedy meshes are single-sided
        // Slope-scaled depth bias fights acne on faces near-parallel to the sun.
        // The clamp caps the slope term: vertical flora quads have near-infinite
        // light-space slope, so they always land ON the clamp — at 0.005 NDC
        // (~0.7 blocks along the light) stems sank into the ground, detaching
        // every flower's shadow from its base and erasing thin grass shadows
        // entirely. Cascade 0 (where that contact detail is visible) gets a
        // 10x tighter clamp and leans on the receiver normal offset for acne;
        // the far cascades keep the wide clamp — their NDC unit spans several
        // blocks, so a tight clamp reintroduces acne at low sun while a
        // ~1-block caster offset is invisible at 20+ blocks away.
        [encoder setDepthBias:1.0f slopeScale:2.5f clamp:(cascade == 0 ? 0.0005f : 0.005f)];
        [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
        [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

        ShadowPassUniforms passUniforms{};
        std::memcpy(&passUniforms.lightViewProj, _shadowMap->cascadeViewProj(cascade).data.data(),
                    sizeof(float) * 16);
        passUniforms.time = _animTime;
        passUniforms.swayStrength = _gfx.wavingFoliage ? 1.0f : 0.0f;
        [encoder setVertexBytes:&passUniforms length:sizeof(passUniforms) atIndex:1];

        for (auto& chunk : loadedChunks) {
            if (!chunk || !chunk->generated)
                continue;
            const ChunkPos key = chunk->pos();
            auto cached = _chunkMeshes.find(key);
            if (cached == _chunkMeshes.end() || !cached->second.uploaded)
                continue;
            const auto& meshState = cached->second;
            if (meshState.opaqueIndexCount == 0)
                continue;
            if (!_shadowMap->cascadeContains(cascade, chunk->getAABB()))
                continue;

            ChunkOrigin origin{};
            origin.origin = simd_make_float4(static_cast<float>(chunk->chunkX * CHUNK_WIDTH),
                                             static_cast<float>(chunk->chunkY * CHUNK_HEIGHT),
                                             static_cast<float>(chunk->chunkZ * CHUNK_DEPTH), 0.0f);
            [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
            [encoder setVertexBuffer:meshState.alloc.vertexBuffer
                              offset:meshState.alloc.vertexOffset
                             atIndex:0];
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:meshState.opaqueIndexCount
                                 indexType:MTLIndexTypeUInt32
                               indexBuffer:meshState.alloc.indexBuffer
                         indexBufferOffset:meshState.alloc.indexOffset];
        }
        [encoder endEncoding];
    }
}

// ---------------------------------------------------------------------------
// renderChunks (opaque pass)
// ---------------------------------------------------------------------------
void RenderPipeline::renderChunks(id<MTLRenderCommandEncoder> encoder, const World& world,
                                  const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                                  const Mat4& viewMatrix, const Mat4& projectionMatrix,
                                  const Vec3& cameraPosition, const float sunDirection[3],
                                  const float sunColor[3], const float ambientColor[3],
                                  const float fogColor[3]) {
    // Bind pipeline state
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];
    // Opaque cube faces use outward CCW winding. Flora explicitly carries
    // reverse indices, so it remains two-sided without disabling culling for
    // the much larger terrain surface.
    [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [encoder setCullMode:MTLCullModeBack];

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

    // Foliage sway clock + the waving toggle (0 freezes blades at rest)
    uniforms.time = _animTime;
    uniforms.swayStrength = _gfx.wavingFoliage ? 1.0f : 0.0f;
    uniforms.wetness = _cameraUnderwater ? 0.0f : _wetness;

    // Upload to GPU (kept for the entity renderer + water vertex stage too)
    _frameUniforms = _frameRing.push(&uniforms, sizeof(Uniforms));

    // Bind the shared atlas + uniforms once; every chunk draw reuses them
    [encoder setVertexBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
    [encoder setFragmentBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

    // Shadow sampling: the cascade array + comparison sampler + uniforms. When
    // shadows are off the depth array still binds (validation requires a
    // texture at the slot) but strength 0 keeps the fragment fully lit.
    FrameRing::Alloc shadowAlloc = _frameRing.push(&_sceneShadowUniforms, sizeof(ShadowUniforms));
    [encoder setFragmentTexture:_shadowMap->depthTexture() atIndex:1];
    [encoder setFragmentSamplerState:_shadowMap->comparisonSampler() atIndex:1];
    [encoder setFragmentBuffer:shadowAlloc.buffer offset:shadowAlloc.offset atIndex:4];
    const FarTerrainOwnershipUniforms noFarOwnership{};
    [encoder setFragmentBytes:&noFarOwnership length:sizeof(noFarOwnership) atIndex:5];

    // Reset seed-owned far state before exact ownership is accumulated. The
    // second call in renderFarTerrain is then an inexpensive no-op.
    resetFarTerrain(world.getSeed());

    // Water draws recorded here render later, in the dedicated water pass
    _waterDraws.clear();

    // Builds only happen within the render radius: the generation radius is
    // one chunk wider, so every meshable chunk has generated neighbors for
    // its snapshot (frontier chunks simply wait their turn).
    const int64_t camChunkX = Chunk::worldToChunk(static_cast<int64_t>(std::floor(camX)));
    const int32_t camChunkY = Chunk::worldToChunkY(static_cast<int32_t>(std::floor(camY)));
    const int64_t camChunkZ = Chunk::worldToChunk(static_cast<int64_t>(std::floor(camZ)));
    const int renderRadius = world.getExactViewDistance();
    const auto meshCandidateSnapshot = world.getMeshCandidateSnapshot();
    const auto shouldMesh = [&](ChunkPos pos) {
        return meshCandidateSnapshot && meshCandidateSnapshot->contains(pos);
    };

    // Recycle regions whose last GPU reader has finished, then sweep mesh
    // allocations of chunks the world has since unloaded — freed space can
    // serve this frame's builds once its deferral window closes.
    _megaBuffer->drainDeferredFrees(_frameRing.completedFrame());
    _liveChunksByPosition.clear();
    for (const auto& chunk : loadedChunks) {
        if (chunk) {
            _liveChunksByPosition.insert_or_assign(chunk->pos(), chunk.get());
        }
    }
    for (auto it = _chunkMeshes.begin(); it != _chunkMeshes.end();) {
        if (!_liveChunksByPosition.contains(it->first) || !shouldMesh(it->first)) {
            // Equal-priority candidates remain selected by the world's
            // sticky cap policy. Reaching this branch therefore means the
            // section left the live set or yielded to higher-priority work.
            setExactSectionOwned(it->first, false);
            if (it->second.uploaded) {
                _megaBuffer->deferFree(it->second.alloc, _frameRing.frameIndex());
            }
            it = _chunkMeshes.erase(it);
        } else {
            ++it;
        }
    }

    // ---- Async meshing: workers build, the render thread only uploads ----
    if (!_meshScheduler) {
        _meshScheduler = std::make_unique<MeshScheduler>(world, EXACT_MESH_WORKER_COUNT);
    }

    // ---- MegaBuffer sized once for the selected exact radius ----
    // Surface columns retain roughly 4.5 exposed vertical sections. Base the
    // allocation on that stable target instead of the currently loaded count,
    // which grows every frame during streaming and would repeatedly discard
    // the whole registry. The 512-chunk horizon does not enter this estimate.
    {
        constexpr double PI = 3.14159265358979323846;
        const double estimatedColumns =
            PI * static_cast<double>((renderRadius + 2) * (renderRadius + 2));
        const uint64_t visibleChunks = std::min<uint64_t>(
            MAX_MESH_RESIDENT_CUBES, static_cast<uint64_t>(std::ceil(estimatedColumns * 4.5)));
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
            clearExactSectionOwnership();
        }
    }

    // Upload one finished mesh into the registry. Returns false on a
    // transient MegaBuffer-full failure (builtVersion stays 0, so the chunk
    // re-requests once space frees up).
    bool allocFailureLogged = false;
    auto hasUnownedMeshVictim = [&] {
        // Sweeps and resets clear ownership, and explicit eviction skips
        // exact owners, so the ownership set remains a registry subset.
        return _exactOwnedSections.size() < _chunkMeshes.size();
    };
    auto canMakeMeshSlot = [&](ChunkPos key) {
        return chunkMeshRegistryCanAdmit(
            _chunkMeshes.size(), MAX_MESH_RESIDENT_CUBES, _chunkMeshes.contains(key),
            _chunkMeshes.size() >= MAX_MESH_RESIDENT_CUBES && hasUnownedMeshVictim());
    };
    auto makeMeshSlot = [&](ChunkPos key) -> ChunkMeshState* {
        auto existing = _chunkMeshes.find(key);
        if (existing != _chunkMeshes.end())
            return &existing->second;
        if (_chunkMeshes.size() >= MAX_MESH_RESIDENT_CUBES) {
            auto victim = _chunkMeshes.end();
            float farthestDistance = -1.0f;
            for (auto it = _chunkMeshes.begin(); it != _chunkMeshes.end(); ++it) {
                if (_exactOwnedSections.contains(it->first))
                    continue;
                const float dx =
                    static_cast<float>(it->first.x * CHUNK_EDGE + CHUNK_EDGE / 2) - camX;
                const float dy =
                    static_cast<float>(it->first.y * CHUNK_EDGE + CHUNK_EDGE / 2) - camY;
                const float dz =
                    static_cast<float>(it->first.z * CHUNK_EDGE + CHUNK_EDGE / 2) - camZ;
                const float distance = dx * dx + dy * dy + dz * dz;
                if (distance > farthestDistance) {
                    farthestDistance = distance;
                    victim = it;
                }
            }
            if (victim == _chunkMeshes.end())
                return nullptr;
            // Reuse the victim node so insertion at the hard cap cannot fail
            // after discarding a live mesh.
            auto node = _chunkMeshes.extract(victim);
            if (node.mapped().uploaded) {
                _megaBuffer->deferFree(node.mapped().alloc, _frameRing.frameIndex());
            }
            node.key() = key;
            node.mapped() = ChunkMeshState{};
            return &_chunkMeshes.insert(std::move(node)).position->second;
        }
        if (_chunkMeshes.size() >= MAX_MESH_RESIDENT_CUBES)
            return nullptr;
        return &_chunkMeshes.try_emplace(key).first->second;
    };
    auto applyMesh = [&](ChunkPos key, const MeshOutput& mesh, uint32_t builtVersion,
                         uint32_t completedRequestVersion) -> bool {
        if (!canMakeMeshSlot(key))
            return false;
        if (mesh.vertices.empty()) {
            ChunkMeshState* state = makeMeshSlot(key);
            if (!state)
                return false;
            state->requestedVersion =
                chunkMeshRequestAfterCompletion(state->requestedVersion, completedRequestVersion);
            if (state->uploaded) {
                _megaBuffer->deferFree(state->alloc, _frameRing.frameIndex());
                state->uploaded = false;
            }
            state->opaqueIndexCount = 0;
            state->builtVersion = builtVersion; // all-air: nothing to draw
            return true;
        }
        std::optional<MegaBuffer::ChunkAllocation> replacement;
        try {
            replacement = _megaBuffer->allocate(static_cast<uint32_t>(mesh.vertices.size()),
                                                static_cast<uint32_t>(mesh.indices.size()));
            _megaBuffer->uploadVertices(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
                                        replacement->vertexOffset);
            _megaBuffer->uploadIndices(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
                                       replacement->indexOffset);
            // Do not evict a resident entry until allocation and upload have
            // succeeded. At the cap makeMeshSlot reuses the victim map node.
            ChunkMeshState* state = makeMeshSlot(key);
            if (!state) {
                _megaBuffer->free(*replacement);
                return false;
            }
            state->requestedVersion =
                chunkMeshRequestAfterCompletion(state->requestedVersion, completedRequestVersion);
            if (state->uploaded) {
                _megaBuffer->deferFree(state->alloc, _frameRing.frameIndex());
            }
            state->alloc = *replacement;
            state->opaqueIndexCount = mesh.opaqueIndexCount;
            state->uploaded = true;
            state->builtVersion = builtVersion;
            return true;
        } catch (const std::exception& e) {
            if (replacement)
                _megaBuffer->free(*replacement);
            if (!allocFailureLogged) {
                RY_LOG_ERROR((std::string("Chunk mesh upload failed: ") + e.what()).c_str());
                allocFailureLogged = true;
            }
            return false;
        }
    };

    // 1. Drain worker results and upload within the per-frame budget; the
    //    leftovers stay in _pendingResults for next frame.
    _meshScheduler->drainCompleted(_pendingResults);
    constexpr int MAX_MESH_UPLOADS_PER_FRAME = 64;
    constexpr size_t MAX_UPLOAD_BYTES_PER_FRAME = 32 * 1024 * 1024;
    constexpr int MAX_ASYNC_UPLOADS_PER_FRAME = MAX_MESH_UPLOADS_PER_FRAME - 2;
    constexpr size_t MAX_ASYNC_UPLOAD_BYTES_PER_FRAME =
        MAX_UPLOAD_BYTES_PER_FRAME - 4 * 1024 * 1024;
    int uploads = 0;
    size_t uploadBytes = 0;
    size_t resultsConsumed = 0;
    for (MeshResult& result : _pendingResults) {
        if (uploads >= MAX_ASYNC_UPLOADS_PER_FRAME ||
            uploadBytes >= MAX_ASYNC_UPLOAD_BYTES_PER_FRAME) {
            break;
        }
        ChunkPos key = result.pos;
        const auto live = _liveChunksByPosition.find(key);
        if (live == _liveChunksByPosition.end()) {
            ++resultsConsumed; // chunk unloaded while meshing — drop
            continue;
        }
        if (!shouldMesh(key)) {
            ++resultsConsumed;
            continue;
        }
        if (!result.snapshotOk) {
            // A snapshot prerequisite was unavailable. Clear only the request
            // that produced this result so a later revision stays in flight.
            auto it = _chunkMeshes.find(key);
            if (it != _chunkMeshes.end()) {
                it->second.requestedVersion = chunkMeshRequestAfterCompletion(
                    it->second.requestedVersion, result.requestedVersion);
            }
            ++resultsConsumed;
            continue;
        }
        auto resident = _chunkMeshes.find(key);
        const uint32_t residentVersion =
            resident == _chunkMeshes.end() ? 0 : resident->second.builtVersion;
        const uint32_t liveVersion = live->second->version.load(std::memory_order_relaxed);
        if (!chunkMeshAsyncResultCanReplace(result.builtVersion, liveVersion, residentVersion)) {
            if (resident != _chunkMeshes.end()) {
                resident->second.requestedVersion = chunkMeshRequestAfterCompletion(
                    resident->second.requestedVersion, result.requestedVersion);
            }
            ++resultsConsumed;
            continue;
        }
        const size_t bytes = result.mesh.vertices.size() * sizeof(Vertex) +
                             result.mesh.indices.size() * sizeof(uint32_t);
        if (uploadBytes + bytes > MAX_ASYNC_UPLOAD_BYTES_PER_FRAME)
            break;
        if (!applyMesh(key, result.mesh, result.builtVersion, result.requestedVersion)) {
            break; // MegaBuffer full: retry this result next frame
        }
        ++uploads;
        uploadBytes += bytes;
        ++resultsConsumed;
    }
    _pendingResults.erase(_pendingResults.begin(),
                          _pendingResults.begin() + static_cast<long>(resultsConsumed));
    _meshScheduler->acknowledgeConsumerPending(_pendingResults.size());

    // 2. Edit fast path: chunks right next to the camera re-mesh
    //    synchronously so breaking a block never shows a stale frame.
    int syncBuilds = 0;
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated || syncBuilds >= 2 || uploads >= MAX_MESH_UPLOADS_PER_FRAME)
            continue;
        if (std::abs(chunk->chunkX - camChunkX) > 2 || std::abs(chunk->chunkY - camChunkY) > 2 ||
            std::abs(chunk->chunkZ - camChunkZ) > 2)
            continue;
        if (!shouldMesh(chunk->pos()))
            continue;
        ChunkPos key = chunk->pos();
        auto it = _chunkMeshes.find(key);
        uint32_t version = chunk->version.load(std::memory_order_relaxed);
        // Only REBUILDS take the sync path (builtVersion != 0): first-time
        // builds stream through the workers like everything else
        if (it == _chunkMeshes.end() || it->second.builtVersion == 0 ||
            it->second.builtVersion == version) {
            continue;
        }
        if (!world.snapshotForMeshing(chunk->pos(), _meshSnapshot)) {
            continue;
        }
        MeshOutput mesh = LODMesher::buildMesh(_meshSnapshot, _meshScratch);
        const size_t bytes =
            mesh.vertices.size() * sizeof(Vertex) + mesh.indices.size() * sizeof(uint32_t);
        if (uploadBytes + bytes > MAX_UPLOAD_BYTES_PER_FRAME)
            continue;
        if (applyMesh(key, mesh, _meshSnapshot.version, 0)) {
            ++syncBuilds;
            ++uploads;
            uploadBytes += bytes;
        }
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
        if (!shouldMesh(chunk->pos()))
            continue;
        ChunkPos key = chunk->pos();
        uint32_t version = chunk->version.load(std::memory_order_relaxed);
        auto it = _chunkMeshes.find(key);
        if (it != _chunkMeshes.end() &&
            (it->second.builtVersion == version || it->second.requestedVersion == version)) {
            continue; // up to date, or a build is already on its way
        }
        float dx = static_cast<float>(chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2) - camX;
        float dy = static_cast<float>(chunk->chunkY * CHUNK_HEIGHT + CHUNK_HEIGHT / 2) - camY;
        float dz = static_cast<float>(chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2) - camZ;
        _meshCandidates.push_back({dx * dx + dy * dy + dz * dz, chunk.get()});
    }
    const auto candidateLane = [&](const Chunk* chunk) {
        const int64_t dx = chunk->chunkX - camChunkX;
        const int64_t dz = chunk->chunkZ - camChunkZ;
        if (dx == 0 && dz == 0)
            return MeshPriorityLane::CAMERA_COLUMN;
        return dx * dx + dz * dz <= EXPLORATION_RADIUS_CHUNKS * EXPLORATION_RADIUS_CHUNKS
                   ? MeshPriorityLane::CAMERA_BAND
                   : MeshPriorityLane::BROAD_SURFACE;
    };
    std::sort(_meshCandidates.begin(), _meshCandidates.end(),
              [&](const auto& left, const auto& right) {
                  const MeshPriorityLane leftLane = candidateLane(left.second);
                  const MeshPriorityLane rightLane = candidateLane(right.second);
                  if (leftLane != rightLane)
                      return static_cast<uint8_t>(leftLane) > static_cast<uint8_t>(rightLane);
                  return left.first < right.first;
              });
    for (const auto& [distSq, chunkPtr] : _meshCandidates) {
        ChunkPos pos = chunkPtr->pos();
        if (!canMakeMeshSlot(pos)) {
            break; // every resident mesh owns exact terrain; wait for a real eviction
        }
        const uint32_t requestedVersion = chunkPtr->version.load(std::memory_order_relaxed);
        const int64_t chunkDx = pos.x - camChunkX;
        const int64_t chunkDy = static_cast<int64_t>(pos.y) - camChunkY;
        const int64_t chunkDz = pos.z - camChunkZ;
        const MeshPriorityLane lane = candidateLane(chunkPtr);
        const uint64_t schedulerDistance =
            static_cast<uint64_t>(chunkDx * chunkDx + chunkDz * chunkDz + chunkDy * chunkDy * 2);
        if (!_meshScheduler->enqueue(pos, requestedVersion, lane, schedulerDistance)) {
            break; // in-flight cap reached — re-prioritized next frame
        }
        ChunkMeshState* state = makeMeshSlot(pos);
        if (!state) {
            break; // defensive: the registry never grows beyond the hard cap
        }
        state->requestedVersion = requestedVersion;
    }

    auto overlayColor = [&](const worldgen::SurfaceSample& sample) {
        constexpr float biomeColors[][3] = {
            {0.02f, 0.08f, 0.30f}, {0.03f, 0.22f, 0.55f}, {0.42f, 0.72f, 0.24f},
            {0.08f, 0.45f, 0.13f}, {0.20f, 0.42f, 0.35f}, {0.88f, 0.74f, 0.28f},
            {0.42f, 0.42f, 0.39f}, {0.17f, 0.39f, 0.25f}, {0.55f, 0.25f, 0.62f},
            {0.75f, 0.89f, 0.95f}, {0.91f, 0.83f, 0.56f}, {0.04f, 0.42f, 0.80f},
            {0.30f, 0.66f, 0.38f}, {0.64f, 0.82f, 0.30f}, {0.69f, 0.67f, 0.20f},
            {0.03f, 0.56f, 0.17f}, {0.10f, 0.50f, 0.32f}, {0.48f, 0.54f, 0.27f},
            {0.64f, 0.59f, 0.26f}, {0.72f, 0.65f, 0.49f}, {0.68f, 0.31f, 0.17f},
            {0.55f, 0.62f, 0.53f}, {0.58f, 0.64f, 0.67f}, {0.13f, 0.44f, 0.29f},
            {0.40f, 0.68f, 0.83f}, {0.29f, 0.23f, 0.22f}, {0.82f, 0.92f, 0.97f},
            {0.52f, 0.61f, 0.25f}, {0.30f, 0.60f, 0.31f}, {0.55f, 0.47f, 0.22f},
            {0.16f, 0.38f, 0.30f}, {0.12f, 0.50f, 0.28f}, {0.49f, 0.53f, 0.20f},
        };
        static_assert(std::size(biomeColors) == static_cast<size_t>(Biome::COUNT));
        switch (_worldgenOverlayMode) {
            case WorldgenOverlayMode::GEOLOGY: {
                const uint64_t id = sample.geology.plateId;
                float r = 0.2f + static_cast<float>((id >> 8U) & 255U) / 425.0f;
                float g = 0.2f + static_cast<float>((id >> 24U) & 255U) / 425.0f;
                float b = 0.2f + static_cast<float>((id >> 40U) & 255U) / 425.0f;
                if (sample.geology.boundary != worldgen::PlateBoundary::NONE) {
                    constexpr float boundaryColors[4][3] = {
                        {0.65f, 0.65f, 0.65f},
                        {0.95f, 0.16f, 0.08f},
                        {0.10f, 0.43f, 1.00f},
                        {1.00f, 0.65f, 0.06f},
                    };
                    const size_t boundary = static_cast<size_t>(sample.geology.boundary);
                    const float proximity = static_cast<float>(
                        std::clamp(1.0 - sample.geology.distanceToBoundary / 900.0, 0.0, 0.85));
                    r = std::lerp(r, boundaryColors[boundary][0], proximity);
                    g = std::lerp(g, boundaryColors[boundary][1], proximity);
                    b = std::lerp(b, boundaryColors[boundary][2], proximity);
                }
                const float volcanic =
                    static_cast<float>(std::clamp(sample.geology.volcanicActivity, 0.0, 1.0));
                return simd_make_float4(std::lerp(r, 1.0f, volcanic * 0.7f),
                                        std::lerp(g, 0.08f, volcanic * 0.7f),
                                        std::lerp(b, 0.02f, volcanic * 0.7f), 0.72f);
            }
            case WorldgenOverlayMode::HYDROLOGY:
                if (sample.hydrology.delta)
                    return simd_make_float4(0.95f, 0.80f, 0.16f, 0.82f);
                if (sample.hydrology.waterfall)
                    return simd_make_float4(0.80f, 0.95f, 1.00f, 0.85f);
                if (sample.hydrology.river)
                    return simd_make_float4(0.02f, 0.65f, 1.00f, 0.82f);
                if (sample.hydrology.lake)
                    return simd_make_float4(0.08f, 0.42f, 0.94f, 0.78f);
                if (sample.hydrology.ocean)
                    return simd_make_float4(0.01f, 0.08f, 0.42f, 0.78f);
                return simd_make_float4(0.18f, 0.16f, 0.11f, 0.65f);
            case WorldgenOverlayMode::CLIMATE: {
                const float temperature = static_cast<float>(
                    std::clamp((sample.climate.temperatureC + 25.0) / 65.0, 0.0, 1.0));
                const float rain = static_cast<float>(
                    std::clamp(sample.climate.annualPrecipitationMm / 3000.0, 0.0, 1.0));
                return simd_make_float4(temperature, rain, 1.0f - temperature, 0.74f);
            }
            case WorldgenOverlayMode::BIOME: {
                const size_t primary = static_cast<size_t>(sample.biome.primary);
                const size_t secondary = static_cast<size_t>(sample.biome.secondary);
                const float blend =
                    static_cast<float>(std::clamp(sample.biome.transition, 0.0, 1.0));
                return simd_make_float4(
                    std::lerp(biomeColors[primary][0], biomeColors[secondary][0], blend),
                    std::lerp(biomeColors[primary][1], biomeColors[secondary][1], blend),
                    std::lerp(biomeColors[primary][2], biomeColors[secondary][2], blend), 0.72f);
            }
            case WorldgenOverlayMode::NONE:
                break;
        }
        return simd_make_float4(0.0f);
    };

    // ---- Draw everything the registry has uploaded ----
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated)
            continue;

        ChunkPos key = chunk->pos();

        auto cached = _chunkMeshes.find(key);
        if (cached == _chunkMeshes.end())
            continue;

        const auto& meshState = cached->second;
        if (farTerrainExactSectionOwnsSurface(_exactOwnedSections.contains(key),
                                              meshState.builtVersion,
                                              chunk->version.load(std::memory_order_relaxed))) {
            setExactSectionOwned(key, true);
        }
        if (!meshState.uploaded)
            continue;

        // Readiness is independent of the current view direction, while draw
        // submission remains frustum culled.
        AABB chunkAABB = chunk->getAABB();
        if (!isChunkInFrustum(chunkAABB))
            continue;

        // Mesh vertices are chunk-local; this restores world space (and keeps
        // fp16 positions exact regardless of how far the chunk is from origin)
        ChunkOrigin origin{};
        origin.origin = simd_make_float4(static_cast<float>(chunk->chunkX * CHUNK_WIDTH),
                                         static_cast<float>(chunk->chunkY * CHUNK_HEIGHT),
                                         static_cast<float>(chunk->chunkZ * CHUNK_DEPTH), 0.0f);
        if (_worldgenOverlayMode != WorldgenOverlayMode::NONE) {
            const int64_t centerX = chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2;
            const int64_t centerZ = chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2;
            if (const auto sample = world.findSurfaceSample(centerX, centerZ)) {
                origin.overlayColorAndStrength = overlayColor(*sample);
            }
        }

        // The chunk's water section renders after the scene resolves
        uint32_t waterIndexCount = meshState.alloc.indexCount - meshState.opaqueIndexCount;
        if (waterIndexCount > 0) {
            float dx = static_cast<float>(chunk->chunkX * CHUNK_WIDTH + CHUNK_WIDTH / 2) - camX;
            float dy = static_cast<float>(chunk->chunkY * CHUNK_HEIGHT + CHUNK_HEIGHT / 2) - camY;
            float dz = static_cast<float>(chunk->chunkZ * CHUNK_DEPTH + CHUNK_DEPTH / 2) - camZ;
            _waterDraws.push_back(WaterDraw{
                origin.origin, origin.overlayColorAndStrength, origin.farMetadata, noFarOwnership,
                meshState.alloc.vertexBuffer, meshState.alloc.indexBuffer,
                meshState.alloc.vertexOffset,
                meshState.alloc.indexOffset + meshState.opaqueIndexCount * sizeof(uint32_t),
                waterIndexCount, dx * dx + dy * dy + dz * dz});
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

    // Fill the exact-to-horizon region after exact cubes. Each ready exact
    // column suppresses its own far terrain, water, canopies, and skirts. An
    // unrelated loading gap therefore cannot reveal coarse geometry over a
    // revision-ready nearby column.
    renderFarTerrain(encoder, world, cameraPosition, fogColor);

    // F3 HUD counters: uploads applied this frame + the workers' build EMA
    _chunkStats.meshBuildsLastFrame = static_cast<uint32_t>(uploads);
    _chunkStats.meshMsAvg = _meshScheduler->meshMsAvg();
    _chunkStats.megaUsedMB =
        static_cast<float>(_megaBuffer->vertexUsed() + _megaBuffer->indexUsed()) /
        (1024.f * 1024.f);
    _chunkStats.megaCapMB =
        static_cast<float>(_megaBuffer->vertexCapacity() + _megaBuffer->indexCapacity()) /
        (1024.f * 1024.f);
    _chunkStats.meshCubeCount = static_cast<uint32_t>(_chunkMeshes.size());
    const MeshSchedulerStats meshStats = _meshScheduler->stats();
    _chunkStats.meshPendingCount =
        static_cast<uint32_t>(meshStats.schedulerOwned + meshStats.consumerPending);
    _chunkStats.meshQueueHighWater = static_cast<uint32_t>(meshStats.highWater);
    _chunkStats.meshCoalescedCount = meshStats.coalesced;
    _chunkStats.meshDroppedStaleCount = meshStats.droppedStale;
}

void RenderPipeline::resetFarTerrain(uint64_t worldSeed) {
    if (_farTerrainSeed && *_farTerrainSeed == worldSeed && _farTerrainScheduler)
        return;

    if (_farTerrainScheduler)
        _farTerrainScheduler->shutdown();
    if (_farMegaBuffer) {
        for (auto& [key, state] : _farTerrainMeshes) {
            (void)key;
            if (state.uploaded) {
                _farMegaBuffer->deferFree(state.alloc, _frameRing.frameIndex());
            }
        }
    }
    _farTerrainMeshes.clear();
    _farTerrainWanted.clear();
    _farTerrainPriorityOrder.clear();
    _farTerrainActiveTiles.clear();
    _farTerrainDesiredByTile.clear();
    _farTerrainDisplayedByTile.clear();
    _farTerrainComplexityByTile.clear();
    _farTerrainTransitions.clear();
    _farTerrainNearGraceStartedAt.clear();
    _farTerrainResults.clear();
    _farTerrainCandidates.clear();
    _farTerrainCachedBaseRequests.clear();
    _farTerrainUrgentRefinementRequests.clear();
    _farTerrainUrgentRefinementKeys.clear();
    _farTerrainCachedRefinementRequests.clear();
    _farTerrainCachedMeshes.clear();
    _farTerrainCenterTile.reset();
    clearExactSectionOwnership();
    _farTerrainResidentWantedCount = 0;
    _farTerrainResidentRefinementCount = 0;

    if (!_farMegaBuffer) {
        _farMegaBuffer = std::make_unique<SegmentedMegaBuffer>(
            _device, FAR_VERTEX_BUFFER_BYTES, FAR_INDEX_BUFFER_BYTES, FAR_VERTEX_BUFFER_SLAB_BYTES,
            FAR_INDEX_BUFFER_SLAB_BYTES);
    }
    _farTerrainScheduler = std::make_unique<FarTerrainScheduler>(worldSeed);
    _farTerrainSeed = worldSeed;
    _farTerrainMeshes.reserve(8192);
    _farTerrainWanted.reserve(8192);
    _farTerrainPriorityOrder.reserve(8192);
    _farTerrainActiveTiles.reserve(4096);
    _farTerrainDesiredByTile.reserve(4096);
    _farTerrainDisplayedByTile.reserve(4096);
    _farTerrainComplexityByTile.reserve(4096);
    _farTerrainTransitions.reserve(64);
    _farTerrainNearGraceStartedAt.reserve(4096);
    _farTerrainCandidates.reserve(4096);
    _farTerrainCachedBaseRequests.reserve(4096);
    _farTerrainUrgentRefinementRequests.reserve(256);
    _farTerrainUrgentRefinementKeys.reserve(320);
    _farTerrainCachedRefinementRequests.reserve(4096);
    _farTerrainCachedMeshes.reserve(FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME);
    _exactOwnedSections.reserve(MAX_MESH_RESIDENT_CUBES);
}

void RenderPipeline::setExactSectionOwned(ChunkPos position, bool owned) {
    if (owned) {
        if (_exactOwnedSections.insert(position).second) {
            _farTerrainExactCoverage.setSectionReady(position, true);
        }
        return;
    }
    if (_exactOwnedSections.erase(position) != 0) {
        _farTerrainExactCoverage.setSectionReady(position, false);
    }
}

void RenderPipeline::clearExactSectionOwnership() {
    _exactOwnedSections.clear();
    _farTerrainExactCoverage.clear();
}

void RenderPipeline::renderFarTerrain(id<MTLRenderCommandEncoder> encoder, const World& world,
                                      const Vec3& cameraPosition, const float fogColor[3]) {
    resetFarTerrain(world.getSeed());
    _farMegaBuffer->drainDeferredFrees(_frameRing.completedFrame());
    _farTerrainActiveTiles.clear();
    _farTerrainCandidates.clear();

    const int exactChunks = world.getExactViewDistance();
    const int visibleChunks = world.getViewDistance();
    const auto exactCoverage = world.getExactSurfaceCoverageSnapshot();
    const int nominalExactChunks =
        exactCoverage ? std::clamp(exactCoverage->nominalRadiusChunks, 0, exactChunks) : 0;
    const float nominalExactBlocks = static_cast<float>(nominalExactChunks * CHUNK_EDGE);
    const uint64_t exactCoverageEpoch = exactCoverage ? exactCoverage->epoch : 0;
    if (!_farTerrainExactCoverage.matches(exactCoverageEpoch, nominalExactChunks)) {
        _farTerrainExactCoverage.rebuild(
            exactCoverageEpoch, nominalExactChunks,
            exactCoverage ? std::span<const ChunkPos>(exactCoverage->requiredSections)
                          : std::span<const ChunkPos>(),
            exactCoverage ? std::span<const ColumnPos>(exactCoverage->unresolvedColumns)
                          : std::span<const ColumnPos>(),
            [&](ChunkPos position) { return _exactOwnedSections.contains(position); });
    }
    const FarTerrainExactHandoff& exactHandoff =
        _farTerrainExactCoverage.sample(cameraPosition.x, cameraPosition.z);
    const MeshSchedulerStats exactMeshStats = _meshScheduler->stats();
    const bool exactStreamingBusy = farTerrainExactStreamingBusy(
        world.getPendingChunkCount(), exactMeshStats.schedulerOwned,
        std::max(exactMeshStats.consumerPending, _pendingResults.size()),
        exactHandoff.requiredSections, exactHandoff.readySections, exactHandoff.unresolvedColumns);
    // Far construction runs at utility priority, while exact generation and
    // meshing retain higher priorities on their separate pools. Keep all far
    // workers available during cold exact streaming so horizon parents and
    // nearby selected targets can converge together on the reference M4 Max.
    _farTerrainScheduler->setWorkerBudget(FarTerrainScheduler::WORKER_COUNT);
    const TerrainHorizonViewpoint viewpoint{cameraPosition.x, cameraPosition.y, cameraPosition.z};
    selectFarTerrainView(cameraPosition.x, cameraPosition.z, visibleChunks, _farTerrainCandidates);
    const int64_t cameraBlockX = static_cast<int64_t>(std::floor(cameraPosition.x));
    const int64_t cameraBlockZ = static_cast<int64_t>(std::floor(cameraPosition.z));
    const ColumnPos centerTile{
        world_coord::floorDiv(cameraBlockX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
        world_coord::floorDiv(cameraBlockZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
    };
    if (_farTerrainCenterTile && (std::abs(centerTile.x - _farTerrainCenterTile->x) > 2 ||
                                  std::abs(centerTile.z - _farTerrainCenterTile->z) > 2)) {
        _farTerrainScheduler->advanceEpoch();
    }
    _farTerrainCenterTile = centerTile;
    for (FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        _farTerrainActiveTiles.insert(coordinate);

        // Retain the maximum observed value while the coordinate is active.
        // Different sample spacings can under-resolve a narrow ridge or river;
        // allowing that observation to fall again would make two tiers chase
        // each other despite the distance hysteresis.
        float& complexity = _farTerrainComplexityByTile[coordinate];
        for (FarTerrainStep step :
             {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            const auto resident = _farTerrainMeshes.find({tile.key.tileX, tile.key.tileZ, step});
            if (resident != _farTerrainMeshes.end()) {
                complexity = std::max(complexity, resident->second.complexity);
            }
        }
        std::optional<FarTerrainStep> previousStep;
        if (const auto previous = _farTerrainDesiredByTile.find(coordinate);
            previous != _farTerrainDesiredByTile.end()) {
            previousStep = previous->second.step;
        }
        if (const auto desired =
                farTerrainStepForMetrics(tile.distanceChunks, complexity, previousStep)) {
            tile.key.step = *desired;
        }
        _farTerrainDesiredByTile.insert_or_assign(coordinate, tile.key);
    }
    for (auto it = _farTerrainDesiredByTile.begin(); it != _farTerrainDesiredByTile.end();) {
        if (!_farTerrainActiveTiles.contains(it->first)) {
            it = _farTerrainDesiredByTile.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = _farTerrainComplexityByTile.begin(); it != _farTerrainComplexityByTile.end();) {
        if (!_farTerrainActiveTiles.contains(it->first)) {
            it = _farTerrainComplexityByTile.erase(it);
        } else {
            ++it;
        }
    }
    // Every visible coordinate owns a step-32 parent independently of its
    // desired detail. Parents occupy the first priority lane in nearest-first
    // order, followed by refinements in the same stable coordinate order.
    const bool residencyMembershipChanged =
        !farTerrainResidencyMembershipMatches(_farTerrainCandidates, _farTerrainWanted);
    if (residencyMembershipChanged) {
        buildFarTerrainResidencyOrder(_farTerrainCandidates, _farTerrainPriorityOrder);
        _farTerrainWanted.clear();
        _farTerrainWanted.insert(_farTerrainPriorityOrder.begin(), _farTerrainPriorityOrder.end());
        _farTerrainScheduler->retainWanted(_farTerrainWanted, _farTerrainPriorityOrder);
    }

    auto isResident = [&](const FarTerrainKey& key) {
        const auto found = _farTerrainMeshes.find(key);
        return found != _farTerrainMeshes.end() && found->second.uploaded;
    };
    if (residencyMembershipChanged) {
        _farTerrainResidentWantedCount = 0;
        _farTerrainResidentRefinementCount = 0;
        for (const auto& [key, state] : _farTerrainMeshes) {
            if (!state.uploaded || !_farTerrainWanted.contains(key))
                continue;
            ++_farTerrainResidentWantedCount;
            if (!farTerrainIsBaseStep(key.step))
                ++_farTerrainResidentRefinementCount;
        }
    }

    size_t uploads = 0;
    size_t baseUploads = 0;
    size_t refinementUploads = 0;
    size_t uploadBytes = 0;
    bool uploadFailureLogged = false;
    auto uploadMesh = [&](const std::shared_ptr<const FarTerrainMesh>& mesh) {
        if (!mesh || _farTerrainWanted.count(mesh->key) == 0 ||
            _farTerrainMeshes.count(mesh->key) != 0) {
            return false;
        }
        const bool base = farTerrainIsBaseStep(mesh->key.step);
        size_t& laneUploads = base ? baseUploads : refinementUploads;
        const size_t laneLimit = exactStreamingBusy
                                     ? (base ? FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME : size_t{4})
                                     : (base ? FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME
                                             : FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME);
        if (laneUploads >= laneLimit)
            return false;
        const size_t bytes =
            mesh->vertices.size() * sizeof(Vertex) + mesh->indices.size() * sizeof(uint32_t);
        if (uploadBytes + bytes > FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            return false;
        std::optional<MegaBuffer::ChunkAllocation> allocation;
        try {
            allocation = _farMegaBuffer->allocate(static_cast<uint32_t>(mesh->vertices.size()),
                                                  static_cast<uint32_t>(mesh->indices.size()));
            _farMegaBuffer->uploadVertices(mesh->vertices.data(),
                                           mesh->vertices.size() * sizeof(Vertex), *allocation);
            _farMegaBuffer->uploadIndices(mesh->indices.data(),
                                          mesh->indices.size() * sizeof(uint32_t), *allocation);
            const auto [_, inserted] = _farTerrainMeshes.emplace(
                mesh->key, FarTerrainMeshState{*allocation, mesh->bounds, mesh->surfaceBounds,
                                               mesh->occluderPatches, mesh->opaqueIndexCount,
                                               mesh->complexity, mesh->deterministicHash, true});
            if (!inserted) {
                _farMegaBuffer->free(*allocation);
                return false;
            }
            ++_farTerrainResidentWantedCount;
            if (!base)
                ++_farTerrainResidentRefinementCount;
            ++uploads;
            ++laneUploads;
            uploadBytes += bytes;
            return true;
        } catch (const std::exception& error) {
            if (allocation)
                _farMegaBuffer->free(*allocation);
            if (!uploadFailureLogged) {
                RY_LOG_ERROR((std::string("Far-terrain upload failed: ") + error.what()).c_str());
                uploadFailureLogged = true;
            }
            return false;
        }
    };

    _farTerrainResults.clear();
    _farTerrainScheduler->drainCompleted(_farTerrainResults);
    const auto distanceSquaredForKey = [&](const FarTerrainKey& key) {
        const double minimumX = static_cast<double>(key.tileX) * FAR_TERRAIN_TILE_EDGE;
        const double maximumX = minimumX + FAR_TERRAIN_TILE_EDGE;
        const double minimumZ = static_cast<double>(key.tileZ) * FAR_TERRAIN_TILE_EDGE;
        const double maximumZ = minimumZ + FAR_TERRAIN_TILE_EDGE;
        const double dx = cameraPosition.x < minimumX   ? minimumX - cameraPosition.x
                          : cameraPosition.x > maximumX ? cameraPosition.x - maximumX
                                                        : 0.0;
        const double dz = cameraPosition.z < minimumZ   ? minimumZ - cameraPosition.z
                          : cameraPosition.z > maximumZ ? cameraPosition.z - maximumZ
                                                        : 0.0;
        return dx * dx + dz * dz;
    };
    std::sort(_farTerrainResults.begin(), _farTerrainResults.end(),
              [&](const FarTerrainResult& first, const FarTerrainResult& second) {
                  const bool firstBase = farTerrainIsBaseStep(first.key.step);
                  const bool secondBase = farTerrainIsBaseStep(second.key.step);
                  if (firstBase != secondBase)
                      return firstBase;
                  const double firstDistance = distanceSquaredForKey(first.key);
                  const double secondDistance = distanceSquaredForKey(second.key);
                  if (firstDistance != secondDistance)
                      return firstDistance < secondDistance;
                  if (first.key.tileX != second.key.tileX)
                      return first.key.tileX < second.key.tileX;
                  if (first.key.tileZ != second.key.tileZ)
                      return first.key.tileZ < second.key.tileZ;
                  return farTerrainStepSize(first.key.step) < farTerrainStepSize(second.key.step);
              });

    // CPU completion order cannot bypass the parent lane. Upload every base
    // result and cached base nearest-first before considering refinements.
    for (const FarTerrainResult& result : _farTerrainResults) {
        if (!result.failed && farTerrainIsBaseStep(result.key.step))
            uploadMesh(result.mesh);
    }
    _farTerrainCachedBaseRequests.clear();
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const FarTerrainKey base{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        if (!isResident(base))
            _farTerrainCachedBaseRequests.push_back(base);
    }
    _farTerrainScheduler->findCachedBatch(_farTerrainCachedBaseRequests,
                                          FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - baseUploads,
                                          _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    const double lodTimeSeconds = CACurrentMediaTime();
    const auto findNearGrace = [&](ColumnPos coordinate) {
        return std::find_if(_farTerrainNearGraceStartedAt.begin(),
                            _farTerrainNearGraceStartedAt.end(),
                            [&](const auto& entry) { return entry.first == coordinate; });
    };
    const auto startNearGrace = [&](ColumnPos coordinate) {
        const auto found = findNearGrace(coordinate);
        if (found != _farTerrainNearGraceStartedAt.end())
            return found->second;
        _farTerrainNearGraceStartedAt.emplace_back(coordinate, lodTimeSeconds);
        return lodTimeSeconds;
    };
    const auto eraseNearGrace = [&](ColumnPos coordinate) {
        const auto found = findNearGrace(coordinate);
        if (found != _farTerrainNearGraceStartedAt.end())
            _farTerrainNearGraceStartedAt.erase(found);
    };
    const auto displayedStepFor = [&](ColumnPos coordinate) {
        const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        if (const auto found = _farTerrainDisplayedByTile.find(coordinate);
            found != _farTerrainDisplayedByTile.end() && isResident(found->second))
            return found->second.step;
        return base.step;
    };
    const auto residentStepMaskFor = [&](ColumnPos coordinate) {
        FarTerrainStepMask mask = 0;
        for (FarTerrainStep step :
             {FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            if (isResident({coordinate.x, coordinate.z, step}))
                mask |= farTerrainStepMask(step);
        }
        return mask;
    };
    const auto requiresFineFallback = [&](ColumnPos coordinate) {
        return farTerrainRequiresCoverageParent(cameraPosition.x, cameraPosition.z, coordinate,
                                                nominalExactBlocks, exactHandoff);
    };
    const auto requiresBlockScaleFallback = [&](ColumnPos coordinate) {
        constexpr float EXPLORATION_BLOCKS = EXPLORATION_RADIUS_CHUNKS * CHUNK_EDGE;
        return requiresFineFallback(coordinate) &&
               farTerrainRequiresCoverageParent(cameraPosition.x, cameraPosition.z, coordinate,
                                                EXPLORATION_BLOCKS, exactHandoff);
    };
    const auto coarsestFallbackFor = [&](ColumnPos coordinate) {
        if (requiresBlockScaleFallback(coordinate))
            return FarTerrainStep::TWO;
        if (requiresFineFallback(coordinate))
            return FarTerrainStep::EIGHT;
        return FAR_TERRAIN_BASE_STEP;
    };
    const auto isDrawableCoverageResident = [&](FarTerrainKey base) {
        if (!isResident(base))
            return false;
        const ColumnPos coordinate{base.tileX, base.tileZ};
        const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
        if (coarsestAllowed == FAR_TERRAIN_BASE_STEP)
            return true;
        if (coarsestAllowed == FarTerrainStep::TWO)
            return isResident({coordinate.x, coordinate.z, FarTerrainStep::TWO});
        return isResident({coordinate.x, coordinate.z, FarTerrainStep::TWO}) ||
               isResident({coordinate.x, coordinate.z, FarTerrainStep::FOUR}) ||
               isResident({coordinate.x, coordinate.z, FarTerrainStep::EIGHT});
    };
    FarTerrainCoverageFrontier parentCoverage =
        farTerrainCoverageFrontier(_farTerrainCandidates, isResident);
    FarTerrainCoverageFrontier coverage =
        farTerrainCoverageFrontier(_farTerrainCandidates, isDrawableCoverageResident);
    // Every connected parent may request its distance-selected child before
    // the complete 8 km parent disk arrives. The bounded worker lane advances
    // the 16, 8, 4, and 2-block rings together instead of leaving an empty
    // middle distance while thousands of horizon parents finish.
    _farTerrainUrgentRefinementRequests.clear();
    constexpr size_t MAX_PROGRESSIVE_REQUESTS_PER_TIER = 64;
    std::array<size_t, 4> progressiveRequestsPerTier{};
    const auto progressiveTierIndex = [](FarTerrainStep step) -> size_t {
        switch (step) {
            case FarTerrainStep::SIXTEEN:
                return 0;
            case FarTerrainStep::EIGHT:
                return 1;
            case FarTerrainStep::FOUR:
                return 2;
            case FarTerrainStep::TWO:
                return 3;
            case FarTerrainStep::ONE:
            case FarTerrainStep::THIRTY_TWO:
                return 4;
        }
        return 4;
    };
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const bool cameraTile = coordinate == centerTile;
        const bool fineFallbackRequired = requiresFineFallback(coordinate);
        const bool blockScaleFallbackRequired = requiresBlockScaleFallback(coordinate);
        if (!cameraTile && !fineFallbackRequired &&
            !farTerrainCoverageDrawEligible(tile.distanceSquared, parentCoverage)) {
            // A same-distance protected tile may follow a ready exact tile in
            // coordinate order, so keep scanning the bounded candidate set.
            continue;
        }
        const size_t tierIndex = progressiveTierIndex(tile.key.step);
        if (tierIndex >= progressiveRequestsPerTier.size() ||
            progressiveRequestsPerTier[tierIndex] >= MAX_PROGRESSIVE_REQUESTS_PER_TIER) {
            continue;
        }
        const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        const bool baseResident = isResident(base);
        const float tileExactHandoffBlocks =
            exactHandoff.distanceBlocksForTile(coordinate, nominalExactBlocks);
        if (!farTerrainConnectedRefinementEligible(tile, tileExactHandoffBlocks, parentCoverage,
                                                   baseResident,
                                                   cameraTile || fineFallbackRequired)) {
            continue;
        }
        const FarTerrainStep displayed = displayedStepFor(coordinate);
        if (farTerrainStepSize(displayed) <= farTerrainStepSize(tile.key.step))
            continue;
        const bool transitionActive = _farTerrainTransitions.contains(coordinate);
        bool deferIntermediate = !baseResident;
        if (!fineFallbackRequired && !transitionActive && baseResident &&
            displayed == FAR_TERRAIN_BASE_STEP && tile.key.step == FarTerrainStep::TWO) {
            const float parentAgeSeconds =
                static_cast<float>(lodTimeSeconds - startNearGrace(coordinate));
            deferIntermediate = deferIntermediate || farTerrainDeferNearIntermediate(
                                                         displayed, tile.key.step,
                                                         FarTerrainStep::EIGHT, parentAgeSeconds);
        }
        _farTerrainUrgentRefinementRequests.push_back(
            {coordinate, displayed, tile.key.step, residentStepMaskFor(coordinate),
             transitionActive, deferIntermediate, fineFallbackRequired,
             blockScaleFallbackRequired});
        ++progressiveRequestsPerTier[tierIndex];
    }

    // Keep the camera target first, then seed one request from each remaining
    // tier in coarse-to-fine order. Four refinement workers therefore build a
    // spatial taper instead of all entering the much slower step-2 ring.
    size_t promotedRequests = 0;
    if (const auto camera = std::find_if(
            _farTerrainUrgentRefinementRequests.begin(), _farTerrainUrgentRefinementRequests.end(),
            [&](const auto& request) { return request.coordinate == centerTile; });
        camera != _farTerrainUrgentRefinementRequests.end()) {
        std::rotate(_farTerrainUrgentRefinementRequests.begin(), camera, std::next(camera));
        promotedRequests = 1;
    }
    for (const FarTerrainStep tier : {FarTerrainStep::SIXTEEN, FarTerrainStep::EIGHT,
                                      FarTerrainStep::FOUR, FarTerrainStep::TWO}) {
        if (promotedRequests >= FAR_TERRAIN_MAX_URGENT_REFINEMENTS_IN_FLIGHT)
            break;
        const auto firstUnpromoted =
            _farTerrainUrgentRefinementRequests.begin() +
            static_cast<std::vector<FarTerrainRefinementCacheRequest>::difference_type>(
                promotedRequests);
        const auto tierRequest =
            std::find_if(firstUnpromoted, _farTerrainUrgentRefinementRequests.end(),
                         [&](const auto& request) { return request.desired == tier; });
        if (tierRequest == _farTerrainUrgentRefinementRequests.end())
            continue;
        std::rotate(firstUnpromoted, tierRequest, std::next(tierRequest));
        ++promotedRequests;
    }
    reserveFarTerrainIntermediateTransitionSlots(_farTerrainUrgentRefinementRequests,
                                                 _farTerrainTransitions.size());
    _farTerrainScheduler->findFinestCachedBatch(_farTerrainUrgentRefinementRequests,
                                                FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME,
                                                _farTerrainCachedMeshes);
    size_t uploadedProgressiveIntermediates = 0;
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        if (!uploadMesh(mesh))
            continue;
        const ColumnPos coordinate{mesh->key.tileX, mesh->key.tileZ};
        const auto desired = _farTerrainDesiredByTile.find(coordinate);
        if (desired != _farTerrainDesiredByTile.end() && desired->second.step != mesh->key.step) {
            ++uploadedProgressiveIntermediates;
        }
    }

    // The broad optional lane begins after the complete visible parent disk is
    // GPU resident. Connected targets and near fallbacks above already advance
    // within the visible prefix. Once coverage completes, one cache lock
    // selects the finest useful results for this broader upload lane. A fresh
    // nearby parent gets 120 ms for its selected target to arrive, but a ready
    // step-2 result bypasses that grace immediately.
    const size_t refinementLaneLimit =
        exactStreamingBusy ? size_t{4} : FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME;
    _farTerrainCachedRefinementRequests.clear();
    if (farTerrainRefinementLaneOpen(parentCoverage, true)) {
        for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
            const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
            const FarTerrainStep displayed = displayedStepFor(coordinate);
            const bool fineFallbackRequired = requiresFineFallback(coordinate);
            const bool blockScaleFallbackRequired = requiresBlockScaleFallback(coordinate);
            float parentAgeSeconds = std::numeric_limits<float>::infinity();
            if (!fineFallbackRequired && displayed == FAR_TERRAIN_BASE_STEP &&
                tile.key.step == FarTerrainStep::TWO &&
                !_farTerrainTransitions.contains(coordinate)) {
                parentAgeSeconds = static_cast<float>(lodTimeSeconds - startNearGrace(coordinate));
            } else {
                eraseNearGrace(coordinate);
            }
            const bool deferIntermediate = farTerrainDeferNearIntermediate(
                displayed, tile.key.step, FarTerrainStep::EIGHT, parentAgeSeconds);
            _farTerrainCachedRefinementRequests.push_back(
                {coordinate, displayed, tile.key.step, residentStepMaskFor(coordinate),
                 _farTerrainTransitions.contains(coordinate), deferIntermediate,
                 fineFallbackRequired, blockScaleFallbackRequired});
        }
    }
    std::erase_if(_farTerrainNearGraceStartedAt, [&](const auto& entry) {
        return !_farTerrainActiveTiles.contains(entry.first) ||
               !isResident({entry.first.x, entry.first.z, FAR_TERRAIN_BASE_STEP});
    });
    reserveFarTerrainIntermediateTransitionSlots(_farTerrainCachedRefinementRequests,
                                                 _farTerrainTransitions.size() +
                                                     uploadedProgressiveIntermediates);
    _farTerrainScheduler->findFinestCachedBatch(_farTerrainCachedRefinementRequests,
                                                refinementLaneLimit - refinementUploads,
                                                _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    parentCoverage = farTerrainCoverageFrontier(_farTerrainCandidates, isResident);
    coverage = farTerrainCoverageFrontier(_farTerrainCandidates, isDrawableCoverageResident);

    // Submission is nearest-first and bounded inside the scheduler. Seed the
    // parent lane before urgent work so a frame-boundary queue drain cannot
    // let every utility worker enter a long refinement. The scheduler keeps
    // four workers on connected coverage and permits four progressive
    // selected targets to advance at the same time.
    size_t refinementOffset = 0;
    size_t baseSubmissions = 0;
    constexpr size_t MAX_BASE_SUBMISSIONS_PER_FRAME = 64;
    for (; refinementOffset < _farTerrainPriorityOrder.size(); ++refinementOffset) {
        const FarTerrainKey key = _farTerrainPriorityOrder[refinementOffset];
        if (!farTerrainIsBaseStep(key.step))
            break;
        if (isResident(key))
            continue;
        if (_farTerrainScheduler->hasSubmissionCapacity()) {
            _farTerrainScheduler->enqueue(key, static_cast<uint32_t>(refinementOffset * 8));
            ++baseSubmissions;
        }
        ++refinementOffset;
        break;
    }

    size_t urgentRefinementSubmissions = 0;
    buildFarTerrainProgressiveSubmissionOrder(_farTerrainUrgentRefinementRequests,
                                              _farTerrainUrgentRefinementKeys);
    for (size_t index = 0;
         index < _farTerrainUrgentRefinementKeys.size() &&
         urgentRefinementSubmissions < FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
         _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey target = _farTerrainUrgentRefinementKeys[index];
        if (isResident(target))
            continue;
        if (_farTerrainScheduler->enqueueUrgentRefinement(
                target, static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX)))) {
            ++urgentRefinementSubmissions;
        }
    }

    for (; refinementOffset < _farTerrainPriorityOrder.size(); ++refinementOffset) {
        if (!_farTerrainScheduler->hasSubmissionCapacity() ||
            baseSubmissions >= MAX_BASE_SUBMISSIONS_PER_FRAME)
            break;
        const FarTerrainKey key = _farTerrainPriorityOrder[refinementOffset];
        if (!farTerrainIsBaseStep(key.step))
            break;
        if (!isResident(key)) {
            _farTerrainScheduler->enqueue(key, static_cast<uint32_t>(refinementOffset * 8));
            ++baseSubmissions;
        }
    }
    const bool parentSubmissionComplete = refinementOffset == _farTerrainCandidates.size();
    for (size_t tileIndex = 0;
         farTerrainRefinementLaneOpen(parentCoverage, parentSubmissionComplete) &&
         tileIndex < _farTerrainCandidates.size();
         ++tileIndex) {
        if (!_farTerrainScheduler->hasSubmissionCapacity())
            break;
        const FarTerrainViewTile& tile = _farTerrainCandidates[tileIndex];
        if (!isResident(tile.key)) {
            _farTerrainScheduler->enqueue(tile.key, static_cast<uint32_t>(tileIndex * 8) + 1U);
        }
    }
    for (size_t tileIndex = 0;
         farTerrainRefinementLaneOpen(parentCoverage, parentSubmissionComplete) &&
         tileIndex < _farTerrainCandidates.size();
         ++tileIndex) {
        const FarTerrainViewTile& tile = _farTerrainCandidates[tileIndex];
        uint32_t stagePriority = 2;
        const FarTerrainRefinementOrder refinement = farTerrainRefinementOrder(tile.key.step);
        for (FarTerrainStep step : std::span(refinement.steps).first(refinement.count)) {
            if (!_farTerrainScheduler->hasSubmissionCapacity())
                break;
            if (step == tile.key.step)
                continue;
            const FarTerrainKey key{tile.key.tileX, tile.key.tileZ, step};
            if (!isResident(key)) {
                _farTerrainScheduler->enqueue(key,
                                              static_cast<uint32_t>(tileIndex * 8) + stagePriority);
            }
            ++stagePriority;
        }
        if (!_farTerrainScheduler->hasSubmissionCapacity())
            break;
    }

    // Finish each monotonic replacement before reevaluating the desired tier.
    // Redirecting a transition mid-flight can make already revealed voxel
    // columns disappear and is perceived as flicker during ordinary travel.
    for (auto it = _farTerrainTransitions.begin(); it != _farTerrainTransitions.end();) {
        if (!_farTerrainActiveTiles.contains(it->first)) {
            it = _farTerrainTransitions.erase(it);
            continue;
        }
        const auto desired = _farTerrainDesiredByTile.find(it->first);
        const FarTerrainStep desiredStep =
            desired == _farTerrainDesiredByTile.end() ? it->second.to.step : desired->second.step;
        const FarTerrainLodAdvance advance =
            advanceFarTerrainLod(it->second.from.step, desiredStep, it->second.to.step,
                                 static_cast<float>(lodTimeSeconds - it->second.startedAtSeconds));
        if (advance.completedTransition) {
            _farTerrainDisplayedByTile.insert_or_assign(
                it->first, FarTerrainKey{it->first.x, it->first.z, advance.displayed});
            it = _farTerrainTransitions.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = _farTerrainDisplayedByTile.begin(); it != _farTerrainDisplayedByTile.end();) {
        if (!_farTerrainActiveTiles.contains(it->first)) {
            it = _farTerrainDisplayedByTile.erase(it);
        } else if (!farTerrainDisplayedStepAllowed(it->second.step,
                                                   coarsestFallbackFor(it->first))) {
            _farTerrainTransitions.erase(it->first);
            it = _farTerrainDisplayedByTile.erase(it);
        } else {
            ++it;
        }
    }

    // A coordinate cannot become visible before its parent is resident, but
    // it initializes directly to the finest resident selected tier. Reentering
    // a cached nearby tile must not briefly display step 32 and transition
    // through a lower-detail canopy again.
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        if (!isResident(base)) {
            _farTerrainDisplayedByTile.erase(coordinate);
            _farTerrainTransitions.erase(coordinate);
            eraseNearGrace(coordinate);
            continue;
        }
        auto displayed = _farTerrainDisplayedByTile.find(coordinate);
        if (displayed != _farTerrainDisplayedByTile.end() && isResident(displayed->second)) {
            continue;
        }
        const auto desired = _farTerrainDesiredByTile.find(coordinate);
        const FarTerrainStep desiredStep =
            desired == _farTerrainDesiredByTile.end() ? tile.key.step : desired->second.step;
        const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
        const std::optional<FarTerrainStep> initial = farTerrainInitialDisplayedStep(
            desiredStep, residentStepMaskFor(coordinate), coarsestAllowed);
        if (initial) {
            _farTerrainDisplayedByTile.insert_or_assign(
                coordinate, FarTerrainKey{coordinate.x, coordinate.z, *initial});
        }
    }

    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (_farTerrainTransitions.size() >= FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS)
            break;
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        if (_farTerrainTransitions.contains(coordinate))
            continue;
        const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
        const auto desired = _farTerrainDesiredByTile.find(coordinate);
        if (displayed == _farTerrainDisplayedByTile.end() ||
            desired == _farTerrainDesiredByTile.end() || displayed->second == desired->second ||
            !isResident(displayed->second)) {
            continue;
        }
        const std::optional<FarTerrainStep> readyTarget = farTerrainReadyTransitionTarget(
            displayed->second.step, desired->second.step, residentStepMaskFor(coordinate), false);
        if (!readyTarget)
            continue;
        float parentAgeSeconds = std::numeric_limits<float>::infinity();
        if (const auto grace = findNearGrace(coordinate);
            grace != _farTerrainNearGraceStartedAt.end()) {
            parentAgeSeconds = static_cast<float>(lodTimeSeconds - grace->second);
        }
        if (farTerrainDeferNearIntermediate(displayed->second.step, desired->second.step,
                                            *readyTarget, parentAgeSeconds)) {
            continue;
        }
        const FarTerrainLodAdvance advance =
            advanceFarTerrainLod(displayed->second.step, *readyTarget);
        if (!advance.transitionTarget)
            continue;
        const FarTerrainKey next{coordinate.x, coordinate.z, *advance.transitionTarget};
        if (!isResident(next))
            continue;
        _farTerrainTransitions.emplace(
            coordinate, FarTerrainLodTransition{displayed->second, next, lodTimeSeconds});
        eraseNearGrace(coordinate);
    }

    // GPU residency follows the full circular horizon rather than the current
    // camera direction. During a replacement, both immutable tiers stay live;
    // all other stale tiers retire through the frame-safe arena immediately.
    for (auto it = _farTerrainMeshes.begin(); it != _farTerrainMeshes.end();) {
        const ColumnPos coordinate{it->first.tileX, it->first.tileZ};
        bool keep = _farTerrainActiveTiles.contains(coordinate);
        if (keep) {
            const auto desired = _farTerrainDesiredByTile.find(coordinate);
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            const auto transition = _farTerrainTransitions.find(coordinate);
            keep =
                farTerrainIsBaseStep(it->first.step) ||
                (coarsestFallbackFor(coordinate) != FAR_TERRAIN_BASE_STEP &&
                 farTerrainDisplayedStepAllowed(it->first.step, coarsestFallbackFor(coordinate))) ||
                (desired != _farTerrainDesiredByTile.end() && desired->second == it->first) ||
                (displayed != _farTerrainDisplayedByTile.end() && displayed->second == it->first) ||
                (transition != _farTerrainTransitions.end() &&
                 (transition->second.from == it->first || transition->second.to == it->first));
        }
        if (!keep) {
            if (it->second.uploaded) {
                if (_farTerrainWanted.contains(it->first)) {
                    if (_farTerrainResidentWantedCount > 0)
                        --_farTerrainResidentWantedCount;
                    if (!farTerrainIsBaseStep(it->first.step) &&
                        _farTerrainResidentRefinementCount > 0) {
                        --_farTerrainResidentRefinementCount;
                    }
                }
                _farMegaBuffer->deferFree(it->second.alloc, _frameRing.frameIndex());
            }
            it = _farTerrainMeshes.erase(it);
        } else {
            ++it;
        }
    }

    const float exactHandoffBlocks = exactHandoff.distanceBlocks;

    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];
    [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [encoder setCullMode:MTLCullModeBack];
    // Exact terrain draws first. The shader clips each far fragment with its
    // destination-column ownership bit; the depth bias remains only as a
    // conservative fallback while an exact column is still cold.
    [encoder setDepthBias:4.0f slopeScale:1.0f clamp:0.000002f];

    TerrainHorizonCuller horizon(viewpoint);
    uint32_t drawn = 0;
    uint32_t baseDrawn = 0;
    uint32_t refinementDrawn = 0;
    uint32_t frustumCulled = 0;
    uint32_t occlusionCulled = 0;
    auto displayedKeyFor = [&](ColumnPos coordinate) -> std::optional<FarTerrainKey> {
        if (!_farTerrainActiveTiles.contains(coordinate))
            return std::nullopt;
        const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        if (!isResident(base))
            return std::nullopt;
        const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
        const auto displayAllowed = [&](FarTerrainStep step) {
            return farTerrainDisplayedStepAllowed(step, coarsestAllowed);
        };

        if (const auto transition = _farTerrainTransitions.find(coordinate);
            transition != _farTerrainTransitions.end()) {
            // Water remains source-owned until completion. Terrain and its
            // skirts swap together under the narrow fog pulse.
            if (isResident(transition->second.from) && displayAllowed(transition->second.from.step))
                return transition->second.from;
            if (isResident(transition->second.to) && displayAllowed(transition->second.to.step))
                return transition->second.to;
        }
        if (const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            displayed != _farTerrainDisplayedByTile.end() && isResident(displayed->second) &&
            displayAllowed(displayed->second.step)) {
            return displayed->second;
        }
        const auto desired = _farTerrainDesiredByTile.find(coordinate);
        const FarTerrainStep desiredStep =
            desired == _farTerrainDesiredByTile.end() ? FarTerrainStep::TWO : desired->second.step;
        const std::optional<FarTerrainStep> fallback = farTerrainInitialDisplayedStep(
            desiredStep, residentStepMaskFor(coordinate), coarsestAllowed);
        if (!fallback)
            return std::nullopt;
        return FarTerrainKey{coordinate.x, coordinate.z, *fallback};
    };
    auto visibleTerrainKeyFor = [&](ColumnPos coordinate) -> std::optional<FarTerrainKey> {
        if (const auto transition = _farTerrainTransitions.find(coordinate);
            transition != _farTerrainTransitions.end() && isResident(transition->second.from) &&
            isResident(transition->second.to)) {
            const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
            if (!farTerrainDisplayedStepAllowed(transition->second.from.step, coarsestAllowed)) {
                if (farTerrainDisplayedStepAllowed(transition->second.to.step, coarsestAllowed))
                    return transition->second.to;
                return displayedKeyFor(coordinate);
            }
            const FarTerrainTransitionSample sample = sampleFarTerrainTransition(
                static_cast<float>(lodTimeSeconds - transition->second.startedAtSeconds));
            uint32_t flags = FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
            if (transition->second.from.step == FarTerrainStep::THIRTY_TWO &&
                transition->second.to.step == FarTerrainStep::TWO) {
                flags |= FAR_TERRAIN_LOD_EMERGENCY_FLAG;
            }
            if (farTerrainLodTerrainVisible(sample.progress, flags | FAR_TERRAIN_LOD_TARGET_FLAG)) {
                return transition->second.to;
            }
            return transition->second.from;
        }
        return displayedKeyFor(coordinate);
    };
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (!farTerrainCoverageDrawEligible(tile.distanceSquared, coverage))
            continue;
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        struct DrawPlan {
            FarTerrainKey key;
            float progress = 0.0F;
            uint32_t flags = FAR_TERRAIN_DRAW_FLAG;
            bool ownsConnectedGeometry = true;
        };
        std::array<DrawPlan, 2> plans{};
        size_t planCount = 0;
        if (const auto transition = _farTerrainTransitions.find(coordinate);
            transition != _farTerrainTransitions.end() && isResident(transition->second.from) &&
            isResident(transition->second.to)) {
            const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
            if (!farTerrainDisplayedStepAllowed(transition->second.from.step, coarsestAllowed)) {
                if (farTerrainDisplayedStepAllowed(transition->second.to.step, coarsestAllowed))
                    plans[planCount++] = {transition->second.to};
            } else {
                const FarTerrainTransitionSample sample = sampleFarTerrainTransition(
                    static_cast<float>(lodTimeSeconds - transition->second.startedAtSeconds));
                const uint32_t transitionFlags =
                    FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG |
                    (transition->second.from.step == FarTerrainStep::THIRTY_TWO &&
                             transition->second.to.step == FarTerrainStep::TWO
                         ? FAR_TERRAIN_LOD_EMERGENCY_FLAG
                         : 0U);
                plans[planCount++] = {transition->second.from, sample.progress, transitionFlags,
                                      true};
                plans[planCount++] = {transition->second.to, sample.progress,
                                      transitionFlags | FAR_TERRAIN_LOD_TARGET_FLAG, false};
            }
        } else if (const std::optional<FarTerrainKey> displayed = displayedKeyFor(coordinate)) {
            plans[planCount++] = {*displayed};
        }
        if (planCount == 0)
            continue;

        FarTerrainBounds visibilityBounds = _farTerrainMeshes.at(plans.front().key).surfaceBounds;
        for (size_t index = 1; index < planCount; ++index) {
            const FarTerrainBounds& bounds = _farTerrainMeshes.at(plans[index].key).surfaceBounds;
            visibilityBounds.minY = std::min(visibilityBounds.minY, bounds.minY);
            visibilityBounds.maxY = std::max(visibilityBounds.maxY, bounds.maxY);
        }
        const AABB aabb{{static_cast<float>(visibilityBounds.minX), visibilityBounds.minY,
                         static_cast<float>(visibilityBounds.minZ)},
                        {static_cast<float>(visibilityBounds.maxX), visibilityBounds.maxY,
                         static_cast<float>(visibilityBounds.maxZ)}};
        if (!isChunkInFrustum(aabb)) {
            ++frustumCulled;
            continue;
        }
        // A replacement uses the union bounds and remains visible throughout
        // its monotonic reveal. Culling either topology independently can
        // punch a transient hole when the camera crosses a frustum or horizon
        // threshold during those 650 milliseconds.
        if (planCount == 1 && horizon.isOccluded(visibilityBounds)) {
            ++occlusionCulled;
            continue;
        }
        const float tileExactHandoffBlocks =
            exactHandoff.distanceBlocksForTile(coordinate, nominalExactBlocks);
        const FarTerrainExactHandoff::ColumnMask readyColumns =
            exactHandoff.readyColumnMask(coordinate);
        const FarTerrainOwnershipUniforms farOwnership =
            farTerrainOwnershipUniforms(coordinate, exactHandoff);
        const double fullyOpaqueFarRadius =
            static_cast<double>(tileExactHandoffBlocks) + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS;
        const double fullyOpaqueFarRadiusSquared = fullyOpaqueFarRadius * fullyOpaqueFarRadius;
        const FarTerrainMeshState& occluderState = _farTerrainMeshes.at(plans.front().key);
        for (const FarTerrainBounds& patch : occluderState.occluderPatches) {
            // A clipped or dithered patch is not a conservative solid
            // occluder. Boundary tiles are few, so omit those patches until
            // their complete bounds are beyond the handoff band.
            const bool fullyOpaque = farTerrainCoveragePatchMayOcclude(
                patch, viewpoint, coverage, FAR_TERRAIN_COVERAGE_FADE_BLOCKS, planCount > 1);
            if (fullyOpaque &&
                !farTerrainOccluderIntersectsExact(patch, coordinate, readyColumns) &&
                TerrainHorizonCuller::horizontalDistanceSquared(patch, viewpoint) >=
                    fullyOpaqueFarRadiusSquared) {
                horizon.addOccluder(patch);
            }
        }

        std::array<std::optional<FarTerrainStep>, 4> displayedNeighborSteps;
        const std::array<ColumnPos, 4> neighborCoordinates = {
            ColumnPos{coordinate.x + 1, coordinate.z}, ColumnPos{coordinate.x - 1, coordinate.z},
            ColumnPos{coordinate.x, coordinate.z + 1}, ColumnPos{coordinate.x, coordinate.z - 1}};
        for (size_t edge = 0; edge < neighborCoordinates.size(); ++edge) {
            if (const auto neighbor = visibleTerrainKeyFor(neighborCoordinates[edge])) {
                displayedNeighborSteps[edge] = neighbor->step;
            }
        }
        for (const DrawPlan& plan : std::span(plans).first(planCount)) {
            // Target canopies overlap source canopies during their monotonic
            // exchange. A slightly smaller target bias makes matching crowns
            // stable without changing exact-cube ownership.
            const bool transitionTarget = (plan.flags & FAR_TERRAIN_LOD_TARGET_FLAG) != 0U;
            [encoder setDepthBias:(planCount > 1 && !transitionTarget ? 6.0f : 4.0f)
                       slopeScale:1.0f
                            clamp:0.000002f];
            const FarTerrainMeshState& state = _farTerrainMeshes.at(plan.key);
            ChunkOrigin origin{};
            origin.origin = simd_make_float4(static_cast<float>(state.bounds.minX), 0.0f,
                                             static_cast<float>(state.bounds.minZ), 0.0F);
            origin.overlayColorAndStrength =
                simd_make_float4(fogColor[0], fogColor[1], fogColor[2], 0.0F);
            origin.farMetadata.x = farTerrainSkirtEdgeMask(plan.key.step, displayedNeighborSteps);
            origin.farMetadata.y = std::bit_cast<uint32_t>(coverage.distanceBlocks);
            origin.farMetadata.z = std::bit_cast<uint32_t>(plan.progress);
            origin.farMetadata.w = plan.flags;

            // Canopies use a monotonic target-in, source-out exchange, so
            // unrelated forest summaries never pass through an empty phase.
            // Target water is not submitted at all, guaranteeing one
            // refractive owner through the replacement.
            const uint32_t waterIndexCount =
                plan.ownsConnectedGeometry ? state.alloc.indexCount - state.opaqueIndexCount : 0;
            if (waterIndexCount > 0) {
                const double centerX =
                    static_cast<double>(state.surfaceBounds.minX) +
                    static_cast<double>(state.surfaceBounds.maxX - state.surfaceBounds.minX) * 0.5;
                const double centerY =
                    static_cast<double>(state.surfaceBounds.minY) +
                    static_cast<double>(state.surfaceBounds.maxY - state.surfaceBounds.minY) * 0.5;
                const double centerZ =
                    static_cast<double>(state.surfaceBounds.minZ) +
                    static_cast<double>(state.surfaceBounds.maxZ - state.surfaceBounds.minZ) * 0.5;
                const double dx = centerX - cameraPosition.x;
                const double dy = centerY - cameraPosition.y;
                const double dz = centerZ - cameraPosition.z;
                _waterDraws.push_back(WaterDraw{
                    origin.origin, origin.overlayColorAndStrength, origin.farMetadata, farOwnership,
                    state.alloc.vertexBuffer, state.alloc.indexBuffer, state.alloc.vertexOffset,
                    state.alloc.indexOffset + state.opaqueIndexCount * sizeof(uint32_t),
                    waterIndexCount, static_cast<float>(dx * dx + dy * dy + dz * dz)});
            }
            if (state.opaqueIndexCount > 0) {
                [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
                [encoder setFragmentBytes:&farOwnership length:sizeof(farOwnership) atIndex:5];
                [encoder setVertexBuffer:state.alloc.vertexBuffer
                                  offset:state.alloc.vertexOffset
                                 atIndex:0];
                [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:state.opaqueIndexCount
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:state.alloc.indexBuffer
                             indexBufferOffset:state.alloc.indexOffset];
            }
        }
        if (farTerrainIsBaseStep(plans.front().key.step)) {
            ++baseDrawn;
        } else {
            ++refinementDrawn;
        }
        ++drawn;
    }
    [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];

    const FarTerrainSchedulerStats schedulerStats = _farTerrainScheduler->stats();
    const uint32_t baseWanted = static_cast<uint32_t>(_farTerrainCandidates.size());
    const uint32_t refinementWanted =
        static_cast<uint32_t>(_farTerrainWanted.size() - _farTerrainCandidates.size());
    const uint32_t refinementResident = static_cast<uint32_t>(_farTerrainResidentRefinementCount);
    _chunkStats.exactSurfaceRequiredCount = static_cast<uint32_t>(
        std::min<size_t>(exactHandoff.requiredSections, std::numeric_limits<uint32_t>::max()));
    _chunkStats.exactSurfaceReadyCount = static_cast<uint32_t>(
        std::min<size_t>(exactHandoff.readySections, std::numeric_limits<uint32_t>::max()));
    _chunkStats.exactSurfaceUnresolvedColumnCount = static_cast<uint32_t>(
        std::min<size_t>(exactHandoff.unresolvedColumns, std::numeric_limits<uint32_t>::max()));
    _chunkStats.exactSurfaceHandoffBlocks = exactHandoffBlocks;
    _chunkStats.farWantedTileCount = static_cast<uint32_t>(_farTerrainWanted.size());
    _chunkStats.farResidentTileCount = static_cast<uint32_t>(_farTerrainResidentWantedCount);
    _chunkStats.farBaseWantedTileCount = baseWanted;
    _chunkStats.farBaseResidentTileCount = baseWanted - parentCoverage.missingBaseTiles;
    _chunkStats.farBaseDrawnTileCount = baseDrawn;
    _chunkStats.farBaseMissingTileCount = parentCoverage.missingBaseTiles;
    _chunkStats.farRefinementWantedTileCount = refinementWanted;
    _chunkStats.farRefinementResidentTileCount = refinementResident;
    _chunkStats.farRefinementDrawnTileCount = refinementDrawn;
    _chunkStats.farDrawnTileCount = drawn;
    _chunkStats.farFrustumCulledTileCount = frustumCulled;
    _chunkStats.farOcclusionCulledTileCount = occlusionCulled;
    _chunkStats.farPendingTileCount =
        static_cast<uint32_t>(schedulerStats.inFlight + schedulerStats.completed);
    _chunkStats.farUploadsLastFrame = static_cast<uint32_t>(uploads);
    _chunkStats.farQueuedBaseTileCount = static_cast<uint32_t>(schedulerStats.queuedBase);
    _chunkStats.farQueuedRefinementTileCount =
        static_cast<uint32_t>(schedulerStats.queuedRefinement);
    _chunkStats.farActiveBaseWorkerCount = static_cast<uint32_t>(schedulerStats.activeBaseWorkers);
    _chunkStats.farReservedBaseWorkerCount =
        static_cast<uint32_t>(schedulerStats.reservedBaseWorkers);
    _chunkStats.farActiveUrgentRefinementCount =
        static_cast<uint32_t>(schedulerStats.activeUrgentRefinement);
    _chunkStats.farWorkerBudget = static_cast<uint32_t>(schedulerStats.workerBudget);
    _chunkStats.farCachedBaseTileCount = static_cast<uint32_t>(schedulerStats.cacheBaseEntries);
    _chunkStats.farCoverageFrontierBlocks = coverage.distanceBlocks;
    _chunkStats.farCacheMB = static_cast<float>(schedulerStats.cacheBytes) / (1024.0f * 1024.0f);
    _chunkStats.farMegaUsedMB =
        static_cast<float>(_farMegaBuffer->vertexUsed() + _farMegaBuffer->indexUsed()) /
        (1024.0f * 1024.0f);
}

void RenderPipeline::shutdownMeshWorkers() {
    if (_meshScheduler) {
        _meshScheduler->shutdown();
    }
    if (_farTerrainScheduler) {
        _farTerrainScheduler->shutdown();
    }
}

FarTerrainGenerationCacheStats RenderPipeline::farGenerationCacheStats() const {
    return _farTerrainScheduler ? _farTerrainScheduler->generationCacheStats()
                                : FarTerrainGenerationCacheStats{};
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
                                 const float fogColor[3]) {
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
    // Water depth differences are only a few blocks, even when absolute world
    // coordinates are large. Remove camera translation before inversion so
    // reconstructing those positions cannot quantize at chunk boundaries.
    simd_float4x4 cameraRelativeView = view;
    cameraRelativeView.columns[3] = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    wu.cameraRelativeViewProjection = simd_mul(proj, cameraRelativeView);
    wu.invCameraRelativeViewProjection = simd_inverse(wu.cameraRelativeViewProjection);
    wu.zenithColor = skyUniforms.zenithColor;
    wu.horizonColor = skyUniforms.horizonColor;
    wu.sunDirection = skyUniforms.sunDirection;
    wu.sunColor = skyUniforms.sunColor;
    wu.cameraPosition = simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    wu.fogColor = simd_make_float3(fogColor[0], fogColor[1], fogColor[2]);
    wu.resolution =
        simd_make_float2(static_cast<float>(_displayWidth), static_cast<float>(_displayHeight));
    wu.fogDensity = _fogDensity;
    wu.time = _animTime; // the shared per-frame animation clock
    wu.cameraUnderwater = cameraUnderwater ? 1.f : 0.f;
    // Screen-space reflections layer onto the fresnel sky term; 0 keeps the
    // pre-SSR look (also the RYCRAFT_SSR=0 / setting-off path).
    wu.ssrStrength = _gfx.waterReflections ? 1.0f : 0.0f;
    wu.skyExposure = _uwSkyExposure;
    wu.waterSurfaceY = _uwSurfaceY;
    FrameRing::Alloc waterAlloc = _frameRing.push(&wu, sizeof(WaterUniforms));

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
        [encoder setVertexBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
        [encoder setVertexBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
        [encoder setFragmentBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
        [encoder setFragmentTexture:_sceneColorCopy atIndex:0];
        [encoder setFragmentTexture:_depthResolve atIndex:1];

        for (const WaterDraw& draw : _waterDraws) {
            ChunkOrigin origin{draw.origin, draw.overlayColorAndStrength, draw.farMetadata};
            [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
            [encoder setFragmentBytes:&draw.farOwnership
                               length:sizeof(draw.farOwnership)
                              atIndex:5];
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
        [encoder setFragmentBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
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

    FrameRing::Alloc highlightAlloc = _frameRing.push(&uniforms, sizeof(Uniforms));

    [encoder setRenderPipelineState:_highlightPipelineState];
    [encoder setDepthStencilState:_noDepthWriteState];

    [encoder setVertexBuffer:_highlightVertexBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:highlightAlloc.buffer offset:highlightAlloc.offset atIndex:1];
    [encoder setFragmentBuffer:highlightAlloc.buffer offset:highlightAlloc.offset atIndex:1];

    // Highlight vertices carry their translation in the model matrix
    ChunkOrigin zeroOrigin{};
    [encoder setVertexBytes:&zeroOrigin length:sizeof(zeroOrigin) atIndex:2];
    // The highlight shares fragmentMain, which now samples the atlas AND the
    // shadow cascade. Bind both (a disabled shadow block — strength 0 — keeps
    // the yellow wireframe fully lit) so it never reads stale chunk-loop state.
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];
    ShadowUniforms noShadows{};
    FrameRing::Alloc noShadowAlloc = _frameRing.push(&noShadows, sizeof(ShadowUniforms));
    [encoder setFragmentTexture:_shadowMap->depthTexture() atIndex:1];
    [encoder setFragmentSamplerState:_shadowMap->comparisonSampler() atIndex:1];
    [encoder setFragmentBuffer:noShadowAlloc.buffer offset:noShadowAlloc.offset atIndex:4];

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
    if (_volumetrics) {
        _volumetrics->resize(_displayWidth, _displayHeight);
    }
    if (_ssao) {
        _ssao->resize(_displayWidth, _displayHeight);
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
// setGraphicsSettings — the engine pushes a copy on init and on every video
// settings change; passes read the copy each frame and skip when disabled.
// ---------------------------------------------------------------------------
void RenderPipeline::setGraphicsSettings(const GraphicsSettings& gfx) {
    _gfx = gfx;
    setBloomIntensity(gfx.bloomIntensity()); // level 5 = stock 1.0; 0 skips
}

// ---------------------------------------------------------------------------
// tickParticles — Update weather particle physics each game tick
// ---------------------------------------------------------------------------
void RenderPipeline::tickParticles(float dt, const World& world, const Vec3& playerPosition,
                                   bool raining) {
    if (!_particles)
        return;
    _particles->tick(dt, world, playerPosition, raining);
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
                                  uint64_t worldTime, const float sunDirection[3],
                                  float sunIntensity) {
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
    cloudUniforms.volumetric = _gfx.cloudMode >= 2 ? 1.0f : 0.0f;
    // Sun height 0..1 from the sky's single source (NOT the active light's
    // direction — that swaps to the MOON below the horizon, whose +y at
    // midnight would keep clouds daytime-bright all night).
    cloudUniforms.sunElevation = sunIntensity;

    FrameRing::Alloc cloudAlloc = _frameRing.push(&cloudUniforms, sizeof(cloudUniforms));

    [encoder setRenderPipelineState:_cloudPipelineState];
    [encoder setDepthStencilState:_cloudDepthState];
    [encoder setVertexBuffer:cloudAlloc.buffer offset:cloudAlloc.offset atIndex:0];
    [encoder setFragmentBuffer:cloudAlloc.buffer offset:cloudAlloc.offset atIndex:0];

    // Draw fullscreen quad (6 vertices)
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}
