#import "render/render_pipeline.hpp"

#include "common/error.hpp"
#include "common/random.hpp"
#include "render/atmosphere.hpp"
#include "render/atmospheric_memory.hpp"
#include "render/block_textures.hpp"
#include "render/bloom.hpp"
#include "render/boat_renderer.hpp"
#include "render/celestial.hpp"
#include "render/cloud_renderer.hpp"
#include "render/entity_renderer.hpp"
#include "render/item_entity_renderer.hpp"
#include "render/lightning_renderer.hpp"
#include "render/lod_mesher.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"
#include "render/post_stack.hpp"
#include "render/screen_space_lighting.hpp"
#include "render/shadow_map.hpp"
#include "render/volumetrics.hpp"

#include "engine/camera.hpp"
#include "render/particles.hpp"
#include "render/ui_hud.hpp"
#include "render/ui_overlay.hpp"
#include "world/chunk.hpp"
#include "world/chunk_pos.hpp"
#include "world/weather.hpp"
#include "world/world.hpp"

static_assert(EXACT_SURFACE_MESH_PRIORITY_RADIUS_CHUNKS == MAX_EXACT_CUBIC_DISTANCE_CHUNKS);

#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
// One frame's constants (chunk/sky/water/highlight/cloud uniform blocks plus
// the particle instance array) sub-allocate from this ring slot; the particle
// array dominates at 192 KB.
static constexpr uint64_t FRAME_RING_SLOT_BYTES = 256 * 1024;
static constexpr uint64_t EXACT_VERTEX_BUFFER_BYTES =
    MAX_MESH_RESIDENT_CUBES * 128ull * 1024 * 13 / 10;
static constexpr uint64_t EXACT_INDEX_BUFFER_BYTES = EXACT_VERTEX_BUFFER_BYTES / 2;
static constexpr uint64_t EXACT_VERTEX_BUFFER_SLAB_BYTES = 256ull * 1024 * 1024;
static constexpr uint64_t EXACT_INDEX_BUFFER_SLAB_BYTES = 128ull * 1024 * 1024;
static_assert((EXACT_VERTEX_BUFFER_BYTES + EXACT_VERTEX_BUFFER_SLAB_BYTES - 1) /
                  EXACT_VERTEX_BUFFER_SLAB_BYTES ==
              (EXACT_INDEX_BUFFER_BYTES + EXACT_INDEX_BUFFER_SLAB_BYTES - 1) /
                  EXACT_INDEX_BUFFER_SLAB_BYTES);
static constexpr uint64_t FAR_VERTEX_BUFFER_BYTES = 2ull * 1024 * 1024 * 1024;
static constexpr uint64_t FAR_INDEX_BUFFER_BYTES = 1ull * 1024 * 1024 * 1024;
static constexpr uint64_t FAR_VERTEX_BUFFER_SLAB_BYTES = 256ull * 1024 * 1024;
static constexpr uint64_t FAR_INDEX_BUFFER_SLAB_BYTES = 128ull * 1024 * 1024;
static constexpr size_t FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET = 8;
static_assert(FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR == 4);
static_assert(FarTerrainExactHandoff::COLUMN_MASK_WORD_COUNT == FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
static_assert(FAR_TERRAIN_TILE_EDGE / CHUNK_EDGE == FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);

static FarTerrainOwnershipUniforms
farTerrainOwnershipUniforms(ColumnPos centerTile, const FarTerrainExactHandoff& surfaceHandoff,
                            const FarTerrainExactHandoff& floraHandoff) {
    FarTerrainOwnershipUniforms ownership{};
    for (int64_t neighborZ = -FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS;
         neighborZ <= FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS; ++neighborZ) {
        for (int64_t neighborX = -FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS;
             neighborX <= FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS; ++neighborX) {
            const size_t tileIndex =
                static_cast<size_t>((neighborZ + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS) *
                                        FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE +
                                    neighborX + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS);
            const ColumnPos tile{centerTile.x + neighborX, centerTile.z + neighborZ};
            const FarTerrainExactHandoff::ColumnMask surfaceMask =
                surfaceHandoff.readyColumnMask(tile);
            const FarTerrainExactHandoff::ColumnMask floraMask = floraHandoff.readyColumnMask(tile);
            for (size_t word = 0; word < surfaceMask.size(); ++word) {
                ownership.readyColumnMasks[tileIndex * FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE +
                                           word / FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR]
                                          [word % FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR] =
                    surfaceMask[word];
                ownership
                    .floraReadyColumnMasks[tileIndex * FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE +
                                           word / FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR]
                                          [word % FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR] =
                    floraMask[word];
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

void buildFarTerrainCanopyRefreshBatch(
    const std::unordered_map<ColumnPos, FarTerrainKey>& displayed,
    const std::unordered_map<ColumnPos, FarTerrainLodTransition>& transitions,
    const std::unordered_map<FarTerrainKey, FarTerrainMeshState, FarTerrainKeyHash>& residents,
    const std::unordered_map<FarTerrainKey, FarCanopyMeshState, FarTerrainKeyHash>& attachments,
    double cameraX, double cameraZ, size_t requestBudget,
    std::vector<FarTerrainCanopyRefreshRequest>& output,
    const FarTerrainExactHandoff* exactFloraHandoff) {
    output.clear();
    if (requestBudget == 0)
        return;

    std::vector<FarTerrainCanopyRefreshRequest> missingAttachments;
    std::vector<FarTerrainCanopyRefreshRequest> provisionalPromotions;
    missingAttachments.reserve(requestBudget);
    provisionalPromotions.reserve(requestBudget);
    const auto before = [](const FarTerrainCanopyRefreshRequest& first,
                           const FarTerrainCanopyRefreshRequest& second) {
        if (first.distanceSquaredBlocks != second.distanceSquaredBlocks)
            return first.distanceSquaredBlocks < second.distanceSquaredBlocks;
        if (first.key.tileX != second.key.tileX)
            return first.key.tileX < second.key.tileX;
        if (first.key.tileZ != second.key.tileZ)
            return first.key.tileZ < second.key.tileZ;
        return farTerrainStepSize(first.key.step) < farTerrainStepSize(second.key.step);
    };
    const auto insertBounded = [&](std::vector<FarTerrainCanopyRefreshRequest>& requests,
                                   const FarTerrainCanopyRefreshRequest& request) {
        if (auto duplicate =
                std::ranges::find(requests, request.key, &FarTerrainCanopyRefreshRequest::key);
            duplicate != requests.end()) {
            duplicate->transitionTarget = duplicate->transitionTarget || request.transitionTarget;
            return;
        }
        if (requests.size() == requestBudget) {
            if (!before(request, requests.back()))
                return;
            requests.pop_back();
        }
        const auto insertion = std::lower_bound(requests.begin(), requests.end(), request, before);
        requests.insert(insertion, request);
    };
    const auto appendIfFinalResident = [&](FarTerrainKey key, bool transitionTarget) {
        const ColumnPos coordinate{key.tileX, key.tileZ};
        if (exactFloraHandoff && exactFloraHandoff->tileFullyOwned(coordinate)) {
            return;
        }
        const auto resident = residents.find(key);
        if (resident == residents.end() || !resident->second.uploaded) {
            return;
        }
        const auto attachment = attachments.find(key);
        const bool matchingAttachment =
            attachment != attachments.end() &&
            farCanopyMatchesSurface(attachment->second.authorityQuality,
                                    attachment->second.groundingQuality,
                                    resident->second.authorityQuality);
        if (matchingAttachment &&
            attachment->second.authorityQuality == FarTerrainAuthorityQuality::FINAL) {
            return;
        }
        const double minimumX = static_cast<double>(key.tileX) * FAR_TERRAIN_TILE_EDGE;
        const double maximumX = minimumX + FAR_TERRAIN_TILE_EDGE;
        const double minimumZ = static_cast<double>(key.tileZ) * FAR_TERRAIN_TILE_EDGE;
        const double maximumZ = minimumZ + FAR_TERRAIN_TILE_EDGE;
        const double dx = cameraX < minimumX   ? minimumX - cameraX
                          : cameraX > maximumX ? cameraX - maximumX
                                               : 0.0;
        const double dz = cameraZ < minimumZ   ? minimumZ - cameraZ
                          : cameraZ > maximumZ ? cameraZ - maximumZ
                                               : 0.0;
        const double distanceSquared = dx * dx + dz * dz;
        const uint32_t absolutePriority = static_cast<uint32_t>(std::min(
            std::sqrt(distanceSquared), static_cast<double>(std::numeric_limits<uint32_t>::max())));
        const FarTerrainCanopyRefreshRequest request{key, resident->second.authorityQuality,
                                                     absolutePriority, transitionTarget,
                                                     distanceSquared};
        // PREVIEW attachments schedule their own FINAL successor when they
        // complete. Keep them eligible for recovery, but do not let a handful
        // of parked promotions occupy the entire bounded request batch while
        // other drawable tiles still have no flora at all.
        insertBounded(matchingAttachment ? provisionalPromotions : missingAttachments, request);
    };
    for (const auto& [_, key] : displayed)
        appendIfFinalResident(key, false);
    for (const auto& [_, transition] : transitions)
        appendIfFinalResident(transition.to, true);

    const size_t missingCount = std::min(requestBudget, missingAttachments.size());
    output.insert(output.end(), missingAttachments.begin(),
                  missingAttachments.begin() + static_cast<std::ptrdiff_t>(missingCount));
    const size_t promotionCount =
        std::min(requestBudget - output.size(), provisionalPromotions.size());
    output.insert(output.end(), provisionalPromotions.begin(),
                  provisionalPromotions.begin() + static_cast<std::ptrdiff_t>(promotionCount));
}

RenderPipeline::RenderPipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                               uint32_t height, uint64_t worldSeed)
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
        } else if (name == "lod") {
            _worldgenOverlayMode = WorldgenOverlayMode::LOD;
        } else if (name == "authority") {
            _worldgenOverlayMode = WorldgenOverlayMode::AUTHORITY;
        } else if (!name.empty()) {
            RY_LOG_ERROR("RYCRAFT_WORLDGEN_OVERLAY must be geology, hydrology, climate, biome, "
                         "lod, or authority");
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

    PixelFormats::configureScenePassPipeline(pipelineDesc);

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    resetMetalObject(pipelineDesc);
    resetMetalObject(vertexFunc);
    resetMetalObject(fragmentFunc);
    if (!_pipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create render pipeline state: %@",
                                                   error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    if (![_device supportsFamily:MTLGPUFamilyApple4] ||
        ![_device supportsTextureSampleCount:PixelFormats::SCENE_SAMPLE_COUNT]) {
        RY_LOG_FATAL("Coherent scene material resolve requires Apple GPU family 4 and 4x MSAA");
    }
    id<MTLFunction> coherentResolveFunction =
        [shaderLibrary newFunctionWithName:@"coherentSceneResolveTileKernel"];
    if (!coherentResolveFunction) {
        RY_LOG_FATAL("Failed to load coherent scene material resolve tile function");
    }
    auto coherentResolveDesc = [[MTLTileRenderPipelineDescriptor alloc] init];
    coherentResolveDesc.label = @"Coherent Nearest-Surface Resolve";
    coherentResolveDesc.tileFunction = coherentResolveFunction;
    coherentResolveDesc.threadgroupSizeMatchesTileSize = YES;
    coherentResolveDesc.rasterSampleCount = PixelFormats::SCENE_SAMPLE_COUNT;
    coherentResolveDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    coherentResolveDesc.colorAttachments[1].pixelFormat = PixelFormats::SURFACE;
    coherentResolveDesc.colorAttachments[2].pixelFormat = PixelFormats::REACTIVE;
    coherentResolveDesc.colorAttachments[3].pixelFormat = PixelFormats::RESOLVE_DEPTH_KEY;
    _coherentResolvePipelineState =
        [_device newRenderPipelineStateWithTileDescriptor:coherentResolveDesc
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:&error];
    resetMetalObject(coherentResolveDesc);
    resetMetalObject(coherentResolveFunction);
    if (!_coherentResolvePipelineState) {
        NSString* msg =
            [NSString stringWithFormat:@"Failed to create coherent scene material resolve: %@",
                                       error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Depth stencil state (opaque) ----
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    resetMetalObject(depthDesc);
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create depth stencil state");
    }

    // ---- Depth-tested, non-writing state (block highlight) ----
    auto noWriteDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    noWriteDepthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    noWriteDepthDesc.depthWriteEnabled = false;
    _noDepthWriteState = [_device newDepthStencilStateWithDescriptor:noWriteDepthDesc];
    resetMetalObject(noWriteDepthDesc);
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
    PixelFormats::configureScenePassPipeline(skyPipelineDesc);
    skyPipelineDesc.colorAttachments[1].writeMask = MTLColorWriteMaskNone;
    skyPipelineDesc.colorAttachments[2].writeMask = MTLColorWriteMaskNone;
    skyPipelineDesc.colorAttachments[3].writeMask = MTLColorWriteMaskNone;

    _skyPipelineState = [_device newRenderPipelineStateWithDescriptor:skyPipelineDesc error:&error];
    resetMetalObject(skyPipelineDesc);
    resetMetalObject(skyVertexFunc);
    resetMetalObject(skyFragmentFunc);
    if (!_skyPipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create sky pipeline state: %@",
                                                   error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // Sky depth state: always pass, never write, the sky sits behind everything
    auto skyDepthDesc = [[MTLDepthStencilDescriptor alloc] init];
    skyDepthDesc.depthCompareFunction = MTLCompareFunctionAlways;
    skyDepthDesc.depthWriteEnabled = false;
    _skyDepthState = [_device newDepthStencilStateWithDescriptor:skyDepthDesc];
    resetMetalObject(skyDepthDesc);
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
    PixelFormats::configureScenePassPipeline(highlightPipelineDesc);
    highlightPipelineDesc.colorAttachments[0].blendingEnabled = true;
    highlightPipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    highlightPipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    highlightPipelineDesc.colorAttachments[0].destinationRGBBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    highlightPipelineDesc.colorAttachments[0].writeMask =
        MTLColorWriteMaskRed | MTLColorWriteMaskGreen | MTLColorWriteMaskBlue;
    highlightPipelineDesc.colorAttachments[1].writeMask = MTLColorWriteMaskNone;
    highlightPipelineDesc.colorAttachments[2].writeMask = MTLColorWriteMaskNone;
    highlightPipelineDesc.colorAttachments[3].writeMask = MTLColorWriteMaskNone;
    _highlightPipelineState = [_device newRenderPipelineStateWithDescriptor:highlightPipelineDesc
                                                                      error:&error];
    resetMetalObject(highlightPipelineDesc);
    resetMetalObject(highlightVertexFunc);
    resetMetalObject(highlightFragmentFunc);
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
        waterDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
        waterDesc.rasterSampleCount = 1;
        _waterPipelineState = [_device newRenderPipelineStateWithDescriptor:waterDesc error:&error];
        resetMetalObject(waterDesc);
        resetMetalObject(waterVertexFunc);
        resetMetalObject(waterFragmentFunc);
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
        overlayDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
        overlayDesc.colorAttachments[0].blendingEnabled = true;
        overlayDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        overlayDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        // Dual-source blending: result = inscatter + scene * transmit. The
        // fragment's color(0) index(0) is the inscattered light and index(1)
        // the per-channel Beer-Lambert transmittance, a single alpha cannot
        // express spectral absorption (red must die faster than blue).
        overlayDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        overlayDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        overlayDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorSource1Color;
        overlayDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
        overlayDesc.rasterSampleCount = 1;
        _underwaterOverlayState = [_device newRenderPipelineStateWithDescriptor:overlayDesc
                                                                          error:&error];
        resetMetalObject(overlayDesc);
        resetMetalObject(overlayVertexFunc);
        resetMetalObject(overlayFragmentFunc);
        if (!_underwaterOverlayState) {
            NSString* msg =
                [NSString stringWithFormat:@"Failed to create underwater overlay pipeline: %@",
                                           error.localizedDescription];
            RY_LOG_FATAL(msg.UTF8String);
        }
    }

    // ---- Lazy segmented arena for exact chunk meshes ----
    // v4 expands its exact radius after safe-spawn qualification. A growable
    // contiguous buffer would invalidate every published spawn allocation at
    // that handoff, so reserve the final bounded capacity and allocate slabs
    // only as exact meshes actually arrive.
    _megaBuffer = std::make_unique<SegmentedMegaBuffer>(
        _device, EXACT_VERTEX_BUFFER_BYTES, EXACT_INDEX_BUFFER_BYTES,
        EXACT_VERTEX_BUFFER_SLAB_BYTES, EXACT_INDEX_BUFFER_SLAB_BYTES);

    // ---- Block textures (procedural, one array layer per face texture) ----
    _blockTextures = std::make_unique<BlockTextureArray>(_device);

    // ---- UIOverlay (screen-space HUD rendering) ----
    _uiOverlay = std::make_unique<UIOverlay>(_device, shaderLibrary, _displayWidth, _displayHeight);
    _uiOverlay->setIconAtlas(_blockTextures->texture(), _blockTextures->sampler());

    // ---- Bloom post-processing (HDR extract + blur) ----
    _bloom = std::make_unique<Bloom>(_device, shaderLibrary, _displayWidth, _displayHeight);
    _bloom->setIntensity(_bloomIntensity);

    // ---- Final composite (tonemap + grade + sharpen) ----
    _postStack = std::make_unique<PostStack>(_device, shaderLibrary);

    // ---- Cascaded shadow maps (share the chunk vertex layout) ----
    _shadowMap = std::make_unique<ShadowMap>(_device, shaderLibrary, vertexDesc);

    // ---- Physical atmosphere, indirect lighting, clouds, and volumetrics ----
    _atmosphere = std::make_unique<AtmosphereRenderer>(_device, shaderLibrary);
    _screenSpaceLighting = std::make_unique<ScreenSpaceLighting>(_device, shaderLibrary,
                                                                 _displayWidth, _displayHeight);
    _clouds = std::make_unique<CloudRenderer>(_device, shaderLibrary, _displayWidth, _displayHeight,
                                              worldSeed);
    _lightning = std::make_unique<LightningRenderer>(_device, shaderLibrary);
    _volumetrics =
        std::make_unique<Volumetrics>(_device, shaderLibrary, _displayWidth, _displayHeight);
    _indirectHistoryState = std::make_unique<IndirectHistoryState>();

    // ---- Weather Particle System ----
    _particles = std::make_unique<ParticleSystem>(_device, shaderLibrary);

    // ---- Animal renderer ----
    _entityRenderer = std::make_unique<EntityRenderer>(_device, shaderLibrary);
    _itemEntityRenderer = std::make_unique<ItemEntityRenderer>(_device, shaderLibrary);
    _boatRenderer = std::make_unique<BoatRenderer>(_device, shaderLibrary);

    // ---- GPU timing (per-pass sampling is a diagnostic opt-in) ----
    const char* counters = std::getenv("RYCRAFT_GPU_COUNTERS");
    _gpuTimer = std::make_unique<GpuFrameTimer>(_device, counters && *counters &&
                                                             std::strcmp(counters, "0") != 0);
}

// ---------------------------------------------------------------------------
// allocateSceneTargets, (re)create the MSAA + resolve textures at the
// current drawable size. MSAA targets are memoryless: their tile contents
// are resolved at pass end (color into _colorResolve, depth into
// _depthResolve for the water pass) and never loaded or stored.
// ---------------------------------------------------------------------------
void RenderPipeline::releaseSceneTargets() {
    resetMetalObject(_colorMSAA);
    resetMetalObject(_surfaceMSAA);
    resetMetalObject(_reactiveMSAA);
    resetMetalObject(_resolveDepthKeyMSAA);
    resetMetalObject(_depthMSAA);
    resetMetalObject(_colorResolve);
    resetMetalObject(_surfaceResolve);
    resetMetalObject(_reactiveResolve);
    resetMetalObject(_depthResolve);
    resetMetalObject(_mediaDepthResolve);
    resetMetalObject(_sceneColorCopy);
}

void RenderPipeline::allocateSceneTargets() {
    releaseSceneTargets();
    auto colorMSAADesc = [[MTLTextureDescriptor alloc] init];
    colorMSAADesc.textureType = MTLTextureType2DMultisample;
    colorMSAADesc.pixelFormat = PixelFormats::SCENE_HDR;
    colorMSAADesc.width = _displayWidth;
    colorMSAADesc.height = _displayHeight;
    colorMSAADesc.sampleCount = PixelFormats::SCENE_SAMPLE_COUNT;
    colorMSAADesc.usage = MTLTextureUsageRenderTarget;
    colorMSAADesc.storageMode = MTLStorageModeMemoryless;
    _colorMSAA = [_device newTextureWithDescriptor:colorMSAADesc];
    if (!_colorMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA color texture");
    }

    MTLTextureDescriptor* surfaceMSAADesc = [colorMSAADesc copy];
    surfaceMSAADesc.pixelFormat = PixelFormats::SURFACE;
    _surfaceMSAA = [_device newTextureWithDescriptor:surfaceMSAADesc];
    MTLTextureDescriptor* reactiveMSAADesc = [colorMSAADesc copy];
    reactiveMSAADesc.pixelFormat = PixelFormats::REACTIVE;
    _reactiveMSAA = [_device newTextureWithDescriptor:reactiveMSAADesc];
    MTLTextureDescriptor* resolveDepthKeyMSAADesc = [colorMSAADesc copy];
    resolveDepthKeyMSAADesc.pixelFormat = PixelFormats::RESOLVE_DEPTH_KEY;
    _resolveDepthKeyMSAA = [_device newTextureWithDescriptor:resolveDepthKeyMSAADesc];
    resetMetalObject(resolveDepthKeyMSAADesc);
    resetMetalObject(reactiveMSAADesc);
    resetMetalObject(surfaceMSAADesc);
    resetMetalObject(colorMSAADesc);
    if (!_surfaceMSAA) {
        RY_LOG_FATAL("Failed to allocate MSAA surface-data texture");
    }
    if (!_reactiveMSAA || !_resolveDepthKeyMSAA) {
        RY_LOG_FATAL("Failed to allocate coherent MSAA material targets");
    }

    auto depthMSAADesc = [[MTLTextureDescriptor alloc] init];
    depthMSAADesc.textureType = MTLTextureType2DMultisample;
    depthMSAADesc.pixelFormat = PixelFormats::SCENE_DEPTH;
    depthMSAADesc.width = _displayWidth;
    depthMSAADesc.height = _displayHeight;
    depthMSAADesc.sampleCount = PixelFormats::SCENE_SAMPLE_COUNT;
    depthMSAADesc.usage = MTLTextureUsageRenderTarget;
    depthMSAADesc.storageMode = MTLStorageModeMemoryless;
    _depthMSAA = [_device newTextureWithDescriptor:depthMSAADesc];
    resetMetalObject(depthMSAADesc);
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

    auto surfaceResolveDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SURFACE
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:false];
    surfaceResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    surfaceResolveDesc.storageMode = MTLStorageModePrivate;
    _surfaceResolve = [_device newTextureWithDescriptor:surfaceResolveDesc];
    if (!_surfaceResolve) {
        RY_LOG_FATAL("Failed to allocate surface-data resolve texture");
    }

    auto reactiveResolveDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::REACTIVE
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:false];
    reactiveResolveDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    reactiveResolveDesc.storageMode = MTLStorageModePrivate;
    _reactiveResolve = [_device newTextureWithDescriptor:reactiveResolveDesc];
    if (!_reactiveResolve) {
        RY_LOG_FATAL("Failed to allocate reactive resolve texture");
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
    _mediaDepthResolve = [_device newTextureWithDescriptor:depthResolveDesc];
    _mediaDepthResolve.label = @"Opaque and Water Media Depth";
    if (!_mediaDepthResolve) {
        RY_LOG_FATAL("Failed to allocate media depth texture");
    }

    // Refraction samples level zero of a copy of the resolved color (a render
    // target cannot sample itself). SSR additionally uses the complete HDR
    // mip pyramid so long grazing reflections cannot minify sharp distant
    // geometry into a checkerboard of hit-or-sky decisions.
    auto sceneCopyDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SCENE_HDR
                                                           width:_displayWidth
                                                          height:_displayHeight
                                                       mipmapped:true];
    sceneCopyDesc.usage = MTLTextureUsageShaderRead;
    sceneCopyDesc.storageMode = MTLStorageModePrivate;
    _sceneColorCopy = [_device newTextureWithDescriptor:sceneCopyDesc];
    _sceneColorCopy.label = @"Water Reflection and Refraction Pyramid";
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
                            double deltaSeconds, std::optional<BlockHighlight> highlightedBlock,
                            const UIFrameState& uiFrame,
                            const std::vector<std::shared_ptr<Entity>>* entities,
                            const std::vector<ItemEntity>* itemEntities,
                            const std::vector<Boat>* boats,
                            std::shared_ptr<const WeatherSnapshot> weatherSnapshot,
                            const std::vector<LightningEvent>* lightningEvents) {
    if (!drawable || !queue)
        return;

    _clouds->beginWorld(world.instanceId(), world.getSeed());
    _particles->beginWorld(world.instanceId(), world.getSeed());

    // Track the true drawable size (pixels, not view points, 2x on Retina)
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
    // basis, computeDayNightUniforms only knows the time of day).
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

    const Vec3 cameraPosition = camera.getPosition();
    const WorldPhysicalScale physicalScale =
        worldPhysicalScale(world.generator().usesLearnedAuthority());
    WeatherSample localWeather{};
    if (weatherSnapshot) {
        localWeather = weatherSnapshot->sample(cameraPosition.x, cameraPosition.z, worldTime);
    }
    const FoliageWindUniforms foliageWind = makeFoliageWindUniforms(
        localWeather.windBlocksPerSecond.x, localWeather.windBlocksPerSecond.y, _gfx.wavingFoliage);
    _clouds->setQuality(_gfx.cloudQuality);

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer)
        return;

    _gpuTimer->beginFrame();

    // Claim a frames-in-flight slot: every per-frame uniform block below
    // sub-allocates from it, so the CPU never rewrites data the GPU reads.
    _frameRing.waitAndBegin();
    if (weatherSnapshot) {
        _clouds->updateWeather(*weatherSnapshot, worldTime,
                               static_cast<uint32_t>(_frameRing.frameIndex()));
    }
    FrameRing::Alloc skyAlloc = _frameRing.push(&skyUniforms, sizeof(SkyUniforms));

    // Slow atmosphere LUTs and the snapped cloud transmittance map precede
    // every geometry pass that samples them.
    const AtmosphereUniforms atmosphereUniforms = earthAtmosphereUniforms(
        cameraPosition.y, physicalScale, skyUniforms.sunDirection,
        simd_make_float3(1.0F, 1.0F, 0.98F), std::max(localWeather.aerosolDensity, 0.08F),
        localWeather.relativeHumidity, static_cast<uint32_t>(_frameRing.frameIndex()));
    _atmosphere->encode(commandBuffer, atmosphereUniforms, false, _gpuTimer.get());
    if (weatherSnapshot && _gfx.cloudQuality > 0 && shadowStrength > 0.0001F) {
        CloudShadowUniforms cloudShadow{};
        cloudShadow.cameraPosition =
            simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
        cloudShadow.sunDirection =
            simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
        cloudShadow.footprintAndTexel =
            simd_make_float4(16'384.0F, 0.0F, 1.0F, static_cast<float>(_frameRing.frameIndex()));
        _clouds->encodeShadow(commandBuffer, cloudShadow, _gpuTimer.get());
    }

    // ---- Scene pass: sky, opaque chunks, entities, and highlights ----
    // Geometry renders into the 4x MSAA targets and resolves once into the
    // HDR color, linear-depth source, and surface-data attachments.
    auto renderPassDesc = [[MTLRenderPassDescriptor alloc] init];
    renderPassDesc.colorAttachments[0].texture = _colorMSAA;
    renderPassDesc.colorAttachments[0].resolveTexture = _colorResolve;
    renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(
        skyUniforms.horizonColor.x, skyUniforms.horizonColor.y, skyUniforms.horizonColor.z, 1.0f);
    renderPassDesc.colorAttachments[1].texture = _surfaceMSAA;
    renderPassDesc.colorAttachments[1].resolveTexture = _surfaceResolve;
    renderPassDesc.colorAttachments[1].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[1].storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.colorAttachments[1].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPassDesc.colorAttachments[2].texture = _reactiveMSAA;
    renderPassDesc.colorAttachments[2].resolveTexture = _reactiveResolve;
    renderPassDesc.colorAttachments[2].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[2].storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.colorAttachments[2].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    renderPassDesc.colorAttachments[3].texture = _resolveDepthKeyMSAA;
    renderPassDesc.colorAttachments[3].loadAction = MTLLoadActionClear;
    renderPassDesc.colorAttachments[3].storeAction = MTLStoreActionDontCare;
    renderPassDesc.colorAttachments[3].clearColor = MTLClearColorMake(1.0, 0.0, 0.0, 0.0);

    // Depth resolves out of tile memory (min filter: nearest sample) so the
    // water pass can depth-test and reconstruct world positions.
    renderPassDesc.depthAttachment.texture = _depthMSAA;
    renderPassDesc.depthAttachment.resolveTexture = _depthResolve;
    renderPassDesc.depthAttachment.loadAction = MTLLoadActionClear;
    renderPassDesc.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
    renderPassDesc.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterMin;
    renderPassDesc.depthAttachment.clearDepth = 1.0;
    renderPassDesc.tileWidth = 16;
    renderPassDesc.tileHeight = 16;
    _gpuTimer->attachPass(renderPassDesc, "scene");

    // One immutable tick snapshot feeds shadows, exact terrain, and entities.
    // The render thread never copies or locks the cubic chunk map.
    const auto loadedSnapshot = world.getLoadedSnapshot();
    static const std::vector<std::shared_ptr<Chunk>> emptyChunks;
    const auto& loadedChunks = loadedSnapshot ? *loadedSnapshot : emptyChunks;

    // ---- Shadow cascades (depth-only passes before the scene pass) ----
    renderShadows(commandBuffer, loadedChunks, camera, sunDirection, shadowStrength, foliageWind,
                  entities, itemEntities, boats);

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];
    if (!encoder) {
        resetMetalObject(renderPassDesc);
        _frameRing.cancelFrame(); // nothing encoded references the slot
        return;
    }

    renderSky(encoder, skyAlloc);

    // Underwater the whole scene sinks into a dense blue veil (light
    // attenuation), owned entirely by the underwater overlay's depth-based
    // scattering, so the scene/water passes apply no fog of their own below the
    // surface (two fogs stacked over-darkened the near water).
    const bool cameraUnderwater = uiFrame.cameraUnderwater;
    _cameraUnderwater = cameraUnderwater;

    // Sky exposure of the camera's water column: 0 when solid ground seals it
    // (aquifers, roofed lakes, the same surface-height gate rain spawning
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
    // Air extinction belongs exclusively to the froxel or LUT aerial-
    // perspective path later in the frame. The geometry and water shaders
    // keep their compatibility fog inputs neutral so atmospheric density is
    // never applied twice. Underwater absorption remains a separate medium.
    const float savedFogDensity = _fogDensity;
    _fogDensity = 0.0F;
    renderChunks(encoder, world, loadedChunks, viewMatrix, projectionMatrix, camera, sunDirection,
                 sunColor, ambientColor, fogColor, foliageWind);

    if (entities && _entityRenderer) {
        _entityRenderer->render(encoder, _frameUniforms.buffer, _frameUniforms.offset, *entities,
                                [this](const AABB& aabb) { return isChunkInFrustum(aabb); });
    }
    if (itemEntities && _itemEntityRenderer) {
        _itemEntityRenderer->render(encoder, _frameUniforms.buffer, _frameUniforms.offset,
                                    *itemEntities,
                                    [this](const AABB& aabb) { return isChunkInFrustum(aabb); });
    }
    if (boats && _boatRenderer) {
        _boatRenderer->render(encoder, _frameUniforms.buffer, _frameUniforms.offset, *boats,
                              [this](const AABB& aabb) { return isChunkInFrustum(aabb); });
    }

    if (highlightedBlock.has_value()) {
        renderBlockHighlight(encoder, highlightedBlock.value(), viewMatrix, projectionMatrix);
    }

    [encoder setRenderPipelineState:_coherentResolvePipelineState];
    [encoder dispatchThreadsPerTile:MTLSizeMake(16, 16, 1)];

    [encoder endEncoding];
    resetMetalObject(renderPassDesc);

    // ---- GTAO, ambient accessibility, and near-field diffuse SSGI ----
    simd_float4x4 projection{};
    simd_float4x4 viewProjection{};
    std::memcpy(&projection, projectionMatrix.data.data(), sizeof(projection));
    std::memcpy(&viewProjection, vpMatrix.data.data(), sizeof(viewProjection));
    if (indirectLightingTimeDiscontinuity(_hasPreviousWorldTime, _previousWorldTime, worldTime)) {
        ++_indirectTimeDiscontinuityRevision;
    }
    if (_hasPreviousWorldTime &&
        (worldTime < _previousWorldTime || worldTime > _previousWorldTime + 4U)) {
        ++_forcedStateRevision;
    }
    const bool weatherSnapshotPresent = weatherSnapshot != nullptr;
    if (weatherSnapshotPresent != _weatherSnapshotWasPresent) {
        ++_forcedStateRevision;
    }
    if (weatherSnapshot) {
        const uint8_t weatherPreset = static_cast<uint8_t>(weatherSnapshot->preset());
        if (_weatherSnapshotWasPresent && weatherPreset != _previousWeatherPreset) {
            ++_forcedStateRevision;
        }
        _previousWeatherPreset = weatherPreset;
    }
    _weatherSnapshotWasPresent = weatherSnapshotPresent;
    IndirectHistoryState currentHistory{};
    currentHistory.width = _displayWidth;
    currentHistory.height = _displayHeight;
    currentHistory.cameraPosition = cameraPosition;
    currentHistory.fovDegrees = camera.FOV();
    currentHistory.worldIdentity = world.instanceId();
    currentHistory.forcedStateRevision = _forcedStateRevision;
    currentHistory.lightEditRevision =
        indirectLightingRevision(world.lightingRevision(), _exactMaterialPublicationRevision);
    currentHistory.timeDiscontinuityRevision = _indirectTimeDiscontinuityRevision;
    currentHistory.quality = _gfx.indirectLightingQuality;
    currentHistory.directLightSource = _activeCelestialSource;
    // Ambient-only mode has no SSGI depth history to invalidate. Treat that
    // state as stable so disabling indirect light cannot erase the independent
    // cloud and froxel temporal histories every frame.
    currentHistory.priorDepthValid =
        _gfx.indirectLightingQuality == 0 || _screenSpaceLighting->historyValid();
    const uint32_t historyReset = indirectHistoryResetMask(*_indirectHistoryState, currentHistory);
    _indirectHistoryResetMask = historyReset;
    _screenSpaceLighting->setQuality(_gfx.indirectLightingQuality);
    if (historyReset != INDIRECT_HISTORY_STABLE) {
        _screenSpaceLighting->resetHistory();
    }
    const uint32_t atmosphereHistoryReset = atmosphericHistoryResetMask(historyReset);
    if (atmosphereHistoryReset != INDIRECT_HISTORY_STABLE) {
        _clouds->resetHistory();
        _volumetrics->resetHistory();
    }
    *_indirectHistoryState = currentHistory;

    IndirectLightingUniforms indirect{};
    indirect.projection = projection;
    indirect.invProjection = simd_inverse(projection);
    indirect.invViewProjection = simd_inverse(viewProjection);
    indirect.previousViewProjection = _previousViewProjection;
    // The final term calibrates one visible bounce against our unit direct
    // sunlight. Keep it below direct illumination while preserving a readable
    // spill from a sunlit voxel into an otherwise dark cave entrance.
    indirect.traceParams = simd_make_float4(8.0F, 0.15F, 0.70F, 0.65F);
    // Color history clamps at two accumulated standard deviations so a
    // sparse-but-real bright source survives, while ambient occlusion clamps
    // at one so disocclusion darkening reacts within a frame.
    indirect.temporalParams = simd_make_float4(0.90F, 2.0F, 1.0F, 4.0F);
    // The day-night sky level (1 in daylight, 0 deep at night) scales the
    // additive SSGI bounce so night auto-exposure cannot amplify the near-field
    // one-bounce into a camera-following ground disk. It reuses the same
    // celestial signal the water tint fades by, so the two never drift.
    const float dayNightSkyLevel = std::clamp(1.0F - skyUniforms.visibilityAndPhase.w, 0.0F, 1.0F);
    indirect.filterParams = simd_make_float4(_gfx.indirectLightingQuality >= 2 ? 24.0F : 16.0F,
                                             4.0F, 4.0F, dayNightSkyLevel);
    indirect.ambientAndFrame = simd_make_float4(ambientColor[0], ambientColor[1], ambientColor[2],
                                                static_cast<float>(_frameRing.frameIndex()));
    _screenSpaceLighting->encode(commandBuffer, _colorResolve, _depthResolve, _surfaceResolve,
                                 _reactiveResolve, indirect, _gpuTimer.get());

    // ---- Quarter-resolution physical volumetric clouds ----
    if (weatherSnapshot && _gfx.cloudQuality > 0) {
        CloudRenderUniforms cloud{};
        cloud.invViewProjection = simd_inverse(viewProjection);
        cloud.previousViewProjection = _previousViewProjection;
        cloud.cameraPosition =
            simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
        cloud.cameraForward = skyUniforms.cameraForward;
        cloud.sunDirection = simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
        cloud.sunRadiance = simd_make_float3(sunColor[0], sunColor[1], sunColor[2]);
        cloud.skyIrradiance = simd_make_float3(ambientColor[0], ambientColor[1], ambientColor[2]);
        const simd_float2 cloudLayerBounds = cloudSnapshotMarchLayerBounds(*weatherSnapshot);
        cloud.layerBounds = simd_make_float4(cloudLayerBounds.x, cloudLayerBounds.y, 16.0F,
                                             CLOUD_HORIZON_VIEW_DEPTH);
        cloud.densityParams = simd_make_float4(1.0F, 0.38F, 0.035F, 0.18F);
        cloud.phaseParams = simd_make_float4(0.78F, -0.20F, 0.82F, 0.0F);
        cloud.resolutionAndFrame.z = static_cast<float>(_frameRing.frameIndex());
        _clouds->encode(commandBuffer, _colorResolve, _depthResolve, cloud, _gpuTimer.get());
    }

    // ---- Deterministic cloud-aware lightning ----
    if (lightningEvents && !lightningEvents->empty()) {
        _lightning->encode(commandBuffer, _colorResolve, _depthResolve, _clouds->resolvedHitDepth(),
                           viewProjection,
                           simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z),
                           *lightningEvents, worldTime, 20.0F, _gpuTimer.get());
        _lastLightningEventId = _lightning->stats().lastEventId;
    } else {
        _lastLightningEventId = 0;
    }

    // ---- Water pass (refraction/reflection/caustics over the resolved scene) ----
    renderWater(commandBuffer, viewMatrix, projectionMatrix, camera.getPosition(), cameraUnderwater,
                skyUniforms, sunDirection, sunColor, fogColor);
    _fogDensity = savedFogDensity;

    // ---- Unified air medium (over opaque, clouds, and water) ----
    // The same weather, atmosphere, terrain-shadow, and cloud-shadow state
    // feeds fog and shafts. Disabling volumetric lighting retains the cheap
    // LUT aerial-perspective path. Underwater absorption remains exclusively
    // in the water renderer.
    FroxelUniforms froxel{};
    froxel.invViewProjection = simd_inverse(viewProjection);
    froxel.previousViewProjection = _previousViewProjection;
    froxel.viewProjection = viewProjection;
    froxel.cameraPosition = simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    froxel.lightDirection = simd_make_float3(sunDirection[0], sunDirection[1], sunDirection[2]);
    froxel.lightRadiance = simd_make_float3(sunColor[0], sunColor[1], sunColor[2]);
    froxel.solarDirection = skyUniforms.sunDirection;
    froxel.volumeDimensions = simd_make_uint4(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH,
                                              static_cast<uint32_t>(_frameRing.frameIndex()));
    froxel.depthParams = simd_make_float4(0.1F, SHADOW_HORIZON_DISTANCE, 16'384.0F,
                                          cameraUnderwater ? _uwSurfaceY : -65'536.0F);
    froxel.mediumParams = simd_make_float4(std::max(savedFogDensity, 0.0F), 0.92F, 0.62F, 800.0F);
    froxel.weatherParams =
        simd_make_float4(std::max(localWeather.aerosolDensity, 0.0F),
                         std::clamp(localWeather.relativeHumidity, 0.0F, 1.0F),
                         std::clamp(localWeather.precipitationIntensity, 0.0F, 1.0F),
                         std::max(localWeather.fogExtinction, 0.0F));
    if (weatherSnapshot) {
        froxel.weatherMap = _clouds->weatherMapForCamera(froxel.cameraPosition);
    }
    froxel.renderParams =
        simd_make_float4(0.90F, atmosphereHistoryReset == INDIRECT_HISTORY_STABLE ? 1.0F : 0.0F,
                         shadowStrength, cameraUnderwater ? 1.0F : 0.0F);
    froxel.physicalScale =
        simd_make_float4(static_cast<float>(physicalScale.horizontalMetersPerBlock),
                         static_cast<float>(physicalScale.positiveVerticalMetersPerBlock),
                         static_cast<float>(physicalScale.altitudeDatumY), 0.0F);
    _volumetrics->encode(
        commandBuffer, _colorResolve, _mediaDepthResolve, _shadowMap->nearDepthTexture(),
        _shadowMap->farDepthTexture(), _shadowMap->horizonDepthTexture(),
        _atmosphere->skyViewTexture(), _clouds ? _clouds->shadowTexture() : nil,
        _clouds ? _clouds->resolvedHitDepth() : nil,
        weatherSnapshot ? _clouds->weatherCloudTexture() : nil,
        weatherSnapshot ? _clouds->weatherLayerTexture() : nil, _shadowMap->comparisonSampler(),
        froxel, _sceneShadowUniforms, _clouds ? _clouds->shadowUniforms() : CloudShadowUniforms{},
        _gfx.volumetricLight, _gpuTimer.get());

    // ---- Depth-tested weather particles ----
    // Draw precipitation after the air medium so particles receive analytic
    // atmospheric attenuation without being fogged a second time.
    if (_particles && weatherParticlesVisible(cameraUnderwater)) {
        auto weatherPass = [[MTLRenderPassDescriptor alloc] init];
        weatherPass.colorAttachments[0].texture = _colorResolve;
        weatherPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        weatherPass.colorAttachments[0].storeAction = MTLStoreActionStore;
        weatherPass.depthAttachment.texture = _mediaDepthResolve;
        weatherPass.depthAttachment.loadAction = MTLLoadActionLoad;
        weatherPass.depthAttachment.storeAction = MTLStoreActionStore;
        _gpuTimer->attachPass(weatherPass, "weatherParticles");
        id<MTLRenderCommandEncoder> weatherEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:weatherPass];
        if (weatherEncoder) {
            weatherEncoder.label = @"Atmospherically Attenuated Weather";
            _particles->render(weatherEncoder, _frameRing, viewMatrix, projectionMatrix,
                               cameraPosition, localWeather, savedFogDensity, physicalScale);
            [weatherEncoder endEncoding];
        }
        resetMetalObject(weatherPass);
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
            // Fade out as the sun crosses the screen edge, the probe has no
            // depth data outside the frame, and a hard cut would pop.
            float edge =
                std::min(std::min(sunUV.x, 1.0f - sunUV.x), std::min(sunUV.y, 1.0f - sunUV.y));
            float fade = std::clamp((edge + 0.1f) / 0.2f, 0.0f, 1.0f);
            flareStrength = skyUniforms.visibilityAndPhase.x * fade;
        }
    }
    if (flareStrength > 0.0f) {
        _postStack->encodeFlareProbe(commandBuffer, _mediaDepthResolve,
                                     _gfx.cloudQuality > 0 ? _clouds->resolvedCloudTexture() : nil,
                                     sunUV);
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
        renderUIOverlay(uiEncoder, uiFrame);
        [uiEncoder endEncoding];
    }
    resetMetalObject(uiPassDesc);

    // ---- Optional frame capture (playtest verification) ----
    if (!_capturePath.empty()) {
        encodeFrameCapture(commandBuffer, drawable.texture);
    }

    _gpuTimer->endFrame(commandBuffer);
    _previousViewProjection = viewProjection;
    _previousWorldTime = worldTime;
    _hasPreviousWorldTime = true;
    _frameRing.signalOnCompletion(commandBuffer);

    // Present and commit
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

void RenderPipeline::renderMenuOnly(id<MTLCommandQueue> queue, id<CAMetalDrawable> drawable,
                                    const UIFrameState& uiFrame) {
    if (!drawable || !queue)
        return;

    if (drawable.texture.width != _displayWidth || drawable.texture.height != _displayHeight) {
        resize(static_cast<uint32_t>(drawable.texture.width),
               static_cast<uint32_t>(drawable.texture.height));
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer)
        return;

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.06, 0.07, 0.10, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (encoder) {
        renderUIOverlay(encoder, uiFrame);
        [encoder endEncoding];
    }

    // Menu screens are capturable too (world selection playtests).
    if (!_capturePath.empty()) {
        encodeFrameCapture(commandBuffer, drawable.texture);
    }

    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

void publishV4PreparationWorldSnapshot(World& world) {
    world.publishLoadedSnapshot();
}

void RenderPipeline::renderV4Preparation(id<MTLCommandQueue> queue,
                                         id<CAMetalDrawable> drawable,
                                         const UIFrameState& uiFrame, World& world,
                                         const Camera& camera) {
    if (!drawable || !queue)
        return;

    if (drawable.texture.width != _displayWidth || drawable.texture.height != _displayHeight) {
        resize(static_cast<uint32_t>(drawable.texture.width),
               static_cast<uint32_t>(drawable.texture.height));
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer)
        return;

    _frameRing.waitAndBegin();
    publishV4PreparationWorldSnapshot(world);
    const auto loadedSnapshot = world.getLoadedSnapshot();
    static const std::vector<std::shared_ptr<Chunk>> emptyChunks;
    const auto& loadedChunks = loadedSnapshot ? *loadedSnapshot : emptyChunks;
    constexpr float LIGHT_DIRECTION[3] = {0.0F, 1.0F, 0.0F};
    constexpr float WHITE[3] = {1.0F, 1.0F, 1.0F};
    constexpr float PREPARATION_FOG[3] = {0.53F, 0.81F, 0.92F};
    const FoliageWindUniforms noWind{};
    const Mat4 identity = Mat4::identity();
    renderChunks(nil, world, loadedChunks, identity, identity, camera, LIGHT_DIRECTION, WHITE,
                 WHITE, PREPARATION_FOG, noWind, false, true);

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.06, 0.07, 0.10, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (!encoder) {
        _frameRing.cancelFrame();
        return;
    }
    renderUIOverlay(encoder, uiFrame);
    [encoder endEncoding];

    if (!_capturePath.empty()) {
        encodeFrameCapture(commandBuffer, drawable.texture);
    }
    _frameRing.signalOnCompletion(commandBuffer);
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
}

void RenderPipeline::endWorldSession() {
    if (_meshScheduler) {
        _meshScheduler->shutdown();
        _meshScheduler.reset();
    }
    _pendingResults.clear();
    if (_megaBuffer) {
        for (auto& [key, state] : _chunkMeshes) {
            (void)key;
            if (state.uploaded) {
                _megaBuffer->deferFree(state.alloc, _frameRing.frameIndex());
            }
        }
    }
    _chunkMeshes.clear();
    _pendingExactLightingPublications.clear();
    _observedWorldLightingRevision = 0;
    _exactMaterialPublicationRevision = 0;
    _exactLightingPublicationBatchPending = false;
    _exactLightingPublicationCompleted = false;
    _liveChunksByPosition.clear();
    clearExactSectionOwnership();
    _exactShadowOwnedSections.clear();
    _waterDraws.clear();

    // Stop both far-terrain lanes and release every World-keyed residency
    // record. A later world with the same seed still receives a new scheduler
    // and generation context because the instance identity is also cleared.
    if (_farTerrainScheduler) {
        _farTerrainScheduler->shutdown();
        _farTerrainScheduler.reset();
    }
    if (_farMegaBuffer) {
        for (auto& [key, state] : _farTerrainMeshes) {
            (void)key;
            if (state.uploaded) {
                _farMegaBuffer->deferFree(state.alloc, _frameRing.frameIndex());
            }
        }
        for (auto& [key, state] : _farCanopyAttachments) {
            (void)key;
            if (state.alloc) {
                _farMegaBuffer->deferFree(*state.alloc, _frameRing.frameIndex());
            }
        }
        for (auto& [key, transition] : _farTerrainAuthorityTransitions) {
            (void)key;
            if (transition.source.uploaded) {
                _farMegaBuffer->deferFree(transition.source.alloc, _frameRing.frameIndex());
            }
            if (transition.sourceCanopy && transition.sourceCanopy->alloc) {
                _farMegaBuffer->deferFree(*transition.sourceCanopy->alloc, _frameRing.frameIndex());
            }
        }
    }
    _farTerrainMeshes.clear();
    _farCanopyAttachments.clear();
    _farTerrainSeed.reset();
    _farTerrainWorldInstanceId.reset();
    _farTerrainCenterTile.reset();
    _farTerrainWanted.clear();
    _farTerrainPriorityOrder.clear();
    _farTerrainActiveTiles.clear();
    _farTerrainDesiredByTile.clear();
    _farTerrainDisplayedByTile.clear();
    _farTerrainTransitions.clear();
    _farCanopyLodFallbacks.clear();
    _farTerrainAuthorityTransitions.clear();
    _farShadowDrawPlans.clear();
    _farTerrainNearGraceStartedAt.clear();
    _farTerrainResults.clear();
    _farCanopyResults.clear();
    _farTerrainSelectionCamera.reset();
    _farTerrainDesiredMetricsCamera.reset();
    _farTerrainSpeculativeCamera.reset();
    _farTerrainSelectionViewDistance = -1;
    _farTerrainProtectedNearHandoff.clear();
    _farTerrainProtectedNearEpoch = 0;
    _farTerrainLocalTerrainDebt = true;
    _farTerrainProtectedRecentMotionX = 0;
    _farTerrainProtectedRecentMotionZ = 0;
    _farTerrainPredictedNearAnchor.reset();
    _farTerrainProtectedNearClosureSnapshot.reset();
    _farTerrainDesiredViewportHeight = 0;
    _farTerrainDesiredVerticalFovRadians = 0.0;
    _farTerrainDesiredDrawGeometry = false;
    _farTerrainDesiredMetricsDirty = true;
    _farTerrainViewEpoch = 0;
    _farTerrainCandidates.clear();
    _farTerrainCachedBaseRequests.clear();
    _farTerrainMissingBaseRequests.clear();
    _farTerrainDistantBaseRequests.clear();
    _farTerrainFinalBaseRequests.clear();
    _farTerrainPerceptualFinalRequests.clear();
    _farTerrainFinalRefinementRequests.clear();
    _farTerrainCanopyRefreshRequests.clear();
    _farTerrainCanopyRefreshKeys.clear();
    _farTerrainUrgentRefinementRequests.clear();
    _farTerrainUrgentRefinementKeys.clear();
    _farTerrainConnectedNearPatchTargets.clear();
    _farTerrainProtectedFinalTerrainRegions.clear();
    _farTerrainPredictedNearPatchTargets.clear();
    _farTerrainPredictedCriticalResidencyKeys.clear();
    _farTerrainCriticalResidencyCoordinates.clear();
    _farTerrainCriticalResidencyTargets.clear();
    _farTerrainCriticalResidencyCoordinateScratch.clear();
    _farTerrainCriticalResidencyTargetScratch.clear();
    _farTerrainCriticalResidencyKeys.clear();
    _farTerrainRefinementSubmissionKeys.clear();
    _farTerrainCachedMeshes.clear();
    _farTerrainCachedCanopies.clear();
    _farTerrainResidentWantedCount = 0;
    _farTerrainResidentRefinementCount = 0;
    _farTerrainPlannerTimings.clear();
    _farTerrainSelectionTimings.clear();
    _farTerrainPublicationTimings.clear();
    _farTerrainResidencyTimings.clear();
    _farTerrainArenaAdmissionDeniedCount = 0;
    _farTerrainNearArenaReclaimCount = 0;
    _farTerrainNearArenaReclaimedBytes = 0;
    _farShadowDrawPlans.clear();
    _chunkStats = {};
    if (_screenSpaceLighting)
        _screenSpaceLighting->resetHistory();
    if (_clouds)
        _clouds->endWorld();
    if (_volumetrics)
        _volumetrics->resetHistory();
    if (_particles)
        _particles->endWorld();
    if (_postStack)
        _postStack->resetHistory();
    if (_indirectHistoryState)
        *_indirectHistoryState = IndirectHistoryState{};
    _previousViewProjection = matrix_identity_float4x4;
    _previousWorldTime = 0;
    _hasPreviousWorldTime = false;
    _weatherSnapshotWasPresent = false;
    _previousWeatherPreset = 0;
    _animTime = 0.0F;
    _animClock = 0.0;
    ++_forcedStateRevision;
    ++_indirectTimeDiscontinuityRevision;
}

void RenderPipeline::cancelV4Preparation() {
    if (_farTerrainScheduler) {
        _farTerrainScheduler->cancelViewPreparation();
    }
    _farTerrainSelectionCamera.reset();
    _farTerrainDesiredMetricsCamera.reset();
    _farTerrainSpeculativeCamera.reset();
    _farTerrainSelectionViewDistance = -1;
    _farTerrainCenterTile.reset();
    _farTerrainProtectedNearHandoff.clear();
    _farTerrainProtectedNearEpoch = 0;
    _farTerrainProtectedRecentMotionX = 0;
    _farTerrainProtectedRecentMotionZ = 0;
    _farTerrainPredictedNearAnchor.reset();
    _farTerrainPredictedNearPatchTargets.clear();
    _farTerrainPredictedCriticalResidencyKeys.clear();
    _farTerrainProtectedNearClosureSnapshot.reset();
    _farTerrainDesiredMetricsDirty = true;
    _farTerrainResults.clear();
    _farCanopyResults.clear();
    _chunkStats = {};
}

// ---------------------------------------------------------------------------
// Frame capture, copy the finished drawable into a shared texture and write
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
    const CelestialState celestial = computeCelestialState(worldTime);
    _lunarPhaseEnergy = celestial.phaseEnergy;
    _lunarPhaseCycle = celestial.phaseCycle;
    _directSpecularFactor = celestial.directSpecularFactor;
    _activeCelestialSource = static_cast<uint8_t>(celestial.directSource);
    sunDirection[0] = celestial.directLightDirection.x;
    sunDirection[1] = celestial.directLightDirection.y;
    sunDirection[2] = celestial.directLightDirection.z;
    sunColor[0] = celestial.directLightRadiance.x;
    sunColor[1] = celestial.directLightRadiance.y;
    sunColor[2] = celestial.directLightRadiance.z;
    ambientColor[0] = celestial.ambientRadiance.x;
    ambientColor[1] = celestial.ambientRadiance.y;
    ambientColor[2] = celestial.ambientRadiance.z;
    shadowStrength = celestial.shadowStrength;

    const float daylight = std::clamp((celestial.sunDirection.y + 0.10F) / 0.45F, 0.0F, 1.0F);
    const float night = celestial.starVisibility;
    const simd_float3 twilightZenith = simd_make_float3(0.15F, 0.10F, 0.30F);
    const simd_float3 twilightHorizon = simd_make_float3(0.60F, 0.30F, 0.20F);
    const simd_float3 dayZenith = simd_make_float3(0.20F, 0.40F, 0.80F);
    const simd_float3 dayHorizon = simd_make_float3(0.53F, 0.81F, 0.92F);
    const simd_float3 nightZenith = simd_make_float3(0.02F, 0.02F, 0.05F);
    const simd_float3 nightHorizon = simd_make_float3(0.05F, 0.05F, 0.10F);
    const auto mixColor = [](simd_float3 from, simd_float3 to, float amount) {
        return from + (to - from) * amount;
    };

    skyUniforms.sunDirection = simd_make_float3(celestial.sunDirection.x, celestial.sunDirection.y,
                                                celestial.sunDirection.z);
    skyUniforms.moonDirection = simd_make_float3(
        celestial.moonDirection.x, celestial.moonDirection.y, celestial.moonDirection.z);
    skyUniforms.sunColor =
        simd_make_float3(celestial.solarDiscRadiance.x, celestial.solarDiscRadiance.y,
                         celestial.solarDiscRadiance.z);
    skyUniforms.moonColor =
        simd_make_float3(celestial.lunarDiscRadiance.x, celestial.lunarDiscRadiance.y,
                         celestial.lunarDiscRadiance.z);
    skyUniforms.zenithColor =
        mixColor(mixColor(twilightZenith, dayZenith, daylight), nightZenith, night);
    skyUniforms.horizonColor =
        mixColor(mixColor(twilightHorizon, dayHorizon, daylight), nightHorizon, night);
    skyUniforms.visibilityAndPhase =
        simd_make_float4(celestial.sunVisibility, celestial.moonVisibility, celestial.phaseEnergy,
                         celestial.starVisibility);
}

// ---------------------------------------------------------------------------
// renderSky, fullscreen gradient drawn first in the scene pass
// ---------------------------------------------------------------------------
void RenderPipeline::renderSky(id<MTLRenderCommandEncoder> encoder,
                               const FrameRing::Alloc& skyUniforms) {
    [encoder setRenderPipelineState:_skyPipelineState];
    [encoder setDepthStencilState:_skyDepthState];
    [encoder setVertexBuffer:skyUniforms.buffer offset:skyUniforms.offset atIndex:1];
    [encoder setFragmentBuffer:skyUniforms.buffer offset:skyUniforms.offset atIndex:1];
    [encoder setFragmentTexture:_atmosphere->skyViewTexture() atIndex:0];
    [encoder setFragmentTexture:_atmosphere->transmittanceTexture() atIndex:1];

    // Draw fullscreen quad (6 vertices, no index buffer)
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

// ---------------------------------------------------------------------------
// renderShadows, depth-only cascade passes. Uses last frame's uploaded mesh
// registry (a streaming chunk casting a shadow one frame late is invisible),
// so it can run before renderChunks rebuilds it. A no-op at shadowQuality 0 or
// zero strength (the horizon crossing), where _sceneShadowUniforms carries
// strength 0 and the chunk fragment falls back to propagated skylight. The
// active light (sun by day, moon by night) and its strength come from
// computeDayNightUniforms. Exact chunks, far terrain, canopies, and entities
// cast into the cascades selected by their shared ownership masks.
// ---------------------------------------------------------------------------
void RenderPipeline::renderShadows(id<MTLCommandBuffer> commandBuffer,
                                   const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                                   const Camera& camera, const float lightDirection[3],
                                   float strength, const FoliageWindUniforms& foliageWind,
                                   const std::vector<std::shared_ptr<Entity>>* entities,
                                   const std::vector<ItemEntity>* itemEntities,
                                   const std::vector<Boat>* boats) {
    _shadowCasterCounts.fill(0U);
    if (_gfx.shadowQuality == 0 || strength <= 0.001f) {
        _shadowRefreshMask = 0U;
        _sceneShadowUniforms = ShadowUniforms{}; // strength 0 → chunk reads full sun
        return;
    }

    _shadowMap->setQuality(static_cast<uint32_t>(_gfx.shadowQuality));

    Vec3 lightDir{lightDirection[0], lightDirection[1], lightDirection[2]};
    _shadowMap->computeCascades(
        camera.getPosition(), camera.forward(), camera.FOV() * static_cast<float>(M_PI) / 180.0f,
        static_cast<float>(_displayWidth) / static_cast<float>(std::max(_displayHeight, 1u)),
        lightDir, strength);
    std::array<uint64_t, SHADOW_CASCADE_COUNT> casterRevisions{};
    const auto addCasterRevision = [&](uint64_t revision, const AABB& bounds) {
        for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
            if (_shadowMap->cascadeContains(cascade, bounds)) {
                ++_shadowCasterCounts[static_cast<size_t>(cascade)];
                casterRevisions[static_cast<size_t>(cascade)] ^= hash64(revision);
            }
        }
    };
    for (const auto& chunk : loadedChunks) {
        if (!chunk) {
            continue;
        }
        const auto mesh = _chunkMeshes.find(chunk->pos());
        if (mesh != _chunkMeshes.end() && mesh->second.uploaded &&
            _exactShadowOwnedSections.contains(chunk->pos())) {
            const ChunkPos position = chunk->pos();
            const uint64_t key = static_cast<uint64_t>(position.x) ^
                                 (static_cast<uint64_t>(static_cast<uint32_t>(position.y)) << 21U) ^
                                 (static_cast<uint64_t>(position.z) << 37U) ^
                                 mesh->second.builtVersion;
            addCasterRevision(key, chunk->getAABB());
        }
    }
    for (const FarShadowDrawPlan& plan : _farShadowDrawPlans) {
        if (plan.state.uploaded) {
            const bool canopyCasts =
                plan.canopy &&
                farCanopyCastsShadow(true, plan.canopy->alloc.has_value(),
                                     plan.canopy->alloc ? plan.canopy->alloc->indexCount : 0U);
            const FarTerrainBounds bounds = farShadowCasterBounds(
                plan.state.surfaceBounds,
                canopyCasts ? std::optional<FarTerrainBounds>{plan.canopy->bounds} : std::nullopt);
            const AABB casterBounds{
                {static_cast<float>(bounds.minX), bounds.minY, static_cast<float>(bounds.minZ)},
                {static_cast<float>(bounds.maxX), bounds.maxY, static_cast<float>(bounds.maxZ)}};
            const uint64_t baseRevision = static_cast<uint64_t>(plan.coordinate.x) ^
                                          (static_cast<uint64_t>(plan.coordinate.z) << 32U) ^
                                          plan.state.deterministicHash ^
                                          (static_cast<uint64_t>(plan.farMetadata.z) << 7U) ^
                                          (static_cast<uint64_t>(plan.farMetadata.w) << 39U);
            const uint64_t revision = farCanopyShadowRevision(
                baseRevision, canopyCasts,
                canopyCasts ? plan.canopy->authorityQuality : FarTerrainAuthorityQuality::FINAL,
                canopyCasts ? plan.canopy->deterministicHash : 0U);
            addCasterRevision(revision, casterBounds);
        }
    }
    if (entities) {
        for (const auto& entity : *entities) {
            if (!entity || !entity->alive) {
                continue;
            }
            const uint64_t revision =
                entity->id ^
                static_cast<uint64_t>(std::bit_cast<uint32_t>(entity->position.x)) << 1U ^
                static_cast<uint64_t>(std::bit_cast<uint32_t>(entity->position.y)) << 22U ^
                static_cast<uint64_t>(std::bit_cast<uint32_t>(entity->position.z)) << 43U;
            for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
                if (_shadowMap->entityCasterAffectsCascade(cascade, entity->aabb)) {
                    ++_shadowCasterCounts[static_cast<size_t>(cascade)];
                    casterRevisions[static_cast<size_t>(cascade)] ^= hash64(revision);
                }
            }
        }
    }
    if (itemEntities) {
        for (size_t index = 0; index < itemEntities->size(); ++index) {
            const ItemEntity& item = (*itemEntities)[index];
            if (item.stack.empty())
                continue;
            const uint64_t revision =
                static_cast<uint64_t>(index) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(item.position.x)) << 1U) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(item.position.y)) << 22U) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(item.position.z)) << 43U) ^
                (static_cast<uint64_t>(item.ageTicks) << 7U);
            const AABB bounds = item.getAABB();
            for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
                if (_shadowMap->entityCasterAffectsCascade(cascade, bounds)) {
                    ++_shadowCasterCounts[static_cast<size_t>(cascade)];
                    casterRevisions[static_cast<size_t>(cascade)] ^= hash64(revision);
                }
            }
        }
    }
    if (boats) {
        for (size_t index = 0; index < boats->size(); ++index) {
            const Boat& boat = (*boats)[index];
            const uint64_t revision =
                static_cast<uint64_t>(index) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(boat.position.x)) << 1U) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(boat.position.y)) << 22U) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(boat.position.z)) << 43U) ^
                (static_cast<uint64_t>(std::bit_cast<uint32_t>(boat.yaw)) << 11U);
            const AABB bounds = boat.getAABB();
            for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
                if (_shadowMap->entityCasterAffectsCascade(cascade, bounds)) {
                    ++_shadowCasterCounts[static_cast<size_t>(cascade)];
                    casterRevisions[static_cast<size_t>(cascade)] ^= hash64(revision);
                }
            }
        }
    }
    const uint32_t refreshMask =
        _shadowMap->selectRefreshMask(_frameRing.frameIndex(), casterRevisions, lightDir);
    _shadowRefreshMask = refreshMask;
    _sceneShadowUniforms = _shadowMap->shadowUniforms();

    for (int cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        if ((refreshMask & (1U << static_cast<uint32_t>(cascade))) == 0U) {
            continue;
        }
        MTLRenderPassDescriptor* passDesc = _shadowMap->passDescriptor(cascade);
        const char* passName = cascade < 2   ? "shadowNear"
                               : cascade < 4 ? "shadowFar"
                                             : "shadowHorizon";
        _gpuTimer->attachPass(passDesc, passName);
        id<MTLRenderCommandEncoder> encoder =
            [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
        if (!encoder) {
            resetMetalObject(passDesc);
            continue;
        }

        [encoder setRenderPipelineState:_shadowMap->chunkPipeline()];
        [encoder setDepthStencilState:_shadowMap->depthState()];
        [encoder setCullMode:MTLCullModeNone]; // greedy meshes are single-sided
        // Slope-scaled depth bias fights acne on faces near-parallel to the sun.
        // The clamp caps the slope term: vertical flora quads have near-infinite
        // light-space slope, so they always land ON the clamp, at 0.005 NDC
        // (~0.7 blocks along the light) stems sank into the ground, detaching
        // every flower's shadow from its base and erasing thin grass shadows
        // entirely. Cascade 0 (where that contact detail is visible) gets a
        // 10x tighter clamp and leans on the receiver normal offset for acne;
        // the far cascades keep the wide clamp, their NDC unit spans several
        // blocks, so a tight clamp reintroduces acne at low sun while a
        // ~1-block caster offset is invisible at 20+ blocks away.
        const float biasClamp = cascade == 0   ? 0.0005f
                                : cascade == 1 ? 0.0010f
                                : cascade < 4  ? 0.0030f
                                               : 0.0060f;
        [encoder setDepthBias:1.0f + 0.35f * static_cast<float>(cascade)
                   slopeScale:2.5f
                        clamp:biasClamp];
        [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
        [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

        ShadowPassUniforms passUniforms{};
        std::memcpy(&passUniforms.lightViewProj, _shadowMap->cascadeViewProj(cascade).data.data(),
                    sizeof(float) * 16);
        const Vec3& projectionOrigin = _shadowMap->cascadeProjectionOrigin(cascade);
        passUniforms.projectionOrigin =
            simd_make_float4(projectionOrigin.x, projectionOrigin.y, projectionOrigin.z, 0.0F);
        passUniforms.foliageWind =
            shadowFoliageWindForCascade(foliageWind, static_cast<uint32_t>(cascade));
        passUniforms.time = _animTime;
        [encoder setVertexBytes:&passUniforms length:sizeof(passUniforms) atIndex:1];
        const FarTerrainOwnershipUniforms noFarOwnership{};
        [encoder setFragmentBytes:&noFarOwnership length:sizeof(noFarOwnership) atIndex:5];

        // Exact meshes cast first. Far terrain then uses the same ownership
        // mask as the color pass, followed by the entity-specific vertex path.
        for (auto& chunk : loadedChunks) {
            if (!chunk || !chunk->generated)
                continue;
            const ChunkPos key = chunk->pos();
            if (!_exactShadowOwnedSections.contains(key))
                continue;
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

        // Replay the prior color pass's exact far draw authority. Transition
        // source and target topologies retain their monotonic flags and the
        // connected-coverage frontier, so hidden islands cannot cast and a
        // retiring canopy never leaves a stale shadow behind.
        for (const FarShadowDrawPlan& plan : _farShadowDrawPlans) {
            if (!plan.state.uploaded) {
                continue;
            }
            const FarTerrainMeshState& state = plan.state;
            const bool canopyCasts =
                plan.canopy &&
                farCanopyCastsShadow(true, plan.canopy->alloc.has_value(),
                                     plan.canopy->alloc ? plan.canopy->alloc->indexCount : 0U);
            const FarTerrainBounds bounds = farShadowCasterBounds(
                state.surfaceBounds,
                canopyCasts ? std::optional<FarTerrainBounds>{plan.canopy->bounds} : std::nullopt);
            const AABB aabb{
                {static_cast<float>(bounds.minX), bounds.minY, static_cast<float>(bounds.minZ)},
                {static_cast<float>(bounds.maxX), bounds.maxY, static_cast<float>(bounds.maxZ)}};
            if (!_shadowMap->cascadeContains(cascade, aabb)) {
                continue;
            }

            ChunkOrigin origin{};
            origin.origin = simd_make_float4(static_cast<float>(state.bounds.minX), 0.0F,
                                             static_cast<float>(state.bounds.minZ), 0.0F);
            origin.farMetadata = plan.farMetadata;
            [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
            [encoder setFragmentBytes:&plan.farOwnership
                               length:sizeof(plan.farOwnership)
                              atIndex:5];
            if (state.opaqueIndexCount > 0) {
                [encoder setVertexBuffer:state.alloc.vertexBuffer
                                  offset:state.alloc.vertexOffset
                                 atIndex:0];
                [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:state.opaqueIndexCount
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:state.alloc.indexBuffer
                             indexBufferOffset:state.alloc.indexOffset];
            }
            if (canopyCasts) {
                const MegaBuffer::ChunkAllocation& canopyAlloc = *plan.canopy->alloc;
                [encoder setVertexBuffer:canopyAlloc.vertexBuffer
                                  offset:canopyAlloc.vertexOffset
                                 atIndex:0];
                [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:canopyAlloc.indexCount
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:canopyAlloc.indexBuffer
                             indexBufferOffset:canopyAlloc.indexOffset];
            }
        }
        if (entities && _entityRenderer) {
            _entityRenderer->renderShadows(
                encoder, passUniforms, *entities, [this, cascade](const AABB& bounds) {
                    return _shadowMap->entityCasterAffectsCascade(cascade, bounds);
                });
        }
        if ((itemEntities && _itemEntityRenderer) || (boats && _boatRenderer)) {
            [encoder setRenderPipelineState:_shadowMap->entityPipeline()];
            [encoder setVertexBytes:&passUniforms length:sizeof(passUniforms) atIndex:1];
            [encoder setCullMode:MTLCullModeBack];
            [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
            const auto affectsCascade = [this, cascade](const AABB& bounds) {
                return _shadowMap->entityCasterAffectsCascade(cascade, bounds);
            };
            if (itemEntities && _itemEntityRenderer) {
                _itemEntityRenderer->renderShadowCasters(encoder, *itemEntities, affectsCascade);
            }
            if (boats && _boatRenderer) {
                _boatRenderer->renderShadowCasters(encoder, *boats, affectsCascade);
            }
        }
        [encoder endEncoding];
        resetMetalObject(passDesc);
    }
}

// ---------------------------------------------------------------------------
// renderChunks (opaque pass)
// ---------------------------------------------------------------------------
void RenderPipeline::renderChunks(id<MTLRenderCommandEncoder> encoder, const World& world,
                                  const std::vector<std::shared_ptr<Chunk>>& loadedChunks,
                                  const Mat4& viewMatrix, const Mat4& projectionMatrix,
                                  const Camera& camera, const float sunDirection[3],
                                  const float sunColor[3], const float ambientColor[3],
                                  const float fogColor[3], const FoliageWindUniforms& foliageWind,
                                  bool drawGeometry, bool prepareProtectedFinal) {
    const Vec3 cameraPosition = camera.getPosition();
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

    // Scene and shadow passes use the same weather wind and animation clock.
    uniforms.foliageWind = foliageWind;
    uniforms.time = _animTime;
    uniforms.wetness = _cameraUnderwater ? 0.0f : _wetness;

    // Upload to GPU (kept for the entity renderer + water vertex stage too)
    _frameUniforms = _frameRing.push(&uniforms, sizeof(Uniforms));

    // Bind the shared atlas + uniforms once; every chunk draw reuses them
    [encoder setVertexBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
    [encoder setFragmentBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentTexture:_blockTextures->emissionMask() atIndex:5];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];

    // Shadow sampling: two detailed arrays, the horizon texture, one
    // comparison sampler, and one shared uniform block. Disabled shadows keep
    // all targets bound for validation while strength zero returns full light.
    FrameRing::Alloc shadowAlloc = _frameRing.push(&_sceneShadowUniforms, sizeof(ShadowUniforms));
    [encoder setFragmentTexture:_shadowMap->nearDepthTexture() atIndex:1];
    [encoder setFragmentTexture:_shadowMap->farDepthTexture() atIndex:2];
    [encoder setFragmentTexture:_shadowMap->horizonDepthTexture() atIndex:3];
    [encoder setFragmentTexture:_clouds->shadowTexture() atIndex:4];
    [encoder setFragmentSamplerState:_shadowMap->comparisonSampler() atIndex:1];
    [encoder setFragmentBuffer:shadowAlloc.buffer offset:shadowAlloc.offset atIndex:4];
    const FarTerrainOwnershipUniforms noFarOwnership{};
    [encoder setFragmentBytes:&noFarOwnership length:sizeof(noFarOwnership) atIndex:5];
    const CloudShadowUniforms& cloudShadow = _clouds->shadowUniforms();
    [encoder setFragmentBytes:&cloudShadow length:sizeof(cloudShadow) atIndex:6];

    // Reset seed-owned far state before exact ownership is accumulated. The
    // second call in renderFarTerrain is then an inexpensive no-op.
    resetFarTerrain(world);

    // Water draws recorded here render later, in the dedicated water pass
    _waterDraws.clear();
    // renderShadows already consumed the prior frame's set. Rebuild the next
    // replay from this frame's stable authority decisions before frustum
    // culling, because an offscreen owner may still cast into a visible region.
    _exactShadowOwnedSections.clear();

    // Builds only happen within the render radius: the generation radius is
    // one chunk wider, so every meshable chunk has generated neighbors for
    // its snapshot (frontier chunks simply wait their turn).
    const int64_t camChunkX = Chunk::worldToChunk(static_cast<int64_t>(std::floor(camX)));
    const int32_t camChunkY = Chunk::worldToChunkY(static_cast<int32_t>(std::floor(camY)));
    const int64_t camChunkZ = Chunk::worldToChunk(static_cast<int64_t>(std::floor(camZ)));
    // World publishes mesh ownership through its active radius, which is one
    // column wider than the nominal exact distance. That ring supplies the
    // cold spawn collision halo and the steady-state handoff boundary. Use
    // the same radius here or a nominal distance of zero can mesh only the
    // center of the five-column entry requirement and never become playable.
    const int renderRadius = exactStreamingMeshRadiusChunks(world.getExactViewDistance());
    const auto meshCandidateSnapshot = world.getMeshCandidateSnapshot();
    const auto shouldMesh = [&](ChunkPos pos) {
        return meshCandidateSnapshot && meshCandidateSnapshot->contains(pos);
    };

    // Recycle regions whose last GPU reader has finished, then sweep mesh
    // allocations of chunks the world has since unloaded, freed space can
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
            _pendingExactLightingPublications.erase(it->first);
            it = _chunkMeshes.erase(it);
        } else {
            ++it;
        }
    }

    // ---- Async meshing: workers build, the render thread only uploads ----
    if (!_meshScheduler) {
        _meshScheduler = std::make_unique<MeshScheduler>(world, EXACT_MESH_WORKER_COUNT);
    }

    // The active-set snapshot is the sole exact residency authority. A camera
    // jump can replace it while the bounded mesh queue still contains work for
    // the previous disk, so release those queued requests before draining or
    // admitting this frame's nearest candidates. Running builds remain bounded
    // and their ordinary stale-result path handles completion.
    const auto clearCanceledMeshRequest = [&](const MeshCanceledRequest& canceled) {
        const auto resident = _chunkMeshes.find(canceled.pos);
        if (resident == _chunkMeshes.end())
            return;
        resident->second.requestedVersion = chunkMeshRequestAfterCompletion(
            resident->second.requestedVersion, canceled.requestedVersion);
    };
    if (meshCandidateSnapshot) {
        for (const MeshCanceledRequest& canceled :
             _meshScheduler->cancelQueuedOutside(*meshCandidateSnapshot)) {
            clearCanceledMeshRequest(canceled);
        }
    }

    // Read the current exact requirement epoch before draining worker
    // completions. Upload ordering must follow the new camera immediately;
    // rebuilding this cache after the drain lets an old completion wave spend
    // the frame budget while nearer required surfaces wait another frame.
    const std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactCoverageForFrame =
        world.getExactSurfaceCoverageSnapshot();
    const int exactCoverageRadius = exactCoverageForFrame
                                        ? std::clamp(exactCoverageForFrame->nominalRadiusChunks, 0,
                                                     world.getExactViewDistance())
                                        : 0;
    const uint64_t exactCoverageEpoch = exactCoverageForFrame ? exactCoverageForFrame->epoch : 0;
    if (!_farTerrainExactCoverage.matches(exactCoverageEpoch, exactCoverageRadius)) {
        _farTerrainExactCoverage.rebuild(
            exactCoverageEpoch, exactCoverageRadius,
            exactCoverageForFrame
                ? std::span<const ChunkPos>(exactCoverageForFrame->requiredSections)
                : std::span<const ChunkPos>(),
            exactCoverageForFrame
                ? std::span<const ColumnPos>(exactCoverageForFrame->unresolvedColumns)
                : std::span<const ColumnPos>(),
            [&](ChunkPos position) { return _exactOwnedSections.contains(position); });
    }
    if (!_farTerrainExactFloraCoverage.matches(exactCoverageEpoch, exactCoverageRadius)) {
        _farTerrainExactFloraCoverage.rebuild(
            exactCoverageEpoch, exactCoverageRadius,
            exactCoverageForFrame
                ? std::span<const ChunkPos>(exactCoverageForFrame->floraRequiredSections)
                : std::span<const ChunkPos>(),
            exactCoverageForFrame
                ? std::span<const ColumnPos>(exactCoverageForFrame->unresolvedColumns)
                : std::span<const ColumnPos>(),
            [&](ChunkPos position) { return _exactOwnedSections.contains(position); });
    }
    // Queued mesh requests can remain selected after a camera move. Refresh
    // their lane and distance in place so a former disk-edge request that is
    // now under the player moves to the front without waiting for, canceling,
    // or duplicating its existing single-flight build.
    _meshScheduler->reprioritizeQueued([&](ChunkPos position) {
        const int64_t dx = position.x - camChunkX;
        const int64_t dy = static_cast<int64_t>(position.y) - camChunkY;
        const int64_t dz = position.z - camChunkZ;
        const bool explorationBand =
            dx * dx + dz * dz <= EXPLORATION_RADIUS_CHUNKS * EXPLORATION_RADIUS_CHUNKS;
        const ExactMeshCandidatePriority priority = exactMeshCandidatePriority(
            dx, dy, dz, explorationBand,
            _farTerrainExactCoverage.sectionRequired(position),
            _farTerrainExactFloraCoverage.sectionRequired(position));
        return MeshRequestPriority{priority.lane, priority.distanceSquared};
    });

    // World lighting changes invalidate SSGI immediately. Track only stale,
    // already-visible exact meshes at that revision so their delayed GPU
    // publication triggers one final reset. Ordinary streaming and unrelated
    // background remeshes never enter this batch.
    const uint64_t worldLightingRevision = world.lightingRevision();
    if (_observedWorldLightingRevision == 0) {
        _observedWorldLightingRevision = worldLightingRevision;
    } else if (_observedWorldLightingRevision != worldLightingRevision) {
        for (const auto& chunk : loadedChunks) {
            if (!chunk || !chunk->generated)
                continue;
            const auto resident = _chunkMeshes.find(chunk->pos());
            if (resident == _chunkMeshes.end() || !resident->second.uploaded)
                continue;
            const uint32_t liveVersion = chunk->version.load(std::memory_order_relaxed);
            if (resident->second.builtVersion != liveVersion) {
                _pendingExactLightingPublications.insert_or_assign(chunk->pos(), liveVersion);
            }
        }
        _exactLightingPublicationBatchPending =
            _exactLightingPublicationBatchPending || !_pendingExactLightingPublications.empty();
        _observedWorldLightingRevision = worldLightingRevision;
    }

    // Upload one finished mesh into the registry. Returns false on a
    // transient MegaBuffer-full failure (builtVersion stays 0, so the chunk
    // re-requests once space frees up).
    constexpr int MAX_MESH_UPLOADS_PER_FRAME = 64;
    bool allocFailureLogged = false;
    const ChunkPos uploadCamera{camChunkX, camChunkY, camChunkZ};
    std::unordered_set<ChunkPos> committedExactMeshesThisDrain;
    committedExactMeshesThisDrain.reserve(MAX_MESH_UPLOADS_PER_FRAME);
    const auto protectCommittedMeshUntilColumnHandoff = [&](ChunkPos key) {
        committedExactMeshesThisDrain.insert(key);
    };
    const auto meshPriorityFor = [&](ChunkPos key) {
        return exactMeshUploadPriority(key, uploadCamera, EXPLORATION_RADIUS_CHUNKS,
                                       _farTerrainExactCoverage.sectionRequired(key),
                                       _farTerrainExactFloraCoverage.sectionRequired(key));
    };
    const auto meshVictimFor = [&](ExactMeshUploadPriority incoming) {
        auto victim = _chunkMeshes.end();
        std::optional<ExactMeshUploadPriority> victimPriority;
        for (auto it = _chunkMeshes.begin(); it != _chunkMeshes.end(); ++it) {
            if (!exactMeshRegistryVictimEligible(
                    _exactOwnedSections.contains(it->first),
                    committedExactMeshesThisDrain.contains(it->first))) {
                continue;
            }
            const ExactMeshUploadPriority priority = meshPriorityFor(it->first);
            if (!victimPriority || exactMeshEvictionRanksBefore(priority, *victimPriority)) {
                victim = it;
                victimPriority = priority;
            }
        }
        if (victim == _chunkMeshes.end() || !victimPriority ||
            !exactMeshRegistryMayReplace(incoming, *victimPriority)) {
            return _chunkMeshes.end();
        }
        return victim;
    };
    auto canMakeMeshSlot = [&](ChunkPos key, ExactMeshUploadPriority incoming) {
        if (_chunkMeshes.contains(key) || _chunkMeshes.size() < MAX_MESH_RESIDENT_CUBES)
            return true;
        return meshVictimFor(incoming) != _chunkMeshes.end();
    };
    auto makeMeshSlot = [&](ChunkPos key, ExactMeshUploadPriority incoming) -> ChunkMeshState* {
        auto existing = _chunkMeshes.find(key);
        if (existing != _chunkMeshes.end())
            return &existing->second;
        if (_chunkMeshes.size() >= MAX_MESH_RESIDENT_CUBES) {
            auto victim = meshVictimFor(incoming);
            if (victim == _chunkMeshes.end())
                return nullptr;
            if (const auto canceled = _meshScheduler->cancelQueued(victim->first)) {
                clearCanceledMeshRequest(*canceled);
            }
            // Reuse the victim node so insertion at the hard cap cannot fail
            // after discarding a live mesh.
            auto node = _chunkMeshes.extract(victim);
            _pendingExactLightingPublications.erase(node.key());
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
    const auto completeTrackedLightingPublication = [&](ChunkPos key, uint32_t builtVersion,
                                                        bool replacedUploadedMesh) {
        const auto target = _pendingExactLightingPublications.find(key);
        if (target == _pendingExactLightingPublications.end() ||
            !exactMeshPublicationInvalidatesHistory(true, replacedUploadedMesh, builtVersion,
                                                    target->second)) {
            return;
        }
        _pendingExactLightingPublications.erase(target);
        _exactLightingPublicationCompleted = true;
    };
    auto applyMesh = [&](ChunkPos key, const MeshOutput& mesh, uint32_t builtVersion,
                         uint32_t completedRequestVersion) -> bool {
        const ExactMeshUploadPriority incoming = meshPriorityFor(key);
        if (!canMakeMeshSlot(key, incoming))
            return false;
        if (mesh.vertices.empty()) {
            ChunkMeshState* state = makeMeshSlot(key, incoming);
            if (!state)
                return false;
            const bool replacedUploadedMesh = state->uploaded;
            state->requestedVersion =
                chunkMeshRequestAfterCompletion(state->requestedVersion, completedRequestVersion);
            if (state->uploaded) {
                _megaBuffer->deferFree(state->alloc, _frameRing.frameIndex());
                state->uploaded = false;
            }
            state->opaqueIndexCount = 0;
            state->builtVersion = builtVersion; // all-air: nothing to draw
            completeTrackedLightingPublication(key, builtVersion, replacedUploadedMesh);
            protectCommittedMeshUntilColumnHandoff(key);
            return true;
        }
        std::optional<MegaBuffer::ChunkAllocation> replacement;
        try {
            replacement = _megaBuffer->allocate(static_cast<uint32_t>(mesh.vertices.size()),
                                                static_cast<uint32_t>(mesh.indices.size()));
            _megaBuffer->uploadVertices(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
                                        *replacement);
            _megaBuffer->uploadIndices(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
                                       *replacement);
            // Do not evict a resident entry until allocation and upload have
            // succeeded. At the cap makeMeshSlot reuses the victim map node.
            ChunkMeshState* state = makeMeshSlot(key, incoming);
            if (!state) {
                _megaBuffer->free(*replacement);
                return false;
            }
            const bool replacedUploadedMesh = state->uploaded;
            state->requestedVersion =
                chunkMeshRequestAfterCompletion(state->requestedVersion, completedRequestVersion);
            if (state->uploaded) {
                _megaBuffer->deferFree(state->alloc, _frameRing.frameIndex());
            }
            state->alloc = *replacement;
            state->opaqueIndexCount = mesh.opaqueIndexCount;
            state->uploaded = true;
            state->builtVersion = builtVersion;
            completeTrackedLightingPublication(key, builtVersion, replacedUploadedMesh);
            protectCommittedMeshUntilColumnHandoff(key);
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
    std::stable_sort(_pendingResults.begin(), _pendingResults.end(),
                     [&](const MeshResult& left, const MeshResult& right) {
                         const bool leftSurfaceRequired =
                             _farTerrainExactCoverage.sectionRequired(left.pos);
                         const bool rightSurfaceRequired =
                             _farTerrainExactCoverage.sectionRequired(right.pos);
                         const bool leftFloraRequired =
                             _farTerrainExactFloraCoverage.sectionRequired(left.pos);
                         const bool rightFloraRequired =
                             _farTerrainExactFloraCoverage.sectionRequired(right.pos);
                         return exactMeshUploadRanksBefore(
                             exactMeshUploadPriority(left.pos, uploadCamera,
                                                     EXPLORATION_RADIUS_CHUNKS,
                                                     leftSurfaceRequired, leftFloraRequired),
                             exactMeshUploadPriority(right.pos, uploadCamera,
                                                     EXPLORATION_RADIUS_CHUNKS,
                                                     rightSurfaceRequired, rightFloraRequired));
                     });
    constexpr size_t MAX_UPLOAD_BYTES_PER_FRAME = 32 * 1024 * 1024;
    // An edit synchronously relights its whole affected neighborhood (home cube
    // plus the face, edge, and corner cubes light can reach), so let the edit
    // fast path rebuild all of them in the same frame instead of trickling two
    // per frame. It only bites on the post-edit frame and stays inside the
    // upload count and byte budgets below.
    constexpr int MAX_EDIT_SYNC_BUILDS_PER_FRAME = 8;
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
            ++resultsConsumed; // chunk unloaded while meshing, drop
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
        if (!canMakeMeshSlot(key, meshPriorityFor(key))) {
            // The placeholder yielded to a strictly more important surface
            // while this worker was finishing. Drop the result so it cannot
            // pin consumer capacity; it will be requested again if it later
            // re-enters the admissible current-camera set.
            if (resident != _chunkMeshes.end()) {
                resident->second.requestedVersion = chunkMeshRequestAfterCompletion(
                    resident->second.requestedVersion, result.requestedVersion);
            }
            ++resultsConsumed;
            continue;
        }
        if (!applyMesh(key, result.mesh, result.builtVersion, result.requestedVersion)) {
            break; // MegaBuffer full: retry this result next frame
        }
        ++uploads;
        uploadBytes += bytes;
        ++resultsConsumed;
    }
    _pendingResults.erase(_pendingResults.begin(),
                          _pendingResults.begin() + static_cast<long>(resultsConsumed));
    // Results that left the current exact candidate set after the drain do
    // not get to occupy shared scheduler capacity until the next frame.
    for (auto iterator = _pendingResults.begin(); iterator != _pendingResults.end();) {
        if (_liveChunksByPosition.contains(iterator->pos) && shouldMesh(iterator->pos)) {
            ++iterator;
            continue;
        }
        clearCanceledMeshRequest({iterator->pos, iterator->requestedVersion});
        iterator = _pendingResults.erase(iterator);
    }
    _meshScheduler->acknowledgeConsumerPending(_pendingResults.size());

    // 2. Edit fast path: chunks right next to the camera re-mesh
    //    synchronously so breaking a block never shows a stale frame.
    int syncBuilds = 0;
    for (auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated || syncBuilds >= MAX_EDIT_SYNC_BUILDS_PER_FRAME ||
            uploads >= MAX_MESH_UPLOADS_PER_FRAME)
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

    if (_exactLightingPublicationBatchPending && _pendingExactLightingPublications.empty()) {
        if (_exactLightingPublicationCompleted) {
            ++_exactMaterialPublicationRevision;
        }
        _exactLightingPublicationBatchPending = false;
        _exactLightingPublicationCompleted = false;
    }

    // 3. Candidate scan: every generated chunk in the render radius whose
    //    mesh is missing or stale, nearest first, until the in-flight cap.
    const auto candidatePriority = [&](const Chunk* chunk) {
        const int64_t dx = chunk->chunkX - camChunkX;
        const int64_t dy = static_cast<int64_t>(chunk->chunkY) - camChunkY;
        const int64_t dz = chunk->chunkZ - camChunkZ;
        const bool explorationBand =
            dx * dx + dz * dz <= EXPLORATION_RADIUS_CHUNKS * EXPLORATION_RADIUS_CHUNKS;
        return exactMeshCandidatePriority(
            dx, dy, dz, explorationBand,
            _farTerrainExactCoverage.sectionRequired(chunk->pos()),
            _farTerrainExactFloraCoverage.sectionRequired(chunk->pos()));
    };
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
        _meshCandidates.push_back({candidatePriority(chunk.get()), chunk.get()});
    }
    std::sort(_meshCandidates.begin(), _meshCandidates.end(),
              [&](const auto& left, const auto& right) {
                  const ExactMeshCandidatePriority leftPriority = left.priority;
                  const ExactMeshCandidatePriority rightPriority = right.priority;
                  if (leftPriority != rightPriority)
                      return exactMeshCandidateRanksBefore(leftPriority, rightPriority);
                  const ChunkPos leftPosition = left.chunk->pos();
                  const ChunkPos rightPosition = right.chunk->pos();
                  if (leftPosition.x != rightPosition.x)
                      return leftPosition.x < rightPosition.x;
                  if (leftPosition.z != rightPosition.z)
                      return leftPosition.z < rightPosition.z;
                  return leftPosition.y < rightPosition.y;
              });
    const auto releaseLowerPriorityPendingResult = [&](ExactMeshUploadPriority incoming) {
        for (size_t offset = _pendingResults.size(); offset > 0; --offset) {
            const size_t index = offset - 1;
            const MeshResult& pending = _pendingResults[index];
            const ExactMeshUploadPriority pendingPriority = meshPriorityFor(pending.pos);
            if (!exactMeshRegistryMayReplace(incoming, pendingPriority))
                continue;
            clearCanceledMeshRequest({pending.pos, pending.requestedVersion});
            _pendingResults.erase(_pendingResults.begin() + static_cast<long>(index));
            _meshScheduler->acknowledgeConsumerPending(_pendingResults.size());
            return true;
        }
        return false;
    };
    for (const auto& candidate : _meshCandidates) {
        const Chunk* chunkPtr = candidate.chunk;
        ChunkPos pos = chunkPtr->pos();
        const ExactMeshUploadPriority incoming{candidate.priority, pos};
        if (!canMakeMeshSlot(pos, incoming)) {
            break; // every resident mesh owns exact terrain; wait for a real eviction
        }
        const uint32_t requestedVersion = chunkPtr->version.load(std::memory_order_relaxed);
        const ExactMeshCandidatePriority priority = candidate.priority;
        std::optional<MeshCanceledRequest> displaced;
        bool admitted = false;
        for (;;) {
            if (_meshScheduler->enqueue(pos, requestedVersion, priority.lane,
                                        priority.distanceSquared, &displaced)) {
                admitted = true;
                break;
            }
            // Consumer-owned distant completions are part of the same hard
            // 64-slot budget. Let a nearer missing surface reclaim one now,
            // rather than waiting a frame while the player remains on coarse
            // collision and visual authority.
            if (!releaseLowerPriorityPendingResult(incoming))
                break;
        }
        if (!admitted)
            break; // only equal-or-higher-priority work remains in flight
        if (displaced)
            clearCanceledMeshRequest(*displaced);
        ChunkMeshState* state = makeMeshSlot(pos, incoming);
        if (!state) {
            break; // defensive: the registry never grows beyond the hard cap
        }
        state->requestedVersion = requestedVersion;
    }

    // Publish every revision-ready ownership section before submitting any
    // exact draw. A column handoff is atomic: while even one required surface
    // section is missing, its resident far parent remains the sole visible
    // surface instead of being layered over the exact sections that happened
    // to finish first.
    for (const auto& chunk : loadedChunks) {
        if (!chunk || !chunk->generated)
            continue;
        const ChunkPos key = chunk->pos();
        const auto cached = _chunkMeshes.find(key);
        if (cached == _chunkMeshes.end())
            continue;
        if (farTerrainExactSectionOwnsSurface(_exactOwnedSections.contains(key),
                                              cached->second.builtVersion,
                                              chunk->version.load(std::memory_order_relaxed))) {
            setExactSectionOwned(key, true);
        }
    }
    const FarTerrainExactHandoff& exactDrawHandoff =
        _farTerrainExactCoverage.sample(cameraPosition.x, cameraPosition.z);
    const FarTerrainExactHandoff& exactFloraDrawHandoff =
        _farTerrainExactFloraCoverage.sample(cameraPosition.x, cameraPosition.z);
    const auto baseResident = [&](const FarTerrainKey& key) {
        const auto resident = _farTerrainMeshes.find(key);
        return resident != _farTerrainMeshes.end() && resident->second.uploaded;
    };
    const FarTerrainCoverageFrontier exactDrawCoverage =
        farTerrainCoverageFrontier(_farTerrainCandidates, baseResident);
    const auto drawableCoverageParentFor = [&](ColumnPos chunkColumn) {
        constexpr int64_t COLUMNS_PER_TILE = FAR_TERRAIN_TILE_EDGE / CHUNK_EDGE;
        const ColumnPos tile{
            world_coord::floorDiv(chunkColumn.x, COLUMNS_PER_TILE),
            world_coord::floorDiv(chunkColumn.z, COLUMNS_PER_TILE),
        };
        const FarTerrainKey parent{tile.x, tile.z, FAR_TERRAIN_BASE_STEP};
        if (!baseResident(parent))
            return false;
        const double minimumX = static_cast<double>(tile.x) * FAR_TERRAIN_TILE_EDGE;
        const double maximumX = minimumX + FAR_TERRAIN_TILE_EDGE;
        const double minimumZ = static_cast<double>(tile.z) * FAR_TERRAIN_TILE_EDGE;
        const double maximumZ = minimumZ + FAR_TERRAIN_TILE_EDGE;
        const double dx = cameraPosition.x < minimumX   ? minimumX - cameraPosition.x
                          : cameraPosition.x > maximumX ? cameraPosition.x - maximumX
                                                        : 0.0;
        const double dz = cameraPosition.z < minimumZ   ? minimumZ - cameraPosition.z
                          : cameraPosition.z > maximumZ ? cameraPosition.z - maximumZ
                                                        : 0.0;
        return farTerrainCoverageDrawEligible(dx * dx + dz * dz, exactDrawCoverage);
    };

    // Collision publication is accumulated in the draw walk below so the
    // render thread does not scan every loaded cube twice.
    _exactCollisionOwnedSections.clear();
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
            case WorldgenOverlayMode::LOD: {
                const auto color = terrainLodOverlayColor(std::nullopt);
                return simd_make_float4(color[0], color[1], color[2], color[3]);
            }
            case WorldgenOverlayMode::AUTHORITY:
                return simd_make_float4(0.10f, 0.95f, 0.28f, 0.78f);
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
        const ColumnPos chunkColumn{key.x, key.z};
        const bool sectionRequired = _farTerrainExactCoverage.sectionRequired(key);
        const bool floraSectionRequired = _farTerrainExactFloraCoverage.sectionRequired(key);
        const bool floraSectionMayPublish = exactFloraSectionMayPublish(
            sectionRequired, floraSectionRequired,
            exactFloraDrawHandoff.columnFullyReady(chunkColumn));
        const bool coverageParentDrawable = drawableCoverageParentFor(chunkColumn);
        const bool revisionReady = farTerrainExactSectionOwnsSurface(
            _exactOwnedSections.contains(key), meshState.builtVersion,
            chunk->version.load(std::memory_order_relaxed));
        if (floraSectionMayPublish && farTerrainExactCollisionOwnsSection(
                sectionRequired, exactDrawHandoff.columnFullyReady(chunkColumn),
                coverageParentDrawable, revisionReady)) {
            // Empty exact meshes publish intentional air even though the
            // visual path below has no allocation to submit.
            _exactCollisionOwnedSections.push_back(key);
        }
        const FarTerrainExactVisualOwnership ownership = farTerrainExactVisualOwnership(
            sectionRequired, exactDrawHandoff.columnFullyReady(chunkColumn),
            coverageParentDrawable, meshState.uploaded);
        if (!ownership.drawExact || !floraSectionMayPublish) {
            continue;
        }
        _exactShadowOwnedSections.insert(key);

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
        if (_worldgenOverlayMode == WorldgenOverlayMode::LOD ||
            _worldgenOverlayMode == WorldgenOverlayMode::AUTHORITY) {
            if (_worldgenOverlayMode == WorldgenOverlayMode::LOD) {
                const auto color = terrainLodOverlayColor(std::nullopt);
                origin.overlayColorAndStrength =
                    simd_make_float4(color[0], color[1], color[2], color[3]);
            } else {
                origin.overlayColorAndStrength =
                    simd_make_float4(0.10f, 0.95f, 0.28f, 0.78f);
            }
        } else if (_worldgenOverlayMode != WorldgenOverlayMode::NONE) {
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

    // The loaded snapshot retains a stable order between membership changes,
    // so vector equality avoids allocating and copying a 16K-entry collision
    // set every render frame. A stale epoch is rejected inside World and is
    // retried next frame without changing the last successful publication.
    const bool collisionPublicationChanged =
        !_publishedExactCollisionCoverageEpoch ||
        *_publishedExactCollisionCoverageEpoch != exactCoverageEpoch ||
        _publishedExactCollisionOwnedSections != _exactCollisionOwnedSections;
    if (collisionPublicationChanged &&
        world.publishExactCollisionOwnership(exactCoverageEpoch,
                                             _exactCollisionOwnedSections)) {
        _publishedExactCollisionCoverageEpoch = exactCoverageEpoch;
        _publishedExactCollisionOwnedSections = _exactCollisionOwnedSections;
    }

    // Fill the exact-to-horizon region after exact cubes. Each ready exact
    // column suppresses its own far terrain, water, and canopies. An
    // unrelated loading gap therefore cannot reveal coarse geometry over a
    // revision-ready nearby column.
    renderFarTerrain(encoder, world, camera, fogColor, drawGeometry, -1, exactCoverageForFrame,
                     prepareProtectedFinal);

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

void RenderPipeline::resetFarTerrain(const World& world) {
    const uint64_t worldSeed = world.getSeed();
    const uint64_t worldInstanceId = world.instanceId();
    if (_farTerrainSeed && *_farTerrainSeed == worldSeed && _farTerrainWorldInstanceId &&
        *_farTerrainWorldInstanceId == worldInstanceId && _farTerrainScheduler)
        return;

    const std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext =
        world.generationContext();

    if (_farTerrainScheduler)
        _farTerrainScheduler->shutdown();
    if (_farMegaBuffer) {
        for (auto& [key, state] : _farTerrainMeshes) {
            (void)key;
            if (state.uploaded) {
                _farMegaBuffer->deferFree(state.alloc, _frameRing.frameIndex());
            }
        }
        for (auto& [key, state] : _farCanopyAttachments) {
            (void)key;
            if (state.alloc)
                _farMegaBuffer->deferFree(*state.alloc, _frameRing.frameIndex());
        }
        for (auto& [key, transition] : _farTerrainAuthorityTransitions) {
            (void)key;
            if (transition.source.uploaded) {
                _farMegaBuffer->deferFree(transition.source.alloc, _frameRing.frameIndex());
            }
            if (transition.sourceCanopy && transition.sourceCanopy->alloc) {
                _farMegaBuffer->deferFree(*transition.sourceCanopy->alloc, _frameRing.frameIndex());
            }
        }
    }
    _farTerrainMeshes.clear();
    _farCanopyAttachments.clear();
    _farTerrainWanted.clear();
    _farTerrainPriorityOrder.clear();
    _farTerrainActiveTiles.clear();
    _farTerrainDesiredByTile.clear();
    _farTerrainDisplayedByTile.clear();
    _farTerrainTransitions.clear();
    _farCanopyLodFallbacks.clear();
    _farTerrainAuthorityTransitions.clear();
    _farTerrainNearGraceStartedAt.clear();
    _farTerrainResults.clear();
    _farCanopyResults.clear();
    _farTerrainSelectionCamera.reset();
    _farTerrainDesiredMetricsCamera.reset();
    _farTerrainSpeculativeCamera.reset();
    _farTerrainSelectionViewDistance = -1;
    _farTerrainProtectedNearHandoff.clear();
    _farTerrainProtectedNearEpoch = 0;
    _farTerrainProtectedRecentMotionX = 0;
    _farTerrainProtectedRecentMotionZ = 0;
    _farTerrainPredictedNearAnchor.reset();
    _farTerrainProtectedNearClosureSnapshot.reset();
    _farTerrainDesiredViewportHeight = 0;
    _farTerrainDesiredVerticalFovRadians = 0.0;
    _farTerrainDesiredDrawGeometry = false;
    _farTerrainDesiredMetricsDirty = true;
    _farTerrainViewEpoch = 0;
    if (_farTerrainWorldEpoch == std::numeric_limits<uint64_t>::max()) {
        _farTerrainWorldEpoch = 1;
    } else {
        ++_farTerrainWorldEpoch;
    }
    _farTerrainCandidates.clear();
    _farTerrainCachedBaseRequests.clear();
    _farTerrainMissingBaseRequests.clear();
    _farTerrainDistantBaseRequests.clear();
    _farTerrainFinalBaseRequests.clear();
    _farTerrainPerceptualFinalRequests.clear();
    _farTerrainFinalRefinementRequests.clear();
    _farTerrainCanopyRefreshRequests.clear();
    _farTerrainCanopyRefreshKeys.clear();
    _farTerrainUrgentRefinementRequests.clear();
    _farTerrainUrgentRefinementKeys.clear();
    _farTerrainPredictedNearPatchTargets.clear();
    _farTerrainPredictedCriticalResidencyKeys.clear();
    _farTerrainConnectedNearPatchTargets.clear();
    _farTerrainProtectedFinalTerrainRegions.clear();
    _farTerrainCriticalResidencyCoordinates.clear();
    _farTerrainCriticalResidencyTargets.clear();
    _farTerrainCriticalResidencyCoordinateScratch.clear();
    _farTerrainCriticalResidencyTargetScratch.clear();
    _farTerrainCriticalResidencyKeys.clear();
    _farTerrainRefinementSubmissionKeys.clear();
    _farTerrainCachedMeshes.clear();
    _farTerrainCachedCanopies.clear();
    _farTerrainCenterTile.reset();
    clearExactSectionOwnership();
    _exactCollisionOwnedSections.clear();
    _publishedExactCollisionOwnedSections.clear();
    _publishedExactCollisionCoverageEpoch.reset();
    _farTerrainResidentWantedCount = 0;
    _farTerrainResidentRefinementCount = 0;
    _farTerrainPlannerTimings.clear();
    _farTerrainSelectionTimings.clear();
    _farTerrainPublicationTimings.clear();
    _farTerrainResidencyTimings.clear();
    _farTerrainArenaAdmissionDeniedCount = 0;
    _farTerrainNearArenaReclaimCount = 0;
    _farTerrainNearArenaReclaimedBytes = 0;

    if (!_farMegaBuffer) {
        _farMegaBuffer = std::make_unique<SegmentedMegaBuffer>(
            _device, FAR_VERTEX_BUFFER_BYTES, FAR_INDEX_BUFFER_BYTES, FAR_VERTEX_BUFFER_SLAB_BYTES,
            FAR_INDEX_BUFFER_SLAB_BYTES);
    }
    _farTerrainScheduler = std::make_unique<FarTerrainScheduler>(
        worldSeed, std::move(generationContext), FarTerrainSchedulerLimits{},
        world.getGenerationSettings());
    _farTerrainSeed = worldSeed;
    _farTerrainWorldInstanceId = worldInstanceId;
    _farTerrainMeshes.reserve(8192);
    _farCanopyAttachments.reserve(8192);
    _farCanopyLodFallbacks.reserve(4096);
    _farTerrainWanted.reserve(FAR_TERRAIN_MAX_RESIDENCY_KEYS);
    _farTerrainPriorityOrder.reserve(FAR_TERRAIN_MAX_RESIDENCY_KEYS);
    _farTerrainActiveTiles.reserve(4096);
    _farTerrainDesiredByTile.reserve(4096);
    _farTerrainDisplayedByTile.reserve(4096);
    _farTerrainTransitions.reserve(64);
    _farTerrainConnectedNearPatchTargets.reserve(128);
    _farTerrainPredictedNearPatchTargets.reserve(FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    _farTerrainPredictedCriticalResidencyKeys.reserve(512);
    _farTerrainCriticalResidencyCoordinates.reserve(256);
    _farTerrainCriticalResidencyTargets.reserve(256);
    _farTerrainCriticalResidencyCoordinateScratch.reserve(256);
    _farTerrainCriticalResidencyTargetScratch.reserve(256);
    _farTerrainCriticalResidencyKeys.reserve(1024);
    _farShadowDrawPlans.reserve(8192);
    _farTerrainNearGraceStartedAt.reserve(4096);
    _farTerrainCandidates.reserve(4096);
    _farTerrainCachedBaseRequests.reserve(4096);
    _farTerrainMissingBaseRequests.reserve(4096);
    _farTerrainDistantBaseRequests.reserve(4096);
    _farTerrainFinalBaseRequests.reserve(4096);
    _farTerrainPerceptualFinalRequests.reserve(1024);
    _farTerrainFinalRefinementRequests.reserve(8192);
    _farTerrainCanopyRefreshRequests.reserve(FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET);
    _farTerrainCanopyRefreshKeys.reserve(FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET);
    _farTerrainUrgentRefinementRequests.reserve(4096);
    _farTerrainUrgentRefinementKeys.reserve(FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
    _farTerrainRefinementSubmissionKeys.reserve(FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
    _farTerrainCachedMeshes.reserve(FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME);
    _farTerrainCachedCanopies.reserve(FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET);
    _exactOwnedSections.reserve(MAX_MESH_RESIDENT_CUBES);
    _exactShadowOwnedSections.reserve(MAX_MESH_RESIDENT_CUBES);
    _exactCollisionOwnedSections.reserve(MAX_MESH_RESIDENT_CUBES);
    _publishedExactCollisionOwnedSections.reserve(MAX_MESH_RESIDENT_CUBES);
}

void RenderPipeline::setExactSectionOwned(ChunkPos position, bool owned) {
    if (owned) {
        if (_exactOwnedSections.insert(position).second) {
            _farTerrainExactCoverage.setSectionReady(position, true);
            _farTerrainExactFloraCoverage.setSectionReady(position, true);
        }
        return;
    }
    if (_exactOwnedSections.erase(position) != 0) {
        _farTerrainExactCoverage.setSectionReady(position, false);
        _farTerrainExactFloraCoverage.setSectionReady(position, false);
    }
}

void RenderPipeline::clearExactSectionOwnership() {
    _exactOwnedSections.clear();
    _farTerrainExactCoverage.clear();
    _farTerrainExactFloraCoverage.clear();
}

bool RenderPipeline::updateFarTerrainSelection(const Vec3& cameraPosition, int visibleChunks) {
    const bool selectionChanged = farTerrainSelectionRequiresRefresh(
        _farTerrainSelectionCamera, cameraPosition.x, cameraPosition.z,
        _farTerrainSelectionViewDistance, visibleChunks);
    if (!selectionChanged)
        return false;
    selectFarTerrainView(cameraPosition.x, cameraPosition.z, visibleChunks, _farTerrainCandidates);
    _farTerrainSelectionCamera = std::pair{cameraPosition.x, cameraPosition.z};
    _farTerrainSelectionViewDistance = visibleChunks;
    ++_farTerrainViewEpoch;
    return true;
}

void RenderPipeline::renderFarTerrain(
    id<MTLRenderCommandEncoder> encoder, const World& world, const Camera& camera,
    const float fogColor[3], bool drawGeometry, int selectedViewDistance,
    std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactCoverageOverride,
    bool prepareProtectedFinal) {
    const Vec3 cameraPosition = camera.getPosition();
    resetFarTerrain(world);
    const bool finalStreamingWorkEnabled =
        farTerrainFinalStreamingWorkEnabled(drawGeometry, prepareProtectedFinal);
    _farTerrainScheduler->setFinalStreamingWorkEnabled(finalStreamingWorkEnabled);
    if (!finalStreamingWorkEnabled) {
        // Preparation publishes only terrain and canonical water. Disable the
        // optional and FINAL lanes before completed PREVIEW parents can enqueue
        // follow-up work on their gameplay budgets.
        _farTerrainScheduler->setCanopyWorkerBudget(0);
        if (_farTerrainProtectedNearHandoff.statusCenter()) {
            _farTerrainProtectedNearHandoff.clear();
            _farTerrainDesiredMetricsDirty = true;
        }
        _farTerrainProtectedNearEpoch = 0;
        _farTerrainProtectedRecentMotionX = 0;
        _farTerrainProtectedRecentMotionZ = 0;
        _farTerrainPredictedNearAnchor.reset();
        _farTerrainPredictedNearPatchTargets.clear();
        _farTerrainPredictedCriticalResidencyKeys.clear();
        _farTerrainProtectedNearClosureSnapshot.reset();
    }
    _farMegaBuffer->drainDeferredFrees(_frameRing.completedFrame());
    const auto farPlannerStartedAt = std::chrono::steady_clock::now();

    const int exactChunks = world.getExactViewDistance();
    const int configuredVisibleChunks = world.generationContext()
                                            ? farTerrainEntryHorizonViewDistance(
                                                  world.getViewDistance())
                                            : world.getViewDistance();
    const int visibleChunks = selectedViewDistance < 0
                                  ? configuredVisibleChunks
                                  : std::clamp(selectedViewDistance, 0, configuredVisibleChunks);
    const auto exactCoverage = exactCoverageOverride ? std::move(exactCoverageOverride)
                                                     : world.getExactSurfaceCoverageSnapshot();
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
    if (!_farTerrainExactFloraCoverage.matches(exactCoverageEpoch, nominalExactChunks)) {
        _farTerrainExactFloraCoverage.rebuild(
            exactCoverageEpoch, nominalExactChunks,
            exactCoverage ? std::span<const ChunkPos>(exactCoverage->floraRequiredSections)
                          : std::span<const ChunkPos>(),
            exactCoverage ? std::span<const ColumnPos>(exactCoverage->unresolvedColumns)
                          : std::span<const ColumnPos>(),
            [&](ChunkPos position) { return _exactOwnedSections.contains(position); });
    }
    const FarTerrainExactHandoff& exactHandoff =
        _farTerrainExactCoverage.sample(cameraPosition.x, cameraPosition.z);
    const FarTerrainExactHandoff& exactFloraHandoff =
        _farTerrainExactFloraCoverage.sample(cameraPosition.x, cameraPosition.z);
    const MeshSchedulerStats exactMeshStats =
        _meshScheduler ? _meshScheduler->stats() : MeshSchedulerStats{};
    const bool exactStreamingBusy = farTerrainExactStreamingBusy(
        world.getPendingChunkCount(), exactMeshStats.schedulerOwned,
        std::max(exactMeshStats.consumerPending, _pendingResults.size()),
        exactHandoff.requiredSections, exactHandoff.readySections, exactHandoff.unresolvedColumns);
    // Exact generation and meshing share physical cores with far construction.
    // Apply the exact-debt floor before this frame computes finer local debt;
    // the completed planner policy below may reduce the budget further but can
    // return to the full horizon pool only after both debts clear.
    _farTerrainScheduler->setWorkerBudget(
        farTerrainWorkerBudget(exactStreamingBusy, _farTerrainLocalTerrainDebt));
    const TerrainHorizonViewpoint viewpoint{cameraPosition.x, cameraPosition.y, cameraPosition.z};
    const int previousSelectionViewDistance = _farTerrainSelectionViewDistance;
    const std::optional<std::pair<double, double>> previousSelectionCamera =
        _farTerrainSelectionCamera;
    const bool selectionChanged = updateFarTerrainSelection(cameraPosition, visibleChunks);
    if (selectionChanged) {
        _farTerrainActiveTiles.clear();
        for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
            _farTerrainActiveTiles.insert({tile.key.tileX, tile.key.tileZ});
        }
        if (!previousSelectionCamera) {
            _farTerrainProtectedRecentMotionX = 0;
            _farTerrainProtectedRecentMotionZ = 0;
        } else {
            const double movementX = cameraPosition.x - previousSelectionCamera->first;
            const double movementZ = cameraPosition.z - previousSelectionCamera->second;
            constexpr double MOTION_SAMPLE_DISTANCE_SQUARED =
                static_cast<double>(CHUNK_EDGE * CHUNK_EDGE);
            if (movementX * movementX + movementZ * movementZ >=
                MOTION_SAMPLE_DISTANCE_SQUARED) {
                _farTerrainProtectedRecentMotionX = (movementX > 0.0) - (movementX < 0.0);
                _farTerrainProtectedRecentMotionZ = (movementZ > 0.0) - (movementZ < 0.0);
            }
        }
    }
    const int64_t cameraBlockX = static_cast<int64_t>(std::floor(cameraPosition.x));
    const int64_t cameraBlockZ = static_cast<int64_t>(std::floor(cameraPosition.z));
    const ColumnPos centerTile{
        world_coord::floorDiv(cameraBlockX, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
        world_coord::floorDiv(cameraBlockZ, static_cast<int64_t>(FAR_TERRAIN_TILE_EDGE)),
    };
    const ColumnPos protectedAnchor =
        farTerrainProtectedNearAnchor(cameraBlockX, cameraBlockZ);
    const bool protectedHandoffChanged =
        finalStreamingWorkEnabled && _farTerrainProtectedNearHandoff.request(protectedAnchor);
    if (protectedHandoffChanged) {
        // The desired map owns scheduler residency. Recompute it immediately so
        // a moving camera retains the published patch while also requesting the
        // complete replacement patch and shell.
        _farTerrainDesiredMetricsDirty = true;
    }
    std::optional<ColumnPos> predictedProtectedAnchor;
    if (drawGeometry && finalStreamingWorkEnabled) {
        predictedProtectedAnchor = farTerrainPredictedProtectedNearAnchor(
            cameraBlockX, cameraBlockZ, _farTerrainProtectedRecentMotionX,
            _farTerrainProtectedRecentMotionZ);
    }
    bool protectedPredictionChanged = false;
    if (predictedProtectedAnchor != _farTerrainPredictedNearAnchor ||
        (predictedProtectedAnchor && selectionChanged)) {
        std::vector<FarTerrainKey> predictedProtectedTargets;
        if (predictedProtectedAnchor) {
            buildFarTerrainProtectedNearTargets(*predictedProtectedAnchor, _farTerrainCandidates,
                                                predictedProtectedTargets);
            if (predictedProtectedTargets.size() != FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT) {
                predictedProtectedAnchor.reset();
                predictedProtectedTargets.clear();
            }
        }
        protectedPredictionChanged =
            predictedProtectedAnchor != _farTerrainPredictedNearAnchor ||
            predictedProtectedTargets != _farTerrainPredictedNearPatchTargets;
        if (protectedPredictionChanged) {
            _farTerrainPredictedNearAnchor = predictedProtectedAnchor;
            _farTerrainPredictedNearPatchTargets = std::move(predictedProtectedTargets);
            buildFarTerrainCriticalResidencyOrder(_farTerrainPredictedNearPatchTargets,
                                                  _farTerrainPredictedCriticalResidencyKeys);
        }
    }
    if (_farTerrainCenterTile && (std::abs(centerTile.x - _farTerrainCenterTile->x) > 2 ||
                                  std::abs(centerTile.z - _farTerrainCenterTile->z) > 2)) {
        _farTerrainScheduler->advanceEpoch();
    }
    _farTerrainCenterTile = centerTile;
    const double farViewportHeightPixels = static_cast<double>(std::max(_displayHeight, 1U));
    const double farVerticalFovRadians =
        static_cast<double>(camera.FOV()) * std::numbers::pi / 180.0;
    const double farProjectionScalePixels =
        farViewportHeightPixels / (2.0 * std::tan(farVerticalFovRadians * 0.5));
    const bool desiredMetricCameraChanged = farTerrainCameraMovementRequiresRefresh(
        _farTerrainDesiredMetricsCamera, cameraPosition.x, cameraPosition.z,
        FAR_TERRAIN_DESIRED_METRIC_REFRESH_BLOCKS);
    const bool recomputeDesiredMetrics = farTerrainDesiredMetricsRequireRefresh(
        selectionChanged || desiredMetricCameraChanged, _farTerrainDesiredMetricsDirty,
        _farTerrainDesiredViewportHeight, _displayHeight, _farTerrainDesiredVerticalFovRadians,
        farVerticalFovRadians, _farTerrainDesiredDrawGeometry, drawGeometry);
    const auto screenMetricsForTile =
        [&](const FarTerrainViewTile& tile) -> std::optional<FarTerrainScreenErrorMetrics> {
        const auto parent =
            _farTerrainMeshes.find({tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP});
        if (parent == _farTerrainMeshes.end() || !parent->second.uploaded)
            return std::nullopt;
        const FarTerrainKey parentKey{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        const auto promotion = _farTerrainAuthorityTransitions.find(parentKey);
        const bool previewMayBeDisplayed =
            parent->second.authorityQuality == FarTerrainAuthorityQuality::PREVIEW ||
            (promotion != _farTerrainAuthorityTransitions.end() &&
             promotion->second.source.authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
        const double relief =
            std::max(0.0, static_cast<double>(parent->second.surfaceBounds.maxY) -
                              static_cast<double>(parent->second.surfaceBounds.minY)) +
            FAR_TERRAIN_STEP32_RELIEF_ENVELOPE * 2.0 +
            (previewMayBeDisplayed ? FAR_TERRAIN_PREVIEW_RESIDUAL_MAX_BLOCKS * 2.0 : 0.0);
        return FarTerrainScreenErrorMetrics{
            // Absolute LOD bands start at the 32-chunk exact handoff, but a
            // temporarily unresolved exact column can be much closer. Use
            // its real nearest-point distance for projected error so the far
            // fallback refines toward block scale instead of pretending it is
            // half a kilometer away.
            .distanceBlocks = std::max(1.0, std::sqrt(tile.distanceSquared)),
            .viewportHeightPixels = farViewportHeightPixels,
            .verticalFovRadians = farVerticalFovRadians,
            .projectionScalePixels = farProjectionScalePixels,
            .tileReliefBlocks = relief,
        };
    };
    const auto currentTileDistanceSquared = [&](const FarTerrainBounds& bounds) {
        const double dx = cameraPosition.x < static_cast<double>(bounds.minX)
                              ? static_cast<double>(bounds.minX) - cameraPosition.x
                          : cameraPosition.x > static_cast<double>(bounds.maxX)
                              ? cameraPosition.x - static_cast<double>(bounds.maxX)
                              : 0.0;
        const double dz = cameraPosition.z < static_cast<double>(bounds.minZ)
                              ? static_cast<double>(bounds.minZ) - cameraPosition.z
                          : cameraPosition.z > static_cast<double>(bounds.maxZ)
                              ? cameraPosition.z - static_cast<double>(bounds.maxZ)
                              : 0.0;
        return dx * dx + dz * dz;
    };
    bool desiredTierChanged = false;
    for (FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        tile.distanceSquared = currentTileDistanceSquared(tile.bounds);
        tile.distanceChunks =
            std::max(static_cast<double>(FAR_TERRAIN_NEAR_CHUNK_RADIUS),
                     std::sqrt(tile.distanceSquared) / static_cast<double>(CHUNK_EDGE));

        if (recomputeDesiredMetrics) {
            std::optional<FarTerrainStep> previousStep;
            if (const auto previous = _farTerrainDesiredByTile.find(coordinate);
                previous != _farTerrainDesiredByTile.end()) {
                previousStep = previous->second.step;
            }
            tile.screenErrorMetrics = drawGeometry ? screenMetricsForTile(tile) : std::nullopt;
            std::optional<FarTerrainStep> desired =
                tile.screenErrorMetrics
                    ? farTerrainStepForScreenMetrics(tile.distanceChunks, *tile.screenErrorMetrics,
                                                     previousStep)
                    : farTerrainStepForMetrics(tile.distanceChunks, previousStep);
            desired = farTerrainProtectedDesiredStep(
                desired,
                farTerrainProtectedNearRequiredStep(_farTerrainProtectedNearHandoff, coordinate));
            if (desired) {
                tile.key.step = *desired;
            }
            desiredTierChanged =
                desiredTierChanged || !previousStep || *previousStep != tile.key.step;
            _farTerrainDesiredByTile.insert_or_assign(coordinate, tile.key);
        }
    }
    if (recomputeDesiredMetrics) {
        _farTerrainDesiredMetricsCamera = std::pair{cameraPosition.x, cameraPosition.z};
        _farTerrainDesiredViewportHeight = _displayHeight;
        _farTerrainDesiredVerticalFovRadians = farVerticalFovRadians;
        _farTerrainDesiredDrawGeometry = drawGeometry;
        _farTerrainDesiredMetricsDirty = false;
    }
    if (selectionChanged) {
        for (auto it = _farTerrainDesiredByTile.begin(); it != _farTerrainDesiredByTile.end();) {
            if (!_farTerrainActiveTiles.contains(it->first)) {
                it = _farTerrainDesiredByTile.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Every visible coordinate owns a step-32 parent independently of its
    // desired detail. Parents occupy the first priority lane. Camera,
    // protected, and unresolved exact-fallback refinements form a second
    // critical lane before the broad global bridge wavefront.
    const bool residencyMembershipChanged =
        (selectionChanged || desiredTierChanged || protectedPredictionChanged) &&
        !farTerrainResidencyMembershipMatches(_farTerrainCandidates, _farTerrainWanted,
                                              _farTerrainPredictedCriticalResidencyKeys);
    std::vector<ColumnPos>& criticalResidencyCoordinates =
        _farTerrainCriticalResidencyCoordinateScratch;
    std::vector<FarTerrainKey>& criticalResidencyTargets =
        _farTerrainCriticalResidencyTargetScratch;
    criticalResidencyCoordinates.clear();
    criticalResidencyTargets.clear();
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const bool critical =
            coordinate == centerTile ||
            farTerrainProtectedNearRequiredStep(_farTerrainProtectedNearHandoff, coordinate)
                .has_value() ||
            farTerrainRequiresCoverageParent(cameraPosition.x, cameraPosition.z, coordinate,
                                             nominalExactBlocks, exactHandoff);
        if (critical) {
            criticalResidencyCoordinates.push_back(coordinate);
            criticalResidencyTargets.push_back(
                {coordinate.x, coordinate.z, farTerrainResidencyTarget(tile)});
        }
    }
    const bool criticalResidencyChanged =
        criticalResidencyCoordinates != _farTerrainCriticalResidencyCoordinates ||
        criticalResidencyTargets != _farTerrainCriticalResidencyTargets;
    if (criticalResidencyChanged) {
        _farTerrainCriticalResidencyCoordinates.swap(criticalResidencyCoordinates);
        _farTerrainCriticalResidencyTargets.swap(criticalResidencyTargets);
    }
    const bool criticalPriorityChanged = criticalResidencyChanged || residencyMembershipChanged ||
                                         protectedHandoffChanged || protectedPredictionChanged;
    if (criticalPriorityChanged) {
        // Critical cache rank is independent of the global base-first
        // submission order. Preserve every required surface before any
        // support copy, then every parent before intermediate bridges. Under
        // exceptional pressure, an adjacent core target can therefore never
        // lose admission to another coordinate's nonpublishable lineage.
        buildFarTerrainTieredCriticalResidencyOrder(
            _farTerrainCriticalResidencyTargets, _farTerrainPredictedNearPatchTargets,
            _farTerrainCriticalResidencyKeys);
    }
    if (residencyMembershipChanged) {
        buildFarTerrainResidencyOrder(_farTerrainCandidates, _farTerrainPriorityOrder,
                                      _farTerrainCriticalResidencyCoordinates);
        _farTerrainWanted.clear();
        _farTerrainWanted.insert(_farTerrainPriorityOrder.begin(), _farTerrainPriorityOrder.end());
        for (const FarTerrainKey key : _farTerrainPredictedCriticalResidencyKeys) {
            if (_farTerrainWanted.insert(key).second)
                _farTerrainPriorityOrder.push_back(key);
        }
        _farTerrainScheduler->retainWanted(_farTerrainWanted, _farTerrainPriorityOrder,
                                           _farTerrainCriticalResidencyKeys);
        // Pause optional dispatch while wanted membership changes. The frame's
        // local-debt policy below may re-admit one worker during gameplay only
        // after the new protected and connected requirements are known.
        _farTerrainScheduler->setCanopyWorkerBudget(0);
    } else if (criticalPriorityChanged) {
        _farTerrainScheduler->refreshCriticalPriorities(_farTerrainCriticalResidencyKeys);
    }
    if (protectedHandoffChanged) {
        // Publish the new critical membership before advancing the movement
        // epoch. The scheduler can then retag exact overlapping work instead
        // of canceling and rebuilding it at every half-tile boundary.
        _farTerrainProtectedNearEpoch = _farTerrainScheduler->advanceProtectedHandoffEpoch();
    }
    if (selectionChanged) {
        std::vector<worldgen::learned::TerrainPageCoordinate> visibleAuthorityPages =
            farTerrainCoarseAuthorityPages(_farTerrainCandidates, cameraPosition.x,
                                           cameraPosition.z);
        std::vector<worldgen::learned::TerrainPageCoordinate> speculativeAuthorityPages;
        bool replaceSpeculativeAuthorityPages = false;
        const bool viewDistanceChanged = previousSelectionViewDistance != visibleChunks;
        if (viewDistanceChanged || !_farTerrainSpeculativeCamera) {
            replaceSpeculativeAuthorityPages = true;
            _farTerrainSpeculativeCamera = std::pair{cameraPosition.x, cameraPosition.z};
        } else {
            const double movementX = cameraPosition.x - _farTerrainSpeculativeCamera->first;
            const double movementZ = cameraPosition.z - _farTerrainSpeculativeCamera->second;
            constexpr double SPECULATIVE_UPDATE_DISTANCE_SQUARED =
                static_cast<double>(CHUNK_EDGE * CHUNK_EDGE);
            if (movementX * movementX + movementZ * movementZ >=
                SPECULATIVE_UPDATE_DISTANCE_SQUARED) {
                speculativeAuthorityPages = farTerrainSpeculativeAuthorityPages(
                    visibleAuthorityPages, _farTerrainSpeculativeCamera->first,
                    _farTerrainSpeculativeCamera->second, cameraPosition.x, cameraPosition.z);
                replaceSpeculativeAuthorityPages = true;
                _farTerrainSpeculativeCamera = std::pair{cameraPosition.x, cameraPosition.z};
            }
        }
        _farTerrainScheduler->setCoarseAuthorityPrefetchPages(std::move(visibleAuthorityPages));
        if (replaceSpeculativeAuthorityPages) {
            _farTerrainScheduler->setSpeculativeAuthorityPrefetchPages(
                std::move(speculativeAuthorityPages));
        }
    }
    if (finalStreamingWorkEnabled)
        _farTerrainScheduler->pumpFinalBaseAuthority();
    // Preparation fills the bounded entry prefix. Once gameplay opens, exact
    // publication and nearer desired LODs own authority before the remaining
    // configured horizon can enqueue more preview pages.
    if (!drawGeometry || (!exactStreamingBusy && !_farTerrainLocalTerrainDebt))
        _farTerrainScheduler->pumpCoarseAuthorityPrefetch();
    // Resuming a parked canopy only moves completed authority work back into
    // its optional queue. Do this every gameplay frame so unrelated visible
    // FINAL debt cannot strand nearby flora indefinitely.
    if (drawGeometry)
        _farTerrainScheduler->pumpCanopyAuthority();
    // Current-frustum canonical replacements still enter the single-flight
    // inference coordinator before speculative movement prediction.
    if (drawGeometry && _farTerrainPerceptualFinalRequests.empty()) {
        _farTerrainScheduler->pumpSpeculativeAuthorityPrefetch();
    }

    auto isResident = [&](const FarTerrainKey& key) {
        const auto found = _farTerrainMeshes.find(key);
        return found != _farTerrainMeshes.end() && found->second.uploaded;
    };
    const auto isFinalResident = [&](const FarTerrainKey& key) {
        const auto found = _farTerrainMeshes.find(key);
        return found != _farTerrainMeshes.end() && found->second.uploaded &&
               found->second.authorityQuality == FarTerrainAuthorityQuality::FINAL;
    };
    const auto isFinalBaseResident = [&](ColumnPos coordinate) {
        return isFinalResident({coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP});
    };
    const auto displayedQualityFor = [&](ColumnPos coordinate, FarTerrainStep step) noexcept {
        const FarTerrainKey key{coordinate.x, coordinate.z, step};
        if (const auto promotion = _farTerrainAuthorityTransitions.find(key);
            promotion != _farTerrainAuthorityTransitions.end()) {
            return promotion->second.source.authorityQuality;
        }
        const auto resident = _farTerrainMeshes.find(key);
        return resident == _farTerrainMeshes.end() ? FarTerrainAuthorityQuality::PREVIEW
                                                   : resident->second.authorityQuality;
    };
    const auto parentAuthorityAllowsDisplayedStep =
        [&](ColumnPos coordinate, FarTerrainAuthorityQuality child, FarTerrainStep step) noexcept {
            const FarTerrainKey parentKey{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
            const auto parent = _farTerrainMeshes.find(parentKey);
            if (parent == _farTerrainMeshes.end() || !parent->second.uploaded)
                return false;
            std::optional<FarTerrainAuthorityQuality> sourceQuality;
            if (const auto promotion = _farTerrainAuthorityTransitions.find(parentKey);
                promotion != _farTerrainAuthorityTransitions.end()) {
                sourceQuality = promotion->second.source.authorityQuality;
            }
            return farTerrainAuthorityAllowsDisplayedStepDuringParentPromotion(
                parent->second.authorityQuality, sourceQuality, child, step);
        };
    const auto tileVisibleForScheduling = [&](ColumnPos coordinate) {
        const auto parent =
            _farTerrainMeshes.find({coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP});
        if (parent == _farTerrainMeshes.end() || !parent->second.uploaded)
            return false;
        const FarTerrainBounds& bounds = parent->second.surfaceBounds;
        const float previewVerticalExpansion =
            displayedQualityFor(coordinate, FAR_TERRAIN_BASE_STEP) ==
                    FarTerrainAuthorityQuality::PREVIEW
                ? static_cast<float>(FAR_TERRAIN_PREVIEW_RESIDUAL_MAX_BLOCKS)
                : 0.0F;
        return isChunkInFrustum(
            {{static_cast<float>(bounds.minX), bounds.minY - previewVerticalExpansion,
              static_cast<float>(bounds.minZ)},
             {static_cast<float>(bounds.maxX), bounds.maxY + previewVerticalExpansion,
              static_cast<float>(bounds.maxZ)}});
    };
    // Capture the closure identity before evaluating any of its targets. The
    // resulting anchor, counts, and epoch remain one logical snapshot even if
    // the requested handoff commits later in this render pass.
    const std::optional<ColumnPos> protectedClosureAnchor =
        _farTerrainProtectedNearHandoff.statusCenter();
    const uint64_t protectedClosureViewEpoch = protectedClosureAnchor ? _farTerrainViewEpoch : 0;
    const uint64_t protectedClosureWorldEpoch = protectedClosureAnchor ? _farTerrainWorldEpoch : 0;
    const uint64_t protectedClosureEpoch =
        protectedClosureAnchor ? _farTerrainProtectedNearEpoch : 0;
    _farTerrainConnectedNearPatchTargets.clear();
    if (protectedClosureAnchor) {
        buildFarTerrainProtectedNearTargets(*protectedClosureAnchor, _farTerrainCandidates,
                                            _farTerrainConnectedNearPatchTargets);
        if (protectedHandoffChanged || selectionChanged ||
            _farTerrainProtectedFinalTerrainRegions.empty()) {
            _farTerrainProtectedFinalTerrainRegions = farTerrainProtectedFinalTerrainRegions(
                _farTerrainConnectedNearPatchTargets);
        }
        if (const auto generationContext = world.generationContext()) {
            const worldgen::learned::ProtectedHandoffEpoch epoch{protectedClosureEpoch};
            for (const worldgen::learned::NativeRect region :
                 _farTerrainProtectedFinalTerrainRegions) {
                const auto prepared = generationContext->queryTransientFinalNativeGrid(
                    region, worldgen::learned::AuthorityRequestPriority::PROTECTED_HANDOFF,
                    epoch);
                if (prepared.status() == worldgen::learned::AuthorityStatus::FAILED) {
                    generationContext->latchFailure(
                        prepared.failure()
                            ? *prepared.failure()
                            : worldgen::learned::GenerationFailure{
                                  .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                                  .message =
                                      "Protected FINAL terrain prewarm failed without a reason",
                                  .retriable = true});
                    break;
                }
            }
        }
    } else {
        _farTerrainProtectedFinalTerrainRegions.clear();
    }
    const auto predictedOnlyTarget = [&](FarTerrainKey key) {
        const bool predicted =
            std::ranges::find(_farTerrainPredictedNearPatchTargets, key) !=
            _farTerrainPredictedNearPatchTargets.end();
        const bool current = std::ranges::find(_farTerrainConnectedNearPatchTargets, key) !=
                             _farTerrainConnectedNearPatchTargets.end();
        return predicted && !current;
    };
    _farTerrainPerceptualFinalRequests.clear();
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (!tile.screenErrorMetrics)
            continue;
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        if (!tileVisibleForScheduling(coordinate))
            continue;
        if (const std::optional<ColumnPos> protectedCenter =
                _farTerrainProtectedNearHandoff.statusCenter();
            protectedCenter && farTerrainProtectedNearRole(*protectedCenter, coordinate) !=
                                   FarTerrainProtectedNearRole::NONE) {
            continue;
        }
        FarTerrainStep displayed = FAR_TERRAIN_BASE_STEP;
        if (const auto found = _farTerrainDisplayedByTile.find(coordinate);
            found != _farTerrainDisplayedByTile.end() && isResident(found->second)) {
            displayed = found->second.step;
        }
        if (displayedQualityFor(coordinate, displayed) != FarTerrainAuthorityQuality::PREVIEW)
            continue;
        const double projectedError = farTerrainProjectedDisplayErrorPixels(
            displayed, FarTerrainAuthorityQuality::PREVIEW, *tile.screenErrorMetrics);
        if (projectedError <= FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS)
            continue;
        _farTerrainPerceptualFinalRequests.push_back({
            .coordinate = coordinate,
            .displayed = displayed,
            .desired = tile.key.step,
            .cameraTile = coordinate == centerTile,
            .visible = true,
            .projectedErrorPixels = projectedError,
            .distanceSquaredBlocks = tile.distanceSquared,
        });
    }
    std::sort(_farTerrainPerceptualFinalRequests.begin(), _farTerrainPerceptualFinalRequests.end(),
              [](const auto& first, const auto& second) {
                  if (first.cameraTile != second.cameraTile)
                      return first.cameraTile;
                  if (first.projectedErrorPixels != second.projectedErrorPixels)
                      return first.projectedErrorPixels > second.projectedErrorPixels;
                  if (first.distanceSquaredBlocks != second.distanceSquaredBlocks)
                      return first.distanceSquaredBlocks < second.distanceSquaredBlocks;
                  if (first.coordinate.x != second.coordinate.x)
                      return first.coordinate.x < second.coordinate.x;
                  return first.coordinate.z < second.coordinate.z;
              });
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
    const auto farPlannerSelectionCompletedAt = std::chrono::steady_clock::now();

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
    const auto canopyStateHasGeometry = [](const FarCanopyMeshState* canopy) {
        return canopy && canopy->alloc && canopy->alloc->indexCount != 0U;
    };
    const auto fallbackCanopyFor =
        [&](ColumnPos coordinate,
            FarTerrainAuthorityQuality surfaceQuality) -> const FarCanopyMeshState* {
        const auto fallback = _farCanopyLodFallbacks.find(coordinate);
        if (fallback == _farCanopyLodFallbacks.end())
            return nullptr;
        const auto canopy = _farCanopyAttachments.find(fallback->second);
        if (canopy == _farCanopyAttachments.end() ||
            !canopyStateHasGeometry(&canopy->second) ||
            !farCanopyMatchesSurface(canopy->second.authorityQuality,
                                     canopy->second.groundingQuality, surfaceQuality)) {
            return nullptr;
        }
        return &canopy->second;
    };
    const auto retireCanopyFallback =
        [&](ColumnPos coordinate, std::optional<FarTerrainKey> replacement) {
        const auto fallback = _farCanopyLodFallbacks.find(coordinate);
        if (fallback == _farCanopyLodFallbacks.end())
            return;
        const FarTerrainKey source = fallback->second;
        _farCanopyLodFallbacks.erase(fallback);
        if (replacement && source == *replacement)
            return;
        const auto canopy = _farCanopyAttachments.find(source);
        if (canopy == _farCanopyAttachments.end())
            return;
        if (canopy->second.alloc) {
            _farMegaBuffer->deferFree(*canopy->second.alloc, _frameRing.frameIndex());
        }
        _farCanopyAttachments.erase(canopy);
    };
    const auto clearCanopyFallbackReference = [&](FarTerrainKey key) {
        const ColumnPos coordinate{key.tileX, key.tileZ};
        const auto fallback = _farCanopyLodFallbacks.find(coordinate);
        if (fallback != _farCanopyLodFallbacks.end() && fallback->second == key) {
            _farCanopyLodFallbacks.erase(fallback);
        }
    };
    const auto completeCanopyLodFallback = [&](ColumnPos coordinate, FarTerrainKey sourceKey,
                                               FarTerrainKey targetKey) {
        const auto targetSurface = _farTerrainMeshes.find(targetKey);
        if (targetSurface == _farTerrainMeshes.end() || !targetSurface->second.uploaded)
            return;
        const FarTerrainAuthorityQuality surfaceQuality = targetSurface->second.authorityQuality;
        if (_farCanopyLodFallbacks.contains(coordinate) &&
            !fallbackCanopyFor(coordinate, surfaceQuality)) {
            retireCanopyFallback(coordinate, std::nullopt);
        }
        const auto source = _farCanopyAttachments.find(sourceKey);
        const bool sourcePresent =
            source != _farCanopyAttachments.end() &&
            canopyStateHasGeometry(&source->second) &&
            farCanopyMatchesSurface(source->second.authorityQuality,
                                    source->second.groundingQuality, surfaceQuality);
        const auto target = _farCanopyAttachments.find(targetKey);
        const bool targetPresent =
            target != _farCanopyAttachments.end() &&
            farCanopyMatchesSurface(target->second.authorityQuality,
                                    target->second.groundingQuality, surfaceQuality);
        const bool fallbackPresent = _farCanopyLodFallbacks.contains(coordinate);
        switch (farCanopyLodCompletionAction(fallbackPresent, sourcePresent, targetPresent)) {
            case FarCanopyLodCompletionAction::ADOPT_SOURCE:
                _farCanopyLodFallbacks.insert_or_assign(coordinate, sourceKey);
                break;
            case FarCanopyLodCompletionAction::RETIRE_FALLBACK:
                retireCanopyFallback(coordinate, targetKey);
                break;
            case FarCanopyLodCompletionAction::RETAIN_FALLBACK:
            case FarCanopyLodCompletionAction::NONE:
                break;
        }
    };
    constexpr double LOCAL_UPLOAD_RADIUS_BLOCKS =
        FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS * CHUNK_EDGE;
    constexpr double LOCAL_UPLOAD_RADIUS_SQUARED =
        LOCAL_UPLOAD_RADIUS_BLOCKS * LOCAL_UPLOAD_RADIUS_BLOCKS;
    size_t uploads = 0;
    size_t baseUploads = 0;
    size_t refinementUploads = 0;
    size_t canopyUploads = 0;
    size_t uploadBytes = 0;
    bool uploadFailureLogged = false;
    uint64_t farArenaVertexUsed = _farMegaBuffer->vertexUsed();
    uint64_t farArenaIndexUsed = _farMegaBuffer->indexUsed();
    const auto alignedArenaBytes = [](uint64_t bytes) {
        return (bytes + MegaBuffer::ALIGNMENT - 1) & ~(MegaBuffer::ALIGNMENT - 1);
    };
    struct FarTerrainGpuVictim {
        FarTerrainKey key;
        bool canopy = false;
        bool demoteToParent = false;
        uint8_t retirementClass = 0;
        double distanceSquared = 0.0;
        uint64_t vertexBytes = 0;
        uint64_t indexBytes = 0;
    };
    const auto protectedClosureTarget = [&](FarTerrainKey key) {
        return farTerrainProtectedNearTargetKey(
                   _farTerrainProtectedNearHandoff.activeCenter(), key) ||
               farTerrainProtectedNearTargetKey(
                   _farTerrainProtectedNearHandoff.requestedCenter(), key);
    };
    const auto reclaimOptionalGpuResidencyForNear = [&](FarTerrainKey preserveKey,
                                                        uint64_t targetVertexBytes,
                                                        uint64_t targetIndexBytes) {
        std::vector<FarTerrainGpuVictim> victims;
        victims.reserve(_farCanopyAttachments.size() + _farTerrainMeshes.size() / 4);
        for (const auto& [key, canopy] : _farCanopyAttachments) {
            if (key == preserveKey || !canopy.alloc)
                continue;
            const ColumnPos coordinate{key.tileX, key.tileZ};
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            const bool current =
                displayed != _farTerrainDisplayedByTile.end() && displayed->second == key;
            const auto transition = _farTerrainTransitions.find(coordinate);
            const bool transitionEndpoint =
                transition != _farTerrainTransitions.end() &&
                (transition->second.from == key || transition->second.to == key);
            const double distanceSquared = distanceSquaredForKey(key);
            if (transitionEndpoint || _farTerrainAuthorityTransitions.contains(key)) {
                continue;
            }
            // Canopy is always optional relative to terrain and water. Retire
            // distant attachments first, but allow even a current nearby
            // attachment to yield when it is the last way to publish protected
            // terrain. Its independent queue can rebuild it afterward.
            victims.push_back({
                .key = key,
                .canopy = true,
                .demoteToParent = false,
                .retirementClass = static_cast<uint8_t>(current ? 1 : 0),
                .distanceSquared = distanceSquared,
                .vertexBytes = alignedArenaBytes(static_cast<uint64_t>(canopy.alloc->vertexCount) *
                                                 sizeof(Vertex)),
                .indexBytes = alignedArenaBytes(static_cast<uint64_t>(canopy.alloc->indexCount) *
                                                sizeof(uint32_t)),
            });
        }
        for (const auto& [key, state] : _farTerrainMeshes) {
            if (key == preserveKey || !state.uploaded || farTerrainIsBaseStep(key.step))
                continue;
            const ColumnPos coordinate{key.tileX, key.tileZ};
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            const bool isDisplayed =
                displayed != _farTerrainDisplayedByTile.end() && displayed->second == key;
            const auto transition = _farTerrainTransitions.find(coordinate);
            const bool lodTransitionEndpoint =
                transition != _farTerrainTransitions.end() &&
                (transition->second.from == key || transition->second.to == key);
            const bool authorityTransitionEndpoint = _farTerrainAuthorityTransitions.contains(key);
            const bool protectedClosure = protectedClosureTarget(key);
            const bool exactFallback = farTerrainRequiresCoverageParent(
                cameraPosition.x, cameraPosition.z, coordinate, nominalExactBlocks, exactHandoff);
            const bool nextCritical = farTerrainProtectedNearTargetKey(
                _farTerrainProtectedNearHandoff.requestedCenter(), key);
            bool demoteToParent = false;
            if (isDisplayed) {
                constexpr std::array<ColumnPos, 4> NEIGHBOR_OFFSETS = {
                    ColumnPos{1, 0}, ColumnPos{-1, 0}, ColumnPos{0, 1}, ColumnPos{0, -1}};
                std::array<std::optional<FarTerrainStep>, 4> neighborSteps;
                bool neighborTransition = false;
                for (size_t edge = 0; edge < NEIGHBOR_OFFSETS.size(); ++edge) {
                    const ColumnPos neighbor{coordinate.x + NEIGHBOR_OFFSETS[edge].x,
                                             coordinate.z + NEIGHBOR_OFFSETS[edge].z};
                    if (_farTerrainTransitions.contains(neighbor)) {
                        neighborTransition = true;
                        continue;
                    }
                    if (const auto neighborDisplayed = _farTerrainDisplayedByTile.find(neighbor);
                        neighborDisplayed != _farTerrainDisplayedByTile.end()) {
                        neighborSteps[edge] = neighborDisplayed->second.step;
                    }
                }
                const FarTerrainKey parent{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
                const auto parentResident = _farTerrainMeshes.find(parent);
                const bool parentReady =
                    parentResident != _farTerrainMeshes.end() && parentResident->second.uploaded &&
                    !_farTerrainAuthorityTransitions.contains(parent);
                demoteToParent = farTerrainDisplayedRefinementMayYieldToParentForNear(
                    key.step, neighborSteps, parentReady, lodTransitionEndpoint,
                    neighborTransition, authorityTransitionEndpoint, protectedClosure,
                    exactFallback, nextCritical);
            }
            if (!demoteToParent &&
                !farTerrainGpuMayEvictForNear(false, isDisplayed, lodTransitionEndpoint,
                                              authorityTransitionEndpoint, protectedClosure,
                                              exactFallback, nextCritical)) {
                continue;
            }
            const auto desired = _farTerrainDesiredByTile.find(coordinate);
            const bool desiredSurface =
                desired != _farTerrainDesiredByTile.end() && desired->second == key;
            const bool visible = tileVisibleForScheduling(coordinate);
            if (desiredSurface && visible && !demoteToParent)
                continue;
            const bool wanted = _farTerrainWanted.contains(key);
            const uint8_t retirementClass = !wanted           ? uint8_t{2}
                                            : !desiredSurface ? uint8_t{3}
                                                              : uint8_t{4};
            victims.push_back({
                .key = key,
                .canopy = false,
                .demoteToParent = demoteToParent,
                .retirementClass =
                    static_cast<uint8_t>(demoteToParent ? uint8_t{5} : retirementClass),
                .distanceSquared = distanceSquaredForKey(key),
                .vertexBytes = alignedArenaBytes(static_cast<uint64_t>(state.alloc.vertexCount) *
                                                 sizeof(Vertex)),
                .indexBytes = alignedArenaBytes(static_cast<uint64_t>(state.alloc.indexCount) *
                                                sizeof(uint32_t)),
            });
        }
        std::sort(victims.begin(), victims.end(), [](const auto& first, const auto& second) {
            if (first.retirementClass != second.retirementClass)
                return first.retirementClass < second.retirementClass;
            if (first.distanceSquared != second.distanceSquared)
                return first.distanceSquared > second.distanceSquared;
            if (first.key.tileX != second.key.tileX)
                return first.key.tileX < second.key.tileX;
            if (first.key.tileZ != second.key.tileZ)
                return first.key.tileZ < second.key.tileZ;
            return farTerrainStepSize(first.key.step) > farTerrainStepSize(second.key.step);
        });

        uint64_t reclaimedVertexBytes = 0;
        uint64_t reclaimedIndexBytes = 0;
        size_t retired = 0;
        for (const FarTerrainGpuVictim& victim : victims) {
            if (reclaimedVertexBytes >= targetVertexBytes &&
                reclaimedIndexBytes >= targetIndexBytes) {
                break;
            }
            if (victim.canopy) {
                const auto found = _farCanopyAttachments.find(victim.key);
                if (found == _farCanopyAttachments.end() || !found->second.alloc)
                    continue;
                _farMegaBuffer->deferFree(*found->second.alloc, _frameRing.frameIndex());
                _farCanopyAttachments.erase(found);
                clearCanopyFallbackReference(victim.key);
            } else {
                const auto found = _farTerrainMeshes.find(victim.key);
                if (found == _farTerrainMeshes.end() || !found->second.uploaded)
                    continue;
                if (victim.demoteToParent) {
                    const ColumnPos coordinate{victim.key.tileX, victim.key.tileZ};
                    const FarTerrainKey parent{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
                    const auto parentResident = _farTerrainMeshes.find(parent);
                    if (parentResident == _farTerrainMeshes.end() ||
                        !parentResident->second.uploaded) {
                        continue;
                    }
                    _farTerrainTransitions.erase(coordinate);
                    _farTerrainDisplayedByTile.insert_or_assign(coordinate, parent);
                }
                if (const auto canopy = _farCanopyAttachments.find(victim.key);
                    canopy != _farCanopyAttachments.end()) {
                    if (canopy->second.alloc) {
                        reclaimedVertexBytes += alignedArenaBytes(
                            static_cast<uint64_t>(canopy->second.alloc->vertexCount) *
                            sizeof(Vertex));
                        reclaimedIndexBytes += alignedArenaBytes(
                            static_cast<uint64_t>(canopy->second.alloc->indexCount) *
                            sizeof(uint32_t));
                        _farMegaBuffer->deferFree(*canopy->second.alloc, _frameRing.frameIndex());
                    }
                    _farCanopyAttachments.erase(canopy);
                    clearCanopyFallbackReference(victim.key);
                }
                if (_farTerrainWanted.contains(victim.key)) {
                    if (_farTerrainResidentWantedCount > 0)
                        --_farTerrainResidentWantedCount;
                    if (_farTerrainResidentRefinementCount > 0)
                        --_farTerrainResidentRefinementCount;
                }
                _farMegaBuffer->deferFree(found->second.alloc, _frameRing.frameIndex());
                _farTerrainMeshes.erase(found);
            }
            reclaimedVertexBytes += victim.vertexBytes;
            reclaimedIndexBytes += victim.indexBytes;
            ++retired;
        }
        if (retired != 0) {
            ++_farTerrainNearArenaReclaimCount;
            _farTerrainNearArenaReclaimedBytes += reclaimedVertexBytes + reclaimedIndexBytes;
        }
        return retired != 0;
    };
    auto uploadMesh = [&](const std::shared_ptr<const FarTerrainMesh>& mesh) {
        if (!mesh || _farTerrainWanted.count(mesh->key) == 0) {
            return false;
        }
        const bool base = farTerrainIsBaseStep(mesh->key.step);
        auto resident = _farTerrainMeshes.find(mesh->key);
        FarTerrainUploadAction uploadAction = farTerrainUploadAction(
            resident == _farTerrainMeshes.end() ? std::nullopt
                                                : std::optional{resident->second.authorityQuality},
            mesh->authorityQuality);
        if (uploadAction == FarTerrainUploadAction::REJECT)
            return false;
        const bool replacingPreviewParent =
            uploadAction == FarTerrainUploadAction::REPLACE_AFTER_UPLOAD;
        const bool coverageCritical = base && resident == _farTerrainMeshes.end();
        const ColumnPos coordinate{mesh->key.tileX, mesh->key.tileZ};
        const std::optional<ColumnPos> protectedCenter =
            _farTerrainProtectedNearHandoff.statusCenter();
        const bool protectedClosure =
            protectedCenter && farTerrainProtectedNearRole(*protectedCenter, coordinate) !=
                                   FarTerrainProtectedNearRole::NONE;
        const bool exactFallback = farTerrainRequiresCoverageParent(
            cameraPosition.x, cameraPosition.z, coordinate, nominalExactBlocks, exactHandoff);
        const bool localUpload = coordinate == centerTile || protectedClosure || exactFallback ||
                                 distanceSquaredForKey(mesh->key) <= LOCAL_UPLOAD_RADIUS_SQUARED;
        const bool criticalRefinement = farTerrainCriticalProtectedRefinement(
            _farTerrainProtectedNearHandoff.requestedCenter(), mesh->key,
            mesh->authorityQuality);
        const bool nearRefinement = !coverageCritical && !criticalRefinement && localUpload;
        const bool localCriticalCoverage = coverageCritical && localUpload;
        const auto displayedSurface = _farTerrainDisplayedByTile.find(coordinate);
        const auto lodTransition = _farTerrainTransitions.find(coordinate);
        const bool visiblePreviewChildDependsOnParent =
            base && displayedSurface != _farTerrainDisplayedByTile.end() &&
            displayedSurface->second != mesh->key &&
            displayedQualityFor(coordinate, displayedSurface->second.step) ==
                FarTerrainAuthorityQuality::PREVIEW;
        const bool replacementIsVisible =
            drawGeometry &&
            ((displayedSurface != _farTerrainDisplayedByTile.end() &&
              displayedSurface->second == mesh->key) ||
             visiblePreviewChildDependsOnParent ||
             (lodTransition != _farTerrainTransitions.end() &&
              (lodTransition->second.from == mesh->key || lodTransition->second.to == mesh->key)));
        auto previewCanopy = _farCanopyAttachments.end();
        std::shared_ptr<const FarCanopyAttachment> finalPromotionCanopy;
        size_t finalPromotionCanopyBytes = 0;
        FarTerrainWaterPromotionAction waterPromotionAction =
            FarTerrainWaterPromotionAction::MATCHED_TOPOLOGY_TRANSITION;
        if (replacingPreviewParent) {
            if (replacementIsVisible &&
                (_farTerrainTransitions.contains(coordinate) ||
                 _farTerrainAuthorityTransitions.contains(mesh->key) ||
                 _farTerrainTransitions.size() + _farTerrainAuthorityTransitions.size() >=
                     FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS)) {
                return false;
            }
            waterPromotionAction =
                farTerrainWaterPromotionAction(resident->second.waterTopology, mesh->waterTopology);
            previewCanopy = _farCanopyAttachments.find(mesh->key);
            if (waterPromotionAction ==
                    FarTerrainWaterPromotionAction::MATCHED_TOPOLOGY_TRANSITION &&
                previewCanopy != _farCanopyAttachments.end() &&
                farCanopyMatchesSurface(previewCanopy->second.authorityQuality,
                                        previewCanopy->second.groundingQuality,
                                        resident->second.authorityQuality)) {
                // If flora is already visible on the preview surface, promote
                // its final-grounded counterpart in the same GPU transaction
                // when it is ready. A bounded optional-lane miss is retried,
                // but it never delays FINAL terrain or canonical water.
                finalPromotionCanopy = _farTerrainScheduler->findCachedCanopy(mesh->key);
                if (!finalPromotionCanopy ||
                    !farCanopyMatchesSurface(finalPromotionCanopy->authorityQuality,
                                             finalPromotionCanopy->groundingQuality,
                                             mesh->authorityQuality)) {
                    // The completion-time enqueue is bounded and may have
                    // lost a full-queue race. Retry every visible promotion
                    // with deterministic highest priority until accepted.
                    _farTerrainScheduler->enqueueCanopy(mesh->key, 0,
                                                        FarTerrainAuthorityQuality::FINAL);
                    finalPromotionCanopy.reset();
                }
                if (finalPromotionCanopy &&
                    !farCanopyAnchorIdentityCompatible(previewCanopy->second.authorityQuality,
                                                       previewCanopy->second.anchorIdentityHash,
                                                       finalPromotionCanopy->authorityQuality,
                                                       finalPromotionCanopy->anchorIdentityHash)) {
                    const std::string message =
                        "Far-canopy FINAL promotion changed stable ecology anchors at tile " +
                        std::to_string(mesh->key.tileX) + "," + std::to_string(mesh->key.tileZ) +
                        " step " + std::to_string(farTerrainStepSize(mesh->key.step));
                    if (const auto context = world.generationContext()) {
                        context->latchFailure({
                            .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                            .message = message,
                            .retriable = false,
                        });
                    }
                    if (!uploadFailureLogged) {
                        RY_LOG_ERROR(message.c_str());
                        uploadFailureLogged = true;
                    }
                    return false;
                }
                if (finalPromotionCanopy &&
                    canopyUploads < FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET) {
                    finalPromotionCanopyBytes =
                        finalPromotionCanopy->vertices.size() * sizeof(Vertex) +
                        finalPromotionCanopy->indices.size() * sizeof(uint32_t);
                } else {
                    finalPromotionCanopy.reset();
                }
            }
        }
        if (!base) {
            if (!parentAuthorityAllowsDisplayedStep(coordinate, mesh->authorityQuality,
                                                    mesh->key.step)) {
                return false;
            }
        }
        size_t& laneUploads = base ? baseUploads : refinementUploads;
        const size_t laneLimit = exactStreamingBusy
                                     ? (base ? FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME : size_t{4})
                                     : (base ? FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME
                                             : FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME);
        if (laneUploads >= laneLimit)
            return false;
        const uint64_t vertexBytes = mesh->vertices.size() * sizeof(Vertex);
        const uint64_t indexBytes = mesh->indices.size() * sizeof(uint32_t);
        const size_t bytes = vertexBytes + indexBytes;
        const bool mayConsumeNearUploadReserve = !drawGeometry || localUpload;
        if (finalPromotionCanopy &&
            !farTerrainUploadFitsPrioritizedFrameBudget(
                uploadBytes, bytes + finalPromotionCanopyBytes,
                FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME,
                FAR_TERRAIN_NEAR_REFINEMENT_UPLOAD_RESERVE_BYTES, mayConsumeNearUploadReserve)) {
            // Surface authority has priority over an opportunistic atomic
            // canopy upload. The cached attachment remains available to the
            // ordinary optional upload lane after the terrain commit.
            finalPromotionCanopy.reset();
            finalPromotionCanopyBytes = 0;
        }
        if (!farTerrainUploadFitsPrioritizedFrameBudget(
                uploadBytes, bytes + finalPromotionCanopyBytes,
                FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME,
                FAR_TERRAIN_NEAR_REFINEMENT_UPLOAD_RESERVE_BYTES, mayConsumeNearUploadReserve))
            return false;
        const FarTerrainGpuArenaClass arenaClass =
            criticalRefinement    ? FarTerrainGpuArenaClass::CRITICAL_REFINEMENT
            : localCriticalCoverage ? FarTerrainGpuArenaClass::CRITICAL_COVERAGE
            : coverageCritical    ? FarTerrainGpuArenaClass::COVERAGE
            : nearRefinement ? FarTerrainGpuArenaClass::NEAR_REFINEMENT
                             : FarTerrainGpuArenaClass::REFINEMENT;
        if (!farTerrainGpuUploadFitsArena(
                farArenaVertexUsed, farArenaIndexUsed, _farMegaBuffer->vertexCapacity(),
                _farMegaBuffer->indexCapacity(), vertexBytes, indexBytes, arenaClass)) {
            ++_farTerrainArenaAdmissionDeniedCount;
            if (criticalRefinement || nearRefinement || localCriticalCoverage) {
                const bool fullArenaAdmission = criticalRefinement || localCriticalCoverage;
                const uint64_t vertexReserve =
                    fullArenaAdmission ? uint64_t{0}
                                       : FAR_TERRAIN_GPU_VERTEX_COVERAGE_RESERVE_BYTES;
                const uint64_t indexReserve =
                    fullArenaAdmission ? uint64_t{0}
                                       : FAR_TERRAIN_GPU_INDEX_COVERAGE_RESERVE_BYTES;
                const uint64_t vertexLimit =
                    _farMegaBuffer->vertexCapacity() > vertexReserve
                        ? _farMegaBuffer->vertexCapacity() - vertexReserve
                        : uint64_t{0};
                const uint64_t indexLimit =
                    _farMegaBuffer->indexCapacity() > indexReserve
                        ? _farMegaBuffer->indexCapacity() - indexReserve
                        : uint64_t{0};
                const uint64_t alignedVertexBytes = alignedArenaBytes(vertexBytes);
                const uint64_t alignedIndexBytes = alignedArenaBytes(indexBytes);
                const uint64_t availableVertexBytes =
                    farArenaVertexUsed < vertexLimit ? vertexLimit - farArenaVertexUsed : 0;
                const uint64_t availableIndexBytes =
                    farArenaIndexUsed < indexLimit ? indexLimit - farArenaIndexUsed : 0;
                const uint64_t requiredVertexBytes = alignedVertexBytes > availableVertexBytes
                                                         ? alignedVertexBytes - availableVertexBytes
                                                         : uint64_t{0};
                const uint64_t requiredIndexBytes = alignedIndexBytes > availableIndexBytes
                                                        ? alignedIndexBytes - availableIndexBytes
                                                        : uint64_t{0};
                reclaimOptionalGpuResidencyForNear(mesh->key, requiredVertexBytes,
                                                   requiredIndexBytes);
            }
            return false;
        }
        uint64_t finalCanopyVertexBytes = 0;
        uint64_t finalCanopyIndexBytes = 0;
        if (finalPromotionCanopy) {
            finalCanopyVertexBytes = finalPromotionCanopy->vertices.size() * sizeof(Vertex);
            finalCanopyIndexBytes = finalPromotionCanopy->indices.size() * sizeof(uint32_t);
            if (!farTerrainGpuUploadFitsArena(farArenaVertexUsed + alignedArenaBytes(vertexBytes),
                                              farArenaIndexUsed + alignedArenaBytes(indexBytes),
                                              _farMegaBuffer->vertexCapacity(),
                                              _farMegaBuffer->indexCapacity(),
                                              finalCanopyVertexBytes, finalCanopyIndexBytes,
                                              FarTerrainGpuArenaClass::FLORA)) {
                finalPromotionCanopy.reset();
                finalPromotionCanopyBytes = 0;
                finalCanopyVertexBytes = 0;
                finalCanopyIndexBytes = 0;
                ++_farTerrainArenaAdmissionDeniedCount;
            }
        }
        std::optional<MegaBuffer::ChunkAllocation> allocation;
        std::optional<MegaBuffer::ChunkAllocation> finalCanopyAllocation;
        bool stagedAuthorityTransition = false;
        try {
            allocation = _farMegaBuffer->allocate(static_cast<uint32_t>(mesh->vertices.size()),
                                                  static_cast<uint32_t>(mesh->indices.size()));
            _farMegaBuffer->uploadVertices(mesh->vertices.data(),
                                           mesh->vertices.size() * sizeof(Vertex), *allocation);
            _farMegaBuffer->uploadIndices(mesh->indices.data(),
                                          mesh->indices.size() * sizeof(uint32_t), *allocation);
            if (finalPromotionCanopy && !finalPromotionCanopy->vertices.empty() &&
                !finalPromotionCanopy->indices.empty()) {
                try {
                    finalCanopyAllocation = _farMegaBuffer->allocate(
                        static_cast<uint32_t>(finalPromotionCanopy->vertices.size()),
                        static_cast<uint32_t>(finalPromotionCanopy->indices.size()));
                    _farMegaBuffer->uploadVertices(finalPromotionCanopy->vertices.data(),
                                                   finalPromotionCanopy->vertices.size() *
                                                       sizeof(Vertex),
                                                   *finalCanopyAllocation);
                    _farMegaBuffer->uploadIndices(finalPromotionCanopy->indices.data(),
                                                  finalPromotionCanopy->indices.size() *
                                                      sizeof(uint32_t),
                                                  *finalCanopyAllocation);
                } catch (...) {
                    // Optional arena pressure cannot reject a valid terrain
                    // promotion. Release a partial canopy allocation and let
                    // the ordinary attachment lane retry independently.
                    if (finalCanopyAllocation)
                        _farMegaBuffer->free(*finalCanopyAllocation);
                    finalCanopyAllocation.reset();
                    finalPromotionCanopy.reset();
                    finalPromotionCanopyBytes = 0;
                }
            }
            if (!farTerrainUploadCommitAllowed(uploadAction, allocation.has_value()))
                return false;
            FarTerrainMeshState state{*allocation,
                                      mesh->bounds,
                                      mesh->surfaceBounds,
                                      mesh->occluderPatches,
                                      mesh->opaqueIndexCount,
                                      mesh->complexity,
                                      mesh->deterministicHash,
                                      mesh->waterTopology,
                                      mesh->authorityQuality,
                                      true,
                                      mesh->surfaceBoundary,
                                      mesh->exactAuthorityCompatible};
            if (replacingPreviewParent && replacementIsVisible) {
                // Allocate the map node before moving any resident ownership.
                // After this point the commit consists only of no-allocation
                // state moves, so an unordered_map failure cannot leave a new
                // allocation published while the catch path frees it.
                const auto [transition, inserted] =
                    _farTerrainAuthorityTransitions.try_emplace(mesh->key);
                if (!inserted) {
                    throw std::logic_error("far terrain authority transition already exists");
                }
                stagedAuthorityTransition = true;
                FarTerrainMeshState previewSource = resident->second;
                std::optional<FarCanopyMeshState> sourceCanopy;
                if (previewCanopy != _farCanopyAttachments.end() &&
                    farCanopyMatchesSurface(previewCanopy->second.authorityQuality,
                                            previewCanopy->second.groundingQuality,
                                            previewSource.authorityQuality)) {
                    sourceCanopy.emplace(previewCanopy->second);
                }
                const bool hadSourceCanopy = sourceCanopy.has_value();
                const auto requestedCenter = _farTerrainProtectedNearHandoff.requestedCenter();
                const FarTerrainProtectedNearRole requestedRole =
                    requestedCenter ? farTerrainProtectedNearRole(*requestedCenter, coordinate)
                                    : FarTerrainProtectedNearRole::NONE;
                const bool pendingProtectedPublication =
                    requestedCenter && requestedRole != FarTerrainProtectedNearRole::NONE;
                const double promotionStartedAt =
                    pendingProtectedPublication ? 0.0 : CACurrentMediaTime();
                FarTerrainAuthorityTransition promoted{mesh->key, std::move(previewSource),
                                                       std::move(sourceCanopy), promotionStartedAt,
                                                       !pendingProtectedPublication};
                std::optional<FarCanopyMeshState> promotedCanopy;
                if (finalPromotionCanopy) {
                    promotedCanopy.emplace(
                        FarCanopyMeshState{finalCanopyAllocation, finalPromotionCanopy->bounds,
                                           finalPromotionCanopy->deterministicHash,
                                           finalPromotionCanopy->anchorIdentityHash,
                                           finalPromotionCanopy->authorityQuality,
                                           finalPromotionCanopy->groundingQuality});
                }
                resident->second = std::move(state);
                transition->second = std::move(promoted);
                if (promotedCanopy) {
                    previewCanopy->second = std::move(*promotedCanopy);
                    ++uploads;
                    ++canopyUploads;
                    uploadBytes += finalPromotionCanopyBytes;
                } else if (hadSourceCanopy) {
                    _farCanopyAttachments.erase(previewCanopy);
                }
                stagedAuthorityTransition = false;
            } else if (replacingPreviewParent) {
                // A preparation pass or a hidden cached tier has never been
                // submitted to a gameplay encoder. Replace it atomically
                // instead of manufacturing a visible preview source whose
                // later midpoint could cut FINAL exact columns into PREVIEW
                // terrain on the first scene frame.
                FarTerrainMeshState previewSource = resident->second;
                resident->second = std::move(state);
                if (previewSource.uploaded) {
                    _farMegaBuffer->deferFree(previewSource.alloc, _frameRing.frameIndex());
                }
                if (previewCanopy != _farCanopyAttachments.end()) {
                    if (previewCanopy->second.alloc) {
                        _farMegaBuffer->deferFree(*previewCanopy->second.alloc,
                                                  _frameRing.frameIndex());
                    }
                    if (finalPromotionCanopy) {
                        previewCanopy->second =
                            FarCanopyMeshState{finalCanopyAllocation,
                                               finalPromotionCanopy->bounds,
                                               finalPromotionCanopy->deterministicHash,
                                               finalPromotionCanopy->anchorIdentityHash,
                                               finalPromotionCanopy->authorityQuality,
                                               finalPromotionCanopy->groundingQuality};
                        ++uploads;
                        ++canopyUploads;
                        uploadBytes += finalPromotionCanopyBytes;
                    } else {
                        _farCanopyAttachments.erase(previewCanopy);
                    }
                }
            } else {
                const auto [_, inserted] = _farTerrainMeshes.emplace(mesh->key, std::move(state));
                if (!inserted) {
                    _farMegaBuffer->free(*allocation);
                    return false;
                }
                ++_farTerrainResidentWantedCount;
                if (!base)
                    ++_farTerrainResidentRefinementCount;
            }
            ++uploads;
            ++laneUploads;
            uploadBytes += bytes;
            farArenaVertexUsed += alignedArenaBytes(vertexBytes);
            farArenaIndexUsed += alignedArenaBytes(indexBytes);
            if (finalCanopyAllocation) {
                farArenaVertexUsed += alignedArenaBytes(finalCanopyVertexBytes);
                farArenaIndexUsed += alignedArenaBytes(finalCanopyIndexBytes);
            }
            if (base)
                _farTerrainDesiredMetricsDirty = true;
            return true;
        } catch (const std::exception& error) {
            if (stagedAuthorityTransition)
                _farTerrainAuthorityTransitions.erase(mesh->key);
            if (finalCanopyAllocation)
                _farMegaBuffer->free(*finalCanopyAllocation);
            if (allocation)
                _farMegaBuffer->free(*allocation);
            if ((criticalRefinement || nearRefinement || localCriticalCoverage) && !allocation) {
                // Aggregate capacity can still hide per-slab fragmentation.
                // Retire a candidate-sized optional set and retry only after
                // the frame ring reports those source allocations complete.
                reclaimOptionalGpuResidencyForNear(mesh->key, alignedArenaBytes(vertexBytes),
                                                   alignedArenaBytes(indexBytes));
            }
            if (!uploadFailureLogged) {
                RY_LOG_ERROR((std::string("Far-terrain upload failed: ") + error.what()).c_str());
                uploadFailureLogged = true;
            }
            return false;
        }
    };

    auto uploadCanopy = [&](const std::shared_ptr<const FarCanopyAttachment>& attachment) {
        if (!attachment || _farTerrainWanted.count(attachment->key) == 0 ||
            canopyUploads >= FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET) {
            return false;
        }
        const auto base = _farTerrainMeshes.find(attachment->key);
        if (base == _farTerrainMeshes.end() || !base->second.uploaded)
            return false;
        auto promotion = _farTerrainAuthorityTransitions.find(attachment->key);
        const bool targetsSource =
            promotion != _farTerrainAuthorityTransitions.end() && promotion->second.sourceCanopy &&
            farCanopyMatchesSurface(attachment->authorityQuality, attachment->groundingQuality,
                                    promotion->second.source.authorityQuality);
        const bool targetsResident =
            farCanopyMatchesSurface(attachment->authorityQuality, attachment->groundingQuality,
                                    base->second.authorityQuality);
        if (!targetsSource && !targetsResident)
            return false;

        auto resident = _farCanopyAttachments.find(attachment->key);
        FarCanopyMeshState* destination = nullptr;
        if (targetsSource) {
            destination =
                promotion->second.sourceCanopy ? &*promotion->second.sourceCanopy : nullptr;
        } else if (resident != _farCanopyAttachments.end()) {
            destination = &resident->second;
        }
        const FarCanopyMeshState* counterpart = nullptr;
        if (targetsSource) {
            if (resident != _farCanopyAttachments.end())
                counterpart = &resident->second;
        } else if (promotion != _farTerrainAuthorityTransitions.end() &&
                   promotion->second.sourceCanopy) {
            counterpart = &*promotion->second.sourceCanopy;
        }
        const auto compatibleAnchors = [&](const FarCanopyMeshState* candidate) {
            return !candidate || farCanopyAnchorIdentityCompatible(
                                     candidate->authorityQuality, candidate->anchorIdentityHash,
                                     attachment->authorityQuality, attachment->anchorIdentityHash);
        };
        if (!compatibleAnchors(destination) || !compatibleAnchors(counterpart)) {
            const std::string message =
                "Far-canopy FINAL promotion changed stable ecology anchors at tile " +
                std::to_string(attachment->key.tileX) + "," +
                std::to_string(attachment->key.tileZ) + " step " +
                std::to_string(farTerrainStepSize(attachment->key.step));
            if (const auto context = world.generationContext()) {
                context->latchFailure({
                    .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                    .message = message,
                    .retriable = false,
                });
            }
            if (!uploadFailureLogged) {
                RY_LOG_ERROR(message.c_str());
                uploadFailureLogged = true;
            }
            return false;
        }
        if (destination &&
            (!farCanopyMayReplace(destination->authorityQuality, destination->groundingQuality,
                                  attachment->authorityQuality, attachment->groundingQuality) ||
             destination->deterministicHash == attachment->deterministicHash)) {
            return false;
        }
        const size_t bytes = attachment->vertices.size() * sizeof(Vertex) +
                             attachment->indices.size() * sizeof(uint32_t);
        if (uploadBytes + bytes > FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            return false;
        const uint64_t vertexBytes = attachment->vertices.size() * sizeof(Vertex);
        const uint64_t indexBytes = attachment->indices.size() * sizeof(uint32_t);
        if (!farTerrainGpuUploadFitsArena(farArenaVertexUsed, farArenaIndexUsed,
                                          _farMegaBuffer->vertexCapacity(),
                                          _farMegaBuffer->indexCapacity(), vertexBytes, indexBytes,
                                          FarTerrainGpuArenaClass::FLORA)) {
            ++_farTerrainArenaAdmissionDeniedCount;
            return false;
        }

        std::optional<MegaBuffer::ChunkAllocation> allocation;
        try {
            if (!attachment->vertices.empty() && !attachment->indices.empty()) {
                allocation =
                    _farMegaBuffer->allocate(static_cast<uint32_t>(attachment->vertices.size()),
                                             static_cast<uint32_t>(attachment->indices.size()));
                _farMegaBuffer->uploadVertices(attachment->vertices.data(),
                                               attachment->vertices.size() * sizeof(Vertex),
                                               *allocation);
                _farMegaBuffer->uploadIndices(attachment->indices.data(),
                                              attachment->indices.size() * sizeof(uint32_t),
                                              *allocation);
            }
            FarCanopyMeshState state{allocation,
                                     attachment->bounds,
                                     attachment->deterministicHash,
                                     attachment->anchorIdentityHash,
                                     attachment->authorityQuality,
                                     attachment->groundingQuality};
            if (targetsSource) {
                if (promotion->second.sourceCanopy && promotion->second.sourceCanopy->alloc) {
                    _farMegaBuffer->deferFree(*promotion->second.sourceCanopy->alloc,
                                              _frameRing.frameIndex());
                }
                promotion->second.sourceCanopy = std::move(state);
            } else if (resident != _farCanopyAttachments.end()) {
                if (resident->second.alloc)
                    _farMegaBuffer->deferFree(*resident->second.alloc, _frameRing.frameIndex());
                resident->second = std::move(state);
            } else {
                _farCanopyAttachments.emplace(attachment->key, std::move(state));
            }
            const ColumnPos coordinate{attachment->key.tileX, attachment->key.tileZ};
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            if (!_farTerrainTransitions.contains(coordinate) &&
                displayed != _farTerrainDisplayedByTile.end() &&
                displayed->second == attachment->key) {
                // The target is now both drawable and resident. Retire an
                // older-tier fallback only after this upload has committed,
                // making the exchange atomic even for an intentionally empty
                // target attachment.
                retireCanopyFallback(coordinate, attachment->key);
            }
            ++uploads;
            ++canopyUploads;
            uploadBytes += bytes;
            if (allocation) {
                farArenaVertexUsed += alignedArenaBytes(vertexBytes);
                farArenaIndexUsed += alignedArenaBytes(indexBytes);
            }
            return true;
        } catch (const std::exception& error) {
            if (allocation)
                _farMegaBuffer->free(*allocation);
            if (!uploadFailureLogged) {
                RY_LOG_ERROR((std::string("Far-canopy upload failed: ") + error.what()).c_str());
                uploadFailureLogged = true;
            }
            return false;
        }
    };

    _farTerrainResults.clear();
    _farTerrainScheduler->drainCompleted(_farTerrainResults);
    _farCanopyResults.clear();
    _farTerrainScheduler->drainCanopyCompleted(_farCanopyResults);
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
                  return first.key.step != second.key.step &&
                         farTerrainStepSize(first.key.step) < farTerrainStepSize(second.key.step);
              });
    std::sort(_farCanopyResults.begin(), _farCanopyResults.end(),
              [&](const FarCanopyResult& first, const FarCanopyResult& second) {
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

    const auto finalBaseUpgradeNeeded = [&](ColumnPos coordinate) {
        if (!finalStreamingWorkEnabled)
            return false;
        if (farTerrainRequiresCoverageParent(cameraPosition.x, cameraPosition.z, coordinate,
                                             nominalExactBlocks, exactHandoff)) {
            return true;
        }
        if (std::ranges::any_of(_farTerrainPerceptualFinalRequests, [&](const auto& request) {
                return request.coordinate == coordinate;
            })) {
            return true;
        }
        const std::optional<ColumnPos> protectedCenter =
            _farTerrainProtectedNearHandoff.statusCenter();
        if (!protectedCenter)
            return false;
        const FarTerrainProtectedNearRole role =
            farTerrainProtectedNearRole(*protectedCenter, coordinate);
        return role != FarTerrainProtectedNearRole::NONE;
    };
    constexpr double LOCAL_BASE_UPLOAD_RADIUS_BLOCKS =
        FAR_TERRAIN_CONNECTED_REFINEMENT_START_CHUNK_RADIUS * CHUNK_EDGE;
    constexpr double LOCAL_BASE_UPLOAD_RADIUS_SQUARED =
        LOCAL_BASE_UPLOAD_RADIUS_BLOCKS * LOCAL_BASE_UPLOAD_RADIUS_BLOCKS;
    const auto basePrecedesCanopy = [&](FarTerrainKey key) {
        return !drawGeometry || distanceSquaredForKey(key) <= LOCAL_BASE_UPLOAD_RADIUS_SQUARED ||
               finalBaseUpgradeNeeded({key.tileX, key.tileZ});
    };
    // Selection changes discover their desired-LOD debt later in this pass.
    // Treat the movement frame as local debt too, so a cached outer parent
    // cannot flash into view one frame before the nearer queue takes control.
    const bool ordinaryCoveragePublicationEnabled =
        !selectionChanged && farTerrainOrdinaryCoverageWorkEnabled(
                                 drawGeometry, exactStreamingBusy, _farTerrainLocalTerrainDebt);
    const auto baseResultEligible = [&](const FarTerrainResult& result) {
        return !result.failed && farTerrainIsBaseStep(result.key.step) && result.mesh &&
               (result.mesh->authorityQuality != FarTerrainAuthorityQuality::FINAL ||
                finalBaseUpgradeNeeded({result.key.tileX, result.key.tileZ}));
    };
    // CPU completion order cannot bypass a camera-critical parent. During
    // gameplay, ordinary horizon parents wait until protected FINAL children
    // have consumed their reserved upload opportunity. Preparation still
    // spends the complete lane on connected parent coverage.
    for (const FarTerrainResult& result : _farTerrainResults) {
        if (baseResultEligible(result) &&
            (!drawGeometry || finalBaseUpgradeNeeded({result.key.tileX, result.key.tileZ}))) {
            uploadMesh(result.mesh);
        }
    }
    // A completed final parent upgrades the same GPU key in place. Check
    // these before missing preview parents so an upload-budget delay cannot
    // leave a nearby final result hidden behind an older cached preview.
    _farTerrainCachedBaseRequests.clear();
    _farTerrainMissingBaseRequests.clear();
    _farTerrainDistantBaseRequests.clear();
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const FarTerrainKey base{tile.key.tileX, tile.key.tileZ, FAR_TERRAIN_BASE_STEP};
        const auto resident = _farTerrainMeshes.find(base);
        if (resident == _farTerrainMeshes.end() || !resident->second.uploaded) {
            (basePrecedesCanopy(base) ? _farTerrainMissingBaseRequests
                                      : _farTerrainDistantBaseRequests)
                .push_back(base);
        } else if (resident->second.authorityQuality != FarTerrainAuthorityQuality::FINAL &&
                   finalBaseUpgradeNeeded({base.tileX, base.tileZ})) {
            _farTerrainCachedBaseRequests.push_back(base);
        }
    }
    _farTerrainScheduler->findCachedBatch(
        _farTerrainCachedBaseRequests, FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - baseUploads,
        _farTerrainCachedMeshes, FarTerrainAuthorityQuality::FINAL);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }
    // Pull protected FINAL payloads before ordinary missing parents. Their
    // FINAL parents were handled above, and the near upload reserve exists so
    // a ready camera surface cannot remain CPU-only behind broad coverage.
    if (_farTerrainProtectedNearHandoff.statusCenter()) {
        for (const FarTerrainKey target : _farTerrainConnectedNearPatchTargets) {
            if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
                break;
            std::shared_ptr<const FarTerrainMesh> cached = _farTerrainScheduler->findCached(target);
            if (cached && cached->authorityQuality != FarTerrainAuthorityQuality::FINAL)
                cached.reset();
            if (cached)
                uploadMesh(cached);
        }
    }
    _farTerrainScheduler->findCachedBatch(_farTerrainMissingBaseRequests,
                                          FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - baseUploads,
                                          _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }
    _farTerrainFinalBaseRequests.clear();
    const uint32_t missingFinalHandoffParents =
        finalStreamingWorkEnabled
            ? buildFarTerrainFinalParentUpgradeOrder(
                  _farTerrainCandidates, cameraPosition.x, cameraPosition.z, nominalExactBlocks,
                  exactHandoff, isFinalResident, _farTerrainFinalBaseRequests)
            : 0;
    const double lodTimeSeconds = CACurrentMediaTime();
    const auto findNearGrace = [&](ColumnPos coordinate) {
        return _farTerrainNearGraceStartedAt.find(coordinate);
    };
    const auto startNearGrace = [&](ColumnPos coordinate) {
        const auto found = findNearGrace(coordinate);
        if (found != _farTerrainNearGraceStartedAt.end())
            return found->second;
        _farTerrainNearGraceStartedAt.emplace(coordinate, lodTimeSeconds);
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
        if (!isResident({coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP}))
            return mask;
        for (FarTerrainStep step :
             {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            const auto resident = _farTerrainMeshes.find({coordinate.x, coordinate.z, step});
            if (resident == _farTerrainMeshes.end() || !resident->second.uploaded ||
                !parentAuthorityAllowsDisplayedStep(coordinate, resident->second.authorityQuality,
                                                    step)) {
                continue;
            }
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
        const auto desired = _farTerrainDesiredByTile.find(coordinate);
        const FarTerrainStep desiredStep = desired == _farTerrainDesiredByTile.end()
                                               ? FarTerrainStep::EIGHT
                                               : desired->second.step;
        return farTerrainCoarsestDrawableFallback(desiredStep, requiresFineFallback(coordinate),
                                                  requiresBlockScaleFallback(coordinate));
    };
    constexpr std::array<ColumnPos, 4> FAR_TERRAIN_NEIGHBOR_OFFSETS = {
        ColumnPos{1, 0}, ColumnPos{-1, 0}, ColumnPos{0, 1}, ColumnPos{0, -1}};
    const auto stepCompatibleAt = [&](ColumnPos coordinate, FarTerrainStep candidate) {
        std::array<std::optional<FarTerrainStep>, 4> neighborSteps;
        for (size_t edge = 0; edge < FAR_TERRAIN_NEIGHBOR_OFFSETS.size(); ++edge) {
            const ColumnPos neighbor{
                coordinate.x + FAR_TERRAIN_NEIGHBOR_OFFSETS[edge].x,
                coordinate.z + FAR_TERRAIN_NEIGHBOR_OFFSETS[edge].z,
            };
            const auto transition = _farTerrainTransitions.find(neighbor);
            if (transition != _farTerrainTransitions.end()) {
                if (!farTerrainStepCompatibleWithNeighbors(
                        candidate, {transition->second.from.step, transition->second.to.step,
                                    std::nullopt, std::nullopt})) {
                    return false;
                }
                continue;
            }
            const auto displayed = _farTerrainDisplayedByTile.find(neighbor);
            if (displayed != _farTerrainDisplayedByTile.end() && isResident(displayed->second)) {
                neighborSteps[edge] = displayed->second.step;
            }
        }
        return farTerrainStepCompatibleWithNeighbors(candidate, neighborSteps);
    };
    const auto compatibleInitialStepFor =
        [&](ColumnPos coordinate, FarTerrainStep coarsestAllowed) -> std::optional<FarTerrainStep> {
        if (!isResident({coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP}))
            return std::nullopt;
        const FarTerrainStepMask readySteps = residentStepMaskFor(coordinate);
        constexpr std::array CANDIDATES = {
            FarTerrainStep::ONE,   FarTerrainStep::TWO,     FarTerrainStep::FOUR,
            FarTerrainStep::EIGHT, FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO,
        };
        for (const FarTerrainStep candidate : CANDIDATES) {
            const auto& requested = _farTerrainProtectedNearHandoff.requestedCenter();
            const bool protectedTarget =
                requested && farTerrainProtectedNearRole(*requested, coordinate) !=
                                 FarTerrainProtectedNearRole::NONE;
            if (candidate != FAR_TERRAIN_BASE_STEP &&
                !farTerrainProtectedIntermediateMayDisplay(
                    protectedTarget, displayedQualityFor(coordinate, candidate))) {
                // The complete protected FINAL closure remains atomic. A
                // PREVIEW bridge may still improve the visible surface while
                // those canonical children build behind it.
                continue;
            }
            if (candidate == FarTerrainStep::ONE) {
                const auto desired = _farTerrainDesiredByTile.find(coordinate);
                if (desired == _farTerrainDesiredByTile.end() ||
                    desired->second.step != FarTerrainStep::ONE) {
                    continue;
                }
            }
            // A cold step-32 parent is the explicit continuous-coverage
            // fallback. Its exact-owned columns are clipped in the shader, so
            // it must not disappear merely because a neighbor has already
            // reached a finer far tier. Finer replacements still use the
            // normal compatibility gate.
            const bool compatible =
                candidate == FAR_TERRAIN_BASE_STEP || stepCompatibleAt(coordinate, candidate);
            const auto resident = _farTerrainMeshes.find({coordinate.x, coordinate.z, candidate});
            if (!parentAuthorityAllowsDisplayedStep(coordinate,
                                                    resident == _farTerrainMeshes.end()
                                                        ? FarTerrainAuthorityQuality::FINAL
                                                        : resident->second.authorityQuality,
                                                    candidate)) {
                continue;
            }
            if (!isResident({coordinate.x, coordinate.z, candidate}) || !compatible ||
                !farTerrainDisplayedStepAllowed(candidate, coarsestAllowed, readySteps)) {
                continue;
            }
            return candidate;
        }
        return std::nullopt;
    };
    FarTerrainCoverageFrontier parentCoverage =
        farTerrainCoverageFrontier(_farTerrainCandidates, isResident);
    const auto protectedNearGeometryStatus = [&] {
        std::vector<FarTerrainProtectedNearSurface> surfaces;
        surfaces.reserve(_farTerrainConnectedNearPatchTargets.size());
        for (const FarTerrainKey target : _farTerrainConnectedNearPatchTargets) {
            const auto targetSurface = _farTerrainMeshes.find(target);
            const auto parentSurface =
                _farTerrainMeshes.find({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
            if (targetSurface == _farTerrainMeshes.end() || !targetSurface->second.uploaded ||
                parentSurface == _farTerrainMeshes.end() || !parentSurface->second.uploaded) {
                continue;
            }
            surfaces.push_back({
                .key = target,
                .authorityQuality = targetSurface->second.authorityQuality,
                .parentAuthorityQuality = parentSurface->second.authorityQuality,
                .exactAuthorityCompatible = targetSurface->second.exactAuthorityCompatible,
                .surfaceBoundary = targetSurface->second.surfaceBoundary,
            });
        }
        return protectedClosureAnchor
                   ? farTerrainProtectedNearGeometryStatus(
                         *protectedClosureAnchor, _farTerrainConnectedNearPatchTargets, surfaces)
                   : FarTerrainProtectedNearGeometryStatus{};
    };
    const FarTerrainProtectedNearGeometryStatus protectedStatusBeforeUploads =
        protectedNearGeometryStatus();
    const bool protectedNearReadyBeforeUploads = protectedStatusBeforeUploads.ready();
    const bool requiredProtectedNearWork = farTerrainConnectedRefinementLaneOpen(parentCoverage) &&
                                           !_farTerrainConnectedNearPatchTargets.empty() &&
                                           !protectedNearReadyBeforeUploads;
    const bool missingProtectedCoverageParent =
        std::ranges::any_of(_farTerrainConnectedNearPatchTargets, [&](FarTerrainKey target) {
            return !isResident({target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP});
        });
    // Exact ownership clips only ready columns from a parent; the same parent
    // remains the drawable coverage authority for every unresolved column.
    FarTerrainCoverageFrontier coverage = parentCoverage;
    // Optional flora stays behind terrain and water. Once the connected prefix
    // exists, one renewable worker remains available even during exact or
    // protected debt so a continuously busy terrain pipeline cannot leave
    // every drawable non-exact surface barren. The second worker waits for
    // stronger publication debt to clear.
    const bool canopyConnectedPrefixReady = farTerrainConnectedRefinementLaneOpen(parentCoverage);
    const bool nearExactPublicationDebt = farTerrainCanopyHasNearExactPublicationDebt(
        exactStreamingBusy, exactHandoff.distanceBlocks, exactFloraHandoff.distanceBlocks,
        EXACT_STREAMING_FLORA_PRIORITY_RADIUS_CHUNKS);
    bool localTerrainDebt = requiredProtectedNearWork || missingProtectedCoverageParent ||
                            missingFinalHandoffParents != 0;
    _farTerrainScheduler->setWorkerBudget(
        farTerrainWorkerBudget(exactStreamingBusy, localTerrainDebt));
    _farTerrainScheduler->setCanopyWorkerBudget(
        farTerrainCanopyWorkerBudget(drawGeometry, canopyConnectedPrefixReady,
                                     localTerrainDebt || nearExactPublicationDebt,
                                     exactStreamingBusy));
    _farTerrainUrgentRefinementRequests.clear();
    const bool ordinaryNearRefinementWorkEnabled = farTerrainOptionalStreamingWorkEnabled(
        drawGeometry, farTerrainConnectedRefinementLaneOpen(parentCoverage));
    const bool nearRefinementWorkEnabled =
        ordinaryNearRefinementWorkEnabled || requiredProtectedNearWork;
    const std::span<const FarTerrainViewTile> urgentCandidates =
        nearRefinementWorkEnabled ? std::span<const FarTerrainViewTile>{_farTerrainCandidates}
                                  : std::span<const FarTerrainViewTile>{};
    for (const FarTerrainViewTile& tile : urgentCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const std::optional<FarTerrainStep> protectedStep =
            farTerrainProtectedNearRequiredStep(_farTerrainProtectedNearHandoff, coordinate);
        if (!ordinaryNearRefinementWorkEnabled && !protectedStep)
            continue;
        const bool cameraTile = coordinate == centerTile;
        const FarTerrainStep residencyTarget = farTerrainResidencyTarget(tile);
        const bool fineFallbackRequired = requiresFineFallback(coordinate);
        const bool blockScaleFallbackRequired = requiresBlockScaleFallback(coordinate);
        if (!cameraTile && !fineFallbackRequired &&
            !farTerrainCoverageDrawEligible(tile.distanceSquared, parentCoverage)) {
            // A same-distance protected tile may follow a ready exact tile in
            // coordinate order, so keep scanning the bounded candidate set.
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
        if (farTerrainStepSize(displayed) <= farTerrainStepSize(residencyTarget))
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
        const double projectedError =
            tile.screenErrorMetrics
                ? farTerrainProjectedDisplayErrorPixels(displayed,
                                                        displayedQualityFor(coordinate, displayed),
                                                        *tile.screenErrorMetrics)
                : 0.0;
        const FarTerrainStep nextBridge = farTerrainNextDisplayedStep(displayed, residencyTarget);
        const bool displayableWavefront =
            protectedStep.has_value() || stepCompatibleAt(coordinate, nextBridge);
        _farTerrainUrgentRefinementRequests.push_back(
            {coordinate, displayed, residencyTarget, residentStepMaskFor(coordinate),
             transitionActive, deferIntermediate, fineFallbackRequired, blockScaleFallbackRequired,
             cameraTile, tileVisibleForScheduling(coordinate), displayableWavefront, projectedError,
             tile.distanceSquared, protectedStep.has_value()});
    }
    // Every connected desired-LOD miss is local debt. Finish these requests
    // nearest-first before expanding the outer parent frontier, not only the
    // small subset that intersects the exact fallback band.
    localTerrainDebt = localTerrainDebt || !_farTerrainUrgentRefinementRequests.empty();
    _farTerrainLocalTerrainDebt = localTerrainDebt;
    _farTerrainScheduler->setNearFirstWorkEnabled(
        drawGeometry && (exactStreamingBusy || localTerrainDebt));
    _farTerrainScheduler->setWorkerBudget(
        farTerrainWorkerBudget(exactStreamingBusy, localTerrainDebt));
    _farTerrainScheduler->setCanopyWorkerBudget(
        farTerrainCanopyWorkerBudget(drawGeometry, canopyConnectedPrefixReady,
                                     localTerrainDebt || nearExactPublicationDebt,
                                     exactStreamingBusy));

    // The pure ordering helper ranks the camera and protected handoff first,
    // followed by the connected neighbor wavefront and largest projected
    // error. Every admitted request builds an adjacent bridge that can become
    // displayable instead of seeding fine targets behind coarse neighbors.
    const size_t sharedTransitionOccupancy =
        std::min(FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS,
                 _farTerrainTransitions.size() + _farTerrainAuthorityTransitions.size());
    reserveFarTerrainIntermediateTransitionSlots(_farTerrainUrgentRefinementRequests,
                                                 sharedTransitionOccupancy);
    buildFarTerrainProgressiveSubmissionOrder(_farTerrainUrgentRefinementRequests,
                                              _farTerrainUrgentRefinementKeys,
                                              FAR_TERRAIN_MAX_PROGRESSIVE_PLANNER_RESULTS);
    _farTerrainRefinementSubmissionKeys.clear();
    for (const FarTerrainKey key : _farTerrainUrgentRefinementKeys) {
        if (!predictedOnlyTarget(key))
            _farTerrainRefinementSubmissionKeys.push_back(key);
    }
    _farTerrainScheduler->findCachedBatch(_farTerrainRefinementSubmissionKeys,
                                          FAR_TERRAIN_MAX_URGENT_REFINEMENT_UPLOADS_PER_FRAME,
                                          _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    // Build the optional lane from every drawable surface. PREVIEW ecology may
    // publish immediately while the same parked job converges on FINAL. Its
    // geometry grounds against the displayed PREVIEW or FINAL payload. A
    // bounded nearest-first insertion keeps this pass linear in the displayed
    // set, and the scheduler receives absolute block distance rather than a
    // rotating batch-local rank. One ready attachment gets an upload
    // opportunity after protected and urgent local terrain but before
    // movement-edge bases and broad refinements.
    _farTerrainCanopyRefreshRequests.clear();
    _farTerrainCanopyRefreshKeys.clear();
    _farTerrainCachedCanopies.clear();
    if (drawGeometry) {
        buildFarTerrainCanopyRefreshBatch(
            _farTerrainDisplayedByTile, _farTerrainTransitions, _farTerrainMeshes,
            _farCanopyAttachments, cameraPosition.x, cameraPosition.z,
            FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET, _farTerrainCanopyRefreshRequests,
            &exactFloraHandoff);
        for (const FarTerrainCanopyRefreshRequest& request : _farTerrainCanopyRefreshRequests)
            _farTerrainCanopyRefreshKeys.push_back(request.key);
        _farTerrainScheduler->findCachedCanopyBatch(_farTerrainCanopyRefreshKeys,
                                                    _farTerrainCanopyRefreshKeys.size(),
                                                    _farTerrainCachedCanopies);
    }
    const auto readyCanopyFor = [&](FarTerrainKey key, FarTerrainAuthorityQuality groundingQuality)
        -> std::shared_ptr<const FarCanopyAttachment> {
        std::shared_ptr<const FarCanopyAttachment> best;
        const auto prefer = [&](const std::shared_ptr<const FarCanopyAttachment>& candidate) {
            if (!candidate || candidate->key != key ||
                candidate->groundingQuality != groundingQuality)
                return;
            if (!best ||
                farTerrainAuthoritySatisfies(candidate->authorityQuality, best->authorityQuality)) {
                best = candidate;
            }
        };
        for (const FarCanopyResult& result : _farCanopyResults) {
            if (result.key == key && !result.failed && result.attachment)
                prefer(result.attachment);
        }
        for (const std::shared_ptr<const FarCanopyAttachment>& attachment :
             _farTerrainCachedCanopies) {
            prefer(attachment);
        }
        return best;
    };
    std::optional<FarTerrainKey> earlyCanopyUpload;
    for (const FarTerrainCanopyRefreshRequest& request : _farTerrainCanopyRefreshRequests) {
        const std::shared_ptr<const FarCanopyAttachment> attachment =
            readyCanopyFor(request.key, request.groundingQuality);
        if (attachment && uploadCanopy(attachment)) {
            earlyCanopyUpload = request.key;
            break;
        }
    }
    for (const FarTerrainCanopyRefreshRequest& request : _farTerrainCanopyRefreshRequests) {
        _farTerrainScheduler->enqueueCanopy(request.key, request.viewPriority,
                                            request.groundingQuality);
    }
    // Start the grounded successor before a requested surface promotion
    // reaches the upload lane. The queued-job coalescer preserves the current
    // PREVIEW-grounded publication first, then retargets its parked FINAL
    // retry. This prewarm is optional, bounded, and cannot consume terrain or
    // water scheduler capacity.
    size_t finalGroundingPrewarms = 0;
    for (const FarTerrainRefinementCacheRequest& finalRequest :
         _farTerrainPerceptualFinalRequests) {
        if (finalGroundingPrewarms >= FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET)
            break;
        const auto displayed = _farTerrainDisplayedByTile.find(finalRequest.coordinate);
        if (displayed == _farTerrainDisplayedByTile.end())
            continue;
        if (exactFloraHandoff.tileFullyOwned(finalRequest.coordinate))
            continue;
        const auto resident = _farTerrainMeshes.find(displayed->second);
        if (resident == _farTerrainMeshes.end() || !resident->second.uploaded ||
            resident->second.authorityQuality != FarTerrainAuthorityQuality::PREVIEW) {
            continue;
        }
        const uint32_t priority = static_cast<uint32_t>(
            std::min(std::sqrt(finalRequest.distanceSquaredBlocks),
                     static_cast<double>(std::numeric_limits<uint32_t>::max())));
        _farTerrainScheduler->enqueueCanopy(displayed->second, priority,
                                            FarTerrainAuthorityQuality::FINAL);
        ++finalGroundingPrewarms;
    }
    if (const std::optional<ColumnPos> protectedCenter =
            _farTerrainProtectedNearHandoff.statusCenter()) {
        for (const FarTerrainCanopyRefreshRequest& request : _farTerrainCanopyRefreshRequests) {
            if (finalGroundingPrewarms >= FAR_TERRAIN_CANOPY_REFRESH_REQUEST_BUDGET)
                break;
            if (request.groundingQuality != FarTerrainAuthorityQuality::PREVIEW ||
                exactFloraHandoff.tileFullyOwned(
                    {request.key.tileX, request.key.tileZ}) ||
                farTerrainProtectedNearRole(*protectedCenter,
                                            {request.key.tileX, request.key.tileZ}) ==
                    FarTerrainProtectedNearRole::NONE) {
                continue;
            }
            _farTerrainScheduler->enqueueCanopy(request.key, request.viewPriority,
                                                FarTerrainAuthorityQuality::FINAL);
            ++finalGroundingPrewarms;
        }
    }

    // Gameplay already owns a connected nearby surface. Movement-edge parent
    // results remain coverage work, but they cannot consume the sole frame
    // opportunity for a ready nearby attachment.
    for (const FarTerrainResult& result : _farTerrainResults) {
        if (ordinaryCoveragePublicationEnabled && baseResultEligible(result) &&
            !basePrecedesCanopy(result.key)) {
            if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
                break;
            uploadMesh(result.mesh);
        }
    }
    _farTerrainScheduler->findCachedBatch(
        ordinaryCoveragePublicationEnabled
            ? std::span<const FarTerrainKey>{_farTerrainDistantBaseRequests}
            : std::span<const FarTerrainKey>{},
        FAR_TERRAIN_MAX_BASE_UPLOADS_PER_FRAME - baseUploads, _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    // The broad optional lane reuses the same bounded, ranked keys after the
    // urgent upload pass. Rebuilding a second 3,336-request vector provided no
    // additional scheduler capacity and dominated settled planner time.
    const size_t refinementLaneLimit =
        exactStreamingBusy ? size_t{4} : FAR_TERRAIN_MAX_REFINEMENT_UPLOADS_PER_FRAME;
    std::erase_if(_farTerrainNearGraceStartedAt, [&](const auto& entry) {
        return !_farTerrainActiveTiles.contains(entry.first) ||
               !isResident({entry.first.x, entry.first.z, FAR_TERRAIN_BASE_STEP});
    });
    _farTerrainRefinementSubmissionKeys.clear();
    if (drawGeometry && farTerrainRefinementLaneOpen(parentCoverage, true)) {
        for (const FarTerrainKey key : _farTerrainUrgentRefinementKeys) {
            if (!isResident(key) && !predictedOnlyTarget(key))
                _farTerrainRefinementSubmissionKeys.push_back(key);
        }
    }
    _farTerrainScheduler->findCachedBatch(_farTerrainRefinementSubmissionKeys,
                                          refinementLaneLimit - refinementUploads,
                                          _farTerrainCachedMeshes);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    // A final refinement has the same geometric key as its preview payload,
    // so it is not a new transition target. Apply cached upgrades only after
    // all useful preview bridges have had first claim on this frame's upload
    // lane, then swap the complete allocation atomically.
    _farTerrainFinalRefinementRequests.clear();
    if (drawGeometry) {
        for (const FarTerrainRefinementCacheRequest& request : _farTerrainPerceptualFinalRequests) {
            const ColumnPos coordinate = request.coordinate;
            if (_farTerrainTransitions.contains(coordinate))
                continue;
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            if (displayed == _farTerrainDisplayedByTile.end() ||
                farTerrainIsBaseStep(displayed->second.step)) {
                continue;
            }
            const FarTerrainKey key = displayed->second;
            if (predictedOnlyTarget(key))
                continue;
            const auto resident = _farTerrainMeshes.find(key);
            if (resident == _farTerrainMeshes.end() || !resident->second.uploaded ||
                resident->second.authorityQuality != FarTerrainAuthorityQuality::PREVIEW ||
                !isFinalBaseResident(coordinate)) {
                continue;
            }
            _farTerrainFinalRefinementRequests.push_back(key);
        }
    }
    _farTerrainScheduler->findCachedBatch(
        _farTerrainFinalRefinementRequests, refinementLaneLimit - refinementUploads,
        _farTerrainCachedMeshes, FarTerrainAuthorityQuality::FINAL);
    for (const std::shared_ptr<const FarTerrainMesh>& mesh : _farTerrainCachedMeshes) {
        if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
            break;
        uploadMesh(mesh);
    }

    // Additional ready attachments may consume capacity left after terrain and
    // water. The early opportunity above is the only capacity ordering change;
    // every other canopy remains strictly optional.
    if (drawGeometry) {
        for (const FarTerrainCanopyRefreshRequest& request : _farTerrainCanopyRefreshRequests) {
            if (earlyCanopyUpload && request.key == *earlyCanopyUpload)
                continue;
            if (uploadBytes >= FAR_TERRAIN_MAX_UPLOAD_BYTES_PER_FRAME)
                break;
            if (const std::shared_ptr<const FarCanopyAttachment> attachment =
                    readyCanopyFor(request.key, request.groundingQuality)) {
                uploadCanopy(attachment);
            }
        }
    }

    const auto farPlannerPublicationCompletedAt = std::chrono::steady_clock::now();

    // Submission is nearest-first and bounded inside the scheduler. During
    // preparation, connected PREVIEW coverage and the protected FINAL closure
    // share the hard-priority lane. Ordinary refinement remains gameplay-only.
    const std::optional<ColumnPos> currentProtectedCenter =
        _farTerrainProtectedNearHandoff.statusCenter();
    size_t urgentRefinementSubmissions = 0;
    size_t baseSubmissions = 0;
    constexpr size_t MAX_BASE_SUBMISSIONS_PER_FRAME = 64;
    const bool ordinaryCoverageWorkEnabled = farTerrainOrdinaryCoverageWorkEnabled(
        drawGeometry, exactStreamingBusy, localTerrainDebt);
    // A missing parent in the current protected closure is the player's only
    // continuous fallback. Promote or admit it through the critical coverage
    // lane before the ordinary horizon scan, including when that scan already
    // parked its previous request at the nonurgent cap.
    if (currentProtectedCenter) {
        for (size_t index = 0;
             index < _farTerrainConnectedNearPatchTargets.size() &&
             urgentRefinementSubmissions <
                 FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
             _farTerrainScheduler->hasUrgentRefinementCapacity();
             ++index) {
            const FarTerrainKey target = _farTerrainConnectedNearPatchTargets[index];
            const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
            if (isResident(parent))
                continue;
            if (_farTerrainScheduler->enqueueUrgentCoverage(
                    parent, static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX)))) {
                ++urgentRefinementSubmissions;
                ++baseSubmissions;
            }
        }
    }

    size_t refinementOffset = 0;
    if (ordinaryCoverageWorkEnabled) {
        for (; refinementOffset < _farTerrainPriorityOrder.size(); ++refinementOffset) {
            const FarTerrainKey key = _farTerrainPriorityOrder[refinementOffset];
            if (!farTerrainIsBaseStep(key.step))
                break;
            if (isResident(key))
                continue;
            if (currentProtectedCenter &&
                farTerrainProtectedNearRole(*currentProtectedCenter, {key.tileX, key.tileZ}) !=
                    FarTerrainProtectedNearRole::NONE) {
                continue;
            }
            if (_farTerrainScheduler->hasSubmissionCapacity()) {
                if (_farTerrainScheduler->enqueue(key,
                                                  static_cast<uint32_t>(refinementOffset * 8))) {
                    ++baseSubmissions;
                }
            }
            ++refinementOffset;
            break;
        }
    }

    // The complete protected representation is one FINAL publication unit.
    // Every one of its 60 parents remains drawable as PREVIEW until the
    // corresponding FINAL parent and child are complete. Submit drawable
    // FINAL children first because they directly retire near-player terrain
    // debt. A child without a provisional bridge remains deferred until the
    // bounded prerequisite lane below creates one.
    if (finalStreamingWorkEnabled && _farTerrainProtectedNearHandoff.statusCenter()) {
        for (size_t index = 0; index < _farTerrainConnectedNearPatchTargets.size() &&
                               urgentRefinementSubmissions <
                                   FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
                               _farTerrainScheduler->hasUrgentRefinementCapacity();
            ++index) {
            const FarTerrainKey target = _farTerrainConnectedNearPatchTargets[index];
            const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
            const auto displayed = _farTerrainDisplayedByTile.find({target.tileX, target.tileZ});
            const bool provisionalBridgeDisplayed =
                displayed != _farTerrainDisplayedByTile.end() &&
                !farTerrainIsBaseStep(displayed->second.step);
            if (!farTerrainProtectedFinalTargetMaySubmit(
                    isFinalResident(target), provisionalBridgeDisplayed,
                    isFinalResident(parent))) {
                // At least one provisional bridge becomes drawable before
                // this coordinate spends capacity on its hidden FINAL target,
                // unless a FINAL parent has already made that PREVIEW bridge
                // authority-incompatible. In that state, build the FINAL child
                // directly and retain atomic publication of the full closure.
                continue;
            }
            const uint32_t priority =
                static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX));
            if (_farTerrainScheduler->enqueueUrgentFinalRefinement(target, priority, true))
                ++urgentRefinementSubmissions;
        }
    }
    const auto missingCurrentProtectedBridge = [&](FarTerrainKey bridge) {
        return bridge.step != FarTerrainStep::ONE && currentProtectedCenter &&
               farTerrainProtectedNearRole(*currentProtectedCenter,
                                           {bridge.tileX, bridge.tileZ}) !=
                   FarTerrainProtectedNearRole::NONE &&
               !isResident(bridge);
    };
    const bool protectedBridgePrerequisiteRequired =
        std::ranges::any_of(_farTerrainUrgentRefinementKeys, missingCurrentProtectedBridge);
    const size_t protectedFinalSubmissionFloor = farTerrainProtectedFinalSubmissionFloor(
        FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME,
        protectedBridgePrerequisiteRequired);
    size_t protectedFinalParentScan = 0;
    const auto submitCurrentFinalParentsUntil = [&](size_t submissionLimit) {
        for (; finalStreamingWorkEnabled &&
               protectedFinalParentScan < _farTerrainConnectedNearPatchTargets.size() &&
               urgentRefinementSubmissions < submissionLimit &&
               _farTerrainScheduler->hasUrgentRefinementCapacity();
             ++protectedFinalParentScan) {
            const size_t index = protectedFinalParentScan;
            const FarTerrainKey target = _farTerrainConnectedNearPatchTargets[index];
            const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
            if (!isResident(parent) || isFinalResident(parent))
                continue;
            if (_farTerrainScheduler->enqueueFinalBase(
                    parent, static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX)), true)) {
                ++urgentRefinementSubmissions;
            }
        }
    };
    // Reserve at most one third of a cold frame for the PREVIEW bridges that
    // make later FINAL children displayable. Canonical FINAL work therefore
    // always receives the first and largest share of current closure capacity.
    submitCurrentFinalParentsUntil(protectedFinalSubmissionFloor);
    size_t protectedBridgeSubmissions = 0;
    for (size_t index = 0;
         index < _farTerrainUrgentRefinementKeys.size() &&
         protectedBridgeSubmissions < FAR_TERRAIN_MAX_PROTECTED_BRIDGE_SUBMISSIONS_PER_FRAME &&
         urgentRefinementSubmissions < FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
         _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey bridge = _farTerrainUrgentRefinementKeys[index];
        if (!missingCurrentProtectedBridge(bridge))
            continue;
        if (_farTerrainScheduler->enqueueUrgentRefinement(
                bridge, static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX)), true)) {
            ++protectedBridgeSubmissions;
            ++urgentRefinementSubmissions;
        }
    }
    // If prerequisites were already queued or cached, return their unused
    // reservation to current FINAL parents before any directional prediction.
    submitCurrentFinalParentsUntil(FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME);
    // Directional preparation uses only capacity left after the complete
    // current closure has submitted its parents and children. Results remain
    // in the CPU cache. They are intentionally absent from connected targets,
    // upload lists, GPU residency, display state, and closure statistics until
    // the canonical half-tile anchor actually advances.
    size_t predictedProtectedSubmissions = 0;
    const auto predictedSubmissionPriority = [&](size_t index) {
        const size_t currentOffset =
            _farTerrainConnectedNearPatchTargets.size() * static_cast<size_t>(8);
        return static_cast<uint32_t>(
            std::min(currentOffset + index * static_cast<size_t>(8),
                     static_cast<size_t>(UINT32_MAX)));
    };
    for (size_t index = 0;
         index < _farTerrainPredictedNearPatchTargets.size() &&
         predictedProtectedSubmissions <
             FAR_TERRAIN_MAX_PROTECTED_PREDICTION_SUBMISSIONS_PER_FRAME &&
         urgentRefinementSubmissions < FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
         _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey target = _farTerrainPredictedNearPatchTargets[index];
        const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
        if (!isResident(parent) || isFinalResident(parent))
            continue;
        if (_farTerrainScheduler->enqueueFinalBase(parent, predictedSubmissionPriority(index),
                                                   true)) {
            ++predictedProtectedSubmissions;
            ++urgentRefinementSubmissions;
        }
    }
    for (size_t index = 0;
         index < _farTerrainPredictedNearPatchTargets.size() &&
         predictedProtectedSubmissions <
             FAR_TERRAIN_MAX_PROTECTED_PREDICTION_SUBMISSIONS_PER_FRAME &&
         urgentRefinementSubmissions < FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
         _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey target = _farTerrainPredictedNearPatchTargets[index];
        const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
        const std::shared_ptr<const FarTerrainMesh> cachedParent =
            _farTerrainScheduler->findCached(parent);
        if (!isFinalResident(parent) &&
            (!cachedParent ||
             cachedParent->authorityQuality != FarTerrainAuthorityQuality::FINAL)) {
            continue;
        }
        if (_farTerrainScheduler->enqueueUrgentFinalRefinement(
                target, predictedSubmissionPriority(index), true)) {
            ++predictedProtectedSubmissions;
            ++urgentRefinementSubmissions;
        }
    }
    // Admit ready same-key FINAL children before adding another wave of proxy
    // bridges. Four bounded slots let canonical learned detail advance even
    // while four FINAL parents remain parked and the visible horizon still
    // contains more proxy candidates than the queue can hold.
    constexpr size_t MAX_VISIBLE_FINAL_CHILD_SUBMISSIONS_PER_FRAME = 4;
    size_t visibleFinalChildSubmissions = 0;
    for (size_t index = 0;
         drawGeometry && index < _farTerrainFinalRefinementRequests.size() &&
         visibleFinalChildSubmissions < MAX_VISIBLE_FINAL_CHILD_SUBMISSIONS_PER_FRAME &&
         urgentRefinementSubmissions < FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME &&
         _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey target = _farTerrainFinalRefinementRequests[index];
        const ColumnPos coordinate{target.tileX, target.tileZ};
        const bool cameraNearCritical =
            coordinate == centerTile || requiresFineFallback(coordinate);
        const uint32_t priority = static_cast<uint32_t>(std::min<size_t>(index, UINT32_MAX));
        const bool enqueued =
            requiresFineFallback(coordinate)
                ? _farTerrainScheduler->enqueueUrgentFinalRefinement(target, priority, true)
                : _farTerrainScheduler->enqueueFinalRefinement(target, priority,
                                                               cameraNearCritical);
        if (enqueued) {
            ++visibleFinalChildSubmissions;
            ++urgentRefinementSubmissions;
        }
    }
    constexpr size_t MAX_VISIBLE_FINAL_PARENT_SUBMISSIONS_PER_FRAME = 4;
    size_t visibleFinalParentSubmissions = 0;
    const size_t proxyBridgeSubmissionLimit =
        FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME -
        (_farTerrainPerceptualFinalRequests.empty()
             ? size_t{0}
             : MAX_VISIBLE_FINAL_PARENT_SUBMISSIONS_PER_FRAME);
    for (size_t index = 0; index < _farTerrainUrgentRefinementKeys.size() &&
                           urgentRefinementSubmissions < proxyBridgeSubmissionLimit &&
                           _farTerrainScheduler->hasUrgentRefinementCapacity();
         ++index) {
        const FarTerrainKey target = _farTerrainUrgentRefinementKeys[index];
        const ColumnPos coordinate{target.tileX, target.tileZ};
        const std::optional<ColumnPos> protectedCenter =
            _farTerrainProtectedNearHandoff.statusCenter();
        const FarTerrainProtectedNearRole protectedRole =
            protectedCenter ? farTerrainProtectedNearRole(*protectedCenter, coordinate)
                            : FarTerrainProtectedNearRole::NONE;
        const bool protectedNearTarget = protectedRole != FarTerrainProtectedNearRole::NONE;
        const std::optional<FarTerrainStep> protectedStep =
            farTerrainProtectedNearRequiredStep(_farTerrainProtectedNearHandoff, coordinate);
        if (protectedNearTarget && protectedStep && target.step == *protectedStep &&
            target.step == FarTerrainStep::ONE) {
            continue;
        }
        const FarTerrainKey parentKey{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        const auto parent = _farTerrainMeshes.find(parentKey);
        std::optional<FarTerrainAuthorityQuality> parentSourceQuality;
        if (const auto promotion = _farTerrainAuthorityTransitions.find(parentKey);
            promotion != _farTerrainAuthorityTransitions.end()) {
            parentSourceQuality = promotion->second.source.authorityQuality;
        }
        const bool finalRequired = !protectedNearTarget && parent != _farTerrainMeshes.end() &&
                                   farTerrainRefinementRequiresFinalAuthority(
                                       parent->second.authorityQuality, parentSourceQuality,
                                       requiresFineFallback(coordinate));
        if ((finalRequired && isFinalResident(target)) || (!finalRequired && isResident(target))) {
            continue;
        }
        const uint32_t priority = static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX));
        const bool cameraNearCritical =
            protectedNearTarget || coordinate == centerTile || requiresFineFallback(coordinate);
        const bool enqueued = finalRequired ? _farTerrainScheduler->enqueueUrgentFinalRefinement(
                                                  target, priority, cameraNearCritical)
                                            : _farTerrainScheduler->enqueueUrgentRefinement(
                                                  target, priority, cameraNearCritical);
        if (enqueued) {
            ++urgentRefinementSubmissions;
        }
    }
    // Preview bridges receive the first ordinary urgent slots because they
    // remove large projected cells without another decoder call. Reserve the
    // remaining slots for visible FINAL parents, which still outrank canopy
    // and speculative work in the inference coordinator.
    for (const FarTerrainRefinementCacheRequest& request : _farTerrainPerceptualFinalRequests) {
        if (visibleFinalParentSubmissions >= MAX_VISIBLE_FINAL_PARENT_SUBMISSIONS_PER_FRAME ||
            urgentRefinementSubmissions >=
                FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME ||
            !_farTerrainScheduler->hasUrgentRefinementCapacity()) {
            break;
        }
        const FarTerrainKey parent{request.coordinate.x, request.coordinate.z,
                                   FAR_TERRAIN_BASE_STEP};
        if (!isResident(parent) || isFinalResident(parent))
            continue;
        const bool exactHandoffRequired = requiresFineFallback(request.coordinate);
        if (_farTerrainScheduler->enqueueFinalBase(
                parent,
                static_cast<uint32_t>(std::min<size_t>(visibleFinalParentSubmissions, UINT32_MAX)),
                exactHandoffRequired)) {
            ++visibleFinalParentSubmissions;
            ++urgentRefinementSubmissions;
        }
    }
    // FINAL child tiers are the fastest way to remove a visibly coarse nearby
    // tile. Spend remaining urgent slots on their FINAL parent replacements
    // without allowing optional outer work to delay the protected handoff.
    if (finalStreamingWorkEnabled && farTerrainConnectedRefinementLaneOpen(parentCoverage)) {
        const size_t finalParentRequestCount = missingFinalHandoffParents;
        for (size_t index = 0; index < finalParentRequestCount; ++index) {
            if (urgentRefinementSubmissions >=
                    FAR_TERRAIN_MAX_URGENT_REFINEMENT_SUBMISSIONS_PER_FRAME ||
                !_farTerrainScheduler->hasUrgentRefinementCapacity()) {
                break;
            }
            const FarTerrainKey base = _farTerrainFinalBaseRequests[index];
            const ColumnPos coordinate{base.tileX, base.tileZ};
            const bool exactHandoffRequired = index < missingFinalHandoffParents;
            if (!isResident(base) || isFinalResident(base) ||
                (coordinate != centerTile && !exactHandoffRequired &&
                 !farTerrainCoverageDrawEligible(distanceSquaredForKey(base), parentCoverage))) {
                continue;
            }
            if (_farTerrainScheduler->enqueueFinalBase(
                    base, static_cast<uint32_t>(std::min<size_t>(index * 8, UINT32_MAX)),
                    exactHandoffRequired)) {
                ++urgentRefinementSubmissions;
            }
        }
    }

    for (; ordinaryCoverageWorkEnabled &&
           refinementOffset < _farTerrainPriorityOrder.size(); ++refinementOffset) {
        if (!_farTerrainScheduler->hasSubmissionCapacity() ||
            baseSubmissions >= MAX_BASE_SUBMISSIONS_PER_FRAME)
            break;
        const FarTerrainKey key = _farTerrainPriorityOrder[refinementOffset];
        if (!farTerrainIsBaseStep(key.step))
            break;
        if (currentProtectedCenter &&
            farTerrainProtectedNearRole(*currentProtectedCenter, {key.tileX, key.tileZ}) !=
                FarTerrainProtectedNearRole::NONE) {
            continue;
        }
        if (!isResident(key)) {
            if (_farTerrainScheduler->enqueue(key,
                                              static_cast<uint32_t>(refinementOffset * 8))) {
                ++baseSubmissions;
            }
        }
    }
    const bool parentSubmissionComplete =
        ordinaryCoverageWorkEnabled && refinementOffset == _farTerrainCandidates.size();
    const std::span<const FarTerrainKey> broadRefinementSubmissions =
        drawGeometry && farTerrainRefinementLaneOpen(parentCoverage, parentSubmissionComplete)
            ? std::span<const FarTerrainKey>{_farTerrainRefinementSubmissionKeys}
            : std::span<const FarTerrainKey>{};
    for (size_t index = 0; index < broadRefinementSubmissions.size(); ++index) {
        if (!_farTerrainScheduler->hasSubmissionCapacity())
            break;
        const FarTerrainKey key = broadRefinementSubmissions[index];
        if (!isResident(key)) {
            _farTerrainScheduler->enqueue(
                key, static_cast<uint32_t>(std::min<size_t>(index, UINT32_MAX)));
        }
    }

    const bool protectedNearReplacementReady =
        _farTerrainProtectedNearHandoff.requestedCenter().has_value() &&
        protectedStatusBeforeUploads.ready();
    if (protectedStatusBeforeUploads.presentTargets ==
            protectedStatusBeforeUploads.expectedTargets &&
        protectedStatusBeforeUploads.finalTargets == protectedStatusBeforeUploads.expectedTargets &&
        protectedStatusBeforeUploads.finalParents ==
            protectedStatusBeforeUploads.expectedFinalParents &&
        protectedStatusBeforeUploads.mismatchedSharedBoundaries != 0) {
        if (const auto context = world.generationContext()) {
            context->latchFailure({
                .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                .message = "Protected FINAL far-terrain patch has incompatible shared boundaries",
                .retriable = true,
            });
        }
    }
    if (protectedNearReplacementReady) {
        // Publish the requested near patch and shell as one enabled set.
        // Same-key GPU replacements were staged with their
        // old sources as sole draw owners; assign one clock before changing
        // any display key so every temporal exchange starts together.
        for (const FarTerrainKey target : _farTerrainConnectedNearPatchTargets) {
            if (auto transition = _farTerrainAuthorityTransitions.find(target);
                transition != _farTerrainAuthorityTransitions.end() &&
                !transition->second.published) {
                transition->second.startedAtSeconds = lodTimeSeconds;
                transition->second.published = true;
            }
        }
        // No encoder observes the display map between these updates, so a
        // flying camera cannot reveal a partial FINAL island or proxy ring.
        for (const FarTerrainKey target : _farTerrainConnectedNearPatchTargets) {
            const ColumnPos coordinate{target.tileX, target.tileZ};
            if (const auto transition = _farTerrainTransitions.find(coordinate);
                transition != _farTerrainTransitions.end()) {
                completeCanopyLodFallback(coordinate, transition->second.from, target);
            }
            _farTerrainTransitions.erase(coordinate);
            _farTerrainDisplayedByTile.insert_or_assign(coordinate, target);
            eraseNearGrace(coordinate);
        }
        if (_farTerrainProtectedNearHandoff.commitRequested(true)) {
            // Old-only coordinates now return to the ordinary screen-error
            // hierarchy. Recompute them next frame and coarsen through legal
            // adjacent tiers while the newly published patch stays fixed.
            _farTerrainDesiredMetricsDirty = true;
        }
    }

    // Finish each monotonic replacement before reevaluating the desired tier.
    // Redirecting a transition mid-flight can make already revealed voxel
    // columns disappear and is perceived as flicker during ordinary travel.
    for (auto it = _farTerrainTransitions.begin(); it != _farTerrainTransitions.end();) {
        if (!_farTerrainActiveTiles.contains(it->first) ||
            !isResident({it->first.x, it->first.z, FAR_TERRAIN_BASE_STEP})) {
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
            const FarTerrainKey target{it->first.x, it->first.z, advance.displayed};
            completeCanopyLodFallback(it->first, it->second.from, target);
            _farTerrainDisplayedByTile.insert_or_assign(it->first, target);
            it = _farTerrainTransitions.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = _farTerrainDisplayedByTile.begin(); it != _farTerrainDisplayedByTile.end();) {
        if (!_farTerrainActiveTiles.contains(it->first)) {
            it = _farTerrainDisplayedByTile.erase(it);
        } else {
            ++it;
        }
    }

    // A coordinate cannot become visible before its parent is resident. It
    // selects the closest resident tier that remains compatible with every
    // displayed neighbor, including both sides of an active replacement.
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
        if (!isResident(base)) {
            _farTerrainDisplayedByTile.erase(coordinate);
            _farTerrainTransitions.erase(coordinate);
            retireCanopyFallback(coordinate, std::nullopt);
            eraseNearGrace(coordinate);
            continue;
        }
        auto displayed = _farTerrainDisplayedByTile.find(coordinate);
        if (displayed != _farTerrainDisplayedByTile.end() && isResident(displayed->second)) {
            continue;
        }
        const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);
        const std::optional<FarTerrainStep> initial =
            compatibleInitialStepFor(coordinate, coarsestAllowed);
        if (initial) {
            _farTerrainDisplayedByTile.insert_or_assign(
                coordinate, FarTerrainKey{coordinate.x, coordinate.z, *initial});
        }
    }

    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (_farTerrainTransitions.size() + _farTerrainAuthorityTransitions.size() >=
            FAR_TERRAIN_MAX_SIMULTANEOUS_LOD_TRANSITIONS)
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
        if (_farTerrainAuthorityTransitions.contains(displayed->second))
            continue;
        const FarTerrainStepMask readySteps = residentStepMaskFor(coordinate);
        const std::optional<FarTerrainStep> readyTarget = farTerrainReadyTransitionTarget(
            displayed->second.step, desired->second.step, readySteps, false);
        if (!readyTarget)
            continue;
        FarTerrainStep transitionTarget = *readyTarget;
        if (const auto& requested = _farTerrainProtectedNearHandoff.requestedCenter();
            requested && !farTerrainProtectedIntermediateMayDisplay(
                             farTerrainProtectedNearRole(*requested, coordinate) !=
                                 FarTerrainProtectedNearRole::NONE,
                             displayedQualityFor(coordinate, transitionTarget))) {
            continue;
        }
        if (!stepCompatibleAt(coordinate, transitionTarget)) {
            const FarTerrainStep bridge =
                farTerrainNextDisplayedStep(displayed->second.step, desired->second.step);
            if (bridge == displayed->second.step ||
                (readySteps & farTerrainStepMask(bridge)) == 0 ||
                !stepCompatibleAt(coordinate, bridge)) {
                continue;
            }
            transitionTarget = bridge;
        }
        float parentAgeSeconds = std::numeric_limits<float>::infinity();
        if (const auto grace = findNearGrace(coordinate);
            grace != _farTerrainNearGraceStartedAt.end()) {
            parentAgeSeconds = static_cast<float>(lodTimeSeconds - grace->second);
        }
        if (farTerrainDeferNearIntermediate(displayed->second.step, desired->second.step,
                                            transitionTarget, parentAgeSeconds)) {
            continue;
        }
        const FarTerrainKey next{coordinate.x, coordinate.z, transitionTarget};
        if (!isResident(next) || _farTerrainAuthorityTransitions.contains(next))
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
        const bool protectedGpuResidency = farTerrainProtectedGpuResidencyRequired(
            it->first, _farTerrainProtectedNearHandoff.activeCenter(),
            _farTerrainProtectedNearHandoff.requestedCenter());
        const bool activeTile = _farTerrainActiveTiles.contains(coordinate);
        bool keep = protectedGpuResidency;
        if (activeTile) {
            const auto desired = _farTerrainDesiredByTile.find(coordinate);
            const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
            const auto transition = _farTerrainTransitions.find(coordinate);
            const bool authorityTransition = _farTerrainAuthorityTransitions.contains(it->first);
            const bool retainedStepOnePrefetch =
                it->first.step == FarTerrainStep::ONE && _farTerrainWanted.contains(it->first);
            const bool progressiveBridge =
                desired != _farTerrainDesiredByTile.end() &&
                displayed != _farTerrainDisplayedByTile.end() &&
                farTerrainRetainsProgressiveStep(it->first.step, displayed->second.step,
                                                 desired->second.step);
            keep = protectedGpuResidency || farTerrainIsBaseStep(it->first.step) ||
                   retainedStepOnePrefetch || progressiveBridge || authorityTransition ||
                   (desired != _farTerrainDesiredByTile.end() && desired->second == it->first) ||
                   (displayed != _farTerrainDisplayedByTile.end() &&
                    displayed->second == it->first) ||
                   (transition != _farTerrainTransitions.end() &&
                    (transition->second.from == it->first || transition->second.to == it->first));
        }
        if (!keep) {
            const auto fallback = _farCanopyLodFallbacks.find(coordinate);
            const bool retainedCanopyFallback =
                fallback != _farCanopyLodFallbacks.end() && fallback->second == it->first;
            if (const auto canopy = _farCanopyAttachments.find(it->first);
                canopy != _farCanopyAttachments.end() && !retainedCanopyFallback) {
                if (canopy->second.alloc) {
                    _farMegaBuffer->deferFree(*canopy->second.alloc, _frameRing.frameIndex());
                }
                _farCanopyAttachments.erase(canopy);
                clearCanopyFallbackReference(it->first);
            }
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
    // Fallbacks are optional and live no longer than their active tile. This
    // sweep also removes any reference retired by arena pressure, bounding the
    // independent lifetime without retaining the source terrain mesh.
    for (auto it = _farCanopyLodFallbacks.begin(); it != _farCanopyLodFallbacks.end();) {
        const bool active = _farTerrainActiveTiles.contains(it->first);
        const auto canopy = _farCanopyAttachments.find(it->second);
        if (!active || canopy == _farCanopyAttachments.end() ||
            !canopyStateHasGeometry(canopy == _farCanopyAttachments.end() ? nullptr
                                                                          : &canopy->second)) {
            if (!active && canopy != _farCanopyAttachments.end()) {
                if (canopy->second.alloc) {
                    _farMegaBuffer->deferFree(*canopy->second.alloc, _frameRing.frameIndex());
                }
                _farCanopyAttachments.erase(canopy);
            }
            it = _farCanopyLodFallbacks.erase(it);
        } else {
            ++it;
        }
    }

    const auto farPlannerCompletedAt = std::chrono::steady_clock::now();
    const auto elapsedMilliseconds = [](auto beginning, auto end) {
        return static_cast<float>(
            std::chrono::duration<double, std::milli>(end - beginning).count());
    };
    const float farPlannerSelectionMilliseconds =
        elapsedMilliseconds(farPlannerStartedAt, farPlannerSelectionCompletedAt);
    const float farPlannerPublicationMilliseconds =
        elapsedMilliseconds(farPlannerSelectionCompletedAt, farPlannerPublicationCompletedAt);
    const float farPlannerResidencyMilliseconds =
        elapsedMilliseconds(farPlannerPublicationCompletedAt, farPlannerCompletedAt);
    const float farPlannerMilliseconds =
        elapsedMilliseconds(farPlannerStartedAt, farPlannerCompletedAt);
    _farTerrainPlannerTimings.record(farPlannerMilliseconds);
    _farTerrainSelectionTimings.record(farPlannerSelectionMilliseconds);
    _farTerrainPublicationTimings.record(farPlannerPublicationMilliseconds);
    _farTerrainResidencyTimings.record(farPlannerResidencyMilliseconds);

    uint32_t drawn = 0;
    uint32_t baseDrawn = 0;
    uint32_t refinementDrawn = 0;
    uint32_t frustumCulled = 0;
    uint32_t occlusionCulled = 0;
    std::array<uint32_t, 6> farTierDrawn{};
    const auto diagnosticTierIndex = [](FarTerrainStep step) -> std::optional<size_t> {
        switch (step) {
            case FarTerrainStep::ONE:
                return 0;
            case FarTerrainStep::TWO:
                return 1;
            case FarTerrainStep::FOUR:
                return 2;
            case FarTerrainStep::EIGHT:
                return 3;
            case FarTerrainStep::SIXTEEN:
                return 4;
            case FarTerrainStep::THIRTY_TWO:
                return 5;
        }
        return std::nullopt;
    };
    const float exactHandoffBlocks = exactHandoff.distanceBlocks;

    if (drawGeometry && encoder) {
        _farShadowDrawPlans.clear();
        [encoder setRenderPipelineState:_pipelineState];
        [encoder setDepthStencilState:_depthState];
        [encoder setFrontFacingWinding:MTLWindingCounterClockwise];
        [encoder setCullMode:MTLCullModeBack];
        // Exact terrain draws first. The shader clips each far fragment with its
        // destination-column ownership bit; the depth bias remains only as a
        // conservative fallback while an exact column is still cold.
        [encoder setDepthBias:4.0f slopeScale:1.0f clamp:0.000002f];

        TerrainHorizonCuller horizon(viewpoint);
        auto displayedKeyFor = [&](ColumnPos coordinate) -> std::optional<FarTerrainKey> {
            if (!_farTerrainActiveTiles.contains(coordinate))
                return std::nullopt;
            const FarTerrainKey base{coordinate.x, coordinate.z, FAR_TERRAIN_BASE_STEP};
            if (!isResident(base))
                return std::nullopt;
            const FarTerrainStep coarsestAllowed = coarsestFallbackFor(coordinate);

            if (const auto transition = _farTerrainTransitions.find(coordinate);
                transition != _farTerrainTransitions.end()) {
                // Water remains source-owned until completion. Complete terrain
                // topologies swap together under the narrow fog pulse.
                if (isResident(transition->second.from))
                    return transition->second.from;
                if (isResident(transition->second.to))
                    return transition->second.to;
            }
            if (const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
                displayed != _farTerrainDisplayedByTile.end() && isResident(displayed->second)) {
                return displayed->second;
            }
            const std::optional<FarTerrainStep> fallback =
                compatibleInitialStepFor(coordinate, coarsestAllowed);
            if (!fallback)
                return std::nullopt;
            return FarTerrainKey{coordinate.x, coordinate.z, *fallback};
        };
        for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
            if (!farTerrainCoverageDrawEligible(tile.distanceSquared, coverage))
                continue;
            const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
            struct DrawPlan {
                FarTerrainKey key;
                const FarTerrainMeshState* state = nullptr;
                float progress = 0.0F;
                uint32_t flags = FAR_TERRAIN_DRAW_FLAG;
                bool ownsConnectedGeometry = true;
                const FarCanopyMeshState* canopy = nullptr;
            };
            std::array<DrawPlan, 2> plans{};
            size_t planCount = 0;
            const auto matchingResidentCanopy =
                [&](const FarTerrainKey& key,
                    FarTerrainAuthorityQuality surfaceQuality) -> const FarCanopyMeshState* {
                const auto canopy = _farCanopyAttachments.find(key);
                if (canopy != _farCanopyAttachments.end() &&
                    farCanopyMatchesSurface(canopy->second.authorityQuality,
                                            canopy->second.groundingQuality, surfaceQuality)) {
                    return &canopy->second;
                }
                return fallbackCanopyFor({key.tileX, key.tileZ}, surfaceQuality);
            };
            if (const auto transition = _farTerrainTransitions.find(coordinate);
                transition != _farTerrainTransitions.end() && isResident(transition->second.from) &&
                isResident(transition->second.to)) {
                const FarTerrainTransitionSample sample = sampleFarTerrainTransition(
                    static_cast<float>(lodTimeSeconds - transition->second.startedAtSeconds));
                const uint32_t transitionFlags =
                    FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG |
                    (transition->second.from.step == FarTerrainStep::THIRTY_TWO &&
                             transition->second.to.step == FarTerrainStep::TWO
                         ? FAR_TERRAIN_LOD_EMERGENCY_FLAG
                         : 0U);
                const bool sourceOwnsConnectedGeometry =
                    farTerrainLodConnectedGeometryVisible(sample.progress, transitionFlags);
                const bool targetOwnsConnectedGeometry = farTerrainLodConnectedGeometryVisible(
                    sample.progress, transitionFlags | FAR_TERRAIN_LOD_TARGET_FLAG);
                const FarCanopyMeshState* sourceCanopy = matchingResidentCanopy(
                    transition->second.from,
                    _farTerrainMeshes.at(transition->second.from).authorityQuality);
                const FarCanopyMeshState* targetCanopy = matchingResidentCanopy(
                    transition->second.to,
                    _farTerrainMeshes.at(transition->second.to).authorityQuality);
                if (farCanopyLodTargetUsesSourceFallback(sourceCanopy != nullptr,
                                                         targetCanopy != nullptr)) {
                    // The same allocation fills the target half of the
                    // monotonic dither until its independently built
                    // attachment arrives. Matching tile-local coordinates
                    // make this safe across geometric LOD keys.
                    targetCanopy = sourceCanopy;
                }
                plans[planCount++] = {
                    transition->second.from,
                    &_farTerrainMeshes.at(transition->second.from),
                    sample.progress,
                    transitionFlags,
                    sourceOwnsConnectedGeometry,
                    sourceCanopy};
                plans[planCount++] = {
                    transition->second.to,
                    &_farTerrainMeshes.at(transition->second.to),
                    sample.progress,
                    transitionFlags | FAR_TERRAIN_LOD_TARGET_FLAG,
                    targetOwnsConnectedGeometry,
                    targetCanopy};
            } else if (const std::optional<FarTerrainKey> displayed = displayedKeyFor(coordinate)) {
                if (const auto promotion = _farTerrainAuthorityTransitions.find(*displayed);
                    promotion != _farTerrainAuthorityTransitions.end() &&
                    isFinalResident(*displayed)) {
                    const FarCanopyMeshState* sourceCanopy =
                        promotion->second.sourceCanopy &&
                                farCanopyMatchesSurface(
                                    promotion->second.sourceCanopy->authorityQuality,
                                    promotion->second.sourceCanopy->groundingQuality,
                                    promotion->second.source.authorityQuality)
                            ? &*promotion->second.sourceCanopy
                            : nullptr;
                    if (!promotion->second.published) {
                        plans[planCount++] = {*displayed, &promotion->second.source,
                                              0.0F,       FAR_TERRAIN_DRAW_FLAG,
                                              true,       sourceCanopy};
                    } else {
                        const FarTerrainTransitionSample sample =
                            sampleFarTerrainTransition(static_cast<float>(
                                lodTimeSeconds - promotion->second.startedAtSeconds));
                        const uint32_t flags =
                            FAR_TERRAIN_DRAW_FLAG | FAR_TERRAIN_LOD_TRANSITION_FLAG;
                        const bool sourceOwnsConnectedGeometry =
                            farTerrainLodConnectedGeometryVisible(sample.progress, flags);
                        const bool targetOwnsConnectedGeometry =
                            farTerrainLodConnectedGeometryVisible(
                                sample.progress, flags | FAR_TERRAIN_LOD_TARGET_FLAG);
                        plans[planCount++] = {
                            *displayed, &promotion->second.source,   sample.progress,
                            flags,      sourceOwnsConnectedGeometry, sourceCanopy};
                        plans[planCount++] = {
                            *displayed,
                            &_farTerrainMeshes.at(*displayed),
                            sample.progress,
                            flags | FAR_TERRAIN_LOD_TARGET_FLAG,
                            targetOwnsConnectedGeometry,
                            matchingResidentCanopy(
                                *displayed, _farTerrainMeshes.at(*displayed).authorityQuality)};
                    }
                } else {
                    const FarTerrainMeshState& state = _farTerrainMeshes.at(*displayed);
                    plans[planCount++] = {
                        *displayed, &state,
                        0.0F,       FAR_TERRAIN_DRAW_FLAG,
                        true,       matchingResidentCanopy(*displayed, state.authorityQuality)};
                }
            }
            if (planCount == 0)
                continue;

            FarTerrainBounds visibilityBounds = plans.front().state->surfaceBounds;
            for (size_t index = 1; index < planCount; ++index) {
                const FarTerrainBounds& bounds = plans[index].state->surfaceBounds;
                visibilityBounds.minY = std::min(visibilityBounds.minY, bounds.minY);
                visibilityBounds.maxY = std::max(visibilityBounds.maxY, bounds.maxY);
            }
            for (const DrawPlan& plan : std::span(plans).first(planCount)) {
                if (!plan.canopy)
                    continue;
                visibilityBounds.minX = std::min(visibilityBounds.minX, plan.canopy->bounds.minX);
                visibilityBounds.maxX = std::max(visibilityBounds.maxX, plan.canopy->bounds.maxX);
                visibilityBounds.minZ = std::min(visibilityBounds.minZ, plan.canopy->bounds.minZ);
                visibilityBounds.maxZ = std::max(visibilityBounds.maxZ, plan.canopy->bounds.maxZ);
                visibilityBounds.minY = std::min(visibilityBounds.minY, plan.canopy->bounds.minY);
                visibilityBounds.maxY = std::max(visibilityBounds.maxY, plan.canopy->bounds.maxY);
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
            FarTerrainOwnershipUniforms farOwnership =
                farTerrainOwnershipUniforms(coordinate, exactHandoff, exactFloraHandoff);
            const double fullyOpaqueFarRadius =
                static_cast<double>(tileExactHandoffBlocks) + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS;
            const double fullyOpaqueFarRadiusSquared = fullyOpaqueFarRadius * fullyOpaqueFarRadius;
            const FarTerrainMeshState& occluderState = *plans.front().state;
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

            for (const DrawPlan& plan : std::span(plans).first(planCount)) {
                // Target canopies overlap source canopies during their monotonic
                // exchange. A slightly smaller target bias makes matching crowns
                // stable without changing exact-cube ownership.
                const bool transitionTarget = (plan.flags & FAR_TERRAIN_LOD_TARGET_FLAG) != 0U;
                [encoder setDepthBias:(planCount > 1 && !transitionTarget ? 6.0f : 4.0f)
                           slopeScale:1.0f
                                clamp:0.000002f];
                const FarTerrainMeshState& state = *plan.state;
                ChunkOrigin origin{};
                origin.origin = simd_make_float4(static_cast<float>(state.bounds.minX), 0.0f,
                                                 static_cast<float>(state.bounds.minZ), 0.0F);
                origin.overlayColorAndStrength =
                    simd_make_float4(fogColor[0], fogColor[1], fogColor[2], 0.0F);
                if (_worldgenOverlayMode == WorldgenOverlayMode::AUTHORITY) {
                    origin.overlayColorAndStrength =
                        state.authorityQuality == FarTerrainAuthorityQuality::FINAL
                            ? simd_make_float4(0.10f, 0.95f, 0.28f, 0.78f)
                            : simd_make_float4(0.96f, 0.10f, 0.72f, 0.78f);
                } else if (_worldgenOverlayMode == WorldgenOverlayMode::LOD) {
                    const auto color = terrainLodOverlayColor(plan.key.step);
                    origin.overlayColorAndStrength =
                        simd_make_float4(color[0], color[1], color[2], color[3]);
                }
                origin.farMetadata.x = 0;
                origin.farMetadata.y = std::bit_cast<uint32_t>(coverage.distanceBlocks);
                origin.farMetadata.z = std::bit_cast<uint32_t>(plan.progress);
                origin.farMetadata.w = plan.flags;

                // Canopies use a monotonic target-in, source-out exchange, so
                // unrelated forest summaries never pass through an empty phase.
                // Submit only the terrain-matched connected-water owner on
                // either side of the fog-covered topology swap.
                const uint32_t waterIndexCount =
                    plan.ownsConnectedGeometry ? state.alloc.indexCount - state.opaqueIndexCount
                                               : 0;
                if (waterIndexCount > 0) {
                    const double centerX =
                        static_cast<double>(state.surfaceBounds.minX) +
                        static_cast<double>(state.surfaceBounds.maxX - state.surfaceBounds.minX) *
                            0.5;
                    const double centerY =
                        static_cast<double>(state.surfaceBounds.minY) +
                        static_cast<double>(state.surfaceBounds.maxY - state.surfaceBounds.minY) *
                            0.5;
                    const double centerZ =
                        static_cast<double>(state.surfaceBounds.minZ) +
                        static_cast<double>(state.surfaceBounds.maxZ - state.surfaceBounds.minZ) *
                            0.5;
                    const double dx = centerX - cameraPosition.x;
                    const double dy = centerY - cameraPosition.y;
                    const double dz = centerZ - cameraPosition.z;
                    _waterDraws.push_back(WaterDraw{
                        origin.origin, origin.overlayColorAndStrength, origin.farMetadata,
                        farOwnership, state.alloc.vertexBuffer, state.alloc.indexBuffer,
                        state.alloc.vertexOffset,
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
                const FarCanopyMeshState* canopy = plan.canopy;
                const bool matchingCanopy =
                    canopy &&
                    farCanopyMatchesSurface(canopy->authorityQuality, canopy->groundingQuality,
                                            state.authorityQuality);
                const std::optional<FarCanopyMeshState> shadowCanopy =
                    matchingCanopy && canopy->alloc && canopy->alloc->indexCount > 0
                        ? std::optional<FarCanopyMeshState>{*canopy}
                        : std::nullopt;
                _farShadowDrawPlans.push_back(
                    {coordinate, plan.key, state, origin.farMetadata, farOwnership, shadowCanopy});
                if (matchingCanopy && canopy->alloc && canopy->alloc->indexCount > 0) {
                    [encoder setVertexBytes:&origin length:sizeof(origin) atIndex:2];
                    [encoder setFragmentBytes:&farOwnership length:sizeof(farOwnership) atIndex:5];
                    [encoder setVertexBuffer:canopy->alloc->vertexBuffer
                                      offset:canopy->alloc->vertexOffset
                                     atIndex:0];
                    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                        indexCount:canopy->alloc->indexCount
                                         indexType:MTLIndexTypeUInt32
                                       indexBuffer:canopy->alloc->indexBuffer
                                 indexBufferOffset:canopy->alloc->indexOffset];
                }
            }
            if (farTerrainIsBaseStep(plans.front().key.step)) {
                ++baseDrawn;
            } else {
                ++refinementDrawn;
            }
            if (const std::optional<size_t> tier = diagnosticTierIndex(plans.front().key.step)) {
                ++farTierDrawn[*tier];
            }
            ++drawn;
        }
        [encoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    }

    // Retire authority sources only after this frame has encoded both exact
    // and far geometry. Processing completion earlier could let far clipping
    // adopt FINAL ownership after the exact pass had already stayed hidden,
    // creating a one-frame empty column at the handoff.
    for (auto it = _farTerrainAuthorityTransitions.begin();
         it != _farTerrainAuthorityTransitions.end();) {
        const ColumnPos coordinate{it->first.tileX, it->first.tileZ};
        const bool targetResident = isFinalResident(it->first);
        const auto previewChildDependsOnParentSource = [&](const FarTerrainKey& candidate) {
            const auto resident = _farTerrainMeshes.find(candidate);
            return resident != _farTerrainMeshes.end() && resident->second.uploaded &&
                   farTerrainPreviewChildDependsOnParentSource(
                       it->first, candidate, displayedQualityFor(coordinate, candidate.step));
        };
        bool visiblePreviewDependency = false;
        if (farTerrainIsBaseStep(it->first.step)) {
            if (const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
                displayed != _farTerrainDisplayedByTile.end()) {
                visiblePreviewDependency = previewChildDependsOnParentSource(displayed->second);
            }
            if (const auto lodTransition = _farTerrainTransitions.find(coordinate);
                lodTransition != _farTerrainTransitions.end()) {
                visiblePreviewDependency =
                    visiblePreviewDependency ||
                    previewChildDependsOnParentSource(lodTransition->second.from) ||
                    previewChildDependsOnParentSource(lodTransition->second.to);
            }
        }
        const bool completed =
            it->second.published &&
            lodTimeSeconds - it->second.startedAtSeconds >= FAR_TERRAIN_LOD_TRANSITION_SECONDS &&
            !visiblePreviewDependency;
        if (!_farTerrainActiveTiles.contains(coordinate) || !targetResident || completed) {
            // The color pass records this source for the following frame's
            // shadow pass. Keep its arena regions alive through that replay,
            // even if the GPU has already completed the current color frame.
            const uint64_t shadowReplayFrame = _frameRing.frameIndex() + 1;
            if (it->second.source.uploaded) {
                _farMegaBuffer->deferFree(it->second.source.alloc, shadowReplayFrame);
            }
            if (it->second.sourceCanopy && it->second.sourceCanopy->alloc) {
                _farMegaBuffer->deferFree(*it->second.sourceCanopy->alloc, shadowReplayFrame);
            }
            it = _farTerrainAuthorityTransitions.erase(it);
        } else {
            ++it;
        }
    }

    const FarTerrainSchedulerStats schedulerStats = _farTerrainScheduler->stats();
    const uint32_t baseWanted = static_cast<uint32_t>(_farTerrainCandidates.size());
    const uint32_t protectedNearWanted =
        static_cast<uint32_t>(_farTerrainConnectedNearPatchTargets.size());
    std::array<uint32_t, 5> protectedNearTargetsByStep{};
    for (const FarTerrainKey target : _farTerrainConnectedNearPatchTargets) {
        switch (target.step) {
            case FarTerrainStep::ONE:
                ++protectedNearTargetsByStep[0];
                break;
            case FarTerrainStep::TWO:
                ++protectedNearTargetsByStep[1];
                break;
            case FarTerrainStep::FOUR:
                ++protectedNearTargetsByStep[2];
                break;
            case FarTerrainStep::EIGHT:
                ++protectedNearTargetsByStep[3];
                break;
            case FarTerrainStep::SIXTEEN:
                ++protectedNearTargetsByStep[4];
                break;
            case FarTerrainStep::THIRTY_TWO:
                break;
        }
    }
    // Entry and movement gates count publishable surfaces, not raw GPU
    // residency. A PREVIEW payload or a FINAL payload with an incompatible
    // shared boundary must never satisfy the protected-patch contract.
    const size_t compatibleProtectedTargets = std::min(
        {protectedStatusBeforeUploads.presentTargets, protectedStatusBeforeUploads.finalTargets,
         protectedStatusBeforeUploads.exactCompatibleTargets});
    const size_t missingProtectedParents =
        protectedStatusBeforeUploads.expectedFinalParents -
        std::min(protectedStatusBeforeUploads.finalParents,
                 protectedStatusBeforeUploads.expectedFinalParents);
    const size_t missingSharedBoundaries =
        protectedStatusBeforeUploads.expectedSharedBoundaries >
                protectedStatusBeforeUploads.matchingSharedBoundaries
            ? protectedStatusBeforeUploads.expectedSharedBoundaries -
                  protectedStatusBeforeUploads.matchingSharedBoundaries
            : 0;
    const uint32_t protectedNearAuthorityMismatch = static_cast<uint32_t>(std::min(
        std::max(protectedStatusBeforeUploads.mismatchedSharedBoundaries, missingSharedBoundaries),
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    uint32_t protectedNearResident =
        static_cast<uint32_t>(std::min(compatibleProtectedTargets > missingProtectedParents
                                           ? compatibleProtectedTargets - missingProtectedParents
                                           : size_t{0},
                                       static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    if ((protectedStatusBeforeUploads.incompatibleLodBoundaries != 0 ||
         protectedNearAuthorityMismatch != 0) &&
        protectedNearResident == protectedNearWanted && protectedNearResident != 0) {
        --protectedNearResident;
    }
    const uint32_t refinementWanted =
        static_cast<uint32_t>(_farTerrainWanted.size() - _farTerrainCandidates.size());
    const uint32_t refinementResident = static_cast<uint32_t>(_farTerrainResidentRefinementCount);
    const uint32_t criticalWanted = static_cast<uint32_t>(std::min<size_t>(
        _farTerrainCriticalResidencyTargets.size(), std::numeric_limits<uint32_t>::max()));
    const uint32_t criticalResident = static_cast<uint32_t>(
        std::min<size_t>(std::ranges::count_if(_farTerrainCriticalResidencyTargets,
                                               [&](FarTerrainKey key) { return isResident(key); }),
                         std::numeric_limits<uint32_t>::max()));
    std::array<uint32_t, 6> farTierDesired{};
    std::array<uint32_t, 6> farTierResident{};
    std::array<uint32_t, 6> farTierDisplayed{};
    std::array<uint32_t, 6> farTierResidentPreview{};
    std::array<uint32_t, 6> farTierResidentFinal{};
    std::array<uint32_t, 6> farTierDisplayedPreview{};
    std::array<uint32_t, 6> farTierDisplayedFinal{};
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (const std::optional<size_t> tier = diagnosticTierIndex(tile.key.step)) {
            ++farTierDesired[*tier];
        }
    }
    for (const auto& [key, state] : _farTerrainMeshes) {
        if (!state.uploaded || !_farTerrainActiveTiles.contains(ColumnPos{key.tileX, key.tileZ}))
            continue;
        if (const std::optional<size_t> tier = diagnosticTierIndex(key.step)) {
            ++farTierResident[*tier];
            if (state.authorityQuality == FarTerrainAuthorityQuality::PREVIEW) {
                ++farTierResidentPreview[*tier];
            } else {
                ++farTierResidentFinal[*tier];
            }
        }
    }
    for (const auto& [coordinate, key] : _farTerrainDisplayedByTile) {
        if (!_farTerrainActiveTiles.contains(coordinate) || !isResident(key))
            continue;
        if (const std::optional<size_t> tier = diagnosticTierIndex(key.step)) {
            ++farTierDisplayed[*tier];
            const FarTerrainMeshState* displayedState = nullptr;
            if (const auto promotion = _farTerrainAuthorityTransitions.find(key);
                promotion != _farTerrainAuthorityTransitions.end() &&
                !promotion->second.published) {
                displayedState = &promotion->second.source;
            } else if (const auto resident = _farTerrainMeshes.find(key);
                       resident != _farTerrainMeshes.end()) {
                displayedState = &resident->second;
            }
            if (displayedState &&
                displayedState->authorityQuality == FarTerrainAuthorityQuality::FINAL) {
                ++farTierDisplayedFinal[*tier];
            } else {
                ++farTierDisplayedPreview[*tier];
            }
        }
    }
    float worstVisibleProjectedError = 0.0F;
    ColumnPos worstVisibleCoordinate{};
    FarTerrainStep worstVisibleDesired = FAR_TERRAIN_BASE_STEP;
    FarTerrainStep worstVisibleDisplayed = FAR_TERRAIN_BASE_STEP;
    FarTerrainAuthorityQuality worstVisibleQuality = FarTerrainAuthorityQuality::PREVIEW;
    uint32_t worstVisiblePreviewResidentMask = 0;
    uint32_t worstVisibleFinalResidentMask = 0;
    uint32_t visibleProjectedErrorViolations = 0;
    for (const FarTerrainViewTile& tile : _farTerrainCandidates) {
        if (!tile.screenErrorMetrics)
            continue;
        const ColumnPos coordinate{tile.key.tileX, tile.key.tileZ};
        if (!tileVisibleForScheduling(coordinate))
            continue;
        const auto displayed = _farTerrainDisplayedByTile.find(coordinate);
        if (displayed == _farTerrainDisplayedByTile.end() || !isResident(displayed->second))
            continue;
        const FarTerrainAuthorityQuality quality =
            displayedQualityFor(coordinate, displayed->second.step);
        const double error = farTerrainProjectedDisplayErrorPixels(displayed->second.step, quality,
                                                                   *tile.screenErrorMetrics);
        if (error > FAR_TERRAIN_SCREEN_ERROR_TARGET_PIXELS &&
            displayed->second.step != FarTerrainStep::ONE)
            ++visibleProjectedErrorViolations;
        if (error <= worstVisibleProjectedError)
            continue;
        worstVisibleProjectedError = static_cast<float>(error);
        worstVisibleCoordinate = coordinate;
        worstVisibleDesired = tile.key.step;
        worstVisibleDisplayed = displayed->second.step;
        worstVisibleQuality = quality;
        worstVisiblePreviewResidentMask = 0;
        worstVisibleFinalResidentMask = 0;
        for (const FarTerrainStep step :
             {FarTerrainStep::ONE, FarTerrainStep::TWO, FarTerrainStep::FOUR, FarTerrainStep::EIGHT,
              FarTerrainStep::SIXTEEN, FarTerrainStep::THIRTY_TWO}) {
            const auto resident = _farTerrainMeshes.find({coordinate.x, coordinate.z, step});
            if (resident == _farTerrainMeshes.end() || !resident->second.uploaded)
                continue;
            uint32_t& mask = resident->second.authorityQuality == FarTerrainAuthorityQuality::FINAL
                                 ? worstVisibleFinalResidentMask
                                 : worstVisiblePreviewResidentMask;
            mask |= static_cast<uint32_t>(farTerrainStepMask(step));
        }
    }
    const auto saturatingUint32 = [](size_t value) {
        return static_cast<uint32_t>(
            std::min(value, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    };
    std::optional<FarTerrainProtectedNearClosureSnapshot> evaluatedProtectedClosure;
    if (protectedClosureAnchor) {
        evaluatedProtectedClosure = FarTerrainProtectedNearClosureSnapshot{
            .wantedTileCount = protectedNearWanted,
            .residentTileCount = protectedNearResident,
            .missingTileCount = protectedNearWanted - protectedNearResident,
            .boundaryMismatchCount =
                saturatingUint32(protectedStatusBeforeUploads.mismatchedSharedBoundaries),
            .targetCountsByStep = protectedNearTargetsByStep,
            .finalParentCount = saturatingUint32(protectedStatusBeforeUploads.finalParents),
            .finalTargetCount = saturatingUint32(protectedStatusBeforeUploads.finalTargets),
            .exactCompatibleTargetCount =
                saturatingUint32(protectedStatusBeforeUploads.exactCompatibleTargets),
            .lodMismatchCount =
                saturatingUint32(protectedStatusBeforeUploads.incompatibleLodBoundaries),
            .authorityMismatchCount = protectedNearAuthorityMismatch,
            .ready = protectedStatusBeforeUploads.ready(),
            .anchor = *protectedClosureAnchor,
            .viewEpoch = protectedClosureViewEpoch,
            .worldEpoch = protectedClosureWorldEpoch,
            .protectedEpoch = protectedClosureEpoch,
        };
    }
    // A cached prior anchor can already be geometrically complete on the same
    // frame that movement advances the requested scheduler epoch. Publish the
    // retained snapshot for that frame, then retain this newly validated
    // atomic snapshot. Startup observes the stale epoch once and can accept
    // the revalidated closure on the following frame.
    const std::optional<FarTerrainProtectedNearClosureSnapshot> publishedProtectedClosure =
        protectedHandoffChanged && evaluatedProtectedClosure && evaluatedProtectedClosure->ready
            ? _farTerrainProtectedNearClosureSnapshot
            : evaluatedProtectedClosure;
    _farTerrainProtectedNearClosureSnapshot = evaluatedProtectedClosure;
    const FarTerrainProtectedNearClosureSnapshot protectedClosureStats =
        publishedProtectedClosure.value_or(FarTerrainProtectedNearClosureSnapshot{});

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
    _chunkStats.farProtectedNearWantedTileCount = protectedClosureStats.wantedTileCount;
    _chunkStats.farProtectedNearResidentTileCount = protectedClosureStats.residentTileCount;
    _chunkStats.farProtectedNearMissingTileCount = protectedClosureStats.missingTileCount;
    _chunkStats.farProtectedNearBoundaryMismatchCount = protectedClosureStats.boundaryMismatchCount;
    _chunkStats.farProtectedNearTargetCountsByStep = protectedClosureStats.targetCountsByStep;
    _chunkStats.farProtectedNearFinalParentCount = protectedClosureStats.finalParentCount;
    _chunkStats.farProtectedNearFinalTargetCount = protectedClosureStats.finalTargetCount;
    _chunkStats.farProtectedNearExactCompatibleTargetCount =
        protectedClosureStats.exactCompatibleTargetCount;
    _chunkStats.farProtectedNearLodMismatchCount = protectedClosureStats.lodMismatchCount;
    _chunkStats.farProtectedNearAuthorityMismatchCount =
        protectedClosureStats.authorityMismatchCount;
    _chunkStats.farProtectedNearReady = protectedClosureStats.ready;
    _chunkStats.farProtectedNearAnchorTileX = protectedClosureStats.anchor.x;
    _chunkStats.farProtectedNearAnchorTileZ = protectedClosureStats.anchor.z;
    _chunkStats.farProtectedNearViewEpoch = protectedClosureStats.viewEpoch;
    _chunkStats.farProtectedNearWorldEpoch = protectedClosureStats.worldEpoch;
    _chunkStats.farProtectedNearCurrentEpoch = _farTerrainProtectedNearEpoch;
    _chunkStats.farProtectedNearClosureEpoch = protectedClosureStats.protectedEpoch;
    _chunkStats.exactSurfaceEpoch = exactCoverageEpoch;
    _chunkStats.farCriticalWantedTileCount = criticalWanted;
    _chunkStats.farCriticalResidentTileCount = criticalResident;
    _chunkStats.farCriticalMissingTileCount = criticalWanted - criticalResident;
    _chunkStats.farTierDesiredTileCounts = farTierDesired;
    _chunkStats.farTierResidentMeshCounts = farTierResident;
    _chunkStats.farTierDisplayedTileCounts = farTierDisplayed;
    _chunkStats.farTierDrawnTileCounts = farTierDrawn;
    _chunkStats.farTierResidentPreviewCounts = farTierResidentPreview;
    _chunkStats.farTierResidentFinalCounts = farTierResidentFinal;
    _chunkStats.farTierDisplayedPreviewCounts = farTierDisplayedPreview;
    _chunkStats.farTierDisplayedFinalCounts = farTierDisplayedFinal;
    _chunkStats.farWorstVisibleProjectedErrorPixels = worstVisibleProjectedError;
    _chunkStats.farWorstVisibleTileX = worstVisibleCoordinate.x;
    _chunkStats.farWorstVisibleTileZ = worstVisibleCoordinate.z;
    _chunkStats.farWorstVisibleDesiredStep =
        static_cast<uint8_t>(farTerrainStepSize(worstVisibleDesired));
    _chunkStats.farWorstVisibleDisplayedStep =
        static_cast<uint8_t>(farTerrainStepSize(worstVisibleDisplayed));
    _chunkStats.farWorstVisibleDisplayedQuality = static_cast<uint8_t>(worstVisibleQuality);
    _chunkStats.farWorstVisiblePreviewResidentMask = worstVisiblePreviewResidentMask;
    _chunkStats.farWorstVisibleFinalResidentMask = worstVisibleFinalResidentMask;
    _chunkStats.farVisibleProjectedErrorViolationCount = visibleProjectedErrorViolations;
    _chunkStats.farVisiblePerceptualFinalRequestCount = static_cast<uint32_t>(std::min<size_t>(
        _farTerrainPerceptualFinalRequests.size(), std::numeric_limits<uint32_t>::max()));
    _chunkStats.farPendingAuthorityTransitionCount = static_cast<uint32_t>(std::min<size_t>(
        _farTerrainAuthorityTransitions.size(), std::numeric_limits<uint32_t>::max()));
    _chunkStats.farExactHandoffMissingFinalParentCount = missingFinalHandoffParents;
    _chunkStats.farBaseViewDistanceChunks = visibleChunks;
    _chunkStats.farBaseCenterTileX = centerTile.x;
    _chunkStats.farBaseCenterTileZ = centerTile.z;
    _chunkStats.farBaseWorldEpoch = _farTerrainWorldEpoch;
    _chunkStats.farBaseViewEpoch = _farTerrainViewEpoch;
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
    _chunkStats.farCriticalSchedulerDisplacementCount = schedulerStats.criticalDisplacements;
    _chunkStats.farWorkerBudget = static_cast<uint32_t>(schedulerStats.workerBudget);
    _chunkStats.farStep32WaterGridCalls = schedulerStats.step32WaterGridCalls;
    _chunkStats.farStep32WaterGridSamples = schedulerStats.step32WaterGridSamples;
    _chunkStats.farStep32WaterPointSamples = schedulerStats.step32WaterPointSamples;
    _chunkStats.farStep32WaterDenseGridCalls = schedulerStats.step32WaterDenseGridCalls;
    _chunkStats.farCachedBaseTileCount = static_cast<uint32_t>(schedulerStats.cacheBaseEntries);
    _chunkStats.farCanopyInFlightCount = static_cast<uint32_t>(schedulerStats.canopyInFlight);
    _chunkStats.farActiveCanopyWorkerCount =
        static_cast<uint32_t>(schedulerStats.activeCanopyWorkers);
    _chunkStats.farQueuedCanopyCount = static_cast<uint32_t>(schedulerStats.queuedCanopy);
    _chunkStats.farParkedCanopyCount = static_cast<uint32_t>(schedulerStats.parkedCanopy);
    _chunkStats.farCompletedCanopyCount = static_cast<uint32_t>(schedulerStats.completedCanopy);
    _chunkStats.farCanopyCacheEntryCount = static_cast<uint32_t>(schedulerStats.canopyCacheEntries);
    _chunkStats.farCanopyFailedCount = schedulerStats.canopyFailed;
    _chunkStats.farCanopyDeferredCount = schedulerStats.canopyDeferred;
    _chunkStats.farCanopyAuthorityCompletionResumeCount =
        schedulerStats.canopyAuthorityCompletionResumes;
    _chunkStats.farCoverageFrontierBlocks = coverage.distanceBlocks;
    _chunkStats.farCacheMB = static_cast<float>(schedulerStats.cacheBytes) / (1024.0f * 1024.0f);
    _chunkStats.farCanopyCacheMB =
        static_cast<float>(schedulerStats.canopyCacheBytes) / (1024.0f * 1024.0f);
    _chunkStats.farMegaUsedMB =
        static_cast<float>(_farMegaBuffer->vertexUsed() + _farMegaBuffer->indexUsed()) /
        (1024.0f * 1024.0f);
    _chunkStats.farPlannerMsLast = farPlannerMilliseconds;
    _chunkStats.farPlannerMsP95 = _farTerrainPlannerTimings.percentile95Milliseconds();
    _chunkStats.farPlannerMsMax = _farTerrainPlannerTimings.maximumMilliseconds();
    _chunkStats.farPlannerSelectionMsP95 = _farTerrainSelectionTimings.percentile95Milliseconds();
    _chunkStats.farPlannerPublicationMsP95 =
        _farTerrainPublicationTimings.percentile95Milliseconds();
    _chunkStats.farPlannerResidencyMsP95 = _farTerrainResidencyTimings.percentile95Milliseconds();
    _chunkStats.farArenaAdmissionDeniedCount = _farTerrainArenaAdmissionDeniedCount;
    _chunkStats.farNearArenaReclaimCount = _farTerrainNearArenaReclaimCount;
    _chunkStats.farNearArenaReclaimedBytes = _farTerrainNearArenaReclaimedBytes;
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

RenderPipeline::AtmosphericRenderStats RenderPipeline::atmosphericRenderStats() const {
    AtmosphericRenderStats stats;
    stats.shadowRefreshMask = _shadowRefreshMask;
    stats.shadowCasterCounts = _shadowCasterCounts;
    if (_shadowMap) {
        for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
            stats.shadowRefreshCounts[cascade] = _shadowMap->refreshCount(cascade);
        }
    }
    stats.indirectHistoryResetMask = _indirectHistoryResetMask;
    stats.indirectHistoryValid = _gfx.indirectLightingQuality == 0 ||
                                 (_screenSpaceLighting && _screenSpaceLighting->historyValid());
    stats.cloudHistoryValid = _gfx.cloudQuality == 0 || (_clouds && _clouds->historyValid());
    stats.froxelHistoryValid =
        !_gfx.volumetricLight || (_volumetrics && _volumetrics->historyValid());
    if (_atmosphere) {
        stats.atmosphereSlowRefreshCount = _atmosphere->slowRefreshCount();
        stats.atmosphereSkyRefreshCount = _atmosphere->skyRefreshCount();
    }
    stats.indirectPersistentBytes =
        _screenSpaceLighting ? _screenSpaceLighting->persistentBytes() : 0U;
    stats.cloudPersistentBytes = _clouds ? _clouds->persistentBytes() : 0U;
    stats.froxelPersistentBytes = _volumetrics ? _volumetrics->persistentBytes() : 0U;
    stats.integratedPersistentBytes =
        atmosphericSceneTargetMemoryBytes(_displayWidth, _displayHeight) +
        (_shadowMap && _shadowMap->quality() != 0U ? shadowMapMemoryBytes(_shadowMap->quality())
                                                   : 0U) +
        stats.indirectPersistentBytes + (_atmosphere ? atmosphereLutMemoryBytes() : 0U) +
        stats.cloudPersistentBytes + stats.froxelPersistentBytes +
        (_lightning ? LIGHTNING_RENDERER_MEMORY_BYTES : 0U);
    stats.lightningEventId = _lastLightningEventId;
    stats.lunarPhaseEnergy = _lunarPhaseEnergy;
    stats.lunarPhaseCycle = _lunarPhaseCycle;
    return stats;
}

// ---------------------------------------------------------------------------
// renderWater, the water surfaces recorded by renderChunks, drawn into the
// resolved scene color with refraction from a copy of the scene. The shader
// rejects water behind opaque depth, and hardware depth writes the nearest
// visible interface into media depth for later atmospheric passes. The
// underwater absorption and scattering overlay runs here when submerged.
// ---------------------------------------------------------------------------
void RenderPipeline::renderWater(id<MTLCommandBuffer> commandBuffer, const Mat4& viewMatrix,
                                 const Mat4& projectionMatrix, const Vec3& cameraPosition,
                                 bool cameraUnderwater, const SkyUniforms& skyUniforms,
                                 const float directLightDirection[3],
                                 const float directLightRadiance[3], const float fogColor[3]) {
    // Preserve opaque depth, then let visible water surfaces replace it with
    // the nearer interface. Later air-medium and precipitation passes stop at
    // this combined depth instead of fogging or raining through water.
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit) {
        [blit copyFromTexture:_depthResolve
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(_depthResolve.width, _depthResolve.height, 1)
                    toTexture:_mediaDepthResolve
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        if (!_waterDraws.empty()) {
            [blit copyFromTexture:_colorResolve
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(_colorResolve.width, _colorResolve.height, 1)
                        toTexture:_sceneColorCopy
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            // The level-zero copy remains the exact refraction source. Build
            // lower levels only while SSR can consume them. The persistent
            // allocation is accounted as a complete pyramid at every quality;
            // this conditional skips only the per-frame filtering work.
            if (_gfx.waterReflections) {
                [blit generateMipmapsForTexture:_sceneColorCopy];
            }
        }
        [blit endEncoding];
    }
    if (_waterDraws.empty() && !cameraUnderwater) {
        return;
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
    wu.directLightDirection =
        simd_make_float3(directLightDirection[0], directLightDirection[1], directLightDirection[2]);
    wu.directLightRadiance =
        simd_make_float3(directLightRadiance[0], directLightRadiance[1], directLightRadiance[2]);
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
    wu.solarDirection = skyUniforms.sunDirection;
    wu.physicalSkyBlend = 1.0F - skyUniforms.visibilityAndPhase.w;
    wu.directSpecularFactor = _directSpecularFactor;
    FrameRing::Alloc waterAlloc = _frameRing.push(&wu, sizeof(WaterUniforms));

    auto passDesc = [[MTLRenderPassDescriptor alloc] init];
    passDesc.colorAttachments[0].texture = _colorResolve;
    passDesc.colorAttachments[0].loadAction = MTLLoadActionLoad;
    passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
    passDesc.depthAttachment.texture = _mediaDepthResolve;
    passDesc.depthAttachment.loadAction = MTLLoadActionLoad;
    passDesc.depthAttachment.storeAction = MTLStoreActionStore;
    _gpuTimer->attachPass(passDesc, "water");

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
    if (!encoder) {
        resetMetalObject(passDesc);
        return;
    }

    if (!_waterDraws.empty()) {
        // Back-to-front: the nearest surface draws last and wins the pixel
        std::sort(_waterDraws.begin(), _waterDraws.end(),
                  [](const WaterDraw& a, const WaterDraw& b) { return a.distSq > b.distSq; });

        [encoder setRenderPipelineState:_waterPipelineState];
        [encoder setDepthStencilState:_depthState];
        [encoder setCullMode:MTLCullModeNone]; // surface visible from below
        [encoder setVertexBuffer:_frameUniforms.buffer offset:_frameUniforms.offset atIndex:1];
        [encoder setVertexBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
        [encoder setFragmentBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
        [encoder setFragmentTexture:_sceneColorCopy atIndex:0];
        [encoder setFragmentTexture:_depthResolve atIndex:1];
        [encoder setFragmentTexture:_atmosphere->skyViewTexture() atIndex:2];
        [encoder setFragmentTexture:_clouds->shadowTexture() atIndex:3];
        [encoder setFragmentTexture:_shadowMap->nearDepthTexture() atIndex:4];
        [encoder setFragmentTexture:_shadowMap->farDepthTexture() atIndex:5];
        [encoder setFragmentTexture:_shadowMap->horizonDepthTexture() atIndex:6];
        [encoder setFragmentSamplerState:_shadowMap->comparisonSampler() atIndex:1];
        FrameRing::Alloc shadowAlloc =
            _frameRing.push(&_sceneShadowUniforms, sizeof(ShadowUniforms));
        [encoder setFragmentBuffer:shadowAlloc.buffer offset:shadowAlloc.offset atIndex:4];
        const CloudShadowUniforms& cloudShadow = _clouds->shadowUniforms();
        [encoder setFragmentBytes:&cloudShadow length:sizeof(cloudShadow) atIndex:6];

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
        [encoder setDepthStencilState:_noDepthWriteState];
        [encoder setFragmentBuffer:waterAlloc.buffer offset:waterAlloc.offset atIndex:3];
        [encoder setFragmentTexture:_depthResolve atIndex:1];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }

    [encoder endEncoding];
    resetMetalObject(passDesc);
}

// ---------------------------------------------------------------------------
// renderBlockHighlight (Task 6.9)
// ---------------------------------------------------------------------------
void RenderPipeline::renderBlockHighlight(id<MTLRenderCommandEncoder> encoder,
                                          const BlockHighlight& highlight, const Mat4& viewMatrix,
                                          const Mat4& projectionMatrix) {
    // Expand the authored bounds slightly on every side to prevent z-fighting.
    // The unit wireframe is scaled before translation, so beds and torches no
    // longer receive a floating full-voxel outline.
    Uniforms uniforms{};
    constexpr float HIGHLIGHT_OFFSET = 0.002F;
    const Vec3 minimum = highlight.blockPosition + highlight.localBounds.min -
                         Vec3{HIGHLIGHT_OFFSET, HIGHLIGHT_OFFSET, HIGHLIGHT_OFFSET};
    const Vec3 extent =
        highlight.localBounds.max - highlight.localBounds.min +
        Vec3{2.0F * HIGHLIGHT_OFFSET, 2.0F * HIGHLIGHT_OFFSET, 2.0F * HIGHLIGHT_OFFSET};
    uniforms.modelMatrix = matrix_identity_float4x4;
    uniforms.modelMatrix.columns[0].x = extent.x;
    uniforms.modelMatrix.columns[1].y = extent.y;
    uniforms.modelMatrix.columns[2].z = extent.z;
    uniforms.modelMatrix.columns[3] = simd_make_float4(minimum.x, minimum.y, minimum.z, 1.0F);

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
    // shadow cascade. Bind both (a disabled shadow block, strength 0, keeps
    // the yellow wireframe fully lit) so it never reads stale chunk-loop state.
    [encoder setFragmentTexture:_blockTextures->texture() atIndex:0];
    [encoder setFragmentTexture:_blockTextures->emissionMask() atIndex:5];
    [encoder setFragmentSamplerState:_blockTextures->sampler() atIndex:0];
    ShadowUniforms noShadows{};
    FrameRing::Alloc noShadowAlloc = _frameRing.push(&noShadows, sizeof(ShadowUniforms));
    [encoder setFragmentTexture:_shadowMap->nearDepthTexture() atIndex:1];
    [encoder setFragmentTexture:_shadowMap->farDepthTexture() atIndex:2];
    [encoder setFragmentTexture:_shadowMap->horizonDepthTexture() atIndex:3];
    [encoder setFragmentTexture:_clouds->shadowTexture() atIndex:4];
    [encoder setFragmentSamplerState:_shadowMap->comparisonSampler() atIndex:1];
    [encoder setFragmentBuffer:noShadowAlloc.buffer offset:noShadowAlloc.offset atIndex:4];
    const FarTerrainOwnershipUniforms noFarOwnership{};
    [encoder setFragmentBytes:&noFarOwnership length:sizeof(noFarOwnership) atIndex:5];
    const CloudShadowUniforms noCloudShadow{};
    [encoder setFragmentBytes:&noCloudShadow length:sizeof(noCloudShadow) atIndex:6];

    // Draw 12 lines (24 vertices) for wireframe box
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:24];
}

// ---------------------------------------------------------------------------
// renderUIOverlay (Task 6.10 + hotbar)
// ---------------------------------------------------------------------------
void RenderPipeline::renderUIOverlay(id<MTLRenderCommandEncoder> encoder,
                                     const UIFrameState& uiFrame) {
    _uiOverlay->beginFrame();
    drawGameHud(*_uiOverlay, uiFrame, _displayWidth, _displayHeight);
    if (uiFrame.screen != GameScreen::PLAYING) {
        drawMenu(*_uiOverlay, uiFrame, _displayWidth, _displayHeight);
    }
    _uiOverlay->flush(encoder);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
RenderPipeline::~RenderPipeline() {
    releaseSceneTargets();
    resetMetalObject(_pipelineState);
    resetMetalObject(_coherentResolvePipelineState);
    resetMetalObject(_depthState);
    resetMetalObject(_skyPipelineState);
    resetMetalObject(_skyDepthState);
    resetMetalObject(_noDepthWriteState);
    resetMetalObject(_highlightPipelineState);
    resetMetalObject(_highlightVertexBuffer);
    resetMetalObject(_waterPipelineState);
    resetMetalObject(_underwaterOverlayState);
}

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
    if (_screenSpaceLighting) {
        _screenSpaceLighting->resize(_displayWidth, _displayHeight);
    }
    if (_clouds) {
        _clouds->resize(_displayWidth, _displayHeight);
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
// setGraphicsSettings, the engine pushes a copy on init and on every video
// settings change; passes read the copy each frame and skip when disabled.
// ---------------------------------------------------------------------------
void RenderPipeline::setGraphicsSettings(const GraphicsSettings& gfx) {
    _gfx = gfx;
    setBloomIntensity(gfx.bloomIntensity()); // level 5 = stock 1.0; 0 skips
    if (_screenSpaceLighting) {
        _screenSpaceLighting->setQuality(gfx.indirectLightingQuality);
    }
    if (_clouds) {
        _clouds->setQuality(gfx.cloudQuality);
    }
}

// ---------------------------------------------------------------------------
// tickParticles, Update weather particle physics each game tick
// ---------------------------------------------------------------------------
void RenderPipeline::tickParticles(float dt, const World& world, const Vec3& playerPosition,
                                   const WeatherSample& weather) {
    if (!_particles)
        return;
    _particles->tick(dt, world, playerPosition, weather);
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
