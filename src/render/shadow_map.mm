#import "render/shadow_map.hpp"

#include "common/error.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// The caster margin pulls each projection back toward the light so geometry
// above its receiver slice remains eligible. Horizon terrain needs more room
// than the detailed receiver slices, but the bound prevents a low sun from
// wasting most of the depth range on empty space.
// ---------------------------------------------------------------------------
namespace {
simd_float4x4 toSimd(const Mat4& m) {
    simd_float4x4 out;
    std::memcpy(&out, m.data.data(), sizeof(float) * 16);
    return out;
}

} // namespace

ShadowMap::ShadowMap(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
                     MTLVertexDescriptor* vertexDescriptor)
    : _device(device) {
    // ---- Depth-only chunk pipeline (cutout-aware) ----
    id<MTLFunction> chunkVertex = [shaderLibrary newFunctionWithName:@"shadowVertexMain"];
    id<MTLFunction> cutoutFragment = [shaderLibrary newFunctionWithName:@"shadowCutoutFragment"];
    if (!chunkVertex || !cutoutFragment) {
        RY_LOG_FATAL("Failed to load shadow shader functions");
    }

    NSError* error = nil;
    auto chunkDesc = [[MTLRenderPipelineDescriptor alloc] init];
    chunkDesc.vertexFunction = chunkVertex;
    chunkDesc.fragmentFunction = cutoutFragment;
    chunkDesc.vertexDescriptor = vertexDescriptor;
    chunkDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
    _chunkPipeline = [_device newRenderPipelineStateWithDescriptor:chunkDesc error:&error];
    resetMetalObject(chunkDesc);
    resetMetalObject(chunkVertex);
    resetMetalObject(cutoutFragment);
    if (!_chunkPipeline) {
        RY_LOG_FATAL("Failed to create shadow chunk pipeline state");
    }

    id<MTLFunction> entityVertex = [shaderLibrary newFunctionWithName:@"entityShadowVertexMain"];
    if (!entityVertex) {
        RY_LOG_FATAL("Failed to load dynamic-object shadow vertex function");
    }
    auto entityDesc = [[MTLRenderPipelineDescriptor alloc] init];
    entityDesc.vertexFunction = entityVertex;
    entityDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
    _entityPipeline = [_device newRenderPipelineStateWithDescriptor:entityDesc error:&error];
    resetMetalObject(entityDesc);
    resetMetalObject(entityVertex);
    if (!_entityPipeline) {
        RY_LOG_FATAL("Failed to create dynamic-object shadow pipeline state");
    }

    // Shadow casters write depth, standard less-than test.
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    resetMetalObject(depthDesc);

    // Hardware PCF: sample_compare against the stored depth, bilinear between
    // the 4 texels for a soft edge at no extra taps.
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.compareFunction = MTLCompareFunctionLess;
    _comparisonSampler = [_device newSamplerStateWithDescriptor:samplerDesc];
    resetMetalObject(samplerDesc);

    allocateTextures();
}

ShadowMap::~ShadowMap() {
    resetMetalObject(_nearDepthTexture);
    resetMetalObject(_farDepthTexture);
    resetMetalObject(_horizonDepthTexture);
    resetMetalObject(_depthState);
    resetMetalObject(_chunkPipeline);
    resetMetalObject(_entityPipeline);
    resetMetalObject(_comparisonSampler);
}

void ShadowMap::allocateTextures() {
    const uint32_t nearResolution = shadowCascadeConfiguration(_quality, 0u).resolution;
    const uint32_t farResolution = shadowCascadeConfiguration(_quality, 2u).resolution;
    const uint32_t horizonResolution =
        shadowCascadeConfiguration(_quality, SHADOW_HORIZON_CASCADE_INDEX).resolution;

    auto arrayDescriptor = [[MTLTextureDescriptor alloc] init];
    arrayDescriptor.textureType = MTLTextureType2DArray;
    arrayDescriptor.pixelFormat = PixelFormats::SCENE_DEPTH;
    arrayDescriptor.arrayLength = 2;
    arrayDescriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    arrayDescriptor.storageMode = MTLStorageModePrivate;

    arrayDescriptor.width = nearResolution;
    arrayDescriptor.height = nearResolution;
    id<MTLTexture> nearDepthTexture = [_device newTextureWithDescriptor:arrayDescriptor];
    nearDepthTexture.label = @"Shadow near cascades";

    arrayDescriptor.width = farResolution;
    arrayDescriptor.height = farResolution;
    id<MTLTexture> farDepthTexture = [_device newTextureWithDescriptor:arrayDescriptor];
    farDepthTexture.label = @"Shadow far cascades";

    auto horizonDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:PixelFormats::SCENE_DEPTH
                                                           width:horizonResolution
                                                          height:horizonResolution
                                                       mipmapped:false];
    horizonDescriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    horizonDescriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> horizonDepthTexture = [_device newTextureWithDescriptor:horizonDescriptor];
    horizonDepthTexture.label = @"Shadow horizon cascade";

    if (!nearDepthTexture || !farDepthTexture || !horizonDepthTexture) {
        RY_LOG_FATAL("Failed to allocate grouped shadow depth textures");
    }
    resetMetalObject(_nearDepthTexture);
    resetMetalObject(_farDepthTexture);
    resetMetalObject(_horizonDepthTexture);
    _nearDepthTexture = nearDepthTexture;
    _farDepthTexture = farDepthTexture;
    _horizonDepthTexture = horizonDepthTexture;
    resetMetalObject(arrayDescriptor);
}

void ShadowMap::setQuality(uint32_t quality) {
    const uint32_t normalizedQuality = quality >= 2u ? 2u : 1u;
    if (normalizedQuality == _quality) {
        return;
    }
    _quality = normalizedQuality;
    _hasRenderedCascades = false;
    allocateTextures();
}

uint32_t ShadowMap::resolution(int cascade) const {
    return shadowCascadeConfiguration(_quality, static_cast<uint32_t>(cascade)).resolution;
}

MTLRenderPassDescriptor* ShadowMap::passDescriptor(int cascade) const {
    const ShadowCascadeConfiguration configuration =
        shadowCascadeConfiguration(_quality, static_cast<uint32_t>(cascade));
    // The caller owns this descriptor and releases it after encoding. Shadow
    // refreshes run every frame for the near cascades, so relying on an outer
    // autorelease pool here would retain descriptor graphs between pool drains
    // in manual-reference-counted builds.
    auto desc = [[MTLRenderPassDescriptor alloc] init];
    switch (configuration.textureGroup) {
        case ShadowTextureGroup::NEAR:
            desc.depthAttachment.texture = _nearDepthTexture;
            desc.depthAttachment.slice = configuration.textureSlice;
            break;
        case ShadowTextureGroup::FAR:
            desc.depthAttachment.texture = _farDepthTexture;
            desc.depthAttachment.slice = configuration.textureSlice;
            break;
        case ShadowTextureGroup::HORIZON:
            desc.depthAttachment.texture = _horizonDepthTexture;
            break;
    }
    desc.depthAttachment.loadAction = MTLLoadActionClear;
    desc.depthAttachment.storeAction = MTLStoreActionStore;
    desc.depthAttachment.clearDepth = 1.0;
    return desc;
}

void ShadowMap::computeCascades(const Vec3& cameraPos, const Vec3& cameraForward, float fovY,
                                float aspect, const Vec3& lightDir, float strength) {
    const Vec3 light = lightDir.normalize();
    const Vec3 forward = cameraForward.normalize();
    _casterCameraPosition = cameraPos;
    _casterCameraForward = forward;
    _casterLightDirection = light;
    _shadowUniforms = ShadowUniforms{};
    _shadowUniforms.cameraPositionAndStrength =
        simd_make_float4(cameraPos.x, cameraPos.y, cameraPos.z, strength);
    _shadowUniforms.cameraForwardAndPadding =
        simd_make_float4(forward.x, forward.y, forward.z, 0.0f);

    const float tanHalf = std::tan(fovY * 0.5f);
    const Vec3 up = std::fabs(light.y) > 0.99f ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 1.f, 0.f};

    for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
        const ShadowCascadeConfiguration configuration =
            shadowCascadeConfiguration(_quality, static_cast<uint32_t>(c));
        const float n = c == 0 ? configuration.nearDepth
                               : shadowCascadeBlendStart(shadowCascadeConfiguration(
                                     _quality, static_cast<uint32_t>(c - 1)));
        const float f = configuration.farDepth;

        // Compute the center and radius analytically. Reconstructing them from
        // absolute corners around z=-111,726 lost a meaningful fraction of a
        // near-cascade texel and made the quantized radius alternate.
        const Vec3 receiverCenter = cameraPos + forward * (0.5f * (n + f));
        const float radius = shadowCascadeBoundingRadius(n, f, tanHalf, aspect);
        const float texelWorldSize = 2.0f * radius / static_cast<float>(configuration.resolution);
        const float normalBias = std::clamp(texelWorldSize * (c < 2 ? 1.5f : 1.0f), 0.015f,
                                            c == SHADOW_HORIZON_CASCADE_INDEX ? 4.0f : 1.5f);
        const float casterMargin =
            shadowCasterMargin(radius, light.y, c == SHADOW_HORIZON_CASCADE_INDEX);
        const float receiverDepthGuard = shadowCascadeReceiverDepthGuard(
            radius, casterMargin, configuration.resolution, normalBias);
        const float depthRange = shadowCascadeDepthRange(radius, casterMargin, receiverDepthGuard);
        const float depthTexelWorldSize = shadowCascadeDepthTexelWorldSize(
            radius, casterMargin, receiverDepthGuard, configuration.resolution);

        // Snap the center in global light-space with double intermediates, then
        // build the matrix around a nearby 256-block anchor. This keeps its
        // translations small without changing the world-aligned shadow grid
        // when the anchor recenters.
        const Vec3 lightZ = light;
        const Vec3 lightX = up.cross(lightZ).normalize();
        const Vec3 lightY = lightZ.cross(lightX);
        const double centerX = shadowPreciseDot(receiverCenter, lightX);
        const double centerY = shadowPreciseDot(receiverCenter, lightY);
        const double centerZ = shadowPreciseDot(receiverCenter, lightZ);
        const double snappedCenterX = shadowSnappedLightCoordinate(centerX, texelWorldSize);
        const double snappedCenterY = shadowSnappedLightCoordinate(centerY, texelWorldSize);
        const double snappedCenterZ = shadowSnappedLightCoordinate(centerZ, depthTexelWorldSize);
        const Vec3 projectionOrigin = shadowProjectionOrigin(cameraPos);
        Vec3 localSphereCenter = receiverCenter - projectionOrigin;
        localSphereCenter += lightX * static_cast<float>(snappedCenterX - centerX);
        localSphereCenter += lightY * static_cast<float>(snappedCenterY - centerY);
        localSphereCenter += lightZ * static_cast<float>(snappedCenterZ - centerZ);

        // Light view sits at the light's side of the sphere (lightDir points
        // toward the sun/moon) looking down at its center; ortho spans the
        // sphere, and the extra depth admits casters above the receiver slice.
        Vec3 eye = localSphereCenter + light * (radius + casterMargin + receiverDepthGuard);
        Mat4 lightView = Mat4::lookAt(eye, localSphereCenter, up);
        Mat4 lightProj = Mat4::orthographic(-radius, radius, -radius, radius, 0.0f, depthRange);

        _cascadeVP[c] = lightProj * lightView;
        _cascadeProjectionOrigins[c] = projectionOrigin;
        _cascadeReceiverCenters[c] = receiverCenter;
        _cascadeReceiverRadii[c] = radius + normalBias;
        ShadowCascadeUniforms& cascade = _shadowUniforms.cascades[c];
        cascade.lightViewProj = toSimd(_cascadeVP[c]);
        cascade.projectionOrigin =
            simd_make_float4(projectionOrigin.x, projectionOrigin.y, projectionOrigin.z, 0.0F);
        cascade.depthRange = simd_make_float4(n, f, shadowCascadeBlendStart(configuration), 1.0f);

        const float filterRadius = c < 2 ? 1.5f : (c < SHADOW_HORIZON_CASCADE_INDEX ? 1.25f : 1.0f);
        const float receiverDepthBias =
            (c < 2 ? 1.5f : 2.0f) / static_cast<float>(configuration.resolution);
        cascade.samplingParams =
            simd_make_float4(texelWorldSize, normalBias, filterRadius, receiverDepthBias);
    }
}

uint32_t
ShadowMap::selectRefreshMask(uint64_t frameIndex,
                             const std::array<uint64_t, SHADOW_CASCADE_COUNT>& casterRevisions,
                             const Vec3& lightDirection) {
    uint32_t refreshMask = 0U;
    const Vec3 normalizedLight = lightDirection.normalize();
    const bool lightChanged =
        !_hasRenderedCascades || normalizedLight.dot(_lastLightDirection) < 0.999999F;
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        bool projectionChanged = !_hasRenderedCascades;
        if (_hasRenderedCascades) {
            // X/Y and depth centers are all texel-snapped before comparison.
            // A changed depth row therefore marks a new sampled coverage
            // authority, not ordinary camera motion. The independent receiver
            // check also catches a fast move before its next snap boundary.
            projectionChanged = shadowCascadeProjectionChanged(
                _cascadeVP[cascade], _cascadeProjectionOrigins[cascade],
                _renderedCascadeVP[cascade], _renderedProjectionOrigins[cascade]);
            if (!projectionChanged) {
                projectionChanged = !shadowCascadeReceiverDepthCovered(
                    _renderedCascadeVP[cascade], _renderedProjectionOrigins[cascade],
                    _cascadeReceiverCenters[cascade], _cascadeReceiverRadii[cascade]);
            }
        }
        const uint64_t elapsed = frameIndex - _lastRefreshFrame[cascade];
        const bool cadenceExpired = elapsed >= shadowCascadeMaximumRefreshInterval(cascade);
        const bool casterChanged = casterRevisions[cascade] != _lastCasterRevision[cascade];
        const bool refresh =
            cascade < 2U || projectionChanged || casterChanged || lightChanged || cadenceExpired;
        if (refresh) {
            refreshMask |= 1U << cascade;
            ++_refreshCounts[cascade];
            _renderedCascadeVP[cascade] = _cascadeVP[cascade];
            _renderedProjectionOrigins[cascade] = _cascadeProjectionOrigins[cascade];
            _renderedCascadeUniforms[cascade] = _shadowUniforms.cascades[cascade];
            _lastRefreshFrame[cascade] = frameIndex;
            _lastCasterRevision[cascade] = casterRevisions[cascade];
        } else {
            _cascadeVP[cascade] = _renderedCascadeVP[cascade];
            _cascadeProjectionOrigins[cascade] = _renderedProjectionOrigins[cascade];
            _shadowUniforms.cascades[cascade] = _renderedCascadeUniforms[cascade];
        }
    }
    _lastLightDirection = normalizedLight;
    _hasRenderedCascades = true;
    return refreshMask;
}

bool ShadowMap::cascadeContains(int cascade, const AABB& aabb) const {
    // Project the 8 AABB corners into the cascade's clip space and test the
    // resulting clip AABB against [-1,1]×[-1,1]×[0,1]. The ortho already spans
    // the caster margin, so a chunk overlapping the box is a relevant caster.
    const Mat4& vp = _cascadeVP[cascade];
    float minX = 1e9f, minY = 1e9f, minZ = 1e9f;
    float maxX = -1e9f, maxY = -1e9f, maxZ = -1e9f;
    for (float x : {aabb.min.x, aabb.max.x}) {
        for (float y : {aabb.min.y, aabb.max.y}) {
            for (float z : {aabb.min.z, aabb.max.z}) {
                Vec3 clip = vp.transformVec3({x - _cascadeProjectionOrigins[cascade].x,
                                              y - _cascadeProjectionOrigins[cascade].y,
                                              z - _cascadeProjectionOrigins[cascade].z});
                minX = std::min(minX, clip.x);
                maxX = std::max(maxX, clip.x);
                minY = std::min(minY, clip.y);
                maxY = std::max(maxY, clip.y);
                minZ = std::min(minZ, clip.z);
                maxZ = std::max(maxZ, clip.z);
            }
        }
    }
    return maxX >= -1.f && minX <= 1.f && maxY >= -1.f && minY <= 1.f && maxZ >= 0.f && minZ <= 1.f;
}

bool ShadowMap::entityCasterAffectsCascade(int cascade, const AABB& aabb) const {
    if (cascade < 0 || cascade >= SHADOW_CASCADE_COUNT) {
        return false;
    }
    const simd_float4 range = _shadowUniforms.cascades[cascade].depthRange;
    return range.w >= 0.5F &&
           shadowEntityCasterReachesDepthSlice(aabb, _casterCameraPosition, _casterCameraForward,
                                               _casterLightDirection, range.x, range.y) &&
           cascadeContains(cascade, aabb);
}
