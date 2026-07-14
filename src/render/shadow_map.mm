#import "render/shadow_map.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Cascade tuning. Splits are world distances from the camera; the log/uniform
// blend packs texel density near the player. shadowDistance (passed in) caps
// the last cascade. The caster margin pulls each cascade's near plane back
// toward the light so geometry above the frustum still casts in.
// ---------------------------------------------------------------------------
namespace {
constexpr float NEAR_SPLIT = 0.5f;
constexpr float SPLIT_LAMBDA = 0.7f; // 0 = uniform, 1 = logarithmic
constexpr float CASTER_MARGIN = 120.0f;

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
    if (!_chunkPipeline) {
        RY_LOG_FATAL("Failed to create shadow chunk pipeline state");
    }

    // Shadow casters write depth, standard less-than test.
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = true;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];

    // Hardware PCF: sample_compare against the stored depth, bilinear between
    // the 4 texels for a soft edge at no extra taps.
    auto samplerDesc = [[MTLSamplerDescriptor alloc] init];
    samplerDesc.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDesc.compareFunction = MTLCompareFunctionLess;
    _comparisonSampler = [_device newSamplerStateWithDescriptor:samplerDesc];

    allocateTexture();
}

void ShadowMap::allocateTexture() {
    auto desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType2DArray;
    desc.pixelFormat = PixelFormats::SCENE_DEPTH;
    desc.width = _resolution;
    desc.height = _resolution;
    desc.arrayLength = SHADOW_CASCADE_COUNT;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    _depthTexture = [_device newTextureWithDescriptor:desc];
    if (!_depthTexture) {
        RY_LOG_FATAL("Failed to allocate shadow depth array");
    }
}

void ShadowMap::setResolution(uint32_t resolution) {
    if (resolution == _resolution) {
        return;
    }
    _resolution = resolution;
    allocateTexture(); // ARC keeps the old texture alive for in-flight frames
}

MTLRenderPassDescriptor* ShadowMap::passDescriptor(int cascade) const {
    auto desc = [MTLRenderPassDescriptor renderPassDescriptor];
    desc.depthAttachment.texture = _depthTexture;
    desc.depthAttachment.slice = static_cast<NSUInteger>(cascade);
    desc.depthAttachment.loadAction = MTLLoadActionClear;
    desc.depthAttachment.storeAction = MTLStoreActionStore;
    desc.depthAttachment.clearDepth = 1.0;
    return desc;
}

void ShadowMap::computeCascades(const Vec3& cameraPos, const Vec3& cameraForward,
                                const Vec3& cameraRight, const Vec3& cameraUp, float fovY,
                                float aspect, const Vec3& lightDir, float shadowDistance,
                                float strength) {
    const Vec3 light = lightDir.normalize();

    // Practical split scheme: blend a uniform and a logarithmic split so the
    // near cascade stays tight (crisp contact shadows) and the far one covers
    // the view distance.
    float splits[SHADOW_CASCADE_COUNT + 1];
    splits[0] = NEAR_SPLIT;
    for (int i = 1; i <= SHADOW_CASCADE_COUNT; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(SHADOW_CASCADE_COUNT);
        float uniform = NEAR_SPLIT + (shadowDistance - NEAR_SPLIT) * t;
        float logSplit = NEAR_SPLIT * std::pow(shadowDistance / NEAR_SPLIT, t);
        splits[i] = SPLIT_LAMBDA * logSplit + (1.0f - SPLIT_LAMBDA) * uniform;
    }

    const float tanHalf = std::tan(fovY * 0.5f);
    const Vec3 up = std::fabs(light.y) > 0.99f ? Vec3{0.f, 0.f, 1.f} : Vec3{0.f, 1.f, 0.f};

    for (int c = 0; c < SHADOW_CASCADE_COUNT; ++c) {
        float n = splits[c];
        float f = splits[c + 1];

        // 8 world-space corners of this frustum slice.
        Vec3 corners[8];
        int idx = 0;
        for (float d : {n, f}) {
            float halfH = d * tanHalf;
            float halfW = halfH * aspect;
            Vec3 center = cameraPos + cameraForward * d;
            for (float sx : {-1.f, 1.f}) {
                for (float sy : {-1.f, 1.f}) {
                    corners[idx++] = center + cameraRight * (halfW * sx) + cameraUp * (halfH * sy);
                }
            }
        }

        // Bounding sphere: center = mean, radius = farthest corner. The radius
        // depends only on (n, f, fov, aspect), so it is stable as the camera
        // rotates — the precondition for texel snapping to kill shimmer.
        Vec3 sphereCenter{0.f, 0.f, 0.f};
        for (const Vec3& corner : corners) {
            sphereCenter = sphereCenter + corner;
        }
        sphereCenter = sphereCenter * (1.0f / 8.0f);
        float radius = 0.f;
        for (const Vec3& corner : corners) {
            Vec3 d = corner - sphereCenter;
            radius = std::max(radius, std::sqrt(d.dot(d)));
        }
        radius = std::ceil(radius * 16.0f) / 16.0f; // quantize so it never jitters

        // Light view sits at the light's side of the sphere (lightDir points
        // toward the sun/moon) looking down at its center; ortho spans the
        // sphere, near→far covers depth + the caster margin above it.
        Vec3 eye = sphereCenter + light * (radius + CASTER_MARGIN);
        Mat4 lightView = Mat4::lookAt(eye, sphereCenter, up);
        Mat4 lightProj = Mat4::orthographic(-radius, radius, -radius, radius, 0.0f,
                                            2.0f * radius + CASTER_MARGIN);

        // Texel snap: round a FIXED world reference's shadow-map position to
        // whole texels and fold the fractional shift back into the projection,
        // so the grid clicks to world-aligned increments instead of swimming
        // as the camera moves. (Snapping the sphere center would be a no-op —
        // the ortho is centered on it, so it always projects to clip 0.)
        Mat4 vp = lightProj * lightView;
        Vec4 originClip = vp.transformVec4({0.f, 0.f, 0.f, 1.f});
        float half = static_cast<float>(_resolution) * 0.5f;
        float snappedX = std::round(originClip.x * half) / half;
        float snappedY = std::round(originClip.y * half) / half;
        lightProj(0, 3) += snappedX - originClip.x;
        lightProj(1, 3) += snappedY - originClip.y;

        _cascadeVP[c] = lightProj * lightView;
        _shadowUniforms.cascadeViewProj[c] = toSimd(_cascadeVP[c]);
        reinterpret_cast<float*>(&_shadowUniforms.cascadeSplitDist)[c] = f;
    }

    // texelWorldSize of cascade 0 seeds the penumbra + normal-bias scale.
    float texel0 = 0.f;
    {
        // radius0 recompute cheaply from the split.
        float d = splits[1];
        float halfH = d * tanHalf;
        float halfW = halfH * aspect;
        float r = std::sqrt(halfW * halfW + halfH * halfH + d * d * 0.25f);
        texel0 = 2.0f * r / static_cast<float>(_resolution);
    }
    _shadowUniforms.shadowParams =
        simd_make_float4(1.5f,                    // penumbra texel radius
                         texel0 * 2.0f,           // depth/normal bias in world units
                         strength,                // 0 disables sampling
                         shadowDistance * 0.85f); // fade start
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
                Vec3 clip = vp.transformVec3({x, y, z});
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
