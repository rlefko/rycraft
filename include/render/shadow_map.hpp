#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

enum class ShadowTextureGroup : uint8_t {
    NEAR,
    FAR,
    HORIZON,
};

struct ShadowCascadeConfiguration {
    float nearDepth;
    float farDepth;
    uint32_t resolution;
    ShadowTextureGroup textureGroup;
    uint32_t textureSlice;
};

// Quality is the existing GraphicsSettings value: 1 is Medium and 2 is High.
// Keeping the complete split, target, and slice table here gives tests and the
// render loop one source of truth.
constexpr ShadowCascadeConfiguration shadowCascadeConfiguration(uint32_t quality,
                                                                uint32_t cascade) {
    const bool high = quality >= 2u;
    switch (cascade) {
        case 0u:
            return {0.5f, high ? 48.0f : 40.0f, high ? 4096u : 2048u, ShadowTextureGroup::NEAR, 0u};
        case 1u:
            return {high ? 48.0f : 40.0f, high ? 160.0f : 128.0f, high ? 4096u : 2048u,
                    ShadowTextureGroup::NEAR, 1u};
        case 2u:
            return {high ? 160.0f : 128.0f, high ? 512.0f : 384.0f, high ? 2048u : 1024u,
                    ShadowTextureGroup::FAR, 0u};
        case 3u:
            return {high ? 512.0f : 384.0f, high ? 1536.0f : 768.0f, high ? 2048u : 1024u,
                    ShadowTextureGroup::FAR, 1u};
        default:
            return {high ? 1536.0f : 768.0f, SHADOW_HORIZON_DISTANCE, high ? 2048u : 1024u,
                    ShadowTextureGroup::HORIZON, 0u};
    }
}

constexpr uint64_t shadowMapMemoryBytes(uint32_t quality) noexcept {
    constexpr uint64_t DEPTH_BYTES = 4U;
    const uint64_t nearEdge = shadowCascadeConfiguration(quality, 0U).resolution;
    const uint64_t farEdge = shadowCascadeConfiguration(quality, 2U).resolution;
    const uint64_t horizonEdge =
        shadowCascadeConfiguration(quality, SHADOW_HORIZON_CASCADE_INDEX).resolution;
    return (nearEdge * nearEdge * 2U + farEdge * farEdge * 2U + horizonEdge * horizonEdge) *
           DEPTH_BYTES;
}

constexpr float shadowCascadeBlendStart(const ShadowCascadeConfiguration& configuration) {
    return configuration.farDepth -
           (configuration.farDepth - configuration.nearDepth) * SHADOW_CASCADE_BLEND_FRACTION;
}

// The mean of the eight frustum corners lies halfway between the split
// planes. Its farthest corner is always on the far plane, so this analytic
// radius is equivalent to measuring all corners without subtracting
// six-digit world coordinates. The 1/16-block quantization retains a stable
// projection scale as the camera rotates and moves through large coordinates.
inline float shadowCascadeBoundingRadius(float nearDepth, float farDepth, float tanHalfFov,
                                         float aspect) noexcept {
    const double halfWidth = static_cast<double>(farDepth) * tanHalfFov * aspect;
    const double halfHeight = static_cast<double>(farDepth) * tanHalfFov;
    const double halfDepth = 0.5 * static_cast<double>(farDepth - nearDepth);
    const double radius =
        std::sqrt(halfWidth * halfWidth + halfHeight * halfHeight + halfDepth * halfDepth);
    return static_cast<float>(std::ceil(radius * 16.0) / 16.0);
}

inline double shadowPreciseDot(const Vec3& left, const Vec3& right) noexcept {
    return static_cast<double>(left.x) * right.x + static_cast<double>(left.y) * right.y +
           static_cast<double>(left.z) * right.z;
}

inline double shadowSnappedLightCoordinate(double coordinate, double texelWorldSize) noexcept {
    return std::round(coordinate / texelWorldSize) * texelWorldSize;
}

inline float shadowProjectionAnchorCoordinate(float coordinate) noexcept {
    constexpr double ANCHOR_EDGE = 256.0;
    return static_cast<float>(std::floor(static_cast<double>(coordinate) / ANCHOR_EDGE) *
                              ANCHOR_EDGE);
}

inline Vec3 shadowProjectionOrigin(const Vec3& cameraPosition) noexcept {
    return {shadowProjectionAnchorCoordinate(cameraPosition.x),
            shadowProjectionAnchorCoordinate(cameraPosition.y),
            shadowProjectionAnchorCoordinate(cameraPosition.z)};
}

constexpr uint32_t shadowCascadeMaximumRefreshInterval(uint32_t cascade) {
    return cascade < 2U ? 1U : cascade == 2U ? 2U : cascade == 3U ? 4U : 8U;
}

// Near maps refresh every frame and can follow animated foliage exactly.
// Deferred maps deliberately keep foliage casters static, avoiding a visible
// 2/4/8-frame jump when their retained depth texture is sampled between refreshes.
constexpr bool shadowCascadeUsesAnimatedFoliage(uint32_t cascade) {
    return shadowCascadeMaximumRefreshInterval(cascade) == 1U;
}

inline FoliageWindUniforms shadowFoliageWindForCascade(FoliageWindUniforms wind,
                                                       uint32_t cascade) noexcept {
    if (!shadowCascadeUsesAnimatedFoliage(cascade)) {
        wind.strength = 0.0F;
    }
    return wind;
}

// The receiver sphere is tight in light-space X/Y, but a deferred map also
// needs a small guard at both depth ends. It covers sub-texel depth motion and
// the normal offset until the snapped depth center advances, without reducing
// cascades two through four to a per-frame refresh path.
inline float shadowCascadeReceiverDepthGuard(float cascadeRadius, float casterMargin,
                                             uint32_t resolution, float normalBias) noexcept {
    const float baseDepthRange = 2.0F * cascadeRadius + casterMargin;
    const float baseDepthTexel = baseDepthRange / static_cast<float>(std::max(resolution, 1U));
    return std::max(baseDepthTexel * 4.0F, normalBias + baseDepthTexel * 2.0F);
}

inline float shadowCascadeDepthRange(float cascadeRadius, float casterMargin,
                                     float receiverDepthGuard) noexcept {
    return 2.0F * cascadeRadius + casterMargin + 2.0F * receiverDepthGuard;
}

inline float shadowCascadeDepthTexelWorldSize(float cascadeRadius, float casterMargin,
                                              float receiverDepthGuard,
                                              uint32_t resolution) noexcept {
    return shadowCascadeDepthRange(cascadeRadius, casterMargin, receiverDepthGuard) /
           static_cast<float>(std::max(resolution, 1U));
}

// Shadow matrices consume positions relative to a nearby origin. Rows zero
// through two define the sampled orthographic X, Y, and depth coverage; row
// three stays [0, 0, 0, 1] for every valid map. Depth is snapped before this
// comparison, so a true row-two change means the retained map is no longer a
// compatible authority rather than ordinary sub-texel camera motion.
inline bool shadowCascadeProjectionChanged(const Mat4& candidateMatrix, const Vec3& candidateOrigin,
                                           const Mat4& renderedMatrix, const Vec3& renderedOrigin,
                                           float epsilon = 1.0e-6F) noexcept {
    if (std::abs(candidateOrigin.x - renderedOrigin.x) > epsilon ||
        std::abs(candidateOrigin.y - renderedOrigin.y) > epsilon ||
        std::abs(candidateOrigin.z - renderedOrigin.z) > epsilon) {
        return true;
    }
    for (size_t column = 0; column < 4; ++column) {
        for (size_t row = 0; row < 3; ++row) {
            if (std::abs(candidateMatrix(row, column) - renderedMatrix(row, column)) > epsilon) {
                return true;
            }
        }
    }
    return false;
}

// An independently checked receiver guard catches a teleport or any other
// displacement that crosses the retained map's light-space far face before a
// snap update. Returning false forces a depth refresh instead of letting the
// shader fall back to exterior light in a moving ring.
inline bool shadowCascadeReceiverDepthCovered(const Mat4& sampledMatrix, const Vec3& sampledOrigin,
                                              const Vec3& receiverCenter,
                                              float receiverRadius) noexcept {
    if (!std::isfinite(receiverRadius) || receiverRadius < 0.0F) {
        return false;
    }
    const Vec3 relativeCenter = receiverCenter - sampledOrigin;
    const Vec4 clip = sampledMatrix.transformVec4({relativeCenter, 1.0F});
    if (!std::isfinite(clip.z) || !std::isfinite(clip.w) || std::abs(clip.w) < 1.0e-6F) {
        return false;
    }
    const float centerDepth = clip.z / clip.w;
    const float depthScale = std::sqrt(sampledMatrix(2, 0) * sampledMatrix(2, 0) +
                                       sampledMatrix(2, 1) * sampledMatrix(2, 1) +
                                       sampledMatrix(2, 2) * sampledMatrix(2, 2)) /
                             std::abs(clip.w);
    const float depthRadius = receiverRadius * depthScale;
    return std::isfinite(centerDepth) && std::isfinite(depthRadius) &&
           centerDepth - depthRadius >= 0.0F && centerDepth + depthRadius <= 1.0F;
}

inline float shadowCasterMargin(float cascadeRadius, float lightElevation, bool horizonCascade) {
    constexpr float minimum = 120.0F;
    constexpr float representativeCasterHeight = 96.0F;
    const float elevation = std::max(std::abs(lightElevation), 0.125F);
    const float lowAngleReach = representativeCasterHeight / elevation;
    const float radiusReach = horizonCascade ? cascadeRadius * 0.125F : minimum;
    const float maximum = horizonCascade ? 1024.0F : 768.0F;
    return std::clamp(std::max(lowAngleReach, radiusReach), minimum, maximum);
}

// A compact caster can affect a receiver only along the ray travelling away
// from the directional light. Bound that extrusion where it reaches the
// world's vertical limits, then compare its camera-forward depth interval to
// the cascade's receiver slice. This rejects nearby animals from coarse
// cascades under a high light while retaining their genuinely long low-sun
// shadows. The clip-volume test remains a second, independent requirement.
inline bool shadowEntityCasterReachesDepthSlice(const AABB& caster, const Vec3& cameraPosition,
                                                const Vec3& cameraForward,
                                                const Vec3& lightDirection, float receiverNear,
                                                float receiverFar) noexcept {
    if (receiverFar < receiverNear) {
        return false;
    }
    const Vec3 forward = cameraForward.normalize();
    const Vec3 light = lightDirection.normalize();
    if (forward.lengthSq() <= 0.0F || light.lengthSq() <= 0.0F) {
        return false;
    }

    float casterNear = std::numeric_limits<float>::infinity();
    float casterFar = -std::numeric_limits<float>::infinity();
    for (float x : {caster.min.x, caster.max.x}) {
        for (float y : {caster.min.y, caster.max.y}) {
            for (float z : {caster.min.z, caster.max.z}) {
                const float depth = (Vec3{x, y, z} - cameraPosition).dot(forward);
                casterNear = std::min(casterNear, depth);
                casterFar = std::max(casterFar, depth);
            }
        }
    }

    const float lightAlongView = light.dot(forward);
    const float shadowDepthRate = -lightAlongView;
    float maximumDistance = std::numeric_limits<float>::infinity();
    if (light.y > 1.0e-5F) {
        maximumDistance =
            std::max((caster.max.y - static_cast<float>(WORLD_MIN_Y)) / light.y, 0.0F);
    } else if (light.y < -1.0e-5F) {
        maximumDistance =
            std::max((static_cast<float>(WORLD_MAX_Y) - caster.min.y) / -light.y, 0.0F);
    }

    float shadowNear = casterNear;
    float shadowFar = casterFar;
    if (std::isfinite(maximumDistance)) {
        const float depthReach = shadowDepthRate * maximumDistance;
        shadowNear += std::min(depthReach, 0.0F);
        shadowFar += std::max(depthReach, 0.0F);
    } else if (shadowDepthRate < 0.0F) {
        shadowNear = -std::numeric_limits<float>::infinity();
    } else if (shadowDepthRate > 0.0F) {
        shadowFar = std::numeric_limits<float>::infinity();
    }
    return shadowFar >= receiverNear && shadowNear <= receiverFar;
}

// ---------------------------------------------------------------------------
// ShadowMap, cascaded sun/moon shadow maps.
//
// Owns two two-slice depth arrays plus one horizon depth texture, the
// cutout-aware chunk pipeline, and the cascade math.
// Each frame computeCascades() fits a stable, texel-snapped ortho box around
// each slice of the camera frustum; the caller then encodes one depth pass per
// cascade, and reads shadowUniforms() into the scene pass for PCF sampling.
//
// RenderPipeline drives the geometry (it owns the mega-buffer); this class
// owns the targets, pipelines, and math so render_pipeline.mm stays smaller.
// ---------------------------------------------------------------------------
class ShadowMap {
public:
    // vertexDescriptor is the shared chunk vertex layout (owned by
    // RenderPipeline) so the shadow pass reads the same 16-byte vertices.
    ShadowMap(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
              MTLVertexDescriptor* vertexDescriptor);
    ~ShadowMap();

    // Allocate the Medium or High texture groups. Called on a quality change.
    void setQuality(uint32_t quality);
    uint32_t quality() const { return _quality; }
    uint32_t resolution(int cascade) const;

    // Fit the cascades to the current camera + light. lightDir points FROM the
    // scene TO the light (sun by day, moon by night). The quality split table
    // caps the detailed and horizon ranges; strength scales the shadow term.
    void computeCascades(const Vec3& cameraPos, const Vec3& cameraForward, float fovY, float aspect,
                         const Vec3& lightDir, float strength);

    // Selects the cascades whose new snapped projection or caster content
    // must be rendered. Skipped records retain their last rendered matrix so
    // sampling can never get ahead of the depth texture it projects through.
    uint32_t selectRefreshMask(uint64_t frameIndex,
                               const std::array<uint64_t, SHADOW_CASCADE_COUNT>& casterRevisions,
                               const Vec3& lightDirection);
    uint64_t lastRefreshFrame(uint32_t cascade) const { return _lastRefreshFrame[cascade]; }
    uint64_t refreshCount(uint32_t cascade) const { return _refreshCounts[cascade]; }

    // Per-cascade light view-projection (for the depth pass) and the packed
    // sampling block (for the scene pass).
    const Mat4& cascadeViewProj(int cascade) const { return _cascadeVP[cascade]; }
    const Vec3& cascadeProjectionOrigin(int cascade) const {
        return _cascadeProjectionOrigins[cascade];
    }
    const ShadowUniforms& shadowUniforms() const { return _shadowUniforms; }

    id<MTLTexture> nearDepthTexture() const { return _nearDepthTexture; }
    id<MTLTexture> farDepthTexture() const { return _farDepthTexture; }
    id<MTLTexture> horizonDepthTexture() const { return _horizonDepthTexture; }
    id<MTLDepthStencilState> depthState() const { return _depthState; }
    id<MTLRenderPipelineState> chunkPipeline() const { return _chunkPipeline; }
    id<MTLRenderPipelineState> entityPipeline() const { return _entityPipeline; }
    id<MTLSamplerState> comparisonSampler() const { return _comparisonSampler; }

    // A render pass descriptor targeting one cascade slice (depth-only clear).
    MTLRenderPassDescriptor* passDescriptor(int cascade) const;

    // World-space AABB test against a cascade's ortho volume, extruded toward
    // the light so casters behind the frustum still draw. Reused for culling.
    bool cascadeContains(int cascade, const struct AABB& aabb) const;

    // Entity revisions and entity shadow draws must share this exact
    // authority or a skipped draw can still force a coarse cascade refresh.
    bool entityCasterAffectsCascade(int cascade, const struct AABB& aabb) const;

private:
    id<MTLDevice> _device;
    id<MTLTexture> _nearDepthTexture{};    // Depth32Float, two-slice 2D array
    id<MTLTexture> _farDepthTexture{};     // Depth32Float, two-slice 2D array
    id<MTLTexture> _horizonDepthTexture{}; // Depth32Float, one 2D texture
    id<MTLDepthStencilState> _depthState;
    id<MTLRenderPipelineState> _chunkPipeline;
    id<MTLRenderPipelineState> _entityPipeline;
    id<MTLSamplerState> _comparisonSampler;
    uint32_t _quality = 1;

    Mat4 _cascadeVP[SHADOW_CASCADE_COUNT];
    Vec3 _cascadeProjectionOrigins[SHADOW_CASCADE_COUNT];
    Vec3 _cascadeReceiverCenters[SHADOW_CASCADE_COUNT];
    float _cascadeReceiverRadii[SHADOW_CASCADE_COUNT]{};
    ShadowUniforms _shadowUniforms{};
    Mat4 _renderedCascadeVP[SHADOW_CASCADE_COUNT];
    Vec3 _renderedProjectionOrigins[SHADOW_CASCADE_COUNT];
    ShadowCascadeUniforms _renderedCascadeUniforms[SHADOW_CASCADE_COUNT]{};
    std::array<uint64_t, SHADOW_CASCADE_COUNT> _lastRefreshFrame{};
    std::array<uint64_t, SHADOW_CASCADE_COUNT> _refreshCounts{};
    std::array<uint64_t, SHADOW_CASCADE_COUNT> _lastCasterRevision{};
    Vec3 _lastLightDirection{};
    Vec3 _casterCameraPosition{};
    Vec3 _casterCameraForward{};
    Vec3 _casterLightDirection{};
    bool _hasRenderedCascades = false;

    void allocateTextures();
};
