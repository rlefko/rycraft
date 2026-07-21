#pragma once

#include <metal_stdlib>
#include <render/shader_types.hpp>

// Shared directional-shadow sampling for opaque surfaces and volumetric
// lighting. Cascade selection, projection validation, overlap blending, and
// texture-group routing live here so a shaft cannot switch cascades at a
// different camera ring than the terrain beneath it.

struct ShadowProjection {
    metal::float2 uv;
    float depth;
    unsigned int valid;
};

constant metal::float2 SHADOW_POISSON_DISK[16] = {
    metal::float2(-0.613f, 0.617f),  metal::float2(0.170f, -0.040f),
    metal::float2(-0.299f, -0.791f), metal::float2(0.645f, 0.493f),
    metal::float2(-0.651f, -0.378f), metal::float2(0.918f, -0.126f),
    metal::float2(0.344f, 0.294f),   metal::float2(-0.108f, 0.987f),
    metal::float2(-0.920f, 0.078f),  metal::float2(0.542f, -0.782f),
    metal::float2(0.098f, -0.967f),  metal::float2(-0.379f, 0.278f),
    metal::float2(0.895f, 0.373f),   metal::float2(-0.759f, -0.727f),
    metal::float2(0.269f, 0.766f),   metal::float2(-0.288f, -0.303f),
};

constant metal::float2 SHADOW_FAR_TAPS[9] = {
    metal::float2(0.0f, 0.0f),      metal::float2(1.0f, 0.0f),      metal::float2(-1.0f, 0.0f),
    metal::float2(0.0f, 1.0f),      metal::float2(0.0f, -1.0f),     metal::float2(0.707f, 0.707f),
    metal::float2(-0.707f, 0.707f), metal::float2(0.707f, -0.707f), metal::float2(-0.707f, -0.707f),
};

static inline float shadowTextureTexel(unsigned int cascade, metal::depth2d_array<float> nearShadow,
                                       metal::depth2d_array<float> farShadow,
                                       metal::depth2d<float> horizonShadow) {
    if (cascade < 2u) {
        return 1.0f / float(nearShadow.get_width());
    }
    if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
        return 1.0f / float(farShadow.get_width());
    }
    return 1.0f / float(horizonShadow.get_width());
}

static inline float shadowRawDepth(unsigned int cascade, metal::float2 uv,
                                   metal::depth2d_array<float> nearShadow,
                                   metal::depth2d_array<float> farShadow,
                                   metal::depth2d<float> horizonShadow) {
    constexpr metal::sampler pointSampler(metal::mag_filter::nearest, metal::min_filter::nearest,
                                          metal::address::clamp_to_edge);
    if (cascade < 2u) {
        return nearShadow.sample(pointSampler, uv, cascade);
    }
    if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
        return farShadow.sample(pointSampler, uv, cascade - 2u);
    }
    return horizonShadow.sample(pointSampler, uv);
}

static inline float shadowCompare(unsigned int cascade, metal::float2 uv, float depth,
                                  metal::depth2d_array<float> nearShadow,
                                  metal::depth2d_array<float> farShadow,
                                  metal::depth2d<float> horizonShadow,
                                  metal::sampler comparisonSampler) {
    if (cascade < 2u) {
        return nearShadow.sample_compare(comparisonSampler, uv, cascade, depth);
    }
    if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
        return farShadow.sample_compare(comparisonSampler, uv, cascade - 2u, depth);
    }
    return horizonShadow.sample_compare(comparisonSampler, uv, depth);
}

static inline ShadowProjection projectShadowPosition(metal::float3 worldPosition,
                                                     metal::float3 normal, unsigned int cascade,
                                                     constant ShadowUniforms& shadow) {
    constant ShadowCascadeUniforms& record = shadow.cascades[cascade];
    const metal::float3 relativePosition =
        (worldPosition - record.projectionOrigin.xyz) + normal * record.samplingParams.y;
    const metal::float4 clip = record.lightViewProj * metal::float4(relativePosition, 1.0f);
    if (metal::abs(clip.w) < 1e-6f || record.depthRange.w < 0.5f) {
        return {metal::float2(0.0f), 0.0f, 0u};
    }
    const metal::float3 ndc = clip.xyz / clip.w;
    metal::float2 uv = ndc.xy * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    const bool valid = metal::all(uv >= metal::float2(0.0f)) &&
                       metal::all(uv <= metal::float2(1.0f)) && ndc.z >= 0.0f && ndc.z <= 1.0f;
    return {uv, ndc.z - record.samplingParams.w, valid ? 1u : 0u};
}

static inline metal::float2 shadowRotation(float angle, metal::float2 value) {
    const metal::float2 axis = metal::float2(metal::cos(angle), metal::sin(angle));
    return metal::float2(axis.x * value.x - axis.y * value.y, axis.y * value.x + axis.x * value.y);
}

static inline float shadowWorldStableRotation(metal::float3 worldPosition, unsigned int cascade) {
    // A screen-space rotation crawls over a stationary shadow whenever the
    // camera moves. Quantize a bounded world-space cell instead. The wrapping
    // retains hash precision around large coordinates and the cascade offset
    // prevents aligned kernels across an overlap band.
    metal::float2 cell = metal::floor(worldPosition.xz * 2.0f);
    cell -= metal::floor(cell * (1.0f / 4096.0f)) * 4096.0f;
    cell += metal::float2(float(cascade) * 37.0f, float(cascade) * 73.0f);
    return interleavedGradientNoise(cell) * 6.2831853f;
}

static inline float
sampleNearShadow(unsigned int cascade, ShadowProjection projection, metal::float3 worldPosition,
                 metal::depth2d_array<float> nearShadow, metal::depth2d_array<float> farShadow,
                 metal::depth2d<float> horizonShadow, metal::sampler comparisonSampler,
                 constant ShadowUniforms& shadow) {
    const float texel = shadowTextureTexel(cascade, nearShadow, farShadow, horizonShadow);
    const float baseRadius = shadow.cascades[cascade].samplingParams.z;
    const float rotation = shadowWorldStableRotation(worldPosition, cascade);
    float blockerSum = 0.0f;
    float blockerCount = 0.0f;
    for (int tap = 0; tap < 16; ++tap) {
        const metal::float2 offset = shadowRotation(rotation, SHADOW_POISSON_DISK[tap]);
        const float blockerDepth =
            shadowRawDepth(cascade, projection.uv + offset * (baseRadius * 3.0f * texel),
                           nearShadow, farShadow, horizonShadow);
        if (blockerDepth < projection.depth) {
            blockerSum += blockerDepth;
            blockerCount += 1.0f;
        }
    }
    if (blockerCount < 0.5f) {
        return 1.0f;
    }

    const float averageBlocker = blockerSum / blockerCount;
    const float penumbra = metal::saturate((projection.depth - averageBlocker) /
                                           metal::max(averageBlocker, 1e-4f) * 40.0f);
    const float filterRadius = metal::mix(baseRadius, baseRadius * 4.0f, penumbra) * texel;
    float visibility = 0.0f;
    for (int tap = 0; tap < 16; ++tap) {
        const metal::float2 offset = shadowRotation(rotation, SHADOW_POISSON_DISK[tap]);
        visibility +=
            shadowCompare(cascade, projection.uv + offset * filterRadius, projection.depth,
                          nearShadow, farShadow, horizonShadow, comparisonSampler);
    }
    return visibility * (1.0f / 16.0f);
}

static inline float
sampleFarShadow(unsigned int cascade, ShadowProjection projection, metal::float3 worldPosition,
                metal::depth2d_array<float> nearShadow, metal::depth2d_array<float> farShadow,
                metal::depth2d<float> horizonShadow, metal::sampler comparisonSampler,
                constant ShadowUniforms& shadow) {
    const float texel = shadowTextureTexel(cascade, nearShadow, farShadow, horizonShadow);
    const float radius = shadow.cascades[cascade].samplingParams.z * texel;
    const float rotation = shadowWorldStableRotation(worldPosition, cascade);
    float visibility = 0.0f;
    for (int tap = 0; tap < 9; ++tap) {
        const metal::float2 offset = shadowRotation(rotation, SHADOW_FAR_TAPS[tap]);
        visibility += shadowCompare(cascade, projection.uv + offset * radius, projection.depth,
                                    nearShadow, farShadow, horizonShadow, comparisonSampler);
    }
    return visibility * (1.0f / 9.0f);
}

static inline float sampleHorizonShadow(unsigned int cascade, ShadowProjection projection,
                                        metal::depth2d_array<float> nearShadow,
                                        metal::depth2d_array<float> farShadow,
                                        metal::depth2d<float> horizonShadow,
                                        metal::sampler comparisonSampler,
                                        constant ShadowUniforms& shadow) {
    const float texel = shadowTextureTexel(cascade, nearShadow, farShadow, horizonShadow) *
                        shadow.cascades[cascade].samplingParams.z;
    const metal::float2 offsets[4] = {
        metal::float2(-0.5f, -0.5f),
        metal::float2(0.5f, -0.5f),
        metal::float2(-0.5f, 0.5f),
        metal::float2(0.5f, 0.5f),
    };
    float visibility = 0.0f;
    for (int tap = 0; tap < 4; ++tap) {
        visibility += shadowCompare(cascade, projection.uv + offsets[tap] * texel, projection.depth,
                                    nearShadow, farShadow, horizonShadow, comparisonSampler);
    }
    return visibility * 0.25f;
}

static inline float sampleShadowCascadeSurface(metal::float3 worldPosition, metal::float3 normal,
                                               unsigned int cascade, float exteriorVisibility,
                                               metal::depth2d_array<float> nearShadow,
                                               metal::depth2d_array<float> farShadow,
                                               metal::depth2d<float> horizonShadow,
                                               metal::sampler comparisonSampler,
                                               constant ShadowUniforms& shadow) {
    const ShadowProjection projection =
        projectShadowPosition(worldPosition, normal, cascade, shadow);
    if (projection.valid == 0u) {
        return exteriorVisibility;
    }
    if (cascade < 2u) {
        return sampleNearShadow(cascade, projection, worldPosition, nearShadow, farShadow,
                                horizonShadow, comparisonSampler, shadow);
    }
    if (cascade < SHADOW_HORIZON_CASCADE_INDEX) {
        return sampleFarShadow(cascade, projection, worldPosition, nearShadow, farShadow,
                               horizonShadow, comparisonSampler, shadow);
    }
    return sampleHorizonShadow(cascade, projection, nearShadow, farShadow, horizonShadow,
                               comparisonSampler, shadow);
}

static inline float
sampleShadowVisibility(metal::float3 worldPosition, metal::float3 normal, float exteriorVisibility,
                       metal::depth2d_array<float> nearShadow,
                       metal::depth2d_array<float> farShadow, metal::depth2d<float> horizonShadow,
                       metal::sampler comparisonSampler, constant ShadowUniforms& shadow) {
    const float strength = shadow.cameraPositionAndStrength.w;
    if (strength <= 0.001f) {
        return metal::saturate(exteriorVisibility);
    }
    exteriorVisibility = metal::saturate(exteriorVisibility);
    const float viewDepth = shadowViewDepth(worldPosition, shadow.cameraPositionAndStrength.xyz,
                                            shadow.cameraForwardAndPadding.xyz);
    const ShadowCascadeSelection selection = shadowCascadeSelection(viewDepth, shadow);
    if (selection.covered == 0u) {
        return exteriorVisibility;
    }

    const ShadowProjection primaryProjection =
        projectShadowPosition(worldPosition, normal, selection.primary, shadow);
    if (primaryProjection.valid == 0u) {
        return exteriorVisibility;
    }

    const float primary =
        sampleShadowCascadeSurface(worldPosition, normal, selection.primary, exteriorVisibility,
                                   nearShadow, farShadow, horizonShadow, comparisonSampler, shadow);
    float visibility = primary;
    if (selection.secondary != selection.primary) {
        const float secondary = sampleShadowCascadeSurface(
            worldPosition, normal, selection.secondary, exteriorVisibility, nearShadow, farShadow,
            horizonShadow, comparisonSampler, shadow);
        visibility = metal::mix(primary, secondary, selection.secondaryWeight);
    }
    visibility = metal::mix(visibility, exteriorVisibility, selection.exteriorWeight);
    return shadowVisibilityWithStrength(visibility, strength);
}

static inline float
sampleShadowCascadeFast(metal::float3 worldPosition, unsigned int cascade, float exteriorVisibility,
                        metal::depth2d_array<float> nearShadow,
                        metal::depth2d_array<float> farShadow, metal::depth2d<float> horizonShadow,
                        metal::sampler comparisonSampler, constant ShadowUniforms& shadow) {
    const ShadowProjection projection =
        projectShadowPosition(worldPosition, metal::float3(0.0f), cascade, shadow);
    if (projection.valid == 0u) {
        return exteriorVisibility;
    }
    if (cascade == SHADOW_HORIZON_CASCADE_INDEX) {
        return sampleHorizonShadow(cascade, projection, nearShadow, farShadow, horizonShadow,
                                   comparisonSampler, shadow);
    }
    return shadowCompare(cascade, projection.uv, projection.depth, nearShadow, farShadow,
                         horizonShadow, comparisonSampler);
}

static inline float sampleShadowVisibilityFast(
    metal::float3 worldPosition, float exteriorVisibility, metal::depth2d_array<float> nearShadow,
    metal::depth2d_array<float> farShadow, metal::depth2d<float> horizonShadow,
    metal::sampler comparisonSampler, constant ShadowUniforms& shadow) {
    const float strength = shadow.cameraPositionAndStrength.w;
    if (strength <= 0.001f) {
        return metal::saturate(exteriorVisibility);
    }
    exteriorVisibility = metal::saturate(exteriorVisibility);
    const float viewDepth = shadowViewDepth(worldPosition, shadow.cameraPositionAndStrength.xyz,
                                            shadow.cameraForwardAndPadding.xyz);
    const ShadowCascadeSelection selection = shadowCascadeSelection(viewDepth, shadow);
    if (selection.covered == 0u) {
        return exteriorVisibility;
    }
    const ShadowProjection primaryProjection =
        projectShadowPosition(worldPosition, metal::float3(0.0f), selection.primary, shadow);
    if (primaryProjection.valid == 0u) {
        return exteriorVisibility;
    }
    const float primary =
        sampleShadowCascadeFast(worldPosition, selection.primary, exteriorVisibility, nearShadow,
                                farShadow, horizonShadow, comparisonSampler, shadow);
    float visibility = primary;
    if (selection.secondary != selection.primary) {
        const float secondary = sampleShadowCascadeFast(worldPosition, selection.secondary,
                                                        exteriorVisibility, nearShadow, farShadow,
                                                        horizonShadow, comparisonSampler, shadow);
        visibility = metal::mix(primary, secondary, selection.secondaryWeight);
    }
    visibility = metal::mix(visibility, exteriorVisibility, selection.exteriorWeight);
    return shadowVisibilityWithStrength(visibility, strength);
}
