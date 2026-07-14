#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Volumetric light — ray-marched sun/moon shafts (the Extreme-VL signature).
//
// A half-resolution pass reconstructs each pixel's world ray from the resolved
// scene depth, marches camera→scene sampling the shadow cascades at each step,
// and accumulates in-scatter weighted by a Henyey-Greenstein phase (bright
// halo toward the light). A per-pixel dithered start offset breaks the low
// step count into a smooth haze. The result composites additively onto the HDR
// scene, so shafts glow through gaps in foliage and terrain. Underwater the
// march also absorbs with depth and tints, replacing the old fake banded
// god-ray overlay.
// ---------------------------------------------------------------------------

struct VolVertexOut {
    float4 clipPosition [[position]];
    float2 vUV;
};

// Fullscreen triangle with V flipped for sampling rendered textures.
static VolVertexOut fullscreenTriangle(uint vertexID) {
    const float2 pos[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    VolVertexOut out;
    out.clipPosition = float4(pos[vertexID], 0.0f, 1.0f);
    out.vUV = float2(pos[vertexID].x * 0.5f + 0.5f, 0.5f - pos[vertexID].y * 0.5f);
    return out;
}

vertex VolVertexOut volumetricVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

// Henyey-Greenstein phase — forward scatter toward the light for g > 0.
static float henyeyGreenstein(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * 3.14159265f * pow(max(denom, 1e-4f), 1.5f));
}

// Sun visibility at a world point: pick the cascade by camera distance, project
// into it, and hardware-PCF a single comparison tap (a shaft only needs a
// coarse in/out test per step).
static float marchVisibility(float3 worldPos, float dist, depth2d_array<float> shadowMap,
                             sampler shadowSampler, constant ShadowUniforms& shadow) {
    int cascade = SHADOW_CASCADE_COUNT - 1;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        if (dist < shadow.cascadeSplitDist[i]) {
            cascade = i;
            break;
        }
    }
    float4 clip = shadow.cascadeViewProj[cascade] * float4(worldPos, 1.0f);
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f || ndc.z > 1.0f) {
        return 1.0f; // outside the shadow map → assume lit
    }
    return shadowMap.sample_compare(shadowSampler, uv, cascade, ndc.z - 0.002f);
}

fragment float4 volumetricFragment(VolVertexOut in [[stage_in]],
                                   depth2d<float> sceneDepth [[texture(0)]],
                                   depth2d_array<float> shadowMap [[texture(1)]],
                                   sampler shadowSampler [[sampler(1)]],
                                   constant VolumetricUniforms& vol [[buffer(0)]],
                                   constant ShadowUniforms& shadow [[buffer(1)]]) {
    constexpr sampler depthSampler(mag_filter::linear, min_filter::linear,
                                   address::clamp_to_edge);

    // Reconstruct the world position of the opaque scene behind this pixel.
    float depth = sceneDepth.sample(depthSampler, in.vUV);
    float4 clip = float4(in.vUV.x * 2.0f - 1.0f, 1.0f - in.vUV.y * 2.0f, depth, 1.0f);
    float4 worldH = vol.invViewProjection * clip;
    float3 sceneWorld = worldH.xyz / worldH.w;

    float3 toScene = sceneWorld - vol.cameraPosition;
    float sceneDist = length(toScene);
    float rayLen = min(sceneDist, vol.maxDistance);
    float3 rayDir = toScene / max(sceneDist, 1e-4f);

    int steps = int(vol.stepCount);
    float stepSize = rayLen / float(steps);

    // Dithered start offset (interleaved gradient noise on the screen pixel +
    // frame index) so the coarse march reads as smooth haze, not slabs.
    float2 px = in.clipPosition.xy + float2(vol.frameIndex % 8u) * 5.588f;
    float dither = fract(52.9829189f * fract(dot(px, float2(0.06711056f, 0.00583715f))));

    float3 sunDir = normalize(vol.sunDirection);
    float cosTheta = dot(rayDir, sunDir);
    float phase = henyeyGreenstein(cosTheta, vol.anisotropy);

    float inscatter = 0.0f;
    float t = dither * stepSize;
    for (int i = 0; i < steps; ++i) {
        float3 p = vol.cameraPosition + rayDir * t;
        float vis = marchVisibility(p, t, shadowMap, shadowSampler, shadow);
        float atten = 1.0f;
        if (vol.underwater > 0.5f) {
            atten = exp(-t * 0.06f); // water absorbs the shaft with depth
        }
        inscatter += vis * atten * stepSize;
        t += stepSize;
    }
    inscatter *= vol.density * phase;

    float3 color = vol.sunColor * inscatter;
    if (vol.underwater > 0.5f) {
        color *= float3(0.4f, 0.7f, 1.0f); // cool underwater tint
    }
    return float4(color, 1.0f);
}

// ---------------------------------------------------------------------------
// Composite — additive upsample of the half-res shafts onto the HDR scene.
// (V flip: samples a rendered texture.)
// ---------------------------------------------------------------------------
vertex VolVertexOut volumetricCompositeVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

fragment float4 volumetricCompositeFragment(VolVertexOut in [[stage_in]],
                                            texture2d<float> volumetricTex [[texture(0)]]) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear,
                                    address::clamp_to_edge);
    return float4(volumetricTex.sample(linearSampler, in.vUV).rgb, 1.0f);
}
