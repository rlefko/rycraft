#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Screen-space ambient occlusion.
//
// Generate (half-res): reconstruct each pixel's view-space position and a
// normal from depth derivatives, sample a hemisphere of points rotated per
// pixel by interleaved-gradient noise, and count how many find geometry
// standing above the receiver's tangent plane. Apply (full-res): box-upsample
// the half-res AO with a 4-tap bilinear average and multiply it onto the HDR
// scene so creases, block corners, and cave interiors darken smoothly.
// Multiply blending means the apply pass never samples the scene it writes.
// ---------------------------------------------------------------------------

struct SsaoVertexOut {
    float4 clipPosition [[position]];
    float2 vUV;
};

static SsaoVertexOut fullscreenTriangle(uint vertexID) {
    const float2 pos[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
    SsaoVertexOut out;
    out.clipPosition = float4(pos[vertexID], 0.0f, 1.0f);
    out.vUV = float2(pos[vertexID].x * 0.5f + 0.5f, 0.5f - pos[vertexID].y * 0.5f);
    return out;
}

vertex SsaoVertexOut ssaoVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

// View-space position of a screen UV + its depth, via the inverse projection.
static float3 viewPosFromDepth(float2 uv, float depth, constant SsaoUniforms& s) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    float4 view = s.invProjection * clip;
    return view.xyz / view.w;
}

// A small hemisphere kernel (fixed directions, scaled by a growing radius so
// samples cluster near the center).
constant float3 SSAO_KERNEL[12] = {
    float3(0.14f, 0.21f, 0.28f),   float3(-0.33f, 0.11f, 0.19f), float3(0.19f, -0.29f, 0.35f),
    float3(-0.12f, -0.16f, 0.44f), float3(0.42f, 0.05f, 0.51f),  float3(-0.28f, 0.34f, 0.22f),
    float3(0.05f, -0.47f, 0.30f),  float3(0.38f, -0.22f, 0.60f), float3(-0.55f, -0.18f, 0.41f),
    float3(0.24f, 0.52f, 0.48f),   float3(-0.44f, 0.40f, 0.66f), float3(0.10f, 0.08f, 0.78f),
};

// How much the self-occlusion guard grows with the surface's per-texel depth
// span. Grazing far ground spans several world units per half-res texel, so a
// fixed bias can't cover the depth-quantization error there and the ground
// bands; scaling by the local span keeps flat grazing ground quiet while
// face-on ground (small span) still darkens creases at the fixed base bias.
constant float SSAO_SLOPE_BIAS = 4.0f;

fragment float4 ssaoGenerateFragment(SsaoVertexOut in [[stage_in]],
                                     depth2d<float> sceneDepth [[texture(0)]],
                                     constant SsaoUniforms& s [[buffer(0)]]) {
    constexpr sampler depthSampler(mag_filter::nearest, min_filter::nearest,
                                   address::clamp_to_edge);
    float depth = sceneDepth.sample(depthSampler, in.vUV);
    if (depth >= 1.0f) {
        return float4(1.0f); // sky → unoccluded
    }

    float3 origin = viewPosFromDepth(in.vUV, depth, s);

    // View-space neighbours on BOTH sides, then reconstruct the normal from the
    // side with the smaller depth step. A one-sided derivative straddles a
    // crease (reading a normal halfway into a wall) and — the cause of the
    // reported scanlines — turns the depth-quantized far ground into a wobbling
    // normal, so flat grazing ground self-occluded in horizontal bands. The
    // best-of-both-sides pick reads the continuous surface in either case.
    float2 texel = 1.0f / s.resolution;
    float3 posL =
        viewPosFromDepth(in.vUV - float2(texel.x, 0.0f),
                         sceneDepth.sample(depthSampler, in.vUV - float2(texel.x, 0.0f)), s);
    float3 posR =
        viewPosFromDepth(in.vUV + float2(texel.x, 0.0f),
                         sceneDepth.sample(depthSampler, in.vUV + float2(texel.x, 0.0f)), s);
    float3 posD =
        viewPosFromDepth(in.vUV - float2(0.0f, texel.y),
                         sceneDepth.sample(depthSampler, in.vUV - float2(0.0f, texel.y)), s);
    float3 posU =
        viewPosFromDepth(in.vUV + float2(0.0f, texel.y),
                         sceneDepth.sample(depthSampler, in.vUV + float2(0.0f, texel.y)), s);
    float3 ddx =
        (abs(posR.z - origin.z) < abs(origin.z - posL.z)) ? (posR - origin) : (origin - posL);
    float3 ddy =
        (abs(posU.z - origin.z) < abs(origin.z - posD.z)) ? (posU - origin) : (origin - posD);
    float3 normal = normalize(cross(ddx, ddy));
    // Force it to face the camera (view +Z): the derivative winding can point
    // it into the surface, which would aim the hemisphere INTO the geometry
    // and read every pixel as fully occluded (near ground went black).
    if (normal.z < 0.0f) {
        normal = -normal;
    }

    // Depth span of one texel along the surface (measured on the continuous
    // side, so a crease keeps a tight guard). Grazing far ground has a large
    // span; the self-occlusion bias grows with it so the ground stops banding.
    float adaptiveBias = s.bias + (abs(ddx.z) + abs(ddy.z)) * SSAO_SLOPE_BIAS;

    // Per-pixel rotation angle (IGN) to decorrelate the fixed kernel.
    float2 fragPx = in.clipPosition.xy + float2(s.frameIndex % 4u) * 7.13f;
    float rnd = interleavedGradientNoise(fragPx);
    float ca = cos(rnd * 6.2831853f);
    float sa = sin(rnd * 6.2831853f);

    float occlusion = 0.0f;
    for (int i = 0; i < 12; ++i) {
        // Rotate the kernel around view Z, then flip into the normal hemisphere.
        float3 k = SSAO_KERNEL[i];
        float3 dir = float3(k.x * ca - k.y * sa, k.x * sa + k.y * ca, k.z);
        if (dot(dir, normal) < 0.0f) {
            dir = -dir;
        }
        float3 samplePos = origin + dir * s.radius;

        // Project the sample back to screen and read the stored depth there.
        float4 clip = s.projection * float4(samplePos, 1.0f);
        float2 sampleUV = (clip.xy / clip.w) * float2(0.5f, -0.5f) + 0.5f;
        if (sampleUV.x < 0.0f || sampleUV.x > 1.0f || sampleUV.y < 0.0f || sampleUV.y > 1.0f) {
            continue;
        }
        float sampleDepth = sceneDepth.sample(depthSampler, sampleUV);
        float3 occluderView = viewPosFromDepth(sampleUV, sampleDepth, s);

        // Occluded when the visible geometry at the sample rises above the
        // receiver's tangent plane by more than the bias. Measuring the rise
        // perpendicular to the surface (not a raw view-z compare) keeps the
        // grazing-ground quantization error IN the plane where it cancels; the
        // depth-span-scaled adaptiveBias then absorbs what's left, while real
        // occluders (a block face above the ground) still stand proud of it.
        float3 toOccluder = occluderView - origin;
        float planeDist = dot(toOccluder, normal);
        float rangeCheck = smoothstep(0.0f, 1.0f, s.radius / max(length(toOccluder), 1e-4f));
        if (planeDist > adaptiveBias) {
            occlusion += rangeCheck;
        }
    }
    // Fade AO as the surface turns edge-on to the view: the depth-derived
    // normal is least reliable at grazing angles (ceilings near the top of
    // the frame), where residual noise reads as banding rather than shading.
    float grazeFade = smoothstep(0.15f, 0.40f, abs(normal.z));
    float ao = 1.0f - (occlusion / 12.0f) * s.strength * grazeFade;
    return float4(saturate(ao));
}

// ---------------------------------------------------------------------------
// Depth-aware bilateral blur of the half-res AO, run between generate and
// apply. The generate pass rotates its kernel with per-pixel IGN and has no
// temporal accumulation to hide that noise under (the renderer keeps MSAA,
// not TAA), so without a blur the diagonal rotation field printed through as
// scan lines on grazing surfaces. The depth weight keeps the blur from
// bleeding AO across block silhouettes — every voxel edge is a depth edge.
// ---------------------------------------------------------------------------
fragment float4 ssaoBlurFragment(SsaoVertexOut in [[stage_in]],
                                 texture2d<float> aoTex [[texture(0)]],
                                 depth2d<float> sceneDepth [[texture(1)]],
                                 constant SsaoUniforms& s [[buffer(0)]]) {
    constexpr sampler pointSampler(mag_filter::nearest, min_filter::nearest,
                                   address::clamp_to_edge);
    float centerDepth = sceneDepth.sample(pointSampler, in.vUV);
    if (centerDepth >= 1.0f) {
        return float4(1.0f); // sky
    }
    float centerZ = viewPosFromDepth(in.vUV, centerDepth, s).z;

    // 5x5 spans ~4 half-res texels — wider than the IGN period — with a
    // gaussian falloff (sigmaS ~1.6 texels, sigmaZ ~0.5 view units).
    float2 texel = 1.0f / s.resolution;
    float sum = 0.0f;
    float wsum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            float2 uv = in.vUV + float2(dx, dy) * texel;
            float d = sceneDepth.sample(pointSampler, uv);
            float z = viewPosFromDepth(uv, d, s).z;
            float wS = exp(-float(dx * dx + dy * dy) / (2.0f * 1.6f * 1.6f));
            float dz = z - centerZ;
            float wZ = exp(-(dz * dz) / (2.0f * 0.5f * 0.5f));
            float w = wS * wZ;
            sum += aoTex.sample(pointSampler, uv).r * w;
            wsum += w;
        }
    }
    float ao = sum / max(wsum, 1e-4f);
    return float4(ao, ao, ao, 1.0f);
}

vertex SsaoVertexOut ssaoApplyVertex(uint vertexID [[vertex_id]]) {
    return fullscreenTriangle(vertexID);
}

fragment float4 ssaoApplyFragment(SsaoVertexOut in [[stage_in]],
                                  texture2d<float> aoTex [[texture(0)]]) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    // 4-tap upsample smooths the half-res AO before it multiplies the scene.
    float2 texel = 1.0f / float2(aoTex.get_width(), aoTex.get_height());
    float ao = 0.0f;
    ao += aoTex.sample(linearSampler, in.vUV + float2(-0.5f, -0.5f) * texel).r;
    ao += aoTex.sample(linearSampler, in.vUV + float2(0.5f, -0.5f) * texel).r;
    ao += aoTex.sample(linearSampler, in.vUV + float2(-0.5f, 0.5f) * texel).r;
    ao += aoTex.sample(linearSampler, in.vUV + float2(0.5f, 0.5f) * texel).r;
    ao *= 0.25f;
    return float4(ao, ao, ao, 1.0f); // multiply-blended onto the HDR scene
}
