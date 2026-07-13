#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Bloom Post-Processing Shaders
//
// Pipeline:
//   1. Extract pass — threshold bright pixels (above 1.0 luminance)
//   2. Kawase blur — 4-level pyramid, separable 8-tap blur
//   3. Composite pass — additive blend bloom onto original scene
//   4. ACES tone mapping applied during composite
// ---------------------------------------------------------------------------

// ---- Fullscreen quad vertex output ----
struct BloomVertexOut {
    float4 clipPosition [[position]];
    float2 vUV;
};

// ============================================================================
// 1. Extract Pass — Threshold bright pixels
// ============================================================================

vertex BloomVertexOut bloomExtractVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[6] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f,  1.0f)
    };
    BloomVertexOut out;
    out.clipPosition = float4(positions[vertexID], 0.0, 1.0);
    // Texture v runs downward in Metal while NDC y runs up; flip v so
    // every sampling pass preserves the image orientation.
    out.vUV = float2(positions[vertexID].x * 0.5f + 0.5f,
                     0.5f - positions[vertexID].y * 0.5f);
    return out;
}

fragment float4 bloomExtractFragment(
    BloomVertexOut in [[stage_in]],
    texture2d<float> sceneTexture [[texture(0)]],
    constant BloomUniforms &uniforms [[buffer(0)]]
) {
    constexpr sampler sceneSampler(mag_filter::linear, min_filter::linear);
    float4 sceneColor = sceneTexture.sample(sceneSampler, in.vUV);

    // Luminance (Rec. 709 weights)
    float luminance = dot(sceneColor.rgb, float3(0.2126f, 0.7152f, 0.0722f));

    // Soft threshold: smoothstep around the threshold value
    float softThreshold = smoothstep(
        uniforms.threshold - 0.5f,
        uniforms.threshold + 0.5f,
        luminance
    );

    // Only pass through pixels above threshold
    float3 brightColor = sceneColor.rgb * softThreshold;
    return float4(brightColor, 1.0);
}

// ============================================================================
// 2. Kawase Blur — Separable 8-tap pattern
// ============================================================================

vertex BloomVertexOut bloomBlurVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[6] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f,  1.0f)
    };
    BloomVertexOut out;
    out.clipPosition = float4(positions[vertexID], 0.0, 1.0);
    // Texture v runs downward in Metal while NDC y runs up; flip v so
    // every sampling pass preserves the image orientation.
    out.vUV = float2(positions[vertexID].x * 0.5f + 0.5f,
                     0.5f - positions[vertexID].y * 0.5f);
    return out;
}

fragment float4 bloomBlurFragment(
    BloomVertexOut in [[stage_in]],
    texture2d<float> inputTexture [[texture(0)]],
    constant BloomUniforms &uniforms [[buffer(0)]]
) {
    constexpr sampler blurSampler(mag_filter::linear, min_filter::linear);

    float radius = uniforms.blurRadius;
    float3 sum = float3(0.0);
    float weightSum = 0.0;

    // 8-tap Kawase pattern (cross-shaped with center)
    // Taps at: (-r,0), (-r/2,-r/2), (-r/2,r/2), (0,-r), (0,r), (r/2,-r/2), (r/2,r/2), (r,0)
    float2 offsets[8] = {
        float2(-radius, 0.0) * uniforms.texelSize,
        float2(-radius * 0.5f, -radius * 0.5f) * uniforms.texelSize,
        float2(-radius * 0.5f, radius * 0.5f) * uniforms.texelSize,
        float2(0.0, -radius) * uniforms.texelSize,
        float2(0.0, radius) * uniforms.texelSize,
        float2(radius * 0.5f, -radius * 0.5f) * uniforms.texelSize,
        float2(radius * 0.5f, radius * 0.5f) * uniforms.texelSize,
        float2(radius, 0.0) * uniforms.texelSize,
    };

    // Gaussian-like weights for each tap
    float weights[8] = {
        0.0625f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f
    };

    for (int i = 0; i < 8; ++i) {
        float2 uv = in.vUV + offsets[i];
        float3 color = inputTexture.sample(blurSampler, uv).rgb;
        sum += color * weights[i];
        weightSum += weights[i];
    }

    return float4(sum / weightSum, 1.0);
}

// ============================================================================
// 3. Composite Pass — Additive blend + ACES tone mapping
// ============================================================================

vertex BloomVertexOut bloomCompositeVertex(uint vertexID [[vertex_id]]) {
    const float2 positions[6] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f,  1.0f)
    };
    BloomVertexOut out;
    out.clipPosition = float4(positions[vertexID], 0.0, 1.0);
    // Texture v runs downward in Metal while NDC y runs up; flip v so
    // every sampling pass preserves the image orientation.
    out.vUV = float2(positions[vertexID].x * 0.5f + 0.5f,
                     0.5f - positions[vertexID].y * 0.5f);
    return out;
}

fragment float4 bloomCompositeFragment(
    BloomVertexOut in [[stage_in]],
    texture2d<float> sceneTexture [[texture(0)]],
    texture2d<float> bloomTexture [[texture(1)]],
    constant BloomUniforms &uniforms [[buffer(0)]]
) {
    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear);

    float3 sceneColor = sceneTexture.sample(linearSampler, in.vUV).rgb;
    float3 bloomColor = bloomTexture.sample(linearSampler, in.vUV).rgb;

    // Additive blend bloom onto scene
    float3 combined = sceneColor + bloomColor * uniforms.intensity;

    // ACES tone mapping: color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
    float3 a = combined * (2.51f * combined + 0.03f);
    float3 b = combined * (2.43f * combined + 0.59f) + 0.14f;
    float3 toneMapped = a / b;

    return float4(toneMapped, 1.0);
}
