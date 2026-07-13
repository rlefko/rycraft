#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Cloud Layer Shader — Procedural volumetric-style clouds
//
// Renders a fullscreen quad at cloud altitude (Y=192) with:
//   • 2D Simplex noise for cloud patterns (frequency 0.005, threshold 0.4)
//   • White clouds with 0.8 opacity, sunlit side brighter
//   • Wind-driven animation via noise coordinate offset
//
// Uniforms (buffer 0):
//   float3 cameraPosition — for parallax and sun direction
//   float3 sunDirection   — for cloud lighting
//   float windOffset      — worldTime * windSpeed (0.02 blocks/tick)
//   float cloudAltitude   — Y level of cloud layer (192)
//   float noiseFrequency  — noise scale (0.005)
//   float cloudThreshold  — noise threshold for cloud presence (0.4)
// ---------------------------------------------------------------------------

struct CloudVertexOut {
    float4 clipPosition [[position]];
    float2 vUV;
    float3 vWorldPos;
};

// ---- Hash-based pseudo-random noise (for simplex-like clouds) ----

// Fast hash function for noise
uint hash2D(uint2 p) {
    p = p * uint2(16843009u, 8884513u);
    return (p.x ^ p.y) * 1274126177u;
}

// Smooth noise using hash + interpolation
float noise2D(float2 p) {
    float2 ip = floor(p);
    float2 fp = fract(p);
    fp = fp * fp * (3.0f - 2.0f * fp); // Smoothstep

    uint2 i00 = uint2(ip);
    uint2 i10 = i00 + uint2(1u, 0u);
    uint2 i01 = i00 + uint2(0u, 1u);
    uint2 i11 = i00 + uint2(1u, 1u);

    float v00 = float(hash2D(i00) % 65536u) / 65535.0f;
    float v10 = float(hash2D(i10) % 65536u) / 65535.0f;
    float v01 = float(hash2D(i01) % 65536u) / 65535.0f;
    float v11 = float(hash2D(i11) % 65536u) / 65535.0f;

    float a = mix(v00, v10, fp.x);
    float b = mix(v01, v11, fp.x);
    return mix(a, b, fp.y);
}

// Fractal noise (multiple octaves)
float fractalNoise(float2 p, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for (int i = 0; i < octaves; ++i) {
        value += amplitude * noise2D(p * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value;
}

// ============================================================================
// Vertex shader — fullscreen quad at cloud altitude
// ============================================================================

vertex CloudVertexOut cloudVertexMain(
    uint vertexID [[vertex_id]],
    constant CloudUniforms &uniforms [[buffer(0)]]
) {
    // Fullscreen quad vertices in NDC
    const float2 ndcPositions[6] = {
        float2(-1.0f, -1.0f),
        float2( 1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f, -1.0f),
        float2( 1.0f,  1.0f),
        float2(-1.0f,  1.0f)
    };

    CloudVertexOut out;
    float2 ndc = ndcPositions[vertexID];
    out.clipPosition = float4(ndc, 0.99f, 1.0f); // Slight Z offset to sit behind scene
    out.vUV = ndc * 0.5f + 0.5f;

    // World position at cloud altitude for noise sampling
    // Spread the quad over a large area for noise sampling
    float spread = 2000.0f;
    out.vWorldPos = float3(
        uniforms.cameraPosition.x + ndc.x * spread,
        uniforms.cloudAltitude,
        uniforms.cameraPosition.z + ndc.y * spread
    );

    return out;
}

// ============================================================================
// Fragment shader — procedural cloud noise with wind animation
// ============================================================================

fragment float4 cloudFragmentMain(
    CloudVertexOut in [[stage_in]],
    constant CloudUniforms &uniforms [[buffer(0)]]
) {
    // Wind-driven noise coordinates
    float2 noiseCoord = in.vWorldPos.xz * uniforms.noiseFrequency;
    noiseCoord.x += uniforms.windOffset;

    // Multi-octave noise for cloud detail
    float cloudDensity = fractalNoise(noiseCoord, 4);

    // Threshold: only render where noise exceeds threshold
    float cloudMask = smoothstep(
        uniforms.cloudThreshold - 0.1f,
        uniforms.cloudThreshold + 0.1f,
        cloudDensity
    );

    // Cloud color: white base
    float3 cloudColor = float3(1.0f);

    // Sunlit side brighter
    float sunDot = max(dot(normalize(uniforms.sunDirection), float3(0.0f, 1.0f, 0.0f)), 0.0f);
    cloudColor *= 0.6f + 0.4f * sunDot;

    // Base opacity
    float alpha = cloudMask * 0.8f;

    // Discard non-cloud pixels
    if (alpha < 0.01f) {
        discard_fragment();
    }

    return float4(cloudColor, alpha);
}
