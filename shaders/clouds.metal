#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Cloud Layer Shader — Procedural clouds on a world-space plane
//
// A fullscreen quad rides at the far end of the depth range so terrain
// occludes it. Each fragment reconstructs its view ray from the camera basis
// and intersects the horizontal cloud plane (Y = cloudAltitude); rays that
// never reach the plane (looking away from it) draw nothing. Cloud shapes
// come from wind-scrolled fractal value noise over the plane's XZ coords, so
// the layer holds still in world space while the camera moves through it.
// ---------------------------------------------------------------------------

struct CloudVertexOut {
    float4 clipPosition [[position]];
    float2 vNdc;
};

// ---- Hash-based value noise ----

// Integer hash with good avalanche behavior (xorshift-multiply mix)
uint hash2D(uint2 p) {
    uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u;
    h ^= h >> 13;
    h *= 0x9E3779B1u;
    h ^= h >> 16;
    return h;
}

// Smooth noise using hash + interpolation
float noise2D(float2 p) {
    float2 ip = floor(p);
    float2 fp = fract(p);
    fp = fp * fp * (3.0f - 2.0f * fp); // Smoothstep

    // Lattice coords can be negative (world space around the camera); go
    // through int2 and reinterpret the two's-complement bits, because a
    // direct float→uint conversion clamps negatives and bands the noise.
    int2 base = int2(ip);
    uint2 i00 = as_type<uint2>(base);
    uint2 i10 = as_type<uint2>(base + int2(1, 0));
    uint2 i01 = as_type<uint2>(base + int2(0, 1));
    uint2 i11 = as_type<uint2>(base + int2(1, 1));

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
// Vertex shader — fullscreen quad parked at the far depth plane
// ============================================================================

vertex CloudVertexOut cloudVertexMain(
    uint vertexID [[vertex_id]],
    constant CloudUniforms &uniforms [[buffer(0)]]
) {
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
    // Near the far plane in Metal's [0,1] depth range: terrain always wins
    // the depth test, sky (which writes no depth) never does.
    out.clipPosition = float4(ndc, 0.9999f, 1.0f);
    out.vNdc = ndc;

    return out;
}

// ============================================================================
// Fragment shader — ray-cast onto the cloud plane, then sample cloud noise
// ============================================================================

fragment float4 cloudFragmentMain(
    CloudVertexOut in [[stage_in]],
    constant CloudUniforms &uniforms [[buffer(0)]]
) {
    // View ray through this pixel from the camera basis + projection shape
    float3 dir = normalize(
        uniforms.cameraForward
        + in.vNdc.x * uniforms.aspect * uniforms.tanHalfFov * uniforms.cameraRight
        + in.vNdc.y * uniforms.tanHalfFov * uniforms.cameraUp);

    // Intersect with the horizontal cloud plane
    float heightToPlane = uniforms.cloudAltitude - uniforms.cameraPosition.y;
    if (fabs(dir.y) < 1e-4f) {
        discard_fragment();
    }
    float t = heightToPlane / dir.y;
    if (t <= 0.0f) {
        discard_fragment();
    }

    float2 planeXZ = uniforms.cameraPosition.xz + dir.xz * t;

    // Wind-driven noise coordinates
    float2 noiseCoord = planeXZ * uniforms.noiseFrequency;
    noiseCoord.x += uniforms.windOffset;

    // Multi-octave noise for cloud detail
    float cloudDensity = fractalNoise(noiseCoord, 4);

    // Threshold: only render where noise exceeds threshold
    float cloudMask = smoothstep(
        uniforms.cloudThreshold - 0.1f,
        uniforms.cloudThreshold + 0.1f,
        cloudDensity
    );

    // Cloud color: white base, sunlit side brighter
    float3 cloudColor = float3(1.0f);
    float sunDot = max(dot(normalize(uniforms.sunDirection), float3(0.0f, 1.0f, 0.0f)), 0.0f);
    cloudColor *= 0.6f + 0.4f * sunDot;

    // Fade with distance along the ray so the layer dissolves at the horizon
    float fade = exp(-t * 0.0015f);
    float alpha = cloudMask * 0.8f * fade;

    if (alpha < 0.01f) {
        discard_fragment();
    }

    return float4(cloudColor, alpha);
}
