#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Cloud Layer Shader — procedural clouds in a world-space slab
//
// A fullscreen quad rides at the far end of the depth range so terrain
// occludes it. Each fragment reconstructs its view ray from the camera basis,
// then either samples the flat plane at Y = cloudAltitude (cloudMode 1) or
// ray-marches the [cloudAltitude, +SLAB_THICKNESS] volume with sun
// self-shadowing (cloudMode 2, CloudUniforms.volumetric). Cloud shapes come
// from wind-scrolled fractal value noise over world XZ, so the layer holds
// still in world space while the camera moves through it.
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

vertex CloudVertexOut cloudVertexMain(uint vertexID [[vertex_id]],
                                      constant CloudUniforms& uniforms [[buffer(0)]]) {
    const float2 ndcPositions[6] = {float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(1.0f, 1.0f),
                                    float2(-1.0f, -1.0f), float2(1.0f, 1.0f),  float2(-1.0f, 1.0f)};

    CloudVertexOut out;
    float2 ndc = ndcPositions[vertexID];
    // Near the far plane in Metal's [0,1] depth range: terrain always wins
    // the depth test, sky (which writes no depth) never does.
    out.clipPosition = float4(ndc, 0.9999f, 1.0f);
    out.vNdc = ndc;

    return out;
}

// ---- Volumetric density ----

// Cloud slab spans [cloudAltitude, cloudAltitude + SLAB_THICKNESS].
constant float SLAB_THICKNESS = 44.0f;

// Density at a world point (0 = clear). Wind-scrolled horizontal fractal noise,
// shaped by a vertical profile (rounded base, wispy top) so the volume reads as
// puffy clouds instead of a solid slab; height slightly warps the noise so
// stacked march samples differ.
static float cloudDensity3D(float3 p, constant CloudUniforms& u) {
    float2 nc = p.xz * u.noiseFrequency;
    nc.x += u.windOffset;
    nc += p.y * 0.02f;
    float base = fractalNoise(nc, 4);
    float h = saturate((p.y - u.cloudAltitude) / SLAB_THICKNESS); // 0 bottom → 1 top
    float profile = smoothstep(0.0f, 0.25f, h) * (1.0f - smoothstep(0.55f, 1.0f, h));
    return max(base - u.cloudThreshold, 0.0f) * profile * 3.0f;
}

// ============================================================================
// Fragment shader — flat plane sample (mode 1) or a ray-marched volume (mode 2)
// ============================================================================

fragment float4 cloudFragmentMain(CloudVertexOut in [[stage_in]],
                                  constant CloudUniforms& uniforms [[buffer(0)]]) {
    // View ray through this pixel from the camera basis + projection shape
    float3 dir =
        normalize(uniforms.cameraForward +
                  in.vNdc.x * uniforms.aspect * uniforms.tanHalfFov * uniforms.cameraRight +
                  in.vNdc.y * uniforms.tanHalfFov * uniforms.cameraUp);

    if (fabs(dir.y) < 1e-4f) {
        discard_fragment();
    }

    if (uniforms.volumetric > 0.5f) {
        // ---- Volumetric march through the cloud slab ----
        // Clamp dir.y away from zero (instead of the flat path's discard) so a
        // camera flying inside the slab looking dead-horizontal still marches
        // its cloud instead of leaving a one-pixel clear seam at the horizon.
        float dy = sign(dir.y) * max(fabs(dir.y), 1e-4f);
        float t0 = (uniforms.cloudAltitude - uniforms.cameraPosition.y) / dy;
        float t1 = (uniforms.cloudAltitude + SLAB_THICKNESS - uniforms.cameraPosition.y) / dy;
        float tEnter = max(min(t0, t1), 0.0f);
        float tExit = max(t0, t1);
        if (tExit <= tEnter) {
            discard_fragment(); // ray never crosses the slab ahead of the camera
        }

        // Horizon fade + night thinning bound the final alpha, so pixels they
        // already extinguish skip the march entirely (a fullscreen band of
        // near-horizon sky otherwise pays 80 noise calls to produce nothing).
        float fade = exp(-tEnter * 0.0008f) * (0.35f + 0.65f * uniforms.sunElevation);
        if (fade < 0.01f) {
            discard_fragment();
        }

        // Cap grazing rays that would otherwise march kilometers of slab; the
        // horizon fade hides the truncation.
        float marchLen = min(tExit - tEnter, 700.0f);
        const int STEPS = 20;
        float stepLen = marchLen / float(STEPS);
        float3 sunDir = normalize(uniforms.sunDirection);
        // Deterministic per-pixel jitter breaks the slab into a soft edge
        // instead of concentric banding.
        float jitter = interleavedGradientNoise(in.clipPosition.xy);

        float transmittance = 1.0f;
        float3 scatter = 0.0f;
        float3 sunTint =
            mix(float3(0.55f, 0.6f, 0.72f), float3(1.0f, 0.96f, 0.88f), uniforms.sunElevation);
        for (int i = 0; i < STEPS; ++i) {
            float t = tEnter + (float(i) + jitter) * stepLen;
            float3 sp = uniforms.cameraPosition + dir * t;
            float density = cloudDensity3D(sp, uniforms);
            if (density > 0.01f) {
                // Short march toward the sun for self-shadowing (Beer), plus a
                // powder term that darkens dense cores' edges.
                float lightDensity = 0.0f;
                for (int j = 1; j <= 3; ++j) {
                    lightDensity += cloudDensity3D(sp + sunDir * (float(j) * 7.0f), uniforms);
                }
                float sunAtten = exp(-lightDensity * 0.55f);
                float powder = 1.0f - exp(-density * 2.0f);
                float3 lit =
                    mix(float3(0.35f, 0.4f, 0.5f), sunTint, sunAtten) * (0.35f + 0.65f * powder);
                float stepT = exp(-density * stepLen * 0.45f);
                scatter += transmittance * (1.0f - stepT) * lit;
                transmittance *= stepT;
                if (transmittance < 0.02f) {
                    break;
                }
            }
        }

        float coverage = 1.0f - transmittance;
        float alpha = coverage * fade;
        if (alpha < 0.01f) {
            discard_fragment();
        }
        // The march accumulates PREmultiplied radiance (scatter ≈ lit ×
        // coverage), but the shared cloud pipeline blends straight alpha —
        // unpremultiply so wisps and edges don't get darkened twice.
        return float4(scatter / max(coverage, 1e-3f), alpha);
    }

    // ---- Flat plane layer (mode 1): single sample onto the cloud plane ----
    float t = (uniforms.cloudAltitude - uniforms.cameraPosition.y) / dir.y;
    if (t <= 0.0f) {
        discard_fragment();
    }
    float2 planeXZ = uniforms.cameraPosition.xz + dir.xz * t;
    float2 noiseCoord = planeXZ * uniforms.noiseFrequency;
    noiseCoord.x += uniforms.windOffset;
    float cloudDensity = fractalNoise(noiseCoord, 4);
    float cloudMask =
        smoothstep(uniforms.cloudThreshold - 0.1f, uniforms.cloudThreshold + 0.1f, cloudDensity);

    float3 cloudColor = float3(1.0f);
    float sunDot = max(dot(normalize(uniforms.sunDirection), float3(0.0f, 1.0f, 0.0f)), 0.0f);
    cloudColor *= 0.6f + 0.4f * sunDot;

    float fade = exp(-t * 0.0015f);
    float alpha = cloudMask * 0.8f * fade;
    if (alpha < 0.01f) {
        discard_fragment();
    }
    return float4(cloudColor, alpha);
}
