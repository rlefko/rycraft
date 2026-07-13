#include <metal_stdlib>
#include <render/shader_types.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Vertex input — bound through the vertex descriptor (not argument buffer)
//
// Attribute layout matches include/render/vertex.hpp:
//   attribute(0)  uint     faceAttr          offset 0   4 bytes  (UInt)
//                          face normal in bits 0-2, texture layer in bits 3+
//   attribute(1)  float3   px, py, pz        offset 4   6 bytes  (Half3)
//                          CHUNK-LOCAL position; ChunkOrigin restores world
//   attribute(2)  float2   u, v              offset 10  4 bytes  (Half2)
//                          spans the quad extent in blocks (repeat-sampled)
//   stride = 16 bytes
// ---------------------------------------------------------------------------
struct VertexInput {
    uint faceAttr [[attribute(0)]];
    float3 position [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

// ---------------------------------------------------------------------------
// Vertex → Fragment inter-stage
// ---------------------------------------------------------------------------
struct VertexOutput {
    float4 clipPosition [[position]];
    float3 vNormal;
    float2 vUV;
    float vLight;
    float vSkyLight;       // column skylight 0-1 (cast shade, cave darkness)
    float3 vWorldPosition; // World-space position for fog calculation
    uint vTextureLayer [[flat]];
};

// ---------------------------------------------------------------------------
// Face normal lookup — indices match FaceNormal enum (0=+X … 5=-Y)
// ---------------------------------------------------------------------------
float3 getFaceNormal(uint index) {
    switch (index) {
        case 0: return float3( 1.0,  0.0,  0.0); // +X
        case 1: return float3(-1.0,  0.0,  0.0); // -X
        case 2: return float3( 0.0,  0.0,  1.0); // +Z
        case 3: return float3( 0.0,  0.0, -1.0); // -Z
        case 4: return float3( 0.0,  1.0,  0.0); // +Y
        case 5: return float3( 0.0, -1.0,  0.0); // -Y
        default: return float3(0.0, 1.0, 0.0);    // Fallback: +Y
    }
}

// ---------------------------------------------------------------------------
// Vertex shader — passes world position for fog calculation
// ---------------------------------------------------------------------------
vertex VertexOutput vertexMain(
    VertexInput in [[stage_in]],
    constant Uniforms &uniforms [[buffer(1)]],
    constant ChunkOrigin &chunkOrigin [[buffer(2)]]
) {
    VertexOutput out;

    // Restore world space from the chunk-local position, then run MVP
    float4 worldPos =
        uniforms.modelMatrix * float4(in.position + chunkOrigin.origin.xyz, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;

    // Pass through UV
    out.vUV = in.uv;

    // Unpack face normal (bits 0-2), texture layer (bits 3-10), and
    // column skylight (bits 11-14)
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    uint normalIdx = in.faceAttr & 7u;
    if (normalIdx == 6u) {
        // Flora cross-quads: orientation-free lighting tracking the sun's
        // elevation, so the two diagonal quads never shade differently
        out.vNormal = float3(0.0, 1.0, 0.0);
        out.vLight = max(uniforms.sunDirection.y, 0.0f) * 0.9f;
        return out;
    }
    float3 normal = getFaceNormal(normalIdx);
    out.vNormal = normalize(
        (uniforms.modelMatrix * float4(normal, 0.0)).xyz
    );

    // Per-vertex diffuse lighting
    float ndotl = dot(out.vNormal, uniforms.sunDirection);
    out.vLight = max(ndotl, 0.0);

    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — with distance fog (Phase 8)
// ---------------------------------------------------------------------------
fragment float4 fragmentMain(
    VertexOutput in [[stage_in]],
    texture2d_array<float> blockTextures [[texture(0)]],
    sampler blockSampler [[sampler(0)]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    // The bound sampler uses repeat addressing + nearest filtering: UVs span
    // the quad extent in blocks, so each block gets one full texture tile.
    float4 texColor = blockTextures.sample(blockSampler, in.vUV, in.vTextureLayer);

    // Alpha cutout for foliage/glass: transparent texels simply don't exist.
    // Runs in the opaque pass with depth writes, so no sorting is needed.
    if (texColor.a < 0.5f) {
        discard_fragment();
    }

    // Combine directional sun light with ambient, both attenuated by the
    // column skylight so covered ground and caves sit in shadow
    float sky = 0.25f + 0.75f * in.vSkyLight;
    float3 litColor =
        texColor.rgb * (uniforms.sunColor * in.vLight * sky + uniforms.ambientColor * sky);

    // ---- Distance fog (Phase 8) ----
    // Use world position passed from vertex shader
    float distanceToFrag = distance(in.vWorldPosition, uniforms.cameraPosition);

    // Exponential fog: fogFactor = 1.0 - exp(-density * distance)
    float fogFactor = 1.0f - exp(-uniforms.fogDensity * distanceToFrag);
    fogFactor = clamp(fogFactor, 0.0f, 1.0f);

    // Blend between fog color and lit color
    float3 finalColor = mix(litColor, uniforms.fogColor, fogFactor);

    return float4(finalColor, 1.0);
}

// ---------------------------------------------------------------------------
// Water pass — runs after the opaque scene resolves, compositing its own
// pixels from the resolved color + depth: screen-space refraction with
// depth-based absorption, procedural caustics on the submerged floor,
// fresnel sky reflection with a sun sparkle, and animated waves. No depth
// attachment: the fragment depth-tests manually against the resolved depth.
// ---------------------------------------------------------------------------
struct WaterVertexOutput {
    float4 clipPosition [[position]];
    float3 vWorldPosition;
    float vSkyLight;
};

vertex WaterVertexOutput waterVertexMain(
    VertexInput in [[stage_in]],
    constant Uniforms &uniforms [[buffer(1)]],
    constant ChunkOrigin &chunkOrigin [[buffer(2)]],
    constant WaterUniforms &water [[buffer(3)]]
) {
    float3 pos = in.position + chunkOrigin.origin.xyz;
    uint normalIdx = in.faceAttr & 7u;
    if (normalIdx == 4u) {
        // Top surfaces bob; world-space input keeps waves continuous
        // across chunk borders
        pos.y += sin(pos.x * 0.55f + water.time * 1.4f) *
                 cos(pos.z * 0.45f + water.time * 1.1f) * 0.06f;
    }

    WaterVertexOutput out;
    float4 worldPos = uniforms.modelMatrix * float4(pos, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    return out;
}

// Analytic normal of the same wave field the vertex stage displaces with,
// plus a finer ripple set for sparkle.
static float3 waterWaveNormal(float2 p, float t) {
    float ddx = 0.55f * cos(p.x * 0.55f + t * 1.4f) * cos(p.y * 0.45f + t * 1.1f) * 0.06f +
                cos(p.x * 1.9f + t * 2.3f) * 0.9f * 0.015f;
    float ddz = -0.45f * sin(p.x * 0.55f + t * 1.4f) * sin(p.y * 0.45f + t * 1.1f) * 0.06f +
                cos(p.y * 2.2f - t * 2.0f) * 1.1f * 0.015f;
    return normalize(float3(-ddx * 6.0f, 1.0f, -ddz * 6.0f));
}

// Interfering sine bands sharpened into bright filaments — the classic
// cheap caustic, projected on the floor's world xz.
static float causticPattern(float2 p, float t) {
    float c1 = sin(p.x * 0.9f + t * 1.7f) + sin(p.y * 1.1f - t * 1.3f);
    float c2 = sin((p.x + p.y) * 0.7f + t * 2.1f) + sin((p.x - p.y) * 0.8f + t * 1.1f);
    float c = (c1 + c2) * 0.25f;
    return pow(saturate(1.0f - abs(c)), 6.0f);
}

fragment float4 waterFragmentMain(
    WaterVertexOutput in [[stage_in]],
    texture2d<float> sceneColor [[texture(0)]],
    depth2d<float> sceneDepth [[texture(1)]],
    constant WaterUniforms &water [[buffer(3)]]
) {
    constexpr sampler screenSampler(mag_filter::linear, min_filter::linear,
                                    address::clamp_to_edge);
    float2 screenUV = in.clipPosition.xy / water.resolution;

    // Manual depth test against the resolved opaque scene
    float opaqueDepth = sceneDepth.sample(screenSampler, screenUV);
    if (in.clipPosition.z > opaqueDepth) {
        discard_fragment();
    }

    float3 V = normalize(water.cameraPosition - in.vWorldPosition);
    float3 N = waterWaveNormal(in.vWorldPosition.xz, water.time);
    bool fromBelow = water.cameraUnderwater > 0.5f;

    // Reconstruct the opaque world position behind this fragment
    float4 clip = float4(screenUV.x * 2.0f - 1.0f, 1.0f - screenUV.y * 2.0f, opaqueDepth, 1.0f);
    float4 behindH = water.invViewProjection * clip;
    float3 behind = behindH.xyz / behindH.w;
    float waterDepth = max(distance(behind, in.vWorldPosition), 0.0f);

    // ---- Refraction: wave-distorted resample of the scene, pinned at the
    // shoreline so shallow edges don't smear
    float distortion = min(waterDepth, 4.0f) * 0.25f;
    float2 refractUV = clamp(screenUV + N.xz * 0.05f * distortion, 0.001f, 0.999f);
    float refractDepth = sceneDepth.sample(screenSampler, refractUV);
    if (refractDepth < in.clipPosition.z) {
        // The distorted tap landed on something in FRONT of the surface —
        // fall back to the undistorted sample
        refractUV = screenUV;
        refractDepth = opaqueDepth;
    }
    float3 refracted = sceneColor.sample(screenSampler, refractUV).rgb;

    // World position of the refracted floor sample (caustics + absorption)
    float4 rclip =
        float4(refractUV.x * 2.0f - 1.0f, 1.0f - refractUV.y * 2.0f, refractDepth, 1.0f);
    float4 rworldH = water.invViewProjection * rclip;
    float3 rworld = rworldH.xyz / rworldH.w;
    float depthBelow = max(in.vWorldPosition.y - rworld.y, 0.0f);

    // ---- Caustics: bright ripple filaments on the shallow floor
    float caustic = causticPattern(rworld.xz * 1.3f, water.time) * exp(-depthBelow * 0.22f);
    refracted += water.sunColor * caustic * 0.4f * in.vSkyLight *
                 saturate(water.sunDirection.y * 2.0f);

    // ---- Absorption: the water column filters toward deep blue
    float3 deepColor = float3(0.02f, 0.12f, 0.25f);
    float absorb = 1.0f - exp(-waterDepth * 0.16f);
    float3 body = mix(refracted * float3(0.85f, 0.95f, 1.0f), deepColor, absorb);

    // ---- Fresnel sky reflection + sun sparkle (skylight gates both, so
    // flooded caves reflect darkness rather than open sky)
    float3 R = reflect(-V, N);
    R.y = abs(R.y);
    float horizonBlend = pow(1.0f - saturate(R.y), 2.0f);
    float3 skyReflection = mix(water.zenithColor, water.horizonColor, horizonBlend);
    float sunAlign = saturate(dot(R, water.sunDirection));
    float sparkle = pow(sunAlign, 240.0f) * 2.0f + pow(sunAlign, 32.0f) * 0.25f;
    float fresnel = mix(0.04f, 1.0f, pow(1.0f - saturate(dot(V, N)), 5.0f));
    if (fromBelow) {
        fresnel *= 0.3f; // looking up at the surface: mostly see through it
    }

    float3 color = mix(body, skyReflection, fresnel * in.vSkyLight);
    color += water.sunColor * sparkle * in.vSkyLight;

    // ---- Distance fog, matching the terrain shader
    float dist = distance(in.vWorldPosition, water.cameraPosition);
    float fogFactor = clamp(1.0f - exp(-water.fogDensity * dist), 0.0f, 1.0f);
    color = mix(color, water.fogColor, fogFactor);

    // Hairline shorelines dissolve into the shore instead of aliasing
    color = mix(refracted, color, saturate(waterDepth * 3.0f));
    return float4(color, 1.0f);
}

// ---------------------------------------------------------------------------
// Underwater overlay — fullscreen veil + god rays when the camera is
// submerged, drawn after the water surfaces with alpha blending.
// ---------------------------------------------------------------------------
struct OverlayVertexOutput {
    float4 clipPosition [[position]];
    float2 uv;
};

vertex OverlayVertexOutput underwaterOverlayVertex(uint vertexID [[vertex_id]]) {
    // Fullscreen triangle
    float2 pos[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    OverlayVertexOutput out;
    out.clipPosition = float4(pos[vertexID], 0.0, 1.0);
    out.uv = float2(pos[vertexID].x * 0.5f + 0.5f, 1.0f - (pos[vertexID].y * 0.5f + 0.5f));
    return out;
}

fragment float4 underwaterOverlayFragment(
    OverlayVertexOutput in [[stage_in]],
    constant WaterUniforms &water [[buffer(3)]]
) {
    // Slanted shafts of light, banded and animated, fading with depth on
    // screen (light enters from the surface above)
    float slant = in.uv.x * 1.4f + in.uv.y * 0.6f;
    float t = water.time;
    float rays = pow(max(sin(slant * 9.0f - t * 0.7f) * 0.5f + 0.5f, 0.0f), 3.0f) * 0.5f +
                 pow(max(sin(slant * 17.0f + t * 0.45f + 1.7f) * 0.5f + 0.5f, 0.0f), 4.0f) * 0.35f;
    float topFade = pow(saturate(1.0f - in.uv.y), 1.5f);
    float sunUp = saturate(water.sunDirection.y);
    float3 rayColor = water.sunColor * rays * topFade * sunUp * 0.9f;

    float3 veil = float3(0.05f, 0.18f, 0.32f);
    return float4(veil + rayColor, 0.35f);
}

// ---------------------------------------------------------------------------
// Entity shaders — voxel-box animal models, lit like terrain
// ---------------------------------------------------------------------------
struct EntityVertexOutput {
    float4 clipPosition [[position]];
    float3 vNormal;
    float3 vColor;
    float3 vWorldPosition;
};

vertex EntityVertexOutput entityVertexMain(
    device const EntityVertex* vertices [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]],
    constant EntityModel &entityModel [[buffer(2)]],
    uint vertexID [[vertex_id]]
) {
    device const EntityVertex& v = vertices[vertexID];

    EntityVertexOutput out;
    float4 worldPos = entityModel.model * float4(v.position, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vNormal = normalize((entityModel.model * float4(v.normal, 0.0)).xyz);
    out.vColor = v.color;
    return out;
}

fragment float4 entityFragmentMain(
    EntityVertexOutput in [[stage_in]],
    constant Uniforms &uniforms [[buffer(1)]]
) {
    float light = max(dot(in.vNormal, uniforms.sunDirection), 0.0f);
    float3 litColor = in.vColor * (uniforms.sunColor * light + uniforms.ambientColor);

    float distanceToFrag = distance(in.vWorldPosition, uniforms.cameraPosition);
    float fogFactor = clamp(1.0f - exp(-uniforms.fogDensity * distanceToFrag), 0.0f, 1.0f);
    return float4(mix(litColor, uniforms.fogColor, fogFactor), 1.0);
}
