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
    float vAO;             // baked corner AO 0.5-1 (per-vertex crease shading)
    float3 vWorldPosition; // World-space position for fog calculation
    uint vTextureLayer [[flat]];
};

// Face indices — must match FaceNormal in include/render/vertex.hpp
constant uint FACE_PLUS_Y = 4u;
constant uint FACE_CROSS = 6u;

// Shared exponential distance fog — one definition so the terrain, water,
// and entity passes can never drift apart.
static float3 applyFog(float3 color, float3 worldPos, float3 cameraPos, float density,
                       float3 fogColor) {
    float dist = distance(worldPos, cameraPos);
    float fogFactor = clamp(1.0f - exp(-density * dist), 0.0f, 1.0f);
    return mix(color, fogColor, fogFactor);
}

// ---------------------------------------------------------------------------
// Face normal lookup — indices match FaceNormal enum (0=+X … 5=-Y)
// ---------------------------------------------------------------------------
float3 getFaceNormal(uint index) {
    switch (index) {
        case 0:
            return float3(1.0, 0.0, 0.0); // +X
        case 1:
            return float3(-1.0, 0.0, 0.0); // -X
        case 2:
            return float3(0.0, 0.0, 1.0); // +Z
        case 3:
            return float3(0.0, 0.0, -1.0); // -Z
        case 4:
            return float3(0.0, 1.0, 0.0); // +Y
        case 5:
            return float3(0.0, -1.0, 0.0); // -Y
        default:
            return float3(0.0, 1.0, 0.0); // Fallback: +Y
    }
}

// ---------------------------------------------------------------------------
// Vertex shader — passes world position for fog calculation
// ---------------------------------------------------------------------------
vertex VertexOutput vertexMain(VertexInput in [[stage_in]],
                               constant Uniforms& uniforms [[buffer(1)]],
                               constant ChunkOrigin& chunkOrigin [[buffer(2)]]) {
    VertexOutput out;

    // Restore world space from the chunk-local position, then run MVP
    float4 worldPos = uniforms.modelMatrix * float4(in.position + chunkOrigin.origin.xyz, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;

    // Pass through UV
    out.vUV = in.uv;

    // Unpack face normal (bits 0-2), texture layer (bits 3-10), column
    // skylight (bits 11-14), and baked corner AO (bits 15-16). The AO level
    // 0..3 maps to 0.5..1.0 — a fully enclosed voxel corner keeps half its
    // light, an open corner none removed.
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    out.vAO = 0.5f + float((in.faceAttr >> 15) & 3u) * (1.0f / 6.0f);
    uint normalIdx = in.faceAttr & 7u;
    if (normalIdx == FACE_CROSS) {
        // Flora cross-quads: orientation-free lighting tracking the sun's
        // elevation, so the two diagonal quads never shade differently
        out.vNormal = float3(0.0, 1.0, 0.0);
        out.vLight = max(uniforms.sunDirection.y, 0.0f) * 0.9f;
        out.vAO = 1.0f;
        return out;
    }
    float3 normal = getFaceNormal(normalIdx);
    out.vNormal = normalize((uniforms.modelMatrix * float4(normal, 0.0)).xyz);

    // Per-vertex diffuse lighting
    float ndotl = dot(out.vNormal, uniforms.sunDirection);
    out.vLight = max(ndotl, 0.0);

    return out;
}

// A 16-point Poisson disk — even angular coverage with no axis-aligned
// banding, reused for both the PCSS blocker search and the PCF filter.
constant float2 POISSON_DISK[16] = {
    float2(-0.613f, 0.617f),  float2(0.170f, -0.040f),  float2(-0.299f, -0.791f),
    float2(0.645f, 0.493f),   float2(-0.651f, -0.378f), float2(0.918f, -0.126f),
    float2(0.344f, 0.294f),   float2(-0.108f, 0.987f),  float2(-0.920f, 0.078f),
    float2(0.542f, -0.782f),  float2(0.098f, -0.967f),  float2(-0.379f, 0.278f),
    float2(0.895f, 0.373f),   float2(-0.759f, -0.727f), float2(0.269f, 0.766f),
    float2(-0.288f, -0.303f),
};

// ---------------------------------------------------------------------------
// Cascaded shadow sampling with PCSS-style variable penumbra — one definition
// so terrain and entities can't drift. Returns a 0..1 lit factor (1 = fully
// lit) that scales the direct sun/moon contribution only; ambient stays so
// shadows never go pure black. A blocker search estimates how far occluders
// float above the receiver, so contact points stay crisp while shadows soften
// with distance from their caster — the Sildur soft-shadow look. The Poisson
// disk is rotated per pixel to trade banding for dithered noise the eye reads
// as a smooth penumbra.
// ---------------------------------------------------------------------------
// Penumbra tuning (multiples of shadowParams.x, the base texel radius).
constant float SHADOW_SEARCH_SCALE = 3.0f;     // blocker-search radius
constant float SHADOW_MAX_FILTER_SCALE = 4.0f; // widest PCF radius (soft edge)
constant float SHADOW_PENUMBRA_GAIN = 40.0f;   // depth gap → penumbra fraction

static float sampleShadow(float3 worldPos, float3 normal, float3 cameraPos, float2 screenPos,
                          depth2d_array<float> shadowMap, sampler shadowSampler,
                          constant ShadowUniforms& shadow) {
    float strength = shadow.shadowParams.z;
    if (strength <= 0.001f) {
        return 1.0f; // shadows disabled — fully lit
    }

    float dist = distance(worldPos, cameraPos);

    // Fade the whole term out near the shadow distance so the cascade edge
    // dissolves into the baked skylight instead of popping.
    float fade = 1.0f - saturate((dist - shadow.shadowParams.w) /
                                 max(shadow.cascadeSplitDist.z - shadow.shadowParams.w, 1.0f));
    if (fade <= 0.0f) {
        return 1.0f;
    }

    int cascade = SHADOW_CASCADE_COUNT - 1;
    for (int i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        if (dist < shadow.cascadeSplitDist[i]) {
            cascade = i;
            break;
        }
    }

    float3 offsetPos = worldPos + normal * shadow.shadowParams.y;
    float4 lightClip = shadow.cascadeViewProj[cascade] * float4(offsetPos, 1.0f);
    float3 ndc = lightClip.xyz / lightClip.w;
    float2 uv = ndc.xy * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y; // Metal texture v runs down
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f || ndc.z > 1.0f) {
        return 1.0f; // outside this cascade → treat as lit
    }

    float depthRef = ndc.z - 0.0015f; // constant slope-independent depth bias
    float texel = 1.0f / float(shadowMap.get_width());

    // Per-pixel Poisson rotation from an interleaved-gradient-noise hash of
    // the SCREEN pixel — bounded coordinates keep float precision (world-space
    // coords reach the thousands and collapse the hash), deterministic and
    // static so the penumbra dither never crawls (convention 9).
    float ign = fract(52.9829189f * fract(dot(screenPos, float2(0.06711056f, 0.00583715f))));
    float ang = ign * 6.2831853f;
    float2 rc = float2(cos(ang), sin(ang));
    float2x2 rot = float2x2(rc.x, -rc.y, rc.y, rc.x);

    // Point sampler for the raw blocker depth: Depth32Float is not linearly
    // filterable off Apple GPUs, and averaging depths across caster edges
    // would fabricate blocker distances anyway.
    constexpr sampler depthSampler(mag_filter::nearest, min_filter::nearest,
                                   address::clamp_to_edge);

    // ---- Blocker search: average depth of occluders in a search radius ----
    float searchRadius = shadow.shadowParams.x * SHADOW_SEARCH_SCALE * texel;
    float blockerSum = 0.0f;
    float blockerCount = 0.0f;
    for (int k = 0; k < 16; ++k) {
        float2 tap = uv + (rot * POISSON_DISK[k]) * searchRadius;
        float d = shadowMap.sample(depthSampler, tap, cascade);
        if (d < depthRef) {
            blockerSum += d;
            blockerCount += 1.0f;
        }
    }
    if (blockerCount < 0.5f) {
        return 1.0f; // no occluder in the search radius → fully lit
    }
    float avgBlocker = blockerSum / blockerCount;

    // ---- Penumbra width from the blocker/receiver depth gap ----
    // Contact (tiny gap) → tight kernel; occluder high above → wide, soft.
    float penumbra =
        saturate((depthRef - avgBlocker) / max(avgBlocker, 1e-4f) * SHADOW_PENUMBRA_GAIN);
    float filterRadius =
        mix(1.0f, shadow.shadowParams.x * SHADOW_MAX_FILTER_SCALE, penumbra) * texel;

    // ---- PCF over the same rotated disk at the penumbra-scaled radius ----
    float lit = 0.0f;
    for (int k = 0; k < 16; ++k) {
        float2 tap = uv + (rot * POISSON_DISK[k]) * filterRadius;
        lit += shadowMap.sample_compare(shadowSampler, tap, cascade, depthRef);
    }
    lit /= 16.0f;

    // Blend toward fully lit at the fade edge, and scale by strength.
    return mix(1.0f, mix(1.0f, lit, fade), strength);
}

// ---------------------------------------------------------------------------
// Fragment shader — sun/moon shadows + skylight + distance fog
// ---------------------------------------------------------------------------
fragment float4 fragmentMain(VertexOutput in [[stage_in]],
                             texture2d_array<float> blockTextures [[texture(0)]],
                             sampler blockSampler [[sampler(0)]],
                             depth2d_array<float> shadowMap [[texture(1)]],
                             sampler shadowSampler [[sampler(1)]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             constant ShadowUniforms& shadow [[buffer(4)]]) {
    // The bound sampler uses repeat addressing + nearest filtering: UVs span
    // the quad extent in blocks, so each block gets one full texture tile.
    float4 texColor = blockTextures.sample(blockSampler, in.vUV, in.vTextureLayer);

    // Alpha cutout for foliage/glass: transparent texels simply don't exist.
    // Runs in the opaque pass with depth writes, so no sorting is needed.
    if (texColor.a < 0.5f) {
        discard_fragment();
    }

    // Sky access from the baked per-column skylight. Because the mesher now
    // treats only OPAQUE blocks as sky-blockers, a tree canopy (non-opaque
    // leaves) no longer casts a fake column shadow on the ground below — that
    // shading comes entirely from the real cascade shadow. Genuine cover
    // (opaque terrain: caves, overhangs) still lowers skylight and darkens.
    float sky = 0.25f + 0.75f * in.vSkyLight;

    // The real cascade shadow gates the direct sun; sampleShadow returns 1.0
    // when shadows are disabled, so the term falls back to the sky access
    // alone (the old fake shadow) with no branch and no doubling.
    float lit = sampleShadow(in.vWorldPosition, in.vNormal, uniforms.cameraPosition,
                             in.clipPosition.xy, shadowMap, shadowSampler, shadow);

    // Baked corner AO darkens ambient and direct alike — the crease shading
    // reads as contact occlusion the same way vanilla applies it. The half-res
    // SSAO pass layers the broader mid-scale occlusion on top in screen space.
    float3 litColor = texColor.rgb * in.vAO *
                      (uniforms.sunColor * in.vLight * sky * lit + uniforms.ambientColor * sky);

    float3 finalColor = applyFog(litColor, in.vWorldPosition, uniforms.cameraPosition,
                                 uniforms.fogDensity, uniforms.fogColor);
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

vertex WaterVertexOutput waterVertexMain(VertexInput in [[stage_in]],
                                         constant Uniforms& uniforms [[buffer(1)]],
                                         constant ChunkOrigin& chunkOrigin [[buffer(2)]],
                                         constant WaterUniforms& water [[buffer(3)]]) {
    float3 pos = in.position + chunkOrigin.origin.xyz;
    uint normalIdx = in.faceAttr & 7u;
    if (normalIdx == FACE_PLUS_Y) {
        // Top surfaces bob gently; world-space input keeps waves continuous
        // across chunk borders
        pos.y +=
            sin(pos.x * 0.55f + water.time * 1.4f) * cos(pos.z * 0.45f + water.time * 1.1f) * 0.04f;
    }

    WaterVertexOutput out;
    float4 worldPos = uniforms.modelMatrix * float4(pos, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    return out;
}

// Analytic normal of the same wave field the vertex stage displaces with,
// plus a faint finer ripple set for sparkle. The slope scale keeps the
// surface glassy — pushing it higher reads as chop.
static float3 waterWaveNormal(float2 p, float t) {
    float ddx = 0.55f * cos(p.x * 0.55f + t * 1.4f) * cos(p.y * 0.45f + t * 1.1f) * 0.04f +
                cos(p.x * 1.9f + t * 2.3f) * 0.9f * 0.008f;
    float ddz = -0.45f * sin(p.x * 0.55f + t * 1.4f) * sin(p.y * 0.45f + t * 1.1f) * 0.04f +
                cos(p.y * 2.2f - t * 2.0f) * 1.1f * 0.008f;
    return normalize(float3(-ddx * 2.5f, 1.0f, -ddz * 2.5f));
}

// Interfering sine bands sharpened into bright filaments — the classic
// cheap caustic, projected on the floor's world xz.
static float causticPattern(float2 p, float t) {
    float c1 = sin(p.x * 0.9f + t * 1.7f) + sin(p.y * 1.1f - t * 1.3f);
    float c2 = sin((p.x + p.y) * 0.7f + t * 2.1f) + sin((p.x - p.y) * 0.8f + t * 1.1f);
    float c = (c1 + c2) * 0.25f;
    return pow(saturate(1.0f - abs(c)), 6.0f);
}

fragment float4 waterFragmentMain(WaterVertexOutput in [[stage_in]],
                                  texture2d<float> sceneColor [[texture(0)]],
                                  depth2d<float> sceneDepth [[texture(1)]],
                                  constant WaterUniforms& water [[buffer(3)]]) {
    constexpr sampler screenSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
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
    float2 refractUV = clamp(screenUV + N.xz * 0.035f * distortion, 0.001f, 0.999f);
    float refractDepth = sceneDepth.sample(screenSampler, refractUV);
    if (refractDepth < in.clipPosition.z) {
        // The distorted tap landed on something in FRONT of the surface —
        // fall back to the undistorted sample
        refractUV = screenUV;
        refractDepth = opaqueDepth;
    }
    float3 refracted = sceneColor.sample(screenSampler, refractUV).rgb;

    // World position of the refracted floor sample (caustics + absorption)
    float4 rclip = float4(refractUV.x * 2.0f - 1.0f, 1.0f - refractUV.y * 2.0f, refractDepth, 1.0f);
    float4 rworldH = water.invViewProjection * rclip;
    float3 rworld = rworldH.xyz / rworldH.w;
    float depthBelow = max(in.vWorldPosition.y - rworld.y, 0.0f);

    // ---- Caustics: bright ripple filaments on the shallow floor
    float caustic = causticPattern(rworld.xz * 0.6f, water.time) * exp(-depthBelow * 0.22f);
    refracted +=
        water.sunColor * caustic * 0.4f * in.vSkyLight * saturate(water.sunDirection.y * 2.0f);

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

    color =
        applyFog(color, in.vWorldPosition, water.cameraPosition, water.fogDensity, water.fogColor);

    // Hairline shorelines dissolve into the shore instead of aliasing
    color = mix(refracted, color, saturate(waterDepth * 3.0f));
    return float4(color, 1.0f);
}

// ---------------------------------------------------------------------------
// Underwater overlay — fullscreen veil + floor caustics when the camera is
// submerged, drawn after the water surfaces with alpha blending. The light
// shafts are now the real ray-marched volumetric pass (volumetrics.metal),
// which replaced the old fake banded screen-space rays here.
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

fragment float4 underwaterOverlayFragment(OverlayVertexOutput in [[stage_in]],
                                          depth2d<float> sceneDepth [[texture(1)]],
                                          constant WaterUniforms& water [[buffer(3)]]) {
    float t = water.time;
    float sunUp = saturate(water.sunDirection.y);

    // Caustics on every submerged surface around the camera — the water
    // pass only shades pixels behind a surface quad, so without this the
    // floor at the player's feet had none. Reconstruct the opaque world
    // position and project the same caustic field onto up-facing geometry.
    constexpr sampler screenSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float depth = sceneDepth.sample(screenSampler, in.uv);
    float4 clip = float4(in.uv.x * 2.0f - 1.0f, 1.0f - in.uv.y * 2.0f, depth, 1.0f);
    float4 worldH = water.invViewProjection * clip;
    float3 world = worldH.xyz / worldH.w;
    // Screen-space derivatives give the surface normal: walls get none
    float3 surfaceNormal = normalize(cross(dfdx(world), dfdy(world)));
    float upFacing = saturate(abs(surfaceNormal.y));
    // SEA_LEVEL (64, chunk.hpp) minus an epsilon: the water surface renders
    // at 63.875, so shore blocks at y >= 64 must not catch caustics
    float submerged = step(world.y, 63.9f);
    float dist = distance(world, water.cameraPosition);
    float caustic = causticPattern(world.xz * 0.85f, t) * exp(-max(64.0f - world.y, 0.0f) * 0.10f) *
                    exp(-dist * 0.03f);
    float3 causticColor = water.sunColor * caustic * upFacing * submerged * sunUp * 2.4f;

    float3 veil = float3(0.05f, 0.18f, 0.32f);
    return float4(veil + causticColor, 0.35f);
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

vertex EntityVertexOutput entityVertexMain(device const EntityVertex* vertices [[buffer(0)]],
                                           constant Uniforms& uniforms [[buffer(1)]],
                                           constant EntityModel& entityModel [[buffer(2)]],
                                           uint vertexID [[vertex_id]]) {
    device const EntityVertex& v = vertices[vertexID];

    EntityVertexOutput out;
    float4 worldPos = entityModel.model * float4(v.position, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vNormal = normalize((entityModel.model * float4(v.normal, 0.0)).xyz);
    out.vColor = v.color;
    return out;
}

fragment float4 entityFragmentMain(EntityVertexOutput in [[stage_in]],
                                   constant Uniforms& uniforms [[buffer(1)]]) {
    float light = max(dot(in.vNormal, uniforms.sunDirection), 0.0f);
    float3 litColor = in.vColor * (uniforms.sunColor * light + uniforms.ambientColor);

    return float4(applyFog(litColor, in.vWorldPosition, uniforms.cameraPosition,
                           uniforms.fogDensity, uniforms.fogColor),
                  1.0);
}
