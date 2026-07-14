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
    float vBlockLight;     // lava block light 0-1 (warm cave glow)
    float vEmissive;       // 1 = self-lit block (lava), 0 = normally lit
    float vFoliage;        // shading class: 0 solid, 1 = cross-quad flora
                           // (two-sided facing + SSS), 2 = leaf cube (SSS only)
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

    // Restore world space from the chunk-local position, sway foliage in the
    // wind (bits 22-23; the shadow pass applies the same displacement), then MVP
    float4 worldPos = uniforms.modelMatrix * float4(in.position + chunkOrigin.origin.xyz, 1.0);
    uint sway = (in.faceAttr >> 22) & 3u;
    worldPos.xyz = applySway(worldPos.xyz, sway, in.uv.y, uniforms.time, uniforms.swayStrength);
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
    // Block light (bits 17-20) and the emissive flag (bit 21) — the lava glow.
    out.vBlockLight = float((in.faceAttr >> 17) & 15u) / 15.0f;
    out.vEmissive = float((in.faceAttr >> 21) & 1u);
    uint normalIdx = in.faceAttr & 7u;
    // Shading class: every cross quad (grass, flowers, reeds, mushrooms)
    // gets the fragment facing term; swaying leaf cubes get SSS only.
    out.vFoliage = normalIdx == FACE_CROSS ? 1.0f : (sway == 2u ? 2.0f : 0.0f);
    if (normalIdx == FACE_CROSS) {
        // Flora cross-quads: elevation-tracked base light; the fragment stage
        // adds the two-sided facing term (it needs per-pixel derivatives).
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
    float ign = interleavedGradientNoise(screenPos);
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

    // Emissive blocks (lava) glow at a fixed HDR level, ignoring sun, shadow,
    // and skylight — the >1.0 output exceeds the bloom threshold naturally and
    // makes lava the light source the block-light term below spreads from.
    // Kept modest so the molten orange survives tonemapping instead of
    // clipping to white when auto-exposure lifts a dark scene.
    constexpr float EMISSIVE_BOOST = 3.0f;
    if (in.vEmissive > 0.5f) {
        float3 glow = texColor.rgb * EMISSIVE_BOOST;
        return float4(applyFog(glow, in.vWorldPosition, uniforms.cameraPosition,
                               uniforms.fogDensity, uniforms.fogColor),
                      1.0f);
    }

    // Rain-soaked surfaces darken (water fills the surface pores) — scaled
    // by sky access so caves and covered builds stay dry during surface rain.
    texColor.rgb *= 1.0f - 0.22f * uniforms.wetness * in.vSkyLight;

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

    // Two-sided flora shading: the geometric normal of the visible blade side
    // (from screen-space derivatives, flipped toward the camera) modulates the
    // direct sun, so the face turned away from the sun reads as self-shaded
    // instead of matching the sunlit side exactly. The term relaxes toward 1
    // as the sun climbs: vertical blades are edge-on to a high sun, so at noon
    // BOTH sides of every blade scored ~0.4 and all flora went uniformly dark
    // — the facing contrast only means something at low sun angles.
    float3 toCamera = normalize(uniforms.cameraPosition - in.vWorldPosition);
    float direct = in.vLight;
    if (in.vFoliage > 0.5f && in.vFoliage < 1.5f) {
        float3 blade = normalize(cross(dfdx(in.vWorldPosition), dfdy(in.vWorldPosition)));
        blade = dot(blade, toCamera) < 0.0f ? -blade : blade;
        float facing = 0.4f + 0.6f * saturate(dot(blade, uniforms.sunDirection));
        direct *= mix(facing, 1.0f, 0.7f * saturate(uniforms.sunDirection.y));
    }

    // Warm block light from lava: a squared falloff so it reads as a punchy
    // pool of orange near the source that fades quickly, added on top of the
    // sun/sky (it is emitted, so the cascade shadow does not gate it).
    constexpr float3 BLOCK_LIGHT_TINT = float3(1.0f, 0.55f, 0.22f);
    constexpr float BLOCK_LIGHT_STRENGTH = 1.5f;
    float3 blockLight = BLOCK_LIGHT_TINT * (in.vBlockLight * in.vBlockLight) * BLOCK_LIGHT_STRENGTH;

    // Baked corner AO darkens ambient and direct alike — the crease shading
    // reads as contact occlusion the same way vanilla applies it. The half-res
    // SSAO pass layers the broader mid-scale occlusion on top in screen space.
    float3 litColor =
        texColor.rgb * in.vAO *
        (uniforms.sunColor * direct * sky * lit + uniforms.ambientColor * sky + blockLight);

    // Rain-wet sheen: up-facing surfaces catch a moving sun gloss while wet.
    if (uniforms.wetness > 0.001f) {
        float3 sheenDir = reflect(-uniforms.sunDirection, in.vNormal);
        litColor += uniforms.sunColor * pow(saturate(dot(sheenDir, toCamera)), 32.0f) *
                    (0.45f * uniforms.wetness * lit * saturate(in.vNormal.y) * sky);
    }

    // Translucent foliage: the sun behind a leaf or blade glows through it
    // toward the camera (cheap subsurface term, shadow-gated so shaded
    // canopies don't self-illuminate).
    if (in.vFoliage > 0.5f) {
        float sss = pow(saturate(dot(-toCamera, uniforms.sunDirection)), 6.0f);
        litColor += texColor.rgb * uniforms.sunColor * (sss * 0.55f * lit * in.vSkyLight);
    }

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
        // Top surfaces ride the shared wave field; world-space input keeps the
        // waves continuous across chunk borders (the uniform tessellation emits
        // coincident border vertices, so both chunks displace them identically)
        pos.y += waterWaveHeight(pos.xz, water.time);
    }

    WaterVertexOutput out;
    float4 worldPos = uniforms.modelMatrix * float4(pos, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    return out;
}

// Cheap 2D value hash -> point in [0,1]^2, for the Voronoi caustic cells.
static float2 causticHash(float2 c) {
    float2 p = float2(dot(c, float2(127.1f, 311.7f)), dot(c, float2(269.5f, 183.3f)));
    return fract(sin(p) * 43758.5453f);
}

// Animated Voronoi caustic web. Each grid cell owns a feature point that
// drifts on a circle over time; the bright filaments sit where two cells are
// nearly equidistant (F2 - F1 small), which reads as refracted-sunlight
// caustics rather than the axis-aligned grid the old summed sines produced.
// Two octaves at different scale and drift keep it organic.
static float causticPattern(float2 p, float t) {
    float web = 0.0f;
    float amp = 1.0f;
    for (int layer = 0; layer < 2; ++layer) {
        float2 sp = p * (1.0f + 0.9f * float(layer)) + float2(3.7f * float(layer));
        float2 cell = floor(sp);
        float2 f = fract(sp);
        float f1 = 8.0f, f2 = 8.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float2 g = float2(dx, dy);
                float2 o = causticHash(cell + g);
                o = 0.5f + 0.5f * sin(t * (0.9f + 0.3f * float(layer)) + 6.2831853f * o);
                float d = length(g + o - f);
                if (d < f1) {
                    f2 = f1;
                    f1 = d;
                } else if (d < f2) {
                    f2 = d;
                }
            }
        }
        web += amp * smoothstep(0.16f, 0.0f, f2 - f1); // bright where cells meet
        amp *= 0.55f;
    }
    return pow(saturate(web), 1.4f);
}

// World position of a screen UV + its stored depth, via the camera inverse.
static float3 reconstructWorld(float2 uv, float depth, constant WaterUniforms& water) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    float4 world = water.invViewProjection * clip;
    return world.xyz / world.w;
}

// ---------------------------------------------------------------------------
// Screen-space reflection for water. Marches the reflected ray through world
// space, projecting each step to screen and comparing its device depth against
// the resolved opaque depth, so the far shore, trees, and terrain mirror in
// the surface. The forward projection already yields both the screen UV and
// the ray's device z, so the crossing test needs no per-step world
// reconstruction — only the final thickness reject reconstructs a world point.
// Returns rgb + a confidence in .a (0 when the ray misses, leaves the screen,
// or only finds sky); the caller falls back to the procedural-sky reflection.
// Depth is point-sampled: linear across a depth edge reads a value between two
// surfaces, so the crossing test would flicker.
// ---------------------------------------------------------------------------
static float4 traceWaterSSR(float3 origin, float3 dir, depth2d<float> sceneDepth,
                            texture2d<float> sceneColor, constant WaterUniforms& water) {
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    constexpr sampler colorLinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    const int STEPS = 24;
    float stride = 1.5f;
    float3 pos = origin;
    for (int i = 0; i < STEPS; ++i) {
        float3 prev = pos;
        pos += dir * stride;
        stride *= 1.13f; // grow the step so distant reflections stay cheap

        float4 clip = water.viewProjection * float4(pos, 1.0f);
        if (clip.w <= 0.0f) {
            break; // stepped behind the camera plane
        }
        float2 uv = (clip.xy / clip.w) * float2(0.5f, -0.5f) + 0.5f;
        if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
            break; // left the screen — no data to reflect
        }
        float sceneZ = sceneDepth.sample(depthPoint, uv);
        if (sceneZ >= 1.0f) {
            continue; // sky pixel: keep marching, maybe the ray dips into terrain
        }
        if (clip.z / clip.w <= sceneZ) {
            continue; // ray still in front of the visible surface
        }

        // Crossed behind the depth buffer — bisect prev..pos for the contact.
        float3 lo = prev, hi = pos;
        for (int j = 0; j < 6; ++j) {
            float3 mid = (lo + hi) * 0.5f;
            float4 mclip = water.viewProjection * float4(mid, 1.0f);
            float2 muv = (mclip.xy / mclip.w) * float2(0.5f, -0.5f) + 0.5f;
            if (mclip.z / mclip.w > sceneDepth.sample(depthPoint, muv)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        float4 hclip = water.viewProjection * float4(hi, 1.0f);
        float2 hitUV = (hclip.xy / hclip.w) * float2(0.5f, -0.5f) + 0.5f;
        // Reject a hit hiding far behind a thick occluder (a false crossing).
        float3 hitWorld = reconstructWorld(hitUV, sceneDepth.sample(depthPoint, hitUV), water);
        if (distance(water.cameraPosition, hi) - distance(water.cameraPosition, hitWorld) > 1.5f) {
            break;
        }
        // Fade as the hit nears a screen edge (data runs out there).
        float2 e = smoothstep(0.0f, 0.12f, hitUV) * smoothstep(0.0f, 0.12f, 1.0f - hitUV);
        return float4(sceneColor.sample(colorLinear, hitUV).rgb, e.x * e.y);
    }
    return float4(0.0f);
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
    float3 N = waterSurfaceNormal(in.vWorldPosition.xz, water.time);
    bool fromBelow = water.cameraUnderwater > 0.5f;

    // Reconstruct the opaque world position behind this fragment
    float3 behind = reconstructWorld(screenUV, opaqueDepth, water);
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
    float3 rworld = reconstructWorld(refractUV, refractDepth, water);
    float depthBelow = max(in.vWorldPosition.y - rworld.y, 0.0f);

    // ---- Caustics: refracted-light web on the shallow floor (~3.5 block cells)
    float caustic = causticPattern(rworld.xz * 0.28f, water.time) * exp(-depthBelow * 0.22f);
    refracted +=
        water.sunColor * caustic * 0.4f * in.vSkyLight * saturate(water.sunDirection.y * 2.0f);

    // ---- Absorption: shallow water reads turquoise and filters toward deep
    // blue with depth (red light dies first), the floor showing through shallows
    float3 shallowTint = float3(0.10f, 0.42f, 0.48f);
    float3 deepTint = float3(0.02f, 0.10f, 0.22f);
    float3 waterColor = mix(shallowTint, deepTint, saturate(waterDepth * 0.12f));
    float absorb = 1.0f - exp(-waterDepth * 0.16f);
    float3 body = mix(refracted * float3(0.75f, 0.92f, 0.96f), waterColor, absorb);

    // ---- Fresnel sky reflection + sun sparkle (skylight gates both, so
    // flooded caves reflect darkness rather than open sky)
    float3 R = reflect(-V, N);                // true reflection, for SSR marching
    float3 Rsky = float3(R.x, abs(R.y), R.z); // up-facing form for sky + sparkle
    float horizonBlend = pow(1.0f - saturate(Rsky.y), 2.0f);
    float3 skyReflection = mix(water.zenithColor, water.horizonColor, horizonBlend);

    // Screen-space reflection layered over the procedural sky: where the
    // reflected ray finds on-screen geometry (far shore, trees), mirror it;
    // elsewhere the sky term shows through. Skipped when looking up from below.
    if (water.ssrStrength > 0.0f && !fromBelow) {
        float4 ssr = traceWaterSSR(in.vWorldPosition, R, sceneDepth, sceneColor, water);
        skyReflection = mix(skyReflection, ssr.rgb, ssr.a * water.ssrStrength);
    }

    float sunAlign = saturate(dot(Rsky, water.sunDirection));
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

    // Shoreline foam: a bright animated band just off the shallow edge, gated
    // by skylight so flooded caves stay dark. Reuses the caustic web to break
    // the band into moving flecks rather than a hard rim.
    float foamBand =
        smoothstep(0.05f, 0.4f, waterDepth) * (1.0f - smoothstep(0.4f, 1.4f, waterDepth));
    float foam =
        foamBand * (0.35f + 0.65f * causticPattern(in.vWorldPosition.xz * 0.5f, water.time));
    color = mix(color, float3(0.92f, 0.96f, 1.0f), saturate(foam) * in.vSkyLight);

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
    float3 world = reconstructWorld(in.uv, depth, water);
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
