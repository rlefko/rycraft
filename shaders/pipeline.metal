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
    uint vFace [[flat]];
    uint vFarCanopy [[flat]];
    uint vFarSkirt [[flat]];
    uint vFarSkirtMask [[flat]];
    float vExactRadius [[flat]];
    float4 vOverlayColor;
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

// The far mesh deliberately overlaps the exact-radius boundary by whole
// 256-block tiles. Water and canopy summaries use this stable world-space
// handoff. Opaque terrain tops remain available as depth-tested fallback while
// exact surface meshes stream in.
static bool keepFarTerrainFragment(float3 worldPosition, float3 cameraPosition, float exactRadius) {
    if (exactRadius <= 0.0f) {
        return true;
    }
    const float horizontalDistance = distance(worldPosition.xz, cameraPosition.xz);
    if (horizontalDistance <= exactRadius) {
        return false;
    }
    if (horizontalDistance >= exactRadius + FAR_TERRAIN_HANDOFF_WIDTH_BLOCKS) {
        return true;
    }
    const float2 ditherCell = metal::floor(worldPosition.xz * 2.0f);
    const float dither = interleavedGradientNoise(ditherCell);
    return farTerrainHandoffVisible(horizontalDistance, exactRadius, dither);
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
    out.vOverlayColor = chunkOrigin.overlayColorAndStrength;

    // Pass through UV
    out.vUV = in.uv;

    // Unpack face normal (bits 0-2), texture layer (bits 3-10), column
    // skylight (bits 11-14), and baked corner AO (bits 15-16). The AO level
    // 0..3 maps to 0.5..1.0 — a fully enclosed voxel corner keeps half its
    // light, an open corner none removed.
    uint normalIdx = in.faceAttr & 7u;
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    out.vFace = normalIdx;
    out.vFarCanopy = (in.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0u;
    out.vFarSkirt = (in.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) != 0u;
    out.vFarSkirtMask = chunkOrigin.farMetadata.x;
    out.vExactRadius = chunkOrigin.origin.w;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    out.vAO = 0.5f + float((in.faceAttr >> 15) & 3u) * (1.0f / 6.0f);
    // Block light (bits 17-20) and the emissive flag (bit 21) — the lava glow.
    out.vBlockLight = float((in.faceAttr >> 17) & 15u) / 15.0f;
    out.vEmissive = float((in.faceAttr >> 21) & 1u);
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
    if (in.vFarCanopy != 0u &&
        !keepFarTerrainFragment(in.vWorldPosition, uniforms.cameraPosition, in.vExactRadius)) {
        discard_fragment();
    }
    const float horizontalDistance = distance(in.vWorldPosition.xz, uniforms.cameraPosition.xz);
    if (in.vFarSkirt != 0u) {
        const uint edgeBit = 1u << in.vFace;
        if ((in.vFarSkirtMask & edgeBit) == 0u ||
            !farTerrainSkirtVisible(horizontalDistance, in.vExactRadius)) {
            discard_fragment();
        }
    }
    // The bound sampler repeats each tile, keeps nearest magnification, and
    // applies trilinear anisotropic filtering during minification. UVs span the
    // quad extent in blocks, so each block gets one full texture tile.
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
        glow = applyFog(glow, in.vWorldPosition, uniforms.cameraPosition, uniforms.fogDensity,
                        uniforms.fogColor);
        glow = mix(glow, in.vOverlayColor.rgb, saturate(in.vOverlayColor.a));
        return float4(glow, 1.0f);
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
    finalColor = mix(finalColor, in.vOverlayColor.rgb, saturate(in.vOverlayColor.a));
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
    float3 vCameraRelativePosition;
    float vSkyLight;
    float4 vOverlayColor;
    uint vFace [[flat]];
    uint vFlow [[flat]];
    uint vFalling [[flat]];
    float vExactRadius [[flat]];
};

// Fluid direction values are packed by the mesher as still, west, east,
// north, and south. Keeping the lookup here makes the wave phase follow the
// actual simulated current without expanding the 16-byte vertex ABI.
static float2 waterFlowVector(uint flow) {
    switch (flow) {
        case 1u:
            return float2(-1.0f, 0.0f);
        case 2u:
            return float2(1.0f, 0.0f);
        case 3u:
            return float2(0.0f, -1.0f);
        case 4u:
            return float2(0.0f, 1.0f);
        default:
            return float2(0.0f);
    }
}

vertex WaterVertexOutput waterVertexMain(VertexInput in [[stage_in]],
                                         constant Uniforms& uniforms [[buffer(1)]],
                                         constant ChunkOrigin& chunkOrigin [[buffer(2)]],
                                         constant WaterUniforms& water [[buffer(3)]]) {
    float3 pos = in.position + chunkOrigin.origin.xyz;
    uint normalIdx = in.faceAttr & 7u;
    uint flow = (in.faceAttr >> 24) & 7u;
    uint falling = (in.faceAttr >> 27) & 1u;
    if (normalIdx == FACE_PLUS_Y) {
        // Top surfaces ride the shared wave field (one definition with the
        // fragment normal, see shader_types.hpp); world-space input keeps the
        // waves continuous across chunk borders, and flow advects the phase in
        // its packed cardinal direction so currents visibly travel while still
        // water keeps the resting interference. The midline is biased DOWN so
        // a full crest (rest plane 0.875 + peak ~0.21) stays below the block
        // top instead of washing over adjacent shoreline blocks, and thin
        // flowing sheets scale their swell by their own fluid level so a
        // near-empty cell never displaces below its floor.
        float levelFrac = fract(in.position.y);
        float waveScale = saturate(levelFrac * 1.143f); // 0.875 (full cell) -> 1
        float2 phase = pos.xz - waterFlowVector(flow) * water.time * 0.7f;
        pos.y += (waterWaveHeight(phase, water.time) - 0.10f) * waveScale;
    }

    WaterVertexOutput out;
    float4 worldPos = uniforms.modelMatrix * float4(pos, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vCameraRelativePosition = worldPos.xyz - water.cameraPosition;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    out.vOverlayColor = chunkOrigin.overlayColorAndStrength;
    out.vFace = normalIdx;
    out.vFlow = flow;
    out.vFalling = falling;
    out.vExactRadius = chunkOrigin.origin.w;
    return out;
}

// Water caustics: the flowing bright web sunlight focuses into as it refracts
// through moving water. This is the standard iterative domain-warped caustic
// (the look Sildur and most Minecraft shaders use): each iteration folds the
// coordinate through the previous one so the pattern converges into curved
// filaments that drift with the surface, rather than a static grid or cells.
// Driven by the water time and warped by the shared wave field so the light
// moves with the actual waves overhead.
// One octave of the iterative web, over a domain already scaled to radians.
// The wrap keeps the iteration numerically bounded at large world
// coordinates, which makes a single octave exactly periodic per tile.
static float causticOctave(float2 wp, float t, int iterations) {
    // GLSL-style positive wrap: MSL fmod follows the dividend's sign, which
    // would flip the pattern's anchor across the world origin.
    float2 p = wp - 6.28318f * floor(wp / 6.28318f) - 250.0f;
    float2 i = p;
    float c = 1.0f;
    const float inten = 0.005f;
    for (int n = 0; n < iterations; ++n) {
        float tt = t * (1.0f - (3.5f / float(n + 1)));
        i = p + float2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
        c += 1.0f / length(float2(p.x / (sin(i.x + tt) / inten), p.y / (cos(i.y + tt) / inten)));
    }
    c /= float(iterations);
    c = 1.17f - pow(c, 1.4f);
    return pow(abs(c), 8.0f);
}

// floorDepth is the shaded point's depth below the water surface in blocks.
// Physically the crisp web is focused by the short ripples (cell size tracks
// ripple wavelength) and the focus blurs away with distance from the surface,
// so shallow floors show fine sharp cells and deep floors only soft, large
// patches from the swells — a fixed web at every depth read as painted-on.
static float causticPattern(float2 worldXZ, float t, float floorDepth) {
    // Warp the lookup by the wave gradient (unscaled world xz, so it uses the
    // real per-block wave frequencies), scaled up so the web arms visibly
    // wiggle with the same ripples that focus them.
    float3 wn = waterSurfaceNormal(worldXZ, t);
    float2 warp = wn.xz * 3.0f;
    // One crisp web, MODULATED by a slow rotated octave. A single wrapped
    // octave is exactly periodic (a visible grid of identical ~2-block cells
    // covered every floor); the incommensurate rotated modulator varies the
    // web's brightness over a beat period of hundreds of blocks, so no
    // repetition survives to the eye — while the arms stay sharp (summing two
    // full webs blurred them into mush instead).
    const float freqA = 0.30f * 6.28318f; // ~3.3 block web cells (ripple scale)
    const float freqB = 0.11f * 6.28318f; // ~9 block modulation (swell scale)
    float2 pA = worldXZ * freqA + warp;
    float2 rot = float2(worldXZ.x * 0.7986f - worldXZ.y * 0.6018f,
                        worldXZ.x * 0.6018f + worldXZ.y * 0.7986f);
    float2 pB = rot * freqB + warp + float2(87.31f, -42.77f);
    float web = causticOctave(pA, t, 5);
    float modulation = saturate(causticOctave(pB, t * 0.7f, 3));
    // Defocus with depth: the crisp ripple web washes out over ~8 blocks,
    // leaving the broad swell-scale patches.
    float defocus = saturate(floorDepth * 0.12f);
    float focused = web * (0.5f + 0.9f * modulation);
    float diffuse = modulation * 0.8f;
    // Saturate: the web centers overshoot 1, and an unclamped HDR caustic times
    // its gain crossed the bloom threshold across whole floors (white-out).
    return saturate(mix(focused, diffuse, defocus));
}

// Camera-relative world position of a screen UV + its stored depth. Keeping
// the camera translation out of the inverse avoids catastrophic precision
// loss when water is rendered far from the world origin.
static float3 reconstructCameraRelative(float2 uv, float depth, constant WaterUniforms& water) {
    float4 clip = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, depth, 1.0f);
    float4 relative = water.invCameraRelativeViewProjection * clip;
    return relative.xyz / relative.w;
}

// ---------------------------------------------------------------------------
// Screen-space reflection for water. Marches the reflected ray through
// camera-relative world space, projecting each step to screen and comparing
// its device depth against the resolved opaque depth, so the far shore, trees,
// and terrain mirror in the surface. The forward projection already yields
// both the screen UV and the ray's device z, so the crossing test needs no
// per-step reconstruction. Only the final thickness reject reconstructs a
// camera-relative point.
// Returns rgb + a confidence in .a (0 when the ray misses, leaves the screen,
// or only finds sky); the caller falls back to the procedural-sky reflection.
// Depth is point-sampled: linear across a depth edge reads a value between two
// surfaces, so the crossing test would flicker.
// ---------------------------------------------------------------------------
// Per-channel water absorption (per block): red dies within a few blocks,
// green by ~15, blue last. ONE definition shared by the underwater overlay
// and the reflected-path attenuation in the SSR, so the water can never
// absorb differently along different ray types.
constant float3 WATER_SIGMA_A = float3(0.16f, 0.05f, 0.028f);

static float4 traceWaterSSR(float3 origin, float3 dir, float2 fragPx, bool underwater,
                            depth2d<float> sceneDepth, texture2d<float> sceneColor,
                            constant WaterUniforms& water) {
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    constexpr sampler colorLinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    const int STEPS = 24;
    // Jitter the first stride per pixel (the engine's IGN convention): a
    // coherent march start turned the coarse steps into visible stair bands
    // across every reflection. The range is deliberately narrow — wide jitter
    // traded the bands for salt-and-pepper speckle, and with MSAA instead of
    // a temporal history there is nothing downstream to average it away.
    float stride = 1.5f * (0.88f + 0.24f * interleavedGradientNoise(fragPx));
    float3 pos = origin;
    for (int i = 0; i < STEPS; ++i) {
        float3 prev = pos;
        pos += dir * stride;
        stride *= 1.13f; // grow the step so distant reflections stay cheap

        float4 clip = water.cameraRelativeViewProjection * float4(pos, 1.0f);
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
            float4 mclip = water.cameraRelativeViewProjection * float4(mid, 1.0f);
            float2 muv = (mclip.xy / mclip.w) * float2(0.5f, -0.5f) + 0.5f;
            if (mclip.z / mclip.w > sceneDepth.sample(depthPoint, muv)) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        float4 hclip = water.cameraRelativeViewProjection * float4(hi, 1.0f);
        float2 hitUV = (hclip.xy / hclip.w) * float2(0.5f, -0.5f) + 0.5f;
        // Reject a hit hiding far behind a thick occluder (a false crossing),
        // but keep marching rather than giving up: bailing out here flipped
        // neighboring jittered pixels between hit and sky fallback, which
        // read as speckle noise along every reflection silhouette.
        float3 hitRelative =
            reconstructCameraRelative(hitUV, sceneDepth.sample(depthPoint, hitUV), water);
        if (length(hi) - length(hitRelative) > 1.5f) {
            pos = hi;
            continue;
        }
        // Fade as the hit nears a screen edge (data runs out there).
        float2 e = smoothstep(0.0f, 0.12f, hitUV) * smoothstep(0.0f, 0.12f, 1.0f - hitUV);
        float3 hit = sceneColor.sample(colorLinear, hitUV).rgb;
        if (underwater) {
            // The reflected ray also travels through water: absorb its path
            // per channel, so distant mirrored geometry dims into the deep
            // instead of reflecting crisp daylight colors (also hides the
            // minification shimmer of far reflections).
            hit *= exp(-WATER_SIGMA_A * distance(origin, hi));
        }
        return float4(hit, e.x * e.y);
    }
    return float4(0.0f);
}

fragment float4 waterFragmentMain(WaterVertexOutput in [[stage_in]],
                                  texture2d<float> sceneColor [[texture(0)]],
                                  depth2d<float> sceneDepth [[texture(1)]],
                                  constant WaterUniforms& water [[buffer(3)]]) {
    if (!keepFarTerrainFragment(in.vWorldPosition, water.cameraPosition, in.vExactRadius)) {
        discard_fragment();
    }
    constexpr sampler screenSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 screenUV = in.clipPosition.xy / water.resolution;

    // Manual depth test against the resolved opaque scene
    float opaqueDepth = sceneDepth.sample(screenSampler, screenUV);
    if (in.clipPosition.z > opaqueDepth) {
        discard_fragment();
    }

    float3 V = normalize(-in.vCameraRelativePosition);
    // The fragment normal samples the same shared wave field the vertex stage
    // displaced with, at the same flow-advected phase, so shading follows the
    // moving geometry exactly.
    float2 flowOffset = waterFlowVector(in.vFlow) * water.time * 0.7f;
    float3 N = in.vFace == FACE_PLUS_Y
                   ? waterSurfaceNormal(in.vWorldPosition.xz - flowOffset, water.time)
                   : getFaceNormal(in.vFace);
    // Falling columns are the only water geometry allowed to expose vertical
    // sides. Give those faces a subtle downward streak normal so waterfalls
    // read as moving sheets while stable shorelines remain top surfaces only.
    if (in.vFalling != 0u && in.vFace != FACE_PLUS_Y && in.vFace != 5u) {
        float streak = sin(in.vWorldPosition.y * 5.5f - water.time * 8.0f +
                           dot(in.vWorldPosition.xz, float2(1.7f, 2.1f))) *
                       0.08f;
        N = normalize(N + float3(0.0f, streak, 0.0f));
    }
    bool fromBelow = water.cameraUnderwater > 0.5f;

    // Reconstruct both points in the same camera-relative frame. Absolute
    // inverse-view-projection matrices lose enough precision at large world
    // coordinates to turn this thickness into a visible chunk grid.
    float3 behindRelative = reconstructCameraRelative(screenUV, opaqueDepth, water);
    float waterDepth = max(distance(behindRelative, in.vCameraRelativePosition), 0.0f);

    // ---- Refraction: wave-distorted resample of the scene, pinned at the
    // shoreline so shallow edges don't smear. From below the distorted tap
    // crosses the surface boundary into unrelated above-water content, so the
    // transmission samples straight through instead.
    float distortion = fromBelow ? 0.0f : min(waterDepth, 4.0f) * 0.25f;
    float2 refractUV = clamp(screenUV + N.xz * 0.035f * distortion, 0.001f, 0.999f);
    float refractDepth = sceneDepth.sample(screenSampler, refractUV);
    if (refractDepth < in.clipPosition.z) {
        // The distorted tap landed on something in FRONT of the surface —
        // fall back to the undistorted sample
        refractUV = screenUV;
        refractDepth = opaqueDepth;
    }
    float3 refracted = sceneColor.sample(screenSampler, refractUV).rgb;

    // Reconstruct the refracted floor camera-relatively. Add the camera only
    // after depth math to anchor the low-frequency caustic pattern globally.
    float3 refractedRelative = reconstructCameraRelative(refractUV, refractDepth, water);
    float3 refractedWorld = refractedRelative + water.cameraPosition;
    float depthBelow = max(in.vCameraRelativePosition.y - refractedRelative.y, 0.0f);

    // ---- Caustics: refracted-light web on the shallow floor, seen from
    // above only. World-anchored (unscaled xz: causticPattern bakes its own
    // cell scale and wave warp). From below the reconstruction lands on
    // above-water content, which painted mis-oriented white bands onto the
    // transmission — the from-below floor gets its caustics from the overlay.
    if (!fromBelow) {
        float caustic =
            causticPattern(refractedWorld.xz, water.time, depthBelow) * exp(-depthBelow * 0.22f);
        refracted +=
            water.sunColor * caustic * 0.4f * in.vSkyLight * saturate(water.sunDirection.y * 2.0f);
    }

    // ---- Absorption: shallow water reads turquoise and filters toward deep
    // blue with depth (red light dies first), the floor showing through
    // shallows. From above, waterDepth is the submerged column behind the
    // surface. From below it is the distance to the sky or shore in the AIR
    // beyond the surface, which absorbs nothing — the underwater overlay
    // already absorbs the eye-to-surface water segment, so absorbing here
    // turned the whole Snell window into opaque flat blue instead of a view
    // of the world above.
    float3 shallowTint = float3(0.10f, 0.42f, 0.48f);
    float3 deepTint = float3(0.02f, 0.10f, 0.22f);
    float3 waterColor = mix(shallowTint, deepTint, saturate(waterDepth * 0.12f));
    float absorb = 1.0f - exp(-waterDepth * 0.16f);
    float3 body =
        fromBelow ? refracted : mix(refracted * float3(0.75f, 0.92f, 0.96f), waterColor, absorb);

    // ---- Fresnel reflection + sun sparkle (skylight gates both, so flooded
    // caves reflect darkness rather than open sky). From below the physics
    // flips: water-to-air refraction hits total internal reflection beyond the
    // critical angle (~48.6 deg), so the surface turns into a mirror of the
    // underwater scene (SSR provides it) instead of a window to the sky.
    float3 R = reflect(-V, N);                // true reflection, for SSR marching
    float3 Rsky = float3(R.x, abs(R.y), R.z); // up-facing form for sky + sparkle
    float cosI = saturate(dot(V, N));
    float fresnel;
    float3 reflection;
    if (fromBelow) {
        const float ETA = 1.33f; // water/air refractive index ratio
        float sinT2 = ETA * ETA * (1.0f - cosI * cosI);
        if (sinT2 >= 1.0f) {
            fresnel = 1.0f; // total internal reflection: pure mirror
        } else {
            // Schlick against the transmitted angle (the dense-side form),
            // eased into the mirror near the critical angle so per-quad wave
            // normals don't flip whole cells into hard-edged panels.
            const float R0 = 0.02f; // ((1.33-1)/(1.33+1))^2
            float cosT = sqrt(1.0f - sinT2);
            fresnel = R0 + (1.0f - R0) * pow(1.0f - cosT, 5.0f);
            fresnel = mix(fresnel, 1.0f, smoothstep(0.90f, 1.0f, sinT2));
        }
        // The mirror shows the underwater scene: SSR marches the downward
        // reflected ray; where it misses, the deep water body shows (the
        // bright shallow tint here read as glowing panels overhead).
        reflection = deepTint;
    } else {
        fresnel = mix(0.04f, 1.0f, pow(1.0f - cosI, 5.0f));
        float horizonBlend = pow(1.0f - saturate(Rsky.y), 2.0f);
        reflection = mix(water.zenithColor, water.horizonColor, horizonBlend);
    }

    // Screen-space reflection layered over the fallback: where the reflected
    // ray finds on-screen geometry (far shore and trees from above, the floor
    // under total internal reflection from below), mirror it.
    if (water.ssrStrength > 0.0f) {
        float4 ssr = traceWaterSSR(in.vCameraRelativePosition, R, in.clipPosition.xy, fromBelow,
                                   sceneDepth, sceneColor, water);
        reflection = mix(reflection, ssr.rgb, ssr.a * water.ssrStrength);
    }

    float3 color = mix(body, reflection, fresnel * (fromBelow ? 1.0f : in.vSkyLight));
    if (!fromBelow) {
        float sunAlign = saturate(dot(Rsky, water.sunDirection));
        float sparkle = pow(sunAlign, 240.0f) * 2.0f + pow(sunAlign, 32.0f) * 0.25f;
        // The glint is a specular reflection, so it obeys the same Fresnel as
        // the sky term: ~2% at normal incidence, rising toward grazing.
        // Unscaled, a zenith sun mirrored in every up-facing wave below the
        // camera and bloomed into one giant white blob on the surface.
        color += water.sunColor * sparkle * fresnel * in.vSkyLight;
    }

    color =
        applyFog(color, in.vWorldPosition, water.cameraPosition, water.fogDensity, water.fogColor);

    // Hairline shorelines dissolve into the shore instead of aliasing
    color = mix(refracted, color, saturate(waterDepth * 3.0f));
    // Shoreline foam: a bright animated band just off the shallow edge, gated
    // by skylight so flooded caves stay dark, and by the above-water view —
    // foam is surface froth, so from below it painted white streaks along the
    // waterline. Reuses the caustic web to break the band into moving flecks.
    if (!fromBelow) {
        float foamBand =
            smoothstep(0.05f, 0.4f, waterDepth) * (1.0f - smoothstep(0.4f, 1.4f, waterDepth));
        float foam =
            foamBand * (0.35f + 0.65f * causticPattern(in.vWorldPosition.xz, water.time, 0.0f));
        color = mix(color, float3(0.92f, 0.96f, 1.0f), saturate(foam) * in.vSkyLight);
    }

    color = mix(color, in.vOverlayColor.rgb, saturate(in.vOverlayColor.a));
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

// Dual-source blending: the pipeline blends result = inscatter + scene *
// transmit, so absorption can multiply the scene PER CHANNEL (Beer-Lambert)
// while the scattered light adds — a single alpha cannot express both.
struct UnderwaterOverlayOut {
    float4 inscatter [[color(0), index(0)]];
    float4 transmit [[color(0), index(1)]];
};

fragment UnderwaterOverlayOut underwaterOverlayFragment(OverlayVertexOutput in [[stage_in]],
                                                        depth2d<float> sceneDepth [[texture(1)]],
                                                        constant WaterUniforms& water
                                                        [[buffer(3)]]) {
    float t = water.time;
    float sunUp = saturate(water.sunDirection.y);

    // Camera-relative reconstruction: absolute inverse matrices lose precision
    // at large world coordinates (the same reason the water pass switched).
    constexpr sampler screenSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float depth = sceneDepth.sample(screenSampler, in.uv);
    float3 relative = reconstructCameraRelative(in.uv, depth, water);
    float3 world = relative + water.cameraPosition;
    float dist = length(relative);

    // ---- Physically based water: per-channel Beer-Lambert absorption
    // (WATER_SIGMA_A, shared with the SSR's reflected-path attenuation), so
    // distance shifts everything through teal into deep blue instead of
    // lerping toward one flat fog color. Absorption accumulates only along
    // the IN-WATER part of the ray: an upward ray exits after a few blocks.
    const float3 SIGMA_A = WATER_SIGMA_A;
    float3 rayDir = relative / max(dist, 1e-4f);
    float eyeY = water.cameraPosition.y;
    float exitDist = (rayDir.y > 0.02f)
                         ? max(water.waterSurfaceY - eyeY, 0.0f) / max(rayDir.y, 0.02f)
                         : 3.4e38f;
    float waterPath = min(dist, exitDist);

    // The light that reached the shaded point also crossed the water column
    // above it (longer when the sun sits low), so deep floors go dark, not
    // just blue. Points seen THROUGH the surface keep their above-water light,
    // and covered water (skyExposure 0) is cave-lit already — no double dark.
    float pointDepth = (dist <= exitDist + 0.5f)
                           ? clamp(water.waterSurfaceY - world.y, 0.0f, 48.0f)
                           : 0.0f;
    float lightSlant = 1.0f / max(sunUp, 0.35f);
    pointDepth *= water.skyExposure;

    // ---- Caustics on up-facing submerged surfaces (walls stay dark). The
    // water surface pass only shades pixels behind a quad, so the floor at
    // the player's feet would have none without this. The normal comes from
    // best-of-both-sides depth taps, not raw screen derivatives: a one-sided
    // derivative straddles block silhouettes and lit dashed lines along every
    // oblique edge (the same defect ssao.metal documents), while picking the
    // continuous side per axis reads the true surface at edges.
    float2 texel = 1.0f / water.resolution;
    float3 pL = reconstructCameraRelative(
        in.uv - float2(texel.x, 0.0f),
        sceneDepth.sample(screenSampler, in.uv - float2(texel.x, 0.0f)), water);
    float3 pR = reconstructCameraRelative(
        in.uv + float2(texel.x, 0.0f),
        sceneDepth.sample(screenSampler, in.uv + float2(texel.x, 0.0f)), water);
    float3 pD = reconstructCameraRelative(
        in.uv - float2(0.0f, texel.y),
        sceneDepth.sample(screenSampler, in.uv - float2(0.0f, texel.y)), water);
    float3 pU = reconstructCameraRelative(
        in.uv + float2(0.0f, texel.y),
        sceneDepth.sample(screenSampler, in.uv + float2(0.0f, texel.y)), water);
    float spanL = abs(dist - length(pL)), spanR = abs(length(pR) - dist);
    float spanD = abs(dist - length(pD)), spanU = abs(length(pU) - dist);
    float3 ddxv = (spanR < spanL) ? (pR - relative) : (relative - pL);
    float3 ddyv = (spanU < spanD) ? (pU - relative) : (relative - pD);
    float3 surfaceNormal = normalize(cross(ddxv, ddyv));
    // The depth falloff is gentle (0.03/block) so the pool floor several
    // blocks down still catches a bright web, not only the near-surface cells.
    float upFacing = saturate(abs(surfaceNormal.y));
    // Feather true silhouettes: when even the continuous side jumps more than
    // a surface at this distance could, the pixel straddles two surfaces and
    // neither normal is trustworthy — fade rather than sparkle the block edge.
    float edgeSpan = max(min(spanL, spanR), min(spanD, spanU));
    upFacing *= 1.0f - smoothstep(dist * 0.04f + 0.15f, dist * 0.08f + 0.5f, edgeSpan);
    // Caustics land only on submerged floors: refracted sunlight never lights
    // shore terrain reconstructed BEHIND the from-below surface (whose pixels
    // the surface pass already shaded), and the focusing decays with depth.
    float submerged = step(world.y, water.waterSurfaceY);
    float throughSurface = 1.0f - smoothstep(exitDist * 0.9f, exitDist * 1.1f, dist);
    float caustic = causticPattern(world.xz, t, pointDepth) * exp(-pointDepth * 0.05f) *
                    upFacing * submerged *
                    throughSurface * sunUp * water.skyExposure;

    // ---- Transmittance (dual-source color 1): the scene is multiplied per
    // channel by the view-path and light-path absorption, and the caustic
    // MODULATES that light instead of adding white — it rides the floor's own
    // shading, so shadowed floors get proportionally dimmer webs.
    float3 transmit = exp(-SIGMA_A * (waterPath + pointDepth * lightSlant)) *
                      (1.0f + caustic * 1.8f);

    // ---- Inscatter (dual-source color 0): sunlight scattered into the view
    // ray by the water itself. Henyey-Greenstein makes looking toward the sun
    // visibly brighter (the underwater silver lining); the light available in
    // the volume decays with the camera's own depth per channel; a tiny
    // ambient floor keeps covered water from reading as a void.
    float camDepth = max(water.waterSurfaceY - eyeY, 0.0f);
    float cosSun = dot(rayDir, normalize(water.sunDirection));
    const float g = 0.45f;
    float phase = (1.0f - g * g) / pow(1.0f + g * g - 2.0f * g * cosSun, 1.5f); // 1 = isotropic
    float phaseN = phase / (1.0f + phase); // capped so the sun lobe brightens, never blows out
    const float3 SCATTER_COLOR = float3(0.02f, 0.10f, 0.17f);
    float3 volLight =
        water.sunColor * sunUp * water.skyExposure * exp(-SIGMA_A * camDepth * 0.7f);
    float buildup = 1.0f - exp(-0.15f * waterPath);
    float3 inscatter = SCATTER_COLOR * volLight * (0.55f + 0.9f * phaseN) * buildup;
    inscatter += float3(0.004f, 0.012f, 0.02f) * buildup;

    UnderwaterOverlayOut out;
    out.inscatter = float4(inscatter, 1.0f);
    out.transmit = float4(transmit, 1.0f);
    return out;
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
