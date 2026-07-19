#include <metal_stdlib>
#include <render/shader_types.hpp>
#include <render/shadow_sampling.hpp>
using namespace metal;

// ---------------------------------------------------------------------------
// Vertex input, bound through the vertex descriptor (not argument buffer)
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
    float vSkyLight;          // propagated skylight 0-1 (ambient access)
    float vAO;                // baked corner AO 0.5-1 (per-vertex crease shading)
    float vBlockLight;        // lava block light 0-1 (warm cave glow)
    float vEmissive;          // 1 = self-lit block (lava), 0 = normally lit
    float vFoliage;           // shading class: 0 solid, 1 = cross-quad flora
                              // (two-sided facing + SSS), 2 = leaf cube (SSS only)
    float3 vWorldPosition;    // Displayed world-space position for fog and light
    float2 vLodWorldPosition; // Unswung world X/Z for stable LOD ownership
    float2 vFarLocalPosition;
    uint vTextureLayer [[flat]];
    uint vFace [[flat]];
    uint vFarCanopy [[flat]];
    uint vFarSkirt [[flat]];
    uint vFarSkirtMask [[flat]];
    uint vFarTerrain [[flat]];
    float vLodTransitionProgress [[flat]];
    float vCoverageFrontier [[flat]];
    float4 vOverlayColor;
};

// Face indices, must match FaceNormal in include/render/vertex.hpp
constant uint FACE_PLUS_Y = 4u;
constant uint FACE_CROSS = 6u;

// Shared exponential distance fog, one definition so the terrain, water,
// and entity passes can never drift apart.
static float3 applyFog(float3 color, float3 worldPos, float3 cameraPos, float density,
                       float3 fogColor) {
    float dist = distance(worldPos, cameraPos);
    float fogFactor = clamp(1.0f - exp(-density * dist), 0.0f, 1.0f);
    return mix(color, fogColor, fogFactor);
}

static float cloudShadowVisibility(float3 worldPosition, texture2d<float> cloudShadow,
                                   constant CloudShadowUniforms& cloud) {
    if (cloud.footprintAndTexel.x <= 0.0f) {
        return 1.0f;
    }
    constexpr sampler cloudSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 center =
        floor(cloud.cameraPosition.xz / cloud.footprintAndTexel.y) * cloud.footprintAndTexel.y;
    const float2 referencePosition =
        cloudShadowReferencePosition(worldPosition, normalize(cloud.sunDirection));
    const float2 uv = (referencePosition - center) / cloud.footprintAndTexel.x + 0.5f;
    if (any(uv < 0.0f) || any(uv > 1.0f)) {
        return 1.0f;
    }
    return mix(1.0f, cloudShadow.sample(cloudSampler, uv).r, saturate(cloud.footprintAndTexel.z));
}

// Far meshes deliberately overlap the exact radius by whole 256-block tiles.
// Revision-matched column ownership below is the authority for that overlap:
// a ready column clips immediately, while an unresolved or stale column keeps
// complete coarse coverage even when it lies inside the nominal radius.
// ---------------------------------------------------------------------------
// Face normal lookup, indices match FaceNormal enum (0=+X … 5=-Y)
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
// Vertex shader, passes world position for fog calculation
// ---------------------------------------------------------------------------
vertex VertexOutput vertexMain(VertexInput in [[stage_in]],
                               constant Uniforms& uniforms [[buffer(1)]],
                               constant ChunkOrigin& chunkOrigin [[buffer(2)]]) {
    VertexOutput out;

    // Restore world space from the chunk-local position, sway foliage in the
    // wind (bits 22-23; the shadow pass applies the same displacement), then MVP
    float4 worldPos = uniforms.modelMatrix * float4(in.position + chunkOrigin.origin.xyz, 1.0);
    const float2 lodWorldPosition = worldPos.xz;
    uint sway = (in.faceAttr >> 22) & 3u;
    worldPos.xyz = applySway(worldPos.xyz, sway, in.uv.y, uniforms.time, uniforms.foliageWind);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vLodWorldPosition = lodWorldPosition;
    out.vFarLocalPosition = in.position.xz;
    out.vOverlayColor = chunkOrigin.overlayColorAndStrength;

    // Pass through UV
    out.vUV = in.uv;

    // Unpack face normal (bits 0-2), texture layer (bits 3-10), column
    // skylight (bits 11-14), and baked corner AO (bits 15-16). The AO level
    // 0..3 maps to 0.5..1.0, a fully enclosed voxel corner keeps half its
    // light, an open corner none removed.
    uint normalIdx = in.faceAttr & 7u;
    out.vTextureLayer = (in.faceAttr >> 3) & 0xFFu;
    out.vFace = normalIdx;
    out.vFarCanopy = (in.faceAttr & FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK) != 0u;
    out.vFarSkirt = (in.faceAttr & FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK) != 0u;
    out.vFarSkirtMask = chunkOrigin.farMetadata.x;
    out.vFarTerrain = chunkOrigin.farMetadata.w;
    out.vLodTransitionProgress = as_type<float>(chunkOrigin.farMetadata.z);
    out.vCoverageFrontier = as_type<float>(chunkOrigin.farMetadata.y);
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    out.vAO = 0.5f + float((in.faceAttr >> 15) & 3u) * (1.0f / 6.0f);
    // Block light (bits 17-20) and the emissive flag (bit 21), the lava glow.
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

// ---------------------------------------------------------------------------
// Fragment shader, sun/moon shadows + skylight + distance fog
// ---------------------------------------------------------------------------
struct SurfaceFragmentOutput {
    float4 scene [[color(0)]];
    float4 surface [[color(1)]];
};

fragment SurfaceFragmentOutput fragmentMain(
    VertexOutput in [[stage_in]], texture2d_array<float> blockTextures [[texture(0)]],
    sampler blockSampler [[sampler(0)]], depth2d_array<float> nearShadow [[texture(1)]],
    depth2d_array<float> farShadow [[texture(2)]], depth2d<float> horizonShadow [[texture(3)]],
    texture2d<float> cloudShadow [[texture(4)]], sampler shadowSampler [[sampler(1)]],
    constant Uniforms& uniforms [[buffer(1)]], constant ShadowUniforms& shadow [[buffer(4)]],
    constant FarTerrainOwnershipUniforms& ownership [[buffer(5)]],
    constant CloudShadowUniforms& cloudShadowUniforms [[buffer(6)]]) {
    const float horizontalDistance = distance(in.vLodWorldPosition, uniforms.cameraPosition.xz);
    if (!farTerrainCoverageVisible(horizontalDistance, in.vCoverageFrontier)) {
        discard_fragment();
    }
    const bool useEmittingColumn =
        farTerrainOpaqueRiserUsesEmittingColumn(in.vFace, in.vFarCanopy != 0u, in.vFarSkirt != 0u);
    if (in.vFarTerrain != 0u) {
        bool exactOwnsFragment = farTerrainExactColumnOwnsFragment(in.vFarLocalPosition, in.vFace,
                                                                   useEmittingColumn, ownership);
        if (in.vFarSkirt != 0u) {
            const bool emittingColumnOwnedByExact =
                farTerrainExactColumnOwnsFragment(in.vFarLocalPosition, in.vFace, true, ownership);
            const float2 receivingPosition =
                farTerrainSkirtReceivingOwnershipSamplePosition(in.vFarLocalPosition, in.vFace);
            const bool receivingColumnOwnedByExact =
                farTerrainExactColumnOwnsFragment(receivingPosition, in.vFace, false, ownership);
            exactOwnsFragment = !farTerrainSkirtOwnersVisible(emittingColumnOwnedByExact,
                                                              receivingColumnOwnedByExact);
        }
        if (exactOwnsFragment) {
            discard_fragment();
        }
    }
    const float lodThreshold = interleavedGradientNoise(floor(in.vLodWorldPosition));
    const bool lodVisible =
        in.vFarSkirt != 0u ? farTerrainLodSkirtVisible(in.vLodTransitionProgress, in.vFarTerrain)
        : in.vFarCanopy != 0u
            ? farTerrainLodCanopyVisible(in.vLodTransitionProgress, lodThreshold, in.vFarTerrain)
            : farTerrainLodTerrainVisible(in.vLodTransitionProgress, in.vFarTerrain);
    if (!lodVisible) {
        discard_fragment();
    }
    const float coverageFog = farTerrainCoverageFog(horizontalDistance, in.vCoverageFrontier);
    const float lodTerrainFog =
        in.vFarCanopy == 0u && in.vFarSkirt == 0u
            ? farTerrainLodTerrainFog(in.vLodTransitionProgress, in.vFarTerrain)
            : 0.0f;
    if (in.vFarSkirt != 0u) {
        const uint edgeBit = 1u << in.vFace;
        if ((in.vFarSkirtMask & edgeBit) == 0u) {
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
    // and skylight, the >1.0 output exceeds the bloom threshold naturally and
    // makes lava the light source the block-light term below spreads from.
    // Kept modest so the molten orange survives tonemapping instead of
    // clipping to white when auto-exposure lifts a dark scene.
    constexpr float EMISSIVE_BOOST = 3.0f;
    if (in.vEmissive > 0.5f) {
        float3 glow = texColor.rgb * EMISSIVE_BOOST;
        glow = applyFog(glow, in.vWorldPosition, uniforms.cameraPosition, uniforms.fogDensity,
                        uniforms.fogColor);
        glow = mix(glow, in.vOverlayColor.rgb, max(saturate(in.vOverlayColor.a), coverageFog));
        SurfaceFragmentOutput result;
        result.scene = float4(glow, 1.0f);
        result.surface = float4(texColor.rgb, 0.0f);
        return result;
    }

    // Rain-soaked surfaces darken (water fills the surface pores), scaled
    // by sky access so caves and covered builds stay dry during surface rain.
    texColor.rgb *= 1.0f - 0.22f * uniforms.wetness * in.vSkyLight;

    // Propagated skylight controls ambient access. Direct sun is controlled by
    // geometry orientation and the shadow map, so an overhead log cannot
    // create a second baked column shadow.
    float sky = in.vSkyLight;

    // The real cascade shadow gates direct sun. Outside valid shadow coverage,
    // propagated exterior access prevents distant sealed spaces from leaking
    // directional light.
    float lit = sampleShadowVisibility(in.vWorldPosition, in.vNormal, in.vSkyLight, nearShadow,
                                       farShadow, horizonShadow, shadowSampler, shadow);
    lit *= cloudShadowVisibility(in.vWorldPosition, cloudShadow, cloudShadowUniforms);

    // Two-sided flora shading: the geometric normal of the visible blade side
    // (from screen-space derivatives, flipped toward the camera) modulates the
    // direct sun, so the face turned away from the sun reads as self-shaded
    // instead of matching the sunlit side exactly. The term relaxes toward 1
    // as the sun climbs: vertical blades are edge-on to a high sun, so at noon
    // BOTH sides of every blade scored ~0.4 and all flora went uniformly dark
    //, the facing contrast only means something at low sun angles.
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

    // Baked corner AO is ambient accessibility, not a shadow on direct or
    // emitted light. Block light remains independent of both ambient terms.
    float ambientAccess = sky * in.vAO;
    float3 litColor = texColor.rgb * (uniforms.sunColor * direct * lit +
                                      uniforms.ambientColor * ambientAccess + blockLight);

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
        litColor += texColor.rgb * uniforms.sunColor * (sss * 0.55f * lit);
    }

    float3 finalColor = applyFog(litColor, in.vWorldPosition, uniforms.cameraPosition,
                                 uniforms.fogDensity, uniforms.fogColor);
    finalColor = mix(finalColor, in.vOverlayColor.rgb,
                     max(max(saturate(in.vOverlayColor.a), coverageFog), lodTerrainFog));
    SurfaceFragmentOutput result;
    result.scene = float4(finalColor, 1.0f);
    result.surface = float4(texColor.rgb, ambientAccess);
    return result;
}

// ---------------------------------------------------------------------------
// Water pass, runs after the opaque scene resolves, compositing its own
// pixels from the resolved color + depth: screen-space refraction with
// depth-based absorption, procedural caustics on the submerged floor,
// fresnel sky reflection with a sun sparkle, and animated waves. No depth
// attachment: the fragment depth-tests manually against the resolved depth.
// ---------------------------------------------------------------------------
struct WaterVertexOutput {
    float4 clipPosition [[position]];
    float3 vWorldPosition;
    float2 vFarLocalPosition;
    float3 vCameraRelativePosition;
    float vSkyLight;             // propagated ambient accessibility
    float vExteriorSky [[flat]]; // binary water-interface authority
    float4 vOverlayColor;
    uint vFace [[flat]];
    uint vFlow [[flat]];
    uint vFalling [[flat]];
    float vCoverageFrontier [[flat]];
    uint vFarTerrain [[flat]];
    float vLodTransitionProgress [[flat]];
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
    // Water geometry stays on the authored fluid plane. Motion remains in the
    // analytic fragment normal below, so reflections animate without changing
    // shoreline clearance or exposing mesh diagonals.

    WaterVertexOutput out;
    float4 worldPos = uniforms.modelMatrix * float4(pos, 1.0);
    out.clipPosition = uniforms.projectionMatrix * uniforms.viewMatrix * worldPos;
    out.vWorldPosition = worldPos.xyz;
    out.vFarLocalPosition = in.position.xz;
    out.vCameraRelativePosition = worldPos.xyz - water.cameraPosition;
    out.vSkyLight = float((in.faceAttr >> 11) & 15u) / 15.0f;
    out.vExteriorSky = (in.faceAttr & (1u << 30u)) != 0u ? 1.0f : 0.0f;
    out.vOverlayColor = chunkOrigin.overlayColorAndStrength;
    out.vFace = normalIdx;
    out.vFlow = flow;
    out.vFalling = falling;
    out.vCoverageFrontier = as_type<float>(chunkOrigin.farMetadata.y);
    out.vFarTerrain = chunkOrigin.farMetadata.w;
    out.vLodTransitionProgress = as_type<float>(chunkOrigin.farMetadata.z);
    return out;
}

// Analytic normal of the shared water-wave field, filtered in screen space.
// The geometry remains flat, but flow still advects the normal's phase. Each
// band fades before it advances too far between neighboring pixels, preventing
// distant ripple aliasing without changing the authored fluid surface.
static float3 filteredWaterSurfaceNormal(float2 p, float t) {
    float2 gradient = float2(0.0f);
    for (int i = 0; i < 3; ++i) {
        WaterWave wave = WATER_WAVES[i];
        const float phase = dot(p, wave.dir) * wave.freq + t * wave.speed;
        const float footprint = length(float2(dfdx(phase), dfdy(phase)));
        gradient += wave.amp * wave.freq * wave.dir * cos(phase) * waterBandVisibility(footprint);
    }

    const float fineX = p.x * 1.9f + t * 2.3f;
    const float fineZ = p.y * 2.2f - t * 2.0f;
    const float diagonalX = (p.x + p.y) * 3.7f + t * 3.1f;
    const float diagonalZ = (p.y - p.x) * 3.3f - t * 2.7f;
    gradient.x +=
        0.014f * cos(fineX) * waterBandVisibility(length(float2(dfdx(fineX), dfdy(fineX))));
    gradient.y +=
        0.014f * cos(fineZ) * waterBandVisibility(length(float2(dfdx(fineZ), dfdy(fineZ))));
    gradient.x += 0.007f * cos(diagonalX) *
                  waterBandVisibility(length(float2(dfdx(diagonalX), dfdy(diagonalX))));
    gradient.y += 0.007f * cos(diagonalZ) *
                  waterBandVisibility(length(float2(dfdx(diagonalZ), dfdy(diagonalZ))));

    const float slope = 1.5f;
    return normalize(float3(-gradient.x * slope, 1.0f, -gradient.y * slope));
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
// patches from the swells, a fixed web at every depth read as painted-on.
static float causticPattern(float2 worldXZ, float t, float floorDepth) {
    // Warp the lookup by the wave gradient (unscaled world xz, so it uses the
    // real per-block wave frequencies), scaled up so the web arms visibly
    // wiggle with the same ripples that focus them.
    float3 wn = filteredWaterSurfaceNormal(worldXZ, t);
    float2 warp = wn.xz * 3.0f;
    // One crisp web, MODULATED by a slow rotated octave. A single wrapped
    // octave is exactly periodic (a visible grid of identical ~2-block cells
    // covered every floor); the incommensurate rotated modulator varies the
    // web's brightness over a beat period of hundreds of blocks, so no
    // repetition survives to the eye, while the arms stay sharp (summing two
    // full webs blurred them into mush instead).
    const float freqA = 0.30f * 6.28318f; // ~3.3 block web cells (ripple scale)
    const float freqB = 0.11f * 6.28318f; // ~9 block modulation (swell scale)
    float2 pA = worldXZ * freqA + warp;
    float2 rot = float2(worldXZ.x * 0.7986f - worldXZ.y * 0.6018f,
                        worldXZ.x * 0.6018f + worldXZ.y * 0.7986f);
    float2 pB = rot * freqB + warp + float2(87.31f, -42.77f);
    const float webFootprint = max(length(dfdx(pA)), length(dfdy(pA)));
    const float modulationFootprint = max(length(dfdx(pB)), length(dfdy(pB)));
    float web = causticOctave(pA, t, 5) * waterBandVisibility(webFootprint);
    float modulation =
        saturate(causticOctave(pB, t * 0.7f, 3)) * waterBandVisibility(modulationFootprint);
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

// Water volume scatter color and ambient floor. ONE definition shared by the
// underwater overlay's inscatter and the total-internal-reflection fallback
// in the water pass, so mirrored water can never glow a different color than
// the volume it reflects.
constant float3 WATER_SCATTER = float3(0.02f, 0.10f, 0.17f);
constant float3 WATER_AMBIENT = float3(0.004f, 0.012f, 0.02f);

// Schlick Fresnel for the air/water interface. From the dense (water) side
// the lobe is evaluated against the TRANSMITTED angle: cosT reaches zero
// exactly at the critical angle, so the reflectance rises continuously to
// the total-internal-reflection mirror with no hand-tuned ease, at a
// fraction of the exact dielectric form's cost.
static float waterFresnel(float cosI, bool fromWater) {
    const float R0 = 0.02f; // ((1.33 - 1) / (1.33 + 1))^2, both directions
    if (fromWater) {
        const float ETA = 1.33f; // water to air
        float sinT2 = ETA * ETA * (1.0f - cosI * cosI);
        if (sinT2 >= 1.0f) {
            return 1.0f; // total internal reflection
        }
        cosI = sqrt(1.0f - sinT2); // Schlick against the transmitted angle
    }
    return R0 + (1.0f - R0) * pow(1.0f - cosI, 5.0f);
}

static float4 traceWaterSSR(float3 origin, float3 dir, float2 fragPx, bool underwater,
                            depth2d<float> sceneDepth, texture2d<float> sceneColor,
                            constant WaterUniforms& water) {
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    constexpr sampler colorLinear(coord::normalized, address::clamp_to_edge, filter::linear,
                                  mip_filter::linear);

    const int STEPS = 24;
    // A fixed stride makes shallow crossings form coherent stair bands. IGN
    // breaks those bands, but at a long grazing path independent jitter
    // becomes a black-and-bright checker because this pass has no history.
    // Taper that bounded jitter as the reflected ray approaches horizontal;
    // mip filtering and the confidence below handle the remaining far tail.
    const float grazing = 1.0f - saturate(abs(dir.y));
    const float jitter = interleavedGradientNoise(fragPx) - 0.5f;
    float stride = 1.5f * (1.0f + jitter * waterSsrJitterAmplitude(grazing));
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
            break; // left the screen, no data to reflect
        }
        float sceneZ = sceneDepth.sample(depthPoint, uv);
        if (sceneZ >= 1.0f) {
            continue; // sky pixel: keep marching, maybe the ray dips into terrain
        }
        if (clip.z / clip.w <= sceneZ) {
            continue; // ray still in front of the visible surface
        }

        // Crossed behind the depth buffer, bisect prev..pos for the contact.
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
        const float hitDistance = distance(origin, hi);
        // A full-resolution source is correct for nearby reflections, but it
        // aliases badly when a grazing ray compresses distant terrain into a
        // few pixels. The copied HDR texture has a complete mip chain, so
        // choose a bounded blur from ray angle and path length rather than
        // sampling unrelated sharp terrain per pixel.
        const float mipLevel = waterSsrReflectionMipLevel(grazing, hitDistance);
        float3 hit = sceneColor.sample(colorLinear, hitUV, level(mipLevel)).rgb;
        if (underwater) {
            // The reflected ray also travels through water: absorb its path
            // per channel, so distant mirrored geometry dims into the deep
            // instead of reflecting crisp daylight colors (also hides the
            // minification shimmer of far reflections).
            hit *= exp(-WATER_SIGMA_A * hitDistance);
        }
        // Screen-space depth cannot represent a stable reflection at the
        // horizon after the ray has traveled far enough. Fade only that
        // unstable region back to the analytic atmosphere reflection instead
        // of toggling neighboring pixels between a dark hit and a sky miss.
        return float4(hit, e.x * e.y * waterSsrStabilityConfidence(grazing, hitDistance));
    }
    return float4(0.0f);
}

fragment float4 waterFragmentMain(
    WaterVertexOutput in [[stage_in]], texture2d<float> sceneColor [[texture(0)]],
    depth2d<float> sceneDepth [[texture(1)]], texture2d<float> atmosphereSky [[texture(2)]],
    texture2d<float> cloudShadow [[texture(3)]], depth2d_array<float> nearShadow [[texture(4)]],
    depth2d_array<float> farShadow [[texture(5)]], depth2d<float> horizonShadow [[texture(6)]],
    sampler shadowSampler [[sampler(1)]], constant WaterUniforms& water [[buffer(3)]],
    constant ShadowUniforms& shadow [[buffer(4)]],
    constant FarTerrainOwnershipUniforms& ownership [[buffer(5)]],
    constant CloudShadowUniforms& cloudShadowUniforms [[buffer(6)]]) {
    const float horizontalDistance = distance(in.vWorldPosition.xz, water.cameraPosition.xz);
    if ((in.vFarTerrain != 0u &&
         farTerrainExactColumnOwnsFragment(in.vFarLocalPosition, in.vFace, false, ownership)) ||
        !farTerrainCoverageVisible(horizontalDistance, in.vCoverageFrontier)) {
        discard_fragment();
    }
    if (!farTerrainLodConnectedGeometryVisible(in.vFarTerrain)) {
        discard_fragment();
    }
    const float coverageFog = farTerrainCoverageFog(horizontalDistance, in.vCoverageFrontier);
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    constexpr sampler colorLinear(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    float2 screenUV = in.clipPosition.xy / water.resolution;

    // Depth is point sampled throughout this pass. Linear depth interpolation
    // invents a sloping floor at voxel silhouettes, which turns far greedy
    // cells and exact/far ownership changes into rectangular absorption bands.
    float opaqueDepth = sceneDepth.sample(depthPoint, screenUV);
    if (in.clipPosition.z > opaqueDepth) {
        discard_fragment();
    }

    float3 V = normalize(-in.vCameraRelativePosition);
    // The fragment normal samples the shared wave field at a flow-advected
    // phase. It supplies filtered motion while the authored voxel plane stays
    // geometrically flat.
    float2 flowOffset = waterFlowVector(in.vFlow) * water.time * 0.7f;
    float surfaceDetail = 1.0f;
    float3 N = in.vFace == FACE_PLUS_Y
                   ? filteredWaterSurfaceNormal(in.vWorldPosition.xz - flowOffset, water.time)
                   : getFaceNormal(in.vFace);
    if (in.vFace == FACE_PLUS_Y) {
        // At a glancing view the reflection projection magnifies a tiny
        // normal change into long horizontal bands. The filtered normal above
        // removes frequency aliases; this view-aware attenuation also keeps
        // the stable low-frequency sky fallback from being striped by a flat
        // surface's shading-only ripples.
        surfaceDetail = waterGrazingWaveDetail(abs(V.y));
        N = normalize(mix(float3(0.0f, 1.0f, 0.0f), N, surfaceDetail));
        // A low-frequency wave can still be under-sampled after the reflection
        // projection, even when its world-space phase passed the filter above.
        // Derive this last gate from the reflected ray itself, which catches
        // only adjacent pixels that would select unrelated sky or SSR samples.
        const float3 prefilteredReflection = reflect(-V, N);
        const float reflectedRayFootprint =
            max(length(dfdx(prefilteredReflection)), length(dfdy(prefilteredReflection)));
        const float reflectionDetail = waterReflectionNormalVisibility(reflectedRayFootprint);
        surfaceDetail *= reflectionDetail;
        N = normalize(mix(float3(0.0f, 1.0f, 0.0f), N, reflectionDetail));
    }
    // Falling columns are the only water geometry allowed to expose vertical
    // sides. Give those faces a subtle downward streak normal so waterfalls
    // read as moving sheets while stable shorelines remain top surfaces only.
    if (in.vFalling != 0u && in.vFace != FACE_PLUS_Y && in.vFace != 5u) {
        const float streakPhase = in.vWorldPosition.y * 5.5f - water.time * 8.0f +
                                  dot(in.vWorldPosition.xz, float2(1.7f, 2.1f));
        const float streakFootprint = length(float2(dfdx(streakPhase), dfdy(streakPhase)));
        float streak = sin(streakPhase) * 0.08f * waterBandVisibility(streakFootprint);
        N = normalize(N + float3(0.0f, streak, 0.0f));
    }
    // Packed skylight stays ambient-only. The separate water bit preserves
    // exterior reflection and shadow-coverage fallback while a vertical light
    // path is conservatively unresolved, without treating the 4-bit value as
    // a fractional reflection multiplier.
    const float exteriorSkyVisibility = waterExteriorSkyVisibility(in.vExteriorSky);
    const float terrainVisibility =
        sampleShadowVisibility(in.vWorldPosition, N, max(in.vSkyLight, exteriorSkyVisibility),
                               nearShadow, farShadow, horizonShadow, shadowSampler, shadow);
    const float cloudVisibility =
        cloudShadowVisibility(in.vWorldPosition, cloudShadow, cloudShadowUniforms);
    const float directVisibility = terrainVisibility * cloudVisibility;
    // The binary exterior authority distinguishes sealed caves from open
    // water without making temporary propagated-light differences visible at
    // an exact or far handoff.
    // Which side of the interface this fragment shows is a per-fragment
    // geometric fact, not a camera flag: an elevated lake's underside seen
    // from dry land is still the water-to-air interface. Branching on the
    // camera alone gave such surfaces air-side Fresnel against a backfacing
    // normal, which saturated into a full sky mirror.
    float NdotV = dot(V, N);
    bool underside = NdotV < 0.0f;
    float cosI = saturate(abs(NdotV));
    // The camera flag still decides what medium the eye and the SSR's
    // reflected rays travel through.
    bool fromBelow = water.cameraUnderwater > 0.5f;

    // Reconstruct both points in the same camera-relative frame. Absolute
    // inverse-view-projection matrices lose enough precision at large world
    // coordinates to turn this thickness into a visible chunk grid.
    // A clear scene-depth pixel has no submerged receiver. Treating the far
    // plane as a refracted floor made horizon pixels borrow arbitrary scene
    // color, which exposed large water panes whenever streaming or a terrain
    // silhouette changed. It is a reflection-only case instead.
    const bool hasOpaqueReceiver = opaqueDepth < 0.99999f;
    const float waterSurfaceDistance = length(in.vCameraRelativePosition);
    const float waterSurfaceFootprint =
        max(length(dfdx(in.vCameraRelativePosition)), length(dfdy(in.vCameraRelativePosition)));
    float3 behindRelative = reconstructCameraRelative(screenUV, opaqueDepth, water);
    const float receiverFootprint = max(length(dfdx(behindRelative)), length(dfdy(behindRelative)));
    float waterDepth =
        hasOpaqueReceiver ? max(distance(behindRelative, in.vCameraRelativePosition), 0.0f) : 0.0f;
    const float verticalWaterDepth =
        hasOpaqueReceiver ? abs(behindRelative.y - in.vCameraRelativePosition.y) : 0.0f;
    // Refraction owns no temporal history. At a grazing path a voxel edge or
    // far-terrain handoff can change the one scene receiver by many blocks per
    // pixel, even though the water interface itself is continuous. Retire only
    // that unstable transmission into the continuous reflection fallback. A
    // shallow distant receiver can be smooth inside one coarse terrain cell,
    // so its water-interface footprint, distance, and physical water-column
    // depth participate as well.
    const float refractionVisibility =
        waterRefractionVisibility(abs(V.y), waterDepth, verticalWaterDepth, waterSurfaceDistance,
                                  waterSurfaceFootprint, receiverFootprint, hasOpaqueReceiver);
    const float opticalWaterDepth = waterStabilizedOpticalDepth(waterDepth, refractionVisibility);

    // ---- Refraction: wave-distorted resample of the scene, pinned at the
    // shoreline so shallow edges don't smear. From below the distorted tap
    // crosses the surface boundary into unrelated above-water content, so the
    // transmission samples straight through instead.
    // At a grazing view, a small screen-space refraction offset can jump
    // across unrelated submerged receivers. Follow the same filtered surface
    // detail as reflection: close refraction stays animated, while the
    // distant horizon settles instead of forming horizontal bands.
    float distortion = underside ? 0.0f : min(waterDepth, 4.0f) * 0.25f * surfaceDetail;
    float2 refractUV = clamp(screenUV + N.xz * 0.035f * distortion, 0.001f, 0.999f);
    float refractDepth = sceneDepth.sample(depthPoint, refractUV);
    if (refractDepth < in.clipPosition.z) {
        // The distorted tap landed on something in FRONT of the surface, so
        // fall back to the undistorted sample
        refractUV = screenUV;
        refractDepth = opaqueDepth;
    }
    float3 refracted = sceneColor.sample(colorLinear, refractUV).rgb;

    // Reconstruct the refracted floor camera-relatively. Add the camera only
    // after depth math to anchor the low-frequency caustic pattern globally.
    float3 refractedRelative = reconstructCameraRelative(refractUV, refractDepth, water);
    float3 refractedWorld = refractedRelative + water.cameraPosition;
    float depthBelow = max(in.vCameraRelativePosition.y - refractedRelative.y, 0.0f);

    // ---- Caustics: refracted-light web on the shallow floor, seen from
    // above only. World-anchored (unscaled xz: causticPattern bakes its own
    // cell scale and wave warp). From below the reconstruction lands on
    // above-water content, which painted mis-oriented white bands onto the
    // transmission, the from-below floor gets its caustics from the overlay.
    if (!underside && hasOpaqueReceiver) {
        float caustic = causticPattern(refractedWorld.xz, water.time, depthBelow) *
                        exp(-depthBelow * 0.22f) * surfaceDetail * refractionVisibility;
        refracted += water.directLightRadiance * caustic * 0.4f * directVisibility *
                     exteriorSkyVisibility * saturate(water.directLightDirection.y * 2.0f);
    }

    // ---- Absorption: shallow water reads turquoise and filters toward deep
    // blue with depth (red light dies first), the floor showing through
    // shallows. From above, waterDepth is the submerged column behind the
    // surface. From below it is the distance to the sky or shore in the AIR
    // beyond the surface, which absorbs nothing, the underwater overlay
    // already absorbs the eye-to-surface water segment, so absorbing here
    // turned the whole Snell window into opaque flat blue instead of a view
    // of the world above.
    // The intrinsic tints are the water's response to received sky light,
    // not emission. Scale them by the surface's sky access and the day-night
    // sky level, otherwise still water glows teal from its constant shallow
    // tint at night and in covered caves, drawing a bright ring on the lake
    // floor around the camera where refraction still outweighs the dark
    // night reflection.
    const float tintIllumination = max(in.vSkyLight, exteriorSkyVisibility) *
                                   mix(0.08f, 1.0f, saturate(water.physicalSkyBlend));
    float3 shallowTint = float3(0.10f, 0.42f, 0.48f) * tintIllumination;
    float3 deepTint = float3(0.02f, 0.10f, 0.22f) * tintIllumination;
    float3 waterColor = mix(shallowTint, deepTint, saturate(opticalWaterDepth * 0.12f));
    float absorb = 1.0f - exp(-opticalWaterDepth * 0.16f);
    const float3 transmitted = mix(refracted * float3(0.75f, 0.92f, 0.96f), waterColor, absorb);
    float3 body = underside ? refracted : mix(waterColor, transmitted, refractionVisibility);

    // ---- Fresnel reflection + sun sparkle. The exterior gate keeps flooded
    // caves from reflecting open sky without turning propagated skylight's
    // 4-bit ambient range into a fractional reflection grid. From below the
    // physics flips: water-to-air refraction hits total internal reflection
    // beyond the critical angle (~48.6 deg), so the surface turns into a
    // mirror of the underwater scene (SSR provides it) instead of a window to
    // the sky.
    float3 R = reflect(-V, N); // symmetric in the normal's sign
    float3 Rsky = R;
    // Exact dielectric Fresnel for whichever side of the interface this
    // fragment shows: from the water side, total internal reflection falls
    // out of Snell's law past ~48.6 degrees, and inside that window the
    // transmission dominates (~2% reflectance near vertical).
    float fresnel = waterFresnel(cosI, underside);
    if (!underside) {
        // When transmission has no trustworthy receiver, the opaque-interface
        // fallback is reflection. This is not an artistic boost to normal
        // Fresnel: it only replaces screen-space information that does not
        // exist or is unstable at a grazing path.
        fresnel = mix(1.0f, fresnel, refractionVisibility);
    }
    float3 reflection;
    if (underside) {
        // The mirror shows the underwater scene: SSR marches the reflected
        // ray; where it misses, the sunlit water volume glows through the
        // same scatter terms as the overlay. A near-black fallback here read
        // as flat dark panels, where a real internal mirror reflects
        // luminous water.
        reflection = WATER_SCATTER * 1.5f * water.directLightRadiance *
                         saturate(water.directLightDirection.y) * water.skyExposure +
                     WATER_AMBIENT;
    } else {
        float horizonBlend = pow(1.0f - saturate(Rsky.y), 2.0f);
        const float3 palette = Rsky.y > 0.0f
                                   ? mix(water.zenithColor, water.horizonColor, horizonBlend)
                                   : water.horizonColor * 0.08f;
        constexpr sampler skySampler(coord::normalized, address::clamp_to_edge, filter::linear);
        const float2 viewHorizontal = normalize(Rsky.xz + float2(1.0e-6f, 0.0f));
        const float2 sunHorizontal = normalize(water.solarDirection.xz + float2(1.0e-6f, 0.0f));
        const float signedAzimuth =
            atan2(viewHorizontal.x * sunHorizontal.y - viewHorizontal.y * sunHorizontal.x,
                  dot(viewHorizontal, sunHorizontal));
        const float2 skyUv =
            float2(signedAzimuth / (2.0f * M_PI_F) + 0.5f, saturate((Rsky.y + 0.08f) / 1.08f));
        const float3 physicalSky = atmosphereSky.sample(skySampler, skyUv).rgb;
        reflection =
            mix(palette, physicalSky, saturate(water.physicalSkyBlend) * step(0.0f, Rsky.y));
    }

    // Screen-space reflection layered over the fallback: where the reflected
    // ray finds on-screen geometry (far shore and trees from above, the floor
    // under total internal reflection from below), mirror it.
    if (water.ssrStrength > 0.0f) {
        float4 ssr = traceWaterSSR(in.vCameraRelativePosition, R, in.clipPosition.xy, fromBelow,
                                   sceneDepth, sceneColor, water);
        reflection = mix(reflection, ssr.rgb, ssr.a * water.ssrStrength);
    }

    float3 color = mix(body, reflection, fresnel * (underside ? 1.0f : exteriorSkyVisibility));
    if (!underside) {
        float sunAlign = saturate(dot(Rsky, water.directLightDirection));
        float sparkle = pow(sunAlign, 240.0f) * 2.0f + pow(sunAlign, 32.0f) * 0.25f;
        // The glint is a specular reflection, so it obeys the same Fresnel as
        // the sky term: ~2% at normal incidence, rising toward grazing.
        // Unscaled, a zenith sun mirrored in every up-facing wave below the
        // camera and bloomed into one giant white blob on the surface.
        const float sourceAboveHorizon = step(0.0f, water.directLightDirection.y);
        const float reflectedSkyRay = step(0.0f, Rsky.y);
        color += water.directLightRadiance * sparkle * fresnel * directVisibility *
                 exteriorSkyVisibility * water.directSpecularFactor * sourceAboveHorizon *
                 reflectedSkyRay;
    }

    color =
        applyFog(color, in.vWorldPosition, water.cameraPosition, water.fogDensity, water.fogColor);

    // Hairline shorelines dissolve into the shore instead of aliasing
    const float3 stableShoreTransmission = mix(waterColor, refracted, refractionVisibility);
    color = mix(stableShoreTransmission, color, saturate(waterDepth * 3.0f));
    // Shoreline foam is a bright animated band just off the shallow edge. The
    // exterior gate keeps flooded caves dark without making harmless skylight
    // nibble differences visible at an exact/far handoff. It is above-water
    // only because from below it painted white streaks along the waterline.
    // It reuses the caustic web to break the band into moving flecks.
    if (!underside) {
        // Kept narrow and well under full white: froth is sparse flecks, and
        // a wide bright band rimmed every water body like a glowing outline.
        float foamBand =
            smoothstep(0.05f, 0.35f, waterDepth) * (1.0f - smoothstep(0.35f, 0.9f, waterDepth));
        float foam =
            foamBand * (0.35f + 0.65f * causticPattern(in.vWorldPosition.xz, water.time, 0.0f));
        color =
            mix(color, float3(0.92f, 0.96f, 1.0f), saturate(foam) * 0.45f * exteriorSkyVisibility);
    }

    color = mix(color, in.vOverlayColor.rgb, max(saturate(in.vOverlayColor.a), coverageFog));
    return float4(color, 1.0f);
}

// ---------------------------------------------------------------------------
// Underwater overlay, fullscreen veil + floor caustics when the camera is
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
// while the scattered light adds, a single alpha cannot express both.
struct UnderwaterOverlayOut {
    float4 inscatter [[color(0), index(0)]];
    float4 transmit [[color(0), index(1)]];
};

fragment UnderwaterOverlayOut underwaterOverlayFragment(OverlayVertexOutput in [[stage_in]],
                                                        depth2d<float> sceneDepth [[texture(1)]],
                                                        constant WaterUniforms& water
                                                        [[buffer(3)]]) {
    float t = water.time;
    float sunUp = saturate(water.directLightDirection.y);

    // Camera-relative reconstruction keeps the depth path precise at large
    // world coordinates. Point sampling preserves voxel silhouettes instead
    // of inventing sloped surfaces between adjacent depth values.
    constexpr sampler depthPoint(mag_filter::nearest, min_filter::nearest, address::clamp_to_edge);
    float depth = sceneDepth.sample(depthPoint, in.uv);
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
    float exitDist =
        (rayDir.y > 0.02f) ? max(water.waterSurfaceY - eyeY, 0.0f) / max(rayDir.y, 0.02f) : 3.4e38f;
    float waterPath = min(dist, exitDist);

    // The light that reached the shaded point also crossed the water column
    // above it (longer when the sun sits low), so deep floors go dark, not
    // just blue. Points seen THROUGH the surface keep their above-water light,
    // and covered water (skyExposure 0) is cave-lit already, no double dark.
    float pointDepth =
        (dist <= exitDist + 0.5f) ? clamp(water.waterSurfaceY - world.y, 0.0f, 48.0f) : 0.0f;
    float lightSlant = 1.0f / max(sunUp, 0.35f);
    pointDepth *= water.skyExposure;

    // ---- Caustics on up-facing submerged surfaces (walls stay dark). The
    // water surface pass only shades pixels behind a quad, so the floor at
    // the player's feet would have none without this. The normal comes from
    // best-of-both-sides depth taps, not raw screen derivatives: a one-sided
    // derivative straddles block silhouettes and lit dashed lines along every
    // oblique edge, while picking the
    // continuous side per axis reads the true surface at edges.
    float2 texel = 1.0f / water.resolution;
    float3 pL = reconstructCameraRelative(
        in.uv - float2(texel.x, 0.0f), sceneDepth.sample(depthPoint, in.uv - float2(texel.x, 0.0f)),
        water);
    float3 pR = reconstructCameraRelative(
        in.uv + float2(texel.x, 0.0f), sceneDepth.sample(depthPoint, in.uv + float2(texel.x, 0.0f)),
        water);
    float3 pD = reconstructCameraRelative(
        in.uv - float2(0.0f, texel.y), sceneDepth.sample(depthPoint, in.uv - float2(0.0f, texel.y)),
        water);
    float3 pU = reconstructCameraRelative(
        in.uv + float2(0.0f, texel.y), sceneDepth.sample(depthPoint, in.uv + float2(0.0f, texel.y)),
        water);
    float spanL = abs(dist - length(pL)), spanR = abs(length(pR) - dist);
    float spanD = abs(dist - length(pD)), spanU = abs(length(pU) - dist);
    float3 ddxv = (spanR < spanL) ? (pR - relative) : (relative - pL);
    float3 ddyv = (spanU < spanD) ? (pU - relative) : (relative - pD);
    float3 surfaceNormal = normalize(cross(ddxv, ddyv));
    // Screen UV grows downward, so the raw cross-product winding can invert
    // a floor. Face the receiver toward the camera before the strict +Y gate:
    // floors below the camera retain caustics, ceilings above it still reject.
    surfaceNormal = orientUnderwaterReceiverNormalTowardCamera(surfaceNormal, relative);
    // The depth falloff is gentle (0.03/block) so the pool floor several
    // blocks down still catches a bright web, not only the near-surface cells.
    // Feather true silhouettes: when even the continuous side jumps more than
    // a surface at this distance could, the pixel straddles two surfaces and
    // neither normal is trustworthy, fade rather than sparkle the block edge.
    float edgeSpan = max(min(spanL, spanR), min(spanD, spanU));
    const float upFacing = underwaterCausticSurfaceConfidence(surfaceNormal.y, edgeSpan, dist);
    // Caustics land only on submerged floors: refracted sunlight never lights
    // shore terrain reconstructed BEHIND the from-below surface (whose pixels
    // the surface pass already shaded), and the focusing decays with depth.
    float submerged = step(world.y, water.waterSurfaceY);
    float throughSurface = 1.0f - smoothstep(exitDist * 0.9f, exitDist * 1.1f, dist);
    const float directEnergy =
        saturate(dot(max(water.directLightRadiance, 0.0f), float3(0.2126f, 0.7152f, 0.0722f)));
    float caustic = causticPattern(world.xz, t, pointDepth) * exp(-pointDepth * 0.05f) * upFacing *
                    submerged * throughSurface * sunUp * water.skyExposure * directEnergy;

    // ---- Transmittance (dual-source color 1): the scene is multiplied per
    // channel by the view-path and light-path absorption, and the caustic
    // MODULATES that light instead of adding white, it rides the floor's own
    // shading, so shadowed floors get proportionally dimmer webs.
    float3 transmit =
        exp(-SIGMA_A * (waterPath + pointDepth * lightSlant)) * (1.0f + caustic * 1.8f);

    // ---- Inscatter (dual-source color 0): sunlight scattered into the view
    // ray by the water itself. Henyey-Greenstein makes looking toward the sun
    // visibly brighter (the underwater silver lining); the light available in
    // the volume decays with the camera's own depth per channel; a tiny
    // ambient floor keeps covered water from reading as a void.
    float camDepth = max(water.waterSurfaceY - eyeY, 0.0f);
    float cosSun = dot(rayDir, normalize(water.directLightDirection));
    const float g = 0.45f;
    float phase = (1.0f - g * g) / pow(1.0f + g * g - 2.0f * g * cosSun, 1.5f); // 1 = isotropic
    float phaseN = phase / (1.0f + phase);      // capped so the sun lobe brightens, never blows out
    const float3 SCATTER_COLOR = WATER_SCATTER; // shared with the TIR mirror fallback
    float3 volLight =
        water.directLightRadiance * sunUp * water.skyExposure * exp(-SIGMA_A * camDepth * 0.7f);
    float buildup = 1.0f - exp(-0.15f * waterPath);
    float3 inscatter = SCATTER_COLOR * volLight * (0.55f + 0.9f * phaseN) * buildup;
    inscatter += WATER_AMBIENT * buildup;

    UnderwaterOverlayOut out;
    out.inscatter = float4(inscatter, 1.0f);
    out.transmit = float4(transmit, 1.0f);
    return out;
}

// ---------------------------------------------------------------------------
// Entity shaders, voxel-box animal models, lit like terrain
// ---------------------------------------------------------------------------
struct EntityVertexOutput {
    float4 clipPosition [[position]];
    float3 vNormal;
    float3 vColor;
    float3 vWorldPosition;
    float vSkyLight;
    float vBlockLight;
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
    out.vSkyLight = saturate(entityModel.lighting.x);
    out.vBlockLight = saturate(entityModel.lighting.y);
    return out;
}

fragment SurfaceFragmentOutput entityFragmentMain(
    EntityVertexOutput in [[stage_in]], depth2d_array<float> nearShadow [[texture(1)]],
    depth2d_array<float> farShadow [[texture(2)]], depth2d<float> horizonShadow [[texture(3)]],
    texture2d<float> cloudShadow [[texture(4)]], sampler shadowSampler [[sampler(1)]],
    constant Uniforms& uniforms [[buffer(1)]], constant ShadowUniforms& shadow [[buffer(4)]],
    constant CloudShadowUniforms& cloudShadowUniforms [[buffer(6)]]) {
    float light = max(dot(in.vNormal, uniforms.sunDirection), 0.0f);
    float visibility =
        sampleShadowVisibility(in.vWorldPosition, in.vNormal, in.vSkyLight, nearShadow, farShadow,
                               horizonShadow, shadowSampler, shadow);
    visibility *= cloudShadowVisibility(in.vWorldPosition, cloudShadow, cloudShadowUniforms);
    constexpr float3 BLOCK_LIGHT_TINT = float3(1.0f, 0.55f, 0.22f);
    constexpr float BLOCK_LIGHT_STRENGTH = 1.5f;
    const float3 blockLight =
        BLOCK_LIGHT_TINT * (in.vBlockLight * in.vBlockLight) * BLOCK_LIGHT_STRENGTH;
    float3 litColor = in.vColor * (uniforms.sunColor * light * visibility +
                                   uniforms.ambientColor * in.vSkyLight + blockLight);

    SurfaceFragmentOutput result;
    result.scene = float4(applyFog(litColor, in.vWorldPosition, uniforms.cameraPosition,
                                   uniforms.fogDensity, uniforms.fogColor),
                          1.0f);
    result.surface = float4(in.vColor, in.vSkyLight);
    return result;
}
