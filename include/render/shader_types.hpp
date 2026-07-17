#pragma once

// Shared GPU data layouts, included by BOTH the C++ engine and the .metal
// shaders (the shader build passes -I include/). simd types have identical
// size and alignment in each language, so one definition serves both sides;
// the static_asserts below make any drift a compile error instead of a
// corrupted frame.
#include <simd/simd.h>

#ifndef __METAL_VERSION__
#include <algorithm>
#include <cmath>
#endif

#define FAR_TERRAIN_COVERAGE_FADE_BLOCKS 256.0f
#define FAR_TERRAIN_COVERAGE_FADE_FRACTION 0.125f
#define FAR_TERRAIN_COVERAGE_MIN_FADE_BLOCKS 16.0f
#define FAR_TERRAIN_TILE_EDGE_BLOCKS 256.0f
#define FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS 16.0f
#define FAR_TERRAIN_EXACT_COLUMNS_PER_TILE 16
#define FAR_TERRAIN_CANOPY_ATTRIBUTE_MASK (1u << 28u)
#define FAR_TERRAIN_SKIRT_ATTRIBUTE_MASK (1u << 29u)
#define FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD 32
#define FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR 4
#define FAR_TERRAIN_EXACT_MASK_WORD_COUNT 8
#define FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE                                                    \
    (FAR_TERRAIN_EXACT_MASK_WORD_COUNT / FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR)
#define FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE 3
#define FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS (FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE / 2)
#define FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT 9
#define FAR_TERRAIN_LOD_TRANSITION_SECONDS_VALUE 0.65f
#define FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS_VALUE 0.10f
#define FAR_TERRAIN_DRAW_FLAG 1u
#define FAR_TERRAIN_LOD_TRANSITION_FLAG 2u
#define FAR_TERRAIN_LOD_TARGET_FLAG 4u
#define FAR_TERRAIN_LOD_EMERGENCY_FLAG 8u
#define FAR_TERRAIN_FACE_PLUS_X 0u
#define FAR_TERRAIN_FACE_MINUS_X 1u
#define FAR_TERRAIN_FACE_PLUS_Z 2u
#define FAR_TERRAIN_FACE_MINUS_Z 3u
#define FAR_TERRAIN_OWNERSHIP_RISER_INSET_BLOCKS 0.03125f

// A far terrain riser is emitted by the higher terrain column behind its
// outward face. Half-open X/Z lookup already selects that column for -X and
// -Z faces, but a +X or +Z face lies exactly on the following column boundary.
// Move only those terrain-riser samples inward by one binary-exact fraction.
// Tops, water, canopies, and skirts keep destination-fragment ownership by
// passing false for useEmittingColumn.
static inline bool farTerrainOpaqueRiserUsesEmittingColumn(unsigned int face, bool canopy,
                                                           bool skirt) {
    return face <= FAR_TERRAIN_FACE_MINUS_Z && !canopy && !skirt;
}

// A skirt joins the edge column that emitted it to the neighboring receiving
// column. It is valid only while both far-terrain owners remain visible.
// Testing just the destination leaves an orphan wall when exact residency
// reaches the emitting side of a tile boundary first.
static inline bool farTerrainSkirtOwnersVisible(bool emittingColumnOwnedByExact,
                                                bool receivingColumnOwnedByExact) {
    return !emittingColumnOwnedByExact && !receivingColumnOwnedByExact;
}

static inline simd_float2 farTerrainExactOwnershipSamplePosition(simd_float2 localPosition,
                                                                 unsigned int face,
                                                                 bool useEmittingColumn) {
    if (!useEmittingColumn) {
        return localPosition;
    }
    if (face == FAR_TERRAIN_FACE_PLUS_X) {
        localPosition.x -= FAR_TERRAIN_OWNERSHIP_RISER_INSET_BLOCKS;
    } else if (face == FAR_TERRAIN_FACE_PLUS_Z) {
        localPosition.y -= FAR_TERRAIN_OWNERSHIP_RISER_INSET_BLOCKS;
    }
    return localPosition;
}

// Positive faces already lie in the receiving half-open column. Negative
// faces lie in the emitting column and need the corresponding outward inset
// to reach their receiving neighbor.
static inline simd_float2 farTerrainSkirtReceivingOwnershipSamplePosition(simd_float2 localPosition,
                                                                          unsigned int face) {
    if (face == FAR_TERRAIN_FACE_MINUS_X) {
        localPosition.x -= FAR_TERRAIN_OWNERSHIP_RISER_INSET_BLOCKS;
    } else if (face == FAR_TERRAIN_FACE_MINUS_Z) {
        localPosition.y -= FAR_TERRAIN_OWNERSHIP_RISER_INSET_BLOCKS;
    }
    return localPosition;
}

// A positive frontier marks the nearest missing GPU-resident step-32 parent.
// Geometry beyond it is suppressed and the preceding tile fades completely
// into the frame fog. Zero is the steady-state disabled value used by exact
// cubes and by a fully covered far view.
static inline bool farTerrainCoverageVisible(float horizontalDistance, float frontierDistance) {
    return frontierDistance <= 0.0f || horizontalDistance < frontierDistance;
}

static inline float farTerrainCoverageFadeBlocks(float frontierDistance) {
    const float proportional = frontierDistance * FAR_TERRAIN_COVERAGE_FADE_FRACTION;
#ifdef __METAL_VERSION__
    return metal::clamp(proportional, FAR_TERRAIN_COVERAGE_MIN_FADE_BLOCKS,
                        FAR_TERRAIN_COVERAGE_FADE_BLOCKS);
#else
    return std::clamp(proportional, FAR_TERRAIN_COVERAGE_MIN_FADE_BLOCKS,
                      FAR_TERRAIN_COVERAGE_FADE_BLOCKS);
#endif
}

static inline float farTerrainCoverageFog(float horizontalDistance, float frontierDistance) {
    if (frontierDistance <= 0.0f) {
        return 0.0f;
    }
    // During cold startup the nearest missing parent can be less than one tile
    // away. Applying the full 256-block horizon taper in that state starts the
    // fade behind the camera and turns every available fallback fragment into
    // an opaque fog wall. Keep only the last eighth of a short connected prefix
    // as its transition, then grow smoothly to the full horizon band.
    const float fadeBlocks = farTerrainCoverageFadeBlocks(frontierDistance);
    float amount = (horizontalDistance - (frontierDistance - fadeBlocks)) / fadeBlocks;
#ifdef __METAL_VERSION__
    amount = metal::clamp(amount, 0.0f, 1.0f);
#else
    amount = std::clamp(amount, 0.0f, 1.0f);
#endif
    return amount * amount * (3.0f - 2.0f * amount);
}

static inline float farTerrainLodTransitionAmount(float amount) {
#ifdef __METAL_VERSION__
    return metal::clamp(amount, 0.0f, 1.0f);
#else
    return std::clamp(amount, 0.0f, 1.0f);
#endif
}

static inline float farTerrainLodTransitionProgressAtSeconds(float elapsedSeconds) {
    const float phase =
        farTerrainLodTransitionAmount(elapsedSeconds / FAR_TERRAIN_LOD_TRANSITION_SECONDS_VALUE);
    return phase * phase * (3.0f - 2.0f * phase);
}

static inline bool farTerrainLodTransitionTarget(unsigned int flags) {
    return (flags & FAR_TERRAIN_LOD_TARGET_FLAG) != 0u;
}

static inline float farTerrainLodTerrainSwapProgress(unsigned int flags) {
    // A direct emergency-parent replacement reaches its hidden terrain swap
    // at the shared emergency time. Normal tier changes use the temporal
    // midpoint.
    return (flags & FAR_TERRAIN_LOD_EMERGENCY_FLAG) != 0u
               ? farTerrainLodTransitionProgressAtSeconds(
                     FAR_TERRAIN_LOD_EMERGENCY_SWAP_SECONDS_VALUE)
               : 0.5f;
}

// Production filtered tiers are not a strict height-min pyramid. Draw one
// complete terrain topology at a time so a source/target height mismatch can
// never expose an unsupported partial sheet.
static inline bool farTerrainLodTerrainVisible(float progress, unsigned int flags) {
    if ((flags & FAR_TERRAIN_LOD_TRANSITION_FLAG) == 0u) {
        return true;
    }
    progress = farTerrainLodTransitionAmount(progress);
    const bool target = farTerrainLodTransitionTarget(flags);
    const bool swapped = progress >= farTerrainLodTerrainSwapProgress(flags);
    return target == swapped;
}

// Skirts belong to the currently visible terrain topology. Water keeps its
// source topology until transition completion, but retaining a source skirt
// after the source terrain swaps out creates a freestanding vertical panel.
static inline bool farTerrainLodSkirtVisible(float progress, unsigned int flags) {
    return farTerrainLodTerrainVisible(progress, flags);
}

// Hide the complete terrain swap behind a narrow terrain-only fog pulse. It is
// intentionally independent of canopy, water, skirt, and coverage fog.
static inline float farTerrainLodTerrainFog(float progress, unsigned int flags) {
    if ((flags & FAR_TERRAIN_LOD_TRANSITION_FLAG) == 0u) {
        return 0.0f;
    }
    progress = farTerrainLodTransitionAmount(progress);
    const bool emergency = (flags & FAR_TERRAIN_LOD_EMERGENCY_FLAG) != 0u;
    const float center = farTerrainLodTerrainSwapProgress(flags);
    const float halfWidth = emergency ? 0.030f : 0.080f;
#ifdef __METAL_VERSION__
    const float amount = metal::clamp(1.0f - metal::abs(progress - center) / halfWidth, 0.0f, 1.0f);
#else
    const float amount = std::clamp(1.0f - std::abs(progress - center) / halfWidth, 0.0f, 1.0f);
#endif
    return amount * amount * (3.0f - 2.0f * amount);
}

// Unrelated canopy sets exchange in two monotonic phases. The target appears
// before the source retires, so a forest never passes through an empty frame
// and no source-only crown disappears at transition completion.
static inline bool farTerrainLodCanopyVisible(float progress, float ditherThreshold,
                                              unsigned int flags) {
    if ((flags & FAR_TERRAIN_LOD_TRANSITION_FLAG) == 0u) {
        return true;
    }
    progress = farTerrainLodTransitionAmount(progress);
    ditherThreshold = farTerrainLodTransitionAmount(ditherThreshold);
    if (farTerrainLodTransitionTarget(flags)) {
        if (progress >= 0.5f) {
            return true;
        }
        return ditherThreshold < farTerrainLodTransitionAmount(progress * 2.0f);
    }
    if (progress <= 0.5f) {
        return true;
    }
    if (progress >= 1.0f) {
        return false;
    }
    return ditherThreshold >= farTerrainLodTransitionAmount((progress - 0.5f) * 2.0f);
}

// Water remains source-owned for the complete transition, and the renderer
// retires it atomically after the target topology becomes authoritative.
static inline bool farTerrainLodConnectedGeometryVisible(unsigned int flags) {
    return (flags & FAR_TERRAIN_LOD_TRANSITION_FLAG) == 0u || !farTerrainLodTransitionTarget(flags);
}

#ifdef __METAL_VERSION__
// Interleaved gradient noise — the engine's one deterministic per-pixel
// dither/rotation source (convention: seeded randomness only, no temporal
// noise). One definition here because five passes sample it and the magic
// constants must never drift apart.
static inline float interleavedGradientNoise(metal::float2 px) {
    return metal::fract(52.9829189f *
                        metal::fract(metal::dot(px, metal::float2(0.06711056f, 0.00583715f))));
}

// Fade procedural water bands before their phase advances too far between
// adjacent pixels. The argument is the screen-space phase footprint in
// radians per pixel. Keeping this transfer function shared and portable lets
// the CPU regression suite pin the threshold while the fragment shader
// derives the footprint from screen-space derivatives.
static inline float waterBandVisibility(float phaseFootprint) {
    float t = metal::clamp((phaseFootprint - 0.45f) / 1.35f, 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

// Foliage wind sway, ONE definition shared by the scene and shadow vertex
// stages so shadows track the displaced geometry exactly (the same
// never-drift rule as applyFog). sway: 1 = flora (v is the cross-quad texture
// v, 0 at the tip — the base stays rooted), 2 = leaves (a continuous field of
// world position, so vertices shared by merged quads displace identically and
// the canopy never cracks). strength 0 (the waving setting off) is a no-op.
static inline metal::float3 applySway(metal::float3 worldPos, uint sway, float v, float time,
                                      float strength) {
    if (sway == 0u || strength <= 0.0f) {
        return worldPos;
    }
    if (sway == 1u) {
        float w = (1.0f - v) * (1.0f - v);
        metal::float2 cell = metal::floor(worldPos.xz) + 0.5f; // whole plant, one phase
        float phase = cell.x * 0.9f + cell.y * 1.3f;
        float gust =
            metal::sin(time * 1.6f + phase) + 0.4f * metal::sin(time * 2.7f + phase * 1.7f);
        worldPos.x += gust * 0.055f * w * strength;
        worldPos.z += metal::cos(time * 1.2f + phase) * 0.045f * w * strength;
    } else {
        float phase = worldPos.x * 0.35f + worldPos.y * 0.21f + worldPos.z * 0.28f;
        worldPos.x += metal::sin(time * 0.9f + phase) * 0.03f * strength;
        worldPos.z += metal::cos(time * 0.7f + phase * 1.3f) * 0.03f * strength;
    }
    return worldPos;
}

// Water surface waves. The three directional waves live in ONE table so the
// vertex displacement (waterWaveHeight) and the fragment normal
// (waterSurfaceNormal) can never drift apart — editing the sea updates both
// (the same never-drift rule as applySway). A gentle sea, not chop, in world
// space so it stays continuous across chunk borders: coincident border
// vertices from the uniform water tessellation displace identically.
struct WaterWave {
    metal::float2 dir; // travel direction (unit-ish)
    float freq;        // spatial frequency (radians per block)
    float amp;         // crest height in blocks
    float speed;       // temporal frequency (radians per second)
};
// Amplitudes total ~0.21 blocks: visible swell at eye level, still far under
// the 0.875 surface inset so a displaced top never pokes above its side quads.
constant WaterWave WATER_WAVES[3] = {
    {metal::float2(0.80f, 0.60f), 0.52f, 0.110f, 1.1f},
    {metal::float2(-0.50f, 0.87f), 0.80f, 0.065f, 1.5f},
    {metal::float2(0.20f, -0.98f), 1.05f, 0.040f, 2.1f},
};

static inline float waterWaveHeight(metal::float2 p, float t) {
    float h = 0.0f;
    for (int i = 0; i < 3; ++i) {
        WaterWave w = WATER_WAVES[i];
        h += w.amp * metal::sin(metal::dot(p, w.dir) * w.freq + t * w.speed);
    }
    return h;
}

// Analytic normal of the same wave field (the gradient of waterWaveHeight,
// d/dp of A*sin(dot(p,D)*K + t*S) = A*D*K*cos(...)) plus a faint fine ripple
// for specular sparkle. The slope scale keeps the surface glassy.
static inline metal::float3 waterSurfaceNormal(metal::float2 p, float t) {
    metal::float2 g = metal::float2(0.0f, 0.0f);
    for (int i = 0; i < 3; ++i) {
        WaterWave w = WATER_WAVES[i];
        g += w.amp * w.freq * w.dir * metal::cos(metal::dot(p, w.dir) * w.freq + t * w.speed);
    }
    // Fine ripple octaves, shading only (not displaced): they carry the
    // perceived chop and sparkle that geometric displacement alone cannot.
    g += 0.014f *
         metal::float2(metal::cos(p.x * 1.9f + t * 2.3f), metal::cos(p.y * 2.2f - t * 2.0f));
    g += 0.007f * metal::float2(metal::cos((p.x + p.y) * 3.7f + t * 3.1f),
                                metal::cos((p.y - p.x) * 3.3f - t * 2.7f));
    const float slope = 1.5f;
    return metal::normalize(metal::float3(-g.x * slope, 1.0f, -g.y * slope));
}
#endif

#ifndef __METAL_VERSION__
// CPU form of waterBandVisibility for deterministic shader-contract tests.
// This deliberately mirrors the Metal expression above rather than depending
// on a graphics framework in otherwise portable render tests.
static inline float waterBandVisibility(float phaseFootprint) {
    const float t = std::clamp((phaseFootprint - 0.45f) / 1.35f, 0.0f, 1.0f);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}
#endif

// Bound at buffer(1) in the main chunk/highlight shaders.
struct Uniforms {
    simd_float4x4 modelMatrix;
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3 sunDirection; // normalized direction to the sun
    simd_float3 sunColor;
    simd_float3 ambientColor;
    simd_float3 fogColor;
    float fogDensity;
    simd_float3 cameraPosition; // for fog distance
    float time;                 // seconds; drives foliage sway
    float swayStrength;         // 0 = waving setting off, 1 = full sway
    float wetness;              // 0 dry .. 1 soaked (rain darkening + sheen)
};

// Per-chunk world offset, bound at buffer(2) via setVertexBytes. Vertices are
// chunk-local so their fp16 coordinates stay exact; this restores world space.
// origin.w is reserved and remains zero. overlayColorAndStrength remains
// available for diagnostic overlays.
// farMetadata.x contains the four displayed-neighbor skirt edge bits ordered
// by FaceNormal values +X, -X, +Z, and -Z. farMetadata.y stores the float bits
// of the temporary coverage frontier distance, or zero when coverage is
// complete. farMetadata.z stores LOD transition progress as float bits.
// farMetadata.w stores FAR_TERRAIN_* flags and is zero for exact cube draws.
struct ChunkOrigin {
    simd_float4 origin;
    simd_float4 overlayColorAndStrength;
    simd_uint4 farMetadata;
};

// Fragment-only ownership for one far tile and its eight immediate neighbors.
// Each tile contributes one 256-bit mask, with one bit per 16x16 exact chunk
// column in row-major local X/Z order. The neighboring masks let a canopy or
// waterfall owned by one tile cross a tile face without reappearing over a
// revision-ready exact column. Exact cube draws bind an all-zero value.
struct FarTerrainOwnershipUniforms {
    simd_uint4 readyColumnMasks[FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT *
                                FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE];
};

// One cascade's light view-projection, bound at buffer(1) via setVertexBytes
// in the depth-only shadow pass (shadow.metal). time/swayStrength must match
// the scene pass exactly or foliage shadows detach from the swaying blades.
struct ShadowPassUniforms {
    simd_float4x4 lightViewProj;
    float time;
    float swayStrength;
};

// A macro, not a constexpr int: MSL rejects a program-scope constant outside
// the constant address space, and both languages accept an array bound here.
#define SHADOW_CASCADE_COUNT 3

// Cascaded shadow-map sampling data for the scene passes (chunks, entities),
// bound at buffer(4). The fragment picks a cascade by camera distance, projects
// the world position into it, and PCF-samples the depth array at texture(1).
struct ShadowUniforms {
    simd_float4x4 cascadeViewProj[3]; // 0 / 64 / 128
    simd_float4 cascadeSplitDist;     // 192: x,y,z = far world distance of each cascade
    // 208: x = penumbra texel radius, y = normal offset (world units),
    // z = strength (0 disables sampling so the pass can be skipped),
    // w = fade-start distance
    simd_float4 shadowParams;
};

// Bound at buffer(3) in the water shaders (vertex: wave time; fragment:
// refraction/reflection/caustics). The water pass composites its own pixels
// from the resolved opaque scene. Screen-space positions reconstruct into a
// camera-relative world frame so large absolute coordinates cannot quantize
// water thickness or caustic placement at cubic chunk boundaries.
struct WaterUniforms {
    simd_float4x4 invCameraRelativeViewProjection; // clip → camera-relative world
    simd_float4x4 cameraRelativeViewProjection;    // camera-relative world → clip
    simd_float3 zenithColor;
    simd_float3 horizonColor;
    simd_float3 sunDirection;
    simd_float3 sunColor;
    simd_float3 cameraPosition;
    simd_float3 fogColor;
    simd_float2 resolution; // scene color/depth texture size in pixels
    float fogDensity;
    float time;             // seconds; drives waves + caustics
    float cameraUnderwater; // 1.0 when the camera is inside water
    float ssrStrength;      // 0 = sky-only reflection (the pre-SSR look)
    float skyExposure;      // 0 when solid ground seals the camera's water
                            // column (aquifers, roofed lakes): no sunlight
                            // reaches covered water, so caustics and the
                            // sun-driven murk must go dark, not track the sun
    float waterSurfaceY;    // world Y of the surface of the water body the
                            // camera is in: upward rays leave the water there,
                            // so murk and caustics must stop at that exit
                            // instead of fogging out to the opaque depth
};

// Atmospheric sky, bound at buffer(1) in sky.metal. The fragment shader
// reconstructs a per-pixel view ray from the camera basis (like clouds.metal)
// so the sun and moon are true direction-projected discs, not screen-space
// blobs, and the gradient tracks where the camera looks. zenith/horizon come
// from computeDayNightUniforms (also the fog palette — one source).
struct SkyUniforms {
    simd_float3 cameraForward;
    simd_float3 cameraRight;
    simd_float3 cameraUp;
    simd_float3 sunDirection;
    simd_float3 moonDirection; // opposes the sun; lights the night sky
    simd_float3 sunColor;
    simd_float3 zenithColor;
    simd_float3 horizonColor;
    float tanHalfFov;
    float aspect;
    float sunIntensity; // 0 at night, 1 at noon
    float starStrength; // 0 by day, 1 deep night — fades the star field
};

// Procedural cloud layer, bound at buffer(0) in clouds.metal. The fragment
// shader ray-casts from the camera through each pixel onto the flat cloud
// plane (mode 1) or marches the volumetric slab above it (mode 2), so the
// camera basis and projection shape ride along.
struct CloudUniforms {
    simd_float3 cameraPosition;
    simd_float3 cameraForward;
    simd_float3 cameraRight;
    simd_float3 cameraUp;
    simd_float3 sunDirection;
    float tanHalfFov;
    float aspect;
    float windOffset;
    float cloudAltitude;
    float noiseFrequency;
    float cloudThreshold;
    float volumetric;   // 0 = flat plane layer, 1 = ray-marched volume
    float sunElevation; // sun height 0..1, dims clouds toward night
};

// Weather particle instance data, bound at buffer(0) in particles.metal.
struct GPUParticle {
    simd_float3 position;
    simd_float3 velocity;
    float lifetime;
    float type; // 0 = rain, 1 = snow
};

// One vertex of an entity voxel-box mesh, indexed directly by vertex_id in
// entities' draw calls (buffer(0) in the entity shaders).
struct EntityVertex {
    simd_float3 position; // model-local (feet-centered)
    simd_float3 normal;
    simd_float3 color;
};

// Per-entity transform, bound at buffer(2) via setVertexBytes.
struct EntityModel {
    simd_float4x4 model;
};

// Bound at buffer(1) in particles.metal.
struct ParticleUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3 cameraPosition;
};

// Bound at buffer(0) in the bloom extract/blur passes (via setFragmentBytes
// — 32 bytes of per-pass state, not a real buffer).
struct BloomUniforms {
    simd_float2 resolution; // output texture width, height
    simd_float2 texelSize;  // 1/width, 1/height
    float threshold;        // extract luminance threshold
    float intensity;        // bloom strength multiplier
    float blurRadius;       // Kawase blur radius in texels
};

// Screen-space ambient occlusion, bound at buffer(0) in ssao.metal. The
// half-res generate pass reconstructs view-space position + a depth-derivative
// normal from the resolved depth, samples a rotated hemisphere, and writes an
// occlusion factor; the apply pass multiplies it (4-tap box upsampled) onto
// the HDR scene to darken creases, corners, and enclosed spaces smoothly.
struct SsaoUniforms {
    simd_float4x4 projection;    // 0: view → clip, to project samples to screen
    simd_float4x4 invProjection; // 64: clip → view, to reconstruct view pos
    simd_float2 resolution;      // 128: half-res target size
    float radius;                // 136: hemisphere radius in view units
    float strength;              // 140: occlusion darkening
    float bias;                  // 144: self-occlusion guard
    uint32_t frameIndex;         // 148: deterministic rotation
};

// Volumetric light march, bound at buffer(0) in volumetrics.metal. The
// half-res pass reconstructs each pixel's world ray from the resolved depth,
// marches camera→scene sampling the shadow cascades, and accumulates sun
// in-scatter (Henyey-Greenstein phase). Composited additively onto the HDR
// scene. The shadow cascade data rides the shared ShadowUniforms at buffer(1).
struct VolumetricUniforms {
    simd_float4x4 invViewProjection; // 0: clip → world, to place the scene hit
    simd_float3 cameraPosition;      // 64
    simd_float3 sunDirection;        // 80: active light (sun or moon)
    simd_float3 sunColor;            // 96
    float stepCount;                 // 112
    float density;                   // 116: in-scatter per world unit
    float anisotropy;                // 120: HG g (forward scatter toward the sun)
    float maxDistance;               // 124: march cap in world units
    float underwater;                // 128: 1 when the camera is submerged
    uint32_t frameIndex;             // 132: deterministic dither offset
};

// Persistent auto-exposure state (device buffer, survives across frames).
// The reduction kernel blends each frame's average scene luminance into
// smoothedLogLum and derives the exposure the composite multiplies by; the
// CPU seeds it once so the first frames aren't black. Bound at buffer(0) in
// exposure.metal and read at buffer(1) by post.metal.
struct ExposureState {
    float smoothedLogLum; // EMA of log2(scene luminance)
    float exposure;       // derived multiplier the composite applies
};

// Bound at buffer(1) in exposure.metal via setBytes.
struct ExposureParams {
    // keyValue sets where typical daylight lands: exposure = keyValue/avgLum,
    // so keyValue ≈ a lit surface's average luminance keeps day near 1.0
    // (mapping the average to middle grey instead over-darkens bright scenes).
    float keyValue;
    float adaptationDownRate; // 0..1 EMA weight when the scene brightens (fast:
                              // the eye stops down quickly facing the sun)
    float minLogLum;          // clamp floor for scene log-luminance
    float maxLogLum;          // clamp ceiling
    simd_uint2 sampleGrid;    // reduction sample count across the frame (e.g. 16×16)
    // Exposure clamp: minExposure keeps bright outdoor scenes from being
    // crushed dim; maxExposure lifts caves/night without blowing up.
    float minExposure;
    float maxExposure;
    float adaptationUpRate; // slower EMA weight when the scene darkens
    // Highlight weighting: a plain mean barely moves when a small bright sun
    // enters the frame, so bright samples get up-weighted —
    // w = 1 + gain * saturate((logLum - knee) / range) — and facing the sun
    // actually stops the scene down.
    float highlightGain;
    float highlightKnee;  // log2 luminance where up-weighting starts
    float highlightRange; // log2 range over which the weight ramps in
};

// Bound at buffer(0) in the final composite (post.metal): the one pass that
// converts the linear HDR scene to the display — exposure, bloom add,
// filmic tonemap, vibrance grade, optional CAS sharpen, dither. It always
// runs, so the frame is tonemapped even with bloom off.
struct PostUniforms {
    simd_float2 resolution;  // drawable size in pixels
    float exposure;          // linear pre-tonemap multiplier
    float bloomIntensity;    // 0 = bloom texture is the black fallback
    float vibrance;          // 0..2 saturation-aware boost; 1 = stock look
    float sharpening;        // 0..1 CAS strength; 0 = skip
    uint32_t frameIndex;     // deterministic dither phase
    float flareStrength;     // 0 = flare off (setting off / sun behind camera)
    simd_float2 sunScreenUV; // sun position in composite UV space
};

// Persistent lens-flare occlusion (device buffer, survives across frames).
// The probe kernel eases visibility toward the fraction of sky depth taps
// around the sun, so the flare fades smoothly behind terrain instead of
// popping at silhouette edges. Bound at buffer(0) by the flareProbe kernel;
// read at buffer(2) by the composite (post.metal).
struct FlareState {
    float visibility; // 0 sun fully occluded .. 1 fully visible
};

#ifndef __METAL_VERSION__
#include <cstddef>

static_assert(sizeof(Uniforms) == 304);
static_assert(sizeof(ChunkOrigin) == 48);
static_assert(offsetof(ChunkOrigin, farMetadata) == 32);
static_assert(sizeof(FarTerrainOwnershipUniforms) == 288);
static_assert(offsetof(FarTerrainOwnershipUniforms, readyColumnMasks) == 0);
static_assert(FAR_TERRAIN_EXACT_MASK_WORD_COUNT * FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD ==
              FAR_TERRAIN_EXACT_COLUMNS_PER_TILE * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE);
static_assert(FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE * FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR ==
              FAR_TERRAIN_EXACT_MASK_WORD_COUNT);
static_assert(sizeof(simd_uint4) / sizeof(uint32_t) == FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR);
static_assert(FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE * FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE ==
              FAR_TERRAIN_EXACT_MASK_NEIGHBOR_COUNT);
static_assert(offsetof(Uniforms, sunDirection) == 192);
static_assert(offsetof(Uniforms, fogColor) == 240);
static_assert(offsetof(Uniforms, fogDensity) == 256);
static_assert(offsetof(Uniforms, cameraPosition) == 272);
static_assert(offsetof(Uniforms, time) == 288);
static_assert(offsetof(Uniforms, swayStrength) == 292);
static_assert(offsetof(Uniforms, wetness) == 296);

static_assert(sizeof(WaterUniforms) == 256);
static_assert(offsetof(WaterUniforms, skyExposure) == 248);
static_assert(offsetof(WaterUniforms, waterSurfaceY) == 252);
static_assert(offsetof(WaterUniforms, cameraRelativeViewProjection) == 64);
static_assert(offsetof(WaterUniforms, zenithColor) == 128);
static_assert(offsetof(WaterUniforms, resolution) == 224);
static_assert(offsetof(WaterUniforms, fogDensity) == 232);
static_assert(offsetof(WaterUniforms, time) == 236);
static_assert(offsetof(WaterUniforms, cameraUnderwater) == 240);
static_assert(offsetof(WaterUniforms, ssrStrength) == 244);

static_assert(sizeof(ShadowPassUniforms) == 80);
static_assert(offsetof(ShadowPassUniforms, time) == 64);
static_assert(offsetof(ShadowPassUniforms, swayStrength) == 68);

static_assert(sizeof(ShadowUniforms) == 224);
static_assert(offsetof(ShadowUniforms, cascadeSplitDist) == 192);
static_assert(offsetof(ShadowUniforms, shadowParams) == 208);

static_assert(sizeof(SkyUniforms) == 144);
static_assert(offsetof(SkyUniforms, moonDirection) == 64);
static_assert(offsetof(SkyUniforms, zenithColor) == 96);
static_assert(offsetof(SkyUniforms, tanHalfFov) == 128);
static_assert(offsetof(SkyUniforms, sunIntensity) == 136);
static_assert(offsetof(SkyUniforms, starStrength) == 140);

static_assert(sizeof(CloudUniforms) == 112);
static_assert(offsetof(CloudUniforms, sunDirection) == 64);
static_assert(offsetof(CloudUniforms, tanHalfFov) == 80);
static_assert(offsetof(CloudUniforms, cloudThreshold) == 100);
static_assert(offsetof(CloudUniforms, volumetric) == 104);
static_assert(offsetof(CloudUniforms, sunElevation) == 108);

static_assert(sizeof(GPUParticle) == 48);
static_assert(offsetof(GPUParticle, velocity) == 16);
static_assert(offsetof(GPUParticle, lifetime) == 32);

static_assert(sizeof(EntityVertex) == 48);
static_assert(offsetof(EntityVertex, normal) == 16);
static_assert(offsetof(EntityVertex, color) == 32);
static_assert(sizeof(EntityModel) == 64);

static_assert(sizeof(ParticleUniforms) == 144);
static_assert(offsetof(ParticleUniforms, cameraPosition) == 128);

static_assert(sizeof(BloomUniforms) == 32);
static_assert(offsetof(BloomUniforms, threshold) == 16);

static_assert(sizeof(PostUniforms) == 40);
static_assert(offsetof(PostUniforms, exposure) == 8);
static_assert(offsetof(PostUniforms, frameIndex) == 24);
static_assert(offsetof(PostUniforms, flareStrength) == 28);
static_assert(offsetof(PostUniforms, sunScreenUV) == 32);
static_assert(sizeof(FlareState) == 4);

static_assert(sizeof(SsaoUniforms) == 160);
static_assert(offsetof(SsaoUniforms, invProjection) == 64);
static_assert(offsetof(SsaoUniforms, resolution) == 128);
static_assert(offsetof(SsaoUniforms, radius) == 136);
static_assert(offsetof(SsaoUniforms, frameIndex) == 148);

static_assert(sizeof(VolumetricUniforms) == 144);
static_assert(offsetof(VolumetricUniforms, cameraPosition) == 64);
static_assert(offsetof(VolumetricUniforms, stepCount) == 112);
static_assert(offsetof(VolumetricUniforms, underwater) == 128);
static_assert(offsetof(VolumetricUniforms, frameIndex) == 132);

static_assert(sizeof(ExposureState) == 8);
static_assert(offsetof(ExposureState, exposure) == 4);

static_assert(sizeof(ExposureParams) == 48);
static_assert(offsetof(ExposureParams, adaptationDownRate) == 4);
static_assert(offsetof(ExposureParams, sampleGrid) == 16);
static_assert(offsetof(ExposureParams, minExposure) == 24);
static_assert(offsetof(ExposureParams, maxExposure) == 28);
static_assert(offsetof(ExposureParams, adaptationUpRate) == 32);
static_assert(offsetof(ExposureParams, highlightGain) == 36);
static_assert(offsetof(ExposureParams, highlightKnee) == 40);
static_assert(offsetof(ExposureParams, highlightRange) == 44);
#endif
