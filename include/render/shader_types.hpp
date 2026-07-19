#pragma once

// Shared GPU data layouts, included by BOTH the C++ engine and the .metal
// shaders (the shader build passes -I include/). simd types have identical
// size and alignment in each language, so one definition serves both sides;
// the static_asserts below make any drift a compile error instead of a
// corrupted frame.
#include <simd/simd.h>
#include <world/weather_grid.hpp>

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
#define FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND 12.0f
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

// The canonical weather wind used by all foliage vertex passes. Direction is
// normalized in world X/Z, speed remains in physical blocks per second, and
// strength reflects the user's waving-foliage setting.
struct FoliageWindUniforms {
    simd_float2 direction;
    float speedBlocksPerSecond;
    float strength;
};

#ifndef __METAL_VERSION__
static inline FoliageWindUniforms makeFoliageWindUniforms(float windXBlocksPerSecond,
                                                          float windZBlocksPerSecond,
                                                          bool enabled) noexcept {
    FoliageWindUniforms result{};
    if (!std::isfinite(windXBlocksPerSecond) || !std::isfinite(windZBlocksPerSecond)) {
        return result;
    }
    const float magnitude = std::hypot(windXBlocksPerSecond, windZBlocksPerSecond);
    if (magnitude > 1e-4f) {
        result.direction =
            simd_make_float2(windXBlocksPerSecond / magnitude, windZBlocksPerSecond / magnitude);
        result.speedBlocksPerSecond =
            std::min(magnitude, static_cast<float>(FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND));
    }
    result.strength = enabled ? 1.0f : 0.0f;
    return result;
}
#endif

#ifdef __METAL_VERSION__
// Interleaved gradient noise, the engine's one deterministic per-pixel
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

// Screen-space reflections have no information past the color/depth buffer
// boundary. At a long grazing path, tiny depth discontinuities otherwise
// make adjacent pixels choose unrelated full-resolution hits or the sky
// fallback. Those discontinuities become visible well before a ray is
// mathematically horizontal, so filter and retire the far grazing region
// conservatively. The analytic atmosphere is continuous there, while nearby
// and non-grazing geometry keeps its full SSR detail.
static inline float waterSsrReflectionMipLevel(float grazing, float hitDistance) {
    const float horizon = metal::smoothstep(0.20f, 0.60f, metal::clamp(grazing, 0.0f, 1.0f));
    const float distance = metal::smoothstep(4.0f, 64.0f, metal::max(hitDistance, 0.0f));
    return 4.0f * horizon * (0.50f + 0.50f * distance);
}

static inline float waterSsrStabilityConfidence(float grazing, float hitDistance) {
    const float horizon = metal::smoothstep(0.20f, 0.58f, metal::clamp(grazing, 0.0f, 1.0f));
    const float distance = metal::smoothstep(12.0f, 56.0f, metal::max(hitDistance, 0.0f));
    return 1.0f - horizon * distance;
}

static inline float waterSsrJitterAmplitude(float grazing) {
    return metal::mix(0.24f, 0.04f,
                      metal::smoothstep(0.50f, 0.90f, metal::clamp(grazing, 0.0f, 1.0f)));
}

// A perfectly flat water plane with high-frequency shading normals maps a
// small normal change to a large sky or reflection change near the horizon.
// This pass has no water history, so retain wave detail for nearby views and
// smoothly reduce it only at the glancing angles where it would alias into
// horizontal reflection bands.
static inline float waterGrazingWaveDetail(float viewCosine) {
    const float grazing = 1.0f - metal::clamp(metal::abs(viewCosine), 0.0f, 1.0f);
    return 1.0f - metal::smoothstep(0.45f, 0.80f, grazing);
}

// A wave can be geometrically continuous yet still alias after it is reflected
// about a grazing interface. Measure the reflected-ray variation directly,
// rather than guessing from wave frequency alone: the camera's ordinary view
// gradient remains small, while a noisy water normal makes adjacent pixels
// point at unrelated sky or scene samples. Retire only that unresolved normal
// detail before reflection and refraction turn it into moving horizontal bands.
static inline float waterReflectionNormalVisibility(float reflectedRayFootprint) {
    if (!metal::isfinite(reflectedRayFootprint)) {
        return 0.0f;
    }
    return 1.0f - metal::smoothstep(0.012f, 0.065f, metal::max(reflectedRayFootprint, 0.0f));
}

// Propagated skylight is an ambient accessibility value, not a fractional
// multiplier for every exterior water reflection. An exposed source-water
// surface can differ by one propagated nibble across an exact or far handoff;
// treating that as a reflection gain paints an artificial grid. Keep a smooth
// sealed-cave gate, then let direct shadows and cloud visibility shape the
// lighting above open water.
static inline float waterExteriorSkyVisibility(float propagatedSkylight) {
    if (!metal::isfinite(propagatedSkylight)) {
        return 0.0f;
    }
    return metal::smoothstep(0.0f, 1.0f / 15.0f, metal::clamp(propagatedSkylight, 0.0f, 1.0f));
}

// The slant distance to a single depth receiver is valid only while that
// receiver is stable. Once grazing transmission falls back, blend its optical
// depth to a modest water-body depth too. This keeps a one-pixel terrain or
// LOD step from leaking back through the small non-reflective Fresnel tail as
// a dark rectangular pane.
static inline float waterStabilizedOpticalDepth(float waterRayDistance,
                                                float transmissionVisibility) {
    constexpr float FALLBACK_DEPTH_BLOCKS = 4.0f;
    if (!metal::isfinite(waterRayDistance) || !metal::isfinite(transmissionVisibility)) {
        return FALLBACK_DEPTH_BLOCKS;
    }
    const float rawDepth = metal::clamp(waterRayDistance, 0.0f, 64.0f);
    return metal::mix(FALLBACK_DEPTH_BLOCKS, rawDepth,
                      metal::clamp(transmissionVisibility, 0.0f, 1.0f));
}

// Refraction reads one opaque scene sample behind the water interface. At a
// long grazing path, that sample can change across a voxel edge or a far
// terrain replacement by several blocks between adjacent pixels. The receiver
// derivative finds discontinuities, but a shallow, distant, flat receiver can
// stay locally smooth inside a coarse tile and still expose a large colored
// pane. Retire that transmission when the water interface itself becomes
// under-sampled or distant. There is no stable transmission history in the
// water pass, so the fallback must be continuous reflection instead. Near,
// non-grazing, continuous receivers retain their full refraction. A missing
// opaque receiver has no valid transmission sample and returns zero.
static inline float waterRefractionVisibility(float viewCosine, float waterRayDistance,
                                              float verticalWaterDepth, float waterSurfaceDistance,
                                              float waterSurfaceFootprint, float receiverFootprint,
                                              bool hasOpaqueReceiver) {
    if (!hasOpaqueReceiver || !metal::isfinite(viewCosine) || !metal::isfinite(waterRayDistance) ||
        !metal::isfinite(verticalWaterDepth) || !metal::isfinite(waterSurfaceDistance) ||
        !metal::isfinite(waterSurfaceFootprint) || !metal::isfinite(receiverFootprint)) {
        return 0.0f;
    }
    const float grazing =
        metal::smoothstep(0.30f, 0.78f, 1.0f - metal::clamp(metal::abs(viewCosine), 0.0f, 1.0f));
    const float longPath = metal::smoothstep(6.0f, 32.0f, metal::max(waterRayDistance, 0.0f));
    const float discontinuity = metal::smoothstep(1.5f, 6.0f, metal::max(receiverFootprint, 0.0f));
    // At a glancing angle, the ray distance is much longer than the physical
    // water column. Reliability depends on the latter: a shallow lake bed can
    // still be many blocks away along the refracted ray.
    const float shallowReceiver =
        1.0f - metal::smoothstep(3.0f, 12.0f, metal::max(verticalWaterDepth, 0.0f));
    // A water pass has no transmission history. Beyond this small near-field
    // region, even a locally smooth receiver can be replaced by a different
    // terrain LOD or stream-in result on the next pixel or frame. That is
    // unsafe at every view angle, not only at the horizon. Reflection is
    // continuous there, while residual refraction exposes large panes.
    const float distantSurface =
        metal::smoothstep(8.0f, 32.0f, metal::max(waterSurfaceDistance, 0.0f));
    const float unresolvedSurface =
        metal::smoothstep(0.025f, 0.125f, metal::max(waterSurfaceFootprint, 0.0f));
    const float shallowUnresolvedTransmission = grazing * shallowReceiver * unresolvedSurface;
    const float instability = metal::max(metal::max(grazing * longPath, grazing * discontinuity),
                                         metal::max(distantSurface, shallowUnresolvedTransmission));
    // A small partial transmission tail still reveals a full coarse receiver
    // through Fresnel. Snap the reliability transition to reflection before
    // that tail can read as a dark terrain slab.
    return 1.0f - metal::smoothstep(0.10f, 0.35f, instability);
}

// Screen-space derivative winding follows the UV convention rather than the
// receiver's physical side. Orient the reconstructed normal toward the camera
// before testing its world Y so floors viewed from above remain +Y while
// ceilings viewed from below remain -Y.
static inline metal::float3
orientUnderwaterReceiverNormalTowardCamera(metal::float3 normal, metal::float3 cameraToReceiver) {
    return metal::dot(normal, -cameraToReceiver) < 0.0f ? -normal : normal;
}

// Underwater caustics belong on stable, upward-facing submerged receivers.
// Never use abs(normalY): that turns an opposite-oriented or
// silhouette-corrupted normal into a false floor.
static inline float underwaterCausticSurfaceConfidence(float normalY, float edgeSpan,
                                                       float viewDistance) {
    if (!metal::isfinite(normalY) || !metal::isfinite(edgeSpan) || !metal::isfinite(viewDistance)) {
        return 0.0f;
    }
    const float upFacing = metal::smoothstep(0.60f, 0.90f, normalY);
    const float distance = metal::max(viewDistance, 0.0f);
    const float stableEdge =
        1.0f - metal::smoothstep(distance * 0.04f + 0.15f, distance * 0.08f + 0.50f,
                                 metal::max(edgeSpan, 0.0f));
    return upFacing * stableEdge;
}

// Foliage wind sway, ONE definition shared by the scene and shadow vertex
// stages so shadows track the displaced geometry exactly (the same
// never-drift rule as applyFog). sway: 1 = flora (v is the cross-quad texture
// v, 0 at the tip, so the base stays rooted), 2 = leaves (a continuous field of
// world position, so vertices shared by merged quads displace identically and
// the canopy never cracks). strength 0 (the waving setting off) is a no-op.
static inline metal::float3 applySway(metal::float3 worldPos, uint sway, float v, float time,
                                      constant FoliageWindUniforms& wind) {
    const float speed =
        metal::clamp(wind.speedBlocksPerSecond, 0.0f, FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND);
    if (sway == 0u || wind.strength <= 0.0f || speed <= 1e-4f) {
        return worldPos;
    }
    const metal::float2 direction = metal::normalize(wind.direction);
    const metal::float2 crosswind = metal::float2(-direction.y, direction.x);
    const float response = metal::sqrt(speed / FOLIAGE_WIND_MAX_BLOCKS_PER_SECOND);
    if (sway == 1u) {
        const float rootWeight = (1.0f - v) * (1.0f - v);
        const metal::float2 cell = metal::floor(worldPos.xz) + 0.5f;
        const float alongWind = metal::dot(cell, direction) - time * speed;
        const float acrossWind = metal::dot(cell, crosswind);
        const float phase = alongWind * 0.75f + acrossWind * 0.21f;
        const float gust = metal::sin(phase) + 0.4f * metal::sin(phase * 1.7f + acrossWind * 0.3f);
        const float flutter = metal::sin(phase * 2.3f - acrossWind * 0.6f);
        const float amplitude =
            (0.015f * response * response + 0.070f * response) * rootWeight * wind.strength;
        worldPos.xz += direction * gust * amplitude + crosswind * flutter * amplitude * 0.22f;
    } else {
        const float alongWind = metal::dot(worldPos.xz, direction) - time * speed;
        const float acrossWind = metal::dot(worldPos.xz, crosswind);
        const float phase = alongWind * 0.35f + worldPos.y * 0.21f + acrossWind * 0.14f;
        const float gust = metal::sin(phase) + 0.32f * metal::sin(phase * 1.9f + acrossWind * 0.2f);
        const float flutter = metal::cos(phase * 1.3f - worldPos.y * 0.17f);
        const float amplitude = (0.006f * response * response + 0.026f * response) * wind.strength;
        worldPos.xz += direction * gust * amplitude + crosswind * flutter * amplitude * 0.18f;
    }
    return worldPos;
}

// Water surface waves. The three directional waves live in one table so the
// filtered fragment normal (waterSurfaceNormal) stays consistent across all
// water representations. Stable water geometry stays planar; phase animation
// changes shading only, which keeps exact and far source planes continuous.
struct WaterWave {
    metal::float2 dir; // travel direction (unit-ish)
    float freq;        // spatial frequency (radians per block)
    float amp;         // crest height in blocks
    float speed;       // temporal frequency (radians per second)
};
// These amplitudes provide visible low-frequency slope at eye level while the
// filtered normal keeps grazing reflection and refraction stable.
constant WaterWave WATER_WAVES[3] = {
    {metal::float2(0.80f, 0.60f), 0.52f, 0.110f, 1.1f},
    {metal::float2(-0.50f, 0.87f), 0.80f, 0.065f, 1.5f},
    {metal::float2(0.20f, -0.98f), 1.05f, 0.040f, 2.1f},
};

// Analytic slope of the canonical wave field
// (d/dp of A*sin(dot(p,D)*K + t*S) = A*D*K*cos(...)) plus a faint fine ripple
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

static inline float waterSsrSmoothstep(float edge0, float edge1, float value) {
    const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// CPU mirrors of the SSR quality transfer functions above. They pin the
// grazing fallback contract without needing a Metal device in unit tests.
static inline float waterSsrReflectionMipLevel(float grazing, float hitDistance) {
    const float horizon = waterSsrSmoothstep(0.20f, 0.60f, std::clamp(grazing, 0.0f, 1.0f));
    const float distance = waterSsrSmoothstep(4.0f, 64.0f, std::max(hitDistance, 0.0f));
    return 4.0f * horizon * (0.50f + 0.50f * distance);
}

static inline float waterSsrStabilityConfidence(float grazing, float hitDistance) {
    const float horizon = waterSsrSmoothstep(0.20f, 0.58f, std::clamp(grazing, 0.0f, 1.0f));
    const float distance = waterSsrSmoothstep(12.0f, 56.0f, std::max(hitDistance, 0.0f));
    return 1.0f - horizon * distance;
}

static inline float waterSsrJitterAmplitude(float grazing) {
    const float blend = waterSsrSmoothstep(0.50f, 0.90f, std::clamp(grazing, 0.0f, 1.0f));
    return 0.24f + (0.04f - 0.24f) * blend;
}

static inline float waterGrazingWaveDetail(float viewCosine) {
    const float grazing = 1.0f - std::clamp(std::abs(viewCosine), 0.0f, 1.0f);
    return 1.0f - waterSsrSmoothstep(0.45f, 0.80f, grazing);
}

// CPU mirror of the reflected-ray derivative filter. This pins the threshold
// that prevents a continuous wave field from choosing unrelated reflection
// samples at a grazing view.
static inline float waterReflectionNormalVisibility(float reflectedRayFootprint) {
    if (!std::isfinite(reflectedRayFootprint)) {
        return 0.0f;
    }
    return 1.0f - waterSsrSmoothstep(0.012f, 0.065f, std::max(reflectedRayFootprint, 0.0f));
}

// CPU mirror of the exterior-versus-sealed water gate. Its low threshold
// avoids exact/far reflection seams from harmless skylight-nibble differences.
static inline float waterExteriorSkyVisibility(float propagatedSkylight) {
    if (!std::isfinite(propagatedSkylight)) {
        return 0.0f;
    }
    return waterSsrSmoothstep(0.0f, 1.0f / 15.0f, std::clamp(propagatedSkylight, 0.0f, 1.0f));
}

// CPU mirror of the stable optical-depth fallback used when grazing
// transmission cannot trust its one-pixel opaque receiver.
static inline float waterStabilizedOpticalDepth(float waterRayDistance,
                                                float transmissionVisibility) {
    constexpr float FALLBACK_DEPTH_BLOCKS = 4.0f;
    if (!std::isfinite(waterRayDistance) || !std::isfinite(transmissionVisibility)) {
        return FALLBACK_DEPTH_BLOCKS;
    }
    const float rawDepth = std::clamp(waterRayDistance, 0.0f, 64.0f);
    return FALLBACK_DEPTH_BLOCKS +
           (rawDepth - FALLBACK_DEPTH_BLOCKS) * std::clamp(transmissionVisibility, 0.0f, 1.0f);
}

// CPU mirror of the water transmission reliability gate above. The tests pin
// its grazing, long-path, distant-surface, discontinuity, and missing-receiver
// behavior without requiring a Metal device.
static inline float waterRefractionVisibility(float viewCosine, float waterRayDistance,
                                              float verticalWaterDepth, float waterSurfaceDistance,
                                              float waterSurfaceFootprint, float receiverFootprint,
                                              bool hasOpaqueReceiver) {
    if (!hasOpaqueReceiver || !std::isfinite(viewCosine) || !std::isfinite(waterRayDistance) ||
        !std::isfinite(verticalWaterDepth) || !std::isfinite(waterSurfaceDistance) ||
        !std::isfinite(waterSurfaceFootprint) || !std::isfinite(receiverFootprint)) {
        return 0.0f;
    }
    const float grazing =
        waterSsrSmoothstep(0.30f, 0.78f, 1.0f - std::clamp(std::abs(viewCosine), 0.0f, 1.0f));
    const float longPath = waterSsrSmoothstep(6.0f, 32.0f, std::max(waterRayDistance, 0.0f));
    const float discontinuity = waterSsrSmoothstep(1.5f, 6.0f, std::max(receiverFootprint, 0.0f));
    const float shallowReceiver =
        1.0f - waterSsrSmoothstep(3.0f, 12.0f, std::max(verticalWaterDepth, 0.0f));
    const float distantSurface =
        waterSsrSmoothstep(8.0f, 32.0f, std::max(waterSurfaceDistance, 0.0f));
    const float unresolvedSurface =
        waterSsrSmoothstep(0.025f, 0.125f, std::max(waterSurfaceFootprint, 0.0f));
    const float shallowUnresolvedTransmission = grazing * shallowReceiver * unresolvedSurface;
    const float instability = std::max(std::max(grazing * longPath, grazing * discontinuity),
                                       std::max(distantSurface, shallowUnresolvedTransmission));
    return 1.0f - waterSsrSmoothstep(0.10f, 0.35f, instability);
}

// CPU mirror of the normal orientation used before the underwater receiver
// gate. Keep it alongside the Metal form so tests can cover floor and ceiling
// winding without a graphics device.
static inline simd_float3
orientUnderwaterReceiverNormalTowardCamera(simd_float3 normal,
                                           simd_float3 cameraToReceiver) noexcept {
    return simd_dot(normal, -cameraToReceiver) < 0.0f ? -normal : normal;
}

// CPU mirror of the underwater caustic receiver gate. Keep the angular and
// discontinuity thresholds alongside the Metal form so focused tests can
// reject wall, ceiling, and silhouette leakage without a graphics device.
static inline float underwaterCausticSurfaceConfidence(float normalY, float edgeSpan,
                                                       float viewDistance) {
    if (!std::isfinite(normalY) || !std::isfinite(edgeSpan) || !std::isfinite(viewDistance)) {
        return 0.0f;
    }
    const float upFacing = waterSsrSmoothstep(0.60f, 0.90f, normalY);
    const float distance = std::max(viewDistance, 0.0f);
    const float stableEdge =
        1.0f - waterSsrSmoothstep(distance * 0.04f + 0.15f, distance * 0.08f + 0.50f,
                                  std::max(edgeSpan, 0.0f));
    return upFacing * stableEdge;
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
    FoliageWindUniforms foliageWind;
    float time;    // seconds; drives foliage sway
    float wetness; // 0 dry .. 1 soaked (rain darkening + sheen)
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

#ifdef __METAL_VERSION__
static inline bool
farTerrainExactColumnOwnsFragment(simd_float2 localPosition, unsigned int face,
                                  bool useEmittingColumn,
                                  constant FarTerrainOwnershipUniforms& ownership) {
    localPosition = farTerrainExactOwnershipSamplePosition(localPosition, face, useEmittingColumn);
    const metal::int2 neighbor =
        metal::int2(metal::floor(localPosition / FAR_TERRAIN_TILE_EDGE_BLOCKS));
    if (metal::any(neighbor < metal::int2(-FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS)) ||
        metal::any(neighbor > metal::int2(FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS))) {
        return false;
    }
    const metal::float2 neighborLocal =
        localPosition - metal::float2(neighbor) * FAR_TERRAIN_TILE_EDGE_BLOCKS;
    const metal::int2 column = metal::clamp(
        metal::int2(metal::floor(neighborLocal / FAR_TERRAIN_EXACT_COLUMN_EDGE_BLOCKS)),
        metal::int2(0), metal::int2(FAR_TERRAIN_EXACT_COLUMNS_PER_TILE - 1));
    const unsigned int bit = unsigned(column.y * FAR_TERRAIN_EXACT_COLUMNS_PER_TILE + column.x);
    const unsigned int word = bit / FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD;
    const unsigned int tileIndex = unsigned((neighbor.y + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS) *
                                                FAR_TERRAIN_EXACT_MASK_NEIGHBOR_EDGE +
                                            neighbor.x + FAR_TERRAIN_EXACT_MASK_NEIGHBOR_RADIUS);
    const simd_uint4 packed =
        ownership.readyColumnMasks[tileIndex * FAR_TERRAIN_EXACT_MASK_VECTORS_PER_TILE +
                                   word / FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR];
    return ((packed[word % FAR_TERRAIN_EXACT_MASK_WORDS_PER_VECTOR] >>
             (bit % FAR_TERRAIN_EXACT_MASK_BITS_PER_WORD)) &
            1u) != 0u;
}
#endif

// One cascade's light view-projection, bound at buffer(1) via setVertexBytes
// in the depth-only shadow pass (shadow.metal). time and foliageWind must
// match the scene pass exactly or foliage shadows detach from the blades.
struct ShadowPassUniforms {
    simd_float4x4 lightViewProj;
    simd_float4 projectionOrigin;
    FoliageWindUniforms foliageWind;
    float time;
};

// Macros, not constexpr values: MSL rejects program-scope constants outside
// the constant address space, and both languages accept these array bounds.
#define SHADOW_DETAILED_CASCADE_COUNT 4
#define SHADOW_CASCADE_COUNT 5
#define SHADOW_HORIZON_CASCADE_INDEX 4
#define SHADOW_CASCADE_BLEND_FRACTION 0.125f
#define SHADOW_HORIZON_DISTANCE 8192.0f

// One directional-shadow projection. depthRange stores the projection near
// depth, selection far depth, overlap start, and valid-coverage flag.
// samplingParams stores world units per texel, receiver normal offset, filter
// radius in texels, and receiver depth bias. Keeping every sampling parameter
// beside its matrix prevents a high-resolution near cascade from inheriting a
// horizon-scale bias or filter width.
struct ShadowCascadeUniforms {
    simd_float4x4 lightViewProj;
    // The matrix consumes coordinates relative to this nearby world anchor.
    // Keeping six-digit world translations out of a 4K near-cascade matrix
    // preserves sub-texel precision at the documented large-coordinate route.
    simd_float4 projectionOrigin;
    simd_float4 depthRange;
    simd_float4 samplingParams;
};

// Scene shadow data bound at buffer(4). Selection uses camera-forward view
// depth, matching the frustum slices used to construct the projections. The
// first two records sample the near array, the next two sample the far array,
// and record four samples the horizon texture.
struct ShadowUniforms {
    ShadowCascadeUniforms cascades[SHADOW_CASCADE_COUNT];
    simd_float4 cameraPositionAndStrength;
    simd_float4 cameraForwardAndPadding;
};

// Shared CPU and Metal cascade selection result. The secondary cascade equals
// the primary outside an overlap, and exteriorWeight fades the final horizon
// band to propagated exterior visibility before shadow coverage ends.
struct ShadowCascadeSelection {
    unsigned int primary;
    unsigned int secondary;
    float secondaryWeight;
    float exteriorWeight;
    unsigned int covered;
};

static inline float shadowViewDepth(simd_float3 worldPosition, simd_float3 cameraPosition,
                                    simd_float3 cameraForward) {
    const simd_float3 offset = worldPosition - cameraPosition;
    return offset.x * cameraForward.x + offset.y * cameraForward.y + offset.z * cameraForward.z;
}

static inline float shadowVisibilityWithStrength(float visibility, float strength) {
#ifdef __METAL_VERSION__
    return metal::mix(1.0f, metal::saturate(visibility), metal::saturate(strength));
#else
    return std::lerp(1.0f, std::clamp(visibility, 0.0f, 1.0f), std::clamp(strength, 0.0f, 1.0f));
#endif
}

static inline ShadowCascadeSelection shadowCascadeSelection(float viewDepth,
#ifdef __METAL_VERSION__
                                                            constant ShadowUniforms& shadow
#else
                                                            const ShadowUniforms& shadow
#endif
) {
    ShadowCascadeSelection result{0u, 0u, 0.0f, 0.0f, 0u};
    if (viewDepth < shadow.cascades[0].depthRange.x ||
        viewDepth > shadow.cascades[SHADOW_HORIZON_CASCADE_INDEX].depthRange.y) {
        return result;
    }

    for (unsigned int cascade = 0u; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        const simd_float4 range = shadow.cascades[cascade].depthRange;
        if (viewDepth > range.y) {
            continue;
        }
        result.primary = cascade;
        result.secondary = cascade;
        result.covered = 1u;
        if (viewDepth >= range.z) {
            float amount = (viewDepth - range.z) / (range.y - range.z);
#ifdef __METAL_VERSION__
            amount = metal::clamp(amount, 0.0f, 1.0f);
#else
            amount = std::clamp(amount, 0.0f, 1.0f);
#endif
            const float smoothAmount = amount * amount * (3.0f - 2.0f * amount);
            if (cascade == SHADOW_HORIZON_CASCADE_INDEX) {
                result.exteriorWeight = smoothAmount;
            } else {
                result.secondary = cascade + 1u;
                result.secondaryWeight = smoothAmount;
            }
        }
        return result;
    }
    return result;
}

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
    simd_float3 directLightDirection;
    simd_float3 directLightRadiance;
    simd_float3 cameraPosition;
    simd_float3 fogColor;
    simd_float2 resolution; // scene color/depth texture size in pixels
    float fogDensity;
    float time;                 // seconds; drives waves + caustics
    float cameraUnderwater;     // 1.0 when the camera is inside water
    float ssrStrength;          // 0 = sky-only reflection (the pre-SSR look)
    float skyExposure;          // 0 when solid ground seals the camera's water
                                // column (aquifers, roofed lakes): no sunlight
                                // reaches covered water, so caustics and the
                                // sun-driven murk must go dark, not track the sun
    float waterSurfaceY;        // world Y of the surface of the water body the
                                // camera is in: upward rays leave the water there,
                                // so murk and caustics must stop at that exit
                                // instead of fogging out to the opaque depth
    simd_float3 solarDirection; // true Sun direction for atmosphere reflection
    float physicalSkyBlend;     // 1 in daylight, 0 after astronomical twilight
    float directSpecularFactor; // 1 for Sun, lunar phase energy for Moon
};

// Atmospheric sky, bound at buffer(1) in sky.metal. The fragment shader
// reconstructs a per-pixel view ray from the camera basis (like clouds.metal)
// so the sun and moon are true direction-projected discs, not screen-space
// blobs, and the gradient tracks where the camera looks. zenith/horizon come
// from computeDayNightUniforms (also the fog palette, one source).
struct SkyUniforms {
    simd_float3 cameraForward;
    simd_float3 cameraRight;
    simd_float3 cameraUp;
    simd_float3 sunDirection;
    simd_float3 moonDirection; // phase-relative direction from the saved clock
    simd_float3 sunColor;
    simd_float3 moonColor;
    simd_float3 zenithColor;
    simd_float3 horizonColor;
    simd_float4 visibilityAndPhase; // Sun, Moon, phase energy, stars
    float tanHalfFov;
    float aspect;
};

#define WEATHER_MAP_EDGE WEATHER_GRID_EDGE
#define WEATHER_MAP_CELL_SPACING WEATHER_GRID_CELL_SPACING_BLOCKS
#define ATMOSPHERE_TRANSMITTANCE_WIDTH 256
#define ATMOSPHERE_TRANSMITTANCE_HEIGHT 64
#define ATMOSPHERE_MULTISCATTER_WIDTH 32
#define ATMOSPHERE_MULTISCATTER_HEIGHT 32
#define ATMOSPHERE_SKY_VIEW_WIDTH 192
#define ATMOSPHERE_SKY_VIEW_HEIGHT 108
#define ATMOSPHERE_TRANSMITTANCE_MU_MIN (-0.15f)
#define ATMOSPHERE_TRANSMITTANCE_MU_MAX 1.0f
// Atmospheric phase functions integrate to one over 4 pi steradians. Rycraft
// terrain and volumetric lighting use scene-normalized directional irradiance,
// so atmosphere LUT radiance restores that solid-angle normalization before
// entering the shared HDR exposure pipeline. A clear, sunlit sky covers most
// of the exposure meter while direct sunlight occupies only a tiny disc, so
// the scene calibration reserves four such solid angles for sky radiance.
// This keeps a clear noon sky visibly blue without raising terrain exposure.
#define ATMOSPHERE_RADIANCE_SCALE (16.0f * 3.14159265358979323846f)

// The transmittance LUT dedicates a small negative-mu interval to near-horizon
// and downward rays. Keep its coordinate transform shared by every producer
// and consumer: sampling with raw mu would shift the multiple-scattering
// directions toward the horizon and no longer address the values written by
// the transmittance kernel.
static inline float atmosphereTransmittanceUnitCoordinate(float value) {
#ifdef __METAL_VERSION__
    const float finiteValue = metal::isfinite(value) ? value : 0.0f;
    return metal::saturate(finiteValue);
#else
    const float finiteValue = std::isfinite(value) ? value : 0.0F;
    return std::clamp(finiteValue, 0.0F, 1.0F);
#endif
}

static inline float atmosphereTransmittanceMuUv(float mu) {
    return atmosphereTransmittanceUnitCoordinate(
        (mu - ATMOSPHERE_TRANSMITTANCE_MU_MIN) /
        (ATMOSPHERE_TRANSMITTANCE_MU_MAX - ATMOSPHERE_TRANSMITTANCE_MU_MIN));
}

static inline float atmosphereTransmittanceUvMu(float uv) {
    return ATMOSPHERE_TRANSMITTANCE_MU_MIN +
           (ATMOSPHERE_TRANSMITTANCE_MU_MAX - ATMOSPHERE_TRANSMITTANCE_MU_MIN) *
               atmosphereTransmittanceUnitCoordinate(uv);
}

// A transmittance lookup follows a ray only until it leaves the atmosphere or
// reaches the planet's lower boundary. Keeping this small geometric contract
// shared with Metal prevents the LUT's downward rows from integrating through
// solid ground toward the far side of the atmosphere.
static inline float atmosphereRayToSphereDistance(float radius, float mu, float sphereRadius) {
    const float discriminant = radius * radius * (mu * mu - 1.0f) + sphereRadius * sphereRadius;
#ifdef __METAL_VERSION__
    return metal::max(-radius * mu + metal::sqrt(metal::max(discriminant, 0.0f)), 0.0f);
#else
    return std::max(-radius * mu + std::sqrt(std::max(discriminant, 0.0F)), 0.0F);
#endif
}

static inline bool atmosphereRayHitsGround(float radius, float mu, float groundRadius) {
    if (mu >= 0.0f) {
        return false;
    }
    const float discriminant = radius * radius * (mu * mu - 1.0f) + groundRadius * groundRadius;
#ifdef __METAL_VERSION__
    return discriminant >= 0.0f && -radius * mu - metal::sqrt(discriminant) > 0.0f;
#else
    return discriminant >= 0.0F && -radius * mu - std::sqrt(discriminant) > 0.0F;
#endif
}

static inline float atmosphereTransmittancePathLength(float radius, float mu, float groundRadius,
                                                      float topRadius) {
    const float topDistance = atmosphereRayToSphereDistance(radius, mu, topRadius);
    if (!atmosphereRayHitsGround(radius, mu, groundRadius)) {
        return topDistance;
    }
    const float discriminant = radius * radius * (mu * mu - 1.0f) + groundRadius * groundRadius;
#ifdef __METAL_VERSION__
    const float groundDistance =
        metal::max(-radius * mu - metal::sqrt(metal::max(discriminant, 0.0f)), 0.0f);
    return metal::min(topDistance, groundDistance);
#else
    const float groundDistance =
        std::max(-radius * mu - std::sqrt(std::max(discriminant, 0.0F)), 0.0F);
    return std::min(topDistance, groundDistance);
#endif
}

#ifdef __METAL_VERSION__
static inline metal::float3 atmosphereSceneRadiance(metal::float3 radiance) {
    return radiance * ATMOSPHERE_RADIANCE_SCALE;
}

// The spherical sky view reaches the physical ground for rays below the
// horizon. Returning a black terminator there exposes a false void wherever
// streamed terrain has not yet filled the frame, so use the same Lambertian
// ground response that supplies the atmosphere's lower boundary condition.
static inline metal::float3 atmosphereGroundRadiance(metal::float3 groundAlbedo,
                                                     metal::float3 directIrradiance,
                                                     float localSunMu,
                                                     metal::float3 diffuseIrradiance) {
    return groundAlbedo *
           (directIrradiance * metal::max(localSunMu, 0.0f) + diffuseIrradiance * 0.20f) *
           0.31830988618379067154f;
}
#else
static inline simd_float3 atmosphereSceneRadiance(simd_float3 radiance) noexcept {
    return radiance * ATMOSPHERE_RADIANCE_SCALE;
}

static inline simd_float3 atmosphereGroundRadiance(simd_float3 groundAlbedo,
                                                   simd_float3 directIrradiance, float localSunMu,
                                                   simd_float3 diffuseIrradiance) noexcept {
    return groundAlbedo *
           (directIrradiance * std::max(localSunMu, 0.0F) + diffuseIrradiance * 0.20F) *
           0.31830988618379067154F;
}
#endif
#define CLOUD_BASE_NOISE_EDGE 128
#define CLOUD_EROSION_NOISE_EDGE 32
#define CLOUD_HIGH_VIEW_STEPS 48
#define CLOUD_MEDIUM_VIEW_STEPS 24
#define CLOUD_HIGH_LIGHT_STEPS 6
#define CLOUD_MEDIUM_LIGHT_STEPS 3
#define CLOUD_NOISE_BLOCK_FREQUENCY 0.00045f
#define CLOUD_MOTION_WRAP_BLOCKS 2222.2222f
#define CLOUD_HORIZON_VIEW_DEPTH 8192.0f
#define FROXEL_WIDTH 160
#define FROXEL_HEIGHT 104
#define FROXEL_DEPTH 64

// Camera-centered weather-map transform. Grid coordinates are reconstructed
// from large-coordinate world X/Z on the CPU before these float values reach
// Metal, keeping the shader sampling neighborhood close to zero.
struct WeatherMapUniforms {
    simd_float2 originXZ;
    float cellSpacing;
    float interpolation;
    simd_uint2 gridSize;
    float motionWrapBlocks;
    float movementMargin;
};

#ifdef __METAL_VERSION__
static inline metal::float2 weatherMapTextureCoordinate(metal::float2 cameraRelativeXZ,
                                                        constant WeatherMapUniforms& weatherMap) {
    const metal::float2 cell = (cameraRelativeXZ - weatherMap.originXZ) / weatherMap.cellSpacing;
    return (cell + 0.5f) / metal::float2(weatherMap.gridSize);
}
#else
static inline simd_float2
weatherMapTextureCoordinate(simd_float2 cameraRelativeXZ,
                            const WeatherMapUniforms& weatherMap) noexcept {
    const simd_float2 cell = (cameraRelativeXZ - weatherMap.originXZ) / weatherMap.cellSpacing;
    return (cell + simd_make_float2(0.5f, 0.5f)) /
           simd_make_float2(static_cast<float>(weatherMap.gridSize.x),
                            static_cast<float>(weatherMap.gridSize.y));
}
#endif

// Physical atmosphere parameters shared by the LUT compute passes and sky.
// Radii and scale heights are kilometers; the renderer converts the voxel
// camera altitude once on the CPU instead of mixing world units in shaders.
struct AtmosphereUniforms {
    simd_float3 cameraPositionKm;
    simd_float3 sunDirection;
    simd_float3 sunRadiance;
    simd_float3 groundAlbedo;
    simd_float4 rayleighScatteringAndScaleHeight;
    simd_float4 mieScatteringAndScaleHeight;
    simd_float4 ozoneAbsorptionAndCenter;
    simd_float4 atmosphereRadii;
    simd_float4 weatherOptics;
    simd_float4 renderParams;
};

// Quarter-resolution volumetric cloud march and temporal reprojection data.
// The regional motion textures carry wrapped per-cell cloud phase. Keeping
// both map transforms lets reprojection sample the exact prior frame field.
struct CloudRenderUniforms {
    simd_float4x4 invViewProjection;
    simd_float4x4 previousViewProjection;
    simd_float3 cameraPosition;
    simd_float3 cameraForward;
    simd_float3 sunDirection;
    simd_float3 sunRadiance;
    simd_float3 skyIrradiance;
    simd_float4 layerBounds;
    simd_float4 densityParams;
    simd_float4 phaseParams;
    simd_float4 renderParams;
    simd_float4 resolutionAndFrame;
    WeatherMapUniforms weatherMap;
    WeatherMapUniforms previousWeatherMap;
};

// Converts the camera-forward horizon contract into distance along one view
// ray. The weather cap keeps every density lookup inside the regional map,
// reserving one cell for filtered samples and the short light march.
static inline float cloudViewDepthRayDistance(simd_float3 ray, simd_float3 cameraForward,
                                              float viewDepth) {
    const float forwardProjection =
        ray.x * cameraForward.x + ray.y * cameraForward.y + ray.z * cameraForward.z;
    if (forwardProjection <= 1.0e-4f) {
        return 0.0f;
    }
    return viewDepth / forwardProjection;
}

static inline float cloudWeatherCoverageRayDistance(simd_float3 ray,
#ifdef __METAL_VERSION__
                                                    constant WeatherMapUniforms& weatherMap
#else
                                                    const WeatherMapUniforms& weatherMap
#endif
) {
    const float guard = weatherMap.cellSpacing;
    const float minimumX = weatherMap.originXZ.x + guard;
    const float minimumZ = weatherMap.originXZ.y + guard;
    const float maximumX =
        weatherMap.originXZ.x + float(weatherMap.gridSize.x - 1u) * weatherMap.cellSpacing - guard;
    const float maximumZ =
        weatherMap.originXZ.y + float(weatherMap.gridSize.y - 1u) * weatherMap.cellSpacing - guard;
    float distance = 1.0e7f;
    if (ray.x > 1.0e-4f) {
#ifdef __METAL_VERSION__
        distance = metal::min(distance, maximumX / ray.x);
#else
        distance = std::min(distance, maximumX / ray.x);
#endif
    } else if (ray.x < -1.0e-4f) {
#ifdef __METAL_VERSION__
        distance = metal::min(distance, minimumX / ray.x);
#else
        distance = std::min(distance, minimumX / ray.x);
#endif
    }
    if (ray.z > 1.0e-4f) {
#ifdef __METAL_VERSION__
        distance = metal::min(distance, maximumZ / ray.z);
#else
        distance = std::min(distance, maximumZ / ray.z);
#endif
    } else if (ray.z < -1.0e-4f) {
#ifdef __METAL_VERSION__
        distance = metal::min(distance, minimumZ / ray.z);
#else
        distance = std::min(distance, minimumZ / ray.z);
#endif
    }
#ifdef __METAL_VERSION__
    return metal::max(distance, 0.0f);
#else
    return std::max(distance, 0.0f);
#endif
}

static inline float cloudMarchRayDistanceLimit(simd_float3 ray, simd_float3 cameraForward,
                                               float viewDepth,
#ifdef __METAL_VERSION__
                                               constant WeatherMapUniforms& weatherMap
#else
                                               const WeatherMapUniforms& weatherMap
#endif
) {
    const float horizon = cloudViewDepthRayDistance(ray, cameraForward, viewDepth);
    const float weather = cloudWeatherCoverageRayDistance(ray, weatherMap);
#ifdef __METAL_VERSION__
    return metal::min(horizon, weather);
#else
    return std::min(horizon, weather);
#endif
}

// Bilateral cloud upscale weights. A clear or geometry-occluded low-resolution
// tap contributes transparent spatial coverage to the normalization term;
// omitting that weight would renormalize one neighboring cloud tap to full
// opacity and bleed clouds across terrain silhouettes. Visible hits retain a
// small distance preference to keep nearer cloud structure crisp.
static inline simd_float2 cloudCompositeTapWeights(float spatialWeight, float hitDepth,
                                                   float opaqueDistance) {
    if (spatialWeight <= 0.0f) {
        return {0.0f, 0.0f};
    }
    if (hitDepth <= 0.0f || hitDepth >= opaqueDistance) {
        return {0.0f, spatialWeight};
    }
    const float visibleWeight = spatialWeight / (1.0f + hitDepth * 0.00025f);
    return {visibleWeight, visibleWeight};
}

// Snapped camera-centered cloud-shadow projection. footprintAndTexel stores
// the world footprint, world units per texel, opacity scale, and frame index.
struct CloudShadowUniforms {
    simd_float3 cameraPosition;
    simd_float3 sunDirection;
    simd_float4 footprintAndTexel;
    WeatherMapUniforms weatherMap;
};

// Cloud transmittance is generated for rays leaving a snapped reference
// plane at this height. Project every receiver onto that same ray before
// looking up the map so mountains, clouds, terrain, water, and fog agree.
#define CLOUD_SHADOW_REFERENCE_HEIGHT 64.0f
static inline simd_float2 cloudShadowReferencePosition(simd_float3 worldPosition,
                                                       simd_float3 lightDirection) {
#ifdef __METAL_VERSION__
    const float vertical = metal::abs(lightDirection.y) >= 0.03f ? lightDirection.y : 0.03f;
#else
    const float vertical = std::abs(lightDirection.y) >= 0.03f ? lightDirection.y : 0.03f;
#endif
    const float distanceToPlane = (CLOUD_SHADOW_REFERENCE_HEIGHT - worldPosition.y) / vertical;
    return {worldPosition.x + lightDirection.x * distanceToPlane,
            worldPosition.z + lightDirection.z * distanceToPlane};
}

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
    simd_float4 lighting; // propagated skylight, block light, reserved, reserved
};

// Bound at buffer(1) in particles.metal.
struct ParticleUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float3 cameraPosition;
    float atmosphericExtinction;
};

// Bound at buffer(0) in the bloom extract/blur passes (via setFragmentBytes
// as 32 bytes of per-pass state, not a real buffer).
struct BloomUniforms {
    simd_float2 resolution; // output texture width, height
    simd_float2 texelSize;  // 1/width, 1/height
    float threshold;        // extract luminance threshold
    float intensity;        // bloom strength multiplier
    float blurRadius;       // Kawase blur radius in texels
};

// Screen-space ray traced indirect light: GTAO-style occlusion plus one
// diffuse bounce, marched through a min-depth Hi-Z pyramid at reduced
// resolution, then temporally accumulated and spatially denoised. The current
// and previous matrices drive motion reprojection without a full TAA pass.
// traceParams carries (aoRadius, thickness, aoStrength, bounceIntensity),
// temporalParams carries (maxHistoryWeight, colorClampGamma, aoClampGamma,
// fireflyMaxLuminance), and filterParams carries (giMaxDistance,
// atrousLuminanceSigma, disocclusionAgeThreshold, reserved).
#define INDIRECT_HIGH_RAY_COUNT 4u
#define INDIRECT_MEDIUM_RAY_COUNT 2u
#define INDIRECT_HIGH_HIZ_ITERATION_CAP 24u
#define INDIRECT_MEDIUM_HIZ_ITERATION_CAP 16u
#define INDIRECT_HIGH_ATROUS_ITERATIONS 3u
#define INDIRECT_MEDIUM_ATROUS_ITERATIONS 2u
#define INDIRECT_HIZ_START_MIP 1u
#define INDIRECT_HIZ_MAX_MIP 7u
#define INDIRECT_HISTORY_MAX_AGE 32.0f
#define INDIRECT_SKY_LINEAR_DEPTH 60000.0f
struct IndirectLightingUniforms {
    simd_float4x4 projection;
    simd_float4x4 invProjection;
    simd_float4x4 invViewProjection;
    simd_float4x4 previousViewProjection;
    simd_float4 resolutionAndQuality;
    simd_float4 traceParams;
    simd_float4 temporalParams;
    simd_float4 filterParams;
    simd_float4 ambientAndFrame;
};

// Screen-space lighting advances its bounded rays in view space, then projects
// each sample for hierarchical depth lookup. Keeping this math shared makes
// the physical trace radius independent of render resolution, camera distance,
// and field of view while its screen footprint naturally follows projection.
static inline simd_float3 screenSpaceTraceViewSample(simd_float3 origin, simd_float3 direction,
                                                     float distance) {
    return origin + direction * distance;
}

static inline simd_float2 screenSpaceProjectViewPosition(simd_float3 viewPosition,
                                                         simd_float4x4 projection) {
#ifdef __METAL_VERSION__
    const metal::float4 clip = projection * metal::float4(viewPosition, 1.0f);
    const metal::float2 ndc = clip.xy / metal::max(clip.w, 1.0e-6f);
    return ndc * metal::float2(0.5f, -0.5f) + 0.5f;
#else
    const simd_float4 clip = simd_mul(
        projection, simd_make_float4(viewPosition.x, viewPosition.y, viewPosition.z, 1.0f));
    const simd_float2 ndc = clip.xy / std::max(clip.w, 1.0e-6f);
    return ndc * simd_make_float2(0.5f, -0.5f) + simd_make_float2(0.5f, 0.5f);
#endif
}

// History rejection operates in linear view depth. Device depth is highly
// compressed at a grazing floor, so a fixed normalized-depth threshold can
// reject every otherwise continuous history sample and expose the four-ray
// noise. This tolerance grows only with the receiver distance and remains far
// below a separate voxel face at normal cave-view ranges.
static inline float screenSpaceTemporalLinearDepthTolerance(float linearDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(linearDepth) || linearDepth <= 0.0f) {
        return 0.0f;
    }
    return metal::max(0.04f, linearDepth * 0.01f);
#else
    if (!std::isfinite(linearDepth) || linearDepth <= 0.0f) {
        return 0.0f;
    }
    return std::max(0.04f, linearDepth * 0.01f);
#endif
}

// The final indirect pass reconstructs a lower-resolution temporal history at
// native resolution. Keep the candidate's view depth close to the full-resolution
// receiver so GTAO and bounced radiance cannot cross a voxel edge. The soft
// falloff preserves smooth surfaces while the hard cutoff makes an unrelated
// block face contribute nothing.
static inline float screenSpaceBilateralDepthWeight(float receiverDepth, float candidateDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(receiverDepth) || !metal::isfinite(candidateDepth) ||
        receiverDepth <= 0.0f || candidateDepth <= 0.0f) {
        return 0.0f;
    }
    const float tolerance = metal::max(receiverDepth * 0.02f, 0.25f);
    const float difference = metal::abs(candidateDepth - receiverDepth);
    return difference > tolerance * 3.0f ? 0.0f : metal::exp(-difference / tolerance);
#else
    if (!std::isfinite(receiverDepth) || !std::isfinite(candidateDepth) || receiverDepth <= 0.0f ||
        candidateDepth <= 0.0f) {
        return 0.0f;
    }
    const float tolerance = std::max(receiverDepth * 0.02f, 0.25f);
    const float difference = std::abs(candidateDepth - receiverDepth);
    return difference > tolerance * 3.0f ? 0.0f : std::exp(-difference / tolerance);
#endif
}

// ---- Hi-Z screen-space ray traversal ----
// The trace kernel marches each ray across the min-depth pyramid in mip-zero
// texel space: advance to the current cell's exit, ascend after proving the
// cell empty, and descend toward mip zero when the ray may pass behind a
// surface. Cell stepping cannot skip a texel, so no crossing heuristic is
// needed. These helpers stay scalar and branch-free where possible so the
// CPU mirrors in tests/test_render.mm pin the exact traversal arithmetic.

// Additive R2 low-discrepancy sequence. Consecutive indices cover the unit
// square evenly, so one pixel's rays decorrelate across frames instead of
// clumping the way independent hashes do. The index wraps at 1024 to keep
// the product exactly representable in single precision.
static inline simd_float2 screenSpaceRaySequenceSample(uint32_t sequenceIndex,
                                                       simd_float2 pixelNoise) {
    const float index = (float)(sequenceIndex & 1023u);
#ifdef __METAL_VERSION__
    return metal::fract(pixelNoise + index * metal::float2(0.7548777f, 0.5698403f));
#else
    const simd_float2 shifted = pixelNoise + index * simd_make_float2(0.7548777f, 0.5698403f);
    return simd_make_float2(shifted.x - std::floor(shifted.x), shifted.y - std::floor(shifted.y));
#endif
}

// Cosine-weighted hemisphere direction around the receiver normal. The
// cosine density cancels the Lambert term, so the bounce estimator is the
// plain mean of hit radiances and small bright sources are sampled in
// proportion to their true solid angle.
static inline simd_float3 screenSpaceCosineHemisphereDirection(simd_float2 xi, simd_float3 normal) {
#ifdef __METAL_VERSION__
    const float lengthSquared = metal::dot(normal, normal);
    if (!metal::isfinite(lengthSquared) || lengthSquared < 1.0e-8f) {
        return metal::float3(0.0f, 0.0f, 1.0f);
    }
    const metal::float3 axis = normal * metal::rsqrt(lengthSquared);
    const metal::float3 tangent =
        metal::abs(axis.z) < 0.999f
            ? metal::normalize(metal::cross(metal::float3(0.0f, 0.0f, 1.0f), axis))
            : metal::float3(1.0f, 0.0f, 0.0f);
    const metal::float3 bitangent = metal::normalize(metal::cross(axis, tangent));
    const float phi = 6.2831853f * metal::saturate(xi.x);
    const float radial = metal::sqrt(metal::saturate(xi.y));
    const float height = metal::sqrt(metal::saturate(1.0f - xi.y));
    return metal::normalize(tangent * (radial * metal::cos(phi)) +
                            bitangent * (radial * metal::sin(phi)) + axis * height);
#else
    const float lengthSquared = simd_dot(normal, normal);
    if (!std::isfinite(lengthSquared) || lengthSquared < 1.0e-8f) {
        return simd_make_float3(0.0f, 0.0f, 1.0f);
    }
    const simd_float3 axis = normal / std::sqrt(lengthSquared);
    const simd_float3 tangent =
        std::abs(axis.z) < 0.999f
            ? simd_normalize(simd_cross(simd_make_float3(0.0f, 0.0f, 1.0f), axis))
            : simd_make_float3(1.0f, 0.0f, 0.0f);
    const simd_float3 bitangent = simd_normalize(simd_cross(axis, tangent));
    const float phi = 6.2831853f * std::clamp(xi.x, 0.0f, 1.0f);
    const float radial = std::sqrt(std::clamp(xi.y, 0.0f, 1.0f));
    const float height = std::sqrt(std::clamp(1.0f - xi.y, 0.0f, 1.0f));
    return simd_normalize(tangent * (radial * std::cos(phi)) +
                          bitangent * (radial * std::sin(phi)) + axis * height);
#endif
}

// Texel distance to leave the pyramid cell containing positionTexels.
// directionTexels must be unit length in mip-zero texel space. The result is
// nudged past the boundary so the next lookup lands in the adjacent cell
// instead of re-testing this one forever.
static inline float screenSpaceHiZCellExit(simd_float2 positionTexels, simd_float2 directionTexels,
                                           float cellSizeTexels) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(cellSizeTexels) || cellSizeTexels <= 0.0f) {
        return 0.05f;
    }
    const metal::float2 cell = metal::floor(positionTexels / cellSizeTexels);
    const float boundaryX = (cell.x + (directionTexels.x >= 0.0f ? 1.0f : 0.0f)) * cellSizeTexels;
    const float boundaryY = (cell.y + (directionTexels.y >= 0.0f ? 1.0f : 0.0f)) * cellSizeTexels;
    const float exitX = metal::abs(directionTexels.x) > 1.0e-6f
                            ? (boundaryX - positionTexels.x) / directionTexels.x
                            : 3.4e38f;
    const float exitY = metal::abs(directionTexels.y) > 1.0e-6f
                            ? (boundaryY - positionTexels.y) / directionTexels.y
                            : 3.4e38f;
    return metal::max(metal::min(exitX, exitY), 0.0f) + 0.05f;
#else
    if (!std::isfinite(cellSizeTexels) || cellSizeTexels <= 0.0f) {
        return 0.05f;
    }
    const float cellX = std::floor(positionTexels.x / cellSizeTexels);
    const float cellY = std::floor(positionTexels.y / cellSizeTexels);
    const float boundaryX = (cellX + (directionTexels.x >= 0.0f ? 1.0f : 0.0f)) * cellSizeTexels;
    const float boundaryY = (cellY + (directionTexels.y >= 0.0f ? 1.0f : 0.0f)) * cellSizeTexels;
    const float exitX = std::abs(directionTexels.x) > 1.0e-6f
                            ? (boundaryX - positionTexels.x) / directionTexels.x
                            : 3.4e38f;
    const float exitY = std::abs(directionTexels.y) > 1.0e-6f
                            ? (boundaryY - positionTexels.y) / directionTexels.y
                            : 3.4e38f;
    return std::max(std::min(exitX, exitY), 0.0f) + 0.05f;
#endif
}

// Perspective-correct linear view depth along a projected ray segment.
// Inverse depth interpolates linearly in screen space, so the endpoints'
// reciprocals are mixed and inverted rather than mixing the depths directly.
static inline float screenSpaceHiZRayDepth(float progress, float startLinearDepth,
                                           float endLinearDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(startLinearDepth) || !metal::isfinite(endLinearDepth)) {
        return INDIRECT_SKY_LINEAR_DEPTH;
    }
    const float clampedProgress = metal::saturate(progress);
    const float inverseStart = 1.0f / metal::max(startLinearDepth, 1.0e-4f);
    const float inverseEnd = 1.0f / metal::max(endLinearDepth, 1.0e-4f);
    const float inverseDepth = inverseStart + (inverseEnd - inverseStart) * clampedProgress;
    return 1.0f / metal::max(inverseDepth, 1.0e-6f);
#else
    if (!std::isfinite(startLinearDepth) || !std::isfinite(endLinearDepth)) {
        return INDIRECT_SKY_LINEAR_DEPTH;
    }
    const float clampedProgress = std::clamp(progress, 0.0f, 1.0f);
    const float inverseStart = 1.0f / std::max(startLinearDepth, 1.0e-4f);
    const float inverseEnd = 1.0f / std::max(endLinearDepth, 1.0e-4f);
    const float inverseDepth = inverseStart + (inverseEnd - inverseStart) * clampedProgress;
    return 1.0f / std::max(inverseDepth, 1.0e-6f);
#endif
}

// A cell is provably empty when the deepest point the ray reaches inside it
// stays in front of the cell's closest surface. Non-finite input classifies
// as a potential hit so the mip-zero exact test makes the final call.
static inline bool screenSpaceHiZAdvances(float rayEntryDepth, float rayExitDepth,
                                          float cellMinDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(rayEntryDepth) || !metal::isfinite(rayExitDepth) ||
        !metal::isfinite(cellMinDepth)) {
        return false;
    }
    return metal::max(rayEntryDepth, rayExitDepth) < cellMinDepth;
#else
    if (!std::isfinite(rayEntryDepth) || !std::isfinite(rayExitDepth) ||
        !std::isfinite(cellMinDepth)) {
        return false;
    }
    return std::max(rayEntryDepth, rayExitDepth) < cellMinDepth;
#endif
}

// Exact mip-zero receiver test: the ray is at or behind the visible surface
// but within its assumed thickness. Sky texels and non-finite values never
// hit, so rays leaving the depth buffer contribute nothing.
static inline bool screenSpaceHiZSurfaceHit(float rayDepth, float surfaceDepth, float thickness) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(rayDepth) || !metal::isfinite(surfaceDepth) ||
        !metal::isfinite(thickness) || surfaceDepth > INDIRECT_SKY_LINEAR_DEPTH) {
        return false;
    }
    const float tolerance = metal::max(thickness, 0.05f);
#else
    if (!std::isfinite(rayDepth) || !std::isfinite(surfaceDepth) || !std::isfinite(thickness) ||
        surfaceDepth > INDIRECT_SKY_LINEAR_DEPTH) {
        return false;
    }
    const float tolerance = std::max(thickness, 0.05f);
#endif
    return rayDepth >= surfaceDepth - 1.0e-3f && rayDepth <= surfaceDepth + tolerance;
}

// Near-field occlusion weight for GTAO-style ambient darkening. Hits beyond
// the AO radius still bounce light but no longer occlude, keeping the
// ambient correction local while the bounce reach extends further.
static inline float screenSpaceOcclusionFalloff(float hitDistance, float radius) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(hitDistance) || !metal::isfinite(radius) || radius <= 0.0f) {
        return 0.0f;
    }
    const float remaining = metal::saturate(1.0f - hitDistance / radius);
#else
    if (!std::isfinite(hitDistance) || !std::isfinite(radius) || radius <= 0.0f) {
        return 0.0f;
    }
    const float remaining = std::clamp(1.0f - hitDistance / radius, 0.0f, 1.0f);
#endif
    return remaining * remaining;
}

// One-bounce source weight: a back-facing source contributes occlusion only,
// and radiance fades over the last quarter of the trace range so the bounce
// reach limit cannot pop as the camera moves.
static inline float screenSpaceBounceSourceWeight(float sourceCosine, float hitDistance,
                                                  float maxDistance) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(sourceCosine) || !metal::isfinite(hitDistance) ||
        !metal::isfinite(maxDistance) || maxDistance <= 0.0f) {
        return 0.0f;
    }
    const float facing = metal::saturate(sourceCosine);
    const float fadeStart = 0.75f * maxDistance;
    const float fade =
        1.0f - metal::smoothstep(fadeStart, maxDistance, metal::max(hitDistance, 0.0f));
    return facing * fade;
#else
    if (!std::isfinite(sourceCosine) || !std::isfinite(hitDistance) ||
        !std::isfinite(maxDistance) || maxDistance <= 0.0f) {
        return 0.0f;
    }
    const float facing = std::clamp(sourceCosine, 0.0f, 1.0f);
    const float fadeStart = 0.75f * maxDistance;
    const float fadeSpan = std::max(maxDistance - fadeStart, 1.0e-4f);
    const float fadeAmount =
        std::clamp((std::max(hitDistance, 0.0f) - fadeStart) / fadeSpan, 0.0f, 1.0f);
    const float fade = 1.0f - fadeAmount * fadeAmount * (3.0f - 2.0f * fadeAmount);
    return facing * fade;
#endif
}

// ---- Temporal accumulation and spatial denoising ----

// Age-driven history blend: a disoccluded pixel restarts from the current
// frame and converges toward the cap in about nine frames. This replaces a
// fixed history weight that ghosted for seconds after any change.
static inline float screenSpaceTemporalBlendWeight(float age, float maximumWeight) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(age) || !metal::isfinite(maximumWeight) || age <= 0.0f) {
        return 0.0f;
    }
    return metal::min(age / (age + 1.0f), metal::saturate(maximumWeight));
#else
    if (!std::isfinite(age) || !std::isfinite(maximumWeight) || age <= 0.0f) {
        return 0.0f;
    }
    return std::min(age / (age + 1.0f), std::clamp(maximumWeight, 0.0f, 1.0f));
#endif
}

static inline float screenSpaceLuminanceVariance(float firstMoment, float secondMoment) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(firstMoment) || !metal::isfinite(secondMoment)) {
        return 0.0f;
    }
    return metal::max(secondMoment - firstMoment * firstMoment, 0.0f);
#else
    if (!std::isfinite(firstMoment) || !std::isfinite(secondMoment)) {
        return 0.0f;
    }
    return std::max(secondMoment - firstMoment * firstMoment, 0.0f);
#endif
}

// Young pixels have too few samples for a trustworthy temporal variance, so
// they fall back to the larger of the spatial and temporal estimates. That
// floor opens the a-trous filter wide on disocclusion, filling fresh regions
// with a smooth spatial estimate instead of black or speckle.
static inline float screenSpaceVarianceForAge(float temporalVariance, float spatialVariance,
                                              float age, float disocclusionAgeThreshold) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(temporalVariance) || !metal::isfinite(spatialVariance) ||
        !metal::isfinite(age)) {
        return 0.0f;
    }
    const float temporal = metal::max(temporalVariance, 0.0f);
    const float spatial = metal::max(spatialVariance, 0.0f);
    return age < disocclusionAgeThreshold ? metal::max(temporal, spatial) : temporal;
#else
    if (!std::isfinite(temporalVariance) || !std::isfinite(spatialVariance) ||
        !std::isfinite(age)) {
        return 0.0f;
    }
    const float temporal = std::max(temporalVariance, 0.0f);
    const float spatial = std::max(spatialVariance, 0.0f);
    return age < disocclusionAgeThreshold ? std::max(temporal, spatial) : temporal;
#endif
}

// Variance-scaled clamp half-range. A converged neighborhood has near-zero
// deviation, so stale bright history collapses to the floor immediately,
// while a genuinely sparse bright source keeps a wide clamp because its
// accumulated variance stays high. This replaces the special case that kept
// reprojected color unclamped over an all-zero neighborhood.
static inline float screenSpaceVarianceClampHalfRange(float standardDeviation, float gamma,
                                                      float minimumRange) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(standardDeviation) || !metal::isfinite(gamma) ||
        !metal::isfinite(minimumRange)) {
        return metal::max(minimumRange, 0.0f);
    }
    return metal::max(metal::max(standardDeviation, 0.0f) * metal::max(gamma, 0.0f),
                      metal::max(minimumRange, 0.0f));
#else
    if (!std::isfinite(standardDeviation) || !std::isfinite(gamma) ||
        !std::isfinite(minimumRange)) {
        return std::max(minimumRange, 0.0f);
    }
    return std::max(std::max(standardDeviation, 0.0f) * std::max(gamma, 0.0f),
                    std::max(minimumRange, 0.0f));
#endif
}

// Hue-preserving firefly clamp applied to the raw trace sample before any
// statistics or accumulation. A single sunlit texel found by one ray cannot
// seed the history with more luminance than the cap.
static inline float screenSpaceFireflyClampScale(float luminance, float maximumLuminance) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(luminance) || !metal::isfinite(maximumLuminance) ||
        maximumLuminance <= 0.0f) {
        return 0.0f;
    }
    return luminance <= maximumLuminance ? 1.0f : maximumLuminance / luminance;
#else
    if (!std::isfinite(luminance) || !std::isfinite(maximumLuminance) || maximumLuminance <= 0.0f) {
        return 0.0f;
    }
    return luminance <= maximumLuminance ? 1.0f : maximumLuminance / luminance;
#endif
}

// Edge-stopping weight for the a-trous wavelet passes. Depth and normal
// terms pin the filter to one voxel face; the luminance term widens with the
// pixel's variance so noisy young regions blur and converged regions stay
// sharp.
static inline float screenSpaceAtrousEdgeWeight(float depthDelta, float depthTolerance,
                                                float normalDot, float luminanceDelta,
                                                float luminanceSigma) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(depthDelta) || !metal::isfinite(normalDot) ||
        !metal::isfinite(luminanceDelta)) {
        return 0.0f;
    }
    const float depthScale = metal::max(depthTolerance, 1.0e-4f);
    if (metal::abs(depthDelta) > depthScale * 3.0f) {
        return 0.0f;
    }
    const float depthWeight = metal::exp(-metal::abs(depthDelta) / depthScale);
    const float alignment = metal::saturate(normalDot);
    const float alignmentSquared = alignment * alignment;
    const float normalWeight =
        alignmentSquared * alignmentSquared * alignmentSquared * alignmentSquared;
    const float luminanceScale = metal::max(luminanceSigma, 1.0e-4f);
    const float luminanceWeight = metal::exp(-metal::abs(luminanceDelta) / luminanceScale);
    return depthWeight * normalWeight * luminanceWeight;
#else
    if (!std::isfinite(depthDelta) || !std::isfinite(normalDot) || !std::isfinite(luminanceDelta)) {
        return 0.0f;
    }
    const float depthScale = std::max(depthTolerance, 1.0e-4f);
    if (std::abs(depthDelta) > depthScale * 3.0f) {
        return 0.0f;
    }
    const float depthWeight = std::exp(-std::abs(depthDelta) / depthScale);
    const float alignment = std::clamp(normalDot, 0.0f, 1.0f);
    const float alignmentSquared = alignment * alignment;
    const float normalWeight =
        alignmentSquared * alignmentSquared * alignmentSquared * alignmentSquared;
    const float luminanceScale = std::max(luminanceSigma, 1.0e-4f);
    const float luminanceWeight = std::exp(-std::abs(luminanceDelta) / luminanceScale);
    return depthWeight * normalWeight * luminanceWeight;
#endif
}

// Unified air-medium froxel injection, front-to-back integration, and
// temporal reprojection parameters. Air extinction is gated above the water
// surface; underwater absorption remains in the dedicated water path.
struct FroxelUniforms {
    simd_float4x4 invViewProjection;
    simd_float4x4 previousViewProjection;
    simd_float4x4 viewProjection;
    simd_float3 cameraPosition;
    simd_float3 lightDirection;
    simd_float3 lightRadiance;
    simd_float3 solarDirection;
    simd_uint4 volumeDimensions;
    simd_float4 depthParams;
    simd_float4 mediumParams;
    simd_float4 weatherParams;
    simd_float4 renderParams;
    WeatherMapUniforms weatherMap;
};

// One deterministic lightning event expanded procedurally by the vertex
// shader. Event identity is split into two 32-bit words to remain exact.
struct LightningUniforms {
    simd_float4x4 viewProjection;
    simd_float3 cameraPosition;
    simd_float3 strikePosition;
    simd_float4 colorAndIntensity;
    simd_uint4 eventAndShape;
};

static inline float beerLambertTransmittance(float extinction, float distance) {
#ifdef __METAL_VERSION__
    return metal::exp(-metal::max(extinction, 0.0f) * metal::max(distance, 0.0f));
#else
    return std::exp(-std::max(extinction, 0.0f) * std::max(distance, 0.0f));
#endif
}

static inline float froxelSliceDepth(unsigned int slice, unsigned int sliceCount, float nearDepth,
                                     float farDepth) {
    const float amount = float(slice) / float(sliceCount);
#ifdef __METAL_VERSION__
    return nearDepth * metal::pow(farDepth / nearDepth, amount);
#else
    return nearDepth * std::pow(farDepth / nearDepth, amount);
#endif
}

// The physical sky is already the integral of the atmosphere to infinity.
// Only opaque geometry or a resolved cloud hit supplies a finite receiver for
// the separate air-medium pass. Treating an empty depth pixel as far geometry
// would attenuate the sky a second time and turn a bright day into a dark fog
// dome while leaving nearby terrain lit.
static inline bool froxelHasFiniteReceiver(float opaqueDeviceDepth, float cloudHitDistance) {
    return opaqueDeviceDepth < 0.99999f || (cloudHitDistance > 0.0f && cloudHitDistance < 65504.0f);
}

// The froxel history stores linear view depth rather than compressed device
// depth. Device depth turns a tiny motion along a grazing cave floor into a
// much larger apparent discontinuity, exposing the fixed froxel grid instead
// of allowing the temporal filter to converge. This tolerance stays below a
// separate nearby voxel face while growing with legitimate receiver distance.
static inline float froxelTemporalLinearDepthTolerance(float linearDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(linearDepth) || linearDepth <= 0.0f) {
        return 0.0f;
    }
    return metal::max(0.05f, linearDepth * 0.008f);
#else
    if (!std::isfinite(linearDepth) || linearDepth <= 0.0f) {
        return 0.0f;
    }
    return std::max(0.05f, linearDepth * 0.008f);
#endif
}

// Half-resolution froxel integration must not blend a lit opening over an
// adjacent cave wall or ceiling. Use the same linear view-depth authority as
// temporal reprojection, with a hard rejection beyond a bounded neighborhood.
static inline float froxelBilateralLinearDepthWeight(float receiverDepth, float candidateDepth) {
#ifdef __METAL_VERSION__
    if (!metal::isfinite(receiverDepth) || !metal::isfinite(candidateDepth) ||
        receiverDepth <= 0.0f || candidateDepth <= 0.0f) {
        return 0.0f;
    }
    const float tolerance = froxelTemporalLinearDepthTolerance(receiverDepth);
    const float difference = metal::abs(candidateDepth - receiverDepth);
    return difference > tolerance * 3.0f ? 0.0f : metal::exp(-difference / tolerance);
#else
    if (!std::isfinite(receiverDepth) || !std::isfinite(candidateDepth) || receiverDepth <= 0.0f ||
        candidateDepth <= 0.0f) {
        return 0.0f;
    }
    const float tolerance = froxelTemporalLinearDepthTolerance(receiverDepth);
    const float difference = std::abs(candidateDepth - receiverDepth);
    return difference > tolerance * 3.0f ? 0.0f : std::exp(-difference / tolerance);
#endif
}

// A deterministic R2 sequence moves the one representative sample within a
// fixed froxel cell and logarithmic slice. Temporal reprojection then averages
// those unbiased positions instead of preserving camera-aligned shadow bands.
// The sequence contains no external state, remains reproducible in captures,
// and leaves the 160 by 104 by 64 froxel contract unchanged.
static inline float froxelLowDiscrepancySample(unsigned int frameIndex, unsigned int dimension) {
    const float multiplier =
        dimension == 0u ? 0.754877666f : (dimension == 1u ? 0.569840296f : 0.438578026f);
#ifdef __METAL_VERSION__
    return metal::fract((float(frameIndex) + 0.5f) * multiplier);
#else
    const float value = (static_cast<float>(frameIndex) + 0.5f) * multiplier;
    return value - std::floor(value);
#endif
}

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
    // enters the frame, so bright samples get up-weighted:
    // w = 1 + gain * saturate((logLum - knee) / range), and facing the sun
    // actually stops the scene down.
    float highlightGain;
    float highlightKnee;  // log2 luminance where up-weighting starts
    float highlightRange; // log2 range over which the weight ramps in
};

// Bound at buffer(0) in the final composite (post.metal): the one pass that
// converts the linear HDR scene to the display: exposure, bloom add,
// filmic tonemap, vibrance grade, optional CAS sharpen, dither. It always
// runs, so the frame is tonemapped even with bloom off.
struct PostUniforms {
    simd_float2 resolution;            // drawable size in pixels
    float exposure;                    // linear pre-tonemap multiplier
    float bloomIntensity;              // 0 = bloom texture is the black fallback
    float vibrance;                    // 0..2 saturation-aware boost; 1 = stock look
    float sharpening;                  // 0..1 CAS strength; 0 = skip
    uint32_t frameIndex;               // deterministic dither phase
    float flareStrength;               // 0 = flare off (setting off / sun behind camera)
    simd_float2 sunScreenUV;           // sun position in composite UV space
    uint32_t flareCloudOpacityTexture; // 1 when probe texture alpha stores cloud opacity
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

static_assert(sizeof(FoliageWindUniforms) == 16);
static_assert(offsetof(FoliageWindUniforms, direction) == 0);
static_assert(offsetof(FoliageWindUniforms, speedBlocksPerSecond) == 8);
static_assert(offsetof(FoliageWindUniforms, strength) == 12);

static_assert(sizeof(Uniforms) == 320);
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
static_assert(offsetof(Uniforms, foliageWind) == 288);
static_assert(offsetof(Uniforms, time) == 304);
static_assert(offsetof(Uniforms, wetness) == 308);

static_assert(sizeof(WaterUniforms) == 288);
static_assert(offsetof(WaterUniforms, skyExposure) == 248);
static_assert(offsetof(WaterUniforms, waterSurfaceY) == 252);
static_assert(offsetof(WaterUniforms, cameraRelativeViewProjection) == 64);
static_assert(offsetof(WaterUniforms, zenithColor) == 128);
static_assert(offsetof(WaterUniforms, resolution) == 224);
static_assert(offsetof(WaterUniforms, fogDensity) == 232);
static_assert(offsetof(WaterUniforms, time) == 236);
static_assert(offsetof(WaterUniforms, cameraUnderwater) == 240);
static_assert(offsetof(WaterUniforms, ssrStrength) == 244);
static_assert(offsetof(WaterUniforms, solarDirection) == 256);
static_assert(offsetof(WaterUniforms, physicalSkyBlend) == 272);
static_assert(offsetof(WaterUniforms, directSpecularFactor) == 276);

static_assert(sizeof(ShadowPassUniforms) == 112);
static_assert(offsetof(ShadowPassUniforms, projectionOrigin) == 64);
static_assert(offsetof(ShadowPassUniforms, foliageWind) == 80);
static_assert(offsetof(ShadowPassUniforms, time) == 96);

static_assert(sizeof(ShadowCascadeUniforms) == 112);
static_assert(offsetof(ShadowCascadeUniforms, projectionOrigin) == 64);
static_assert(offsetof(ShadowCascadeUniforms, depthRange) == 80);
static_assert(offsetof(ShadowCascadeUniforms, samplingParams) == 96);
static_assert(sizeof(ShadowUniforms) == 592);
static_assert(offsetof(ShadowUniforms, cameraPositionAndStrength) == 560);
static_assert(offsetof(ShadowUniforms, cameraForwardAndPadding) == 576);

static_assert(sizeof(SkyUniforms) == 176);
static_assert(offsetof(SkyUniforms, moonDirection) == 64);
static_assert(offsetof(SkyUniforms, moonColor) == 96);
static_assert(offsetof(SkyUniforms, zenithColor) == 112);
static_assert(offsetof(SkyUniforms, visibilityAndPhase) == 144);
static_assert(offsetof(SkyUniforms, tanHalfFov) == 160);

static_assert(sizeof(WeatherMapUniforms) == 32);
static_assert(offsetof(WeatherMapUniforms, gridSize) == 16);
static_assert(offsetof(WeatherMapUniforms, motionWrapBlocks) == 24);
static_assert(sizeof(AtmosphereUniforms) == 160);
static_assert(offsetof(AtmosphereUniforms, weatherOptics) == 128);
static_assert(sizeof(CloudRenderUniforms) == 352);
static_assert(offsetof(CloudRenderUniforms, cameraForward) == 144);
static_assert(offsetof(CloudRenderUniforms, skyIrradiance) == 192);
static_assert(offsetof(CloudRenderUniforms, weatherMap) == 288);
static_assert(offsetof(CloudRenderUniforms, previousWeatherMap) == 320);
static_assert(sizeof(CloudShadowUniforms) == 80);
static_assert(offsetof(CloudShadowUniforms, weatherMap) == 48);

static_assert(sizeof(GPUParticle) == 48);
static_assert(offsetof(GPUParticle, velocity) == 16);
static_assert(offsetof(GPUParticle, lifetime) == 32);

static_assert(sizeof(EntityVertex) == 48);
static_assert(offsetof(EntityVertex, normal) == 16);
static_assert(offsetof(EntityVertex, color) == 32);
static_assert(sizeof(EntityModel) == 80);
static_assert(offsetof(EntityModel, lighting) == 64);

static_assert(sizeof(ParticleUniforms) == 160);
static_assert(offsetof(ParticleUniforms, cameraPosition) == 128);
static_assert(offsetof(ParticleUniforms, atmosphericExtinction) == 144);

static_assert(sizeof(BloomUniforms) == 32);
static_assert(offsetof(BloomUniforms, threshold) == 16);

static_assert(sizeof(PostUniforms) == 48);
static_assert(offsetof(PostUniforms, exposure) == 8);
static_assert(offsetof(PostUniforms, frameIndex) == 24);
static_assert(offsetof(PostUniforms, flareStrength) == 28);
static_assert(offsetof(PostUniforms, sunScreenUV) == 32);
static_assert(offsetof(PostUniforms, flareCloudOpacityTexture) == 40);
static_assert(sizeof(FlareState) == 4);

static_assert(sizeof(IndirectLightingUniforms) == 336);
static_assert(offsetof(IndirectLightingUniforms, traceParams) == 272);
static_assert(offsetof(IndirectLightingUniforms, filterParams) == 304);
static_assert(offsetof(IndirectLightingUniforms, ambientAndFrame) == 320);

static_assert(sizeof(FroxelUniforms) == 368);
static_assert(offsetof(FroxelUniforms, cameraPosition) == 192);
static_assert(offsetof(FroxelUniforms, volumeDimensions) == 256);
static_assert(offsetof(FroxelUniforms, renderParams) == 320);
static_assert(offsetof(FroxelUniforms, weatherMap) == 336);
static_assert(sizeof(LightningUniforms) == 128);
static_assert(offsetof(LightningUniforms, eventAndShape) == 112);

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
