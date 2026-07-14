#pragma once

// Shared GPU data layouts, included by BOTH the C++ engine and the .metal
// shaders (the shader build passes -I include/). simd types have identical
// size and alignment in each language, so one definition serves both sides;
// the static_asserts below make any drift a compile error instead of a
// corrupted frame.
#include <simd/simd.h>

#ifdef __METAL_VERSION__
// Interleaved gradient noise — the engine's one deterministic per-pixel
// dither/rotation source (convention: seeded randomness only, no temporal
// noise). One definition here because five passes sample it and the magic
// constants must never drift apart.
static inline float interleavedGradientNoise(metal::float2 px) {
    return metal::fract(52.9829189f *
                        metal::fract(metal::dot(px, metal::float2(0.06711056f, 0.00583715f))));
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
constant WaterWave WATER_WAVES[3] = {
    {metal::float2(0.80f, 0.60f), 0.52f, 0.060f, 1.1f},
    {metal::float2(-0.50f, 0.87f), 0.80f, 0.035f, 1.5f},
    {metal::float2(0.20f, -0.98f), 1.05f, 0.020f, 2.1f},
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
    // Faint fine ripple, shading only (not displaced), for sparkle
    g += 0.008f *
         metal::float2(metal::cos(p.x * 1.9f + t * 2.3f), metal::cos(p.y * 2.2f - t * 2.0f));
    const float slope = 1.5f;
    return metal::normalize(metal::float3(-g.x * slope, 1.0f, -g.y * slope));
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
struct ChunkOrigin {
    simd_float4 origin;
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
// from the resolved opaque scene, so it needs the full camera inverse and
// the sky palette that the fresnel reflection samples procedurally.
struct WaterUniforms {
    simd_float4x4 invViewProjection; // clip → world, for depth reconstruction
    simd_float4x4 viewProjection;    // world → clip, for SSR ray projection
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
    float adaptationRate;  // 0..1 EMA weight for this frame (eye adaptation speed)
    float minLogLum;       // clamp floor for scene log-luminance
    float maxLogLum;       // clamp ceiling
    simd_uint2 sampleGrid; // reduction sample count across the frame (e.g. 16×16)
    // Exposure clamp: minExposure well above 0 keeps bright outdoor scenes
    // from being crushed dim; maxExposure lifts caves/night without blowing up.
    float minExposure;
    float maxExposure;
};

// Bound at buffer(0) in the final composite (post.metal): the one pass that
// converts the linear HDR scene to the display — exposure, bloom add,
// Uchimura tonemap, vibrance grade, optional CAS sharpen, dither. It always
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
static_assert(offsetof(Uniforms, sunDirection) == 192);
static_assert(offsetof(Uniforms, fogColor) == 240);
static_assert(offsetof(Uniforms, fogDensity) == 256);
static_assert(offsetof(Uniforms, cameraPosition) == 272);
static_assert(offsetof(Uniforms, time) == 288);
static_assert(offsetof(Uniforms, swayStrength) == 292);
static_assert(offsetof(Uniforms, wetness) == 296);

static_assert(sizeof(WaterUniforms) == 256);
static_assert(offsetof(WaterUniforms, viewProjection) == 64);
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

static_assert(sizeof(ExposureParams) == 32);
static_assert(offsetof(ExposureParams, adaptationRate) == 4);
static_assert(offsetof(ExposureParams, sampleGrid) == 16);
static_assert(offsetof(ExposureParams, minExposure) == 24);
static_assert(offsetof(ExposureParams, maxExposure) == 28);
#endif
