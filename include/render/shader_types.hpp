#pragma once

// Shared GPU data layouts, included by BOTH the C++ engine and the .metal
// shaders (the shader build passes -I include/). simd types have identical
// size and alignment in each language, so one definition serves both sides;
// the static_asserts below make any drift a compile error instead of a
// corrupted frame.
#include <simd/simd.h>

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
};

// Per-chunk world offset, bound at buffer(2) via setVertexBytes. Vertices are
// chunk-local so their fp16 coordinates stay exact; this restores world space.
struct ChunkOrigin {
    simd_float4 origin;
};

// Bound at buffer(3) in the water shaders (vertex: wave time; fragment:
// refraction/reflection/caustics). The water pass composites its own pixels
// from the resolved opaque scene, so it needs the full camera inverse and
// the sky palette that the fresnel reflection samples procedurally.
struct WaterUniforms {
    simd_float4x4 invViewProjection; // clip → world, for depth reconstruction
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
};

// Sky gradient + sun disc, bound at buffer(1) in sky.metal.
struct SkyUniforms {
    simd_float3 zenithColor;
    simd_float3 horizonColor;
    simd_float3 sunDirection;
    simd_float3 sunColor;
    float sunIntensity; // 0 at night, 1 at noon
};

// Procedural cloud layer, bound at buffer(0) in clouds.metal. The fragment
// shader ray-casts from the camera through each pixel onto the horizontal
// cloud plane, so the camera basis and projection shape ride along.
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

// Bound at buffer(0) in the final composite (post.metal): the one pass that
// converts the linear HDR scene to the display — exposure, bloom add,
// Uchimura tonemap, vibrance grade, optional CAS sharpen, dither. It always
// runs, so the frame is tonemapped even with bloom off.
struct PostUniforms {
    simd_float2 resolution; // drawable size in pixels
    float exposure;         // linear pre-tonemap multiplier
    float bloomIntensity;   // 0 = bloom texture is the black fallback
    float vibrance;         // 0..2 saturation-aware boost; 1 = stock look
    float sharpening;       // 0..1 CAS strength; 0 = skip
    uint32_t frameIndex;    // deterministic dither phase
};

#ifndef __METAL_VERSION__
#include <cstddef>

static_assert(sizeof(Uniforms) == 288);
static_assert(offsetof(Uniforms, sunDirection) == 192);
static_assert(offsetof(Uniforms, fogColor) == 240);
static_assert(offsetof(Uniforms, fogDensity) == 256);
static_assert(offsetof(Uniforms, cameraPosition) == 272);

static_assert(sizeof(WaterUniforms) == 192);
static_assert(offsetof(WaterUniforms, zenithColor) == 64);
static_assert(offsetof(WaterUniforms, resolution) == 160);
static_assert(offsetof(WaterUniforms, fogDensity) == 168);
static_assert(offsetof(WaterUniforms, time) == 172);
static_assert(offsetof(WaterUniforms, cameraUnderwater) == 176);

static_assert(sizeof(SkyUniforms) == 80);
static_assert(offsetof(SkyUniforms, sunIntensity) == 64);

static_assert(sizeof(CloudUniforms) == 112);
static_assert(offsetof(CloudUniforms, sunDirection) == 64);
static_assert(offsetof(CloudUniforms, tanHalfFov) == 80);
static_assert(offsetof(CloudUniforms, cloudThreshold) == 100);

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

static_assert(sizeof(PostUniforms) == 32);
static_assert(offsetof(PostUniforms, exposure) == 8);
static_assert(offsetof(PostUniforms, frameIndex) == 24);
#endif
