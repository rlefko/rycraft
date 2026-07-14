#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "render/frame_ring.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"

#include <cstdint>

// Forward declarations
class World;

// ---------------------------------------------------------------------------
// ParticleType — Weather particle kind
// ---------------------------------------------------------------------------
enum class ParticleType : uint8_t {
    RAIN = 0,
    SNOW = 1,
};

// ---------------------------------------------------------------------------
// Particle — Single weather particle (CPU simulation state)
//
// Layout: position(12) + velocity(12) + lifetime(4) + maxLifetime(4) +
//         type(1) + active(1) + padding(2) = 36 bytes
// Padded to 40 bytes for 16-byte alignment in arrays.
// ---------------------------------------------------------------------------
struct alignas(16) Particle {
    float position[3];
    float velocity[3];
    float lifetime;
    float maxLifetime;
    ParticleType type;
    bool active;
    uint8_t _pad[2];
};

// GPUParticle and ParticleUniforms live in render/shader_types.hpp, shared
// with particles.metal so the two sides can never disagree on layout.

// ---------------------------------------------------------------------------
// ParticleSystem — CPU-simulated, GPU-rendered weather particles.
//
// Pool-based with fixed 4096 particle capacity. Biome-aware: rain in most
// biomes, snow in cold biomes (IceSpikes, Taiga).
//
// Rain: fall ~10 blocks/s, slight wind drift, lifetime ~2s
// Snow: fall ~3 blocks/s, sinusoidal horizontal drift, lifetime ~5s
// ---------------------------------------------------------------------------
class ParticleSystem {
public:
    static constexpr size_t MAX_PARTICLES = 4096;

    ParticleSystem(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~ParticleSystem();

    // Update particle physics (call each game tick).
    void tick(float dt, const World& world, const Vec3& playerPosition);

    // Render active particles as billboards (call during main render pass).
    // Instance data + uniforms sub-allocate from the caller's frame ring so
    // the per-frame rewrite never races frames the GPU still reads.
    void render(id<MTLRenderCommandEncoder> encoder, FrameRing& frameRing, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const Vec3& cameraPosition);

private:
    // ---- CPU particle pool ----
    Particle particles_[MAX_PARTICLES];

    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // ---- Spawn helpers ----
    void spawnRainParticle(Particle& p, const Vec3& playerPos, float spawnRadius);
    void spawnSnowParticle(Particle& p, const Vec3& playerPos, float spawnRadius);

    // Check if biome at position should produce snow
    bool isSnowBiome(const World& world, int x, int z) const;
};
