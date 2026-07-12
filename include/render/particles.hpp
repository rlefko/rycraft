#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
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

// ---------------------------------------------------------------------------
// GPUParticle — Per-particle data uploaded to GPU each frame
//
// Layout: position(12) + velocity(12) + lifetime(4) + type(4) = 32 bytes
// Aligned to 16 bytes for Metal buffer stride.
// ---------------------------------------------------------------------------
struct alignas(16) GPUParticle {
    float position[3];
    float _pad0;
    float velocity[3];
    float _pad1;
    float lifetime;
    float type;
};

// ---------------------------------------------------------------------------
// ParticleUniforms — GPU uniform buffer for particle vertex shader
// ---------------------------------------------------------------------------
struct alignas(16) ParticleUniforms {
    float viewMatrix[16];
    float projectionMatrix[16];
    float cameraPosition[3];
    float _pad0;
};

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
    void render(id<MTLRenderCommandEncoder> encoder,
                const Mat4& viewMatrix,
                const Mat4& projectionMatrix,
                const Vec3& cameraPosition);

private:
    // ---- CPU particle pool ----
    Particle particles_[MAX_PARTICLES];

    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // GPU particle buffer (uploaded every frame)
    id<MTLBuffer> _particleBuffer;

    // Uniform buffer (view/projection/camera)
    id<MTLBuffer> _uniformsBuffer;

    // ---- Spawn helpers ----
    void spawnRainParticle(Particle& p, const Vec3& playerPos, float spawnRadius);
    void spawnSnowParticle(Particle& p, const Vec3& playerPos, float spawnRadius);

    // Check if biome at position should produce snow
    bool isSnowBiome(const World& world, int x, int z) const;
};
