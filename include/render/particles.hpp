#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "render/frame_ring.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"
#include "world/weather.hpp"

#include <cstdint>

// Forward declarations
class World;

// ---------------------------------------------------------------------------
// ParticleType, Weather particle kind
// ---------------------------------------------------------------------------
enum class ParticleType : uint8_t {
    RAIN = 0,
    SNOW = 1,
};

// ---------------------------------------------------------------------------
// Particle, Single weather particle (CPU simulation state)
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
// ParticleSystem, CPU-simulated, GPU-rendered weather particles.
//
// Pool-based with fixed 4096 particle capacity. The canonical weather sample
// supplies precipitation kind, intensity, and wind in physical world units.
//
// Rain: fall ~10 blocks/s, slight wind drift, lifetime 5s (spawned just
// above the camera so drops sweep past eye level before dying)
// Snow: fall ~3 blocks/s, sinusoidal horizontal drift, lifetime 10s
// ---------------------------------------------------------------------------
class ParticleSystem {
public:
    static constexpr size_t MAX_PARTICLES = 4096;

    ParticleSystem(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~ParticleSystem();

    // Update particle physics (call each game tick). When precipitation stops,
    // no new particles spawn and live particles finish their lifetimes.
    void tick(float dt, const World& world, const Vec3& playerPosition,
              const WeatherSample& weather);

    // Render active particles as billboards after the air-medium composite.
    // Instance data + uniforms sub-allocate from the caller's frame ring so
    // the per-frame rewrite never races frames the GPU still reads.
    void render(id<MTLRenderCommandEncoder> encoder, FrameRing& frameRing, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const Vec3& cameraPosition,
                const WeatherSample& weather, float baseExtinction);

private:
    // ---- CPU particle pool ----
    Particle particles_[MAX_PARTICLES];

    // ---- Metal resources ----
    id<MTLDevice> _device;
    id<MTLRenderPipelineState> _pipelineState;
    id<MTLDepthStencilState> _depthState;

    // ---- Spawn helpers (false = column not sky-exposed; nothing spawned) ----
    bool spawnRainParticle(Particle& p, const World& world, const Vec3& playerPos,
                           float spawnRadius, const WeatherSample& weather);
    bool spawnSnowParticle(Particle& p, const World& world, const Vec3& playerPos,
                           float spawnRadius, const WeatherSample& weather);
};

// Air precipitation is not rendered from within the underwater medium. This
// prevents droplets in front of the water surface from being composited over
// the underwater overlay while still allowing them to age normally.
constexpr bool weatherParticlesVisible(bool cameraUnderwater) {
    return !cameraUnderwater;
}
