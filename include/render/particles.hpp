#pragma once

#import <Metal/Metal.h>

#include "common/math.hpp"
#include "common/random.hpp"
#include "render/frame_ring.hpp"
#include "render/shader_types.hpp"
#include "world/chunk.hpp"
#include "world/physical_scale.hpp"
#include "world/weather.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

// Forward declarations
class World;

// ---------------------------------------------------------------------------
// ParticleType, Weather particle kind
// ---------------------------------------------------------------------------
enum class ParticleType : uint8_t {
    RAIN = 0,
    SNOW = 1,
};

// Weather follows the expanded v4 world bounds and the world's 64-bit
// horizontal coordinate contract. These helpers stay independent of Metal so
// boundary regressions can run in ordinary CI.
constexpr bool weatherParticleBelowWorld(float y) {
    return y < static_cast<float>(WORLD_MIN_Y);
}

inline int64_t weatherBlockCoordinate(float coordinate) {
    return static_cast<int64_t>(std::floor(coordinate));
}

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

inline constexpr size_t WEATHER_PARTICLE_CAPACITY = 4096;

constexpr uint64_t weatherParticleRandomSeed(uint64_t worldSeed) noexcept {
    return hash64(worldSeed ^ 0x5259435250524350ULL);
}

// CPU simulation state is independently bindable so world teardown can clear
// precipitation without waiting for Metal work. A fresh binding of the same
// seed restarts the identical serial effect stream, while a separate seed
// receives a distinct stream.
class WeatherParticleSessionState {
public:
    bool beginWorld(uint64_t instanceId, uint64_t seed) noexcept;
    void endWorld() noexcept;
    void clear() noexcept;

    std::span<Particle> particles() noexcept { return particles_; }
    std::span<const Particle> particles() const noexcept { return particles_; }
    float nextRandomFloat() noexcept { return random_.nextFloat(); }
    size_t activeCount() const noexcept;
    bool bound() const noexcept { return worldInstanceId_.has_value(); }

private:
    std::array<Particle, WEATHER_PARTICLE_CAPACITY> particles_{};
    SeededRng random_{weatherParticleRandomSeed(0)};
    std::optional<uint64_t> worldInstanceId_;
    uint64_t worldSeed_ = 0;
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
    static constexpr size_t MAX_PARTICLES = WEATHER_PARTICLE_CAPACITY;

    ParticleSystem(id<MTLDevice> device, id<MTLLibrary> shaderLibrary);
    ~ParticleSystem();

    void beginWorld(uint64_t instanceId, uint64_t seed);
    void endWorld();

    // Update particle physics (call each game tick). When precipitation stops,
    // no new particles spawn and live particles finish their lifetimes.
    void tick(float dt, const World& world, const Vec3& playerPosition,
              const WeatherSample& weather);

    // Render active particles as block-space billboards after the air-medium
    // composite. Physical scale affects only their Beer-Lambert optical path.
    // Instance data + uniforms sub-allocate from the caller's frame ring so
    // the per-frame rewrite never races frames the GPU still reads.
    void render(id<MTLRenderCommandEncoder> encoder, FrameRing& frameRing, const Mat4& viewMatrix,
                const Mat4& projectionMatrix, const Vec3& cameraPosition,
                const WeatherSample& weather, float baseExtinction,
                WorldPhysicalScale physicalScale);

private:
    // ---- CPU particle pool ----
    WeatherParticleSessionState session_;

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
