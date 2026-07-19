#import "render/particles.hpp"

#include "common/error.hpp"
#include "render/pixel_formats.hpp"
#include "world/world.hpp"

#include "common/random.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ParticleSystem::ParticleSystem(id<MTLDevice> device, id<MTLLibrary> shaderLibrary)
    : _device(device), _pipelineState(nil), _depthState(nil) {
    // Zero-initialize all particles
    std::memset(particles_, 0, sizeof(particles_));

    // ---- Load particle shader functions ----
    id<MTLFunction> vertexFunc = [shaderLibrary newFunctionWithName:@"particleVertexMain"];
    if (!vertexFunc) {
        RY_LOG_FATAL("Failed to load particle vertex shader 'particleVertexMain'");
    }

    id<MTLFunction> fragmentFunc = [shaderLibrary newFunctionWithName:@"particleFragmentMain"];
    if (!fragmentFunc) {
        RY_LOG_FATAL("Failed to load particle fragment shader 'particleFragmentMain'");
    }

    // ---- Render pipeline state (alpha-blended points) ----
    // The vertex shader indexes the GPUParticle buffer directly by vertex_id,
    // so no vertex descriptor is involved.
    auto pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDesc.vertexFunction = vertexFunc;
    pipelineDesc.fragmentFunction = fragmentFunc;

    pipelineDesc.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    pipelineDesc.colorAttachments[0].blendingEnabled = true;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor =
        MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.depthAttachmentPixelFormat = PixelFormats::SCENE_DEPTH;
    // Weather is composited after the resolved froxel pass.
    pipelineDesc.rasterSampleCount = 1;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
    if (!_pipelineState) {
        NSString* msg = [NSString stringWithFormat:@"Failed to create particle pipeline state: %@",
                                                   error.localizedDescription];
        RY_LOG_FATAL(msg.UTF8String);
    }

    // ---- Depth stencil state (depth test, no write) ----
    auto depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLess;
    depthDesc.depthWriteEnabled = false;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthDesc];
    if (!_depthState) {
        RY_LOG_FATAL("Failed to create particle depth stencil state");
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
ParticleSystem::~ParticleSystem() {
    // Metal objects released via ARC (nil assignment)
    _pipelineState = nil;
    _depthState = nil;
}

// ---------------------------------------------------------------------------
// tick, Update particle physics each game tick
// ---------------------------------------------------------------------------
void ParticleSystem::tick(float dt, const World& world, const Vec3& playerPosition,
                          const WeatherSample& weather) {
    if (dt <= 0.f)
        return;

    // Clamp dt to prevent physics explosions after long pauses
    float clampedDt = std::min(dt, 0.1f);

    // Spawn radius based on view distance (use player position as center)
    constexpr float SPAWN_RADIUS = 64.f;

    // ---- Update existing particles and count survivors ----
    size_t activeCount = 0;
    for (size_t i = 0; i < MAX_PARTICLES; ++i) {
        Particle& p = particles_[i];
        if (!p.active)
            continue;

        // Advance lifetime
        p.lifetime += clampedDt;

        // Apply velocity
        p.position[0] += p.velocity[0] * clampedDt;
        p.position[1] += p.velocity[1] * clampedDt;
        p.position[2] += p.velocity[2] * clampedDt;

        const float windScale = p.type == ParticleType::RAIN ? 0.65f : 0.82f;
        const float response = std::min(1.0f, clampedDt * 1.8f);
        p.velocity[0] += (weather.windBlocksPerSecond.x * windScale - p.velocity[0]) * response;
        p.velocity[2] += (weather.windBlocksPerSecond.y * windScale - p.velocity[2]) * response;

        // Snow: sinusoidal horizontal drift
        if (p.type == ParticleType::SNOW) {
            float driftAngle = p.lifetime * 2.0f + p.position[0] * 0.1f;
            p.velocity[0] += std::sin(driftAngle) * 0.5f * clampedDt;
            p.velocity[2] += std::cos(driftAngle) * 0.5f * clampedDt;
        }

        // Count after deaths, including particles that crossed the lower
        // boundary this tick. The former pre-update count underfilled the pool
        // whenever a large cohort expired together.
        if (p.lifetime >= p.maxLifetime || p.position[1] < 0.f) {
            p.active = false;
            continue;
        }
        ++activeCount;
    }

    // ---- Spawn new particles to fill pool ----
    // Intensity scales the same bounded pool instead of turning all
    // precipitation into a binary 70-percent load.
    const float intensity = std::clamp(weather.precipitationIntensity, 0.0f, 1.0f);
    const bool precipitating = weather.precipitationKind != PrecipitationKind::NONE;
    const size_t targetActive =
        precipitating ? static_cast<size_t>(static_cast<float>(MAX_PARTICLES) * 0.7f * intensity)
                      : 0;
    size_t toSpawn = (activeCount < targetActive) ? (targetActive - activeCount) : 0;

    // Limit spawn rate per tick to prevent sudden bursts
    constexpr size_t MAX_SPAWN_PER_TICK = 256;
    toSpawn = std::min(toSpawn, MAX_SPAWN_PER_TICK);

    for (size_t spawned = 0; spawned < toSpawn; ++spawned) {
        // Find a dead particle slot
        size_t slot = spawned % MAX_PARTICLES;
        if (particles_[slot].active) {
            // Linear probe for next dead slot
            bool found = false;
            for (size_t probe = 1; probe < MAX_PARTICLES; ++probe) {
                slot = (spawned + probe) % MAX_PARTICLES;
                if (!particles_[slot].active) {
                    found = true;
                    break;
                }
            }
            if (!found)
                continue; // Pool full, skip this spawn
        }
        Particle& p = particles_[slot];

        if (weather.precipitationKind == PrecipitationKind::SNOW) {
            spawnSnowParticle(p, world, playerPosition, SPAWN_RADIUS, weather);
        } else if (weather.precipitationKind == PrecipitationKind::RAIN) {
            spawnRainParticle(p, world, playerPosition, SPAWN_RADIUS, weather);
        }
    }
}

// ---------------------------------------------------------------------------
// spawnRainParticle
// ---------------------------------------------------------------------------
bool ParticleSystem::spawnRainParticle(Particle& p, const World& world, const Vec3& playerPos,
                                       float spawnRadius, const WeatherSample& weather) {
    // Deterministic weather: the same seed always rains the same way
    static thread_local SeededRng rng(0x52594352u /* 'RYCR' */);

    float angle = rng.nextFloat() * 2.f * static_cast<float>(M_PI);
    float radius = rng.nextFloat() * spawnRadius;

    // Spawn in a band just above the camera: at 10 blocks/s a drop lives
    // long enough to fall PAST eye level and below the feet, the old
    // +128-block spawn with a 2 s lifetime died ~110 blocks up, so rain was
    // simulated but never once visible on screen.
    p.position[0] = playerPos.x + std::cos(angle) * radius;
    p.position[1] = playerPos.y + 12.f + rng.nextFloat() * 32.f;
    p.position[2] = playerPos.z + std::sin(angle) * radius;

    // Only sky-exposed columns rain: a spawn point at or below the column's
    // surface is inside terrain (cave, overhang) and would rain indoors.
    auto surface = world.surfaceHeightIfLoaded(static_cast<int64_t>(std::floor(p.position[0])),
                                               static_cast<int64_t>(std::floor(p.position[2])));
    if (!surface || p.position[1] <= static_cast<float>(*surface + 1)) {
        return false;
    }

    // Fall speed ~10 blocks/s with drift from the canonical weather wind.
    p.velocity[0] = weather.windBlocksPerSecond.x * 0.65f + (rng.nextFloat() - 0.5f);
    p.velocity[1] = -10.f;
    p.velocity[2] = weather.windBlocksPerSecond.y * 0.65f + (rng.nextFloat() - 0.5f);

    p.lifetime = 0.f;
    p.maxLifetime = 5.0f;
    p.type = ParticleType::RAIN;
    p.active = true;
    return true;
}

// ---------------------------------------------------------------------------
// spawnSnowParticle
// ---------------------------------------------------------------------------
bool ParticleSystem::spawnSnowParticle(Particle& p, const World& world, const Vec3& playerPos,
                                       float spawnRadius, const WeatherSample& weather) {
    // Deterministic weather: the same seed always rains the same way
    static thread_local SeededRng rng(0x52594352u /* 'RYCR' */);

    float angle = rng.nextFloat() * 2.f * static_cast<float>(M_PI);
    float radius = rng.nextFloat() * spawnRadius;

    // Spawn just above the camera (see the rain comment: a high spawn band
    // with a short lifetime kept every flake far above the screen).
    p.position[0] = playerPos.x + std::cos(angle) * radius;
    p.position[1] = playerPos.y + 8.f + rng.nextFloat() * 20.f;
    p.position[2] = playerPos.z + std::sin(angle) * radius;

    // Same sky-exposure gate as rain (see spawnRainParticle).
    auto surface = world.surfaceHeightIfLoaded(static_cast<int64_t>(std::floor(p.position[0])),
                                               static_cast<int64_t>(std::floor(p.position[2])));
    if (!surface || p.position[1] <= static_cast<float>(*surface + 1)) {
        return false;
    }

    // Fall speed ~3 blocks/s with gentle drift around the shared weather wind.
    p.velocity[0] = weather.windBlocksPerSecond.x * 0.82f + (rng.nextFloat() - 0.5f) * 0.5f;
    p.velocity[1] = -3.f;
    p.velocity[2] = weather.windBlocksPerSecond.y * 0.82f + (rng.nextFloat() - 0.5f) * 0.5f;

    p.lifetime = 0.f;
    p.maxLifetime = 10.0f;
    p.type = ParticleType::SNOW;
    p.active = true;
    return true;
}

// ---------------------------------------------------------------------------
// render, Draw active particles as billboards
// ---------------------------------------------------------------------------
void ParticleSystem::render(id<MTLRenderCommandEncoder> encoder, FrameRing& frameRing,
                            const Mat4& viewMatrix, const Mat4& projectionMatrix,
                            const Vec3& cameraPosition, const WeatherSample& weather,
                            float baseExtinction) {
    if (!encoder || !_pipelineState)
        return;

    // ---- Upload particle data into this frame's ring slot ----
    FrameRing::Alloc instances = frameRing.alloc(sizeof(GPUParticle) * MAX_PARTICLES);
    GPUParticle* gpuParticles = static_cast<GPUParticle*>(instances.ptr);
    size_t activeCount = 0;

    for (size_t i = 0; i < MAX_PARTICLES; ++i) {
        const Particle& p = particles_[i];
        if (!p.active)
            continue;

        GPUParticle& gp = gpuParticles[activeCount];
        gp.position = simd_make_float3(p.position[0], p.position[1], p.position[2]);
        gp.velocity = simd_make_float3(p.velocity[0], p.velocity[1], p.velocity[2]);
        gp.lifetime = p.lifetime;
        gp.type = static_cast<float>(static_cast<uint8_t>(p.type));
        ++activeCount;
    }

    // Skip draw if no active particles
    if (activeCount == 0)
        return;

    // ---- Upload uniforms ----
    ParticleUniforms uniforms{};
    std::memcpy(&uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(&uniforms.projectionMatrix, projectionMatrix.data.data(),
                sizeof(uniforms.projectionMatrix));
    uniforms.cameraPosition =
        simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    uniforms.atmosphericExtinction =
        std::max(baseExtinction, 0.0F) * (0.35F + std::max(weather.aerosolDensity, 0.0F)) +
        std::max(weather.fogExtinction, 0.0F);
    FrameRing::Alloc uniformsAlloc = frameRing.push(&uniforms, sizeof(uniforms));

    // ---- Bind and draw ----
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];

    [encoder setVertexBuffer:instances.buffer offset:instances.offset atIndex:0];
    [encoder setVertexBuffer:uniformsAlloc.buffer offset:uniformsAlloc.offset atIndex:1];
    [encoder setFragmentBuffer:uniformsAlloc.buffer offset:uniformsAlloc.offset atIndex:1];

    // Draw as point primitives (one point per particle)
    [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:activeCount];
}
