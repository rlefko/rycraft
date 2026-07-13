#import "render/particles.hpp"

#include "common/error.hpp"
#include "world/world.hpp"

#include <cmath>
#include <cstring>
#include <random>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ParticleSystem::ParticleSystem(id<MTLDevice> device, id<MTLLibrary> shaderLibrary)
    : _device(device)
    , _pipelineState(nil)
    , _depthState(nil)
    , _particleBuffer(nil)
    , _uniformsBuffer(nil)
{
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

    pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipelineDesc.colorAttachments[0].blendingEnabled = true;
    pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    NSError* error = nil;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDesc
                                                             error:&error];
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

    // ---- GPU particle buffer ----
    _particleBuffer = [_device newBufferWithLength:sizeof(GPUParticle) * MAX_PARTICLES
                                             options:MTLResourceStorageModeShared];
    if (!_particleBuffer) {
        RY_LOG_FATAL("Failed to allocate particle GPU buffer");
    }

    // ---- Uniform buffer ----
    _uniformsBuffer = [_device newBufferWithLength:sizeof(ParticleUniforms)
                                             options:MTLResourceStorageModeShared];
    if (!_uniformsBuffer) {
        RY_LOG_FATAL("Failed to allocate particle uniforms buffer");
    }
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
ParticleSystem::~ParticleSystem() {
    // Metal objects released via ARC (nil assignment)
    _pipelineState = nil;
    _depthState = nil;
    _particleBuffer = nil;
    _uniformsBuffer = nil;
}

// ---------------------------------------------------------------------------
// tick — Update particle physics each game tick
// ---------------------------------------------------------------------------
void ParticleSystem::tick(float dt, const World& world, const Vec3& playerPosition) {
    if (dt <= 0.f) return;

    // Clamp dt to prevent physics explosions after long pauses
    float clampedDt = std::min(dt, 0.1f);

    // Spawn radius based on view distance (use player position as center)
    constexpr float SPAWN_RADIUS = 64.f;

    // ---- Count active particles ----
    size_t activeCount = 0;
    for (size_t i = 0; i < MAX_PARTICLES; ++i) {
        if (particles_[i].active) {
            ++activeCount;
        }
    }

    // ---- Update existing particles ----
    for (size_t i = 0; i < MAX_PARTICLES; ++i) {
        Particle& p = particles_[i];
        if (!p.active) continue;

        // Advance lifetime
        p.lifetime += clampedDt;

        // Kill if exceeded max lifetime or fell below Y=0
        if (p.lifetime >= p.maxLifetime || p.position[1] < 0.f) {
            p.active = false;
            continue;
        }

        // Apply velocity
        p.position[0] += p.velocity[0] * clampedDt;
        p.position[1] += p.velocity[1] * clampedDt;
        p.position[2] += p.velocity[2] * clampedDt;

        // Snow: sinusoidal horizontal drift
        if (p.type == ParticleType::SNOW) {
            float driftAngle = p.lifetime * 2.0f + p.position[0] * 0.1f;
            p.velocity[0] += std::sin(driftAngle) * 0.5f * clampedDt;
            p.velocity[2] += std::cos(driftAngle) * 0.5f * clampedDt;

            // Dampen horizontal velocity to prevent runaway drift
            p.velocity[0] *= 0.99f;
            p.velocity[2] *= 0.99f;
        }
    }

    // ---- Spawn new particles to fill pool ----
    // Target: keep ~70% of pool active
    constexpr size_t TARGET_ACTIVE = static_cast<size_t>(MAX_PARTICLES * 0.7);
    size_t toSpawn = (activeCount < TARGET_ACTIVE) ? (TARGET_ACTIVE - activeCount) : 0;

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
            if (!found) continue; // Pool full, skip this spawn
        }
        Particle& p = particles_[slot];

        // Determine biome at spawn location
        int spawnX = static_cast<int>(playerPosition.x);
        int spawnZ = static_cast<int>(playerPosition.z);

        if (isSnowBiome(world, spawnX, spawnZ)) {
            spawnSnowParticle(p, playerPosition, SPAWN_RADIUS);
        } else {
            spawnRainParticle(p, playerPosition, SPAWN_RADIUS);
        }
    }
}

// ---------------------------------------------------------------------------
// spawnRainParticle
// ---------------------------------------------------------------------------
void ParticleSystem::spawnRainParticle(Particle& p, const Vec3& playerPos, float spawnRadius) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> angleDist(0.f, 2.f * static_cast<float>(M_PI));
    std::uniform_real_distribution<float> radiusDist(0.f, spawnRadius);
    std::uniform_real_distribution<float> unitDist(0.f, 1.f);

    float angle = angleDist(rng);
    float radius = radiusDist(rng);

    // Spawn above player at high altitude
    p.position[0] = playerPos.x + std::cos(angle) * radius;
    p.position[1] = playerPos.y + 128.f + unitDist(rng) * 32.f;
    p.position[2] = playerPos.z + std::sin(angle) * radius;

    // Fall speed ~10 blocks/s with slight wind drift
    p.velocity[0] = (unitDist(rng) - 0.5f) * 1.0f;
    p.velocity[1] = -10.f;
    p.velocity[2] = (unitDist(rng) - 0.5f) * 1.0f;

    p.lifetime = 0.f;
    p.maxLifetime = 2.0f;
    p.type = ParticleType::RAIN;
    p.active = true;
}

// ---------------------------------------------------------------------------
// spawnSnowParticle
// ---------------------------------------------------------------------------
void ParticleSystem::spawnSnowParticle(Particle& p, const Vec3& playerPos, float spawnRadius) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> angleDist(0.f, 2.f * static_cast<float>(M_PI));
    std::uniform_real_distribution<float> radiusDist(0.f, spawnRadius);
    std::uniform_real_distribution<float> unitDist(0.f, 1.f);

    float angle = angleDist(rng);
    float radius = radiusDist(rng);

    // Spawn above player
    p.position[0] = playerPos.x + std::cos(angle) * radius;
    p.position[1] = playerPos.y + 96.f + unitDist(rng) * 32.f;
    p.position[2] = playerPos.z + std::sin(angle) * radius;

    // Fall speed ~3 blocks/s with gentle initial drift
    p.velocity[0] = (unitDist(rng) - 0.5f) * 0.5f;
    p.velocity[1] = -3.f;
    p.velocity[2] = (unitDist(rng) - 0.5f) * 0.5f;

    p.lifetime = 0.f;
    p.maxLifetime = 5.0f;
    p.type = ParticleType::SNOW;
    p.active = true;
}

// ---------------------------------------------------------------------------
// isSnowBiome
// ---------------------------------------------------------------------------
bool ParticleSystem::isSnowBiome(const World& world, int x, int z) const {
    Biome biome = world.getBiome(x, z);
    return biome == Biome::IceSpikes || biome == Biome::Taiga;
}

// ---------------------------------------------------------------------------
// render — Draw active particles as billboards
// ---------------------------------------------------------------------------
void ParticleSystem::render(id<MTLRenderCommandEncoder> encoder,
                            const Mat4& viewMatrix,
                            const Mat4& projectionMatrix,
                            const Vec3& cameraPosition)
{
    if (!encoder || !_pipelineState) return;

    // ---- Upload particle data to GPU ----
    GPUParticle* gpuParticles = reinterpret_cast<GPUParticle*>(_particleBuffer.contents);
    size_t activeCount = 0;

    for (size_t i = 0; i < MAX_PARTICLES; ++i) {
        const Particle& p = particles_[i];
        if (!p.active) continue;

        GPUParticle& gp = gpuParticles[activeCount];
        gp.position = simd_make_float3(p.position[0], p.position[1], p.position[2]);
        gp.velocity = simd_make_float3(p.velocity[0], p.velocity[1], p.velocity[2]);
        gp.lifetime = p.lifetime;
        gp.type = static_cast<float>(static_cast<uint8_t>(p.type));
        ++activeCount;
    }

    // Skip draw if no active particles
    if (activeCount == 0) return;

    // ---- Upload uniforms ----
    ParticleUniforms uniforms{};
    std::memcpy(&uniforms.viewMatrix, viewMatrix.data.data(), sizeof(uniforms.viewMatrix));
    std::memcpy(&uniforms.projectionMatrix, projectionMatrix.data.data(),
                sizeof(uniforms.projectionMatrix));
    uniforms.cameraPosition = simd_make_float3(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    std::memcpy((void*)_uniformsBuffer.contents, &uniforms, sizeof(uniforms));

    // ---- Bind and draw ----
    [encoder setRenderPipelineState:_pipelineState];
    [encoder setDepthStencilState:_depthState];

    [encoder setVertexBuffer:_particleBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:_uniformsBuffer offset:0 atIndex:1];
    [encoder setFragmentBuffer:_uniformsBuffer offset:0 atIndex:1];

    // Draw as point primitives (one point per particle)
    [encoder drawPrimitives:MTLPrimitiveTypePoint
                  vertexStart:0
                   vertexCount:activeCount];
}
