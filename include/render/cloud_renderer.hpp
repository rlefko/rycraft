#pragma once

#import <Metal/Metal.h>

#include "render/shader_types.hpp"
#include "world/weather.hpp"

#include <array>
#include <cstdint>
#include <vector>

class GpuFrameTimer;

// Periodic deterministic noise used for startup texture generation and
// contract tests. Every axis tiles at edge without a seam.
float cloudBaseNoise(int x, int y, int z, int edge, uint64_t seed) noexcept;

struct CloudNoiseGenerationStats {
    uint64_t voxelCount = 0;
    uint64_t hashEvaluations = 0;
    uint64_t worleyFeatureTests = 0;
};

// Builds the startup R8 volume from precomputed periodic value lattices and
// Worley feature offsets. The result matches cloudBaseNoise after R8
// quantization without hashing independently for every voxel.
std::vector<uint8_t> generateCloudBaseNoiseVolume(int edge, uint64_t seed,
                                                  CloudNoiseGenerationStats* stats = nullptr);

double wrappedCloudOffset(double current, double velocityBlocksPerSecond,
                          double deltaSeconds) noexcept;
simd_float4 encodeCloudMotionOffset(worldgen::Vector2d offset) noexcept;
simd_float4 encodeCloudMotion(const WeatherSample& sample) noexcept;
simd_float2 decodeCloudMotion(simd_float4 encoded) noexcept;
simd_float2 cloudMotionDelta(simd_float2 current, simd_float2 previous) noexcept;

// Texture payload retained by the cloud renderer. Driver page alignment is
// device-specific and is intentionally outside this budget.
struct CloudRendererMemoryFootprint {
    uint32_t quarterWidth = 1;
    uint32_t quarterHeight = 1;
    uint32_t shadowEdge = 0;
    uint64_t noiseBytes = 0;
    uint64_t weatherBytes = 0;
    uint64_t neutralShadowBytes = 0;
    uint64_t frameTargetBytes = 0;
    uint64_t shadowBytes = 0;

    uint64_t totalBytes() const noexcept {
        return noiseBytes + weatherBytes + neutralShadowBytes + frameTargetBytes + shadowBytes;
    }
};

CloudRendererMemoryFootprint cloudRendererMemoryFootprint(uint32_t width, uint32_t height,
                                                          int quality) noexcept;

class CloudRenderer {
public:
    CloudRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                  uint32_t height, uint64_t worldSeed);
    ~CloudRenderer();

    void resize(uint32_t width, uint32_t height);
    void setQuality(int quality);
    void resetHistory();
    void updateWeather(const WeatherSnapshot& snapshot, uint64_t worldTick, uint32_t frameSlot);

    // Runs before opaque shading so every later pass sees the same snapped
    // cloud transmittance field.
    void encodeShadow(id<MTLCommandBuffer> commandBuffer, CloudShadowUniforms uniforms,
                      GpuFrameTimer* timer = nullptr);

    // Marches, temporally resolves, and composites clouds over scene HDR.
    void encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                id<MTLTexture> sceneDepth, CloudRenderUniforms uniforms,
                GpuFrameTimer* timer = nullptr);

    id<MTLTexture> shadowTexture() const {
        return _quality > 0 && _weatherValid ? _cloudShadow : _neutralCloudShadow;
    }
    id<MTLTexture> weatherCloudTexture() const { return _weatherCloud[_activeWeatherSlot]; }
    id<MTLTexture> weatherLayerTexture() const { return _weatherLayer[_activeWeatherSlot]; }
    id<MTLTexture> weatherMotionTexture() const { return _weatherMotion[_activeWeatherSlot]; }
    WeatherMapUniforms weatherMapForCamera(simd_float3 cameraPosition) const;
    id<MTLTexture> resolvedHitDepth() const {
        return _historyValid ? _historyDepth[_historyIndex] : nil;
    }
    id<MTLTexture> resolvedCloudTexture() const {
        return _historyValid ? _historyCloud[_historyIndex] : nil;
    }
    const CloudShadowUniforms& shadowUniforms() const { return _shadowUniforms; }
    bool historyValid() const { return _historyValid; }
    uint64_t persistentBytes() const { return _persistentBytes; }

private:
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _marchPipeline;
    id<MTLComputePipelineState> _temporalPipeline;
    id<MTLComputePipelineState> _shadowPipeline;
    id<MTLRenderPipelineState> _compositePipeline;
    id<MTLTexture> _baseNoise;
    id<MTLTexture> _erosionNoise;
    id<MTLTexture> _curlNoise;
    static constexpr uint32_t WEATHER_TEXTURE_SLOTS = 3;
    id<MTLTexture> _weatherCloud[WEATHER_TEXTURE_SLOTS]{};
    id<MTLTexture> _weatherLayer[WEATHER_TEXTURE_SLOTS]{};
    id<MTLTexture> _weatherMotion[WEATHER_TEXTURE_SLOTS]{};
    id<MTLTexture> _currentCloud{};
    id<MTLTexture> _currentDepth{};
    id<MTLTexture> _historyCloud[2]{};
    id<MTLTexture> _historyDepth[2]{};
    id<MTLTexture> _cloudShadow{};
    id<MTLTexture> _neutralCloudShadow;
    uint32_t _displayWidth;
    uint32_t _displayHeight;
    uint64_t _worldSeed = 0;
    uint32_t _quarterWidth = 1;
    uint32_t _quarterHeight = 1;
    int _quality = 0;
    uint32_t _historyIndex = 0;
    bool _historyValid = false;
    bool _weatherValid = false;
    WeatherMapUniforms _weatherMap{};
    WeatherMapUniforms _previousWeatherMap{};
    CloudShadowUniforms _shadowUniforms{};
    std::array<uint64_t, WEATHER_TEXTURE_SLOTS> _weatherSignatures{};
    std::array<bool, WEATHER_TEXTURE_SLOTS> _weatherSlotValid{};
    std::array<std::vector<simd_float4>, 2> _weatherCloudUpload;
    std::array<std::vector<simd_float4>, 2> _weatherLayerUpload;
    std::array<std::vector<simd_float4>, 4> _weatherMotionUpload;
    uint64_t _weatherUploadSignature = 0;
    bool _weatherUploadValid = false;
    uint64_t _currentWeatherSignature = 0;
    int64_t _weatherOriginX = 0;
    int64_t _weatherOriginZ = 0;
    int64_t _previousWeatherOriginX = 0;
    int64_t _previousWeatherOriginZ = 0;
    uint32_t _activeWeatherSlot = 0;
    uint32_t _previousWeatherSlot = 0;
    uint64_t _lastShadowFrame = 0;
    simd_float3 _lastShadowLightDirection = simd_make_float3(0.0F, 1.0F, 0.0F);
    bool _shadowValid = false;
    bool _shadowDirty = true;
    uint64_t _persistentBytes = 0;

    void allocateNoise(uint64_t worldSeed);
    void allocateFrameTargets();
    void allocateWeatherTextures();
    static WeatherMapUniforms weatherMapForCamera(const WeatherMapUniforms& weatherMap,
                                                  int64_t originX, int64_t originZ,
                                                  simd_float3 cameraPosition);
    static void dispatch2D(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline, NSUInteger width,
                           NSUInteger height);
};
