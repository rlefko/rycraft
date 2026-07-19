#import "render/cloud_renderer.hpp"

#include "common/error.hpp"
#include "common/random.hpp"
#include "render/gpu_timer.hpp"
#include "render/metal_ownership.hpp"
#include "render/pixel_formats.hpp"
#include "world/chunk_pos.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

static_assert(WeatherSnapshot::GRID_EDGE == WEATHER_MAP_EDGE);
static_assert(WeatherSnapshot::GRID_SPACING == static_cast<int>(WEATHER_MAP_CELL_SPACING));

namespace {

constexpr double CLOUD_OFFSET_WRAP = static_cast<double>(CLOUD_MOTION_WRAP_BLOCKS);
constexpr double TWO_PI = 6.28318530717958647692;

float hashUnit(int x, int y, int z, int edge, uint64_t seed) {
    x = world_coord::floorMod(x, edge);
    y = world_coord::floorMod(y, edge);
    z = world_coord::floorMod(z, edge);
    uint64_t value = seed;
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(x)) * 0x9E3779B185EBCA87ULL;
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(y)) * 0xC2B2AE3D27D4EB4FULL;
    value ^= static_cast<uint64_t>(static_cast<uint32_t>(z)) * 0x165667B19E3779F9ULL;
    return static_cast<float>(hash64(value) >> 40U) / static_cast<float>(1U << 24U);
}

float smooth(float value) {
    return value * value * (3.0F - 2.0F * value);
}

template <typename Sample>
float evaluatePeriodicValueNoise(float x, float y, float z, int edge, int frequency,
                                 const Sample& sample) {
    const float gridScale = static_cast<float>(frequency) / static_cast<float>(edge);
    x *= gridScale;
    y *= gridScale;
    z *= gridScale;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const float fx = smooth(x - std::floor(x));
    const float fy = smooth(y - std::floor(y));
    const float fz = smooth(z - std::floor(z));
    const float c00 = std::lerp(sample(x0, y0, z0), sample(x0 + 1, y0, z0), fx);
    const float c10 = std::lerp(sample(x0, y0 + 1, z0), sample(x0 + 1, y0 + 1, z0), fx);
    const float c01 = std::lerp(sample(x0, y0, z0 + 1), sample(x0 + 1, y0, z0 + 1), fx);
    const float c11 = std::lerp(sample(x0, y0 + 1, z0 + 1), sample(x0 + 1, y0 + 1, z0 + 1), fx);
    return std::lerp(std::lerp(c00, c10, fy), std::lerp(c01, c11, fy), fz);
}

float periodicValueNoise(float x, float y, float z, int edge, int frequency, uint64_t seed) {
    return evaluatePeriodicValueNoise(x, y, z, edge, frequency, [&](int sx, int sy, int sz) {
        return hashUnit(sx, sy, sz, frequency, seed);
    });
}

struct WorleyFeatureOffset {
    float x;
    float y;
    float z;
};

template <typename FeatureSample>
float evaluatePeriodicWorley(float x, float y, float z, int edge, int cells,
                             const FeatureSample& sample) {
    const float scale = static_cast<float>(cells) / static_cast<float>(edge);
    const float px = x * scale;
    const float py = y * scale;
    const float pz = z * scale;
    const int cx = static_cast<int>(std::floor(px));
    const int cy = static_cast<int>(std::floor(py));
    const int cz = static_cast<int>(std::floor(pz));
    float minimumDistanceSquared = 3.0F;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int cellX = cx + dx;
                const int cellY = cy + dy;
                const int cellZ = cz + dz;
                const WorleyFeatureOffset feature = sample(cellX, cellY, cellZ);
                const float ddx = static_cast<float>(cellX) + feature.x - px;
                const float ddy = static_cast<float>(cellY) + feature.y - py;
                const float ddz = static_cast<float>(cellZ) + feature.z - pz;
                minimumDistanceSquared =
                    std::min(minimumDistanceSquared, ddx * ddx + ddy * ddy + ddz * ddz);
            }
        }
    }
    return std::clamp(1.0F - std::sqrt(minimumDistanceSquared), 0.0F, 1.0F);
}

float periodicWorley(float x, float y, float z, int edge, int cells, uint64_t seed) {
    return evaluatePeriodicWorley(x, y, z, edge, cells, [&](int cellX, int cellY, int cellZ) {
        return WorleyFeatureOffset{
            hashUnit(cellX, cellY, cellZ, cells, seed),
            hashUnit(cellX, cellY, cellZ, cells, seed ^ 0xA53A9B1DULL),
            hashUnit(cellX, cellY, cellZ, cells, seed ^ 0xC13FA9A9ULL),
        };
    });
}

size_t periodicIndex(int x, int y, int z, int edge) {
    x = world_coord::floorMod(x, edge);
    y = world_coord::floorMod(y, edge);
    z = world_coord::floorMod(z, edge);
    return (static_cast<size_t>(z) * static_cast<size_t>(edge) + static_cast<size_t>(y)) *
               static_cast<size_t>(edge) +
           static_cast<size_t>(x);
}

class PeriodicValueLattice {
public:
    PeriodicValueLattice(int frequency, uint64_t seed, uint64_t* hashEvaluations = nullptr)
        : _frequency(frequency), _values(static_cast<size_t>(frequency) * frequency * frequency) {
        for (int z = 0; z < frequency; ++z) {
            for (int y = 0; y < frequency; ++y) {
                for (int x = 0; x < frequency; ++x) {
                    _values[periodicIndex(x, y, z, frequency)] = hashUnit(x, y, z, frequency, seed);
                }
            }
        }
        if (hashEvaluations) {
            *hashEvaluations += _values.size();
        }
    }

    float evaluate(float x, float y, float z, int edge) const {
        return evaluatePeriodicValueNoise(x, y, z, edge, _frequency,
                                          [&](int sx, int sy, int sz) { return at(sx, sy, sz); });
    }

private:
    int _frequency;
    std::vector<float> _values;

    float at(int x, int y, int z) const { return _values[periodicIndex(x, y, z, _frequency)]; }
};

class PeriodicWorleyFeatures {
public:
    PeriodicWorleyFeatures(int cells, uint64_t seed, uint64_t* hashEvaluations = nullptr)
        : _cells(cells), _offsets(static_cast<size_t>(cells) * cells * cells) {
        for (int z = 0; z < cells; ++z) {
            for (int y = 0; y < cells; ++y) {
                for (int x = 0; x < cells; ++x) {
                    WorleyFeatureOffset& feature = _offsets[periodicIndex(x, y, z, cells)];
                    feature.x = hashUnit(x, y, z, cells, seed);
                    feature.y = hashUnit(x, y, z, cells, seed ^ 0xA53A9B1DULL);
                    feature.z = hashUnit(x, y, z, cells, seed ^ 0xC13FA9A9ULL);
                }
            }
        }
        if (hashEvaluations) {
            *hashEvaluations += _offsets.size() * 3U;
        }
    }

    float evaluate(float x, float y, float z, int edge) const {
        return evaluatePeriodicWorley(x, y, z, edge, _cells, [&](int cellX, int cellY, int cellZ) {
            return at(cellX, cellY, cellZ);
        });
    }

private:
    int _cells;
    std::vector<WorleyFeatureOffset> _offsets;

    const WorleyFeatureOffset& at(int x, int y, int z) const {
        return _offsets[periodicIndex(x, y, z, _cells)];
    }
};

id<MTLComputePipelineState> makeComputePipeline(id<MTLDevice> device, id<MTLLibrary> shaderLibrary,
                                                NSString* functionName) {
    id<MTLFunction> function = [shaderLibrary newFunctionWithName:functionName];
    if (!function) {
        RY_LOG_FATAL("Failed to load cloud compute function");
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    resetMetalObject(function);
    if (!pipeline) {
        RY_LOG_FATAL("Failed to create cloud compute pipeline");
    }
    return pipeline;
}

id<MTLTexture> make2D(id<MTLDevice> device, MTLPixelFormat format, NSUInteger width,
                      NSUInteger height, MTLTextureUsage usage, NSString* label) {
    auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                         width:width
                                                                        height:height
                                                                     mipmapped:false];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = usage;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    texture.label = label;
    if (!texture) {
        RY_LOG_FATAL("Failed to allocate cloud render texture");
    }
    return texture;
}

void releaseTexture(id<MTLTexture> __strong& texture) {
    resetMetalObject(texture);
}

} // namespace

float cloudBaseNoise(int x, int y, int z, int edge, uint64_t seed) noexcept {
    if (edge <= 0) {
        return 0.0F;
    }
    const float coarse = periodicValueNoise(static_cast<float>(x), static_cast<float>(y),
                                            static_cast<float>(z), edge, 4, seed);
    const float detail =
        periodicValueNoise(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z),
                           edge, 8, seed ^ 0xD1B54A32D192ED03ULL);
    const float worley =
        periodicWorley(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), edge, 8,
                       seed ^ 0x94D049BB133111EBULL);
    return std::clamp(coarse * 0.50F + detail * 0.20F + worley * 0.30F, 0.0F, 1.0F);
}

std::vector<uint8_t> generateCloudBaseNoiseVolume(int edge, uint64_t seed,
                                                  CloudNoiseGenerationStats* stats) {
    CloudNoiseGenerationStats measured{};
    if (edge <= 0) {
        if (stats) {
            *stats = measured;
        }
        return {};
    }

    PeriodicValueLattice coarse(4, seed, &measured.hashEvaluations);
    PeriodicValueLattice detail(8, seed ^ 0xD1B54A32D192ED03ULL, &measured.hashEvaluations);
    PeriodicWorleyFeatures worley(8, seed ^ 0x94D049BB133111EBULL, &measured.hashEvaluations);
    const size_t voxelCount = static_cast<size_t>(edge) * edge * edge;
    std::vector<uint8_t> volume(voxelCount);
    for (int z = 0; z < edge; ++z) {
        for (int y = 0; y < edge; ++y) {
            for (int x = 0; x < edge; ++x) {
                const float broad = coarse.evaluate(static_cast<float>(x), static_cast<float>(y),
                                                    static_cast<float>(z), edge);
                const float fine = detail.evaluate(static_cast<float>(x), static_cast<float>(y),
                                                   static_cast<float>(z), edge);
                const float cells = worley.evaluate(static_cast<float>(x), static_cast<float>(y),
                                                    static_cast<float>(z), edge);
                const float noise =
                    std::clamp(broad * 0.50F + fine * 0.20F + cells * 0.30F, 0.0F, 1.0F);
                volume[(static_cast<size_t>(z) * edge + static_cast<size_t>(y)) * edge +
                       static_cast<size_t>(x)] = static_cast<uint8_t>(noise * 255.0F + 0.5F);
            }
        }
    }
    measured.voxelCount = voxelCount;
    measured.worleyFeatureTests = static_cast<uint64_t>(voxelCount) * 27U;
    if (stats) {
        *stats = measured;
    }
    return volume;
}

double wrappedCloudOffset(double current, double velocityBlocksPerSecond,
                          double deltaSeconds) noexcept {
    double result = std::fmod(current + velocityBlocksPerSecond * deltaSeconds, CLOUD_OFFSET_WRAP);
    return result < 0.0 ? result + CLOUD_OFFSET_WRAP : result;
}

simd_float4 encodeCloudMotionOffset(worldgen::Vector2d offset) noexcept {
    const double x = wrappedCloudOffset(offset.x, 0.0, 0.0);
    const double z = wrappedCloudOffset(offset.z, 0.0, 0.0);
    const double xAngle = x * TWO_PI / CLOUD_OFFSET_WRAP;
    const double zAngle = z * TWO_PI / CLOUD_OFFSET_WRAP;
    return simd_make_float4(
        static_cast<float>(std::cos(xAngle)), static_cast<float>(std::sin(xAngle)),
        static_cast<float>(std::cos(zAngle)), static_cast<float>(std::sin(zAngle)));
}

simd_float4 encodeCloudMotion(const WeatherSample& sample) noexcept {
    return encodeCloudMotionOffset(sample.cloudOffsetBlocks);
}

simd_float2 decodeCloudMotion(simd_float4 encoded) noexcept {
    double xAngle = std::atan2(static_cast<double>(encoded.y), static_cast<double>(encoded.x));
    double zAngle = std::atan2(static_cast<double>(encoded.w), static_cast<double>(encoded.z));
    if (xAngle < 0.0)
        xAngle += TWO_PI;
    if (zAngle < 0.0)
        zAngle += TWO_PI;
    return simd_make_float2(static_cast<float>(xAngle * CLOUD_OFFSET_WRAP / TWO_PI),
                            static_cast<float>(zAngle * CLOUD_OFFSET_WRAP / TWO_PI));
}

simd_float2 cloudMotionDelta(simd_float2 current, simd_float2 previous) noexcept {
    simd_float2 delta = current - previous;
    delta.x -= std::round(delta.x / CLOUD_MOTION_WRAP_BLOCKS) * CLOUD_MOTION_WRAP_BLOCKS;
    delta.y -= std::round(delta.y / CLOUD_MOTION_WRAP_BLOCKS) * CLOUD_MOTION_WRAP_BLOCKS;
    return delta;
}

CloudRendererMemoryFootprint cloudRendererMemoryFootprint(uint32_t width, uint32_t height,
                                                          int quality) noexcept {
    constexpr uint32_t CURL_NOISE_EDGE = 128U;
    constexpr uint32_t WEATHER_TEXTURE_SLOTS = 3U;
    CloudRendererMemoryFootprint footprint;
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    quality = std::clamp(quality, 0, 2);
    footprint.noiseBytes = static_cast<uint64_t>(CLOUD_BASE_NOISE_EDGE) * CLOUD_BASE_NOISE_EDGE *
                               CLOUD_BASE_NOISE_EDGE +
                           static_cast<uint64_t>(CLOUD_EROSION_NOISE_EDGE) *
                               CLOUD_EROSION_NOISE_EDGE * CLOUD_EROSION_NOISE_EDGE +
                           static_cast<uint64_t>(CURL_NOISE_EDGE) * CURL_NOISE_EDGE * 2U;
    // Cloud and layer fields each use two time slices. Motion uses four so
    // low and high layers preserve independent circular phases.
    constexpr uint32_t WEATHER_SLICES_PER_SLOT = 2U + 2U + 4U;
    footprint.weatherBytes = static_cast<uint64_t>(WEATHER_MAP_EDGE) * WEATHER_MAP_EDGE * 16U *
                             WEATHER_SLICES_PER_SLOT * WEATHER_TEXTURE_SLOTS;
    footprint.neutralShadowBytes = 2U;
    if (quality == 0) {
        return footprint;
    }

    footprint.quarterWidth = std::max(width / 4U, 1U);
    footprint.quarterHeight = std::max(height / 4U, 1U);
    footprint.shadowEdge = quality >= 2 ? 2048U : 1024U;
    const uint64_t quarterPixels =
        static_cast<uint64_t>(footprint.quarterWidth) * footprint.quarterHeight;
    footprint.frameTargetBytes = quarterPixels * (8U * 3U + 2U * 3U);
    footprint.shadowBytes = static_cast<uint64_t>(footprint.shadowEdge) * footprint.shadowEdge * 2U;
    return footprint;
}

CloudRenderer::CloudRenderer(id<MTLDevice> device, id<MTLLibrary> shaderLibrary, uint32_t width,
                             uint32_t height, uint64_t worldSeed)
    : _device(device), _displayWidth(std::max(width, 1U)), _displayHeight(std::max(height, 1U)),
      _worldSeed(worldSeed) {
    _marchPipeline = makeComputePipeline(device, shaderLibrary, @"volumetricCloudMarchKernel");
    _temporalPipeline =
        makeComputePipeline(device, shaderLibrary, @"volumetricCloudTemporalKernel");
    _shadowPipeline = makeComputePipeline(device, shaderLibrary, @"cloudShadowKernel");

    id<MTLFunction> vertex = [shaderLibrary newFunctionWithName:@"cloudCompositeVertex"];
    id<MTLFunction> fragment = [shaderLibrary newFunctionWithName:@"cloudCompositeFragment"];
    if (!vertex || !fragment) {
        RY_LOG_FATAL("Failed to load cloud composite functions");
    }
    auto descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.vertexFunction = vertex;
    descriptor.fragmentFunction = fragment;
    descriptor.colorAttachments[0].pixelFormat = PixelFormats::SCENE_HDR;
    descriptor.colorAttachments[0].blendingEnabled = true;
    descriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
    descriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    descriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    descriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    NSError* error = nil;
    _compositePipeline = [_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    resetMetalObject(descriptor);
    resetMetalObject(vertex);
    resetMetalObject(fragment);
    if (!_compositePipeline) {
        RY_LOG_FATAL("Failed to create cloud composite pipeline");
    }

    allocateNoise(worldSeed);
    allocateWeatherTextures();
    auto neutralDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:false];
    neutralDescriptor.storageMode = MTLStorageModeShared;
    neutralDescriptor.usage = MTLTextureUsageShaderRead;
    _neutralCloudShadow = [_device newTextureWithDescriptor:neutralDescriptor];
    _neutralCloudShadow.label = @"Neutral Cloud Transmittance";
    if (!_neutralCloudShadow) {
        RY_LOG_FATAL("Failed to allocate neutral cloud transmittance");
    }
    constexpr uint16_t HALF_ONE = 0x3C00U;
    [_neutralCloudShadow replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                           mipmapLevel:0
                             withBytes:&HALF_ONE
                           bytesPerRow:sizeof(HALF_ONE)];
    allocateFrameTargets();
}

CloudRenderer::~CloudRenderer() {
    releaseTexture(_baseNoise);
    releaseTexture(_erosionNoise);
    releaseTexture(_curlNoise);
    for (uint32_t slot = 0; slot < WEATHER_TEXTURE_SLOTS; ++slot) {
        releaseTexture(_weatherCloud[slot]);
        releaseTexture(_weatherLayer[slot]);
        releaseTexture(_weatherMotion[slot]);
    }
    releaseTexture(_currentCloud);
    releaseTexture(_currentDepth);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_historyCloud[index]);
        releaseTexture(_historyDepth[index]);
    }
    releaseTexture(_cloudShadow);
    releaseTexture(_neutralCloudShadow);
    resetMetalObject(_marchPipeline);
    resetMetalObject(_temporalPipeline);
    resetMetalObject(_shadowPipeline);
    resetMetalObject(_compositePipeline);
}

void CloudRenderer::dispatch2D(id<MTLComputeCommandEncoder> encoder,
                               id<MTLComputePipelineState> pipeline, NSUInteger width,
                               NSUInteger height) {
    const NSUInteger threadWidth = std::min<NSUInteger>(pipeline.threadExecutionWidth, 16);
    const NSUInteger threadHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(
               pipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1), 16));
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(threadWidth, threadHeight, 1)];
}

void CloudRenderer::allocateNoise(uint64_t worldSeed) {
    auto baseDescriptor = [[MTLTextureDescriptor alloc] init];
    baseDescriptor.textureType = MTLTextureType3D;
    baseDescriptor.pixelFormat = MTLPixelFormatR8Unorm;
    baseDescriptor.width = CLOUD_BASE_NOISE_EDGE;
    baseDescriptor.height = CLOUD_BASE_NOISE_EDGE;
    baseDescriptor.depth = CLOUD_BASE_NOISE_EDGE;
    baseDescriptor.storageMode = MTLStorageModeShared;
    baseDescriptor.usage = MTLTextureUsageShaderRead;
    _baseNoise = [_device newTextureWithDescriptor:baseDescriptor];
    _baseNoise.label = @"Perlin-Worley Cloud Base";
    const std::vector<uint8_t> base =
        generateCloudBaseNoiseVolume(CLOUD_BASE_NOISE_EDGE, worldSeed);
    [_baseNoise replaceRegion:MTLRegionMake3D(0, 0, 0, CLOUD_BASE_NOISE_EDGE, CLOUD_BASE_NOISE_EDGE,
                                              CLOUD_BASE_NOISE_EDGE)
                  mipmapLevel:0
                        slice:0
                    withBytes:base.data()
                  bytesPerRow:CLOUD_BASE_NOISE_EDGE
                bytesPerImage:CLOUD_BASE_NOISE_EDGE * CLOUD_BASE_NOISE_EDGE];

    MTLTextureDescriptor* erosionDescriptor = [baseDescriptor copy];
    erosionDescriptor.width = CLOUD_EROSION_NOISE_EDGE;
    erosionDescriptor.height = CLOUD_EROSION_NOISE_EDGE;
    erosionDescriptor.depth = CLOUD_EROSION_NOISE_EDGE;
    _erosionNoise = [_device newTextureWithDescriptor:erosionDescriptor];
    _erosionNoise.label = @"Worley Cloud Erosion";
    const PeriodicWorleyFeatures erosionFeatures(8, worldSeed ^ 0x6A09E667F3BCC909ULL);
    std::vector<uint8_t> erosion(static_cast<size_t>(CLOUD_EROSION_NOISE_EDGE) *
                                 CLOUD_EROSION_NOISE_EDGE * CLOUD_EROSION_NOISE_EDGE);
    for (int z = 0; z < CLOUD_EROSION_NOISE_EDGE; ++z) {
        for (int y = 0; y < CLOUD_EROSION_NOISE_EDGE; ++y) {
            for (int x = 0; x < CLOUD_EROSION_NOISE_EDGE; ++x) {
                const size_t index = (static_cast<size_t>(z) * CLOUD_EROSION_NOISE_EDGE + y) *
                                         CLOUD_EROSION_NOISE_EDGE +
                                     x;
                erosion[index] = static_cast<uint8_t>(
                    erosionFeatures.evaluate(static_cast<float>(x), static_cast<float>(y),
                                             static_cast<float>(z), CLOUD_EROSION_NOISE_EDGE) *
                        255.0F +
                    0.5F);
            }
        }
    }
    [_erosionNoise replaceRegion:MTLRegionMake3D(0, 0, 0, CLOUD_EROSION_NOISE_EDGE,
                                                 CLOUD_EROSION_NOISE_EDGE, CLOUD_EROSION_NOISE_EDGE)
                     mipmapLevel:0
                           slice:0
                       withBytes:erosion.data()
                     bytesPerRow:CLOUD_EROSION_NOISE_EDGE
                   bytesPerImage:CLOUD_EROSION_NOISE_EDGE * CLOUD_EROSION_NOISE_EDGE];

    constexpr int CURL_EDGE = 128;
    auto curlDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG8Snorm
                                                           width:CURL_EDGE
                                                          height:CURL_EDGE
                                                       mipmapped:false];
    curlDescriptor.storageMode = MTLStorageModeShared;
    curlDescriptor.usage = MTLTextureUsageShaderRead;
    _curlNoise = [_device newTextureWithDescriptor:curlDescriptor];
    _curlNoise.label = @"Cloud Curl Noise";
    const PeriodicValueLattice curlLattice(8, worldSeed);
    std::vector<int8_t> curl(static_cast<size_t>(CURL_EDGE) * CURL_EDGE * 2);
    for (int y = 0; y < CURL_EDGE; ++y) {
        for (int x = 0; x < CURL_EDGE; ++x) {
            const float left = curlLattice.evaluate(static_cast<float>(x - 1),
                                                    static_cast<float>(y), 0.0F, CURL_EDGE);
            const float right = curlLattice.evaluate(static_cast<float>(x + 1),
                                                     static_cast<float>(y), 0.0F, CURL_EDGE);
            const float down = curlLattice.evaluate(static_cast<float>(x),
                                                    static_cast<float>(y - 1), 0.0F, CURL_EDGE);
            const float up = curlLattice.evaluate(static_cast<float>(x), static_cast<float>(y + 1),
                                                  0.0F, CURL_EDGE);
            const size_t index = (static_cast<size_t>(y) * CURL_EDGE + x) * 2;
            curl[index] = static_cast<int8_t>(std::clamp((up - down) * 127.0F, -127.0F, 127.0F));
            curl[index + 1] =
                static_cast<int8_t>(std::clamp((left - right) * 127.0F, -127.0F, 127.0F));
        }
    }
    [_curlNoise replaceRegion:MTLRegionMake2D(0, 0, CURL_EDGE, CURL_EDGE)
                  mipmapLevel:0
                    withBytes:curl.data()
                  bytesPerRow:CURL_EDGE * 2];
    resetMetalObject(erosionDescriptor);
    resetMetalObject(baseDescriptor);
}

void CloudRenderer::allocateWeatherTextures() {
    auto descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.textureType = MTLTextureType2DArray;
    descriptor.pixelFormat = MTLPixelFormatRGBA32Float;
    descriptor.width = WEATHER_MAP_EDGE;
    descriptor.height = WEATHER_MAP_EDGE;
    descriptor.arrayLength = 2;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead;
    for (uint32_t slot = 0; slot < WEATHER_TEXTURE_SLOTS; ++slot) {
        _weatherCloud[slot] = [_device newTextureWithDescriptor:descriptor];
        _weatherLayer[slot] = [_device newTextureWithDescriptor:descriptor];
        descriptor.arrayLength = 4;
        _weatherMotion[slot] = [_device newTextureWithDescriptor:descriptor];
        descriptor.arrayLength = 2;
        _weatherCloud[slot].label = @"Weather Cloud Fields";
        _weatherLayer[slot].label = @"Weather Cloud Layers";
        _weatherMotion[slot].label = @"Weather Cloud Motion";
        if (!_weatherCloud[slot] || !_weatherLayer[slot] || !_weatherMotion[slot]) {
            RY_LOG_FATAL("Failed to allocate ringed cloud weather textures");
        }
    }
    resetMetalObject(descriptor);
}

void CloudRenderer::allocateFrameTargets() {
    const CloudRendererMemoryFootprint footprint =
        cloudRendererMemoryFootprint(_displayWidth, _displayHeight, _quality);
    _quarterWidth = footprint.quarterWidth;
    _quarterHeight = footprint.quarterHeight;
    _persistentBytes = footprint.totalBytes();
    releaseTexture(_currentCloud);
    releaseTexture(_currentDepth);
    for (uint32_t index = 0; index < 2; ++index) {
        releaseTexture(_historyCloud[index]);
        releaseTexture(_historyDepth[index]);
    }
    releaseTexture(_cloudShadow);
    _historyIndex = 0;
    _historyValid = false;
    _shadowValid = false;
    _shadowDirty = true;
    if (_quality == 0) {
        return;
    }

    constexpr MTLTextureUsage RW = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    _currentCloud = make2D(_device, MTLPixelFormatRGBA16Float, _quarterWidth, _quarterHeight, RW,
                           @"Cloud Current Scattering");
    _currentDepth = make2D(_device, MTLPixelFormatR16Float, _quarterWidth, _quarterHeight, RW,
                           @"Cloud Current Hit Depth");
    for (uint32_t index = 0; index < 2; ++index) {
        _historyCloud[index] =
            make2D(_device, MTLPixelFormatRGBA16Float, _quarterWidth, _quarterHeight, RW,
                   index == 0 ? @"Cloud History A" : @"Cloud History B");
        _historyDepth[index] =
            make2D(_device, MTLPixelFormatR16Float, _quarterWidth, _quarterHeight, RW,
                   index == 0 ? @"Cloud Hit History A" : @"Cloud Hit History B");
    }
    _cloudShadow = make2D(_device, MTLPixelFormatR16Float, footprint.shadowEdge,
                          footprint.shadowEdge, RW, @"Cloud Shadow Transmittance");
}

void CloudRenderer::resize(uint32_t width, uint32_t height) {
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    if (width == _displayWidth && height == _displayHeight) {
        return;
    }
    _displayWidth = width;
    _displayHeight = height;
    allocateFrameTargets();
}

void CloudRenderer::setQuality(int quality) {
    quality = std::clamp(quality, 0, 2);
    if (_quality == quality) {
        return;
    }
    _quality = quality;
    allocateFrameTargets();
}

void CloudRenderer::resetHistory() {
    _historyValid = false;
}

void CloudRenderer::updateWeather(const WeatherSnapshot& snapshot, uint64_t worldTick,
                                  uint32_t frameSlot) {
    const bool hadWeather = _weatherValid;
    if (hadWeather) {
        _previousWeatherSlot = _activeWeatherSlot;
        _previousWeatherMap = _weatherMap;
        _previousWeatherOriginX = _weatherOriginX;
        _previousWeatherOriginZ = _weatherOriginZ;
    }
    _activeWeatherSlot = frameSlot % WEATHER_TEXTURE_SLOTS;
    const uint64_t signature =
        hash64(_worldSeed ^ snapshot.requestId() ^ static_cast<uint64_t>(snapshot.centerX()) ^
               (static_cast<uint64_t>(snapshot.centerZ()) << 1U) ^ (snapshot.firstTick() << 7U) ^
               (static_cast<uint64_t>(snapshot.preset()) << 59U));
    if (_currentWeatherSignature != signature) {
        _currentWeatherSignature = signature;
        _shadowDirty = true;
    }
    if (!_weatherUploadValid || _weatherUploadSignature != signature) {
        for (int slice = 0; slice < 2; ++slice) {
            _weatherCloudUpload[slice].resize(WeatherSnapshot::GRID_SAMPLE_COUNT);
            _weatherLayerUpload[slice].resize(WeatherSnapshot::GRID_SAMPLE_COUNT);
            _weatherMotionUpload[slice].resize(WeatherSnapshot::GRID_SAMPLE_COUNT);
            _weatherMotionUpload[slice + 2].resize(WeatherSnapshot::GRID_SAMPLE_COUNT);
            size_t index = 0;
            for (const WeatherSample& sample : snapshot.timeSlice(slice)) {
                _weatherCloudUpload[slice][index] = simd_make_float4(
                    sample.cloudCoverage, sample.relativeHumidity, sample.stormPotential,
                    static_cast<float>(sample.cloudType) /
                        static_cast<float>(CloudType::CUMULONIMBUS));
                _weatherLayerUpload[slice][index] =
                    simd_make_float4(sample.cloudBaseY, sample.cloudTopY,
                                     sample.precipitationIntensity, sample.aerosolDensity);
                _weatherMotionUpload[slice][index] = encodeCloudMotion(sample);
                _weatherMotionUpload[slice + 2][index] =
                    encodeCloudMotionOffset(sample.highCloudOffsetBlocks);
                ++index;
            }
        }
        _weatherUploadSignature = signature;
        _weatherUploadValid = true;
    }
    if (!_weatherSlotValid[_activeWeatherSlot] ||
        _weatherSignatures[_activeWeatherSlot] != signature) {
        for (int slice = 0; slice < 2; ++slice) {
            const MTLRegion region = MTLRegionMake2D(0, 0, WEATHER_MAP_EDGE, WEATHER_MAP_EDGE);
            [_weatherCloud[_activeWeatherSlot]
                replaceRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(slice)
                    withBytes:_weatherCloudUpload[slice].data()
                  bytesPerRow:WEATHER_MAP_EDGE * sizeof(simd_float4)
                bytesPerImage:WEATHER_MAP_EDGE * WEATHER_MAP_EDGE * sizeof(simd_float4)];
            [_weatherLayer[_activeWeatherSlot]
                replaceRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(slice)
                    withBytes:_weatherLayerUpload[slice].data()
                  bytesPerRow:WEATHER_MAP_EDGE * sizeof(simd_float4)
                bytesPerImage:WEATHER_MAP_EDGE * WEATHER_MAP_EDGE * sizeof(simd_float4)];
            [_weatherMotion[_activeWeatherSlot]
                replaceRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(slice)
                    withBytes:_weatherMotionUpload[slice].data()
                  bytesPerRow:WEATHER_MAP_EDGE * sizeof(simd_float4)
                bytesPerImage:WEATHER_MAP_EDGE * WEATHER_MAP_EDGE * sizeof(simd_float4)];
            [_weatherMotion[_activeWeatherSlot]
                replaceRegion:region
                  mipmapLevel:0
                        slice:static_cast<NSUInteger>(slice + 2)
                    withBytes:_weatherMotionUpload[slice + 2].data()
                  bytesPerRow:WEATHER_MAP_EDGE * sizeof(simd_float4)
                bytesPerImage:WEATHER_MAP_EDGE * WEATHER_MAP_EDGE * sizeof(simd_float4)];
        }
        _weatherSignatures[_activeWeatherSlot] = signature;
        _weatherSlotValid[_activeWeatherSlot] = true;
    }
    const float interpolation =
        worldTick <= snapshot.firstTick()
            ? 0.0F
            : std::clamp(static_cast<float>(worldTick - snapshot.firstTick()) /
                             static_cast<float>(WeatherSnapshot::TIME_SLICE_TICKS),
                         0.0F, 1.0F);
    _weatherOriginX = snapshot.originX();
    _weatherOriginZ = snapshot.originZ();
    _weatherMap.originXZ = simd_make_float2(0.0F, 0.0F);
    _weatherMap.cellSpacing = static_cast<float>(WeatherSnapshot::GRID_SPACING);
    _weatherMap.interpolation = interpolation;
    _weatherMap.gridSize = simd_make_uint2(WeatherSnapshot::GRID_EDGE, WeatherSnapshot::GRID_EDGE);
    _weatherMap.motionWrapBlocks = CLOUD_MOTION_WRAP_BLOCKS;
    _weatherMap.movementMargin = static_cast<float>(WeatherSystem::RECENTER_DISTANCE);
    if (!hadWeather) {
        _previousWeatherSlot = _activeWeatherSlot;
        _previousWeatherMap = _weatherMap;
        _previousWeatherOriginX = _weatherOriginX;
        _previousWeatherOriginZ = _weatherOriginZ;
    }
    _weatherValid = true;
}

WeatherMapUniforms CloudRenderer::weatherMapForCamera(simd_float3 cameraPosition) const {
    return weatherMapForCamera(_weatherMap, _weatherOriginX, _weatherOriginZ, cameraPosition);
}

WeatherMapUniforms CloudRenderer::weatherMapForCamera(const WeatherMapUniforms& weatherMap,
                                                      int64_t originX, int64_t originZ,
                                                      simd_float3 cameraPosition) {
    WeatherMapUniforms result = weatherMap;
    result.originXZ = simd_make_float2(
        static_cast<float>(static_cast<double>(originX) - static_cast<double>(cameraPosition.x)),
        static_cast<float>(static_cast<double>(originZ) - static_cast<double>(cameraPosition.z)));
    return result;
}

void CloudRenderer::encodeShadow(id<MTLCommandBuffer> commandBuffer, CloudShadowUniforms uniforms,
                                 GpuFrameTimer* timer) {
    if (!commandBuffer || _quality == 0 || !_weatherValid) {
        return;
    }
    const uint64_t frameIndex = static_cast<uint64_t>(std::max(uniforms.footprintAndTexel.w, 0.0F));
    const simd_float3 lightDirection = simd_normalize(uniforms.sunDirection);
    const uint64_t refreshInterval = _quality >= 2 ? 2U : 4U;
    const bool lightChanged =
        !_shadowValid || simd_dot(lightDirection, _lastShadowLightDirection) < 0.999F;
    if (_shadowValid && !_shadowDirty && !lightChanged &&
        frameIndex - _lastShadowFrame < refreshInterval) {
        return;
    }
    uniforms.footprintAndTexel.x = 16'384.0F;
    uniforms.footprintAndTexel.y = 16'384.0F / static_cast<float>(_cloudShadow.width);
    uniforms.footprintAndTexel.z = static_cast<float>(_quality);
    uniforms.weatherMap = weatherMapForCamera(uniforms.cameraPosition);
    _shadowUniforms = uniforms;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Cloud Shadows";
    const uint32_t timerToken =
        timer ? timer->beginComputePass(encoder, "cloudShadow") : UINT32_MAX;
    [encoder setComputePipelineState:_shadowPipeline];
    [encoder setTexture:_baseNoise atIndex:0];
    [encoder setTexture:_erosionNoise atIndex:1];
    [encoder setTexture:_curlNoise atIndex:2];
    [encoder setTexture:_weatherCloud[_activeWeatherSlot] atIndex:3];
    [encoder setTexture:_weatherLayer[_activeWeatherSlot] atIndex:4];
    [encoder setTexture:_weatherMotion[_activeWeatherSlot] atIndex:5];
    [encoder setTexture:_cloudShadow atIndex:6];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    dispatch2D(encoder, _shadowPipeline, _cloudShadow.width, _cloudShadow.height);
    if (timer) {
        timer->endComputePass(encoder, timerToken);
    }
    [encoder endEncoding];
    _lastShadowFrame = frameIndex;
    _lastShadowLightDirection = lightDirection;
    _shadowValid = true;
    _shadowDirty = false;
}

void CloudRenderer::encode(id<MTLCommandBuffer> commandBuffer, id<MTLTexture> sceneHDR,
                           id<MTLTexture> sceneDepth, CloudRenderUniforms uniforms,
                           GpuFrameTimer* timer) {
    if (!commandBuffer || !sceneHDR || !sceneDepth || _quality == 0 || !_weatherValid) {
        return;
    }
    uniforms.renderParams.x =
        static_cast<float>(_quality >= 2 ? CLOUD_HIGH_VIEW_STEPS : CLOUD_MEDIUM_VIEW_STEPS);
    uniforms.renderParams.y =
        static_cast<float>(_quality >= 2 ? CLOUD_HIGH_LIGHT_STEPS : CLOUD_MEDIUM_LIGHT_STEPS);
    uniforms.renderParams.z = _historyValid ? 1.0F : 0.0F;
    uniforms.resolutionAndFrame.x = static_cast<float>(_quarterWidth);
    uniforms.resolutionAndFrame.y = static_cast<float>(_quarterHeight);
    uniforms.weatherMap = weatherMapForCamera(uniforms.cameraPosition);
    uniforms.previousWeatherMap =
        weatherMapForCamera(_previousWeatherMap, _previousWeatherOriginX, _previousWeatherOriginZ,
                            uniforms.cameraPosition);

    const uint32_t writeIndex = _historyIndex ^ 1U;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Volumetric Clouds";
    const uint32_t timerToken =
        timer ? timer->beginComputePass(encoder, "cloudMarchTemporal") : UINT32_MAX;
    [encoder setComputePipelineState:_marchPipeline];
    [encoder setTexture:_baseNoise atIndex:0];
    [encoder setTexture:_erosionNoise atIndex:1];
    [encoder setTexture:_curlNoise atIndex:2];
    [encoder setTexture:_weatherCloud[_activeWeatherSlot] atIndex:3];
    [encoder setTexture:_weatherLayer[_activeWeatherSlot] atIndex:4];
    [encoder setTexture:_weatherMotion[_activeWeatherSlot] atIndex:5];
    [encoder setTexture:sceneDepth atIndex:6];
    [encoder setTexture:_currentCloud atIndex:7];
    [encoder setTexture:_currentDepth atIndex:8];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    dispatch2D(encoder, _marchPipeline, _quarterWidth, _quarterHeight);

    [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];

    [encoder setComputePipelineState:_temporalPipeline];
    [encoder setTexture:_currentCloud atIndex:0];
    [encoder setTexture:_currentDepth atIndex:1];
    [encoder setTexture:_historyCloud[_historyIndex] atIndex:2];
    [encoder setTexture:_historyDepth[_historyIndex] atIndex:3];
    [encoder setTexture:_historyCloud[writeIndex] atIndex:4];
    [encoder setTexture:_historyDepth[writeIndex] atIndex:5];
    [encoder setTexture:_weatherMotion[_activeWeatherSlot] atIndex:6];
    [encoder setTexture:_weatherMotion[_previousWeatherSlot] atIndex:7];
    [encoder setTexture:_weatherCloud[_activeWeatherSlot] atIndex:8];
    [encoder setTexture:_weatherLayer[_activeWeatherSlot] atIndex:9];
    [encoder setTexture:_weatherCloud[_previousWeatherSlot] atIndex:10];
    [encoder setTexture:_weatherLayer[_previousWeatherSlot] atIndex:11];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    dispatch2D(encoder, _temporalPipeline, _quarterWidth, _quarterHeight);
    if (timer) {
        timer->endComputePass(encoder, timerToken);
    }
    [encoder endEncoding];

    auto pass = [[MTLRenderPassDescriptor alloc] init];
    pass.colorAttachments[0].texture = sceneHDR;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (timer) {
        timer->attachPass(pass, "cloudComposite");
    }
    id<MTLRenderCommandEncoder> composite = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (!composite) {
        resetMetalObject(pass);
        return;
    }
    composite.label = @"Composite Volumetric Clouds";
    [composite setRenderPipelineState:_compositePipeline];
    [composite setFragmentTexture:_historyCloud[writeIndex] atIndex:0];
    [composite setFragmentTexture:_historyDepth[writeIndex] atIndex:1];
    [composite setFragmentTexture:sceneDepth atIndex:2];
    [composite setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [composite drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [composite endEncoding];
    resetMetalObject(pass);

    _historyIndex = writeIndex;
    _historyValid = true;
}
