#pragma once

#include "world/generator_v4.hpp"

#include <array>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace worldgen {
class NativeHydrologyIdentityRegistry;
class NativeHydrologyRouter;
} // namespace worldgen

namespace worldgen::learned {

inline constexpr int MODEL_BLOCK_SCALE = 4;
inline constexpr int AUTHORITY_PAGE_BLOCK_EDGE = 1'024;
inline constexpr int AUTHORITY_PAGE_NATIVE_EDGE = 256;
inline constexpr size_t AUTHORITY_PAGE_SAMPLE_COUNT =
    static_cast<size_t>(AUTHORITY_PAGE_NATIVE_EDGE) * AUTHORITY_PAGE_NATIVE_EDGE;
inline constexpr uint16_t TERRAIN_AUTHORITY_SCHEMA_VERSION = 1;
inline constexpr uint32_t TERRAIN_CHANNEL_MASK = 0x3FU;
inline constexpr float WINDOW_WEIGHT_EPSILON = 0.001F;
inline constexpr double WORLD_METERS_PER_BLOCK = 7.5;
inline constexpr int LEARNED_SEA_LEVEL = 64;
inline constexpr uint16_t GENERATOR_V4_RNG_REVISION = 1;
inline constexpr uint16_t GENERATOR_V4_QUANTIZATION_REVISION = 2;
// Revision 12 freezes branch-specific fall ownership, reach-scoped channel
// projection, and receiver-owned backwater without proximity-synthesized falls.
inline constexpr uint16_t GENERATOR_V4_HYDROLOGY_REVISION = 12;
// Revision 9 reconstructs PREVIEW from the FINAL latent low-frequency lineage
// with the same cleanup operator, while retaining center-aligned interpolation
// and the canonical postprocessing consumed by exact and far generation.
inline constexpr uint16_t GENERATOR_V4_POSTPROCESSING_REVISION = 9;
inline constexpr size_t MAXIMUM_AUTHORITY_QUERY_PAGES = 64;
inline constexpr size_t MAXIMUM_AUTHORITY_QUEUED_REQUESTS = 64;
// Preview and speculative work may occupy at most half of the shared
// inference queue. The remaining requests stay available to spawn, exact
// exploration, protected handoff, and visible final refinement.
inline constexpr size_t MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS = 32;
// Visible final refinement and both lower-priority lanes share at most three
// quarters of the queue. Sixteen requests therefore remain available to
// protected handoff, exact exploration, and spawn after both lower tiers
// saturate.
inline constexpr size_t MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS = 48;
// A 2x2 group of adjacent 517-sample hydrology inputs spans 1,029 by 1,029
// samples after their shared apron is removed. Admit that bounded union so a
// protected near closure can infer it once and crop each canonical owner from
// the immutable result.
inline constexpr size_t MAXIMUM_AUTHORITY_QUERY_SAMPLES = 1'100'000;
inline constexpr uint16_t MAXIMUM_COARSE_SPAWN_GRID_EDGE = 128;
static_assert(MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS <= MAXIMUM_AUTHORITY_QUEUED_REQUESTS);
static_assert(MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS <=
              MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS);
static_assert(MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS <= MAXIMUM_AUTHORITY_QUEUED_REQUESTS);

struct WindowGeometry {
    uint16_t edge = 0;
    uint16_t stride = 0;
    uint16_t inferenceSteps = 0;
    uint16_t batchSize = 0;

    auto operator<=>(const WindowGeometry&) const = default;
};

inline constexpr WindowGeometry COARSE_WINDOW{
    .edge = 64,
    .stride = 48,
    .inferenceSteps = 20,
    .batchSize = 1,
};
inline constexpr WindowGeometry LATENT_WINDOW{
    .edge = 64,
    .stride = 32,
    .inferenceSteps = 2,
    .batchSize = 4,
};
inline constexpr WindowGeometry DECODER_WINDOW{
    .edge = 256,
    .stride = 192,
    .inferenceSteps = 1,
    .batchSize = 4,
};
// Direct backend qualification may pass one bounded union of pages so tests
// can verify cross-page tensor reuse independently from coordinator policy.
inline constexpr size_t MAXIMUM_FINAL_AUTHORITY_BATCH_PAGES = 24;
// Runtime scheduling yields after each fixed four-page group so a newly
// arrived urgent request is reconsidered before another model call begins.
inline constexpr size_t MAXIMUM_COORDINATOR_PAGE_GROUP_PAGES = LATENT_WINDOW.batchSize;
static_assert(MAXIMUM_COORDINATOR_PAGE_GROUP_PAGES == 4);

enum class AuthorityQuality : uint8_t {
    PREVIEW = 0,
    FINAL = 1,
};

enum class AuthorityStatus : uint8_t {
    READY,
    DEFERRED,
    FAILED,
};

// Lower values are serviced first. These lanes are shared by exact and far
// generation so one slow inference call cannot let speculative work pass the
// safe spawn or connected handoff bands.
enum class AuthorityRequestPriority : uint8_t {
    SPAWN = 0,
    EXPLORATION_EXACT = 1,
    PROTECTED_HANDOFF = 2,
    VISIBLE_FINAL_REFINEMENT = 3,
    COARSE_PREVIEW = 4,
    SPECULATIVE_PREFETCH = 5,
};

// A monotonically increasing view generation attached only to production
// protected-handoff work. The cached coordinator uses it to distinguish the
// camera's current closure from abandoned work at an older view position.
// Zero preserves the unversioned compatibility path used by startup, exact
// generation, and simple test authorities.
struct ProtectedHandoffEpoch {
    uint64_t value = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }
    auto operator<=>(const ProtectedHandoffEpoch&) const = default;
};

enum class GenerationFailureCode : uint8_t {
    NONE,
    INVALID_REQUEST,
    PAGE_NOT_FOUND,
    QUEUE_FULL,
    BACKEND_UNAVAILABLE,
    INFERENCE_FAILED,
    IO_ERROR,
    CORRUPT_PAGE,
    INCOMPATIBLE_FINGERPRINT,
};

struct GenerationFailure {
    GenerationFailureCode code = GenerationFailureCode::NONE;
    std::string message;
    bool retriable = false;

    auto operator<=>(const GenerationFailure&) const = default;
};

template <typename Value>
class AuthorityResult {
public:
    static AuthorityResult ready(Value value) {
        AuthorityResult result;
        result.status_ = AuthorityStatus::READY;
        result.value_.emplace(std::move(value));
        return result;
    }

    static AuthorityResult deferred(GenerationFailure failure) {
        AuthorityResult result;
        result.status_ = AuthorityStatus::DEFERRED;
        result.failure_ = std::move(failure);
        return result;
    }

    static AuthorityResult failed(GenerationFailure failure) {
        AuthorityResult result;
        result.status_ = AuthorityStatus::FAILED;
        result.failure_ = std::move(failure);
        return result;
    }

    [[nodiscard]] AuthorityStatus status() const noexcept { return status_; }
    [[nodiscard]] bool isReady() const noexcept { return status_ == AuthorityStatus::READY; }
    [[nodiscard]] const Value* value() const noexcept {
        return value_ ? std::addressof(*value_) : nullptr;
    }
    [[nodiscard]] Value* value() noexcept { return value_ ? std::addressof(*value_) : nullptr; }
    [[nodiscard]] const GenerationFailure* failure() const noexcept {
        return failure_ ? std::addressof(*failure_) : nullptr;
    }

private:
    AuthorityStatus status_ = AuthorityStatus::FAILED;
    std::optional<Value> value_;
    std::optional<GenerationFailure> failure_;
};

using Sha256Digest = std::array<uint8_t, 32>;

Sha256Digest sha256(std::span<const uint8_t> bytes);
std::optional<Sha256Digest> parseSha256(std::string_view hexadecimal);
std::string sha256Hex(const Sha256Digest& digest);

enum class InferenceProvider : uint8_t {
    CORE_ML = 1,
};

inline constexpr uint32_t CORE_ML_STATIC_SHAPES = 1U << 0U;
inline constexpr uint32_t CORE_ML_ML_PROGRAM = 1U << 1U;
inline constexpr uint32_t CORE_ML_ALL_COMPUTE_UNITS = 1U << 2U;
inline constexpr uint32_t CORE_ML_SEQUENTIAL_EXECUTION = 1U << 3U;
inline constexpr uint32_t CORE_ML_MODEL_CACHE = 1U << 4U;
inline constexpr uint32_t CORE_ML_STATIC_BASE_BATCH_FOUR = 1U << 5U;
inline constexpr uint32_t CORE_ML_STATIC_DECODER_BATCH_FOUR = 1U << 6U;
inline constexpr uint32_t CORE_ML_STATIC_DECODER_SPATIAL_256 = 1U << 7U;
inline constexpr uint32_t CORE_ML_REQUIRED_FLAGS =
    CORE_ML_STATIC_SHAPES | CORE_ML_ML_PROGRAM | CORE_ML_ALL_COMPUTE_UNITS |
    CORE_ML_SEQUENTIAL_EXECUTION | CORE_ML_MODEL_CACHE | CORE_ML_STATIC_BASE_BATCH_FOUR |
    CORE_ML_STATIC_DECODER_BATCH_FOUR | CORE_ML_STATIC_DECODER_SPATIAL_256;

struct ProviderConfiguration {
    InferenceProvider provider = InferenceProvider::CORE_ML;
    uint16_t onnxRuntimeMajorVersion = 1;
    uint16_t onnxRuntimeMinorVersion = 27;
    uint16_t onnxRuntimePatchVersion = 1;
    uint32_t flags = CORE_ML_REQUIRED_FLAGS;

    auto operator<=>(const ProviderConfiguration&) const = default;
};

inline constexpr ProviderConfiguration GENERATOR_V4_PROVIDER_CONFIGURATION{
    .provider = InferenceProvider::CORE_ML,
    .onnxRuntimeMajorVersion = 1,
    .onnxRuntimeMinorVersion = 27,
    .onnxRuntimePatchVersion = 1,
    .flags = CORE_ML_REQUIRED_FLAGS,
};

struct GenerationIdentity {
    uint32_t generatorVersion = GENERATOR_V4_VERSION;
    uint64_t seed = 0;
    Sha256Digest modelPackHash{};
    Sha256Digest runtimeHash{};
    ProviderConfiguration provider = GENERATOR_V4_PROVIDER_CONFIGURATION;
    uint16_t modelBlockScale = MODEL_BLOCK_SCALE;
    uint16_t rngRevision = GENERATOR_V4_RNG_REVISION;
    uint16_t quantizationRevision = GENERATOR_V4_QUANTIZATION_REVISION;
    uint16_t hydrologyRevision = GENERATOR_V4_HYDROLOGY_REVISION;
    uint16_t postprocessingRevision = GENERATOR_V4_POSTPROCESSING_REVISION;
    WindowGeometry coarseWindow = COARSE_WINDOW;
    WindowGeometry latentWindow = LATENT_WINDOW;
    WindowGeometry decoderWindow = DECODER_WINDOW;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] Sha256Digest fingerprint() const;

    auto operator<=>(const GenerationIdentity&) const = default;
};

struct NativeRect {
    int64_t rowBegin = 0;
    int64_t columnBegin = 0;
    int64_t rowEnd = 0;
    int64_t columnEnd = 0;

    [[nodiscard]] bool valid() const noexcept {
        return rowBegin < rowEnd && columnBegin < columnEnd;
    }
    [[nodiscard]] uint64_t height() const noexcept {
        return static_cast<uint64_t>(rowEnd) - static_cast<uint64_t>(rowBegin);
    }
    [[nodiscard]] uint64_t width() const noexcept {
        return static_cast<uint64_t>(columnEnd) - static_cast<uint64_t>(columnBegin);
    }

    auto operator<=>(const NativeRect&) const = default;
};

// Coordinates in the learned coarse model grid. One coarse cell spans one
// immutable authority page, or 256 native pixels and 1,024 world blocks.
// Spawn selection is the only consumer: it deliberately remains separate
// from native-field queries so it never creates a preview authority page.
struct CoarseSpawnRegion {
    int64_t rowBegin = 0;
    int64_t columnBegin = 0;
    int64_t rowEnd = 0;
    int64_t columnEnd = 0;

    [[nodiscard]] bool valid() const noexcept {
        return rowBegin < rowEnd && columnBegin < columnEnd;
    }
    [[nodiscard]] uint64_t height() const noexcept {
        return static_cast<uint64_t>(rowEnd) - static_cast<uint64_t>(rowBegin);
    }
    [[nodiscard]] uint64_t width() const noexcept {
        return static_cast<uint64_t>(columnEnd) - static_cast<uint64_t>(columnBegin);
    }

    auto operator<=>(const CoarseSpawnRegion&) const = default;
};

struct NativePoint {
    int64_t row = 0;
    int64_t column = 0;

    auto operator<=>(const NativePoint&) const = default;
};

struct WorldBlockPoint {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const WorldBlockPoint&) const = default;
};

// InfiniteDiffusion treats model rows as world Z and model columns as world X.
// This returns the containing native pixel. Block-resolution physical queries
// use the reference scale-four, align_corners=false bilinear reconstruction.
NativePoint worldBlockToNative(int64_t worldX, int64_t worldZ);

// Matches the scale-four conversion used by the reference implementation,
// including its compressed bathymetry and Rycraft's Y=64 sea plane.
double learnedElevationMetersToWorldHeight(double elevationMeters) noexcept;

struct WindowIndex {
    int64_t row = 0;
    int64_t column = 0;

    auto operator<=>(const WindowIndex&) const = default;
};

// Compatibility boundary for the published model-pipeline vectors. New world
// coordinate code should use world_coord::floorDiv directly.
int64_t floorDivide(int64_t value, int64_t divisor);
std::vector<WindowIndex> intersectingWindows(const NativeRect& region, WindowGeometry geometry);
float linearWindowWeight(size_t offset, size_t edge);

class WeightedWindowAccumulator {
public:
    WeightedWindowAccumulator(NativeRect target, WindowGeometry geometry, size_t channels);
    ~WeightedWindowAccumulator();

    WeightedWindowAccumulator(WeightedWindowAccumulator&&) noexcept;
    WeightedWindowAccumulator& operator=(WeightedWindowAccumulator&&) noexcept;
    WeightedWindowAccumulator(const WeightedWindowAccumulator&) = delete;
    WeightedWindowAccumulator& operator=(const WeightedWindowAccumulator&) = delete;

    [[nodiscard]] bool addWindow(WindowIndex index, std::span<const float> channelMajorValues);
    [[nodiscard]] AuthorityResult<std::vector<float>> resolve() const;
    [[nodiscard]] size_t windowCount() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class PortablePcg64 {
public:
    explicit PortablePcg64(uint64_t seed) : state_(seed) {}

    uint32_t next32() noexcept;
    [[nodiscard]] uint64_t state() const noexcept { return state_; }

private:
    uint64_t state_;
};

uint64_t terrainTileSeed(uint64_t baseSeed, int64_t tileRow, int64_t tileColumn) noexcept;
void fillStandardNormal(uint64_t seed, std::span<float> output);
AuthorityResult<std::vector<float>>
gaussianNoisePatch(uint64_t baseSeed, NativeRect region, size_t channels,
                   int64_t tileEdge = AUTHORITY_PAGE_NATIVE_EDGE);

struct TerrainPageCoordinate {
    int64_t row = 0;
    int64_t column = 0;

    auto operator<=>(const TerrainPageCoordinate&) const = default;
};

// Converts a native model pixel to its immutable half-open authority page.
// Negative coordinates use mathematical floor division, so native -1 belongs
// to page -1 and native -256 is that page's first sample.
TerrainPageCoordinate terrainPageCoordinateFor(NativePoint point) noexcept;

// Returns the zero-based native sample offset within an authority page.
size_t terrainPageLocalCoordinate(int64_t coordinate) noexcept;

// Returns the exact half-open native rectangle owned by a page, or no value
// when either endpoint cannot be represented by signed 64-bit coordinates.
std::optional<NativeRect> terrainPageNativeRect(TerrainPageCoordinate coordinate) noexcept;

struct TerrainPageKey {
    AuthorityQuality quality = AuthorityQuality::FINAL;
    TerrainPageCoordinate coordinate;

    auto operator<=>(const TerrainPageKey&) const = default;
};

struct QuantizedTerrainSample {
    int16_t elevationMeters = 0;
    int16_t meanTemperatureCentidegrees = 0;
    uint16_t temperatureVariabilityCentidegrees = 0;
    uint16_t annualPrecipitationMillimeters = 0;
    uint16_t precipitationCoefficientBasisPoints = 0;
    int16_t lapseRateMicrodegreesPerMeter = 0;

    auto operator<=>(const QuantizedTerrainSample&) const = default;
};

static_assert(sizeof(QuantizedTerrainSample) == 12);

// A direct backend group retains only the quantized immutable outputs until
// its caller accepts them. Keep the largest qualification handoff below the
// temporary-output bound.
inline constexpr size_t MAXIMUM_FINAL_AUTHORITY_BATCH_OUTPUT_BYTES =
    MAXIMUM_FINAL_AUTHORITY_BATCH_PAGES * AUTHORITY_PAGE_SAMPLE_COUNT *
    sizeof(QuantizedTerrainSample);
static_assert(MAXIMUM_FINAL_AUTHORITY_BATCH_OUTPUT_BYTES <= 64ULL * 1'024 * 1'024);

struct PhysicalTerrainSample {
    double elevationMeters = 0.0;
    double meanTemperatureC = 0.0;
    double temperatureVariabilityC = 0.0;
    double annualPrecipitationMm = 0.0;
    double precipitationCoefficientOfVariation = 0.0;
    double lapseRateCPerMeter = 0.0;

    auto operator<=>(const PhysicalTerrainSample&) const = default;
};

// Converts an already canonical physical sample back to the immutable RYTA
// channel units. Transient FINAL grids deliberately contain dequantized RYTA
// values, so this inverse is exact and lets a containing grid publish the
// same page payload as direct page inference without another model call.
QuantizedTerrainSample quantizePhysicalTerrainSample(
    const PhysicalTerrainSample& sample) noexcept;
PhysicalTerrainSample dequantizeTerrainSample(const QuantizedTerrainSample& sample) noexcept;

struct PhysicalTerrainGrid {
    NativeRect region;
    std::vector<PhysicalTerrainSample> samples;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const PhysicalTerrainSample* sample(int64_t row, int64_t column) const noexcept;
};

// Nonpersistent, low-frequency elevation sampled directly from the coarse
// model output. Values are physical meters after signed-square-root decoding.
// This is only a land proposal field. Final authority and hydrology remain
// the sole source for collision and a playable spawn.
struct CoarseSpawnGrid {
    CoarseSpawnRegion region;
    std::vector<float> elevationMeters;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const float* sample(int64_t row, int64_t column) const noexcept;
};

struct TerrainAuthorityPage {
    TerrainPageKey key;
    uint64_t generationSeed = 0;
    Sha256Digest generationFingerprint{};
    std::vector<QuantizedTerrainSample> samples;

    [[nodiscard]] bool valid() const noexcept {
        return samples.size() == AUTHORITY_PAGE_SAMPLE_COUNT;
    }
    [[nodiscard]] size_t byteSize() const noexcept {
        return samples.size() * sizeof(QuantizedTerrainSample);
    }
    [[nodiscard]] bool matches(const GenerationIdentity& identity) const {
        return generationSeed == identity.seed && generationFingerprint == identity.fingerprint();
    }
    [[nodiscard]] const QuantizedTerrainSample* sample(size_t row, size_t column) const noexcept;
};

class TerrainInferenceBackend {
public:
    virtual ~TerrainInferenceBackend() = default;
    virtual AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                            TerrainPageKey key) = 0;
    virtual AuthorityResult<TerrainAuthorityPage>
    inferPageForRequest(const GenerationIdentity& identity, TerrainPageKey key,
                        AuthorityRequestPriority priority);
    // The default preserves the single-page contract for deterministic fake
    // and test backends. The production backend overrides it to prepare the
    // shared latent windows for one bounded FINAL page group before emitting
    // the same immutable pages it would emit one at a time. Results retain
    // the input-key order.
    virtual AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPages(const GenerationIdentity& identity, std::span<const TerrainPageKey> keys);
    virtual AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPagesForRequest(const GenerationIdentity& identity, std::span<const TerrainPageKey> keys,
                         AuthorityRequestPriority priority);
    // Canonical hydrology consumes a narrow apron around a 2,048-block
    // owner. Materializing every intersected 1,024-block persistence page
    // would infer twelve mostly unused pages. Production backends may answer
    // one exact, quantized FINAL rectangle through the same global window
    // lattice without publishing those unrelated page interiors.
    virtual AuthorityResult<PhysicalTerrainGrid> inferFinalNativeGrid(const GenerationIdentity&,
                                                                      NativeRect);
    virtual AuthorityResult<PhysicalTerrainGrid>
    inferFinalNativeGridForRequest(const GenerationIdentity&, NativeRect, AuthorityRequestPriority);
    // The production backend overrides this with the model's coarse-window
    // output. Backends without an explicit coarse implementation fail closed.
    virtual AuthorityResult<CoarseSpawnGrid> inferCoarseSpawnGrid(const GenerationIdentity&,
                                                                  CoarseSpawnRegion);
    virtual AuthorityResult<CoarseSpawnGrid>
    inferCoarseSpawnGridForRequest(const GenerationIdentity&, CoarseSpawnRegion,
                                   AuthorityRequestPriority);
};

class DeterministicFakeTerrainBackend final : public TerrainInferenceBackend {
public:
    explicit DeterministicFakeTerrainBackend(
        std::chrono::milliseconds latency = std::chrono::milliseconds{0});
    ~DeterministicFakeTerrainBackend() override;

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override;
    AuthorityResult<PhysicalTerrainGrid> inferFinalNativeGrid(const GenerationIdentity& identity,
                                                              NativeRect region) override;
    AuthorityResult<CoarseSpawnGrid> inferCoarseSpawnGrid(const GenerationIdentity& identity,
                                                          CoarseSpawnRegion region) override;

    [[nodiscard]] uint64_t callCount() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class TerrainPageStore {
public:
    struct TestHooks {
        // Test-only seam used to make cross-process-style publication races
        // deterministic. It runs after staging is durable and before the
        // per-page exclusive publication lock is acquired.
        std::function<void()> beforeExclusivePublish;
    };

    TerrainPageStore(std::filesystem::path root, GenerationIdentity identity,
                     std::shared_ptr<const TestHooks> testHooks = nullptr);

    [[nodiscard]] AuthorityResult<TerrainAuthorityPage> loadPage(TerrainPageKey key) const;
    [[nodiscard]] AuthorityResult<bool> writePage(const TerrainAuthorityPage& page) const;
    [[nodiscard]] std::filesystem::path pagePath(TerrainPageKey key) const;

private:
    std::filesystem::path root_;
    GenerationIdentity identity_;
    std::shared_ptr<const TestHooks> testHooks_;
};

struct TerrainAuthorityCacheConfig {
    size_t maximumEntries = 1'024;
    size_t byteBudget = 512ULL * 1'024 * 1'024;
    size_t maximumOutstandingRequests = MAXIMUM_AUTHORITY_QUEUED_REQUESTS;
    size_t maximumConcurrentBuilds = 1;
    size_t maximumQueryPages = MAXIMUM_AUTHORITY_QUERY_PAGES;
    size_t maximumQuerySamples = MAXIMUM_AUTHORITY_QUERY_SAMPLES;
};

struct TerrainAuthorityCacheMetrics {
    size_t entries = 0;
    size_t bytes = 0;
    // Model evaluation remains exactly one-at-a-time. Immutable page
    // compression and atomic publication run in a separate bounded lane so
    // file synchronization cannot stall the next learned inference request.
    size_t activeBuilds = 0;
    size_t queuedBuilds = 0;
    // Reservation occupancy includes queued, active, publishing, and
    // completed single flights until their authoritative result is consumed.
    size_t lowPriorityOutstandingRequests = 0;
    size_t visibleOrLowerOutstandingRequests = 0;
    size_t activePublications = 0;
    size_t queuedPublications = 0;
    size_t peakConcurrentPublications = 0;
    bool coordinatorStarted = false;
    bool publicationWorkersStarted = false;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t diskLoads = 0;
    uint64_t transientDiskLoads = 0;
    uint64_t builds = 0;
    // Advances only after a page publication, transient rectangle, coarse
    // result, or terminal failure is observable to a caller. Schedulers use
    // this generation to park dynamically discovered dependencies without
    // polling an in-flight model request from worker threads.
    uint64_t completionGeneration = 0;
    // One batch remains one active inference lane. These counters distinguish
    // coordinator groups from the immutable pages they produced.
    uint64_t batches = 0;
    uint64_t batchedPages = 0;
    uint64_t repairs = 0;
    uint64_t publicationWrites = 0;
    uint64_t transientRepairs = 0;
    uint64_t transientPublicationWrites = 0;
    uint64_t evictions = 0;
    uint64_t singleFlightDeferrals = 0;
    uint64_t deferredRequests = 0;
    uint64_t lowPriorityDeferredRequests = 0;
    uint64_t visibleOrLowerDeferredRequests = 0;
    uint64_t currentProtectedHandoffEpoch = 0;
    // Number of unstarted lower-priority requests displaced so SPAWN,
    // EXPLORATION_EXACT, or current protected handoff authority can enter a
    // saturated coordinator without waiting for distant work to drain.
    uint64_t higherPriorityPreemptions = 0;
    uint64_t protectedHandoffPreemptions = 0;
    uint64_t staleProtectedHandoffDeferrals = 0;
};

class TerrainAuthority {
public:
    virtual ~TerrainAuthority() = default;
    [[nodiscard]] virtual const GenerationIdentity& generationIdentity() const noexcept = 0;
    virtual AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    preparePage(TerrainPageKey key, AuthorityRequestPriority priority) = 0;
    // Epoch-aware requests retain source compatibility for custom authorities.
    // CachedTerrainAuthority overrides these entry points; simpler authorities
    // delegate to their ordinary request implementation.
    virtual AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    preparePage(TerrainPageKey key, AuthorityRequestPriority priority,
                ProtectedHandoffEpoch epoch) {
        static_cast<void>(epoch);
        return preparePage(key, priority);
    }
    // Admit an ordered closure of immutable pages together. The production
    // cache overrides this so no coordinator can begin the first page before
    // every member of a bounded spawn closure has entered the same priority
    // lane. The compatibility default delegates to preparePage for simple
    // test authorities and other non-cached implementations.
    virtual AuthorityResult<bool> preparePages(std::span<const TerrainPageKey> keys,
                                               AuthorityRequestPriority priority);
    virtual AuthorityResult<bool> preparePages(std::span<const TerrainPageKey> keys,
                                               AuthorityRequestPriority priority,
                                               ProtectedHandoffEpoch epoch) {
        static_cast<void>(epoch);
        return preparePages(keys, priority);
    }
    virtual AuthorityResult<PhysicalTerrainGrid>
    queryNative(NativeRect region, AuthorityQuality quality, AuthorityRequestPriority priority) = 0;
    virtual AuthorityResult<std::vector<PhysicalTerrainSample>>
    queryNativePoints(std::span<const NativePoint> points, AuthorityQuality quality,
                      AuthorityRequestPriority priority) = 0;
    virtual AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(NativeRect, AuthorityRequestPriority) {
        return AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>::failed({
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "The terrain authority does not provide transient final rectangles",
            .retriable = false,
        });
    }
    virtual AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(NativeRect region, AuthorityRequestPriority priority,
                                  ProtectedHandoffEpoch epoch) {
        static_cast<void>(epoch);
        return queryTransientFinalNativeGrid(region, priority);
    }
    // This narrow, nonpersistent query is intentionally optional for test
    // authorities. A production CachedTerrainAuthority schedules it through
    // the same one-active-call coordinator used by page inference.
    virtual AuthorityResult<CoarseSpawnGrid> queryCoarseSpawnGrid(CoarseSpawnRegion,
                                                                  AuthorityRequestPriority) {
        return AuthorityResult<CoarseSpawnGrid>::failed({
            .code = GenerationFailureCode::BACKEND_UNAVAILABLE,
            .message = "The terrain authority does not provide coarse spawn selection",
            .retriable = false,
        });
    }
    [[nodiscard]] virtual TerrainAuthorityCacheMetrics cacheMetrics() const = 0;
};

class CachedTerrainAuthority final : public TerrainAuthority {
public:
    CachedTerrainAuthority(GenerationIdentity identity, std::filesystem::path storeRoot,
                           std::shared_ptr<TerrainInferenceBackend> backend,
                           TerrainAuthorityCacheConfig config = {});
    ~CachedTerrainAuthority() override;

    CachedTerrainAuthority(const CachedTerrainAuthority&) = delete;
    CachedTerrainAuthority& operator=(const CachedTerrainAuthority&) = delete;

    AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>> preparePage(
        TerrainPageKey key,
        AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) override;
    AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    preparePage(TerrainPageKey key, AuthorityRequestPriority priority,
                ProtectedHandoffEpoch epoch) override;
    AuthorityResult<bool> preparePages(
        std::span<const TerrainPageKey> keys,
        AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) override;
    AuthorityResult<bool> preparePages(std::span<const TerrainPageKey> keys,
                                       AuthorityRequestPriority priority,
                                       ProtectedHandoffEpoch epoch) override;
    [[nodiscard]] const GenerationIdentity& generationIdentity() const noexcept override;
    AuthorityResult<PhysicalTerrainGrid> queryNative(
        NativeRect region, AuthorityQuality quality,
        AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) override;
    AuthorityResult<std::vector<PhysicalTerrainSample>> queryNativePoints(
        std::span<const NativePoint> points, AuthorityQuality quality,
        AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) override;
    AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>> queryTransientFinalNativeGrid(
        NativeRect region,
        AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) override;
    AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>> queryTransientFinalNativeGrid(
        NativeRect region, AuthorityRequestPriority priority,
        ProtectedHandoffEpoch epoch) override;
    AuthorityResult<CoarseSpawnGrid> queryCoarseSpawnGrid(
        CoarseSpawnRegion region,
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPAWN) override;
    [[nodiscard]] TerrainAuthorityCacheMetrics cacheMetrics() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class GenerationFailureException final : public std::runtime_error {
public:
    GenerationFailureException(AuthorityStatus status, GenerationFailure failure);

    [[nodiscard]] AuthorityStatus status() const noexcept { return status_; }
    [[nodiscard]] const GenerationFailure& failure() const noexcept { return failure_; }

private:
    AuthorityStatus status_;
    GenerationFailure failure_;
};

struct WorldGenerationMetrics {
    AuthorityQuality quality = AuthorityQuality::FINAL;
    AuthorityRequestPriority requestPriority = AuthorityRequestPriority::EXPLORATION_EXACT;
    Sha256Digest generationFingerprint{};
    uint64_t queries = 0;
    uint64_t requestedSamples = 0;
    uint64_t readyQueries = 0;
    uint64_t deferredQueries = 0;
    uint64_t failedQueries = 0;
    TerrainAuthorityCacheMetrics authorityCache;
};

// Shared immutable generation identity and learned authority for every v4
// consumer. The first production failure is latched so callers cannot fall
// back to a different generator after publishing part of a world.
class WorldGenerationContext {
public:
    WorldGenerationContext(GenerationIdentity identity, std::shared_ptr<TerrainAuthority> authority,
                           AuthorityQuality quality,
                           std::filesystem::path hydrologyAuthorityRoot = {});
    WorldGenerationContext(GenerationIdentity identity, std::shared_ptr<TerrainAuthority> authority,
                           AuthorityQuality quality, AuthorityRequestPriority requestPriority,
                           std::filesystem::path hydrologyAuthorityRoot = {});
    ~WorldGenerationContext();

    WorldGenerationContext(const WorldGenerationContext&) = delete;
    WorldGenerationContext& operator=(const WorldGenerationContext&) = delete;

    [[nodiscard]] const GenerationIdentity& identity() const noexcept;
    [[nodiscard]] const Sha256Digest& fingerprint() const noexcept;
    [[nodiscard]] AuthorityQuality quality() const noexcept;
    [[nodiscard]] AuthorityRequestPriority requestPriority() const noexcept;
    [[nodiscard]] const std::filesystem::path& hydrologyAuthorityRoot() const noexcept;
    [[nodiscard]] std::shared_ptr<worldgen::NativeHydrologyRouter>
    nativeHydrologyRouter() const noexcept;
    [[nodiscard]] std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
    nativeHydrologyIdentityRegistry() const noexcept;
    // Records that one FINAL native-hydrology owner has been built or
    // semantically loaded through the shared router. Derived priority
    // contexts share this process-local proof. It is never serialized as a
    // substitute for the fingerprinted hydrology authority itself.
    void recordPreparedNativeHydrologyOwner(int64_t ownerPageX, int64_t ownerPageZ) const;
    [[nodiscard]] bool nativeHydrologyOwnerPrepared(int64_t ownerPageX, int64_t ownerPageZ) const;
    [[nodiscard]] size_t preparedNativeHydrologyOwnerCount() const;
    [[nodiscard]] std::shared_ptr<WorldGenerationContext>
    withQuality(AuthorityQuality quality) const;
    [[nodiscard]] std::shared_ptr<WorldGenerationContext>
    withRequestPriority(AuthorityRequestPriority priority) const;

    // Enqueues one immutable authority page. Startup uses the SPAWN lane for
    // every page required by the native hydrology owner and its apron before
    // exact workers begin. Requests are nonblocking and do not change this
    // context's later default lane.
    AuthorityResult<bool>
    requestAuthorityPage(TerrainPageCoordinate coordinate,
                         AuthorityRequestPriority priority = AuthorityRequestPriority::SPAWN,
                         ProtectedHandoffEpoch epoch = {}) const;
    // Returns one immutable quantized page to a bounded exact consumer after
    // validating its key, payload, seed, and complete generation fingerprint.
    // Column plans copy their compact native stencil before releasing this
    // handle, so page lifetime remains inside the authority cache budget.
    AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    retainAuthorityPage(TerrainPageCoordinate coordinate,
                        std::optional<AuthorityRequestPriority> priority = std::nullopt) const;
    // Atomically admits a bounded closure to the underlying production cache.
    // The supplied coordinates retain their context quality and are consumed
    // in lexical order after duplicate removal. Deferred means at least one
    // page is still queued, building, or waiting for queue capacity. Only a
    // ready result proves the complete closure is cached and published.
    AuthorityResult<bool> requestAuthorityPages(
        std::span<const TerrainPageCoordinate> coordinates,
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPAWN,
        ProtectedHandoffEpoch epoch = {}) const;
    // Enqueues the one to four pages that support scale-four bilinear
    // reconstruction at a block coordinate.
    AuthorityResult<bool>
    requestWorldPage(int64_t worldX, int64_t worldZ,
                     AuthorityRequestPriority priority = AuthorityRequestPriority::SPAWN) const;
    AuthorityResult<PhysicalTerrainGrid>
    queryNative(NativeRect region,
                std::optional<AuthorityRequestPriority> priority = std::nullopt) const;
    AuthorityResult<std::vector<PhysicalTerrainSample>>
    queryNativePoints(std::span<const NativePoint> points,
                      std::optional<AuthorityRequestPriority> priority = std::nullopt) const;
    AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>> queryTransientFinalNativeGrid(
        NativeRect region, std::optional<AuthorityRequestPriority> priority = std::nullopt,
        ProtectedHandoffEpoch epoch = {}) const;
    AuthorityResult<CoarseSpawnGrid>
    queryCoarseSpawnGrid(CoarseSpawnRegion region,
                         std::optional<AuthorityRequestPriority> priority = std::nullopt) const;
    AuthorityResult<std::vector<PhysicalTerrainSample>>
    queryWorldPoints(std::span<const WorldBlockPoint> points,
                     std::optional<AuthorityRequestPriority> priority = std::nullopt) const;
    AuthorityResult<PhysicalTerrainSample>
    sampleWorld(int64_t worldX, int64_t worldZ,
                std::optional<AuthorityRequestPriority> priority = std::nullopt) const;

    void latchFailure(GenerationFailure failure) const;
    bool clearRetriableFailure() const;
    [[nodiscard]] std::optional<GenerationFailure> failure() const;
    [[nodiscard]] WorldGenerationMetrics metrics() const;

private:
    class FailureLatch;
    class HydrologyPreparationRegistry;
    class Impl;
    class MakeSharedEnabler;
    WorldGenerationContext(GenerationIdentity identity, std::shared_ptr<TerrainAuthority> authority,
                           AuthorityQuality quality, AuthorityRequestPriority requestPriority,
                           std::shared_ptr<FailureLatch> failureLatch,
                           std::shared_ptr<HydrologyPreparationRegistry> hydrologyPreparation,
                           std::filesystem::path hydrologyAuthorityRoot,
                           std::shared_ptr<worldgen::NativeHydrologyRouter> nativeHydrologyRouter,
                           std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
                               nativeHydrologyIdentityRegistry);
    std::unique_ptr<Impl> impl_;
};

} // namespace worldgen::learned
