#pragma once

#include "world/learned_terrain.hpp"
#include "world/terrain_bootstrap.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace worldgen::runtime {

inline constexpr std::string_view ONNX_RUNTIME_VERSION = "1.27.1";
inline constexpr std::string_view ONNX_RUNTIME_ARCHIVE = "onnxruntime-osx-arm64-1.27.1.tgz";
inline constexpr std::string_view ONNX_RUNTIME_DIRECTORY = "onnxruntime-osx-arm64-1.27.1";
inline constexpr std::string_view ONNX_RUNTIME_DYLIB = "lib/libonnxruntime.1.27.1.dylib";
inline constexpr std::string_view CORE_ML_CACHE_DIRECTORY =
    "coreml-cache-v3-base4-decoder4x256";
inline constexpr std::string_view CORE_ML_PROVIDER_NAME = "CoreML";
struct CoreMlProviderOption {
    std::string_view key;
    std::string_view value;
};
inline constexpr std::array<CoreMlProviderOption, 4> CORE_ML_PINNED_PROVIDER_OPTIONS{{
    {.key = "ModelFormat", .value = "MLProgram"},
    {.key = "MLComputeUnits", .value = "ALL"},
    {.key = "RequireStaticInputShapes", .value = "1"},
    {.key = "EnableOnSubgraphs", .value = "0"},
}};
inline constexpr uint32_t QUALIFICATION_SCHEMA_VERSION = 2;
inline constexpr double QUALIFICATION_QUANTIZATION_SCALE = 256.0;
inline constexpr uint32_t MAXIMUM_CPU_FALLBACK_THREADS = 16;
inline constexpr uint32_t GLOBAL_INTER_OP_THREAD_COUNT = 1;

// The baseline is recorded by the explicit local real-model qualification
// suite. Production refuses startup if the three pinned graphs do not produce
// this digest after provider-stable quantization.
inline constexpr std::string_view CANONICAL_QUALIFICATION_HASH =
    "6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df";

enum class TerrainRuntimeModel : uint8_t {
    Coarse = 0,
    Base = 1,
    Decoder = 2,
};

inline constexpr std::array<int64_t, 4> COARSE_MODEL_OUTPUT_SHAPE{1, 6, 64, 64};
inline constexpr std::array<int64_t, 4> BASE_MODEL_OUTPUT_SHAPE{4, 5, 64, 64};
inline constexpr std::array<int64_t, 4> DECODER_MODEL_OUTPUT_SHAPE{4, 1, 256, 256};

struct TerrainRuntimeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;

    [[nodiscard]] bool valid() const noexcept;
};

struct TerrainRuntimeInferenceResult {
    bool succeeded = false;
    TerrainRuntimeTensor output;
    std::string message;
};

struct TerrainQualificationOutput {
    TerrainRuntimeModel model = TerrainRuntimeModel::Coarse;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

[[nodiscard]] std::vector<TerrainRuntimeTensor>
terrainQualificationInputs(TerrainRuntimeModel model);
[[nodiscard]] learned::Sha256Digest
quantizedQualificationDigest(std::span<const TerrainQualificationOutput> outputs);
[[nodiscard]] bool qualificationDigestMatches(const learned::Sha256Digest& digest,
                                              std::string_view expectedHex);
[[nodiscard]] learned::GenerationIdentity productionGenerationIdentity(uint64_t seed);
// ONNX Runtime counts the calling thread as an intra-op worker. Keep the shared
// CPU fallback pool within the physical-core budget even when all three static
// sessions stay resident.
[[nodiscard]] uint32_t boundedCpuFallbackThreadCount(uint32_t physicalCpuCount) noexcept;

// All resident ONNX Runtime sessions share this one environment-level pool. The
// inference coordinator permits one graph call at a time, so the pool cannot
// multiply CPU fallback workers when the coarse, base, and decoder sessions exist
// together.
struct TerrainRuntimeThreadPoolConfiguration {
    uint32_t globalIntraOpThreads = 1;
    uint32_t globalInterOpThreads = GLOBAL_INTER_OP_THREAD_COUNT;
    bool usesGlobalThreadPool = true;
    bool disablesPerSessionThreads = true;
};

// The verified ONNX Runtime image remains mapped until process exit because
// its ONNX operator registries own static objects that outlive any one runtime
// instance. Sessions and environments still have deterministic instance
// lifetimes. These counters make retry and teardown behavior observable without
// exposing the native loader handle.
struct TerrainRuntimeLibraryMetrics {
    uint64_t loadAttempts = 0;
    uint64_t successfulLoads = 0;
    uint64_t reuseCount = 0;
    bool resident = false;
};

[[nodiscard]] TerrainRuntimeLibraryMetrics terrainRuntimeLibraryMetrics();

// Inference attribution is diagnostic only and never participates in the
// generation fingerprint. The authority coordinator still owns one active
// model call, while each call records the reason that admitted its window.
enum class TerrainRuntimeInferencePhase : uint8_t {
    Qualification = 0,
    DrySpawnCoarseSearch,
    FinalSpawnCertification,
    HorizonPreview,
    ProtectedFinal,
    Other,
    ExplorationExact,
    VisibleFinalRefinement,
    Count,
};

inline constexpr size_t TERRAIN_RUNTIME_INFERENCE_PHASE_COUNT =
    static_cast<size_t>(TerrainRuntimeInferencePhase::Count);

[[nodiscard]] constexpr TerrainRuntimeThreadPoolConfiguration
terrainRuntimeThreadPoolConfiguration(uint32_t physicalCpuCount) noexcept {
    return {
        .globalIntraOpThreads =
            physicalCpuCount == 0 ? 1U : std::min(physicalCpuCount, MAXIMUM_CPU_FALLBACK_THREADS),
        .globalInterOpThreads = GLOBAL_INTER_OP_THREAD_COUNT,
        .usesGlobalThreadPool = true,
        .disablesPerSessionThreads = true,
    };
}

struct TerrainRuntimeMetrics {
    struct Model {
        uint64_t sessionCreations = 0;
        uint64_t sessionCreationNanoseconds = 0;
        uint64_t calls = 0;
        uint64_t inferenceNanoseconds = 0;
        uint64_t maximumInferenceNanoseconds = 0;
    };

    struct Phase {
        uint64_t calls = 0;
        uint64_t inferenceNanoseconds = 0;
        std::array<uint64_t, 3> modelCalls{};
    };

    uint64_t compiledSessions = 0;
    uint32_t residentSessions = 0;
    uint32_t peakResidentSessions = 0;
    uint32_t globalIntraOpThreads = 0;
    uint32_t globalInterOpThreads = 0;
    bool usesGlobalThreadPool = false;
    bool perSessionThreadsDisabled = false;
    uint32_t cpuFallbackIntraOpThreads = 0;
    uint64_t inferenceCalls = 0;
    uint64_t inferenceFailures = 0;
    uint64_t coreMlPartitions = 0;
    uint64_t coreMlNodes = 0;
    uint64_t cpuFallbackPartitions = 0;
    uint64_t cpuFallbackNodes = 0;
    uint64_t otherPartitions = 0;
    uint64_t otherNodes = 0;
    uint32_t activeInferenceCalls = 0;
    uint32_t queuedInferenceCalls = 0;
    uint32_t maximumConcurrentInferenceCalls = 0;
    std::array<Model, 3> models{};
    std::array<Phase, TERRAIN_RUNTIME_INFERENCE_PHASE_COUNT> phases{};
    std::optional<learned::Sha256Digest> qualificationDigest;
};

class TerrainModelExecutor {
public:
    virtual ~TerrainModelExecutor() = default;
    virtual TerrainRuntimeInferenceResult
    runModel(TerrainRuntimeModel model, std::span<const TerrainRuntimeTensor> inputs,
             TerrainRuntimeInferencePhase phase = TerrainRuntimeInferencePhase::Other,
             const bootstrap::TerrainBootstrapCancellation* cancellation = nullptr) = 0;
};

class ProductionTerrainRuntime final : public bootstrap::TerrainRuntimePreparation {
public:
    explicit ProductionTerrainRuntime(uint64_t seed, std::string expectedQualificationHash =
                                                         std::string(CANONICAL_QUALIFICATION_HASH));
    ~ProductionTerrainRuntime() override;

    ProductionTerrainRuntime(const ProductionTerrainRuntime&) = delete;
    ProductionTerrainRuntime& operator=(const ProductionTerrainRuntime&) = delete;

    bootstrap::TerrainRuntimeStepResult qualifyPlatform() override;
    bootstrap::TerrainRuntimeStepResult
    compile(const std::filesystem::path& installedPack,
            const bootstrap::TerrainBootstrapCancellation& cancellation) override;
    bootstrap::TerrainRuntimeStepResult
    loadAndQualify(const std::filesystem::path& installedPack,
                   const bootstrap::TerrainBootstrapCancellation& cancellation) override;
    bootstrap::TerrainRuntimeStepResult
    bindWorldProfile(const std::filesystem::path& worldPath) override;
    [[nodiscard]] std::optional<std::string> qualifiedGenerationFingerprint() const override;
    [[nodiscard]] std::shared_ptr<learned::WorldGenerationContext>
    qualifiedGenerationContext() const override;

    TerrainRuntimeInferenceResult
    runModel(TerrainRuntimeModel model, std::span<const TerrainRuntimeTensor> inputs,
             TerrainRuntimeInferencePhase phase = TerrainRuntimeInferencePhase::Other,
             const bootstrap::TerrainBootstrapCancellation* cancellation = nullptr);

    [[nodiscard]] const learned::GenerationIdentity& generationIdentity() const noexcept;
    [[nodiscard]] TerrainRuntimeMetrics metrics() const;
    [[nodiscard]] std::filesystem::path runtimeDylibPath() const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
    mutable std::mutex contextMutex_;
    std::shared_ptr<learned::WorldGenerationContext> generationContext_;
};

[[nodiscard]] std::unique_ptr<bootstrap::TerrainRuntimePreparation>
makeProductionTerrainRuntime(uint64_t seed);

} // namespace worldgen::runtime
