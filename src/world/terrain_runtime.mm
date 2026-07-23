#import <Foundation/Foundation.h>

#include "world/terrain_runtime.hpp"

#include "world/infinite_diffusion_backend.hpp"
#include "world/onnxruntime_c_api_v27.hpp"
#include "world/save_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <spawn.h>
#include <string>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

extern char** environ;

namespace worldgen::runtime {
namespace {

using namespace ort_v27;

constexpr size_t MAXIMUM_INFERENCE_FLOATS = 64ULL * 1'024ULL * 1'024ULL;
constexpr std::array<TerrainRuntimeModel, 3> MODELS{
    TerrainRuntimeModel::Coarse,
    TerrainRuntimeModel::Base,
    TerrainRuntimeModel::Decoder,
};

std::string_view modelFileName(TerrainRuntimeModel model) {
    switch (model) {
        case TerrainRuntimeModel::Coarse:
            return "coarse_model.onnx";
        case TerrainRuntimeModel::Base:
            return "base_model.onnx";
        case TerrainRuntimeModel::Decoder:
            return "decoder_model.onnx";
    }
    return {};
}

std::string_view modelCacheName(TerrainRuntimeModel model) {
    switch (model) {
        case TerrainRuntimeModel::Coarse:
            return "coarse";
        case TerrainRuntimeModel::Base:
            return "base";
        case TerrainRuntimeModel::Decoder:
            return "decoder";
    }
    return {};
}

void appendU32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void appendU64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

int32_t quantizeQualificationValue(float value) {
    if (std::isnan(value))
        return std::numeric_limits<int32_t>::min();
    if (value == std::numeric_limits<float>::infinity())
        return std::numeric_limits<int32_t>::max();
    if (value == -std::numeric_limits<float>::infinity())
        return std::numeric_limits<int32_t>::min() + 1;
    const double scaled = std::round(static_cast<double>(value) * QUALIFICATION_QUANTIZATION_SCALE);
    return static_cast<int32_t>(
        std::clamp(scaled, static_cast<double>(std::numeric_limits<int32_t>::min() + 2),
                   static_cast<double>(std::numeric_limits<int32_t>::max() - 1)));
}

std::optional<size_t> tensorElementCount(std::span<const int64_t> shape) {
    if (shape.empty())
        return std::nullopt;
    size_t count = 1;
    for (int64_t dimension : shape) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) >
                                  static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return std::nullopt;
        }
        const size_t value = static_cast<size_t>(dimension);
        if (count > std::numeric_limits<size_t>::max() / value)
            return std::nullopt;
        count *= value;
    }
    return count;
}

std::vector<float> qualificationValues(size_t count, uint64_t seed) {
    std::vector<float> values(count);
    uint64_t state = seed;
    for (float& value : values) {
        state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
        const uint32_t mantissa = static_cast<uint32_t>(state >> 40U);
        value = (static_cast<float>(mantissa) / 16'777'216.0F - 0.5F) * 0.25F;
    }
    return values;
}

std::vector<TerrainRuntimeTensor> canonicalInputs(TerrainRuntimeModel model) {
    switch (model) {
        case TerrainRuntimeModel::Coarse:
            return {
                {.name = "x",
                 .shape = {1, 11, 64, 64},
                 .values = qualificationValues(11ULL * 64 * 64, 0x434F41525345ULL)},
                {.name = "noise_labels", .shape = {1}, .values = {0.125F}},
                {.name = "cond_0", .shape = {1}, .values = {-0.25F}},
                {.name = "cond_1", .shape = {1}, .values = {-0.125F}},
                {.name = "cond_2", .shape = {1}, .values = {0.0F}},
                {.name = "cond_3", .shape = {1}, .values = {0.125F}},
                {.name = "cond_4", .shape = {1}, .values = {0.25F}},
            };
        case TerrainRuntimeModel::Base:
            return {
                {.name = "x",
                 .shape = {4, 5, 64, 64},
                 .values = qualificationValues(4ULL * 5 * 64 * 64, 0x42415345ULL)},
                {.name = "noise_labels", .shape = {4}, .values = {0.35F, 0.35F, 0.35F, 0.35F}},
                {.name = "cond_0",
                 .shape = {4, 58},
                 .values = qualificationValues(4ULL * 58, 0x434F4E4430ULL)},
            };
        case TerrainRuntimeModel::Decoder:
            return {
                {.name = "x",
                 .shape = {4, 5, 256, 256},
                 .values = qualificationValues(4ULL * 5 * 256 * 256, 0x4445434F444552ULL)},
                {.name = "noise_labels",
                 .shape = {4},
                 .values = {1.564546F, 1.564546F, 1.564546F, 1.564546F}},
            };
    }
    return {};
}

std::filesystem::path uniqueSibling(const std::filesystem::path& parent, std::string_view suffix) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return parent /
           (std::string(suffix) + "." + std::to_string(::getpid()) + "." + std::to_string(ticks));
}

struct DirectoryCleanup {
    std::filesystem::path path;
    bool keep = false;

    ~DirectoryCleanup() {
        if (keep || path.empty())
            return;
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::optional<std::string>
extractRuntimeArchive(const std::filesystem::path& archive,
                      const std::filesystem::path& destination,
                      const bootstrap::TerrainBootstrapCancellation& cancellation) {
    std::error_code error;
    const std::filesystem::path expected =
        destination / ONNX_RUNTIME_DIRECTORY / ONNX_RUNTIME_DYLIB;
    if (std::filesystem::is_regular_file(expected, error) && !error)
        return std::nullopt;
    error.clear();

    const std::filesystem::path parent = destination.parent_path();
    std::filesystem::create_directories(parent, error);
    if (error)
        return "Could not create the terrain runtime directory: " + error.message();

    const std::filesystem::path staging = uniqueSibling(parent, ".runtime-staging");
    if (!std::filesystem::create_directory(staging, error) || error)
        return "Could not create the terrain runtime staging directory: " + error.message();
    DirectoryCleanup stagingCleanup{staging};

    const std::string archiveString = archive.string();
    const std::string stagingString = staging.string();
    std::array<char*, 6> arguments{
        const_cast<char*>("/usr/bin/tar"),        const_cast<char*>("-xzf"),
        const_cast<char*>(archiveString.c_str()), const_cast<char*>("-C"),
        const_cast<char*>(stagingString.c_str()), nullptr,
    };
    pid_t process = 0;
    const int spawnError =
        ::posix_spawn(&process, arguments[0], nullptr, nullptr, arguments.data(), environ);
    if (spawnError != 0)
        return "Could not start the verified terrain runtime extractor: " +
               std::error_code(spawnError, std::generic_category()).message();

    int processStatus = 0;
    while (true) {
        const pid_t waited = ::waitpid(process, &processStatus, WNOHANG);
        if (waited == process)
            break;
        if (waited < 0)
            return "Could not wait for terrain runtime extraction: " +
                   std::error_code(errno, std::generic_category()).message();
        if (cancellation.canceled()) {
            ::kill(process, SIGTERM);
            while (::waitpid(process, &processStatus, 0) < 0 && errno == EINTR) {
            }
            return "Terrain runtime extraction canceled";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!WIFEXITED(processStatus) || WEXITSTATUS(processStatus) != 0)
        return "The verified terrain runtime archive could not be extracted";

    const std::filesystem::path stagedDylib = staging / ONNX_RUNTIME_DIRECTORY / ONNX_RUNTIME_DYLIB;
    if (!std::filesystem::is_regular_file(stagedDylib, error) || error)
        return "The verified terrain runtime archive has an incompatible layout";

    std::optional<std::filesystem::path> replaced;
    if (std::filesystem::exists(destination, error) && !error) {
        replaced = uniqueSibling(parent, ".runtime-replaced");
        std::filesystem::rename(destination, *replaced, error);
        if (error)
            return "Could not replace the invalid terrain runtime: " + error.message();
    }
    error.clear();
    std::filesystem::rename(staging, destination, error);
    if (error) {
        if (replaced) {
            std::error_code restoreError;
            std::filesystem::rename(*replaced, destination, restoreError);
        }
        return "Could not publish the terrain runtime: " + error.message();
    }
    stagingCleanup.keep = true;
    if (replaced) {
        std::error_code removeError;
        std::filesystem::remove_all(*replaced, removeError);
    }
    return std::nullopt;
}

std::string dynamicLoaderError() {
    const char* error = ::dlerror();
    return error == nullptr ? "unknown dynamic loader error" : std::string(error);
}

struct RuntimeLibraryAcquisition {
    const void* api = nullptr;
    std::optional<std::string> error;
};

class ProcessLifetimeRuntimeLibrary {
public:
    RuntimeLibraryAcquisition acquire(const std::filesystem::path& path) {
        std::error_code filesystemError;
        if (!std::filesystem::is_regular_file(path, filesystemError) || filesystemError) {
            return {.error = "The extracted ONNX Runtime library is missing"};
        }

        std::lock_guard lock(mutex_);
        if (api_ != nullptr) {
            ++metrics_.reuseCount;
            return {.api = api_};
        }
        if (library_ != nullptr) {
            return {.error = terminalError_.value_or(
                        "The resident ONNX Runtime library is incompatible")};
        }

        ++metrics_.loadAttempts;
        ::dlerror();
        library_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library_ == nullptr) {
            return {.error = "Could not load the verified ONNX Runtime library: " +
                             dynamicLoaderError()};
        }
        // Once dlopen succeeds, ONNX may have registered process static
        // operator schemas. Never call dlclose on this handle, including an
        // incompatibility path. A failed validation poisons this process-local
        // acquisition instead of loading a second image with another registry.
        ::dlerror();
        const void* symbol = ::dlsym(library_, "OrtGetApiBase");
        if (const char* loaderError = ::dlerror(); loaderError != nullptr || symbol == nullptr) {
            const std::string detail =
                loaderError == nullptr ? "OrtGetApiBase is missing" : loaderError;
            return terminalFailure("The verified ONNX Runtime library is incompatible: " + detail);
        }
        OrtGetApiBaseFunction getApiBase = nullptr;
        static_assert(sizeof(getApiBase) == sizeof(symbol));
        std::memcpy(&getApiBase, &symbol, sizeof(getApiBase));
        const OrtApiBase* base = getApiBase();
        if (base == nullptr || base->GetVersionString == nullptr || base->GetApi == nullptr) {
            return terminalFailure("The verified ONNX Runtime library did not provide its C API");
        }
        const char* version = base->GetVersionString();
        if (version == nullptr || std::string_view(version) != ONNX_RUNTIME_VERSION) {
            const std::string actual = version == nullptr ? "unknown" : version;
            return terminalFailure("ONNX Runtime version mismatch: expected 1.27.1, found " +
                                   actual);
        }
        api_ = base->GetApi(API_VERSION);
        if (api_ == nullptr)
            return terminalFailure("ONNX Runtime did not provide C API version 27");
        ++metrics_.successfulLoads;
        return {.api = api_};
    }

    TerrainRuntimeLibraryMetrics metrics() const {
        std::lock_guard lock(mutex_);
        TerrainRuntimeLibraryMetrics result = metrics_;
        result.resident = library_ != nullptr;
        return result;
    }

private:
    RuntimeLibraryAcquisition terminalFailure(std::string message) {
        terminalError_ = std::move(message);
        return {.error = *terminalError_};
    }

    mutable std::mutex mutex_;
    void* library_ = nullptr;
    const void* api_ = nullptr;
    std::optional<std::string> terminalError_;
    TerrainRuntimeLibraryMetrics metrics_;
};

ProcessLifetimeRuntimeLibrary& processLifetimeRuntimeLibrary() {
    // The raw loader handle intentionally has no closing owner. The manager
    // itself may destruct normally without calling dlclose, and process
    // termination reclaims the mapping.
    static ProcessLifetimeRuntimeLibrary owner;
    return owner;
}

bool providerNameIsCoreMl(std::string_view name) {
    return name == "CoreML" || name == "CoreMLExecutionProvider";
}

bool providerNameIsCpu(std::string_view name) {
    return name == "CPU" || name == "CPUExecutionProvider";
}

uint32_t detectedPhysicalCpuCount() noexcept {
#if defined(__APPLE__)
    int physicalCpuCount = 0;
    size_t byteCount = sizeof(physicalCpuCount);
    if (::sysctlbyname("hw.physicalcpu", &physicalCpuCount, &byteCount, nullptr, 0) == 0 &&
        physicalCpuCount > 0) {
        return static_cast<uint32_t>(physicalCpuCount);
    }
#endif
    return std::thread::hardware_concurrency();
}

} // namespace

bool TerrainRuntimeTensor::valid() const noexcept {
    const std::optional<size_t> count = tensorElementCount(shape);
    return !name.empty() && count && *count == values.size() &&
           values.size() <= MAXIMUM_INFERENCE_FLOATS;
}

std::vector<TerrainRuntimeTensor> terrainQualificationInputs(TerrainRuntimeModel model) {
    return canonicalInputs(model);
}

learned::Sha256Digest
quantizedQualificationDigest(std::span<const TerrainQualificationOutput> outputs) {
    std::vector<uint8_t> bytes;
    size_t valueCount = 0;
    for (const TerrainQualificationOutput& output : outputs)
        valueCount += output.values.size();
    bytes.reserve(16 + outputs.size() * 48 + valueCount * sizeof(int32_t));
    appendU32(bytes, QUALIFICATION_SCHEMA_VERSION);
    appendU32(bytes, static_cast<uint32_t>(QUALIFICATION_QUANTIZATION_SCALE));
    appendU32(bytes, static_cast<uint32_t>(outputs.size()));
    for (const TerrainQualificationOutput& output : outputs) {
        bytes.push_back(static_cast<uint8_t>(output.model));
        appendU32(bytes, static_cast<uint32_t>(output.shape.size()));
        for (int64_t dimension : output.shape)
            appendU64(bytes, static_cast<uint64_t>(dimension));
        appendU64(bytes, static_cast<uint64_t>(output.values.size()));
        for (float value : output.values)
            appendU32(bytes, static_cast<uint32_t>(quantizeQualificationValue(value)));
    }
    return learned::sha256(bytes);
}

bool qualificationDigestMatches(const learned::Sha256Digest& digest, std::string_view expectedHex) {
    const std::optional<learned::Sha256Digest> expected = learned::parseSha256(expectedHex);
    return expected && *expected == digest;
}

uint32_t boundedCpuFallbackThreadCount(uint32_t physicalCpuCount) noexcept {
    return terrainRuntimeThreadPoolConfiguration(physicalCpuCount).globalIntraOpThreads;
}

TerrainRuntimeLibraryMetrics terrainRuntimeLibraryMetrics() {
    return processLifetimeRuntimeLibrary().metrics();
}

learned::GenerationIdentity productionGenerationIdentity(uint64_t seed) {
    std::vector<uint8_t> modelBytes;
    learned::Sha256Digest runtimeHash{};
    bool runtimeAssetFound = false;
    bool pinsAreValid = true;
    for (const bootstrap::TerrainAssetSpec& asset : bootstrap::pinnedTerrainAssets()) {
        const std::optional<learned::Sha256Digest> digest = learned::parseSha256(asset.sha256);
        if (!digest) {
            pinsAreValid = false;
            continue;
        }
        if (asset.kind == bootstrap::TerrainAssetKind::Model) {
            modelBytes.insert(modelBytes.end(), digest->begin(), digest->end());
        } else if (asset.kind == bootstrap::TerrainAssetKind::Runtime) {
            if (runtimeAssetFound)
                pinsAreValid = false;
            runtimeHash = *digest;
            runtimeAssetFound = true;
        }
    }

    learned::GenerationIdentity identity;
    identity.seed = seed;
    if (pinsAreValid && !modelBytes.empty() && runtimeAssetFound) {
        identity.modelPackHash = learned::sha256(modelBytes);
        identity.runtimeHash = runtimeHash;
    }
    identity.provider = learned::GENERATOR_V4_PROVIDER_CONFIGURATION;
    return identity;
}

class ProductionTerrainRuntime::Impl final : public TerrainModelExecutor {
public:
    Impl(uint64_t seed, std::string expectedHash)
        : identity(productionGenerationIdentity(seed)),
          expectedQualificationHash(std::move(expectedHash)) {}

    ~Impl() { resetRuntime(); }

    using GetErrorMessageFunction = const char* (*)(const OrtStatus*) noexcept;
    using ReleaseStatusFunction = void (*)(OrtStatus*) noexcept;

    std::string consumeStatus(OrtStatus* status) const {
        if (status == nullptr)
            return {};
        std::string message = "ONNX Runtime reported an error";
        if (api != nullptr) {
            const auto getMessage =
                apiFunction<GetErrorMessageFunction>(api, ApiSlot::GetErrorMessage);
            if (getMessage != nullptr) {
                if (const char* text = getMessage(status); text != nullptr)
                    message = text;
            }
            const auto release = apiFunction<ReleaseStatusFunction>(api, ApiSlot::ReleaseStatus);
            if (release != nullptr)
                release(status);
        }
        return message;
    }

    void releaseSession(OrtSession*& session) const {
        if (session == nullptr || api == nullptr)
            return;
        using Function = void (*)(OrtSession*) noexcept;
        apiFunction<Function>(api, ApiSlot::ReleaseSession)(session);
        session = nullptr;
    }

    void updateResidentSessionMetrics() {
        uint32_t residentSessions = 0;
        for (const OrtSession* session : sessions)
            if (session != nullptr)
                ++residentSessions;
        std::lock_guard lock(metricsMutex);
        runtimeMetrics.residentSessions = residentSessions;
        runtimeMetrics.peakResidentSessions =
            std::max(runtimeMetrics.peakResidentSessions, residentSessions);
    }

    void releaseSessions() {
        for (OrtSession*& session : sessions)
            releaseSession(session);
        updateResidentSessionMetrics();
    }

    void setThreadPoolConfiguration(TerrainRuntimeThreadPoolConfiguration configuration) {
        threadPoolConfiguration = configuration;
        std::lock_guard lock(metricsMutex);
        runtimeMetrics.globalIntraOpThreads = configuration.globalIntraOpThreads;
        runtimeMetrics.globalInterOpThreads = configuration.globalInterOpThreads;
        runtimeMetrics.usesGlobalThreadPool = configuration.usesGlobalThreadPool;
        runtimeMetrics.perSessionThreadsDisabled = configuration.disablesPerSessionThreads;
        // Retain this metric as the legacy inspector field. CPU fallback work now
        // uses the shared environment pool rather than a pool per model session.
        runtimeMetrics.cpuFallbackIntraOpThreads = configuration.globalIntraOpThreads;
    }

    void resetRuntime() {
        releaseSessions();
        if (environment != nullptr && api != nullptr) {
            using Function = void (*)(OrtEnv*) noexcept;
            apiFunction<Function>(api, ApiSlot::ReleaseEnv)(environment);
            environment = nullptr;
        }
        api = nullptr;
        compiled = false;
        qualified = false;
    }

    std::optional<std::string> loadRuntimeLibrary() {
        if (api != nullptr && environment != nullptr)
            return std::nullopt;
        resetRuntime();
        RuntimeLibraryAcquisition acquisition = processLifetimeRuntimeLibrary().acquire(dylibPath);
        if (acquisition.error)
            return acquisition.error;
        api = acquisition.api;

        using CreateThreadingOptionsFunction = OrtStatus* (*)(OrtThreadingOptions**) noexcept;
        using SetGlobalThreadsFunction = OrtStatus* (*)(OrtThreadingOptions*, int) noexcept;
        using ReleaseThreadingOptionsFunction = void (*)(OrtThreadingOptions*) noexcept;
        using CreateEnvFunction = OrtStatus* (*)(LoggingLevel, const char*,
                                                 const OrtThreadingOptions*, OrtEnv**) noexcept;
        const auto createThreadingOptions =
            apiFunction<CreateThreadingOptionsFunction>(api, ApiSlot::CreateThreadingOptions);
        const auto setGlobalIntraOpThreads =
            apiFunction<SetGlobalThreadsFunction>(api, ApiSlot::SetGlobalIntraOpNumThreads);
        const auto setGlobalInterOpThreads =
            apiFunction<SetGlobalThreadsFunction>(api, ApiSlot::SetGlobalInterOpNumThreads);
        const auto releaseThreadingOptions =
            apiFunction<ReleaseThreadingOptionsFunction>(api, ApiSlot::ReleaseThreadingOptions);
        const auto createEnv =
            apiFunction<CreateEnvFunction>(api, ApiSlot::CreateEnvWithGlobalThreadPools);
        if (createThreadingOptions == nullptr || setGlobalIntraOpThreads == nullptr ||
            setGlobalInterOpThreads == nullptr || releaseThreadingOptions == nullptr ||
            createEnv == nullptr) {
            resetRuntime();
            return "The verified ONNX Runtime library is missing global thread-pool APIs";
        }

        OrtThreadingOptions* threadingOptions = nullptr;
        std::string error = consumeStatus(createThreadingOptions(&threadingOptions));
        if (!error.empty() || threadingOptions == nullptr) {
            if (threadingOptions != nullptr)
                releaseThreadingOptions(threadingOptions);
            resetRuntime();
            return "Could not create ONNX Runtime global thread-pool options" +
                   (error.empty() ? std::string{} : ": " + error);
        }
        error = consumeStatus(setGlobalIntraOpThreads(
            threadingOptions, static_cast<int>(threadPoolConfiguration.globalIntraOpThreads)));
        if (error.empty()) {
            error = consumeStatus(setGlobalInterOpThreads(
                threadingOptions, static_cast<int>(threadPoolConfiguration.globalInterOpThreads)));
        }
        if (!error.empty()) {
            releaseThreadingOptions(threadingOptions);
            resetRuntime();
            return "Could not configure ONNX Runtime global thread pools: " + error;
        }
        error = consumeStatus(
            createEnv(LoggingLevel::Error, "rycraft-terrain-v4", threadingOptions, &environment));
        releaseThreadingOptions(threadingOptions);
        if (!error.empty()) {
            resetRuntime();
            return "Could not create the ONNX Runtime environment: " + error;
        }
        return std::nullopt;
    }

    std::optional<std::string> requireCoreMlProvider() const {
        using GetFunction = OrtStatus* (*)(char***, int*) noexcept;
        using ReleaseFunction = OrtStatus* (*)(char**, int) noexcept;
        char** providers = nullptr;
        int providerCount = 0;
        const std::string error = consumeStatus(apiFunction<GetFunction>(
            api, ApiSlot::GetAvailableProviders)(&providers, &providerCount));
        if (!error.empty())
            return "Could not inspect ONNX Runtime providers: " + error;
        bool found = false;
        for (int index = 0; index < providerCount; ++index) {
            if (providers[index] != nullptr && providerNameIsCoreMl(providers[index]))
                found = true;
        }
        const std::string releaseError = consumeStatus(apiFunction<ReleaseFunction>(
            api, ApiSlot::ReleaseAvailableProviders)(providers, providerCount));
        if (!releaseError.empty())
            return "Could not release the ONNX Runtime provider list: " + releaseError;
        if (!found)
            return "The verified ONNX Runtime archive does not contain Core ML";
        return std::nullopt;
    }

    std::optional<std::string> configureOptions(OrtSessionOptions* options,
                                                TerrainRuntimeModel model) const {
        using SetExecutionFunction = OrtStatus* (*)(OrtSessionOptions*, ExecutionMode) noexcept;
        using SetOptimizationFunction =
            OrtStatus* (*)(OrtSessionOptions*, GraphOptimizationLevel) noexcept;
        using SetDeterminismFunction = OrtStatus* (*)(OrtSessionOptions*, bool) noexcept;
        using DisablePerSessionThreadsFunction = OrtStatus* (*)(OrtSessionOptions*) noexcept;
        using AddFreeDimensionOverrideFunction =
            OrtStatus* (*)(OrtSessionOptions*, const char*, int64_t) noexcept;
        using AddConfigFunction =
            OrtStatus* (*)(OrtSessionOptions*, const char*, const char*) noexcept;
        using AppendProviderFunction =
            OrtStatus* (*)(OrtSessionOptions*, const char*, const char* const*, const char* const*,
                           size_t) noexcept;

        auto checked = [this](OrtStatus* status,
                              std::string_view action) -> std::optional<std::string> {
            const std::string error = consumeStatus(status);
            if (error.empty())
                return std::nullopt;
            return std::string(action) + ": " + error;
        };
        if (auto error =
                checked(apiFunction<SetExecutionFunction>(api, ApiSlot::SetSessionExecutionMode)(
                            options, ExecutionMode::Sequential),
                        "Could not require sequential inference"))
            return error;
        // Every static model session must share the one environment-level pool.
        // The runtime serializes graph calls, preserving deterministic model
        // invocation while preventing three resident sessions from triplicating
        // CPU fallback workers.
        if (auto error = checked(apiFunction<DisablePerSessionThreadsFunction>(
                                     api, ApiSlot::DisablePerSessionThreads)(options),
                                 "Could not share the global ONNX Runtime thread pool"))
            return error;
        if (auto error = checked(apiFunction<SetOptimizationFunction>(
                                     api, ApiSlot::SetSessionGraphOptimizationLevel)(
                                     options, GraphOptimizationLevel::EnableAll),
                                 "Could not enable graph optimization"))
            return error;
        if (auto error = checked(apiFunction<SetDeterminismFunction>(
                                     api, ApiSlot::SetDeterministicCompute)(options, true),
                                 "Could not require deterministic compute"))
            return error;
        // The Base and Decoder graphs expose symbolic dimensions even though
        // the v4 window contract always invokes fixed shapes. Publish those
        // shapes during session construction so RequireStaticInputShapes can
        // partition the actual graph instead of retaining dynamic CPU paths.
        if (model == TerrainRuntimeModel::Base || model == TerrainRuntimeModel::Decoder) {
            static_assert(learned::LATENT_WINDOW.batchSize == 4);
            static_assert(learned::DECODER_WINDOW.batchSize == 4);
            static_assert(learned::DECODER_WINDOW.edge == 256);
            const auto addFreeDimensionOverride = apiFunction<AddFreeDimensionOverrideFunction>(
                api, ApiSlot::AddFreeDimensionOverrideByName);
            if (addFreeDimensionOverride == nullptr)
                return "The verified ONNX Runtime library cannot specialize terrain model shapes";
            const int64_t batch = model == TerrainRuntimeModel::Base
                                      ? learned::LATENT_WINDOW.batchSize
                                      : learned::DECODER_WINDOW.batchSize;
            if (auto error = checked(addFreeDimensionOverride(options, "batch", batch),
                                     "Could not specialize the fixed terrain model batch"))
                return error;
            if (model == TerrainRuntimeModel::Decoder) {
                if (auto error = checked(
                        addFreeDimensionOverride(options, "height", learned::DECODER_WINDOW.edge),
                        "Could not specialize the Decoder height"))
                    return error;
                if (auto error = checked(
                        addFreeDimensionOverride(options, "width", learned::DECODER_WINDOW.edge),
                        "Could not specialize the Decoder width"))
                    return error;
            }
        }
        if (auto error =
                checked(apiFunction<AddConfigFunction>(api, ApiSlot::AddSessionConfigEntry)(
                            options, "session.record_ep_graph_assignment_info", "1"),
                        "Could not enable provider partition metrics"))
            return error;

        const std::filesystem::path modelCache = cachePath / modelCacheName(model);
        std::error_code filesystemError;
        std::filesystem::create_directories(modelCache, filesystemError);
        if (filesystemError)
            return "Could not create the Core ML model cache: " + filesystemError.message();
        const std::string modelCacheString = modelCache.string();
        const std::array<const char*, 6> keys{
            CORE_ML_PINNED_PROVIDER_OPTIONS[0].key.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[1].key.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[2].key.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[3].key.data(),
            "ModelCacheDirectory",
            "ProfileComputePlan",
        };
        const std::array<const char*, 6> values{
            CORE_ML_PINNED_PROVIDER_OPTIONS[0].value.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[1].value.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[2].value.data(),
            CORE_ML_PINNED_PROVIDER_OPTIONS[3].value.data(),
            modelCacheString.c_str(),
            "0",
        };
        if (auto error = checked(
                apiFunction<AppendProviderFunction>(api,
                                                    ApiSlot::SessionOptionsAppendExecutionProvider)(
                    options, CORE_ML_PROVIDER_NAME.data(), keys.data(), values.data(), keys.size()),
                "Could not configure the Core ML execution provider"))
            return error;
        return std::nullopt;
    }

    struct AssignmentCounts {
        uint64_t coreMlPartitions = 0;
        uint64_t coreMlNodes = 0;
        uint64_t cpuPartitions = 0;
        uint64_t cpuNodes = 0;
        uint64_t otherPartitions = 0;
        uint64_t otherNodes = 0;
    };

    std::optional<std::string> assignmentCounts(OrtSession* session,
                                                AssignmentCounts& counts) const {
        using GetAssignmentsFunction =
            OrtStatus* (*)(const OrtSession*, const OrtEpAssignedSubgraph* const**,
                           size_t*) noexcept;
        using GetNameFunction = OrtStatus* (*)(const OrtEpAssignedSubgraph*, const char**) noexcept;
        using GetNodesFunction = OrtStatus* (*)(const OrtEpAssignedSubgraph*,
                                                const OrtEpAssignedNode* const**, size_t*) noexcept;
        const OrtEpAssignedSubgraph* const* assignments = nullptr;
        size_t assignmentCount = 0;
        std::string error = consumeStatus(
            apiFunction<GetAssignmentsFunction>(api, ApiSlot::SessionGetEpGraphAssignmentInfo)(
                session, &assignments, &assignmentCount));
        if (!error.empty())
            return "Could not read provider partition metrics: " + error;
        for (size_t index = 0; index < assignmentCount; ++index) {
            const char* provider = nullptr;
            error = consumeStatus(apiFunction<GetNameFunction>(
                api, ApiSlot::EpAssignedSubgraphGetEpName)(assignments[index], &provider));
            if (!error.empty())
                return "Could not read a provider partition name: " + error;
            const OrtEpAssignedNode* const* nodes = nullptr;
            size_t nodeCount = 0;
            error = consumeStatus(apiFunction<GetNodesFunction>(
                api, ApiSlot::EpAssignedSubgraphGetNodes)(assignments[index], &nodes, &nodeCount));
            if (!error.empty())
                return "Could not read provider partition nodes: " + error;
            (void)nodes;
            const std::string_view name = provider == nullptr ? std::string_view{} : provider;
            if (providerNameIsCoreMl(name)) {
                ++counts.coreMlPartitions;
                counts.coreMlNodes += nodeCount;
            } else if (providerNameIsCpu(name)) {
                ++counts.cpuPartitions;
                counts.cpuNodes += nodeCount;
            } else {
                ++counts.otherPartitions;
                counts.otherNodes += nodeCount;
            }
        }
        if (counts.coreMlPartitions == 0 || counts.coreMlNodes == 0)
            return "Core ML did not receive any nodes from the pinned terrain model";
        return std::nullopt;
    }

    std::pair<OrtSession*, std::optional<std::string>> createSession(TerrainRuntimeModel model) {
        using CreateOptionsFunction = OrtStatus* (*)(OrtSessionOptions**) noexcept;
        using ReleaseOptionsFunction = void (*)(OrtSessionOptions*) noexcept;
        using CreateSessionFunction =
            OrtStatus* (*)(const OrtEnv*, const char*, const OrtSessionOptions*,
                           OrtSession**) noexcept;

        const auto started = std::chrono::steady_clock::now();
        OrtSessionOptions* options = nullptr;
        std::string error = consumeStatus(
            apiFunction<CreateOptionsFunction>(api, ApiSlot::CreateSessionOptions)(&options));
        if (!error.empty())
            return {nullptr, "Could not create terrain session options: " + error};
        const auto releaseOptions =
            apiFunction<ReleaseOptionsFunction>(api, ApiSlot::ReleaseSessionOptions);
        const std::optional<std::string> configurationError = configureOptions(options, model);
        if (configurationError) {
            releaseOptions(options);
            return {nullptr, configurationError};
        }

        const std::filesystem::path path = packPath / modelFileName(model);
        OrtSession* session = nullptr;
        error = consumeStatus(apiFunction<CreateSessionFunction>(api, ApiSlot::CreateSession)(
            environment, path.c_str(), options, &session));
        releaseOptions(options);
        if (!error.empty()) {
            if (session != nullptr)
                releaseSession(session);
            return {nullptr, "Could not load " + std::string(modelFileName(model)) + ": " + error};
        }

        AssignmentCounts counts;
        const std::optional<std::string> assignmentError = assignmentCounts(session, counts);
        if (assignmentError) {
            releaseSession(session);
            return {nullptr, assignmentError};
        }
        {
            std::lock_guard lock(metricsMutex);
            ++runtimeMetrics.compiledSessions;
            TerrainRuntimeMetrics::Model& modelMetrics =
                runtimeMetrics.models[static_cast<size_t>(model)];
            ++modelMetrics.sessionCreations;
            modelMetrics.sessionCreationNanoseconds +=
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
            runtimeMetrics.coreMlPartitions += counts.coreMlPartitions;
            runtimeMetrics.coreMlNodes += counts.coreMlNodes;
            runtimeMetrics.cpuFallbackPartitions += counts.cpuPartitions;
            runtimeMetrics.cpuFallbackNodes += counts.cpuNodes;
            runtimeMetrics.otherPartitions += counts.otherPartitions;
            runtimeMetrics.otherNodes += counts.otherNodes;
        }
        return {session, std::nullopt};
    }

    TerrainRuntimeInferenceResult
    runLocked(TerrainRuntimeModel model, std::span<const TerrainRuntimeTensor> inputs,
              const bootstrap::TerrainBootstrapCancellation* cancellation) {
        if (!compiled || api == nullptr || environment == nullptr)
            return {.message = "The terrain inference runtime is not compiled and loaded"};
        if (cancellation != nullptr && cancellation->canceled())
            return {.message = "Terrain inference was canceled before execution"};
        if (inputs.empty())
            return {.message = "Terrain inference requires at least one input"};

        size_t totalValues = 0;
        std::set<std::string_view> names;
        for (const TerrainRuntimeTensor& input : inputs) {
            if (!input.valid())
                return {.message = "Terrain inference received an invalid tensor"};
            if (!names.insert(input.name).second)
                return {.message = "Terrain inference input names must be unique"};
            if (totalValues > MAXIMUM_INFERENCE_FLOATS - input.values.size())
                return {.message = "Terrain inference input exceeds the bounded tensor budget"};
            totalValues += input.values.size();
        }
        OrtSession* const session = sessions[static_cast<size_t>(model)];
        if (session == nullptr)
            return {.message = "The pinned terrain model does not have a resident session"};

        using CountFunction = OrtStatus* (*)(const OrtSession*, size_t*) noexcept;
        size_t expectedInputs = 0;
        std::string error = consumeStatus(apiFunction<CountFunction>(
            api, ApiSlot::SessionGetInputCount)(session, &expectedInputs));
        if (!error.empty())
            return {.message = "Could not inspect model inputs: " + error};
        if (expectedInputs != inputs.size())
            return {.message = "Terrain inference input count does not match the pinned model"};
        size_t outputCount = 0;
        error = consumeStatus(
            apiFunction<CountFunction>(api, ApiSlot::SessionGetOutputCount)(session, &outputCount));
        if (!error.empty())
            return {.message = "Could not inspect model outputs: " + error};
        if (outputCount != 1)
            return {.message = "The pinned terrain model must have exactly one output"};

        using CreateMemoryFunction =
            OrtStatus* (*)(AllocatorType, MemoryType, OrtMemoryInfo**) noexcept;
        using ReleaseMemoryFunction = void (*)(OrtMemoryInfo*) noexcept;
        using CreateTensorFunction =
            OrtStatus* (*)(const OrtMemoryInfo*, void*, size_t, const int64_t*, size_t,
                           TensorElementDataType, OrtValue**) noexcept;
        using ReleaseValueFunction = void (*)(OrtValue*) noexcept;
        using AllocatorFunction = OrtStatus* (*)(OrtAllocator**) noexcept;
        using GetNameFunction =
            OrtStatus* (*)(const OrtSession*, size_t, OrtAllocator*, char**) noexcept;
        using AllocatorFreeFunction = OrtStatus* (*)(OrtAllocator*, void*) noexcept;
        using RunFunction = OrtStatus* (*)(OrtSession*, const OrtRunOptions*, const char* const*,
                                           const OrtValue* const*, size_t, const char* const*,
                                           size_t, OrtValue**) noexcept;

        OrtMemoryInfo* memory = nullptr;
        error = consumeStatus(apiFunction<CreateMemoryFunction>(api, ApiSlot::CreateCpuMemoryInfo)(
            AllocatorType::Arena, MemoryType::Default, &memory));
        if (!error.empty())
            return {.message = "Could not create CPU tensor memory: " + error};
        const auto releaseMemory =
            apiFunction<ReleaseMemoryFunction>(api, ApiSlot::ReleaseMemoryInfo);
        const auto releaseValue = apiFunction<ReleaseValueFunction>(api, ApiSlot::ReleaseValue);

        std::vector<OrtValue*> values(inputs.size(), nullptr);
        std::vector<const char*> inputNames;
        inputNames.reserve(inputs.size());
        for (size_t index = 0; index < inputs.size(); ++index) {
            const TerrainRuntimeTensor& input = inputs[index];
            inputNames.push_back(input.name.c_str());
            error = consumeStatus(
                apiFunction<CreateTensorFunction>(api, ApiSlot::CreateTensorWithDataAsOrtValue)(
                    memory, const_cast<float*>(input.values.data()),
                    input.values.size() * sizeof(float), input.shape.data(), input.shape.size(),
                    TensorElementDataType::Float, &values[index]));
            if (!error.empty()) {
                for (OrtValue* value : values)
                    if (value != nullptr)
                        releaseValue(value);
                releaseMemory(memory);
                return {.message = "Could not create an input tensor: " + error};
            }
        }

        OrtAllocator* allocator = nullptr;
        error = consumeStatus(apiFunction<AllocatorFunction>(
            api, ApiSlot::GetAllocatorWithDefaultOptions)(&allocator));
        if (!error.empty()) {
            for (OrtValue* value : values)
                releaseValue(value);
            releaseMemory(memory);
            return {.message = "Could not access the ONNX Runtime allocator: " + error};
        }
        char* outputName = nullptr;
        error = consumeStatus(apiFunction<GetNameFunction>(api, ApiSlot::SessionGetOutputName)(
            session, 0, allocator, &outputName));
        if (!error.empty()) {
            for (OrtValue* value : values)
                releaseValue(value);
            releaseMemory(memory);
            return {.message = "Could not read the model output name: " + error};
        }

        OrtValue* output = nullptr;
        const std::array<const char*, 1> outputNames{outputName};
        error = consumeStatus(apiFunction<RunFunction>(api, ApiSlot::Run)(
            session, nullptr, inputNames.data(), const_cast<const OrtValue* const*>(values.data()),
            values.size(), outputNames.data(), outputNames.size(), &output));
        const std::string freeError = consumeStatus(
            apiFunction<AllocatorFreeFunction>(api, ApiSlot::AllocatorFree)(allocator, outputName));
        for (OrtValue* value : values)
            releaseValue(value);
        releaseMemory(memory);
        if (!error.empty()) {
            if (output != nullptr)
                releaseValue(output);
            return {.message = "Terrain model inference failed: " + error};
        }
        if (!freeError.empty()) {
            releaseValue(output);
            return {.message = "Could not release the model output name: " + freeError};
        }

        using GetShapeFunction =
            OrtStatus* (*)(const OrtValue*, OrtTensorTypeAndShapeInfo**) noexcept;
        using ReleaseShapeFunction = void (*)(OrtTensorTypeAndShapeInfo*) noexcept;
        using GetTypeFunction =
            OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, TensorElementDataType*) noexcept;
        using GetDimensionCountFunction =
            OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, size_t*) noexcept;
        using GetDimensionsFunction =
            OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, int64_t*, size_t) noexcept;
        using GetElementCountFunction =
            OrtStatus* (*)(const OrtTensorTypeAndShapeInfo*, size_t*) noexcept;
        using GetDataFunction = OrtStatus* (*)(OrtValue*, void**) noexcept;

        OrtTensorTypeAndShapeInfo* shapeInfo = nullptr;
        error = consumeStatus(
            apiFunction<GetShapeFunction>(api, ApiSlot::GetTensorTypeAndShape)(output, &shapeInfo));
        if (!error.empty()) {
            releaseValue(output);
            return {.message = "Could not inspect the model output tensor: " + error};
        }
        const auto releaseShape =
            apiFunction<ReleaseShapeFunction>(api, ApiSlot::ReleaseTensorTypeAndShapeInfo);
        TensorElementDataType elementType = TensorElementDataType::Undefined;
        error = consumeStatus(apiFunction<GetTypeFunction>(api, ApiSlot::GetTensorElementType)(
            shapeInfo, &elementType));
        if (!error.empty() || elementType != TensorElementDataType::Float) {
            releaseShape(shapeInfo);
            releaseValue(output);
            return {.message = error.empty() ? "The terrain model output is not float32"
                                             : "Could not inspect model output type: " + error};
        }
        size_t dimensionCount = 0;
        error = consumeStatus(apiFunction<GetDimensionCountFunction>(
            api, ApiSlot::GetDimensionsCount)(shapeInfo, &dimensionCount));
        if (!error.empty()) {
            releaseShape(shapeInfo);
            releaseValue(output);
            return {.message = "Could not inspect model output dimensions: " + error};
        }
        std::vector<int64_t> shape(dimensionCount);
        error = consumeStatus(apiFunction<GetDimensionsFunction>(api, ApiSlot::GetDimensions)(
            shapeInfo, shape.data(), shape.size()));
        size_t elementCount = 0;
        if (error.empty()) {
            error = consumeStatus(apiFunction<GetElementCountFunction>(
                api, ApiSlot::GetTensorShapeElementCount)(shapeInfo, &elementCount));
        }
        releaseShape(shapeInfo);
        if (!error.empty() || elementCount > MAXIMUM_INFERENCE_FLOATS) {
            releaseValue(output);
            return {.message = error.empty() ? "Terrain model output exceeds the tensor budget"
                                             : "Could not inspect model output shape: " + error};
        }
        void* outputData = nullptr;
        error = consumeStatus(
            apiFunction<GetDataFunction>(api, ApiSlot::GetTensorMutableData)(output, &outputData));
        if (!error.empty() || (elementCount != 0 && outputData == nullptr)) {
            releaseValue(output);
            return {.message = error.empty() ? "Terrain model returned a null output tensor"
                                             : "Could not read model output data: " + error};
        }
        const auto* floats = static_cast<const float*>(outputData);
        std::vector<float> outputValues(floats, floats + elementCount);
        releaseValue(output);
        return {
            .succeeded = true,
            .output = {.name = std::string(modelFileName(model)),
                       .shape = std::move(shape),
                       .values = std::move(outputValues)},
        };
    }

    TerrainRuntimeInferenceResult
    runModel(TerrainRuntimeModel model, std::span<const TerrainRuntimeTensor> inputs,
             TerrainRuntimeInferencePhase phase,
             const bootstrap::TerrainBootstrapCancellation* cancellation) override {
        {
            std::lock_guard lock(metricsMutex);
            ++runtimeMetrics.queuedInferenceCalls;
        }
        std::unique_lock inferenceLock(inferenceMutex);
        {
            std::lock_guard lock(metricsMutex);
            --runtimeMetrics.queuedInferenceCalls;
            ++runtimeMetrics.activeInferenceCalls;
            runtimeMetrics.maximumConcurrentInferenceCalls =
                std::max(runtimeMetrics.maximumConcurrentInferenceCalls,
                         runtimeMetrics.activeInferenceCalls);
            ++runtimeMetrics.inferenceCalls;
        }
        const auto inferenceStarted = std::chrono::steady_clock::now();
        TerrainRuntimeInferenceResult result = runLocked(model, inputs, cancellation);
        const uint64_t elapsedNanoseconds =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - inferenceStarted)
                                      .count());
        {
            std::lock_guard lock(metricsMutex);
            --runtimeMetrics.activeInferenceCalls;
            if (!result.succeeded)
                ++runtimeMetrics.inferenceFailures;
            TerrainRuntimeMetrics::Model& modelMetrics =
                runtimeMetrics.models[static_cast<size_t>(model)];
            ++modelMetrics.calls;
            modelMetrics.inferenceNanoseconds += elapsedNanoseconds;
            modelMetrics.maximumInferenceNanoseconds =
                std::max(modelMetrics.maximumInferenceNanoseconds, elapsedNanoseconds);
            const size_t phaseIndex = static_cast<size_t>(phase);
            if (phaseIndex < runtimeMetrics.phases.size()) {
                TerrainRuntimeMetrics::Phase& phaseMetrics =
                    runtimeMetrics.phases[phaseIndex];
                ++phaseMetrics.calls;
                phaseMetrics.inferenceNanoseconds += elapsedNanoseconds;
                ++phaseMetrics.modelCalls[static_cast<size_t>(model)];
            }
        }
        return result;
    }

    learned::GenerationIdentity identity;
    std::string expectedQualificationHash;
    std::filesystem::path packPath;
    std::filesystem::path extractionPath;
    std::filesystem::path dylibPath;
    std::filesystem::path cachePath;
    const void* api = nullptr;
    OrtEnv* environment = nullptr;
    std::array<OrtSession*, MODELS.size()> sessions{};
    TerrainRuntimeThreadPoolConfiguration threadPoolConfiguration;
    bool compiled = false;
    bool qualified = false;
    mutable std::mutex inferenceMutex;
    mutable std::mutex metricsMutex;
    TerrainRuntimeMetrics runtimeMetrics;
};

ProductionTerrainRuntime::ProductionTerrainRuntime(uint64_t seed,
                                                   std::string expectedQualificationHash)
    : impl_(std::make_shared<Impl>(seed, std::move(expectedQualificationHash))) {}

ProductionTerrainRuntime::~ProductionTerrainRuntime() = default;

bootstrap::TerrainRuntimeStepResult ProductionTerrainRuntime::qualifyPlatform() {
#if !defined(__arm64__) && !defined(__aarch64__)
    return bootstrap::TerrainRuntimeStepResult::failureResult(
        bootstrap::TerrainBootstrapFailureCode::UnsupportedPlatform,
        "Generator v4 requires an Apple Silicon process", false);
#else
    const NSOperatingSystemVersion minimum{14, 0, 0};
    if (![[NSProcessInfo processInfo] isOperatingSystemAtLeastVersion:minimum]) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::UnsupportedPlatform,
            "Generator v4 requires macOS 14 or newer", false);
    }
    return bootstrap::TerrainRuntimeStepResult::success();
#endif
}

bootstrap::TerrainRuntimeStepResult
ProductionTerrainRuntime::compile(const std::filesystem::path& installedPack,
                                  const bootstrap::TerrainBootstrapCancellation& cancellation) {
    {
        std::lock_guard lock(contextMutex_);
        generationContext_.reset();
    }
    if (cancellation.canceled()) {
        std::unique_lock inferenceLock(impl_->inferenceMutex);
        impl_->resetRuntime();
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Canceled,
            "Terrain runtime compilation canceled", true);
    }
    std::unique_lock inferenceLock(impl_->inferenceMutex);
    impl_->resetRuntime();
    impl_->setThreadPoolConfiguration(
        terrainRuntimeThreadPoolConfiguration(detectedPhysicalCpuCount()));
    impl_->packPath = installedPack;
    impl_->extractionPath = installedPack / "runtime-v1.27.1";
    impl_->dylibPath = impl_->extractionPath / ONNX_RUNTIME_DIRECTORY / ONNX_RUNTIME_DYLIB;
    impl_->cachePath = installedPack / CORE_ML_CACHE_DIRECTORY;

    std::error_code error;
    const std::filesystem::path archive = installedPack / ONNX_RUNTIME_ARCHIVE;
    if (!std::filesystem::is_regular_file(archive, error) || error) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation,
            "The verified ONNX Runtime archive is missing", true);
    }
    if (const std::optional<std::string> extractionError =
            extractRuntimeArchive(archive, impl_->extractionPath, cancellation)) {
        const bool canceled = cancellation.canceled();
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            canceled ? bootstrap::TerrainBootstrapFailureCode::Canceled
                     : bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation,
            *extractionError, true);
    }
    if (const std::optional<std::string> loadError = impl_->loadRuntimeLibrary()) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation, *loadError, true);
    }
    if (const std::optional<std::string> providerError = impl_->requireCoreMlProvider()) {
        impl_->resetRuntime();
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation, *providerError, true);
    }

    for (TerrainRuntimeModel model : MODELS) {
        if (cancellation.canceled()) {
            impl_->resetRuntime();
            return bootstrap::TerrainRuntimeStepResult::failureResult(
                bootstrap::TerrainBootstrapFailureCode::Canceled,
                "Terrain runtime compilation canceled", true);
        }
        auto [session, sessionError] = impl_->createSession(model);
        if (sessionError) {
            impl_->resetRuntime();
            return bootstrap::TerrainRuntimeStepResult::failureResult(
                bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation, *sessionError, true);
        }
        impl_->sessions[static_cast<size_t>(model)] = session;
        impl_->updateResidentSessionMetrics();
    }
    impl_->compiled = true;
    return bootstrap::TerrainRuntimeStepResult::success();
}

bootstrap::TerrainRuntimeStepResult ProductionTerrainRuntime::loadAndQualify(
    const std::filesystem::path& installedPack,
    const bootstrap::TerrainBootstrapCancellation& cancellation) {
    if (installedPack != impl_->packPath || !impl_->compiled) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::RuntimeLoading,
            "Terrain runtime qualification requires the compiled verified model pack", true);
    }
    std::vector<TerrainQualificationOutput> outputs;
    outputs.reserve(MODELS.size());
    for (TerrainRuntimeModel model : MODELS) {
        const std::vector<TerrainRuntimeTensor> inputs = terrainQualificationInputs(model);
        TerrainRuntimeInferenceResult result =
            runModel(model, inputs, TerrainRuntimeInferencePhase::Qualification, &cancellation);
        if (!result.succeeded) {
            const bool canceled = cancellation.canceled();
            return bootstrap::TerrainRuntimeStepResult::failureResult(
                canceled ? bootstrap::TerrainBootstrapFailureCode::Canceled
                         : bootstrap::TerrainBootstrapFailureCode::Qualification,
                result.message, true);
        }
        outputs.push_back({.model = model,
                           .shape = std::move(result.output.shape),
                           .values = std::move(result.output.values)});
    }
    const learned::Sha256Digest digest = quantizedQualificationDigest(outputs);
    {
        std::lock_guard lock(impl_->metricsMutex);
        impl_->runtimeMetrics.qualificationDigest = digest;
    }
    if (!qualificationDigestMatches(digest, impl_->expectedQualificationHash)) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Qualification,
            "Canonical terrain qualification hash mismatch: expected " +
                impl_->expectedQualificationHash + ", found " + learned::sha256Hex(digest),
            true);
    }

    const std::filesystem::path applicationSupportRoot =
        installedPack.parent_path().parent_path().parent_path();
    if (applicationSupportRoot.empty()) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Qualification,
            "The verified terrain model pack is outside the application support layout", true);
    }
    try {
        std::shared_ptr<learned::TerrainInferenceBackend> backend =
            makeInfiniteDiffusionTerrainBackend(impl_->identity.seed, installedPack, impl_);
        auto authority = std::make_shared<learned::CachedTerrainAuthority>(
            impl_->identity,
            applicationSupportRoot / bootstrap::V4_WORLD_DIRECTORY /
                SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
            std::move(backend));
        auto context = std::make_shared<learned::WorldGenerationContext>(
            impl_->identity, std::move(authority), learned::AuthorityQuality::FINAL,
            applicationSupportRoot / bootstrap::V4_WORLD_DIRECTORY /
                SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
        std::lock_guard lock(contextMutex_);
        generationContext_ = std::move(context);
    } catch (const std::exception& exception) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Qualification,
            "Could not construct the qualified terrain authority: " + std::string(exception.what()),
            true);
    }
    impl_->qualified = true;
    return bootstrap::TerrainRuntimeStepResult::success();
}

bootstrap::TerrainRuntimeStepResult
ProductionTerrainRuntime::bindWorldProfile(const std::filesystem::path& worldPath) {
    if (!impl_->qualified || impl_->packPath.empty()) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::RuntimeLoading,
            "Terrain runtime profile binding requires qualified model sessions", true);
    }
    std::error_code error;
    if (worldPath.empty() || !std::filesystem::is_directory(worldPath, error) || error) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Filesystem,
            "The selected generator v4 profile directory is unavailable", true);
    }

    try {
        std::shared_ptr<learned::TerrainInferenceBackend> backend =
            makeInfiniteDiffusionTerrainBackend(impl_->identity.seed, impl_->packPath, impl_);
        auto authority = std::make_shared<learned::CachedTerrainAuthority>(
            impl_->identity, worldPath / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
            std::move(backend));
        auto context = std::make_shared<learned::WorldGenerationContext>(
            impl_->identity, std::move(authority), learned::AuthorityQuality::FINAL,
            worldPath / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
        if (learned::sha256Hex(context->fingerprint()) !=
            learned::sha256Hex(impl_->identity.fingerprint())) {
            return bootstrap::TerrainRuntimeStepResult::failureResult(
                bootstrap::TerrainBootstrapFailureCode::Qualification,
                "The selected generator v4 profile does not match the qualified identity", false);
        }
        std::lock_guard lock(contextMutex_);
        generationContext_ = std::move(context);
    } catch (const std::exception& exception) {
        return bootstrap::TerrainRuntimeStepResult::failureResult(
            bootstrap::TerrainBootstrapFailureCode::Qualification,
            "Could not bind terrain authority to the selected generator v4 profile: " +
                std::string(exception.what()),
            true);
    }
    return bootstrap::TerrainRuntimeStepResult::success();
}

std::optional<std::string> ProductionTerrainRuntime::qualifiedGenerationFingerprint() const {
    if (!impl_->qualified)
        return std::nullopt;
    return learned::sha256Hex(impl_->identity.fingerprint());
}

std::shared_ptr<learned::WorldGenerationContext>
ProductionTerrainRuntime::qualifiedGenerationContext() const {
    if (!impl_->qualified)
        return nullptr;
    std::lock_guard lock(contextMutex_);
    return generationContext_;
}

TerrainRuntimeInferenceResult
ProductionTerrainRuntime::runModel(TerrainRuntimeModel model,
                                   std::span<const TerrainRuntimeTensor> inputs,
                                   TerrainRuntimeInferencePhase phase,
                                   const bootstrap::TerrainBootstrapCancellation* cancellation) {
    return impl_->runModel(model, inputs, phase, cancellation);
}

const learned::GenerationIdentity& ProductionTerrainRuntime::generationIdentity() const noexcept {
    return impl_->identity;
}

TerrainRuntimeMetrics ProductionTerrainRuntime::metrics() const {
    std::lock_guard lock(impl_->metricsMutex);
    return impl_->runtimeMetrics;
}

std::filesystem::path ProductionTerrainRuntime::runtimeDylibPath() const {
    return impl_->dylibPath;
}

std::unique_ptr<bootstrap::TerrainRuntimePreparation> makeProductionTerrainRuntime(uint64_t seed) {
    return std::make_unique<ProductionTerrainRuntime>(seed);
}

} // namespace worldgen::runtime
