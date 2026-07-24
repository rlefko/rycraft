#include <catch2/catch_all.hpp>

#include "engine/v4_world_startup.hpp"
#include "render/far_terrain.hpp"
#include "test_helpers.hpp"
#include "world/artifact_analysis.hpp"
#include "world/infinite_diffusion_backend.hpp"
#include "world/native_hydrology.hpp"
#include "world/save_manager.hpp"
#include "world/terrain_runtime.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

using namespace worldgen;
using namespace worldgen::runtime;

class ZeroTerrainExecutor final : public TerrainModelExecutor {
public:
    explicit ZeroTerrainExecutor(bool recordDecoderBatches = false)
        : recordDecoderBatches_(recordDecoderBatches) {}

    TerrainRuntimeInferenceResult
    runModel(TerrainRuntimeModel model, std::span<const TerrainRuntimeTensor> inputs,
             TerrainRuntimeInferencePhase phase,
             const bootstrap::TerrainBootstrapCancellation* cancellation) override {
        phases[static_cast<size_t>(phase)].fetch_add(1, std::memory_order_relaxed);
        phaseModelCalls[static_cast<size_t>(phase)][static_cast<size_t>(model)].fetch_add(
            1, std::memory_order_relaxed);
        if (cancellation != nullptr && cancellation->canceled())
            return {.message = "Canceled"};
        ++calls;
        ++modelCalls[static_cast<size_t>(model)];
        switch (model) {
            case TerrainRuntimeModel::Coarse:
                if (inputs.size() != 7 ||
                    inputs.front().shape != std::vector<int64_t>{1, 11, 64, 64})
                    return {.message = "Invalid coarse inputs"};
                return coarseResult(inputs.front().values);
            case TerrainRuntimeModel::Base:
                if (inputs.size() != 3 ||
                    inputs.front().shape != std::vector<int64_t>{4, 5, 64, 64})
                    return {.message = "Invalid base inputs"};
                return zeroResult({4, 5, 64, 64});
            case TerrainRuntimeModel::Decoder:
                if (inputs.size() != 2 ||
                    inputs.front().shape != std::vector<int64_t>{4, 5, 256, 256} ||
                    inputs[1].shape != std::vector<int64_t>{4})
                    return {.message = "Invalid decoder inputs"};
                if (recordDecoderBatches_)
                    recordDecoderBatch(inputs.front().values);
                return zeroResult({4, 1, 256, 256});
        }
        return {.message = "Unknown terrain model"};
    }

    std::atomic<uint64_t> calls{0};
    std::array<std::atomic<uint64_t>, 3> modelCalls{};
    std::array<std::atomic<uint64_t>, TERRAIN_RUNTIME_INFERENCE_PHASE_COUNT> phases{};
    std::array<std::array<std::atomic<uint64_t>, 3>, TERRAIN_RUNTIME_INFERENCE_PHASE_COUNT>
        phaseModelCalls{};

    [[nodiscard]] uint64_t callCount(TerrainRuntimeModel model) const noexcept {
        return modelCalls[static_cast<size_t>(model)].load();
    }

    [[nodiscard]] uint64_t phaseCount(TerrainRuntimeInferencePhase phase) const noexcept {
        return phases[static_cast<size_t>(phase)].load();
    }

    [[nodiscard]] uint64_t phaseModelCount(TerrainRuntimeInferencePhase phase,
                                           TerrainRuntimeModel model) const noexcept {
        return phaseModelCalls[static_cast<size_t>(phase)][static_cast<size_t>(model)].load();
    }

    [[nodiscard]] std::vector<size_t> decoderBatchUniqueCounts() const {
        std::lock_guard lock(decoderBatchMutex_);
        return decoderBatchUniqueCounts_;
    }

private:
    void recordDecoderBatch(std::span<const float> values) {
        constexpr size_t VALUES_PER_ITEM = 5ULL * 256 * 256;
        size_t uniqueItems = 0;
        for (size_t item = 0; item < 4; ++item) {
            const auto begin = values.begin() + static_cast<std::ptrdiff_t>(item * VALUES_PER_ITEM);
            bool matchesEarlier = false;
            for (size_t earlier = 0; earlier < item; ++earlier) {
                const auto earlierBegin =
                    values.begin() + static_cast<std::ptrdiff_t>(earlier * VALUES_PER_ITEM);
                if (std::equal(begin, begin + static_cast<std::ptrdiff_t>(VALUES_PER_ITEM),
                               earlierBegin)) {
                    matchesEarlier = true;
                    break;
                }
            }
            if (!matchesEarlier)
                ++uniqueItems;
        }
        std::lock_guard lock(decoderBatchMutex_);
        decoderBatchUniqueCounts_.push_back(uniqueItems);
    }

    static TerrainRuntimeInferenceResult coarseResult(std::span<const float> input) {
        constexpr size_t AREA = 64 * 64;
        std::vector<float> values(6 * AREA);
        for (size_t channel = 0; channel < 6; ++channel) {
            for (size_t pixel = 0; pixel < AREA; ++pixel) {
                values[channel * AREA + pixel] = input[channel * AREA + pixel] * 0.07F +
                                                 input[(6 + channel % 5) * AREA + pixel] * 0.015F;
            }
        }
        return {.succeeded = true,
                .output = {.name = "output", .shape = {1, 6, 64, 64}, .values = std::move(values)}};
    }

    static TerrainRuntimeInferenceResult zeroResult(std::vector<int64_t> shape) {
        size_t count = 1;
        for (int64_t dimension : shape)
            count *= static_cast<size_t>(dimension);
        return {.succeeded = true,
                .output = {.name = "output",
                           .shape = std::move(shape),
                           .values = std::vector<float>(count)}};
    }

    bool recordDecoderBatches_ = false;
    mutable std::mutex decoderBatchMutex_;
    std::vector<size_t> decoderBatchUniqueCounts_;
};

void writeFakePipelineData(const std::filesystem::path& path) {
    std::ofstream output(path);
    REQUIRE(output.is_open());
    auto writeTables = [&output](std::string_view name) {
        output << '"' << name << "\":[";
        for (int channel = 0; channel < 5; ++channel) {
            output << '[';
            for (int index = 0; index < 64; ++index) {
                output << (-4.0 + static_cast<double>(index) * 8.0 / 63.0);
                if (index != 63)
                    output << ',';
            }
            output << ']';
            if (channel != 4)
                output << ',';
        }
        output << ']';
    };
    output << '{';
    writeTables("noise_quantile_tables");
    output << ',';
    writeTables("data_quantile_tables");
    output << R"(,"a_temp_std":0.1,"b_temp_std":2.0,"temp_std_p1":0.0,"temp_std_p99":10.0})";
    output.close();
    REQUIRE(output.good());
}

void writeFakeModelConfiguration(const std::filesystem::path& path) {
    std::ofstream output(path);
    REQUIRE(output.is_open());
    output << R"({
  "_class_name": "WorldPipeline",
  "coarse_means": [-37.70000792952155, 1.1403065255556186, 18.102486588653473, 332.8342598198454, 1332.2078969994473, 52.660088206981435],
  "coarse_pooling": 1,
  "coarse_stds": [39.741999742263, 1.7681844104569366, 8.92146918789914, 321.7660336396054, 842.9293648884745, 31.079985318715785],
  "cond_snr": [0.5, 0.5, 0.5, 0.5, 0.5],
  "drop_water_pct": 0.5,
  "elev_coarse_pool_mode": "avg",
  "frequency_mult": [1.0, 1.0, 1.0, 1.0, 1.0],
  "histogram_raw": null,
  "latent_compression": 8,
  "native_resolution": 30,
  "p5_coarse_pool_mode": "avg",
  "residual_mean": 0.0,
  "residual_std": 0.7
})";
    output.close();
    REQUIRE(output.good());
}

constexpr uint64_t REAL_REGION_SEED = 0x5259'4352'4146'5404ULL;

constexpr std::array<learned::TerrainPageKey, 9> ORIGIN_HYDROLOGY_TERRAIN_PAGES{{
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -1, .column = -1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -1, .column = 0}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -1, .column = 1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 0, .column = -1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 0, .column = 0}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 0, .column = 1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 1, .column = -1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 1, .column = 0}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 1, .column = 1}},
}};

constexpr std::array<learned::TerrainPageKey, 4> ORIGIN_HYDROLOGY_CORE_TERRAIN_PAGES{{
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 0, .column = 0}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 0, .column = 1}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 1, .column = 0}},
    {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = 1, .column = 1}},
}};

struct RealRegionRun {
    double setupSeconds = 0.0;
    double regionSeconds = 0.0;
    learned::Sha256Digest regionHash{};
    TerrainRuntimeMetrics beforeRegion;
    TerrainRuntimeMetrics afterRegion;
    learned::WorldGenerationMetrics contextMetrics;
};

std::filesystem::path stageRealModelPack(const std::filesystem::path& sourcePack,
                                         const std::filesystem::path& applicationSupportRoot) {
    const std::filesystem::path stagedPack = applicationSupportRoot / "terrain-models" /
                                             bootstrap::TERRAIN_MODEL_DIRECTORY /
                                             bootstrap::TERRAIN_MODEL_REVISION;
    std::error_code error;
    std::filesystem::create_directories(stagedPack, error);
    CAPTURE(stagedPack, error.message());
    REQUIRE_FALSE(error);
    for (const bootstrap::TerrainAssetSpec& asset : bootstrap::pinnedTerrainAssets()) {
        const std::filesystem::path source = sourcePack / asset.fileName;
        error.clear();
        const bool regular = std::filesystem::is_regular_file(source, error);
        CAPTURE(source, error.message());
        REQUIRE(regular);
        REQUIRE_FALSE(error);
        CHECK(std::filesystem::file_size(source) == asset.byteSize);
        error.clear();
        std::filesystem::create_symlink(std::filesystem::absolute(source),
                                        stagedPack / asset.fileName, error);
        CAPTURE(source, stagedPack, error.message());
        REQUIRE_FALSE(error);
    }
    return stagedPack;
}

learned::Sha256Digest loadRealRegionHash(const std::filesystem::path& applicationSupportRoot,
                                         const learned::GenerationIdentity& identity,
                                         std::span<const learned::TerrainPageKey> pages) {
    learned::TerrainPageStore store(
        applicationSupportRoot / bootstrap::V4_WORLD_DIRECTORY / "terrain-authority-v1", identity);
    std::vector<uint8_t> pageHashes;
    pageHashes.reserve(pages.size() * learned::Sha256Digest{}.size());
    for (const learned::TerrainPageKey key : pages) {
        auto page = store.loadPage(key);
        CAPTURE(key.coordinate.row, key.coordinate.column,
                page.failure() == nullptr ? std::string{} : page.failure()->message);
        REQUIRE(page.isReady());
        const auto* bytes = reinterpret_cast<const uint8_t*>(page.value()->samples.data());
        const learned::Sha256Digest pageHash =
            learned::sha256(std::span<const uint8_t>(bytes, page.value()->byteSize()));
        pageHashes.insert(pageHashes.end(), pageHash.begin(), pageHash.end());
    }
    return learned::sha256(pageHashes);
}

RealRegionRun runRealRegion(const std::filesystem::path& stagedPack,
                            const std::filesystem::path& applicationSupportRoot,
                            std::string_view expectedQualificationHash,
                            std::span<const learned::TerrainPageKey> pages, bool reverseOrder) {
    RealRegionRun run;
    const auto setupStart = std::chrono::steady_clock::now();
    ProductionTerrainRuntime runtime(REAL_REGION_SEED, std::string(expectedQualificationHash));
    bootstrap::TerrainBootstrapCancellation cancellation;
    REQUIRE(runtime.qualifyPlatform().succeeded);
    const bootstrap::TerrainRuntimeStepResult compilation =
        runtime.compile(stagedPack, cancellation);
    CAPTURE(compilation.failure.message);
    REQUIRE(compilation.succeeded);
    const bootstrap::TerrainRuntimeStepResult qualification =
        runtime.loadAndQualify(stagedPack, cancellation);
    CAPTURE(qualification.failure.message);
    REQUIRE(qualification.succeeded);
    const std::shared_ptr<learned::WorldGenerationContext> context =
        runtime.qualifiedGenerationContext();
    REQUIRE(context);
    run.setupSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - setupStart).count();
    run.beforeRegion = runtime.metrics();

    std::vector<bool> ready(pages.size());
    size_t readyCount = 0;
    auto request = [&](size_t index) {
        const learned::TerrainPageKey key = pages[index];
        const int64_t worldX = key.coordinate.column * learned::AUTHORITY_PAGE_BLOCK_EDGE;
        const int64_t worldZ = key.coordinate.row * learned::AUTHORITY_PAGE_BLOCK_EDGE;
        auto result =
            context->requestWorldPage(worldX, worldZ, learned::AuthorityRequestPriority::SPAWN);
        if (result.status() == learned::AuthorityStatus::FAILED) {
            CAPTURE(key.coordinate.row, key.coordinate.column,
                    result.failure() == nullptr ? std::string{} : result.failure()->message);
            FAIL("Real terrain authority region generation failed");
        }
        if (result.isReady() && !ready[index]) {
            ready[index] = true;
            ++readyCount;
        }
    };

    const auto regionStart = std::chrono::steady_clock::now();
    if (reverseOrder) {
        for (size_t reverseIndex = pages.size(); reverseIndex > 0; --reverseIndex)
            request(reverseIndex - 1);
    } else {
        for (size_t index = 0; index < pages.size(); ++index)
            request(index);
    }

    const auto deadline = regionStart + std::chrono::minutes(5);
    while (readyCount != ready.size() && std::chrono::steady_clock::now() < deadline) {
        for (size_t index = 0; index < ready.size(); ++index) {
            if (!ready[index])
                request(index);
        }
        if (readyCount != ready.size())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    run.regionSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - regionStart).count();
    CAPTURE(readyCount, run.regionSeconds);
    REQUIRE(readyCount == ready.size());
    run.afterRegion = runtime.metrics();
    run.contextMetrics = context->metrics();
    run.regionHash =
        loadRealRegionHash(applicationSupportRoot, runtime.generationIdentity(), pages);
    return run;
}

void reportRealRegionRun(std::string_view label, std::string_view regionDescription,
                         const RealRegionRun& run) {
    WARN(label << " real terrain setup seconds: " << run.setupSeconds);
    WARN(label << ' ' << regionDescription << " seconds: " << run.regionSeconds);
    WARN(label << ' ' << regionDescription << " hash: " << learned::sha256Hex(run.regionHash));
    WARN(label << " authority builds: " << run.contextMetrics.authorityCache.builds
               << ", disk loads: " << run.contextMetrics.authorityCache.diskLoads
               << ", deferred requests: " << run.contextMetrics.deferredQueries);
    static constexpr std::array<std::string_view, 3> MODEL_NAMES{
        "coarse",
        "base",
        "decoder",
    };
    for (size_t model = 0; model < MODEL_NAMES.size(); ++model) {
        const auto& before = run.beforeRegion.models[model];
        const auto& after = run.afterRegion.models[model];
        WARN(
            label << ' ' << MODEL_NAMES[model] << " region calls: " << after.calls - before.calls
                  << ", inference seconds: "
                  << static_cast<double>(after.inferenceNanoseconds - before.inferenceNanoseconds) /
                         1.0E9
                  << ", maximum call seconds: "
                  << static_cast<double>(after.maximumInferenceNanoseconds) / 1.0E9
                  << ", session creation seconds: "
                  << static_cast<double>(after.sessionCreationNanoseconds -
                                         before.sessionCreationNanoseconds) /
                         1.0E9);
    }
}

std::string compactJson(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(),
                              [](unsigned char value) { return std::isspace(value) != 0; }),
               text.end());
    return text;
}

size_t occurrenceCount(std::string_view text, std::string_view needle) {
    size_t count = 0;
    for (size_t offset = 0; (offset = text.find(needle, offset)) != std::string_view::npos;
         offset += needle.size()) {
        ++count;
    }
    return count;
}

std::string jsonShape(std::span<const int64_t> shape) {
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0)
            output << ',';
        output << shape[index];
    }
    output << ']';
    return output.str();
}

std::string qualificationInputShapes(TerrainRuntimeModel model) {
    const std::vector<TerrainRuntimeTensor> inputs = terrainQualificationInputs(model);
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < inputs.size(); ++index) {
        if (index != 0)
            output << ',';
        output << jsonShape(inputs[index].shape);
    }
    output << ']';
    return output.str();
}

std::string windowManifestEntry(std::string_view name, learned::WindowGeometry geometry) {
    std::ostringstream output;
    output << '"' << name << "\":{\"edge\":" << geometry.edge << ",\"stride\":" << geometry.stride
           << ",\"inference_steps\":" << geometry.inferenceSteps
           << ",\"batch_size\":" << geometry.batchSize << '}';
    return output.str();
}

constexpr std::array<int64_t, 6> LEARNED_AUTHORITY_CONTROL_SPACINGS{
    16, 256, 768, 1'024, 2'048, 8'192,
};
constexpr std::array<int64_t, 8> LEARNED_AUTHORITY_NEARBY_NATIVE_OFFSETS{
    -7, -5, -3, -1, 1, 3, 5, 7,
};
constexpr int64_t LEARNED_AUTHORITY_ALONG_HALF_WINDOW = 96;
struct LearnedAuthorityProbeAnchor {
    int64_t lineMultiplier;
    int64_t alongMultiplier;
    int64_t alongOffset;
};
constexpr std::array<LearnedAuthorityProbeAnchor, 8> LEARNED_AUTHORITY_PROBE_ANCHORS{{
    {.lineMultiplier = 7, .alongMultiplier = -13, .alongOffset = -431},
    {.lineMultiplier = -11, .alongMultiplier = 17, .alongOffset = 617},
    {.lineMultiplier = 19, .alongMultiplier = -29, .alongOffset = 1'211},
    {.lineMultiplier = -23, .alongMultiplier = 37, .alongOffset = -1'597},
    {.lineMultiplier = 31, .alongMultiplier = -41, .alongOffset = 2'003},
    {.lineMultiplier = -37, .alongMultiplier = 53, .alongOffset = -2'411},
    {.lineMultiplier = 43, .alongMultiplier = -59, .alongOffset = 2'819},
    {.lineMultiplier = -47, .alongMultiplier = 67, .alongOffset = -3'227},
}};

enum class LearnedAuthorityField : uint8_t {
    Elevation,
    MeanTemperature,
    TemperatureVariability,
    AnnualPrecipitation,
    PrecipitationVariability,
    LapseRate,
    Count,
};

const char* learnedAuthorityFieldName(LearnedAuthorityField field) {
    switch (field) {
        case LearnedAuthorityField::Elevation:
            return "elevation";
        case LearnedAuthorityField::MeanTemperature:
            return "mean temperature";
        case LearnedAuthorityField::TemperatureVariability:
            return "temperature variability";
        case LearnedAuthorityField::AnnualPrecipitation:
            return "annual precipitation";
        case LearnedAuthorityField::PrecipitationVariability:
            return "precipitation variability";
        case LearnedAuthorityField::LapseRate:
            return "lapse rate";
        case LearnedAuthorityField::Count:
            return "invalid";
    }
    return "invalid";
}

double learnedAuthorityValue(const learned::PhysicalTerrainSample& sample,
                             LearnedAuthorityField field) {
    switch (field) {
        case LearnedAuthorityField::Elevation:
            return sample.elevationMeters;
        case LearnedAuthorityField::MeanTemperature:
            return sample.meanTemperatureC;
        case LearnedAuthorityField::TemperatureVariability:
            return sample.temperatureVariabilityC;
        case LearnedAuthorityField::AnnualPrecipitation:
            return sample.annualPrecipitationMm;
        case LearnedAuthorityField::PrecipitationVariability:
            return sample.precipitationCoefficientOfVariation;
        case LearnedAuthorityField::LapseRate:
            return sample.lapseRateCPerMeter;
        case LearnedAuthorityField::Count:
            return 0.0;
    }
    return 0.0;
}

struct LearnedAuthorityEnergy {
    double squaredSum = 0.0;
    size_t count = 0;

    void add(double derivative) {
        if (!std::isfinite(derivative))
            return;
        squaredSum += derivative * derivative;
        ++count;
    }

    [[nodiscard]] double mean() const {
        return count == 0 ? 0.0 : squaredSum / static_cast<double>(count);
    }
};

struct LearnedAuthorityMetric {
    LearnedAuthorityEnergy formerLine;
    LearnedAuthorityEnergy nearby;
    worldgen::artifact_analysis::OrientationHistogram orientation;
};

double seamlessAuthoritySignal(int64_t row, int64_t column, size_t channel) {
    const double phase = static_cast<double>(channel + 1) * 0.731;
    const double x = static_cast<double>(column);
    const double z = static_cast<double>(row);
    return (std::sin((x * 0.809 + z * 0.588) / 7.3 + phase) +
            std::cos((-x * 0.374 + z * 0.927) / 9.7 - phase * 1.3) +
            std::sin((x * 0.643 - z * 0.766) / 13.1 + phase * 0.7) +
            std::cos((x * 0.966 + z * 0.259) / 17.9 - phase * 1.9)) *
           0.25;
}

learned::QuantizedTerrainSample seamlessAuthoritySample(int64_t row, int64_t column) {
    return quantizeInfiniteDiffusionSample(
        static_cast<float>(480.0 + 240.0 * seamlessAuthoritySignal(row, column, 0)),
        static_cast<float>(15.0 + 6.0 * seamlessAuthoritySignal(row, column, 1)),
        static_cast<float>(600.0 + 180.0 * seamlessAuthoritySignal(row, column, 2)),
        static_cast<float>(1'200.0 + 350.0 * seamlessAuthoritySignal(row, column, 3)),
        static_cast<float>(30.0 + 8.0 * seamlessAuthoritySignal(row, column, 4)),
        static_cast<float>(-0.0065 + 0.0008 * seamlessAuthoritySignal(row, column, 5)));
}

class SeamlessLearnedAuthorityBackend final : public learned::TerrainInferenceBackend {
public:
    learned::AuthorityResult<learned::TerrainAuthorityPage>
    inferPage(const learned::GenerationIdentity& identity, learned::TerrainPageKey key) override {
        learned::TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(learned::AUTHORITY_PAGE_SAMPLE_COUNT);
        const int64_t rowBegin = key.coordinate.row * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        const int64_t columnBegin = key.coordinate.column * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        for (int row = 0; row < learned::AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < learned::AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                page.samples[static_cast<size_t>(row * learned::AUTHORITY_PAGE_NATIVE_EDGE +
                                                 column)] =
                    seamlessAuthoritySample(rowBegin + row, columnBegin + column);
            }
        }
        ++pageBuilds;
        return learned::AuthorityResult<learned::TerrainAuthorityPage>::ready(std::move(page));
    }

    learned::AuthorityResult<learned::PhysicalTerrainGrid>
    inferFinalNativeGrid(const learned::GenerationIdentity&, learned::NativeRect region) override {
        if (!region.valid()) {
            return learned::AuthorityResult<learned::PhysicalTerrainGrid>::failed({
                .code = learned::GenerationFailureCode::INVALID_REQUEST,
                .message = "Invalid seamless learned-authority fixture region",
                .retriable = false,
            });
        }
        learned::PhysicalTerrainGrid grid{
            .region = region,
            .samples = std::vector<learned::PhysicalTerrainSample>(
                static_cast<size_t>(region.height() * region.width())),
        };
        size_t index = 0;
        for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
            for (int64_t column = region.columnBegin; column < region.columnEnd; ++column) {
                grid.samples[index++] =
                    learned::dequantizeTerrainSample(seamlessAuthoritySample(row, column));
            }
        }
        ++transientBuilds;
        return learned::AuthorityResult<learned::PhysicalTerrainGrid>::ready(std::move(grid));
    }

    std::atomic<uint64_t> pageBuilds{0};
    std::atomic<uint64_t> transientBuilds{0};
};

const learned::PhysicalTerrainSample& nativeGridSample(const learned::PhysicalTerrainGrid& grid,
                                                       int64_t row, int64_t column) {
    const learned::PhysicalTerrainSample* sample = grid.sample(row, column);
    if (sample == nullptr)
        throw std::out_of_range("Learned authority probe left its fixture grid");
    return *sample;
}

void recordLearnedAuthorityAxis(
    const learned::PhysicalTerrainGrid& grid, bool vertical, int64_t line, int64_t alongCenter,
    std::array<LearnedAuthorityMetric, static_cast<size_t>(LearnedAuthorityField::Count)>&
        metrics) {
    const auto point = [vertical](int64_t normal, int64_t along) {
        return vertical ? learned::NativePoint{.row = along, .column = normal}
                        : learned::NativePoint{.row = normal, .column = along};
    };
    const auto valueAt = [&](LearnedAuthorityField field, int64_t normal, int64_t along) {
        const learned::NativePoint position = point(normal, along);
        return learnedAuthorityValue(nativeGridSample(grid, position.row, position.column), field);
    };
    const auto addOrientation = [&](worldgen::artifact_analysis::OrientationBins& bins,
                                    LearnedAuthorityField field, int64_t normal, int64_t along) {
        const learned::NativePoint center = point(normal, along);
        const double gradientX =
            (learnedAuthorityValue(nativeGridSample(grid, center.row, center.column + 1), field) -
             learnedAuthorityValue(nativeGridSample(grid, center.row, center.column - 1), field)) *
            0.5;
        const double gradientZ =
            (learnedAuthorityValue(nativeGridSample(grid, center.row + 1, center.column), field) -
             learnedAuthorityValue(nativeGridSample(grid, center.row - 1, center.column), field)) *
            0.5;
        if (!std::isfinite(gradientX) || !std::isfinite(gradientZ) ||
            std::hypot(gradientX, gradientZ) <= 1.0e-8) {
            return;
        }
        ++bins[worldgen::artifact_analysis::orientationBin(gradientX, gradientZ)];
    };

    for (int64_t along = alongCenter - LEARNED_AUTHORITY_ALONG_HALF_WINDOW;
         along <= alongCenter + LEARNED_AUTHORITY_ALONG_HALF_WINDOW; along += 2) {
        for (size_t fieldIndex = 0; fieldIndex < metrics.size(); ++fieldIndex) {
            const auto field = static_cast<LearnedAuthorityField>(fieldIndex);
            LearnedAuthorityMetric& metric = metrics[fieldIndex];
            metric.formerLine.add(
                (valueAt(field, line + 1, along) - valueAt(field, line - 1, along)) * 0.5);
            addOrientation(metric.orientation.formerLine, field, line, along);
            for (const int64_t nearby : LEARNED_AUTHORITY_NEARBY_NATIVE_OFFSETS) {
                metric.nearby.add((valueAt(field, line + nearby + 1, along) -
                                   valueAt(field, line + nearby - 1, along)) *
                                  0.5);
                addOrientation(metric.orientation.nearby, field, line + nearby, along);
            }
        }
    }
}

} // namespace

TEST_CASE("Checked-in terrain manifest matches production generation and runtime contracts",
          "[terrain-runtime][bootstrap][manifest][regression]") {
    const std::filesystem::path manifestPath =
        std::filesystem::path(RYCRAFT_SOURCE_ROOT) / "resources/config/terrain_model_manifest.json";
    std::ifstream input(manifestPath, std::ios::binary);
    REQUIRE(input.is_open());
    const std::string manifest = compactJson(
        std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));

    const std::string manifestHeader =
        "{\"schema_version\":" + std::to_string(bootstrap::TERRAIN_MODEL_MANIFEST_SCHEMA_VERSION) +
        ",\"model_revision\":\"" + std::string(bootstrap::TERRAIN_MODEL_REVISION) + "\"";
    CHECK(manifest.starts_with(manifestHeader));

    std::ostringstream generation;
    generation << "\"generation\":{\"model_block_scale\":" << learned::MODEL_BLOCK_SCALE
               << ",\"meters_per_block\":" << learned::WORLD_METERS_PER_BLOCK
               << ",\"terrain_authority_schema_version\":"
               << learned::TERRAIN_AUTHORITY_SCHEMA_VERSION << ",\"windows\":{"
               << windowManifestEntry("coarse", learned::COARSE_WINDOW) << ','
               << windowManifestEntry("latent", learned::LATENT_WINDOW) << ','
               << windowManifestEntry("decoder", learned::DECODER_WINDOW) << "}}";
    INFO(generation.str());
    CHECK(manifest.contains(generation.str()));

    const auto& assets = bootstrap::pinnedTerrainAssets();
    REQUIRE(assets.size() == 6);
    CHECK(occurrenceCount(manifest, "\"bytes\":") == assets.size());
    const bootstrap::TerrainAssetSpec* runtimeAsset = nullptr;
    for (const bootstrap::TerrainAssetSpec& asset : assets) {
        if (asset.kind == bootstrap::TerrainAssetKind::Runtime) {
            REQUIRE(runtimeAsset == nullptr);
            runtimeAsset = &asset;
            continue;
        }
        const std::string entry =
            "{\"name\":\"" + asset.fileName + "\",\"bytes\":" + std::to_string(asset.byteSize) +
            ",\"sha256\":\"" + asset.sha256 + "\",\"url\":\"" + asset.url + "\"}";
        INFO(entry);
        CHECK(manifest.contains(entry));
    }
    REQUIRE(runtimeAsset != nullptr);
    std::ostringstream runtime;
    runtime << "\"runtime\":{\"name\":\"" << runtimeAsset->fileName << "\",\"version\":\""
            << ONNX_RUNTIME_VERSION << "\",\"directory\":\"" << ONNX_RUNTIME_DIRECTORY
            << "\",\"dylib\":\"" << ONNX_RUNTIME_DYLIB << "\",\"bytes\":" << runtimeAsset->byteSize
            << ",\"sha256\":\"" << runtimeAsset->sha256 << "\",\"url\":\"" << runtimeAsset->url
            << "\"}";
    INFO(runtime.str());
    CHECK(manifest.contains(runtime.str()));
    CHECK(runtimeAsset->fileName == ONNX_RUNTIME_ARCHIVE);

    std::ostringstream providerOptions;
    providerOptions << "\"provider_options\":{";
    for (size_t index = 0; index < CORE_ML_PINNED_PROVIDER_OPTIONS.size(); ++index) {
        if (index != 0)
            providerOptions << ',';
        providerOptions << '"' << CORE_ML_PINNED_PROVIDER_OPTIONS[index].key << "\":\""
                        << CORE_ML_PINNED_PROVIDER_OPTIONS[index].value << '"';
    }
    providerOptions << '}';
    std::ostringstream qualification;
    qualification << "\"qualification\":{\"schema_version\":" << QUALIFICATION_SCHEMA_VERSION
                  << ",\"quantization_scale\":" << QUALIFICATION_QUANTIZATION_SCALE
                  << ",\"sha256\":\"" << CANONICAL_QUALIFICATION_HASH
                  << "\",\"execution_mode\":\"sequential\",\"provider\":\"" << CORE_ML_PROVIDER_NAME
                  << "\",\"model_cache_directory\":\"" << CORE_ML_CACHE_DIRECTORY << "\","
                  << providerOptions.str();
    INFO(qualification.str());
    CHECK(manifest.contains(qualification.str()));
    CHECK((learned::GENERATOR_V4_PROVIDER_CONFIGURATION.flags & learned::CORE_ML_REQUIRED_FLAGS) ==
          learned::CORE_ML_REQUIRED_FLAGS);

    struct ModelContract {
        TerrainRuntimeModel model;
        std::string_view fileName;
        std::span<const int64_t> outputShape;
    };
    const std::array contracts{
        ModelContract{TerrainRuntimeModel::Coarse, "coarse_model.onnx", COARSE_MODEL_OUTPUT_SHAPE},
        ModelContract{TerrainRuntimeModel::Base, "base_model.onnx", BASE_MODEL_OUTPUT_SHAPE},
        ModelContract{TerrainRuntimeModel::Decoder, "decoder_model.onnx",
                      DECODER_MODEL_OUTPUT_SHAPE},
    };
    CHECK(occurrenceCount(manifest, "\"inputs\":") == contracts.size());
    CHECK(occurrenceCount(manifest, "\"output\":") == contracts.size());
    for (const ModelContract& contract : contracts) {
        const std::string entry = "{\"name\":\"" + std::string(contract.fileName) +
                                  "\",\"inputs\":" + qualificationInputShapes(contract.model) +
                                  ",\"output\":" + jsonShape(contract.outputShape) + "}";
        INFO(entry);
        CHECK(manifest.contains(entry));
    }
}

TEST_CASE("Production window geometry preserves coordinate-pure learned fields at former grids",
          "[terrain-runtime][pipeline][continuity][artifact][window-fixture]") {
    constexpr std::array geometries{
        learned::COARSE_WINDOW,
        learned::LATENT_WINDOW,
        learned::DECODER_WINDOW,
    };
    constexpr size_t CHANNELS = static_cast<size_t>(LearnedAuthorityField::Count);
    for (const int64_t blockSpacing : LEARNED_AUTHORITY_CONTROL_SPACINGS) {
        const int64_t nativeLine = blockSpacing / learned::MODEL_BLOCK_SCALE;
        const learned::NativeRect target{
            .rowBegin = nativeLine - 5,
            .columnBegin = -nativeLine - 5,
            .rowEnd = nativeLine + 6,
            .columnEnd = -nativeLine + 6,
        };
        for (const learned::WindowGeometry geometry : geometries) {
            learned::WeightedWindowAccumulator accumulator(target, geometry, CHANNELS);
            std::vector<learned::WindowIndex> windows =
                learned::intersectingWindows(target, geometry);
            REQUIRE(windows.size() > 1);
            if ((blockSpacing / learned::MODEL_BLOCK_SCALE + geometry.stride) % 2 != 0)
                std::reverse(windows.begin(), windows.end());
            for (const learned::WindowIndex window : windows) {
                const int64_t rowBegin = window.row * geometry.stride;
                const int64_t columnBegin = window.column * geometry.stride;
                const size_t area = static_cast<size_t>(geometry.edge) * geometry.edge;
                std::vector<float> prediction(CHANNELS * area);
                for (size_t channel = 0; channel < CHANNELS; ++channel) {
                    for (int row = 0; row < geometry.edge; ++row) {
                        for (int column = 0; column < geometry.edge; ++column) {
                            prediction[channel * area + static_cast<size_t>(row) * geometry.edge +
                                       column] =
                                static_cast<float>(seamlessAuthoritySignal(
                                    rowBegin + row, columnBegin + column, channel));
                        }
                    }
                }
                REQUIRE(accumulator.addWindow(window, prediction));
            }
            const auto resolved = accumulator.resolve();
            REQUIRE(resolved.isReady());
            REQUIRE(accumulator.windowCount() == windows.size());
            const size_t targetArea = static_cast<size_t>(target.height() * target.width());
            for (size_t channel = 0; channel < CHANNELS; ++channel) {
                size_t pixel = 0;
                for (int64_t row = target.rowBegin; row < target.rowEnd; ++row) {
                    for (int64_t column = target.columnBegin; column < target.columnEnd; ++column) {
                        CAPTURE(blockSpacing, geometry.edge, geometry.stride, channel, row, column);
                        CHECK(resolved.value()->at(channel * targetArea + pixel) ==
                              Catch::Approx(seamlessAuthoritySignal(row, column, channel))
                                  .margin(2.0e-6));
                        ++pixel;
                    }
                }
            }
        }
    }
}

TEST_CASE("Absolute learned authority has no derivative or orientation energy at former grids",
          "[terrain-runtime][pipeline][continuity][artifact][absolute]") {
    TempDir directory("terrain_runtime_absolute_continuity");
    constexpr uint64_t SEED = 0x434F'4E54'494E'5549ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    auto backend = std::make_shared<SeamlessLearnedAuthorityBackend>();
    learned::CachedTerrainAuthority authority(identity, directory.path(), backend);
    const auto queryGrid = [&](learned::NativeRect region) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        auto result = authority.queryTransientFinalNativeGrid(
            region, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
        while (result.status() == learned::AuthorityStatus::DEFERRED &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            result = authority.queryTransientFinalNativeGrid(
                region, learned::AuthorityRequestPriority::EXPLORATION_EXACT);
        }
        return result;
    };

    for (const int64_t blockSpacing : LEARNED_AUTHORITY_CONTROL_SPACINGS) {
        REQUIRE(blockSpacing % learned::MODEL_BLOCK_SCALE == 0);
        std::array<LearnedAuthorityMetric, static_cast<size_t>(LearnedAuthorityField::Count)>
            metrics{};
        for (const LearnedAuthorityProbeAnchor anchor : LEARNED_AUTHORITY_PROBE_ANCHORS) {
            const int64_t nativeSpacing = blockSpacing / learned::MODEL_BLOCK_SCALE;
            const int64_t nativeLine = nativeSpacing * anchor.lineMultiplier;
            const int64_t alongCenter = nativeSpacing * anchor.alongMultiplier + anchor.alongOffset;
            const learned::NativeRect verticalRegion{
                .rowBegin = alongCenter - LEARNED_AUTHORITY_ALONG_HALF_WINDOW - 1,
                .columnBegin = nativeLine - 8,
                .rowEnd = alongCenter + LEARNED_AUTHORITY_ALONG_HALF_WINDOW + 2,
                .columnEnd = nativeLine + 9,
            };
            const auto vertical = queryGrid(verticalRegion);
            REQUIRE(vertical.isReady());
            REQUIRE((*vertical.value())->valid());
            recordLearnedAuthorityAxis(**vertical.value(), true, nativeLine, alongCenter, metrics);

            const learned::NativeRect horizontalRegion{
                .rowBegin = nativeLine - 8,
                .columnBegin = alongCenter - LEARNED_AUTHORITY_ALONG_HALF_WINDOW - 1,
                .rowEnd = nativeLine + 9,
                .columnEnd = alongCenter + LEARNED_AUTHORITY_ALONG_HALF_WINDOW + 2,
            };
            const auto horizontal = queryGrid(horizontalRegion);
            REQUIRE(horizontal.isReady());
            REQUIRE((*horizontal.value())->valid());
            recordLearnedAuthorityAxis(**horizontal.value(), false, nativeLine, alongCenter,
                                       metrics);
        }

        for (size_t fieldIndex = 0; fieldIndex < metrics.size(); ++fieldIndex) {
            const auto field = static_cast<LearnedAuthorityField>(fieldIndex);
            const LearnedAuthorityMetric& metric = metrics[fieldIndex];
            const double derivativeRatio = worldgen::artifact_analysis::energyRatio(
                metric.formerLine.mean(), metric.nearby.mean());
            const double orientationRatio =
                worldgen::artifact_analysis::structuredOrientationRatio(metric.orientation);
            INFO("spacing " << blockSpacing << " learned field " << learnedAuthorityFieldName(field)
                            << " derivative ratio " << derivativeRatio << " orientation ratio "
                            << orientationRatio << " former samples " << metric.formerLine.count
                            << " nearby samples " << metric.nearby.count);
            REQUIRE(metric.formerLine.count >= 1'550);
            REQUIRE(metric.nearby.count >= 12'400);
            CHECK(derivativeRatio >= worldgen::artifact_analysis::DERIVATIVE_RATIO_MINIMUM);
            CHECK(derivativeRatio <= worldgen::artifact_analysis::DERIVATIVE_RATIO_MAXIMUM);
            CHECK(orientationRatio <= worldgen::artifact_analysis::STRUCTURED_ORIENTATION_LIMIT);
        }
    }

    CHECK(backend->transientBuilds ==
          LEARNED_AUTHORITY_CONTROL_SPACINGS.size() * LEARNED_AUTHORITY_PROBE_ANCHORS.size() * 2);
}

TEST_CASE("Continuity qualification detects aligned derivative and orientation seams",
          "[terrain-runtime][continuity][artifact][negative-control]") {
    for (const int64_t spacing : LEARNED_AUTHORITY_CONTROL_SPACINGS) {
        const auto alignedSeam = [spacing](int64_t x, int64_t z) {
            const double smooth =
                std::sin((static_cast<double>(x) * 0.809 + static_cast<double>(z) * 0.588) / 17.3) +
                std::cos((-static_cast<double>(x) * 0.374 + static_cast<double>(z) * 0.927) / 23.7);
            return smooth + (x >= spacing ? 100.0 : 0.0);
        };
        const double former =
            worldgen::artifact_analysis::derivativeEnergy(alignedSeam, spacing, 137);
        double nearby = 0.0;
        for (const int offset : worldgen::artifact_analysis::NEARBY_OFFSETS) {
            nearby +=
                worldgen::artifact_analysis::derivativeEnergy(alignedSeam, spacing + offset, 137);
        }
        nearby /= static_cast<double>(worldgen::artifact_analysis::NEARBY_OFFSETS.size());
        const auto orientation =
            worldgen::artifact_analysis::orientationHistogram(alignedSeam, spacing, 137);
        const double derivativeRatio = worldgen::artifact_analysis::energyRatio(former, nearby);
        const double orientationRatio =
            worldgen::artifact_analysis::structuredOrientationRatio(orientation);
        INFO("spacing " << spacing << " negative derivative ratio " << derivativeRatio
                        << " negative orientation ratio " << orientationRatio);
        CHECK(derivativeRatio > worldgen::artifact_analysis::DERIVATIVE_RATIO_MAXIMUM);
        CHECK(orientationRatio > worldgen::artifact_analysis::STRUCTURED_ORIENTATION_LIMIT);
    }
}

TEST_CASE("Production terrain identity locks model and runtime revisions", "[terrain-runtime]") {
    constexpr uint64_t SEED = 0xFEDC'BA98'7654'3210ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    CHECK(learned::sha256Hex(productionGenerationIdentity(42).fingerprint()) ==
          "2375071d3919e7acbdf335e59927ef52725266f2e0256b69df39cf01549e886a");
    REQUIRE(identity.valid());
    CHECK(identity.seed == SEED);
    CHECK(identity.generatorVersion == 4);
    CHECK(identity.modelBlockScale == 4);
    CHECK(identity.quantizationRevision == 2);
    CHECK(identity.hydrologyRevision == learned::GENERATOR_V4_HYDROLOGY_REVISION);
    CHECK(identity.postprocessingRevision == 9);
    CHECK(identity.provider.provider == learned::InferenceProvider::CORE_ML);
    CHECK(identity.provider.onnxRuntimeMajorVersion == 1);
    CHECK(identity.provider.onnxRuntimeMinorVersion == 27);
    CHECK(identity.provider.onnxRuntimePatchVersion == 1);
    CHECK(identity.provider.flags == learned::CORE_ML_REQUIRED_FLAGS);
    CHECK((identity.provider.flags & learned::CORE_ML_STATIC_BASE_BATCH_FOUR) != 0);
    CHECK((identity.provider.flags & learned::CORE_ML_STATIC_DECODER_BATCH_FOUR) != 0);
    CHECK((identity.provider.flags & learned::CORE_ML_STATIC_DECODER_SPATIAL_256) != 0);
    CHECK(CORE_ML_CACHE_DIRECTORY == "coreml-cache-v3-base4-decoder4x256");
    CHECK(ONNX_RUNTIME_VERSION == "1.27.1");
    CHECK(CANONICAL_QUALIFICATION_HASH ==
          "6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df");
    CHECK(learned::parseSha256(CANONICAL_QUALIFICATION_HASH));
    CHECK_FALSE(qualificationDigestMatches(identity.runtimeHash, std::string(64, '0')));

    std::vector<uint8_t> modelHashBytes;
    std::optional<learned::Sha256Digest> pinnedRuntimeHash;
    for (const bootstrap::TerrainAssetSpec& asset : bootstrap::pinnedTerrainAssets()) {
        const std::optional<learned::Sha256Digest> digest = learned::parseSha256(asset.sha256);
        REQUIRE(digest);
        if (asset.kind == bootstrap::TerrainAssetKind::Model) {
            modelHashBytes.insert(modelHashBytes.end(), digest->begin(), digest->end());
        } else {
            REQUIRE_FALSE(pinnedRuntimeHash);
            pinnedRuntimeHash = digest;
        }
    }
    REQUIRE(pinnedRuntimeHash);
    CHECK(identity.modelPackHash == learned::sha256(modelHashBytes));
    CHECK(identity.runtimeHash == *pinnedRuntimeHash);
}

TEST_CASE("CPU fallback uses no more than the physical-core budget", "[terrain-runtime]") {
    CHECK(boundedCpuFallbackThreadCount(0) == 1);
    CHECK(boundedCpuFallbackThreadCount(1) == 1);
    CHECK(boundedCpuFallbackThreadCount(8) == 8);
    CHECK(boundedCpuFallbackThreadCount(MAXIMUM_CPU_FALLBACK_THREADS) ==
          MAXIMUM_CPU_FALLBACK_THREADS);
    CHECK(boundedCpuFallbackThreadCount(MAXIMUM_CPU_FALLBACK_THREADS + 1) ==
          MAXIMUM_CPU_FALLBACK_THREADS);
    CHECK(boundedCpuFallbackThreadCount(64) == MAXIMUM_CPU_FALLBACK_THREADS);
}

TEST_CASE("Terrain sessions share one bounded global ONNX Runtime thread pool",
          "[terrain-runtime]") {
    const TerrainRuntimeThreadPoolConfiguration unknown = terrainRuntimeThreadPoolConfiguration(0);
    CHECK(unknown.globalIntraOpThreads == 1);
    CHECK(unknown.globalInterOpThreads == 1);
    CHECK(unknown.usesGlobalThreadPool);
    CHECK(unknown.disablesPerSessionThreads);

    const TerrainRuntimeThreadPoolConfiguration small = terrainRuntimeThreadPoolConfiguration(8);
    CHECK(small.globalIntraOpThreads == 8);
    CHECK(small.globalInterOpThreads == GLOBAL_INTER_OP_THREAD_COUNT);
    CHECK(small.usesGlobalThreadPool);
    CHECK(small.disablesPerSessionThreads);

    const TerrainRuntimeThreadPoolConfiguration capped = terrainRuntimeThreadPoolConfiguration(64);
    CHECK(capped.globalIntraOpThreads == MAXIMUM_CPU_FALLBACK_THREADS);
    CHECK(capped.globalInterOpThreads == 1);
    CHECK(capped.usesGlobalThreadPool);
    CHECK(capped.disablesPerSessionThreads);
}

TEST_CASE("Qualification digest is quantized and order sensitive", "[terrain-runtime]") {
    const std::vector<TerrainQualificationOutput> outputs{
        {.model = TerrainRuntimeModel::Coarse, .shape = {1, 3}, .values = {0.0F, 0.5F, -1.25F}},
        {.model = TerrainRuntimeModel::Decoder, .shape = {1, 2}, .values = {2.0F, -0.125F}},
    };
    const learned::Sha256Digest first = quantizedQualificationDigest(outputs);
    const learned::Sha256Digest repeated = quantizedQualificationDigest(outputs);
    CHECK(first == repeated);
    INFO(learned::sha256Hex(first));
    CHECK(learned::sha256Hex(first) ==
          "6faf5f54d8d3f2f3867dbab903e555e799296f4832d73c42a76bffb07540e19f");

    std::vector<TerrainQualificationOutput> reversed = outputs;
    std::reverse(reversed.begin(), reversed.end());
    CHECK(quantizedQualificationDigest(reversed) != first);
    CHECK_FALSE(qualificationDigestMatches(first, "invalid"));
}

TEST_CASE("InfiniteDiffusion scheduler matches the published Java vector",
          "[terrain-runtime][pipeline-golden]") {
    static constexpr std::array<uint32_t, 21> EXPECTED_BITS{
        0x429ffffeU, 0x426ea148U, 0x422fae58U, 0x41ff134bU, 0x41b65a51U, 0x41802dabU, 0x4130dbceU,
        0x40ef014aU, 0x409dc7f1U, 0x404aef10U, 0x3ffd5cf4U, 0x3f98df1cU, 0x3f315ce4U, 0x3ec488aaU,
        0x3e4e3ce5U, 0x3dcab284U, 0x3d37d6c9U, 0x3c96bd58U, 0x3bd8fadbU, 0x3b03126dU, 0x00000000U,
    };
    const std::array<float, 21> actual = infiniteDiffusionEdmSigmaSchedule();
    for (size_t index = 0; index < actual.size(); ++index) {
        CAPTURE(index);
        CHECK(std::bit_cast<uint32_t>(actual[index]) == EXPECTED_BITS[index]);
    }
}

TEST_CASE("InfiniteDiffusion climate channels use WorldClim storage units",
          "[terrain-runtime][pipeline-golden]") {
    const learned::QuantizedTerrainSample sample =
        quantizeInfiniteDiffusionSample(-257.4F, 18.53F, 332.8F, 769.4F, 69.4F, -0.0065F);
    CHECK(sample.elevationMeters == -257);
    CHECK(sample.meanTemperatureCentidegrees == 1'853);
    CHECK(sample.temperatureVariabilityCentidegrees == 333);
    CHECK(sample.annualPrecipitationMillimeters == 769);
    CHECK(sample.precipitationCoefficientBasisPoints == 6'940);
    CHECK(sample.lapseRateMicrodegreesPerMeter == -6'500);
}

TEST_CASE("InfiniteDiffusion Laplacian postprocessing matches Torchvision 0.19.1",
          "[terrain-runtime][pipeline-golden]") {
    std::vector<float> residual(32 * 32);
    for (int row = 0; row < 32; ++row) {
        for (int column = 0; column < 32; ++column) {
            residual[static_cast<size_t>(row) * 32 + column] =
                static_cast<float>((row * 17 + column * 13) % 29 - 14) / 10.0F;
        }
    }
    std::vector<float> lowFrequency(8 * 8);
    for (int row = 0; row < 8; ++row) {
        for (int column = 0; column < 8; ++column) {
            lowFrequency[static_cast<size_t>(row) * 8 + column] =
                static_cast<float>((row * 7 + column * 11) % 19 - 9) / 4.0F;
        }
    }
    static constexpr std::array<float, 64> EXPECTED_LOW_FREQUENCY{
        0.150287777F,    0.105048031F,   0.00862736162F,  0.0493114516F,  0.0068647922F,
        -0.0316329151F,  0.0204012189F,  0.0752864406F,   0.0817262828F,  0.0603650473F,
        -0.00754746702F, 0.0360313691F,  -0.00649276609F, -0.063192293F,  -0.0312428605F,
        0.0039878767F,   -0.0170513168F, -0.0152487317F,  -0.0356872678F, -0.00956248306F,
        -0.0184046943F,  -0.0539142117F, -0.0149611915F,  0.0120107178F,  -0.0188389514F,
        -0.0169273075F,  -0.0420818143F, -0.0854578316F,  -0.0650615692F, -0.0180856362F,
        0.0691544861F,   0.101275779F,   -0.125339001F,   -0.0757046118F, -0.051487457F,
        -0.0780821517F,  -0.0324785449F, 0.0290223267F,   0.0805566907F,  0.0940441936F,
        -0.181305304F,   -0.111306973F,  -0.0264466833F,  -0.0161780864F, 0.0360641964F,
        0.0886478499F,   0.079203248F,   0.0476834774F,   -0.10472279F,   -0.0285298564F,
        0.0673353001F,   0.0778545141F,  0.090419434F,    0.115278535F,   0.0521779545F,
        -0.00442372914F, -0.0455354899F, 0.0334531851F,   0.108794987F,   0.0979398489F,
        0.0719335973F,   0.0896232352F,  -0.00281765568F, -0.0634050444F,
    };
    const InfiniteDiffusionLaplacianResult result =
        runInfiniteDiffusionLaplacian(residual, 32, 32, lowFrequency, 8, 8, 2.0F);
    REQUIRE(result.lowFrequency.size() == EXPECTED_LOW_FREQUENCY.size());
    for (size_t index = 0; index < result.lowFrequency.size(); ++index) {
        CAPTURE(index);
        CHECK(result.lowFrequency[index] ==
              Catch::Approx(EXPECTED_LOW_FREQUENCY[index]).margin(2.0E-6));
    }
    static constexpr std::array<std::pair<size_t, float>, 11> EXPECTED_DECODED{{
        {0, -1.24971223F},
        {1, 0.0502877757F},
        {32, 0.450287789F},
        {3 * 32 + 7, 1.25614321F},
        {7 * 32 + 3, -0.0600683689F},
        {12 * 32 + 12, -0.250504613F},
        {15 * 32 + 16, 1.33596361F},
        {16 * 32 + 15, -1.16729152F},
        {23 * 32 + 27, 0.353721708F},
        {30 * 32 + 30, -1.36340499F},
        {31 * 32 + 31, -1.26340508F},
    }};
    for (const auto& [index, expected] : EXPECTED_DECODED) {
        CAPTURE(index);
        CHECK(result.decoded[index] == Catch::Approx(expected).margin(2.0E-6));
    }
}

TEST_CASE("Production runtime fails closed before verified compilation", "[terrain-runtime]") {
    TempDir directory("terrain_runtime_closed");
    ProductionTerrainRuntime runtime(91);
    const TerrainRuntimeInferenceResult early = runtime.runModel(
        TerrainRuntimeModel::Coarse,
        std::vector<TerrainRuntimeTensor>{{.name = "x", .shape = {1}, .values = {0.0F}}});
    CHECK_FALSE(early.succeeded);
    CHECK_FALSE(runtime.qualifiedGenerationFingerprint());
    CHECK_FALSE(runtime.qualifiedGenerationContext());

    bootstrap::TerrainBootstrapCancellation cancellation;
    const bootstrap::TerrainRuntimeStepResult compilation =
        runtime.compile(directory.path(), cancellation);
    CHECK_FALSE(compilation.succeeded);
    CHECK(compilation.failure.code == bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation);
    CHECK_FALSE(runtime.qualifiedGenerationFingerprint());
    CHECK_FALSE(runtime.qualifiedGenerationContext());
}

TEST_CASE("Repeated runtime construction failures retain no instance resources",
          "[terrain-runtime][runtime-library][shutdown][regression]") {
    TempDir directory("terrain_runtime_repeated_failure");
    const TerrainRuntimeLibraryMetrics libraryBefore = terrainRuntimeLibraryMetrics();

    for (uint64_t attempt = 0; attempt < 8; ++attempt) {
        ProductionTerrainRuntime runtime(0x5259'0000'0000'0000ULL + attempt);
        bootstrap::TerrainBootstrapCancellation cancellation;
        const bootstrap::TerrainRuntimeStepResult compilation =
            runtime.compile(directory.path(), cancellation);
        CAPTURE(attempt, compilation.failure.message);
        CHECK_FALSE(compilation.succeeded);
        CHECK(compilation.failure.code ==
              bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation);
        CHECK(runtime.metrics().residentSessions == 0);
    }

    const TerrainRuntimeLibraryMetrics libraryAfter = terrainRuntimeLibraryMetrics();
    CHECK(libraryAfter.loadAttempts == libraryBefore.loadAttempts);
    CHECK(libraryAfter.successfulLoads == libraryBefore.successfulLoads);
    CHECK(libraryAfter.reuseCount == libraryBefore.reuseCount);
    CHECK(libraryAfter.resident == libraryBefore.resident);
}

TEST_CASE("Verified ONNX library survives repeated retry and runtime teardown",
          "[terrain-runtime][real-model][runtime-library][shutdown]"
          "[.real-runtime-lifetime]") {
    const char* enabledEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_LIFETIME");
    if (enabledEnvironment == nullptr || std::string_view(enabledEnvironment) != "1")
        SKIP("Set RYCRAFT_TERRAIN_REAL_LIFETIME=1 to run the runtime lifetime regression");
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");

    TempDir directory("terrain_runtime_real_lifetime");
    const std::filesystem::path stagedPack = stageRealModelPack(packEnvironment, directory.path());
    const std::filesystem::path baseModel = stagedPack / "base_model.onnx";
    const std::filesystem::path heldBaseModel = stagedPack / "base_model.onnx.held";
    const TerrainRuntimeLibraryMetrics libraryBefore = terrainRuntimeLibraryMetrics();
    bootstrap::TerrainBootstrapCancellation cancellation;

    {
        ProductionTerrainRuntime runtime(42);
        REQUIRE(runtime.qualifyPlatform().succeeded);
        std::error_code renameError;
        std::filesystem::rename(baseModel, heldBaseModel, renameError);
        CAPTURE(baseModel, heldBaseModel, renameError.message());
        REQUIRE_FALSE(renameError);

        const bootstrap::TerrainRuntimeStepResult failedCompilation =
            runtime.compile(stagedPack, cancellation);

        std::error_code restoreError;
        std::filesystem::rename(heldBaseModel, baseModel, restoreError);
        CAPTURE(restoreError.message(), failedCompilation.failure.message);
        REQUIRE_FALSE(restoreError);
        CHECK_FALSE(failedCompilation.succeeded);
        CHECK(failedCompilation.failure.code ==
              bootstrap::TerrainBootstrapFailureCode::RuntimeCompilation);
        CHECK(runtime.metrics().residentSessions == 0);

        const bootstrap::TerrainRuntimeStepResult retry = runtime.compile(stagedPack, cancellation);
        CAPTURE(retry.failure.message);
        REQUIRE(retry.succeeded);
        CHECK(runtime.metrics().residentSessions == 3);
    }

    constexpr uint64_t REPEATED_RUNTIME_COUNT = 3;
    for (uint64_t repetition = 0; repetition < REPEATED_RUNTIME_COUNT; ++repetition) {
        ProductionTerrainRuntime runtime(42 + repetition);
        REQUIRE(runtime.qualifyPlatform().succeeded);
        const bootstrap::TerrainRuntimeStepResult compilation =
            runtime.compile(stagedPack, cancellation);
        CAPTURE(repetition, compilation.failure.message);
        REQUIRE(compilation.succeeded);
        CHECK(runtime.metrics().residentSessions == 3);
    }

    const TerrainRuntimeLibraryMetrics libraryAfter = terrainRuntimeLibraryMetrics();
    CHECK(libraryAfter.resident);
    CHECK(libraryAfter.reuseCount >= libraryBefore.reuseCount + 1 + REPEATED_RUNTIME_COUNT);
    if (libraryBefore.resident) {
        CHECK(libraryAfter.loadAttempts == libraryBefore.loadAttempts);
        CHECK(libraryAfter.successfulLoads == libraryBefore.successfulLoads);
    } else {
        CHECK(libraryAfter.loadAttempts == libraryBefore.loadAttempts + 1);
        CHECK(libraryAfter.successfulLoads == libraryBefore.successfulLoads + 1);
    }
}

TEST_CASE("InfiniteDiffusion coarse conditioning matches the Python and Java half-pixel grid",
          "[terrain-runtime][pipeline-golden][conditioning][coordinates]") {
    const auto ramp = [](int64_t coarseRow, int64_t coarseColumn) {
        std::array<float, INFINITE_DIFFUSION_COARSE_CONDITIONING_VALUES> fields{};
        constexpr size_t AREA = INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE *
                                INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE;
        for (size_t channel = 0; channel < INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS;
             ++channel) {
            for (size_t row = 0; row < INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE; ++row) {
                for (size_t column = 0; column < INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE;
                     ++column) {
                    const float globalRow =
                        static_cast<float>(coarseRow - 1 + static_cast<int64_t>(row));
                    const float globalColumn =
                        static_cast<float>(coarseColumn - 1 + static_cast<int64_t>(column));
                    fields[channel * AREA + row * INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE +
                           column] =
                        static_cast<float>(channel * 100) + globalRow * 10.0F + globalColumn;
                }
            }
        }
        return fields;
    };
    const auto requireGolden = [](const auto& actual, const std::array<float, 6>& expected) {
        for (size_t channel = 0; channel < expected.size(); ++channel) {
            CAPTURE(channel, actual[channel], expected[channel]);
            REQUIRE(actual[channel] == Catch::Approx(expected[channel]).margin(2.0E-5F));
        }
    };

    // These physical coarse-ramp values were recorded from the local Python
    // grid_sample(..., align_corners=False) and Java bilinear reference paths.
    // The former +1.0 phase yields values 5.5 units larger on this ramp.
    const auto positive = ramp(0, 0);
    requireGolden(interpolateInfiniteDiffusionCoarseConditioning(positive, 0, 0, 0, 0),
                  {-5.478515625F, 94.521484375F, 194.521484375F, 294.521484375F, 394.521484375F,
                   494.521484375F});
    requireGolden(interpolateInfiniteDiffusionCoarseConditioning(positive, 255, 255, 0, 0),
                  {5.478515625F, 105.478515625F, 205.478515625F, 305.478515625F, 405.478515625F,
                   505.478515625F});

    const auto negative = ramp(-1, -1);
    requireGolden(interpolateInfiniteDiffusionCoarseConditioning(negative, -256, -1, -1, -1),
                  {-15.482421875F, 84.517578125F, 184.517578125F, 284.517578125F, 384.517578125F,
                   484.517578125F});
    requireGolden(interpolateInfiniteDiffusionCoarseConditioning(negative, -1, -256, -1, -1),
                  {-6.517578125F, 93.482421875F, 193.482421875F, 293.482421875F, 393.482421875F,
                   493.482421875F});

    const auto leftOfOrigin =
        interpolateInfiniteDiffusionCoarseConditioning(negative, -1, -1, -1, -1);
    const auto atOrigin = interpolateInfiniteDiffusionCoarseConditioning(positive, 0, 0, 0, 0);
    for (size_t channel = 0; channel < atOrigin.size(); ++channel) {
        CAPTURE(channel, leftOfOrigin[channel], atOrigin[channel]);
        CHECK(atOrigin[channel] - leftOfOrigin[channel] ==
              Catch::Approx(11.0F / 256.0F).margin(2.0E-5F));
    }
}

TEST_CASE("InfiniteDiffusion preview uses deterministic Base lineage without Decoder calls",
          "[terrain-runtime][preview][lineage]") {
    TempDir directory("terrain_runtime_backend");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    auto executor = std::make_shared<ZeroTerrainExecutor>();
    constexpr uint64_t SEED = 0x1'0000'0001ULL;
    std::shared_ptr<learned::TerrainInferenceBackend> backend =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    const learned::TerrainPageKey key{
        .quality = learned::AuthorityQuality::PREVIEW,
        .coordinate = {.row = -1, .column = 0},
    };

    const learned::CoarseSpawnRegion coarseRegion{
        .rowBegin = -2,
        .columnBegin = -1,
        .rowEnd = 2,
        .columnEnd = 3,
    };
    auto coarse = backend->inferCoarseSpawnGrid(identity, coarseRegion);
    REQUIRE(coarse.isReady());
    REQUIRE(coarse.value()->valid());
    REQUIRE(coarse.value()->region == coarseRegion);
    REQUIRE(coarse.value()->elevationMeters.size() == 16);
    REQUIRE(std::ranges::all_of(coarse.value()->elevationMeters,
                                [](float elevation) { return std::isfinite(elevation); }));
    const uint64_t callsAfterCoarse = executor->calls.load();
    CHECK(callsAfterCoarse == 80);

    auto first = backend->inferPage(identity, key);
    REQUIRE(first.isReady());
    REQUIRE(first.value()->valid());
    CHECK(first.value()->matches(identity));
    CHECK(first.value()->key == key);
    const std::span<const uint8_t> pageBytes{
        reinterpret_cast<const uint8_t*>(first.value()->samples.data()),
        first.value()->samples.size() * sizeof(learned::QuantizedTerrainSample)};
    const std::string pageHash = learned::sha256Hex(learned::sha256(pageBytes));
    INFO(pageHash);
    CHECK(pageHash == "f51cc9457bb9d41371935862cf869abfe3b5131e82cadf7c3fc2e341b8929274");
    const uint64_t callsAfterFirst = executor->calls.load();
    CHECK(callsAfterFirst > callsAfterCoarse);
    CHECK(executor->callCount(TerrainRuntimeModel::Base) > 0);
    CHECK(executor->callCount(TerrainRuntimeModel::Decoder) == 0);

    auto repeated = backend->inferPage(identity, key);
    REQUIRE(repeated.isReady());
    CHECK(repeated.value()->samples == first.value()->samples);
    CHECK(executor->calls.load() == callsAfterFirst);

    learned::GenerationIdentity wrongIdentity = identity;
    ++wrongIdentity.seed;
    auto rejected = backend->inferPage(wrongIdentity, key);
    REQUIRE_FALSE(rejected.isReady());
    REQUIRE(rejected.failure());
    CHECK(rejected.failure()->code == learned::GenerationFailureCode::INFERENCE_FAILED);
    CHECK_FALSE(rejected.failure()->retriable);

    {
        std::ofstream invalid(pack / "world_pipeline_config.json", std::ios::trunc);
        invalid << R"({"latent_compression":4})";
    }
    auto unusedExecutor = std::make_shared<ZeroTerrainExecutor>();
    auto incompatibleBackend = makeInfiniteDiffusionTerrainBackend(SEED, pack, unusedExecutor);
    auto incompatible = incompatibleBackend->inferPage(identity, key);
    REQUIRE_FALSE(incompatible.isReady());
    REQUIRE(incompatible.failure());
    CHECK_FALSE(incompatible.failure()->retriable);
    CHECK(unusedExecutor->calls.load() == 0);
}

TEST_CASE("InfiniteDiffusion attributes inference to its authority phase",
          "[terrain-runtime][pipeline][phase][diagnostics][regression]") {
    TempDir directory("terrain_runtime_phase_attribution");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0xA77B'1B07'10A0'0001ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    auto executor = std::make_shared<ZeroTerrainExecutor>();

    const auto backendForPhase = [&] {
        return makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
    };
    const auto dryBackend = backendForPhase();
    REQUIRE(dryBackend
                ->inferCoarseSpawnGridForRequest(
                    identity, {.rowBegin = -2, .columnBegin = -2, .rowEnd = 2, .columnEnd = 2},
                    learned::AuthorityRequestPriority::SPAWN)
                .isReady());

    const auto previewBackend = backendForPhase();
    REQUIRE(previewBackend
                ->inferPageForRequest(identity,
                                      {.quality = learned::AuthorityQuality::PREVIEW,
                                       .coordinate = {.row = 48, .column = -48}},
                                      learned::AuthorityRequestPriority::COARSE_PREVIEW)
                .isReady());

    const auto spawnBackend = backendForPhase();
    REQUIRE(spawnBackend
                ->inferPageForRequest(identity,
                                      {.quality = learned::AuthorityQuality::FINAL,
                                       .coordinate = {.row = -32, .column = 32}},
                                      learned::AuthorityRequestPriority::SPAWN)
                .isReady());

    const auto exactBackend = backendForPhase();
    REQUIRE(exactBackend
                ->inferPageForRequest(identity,
                                      {.quality = learned::AuthorityQuality::FINAL,
                                       .coordinate = {.row = 64, .column = -64}},
                                      learned::AuthorityRequestPriority::EXPLORATION_EXACT)
                .isReady());

    const auto protectedBackend = backendForPhase();
    REQUIRE(
        protectedBackend
            ->inferFinalNativeGridForRequest(identity,
                                             {.rowBegin = 16'382,
                                              .columnBegin = -16'386,
                                              .rowEnd = 16'398,
                                              .columnEnd = -16'370},
                                             learned::AuthorityRequestPriority::PROTECTED_HANDOFF)
            .isReady());

    const auto visibleFinalBackend = backendForPhase();
    REQUIRE(visibleFinalBackend
                ->inferFinalNativeGridForRequest(
                    identity,
                    {.rowBegin = -32'770,
                     .columnBegin = 32'766,
                     .rowEnd = -32'754,
                     .columnEnd = 32'782},
                    learned::AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT)
                .isReady());

    using Phase = TerrainRuntimeInferencePhase;
    CHECK(executor->phaseCount(Phase::Qualification) == 0);
    CHECK(executor->phaseCount(Phase::Other) == 0);
    CHECK(executor->phaseModelCount(Phase::DrySpawnCoarseSearch, TerrainRuntimeModel::Coarse) > 0);
    CHECK(executor->phaseModelCount(Phase::DrySpawnCoarseSearch, TerrainRuntimeModel::Base) == 0);
    CHECK(executor->phaseModelCount(Phase::DrySpawnCoarseSearch, TerrainRuntimeModel::Decoder) ==
          0);
    CHECK(executor->phaseModelCount(Phase::HorizonPreview, TerrainRuntimeModel::Coarse) > 0);
    CHECK(executor->phaseModelCount(Phase::HorizonPreview, TerrainRuntimeModel::Base) > 0);
    CHECK(executor->phaseModelCount(Phase::HorizonPreview, TerrainRuntimeModel::Decoder) == 0);
    for (const Phase phase : {Phase::FinalSpawnCertification, Phase::ExplorationExact,
                              Phase::ProtectedFinal, Phase::VisibleFinalRefinement}) {
        CAPTURE(static_cast<unsigned>(phase));
        CHECK(executor->phaseModelCount(phase, TerrainRuntimeModel::Coarse) > 0);
        CHECK(executor->phaseModelCount(phase, TerrainRuntimeModel::Base) > 0);
        CHECK(executor->phaseModelCount(phase, TerrainRuntimeModel::Decoder) > 0);
    }
}

TEST_CASE("InfiniteDiffusion decoder batches deterministically repeat the final tail window",
          "[terrain-runtime][pipeline][decoder][batch][determinism]") {
    TempDir directory("terrain_runtime_decoder_batch_tail");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0xDEC0'DE40'0000'0001ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);

    auto executor = std::make_shared<ZeroTerrainExecutor>(true);
    const auto backend = makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
    const auto page = backend->inferPage(identity, {.quality = learned::AuthorityQuality::FINAL,
                                                    .coordinate = {.row = 0, .column = 0}});
    REQUIRE(page.isReady());
    CHECK(executor->callCount(TerrainRuntimeModel::Decoder) == 3);
    CHECK(executor->decoderBatchUniqueCounts() == std::vector<size_t>{4, 4, 1});

    const auto repeated = backend->inferPage(identity, {.quality = learned::AuthorityQuality::FINAL,
                                                        .coordinate = {.row = 0, .column = 0}});
    REQUIRE(repeated.isReady());
    CHECK(repeated.value()->samples == page.value()->samples);
    CHECK(executor->callCount(TerrainRuntimeModel::Decoder) == 3);
}

TEST_CASE("InfiniteDiffusion final page groups preserve quantized pages and fill fixed batches",
          "[terrain-runtime][pipeline][batch][determinism]") {
    TempDir directory("terrain_runtime_final_batch");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0x7B4C'9031'1020'FEEDULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    std::vector<learned::TerrainPageKey> keys;
    keys.reserve(24);
    for (int64_t row = -2; row < 2; ++row) {
        for (int64_t column = 3; column < 9; ++column) {
            keys.push_back(
                {.quality = learned::AuthorityQuality::FINAL, .coordinate = {row, column}});
        }
    }

    const auto payloadHash = [](const learned::TerrainAuthorityPage& page) {
        const std::span<const uint8_t> bytes{reinterpret_cast<const uint8_t*>(page.samples.data()),
                                             page.byteSize()};
        return learned::sha256(bytes);
    };

    auto serialExecutor = std::make_shared<ZeroTerrainExecutor>();
    const std::shared_ptr<learned::TerrainInferenceBackend> serial =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, serialExecutor);
    std::vector<learned::Sha256Digest> serialHashes(keys.size());
    for (size_t index = 0; index < keys.size(); ++index) {
        const auto page = serial->inferPage(identity, keys[index]);
        REQUIRE(page.isReady());
        serialHashes[index] = payloadHash(*page.value());
    }
    const uint64_t serialBaseCalls = serialExecutor->callCount(TerrainRuntimeModel::Base);

    auto groupedExecutor = std::make_shared<ZeroTerrainExecutor>();
    const std::shared_ptr<learned::TerrainInferenceBackend> grouped =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, groupedExecutor);
    const auto pages = grouped->inferPages(identity, keys);
    REQUIRE(pages.isReady());
    REQUIRE(pages.value()->size() == keys.size());
    for (size_t index = 0; index < keys.size(); ++index) {
        REQUIRE(pages.value()->at(index).key == keys[index]);
        CHECK(payloadHash(pages.value()->at(index)) == serialHashes[index]);
    }
    const uint64_t groupedBaseCalls = groupedExecutor->callCount(TerrainRuntimeModel::Base);
    INFO("serial Base calls=" << serialBaseCalls << " grouped Base calls=" << groupedBaseCalls);
    CHECK(serialBaseCalls == 54);
    CHECK(groupedBaseCalls == 32);
    CHECK(groupedBaseCalls < serialBaseCalls);

    // The coordinator can begin one page while the spawn prequeue is still
    // admitting the rest. The remaining closure still shares its union of
    // windows and must retain the page bytes of a fully serial request.
    auto splitExecutor = std::make_shared<ZeroTerrainExecutor>();
    const std::shared_ptr<learned::TerrainInferenceBackend> split =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, splitExecutor);
    const auto firstSplitPage = split->inferPage(identity, keys.front());
    REQUIRE(firstSplitPage.isReady());
    CHECK(payloadHash(*firstSplitPage.value()) == serialHashes.front());
    const std::vector<learned::TerrainPageKey> tailKeys(keys.begin() + 1, keys.end());
    const auto splitTailPages = split->inferPages(identity, tailKeys);
    REQUIRE(splitTailPages.isReady());
    for (size_t index = 0; index < tailKeys.size(); ++index)
        CHECK(payloadHash(splitTailPages.value()->at(index)) == serialHashes[index + 1]);
    const uint64_t splitBaseCalls = splitExecutor->callCount(TerrainRuntimeModel::Base);
    INFO("first-page-plus-grouped-tail Base calls=" << splitBaseCalls);
    CHECK(splitBaseCalls == 32);
    CHECK(splitBaseCalls < serialBaseCalls);

    auto reverseExecutor = std::make_shared<ZeroTerrainExecutor>();
    const std::shared_ptr<learned::TerrainInferenceBackend> reverse =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, reverseExecutor);
    std::vector<learned::TerrainPageKey> reverseKeys(keys.rbegin(), keys.rend());
    const auto reversed = reverse->inferPages(identity, reverseKeys);
    REQUIRE(reversed.isReady());
    for (size_t reverseIndex = 0; reverseIndex < reverseKeys.size(); ++reverseIndex) {
        const auto expected = std::find(keys.begin(), keys.end(), reverseKeys[reverseIndex]);
        REQUIRE(expected != keys.end());
        const size_t expectedIndex = static_cast<size_t>(expected - keys.begin());
        CHECK(payloadHash(reversed.value()->at(reverseIndex)) == serialHashes[expectedIndex]);
    }
}

TEST_CASE("InfiniteDiffusion preview page groups union Base windows without Decoder inference",
          "[terrain-runtime][pipeline][preview][batch][determinism]") {
    TempDir directory("terrain_runtime_preview_batch");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0x5052'4556'4945'5709ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    std::vector<learned::TerrainPageKey> keys;
    for (int64_t row = -3; row < 1; ++row) {
        for (int64_t column = -5; column < 1; ++column) {
            keys.push_back(
                {.quality = learned::AuthorityQuality::PREVIEW, .coordinate = {row, column}});
        }
    }
    const auto payloadHash = [](const learned::TerrainAuthorityPage& page) {
        const std::span<const uint8_t> bytes{reinterpret_cast<const uint8_t*>(page.samples.data()),
                                             page.byteSize()};
        return learned::sha256(bytes);
    };

    auto serialExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto serial = makeInfiniteDiffusionTerrainBackend(SEED, pack, serialExecutor);
    std::vector<learned::Sha256Digest> serialHashes;
    serialHashes.reserve(keys.size());
    for (const learned::TerrainPageKey key : keys) {
        const auto page = serial->inferPage(identity, key);
        REQUIRE(page.isReady());
        serialHashes.push_back(payloadHash(*page.value()));
    }
    const uint64_t serialBaseCalls = serialExecutor->callCount(TerrainRuntimeModel::Base);
    CHECK(serialExecutor->callCount(TerrainRuntimeModel::Decoder) == 0);

    auto groupedExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto grouped = makeInfiniteDiffusionTerrainBackend(SEED, pack, groupedExecutor);
    std::vector<learned::TerrainPageKey> requested(keys.rbegin(), keys.rend());
    const auto pages = grouped->inferPages(identity, requested);
    REQUIRE(pages.isReady());
    REQUIRE(pages.value()->size() == requested.size());
    for (size_t index = 0; index < requested.size(); ++index) {
        REQUIRE(pages.value()->at(index).key == requested[index]);
        const auto expected = std::find(keys.begin(), keys.end(), requested[index]);
        REQUIRE(expected != keys.end());
        CHECK(payloadHash(pages.value()->at(index)) ==
              serialHashes[static_cast<size_t>(expected - keys.begin())]);
    }
    const uint64_t groupedBaseCalls = groupedExecutor->callCount(TerrainRuntimeModel::Base);
    INFO("serial Base calls=" << serialBaseCalls << " grouped Base calls=" << groupedBaseCalls);
    CHECK(groupedBaseCalls < serialBaseCalls);
    CHECK(groupedExecutor->callCount(TerrainRuntimeModel::Decoder) == 0);

    // The backend accepts one coordinator batch containing both immutable
    // qualities, preserves caller order, and reuses Base tensors between them.
    auto mixedExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto mixed = makeInfiniteDiffusionTerrainBackend(SEED, pack, mixedExecutor);
    const std::array<learned::TerrainPageKey, 4> mixedKeys{{
        keys[0],
        {.quality = learned::AuthorityQuality::FINAL, .coordinate = keys[0].coordinate},
        keys[7],
        {.quality = learned::AuthorityQuality::FINAL, .coordinate = keys[7].coordinate},
    }};
    const auto mixedPages = mixed->inferPages(identity, mixedKeys);
    REQUIRE(mixedPages.isReady());
    REQUIRE(mixedPages.value()->size() == mixedKeys.size());
    for (size_t index = 0; index < mixedKeys.size(); ++index)
        CHECK(mixedPages.value()->at(index).key == mixedKeys[index]);
    CHECK(mixedExecutor->callCount(TerrainRuntimeModel::Base) > 0);
    CHECK(mixedExecutor->callCount(TerrainRuntimeModel::Decoder) > 0);
}

TEST_CASE(
    "InfiniteDiffusion transient hydrology rectangles preserve final samples with fewer calls",
    "[terrain-runtime][pipeline][hydrology][transient][performance][determinism]") {
    TempDir directory("terrain_runtime_transient_hydrology");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0x4859'4452'4F4C'4F47ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);

    std::vector<learned::TerrainPageKey> closure;
    for (int64_t row = -1; row <= 2; ++row) {
        for (int64_t column = -1; column <= 2; ++column) {
            closure.push_back(
                {.quality = learned::AuthorityQuality::FINAL, .coordinate = {row, column}});
        }
    }
    auto pageExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto pageBackend = makeInfiniteDiffusionTerrainBackend(SEED, pack, pageExecutor);
    const auto pages = pageBackend->inferPages(identity, closure);
    REQUIRE(pages.isReady());
    REQUIRE(pages.value()->size() == closure.size());

    // A 2,048-block owner has 513 core raster samples plus a two-sample
    // apron on both sides: native coordinates [-2, 515) on each axis.
    constexpr learned::NativeRect HYDROLOGY_RASTER{
        .rowBegin = -2,
        .columnBegin = -2,
        .rowEnd = 515,
        .columnEnd = 515,
    };
    auto transientExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto transientBackend =
        makeInfiniteDiffusionTerrainBackend(SEED, pack, transientExecutor);
    const auto transient = transientBackend->inferFinalNativeGrid(identity, HYDROLOGY_RASTER);
    REQUIRE(transient.isReady());
    REQUIRE(transient.value()->valid());

    std::map<learned::TerrainPageCoordinate, const learned::TerrainAuthorityPage*> pagesByKey;
    for (const learned::TerrainAuthorityPage& page : *pages.value())
        pagesByKey.emplace(page.key.coordinate, &page);
    std::vector<learned::PhysicalTerrainSample> expectedSamples;
    expectedSamples.reserve(transient.value()->samples.size());
    for (int64_t row = HYDROLOGY_RASTER.rowBegin; row < HYDROLOGY_RASTER.rowEnd; ++row) {
        for (int64_t column = HYDROLOGY_RASTER.columnBegin; column < HYDROLOGY_RASTER.columnEnd;
             ++column) {
            const learned::TerrainPageCoordinate coordinate{
                .row = learned::floorDivide(row, learned::AUTHORITY_PAGE_NATIVE_EDGE),
                .column = learned::floorDivide(column, learned::AUTHORITY_PAGE_NATIVE_EDGE),
            };
            const auto found = pagesByKey.find(coordinate);
            if (found == pagesByKey.end())
                FAIL("Persistent hydrology closure lost a page");
            int64_t localRow = row % learned::AUTHORITY_PAGE_NATIVE_EDGE;
            int64_t localColumn = column % learned::AUTHORITY_PAGE_NATIVE_EDGE;
            if (localRow < 0)
                localRow += learned::AUTHORITY_PAGE_NATIVE_EDGE;
            if (localColumn < 0)
                localColumn += learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const size_t localIndex =
                static_cast<size_t>(localRow) * learned::AUTHORITY_PAGE_NATIVE_EDGE +
                static_cast<size_t>(localColumn);
            expectedSamples.push_back(
                learned::dequantizeTerrainSample(found->second->samples[localIndex]));
        }
    }
    CHECK(transient.value()->samples == expectedSamples);

    const uint64_t pageBaseCalls = pageExecutor->callCount(TerrainRuntimeModel::Base);
    const uint64_t pageDecoderCalls = pageExecutor->callCount(TerrainRuntimeModel::Decoder);
    const uint64_t transientBaseCalls = transientExecutor->callCount(TerrainRuntimeModel::Base);
    const uint64_t transientDecoderCalls =
        transientExecutor->callCount(TerrainRuntimeModel::Decoder);
    INFO("persistent closure calls: base=" << pageBaseCalls << " decoder=" << pageDecoderCalls);
    INFO("transient raster calls: base=" << transientBaseCalls
                                         << " decoder=" << transientDecoderCalls);
    CHECK(transientBaseCalls == 14);
    CHECK(transientDecoderCalls == 4);
    CHECK(transientBaseCalls < pageBaseCalls);
    CHECK(transientDecoderCalls < pageDecoderCalls);
}

TEST_CASE(
    "InfiniteDiffusion grouped protected hydrology owners preserve exact final samples",
    "[terrain-runtime][pipeline][hydrology][transient][protected][performance][determinism]") {
    TempDir directory("terrain_runtime_grouped_protected_hydrology");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0x4752'4F55'5045'4404ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);

    constexpr int64_t OWNER_NATIVE_EDGE = 2 * learned::AUTHORITY_PAGE_NATIVE_EDGE;
    constexpr int64_t OWNER_APRON = 2;
    constexpr int64_t OWNER_SAMPLE_EDGE = OWNER_NATIVE_EDGE + 2 * OWNER_APRON + 1;
    const auto ownerRegion = [](int64_t ownerRow, int64_t ownerColumn) {
        const int64_t rowBegin = ownerRow * OWNER_NATIVE_EDGE - OWNER_APRON;
        const int64_t columnBegin = ownerColumn * OWNER_NATIVE_EDGE - OWNER_APRON;
        return learned::NativeRect{
            .rowBegin = rowBegin,
            .columnBegin = columnBegin,
            .rowEnd = rowBegin + OWNER_SAMPLE_EDGE,
            .columnEnd = columnBegin + OWNER_SAMPLE_EDGE,
        };
    };
    constexpr int64_t FIRST_OWNER_ROW = -17;
    constexpr int64_t FIRST_OWNER_COLUMN = 29;
    const std::array<learned::NativeRect, 4> owners{{
        ownerRegion(FIRST_OWNER_ROW, FIRST_OWNER_COLUMN),
        ownerRegion(FIRST_OWNER_ROW, FIRST_OWNER_COLUMN + 1),
        ownerRegion(FIRST_OWNER_ROW + 1, FIRST_OWNER_COLUMN),
        ownerRegion(FIRST_OWNER_ROW + 1, FIRST_OWNER_COLUMN + 1),
    }};
    const learned::NativeRect groupedRegion{
        .rowBegin = owners[0].rowBegin,
        .columnBegin = owners[0].columnBegin,
        .rowEnd = owners[3].rowEnd,
        .columnEnd = owners[3].columnEnd,
    };
    REQUIRE(groupedRegion.rowBegin < 0);
    REQUIRE(groupedRegion.columnBegin > 0);
    REQUIRE(groupedRegion.height() == 1'029);
    REQUIRE(groupedRegion.width() == 1'029);

    std::array<learned::PhysicalTerrainGrid, 4> individual;
    uint64_t separateBaseCalls = 0;
    uint64_t separateDecoderCalls = 0;
    for (size_t index = 0; index < owners.size(); ++index) {
        REQUIRE(owners[index].height() == 517);
        REQUIRE(owners[index].width() == 517);
        auto executor = std::make_shared<ZeroTerrainExecutor>();
        const auto backend = makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
        const auto result = backend->inferFinalNativeGridForRequest(
            identity, owners[index], learned::AuthorityRequestPriority::PROTECTED_HANDOFF);
        REQUIRE(result.isReady());
        REQUIRE(result.value()->valid());
        REQUIRE(result.value()->region == owners[index]);
        individual[index] = *result.value();
        separateBaseCalls += executor->callCount(TerrainRuntimeModel::Base);
        separateDecoderCalls += executor->callCount(TerrainRuntimeModel::Decoder);
    }

    auto groupedExecutor = std::make_shared<ZeroTerrainExecutor>();
    const auto groupedBackend = makeInfiniteDiffusionTerrainBackend(SEED, pack, groupedExecutor);
    const auto grouped = groupedBackend->inferFinalNativeGridForRequest(
        identity, groupedRegion, learned::AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(grouped.isReady());
    REQUIRE(grouped.value()->valid());
    REQUIRE(grouped.value()->region == groupedRegion);
    for (size_t index = 0; index < owners.size(); ++index) {
        std::vector<learned::PhysicalTerrainSample> crop;
        crop.reserve(individual[index].samples.size());
        for (int64_t row = owners[index].rowBegin; row < owners[index].rowEnd; ++row) {
            const learned::PhysicalTerrainSample* first =
                grouped.value()->sample(row, owners[index].columnBegin);
            REQUIRE(first != nullptr);
            crop.insert(crop.end(), first,
                        first + static_cast<std::ptrdiff_t>(owners[index].width()));
        }
        REQUIRE(crop == individual[index].samples);
    }

    const uint64_t groupedBaseCalls = groupedExecutor->callCount(TerrainRuntimeModel::Base);
    const uint64_t groupedDecoderCalls = groupedExecutor->callCount(TerrainRuntimeModel::Decoder);
    INFO("separate owner calls: Base=" << separateBaseCalls << " Decoder=" << separateDecoderCalls);
    INFO("grouped owner calls: Base=" << groupedBaseCalls << " Decoder=" << groupedDecoderCalls);
    CHECK(separateBaseCalls == 56);
    CHECK(separateDecoderCalls == 21);
    CHECK(groupedBaseCalls == 26);
    CHECK(groupedDecoderCalls == 13);
    CHECK(groupedBaseCalls < separateBaseCalls);
    CHECK(groupedDecoderCalls < separateDecoderCalls);
}

TEST_CASE("InfiniteDiffusion hydrology owner phases retain bounded final call counts",
          "[terrain-runtime][pipeline][hydrology][transient][performance]") {
    TempDir directory("terrain_runtime_transient_hydrology_phases");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 0x5048'4153'4553'0001ULL;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);

    uint64_t minimumBaseCalls = std::numeric_limits<uint64_t>::max();
    uint64_t maximumBaseCalls = 0;
    uint64_t minimumDecoderCalls = std::numeric_limits<uint64_t>::max();
    uint64_t maximumDecoderCalls = 0;
    for (int64_t ownerRow = 0; ownerRow < 3; ++ownerRow) {
        for (int64_t ownerColumn = 0; ownerColumn < 3; ++ownerColumn) {
            constexpr int64_t OWNER_NATIVE_EDGE = 2 * learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const learned::NativeRect region{
                .rowBegin = ownerRow * OWNER_NATIVE_EDGE - 2,
                .columnBegin = ownerColumn * OWNER_NATIVE_EDGE - 2,
                .rowEnd = ownerRow * OWNER_NATIVE_EDGE + 515,
                .columnEnd = ownerColumn * OWNER_NATIVE_EDGE + 515,
            };
            auto executor = std::make_shared<ZeroTerrainExecutor>();
            const auto backend = makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
            const auto result = backend->inferFinalNativeGrid(identity, region);
            REQUIRE(result.isReady());
            const uint64_t baseCalls = executor->callCount(TerrainRuntimeModel::Base);
            const uint64_t decoderCalls = executor->callCount(TerrainRuntimeModel::Decoder);
            INFO("owner phase row=" << ownerRow << " column=" << ownerColumn
                                    << " base=" << baseCalls << " decoder=" << decoderCalls);
            minimumBaseCalls = std::min(minimumBaseCalls, baseCalls);
            maximumBaseCalls = std::max(maximumBaseCalls, baseCalls);
            minimumDecoderCalls = std::min(minimumDecoderCalls, decoderCalls);
            maximumDecoderCalls = std::max(maximumDecoderCalls, decoderCalls);
        }
    }
    INFO("hydrology owner phase calls: Base " << minimumBaseCalls << ".." << maximumBaseCalls
                                              << ", Decoder " << minimumDecoderCalls << ".."
                                              << maximumDecoderCalls);
    CHECK(minimumBaseCalls == 14);
    CHECK(maximumBaseCalls == 14);
    CHECK(minimumDecoderCalls == 4);
    CHECK(maximumDecoderCalls == 7);
}

TEST_CASE("InfiniteDiffusion seed 42 spawn authority keeps bounded model call counts",
          "[terrain-runtime][pipeline][hydrology][spawn][performance]") {
    TempDir directory("terrain_runtime_spawn_calls");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 42;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    constexpr learned::TerrainPageCoordinate SEED_42_SPAWN_PAGE{.row = -5, .column = 5};

    auto executor = std::make_shared<ZeroTerrainExecutor>();
    const auto backend = makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
    const auto page = backend->inferPage(
        identity, {.quality = learned::AuthorityQuality::FINAL, .coordinate = SEED_42_SPAWN_PAGE});
    REQUIRE(page.isReady());
    CHECK(executor->callCount(TerrainRuntimeModel::Base) == 8);
    CHECK(executor->callCount(TerrainRuntimeModel::Decoder) == 3);

    constexpr learned::NativeRect OWNER_REGION{
        .rowBegin = -1'538,
        .columnBegin = 1'022,
        .rowEnd = -1'021,
        .columnEnd = 1'539,
    };
    const auto result = backend->inferFinalNativeGrid(identity, OWNER_REGION);
    REQUIRE(result.isReady());
    REQUIRE(result.value()->valid());
    const uint64_t baseCalls = executor->callCount(TerrainRuntimeModel::Base);
    const uint64_t decoderCalls = executor->callCount(TerrainRuntimeModel::Decoder);
    INFO("seed 42 spawn calls: Base=" << baseCalls << " Decoder=" << decoderCalls);
    CHECK(baseCalls == 14);
    CHECK(decoderCalls == 6);
}

TEST_CASE("Accepted seed 42 hydrology owners reuse tensors for exact refinement pages",
          "[terrain-runtime][pipeline][hydrology][spawn][reuse][performance][determinism]") {
    TempDir directory("terrain_runtime_spawn_owner_reuse");
    const std::filesystem::path pack = directory.path();
    std::filesystem::create_directories(pack);
    writeFakePipelineData(pack / "pipeline_data.json");
    writeFakeModelConfiguration(pack / "world_pipeline_config.json");
    constexpr uint64_t SEED = 42;
    const learned::GenerationIdentity identity = productionGenerationIdentity(SEED);
    constexpr learned::TerrainPageKey SPAWN_PAGE{
        .quality = learned::AuthorityQuality::FINAL,
        .coordinate = {.row = -5, .column = 5},
    };
    constexpr learned::NativeRect FIRST_OWNER{
        .rowBegin = -1'538,
        .columnBegin = 1'022,
        .rowEnd = -1'021,
        .columnEnd = 1'539,
    };
    constexpr learned::NativeRect SECOND_OWNER{
        .rowBegin = -1'026,
        .columnBegin = 1'022,
        .rowEnd = -509,
        .columnEnd = 1'539,
    };
    const std::array<learned::TerrainPageKey, 4> refinement{{
        {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -5, .column = 4}},
        SPAWN_PAGE,
        {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -4, .column = 4}},
        {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -4, .column = 5}},
    }};

    auto executor = std::make_shared<ZeroTerrainExecutor>();
    const auto backend = makeInfiniteDiffusionTerrainBackend(SEED, pack, executor);
    const auto selected = backend->inferPage(identity, SPAWN_PAGE);
    REQUIRE(selected.isReady());
    const auto firstOwner = backend->inferFinalNativeGrid(identity, FIRST_OWNER);
    REQUIRE(firstOwner.isReady());
    const uint64_t rejectedBaseCalls = executor->callCount(TerrainRuntimeModel::Base);
    const uint64_t rejectedDecoderCalls = executor->callCount(TerrainRuntimeModel::Decoder);
    REQUIRE(rejectedBaseCalls == 14);
    REQUIRE(rejectedDecoderCalls == 6);

    // Only an accepted candidate reaches the adjacent owner and exact-page
    // publication. Both owner rectangles and the pages share one global
    // deterministic tensor cache.
    const auto secondOwner = backend->inferFinalNativeGrid(identity, SECOND_OWNER);
    REQUIRE(secondOwner.isReady());
    const auto pages = backend->inferPages(identity, refinement);
    REQUIRE(pages.isReady());
    REQUIRE(pages.value()->size() == refinement.size());
    const uint64_t acceptedBaseCalls = executor->callCount(TerrainRuntimeModel::Base);
    const uint64_t acceptedDecoderCalls = executor->callCount(TerrainRuntimeModel::Decoder);
    INFO("accepted owner reuse calls: Base=" << acceptedBaseCalls
                                             << " Decoder=" << acceptedDecoderCalls);
    CHECK(acceptedBaseCalls == 20);
    CHECK(acceptedDecoderCalls == 10);

    const auto repeated = backend->inferPages(identity, refinement);
    REQUIRE(repeated.isReady());
    REQUIRE(repeated.value()->size() == refinement.size());
    for (size_t index = 0; index < refinement.size(); ++index)
        CHECK(repeated.value()->at(index).samples == pages.value()->at(index).samples);
    CHECK(executor->callCount(TerrainRuntimeModel::Base) == acceptedBaseCalls);
    CHECK(executor->callCount(TerrainRuntimeModel::Decoder) == acceptedDecoderCalls);
}

TEST_CASE("Pinned models expose a reproducible Core ML qualification digest",
          "[terrain-runtime][real-model]") {
    constexpr uint64_t SEED = 42;
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to run the local real-model qualification");

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    ProductionTerrainRuntime runtime(SEED, expected);
    bootstrap::TerrainBootstrapCancellation cancellation;
    REQUIRE(runtime.qualifyPlatform().succeeded);
    const bootstrap::TerrainRuntimeStepResult compilation =
        runtime.compile(packEnvironment, cancellation);
    CAPTURE(compilation.failure.message);
    REQUIRE(compilation.succeeded);
    const bootstrap::TerrainRuntimeStepResult qualification =
        runtime.loadAndQualify(packEnvironment, cancellation);
    const TerrainRuntimeMetrics metrics = runtime.metrics();
    REQUIRE(metrics.qualificationDigest);
    const std::string observed = learned::sha256Hex(*metrics.qualificationDigest);
    WARN("Observed terrain qualification digest: " << observed);
    WARN("Core ML qualification metrics: "
         << metrics.coreMlPartitions << " partitions, " << metrics.coreMlNodes << " nodes, "
         << metrics.cpuFallbackPartitions << " CPU fallback partitions, "
         << metrics.cpuFallbackNodes << " CPU fallback nodes");
    WARN("Qualification inference milliseconds: coarse="
         << static_cast<double>(metrics.models[0].inferenceNanoseconds) / 1'000'000.0 << " base="
         << static_cast<double>(metrics.models[1].inferenceNanoseconds) / 1'000'000.0 << " decoder="
         << static_cast<double>(metrics.models[2].inferenceNanoseconds) / 1'000'000.0);
    CAPTURE(qualification.failure.message, observed);
    REQUIRE(qualification.succeeded);
    REQUIRE(runtime.qualifiedGenerationFingerprint());
    REQUIRE(runtime.qualifiedGenerationContext());
    CHECK(runtime.qualifiedGenerationContext()->identity() == runtime.generationIdentity());
    CHECK(metrics.compiledSessions == 3);
    CHECK(metrics.residentSessions == 3);
    CHECK(metrics.peakResidentSessions == 3);
    CHECK(metrics.usesGlobalThreadPool);
    CHECK(metrics.perSessionThreadsDisabled);
    CHECK(metrics.globalIntraOpThreads >= 1);
    CHECK(metrics.globalIntraOpThreads <= MAXIMUM_CPU_FALLBACK_THREADS);
    CHECK(metrics.globalInterOpThreads == GLOBAL_INTER_OP_THREAD_COUNT);
    CHECK(metrics.cpuFallbackIntraOpThreads == metrics.globalIntraOpThreads);
    CHECK(metrics.cpuFallbackIntraOpThreads >= 1);
    CHECK(metrics.cpuFallbackIntraOpThreads <= MAXIMUM_CPU_FALLBACK_THREADS);
    for (const TerrainRuntimeMetrics::Model& model : metrics.models)
        CHECK(model.sessionCreations == 1);
    if (const char* pageEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_PAGE");
        pageEnvironment != nullptr && std::string_view(pageEnvironment) == "1") {
        TempDir pageDirectory("terrain_runtime_real_page");
        const std::filesystem::path pageProfile =
            std::filesystem::path(pageDirectory.path()) / "v4";
        std::error_code pageProfileError;
        std::filesystem::create_directories(pageProfile, pageProfileError);
        CAPTURE(pageProfile, pageProfileError.message());
        REQUIRE_FALSE(pageProfileError);
        const bootstrap::TerrainRuntimeStepResult pageBinding =
            runtime.bindWorldProfile(pageProfile);
        CAPTURE(pageBinding.failure.message);
        REQUIRE(pageBinding.succeeded);
        auto context = runtime.qualifiedGenerationContext();
        const auto pageStart = std::chrono::steady_clock::now();
        auto sample = context->sampleWorld(-4, -4);
        const auto deadline = pageStart + std::chrono::minutes(5);
        while (sample.status() == learned::AuthorityStatus::DEFERRED &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            sample = context->sampleWorld(-4, -4);
        }
        const double elapsedSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - pageStart).count();
        WARN("Generated final terrain authority sample in " << elapsedSeconds << " seconds");
        const TerrainRuntimeMetrics pageMetrics = runtime.metrics();
        WARN("Final page inference calls: "
             << pageMetrics.inferenceCalls - metrics.inferenceCalls
             << ", total Core ML partitions: " << pageMetrics.coreMlPartitions
             << ", total CPU fallback partitions: " << pageMetrics.cpuFallbackPartitions);
        CHECK(pageMetrics.residentSessions == 3);
        CHECK(pageMetrics.peakResidentSessions == 3);
        for (size_t model = 0; model < metrics.models.size(); ++model) {
            CHECK(pageMetrics.models[model].sessionCreations ==
                  metrics.models[model].sessionCreations);
            CHECK(pageMetrics.models[model].sessionCreationNanoseconds ==
                  metrics.models[model].sessionCreationNanoseconds);
        }
        static constexpr std::array<std::string_view, 3> MODEL_NAMES{
            "coarse",
            "base",
            "decoder",
        };
        for (size_t model = 0; model < MODEL_NAMES.size(); ++model) {
            const auto& before = metrics.models[model];
            const auto& after = pageMetrics.models[model];
            WARN(MODEL_NAMES[model]
                 << " page calls " << after.calls - before.calls << ", inference seconds "
                 << static_cast<double>(after.inferenceNanoseconds - before.inferenceNanoseconds) /
                        1.0E9
                 << ", maximum call seconds "
                 << static_cast<double>(after.maximumInferenceNanoseconds) / 1.0E9
                 << ", new session seconds "
                 << static_cast<double>(after.sessionCreationNanoseconds -
                                        before.sessionCreationNanoseconds) /
                        1.0E9);
        }
        REQUIRE(sample.isReady());
        CHECK(std::isfinite(sample.value()->elevationMeters));
        CHECK(std::isfinite(sample.value()->meanTemperatureC));
        CHECK(sample.value()->annualPrecipitationMm >= 0.0);
        WARN("Final sample elevation " << sample.value()->elevationMeters << " meters, temperature "
                                       << sample.value()->meanTemperatureC << " C, precipitation "
                                       << sample.value()->annualPrecipitationMm << " mm");
        learned::TerrainPageStore store(pageProfile / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
                                        runtime.generationIdentity());
        auto page = store.loadPage(
            {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -1, .column = -1}});
        REQUIRE(page.isReady());
        CHECK(runtime.generationIdentity().seed == SEED);
        CHECK(page.value()->generationSeed == SEED);
        const std::span<const uint8_t> pageBytes{
            reinterpret_cast<const uint8_t*>(page.value()->samples.data()),
            page.value()->samples.size() * sizeof(learned::QuantizedTerrainSample)};
        const std::string pageHash = learned::sha256Hex(learned::sha256(pageBytes));
        WARN("Final quantized terrain page hash: " << pageHash);
        CHECK(pageHash == "d21220e869d92ad4c20201450bcaab05ae735b5657b26a502fd56a8b69c7896a");

        // Rebinding creates a fresh authority and InfiniteDiffusion tensor
        // cache while retaining the three qualified Core ML sessions. Request
        // the eastern page first, then overlap two requests for the golden
        // page. This covers cache-cleared repeatability, reverse request order,
        // and concurrent single flight in one bounded two-page generation.
        context.reset();
        TempDir repeatedPageDirectory("terrain_runtime_real_page_repeat");
        const std::filesystem::path repeatedPageProfile =
            std::filesystem::path(repeatedPageDirectory.path()) / "v4";
        std::error_code repeatedProfileError;
        std::filesystem::create_directories(repeatedPageProfile, repeatedProfileError);
        CAPTURE(repeatedPageProfile, repeatedProfileError.message());
        REQUIRE_FALSE(repeatedProfileError);
        const bootstrap::TerrainRuntimeStepResult repeatedBinding =
            runtime.bindWorldProfile(repeatedPageProfile);
        CAPTURE(repeatedBinding.failure.message);
        REQUIRE(repeatedBinding.succeeded);
        const auto repeatedContext = runtime.qualifiedGenerationContext();
        REQUIRE(repeatedContext);
        auto repeatedNeighbor = repeatedContext->requestAuthorityPage(
            {.row = -1, .column = 0}, learned::AuthorityRequestPriority::SPAWN);
        auto repeatedFirst = repeatedContext->sampleWorld(-4, -4);
        auto repeatedSecond = repeatedContext->sampleWorld(-4, -4);
        const auto repeatedDeadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
        while ((repeatedNeighbor.status() == learned::AuthorityStatus::DEFERRED ||
                repeatedFirst.status() == learned::AuthorityStatus::DEFERRED ||
                repeatedSecond.status() == learned::AuthorityStatus::DEFERRED) &&
               std::chrono::steady_clock::now() < repeatedDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (repeatedNeighbor.status() == learned::AuthorityStatus::DEFERRED) {
                repeatedNeighbor = repeatedContext->requestAuthorityPage(
                    {.row = -1, .column = 0}, learned::AuthorityRequestPriority::SPAWN);
            }
            if (repeatedFirst.status() == learned::AuthorityStatus::DEFERRED)
                repeatedFirst = repeatedContext->sampleWorld(-4, -4);
            if (repeatedSecond.status() == learned::AuthorityStatus::DEFERRED)
                repeatedSecond = repeatedContext->sampleWorld(-4, -4);
        }
        REQUIRE(repeatedNeighbor.isReady());
        REQUIRE(repeatedFirst.isReady());
        REQUIRE(repeatedSecond.isReady());
        CHECK(*repeatedFirst.value() == *repeatedSecond.value());
        const learned::WorldGenerationMetrics repeatedContextMetrics = repeatedContext->metrics();
        CHECK(repeatedContextMetrics.authorityCache.builds == 2);
        CHECK(repeatedContextMetrics.authorityCache.diskLoads == 0);
        CHECK(repeatedContextMetrics.authorityCache.publicationWrites == 2);
        learned::TerrainPageStore repeatedStore(repeatedPageProfile /
                                                    SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY,
                                                runtime.generationIdentity());
        auto repeatedPage = repeatedStore.loadPage(
            {.quality = learned::AuthorityQuality::FINAL, .coordinate = {.row = -1, .column = -1}});
        REQUIRE(repeatedPage.isReady());
        const std::span<const uint8_t> repeatedPageBytes{
            reinterpret_cast<const uint8_t*>(repeatedPage.value()->samples.data()),
            repeatedPage.value()->samples.size() * sizeof(learned::QuantizedTerrainSample)};
        const std::string repeatedPageHash = learned::sha256Hex(learned::sha256(repeatedPageBytes));
        WARN("Cache-cleared concurrent final page hash: " << repeatedPageHash);
        CHECK(repeatedPageHash == pageHash);
        CHECK(runtime.metrics().maximumConcurrentInferenceCalls == 1);
    }
    CHECK(metrics.coreMlPartitions > 0);
    CHECK(metrics.coreMlNodes > 0);
    CHECK(metrics.maximumConcurrentInferenceCalls == 1);

    TempDir profileDirectory("terrain_runtime_profile_binding");
    const std::filesystem::path profilePath = std::filesystem::path(profileDirectory.path()) / "v4";
    std::error_code profileError;
    std::filesystem::create_directories(profilePath, profileError);
    REQUIRE_FALSE(profileError);
    const bootstrap::TerrainRuntimeStepResult binding = runtime.bindWorldProfile(profilePath);
    CAPTURE(binding.failure.message);
    REQUIRE(binding.succeeded);
    const std::shared_ptr<learned::WorldGenerationContext> profileContext =
        runtime.qualifiedGenerationContext();
    REQUIRE(profileContext);
    CHECK(profileContext->identity() == runtime.generationIdentity());
    CHECK(profileContext->hydrologyAuthorityRoot() ==
          profilePath / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
}

TEST_CASE("Seed 42 reported water scene retains exact static routed tops",
          "[terrain-runtime][real-model][water][.real-water-scene]") {
    constexpr uint64_t SEED = 42;
    constexpr int64_t ORIGIN_X = 4'788;
    constexpr int64_t ORIGIN_Z = -4'506;
    constexpr int EDGE = 257;
    constexpr auto TIME_LIMIT = std::chrono::minutes(5);

    const char* enabledEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_WATER");
    if (enabledEnvironment == nullptr || std::string_view(enabledEnvironment) != "1") {
        SKIP("Set RYCRAFT_TERRAIN_REAL_WATER=1 to run the reported real-model water scene");
    }
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");
    const char* profileEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_WATER_PROFILE");
    if (profileEnvironment == nullptr || std::string_view(profileEnvironment).empty()) {
        SKIP("Set RYCRAFT_TERRAIN_REAL_WATER_PROFILE to a disposable matching v4 profile");
    }
    const std::filesystem::path profilePath(profileEnvironment);
    CAPTURE(profilePath);
    REQUIRE(profilePath.is_absolute());
    std::error_code profileError;
    REQUIRE(std::filesystem::is_directory(profilePath, profileError));
    REQUIRE_FALSE(profileError);

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    ProductionTerrainRuntime runtime(SEED, expected);
    bootstrap::TerrainBootstrapCancellation cancellation;
    bootstrap::TerrainRuntimeStepResult step = runtime.qualifyPlatform();
    if (step.succeeded)
        step = runtime.compile(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.loadAndQualify(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.bindWorldProfile(profilePath);
    CAPTURE(step.failure.message);
    REQUIRE(step.succeeded);
    const std::shared_ptr<learned::WorldGenerationContext> context =
        runtime.qualifiedGenerationContext();
    REQUIRE(context);
    ChunkGenerator generator(SEED, context);

    const auto deadline = std::chrono::steady_clock::now() + TIME_LIMIT;
    const auto awaitGeneration = [&](auto&& operation) {
        while (std::chrono::steady_clock::now() < deadline) {
            try {
                operation();
                return;
            } catch (const learned::GenerationFailureException& failure) {
                if (failure.status() != learned::AuthorityStatus::DEFERRED)
                    throw;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        FAIL("The reported real-model water scene did not complete within five minutes");
    };

    std::vector<worldgen::SurfaceSample> native(static_cast<size_t>(EDGE * EDGE));
    std::vector<worldgen::SurfaceSample> exact(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration([&] {
        generator.sampleNativeHydrologyGeometryGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE, native);
    });
    awaitGeneration([&] { generator.sampleExactSurfaceGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, exact); });

    const auto index = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    const auto explicitFall = [](const worldgen::SurfaceSample& sample) {
        return sample.hydrology.waterfall &&
               sample.hydrology.transitionOwnerKind ==
                   worldgen::WaterTransitionKind::EXPLICIT_FALL &&
               sample.hydrology.transitionOwnerId != 0;
    };
    size_t nativeWet = 0;
    size_t exactWet = 0;
    size_t wetMaskMismatches = 0;
    size_t unsupportedWet = 0;
    size_t unownedStageJumps = 0;
    double maximumUnownedStageJump = 0.0;
    ColumnPos firstUnownedPosition{};
    ColumnPos firstUnownedNeighbor{};
    worldgen::SurfaceSample firstUnowned{};
    worldgen::SurfaceSample firstUnownedAdjacent{};
    ColumnPos maximumUnownedPosition{};
    ColumnPos maximumUnownedNeighbor{};
    worldgen::SurfaceSample maximumUnowned{};
    worldgen::SurfaceSample maximumUnownedAdjacent{};
    size_t uphillRouteSteps = 0;
    double maximumUphillRouteStep = 0.0;
    std::optional<ColumnPos> partialRoute;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const worldgen::GeneratedFluidColumn nativeFluid =
                worldgen::generatedFluidColumn(native[index(x, z)]);
            const worldgen::SurfaceSample& sample = exact[index(x, z)];
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(sample);
            nativeWet += nativeFluid.wet ? 1U : 0U;
            exactWet += fluid.wet ? 1U : 0U;
            wetMaskMismatches += nativeFluid.wet != fluid.wet ? 1U : 0U;
            unsupportedWet +=
                fluid.wet && fluid.visibleSurface <= worldgen::geometryTerrainHeight(sample) + 0.01
                    ? 1U
                    : 0U;
            if (!partialRoute && sample.hydrology.river && !sample.hydrology.waterfall &&
                fluid.wet && !fluid.topState.isSource() &&
                fluid.topY >=
                    static_cast<int>(std::llround(worldgen::geometryTerrainHeight(sample))) + 1) {
                partialRoute = ColumnPos{ORIGIN_X + x, ORIGIN_Z + z};
            }

            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE || !fluid.wet)
                    continue;
                const worldgen::SurfaceSample& neighbor = exact[index(x + offsetX, z + offsetZ)];
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                if (!neighborFluid.wet || explicitFall(sample) || explicitFall(neighbor))
                    continue;
                const double difference =
                    std::abs(fluid.visibleSurface - neighborFluid.visibleSurface);
                if (difference > 0.125001) {
                    if (unownedStageJumps == 0) {
                        firstUnownedPosition = {ORIGIN_X + x, ORIGIN_Z + z};
                        firstUnownedNeighbor = {ORIGIN_X + x + offsetX, ORIGIN_Z + z + offsetZ};
                        firstUnowned = sample;
                        firstUnownedAdjacent = neighbor;
                    }
                    ++unownedStageJumps;
                    if (difference > maximumUnownedStageJump) {
                        maximumUnownedStageJump = difference;
                        maximumUnownedPosition = {ORIGIN_X + x, ORIGIN_Z + z};
                        maximumUnownedNeighbor = {ORIGIN_X + x + offsetX, ORIGIN_Z + z + offsetZ};
                        maximumUnowned = sample;
                        maximumUnownedAdjacent = neighbor;
                    }
                }
            }

            if (sample.hydrology.river && fluid.wet && !explicitFall(sample)) {
                const double flowX = sample.hydrology.flowDirection.x;
                const double flowZ = sample.hydrology.flowDirection.z;
                const int flowOffsetX =
                    std::abs(flowX) >= std::abs(flowZ) ? (flowX < 0.0 ? -1 : 1) : 0;
                const int flowOffsetZ = flowOffsetX == 0 ? (flowZ < 0.0 ? -1 : 1) : 0;
                if (x + flowOffsetX >= 0 && x + flowOffsetX < EDGE && z + flowOffsetZ >= 0 &&
                    z + flowOffsetZ < EDGE) {
                    const worldgen::SurfaceSample& downstream =
                        exact[index(x + flowOffsetX, z + flowOffsetZ)];
                    const worldgen::GeneratedFluidColumn downstreamFluid =
                        worldgen::generatedFluidColumn(downstream);
                    if (downstream.hydrology.river && downstreamFluid.wet &&
                        downstream.hydrology.waterBodyId == sample.hydrology.waterBodyId &&
                        !explicitFall(downstream)) {
                        const double rise = downstreamFluid.visibleSurface - fluid.visibleSurface;
                        if (rise > 0.125001) {
                            ++uphillRouteSteps;
                            maximumUphillRouteStep = std::max(maximumUphillRouteStep, rise);
                        }
                    }
                }
            }
        }
    }
    WARN("Reported water scene wet="
         << exactWet << " unsupported=" << unsupportedWet << " unowned_jumps=" << unownedStageJumps
         << " max_jump=" << maximumUnownedStageJump << " uphill_steps=" << uphillRouteSteps
         << " max_uphill=" << maximumUphillRouteStep);
    const NativeHydrologyCacheMetrics stageMetrics =
        context->nativeHydrologyRouter()->cacheMetrics();
    WARN("Ordinary stage tiles builds="
         << stageMetrics.ordinaryStageTileBuilds
         << " expanded=" << stageMetrics.ordinaryStageTileExpandedBuilds
         << " waits=" << stageMetrics.ordinaryStageTileBuildWaits
         << " bytes=" << stageMetrics.ordinaryStageTileBytes
         << " peak_page_bytes=" << stageMetrics.ordinaryStageTilePeakPageBytes << " build_seconds="
         << static_cast<double>(stageMetrics.ordinaryStageTileBuildNanoseconds) / 1.0E9);
    REQUIRE(stageMetrics.ordinaryStageTileBuilds > 0);
    REQUIRE(stageMetrics.ordinaryStageTileFailures == 0);
    REQUIRE(stageMetrics.ordinaryStageTilePeakPageBytes <=
            NATIVE_HYDROLOGY_ORDINARY_STAGE_TILE_CACHE_BYTE_BUDGET);
    CAPTURE(nativeWet, exactWet, wetMaskMismatches, unsupportedWet, unownedStageJumps,
            maximumUnownedStageJump, uphillRouteSteps, maximumUphillRouteStep,
            firstUnownedPosition.x, firstUnownedPosition.z, firstUnownedNeighbor.x,
            firstUnownedNeighbor.z, firstUnowned.waterSurface, firstUnownedAdjacent.waterSurface,
            firstUnowned.hydrology.waterBodyId, firstUnownedAdjacent.hydrology.waterBodyId,
            firstUnowned.hydrology.flowDirection.x, firstUnowned.hydrology.flowDirection.z,
            firstUnownedAdjacent.hydrology.flowDirection.x,
            firstUnownedAdjacent.hydrology.flowDirection.z, firstUnowned.hydrology.channelDistance,
            firstUnowned.hydrology.channelWidth, firstUnownedAdjacent.hydrology.channelDistance,
            firstUnownedAdjacent.hydrology.channelWidth, maximumUnownedPosition.x,
            maximumUnownedPosition.z, maximumUnownedNeighbor.x, maximumUnownedNeighbor.z,
            maximumUnowned.waterSurface, maximumUnownedAdjacent.waterSurface,
            maximumUnowned.hydrology.waterBodyId, maximumUnownedAdjacent.hydrology.waterBodyId,
            maximumUnowned.hydrology.flowDirection.x, maximumUnowned.hydrology.flowDirection.z,
            maximumUnownedAdjacent.hydrology.flowDirection.x,
            maximumUnownedAdjacent.hydrology.flowDirection.z,
            maximumUnowned.hydrology.channelDistance, maximumUnowned.hydrology.channelWidth,
            maximumUnownedAdjacent.hydrology.channelDistance,
            maximumUnownedAdjacent.hydrology.channelWidth);
    REQUIRE(nativeWet > 1'000);
    REQUIRE(exactWet == nativeWet);
    REQUIRE(wetMaskMismatches == 0);
    REQUIRE(unsupportedWet == 0);
    REQUIRE(unownedStageJumps == 0);
    REQUIRE(uphillRouteSteps == 0);
    REQUIRE(partialRoute);

    const worldgen::SurfaceSample partial =
        generator.sampleExactSurface(partialRoute->x, partialRoute->z);
    const worldgen::GeneratedFluidColumn partialFluid = worldgen::generatedFluidColumn(partial);
    const auto plan = generator.getColumnPlan(
        {Chunk::worldToChunk(partialRoute->x), Chunk::worldToChunk(partialRoute->z)});
    const int surfaceY =
        plan->surfaceY(Chunk::worldToLocal(partialRoute->x), Chunk::worldToLocal(partialRoute->z));
    World world(SEED, 4, 32, context);
    for (int32_t section = Chunk::worldToChunkY(surfaceY);
         section <= Chunk::worldToChunkY(partialFluid.topY); ++section) {
        std::shared_ptr<Chunk> cube;
        while (!cube && std::chrono::steady_clock::now() < deadline) {
            cube = world.getChunk({Chunk::worldToChunk(partialRoute->x), section,
                                   Chunk::worldToChunk(partialRoute->z)});
            if (!cube)
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        REQUIRE(cube);
    }
    const FluidCell bed = world.readFluidCell({partialRoute->x, surfaceY, partialRoute->z});
    REQUIRE(bed.loaded);
    REQUIRE(isSolid(bed.block));
    for (int y = surfaceY + 1; y < partialFluid.topY; ++y) {
        const FluidCell covered = world.readFluidCell({partialRoute->x, y, partialRoute->z});
        REQUIRE(covered.loaded);
        REQUIRE(covered.block == BlockType::WATER);
        REQUIRE(covered.state.isSource());
    }
    const FluidCell top =
        world.readFluidCell({partialRoute->x, partialFluid.topY, partialRoute->z});
    REQUIRE(top.loaded);
    REQUIRE(top.block == BlockType::WATER);
    REQUIRE(top.state == partialFluid.topState);
    REQUIRE_FALSE(top.state.isSource());
    REQUIRE(world.getPendingFluidCount() == 0);
    REQUIRE(world.tickFluids(1.0) == 0);
    REQUIRE(world.getPendingFluidCount() == 0);
}

TEST_CASE("Seed 42 provisional PREVIEW and FINAL water remain independently publishable",
          "[terrain-runtime][real-model][water][far-terrain][.real-water-promotion]") {
    constexpr uint64_t SEED = 42;
    constexpr auto TIME_LIMIT = std::chrono::minutes(5);
    constexpr FarTerrainKey REPORTED_TILE{19, -19, FarTerrainStep::THIRTY_TWO};

    const char* enabledEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_WATER");
    if (enabledEnvironment == nullptr || std::string_view(enabledEnvironment) != "1") {
        SKIP("Set RYCRAFT_TERRAIN_REAL_WATER=1 to run the reported water promotion");
    }
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");
    const char* profileEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_WATER_PROFILE");
    if (profileEnvironment == nullptr || std::string_view(profileEnvironment).empty()) {
        SKIP("Set RYCRAFT_TERRAIN_REAL_WATER_PROFILE to a disposable matching v4 profile");
    }
    const std::filesystem::path profilePath(profileEnvironment);
    CAPTURE(profilePath);
    REQUIRE(profilePath.is_absolute());
    std::error_code profileError;
    REQUIRE(std::filesystem::is_directory(profilePath, profileError));
    REQUIRE_FALSE(profileError);

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    ProductionTerrainRuntime runtime(SEED, expected);
    bootstrap::TerrainBootstrapCancellation cancellation;
    bootstrap::TerrainRuntimeStepResult step = runtime.qualifyPlatform();
    if (step.succeeded)
        step = runtime.compile(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.loadAndQualify(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.bindWorldProfile(profilePath);
    CAPTURE(step.failure.message);
    REQUIRE(step.succeeded);
    const std::shared_ptr<learned::WorldGenerationContext> finalContext =
        runtime.qualifiedGenerationContext();
    REQUIRE(finalContext);
    const std::shared_ptr<learned::WorldGenerationContext> previewContext =
        finalContext->withQuality(learned::AuthorityQuality::PREVIEW);
    auto previewGenerator = std::make_shared<ChunkGenerator>(SEED, previewContext);
    auto finalGenerator = std::make_shared<ChunkGenerator>(SEED, finalContext);
    const FarTerrainSource previewSource =
        FarTerrainMesher::generatorGeometrySource(previewGenerator);
    const FarTerrainSource finalSource = FarTerrainMesher::generatorGeometrySource(finalGenerator);
    const auto deadline = std::chrono::steady_clock::now() + TIME_LIMIT;
    const auto awaitMesh = [&](const FarTerrainSource& source, FarTerrainAuthorityQuality quality) {
        while (std::chrono::steady_clock::now() < deadline) {
            try {
                return FarTerrainMesher::build(REPORTED_TILE, source, quality);
            } catch (const learned::GenerationFailureException& failure) {
                if (failure.status() != learned::AuthorityStatus::DEFERRED)
                    throw;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        FAIL("The reported real-model water promotion did not complete within five minutes");
    };

    const std::shared_ptr<const FarTerrainMesh> previewMesh =
        awaitMesh(previewSource, FarTerrainAuthorityQuality::PREVIEW);
    const std::shared_ptr<const FarTerrainMesh> finalMesh =
        awaitMesh(finalSource, FarTerrainAuthorityQuality::FINAL);
    REQUIRE(previewMesh);
    REQUIRE(finalMesh);
    REQUIRE(previewMesh->authorityQuality == FarTerrainAuthorityQuality::PREVIEW);
    REQUIRE(finalMesh->authorityQuality == FarTerrainAuthorityQuality::FINAL);
    WARN("Seed 42 tile preview water quads="
         << previewMesh->waterQuadCount
         << " bodies=" << previewMesh->waterTopology.bodyIdentityCount << " final water quads="
         << finalMesh->waterQuadCount << " bodies=" << finalMesh->waterTopology.bodyIdentityCount);
    const FarTerrainWaterPromotionAction action =
        farTerrainWaterPromotionAction(previewMesh->waterTopology, finalMesh->waterTopology);
    if (previewMesh->waterTopology == finalMesh->waterTopology) {
        REQUIRE(action == FarTerrainWaterPromotionAction::MATCHED_TOPOLOGY_TRANSITION);
    } else {
        REQUIRE(action == FarTerrainWaterPromotionAction::ATOMIC_TOPOLOGY_SWAP);
    }
}

TEST_CASE("Reported seed step-32 tile keeps sparse water probes plan-free",
          "[terrain-runtime][real-model][water][far-terrain][hydrology]"
          "[.real-reported-step32]") {
    constexpr uint64_t SEED = 11'940'042'767'486'971'292ULL;
    constexpr FarTerrainKey REPORTED_TILE{30, -5, FarTerrainStep::THIRTY_TWO};
    constexpr auto TIME_LIMIT = std::chrono::minutes(5);

    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");
    const char* profileEnvironment = std::getenv("RYCRAFT_TERRAIN_REPORTED_TILE_PROFILE");
    if (profileEnvironment == nullptr || std::string_view(profileEnvironment).empty()) {
        SKIP("Set RYCRAFT_TERRAIN_REPORTED_TILE_PROFILE to a disposable copy of the reported "
             "world profile");
    }
    const std::filesystem::path profilePath(profileEnvironment);
    CAPTURE(profilePath);
    REQUIRE(profilePath.is_absolute());
    std::error_code profileError;
    REQUIRE(std::filesystem::is_directory(profilePath, profileError));
    REQUIRE_FALSE(profileError);

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    ProductionTerrainRuntime runtime(SEED, expected);
    bootstrap::TerrainBootstrapCancellation cancellation;
    bootstrap::TerrainRuntimeStepResult step = runtime.qualifyPlatform();
    if (step.succeeded)
        step = runtime.compile(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.loadAndQualify(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.bindWorldProfile(profilePath);
    CAPTURE(step.failure.message);
    REQUIRE(step.succeeded);
    const std::shared_ptr<learned::WorldGenerationContext> finalContext =
        runtime.qualifiedGenerationContext();
    REQUIRE(finalContext);
    const std::shared_ptr<learned::WorldGenerationContext> previewContext =
        finalContext->withQuality(learned::AuthorityQuality::PREVIEW);
    const auto generator = std::make_shared<ChunkGenerator>(SEED, previewContext);
    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    REQUIRE(source.planFreeCoarseAuthority);
    const NativeHydrologyCacheMetrics before =
        previewContext->nativeHydrologyRouter()->cacheMetrics();
    const auto deadline = std::chrono::steady_clock::now() + TIME_LIMIT;
    std::shared_ptr<const FarTerrainMesh> mesh;
    while (!mesh && std::chrono::steady_clock::now() < deadline) {
        try {
            mesh =
                FarTerrainMesher::build(REPORTED_TILE, source, FarTerrainAuthorityQuality::PREVIEW);
        } catch (const learned::GenerationFailureException& failure) {
            if (failure.status() != learned::AuthorityStatus::DEFERRED)
                throw;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    REQUIRE(mesh);
    const NativeHydrologyCacheMetrics after =
        previewContext->nativeHydrologyRouter()->cacheMetrics();
    CAPTURE(before.ordinaryStageTileBuilds, after.ordinaryStageTileBuilds,
            before.ordinaryStageTileFailures, after.ordinaryStageTileFailures,
            before.ordinaryStageCoarseGridSamples, after.ordinaryStageCoarseGridSamples,
            mesh->waterQuadCount);
    REQUIRE(after.ordinaryStageTileBuilds == before.ordinaryStageTileBuilds);
    REQUIRE(after.ordinaryStageTileFailures == before.ordinaryStageTileFailures);
    REQUIRE(after.ordinaryStageCoarseGridSamples > before.ordinaryStageCoarseGridSamples);
}

TEST_CASE("Seed 42 real-model entry audit completes exact and far coverage",
          "[terrain-runtime][real-model][.real-entry-profile]") {
    using Clock = std::chrono::steady_clock;
    constexpr uint64_t SEED = 42;
    constexpr int CONFIGURED_HORIZON_CHUNKS = MAX_RENDER_DISTANCE_CHUNKS;
    constexpr int ENTRY_HORIZON_CHUNKS = V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS;
    constexpr auto PHASE_LIMIT = std::chrono::minutes(5);
    STATIC_REQUIRE(farTerrainEntryHorizonViewDistance(CONFIGURED_HORIZON_CHUNKS) ==
                   CONFIGURED_HORIZON_CHUNKS);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(CONFIGURED_HORIZON_CHUNKS) ==
                   ENTRY_HORIZON_CHUNKS);

    const char* profileEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_ENTRY_PROFILE");
    if (profileEnvironment == nullptr || std::string_view(profileEnvironment).empty()) {
        SKIP("Set RYCRAFT_TERRAIN_REAL_ENTRY_PROFILE to an absolute external profile to run "
             "the entry audit");
    }
    const std::filesystem::path profilePath(profileEnvironment);
    CAPTURE(profilePath);
    REQUIRE(profilePath.is_absolute());
    std::error_code profileError;
    REQUIRE(std::filesystem::is_directory(profilePath, profileError));
    REQUIRE_FALSE(profileError);

    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty()) {
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");
    }
    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    const char* expectColdEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_ENTRY_EXPECT_COLD");
    const bool expectCold =
        expectColdEnvironment != nullptr && std::string_view(expectColdEnvironment) == "1";

    const auto setupStart = Clock::now();
    const auto auditDeadline = setupStart + PHASE_LIMIT;
    ProductionTerrainRuntime runtime(SEED, expected);
    bootstrap::TerrainBootstrapCancellation cancellation;
    bootstrap::TerrainRuntimeStepResult step = runtime.qualifyPlatform();
    if (step.succeeded)
        step = runtime.compile(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.loadAndQualify(packEnvironment, cancellation);
    if (step.succeeded)
        step = runtime.bindWorldProfile(profilePath);
    const double setupSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
    WARN("Real entry setup seconds=" << setupSeconds << " failure="
                                     << (step.succeeded ? "none" : step.failure.message));
    CAPTURE(step.failure.message, setupSeconds);
    REQUIRE(step.succeeded);
    REQUIRE(Clock::now() <= auditDeadline);
    const std::shared_ptr<learned::WorldGenerationContext> context =
        runtime.qualifiedGenerationContext();
    REQUIRE(context);
    REQUIRE(context->identity().seed == SEED);
    REQUIRE(context->quality() == learned::AuthorityQuality::FINAL);
    REQUIRE(context->hydrologyAuthorityRoot() ==
            profilePath / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);

    TerrainRuntimeMetrics previousRuntimeMetrics = runtime.metrics();
    WARN("setup model calls: coarse=" << previousRuntimeMetrics.models[0].calls
                                      << " base=" << previousRuntimeMetrics.models[1].calls
                                      << " decoder=" << previousRuntimeMetrics.models[2].calls
                                      << " total=" << previousRuntimeMetrics.inferenceCalls);
    learned::WorldGenerationMetrics previousContextMetrics = context->metrics();
    NativeHydrologyCacheMetrics previousHydrologyMetrics =
        context->nativeHydrologyRouter()->cacheMetrics();
    const auto warnCallDeltas = [&](std::string_view phase) {
        const TerrainRuntimeMetrics runtimeMetrics = runtime.metrics();
        const learned::WorldGenerationMetrics contextMetrics = context->metrics();
        const NativeHydrologyCacheMetrics hydrologyMetrics =
            context->nativeHydrologyRouter()->cacheMetrics();
        WARN(phase << " model calls: coarse="
                   << runtimeMetrics.models[0].calls - previousRuntimeMetrics.models[0].calls
                   << " base="
                   << runtimeMetrics.models[1].calls - previousRuntimeMetrics.models[1].calls
                   << " decoder="
                   << runtimeMetrics.models[2].calls - previousRuntimeMetrics.models[2].calls
                   << " total="
                   << runtimeMetrics.inferenceCalls - previousRuntimeMetrics.inferenceCalls
                   << " queued=" << runtimeMetrics.queuedInferenceCalls << " failures="
                   << runtimeMetrics.inferenceFailures - previousRuntimeMetrics.inferenceFailures);
        WARN(phase << " model seconds: coarse="
                   << static_cast<double>(runtimeMetrics.models[0].inferenceNanoseconds -
                                          previousRuntimeMetrics.models[0].inferenceNanoseconds) /
                          1.0E9
                   << " base="
                   << static_cast<double>(runtimeMetrics.models[1].inferenceNanoseconds -
                                          previousRuntimeMetrics.models[1].inferenceNanoseconds) /
                          1.0E9
                   << " decoder="
                   << static_cast<double>(runtimeMetrics.models[2].inferenceNanoseconds -
                                          previousRuntimeMetrics.models[2].inferenceNanoseconds) /
                          1.0E9);
        WARN(
            phase
            << " authority: queries=" << contextMetrics.queries - previousContextMetrics.queries
            << " ready=" << contextMetrics.readyQueries - previousContextMetrics.readyQueries
            << " deferred="
            << contextMetrics.deferredQueries - previousContextMetrics.deferredQueries << " builds="
            << contextMetrics.authorityCache.builds - previousContextMetrics.authorityCache.builds
            << " batches="
            << contextMetrics.authorityCache.batches - previousContextMetrics.authorityCache.batches
            << " hits="
            << contextMetrics.authorityCache.hits - previousContextMetrics.authorityCache.hits
            << " disk_loads="
            << contextMetrics.authorityCache.diskLoads -
                   previousContextMetrics.authorityCache.diskLoads
            << " coordinator_queued=" << contextMetrics.authorityCache.queuedBuilds
            << " coordinator_active=" << contextMetrics.authorityCache.activeBuilds
            << " coordinator_completed="
            << contextMetrics.authorityCache.builds - previousContextMetrics.authorityCache.builds
            << " coordinator_deferred="
            << contextMetrics.authorityCache.deferredRequests -
                   previousContextMetrics.authorityCache.deferredRequests);
        WARN(phase << " hydrology: builds="
                   << hydrologyMetrics.builds - previousHydrologyMetrics.builds << " hits="
                   << hydrologyMetrics.hits - previousHydrologyMetrics.hits << " disk_loads="
                   << hydrologyMetrics.persistedLoads - previousHydrologyMetrics.persistedLoads
                   << " active=" << hydrologyMetrics.activeBuilds << " deferred="
                   << hydrologyMetrics.deferredBuilds - previousHydrologyMetrics.deferredBuilds
                   << " failures="
                   << hydrologyMetrics.failures - previousHydrologyMetrics.failures);
        previousRuntimeMetrics = runtimeMetrics;
        previousContextMetrics = contextMetrics;
        previousHydrologyMetrics = hydrologyMetrics;
    };

    std::optional<Vec3> drySpawn;
    std::optional<std::string> spawnFailure;
    uint32_t drySpawnOrdinal = 0;
    uint32_t waterRejections = 0;
    const auto spawnStart = Clock::now();
    const auto spawnDeadline = auditDeadline;
    V4SpawnWaterScreen waterScreen;
    while (drySpawnOrdinal < V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES && Clock::now() < spawnDeadline) {
        learned::AuthorityResult<std::optional<Vec3>> selected =
            findV4DryLandSpawnCandidate(context, 0, 0, drySpawnOrdinal);
        while (selected.status() == learned::AuthorityStatus::DEFERRED &&
               Clock::now() < spawnDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            selected = findV4DryLandSpawnCandidate(context, 0, 0, drySpawnOrdinal);
        }
        if (selected.status() == learned::AuthorityStatus::FAILED) {
            spawnFailure =
                selected.failure() ? selected.failure()->message : "Dry-spawn selection failed";
            break;
        }
        if (!selected.isReady())
            break;
        if (!selected.value() || !*selected.value()) {
            ++drySpawnOrdinal;
            continue;
        }

        const Vec3 candidate = **selected.value();
        V4SpawnWaterScreenResult screened = waterScreen.screen(context, candidate);
        while (screened.deferred() && Clock::now() < spawnDeadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            screened = waterScreen.screen(context, candidate);
        }
        if (screened.failed()) {
            spawnFailure = screened.failure ? screened.failure->message
                                            : "Canonical spawn water screen failed";
            break;
        }
        if (screened.deferred())
            break;
        if (screened.water()) {
            ++waterRejections;
            ++drySpawnOrdinal;
            waterScreen.reset();
            continue;
        }
        drySpawn = screened.resolvedCandidate.value_or(candidate);
        waterScreen.reset();
        break;
    }
    if (!drySpawn && !spawnFailure) {
        spawnFailure = Clock::now() >= spawnDeadline
                           ? "Dry-spawn phase reached the global five-minute deadline"
                           : "No canonical dry spawn was found";
    }
    const double spawnSeconds = std::chrono::duration<double>(Clock::now() - spawnStart).count();
    WARN("Real entry dry-spawn seconds=" << spawnSeconds << " ordinal=" << drySpawnOrdinal
                                         << " water_rejections=" << waterRejections
                                         << " failure=" << (spawnFailure ? *spawnFailure : "none"));
    const TerrainRuntimeMetrics drySpawnMetrics = runtime.metrics();
    const TerrainRuntimeMetrics::Phase& dryCoarsePhase =
        drySpawnMetrics
            .phases[static_cast<size_t>(TerrainRuntimeInferencePhase::DrySpawnCoarseSearch)];
    const TerrainRuntimeMetrics::Phase& finalSpawnPhase =
        drySpawnMetrics
            .phases[static_cast<size_t>(TerrainRuntimeInferencePhase::FinalSpawnCertification)];
    WARN("Real entry dry-spawn phase calls: coarse="
         << dryCoarsePhase.modelCalls[0] << " base=" << finalSpawnPhase.modelCalls[1]
         << " decoder=" << finalSpawnPhase.modelCalls[2]);
    warnCallDeltas("dry-spawn");
    CAPTURE(spawnFailure, spawnSeconds, drySpawnOrdinal, waterRejections);
    REQUIRE(drySpawn);
    REQUIRE(Clock::now() <= spawnDeadline);
    if (expectCold) {
        REQUIRE(dryCoarsePhase.modelCalls[0] == 80);
        REQUIRE(finalSpawnPhase.modelCalls[0] == 0);
        REQUIRE(finalSpawnPhase.modelCalls[1] == 14);
        // Decoder uses the qualified fixed batch of four windows. The same
        // spawn certification that formerly issued sixteen scalar calls now
        // completes in five deterministic batched calls, including its tail.
        REQUIRE(finalSpawnPhase.modelCalls[2] == 5);
    } else {
        REQUIRE(dryCoarsePhase.modelCalls[0] <= 80);
        REQUIRE(finalSpawnPhase.modelCalls[0] == 0);
        REQUIRE(finalSpawnPhase.modelCalls[1] <= 14);
        REQUIRE(finalSpawnPhase.modelCalls[2] <= 5);
    }
    WARN("Real entry dry spawn x=" << drySpawn->x << " y=" << drySpawn->y << " z=" << drySpawn->z);
    const int64_t drySpawnX = static_cast<int64_t>(std::floor(drySpawn->x));
    const int64_t drySpawnZ = static_cast<int64_t>(std::floor(drySpawn->z));
    const std::optional<std::vector<V4SpawnAlignedCandidate>> certifiedExactFootprint =
        v4SpawnCertificationExactFootprintPoints(Chunk::worldToChunk(drySpawnX),
                                                 Chunk::worldToChunk(drySpawnZ));
    REQUIRE(certifiedExactFootprint);
    std::vector<ColumnPos> certifiedExactColumns;
    certifiedExactColumns.reserve(certifiedExactFootprint->size());
    for (const V4SpawnAlignedCandidate point : *certifiedExactFootprint)
        certifiedExactColumns.emplace_back(point.worldX, point.worldZ);
    worldgen::MacroGenerationSampler spawnSampler(SEED, context);
    REQUIRE(spawnSampler.nativeHydrologyDryFootprintContains(certifiedExactColumns));
    struct ExactProgress {
        double seconds = 0.0;
        size_t loaded = 0;
        size_t pending = 0;
        uint64_t completedPlans = 0;
    };
    std::vector<ExactProgress> exactProgress;
    const auto exactStart = Clock::now();
    const auto exactDeadline = auditDeadline;
    World world(SEED, CONFIGURED_HORIZON_CHUNKS, MAX_LOADED_CUBES, context);
    world.setExactStreamingDistance(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    Vec3 exactCenter = *drySpawn;
    std::optional<Vec3> safeSpawn;
    auto nextProgressAt = exactStart;
    for (int recenter = 0; recenter < 4 && Clock::now() < exactDeadline; ++recenter) {
        const int64_t centerX = static_cast<int64_t>(std::floor(exactCenter.x));
        const int32_t centerY = static_cast<int32_t>(std::floor(exactCenter.y));
        const int64_t centerZ = static_cast<int64_t>(std::floor(exactCenter.z));
        world.updatePlayerPosition(centerX, centerY, centerZ);
        while (!world.exactSpawnBandReady(centerX, centerY, centerZ,
                                          V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS) &&
               Clock::now() < exactDeadline && !world.generationFailure()) {
            world.updatePlayerPosition(centerX, centerY, centerZ);
            world.pumpGeneration();
            if (Clock::now() >= nextProgressAt) {
                const StreamingWorkStats stats = world.getStreamingWorkStats();
                exactProgress.push_back({
                    .seconds = std::chrono::duration<double>(Clock::now() - exactStart).count(),
                    .loaded = world.getLoadedChunkCount(),
                    .pending = world.getPendingChunkCount(),
                    .completedPlans = stats.completedColumnPlans,
                });
                nextProgressAt = Clock::now() + std::chrono::seconds(5);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (world.generationFailure())
            break;
        if (!world.exactSpawnBandReady(centerX, centerY, centerZ,
                                       V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS)) {
            break;
        }
        safeSpawn = world.safeSpawnFromReadyPlans(centerX, centerZ,
                                                  V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS);
        if (!safeSpawn)
            break;
        const int64_t safeX = static_cast<int64_t>(std::floor(safeSpawn->x));
        const int64_t safeZ = static_cast<int64_t>(std::floor(safeSpawn->z));
        if (safeX == centerX && safeZ == centerZ)
            break;

        exactCenter = *safeSpawn;
        safeSpawn.reset();
    }
    const StreamingWorkStats exactStats = world.getStreamingWorkStats();
    exactProgress.push_back({
        .seconds = std::chrono::duration<double>(Clock::now() - exactStart).count(),
        .loaded = world.getLoadedChunkCount(),
        .pending = world.getPendingChunkCount(),
        .completedPlans = exactStats.completedColumnPlans,
    });
    for (const ExactProgress& progress : exactProgress) {
        WARN("Real entry exact progress seconds="
             << progress.seconds << " loaded=" << progress.loaded << " pending=" << progress.pending
             << " completed_plans=" << progress.completedPlans);
    }
    const double exactSeconds = std::chrono::duration<double>(Clock::now() - exactStart).count();
    const std::optional<std::string> exactFailure = world.generationFailure();
    WARN("Real entry exact-center seconds="
         << exactSeconds << " loaded=" << world.getLoadedChunkCount() << " pending="
         << world.getPendingChunkCount() << " completed_plans=" << exactStats.completedColumnPlans
         << " failure=" << (exactFailure ? *exactFailure : "none"));
    const TerrainRuntimeMetrics exactMetrics = runtime.metrics();
    const uint64_t exactModelCalls =
        exactMetrics.inferenceCalls - previousRuntimeMetrics.inferenceCalls;
    WARN("Real entry exact-center additional model calls=" << exactModelCalls);
    warnCallDeltas("exact-center");
    CAPTURE(exactSeconds, exactFailure);
    REQUIRE_FALSE(exactFailure);
    REQUIRE(safeSpawn);
    REQUIRE(Clock::now() <= exactDeadline);
    // The public cold-entry gate covers setup, dry-spawn selection, exact
    // collision safety, connected preview coverage, and protected FINAL
    // geometry together in thirty seconds. A brand-new profile may spend up
    // to ten of those seconds on exact collision safety because none of its
    // learned or hydrology authority exists on disk yet. Re-entry retains the
    // tighter five-second component budget.
    CHECK(exactSeconds <= (expectCold ? 10.0 : 5.0));
    const TerrainRuntimeMetrics::Phase& exactSpawnPhase =
        exactMetrics
            .phases[static_cast<size_t>(TerrainRuntimeInferencePhase::FinalSpawnCertification)];
    REQUIRE(exactSpawnPhase.modelCalls == finalSpawnPhase.modelCalls);

    const double cameraX = static_cast<double>(safeSpawn->x);
    const double cameraZ = static_cast<double>(safeSpawn->z);
    world.updatePlayerPosition(static_cast<int64_t>(std::floor(cameraX)),
                               static_cast<int32_t>(std::floor(safeSpawn->y)),
                               static_cast<int64_t>(std::floor(cameraZ)));
    world.publishLoadedSnapshot();

    std::vector<FarTerrainViewTile> horizonTiles;
    selectFarTerrainView(cameraX, cameraZ,
                         farTerrainEntryHorizonViewDistance(world.getViewDistance()), horizonTiles);
    REQUIRE_FALSE(horizonTiles.empty());
    std::vector<FarTerrainViewTile> entryHorizonTiles;
    selectFarTerrainView(cameraX, cameraZ, ENTRY_HORIZON_CHUNKS, entryHorizonTiles);
    REQUIRE_FALSE(entryHorizonTiles.empty());
    REQUIRE(entryHorizonTiles.size() < horizonTiles.size());
    const std::vector<learned::TerrainPageCoordinate> entryPreviewPages =
        farTerrainCoarseAuthorityPages(entryHorizonTiles, cameraX, cameraZ);
    FarTerrainScheduler scheduler(SEED, context);
    scheduler.setCanopyWorkerBudget(0);
    scheduler.setCoarseAuthorityPrefetchPages(entryPreviewPages);

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> previewMeshes;
    std::optional<std::string> previewFailure;
    const auto previewStart = Clock::now();
    const auto previewDeadline = auditDeadline;
    while (previewMeshes.size() != entryHorizonTiles.size() && Clock::now() < previewDeadline) {
        scheduler.pumpCoarseAuthorityPrefetch();
        for (size_t index = 0; index < entryHorizonTiles.size(); ++index) {
            const FarTerrainKey key{entryHorizonTiles[index].key.tileX,
                                    entryHorizonTiles[index].key.tileZ, FAR_TERRAIN_BASE_STEP};
            if (!previewMeshes.contains(key))
                static_cast<void>(scheduler.enqueue(key, static_cast<uint32_t>(index)));
        }
        std::vector<FarTerrainResult> completed;
        scheduler.drainCompleted(completed);
        for (const FarTerrainResult& result : completed) {
            if (result.failed || !result.mesh) {
                previewFailure = "A preview horizon mesh failed";
                continue;
            }
            if (result.mesh->authorityQuality != FarTerrainAuthorityQuality::PREVIEW) {
                previewFailure = "A preview horizon mesh used FINAL authority";
                continue;
            }
            previewMeshes.insert(result.key);
        }
        if (context->failure()) {
            previewFailure = context->failure()->message;
            break;
        }
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (previewMeshes.size() != entryHorizonTiles.size() && !previewFailure) {
        previewFailure = Clock::now() >= previewDeadline
                             ? "Entry preview prefix reached the global five-minute deadline"
                             : "Entry preview prefix did not complete";
    }
    FarTerrainSchedulerStats farStats = scheduler.stats();
    const double entryPreviewSeconds =
        std::chrono::duration<double>(Clock::now() - previewStart).count();
    WARN("Real entry connected preview seconds="
         << entryPreviewSeconds << " radius_chunks=" << ENTRY_HORIZON_CHUNKS
         << " tiles=" << entryHorizonTiles.size() << " pages=" << entryPreviewPages.size()
         << " meshes=" << previewMeshes.size() << " submitted=" << farStats.submitted << " built="
         << farStats.built << " deferred=" << farStats.deferred << " failed=" << farStats.failed
         << " failure=" << (previewFailure ? *previewFailure : "none"));
    warnCallDeltas("entry-preview-prefix");
    CAPTURE(entryPreviewSeconds, ENTRY_HORIZON_CHUNKS, previewMeshes.size(),
            entryHorizonTiles.size(), previewFailure);
    REQUIRE_FALSE(previewFailure);
    REQUIRE(previewMeshes.size() == entryHorizonTiles.size());
    REQUIRE(Clock::now() <= previewDeadline);

    const ColumnPos protectedAnchor = farTerrainProtectedNearAnchor(
        static_cast<int64_t>(std::floor(cameraX)), static_cast<int64_t>(std::floor(cameraZ)));
    std::vector<FarTerrainKey> protectedTargets;
    buildFarTerrainProtectedNearTargets(protectedAnchor, horizonTiles, protectedTargets);
    REQUIRE(protectedTargets.size() == FAR_TERRAIN_PROTECTED_NEAR_TARGET_COUNT);
    scheduler.setFinalStreamingWorkEnabled(true);
    const uint64_t protectedEpoch = scheduler.advanceProtectedHandoffEpoch();
    const std::vector<learned::NativeRect> protectedTerrainRegions =
        farTerrainProtectedFinalTerrainRegions(protectedTargets);
    REQUIRE_FALSE(protectedTerrainRegions.empty());
    std::unordered_map<FarTerrainKey, std::shared_ptr<const FarTerrainMesh>, FarTerrainKeyHash>
        protectedParents;
    std::unordered_map<FarTerrainKey, std::shared_ptr<const FarTerrainMesh>, FarTerrainKeyHash>
        protectedChildren;
    std::optional<std::string> protectedFailure;
    const auto protectedStart = Clock::now();
    while ((protectedParents.size() != protectedTargets.size() ||
            protectedChildren.size() != protectedTargets.size()) &&
           Clock::now() < previewDeadline) {
        for (const learned::NativeRect region : protectedTerrainRegions) {
            const auto prepared = context->queryTransientFinalNativeGrid(
                region, learned::AuthorityRequestPriority::PROTECTED_HANDOFF,
                learned::ProtectedHandoffEpoch{protectedEpoch});
            if (prepared.status() == learned::AuthorityStatus::FAILED) {
                protectedFailure = prepared.failure() ? prepared.failure()->message
                                                      : "Protected FINAL terrain prewarm failed";
                break;
            }
        }
        if (protectedFailure)
            break;
        scheduler.pumpFinalBaseAuthority();
        for (size_t index = 0; index < protectedTargets.size(); ++index) {
            const FarTerrainKey target = protectedTargets[index];
            const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
            if (!protectedParents.contains(parent)) {
                const std::shared_ptr<const FarTerrainMesh> cached = scheduler.findCached(parent);
                if (cached && cached->authorityQuality == FarTerrainAuthorityQuality::FINAL)
                    protectedParents.emplace(parent, cached);
                else
                    static_cast<void>(
                        scheduler.enqueueFinalBase(parent, static_cast<uint32_t>(index), true));
            }
            if (!protectedChildren.contains(target)) {
                const std::shared_ptr<const FarTerrainMesh> cached = scheduler.findCached(target);
                if (cached && cached->authorityQuality == FarTerrainAuthorityQuality::FINAL) {
                    protectedChildren.emplace(target, cached);
                } else if (protectedParents.contains(parent)) {
                    static_cast<void>(scheduler.enqueueUrgentFinalRefinement(
                        target, static_cast<uint32_t>(index), true));
                }
            }
        }
        std::vector<FarTerrainResult> completed;
        scheduler.drainCompleted(completed);
        for (const FarTerrainResult& result : completed) {
            if (result.failed || !result.mesh) {
                protectedFailure = "A protected FINAL terrain mesh failed";
                continue;
            }
            if (result.mesh->authorityQuality != FarTerrainAuthorityQuality::FINAL)
                continue;
            if (farTerrainIsBaseStep(result.key.step)) {
                const auto target = std::ranges::find_if(protectedTargets, [&](FarTerrainKey key) {
                    return key.tileX == result.key.tileX && key.tileZ == result.key.tileZ;
                });
                if (target != protectedTargets.end())
                    protectedParents.insert_or_assign(result.key, result.mesh);
            } else if (std::ranges::find(protectedTargets, result.key) != protectedTargets.end()) {
                protectedChildren.insert_or_assign(result.key, result.mesh);
            }
        }
        if (context->failure()) {
            protectedFailure = context->failure()->message;
            break;
        }
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if ((protectedParents.size() != protectedTargets.size() ||
         protectedChildren.size() != protectedTargets.size()) &&
        !protectedFailure) {
        protectedFailure = Clock::now() >= previewDeadline
                               ? "Protected FINAL closure reached the global five-minute deadline"
                               : "Protected FINAL closure did not complete";
    }
    std::vector<FarTerrainProtectedNearSurface> protectedSurfaces;
    protectedSurfaces.reserve(protectedTargets.size());
    if (!protectedFailure) {
        for (const FarTerrainKey target : protectedTargets) {
            const FarTerrainKey parent{target.tileX, target.tileZ, FAR_TERRAIN_BASE_STEP};
            const std::shared_ptr<const FarTerrainMesh>& child = protectedChildren.at(target);
            const std::shared_ptr<const FarTerrainMesh>& base = protectedParents.at(parent);
            protectedSurfaces.push_back({
                .key = target,
                .authorityQuality = child->authorityQuality,
                .parentAuthorityQuality = base->authorityQuality,
                .exactAuthorityCompatible = child->exactAuthorityCompatible,
                .surfaceBoundary = child->surfaceBoundary,
            });
        }
    }
    const FarTerrainProtectedNearGeometryStatus protectedStatus =
        farTerrainProtectedNearGeometryStatus(protectedAnchor, protectedTargets, protectedSurfaces);
    const double protectedSeconds =
        std::chrono::duration<double>(Clock::now() - protectedStart).count();
    farStats = scheduler.stats();
    WARN("Real entry protected FINAL closure seconds="
         << protectedSeconds << " parents=" << protectedParents.size()
         << " children=" << protectedChildren.size()
         << " matching_boundaries=" << protectedStatus.matchingSharedBoundaries << "/"
         << protectedStatus.expectedSharedBoundaries << " submitted=" << farStats.submitted
         << " built=" << farStats.built << " deferred=" << farStats.deferred << " failed="
         << farStats.failed << " failure=" << (protectedFailure ? *protectedFailure : "none"));
    warnCallDeltas("protected-final-closure");
    CAPTURE(protectedSeconds, protectedParents.size(), protectedChildren.size(), protectedFailure,
            protectedStatus.mismatchedSharedBoundaries, protectedStatus.incompatibleLodBoundaries);
    REQUIRE_FALSE(protectedFailure);
    REQUIRE(protectedStatus.ready());

    const double firstEntrySeconds =
        std::chrono::duration<double>(Clock::now() - setupStart).count();
    WARN("Real entry first-playable seconds=" << firstEntrySeconds);
    CHECK(Clock::now() <= setupStart + std::chrono::seconds(30));

    const std::vector<learned::TerrainPageCoordinate> configuredPreviewPages =
        farTerrainCoarseAuthorityPages(horizonTiles, cameraX, cameraZ);
    scheduler.setCoarseAuthorityPrefetchPages(configuredPreviewPages);
    const auto settlementStart = Clock::now();
    while (previewMeshes.size() != horizonTiles.size() && Clock::now() < auditDeadline) {
        scheduler.pumpCoarseAuthorityPrefetch();
        for (size_t index = 0; index < horizonTiles.size(); ++index) {
            const FarTerrainKey key{horizonTiles[index].key.tileX, horizonTiles[index].key.tileZ,
                                    FAR_TERRAIN_BASE_STEP};
            if (!previewMeshes.contains(key))
                static_cast<void>(scheduler.enqueue(key, static_cast<uint32_t>(index)));
        }
        std::vector<FarTerrainResult> completed;
        scheduler.drainCompleted(completed);
        for (const FarTerrainResult& result : completed) {
            if (result.failed || !result.mesh) {
                previewFailure = "A configured-horizon terrain mesh failed";
                continue;
            }
            if (farTerrainIsBaseStep(result.key.step))
                previewMeshes.insert(result.key);
        }
        if (context->failure()) {
            previewFailure = context->failure()->message;
            break;
        }
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (previewMeshes.size() != horizonTiles.size() && !previewFailure) {
        previewFailure = Clock::now() >= auditDeadline
                             ? "Configured preview horizon reached the five-minute deadline"
                             : "Configured preview horizon did not complete";
    }
    const double settlementSeconds =
        std::chrono::duration<double>(Clock::now() - settlementStart).count();
    farStats = scheduler.stats();
    const FarTerrainSchedulerStats previewFarStats = farStats;
    WARN("Real entry configured preview settlement seconds="
         << settlementSeconds << " radius_chunks=" << CONFIGURED_HORIZON_CHUNKS
         << " tiles=" << horizonTiles.size() << " pages=" << configuredPreviewPages.size()
         << " meshes=" << previewMeshes.size() << " submitted=" << farStats.submitted << " built="
         << farStats.built << " deferred=" << farStats.deferred << " failed=" << farStats.failed
         << " failure=" << (previewFailure ? *previewFailure : "none"));
    warnCallDeltas("configured-preview-settlement");
    CAPTURE(settlementSeconds, CONFIGURED_HORIZON_CHUNKS, previewMeshes.size(), horizonTiles.size(),
            previewFailure);
    REQUIRE_FALSE(previewFailure);
    REQUIRE(previewMeshes.size() == horizonTiles.size());
    REQUIRE(Clock::now() <= auditDeadline);

    const int64_t settledCenterX = static_cast<int64_t>(std::floor(safeSpawn->x));
    const int32_t settledCenterY = static_cast<int32_t>(std::floor(safeSpawn->y));
    const int64_t settledCenterZ = static_cast<int64_t>(std::floor(safeSpawn->z));
    const auto exactBandStart = Clock::now();
    while (!world.exactSpawnBandReady(settledCenterX, settledCenterY, settledCenterZ,
                                      COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) &&
           Clock::now() < auditDeadline && !world.generationFailure()) {
        world.updatePlayerPosition(settledCenterX, settledCenterY, settledCenterZ);
        world.pumpGeneration();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const double exactBandSeconds =
        std::chrono::duration<double>(Clock::now() - exactBandStart).count();
    WARN("Real entry post-entry exact-band seconds="
         << exactBandSeconds << " loaded=" << world.getLoadedChunkCount()
         << " pending=" << world.getPendingChunkCount());
    warnCallDeltas("post-entry-exact-band");
    CAPTURE(exactBandSeconds, world.generationFailure());
    REQUIRE_FALSE(world.generationFailure());
    REQUIRE(world.exactSpawnBandReady(settledCenterX, settledCenterY, settledCenterZ,
                                      COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS));
    REQUIRE(Clock::now() <= auditDeadline);
    world.publishLoadedSnapshot();

    const std::shared_ptr<const ExactSurfaceCoverageSnapshot> exactCoverage =
        world.getExactSurfaceCoverageSnapshot();
    REQUIRE(exactCoverage);
    const FarTerrainExactHandoff handoff =
        farTerrainExactHandoff(cameraX, cameraZ, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS,
                               exactCoverage->requiredSections, exactCoverage->unresolvedColumns,
                               [&](ChunkPos section) { return world.isChunkLoaded(section); });
    const auto isFinalResident = [&](const FarTerrainKey& key) {
        const std::shared_ptr<const FarTerrainMesh> mesh = scheduler.findCached(key);
        return mesh && mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL;
    };
    std::vector<FarTerrainKey> finalUpgradeOrder;
    const uint32_t requiredFinalParents = buildFarTerrainFinalParentUpgradeOrder(
        horizonTiles, cameraX, cameraZ,
        static_cast<float>(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS * CHUNK_EDGE), handoff,
        isFinalResident, finalUpgradeOrder);
    REQUIRE(requiredFinalParents <= finalUpgradeOrder.size());

    std::unordered_set<FarTerrainKey, FarTerrainKeyHash> finalMeshes;
    std::optional<std::string> finalFailure;
    const auto finalStart = Clock::now();
    const auto finalDeadline = auditDeadline;
    while (finalMeshes.size() != requiredFinalParents && Clock::now() < finalDeadline) {
        scheduler.pumpFinalBaseAuthority();
        for (size_t index = 0; index < requiredFinalParents; ++index) {
            const FarTerrainKey key = finalUpgradeOrder[index];
            if (!finalMeshes.contains(key)) {
                static_cast<void>(scheduler.enqueueFinalBase(key, static_cast<uint32_t>(index)));
            }
        }
        std::vector<FarTerrainResult> completed;
        scheduler.drainCompleted(completed);
        for (const FarTerrainResult& result : completed) {
            if (result.failed || !result.mesh) {
                finalFailure = "A FINAL exact-handoff parent failed";
                continue;
            }
            if (result.mesh->authorityQuality == FarTerrainAuthorityQuality::FINAL &&
                std::ranges::find(std::span(finalUpgradeOrder).first(requiredFinalParents),
                                  result.key) !=
                    std::span(finalUpgradeOrder).first(requiredFinalParents).end()) {
                finalMeshes.insert(result.key);
            }
        }
        if (context->failure()) {
            finalFailure = context->failure()->message;
            break;
        }
        if (completed.empty())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (finalMeshes.size() != requiredFinalParents && !finalFailure) {
        finalFailure = Clock::now() >= finalDeadline
                           ? "FINAL exact-handoff parents reached the global five-minute deadline"
                           : "FINAL exact-handoff parents did not complete";
    }
    farStats = scheduler.stats();
    const double finalSeconds = std::chrono::duration<double>(Clock::now() - finalStart).count();
    WARN("Real entry FINAL parents seconds="
         << finalSeconds << " required=" << requiredFinalParents << " built=" << finalMeshes.size()
         << " exact_required_sections=" << exactCoverage->requiredSections.size()
         << " unresolved_columns=" << exactCoverage->unresolvedColumns.size()
         << " scheduler_submitted=" << farStats.submitted - previewFarStats.submitted
         << " scheduler_built=" << farStats.built - previewFarStats.built << " scheduler_deferred="
         << farStats.deferred - previewFarStats.deferred << " scheduler_authority_resumes="
         << farStats.authorityCompletionResumes - previewFarStats.authorityCompletionResumes
         << " scheduler_failed=" << farStats.failed - previewFarStats.failed
         << " failure=" << (finalFailure ? *finalFailure : "none"));
    warnCallDeltas("final-handoff-parents");
    CAPTURE(finalSeconds, requiredFinalParents, finalMeshes.size(), finalFailure);
    REQUIRE_FALSE(finalFailure);
    REQUIRE(finalMeshes.size() == requiredFinalParents);
    REQUIRE(Clock::now() <= finalDeadline);
    const double settledSeconds = std::chrono::duration<double>(Clock::now() - setupStart).count();
    WARN("Real entry protected FINAL parents settled seconds=" << settledSeconds);
    CHECK(runtime.metrics().maximumConcurrentInferenceCalls == 1);
}

TEST_CASE("Pinned models report steady Core ML inference latency",
          "[terrain-runtime][real-model][.real-model-steady]") {
    const char* steadyEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_STEADY");
    if (steadyEnvironment == nullptr || std::string_view(steadyEnvironment) != "1")
        SKIP("Set RYCRAFT_TERRAIN_REAL_STEADY=1 to run the steady inference benchmark");
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");

    TempDir directory("terrain_runtime_real_steady");
    const std::filesystem::path stagedPack = stageRealModelPack(packEnvironment, directory.path());
    ProductionTerrainRuntime runtime(REAL_REGION_SEED);
    bootstrap::TerrainBootstrapCancellation cancellation;
    REQUIRE(runtime.qualifyPlatform().succeeded);
    const bootstrap::TerrainRuntimeStepResult compilation =
        runtime.compile(stagedPack, cancellation);
    CAPTURE(compilation.failure.message);
    REQUIRE(compilation.succeeded);
    const bootstrap::TerrainRuntimeStepResult qualification =
        runtime.loadAndQualify(stagedPack, cancellation);
    CAPTURE(qualification.failure.message);
    REQUIRE(qualification.succeeded);
    const TerrainRuntimeMetrics setupMetrics = runtime.metrics();
    REQUIRE(setupMetrics.compiledSessions == 3);
    REQUIRE(setupMetrics.residentSessions == 3);
    REQUIRE(setupMetrics.peakResidentSessions == 3);
    REQUIRE(setupMetrics.cpuFallbackIntraOpThreads >= 1);
    REQUIRE(setupMetrics.cpuFallbackIntraOpThreads <= MAXIMUM_CPU_FALLBACK_THREADS);
    for (const TerrainRuntimeMetrics::Model& model : setupMetrics.models)
        REQUIRE(model.sessionCreations == 1);

    static constexpr size_t MEASURED_CALLS = 3;
    static constexpr std::array<std::string_view, 3> MODEL_NAMES{
        "coarse",
        "base",
        "decoder",
    };
    const std::array<TerrainRuntimeModel, 4> switchOrder{
        TerrainRuntimeModel::Coarse,
        TerrainRuntimeModel::Base,
        TerrainRuntimeModel::Decoder,
        TerrainRuntimeModel::Coarse,
    };
    for (TerrainRuntimeModel model : switchOrder) {
        const TerrainRuntimeInferenceResult switched =
            runtime.runModel(model, terrainQualificationInputs(model));
        CAPTURE(MODEL_NAMES[static_cast<size_t>(model)], switched.message);
        REQUIRE(switched.succeeded);
    }
    const TerrainRuntimeMetrics afterSwitches = runtime.metrics();
    CHECK(afterSwitches.residentSessions == setupMetrics.residentSessions);
    CHECK(afterSwitches.peakResidentSessions == setupMetrics.peakResidentSessions);
    for (size_t model = 0; model < setupMetrics.models.size(); ++model)
        CHECK(afterSwitches.models[model].sessionCreations ==
              setupMetrics.models[model].sessionCreations);
    for (TerrainRuntimeModel model :
         {TerrainRuntimeModel::Coarse, TerrainRuntimeModel::Base, TerrainRuntimeModel::Decoder}) {
        const size_t modelIndex = static_cast<size_t>(model);
        const std::vector<TerrainRuntimeTensor> inputs = terrainQualificationInputs(model);
        TerrainRuntimeInferenceResult warm = runtime.runModel(model, inputs);
        CAPTURE(MODEL_NAMES[modelIndex], warm.message);
        REQUIRE(warm.succeeded);
        const std::array<TerrainQualificationOutput, 1> warmOutput{{
            {.model = model, .shape = warm.output.shape, .values = warm.output.values},
        }};
        const learned::Sha256Digest expectedDigest = quantizedQualificationDigest(warmOutput);

        const TerrainRuntimeMetrics before = runtime.metrics();
        uint64_t maximumWallNanoseconds = 0;
        for (size_t call = 0; call < MEASURED_CALLS; ++call) {
            const auto started = std::chrono::steady_clock::now();
            TerrainRuntimeInferenceResult measured = runtime.runModel(model, inputs);
            const uint64_t wallNanoseconds =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
            maximumWallNanoseconds = std::max(maximumWallNanoseconds, wallNanoseconds);
            CAPTURE(MODEL_NAMES[modelIndex], call, measured.message);
            REQUIRE(measured.succeeded);
            const std::array<TerrainQualificationOutput, 1> measuredOutput{{
                {.model = model, .shape = measured.output.shape, .values = measured.output.values},
            }};
            CHECK(quantizedQualificationDigest(measuredOutput) == expectedDigest);
        }
        const TerrainRuntimeMetrics after = runtime.metrics();
        const auto& beforeModel = before.models[modelIndex];
        const auto& afterModel = after.models[modelIndex];
        REQUIRE(afterModel.calls - beforeModel.calls == MEASURED_CALLS);
        CHECK(afterModel.sessionCreations == beforeModel.sessionCreations);
        const double averageSeconds = static_cast<double>(afterModel.inferenceNanoseconds -
                                                          beforeModel.inferenceNanoseconds) /
                                      static_cast<double>(MEASURED_CALLS) / 1.0E9;
        WARN(MODEL_NAMES[modelIndex]
             << " steady average seconds: " << averageSeconds
             << ", maximum wall seconds: " << static_cast<double>(maximumWallNanoseconds) / 1.0E9
             << ", digest: " << learned::sha256Hex(expectedDigest));
    }
}

TEST_CASE("Origin hydrology region qualifies real final terrain authority",
          "[terrain-runtime][real-model][.real-model-region]") {
    const char* regionEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_REGION");
    if (regionEnvironment == nullptr || std::string_view(regionEnvironment) != "1")
        SKIP("Set RYCRAFT_TERRAIN_REAL_REGION=1 to run the local 3x3 region gate");
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    TempDir directory("terrain_runtime_real_region");
    const std::filesystem::path applicationSupportRoot = directory.path();
    const std::filesystem::path stagedPack =
        stageRealModelPack(packEnvironment, applicationSupportRoot);

    const RealRegionRun forward = runRealRegion(stagedPack, applicationSupportRoot, expected,
                                                ORIGIN_HYDROLOGY_TERRAIN_PAGES, false);
    reportRealRegionRun("Forward", "3x3 final authority region", forward);
    CHECK(forward.afterRegion.maximumConcurrentInferenceCalls == 1);
    CHECK(forward.contextMetrics.authorityCache.builds == ORIGIN_HYDROLOGY_TERRAIN_PAGES.size());
    CHECK(forward.regionSeconds <= 30.0);

    const char* reverseEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_REGION_REVERSE");
    if (reverseEnvironment != nullptr && std::string_view(reverseEnvironment) == "1") {
        std::error_code error;
        std::filesystem::remove_all(
            applicationSupportRoot / bootstrap::V4_WORLD_DIRECTORY / "terrain-authority-v1", error);
        if (error) {
            CAPTURE(error.message());
            REQUIRE_FALSE(error);
        }
        const RealRegionRun reverse = runRealRegion(stagedPack, applicationSupportRoot, expected,
                                                    ORIGIN_HYDROLOGY_TERRAIN_PAGES, true);
        reportRealRegionRun("Reverse", "3x3 final authority region", reverse);
        CHECK(reverse.afterRegion.maximumConcurrentInferenceCalls == 1);
        CHECK(reverse.contextMetrics.authorityCache.builds ==
              ORIGIN_HYDROLOGY_TERRAIN_PAGES.size());
        CHECK(reverse.regionHash == forward.regionHash);
        CHECK(reverse.regionSeconds <= 30.0);
    }
}

TEST_CASE("Origin hydrology core records the real final authority lower bound",
          "[terrain-runtime][real-model][.real-model-region-lower-bound]") {
    const char* regionEnvironment = std::getenv("RYCRAFT_TERRAIN_REAL_REGION_2X2");
    if (regionEnvironment == nullptr || std::string_view(regionEnvironment) != "1")
        SKIP("Set RYCRAFT_TERRAIN_REAL_REGION_2X2=1 to run the local 2x2 lower bound");
    const char* packEnvironment = std::getenv("RYCRAFT_TERRAIN_MODEL_PACK");
    if (packEnvironment == nullptr || std::string_view(packEnvironment).empty())
        SKIP("Set RYCRAFT_TERRAIN_MODEL_PACK to the verified external model pack");

    const char* expectedEnvironment = std::getenv("RYCRAFT_TERRAIN_QUALIFICATION_HASH");
    const std::string expected =
        expectedEnvironment == nullptr || std::string_view(expectedEnvironment).empty()
            ? std::string(CANONICAL_QUALIFICATION_HASH)
            : std::string(expectedEnvironment);
    TempDir directory("terrain_runtime_real_region_2x2");
    const std::filesystem::path applicationSupportRoot = directory.path();
    const std::filesystem::path stagedPack =
        stageRealModelPack(packEnvironment, applicationSupportRoot);

    const RealRegionRun run = runRealRegion(stagedPack, applicationSupportRoot, expected,
                                            ORIGIN_HYDROLOGY_CORE_TERRAIN_PAGES, false);
    reportRealRegionRun("Core", "2x2 final authority lower bound", run);
    CHECK(run.afterRegion.maximumConcurrentInferenceCalls == 1);
    CHECK(run.contextMetrics.authorityCache.builds == ORIGIN_HYDROLOGY_CORE_TERRAIN_PAGES.size());
    CHECK(run.regionSeconds <= 30.0);
}
