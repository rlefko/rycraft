#include <catch2/catch_all.hpp>

#include "engine/v4_world_startup.hpp"
#include "test_helpers.hpp"
#include "world/learned_terrain.hpp"
#include "world/macro_generation.hpp"
#include "world/native_hydrology.hpp"
#include "world/physical_scale.hpp"
#include "world/terrain_bootstrap.hpp"
#include "world/terrain_runtime.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace std::chrono_literals;
using namespace worldgen::bootstrap;

TEST_CASE("Generator v4 scale and profile paths share their canonical authority",
          "[bootstrap][generator-v4][scale][profile]") {
    STATIC_REQUIRE(V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == worldgen::learned::MODEL_BLOCK_SCALE);
    STATIC_REQUIRE(GENERATOR_V4_PHYSICAL_SCALE.horizontalMetersPerBlock ==
                   worldgen::learned::WORLD_METERS_PER_BLOCK);
    STATIC_REQUIRE(GENERATOR_V4_PHYSICAL_SCALE.positiveVerticalMetersPerBlock ==
                   worldgen::learned::WORLD_METERS_PER_BLOCK);
    STATIC_REQUIRE(GENERATOR_V4_PHYSICAL_SCALE.altitudeDatumY ==
                   worldgen::learned::LEARNED_SEA_LEVEL);
    STATIC_REQUIRE(SaveManager::GENERATOR_V4_VERSION == worldgen::learned::GENERATOR_V4_VERSION);
    STATIC_REQUIRE(V4_WORLD_DIRECTORY == worldgen::v4_profile::WORLD_DIRECTORY);
    STATIC_REQUIRE(std::string_view(SaveManager::V4_REGIONS_DIRECTORY) ==
                   worldgen::v4_profile::REGIONS_DIRECTORY);
    STATIC_REQUIRE(std::string_view(SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY) ==
                   worldgen::v4_profile::TERRAIN_AUTHORITY_DIRECTORY);
    STATIC_REQUIRE(std::string_view(SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY) ==
                   worldgen::v4_profile::HYDROLOGY_AUTHORITY_DIRECTORY);
}

TEST_CASE("Generator v4 spawn placement minimizes protected FINAL owner work",
          "[bootstrap][generator-v4][spawn][authority][performance][regression]") {
    const auto reported = v4SpawnPlacementAuthorityCost(3'936, -5'200);
    const auto aligned = v4SpawnPlacementAuthorityCost(3'936, -5'568);
    REQUIRE(reported);
    REQUIRE(aligned);
    CHECK(*reported == V4SpawnPlacementAuthorityCost{.protectedDirectOwnerCount = 6,
                                                     .protectedExactOverlapCount = 1,
                                                     .exactOwnerCount = 1,
                                                     .finalRefinementPageCount = 2});
    CHECK(*aligned == V4SpawnPlacementAuthorityCost{.protectedDirectOwnerCount = 4,
                                                    .protectedExactOverlapCount = 1,
                                                    .exactOwnerCount = 1,
                                                    .finalRefinementPageCount = 1});
    CHECK(v4SpawnPlacementAuthorityPreferred(*aligned, *reported));
    CHECK_FALSE(v4SpawnPlacementAuthorityPreferred(*reported, *aligned));

    // Reflecting integer columns around the origin preserves every half-open
    // owner count, including the boundary-sensitive negative floor divisions.
    const auto reflected = v4SpawnPlacementAuthorityCost(-3'937, 5'199);
    REQUIRE(reflected);
    CHECK(*reflected == *reported);
}

TEST_CASE("Generator v4 spawn placement reuses exact owners at signed boundaries",
          "[bootstrap][generator-v4][spawn][authority][signed][regression]") {
    const auto interior = v4SpawnPlacementAuthorityCost(-4'224, -4'224);
    const auto seam = v4SpawnPlacementAuthorityCost(-4'097, -4'097);
    const auto reflectedSeam = v4SpawnPlacementAuthorityCost(4'096, 4'096);
    REQUIRE(interior);
    REQUIRE(seam);
    REQUIRE(reflectedSeam);
    CHECK(*interior == V4SpawnPlacementAuthorityCost{.protectedDirectOwnerCount = 4,
                                                     .protectedExactOverlapCount = 1,
                                                     .exactOwnerCount = 1,
                                                     .finalRefinementPageCount = 1});
    CHECK(*seam == V4SpawnPlacementAuthorityCost{.protectedDirectOwnerCount = 4,
                                                 .protectedExactOverlapCount = 4,
                                                 .exactOwnerCount = 4,
                                                 .finalRefinementPageCount = 4});
    CHECK(*reflectedSeam == *seam);
    CHECK(v4SpawnPlacementAuthorityPreferred(*seam, *interior));

    CHECK_FALSE(v4SpawnPlacementAuthorityCost(std::numeric_limits<int64_t>::min(), 0));
    CHECK_FALSE(v4SpawnPlacementAuthorityCost(std::numeric_limits<int64_t>::max(), 0));
}

TEST_CASE("Generator v4 spawn placement counts the protected diamond without its corners",
          "[bootstrap][generator-v4][spawn][authority][regression]") {
    const auto fiveOwners = v4SpawnPlacementAuthorityCost(640, 896);
    const auto sixOwners = v4SpawnPlacementAuthorityCost(640, 640);
    REQUIRE(fiveOwners);
    REQUIRE(sixOwners);
    CHECK(fiveOwners->protectedDirectOwnerCount == 5);
    CHECK(sixOwners->protectedDirectOwnerCount == 6);
}

TEST_CASE("Generator v4 spawn placement authority ordering retains existing tie breaks",
          "[bootstrap][generator-v4][spawn][authority][ordering]") {
    const V4SpawnPlacementAuthorityCost baseline{
        .protectedDirectOwnerCount = 6,
        .protectedExactOverlapCount = 1,
        .exactOwnerCount = 2,
        .finalRefinementPageCount = 4,
    };
    CHECK(v4SpawnPlacementAuthorityPreferred({.protectedDirectOwnerCount = 4,
                                              .protectedExactOverlapCount = 0,
                                              .exactOwnerCount = 4,
                                              .finalRefinementPageCount = 8},
                                             baseline));
    CHECK(v4SpawnPlacementAuthorityPreferred({.protectedDirectOwnerCount = 6,
                                              .protectedExactOverlapCount = 2,
                                              .exactOwnerCount = 4,
                                              .finalRefinementPageCount = 8},
                                             baseline));
    CHECK(v4SpawnPlacementAuthorityPreferred({.protectedDirectOwnerCount = 6,
                                              .protectedExactOverlapCount = 1,
                                              .exactOwnerCount = 1,
                                              .finalRefinementPageCount = 8},
                                             baseline));
    CHECK(v4SpawnPlacementAuthorityPreferred({.protectedDirectOwnerCount = 6,
                                              .protectedExactOverlapCount = 1,
                                              .exactOwnerCount = 2,
                                              .finalRefinementPageCount = 3},
                                             baseline));
    CHECK_FALSE(v4SpawnPlacementAuthorityPreferred(baseline, baseline));
}

class ScopedEnvironmentVariable {
public:
    ScopedEnvironmentVariable(const char* name, std::string value) : name_(name) {
        if (const char* existing = std::getenv(name))
            previous_ = existing;
        REQUIRE(setenv(name, value.c_str(), 1) == 0);
    }

    ~ScopedEnvironmentVariable() {
        if (previous_)
            (void)setenv(name_.c_str(), previous_->c_str(), 1);
        else
            (void)unsetenv(name_.c_str());
    }

private:
    std::string name_;
    std::optional<std::string> previous_;
};

TerrainAssetSpec tinyAsset() {
    return {
        .fileName = "tiny-model.bin",
        .byteSize = 3,
        .sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        .url = "https://example.invalid/tiny-model.bin",
        .kind = TerrainAssetKind::Model,
    };
}

TerrainAssetSpec secondaryTinyAsset() {
    return {
        .fileName = "decoder_model.onnx",
        .byteSize = 3,
        .sha256 = "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34",
        .url = "https://example.invalid/decoder_model.onnx",
        .kind = TerrainAssetKind::Model,
    };
}

TerrainAssetSpec resumableBaseAsset() {
    return {
        .fileName = "base_model.onnx",
        .byteSize = 8,
        .sha256 = "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab",
        .url = "https://example.invalid/base_model.onnx",
        .kind = TerrainAssetKind::Model,
    };
}

class FakeTransport final : public TerrainModelTransport {
public:
    std::string payload = "abc";
    std::unordered_map<std::string, std::string> payloads;
    int failuresRemaining = 0;
    std::optional<size_t> failAfterAdditionalBytes;
    int calls = 0;
    uint64_t transferredBytes = 0;
    std::vector<uint64_t> startingOffsets;
    std::vector<std::string> requestedAssets;
    std::filesystem::path pathThatMustNotExist;

    TerrainTransferResult download(const TerrainAssetSpec& asset,
                                   const std::filesystem::path& destination,
                                   const TerrainDownloadProgress& progress,
                                   const TerrainBootstrapCancellation& cancellation) override {
        ++calls;
        requestedAssets.push_back(asset.fileName);
        if (!pathThatMustNotExist.empty() && std::filesystem::exists(pathThatMustNotExist))
            return TerrainTransferResult::failure("Final pack became visible before verification");
        if (failuresRemaining > 0) {
            --failuresRemaining;
            return TerrainTransferResult::failure("Injected network failure");
        }
        if (cancellation.canceled())
            return TerrainTransferResult::cancellation();

        const std::string& contents =
            payloads.contains(asset.fileName) ? payloads.at(asset.fileName) : payload;
        std::error_code error;
        uint64_t startingOffset = 0;
        const bool destinationExists = std::filesystem::exists(destination, error);
        if (!error && destinationExists && std::filesystem::is_regular_file(destination, error)) {
            startingOffset = static_cast<uint64_t>(std::filesystem::file_size(destination, error));
        }
        startingOffsets.push_back(startingOffset);
        if (error || startingOffset > contents.size())
            return TerrainTransferResult::failure("Could not inspect fake destination");
        std::ofstream output(destination, std::ios::binary | std::ios::app);
        if (!output.is_open())
            return TerrainTransferResult::failure("Could not open fake destination");
        size_t additionalBytes = 0;
        for (size_t index = static_cast<size_t>(startingOffset); index < contents.size(); ++index) {
            output.put(contents[index]);
            ++additionalBytes;
            ++transferredBytes;
            if (!progress(index + 1))
                return TerrainTransferResult::cancellation();
            if (failAfterAdditionalBytes && additionalBytes == *failAfterAdditionalBytes) {
                failAfterAdditionalBytes.reset();
                output.flush();
                return TerrainTransferResult::failure("Injected partial network failure");
            }
        }
        output.close();
        return output.good() && contents.size() == asset.byteSize
                   ? TerrainTransferResult::success()
                   : TerrainTransferResult::failure("Fake payload length mismatch");
    }
};

class CountingVerifier final : public TerrainAssetVerifier {
public:
    TerrainVerificationResult
    verify(const std::filesystem::path& path, const TerrainAssetSpec& asset,
           const TerrainBootstrapCancellation* cancellation = nullptr) const override {
        ++calls;
        return verifier.verify(path, asset, cancellation);
    }

    mutable uint32_t calls = 0;

private:
    Sha256TerrainAssetVerifier verifier;
};

class BlockingTransport final : public TerrainModelTransport {
public:
    std::atomic<bool> entered{false};

    TerrainTransferResult download(const TerrainAssetSpec& asset,
                                   const std::filesystem::path& destination,
                                   const TerrainDownloadProgress& progress,
                                   const TerrainBootstrapCancellation& cancellation) override {
        (void)asset;
        (void)destination;
        (void)progress;
        entered.store(true, std::memory_order_release);
        while (!cancellation.canceled())
            std::this_thread::yield();
        return TerrainTransferResult::cancellation();
    }
};

class FakeRuntime final : public TerrainRuntimePreparation {
public:
    bool platformSupported = true;
    int loadingFailuresRemaining = 0;
    int qualificationCalls = 0;
    int compilationCalls = 0;
    int loadingCalls = 0;
    std::string fingerprint;
    std::vector<std::filesystem::path> boundProfiles;

    FakeRuntime() { selectIdentity(1); }

    void selectIdentity(uint64_t seed) { selectIdentity(seed, 0x11U, 0x22U); }

    void selectIdentity(uint64_t seed, uint8_t modelPackHash, uint8_t runtimeHash) {
        worldgen::learned::GenerationIdentity identity;
        identity.seed = seed;
        identity.modelPackHash.fill(modelPackHash);
        identity.runtimeHash.fill(runtimeHash);
        identity_ = identity;
        auto backend = std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>();
        auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
            identity, std::filesystem::path{}, std::move(backend));
        context = std::make_shared<worldgen::learned::WorldGenerationContext>(
            identity, std::move(authority), worldgen::learned::AuthorityQuality::FINAL);
        fingerprint = worldgen::learned::sha256Hex(identity.fingerprint());
    }

    TerrainRuntimeStepResult qualifyPlatform() override {
        ++qualificationCalls;
        if (!platformSupported) {
            return TerrainRuntimeStepResult::failureResult(
                TerrainBootstrapFailureCode::UnsupportedPlatform,
                "Generator v4 requires Apple Silicon and macOS 14 or newer", false);
        }
        return TerrainRuntimeStepResult::success();
    }

    TerrainRuntimeStepResult compile(const std::filesystem::path& installedPack,
                                     const TerrainBootstrapCancellation& cancellation) override {
        ++compilationCalls;
        if (cancellation.canceled()) {
            return TerrainRuntimeStepResult::failureResult(TerrainBootstrapFailureCode::Canceled,
                                                           "Compilation canceled", true);
        }
        if (!std::filesystem::exists(installedPack / "tiny-model.bin")) {
            return TerrainRuntimeStepResult::failureResult(
                TerrainBootstrapFailureCode::RuntimeCompilation,
                "Compiler did not receive a verified pack", true);
        }
        return TerrainRuntimeStepResult::success();
    }

    TerrainRuntimeStepResult
    loadAndQualify(const std::filesystem::path& installedPack,
                   const TerrainBootstrapCancellation& cancellation) override {
        (void)installedPack;
        ++loadingCalls;
        if (cancellation.canceled()) {
            return TerrainRuntimeStepResult::failureResult(TerrainBootstrapFailureCode::Canceled,
                                                           "Loading canceled", true);
        }
        if (loadingFailuresRemaining > 0) {
            --loadingFailuresRemaining;
            return TerrainRuntimeStepResult::failureResult(
                TerrainBootstrapFailureCode::Qualification,
                "Canonical model qualification hash did not match", true);
        }
        return TerrainRuntimeStepResult::success();
    }

    TerrainRuntimeStepResult bindWorldProfile(const std::filesystem::path& worldPath) override {
        if (worldPath.empty()) {
            return TerrainRuntimeStepResult::failureResult(TerrainBootstrapFailureCode::Filesystem,
                                                           "A profile path is required", false);
        }
        auto backend = std::make_shared<worldgen::learned::DeterministicFakeTerrainBackend>();
        auto authority = std::make_shared<worldgen::learned::CachedTerrainAuthority>(
            identity_, worldPath / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY, std::move(backend));
        context = std::make_shared<worldgen::learned::WorldGenerationContext>(
            identity_, std::move(authority), worldgen::learned::AuthorityQuality::FINAL,
            worldPath / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
        boundProfiles.push_back(worldPath);
        return TerrainRuntimeStepResult::success();
    }

    std::optional<std::string> qualifiedGenerationFingerprint() const override {
        return fingerprint;
    }

    std::shared_ptr<worldgen::learned::WorldGenerationContext>
    qualifiedGenerationContext() const override {
        return context;
    }

private:
    worldgen::learned::GenerationIdentity identity_;
    std::shared_ptr<worldgen::learned::WorldGenerationContext> context;
};

class RecordingTerrainAuthority final : public worldgen::learned::TerrainAuthority {
public:
    struct Request {
        worldgen::learned::TerrainPageKey key;
        worldgen::learned::AuthorityRequestPriority priority;

        auto operator<=>(const Request&) const = default;
    };

    explicit RecordingTerrainAuthority(
        worldgen::learned::GenerationIdentity identity,
        std::optional<worldgen::learned::GenerationFailureCode> deferredCode = std::nullopt)
        : identity_(std::move(identity)), deferredCode_(deferredCode) {}

    const worldgen::learned::GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey key,
                worldgen::learned::AuthorityRequestPriority priority) override {
        requests.push_back({key, priority});
        if (terminalFailure_) {
            return worldgen::learned::AuthorityResult<std::shared_ptr<
                const worldgen::learned::TerrainAuthorityPage>>::failed(*terminalFailure_);
        }
        if (!deferredCode_) {
            return worldgen::learned::AuthorityResult<
                std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::
                ready(std::make_shared<const worldgen::learned::TerrainAuthorityPage>());
        }
        return worldgen::learned::AuthorityResult<
            std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::deferred({
            .code = *deferredCode_,
            .message = "Recorded startup request",
            .retriable = true,
        });
    }

    worldgen::learned::AuthorityResult<bool>
    preparePages(std::span<const worldgen::learned::TerrainPageKey> keys,
                 worldgen::learned::AuthorityRequestPriority priority) override {
        std::vector<Request> batch;
        batch.reserve(keys.size());
        for (const worldgen::learned::TerrainPageKey key : keys)
            batch.push_back({key, priority});
        batches.push_back(std::move(batch));
        return worldgen::learned::TerrainAuthority::preparePages(keys, priority);
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    queryNative(worldgen::learned::NativeRect, worldgen::learned::AuthorityQuality,
                worldgen::learned::AuthorityRequestPriority) override {
        return invalid<worldgen::learned::PhysicalTerrainGrid>();
    }

    worldgen::learned::AuthorityResult<std::vector<worldgen::learned::PhysicalTerrainSample>>
    queryNativePoints(std::span<const worldgen::learned::NativePoint>,
                      worldgen::learned::AuthorityQuality,
                      worldgen::learned::AuthorityRequestPriority) override {
        return invalid<std::vector<worldgen::learned::PhysicalTerrainSample>>();
    }

    worldgen::learned::TerrainAuthorityCacheMetrics cacheMetrics() const override { return {}; }

    void setDeferredCode(worldgen::learned::GenerationFailureCode code) noexcept {
        terminalFailure_.reset();
        deferredCode_ = code;
    }

    void setFailure(worldgen::learned::GenerationFailureCode code, bool retriable) {
        deferredCode_.reset();
        terminalFailure_ = worldgen::learned::GenerationFailure{
            .code = code,
            .message = "Recorded startup failure",
            .retriable = retriable,
        };
    }

    void setReady() noexcept {
        deferredCode_.reset();
        terminalFailure_.reset();
    }

    std::vector<Request> requests;
    std::vector<std::vector<Request>> batches;

private:
    template <typename Value> static worldgen::learned::AuthorityResult<Value> invalid() {
        return worldgen::learned::AuthorityResult<Value>::failed({
            .code = worldgen::learned::GenerationFailureCode::INVALID_REQUEST,
            .message = "Unexpected authority query",
            .retriable = false,
        });
    }

    worldgen::learned::GenerationIdentity identity_;
    std::optional<worldgen::learned::GenerationFailureCode> deferredCode_;
    std::optional<worldgen::learned::GenerationFailure> terminalFailure_;
};

class SpawnTerrainBackend final : public worldgen::learned::TerrainInferenceBackend {
public:
    explicit SpawnTerrainBackend(worldgen::learned::TerrainPageCoordinate dryPage)
        : coarseDryPage_(dryPage), finalDryPage_(dryPage) {}

    SpawnTerrainBackend(worldgen::learned::TerrainPageCoordinate coarseDryPage,
                        worldgen::learned::TerrainPageCoordinate finalDryPage)
        : coarseDryPage_(coarseDryPage), finalDryPage_(finalDryPage) {}

    SpawnTerrainBackend(worldgen::learned::TerrainPageCoordinate coarseDryPage,
                        worldgen::learned::TerrainPageCoordinate finalDryPage,
                        int64_t dryPatchRadius, bool firstCandidateLake = false,
                        uint16_t annualPrecipitationMillimeters = 800)
        : coarseDryPage_(coarseDryPage), finalDryPage_(finalDryPage),
          dryPatchRadius_(dryPatchRadius), firstCandidateLake_(firstCandidateLake),
          annualPrecipitationMillimeters_(annualPrecipitationMillimeters) {}

    SpawnTerrainBackend(worldgen::learned::TerrainPageCoordinate coarseDryPage,
                        worldgen::learned::TerrainPageCoordinate finalDryPage,
                        int16_t finalDryElevation,
                        std::optional<worldgen::learned::NativeRect> finalNarrowDryPatch)
        : coarseDryPage_(coarseDryPage), finalDryPage_(finalDryPage),
          finalDryElevation_(finalDryElevation),
          finalNarrowDryPatch_(std::move(finalNarrowDryPatch)) {}

    worldgen::learned::AuthorityResult<worldgen::learned::TerrainAuthorityPage>
    inferPage(const worldgen::learned::GenerationIdentity& identity,
              worldgen::learned::TerrainPageKey key) override {
        using namespace worldgen::learned;
        if (key.quality == AuthorityQuality::PREVIEW)
            ++previewPageCalls_;
        else
            ++finalPageCalls_;
        const TerrainPageCoordinate dryPage =
            key.quality == AuthorityQuality::PREVIEW ? coarseDryPage_ : finalDryPage_;
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        const bool dry = inDryPatch(key.coordinate, dryPage, dryPatchRadius_);
        const bool narrowFinalPatch = finalNarrowDryPatch_ &&
                                      key.quality == AuthorityQuality::FINAL &&
                                      key.coordinate == finalDryPage_;
        const int16_t elevationMeters =
            dry && !narrowFinalPatch
                ? (firstCandidateLake_ && key.quality == AuthorityQuality::FINAL
                       ? 350
                       : finalDryElevation_)
                : -250;
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = elevationMeters,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = annualPrecipitationMillimeters_,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = -6'500,
                            });
        if (dry && firstCandidateLake_ && key.quality == AuthorityQuality::FINAL) {
            // Candidate ordinal zero begins at native (32, 32) for an origin
            // at world (0, 0). A shallow, flat floor inside a high rim is
            // dry in the learned elevation field but canonical lake water
            // after Priority-Flood Fill-Spill-Merge routing.
            constexpr int64_t LAKE_RIM_BEGIN = 0;
            constexpr int64_t LAKE_RIM_END = 256;
            constexpr int64_t LAKE_BEGIN = 1;
            constexpr int64_t LAKE_END = 255;
            constexpr int64_t DRY_PEAK_ROW = 240;
            constexpr int64_t DRY_PEAK_COLUMN = 240;
            constexpr int64_t DRY_PEAK_RADIUS = 8;
            const int64_t pageRowOrigin = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE;
            const int64_t pageColumnOrigin = key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE;
            for (int64_t row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
                const int64_t nativeRow = pageRowOrigin + row;
                for (int64_t column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                    const int64_t nativeColumn = pageColumnOrigin + column;
                    QuantizedTerrainSample& sample =
                        page.samples[static_cast<size_t>(row) * AUTHORITY_PAGE_NATIVE_EDGE +
                                     static_cast<size_t>(column)];
                    if (nativeRow >= LAKE_RIM_BEGIN && nativeRow < LAKE_RIM_END &&
                        nativeColumn >= LAKE_RIM_BEGIN && nativeColumn < LAKE_RIM_END) {
                        sample.elevationMeters = 650;
                    }
                    if (nativeRow >= LAKE_BEGIN && nativeRow < LAKE_END &&
                        nativeColumn >= LAKE_BEGIN && nativeColumn < LAKE_END) {
                        sample.elevationMeters = 420;
                    }
                    // Ordinal one is a small local peak. It has no upstream
                    // contributors at its exact center, so canonical routing
                    // cannot classify the candidate column as a river.
                    const int64_t peakDistance = std::max(std::abs(nativeRow - DRY_PEAK_ROW),
                                                          std::abs(nativeColumn - DRY_PEAK_COLUMN));
                    if (peakDistance <= DRY_PEAK_RADIUS) {
                        sample.elevationMeters = static_cast<int16_t>(800 - peakDistance * 8);
                    }
                }
            }
        }
        if (narrowFinalPatch) {
            const NativeRect& patch = *finalNarrowDryPatch_;
            const int64_t pageRowOrigin = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE;
            const int64_t pageColumnOrigin = key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE;
            for (int64_t row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
                const int64_t nativeRow = pageRowOrigin + row;
                if (nativeRow < patch.rowBegin || nativeRow >= patch.rowEnd)
                    continue;
                for (int64_t column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                    const int64_t nativeColumn = pageColumnOrigin + column;
                    if (nativeColumn < patch.columnBegin || nativeColumn >= patch.columnEnd)
                        continue;
                    page.samples[static_cast<size_t>(row) * AUTHORITY_PAGE_NATIVE_EDGE +
                                 static_cast<size_t>(column)]
                        .elevationMeters = finalDryElevation_;
                }
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    inferFinalNativeGrid(const worldgen::learned::GenerationIdentity&,
                         worldgen::learned::NativeRect region) override {
        using namespace worldgen::learned;
        ++transientGridCalls_;
        PhysicalTerrainGrid grid{
            .region = region,
            .samples = std::vector<PhysicalTerrainSample>(static_cast<size_t>(region.height()) *
                                                          static_cast<size_t>(region.width())),
        };
        size_t index = 0;
        for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
            for (int64_t column = region.columnBegin; column < region.columnEnd; ++column) {
                const TerrainPageCoordinate page{
                    .row = floorDivide(row, AUTHORITY_PAGE_NATIVE_EDGE),
                    .column = floorDivide(column, AUTHORITY_PAGE_NATIVE_EDGE),
                };
                const bool dry = inDryPatch(page, finalDryPage_, dryPatchRadius_);
                const bool narrowFinalPatch = finalNarrowDryPatch_ && page == finalDryPage_;
                int16_t elevation = dry && !narrowFinalPatch
                                        ? (firstCandidateLake_ ? 350 : finalDryElevation_)
                                        : -250;
                if (dry && firstCandidateLake_) {
                    constexpr int64_t LAKE_RIM_BEGIN = 0;
                    constexpr int64_t LAKE_RIM_END = 256;
                    constexpr int64_t LAKE_BEGIN = 1;
                    constexpr int64_t LAKE_END = 255;
                    constexpr int64_t DRY_PEAK_ROW = 240;
                    constexpr int64_t DRY_PEAK_COLUMN = 240;
                    constexpr int64_t DRY_PEAK_RADIUS = 8;
                    if (row >= LAKE_RIM_BEGIN && row < LAKE_RIM_END && column >= LAKE_RIM_BEGIN &&
                        column < LAKE_RIM_END) {
                        elevation = 650;
                    }
                    if (row >= LAKE_BEGIN && row < LAKE_END && column >= LAKE_BEGIN &&
                        column < LAKE_END) {
                        elevation = 420;
                    }
                    const int64_t peakDistance =
                        std::max(std::abs(row - DRY_PEAK_ROW), std::abs(column - DRY_PEAK_COLUMN));
                    if (peakDistance <= DRY_PEAK_RADIUS)
                        elevation = static_cast<int16_t>(800 - peakDistance * 8);
                }
                if (narrowFinalPatch && row >= finalNarrowDryPatch_->rowBegin &&
                    row < finalNarrowDryPatch_->rowEnd &&
                    column >= finalNarrowDryPatch_->columnBegin &&
                    column < finalNarrowDryPatch_->columnEnd) {
                    elevation = finalDryElevation_;
                }
                QuantizedTerrainSample quantized{
                    .elevationMeters = elevation,
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = annualPrecipitationMillimeters_,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
                grid.samples[index++] = dequantizeTerrainSample(quantized);
            }
        }
        return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(grid));
    }

    worldgen::learned::AuthorityResult<worldgen::learned::CoarseSpawnGrid>
    inferCoarseSpawnGrid(const worldgen::learned::GenerationIdentity&,
                         worldgen::learned::CoarseSpawnRegion region) override {
        using namespace worldgen::learned;
        if (!region.valid() || region.height() > MAXIMUM_COARSE_SPAWN_GRID_EDGE ||
            region.width() > MAXIMUM_COARSE_SPAWN_GRID_EDGE) {
            return AuthorityResult<CoarseSpawnGrid>::failed({
                .code = GenerationFailureCode::INVALID_REQUEST,
                .message = "Synthetic coarse spawn region is invalid",
                .retriable = false,
            });
        }
        {
            std::lock_guard lock(coarseRegionsMutex_);
            coarseRegions_.push_back(region);
        }
        ++coarseCalls_;
        CoarseSpawnGrid grid{
            .region = region,
            .elevationMeters = std::vector<float>(static_cast<size_t>(region.height()) *
                                                      static_cast<size_t>(region.width()),
                                                  -250.0F),
        };
        for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
            for (int64_t column = region.columnBegin; column < region.columnEnd; ++column) {
                if (!inDryPatch({.row = row, .column = column}, coarseDryPage_, dryPatchRadius_))
                    continue;
                const size_t index = static_cast<size_t>(row - region.rowBegin) *
                                         static_cast<size_t>(region.width()) +
                                     static_cast<size_t>(column - region.columnBegin);
                grid.elevationMeters[index] = 420.0F;
            }
        }
        return AuthorityResult<CoarseSpawnGrid>::ready(std::move(grid));
    }

    [[nodiscard]] uint64_t previewPageCalls() const noexcept { return previewPageCalls_; }
    [[nodiscard]] uint64_t finalPageCalls() const noexcept { return finalPageCalls_; }
    [[nodiscard]] uint64_t coarseCalls() const noexcept { return coarseCalls_; }
    [[nodiscard]] uint64_t transientGridCalls() const noexcept { return transientGridCalls_; }
    [[nodiscard]] std::vector<worldgen::learned::CoarseSpawnRegion> coarseRegions() const {
        std::lock_guard lock(coarseRegionsMutex_);
        return coarseRegions_;
    }

private:
    static bool inDryPatch(worldgen::learned::TerrainPageCoordinate candidate,
                           worldgen::learned::TerrainPageCoordinate center, int64_t radius) {
        return std::abs(candidate.row - center.row) <= radius &&
               std::abs(candidate.column - center.column) <= radius;
    }

    worldgen::learned::TerrainPageCoordinate coarseDryPage_;
    worldgen::learned::TerrainPageCoordinate finalDryPage_;
    int64_t dryPatchRadius_ = 0;
    bool firstCandidateLake_ = false;
    uint16_t annualPrecipitationMillimeters_ = 800;
    int16_t finalDryElevation_ = 420;
    std::optional<worldgen::learned::NativeRect> finalNarrowDryPatch_;
    std::atomic<uint64_t> previewPageCalls_{0};
    std::atomic<uint64_t> finalPageCalls_{0};
    std::atomic<uint64_t> coarseCalls_{0};
    std::atomic<uint64_t> transientGridCalls_{0};
    mutable std::mutex coarseRegionsMutex_;
    std::vector<worldgen::learned::CoarseSpawnRegion> coarseRegions_;
};

class BlockingDrySpawnAuthority final : public worldgen::learned::TerrainAuthority {
public:
    explicit BlockingDrySpawnAuthority(worldgen::learned::GenerationIdentity identity)
        : identity_(std::move(identity)) {}

    const worldgen::learned::GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey,
                worldgen::learned::AuthorityRequestPriority) override {
        return worldgen::learned::
            AuthorityResult<std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>::ready(
                std::make_shared<const worldgen::learned::TerrainAuthorityPage>());
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    queryNative(worldgen::learned::NativeRect region, worldgen::learned::AuthorityQuality,
                worldgen::learned::AuthorityRequestPriority) override {
        return worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>::ready(
            makeGrid(region));
    }

    worldgen::learned::AuthorityResult<std::vector<worldgen::learned::PhysicalTerrainSample>>
    queryNativePoints(std::span<const worldgen::learned::NativePoint> points,
                      worldgen::learned::AuthorityQuality,
                      worldgen::learned::AuthorityRequestPriority) override {
        {
            std::unique_lock lock(blockMutex_);
            if (blockNextPointQuery_) {
                blockNextPointQuery_ = false;
                pointQueryBlocked_ = true;
                blockChanged_.notify_all();
                blockChanged_.wait(lock, [&] { return releasePointQuery_; });
            }
        }
        std::vector<worldgen::learned::PhysicalTerrainSample> samples;
        samples.reserve(points.size());
        for (const worldgen::learned::NativePoint point : points)
            samples.push_back(sample(point.row, point.column));
        return worldgen::learned::AuthorityResult<
            std::vector<worldgen::learned::PhysicalTerrainSample>>::ready(std::move(samples));
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(worldgen::learned::NativeRect region,
                                  worldgen::learned::AuthorityRequestPriority) override {
        return worldgen::learned::
            AuthorityResult<std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>::ready(
                std::make_shared<const worldgen::learned::PhysicalTerrainGrid>(makeGrid(region)));
    }

    worldgen::learned::TerrainAuthorityCacheMetrics cacheMetrics() const override { return {}; }

    void blockNextPointQuery() {
        std::lock_guard lock(blockMutex_);
        blockNextPointQuery_ = true;
        pointQueryBlocked_ = false;
        releasePointQuery_ = false;
    }

    [[nodiscard]] bool waitForBlockedPointQuery(std::chrono::milliseconds timeout) {
        std::unique_lock lock(blockMutex_);
        return blockChanged_.wait_for(lock, timeout, [&] { return pointQueryBlocked_; });
    }

    void releasePointQuery() {
        std::lock_guard lock(blockMutex_);
        releasePointQuery_ = true;
        blockChanged_.notify_all();
    }

private:
    static worldgen::learned::PhysicalTerrainSample sample(int64_t row, int64_t column) {
        using namespace worldgen::learned;
        const bool dry = floorDivide(row, AUTHORITY_PAGE_NATIVE_EDGE) == 0 &&
                         floorDivide(column, AUTHORITY_PAGE_NATIVE_EDGE) == 0;
        return {
            .elevationMeters = dry ? 420.0 : -250.0,
            .meanTemperatureC = 18.0,
            .temperatureVariabilityC = 6.0,
            .annualPrecipitationMm = 0.0,
            .precipitationCoefficientOfVariation = 0.25,
            .lapseRateCPerMeter = -0.0065,
        };
    }

    static worldgen::learned::PhysicalTerrainGrid makeGrid(worldgen::learned::NativeRect region) {
        worldgen::learned::PhysicalTerrainGrid grid{.region = region};
        if (!region.valid())
            return grid;
        grid.samples.reserve(static_cast<size_t>(region.height()) *
                             static_cast<size_t>(region.width()));
        for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
            for (int64_t column = region.columnBegin; column < region.columnEnd; ++column)
                grid.samples.push_back(sample(row, column));
        }
        return grid;
    }

    worldgen::learned::GenerationIdentity identity_;
    std::mutex blockMutex_;
    std::condition_variable blockChanged_;
    bool blockNextPointQuery_ = false;
    bool pointQueryBlocked_ = false;
    bool releasePointQuery_ = false;
};

class PerpetuallyDeferredSpawnWaterAuthority final : public worldgen::learned::TerrainAuthority {
public:
    explicit PerpetuallyDeferredSpawnWaterAuthority(worldgen::learned::GenerationIdentity identity,
                                                    bool reportActiveBuild = false,
                                                    bool reportContinuousProgress = false)
        : identity_(std::move(identity)), reportActiveBuild_(reportActiveBuild),
          reportContinuousProgress_(reportContinuousProgress) {}

    const worldgen::learned::GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>
    preparePage(worldgen::learned::TerrainPageKey,
                worldgen::learned::AuthorityRequestPriority) override {
        ++requests_;
        return deferred<std::shared_ptr<const worldgen::learned::TerrainAuthorityPage>>();
    }

    worldgen::learned::AuthorityResult<worldgen::learned::PhysicalTerrainGrid>
    queryNative(worldgen::learned::NativeRect, worldgen::learned::AuthorityQuality,
                worldgen::learned::AuthorityRequestPriority) override {
        ++requests_;
        return deferred<worldgen::learned::PhysicalTerrainGrid>();
    }

    worldgen::learned::AuthorityResult<std::vector<worldgen::learned::PhysicalTerrainSample>>
    queryNativePoints(std::span<const worldgen::learned::NativePoint>,
                      worldgen::learned::AuthorityQuality,
                      worldgen::learned::AuthorityRequestPriority) override {
        ++requests_;
        return deferred<std::vector<worldgen::learned::PhysicalTerrainSample>>();
    }

    worldgen::learned::AuthorityResult<
        std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>
    queryTransientFinalNativeGrid(worldgen::learned::NativeRect,
                                  worldgen::learned::AuthorityRequestPriority) override {
        ++requests_;
        return deferred<std::shared_ptr<const worldgen::learned::PhysicalTerrainGrid>>();
    }

    worldgen::learned::TerrainAuthorityCacheMetrics cacheMetrics() const override {
        return {
            .activeBuilds = reportActiveBuild_ ? 1U : 0U,
            .batches = reportContinuousProgress_ ? ++reportedBatches_ : 0U,
        };
    }

    [[nodiscard]] uint64_t requests() const noexcept { return requests_.load(); }

private:
    template <typename Value> static worldgen::learned::AuthorityResult<Value> deferred() {
        return worldgen::learned::AuthorityResult<Value>::deferred({
            .code = worldgen::learned::GenerationFailureCode::PAGE_NOT_FOUND,
            .message = "Synthetic spawn water authority remains deferred",
            .retriable = true,
        });
    }

    worldgen::learned::GenerationIdentity identity_;
    bool reportActiveBuild_ = false;
    bool reportContinuousProgress_ = false;
    mutable std::atomic<uint64_t> reportedBatches_{0};
    std::atomic<uint64_t> requests_{0};
};

worldgen::learned::GenerationIdentity spawnTestIdentity(uint64_t seed) {
    worldgen::learned::GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    return identity;
}

worldgen::learned::AuthorityResult<std::optional<Vec3>> awaitDryLandSpawnCandidate(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context, int64_t originWorldX,
    int64_t originWorldZ, uint32_t ordinal) {
    using namespace worldgen::learned;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    AuthorityResult<std::optional<Vec3>> result =
        findV4DryLandSpawnCandidate(context, originWorldX, originWorldZ, ordinal);
    while (result.status() == AuthorityStatus::DEFERRED &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
        result = findV4DryLandSpawnCandidate(context, originWorldX, originWorldZ, ordinal);
    }
    return result;
}

V4SpawnWaterScreenResult
awaitV4SpawnWaterScreen(V4SpawnWaterScreen& screen,
                        const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context,
                        Vec3 candidate) {
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    V4SpawnWaterScreenResult result = screen.screen(context, candidate);
    while (result.deferred() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
        result = screen.screen(context, candidate);
    }
    return result;
}

std::vector<TerrainBootstrapState> compactStates(const std::vector<TerrainBootstrapState>& states) {
    std::vector<TerrainBootstrapState> compact;
    for (const TerrainBootstrapState state : states) {
        if (compact.empty() || compact.back() != state)
            compact.push_back(state);
    }
    return compact;
}

bool hasStagingDirectory(const TerrainModelInstaller& installer) {
    const std::filesystem::path modelsRoot = installer.installedPackPath().parent_path();
    if (!std::filesystem::exists(modelsRoot))
        return false;
    for (const auto& entry : std::filesystem::directory_iterator(modelsRoot)) {
        if (entry.path().filename().string().starts_with(".staging-"))
            return true;
    }
    return false;
}

} // namespace

TEST_CASE("Generator v4 repair prequeues bounded final topology and refinement at spawn priority",
          "[bootstrap][generator-v4][authority][repair][qualification]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    for (const AuthorityQuality inputQuality :
         {AuthorityQuality::FINAL, AuthorityQuality::PREVIEW}) {
        auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
        auto context = std::make_shared<WorldGenerationContext>(identity, authority, inputQuality);

        const V4SpawnAuthorityPrequeueResult result =
            prequeueV4SpawnAuthority(context, 2'048, 2'048);
        REQUIRE(result.ready());
        REQUIRE_FALSE(result.deferred());
        REQUIRE_FALSE(result.failed());
        REQUIRE(result.finalTopologyPageCount == 36);
        REQUIRE(result.finalRefinementPageCount == 4);
        REQUIRE(result.finalPageCount == 36);
        REQUIRE(authority->batches.size() == 1);
        REQUIRE(authority->batches.front().size() == 36);
        REQUIRE(authority->requests.size() == 36);
        REQUIRE(authority->batches.front() == authority->requests);
        REQUIRE(std::ranges::all_of(authority->requests, [](const auto& request) {
            return request.priority == AuthorityRequestPriority::SPAWN &&
                   request.key.quality == AuthorityQuality::FINAL;
        }));
        std::set<TerrainPageKey> unique;
        for (const auto& request : authority->requests)
            unique.insert(request.key);
        REQUIRE(unique.size() == authority->requests.size());
    }
}

TEST_CASE("Generator v4 repair prequeues the complete cold exact band",
          "[bootstrap][generator-v4][authority][repair][qualification][cold-band][regression]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const V4SpawnAuthorityPrequeueResult result =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(result.ready());
    CHECK(result.finalTopologyPageCount == 36);
    CHECK(result.finalRefinementPageCount == 4);
    CHECK(result.finalPageCount == 36);
    REQUIRE(authority->batches.size() == 1);
    REQUIRE(authority->batches.front().size() == 36);
    REQUIRE(authority->requests.size() == 36);
    CHECK(std::ranges::all_of(authority->requests, [](const auto& request) {
        return request.priority == AuthorityRequestPriority::SPAWN &&
               request.key.quality == AuthorityQuality::FINAL;
    }));
}

TEST_CASE("Generator v4 repair keeps an interior cold exact footprint in one hydrology owner",
          "[bootstrap][generator-v4][authority][repair][qualification][cold-band]"
          "[hydrology-boundary][regression]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    // The radius-one cold active set, mesh halo, and plan apron all remain in
    // owner (2, -3) at this interior point. Startup must not open owner Z=-2.
    const V4SpawnAuthorityPrequeueResult result =
        prequeueV4SpawnAuthority(context, 5'250, -4'238, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(result.ready());
    REQUIRE(result.hydrologyOwnerCount == 1);
    REQUIRE(result.finalTopologyPageCount == 16);
    REQUIRE(result.finalRefinementPageCount == 1);
    REQUIRE(result.finalPageCount == 16);
    REQUIRE(authority->batches.size() == 1);
    REQUIRE(authority->batches.front().size() == 16);
    REQUIRE(authority->requests.size() == 16);
    const TerrainPageKey adjacentOwnerPage{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = -2, .column = 3},
    };
    REQUIRE_FALSE(std::ranges::any_of(authority->requests, [&](const auto& request) {
        return request.key == adjacentOwnerPage &&
               request.priority == AuthorityRequestPriority::SPAWN;
    }));
}

TEST_CASE("Prepared repair hydrology owners reduce prequeue to exact refinement pages",
          "[bootstrap][generator-v4][authority][repair][qualification][reuse][performance]"
          "[regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(42);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    // The accepted interior spawn band stays in owner (2, -3). Record through
    // a derived priority context to prove that the semantic preparation
    // registry is shared with the original FINAL context.
    const std::shared_ptr<WorldGenerationContext> spawnContext =
        context->withRequestPriority(AuthorityRequestPriority::SPAWN);
    spawnContext->recordPreparedNativeHydrologyOwner(2, -3);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 1);

    const V4SpawnAuthorityPrequeueResult reused =
        prequeueV4SpawnAuthority(context, 5'250, -4'238, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(reused.ready());
    REQUIRE(reused.reusedPreparedHydrology);
    REQUIRE(reused.hydrologyOwnerCount == 1);
    REQUIRE(reused.preparedHydrologyOwnerCount == 1);
    REQUIRE(reused.finalTopologyPageCount == 16);
    REQUIRE(reused.finalRefinementPageCount == 1);
    REQUIRE(reused.finalPageCount == 1);
    REQUIRE(authority->requests.size() == 1);
    const std::set<TerrainPageCoordinate> expected{
        {.row = -5, .column = 5},
    };
    std::set<TerrainPageCoordinate> requested;
    for (const auto& request : authority->requests) {
        REQUIRE(request.priority == AuthorityRequestPriority::SPAWN);
        REQUIRE(request.key.quality == AuthorityQuality::FINAL);
        requested.insert(request.key.coordinate);
    }
    REQUIRE(requested == expected);

    // The proof is deliberately process-local. A restart must semantically
    // load its RYHY owners again before it may omit topology terrain pages.
    auto restartAuthority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto restartContext = std::make_shared<WorldGenerationContext>(identity, restartAuthority,
                                                                   AuthorityQuality::FINAL);
    const V4SpawnAuthorityPrequeueResult restart = prequeueV4SpawnAuthority(
        restartContext, 5'250, -4'238, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(restart.ready());
    REQUIRE_FALSE(restart.reusedPreparedHydrology);
    REQUIRE(restart.preparedHydrologyOwnerCount == 0);
    REQUIRE(restart.finalPageCount == 16);
    REQUIRE(restartAuthority->requests.size() == 16);

    // A PREVIEW route is not semantic proof for exact startup, even when a
    // FINAL context is later derived from it.
    auto previewAuthority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto previewContext = std::make_shared<WorldGenerationContext>(identity, previewAuthority,
                                                                   AuthorityQuality::PREVIEW);
    previewContext->recordPreparedNativeHydrologyOwner(2, -3);
    CHECK(previewContext->preparedNativeHydrologyOwnerCount() == 0);
    const V4SpawnAuthorityPrequeueResult previewFallback = prequeueV4SpawnAuthority(
        previewContext, 5'250, -4'238, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(previewFallback.ready());
    CHECK_FALSE(previewFallback.reusedPreparedHydrology);
    CHECK(previewFallback.finalPageCount == 16);
}

TEST_CASE("Partial repair hydrology preparation keeps the complete topology fallback",
          "[bootstrap][generator-v4][authority][repair][qualification][reuse][fallback]"
          "[regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(42);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    context->recordPreparedNativeHydrologyOwner(0, 0);

    const V4SpawnAuthorityPrequeueResult result =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(result.ready());
    REQUIRE_FALSE(result.reusedPreparedHydrology);
    REQUIRE(result.hydrologyOwnerCount == 4);
    REQUIRE(result.preparedHydrologyOwnerCount == 1);
    REQUIRE(result.finalPageCount == 36);
    REQUIRE(authority->requests.size() == 36);
}

TEST_CASE("Generator v4 defers repair prequeue when final authority is full",
          "[bootstrap][generator-v4][authority][repair][qualification][queue]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    auto authority =
        std::make_shared<RecordingTerrainAuthority>(identity, GenerationFailureCode::QUEUE_FULL);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const V4SpawnAuthorityPrequeueResult result = prequeueV4SpawnAuthority(context, 2'048, 2'048);
    REQUIRE(result.deferred());
    REQUIRE_FALSE(result.ready());
    REQUIRE_FALSE(result.failed());
    REQUIRE(result.failure);
    REQUIRE(result.failure->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(result.finalTopologyPageCount == 36);
    REQUIRE(result.finalRefinementPageCount == 4);
    REQUIRE(result.finalPageCount == 36);
    REQUIRE(authority->batches.size() == 1);
    REQUIRE(authority->batches.front().size() == 36);
    REQUIRE(authority->requests.size() == 1);
    REQUIRE(authority->requests.front().priority == AuthorityRequestPriority::SPAWN);
    REQUIRE(authority->requests.front().key.quality == AuthorityQuality::FINAL);

    authority->setDeferredCode(GenerationFailureCode::PAGE_NOT_FOUND);
    const V4SpawnAuthorityPrequeueResult retry = prequeueV4SpawnAuthority(context, 2'048, 2'048);
    REQUIRE_FALSE(retry.ready());
    REQUIRE(retry.deferred());
    REQUIRE_FALSE(retry.failed());
    REQUIRE(retry.failure);
    REQUIRE(retry.failure->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(authority->batches.size() == 2);
    REQUIRE(authority->batches.back().size() == 36);
    REQUIRE(authority->requests.size() == 37);

    authority->setReady();
    const V4SpawnAuthorityPrequeueResult ready = prequeueV4SpawnAuthority(context, 2'048, 2'048);
    REQUIRE(ready.ready());
    REQUIRE_FALSE(ready.deferred());
    REQUIRE_FALSE(ready.failed());
    REQUIRE(authority->batches.size() == 3);
    REQUIRE(authority->batches.back().size() == 36);
    REQUIRE(authority->requests.size() == 73);
}

TEST_CASE("Queued final repair pages cannot start World or ColumnPlan construction",
          "[bootstrap][generator-v4][authority][repair][qualification][readiness][restart]"
          "[regression]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    auto authority = std::make_shared<RecordingTerrainAuthority>(
        identity, GenerationFailureCode::PAGE_NOT_FOUND);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    size_t worldConstructionCount = 0;
    size_t columnPlanStartCount = 0;
    const auto applyStartupGate = [&](const V4SpawnAuthorityPrequeueResult& result) {
        if (!result.allowsWorldConstruction())
            return;
        ++worldConstructionCount;
        ++columnPlanStartCount;
    };

    const V4SpawnAuthorityPrequeueResult building =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(building.deferred());
    REQUIRE(building.failure);
    REQUIRE(building.failure->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(building.finalPageCount == 36);
    REQUIRE(authority->requests.size() == building.finalPageCount);
    applyStartupGate(building);
    REQUIRE(worldConstructionCount == 0);
    REQUIRE(columnPlanStartCount == 0);

    authority->setReady();
    const V4SpawnAuthorityPrequeueResult ready =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(ready.ready());
    applyStartupGate(ready);
    REQUIRE(worldConstructionCount == 1);
    REQUIRE(columnPlanStartCount == 1);
}

TEST_CASE("Retriable final repair authority failure stays pre-world until explicit retry",
          "[bootstrap][generator-v4][authority][repair][qualification][readiness][retry]"
          "[regression]") {
    using namespace worldgen::learned;
    GenerationIdentity identity;
    identity.seed = 42;
    identity.modelPackHash.fill(0x11U);
    identity.runtimeHash.fill(0x22U);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    authority->setFailure(GenerationFailureCode::INFERENCE_FAILED, true);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const V4SpawnAuthorityPrequeueResult failed =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(failed.failed());
    REQUIRE_FALSE(failed.allowsWorldConstruction());
    REQUIRE(failed.failure);
    REQUIRE(failed.failure->code == GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(failed.failure->retriable);

    authority->setReady();
    const V4SpawnAuthorityPrequeueResult latched =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(latched.failed());
    REQUIRE_FALSE(latched.allowsWorldConstruction());
    REQUIRE(context->clearRetriableFailure());

    const V4SpawnAuthorityPrequeueResult retried =
        prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(retried.ready());
    REQUIRE(retried.allowsWorldConstruction());
}

TEST_CASE("Ready final repair closure is durable and restart loads it without inference",
          "[bootstrap][generator-v4][authority][repair][qualification][readiness][restart]"
          "[persistence]") {
    using namespace worldgen::learned;
    TempDir directory("v4_spawn_ready_restart");
    const GenerationIdentity identity = spawnTestIdentity(0x5A17'CAFE'0001ULL);
    size_t expectedPageCount = 0;

    {
        auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
        auto authority =
            std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
        auto context =
            std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
        V4SpawnAuthorityPrequeueResult preparation =
            prequeueV4SpawnAuthority(context, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
        REQUIRE(preparation.deferred());
        REQUIRE_FALSE(preparation.allowsWorldConstruction());
        expectedPageCount = preparation.finalPageCount;
        REQUIRE(expectedPageCount == 36);

        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (preparation.deferred() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
            preparation = prequeueV4SpawnAuthority(context, 2'048, 2'048,
                                                   COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
        }
        REQUIRE(preparation.ready());
        REQUIRE(preparation.allowsWorldConstruction());
        REQUIRE(backend->callCount() == expectedPageCount);
        const TerrainAuthorityCacheMetrics metrics = authority->cacheMetrics();
        REQUIRE(metrics.publicationWrites == expectedPageCount);
        REQUIRE(metrics.entries >= expectedPageCount);
    }

    auto restartBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    auto restartAuthority =
        std::make_shared<CachedTerrainAuthority>(identity, directory.path(), restartBackend);
    auto restartContext = std::make_shared<WorldGenerationContext>(identity, restartAuthority,
                                                                   AuthorityQuality::FINAL);
    V4SpawnAuthorityPrequeueResult restarted = prequeueV4SpawnAuthority(
        restartContext, 2'048, 2'048, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    REQUIRE(restarted.deferred());
    REQUIRE_FALSE(restarted.allowsWorldConstruction());

    const auto restartDeadline = std::chrono::steady_clock::now() + 10s;
    while (restarted.deferred() && std::chrono::steady_clock::now() < restartDeadline) {
        std::this_thread::sleep_for(1ms);
        restarted = prequeueV4SpawnAuthority(restartContext, 2'048, 2'048,
                                             COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    }
    REQUIRE(restarted.ready());
    REQUIRE(restarted.allowsWorldConstruction());
    REQUIRE(restartBackend->callCount() == 0);
    REQUIRE(restartAuthority->cacheMetrics().diskLoads == expectedPageCount);
}

TEST_CASE("Generator v4 selects a bounded dry learned spawn page and rejects ocean pages",
          "[bootstrap][generator-v4][spawn][dry-land][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_dry_spawn_candidate");
    const GenerationIdentity identity = spawnTestIdentity(0xD4A5'BEEF'0001ULL);

    // The coarse query proposes a bounded patch of inland cells without
    // writing a preview RYTA page. Final authority remains responsible for
    // accepting its precise spawn location.
    constexpr TerrainPageCoordinate DRY_PAGE{.row = 3, .column = -3};
    auto backend = std::make_shared<SpawnTerrainBackend>(DRY_PAGE);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto dry = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(dry.isReady());
    REQUIRE(*dry.value());
    CHECK(backend->coarseCalls() > 0);
    CHECK(backend->previewPageCalls() == 0);
    CHECK(backend->finalPageCalls() > 0);
    const Vec3 candidate = **dry.value();
    const NativePoint native = worldBlockToNative(static_cast<int64_t>(std::floor(candidate.x)),
                                                  static_cast<int64_t>(std::floor(candidate.z)));
    CAPTURE(candidate.x, candidate.y, candidate.z, native.row, native.column);
    CHECK(std::abs(floorDivide(native.row, AUTHORITY_PAGE_NATIVE_EDGE) - DRY_PAGE.row) <= 1);
    CHECK(std::abs(floorDivide(native.column, AUTHORITY_PAGE_NATIVE_EDGE) - DRY_PAGE.column) <= 1);
    CHECK(candidate.y > LEARNED_SEA_LEVEL + 8.0F);

    TempDir oceanDirectory("v4_all_ocean_spawn_candidate");
    auto oceanAuthority = std::make_shared<CachedTerrainAuthority>(
        identity, oceanDirectory.path(),
        std::make_shared<SpawnTerrainBackend>(TerrainPageCoordinate{.row = 99, .column = 99}));
    auto oceanContext =
        std::make_shared<WorldGenerationContext>(identity, oceanAuthority, AuthorityQuality::FINAL);
    const auto allOcean = awaitDryLandSpawnCandidate(oceanContext, 0, 0, 0);
    REQUIRE(allOcean.status() == AuthorityStatus::FAILED);
    REQUIRE(allOcean.failure());
    CHECK(allOcean.failure()->code == GenerationFailureCode::INVALID_REQUEST);
}

TEST_CASE("Generator v4 falls back from a dry coarse border to isolated learned land",
          "[bootstrap][generator-v4][spawn][dry-land][coarse-fallback][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_isolated_coarse_dry_spawn");
    const GenerationIdentity identity = spawnTestIdentity(0xD4A5'BEEF'0002ULL);

    // This raw learned land page is deliberately isolated at the bounded
    // coarse grid's half-open edge. It cannot have an in-grid 3x3 dry border,
    // but is still inside the physical search and must reach final authority
    // instead of being misreported as an all-ocean world.
    constexpr TerrainPageCoordinate ISOLATED_DRY_PAGE{.row = 7, .column = -8};
    auto backend = std::make_shared<SpawnTerrainBackend>(ISOLATED_DRY_PAGE, ISOLATED_DRY_PAGE, 0);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    REQUIRE(V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES == 81);
    const auto selected = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(selected.isReady());
    REQUIRE(*selected.value());
    const Vec3 candidate = **selected.value();
    const NativePoint native = worldBlockToNative(static_cast<int64_t>(std::floor(candidate.x)),
                                                  static_cast<int64_t>(std::floor(candidate.z)));
    CHECK((TerrainPageCoordinate{
              .row = floorDivide(native.row, AUTHORITY_PAGE_NATIVE_EDGE),
              .column = floorDivide(native.column, AUTHORITY_PAGE_NATIVE_EDGE),
          }) == ISOLATED_DRY_PAGE);
    CHECK(backend->finalPageCalls() > 0);
}

TEST_CASE("Generator v4 routes cold spawn hydrology through a bounded transient closure",
          "[bootstrap][generator-v4][spawn][dry-land][transient-hydrology][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_spawn_candidate_hydrology_closure");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'000CULL);
    constexpr TerrainPageCoordinate DRY_PAGE{.row = 0, .column = 0};
    auto backend = std::make_shared<SpawnTerrainBackend>(DRY_PAGE);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    // The canonical owner still has the same sixteen-page persistent
    // intersection. Cold spawn must not infer those mostly unused pages.
    const worldgen::NativeHydrologyAuthorityRequirements requirements =
        worldgen::nativeHydrologyAuthorityRequirementsForWorldRect(0, 0, 1'024, 1'024);
    REQUIRE(requirements.finalTopologyPages.size() == 16);
    REQUIRE(requirements.totalPageCount() == 16);

    const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(candidate.isReady());
    REQUIRE(*candidate.value());

    const TerrainAuthorityCacheMetrics afterSelection = authority->cacheMetrics();
    CHECK(afterSelection.builds == 2);
    CHECK(afterSelection.batches == 0);
    CHECK(afterSelection.batchedPages == 0);
    CHECK(backend->finalPageCalls() == 1);
    CHECK(backend->transientGridCalls() == 0);

    // The first canonical water query executes the exact 517-by-517 owner
    // rectangle, then follows the one conservative open-basin edge exposed
    // by this flat synthetic fixture. Both rectangles share tensor windows
    // with the selected page. The router does not publish the other fifteen
    // persistence pages merely because they intersect the apron.
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    const Vec3 selected = **candidate.value();
    std::optional<worldgen::HydrologySample> hydrology;
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!hydrology && std::chrono::steady_clock::now() < deadline) {
        try {
            hydrology = sampler.sampleHydrology(static_cast<double>(std::floor(selected.x)),
                                                static_cast<double>(std::floor(selected.z)));
        } catch (const GenerationFailureException& failure) {
            if (failure.status() != AuthorityStatus::DEFERRED)
                throw;
            std::this_thread::sleep_for(1ms);
        }
    }
    REQUIRE(hydrology);
    const TerrainAuthorityCacheMetrics afterHydrology = authority->cacheMetrics();
    CHECK(afterHydrology.builds == afterSelection.builds + 1);
    CHECK(afterHydrology.batches == 0);
    CHECK(afterHydrology.batchedPages == 0);
    CHECK(backend->finalPageCalls() == 1);
    CHECK(backend->transientGridCalls() == 1);
}

TEST_CASE("Generator v4 accepts only final-authority land for a spawn candidate",
          "[bootstrap][generator-v4][spawn][dry-land][final-authority][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_final_spawn_authority");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0001ULL);

    // The coarse selector is useful for proposing a search region, but it is not
    // collision or hydrology authority. The final page at the same coordinate
    // is ocean, so this must not be admitted as a spawn candidate.
    constexpr TerrainPageCoordinate COARSE_LAND{.row = 0, .column = 0};
    constexpr TerrainPageCoordinate FINAL_LAND_ELSEWHERE{.row = 9, .column = 9};
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<SpawnTerrainBackend>(COARSE_LAND, FINAL_LAND_ELSEWHERE));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(candidate.isReady());
    CHECK_FALSE(*candidate.value());
}

TEST_CASE("Generator v4 falls back to low final dry land for spawn selection",
          "[bootstrap][generator-v4][spawn][dry-land][low-elevation][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_low_dry_spawn_candidate");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0006ULL);
    constexpr TerrainPageCoordinate DRY_PAGE{.row = 0, .column = 0};

    // The former inland-only heuristic rejected this 32-meter plain before
    // canonical water or exact collision could validate it. It is dry final
    // authority and must remain a selectable proposal.
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<SpawnTerrainBackend>(DRY_PAGE, DRY_PAGE, int16_t{32}, std::nullopt));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(candidate.isReady());
    REQUIRE(*candidate.value());
    CHECK((**candidate.value()).y < 90.0F);
}

TEST_CASE("Generator v4 samples narrow final dry benches between inland probes",
          "[bootstrap][generator-v4][spawn][dry-land][narrow-bench][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_narrow_dry_spawn_candidate");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0007ULL);
    constexpr TerrainPageCoordinate DRY_PAGE{.row = 0, .column = 0};
    // A 12-block bench occupies native samples 65 through 67. The original
    // 16-block probe visits 64 and 68, so it could not discover this valid
    // center. The one-native-pixel fallback must return the center at 66.
    constexpr NativeRect NARROW_DRY_BENCH{
        .rowBegin = 65,
        .columnBegin = 65,
        .rowEnd = 68,
        .columnEnd = 68,
    };
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<SpawnTerrainBackend>(DRY_PAGE, DRY_PAGE, int16_t{32}, NARROW_DRY_BENCH));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(candidate.isReady());
    REQUIRE(*candidate.value());
    const NativePoint native =
        worldBlockToNative(static_cast<int64_t>(std::floor((**candidate.value()).x)),
                           static_cast<int64_t>(std::floor((**candidate.value()).z)));
    CHECK(native.row == 66);
    CHECK(native.column == 66);
}

TEST_CASE("Generator v4 relocates a wet proposal within one certified dry owner",
          "[bootstrap][generator-v4][spawn][dry-land][water-screen][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_canonical_water_spawn_screen");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0005ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};

    // Every coarse candidate is land. FINAL candidate zero has a 420-meter
    // basin inside a 650-meter rim, so it passes the learned proposal but
    // canonical hydrology turns that point into a lake. The same 2,048-block
    // owner also contains a provably dry local site, so startup should
    // relocate inside the owner rather than opening a neighbor or rejecting
    // the entire coarse region.
    auto backend = std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, 0, true);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto firstCandidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(firstCandidate.isReady());
    REQUIRE(*firstCandidate.value());

    V4SpawnWaterScreen screen;
    const V4SpawnWaterScreenResult firstWater =
        awaitV4SpawnWaterScreen(screen, context, **firstCandidate.value());
    REQUIRE_FALSE(firstWater.failed());
    REQUIRE_FALSE(firstWater.deferred());
    REQUIRE(firstWater.dry());
    REQUIRE(firstWater.resolvedCandidate);
    CHECK((firstWater.resolvedCandidate->x != (**firstCandidate.value()).x ||
           firstWater.resolvedCandidate->z != (**firstCandidate.value()).z));
    CHECK(world_coord::floorDiv(static_cast<int64_t>(std::floor(firstWater.resolvedCandidate->x)),
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) ==
          world_coord::floorDiv(static_cast<int64_t>(std::floor((**firstCandidate.value()).x)),
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)));
    CHECK(world_coord::floorDiv(static_cast<int64_t>(std::floor(firstWater.resolvedCandidate->z)),
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) ==
          world_coord::floorDiv(static_cast<int64_t>(std::floor((**firstCandidate.value()).z)),
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)));

    // A full cold exact prequeue at this origin needs 36 FINAL authority
    // pages. The screen directly warms one transient native owner and does not
    // turn the surrounding exact band into dry-land selection work.
    CHECK(backend->finalPageCalls() < 36);
    CHECK(backend->transientGridCalls() == 1);
    CHECK(context->preparedNativeHydrologyOwnerCount() == 0);

    constexpr int BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
    constexpr int SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    const int64_t resolvedX = static_cast<int64_t>(std::floor(firstWater.resolvedCandidate->x));
    const int64_t resolvedZ = static_cast<int64_t>(std::floor(firstWater.resolvedCandidate->z));
    std::vector<ColumnPos> localFootprint;
    for (int offsetZ = -BUFFER; offsetZ <= BUFFER; ++offsetZ) {
        for (int offsetX = -BUFFER; offsetX <= BUFFER; ++offsetX) {
            localFootprint.emplace_back(resolvedX + offsetX * SPACING,
                                        resolvedZ + offsetZ * SPACING);
        }
    }
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK(sampler.nativeHydrologyDryFootprintContains(localFootprint));

    const std::optional<std::vector<V4SpawnAlignedCandidate>> exactFootprint =
        v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(resolvedX),
                                        Chunk::worldToChunk(resolvedZ));
    REQUIRE(exactFootprint);
    std::vector<ColumnPos> exactColumns;
    exactColumns.reserve(exactFootprint->size());
    for (const V4SpawnAlignedCandidate point : *exactFootprint)
        exactColumns.emplace_back(point.worldX, point.worldZ);
    CHECK_FALSE(sampler.nativeHydrologyDryFootprintContains(exactColumns));
}

TEST_CASE("Generator v4 proves a dry proposed chunk before scanning its complete owner",
          "[bootstrap][generator-v4][spawn][dry-land][water-screen][performance][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_direct_dry_spawn_footprint");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0009ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};
    // Zero runoff makes this ocean-backed final plateau strictly dry. Resolve
    // the normal learned proposal first so its immutable page is already
    // resident, exactly as it is in production before canonical water
    // screening begins.
    auto backend =
        std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, 0, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    const AuthorityResult<std::optional<Vec3>> selected =
        awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(selected.isReady());
    REQUIRE(*selected.value());
    // The request is just inside chunk eight, while its globally nearest
    // four-block-aligned point and complete dry footprint lie in chunk nine.
    // This preserves the exhaustive path's cross-chunk distance ordering.
    const Vec3 requested{143.9F, (**selected.value()).y, 143.9F};
    V4SpawnWaterScreen screen;
    const V4SpawnWaterScreenResult result = awaitV4SpawnWaterScreen(screen, context, requested);

    REQUIRE_FALSE(result.failed());
    REQUIRE_FALSE(result.deferred());
    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK_FALSE(result.provisionalLearnedDry);
    CHECK(result.resolvedCandidate->x == 144.5F);
    CHECK(result.resolvedCandidate->z == 144.5F);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> exactFootprint =
        v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(144), Chunk::worldToChunk(144));
    REQUIRE(exactFootprint);
    CHECK(context->nativeHydrologyRouter()->cacheMetrics().dryCertificateSamples ==
          exactFootprint->size());
    std::vector<ColumnPos> exactColumns;
    exactColumns.reserve(exactFootprint->size());
    for (const V4SpawnAlignedCandidate point : *exactFootprint)
        exactColumns.emplace_back(point.worldX, point.worldZ);
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK(sampler.nativeHydrologyDryFootprintContains(exactColumns));
    CHECK(backend->transientGridCalls() == 1);
}

TEST_CASE("Generator v4 defers continental water authority to the radius zero exact plan",
          "[bootstrap][generator-v4][spawn][water-screen][continental][exact-validation]"
          "[regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_continental_spawn_provisional");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0015ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};
    constexpr size_t LOCAL_SAMPLE_COUNT =
        static_cast<size_t>(V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES * 2 + 1) *
        static_cast<size_t>(V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES * 2 + 1);

    // The complete owner and its learned input apron are positive land. That
    // intentionally gives the conservative locality proof no permanent ocean
    // terminal. Startup must not open a connected multi-owner lake walk or
    // install learned terrain as a canonical certificate.
    auto backend =
        std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, 2, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    const Vec3 requested{500.5F, 120.0F, 500.5F};
    V4SpawnWaterScreen screen;
    const V4SpawnWaterScreenResult result = awaitV4SpawnWaterScreen(screen, context, requested);

    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK(result.provisionalLearnedDry);
    CHECK_FALSE(result.canonicalDryCertificateInstalled());
    CHECK(result.requiresExactPlanValidation());
    STATIC_REQUIRE(V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS == 0);
    CHECK(backend->transientGridCalls() == 1);
    constexpr int64_t OWNER_MARGIN = worldgen::NATIVE_HYDROLOGY_HANDOFF_BLOCKS + 1;
    constexpr int64_t OWNER_FIRST = ((OWNER_MARGIN + V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS - 1) /
                                     V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS) *
                                    V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    constexpr int64_t OWNER_LAST = ((worldgen::NATIVE_HYDROLOGY_PAGE_EDGE - OWNER_MARGIN) /
                                    V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS) *
                                   V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    constexpr uint64_t OWNER_LATTICE_EDGE =
        (OWNER_LAST - OWNER_FIRST) / V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS + 1;
    STATIC_REQUIRE(OWNER_LATTICE_EDGE == 507);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> exactFootprint =
        v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(500), Chunk::worldToChunk(500));
    REQUIRE(exactFootprint);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> certificationFootprint =
        v4SpawnCertificationExactFootprintPoints(Chunk::worldToChunk(500),
                                                 Chunk::worldToChunk(500));
    REQUIRE(certificationFootprint);
    CHECK(context->nativeHydrologyRouter()->cacheMetrics().dryCertificateSamples ==
          exactFootprint->size() + certificationFootprint->size() + LOCAL_SAMPLE_COUNT +
              OWNER_LATTICE_EDGE * OWNER_LATTICE_EDGE);

    constexpr int BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
    constexpr int SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    const int64_t resolvedX = static_cast<int64_t>(std::floor(result.resolvedCandidate->x));
    const int64_t resolvedZ = static_cast<int64_t>(std::floor(result.resolvedCandidate->z));
    std::vector<ColumnPos> localFootprint;
    for (int offsetZ = -BUFFER; offsetZ <= BUFFER; ++offsetZ) {
        for (int offsetX = -BUFFER; offsetX <= BUFFER; ++offsetX) {
            localFootprint.emplace_back(resolvedX + offsetX * SPACING,
                                        resolvedZ + offsetZ * SPACING);
        }
    }
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK_FALSE(sampler.nativeHydrologyDryFootprintContains(localFootprint));
}

TEST_CASE("Generator v4 spawn alignment stays inside positive half-open owner edges",
          "[bootstrap][generator-v4][spawn][water-screen][owner-edge][positive][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_positive_spawn_owner_edge");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0016ULL);
    constexpr TerrainPageCoordinate DRY_PAGE{.row = 0, .column = 1};
    auto backend = std::make_shared<SpawnTerrainBackend>(DRY_PAGE, DRY_PAGE, 0, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen;
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{2'047.9F, 120.0F, 1'000.1F});

    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK_FALSE(result.provisionalLearnedDry);
    const int64_t resolvedX = static_cast<int64_t>(std::floor(result.resolvedCandidate->x));
    const int64_t resolvedZ = static_cast<int64_t>(std::floor(result.resolvedCandidate->z));
    CHECK(resolvedX <= 2'028);
    CHECK(resolvedX >= 0);
    CHECK(resolvedX % V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == 0);
    CHECK(std::abs(resolvedZ - 1'000) <= 64);
    CHECK(world_coord::floorDiv(resolvedX,
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) == 0);
    CHECK(world_coord::floorDiv(resolvedZ,
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) == 0);
    CHECK(terrainPageCoordinateFor(worldBlockToNative(resolvedX, resolvedZ)) == DRY_PAGE);
    CHECK(backend->finalPageCalls() <= 1);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> certificationFootprint =
        v4SpawnCertificationExactFootprintPoints(Chunk::worldToChunk(resolvedX),
                                                 Chunk::worldToChunk(resolvedZ));
    REQUIRE(certificationFootprint);
    std::vector<ColumnPos> certificationColumns;
    certificationColumns.reserve(certificationFootprint->size());
    for (const V4SpawnAlignedCandidate point : *certificationFootprint)
        certificationColumns.emplace_back(point.worldX, point.worldZ);
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK(sampler.nativeHydrologyDryFootprintContains(certificationColumns));
}

TEST_CASE("Generator v4 spawn alignment stays inside negative half-open owner edges",
          "[bootstrap][generator-v4][spawn][water-screen][owner-edge][negative][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_negative_spawn_owner_edge");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0017ULL);
    constexpr TerrainPageCoordinate DRY_PAGE{.row = -1, .column = -1};
    auto backend = std::make_shared<SpawnTerrainBackend>(DRY_PAGE, DRY_PAGE, 0, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen;
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{-0.1F, 120.0F, -1'000.1F});

    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK_FALSE(result.provisionalLearnedDry);
    const int64_t resolvedX = static_cast<int64_t>(std::floor(result.resolvedCandidate->x));
    const int64_t resolvedZ = static_cast<int64_t>(std::floor(result.resolvedCandidate->z));
    CHECK(resolvedX <= -20);
    CHECK(resolvedX >= -2'048);
    CHECK(resolvedX % V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == 0);
    CHECK(std::abs(resolvedZ - -1'000) <= 64);
    CHECK(world_coord::floorDiv(resolvedX,
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) == -1);
    CHECK(world_coord::floorDiv(resolvedZ,
                                static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) == -1);
    CHECK(terrainPageCoordinateFor(worldBlockToNative(resolvedX, resolvedZ)) == DRY_PAGE);
    CHECK(backend->finalPageCalls() <= 1);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> certificationFootprint =
        v4SpawnCertificationExactFootprintPoints(Chunk::worldToChunk(resolvedX),
                                                 Chunk::worldToChunk(resolvedZ));
    REQUIRE(certificationFootprint);
    std::vector<ColumnPos> certificationColumns;
    certificationColumns.reserve(certificationFootprint->size());
    for (const V4SpawnAlignedCandidate point : *certificationFootprint)
        certificationColumns.emplace_back(point.worldX, point.worldZ);
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK(sampler.nativeHydrologyDryFootprintContains(certificationColumns));
}

TEST_CASE("Generator v4 safe-spawn requests reject nonrepresentable candidate coordinates",
          "[bootstrap][generator-v4][spawn][water-screen][validation][regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0010ULL);
    auto authority = std::make_shared<RecordingTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen;

    for (const Vec3 candidate : {Vec3{std::numeric_limits<float>::quiet_NaN(), 96.0F, 2.5F},
                                 Vec3{2.5F, 96.0F, std::numeric_limits<float>::infinity()},
                                 Vec3{std::numeric_limits<float>::max(), 96.0F, 2.5F},
                                 Vec3{-std::numeric_limits<float>::max(), 96.0F, 2.5F}}) {
        const V4SpawnWaterScreenResult result = screen.screen(context, candidate);
        REQUIRE(result.failed());
        REQUIRE(result.failure);
        CHECK(result.failure->code == GenerationFailureCode::INVALID_REQUEST);
        CHECK_FALSE(result.failure->retriable);
    }
    CHECK(authority->requests.empty());
}

TEST_CASE("Generator v4 safe-spawn request identity preserves fractional alignment",
          "[bootstrap][generator-v4][spawn][water-screen][request-identity][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_fractional_spawn_request_identity");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0011ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};
    auto backend =
        std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, 1, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen;

    CHECK(screen.screen(context, Vec3{146.49F, 96.0F, 143.9F}).deferred());
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{146.51F, 96.0F, 143.9F});

    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK(result.resolvedCandidate->x == 148.5F);
    CHECK(result.resolvedCandidate->z == 144.5F);
}

TEST_CASE("Generator v4 reset cancels obsolete spawn proof before footprint installation",
          "[bootstrap][generator-v4][spawn][water-screen][cancellation][regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0012ULL);
    auto authority = std::make_shared<BlockingDrySpawnAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen;

    authority->blockNextPointQuery();
    CHECK(screen.screen(context, Vec3{143.9F, 96.0F, 143.9F}).deferred());
    const bool pointQueryBlocked = authority->waitForBlockedPointQuery(10s);
    if (!pointQueryBlocked)
        authority->releasePointQuery();
    REQUIRE(pointQueryBlocked);
    screen.reset();
    CHECK(screen.screen(context, Vec3{399.9F, 96.0F, 399.9F}).deferred());
    authority->releasePointQuery();
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{399.9F, 96.0F, 399.9F});

    REQUIRE(result.dry());
    REQUIRE(result.resolvedCandidate);
    CHECK(result.resolvedCandidate->x == 400.5F);
    CHECK(result.resolvedCandidate->z == 400.5F);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> exactFootprint =
        v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(400), Chunk::worldToChunk(400));
    REQUIRE(exactFootprint);
    CHECK(context->nativeHydrologyRouter()->cacheMetrics().dryCertificateSamples ==
          2 * exactFootprint->size());
}

TEST_CASE("Generator v4 deadline stops a completed local proof before footprint installation",
          "[bootstrap][generator-v4][spawn][water-screen][deadline][regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0013ULL);
    auto authority = std::make_shared<BlockingDrySpawnAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen({
        .retryInterval = 1ms,
        .noProgressTimeout = 5ms,
        .activeWorkTimeout = 1'000ms,
    });

    authority->blockNextPointQuery();
    CHECK(screen.screen(context, Vec3{143.9F, 96.0F, 143.9F}).deferred());
    const bool pointQueryBlocked = authority->waitForBlockedPointQuery(10s);
    if (!pointQueryBlocked)
        authority->releasePointQuery();
    REQUIRE(pointQueryBlocked);
    std::this_thread::sleep_for(1'100ms);
    authority->releasePointQuery();
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{143.9F, 96.0F, 143.9F});

    REQUIRE(result.failed());
    REQUIRE(result.failure);
    CHECK(result.failure->code == GenerationFailureCode::INFERENCE_FAILED);
    CHECK(result.failure->message.find("absolute 1000 millisecond request deadline") !=
          std::string::npos);
    const std::optional<std::vector<V4SpawnAlignedCandidate>> footprint =
        v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(144), Chunk::worldToChunk(144));
    REQUIRE(footprint);
    CHECK(context->nativeHydrologyRouter()->cacheMetrics().dryCertificateSamples ==
          footprint->size());
    std::vector<ColumnPos> columns;
    columns.reserve(footprint->size());
    for (const V4SpawnAlignedCandidate point : *footprint)
        columns.emplace_back(point.worldX, point.worldZ);
    worldgen::MacroGenerationSampler sampler(identity.seed, context);
    CHECK_FALSE(sampler.nativeHydrologyDryFootprintContains(columns));
}

TEST_CASE("Generator v4 safe-spawn screening rejects every canonical surface-water state",
          "[bootstrap][generator-v4][spawn][water-screen][canonical-state][regression]") {
    worldgen::HydrologySample dry;
    dry.surfaceElevation = 132.0;
    dry.waterSurface = 0.0;
    CHECK_FALSE(v4SpawnCandidateHasCanonicalSurfaceWater(dry));

    worldgen::HydrologySample delta = dry;
    delta.delta = true;
    CHECK(v4SpawnCandidateHasCanonicalSurfaceWater(delta));

    worldgen::HydrologySample transition = dry;
    transition.transitionOwnerKind = worldgen::WaterTransitionKind::OUTLET_CORRIDOR;
    CHECK(v4SpawnCandidateHasCanonicalSurfaceWater(transition));

    worldgen::HydrologySample supportedStage = dry;
    supportedStage.waterSurface = supportedStage.surfaceElevation + 0.125;
    CHECK(v4SpawnCandidateHasCanonicalSurfaceWater(supportedStage));

    worldgen::HydrologySample invalidStage = dry;
    invalidStage.waterSurface = std::numeric_limits<double>::quiet_NaN();
    CHECK(v4SpawnCandidateHasCanonicalSurfaceWater(invalidStage));
}

TEST_CASE("Generator v4 cold spawn footprint matches production exact dependencies",
          "[bootstrap][generator-v4][spawn][cold-band][footprint][regression]") {
    const std::optional<std::vector<V4SpawnAlignedCandidate>> certification =
        v4SpawnCertificationExactFootprintPoints(-3, 2);
    REQUIRE(certification);
    REQUIRE(certification->size() == 113 * 113);
    CHECK((certification->front() == V4SpawnAlignedCandidate{.worldX = -96, .worldZ = -16}));
    CHECK((certification->back() == V4SpawnAlignedCandidate{.worldX = 16, .worldZ = 96}));
    CHECK(std::ranges::adjacent_find(*certification) == certification->end());
    CHECK_FALSE(v4SpawnCertificationExactFootprintPoints(std::numeric_limits<int64_t>::max(), 0));
    CHECK_FALSE(v4SpawnCertificationExactFootprintPoints(std::numeric_limits<int64_t>::min(), 0));

    const std::vector<V4SpawnExactFootprintRow> rows = v4ColdSpawnExactFootprintRows();
    REQUIRE(rows.size() == 177);
    REQUIRE((rows.front() == V4SpawnExactFootprintRow{
                                 .zOffset = -80,
                                 .minimumXOffset = -64,
                                 .maximumXOffset = 80,
                             }));
    REQUIRE((rows[32] == V4SpawnExactFootprintRow{
                             .zOffset = -48,
                             .minimumXOffset = -80,
                             .maximumXOffset = 96,
                         }));
    REQUIRE((rows.back() == V4SpawnExactFootprintRow{
                                .zOffset = 96,
                                .minimumXOffset = -64,
                                .maximumXOffset = 80,
                            }));
    size_t pointCount = 0;
    int64_t minimumX = std::numeric_limits<int64_t>::max();
    int64_t maximumX = std::numeric_limits<int64_t>::min();
    for (const V4SpawnExactFootprintRow row : rows) {
        pointCount += row.sampleCount();
        minimumX = std::min(minimumX, row.minimumXOffset);
        maximumX = std::max(maximumX, row.maximumXOffset);
    }
    CHECK(pointCount == 30'305);
    CHECK(minimumX == -80);
    CHECK(maximumX == 96);

    const std::optional<std::vector<V4SpawnAlignedCandidate>> signedPoints =
        v4ColdSpawnExactFootprintPoints(-3, 2);
    REQUIRE(signedPoints);
    REQUIRE(signedPoints->size() == pointCount);
    CHECK((signedPoints->front() == V4SpawnAlignedCandidate{.worldX = -112, .worldZ = -48}));
    CHECK((signedPoints->back() == V4SpawnAlignedCandidate{.worldX = 32, .worldZ = 128}));
    CHECK(std::ranges::adjacent_find(*signedPoints) == signedPoints->end());
}

TEST_CASE("Cold zero-radius footprint stays within signed half-open authority pages",
          "[bootstrap][generator-v4][spawn][cold-band][footprint][boundary][negative]"
          "[regression]") {
    using worldgen::learned::TerrainPageCoordinate;

    struct BoundaryCase {
        int64_t centerChunk = 0;
        int64_t minimumWorld = 0;
        int64_t maximumWorld = 0;
        TerrainPageCoordinate page{};
        int64_t hydrologyOwner = 0;
    };
    constexpr std::array CASES{
        BoundaryCase{.centerChunk = 57,
                     .minimumWorld = 832,
                     .maximumWorld = 1'008,
                     .page = {.row = 0, .column = 0},
                     .hydrologyOwner = 0},
        BoundaryCase{.centerChunk = -59,
                     .minimumWorld = -1'024,
                     .maximumWorld = -848,
                     .page = {.row = -1, .column = -1},
                     .hydrologyOwner = -1},
    };

    for (const BoundaryCase& boundary : CASES) {
        const std::optional<std::vector<V4SpawnAlignedCandidate>> points =
            v4ColdSpawnExactFootprintPoints(boundary.centerChunk, boundary.centerChunk);
        REQUIRE(points);
        REQUIRE_FALSE(points->empty());
        int64_t minimumX = std::numeric_limits<int64_t>::max();
        int64_t minimumZ = std::numeric_limits<int64_t>::max();
        int64_t maximumX = std::numeric_limits<int64_t>::min();
        int64_t maximumZ = std::numeric_limits<int64_t>::min();
        for (const V4SpawnAlignedCandidate point : *points) {
            minimumX = std::min(minimumX, point.worldX);
            minimumZ = std::min(minimumZ, point.worldZ);
            maximumX = std::max(maximumX, point.worldX);
            maximumZ = std::max(maximumZ, point.worldZ);
            const worldgen::learned::TerrainPageCoordinate page =
                worldgen::learned::terrainPageCoordinateFor(
                    worldgen::learned::worldBlockToNative(point.worldX, point.worldZ));
            CAPTURE(boundary.centerChunk, point.worldX, point.worldZ, page.row, page.column);
            REQUIRE(page == boundary.page);
            REQUIRE(world_coord::floorDiv(
                        point.worldX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) ==
                    boundary.hydrologyOwner);
            REQUIRE(world_coord::floorDiv(
                        point.worldZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)) ==
                    boundary.hydrologyOwner);
        }
        CHECK(minimumX == boundary.minimumWorld);
        CHECK(minimumZ == boundary.minimumWorld);
        CHECK(maximumX == boundary.maximumWorld);
        CHECK(maximumZ == boundary.maximumWorld);
    }
}

TEST_CASE(
    "Generator v4 ranks locally safe owner candidates without requiring a dry generation band",
    "[bootstrap][generator-v4][spawn][dry-certificate][mask][regression]") {
    constexpr int EDGE = 128;
    constexpr int SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    std::vector<uint8_t> certified(static_cast<size_t>(EDGE) * EDGE, uint8_t{1});
    const Vec3 requested{250.5F, 100.0F, 250.5F};
    const std::vector<V4SpawnAlignedCandidate> first =
        v4RankCertifiedDrySpawnCandidates(requested, 0, 0, EDGE, EDGE, certified);
    REQUIRE_FALSE(first.empty());
    CHECK((first.front() == V4SpawnAlignedCandidate{.worldX = 248, .worldZ = 248}));
    CHECK(v4RankCertifiedDrySpawnCandidates(requested, 0, 0, EDGE, EDGE, certified) == first);

    // A wet point beyond the local five-by-five buffer is ordinary streaming
    // work and cannot reject an otherwise safe center.
    certified.front() = 0;

    const std::vector<V4SpawnAlignedCandidate> withSurroundingWater =
        v4RankCertifiedDrySpawnCandidates(requested, 0, 0, EDGE, EDGE, certified);
    REQUIRE_FALSE(withSurroundingWater.empty());
    CHECK(withSurroundingWater.front() == first.front());

    constexpr int64_t LOCAL_BUFFER_OFFSET =
        V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES * V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    const int64_t localX = first.front().worldX - LOCAL_BUFFER_OFFSET;
    const int64_t localZ = first.front().worldZ - LOCAL_BUFFER_OFFSET;
    REQUIRE(localX >= 0);
    REQUIRE(localZ >= 0);
    certified[static_cast<size_t>(localZ / SPACING) * EDGE +
              static_cast<size_t>(localX / SPACING)] = 0;

    const std::vector<V4SpawnAlignedCandidate> afterLocalFailure =
        v4RankCertifiedDrySpawnCandidates(requested, 0, 0, EDGE, EDGE, certified);
    REQUIRE_FALSE(afterLocalFailure.empty());
    CHECK(std::ranges::find(afterLocalFailure, first.front()) == afterLocalFailure.end());
}

TEST_CASE("Generator v4 canonical-water fallback retains lower protected owner cost",
          "[bootstrap][generator-v4][spawn][water-screen][relocation][authority][performance]"
          "[regression]") {
    constexpr int EDGE =
        worldgen::NATIVE_HYDROLOGY_PAGE_EDGE / V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    constexpr int64_t ORIGIN_X = 0;
    constexpr int64_t ORIGIN_Z = 0;
    constexpr V4SpawnAlignedCandidate NEAREST_HIGH_COST{.worldX = 640, .worldZ = 128};
    constexpr V4SpawnAlignedCandidate LOWER_COST{.worldX = 636, .worldZ = 128};
    const std::vector<uint8_t> certified(static_cast<size_t>(EDGE) * EDGE, uint8_t{1});
    const Vec3 requested{static_cast<float>(NEAREST_HIGH_COST.worldX) + 0.5F, 120.0F,
                         static_cast<float>(NEAREST_HIGH_COST.worldZ) + 0.5F};

    const std::vector<V4SpawnAlignedCandidate> ranked =
        v4RankCertifiedDrySpawnCandidates(requested, ORIGIN_X, ORIGIN_Z, EDGE, EDGE, certified);
    REQUIRE_FALSE(ranked.empty());
    const auto nearest = std::ranges::find(ranked, NEAREST_HIGH_COST);
    const auto lower = std::ranges::find(ranked, LOWER_COST);
    REQUIRE(nearest != ranked.end());
    REQUIRE(lower != ranked.end());

    const auto nearestCost =
        v4SpawnPlacementAuthorityCost(NEAREST_HIGH_COST.worldX, NEAREST_HIGH_COST.worldZ);
    const auto lowerCost = v4SpawnPlacementAuthorityCost(LOWER_COST.worldX, LOWER_COST.worldZ);
    const auto selectedCost =
        v4SpawnPlacementAuthorityCost(ranked.front().worldX, ranked.front().worldZ);
    REQUIRE(nearestCost);
    REQUIRE(lowerCost);
    REQUIRE(selectedCost);
    CHECK(nearestCost->protectedDirectOwnerCount == 6);
    CHECK(lowerCost->protectedDirectOwnerCount == 4);
    CHECK(v4SpawnPlacementAuthorityPreferred(*lowerCost, *nearestCost));
    CHECK(lower < nearest);
    CHECK(selectedCost->protectedDirectOwnerCount == lowerCost->protectedDirectOwnerCount);
    CHECK(v4SpawnPlacementAuthorityPreferred(*selectedCost, *nearestCost));
    CHECK(v4RankCertifiedDrySpawnCandidates(requested, ORIGIN_X, ORIGIN_Z, EDGE, EDGE, certified) ==
          ranked);
}

TEST_CASE("Generator v4 certified spawn ranking preserves negative half-open chunks",
          "[bootstrap][generator-v4][spawn][dry-certificate][negative][regression]") {
    constexpr int EDGE = 128;
    constexpr int64_t ORIGIN = -512;
    const std::vector<uint8_t> certified(static_cast<size_t>(EDGE) * EDGE, uint8_t{1});
    const std::vector<V4SpawnAlignedCandidate> ranked = v4RankCertifiedDrySpawnCandidates(
        Vec3{-250.5F, 100.0F, -250.5F}, ORIGIN, ORIGIN, EDGE, EDGE, certified);
    REQUIRE_FALSE(ranked.empty());
    CHECK((ranked.front() == V4SpawnAlignedCandidate{.worldX = -252, .worldZ = -252}));
    CHECK(Chunk::worldToChunk(ranked.front().worldX) == -16);
    CHECK(Chunk::worldToChunk(ranked.front().worldZ) == -16);
    CHECK(ranked.front().worldX % V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == 0);
    CHECK(ranked.front().worldZ % V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == 0);
}

TEST_CASE("Generator v4 resolves a narrow river candidate to a buffered dry bank",
          "[bootstrap][generator-v4][spawn][water-screen][local-refinement][river][regression]") {
    constexpr int EDGE = 11;
    constexpr int RIVER_COLUMN = 5;
    std::vector<worldgen::HydrologySample> samples(static_cast<size_t>(EDGE) * EDGE);
    for (worldgen::HydrologySample& sample : samples) {
        sample.surfaceElevation = 100.0;
        sample.waterSurface = 0.0;
    }
    for (int z = 0; z < EDGE; ++z) {
        worldgen::HydrologySample& river =
            samples[static_cast<size_t>(z) * EDGE + static_cast<size_t>(RIVER_COLUMN)];
        river.river = true;
        river.surfaceElevation = 99.0;
        river.waterSurface = 100.0;
    }

    const Vec3 requested{
        static_cast<float>(RIVER_COLUMN * V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS) + 0.5F,
        100.05F,
        static_cast<float>(RIVER_COLUMN * V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS) + 0.5F,
    };
    const std::optional<Vec3> resolved =
        v4SelectLocalDrySpawnCandidate(requested, 0, 0, EDGE, EDGE, samples);
    REQUIRE(resolved);
    // A five-by-five dry buffer needs two native samples between its center
    // and the river. The screen evaluates native raster coordinates, while
    // the request is centered inside its native cell, so the east bank is
    // half a block closer and wins deterministically.
    CHECK(resolved->x == 32.5F);
    CHECK(resolved->z == 20.5F);
    CHECK(resolved->y == 100.05F);
}

TEST_CASE("Generator v4 requires the complete dry buffer for every resolved spawn",
          "[bootstrap][generator-v4][spawn][water-screen][local-refinement][buffer][regression]") {
    constexpr int EDGE = 5;
    std::vector<worldgen::HydrologySample> samples(static_cast<size_t>(EDGE) * EDGE);
    for (worldgen::HydrologySample& sample : samples) {
        sample.surfaceElevation = 100.0;
        sample.waterSurface = 0.0;
    }
    // The requested center itself is dry. A single river sample in its
    // required five-by-five neighborhood must still reject it, and this
    // compact grid leaves no alternate local point to return.
    samples.front().river = true;
    samples.front().surfaceElevation = 99.0;
    samples.front().waterSurface = 100.0;
    const std::optional<Vec3> resolved =
        v4SelectLocalDrySpawnCandidate(Vec3{8.5F, 100.05F, 8.5F}, 0, 0, EDGE, EDGE, samples);
    CHECK_FALSE(resolved);
}

TEST_CASE(
    "Generator v4 local refinement preserves a negative half-open grid",
    "[bootstrap][generator-v4][spawn][water-screen][local-refinement][negative][regression]") {
    constexpr int EDGE = 5;
    std::vector<worldgen::HydrologySample> samples(static_cast<size_t>(EDGE) * EDGE);
    for (worldgen::HydrologySample& sample : samples) {
        sample.surfaceElevation = 100.0;
        sample.waterSurface = 0.0;
    }
    // The last native point is world -4, immediately before page zero's
    // half-open boundary. Floor division must keep every selected coordinate
    // in page -1 rather than truncating it into page zero.
    const std::optional<Vec3> resolved = v4SelectLocalDrySpawnCandidate(
        Vec3{-11.5F, 100.05F, -11.5F}, -20, -20, EDGE, EDGE, samples);
    REQUIRE(resolved);
    CHECK(resolved->x == -11.5F);
    CHECK(resolved->z == -11.5F);
    const worldgen::learned::NativePoint native =
        worldgen::learned::worldBlockToNative(static_cast<int64_t>(std::floor(resolved->x)),
                                              static_cast<int64_t>(std::floor(resolved->z)));
    CHECK(worldgen::learned::floorDivide(native.row,
                                         worldgen::learned::AUTHORITY_PAGE_NATIVE_EDGE) == -1);
    CHECK(worldgen::learned::floorDivide(native.column,
                                         worldgen::learned::AUTHORITY_PAGE_NATIVE_EDGE) == -1);
}

TEST_CASE("Generator v4 five-by-five spawn validation cannot cross a chunk boundary",
          "[bootstrap][generator-v4][spawn][water-screen][local-refinement][chunk-boundary]"
          "[regression]") {
    STATIC_REQUIRE(V4_CERTIFIED_SPAWN_READY_PLAN_RADIUS_CHUNKS == 0);
    constexpr int EDGE = 5;
    constexpr int64_t ORIGIN = -8;
    std::vector<worldgen::HydrologySample> samples(static_cast<size_t>(EDGE) * EDGE);
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            worldgen::HydrologySample& sample =
                samples[static_cast<size_t>(z) * EDGE + static_cast<size_t>(x)];
            sample.surfaceElevation = 100.0 + std::abs(x - 2) * 0.5 + std::abs(z - 2) * 0.25;
            sample.waterSurface = 0.0;
        }
    }
    const std::optional<Vec3> resolved = v4SelectLocalDrySpawnCandidate(
        Vec3{0.5F, 100.0F, 0.5F}, ORIGIN, ORIGIN, EDGE, EDGE, samples);
    REQUIRE(resolved);
    CHECK(resolved->x == 0.5F);
    CHECK(resolved->z == 0.5F);
    CHECK(Chunk::worldToChunk(static_cast<int64_t>(std::floor(resolved->x))) == 0);
    CHECK(Chunk::worldToChunk(static_cast<int64_t>(std::floor(resolved->z))) == 0);
}

TEST_CASE("Generator v4 safe-spawn water screening fails closed after no authority progress",
          "[bootstrap][generator-v4][spawn][water-screen][watchdog][regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0008ULL);
    auto authority = std::make_shared<PerpetuallyDeferredSpawnWaterAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen({
        .retryInterval = 1ms,
        .noProgressTimeout = 25ms,
    });

    const auto startedAt = std::chrono::steady_clock::now();
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{2.5F, 96.0F, 2.5F});
    const auto elapsed = std::chrono::steady_clock::now() - startedAt;
    REQUIRE(result.failed());
    REQUIRE(result.failure);
    CHECK(result.failure->code == GenerationFailureCode::INFERENCE_FAILED);
    CHECK(result.failure->retriable);
    CHECK(result.failure->message.find("no authority or hydrology progress") != std::string::npos);
    CHECK(authority->requests() > 0);
    CHECK(elapsed < 2s);
    // The watchdog owns only this screen. It fails closed for startup while
    // leaving the shared authority retriable when the user selects Retry.
    CHECK_FALSE(context->failure());

    screen.reset();
    const V4SpawnWaterScreenResult retry = screen.screen(context, Vec3{2.5F, 96.0F, 2.5F});
    CHECK(retry.deferred());
    screen.reset();
}

TEST_CASE("Generator v4 safe-spawn water screening uses an immutable absolute deadline",
          "[bootstrap][generator-v4][spawn][water-screen][watchdog][active-work][regression]") {
    using namespace worldgen::learned;
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'000BULL);
    // The fake authority remains deferred, advertises one admitted active
    // build, and advances a completion counter on every observation. Neither
    // active work nor continuous reported progress may renew the request's
    // absolute deadline.
    auto authority = std::make_shared<PerpetuallyDeferredSpawnWaterAuthority>(identity, true, true);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    V4SpawnWaterScreen screen({
        .retryInterval = 1ms,
        .noProgressTimeout = 5ms,
        .activeWorkTimeout = 35ms,
    });

    const auto startedAt = std::chrono::steady_clock::now();
    const V4SpawnWaterScreenResult result =
        awaitV4SpawnWaterScreen(screen, context, Vec3{2.5F, 96.0F, 2.5F});
    const auto elapsed = std::chrono::steady_clock::now() - startedAt;
    REQUIRE(result.failed());
    REQUIRE(result.failure);
    CHECK(result.failure->code == GenerationFailureCode::INFERENCE_FAILED);
    CHECK(result.failure->message.find("absolute 35 millisecond request deadline") !=
          std::string::npos);
    CHECK(elapsed >= 25ms);
    CHECK(elapsed < 2s);
    CHECK(authority->requests() > 0);
    CHECK_FALSE(context->failure());
    screen.reset();
}

TEST_CASE(
    "Generator v4 dry-land search uses one stable grid inside its physical 64-kilometer bound",
    "[bootstrap][generator-v4][spawn][dry-land][search-range][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_expanded_spawn_search");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0002ULL);

    // This patch lies in the outer part of the one stable 16-cell page-aligned
    // grid, whose half-edge is 61.44 km at Rycraft's 7.5-meter block scale.
    constexpr TerrainPageCoordinate LAND_IN_STABLE_SEARCH{.row = -5, .column = -5};
    auto backend = std::make_shared<SpawnTerrainBackend>(LAND_IN_STABLE_SEARCH);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    REQUIRE(V4_DRY_SPAWN_SEARCH_TARGET_RADIUS_METERS == 64'000.0);
    REQUIRE(V4_DRY_SPAWN_COARSE_CELL_EDGE_METERS == 7'680.0);
    REQUIRE(V4_DRY_SPAWN_SEARCH_REPRESENTABLE_HALF_EDGE_METERS == 61'440.0);
    REQUIRE(V4_DRY_SPAWN_SEARCH_REPRESENTABLE_HALF_EDGE_METERS <=
            V4_DRY_SPAWN_SEARCH_TARGET_RADIUS_METERS);
    REQUIRE(V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE == 16);
    REQUIRE(V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES == 81);
    const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(candidate.isReady());
    REQUIRE(*candidate.value());
    const Vec3 position = **candidate.value();
    const NativePoint native = worldBlockToNative(static_cast<int64_t>(std::floor(position.x)),
                                                  static_cast<int64_t>(std::floor(position.z)));
    CHECK(std::abs(floorDivide(native.row, AUTHORITY_PAGE_NATIVE_EDGE) -
                   LAND_IN_STABLE_SEARCH.row) <= 1);
    CHECK(std::abs(floorDivide(native.column, AUTHORITY_PAGE_NATIVE_EDGE) -
                   LAND_IN_STABLE_SEARCH.column) <= 1);
    const std::vector<CoarseSpawnRegion> requests = backend->coarseRegions();
    REQUIRE(requests.size() == 1);
    CHECK(requests.front().height() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
    CHECK(requests.front().width() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
}

TEST_CASE("Generator v4 dry-land ordinals enumerate a stable coarse grid without repeats",
          "[bootstrap][generator-v4][spawn][dry-land][ordinal][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_stable_dry_spawn_ordinals");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0004ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};
    const int64_t allLandRadius = V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE / 2;
    auto backend = std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, allLandRadius);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    constexpr uint32_t CHECKED_ORDINALS = 9;
    std::set<std::pair<int64_t, int64_t>> seenOwners;
    for (uint32_t ordinal = 0; ordinal < CHECKED_ORDINALS; ++ordinal) {
        const auto candidate = awaitDryLandSpawnCandidate(context, 0, 0, ordinal);
        REQUIRE(candidate.isReady());
        REQUIRE(*candidate.value());
        const Vec3 position = **candidate.value();
        const NativePoint native = worldBlockToNative(static_cast<int64_t>(std::floor(position.x)),
                                                      static_cast<int64_t>(std::floor(position.z)));
        const TerrainPageCoordinate page{
            .row = floorDivide(native.row, AUTHORITY_PAGE_NATIVE_EDGE),
            .column = floorDivide(native.column, AUTHORITY_PAGE_NATIVE_EDGE),
        };
        CAPTURE(ordinal, page.row, page.column);
        const std::pair<int64_t, int64_t> owner{
            floorDivide(page.column, int64_t{2}),
            floorDivide(page.row, int64_t{2}),
        };
        CHECK(seenOwners.insert(owner).second);
    }

    CHECK(backend->coarseCalls() == 1);
    const std::vector<CoarseSpawnRegion> requests = backend->coarseRegions();
    REQUIRE(requests.size() == 1);
    CHECK(requests.front().height() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
    CHECK(requests.front().width() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
}

TEST_CASE("Generator v4 tries ocean-backed land before owner-ambiguous interiors",
          "[bootstrap][generator-v4][spawn][dry-land][coarse-priority][performance]"
          "[regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_ocean_backed_spawn_priority");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0014ULL);
    constexpr TerrainPageCoordinate ORIGIN_PAGE{.row = 0, .column = 0};
    auto backend =
        std::make_shared<SpawnTerrainBackend>(ORIGIN_PAGE, ORIGIN_PAGE, 1, false, uint16_t{0});
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const AuthorityResult<std::optional<Vec3>> selected =
        awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(selected.isReady());
    REQUIRE(*selected.value());
    const Vec3 proposal = **selected.value();
    const NativePoint native = worldBlockToNative(static_cast<int64_t>(std::floor(proposal.x)),
                                                  static_cast<int64_t>(std::floor(proposal.z)));
    CHECK((TerrainPageCoordinate{
               .row = floorDivide(native.row, AUTHORITY_PAGE_NATIVE_EDGE),
               .column = floorDivide(native.column, AUTHORITY_PAGE_NATIVE_EDGE),
           } == TerrainPageCoordinate{.row = -1, .column = -1}));

    // Canonical hydrology still decides whether this proposal is playable.
    // This regression covers only the cheap ordering decision that prevents
    // startup from materializing owner-ambiguous interior pages first.
    CHECK(backend->transientGridCalls() == 0);
}

TEST_CASE("Generator v4 fails closed when no eligible land lies inside the physical search",
          "[bootstrap][generator-v4][spawn][dry-land][fail-closed][regression]") {
    using namespace worldgen::learned;
    TempDir directory("v4_physical_bounded_ocean_spawn");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0003ULL);
    // This patch starts on the positive half-open edge of the 16-cell square.
    // It must not trigger final authority outside the 61.44-kilometer search.
    constexpr TerrainPageCoordinate LAND_BEYOND_REPRESENTABLE_EDGE{.row = 8, .column = 8};
    auto backend = std::make_shared<SpawnTerrainBackend>(LAND_BEYOND_REPRESENTABLE_EDGE,
                                                         LAND_BEYOND_REPRESENTABLE_EDGE, 0);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto result = awaitDryLandSpawnCandidate(context, 0, 0, 0);
    REQUIRE(result.status() == AuthorityStatus::FAILED);
    REQUIRE(result.failure());
    CHECK(result.failure()->code == GenerationFailureCode::INVALID_REQUEST);
    CHECK_FALSE(result.value());
    CHECK(backend->finalPageCalls() == 0);
    const std::vector<CoarseSpawnRegion> requests = backend->coarseRegions();
    REQUIRE(requests.size() == 1);
    CHECK(requests.front().height() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
    CHECK(requests.front().width() == V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
}

TEST_CASE("Generator v4 revalidates legacy finalized spawns before world entry",
          "[bootstrap][generator-v4][spawn][dry-land][migration]") {
    constexpr uint32_t CURRENT = SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
    REQUIRE(v4SpawnRequiresStrictDryValidation(false, 0, false));
    REQUIRE(v4SpawnRequiresStrictDryValidation(true, 0, false));
    REQUIRE(v4SpawnRequiresStrictDryValidation(true, CURRENT - 1, true));
    REQUIRE(v4SpawnRequiresStrictDryValidation(true, CURRENT, false));
    REQUIRE_FALSE(v4SpawnRequiresStrictDryValidation(true, CURRENT, true));
    // An unknown future or corrupted revision cannot be presumed to have
    // applied this build's canonical dry-spawn contract.
    REQUIRE(v4SpawnRequiresStrictDryValidation(true, CURRENT + 1, true));
}

TEST_CASE("Generator v4 recovers legacy ocean resumes from the requested spawn anchor",
          "[bootstrap][generator-v4][spawn][dry-land][migration][ocean-resume][regression]") {
    using namespace worldgen::learned;
    constexpr Vec3 REQUESTED_SPAWN = GENERATOR_V4_INITIAL_SPAWN;
    constexpr Vec3 STALE_OCEAN_PLAYER{250'000.5F, 63.0F, -250'000.5F};
    constexpr Vec3 VERIFIED_SAFE_SPAWN{4'098.5F, 96.0F, -2'046.5F};

    // A record without safeSpawnPos has no trustworthy land anchor. Its old
    // player position may be many ocean-only search regions away from the
    // canonical fresh-world start, so it must not control recovery.
    const Vec3 recoveredAnchor = v4DrySpawnSearchOrigin(std::nullopt, REQUESTED_SPAWN);
    CHECK(recoveredAnchor == REQUESTED_SPAWN);
    CHECK(recoveredAnchor != STALE_OCEAN_PLAYER);
    CHECK(v4DrySpawnSearchOrigin(VERIFIED_SAFE_SPAWN, REQUESTED_SPAWN) == VERIFIED_SAFE_SPAWN);

    TempDir directory("v4_stale_ocean_recovery_anchor");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'0009ULL);
    constexpr TerrainPageCoordinate REQUESTED_LAND_PAGE{.row = 0, .column = 0};
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<SpawnTerrainBackend>(REQUESTED_LAND_PAGE));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    // The stale ocean coordinate has no coarse land within its bounded
    // physical search, whereas recovery from the requested anchor does.
    const auto stale =
        awaitDryLandSpawnCandidate(context, static_cast<int64_t>(std::floor(STALE_OCEAN_PLAYER.x)),
                                   static_cast<int64_t>(std::floor(STALE_OCEAN_PLAYER.z)), 0);
    REQUIRE(stale.status() == AuthorityStatus::FAILED);

    const auto recovered =
        awaitDryLandSpawnCandidate(context, static_cast<int64_t>(std::floor(recoveredAnchor.x)),
                                   static_cast<int64_t>(std::floor(recoveredAnchor.z)), 0);
    REQUIRE(recovered.isReady());
    REQUIRE(*recovered.value());
}

TEST_CASE("Generator v4 retries a stale legacy safe spawn from the requested anchor",
          "[bootstrap][generator-v4][spawn][dry-land][migration][safe-ocean][regression]") {
    using namespace worldgen::learned;
    constexpr Vec3 REQUESTED_SPAWN = GENERATOR_V4_INITIAL_SPAWN;
    constexpr Vec3 STALE_LEGACY_SAFE_SPAWN{250'000.5F, 63.0F, -250'000.5F};

    const V4DrySpawnRecoverySearch legacySearch =
        v4DrySpawnRecoverySearch(STALE_LEGACY_SAFE_SPAWN, REQUESTED_SPAWN, true);
    CHECK(legacySearch.primary == STALE_LEGACY_SAFE_SPAWN);
    REQUIRE(legacySearch.fallback.has_value());
    CHECK(*legacySearch.fallback == REQUESTED_SPAWN);

    // A current-revision record retains an intentional water resume. It never
    // enters strict recovery or substitutes the requested fresh-world anchor.
    const V4DrySpawnRecoverySearch currentSearch =
        v4DrySpawnRecoverySearch(STALE_LEGACY_SAFE_SPAWN, REQUESTED_SPAWN, false);
    CHECK(currentSearch.primary == STALE_LEGACY_SAFE_SPAWN);
    CHECK_FALSE(currentSearch.fallback.has_value());

    const V4DrySpawnRecoverySearch sameHorizontalSearch = v4DrySpawnRecoverySearch(
        Vec3{REQUESTED_SPAWN.x, 63.0F, REQUESTED_SPAWN.z}, REQUESTED_SPAWN, true);
    CHECK_FALSE(sameHorizontalSearch.fallback.has_value());

    TempDir directory("v4_stale_safe_spawn_fallback");
    const GenerationIdentity identity = spawnTestIdentity(0xF1A1'0000'000AULL);
    constexpr TerrainPageCoordinate REQUESTED_LAND_PAGE{.row = 0, .column = 0};
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<SpawnTerrainBackend>(REQUESTED_LAND_PAGE));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    const auto stale = awaitDryLandSpawnCandidate(
        context, static_cast<int64_t>(std::floor(legacySearch.primary.x)),
        static_cast<int64_t>(std::floor(legacySearch.primary.z)), 0);
    REQUIRE(stale.status() == AuthorityStatus::FAILED);

    const auto recovered = awaitDryLandSpawnCandidate(
        context, static_cast<int64_t>(std::floor(legacySearch.fallback->x)),
        static_cast<int64_t>(std::floor(legacySearch.fallback->z)), 0);
    REQUIRE(recovered.isReady());
    REQUIRE(*recovered.value());
}

TEST_CASE("Generator v4 keeps a verified safe spawn separate from the player position",
          "[bootstrap][generator-v4][spawn][persistence][regression]") {
    TempDir directory("v4_safe_spawn_separation");
    constexpr uint64_t SEED = 0x0123'4567'89AB'CDEFULL;
    constexpr std::string_view FINGERPRINT{
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    constexpr Vec3 OCEAN_PLAYER_POSITION{4'096.5F, 63.0F, -2'048.5F};
    constexpr Vec3 VERIFIED_SAFE_SPAWN{128.5F, 83.0F, -64.5F};

    SaveManager saves(directory.path(), SaveManager::Profile::GeneratorV4);
    REQUIRE(saves.saveV4Metadata(SEED, FINGERPRINT, OCEAN_PLAYER_POSITION, VERIFIED_SAFE_SPAWN,
                                 9'876, true, SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION));
    const auto saved = saves.loadMetadata();
    REQUIRE(saved.has_value());
    CHECK(saved->playerPos == OCEAN_PLAYER_POSITION);
    REQUIRE(saved->safeSpawnPos.has_value());
    CHECK(*saved->safeSpawnPos == VERIFIED_SAFE_SPAWN);
    CHECK_FALSE(saves.saveV4Metadata(SEED, FINGERPRINT, OCEAN_PLAYER_POSITION, std::nullopt, 9'876,
                                     true, SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION));
}

TEST_CASE("Generator v4 revalidates revision-three ocean resumes but preserves revision-four ones",
          "[bootstrap][generator-v4][spawn][persistence][migration][ocean-resume][regression]") {
    TempDir directory("v4_ocean_resume_safety_revision");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    constexpr uint64_t SEED = 0xCAFE'BEEF'1020'3040ULL;
    constexpr Vec3 OCEAN_PLAYER_POSITION{4'096.5F, 63.0F, -2'048.5F};
    constexpr Vec3 VERIFIED_SAFE_SPAWN{128.5F, 83.0F, -64.5F};
    static_assert(SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION == 4);
    constexpr uint32_t PREVIOUS_SAFETY_REVISION =
        SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION - 1;

    FakeRuntime runtime;
    runtime.selectIdentity(SEED);
    TerrainGenerationBootstrap bootstrap(installer, runtime);
    REQUIRE(bootstrap.run());

    V4WorldOpenResult created = openQualifiedV4World(bootstrap, SEED);
    REQUIRE(created.ready());
    REQUIRE(created.saveManager->saveV4Metadata(SEED, runtime.fingerprint, OCEAN_PLAYER_POSITION,
                                                VERIFIED_SAFE_SPAWN, 12'345, true,
                                                PREVIOUS_SAFETY_REVISION));
    created.saveManager.reset();

    // Revision three may retain a separate safe spawn, but it predates the
    // current selector and must therefore reopen as provisional rather than
    // accepting the stale ocean player coordinate as a completed start.
    V4WorldOpenResult revisionThree = openQualifiedV4World(bootstrap, SEED);
    REQUIRE(revisionThree.ready());
    CHECK(revisionThree.fresh);
    CHECK(revisionThree.metadata.playerPos == OCEAN_PLAYER_POSITION);
    REQUIRE(revisionThree.metadata.safeSpawnPos.has_value());
    CHECK(*revisionThree.metadata.safeSpawnPos == VERIFIED_SAFE_SPAWN);
    CHECK(v4SpawnRequiresStrictDryValidation(revisionThree.metadata.spawnFinalized,
                                             revisionThree.metadata.spawnSafetyRevision,
                                             revisionThree.metadata.safeSpawnPos.has_value()));

    // A revision-four record retains an intentional ocean resume separately
    // from the verified return point. Startup can resume playerPos without
    // replacing or weakening the safe-spawn contract.
    REQUIRE(revisionThree.saveManager->saveV4Metadata(
        SEED, runtime.fingerprint, OCEAN_PLAYER_POSITION, VERIFIED_SAFE_SPAWN, 12'346, true,
        SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION));
    revisionThree.saveManager.reset();

    V4WorldOpenResult revisionFour = openQualifiedV4World(bootstrap, SEED);
    REQUIRE(revisionFour.ready());
    CHECK_FALSE(revisionFour.fresh);
    CHECK(revisionFour.metadata.playerPos == OCEAN_PLAYER_POSITION);
    REQUIRE(revisionFour.metadata.safeSpawnPos.has_value());
    CHECK(*revisionFour.metadata.safeSpawnPos == VERIFIED_SAFE_SPAWN);
    CHECK_FALSE(v4SpawnRequiresStrictDryValidation(revisionFour.metadata.spawnFinalized,
                                                   revisionFour.metadata.spawnSafetyRevision,
                                                   revisionFour.metadata.safeSpawnPos.has_value()));
}

TEST_CASE("Generator v4 treats pre-split metadata as unvalidated",
          "[bootstrap][generator-v4][spawn][persistence][migration]") {
    TempDir directory("v4_legacy_spawn_metadata");
    constexpr uint64_t SEED = 0x0BAD'F00D'1234'5678ULL;
    constexpr std::string_view FINGERPRINT{
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"};
    constexpr Vec3 LEGACY_PLAYER_POSITION{1'024.5F, 64.0F, -512.5F};

    SaveManager saves(directory.path(), SaveManager::Profile::GeneratorV4);
    const std::filesystem::path metadataPath =
        std::filesystem::path(directory.path()) / "metadata.json";
    std::ofstream legacy(metadataPath, std::ios::trunc);
    REQUIRE(legacy.is_open());
    legacy << "{\n"
           << "  \"seed\": " << SEED << ",\n"
           << "  \"generationFingerprint\": \"" << FINGERPRINT << "\",\n"
           << "  \"spawnFinalized\": true,\n"
           << "  \"spawnSafetyRevision\": " << SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION
           << ",\n"
           << "  \"spawnPos\": {\n"
           << "    \"x\": " << LEGACY_PLAYER_POSITION.x << ",\n"
           << "    \"y\": " << LEGACY_PLAYER_POSITION.y << ",\n"
           << "    \"z\": " << LEGACY_PLAYER_POSITION.z << "\n"
           << "  },\n"
           << "  \"worldTime\": 123,\n"
           << "  \"chunkFormatVersion\": " << CHUNK_VERSION << ",\n"
           << "  \"generatorVersion\": " << SaveManager::GENERATOR_V4_VERSION << "\n"
           << "}\n";
    legacy.close();
    REQUIRE(legacy.good());

    const auto loaded = saves.loadMetadata();
    REQUIRE(loaded.has_value());
    CHECK(loaded->playerPos == LEGACY_PLAYER_POSITION);
    CHECK_FALSE(loaded->safeSpawnPos.has_value());
    REQUIRE(v4SpawnRequiresStrictDryValidation(loaded->spawnFinalized, loaded->spawnSafetyRevision,
                                               loaded->safeSpawnPos.has_value()));
}

TEST_CASE("Generator v4 drops a stale safe spawn from a provisional record",
          "[bootstrap][generator-v4][spawn][persistence][corruption][regression]") {
    TempDir directory("v4_provisional_safe_spawn_metadata");
    constexpr uint64_t SEED = 0x0BAD'F00D'FEDC'BA98ULL;
    constexpr std::string_view FINGERPRINT{
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    constexpr Vec3 PLAYER_POSITION{1'024.5F, 64.0F, -512.5F};
    constexpr Vec3 STALE_SAFE_POSITION{-4'096.5F, 96.0F, 2'048.5F};

    SaveManager saves(directory.path(), SaveManager::Profile::GeneratorV4);
    const std::filesystem::path metadataPath =
        std::filesystem::path(directory.path()) / "metadata.json";
    std::ofstream metadata(metadataPath, std::ios::trunc);
    REQUIRE(metadata.is_open());
    metadata << "{\n"
             << "  \"seed\": " << SEED << ",\n"
             << "  \"generationFingerprint\": \"" << FINGERPRINT << "\",\n"
             << "  \"spawnFinalized\": false,\n"
             << "  \"spawnSafetyRevision\": " << SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION
             << ",\n"
             << "  \"playerPos\": {\n"
             << "    \"x\": " << PLAYER_POSITION.x << ",\n"
             << "    \"y\": " << PLAYER_POSITION.y << ",\n"
             << "    \"z\": " << PLAYER_POSITION.z << "\n"
             << "  },\n"
             << "  \"safeSpawnPos\": {\n"
             << "    \"x\": " << STALE_SAFE_POSITION.x << ",\n"
             << "    \"y\": " << STALE_SAFE_POSITION.y << ",\n"
             << "    \"z\": " << STALE_SAFE_POSITION.z << "\n"
             << "  },\n"
             << "  \"worldTime\": 123,\n"
             << "  \"chunkFormatVersion\": " << CHUNK_VERSION << ",\n"
             << "  \"generatorVersion\": " << SaveManager::GENERATOR_V4_VERSION << "\n"
             << "}\n";
    metadata.close();
    REQUIRE(metadata.good());

    const auto loaded = saves.loadMetadata();
    REQUIRE(loaded.has_value());
    CHECK(loaded->playerPos == PLAYER_POSITION);
    CHECK_FALSE(loaded->safeSpawnPos.has_value());
    CHECK(v4SpawnRequiresStrictDryValidation(loaded->spawnFinalized, loaded->spawnSafetyRevision,
                                             loaded->safeSpawnPos.has_value()));
}

TEST_CASE("Generator v4 cannot prepare a horizon before final spawn validation",
          "[bootstrap][generator-v4][spawn][horizon][regression]") {
    REQUIRE_FALSE(v4CanonicalDrySpawnAccepted(false, false, false));
    REQUIRE_FALSE(v4CanonicalDrySpawnAccepted(false, true, false));
    REQUIRE(v4CanonicalDrySpawnAccepted(true, false, false));
    REQUIRE_FALSE(v4CanonicalDrySpawnAccepted(true, true, false));
    REQUIRE(v4CanonicalDrySpawnAccepted(false, false, true));
    REQUIRE(v4CanonicalDrySpawnAccepted(false, true, true));
    REQUIRE(v4CanonicalDrySpawnAccepted(true, false, true));
    REQUIRE(v4CanonicalDrySpawnAccepted(true, true, true));

    REQUIRE_FALSE(v4MayPrepareHorizon(false, false));
    REQUIRE_FALSE(v4MayPrepareHorizon(false, true));
    REQUIRE_FALSE(v4MayPrepareHorizon(true, false));
    REQUIRE(v4MayPrepareHorizon(true, true));

    STATIC_REQUIRE(V4_ENTRY_CONNECTED_PARENT_RADIUS_CHUNKS == 96);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(-1) == 0);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(0) == 0);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(64) == 64);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(96) == 96);
    STATIC_REQUIRE(v4RequiredEntryParentRadiusChunks(512) == 96);
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(0, 0, 0.0F, 0, 0, 0));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 0, 0.0F, 400, 0, 400));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(64, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 64, 64.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 96, 95.99F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 96, 96.0F, 400, 0, 400));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 96, 512.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 96, 513.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(512, 96, 96.0F, 400, 100, 299));
    STATIC_REQUIRE_FALSE(v4EntryConnectedParentReady(64, 64, 64.0F, 100, 99, 1));
    STATIC_REQUIRE(v4EntryConnectedParentReady(512, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE(v4EntryConnectedParentReady(64, 64, 0.0F, 100, 100, 0));

    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(false, true, 2, 512, 512, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, false, 2, 512, 512, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, true, 1, 512, 512, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, true, 2, 0, 0, 0, 0.0F, 0, 0, 0));
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, true, 2, 512, 64, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, true, 2, 512, 512, 64, 64.0F, 400, 100, 300));
    // One missing connected PREVIEW terrain-and-water parent still blocks
    // entry even if every other selected parent is resident.
    STATIC_REQUIRE_FALSE(v4EntryHorizonReady(true, true, 2, 512, 512, 96, 95.99F, 400, 100, 300));
    // The remaining 300 configured-horizon parents may continue streaming
    // after entry because the first 96 chunks are one connected prefix. The
    // 60 protected FINAL targets advance concurrently and are evaluated by the
    // separate atomic near-entry closure gate.
    STATIC_REQUIRE(V4_ENTRY_FINAL_TARGET_COUNT == 60);
    STATIC_REQUIRE(v4EntryHorizonReady(true, true, 2, 512, 512, 96, 96.0F, 400, 100, 300));
    STATIC_REQUIRE(v4EntryHorizonReady(true, true, 2, 64, 64, 64, 0.0F, 100, 100, 0));
}

TEST_CASE("Generator v4 pins every model and runtime asset", "[bootstrap]") {
    const auto& assets = pinnedTerrainAssets();
    REQUIRE(assets.size() == 6);
    REQUIRE(TERRAIN_MODEL_REVISION == "ad2df557eca5645f588766101cf3bc3682455c3e");

    CHECK(assets[0].fileName == "base_model.onnx");
    CHECK(assets[0].byteSize == 2'029'994'361ULL);
    CHECK(assets[0].sha256 == "543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0");
    CHECK(assets[1].fileName == "coarse_model.onnx");
    CHECK(assets[1].byteSize == 22'497'125ULL);
    CHECK(assets[1].sha256 == "d6ca15b21b2e35d5e594a9ac7a4249a2376590c0ad2b5b49a1e6e2d033450008");
    CHECK(assets[2].fileName == "decoder_model.onnx");
    CHECK(assets[2].byteSize == 223'854'143ULL);
    CHECK(assets[2].sha256 == "6473ae47ca6ec4d743d30fe4f5d381fe4158899714eff09b762005bdbdef68c1");
    CHECK(assets[3].fileName == "pipeline_data.json");
    CHECK(assets[3].byteSize == 12'226ULL);
    CHECK(assets[3].sha256 == "e3132c3ef0c65d8613615f9278ffe23bbd9363ddcd87f1cc6f18456bcc9efe5c");
    CHECK(assets[4].fileName == "world_pipeline_config.json");
    CHECK(assets[4].byteSize == 774ULL);
    CHECK(assets[4].sha256 == "c60f0b74d89317e64cfc623fbfdd828f1b5b2e50aa75020ac4001103381853bd");
    CHECK(assets[5].fileName == "onnxruntime-osx-arm64-1.27.1.tgz");
    CHECK(assets[5].byteSize == 31'959'937ULL);
    CHECK(assets[5].sha256 == "e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6");
    CHECK(assets[5].kind == TerrainAssetKind::Runtime);
    CHECK(defaultRycraftApplicationSupportPath().filename() == "rycraft");
}

TEST_CASE("Terrain asset verification checks size and SHA-256", "[bootstrap]") {
    TempDir directory("terrain_asset_verifier");
    const std::filesystem::path path = std::filesystem::path(directory.path()) / "tiny-model.bin";
    std::filesystem::create_directories(path.parent_path());
    {
        std::ofstream output(path, std::ios::binary);
        output << "abc";
    }

    Sha256TerrainAssetVerifier verifier;
    const TerrainVerificationResult valid = verifier.verify(path, tinyAsset());
    REQUIRE(valid.valid);
    CHECK(valid.actualBytes == 3);
    CHECK(valid.actualSha256 == tinyAsset().sha256);

    TerrainAssetSpec wrongSize = tinyAsset();
    wrongSize.byteSize = 4;
    CHECK_FALSE(verifier.verify(path, wrongSize).valid);

    TerrainAssetSpec wrongHash = tinyAsset();
    wrongHash.sha256.assign(64, '0');
    const TerrainVerificationResult corrupt = verifier.verify(path, wrongHash);
    CHECK_FALSE(corrupt.valid);
    CHECK(corrupt.actualSha256 == tinyAsset().sha256);
}

TEST_CASE("Terrain model installation publishes only a fully verified pack", "[bootstrap]") {
    TempDir directory("terrain_model_install");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    transport.pathThatMustNotExist = installer.installedPackPath();
    TerrainBootstrapCancellation cancellation;
    std::vector<TerrainBootstrapSnapshot> snapshots;

    const TerrainInstallResult first =
        installer.prepare(cancellation, [&](const TerrainBootstrapSnapshot& snapshot) {
            snapshots.push_back(snapshot);
        });
    REQUIRE(first.installed());
    CHECK(first.installedPath == installer.installedPackPath());
    CHECK(std::filesystem::exists(first.installedPath / "tiny-model.bin"));
    CHECK(std::filesystem::exists(first.installedPath / ".verified-pack-v1"));
    CHECK(transport.calls == 1);
    REQUIRE_FALSE(snapshots.empty());
    CHECK(snapshots.front().state == TerrainBootstrapState::ModelRequired);
    CHECK(snapshots.back().completedBytes == installer.totalDownloadBytes());

    const TerrainInstallResult reused = installer.prepare(cancellation, {});
    REQUIRE(reused.installed());
    CHECK(transport.calls == 1);
    CHECK(installer.hasInstalledPackCandidate());

    FakeTransport restartedTransport;
    TerrainModelInstaller restartedInstaller(directory.path(), restartedTransport, verifier,
                                             {tinyAsset()});
    const TerrainInstallResult restarted = restartedInstaller.prepare(cancellation, {});
    REQUIRE(restarted.installed());
    CHECK(restartedTransport.calls == 0);
    transport.pathThatMustNotExist.clear();

    installer.requestRepair();
    const TerrainInstallResult refreshed = installer.prepare(cancellation, {});
    REQUIRE(refreshed.installed());
    CHECK(transport.calls == 1);

    {
        std::ofstream corrupt(first.installedPath / "tiny-model.bin",
                              std::ios::binary | std::ios::trunc);
        corrupt << "abd";
    }
    const TerrainInstallResult rejected = installer.prepare(cancellation, {});
    CHECK_FALSE(rejected.installed());
    CHECK(rejected.failure.code == TerrainBootstrapFailureCode::Integrity);
    CHECK(transport.calls == 1);

    installer.requestRepair();
    const TerrainInstallResult repaired = installer.prepare(cancellation, {});
    REQUIRE(repaired.installed());
    CHECK(transport.calls == 2);
    CHECK(verifier.verify(repaired.installedPath / "tiny-model.bin", tinyAsset()).valid);
}

TEST_CASE("Terrain model setup accepts an explicit external application support root",
          "[bootstrap][playtest]") {
    TempDir directory("terrain_external_application_support");
    ScopedEnvironmentVariable root("RYCRAFT_APPLICATION_SUPPORT_ROOT", directory.path());
    CHECK(defaultRycraftApplicationSupportPath() == directory.path());
}

TEST_CASE("Generator v4 restart reuses the installed model pack without downloading",
          "[bootstrap][restart][regression]") {
    TempDir directory("terrain_bootstrap_restart");
    Sha256TerrainAssetVerifier verifier;
    TerrainBootstrapCancellation cancellation;
    FakeTransport initialTransport;
    TerrainModelInstaller initialInstaller(directory.path(), initialTransport, verifier,
                                           {tinyAsset()});
    REQUIRE(initialInstaller.prepare(cancellation, {}).installed());
    REQUIRE(initialTransport.calls == 1);

    FakeTransport restartTransport;
    TerrainModelInstaller restartInstaller(directory.path(), restartTransport, verifier,
                                           {tinyAsset()});
    FakeRuntime runtime;
    std::vector<TerrainBootstrapState> states;
    TerrainGenerationBootstrap bootstrap(
        restartInstaller, runtime,
        [&](const TerrainBootstrapSnapshot& snapshot) { states.push_back(snapshot.state); });

    REQUIRE(restartInstaller.hasInstalledPackCandidate());
    REQUIRE(bootstrap.run());
    CHECK(restartTransport.calls == 0);
    CHECK(compactStates(states) ==
          std::vector<TerrainBootstrapState>{
              TerrainBootstrapState::Verifying, TerrainBootstrapState::Compiling,
              TerrainBootstrapState::Loading, TerrainBootstrapState::Ready});
}

TEST_CASE("Generator v4 restart trusts an unchanged verified-pack stamp",
          "[bootstrap][restart][reuse][performance]") {
    TempDir directory("terrain_bootstrap_marker_fast_path");
    CountingVerifier verifier;
    TerrainBootstrapCancellation cancellation;
    FakeTransport initialTransport;
    TerrainModelInstaller initialInstaller(directory.path(), initialTransport, verifier,
                                           {tinyAsset()});
    REQUIRE(initialInstaller.prepare(cancellation, {}).installed());
    REQUIRE(initialTransport.calls == 1);
    REQUIRE(verifier.calls > 0);

    verifier.calls = 0;
    FakeTransport restartTransport;
    TerrainModelInstaller restartInstaller(directory.path(), restartTransport, verifier,
                                           {tinyAsset()});
    std::vector<TerrainBootstrapSnapshot> snapshots;
    const TerrainInstallResult restarted =
        restartInstaller.prepare(cancellation, [&](const TerrainBootstrapSnapshot& snapshot) {
            snapshots.push_back(snapshot);
        });

    REQUIRE(restarted.installed());
    CHECK(restarted.reusedInstalledPack);
    CHECK(restartTransport.calls == 0);
    CHECK(verifier.calls == 0);
    REQUIRE(snapshots.size() == 1);
    CHECK(snapshots.front().state == TerrainBootstrapState::Verifying);
    CHECK(snapshots.front().reusingInstalledPack);
    CHECK(snapshots.front().detail.find("no download") != std::string::npos);

    const std::filesystem::path asset = restartInstaller.installedPackPath() / "tiny-model.bin";
    const std::filesystem::file_time_type originalTime = std::filesystem::last_write_time(asset);
    std::filesystem::last_write_time(asset, originalTime + std::chrono::seconds(1));

    verifier.calls = 0;
    FakeTransport changedTransport;
    TerrainModelInstaller changedInstaller(directory.path(), changedTransport, verifier,
                                           {tinyAsset()});
    const TerrainInstallResult audited = changedInstaller.prepare(cancellation, {});
    REQUIRE(audited.installed());
    CHECK(audited.reusedInstalledPack);
    CHECK(changedTransport.calls == 0);
    CHECK(verifier.calls == 1);
}

TEST_CASE("Generator v4 restores a missing completion marker without downloading",
          "[bootstrap][restart][reuse][regression]") {
    TempDir directory("terrain_bootstrap_marker_recovery");
    Sha256TerrainAssetVerifier verifier;
    TerrainBootstrapCancellation cancellation;
    FakeTransport initialTransport;
    TerrainModelInstaller initialInstaller(directory.path(), initialTransport, verifier,
                                           {tinyAsset()});
    REQUIRE(initialInstaller.prepare(cancellation, {}).installed());
    REQUIRE(initialTransport.calls == 1);

    const std::filesystem::path marker = initialInstaller.installedPackPath() / ".verified-pack-v1";
    REQUIRE(std::filesystem::remove(marker));
    REQUIRE(
        std::filesystem::is_regular_file(initialInstaller.installedPackPath() / "tiny-model.bin"));

    FakeTransport restartTransport;
    TerrainModelInstaller restartInstaller(directory.path(), restartTransport, verifier,
                                           {tinyAsset()});
    std::vector<TerrainBootstrapSnapshot> snapshots;
    const TerrainInstallResult recovered =
        restartInstaller.prepare(cancellation, [&](const TerrainBootstrapSnapshot& snapshot) {
            snapshots.push_back(snapshot);
        });

    REQUIRE(recovered.installed());
    CHECK(restartTransport.calls == 0);
    CHECK(std::filesystem::is_regular_file(marker));
    REQUIRE_FALSE(snapshots.empty());
    CHECK(std::ranges::all_of(snapshots, [](const TerrainBootstrapSnapshot& snapshot) {
        return snapshot.state == TerrainBootstrapState::Verifying;
    }));
    CHECK(snapshots.back().detail ==
          "Refreshed the verified local terrain model completion marker");
    CHECK(verifier.verify(recovered.installedPath / "tiny-model.bin", tinyAsset()).valid);
}

TEST_CASE("Generator v4 bootstrap gates world construction on qualification", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_ready");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    std::vector<TerrainBootstrapState> states;
    TerrainGenerationBootstrap bootstrap(
        installer, runtime,
        [&](const TerrainBootstrapSnapshot& snapshot) { states.push_back(snapshot.state); });

    CHECK_FALSE(bootstrap.ready());
    CHECK_FALSE(bootstrap.worldPath());
    CHECK_FALSE(
        std::filesystem::exists(std::filesystem::path(directory.path()) / V4_WORLD_DIRECTORY));
    REQUIRE(bootstrap.run());
    REQUIRE(bootstrap.ready());
    REQUIRE(bootstrap.worldPath());
    REQUIRE(bootstrap.qualifiedGenerationFingerprint());
    CHECK(*bootstrap.qualifiedGenerationFingerprint() == runtime.fingerprint);
    CHECK(bootstrap.worldPath()->filename() == V4_WORLD_DIRECTORY);
    CHECK_FALSE(std::filesystem::exists(*bootstrap.worldPath()));

    const std::vector<TerrainBootstrapState> expected{
        TerrainBootstrapState::ModelRequired, TerrainBootstrapState::Downloading,
        TerrainBootstrapState::Verifying,     TerrainBootstrapState::Compiling,
        TerrainBootstrapState::Loading,       TerrainBootstrapState::Ready,
    };
    CHECK(compactStates(states) == expected);
    CHECK(runtime.qualificationCalls == 1);
    CHECK(runtime.compilationCalls == 1);
    CHECK(runtime.loadingCalls == 1);
}

TEST_CASE("Generator v4 launch profile overrides resolve to one exact path",
          "[bootstrap][save][generator-v4][startup][regression]") {
    const std::filesystem::path applicationSupport =
        "/Users/test/Library/Application Support/rycraft";
    CHECK_FALSE(resolveV4LaunchProfilePath(applicationSupport, ""));
    REQUIRE(resolveV4LaunchProfilePath(applicationSupport, V4_WORLD_DIRECTORY));
    CHECK(*resolveV4LaunchProfilePath(applicationSupport, V4_WORLD_DIRECTORY) ==
          applicationSupport / V4_WORLD_DIRECTORY);

    const std::filesystem::path sibling =
        applicationSupport / "rycraft_world_v4-seed-000000000000002a-fingerprint-0123456789abcdef";
    REQUIRE(resolveV4LaunchProfilePath(applicationSupport, sibling.string()));
    CHECK(*resolveV4LaunchProfilePath(applicationSupport, sibling.string()) == sibling);
    REQUIRE(resolveV4LaunchProfilePath(applicationSupport, sibling.string() + "/"));
    CHECK(*resolveV4LaunchProfilePath(applicationSupport, sibling.string() + "/") == sibling);

    REQUIRE(resolveV4LaunchProfilePath(applicationSupport, "profiles/../rycraft_world_v4"));
    CHECK(*resolveV4LaunchProfilePath(applicationSupport, "profiles/../rycraft_world_v4") ==
          applicationSupport / V4_WORLD_DIRECTORY);
}

TEST_CASE("Generator v4 selected paths never become implicit world creation requests",
          "[bootstrap][save][generator-v4][startup][regression]") {
    TempDir directory("v4_selected_profile_must_exist");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    constexpr uint64_t SEED = 42;
    runtime.selectIdentity(SEED);
    TerrainGenerationBootstrap bootstrap(installer, runtime);
    REQUIRE(bootstrap.run());

    const std::filesystem::path selected =
        std::filesystem::path(directory.path()) / V4_WORLD_DIRECTORY;
    const V4WorldOpenResult missing =
        openQualifiedV4World(bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN, 0, nullptr, selected);
    CHECK_FALSE(missing.ready());
    CHECK(missing.status == V4WorldOpenStatus::MissingMetadata);
    CHECK(missing.profilePath == selected);
    CHECK_FALSE(std::filesystem::exists(selected));

    std::filesystem::create_directories(selected);
    const V4WorldOpenResult empty =
        openQualifiedV4World(bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN, 0, nullptr, selected);
    CHECK_FALSE(empty.ready());
    CHECK(empty.status == V4WorldOpenStatus::MissingMetadata);
    CHECK(std::filesystem::is_empty(selected));
}

TEST_CASE("Qualified startup creates only the fingerprinted v4 persistence profile",
          "[bootstrap][save][generator-v4]") {
    TempDir directory("qualified_v4_profile");
    const std::filesystem::path applicationSupport = directory.path();
    const std::filesystem::path legacySentinel = applicationSupport / "rycraft_world" /
                                                 SaveManager::CURRENT_REGIONS_DIRECTORY /
                                                 "sentinel.txt";
    std::filesystem::create_directories(legacySentinel.parent_path());
    {
        std::ofstream sentinel(legacySentinel);
        sentinel << "legacy profile must remain untouched";
    }

    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(applicationSupport, transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    constexpr uint64_t SEED = 0xFEDC'BA98'7654'3210ULL;
    runtime.selectIdentity(SEED);
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    const V4WorldOpenResult gated = openQualifiedV4World(bootstrap, SEED);
    CHECK(gated.status == V4WorldOpenStatus::BootstrapNotReady);
    CHECK_FALSE(std::filesystem::exists(applicationSupport / V4_WORLD_DIRECTORY));

    REQUIRE(bootstrap.run());
    V4WorldOpenResult opened = openQualifiedV4World(bootstrap, SEED, Vec3{4.f, 120.f, -9.f}, 321);
    REQUIRE(opened.ready());
    CHECK(opened.fresh);
    CHECK(opened.metadata.seed == SEED);
    CHECK(opened.metadata.generationFingerprint == runtime.fingerprint);
    CHECK(opened.metadata.generatorVersion == SaveManager::GENERATOR_V4_VERSION);
    CHECK_FALSE(opened.metadata.spawnFinalized);
    CHECK(opened.metadata.spawnSafetyRevision == 0);
    CHECK(opened.metadata.playerPos == Vec3{4.f, 120.f, -9.f});
    CHECK_FALSE(opened.metadata.safeSpawnPos.has_value());
    CHECK(opened.metadata.worldTime == 321);
    CHECK(opened.saveManager->profile() == SaveManager::Profile::GeneratorV4);

    const std::filesystem::path v4World = applicationSupport / V4_WORLD_DIRECTORY;
    CHECK(std::filesystem::exists(v4World / SaveManager::V4_REGIONS_DIRECTORY));
    CHECK(std::filesystem::exists(v4World / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY));
    CHECK(std::filesystem::exists(v4World / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY));
    REQUIRE_FALSE(runtime.boundProfiles.empty());
    CHECK(runtime.boundProfiles.back() == v4World);
    REQUIRE(bootstrap.qualifiedGenerationContext());
    CHECK(bootstrap.qualifiedGenerationContext()->hydrologyAuthorityRoot() ==
          v4World / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
    CHECK_FALSE(std::filesystem::exists(v4World / SaveManager::CURRENT_REGIONS_DIRECTORY));
    CHECK_FALSE(opened.saveManager->saveMetadata(7, {}, 0));
    CHECK_FALSE(opened.saveManager->saveV4Metadata(SEED, "invalid", {}, std::nullopt, 0));
    opened.saveManager.reset();

    V4WorldOpenResult provisional = openQualifiedV4World(bootstrap, SEED);
    REQUIRE(provisional.ready());
    CHECK(provisional.fresh);
    CHECK_FALSE(provisional.metadata.spawnFinalized);
    constexpr Vec3 PLAYER_POSITION{2'304.5F, 63.0F, -1'792.5F};
    constexpr Vec3 SAFE_SPAWN{8.5F, 65.05F, -3.5F};
    constexpr uint32_t SPAWN_SAFETY_REVISION = SaveManager::GENERATOR_V4_SPAWN_SAFETY_REVISION;
    REQUIRE(provisional.saveManager->saveV4Metadata(SEED, runtime.fingerprint, PLAYER_POSITION,
                                                    SAFE_SPAWN, 322, true, SPAWN_SAFETY_REVISION));
    provisional.saveManager.reset();
    V4WorldOpenResult finalized = openQualifiedV4World(bootstrap, SEED);
    REQUIRE(finalized.ready());
    CHECK_FALSE(finalized.fresh);
    CHECK(finalized.metadata.spawnFinalized);
    CHECK(finalized.metadata.spawnSafetyRevision == SPAWN_SAFETY_REVISION);
    CHECK(finalized.metadata.playerPos == PLAYER_POSITION);
    REQUIRE(finalized.metadata.safeSpawnPos.has_value());
    CHECK(*finalized.metadata.safeSpawnPos == SAFE_SPAWN);

    const std::filesystem::path preservedPayload = v4World / "preserve-original-profile.txt";
    {
        std::ofstream preserved(preservedPayload);
        preserved << "do not rewrite this profile";
    }
    const auto readText = [](const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    };
    const std::string originalMetadata = readText(v4World / "metadata.json");
    finalized.saveManager.reset();

    std::ifstream sentinel(legacySentinel);
    std::string legacyContents((std::istreambuf_iterator<char>(sentinel)),
                               std::istreambuf_iterator<char>());
    CHECK(legacyContents == "legacy profile must remain untouched");

    FakeRuntime alternateSeedRuntime;
    alternateSeedRuntime.selectIdentity(SEED + 1);
    TerrainGenerationBootstrap alternateSeedBootstrap(installer, alternateSeedRuntime);
    REQUIRE(alternateSeedBootstrap.run());
    const std::string alternateSeedProfileName = std::string(V4_WORLD_DIRECTORY) +
                                                 "-seed-fedcba9876543211-fingerprint-" +
                                                 alternateSeedRuntime.fingerprint;
    const std::filesystem::path occupiedAlternateSeedPath =
        v4World.parent_path() / alternateSeedProfileName;
    std::filesystem::create_directories(occupiedAlternateSeedPath);
    {
        std::ofstream occupied(occupiedAlternateSeedPath / "keep-this-file.txt");
        occupied << "unrelated profile collision";
    }
    const V4WorldOpenResult conflict = openQualifiedV4World(alternateSeedBootstrap, SEED + 1);
    CHECK_FALSE(conflict.ready());
    CHECK(conflict.status == V4WorldOpenStatus::IdentityConflict);
    CHECK_FALSE(conflict.usingSeparateProfile);
    CHECK(conflict.profilePath == v4World);
    CHECK(conflict.message.find("select a world or create a new one") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(v4World.parent_path() / (alternateSeedProfileName + "-1")));

    const V4WorldOpenResult selectedConflict = openQualifiedV4World(
        alternateSeedBootstrap, SEED + 1, GENERATOR_V4_INITIAL_SPAWN, 0, nullptr, v4World);
    CHECK_FALSE(selectedConflict.ready());
    CHECK(selectedConflict.status == V4WorldOpenStatus::IdentityConflict);
    CHECK(selectedConflict.profilePath == v4World);
    CHECK(selectedConflict.message.find("selected generator v4 profile") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(v4World.parent_path() / (alternateSeedProfileName + "-1")));

    const V4WorldOpenResult reopenedConflict =
        openQualifiedV4World(alternateSeedBootstrap, SEED + 1);
    CHECK_FALSE(reopenedConflict.ready());
    CHECK(reopenedConflict.status == V4WorldOpenStatus::IdentityConflict);
    CHECK(reopenedConflict.profilePath == v4World);

    CHECK(readText(v4World / "metadata.json") == originalMetadata);
    CHECK(readText(preservedPayload) == "do not rewrite this profile");
    CHECK(readText(occupiedAlternateSeedPath / "keep-this-file.txt") ==
          "unrelated profile collision");

    FakeRuntime incompatibleRuntime;
    incompatibleRuntime.selectIdentity(SEED, 0x12U, 0x22U);
    TerrainGenerationBootstrap incompatibleBootstrap(installer, incompatibleRuntime);
    REQUIRE(incompatibleBootstrap.run());
    const V4WorldOpenResult fingerprintConflict = openQualifiedV4World(incompatibleBootstrap, SEED);
    CHECK_FALSE(fingerprintConflict.ready());
    CHECK(fingerprintConflict.status == V4WorldOpenStatus::IdentityConflict);
    CHECK_FALSE(fingerprintConflict.usingSeparateProfile);
    CHECK(fingerprintConflict.profilePath == v4World);
    CHECK(readText(v4World / "metadata.json") == originalMetadata);
    CHECK(readText(preservedPayload) == "do not rewrite this profile");
    CHECK(std::filesystem::exists(legacySentinel));
}

TEST_CASE("Explicit v4 creation never reuses another world with the same identity",
          "[bootstrap][save][generator-v4][worlds][creation][regression]") {
    TempDir directory("qualified_v4_distinct_creation");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    constexpr uint64_t SEED = 0xFFFF'FFFF'1234'5678ULL;
    runtime.selectIdentity(SEED);
    TerrainGenerationBootstrap bootstrap(installer, runtime);
    REQUIRE(bootstrap.run());

    V4WorldCreationRequest firstRequest;
    firstRequest.displayName = "First world";
    firstRequest.gameMode = GameMode::SURVIVAL;
    firstRequest.generation.fauna = false;
    firstRequest.player.inventory.fill(ItemStack{});
    V4WorldOpenResult first = openQualifiedV4World(bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN,
                                                   6'000, nullptr, std::nullopt, firstRequest);
    REQUIRE(first.ready());
    REQUIRE(first.newlyCreated);
    CHECK(first.profilePath.filename() == V4_WORLD_DIRECTORY);
    CHECK(first.metadata.name == "First world");
    CHECK(first.metadata.gameMode == GameMode::SURVIVAL);
    CHECK_FALSE(first.metadata.generation.fauna);
    CHECK(first.metadata.worldTime == 6'000);
    const std::filesystem::path firstPath = first.profilePath;
    first.saveManager.reset();
    std::ifstream beforeInput(firstPath / "metadata.json", std::ios::binary);
    const std::string before((std::istreambuf_iterator<char>(beforeInput)),
                             std::istreambuf_iterator<char>());

    V4WorldCreationRequest secondRequest;
    secondRequest.displayName = "Second world";
    secondRequest.gameMode = GameMode::CREATIVE;
    V4WorldOpenResult second = openQualifiedV4World(bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN, 0,
                                                    nullptr, std::nullopt, secondRequest);
    REQUIRE(second.ready());
    REQUIRE(second.newlyCreated);
    CHECK(second.usingSeparateProfile);
    CHECK(second.profilePath != firstPath);
    CHECK(second.profilePath.filename().string().starts_with(
        std::string(V4_WORLD_DIRECTORY) + "-seed-ffffffff12345678-fingerprint-"));
    CHECK(second.metadata.name == "Second world");
    CHECK(second.metadata.seed == SEED);
    const std::filesystem::path secondPath = second.profilePath;
    second.saveManager.reset();

    V4WorldOpenResult selected =
        openQualifiedV4World(bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN, 0, nullptr, secondPath);
    REQUIRE(selected.ready());
    CHECK(selected.profilePath == secondPath);
    CHECK(selected.usingSeparateProfile);
    CHECK_FALSE(selected.newlyCreated);
    CHECK(selected.metadata.name == "Second world");
    selected.saveManager.reset();

    std::ifstream afterInput(firstPath / "metadata.json", std::ios::binary);
    const std::string after((std::istreambuf_iterator<char>(afterInput)),
                            std::istreambuf_iterator<char>());
    CHECK(after == before);

    const V4WorldOpenResult invalid = openQualifiedV4World(
        bootstrap, SEED, GENERATOR_V4_INITIAL_SPAWN, 0, nullptr, firstPath, secondRequest);
    CHECK(invalid.status == V4WorldOpenStatus::InvalidWorldDirectory);
    CHECK(after == before);
}

TEST_CASE("Fresh v4 metadata failure leaves no visible partial world",
          "[bootstrap][save][generator-v4][atomic][recovery]") {
    TempDir directory("qualified_v4_atomic_failure");
    const std::filesystem::path applicationSupport = directory.path();
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(applicationSupport, transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    runtime.selectIdentity(41);
    TerrainGenerationBootstrap bootstrap(installer, runtime);
    REQUIRE(bootstrap.run());

    auto failures = std::make_shared<SaveManager::TestHooks>();
    failures->writeFailuresRemaining.store(3, std::memory_order_release);
    const V4WorldOpenResult interrupted = openQualifiedV4World(bootstrap, 41, {}, 0, failures);
    CHECK(interrupted.status == V4WorldOpenStatus::PersistenceFailure);
    CHECK_FALSE(std::filesystem::exists(applicationSupport / V4_WORLD_DIRECTORY));
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(applicationSupport)) {
        CHECK_FALSE(entry.path().filename().string().starts_with(std::string(V4_WORLD_DIRECTORY) +
                                                                 ".creating."));
    }

    V4WorldOpenResult retried = openQualifiedV4World(bootstrap, 41);
    REQUIRE(retried.ready());
    CHECK(retried.metadata.seed == 41);
}

TEST_CASE("Fresh v4 startup repairs only empty expected profile residue",
          "[bootstrap][save][generator-v4][recovery]") {
    TempDir directory("qualified_v4_residue");
    const std::filesystem::path applicationSupport = directory.path();
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(applicationSupport, transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    runtime.selectIdentity(99);
    TerrainGenerationBootstrap bootstrap(installer, runtime);
    REQUIRE(bootstrap.run());

    const std::filesystem::path world = applicationSupport / V4_WORLD_DIRECTORY;
    std::filesystem::create_directories(world / SaveManager::V4_REGIONS_DIRECTORY);
    std::filesystem::create_directories(world / SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY);
    std::filesystem::create_directories(world / SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY);
    {
        std::ofstream temporary(world / "metadata.json.tmp");
        temporary << "interrupted";
    }

    V4WorldOpenResult recovered = openQualifiedV4World(bootstrap, 99);
    REQUIRE(recovered.ready());
    CHECK(recovered.metadata.seed == 99);

    recovered.saveManager.reset();
    std::filesystem::remove(world / "metadata.json");
    {
        std::ofstream unknown(world / "do-not-delete.txt");
        unknown << "user data";
    }
    const V4WorldOpenResult rejected = openQualifiedV4World(bootstrap, 99);
    CHECK(rejected.status == V4WorldOpenStatus::MissingMetadata);
    CHECK(std::filesystem::exists(world / "do-not-delete.txt"));
}

TEST_CASE("Generator v4 rejects a runtime without a canonical fingerprint", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_fingerprint");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    runtime.fingerprint = "not-a-fingerprint";
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    CHECK_FALSE(bootstrap.run());
    CHECK(bootstrap.snapshot().state == TerrainBootstrapState::Failed);
    REQUIRE(bootstrap.snapshot().failure);
    CHECK(bootstrap.snapshot().failure->code == TerrainBootstrapFailureCode::Qualification);
    CHECK_FALSE(bootstrap.worldPath());
    CHECK_FALSE(bootstrap.qualifiedGenerationFingerprint());
}

TEST_CASE("Generator v4 bootstrap latches failure until an explicit retry", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_retry");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    runtime.loadingFailuresRemaining = 1;
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    CHECK_FALSE(bootstrap.run());
    CHECK(bootstrap.snapshot().state == TerrainBootstrapState::Failed);
    REQUIRE(bootstrap.snapshot().failure);
    CHECK(bootstrap.snapshot().failure->code == TerrainBootstrapFailureCode::Qualification);
    CHECK_FALSE(bootstrap.worldPath());
    CHECK_FALSE(bootstrap.run());
    CHECK(runtime.loadingCalls == 1);

    REQUIRE(bootstrap.retry());
    CHECK(bootstrap.ready());
    CHECK(runtime.loadingCalls == 2);
    CHECK(transport.calls == 1);
}

TEST_CASE("Generator v4 bootstrap can explicitly repair a qualified runtime", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_repair");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    REQUIRE(bootstrap.run());
    REQUIRE(bootstrap.ready());
    CHECK(transport.calls == 1);
    CHECK(runtime.loadingCalls == 1);

    REQUIRE(bootstrap.repair());
    CHECK(bootstrap.ready());
    CHECK(transport.calls == 1);
    CHECK(runtime.compilationCalls == 2);
    CHECK(runtime.loadingCalls == 2);
    CHECK(bootstrap.qualifiedGenerationFingerprint() == runtime.fingerprint);
}

TEST_CASE("Generator v4 bootstrap retries a failed staged download", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_download_retry");
    FakeTransport transport;
    transport.failuresRemaining = 1;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    CHECK_FALSE(bootstrap.run());
    REQUIRE(bootstrap.snapshot().failure);
    CHECK(bootstrap.snapshot().failure->code == TerrainBootstrapFailureCode::Download);
    CHECK(transport.calls == 1);
    CHECK_FALSE(std::filesystem::exists(installer.installedPackPath()));
    CHECK(hasStagingDirectory(installer));

    REQUIRE(bootstrap.retry());
    CHECK(bootstrap.ready());
    CHECK(transport.calls == 2);
    CHECK(std::filesystem::exists(installer.installedPackPath()));
    CHECK_FALSE(hasStagingDirectory(installer));
}

TEST_CASE("Interrupted base model downloads continue from persistent staged bytes",
          "[bootstrap][download][resume][regression]") {
    TempDir directory("terrain_bootstrap_resume");
    FakeTransport transport;
    transport.payload = "abcdefgh";
    transport.failAfterAdditionalBytes = 3;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {resumableBaseAsset()});
    TerrainBootstrapCancellation cancellation;

    const TerrainInstallResult interrupted = installer.prepare(cancellation, {});
    CHECK_FALSE(interrupted.installed());
    REQUIRE(transport.startingOffsets == std::vector<uint64_t>{0});
    CHECK(transport.transferredBytes == 3);
    REQUIRE(std::filesystem::is_regular_file(installer.stagingPackPath() / "base_model.onnx"));
    CHECK(std::filesystem::file_size(installer.stagingPackPath() / "base_model.onnx") == 3);

    const TerrainInstallResult resumed = installer.prepare(cancellation, {});
    REQUIRE(resumed.installed());
    CHECK(transport.startingOffsets == std::vector<uint64_t>{0, 3});
    CHECK(transport.transferredBytes == resumableBaseAsset().byteSize);
    CHECK(verifier.verify(resumed.installedPath / "base_model.onnx", resumableBaseAsset()).valid);
    CHECK_FALSE(std::filesystem::exists(installer.stagingPackPath()));
}

TEST_CASE("Terrain model repair downloads only invalid assets and preserves compiled caches",
          "[bootstrap][repair][reuse][regression]") {
    TempDir directory("terrain_bootstrap_selective_repair");
    FakeTransport transport;
    transport.payloads = {{"tiny-model.bin", "abc"}, {"decoder_model.onnx", "def"}};
    Sha256TerrainAssetVerifier verifier;
    const std::vector assets{tinyAsset(), secondaryTinyAsset()};
    TerrainModelInstaller installer(directory.path(), transport, verifier, assets);
    TerrainBootstrapCancellation cancellation;

    REQUIRE(installer.prepare(cancellation, {}).installed());
    REQUIRE(transport.calls == 2);
    const std::filesystem::path cacheSentinel = installer.installedPackPath() /
                                                worldgen::runtime::CORE_ML_CACHE_DIRECTORY /
                                                "base" / "compiled.bin";
    std::filesystem::create_directories(cacheSentinel.parent_path());
    {
        std::ofstream cache(cacheSentinel, std::ios::binary);
        cache << "compiled cache";
    }
    {
        std::ofstream corrupt(installer.installedPackPath() / "decoder_model.onnx",
                              std::ios::binary | std::ios::trunc);
        corrupt << "bad";
    }

    installer.requestRepair();
    const TerrainInstallResult repaired = installer.prepare(cancellation, {});
    REQUIRE(repaired.installed());
    CHECK(transport.calls == 3);
    CHECK(transport.requestedAssets ==
          std::vector<std::string>{"tiny-model.bin", "decoder_model.onnx", "decoder_model.onnx"});
    CHECK(verifier.verify(repaired.installedPath / "tiny-model.bin", tinyAsset()).valid);
    CHECK(
        verifier.verify(repaired.installedPath / "decoder_model.onnx", secondaryTinyAsset()).valid);
    CHECK(std::filesystem::is_regular_file(cacheSentinel));
    std::ifstream cache(cacheSentinel, std::ios::binary);
    CHECK(std::string((std::istreambuf_iterator<char>(cache)), std::istreambuf_iterator<char>()) ==
          "compiled cache");
}

TEST_CASE("Generator v4 bootstrap cancels an active staged download", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_cancel");
    BlockingTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    auto running = std::async(std::launch::async, [&] { return bootstrap.run(); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!transport.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    REQUIRE(transport.entered.load(std::memory_order_acquire));
    bootstrap.cancel();
    REQUIRE(running.wait_for(2s) == std::future_status::ready);
    CHECK_FALSE(running.get());
    CHECK(bootstrap.snapshot().state == TerrainBootstrapState::Failed);
    REQUIRE(bootstrap.snapshot().failure);
    CHECK(bootstrap.snapshot().failure->code == TerrainBootstrapFailureCode::Canceled);
    CHECK_FALSE(std::filesystem::exists(installer.installedPackPath()));
    CHECK(hasStagingDirectory(installer));
}

TEST_CASE("Unsupported platforms fail before download or world creation", "[bootstrap]") {
    TempDir directory("terrain_bootstrap_platform");
    FakeTransport transport;
    Sha256TerrainAssetVerifier verifier;
    TerrainModelInstaller installer(directory.path(), transport, verifier, {tinyAsset()});
    FakeRuntime runtime;
    runtime.platformSupported = false;
    TerrainGenerationBootstrap bootstrap(installer, runtime);

    CHECK_FALSE(bootstrap.run());
    CHECK(bootstrap.snapshot().state == TerrainBootstrapState::Failed);
    REQUIRE(bootstrap.snapshot().failure);
    CHECK(bootstrap.snapshot().failure->code == TerrainBootstrapFailureCode::UnsupportedPlatform);
    CHECK(transport.calls == 0);
    CHECK_FALSE(bootstrap.worldPath());
    CHECK_FALSE(std::filesystem::exists(installer.installedPackPath()));
}
