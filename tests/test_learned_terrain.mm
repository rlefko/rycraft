#include <catch2/catch_all.hpp>

#include "render/far_terrain.hpp"
#include "test_helpers.hpp"
#include "world/chunk_generator.hpp"
#include "world/features.hpp"
#include "world/learned_terrain.hpp"
#include "world/macro_generation.hpp"
#include "world/surface_material.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono_literals;
using namespace worldgen::learned;

GenerationIdentity testIdentity(uint64_t seed = 0x1234'5678'9ABC'DEF0ULL) {
    GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash =
        *parseSha256("543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0");
    identity.runtimeHash =
        *parseSha256("e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6");
    return identity;
}

TerrainPageKey finalPage(int64_t row, int64_t column) {
    return TerrainPageKey{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = row, .column = column},
    };
}

TerrainPageKey previewPage(int64_t row, int64_t column) {
    return TerrainPageKey{
        .quality = AuthorityQuality::PREVIEW,
        .coordinate = {.row = row, .column = column},
    };
}

void waitForBackendCall(const DeterministicFakeTerrainBackend& backend) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (backend.callCount() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    REQUIRE(backend.callCount() != 0);
}

template <typename Query>
auto awaitAuthority(Query&& query, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    auto result = query();
    while (result.status() == AuthorityStatus::DEFERRED &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
        result = query();
    }
    return result;
}

auto awaitPage(CachedTerrainAuthority& authority, TerrainPageKey key,
               AuthorityRequestPriority priority = AuthorityRequestPriority::EXPLORATION_EXACT) {
    return awaitAuthority([&] { return authority.preparePage(key, priority); });
}

std::vector<uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    REQUIRE(input.is_open());
    const auto size = input.tellg();
    REQUIRE(size > 0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    REQUIRE(input.good());
    return bytes;
}

void writeBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.is_open());
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    REQUIRE(output.good());
}

uint32_t terrainPageFixtureCrc32(std::span<const uint8_t> bytes) {
    uint32_t checksum = 0xFFFFFFFFU;
    for (const uint8_t byte : bytes) {
        checksum ^= byte;
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t polynomialMask = 0U - (checksum & 1U);
            checksum = (checksum >> 1U) ^ (0xEDB88320U & polynomialMask);
        }
    }
    return ~checksum;
}

void rewriteTerrainPageFingerprint(const std::filesystem::path& path,
                                   const Sha256Digest& fingerprint) {
    constexpr size_t FINGERPRINT_OFFSET = 36;
    constexpr size_t HEADER_CHECKSUM_OFFSET = 88;
    constexpr size_t HEADER_BYTES = 92;
    std::vector<uint8_t> bytes = readBytes(path);
    REQUIRE(bytes.size() >= HEADER_BYTES);
    std::copy(fingerprint.begin(), fingerprint.end(), bytes.begin() + FINGERPRINT_OFFSET);
    const uint32_t checksum =
        terrainPageFixtureCrc32(std::span<const uint8_t>(bytes).first(HEADER_CHECKSUM_OFFSET));
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes[HEADER_CHECKSUM_OFFSET + shift / 8] = static_cast<uint8_t>(checksum >> shift);
    }
    writeBytes(path, bytes);
}

class FailingTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity&,
                                                    TerrainPageKey) override {
        ++calls;
        return AuthorityResult<TerrainAuthorityPage>::failed({
            .code = GenerationFailureCode::INFERENCE_FAILED,
            .message = "Synthetic inference failure",
            .retriable = true,
        });
    }

    uint64_t calls = 0;
};

class ConstantTerrainBackend final : public TerrainInferenceBackend {
public:
    explicit ConstantTerrainBackend(int16_t elevationMeters = 750,
                                    int16_t lapseRateMicrodegreesPerMeter = 0)
        : elevationMeters_(elevationMeters),
          lapseRateMicrodegreesPerMeter_(lapseRateMicrodegreesPerMeter) {}

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = elevationMeters_,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = 800,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = lapseRateMicrodegreesPerMeter_,
                            });
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

private:
    int16_t elevationMeters_ = 750;
    int16_t lapseRateMicrodegreesPerMeter_ = 0;
};

class RegionTaggedTransientTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        ++pageCalls_;
        TerrainAuthorityPage page{
            .key = key,
            .generationSeed = identity.seed,
            .generationFingerprint = identity.fingerprint(),
            .samples = std::vector<QuantizedTerrainSample>(AUTHORITY_PAGE_SAMPLE_COUNT),
        };
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    AuthorityResult<PhysicalTerrainGrid> inferFinalNativeGrid(const GenerationIdentity& identity,
                                                              NativeRect region) override {
        if (!identity.valid() || !region.valid()) {
            return AuthorityResult<PhysicalTerrainGrid>::failed({
                .code = GenerationFailureCode::INVALID_REQUEST,
                .message = "Region-tagged transient request is invalid",
                .retriable = false,
            });
        }
        ++gridCalls_;
        const int16_t marker = region.columnBegin == -20 ? 222 : 111;
        PhysicalTerrainGrid grid{
            .region = region,
            .samples = std::vector<PhysicalTerrainSample>(
                static_cast<size_t>(region.height() * region.width()),
                dequantizeTerrainSample({
                    .elevationMeters = marker,
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 800,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                })),
        };
        return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(grid));
    }

    [[nodiscard]] uint64_t pageCalls() const noexcept { return pageCalls_.load(); }
    [[nodiscard]] uint64_t gridCalls() const noexcept { return gridCalls_.load(); }

private:
    std::atomic<uint64_t> pageCalls_{0};
    std::atomic<uint64_t> gridCalls_{0};
};

class GatedTerrainBackend final : public TerrainInferenceBackend {
public:
    GatedTerrainBackend()
        : enteredFuture_(entered_.get_future().share()),
          releaseFuture_(release_.get_future().share()) {}

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        entered_.set_value();
        releaseFuture_.wait();
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = 750,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = 800,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = -6'500,
                            });
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    [[nodiscard]] bool waitUntilEntered(std::chrono::milliseconds timeout) const {
        return enteredFuture_.wait_for(timeout) == std::future_status::ready;
    }

    void release() { release_.set_value(); }

private:
    std::promise<void> entered_;
    std::shared_future<void> enteredFuture_;
    std::promise<void> release_;
    std::shared_future<void> releaseFuture_;
};

class QueueReservationBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        waitForRelease();
        return delegate_.inferPage(identity, key);
    }

    AuthorityResult<PhysicalTerrainGrid> inferFinalNativeGrid(const GenerationIdentity& identity,
                                                              NativeRect region) override {
        waitForRelease();
        return delegate_.inferFinalNativeGrid(identity, region);
    }

    AuthorityResult<CoarseSpawnGrid> inferCoarseSpawnGrid(const GenerationIdentity& identity,
                                                          CoarseSpawnRegion region) override {
        waitForRelease();
        return delegate_.inferCoarseSpawnGrid(identity, region);
    }

    [[nodiscard]] bool waitUntilEntered(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return enteredCondition_.wait_for(lock, timeout, [this] { return entered_; });
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        releaseCondition_.notify_all();
    }

private:
    void waitForRelease() {
        std::unique_lock lock(mutex_);
        entered_ = true;
        enteredCondition_.notify_all();
        releaseCondition_.wait(lock, [this] { return released_; });
    }

    std::mutex mutex_;
    std::condition_variable enteredCondition_;
    std::condition_variable releaseCondition_;
    bool entered_ = false;
    bool released_ = false;
    DeterministicFakeTerrainBackend delegate_;
};

class HandoffEpochTrackingBackend final : public TerrainInferenceBackend {
public:
    enum class Kind : uint8_t {
        Page,
        PageBatch,
        TransientGrid,
    };

    struct Call {
        Kind kind = Kind::Page;
        std::vector<TerrainPageKey> pages;
        NativeRect region;
    };

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        recordAndBlockFirst({.kind = Kind::Page, .pages = {key}});
        return AuthorityResult<TerrainAuthorityPage>::ready(makePage(identity, key));
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPages(const GenerationIdentity& identity, std::span<const TerrainPageKey> keys) override {
        recordAndBlockFirst({.kind = Kind::PageBatch,
                             .pages = std::vector<TerrainPageKey>(keys.begin(), keys.end())});
        std::vector<TerrainAuthorityPage> pages;
        pages.reserve(keys.size());
        for (const TerrainPageKey key : keys)
            pages.push_back(makePage(identity, key));
        return AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(pages));
    }

    AuthorityResult<PhysicalTerrainGrid> inferFinalNativeGrid(const GenerationIdentity&,
                                                              NativeRect region) override {
        recordAndBlockFirst({.kind = Kind::TransientGrid, .region = region});
        PhysicalTerrainGrid grid{
            .region = region,
            .samples = std::vector<PhysicalTerrainSample>(
                static_cast<size_t>(region.height() * region.width()),
                dequantizeTerrainSample({
                    .elevationMeters = 800,
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 900,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                })),
        };
        return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(grid));
    }

    [[nodiscard]] bool waitForCallCount(size_t count, std::chrono::milliseconds timeout) const {
        std::unique_lock lock(mutex_);
        return callsChanged_.wait_for(lock, timeout, [&] { return calls_.size() >= count; });
    }

    [[nodiscard]] std::vector<Call> calls() const {
        std::lock_guard lock(mutex_);
        return calls_;
    }

    void releaseFirst() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        releaseChanged_.notify_all();
    }

private:
    static TerrainAuthorityPage makePage(const GenerationIdentity& identity, TerrainPageKey key) {
        TerrainAuthorityPage page{
            .key = key,
            .generationSeed = identity.seed,
            .generationFingerprint = identity.fingerprint(),
            .samples = std::vector<QuantizedTerrainSample>(AUTHORITY_PAGE_SAMPLE_COUNT),
        };
        return page;
    }

    void recordAndBlockFirst(Call call) {
        std::unique_lock lock(mutex_);
        calls_.push_back(std::move(call));
        const bool first = calls_.size() == 1;
        callsChanged_.notify_all();
        if (first)
            releaseChanged_.wait(lock, [this] { return released_; });
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable callsChanged_;
    std::condition_variable releaseChanged_;
    std::vector<Call> calls_;
    bool released_ = false;
};

class PlanarTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeRow = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE + row;
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                const int64_t elevation = std::clamp<int64_t>(
                    2'000 - nativeColumn / 16 + nativeRow / 24, -30'000, 30'000);
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = static_cast<int16_t>(elevation),
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 800,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }
};

class DrySlopeTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                const int64_t elevation =
                    std::clamp<int64_t>(8'000 + nativeColumn * 8, -30'000, 30'000);
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = static_cast<int16_t>(elevation),
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 0,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }
};

class SubBlockInclineTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeRow = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE + row;
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                const int64_t elevation =
                    std::clamp<int64_t>(1'005 - nativeColumn - nativeRow, -30'000, 30'000);
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = static_cast<int16_t>(elevation),
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = 0,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }
};

class RecordingQualityTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        {
            std::lock_guard lock(mutex_);
            keys_.push_back(key);
        }
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = 750,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = 800,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = -6'500,
                            });
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    [[nodiscard]] std::vector<TerrainPageKey> keys() const {
        std::lock_guard lock(mutex_);
        return keys_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<TerrainPageKey> keys_;
};

class ClosureRecordingDeferredTerrainAuthority final : public TerrainAuthority {
public:
    explicit ClosureRecordingDeferredTerrainAuthority(GenerationIdentity identity)
        : identity_(std::move(identity)) {}

    [[nodiscard]] const GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    preparePage(TerrainPageKey key, AuthorityRequestPriority priority) override {
        prepared_.push_back({key, priority});
        return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::deferred(
            deferredFailure());
    }

    AuthorityResult<bool> preparePages(std::span<const TerrainPageKey> keys,
                                       AuthorityRequestPriority priority) override {
        std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>> batch;
        batch.reserve(keys.size());
        for (const TerrainPageKey key : keys) {
            const std::pair request{key, priority};
            prepared_.push_back(request);
            batch.push_back(request);
        }
        batches_.push_back(std::move(batch));
        return AuthorityResult<bool>::deferred(deferredFailure());
    }

    AuthorityResult<PhysicalTerrainGrid> queryNative(NativeRect, AuthorityQuality,
                                                     AuthorityRequestPriority) override {
        return AuthorityResult<PhysicalTerrainGrid>::deferred(deferredFailure());
    }

    AuthorityResult<std::vector<PhysicalTerrainSample>>
    queryNativePoints(std::span<const NativePoint>, AuthorityQuality,
                      AuthorityRequestPriority priority) override {
        if (rasterQueryCount_++ == 0) {
            preparedAtFirstRasterQuery_ = prepared_;
            rasterPriority_ = priority;
        }
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::deferred(deferredFailure());
    }

    [[nodiscard]] TerrainAuthorityCacheMetrics cacheMetrics() const override { return {}; }

    [[nodiscard]] const std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>>&
    preparedAtFirstRasterQuery() const {
        return preparedAtFirstRasterQuery_;
    }

    [[nodiscard]] size_t rasterQueryCount() const noexcept { return rasterQueryCount_; }
    [[nodiscard]] AuthorityRequestPriority rasterPriority() const noexcept {
        return rasterPriority_;
    }
    [[nodiscard]] const std::vector<
        std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>>>&
    batches() const noexcept {
        return batches_;
    }

private:
    static GenerationFailure deferredFailure() {
        return {
            .code = GenerationFailureCode::PAGE_NOT_FOUND,
            .message = "Synthetic deferred terrain authority page",
            .retriable = true,
        };
    }

    GenerationIdentity identity_;
    std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>> prepared_;
    std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>> preparedAtFirstRasterQuery_;
    std::vector<std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>>> batches_;
    size_t rasterQueryCount_ = 0;
    AuthorityRequestPriority rasterPriority_ = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
};

enum class RetainedPageFault : uint8_t {
    NONE,
    NULL_PAGE,
    INVALID_PAYLOAD,
    WRONG_QUALITY,
    WRONG_COORDINATE,
    WRONG_SEED,
    WRONG_FINGERPRINT,
};

class EphemeralRetainedPageAuthority final : public TerrainAuthority {
public:
    explicit EphemeralRetainedPageAuthority(GenerationIdentity identity,
                                            RetainedPageFault fault = RetainedPageFault::NONE)
        : identity_(std::move(identity)), fault_(fault) {}

    [[nodiscard]] const GenerationIdentity& generationIdentity() const noexcept override {
        return identity_;
    }

    AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
    preparePage(TerrainPageKey key, AuthorityRequestPriority) override {
        auto page = std::make_shared<TerrainAuthorityPage>();
        page->key = key;
        page->generationSeed = identity_.seed;
        page->generationFingerprint = identity_.fingerprint();
        const std::optional<NativeRect> region = terrainPageNativeRect(key.coordinate);
        if (!region) {
            return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::failed({
                .code = GenerationFailureCode::INVALID_REQUEST,
                .message = "Synthetic retained page coordinate is out of range",
                .retriable = false,
            });
        }
        page->samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (size_t row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (size_t column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                page->samples[row * AUTHORITY_PAGE_NATIVE_EDGE + column] = sampleAt({
                    .row = region->rowBegin + static_cast<int64_t>(row),
                    .column = region->columnBegin + static_cast<int64_t>(column),
                });
            }
        }

        switch (fault_) {
            case RetainedPageFault::NONE:
            case RetainedPageFault::NULL_PAGE:
                break;
            case RetainedPageFault::INVALID_PAYLOAD:
                page->samples.pop_back();
                break;
            case RetainedPageFault::WRONG_QUALITY:
                page->key.quality = key.quality == AuthorityQuality::FINAL
                                        ? AuthorityQuality::PREVIEW
                                        : AuthorityQuality::FINAL;
                break;
            case RetainedPageFault::WRONG_COORDINATE:
                ++page->key.coordinate.column;
                break;
            case RetainedPageFault::WRONG_SEED:
                page->generationSeed ^= 1U;
                break;
            case RetainedPageFault::WRONG_FINGERPRINT:
                page->generationFingerprint.front() ^= 0xFFU;
                break;
        }

        std::shared_ptr<const TerrainAuthorityPage> published = std::move(page);
        observedPages_.push_back(published);
        if (fault_ == RetainedPageFault::NULL_PAGE)
            published.reset();
        return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::ready(
            std::move(published));
    }

    AuthorityResult<bool> preparePages(std::span<const TerrainPageKey>,
                                       AuthorityRequestPriority) override {
        return AuthorityResult<bool>::ready(true);
    }

    AuthorityResult<PhysicalTerrainGrid> queryNative(NativeRect, AuthorityQuality,
                                                     AuthorityRequestPriority) override {
        return AuthorityResult<PhysicalTerrainGrid>::failed({
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "Synthetic retained page authority does not provide raster queries",
            .retriable = false,
        });
    }

    AuthorityResult<std::vector<PhysicalTerrainSample>>
    queryNativePoints(std::span<const NativePoint> points, AuthorityQuality,
                      AuthorityRequestPriority) override {
        std::vector<PhysicalTerrainSample> result;
        result.reserve(points.size());
        for (const NativePoint point : points)
            result.push_back(dequantizeTerrainSample(sampleAt(point)));
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::ready(std::move(result));
    }

    [[nodiscard]] TerrainAuthorityCacheMetrics cacheMetrics() const override { return {}; }

    [[nodiscard]] const std::vector<std::weak_ptr<const TerrainAuthorityPage>>&
    observedPages() const noexcept {
        return observedPages_;
    }

private:
    static QuantizedTerrainSample sampleAt(NativePoint point) noexcept {
        uint64_t mixed = static_cast<uint64_t>(point.row) * 0x9E37'79B9'7F4A'7C15ULL;
        mixed ^= static_cast<uint64_t>(point.column) * 0xD1B5'4A32'D192'ED03ULL;
        mixed ^= mixed >> 29U;
        mixed *= 0x94D0'49BB'1331'11EBULL;
        mixed ^= mixed >> 31U;
        return {
            .elevationMeters = static_cast<int16_t>(static_cast<int64_t>(mixed % 4'001U) - 2'000),
            .meanTemperatureCentidegrees =
                static_cast<int16_t>(static_cast<int64_t>((mixed >> 7U) % 6'001U) - 2'000),
            .temperatureVariabilityCentidegrees =
                static_cast<uint16_t>(100U + (mixed >> 17U) % 2'901U),
            .annualPrecipitationMillimeters = static_cast<uint16_t>(100U + (mixed >> 27U) % 5'001U),
            .precipitationCoefficientBasisPoints =
                static_cast<uint16_t>(500U + (mixed >> 37U) % 8'501U),
            .lapseRateMicrodegreesPerMeter =
                static_cast<int16_t>(-4'000 - static_cast<int16_t>((mixed >> 47U) % 4'001U)),
        };
    }

    GenerationIdentity identity_;
    RetainedPageFault fault_ = RetainedPageFault::NONE;
    std::vector<std::weak_ptr<const TerrainAuthorityPage>> observedPages_;
};

class QualitySplitTerrainBackend final : public TerrainInferenceBackend {
public:
    QualitySplitTerrainBackend(int16_t previewElevationMeters, int16_t finalElevationMeters,
                               bool varyPreviewElevation = false,
                               uint16_t annualPrecipitationMillimeters = 3'200)
        : previewElevationMeters_(previewElevationMeters),
          finalElevationMeters_(finalElevationMeters), varyPreviewElevation_(varyPreviewElevation),
          annualPrecipitationMillimeters_(annualPrecipitationMillimeters) {}

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        {
            std::lock_guard lock(mutex_);
            keys_.push_back(key);
        }
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeRow = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE + row;
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                const int64_t previewVariation =
                    varyPreviewElevation_
                        ? static_cast<int64_t>(std::lround(
                              std::sin(static_cast<double>(nativeColumn) / 45.0) * 120.0 +
                              std::cos(static_cast<double>(nativeRow) / 52.0) * 110.0 +
                              std::sin(static_cast<double>(nativeColumn + nativeRow) / 31.0) *
                                  60.0))
                        : 0;
                const int16_t elevationMeters =
                    key.quality == AuthorityQuality::PREVIEW
                        ? static_cast<int16_t>(std::clamp<int64_t>(
                              static_cast<int64_t>(previewElevationMeters_) - previewVariation,
                              std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()))
                        : static_cast<int16_t>(std::clamp<int64_t>(
                              static_cast<int64_t>(finalElevationMeters_) -
                                  world_coord::floorDiv(nativeColumn + nativeRow, int64_t{16}),
                              std::numeric_limits<int16_t>::min(),
                              std::numeric_limits<int16_t>::max()));
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = elevationMeters,
                    .meanTemperatureCentidegrees = 1'800,
                    .temperatureVariabilityCentidegrees = 600,
                    .annualPrecipitationMillimeters = annualPrecipitationMillimeters_,
                    .precipitationCoefficientBasisPoints = 2'500,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    [[nodiscard]] std::vector<TerrainPageKey> keys() const {
        std::lock_guard lock(mutex_);
        return keys_;
    }

private:
    int16_t previewElevationMeters_ = 0;
    int16_t finalElevationMeters_ = 0;
    bool varyPreviewElevation_ = false;
    uint16_t annualPrecipitationMillimeters_ = 3'200;
    mutable std::mutex mutex_;
    std::vector<TerrainPageKey> keys_;
};

template <typename Query>
auto awaitGeneration(Query&& query,
                     std::chrono::milliseconds timeout = 5s) -> std::invoke_result_t<Query> {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        try {
            return query();
        } catch (const GenerationFailureException& failure) {
            if (failure.status() != AuthorityStatus::DEFERRED)
                throw;
            std::this_thread::sleep_for(1ms);
        }
    }
    throw std::runtime_error("Learned generation did not become ready before the test deadline");
}

class SerialTrackingTerrainBackend final : public TerrainInferenceBackend {
public:
    explicit SerialTrackingTerrainBackend(std::chrono::milliseconds latency) : delegate_(latency) {}

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        {
            std::lock_guard lock(mutex_);
            ++activeCalls_;
            maximumActiveCalls_ = std::max(maximumActiveCalls_, activeCalls_);
            requestedKeys_.push_back(key);
            requestThreads_.push_back(std::this_thread::get_id());
        }
        auto result = delegate_.inferPage(identity, key);
        {
            std::lock_guard lock(mutex_);
            --activeCalls_;
        }
        return result;
    }

    [[nodiscard]] size_t maximumActiveCalls() const {
        std::lock_guard lock(mutex_);
        return maximumActiveCalls_;
    }

    [[nodiscard]] std::vector<TerrainPageKey> requestedKeys() const {
        std::lock_guard lock(mutex_);
        return requestedKeys_;
    }

    [[nodiscard]] std::vector<std::thread::id> requestThreads() const {
        std::lock_guard lock(mutex_);
        return requestThreads_;
    }

private:
    DeterministicFakeTerrainBackend delegate_;
    mutable std::mutex mutex_;
    size_t activeCalls_ = 0;
    size_t maximumActiveCalls_ = 0;
    std::vector<TerrainPageKey> requestedKeys_;
    std::vector<std::thread::id> requestThreads_;
};

class BatchTrackingTerrainBackend final : public TerrainInferenceBackend {
public:
    explicit BatchTrackingTerrainBackend(std::chrono::milliseconds firstPageLatency)
        : firstPageLatency_(firstPageLatency) {}

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        enter();
        bool first = false;
        {
            std::lock_guard lock(mutex_);
            singletonKeys_.push_back(key);
            first = singletonKeys_.size() == 1;
        }
        if (first && firstPageLatency_.count() > 0)
            std::this_thread::sleep_for(firstPageLatency_);
        TerrainAuthorityPage page = makePage(identity, key);
        leave();
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPages(const GenerationIdentity& identity, std::span<const TerrainPageKey> keys) override {
        enter();
        {
            std::lock_guard lock(mutex_);
            batches_.emplace_back(keys.begin(), keys.end());
        }
        std::vector<TerrainAuthorityPage> pages;
        pages.reserve(keys.size());
        for (const TerrainPageKey key : keys)
            pages.push_back(makePage(identity, key));
        leave();
        return AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(pages));
    }

    [[nodiscard]] size_t maximumActiveCalls() const {
        std::lock_guard lock(mutex_);
        return maximumActiveCalls_;
    }

    [[nodiscard]] std::vector<std::vector<TerrainPageKey>> batches() const {
        std::lock_guard lock(mutex_);
        return batches_;
    }

    [[nodiscard]] size_t singletonCount() const {
        std::lock_guard lock(mutex_);
        return singletonKeys_.size();
    }

private:
    static TerrainAuthorityPage makePage(const GenerationIdentity& identity, TerrainPageKey key) {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        const int16_t elevation = static_cast<int16_t>(std::clamp<int64_t>(
            key.coordinate.row * 31 + key.coordinate.column * 17, -30'000, 30'000));
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = elevation,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = 800,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = -6'500,
                            });
        return page;
    }

    void enter() {
        std::lock_guard lock(mutex_);
        ++activeCalls_;
        maximumActiveCalls_ = std::max(maximumActiveCalls_, activeCalls_);
    }

    void leave() {
        std::lock_guard lock(mutex_);
        if (activeCalls_ != 0)
            --activeCalls_;
    }

    std::chrono::milliseconds firstPageLatency_;
    mutable std::mutex mutex_;
    size_t activeCalls_ = 0;
    size_t maximumActiveCalls_ = 0;
    std::vector<TerrainPageKey> singletonKeys_;
    std::vector<std::vector<TerrainPageKey>> batches_;
};

class PriorityReconsiderationTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        {
            std::unique_lock lock(mutex_);
            calls_.push_back({key});
            if (!initialPageEntered_) {
                initialPageEntered_ = true;
                condition_.notify_all();
                condition_.wait(lock, [this] { return initialPageReleased_; });
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(makePage(identity, key));
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPages(const GenerationIdentity& identity, std::span<const TerrainPageKey> keys) override {
        {
            std::unique_lock lock(mutex_);
            calls_.emplace_back(keys.begin(), keys.end());
            if (!firstBatchEntered_) {
                firstBatchEntered_ = true;
                condition_.notify_all();
                condition_.wait(lock, [this] { return firstBatchReleased_; });
            }
        }
        std::vector<TerrainAuthorityPage> pages;
        pages.reserve(keys.size());
        for (const TerrainPageKey key : keys)
            pages.push_back(makePage(identity, key));
        return AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(pages));
    }

    [[nodiscard]] bool waitForInitialPage(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [this] { return initialPageEntered_; });
    }

    [[nodiscard]] bool waitForFirstBatch(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [this] { return firstBatchEntered_; });
    }

    void releaseInitialPage() {
        {
            std::lock_guard lock(mutex_);
            initialPageReleased_ = true;
        }
        condition_.notify_all();
    }

    void releaseFirstBatch() {
        {
            std::lock_guard lock(mutex_);
            firstBatchReleased_ = true;
        }
        condition_.notify_all();
    }

    void releaseAll() {
        {
            std::lock_guard lock(mutex_);
            initialPageReleased_ = true;
            firstBatchReleased_ = true;
        }
        condition_.notify_all();
    }

    [[nodiscard]] std::vector<std::vector<TerrainPageKey>> calls() const {
        std::lock_guard lock(mutex_);
        return calls_;
    }

private:
    static TerrainAuthorityPage makePage(const GenerationIdentity& identity, TerrainPageKey key) {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.assign(AUTHORITY_PAGE_SAMPLE_COUNT,
                            QuantizedTerrainSample{
                                .elevationMeters = 750,
                                .meanTemperatureCentidegrees = 1'800,
                                .temperatureVariabilityCentidegrees = 600,
                                .annualPrecipitationMillimeters = 800,
                                .precipitationCoefficientBasisPoints = 2'500,
                                .lapseRateMicrodegreesPerMeter = -6'500,
                            });
        return page;
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool initialPageEntered_ = false;
    bool initialPageReleased_ = false;
    bool firstBatchEntered_ = false;
    bool firstBatchReleased_ = false;
    std::vector<std::vector<TerrainPageKey>> calls_;
};

// This deliberately simple physical fixture gives the water regressions a
// large, humid catchment that converges on one steep coastal outlet. It keeps
// the tests about canonical high-flow falls rather than the incidental
// low-flow channels in the general-purpose noise backend.
class HighFlowWaterfallTerrainBackend final : public TerrainInferenceBackend {
public:
    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        ++calls_;
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
        for (int row = 0; row < AUTHORITY_PAGE_NATIVE_EDGE; ++row) {
            for (int column = 0; column < AUTHORITY_PAGE_NATIVE_EDGE; ++column) {
                const int64_t nativeRow = key.coordinate.row * AUTHORITY_PAGE_NATIVE_EDGE + row;
                const int64_t nativeColumn =
                    key.coordinate.column * AUTHORITY_PAGE_NATIVE_EDGE + column;
                page.samples[static_cast<size_t>(row * AUTHORITY_PAGE_NATIVE_EDGE + column)] = {
                    .elevationMeters = static_cast<int16_t>(
                        std::llround(elevationMeters(nativeColumn, nativeRow))),
                    .meanTemperatureCentidegrees = 1'000,
                    .temperatureVariabilityCentidegrees = 200,
                    .annualPrecipitationMillimeters = 1'800,
                    .precipitationCoefficientBasisPoints = 800,
                    .lapseRateMicrodegreesPerMeter = -6'500,
                };
            }
        }
        return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
    }

    [[nodiscard]] uint64_t calls() const noexcept { return calls_.load(); }

private:
    // The test scans the native page at X=[0, 2,048) and Z=[-2,048, 0).
    // A bounded, humid lake drains through one natural spill into a steep
    // coastal outlet. The flat lake avoids inventing lateral feeder falls at
    // the lip while the outlet retains a large, physically routed catchment.
    static double elevationMeters(int64_t nativeColumn, int64_t nativeRow) {
        constexpr int64_t OUTLET_COLUMN = 256;
        constexpr int64_t LAKE_NORTH_ROW = -300;
        constexpr int64_t OUTLET_ROW = -213;
        constexpr int64_t OCEAN_ROW = OUTLET_ROW + 1;
        constexpr double LAKE_HALF_WIDTH_NATIVE_CELLS = 32.0;
        constexpr double RIM_ELEVATION_METERS = 160.0;
        constexpr double LAKE_FLOOR_ELEVATION_METERS = 70.0;
        constexpr double SPILL_ELEVATION_METERS = 120.0;
        constexpr double OCEAN_FLOOR_ELEVATION_METERS = -100.0;
        const double lateral =
            std::abs(static_cast<double>(nativeColumn) - static_cast<double>(OUTLET_COLUMN));
        if (nativeRow > LAKE_NORTH_ROW && nativeRow < OUTLET_ROW &&
            lateral < LAKE_HALF_WIDTH_NATIVE_CELLS) {
            return LAKE_FLOOR_ELEVATION_METERS;
        }
        if (nativeRow == OUTLET_ROW && lateral < 0.5)
            return SPILL_ELEVATION_METERS;
        if (nativeRow >= OCEAN_ROW && (lateral < 4.5 || nativeRow >= OCEAN_ROW + 4)) {
            return OCEAN_FLOOR_ELEVATION_METERS;
        }
        return RIM_ELEVATION_METERS;
    }

    std::atomic<uint64_t> calls_{0};
};

struct CanonicalWaterRegressionFixture {
    static constexpr uint64_t SEED = 42;

    TempDir directory{"learned_canonical_water_regression"};
    GenerationIdentity identity = testIdentity(SEED);
    std::shared_ptr<CachedTerrainAuthority> authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<HighFlowWaterfallTerrainBackend>());
    std::shared_ptr<WorldGenerationContext> context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator{SEED, context};
    worldgen::MacroGenerationSampler macro{SEED, context};
};

CanonicalWaterRegressionFixture& canonicalWaterRegressionFixture() {
    static CanonicalWaterRegressionFixture fixture;
    return fixture;
}

struct CanonicalWaterfallScene {
    int64_t highX = 0;
    int64_t highZ = 0;
    int64_t lowX = 0;
    int64_t lowZ = 0;
    double discharge = 0.0;
    double gradient = 0.0;
    worldgen::WaterBodyId body = worldgen::NO_WATER_BODY;
    uint64_t transitionOwnerId = 0;
};

const CanonicalWaterfallScene& canonicalWaterfallScene() {
    static const CanonicalWaterfallScene scene = [] {
        CanonicalWaterRegressionFixture& fixture = canonicalWaterRegressionFixture();
        constexpr int64_t ORIGIN_X = 960;
        constexpr int64_t ORIGIN_Z = -912;
        constexpr int SPACING = 1;
        constexpr int EDGE = 129;
        std::vector<worldgen::HydrologySample> samples(static_cast<size_t>(EDGE * EDGE));
        awaitGeneration(
            [&] {
                fixture.macro.sampleHydrologyGrid(ORIGIN_X, ORIGIN_Z, SPACING, SPACING, EDGE, EDGE,
                                                  samples);
                return true;
            },
            60s);

        constexpr double MINIMUM_RENDERABLE_FLOW_MULTIPLIER = 4.0;
        constexpr double MINIMUM_STEEP_FALL_GRADIENT = 1.5;
        std::optional<CanonicalWaterfallScene> best;
        for (int row = 0; row < EDGE; ++row) {
            for (int column = 0; column < EDGE; ++column) {
                const size_t index = static_cast<size_t>(row * EDGE + column);
                const worldgen::HydrologySample& high = samples[index];
                if (!high.river || !high.waterfall || high.waterBodyId == worldgen::NO_WATER_BODY ||
                    high.discharge < worldgen::NATIVE_HYDROLOGY_MINIMUM_CHANNEL_DISCHARGE *
                                         MINIMUM_RENDERABLE_FLOW_MULTIPLIER ||
                    high.channelGradient < MINIMUM_STEEP_FALL_GRADIENT) {
                    continue;
                }
                const CanonicalWaterfallScene candidate{
                    .highX = ORIGIN_X + static_cast<int64_t>(column) * SPACING,
                    .highZ = ORIGIN_Z + static_cast<int64_t>(row) * SPACING,
                    .discharge = high.discharge,
                    .gradient = high.channelGradient,
                    .body = high.waterBodyId,
                    .transitionOwnerId = high.transitionOwnerId,
                };
                if (!best || candidate.discharge > best->discharge ||
                    (candidate.discharge == best->discharge &&
                     std::pair{candidate.highX, candidate.highZ} <
                         std::pair{best->highX, best->highZ})) {
                    best = candidate;
                }
            }
        }
        if (!best)
            throw std::runtime_error("High-flow final authority contains no canonical waterfall");
        // Wide high-flow ribbons place their canonical integer anchor on the
        // outer edge of the fall curtain. Channel width is capped at 36
        // blocks, so a four-block neighborhood can exclude a valid anchor.
        constexpr int ANCHOR_RADIUS = 24;
        constexpr int ANCHOR_EDGE = ANCHOR_RADIUS * 2 + 1;
        std::array<worldgen::HydrologySample, ANCHOR_EDGE * ANCHOR_EDGE> anchorSamples{};
        awaitGeneration(
            [&] {
                fixture.macro.sampleHydrologyGrid(best->highX - ANCHOR_RADIUS,
                                                  best->highZ - ANCHOR_RADIUS, 1, 1, ANCHOR_EDGE,
                                                  ANCHOR_EDGE, anchorSamples);
                return true;
            },
            60s);
        std::optional<CanonicalWaterfallScene> anchored;
        for (int row = 0; row < ANCHOR_EDGE; ++row) {
            for (int column = 0; column < ANCHOR_EDGE; ++column) {
                const worldgen::HydrologySample& sample =
                    anchorSamples[static_cast<size_t>(row * ANCHOR_EDGE + column)];
                if (!sample.waterfall || !sample.waterfallAnchor ||
                    sample.transitionOwnerId != best->transitionOwnerId)
                    continue;
                CanonicalWaterfallScene candidate = *best;
                candidate.highX = best->highX - ANCHOR_RADIUS + column;
                candidate.highZ = best->highZ - ANCHOR_RADIUS + row;
                candidate.discharge = sample.discharge;
                candidate.gradient = sample.channelGradient;
                candidate.body = sample.waterBodyId;
                if (!anchored || candidate.discharge > anchored->discharge ||
                    (candidate.discharge == anchored->discharge &&
                     std::pair{candidate.highX, candidate.highZ} <
                         std::pair{anchored->highX, anchored->highZ})) {
                    anchored = candidate;
                }
            }
        }
        if (!anchored)
            throw std::runtime_error("Canonical waterfall contains no integer anchor");
        best = anchored;
        std::optional<std::pair<int64_t, int64_t>> nearestOcean;
        uint64_t nearestDistanceSquared = std::numeric_limits<uint64_t>::max();
        for (int row = 0; row < EDGE; ++row) {
            for (int column = 0; column < EDGE; ++column) {
                const worldgen::HydrologySample& low =
                    samples[static_cast<size_t>(row * EDGE + column)];
                if (!low.ocean || low.river || low.waterfall ||
                    low.waterBodyId == worldgen::NO_WATER_BODY ||
                    low.waterSurface - low.surfaceElevation <= 1.0) {
                    continue;
                }
                const int64_t x = ORIGIN_X + static_cast<int64_t>(column) * SPACING;
                const int64_t z = ORIGIN_Z + static_cast<int64_t>(row) * SPACING;
                const int64_t deltaX = x - best->highX;
                const int64_t deltaZ = z - best->highZ;
                const uint64_t distanceSquared =
                    static_cast<uint64_t>(deltaX * deltaX) + static_cast<uint64_t>(deltaZ * deltaZ);
                if (!nearestOcean || distanceSquared < nearestDistanceSquared ||
                    (distanceSquared == nearestDistanceSquared &&
                     std::pair{x, z} < *nearestOcean)) {
                    nearestDistanceSquared = distanceSquared;
                    nearestOcean = {x, z};
                }
            }
        }
        if (!nearestOcean)
            throw std::runtime_error("Fake final authority contains no canonical ocean");
        best->lowX = nearestOcean->first;
        best->lowZ = nearestOcean->second;
        return *best;
    }();
    return scene;
}

Sha256Digest quantizedPayloadHash(const TerrainAuthorityPage& page) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(page.samples.data());
    return sha256(std::span<const uint8_t>(bytes, page.byteSize()));
}

} // namespace

TEST_CASE("Learned terrain SHA-256 and generation fingerprints are canonical", "[learned]") {
    constexpr std::array<uint8_t, 3> ABC{'a', 'b', 'c'};
    REQUIRE(sha256Hex(sha256(ABC)) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE_FALSE(parseSha256("not-a-digest"));

    GenerationIdentity identity = testIdentity();
    REQUIRE(identity.valid());
    STATIC_REQUIRE((COARSE_WINDOW == WindowGeometry{64, 48, 20, 1}));
    STATIC_REQUIRE((LATENT_WINDOW == WindowGeometry{64, 32, 2, 4}));
    STATIC_REQUIRE((DECODER_WINDOW == WindowGeometry{256, 192, 1, 4}));
    STATIC_REQUIRE(TerrainAuthorityCacheConfig{}.byteBudget == 512ULL * 1'024 * 1'024);
    STATIC_REQUIRE(TerrainAuthorityCacheConfig{}.maximumOutstandingRequests == 64);
    STATIC_REQUIRE(TerrainAuthorityCacheConfig{}.maximumConcurrentBuilds == 1);
    STATIC_REQUIRE(GENERATOR_V4_HYDROLOGY_REVISION == 12);
    STATIC_REQUIRE(GENERATOR_V4_POSTPROCESSING_REVISION == 9);
    REQUIRE(identity.coarseWindow == COARSE_WINDOW);
    REQUIRE(identity.latentWindow == LATENT_WINDOW);
    REQUIRE(identity.decoderWindow == DECODER_WINDOW);
    REQUIRE(identity.fingerprint() == testIdentity().fingerprint());
    REQUIRE(sha256Hex(identity.fingerprint()) ==
            "6b3da3b338a11a44c2adc51d3f258f09e88561f9c672c45fca28da95c323a440");

    GenerationIdentity changed = identity;
    ++changed.seed;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    changed = identity;
    ++changed.quantizationRevision;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    changed = identity;
    changed.latentWindow.stride = 16;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    changed = identity;
    changed.provider.flags &= ~CORE_ML_STATIC_BASE_BATCH_FOUR;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    CHECK_FALSE(changed.valid());
    changed = identity;
    changed.provider.flags &= ~CORE_ML_STATIC_DECODER_BATCH_FOUR;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    CHECK_FALSE(changed.valid());
    changed = identity;
    changed.provider.flags &= ~CORE_ML_STATIC_DECODER_SPATIAL_256;
    REQUIRE(changed.fingerprint() != identity.fingerprint());
    CHECK_FALSE(changed.valid());

    changed = identity;
    changed.modelBlockScale = 8;
    CHECK_FALSE(changed.valid());
    changed = identity;
    changed.coarseWindow.stride = 32;
    CHECK_FALSE(changed.valid());
    changed = identity;
    changed.latentWindow.batchSize = 2;
    CHECK_FALSE(changed.valid());
    changed = identity;
    changed.decoderWindow.edge = 128;
    CHECK_FALSE(changed.valid());
    changed = identity;
    ++changed.provider.flags;
    CHECK_FALSE(changed.valid());
    changed = identity;
    ++changed.hydrologyRevision;
    CHECK_FALSE(changed.valid());
    changed = identity;
    ++changed.postprocessingRevision;
    CHECK_FALSE(changed.valid());
}

TEST_CASE("InfiniteDiffusion windows use a global half-open lattice", "[learned]") {
    REQUIRE(floorDivide(-65, 32) == -3);
    REQUIRE(floorDivide(-64, 32) == -2);
    REQUIRE(floorDivide(-1, 32) == -1);
    REQUIRE(floorDivide(0, 32) == 0);

    const auto crossingOrigin = intersectingWindows(
        NativeRect{.rowBegin = -1, .columnBegin = -1, .rowEnd = 1, .columnEnd = 1}, LATENT_WINDOW);
    REQUIRE(crossingOrigin.size() == 9);
    REQUIRE(crossingOrigin.front() == WindowIndex{.row = -2, .column = -2});
    REQUIRE(crossingOrigin.back() == WindowIndex{.row = 0, .column = 0});
    REQUIRE(std::is_sorted(crossingOrigin.begin(), crossingOrigin.end()));

    const auto leftOfBoundary = intersectingWindows(
        NativeRect{.rowBegin = 0, .columnBegin = 0, .rowEnd = 32, .columnEnd = 32}, LATENT_WINDOW);
    REQUIRE(leftOfBoundary == std::vector<WindowIndex>{{.row = -1, .column = -1},
                                                       {.row = -1, .column = 0},
                                                       {.row = 0, .column = -1},
                                                       {.row = 0, .column = 0}});

    const auto rightOfBoundary = intersectingWindows(
        NativeRect{.rowBegin = 32, .columnBegin = 32, .rowEnd = 33, .columnEnd = 33},
        LATENT_WINDOW);
    REQUIRE(rightOfBoundary == std::vector<WindowIndex>{{.row = 0, .column = 0},
                                                        {.row = 0, .column = 1},
                                                        {.row = 1, .column = 0},
                                                        {.row = 1, .column = 1}});
}

TEST_CASE("Learned authority pages preserve negative half-open ownership", "[learned]") {
    struct BoundaryCase {
        int64_t native = 0;
        int64_t page = 0;
        size_t local = 0;
    };
    constexpr std::array CASES{
        BoundaryCase{-513, -3, 255}, BoundaryCase{-512, -2, 0}, BoundaryCase{-257, -2, 255},
        BoundaryCase{-256, -1, 0},   BoundaryCase{-1, -1, 255}, BoundaryCase{0, 0, 0},
        BoundaryCase{255, 0, 255},   BoundaryCase{256, 1, 0},
    };

    for (const BoundaryCase boundary : CASES) {
        const TerrainPageCoordinate page =
            terrainPageCoordinateFor({.row = boundary.native, .column = boundary.native});
        CHECK(page == TerrainPageCoordinate{.row = boundary.page, .column = boundary.page});
        CHECK(terrainPageLocalCoordinate(boundary.native) == boundary.local);

        const std::optional<NativeRect> region = terrainPageNativeRect(page);
        REQUIRE(region);
        CHECK(boundary.native >= region->rowBegin);
        CHECK(boundary.native < region->rowEnd);
        CHECK(boundary.native >= region->columnBegin);
        CHECK(boundary.native < region->columnEnd);
        CHECK(region->height() == AUTHORITY_PAGE_NATIVE_EDGE);
        CHECK(region->width() == AUTHORITY_PAGE_NATIVE_EDGE);
    }

    CHECK(world_coord::floorMultiple(int64_t{-257}, int64_t{256}) == -512);
    CHECK(world_coord::floorMultiple(int64_t{-256}, int64_t{256}) == -256);
    CHECK(world_coord::floorMultiple(int64_t{-1}, int64_t{256}) == -256);
    CHECK(world_coord::floorMultiple(int64_t{0}, int64_t{256}) == 0);
    CHECK(world_coord::floorMultiple(int64_t{255}, int64_t{256}) == 0);
    CHECK(world_coord::floorMultiple(int64_t{256}, int64_t{256}) == 256);

    const TerrainPageCoordinate minimumPage =
        terrainPageCoordinateFor({.row = std::numeric_limits<int64_t>::min(),
                                  .column = std::numeric_limits<int64_t>::min()});
    const std::optional<NativeRect> minimumRegion = terrainPageNativeRect(minimumPage);
    REQUIRE(minimumRegion);
    CHECK(minimumRegion->rowBegin == std::numeric_limits<int64_t>::min());
    CHECK(minimumRegion->columnBegin == std::numeric_limits<int64_t>::min());
    CHECK_FALSE(terrainPageNativeRect({.row = std::numeric_limits<int64_t>::max(),
                                       .column = std::numeric_limits<int64_t>::max()}));
}

TEST_CASE("Weighted window accumulation is insertion-order independent", "[learned]") {
    constexpr WindowGeometry geometry{
        .edge = 3,
        .stride = 2,
        .inferenceSteps = 1,
        .batchSize = 1,
    };
    const NativeRect target{.rowBegin = 0, .columnBegin = 0, .rowEnd = 3, .columnEnd = 5};
    const std::vector<float> first(9, 10.0F);
    const std::vector<float> second(9, 20.0F);

    WeightedWindowAccumulator forward(target, geometry, 1);
    REQUIRE(forward.addWindow({.row = 0, .column = 0}, first));
    REQUIRE(forward.addWindow({.row = 0, .column = 1}, second));
    REQUIRE_FALSE(forward.addWindow({.row = 0, .column = 1}, second));
    auto forwardResult = forward.resolve();
    REQUIRE(forwardResult.isReady());

    WeightedWindowAccumulator reverse(target, geometry, 1);
    REQUIRE(reverse.addWindow({.row = 0, .column = 1}, second));
    REQUIRE(reverse.addWindow({.row = 0, .column = 0}, first));
    auto reverseResult = reverse.resolve();
    REQUIRE(reverseResult.isReady());
    REQUIRE(*forwardResult.value() == *reverseResult.value());

    for (size_t row = 0; row < 3; ++row) {
        REQUIRE((*forwardResult.value())[row * 5] == Catch::Approx(10.0F));
        REQUIRE((*forwardResult.value())[row * 5 + 2] == Catch::Approx(15.0F));
        REQUIRE((*forwardResult.value())[row * 5 + 4] == Catch::Approx(20.0F));
    }
    REQUIRE(linearWindowWeight(0, 3) == Catch::Approx(WINDOW_WEIGHT_EPSILON));
    REQUIRE(linearWindowWeight(1, 3) == Catch::Approx(1.0F));
    REQUIRE(linearWindowWeight(2, 3) == Catch::Approx(WINDOW_WEIGHT_EPSILON));
}

TEST_CASE("Portable PCG64 and Marsaglia noise match the published implementation", "[learned]") {
    PortablePcg64 generator(0x1234'5678'9ABC'DEF0ULL);
    constexpr std::array<uint32_t, 6> EXPECTED{
        0x7B75F3C1U, 0x2C1F919AU, 0x7EC843F4U, 0x71F83B2EU, 0x29019EA8U, 0x57153F69U,
    };
    for (uint32_t expected : EXPECTED)
        REQUIRE(generator.next32() == expected);
    REQUIRE(generator.state() == 0xDA55F3DAA164B576ULL);

    REQUIRE(terrainTileSeed(42, 0, 0) == 0x0AE607774D7D030AULL);
    REQUIRE(terrainTileSeed(42, -1, -1) == 0xA91D8130AF458950ULL);
    REQUIRE(terrainTileSeed(UINT64_MAX, -123, 456) == 0x3C6EF32717FA1E34ULL);

    std::array<float, 8> normal{};
    fillStandardNormal(0x1234'5678'9ABC'DEF0ULL, normal);
    constexpr std::array<float, 8> EXPECTED_NORMAL{
        -0.0701443031F, -1.2961331606F, -0.2569115162F, -2.9601044655F,
        -0.9683197141F, -0.4554439187F, -0.4294789433F, 0.0498145446F,
    };
    for (size_t index = 0; index < normal.size(); ++index)
        REQUIRE(normal[index] == Catch::Approx(EXPECTED_NORMAL[index]).margin(1.0e-7F));
}

TEST_CASE("Coordinate-addressed Gaussian patches agree across negative tile seams", "[learned]") {
    auto complete = gaussianNoisePatch(
        99, NativeRect{.rowBegin = -3, .columnBegin = -3, .rowEnd = 3, .columnEnd = 3}, 2, 4);
    REQUIRE(complete.isReady());
    constexpr size_t COMPLETE_EDGE = 6;
    for (int64_t row = -3; row < 3; ++row) {
        for (int64_t column = -3; column < 3; ++column) {
            auto point = gaussianNoisePatch(99,
                                            NativeRect{.rowBegin = row,
                                                       .columnBegin = column,
                                                       .rowEnd = row + 1,
                                                       .columnEnd = column + 1},
                                            2, 4);
            REQUIRE(point.isReady());
            const size_t pixel =
                static_cast<size_t>(row + 3) * COMPLETE_EDGE + static_cast<size_t>(column + 3);
            REQUIRE((*point.value())[0] == (*complete.value())[pixel]);
            REQUIRE((*point.value())[1] ==
                    (*complete.value())[COMPLETE_EDGE * COMPLETE_EDGE + pixel]);
        }
    }
}

TEST_CASE("Learned authority teardown joins active inference and publication workers",
          "[learned][thread][shutdown][regression]") {
    TempDir directory("learned_authority_teardown");
    const GenerationIdentity identity = testIdentity(0x7EA2'D04E'0001ULL);
    auto backend = std::make_shared<GatedTerrainBackend>();
    auto authority = std::make_unique<CachedTerrainAuthority>(identity, directory.path(), backend);
    REQUIRE(authority->preparePage(finalPage(0, 0), AuthorityRequestPriority::EXPLORATION_EXACT)
                .status() == AuthorityStatus::DEFERRED);
    const bool entered = backend->waitUntilEntered(2s);
    if (!entered)
        backend->release();
    REQUIRE(entered);

    std::future<void> teardown = std::async(std::launch::async, [&] { authority.reset(); });
    REQUIRE(teardown.wait_for(20ms) == std::future_status::timeout);
    backend->release();
    REQUIRE(teardown.wait_for(5s) == std::future_status::ready);
    REQUIRE_NOTHROW(teardown.get());
}

TEST_CASE("Coarse spawn grids are bounded, single-flight, and nonpersistent", "[learned][spawn]") {
    TempDir directory("learned_coarse_spawn_grid");
    const GenerationIdentity identity = testIdentity(0xBADC'0FFE'0001ULL);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(2ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const CoarseSpawnRegion region{
        .rowBegin = -16,
        .columnBegin = -16,
        .rowEnd = 16,
        .columnEnd = 16,
    };
    const uint64_t completionBefore = authority.cacheMetrics().completionGeneration;

    const auto first = authority.queryCoarseSpawnGrid(region, AuthorityRequestPriority::SPAWN);
    REQUIRE(first.status() == AuthorityStatus::DEFERRED);
    const auto second = authority.queryCoarseSpawnGrid(region, AuthorityRequestPriority::SPAWN);
    REQUIRE(second.status() == AuthorityStatus::DEFERRED);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore);
    const auto ready = awaitAuthority(
        [&] { return authority.queryCoarseSpawnGrid(region, AuthorityRequestPriority::SPAWN); });
    REQUIRE(ready.isReady());
    REQUIRE(ready.value()->valid());
    REQUIRE(ready.value()->region == region);
    REQUIRE(ready.value()->elevationMeters.size() == 32 * 32);
    REQUIRE(backend->callCount() == 1);
    REQUIRE(authority.cacheMetrics().singleFlightDeferrals >= 1);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore + 1);

    const auto cached = authority.queryCoarseSpawnGrid(region, AuthorityRequestPriority::SPAWN);
    REQUIRE(cached.isReady());
    CHECK(cached.value()->elevationMeters == ready.value()->elevationMeters);
    CHECK(backend->callCount() == 1);
    CHECK(authority.cacheMetrics().completionGeneration == completionBefore + 1);
    const TerrainPageStore store(directory.path(), identity);
    CHECK_FALSE(std::filesystem::exists(store.pagePath(
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = 0, .column = 0}})));

    const CoarseSpawnRegion tooLarge{
        .rowBegin = 0,
        .columnBegin = 0,
        .rowEnd = MAXIMUM_COARSE_SPAWN_GRID_EDGE + 1,
        .columnEnd = 1,
    };
    const auto rejected = authority.queryCoarseSpawnGrid(tooLarge, AuthorityRequestPriority::SPAWN);
    REQUIRE(rejected.status() == AuthorityStatus::FAILED);
    REQUIRE(rejected.failure());
    CHECK(rejected.failure()->code == GenerationFailureCode::INVALID_REQUEST);
}

TEST_CASE("Transient final rectangles advance completion once after becoming observable",
          "[learned][transient][concurrency][regression]") {
    TempDir directory("learned_transient_completion_generation");
    const GenerationIdentity identity = testIdentity(0x7A11'51E4'0001ULL);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(2ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    constexpr NativeRect REGION{
        .rowBegin = -2,
        .columnBegin = 510,
        .rowEnd = 6,
        .columnEnd = 518,
    };

    const uint64_t completionBefore = authority.cacheMetrics().completionGeneration;
    REQUIRE(
        authority.queryTransientFinalNativeGrid(REGION, AuthorityRequestPriority::PROTECTED_HANDOFF)
            .status() == AuthorityStatus::DEFERRED);
    REQUIRE(
        authority.queryTransientFinalNativeGrid(REGION, AuthorityRequestPriority::PROTECTED_HANDOFF)
            .status() == AuthorityStatus::DEFERRED);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore);

    const auto ready = awaitAuthority([&] {
        return authority.queryTransientFinalNativeGrid(REGION,
                                                       AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(ready.isReady());
    REQUIRE(*ready.value());
    REQUIRE((*ready.value())->valid());
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore + 1);

    const auto cached = authority.queryTransientFinalNativeGrid(
        REGION, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(cached.isReady());
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore + 1);
}

TEST_CASE("Protected transient final authority survives restart without another inference",
          "[learned][transient][persistence][restart][performance][regression]") {
    TempDir directory("learned_transient_restart");
    const GenerationIdentity identity = testIdentity(0x7A11'51E4'0002ULL);
    constexpr NativeRect REGION{
        .rowBegin = -258,
        .columnBegin = 510,
        .rowEnd = -250,
        .columnEnd = 518,
    };
    std::vector<PhysicalTerrainSample> expected;
    {
        auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
        CachedTerrainAuthority authority(identity, directory.path(), backend);
        const auto ready = awaitAuthority([&] {
            return authority.queryTransientFinalNativeGrid(
                REGION, AuthorityRequestPriority::PROTECTED_HANDOFF);
        });
        REQUIRE(ready.isReady());
        REQUIRE(*ready.value());
        expected = (*ready.value())->samples;
        CHECK(backend->callCount() == 1);
        CHECK(authority.cacheMetrics().transientPublicationWrites == 1);
    }

    auto restartedBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority restarted(identity, directory.path(), restartedBackend);
    const auto restored = awaitAuthority([&] {
        return restarted.queryTransientFinalNativeGrid(REGION,
                                                       AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(restored.isReady());
    REQUIRE(*restored.value());
    CHECK((*restored.value())->samples == expected);
    CHECK(restartedBackend->callCount() == 0);
    CHECK(restarted.cacheMetrics().transientDiskLoads == 1);
    CHECK(restarted.cacheMetrics().transientPublicationWrites == 0);

    GenerationIdentity incompatibleIdentity = identity;
    ++incompatibleIdentity.seed;
    auto incompatibleBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority incompatible(incompatibleIdentity, directory.path(),
                                        incompatibleBackend);
    const auto rejected = awaitAuthority([&] {
        return incompatible.queryTransientFinalNativeGrid(
            REGION, AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(rejected.status() == AuthorityStatus::FAILED);
    REQUIRE(rejected.failure());
    CHECK(rejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    CHECK(incompatibleBackend->callCount() == 0);
}

TEST_CASE("Corrupt protected transient authority is inferred and atomically repaired",
          "[learned][transient][persistence][corruption][regression]") {
    TempDir directory("learned_transient_repair");
    const GenerationIdentity identity = testIdentity(0x7A11'51E4'0003ULL);
    constexpr NativeRect REGION{
        .rowBegin = 250,
        .columnBegin = -518,
        .rowEnd = 258,
        .columnEnd = -510,
    };
    {
        auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
        CachedTerrainAuthority authority(identity, directory.path(), backend);
        REQUIRE(awaitAuthority([&] {
                    return authority.queryTransientFinalNativeGrid(REGION,
                                                                   AuthorityRequestPriority::SPAWN);
                }).isReady());
        REQUIRE(backend->callCount() == 1);
    }

    std::vector<std::filesystem::path> grids;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory.path())) {
        if (entry.is_regular_file() && entry.path().extension() == ".rytg")
            grids.push_back(entry.path());
    }
    REQUIRE(grids.size() == 1);
    std::vector<uint8_t> corrupt = readBytes(grids.front());
    REQUIRE(corrupt.size() > 104);
    corrupt.back() ^= 0xA5U;
    writeBytes(grids.front(), corrupt);

    auto repairBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    {
        CachedTerrainAuthority repaired(identity, directory.path(), repairBackend);
        REQUIRE(awaitAuthority([&] {
                    return repaired.queryTransientFinalNativeGrid(
                        REGION, AuthorityRequestPriority::PROTECTED_HANDOFF);
                }).isReady());
        CHECK(repairBackend->callCount() == 1);
        CHECK(repaired.cacheMetrics().transientRepairs == 1);
        CHECK(repaired.cacheMetrics().transientPublicationWrites == 1);
    }

    auto verificationBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority verified(identity, directory.path(), verificationBackend);
    REQUIRE(awaitAuthority([&] {
                return verified.queryTransientFinalNativeGrid(
                    REGION, AuthorityRequestPriority::PROTECTED_HANDOFF);
            }).isReady());
    CHECK(verificationBackend->callCount() == 0);
    CHECK(verified.cacheMetrics().transientDiskLoads == 1);
}

TEST_CASE("Contained transient final rectangles reuse the smallest cached authority grid",
          "[learned][transient][containment][cache][regression]") {
    TempDir directory("learned_transient_containment");
    const GenerationIdentity identity = testIdentity(0xC0A7'A11E'0001ULL);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    constexpr NativeRect PARENT{
        .rowBegin = -20,
        .columnBegin = 500,
        .rowEnd = -4,
        .columnEnd = 516,
    };
    constexpr NativeRect CHILD{
        .rowBegin = -15,
        .columnBegin = 506,
        .rowEnd = -10,
        .columnEnd = 511,
    };
    TerrainAuthorityCacheConfig config;
    config.byteBudget =
        static_cast<size_t>(PARENT.height() * PARENT.width()) * sizeof(PhysicalTerrainSample);
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);

    const auto parent = awaitAuthority([&] {
        return authority.queryTransientFinalNativeGrid(PARENT,
                                                       AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(parent.isReady());
    REQUIRE(*parent.value());
    REQUIRE((*parent.value())->valid());
    REQUIRE(backend->callCount() == 1);
    const uint64_t completion = authority.cacheMetrics().completionGeneration;

    const auto child =
        authority.queryTransientFinalNativeGrid(CHILD, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(child.isReady());
    REQUIRE(*child.value());
    REQUIRE((*child.value())->region == CHILD);
    REQUIRE((*child.value())->valid());
    CHECK(backend->callCount() == 1);
    CHECK(authority.cacheMetrics().completionGeneration == completion);
    CHECK(authority.cacheMetrics().bytes <= config.byteBudget);

    for (int64_t row = CHILD.rowBegin; row < CHILD.rowEnd; ++row) {
        for (int64_t column = CHILD.columnBegin; column < CHILD.columnEnd; ++column) {
            const PhysicalTerrainSample* expected = (*parent.value())->sample(row, column);
            const PhysicalTerrainSample* actual = (*child.value())->sample(row, column);
            REQUIRE(expected);
            REQUIRE(actual);
            CHECK(actual->elevationMeters == expected->elevationMeters);
            CHECK(actual->meanTemperatureC == expected->meanTemperatureC);
            CHECK(actual->temperatureVariabilityC == expected->temperatureVariabilityC);
            CHECK(actual->annualPrecipitationMm == expected->annualPrecipitationMm);
            CHECK(actual->precipitationCoefficientOfVariation ==
                  expected->precipitationCoefficientOfVariation);
            CHECK(actual->lapseRateCPerMeter == expected->lapseRateCPerMeter);
        }
    }

    const auto native = authority.queryNative(CHILD, AuthorityQuality::FINAL,
                                              AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(native.isReady());
    CHECK(native.value()->samples == (*child.value())->samples);
    constexpr std::array<NativePoint, 3> POINTS{{
        {.row = -15, .column = 506},
        {.row = -12, .column = 509},
        {.row = -11, .column = 510},
    }};
    const auto points = authority.queryNativePoints(POINTS, AuthorityQuality::FINAL,
                                                    AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(points.isReady());
    REQUIRE(points.value()->size() == POINTS.size());
    for (size_t index = 0; index < POINTS.size(); ++index) {
        const PhysicalTerrainSample* expected =
            (*child.value())->sample(POINTS[index].row, POINTS[index].column);
        REQUIRE(expected);
        CHECK(points.value()->at(index).elevationMeters == expected->elevationMeters);
    }
    CHECK(backend->callCount() == 1);
    CHECK(authority.cacheMetrics().bytes <= config.byteBudget);
}

TEST_CASE("Contained transient requests join an in-flight parent rectangle",
          "[learned][transient][containment][single-flight][performance][regression]") {
    TempDir directory("learned_transient_containing_flight");
    const GenerationIdentity identity = testIdentity(0xC0A7'A11E'0002ULL);
    auto backend = std::make_shared<QueueReservationBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    constexpr NativeRect PARENT{
        .rowBegin = -258,
        .columnBegin = 510,
        .rowEnd = 771,
        .columnEnd = 1'539,
    };
    constexpr NativeRect CHILD{
        .rowBegin = -2,
        .columnBegin = 512,
        .rowEnd = 515,
        .columnEnd = 1'029,
    };
    STATIC_REQUIRE(1'029ULL * 1'029ULL <= MAXIMUM_AUTHORITY_QUERY_SAMPLES);

    REQUIRE(
        authority.queryTransientFinalNativeGrid(PARENT, AuthorityRequestPriority::PROTECTED_HANDOFF)
            .status() == AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitUntilEntered(1s));
    const TerrainAuthorityCacheMetrics beforeChild = authority.cacheMetrics();
    REQUIRE(beforeChild.misses == 1);

    const auto childWhileParentRuns =
        authority.queryTransientFinalNativeGrid(CHILD, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(childWhileParentRuns.status() == AuthorityStatus::DEFERRED);
    const TerrainAuthorityCacheMetrics joined = authority.cacheMetrics();
    CHECK(joined.misses == beforeChild.misses);
    CHECK(joined.singleFlightDeferrals == beforeChild.singleFlightDeferrals + 1);

    backend->release();
    const auto child = awaitAuthority([&] {
        return authority.queryTransientFinalNativeGrid(CHILD,
                                                       AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(child.isReady());
    REQUIRE(*child.value());
    CHECK((*child.value())->region == CHILD);
    CHECK((*child.value())->valid());
    CHECK(authority.cacheMetrics().misses == 1);
}

TEST_CASE("FINAL pages materialized from transient authority match direct inference",
          "[learned][transient][materialization][determinism][golden]") {
    TempDir directDirectory("learned_transient_page_direct");
    TempDir transientDirectory("learned_transient_page_materialized");
    const GenerationIdentity identity = testIdentity(0x7A11'5EED'1001ULL);
    constexpr TerrainPageKey KEY{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = 2, .column = -3},
    };

    auto directBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority directAuthority(identity, directDirectory.path(), directBackend);
    const auto direct = awaitPage(directAuthority, KEY);
    REQUIRE(direct.isReady());
    REQUIRE(*direct.value());
    REQUIRE(directBackend->callCount() == 1);
    const Sha256Digest directHash = quantizedPayloadHash(**direct.value());
    CHECK(sha256Hex(directHash) ==
          "74a4d64661d38fc9a63bf4ee4ded692e69935e261132ca1a5398dc40d96c04de");

    const NativeRect pageRegion = *terrainPageNativeRect(KEY.coordinate);
    const NativeRect enclosingRegion{
        .rowBegin = pageRegion.rowBegin - 7,
        .columnBegin = pageRegion.columnBegin - 11,
        .rowEnd = pageRegion.rowEnd + 13,
        .columnEnd = pageRegion.columnEnd + 5,
    };
    auto transientBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority transientAuthority(identity, transientDirectory.path(),
                                              transientBackend);
    const auto transient = awaitAuthority([&] {
        return transientAuthority.queryTransientFinalNativeGrid(
            enclosingRegion, AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(transient.isReady());
    REQUIRE(*transient.value());
    REQUIRE(transientBackend->callCount() == 1);

    const auto materialized = awaitPage(transientAuthority, KEY);
    REQUIRE(materialized.isReady());
    REQUIRE(*materialized.value());
    CHECK(transientBackend->callCount() == 1);
    CHECK((*materialized.value())->samples == (*direct.value())->samples);
    CHECK(quantizedPayloadHash(**materialized.value()) == directHash);

    const TerrainPageStore store(transientDirectory.path(), identity);
    const auto persisted = store.loadPage(KEY);
    REQUIRE(persisted.isReady());
    CHECK(persisted.value()->samples == (*direct.value())->samples);
    const std::vector<uint8_t> bytes = readBytes(store.pagePath(KEY));
    const TerrainPageStore directStore(directDirectory.path(), identity);
    CHECK(bytes == readBytes(directStore.pagePath(KEY)));
    REQUIRE(bytes.size() > 92);
    CHECK(std::equal(bytes.begin(), bytes.begin() + 4,
                     std::array<uint8_t, 4>{'R', 'Y', 'T', 'A'}.begin()));
}

TEST_CASE("FINAL page materialization selects the smallest containing transient grid",
          "[learned][transient][materialization][containment][regression]") {
    TempDir directory("learned_transient_page_smallest");
    const GenerationIdentity identity = testIdentity(0x7A11'5EED'1005ULL);
    constexpr TerrainPageKey KEY{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = 0, .column = 0},
    };
    constexpr NativeRect LARGE{
        .rowBegin = -20,
        .columnBegin = -4,
        .rowEnd = 280,
        .columnEnd = 270,
    };
    constexpr NativeRect SMALL{
        .rowBegin = -4,
        .columnBegin = -20,
        .rowEnd = 270,
        .columnEnd = 270,
    };
    REQUIRE(SMALL.height() * SMALL.width() < LARGE.height() * LARGE.width());

    auto backend = std::make_shared<RegionTaggedTransientTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    LARGE, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
            }).isReady());
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    SMALL, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
            }).isReady());
    REQUIRE(backend->gridCalls() == 2);

    const auto page = awaitPage(authority, KEY);
    REQUIRE(page.isReady());
    REQUIRE(*page.value());
    CHECK(backend->pageCalls() == 0);
    CHECK(backend->gridCalls() == 2);
    REQUIRE_FALSE((*page.value())->samples.empty());
    CHECK(std::ranges::all_of((*page.value())->samples, [](const QuantizedTerrainSample& sample) {
        return sample.elevationMeters == 222;
    }));
}

TEST_CASE("Negative FINAL pages crop from transient authority with half-open ownership",
          "[learned][transient][materialization][negative][golden]") {
    TempDir directDirectory("learned_transient_negative_direct");
    TempDir transientDirectory("learned_transient_negative_materialized");
    const GenerationIdentity identity = testIdentity(0x7A11'5EED'1002ULL);
    constexpr TerrainPageKey KEY{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = -2, .column = -3},
    };
    const NativeRect pageRegion = *terrainPageNativeRect(KEY.coordinate);
    REQUIRE(pageRegion ==
            (NativeRect{.rowBegin = -512, .columnBegin = -768, .rowEnd = -256, .columnEnd = -512}));

    auto directBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority directAuthority(identity, directDirectory.path(), directBackend);
    const auto direct = awaitPage(directAuthority, KEY);
    REQUIRE(direct.isReady());
    REQUIRE(*direct.value());

    auto transientBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority transientAuthority(identity, transientDirectory.path(),
                                              transientBackend);
    constexpr NativeRect ENCLOSING{
        .rowBegin = -519,
        .columnBegin = -779,
        .rowEnd = -249,
        .columnEnd = -501,
    };
    const auto transient = awaitAuthority([&] {
        return transientAuthority.queryTransientFinalNativeGrid(
            ENCLOSING, AuthorityRequestPriority::PROTECTED_HANDOFF);
    });
    REQUIRE(transient.isReady());
    REQUIRE(*transient.value());
    const auto materialized = awaitPage(transientAuthority, KEY);
    REQUIRE(materialized.isReady());
    REQUIRE(*materialized.value());
    CHECK(transientBackend->callCount() == 1);
    const Sha256Digest directHash = quantizedPayloadHash(**direct.value());
    CHECK(sha256Hex(directHash) ==
          "adcb3bbea032e3121eeaa95e75b8bee0459514166c467f0329a065dc9ef5d653");
    CHECK(quantizedPayloadHash(**materialized.value()) == directHash);

    const PhysicalTerrainSample* first =
        (*transient.value())->sample(pageRegion.rowBegin, pageRegion.columnBegin);
    const PhysicalTerrainSample* last =
        (*transient.value())->sample(pageRegion.rowEnd - 1, pageRegion.columnEnd - 1);
    REQUIRE(first);
    REQUIRE(last);
    CHECK(dequantizeTerrainSample((*materialized.value())->samples.front()) == *first);
    CHECK(dequantizeTerrainSample((*materialized.value())->samples.back()) == *last);
}

TEST_CASE("Materialized FINAL pages survive restart without model inference",
          "[learned][transient][materialization][persistence][restart][regression]") {
    TempDir directory("learned_transient_page_restart");
    const GenerationIdentity identity = testIdentity(0x7A11'5EED'1003ULL);
    constexpr TerrainPageKey KEY{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = -1, .column = 4},
    };
    const NativeRect pageRegion = *terrainPageNativeRect(KEY.coordinate);
    const NativeRect enclosing{
        .rowBegin = pageRegion.rowBegin - 3,
        .columnBegin = pageRegion.columnBegin - 5,
        .rowEnd = pageRegion.rowEnd + 7,
        .columnEnd = pageRegion.columnEnd + 9,
    };
    Sha256Digest expectedHash{};
    {
        auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
        CachedTerrainAuthority authority(identity, directory.path(), backend);
        REQUIRE(awaitAuthority([&] {
                    return authority.queryTransientFinalNativeGrid(
                        enclosing, AuthorityRequestPriority::PROTECTED_HANDOFF);
                }).isReady());
        const auto page = awaitPage(authority, KEY);
        REQUIRE(page.isReady());
        REQUIRE(*page.value());
        expectedHash = quantizedPayloadHash(**page.value());
        CHECK(backend->callCount() == 1);
        CHECK(authority.cacheMetrics().publicationWrites == 1);
    }

    auto restartedBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority restarted(identity, directory.path(), restartedBackend);
    const auto restored = awaitPage(restarted, KEY);
    REQUIRE(restored.isReady());
    REQUIRE(*restored.value());
    CHECK(quantizedPayloadHash(**restored.value()) == expectedHash);
    CHECK(restartedBackend->callCount() == 0);
    CHECK(restarted.cacheMetrics().diskLoads == 1);
}

TEST_CASE("Concurrent FINAL page requests share one transient materialization",
          "[learned][transient][materialization][concurrency][single-flight][regression]") {
    TempDir directory("learned_transient_page_concurrency");
    const GenerationIdentity identity = testIdentity(0x7A11'5EED'1004ULL);
    constexpr TerrainPageKey KEY{
        .quality = AuthorityQuality::FINAL,
        .coordinate = {.row = 3, .column = 2},
    };
    const NativeRect pageRegion = *terrainPageNativeRect(KEY.coordinate);
    const NativeRect enclosing{
        .rowBegin = pageRegion.rowBegin - 5,
        .columnBegin = pageRegion.columnBegin - 5,
        .rowEnd = pageRegion.rowEnd + 5,
        .columnEnd = pageRegion.columnEnd + 5,
    };
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    enclosing, AuthorityRequestPriority::PROTECTED_HANDOFF);
            }).isReady());
    REQUIRE(backend->callCount() == 1);

    constexpr size_t REQUEST_COUNT = 12;
    std::barrier start(static_cast<std::ptrdiff_t>(REQUEST_COUNT + 1));
    std::array<std::future<AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>>,
               REQUEST_COUNT>
        requests;
    for (auto& request : requests) {
        request = std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return awaitPage(authority, KEY, AuthorityRequestPriority::PROTECTED_HANDOFF);
        });
    }
    start.arrive_and_wait();

    std::optional<Sha256Digest> expectedHash;
    for (auto& request : requests) {
        const auto result = request.get();
        REQUIRE(result.isReady());
        REQUIRE(*result.value());
        const Sha256Digest hash = quantizedPayloadHash(**result.value());
        if (!expectedHash)
            expectedHash = hash;
        else
            CHECK(hash == *expectedHash);
    }
    CHECK(backend->callCount() == 1);
    CHECK(authority.cacheMetrics().publicationWrites == 1);
    const TerrainPageStore store(directory.path(), identity);
    const auto persisted = store.loadPage(KEY);
    REQUIRE(persisted.isReady());
    REQUIRE(expectedHash);
    CHECK(quantizedPayloadHash(*persisted.value()) == *expectedHash);
}

TEST_CASE("Spawn transient authority survives coarse page cache pressure",
          "[learned][transient][spawn][cache][retention][regression]") {
    TempDir directory("learned_spawn_transient_retention");
    const GenerationIdentity identity = testIdentity(0x5A11'CA4E'0001ULL);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    constexpr NativeRect PARENT{
        .rowBegin = -8,
        .columnBegin = 504,
        .rowEnd = 8,
        .columnEnd = 520,
    };
    constexpr NativeRect CHILD{
        .rowBegin = -4,
        .columnBegin = 508,
        .rowEnd = 4,
        .columnEnd = 516,
    };
    const size_t parentBytes =
        static_cast<size_t>(PARENT.height() * PARENT.width()) * sizeof(PhysicalTerrainSample);
    constexpr size_t PAGE_BYTES = AUTHORITY_PAGE_SAMPLE_COUNT * sizeof(QuantizedTerrainSample);
    TerrainAuthorityCacheConfig config;
    config.byteBudget = parentBytes + PAGE_BYTES;
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);

    const auto parent = awaitAuthority([&] {
        return authority.queryTransientFinalNativeGrid(PARENT, AuthorityRequestPriority::SPAWN);
    });
    REQUIRE(parent.isReady());
    REQUIRE(*parent.value());

    for (const TerrainPageKey key : {
             TerrainPageKey{.quality = AuthorityQuality::PREVIEW,
                            .coordinate = {.row = 0, .column = 0}},
             TerrainPageKey{.quality = AuthorityQuality::PREVIEW,
                            .coordinate = {.row = 0, .column = 1}},
         }) {
        const auto page = awaitAuthority(
            [&] { return authority.preparePage(key, AuthorityRequestPriority::COARSE_PREVIEW); });
        REQUIRE(page.isReady());
    }

    const size_t callsBeforeReuse = backend->callCount();
    const auto child =
        authority.queryTransientFinalNativeGrid(CHILD, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(child.isReady());
    REQUIRE(*child.value());
    CHECK((*child.value())->region == CHILD);
    CHECK(backend->callCount() == callsBeforeReuse);
    CHECK(authority.cacheMetrics().bytes <= config.byteBudget);
}

TEST_CASE("RYTA pages round-trip atomically and reject incompatible fingerprints", "[learned]") {
    TempDir directory("learned_page");
    const GenerationIdentity identity = testIdentity();
    TerrainPageStore store(directory.path(), identity);
    DeterministicFakeTerrainBackend backend;
    const TerrainPageKey key = finalPage(-7, 11);
    auto inferred = backend.inferPage(identity, key);
    REQUIRE(inferred.isReady());

    auto written = store.writePage(*inferred.value());
    REQUIRE(written.isReady());
    const std::filesystem::path path = store.pagePath(key);
    REQUIRE(std::filesystem::exists(path));
    REQUIRE(readBytes(path)[0] == 'R');
    REQUIRE(readBytes(path)[1] == 'Y');
    REQUIRE(readBytes(path)[2] == 'T');
    REQUIRE(readBytes(path)[3] == 'A');
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path()))
        REQUIRE(entry.path().extension() != ".tmp");

    auto loaded = store.loadPage(key);
    REQUIRE(loaded.isReady());
    REQUIRE(loaded.value()->key == key);
    REQUIRE(loaded.value()->samples == inferred.value()->samples);

    GenerationIdentity incompatibleIdentity = identity;
    ++incompatibleIdentity.seed;
    TerrainPageStore incompatible(directory.path(), incompatibleIdentity);
    auto rejected = incompatible.loadPage(key);
    REQUIRE(rejected.status() == AuthorityStatus::FAILED);
    REQUIRE(rejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE_FALSE(rejected.failure()->retriable);
    auto rejectedWrite = incompatible.writePage(*inferred.value());
    REQUIRE(rejectedWrite.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedWrite.failure()->code == GenerationFailureCode::INVALID_REQUEST);
    REQUIRE(store.loadPage(key).isReady());
}

TEST_CASE("Stale PREVIEW pages remain fail closed until profile identity selection",
          "[learned][persistence][preview][fingerprint][regression]") {
    TempDir directory("learned_stale_preview_rejection");
    const GenerationIdentity identity = testIdentity(0xC0A2'5E00'0009ULL);
    GenerationIdentity previousIdentity = identity;
    --previousIdentity.postprocessingRevision;
    STATIC_REQUIRE(GENERATOR_V4_POSTPROCESSING_REVISION == 9);
    REQUIRE(previousIdentity.postprocessingRevision == 8);
    REQUIRE(previousIdentity.fingerprint() != identity.fingerprint());

    constexpr TerrainPageKey PREVIEW_KEY{
        .quality = AuthorityQuality::PREVIEW,
        .coordinate = {.row = -7, .column = 11},
    };
    const TerrainPageKey finalKey = finalPage(-7, 11);
    DeterministicFakeTerrainBackend seedBackend;
    TerrainPageStore store(directory.path(), identity);
    std::map<TerrainPageKey, std::vector<uint8_t>> staleBytes;
    for (const TerrainPageKey key : {PREVIEW_KEY, finalKey}) {
        const auto page = seedBackend.inferPage(identity, key);
        REQUIRE(page.isReady());
        REQUIRE(store.writePage(*page.value()).isReady());
        rewriteTerrainPageFingerprint(store.pagePath(key), previousIdentity.fingerprint());
        staleBytes.emplace(key, readBytes(store.pagePath(key)));
        const auto stale = store.loadPage(key);
        REQUIRE(stale.status() == AuthorityStatus::FAILED);
        REQUIRE(stale.failure());
        CHECK(stale.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
        CHECK_FALSE(stale.failure()->retriable);
    }

    const auto currentPreview = seedBackend.inferPage(identity, PREVIEW_KEY);
    REQUIRE(currentPreview.isReady());
    const auto rejectedWrite = store.writePage(*currentPreview.value());
    REQUIRE(rejectedWrite.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedWrite.failure());
    CHECK(rejectedWrite.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    CHECK_FALSE(rejectedWrite.failure()->retriable);
    CHECK(readBytes(store.pagePath(PREVIEW_KEY)) == staleBytes.at(PREVIEW_KEY));

    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const auto rejectedPreview =
        awaitPage(authority, PREVIEW_KEY, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(rejectedPreview.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedPreview.failure());
    CHECK(rejectedPreview.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    CHECK_FALSE(rejectedPreview.failure()->retriable);
    CHECK(backend->callCount() == 0);
    CHECK(authority.cacheMetrics().repairs == 0);
    CHECK(readBytes(store.pagePath(PREVIEW_KEY)) == staleBytes.at(PREVIEW_KEY));

    const auto rejectedFinal = awaitPage(authority, finalKey);
    REQUIRE(rejectedFinal.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedFinal.failure());
    CHECK(rejectedFinal.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    CHECK_FALSE(rejectedFinal.failure()->retriable);
    CHECK(backend->callCount() == 0);
    CHECK(readBytes(store.pagePath(finalKey)) == staleBytes.at(finalKey));

    GenerationIdentity otherSeedIdentity = identity;
    ++otherSeedIdentity.seed;
    auto otherSeedBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority otherSeedAuthority(otherSeedIdentity, directory.path(),
                                              otherSeedBackend);
    const auto rejectedSeed =
        awaitPage(otherSeedAuthority, PREVIEW_KEY, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(rejectedSeed.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedSeed.failure());
    CHECK(rejectedSeed.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    CHECK_FALSE(rejectedSeed.failure()->retriable);
    CHECK(otherSeedBackend->callCount() == 0);
}

TEST_CASE("RYTA concurrent publishers preserve the first immutable payload",
          "[learned][persistence][concurrency]") {
    TempDir directory("learned_concurrent_immutable_publication");
    const GenerationIdentity identity = testIdentity(0xC0FF'EE00'0001ULL);
    const TerrainPageKey key = finalPage(-19, 27);
    DeterministicFakeTerrainBackend backend;
    const auto inferred = backend.inferPage(identity, key);
    REQUIRE(inferred.isReady());

    TerrainAuthorityPage firstPage = *inferred.value();
    TerrainAuthorityPage conflictingPage = firstPage;
    conflictingPage.samples.front().elevationMeters ^= 1;
    REQUIRE(conflictingPage.samples != firstPage.samples);

    TerrainPageStore seedStore(directory.path(), identity);
    REQUIRE(seedStore.writePage(firstPage).isReady());
    std::vector<uint8_t> corrupted = readBytes(seedStore.pagePath(key));
    corrupted.back() ^= 0xA5U;
    writeBytes(seedStore.pagePath(key), corrupted);
    const auto corrupt = seedStore.loadPage(key);
    REQUIRE(corrupt.status() == AuthorityStatus::FAILED);
    REQUIRE(corrupt.failure());
    REQUIRE(corrupt.failure()->code == GenerationFailureCode::CORRUPT_PAGE);

    auto publishBarrier = std::make_shared<std::barrier<>>(2);
    auto hooks = std::make_shared<TerrainPageStore::TestHooks>();
    hooks->beforeExclusivePublish = [publishBarrier] { publishBarrier->arrive_and_wait(); };
    TerrainPageStore firstStore(directory.path(), identity, hooks);
    TerrainPageStore conflictingStore(directory.path(), identity, hooks);

    AuthorityResult<bool> firstResult;
    AuthorityResult<bool> conflictingResult;
    std::thread firstWriter([&] { firstResult = firstStore.writePage(firstPage); });
    std::thread conflictingWriter(
        [&] { conflictingResult = conflictingStore.writePage(conflictingPage); });
    firstWriter.join();
    conflictingWriter.join();

    const bool firstPublished = firstResult.isReady();
    const bool conflictingPublished = conflictingResult.isReady();
    REQUIRE(static_cast<int>(firstPublished) + static_cast<int>(conflictingPublished) == 1);
    const AuthorityResult<bool>& rejected = firstPublished ? conflictingResult : firstResult;
    REQUIRE(rejected.status() == AuthorityStatus::FAILED);
    REQUIRE(rejected.failure());
    REQUIRE(rejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE_FALSE(rejected.failure()->retriable);

    const TerrainAuthorityPage& expected = firstPublished ? firstPage : conflictingPage;
    TerrainPageStore observer(directory.path(), identity);
    const auto persisted = observer.loadPage(key);
    REQUIRE(persisted.isReady());
    REQUIRE(persisted.value()->samples == expected.samples);
    REQUIRE(observer.writePage(expected).isReady());
    const TerrainAuthorityPage& rejectedPage = firstPublished ? conflictingPage : firstPage;
    const auto rejectedRetry = observer.writePage(rejectedPage);
    REQUIRE(rejectedRetry.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedRetry.failure());
    REQUIRE(rejectedRetry.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE(std::ranges::none_of(
        std::filesystem::directory_iterator(observer.pagePath(key).parent_path()),
        [](const auto& entry) { return entry.path().filename().string().contains(".tmp."); }));
}

TEST_CASE("RYTA payload corruption is detected and can be repaired", "[learned]") {
    TempDir directory("learned_repair");
    const GenerationIdentity identity = testIdentity();
    const TerrainPageKey key = finalPage(2, -4);
    auto firstBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    {
        CachedTerrainAuthority authority(identity, directory.path(), firstBackend);
        REQUIRE(awaitPage(authority, key).isReady());
    }

    TerrainPageStore store(directory.path(), identity);
    std::vector<uint8_t> bytes = readBytes(store.pagePath(key));
    bytes.back() ^= 0x5AU;
    writeBytes(store.pagePath(key), bytes);
    auto corrupt = store.loadPage(key);
    REQUIRE(corrupt.status() == AuthorityStatus::FAILED);
    REQUIRE(corrupt.failure()->code == GenerationFailureCode::CORRUPT_PAGE);

    auto repairBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority repaired(identity, directory.path(), repairBackend);
    REQUIRE(awaitPage(repaired, key).isReady());
    REQUIRE(repairBackend->callCount() == 1);
    REQUIRE(repaired.cacheMetrics().repairs == 1);
    REQUIRE(store.loadPage(key).isReady());
}

TEST_CASE("RYTA publication ignores an interrupted staging file", "[learned][persistence]") {
    TempDir directory("learned_interrupted_write");
    const GenerationIdentity identity = testIdentity();
    const TerrainPageKey key = finalPage(-11, -13);
    TerrainPageStore store(directory.path(), identity);
    const std::filesystem::path canonicalPath = store.pagePath(key);
    REQUIRE(std::filesystem::create_directories(canonicalPath.parent_path()));
    std::filesystem::path interruptedPath = canonicalPath;
    interruptedPath += ".tmp.interrupted";
    writeBytes(interruptedPath, {'R', 'Y', 'T', 'A'});

    auto missing = store.loadPage(key);
    REQUIRE(missing.status() == AuthorityStatus::DEFERRED);
    REQUIRE(missing.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE_FALSE(std::filesystem::exists(canonicalPath));

    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    REQUIRE(awaitPage(authority, key).isReady());
    REQUIRE(backend->callCount() == 1);
    REQUIRE(store.loadPage(key).isReady());
    REQUIRE(std::filesystem::exists(interruptedPath));
}

TEST_CASE("Learned terrain cache coalesces identical concurrent requests", "[learned]") {
    TempDir directory("learned_single_flight");
    const GenerationIdentity identity = testIdentity();
    const TerrainPageKey key = finalPage(0, 0);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(75ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const uint64_t completionBefore = authority.cacheMetrics().completionGeneration;

    const auto firstStart = std::chrono::steady_clock::now();
    auto first = authority.preparePage(key);
    const auto firstDuration = std::chrono::steady_clock::now() - firstStart;
    REQUIRE(first.status() == AuthorityStatus::DEFERRED);
    REQUIRE(firstDuration < 20ms);
    waitForBackendCall(*backend);
    const auto secondStart = std::chrono::steady_clock::now();
    auto second = authority.preparePage(key);
    const auto secondDuration = std::chrono::steady_clock::now() - secondStart;
    REQUIRE(second.status() == AuthorityStatus::DEFERRED);
    REQUIRE(secondDuration < 20ms);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore);
    REQUIRE(awaitPage(authority, key).isReady());
    REQUIRE(backend->callCount() == 1);
    REQUIRE(authority.cacheMetrics().singleFlightDeferrals >= 1);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore + 1);

    const uint64_t hitsBefore = authority.cacheMetrics().hits;
    REQUIRE(authority.preparePage(key).isReady());
    REQUIRE(authority.cacheMetrics().hits == hitsBefore + 1);
    REQUIRE(authority.cacheMetrics().completionGeneration == completionBefore + 1);
    REQUIRE(backend->callCount() == 1);
}

TEST_CASE("Learned terrain authority bounds outstanding work and reloads persisted pages",
          "[learned]") {
    TempDir directory("learned_bounded");
    const GenerationIdentity identity = testIdentity();
    const TerrainPageKey firstKey = finalPage(1, 1);
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(75ms);
    TerrainAuthorityCacheConfig config;
    config.maximumEntries = 1;
    config.maximumOutstandingRequests = 1;
    config.maximumConcurrentBuilds = 1;
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);

    auto first = authority.preparePage(firstKey);
    REQUIRE(first.status() == AuthorityStatus::DEFERRED);
    waitForBackendCall(*backend);
    auto deferred = authority.preparePage(finalPage(2, 2));
    REQUIRE(deferred.status() == AuthorityStatus::DEFERRED);
    REQUIRE(deferred.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(deferred.failure()->retriable);
    REQUIRE(awaitPage(authority, firstKey).isReady());
    REQUIRE(authority.cacheMetrics().deferredRequests == 1);

    REQUIRE(awaitPage(authority, finalPage(3, 3)).isReady());
    REQUIRE(authority.cacheMetrics().entries == 1);
    REQUIRE(authority.cacheMetrics().evictions == 1);

    auto reloadingBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority reloaded(identity, directory.path(), reloadingBackend, config);
    REQUIRE(awaitPage(reloaded, firstKey).isReady());
    REQUIRE(reloadingBackend->callCount() == 0);
    REQUIRE(reloaded.cacheMetrics().diskLoads == 1);
}

TEST_CASE("Distant decoded pages cannot evict current-player exact authority",
          "[learned][cache][priority][exact][regression]") {
    TempDir directory("learned_priority_page_cache");
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumEntries = 2;
    CachedTerrainAuthority authority(testIdentity(0xCA4E'0000'0000'0001ULL), directory.path(),
                                     backend, config);
    const TerrainPageKey exact = finalPage(610, 0);
    const TerrainPageKey firstDistant = finalPage(610, 1);
    const TerrainPageKey secondDistant = finalPage(610, 2);

    REQUIRE(awaitPage(authority, exact, AuthorityRequestPriority::EXPLORATION_EXACT).isReady());
    REQUIRE(awaitPage(authority, firstDistant, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT)
                .isReady());
    REQUIRE(awaitPage(authority, secondDistant, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT)
                .isReady());
    REQUIRE(authority.cacheMetrics().entries == 2);
    REQUIRE(authority.cacheMetrics().evictions == 1);

    CHECK(authority.preparePage(exact, AuthorityRequestPriority::EXPLORATION_EXACT).isReady());
    const auto evicted =
        authority.preparePage(firstDistant, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    CHECK(evicted.status() == AuthorityStatus::DEFERRED);
    CHECK(evicted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(awaitPage(authority, firstDistant, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT)
                .isReady());
    CHECK(backend->callCount() == 3);
}

TEST_CASE("Distant transient grids cannot evict current-player exact authority",
          "[learned][transient][cache][priority][exact][regression]") {
    TempDir directory("learned_priority_transient_cache");
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    constexpr NativeRect EXACT{
        .rowBegin = 0,
        .columnBegin = 0,
        .rowEnd = 8,
        .columnEnd = 8,
    };
    constexpr NativeRect FIRST_DISTANT{
        .rowBegin = 16,
        .columnBegin = 16,
        .rowEnd = 24,
        .columnEnd = 24,
    };
    constexpr NativeRect SECOND_DISTANT{
        .rowBegin = 32,
        .columnBegin = 32,
        .rowEnd = 40,
        .columnEnd = 40,
    };
    constexpr size_t GRID_BYTES = 8U * 8U * sizeof(PhysicalTerrainSample);
    TerrainAuthorityCacheConfig config;
    config.byteBudget = GRID_BYTES * 2;
    CachedTerrainAuthority authority(testIdentity(0xCA4E'0000'0000'0002ULL), directory.path(),
                                     backend, config);

    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    EXACT, AuthorityRequestPriority::EXPLORATION_EXACT);
            }).isReady());
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    FIRST_DISTANT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
            }).isReady());
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    SECOND_DISTANT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
            }).isReady());
    REQUIRE(authority.cacheMetrics().entries == 2);
    REQUIRE(authority.cacheMetrics().evictions == 1);

    CHECK(
        authority.queryTransientFinalNativeGrid(EXACT, AuthorityRequestPriority::EXPLORATION_EXACT)
            .isReady());
    const auto evicted = authority.queryTransientFinalNativeGrid(
        FIRST_DISTANT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    CHECK(evicted.status() == AuthorityStatus::DEFERRED);
    CHECK(evicted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    FIRST_DISTANT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
            }).isReady());
    CHECK(backend->callCount() == 4);
}

TEST_CASE("Authority prefetchers surface terminal failures before a full queue can stall",
          "[learned][concurrency][failure]") {
    TempDir directory("learned_prefetch_terminal_failure");
    const GenerationIdentity identity = testIdentity();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 1;
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<FailingTerrainBackend>(), config);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::PREVIEW);
    constexpr TerrainPageCoordinate FIRST{.row = 0, .column = 0};
    constexpr TerrainPageCoordinate SECOND{.row = 0, .column = 1};

    const auto initial =
        context->requestAuthorityPage(FIRST, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(initial.status() == AuthorityStatus::DEFERRED);
    const auto full =
        context->requestAuthorityPage(SECOND, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(full.status() == AuthorityStatus::DEFERRED);
    REQUIRE(full.failure()->code == GenerationFailureCode::QUEUE_FULL);

    // The page accepted before the bound was reached must be polled. Its
    // terminal result latches the shared context, so a horizon scheduler can
    // present repair UI rather than repeatedly retrying an unrelated page.
    const auto terminal = awaitAuthority([&] {
        return context->requestAuthorityPage(FIRST, AuthorityRequestPriority::COARSE_PREVIEW);
    });
    REQUIRE(terminal.status() == AuthorityStatus::FAILED);
    REQUIRE(terminal.failure()->code == GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(context->failure());
    REQUIRE(authority->cacheMetrics().completionGeneration == 1);

    const auto afterFailure =
        context->requestAuthorityPage(SECOND, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(afterFailure.status() == AuthorityStatus::FAILED);
    REQUIRE(afterFailure.failure()->code == GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(authority->cacheMetrics().completionGeneration == 1);
}

TEST_CASE("Learned terrain coordinator starts lazily on its first valid page request",
          "[learned][concurrency][bootstrap]") {
    TempDir directory("learned_lazy_coordinator");
    const GenerationIdentity identity = testIdentity();
    CachedTerrainAuthority authority(identity, directory.path(),
                                     std::make_shared<DeterministicFakeTerrainBackend>());
    REQUIRE_FALSE(authority.cacheMetrics().coordinatorStarted);
    const auto first = authority.preparePage(finalPage(0, 0), AuthorityRequestPriority::SPAWN);
    REQUIRE(first.status() == AuthorityStatus::DEFERRED);
    REQUIRE(authority.cacheMetrics().coordinatorStarted);
    REQUIRE(awaitPage(authority, finalPage(0, 0), AuthorityRequestPriority::SPAWN).isReady());
}

TEST_CASE("Learned terrain coordinator clamps requested queue capacity to sixty four pages",
          "[learned][concurrency][work-limit]") {
    TempDir directory("learned_hard_queue_bound");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(500ms);
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 1'000;
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);

    REQUIRE(authority.preparePage(finalPage(0, 0)).status() == AuthorityStatus::DEFERRED);
    waitForBackendCall(*backend);
    for (int64_t index = 1; index < 64; ++index) {
        const auto queued = authority.preparePage(finalPage(index, 0));
        REQUIRE(queued.status() == AuthorityStatus::DEFERRED);
        REQUIRE(queued.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }
    const auto rejected = authority.preparePage(finalPage(64, 0));
    REQUIRE(rejected.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejected.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().activeBuilds == 1);
    REQUIRE(authority.cacheMetrics().queuedBuilds == 63);
}

TEST_CASE("Low-priority terrain work preserves half the shared queue for urgent requests",
          "[learned][concurrency][priority][work-limit][regression]") {
    TempDir directory("learned_low_priority_reservation");
    const GenerationIdentity identity = testIdentity(0x10A0'B0C0'D0E0'0001ULL);
    auto backend = std::make_shared<QueueReservationBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    struct ReleaseBackendOnExit {
        std::shared_ptr<QueueReservationBackend> backend;
        ~ReleaseBackendOnExit() { backend->release(); }
    } releaseBackendOnExit{backend};

    const TerrainPageKey blocker = finalPage(-900, -900);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitUntilEntered(2s));

    // Direct PREVIEW calls default to an exact service priority, but their
    // quality still makes them horizon work for admission accounting.
    for (int64_t index = 0;
         index < static_cast<int64_t>(MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS) - 2; ++index) {
        const TerrainPageKey preview{
            .quality = AuthorityQuality::PREVIEW,
            .coordinate = {.row = 100, .column = index},
        };
        const auto admitted = authority.preparePage(preview);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }

    const TerrainPageKey promotable = finalPage(-899, -899);
    const auto queuedPromotable =
        authority.preparePage(promotable, AuthorityRequestPriority::SPECULATIVE_PREFETCH);
    REQUIRE(queuedPromotable.status() == AuthorityStatus::DEFERRED);
    REQUIRE(queuedPromotable.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(authority.cacheMetrics().lowPriorityOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS);
    REQUIRE(authority.cacheMetrics().visibleOrLowerOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS);

    // A duplicate urgent request promotes an unstarted FINAL single flight
    // instead of allocating another request or remaining in the low lane.
    const auto promoted = authority.preparePage(promotable, AuthorityRequestPriority::SPAWN);
    REQUIRE(promoted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(promoted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(authority.cacheMetrics().lowPriorityOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS - 1);
    REQUIRE(authority.cacheMetrics().visibleOrLowerOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS - 1);

    const std::array<TerrainPageKey, 2> previewClosure{{
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = 101, .column = 0}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = 101, .column = 1}},
    }};
    const size_t queuedBeforeClosure = authority.cacheMetrics().queuedBuilds;
    const auto rejectedClosure =
        authority.preparePages(previewClosure, AuthorityRequestPriority::SPAWN);
    REQUIRE(rejectedClosure.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedClosure.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().queuedBuilds == queuedBeforeClosure);

    constexpr NativeRect LOW_TRANSIENT{
        .rowBegin = 0,
        .columnBegin = 0,
        .rowEnd = 8,
        .columnEnd = 8,
    };
    const auto lowTransient = authority.queryTransientFinalNativeGrid(
        LOW_TRANSIENT, AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(lowTransient.status() == AuthorityStatus::DEFERRED);
    REQUIRE(lowTransient.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE(authority.cacheMetrics().lowPriorityOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS);

    constexpr CoarseSpawnRegion LOW_COARSE{
        .rowBegin = 0,
        .columnBegin = 0,
        .rowEnd = 4,
        .columnEnd = 4,
    };
    const auto rejectedCoarse =
        authority.queryCoarseSpawnGrid(LOW_COARSE, AuthorityRequestPriority::SPECULATIVE_PREFETCH);
    REQUIRE(rejectedCoarse.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedCoarse.failure()->code == GenerationFailureCode::QUEUE_FULL);

    const TerrainPageKey rejectedPreview{
        .quality = AuthorityQuality::PREVIEW,
        .coordinate = {.row = 102, .column = 0},
    };
    const auto rejectedPage = authority.preparePage(rejectedPreview);
    REQUIRE(rejectedPage.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedPage.failure()->code == GenerationFailureCode::QUEUE_FULL);

    // Every urgent entry point remains able to use the reservation shared by
    // pages, transient grids, and coarse spawn work.
    constexpr CoarseSpawnRegion HIGH_COARSE{
        .rowBegin = 8,
        .columnBegin = 8,
        .rowEnd = 12,
        .columnEnd = 12,
    };
    const auto highCoarse =
        authority.queryCoarseSpawnGrid(HIGH_COARSE, AuthorityRequestPriority::SPAWN);
    REQUIRE(highCoarse.status() == AuthorityStatus::DEFERRED);
    REQUIRE(highCoarse.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    constexpr NativeRect HIGH_TRANSIENT{
        .rowBegin = 16,
        .columnBegin = 16,
        .rowEnd = 24,
        .columnEnd = 24,
    };
    const auto highTransient = authority.queryTransientFinalNativeGrid(
        HIGH_TRANSIENT, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(highTransient.status() == AuthorityStatus::DEFERRED);
    REQUIRE(highTransient.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    for (int64_t index = 0; index < 29; ++index) {
        const auto admitted = authority.preparePage(finalPage(200, index),
                                                    AuthorityRequestPriority::EXPLORATION_EXACT);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }

    const auto full = authority.preparePage(finalPage(200, 29), AuthorityRequestPriority::SPAWN);
    REQUIRE(full.status() == AuthorityStatus::DEFERRED);
    REQUIRE(full.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.activeBuilds == 1);
    REQUIRE(metrics.queuedBuilds == MAXIMUM_AUTHORITY_QUEUED_REQUESTS - 1);
    REQUIRE(metrics.lowPriorityOutstandingRequests == MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS - 1);
    REQUIRE(metrics.visibleOrLowerOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS - 1);
    REQUIRE(metrics.lowPriorityDeferredRequests == 3);
    REQUIRE(metrics.visibleOrLowerDeferredRequests == 0);
    REQUIRE(metrics.deferredRequests == 3);
    REQUIRE(metrics.singleFlightDeferrals == 1);
    REQUIRE(metrics.higherPriorityPreemptions == 1);
}

TEST_CASE("Visible refinement preserves sixteen shared queue slots for exact authority",
          "[learned][concurrency][priority][work-limit][regression]") {
    TempDir directory("learned_visible_priority_reservation");
    const GenerationIdentity identity = testIdentity(0x5151'B1E0'0000'0001ULL);
    auto backend = std::make_shared<QueueReservationBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    struct ReleaseBackendOnExit {
        std::shared_ptr<QueueReservationBackend> backend;
        ~ReleaseBackendOnExit() { backend->release(); }
    } releaseBackendOnExit{backend};

    const TerrainPageKey blocker = finalPage(-800, -800);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitUntilEntered(2s));

    for (int64_t index = 0;
         index < static_cast<int64_t>(MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS) - 1; ++index) {
        const TerrainPageKey preview{
            .quality = AuthorityQuality::PREVIEW,
            .coordinate = {.row = 300, .column = index},
        };
        const auto admitted = authority.preparePage(preview);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }
    REQUIRE(authority.cacheMetrics().lowPriorityOutstandingRequests ==
            MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS);

    constexpr NativeRect VISIBLE_TRANSIENT{
        .rowBegin = 32,
        .columnBegin = 32,
        .rowEnd = 40,
        .columnEnd = 40,
    };
    const auto visibleTransient = authority.queryTransientFinalNativeGrid(
        VISIBLE_TRANSIENT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(visibleTransient.status() == AuthorityStatus::DEFERRED);
    REQUIRE(visibleTransient.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);

    constexpr CoarseSpawnRegion VISIBLE_COARSE{
        .rowBegin = 16,
        .columnBegin = 16,
        .rowEnd = 20,
        .columnEnd = 20,
    };
    const auto visibleCoarse = authority.queryCoarseSpawnGrid(
        VISIBLE_COARSE, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(visibleCoarse.status() == AuthorityStatus::DEFERRED);
    REQUIRE(visibleCoarse.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);

    for (int64_t index = 0; index < 14; ++index) {
        const auto admitted = authority.preparePage(
            finalPage(301, index), AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }
    REQUIRE(authority.cacheMetrics().visibleOrLowerOutstandingRequests ==
            MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS);

    const std::array<TerrainPageKey, 2> visibleClosure{
        finalPage(302, 0),
        finalPage(302, 1),
    };
    const size_t queuedBeforeClosure = authority.cacheMetrics().queuedBuilds;
    const auto rejectedClosure =
        authority.preparePages(visibleClosure, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(rejectedClosure.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedClosure.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().queuedBuilds == queuedBeforeClosure);

    constexpr NativeRect REJECTED_TRANSIENT{
        .rowBegin = 48,
        .columnBegin = 48,
        .rowEnd = 56,
        .columnEnd = 56,
    };
    const auto rejectedTransient = authority.queryTransientFinalNativeGrid(
        REJECTED_TRANSIENT, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(rejectedTransient.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedTransient.failure()->code == GenerationFailureCode::QUEUE_FULL);

    constexpr CoarseSpawnRegion REJECTED_COARSE{
        .rowBegin = 24,
        .columnBegin = 24,
        .rowEnd = 28,
        .columnEnd = 28,
    };
    const auto rejectedCoarse = authority.queryCoarseSpawnGrid(
        REJECTED_COARSE, AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(rejectedCoarse.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedCoarse.failure()->code == GenerationFailureCode::QUEUE_FULL);

    const auto rejectedPage = authority.preparePage(
        finalPage(303, 0), AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT);
    REQUIRE(rejectedPage.status() == AuthorityStatus::DEFERRED);
    REQUIRE(rejectedPage.failure()->code == GenerationFailureCode::QUEUE_FULL);

    constexpr NativeRect HIGH_TRANSIENT{
        .rowBegin = 64,
        .columnBegin = 64,
        .rowEnd = 72,
        .columnEnd = 72,
    };
    const auto highTransient = authority.queryTransientFinalNativeGrid(
        HIGH_TRANSIENT, AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(highTransient.status() == AuthorityStatus::DEFERRED);
    REQUIRE(highTransient.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);

    constexpr CoarseSpawnRegion HIGH_COARSE{
        .rowBegin = 32,
        .columnBegin = 32,
        .rowEnd = 36,
        .columnEnd = 36,
    };
    const auto highCoarse =
        authority.queryCoarseSpawnGrid(HIGH_COARSE, AuthorityRequestPriority::SPAWN);
    REQUIRE(highCoarse.status() == AuthorityStatus::DEFERRED);
    REQUIRE(highCoarse.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);

    for (int64_t index = 0; index < 14; ++index) {
        const auto admitted = authority.preparePage(finalPage(304, index),
                                                    AuthorityRequestPriority::EXPLORATION_EXACT);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }

    const auto full = authority.preparePage(finalPage(305, 0), AuthorityRequestPriority::SPAWN);
    REQUIRE(full.status() == AuthorityStatus::DEFERRED);
    REQUIRE(full.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.activeBuilds == 1);
    REQUIRE(metrics.queuedBuilds == MAXIMUM_AUTHORITY_QUEUED_REQUESTS - 1);
    REQUIRE(metrics.lowPriorityOutstandingRequests == MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS - 1);
    REQUIRE(metrics.visibleOrLowerOutstandingRequests ==
            MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS - 1);
    REQUIRE(metrics.lowPriorityDeferredRequests == 0);
    REQUIRE(metrics.visibleOrLowerDeferredRequests == 4);
    REQUIRE(metrics.deferredRequests == 4);
    REQUIRE(metrics.higherPriorityPreemptions == 1);
}

TEST_CASE("Current-player exact authority displaces queued distant work at capacity",
          "[learned][concurrency][priority][exact][preemption][regression]") {
    TempDir directory("learned_exact_priority_preemption");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 4;
    CachedTerrainAuthority authority(testIdentity(0xE0AC'7000'0000'0001ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'400, -1'400);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));
    for (int64_t index = 0; index < 3; ++index) {
        REQUIRE(authority
                    .preparePage(finalPage(540, index),
                                 AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT)
                    .status() == AuthorityStatus::DEFERRED);
    }

    const TerrainPageKey exact = finalPage(541, 0);
    const auto admitted = authority.preparePage(exact, AuthorityRequestPriority::EXPLORATION_EXACT);
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics saturated = authority.cacheMetrics();
    REQUIRE(saturated.activeBuilds == 1);
    REQUIRE(saturated.queuedBuilds == 3);
    REQUIRE(saturated.higherPriorityPreemptions == 1);
    REQUIRE(saturated.protectedHandoffPreemptions == 0);

    backend->releaseFirst();
    REQUIRE(awaitPage(authority, exact, AuthorityRequestPriority::EXPLORATION_EXACT).isReady());
    REQUIRE(backend->waitForCallCount(2, 2s));
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(calls[0].pages == std::vector<TerrainPageKey>{blocker});
    REQUIRE(calls[1].pages == std::vector<TerrainPageKey>{exact});
}

TEST_CASE("A current protected page preempts queued lower-priority authority at capacity",
          "[learned][concurrency][priority][handoff-epoch][page][regression]") {
    TempDir directory("learned_handoff_epoch_page_preemption");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 4;
    CachedTerrainAuthority authority(testIdentity(0xE001'0000'0000'0001ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'000, -1'000);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));

    for (int64_t index = 0; index < 3; ++index) {
        const auto queued = authority.preparePage(finalPage(500, index),
                                                  AuthorityRequestPriority::SPECULATIVE_PREFETCH);
        REQUIRE(queued.status() == AuthorityStatus::DEFERRED);
        REQUIRE(queued.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }
    REQUIRE(authority.cacheMetrics().queuedBuilds == 3);

    const TerrainPageKey current = finalPage(501, 0);
    const auto admitted = authority.preparePage(
        current, AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{10});
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics saturated = authority.cacheMetrics();
    REQUIRE(saturated.activeBuilds == 1);
    REQUIRE(saturated.queuedBuilds == 3);
    REQUIRE(saturated.lowPriorityOutstandingRequests == 3);
    REQUIRE(saturated.visibleOrLowerOutstandingRequests == 3);
    REQUIRE(saturated.currentProtectedHandoffEpoch == 10);
    REQUIRE(saturated.protectedHandoffPreemptions == 1);

    const size_t beforeStale = saturated.queuedBuilds;
    const auto stale = authority.preparePage(
        finalPage(502, 0), AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{9});
    REQUIRE(stale.status() == AuthorityStatus::DEFERRED);
    REQUIRE(stale.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().queuedBuilds == beforeStale);
    REQUIRE(authority.cacheMetrics().staleProtectedHandoffDeferrals == 1);

    backend->releaseFirst();
    REQUIRE(awaitAuthority([&] {
                return authority.preparePage(current, AuthorityRequestPriority::PROTECTED_HANDOFF,
                                             ProtectedHandoffEpoch{10});
            }).isReady());
    REQUIRE(backend->waitForCallCount(2, 2s));
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(calls[0].pages == std::vector<TerrainPageKey>{blocker});
    REQUIRE(calls[1].pages == std::vector<TerrainPageKey>{current});
}

TEST_CASE("A current protected page treats preview authority as a safe preemption victim",
          "[learned][concurrency][priority][handoff-epoch][page][preview][regression]") {
    TempDir directory("learned_handoff_epoch_preview_preemption");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 4;
    CachedTerrainAuthority authority(testIdentity(0xE001'0000'0000'0005ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'050, -1'050);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));

    const TerrainPageKey spawn = finalPage(505, 0);
    const TerrainPageKey exact = finalPage(505, 1);
    const TerrainPageKey preview = previewPage(505, 2);
    REQUIRE(authority.preparePage(spawn, AuthorityRequestPriority::SPAWN).status() ==
            AuthorityStatus::DEFERRED);
    REQUIRE(authority.preparePage(exact, AuthorityRequestPriority::EXPLORATION_EXACT).status() ==
            AuthorityStatus::DEFERRED);
    REQUIRE(authority.preparePage(preview, AuthorityRequestPriority::EXPLORATION_EXACT).status() ==
            AuthorityStatus::DEFERRED);

    const TerrainPageKey current = finalPage(506, 0);
    const auto admitted = authority.preparePage(
        current, AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{12});
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics saturated = authority.cacheMetrics();
    REQUIRE(saturated.activeBuilds == 1);
    REQUIRE(saturated.queuedBuilds == 3);
    REQUIRE(saturated.protectedHandoffPreemptions == 1);
    REQUIRE(saturated.currentProtectedHandoffEpoch == 12);

    backend->releaseFirst();
    REQUIRE(awaitAuthority([&] {
                return authority.preparePage(current, AuthorityRequestPriority::PROTECTED_HANDOFF,
                                             ProtectedHandoffEpoch{12});
            }).isReady());
    REQUIRE(backend->waitForCallCount(4, 2s));
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(std::ranges::none_of(calls, [&](const auto& call) {
        return call.pages == std::vector<TerrainPageKey>{preview};
    }));
    REQUIRE(calls[1].pages == std::vector<TerrainPageKey>{spawn});
    REQUIRE(calls[2].pages == std::vector<TerrainPageKey>{exact});
    REQUIRE(calls[3].pages == std::vector<TerrainPageKey>{current});
}

TEST_CASE("A current protected transient request replaces only a stale protected flight",
          "[learned][concurrency][priority][handoff-epoch][transient][regression]") {
    TempDir directory("learned_handoff_epoch_transient_preemption");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 4;
    CachedTerrainAuthority authority(testIdentity(0xE001'0000'0000'0002ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'100, -1'100);
    const TerrainPageKey spawn = finalPage(510, 0);
    const TerrainPageKey exact = finalPage(510, 1);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));
    REQUIRE(authority.preparePage(spawn, AuthorityRequestPriority::SPAWN).status() ==
            AuthorityStatus::DEFERRED);
    REQUIRE(authority.preparePage(exact, AuthorityRequestPriority::EXPLORATION_EXACT).status() ==
            AuthorityStatus::DEFERRED);

    constexpr NativeRect OLD_REGION{.rowBegin = 0, .columnBegin = 0, .rowEnd = 2, .columnEnd = 2};
    constexpr NativeRect CURRENT_REGION{
        .rowBegin = 4,
        .columnBegin = 4,
        .rowEnd = 6,
        .columnEnd = 6,
    };
    REQUIRE(authority
                .queryTransientFinalNativeGrid(OLD_REGION,
                                               AuthorityRequestPriority::PROTECTED_HANDOFF,
                                               ProtectedHandoffEpoch{1})
                .status() == AuthorityStatus::DEFERRED);
    REQUIRE(authority.cacheMetrics().queuedBuilds == 3);

    const auto admitted = authority.queryTransientFinalNativeGrid(
        CURRENT_REGION, AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{2});
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics saturated = authority.cacheMetrics();
    REQUIRE(saturated.queuedBuilds == 3);
    REQUIRE(saturated.protectedHandoffPreemptions == 1);
    REQUIRE(saturated.currentProtectedHandoffEpoch == 2);

    const auto stale = authority.queryTransientFinalNativeGrid(
        OLD_REGION, AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{1});
    REQUIRE(stale.status() == AuthorityStatus::DEFERRED);
    REQUIRE(stale.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().queuedBuilds == 3);

    backend->releaseFirst();
    REQUIRE(awaitAuthority([&] {
                return authority.queryTransientFinalNativeGrid(
                    CURRENT_REGION, AuthorityRequestPriority::PROTECTED_HANDOFF,
                    ProtectedHandoffEpoch{2});
            }).isReady());
    REQUIRE(backend->waitForCallCount(4, 2s));
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(calls[1].pages == std::vector<TerrainPageKey>{spawn});
    REQUIRE(calls[2].pages == std::vector<TerrainPageKey>{exact});
    REQUIRE(calls[3].kind == HandoffEpochTrackingBackend::Kind::TransientGrid);
    REQUIRE(calls[3].region == CURRENT_REGION);
    REQUIRE(std::ranges::none_of(calls, [&](const auto& call) {
        return call.kind == HandoffEpochTrackingBackend::Kind::TransientGrid &&
               call.region == OLD_REGION;
    }));
}

TEST_CASE("A protected page closure preempts enough lower work atomically",
          "[learned][concurrency][priority][handoff-epoch][batch][regression]") {
    TempDir directory("learned_handoff_epoch_batch_preemption");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 6;
    CachedTerrainAuthority authority(testIdentity(0xE001'0000'0000'0003ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'200, -1'200);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));
    for (int64_t index = 0; index < 5; ++index) {
        REQUIRE(
            authority
                .preparePage(finalPage(520, index), AuthorityRequestPriority::SPECULATIVE_PREFETCH)
                .status() == AuthorityStatus::DEFERRED);
    }

    const std::array<TerrainPageKey, 3> closure{
        finalPage(521, 2),
        finalPage(521, 0),
        finalPage(521, 1),
    };
    const auto admitted = authority.preparePages(
        closure, AuthorityRequestPriority::PROTECTED_HANDOFF, ProtectedHandoffEpoch{4});
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    const TerrainAuthorityCacheMetrics saturated = authority.cacheMetrics();
    REQUIRE(saturated.activeBuilds == 1);
    REQUIRE(saturated.queuedBuilds == 5);
    REQUIRE(saturated.protectedHandoffPreemptions == closure.size());
    REQUIRE(saturated.currentProtectedHandoffEpoch == 4);
    REQUIRE(saturated.lowPriorityOutstandingRequests == 3);
    REQUIRE(saturated.visibleOrLowerOutstandingRequests == 3);

    backend->releaseFirst();
    REQUIRE(awaitAuthority([&] {
                return authority.preparePages(closure, AuthorityRequestPriority::PROTECTED_HANDOFF,
                                              ProtectedHandoffEpoch{4});
            }).isReady());
    REQUIRE(backend->waitForCallCount(2, 2s));
    std::vector<TerrainPageKey> canonical(closure.begin(), closure.end());
    std::sort(canonical.begin(), canonical.end());
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(calls[1].kind == HandoffEpochTrackingBackend::Kind::PageBatch);
    REQUIRE(calls[1].pages == canonical);
}

TEST_CASE("Continuous protected-center movement keeps stale authority bounded",
          "[learned][concurrency][priority][handoff-epoch][movement][regression]") {
    TempDir directory("learned_handoff_epoch_continuous_movement");
    auto backend = std::make_shared<HandoffEpochTrackingBackend>();
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 8;
    CachedTerrainAuthority authority(testIdentity(0xE001'0000'0000'0004ULL), directory.path(),
                                     backend, config);
    struct ReleaseOnExit {
        std::shared_ptr<HandoffEpochTrackingBackend> backend;
        ~ReleaseOnExit() { backend->releaseFirst(); }
    } releaseOnExit{backend};

    const TerrainPageKey blocker = finalPage(-1'300, -1'300);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForCallCount(1, 2s));

    constexpr uint64_t LAST_EPOCH = 32;
    for (uint64_t epoch = 1; epoch <= LAST_EPOCH; ++epoch) {
        const auto moved = authority.preparePage(finalPage(530, static_cast<int64_t>(epoch)),
                                                 AuthorityRequestPriority::PROTECTED_HANDOFF,
                                                 ProtectedHandoffEpoch{epoch});
        CAPTURE(epoch);
        REQUIRE(moved.status() == AuthorityStatus::DEFERRED);
        REQUIRE(moved.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
        const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
        REQUIRE(metrics.activeBuilds == 1);
        REQUIRE(metrics.queuedBuilds <= config.maximumOutstandingRequests - 1);
        REQUIRE(metrics.currentProtectedHandoffEpoch == epoch);
        REQUIRE(metrics.lowPriorityOutstandingRequests == 1);
        REQUIRE(metrics.visibleOrLowerOutstandingRequests == 1);
    }
    const TerrainAuthorityCacheMetrics moved = authority.cacheMetrics();
    REQUIRE(moved.queuedBuilds == config.maximumOutstandingRequests - 1);
    REQUIRE(moved.protectedHandoffPreemptions ==
            LAST_EPOCH - (config.maximumOutstandingRequests - 1));

    const size_t queuedBeforeStale = moved.queuedBuilds;
    const auto stale =
        authority.preparePage(finalPage(531, 0), AuthorityRequestPriority::PROTECTED_HANDOFF,
                              ProtectedHandoffEpoch{LAST_EPOCH - 1});
    REQUIRE(stale.status() == AuthorityStatus::DEFERRED);
    REQUIRE(stale.failure()->code == GenerationFailureCode::QUEUE_FULL);
    REQUIRE(authority.cacheMetrics().queuedBuilds == queuedBeforeStale);

    const TerrainPageKey current = finalPage(530, static_cast<int64_t>(LAST_EPOCH));
    backend->releaseFirst();
    REQUIRE(awaitAuthority([&] {
                return authority.preparePage(current, AuthorityRequestPriority::PROTECTED_HANDOFF,
                                             ProtectedHandoffEpoch{LAST_EPOCH});
            }).isReady());
    REQUIRE(backend->waitForCallCount(2, 2s));
    const std::vector<HandoffEpochTrackingBackend::Call> calls = backend->calls();
    REQUIRE(calls[1].pages == std::vector<TerrainPageKey>{current});
}

TEST_CASE("Learned terrain inference stays single-call across distinct queued pages",
          "[learned][concurrency]") {
    TempDir directory("learned_serial_inference");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<SerialTrackingTerrainBackend>(15ms);
    TerrainAuthorityCacheConfig config;
    config.maximumEntries = 16;
    config.byteBudget = 16ULL * 1'024 * 1'024;
    config.maximumOutstandingRequests = 64;
    config.maximumConcurrentBuilds = 1;
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);
    constexpr std::array<TerrainPageKey, 8> KEYS = {
        TerrainPageKey{AuthorityQuality::FINAL, {-2, 3}},
        TerrainPageKey{AuthorityQuality::FINAL, {1, -4}},
        TerrainPageKey{AuthorityQuality::FINAL, {0, 0}},
        TerrainPageKey{AuthorityQuality::PREVIEW, {-1, -1}},
        TerrainPageKey{AuthorityQuality::FINAL, {5, 2}},
        TerrainPageKey{AuthorityQuality::PREVIEW, {4, -3}},
        TerrainPageKey{AuthorityQuality::FINAL, {-6, 7}},
        TerrainPageKey{AuthorityQuality::FINAL, {8, -9}},
    };

    for (const TerrainPageKey key : KEYS) {
        REQUIRE(authority.preparePage(key).status() == AuthorityStatus::DEFERRED);
    }
    for (const TerrainPageKey key : KEYS)
        REQUIRE(awaitPage(authority, key).isReady());

    REQUIRE(backend->maximumActiveCalls() == 1);
    auto requested = backend->requestedKeys();
    REQUIRE(requested.size() == KEYS.size());
    std::ranges::sort(requested);
    auto expected = KEYS;
    std::ranges::sort(expected);
    REQUIRE(requested == std::vector<TerrainPageKey>(expected.begin(), expected.end()));
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.builds == KEYS.size());
    REQUIRE(metrics.activeBuilds == 0);
    REQUIRE(metrics.queuedBuilds == 0);
    REQUIRE(metrics.publicationWorkersStarted);
    REQUIRE(metrics.publicationWrites == KEYS.size());
    REQUIRE(metrics.activePublications == 0);
    REQUIRE(metrics.queuedPublications == 0);
    REQUIRE(metrics.peakConcurrentPublications >= 1);
    REQUIRE(metrics.bytes <= config.byteBudget);
}

TEST_CASE("Final spawn closures batch contiguous final pages without changing priority",
          "[learned][concurrency][spawn][batch][determinism]") {
    TempDir directory("learned_final_spawn_batch");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<BatchTrackingTerrainBackend>(60ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const TerrainPageKey blocker = finalPage(-99, -99);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    const auto activeDeadline = std::chrono::steady_clock::now() + 2s;
    while (backend->singletonCount() == 0 && std::chrono::steady_clock::now() < activeDeadline)
        std::this_thread::yield();
    REQUIRE(backend->singletonCount() == 1);

    const std::array<TerrainPageKey, 4> closure{
        finalPage(-3, 5),
        finalPage(-3, 6),
        finalPage(-2, 5),
        finalPage(-2, 6),
    };
    for (const TerrainPageKey key : closure) {
        REQUIRE(authority.preparePage(key, AuthorityRequestPriority::SPAWN).status() ==
                AuthorityStatus::DEFERRED);
    }
    REQUIRE(
        awaitPage(authority, blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).isReady());
    for (const TerrainPageKey key : closure)
        REQUIRE(awaitPage(authority, key, AuthorityRequestPriority::SPAWN).isReady());

    const std::vector<std::vector<TerrainPageKey>> batches = backend->batches();
    REQUIRE(batches == std::vector<std::vector<TerrainPageKey>>{
                           std::vector<TerrainPageKey>(closure.begin(), closure.end())});
    REQUIRE(backend->maximumActiveCalls() == 1);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.builds == closure.size() + 1);
    REQUIRE(metrics.batches == 1);
    REQUIRE(metrics.batchedPages == closure.size());
}

TEST_CASE("Preview horizon closures batch contiguous pages on one inference lane",
          "[learned][concurrency][preview][batch][performance][regression]") {
    TempDir directory("learned_preview_horizon_batch");
    const GenerationIdentity identity = testIdentity(0x5052'4556'4945'5709ULL);
    auto backend = std::make_shared<BatchTrackingTerrainBackend>(60ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const TerrainPageKey blocker = finalPage(99, -99);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    const auto activeDeadline = std::chrono::steady_clock::now() + 2s;
    while (backend->singletonCount() == 0 && std::chrono::steady_clock::now() < activeDeadline)
        std::this_thread::yield();
    REQUIRE(backend->singletonCount() == 1);

    const std::array<TerrainPageKey, 6> closure{{
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -5, .column = -3}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -5, .column = -2}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -5, .column = -1}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -4, .column = -3}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -4, .column = -2}},
        {.quality = AuthorityQuality::PREVIEW, .coordinate = {.row = -4, .column = -1}},
    }};
    for (const TerrainPageKey key : closure) {
        REQUIRE(authority.preparePage(key, AuthorityRequestPriority::COARSE_PREVIEW).status() ==
                AuthorityStatus::DEFERRED);
    }
    REQUIRE(
        awaitPage(authority, blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).isReady());
    for (const TerrainPageKey key : closure)
        REQUIRE(awaitPage(authority, key, AuthorityRequestPriority::COARSE_PREVIEW).isReady());

    REQUIRE(backend->batches() ==
            std::vector<std::vector<TerrainPageKey>>{
                std::vector<TerrainPageKey>(closure.begin(), closure.begin() + 4),
                std::vector<TerrainPageKey>(closure.begin() + 4, closure.end()),
            });
    REQUIRE(backend->maximumActiveCalls() == 1);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.batches == 2);
    REQUIRE(metrics.batchedPages == closure.size());
}

TEST_CASE("Protected final handoff batches contiguous pages on one inference lane",
          "[learned][concurrency][protected][batch][performance][regression]") {
    TempDir directory("learned_protected_final_batch");
    const GenerationIdentity identity = testIdentity(0xBA7C'4000'0001ULL);
    auto backend = std::make_shared<BatchTrackingTerrainBackend>(60ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const TerrainPageKey blocker = finalPage(-99, 99);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    const auto activeDeadline = std::chrono::steady_clock::now() + 2s;
    while (backend->singletonCount() == 0 && std::chrono::steady_clock::now() < activeDeadline)
        std::this_thread::yield();
    REQUIRE(backend->singletonCount() == 1);

    const std::array<TerrainPageKey, 4> closure{
        finalPage(7, -4),
        finalPage(7, -3),
        finalPage(8, -4),
        finalPage(8, -3),
    };
    for (const TerrainPageKey key : closure) {
        REQUIRE(authority.preparePage(key, AuthorityRequestPriority::PROTECTED_HANDOFF).status() ==
                AuthorityStatus::DEFERRED);
    }
    REQUIRE(
        awaitPage(authority, blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).isReady());
    for (const TerrainPageKey key : closure)
        REQUIRE(awaitPage(authority, key, AuthorityRequestPriority::PROTECTED_HANDOFF).isReady());

    REQUIRE(backend->batches() == std::vector<std::vector<TerrainPageKey>>{
                                      std::vector<TerrainPageKey>(closure.begin(), closure.end())});
    REQUIRE(backend->maximumActiveCalls() == 1);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.batches == 1);
    REQUIRE(metrics.batchedPages == closure.size());
}

TEST_CASE("The page coordinator reconsiders priority after each four-page group",
          "[learned][concurrency][priority][batch][regression]") {
    TempDir directory("learned_batch_priority_reconsideration");
    const GenerationIdentity identity = testIdentity(0xBA7C'4000'0002ULL);
    auto backend = std::make_shared<PriorityReconsiderationTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    struct ReleaseBackendOnExit {
        std::shared_ptr<PriorityReconsiderationTerrainBackend> backend;
        ~ReleaseBackendOnExit() { backend->releaseAll(); }
    } releaseBackendOnExit{backend};

    const TerrainPageKey blocker = finalPage(-70, -70);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);
    REQUIRE(backend->waitForInitialPage(2s));

    const std::array<TerrainPageKey, 8> refinement{
        finalPage(70, 0), finalPage(70, 1), finalPage(70, 2), finalPage(70, 3),
        finalPage(70, 4), finalPage(70, 5), finalPage(70, 6), finalPage(70, 7),
    };
    for (const TerrainPageKey key : refinement) {
        const auto admitted = authority.preparePage(key, AuthorityRequestPriority::COARSE_PREVIEW);
        REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
        REQUIRE(admitted.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    }

    backend->releaseInitialPage();
    REQUIRE(backend->waitForFirstBatch(2s));
    const std::vector<std::vector<TerrainPageKey>> callsBeforeUrgent = backend->calls();
    REQUIRE(callsBeforeUrgent.size() == 2);
    REQUIRE(callsBeforeUrgent[0] == std::vector<TerrainPageKey>{blocker});
    REQUIRE(callsBeforeUrgent[1] ==
            std::vector<TerrainPageKey>(refinement.begin(), refinement.begin() + 4));

    const TerrainPageKey urgent = finalPage(-69, -69);
    const auto urgentAdmission = authority.preparePage(urgent, AuthorityRequestPriority::SPAWN);
    REQUIRE(urgentAdmission.status() == AuthorityStatus::DEFERRED);
    REQUIRE(urgentAdmission.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    backend->releaseFirstBatch();

    REQUIRE(
        awaitPage(authority, blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).isReady());
    REQUIRE(awaitPage(authority, urgent, AuthorityRequestPriority::SPAWN).isReady());
    for (const TerrainPageKey key : refinement)
        REQUIRE(awaitPage(authority, key, AuthorityRequestPriority::COARSE_PREVIEW).isReady());

    const std::vector<std::vector<TerrainPageKey>> calls = backend->calls();
    REQUIRE(calls == std::vector<std::vector<TerrainPageKey>>{
                         {blocker},
                         std::vector<TerrainPageKey>(refinement.begin(), refinement.begin() + 4),
                         {urgent},
                         std::vector<TerrainPageKey>(refinement.begin() + 4, refinement.end()),
                     });
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.batches == 2);
    REQUIRE(metrics.batchedPages == refinement.size());
}

TEST_CASE("Atomic final spawn admission preserves a hydrology closure across four-page groups",
          "[learned][concurrency][spawn][batch][hydrology][regression]") {
    TempDir directory("learned_atomic_hydrology_spawn_batch");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<BatchTrackingTerrainBackend>(0ms);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);

    std::vector<TerrainPageKey> expected;
    for (int64_t row = -1; row <= 2; ++row) {
        for (int64_t column = -1; column <= 2; ++column)
            expected.push_back(finalPage(row, column));
    }
    std::vector<TerrainPageCoordinate> submitted;
    submitted.reserve(expected.size());
    for (const TerrainPageKey key : expected)
        submitted.push_back(key.coordinate);
    std::reverse(submitted.begin(), submitted.end());

    const AuthorityResult<bool> admitted =
        context->requestAuthorityPages(submitted, AuthorityRequestPriority::SPAWN);
    REQUIRE(admitted.status() == AuthorityStatus::DEFERRED);
    for (const TerrainPageKey key : expected)
        REQUIRE(awaitPage(*authority, key, AuthorityRequestPriority::SPAWN).isReady());

    std::vector<std::vector<TerrainPageKey>> expectedBatches;
    for (size_t begin = 0; begin < expected.size(); begin += 4) {
        const size_t end = std::min(expected.size(), begin + 4);
        expectedBatches.emplace_back(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                                     expected.begin() + static_cast<std::ptrdiff_t>(end));
    }
    REQUIRE(backend->singletonCount() == 0);
    REQUIRE(backend->batches() == expectedBatches);
    REQUIRE(backend->maximumActiveCalls() == 1);
    const TerrainAuthorityCacheMetrics metrics = authority->cacheMetrics();
    REQUIRE(metrics.builds == expected.size());
    REQUIRE(metrics.batches == expectedBatches.size());
    REQUIRE(metrics.batchedPages == expected.size());
}

TEST_CASE("Atomic final spawn admission rejects an oversized closure without partial work",
          "[learned][concurrency][spawn][batch][queue]") {
    TempDir directory("learned_atomic_hydrology_spawn_queue");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<BatchTrackingTerrainBackend>(0ms);
    TerrainAuthorityCacheConfig config;
    config.maximumOutstandingRequests = 15;
    CachedTerrainAuthority authority(identity, directory.path(), backend, config);

    std::vector<TerrainPageKey> closure;
    for (int64_t row = -1; row <= 2; ++row) {
        for (int64_t column = -1; column <= 2; ++column)
            closure.push_back(finalPage(row, column));
    }
    const AuthorityResult<bool> result =
        authority.preparePages(closure, AuthorityRequestPriority::SPAWN);
    REQUIRE(result.status() == AuthorityStatus::DEFERRED);
    REQUIRE(result.failure());
    REQUIRE(result.failure()->code == GenerationFailureCode::QUEUE_FULL);
    const TerrainAuthorityCacheMetrics metrics = authority.cacheMetrics();
    REQUIRE(metrics.queuedBuilds == 0);
    REQUIRE(metrics.builds == 0);
    REQUIRE(metrics.deferredRequests == 1);
    REQUIRE(backend->singletonCount() == 0);
    REQUIRE(backend->batches().empty());
}

TEST_CASE("Learned terrain coordinator services fixed priority lanes off caller threads",
          "[learned][concurrency][priority]") {
    TempDir directory("learned_priority_coordinator");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<SerialTrackingTerrainBackend>(50ms);
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const std::thread::id callerThread = std::this_thread::get_id();
    const TerrainPageKey blocker = finalPage(99, 99);
    REQUIRE(
        authority.preparePage(blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).status() ==
        AuthorityStatus::DEFERRED);

    const auto activeDeadline = std::chrono::steady_clock::now() + 2s;
    while (backend->requestedKeys().empty() && std::chrono::steady_clock::now() < activeDeadline)
        std::this_thread::yield();
    REQUIRE(backend->requestedKeys() == std::vector<TerrainPageKey>{blocker});

    struct Request {
        TerrainPageKey key;
        AuthorityRequestPriority priority;
    };
    const std::array requests{
        Request{finalPage(5, 0), AuthorityRequestPriority::SPECULATIVE_PREFETCH},
        Request{finalPage(4, 0), AuthorityRequestPriority::COARSE_PREVIEW},
        Request{finalPage(3, 0), AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT},
        Request{finalPage(2, 0), AuthorityRequestPriority::PROTECTED_HANDOFF},
        Request{finalPage(1, 0), AuthorityRequestPriority::EXPLORATION_EXACT},
        Request{finalPage(0, 0), AuthorityRequestPriority::SPAWN},
    };
    for (const Request& request : requests) {
        REQUIRE(authority.preparePage(request.key, request.priority).status() ==
                AuthorityStatus::DEFERRED);
    }

    REQUIRE(
        awaitPage(authority, blocker, AuthorityRequestPriority::SPECULATIVE_PREFETCH).isReady());
    for (const Request& request : requests)
        REQUIRE(awaitPage(authority, request.key, request.priority).isReady());

    const std::vector<TerrainPageKey> expected{
        blocker,         requests[5].key, requests[4].key, requests[3].key,
        requests[2].key, requests[1].key, requests[0].key,
    };
    REQUIRE(backend->requestedKeys() == expected);
    const std::vector<std::thread::id> requestThreads = backend->requestThreads();
    REQUIRE(requestThreads.size() == expected.size());
    REQUIRE(std::ranges::all_of(requestThreads,
                                [&](std::thread::id id) { return id != callerThread; }));
    REQUIRE(std::ranges::all_of(requestThreads,
                                [&](std::thread::id id) { return id == requestThreads.front(); }));
    REQUIRE(backend->maximumActiveCalls() == 1);
}

TEST_CASE("Safe spawn prewarm uses a bounded lane without reclassifying exploration",
          "[learned][startup][priority]") {
    TempDir directory("learned_spawn_prewarm");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>(20ms);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    WorldGenerationContext context(identity, authority, AuthorityQuality::FINAL);
    REQUIRE(context.requestPriority() == AuthorityRequestPriority::EXPLORATION_EXACT);

    auto requested = context.requestWorldPage(-1, -1, AuthorityRequestPriority::SPAWN);
    REQUIRE(requested.status() == AuthorityStatus::DEFERRED);
    REQUIRE(context.requestPriority() == AuthorityRequestPriority::EXPLORATION_EXACT);
    REQUIRE_FALSE(context.failure());
    auto coldExploration = context.sampleWorld(-1, -1);
    REQUIRE(coldExploration.status() == AuthorityStatus::DEFERRED);
    REQUIRE_FALSE(context.failure());
    REQUIRE(awaitAuthority([&] {
                return context.requestWorldPage(-1, -1, AuthorityRequestPriority::SPAWN);
            }).isReady());
    REQUIRE(context.sampleWorld(-1, -1).isReady());
    REQUIRE(context.requestPriority() == AuthorityRequestPriority::EXPLORATION_EXACT);
    REQUIRE(backend->callCount() == 4);
}

TEST_CASE("Learned terrain page hashes survive reverse requests and restart",
          "[learned][determinism][persistence]") {
    TempDir directory("learned_reverse_hashes");
    const GenerationIdentity identity = testIdentity(0xCAFE'BEEF'0123'4567ULL);
    constexpr std::array<TerrainPageKey, 4> KEYS = {
        TerrainPageKey{AuthorityQuality::FINAL, {-3, 2}},
        TerrainPageKey{AuthorityQuality::FINAL, {0, -1}},
        TerrainPageKey{AuthorityQuality::FINAL, {2, 4}},
        TerrainPageKey{AuthorityQuality::FINAL, {-1, -2}},
    };
    std::array<Sha256Digest, KEYS.size()> forwardHashes{};
    {
        auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
        CachedTerrainAuthority authority(identity, directory.path(), backend);
        for (size_t index = 0; index < KEYS.size(); ++index) {
            auto page = awaitPage(authority, KEYS[index]);
            REQUIRE(page.isReady());
            forwardHashes[index] = quantizedPayloadHash(**page.value());
        }
        REQUIRE(backend->callCount() == KEYS.size());
    }

    auto restartBackend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority restarted(identity, directory.path(), restartBackend);
    for (size_t reverseIndex = KEYS.size(); reverseIndex > 0; --reverseIndex) {
        const size_t index = reverseIndex - 1;
        auto page = awaitPage(restarted, KEYS[index]);
        REQUIRE(page.isReady());
        REQUIRE(quantizedPayloadHash(**page.value()) == forwardHashes[index]);
    }
    REQUIRE(restartBackend->callCount() == 0);
    REQUIRE(restarted.cacheMetrics().diskLoads == KEYS.size());

    constexpr std::array<NativePoint, 4> SHUFFLED_POINTS = {
        NativePoint{.row = 2 * AUTHORITY_PAGE_NATIVE_EDGE + 17,
                    .column = 4 * AUTHORITY_PAGE_NATIVE_EDGE + 5},
        NativePoint{.row = -3 * AUTHORITY_PAGE_NATIVE_EDGE + 1,
                    .column = 2 * AUTHORITY_PAGE_NATIVE_EDGE + 250},
        NativePoint{.row = 1, .column = -1},
        NativePoint{.row = -AUTHORITY_PAGE_NATIVE_EDGE + 40,
                    .column = -2 * AUTHORITY_PAGE_NATIVE_EDGE + 90},
    };
    auto first = restarted.queryNativePoints(SHUFFLED_POINTS, AuthorityQuality::FINAL);
    REQUIRE(first.isReady());
    const std::array reversePoints{SHUFFLED_POINTS[3], SHUFFLED_POINTS[2], SHUFFLED_POINTS[1],
                                   SHUFFLED_POINTS[0]};
    auto reverse = restarted.queryNativePoints(reversePoints, AuthorityQuality::FINAL);
    REQUIRE(reverse.isReady());
    for (size_t index = 0; index < SHUFFLED_POINTS.size(); ++index) {
        REQUIRE((*first.value())[index] == (*reverse.value())[SHUFFLED_POINTS.size() - 1 - index]);
    }
    REQUIRE(restartBackend->callCount() == 0);
}

TEST_CASE("Completed learned page lookups stay inside the fixed-thread budget",
          "[learned][performance]") {
    TempDir directory("learned_lookup_performance");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);
    const TerrainPageKey key = finalPage(0, 0);
    REQUIRE(awaitPage(authority, key).isReady());
    const uint64_t hitsBefore = authority.cacheMetrics().hits;

    constexpr size_t SAMPLE_COUNT = 256;
    std::array<std::chrono::nanoseconds, SAMPLE_COUNT> durations{};
    for (std::chrono::nanoseconds& duration : durations) {
        const auto start = std::chrono::steady_clock::now();
        const auto page = authority.preparePage(key);
        duration = std::chrono::steady_clock::now() - start;
        REQUIRE(page.isReady());
    }
    std::ranges::sort(durations);
    constexpr size_t P95_INDEX = (SAMPLE_COUNT * 95 + 99) / 100 - 1;
    CAPTURE(std::chrono::duration_cast<std::chrono::microseconds>(durations[P95_INDEX]).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(durations.back()).count());
    REQUIRE(durations[P95_INDEX] <= 250us);
    REQUIRE(durations.back() <= 1ms);
    REQUIRE(backend->callCount() == 1);
    REQUIRE(authority.cacheMetrics().hits == hitsBefore + SAMPLE_COUNT);
}

TEST_CASE("Learned terrain uses scale-four row-column coordinates and physical units",
          "[learned][v4]") {
    REQUIRE(worldBlockToNative(0, 0) == NativePoint{.row = 0, .column = 0});
    REQUIRE(worldBlockToNative(3, 3) == NativePoint{.row = 0, .column = 0});
    REQUIRE(worldBlockToNative(4, 7) == NativePoint{.row = 1, .column = 1});
    REQUIRE(worldBlockToNative(-1, -1) == NativePoint{.row = -1, .column = -1});
    REQUIRE(worldBlockToNative(-4, -4) == NativePoint{.row = -1, .column = -1});
    REQUIRE(worldBlockToNative(-5, -8) == NativePoint{.row = -2, .column = -2});
    REQUIRE(worldBlockToNative(20, -12) == NativePoint{.row = -3, .column = 5});

    REQUIRE(learnedElevationMetersToWorldHeight(0.0) == 64.0);
    REQUIRE(learnedElevationMetersToWorldHeight(7.49) == 64.0);
    REQUIRE(learnedElevationMetersToWorldHeight(7.5) == 65.0);
    REQUIRE(learnedElevationMetersToWorldHeight(15.0) == 66.0);
    REQUIRE(learnedElevationMetersToWorldHeight(-1.0) == 63.0);
    REQUIRE(learnedElevationMetersToWorldHeight(-90.0) == 57.0);

    const PhysicalTerrainSample physical = dequantizeTerrainSample({
        .elevationMeters = -321,
        .meanTemperatureCentidegrees = 1'234,
        .temperatureVariabilityCentidegrees = 567,
        .annualPrecipitationMillimeters = 2'345,
        .precipitationCoefficientBasisPoints = 4'321,
        .lapseRateMicrodegreesPerMeter = -6'500,
    });
    REQUIRE(physical.elevationMeters == -321.0);
    REQUIRE(physical.meanTemperatureC == Catch::Approx(12.34));
    REQUIRE(physical.temperatureVariabilityC == Catch::Approx(5.67));
    REQUIRE(physical.annualPrecipitationMm == 2'345.0);
    REQUIRE(physical.precipitationCoefficientOfVariation == Catch::Approx(0.4321));
    REQUIRE(physical.lapseRateCPerMeter == Catch::Approx(-0.0065));
}

TEST_CASE("Bounded physical queries are seamless and request-order independent", "[learned][v4]") {
    TempDir directory("learned_physical_query");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    CachedTerrainAuthority authority(identity, directory.path(), backend);

    const NativeRect crossing{
        .rowBegin = -1,
        .columnBegin = 255,
        .rowEnd = 2,
        .columnEnd = 258,
    };
    auto grid =
        awaitAuthority([&] { return authority.queryNative(crossing, AuthorityQuality::FINAL); });
    REQUIRE(grid.isReady());
    REQUIRE(grid.value()->valid());
    REQUIRE(grid.value()->samples.size() == 9);
    REQUIRE(backend->callCount() == 4);

    const std::array forward{
        NativePoint{.row = -1, .column = 255},
        NativePoint{.row = 0, .column = 256},
        NativePoint{.row = 1, .column = 257},
        NativePoint{.row = 0, .column = 255},
    };
    auto forwardResult = authority.queryNativePoints(forward, AuthorityQuality::FINAL);
    REQUIRE(forwardResult.isReady());
    const std::array reverse{forward[3], forward[2], forward[1], forward[0]};
    auto reverseResult = authority.queryNativePoints(reverse, AuthorityQuality::FINAL);
    REQUIRE(reverseResult.isReady());
    REQUIRE(backend->callCount() == 4);
    for (size_t index = 0; index < forward.size(); ++index) {
        REQUIRE((*forwardResult.value())[index] ==
                (*reverseResult.value())[forward.size() - 1 - index]);
        REQUIRE(grid.value()->sample(forward[index].row, forward[index].column) != nullptr);
        REQUIRE(*grid.value()->sample(forward[index].row, forward[index].column) ==
                (*forwardResult.value())[index]);
    }

    TerrainAuthorityCacheConfig bounded;
    bounded.maximumQueryPages = 1;
    bounded.maximumQuerySamples = 4;
    CachedTerrainAuthority small(identity, std::filesystem::path(directory.path()) / "bounded",
                                 backend, bounded);
    auto tooManyPages = small.queryNative(crossing, AuthorityQuality::FINAL);
    REQUIRE(tooManyPages.status() == AuthorityStatus::FAILED);
    REQUIRE(tooManyPages.failure()->code == GenerationFailureCode::INVALID_REQUEST);
    const std::array<NativePoint, 5> tooManyPoints{};
    auto tooManySamples = small.queryNativePoints(tooManyPoints, AuthorityQuality::FINAL);
    REQUIRE(tooManySamples.status() == AuthorityStatus::FAILED);
    REQUIRE(tooManySamples.failure()->code == GenerationFailureCode::INVALID_REQUEST);
}

TEST_CASE("World generation context owns quality fingerprint metrics and failure state",
          "[learned][v4]") {
    TempDir directory("learned_context");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    const std::filesystem::path hydrologyRoot =
        std::filesystem::path(directory.path()) / "hydrology-authority-v1";
    WorldGenerationContext context(identity, authority, AuthorityQuality::PREVIEW, hydrologyRoot);

    const std::array points{
        WorldBlockPoint{.x = 0, .z = 0},
        WorldBlockPoint{.x = 3, .z = 3},
        WorldBlockPoint{.x = -1, .z = -1},
        WorldBlockPoint{.x = -4, .z = -4},
    };
    auto samples = awaitAuthority([&] { return context.queryWorldPoints(points); });
    REQUIRE(samples.isReady());
    const std::array nativeStencil{
        NativePoint{.row = -1, .column = -1},
        NativePoint{.row = -1, .column = 0},
        NativePoint{.row = 0, .column = -1},
        NativePoint{.row = 0, .column = 0},
    };
    auto native = awaitAuthority([&] { return context.queryNativePoints(nativeStencil); });
    REQUIRE(native.isReady());
    const auto bilinear = [&](double rowFraction, double columnFraction, auto member) {
        const double north =
            std::lerp((*native.value())[0].*member, (*native.value())[1].*member, columnFraction);
        const double south =
            std::lerp((*native.value())[2].*member, (*native.value())[3].*member, columnFraction);
        return std::lerp(north, south, rowFraction);
    };
    REQUIRE((*samples.value())[0].elevationMeters ==
            Catch::Approx(bilinear(0.625, 0.625, &PhysicalTerrainSample::elevationMeters)));
    REQUIRE((*samples.value())[0].meanTemperatureC ==
            Catch::Approx(bilinear(0.625, 0.625, &PhysicalTerrainSample::meanTemperatureC)));
    REQUIRE((*samples.value())[0] != (*samples.value())[1]);
    REQUIRE((*samples.value())[2] != (*samples.value())[3]);
    REQUIRE(context.identity() == identity);
    REQUIRE(context.fingerprint() == identity.fingerprint());
    REQUIRE(context.quality() == AuthorityQuality::PREVIEW);
    REQUIRE(context.requestPriority() == AuthorityRequestPriority::COARSE_PREVIEW);
    REQUIRE(context.hydrologyAuthorityRoot() == hydrologyRoot);
    REQUIRE(context.nativeHydrologyRouter());
    REQUIRE(context.nativeHydrologyIdentityRegistry());
    REQUIRE_FALSE(context.failure());
    const WorldGenerationMetrics metrics = context.metrics();
    REQUIRE(metrics.quality == AuthorityQuality::PREVIEW);
    REQUIRE(metrics.generationFingerprint == identity.fingerprint());
    REQUIRE(metrics.queries >= 2);
    REQUIRE(metrics.requestedSamples == metrics.queries * points.size());
    REQUIRE(metrics.readyQueries == 2);
    REQUIRE(metrics.deferredQueries == metrics.queries - metrics.readyQueries);

    const std::shared_ptr<WorldGenerationContext> finalView =
        context.withQuality(AuthorityQuality::FINAL);
    REQUIRE(finalView->identity() == identity);
    REQUIRE(finalView->quality() == AuthorityQuality::FINAL);
    REQUIRE(finalView->requestPriority() == AuthorityRequestPriority::EXPLORATION_EXACT);
    REQUIRE(finalView->hydrologyAuthorityRoot() == hydrologyRoot);
    REQUIRE(finalView->nativeHydrologyRouter());
    REQUIRE(finalView->nativeHydrologyRouter() != context.nativeHydrologyRouter());
    REQUIRE(finalView->nativeHydrologyIdentityRegistry() ==
            context.nativeHydrologyIdentityRegistry());
    REQUIRE(awaitAuthority([&] { return finalView->queryWorldPoints(points); }).isReady());
    const auto handoffView =
        finalView->withRequestPriority(AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(handoffView->requestPriority() == AuthorityRequestPriority::PROTECTED_HANDOFF);
    REQUIRE(handoffView->hydrologyAuthorityRoot() == hydrologyRoot);
    REQUIRE(handoffView->nativeHydrologyRouter() == finalView->nativeHydrologyRouter());
    REQUIRE(handoffView->nativeHydrologyIdentityRegistry() ==
            finalView->nativeHydrologyIdentityRegistry());
    REQUIRE(finalView->withQuality(AuthorityQuality::FINAL)->nativeHydrologyRouter() ==
            finalView->nativeHydrologyRouter());
    REQUIRE(handoffView->failure() == finalView->failure());
    context.latchFailure({.code = GenerationFailureCode::INFERENCE_FAILED,
                          .message = "Synthetic shared-view failure",
                          .retriable = true});
    REQUIRE(finalView->failure());
    REQUIRE(handoffView->failure() == finalView->failure());
    REQUIRE(finalView->sampleWorld(0, 0).status() == AuthorityStatus::FAILED);
    REQUIRE(finalView->clearRetriableFailure());
    REQUIRE_FALSE(context.failure());
    REQUIRE_FALSE(finalView->failure());

    GenerationIdentity incompatible = identity;
    ++incompatible.seed;
    WorldGenerationContext rejected(incompatible, authority, AuthorityQuality::FINAL);
    REQUIRE(rejected.failure());
    REQUIRE(rejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    auto failed = rejected.sampleWorld(0, 0);
    REQUIRE(failed.status() == AuthorityStatus::FAILED);
    REQUIRE(failed.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
}

TEST_CASE("V4 macro terrain and climate come only from learned authority", "[learned][v4]") {
    TempDir directory("learned_macro");
    constexpr uint64_t SEED = 0xFEDC'BA98'7654'3210ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    constexpr int64_t WORLD_X = -5;
    constexpr int64_t WORLD_Z = 1'027;
    auto learned = awaitAuthority([&] { return context->sampleWorld(WORLD_X, WORLD_Z); });
    REQUIRE(learned.isReady());
    const double baseHeight = learnedElevationMetersToWorldHeight(learned.value()->elevationMeters);
    REQUIRE(macro.preliminaryElevation(WORLD_X, WORLD_Z) == baseHeight);
    auto neighbor = awaitAuthority([&] { return context->sampleWorld(WORLD_X - 2, WORLD_Z - 2); });
    REQUIRE(neighbor.isReady());
    const double neighborHeight =
        learnedElevationMetersToWorldHeight(neighbor.value()->elevationMeters);
    REQUIRE(macro.preliminaryElevation(WORLD_X - 2, WORLD_Z - 2) == neighborHeight);

    const worldgen::ClimateFields climate = macro.sampleClimate(WORLD_X, WORLD_Z, baseHeight);
    REQUIRE(climate.temperatureC == Catch::Approx(learned.value()->meanTemperatureC));
    REQUIRE(climate.temperatureVariabilityC ==
            Catch::Approx(learned.value()->temperatureVariabilityC));
    REQUIRE(climate.annualPrecipitationMm == Catch::Approx(learned.value()->annualPrecipitationMm));
    REQUIRE(climate.precipitationCoefficientOfVariation ==
            Catch::Approx(learned.value()->precipitationCoefficientOfVariation));
    REQUIRE(climate.lapseRateCPerMeter == Catch::Approx(learned.value()->lapseRateCPerMeter));
    const worldgen::ClimateFields raised = macro.sampleClimate(WORLD_X, WORLD_Z, baseHeight + 10.0);
    REQUIRE(raised.temperatureC == Catch::Approx(learned.value()->meanTemperatureC +
                                                 learned.value()->lapseRateCPerMeter * 75.0));

    std::array<worldgen::SurfaceSample, 9> weatherClimate{};
    awaitGeneration([&] { macro.sampleClimateGrid(-256, 768, 256, 3, weatherClimate); });
    for (const worldgen::SurfaceSample& sample : weatherClimate) {
        REQUIRE(std::isfinite(sample.terrainHeight));
        REQUIRE(std::isfinite(sample.climate.temperatureC));
        REQUIRE(std::isfinite(sample.climate.annualPrecipitationMm));
    }
    REQUIRE(macro.nativeHydrologyCacheMetrics().builds == 0);
    REQUIRE(context->preparedNativeHydrologyOwnerCount() == 0);

    std::optional<ColumnPos> activeUplift;
    for (int64_t z = -32'768; z <= 32'768 && !activeUplift; z += 2'048) {
        for (int64_t x = -32'768; x <= 32'768; x += 2'048) {
            if (macro.sampleGeology(x, z).uplift <= 0.75)
                continue;
            activeUplift = ColumnPos{x, z};
            break;
        }
    }
    REQUIRE(activeUplift.has_value());
    auto upliftAuthority =
        awaitAuthority([&] { return context->sampleWorld(activeUplift->x, activeUplift->z); });
    REQUIRE(upliftAuthority.isReady());
    const double upliftAuthorityHeight =
        learnedElevationMetersToWorldHeight(upliftAuthority.value()->elevationMeters);
    REQUIRE(macro.preliminaryElevation(activeUplift->x, activeUplift->z) == upliftAuthorityHeight);

    worldgen::MacroGenerationSampler v3(SEED);
    worldgen::MacroGenerationSampler explicitV3(SEED, std::shared_ptr<WorldGenerationContext>{});
    REQUIRE(v3.preliminaryElevation(371.0, -812.0) ==
            explicitV3.preliminaryElevation(371.0, -812.0));
    REQUIRE(v3.sampleClimate(371.0, -812.0, 93.0).temperatureC ==
            explicitV3.sampleClimate(371.0, -812.0, 93.0).temperatureC);

    ChunkGenerator generator(SEED, context);
    REQUIRE(generator.seed() == SEED);
}

TEST_CASE("V4 retires legacy erosion while retaining bounded geology caves and ores",
          "[learned][v4][erosion][geology][cave][ore][regression]") {
    ClimateSampler caveMask(0x51A7'0004U);
    for (const ColumnPos position :
         std::array{ColumnPos{-257, 511}, ColumnPos{0, 0}, ColumnPos{1'024, -2'048}}) {
        REQUIRE(caveMask.caveEntrance(position.x, position.z) ==
                caveMask.shapeColumn(position.x, position.z).entrance);
    }

    TempDir directory("learned_without_legacy_erosion");
    constexpr uint64_t SEED = 0xE205'10A0'0004ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<PlanarTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);
    ChunkGenerator generator(SEED, context);

    constexpr int64_t X = 24;
    constexpr int64_t Z = -40;
    const PhysicalTerrainSample learned =
        *awaitAuthority([&] { return context->sampleWorld(X, Z); }).value();
    const double undisturbed = learnedElevationMetersToWorldHeight(learned.elevationMeters);
    const worldgen::HydrologySample hydrology =
        awaitGeneration([&] { return macro.sampleHydrology(X, Z); }, std::chrono::seconds(20));
    const worldgen::SurfaceSample surface = awaitGeneration(
        [&] { return macro.sampleSurface(X, Z, worldgen::SurfaceFootprint::BLOCK_1); },
        std::chrono::seconds(20));

    CAPTURE(undisturbed, hydrology.surfaceElevation, hydrology.erosionDepth, surface.terrainHeight,
            surface.geology.plateId, surface.geology.erosionResistance);
    REQUIRE(hydrology.surfaceElevation <= undisturbed + 1.0e-9);
    REQUIRE(hydrology.erosionDepth ==
            Catch::Approx(undisturbed - hydrology.surfaceElevation).margin(1.0e-9));
    REQUIRE_FALSE(hydrology.channelBank);
    REQUIRE_FALSE(hydrology.lakeBank);
    REQUIRE(surface.terrainHeight == Catch::Approx(hydrology.surfaceElevation).margin(1.0e-9));
    REQUIRE(surface.hydrology.surfaceElevation ==
            Catch::Approx(hydrology.surfaceElevation).margin(1.0e-9));
    REQUIRE(surface.climate.temperatureC == Catch::Approx(learned.meanTemperatureC));
    REQUIRE(macro.basinCacheMetrics().builds == 0);
    REQUIRE(macro.basinCacheMetrics().erosionEpochs == 0);
    REQUIRE(macro.nativeHydrologyCacheMetrics().builds > 0);

    // Synthetic plate fields remain bounded material and cubic-system inputs.
    // They may choose strata, cave character, and ore hosts, but the assertions
    // above ensure they cannot lift or erode the learned macro surface.
    REQUIRE(surface.geology.erosionResistance >= 0.35);
    REQUIRE(surface.geology.erosionResistance <= 1.25);
    REQUIRE(surface.geology.rock == worldgen::RockType::LIMESTONE);
    REQUIRE(std::isfinite(surface.geology.continentalFraction));
    REQUIRE(std::isfinite(surface.geology.distanceToBoundary));

    const auto isOre = [](BlockType block) {
        return block == BlockType::COAL_ORE || block == BlockType::IRON_ORE ||
               block == BlockType::GOLD_ORE || block == BlockType::DIAMOND_ORE;
    };
    bool foundCave = false;
    bool foundOre = false;
    bool foundHostRock = false;
    for (int chunkZ = -1; chunkZ <= 1 && !(foundCave && foundOre && foundHostRock); ++chunkZ) {
        for (int chunkX = -1; chunkX <= 1 && !(foundCave && foundOre && foundHostRock); ++chunkX) {
            Chunk cube(ChunkPos{chunkX, -2, chunkZ});
            awaitGeneration(
                [&] {
                    generator.generate(cube);
                    return true;
                },
                std::chrono::seconds(30));
            for (const BlockType block : cube.copyBlocks()) {
                foundCave = foundCave || block == BlockType::AIR;
                foundOre = foundOre || isOre(block);
                foundHostRock = foundHostRock || block == BlockType::STONE ||
                                block == BlockType::BASALT || block == BlockType::LIMESTONE ||
                                block == BlockType::SANDSTONE || block == BlockType::ANDESITE;
            }
        }
    }
    CAPTURE(foundCave, foundOre, foundHostRock);
    REQUIRE(foundCave);
    REQUIRE(foundOre);
    REQUIRE(foundHostRock);
}

TEST_CASE("V4 physical climate retains semiarid soil moisture for compatibility flora",
          "[learned][v4][ecology][flora][climate][regression]") {
    TempDir directory("learned_physical_climate_flora");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler v4(SEED, context);
    worldgen::MacroGenerationSampler v3(SEED);

    worldgen::GeologySample geology;
    geology.rock = worldgen::RockType::GRANITE;
    geology.distanceToBoundary = 1'024.0;
    worldgen::HydrologySample hydrology;
    hydrology.surfaceElevation = 139.0;
    hydrology.waterSurface = SEA_LEVEL;
    hydrology.channelDistance = 4'096.0;
    worldgen::ClimateFields climate;
    climate.temperatureC = 2.104219;
    climate.temperatureVariabilityC = 7.47;
    climate.annualPrecipitationMm = 302.921875;
    climate.precipitationCoefficientOfVariation = 0.940238;
    climate.potentialEvapotranspirationMm =
        std::clamp(300.0 + climate.temperatureC * 31.0 + climate.temperatureVariabilityC * 8.0,
                   120.0, 1'800.0);
    climate.relativeHumidity =
        climate.annualPrecipitationMm /
        (climate.annualPrecipitationMm + climate.potentialEvapotranspirationMm *
                                             (0.75 + climate.precipitationCoefficientOfVariation));
    climate.aridity = climate.potentialEvapotranspirationMm / climate.annualPrecipitationMm;

    const worldgen::SoilSample learned =
        v4.sampleSoil(5'228.0, -4'247.0, geology, hydrology, climate);
    const worldgen::SoilSample legacy =
        v3.sampleSoil(5'228.0, -4'247.0, geology, hydrology, climate);
    CAPTURE(learned.moisture, learned.fertility, learned.drainage, legacy.moisture,
            legacy.fertility, climate.aridity);
    REQUIRE(legacy.moisture == 0.0);
    REQUIRE(learned.moisture > 0.30);
    REQUIRE(learned.fertility > 0.25);

    worldgen::SurfaceSample surface;
    surface.geology = geology;
    surface.hydrology = hydrology;
    surface.climate = climate;
    surface.soil = learned;
    surface.terrainHeight = hydrology.surfaceElevation;
    surface.slope = 0.10;
    surface.suitability = v4.biomeSuitability(geology, hydrology, climate, learned,
                                              surface.terrainHeight, surface.slope);
    surface.biome = worldgen::MacroGenerationSampler::selectBiome(surface.suitability);
    REQUIRE(feature_generation::treeCoverDensity(surface) > 0.05);
    REQUIRE(worldgen::surface_material::supportsTreeRooting(BlockType::GRASS));

    const int groundY = static_cast<int>(std::ceil(worldgen::geometryTerrainHeight(surface))) - 1;
    bool viableTree = false;
    for (size_t index = 0; index < static_cast<size_t>(feature_generation::TreeSpecies::COUNT);
         ++index) {
        const auto habitat = feature_generation::evaluateTreeHabitat(
            static_cast<feature_generation::TreeSpecies>(index), surface, groundY);
        viableTree = viableTree || habitat.allowed;
    }
    REQUIRE(viableTree);
}

TEST_CASE("V4 learned habitat emits deterministic nonzero coarse flora",
          "[learned][v4][ecology][flora][canopy][regression]") {
    TempDir directory("learned_nonzero_coarse_flora");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<QualitySplitTerrainBackend>(750, 750));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);

    const std::vector<FarCanopy> canopies = awaitGeneration(
        [&] { return generator.collectFarCanopiesForLod(5'120, -4'352, 5'376, -4'096, 4); },
        std::chrono::seconds(30));
    const std::vector<FarCanopy> repeated =
        generator.collectFarCanopiesForLod(5'120, -4'352, 5'376, -4'096, 4);
    REQUIRE_FALSE(canopies.empty());
    REQUIRE(repeated == canopies);
    REQUIRE(std::ranges::all_of(canopies, [](const FarCanopy& canopy) {
        return canopy.anchorId != 0 && canopy.species != feature_generation::TreeSpecies::COUNT;
    }));
}

TEST_CASE("V4 preview ecology emits deterministic dry flora without exact density work",
          "[learned][v4][preview][ecology][flora][canopy][regression]") {
    TempDir directory("learned_preview_coarse_flora");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(),
        std::make_shared<QualitySplitTerrainBackend>(750, 750, true, 800));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::PREVIEW);
    ChunkGenerator generator(SEED, context);

    const uint64_t densityBefore = generator.densityEvaluationCount();
    const std::vector<FarCanopy> canopies = awaitGeneration(
        [&] { return generator.collectFarCanopiesForLod(5'120, -4'352, 5'376, -4'096, 32); },
        std::chrono::seconds(30));
    const std::vector<FarFlora> flora = awaitGeneration(
        [&] { return generator.collectFarFloraForLod(5'120, -4'352, 5'376, -4'096, 32); },
        std::chrono::seconds(30));

    CAPTURE(canopies.size(), flora.size());
    REQUIRE_FALSE(canopies.empty());
    REQUIRE_FALSE(flora.empty());
    REQUIRE(generator.collectFarCanopiesForLod(5'120, -4'352, 5'376, -4'096, 32) == canopies);
    REQUIRE(generator.collectFarFloraForLod(5'120, -4'352, 5'376, -4'096, 32) == flora);
    REQUIRE(generator.densityEvaluationCount() == densityBefore);
    REQUIRE(generator.cachedColumnPlanCount() == 0);
}

TEST_CASE("V4 learned habitat emits the same accepted trees in exact cubes",
          "[learned][v4][ecology][flora][tree][exact][regression]") {
    TempDir directory("learned_nonzero_exact_flora");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<QualitySplitTerrainBackend>(750, 750));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);

    const std::vector<FarCanopy> canopies =
        awaitGeneration([&] { return generator.collectFarCanopies(5'120, -4'352, 5'376, -4'096); },
                        std::chrono::seconds(30));
    const auto acceptedTree = std::ranges::find_if(canopies, [](const FarCanopy& canopy) {
        return canopy.logBlock != BlockType::AIR &&
               canopy.species != feature_generation::TreeSpecies::FALLEN_LOG;
    });
    REQUIRE(acceptedTree != canopies.end());

    const ChunkPos rootCubePosition{Chunk::worldToChunk(acceptedTree->x),
                                    Chunk::worldToChunkY(acceptedTree->baseY),
                                    Chunk::worldToChunk(acceptedTree->z)};
    Chunk rootCube(rootCubePosition);
    awaitGeneration(
        [&] {
            generator.generate(rootCube);
            return true;
        },
        std::chrono::seconds(30));
    REQUIRE(rootCube.getBlockWorld(acceptedTree->x, acceptedTree->baseY, acceptedTree->z) ==
            acceptedTree->logBlock);
    REQUIRE(
        std::ranges::any_of(rootCube.copyBlocks(), [](BlockType block) { return isFlora(block); }));
}

TEST_CASE("V4 coarse flora rejects canonical ocean roots",
          "[learned][v4][ecology][flora][water][shoreline][regression]") {
    TempDir directory("learned_coarse_flora_ocean_roots");
    constexpr uint64_t SEED = 0xEC01'0A74'0002ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>(-25));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);

    const std::vector<FarCanopy> canopies =
        awaitGeneration([&] { return generator.collectFarCanopiesForLod(0, 0, 256, 256, 4); },
                        std::chrono::seconds(20));
    REQUIRE(canopies.empty());
}

TEST_CASE("V4 preview ecology rejects every canonical surface water root",
          "[learned][v4][preview][ecology][flora][water][regression]") {
    worldgen::SurfaceSample surface;
    REQUIRE_FALSE(feature_generation::previewFarEcologyRejectsRoot(surface));

    surface.hydrology.ocean = true;
    REQUIRE(feature_generation::previewFarEcologyRejectsRoot(surface));
    surface.hydrology = {};
    surface.hydrology.river = true;
    REQUIRE(feature_generation::previewFarEcologyRejectsRoot(surface));
    surface.hydrology = {};
    surface.hydrology.lake = true;
    REQUIRE(feature_generation::previewFarEcologyRejectsRoot(surface));
    surface.hydrology = {};
    surface.hydrology.wetland = true;
    REQUIRE(feature_generation::previewFarEcologyRejectsRoot(surface));
    surface.hydrology = {};
    surface.hydrology.waterfall = true;
    surface.hydrology.transitionOwnerKind = worldgen::WaterTransitionKind::EXPLICIT_FALL;
    REQUIRE(feature_generation::previewFarEcologyRejectsRoot(surface));
}

TEST_CASE("V4 preview ecology rejects provisional ocean flora without exact density work",
          "[learned][v4][preview][ecology][flora][water][regression]") {
    TempDir directory("learned_preview_coarse_flora_ocean_roots");
    constexpr uint64_t SEED = 0xEC01'0A74'0003ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>(-25));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::PREVIEW);
    ChunkGenerator generator(SEED, context);

    const uint64_t densityBefore = generator.densityEvaluationCount();
    const std::vector<FarCanopy> canopies =
        awaitGeneration([&] { return generator.collectFarCanopiesForLod(0, 0, 256, 256, 32); },
                        std::chrono::seconds(20));
    const std::vector<FarFlora> flora =
        awaitGeneration([&] { return generator.collectFarFloraForLod(0, 0, 256, 256, 32); },
                        std::chrono::seconds(20));

    REQUIRE(canopies.empty());
    REQUIRE(flora.empty());
    REQUIRE(generator.densityEvaluationCount() == densityBefore);
    REQUIRE(generator.cachedColumnPlanCount() == 0);
}

TEST_CASE("V4 native hydrology admits its full learned closure before a deferred raster query",
          "[learned][v4][hydrology][authority][admission][regression]") {
    constexpr uint64_t SEED = 0xC105'F1E0'0004ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<ClosureRecordingDeferredTerrainAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    try {
        static_cast<void>(macro.sampleHydrology(1'024.0, 1'024.0));
        FAIL("Deferred learned hydrology unexpectedly completed");
    } catch (const GenerationFailureException& failure) {
        REQUIRE(failure.status() == AuthorityStatus::DEFERRED);
    }

    std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>> expected;
    for (int64_t row = -1; row <= 2; ++row) {
        for (int64_t column = -1; column <= 2; ++column) {
            expected.push_back({
                TerrainPageKey{
                    .quality = AuthorityQuality::FINAL,
                    .coordinate = {.row = row, .column = column},
                },
                AuthorityRequestPriority::EXPLORATION_EXACT,
            });
        }
    }

    REQUIRE(authority->rasterQueryCount() == 1);
    REQUIRE(authority->rasterPriority() == AuthorityRequestPriority::EXPLORATION_EXACT);
    REQUIRE(
        authority->batches() ==
        std::vector<std::vector<std::pair<TerrainPageKey, AuthorityRequestPriority>>>{expected});
    REQUIRE(authority->preparedAtFirstRasterQuery() == expected);
}

TEST_CASE("Retained authority pages require the requested key and complete identity",
          "[learned][context][page][validation]") {
    constexpr uint64_t SEED = 0x51E1'CA11'DA7AULL;
    const GenerationIdentity identity = testIdentity(SEED);
    const std::array cases{
        std::pair{RetainedPageFault::NULL_PAGE, GenerationFailureCode::CORRUPT_PAGE},
        std::pair{RetainedPageFault::INVALID_PAYLOAD, GenerationFailureCode::CORRUPT_PAGE},
        std::pair{RetainedPageFault::WRONG_QUALITY, GenerationFailureCode::CORRUPT_PAGE},
        std::pair{RetainedPageFault::WRONG_COORDINATE, GenerationFailureCode::CORRUPT_PAGE},
        std::pair{RetainedPageFault::WRONG_SEED, GenerationFailureCode::INCOMPATIBLE_FINGERPRINT},
        std::pair{RetainedPageFault::WRONG_FINGERPRINT,
                  GenerationFailureCode::INCOMPATIBLE_FINGERPRINT},
    };
    for (const auto [fault, expectedCode] : cases) {
        INFO("retained page fault=" << static_cast<int>(fault));
        auto authority = std::make_shared<EphemeralRetainedPageAuthority>(identity, fault);
        auto context =
            std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
        const auto retained = context->retainAuthorityPage({.row = 0, .column = 0});
        REQUIRE(retained.status() == AuthorityStatus::FAILED);
        REQUIRE(retained.failure() != nullptr);
        REQUIRE(retained.failure()->code == expectedCode);
        REQUIRE(context->failure().has_value());
        REQUIRE(context->failure()->code == expectedCode);
    }

    auto authority = std::make_shared<EphemeralRetainedPageAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    const auto retained = context->retainAuthorityPage({.row = -1, .column = 0});
    REQUIRE(retained.isReady());
    REQUIRE(retained.value() != nullptr);
    REQUIRE(*retained.value());
    REQUIRE((*retained.value())->key == finalPage(-1, 0));
    REQUIRE((*retained.value())->matches(identity));
    REQUIRE_FALSE(context->failure().has_value());
}

TEST_CASE("Exact learned controls copy a compact stencil and release authority pages",
          "[learned][v4][column-plan][stencil][cache][regression]") {
    constexpr uint64_t SEED = 0xC011'EC7A'570C'11ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<EphemeralRetainedPageAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);
    const worldgen::MacroControlView control = macro.controlView({0, 0});
    control.prepareLearnedAuthority();

    // Chunk zero needs four pages because its six-sample stencil crosses the
    // negative native boundary on both axes. None may remain pinned outside
    // the authority's reported cache bytes once the 432-byte copy is ready.
    REQUIRE(authority->observedPages().size() == 4);
    for (const auto& page : authority->observedPages())
        REQUIRE(page.expired());
    REQUIRE(context->metrics().authorityCache.bytes == 0);

    constexpr std::array<WorldBlockPoint, 6> POINTS{{
        {.x = 0, .z = 0},
        {.x = 1, .z = 3},
        {.x = 7, .z = 5},
        {.x = 11, .z = 14},
        {.x = 15, .z = 15},
        {.x = 16, .z = 16},
    }};
    for (const WorldBlockPoint point : POINTS) {
        const auto direct = context->sampleWorld(point.x, point.z);
        REQUIRE(direct.isReady());
        worldgen::SurfaceSample reconstructed;
        reconstructed.terrainHeight = 91.25;
        reconstructed.hydrology.terrainSlope = 0.375;
        const uint64_t beforeQueries = context->metrics().queries;
        control.reconstructContinuous(static_cast<double>(point.x), static_cast<double>(point.z),
                                      reconstructed);
        REQUIRE(context->metrics().queries == beforeQueries);

        const PhysicalTerrainSample& physical = *direct.value();
        const double nativeHeight = learnedElevationMetersToWorldHeight(physical.elevationMeters);
        const double expectedTemperature =
            physical.meanTemperatureC + physical.lapseRateCPerMeter *
                                            (reconstructed.terrainHeight - nativeHeight) *
                                            WORLD_METERS_PER_BLOCK;
        REQUIRE(reconstructed.climate.temperatureC == Catch::Approx(expectedTemperature));
        REQUIRE(reconstructed.climate.temperatureVariabilityC ==
                Catch::Approx(physical.temperatureVariabilityC));
        REQUIRE(reconstructed.climate.annualPrecipitationMm ==
                Catch::Approx(physical.annualPrecipitationMm));
        REQUIRE(reconstructed.climate.precipitationCoefficientOfVariation ==
                Catch::Approx(physical.precipitationCoefficientOfVariation));
        REQUIRE(reconstructed.climate.lapseRateCPerMeter ==
                Catch::Approx(physical.lapseRateCPerMeter));
        REQUIRE(reconstructed.slope == 0.375);
    }
}

TEST_CASE("Exact learned controls reject out-of-chunk reconstruction without latching failure",
          "[learned][v4][column-plan][stencil][bounds][regression]") {
    constexpr uint64_t SEED = 0xB0A0'D5CA'570C'11ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<EphemeralRetainedPageAuthority>(identity);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);
    const worldgen::MacroControlView control = macro.controlView({0, 0});
    control.prepareLearnedAuthority();
    const uint64_t beforeQueries = context->metrics().queries;

    worldgen::SurfaceSample destination;
    destination.slope = 0.75;
    try {
        control.reconstructContinuous(17.0, 8.0, destination);
        FAIL("Out-of-chunk learned reconstruction unexpectedly succeeded");
    } catch (const GenerationFailureException& failure) {
        REQUIRE(failure.status() == AuthorityStatus::FAILED);
        REQUIRE(failure.failure().code == GenerationFailureCode::INVALID_REQUEST);
    }
    REQUIRE(destination.slope == 0.75);
    REQUIRE(context->metrics().queries == beforeQueries);
    REQUIRE_FALSE(context->failure().has_value());

    destination.terrainHeight = 80.0;
    destination.hydrology.terrainSlope = 0.25;
    REQUIRE_NOTHROW(control.reconstructContinuous(16.0, 8.0, destination));
    REQUIRE(destination.slope == 0.25);
    REQUIRE_FALSE(context->failure().has_value());
}

TEST_CASE("V4 scalar control and geometry grids consume identical canonical water",
          "[learned][v4][hydrology][grid][control]") {
    TempDir directory("learned_canonical_water_paths");
    constexpr uint64_t SEED = 0xCA10'CA1E'0004ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);
    constexpr int64_t X = 128;
    constexpr int64_t Z = 128;

    const worldgen::HydrologySample canonical =
        awaitGeneration([&] { return macro.sampleHydrology(X, Z); }, std::chrono::seconds(15));
    std::array<worldgen::HydrologySample, 1> hydrologyGrid{};
    awaitGeneration(
        [&] {
            macro.sampleHydrologyGrid(X, Z, 1, 1, 1, 1, hydrologyGrid);
            return true;
        },
        std::chrono::seconds(15));
    const worldgen::SurfaceSample scalar = awaitGeneration(
        [&] { return macro.sampleSurface(X, Z, worldgen::SurfaceFootprint::BLOCK_1); },
        std::chrono::seconds(15));
    std::array<worldgen::SurfaceSample, 1> geometryGrid{};
    awaitGeneration(
        [&] {
            macro.sampleGeometryGrid(X, Z, 1, 1, 1, 1, worldgen::SurfaceFootprint::BLOCK_1,
                                     geometryGrid);
            return true;
        },
        std::chrono::seconds(15));

    const auto requireCanonicalWater = [&](const worldgen::HydrologySample& sample) {
        REQUIRE(sample.ocean == canonical.ocean);
        REQUIRE(sample.lake == canonical.lake);
        REQUIRE(sample.river == canonical.river);
        REQUIRE(sample.waterBodyId == canonical.waterBodyId);
        REQUIRE(sample.waterSurface == canonical.waterSurface);
        // V4 does not add legacy synthetic relief after canonical routing.
        // Every terrain path must preserve the solved bed, shore, and outlet
        // elevation exactly, including dry land and ocean floors.
        REQUIRE(sample.surfaceElevation == canonical.surfaceElevation);
        REQUIRE(sample.channelDistance == canonical.channelDistance);
        REQUIRE(sample.lakeShoreDistance == canonical.lakeShoreDistance);
    };
    requireCanonicalWater(hydrologyGrid[0]);
    requireCanonicalWater(scalar.hydrology);
    requireCanonicalWater(geometryGrid[0].hydrology);
    REQUIRE(scalar.terrainHeight == canonical.surfaceElevation);
    REQUIRE(geometryGrid[0].terrainHeight == canonical.surfaceElevation);
    const worldgen::SurfaceSample filtered = awaitGeneration(
        [&] { return macro.sampleSurface(X, Z, worldgen::SurfaceFootprint::BLOCK_16); },
        std::chrono::seconds(15));
    requireCanonicalWater(filtered.hydrology);
    worldgen::SurfaceSample reconstructed = scalar;
    const worldgen::MacroControlView control =
        macro.controlView({Chunk::worldToChunk(X), Chunk::worldToChunk(Z)});
    awaitGeneration(
        [&] {
            control.prepareLearnedAuthority();
            return true;
        },
        std::chrono::seconds(15));
    const WorldGenerationMetrics beforeControl = context->metrics();
    control.reconstructContinuous(X, Z, reconstructed);
    const WorldGenerationMetrics afterFirstControl = context->metrics();
    worldgen::SurfaceSample repeated = scalar;
    control.reconstructContinuous(X, Z, repeated);
    const WorldGenerationMetrics afterRepeatedControl = context->metrics();
    REQUIRE(reconstructed.slope == scalar.hydrology.terrainSlope);
    REQUIRE(reconstructed.climate.temperatureC == scalar.climate.temperatureC);
    REQUIRE(repeated.climate.temperatureC == reconstructed.climate.temperatureC);
    REQUIRE(repeated.climate.temperatureVariabilityC ==
            reconstructed.climate.temperatureVariabilityC);
    REQUIRE(repeated.climate.annualPrecipitationMm == reconstructed.climate.annualPrecipitationMm);
    REQUIRE(repeated.climate.precipitationCoefficientOfVariation ==
            reconstructed.climate.precipitationCoefficientOfVariation);
    REQUIRE(repeated.climate.lapseRateCPerMeter == reconstructed.climate.lapseRateCPerMeter);
    // Once the associated chunk's bounded native stencil is copied,
    // reconstruction performs no synchronized context query.
    REQUIRE(afterFirstControl.queries == beforeControl.queries);
    REQUIRE(afterRepeatedControl.queries == afterFirstControl.queries);
    // V4 samples physical authority directly. A scalar far footprint and an
    // exact control view must not materialize a legacy macro or far-climate
    // control tile merely to discard its interpolation result.
    REQUIRE(macro.macroControlCacheMetrics().entries == 0);
    REQUIRE(macro.macroControlCacheMetrics().builds == 0);
    REQUIRE(macro.farClimateControlCacheMetrics().entries == 0);
    REQUIRE(macro.farClimateControlCacheMetrics().builds == 0);
    REQUIRE(macro.basinCacheMetrics().builds == 0);
    REQUIRE(macro.nativeHydrologyCacheMetrics().builds > 0);
}

TEST_CASE("V4 exact columns restore direct continuous geology after compact plan sampling",
          "[learned][v4][geology][column-plan][regression]") {
    TempDir directory("learned_exact_continuous_geology");
    constexpr uint64_t SEED = 0xCA10'CA1E'0006ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    // The compact plan intentionally owns only block-resolution categorical
    // geology. Find a non-lattice point where its retained lattice still
    // differs from the coordinate-pure continuous geology, then ensure the
    // public exact column restores the latter before ecology consumes it.
    constexpr ColumnPos COLUMN{151, -227};
    const int64_t baseX = COLUMN.x * CHUNK_EDGE;
    const int64_t baseZ = COLUMN.z * CHUNK_EDGE;
    const std::shared_ptr<const ColumnPlan> plan =
        awaitGeneration([&] { return generator.getColumnPlan(COLUMN); }, std::chrono::seconds(20));
    const WorldGenerationMetrics beforePlanSamples = context->metrics();

    struct Candidate {
        ColumnPos position;
        worldgen::GeologySample compact;
        worldgen::GeologySample direct;
    };
    std::optional<Candidate> candidate;
    for (int localZ = 1; localZ < CHUNK_EDGE && !candidate; ++localZ) {
        for (int localX = 1; localX < CHUNK_EDGE; ++localX) {
            const int64_t x = baseX + localX;
            const int64_t z = baseZ + localZ;
            const worldgen::GeologySample compact = plan->sample(localX, localZ).geology;
            const worldgen::GeologySample direct = macro.sampleGeology(x, z);
            const double difference = std::max({
                std::abs(compact.erosionResistance - direct.erosionResistance),
                std::abs(compact.continentalFraction - direct.continentalFraction),
                std::abs(compact.distanceToBoundary - direct.distanceToBoundary),
                std::abs(compact.uplift - direct.uplift),
                std::abs(compact.rift - direct.rift),
                std::abs(compact.faultStrength - direct.faultStrength),
                std::abs(compact.hotspotInfluence - direct.hotspotInfluence),
                std::abs(compact.volcanicActivity - direct.volcanicActivity),
            });
            if (difference > 1.0e-9) {
                candidate = Candidate{{x, z}, compact, direct};
                break;
            }
        }
    }
    REQUIRE(candidate.has_value());
    REQUIRE(context->metrics().queries == beforePlanSamples.queries);

    const worldgen::SurfaceSample exact = awaitGeneration(
        [&] { return generator.sampleSurface(candidate->position.x, candidate->position.z); },
        std::chrono::seconds(20));
    const worldgen::SurfaceSample coordinatePure = awaitGeneration(
        [&] {
            return macro.sampleSurface(candidate->position.x, candidate->position.z,
                                       worldgen::SurfaceFootprint::BLOCK_1);
        },
        std::chrono::seconds(20));
    CAPTURE(candidate->position.x, candidate->position.z, candidate->compact.erosionResistance,
            candidate->direct.erosionResistance, exact.geology.erosionResistance,
            candidate->compact.continentalFraction, candidate->direct.continentalFraction,
            exact.geology.continentalFraction, candidate->compact.distanceToBoundary,
            candidate->direct.distanceToBoundary, exact.geology.distanceToBoundary,
            coordinatePure.climate.potentialEvapotranspirationMm,
            exact.climate.potentialEvapotranspirationMm, coordinatePure.climate.aridity,
            exact.climate.aridity);
    REQUIRE(exact.geology.erosionResistance == Catch::Approx(candidate->direct.erosionResistance));
    REQUIRE(exact.geology.plateVelocity.x == Catch::Approx(candidate->direct.plateVelocity.x));
    REQUIRE(exact.geology.plateVelocity.z == Catch::Approx(candidate->direct.plateVelocity.z));
    REQUIRE(exact.geology.continentalFraction ==
            Catch::Approx(candidate->direct.continentalFraction));
    REQUIRE(exact.geology.crustAge == Catch::Approx(candidate->direct.crustAge));
    REQUIRE(exact.geology.crustThickness == Catch::Approx(candidate->direct.crustThickness));
    REQUIRE(exact.geology.crustDensity == Catch::Approx(candidate->direct.crustDensity));
    REQUIRE(exact.geology.distanceToBoundary ==
            Catch::Approx(candidate->direct.distanceToBoundary));
    REQUIRE(exact.geology.uplift == Catch::Approx(candidate->direct.uplift));
    REQUIRE(exact.geology.rift == Catch::Approx(candidate->direct.rift));
    REQUIRE(exact.geology.faultStrength == Catch::Approx(candidate->direct.faultStrength));
    REQUIRE(exact.geology.hotspotInfluence == Catch::Approx(candidate->direct.hotspotInfluence));
    REQUIRE(exact.geology.volcanicActivity == Catch::Approx(candidate->direct.volcanicActivity));
    REQUIRE(exact.climate.potentialEvapotranspirationMm ==
            Catch::Approx(coordinatePure.climate.potentialEvapotranspirationMm));
    REQUIRE(exact.climate.aridity == Catch::Approx(coordinatePure.climate.aridity));
    REQUIRE(exact.soil.moisture == Catch::Approx(coordinatePure.soil.moisture));
    REQUIRE(exact.soil.fertility == Catch::Approx(coordinatePure.soil.fertility));
    REQUIRE(exact.soil.drainage == Catch::Approx(coordinatePure.soil.drainage));
    REQUIRE(exact.soil.waterTable == Catch::Approx(coordinatePure.soil.waterTable));
    REQUIRE(exact.biome.primary == coordinatePure.biome.primary);
    REQUIRE(exact.biome.secondary == coordinatePure.biome.secondary);
    REQUIRE(exact.biome.transition == Catch::Approx(coordinatePure.biome.transition));
    for (size_t index = 0; index < exact.suitability.scores.size(); ++index) {
        REQUIRE(exact.suitability.scores[index] ==
                Catch::Approx(coordinatePure.suitability.scores[index]));
    }
}

TEST_CASE("V4 dry learned relief is slope gated after canonical routing",
          "[learned][v4][relief][regression]") {
    TempDir directory("learned_v4_slope_gated_relief");
    constexpr uint64_t SEED = 0xD37A'110C'0006ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DrySlopeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    std::optional<ColumnPos> selected;
    worldgen::HydrologySample routed;
    worldgen::SurfaceSample detailed;
    for (int64_t z = 64; z <= 320 && !selected; z += 16) {
        for (int64_t x = 64; x <= 320; x += 16) {
            const worldgen::HydrologySample candidate =
                awaitGeneration([&] { return macro.sampleHydrology(x, z); }, 20s);
            const worldgen::SurfaceSample surface = awaitGeneration(
                [&] { return macro.sampleSurface(x, z, worldgen::SurfaceFootprint::BLOCK_1); },
                20s);
            if (!candidate.ocean && !candidate.river && !candidate.lake && !candidate.wetland &&
                !candidate.waterfall &&
                std::abs(surface.terrainHeight - candidate.surfaceElevation) > 0.02) {
                selected = ColumnPos{x, z};
                routed = candidate;
                detailed = surface;
                break;
            }
        }
    }
    REQUIRE(selected);
    CAPTURE(selected->x, selected->z, routed.surfaceElevation, routed.terrainSlope,
            routed.channelDistance, routed.channelWidth, detailed.terrainHeight);
    REQUIRE(std::abs(detailed.terrainHeight - routed.surfaceElevation) <= 1.5);
    REQUIRE(detailed.terrainHeight > SEA_LEVEL);
    REQUIRE_FALSE(detailed.hydrology.ocean);
    REQUIRE_FALSE(detailed.hydrology.river);
    REQUIRE_FALSE(detailed.hydrology.lake);
    REQUIRE_FALSE(detailed.hydrology.wetland);
    REQUIRE(detailed.hydrology.waterBodyId == routed.waterBodyId);
    REQUIRE(detailed.hydrology.surfaceElevation == detailed.terrainHeight);

    const worldgen::SurfaceSample farGeometry = awaitGeneration(
        [&] { return generator.sampleFarGeometrySurface(selected->x, selected->z); }, 20s);
    REQUIRE(farGeometry.terrainHeight == Catch::Approx(detailed.terrainHeight).margin(1.0e-9));
    REQUIRE(farGeometry.hydrology.waterBodyId == detailed.hydrology.waterBodyId);

    const worldgen::SurfaceSample exact =
        awaitGeneration([&] { return generator.sampleSurface(selected->x, selected->z); }, 20s);
    std::array<worldgen::SurfaceSample, 1> habitat{};
    const std::array<ColumnPos, 1> positions{*selected};
    awaitGeneration(
        [&] {
            generator.sampleFarHabitatPoints(positions, habitat);
            return true;
        },
        20s);
    CAPTURE(exact.terrainHeight, habitat.front().terrainHeight, detailed.climate.temperatureC,
            exact.climate.temperatureC, habitat.front().climate.temperatureC,
            detailed.climate.potentialEvapotranspirationMm,
            exact.climate.potentialEvapotranspirationMm,
            habitat.front().climate.potentialEvapotranspirationMm);
    REQUIRE(exact.terrainHeight == Catch::Approx(detailed.terrainHeight).margin(1.0e-6));
    REQUIRE(habitat.front().terrainHeight == Catch::Approx(exact.terrainHeight).margin(1.0e-6));
    REQUIRE(exact.climate.temperatureC ==
            Catch::Approx(detailed.climate.temperatureC).margin(1.0e-6));
    REQUIRE(habitat.front().climate.temperatureC ==
            Catch::Approx(exact.climate.temperatureC).margin(1.0e-6));
    REQUIRE(habitat.front().climate.potentialEvapotranspirationMm ==
            Catch::Approx(exact.climate.potentialEvapotranspirationMm).margin(1.0e-6));
    REQUIRE(habitat.front().climate.relativeHumidity ==
            Catch::Approx(exact.climate.relativeHumidity).margin(1.0e-9));
    REQUIRE(habitat.front().climate.aridity == Catch::Approx(exact.climate.aridity).margin(1.0e-9));
    REQUIRE(habitat.front().hydrology.waterBodyId == exact.hydrology.waterBodyId);

    const worldgen::SurfaceSample coarse = awaitGeneration(
        [&] {
            return generator.sampleFarGeometrySurface(selected->x, selected->z,
                                                      worldgen::SurfaceFootprint::BLOCK_32);
        },
        20s);
    REQUIRE(std::abs(coarse.terrainHeight - routed.surfaceElevation) <= 1.5);
    REQUIRE(coarse.hydrology.waterBodyId == detailed.hydrology.waterBodyId);
}

TEST_CASE("V4 chunk paths retain fake canonical ocean beds without legacy relief",
          "[learned][v4][hydrology][relief][regression]") {
    TempDir directory("learned_chunk_canonical_ocean_relief");
    constexpr uint64_t SEED = 0xCA10'CA1E'0005ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>(-25));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    // The intentionally negative fake authority makes every candidate ocean.
    // Pick a coordinate where the retired v3 ocean-floor detail would move the
    // integer emitted bed, so this protects the complete ChunkGenerator path
    // instead of merely comparing a floating-point no-op.
    ColumnPos position{};
    worldgen::HydrologySample canonical{};
    bool found = false;
    for (int z = -128; z <= 128 && !found; z += 8) {
        for (int x = -128; x <= 128; x += 8) {
            const worldgen::HydrologySample candidate = awaitGeneration(
                [&] { return macro.sampleHydrology(x, z); }, std::chrono::seconds(15));
            const double retiredDetail =
                macro.reliefDetail(x, z, worldgen::SurfaceFootprint::BLOCK_1) *
                worldgen::OCEAN_FLOOR_DETAIL_SCALE;
            if (!candidate.ocean || retiredDetail <= 0.75)
                continue;
            position = {x, z};
            canonical = candidate;
            found = true;
            break;
        }
    }
    REQUIRE(found);
    REQUIRE(canonical.surfaceElevation < SEA_LEVEL);

    const worldgen::SurfaceSample far =
        awaitGeneration([&] { return generator.sampleFarGeometrySurface(position.x, position.z); },
                        std::chrono::seconds(20));
    const std::shared_ptr<const ColumnPlan> plan = awaitGeneration(
        [&] {
            return generator.getColumnPlan(
                {Chunk::worldToChunk(position.x), Chunk::worldToChunk(position.z)});
        },
        std::chrono::seconds(20));
    const worldgen::SurfaceSample planned =
        plan->sample(Chunk::worldToLocal(position.x), Chunk::worldToLocal(position.z));
    const worldgen::SurfaceSample exact = awaitGeneration(
        [&] { return generator.sampleSurface(position.x, position.z); }, std::chrono::seconds(20));
    std::array<worldgen::SurfaceSample, 1> habitat{};
    const std::array<ColumnPos, 1> positions{position};
    awaitGeneration(
        [&] {
            generator.sampleFarHabitatPoints(positions, habitat);
            return true;
        },
        std::chrono::seconds(20));

    CAPTURE(position.x, position.z, canonical.surfaceElevation, far.terrainHeight,
            planned.hydrology.surfaceElevation, exact.terrainHeight, habitat.front().terrainHeight);
    REQUIRE(far.hydrology.surfaceElevation == Catch::Approx(canonical.surfaceElevation));
    REQUIRE(far.terrainHeight == Catch::Approx(canonical.surfaceElevation));
    REQUIRE(planned.hydrology.surfaceElevation == Catch::Approx(canonical.surfaceElevation));
    REQUIRE(exact.terrainHeight == Catch::Approx(canonical.surfaceElevation));
    REQUIRE(habitat.front().terrainHeight == Catch::Approx(canonical.surfaceElevation));
}

TEST_CASE("V4 native far-water authority grids retain canonical hydrology without geometry work",
          "[learned][v4][hydrology][far-water][performance]") {
    TempDir directory("learned_native_far_water_authority");
    constexpr uint64_t SEED = 0xC0A5'7EED'0004ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    constexpr int64_t ORIGIN_X = 384;
    constexpr int64_t ORIGIN_Z = -640;
    constexpr int SPACING = 4;
    constexpr int EDGE = 3;
    std::array<worldgen::HydrologySample, EDGE * EDGE> authorityGrid{};
    std::array<worldgen::HydrologySample, EDGE * EDGE> macroGrid{};
    awaitGeneration(
        [&] {
            generator.sampleNativeHydrologyAuthorityGrid(ORIGIN_X, ORIGIN_Z, SPACING, SPACING, EDGE,
                                                         EDGE, authorityGrid);
            return true;
        },
        std::chrono::seconds(15));
    awaitGeneration(
        [&] {
            macro.sampleHydrologyGrid(ORIGIN_X, ORIGIN_Z, SPACING, SPACING, EDGE, EDGE, macroGrid);
            return true;
        },
        std::chrono::seconds(15));

    REQUIRE(generator.usesLearnedAuthority());
    for (size_t index = 0; index < authorityGrid.size(); ++index) {
        const worldgen::HydrologySample& actual = authorityGrid[index];
        const worldgen::HydrologySample& expected = macroGrid[index];
        CHECK(actual.ocean == expected.ocean);
        CHECK(actual.lake == expected.lake);
        CHECK(actual.river == expected.river);
        CHECK(actual.wetland == expected.wetland);
        CHECK(actual.waterfall == expected.waterfall);
        CHECK(actual.waterBodyId == expected.waterBodyId);
        CHECK(actual.surfaceElevation == expected.surfaceElevation);
        CHECK(actual.waterSurface == expected.waterSurface);
        CHECK(actual.channelDepth == expected.channelDepth);
        CHECK(actual.discharge == expected.discharge);
    }

    const std::array<ColumnPos, 3> points{{
        {ORIGIN_X, ORIGIN_Z},
        {ORIGIN_X + 17, ORIGIN_Z + 9},
        {ORIGIN_X + 32, ORIGIN_Z + 32},
    }};
    std::array<worldgen::HydrologySample, points.size()> pointAuthority{};
    std::array<worldgen::HydrologySample, points.size()> pointMacro{};
    awaitGeneration(
        [&] {
            generator.sampleNativeHydrologyAuthorityPoints(points, pointAuthority);
            return true;
        },
        std::chrono::seconds(15));
    awaitGeneration(
        [&] {
            macro.sampleHydrologyPoints(points, pointMacro);
            return true;
        },
        std::chrono::seconds(15));
    for (size_t index = 0; index < pointAuthority.size(); ++index) {
        const worldgen::HydrologySample& actual = pointAuthority[index];
        const worldgen::HydrologySample& expected = pointMacro[index];
        CHECK(actual.ocean == expected.ocean);
        CHECK(actual.lake == expected.lake);
        CHECK(actual.river == expected.river);
        CHECK(actual.wetland == expected.wetland);
        CHECK(actual.waterfall == expected.waterfall);
        CHECK(actual.waterBodyId == expected.waterBodyId);
        CHECK(actual.surfaceElevation == expected.surfaceElevation);
        CHECK(actual.waterSurface == expected.waterSurface);
        CHECK(actual.channelDepth == expected.channelDepth);
        CHECK(actual.discharge == expected.discharge);
    }

    ChunkGenerator legacy(static_cast<uint32_t>(SEED));
    REQUIRE_FALSE(legacy.usesLearnedAuthority());
}

TEST_CASE("V4 macro hydrology keeps sub-block learned meter slopes",
          "[learned][v4][hydrology][meters][d-infinity][regression]") {
    TempDir directory("learned_native_meter_incline");
    constexpr uint64_t SEED = 0x5A0F'1E00'0004ULL;
    constexpr int64_t X = 1'024;
    constexpr int64_t Z = 1'024;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<SubBlockInclineTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    const auto metersAt = [](int64_t x, int64_t z) {
        return 1'005.0 - static_cast<double>(x / MODEL_BLOCK_SCALE) -
               static_cast<double>(z / MODEL_BLOCK_SCALE);
    };
    const double emittedHeight = learnedElevationMetersToWorldHeight(metersAt(X, Z));
    REQUIRE(learnedElevationMetersToWorldHeight(metersAt(X + MODEL_BLOCK_SCALE, Z)) ==
            emittedHeight);
    REQUIRE(learnedElevationMetersToWorldHeight(metersAt(X, Z + MODEL_BLOCK_SCALE)) ==
            emittedHeight);

    const worldgen::HydrologySample sample = awaitGeneration(
        [&] { return macro.sampleHydrology(static_cast<double>(X), static_cast<double>(Z)); }, 20s);
    REQUIRE_FALSE(sample.ocean);
    REQUIRE_FALSE(sample.lake);
    REQUIRE_FALSE(sample.river);
    REQUIRE(sample.surfaceElevation == emittedHeight);
    REQUIRE(sample.flowDirection.x > 0.65);
    REQUIRE(sample.flowDirection.x < 0.75);
    REQUIRE(sample.flowDirection.z > 0.65);
    REQUIRE(sample.flowDirection.z < 0.75);
    REQUIRE(std::abs(std::hypot(sample.flowDirection.x, sample.flowDirection.z) - 1.0) < 1.0e-6);
}

TEST_CASE("V4 native water grids preserve world-point reconstruction with a structured stencil",
          "[learned][v4][hydrology][far-water][performance]") {
    TempDir directory("learned_native_water_grid_stencil");
    constexpr uint64_t SEED = 0xA114'7EED'0004ULL;
    constexpr int64_t ORIGIN_X = -4;
    constexpr int64_t ORIGIN_Z = -8;
    constexpr int EDGE = 4;
    constexpr int SPACING = 4;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    std::array<ColumnPos, EDGE * EDGE> positions{};
    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            positions[static_cast<size_t>(sampleZ * EDGE + sampleX)] = {
                ORIGIN_X + static_cast<int64_t>(sampleX) * SPACING,
                ORIGIN_Z + static_cast<int64_t>(sampleZ) * SPACING,
            };
        }
    }

    // Point sampling uses the general queryWorldPoints path. It warms exactly
    // the native stencil later consumed by the grid path and is the reference
    // for its align_corners=false bilinear reconstruction.
    std::array<worldgen::HydrologySample, EDGE * EDGE> pointSamples{};
    awaitGeneration(
        [&] {
            macro.sampleHydrologyPoints(positions, pointSamples);
            return true;
        },
        std::chrono::seconds(15));

    const WorldGenerationMetrics beforeGrid = context->metrics();
    std::array<worldgen::HydrologySample, EDGE * EDGE> gridSamples{};
    awaitGeneration(
        [&] {
            macro.sampleHydrologyGrid(ORIGIN_X, ORIGIN_Z, SPACING, SPACING, EDGE, EDGE,
                                      gridSamples);
            return true;
        },
        std::chrono::seconds(15));
    const WorldGenerationMetrics afterGrid = context->metrics();

    // An aligned 4 by 4 world grid needs its shared 5 by 5 native stencil,
    // rather than four independently deduplicated points per output sample.
    REQUIRE(afterGrid.queries - beforeGrid.queries == 1);
    REQUIRE(afterGrid.readyQueries - beforeGrid.readyQueries == 1);
    REQUIRE(afterGrid.requestedSamples - beforeGrid.requestedSamples ==
            static_cast<uint64_t>((EDGE + 1) * (EDGE + 1)));

    const auto exactlyEqual = [](const worldgen::HydrologySample& actual,
                                 const worldgen::HydrologySample& expected) {
        return actual.waterBodyId == expected.waterBodyId &&
               actual.generatedFluidLevel == expected.generatedFluidLevel &&
               actual.transitionOwnerKind == expected.transitionOwnerKind &&
               actual.transitionOwnerId == expected.transitionOwnerId &&
               actual.flowDirection.x == expected.flowDirection.x &&
               actual.flowDirection.z == expected.flowDirection.z &&
               actual.surfaceElevation == expected.surfaceElevation &&
               actual.waterSurface == expected.waterSurface &&
               actual.discharge == expected.discharge && actual.sediment == expected.sediment &&
               actual.channelDistance == expected.channelDistance &&
               actual.channelWidth == expected.channelWidth &&
               actual.channelDepth == expected.channelDepth &&
               actual.channelGradient == expected.channelGradient &&
               actual.erosionDepth == expected.erosionDepth &&
               actual.lakeDepth == expected.lakeDepth &&
               actual.lakeShoreDistance == expected.lakeShoreDistance &&
               actual.shoreWaterSurface == expected.shoreWaterSurface &&
               actual.lakeBankTarget == expected.lakeBankTarget &&
               actual.lakeBankInfluence == expected.lakeBankInfluence &&
               actual.lakeAreaSquareKilometers == expected.lakeAreaSquareKilometers &&
               actual.lakeVolumeCubicMeters == expected.lakeVolumeCubicMeters &&
               actual.lakeRunoffMmSquareKilometers == expected.lakeRunoffMmSquareKilometers &&
               actual.lakeLossMm == expected.lakeLossMm &&
               actual.lakeOverflowMmSquareKilometers == expected.lakeOverflowMmSquareKilometers &&
               actual.lakeSpillSurface == expected.lakeSpillSurface &&
               actual.baseflow == expected.baseflow &&
               actual.precipitationSeasonality == expected.precipitationSeasonality &&
               actual.groundwaterRechargeMm == expected.groundwaterRechargeMm &&
               actual.groundwaterHead == expected.groundwaterHead &&
               actual.hydroperiod == expected.hydroperiod &&
               actual.waterfallTop == expected.waterfallTop &&
               actual.waterfallBottom == expected.waterfallBottom &&
               actual.waterfallWidth == expected.waterfallWidth &&
               actual.streamOrder == expected.streamOrder &&
               actual.distributaryCount == expected.distributaryCount &&
               actual.ocean == expected.ocean && actual.river == expected.river &&
               actual.lake == expected.lake && actual.lakeBank == expected.lakeBank &&
               actual.channelBank == expected.channelBank &&
               actual.endorheic == expected.endorheic && actual.waterfall == expected.waterfall &&
               actual.waterfallAnchor == expected.waterfallAnchor &&
               actual.delta == expected.delta && actual.estuary == expected.estuary &&
               actual.brackish == expected.brackish && actual.perennial == expected.perennial &&
               actual.ephemeral == expected.ephemeral && actual.wetland == expected.wetland;
    };
    for (size_t index = 0; index < gridSamples.size(); ++index) {
        CAPTURE(index, positions[index].x, positions[index].z);
        REQUIRE(exactlyEqual(gridSamples[index], pointSamples[index]));
    }
}

TEST_CASE("V4 geometry batches reuse canonical authority samples for climate",
          "[learned][v4][hydrology][geometry][performance]") {
    TempDir directory("learned_geometry_climate_batch");
    constexpr uint64_t SEED = 0xBA7C'4ED0'0004ULL;
    constexpr int64_t ORIGIN = 128;
    constexpr int SPACING = 16;
    constexpr int EDGE = 3;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    std::array<ColumnPos, EDGE * EDGE> positions{};
    std::array<WorldBlockPoint, EDGE * EDGE> authorityPositions{};
    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * EDGE + sampleX);
            const int64_t x = ORIGIN + static_cast<int64_t>(sampleX) * SPACING;
            const int64_t z = ORIGIN + static_cast<int64_t>(sampleZ) * SPACING;
            positions[index] = {x, z};
            authorityPositions[index] = {.x = x, .z = z};
        }
    }

    // Warm the native owner and final authority pages without constructing the
    // far climate tile. V4 geometry consumes the physical terrain samples
    // returned alongside its hydrology batch, so the first geometry call
    // below must issue one logical authority request for its exact positions.
    std::array<worldgen::HydrologySample, 25> farControlHydrology{};
    awaitGeneration(
        [&] {
            macro.sampleHydrologyGrid(ORIGIN - 128, ORIGIN - 128, 128, 128, 5, 5,
                                      farControlHydrology);
            return true;
        },
        std::chrono::seconds(30));

    const WorldGenerationMetrics coldGeometryBefore = context->metrics();
    std::array<worldgen::SurfaceSample, EDGE * EDGE> pointWarmup{};
    awaitGeneration(
        [&] {
            macro.sampleGeometryPoints(positions, worldgen::SurfaceFootprint::BLOCK_16,
                                       pointWarmup);
            return true;
        },
        std::chrono::seconds(30));
    const WorldGenerationMetrics coldGeometryAfter = context->metrics();
    REQUIRE(coldGeometryAfter.queries - coldGeometryBefore.queries == 1);
    REQUIRE(coldGeometryAfter.readyQueries - coldGeometryBefore.readyQueries == 1);
    REQUIRE(coldGeometryAfter.requestedSamples - coldGeometryBefore.requestedSamples ==
            positions.size());
    const auto authoritative =
        awaitAuthority([&] { return context->queryWorldPoints(authorityPositions); }, 30s);
    REQUIRE(authoritative.isReady());
    for (size_t index = 0; index < pointWarmup.size(); ++index) {
        const PhysicalTerrainSample& physical = (*authoritative.value())[index];
        const double nativeHeight = learnedElevationMetersToWorldHeight(physical.elevationMeters);
        const double expectedTemperature =
            physical.meanTemperatureC + physical.lapseRateCPerMeter *
                                            (pointWarmup[index].terrainHeight - nativeHeight) *
                                            WORLD_METERS_PER_BLOCK;
        REQUIRE(pointWarmup[index].climate.temperatureC == expectedTemperature);
        REQUIRE(pointWarmup[index].climate.temperatureVariabilityC ==
                physical.temperatureVariabilityC);
        REQUIRE(pointWarmup[index].climate.annualPrecipitationMm == physical.annualPrecipitationMm);
        REQUIRE(pointWarmup[index].climate.precipitationCoefficientOfVariation ==
                physical.precipitationCoefficientOfVariation);
        REQUIRE(pointWarmup[index].climate.lapseRateCPerMeter == physical.lapseRateCPerMeter);
    }

    const WorldGenerationMetrics pointBefore = context->metrics();
    std::array<worldgen::SurfaceSample, EDGE * EDGE> pointRepeat{};
    awaitGeneration(
        [&] {
            macro.sampleGeometryPoints(positions, worldgen::SurfaceFootprint::BLOCK_16,
                                       pointRepeat);
            return true;
        },
        std::chrono::seconds(30));
    const WorldGenerationMetrics pointAfter = context->metrics();
    REQUIRE(pointAfter.queries - pointBefore.queries == 1);
    REQUIRE(pointAfter.readyQueries - pointBefore.readyQueries == 1);
    REQUIRE(pointAfter.requestedSamples - pointBefore.requestedSamples == positions.size());
    for (size_t index = 0; index < pointRepeat.size(); ++index) {
        REQUIRE(pointRepeat[index].terrainHeight == pointWarmup[index].terrainHeight);
        REQUIRE(pointRepeat[index].hydrology.surfaceElevation ==
                pointWarmup[index].hydrology.surfaceElevation);
        REQUIRE(pointRepeat[index].climate.temperatureC == pointWarmup[index].climate.temperatureC);
    }

    std::array<worldgen::SurfaceSample, EDGE * EDGE> gridWarmup{};
    awaitGeneration(
        [&] {
            macro.sampleGeometryGrid(ORIGIN, ORIGIN, SPACING, SPACING, EDGE, EDGE,
                                     worldgen::SurfaceFootprint::BLOCK_16, gridWarmup);
            return true;
        },
        std::chrono::seconds(30));
    const WorldGenerationMetrics gridBefore = context->metrics();
    std::array<worldgen::SurfaceSample, EDGE * EDGE> gridRepeat{};
    awaitGeneration(
        [&] {
            macro.sampleGeometryGrid(ORIGIN, ORIGIN, SPACING, SPACING, EDGE, EDGE,
                                     worldgen::SurfaceFootprint::BLOCK_16, gridRepeat);
            return true;
        },
        std::chrono::seconds(30));
    const WorldGenerationMetrics gridAfter = context->metrics();
    REQUIRE(gridAfter.queries - gridBefore.queries == 1);
    REQUIRE(gridAfter.readyQueries - gridBefore.readyQueries == 1);
    REQUIRE(gridAfter.requestedSamples - gridBefore.requestedSamples == positions.size());
    for (size_t index = 0; index < gridRepeat.size(); ++index) {
        REQUIRE(gridRepeat[index].terrainHeight == pointWarmup[index].terrainHeight);
        REQUIRE(gridRepeat[index].hydrology.surfaceElevation ==
                pointWarmup[index].hydrology.surfaceElevation);
        REQUIRE(gridRepeat[index].climate.temperatureC == pointWarmup[index].climate.temperatureC);
    }
}

TEST_CASE("V4 far surface grids reuse batched learned climate authority",
          "[learned][v4][hydrology][surface-grid][far-climate][performance]") {
    TempDir directory("learned_surface_grid_climate_batch");
    constexpr uint64_t SEED = 0xBA7C'4ED0'0005ULL;
    constexpr int64_t ORIGIN = 128;
    constexpr int SPACING = 16;
    constexpr int EDGE = 5;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(SEED, context);

    // Keep the batch inside one far climate control tile, then warm that tile
    // and its native hydrology owner before measuring the surface-grid path.
    // The measured call must issue exactly one final-authority query for all
    // grid samples, not a scalar climate query for every vertex.
    awaitGeneration(
        [&] {
            (void)macro.sampleSurface(ORIGIN, ORIGIN, worldgen::SurfaceFootprint::BLOCK_16);
            return true;
        },
        std::chrono::seconds(30));

    std::array<worldgen::SurfaceSample, EDGE * EDGE> grid{};
    const WorldGenerationMetrics before = context->metrics();
    awaitGeneration(
        [&] {
            macro.sampleSurfaceGrid(ORIGIN, ORIGIN, SPACING, EDGE,
                                    worldgen::SurfaceFootprint::BLOCK_16, grid);
            return true;
        },
        std::chrono::seconds(30));
    const WorldGenerationMetrics after = context->metrics();
    REQUIRE(after.queries - before.queries == 1);
    REQUIRE(after.readyQueries - before.readyQueries == 1);
    REQUIRE(after.requestedSamples - before.requestedSamples == grid.size());

    for (int sampleZ = 0; sampleZ < EDGE; ++sampleZ) {
        for (int sampleX = 0; sampleX < EDGE; ++sampleX) {
            const size_t index = static_cast<size_t>(sampleZ * EDGE + sampleX);
            const int64_t worldX = ORIGIN + static_cast<int64_t>(sampleX) * SPACING;
            const int64_t worldZ = ORIGIN + static_cast<int64_t>(sampleZ) * SPACING;
            const worldgen::SurfaceSample scalar = awaitGeneration(
                [&] {
                    return macro.sampleSurface(worldX, worldZ,
                                               worldgen::SurfaceFootprint::BLOCK_16);
                },
                std::chrono::seconds(30));
            CAPTURE(worldX, worldZ);
            REQUIRE(grid[index].climate.temperatureC == scalar.climate.temperatureC);
            REQUIRE(grid[index].climate.temperatureVariabilityC ==
                    scalar.climate.temperatureVariabilityC);
            REQUIRE(grid[index].climate.annualPrecipitationMm ==
                    scalar.climate.annualPrecipitationMm);
            REQUIRE(grid[index].climate.precipitationCoefficientOfVariation ==
                    scalar.climate.precipitationCoefficientOfVariation);
            REQUIRE(grid[index].climate.lapseRateCPerMeter == scalar.climate.lapseRateCPerMeter);
            REQUIRE(grid[index].climate.potentialEvapotranspirationMm ==
                    scalar.climate.potentialEvapotranspirationMm);
            REQUIRE(grid[index].climate.aridity == scalar.climate.aridity);
            REQUIRE(grid[index].climate.relativeHumidity == scalar.climate.relativeHumidity);
        }
    }
}

TEST_CASE("V4 exact and far hydrology route against final authority while preview stays isolated",
          "[learned][v4][hydrology][preview][final][regression]") {
    TempDir directory("learned_preview_topology");
    constexpr uint64_t SEED = 0xC0A2'5E00'0004ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto backend = std::make_shared<QualitySplitTerrainBackend>(-25, 750);
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto finalContext =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    const auto previewContext = finalContext->withQuality(AuthorityQuality::PREVIEW);
    worldgen::MacroGenerationSampler finalMacro(SEED, finalContext);
    worldgen::MacroGenerationSampler previewMacro(SEED, previewContext);
    ChunkGenerator generator(SEED, finalContext);

    constexpr int64_t X = 512;
    constexpr int64_t Z = 512;
    const worldgen::HydrologySample finalHydrology = awaitGeneration(
        [&] { return finalMacro.sampleHydrology(static_cast<double>(X), static_cast<double>(Z)); },
        std::chrono::seconds(20));
    const worldgen::SurfaceSample exact = awaitGeneration(
        [&] { return generator.sampleExactSurface(X, Z); }, std::chrono::seconds(20));
    const worldgen::SurfaceSample far = awaitGeneration(
        [&] { return generator.sampleFarGeometrySurface(X, Z); }, std::chrono::seconds(20));

    // The preview page makes this coordinate ocean, while the final page is
    // well above sea level and routes a river. Final routing must consume the
    // final page itself, not route preview water and clip it against final
    // terrain afterward.
    REQUIRE_FALSE(finalHydrology.ocean);
    REQUIRE(finalHydrology.river);
    REQUIRE_FALSE(exact.hydrology.ocean);
    REQUIRE(exact.hydrology.river);
    REQUIRE_FALSE(far.hydrology.ocean);
    REQUIRE(far.hydrology.river);
    REQUIRE(exact.hydrology.waterBodyId == finalHydrology.waterBodyId);
    REQUIRE(far.hydrology.waterBodyId == finalHydrology.waterBodyId);
    // Exact columns quantize their bed to a block boundary. Their topology and
    // water body must nevertheless come from the same final routed authority.
    REQUIRE(far.hydrology.surfaceElevation == finalHydrology.surfaceElevation);

    const std::vector<TerrainPageKey> finalKeys = backend->keys();
    REQUIRE(std::ranges::count_if(finalKeys, [](TerrainPageKey key) {
                return key.quality == AuthorityQuality::PREVIEW;
            }) == 0);
    REQUIRE(std::ranges::count_if(finalKeys, [](TerrainPageKey key) {
                return key.quality == AuthorityQuality::FINAL;
            }) > 0);

    const worldgen::HydrologySample previewHydrology = awaitGeneration(
        [&] {
            return previewMacro.sampleHydrology(static_cast<double>(X), static_cast<double>(Z));
        },
        std::chrono::seconds(20));
    REQUIRE(previewHydrology.ocean);
    REQUIRE(previewHydrology.waterSurface == SEA_LEVEL);

    const std::vector<TerrainPageKey> isolatedKeys = backend->keys();
    REQUIRE(std::ranges::count_if(isolatedKeys, [](TerrainPageKey key) {
                return key.quality == AuthorityQuality::PREVIEW;
            }) > 0);
}

TEST_CASE("V4 keeps learned ecology physical while exact geometry uses its emitted top",
          "[learned][v4][ecology][geometry][far][regression]") {
    TempDir directory("learned_physical_ecology_emitted_geometry");
    constexpr uint64_t SEED = 0xEC01'0A74'0001ULL;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<DeterministicFakeTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    auto generator = std::make_shared<ChunkGenerator>(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    struct Candidate {
        int64_t x = 0;
        int64_t z = 0;
        worldgen::SurfaceSample physical;
        worldgen::SurfaceSample exact;
    };
    std::optional<Candidate> candidate;
    for (int64_t chunkZ = 30; chunkZ < 34 && !candidate.has_value(); ++chunkZ) {
        for (int64_t chunkX = 30; chunkX < 34 && !candidate.has_value(); ++chunkX) {
            const std::shared_ptr<const ColumnPlan> plan =
                awaitGeneration([&] { return generator->getColumnPlan({chunkX, chunkZ}); },
                                std::chrono::seconds(20));
            for (int localZ = 0; localZ < CHUNK_EDGE && !candidate.has_value(); ++localZ) {
                for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
                    const int64_t x = chunkX * CHUNK_EDGE + localX;
                    const int64_t z = chunkZ * CHUNK_EDGE + localZ;
                    const worldgen::SurfaceSample exact =
                        awaitGeneration([&] { return generator->sampleExactSurface(x, z); },
                                        std::chrono::seconds(20));
                    const double emittedTop = worldgen::geometryTerrainHeight(exact);
                    if (std::abs(emittedTop - exact.terrainHeight) < 0.5)
                        continue;
                    const worldgen::SurfaceSample physical = plan->sample(localX, localZ);
                    if (std::abs(exact.terrainHeight - physical.terrainHeight) > 1.0e-9)
                        continue;
                    candidate = Candidate{.x = x, .z = z, .physical = physical, .exact = exact};
                    break;
                }
            }
        }
    }

    REQUIRE(candidate.has_value());
    const Candidate& selected = *candidate;
    const double emittedTop = worldgen::geometryTerrainHeight(selected.exact);
    CAPTURE(selected.x, selected.z, selected.physical.terrainHeight, emittedTop,
            selected.exact.hydrology.surfaceElevation);
    REQUIRE(emittedTop == generator->surfaceYAt(selected.x, selected.z) + 1.0);
    REQUIRE(std::abs(emittedTop - selected.exact.terrainHeight) >= 0.5);
    REQUIRE(selected.exact.terrainHeight == Catch::Approx(selected.physical.terrainHeight));
    REQUIRE(selected.exact.hydrology.surfaceElevation ==
            Catch::Approx(selected.physical.hydrology.surfaceElevation));
    REQUIRE(selected.exact.hydrology.waterSurface ==
            Catch::Approx(selected.physical.hydrology.waterSurface));
    REQUIRE(selected.exact.hydrology.waterBodyId == selected.physical.hydrology.waterBodyId);

    worldgen::SurfaceSample expected = selected.physical;
    expected.hydrology.lakeDepth =
        expected.hydrology.lake
            ? std::max(0.0, expected.hydrology.waterSurface - expected.terrainHeight)
            : 0.0;
    expected.waterSurface = expected.hydrology.waterSurface;
    // The learned physical sample already carries the variability-aware PET
    // and aridity adapter. Exact voxel geometry must not replace it with the
    // legacy wind-based climate formula.
    expected.soil =
        macro.sampleSoil(static_cast<double>(selected.x), static_cast<double>(selected.z),
                         expected.geology, expected.hydrology, expected.climate);
    expected.suitability =
        macro.biomeSuitability(expected.geology, expected.hydrology, expected.climate,
                               expected.soil, expected.terrainHeight, expected.slope);
    expected.biome = worldgen::MacroGenerationSampler::selectBiome(expected.suitability);

    REQUIRE(selected.exact.climate.temperatureC == Catch::Approx(expected.climate.temperatureC));
    REQUIRE(selected.exact.climate.potentialEvapotranspirationMm ==
            Catch::Approx(expected.climate.potentialEvapotranspirationMm));
    REQUIRE(selected.exact.climate.aridity == Catch::Approx(expected.climate.aridity));
    REQUIRE(selected.exact.soil.moisture == Catch::Approx(expected.soil.moisture));
    REQUIRE(selected.exact.soil.fertility == Catch::Approx(expected.soil.fertility));
    REQUIRE(selected.exact.soil.drainage == Catch::Approx(expected.soil.drainage));
    REQUIRE(selected.exact.soil.waterTable == Catch::Approx(expected.soil.waterTable));
    for (size_t index = 0; index < expected.suitability.scores.size(); ++index) {
        REQUIRE(selected.exact.suitability.scores[index] ==
                Catch::Approx(expected.suitability.scores[index]));
    }
    REQUIRE(selected.exact.biome.primary == expected.biome.primary);
    REQUIRE(selected.exact.biome.secondary == expected.biome.secondary);
    REQUIRE(selected.exact.biome.transition == Catch::Approx(expected.biome.transition));

    const FarTerrainSource source = FarTerrainMesher::generatorGeometrySource(generator);
    const FarSurfaceSample scalar =
        source.sample(selected.x, selected.z, worldgen::SurfaceFootprint::BLOCK_1);
    std::array<FarSurfaceSample, 1> grid{};
    source.sampleGrid(selected.x, selected.z, 1, 1, worldgen::SurfaceFootprint::BLOCK_1, grid);
    REQUIRE(scalar.geometry.terrainHeight == Catch::Approx(emittedTop));
    REQUIRE(grid.front().geometry.terrainHeight == Catch::Approx(emittedTop));
}

TEST_CASE("V4 ordinary river stages map to immutable eighth-block tops",
          "[learned][v4][hydrology][water][fluid][regression]") {
    struct Case {
        double analyticalStage = 0.0;
        double emittedStage = 0.0;
        uint8_t level = 0;
    };
    constexpr std::array cases{
        Case{84.04, 84.0, 0},  Case{84.08, 84.125, 7}, Case{84.33, 84.375, 5},
        Case{84.74, 84.75, 2}, Case{84.94, 85.0, 0},
    };

    for (const Case& test : cases) {
        worldgen::HydrologySample hydrology;
        hydrology.river = true;
        hydrology.waterBodyId = 17;
        hydrology.waterSurface = test.analyticalStage;
        hydrology.surfaceElevation = 83.0;
        hydrology.channelDepth = 0.5;
        worldgen::quantizeGeneratedRiverSurface(hydrology);

        CAPTURE(test.analyticalStage, hydrology.waterSurface, hydrology.generatedFluidLevel);
        REQUIRE(hydrology.waterSurface == test.emittedStage);
        REQUIRE(hydrology.generatedFluidLevel == test.level);
        REQUIRE(hydrology.surfaceElevation <= hydrology.waterSurface - 0.125);
        REQUIRE(hydrology.channelDepth >= hydrology.waterSurface - hydrology.surfaceElevation);

        worldgen::SurfaceSample surface;
        surface.hydrology = hydrology;
        surface.terrainHeight = hydrology.surfaceElevation;
        surface.emittedTerrainHeight = 83.0F;
        surface.waterSurface = hydrology.waterSurface;
        const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(surface);
        REQUIRE(fluid.wet);
        REQUIRE(fluid.visibleSurface == test.emittedStage);
        REQUIRE(fluid.topState.level() == test.level);
        REQUIRE(fluid.topState.isSource() == (test.level == 0));
        REQUIRE_FALSE(fluid.topState.isFalling());
    }

    worldgen::HydrologySample lake;
    lake.river = true;
    lake.lake = true;
    lake.waterSurface = 84.33;
    worldgen::quantizeGeneratedRiverSurface(lake);
    REQUIRE(lake.waterSurface == 84.33);
    REQUIRE(lake.generatedFluidLevel == 0);

    worldgen::HydrologySample fall;
    fall.river = true;
    fall.waterfall = true;
    fall.waterSurface = 84.33;
    fall.generatedFluidLevel = 7;
    worldgen::quantizeGeneratedRiverSurface(fall);
    REQUIRE(fall.waterSurface == 84.33);
    REQUIRE(fall.generatedFluidLevel == 7);
}

TEST_CASE("V4 partial river tops remain static across exact cubes and restart",
          "[learned][v4][hydrology][water][fluid][world][persistence][regression]") {
    constexpr uint64_t SEED = 42;
    constexpr int64_t ORIGIN_X = 1'088;
    constexpr int64_t ORIGIN_Z = -1'088;
    constexpr int EDGE = 257;
    TempDir directory("learned_static_partial_river");
    const GenerationIdentity identity = testIdentity(SEED);
    const std::filesystem::path terrainRoot =
        std::filesystem::path(directory.path()) / "terrain-authority-v1";
    const std::filesystem::path hydrologyRoot =
        std::filesystem::path(directory.path()) / "hydrology-authority-v1";

    auto backend = std::make_shared<HighFlowWaterfallTerrainBackend>();
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, terrainRoot, backend);
    auto context = std::make_shared<WorldGenerationContext>(identity, authority,
                                                            AuthorityQuality::FINAL, hydrologyRoot);
    auto generator = std::make_unique<ChunkGenerator>(SEED, context);

    std::vector<worldgen::SurfaceSample> native(static_cast<size_t>(EDGE * EDGE));
    std::vector<worldgen::SurfaceSample> exact(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration(
        [&] {
            generator->sampleNativeHydrologyGeometryGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE, EDGE,
                                                         native);
            return true;
        },
        60s);
    awaitGeneration(
        [&] {
            generator->sampleExactSurfaceGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, exact);
            return true;
        },
        60s);

    const auto index = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    const auto explicitFall = [](const worldgen::SurfaceSample& sample) {
        return sample.hydrology.waterfall &&
               sample.hydrology.transitionOwnerKind ==
                   worldgen::WaterTransitionKind::EXPLICIT_FALL &&
               sample.hydrology.transitionOwnerId != 0;
    };
    size_t nativeWet = 0;
    size_t exactWet = 0;
    size_t mismatchedWet = 0;
    size_t unsupportedWet = 0;
    size_t unownedStageJumps = 0;
    ColumnPos firstUnownedPosition{};
    ColumnPos firstUnownedNeighbor{};
    worldgen::SurfaceSample firstUnowned{};
    worldgen::SurfaceSample firstUnownedAdjacent{};
    std::optional<std::pair<int, int>> boundaryRoute;
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const worldgen::GeneratedFluidColumn nativeFluid =
                worldgen::generatedFluidColumn(native[index(x, z)]);
            const worldgen::SurfaceSample& sample = exact[index(x, z)];
            const worldgen::GeneratedFluidColumn fluid = worldgen::generatedFluidColumn(sample);
            nativeWet += nativeFluid.wet ? 1U : 0U;
            exactWet += fluid.wet ? 1U : 0U;
            mismatchedWet += nativeFluid.wet != fluid.wet ? 1U : 0U;
            unsupportedWet +=
                fluid.wet && fluid.visibleSurface <= worldgen::geometryTerrainHeight(sample) + 0.01
                    ? 1U
                    : 0U;

            for (const auto [offsetX, offsetZ] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + offsetX >= EDGE || z + offsetZ >= EDGE || !fluid.wet)
                    continue;
                const worldgen::SurfaceSample& neighbor = exact[index(x + offsetX, z + offsetZ)];
                const worldgen::GeneratedFluidColumn neighborFluid =
                    worldgen::generatedFluidColumn(neighbor);
                if (!neighborFluid.wet || explicitFall(sample) || explicitFall(neighbor))
                    continue;
                if (std::abs(fluid.visibleSurface - neighborFluid.visibleSurface) > 0.125001) {
                    if (unownedStageJumps == 0) {
                        firstUnownedPosition = {ORIGIN_X + x, ORIGIN_Z + z};
                        firstUnownedNeighbor = {firstUnownedPosition.x + offsetX,
                                                firstUnownedPosition.z + offsetZ};
                        firstUnowned = sample;
                        firstUnownedAdjacent = neighbor;
                    }
                    ++unownedStageJumps;
                }
            }

            if (boundaryRoute || x + 1 >= EDGE || z == 0 || z + 1 >= EDGE ||
                Chunk::worldToLocal(ORIGIN_X + x) != CHUNK_EDGE - 1 || !sample.hydrology.river ||
                sample.hydrology.waterfall || !fluid.wet ||
                sample.hydrology.generatedFluidLevel == 0) {
                continue;
            }
            const worldgen::SurfaceSample& east = exact[index(x + 1, z)];
            const worldgen::GeneratedFluidColumn eastFluid = worldgen::generatedFluidColumn(east);
            const int surfaceY =
                static_cast<int>(std::llround(worldgen::geometryTerrainHeight(sample))) - 1;
            if (east.hydrology.river && !east.hydrology.waterfall && eastFluid.wet &&
                east.hydrology.generatedFluidLevel > 0 &&
                east.hydrology.waterBodyId == sample.hydrology.waterBodyId &&
                std::abs(eastFluid.visibleSurface - fluid.visibleSurface) <= 0.125001 &&
                fluid.topY - surfaceY >= 2) {
                boundaryRoute = std::pair{x, z};
            }
        }
    }
    CAPTURE(nativeWet, exactWet, mismatchedWet, unsupportedWet, unownedStageJumps,
            firstUnownedPosition.x, firstUnownedPosition.z, firstUnownedNeighbor.x,
            firstUnownedNeighbor.z, firstUnowned.waterSurface, firstUnownedAdjacent.waterSurface,
            firstUnowned.hydrology.ocean, firstUnowned.hydrology.river, firstUnowned.hydrology.lake,
            firstUnowned.hydrology.waterfall, firstUnowned.hydrology.flowDirection.x,
            firstUnowned.hydrology.flowDirection.z, firstUnowned.hydrology.channelDistance,
            firstUnowned.hydrology.channelWidth, firstUnownedAdjacent.hydrology.ocean,
            firstUnownedAdjacent.hydrology.river, firstUnownedAdjacent.hydrology.lake,
            firstUnownedAdjacent.hydrology.waterfall,
            firstUnownedAdjacent.hydrology.flowDirection.x,
            firstUnownedAdjacent.hydrology.flowDirection.z,
            firstUnownedAdjacent.hydrology.channelDistance,
            firstUnownedAdjacent.hydrology.channelWidth);
    REQUIRE(nativeWet > 1'000);
    REQUIRE(exactWet == nativeWet);
    REQUIRE(mismatchedWet == 0);
    REQUIRE(unsupportedWet == 0);
    REQUIRE(unownedStageJumps == 0);
    REQUIRE(boundaryRoute);

    const std::array<ColumnPos, 2> routePositions{
        ColumnPos{ORIGIN_X + boundaryRoute->first, ORIGIN_Z + boundaryRoute->second},
        ColumnPos{ORIGIN_X + boundaryRoute->first + 1, ORIGIN_Z + boundaryRoute->second},
    };
    struct EmittedColumn {
        worldgen::SurfaceSample surface;
        int surfaceY = WORLD_MIN_Y;
        std::vector<std::pair<BlockType, FluidState>> cells;
    };
    const auto captureColumns = [&](ChunkGenerator& source) {
        std::map<std::tuple<int64_t, int32_t, int64_t>, std::unique_ptr<Chunk>> cubes;
        const auto cellAt = [&](ColumnPos position, int y) {
            const ChunkPos cubePosition{Chunk::worldToChunk(position.x), Chunk::worldToChunkY(y),
                                        Chunk::worldToChunk(position.z)};
            const auto key = std::tuple{cubePosition.x, cubePosition.y, cubePosition.z};
            auto found = cubes.find(key);
            if (found == cubes.end()) {
                auto cube = std::make_unique<Chunk>(cubePosition);
                awaitGeneration(
                    [&] {
                        source.generate(*cube);
                        return true;
                    },
                    30s);
                found = cubes.emplace(key, std::move(cube)).first;
            }
            return std::pair{
                found->second->getBlock(Chunk::worldToLocal(position.x), Chunk::worldToLocalY(y),
                                        Chunk::worldToLocal(position.z)),
                found->second->getFluidState(Chunk::worldToLocal(position.x),
                                             Chunk::worldToLocalY(y),
                                             Chunk::worldToLocal(position.z)),
            };
        };

        std::array<EmittedColumn, routePositions.size()> result;
        for (size_t routeIndex = 0; routeIndex < routePositions.size(); ++routeIndex) {
            const ColumnPos position = routePositions[routeIndex];
            EmittedColumn& emitted = result[routeIndex];
            emitted.surface = awaitGeneration(
                [&] { return source.sampleExactSurface(position.x, position.z); }, 30s);
            const auto plan = awaitGeneration(
                [&] {
                    return source.getColumnPlan(
                        {Chunk::worldToChunk(position.x), Chunk::worldToChunk(position.z)});
                },
                30s);
            emitted.surfaceY =
                plan->surfaceY(Chunk::worldToLocal(position.x), Chunk::worldToLocal(position.z));
            const worldgen::GeneratedFluidColumn fluid =
                worldgen::generatedFluidColumn(emitted.surface);
            for (int y = emitted.surfaceY; y <= fluid.topY; ++y)
                emitted.cells.push_back(cellAt(position, y));
        }
        return result;
    };

    const auto firstEmission = captureColumns(*generator);
    for (size_t routeIndex = 0; routeIndex < firstEmission.size(); ++routeIndex) {
        const EmittedColumn& emitted = firstEmission[routeIndex];
        const worldgen::GeneratedFluidColumn fluid =
            worldgen::generatedFluidColumn(emitted.surface);
        CAPTURE(routePositions[routeIndex].x, routePositions[routeIndex].z, emitted.surfaceY,
                fluid.topY, fluid.visibleSurface, emitted.surface.hydrology.generatedFluidLevel);
        REQUIRE(fluid.wet);
        REQUIRE_FALSE(fluid.topState.isSource());
        REQUIRE(isSolid(emitted.cells.front().first));
        REQUIRE(emitted.cells.back().first == BlockType::WATER);
        REQUIRE(emitted.cells.back().second == fluid.topState);
        for (size_t cell = 1; cell + 1 < emitted.cells.size(); ++cell) {
            REQUIRE(emitted.cells[cell].first == BlockType::WATER);
            REQUIRE(emitted.cells[cell].second.isSource());
        }
    }

    const auto awaitWorldChunk = [](World& world, ChunkPos position) {
        const auto deadline = std::chrono::steady_clock::now() + 30s;
        while (std::chrono::steady_clock::now() < deadline) {
            if (std::shared_ptr<Chunk> cube = world.getChunk(position))
                return cube;
            std::this_thread::sleep_for(1ms);
        }
        return std::shared_ptr<Chunk>{};
    };
    {
        World world(SEED, 4, 128, context);
        REQUIRE(world.getPendingFluidCount() == 0);
        for (size_t routeIndex = 0; routeIndex < routePositions.size(); ++routeIndex) {
            const ColumnPos position = routePositions[routeIndex];
            const EmittedColumn& emitted = firstEmission[routeIndex];
            const worldgen::GeneratedFluidColumn fluid =
                worldgen::generatedFluidColumn(emitted.surface);
            for (int32_t section = Chunk::worldToChunkY(emitted.surfaceY);
                 section <= Chunk::worldToChunkY(fluid.topY); ++section) {
                REQUIRE(awaitWorldChunk(world, {Chunk::worldToChunk(position.x), section,
                                                Chunk::worldToChunk(position.z)}));
            }
            const FluidCell top = world.readFluidCell({position.x, fluid.topY, position.z});
            REQUIRE(top.loaded);
            REQUIRE(top.block == BlockType::WATER);
            REQUIRE(top.state == fluid.topState);
            REQUIRE(world.getFluidHeightIfLoaded(position.x, fluid.topY, position.z) ==
                    Catch::Approx(fluidSurfaceHeight(fluid.topState)));
        }
        REQUIRE(world.getPendingFluidCount() == 0);
        REQUIRE(world.tickFluids(1.0) == 0);
        REQUIRE(world.getPendingFluidCount() == 0);

        const ColumnPos edited = routePositions.front();
        const int editedY = worldgen::generatedFluidColumn(firstEmission.front().surface).topY + 1;
        world.setBlock(edited.x, editedY, edited.z, BlockType::WATER);
        REQUIRE(world.getPendingFluidCount() > 0);
    }

    generator.reset();
    context.reset();
    authority.reset();
    const uint64_t firstBackendCalls = backend->calls();
    backend.reset();
    REQUIRE(firstBackendCalls > 0);

    auto restartedBackend = std::make_shared<HighFlowWaterfallTerrainBackend>();
    auto restartedAuthority =
        std::make_shared<CachedTerrainAuthority>(identity, terrainRoot, restartedBackend);
    auto restartedContext = std::make_shared<WorldGenerationContext>(
        identity, restartedAuthority, AuthorityQuality::FINAL, hydrologyRoot);
    ChunkGenerator restarted(SEED, restartedContext);
    const auto restartedEmission = captureColumns(restarted);
    REQUIRE(restartedBackend->calls() == 0);
    for (size_t routeIndex = 0; routeIndex < firstEmission.size(); ++routeIndex) {
        CAPTURE(routePositions[routeIndex].x, routePositions[routeIndex].z);
        REQUIRE(restartedEmission[routeIndex].surface.hydrology.waterBodyId ==
                firstEmission[routeIndex].surface.hydrology.waterBodyId);
        REQUIRE(restartedEmission[routeIndex].surface.waterSurface ==
                firstEmission[routeIndex].surface.waterSurface);
        REQUIRE(restartedEmission[routeIndex].surface.hydrology.generatedFluidLevel ==
                firstEmission[routeIndex].surface.hydrology.generatedFluidLevel);
        REQUIRE(restartedEmission[routeIndex].surfaceY == firstEmission[routeIndex].surfaceY);
        REQUIRE(restartedEmission[routeIndex].cells == firstEmission[routeIndex].cells);
    }
}

TEST_CASE("Final exact terrain preserves routed water and supported lateral banks",
          "[learned][v4][hydrology][water][exact][bank][regression]") {
    CanonicalWaterRegressionFixture& fixture = canonicalWaterRegressionFixture();
    const CanonicalWaterfallScene& scene = canonicalWaterfallScene();
    constexpr int RADIUS = 48;
    constexpr int EDGE = 97;
    const int64_t ORIGIN_X = scene.highX - RADIUS;
    const int64_t ORIGIN_Z = scene.highZ - RADIUS;
    std::vector<worldgen::SurfaceSample> grid(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration(
        [&] {
            fixture.generator.sampleExactSurfaceGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, grid);
            return true;
        },
        30s);

    const int highLocalX = static_cast<int>(scene.highX - ORIGIN_X);
    const int highLocalZ = static_cast<int>(scene.highZ - ORIGIN_Z);
    REQUIRE(highLocalX >= 0);
    REQUIRE(highLocalX < EDGE);
    REQUIRE(highLocalZ >= 0);
    REQUIRE(highLocalZ < EDGE);
    const int highIndex = highLocalZ * EDGE + highLocalX;
    const int64_t highX = scene.highX;
    const int64_t highZ = scene.highZ;
    const int64_t lowX = scene.lowX;
    const int64_t lowZ = scene.lowZ;
    const worldgen::HydrologySample canonicalLow =
        awaitGeneration([&] { return fixture.macro.sampleHydrology(lowX, lowZ); }, 30s);
    const worldgen::HydrologySample canonicalHigh =
        awaitGeneration([&] { return fixture.macro.sampleHydrology(highX, highZ); }, 30s);
    const worldgen::SurfaceSample exactHigh = grid[static_cast<size_t>(highIndex)];
    const worldgen::SurfaceSample exactLow =
        awaitGeneration([&] { return fixture.generator.sampleExactSurface(lowX, lowZ); }, 30s);
    const worldgen::SurfaceSample scalarHigh =
        awaitGeneration([&] { return fixture.generator.sampleExactSurface(highX, highZ); }, 30s);
    const std::array<ColumnPos, 2> waterPositions{
        ColumnPos{lowX, lowZ},
        ColumnPos{highX, highZ},
    };
    std::array<worldgen::HydrologySample, waterPositions.size()> generatedAuthority{};
    std::array<worldgen::SurfaceSample, waterPositions.size()> generatedGeometry{};
    std::array<worldgen::SurfaceSample, waterPositions.size()> farBlockSamples{};
    awaitGeneration(
        [&] {
            fixture.generator.sampleGeneratedWaterAuthorityPoints(waterPositions,
                                                                  generatedAuthority);
            return true;
        },
        30s);
    awaitGeneration(
        [&] {
            fixture.generator.sampleGeneratedWaterGeometryPoints(waterPositions, generatedGeometry);
            return true;
        },
        30s);
    awaitGeneration(
        [&] {
            fixture.generator.sampleFarSurfacePoints(
                waterPositions, worldgen::SurfaceFootprint::BLOCK_1, farBlockSamples);
            return true;
        },
        30s);

    const worldgen::GeneratedFluidColumn lowFluid = worldgen::generatedFluidColumn(exactLow);
    const worldgen::GeneratedFluidColumn highFluid = worldgen::generatedFluidColumn(exactHigh);
    CAPTURE(highX, highZ, lowX, lowZ, canonicalLow.surfaceElevation, canonicalLow.waterSurface,
            canonicalHigh.surfaceElevation, canonicalHigh.waterSurface, exactLow.terrainHeight,
            exactLow.waterSurface, exactHigh.terrainHeight, exactHigh.waterSurface,
            exactHigh.hydrology.transitionOwnerId);
    REQUIRE(canonicalLow.ocean);
    REQUIRE(canonicalHigh.waterfall);
    // The unique integer anchor can lie on the receiving ocean's categorical
    // side of a wide fall. Explicit transition ownership, rather than the
    // mutually exclusive standing/river label, is the cross-path invariant.
    REQUIRE(canonicalHigh.waterfallAnchor);
    REQUIRE(canonicalHigh.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(canonicalHigh.generatedFluidLevel == 7);
    REQUIRE(canonicalHigh.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL);
    REQUIRE(canonicalHigh.transitionOwnerId != 0);
    REQUIRE(exactLow.hydrology.ocean);
    REQUIRE(lowFluid.wet);
    REQUIRE(exactLow.terrainHeight < exactLow.waterSurface - 1.0);
    REQUIRE(highFluid.wet);
    REQUIRE(exactHigh.hydrology.waterfall);
    REQUIRE(exactHigh.hydrology.waterfallAnchor);
    REQUIRE(exactHigh.hydrology.waterBodyId == canonicalHigh.waterBodyId);
    REQUIRE(exactHigh.hydrology.generatedFluidLevel == canonicalHigh.generatedFluidLevel);
    REQUIRE(exactHigh.hydrology.transitionOwnerKind == canonicalHigh.transitionOwnerKind);
    REQUIRE(exactHigh.hydrology.transitionOwnerId == canonicalHigh.transitionOwnerId);
    REQUIRE(exactHigh.hydrology.waterfallTop >= exactHigh.hydrology.waterfallBottom + 0.5);
    REQUIRE(scalarHigh.terrainHeight == exactHigh.terrainHeight);
    REQUIRE(scalarHigh.hydrology.transitionOwnerId == exactHigh.hydrology.transitionOwnerId);

    // Exact cube generation, canonical far-water geometry, and the block
    // footprint far path may refine terrain contact, but they cannot invent a
    // fall or clear the native router's wet body ownership.
    const std::array<worldgen::HydrologySample, waterPositions.size()> nativeAuthority{
        canonicalLow,
        canonicalHigh,
    };
    const std::array<worldgen::HydrologySample, waterPositions.size()> exactAuthority{
        exactLow.hydrology,
        exactHigh.hydrology,
    };
    for (size_t index = 0; index < waterPositions.size(); ++index) {
        const worldgen::HydrologySample& native = nativeAuthority[index];
        const auto requireNativeWater = [&](const worldgen::HydrologySample& sampled) {
            REQUIRE(sampled.waterBodyId == native.waterBodyId);
            REQUIRE(sampled.waterfall == native.waterfall);
            REQUIRE(sampled.generatedFluidLevel == native.generatedFluidLevel);
            REQUIRE(sampled.transitionOwnerKind == native.transitionOwnerKind);
            REQUIRE(sampled.transitionOwnerId == native.transitionOwnerId);
        };
        requireNativeWater(generatedAuthority[index]);
        requireNativeWater(generatedGeometry[index].hydrology);
        requireNativeWater(farBlockSamples[index].hydrology);
        // The native router owns body identity and stage, while the generated
        // water authority owns the emitted bed used by exact cubes. A v4 far
        // query must not retain a stale fractional native bed beneath a wet
        // column.
        REQUIRE(generatedAuthority[index].surfaceElevation ==
                exactAuthority[index].surfaceElevation);
        REQUIRE(worldgen::generatedFluidColumn(generatedGeometry[index]).wet);
        REQUIRE(worldgen::generatedFluidColumn(farBlockSamples[index]).wet);
    }
}

TEST_CASE("Canonical chasm falls use one receiving-stage curtain",
          "[learned][v4][hydrology][water][waterfall][chasm][regression]") {
    CanonicalWaterRegressionFixture& fixture = canonicalWaterRegressionFixture();
    const CanonicalWaterfallScene& scene = canonicalWaterfallScene();
    const worldgen::HydrologySample fall = awaitGeneration(
        [&] { return fixture.macro.sampleHydrology(scene.highX, scene.highZ); }, 30s);
    const worldgen::HydrologySample receiver =
        awaitGeneration([&] { return fixture.macro.sampleHydrology(scene.lowX, scene.lowZ); }, 30s);

    CAPTURE(scene.highX, scene.highZ, scene.lowX, scene.lowZ);
    REQUIRE(fall.waterfall);
    REQUIRE(fall.waterfallAnchor);
    REQUIRE(fall.transitionOwnerKind == worldgen::WaterTransitionKind::EXPLICIT_FALL);
    REQUIRE(fall.transitionOwnerId != 0);
    REQUIRE(receiver.ocean);
    REQUIRE(receiver.waterSurface == Catch::Approx(SEA_LEVEL).margin(1.0e-6));
    REQUIRE(fall.waterfallBottom == Catch::Approx(receiver.waterSurface).margin(1.0e-6));
    REQUIRE(fall.waterSurface == Catch::Approx(fall.waterfallBottom).margin(1.0e-6));

    constexpr int RADIUS = 24;
    constexpr int EDGE = RADIUS * 2 + 1;
    std::vector<worldgen::SurfaceSample> exact(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration(
        [&] {
            fixture.generator.sampleExactSurfaceGrid(scene.highX - RADIUS, scene.highZ - RADIUS, 1,
                                                     EDGE, exact);
            return true;
        },
        30s);

    const double flowLength = std::hypot(fall.flowDirection.x, fall.flowDirection.z);
    REQUIRE(flowLength > 1.0e-9);
    const double flowX = fall.flowDirection.x / flowLength;
    const double flowZ = fall.flowDirection.z / flowLength;
    size_t curtainColumns = 0;
    size_t anchors = 0;
    double minimumAlong = std::numeric_limits<double>::infinity();
    double maximumAlong = -std::numeric_limits<double>::infinity();
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const worldgen::SurfaceSample& sample = exact[static_cast<size_t>(z * EDGE + x)];
            if (sample.hydrology.transitionOwnerId != fall.transitionOwnerId)
                continue;
            CAPTURE(x, z, sample.waterSurface, sample.hydrology.waterSurface,
                    sample.hydrology.ocean, sample.hydrology.river, sample.hydrology.waterfallTop,
                    sample.hydrology.waterfallBottom);
            ++curtainColumns;
            anchors += sample.hydrology.waterfallAnchor ? 1U : 0U;
            REQUIRE(sample.hydrology.waterfall);
            REQUIRE(sample.hydrology.transitionOwnerKind ==
                    worldgen::WaterTransitionKind::EXPLICIT_FALL);
            REQUIRE(sample.hydrology.waterfallTop ==
                    Catch::Approx(fall.waterfallTop).margin(1.0e-6));
            REQUIRE(sample.hydrology.waterfallBottom ==
                    Catch::Approx(receiver.waterSurface).margin(1.0e-6));
            REQUIRE(sample.waterSurface ==
                    Catch::Approx(sample.hydrology.waterfallBottom).margin(1.0e-6));
            const double offsetX = static_cast<double>(x - RADIUS);
            const double offsetZ = static_cast<double>(z - RADIUS);
            const double along = offsetX * flowX + offsetZ * flowZ;
            minimumAlong = std::min(minimumAlong, along);
            maximumAlong = std::max(maximumAlong, along);
        }
    }
    CAPTURE(curtainColumns, anchors, minimumAlong, maximumAlong, fall.waterfallWidth,
            fall.waterfallTop, fall.waterfallBottom);
    REQUIRE(curtainColumns > 0);
    REQUIRE(anchors == 1);
    // The source-plane clip removes the wide ribbon's round upstream cap. A
    // bowed centerline can rotate the sample-local flow slightly away from
    // the receiver axis, so allow one block of projected lateral skew beyond
    // the half-open four-block receiver edge. The old round cap spanned almost
    // fourteen blocks in this fixed scene.
    REQUIRE(maximumAlong - minimumAlong <= worldgen::NATIVE_HYDROLOGY_RASTER_SPACING + 1.001);

    std::map<int32_t, std::unique_ptr<Chunk>> cubes;
    const auto cellAt = [&](int y) {
        const int32_t section = Chunk::worldToChunkY(y);
        auto found = cubes.find(section);
        if (found == cubes.end()) {
            auto cube = std::make_unique<Chunk>(ChunkPos{Chunk::worldToChunk(scene.highX), section,
                                                         Chunk::worldToChunk(scene.highZ)});
            awaitGeneration(
                [&] {
                    fixture.generator.generate(*cube);
                    return true;
                },
                30s);
            found = cubes.emplace(section, std::move(cube)).first;
        }
        return std::pair{
            found->second->getBlock(Chunk::worldToLocal(scene.highX), Chunk::worldToLocalY(y),
                                    Chunk::worldToLocal(scene.highZ)),
            found->second->getFluidState(Chunk::worldToLocal(scene.highX), Chunk::worldToLocalY(y),
                                         Chunk::worldToLocal(scene.highZ)),
        };
    };

    const int receivingTopY = static_cast<int>(std::ceil(fall.waterfallBottom)) - 1;
    const int fallingStartY = static_cast<int>(std::ceil(fall.waterfallBottom));
    const int fallingTopY = static_cast<int>(std::ceil(fall.waterfallTop)) - 1;
    const auto receivingCell = cellAt(receivingTopY);
    REQUIRE(receivingCell.first == BlockType::WATER);
    REQUIRE_FALSE(receivingCell.second.isFalling());
    for (int y = fallingStartY; y < fallingTopY; ++y) {
        const auto cell = cellAt(y);
        REQUIRE(cell.first == BlockType::WATER);
        REQUIRE(cell.second == FluidState::falling(7));
    }
    const auto lip = cellAt(fallingTopY);
    REQUIRE(lip.first == BlockType::WATER);
    REQUIRE(lip.second == FluidState::flowing(7));
}

TEST_CASE("Exact and native water share the align-corners block-center phase",
          "[learned][v4][hydrology][water][phase][regression]") {
    CanonicalWaterRegressionFixture& fixture = canonicalWaterRegressionFixture();
    constexpr int64_t ORIGIN_X = 1'016;
    constexpr int64_t ORIGIN_Z = -856;
    constexpr int EDGE = 17;
    std::array<worldgen::SurfaceSample, EDGE * EDGE> native{};
    std::array<worldgen::SurfaceSample, EDGE * EDGE> exact{};
    awaitGeneration(
        [&] {
            fixture.generator.sampleNativeHydrologyGeometryGrid(ORIGIN_X, ORIGIN_Z, 1, 1, EDGE,
                                                                EDGE, native);
            return true;
        },
        30s);
    awaitGeneration(
        [&] {
            fixture.generator.sampleExactSurfaceGrid(ORIGIN_X, ORIGIN_Z, 1, EDGE, exact);
            return true;
        },
        30s);

    for (size_t index = 0; index < native.size(); ++index) {
        CAPTURE(index);
        REQUIRE(exact[index].hydrology.ocean == native[index].hydrology.ocean);
        REQUIRE(exact[index].hydrology.river == native[index].hydrology.river);
        REQUIRE(exact[index].hydrology.lake == native[index].hydrology.lake);
        REQUIRE(exact[index].hydrology.waterfall == native[index].hydrology.waterfall);
        REQUIRE(exact[index].hydrology.waterBodyId == native[index].hydrology.waterBodyId);
        REQUIRE(exact[index].hydrology.waterSurface ==
                Catch::Approx(native[index].hydrology.waterSurface).margin(1.0e-6));
        REQUIRE(worldgen::generatedFluidColumn(exact[index]).wet ==
                worldgen::generatedFluidColumn(native[index]).wet);
    }
}

TEST_CASE("Generated channel beds and visible water remain continuous in the second scene",
          "[learned][v4][hydrology][water][continuity][bed][regression]") {
    CanonicalWaterRegressionFixture& fixture = canonicalWaterRegressionFixture();
    const CanonicalWaterfallScene& scene = canonicalWaterfallScene();
    const int64_t CENTER_X = scene.highX;
    const int64_t CENTER_Z = scene.highZ;
    constexpr int RADIUS = 24;
    constexpr int EDGE = RADIUS * 2 + 1;
    std::vector<worldgen::SurfaceSample> exact(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration(
        [&] {
            fixture.generator.sampleExactSurfaceGrid(CENTER_X - RADIUS, CENTER_Z - RADIUS, 1, EDGE,
                                                     exact);
            return true;
        },
        60s);
    std::vector<worldgen::HydrologySample> canonical(static_cast<size_t>(EDGE * EDGE));
    awaitGeneration(
        [&] {
            fixture.macro.sampleHydrologyGrid(CENTER_X - RADIUS, CENTER_Z - RADIUS, 1, 1, EDGE,
                                              EDGE, canonical);
            return true;
        },
        60s);
    size_t wet = 0;
    size_t adjacentWet = 0;
    size_t untagged = 0;
    size_t explicitFallFaces = 0;
    size_t unsupportedBanks = 0;
    std::array<bool, EDGE * EDGE> unsupportedEast{};
    std::array<bool, EDGE * EDGE> unsupportedSouth{};
    size_t supportableCanonical = 0;
    size_t deletedCanonical = 0;
    double maximumStep = 0.0;
    ColumnPos worstPosition{};
    ColumnPos worstNeighbor{};
    worldgen::SurfaceSample worst{};
    worldgen::SurfaceSample worstAdjacent{};
    double maximumUntaggedStep = 0.0;
    ColumnPos worstUntaggedPosition{};
    ColumnPos worstUntaggedNeighbor{};
    worldgen::SurfaceSample worstUntagged{};
    worldgen::SurfaceSample worstUntaggedAdjacent{};
    ColumnPos firstUnsupportedPosition{};
    ColumnPos firstUnsupportedNeighbor{};
    worldgen::SurfaceSample firstUnsupportedWet{};
    worldgen::SurfaceSample firstUnsupportedDry{};
    worldgen::HydrologySample firstUnsupportedCanonicalWet{};
    worldgen::HydrologySample firstUnsupportedCanonicalDry{};
    const auto index = [](int x, int z) { return static_cast<size_t>(z * EDGE + x); };
    for (int z = 0; z < EDGE; ++z) {
        for (int x = 0; x < EDGE; ++x) {
            const auto& sample = exact[index(x, z)];
            const auto fluid = worldgen::generatedFluidColumn(sample);
            wet += fluid.wet ? 1U : 0U;
            const worldgen::HydrologySample& authority = canonical[index(x, z)];
            if ((authority.ocean || authority.river || authority.lake) &&
                authority.waterSurface > sample.terrainHeight + 0.01) {
                ++supportableCanonical;
                deletedCanonical += fluid.wet ? 0U : 1U;
            }
            for (const auto [dx, dz] : {std::pair{1, 0}, std::pair{0, 1}}) {
                if (x + dx >= EDGE || z + dz >= EDGE)
                    continue;
                const auto& neighbor = exact[index(x + dx, z + dz)];
                const auto neighborFluid = worldgen::generatedFluidColumn(neighbor);
                if (fluid.wet && neighborFluid.wet) {
                    ++adjacentWet;
                    const double step =
                        std::abs(fluid.visibleSurface - neighborFluid.visibleSurface);
                    if (step > maximumStep) {
                        maximumStep = step;
                        worstPosition = {CENTER_X - RADIUS + x, CENTER_Z - RADIUS + z};
                        worstNeighbor = {worstPosition.x + dx, worstPosition.z + dz};
                        worst = sample;
                        worstAdjacent = neighbor;
                    }
                    const auto explicitFall = [](const worldgen::SurfaceSample& value) {
                        return value.hydrology.waterfall &&
                               value.hydrology.transitionOwnerKind ==
                                   worldgen::WaterTransitionKind::EXPLICIT_FALL &&
                               value.hydrology.transitionOwnerId != 0;
                    };
                    const bool tagged = explicitFall(sample) || explicitFall(neighbor);
                    explicitFallFaces += tagged ? 1U : 0U;
                    if (!tagged && step > 0.125001) {
                        ++untagged;
                        if (step > maximumUntaggedStep) {
                            maximumUntaggedStep = step;
                            worstUntaggedPosition = {
                                CENTER_X - RADIUS + x,
                                CENTER_Z - RADIUS + z,
                            };
                            worstUntaggedNeighbor = {
                                worstUntaggedPosition.x + dx,
                                worstUntaggedPosition.z + dz,
                            };
                            worstUntagged = sample;
                            worstUntaggedAdjacent = neighbor;
                        }
                    }
                } else if (fluid.wet != neighborFluid.wet) {
                    const auto& wetSample = fluid.wet ? sample : neighbor;
                    const auto& drySample = fluid.wet ? neighbor : sample;
                    const auto wetFluid = fluid.wet ? fluid : neighborFluid;
                    const double normalFlow = dx != 0
                                                  ? std::abs(wetSample.hydrology.flowDirection.x)
                                                  : std::abs(wetSample.hydrology.flowDirection.z);
                    const double tangentialFlow =
                        dx != 0 ? std::abs(wetSample.hydrology.flowDirection.z)
                                : std::abs(wetSample.hydrology.flowDirection.x);
                    const bool standing = std::hypot(wetSample.hydrology.flowDirection.x,
                                                     wetSample.hydrology.flowDirection.z) <= 1.0e-6;
                    const bool explicitWetFall = wetSample.hydrology.waterfall &&
                                                 wetSample.hydrology.transitionOwnerKind ==
                                                     worldgen::WaterTransitionKind::EXPLICIT_FALL &&
                                                 wetSample.hydrology.transitionOwnerId != 0;
                    if (!explicitWetFall && (standing || normalFlow + 1.0e-6 < tangentialFlow) &&
                        wetFluid.visibleSurface - drySample.terrainHeight > 1.000001) {
                        if (unsupportedBanks == 0) {
                            firstUnsupportedPosition = {
                                CENTER_X - RADIUS + x,
                                CENTER_Z - RADIUS + z,
                            };
                            firstUnsupportedNeighbor = {
                                firstUnsupportedPosition.x + dx,
                                firstUnsupportedPosition.z + dz,
                            };
                            firstUnsupportedWet = wetSample;
                            firstUnsupportedDry = drySample;
                            firstUnsupportedCanonicalWet =
                                canonical[index(fluid.wet ? x : x + dx, fluid.wet ? z : z + dz)];
                            firstUnsupportedCanonicalDry =
                                canonical[index(fluid.wet ? x + dx : x, fluid.wet ? z + dz : z)];
                        }
                        ++unsupportedBanks;
                        if (dx != 0)
                            unsupportedEast[index(x, z)] = true;
                        else
                            unsupportedSouth[index(x, z)] = true;
                    }
                }
            }
        }
    }
    const auto explicitFall = [](const worldgen::SurfaceSample& value) {
        return value.hydrology.waterfall &&
               value.hydrology.transitionOwnerKind ==
                   worldgen::WaterTransitionKind::EXPLICIT_FALL &&
               value.hydrology.transitionOwnerId != 0;
    };
    const auto discontinuousFloor = [&](int firstX, int firstZ, int secondX, int secondZ) {
        const worldgen::SurfaceSample& first = exact[index(firstX, firstZ)];
        const worldgen::SurfaceSample& second = exact[index(secondX, secondZ)];
        const bool bothWet =
            worldgen::generatedFluidColumn(first).wet && worldgen::generatedFluidColumn(second).wet;
        const bool openOceanBathymetry = first.hydrology.ocean && second.hydrology.ocean;
        return bothWet && !openOceanBathymetry && !explicitFall(first) && !explicitFall(second) &&
               std::abs(first.terrainHeight - second.terrainHeight) > 4.001;
    };
    size_t longCardinalFloorRuns = 0;
    size_t maximumUnsupportedBankRun = 0;
    for (int faceX = 1; faceX < EDGE; ++faceX) {
        int run = 0;
        for (int z = 0; z < EDGE; ++z) {
            if (discontinuousFloor(faceX - 1, z, faceX, z)) {
                ++run;
            } else {
                longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }
    for (int faceZ = 1; faceZ < EDGE; ++faceZ) {
        int run = 0;
        for (int x = 0; x < EDGE; ++x) {
            if (discontinuousFloor(x, faceZ - 1, x, faceZ)) {
                ++run;
            } else {
                longCardinalFloorRuns += run >= 8 ? 1U : 0U;
                run = 0;
            }
        }
        longCardinalFloorRuns += run >= 8 ? 1U : 0U;
    }
    for (int x = 0; x + 1 < EDGE; ++x) {
        size_t run = 0;
        for (int z = 0; z < EDGE; ++z) {
            if (unsupportedEast[index(x, z)]) {
                maximumUnsupportedBankRun = std::max(maximumUnsupportedBankRun, ++run);
            } else {
                run = 0;
            }
        }
    }
    for (int z = 0; z + 1 < EDGE; ++z) {
        size_t run = 0;
        for (int x = 0; x < EDGE; ++x) {
            if (unsupportedSouth[index(x, z)]) {
                maximumUnsupportedBankRun = std::max(maximumUnsupportedBankRun, ++run);
            } else {
                run = 0;
            }
        }
    }

    CAPTURE(
        wet, adjacentWet, supportableCanonical, deletedCanonical, unsupportedBanks,
        maximumUnsupportedBankRun, untagged, maximumUntaggedStep, explicitFallFaces,
        longCardinalFloorRuns, maximumStep, worstPosition.x, worstPosition.z, worstNeighbor.x,
        worstNeighbor.z, worst.waterSurface, worstAdjacent.waterSurface, worstUntaggedPosition.x,
        worstUntaggedPosition.z, worstUntaggedNeighbor.x, worstUntaggedNeighbor.z,
        worstUntagged.waterSurface, worstUntaggedAdjacent.waterSurface,
        worstUntagged.hydrology.flowDirection.x, worstUntagged.hydrology.flowDirection.z,
        worstUntagged.hydrology.ocean, worstUntagged.hydrology.river, worstUntagged.hydrology.lake,
        worstUntagged.hydrology.waterfall, worstUntagged.hydrology.transitionOwnerId,
        worstUntaggedAdjacent.hydrology.flowDirection.x,
        worstUntaggedAdjacent.hydrology.flowDirection.z, worstUntaggedAdjacent.hydrology.ocean,
        worstUntaggedAdjacent.hydrology.river, worstUntaggedAdjacent.hydrology.lake,
        worstUntaggedAdjacent.hydrology.waterfall,
        worstUntaggedAdjacent.hydrology.transitionOwnerId, firstUnsupportedPosition.x,
        firstUnsupportedPosition.z, firstUnsupportedNeighbor.x, firstUnsupportedNeighbor.z,
        firstUnsupportedWet.terrainHeight, firstUnsupportedWet.waterSurface,
        firstUnsupportedWet.hydrology.ocean, firstUnsupportedWet.hydrology.river,
        firstUnsupportedWet.hydrology.lake, firstUnsupportedWet.hydrology.flowDirection.x,
        firstUnsupportedWet.hydrology.flowDirection.z, firstUnsupportedDry.terrainHeight,
        firstUnsupportedDry.waterSurface, firstUnsupportedDry.hydrology.channelBank,
        firstUnsupportedDry.hydrology.lakeBank, firstUnsupportedDry.hydrology.channelDistance,
        firstUnsupportedDry.hydrology.channelWidth, firstUnsupportedDry.hydrology.lakeShoreDistance,
        firstUnsupportedDry.hydrology.shoreWaterSurface,
        firstUnsupportedCanonicalWet.surfaceElevation, firstUnsupportedCanonicalWet.waterSurface,
        firstUnsupportedCanonicalWet.ocean, firstUnsupportedCanonicalWet.river,
        firstUnsupportedCanonicalWet.lake, firstUnsupportedCanonicalDry.surfaceElevation,
        firstUnsupportedCanonicalDry.waterSurface, firstUnsupportedCanonicalDry.ocean,
        firstUnsupportedCanonicalDry.river, firstUnsupportedCanonicalDry.lake);
    REQUIRE(wet >= 1'000);
    REQUIRE(adjacentWet > wet);
    REQUIRE(supportableCanonical > 0);
    REQUIRE(deletedCanonical == 0);
    REQUIRE(unsupportedBanks == 0);
    REQUIRE(untagged == 0);
    REQUIRE(maximumUntaggedStep <= 0.125001);
    REQUIRE(explicitFallFaces > 0);
    REQUIRE(longCardinalFloorRuns == 0);
}

TEST_CASE("V4 volcanic relief enters canonical routing before climate adaptation",
          "[learned][v4][climate][volcano]") {
    TempDir directory("learned_zero_lapse");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    std::vector<VolcanoPrimitive> volcanoes;
    for (int64_t cellZ = -2; cellZ <= 2 && volcanoes.empty(); ++cellZ) {
        for (int64_t cellX = -4; cellX <= 0 && volcanoes.empty(); ++cellX) {
            volcanoes = awaitGeneration(
                [&] { return generator.hotspotVolcanoesForCell(cellX, cellZ); }, 20s);
        }
    }
    REQUIRE_FALSE(volcanoes.empty());
    const VolcanoPrimitive& volcano = volcanoes.front();
    const int64_t worldX = static_cast<int64_t>(std::llround(volcano.centerX));
    const int64_t worldZ = static_cast<int64_t>(std::llround(volcano.centerZ));
    const worldgen::SurfaceSample macroSurface = awaitGeneration(
        [&] {
            return macro.sampleSurface(static_cast<double>(worldX), static_cast<double>(worldZ));
        },
        20s);
    const worldgen::SurfaceSample emitted =
        awaitGeneration([&] { return generator.sampleFarGeometrySurface(worldX, worldZ); }, 20s);
    const double heightAdjustment = emitted.terrainHeight - macroSurface.terrainHeight;
    const double unmodifiedLearnedHeight = learnedElevationMetersToWorldHeight(750.0);
    CAPTURE(worldX, worldZ, heightAdjustment, macroSurface.climate.temperatureC,
            emitted.climate.temperatureC, macroSurface.terrainHeight, unmodifiedLearnedHeight);
    REQUIRE(std::abs(macroSurface.terrainHeight - unmodifiedLearnedHeight) > 1.0);
    REQUIRE(std::abs(heightAdjustment) <= 1.0e-9);
    REQUIRE(macroSurface.climate.lapseRateCPerMeter == 0.0);
    REQUIRE(emitted.climate.lapseRateCPerMeter == 0.0);
    REQUIRE(emitted.climate.temperatureC ==
            Catch::Approx(macroSurface.climate.temperatureC).margin(1.0e-9));
}

TEST_CASE("V4 far volcano culling does not request an irrelevant crater datum",
          "[learned][v4][volcano][horizon][regression]") {
    TempDir directory("learned_far_volcano_culling");
    constexpr uint64_t SEED = 764'891;
    const GenerationIdentity identity = testIdentity(SEED);
    auto backend = std::make_shared<RecordingQualityTerrainBackend>();
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);

    // This is the representative outer-horizon parent at tile (-23, 22).
    // It inspects the active hotspot cell but lies more than two kilometers
    // from one of its cones. Before deferred crater data, constructing that
    // unrelated cone requested the learned page for its center at native
    // (row 1792, column -2050), outside the far tile's authority closure.
    const worldgen::SurfaceSample surface =
        awaitGeneration([&] { return generator.sampleFarGeometrySurface(-5'888, 5'632); }, 20s);
    REQUIRE(std::isfinite(surface.terrainHeight));

    const std::vector<TerrainPageKey> keys = backend->keys();
    REQUIRE_FALSE(keys.empty());
    const auto irrelevantDatum = std::ranges::find_if(keys, [](TerrainPageKey key) {
        return key.coordinate == TerrainPageCoordinate{.row = 7, .column = -9};
    });
    REQUIRE(irrelevantDatum == keys.end());
}

TEST_CASE("V4 far water uses native topology instead of analytical crater candidates",
          "[learned][v4][volcano][water][far][regression]") {
    TempDir directory("learned_far_crater_water_candidates");
    constexpr uint64_t SEED = 764'891;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<PlanarTerrainBackend>());
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    // This seed owns the crater lake exercised by the ordinary cubic-world
    // volcano regression. Keep the native-topology retirement check on that
    // same stable feature without requiring a real ONNX model in CI.
    constexpr int64_t centerX = 23'029;
    constexpr int64_t centerZ = -111'486;
    const int64_t centerCellOriginX = world_coord::floorDiv(centerX, int64_t{32}) * 32;
    const int64_t centerCellOriginZ = world_coord::floorDiv(centerZ, int64_t{32}) * 32;
    const int64_t topologyOriginX = centerCellOriginX - 64;
    const int64_t topologyOriginZ = centerCellOriginZ - 64;
    std::array<uint8_t, 25> candidates{};
    std::array<worldgen::NativeHydrologyTopologyCell, 25> topology{};
    const std::vector<VolcanoPrimitive> volcanoes =
        awaitGeneration([&] { return generator.hotspotVolcanoesForCell(1, -7); }, 20s);
    const auto primitive = std::ranges::find_if(volcanoes, [](const VolcanoPrimitive& volcano) {
        return std::abs(volcano.centerX - 23'029.177516) < 0.01 &&
               std::abs(volcano.centerZ + 111'485.810195) < 0.01;
    });
    awaitGeneration(
        [&] {
            generator.markVolcanicWaterCandidates(topologyOriginX, topologyOriginZ, 32, 5, 5,
                                                  candidates);
            generator.sampleNativeHydrologyTopologyGrid(topologyOriginX, topologyOriginZ, 5, 5,
                                                        topology);
            return true;
        },
        20s);
    const worldgen::SurfaceSample surface =
        awaitGeneration([&] { return generator.sampleFarGeometrySurface(centerX, centerZ); }, 20s);
    const worldgen::HydrologySample canonical =
        awaitGeneration([&] { return macro.sampleHydrology(centerX, centerZ); }, 20s);
    CAPTURE(volcanoes.size(), primitive != volcanoes.end(), canonical.ocean, canonical.river,
            canonical.lake, canonical.waterSurface, canonical.waterBodyId, surface.terrainHeight,
            surface.waterSurface, surface.hydrology.lake, surface.hydrology.ocean,
            surface.hydrology.waterBodyId);
    REQUIRE(primitive != volcanoes.end());
    CAPTURE(primitive->craterLake, primitive->craterLakeRadius, primitive->craterLakeSurface,
            primitive->craterRadius, primitive->craterDatumElevation);
    REQUIRE(std::ranges::none_of(candidates, [](uint8_t value) { return value != 0; }));
    REQUIRE(
        std::ranges::any_of(topology, [](const auto& cell) { return cell.waterTopologyPossible; }));
    // The crater participates in the native solve, so this fixture now owns a
    // naturally spilled canonical lake. The old analytical crater stage is
    // diagnostic only and cannot override the routed body or its stage.
    REQUIRE_FALSE(canonical.ocean);
    REQUIRE_FALSE(canonical.river);
    REQUIRE(canonical.lake);
    REQUIRE(canonical.waterBodyId != worldgen::NO_WATER_BODY);
    REQUIRE(canonical.waterSurface > canonical.surfaceElevation);
    REQUIRE_FALSE(surface.hydrology.ocean);
    REQUIRE_FALSE(surface.hydrology.river);
    REQUIRE(surface.hydrology.lake);
    REQUIRE(surface.hydrology.waterBodyId == canonical.waterBodyId);
    REQUIRE(surface.waterSurface == canonical.waterSurface);
    REQUIRE(surface.waterSurface != Catch::Approx(primitive->craterLakeSurface).margin(1.0e-6));
}

TEST_CASE("V4 learned peaks retain the expanded height and lazy top density",
          "[learned][v4][height][density]") {
    TempDir directory("learned_high_peak_density");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>(10'050));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    REQUIRE(generator.densityLatticeVerticalSpacing() == LEARNED_DENSITY_LATTICE_Y);
    REQUIRE(generator.densityLatticeLevelCount() == 193);

    const worldgen::SurfaceSample highSurface =
        awaitGeneration([&] { return generator.sampleFarGeometrySurface(0, 0); }, 20s);
    CAPTURE(highSurface.terrainHeight);
    REQUIRE(highSurface.terrainHeight > 480.0);
    REQUIRE(highSurface.terrainHeight <= WORLD_MAX_Y);

    const uint64_t beforePlan = generator.densityEvaluationCount();
    const std::shared_ptr<const ColumnPlan> plan =
        awaitGeneration([&] { return generator.getColumnPlan({0, 0}); }, 30s);
    REQUIRE(plan);
    const uint64_t planEvaluations = generator.densityEvaluationCount() - beforePlan;
    Chunk top(ChunkPos{0, WORLD_MAX_CHUNK_Y, 0});
    const uint64_t before = generator.densityEvaluationCount();
    awaitGeneration(
        [&] {
            generator.generate(top);
            return true;
        },
        30s);
    const uint64_t evaluated = generator.densityEvaluationCount() - before;
    const uint64_t cubeEvaluations = generator.lastCubeDensityEvaluationCount();
    CAPTURE(planEvaluations, cubeEvaluations, evaluated);
    REQUIRE(top.generated);
    REQUIRE(planEvaluations < 128);
    REQUIRE(cubeEvaluations > 0);
    REQUIRE(cubeEvaluations < 256);
    REQUIRE(evaluated > 0);
    REQUIRE(evaluated < 2'048);
    bool emittedTopTerrain = false;
    for (int z = 0; z < CHUNK_EDGE; ++z) {
        for (int x = 0; x < CHUNK_EDGE; ++x) {
            for (int y = 0; y < CHUNK_EDGE; ++y) {
                emittedTopTerrain |= top.getBlock(x, y, z) != BlockType::AIR;
            }
        }
    }
    REQUIRE(emittedTopTerrain);

    Chunk bottom(ChunkPos{0, WORLD_MIN_CHUNK_Y, 0});
    const uint64_t beforeBottom = generator.densityEvaluationCount();
    awaitGeneration(
        [&] {
            generator.generate(bottom);
            return true;
        },
        30s);
    const uint64_t bottomEvaluated = generator.densityEvaluationCount() - beforeBottom;
    const uint64_t bottomCubeEvaluations = generator.lastCubeDensityEvaluationCount();
    CAPTURE(bottomCubeEvaluations, bottomEvaluated);
    REQUIRE(bottom.generated);
    REQUIRE(bottomCubeEvaluations > 0);
    REQUIRE(bottomCubeEvaluations < 256);
    REQUIRE(bottomEvaluated < 2'048);
}

TEST_CASE("Legacy density retains its four block vertical lattice",
          "[density][v3][compatibility]") {
    ChunkGenerator generator(42U);
    REQUIRE(generator.densityLatticeVerticalSpacing() == LEGACY_DENSITY_LATTICE_Y);
    REQUIRE(generator.densityLatticeLevelCount() ==
            (WORLD_MAX_Y - WORLD_MIN_Y + 1) / LEGACY_DENSITY_LATTICE_Y + 1);
    REQUIRE(generator.densityLatticeLevelCount() == 385);
}

TEST_CASE("V4 volcanism cannot delete canonical wet topology",
          "[learned][v4][volcano][hydrology]") {
    TempDir directory("learned_volcanic_water_preservation");
    constexpr uint64_t SEED = 42;
    const GenerationIdentity identity = testIdentity(SEED);
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, directory.path(), std::make_shared<ConstantTerrainBackend>(-25));
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    ChunkGenerator generator(SEED, context);
    worldgen::MacroGenerationSampler macro(SEED, context);

    std::vector<VolcanoPrimitive> volcanoes;
    for (int64_t cellZ = -2; cellZ <= 2 && volcanoes.empty(); ++cellZ) {
        for (int64_t cellX = -4; cellX <= 0 && volcanoes.empty(); ++cellX) {
            volcanoes = awaitGeneration(
                [&] { return generator.hotspotVolcanoesForCell(cellX, cellZ); }, 20s);
        }
    }
    REQUIRE_FALSE(volcanoes.empty());
    const VolcanoPrimitive& volcano = volcanoes.front();
    const int64_t centerX = static_cast<int64_t>(std::llround(volcano.centerX));
    const int64_t centerZ = static_cast<int64_t>(std::llround(volcano.centerZ));
    std::vector<ColumnPos> positions;
    for (int offsetZ = -4; offsetZ <= 4; ++offsetZ) {
        for (int offsetX = -4; offsetX <= 4; ++offsetX) {
            positions.push_back({centerX + offsetX, centerZ + offsetZ});
        }
    }
    std::vector<worldgen::HydrologySample> canonical(positions.size());
    awaitGeneration(
        [&] {
            macro.sampleHydrologyPoints(positions, canonical);
            return true;
        },
        20s);
    std::vector<worldgen::SurfaceSample> emitted(positions.size());
    awaitGeneration(
        [&] {
            generator.sampleFarGeometryPoints(positions, worldgen::SurfaceFootprint::BLOCK_1,
                                              emitted);
            return true;
        },
        20s);

    size_t canonicalWet = 0;
    size_t retainedWet = 0;
    for (size_t index = 0; index < positions.size(); ++index) {
        CAPTURE(index, positions[index].x, positions[index].z);
        const worldgen::HydrologySample& before = canonical[index];
        const worldgen::HydrologySample& after = emitted[index].hydrology;
        REQUIRE(before.ocean);
        ++canonicalWet;
        retainedWet += after.ocean ? 1U : 0U;
        CHECK(after.ocean == before.ocean);
        CHECK(after.river == before.river);
        CHECK(after.lake == before.lake);
        CHECK(after.wetland == before.wetland);
        CHECK(after.delta == before.delta);
        CHECK(after.estuary == before.estuary);
        CHECK(after.brackish == before.brackish);
        CHECK(after.waterfall == before.waterfall);
        CHECK(after.channelBank == before.channelBank);
        CHECK(after.lakeBank == before.lakeBank);
        CHECK(after.waterBodyId == before.waterBodyId);
        CHECK(after.generatedFluidLevel == before.generatedFluidLevel);
        CHECK(after.surfaceElevation == before.surfaceElevation);
        CHECK(emitted[index].terrainHeight == before.surfaceElevation);
        CHECK(after.waterSurface == before.waterSurface);
        CHECK(after.channelDepth == before.channelDepth);
        CHECK(after.erosionDepth == before.erosionDepth);
        CHECK(after.lakeDepth == before.lakeDepth);
        CHECK(after.lakeShoreDistance == before.lakeShoreDistance);
        CHECK(after.shoreWaterSurface == before.shoreWaterSurface);
        CHECK(after.transitionOwnerKind == before.transitionOwnerKind);
        CHECK(after.transitionOwnerId == before.transitionOwnerId);
    }
    REQUIRE(canonicalWet == positions.size());
    REQUIRE(retainedWet == canonicalWet);
}

TEST_CASE("V4 learned failures latch and never fall back to synthetic terrain", "[learned][v4]") {
    TempDir directory("learned_failure");
    const GenerationIdentity identity = testIdentity();
    auto backend = std::make_shared<FailingTerrainBackend>();
    auto authority = std::make_shared<CachedTerrainAuthority>(identity, directory.path(), backend);
    auto context =
        std::make_shared<WorldGenerationContext>(identity, authority, AuthorityQuality::FINAL);
    worldgen::MacroGenerationSampler macro(identity.seed, context);

    try {
        static_cast<void>(macro.preliminaryElevation(12.0, 34.0));
        FAIL("Cold learned terrain unexpectedly completed on the caller thread");
    } catch (const GenerationFailureException& error) {
        REQUIRE(error.status() == AuthorityStatus::DEFERRED);
    }
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!context->failure() && std::chrono::steady_clock::now() < deadline) {
        try {
            static_cast<void>(macro.preliminaryElevation(12.0, 34.0));
        } catch (const GenerationFailureException&) {
        }
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(context->failure());
    REQUIRE(context->failure()->code == GenerationFailureCode::INFERENCE_FAILED);
    REQUIRE(backend->calls == 1);
    REQUIRE_THROWS_AS(macro.preliminaryElevation(12.0, 34.0), GenerationFailureException);
    REQUIRE(backend->calls == 1);
    REQUIRE(context->metrics().failedQueries == 2);
}

TEST_CASE("V4 corrupt native hydrology rebuilds through learned authority",
          "[learned][v4][hydrology][persistence][repair]") {
    TempDir directory("learned_hydrology_failure");
    const GenerationIdentity identity = testIdentity(0xC077'B10B'0002ULL);
    const std::filesystem::path hydrologyRoot =
        std::filesystem::path(directory.path()) / "hydrology-authority-v1";
    worldgen::hydrology::HydrologyAuthorityStore store(hydrologyRoot / "final", identity,
                                                       AuthorityQuality::FINAL);
    const std::array<uint8_t, 8> invalidPayload{'N', 'H', '4', 'P', 0, 0, 0, 0};
    REQUIRE(store.write({0, 0}, invalidPayload).isReady());
    auto backend = std::make_shared<DeterministicFakeTerrainBackend>();
    auto authority = std::make_shared<CachedTerrainAuthority>(
        identity, std::filesystem::path(directory.path()) / "terrain", backend);
    auto context = std::make_shared<WorldGenerationContext>(identity, authority,
                                                            AuthorityQuality::FINAL, hydrologyRoot);
    worldgen::MacroGenerationSampler macro(identity.seed, context);

    const worldgen::HydrologySample repaired = awaitGeneration(
        [&] { return macro.sampleHydrology(1'024.0, 1'024.0); }, std::chrono::seconds(20));
    REQUIRE_FALSE(context->failure());
    REQUIRE(backend->callCount() > 0);
    REQUIRE(repaired.surfaceElevation != 0.0);
    const worldgen::NativeHydrologyCacheMetrics metrics = macro.nativeHydrologyCacheMetrics();
    REQUIRE(metrics.persistedLoads == 0);
    // The repaired owner may solve and persist its bounded canonical closure
    // with neighboring pages. Only the corrupt owner counts as a repair.
    REQUIRE(metrics.persistedWrites >= 1);
    REQUIRE(metrics.persistedRepairs == 1);
    const auto repairedPayload = store.load({0, 0});
    REQUIRE(repairedPayload.isReady());
    REQUIRE(repairedPayload.value()->size() > 52);
    REQUIRE(std::ranges::equal(std::span(*repairedPayload.value()).first(4),
                               std::array<uint8_t, 4>{'N', 'H', '4', 'P'}));
}
