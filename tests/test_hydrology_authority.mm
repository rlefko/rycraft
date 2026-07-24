#include <catch2/catch_all.hpp>

#include "test_helpers.hpp"
#include "world/hydrology_authority.hpp"

#include <algorithm>
#include <barrier>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace worldgen::hydrology;
using namespace worldgen::learned;

GenerationIdentity hydrologyTestIdentity(uint64_t seed = 0xABCD'0123'4567'89EFULL) {
    GenerationIdentity identity;
    identity.seed = seed;
    identity.modelPackHash =
        *parseSha256("543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0");
    identity.runtimeHash =
        *parseSha256("e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6");
    return identity;
}

std::vector<uint8_t> hydrologyPayload(size_t size = 32'768) {
    std::vector<uint8_t> payload(size);
    for (size_t index = 0; index < payload.size(); ++index)
        payload[index] = static_cast<uint8_t>((index * 73U + index / 11U) & 0xFFU);
    return payload;
}

std::vector<uint8_t> readHydrologyBytes(const std::filesystem::path& path) {
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

void writeHydrologyBytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    REQUIRE(output.is_open());
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    REQUIRE(output.good());
}

TEST_CASE("RYHY pages round-trip signed coordinates with atomic validation",
          "[hydrology-authority][persistence]") {
    TempDir directory("hydrology_authority_round_trip");
    const GenerationIdentity identity = hydrologyTestIdentity();
    HydrologyAuthorityStore store(std::filesystem::path(directory.path()) /
                                      "hydrology-authority-v1",
                                  identity, AuthorityQuality::FINAL);
    constexpr HydrologyPageCoordinate coordinate{-17, 23};
    const std::vector<uint8_t> payload = hydrologyPayload();

    REQUIRE(store.pagePath(coordinate).filename() == "p.-17.23.ryhy");
    auto written = store.write(coordinate, payload);
    REQUIRE(written.isReady());

    const std::filesystem::path path = store.pagePath(coordinate);
    REQUIRE(std::filesystem::exists(path));
    const std::vector<uint8_t> fileBytes = readHydrologyBytes(path);
    REQUIRE(fileBytes.size() > 88);
    REQUIRE(std::ranges::equal(std::span(fileBytes).first(4),
                               std::array<uint8_t, 4>{'R', 'Y', 'H', 'Y'}));
    REQUIRE(fileBytes[8] == 1);
    REQUIRE(fileBytes[9] == static_cast<uint8_t>(AuthorityQuality::FINAL));
    REQUIRE(std::ranges::none_of(
        std::filesystem::directory_iterator(path.parent_path()),
        [](const auto& entry) { return entry.path().filename().string().contains(".tmp."); }));

    auto loaded = store.load(coordinate);
    REQUIRE(loaded.isReady());
    REQUIRE(*loaded.value() == payload);
    REQUIRE(store.write(coordinate, payload).isReady());
}

TEST_CASE("RYHY concurrent publishers preserve the first immutable payload",
          "[hydrology-authority][persistence][concurrency]") {
    TempDir directory("hydrology_concurrent_immutable_publication");
    const GenerationIdentity identity = hydrologyTestIdentity(0xC0FF'EE00'0002ULL);
    constexpr HydrologyPageCoordinate coordinate{-21, 31};
    const std::vector<uint8_t> firstPayload = hydrologyPayload(4'096);
    std::vector<uint8_t> conflictingPayload = firstPayload;
    conflictingPayload.front() ^= 0xFFU;
    REQUIRE(conflictingPayload != firstPayload);

    HydrologyAuthorityStore seedStore(directory.path(), identity, AuthorityQuality::FINAL);
    REQUIRE(seedStore.write(coordinate, firstPayload).isReady());
    std::vector<uint8_t> corrupted = readHydrologyBytes(seedStore.pagePath(coordinate));
    corrupted.back() ^= 0xA5U;
    writeHydrologyBytes(seedStore.pagePath(coordinate), corrupted);
    const auto corrupt = seedStore.load(coordinate);
    REQUIRE(corrupt.status() == AuthorityStatus::FAILED);
    REQUIRE(corrupt.failure());
    REQUIRE(corrupt.failure()->code == GenerationFailureCode::CORRUPT_PAGE);

    auto publishBarrier = std::make_shared<std::barrier<>>(2);
    auto hooks = std::make_shared<HydrologyAuthorityStore::TestHooks>();
    hooks->beforeExclusivePublish = [publishBarrier] { publishBarrier->arrive_and_wait(); };
    HydrologyAuthorityStore firstStore(directory.path(), identity, AuthorityQuality::FINAL, hooks);
    HydrologyAuthorityStore conflictingStore(directory.path(), identity, AuthorityQuality::FINAL,
                                             hooks);

    AuthorityResult<bool> firstResult;
    AuthorityResult<bool> conflictingResult;
    std::thread firstWriter([&] { firstResult = firstStore.write(coordinate, firstPayload); });
    std::thread conflictingWriter(
        [&] { conflictingResult = conflictingStore.write(coordinate, conflictingPayload); });
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

    const std::vector<uint8_t>& expected = firstPublished ? firstPayload : conflictingPayload;
    HydrologyAuthorityStore observer(directory.path(), identity, AuthorityQuality::FINAL);
    const auto persisted = observer.load(coordinate);
    REQUIRE(persisted.isReady());
    REQUIRE(*persisted.value() == expected);
    REQUIRE(observer.write(coordinate, expected).isReady());
    const std::vector<uint8_t>& rejectedPayload =
        firstPublished ? conflictingPayload : firstPayload;
    const auto rejectedRetry = observer.write(coordinate, rejectedPayload);
    REQUIRE(rejectedRetry.status() == AuthorityStatus::FAILED);
    REQUIRE(rejectedRetry.failure());
    REQUIRE(rejectedRetry.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE(std::ranges::none_of(
        std::filesystem::directory_iterator(observer.pagePath(coordinate).parent_path()),
        [](const auto& entry) { return entry.path().filename().string().contains(".tmp."); }));
}

TEST_CASE("RYHY pages reject identity mismatches and unsupported hydrology revisions",
          "[hydrology-authority][persistence][fingerprint]") {
    TempDir directory("hydrology_authority_identity");
    const GenerationIdentity identity = hydrologyTestIdentity();
    HydrologyAuthorityStore store(directory.path(), identity, AuthorityQuality::FINAL);
    constexpr HydrologyPageCoordinate coordinate{4, -9};
    const std::vector<uint8_t> payload = hydrologyPayload(8'192);
    REQUIRE(store.write(coordinate, payload).isReady());

    GenerationIdentity anotherSeed = identity;
    ++anotherSeed.seed;
    HydrologyAuthorityStore seedStore(directory.path(), anotherSeed, AuthorityQuality::FINAL);
    auto seedRejected = seedStore.load(coordinate);
    REQUIRE(seedRejected.status() == AuthorityStatus::FAILED);
    REQUIRE(seedRejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE_FALSE(seedRejected.failure()->retriable);
    REQUIRE(seedStore.write(coordinate, payload).failure()->code ==
            GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);

    GenerationIdentity anotherRevision = identity;
    ++anotherRevision.hydrologyRevision;
    HydrologyAuthorityStore revisionStore(directory.path(), anotherRevision,
                                          AuthorityQuality::FINAL);
    auto revisionRejected = revisionStore.load(coordinate);
    REQUIRE(revisionRejected.status() == AuthorityStatus::FAILED);
    REQUIRE(revisionRejected.failure()->code == GenerationFailureCode::INVALID_REQUEST);
    REQUIRE_FALSE(revisionRejected.failure()->retriable);
    REQUIRE(store.load(coordinate).isReady());
}

TEST_CASE("RYHY pages reject preview payloads copied into final authority",
          "[hydrology-authority][persistence][quality]") {
    TempDir directory("hydrology_authority_quality");
    const GenerationIdentity identity = hydrologyTestIdentity();
    HydrologyAuthorityStore previewStore(std::filesystem::path(directory.path()) / "preview",
                                         identity, AuthorityQuality::PREVIEW);
    HydrologyAuthorityStore finalStore(std::filesystem::path(directory.path()) / "final", identity,
                                       AuthorityQuality::FINAL);
    constexpr HydrologyPageCoordinate coordinate{-12, 7};
    const std::vector<uint8_t> payload = hydrologyPayload(8'192);
    REQUIRE(previewStore.write(coordinate, payload).isReady());
    REQUIRE(std::filesystem::create_directories(finalStore.pagePath(coordinate).parent_path()));
    REQUIRE(std::filesystem::copy_file(previewStore.pagePath(coordinate),
                                       finalStore.pagePath(coordinate)));

    const auto rejected = finalStore.load(coordinate);
    REQUIRE(rejected.status() == AuthorityStatus::FAILED);
    REQUIRE(rejected.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE_FALSE(rejected.failure()->retriable);
    REQUIRE(previewStore.load(coordinate).isReady());
}

TEST_CASE("RYHY pages detect corruption and preserve immutable payloads",
          "[hydrology-authority][persistence][corruption]") {
    TempDir directory("hydrology_authority_corruption");
    const GenerationIdentity identity = hydrologyTestIdentity();
    HydrologyAuthorityStore store(directory.path(), identity, AuthorityQuality::FINAL);
    constexpr HydrologyPageCoordinate coordinate{-3, -8};
    const std::vector<uint8_t> payload = hydrologyPayload(16'384);
    REQUIRE(store.write(coordinate, payload).isReady());

    std::vector<uint8_t> bytes = readHydrologyBytes(store.pagePath(coordinate));
    bytes.back() ^= 0xA5U;
    writeHydrologyBytes(store.pagePath(coordinate), bytes);
    auto corrupt = store.load(coordinate);
    REQUIRE(corrupt.status() == AuthorityStatus::FAILED);
    REQUIRE(corrupt.failure()->code == GenerationFailureCode::CORRUPT_PAGE);
    REQUIRE(corrupt.failure()->retriable);

    REQUIRE(store.write(coordinate, payload).isReady());
    REQUIRE(*store.load(coordinate).value() == payload);

    std::vector<uint8_t> conflicting = payload;
    conflicting.front() ^= 0xFFU;
    auto immutable = store.write(coordinate, conflicting);
    REQUIRE(immutable.status() == AuthorityStatus::FAILED);
    REQUIRE(immutable.failure()->code == GenerationFailureCode::INCOMPATIBLE_FINGERPRINT);
    REQUIRE_FALSE(immutable.failure()->retriable);
    REQUIRE(*store.load(coordinate).value() == payload);

    bytes = readHydrologyBytes(store.pagePath(coordinate));
    bytes.push_back(0);
    writeHydrologyBytes(store.pagePath(coordinate), bytes);
    REQUIRE(store.load(coordinate).failure()->code == GenerationFailureCode::CORRUPT_PAGE);
}

TEST_CASE("RYHY loads ignore interrupted staging files and enforce blob bounds",
          "[hydrology-authority][persistence][bounds]") {
    TempDir directory("hydrology_authority_interrupted");
    const GenerationIdentity identity = hydrologyTestIdentity();
    HydrologyAuthorityStore store(directory.path(), identity, AuthorityQuality::FINAL);
    constexpr HydrologyPageCoordinate coordinate{-1, 0};
    const std::filesystem::path canonical = store.pagePath(coordinate);
    REQUIRE(std::filesystem::create_directories(canonical.parent_path()));
    std::filesystem::path interrupted = canonical;
    interrupted += ".tmp.interrupted";
    writeHydrologyBytes(interrupted, {'R', 'Y', 'H', 'Y'});

    auto missing = store.load(coordinate);
    REQUIRE(missing.status() == AuthorityStatus::DEFERRED);
    REQUIRE(missing.failure()->code == GenerationFailureCode::PAGE_NOT_FOUND);
    REQUIRE_FALSE(std::filesystem::exists(canonical));

    const std::vector<uint8_t> payload = hydrologyPayload(4'096);
    REQUIRE(store.write(coordinate, payload).isReady());
    REQUIRE(*store.load(coordinate).value() == payload);
    REQUIRE(std::filesystem::exists(interrupted));

    auto empty = store.write({9, 9}, {});
    REQUIRE(empty.status() == AuthorityStatus::FAILED);
    REQUIRE(empty.failure()->code == GenerationFailureCode::INVALID_REQUEST);
    REQUIRE_FALSE(empty.failure()->retriable);

    HydrologyAuthorityStore emptyRoot({}, identity, AuthorityQuality::FINAL);
    auto invalidRoot = emptyRoot.load({0, 0});
    REQUIRE(invalidRoot.status() == AuthorityStatus::FAILED);
    REQUIRE(invalidRoot.failure()->code == GenerationFailureCode::INVALID_REQUEST);

    constexpr HydrologyPageCoordinate truncatedCoordinate{7, -5};
    writeHydrologyBytes(store.pagePath(truncatedCoordinate), {'R', 'Y', 'H', 'Y'});
    auto truncated = store.load(truncatedCoordinate);
    REQUIRE(truncated.status() == AuthorityStatus::FAILED);
    REQUIRE(truncated.failure()->code == GenerationFailureCode::CORRUPT_PAGE);

    constexpr HydrologyPageCoordinate copiedCoordinate{8, -5};
    std::filesystem::copy_file(canonical, store.pagePath(copiedCoordinate));
    auto wrongPath = store.load(copiedCoordinate);
    REQUIRE(wrongPath.status() == AuthorityStatus::FAILED);
    REQUIRE(wrongPath.failure()->code == GenerationFailureCode::CORRUPT_PAGE);
}

} // namespace
