#include "world/terrain_bootstrap.hpp"

#include "world/learned_terrain.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include <sys/stat.h>
#include <unistd.h>

namespace worldgen::bootstrap {

namespace {

constexpr std::string_view COMPLETION_MARKER = ".verified-pack-v1";
constexpr uint32_t COMPLETION_MARKER_SCHEMA_VERSION = 2;
constexpr size_t SHA256_BLOCK_BYTES = 64;
constexpr size_t HASH_READ_BYTES = 1024 * 1024;

constexpr std::array<uint32_t, 64> SHA256_ROUND_CONSTANTS{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

class StreamingSha256 {
public:
    void update(std::span<const uint8_t> bytes) {
        if (bytes.empty()) return;

        totalBytes_ += bytes.size();
        size_t offset = 0;
        if (bufferSize_ != 0) {
            const size_t copied = std::min(SHA256_BLOCK_BYTES - bufferSize_, bytes.size());
            std::copy_n(bytes.begin(), copied,
                        buffer_.begin() + static_cast<ptrdiff_t>(bufferSize_));
            bufferSize_ += copied;
            offset += copied;
            if (bufferSize_ == SHA256_BLOCK_BYTES) {
                transform(buffer_);
                bufferSize_ = 0;
            }
        }

        while (bytes.size() - offset >= SHA256_BLOCK_BYTES) {
            std::array<uint8_t, SHA256_BLOCK_BYTES> block{};
            std::copy_n(bytes.begin() + static_cast<ptrdiff_t>(offset), SHA256_BLOCK_BYTES,
                        block.begin());
            transform(block);
            offset += SHA256_BLOCK_BYTES;
        }

        if (offset < bytes.size()) {
            bufferSize_ = bytes.size() - offset;
            std::copy(bytes.begin() + static_cast<ptrdiff_t>(offset), bytes.end(), buffer_.begin());
        }
    }

    [[nodiscard]] std::array<uint8_t, 32> finish() {
        const uint64_t bitLength = totalBytes_ * 8U;
        buffer_[bufferSize_++] = 0x80U;
        if (bufferSize_ > 56) {
            std::fill(buffer_.begin() + static_cast<ptrdiff_t>(bufferSize_), buffer_.end(), 0U);
            transform(buffer_);
            bufferSize_ = 0;
        }
        std::fill(buffer_.begin() + static_cast<ptrdiff_t>(bufferSize_), buffer_.begin() + 56, 0U);
        for (size_t index = 0; index < sizeof(bitLength); ++index) {
            const size_t shift = (sizeof(bitLength) - 1U - index) * 8U;
            buffer_[56 + index] = static_cast<uint8_t>(bitLength >> shift);
        }
        transform(buffer_);

        std::array<uint8_t, 32> digest{};
        for (size_t word = 0; word < state_.size(); ++word) {
            digest[word * 4] = static_cast<uint8_t>(state_[word] >> 24U);
            digest[word * 4 + 1] = static_cast<uint8_t>(state_[word] >> 16U);
            digest[word * 4 + 2] = static_cast<uint8_t>(state_[word] >> 8U);
            digest[word * 4 + 3] = static_cast<uint8_t>(state_[word]);
        }
        return digest;
    }

private:
    std::array<uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                   0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<uint8_t, SHA256_BLOCK_BYTES> buffer_{};
    size_t bufferSize_ = 0;
    uint64_t totalBytes_ = 0;

    void transform(const std::array<uint8_t, SHA256_BLOCK_BYTES>& block) {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t offset = index * 4;
            words[index] = static_cast<uint32_t>(block[offset]) << 24U |
                           static_cast<uint32_t>(block[offset + 1]) << 16U |
                           static_cast<uint32_t>(block[offset + 2]) << 8U |
                           static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t previous15 = words[index - 15];
            const uint32_t previous2 = words[index - 2];
            const uint32_t sigma0 =
                std::rotr(previous15, 7) ^ std::rotr(previous15, 18) ^ (previous15 >> 3U);
            const uint32_t sigma1 =
                std::rotr(previous2, 17) ^ std::rotr(previous2, 19) ^ (previous2 >> 10U);
            words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t bigSigma1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const uint32_t choose = (e & f) ^ (~e & g);
            const uint32_t temporary1 =
                h + bigSigma1 + choose + SHA256_ROUND_CONSTANTS[index] + words[index];
            const uint32_t bigSigma0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = bigSigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

std::string digestHex(const std::array<uint8_t, 32>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const uint8_t byte : digest)
        output << std::setw(2) << static_cast<unsigned int>(byte);
    return output.str();
}

struct CompletionMarkerFileStamp {
    uintmax_t device = 0;
    uintmax_t inode = 0;
    uint64_t byteSize = 0;
    int64_t modifiedSeconds = 0;
    int64_t modifiedNanoseconds = 0;
    int64_t changedSeconds = 0;
    int64_t changedNanoseconds = 0;
};

std::optional<CompletionMarkerFileStamp>
completionMarkerFileStamp(const std::filesystem::path& path, uint64_t expectedBytes,
                          std::string& errorMessage) {
    struct stat status{};
    if (::lstat(path.c_str(), &status) != 0) {
        errorMessage =
            "Could not inspect terrain model completion marker input: " + path.filename().string();
        return std::nullopt;
    }
    if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) != expectedBytes) {
        errorMessage = "Terrain model completion marker input changed: " + path.filename().string();
        return std::nullopt;
    }

    CompletionMarkerFileStamp stamp{
        .device = static_cast<uintmax_t>(status.st_dev),
        .inode = static_cast<uintmax_t>(status.st_ino),
        .byteSize = static_cast<uint64_t>(status.st_size),
    };
#if defined(__APPLE__)
    stamp.modifiedSeconds = static_cast<int64_t>(status.st_mtimespec.tv_sec);
    stamp.modifiedNanoseconds = static_cast<int64_t>(status.st_mtimespec.tv_nsec);
    stamp.changedSeconds = static_cast<int64_t>(status.st_ctimespec.tv_sec);
    stamp.changedNanoseconds = static_cast<int64_t>(status.st_ctimespec.tv_nsec);
#elif defined(__linux__)
    stamp.modifiedSeconds = static_cast<int64_t>(status.st_mtim.tv_sec);
    stamp.modifiedNanoseconds = static_cast<int64_t>(status.st_mtim.tv_nsec);
    stamp.changedSeconds = static_cast<int64_t>(status.st_ctim.tv_sec);
    stamp.changedNanoseconds = static_cast<int64_t>(status.st_ctim.tv_nsec);
#else
    const auto modified = std::filesystem::last_write_time(path);
    stamp.modifiedSeconds = static_cast<int64_t>(modified.time_since_epoch().count());
    stamp.changedSeconds = stamp.modifiedSeconds;
#endif
    return stamp;
}

std::optional<std::string> completionMarkerContents(const std::filesystem::path& directory,
                                                    std::span<const TerrainAssetSpec> assets,
                                                    std::string& errorMessage) {
    std::ostringstream output;
    output << "schema=" << COMPLETION_MARKER_SCHEMA_VERSION
           << "\nrevision=" << TERRAIN_MODEL_REVISION << '\n';
    for (const TerrainAssetSpec& asset : assets) {
        const std::optional<CompletionMarkerFileStamp> stamp =
            completionMarkerFileStamp(directory / asset.fileName, asset.byteSize, errorMessage);
        if (!stamp) return std::nullopt;
        output << asset.fileName << ' ' << asset.byteSize << ' ' << asset.sha256 << ' '
               << stamp->device << ' ' << stamp->inode << ' ' << stamp->modifiedSeconds << ' '
               << stamp->modifiedNanoseconds << ' ' << stamp->changedSeconds << ' '
               << stamp->changedNanoseconds << '\n';
    }
    return output.str();
}

bool writeCompletionMarker(const std::filesystem::path& directory,
                           std::span<const TerrainAssetSpec> assets, std::string& errorMessage) {
    const std::optional<std::string> contents =
        completionMarkerContents(directory, assets, errorMessage);
    if (!contents) return false;
    const std::filesystem::path markerPath = directory / COMPLETION_MARKER;
    const std::filesystem::path temporaryPath = directory / ".verified-pack-v1.tmp";
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        errorMessage = "Could not create terrain model completion marker";
        return false;
    }
    output.write(contents->data(), static_cast<std::streamsize>(contents->size()));
    output.flush();
    if (!output.good()) {
        output.close();
        std::error_code removeError;
        std::filesystem::remove(temporaryPath, removeError);
        errorMessage = "Could not finish terrain model completion marker";
        return false;
    }
    output.close();
    if (std::rename(temporaryPath.c_str(), markerPath.c_str()) != 0) {
        std::error_code removeError;
        std::filesystem::remove(temporaryPath, removeError);
        errorMessage = "Could not publish terrain model completion marker";
        return false;
    }
    return true;
}

bool completionMarkerMatches(const std::filesystem::path& directory,
                             std::span<const TerrainAssetSpec> assets) {
    std::ifstream input(directory / COMPLETION_MARKER, std::ios::binary);
    if (!input.is_open()) return false;
    const std::string contents((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) return false;
    std::string ignoredError;
    const std::optional<std::string> expected =
        completionMarkerContents(directory, assets, ignoredError);
    return expected && contents == *expected;
}

TerrainBootstrapSnapshot progressSnapshot(TerrainBootstrapState state, uint64_t completed,
                                          uint64_t total, std::string currentAsset,
                                          std::string detail, bool reusingInstalledPack = false) {
    return TerrainBootstrapSnapshot{
        .state = state,
        .completedBytes = completed,
        .totalBytes = total,
        .reusingInstalledPack = reusingInstalledPack,
        .currentAsset = std::move(currentAsset),
        .detail = std::move(detail),
    };
}

TerrainInstallResult installFailure(TerrainBootstrapFailureCode code, std::string message,
                                    bool retryable) {
    return TerrainInstallResult{
        .status = code == TerrainBootstrapFailureCode::Canceled ? TerrainInstallStatus::Canceled
                                                                : TerrainInstallStatus::Failed,
        .failure = {.code = code, .message = std::move(message), .retryable = retryable},
    };
}

bool validFingerprint(std::string_view fingerprint) {
    return fingerprint.size() == 64 && std::ranges::all_of(fingerprint, [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}

} // namespace

const std::vector<TerrainAssetSpec>& pinnedTerrainAssets() {
    static const std::vector<TerrainAssetSpec> assets{
        {.fileName = "base_model.onnx",
         .byteSize = 2'029'994'361ULL,
         .sha256 = "543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0",
         .url = "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/"
                "ad2df557eca5645f588766101cf3bc3682455c3e/base_model.onnx",
         .kind = TerrainAssetKind::Model},
        {.fileName = "coarse_model.onnx",
         .byteSize = 22'497'125ULL,
         .sha256 = "d6ca15b21b2e35d5e594a9ac7a4249a2376590c0ad2b5b49a1e6e2d033450008",
         .url = "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/"
                "ad2df557eca5645f588766101cf3bc3682455c3e/coarse_model.onnx",
         .kind = TerrainAssetKind::Model},
        {.fileName = "decoder_model.onnx",
         .byteSize = 223'854'143ULL,
         .sha256 = "6473ae47ca6ec4d743d30fe4f5d381fe4158899714eff09b762005bdbdef68c1",
         .url = "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/"
                "ad2df557eca5645f588766101cf3bc3682455c3e/decoder_model.onnx",
         .kind = TerrainAssetKind::Model},
        {.fileName = "pipeline_data.json",
         .byteSize = 12'226ULL,
         .sha256 = "e3132c3ef0c65d8613615f9278ffe23bbd9363ddcd87f1cc6f18456bcc9efe5c",
         .url = "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/"
                "ad2df557eca5645f588766101cf3bc3682455c3e/pipeline_data.json",
         .kind = TerrainAssetKind::Model},
        {.fileName = "world_pipeline_config.json",
         .byteSize = 774ULL,
         .sha256 = "c60f0b74d89317e64cfc623fbfdd828f1b5b2e50aa75020ac4001103381853bd",
         .url = "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/"
                "ad2df557eca5645f588766101cf3bc3682455c3e/world_pipeline_config.json",
         .kind = TerrainAssetKind::Model},
        {.fileName = "onnxruntime-osx-arm64-1.27.1.tgz",
         .byteSize = 31'959'937ULL,
         .sha256 = "e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6",
         .url = "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/"
                "onnxruntime-osx-arm64-1.27.1.tgz",
         .kind = TerrainAssetKind::Runtime},
    };
    return assets;
}

TerrainTransferResult TerrainTransferResult::success() {
    return {.succeeded = true};
}

TerrainTransferResult TerrainTransferResult::cancellation(std::string message) {
    return {.canceled = true, .message = std::move(message)};
}

TerrainTransferResult TerrainTransferResult::failure(std::string message) {
    return {.message = std::move(message)};
}

TerrainVerificationResult
Sha256TerrainAssetVerifier::verify(const std::filesystem::path& path, const TerrainAssetSpec& asset,
                                   const TerrainBootstrapCancellation* cancellation) const {
    std::error_code error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
    if (error || !std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status)) {
        return {.message = "Required file is missing or is not a regular file: " +
                           path.filename().string()};
    }

    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    if (error || fileSize > std::numeric_limits<uint64_t>::max())
        return {.message = "Could not read file size: " + path.filename().string()};
    const uint64_t actualBytes = static_cast<uint64_t>(fileSize);
    if (actualBytes != asset.byteSize) {
        return {.actualBytes = actualBytes,
                .message =
                    "File size does not match the pinned manifest: " + path.filename().string()};
    }

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open())
        return {.actualBytes = actualBytes,
                .message = "Could not open file for verification: " + path.filename().string()};

    StreamingSha256 sha;
    std::vector<uint8_t> buffer(HASH_READ_BYTES);
    while (input) {
        if (cancellation != nullptr && cancellation->canceled()) {
            return {.actualBytes = actualBytes, .message = "Verification canceled"};
        }
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize readCount = input.gcount();
        if (readCount > 0) {
            sha.update(std::span<const uint8_t>(buffer.data(), static_cast<size_t>(readCount)));
        }
    }
    if (!input.eof()) {
        return {.actualBytes = actualBytes,
                .message = "Could not read file for verification: " + path.filename().string()};
    }

    const std::string actualSha256 = digestHex(sha.finish());
    if (actualSha256 != asset.sha256) {
        return {.actualBytes = actualBytes,
                .actualSha256 = actualSha256,
                .message =
                    "SHA-256 does not match the pinned manifest: " + path.filename().string()};
    }
    return {.valid = true, .actualBytes = actualBytes, .actualSha256 = actualSha256};
}

TerrainModelInstaller::TerrainModelInstaller(std::filesystem::path applicationSupportRoot,
                                             TerrainModelTransport& transport,
                                             const TerrainAssetVerifier& verifier,
                                             std::vector<TerrainAssetSpec> assets)
    : applicationSupportRoot_(std::move(applicationSupportRoot))
    , modelsRoot_(applicationSupportRoot_ / "terrain-models" / TERRAIN_MODEL_DIRECTORY)
    , installedPackPath_(modelsRoot_ / TERRAIN_MODEL_REVISION)
    , stagingPackPath_(modelsRoot_ /
                       (std::string(".staging-") + std::string(TERRAIN_MODEL_REVISION)))
    , transport_(transport)
    , verifier_(verifier)
    , assets_(std::move(assets)) {}

uint64_t TerrainModelInstaller::totalDownloadBytes() const noexcept {
    uint64_t total = 0;
    for (const TerrainAssetSpec& asset : assets_)
        total += asset.byteSize;
    return total;
}

bool TerrainModelInstaller::hasInstalledPackCandidate() const {
    std::error_code error;
    return std::filesystem::is_directory(installedPackPath_, error) && !error;
}

TerrainInstallResult
TerrainModelInstaller::verifyPack(const std::filesystem::path& packPath, uint64_t completedBytes,
                                  uint64_t totalBytes,
                                  const TerrainBootstrapCancellation& cancellation,
                                  const TerrainInstallObserver& observer) const {
    // The marker is written only after every pinned asset has passed its
    // SHA-256 check and the pack has been atomically published. Its current
    // file stamps make the normal restart check metadata-only. A missing,
    // old, or changed marker falls back to a full SHA-256 audit without ever
    // downloading over the installed pack.
    const bool markerMatches = completionMarkerMatches(packPath, assets_);
    if (markerMatches) {
        if (observer) {
            observer(progressSnapshot(TerrainBootstrapState::Verifying, totalBytes, totalBytes, {},
                                      "Reusing the verified local terrain model pack; no download "
                                      "is running",
                                      true));
        }
        return {.status = TerrainInstallStatus::Installed,
                .installedPath = packPath,
                .reusedInstalledPack = true};
    }

    uint64_t verifiedBytes = 0;
    for (const TerrainAssetSpec& asset : assets_) {
        if (cancellation.canceled())
            return installFailure(TerrainBootstrapFailureCode::Canceled,
                                  "Terrain model verification canceled", true);
        if (observer) {
            observer(progressSnapshot(
                TerrainBootstrapState::Verifying,
                std::min(totalBytes, completedBytes + verifiedBytes), totalBytes, asset.fileName,
                "Auditing the installed local terrain model pack; no download "
                "is running",
                true));
        }
        const TerrainVerificationResult verification =
            verifier_.verify(packPath / asset.fileName, asset, &cancellation);
        if (!verification.valid) {
            if (cancellation.canceled())
                return installFailure(TerrainBootstrapFailureCode::Canceled,
                                      "Terrain model verification canceled", true);
            return installFailure(TerrainBootstrapFailureCode::Integrity, verification.message,
                                  true);
        }
        verifiedBytes += asset.byteSize;
    }

    std::string markerError;
    if (!writeCompletionMarker(packPath, assets_, markerError)) {
        return installFailure(TerrainBootstrapFailureCode::Filesystem, std::move(markerError),
                              true);
    }
    if (observer) {
        observer(progressSnapshot(TerrainBootstrapState::Verifying, totalBytes, totalBytes, {},
                                  "Refreshed the verified local terrain model completion marker",
                                  true));
    }
    return {.status = TerrainInstallStatus::Installed,
            .installedPath = packPath,
            .reusedInstalledPack = true};
}

TerrainInstallResult
TerrainModelInstaller::prepare(const TerrainBootstrapCancellation& cancellation,
                               const TerrainInstallObserver& observer) const {
    const uint64_t totalBytes = totalDownloadBytes();
    const bool forceRepair = forceRepair_.exchange(false, std::memory_order_acq_rel);
    std::error_code error;
    const bool installedExists = std::filesystem::exists(installedPackPath_, error);
    if (error) {
        return installFailure(
            TerrainBootstrapFailureCode::Filesystem,
            "Could not inspect the terrain model installation path: " + error.message(), true);
    }
    if (!forceRepair && installedExists) {
        TerrainInstallResult existing =
            verifyPack(installedPackPath_, 0, totalBytes, cancellation, observer);
        // A failed installed pack is never replaced as a side effect of an
        // ordinary launch or retry. The explicit repair action is the only
        // path authorized to download a replacement.
        return existing;
    }

    if (cancellation.canceled())
        return installFailure(TerrainBootstrapFailureCode::Canceled,
                              "Terrain model installation canceled", true);
    if (observer) {
        observer(progressSnapshot(TerrainBootstrapState::ModelRequired, 0, totalBytes, {},
                                  forceRepair
                                      ? "A fresh verified terrain model pack was requested"
                                      : "The verified generator v4 terrain model is required"));
    }

    std::filesystem::create_directories(modelsRoot_, error);
    if (error) {
        return installFailure(
            TerrainBootstrapFailureCode::Filesystem,
            "Could not create the terrain model installation directory: " + error.message(), true);
    }

    if (installedExists && !std::filesystem::is_directory(installedPackPath_, error)) {
        return installFailure(TerrainBootstrapFailureCode::Filesystem,
                              "The terrain model installation path is not a directory", true);
    }
    error.clear();
    std::filesystem::create_directories(stagingPackPath_, error);
    if (error) {
        return installFailure(
            TerrainBootstrapFailureCode::Filesystem,
            "Could not create the terrain model staging directory: " + error.message(), true);
    }

    uint64_t completedBytes = 0;
    std::vector<const TerrainAssetSpec*> replacements;
    replacements.reserve(assets_.size());
    for (const TerrainAssetSpec& asset : assets_) {
        if (cancellation.canceled())
            return installFailure(TerrainBootstrapFailureCode::Canceled,
                                  "Terrain model installation canceled", true);

        if (installedExists) {
            const TerrainVerificationResult installed =
                verifier_.verify(installedPackPath_ / asset.fileName, asset, &cancellation);
            if (installed.valid) {
                completedBytes += asset.byteSize;
                continue;
            }
            if (!forceRepair) {
                return installFailure(TerrainBootstrapFailureCode::Integrity, installed.message,
                                      true);
            }
        }

        const std::filesystem::path stagedAsset = stagingPackPath_ / asset.fileName;
        TerrainVerificationResult staged;
        const bool stagedPathExists = std::filesystem::exists(stagedAsset, error);
        if (error) {
            return installFailure(
                TerrainBootstrapFailureCode::Filesystem,
                "Could not inspect a staged terrain model asset: " + error.message(), true);
        }
        if (stagedPathExists) {
            const std::filesystem::file_status status =
                std::filesystem::symlink_status(stagedAsset, error);
            if (error) {
                return installFailure(
                    TerrainBootstrapFailureCode::Filesystem,
                    "Could not inspect a staged terrain model asset: " + error.message(), true);
            }
            if (std::filesystem::is_symlink(status) || !std::filesystem::is_regular_file(status)) {
                std::filesystem::remove(stagedAsset, error);
                if (error) {
                    return installFailure(
                        TerrainBootstrapFailureCode::Filesystem,
                        "Could not reset an invalid staged terrain model asset: " + error.message(),
                        true);
                }
            } else {
                staged = verifier_.verify(stagedAsset, asset, &cancellation);
            }
        }
        if (!staged.valid) {
            const bool oversizedOrCorruptComplete =
                staged.actualBytes >= asset.byteSize && staged.actualBytes != 0;
            if (oversizedOrCorruptComplete) {
                std::filesystem::remove(stagedAsset, error);
                if (error) {
                    return installFailure(
                        TerrainBootstrapFailureCode::Filesystem,
                        "Could not reset an invalid staged terrain model asset: " + error.message(),
                        true);
                }
            }

            uint64_t stagedBytes = 0;
            if (std::filesystem::is_regular_file(stagedAsset, error) && !error) {
                const uintmax_t size = std::filesystem::file_size(stagedAsset, error);
                if (!error && size <= asset.byteSize) stagedBytes = static_cast<uint64_t>(size);
            }
            error.clear();
            if (observer) {
                observer(progressSnapshot(TerrainBootstrapState::Downloading,
                                          completedBytes + stagedBytes, totalBytes, asset.fileName,
                                          stagedBytes == 0
                                              ? "Downloading the pinned terrain model pack"
                                              : "Continuing the pinned terrain model download"));
            }
            const TerrainTransferResult transfer = transport_.download(
                asset, stagedAsset,
                [&](uint64_t assetBytes) {
                    const uint64_t bounded = std::min(assetBytes, asset.byteSize);
                    if (observer) {
                        observer(progressSnapshot(
                            TerrainBootstrapState::Downloading, completedBytes + bounded,
                            totalBytes, asset.fileName,
                            stagedBytes == 0 ? "Downloading the pinned terrain model pack"
                                             : "Continuing the pinned terrain model download"));
                    }
                    return !cancellation.canceled();
                },
                cancellation);
            if (!transfer.succeeded) {
                if (transfer.canceled || cancellation.canceled()) {
                    return installFailure(TerrainBootstrapFailureCode::Canceled,
                                          transfer.message.empty()
                                              ? "Terrain model download canceled"
                                              : transfer.message,
                                          true);
                }
                return installFailure(TerrainBootstrapFailureCode::Download,
                                      transfer.message.empty() ? "Terrain model download failed"
                                                               : transfer.message,
                                      true);
            }

            if (observer) {
                observer(progressSnapshot(TerrainBootstrapState::Verifying, completedBytes,
                                          totalBytes, asset.fileName,
                                          "Verifying the downloaded terrain model asset"));
            }
            staged = verifier_.verify(stagedAsset, asset, &cancellation);
            if (!staged.valid) {
                if (cancellation.canceled()) {
                    return installFailure(TerrainBootstrapFailureCode::Canceled,
                                          "Terrain model verification canceled", true);
                }
                return installFailure(TerrainBootstrapFailureCode::Integrity, staged.message, true);
            }
        }

        replacements.push_back(&asset);
        completedBytes += asset.byteSize;
    }

    if (!installedExists) {
        std::string markerError;
        if (!writeCompletionMarker(stagingPackPath_, assets_, markerError)) {
            return installFailure(TerrainBootstrapFailureCode::Filesystem, std::move(markerError),
                                  true);
        }
        std::filesystem::rename(stagingPackPath_, installedPackPath_, error);
        if (error) {
            return installFailure(
                TerrainBootstrapFailureCode::Filesystem,
                "Could not publish the verified terrain model pack: " + error.message(), true);
        }
    } else {
        if (!replacements.empty()) {
            std::filesystem::remove(installedPackPath_ / COMPLETION_MARKER, error);
            if (error) {
                return installFailure(TerrainBootstrapFailureCode::Filesystem,
                                      "Could not mark the invalid terrain model pack for repair: " +
                                          error.message(),
                                      true);
            }
        }
        for (const TerrainAssetSpec* asset : replacements) {
            const std::filesystem::path source = stagingPackPath_ / asset->fileName;
            const std::filesystem::path destination = installedPackPath_ / asset->fileName;
            if (std::rename(source.c_str(), destination.c_str()) != 0) {
                return installFailure(
                    TerrainBootstrapFailureCode::Filesystem,
                    "Could not publish a repaired terrain model asset: " + asset->fileName, true);
            }
        }
        std::string markerError;
        if (!writeCompletionMarker(installedPackPath_, assets_, markerError)) {
            return installFailure(TerrainBootstrapFailureCode::Filesystem, std::move(markerError),
                                  true);
        }
        std::filesystem::remove_all(stagingPackPath_, error);
    }

    if (observer) {
        observer(progressSnapshot(TerrainBootstrapState::Verifying, totalBytes, totalBytes, {},
                                  "Verified terrain model pack installed"));
    }
    return {.status = TerrainInstallStatus::Installed, .installedPath = installedPackPath_};
}

TerrainRuntimeStepResult TerrainRuntimeStepResult::success() {
    return {.succeeded = true};
}

TerrainRuntimeStepResult TerrainRuntimeStepResult::failureResult(TerrainBootstrapFailureCode code,
                                                                 std::string message,
                                                                 bool retryable) {
    return {.failure = {.code = code, .message = std::move(message), .retryable = retryable}};
}

TerrainGenerationBootstrap::TerrainGenerationBootstrap(TerrainModelInstaller& installer,
                                                       TerrainRuntimePreparation& runtime,
                                                       TerrainInstallObserver observer)
    : installer_(installer)
    , runtime_(runtime)
    , observer_(std::move(observer)) {
    snapshot_.totalBytes = installer_.totalDownloadBytes();
    snapshot_.detail = "The generator v4 terrain model has not been prepared";
}

void TerrainGenerationBootstrap::publish(TerrainBootstrapSnapshot snapshot) {
    TerrainInstallObserver observer;
    TerrainBootstrapSnapshot published;
    {
        std::lock_guard lock(mutex_);
        snapshot_ = std::move(snapshot);
        observer = observer_;
        published = snapshot_;
    }
    if (observer) observer(published);
}

bool TerrainGenerationBootstrap::fail(TerrainBootstrapFailure failure) {
    TerrainBootstrapSnapshot failed;
    {
        std::lock_guard lock(mutex_);
        failed = snapshot_;
    }
    failed.state = TerrainBootstrapState::Failed;
    failed.detail = failure.message;
    failed.failure = std::move(failure);
    publish(std::move(failed));
    return false;
}

bool TerrainGenerationBootstrap::run() {
    {
        std::lock_guard lock(mutex_);
        if (running_ || snapshot_.state == TerrainBootstrapState::Failed ||
            snapshot_.state == TerrainBootstrapState::Ready) {
            return snapshot_.state == TerrainBootstrapState::Ready;
        }
        running_ = true;
    }
    struct RunningReset {
        TerrainGenerationBootstrap& bootstrap;
        ~RunningReset() {
            std::lock_guard lock(bootstrap.mutex_);
            bootstrap.running_ = false;
        }
    } runningReset{*this};

    const TerrainRuntimeStepResult platform = runtime_.qualifyPlatform();
    if (!platform.succeeded) return fail(platform.failure);

    const TerrainInstallResult installation = installer_.prepare(
        cancellation_, [this](const TerrainBootstrapSnapshot& update) { publish(update); });
    if (!installation.installed()) return fail(installation.failure);

    publish(progressSnapshot(TerrainBootstrapState::Compiling, installer_.totalDownloadBytes(),
                             installer_.totalDownloadBytes(), {},
                             installation.reusedInstalledPack
                                 ? "Reusing the local model pack; preparing Core ML sessions"
                                 : "Compiling static Core ML model partitions",
                             installation.reusedInstalledPack));
    const TerrainRuntimeStepResult compilation =
        runtime_.compile(installation.installedPath, cancellation_);
    if (!compilation.succeeded) return fail(compilation.failure);
    if (cancellation_.canceled()) {
        return fail({.code = TerrainBootstrapFailureCode::Canceled,
                     .message = "Terrain model compilation canceled",
                     .retryable = true});
    }

    publish(progressSnapshot(TerrainBootstrapState::Loading, installer_.totalDownloadBytes(),
                             installer_.totalDownloadBytes(), {},
                             installation.reusedInstalledPack
                                 ? "Reusing the local model pack; loading and qualifying runtime"
                                 : "Loading and qualifying the terrain inference runtime",
                             installation.reusedInstalledPack));
    const TerrainRuntimeStepResult loading =
        runtime_.loadAndQualify(installation.installedPath, cancellation_);
    if (!loading.succeeded) return fail(loading.failure);
    if (cancellation_.canceled()) {
        return fail({.code = TerrainBootstrapFailureCode::Canceled,
                     .message = "Terrain runtime loading canceled",
                     .retryable = true});
    }

    const std::optional<std::string> fingerprint = runtime_.qualifiedGenerationFingerprint();
    if (!fingerprint || !validFingerprint(*fingerprint)) {
        return fail({.code = TerrainBootstrapFailureCode::Qualification,
                     .message = "Terrain runtime did not provide a valid generation fingerprint",
                     .retryable = true});
    }
    const std::shared_ptr<learned::WorldGenerationContext> context =
        runtime_.qualifiedGenerationContext();
    if (!context || learned::sha256Hex(context->fingerprint()) != *fingerprint) {
        return fail(
            {.code = TerrainBootstrapFailureCode::Qualification,
             .message = "Terrain runtime did not provide the qualified generation authority",
             .retryable = true});
    }
    {
        std::lock_guard lock(mutex_);
        qualifiedFingerprint_ = fingerprint;
    }

    publish(
        progressSnapshot(TerrainBootstrapState::Ready, installer_.totalDownloadBytes(),
                         installer_.totalDownloadBytes(), {},
                         installation.reusedInstalledPack
                             ? "Generator v4 terrain authority is ready from the local model pack"
                             : "Generator v4 terrain authority is ready",
                         installation.reusedInstalledPack));
    return true;
}

bool TerrainGenerationBootstrap::retry() {
    {
        std::lock_guard lock(mutex_);
        if (running_ || snapshot_.state != TerrainBootstrapState::Failed || !snapshot_.failure ||
            !snapshot_.failure->retryable) {
            return false;
        }
        snapshot_ = TerrainBootstrapSnapshot{
            .state = TerrainBootstrapState::ModelRequired,
            .totalBytes = installer_.totalDownloadBytes(),
            .detail = "Retrying generator v4 terrain preparation",
        };
        qualifiedFingerprint_.reset();
    }
    cancellation_.reset();
    return run();
}

bool TerrainGenerationBootstrap::repair() {
    {
        std::lock_guard lock(mutex_);
        if (running_) return false;
        snapshot_ = TerrainBootstrapSnapshot{
            .state = TerrainBootstrapState::ModelRequired,
            .totalBytes = installer_.totalDownloadBytes(),
            .detail = "Repairing generator v4 terrain preparation",
        };
        qualifiedFingerprint_.reset();
    }
    cancellation_.reset();
    installer_.requestRepair();
    return run();
}

void TerrainGenerationBootstrap::cancel() noexcept {
    cancellation_.cancel();
}

TerrainBootstrapSnapshot TerrainGenerationBootstrap::snapshot() const {
    std::lock_guard lock(mutex_);
    return snapshot_;
}

bool TerrainGenerationBootstrap::ready() const {
    return snapshot().state == TerrainBootstrapState::Ready;
}

std::optional<std::filesystem::path> TerrainGenerationBootstrap::worldPath() const {
    if (!ready()) return std::nullopt;
    return installer_.applicationSupportRoot() / V4_WORLD_DIRECTORY;
}

std::optional<std::string> TerrainGenerationBootstrap::qualifiedGenerationFingerprint() const {
    std::lock_guard lock(mutex_);
    if (snapshot_.state != TerrainBootstrapState::Ready) return std::nullopt;
    return qualifiedFingerprint_;
}

std::shared_ptr<learned::WorldGenerationContext>
TerrainGenerationBootstrap::qualifiedGenerationContext() const {
    std::lock_guard lock(mutex_);
    if (snapshot_.state != TerrainBootstrapState::Ready) return nullptr;
    return runtime_.qualifiedGenerationContext();
}

bool TerrainGenerationBootstrap::bindWorldProfile(const std::filesystem::path& worldPath) {
    {
        std::lock_guard lock(mutex_);
        if (running_ || snapshot_.state != TerrainBootstrapState::Ready) return false;
    }
    if (worldPath.empty()) {
        return fail({.code = TerrainBootstrapFailureCode::Filesystem,
                     .message = "The generator v4 profile path is empty",
                     .retryable = false});
    }

    const TerrainRuntimeStepResult binding = runtime_.bindWorldProfile(worldPath);
    if (!binding.succeeded) return fail(binding.failure);

    const std::optional<std::string> fingerprint = runtime_.qualifiedGenerationFingerprint();
    const std::shared_ptr<learned::WorldGenerationContext> context =
        runtime_.qualifiedGenerationContext();
    if (!fingerprint || !validFingerprint(*fingerprint) || !context ||
        learned::sha256Hex(context->fingerprint()) != *fingerprint) {
        return fail({.code = TerrainBootstrapFailureCode::Qualification,
                     .message = "Terrain runtime did not bind the qualified authority to the "
                                "selected generator v4 profile",
                     .retryable = true});
    }
    return true;
}

} // namespace worldgen::bootstrap
