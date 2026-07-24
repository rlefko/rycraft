#pragma once

#include "world/generator_v4.hpp"

#include <atomic>
#include <compare>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace worldgen::learned {
class WorldGenerationContext;
}

namespace worldgen::bootstrap {

inline constexpr std::string_view TERRAIN_MODEL_REVISION =
    "ad2df557eca5645f588766101cf3bc3682455c3e";
inline constexpr uint32_t TERRAIN_MODEL_MANIFEST_SCHEMA_VERSION = 1;
inline constexpr std::string_view TERRAIN_MODEL_DIRECTORY = "terrain-diffusion-30m-onnx";
inline constexpr std::string_view V4_WORLD_DIRECTORY = worldgen::v4_profile::WORLD_DIRECTORY;

enum class TerrainAssetKind : uint8_t {
    Model,
    Runtime,
};

struct TerrainAssetSpec {
    std::string fileName;
    uint64_t byteSize = 0;
    std::string sha256;
    std::string url;
    TerrainAssetKind kind = TerrainAssetKind::Model;

    auto operator<=>(const TerrainAssetSpec&) const = default;
};

[[nodiscard]] const std::vector<TerrainAssetSpec>& pinnedTerrainAssets();
[[nodiscard]] std::filesystem::path defaultRycraftApplicationSupportPath();

enum class TerrainBootstrapState : uint8_t {
    ModelRequired,
    Downloading,
    Verifying,
    Compiling,
    Loading,
    Ready,
    Failed,
};

enum class TerrainBootstrapFailureCode : uint8_t {
    None,
    Canceled,
    UnsupportedPlatform,
    Filesystem,
    Download,
    Integrity,
    RuntimeCompilation,
    RuntimeLoading,
    Qualification,
};

struct TerrainBootstrapFailure {
    TerrainBootstrapFailureCode code = TerrainBootstrapFailureCode::None;
    std::string message;
    bool retryable = false;

    auto operator<=>(const TerrainBootstrapFailure&) const = default;
};

struct TerrainBootstrapSnapshot {
    TerrainBootstrapState state = TerrainBootstrapState::ModelRequired;
    uint64_t completedBytes = 0;
    uint64_t totalBytes = 0;
    // True after startup has selected an already installed pack. This is
    // deliberately separate from progress bytes: a local integrity check or
    // a Core ML session setup must never look like another model download.
    bool reusingInstalledPack = false;
    std::string currentAsset;
    std::string detail;
    std::optional<TerrainBootstrapFailure> failure;
};

class TerrainBootstrapCancellation {
public:
    void cancel() noexcept { canceled_.store(true, std::memory_order_release); }
    void reset() noexcept { canceled_.store(false, std::memory_order_release); }
    [[nodiscard]] bool canceled() const noexcept {
        return canceled_.load(std::memory_order_acquire);
    }

private:
    std::atomic<bool> canceled_{false};
};

using TerrainDownloadProgress = std::function<bool(uint64_t completedBytes)>;

struct TerrainTransferResult {
    bool succeeded = false;
    bool canceled = false;
    std::string message;

    static TerrainTransferResult success();
    static TerrainTransferResult cancellation(std::string message = "Download canceled");
    static TerrainTransferResult failure(std::string message);
};

class TerrainModelTransport {
public:
    virtual ~TerrainModelTransport() = default;

    virtual TerrainTransferResult download(const TerrainAssetSpec& asset,
                                           const std::filesystem::path& destination,
                                           const TerrainDownloadProgress& progress,
                                           const TerrainBootstrapCancellation& cancellation) = 0;
};

// Writes into the installer-provided persistent staging path. An existing
// short file is continued with an HTTP range request, so a retry does not
// discard bytes that already reached disk. Tests inject a transport and never
// access the network.
[[nodiscard]] std::unique_ptr<TerrainModelTransport> makeAppleTerrainModelTransport();

struct TerrainVerificationResult {
    bool valid = false;
    uint64_t actualBytes = 0;
    std::string actualSha256;
    std::string message;
};

class TerrainAssetVerifier {
public:
    virtual ~TerrainAssetVerifier() = default;
    virtual TerrainVerificationResult
    verify(const std::filesystem::path& path, const TerrainAssetSpec& asset,
           const TerrainBootstrapCancellation* cancellation = nullptr) const = 0;
};

class Sha256TerrainAssetVerifier final : public TerrainAssetVerifier {
public:
    TerrainVerificationResult
    verify(const std::filesystem::path& path, const TerrainAssetSpec& asset,
           const TerrainBootstrapCancellation* cancellation = nullptr) const override;
};

enum class TerrainInstallStatus : uint8_t {
    Installed,
    Canceled,
    Failed,
};

struct TerrainInstallResult {
    TerrainInstallStatus status = TerrainInstallStatus::Failed;
    std::filesystem::path installedPath;
    TerrainBootstrapFailure failure;
    bool reusedInstalledPack = false;

    [[nodiscard]] bool installed() const noexcept {
        return status == TerrainInstallStatus::Installed;
    }
};

using TerrainInstallObserver = std::function<void(const TerrainBootstrapSnapshot& snapshot)>;

class TerrainModelInstaller {
public:
    TerrainModelInstaller(std::filesystem::path applicationSupportRoot,
                          TerrainModelTransport& transport, const TerrainAssetVerifier& verifier,
                          std::vector<TerrainAssetSpec> assets = pinnedTerrainAssets());

    TerrainInstallResult prepare(const TerrainBootstrapCancellation& cancellation,
                                 const TerrainInstallObserver& observer) const;

    [[nodiscard]] const std::filesystem::path& applicationSupportRoot() const noexcept {
        return applicationSupportRoot_;
    }
    [[nodiscard]] const std::filesystem::path& installedPackPath() const noexcept {
        return installedPackPath_;
    }
    [[nodiscard]] const std::filesystem::path& stagingPackPath() const noexcept {
        return stagingPackPath_;
    }
    // A pack candidate is enough to start verification automatically. The
    // installer still validates every pinned asset before returning it.
    [[nodiscard]] bool hasInstalledPackCandidate() const;
    [[nodiscard]] uint64_t totalDownloadBytes() const noexcept;
    void requestRepair() noexcept { forceRepair_.store(true, std::memory_order_release); }

private:
    std::filesystem::path applicationSupportRoot_;
    std::filesystem::path modelsRoot_;
    std::filesystem::path installedPackPath_;
    std::filesystem::path stagingPackPath_;
    TerrainModelTransport& transport_;
    const TerrainAssetVerifier& verifier_;
    std::vector<TerrainAssetSpec> assets_;
    mutable std::atomic<bool> forceRepair_{false};

    [[nodiscard]] TerrainInstallResult verifyPack(const std::filesystem::path& packPath,
                                                  uint64_t completedBytes, uint64_t totalBytes,
                                                  const TerrainBootstrapCancellation& cancellation,
                                                  const TerrainInstallObserver& observer) const;
};

struct TerrainRuntimeStepResult {
    bool succeeded = false;
    TerrainBootstrapFailure failure;

    static TerrainRuntimeStepResult success();
    static TerrainRuntimeStepResult failureResult(TerrainBootstrapFailureCode code,
                                                  std::string message, bool retryable);
};

class TerrainRuntimePreparation {
public:
    virtual ~TerrainRuntimePreparation() = default;

    virtual TerrainRuntimeStepResult qualifyPlatform() = 0;
    virtual TerrainRuntimeStepResult compile(const std::filesystem::path& installedPack,
                                             const TerrainBootstrapCancellation& cancellation) = 0;
    virtual TerrainRuntimeStepResult
    loadAndQualify(const std::filesystem::path& installedPack,
                   const TerrainBootstrapCancellation& cancellation) = 0;
    // Binds the qualified authority's persisted terrain and hydrology stores
    // to a profile that has already passed its immutable metadata check.
    // Implementations must not retain a prior profile's storage roots.
    virtual TerrainRuntimeStepResult bindWorldProfile(const std::filesystem::path& worldPath) = 0;
    [[nodiscard]] virtual std::optional<std::string> qualifiedGenerationFingerprint() const {
        return std::nullopt;
    }
    [[nodiscard]] virtual std::shared_ptr<learned::WorldGenerationContext>
    qualifiedGenerationContext() const {
        return nullptr;
    }
};

// Coordinates installer and inference-runtime work without constructing a
// SaveManager, World, or v4 directory. A caller may construct the v4 world
// only after worldPath() returns a value.
class TerrainGenerationBootstrap {
public:
    TerrainGenerationBootstrap(TerrainModelInstaller& installer, TerrainRuntimePreparation& runtime,
                               TerrainInstallObserver observer = {});

    bool run();
    bool retry();
    bool repair();
    void cancel() noexcept;

    [[nodiscard]] TerrainBootstrapSnapshot snapshot() const;
    [[nodiscard]] bool ready() const;
    [[nodiscard]] std::optional<std::filesystem::path> worldPath() const;
    [[nodiscard]] std::optional<std::string> qualifiedGenerationFingerprint() const;
    [[nodiscard]] std::shared_ptr<learned::WorldGenerationContext>
    qualifiedGenerationContext() const;
    // Rebinds the already-qualified authority to the profile selected by
    // openQualifiedV4World. A failure returns the bootstrap to its visible,
    // retryable failure state before any World is constructed.
    bool bindWorldProfile(const std::filesystem::path& worldPath);

private:
    TerrainModelInstaller& installer_;
    TerrainRuntimePreparation& runtime_;
    TerrainInstallObserver observer_;
    TerrainBootstrapCancellation cancellation_;
    mutable std::mutex mutex_;
    TerrainBootstrapSnapshot snapshot_;
    std::optional<std::string> qualifiedFingerprint_;
    bool running_ = false;

    void publish(TerrainBootstrapSnapshot snapshot);
    bool fail(TerrainBootstrapFailure failure);
};

} // namespace worldgen::bootstrap
