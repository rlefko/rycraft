#include "world/hydrology_authority.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#include <fcntl.h>
#include <lz4.h>
#include <sys/file.h>
#include <unistd.h>

namespace worldgen::hydrology {

namespace {

using learned::AuthorityQuality;
using learned::AuthorityResult;
using learned::GenerationFailure;
using learned::GenerationFailureCode;
using learned::Sha256Digest;

constexpr size_t HEADER_BYTES = 88;
constexpr size_t HEADER_CHECKSUM_OFFSET = 84;
constexpr uint8_t LZ4_COMPRESSION = 1;

GenerationFailure makeFailure(GenerationFailureCode code, std::string message, bool retriable) {
    return GenerationFailure{
        .code = code,
        .message = std::move(message),
        .retriable = retriable,
    };
}

void appendU16(std::vector<uint8_t>& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8U));
}

void appendU32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

void appendU64(std::vector<uint8_t>& bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<uint8_t>(value >> shift));
}

bool readU16(std::span<const uint8_t> bytes, size_t& offset, uint16_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint16_t)) return false;
    value = static_cast<uint16_t>(bytes[offset]) |
            static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8U);
    offset += sizeof(uint16_t);
    return true;
}

bool readU32(std::span<const uint8_t> bytes, size_t& offset, uint32_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint32_t)) return false;
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    return true;
}

bool readU64(std::span<const uint8_t> bytes, size_t& offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint64_t)) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    return true;
}

uint32_t crc32(std::span<const uint8_t> bytes) {
    static constexpr std::array<uint32_t, 256> TABLE = [] {
        std::array<uint32_t, 256> result{};
        for (uint32_t value = 0; value < result.size(); ++value) {
            uint32_t remainder = value;
            for (unsigned bit = 0; bit < 8; ++bit) {
                const uint32_t polynomialMask = 0U - (remainder & 1U);
                remainder = (remainder >> 1U) ^ (0xEDB88320U & polynomialMask);
            }
            result[value] = remainder;
        }
        return result;
    }();
    uint32_t checksum = 0xFFFFFFFFU;
    for (const uint8_t byte : bytes)
        checksum = (checksum >> 8U) ^ TABLE[(checksum ^ byte) & 0xFFU];
    return ~checksum;
}

struct Header {
    AuthorityQuality quality = AuthorityQuality::FINAL;
    uint16_t hydrologyRevision = 0;
    HydrologyPageCoordinate coordinate;
    uint64_t seed = 0;
    Sha256Digest fingerprint{};
    uint32_t payloadBytes = 0;
    uint32_t compressedBytes = 0;
    uint32_t payloadChecksum = 0;
};

AuthorityResult<Header> decodeHeader(std::span<const uint8_t> bytes) {
    if (bytes.size() < HEADER_BYTES) {
        return AuthorityResult<Header>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority header is truncated", true));
    }
    if (bytes[0] != 'R' || bytes[1] != 'Y' || bytes[2] != 'H' || bytes[3] != 'Y') {
        return AuthorityResult<Header>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority magic is invalid", true));
    }

    size_t offset = 4;
    uint16_t schema = 0;
    uint16_t headerBytes = 0;
    if (!readU16(bytes, offset, schema) || !readU16(bytes, offset, headerBytes) ||
        schema != HYDROLOGY_AUTHORITY_SCHEMA_VERSION || headerBytes != HEADER_BYTES) {
        return AuthorityResult<Header>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority schema is invalid", true));
    }

    const uint8_t compression = bytes[offset++];
    const uint8_t quality = bytes[offset++];
    Header header;
    if (quality > static_cast<uint8_t>(AuthorityQuality::FINAL)) {
        return AuthorityResult<Header>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority quality is invalid", true));
    }
    header.quality = static_cast<AuthorityQuality>(quality);
    if (!readU16(bytes, offset, header.hydrologyRevision) || compression != LZ4_COMPRESSION ||
        header.hydrologyRevision == 0) {
        return AuthorityResult<Header>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority encoding is invalid", true));
    }

    uint64_t pageX = 0;
    uint64_t pageZ = 0;
    if (!readU64(bytes, offset, pageX) || !readU64(bytes, offset, pageZ) ||
        !readU64(bytes, offset, header.seed)) {
        return AuthorityResult<Header>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority coordinates are truncated", true));
    }
    header.coordinate = {
        .x = static_cast<int64_t>(pageX),
        .z = static_cast<int64_t>(pageZ),
    };
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), header.fingerprint.size(),
                header.fingerprint.begin());
    offset += header.fingerprint.size();

    uint16_t pageEdge = 0;
    uint16_t reservedWord = 0;
    uint32_t headerChecksum = 0;
    if (!readU16(bytes, offset, pageEdge) || !readU16(bytes, offset, reservedWord) ||
        !readU32(bytes, offset, header.payloadBytes) ||
        !readU32(bytes, offset, header.compressedBytes) ||
        !readU32(bytes, offset, header.payloadChecksum) ||
        !readU32(bytes, offset, headerChecksum) || offset != HEADER_BYTES ||
        pageEdge != HYDROLOGY_AUTHORITY_PAGE_BLOCK_EDGE || reservedWord != 0 ||
        header.payloadBytes == 0 || header.payloadBytes > HYDROLOGY_AUTHORITY_MAX_PAYLOAD_BYTES ||
        header.compressedBytes == 0 ||
        headerChecksum != crc32(bytes.first(HEADER_CHECKSUM_OFFSET))) {
        return AuthorityResult<Header>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority header validation failed", true));
    }
    const int maximumCompressed = LZ4_compressBound(static_cast<int>(header.payloadBytes));
    if (maximumCompressed <= 0 ||
        header.compressedBytes > static_cast<uint32_t>(maximumCompressed)) {
        return AuthorityResult<Header>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority compressed size is invalid", true));
    }
    return AuthorityResult<Header>::ready(header);
}

bool writeAll(int descriptor, std::span<const uint8_t> bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

AuthorityResult<int> acquireExclusivePagePublicationLock(const std::filesystem::path& pagePath) {
    std::filesystem::path lockPath = pagePath;
    lockPath += ".lock";
    const int descriptor = ::open(lockPath.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        return AuthorityResult<int>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority publication lock could not be opened", true));
    }
    while (::flock(descriptor, LOCK_EX) != 0) {
        if (errno == EINTR) continue;
        static_cast<void>(::close(descriptor));
        return AuthorityResult<int>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority publication lock could not be acquired", true));
    }
    return AuthorityResult<int>::ready(descriptor);
}

class ScopedPagePublicationLock {
public:
    explicit ScopedPagePublicationLock(int descriptor) : descriptor_(descriptor) {}
    ~ScopedPagePublicationLock() {
        if (descriptor_ >= 0) static_cast<void>(::close(descriptor_));
    }

    ScopedPagePublicationLock(const ScopedPagePublicationLock&) = delete;
    ScopedPagePublicationLock& operator=(const ScopedPagePublicationLock&) = delete;

private:
    int descriptor_ = -1;
};

void synchronizeContainingDirectory(const std::filesystem::path& path) noexcept {
    const int descriptor = ::open(path.parent_path().c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return;
    static_cast<void>(::fsync(descriptor));
    static_cast<void>(::close(descriptor));
}

} // namespace

HydrologyAuthorityStore::HydrologyAuthorityStore(std::filesystem::path root,
                                                 learned::GenerationIdentity identity,
                                                 learned::AuthorityQuality quality,
                                                 std::shared_ptr<const TestHooks> testHooks)
    : root_(std::move(root))
    , identity_(std::move(identity))
    , quality_(quality)
    , testHooks_(std::move(testHooks)) {}

std::filesystem::path HydrologyAuthorityStore::pagePath(HydrologyPageCoordinate coordinate) const {
    return root_ /
           ("p." + std::to_string(coordinate.x) + "." + std::to_string(coordinate.z) + ".ryhy");
}

AuthorityResult<std::vector<uint8_t>>
HydrologyAuthorityStore::load(HydrologyPageCoordinate coordinate) const {
    if (root_.empty() || !identity_.valid() ||
        (quality_ != AuthorityQuality::PREVIEW && quality_ != AuthorityQuality::FINAL)) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Hydrology authority load received an invalid identity or quality", false));
    }
    const std::filesystem::path path = pagePath(coordinate);
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority page could not be inspected", true));
    }
    if (!exists) {
        return AuthorityResult<std::vector<uint8_t>>::deferred(
            makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                        "Hydrology authority page is not persisted", true));
    }
    if (!std::filesystem::is_regular_file(path, error) || error) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority path is not a regular file", true));
    }

    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    const int maximumCompressed =
        LZ4_compressBound(static_cast<int>(HYDROLOGY_AUTHORITY_MAX_PAYLOAD_BYTES));
    if (error || maximumCompressed <= 0 || fileSize < HEADER_BYTES ||
        fileSize > HEADER_BYTES + static_cast<uintmax_t>(maximumCompressed)) {
        return AuthorityResult<std::vector<uint8_t>>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Hydrology authority file size is invalid", true));
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return AuthorityResult<std::vector<uint8_t>>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Hydrology authority page could not be opened", true));
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input.good()) {
        return AuthorityResult<std::vector<uint8_t>>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Hydrology authority page could not be read", true));
    }

    auto decoded = decodeHeader(bytes);
    if (!decoded.isReady())
        return AuthorityResult<std::vector<uint8_t>>::failed(*decoded.failure());
    const Header& header = *decoded.value();
    if (header.coordinate != coordinate) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority page key does not match its path", true));
    }
    if (header.quality != quality_) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                        "Hydrology authority page belongs to another authority quality", false));
    }
    if (header.seed != identity_.seed || header.fingerprint != identity_.fingerprint() ||
        header.hydrologyRevision != identity_.hydrologyRevision) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                        "Hydrology authority page belongs to another generation identity", false));
    }
    if (bytes.size() != HEADER_BYTES + header.compressedBytes) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority compressed payload is truncated", true));
    }

    std::vector<uint8_t> payload(header.payloadBytes);
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(bytes.data() + HEADER_BYTES),
        reinterpret_cast<char*>(payload.data()), static_cast<int>(header.compressedBytes),
        static_cast<int>(payload.size()));
    if (written != static_cast<int>(payload.size()) || crc32(payload) != header.payloadChecksum) {
        return AuthorityResult<std::vector<uint8_t>>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Hydrology authority payload validation failed", true));
    }
    return AuthorityResult<std::vector<uint8_t>>::ready(std::move(payload));
}

AuthorityResult<bool> HydrologyAuthorityStore::write(HydrologyPageCoordinate coordinate,
                                                     std::span<const uint8_t> payload) const {
    return writeImpl(coordinate, payload, std::nullopt);
}

AuthorityResult<bool>
HydrologyAuthorityStore::replaceCorruptPayload(HydrologyPageCoordinate coordinate,
                                               std::span<const uint8_t> expectedCorruptPayload,
                                               std::span<const uint8_t> replacementPayload) const {
    if (expectedCorruptPayload.empty()) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST,
            "Hydrology authority corrupt-payload replacement requires expected bytes", false));
    }
    return writeImpl(coordinate, replacementPayload, expectedCorruptPayload);
}

AuthorityResult<bool> HydrologyAuthorityStore::writeImpl(
    HydrologyPageCoordinate coordinate, std::span<const uint8_t> payload,
    std::optional<std::span<const uint8_t>> expectedCorruptPayload) const {
    if (root_.empty() || !identity_.valid() ||
        (quality_ != AuthorityQuality::PREVIEW && quality_ != AuthorityQuality::FINAL) ||
        payload.empty() || payload.size() > HYDROLOGY_AUTHORITY_MAX_PAYLOAD_BYTES ||
        payload.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST,
            "Hydrology authority write received an invalid identity or payload", false));
    }

    const auto matchesExisting = [&](std::span<const uint8_t> existing) {
        if (std::ranges::equal(existing, payload)) return AuthorityResult<bool>::ready(true);
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
            "Hydrology authority page is immutable for its generation identity", false));
    };
    const auto matchesExpectedCorruptPayload = [&](std::span<const uint8_t> existing) {
        return expectedCorruptPayload && std::ranges::equal(existing, *expectedCorruptPayload);
    };
    const auto changedDuringRepair = [&] {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
            "Hydrology authority page changed while a corrupt payload was being rebuilt", false));
    };

    auto existing = load(coordinate);
    if (existing.isReady()) {
        const auto matchingReplacement = matchesExisting(*existing.value());
        if (matchingReplacement.isReady() || !expectedCorruptPayload) return matchingReplacement;
        if (!matchesExpectedCorruptPayload(*existing.value())) return changedDuringRepair();
    }
    if (existing.status() == learned::AuthorityStatus::FAILED &&
        (!existing.failure() || existing.failure()->code != GenerationFailureCode::CORRUPT_PAGE)) {
        return AuthorityResult<bool>::failed(
            existing.failure()
                ? *existing.failure()
                : makeFailure(GenerationFailureCode::IO_ERROR,
                              "Hydrology authority load failed without a failure reason", true));
    }

    const int maximum = LZ4_compressBound(static_cast<int>(payload.size()));
    if (maximum <= 0) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Hydrology authority compression bound failed", true));
    }
    std::vector<uint8_t> compressed(static_cast<size_t>(maximum));
    const int compressedBytes = LZ4_compress_default(reinterpret_cast<const char*>(payload.data()),
                                                     reinterpret_cast<char*>(compressed.data()),
                                                     static_cast<int>(payload.size()), maximum);
    if (compressedBytes <= 0) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Hydrology authority compression failed", true));
    }
    compressed.resize(static_cast<size_t>(compressedBytes));

    std::vector<uint8_t> fileBytes;
    fileBytes.reserve(HEADER_BYTES + compressed.size());
    fileBytes.insert(fileBytes.end(), {'R', 'Y', 'H', 'Y'});
    appendU16(fileBytes, HYDROLOGY_AUTHORITY_SCHEMA_VERSION);
    appendU16(fileBytes, static_cast<uint16_t>(HEADER_BYTES));
    fileBytes.push_back(LZ4_COMPRESSION);
    fileBytes.push_back(static_cast<uint8_t>(quality_));
    appendU16(fileBytes, identity_.hydrologyRevision);
    appendU64(fileBytes, static_cast<uint64_t>(coordinate.x));
    appendU64(fileBytes, static_cast<uint64_t>(coordinate.z));
    appendU64(fileBytes, identity_.seed);
    const learned::Sha256Digest fingerprint = identity_.fingerprint();
    fileBytes.insert(fileBytes.end(), fingerprint.begin(), fingerprint.end());
    appendU16(fileBytes, HYDROLOGY_AUTHORITY_PAGE_BLOCK_EDGE);
    appendU16(fileBytes, 0);
    appendU32(fileBytes, static_cast<uint32_t>(payload.size()));
    appendU32(fileBytes, static_cast<uint32_t>(compressed.size()));
    appendU32(fileBytes, crc32(payload));
    appendU32(fileBytes, crc32(fileBytes));
    if (fileBytes.size() != HEADER_BYTES) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority header construction failed", false));
    }
    fileBytes.insert(fileBytes.end(), compressed.begin(), compressed.end());

    const std::filesystem::path path = pagePath(coordinate);
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority directory could not be created", true));
    }
    static std::atomic<uint64_t> temporarySequence{0};
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(::getpid()) + "." +
        std::to_string(temporarySequence.fetch_add(1, std::memory_order_relaxed));
    const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority temporary file could not be opened", true));
    }
    const bool wrote = writeAll(descriptor, fileBytes);
    const bool synchronized = wrote && ::fsync(descriptor) == 0;
    const bool closed = ::close(descriptor) == 0;
    if (!wrote || !synchronized || !closed) {
        std::filesystem::remove(temporary, error);
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority staging file could not be synchronized", true));
    }

    const auto removeTemporary = [&] {
        std::error_code cleanupError;
        std::filesystem::remove(temporary, cleanupError);
    };
    if (testHooks_ && testHooks_->beforeExclusivePublish) {
        try {
            testHooks_->beforeExclusivePublish();
        } catch (const std::exception& exception) {
            removeTemporary();
            return AuthorityResult<bool>::failed(makeFailure(
                GenerationFailureCode::IO_ERROR,
                std::string("Hydrology authority publication test hook threw: ") + exception.what(),
                true));
        } catch (...) {
            removeTemporary();
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Hydrology authority publication test hook threw", true));
        }
    }

    auto lock = acquireExclusivePagePublicationLock(path);
    if (!lock.isReady()) {
        removeTemporary();
        return AuthorityResult<bool>::failed(*lock.failure());
    }
    const ScopedPagePublicationLock heldLock(*lock.value());

    existing = load(coordinate);
    bool replacingCorruptPage = false;
    if (existing.isReady()) {
        const auto matchingReplacement = matchesExisting(*existing.value());
        if (matchingReplacement.isReady() || !expectedCorruptPayload) {
            removeTemporary();
            return matchingReplacement;
        }
        if (!matchesExpectedCorruptPayload(*existing.value())) {
            removeTemporary();
            return changedDuringRepair();
        }
        replacingCorruptPage = true;
    }
    replacingCorruptPage =
        replacingCorruptPage ||
        (existing.status() == learned::AuthorityStatus::FAILED && existing.failure() &&
         existing.failure()->code == GenerationFailureCode::CORRUPT_PAGE);
    if (existing.status() == learned::AuthorityStatus::FAILED && !replacingCorruptPage) {
        removeTemporary();
        return AuthorityResult<bool>::failed(
            existing.failure()
                ? *existing.failure()
                : makeFailure(GenerationFailureCode::IO_ERROR,
                              "Hydrology authority load failed without a failure reason", true));
    }

    const int publishResult = replacingCorruptPage ? ::rename(temporary.c_str(), path.c_str())
                                                   : ::link(temporary.c_str(), path.c_str());
    if (publishResult != 0) {
        const int publishError = errno;
        removeTemporary();
        if (!replacingCorruptPage && publishError == EEXIST) {
            // An unsynchronized or older process may have won publication.
            // Never replace it: reread the immutable payload and require an
            // exact byte match before accepting the contention outcome.
            const auto contender = load(coordinate);
            if (contender.isReady()) return matchesExisting(*contender.value());
            if (contender.failure()) return AuthorityResult<bool>::failed(*contender.failure());
        }
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Hydrology authority page could not be published exclusively", true));
    }
    if (!replacingCorruptPage) removeTemporary();
    synchronizeContainingDirectory(path);

    const auto published = load(coordinate);
    if (published.isReady()) return matchesExisting(*published.value());
    if (published.failure()) return AuthorityResult<bool>::failed(*published.failure());
    return AuthorityResult<bool>::failed(
        makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                    "Hydrology authority post-write validation failed", true));
}

} // namespace worldgen::hydrology
