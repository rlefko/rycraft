#include "world/learned_terrain.hpp"

#include "common/trace.hpp"
#include "world/chunk_pos.hpp"
#include "world/learned_authority_graph.hpp"
#include "world/native_hydrology.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <lz4.h>
#include <sys/file.h>
#include <unistd.h>

namespace worldgen::learned {

namespace {

constexpr uint64_t PCG_MULTIPLIER = 6'364'136'223'846'793'005ULL;
constexpr uint64_t PCG_INCREMENT = 1'442'695'040'888'963'407ULL;
constexpr double INVERSE_TWO_TO_32 = 1.0 / 4'294'967'296.0;
constexpr size_t AUTHORITY_HEADER_BYTES = 92;
constexpr size_t AUTHORITY_HEADER_CHECKSUM_OFFSET = 88;
constexpr uint8_t LZ4_COMPRESSION = 1;
constexpr size_t TERRAIN_CHANNEL_COUNT = 6;
constexpr size_t TERRAIN_PAYLOAD_BYTES =
    AUTHORITY_PAGE_SAMPLE_COUNT * TERRAIN_CHANNEL_COUNT * sizeof(uint16_t);
constexpr uint16_t TRANSIENT_GRID_SCHEMA_VERSION = 1;
constexpr size_t TRANSIENT_GRID_HEADER_BYTES = 104;
constexpr size_t TRANSIENT_GRID_HEADER_CHECKSUM_OFFSET = 100;

GenerationFailure makeFailure(GenerationFailureCode code, std::string message, bool retriable) {
    return GenerationFailure{
        .code = code,
        .message = std::move(message),
        .retriable = retriable,
    };
}

bool validQuality(AuthorityQuality quality) {
    return quality == AuthorityQuality::PREVIEW || quality == AuthorityQuality::FINAL;
}

bool validRequestPriority(AuthorityRequestPriority priority) {
    return priority >= AuthorityRequestPriority::SPAWN &&
           priority <= AuthorityRequestPriority::SPECULATIVE_PREFETCH;
}

AuthorityRequestPriority defaultRequestPriority(AuthorityQuality quality) {
    return quality == AuthorityQuality::PREVIEW ? AuthorityRequestPriority::COARSE_PREVIEW
                                                : AuthorityRequestPriority::EXPLORATION_EXACT;
}

bool validGeometry(WindowGeometry geometry) {
    return geometry.edge != 0 && geometry.stride != 0 && geometry.stride <= geometry.edge &&
           geometry.inferenceSteps != 0 && geometry.batchSize != 0;
}

bool hasNonzeroByte(const Sha256Digest& digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte != 0; });
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

void appendDigest(std::vector<uint8_t>& bytes, const Sha256Digest& digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
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
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    }
    return true;
}

bool readU64(std::span<const uint8_t> bytes, size_t& offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(uint64_t)) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    }
    return true;
}

uint32_t crc32(std::span<const uint8_t> bytes) {
    uint32_t checksum = 0xFFFFFFFFU;
    for (uint8_t byte : bytes) {
        checksum ^= byte;
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t polynomialMask = 0U - (checksum & 1U);
            checksum = (checksum >> 1U) ^ (0xEDB88320U & polynomialMask);
        }
    }
    return ~checksum;
}

constexpr std::array<uint32_t, 64> SHA256_CONSTANTS{
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U,
    0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU,
    0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU,
    0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U,
    0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU,
    0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
    0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U,
    0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U,
    0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U,
    0xC67178F2U,
};

void sha256Transform(std::array<uint32_t, 8>& state, const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (size_t index = 0; index < 16; ++index) {
        const size_t offset = index * 4;
        words[index] = static_cast<uint32_t>(block[offset]) << 24U |
                       static_cast<uint32_t>(block[offset + 1]) << 16U |
                       static_cast<uint32_t>(block[offset + 2]) << 8U |
                       static_cast<uint32_t>(block[offset + 3]);
    }
    for (size_t index = 16; index < words.size(); ++index) {
        const uint32_t s0 = std::rotr(words[index - 15], 7) ^ std::rotr(words[index - 15], 18) ^
                            (words[index - 15] >> 3U);
        const uint32_t s1 = std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^
                            (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (size_t index = 0; index < words.size(); ++index) {
        const uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t temporary1 = h + sum1 + choose + SHA256_CONSTANTS[index] + words[index];
        const uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

int hexadecimalDigit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

bool checkedProduct(size_t first, size_t second, size_t& product) {
    if (first != 0 && second > std::numeric_limits<size_t>::max() / first) return false;
    product = first * second;
    return true;
}

std::optional<int64_t> checkedInt64(__int128 value) {
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max()) {
        return std::nullopt;
    }
    return static_cast<int64_t>(value);
}

__int128 floorDivideWide(__int128 value, int64_t divisor) {
    __int128 quotient = value / divisor;
    const __int128 remainder = value % divisor;
    if (remainder < 0) --quotient;
    return quotient;
}

__int128 ceilDivideWide(__int128 value, int64_t divisor) {
    __int128 quotient = value / divisor;
    const __int128 remainder = value % divisor;
    if (remainder > 0) ++quotient;
    return quotient;
}

struct DecodedPageHeader {
    AuthorityQuality quality = AuthorityQuality::FINAL;
    TerrainPageCoordinate coordinate;
    uint64_t seed = 0;
    Sha256Digest fingerprint{};
    uint32_t uncompressedBytes = 0;
    uint32_t compressedBytes = 0;
    uint32_t payloadChecksum = 0;
};

AuthorityResult<DecodedPageHeader> decodePageHeader(std::span<const uint8_t> bytes) {
    if (bytes.size() < AUTHORITY_HEADER_BYTES) {
        return AuthorityResult<DecodedPageHeader>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority header is truncated", true));
    }
    if (bytes[0] != 'R' || bytes[1] != 'Y' || bytes[2] != 'T' || bytes[3] != 'A') {
        return AuthorityResult<DecodedPageHeader>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority magic is invalid", true));
    }

    size_t offset = 4;
    uint16_t schema = 0;
    uint16_t headerBytes = 0;
    if (!readU16(bytes, offset, schema) || !readU16(bytes, offset, headerBytes) ||
        schema != TERRAIN_AUTHORITY_SCHEMA_VERSION || headerBytes != AUTHORITY_HEADER_BYTES) {
        return AuthorityResult<DecodedPageHeader>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority schema is invalid", true));
    }

    const uint8_t qualityByte = bytes[offset++];
    const uint8_t compression = bytes[offset++];
    uint16_t reserved = 0;
    if (!readU16(bytes, offset, reserved) || qualityByte > 1 || compression != LZ4_COMPRESSION ||
        reserved != 0) {
        return AuthorityResult<DecodedPageHeader>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority encoding is invalid", true));
    }

    DecodedPageHeader result;
    result.quality = static_cast<AuthorityQuality>(qualityByte);
    uint64_t row = 0;
    uint64_t column = 0;
    if (!readU64(bytes, offset, row) || !readU64(bytes, offset, column) ||
        !readU64(bytes, offset, result.seed)) {
        return AuthorityResult<DecodedPageHeader>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority coordinates are truncated", true));
    }
    result.coordinate.row = static_cast<int64_t>(row);
    result.coordinate.column = static_cast<int64_t>(column);
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), result.fingerprint.size(),
                result.fingerprint.begin());
    offset += result.fingerprint.size();

    uint16_t nativeEdge = 0;
    uint16_t channelCount = 0;
    uint32_t channelMask = 0;
    if (!readU16(bytes, offset, nativeEdge) || !readU16(bytes, offset, channelCount) ||
        !readU32(bytes, offset, channelMask) || !readU32(bytes, offset, result.uncompressedBytes) ||
        !readU32(bytes, offset, result.compressedBytes) ||
        !readU32(bytes, offset, result.payloadChecksum)) {
        return AuthorityResult<DecodedPageHeader>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority layout is truncated", true));
    }
    uint32_t headerChecksum = 0;
    if (!readU32(bytes, offset, headerChecksum) || offset != AUTHORITY_HEADER_BYTES ||
        nativeEdge != AUTHORITY_PAGE_NATIVE_EDGE || channelCount != TERRAIN_CHANNEL_COUNT ||
        channelMask != TERRAIN_CHANNEL_MASK || result.uncompressedBytes != TERRAIN_PAYLOAD_BYTES ||
        result.compressedBytes == 0 ||
        headerChecksum != crc32(bytes.first(AUTHORITY_HEADER_CHECKSUM_OFFSET))) {
        return AuthorityResult<DecodedPageHeader>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority header validation failed", true));
    }
    return AuthorityResult<DecodedPageHeader>::ready(result);
}

std::vector<uint8_t> encodePayload(const TerrainAuthorityPage& page) {
    std::vector<uint8_t> payload;
    payload.reserve(TERRAIN_PAYLOAD_BYTES);
    for (const QuantizedTerrainSample& sample : page.samples) {
        appendU16(payload, static_cast<uint16_t>(sample.elevationMeters));
        appendU16(payload, static_cast<uint16_t>(sample.meanTemperatureCentidegrees));
        appendU16(payload, sample.temperatureVariabilityCentidegrees);
        appendU16(payload, sample.annualPrecipitationMillimeters);
        appendU16(payload, sample.precipitationCoefficientBasisPoints);
        appendU16(payload, static_cast<uint16_t>(sample.lapseRateMicrodegreesPerMeter));
    }
    return payload;
}

AuthorityResult<TerrainAuthorityPage> decodePayload(TerrainPageKey key,
                                                    std::span<const uint8_t> payload) {
    if (payload.size() != TERRAIN_PAYLOAD_BYTES) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority payload has the wrong size", true));
    }
    TerrainAuthorityPage page;
    page.key = key;
    page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
    size_t offset = 0;
    for (QuantizedTerrainSample& sample : page.samples) {
        uint16_t elevation = 0;
        uint16_t temperature = 0;
        uint16_t lapseRate = 0;
        if (!readU16(payload, offset, elevation) || !readU16(payload, offset, temperature) ||
            !readU16(payload, offset, sample.temperatureVariabilityCentidegrees) ||
            !readU16(payload, offset, sample.annualPrecipitationMillimeters) ||
            !readU16(payload, offset, sample.precipitationCoefficientBasisPoints) ||
            !readU16(payload, offset, lapseRate)) {
            return AuthorityResult<TerrainAuthorityPage>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Terrain authority payload is truncated", true));
        }
        sample.elevationMeters = static_cast<int16_t>(elevation);
        sample.meanTemperatureCentidegrees = static_cast<int16_t>(temperature);
        sample.lapseRateMicrodegreesPerMeter = static_cast<int16_t>(lapseRate);
    }
    return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
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
                        "Terrain authority publication lock could not be opened", true));
    }
    while (::flock(descriptor, LOCK_EX) != 0) {
        if (errno == EINTR) continue;
        static_cast<void>(::close(descriptor));
        return AuthorityResult<int>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority publication lock could not be acquired", true));
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

template <typename Integer>
Integer clampedRound(double value) {
    const double minimum = static_cast<double>(std::numeric_limits<Integer>::min());
    const double maximum = static_cast<double>(std::numeric_limits<Integer>::max());
    return static_cast<Integer>(std::lround(std::clamp(value, minimum, maximum)));
}

} // namespace

Sha256Digest sha256(std::span<const uint8_t> bytes) {
    std::array<uint32_t, 8> state{
        0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
        0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U,
    };
    size_t offset = 0;
    while (bytes.size() - offset >= 64) {
        sha256Transform(state, bytes.data() + offset);
        offset += 64;
    }

    std::array<uint8_t, 128> tail{};
    const size_t remaining = bytes.size() - offset;
    if (remaining != 0) std::copy_n(bytes.data() + offset, remaining, tail.data());
    tail[remaining] = 0x80;
    const size_t paddedBytes = remaining < 56 ? 64 : 128;
    const uint64_t bitCount = static_cast<uint64_t>(bytes.size()) * 8U;
    for (size_t byte = 0; byte < 8; ++byte) {
        tail[paddedBytes - 1 - byte] = static_cast<uint8_t>(bitCount >> (byte * 8U));
    }
    sha256Transform(state, tail.data());
    if (paddedBytes == 128) sha256Transform(state, tail.data() + 64);

    Sha256Digest digest{};
    for (size_t word = 0; word < state.size(); ++word) {
        for (size_t byte = 0; byte < 4; ++byte) {
            digest[word * 4 + byte] = static_cast<uint8_t>(state[word] >> ((3U - byte) * 8U));
        }
    }
    return digest;
}

std::optional<Sha256Digest> parseSha256(std::string_view hexadecimal) {
    if (hexadecimal.size() != 64) return std::nullopt;
    Sha256Digest digest{};
    for (size_t index = 0; index < digest.size(); ++index) {
        const int high = hexadecimalDigit(hexadecimal[index * 2]);
        const int low = hexadecimalDigit(hexadecimal[index * 2 + 1]);
        if (high < 0 || low < 0) return std::nullopt;
        digest[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return digest;
}

std::string sha256Hex(const Sha256Digest& digest) {
    constexpr char DIGITS[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = DIGITS[digest[index] >> 4U];
        result[index * 2 + 1] = DIGITS[digest[index] & 0x0FU];
    }
    return result;
}

bool GenerationIdentity::valid() const noexcept {
    return generatorVersion == GENERATOR_V4_VERSION && hasNonzeroByte(modelPackHash) &&
           hasNonzeroByte(runtimeHash) && provider == GENERATOR_V4_PROVIDER_CONFIGURATION &&
           modelBlockScale == MODEL_BLOCK_SCALE && rngRevision == GENERATOR_V4_RNG_REVISION &&
           quantizationRevision == GENERATOR_V4_QUANTIZATION_REVISION &&
           hydrologyRevision == GENERATOR_V4_HYDROLOGY_REVISION &&
           postprocessingRevision == GENERATOR_V4_POSTPROCESSING_REVISION &&
           coarseWindow == COARSE_WINDOW && latentWindow == LATENT_WINDOW &&
           decoderWindow == DECODER_WINDOW;
}

Sha256Digest GenerationIdentity::fingerprint() const {
    std::vector<uint8_t> bytes;
    bytes.reserve(160);
    appendU32(bytes, generatorVersion);
    appendU64(bytes, seed);
    appendDigest(bytes, modelPackHash);
    appendDigest(bytes, runtimeHash);
    bytes.push_back(static_cast<uint8_t>(provider.provider));
    appendU16(bytes, provider.onnxRuntimeMajorVersion);
    appendU16(bytes, provider.onnxRuntimeMinorVersion);
    appendU16(bytes, provider.onnxRuntimePatchVersion);
    appendU32(bytes, provider.flags);
    appendU16(bytes, modelBlockScale);
    appendU16(bytes, rngRevision);
    appendU16(bytes, quantizationRevision);
    appendU16(bytes, hydrologyRevision);
    appendU16(bytes, postprocessingRevision);
    for (WindowGeometry geometry : {coarseWindow, latentWindow, decoderWindow}) {
        appendU16(bytes, geometry.edge);
        appendU16(bytes, geometry.stride);
        appendU16(bytes, geometry.inferenceSteps);
        appendU16(bytes, geometry.batchSize);
    }
    return sha256(bytes);
}

int64_t floorDivide(int64_t value, int64_t divisor) {
    if (divisor <= 0) throw std::invalid_argument("floorDivide requires a positive divisor");
    return world_coord::floorDiv(value, divisor);
}

NativePoint worldBlockToNative(int64_t worldX, int64_t worldZ) {
    return NativePoint{
        .row = world_coord::floorDiv(worldZ, static_cast<int64_t>(MODEL_BLOCK_SCALE)),
        .column = world_coord::floorDiv(worldX, static_cast<int64_t>(MODEL_BLOCK_SCALE)),
    };
}

TerrainPageCoordinate terrainPageCoordinateFor(NativePoint point) noexcept {
    return TerrainPageCoordinate{
        .row = world_coord::floorDiv(point.row, static_cast<int64_t>(AUTHORITY_PAGE_NATIVE_EDGE)),
        .column =
            world_coord::floorDiv(point.column, static_cast<int64_t>(AUTHORITY_PAGE_NATIVE_EDGE)),
    };
}

size_t terrainPageLocalCoordinate(int64_t coordinate) noexcept {
    return static_cast<size_t>(
        world_coord::floorMod(coordinate, static_cast<int32_t>(AUTHORITY_PAGE_NATIVE_EDGE)));
}

std::optional<NativeRect> terrainPageNativeRect(TerrainPageCoordinate coordinate) noexcept {
    const auto rowBegin =
        checkedInt64(static_cast<__int128>(coordinate.row) * AUTHORITY_PAGE_NATIVE_EDGE);
    const auto columnBegin =
        checkedInt64(static_cast<__int128>(coordinate.column) * AUTHORITY_PAGE_NATIVE_EDGE);
    const auto rowEnd = checkedInt64(static_cast<__int128>(coordinate.row + __int128{1}) *
                                     AUTHORITY_PAGE_NATIVE_EDGE);
    const auto columnEnd = checkedInt64(static_cast<__int128>(coordinate.column + __int128{1}) *
                                        AUTHORITY_PAGE_NATIVE_EDGE);
    if (!rowBegin || !columnBegin || !rowEnd || !columnEnd) return std::nullopt;
    return NativeRect{.rowBegin = *rowBegin,
                      .columnBegin = *columnBegin,
                      .rowEnd = *rowEnd,
                      .columnEnd = *columnEnd};
}

double learnedElevationMetersToWorldHeight(double elevationMeters) noexcept {
    if (!std::isfinite(elevationMeters)) return LEARNED_SEA_LEVEL;
    if (elevationMeters >= 0.0) {
        return std::trunc(elevationMeters / WORLD_METERS_PER_BLOCK) + LEARNED_SEA_LEVEL;
    }
    return std::trunc(-std::sqrt(std::abs(elevationMeters) + 10.0) + std::sqrt(10.0)) - 1.0 +
           LEARNED_SEA_LEVEL;
}

std::vector<WindowIndex> intersectingWindows(const NativeRect& region, WindowGeometry geometry) {
    if (!region.valid() || !validGeometry(geometry)) return {};
    const int64_t edge = geometry.edge;
    const int64_t stride = geometry.stride;
    const __int128 firstRowWide =
        floorDivideWide(static_cast<__int128>(region.rowBegin) - edge, stride) + 1;
    const __int128 firstColumnWide =
        floorDivideWide(static_cast<__int128>(region.columnBegin) - edge, stride) + 1;
    const __int128 rowEndWide = ceilDivideWide(region.rowEnd, stride);
    const __int128 columnEndWide = ceilDivideWide(region.columnEnd, stride);
    const auto firstRowValue = checkedInt64(firstRowWide);
    const auto firstColumnValue = checkedInt64(firstColumnWide);
    const auto rowEndValue = checkedInt64(rowEndWide);
    const auto columnEndValue = checkedInt64(columnEndWide);
    if (!firstRowValue || !firstColumnValue || !rowEndValue || !columnEndValue) return {};
    const int64_t firstRow = *firstRowValue;
    const int64_t firstColumn = *firstColumnValue;
    const int64_t rowEnd = *rowEndValue;
    const int64_t columnEnd = *columnEndValue;
    if (firstRow >= rowEnd || firstColumn >= columnEnd) return {};

    const __int128 rowCount = static_cast<__int128>(rowEnd) - firstRow;
    const __int128 columnCount = static_cast<__int128>(columnEnd) - firstColumn;
    const __int128 count = rowCount * columnCount;
    if (count > static_cast<__int128>(std::numeric_limits<size_t>::max())) return {};

    std::vector<WindowIndex> result;
    result.reserve(static_cast<size_t>(count));
    for (int64_t row = firstRow; row < rowEnd; ++row) {
        for (int64_t column = firstColumn; column < columnEnd; ++column) {
            result.push_back({.row = row, .column = column});
            if (column == std::numeric_limits<int64_t>::max()) break;
        }
        if (row == std::numeric_limits<int64_t>::max()) break;
    }
    return result;
}

float linearWindowWeight(size_t offset, size_t edge) {
    if (edge == 0 || offset >= edge) return 0.0F;
    if (edge == 1) return 1.0F;
    const double midpoint = static_cast<double>(edge - 1) * 0.5;
    const double distance = std::abs(static_cast<double>(offset) - midpoint) / midpoint;
    return static_cast<float>(1.0 - (1.0 - WINDOW_WEIGHT_EPSILON) * std::clamp(distance, 0.0, 1.0));
}

class WeightedWindowAccumulator::Impl {
public:
    struct Prediction {
        std::vector<float> values;
    };

    NativeRect target;
    WindowGeometry geometry;
    size_t channels = 0;
    bool valid = false;
    std::map<WindowIndex, Prediction> predictions;
};

WeightedWindowAccumulator::WeightedWindowAccumulator(NativeRect target, WindowGeometry geometry,
                                                     size_t channels)
    : impl_(std::make_unique<Impl>()) {
    impl_->target = target;
    impl_->geometry = geometry;
    impl_->channels = channels;
    impl_->valid = target.valid() && validGeometry(geometry) && channels != 0;
}

WeightedWindowAccumulator::~WeightedWindowAccumulator() = default;
WeightedWindowAccumulator::WeightedWindowAccumulator(WeightedWindowAccumulator&&) noexcept =
    default;
WeightedWindowAccumulator&
WeightedWindowAccumulator::operator=(WeightedWindowAccumulator&&) noexcept = default;

bool WeightedWindowAccumulator::addWindow(WindowIndex index,
                                          std::span<const float> channelMajorValues) {
    if (!impl_->valid) return false;
    size_t pixels = 0;
    size_t expected = 0;
    if (!checkedProduct(impl_->geometry.edge, impl_->geometry.edge, pixels) ||
        !checkedProduct(pixels, impl_->channels, expected) ||
        channelMajorValues.size() != expected) {
        return false;
    }
    const auto [unused, inserted] = impl_->predictions.emplace(
        index, Impl::Prediction{.values = std::vector<float>(channelMajorValues.begin(),
                                                             channelMajorValues.end())});
    static_cast<void>(unused);
    return inserted;
}

AuthorityResult<std::vector<float>> WeightedWindowAccumulator::resolve() const {
    if (!impl_->valid || impl_->predictions.empty()) {
        return AuthorityResult<std::vector<float>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Weighted accumulation has no valid predictions", false));
    }
    const uint64_t height = impl_->target.height();
    const uint64_t width = impl_->target.width();
    size_t pixels = 0;
    size_t valueCount = 0;
    if (height > std::numeric_limits<size_t>::max() || width > std::numeric_limits<size_t>::max() ||
        !checkedProduct(static_cast<size_t>(height), static_cast<size_t>(width), pixels) ||
        !checkedProduct(pixels, impl_->channels, valueCount)) {
        return AuthorityResult<std::vector<float>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Weighted accumulation region is too large", false));
    }

    std::vector<double> numerator(valueCount, 0.0);
    std::vector<double> denominator(pixels, 0.0);
    const int64_t edge = impl_->geometry.edge;
    for (const auto& [index, prediction] : impl_->predictions) {
        const auto originRow =
            checkedInt64(static_cast<__int128>(index.row) * impl_->geometry.stride);
        const auto originColumn =
            checkedInt64(static_cast<__int128>(index.column) * impl_->geometry.stride);
        if (!originRow || !originColumn) {
            return AuthorityResult<std::vector<float>>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "Weighted accumulation window coordinate overflowed", false));
        }
        const auto windowRowEnd = checkedInt64(static_cast<__int128>(*originRow) + edge);
        const auto windowColumnEnd = checkedInt64(static_cast<__int128>(*originColumn) + edge);
        if (!windowRowEnd || !windowColumnEnd) {
            return AuthorityResult<std::vector<float>>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "Weighted accumulation window extent overflowed", false));
        }
        const int64_t rowBegin = std::max(impl_->target.rowBegin, *originRow);
        const int64_t rowEnd = std::min(impl_->target.rowEnd, *windowRowEnd);
        const int64_t columnBegin = std::max(impl_->target.columnBegin, *originColumn);
        const int64_t columnEnd = std::min(impl_->target.columnEnd, *windowColumnEnd);
        if (rowBegin >= rowEnd || columnBegin >= columnEnd) continue;

        for (int64_t row = rowBegin; row < rowEnd; ++row) {
            const size_t targetRow = static_cast<size_t>(row - impl_->target.rowBegin);
            const size_t windowRow = static_cast<size_t>(row - *originRow);
            const double rowWeight = linearWindowWeight(windowRow, impl_->geometry.edge);
            for (int64_t column = columnBegin; column < columnEnd; ++column) {
                const size_t targetColumn = static_cast<size_t>(column - impl_->target.columnBegin);
                const size_t windowColumn = static_cast<size_t>(column - *originColumn);
                const double weight =
                    rowWeight * linearWindowWeight(windowColumn, impl_->geometry.edge);
                const size_t targetPixel = targetRow * static_cast<size_t>(width) + targetColumn;
                const size_t windowPixel = windowRow * impl_->geometry.edge + windowColumn;
                denominator[targetPixel] += weight;
                for (size_t channel = 0; channel < impl_->channels; ++channel) {
                    const size_t targetIndex = channel * pixels + targetPixel;
                    const size_t windowIndex =
                        channel * impl_->geometry.edge * impl_->geometry.edge + windowPixel;
                    numerator[targetIndex] += weight * prediction.values[windowIndex];
                }
            }
        }
    }

    if (std::any_of(denominator.begin(), denominator.end(),
                    [](double weight) { return weight <= 0.0; })) {
        return AuthorityResult<std::vector<float>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Weighted predictions do not cover the requested region", false));
    }
    std::vector<float> output(valueCount);
    for (size_t channel = 0; channel < impl_->channels; ++channel) {
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            output[channel * pixels + pixel] =
                static_cast<float>(numerator[channel * pixels + pixel] / denominator[pixel]);
        }
    }
    return AuthorityResult<std::vector<float>>::ready(std::move(output));
}

size_t WeightedWindowAccumulator::windowCount() const noexcept {
    return impl_->predictions.size();
}

uint32_t PortablePcg64::next32() noexcept {
    state_ = state_ * PCG_MULTIPLIER + PCG_INCREMENT;
    const uint32_t xorshifted = static_cast<uint32_t>(((state_ >> 18U) ^ state_) >> 27U);
    const uint32_t rotation = static_cast<uint32_t>(state_ >> 59U);
    return std::rotr(xorshifted, static_cast<int>(rotation));
}

uint64_t terrainTileSeed(uint64_t baseSeed, int64_t tileRow, int64_t tileColumn) noexcept {
    uint64_t hash = baseSeed * 0x9E3779B9ULL;
    hash += static_cast<uint32_t>(tileRow);
    hash = hash * 0x9E3779B9ULL + static_cast<uint32_t>(tileColumn);
    return hash;
}

void fillStandardNormal(uint64_t seed, std::span<float> output) {
    PortablePcg64 generator(seed);
    size_t index = 0;
    while (index < output.size()) {
        const double first =
            2.0 * (static_cast<double>(generator.next32()) + 1.0) * INVERSE_TWO_TO_32 - 1.0;
        const double second =
            2.0 * (static_cast<double>(generator.next32()) + 1.0) * INVERSE_TWO_TO_32 - 1.0;
        const double radiusSquared = first * first + second * second;
        if (radiusSquared <= 0.0 || radiusSquared >= 1.0) continue;
        const double factor = std::sqrt(-2.0 * std::log(radiusSquared) / radiusSquared);
        output[index++] = static_cast<float>(first * factor);
        if (index < output.size()) output[index++] = static_cast<float>(second * factor);
    }
}

AuthorityResult<std::vector<float>> gaussianNoisePatch(uint64_t baseSeed, NativeRect region,
                                                       size_t channels, int64_t tileEdge) {
    if (!region.valid() || channels == 0 || tileEdge <= 0 || tileEdge > 4'096) {
        return AuthorityResult<std::vector<float>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Gaussian noise patch dimensions are invalid", false));
    }
    const uint64_t height = region.height();
    const uint64_t width = region.width();
    size_t pixels = 0;
    size_t outputCount = 0;
    size_t tilePixels = 0;
    size_t tileCount = 0;
    if (height > std::numeric_limits<size_t>::max() || width > std::numeric_limits<size_t>::max() ||
        !checkedProduct(static_cast<size_t>(height), static_cast<size_t>(width), pixels) ||
        !checkedProduct(pixels, channels, outputCount) ||
        !checkedProduct(static_cast<size_t>(tileEdge), static_cast<size_t>(tileEdge), tilePixels) ||
        !checkedProduct(tilePixels, channels, tileCount)) {
        return AuthorityResult<std::vector<float>>::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST, "Gaussian noise patch is too large", false));
    }

    std::vector<float> output(outputCount);
    const int64_t tileRowBegin = world_coord::floorDiv(region.rowBegin, tileEdge);
    const int64_t tileRowEnd = world_coord::floorDiv(region.rowEnd - 1, tileEdge);
    const int64_t tileColumnBegin = world_coord::floorDiv(region.columnBegin, tileEdge);
    const int64_t tileColumnEnd = world_coord::floorDiv(region.columnEnd - 1, tileEdge);
    for (int64_t tileRow = tileRowBegin; tileRow <= tileRowEnd; ++tileRow) {
        const auto tileOriginRow = checkedInt64(static_cast<__int128>(tileRow) * tileEdge);
        if (!tileOriginRow) {
            return AuthorityResult<std::vector<float>>::failed(makeFailure(
                GenerationFailureCode::INVALID_REQUEST, "Gaussian tile row overflowed", false));
        }
        for (int64_t tileColumn = tileColumnBegin; tileColumn <= tileColumnEnd; ++tileColumn) {
            const auto tileOriginColumn =
                checkedInt64(static_cast<__int128>(tileColumn) * tileEdge);
            if (!tileOriginColumn) {
                return AuthorityResult<std::vector<float>>::failed(
                    makeFailure(GenerationFailureCode::INVALID_REQUEST,
                                "Gaussian tile column overflowed", false));
            }
            const auto tileEndRow = checkedInt64(static_cast<__int128>(*tileOriginRow) + tileEdge);
            const auto tileEndColumn =
                checkedInt64(static_cast<__int128>(*tileOriginColumn) + tileEdge);
            if (!tileEndRow || !tileEndColumn) {
                return AuthorityResult<std::vector<float>>::failed(
                    makeFailure(GenerationFailureCode::INVALID_REQUEST,
                                "Gaussian tile extent overflowed", false));
            }
            std::vector<float> tile(tileCount);
            fillStandardNormal(terrainTileSeed(baseSeed, tileRow, tileColumn), tile);
            const int64_t rowBegin = std::max(region.rowBegin, *tileOriginRow);
            const int64_t rowEnd = std::min(region.rowEnd, *tileEndRow);
            const int64_t columnBegin = std::max(region.columnBegin, *tileOriginColumn);
            const int64_t columnEnd = std::min(region.columnEnd, *tileEndColumn);
            for (size_t channel = 0; channel < channels; ++channel) {
                for (int64_t row = rowBegin; row < rowEnd; ++row) {
                    const size_t outputRow = static_cast<size_t>(row - region.rowBegin);
                    const size_t tileLocalRow = static_cast<size_t>(row - *tileOriginRow);
                    for (int64_t column = columnBegin; column < columnEnd; ++column) {
                        const size_t outputColumn =
                            static_cast<size_t>(column - region.columnBegin);
                        const size_t tileLocalColumn =
                            static_cast<size_t>(column - *tileOriginColumn);
                        output[channel * pixels + outputRow * static_cast<size_t>(width) +
                               outputColumn] =
                            tile[channel * tilePixels +
                                 tileLocalRow * static_cast<size_t>(tileEdge) + tileLocalColumn];
                    }
                }
            }
            if (tileColumn == std::numeric_limits<int64_t>::max()) break;
        }
        if (tileRow == std::numeric_limits<int64_t>::max()) break;
    }
    return AuthorityResult<std::vector<float>>::ready(std::move(output));
}

const QuantizedTerrainSample* TerrainAuthorityPage::sample(size_t row,
                                                           size_t column) const noexcept {
    if (row >= AUTHORITY_PAGE_NATIVE_EDGE || column >= AUTHORITY_PAGE_NATIVE_EDGE || !valid()) {
        return nullptr;
    }
    return std::addressof(samples[row * AUTHORITY_PAGE_NATIVE_EDGE + column]);
}

QuantizedTerrainSample quantizePhysicalTerrainSample(const PhysicalTerrainSample& sample) noexcept {
    return QuantizedTerrainSample{
        .elevationMeters = clampedRound<int16_t>(sample.elevationMeters),
        .meanTemperatureCentidegrees = clampedRound<int16_t>(sample.meanTemperatureC * 100.0),
        .temperatureVariabilityCentidegrees =
            clampedRound<uint16_t>(sample.temperatureVariabilityC * 100.0),
        .annualPrecipitationMillimeters = clampedRound<uint16_t>(sample.annualPrecipitationMm),
        .precipitationCoefficientBasisPoints =
            clampedRound<uint16_t>(sample.precipitationCoefficientOfVariation * 10'000.0),
        .lapseRateMicrodegreesPerMeter =
            clampedRound<int16_t>(sample.lapseRateCPerMeter * 1'000'000.0),
    };
}

PhysicalTerrainSample dequantizeTerrainSample(const QuantizedTerrainSample& sample) noexcept {
    return PhysicalTerrainSample{
        .elevationMeters = static_cast<double>(sample.elevationMeters),
        .meanTemperatureC = static_cast<double>(sample.meanTemperatureCentidegrees) / 100.0,
        .temperatureVariabilityC =
            static_cast<double>(sample.temperatureVariabilityCentidegrees) / 100.0,
        .annualPrecipitationMm = static_cast<double>(sample.annualPrecipitationMillimeters),
        .precipitationCoefficientOfVariation =
            static_cast<double>(sample.precipitationCoefficientBasisPoints) / 10'000.0,
        .lapseRateCPerMeter =
            static_cast<double>(sample.lapseRateMicrodegreesPerMeter) / 1'000'000.0,
    };
}

bool PhysicalTerrainGrid::valid() const noexcept {
    if (!region.valid() || region.height() > std::numeric_limits<size_t>::max() ||
        region.width() > std::numeric_limits<size_t>::max()) {
        return false;
    }
    size_t expected = 0;
    return checkedProduct(static_cast<size_t>(region.height()), static_cast<size_t>(region.width()),
                          expected) &&
           samples.size() == expected;
}

const PhysicalTerrainSample* PhysicalTerrainGrid::sample(int64_t row,
                                                         int64_t column) const noexcept {
    if (!valid() || row < region.rowBegin || row >= region.rowEnd || column < region.columnBegin ||
        column >= region.columnEnd) {
        return nullptr;
    }
    const size_t localRow =
        static_cast<size_t>(static_cast<uint64_t>(row) - static_cast<uint64_t>(region.rowBegin));
    const size_t localColumn = static_cast<size_t>(static_cast<uint64_t>(column) -
                                                   static_cast<uint64_t>(region.columnBegin));
    return std::addressof(samples[localRow * static_cast<size_t>(region.width()) + localColumn]);
}

bool CoarseSpawnGrid::valid() const noexcept {
    if (!region.valid() || region.height() > MAXIMUM_COARSE_SPAWN_GRID_EDGE ||
        region.width() > MAXIMUM_COARSE_SPAWN_GRID_EDGE) {
        return false;
    }
    size_t expected = 0;
    return checkedProduct(static_cast<size_t>(region.height()), static_cast<size_t>(region.width()),
                          expected) &&
           elevationMeters.size() == expected;
}

const float* CoarseSpawnGrid::sample(int64_t row, int64_t column) const noexcept {
    if (!valid() || row < region.rowBegin || row >= region.rowEnd || column < region.columnBegin ||
        column >= region.columnEnd) {
        return nullptr;
    }
    const size_t localRow =
        static_cast<size_t>(static_cast<uint64_t>(row) - static_cast<uint64_t>(region.rowBegin));
    const size_t localColumn = static_cast<size_t>(static_cast<uint64_t>(column) -
                                                   static_cast<uint64_t>(region.columnBegin));
    return std::addressof(
        elevationMeters[localRow * static_cast<size_t>(region.width()) + localColumn]);
}

AuthorityResult<TerrainAuthorityPage>
TerrainInferenceBackend::inferPageForRequest(const GenerationIdentity& identity, TerrainPageKey key,
                                             AuthorityRequestPriority) {
    return inferPage(identity, key);
}

AuthorityResult<CoarseSpawnGrid>
TerrainInferenceBackend::inferCoarseSpawnGrid(const GenerationIdentity&, CoarseSpawnRegion) {
    return AuthorityResult<CoarseSpawnGrid>::failed(
        makeFailure(GenerationFailureCode::BACKEND_UNAVAILABLE,
                    "Terrain inference backend does not provide coarse spawn selection", false));
}

AuthorityResult<CoarseSpawnGrid> TerrainInferenceBackend::inferCoarseSpawnGridForRequest(
    const GenerationIdentity& identity, CoarseSpawnRegion region, AuthorityRequestPriority) {
    return inferCoarseSpawnGrid(identity, region);
}

AuthorityResult<PhysicalTerrainGrid>
TerrainInferenceBackend::inferFinalNativeGrid(const GenerationIdentity&, NativeRect) {
    return AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
        GenerationFailureCode::INVALID_REQUEST,
        "Terrain inference backend does not provide transient final rectangles", false));
}

AuthorityResult<PhysicalTerrainGrid> TerrainInferenceBackend::inferFinalNativeGridForRequest(
    const GenerationIdentity& identity, NativeRect region, AuthorityRequestPriority) {
    return inferFinalNativeGrid(identity, region);
}

AuthorityResult<std::vector<TerrainAuthorityPage>>
TerrainInferenceBackend::inferPages(const GenerationIdentity& identity,
                                    std::span<const TerrainPageKey> keys) {
    if (keys.empty() || keys.size() > MAXIMUM_FINAL_AUTHORITY_BATCH_PAGES) {
        return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain inference batch has an invalid page count", false));
    }
    std::vector<TerrainAuthorityPage> pages;
    pages.reserve(keys.size());
    for (const TerrainPageKey key : keys) {
        AuthorityResult<TerrainAuthorityPage> inferred = inferPage(identity, key);
        if (!inferred.isReady()) {
            return inferred.status() == AuthorityStatus::DEFERRED
                       ? AuthorityResult<std::vector<TerrainAuthorityPage>>::deferred(
                             *inferred.failure())
                       : AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                             *inferred.failure());
        }
        pages.push_back(std::move(*inferred.value()));
    }
    return AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(pages));
}

AuthorityResult<std::vector<TerrainAuthorityPage>>
TerrainInferenceBackend::inferPagesForRequest(const GenerationIdentity& identity,
                                              std::span<const TerrainPageKey> keys,
                                              AuthorityRequestPriority) {
    return inferPages(identity, keys);
}

AuthorityResult<bool> TerrainAuthority::preparePages(std::span<const TerrainPageKey> keys,
                                                     AuthorityRequestPriority priority) {
    if (keys.empty()) {
        return AuthorityResult<bool>::failed(makeFailure(GenerationFailureCode::INVALID_REQUEST,
                                                         "Terrain authority page closure is empty",
                                                         false));
    }

    std::optional<GenerationFailure> deferred;
    for (const TerrainPageKey key : keys) {
        const AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>> prepared =
            preparePage(key, priority);
        if (prepared.isReady()) continue;
        const GenerationFailure failure =
            prepared.failure()
                ? *prepared.failure()
                : makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                              "Terrain authority page closure returned no failure", true);
        if (prepared.status() == AuthorityStatus::FAILED)
            return AuthorityResult<bool>::failed(failure);

        // The compatibility fallback cannot roll back a custom authority's
        // individual admission. Preserve the previous bounded-queue result
        // immediately, while still allowing ordinary deferred pages to admit
        // the remaining closure members in lexical caller order.
        if (failure.code == GenerationFailureCode::QUEUE_FULL)
            return AuthorityResult<bool>::deferred(failure);
        if (!deferred) deferred = failure;
    }
    if (deferred) return AuthorityResult<bool>::deferred(std::move(*deferred));
    return AuthorityResult<bool>::ready(true);
}

class DeterministicFakeTerrainBackend::Impl {
public:
    explicit Impl(std::chrono::milliseconds requestedLatency) : latency(requestedLatency) {}

    std::chrono::milliseconds latency;
    std::atomic<uint64_t> calls{0};
};

DeterministicFakeTerrainBackend::DeterministicFakeTerrainBackend(std::chrono::milliseconds latency)
    : impl_(std::make_unique<Impl>(latency)) {}

DeterministicFakeTerrainBackend::~DeterministicFakeTerrainBackend() = default;

AuthorityResult<TerrainAuthorityPage>
DeterministicFakeTerrainBackend::inferPage(const GenerationIdentity& identity, TerrainPageKey key) {
    impl_->calls.fetch_add(1, std::memory_order_relaxed);
    if (!identity.valid() || !validQuality(key.quality)) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Fake terrain inference received an invalid identity or key", false));
    }
    if (impl_->latency.count() > 0) std::this_thread::sleep_for(impl_->latency);

    const std::optional<NativeRect> region = terrainPageNativeRect(key.coordinate);
    if (!region) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Fake terrain page coordinate overflowed", false));
    }
    auto noise = gaussianNoisePatch(identity.seed, *region, TERRAIN_CHANNEL_COUNT);
    if (!noise.isReady()) {
        return AuthorityResult<TerrainAuthorityPage>::failed(*noise.failure());
    }

    TerrainAuthorityPage page;
    page.key = key;
    page.generationSeed = identity.seed;
    page.generationFingerprint = identity.fingerprint();
    page.samples.resize(AUTHORITY_PAGE_SAMPLE_COUNT);
    const std::vector<float>& fields = *noise.value();
    for (size_t index = 0; index < page.samples.size(); ++index) {
        QuantizedTerrainSample& sample = page.samples[index];
        sample.elevationMeters = clampedRound<int16_t>(fields[index] * 900.0);
        sample.meanTemperatureCentidegrees = clampedRound<int16_t>(
            (12.0 + fields[AUTHORITY_PAGE_SAMPLE_COUNT + index] * 10.0) * 100.0);
        sample.temperatureVariabilityCentidegrees = clampedRound<uint16_t>(
            std::abs(fields[2 * AUTHORITY_PAGE_SAMPLE_COUNT + index]) * 800.0 + 200.0);
        sample.annualPrecipitationMillimeters = clampedRound<uint16_t>(
            std::max(0.0, 900.0 + fields[3 * AUTHORITY_PAGE_SAMPLE_COUNT + index] * 300.0));
        sample.precipitationCoefficientBasisPoints = clampedRound<uint16_t>(
            std::abs(fields[4 * AUTHORITY_PAGE_SAMPLE_COUNT + index]) * 1'500.0 + 2'000.0);
        sample.lapseRateMicrodegreesPerMeter = clampedRound<int16_t>(
            -6'500.0 + fields[5 * AUTHORITY_PAGE_SAMPLE_COUNT + index] * 250.0);
    }
    return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
}

AuthorityResult<PhysicalTerrainGrid>
DeterministicFakeTerrainBackend::inferFinalNativeGrid(const GenerationIdentity& identity,
                                                      NativeRect region) {
    impl_->calls.fetch_add(1, std::memory_order_relaxed);
    if (!identity.valid() || !region.valid() ||
        region.height() > std::numeric_limits<size_t>::max() ||
        region.width() > std::numeric_limits<size_t>::max() ||
        (region.height() != 0 &&
         region.width() > std::numeric_limits<size_t>::max() / region.height())) {
        return AuthorityResult<PhysicalTerrainGrid>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Fake transient terrain inference received an invalid rectangle", false));
    }
    const size_t sampleCount =
        static_cast<size_t>(region.height()) * static_cast<size_t>(region.width());
    if (sampleCount > MAXIMUM_AUTHORITY_QUERY_SAMPLES) {
        return AuthorityResult<PhysicalTerrainGrid>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Fake transient terrain inference exceeded its sample bound", false));
    }
    if (impl_->latency.count() > 0) std::this_thread::sleep_for(impl_->latency);

    auto noise = gaussianNoisePatch(identity.seed, region, TERRAIN_CHANNEL_COUNT);
    if (!noise.isReady()) return AuthorityResult<PhysicalTerrainGrid>::failed(*noise.failure());

    PhysicalTerrainGrid grid{.region = region,
                             .samples = std::vector<PhysicalTerrainSample>(sampleCount)};
    const std::vector<float>& fields = *noise.value();
    for (size_t index = 0; index < sampleCount; ++index) {
        QuantizedTerrainSample quantized;
        quantized.elevationMeters = clampedRound<int16_t>(fields[index] * 900.0);
        quantized.meanTemperatureCentidegrees =
            clampedRound<int16_t>((12.0 + fields[sampleCount + index] * 10.0) * 100.0);
        quantized.temperatureVariabilityCentidegrees =
            clampedRound<uint16_t>(std::abs(fields[2 * sampleCount + index]) * 800.0 + 200.0);
        quantized.annualPrecipitationMillimeters =
            clampedRound<uint16_t>(std::max(0.0, 900.0 + fields[3 * sampleCount + index] * 300.0));
        quantized.precipitationCoefficientBasisPoints =
            clampedRound<uint16_t>(std::abs(fields[4 * sampleCount + index]) * 1'500.0 + 2'000.0);
        quantized.lapseRateMicrodegreesPerMeter =
            clampedRound<int16_t>(-6'500.0 + fields[5 * sampleCount + index] * 250.0);
        grid.samples[index] = dequantizeTerrainSample(quantized);
    }
    return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(grid));
}

AuthorityResult<CoarseSpawnGrid>
DeterministicFakeTerrainBackend::inferCoarseSpawnGrid(const GenerationIdentity& identity,
                                                      CoarseSpawnRegion region) {
    impl_->calls.fetch_add(1, std::memory_order_relaxed);
    if (!identity.valid() || !region.valid() || region.height() > MAXIMUM_COARSE_SPAWN_GRID_EDGE ||
        region.width() > MAXIMUM_COARSE_SPAWN_GRID_EDGE) {
        return AuthorityResult<CoarseSpawnGrid>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Fake coarse spawn inference received an invalid request", false));
    }
    if (impl_->latency.count() > 0) std::this_thread::sleep_for(impl_->latency);

    NativeRect noiseRegion{
        .rowBegin = region.rowBegin,
        .columnBegin = region.columnBegin,
        .rowEnd = region.rowEnd,
        .columnEnd = region.columnEnd,
    };
    const auto noise = gaussianNoisePatch(identity.seed, noiseRegion, 1);
    if (!noise.isReady()) {
        return AuthorityResult<CoarseSpawnGrid>::failed(*noise.failure());
    }
    CoarseSpawnGrid result{
        .region = region,
        .elevationMeters = std::move(*noise.value()),
    };
    for (float& elevation : result.elevationMeters)
        elevation *= 900.0F;
    return AuthorityResult<CoarseSpawnGrid>::ready(std::move(result));
}

uint64_t DeterministicFakeTerrainBackend::callCount() const noexcept {
    return impl_->calls.load(std::memory_order_relaxed);
}

TerrainPageStore::TerrainPageStore(std::filesystem::path root, GenerationIdentity identity,
                                   std::shared_ptr<const TestHooks> testHooks)
    : root_(std::move(root))
    , identity_(std::move(identity))
    , testHooks_(std::move(testHooks)) {}

std::filesystem::path TerrainPageStore::pagePath(TerrainPageKey key) const {
    const char* quality = key.quality == AuthorityQuality::PREVIEW ? "preview" : "final";
    return root_ / quality /
           ("p." + std::to_string(key.coordinate.row) + "." +
            std::to_string(key.coordinate.column) + ".ryta");
}

AuthorityResult<TerrainAuthorityPage> TerrainPageStore::loadPage(TerrainPageKey key) const {
    if (!identity_.valid() || !validQuality(key.quality)) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority load received an invalid identity or key", false));
    }
    const std::filesystem::path path = pagePath(key);
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority page could not be inspected", true));
    }
    if (!exists) {
        return AuthorityResult<TerrainAuthorityPage>::deferred(
            makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                        "Terrain authority page is not persisted", true));
    }

    const uintmax_t fileSize = std::filesystem::file_size(path, error);
    const int maximumCompressed = LZ4_compressBound(static_cast<int>(TERRAIN_PAYLOAD_BYTES));
    if (error || fileSize < AUTHORITY_HEADER_BYTES || maximumCompressed <= 0 ||
        fileSize > AUTHORITY_HEADER_BYTES + static_cast<uintmax_t>(maximumCompressed)) {
        return AuthorityResult<TerrainAuthorityPage>::failed(makeFailure(
            GenerationFailureCode::CORRUPT_PAGE, "Terrain authority file size is invalid", true));
    }
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return AuthorityResult<TerrainAuthorityPage>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Terrain authority page could not be opened", true));
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file.good()) {
        return AuthorityResult<TerrainAuthorityPage>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Terrain authority page could not be read", true));
    }

    auto decoded = decodePageHeader(bytes);
    if (!decoded.isReady())
        return AuthorityResult<TerrainAuthorityPage>::failed(*decoded.failure());
    const DecodedPageHeader& header = *decoded.value();
    if (header.quality != key.quality || header.coordinate != key.coordinate) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority page key does not match its path", true));
    }
    if (header.seed != identity_.seed || header.fingerprint != identity_.fingerprint()) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                        "Terrain authority page belongs to another generation identity", false));
    }
    if (bytes.size() != AUTHORITY_HEADER_BYTES + header.compressedBytes) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority compressed payload is truncated", true));
    }

    std::vector<uint8_t> payload(header.uncompressedBytes);
    const int written = LZ4_decompress_safe(
        reinterpret_cast<const char*>(bytes.data() + AUTHORITY_HEADER_BYTES),
        reinterpret_cast<char*>(payload.data()), static_cast<int>(header.compressedBytes),
        static_cast<int>(payload.size()));
    if (written != static_cast<int>(payload.size()) || crc32(payload) != header.payloadChecksum) {
        return AuthorityResult<TerrainAuthorityPage>::failed(
            makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                        "Terrain authority payload validation failed", true));
    }
    auto page = decodePayload(key, payload);
    if (page.isReady()) {
        page.value()->generationSeed = header.seed;
        page.value()->generationFingerprint = header.fingerprint;
    }
    return page;
}

AuthorityResult<bool> TerrainPageStore::writePage(const TerrainAuthorityPage& page) const {
    if (!identity_.valid() || !validQuality(page.key.quality) || !page.valid() ||
        !page.matches(identity_)) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST, "Terrain authority page is invalid", false));
    }
    const auto matchesExisting = [&](const TerrainAuthorityPage& existing) {
        if (existing.samples == page.samples) return AuthorityResult<bool>::ready(true);
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                        "Terrain authority page is immutable for its generation identity", false));
    };

    auto existing = loadPage(page.key);
    if (existing.isReady()) {
        return matchesExisting(*existing.value());
    }
    if (existing.status() == AuthorityStatus::FAILED &&
        (!existing.failure() || existing.failure()->code != GenerationFailureCode::CORRUPT_PAGE)) {
        return AuthorityResult<bool>::failed(
            existing.failure()
                ? *existing.failure()
                : makeFailure(GenerationFailureCode::IO_ERROR,
                              "Terrain authority load failed without a failure reason", true));
    }
    const std::vector<uint8_t> payload = encodePayload(page);
    const int maximum = LZ4_compressBound(static_cast<int>(payload.size()));
    if (maximum <= 0) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Terrain authority compression bound failed", true));
    }
    std::vector<uint8_t> compressed(static_cast<size_t>(maximum));
    const int compressedBytes = LZ4_compress_default(reinterpret_cast<const char*>(payload.data()),
                                                     reinterpret_cast<char*>(compressed.data()),
                                                     static_cast<int>(payload.size()), maximum);
    if (compressedBytes <= 0) {
        return AuthorityResult<bool>::failed(makeFailure(
            GenerationFailureCode::IO_ERROR, "Terrain authority compression failed", true));
    }
    compressed.resize(static_cast<size_t>(compressedBytes));

    std::vector<uint8_t> fileBytes;
    fileBytes.reserve(AUTHORITY_HEADER_BYTES + compressed.size());
    fileBytes.insert(fileBytes.end(), {'R', 'Y', 'T', 'A'});
    appendU16(fileBytes, TERRAIN_AUTHORITY_SCHEMA_VERSION);
    appendU16(fileBytes, static_cast<uint16_t>(AUTHORITY_HEADER_BYTES));
    fileBytes.push_back(static_cast<uint8_t>(page.key.quality));
    fileBytes.push_back(LZ4_COMPRESSION);
    appendU16(fileBytes, 0);
    appendU64(fileBytes, static_cast<uint64_t>(page.key.coordinate.row));
    appendU64(fileBytes, static_cast<uint64_t>(page.key.coordinate.column));
    appendU64(fileBytes, identity_.seed);
    appendDigest(fileBytes, identity_.fingerprint());
    appendU16(fileBytes, AUTHORITY_PAGE_NATIVE_EDGE);
    appendU16(fileBytes, TERRAIN_CHANNEL_COUNT);
    appendU32(fileBytes, TERRAIN_CHANNEL_MASK);
    appendU32(fileBytes, static_cast<uint32_t>(payload.size()));
    appendU32(fileBytes, static_cast<uint32_t>(compressed.size()));
    appendU32(fileBytes, crc32(payload));
    appendU32(fileBytes, crc32(fileBytes));
    if (fileBytes.size() != AUTHORITY_HEADER_BYTES) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority header construction failed", false));
    }
    fileBytes.insert(fileBytes.end(), compressed.begin(), compressed.end());

    const std::filesystem::path path = pagePath(page.key);
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority directory could not be created", true));
    }
    static std::atomic<uint64_t> temporarySequence{0};
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(::getpid()) + "." +
        std::to_string(temporarySequence.fetch_add(1, std::memory_order_relaxed));
    const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority temporary file could not be opened", true));
    }
    const bool wrote = writeAll(descriptor, fileBytes);
    const bool synchronized = wrote && ::fsync(descriptor) == 0;
    const bool closed = ::close(descriptor) == 0;
    if (!wrote || !synchronized || !closed) {
        std::filesystem::remove(temporary, error);
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority staging file could not be synchronized", true));
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
                std::string("Terrain authority publication test hook threw: ") + exception.what(),
                true));
        } catch (...) {
            removeTemporary();
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Terrain authority publication test hook threw", true));
        }
    }

    auto lock = acquireExclusivePagePublicationLock(path);
    if (!lock.isReady()) {
        removeTemporary();
        return AuthorityResult<bool>::failed(*lock.failure());
    }
    const ScopedPagePublicationLock heldLock(*lock.value());

    existing = loadPage(page.key);
    if (existing.isReady()) {
        removeTemporary();
        return matchesExisting(*existing.value());
    }
    const bool replacingCorruptPage =
        existing.status() == AuthorityStatus::FAILED && existing.failure() &&
        existing.failure()->code == GenerationFailureCode::CORRUPT_PAGE;
    if (existing.status() == AuthorityStatus::FAILED && !replacingCorruptPage) {
        removeTemporary();
        return AuthorityResult<bool>::failed(
            existing.failure()
                ? *existing.failure()
                : makeFailure(GenerationFailureCode::IO_ERROR,
                              "Terrain authority load failed without a failure reason", true));
    }

    const int publishResult = replacingCorruptPage ? ::rename(temporary.c_str(), path.c_str())
                                                   : ::link(temporary.c_str(), path.c_str());
    if (publishResult != 0) {
        const int publishError = errno;
        removeTemporary();
        if (!replacingCorruptPage && publishError == EEXIST) {
            // An unsynchronized or older process may have won publication.
            // Never replace it: reread the immutable payload and require an
            // exact quantized match before accepting the contention outcome.
            const auto contender = loadPage(page.key);
            if (contender.isReady()) return matchesExisting(*contender.value());
            if (contender.failure()) return AuthorityResult<bool>::failed(*contender.failure());
        }
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Terrain authority page could not be published exclusively", true));
    }
    if (!replacingCorruptPage) removeTemporary();
    synchronizeContainingDirectory(path);

    const auto published = loadPage(page.key);
    if (published.isReady()) return matchesExisting(*published.value());
    if (published.failure()) return AuthorityResult<bool>::failed(*published.failure());
    return AuthorityResult<bool>::failed(
        makeFailure(GenerationFailureCode::IO_ERROR,
                    "Terrain authority publication could not be validated", true));
}

class TransientTerrainGridStore {
public:
    TransientTerrainGridStore(std::filesystem::path root, GenerationIdentity identity)
        : root_(std::move(root))
        , identity_(std::move(identity)) {}

    [[nodiscard]] std::filesystem::path gridPath(NativeRect region) const {
        return root_ / ("g." + std::to_string(region.rowBegin) + "." +
                        std::to_string(region.columnBegin) + "." + std::to_string(region.rowEnd) +
                        "." + std::to_string(region.columnEnd) + ".rytg");
    }

    AuthorityResult<PhysicalTerrainGrid> loadGrid(NativeRect region) const {
        size_t sampleCount = 0;
        size_t payloadBytes = 0;
        if (!identity_.valid() || !region.valid() ||
            region.height() > std::numeric_limits<size_t>::max() ||
            region.width() > std::numeric_limits<size_t>::max() ||
            !checkedProduct(static_cast<size_t>(region.height()),
                            static_cast<size_t>(region.width()), sampleCount) ||
            sampleCount > MAXIMUM_AUTHORITY_QUERY_SAMPLES ||
            !checkedProduct(sampleCount, sizeof(QuantizedTerrainSample), payloadBytes) ||
            payloadBytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
                GenerationFailureCode::INVALID_REQUEST,
                "Transient terrain authority load received an invalid rectangle", false));
        }

        const std::filesystem::path path = gridPath(region);
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        if (error) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority could not be inspected", true));
        }
        if (!exists) {
            return AuthorityResult<PhysicalTerrainGrid>::deferred(
                makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                            "Transient terrain authority is not persisted", true));
        }

        const uintmax_t fileSize = std::filesystem::file_size(path, error);
        const int maximumCompressed = LZ4_compressBound(static_cast<int>(payloadBytes));
        if (error || fileSize < TRANSIENT_GRID_HEADER_BYTES || maximumCompressed <= 0 ||
            fileSize > TRANSIENT_GRID_HEADER_BYTES + static_cast<uintmax_t>(maximumCompressed)) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority file size is invalid", true));
        }
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority could not be opened", true));
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
        file.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        if (!file.good()) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority could not be read", true));
        }
        if (bytes.size() < TRANSIENT_GRID_HEADER_BYTES || bytes[0] != 'R' || bytes[1] != 'Y' ||
            bytes[2] != 'T' || bytes[3] != 'G') {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority magic is invalid", true));
        }

        size_t offset = 4;
        uint16_t schema = 0;
        uint16_t headerBytes = 0;
        if (!readU16(bytes, offset, schema) || !readU16(bytes, offset, headerBytes) ||
            schema != TRANSIENT_GRID_SCHEMA_VERSION || headerBytes != TRANSIENT_GRID_HEADER_BYTES ||
            bytes[offset++] != LZ4_COMPRESSION || bytes[offset++] != 0 || bytes[offset++] != 0 ||
            bytes[offset++] != 0) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority schema is invalid", true));
        }
        uint64_t seed = 0;
        if (!readU64(bytes, offset, seed)) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority seed is truncated", true));
        }
        Sha256Digest fingerprint{};
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), fingerprint.size(),
                    fingerprint.begin());
        offset += fingerprint.size();
        std::array<uint64_t, 4> encodedRegion{};
        for (uint64_t& coordinate : encodedRegion) {
            if (!readU64(bytes, offset, coordinate)) {
                return AuthorityResult<PhysicalTerrainGrid>::failed(
                    makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                                "Transient terrain authority rectangle is truncated", true));
            }
        }
        const NativeRect storedRegion{
            .rowBegin = static_cast<int64_t>(encodedRegion[0]),
            .columnBegin = static_cast<int64_t>(encodedRegion[1]),
            .rowEnd = static_cast<int64_t>(encodedRegion[2]),
            .columnEnd = static_cast<int64_t>(encodedRegion[3]),
        };
        uint32_t storedSampleCount = 0;
        uint32_t uncompressedBytes = 0;
        uint32_t compressedBytes = 0;
        uint32_t payloadChecksum = 0;
        uint32_t headerChecksum = 0;
        if (!readU32(bytes, offset, storedSampleCount) ||
            !readU32(bytes, offset, uncompressedBytes) ||
            !readU32(bytes, offset, compressedBytes) || !readU32(bytes, offset, payloadChecksum) ||
            !readU32(bytes, offset, headerChecksum) || offset != TRANSIENT_GRID_HEADER_BYTES ||
            storedRegion != region || storedSampleCount != sampleCount ||
            uncompressedBytes != payloadBytes || compressedBytes == 0 ||
            bytes.size() != TRANSIENT_GRID_HEADER_BYTES + compressedBytes ||
            headerChecksum != crc32(std::span<const uint8_t>(bytes).first(
                                  TRANSIENT_GRID_HEADER_CHECKSUM_OFFSET))) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority header validation failed", true));
        }
        if (seed != identity_.seed || fingerprint != identity_.fingerprint()) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
                GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                "Transient terrain authority belongs to another generation identity", false));
        }

        std::vector<uint8_t> payload(payloadBytes);
        const int written = LZ4_decompress_safe(
            reinterpret_cast<const char*>(bytes.data() + TRANSIENT_GRID_HEADER_BYTES),
            reinterpret_cast<char*>(payload.data()), static_cast<int>(compressedBytes),
            static_cast<int>(payload.size()));
        if (written != static_cast<int>(payload.size()) || crc32(payload) != payloadChecksum) {
            return AuthorityResult<PhysicalTerrainGrid>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain authority payload validation failed", true));
        }

        PhysicalTerrainGrid grid{.region = region,
                                 .samples = std::vector<PhysicalTerrainSample>(sampleCount)};
        offset = 0;
        for (PhysicalTerrainSample& sample : grid.samples) {
            QuantizedTerrainSample quantized;
            uint16_t elevation = 0;
            uint16_t temperature = 0;
            uint16_t lapseRate = 0;
            if (!readU16(payload, offset, elevation) || !readU16(payload, offset, temperature) ||
                !readU16(payload, offset, quantized.temperatureVariabilityCentidegrees) ||
                !readU16(payload, offset, quantized.annualPrecipitationMillimeters) ||
                !readU16(payload, offset, quantized.precipitationCoefficientBasisPoints) ||
                !readU16(payload, offset, lapseRate)) {
                return AuthorityResult<PhysicalTerrainGrid>::failed(
                    makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                                "Transient terrain authority payload is truncated", true));
            }
            quantized.elevationMeters = static_cast<int16_t>(elevation);
            quantized.meanTemperatureCentidegrees = static_cast<int16_t>(temperature);
            quantized.lapseRateMicrodegreesPerMeter = static_cast<int16_t>(lapseRate);
            sample = dequantizeTerrainSample(quantized);
        }
        return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(grid));
    }

    AuthorityResult<bool> writeGrid(const PhysicalTerrainGrid& grid) const {
        if (!identity_.valid() || !grid.valid() ||
            grid.samples.size() > MAXIMUM_AUTHORITY_QUERY_SAMPLES ||
            grid.samples.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "Transient terrain authority grid is invalid", false));
        }

        std::vector<uint8_t> payload;
        payload.reserve(grid.samples.size() * sizeof(QuantizedTerrainSample));
        for (const PhysicalTerrainSample& sample : grid.samples) {
            const QuantizedTerrainSample quantized = quantizePhysicalTerrainSample(sample);
            appendU16(payload, static_cast<uint16_t>(quantized.elevationMeters));
            appendU16(payload, static_cast<uint16_t>(quantized.meanTemperatureCentidegrees));
            appendU16(payload, quantized.temperatureVariabilityCentidegrees);
            appendU16(payload, quantized.annualPrecipitationMillimeters);
            appendU16(payload, quantized.precipitationCoefficientBasisPoints);
            appendU16(payload, static_cast<uint16_t>(quantized.lapseRateMicrodegreesPerMeter));
        }

        const auto matchesExisting = [&](const PhysicalTerrainGrid& existing) {
            if (existing.region == grid.region && existing.samples == grid.samples)
                return AuthorityResult<bool>::ready(true);
            return AuthorityResult<bool>::failed(makeFailure(
                GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                "Transient terrain authority is immutable for its generation identity", false));
        };
        auto existing = loadGrid(grid.region);
        if (existing.isReady()) return matchesExisting(*existing.value());
        if (existing.status() == AuthorityStatus::FAILED &&
            (!existing.failure() ||
             existing.failure()->code != GenerationFailureCode::CORRUPT_PAGE)) {
            return AuthorityResult<bool>::failed(
                existing.failure() ? *existing.failure()
                                   : makeFailure(GenerationFailureCode::IO_ERROR,
                                                 "Transient terrain authority load failed", true));
        }

        const int maximum = LZ4_compressBound(static_cast<int>(payload.size()));
        if (maximum <= 0) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority compression bound failed", true));
        }
        std::vector<uint8_t> compressed(static_cast<size_t>(maximum));
        const int compressedBytes = LZ4_compress_default(
            reinterpret_cast<const char*>(payload.data()),
            reinterpret_cast<char*>(compressed.data()), static_cast<int>(payload.size()), maximum);
        if (compressedBytes <= 0) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority compression failed", true));
        }
        compressed.resize(static_cast<size_t>(compressedBytes));

        std::vector<uint8_t> fileBytes;
        fileBytes.reserve(TRANSIENT_GRID_HEADER_BYTES + compressed.size());
        fileBytes.insert(fileBytes.end(), {'R', 'Y', 'T', 'G'});
        appendU16(fileBytes, TRANSIENT_GRID_SCHEMA_VERSION);
        appendU16(fileBytes, static_cast<uint16_t>(TRANSIENT_GRID_HEADER_BYTES));
        fileBytes.push_back(LZ4_COMPRESSION);
        fileBytes.insert(fileBytes.end(), {0, 0, 0});
        appendU64(fileBytes, identity_.seed);
        appendDigest(fileBytes, identity_.fingerprint());
        appendU64(fileBytes, static_cast<uint64_t>(grid.region.rowBegin));
        appendU64(fileBytes, static_cast<uint64_t>(grid.region.columnBegin));
        appendU64(fileBytes, static_cast<uint64_t>(grid.region.rowEnd));
        appendU64(fileBytes, static_cast<uint64_t>(grid.region.columnEnd));
        appendU32(fileBytes, static_cast<uint32_t>(grid.samples.size()));
        appendU32(fileBytes, static_cast<uint32_t>(payload.size()));
        appendU32(fileBytes, static_cast<uint32_t>(compressed.size()));
        appendU32(fileBytes, crc32(payload));
        appendU32(fileBytes, crc32(fileBytes));
        if (fileBytes.size() != TRANSIENT_GRID_HEADER_BYTES) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority header construction failed", false));
        }
        fileBytes.insert(fileBytes.end(), compressed.begin(), compressed.end());

        const std::filesystem::path path = gridPath(grid.region);
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Transient terrain authority directory could not be created", true));
        }
        static std::atomic<uint64_t> temporarySequence{0};
        const std::filesystem::path temporary =
            path.string() + ".tmp." + std::to_string(::getpid()) + "." +
            std::to_string(temporarySequence.fetch_add(1, std::memory_order_relaxed));
        const int descriptor =
            ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (descriptor < 0) {
            return AuthorityResult<bool>::failed(makeFailure(
                GenerationFailureCode::IO_ERROR,
                "Transient terrain authority temporary file could not be opened", true));
        }
        const bool wrote = writeAll(descriptor, fileBytes);
        const bool synchronized = wrote && ::fsync(descriptor) == 0;
        const bool closed = ::close(descriptor) == 0;
        if (!wrote || !synchronized || !closed) {
            std::filesystem::remove(temporary, error);
            return AuthorityResult<bool>::failed(makeFailure(
                GenerationFailureCode::IO_ERROR,
                "Transient terrain authority staging file could not be synchronized", true));
        }
        const auto removeTemporary = [&] {
            std::error_code cleanupError;
            std::filesystem::remove(temporary, cleanupError);
        };
        auto lock = acquireExclusivePagePublicationLock(path);
        if (!lock.isReady()) {
            removeTemporary();
            return AuthorityResult<bool>::failed(*lock.failure());
        }
        const ScopedPagePublicationLock heldLock(*lock.value());
        existing = loadGrid(grid.region);
        if (existing.isReady()) {
            removeTemporary();
            return matchesExisting(*existing.value());
        }
        const bool replacingCorrupt =
            existing.status() == AuthorityStatus::FAILED && existing.failure() &&
            existing.failure()->code == GenerationFailureCode::CORRUPT_PAGE;
        if (existing.status() == AuthorityStatus::FAILED && !replacingCorrupt) {
            removeTemporary();
            return AuthorityResult<bool>::failed(
                existing.failure()
                    ? *existing.failure()
                    : makeFailure(GenerationFailureCode::IO_ERROR,
                                  "Transient terrain authority load failed without a reason",
                                  true));
        }
        const int publishResult = replacingCorrupt ? ::rename(temporary.c_str(), path.c_str())
                                                   : ::link(temporary.c_str(), path.c_str());
        if (publishResult != 0) {
            const int publishError = errno;
            removeTemporary();
            if (!replacingCorrupt && publishError == EEXIST) {
                const auto contender = loadGrid(grid.region);
                if (contender.isReady()) return matchesExisting(*contender.value());
                if (contender.failure()) return AuthorityResult<bool>::failed(*contender.failure());
            }
            return AuthorityResult<bool>::failed(makeFailure(
                GenerationFailureCode::IO_ERROR,
                "Transient terrain authority could not be published exclusively", true));
        }
        if (!replacingCorrupt) removeTemporary();
        synchronizeContainingDirectory(path);

        const auto published = loadGrid(grid.region);
        if (published.isReady()) return matchesExisting(*published.value());
        if (published.failure()) return AuthorityResult<bool>::failed(*published.failure());
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::IO_ERROR,
                        "Transient terrain authority publication could not be validated", true));
    }

private:
    std::filesystem::path root_;
    GenerationIdentity identity_;
};

class CachedTerrainAuthority::Impl {
public:
    using PagePointer = std::shared_ptr<const TerrainAuthorityPage>;
    using PageResult = AuthorityResult<PagePointer>;
    using CoarseResult = AuthorityResult<CoarseSpawnGrid>;
    using GridPointer = std::shared_ptr<const PhysicalTerrainGrid>;
    using GridResult = AuthorityResult<GridPointer>;

    struct CacheEntry {
        PagePointer page;
        size_t bytes = 0;
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        std::list<TerrainPageKey>::iterator recency;
    };

    struct Flight {
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        ProtectedHandoffEpoch handoffEpoch;
        uint64_t sequence = 0;
        bool started = false;
        bool publishing = false;
        bool done = false;
        PageResult result =
            PageResult::failed(makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                           "Terrain authority request did not complete", true));
    };

    struct CoarseSpawnFlight {
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        ProtectedHandoffEpoch handoffEpoch;
        uint64_t sequence = 0;
        bool started = false;
        bool done = false;
        CoarseResult result =
            CoarseResult::failed(makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                             "Coarse spawn request did not complete", true));
    };

    struct TransientGridFlight {
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        ProtectedHandoffEpoch handoffEpoch;
        uint64_t sequence = 0;
        bool started = false;
        bool done = false;
        GridResult result = GridResult::failed(
            makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                        "Transient final terrain request did not complete", true));
    };

    struct CoarseSpawnCacheEntry {
        CoarseSpawnGrid grid;
        uint64_t recency = 0;
    };

    struct TransientGridCacheEntry {
        GridPointer grid;
        size_t bytes = 0;
        uint64_t recency = 0;
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
    };

    struct PublicationTask {
        TerrainPageKey key;
        std::shared_ptr<Flight> flight;
        PagePointer page;
        bool repair = false;
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        ProtectedHandoffEpoch handoffEpoch;
        uint64_t sequence = 0;
    };

    struct PageBuildWork {
        TerrainPageKey key;
        std::shared_ptr<Flight> flight;
    };

    struct PageBuildResult {
        TerrainPageKey key;
        PageResult result = PageResult::failed(
            makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                        "Terrain authority batch request did not complete", true));
        bool requiresPublication = false;
        bool repair = false;
    };

    Impl(GenerationIdentity requestedIdentity, std::filesystem::path storeRoot,
         std::shared_ptr<TerrainInferenceBackend> requestedBackend,
         TerrainAuthorityCacheConfig requestedConfig)
        : identity(std::move(requestedIdentity))
        , store(storeRoot, identity)
        , transientStore(std::move(storeRoot) / "transient-final-v1", identity)
        , backend(std::move(requestedBackend))
        , config(requestedConfig) {
        config.maximumOutstandingRequests = std::clamp(config.maximumOutstandingRequests, size_t{1},
                                                       MAXIMUM_AUTHORITY_QUEUED_REQUESTS);
        // Production inference is deliberately single-call. Keep the legacy
        // configuration field fingerprint-neutral while enforcing the one
        // coordinator thread contract here.
        config.maximumConcurrentBuilds = 1;
        config.maximumQueryPages =
            std::clamp(config.maximumQueryPages, size_t{1}, MAXIMUM_AUTHORITY_QUERY_PAGES);
        config.maximumQuerySamples =
            std::clamp(config.maximumQuerySamples, size_t{1}, MAXIMUM_AUTHORITY_QUERY_SAMPLES);
    }

    static bool isLowPriorityRequest(AuthorityRequestPriority priority) noexcept {
        return priority >= AuthorityRequestPriority::COARSE_PREVIEW;
    }

    static bool isLowPriorityRequest(TerrainPageKey key,
                                     AuthorityRequestPriority priority) noexcept {
        // PREVIEW data is always horizon-quality work, even when a caller
        // reaches the direct preparePage default or accidentally assigns it
        // an urgent service lane. Priority may accelerate an existing
        // preview flight, but it may not consume the urgent reservation.
        return key.quality == AuthorityQuality::PREVIEW || isLowPriorityRequest(priority);
    }

    static bool isVisibleOrLowerRequest(AuthorityRequestPriority priority) noexcept {
        return priority >= AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT;
    }

    static bool isVisibleOrLowerRequest(TerrainPageKey key,
                                        AuthorityRequestPriority priority) noexcept {
        return key.quality == AuthorityQuality::PREVIEW || isVisibleOrLowerRequest(priority);
    }

    static AuthorityRequestPriority effectivePriority(TerrainPageKey key,
                                                      AuthorityRequestPriority priority) noexcept {
        return key.quality == AuthorityQuality::PREVIEW
                   ? std::max(priority, AuthorityRequestPriority::COARSE_PREVIEW)
                   : priority;
    }

    static bool validHandoffEpoch(AuthorityRequestPriority priority,
                                  ProtectedHandoffEpoch epoch) noexcept {
        return !epoch.valid() || priority == AuthorityRequestPriority::PROTECTED_HANDOFF;
    }

    static bool validHandoffEpoch(TerrainPageKey key, AuthorityRequestPriority priority,
                                  ProtectedHandoffEpoch epoch) noexcept {
        return validHandoffEpoch(priority, epoch) &&
               (!epoch.valid() || key.quality == AuthorityQuality::FINAL);
    }

    void observeHandoffEpochLocked(ProtectedHandoffEpoch epoch) noexcept {
        if (!epoch.valid() || epoch.value <= currentProtectedHandoffEpoch.value) return;
        currentProtectedHandoffEpoch = epoch;
        metrics.currentProtectedHandoffEpoch = epoch.value;
    }

    [[nodiscard]] bool staleHandoffEpochLocked(ProtectedHandoffEpoch epoch) const noexcept {
        return epoch.valid() && currentProtectedHandoffEpoch.valid() &&
               epoch.value < currentProtectedHandoffEpoch.value;
    }

    [[nodiscard]] bool requestBeforeLocked(AuthorityRequestPriority leftPriority,
                                           ProtectedHandoffEpoch leftEpoch, uint64_t leftSequence,
                                           AuthorityRequestPriority rightPriority,
                                           ProtectedHandoffEpoch rightEpoch,
                                           uint64_t rightSequence) const noexcept {
        if (leftPriority != rightPriority) return leftPriority < rightPriority;
        if (leftPriority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
            leftEpoch != rightEpoch) {
            const bool leftCurrent = leftEpoch.valid() && leftEpoch == currentProtectedHandoffEpoch;
            const bool rightCurrent =
                rightEpoch.valid() && rightEpoch == currentProtectedHandoffEpoch;
            if (leftCurrent != rightCurrent) return leftCurrent;
            return leftEpoch.value > rightEpoch.value;
        }
        return leftSequence < rightSequence;
    }

    struct PreemptionCandidate {
        enum class Kind : uint8_t {
            Page,
            CoarseSpawn,
            TransientGrid,
        } kind = Kind::Page;
        TerrainPageKey pageKey;
        CoarseSpawnRegion coarseRegion;
        NativeRect transientRegion;
        AuthorityRequestPriority priority = AuthorityRequestPriority::SPECULATIVE_PREFETCH;
        ProtectedHandoffEpoch handoffEpoch;
        uint64_t sequence = 0;
    };

    [[nodiscard]] bool
    preemptForPriorityLocked(size_t requiredSlots, AuthorityRequestPriority incomingPriority,
                             ProtectedHandoffEpoch incomingEpoch = {},
                             std::span<const TerrainPageKey> excludedPageKeys = {}) {
        if (requiredSlots == 0) return true;
        std::vector<PreemptionCandidate> candidates;
        candidates.reserve(outstandingRequestCountLocked());
        const auto eligible = [&](AuthorityRequestPriority priority, ProtectedHandoffEpoch epoch,
                                  bool started, bool publishing) {
            if (started || publishing) return false;
            if (priority > incomingPriority) return true;
            return incomingPriority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
                   incomingEpoch.valid() &&
                   priority == AuthorityRequestPriority::PROTECTED_HANDOFF && epoch.valid() &&
                   staleHandoffEpochLocked(epoch);
        };
        for (const auto& [key, flight] : flights) {
            if (std::ranges::find(excludedPageKeys, key) != excludedPageKeys.end()) continue;
            const AuthorityRequestPriority requestPriority =
                effectivePriority(key, flight->priority);
            if (!eligible(requestPriority, flight->handoffEpoch, flight->started,
                          flight->publishing)) {
                continue;
            }
            candidates.push_back({.kind = PreemptionCandidate::Kind::Page,
                                  .pageKey = key,
                                  .priority = requestPriority,
                                  .handoffEpoch = flight->handoffEpoch,
                                  .sequence = flight->sequence});
        }
        for (const auto& [region, flight] : coarseSpawnFlights) {
            if (!eligible(flight->priority, flight->handoffEpoch, flight->started, false)) continue;
            candidates.push_back({.kind = PreemptionCandidate::Kind::CoarseSpawn,
                                  .coarseRegion = region,
                                  .priority = flight->priority,
                                  .handoffEpoch = flight->handoffEpoch,
                                  .sequence = flight->sequence});
        }
        for (const auto& [region, flight] : transientGridFlights) {
            if (!eligible(flight->priority, flight->handoffEpoch, flight->started, false)) continue;
            candidates.push_back({.kind = PreemptionCandidate::Kind::TransientGrid,
                                  .transientRegion = region,
                                  .priority = flight->priority,
                                  .handoffEpoch = flight->handoffEpoch,
                                  .sequence = flight->sequence});
        }
        if (candidates.size() < requiredSlots) return false;
        std::sort(candidates.begin(), candidates.end(),
                  [](const PreemptionCandidate& left, const PreemptionCandidate& right) {
                      if (left.priority != right.priority) return left.priority > right.priority;
                      if (left.priority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
                          left.handoffEpoch != right.handoffEpoch) {
                          return left.handoffEpoch.value < right.handoffEpoch.value;
                      }
                      return left.sequence > right.sequence;
                  });
        candidates.resize(requiredSlots);
        for (const PreemptionCandidate& candidate : candidates) {
            switch (candidate.kind) {
                case PreemptionCandidate::Kind::Page:
                    std::erase(queue, candidate.pageKey);
                    flights.erase(candidate.pageKey);
                    break;
                case PreemptionCandidate::Kind::CoarseSpawn:
                    std::erase(coarseSpawnQueue, candidate.coarseRegion);
                    coarseSpawnFlights.erase(candidate.coarseRegion);
                    break;
                case PreemptionCandidate::Kind::TransientGrid:
                    std::erase(transientGridQueue, candidate.transientRegion);
                    transientGridFlights.erase(candidate.transientRegion);
                    break;
            }
        }
        metrics.higherPriorityPreemptions += candidates.size();
        if (incomingPriority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
            incomingEpoch.valid()) {
            metrics.protectedHandoffPreemptions += candidates.size();
        }
        return true;
    }

    [[nodiscard]] size_t outstandingRequestCountLocked() const noexcept {
        return flights.size() + coarseSpawnFlights.size() + transientGridFlights.size();
    }

    [[nodiscard]] size_t lowPriorityOutstandingRequestCountLocked() const noexcept {
        size_t count = 0;
        for (const auto& [key, flight] : flights)
            count += isLowPriorityRequest(key, flight->priority) ? 1U : 0U;
        for (const auto& [region, flight] : coarseSpawnFlights) {
            static_cast<void>(region);
            count += isLowPriorityRequest(flight->priority) ? 1U : 0U;
        }
        for (const auto& [region, flight] : transientGridFlights) {
            static_cast<void>(region);
            count += isLowPriorityRequest(flight->priority) ? 1U : 0U;
        }
        return count;
    }

    [[nodiscard]] size_t visibleOrLowerOutstandingRequestCountLocked() const noexcept {
        size_t count = 0;
        for (const auto& [key, flight] : flights)
            count += isVisibleOrLowerRequest(key, flight->priority) ? 1U : 0U;
        for (const auto& [region, flight] : coarseSpawnFlights) {
            static_cast<void>(region);
            count += isVisibleOrLowerRequest(flight->priority) ? 1U : 0U;
        }
        for (const auto& [region, flight] : transientGridFlights) {
            static_cast<void>(region);
            count += isVisibleOrLowerRequest(flight->priority) ? 1U : 0U;
        }
        return count;
    }

    [[nodiscard]] size_t lowPriorityRequestLimit() const noexcept {
        return std::min(config.maximumOutstandingRequests, MAXIMUM_LOW_PRIORITY_AUTHORITY_REQUESTS);
    }

    [[nodiscard]] size_t visibleOrLowerRequestLimit() const noexcept {
        return std::min(config.maximumOutstandingRequests,
                        MAXIMUM_VISIBLE_OR_LOWER_AUTHORITY_REQUESTS);
    }

    [[nodiscard]] bool hasTotalCapacityLocked(size_t additionalRequests) const noexcept {
        const size_t occupied = outstandingRequestCountLocked();
        return occupied <= config.maximumOutstandingRequests &&
               additionalRequests <= config.maximumOutstandingRequests - occupied;
    }

    [[nodiscard]] bool hasLowPriorityCapacityLocked(size_t currentLowPriorityRequests,
                                                    size_t additionalRequests) const noexcept {
        const size_t limit = lowPriorityRequestLimit();
        return currentLowPriorityRequests <= limit &&
               additionalRequests <= limit - currentLowPriorityRequests;
    }

    [[nodiscard]] bool hasLowPriorityCapacityLocked(size_t additionalRequests) const noexcept {
        return hasLowPriorityCapacityLocked(lowPriorityOutstandingRequestCountLocked(),
                                            additionalRequests);
    }

    [[nodiscard]] bool hasVisibleOrLowerCapacityLocked(size_t currentVisibleOrLowerRequests,
                                                       size_t additionalRequests) const noexcept {
        const size_t limit = visibleOrLowerRequestLimit();
        return currentVisibleOrLowerRequests <= limit &&
               additionalRequests <= limit - currentVisibleOrLowerRequests;
    }

    [[nodiscard]] bool hasVisibleOrLowerCapacityLocked(size_t additionalRequests) const noexcept {
        return hasVisibleOrLowerCapacityLocked(visibleOrLowerOutstandingRequestCountLocked(),
                                               additionalRequests);
    }

    static const char* capacityFailureMessage(bool hasTotalCapacity,
                                              bool hasLowPriorityCapacity) noexcept {
        if (!hasTotalCapacity) return "Terrain authority request queue is full";
        if (!hasLowPriorityCapacity) return "Terrain authority low-priority reservation is full";
        return "Terrain authority visible-final reservation is full";
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        if (coordinator.joinable()) coordinator.join();
        {
            std::lock_guard lock(mutex);
            publicationStopping = true;
        }
        publicationCondition.notify_all();
        for (std::thread& worker : publicationWorkers) {
            if (worker.joinable()) worker.join();
        }
    }

    void touch(std::map<TerrainPageKey, CacheEntry>::iterator entry) {
        recency.splice(recency.begin(), recency, entry->second.recency);
        entry->second.recency = recency.begin();
    }

    void ensureCoordinatorStartedLocked() {
        if (coordinator.joinable() || stopping) return;
        coordinator = std::thread([this] { coordinatorLoop(); });
        metrics.coordinatorStarted = true;
    }

    void ensurePublicationWorkersStartedLocked() {
        if (!publicationWorkers.empty() || publicationStopping) return;
        constexpr size_t MAXIMUM_PUBLICATION_WORKERS = 4;
        const size_t available = std::thread::hardware_concurrency() == 0
                                     ? 1U
                                     : static_cast<size_t>(std::thread::hardware_concurrency());
        const size_t workerCount = std::clamp<size_t>(available, 1, MAXIMUM_PUBLICATION_WORKERS);
        publicationWorkers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            publicationWorkers.emplace_back([this] { publicationLoop(); });
        metrics.publicationWorkersStarted = true;
    }

    bool evictPageFor(AuthorityRequestPriority incomingPriority) {
        if (cache.empty()) return false;
        auto victim = cache.end();
        for (auto key = recency.rbegin(); key != recency.rend(); ++key) {
            const auto found = cache.find(*key);
            if (found == cache.end()) std::terminate();
            if (found->second.priority < incomingPriority) continue;
            if (victim == cache.end() || found->second.priority > victim->second.priority)
                victim = found;
        }
        if (victim == cache.end()) return false;
        cacheBytes -= victim->second.bytes;
        recency.erase(victim->second.recency);
        cache.erase(victim);
        ++metrics.evictions;
        return true;
    }

    bool evictTransientGridFor(AuthorityRequestPriority incomingPriority) {
        if (transientGridCache.empty()) return false;
        const auto victim =
            std::max_element(transientGridCache.begin(), transientGridCache.end(),
                             [incomingPriority](const auto& left, const auto& right) {
                                 const bool leftEligible = left.second.priority >= incomingPriority;
                                 const bool rightEligible =
                                     right.second.priority >= incomingPriority;
                                 if (leftEligible != rightEligible) return !leftEligible;
                                 if (!leftEligible) return false;
                                 if (left.second.priority != right.second.priority)
                                     return left.second.priority < right.second.priority;
                                 return left.second.recency > right.second.recency;
                             });
        if (victim == transientGridCache.end() || victim->second.priority < incomingPriority)
            return false;
        transientGridCacheBytes -= victim->second.bytes;
        transientGridCache.erase(victim);
        ++metrics.evictions;
        return true;
    }

    static bool nativeRectContains(NativeRect outer, NativeRect inner) noexcept {
        return outer.rowBegin <= inner.rowBegin && outer.columnBegin <= inner.columnBegin &&
               outer.rowEnd >= inner.rowEnd && outer.columnEnd >= inner.columnEnd;
    }

    std::map<NativeRect, TransientGridCacheEntry>::iterator
    smallestContainingTransientGrid(NativeRect region) {
        auto best = transientGridCache.end();
        uint64_t bestArea = std::numeric_limits<uint64_t>::max();
        for (auto candidate = transientGridCache.begin(); candidate != transientGridCache.end();
             ++candidate) {
            if (!nativeRectContains(candidate->first, region)) continue;
            const uint64_t area = candidate->first.height() * candidate->first.width();
            if (best == transientGridCache.end() || area < bestArea ||
                (area == bestArea && candidate->first < best->first)) {
                best = candidate;
                bestArea = area;
            }
        }
        return best;
    }

    std::map<NativeRect, std::shared_ptr<TransientGridFlight>>::iterator
    smallestContainingTransientGridFlight(NativeRect region) {
        auto best = transientGridFlights.end();
        uint64_t bestArea = std::numeric_limits<uint64_t>::max();
        for (auto candidate = transientGridFlights.begin(); candidate != transientGridFlights.end();
             ++candidate) {
            if (candidate->second->done || !nativeRectContains(candidate->first, region)) continue;
            const uint64_t area = candidate->first.height() * candidate->first.width();
            if (best == transientGridFlights.end() || area < bestArea ||
                (area == bestArea && candidate->first < best->first)) {
                best = candidate;
                bestArea = area;
            }
        }
        return best;
    }

    static GridPointer cropTransientGrid(const PhysicalTerrainGrid& source, NativeRect region) {
        if (!source.valid() || !region.valid() || !nativeRectContains(source.region, region) ||
            region.height() > std::numeric_limits<size_t>::max() ||
            region.width() > std::numeric_limits<size_t>::max()) {
            return nullptr;
        }
        size_t sampleCount = 0;
        if (!checkedProduct(static_cast<size_t>(region.height()),
                            static_cast<size_t>(region.width()), sampleCount)) {
            return nullptr;
        }
        PhysicalTerrainGrid crop{
            .region = region,
            .samples = std::vector<PhysicalTerrainSample>(sampleCount),
        };
        const size_t sourceWidth = static_cast<size_t>(source.region.width());
        const size_t cropWidth = static_cast<size_t>(region.width());
        const size_t sourceColumn =
            static_cast<size_t>(static_cast<uint64_t>(region.columnBegin) -
                                static_cast<uint64_t>(source.region.columnBegin));
        for (size_t row = 0; row < static_cast<size_t>(region.height()); ++row) {
            const size_t sourceRow =
                static_cast<size_t>(static_cast<uint64_t>(region.rowBegin) -
                                    static_cast<uint64_t>(source.region.rowBegin)) +
                row;
            std::copy_n(source.samples.begin() +
                            static_cast<std::ptrdiff_t>(sourceRow * sourceWidth + sourceColumn),
                        cropWidth,
                        crop.samples.begin() + static_cast<std::ptrdiff_t>(row * cropWidth));
        }
        return std::make_shared<const PhysicalTerrainGrid>(std::move(crop));
    }

    void insert(PagePointer page, AuthorityRequestPriority priority) {
        priority = effectivePriority(page->key, priority);
        if (config.maximumEntries == 0 || config.byteBudget == 0 ||
            page->byteSize() > config.byteBudget) {
            return;
        }
        auto existing = cache.find(page->key);
        if (existing != cache.end()) {
            existing->second.priority = std::min(existing->second.priority, priority);
            touch(existing);
            return;
        }
        while (cache.size() >= config.maximumEntries)
            if (!evictPageFor(priority)) return;
        while (cacheBytes + transientGridCacheBytes > config.byteBudget - page->byteSize()) {
            // A new distant decode must never displace a stronger exact or
            // spawn entry. Within the same lane, ordinary LRU behavior still
            // bounds a moving working set.
            if (!evictTransientGridFor(priority) && !evictPageFor(priority)) return;
        }
        const TerrainPageKey key = page->key;
        const size_t bytes = page->byteSize();
        recency.push_front(key);
        cacheBytes += bytes;
        cache.emplace(key, CacheEntry{
                               .page = std::move(page),
                               .bytes = bytes,
                               .priority = priority,
                               .recency = recency.begin(),
                           });
    }

    void insertCoarseSpawnGrid(CoarseSpawnGrid grid) {
        constexpr size_t MAXIMUM_COARSE_SPAWN_GRID_CACHE_ENTRIES = 16;
        const CoarseSpawnRegion key = grid.region;
        const uint64_t recency = ++coarseSpawnRecency;
        const auto existing = coarseSpawnCache.find(key);
        if (existing != coarseSpawnCache.end()) {
            existing->second.grid = std::move(grid);
            existing->second.recency = recency;
            return;
        }
        if (coarseSpawnCache.size() >= MAXIMUM_COARSE_SPAWN_GRID_CACHE_ENTRIES) {
            const auto oldest =
                std::min_element(coarseSpawnCache.begin(), coarseSpawnCache.end(),
                                 [](const auto& left, const auto& right) {
                                     return left.second.recency < right.second.recency;
                                 });
            if (oldest != coarseSpawnCache.end()) coarseSpawnCache.erase(oldest);
        }
        coarseSpawnCache.emplace(key, CoarseSpawnCacheEntry{
                                          .grid = std::move(grid),
                                          .recency = recency,
                                      });
    }

    bool insertTransientGrid(GridPointer grid, AuthorityRequestPriority priority) {
        // One exact far parent can straddle both axes of a native-hydrology
        // seam and therefore needs four distinct owner inputs to remain
        // resident until its worker constructs those owners. Use the same
        // hard request bound as the coordinator instead of a two-entry LRU;
        // the decoded-authority byte budget remains the primary memory bound.
        constexpr size_t MAXIMUM_TRANSIENT_GRID_CACHE_ENTRIES = MAXIMUM_AUTHORITY_QUEUED_REQUESTS;
        const NativeRect key = grid->region;
        if (grid->samples.size() >
            std::numeric_limits<size_t>::max() / sizeof(PhysicalTerrainSample)) {
            return false;
        }
        const size_t bytes = grid->samples.size() * sizeof(PhysicalTerrainSample);
        if (config.byteBudget == 0 || bytes > config.byteBudget) return false;
        const uint64_t recency = ++transientGridRecency;
        const auto existing = transientGridCache.find(key);
        if (existing != transientGridCache.end()) {
            existing->second.priority = std::min(existing->second.priority, priority);
            existing->second.recency = recency;
            return true;
        }
        if (transientGridCache.size() >= MAXIMUM_TRANSIENT_GRID_CACHE_ENTRIES) {
            if (!evictTransientGridFor(priority)) return false;
        }
        while (cacheBytes + transientGridCacheBytes > config.byteBudget - bytes) {
            if (!evictTransientGridFor(priority) && !evictPageFor(priority)) return false;
        }
        transientGridCacheBytes += bytes;
        transientGridCache.emplace(key, TransientGridCacheEntry{
                                            .grid = std::move(grid),
                                            .bytes = bytes,
                                            .recency = recency,
                                            .priority = priority,
                                        });
        return true;
    }

    PageResult materializeFinalPageFromTransient(TerrainPageKey key,
                                                 AuthorityRequestPriority priority) {
        if (key.quality != AuthorityQuality::FINAL) {
            return PageResult::deferred(
                makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                            "No cached transient terrain can materialize a preview page", true));
        }
        const std::optional<NativeRect> pageRegion = terrainPageNativeRect(key.coordinate);
        if (!pageRegion) {
            return PageResult::failed(makeFailure(
                GenerationFailureCode::INVALID_REQUEST,
                "Terrain authority page coordinate overflowed during transient materialization",
                false));
        }

        GridPointer containingGrid;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto containing = smallestContainingTransientGrid(*pageRegion);
            if (containing == transientGridCache.end()) {
                return PageResult::deferred(
                    makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                                "No cached transient terrain contains the authority page", true));
            }
            containingGrid = containing->second.grid;
            containing->second.recency = ++transientGridRecency;
            containing->second.priority = std::min(containing->second.priority, priority);
            ++metrics.hits;
        }

        const GridPointer crop = cropTransientGrid(*containingGrid, *pageRegion);
        if (!crop || crop->samples.size() != AUTHORITY_PAGE_SAMPLE_COUNT) {
            return PageResult::failed(makeFailure(
                GenerationFailureCode::CORRUPT_PAGE,
                "Cached transient terrain could not be cropped to an authority page", true));
        }

        TerrainAuthorityPage page{
            .key = key,
            .generationSeed = identity.seed,
            .generationFingerprint = identity.fingerprint(),
            .samples = std::vector<QuantizedTerrainSample>(AUTHORITY_PAGE_SAMPLE_COUNT),
        };
        std::transform(crop->samples.begin(), crop->samples.end(), page.samples.begin(),
                       quantizePhysicalTerrainSample);
        if (!page.valid() || !page.matches(identity)) {
            return PageResult::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Transient terrain produced an invalid authority page", true));
        }
        return PageResult::ready(std::make_shared<const TerrainAuthorityPage>(std::move(page)));
    }

    std::vector<PageBuildResult> inferOrLoadBatch(std::span<const TerrainPageKey> keys,
                                                  AuthorityRequestPriority priority) {
        // Observability only: one authority build span per batch, keyed by the
        // batch's lead page (see common/trace.hpp).
        trace::Scope span(
            trace::Track::LearnedAuthority, trace::Name::AuthorityPageBuild,
            {.spatialKey =
                 keys.empty()
                     ? 0
                     : trace::packCoord(keys.front().coordinate.row, keys.front().coordinate.column,
                                        static_cast<uint8_t>(keys.front().quality)),
             .quality = keys.empty() ? uint8_t{0} : static_cast<uint8_t>(keys.front().quality),
             .priority = static_cast<uint8_t>(priority)});
        std::vector<PageBuildResult> results;
        results.reserve(keys.size());
        std::vector<size_t> missing;
        missing.reserve(keys.size());
        for (const TerrainPageKey key : keys) {
            results.push_back({.key = key});
            PageBuildResult& result = results.back();
            const auto loaded = store.loadPage(key);
            if (loaded.isReady()) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    ++metrics.diskLoads;
                }
                result.result = PageResult::ready(
                    std::make_shared<const TerrainAuthorityPage>(std::move(*loaded.value())));
                continue;
            }
            if (loaded.status() == AuthorityStatus::FAILED) {
                if (!loaded.failure()) {
                    result.result = PageResult::failed(makeFailure(
                        GenerationFailureCode::IO_ERROR,
                        "Terrain authority page loading failed without a reason", true));
                    continue;
                }
                if (loaded.failure()->code == GenerationFailureCode::CORRUPT_PAGE) {
                    result.repair = true;
                } else {
                    result.result = PageResult::failed(*loaded.failure());
                    continue;
                }
            }

            PageResult materialized = materializeFinalPageFromTransient(key, priority);
            if (materialized.isReady()) {
                result.requiresPublication = true;
                result.result = std::move(materialized);
                continue;
            }
            if (materialized.status() == AuthorityStatus::FAILED) {
                result.result = std::move(materialized);
                continue;
            }
            missing.push_back(results.size() - 1);
        }
        if (missing.empty()) return results;

        if (!backend) {
            const GenerationFailure failure =
                makeFailure(GenerationFailureCode::BACKEND_UNAVAILABLE,
                            "Terrain inference backend is unavailable", true);
            for (const size_t index : missing)
                results[index].result = PageResult::failed(failure);
            return results;
        }

        std::vector<TerrainPageKey> missingKeys;
        missingKeys.reserve(missing.size());
        for (const size_t index : missing)
            missingKeys.push_back(results[index].key);

        AuthorityResult<std::vector<TerrainAuthorityPage>> inferred =
            AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "Terrain inference backend did not return a page group", true));
        try {
            if (missingKeys.size() == 1) {
                AuthorityResult<TerrainAuthorityPage> one =
                    backend->inferPageForRequest(identity, missingKeys.front(), priority);
                if (one.isReady()) {
                    std::vector<TerrainAuthorityPage> pages;
                    pages.push_back(std::move(*one.value()));
                    inferred =
                        AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(pages));
                } else {
                    inferred = one.status() == AuthorityStatus::DEFERRED
                                   ? AuthorityResult<std::vector<TerrainAuthorityPage>>::deferred(
                                         *one.failure())
                                   : AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                                         *one.failure());
                }
            } else {
                inferred = backend->inferPagesForRequest(identity, missingKeys, priority);
            }
        } catch (const std::exception& exception) {
            inferred = AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                std::string("Terrain inference backend batch threw: ") + exception.what(), true));
        } catch (...) {
            inferred = AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "Terrain inference backend batch threw an unknown exception", true));
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            metrics.builds += missingKeys.size();
            if (missingKeys.size() > 1) {
                ++metrics.batches;
                metrics.batchedPages += missingKeys.size();
            }
        }
        if (!inferred.isReady()) {
            const GenerationFailure failure =
                inferred.failure()
                    ? *inferred.failure()
                    : makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                  "Terrain inference backend batch failed without a reason", true);
            for (const size_t index : missing) {
                results[index].result = inferred.status() == AuthorityStatus::DEFERRED
                                            ? PageResult::deferred(failure)
                                            : PageResult::failed(failure);
            }
            return results;
        }
        if (inferred.value()->size() != missing.size()) {
            const GenerationFailure failure =
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "Terrain inference backend returned an incompatible page group", true);
            for (const size_t index : missing)
                results[index].result = PageResult::failed(failure);
            return results;
        }
        std::vector<TerrainAuthorityPage> pages = std::move(*inferred.value());
        for (size_t offset = 0; offset < missing.size(); ++offset) {
            PageBuildResult& result = results[missing[offset]];
            TerrainAuthorityPage& page = pages[offset];
            if (!page.valid() || page.key != result.key || !page.matches(identity)) {
                result.result = PageResult::failed(makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    "Terrain inference backend returned a malformed page in its group", true));
                continue;
            }
            result.requiresPublication = true;
            result.result =
                PageResult::ready(std::make_shared<const TerrainAuthorityPage>(std::move(page)));
        }
        return results;
    }

    CoarseResult inferCoarseSpawn(CoarseSpawnRegion region, AuthorityRequestPriority priority) {
        if (!backend) {
            return CoarseResult::failed(makeFailure(GenerationFailureCode::BACKEND_UNAVAILABLE,
                                                    "Terrain inference backend is unavailable",
                                                    true));
        }
        AuthorityResult<CoarseSpawnGrid> inferred = CoarseResult::failed(
            makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                        "Terrain inference backend did not return coarse spawn data", true));
        try {
            inferred = backend->inferCoarseSpawnGridForRequest(identity, region, priority);
        } catch (const std::exception& exception) {
            return CoarseResult::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                std::string("Terrain coarse spawn inference threw: ") + exception.what(), true));
        } catch (...) {
            return CoarseResult::failed(
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "Terrain coarse spawn inference threw an unknown exception", true));
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.builds;
        }
        if (!inferred.isReady()) {
            return inferred.status() == AuthorityStatus::DEFERRED
                       ? CoarseResult::deferred(*inferred.failure())
                       : CoarseResult::failed(*inferred.failure());
        }
        if (!inferred.value()->valid() || inferred.value()->region != region) {
            return CoarseResult::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                "Terrain inference backend returned malformed coarse spawn data", true));
        }
        return CoarseResult::ready(std::move(*inferred.value()));
    }

    GridResult inferTransientFinalGrid(NativeRect region, AuthorityRequestPriority priority) {
        const AuthorityResult<PhysicalTerrainGrid> loaded = transientStore.loadGrid(region);
        if (loaded.isReady()) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++metrics.transientDiskLoads;
            }
            return GridResult::ready(
                std::make_shared<const PhysicalTerrainGrid>(std::move(*loaded.value())));
        }
        bool repair = false;
        if (loaded.status() == AuthorityStatus::FAILED) {
            const GenerationFailure failure =
                loaded.failure()
                    ? *loaded.failure()
                    : makeFailure(GenerationFailureCode::IO_ERROR,
                                  "Transient terrain authority load returned no failure", true);
            if (failure.code == GenerationFailureCode::CORRUPT_PAGE) {
                repair = true;
            } else {
                return GridResult::failed(failure);
            }
        }
        if (!backend) {
            return GridResult::failed(makeFailure(GenerationFailureCode::BACKEND_UNAVAILABLE,
                                                  "Terrain inference backend is unavailable",
                                                  true));
        }
        AuthorityResult<PhysicalTerrainGrid> inferred =
            AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                "Terrain inference backend did not return a transient final rectangle", true));
        try {
            inferred = backend->inferFinalNativeGridForRequest(identity, region, priority);
        } catch (const std::exception& exception) {
            return GridResult::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                std::string("Terrain transient final inference threw: ") + exception.what(), true));
        } catch (...) {
            return GridResult::failed(
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "Terrain transient final inference threw an unknown exception", true));
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.builds;
        }
        if (!inferred.isReady()) {
            const GenerationFailure failure =
                inferred.failure()
                    ? *inferred.failure()
                    : makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                  "Terrain transient final inference returned no failure", true);
            return inferred.status() == AuthorityStatus::DEFERRED ? GridResult::deferred(failure)
                                                                  : GridResult::failed(failure);
        }
        if (!inferred.value()->valid() || inferred.value()->region != region) {
            return GridResult::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                "Terrain inference backend returned a malformed transient final rectangle", true));
        }
        // Exact spawn and protected-handoff rasters are immutable learned
        // authority. Persist those bounded owner inputs so a finalized world
        // can restart without reconstructing the same coarse, latent, and
        // decoder windows. Lower-priority visible and speculative rectangles
        // remain memory-only to avoid unbounded background disk growth.
        if (priority <= AuthorityRequestPriority::PROTECTED_HANDOFF) {
            const AuthorityResult<bool> published = transientStore.writeGrid(*inferred.value());
            if (!published.isReady()) {
                return GridResult::failed(
                    published.failure()
                        ? *published.failure()
                        : makeFailure(GenerationFailureCode::IO_ERROR,
                                      "Transient terrain authority publication returned no failure",
                                      true));
            }
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.transientPublicationWrites;
            if (repair) ++metrics.transientRepairs;
        }
        return GridResult::ready(
            std::make_shared<const PhysicalTerrainGrid>(std::move(*inferred.value())));
    }

    void publicationLoop() {
        for (;;) {
            PublicationTask task;
            {
                std::unique_lock lock(mutex);
                publicationCondition.wait(
                    lock, [this] { return publicationStopping || !publicationQueue.empty(); });
                if (publicationQueue.empty() && publicationStopping) return;
                const auto selected = std::min_element(
                    publicationQueue.begin(), publicationQueue.end(),
                    [this](const PublicationTask& left, const PublicationTask& right) {
                        return requestBeforeLocked(left.priority, left.handoffEpoch, left.sequence,
                                                   right.priority, right.handoffEpoch,
                                                   right.sequence);
                    });
                task = std::move(*selected);
                publicationQueue.erase(selected);
                ++activePublications;
                metrics.peakConcurrentPublications =
                    std::max(metrics.peakConcurrentPublications, activePublications);
            }

            PageResult result = PageResult::failed(
                makeFailure(GenerationFailureCode::IO_ERROR,
                            "Terrain authority publication did not complete", true));
            try {
                const AuthorityResult<bool> persisted = store.writePage(*task.page);
                if (persisted.isReady()) {
                    result = PageResult::ready(std::move(task.page));
                } else {
                    result = persisted.status() == AuthorityStatus::DEFERRED
                                 ? PageResult::deferred(*persisted.failure())
                                 : PageResult::failed(*persisted.failure());
                }
            } catch (const std::exception& exception) {
                result = PageResult::failed(makeFailure(
                    GenerationFailureCode::IO_ERROR,
                    std::string("Terrain authority publication threw: ") + exception.what(), true));
            } catch (...) {
                result = PageResult::failed(
                    makeFailure(GenerationFailureCode::IO_ERROR,
                                "Terrain authority publication threw an unknown exception", true));
            }

            {
                std::lock_guard lock(mutex);
                if (activePublications == 0) std::terminate();
                --activePublications;
                const auto found = flights.find(task.key);
                if (found != flights.end() && found->second == task.flight) {
                    if (result.isReady()) {
                        insert(*result.value(), task.priority);
                        ++metrics.publicationWrites;
                        if (task.repair) ++metrics.repairs;
                        flights.erase(found);
                    } else {
                        task.flight->publishing = false;
                        task.flight->result = std::move(result);
                        task.flight->done = true;
                    }
                    ++metrics.completionGeneration;
                }
            }
            condition.notify_all();
        }
    }

    void coordinatorLoop() {
        for (;;) {
            enum class WorkKind : uint8_t {
                Page,
                CoarseSpawn,
                TransientGrid,
            };
            WorkKind kind = WorkKind::Page;
            std::vector<PageBuildWork> pageWork;
            CoarseSpawnRegion coarseRegion;
            std::shared_ptr<CoarseSpawnFlight> coarseFlight;
            NativeRect transientRegion;
            std::shared_ptr<TransientGridFlight> transientFlight;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [this] {
                    return stopping || !queue.empty() || !coarseSpawnQueue.empty() ||
                           !transientGridQueue.empty();
                });
                if (stopping) return;
                const auto pageSelected =
                    queue.empty()
                        ? queue.end()
                        : std::min_element(queue.begin(), queue.end(),
                                           [this](TerrainPageKey left, TerrainPageKey right) {
                                               const auto leftFlight = flights.find(left);
                                               const auto rightFlight = flights.find(right);
                                               if (leftFlight == flights.end()) return false;
                                               if (rightFlight == flights.end()) return true;
                                               return requestBeforeLocked(
                                                   leftFlight->second->priority,
                                                   leftFlight->second->handoffEpoch,
                                                   leftFlight->second->sequence,
                                                   rightFlight->second->priority,
                                                   rightFlight->second->handoffEpoch,
                                                   rightFlight->second->sequence);
                                           });
                const auto coarseSelected =
                    coarseSpawnQueue.empty()
                        ? coarseSpawnQueue.end()
                        : std::min_element(
                              coarseSpawnQueue.begin(), coarseSpawnQueue.end(),
                              [this](CoarseSpawnRegion left, CoarseSpawnRegion right) {
                                  const auto leftFlight = coarseSpawnFlights.find(left);
                                  const auto rightFlight = coarseSpawnFlights.find(right);
                                  if (leftFlight == coarseSpawnFlights.end()) return false;
                                  if (rightFlight == coarseSpawnFlights.end()) return true;
                                  return requestBeforeLocked(leftFlight->second->priority,
                                                             leftFlight->second->handoffEpoch,
                                                             leftFlight->second->sequence,
                                                             rightFlight->second->priority,
                                                             rightFlight->second->handoffEpoch,
                                                             rightFlight->second->sequence);
                              });
                const auto transientSelected =
                    transientGridQueue.empty()
                        ? transientGridQueue.end()
                        : std::min_element(
                              transientGridQueue.begin(), transientGridQueue.end(),
                              [this](NativeRect left, NativeRect right) {
                                  const auto leftFlight = transientGridFlights.find(left);
                                  const auto rightFlight = transientGridFlights.find(right);
                                  if (leftFlight == transientGridFlights.end()) return false;
                                  if (rightFlight == transientGridFlights.end()) return true;
                                  return requestBeforeLocked(leftFlight->second->priority,
                                                             leftFlight->second->handoffEpoch,
                                                             leftFlight->second->sequence,
                                                             rightFlight->second->priority,
                                                             rightFlight->second->handoffEpoch,
                                                             rightFlight->second->sequence);
                              });

                bool selected = false;
                AuthorityRequestPriority selectedPriority =
                    AuthorityRequestPriority::SPECULATIVE_PREFETCH;
                ProtectedHandoffEpoch selectedEpoch;
                uint64_t selectedSequence = 0;
                const auto consider =
                    [&](WorkKind candidateKind, AuthorityRequestPriority candidatePriority,
                        ProtectedHandoffEpoch candidateEpoch, uint64_t candidateSequence) {
                        if (!selected || requestBeforeLocked(candidatePriority, candidateEpoch,
                                                             candidateSequence, selectedPriority,
                                                             selectedEpoch, selectedSequence)) {
                            selected = true;
                            kind = candidateKind;
                            selectedPriority = candidatePriority;
                            selectedEpoch = candidateEpoch;
                            selectedSequence = candidateSequence;
                        }
                    };
                if (pageSelected != queue.end()) {
                    const auto flight = flights.find(*pageSelected);
                    if (flight != flights.end())
                        consider(WorkKind::Page, flight->second->priority,
                                 flight->second->handoffEpoch, flight->second->sequence);
                }
                if (coarseSelected != coarseSpawnQueue.end()) {
                    const auto flight = coarseSpawnFlights.find(*coarseSelected);
                    if (flight != coarseSpawnFlights.end())
                        consider(WorkKind::CoarseSpawn, flight->second->priority,
                                 flight->second->handoffEpoch, flight->second->sequence);
                }
                if (transientSelected != transientGridQueue.end()) {
                    const auto flight = transientGridFlights.find(*transientSelected);
                    if (flight != transientGridFlights.end())
                        consider(WorkKind::TransientGrid, flight->second->priority,
                                 flight->second->handoffEpoch, flight->second->sequence);
                }
                if (!selected) continue;

                if (kind == WorkKind::CoarseSpawn) {
                    kind = WorkKind::CoarseSpawn;
                    coarseRegion = *coarseSelected;
                    coarseSpawnQueue.erase(coarseSelected);
                    const auto found = coarseSpawnFlights.find(coarseRegion);
                    if (found == coarseSpawnFlights.end()) continue;
                    coarseFlight = found->second;
                    coarseFlight->started = true;
                } else if (kind == WorkKind::TransientGrid) {
                    transientRegion = *transientSelected;
                    transientGridQueue.erase(transientSelected);
                    const auto found = transientGridFlights.find(transientRegion);
                    if (found == transientGridFlights.end()) continue;
                    transientFlight = found->second;
                    transientFlight->started = true;
                } else {
                    const TerrainPageKey key = *pageSelected;
                    queue.erase(pageSelected);
                    const auto found = flights.find(key);
                    if (found == flights.end()) continue;
                    pageWork.push_back({.key = key, .flight = found->second});
                    pageWork.back().flight->started = true;

                    // Admit at most one fixed four-page, same-priority,
                    // same-quality group in sequence order. The coordinator
                    // returns to global selection after that model call, so a
                    // newly arrived coarse request, transient rectangle, or
                    // higher-priority page waits behind no more than this one
                    // group.
                    const AuthorityRequestPriority groupPriority =
                        pageWork.front().flight->priority;
                    const ProtectedHandoffEpoch groupEpoch = pageWork.front().flight->handoffEpoch;
                    const AuthorityQuality groupQuality = pageWork.front().key.quality;
                    while (pageWork.size() < MAXIMUM_COORDINATOR_PAGE_GROUP_PAGES) {
                        const auto nextPage =
                            queue.empty() ? queue.end()
                                          : std::min_element(
                                                queue.begin(), queue.end(),
                                                [this](TerrainPageKey left, TerrainPageKey right) {
                                                    const auto leftFlight = flights.find(left);
                                                    const auto rightFlight = flights.find(right);
                                                    if (leftFlight == flights.end()) return false;
                                                    if (rightFlight == flights.end()) return true;
                                                    return requestBeforeLocked(
                                                        leftFlight->second->priority,
                                                        leftFlight->second->handoffEpoch,
                                                        leftFlight->second->sequence,
                                                        rightFlight->second->priority,
                                                        rightFlight->second->handoffEpoch,
                                                        rightFlight->second->sequence);
                                                });
                        if (nextPage == queue.end()) break;
                        const auto nextFlight = flights.find(*nextPage);
                        if (nextFlight == flights.end()) break;
                        const auto nextCoarse =
                            coarseSpawnQueue.empty()
                                ? coarseSpawnQueue.end()
                                : std::min_element(
                                      coarseSpawnQueue.begin(), coarseSpawnQueue.end(),
                                      [this](CoarseSpawnRegion left, CoarseSpawnRegion right) {
                                          const auto leftFlight = coarseSpawnFlights.find(left);
                                          const auto rightFlight = coarseSpawnFlights.find(right);
                                          if (leftFlight == coarseSpawnFlights.end()) return false;
                                          if (rightFlight == coarseSpawnFlights.end()) return true;
                                          return requestBeforeLocked(
                                              leftFlight->second->priority,
                                              leftFlight->second->handoffEpoch,
                                              leftFlight->second->sequence,
                                              rightFlight->second->priority,
                                              rightFlight->second->handoffEpoch,
                                              rightFlight->second->sequence);
                                      });
                        if (nextCoarse != coarseSpawnQueue.end()) {
                            const auto coarseFlight = coarseSpawnFlights.find(*nextCoarse);
                            if (coarseFlight != coarseSpawnFlights.end() &&
                                requestBeforeLocked(coarseFlight->second->priority,
                                                    coarseFlight->second->handoffEpoch,
                                                    coarseFlight->second->sequence,
                                                    nextFlight->second->priority,
                                                    nextFlight->second->handoffEpoch,
                                                    nextFlight->second->sequence)) {
                                break;
                            }
                        }
                        if (transientSelected != transientGridQueue.end()) {
                            const auto gridFlight = transientGridFlights.find(*transientSelected);
                            if (gridFlight != transientGridFlights.end() &&
                                requestBeforeLocked(
                                    gridFlight->second->priority, gridFlight->second->handoffEpoch,
                                    gridFlight->second->sequence, nextFlight->second->priority,
                                    nextFlight->second->handoffEpoch,
                                    nextFlight->second->sequence)) {
                                break;
                            }
                        }
                        if (nextFlight->second->priority != groupPriority ||
                            nextFlight->second->handoffEpoch != groupEpoch ||
                            nextPage->quality != groupQuality) {
                            break;
                        }
                        const TerrainPageKey groupedKey = *nextPage;
                        const std::shared_ptr<Flight> groupedFlight = nextFlight->second;
                        queue.erase(nextPage);
                        groupedFlight->started = true;
                        pageWork.push_back({.key = groupedKey, .flight = groupedFlight});
                    }
                }
                activeBuilds = 1;
            }

            if (kind == WorkKind::CoarseSpawn) {
                CoarseResult result = CoarseResult::failed(makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    "Terrain authority coordinator coarse spawn build did not complete", true));
                try {
                    result = inferCoarseSpawn(coarseRegion, coarseFlight->priority);
                } catch (const std::exception& exception) {
                    result = CoarseResult::failed(
                        makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                    std::string("Terrain authority coarse spawn build threw: ") +
                                        exception.what(),
                                    true));
                } catch (...) {
                    result = CoarseResult::failed(makeFailure(
                        GenerationFailureCode::INFERENCE_FAILED,
                        "Terrain authority coarse spawn build threw an unknown exception", true));
                }

                {
                    std::lock_guard lock(mutex);
                    activeBuilds = 0;
                    const auto found = coarseSpawnFlights.find(coarseRegion);
                    if (found != coarseSpawnFlights.end() && found->second == coarseFlight) {
                        if (result.isReady()) {
                            insertCoarseSpawnGrid(std::move(*result.value()));
                            coarseSpawnFlights.erase(found);
                        } else {
                            coarseFlight->result = std::move(result);
                            coarseFlight->done = true;
                        }
                        ++metrics.completionGeneration;
                    }
                }
                condition.notify_all();
                continue;
            }

            if (kind == WorkKind::TransientGrid) {
                GridResult result = GridResult::failed(
                    makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                "Terrain authority transient final build did not complete", true));
                try {
                    result = inferTransientFinalGrid(transientRegion, transientFlight->priority);
                } catch (const std::exception& exception) {
                    result = GridResult::failed(
                        makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                    std::string("Terrain authority transient final build threw: ") +
                                        exception.what(),
                                    true));
                } catch (...) {
                    result = GridResult::failed(makeFailure(
                        GenerationFailureCode::INFERENCE_FAILED,
                        "Terrain authority transient final build threw an unknown exception",
                        true));
                }

                {
                    std::lock_guard lock(mutex);
                    activeBuilds = 0;
                    const auto found = transientGridFlights.find(transientRegion);
                    if (found != transientGridFlights.end() && found->second == transientFlight) {
                        if (result.isReady()) {
                            if (insertTransientGrid(*result.value(), transientFlight->priority)) {
                                transientGridFlights.erase(found);
                            } else {
                                // A caller configured with a cache budget below
                                // the bounded rectangle size still receives the
                                // completed single-flight result exactly once.
                                transientFlight->result = std::move(result);
                                transientFlight->done = true;
                            }
                        } else {
                            transientFlight->result = std::move(result);
                            transientFlight->done = true;
                        }
                        ++metrics.completionGeneration;
                    }
                }
                condition.notify_all();
                continue;
            }

            std::vector<TerrainPageKey> pageKeys;
            pageKeys.reserve(pageWork.size());
            for (const PageBuildWork& work : pageWork)
                pageKeys.push_back(work.key);
            std::vector<PageBuildResult> results;
            try {
                results = inferOrLoadBatch(pageKeys, pageWork.front().flight->priority);
            } catch (const std::exception& exception) {
                const GenerationFailure failure = makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    std::string("Terrain authority batch build threw: ") + exception.what(), true);
                results.reserve(pageKeys.size());
                for (const TerrainPageKey key : pageKeys)
                    results.push_back({.key = key, .result = PageResult::failed(failure)});
            } catch (...) {
                const GenerationFailure failure =
                    makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                "Terrain authority batch build threw an unknown exception", true);
                results.reserve(pageKeys.size());
                for (const TerrainPageKey key : pageKeys)
                    results.push_back({.key = key, .result = PageResult::failed(failure)});
            }
            if (results.size() != pageWork.size()) {
                const GenerationFailure failure = makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    "Terrain authority batch build returned an incompatible result count", true);
                results.clear();
                results.reserve(pageKeys.size());
                for (const TerrainPageKey key : pageKeys)
                    results.push_back({.key = key, .result = PageResult::failed(failure)});
            }

            bool queuedPublication = false;
            {
                std::lock_guard lock(mutex);
                activeBuilds = 0;
                for (size_t index = 0; index < pageWork.size(); ++index) {
                    const PageBuildWork& work = pageWork[index];
                    PageBuildResult& result = results[index];
                    const auto found = flights.find(work.key);
                    if (found == flights.end() || found->second != work.flight) continue;
                    if (result.result.isReady()) {
                        if (result.requiresPublication) {
                            work.flight->publishing = true;
                            publicationQueue.push_back({
                                .key = work.key,
                                .flight = work.flight,
                                .page = *result.result.value(),
                                .repair = result.repair,
                                .priority = work.flight->priority,
                                .handoffEpoch = work.flight->handoffEpoch,
                                .sequence = work.flight->sequence,
                            });
                            ensurePublicationWorkersStartedLocked();
                            queuedPublication = true;
                        } else {
                            insert(*result.result.value(), work.flight->priority);
                            flights.erase(found);
                            ++metrics.completionGeneration;
                        }
                    } else {
                        work.flight->result = std::move(result.result);
                        work.flight->done = true;
                        ++metrics.completionGeneration;
                    }
                }
            }
            condition.notify_all();
            if (queuedPublication) publicationCondition.notify_all();
        }
    }

    GenerationIdentity identity;
    TerrainPageStore store;
    TransientTerrainGridStore transientStore;
    std::shared_ptr<TerrainInferenceBackend> backend;
    TerrainAuthorityCacheConfig config;

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable publicationCondition;
    std::map<TerrainPageKey, CacheEntry> cache;
    std::list<TerrainPageKey> recency;
    size_t cacheBytes = 0;
    std::map<TerrainPageKey, std::shared_ptr<Flight>> flights;
    std::vector<TerrainPageKey> queue;
    std::map<CoarseSpawnRegion, CoarseSpawnCacheEntry> coarseSpawnCache;
    std::map<CoarseSpawnRegion, std::shared_ptr<CoarseSpawnFlight>> coarseSpawnFlights;
    std::vector<CoarseSpawnRegion> coarseSpawnQueue;
    std::map<NativeRect, TransientGridCacheEntry> transientGridCache;
    size_t transientGridCacheBytes = 0;
    std::map<NativeRect, std::shared_ptr<TransientGridFlight>> transientGridFlights;
    std::vector<NativeRect> transientGridQueue;
    std::vector<PublicationTask> publicationQueue;
    size_t activeBuilds = 0;
    size_t activePublications = 0;
    ProtectedHandoffEpoch currentProtectedHandoffEpoch;
    uint64_t nextSequence = 0;
    uint64_t coarseSpawnRecency = 0;
    uint64_t transientGridRecency = 0;
    bool stopping = false;
    bool publicationStopping = false;
    std::thread coordinator;
    std::vector<std::thread> publicationWorkers;
    TerrainAuthorityCacheMetrics metrics;
};

CachedTerrainAuthority::CachedTerrainAuthority(GenerationIdentity identity,
                                               std::filesystem::path storeRoot,
                                               std::shared_ptr<TerrainInferenceBackend> backend,
                                               TerrainAuthorityCacheConfig config)
    : impl_(std::make_unique<Impl>(std::move(identity), std::move(storeRoot), std::move(backend),
                                   config)) {}

CachedTerrainAuthority::~CachedTerrainAuthority() = default;

AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
CachedTerrainAuthority::preparePage(TerrainPageKey key, AuthorityRequestPriority priority) {
    return preparePage(key, priority, {});
}

AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
CachedTerrainAuthority::preparePage(TerrainPageKey key, AuthorityRequestPriority priority,
                                    ProtectedHandoffEpoch epoch) {
    using PageResult = Impl::PageResult;
    if (!impl_->identity.valid() || !validQuality(key.quality) || !validRequestPriority(priority) ||
        !Impl::validHandoffEpoch(key, priority, epoch)) {
        const std::string detail =
            "Terrain authority request is invalid: identity=" +
            std::to_string(static_cast<unsigned>(impl_->identity.valid())) +
            " quality=" + std::to_string(static_cast<unsigned>(key.quality)) +
            " priority=" + std::to_string(static_cast<unsigned>(priority)) +
            " epoch=" + std::to_string(epoch.value) + " row=" + std::to_string(key.coordinate.row) +
            " column=" + std::to_string(key.coordinate.column);
        return PageResult::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST, detail, false));
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->ensureCoordinatorStartedLocked();
        impl_->observeHandoffEpochLocked(epoch);
        auto cached = impl_->cache.find(key);
        if (cached != impl_->cache.end()) {
            ++impl_->metrics.hits;
            const auto page = cached->second.page;
            cached->second.priority =
                std::min(cached->second.priority, Impl::effectivePriority(key, priority));
            impl_->touch(cached);
            return PageResult::ready(page);
        }
        if (impl_->staleHandoffEpochLocked(epoch)) {
            ++impl_->metrics.deferredRequests;
            ++impl_->metrics.staleProtectedHandoffDeferrals;
            return PageResult::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                "Terrain authority protected handoff belongs to a stale camera epoch", true));
        }
        auto active = impl_->flights.find(key);
        if (active != impl_->flights.end()) {
            if (active->second->done) {
                PageResult completed = std::move(active->second->result);
                impl_->flights.erase(active);
                return completed;
            }
            ++impl_->metrics.singleFlightDeferrals;
            if (!active->second->started) {
                if (priority < active->second->priority) active->second->priority = priority;
                if (active->second->priority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
                    epoch.valid() && epoch.value > active->second->handoffEpoch.value) {
                    active->second->handoffEpoch = epoch;
                }
            }
            return PageResult::deferred(makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                                                    "Terrain authority page is queued or building",
                                                    true));
        }
        bool hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
        if (!hasTotalCapacity) {
            const size_t occupied = impl_->outstandingRequestCountLocked();
            const size_t shortage = occupied + 1 - impl_->config.maximumOutstandingRequests;
            if (impl_->preemptForPriorityLocked(shortage, Impl::effectivePriority(key, priority),
                                                epoch)) {
                hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
            }
        }
        const bool lowPriorityRequest = Impl::isLowPriorityRequest(key, priority);
        const bool hasLowPriorityCapacity =
            !lowPriorityRequest || impl_->hasLowPriorityCapacityLocked(1);
        const bool visibleOrLowerRequest = Impl::isVisibleOrLowerRequest(key, priority);
        const bool hasVisibleOrLowerCapacity =
            !visibleOrLowerRequest || impl_->hasVisibleOrLowerCapacityLocked(1);
        if (!hasTotalCapacity || !hasLowPriorityCapacity || !hasVisibleOrLowerCapacity) {
            ++impl_->metrics.deferredRequests;
            if (!hasLowPriorityCapacity) ++impl_->metrics.lowPriorityDeferredRequests;
            if (!hasVisibleOrLowerCapacity) ++impl_->metrics.visibleOrLowerDeferredRequests;
            return PageResult::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                Impl::capacityFailureMessage(hasTotalCapacity, hasLowPriorityCapacity), true));
        }

        ++impl_->metrics.misses;
        auto flight = std::make_shared<Impl::Flight>();
        flight->priority = priority;
        flight->handoffEpoch = epoch;
        flight->sequence = impl_->nextSequence++;
        impl_->flights.emplace(key, flight);
        impl_->queue.push_back(key);
    }
    impl_->condition.notify_one();
    return PageResult::deferred(makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                                            "Terrain authority page was enqueued", true));
}

AuthorityResult<bool> CachedTerrainAuthority::preparePages(std::span<const TerrainPageKey> keys,
                                                           AuthorityRequestPriority priority) {
    return preparePages(keys, priority, {});
}

AuthorityResult<bool> CachedTerrainAuthority::preparePages(std::span<const TerrainPageKey> keys,
                                                           AuthorityRequestPriority priority,
                                                           ProtectedHandoffEpoch epoch) {
    if (!impl_->identity.valid() || !validRequestPriority(priority) || keys.empty() ||
        keys.size() > MAXIMUM_AUTHORITY_QUEUED_REQUESTS ||
        !Impl::validHandoffEpoch(priority, epoch)) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority page closure or handoff epoch is invalid", false));
    }

    std::vector<TerrainPageKey> canonical(keys.begin(), keys.end());
    std::sort(canonical.begin(), canonical.end());
    if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end() ||
        std::any_of(canonical.begin(), canonical.end(), [epoch](TerrainPageKey key) {
            return !validQuality(key.quality) ||
                   (epoch.valid() && key.quality != AuthorityQuality::FINAL);
        })) {
        return AuthorityResult<bool>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority page closure has invalid or duplicate keys", false));
    }

    bool enqueued = false;
    bool hasOutstandingFlight = false;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->ensureCoordinatorStartedLocked();
        impl_->observeHandoffEpochLocked(epoch);
        if (impl_->staleHandoffEpochLocked(epoch)) {
            ++impl_->metrics.deferredRequests;
            ++impl_->metrics.staleProtectedHandoffDeferrals;
            return AuthorityResult<bool>::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                "Terrain authority protected closure belongs to a stale camera epoch", true));
        }

        std::vector<TerrainPageKey> missingKeys;
        missingKeys.reserve(canonical.size());
        std::vector<std::pair<TerrainPageKey, std::shared_ptr<Impl::Flight>>> priorityPromotions;
        priorityPromotions.reserve(canonical.size());
        for (const TerrainPageKey key : canonical) {
            if (auto cached = impl_->cache.find(key); cached != impl_->cache.end()) {
                ++impl_->metrics.hits;
                cached->second.priority =
                    std::min(cached->second.priority, Impl::effectivePriority(key, priority));
                impl_->touch(cached);
                continue;
            }
            const auto active = impl_->flights.find(key);
            if (active == impl_->flights.end()) {
                missingKeys.push_back(key);
                continue;
            }
            if (active->second->done) {
                Impl::PageResult completed = std::move(active->second->result);
                impl_->flights.erase(active);
                if (completed.isReady()) {
                    impl_->insert(*completed.value(), priority);
                    if (!impl_->cache.contains(key)) missingKeys.push_back(key);
                    continue;
                }
                const GenerationFailure failure =
                    completed.failure()
                        ? *completed.failure()
                        : makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                      "Terrain authority page closure completed without a result",
                                      true);
                return completed.status() == AuthorityStatus::DEFERRED
                           ? AuthorityResult<bool>::deferred(failure)
                           : AuthorityResult<bool>::failed(failure);
            }
            hasOutstandingFlight = true;
            priorityPromotions.emplace_back(key, active->second);
        }

        const auto prospectiveReservationCounts = [&] {
            size_t low = impl_->lowPriorityOutstandingRequestCountLocked();
            size_t visibleOrLower = impl_->visibleOrLowerOutstandingRequestCountLocked();
            for (const auto& [key, flight] : priorityPromotions) {
                if (flight->started || priority >= flight->priority) continue;
                const bool wasLow = Impl::isLowPriorityRequest(key, flight->priority);
                const bool willBeLow = Impl::isLowPriorityRequest(key, priority);
                if (wasLow && !willBeLow) {
                    --low;
                } else if (!wasLow && willBeLow) {
                    ++low;
                }
                const bool wasVisibleOrLower = Impl::isVisibleOrLowerRequest(key, flight->priority);
                const bool willBeVisibleOrLower = Impl::isVisibleOrLowerRequest(key, priority);
                if (wasVisibleOrLower && !willBeVisibleOrLower) {
                    --visibleOrLower;
                } else if (!wasVisibleOrLower && willBeVisibleOrLower) {
                    ++visibleOrLower;
                }
            }
            return std::pair{low, visibleOrLower};
        };
        const size_t additionalLowPriorityRequests = static_cast<size_t>(
            std::count_if(missingKeys.begin(), missingKeys.end(), [priority](TerrainPageKey key) {
                return Impl::isLowPriorityRequest(key, priority);
            }));
        const size_t additionalVisibleOrLowerRequests = static_cast<size_t>(
            std::count_if(missingKeys.begin(), missingKeys.end(), [priority](TerrainPageKey key) {
                return Impl::isVisibleOrLowerRequest(key, priority);
            }));
        bool hasTotalCapacity = impl_->hasTotalCapacityLocked(missingKeys.size());
        if (!hasTotalCapacity) {
            const size_t occupied = impl_->outstandingRequestCountLocked();
            const size_t shortage =
                occupied + missingKeys.size() - impl_->config.maximumOutstandingRequests;
            AuthorityRequestPriority incomingPriority = priority;
            for (const TerrainPageKey key : missingKeys) {
                incomingPriority =
                    std::max(incomingPriority, Impl::effectivePriority(key, priority));
            }
            if (impl_->preemptForPriorityLocked(shortage, incomingPriority, epoch, canonical)) {
                hasTotalCapacity = impl_->hasTotalCapacityLocked(missingKeys.size());
            }
        }
        const auto [prospectiveLowPriorityRequests, prospectiveVisibleOrLowerRequests] =
            prospectiveReservationCounts();
        const bool hasLowPriorityCapacity = impl_->hasLowPriorityCapacityLocked(
            prospectiveLowPriorityRequests, additionalLowPriorityRequests);
        const bool hasVisibleOrLowerCapacity = impl_->hasVisibleOrLowerCapacityLocked(
            prospectiveVisibleOrLowerRequests, additionalVisibleOrLowerRequests);
        if (!hasTotalCapacity || !hasLowPriorityCapacity || !hasVisibleOrLowerCapacity) {
            ++impl_->metrics.deferredRequests;
            if (!hasLowPriorityCapacity) ++impl_->metrics.lowPriorityDeferredRequests;
            if (!hasVisibleOrLowerCapacity) ++impl_->metrics.visibleOrLowerDeferredRequests;
            return AuthorityResult<bool>::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                Impl::capacityFailureMessage(hasTotalCapacity, hasLowPriorityCapacity), true));
        }

        for (const auto& [key, flight] : priorityPromotions) {
            static_cast<void>(key);
            ++impl_->metrics.singleFlightDeferrals;
            if (!flight->started) {
                if (priority < flight->priority) flight->priority = priority;
                if (flight->priority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
                    epoch.valid() && epoch.value > flight->handoffEpoch.value) {
                    flight->handoffEpoch = epoch;
                }
            }
        }
        for (const TerrainPageKey key : missingKeys) {
            ++impl_->metrics.misses;
            auto flight = std::make_shared<Impl::Flight>();
            flight->priority = priority;
            flight->handoffEpoch = epoch;
            flight->sequence = impl_->nextSequence++;
            impl_->flights.emplace(key, flight);
            impl_->queue.push_back(key);
            enqueued = true;
        }
    }
    if (enqueued) impl_->condition.notify_one();
    if (enqueued || hasOutstandingFlight) {
        return AuthorityResult<bool>::deferred(
            makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                        "Terrain authority page closure is queued or building", true));
    }
    return AuthorityResult<bool>::ready(true);
}

const GenerationIdentity& CachedTerrainAuthority::generationIdentity() const noexcept {
    return impl_->identity;
}

namespace {

template <typename Destination, typename Source>
AuthorityResult<Destination> forwardFailure(const AuthorityResult<Source>& source) {
    if (source.status() == AuthorityStatus::DEFERRED)
        return AuthorityResult<Destination>::deferred(*source.failure());
    return AuthorityResult<Destination>::failed(*source.failure());
}

} // namespace

AuthorityResult<PhysicalTerrainGrid>
CachedTerrainAuthority::queryNative(NativeRect region, AuthorityQuality quality,
                                    AuthorityRequestPriority priority) {
    if (!impl_->identity.valid() || !validQuality(quality) || !validRequestPriority(priority) ||
        !region.valid() || region.height() > std::numeric_limits<size_t>::max() ||
        region.width() > std::numeric_limits<size_t>::max()) {
        return AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST, "Terrain authority query is invalid", false));
    }
    size_t sampleCount = 0;
    if (!checkedProduct(static_cast<size_t>(region.height()), static_cast<size_t>(region.width()),
                        sampleCount) ||
        sampleCount > impl_->config.maximumQuerySamples) {
        return AuthorityResult<PhysicalTerrainGrid>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority query exceeds the bounded sample count", false));
    }

    if (quality == AuthorityQuality::FINAL) {
        Impl::GridPointer containingGrid;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            const auto containing = impl_->smallestContainingTransientGrid(region);
            if (containing != impl_->transientGridCache.end()) {
                containingGrid = containing->second.grid;
                containing->second.recency = ++impl_->transientGridRecency;
                containing->second.priority = std::min(containing->second.priority, priority);
                ++impl_->metrics.hits;
            }
        }
        if (containingGrid) {
            const Impl::GridPointer crop = Impl::cropTransientGrid(*containingGrid, region);
            if (!crop) {
                return AuthorityResult<PhysicalTerrainGrid>::failed(makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    "Cached transient terrain could not satisfy a contained query", true));
            }
            return AuthorityResult<PhysicalTerrainGrid>::ready(*crop);
        }
    }

    const TerrainPageCoordinate firstPage =
        terrainPageCoordinateFor({.row = region.rowBegin, .column = region.columnBegin});
    const TerrainPageCoordinate lastPage =
        terrainPageCoordinateFor({.row = region.rowEnd - 1, .column = region.columnEnd - 1});
    const __int128 pageRows = static_cast<__int128>(lastPage.row) - firstPage.row + 1;
    const __int128 pageColumns = static_cast<__int128>(lastPage.column) - firstPage.column + 1;
    const __int128 pageCount = pageRows * pageColumns;
    if (pageCount <= 0 || pageCount > static_cast<__int128>(impl_->config.maximumQueryPages)) {
        return AuthorityResult<PhysicalTerrainGrid>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority query exceeds the bounded page count", false));
    }

    std::map<TerrainPageCoordinate, std::shared_ptr<const TerrainAuthorityPage>> pages;
    std::optional<GenerationFailure> deferredFailure;
    for (int64_t pageRow = firstPage.row; pageRow <= lastPage.row; ++pageRow) {
        for (int64_t pageColumn = firstPage.column; pageColumn <= lastPage.column; ++pageColumn) {
            const TerrainPageKey key{
                .quality = quality,
                .coordinate = {.row = pageRow, .column = pageColumn},
            };
            auto prepared = preparePage(key, priority);
            if (prepared.isReady()) {
                pages.emplace(key.coordinate, *prepared.value());
            } else if (prepared.status() == AuthorityStatus::FAILED) {
                return forwardFailure<PhysicalTerrainGrid>(prepared);
            } else if (!deferredFailure) {
                deferredFailure = *prepared.failure();
            }
            if (pageColumn == std::numeric_limits<int64_t>::max()) break;
        }
        if (pageRow == std::numeric_limits<int64_t>::max()) break;
    }
    if (deferredFailure)
        return AuthorityResult<PhysicalTerrainGrid>::deferred(std::move(*deferredFailure));

    PhysicalTerrainGrid output{
        .region = region,
        .samples = std::vector<PhysicalTerrainSample>(sampleCount),
    };
    size_t outputIndex = 0;
    for (int64_t row = region.rowBegin; row < region.rowEnd; ++row) {
        for (int64_t column = region.columnBegin; column < region.columnEnd; ++column) {
            const NativePoint point{.row = row, .column = column};
            const auto page = pages.find(terrainPageCoordinateFor(point));
            if (page == pages.end()) {
                return AuthorityResult<PhysicalTerrainGrid>::failed(
                    makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                "Terrain authority query lost a prepared page", true));
            }
            const QuantizedTerrainSample* quantized = page->second->sample(
                terrainPageLocalCoordinate(row), terrainPageLocalCoordinate(column));
            if (!quantized) {
                return AuthorityResult<PhysicalTerrainGrid>::failed(
                    makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                                "Terrain authority query found a malformed page", true));
            }
            output.samples[outputIndex++] = dequantizeTerrainSample(*quantized);
        }
    }
    return AuthorityResult<PhysicalTerrainGrid>::ready(std::move(output));
}

AuthorityResult<std::vector<PhysicalTerrainSample>>
CachedTerrainAuthority::queryNativePoints(std::span<const NativePoint> points,
                                          AuthorityQuality quality,
                                          AuthorityRequestPriority priority) {
    if (!impl_->identity.valid() || !validQuality(quality) || !validRequestPriority(priority) ||
        points.size() > impl_->config.maximumQuerySamples) {
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority point query is invalid or exceeds its bound", false));
    }
    if (quality == AuthorityQuality::FINAL && !points.empty()) {
        int64_t minimumRow = points.front().row;
        int64_t maximumRow = points.front().row;
        int64_t minimumColumn = points.front().column;
        int64_t maximumColumn = points.front().column;
        for (NativePoint point : points) {
            minimumRow = std::min(minimumRow, point.row);
            maximumRow = std::max(maximumRow, point.row);
            minimumColumn = std::min(minimumColumn, point.column);
            maximumColumn = std::max(maximumColumn, point.column);
        }
        Impl::GridPointer containingGrid;
        if (maximumRow != std::numeric_limits<int64_t>::max() &&
            maximumColumn != std::numeric_limits<int64_t>::max()) {
            const NativeRect bounds{
                .rowBegin = minimumRow,
                .columnBegin = minimumColumn,
                .rowEnd = maximumRow + 1,
                .columnEnd = maximumColumn + 1,
            };
            std::lock_guard<std::mutex> lock(impl_->mutex);
            const auto containing = impl_->smallestContainingTransientGrid(bounds);
            if (containing != impl_->transientGridCache.end()) {
                containingGrid = containing->second.grid;
                containing->second.recency = ++impl_->transientGridRecency;
                containing->second.priority = std::min(containing->second.priority, priority);
                ++impl_->metrics.hits;
            }
        }
        if (containingGrid) {
            std::vector<PhysicalTerrainSample> output;
            output.reserve(points.size());
            for (NativePoint point : points) {
                const PhysicalTerrainSample* sample =
                    containingGrid->sample(point.row, point.column);
                if (!sample) {
                    return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(
                        makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                                    "Cached transient terrain lost a contained point", true));
                }
                output.push_back(*sample);
            }
            return AuthorityResult<std::vector<PhysicalTerrainSample>>::ready(std::move(output));
        }
    }
    std::map<TerrainPageCoordinate, std::shared_ptr<const TerrainAuthorityPage>> pages;
    for (NativePoint point : points)
        pages.try_emplace(terrainPageCoordinateFor(point));
    if (pages.size() > impl_->config.maximumQueryPages) {
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Terrain authority point query exceeds the bounded page count", false));
    }
    std::optional<GenerationFailure> deferredFailure;
    for (auto& [coordinate, page] : pages) {
        auto prepared =
            preparePage(TerrainPageKey{.quality = quality, .coordinate = coordinate}, priority);
        if (prepared.isReady()) {
            page = *prepared.value();
        } else if (prepared.status() == AuthorityStatus::FAILED) {
            return forwardFailure<std::vector<PhysicalTerrainSample>>(prepared);
        } else if (!deferredFailure) {
            deferredFailure = *prepared.failure();
            const auto point = std::find_if(points.begin(), points.end(), [&](NativePoint point) {
                return terrainPageCoordinateFor(point) == coordinate;
            });
            deferredFailure->message += " at authority page row " + std::to_string(coordinate.row) +
                                        " column " + std::to_string(coordinate.column);
            if (point != points.end()) {
                deferredFailure->message += " for native row " + std::to_string(point->row) +
                                            " column " + std::to_string(point->column);
            }
        }
    }
    if (deferredFailure)
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::deferred(
            std::move(*deferredFailure));

    std::vector<PhysicalTerrainSample> output;
    output.reserve(points.size());
    for (NativePoint point : points) {
        const auto page = pages.find(terrainPageCoordinateFor(point));
        const QuantizedTerrainSample* quantized =
            page == pages.end() ? nullptr
                                : page->second->sample(terrainPageLocalCoordinate(point.row),
                                                       terrainPageLocalCoordinate(point.column));
        if (!quantized) {
            return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Terrain authority point query found a malformed page", true));
        }
        output.push_back(dequantizeTerrainSample(*quantized));
    }
    return AuthorityResult<std::vector<PhysicalTerrainSample>>::ready(std::move(output));
}

AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>
CachedTerrainAuthority::queryTransientFinalNativeGrid(NativeRect region,
                                                      AuthorityRequestPriority priority) {
    return queryTransientFinalNativeGrid(region, priority, {});
}

AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>
CachedTerrainAuthority::queryTransientFinalNativeGrid(NativeRect region,
                                                      AuthorityRequestPriority priority,
                                                      ProtectedHandoffEpoch epoch) {
    using GridResult = Impl::GridResult;
    if (!impl_->identity.valid() || !validRequestPriority(priority) || !region.valid() ||
        !Impl::validHandoffEpoch(priority, epoch) ||
        region.height() > std::numeric_limits<size_t>::max() ||
        region.width() > std::numeric_limits<size_t>::max() ||
        (region.height() != 0 &&
         region.width() > std::numeric_limits<size_t>::max() / region.height())) {
        return GridResult::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Transient final terrain query has an invalid rectangle", false));
    }
    const size_t sampleCount =
        static_cast<size_t>(region.height()) * static_cast<size_t>(region.width());
    if (sampleCount > impl_->config.maximumQuerySamples) {
        return GridResult::failed(
            makeFailure(GenerationFailureCode::INVALID_REQUEST,
                        "Transient final terrain query exceeds the bounded sample count", false));
    }

    Impl::GridPointer containingGrid;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->ensureCoordinatorStartedLocked();
        impl_->observeHandoffEpochLocked(epoch);
        const auto cached = impl_->smallestContainingTransientGrid(region);
        if (cached != impl_->transientGridCache.end()) {
            containingGrid = cached->second.grid;
            cached->second.recency = ++impl_->transientGridRecency;
            cached->second.priority = std::min(cached->second.priority, priority);
            ++impl_->metrics.hits;
        } else if (impl_->staleHandoffEpochLocked(epoch)) {
            ++impl_->metrics.deferredRequests;
            ++impl_->metrics.staleProtectedHandoffDeferrals;
            return GridResult::deferred(
                makeFailure(GenerationFailureCode::QUEUE_FULL,
                            "Transient protected handoff belongs to a stale camera epoch", true));
        }
    }
    if (containingGrid) {
        Impl::GridPointer result = containingGrid;
        if (containingGrid->region != region) {
            result = Impl::cropTransientGrid(*containingGrid, region);
            if (!result) {
                return GridResult::failed(makeFailure(
                    GenerationFailureCode::INFERENCE_FAILED,
                    "Cached transient terrain could not produce a contained rectangle", true));
            }
            std::lock_guard<std::mutex> lock(impl_->mutex);
            static_cast<void>(impl_->insertTransientGrid(result, priority));
            if (const auto inserted = impl_->transientGridCache.find(region);
                inserted != impl_->transientGridCache.end()) {
                result = inserted->second.grid;
            }
        }
        return GridResult::ready(std::move(result));
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->ensureCoordinatorStartedLocked();
        if (impl_->staleHandoffEpochLocked(epoch)) {
            ++impl_->metrics.deferredRequests;
            ++impl_->metrics.staleProtectedHandoffDeferrals;
            return GridResult::deferred(
                makeFailure(GenerationFailureCode::QUEUE_FULL,
                            "Transient protected handoff belongs to a stale camera epoch", true));
        }
        const auto active = impl_->transientGridFlights.find(region);
        if (active != impl_->transientGridFlights.end()) {
            if (active->second->done) {
                GridResult completed = std::move(active->second->result);
                impl_->transientGridFlights.erase(active);
                return completed;
            }
            ++impl_->metrics.singleFlightDeferrals;
            if (!active->second->started) {
                if (priority < active->second->priority) active->second->priority = priority;
                if (active->second->priority == AuthorityRequestPriority::PROTECTED_HANDOFF &&
                    epoch.valid() && epoch.value > active->second->handoffEpoch.value) {
                    active->second->handoffEpoch = epoch;
                }
            }
            return GridResult::deferred(
                makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                            "Transient final terrain rectangle is queued or building", true));
        }
        const auto containingFlight = impl_->smallestContainingTransientGridFlight(region);
        if (containingFlight != impl_->transientGridFlights.end()) {
            ++impl_->metrics.singleFlightDeferrals;
            if (!containingFlight->second->started) {
                if (priority < containingFlight->second->priority)
                    containingFlight->second->priority = priority;
                if (containingFlight->second->priority ==
                        AuthorityRequestPriority::PROTECTED_HANDOFF &&
                    epoch.valid() && epoch.value > containingFlight->second->handoffEpoch.value) {
                    containingFlight->second->handoffEpoch = epoch;
                }
            }
            return GridResult::deferred(makeFailure(
                GenerationFailureCode::PAGE_NOT_FOUND,
                "A containing transient final terrain rectangle is queued or building", true));
        }
        bool hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
        if (!hasTotalCapacity) {
            const size_t occupied = impl_->outstandingRequestCountLocked();
            const size_t shortage = occupied + 1 - impl_->config.maximumOutstandingRequests;
            if (impl_->preemptForPriorityLocked(shortage, priority, epoch)) {
                hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
            }
        }
        const bool lowPriorityRequest = Impl::isLowPriorityRequest(priority);
        const bool hasLowPriorityCapacity =
            !lowPriorityRequest || impl_->hasLowPriorityCapacityLocked(1);
        const bool visibleOrLowerRequest = Impl::isVisibleOrLowerRequest(priority);
        const bool hasVisibleOrLowerCapacity =
            !visibleOrLowerRequest || impl_->hasVisibleOrLowerCapacityLocked(1);
        if (!hasTotalCapacity || !hasLowPriorityCapacity || !hasVisibleOrLowerCapacity) {
            ++impl_->metrics.deferredRequests;
            if (!hasLowPriorityCapacity) ++impl_->metrics.lowPriorityDeferredRequests;
            if (!hasVisibleOrLowerCapacity) ++impl_->metrics.visibleOrLowerDeferredRequests;
            return GridResult::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                Impl::capacityFailureMessage(hasTotalCapacity, hasLowPriorityCapacity), true));
        }

        ++impl_->metrics.misses;
        auto flight = std::make_shared<Impl::TransientGridFlight>();
        flight->priority = priority;
        flight->handoffEpoch = epoch;
        flight->sequence = impl_->nextSequence++;
        impl_->transientGridFlights.emplace(region, flight);
        impl_->transientGridQueue.push_back(region);
    }
    impl_->condition.notify_one();
    return GridResult::deferred(makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                                            "Transient final terrain rectangle was enqueued",
                                            true));
}

AuthorityResult<CoarseSpawnGrid>
CachedTerrainAuthority::queryCoarseSpawnGrid(CoarseSpawnRegion region,
                                             AuthorityRequestPriority priority) {
    using CoarseResult = Impl::CoarseResult;
    if (!impl_->identity.valid() || !validRequestPriority(priority) || !region.valid() ||
        region.height() > MAXIMUM_COARSE_SPAWN_GRID_EDGE ||
        region.width() > MAXIMUM_COARSE_SPAWN_GRID_EDGE) {
        return CoarseResult::failed(makeFailure(
            GenerationFailureCode::INVALID_REQUEST,
            "Terrain coarse spawn query is invalid or exceeds its bounded region", false));
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->ensureCoordinatorStartedLocked();
        const auto cached = impl_->coarseSpawnCache.find(region);
        if (cached != impl_->coarseSpawnCache.end()) {
            ++impl_->metrics.hits;
            cached->second.recency = ++impl_->coarseSpawnRecency;
            return CoarseResult::ready(cached->second.grid);
        }
        const auto active = impl_->coarseSpawnFlights.find(region);
        if (active != impl_->coarseSpawnFlights.end()) {
            if (active->second->done) {
                CoarseResult completed = std::move(active->second->result);
                impl_->coarseSpawnFlights.erase(active);
                return completed;
            }
            ++impl_->metrics.singleFlightDeferrals;
            if (!active->second->started && priority < active->second->priority)
                active->second->priority = priority;
            return CoarseResult::deferred(
                makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                            "Terrain coarse spawn data is queued or building", true));
        }
        bool hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
        if (!hasTotalCapacity) {
            const size_t occupied = impl_->outstandingRequestCountLocked();
            const size_t shortage = occupied + 1 - impl_->config.maximumOutstandingRequests;
            if (impl_->preemptForPriorityLocked(shortage, priority)) {
                hasTotalCapacity = impl_->hasTotalCapacityLocked(1);
            }
        }
        const bool lowPriorityRequest = Impl::isLowPriorityRequest(priority);
        const bool hasLowPriorityCapacity =
            !lowPriorityRequest || impl_->hasLowPriorityCapacityLocked(1);
        const bool visibleOrLowerRequest = Impl::isVisibleOrLowerRequest(priority);
        const bool hasVisibleOrLowerCapacity =
            !visibleOrLowerRequest || impl_->hasVisibleOrLowerCapacityLocked(1);
        if (!hasTotalCapacity || !hasLowPriorityCapacity || !hasVisibleOrLowerCapacity) {
            ++impl_->metrics.deferredRequests;
            if (!hasLowPriorityCapacity) ++impl_->metrics.lowPriorityDeferredRequests;
            if (!hasVisibleOrLowerCapacity) ++impl_->metrics.visibleOrLowerDeferredRequests;
            return CoarseResult::deferred(makeFailure(
                GenerationFailureCode::QUEUE_FULL,
                Impl::capacityFailureMessage(hasTotalCapacity, hasLowPriorityCapacity), true));
        }

        ++impl_->metrics.misses;
        auto flight = std::make_shared<Impl::CoarseSpawnFlight>();
        flight->priority = priority;
        flight->sequence = impl_->nextSequence++;
        impl_->coarseSpawnFlights.emplace(region, std::move(flight));
        impl_->coarseSpawnQueue.push_back(region);
    }
    impl_->condition.notify_one();
    return CoarseResult::deferred(makeFailure(GenerationFailureCode::PAGE_NOT_FOUND,
                                              "Terrain coarse spawn data was enqueued", true));
}

TerrainAuthorityCacheMetrics CachedTerrainAuthority::cacheMetrics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    TerrainAuthorityCacheMetrics result = impl_->metrics;
    result.entries = impl_->cache.size() + impl_->transientGridCache.size();
    result.bytes = impl_->cacheBytes + impl_->transientGridCacheBytes;
    result.activeBuilds = impl_->activeBuilds;
    result.queuedBuilds =
        impl_->queue.size() + impl_->coarseSpawnQueue.size() + impl_->transientGridQueue.size();
    result.lowPriorityOutstandingRequests = impl_->lowPriorityOutstandingRequestCountLocked();
    result.visibleOrLowerOutstandingRequests = impl_->visibleOrLowerOutstandingRequestCountLocked();
    result.activePublications = impl_->activePublications;
    result.queuedPublications = impl_->publicationQueue.size();
    return result;
}

std::shared_ptr<void>
CachedTerrainAuthority::retainWindowGraph(const LearnedAuthorityGraph& graph) {
    return impl_->backend ? impl_->backend->retainWindowGraph(graph) : nullptr;
}

GenerationFailureException::GenerationFailureException(AuthorityStatus status,
                                                       GenerationFailure failure)
    : std::runtime_error(failure.message.empty() ? "World generation authority failed"
                                                 : failure.message)
    , status_(status)
    , failure_(std::move(failure)) {}

class WorldGenerationContext::FailureLatch {
public:
    mutable std::mutex mutex;
    std::optional<GenerationFailure> failure;
};

class WorldGenerationContext::HydrologyPreparationRegistry {
public:
    mutable std::mutex mutex;
    std::map<std::pair<int64_t, int64_t>, bool> preparedOwners;
};

std::shared_ptr<worldgen::NativeHydrologyRouter> makeContextHydrologyRouter(
    const GenerationIdentity& identity, const std::filesystem::path& root, AuthorityQuality quality,
    std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry> identityRegistry) {
    std::shared_ptr<hydrology::HydrologyAuthorityStore> store;
    if (!root.empty()) {
        store = std::make_shared<hydrology::HydrologyAuthorityStore>(
            root / (quality == AuthorityQuality::PREVIEW ? "preview" : "final"), identity, quality);
    }
    return std::make_shared<worldgen::NativeHydrologyRouter>(
        identity.seed, std::move(store), std::move(identityRegistry),
        worldgen::NATIVE_HYDROLOGY_VISIBLE_HORIZON_CACHE_BYTE_BUDGET);
}

class WorldGenerationContext::Impl {
public:
    Impl(GenerationIdentity requestedIdentity, std::shared_ptr<TerrainAuthority> requestedAuthority,
         AuthorityQuality requestedQuality, AuthorityRequestPriority requestedPriority,
         std::shared_ptr<FailureLatch> requestedFailureLatch,
         std::shared_ptr<HydrologyPreparationRegistry> requestedHydrologyPreparation,
         std::filesystem::path requestedHydrologyAuthorityRoot,
         std::shared_ptr<worldgen::NativeHydrologyRouter> requestedNativeHydrologyRouter,
         std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
             requestedNativeHydrologyIdentityRegistry)
        : identity(std::move(requestedIdentity))
        , generationFingerprint(identity.fingerprint())
        , authority(std::move(requestedAuthority))
        , quality(requestedQuality)
        , requestPriority(requestedPriority)
        , failureLatch(std::move(requestedFailureLatch))
        , hydrologyPreparation(std::move(requestedHydrologyPreparation))
        , hydrologyAuthorityRoot(std::move(requestedHydrologyAuthorityRoot))
        , nativeHydrologyRouter(std::move(requestedNativeHydrologyRouter))
        , nativeHydrologyIdentityRegistry(std::move(requestedNativeHydrologyIdentityRegistry)) {
        if (!nativeHydrologyIdentityRegistry) {
            nativeHydrologyIdentityRegistry =
                std::make_shared<worldgen::NativeHydrologyIdentityRegistry>(identity.seed);
        }
    }

    [[nodiscard]] std::optional<GenerationFailure> failure() const {
        std::lock_guard lock(failureLatch->mutex);
        return failureLatch->failure;
    }

    void latch(GenerationFailure failure) const {
        if (failure.code == GenerationFailureCode::NONE) return;
        std::lock_guard lock(failureLatch->mutex);
        if (!failureLatch->failure) failureLatch->failure = std::move(failure);
    }

    template <typename Value, typename Query>
    AuthorityResult<Value> executeQuery(uint64_t sampleCount, Query&& query) const {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.queries;
            if (std::numeric_limits<uint64_t>::max() - metrics.requestedSamples < sampleCount)
                metrics.requestedSamples = std::numeric_limits<uint64_t>::max();
            else
                metrics.requestedSamples += sampleCount;
        }
        if (const std::optional<GenerationFailure> current = failure()) {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.failedQueries;
            return AuthorityResult<Value>::failed(*current);
        }

        AuthorityResult<Value> result = AuthorityResult<Value>::failed(
            makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                        "World generation authority query did not complete", true));
        try {
            result = std::forward<Query>(query)();
        } catch (const std::exception& exception) {
            result = AuthorityResult<Value>::failed(makeFailure(
                GenerationFailureCode::INFERENCE_FAILED,
                std::string("World generation authority query threw: ") + exception.what(), true));
        } catch (...) {
            result = AuthorityResult<Value>::failed(
                makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                            "World generation authority query threw an unknown exception", true));
        }
        if (const std::optional<GenerationFailure> current = failure()) {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.failedQueries;
            return AuthorityResult<Value>::failed(*current);
        }
        if (result.isReady()) {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.readyQueries;
            return result;
        }
        if (result.status() == AuthorityStatus::DEFERRED) {
            std::lock_guard<std::mutex> lock(mutex);
            ++metrics.deferredQueries;
            return result;
        }
        if (result.failure() && result.failure()->code != GenerationFailureCode::INVALID_REQUEST &&
            result.failure()->code != GenerationFailureCode::NONE) {
            latch(*result.failure());
        }
        std::lock_guard<std::mutex> lock(mutex);
        ++metrics.failedQueries;
        return result;
    }

    GenerationIdentity identity;
    Sha256Digest generationFingerprint;
    std::shared_ptr<TerrainAuthority> authority;
    AuthorityQuality quality;
    AuthorityRequestPriority requestPriority;
    std::shared_ptr<FailureLatch> failureLatch;
    std::shared_ptr<HydrologyPreparationRegistry> hydrologyPreparation;
    std::filesystem::path hydrologyAuthorityRoot;
    std::shared_ptr<worldgen::NativeHydrologyRouter> nativeHydrologyRouter;
    std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
        nativeHydrologyIdentityRegistry;

    mutable std::mutex mutex;
    mutable WorldGenerationMetrics metrics;
};

WorldGenerationContext::WorldGenerationContext(GenerationIdentity identity,
                                               std::shared_ptr<TerrainAuthority> authority,
                                               AuthorityQuality quality,
                                               std::filesystem::path hydrologyAuthorityRoot)
    : WorldGenerationContext(std::move(identity), std::move(authority), quality,
                             defaultRequestPriority(quality), std::make_shared<FailureLatch>(),
                             std::make_shared<HydrologyPreparationRegistry>(),
                             std::move(hydrologyAuthorityRoot), nullptr, nullptr) {}

WorldGenerationContext::WorldGenerationContext(GenerationIdentity identity,
                                               std::shared_ptr<TerrainAuthority> authority,
                                               AuthorityQuality quality,
                                               AuthorityRequestPriority requestPriority,
                                               std::filesystem::path hydrologyAuthorityRoot)
    : WorldGenerationContext(std::move(identity), std::move(authority), quality, requestPriority,
                             std::make_shared<FailureLatch>(),
                             std::make_shared<HydrologyPreparationRegistry>(),
                             std::move(hydrologyAuthorityRoot), nullptr, nullptr) {}

WorldGenerationContext::WorldGenerationContext(
    GenerationIdentity identity, std::shared_ptr<TerrainAuthority> authority,
    AuthorityQuality quality, AuthorityRequestPriority requestPriority,
    std::shared_ptr<FailureLatch> failureLatch,
    std::shared_ptr<HydrologyPreparationRegistry> hydrologyPreparation,
    std::filesystem::path hydrologyAuthorityRoot,
    std::shared_ptr<worldgen::NativeHydrologyRouter> nativeHydrologyRouter,
    std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
        nativeHydrologyIdentityRegistry)
    : impl_(std::make_unique<Impl>(
          std::move(identity), std::move(authority), quality, requestPriority,
          std::move(failureLatch), std::move(hydrologyPreparation),
          std::move(hydrologyAuthorityRoot), std::move(nativeHydrologyRouter),
          std::move(nativeHydrologyIdentityRegistry))) {
    if (!impl_->nativeHydrologyRouter) {
        impl_->nativeHydrologyRouter =
            makeContextHydrologyRouter(impl_->identity, impl_->hydrologyAuthorityRoot, quality,
                                       impl_->nativeHydrologyIdentityRegistry);
    }
    impl_->metrics.quality = quality;
    impl_->metrics.requestPriority = requestPriority;
    impl_->metrics.generationFingerprint = impl_->generationFingerprint;
    if (!impl_->identity.valid() || !validQuality(quality) ||
        !validRequestPriority(requestPriority)) {
        impl_->latch(makeFailure(GenerationFailureCode::INVALID_REQUEST,
                                 "World generation context has an invalid identity or quality",
                                 false));
    } else if (!impl_->authority) {
        impl_->latch(makeFailure(GenerationFailureCode::BACKEND_UNAVAILABLE,
                                 "World generation authority is unavailable", true));
    } else if (impl_->authority->generationIdentity() != impl_->identity) {
        impl_->latch(makeFailure(
            GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
            "World generation authority does not match the requested generation identity", false));
    }
}

WorldGenerationContext::~WorldGenerationContext() = default;

class WorldGenerationContext::MakeSharedEnabler final : public WorldGenerationContext {
public:
    MakeSharedEnabler(GenerationIdentity identity, std::shared_ptr<TerrainAuthority> authority,
                      AuthorityQuality quality, AuthorityRequestPriority requestPriority,
                      std::shared_ptr<FailureLatch> failureLatch,
                      std::shared_ptr<HydrologyPreparationRegistry> hydrologyPreparation,
                      std::filesystem::path hydrologyAuthorityRoot,
                      std::shared_ptr<worldgen::NativeHydrologyRouter> nativeHydrologyRouter,
                      std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
                          nativeHydrologyIdentityRegistry)
        : WorldGenerationContext(std::move(identity), std::move(authority), quality,
                                 requestPriority, std::move(failureLatch),
                                 std::move(hydrologyPreparation), std::move(hydrologyAuthorityRoot),
                                 std::move(nativeHydrologyRouter),
                                 std::move(nativeHydrologyIdentityRegistry)) {}
};

const GenerationIdentity& WorldGenerationContext::identity() const noexcept {
    return impl_->identity;
}

const Sha256Digest& WorldGenerationContext::fingerprint() const noexcept {
    return impl_->generationFingerprint;
}

AuthorityQuality WorldGenerationContext::quality() const noexcept {
    return impl_->quality;
}

AuthorityRequestPriority WorldGenerationContext::requestPriority() const noexcept {
    return impl_->requestPriority;
}

const std::filesystem::path& WorldGenerationContext::hydrologyAuthorityRoot() const noexcept {
    return impl_->hydrologyAuthorityRoot;
}

std::shared_ptr<worldgen::NativeHydrologyRouter>
WorldGenerationContext::nativeHydrologyRouter() const noexcept {
    return impl_->nativeHydrologyRouter;
}

std::shared_ptr<const worldgen::NativeHydrologyIdentityRegistry>
WorldGenerationContext::nativeHydrologyIdentityRegistry() const noexcept {
    return impl_->nativeHydrologyIdentityRegistry;
}

void WorldGenerationContext::recordPreparedNativeHydrologyOwner(int64_t ownerPageX,
                                                                int64_t ownerPageZ) const {
    // PREVIEW routing cannot prove that the FINAL terrain and hydrology
    // fingerprints required by exact startup have been prepared.
    if (impl_->quality != AuthorityQuality::FINAL) return;
    std::lock_guard lock(impl_->hydrologyPreparation->mutex);
    impl_->hydrologyPreparation->preparedOwners[{ownerPageX, ownerPageZ}] = true;
}

bool WorldGenerationContext::nativeHydrologyOwnerPrepared(int64_t ownerPageX,
                                                          int64_t ownerPageZ) const {
    std::lock_guard lock(impl_->hydrologyPreparation->mutex);
    return impl_->hydrologyPreparation->preparedOwners.contains({ownerPageX, ownerPageZ});
}

std::shared_ptr<void> WorldGenerationContext::retainProtectedAuthorityWindows(
    std::span<const NativeRect> finalRegions) const {
    if (!impl_->authority || finalRegions.empty()) return nullptr;
    std::vector<LearnedAuthorityRequest> requests;
    requests.reserve(finalRegions.size());
    for (const NativeRect region : finalRegions)
        requests.push_back({.quality = AuthorityQuality::FINAL, .region = region});
    const auto plan = LearnedAuthorityGraph::build(requests);
    if (!plan.isReady()) return nullptr; // over a bound: skip pinning, never fatal
    return impl_->authority->retainWindowGraph(**plan.value());
}

size_t WorldGenerationContext::preparedNativeHydrologyOwnerCount() const {
    std::lock_guard lock(impl_->hydrologyPreparation->mutex);
    return impl_->hydrologyPreparation->preparedOwners.size();
}

std::shared_ptr<WorldGenerationContext>
WorldGenerationContext::withQuality(AuthorityQuality quality) const {
    const std::shared_ptr<worldgen::NativeHydrologyRouter> sharedRouter =
        quality == impl_->quality ? impl_->nativeHydrologyRouter : nullptr;
    return std::make_shared<MakeSharedEnabler>(
        impl_->identity, impl_->authority, quality, defaultRequestPriority(quality),
        impl_->failureLatch, impl_->hydrologyPreparation, impl_->hydrologyAuthorityRoot,
        sharedRouter, impl_->nativeHydrologyIdentityRegistry);
}

std::shared_ptr<WorldGenerationContext>
WorldGenerationContext::withRequestPriority(AuthorityRequestPriority priority) const {
    return std::make_shared<MakeSharedEnabler>(
        impl_->identity, impl_->authority, impl_->quality, priority, impl_->failureLatch,
        impl_->hydrologyPreparation, impl_->hydrologyAuthorityRoot, impl_->nativeHydrologyRouter,
        impl_->nativeHydrologyIdentityRegistry);
}

AuthorityResult<bool>
WorldGenerationContext::requestAuthorityPage(TerrainPageCoordinate coordinate,
                                             AuthorityRequestPriority priority,
                                             ProtectedHandoffEpoch epoch) const {
    return impl_->executeQuery<bool>(0, [&] {
        auto prepared = impl_->authority->preparePage(
            TerrainPageKey{.quality = impl_->quality, .coordinate = coordinate}, priority, epoch);
        if (prepared.isReady()) return AuthorityResult<bool>::ready(true);
        return forwardFailure<bool>(prepared);
    });
}

AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>
WorldGenerationContext::retainAuthorityPage(
    TerrainPageCoordinate coordinate, std::optional<AuthorityRequestPriority> priority) const {
    return impl_->executeQuery<std::shared_ptr<const TerrainAuthorityPage>>(0, [&] {
        const TerrainPageKey requestedKey{.quality = impl_->quality, .coordinate = coordinate};
        auto result =
            impl_->authority->preparePage(requestedKey, priority.value_or(impl_->requestPriority));
        if (!result.isReady()) return result;

        const auto* retained = result.value();
        if (retained == nullptr || !*retained) {
            return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Terrain authority returned a null retained page", true));
        }
        const TerrainAuthorityPage& page = **retained;
        if (!page.valid()) {
            return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::failed(
                makeFailure(GenerationFailureCode::CORRUPT_PAGE,
                            "Terrain authority returned an invalid retained page payload", true));
        }
        if (page.key != requestedKey) {
            return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::failed(makeFailure(
                GenerationFailureCode::CORRUPT_PAGE,
                "Terrain authority returned a retained page for a different key", true));
        }
        if (page.generationSeed != impl_->identity.seed ||
            page.generationFingerprint != impl_->generationFingerprint) {
            return AuthorityResult<std::shared_ptr<const TerrainAuthorityPage>>::failed(
                makeFailure(GenerationFailureCode::INCOMPATIBLE_FINGERPRINT,
                            "Terrain authority returned a retained page from a different "
                            "generation identity",
                            false));
        }
        return result;
    });
}

AuthorityResult<bool>
WorldGenerationContext::requestAuthorityPages(std::span<const TerrainPageCoordinate> coordinates,
                                              AuthorityRequestPriority priority,
                                              ProtectedHandoffEpoch epoch) const {
    return impl_->executeQuery<bool>(0, [&] {
        if (coordinates.empty()) {
            return AuthorityResult<bool>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "World generation authority page closure is empty", false));
        }
        std::vector<TerrainPageKey> keys;
        keys.reserve(coordinates.size());
        for (const TerrainPageCoordinate coordinate : coordinates) {
            keys.push_back({.quality = impl_->quality, .coordinate = coordinate});
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return impl_->authority->preparePages(keys, priority, epoch);
    });
}

AuthorityResult<bool>
WorldGenerationContext::requestWorldPage(int64_t worldX, int64_t worldZ,
                                         AuthorityRequestPriority priority) const {
    return impl_->executeQuery<bool>(0, [&] {
        const auto nativeAxis = [](int64_t coordinate) {
            const int64_t containing =
                world_coord::floorDiv(coordinate, static_cast<int64_t>(MODEL_BLOCK_SCALE));
            const int64_t remainder =
                coordinate - containing * static_cast<int64_t>(MODEL_BLOCK_SCALE);
            return remainder < MODEL_BLOCK_SCALE / 2 ? std::pair{containing - 1, containing}
                                                     : std::pair{containing, containing + 1};
        };
        const auto [column0, column1] = nativeAxis(worldX);
        const auto [row0, row1] = nativeAxis(worldZ);
        std::array<TerrainPageCoordinate, 4> coordinates{
            terrainPageCoordinateFor({.row = row0, .column = column0}),
            terrainPageCoordinateFor({.row = row0, .column = column1}),
            terrainPageCoordinateFor({.row = row1, .column = column0}),
            terrainPageCoordinateFor({.row = row1, .column = column1}),
        };
        std::sort(coordinates.begin(), coordinates.end());
        const auto uniqueEnd = std::unique(coordinates.begin(), coordinates.end());
        std::optional<GenerationFailure> deferred;
        for (auto current = coordinates.begin(); current != uniqueEnd; ++current) {
            auto prepared = impl_->authority->preparePage(
                TerrainPageKey{.quality = impl_->quality, .coordinate = *current}, priority);
            if (prepared.status() == AuthorityStatus::FAILED) return forwardFailure<bool>(prepared);
            if (prepared.status() == AuthorityStatus::DEFERRED && !deferred)
                deferred = *prepared.failure();
        }
        if (deferred) return AuthorityResult<bool>::deferred(std::move(*deferred));
        return AuthorityResult<bool>::ready(true);
    });
}

AuthorityResult<PhysicalTerrainGrid>
WorldGenerationContext::queryNative(NativeRect region,
                                    std::optional<AuthorityRequestPriority> priority) const {
    uint64_t sampleCount = 0;
    if (region.valid()) {
        sampleCount =
            region.height() != 0 &&
                    region.width() > std::numeric_limits<uint64_t>::max() / region.height()
                ? std::numeric_limits<uint64_t>::max()
                : region.height() * region.width();
    }
    return impl_->executeQuery<PhysicalTerrainGrid>(sampleCount, [&] {
        return impl_->authority->queryNative(region, impl_->quality,
                                             priority.value_or(impl_->requestPriority));
    });
}

AuthorityResult<CoarseSpawnGrid> WorldGenerationContext::queryCoarseSpawnGrid(
    CoarseSpawnRegion region, std::optional<AuthorityRequestPriority> priority) const {
    uint64_t sampleCount = 0;
    if (region.valid()) {
        sampleCount =
            region.height() != 0 &&
                    region.width() > std::numeric_limits<uint64_t>::max() / region.height()
                ? std::numeric_limits<uint64_t>::max()
                : region.height() * region.width();
    }
    return impl_->executeQuery<CoarseSpawnGrid>(sampleCount, [&] {
        return impl_->authority->queryCoarseSpawnGrid(
            region, priority.value_or(AuthorityRequestPriority::SPAWN));
    });
}

AuthorityResult<std::vector<PhysicalTerrainSample>>
WorldGenerationContext::queryNativePoints(std::span<const NativePoint> points,
                                          std::optional<AuthorityRequestPriority> priority) const {
    return impl_->executeQuery<std::vector<PhysicalTerrainSample>>(points.size(), [&] {
        return impl_->authority->queryNativePoints(points, impl_->quality,
                                                   priority.value_or(impl_->requestPriority));
    });
}

AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>
WorldGenerationContext::queryTransientFinalNativeGrid(
    NativeRect region, std::optional<AuthorityRequestPriority> priority,
    ProtectedHandoffEpoch epoch) const {
    const uint64_t sampleCount =
        region.valid() && region.height() <= std::numeric_limits<uint64_t>::max() /
                                                 std::max<uint64_t>(1, region.width())
            ? region.height() * region.width()
            : std::numeric_limits<uint64_t>::max();
    return impl_->executeQuery<std::shared_ptr<const PhysicalTerrainGrid>>(sampleCount, [&] {
        if (impl_->quality != AuthorityQuality::FINAL) {
            return AuthorityResult<std::shared_ptr<const PhysicalTerrainGrid>>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "Transient learned terrain rectangles require final authority", false));
        }
        return impl_->authority->queryTransientFinalNativeGrid(
            region, priority.value_or(impl_->requestPriority), epoch);
    });
}

AuthorityResult<std::vector<PhysicalTerrainSample>>
WorldGenerationContext::queryWorldPoints(std::span<const WorldBlockPoint> points,
                                         std::optional<AuthorityRequestPriority> priority) const {
    return impl_->executeQuery<std::vector<PhysicalTerrainSample>>(points.size(), [&] {
        if (points.size() > MAXIMUM_AUTHORITY_QUERY_SAMPLES) {
            return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(makeFailure(
                GenerationFailureCode::INVALID_REQUEST,
                "World generation point query exceeds the bounded sample count", false));
        }
        struct AxisInterpolation {
            int64_t lower = 0;
            int64_t upper = 0;
            double fraction = 0.0;
        };
        const auto interpolationAxis = [](int64_t worldCoordinate) {
            const int64_t containing =
                world_coord::floorDiv(worldCoordinate, static_cast<int64_t>(MODEL_BLOCK_SCALE));
            const int64_t remainder =
                worldCoordinate - containing * static_cast<int64_t>(MODEL_BLOCK_SCALE);
            const double offset = (static_cast<double>(remainder) + 0.5) / MODEL_BLOCK_SCALE - 0.5;
            if (offset < 0.0) {
                return AxisInterpolation{
                    .lower = containing - 1,
                    .upper = containing,
                    .fraction = offset + 1.0,
                };
            }
            return AxisInterpolation{
                .lower = containing,
                .upper = containing + 1,
                .fraction = offset,
            };
        };
        struct PointInterpolation {
            std::array<size_t, 4> samples{};
            double rowFraction = 0.0;
            double columnFraction = 0.0;
        };

        std::map<NativePoint, size_t> uniqueIndices;
        std::vector<NativePoint> nativePoints;
        nativePoints.reserve(points.size());
        std::vector<PointInterpolation> interpolation;
        interpolation.reserve(points.size());
        const auto retain = [&](NativePoint point) {
            const auto [found, inserted] = uniqueIndices.try_emplace(point, nativePoints.size());
            if (inserted) nativePoints.push_back(point);
            return found->second;
        };
        for (WorldBlockPoint point : points) {
            const AxisInterpolation row = interpolationAxis(point.z);
            const AxisInterpolation column = interpolationAxis(point.x);
            interpolation.push_back({
                .samples =
                    {
                        retain({.row = row.lower, .column = column.lower}),
                        retain({.row = row.lower, .column = column.upper}),
                        retain({.row = row.upper, .column = column.lower}),
                        retain({.row = row.upper, .column = column.upper}),
                    },
                .rowFraction = row.fraction,
                .columnFraction = column.fraction,
            });
        }
        if (nativePoints.size() > MAXIMUM_AUTHORITY_QUERY_SAMPLES) {
            return AuthorityResult<std::vector<PhysicalTerrainSample>>::failed(
                makeFailure(GenerationFailureCode::INVALID_REQUEST,
                            "Bilinear world query exceeds the bounded native sample count", false));
        }
        auto native = impl_->authority->queryNativePoints(
            nativePoints, impl_->quality, priority.value_or(impl_->requestPriority));
        if (!native.isReady()) return forwardFailure<std::vector<PhysicalTerrainSample>>(native);

        std::vector<PhysicalTerrainSample> output;
        output.reserve(points.size());
        const auto bilinear = [&](const PointInterpolation& weights, auto member) {
            const auto& samples = *native.value();
            const double north =
                std::lerp(samples[weights.samples[0]].*member, samples[weights.samples[1]].*member,
                          weights.columnFraction);
            const double south =
                std::lerp(samples[weights.samples[2]].*member, samples[weights.samples[3]].*member,
                          weights.columnFraction);
            return std::lerp(north, south, weights.rowFraction);
        };
        for (const PointInterpolation& weights : interpolation) {
            output.push_back({
                .elevationMeters = bilinear(weights, &PhysicalTerrainSample::elevationMeters),
                .meanTemperatureC = bilinear(weights, &PhysicalTerrainSample::meanTemperatureC),
                .temperatureVariabilityC =
                    bilinear(weights, &PhysicalTerrainSample::temperatureVariabilityC),
                .annualPrecipitationMm =
                    bilinear(weights, &PhysicalTerrainSample::annualPrecipitationMm),
                .precipitationCoefficientOfVariation =
                    bilinear(weights, &PhysicalTerrainSample::precipitationCoefficientOfVariation),
                .lapseRateCPerMeter = bilinear(weights, &PhysicalTerrainSample::lapseRateCPerMeter),
            });
        }
        return AuthorityResult<std::vector<PhysicalTerrainSample>>::ready(std::move(output));
    });
}

AuthorityResult<PhysicalTerrainSample>
WorldGenerationContext::sampleWorld(int64_t worldX, int64_t worldZ,
                                    std::optional<AuthorityRequestPriority> priority) const {
    const std::array point{WorldBlockPoint{.x = worldX, .z = worldZ}};
    auto queried = queryWorldPoints(point, priority);
    if (!queried.isReady()) return forwardFailure<PhysicalTerrainSample>(queried);
    if (queried.value()->size() != 1) {
        const GenerationFailure invalid =
            makeFailure(GenerationFailureCode::INFERENCE_FAILED,
                        "World generation authority returned the wrong point count", true);
        latchFailure(invalid);
        return AuthorityResult<PhysicalTerrainSample>::failed(invalid);
    }
    return AuthorityResult<PhysicalTerrainSample>::ready(queried.value()->front());
}

void WorldGenerationContext::latchFailure(GenerationFailure failure) const {
    impl_->latch(std::move(failure));
}

bool WorldGenerationContext::clearRetriableFailure() const {
    std::lock_guard lock(impl_->failureLatch->mutex);
    if (!impl_->failureLatch->failure || !impl_->failureLatch->failure->retriable) return false;
    impl_->failureLatch->failure.reset();
    return true;
}

std::optional<GenerationFailure> WorldGenerationContext::failure() const {
    return impl_->failure();
}

WorldGenerationMetrics WorldGenerationContext::metrics() const {
    WorldGenerationMetrics result;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        result = impl_->metrics;
    }
    if (impl_->authority) result.authorityCache = impl_->authority->cacheMetrics();
    return result;
}

} // namespace worldgen::learned
