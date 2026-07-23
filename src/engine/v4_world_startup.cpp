#include "engine/v4_world_startup.hpp"

#include "render/far_terrain.hpp"
#include "world/chunk.hpp"
#include "world/learned_terrain.hpp"
#include "world/macro_generation.hpp"
#include "world/native_hydrology.hpp"
#include "world/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <system_error>
#include <thread>
#include <vector>

namespace {

bool isRecoverableV4Residue(const std::filesystem::path& worldPath) {
    std::error_code error;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(worldPath, error)) {
        if (error) return false;
        const std::string name = entry.path().filename().string();
        if (name == "metadata.json.tmp") {
            if (!entry.is_regular_file(error) || error) return false;
            continue;
        }
        const bool expectedDirectory = name == SaveManager::V4_REGIONS_DIRECTORY ||
                                       name == SaveManager::V4_TERRAIN_AUTHORITY_DIRECTORY ||
                                       name == SaveManager::V4_HYDROLOGY_AUTHORITY_DIRECTORY;
        if (!expectedDirectory || !entry.is_directory(error) || error ||
            !std::filesystem::is_empty(entry.path(), error) || error) {
            return false;
        }
    }
    return !error;
}

enum class V4ProfileInspectionKind : uint8_t {
    Available,
    Compatible,
    IdentityConflict,
    InvalidDirectory,
    MissingMetadata,
};

struct V4ProfileInspection {
    V4ProfileInspectionKind kind = V4ProfileInspectionKind::InvalidDirectory;
    bool existed = false;
    bool recoverableResidue = false;
    std::optional<SaveManager::WorldMetadata> metadata;
    std::string message;
};

bool v4MetadataMatches(const SaveManager::WorldMetadata& metadata, uint64_t seed,
                       std::string_view fingerprint) {
    return metadata.generatorVersion == SaveManager::GENERATOR_V4_VERSION &&
           metadata.chunkFormatVersion == CHUNK_VERSION && metadata.seed == seed &&
           metadata.generationFingerprint == fingerprint;
}

V4ProfileInspection inspectV4Profile(const std::filesystem::path& worldPath, uint64_t seed,
                                     std::string_view fingerprint) {
    std::error_code error;
    const bool existed = std::filesystem::exists(worldPath, error);
    if (error || (existed && !std::filesystem::is_directory(worldPath, error)) || error) {
        return {.kind = V4ProfileInspectionKind::InvalidDirectory,
                .existed = existed,
                .message = "The generator v4 world path is not a usable directory"};
    }
    if (!existed) return {.kind = V4ProfileInspectionKind::Available};

    const std::filesystem::directory_iterator begin(worldPath, error);
    if (error) {
        return {.kind = V4ProfileInspectionKind::InvalidDirectory,
                .existed = true,
                .message = "The generator v4 world directory could not be inspected"};
    }
    if (begin == std::filesystem::directory_iterator{}) {
        return {.kind = V4ProfileInspectionKind::Available, .existed = true};
    }

    const std::optional<SaveManager::WorldMetadata> metadata =
        SaveManager::inspectMetadata(worldPath.string(), SaveManager::Profile::GeneratorV4);
    if (!metadata) {
        if (isRecoverableV4Residue(worldPath)) {
            return {.kind = V4ProfileInspectionKind::Available,
                    .existed = true,
                    .recoverableResidue = true};
        }
        return {.kind = V4ProfileInspectionKind::MissingMetadata,
                .existed = true,
                .message = "The existing generator v4 world metadata is missing or corrupt"};
    }
    if (v4MetadataMatches(*metadata, seed, fingerprint)) {
        return {.kind = V4ProfileInspectionKind::Compatible, .existed = true, .metadata = metadata};
    }
    return {
        .kind = V4ProfileInspectionKind::IdentityConflict, .existed = true, .metadata = metadata};
}

std::filesystem::path separateV4ProfilePath(const std::filesystem::path& defaultPath, uint64_t seed,
                                            std::string_view fingerprint,
                                            uint32_t collisionOrdinal) {
    char seedHex[17]{};
    std::snprintf(seedHex, sizeof(seedHex), "%016llx", static_cast<unsigned long long>(seed));
    std::string name = defaultPath.filename().string() + "-seed-" + seedHex + "-fingerprint-" +
                       std::string(fingerprint);
    if (collisionOrdinal != 0) name += "-" + std::to_string(collisionOrdinal);
    return defaultPath.parent_path() / name;
}

constexpr uint32_t MAXIMUM_SEPARATE_V4_PROFILE_COLLISIONS = 4'096;

std::optional<std::filesystem::path>
createV4StagingDirectory(const std::filesystem::path& worldPath, std::string& message) {
    std::error_code error;
    std::filesystem::create_directories(worldPath.parent_path(), error);
    if (error) {
        message = "The generator v4 world parent directory could not be created";
        return std::nullopt;
    }
    const uint64_t nonce =
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    for (uint64_t attempt = 0; attempt < 128; ++attempt) {
        std::filesystem::path candidate =
            worldPath.parent_path() /
            (worldPath.filename().string() + ".creating." + std::to_string(nonce + attempt));
        if (std::filesystem::create_directory(candidate, error)) return candidate;
        if (error && error != std::errc::file_exists) {
            message = "The generator v4 staging directory could not be created";
            return std::nullopt;
        }
        error.clear();
    }
    message = "A unique generator v4 staging directory could not be reserved";
    return std::nullopt;
}

struct StagingWorldCleanup {
    std::filesystem::path path;
    bool committed = false;

    ~StagingWorldCleanup() {
        if (committed || path.empty()) return;
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::optional<int64_t> checkedSpawnCoordinate(__int128 value) {
    if (value < std::numeric_limits<int64_t>::min() ||
        value > std::numeric_limits<int64_t>::max()) {
        return std::nullopt;
    }
    return static_cast<int64_t>(value);
}

std::optional<int64_t> checkedAdd(int64_t value, int64_t offset) {
    return checkedSpawnCoordinate(static_cast<__int128>(value) + offset);
}

std::optional<int64_t> checkedMultiply(int64_t value, int64_t factor) {
    return checkedSpawnCoordinate(static_cast<__int128>(value) * factor);
}

std::optional<int64_t> checkedFloorSpawnCoordinate(float value) {
    if (!std::isfinite(value)) return std::nullopt;
    const long double floored = std::floor(static_cast<long double>(value));
    constexpr long double MINIMUM = -9'223'372'036'854'775'808.0L;
    constexpr long double MAXIMUM_EXCLUSIVE = 9'223'372'036'854'775'808.0L;
    if (floored < MINIMUM || floored >= MAXIMUM_EXCLUSIVE) {
        return std::nullopt;
    }
    return static_cast<int64_t>(floored);
}

std::optional<int64_t> firstGloballyAlignedSpawnCoordinate(int64_t minimum) {
    constexpr int64_t SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    const std::optional<int64_t> aligned =
        checkedMultiply(world_coord::floorDiv(minimum, SPACING), SPACING);
    if (!aligned) return std::nullopt;
    return *aligned < minimum ? checkedAdd(*aligned, SPACING) : aligned;
}

std::optional<int64_t> nearestGloballyAlignedSpawnCoordinate(int64_t minimum, int64_t maximum,
                                                             float requested) {
    if (!std::isfinite(requested) || minimum > maximum) return std::nullopt;
    const std::optional<int64_t> first = firstGloballyAlignedSpawnCoordinate(minimum);
    if (!first || *first > maximum) return std::nullopt;
    std::optional<int64_t> best;
    long double bestDistance = std::numeric_limits<long double>::infinity();
    for (int64_t coordinate = *first;;) {
        const long double delta = static_cast<long double>(coordinate) + 0.5L - requested;
        const long double distance = delta * delta;
        if (!best || distance < bestDistance || (distance == bestDistance && coordinate < *best)) {
            best = coordinate;
            bestDistance = distance;
        }
        const std::optional<int64_t> next =
            checkedAdd(coordinate, V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS);
        if (!next || *next > maximum) break;
        coordinate = *next;
    }
    return best;
}

std::optional<V4SpawnAlignedCandidate> nearestAlignedSpawnCandidate(Vec3 requested, int64_t worldX,
                                                                    int64_t worldZ) {
    constexpr int64_t EDGE = worldgen::NATIVE_HYDROLOGY_PAGE_EDGE;
    constexpr int64_t BUFFER_REACH =
        V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES * V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    constexpr int64_t INTERIOR_MARGIN =
        worldgen::NATIVE_HYDROLOGY_HANDOFF_BLOCKS + 1 + BUFFER_REACH;
    const std::optional<int64_t> ownerX =
        checkedMultiply(world_coord::floorDiv(worldX, EDGE), EDGE);
    const std::optional<int64_t> ownerZ =
        checkedMultiply(world_coord::floorDiv(worldZ, EDGE), EDGE);
    const std::optional<int64_t> minimumX =
        ownerX ? checkedAdd(*ownerX, INTERIOR_MARGIN) : std::nullopt;
    const std::optional<int64_t> minimumZ =
        ownerZ ? checkedAdd(*ownerZ, INTERIOR_MARGIN) : std::nullopt;
    const std::optional<int64_t> ownerEndX = ownerX ? checkedAdd(*ownerX, EDGE) : std::nullopt;
    const std::optional<int64_t> ownerEndZ = ownerZ ? checkedAdd(*ownerZ, EDGE) : std::nullopt;
    const std::optional<int64_t> maximumX =
        ownerEndX ? checkedAdd(*ownerEndX, -INTERIOR_MARGIN) : std::nullopt;
    const std::optional<int64_t> maximumZ =
        ownerEndZ ? checkedAdd(*ownerEndZ, -INTERIOR_MARGIN) : std::nullopt;
    if (!minimumX || !minimumZ || !maximumX || !maximumZ) return std::nullopt;
    const std::optional<int64_t> alignedX =
        nearestGloballyAlignedSpawnCoordinate(*minimumX, *maximumX, requested.x);
    const std::optional<int64_t> alignedZ =
        nearestGloballyAlignedSpawnCoordinate(*minimumZ, *maximumZ, requested.z);
    if (!alignedX || !alignedZ) return std::nullopt;
    return V4SpawnAlignedCandidate{.worldX = *alignedX, .worldZ = *alignedZ};
}

struct V4SpawnHydrologyOwner {
    int64_t x = 0;
    int64_t z = 0;

    auto operator<=>(const V4SpawnHydrologyOwner&) const = default;
};

struct V4SpawnAuthorityPlan {
    worldgen::NativeHydrologyAuthorityRequirements requirements;
    std::vector<V4SpawnHydrologyOwner> hydrologyOwners;
};

struct V4SpawnAuthorityAxisBounds {
    int64_t minimum = 0;
    int64_t maximumExclusive = 0;
    int64_t firstOwner = 0;
    int64_t lastOwner = 0;
};

std::optional<V4SpawnAuthorityAxisBounds> v4SpawnAuthorityAxisBounds(int64_t worldCoordinate,
                                                                     int radiusChunks) {
    const int radius = boundedColdStartExactRadiusChunks(radiusChunks);
    const int planCoverageRadius = exactStreamingPlanCoverageRadiusChunks(radius);
    const int64_t centerChunk = Chunk::worldToChunk(worldCoordinate);
    const std::optional<int64_t> firstChunk =
        checkedAdd(centerChunk, -static_cast<int64_t>(planCoverageRadius));
    const std::optional<int64_t> lastChunk =
        checkedAdd(centerChunk, static_cast<int64_t>(planCoverageRadius));
    const std::optional<int64_t> firstBase =
        firstChunk ? checkedMultiply(*firstChunk, CHUNK_EDGE) : std::nullopt;
    const std::optional<int64_t> lastBase =
        lastChunk ? checkedMultiply(*lastChunk, CHUNK_EDGE) : std::nullopt;
    const std::optional<int64_t> minimum =
        firstBase ? checkedAdd(*firstBase, -COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS) : std::nullopt;
    const std::optional<int64_t> lastSampleOrigin =
        lastBase ? checkedAdd(*lastBase, -COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS) : std::nullopt;
    const std::optional<int64_t> maximumExclusive =
        lastSampleOrigin ? checkedAdd(*lastSampleOrigin, COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE)
                         : std::nullopt;
    if (!minimum || !maximumExclusive || *minimum >= *maximumExclusive) {
        return std::nullopt;
    }
    return V4SpawnAuthorityAxisBounds{
        .minimum = *minimum,
        .maximumExclusive = *maximumExclusive,
        .firstOwner = world_coord::floorDiv(
            *minimum, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)),
        .lastOwner = world_coord::floorDiv(
            *maximumExclusive - 1, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE)),
    };
}

std::optional<V4SpawnAuthorityPlan> v4SpawnAuthorityPlan(int64_t worldX, int64_t worldZ,
                                                         int radiusChunks) {
    const std::optional<V4SpawnAuthorityAxisBounds> x =
        v4SpawnAuthorityAxisBounds(worldX, radiusChunks);
    const std::optional<V4SpawnAuthorityAxisBounds> z =
        v4SpawnAuthorityAxisBounds(worldZ, radiusChunks);
    if (!x || !z) return std::nullopt;

    V4SpawnAuthorityPlan plan;
    plan.requirements = worldgen::nativeHydrologyAuthorityRequirementsForWorldRect(
        x->minimum, z->minimum, x->maximumExclusive, z->maximumExclusive);
    for (int64_t ownerZ = z->firstOwner;; ++ownerZ) {
        for (int64_t ownerX = x->firstOwner;; ++ownerX) {
            plan.hydrologyOwners.push_back({.x = ownerX, .z = ownerZ});
            if (ownerX == x->lastOwner) break;
        }
        if (ownerZ == z->lastOwner) break;
    }
    return plan;
}

std::optional<worldgen::learned::CoarseSpawnRegion>
coarseSpawnSearchRegion(int64_t originRow, int64_t originColumn, uint16_t edge) {
    using namespace worldgen::learned;
    if (edge == 0 || edge > V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE || edge % 2 != 0) {
        return std::nullopt;
    }
    const int64_t halfEdge = static_cast<int64_t>(edge / 2);
    const std::optional<int64_t> rowBegin = checkedAdd(originRow, -halfEdge);
    const std::optional<int64_t> columnBegin = checkedAdd(originColumn, -halfEdge);
    if (!rowBegin || !columnBegin) return std::nullopt;
    const std::optional<int64_t> rowEnd = checkedAdd(*rowBegin, edge);
    const std::optional<int64_t> columnEnd = checkedAdd(*columnBegin, edge);
    if (!rowEnd || !columnEnd) return std::nullopt;
    return CoarseSpawnRegion{
        .rowBegin = *rowBegin,
        .columnBegin = *columnBegin,
        .rowEnd = *rowEnd,
        .columnEnd = *columnEnd,
    };
}

// A learned spawn page is only a proposal. Materialize its FINAL samples
// before selecting a point. Canonical hydrology subsequently requests its
// exact 517-by-517 native raster as one transient coordinator flight, so the
// two-cell apron no longer forces twelve unrelated full pages here.
worldgen::learned::AuthorityResult<bool> requestSpawnCandidateFinalPage(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
    const worldgen::learned::NativeRect& nativeRegion) {
    using namespace worldgen::learned;
    if (!finalContext || !nativeRegion.valid()) {
        return AuthorityResult<bool>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message = "The dry-land spawn page has an invalid learned terrain rectangle",
             .retriable = false});
    }

    const TerrainPageCoordinate coordinate = terrainPageCoordinateFor(
        {.row = nativeRegion.rowBegin, .column = nativeRegion.columnBegin});
    const std::optional<NativeRect> canonicalRegion = terrainPageNativeRect(coordinate);
    if (!canonicalRegion || *canonicalRegion != nativeRegion) {
        return AuthorityResult<bool>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message = "The dry-land spawn final page is not canonically aligned",
             .retriable = false});
    }
    return finalContext->requestAuthorityPage(coordinate, AuthorityRequestPriority::SPAWN);
}

std::optional<worldgen::learned::TerrainPageCoordinate>
coarseLandSpawnPage(const worldgen::learned::CoarseSpawnGrid& grid, int64_t originRow,
                    int64_t originColumn, uint32_t ordinal) {
    using namespace worldgen::learned;
    if (!grid.valid() || ordinal >= V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES ||
        originRow < grid.region.rowBegin || originRow >= grid.region.rowEnd ||
        originColumn < grid.region.columnBegin || originColumn >= grid.region.columnEnd) {
        return std::nullopt;
    }

    const int64_t gridHeight = static_cast<int64_t>(grid.region.height());
    const int64_t gridWidth = static_cast<int64_t>(grid.region.width());
    struct Candidate {
        TerrainPageCoordinate coordinate;
        V4SpawnHydrologyOwner owner;
        float neighborhoodFloorMeters = 0.0F;
        float centerElevationMeters = 0.0F;
        int64_t distance = 0;
        bool ownerHasCoarseOcean = false;
        bool fullyLand = false;
    };
    std::vector<Candidate> candidates;
    std::vector<Candidate> oceanBacked;
    std::vector<Candidate> inland;
    std::vector<Candidate> coastal;
    candidates.reserve(static_cast<size_t>(gridHeight * gridWidth));
    for (int64_t row = 0; row < gridHeight; ++row) {
        for (int64_t column = 0; column < gridWidth; ++column) {
            const int64_t globalRow = grid.region.rowBegin + row;
            const int64_t globalColumn = grid.region.columnBegin + column;
            const float* center = grid.sample(globalRow, globalColumn);
            if (!center || !std::isfinite(*center) || *center <= 0.0F) continue;

            // A positive 3-by-3 coarse floor is a 23-kilometer-scale inland
            // signal. Rank that floor before proximity so startup tries the
            // strongest continental interior instead of materializing every
            // barely positive coastal or lake-basin page near the origin.
            bool fullyLand =
                row > 0 && column > 0 && row + 1 < gridHeight && column + 1 < gridWidth;
            float neighborhoodFloor = *center;
            for (int64_t neighborRow = row - 1; fullyLand && neighborRow <= row + 1;
                 ++neighborRow) {
                for (int64_t neighborColumn = column - 1; neighborColumn <= column + 1;
                     ++neighborColumn) {
                    const float* elevation = grid.sample(grid.region.rowBegin + neighborRow,
                                                         grid.region.columnBegin + neighborColumn);
                    fullyLand = elevation && std::isfinite(*elevation) && *elevation > 0.0F;
                    if (!fullyLand) break;
                    neighborhoodFloor = std::min(neighborhoodFloor, *elevation);
                }
            }
            Candidate candidate{
                .coordinate = {.row = globalRow, .column = globalColumn},
                .owner = {.x = floorDivide(globalColumn, int64_t{2}),
                          .z = floorDivide(globalRow, int64_t{2})},
                .neighborhoodFloorMeters = neighborhoodFloor,
                .centerElevationMeters = *center,
                .distance = std::max(std::abs(globalRow - originRow),
                                     std::abs(globalColumn - originColumn)),
                .fullyLand = fullyLand,
            };
            const int64_t ownerFirstRow = candidate.owner.z * 2;
            const int64_t ownerFirstColumn = candidate.owner.x * 2;
            for (int64_t ownerRow = ownerFirstRow;
                 !candidate.ownerHasCoarseOcean && ownerRow < ownerFirstRow + 2; ++ownerRow) {
                for (int64_t ownerColumn = ownerFirstColumn; ownerColumn < ownerFirstColumn + 2;
                     ++ownerColumn) {
                    const float* ownerElevation = grid.sample(ownerRow, ownerColumn);
                    if (ownerElevation && std::isfinite(*ownerElevation) &&
                        *ownerElevation < 0.0F) {
                        candidate.ownerHasCoarseOcean = true;
                        break;
                    }
                }
            }
            candidates.push_back(candidate);
        }
    }
    const auto rank = [](const Candidate& left, const Candidate& right) {
        if (left.fullyLand != right.fullyLand) return left.fullyLand;
        if (left.neighborhoodFloorMeters != right.neighborhoodFloorMeters) {
            return left.neighborhoodFloorMeters > right.neighborhoodFloorMeters;
        }
        if (left.centerElevationMeters != right.centerElevationMeters) {
            return left.centerElevationMeters > right.centerElevationMeters;
        }
        if (left.distance != right.distance) return left.distance < right.distance;
        return left.coordinate < right.coordinate;
    };
    const auto rankOceanBacked = [](const Candidate& left, const Candidate& right) {
        if (left.centerElevationMeters != right.centerElevationMeters) {
            return left.centerElevationMeters > right.centerElevationMeters;
        }
        if (left.distance != right.distance) return left.distance < right.distance;
        return left.coordinate < right.coordinate;
    };
    std::map<V4SpawnHydrologyOwner, Candidate> bestByOwner;
    for (const Candidate& candidate : candidates) {
        const auto [found, inserted] = bestByOwner.try_emplace(candidate.owner, candidate);
        if (inserted) continue;
        const bool candidatePreferred = candidate.ownerHasCoarseOcean
                                            ? rankOceanBacked(candidate, found->second)
                                            : rank(candidate, found->second);
        if (candidatePreferred) found->second = candidate;
    }
    for (const auto& [owner, candidate] : bestByOwner) {
        static_cast<void>(owner);
        if (candidate.ownerHasCoarseOcean)
            oceanBacked.push_back(candidate);
        else if (candidate.fullyLand)
            inland.push_back(candidate);
        else
            coastal.push_back(candidate);
    }
    std::ranges::sort(oceanBacked, rankOceanBacked);
    std::ranges::sort(inland, rank);
    std::ranges::sort(coastal, rank);
    if (ordinal < oceanBacked.size()) return oceanBacked[ordinal].coordinate;
    const size_t inlandOrdinal = static_cast<size_t>(ordinal) - oceanBacked.size();
    if (inlandOrdinal < inland.size()) return inland[inlandOrdinal].coordinate;
    // Do not turn an inland preference into a false "no land" result. The
    // fallback is still raw learned land only, and final authority, canonical
    // hydrology, and exact collision separately reject unsafe coastal cells.
    const size_t coastalOrdinal = inlandOrdinal - inland.size();
    if (coastalOrdinal < coastal.size()) return coastal[coastalOrdinal].coordinate;
    return std::nullopt;
}

worldgen::learned::AuthorityResult<bool> prepareAcceptedV4SpawnAuthority(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
    worldgen::MacroGenerationSampler& sampler, Vec3 resolvedCandidate,
    std::span<const V4SpawnAlignedCandidate> footprint) {
    using namespace worldgen::learned;
    if (!finalContext || !std::isfinite(resolvedCandidate.x) ||
        !std::isfinite(resolvedCandidate.z) || footprint.empty()) {
        return AuthorityResult<bool>::failed({
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "The accepted safe-spawn local authority is invalid",
            .retriable = false,
        });
    }

    std::vector<ColumnPos> columns;
    columns.reserve(footprint.size());
    for (const V4SpawnAlignedCandidate point : footprint)
        columns.emplace_back(point.worldX, point.worldZ);
    if (!sampler.nativeHydrologyDryFootprintContains(columns)) {
        return AuthorityResult<bool>::failed({
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "The accepted safe-spawn local dry certificate is incomplete",
            .retriable = false,
        });
    }
    // Owner preparation and the installed local certificate already used
    // FINAL learned authority. Wider exact terrain and hydrology are ordinary
    // streaming work. Making them part of dry-land selection can walk a lake
    // or wetland component through many owners before the World even exists.
    return AuthorityResult<bool>::ready(true);
}

std::optional<Vec3> inlandSpawnCandidate(const worldgen::learned::PhysicalTerrainGrid& grid,
                                         int64_t originWorldX, int64_t originWorldZ) {
    using namespace worldgen::learned;
    // The final exact spawn band reaches six chunks in every direction. Keep
    // the candidate 128 blocks inside its page. The learned-field selector
    // only chooses where to try: canonical water and exact collision decide
    // whether the candidate is legal. It must therefore prefer high inland
    // terrain without turning that preference into a false no-land result.
    constexpr int64_t PAGE_EDGE_MARGIN = 32;
    struct CandidateTier {
        int64_t nativeStride = 1;
        double minimumElevationMeters = 0.0;
        double maximumNeighborSpanMeters = std::numeric_limits<double>::infinity();
        double preferredElevationMeters = 0.0;
        double elevationPenalty = 0.0;
        double roughnessPenalty = 0.0;
    };
    // Tier one preserves the original inland preference. The second and
    // third tiers deliberately inspect every native sample: a final dry
    // coastal plain or narrow bench is still a valid proposal, and the
    // canonical water screen below rejects any wet one before exact work.
    constexpr std::array<CandidateTier, 3> CANDIDATE_TIERS{{
        {.nativeStride = 4,
         .minimumElevationMeters = 120.0,
         .maximumNeighborSpanMeters = 72.0,
         .preferredElevationMeters = 420.0,
         .elevationPenalty = 8.0,
         .roughnessPenalty = 64.0},
        {.nativeStride = 1,
         .minimumElevationMeters = 8.0,
         .maximumNeighborSpanMeters = 32.0,
         .preferredElevationMeters = 96.0,
         .elevationPenalty = 4.0,
         .roughnessPenalty = 96.0},
        {.nativeStride = 1,
         .minimumElevationMeters = 0.0,
         .maximumNeighborSpanMeters = std::numeric_limits<double>::infinity(),
         .preferredElevationMeters = 32.0,
         .elevationPenalty = 1.0,
         .roughnessPenalty = 128.0},
    }};

    struct Candidate {
        int64_t row = 0;
        int64_t column = 0;
        double elevationMeters = 0.0;
        V4SpawnPlacementAuthorityCost authorityCost;
        double score = std::numeric_limits<double>::infinity();
    };
    const int64_t rowStart = grid.region.rowBegin + PAGE_EDGE_MARGIN;
    const int64_t rowEnd = grid.region.rowEnd - PAGE_EDGE_MARGIN;
    const int64_t columnStart = grid.region.columnBegin + PAGE_EDGE_MARGIN;
    const int64_t columnEnd = grid.region.columnEnd - PAGE_EDGE_MARGIN;
    if (rowStart >= rowEnd || columnStart >= columnEnd) return std::nullopt;

    for (const CandidateTier& tier : CANDIDATE_TIERS) {
        std::optional<Candidate> best;
        for (int64_t row = rowStart; row < rowEnd; row += tier.nativeStride) {
            for (int64_t column = columnStart; column < columnEnd; column += tier.nativeStride) {
                const PhysicalTerrainSample* center = grid.sample(row, column);
                const PhysicalTerrainSample* north = grid.sample(row - 1, column);
                const PhysicalTerrainSample* south = grid.sample(row + 1, column);
                const PhysicalTerrainSample* west = grid.sample(row, column - 1);
                const PhysicalTerrainSample* east = grid.sample(row, column + 1);
                if (!center || !north || !south || !west || !east ||
                    !std::isfinite(center->elevationMeters) ||
                    center->elevationMeters <= tier.minimumElevationMeters) {
                    continue;
                }
                const double low = std::min({center->elevationMeters, north->elevationMeters,
                                             south->elevationMeters, west->elevationMeters,
                                             east->elevationMeters});
                const double high = std::max({center->elevationMeters, north->elevationMeters,
                                              south->elevationMeters, west->elevationMeters,
                                              east->elevationMeters});
                if (!std::isfinite(low) || !std::isfinite(high) ||
                    high - low > tier.maximumNeighborSpanMeters) {
                    continue;
                }
                const std::optional<int64_t> candidateWorldBaseX =
                    checkedMultiply(column, MODEL_BLOCK_SCALE);
                const std::optional<int64_t> candidateWorldBaseZ =
                    checkedMultiply(row, MODEL_BLOCK_SCALE);
                const std::optional<int64_t> candidateWorldX =
                    candidateWorldBaseX ? checkedAdd(*candidateWorldBaseX, 2) : std::nullopt;
                const std::optional<int64_t> candidateWorldZ =
                    candidateWorldBaseZ ? checkedAdd(*candidateWorldBaseZ, 2) : std::nullopt;
                if (!candidateWorldX || !candidateWorldZ) continue;
                const std::optional<V4SpawnPlacementAuthorityCost> authorityCost =
                    v4SpawnPlacementAuthorityCost(*candidateWorldX, *candidateWorldZ);
                if (!authorityCost) continue;
                const long double worldX = static_cast<long double>(*candidateWorldX);
                const long double worldZ = static_cast<long double>(*candidateWorldZ);
                const long double deltaX = worldX - static_cast<long double>(originWorldX);
                const long double deltaZ = worldZ - static_cast<long double>(originWorldZ);
                const long double distancePenalty = (deltaX * deltaX + deltaZ * deltaZ) / 1024.0L;
                const double elevationPenalty =
                    std::abs(center->elevationMeters - tier.preferredElevationMeters) *
                    tier.elevationPenalty;
                const double roughnessPenalty = (high - low) * tier.roughnessPenalty;
                const double score =
                    static_cast<double>(distancePenalty) + elevationPenalty + roughnessPenalty;
                // Decoder work for the protected FINAL closure dominates a
                // cold entry. Keep suitability tiers authoritative, then
                // minimize that direct owner footprint and maximize reuse of
                // exact-spawn owners before the existing page and terrain
                // score tie-breaks.
                const bool sameAuthorityCost = best && *authorityCost == best->authorityCost;
                if (!best ||
                    v4SpawnPlacementAuthorityPreferred(*authorityCost, best->authorityCost) ||
                    (sameAuthorityCost && score < best->score) ||
                    (sameAuthorityCost && score == best->score &&
                     std::pair{row, column} < std::pair{best->row, best->column})) {
                    best =
                        Candidate{.row = row,
                                  .column = column,
                                  .elevationMeters = center->elevationMeters,
                                  .authorityCost = *authorityCost,
                                  .score = score};
                }
            }
        }
        if (best) {
            const std::optional<int64_t> worldX = checkedMultiply(best->column, MODEL_BLOCK_SCALE);
            const std::optional<int64_t> worldZ = checkedMultiply(best->row, MODEL_BLOCK_SCALE);
            if (!worldX || !worldZ) return std::nullopt;
            const double terrainY = learnedElevationMetersToWorldHeight(best->elevationMeters);
            const float spawnY =
                static_cast<float>(std::clamp(terrainY + 8.0, static_cast<double>(WORLD_MIN_Y + 2),
                                              static_cast<double>(WORLD_MAX_Y - 2)));
            return Vec3{static_cast<float>(*worldX) + 2.5F, spawnY,
                        static_cast<float>(*worldZ) + 2.5F};
        }
    }
    return std::nullopt;
}

constexpr double V4_SPAWN_LOCAL_DRY_MAX_HEIGHT_SPAN_BLOCKS = 8.0;
constexpr double V4_SPAWN_LOCAL_DRY_MAX_AXIS_DELTA_BLOCKS = 4.0;
static_assert(V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS == worldgen::NATIVE_HYDROLOGY_RASTER_SPACING);

} // namespace

std::optional<V4SpawnPlacementAuthorityCost>
v4SpawnPlacementAuthorityCost(int64_t worldX, int64_t worldZ) noexcept {
    const std::optional<V4SpawnAuthorityAxisBounds> exactX =
        v4SpawnAuthorityAxisBounds(worldX, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    const std::optional<V4SpawnAuthorityAxisBounds> exactZ =
        v4SpawnAuthorityAxisBounds(worldZ, COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    if (!exactX || !exactZ) return std::nullopt;

    const ColumnPos protectedAnchor = farTerrainProtectedNearAnchor(worldX, worldZ);
    constexpr int64_t OUTER_RING =
        FAR_TERRAIN_PROTECTED_NEAR_STEP_SIXTEEN_DISTANCE_TILES;
    constexpr int64_t CORE_EDGE = FAR_TERRAIN_PROTECTED_NEAR_CORE_EDGE_TILES;
    constexpr int64_t TILE_EDGE = FAR_TERRAIN_TILE_EDGE;
    constexpr int64_t PARENT_APRON = farTerrainStepSize(FAR_TERRAIN_BASE_STEP);
    static_assert(OUTER_RING == 4);
    static_assert(CORE_EDGE == 2);
    static_assert(PARENT_APRON == worldgen::NATIVE_HYDROLOGY_TOPOLOGY_CELL_EDGE);

    // The protected region is a 2x2 core with Manhattan rings, not its 10x10
    // bounding square. At some authority-page alignments the skipped corners
    // are the only samples inside a diagonal owner, so count the actual 60
    // targets. Their complete base-parent support spans at most nine owners.
    std::array<std::pair<int64_t, int64_t>, 9> protectedOwners{};
    size_t protectedOwnerCount = 0;
    size_t protectedExactOverlapCount = 0;
    for (int64_t tileOffsetZ = -OUTER_RING; tileOffsetZ < CORE_EDGE + OUTER_RING;
         ++tileOffsetZ) {
        const int64_t distanceZ =
            tileOffsetZ < 0 ? -tileOffsetZ
                            : (tileOffsetZ >= CORE_EDGE ? tileOffsetZ - CORE_EDGE + 1 : 0);
        for (int64_t tileOffsetX = -OUTER_RING; tileOffsetX < CORE_EDGE + OUTER_RING;
             ++tileOffsetX) {
            const int64_t distanceX =
                tileOffsetX < 0
                    ? -tileOffsetX
                    : (tileOffsetX >= CORE_EDGE ? tileOffsetX - CORE_EDGE + 1 : 0);
            if (distanceX + distanceZ > OUTER_RING) continue;

            const std::optional<int64_t> tileX =
                checkedAdd(protectedAnchor.x, tileOffsetX);
            const std::optional<int64_t> tileZ =
                checkedAdd(protectedAnchor.z, tileOffsetZ);
            const std::optional<int64_t> originX =
                tileX ? checkedMultiply(*tileX, TILE_EDGE) : std::nullopt;
            const std::optional<int64_t> originZ =
                tileZ ? checkedMultiply(*tileZ, TILE_EDGE) : std::nullopt;
            const std::optional<int64_t> minimumX =
                originX ? checkedAdd(*originX, -PARENT_APRON) : std::nullopt;
            const std::optional<int64_t> minimumZ =
                originZ ? checkedAdd(*originZ, -PARENT_APRON) : std::nullopt;
            const std::optional<int64_t> maximumX =
                originX ? checkedAdd(*originX, TILE_EDGE + PARENT_APRON) : std::nullopt;
            const std::optional<int64_t> maximumZ =
                originZ ? checkedAdd(*originZ, TILE_EDGE + PARENT_APRON) : std::nullopt;
            if (!minimumX || !minimumZ || !maximumX || !maximumZ) return std::nullopt;

            const int64_t firstOwnerX = world_coord::floorDiv(
                *minimumX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t lastOwnerX = world_coord::floorDiv(
                *maximumX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t firstOwnerZ = world_coord::floorDiv(
                *minimumZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t lastOwnerZ = world_coord::floorDiv(
                *maximumZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            for (int64_t ownerZ = firstOwnerZ;; ++ownerZ) {
                for (int64_t ownerX = firstOwnerX;; ++ownerX) {
                    const std::pair owner{ownerX, ownerZ};
                    const auto existing =
                        std::find(protectedOwners.begin(),
                                  protectedOwners.begin() + protectedOwnerCount, owner);
                    if (existing == protectedOwners.begin() + protectedOwnerCount) {
                        if (protectedOwnerCount == protectedOwners.size()) return std::nullopt;
                        protectedOwners[protectedOwnerCount++] = owner;
                        if (ownerX >= exactX->firstOwner && ownerX <= exactX->lastOwner &&
                            ownerZ >= exactZ->firstOwner && ownerZ <= exactZ->lastOwner) {
                            ++protectedExactOverlapCount;
                        }
                    }
                    if (ownerX == lastOwnerX) break;
                }
                if (ownerZ == lastOwnerZ) break;
            }
        }
    }

    const auto axisWidth = [](int64_t first, int64_t last) -> __int128 {
        return last < first ? 0 : static_cast<__int128>(last) - first + 1;
    };
    const __int128 exactWidth = axisWidth(exactX->firstOwner, exactX->lastOwner);
    const __int128 exactHeight = axisWidth(exactZ->firstOwner, exactZ->lastOwner);

    const int64_t firstFinalColumn = world_coord::floorDiv(
        exactX->minimum, static_cast<int64_t>(worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastFinalColumn =
        world_coord::floorDiv(exactX->maximumExclusive - 1,
                              static_cast<int64_t>(worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t firstFinalRow = world_coord::floorDiv(
        exactZ->minimum, static_cast<int64_t>(worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const int64_t lastFinalRow =
        world_coord::floorDiv(exactZ->maximumExclusive - 1,
                              static_cast<int64_t>(worldgen::learned::AUTHORITY_PAGE_BLOCK_EDGE));
    const __int128 finalWidth = axisWidth(firstFinalColumn, lastFinalColumn);
    const __int128 finalHeight = axisWidth(firstFinalRow, lastFinalRow);

    const __int128 exactCount = exactWidth * exactHeight;
    const __int128 finalPageCount = finalWidth * finalHeight;
    constexpr __int128 MAXIMUM_SIZE = std::numeric_limits<size_t>::max();
    if (protectedOwnerCount == 0 || exactCount <= 0 || exactCount > MAXIMUM_SIZE ||
        finalPageCount <= 0 || finalPageCount > MAXIMUM_SIZE) {
        return std::nullopt;
    }
    return V4SpawnPlacementAuthorityCost{
        .protectedDirectOwnerCount = protectedOwnerCount,
        .protectedExactOverlapCount = protectedExactOverlapCount,
        .exactOwnerCount = static_cast<size_t>(exactCount),
        .finalRefinementPageCount = static_cast<size_t>(finalPageCount),
    };
}

std::vector<V4SpawnExactFootprintRow> v4ColdSpawnExactFootprintRows() {
    // Match World::rebuildActiveSet exactly: its visible columns form a disk,
    // while mesh and ColumnPlan dependencies expand each retained column with
    // square neighborhoods. Each requested plan then owns one inclusive
    // 49-by-49 hydrology raster.
    const int activeRadius =
        exactStreamingActiveSetRadiusChunks(COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS);
    std::set<std::pair<int64_t, int64_t>> planCenters;
    for (int activeZ = -activeRadius; activeZ <= activeRadius; ++activeZ) {
        for (int activeX = -activeRadius; activeX <= activeRadius; ++activeX) {
            if (!withinExactStreamingRadius(activeX, activeZ, activeRadius)) continue;
            for (int haloZ = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                 haloZ <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++haloZ) {
                for (int haloX = -EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS;
                     haloX <= EXACT_STREAMING_HORIZONTAL_MESH_HALO_CHUNKS; ++haloX) {
                    for (int apronZ = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
                         apronZ <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++apronZ) {
                        for (int apronX = -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS;
                             apronX <= EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS; ++apronX) {
                            planCenters.emplace(activeX + haloX + apronX, activeZ + haloZ + apronZ);
                        }
                    }
                }
            }
        }
    }

    std::map<int64_t, std::vector<std::pair<int64_t, int64_t>>> intervalsByRow;
    for (const auto [planX, planZ] : planCenters) {
        const int64_t planBaseX = planX * CHUNK_EDGE;
        const int64_t planBaseZ = planZ * CHUNK_EDGE;
        const int64_t minimumX = planBaseX - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS;
        const int64_t maximumX =
            minimumX + static_cast<int64_t>(COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE) - 1;
        const int64_t minimumZ = planBaseZ - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS;
        const int64_t maximumZ =
            minimumZ + static_cast<int64_t>(COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE) - 1;
        for (int64_t z = minimumZ; z <= maximumZ; ++z)
            intervalsByRow[z].emplace_back(minimumX, maximumX);
    }

    std::vector<V4SpawnExactFootprintRow> rows;
    for (auto& [z, intervals] : intervalsByRow) {
        std::ranges::sort(intervals);
        int64_t minimumX = intervals.front().first;
        int64_t maximumX = intervals.front().second;
        for (size_t index = 1; index < intervals.size(); ++index) {
            if (intervals[index].first <= maximumX + 1) {
                maximumX = std::max(maximumX, intervals[index].second);
                continue;
            }
            rows.push_back({.zOffset = z, .minimumXOffset = minimumX, .maximumXOffset = maximumX});
            minimumX = intervals[index].first;
            maximumX = intervals[index].second;
        }
        rows.push_back({.zOffset = z, .minimumXOffset = minimumX, .maximumXOffset = maximumX});
    }
    return rows;
}

std::optional<std::vector<V4SpawnAlignedCandidate>>
v4ColdSpawnExactFootprintPoints(int64_t centerChunkX, int64_t centerChunkZ) {
    const std::optional<int64_t> centerBaseX = checkedMultiply(centerChunkX, CHUNK_EDGE);
    const std::optional<int64_t> centerBaseZ = checkedMultiply(centerChunkZ, CHUNK_EDGE);
    if (!centerBaseX || !centerBaseZ) return std::nullopt;

    const std::vector<V4SpawnExactFootprintRow> rows = v4ColdSpawnExactFootprintRows();
    size_t pointCount = 0;
    for (const V4SpawnExactFootprintRow row : rows) {
        if (row.sampleCount() > std::numeric_limits<size_t>::max() - pointCount)
            return std::nullopt;
        pointCount += row.sampleCount();
    }
    if (pointCount == 0 || pointCount > worldgen::NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES)
        return std::nullopt;

    std::vector<V4SpawnAlignedCandidate> points;
    points.reserve(pointCount);
    for (const V4SpawnExactFootprintRow row : rows) {
        const std::optional<int64_t> worldZ = checkedAdd(*centerBaseZ, row.zOffset);
        const std::optional<int64_t> minimumX = checkedAdd(*centerBaseX, row.minimumXOffset);
        const std::optional<int64_t> maximumX = checkedAdd(*centerBaseX, row.maximumXOffset);
        if (!worldZ || !minimumX || !maximumX || *maximumX < *minimumX) return std::nullopt;
        for (int64_t worldX = *minimumX;; ++worldX) {
            points.push_back({.worldX = worldX, .worldZ = *worldZ});
            if (worldX == *maximumX) break;
        }
    }
    return points;
}

namespace {

constexpr int64_t V4_SPAWN_CERTIFICATION_MINIMUM_OFFSET =
    -EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS * CHUNK_EDGE - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS;
constexpr int64_t V4_SPAWN_CERTIFICATION_MAXIMUM_OFFSET =
    EXACT_STREAMING_PLAN_DEPENDENCY_APRON_CHUNKS * CHUNK_EDGE - COLUMN_PLAN_HYDROLOGY_APRON_BLOCKS +
    COLUMN_PLAN_HYDROLOGY_SAMPLE_EDGE - 1;

struct V4SpawnCertificationFootprintBounds {
    int64_t minimumX = 0;
    int64_t maximumX = 0;
    int64_t minimumZ = 0;
    int64_t maximumZ = 0;
};

std::optional<V4SpawnCertificationFootprintBounds>
v4SpawnCertificationExactFootprintBounds(int64_t centerChunkX, int64_t centerChunkZ) {
    const std::optional<int64_t> centerBaseX = checkedMultiply(centerChunkX, CHUNK_EDGE);
    const std::optional<int64_t> centerBaseZ = checkedMultiply(centerChunkZ, CHUNK_EDGE);
    const std::optional<int64_t> minimumX =
        centerBaseX ? checkedAdd(*centerBaseX, V4_SPAWN_CERTIFICATION_MINIMUM_OFFSET)
                    : std::nullopt;
    const std::optional<int64_t> maximumX =
        centerBaseX ? checkedAdd(*centerBaseX, V4_SPAWN_CERTIFICATION_MAXIMUM_OFFSET)
                    : std::nullopt;
    const std::optional<int64_t> minimumZ =
        centerBaseZ ? checkedAdd(*centerBaseZ, V4_SPAWN_CERTIFICATION_MINIMUM_OFFSET)
                    : std::nullopt;
    const std::optional<int64_t> maximumZ =
        centerBaseZ ? checkedAdd(*centerBaseZ, V4_SPAWN_CERTIFICATION_MAXIMUM_OFFSET)
                    : std::nullopt;
    if (!minimumX || !maximumX || !minimumZ || !maximumZ) return std::nullopt;
    return V4SpawnCertificationFootprintBounds{
        .minimumX = *minimumX, .maximumX = *maximumX, .minimumZ = *minimumZ, .maximumZ = *maximumZ};
}

} // namespace

std::optional<std::vector<V4SpawnAlignedCandidate>>
v4SpawnCertificationExactFootprintPoints(int64_t centerChunkX, int64_t centerChunkZ) {
    constexpr size_t EDGE = static_cast<size_t>(V4_SPAWN_CERTIFICATION_MAXIMUM_OFFSET -
                                                V4_SPAWN_CERTIFICATION_MINIMUM_OFFSET + 1);
    static_assert(EDGE == 113);
    static_assert(EDGE * EDGE <= worldgen::NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES);

    const std::optional<V4SpawnCertificationFootprintBounds> bounds =
        v4SpawnCertificationExactFootprintBounds(centerChunkX, centerChunkZ);
    if (!bounds) return std::nullopt;

    std::vector<V4SpawnAlignedCandidate> points;
    points.reserve(EDGE * EDGE);
    for (int64_t worldZ = bounds->minimumZ;; ++worldZ) {
        for (int64_t worldX = bounds->minimumX;; ++worldX) {
            points.push_back({.worldX = worldX, .worldZ = worldZ});
            if (worldX == bounds->maximumX) break;
        }
        if (worldZ == bounds->maximumZ) break;
    }
    return points;
}

std::vector<V4SpawnAlignedCandidate>
v4RankCertifiedDrySpawnCandidates(Vec3 requestedCandidate, int64_t originWorldX,
                                  int64_t originWorldZ, int sampleWidth, int sampleHeight,
                                  std::span<const uint8_t> certified) {
    constexpr int64_t BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
    constexpr int64_t SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    if (!std::isfinite(requestedCandidate.x) || !std::isfinite(requestedCandidate.z) ||
        sampleWidth < BUFFER * 2 + 1 || sampleHeight < BUFFER * 2 + 1 ||
        originWorldX % SPACING != 0 || originWorldZ % SPACING != 0 ||
        certified.size() != static_cast<size_t>(sampleWidth) * static_cast<size_t>(sampleHeight)) {
        return {};
    }

    struct RankedCandidate {
        V4SpawnAlignedCandidate candidate;
        V4SpawnPlacementAuthorityCost authorityCost;
        long double distanceSquared = 0.0L;
        uint64_t proximityTier = 0;
    };
    // Relocation remains local before it optimizes cold authority work. A
    // 64-block squared-distance band preserves the established owner-edge
    // bound while allowing adjacent chunks around a protected handoff seam
    // to retain the placement that opens fewer FINAL hydrology owners.
    constexpr long double PROXIMITY_TIER_SQUARED = 64.0L * 64.0L;
    const auto proximityTier = [](long double distanceSquared) noexcept {
        if (!std::isfinite(distanceSquared) || distanceSquared < 0.0L) {
            return std::numeric_limits<uint64_t>::max();
        }
        const long double tier = distanceSquared / PROXIMITY_TIER_SQUARED;
        constexpr long double MAXIMUM_TIER =
            static_cast<long double>(std::numeric_limits<uint64_t>::max());
        return tier >= MAXIMUM_TIER ? std::numeric_limits<uint64_t>::max()
                                   : static_cast<uint64_t>(tier);
    };
    std::map<std::pair<int64_t, int64_t>, RankedCandidate> bestByChunk;
    for (int z = static_cast<int>(BUFFER); z < sampleHeight - BUFFER; ++z) {
        for (int x = static_cast<int>(BUFFER); x < sampleWidth - BUFFER; ++x) {
            bool localBufferCertified = true;
            for (int offsetZ = -static_cast<int>(BUFFER);
                 offsetZ <= static_cast<int>(BUFFER) && localBufferCertified; ++offsetZ) {
                for (int offsetX = -static_cast<int>(BUFFER); offsetX <= static_cast<int>(BUFFER);
                     ++offsetX) {
                    const size_t sample =
                        static_cast<size_t>(z + offsetZ) * static_cast<size_t>(sampleWidth) +
                        static_cast<size_t>(x + offsetX);
                    if (certified[sample] == 0) {
                        localBufferCertified = false;
                        break;
                    }
                }
            }
            if (!localBufferCertified) continue;

            const std::optional<int64_t> xOffset = checkedMultiply(x, SPACING);
            const std::optional<int64_t> zOffset = checkedMultiply(z, SPACING);
            const std::optional<int64_t> worldX =
                xOffset ? checkedAdd(originWorldX, *xOffset) : std::nullopt;
            const std::optional<int64_t> worldZ =
                zOffset ? checkedAdd(originWorldZ, *zOffset) : std::nullopt;
            if (!worldX || !worldZ) return {};
            const long double deltaX =
                static_cast<long double>(*worldX) + 0.5L - requestedCandidate.x;
            const long double deltaZ =
                static_cast<long double>(*worldZ) + 0.5L - requestedCandidate.z;
            const long double distanceSquared = deltaX * deltaX + deltaZ * deltaZ;
            const RankedCandidate candidate{
                .candidate = {.worldX = *worldX, .worldZ = *worldZ},
                .distanceSquared = distanceSquared,
                .proximityTier = proximityTier(distanceSquared),
            };
            const std::pair chunk{Chunk::worldToChunk(*worldX), Chunk::worldToChunk(*worldZ)};
            const auto [found, inserted] = bestByChunk.try_emplace(chunk, candidate);
            if (!inserted && (candidate.distanceSquared < found->second.distanceSquared ||
                              (candidate.distanceSquared == found->second.distanceSquared &&
                               candidate.candidate < found->second.candidate))) {
                found->second = candidate;
            }
        }
    }
    std::vector<RankedCandidate> ranked;
    ranked.reserve(bestByChunk.size());
    for (const auto& [chunk, candidate] : bestByChunk) {
        static_cast<void>(chunk);
        const std::optional<V4SpawnPlacementAuthorityCost> authorityCost =
            v4SpawnPlacementAuthorityCost(candidate.candidate.worldX, candidate.candidate.worldZ);
        if (!authorityCost) continue;
        RankedCandidate rankedCandidate = candidate;
        rankedCandidate.authorityCost = *authorityCost;
        ranked.push_back(std::move(rankedCandidate));
    }
    std::ranges::sort(ranked, [](const RankedCandidate& left, const RankedCandidate& right) {
        if (left.proximityTier != right.proximityTier)
            return left.proximityTier < right.proximityTier;
        if (left.authorityCost != right.authorityCost) {
            return v4SpawnPlacementAuthorityPreferred(left.authorityCost, right.authorityCost);
        }
        if (left.distanceSquared != right.distanceSquared)
            return left.distanceSquared < right.distanceSquared;
        return std::pair{left.candidate.worldZ, left.candidate.worldX} <
               std::pair{right.candidate.worldZ, right.candidate.worldX};
    });

    std::vector<V4SpawnAlignedCandidate> candidates;
    candidates.reserve(ranked.size());
    for (const RankedCandidate& candidate : ranked)
        candidates.push_back(candidate.candidate);
    return candidates;
}

bool v4SpawnCandidateHasCanonicalSurfaceWater(const worldgen::HydrologySample& hydrology) noexcept {
    // The native solver emits at least an eighth of a block of supported
    // water, so this tolerance absorbs only floating-point reconstruction
    // noise. It cannot admit a real standing or flowing water column.
    constexpr double STAGE_ABOVE_BED_EPSILON = 0.01;
    if (!std::isfinite(hydrology.surfaceElevation) || !std::isfinite(hydrology.waterSurface))
        return true;
    return hydrology.ocean || hydrology.lake || hydrology.river || hydrology.wetland ||
           hydrology.waterfall || hydrology.delta ||
           hydrology.transitionOwnerKind != worldgen::WaterTransitionKind::NONE ||
           hydrology.waterSurface > hydrology.surfaceElevation + STAGE_ABOVE_BED_EPSILON;
}

std::optional<Vec3>
v4SelectLocalDrySpawnCandidate(Vec3 requestedCandidate, int64_t originWorldX, int64_t originWorldZ,
                               int sampleWidth, int sampleHeight,
                               std::span<const worldgen::HydrologySample> samples) noexcept {
    constexpr int BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
    constexpr int64_t SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
    if (!std::isfinite(requestedCandidate.x) || !std::isfinite(requestedCandidate.z) ||
        sampleWidth < 2 * BUFFER + 1 || sampleHeight < 2 * BUFFER + 1 ||
        samples.size() != static_cast<size_t>(sampleWidth) * static_cast<size_t>(sampleHeight)) {
        return std::nullopt;
    }

    struct Candidate {
        Vec3 position{};
        double score = std::numeric_limits<double>::infinity();
        int64_t worldX = 0;
        int64_t worldZ = 0;
    };
    std::optional<Candidate> best;
    const auto sampleAt = [&](int x, int z) -> const worldgen::HydrologySample& {
        return samples[static_cast<size_t>(z) * static_cast<size_t>(sampleWidth) +
                       static_cast<size_t>(x)];
    };
    for (int z = BUFFER; z < sampleHeight - BUFFER; ++z) {
        for (int x = BUFFER; x < sampleWidth - BUFFER; ++x) {
            const worldgen::HydrologySample& center = sampleAt(x, z);
            if (v4SpawnCandidateHasCanonicalSurfaceWater(center) ||
                !std::isfinite(center.surfaceElevation)) {
                continue;
            }

            double minimumHeight = center.surfaceElevation;
            double maximumHeight = center.surfaceElevation;
            bool bufferDryAndFinite = true;
            for (int bufferZ = z - BUFFER; bufferZ <= z + BUFFER && bufferDryAndFinite; ++bufferZ) {
                for (int bufferX = x - BUFFER; bufferX <= x + BUFFER; ++bufferX) {
                    const worldgen::HydrologySample& nearby = sampleAt(bufferX, bufferZ);
                    if (v4SpawnCandidateHasCanonicalSurfaceWater(nearby) ||
                        !std::isfinite(nearby.surfaceElevation)) {
                        bufferDryAndFinite = false;
                        break;
                    }
                    minimumHeight = std::min(minimumHeight, nearby.surfaceElevation);
                    maximumHeight = std::max(maximumHeight, nearby.surfaceElevation);
                }
            }
            if (!bufferDryAndFinite ||
                maximumHeight - minimumHeight > V4_SPAWN_LOCAL_DRY_MAX_HEIGHT_SPAN_BLOCKS) {
                continue;
            }

            const double eastWestDelta = std::abs(sampleAt(x + BUFFER, z).surfaceElevation -
                                                  sampleAt(x - BUFFER, z).surfaceElevation);
            const double northSouthDelta = std::abs(sampleAt(x, z + BUFFER).surfaceElevation -
                                                    sampleAt(x, z - BUFFER).surfaceElevation);
            if (!std::isfinite(eastWestDelta) || !std::isfinite(northSouthDelta) ||
                eastWestDelta > V4_SPAWN_LOCAL_DRY_MAX_AXIS_DELTA_BLOCKS ||
                northSouthDelta > V4_SPAWN_LOCAL_DRY_MAX_AXIS_DELTA_BLOCKS) {
                continue;
            }

            const __int128 worldXWide =
                static_cast<__int128>(originWorldX) + static_cast<__int128>(x) * SPACING;
            const __int128 worldZWide =
                static_cast<__int128>(originWorldZ) + static_cast<__int128>(z) * SPACING;
            if (worldXWide < std::numeric_limits<int64_t>::min() ||
                worldXWide > std::numeric_limits<int64_t>::max() ||
                worldZWide < std::numeric_limits<int64_t>::min() ||
                worldZWide > std::numeric_limits<int64_t>::max()) {
                continue;
            }
            const int64_t worldX = static_cast<int64_t>(worldXWide);
            const int64_t worldZ = static_cast<int64_t>(worldZWide);
            const double feet = std::ceil(center.surfaceElevation);
            if (!std::isfinite(feet) || feet < static_cast<double>(WORLD_MIN_Y + 1) ||
                feet + 1.0 > static_cast<double>(WORLD_MAX_Y)) {
                continue;
            }

            const long double deltaX = static_cast<long double>(worldX) - requestedCandidate.x;
            const long double deltaZ = static_cast<long double>(worldZ) - requestedCandidate.z;
            const double slopePenalty =
                std::max(eastWestDelta, northSouthDelta) / (2.0 * static_cast<double>(SPACING));
            const double score = static_cast<double>(deltaX * deltaX + deltaZ * deltaZ) +
                                 slopePenalty * 256.0 + (maximumHeight - minimumHeight) * 16.0;
            if (!best || score < best->score ||
                (score == best->score &&
                 std::pair{worldZ, worldX} < std::pair{best->worldZ, best->worldX})) {
                best = Candidate{
                    .position = {static_cast<float>(worldX) + 0.5F,
                                 static_cast<float>(feet) + 0.05F,
                                 static_cast<float>(worldZ) + 0.5F},
                    .score = score,
                    .worldX = worldX,
                    .worldZ = worldZ,
                };
            }
        }
    }
    return best ? std::optional<Vec3>{best->position} : std::nullopt;
}

class V4SpawnWaterScreen::Impl {
public:
    explicit Impl(V4SpawnWaterScreenTiming timing)
        : timing_(boundedTiming(timing))
        , worker_([this](std::stop_token stopToken) { run(stopToken); }) {}

    ~Impl() {
        {
            std::lock_guard lock(mutex_);
            request_.reset();
            ++requestEpoch_;
            worker_.request_stop();
        }
        wake_.notify_all();
    }

    V4SpawnWaterScreenResult
    screen(const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
           Vec3 candidate) {
        using namespace worldgen::learned;
        if (!finalContext) {
            return {
                .status = V4SpawnWaterScreenStatus::Failed,
                .failure =
                    GenerationFailure{
                        .code = GenerationFailureCode::BACKEND_UNAVAILABLE,
                        .message = "The learned terrain authority is unavailable for safe-spawn "
                                   "water screening",
                        .retriable = true,
                    },
            };
        }

        const std::optional<int64_t> worldX = checkedFloorSpawnCoordinate(candidate.x);
        const std::optional<int64_t> worldZ = checkedFloorSpawnCoordinate(candidate.z);
        if (!worldX || !worldZ) {
            reset();
            return {
                .status = V4SpawnWaterScreenStatus::Failed,
                .failure =
                    GenerationFailure{
                        .code = GenerationFailureCode::INVALID_REQUEST,
                        .message = "The safe-spawn water-screen candidate is outside the supported "
                                   "coordinate range",
                        .retriable = false,
                    },
            };
        }
        candidate.x = candidate.x == 0.0F ? 0.0F : candidate.x;
        candidate.z = candidate.z == 0.0F ? 0.0F : candidate.z;
        std::lock_guard lock(mutex_);
        const bool sameRequest = request_ && request_->context.get() == finalContext.get() &&
                                 request_->worldX == *worldX && request_->worldZ == *worldZ &&
                                 request_->candidate.x == candidate.x &&
                                 request_->candidate.z == candidate.z;
        if (!sameRequest) {
            request_ = Request{.context = finalContext,
                               .candidate = candidate,
                               .worldX = *worldX,
                               .worldZ = *worldZ,
                               .startedAt = std::chrono::steady_clock::now(),
                               .epoch = ++requestEpoch_};
            result_ = {};
            wake_.notify_all();
        }
        return result_;
    }

    void reset() {
        std::lock_guard lock(mutex_);
        request_.reset();
        ++requestEpoch_;
        result_ = {};
        wake_.notify_all();
    }

private:
    struct Request {
        std::shared_ptr<worldgen::learned::WorldGenerationContext> context;
        Vec3 candidate{};
        int64_t worldX = 0;
        int64_t worldZ = 0;
        std::chrono::steady_clock::time_point startedAt{};
        uint64_t epoch = 0;
    };

    struct CertifiedOwnerMask {
        int64_t originWorldX = 0;
        int64_t originWorldZ = 0;
        int sampleWidth = 0;
        int sampleHeight = 0;
        std::vector<uint8_t> certified;
    };

    struct ScreenDeadlineExceeded {};
    struct ScreenRequestObsolete {};

    [[nodiscard]] bool requestIsCurrent(uint64_t epoch) const {
        std::lock_guard lock(mutex_);
        return request_ && request_->epoch == epoch;
    }

    void throwIfRequestCannotContinue(std::stop_token stopToken, uint64_t epoch,
                                      std::chrono::steady_clock::time_point deadline) const {
        if (stopToken.stop_requested() || !requestIsCurrent(epoch)) throw ScreenRequestObsolete{};
        throwIfScreenDeadlineExceeded(deadline);
    }

    void publish(uint64_t epoch, V4SpawnWaterScreenResult result) {
        std::lock_guard lock(mutex_);
        if (!request_ || request_->epoch != epoch) return;
        result_ = std::move(result);
    }

    struct ProgressSnapshot {
        size_t authorityActiveBuilds = 0;
        size_t authorityQueuedBuilds = 0;
        size_t authorityActivePublications = 0;
        size_t authorityQueuedPublications = 0;
        uint64_t authorityBatches = 0;
        uint64_t authorityBatchedPages = 0;
        uint64_t authorityDiskLoads = 0;
        uint64_t authorityPublicationWrites = 0;
        uint64_t authorityRepairs = 0;
        size_t hydrologyActiveBuilds = 0;
        uint64_t hydrologyBuilds = 0;
        uint64_t hydrologyPersistedLoads = 0;
        uint64_t hydrologyPersistedWrites = 0;
        uint64_t hydrologyPersistedRepairs = 0;

        // Completion counters only. Repeated deferred replies and failed
        // native-page attempts are intentionally absent: neither proves that
        // a candidate is getting closer to a safe answer.
        [[nodiscard]] bool completedWorkSince(const ProgressSnapshot& other) const noexcept {
            return authorityBatches != other.authorityBatches ||
                   authorityBatchedPages != other.authorityBatchedPages ||
                   authorityDiskLoads != other.authorityDiskLoads ||
                   authorityPublicationWrites != other.authorityPublicationWrites ||
                   authorityRepairs != other.authorityRepairs ||
                   hydrologyBuilds != other.hydrologyBuilds ||
                   hydrologyPersistedLoads != other.hydrologyPersistedLoads ||
                   hydrologyPersistedWrites != other.hydrologyPersistedWrites ||
                   hydrologyPersistedRepairs != other.hydrologyPersistedRepairs;
        }

        // Admission and running-state changes count once as observable
        // progress, but a static active count does not repeatedly extend the
        // absolute request deadline below.
        [[nodiscard]] bool workStateChangedSince(const ProgressSnapshot& other) const noexcept {
            return authorityActiveBuilds != other.authorityActiveBuilds ||
                   authorityQueuedBuilds != other.authorityQueuedBuilds ||
                   authorityActivePublications != other.authorityActivePublications ||
                   authorityQueuedPublications != other.authorityQueuedPublications ||
                   hydrologyActiveBuilds != other.hydrologyActiveBuilds;
        }

        [[nodiscard]] bool hasAdmittedOrRunningWork() const noexcept {
            return authorityActiveBuilds != 0 || authorityQueuedBuilds != 0 ||
                   authorityActivePublications != 0 || authorityQueuedPublications != 0 ||
                   hydrologyActiveBuilds != 0;
        }
    };

    static V4SpawnWaterScreenTiming boundedTiming(V4SpawnWaterScreenTiming timing) noexcept {
        constexpr std::chrono::milliseconds MINIMUM_INTERVAL{1};
        constexpr std::chrono::milliseconds MAXIMUM_INTERVAL{100};
        timing.retryInterval = std::clamp(timing.retryInterval, MINIMUM_INTERVAL, MAXIMUM_INTERVAL);
        timing.noProgressTimeout = std::clamp(timing.noProgressTimeout, MINIMUM_INTERVAL,
                                              V4_SPAWN_WATER_SCREEN_NO_PROGRESS_TIMEOUT);
        timing.activeWorkTimeout = std::clamp(timing.activeWorkTimeout, timing.noProgressTimeout,
                                              V4_SPAWN_WATER_SCREEN_ACTIVE_WORK_TIMEOUT);
        return timing;
    }

    static ProgressSnapshot
    progressSnapshot(const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context) {
        const worldgen::learned::WorldGenerationMetrics metrics = context->metrics();
        const worldgen::learned::TerrainAuthorityCacheMetrics& authority = metrics.authorityCache;
        const worldgen::NativeHydrologyCacheMetrics hydrology =
            context->nativeHydrologyRouter()->cacheMetrics();
        return {
            .authorityActiveBuilds = authority.activeBuilds,
            .authorityQueuedBuilds = authority.queuedBuilds,
            .authorityActivePublications = authority.activePublications,
            .authorityQueuedPublications = authority.queuedPublications,
            .authorityBatches = authority.batches,
            .authorityBatchedPages = authority.batchedPages,
            .authorityDiskLoads = authority.diskLoads,
            .authorityPublicationWrites = authority.publicationWrites,
            .authorityRepairs = authority.repairs,
            .hydrologyActiveBuilds = hydrology.activeBuilds,
            .hydrologyBuilds = hydrology.builds,
            .hydrologyPersistedLoads = hydrology.persistedLoads,
            .hydrologyPersistedWrites = hydrology.persistedWrites,
            .hydrologyPersistedRepairs = hydrology.persistedRepairs,
        };
    }

    [[nodiscard]] static bool
    screenDeadlineExceeded(std::chrono::steady_clock::time_point deadline) noexcept {
        return std::chrono::steady_clock::now() >= deadline;
    }

    static void throwIfScreenDeadlineExceeded(std::chrono::steady_clock::time_point deadline) {
        if (screenDeadlineExceeded(deadline)) throw ScreenDeadlineExceeded{};
    }

    [[nodiscard]] V4SpawnWaterScreenResult screenDeadlineFailure() const {
        return {
            .status = V4SpawnWaterScreenStatus::Failed,
            .failure =
                worldgen::learned::GenerationFailure{
                    .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                    .message = "Canonical safe-spawn water screening did not complete within its "
                               "absolute " +
                               std::to_string(timing_.activeWorkTimeout.count()) +
                               " millisecond request deadline",
                    .retriable = true,
                },
        };
    }

    [[nodiscard]] std::optional<CertifiedOwnerMask>
    scanStrictOwner(worldgen::MacroGenerationSampler& sampler, int64_t ownerX, int64_t ownerZ,
                    std::stop_token stopToken, uint64_t requestEpoch,
                    std::chrono::steady_clock::time_point deadline) const {
        constexpr int64_t EDGE = worldgen::NATIVE_HYDROLOGY_PAGE_EDGE;
        constexpr int64_t MARGIN = worldgen::NATIVE_HYDROLOGY_HANDOFF_BLOCKS + 1;
        constexpr int64_t SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;

        throwIfScreenDeadlineExceeded(deadline);

        const std::optional<int64_t> ownerOriginX = checkedMultiply(ownerX, EDGE);
        const std::optional<int64_t> ownerOriginZ = checkedMultiply(ownerZ, EDGE);
        const std::optional<int64_t> ownerEndX =
            ownerOriginX ? checkedAdd(*ownerOriginX, EDGE) : std::nullopt;
        const std::optional<int64_t> ownerEndZ =
            ownerOriginZ ? checkedAdd(*ownerOriginZ, EDGE) : std::nullopt;
        const std::optional<int64_t> interiorMinimumX =
            ownerOriginX ? checkedAdd(*ownerOriginX, MARGIN) : std::nullopt;
        const std::optional<int64_t> interiorMinimumZ =
            ownerOriginZ ? checkedAdd(*ownerOriginZ, MARGIN) : std::nullopt;
        const std::optional<int64_t> interiorMaximumX =
            ownerEndX ? checkedAdd(*ownerEndX, -MARGIN) : std::nullopt;
        const std::optional<int64_t> interiorMaximumZ =
            ownerEndZ ? checkedAdd(*ownerEndZ, -MARGIN) : std::nullopt;
        const std::optional<int64_t> interiorOriginX =
            interiorMinimumX ? firstGloballyAlignedSpawnCoordinate(*interiorMinimumX)
                             : std::nullopt;
        const std::optional<int64_t> interiorOriginZ =
            interiorMinimumZ ? firstGloballyAlignedSpawnCoordinate(*interiorMinimumZ)
                             : std::nullopt;
        const std::optional<int64_t> interiorLastX =
            interiorMaximumX
                ? checkedMultiply(world_coord::floorDiv(*interiorMaximumX, SPACING), SPACING)
                : std::nullopt;
        const std::optional<int64_t> interiorLastZ =
            interiorMaximumZ
                ? checkedMultiply(world_coord::floorDiv(*interiorMaximumZ, SPACING), SPACING)
                : std::nullopt;
        if (!interiorOriginX || !interiorOriginZ || !interiorLastX || !interiorLastZ ||
            *interiorOriginX > *interiorLastX || *interiorOriginZ > *interiorLastZ) {
            return std::nullopt;
        }
        const int64_t sampleWidth64 = (*interiorLastX - *interiorOriginX) / SPACING + 1;
        const int64_t sampleHeight64 = (*interiorLastZ - *interiorOriginZ) / SPACING + 1;
        if (sampleWidth64 <= 0 || sampleHeight64 <= 0 ||
            sampleWidth64 > std::numeric_limits<int>::max() ||
            sampleHeight64 > std::numeric_limits<int>::max()) {
            return std::nullopt;
        }

        CertifiedOwnerMask scan{
            .originWorldX = *interiorOriginX,
            .originWorldZ = *interiorOriginZ,
            .sampleWidth = static_cast<int>(sampleWidth64),
            .sampleHeight = static_cast<int>(sampleHeight64),
            .certified = std::vector<uint8_t>(static_cast<size_t>(sampleWidth64) *
                                              static_cast<size_t>(sampleHeight64)),
        };
        throwIfScreenDeadlineExceeded(deadline);
        constexpr size_t BATCH_SIZE = worldgen::NATIVE_HYDROLOGY_MAX_DRY_CERTIFICATE_SAMPLES;
        const size_t batchCount = (scan.certified.size() + BATCH_SIZE - 1) / BATCH_SIZE;
        const size_t hardwareThreads = std::max<size_t>(1, std::thread::hardware_concurrency());
        const size_t workerCount =
            std::min({batchCount, hardwareThreads, worldgen::NATIVE_HYDROLOGY_MAX_PARALLEL_BUILDS});
        std::atomic<size_t> nextBatch{0};
        std::atomic<bool> canceled{false};
        std::mutex failureMutex;
        std::exception_ptr failure;
        const auto worker = [&] {
            std::vector<ColumnPos> positions;
            std::vector<uint8_t> batchMask;
            positions.reserve(BATCH_SIZE);
            batchMask.reserve(BATCH_SIZE);
            try {
                while (!canceled.load(std::memory_order_acquire)) {
                    const size_t batch = nextBatch.fetch_add(1, std::memory_order_relaxed);
                    if (batch >= batchCount) return;
                    if (stopToken.stop_requested() || !requestIsCurrent(requestEpoch) ||
                        screenDeadlineExceeded(deadline)) {
                        canceled.store(true, std::memory_order_release);
                        return;
                    }
                    const size_t first = batch * BATCH_SIZE;
                    const size_t count = std::min(BATCH_SIZE, scan.certified.size() - first);
                    positions.clear();
                    batchMask.assign(count, uint8_t{0});
                    for (size_t index = 0; index < count; ++index) {
                        const size_t linear = first + index;
                        const int64_t x = static_cast<int64_t>(linear % scan.sampleWidth);
                        const int64_t z = static_cast<int64_t>(linear / scan.sampleWidth);
                        positions.emplace_back(*interiorOriginX + x * SPACING,
                                               *interiorOriginZ + z * SPACING);
                    }
                    sampler.certifyNativeHydrologyDryMask(positions, batchMask);
                    std::copy(batchMask.begin(), batchMask.end(), scan.certified.begin() + first);
                }
            } catch (...) {
                std::lock_guard lock(failureMutex);
                if (!failure) failure = std::current_exception();
                canceled.store(true, std::memory_order_release);
            }
        };
        std::vector<std::jthread> workers;
        workers.reserve(workerCount);
        for (size_t index = 0; index < workerCount; ++index)
            workers.emplace_back(worker);
        workers.clear();
        throwIfScreenDeadlineExceeded(deadline);
        if (failure) std::rethrow_exception(failure);
        if (canceled.load(std::memory_order_acquire)) return std::nullopt;
        return scan;
    }

    struct AcceptedCandidate {
        Vec3 resolved{};
        std::vector<V4SpawnAlignedCandidate> footprint;
        bool provisionalLearnedDry = false;
    };

    struct CandidateCertificationPolicy {
        bool coldFootprint = false;
        bool exactSafetyFootprint = false;
        bool localFootprint = false;
    };

    [[nodiscard]] static bool
    certificationFootprintFitsTerrainPage(V4SpawnAlignedCandidate candidate,
                                          worldgen::learned::TerrainPageCoordinate terrainPage) {
        const std::optional<V4SpawnCertificationFootprintBounds> bounds =
            v4SpawnCertificationExactFootprintBounds(Chunk::worldToChunk(candidate.worldX),
                                                     Chunk::worldToChunk(candidate.worldZ));
        if (!bounds) return false;
        const auto pageAt = [](int64_t worldX, int64_t worldZ) {
            return worldgen::learned::terrainPageCoordinateFor(
                worldgen::learned::worldBlockToNative(worldX, worldZ));
        };
        return pageAt(bounds->minimumX, bounds->minimumZ) == terrainPage &&
               pageAt(bounds->maximumX, bounds->maximumZ) == terrainPage;
    }

    [[nodiscard]] static bool
    certificationFootprintPassesOwnerMask(V4SpawnAlignedCandidate candidate,
                                          const CertifiedOwnerMask& mask) {
        constexpr int64_t SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
        const std::optional<V4SpawnCertificationFootprintBounds> bounds =
            v4SpawnCertificationExactFootprintBounds(Chunk::worldToChunk(candidate.worldX),
                                                     Chunk::worldToChunk(candidate.worldZ));
        if (!bounds || mask.sampleWidth <= 0 || mask.sampleHeight <= 0) return false;

        if (bounds->minimumX % SPACING != 0 || bounds->minimumZ % SPACING != 0 ||
            bounds->maximumX % SPACING != 0 || bounds->maximumZ % SPACING != 0) {
            return false;
        }
        const int64_t firstX = (bounds->minimumX - mask.originWorldX) / SPACING;
        const int64_t firstZ = (bounds->minimumZ - mask.originWorldZ) / SPACING;
        const int64_t lastX = (bounds->maximumX - mask.originWorldX) / SPACING;
        const int64_t lastZ = (bounds->maximumZ - mask.originWorldZ) / SPACING;
        if (firstX < 0 || firstZ < 0 || lastX < firstX || lastZ < firstZ ||
            lastX >= mask.sampleWidth || lastZ >= mask.sampleHeight) {
            return false;
        }
        for (int64_t z = firstZ; z <= lastZ; ++z) {
            for (int64_t x = firstX; x <= lastX; ++x) {
                const size_t index =
                    static_cast<size_t>(z) * static_cast<size_t>(mask.sampleWidth) +
                    static_cast<size_t>(x);
                if (index >= mask.certified.size() || mask.certified[index] == 0) return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool replaceDryFootprintForCurrentRequest(
        worldgen::MacroGenerationSampler& sampler, std::span<const ColumnPos> columns,
        std::span<worldgen::HydrologySample> installed, std::stop_token stopToken,
        uint64_t requestEpoch, std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> requestLease;
        const auto acquireInstallLease = [&] {
            requestLease = std::unique_lock(mutex_);
            if (stopToken.stop_requested() || !request_ || request_->epoch != requestEpoch)
                throw ScreenRequestObsolete{};
            throwIfScreenDeadlineExceeded(deadline);
        };
        return sampler.replaceNativeHydrologyDryFootprint(columns, installed, acquireInstallLease);
    }

    [[nodiscard]] std::optional<AcceptedCandidate> certifyAndInstallLocalCandidate(
        worldgen::MacroGenerationSampler& sampler, V4SpawnAlignedCandidate candidate,
        std::stop_token stopToken, uint64_t requestEpoch,
        std::chrono::steady_clock::time_point deadline, CandidateCertificationPolicy policy) {
        constexpr int BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
        constexpr int EDGE = BUFFER * 2 + 1;
        constexpr int SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
        const std::optional<int64_t> originX = checkedAdd(candidate.worldX, -BUFFER * SPACING);
        const std::optional<int64_t> originZ = checkedAdd(candidate.worldZ, -BUFFER * SPACING);
        if (!originX || !originZ) return std::nullopt;

        std::vector<V4SpawnAlignedCandidate> localFootprint;
        localFootprint.reserve(static_cast<size_t>(EDGE) * EDGE);
        for (int offsetZ = -BUFFER; offsetZ <= BUFFER; ++offsetZ) {
            for (int offsetX = -BUFFER; offsetX <= BUFFER; ++offsetX) {
                const std::optional<int64_t> worldX =
                    checkedAdd(candidate.worldX, static_cast<int64_t>(offsetX) * SPACING);
                const std::optional<int64_t> worldZ =
                    checkedAdd(candidate.worldZ, static_cast<int64_t>(offsetZ) * SPACING);
                if (!worldX || !worldZ) return std::nullopt;
                localFootprint.push_back({.worldX = *worldX, .worldZ = *worldZ});
            }
        }

        const auto tryFootprint = [&](std::span<const V4SpawnAlignedCandidate> footprint)
            -> std::optional<AcceptedCandidate> {
            std::vector<ColumnPos> columns;
            columns.reserve(footprint.size());
            for (const V4SpawnAlignedCandidate point : footprint)
                columns.emplace_back(point.worldX, point.worldZ);
            throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
            std::vector<worldgen::HydrologySample> installed(columns.size());
            if (!replaceDryFootprintForCurrentRequest(sampler, columns, installed, stopToken,
                                                      requestEpoch, deadline)) {
                throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
                return std::nullopt;
            }
            throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
            if (!sampler.nativeHydrologyDryFootprintContains(columns)) {
                throw std::runtime_error(
                    "The safe-spawn dry footprint was not installed atomically");
            }

            std::array<worldgen::HydrologySample, EDGE * EDGE> localInstalled{};
            if (footprint.size() == localFootprint.size() &&
                std::ranges::equal(footprint, localFootprint)) {
                std::copy(installed.begin(), installed.end(), localInstalled.begin());
            } else {
                std::vector<ColumnPos> localColumns;
                localColumns.reserve(localFootprint.size());
                for (const V4SpawnAlignedCandidate point : localFootprint)
                    localColumns.emplace_back(point.worldX, point.worldZ);
                sampler.sampleHydrologyPoints(localColumns, localInstalled);
            }
            throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
            const std::optional<Vec3> resolved = v4SelectLocalDrySpawnCandidate(
                Vec3{static_cast<float>(candidate.worldX) + 0.5F, 0.0F,
                     static_cast<float>(candidate.worldZ) + 0.5F},
                *originX, *originZ, EDGE, EDGE, localInstalled);
            if (!resolved || static_cast<int64_t>(std::floor(resolved->x)) != candidate.worldX ||
                static_cast<int64_t>(std::floor(resolved->z)) != candidate.worldZ) {
                sampler.clearNativeHydrologyDryFootprint();
                return std::nullopt;
            }
            return AcceptedCandidate{
                .resolved = *resolved,
                .footprint =
                    std::vector<V4SpawnAlignedCandidate>(footprint.begin(), footprint.end()),
            };
        };

        // The normal cold World asks for a fixed set of plan and mesh
        // dependencies before it can certify collision and headroom. When
        // that complete footprint is page-locally dry, install the stronger
        // proof now. Every startup hydrology read then reuses the already
        // prepared FINAL owner instead of opening its four neighbors merely
        // to reconstruct dry columns. A shoreline or river inside the wider
        // footprint simply rejects this optimization and retains the strict
        // five-by-five safety proof below.
        if (policy.coldFootprint) {
            const std::optional<std::vector<V4SpawnAlignedCandidate>> exactFootprint =
                v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(candidate.worldX),
                                                Chunk::worldToChunk(candidate.worldZ));
            if (exactFootprint) {
                if (const std::optional<AcceptedCandidate> accepted =
                        tryFootprint(*exactFootprint)) {
                    return accepted;
                }
            }
        }
        if (policy.exactSafetyFootprint) {
            const std::optional<std::vector<V4SpawnAlignedCandidate>> certificationFootprint =
                v4SpawnCertificationExactFootprintPoints(Chunk::worldToChunk(candidate.worldX),
                                                         Chunk::worldToChunk(candidate.worldZ));
            if (certificationFootprint) {
                if (const std::optional<AcceptedCandidate> accepted =
                        tryFootprint(*certificationFootprint)) {
                    return accepted;
                }
            }
        }
        return policy.localFootprint ? tryFootprint(localFootprint) : std::nullopt;
    }

    // An all-positive continental owner has no immutable ocean terminal, so
    // the conservative page-local proof cannot certify it. Do not invoke the
    // connected lake resolver here: that can open dozens of neighboring
    // owners before World exists. Instead, admit only a flat, positive FINAL
    // learned five-by-five proposal. It carries no dry certificate and cannot
    // finalize metadata. World must still build the center plan and pass its
    // radius-zero canonical hydrology, support, collision, and headroom check.
    [[nodiscard]] std::optional<AcceptedCandidate> tryProvisionalLearnedCandidate(
        const std::shared_ptr<worldgen::learned::WorldGenerationContext>& context, Vec3 requested,
        int64_t worldX, int64_t worldZ, std::stop_token stopToken, uint64_t requestEpoch,
        std::chrono::steady_clock::time_point deadline) {
        using namespace worldgen::learned;
        constexpr int BUFFER = V4_SPAWN_LOCAL_DRY_BUFFER_RADIUS_SAMPLES;
        constexpr int EDGE = BUFFER * 2 + 1;
        constexpr int SPACING = V4_SPAWN_LOCAL_DRY_GRID_SPACING_BLOCKS;
        const std::optional<V4SpawnAlignedCandidate> candidate =
            nearestAlignedSpawnCandidate(requested, worldX, worldZ);
        if (!candidate) return std::nullopt;
        const std::optional<int64_t> originX = checkedAdd(candidate->worldX, -BUFFER * SPACING);
        const std::optional<int64_t> originZ = checkedAdd(candidate->worldZ, -BUFFER * SPACING);
        if (!originX || !originZ) return std::nullopt;

        std::array<WorldBlockPoint, EDGE * EDGE> positions{};
        for (int z = 0; z < EDGE; ++z) {
            for (int x = 0; x < EDGE; ++x) {
                positions[static_cast<size_t>(z) * EDGE + static_cast<size_t>(x)] = {
                    .x = *originX + static_cast<int64_t>(x) * SPACING,
                    .z = *originZ + static_cast<int64_t>(z) * SPACING,
                };
            }
        }
        throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
        const AuthorityResult<std::vector<PhysicalTerrainSample>> learned =
            context->queryWorldPoints(positions, AuthorityRequestPriority::SPAWN);
        if (!learned.isReady()) {
            throw GenerationFailureException(
                learned.status(),
                learned.failure()
                    ? *learned.failure()
                    : GenerationFailure{
                          .code = GenerationFailureCode::INFERENCE_FAILED,
                          .message = "The provisional continental spawn terrain is not ready",
                          .retriable = true,
                      });
        }
        throwIfRequestCannotContinue(stopToken, requestEpoch, deadline);
        if (!learned.value() || learned.value()->size() != positions.size()) {
            throw GenerationFailureException(
                AuthorityStatus::FAILED,
                {.code = GenerationFailureCode::INFERENCE_FAILED,
                 .message = "The provisional continental spawn terrain is incomplete",
                 .retriable = true});
        }

        std::array<worldgen::HydrologySample, EDGE * EDGE> provisional{};
        for (size_t index = 0; index < provisional.size(); ++index) {
            const double elevationMeters = (*learned.value())[index].elevationMeters;
            if (!std::isfinite(elevationMeters) || elevationMeters <= 0.0) return std::nullopt;
            provisional[index].surfaceElevation =
                learnedElevationMetersToWorldHeight(elevationMeters);
            provisional[index].waterSurface = LEARNED_SEA_LEVEL;
        }
        const std::optional<Vec3> resolved =
            v4SelectLocalDrySpawnCandidate(Vec3{static_cast<float>(candidate->worldX) + 0.5F, 0.0F,
                                                static_cast<float>(candidate->worldZ) + 0.5F},
                                           *originX, *originZ, EDGE, EDGE, provisional);
        if (!resolved || static_cast<int64_t>(std::floor(resolved->x)) != candidate->worldX ||
            static_cast<int64_t>(std::floor(resolved->z)) != candidate->worldZ) {
            return std::nullopt;
        }
        return AcceptedCandidate{
            .resolved = *resolved,
            .provisionalLearnedDry = true,
        };
    }

    // Try the learned proposal before scanning the complete owner. A dry and
    // locally safe center is sufficient even when its wider generation
    // footprint contains canonical water. Only its five-by-five safety buffer
    // enters the immutable fast path. Wider water remains normal streaming
    // work and cannot delay dry-land selection. Only a wet or locally
    // unsuitable center falls through to the exhaustive owner search.
    void run(std::stop_token stopToken) {
        uint64_t observedEpoch = 0;
        while (!stopToken.stop_requested()) {
            Request request;
            {
                std::unique_lock lock(mutex_);
                wake_.wait_for(lock, std::chrono::milliseconds{10}, [&] {
                    return stopToken.stop_requested() ||
                           (request_ && request_->epoch != observedEpoch);
                });
                if (stopToken.stop_requested()) return;
                if (!request_ || request_->epoch == observedEpoch) continue;
                request = *request_;
                observedEpoch = request.epoch;
            }

            const std::chrono::steady_clock::time_point requestStartedAt = request.startedAt;
            const std::chrono::steady_clock::time_point requestDeadline =
                requestStartedAt + timing_.activeWorkTimeout;

            // A screen has to use FINAL authority and the SPAWN priority
            // lane, but shares the context's identity, failure latch, native
            // hydrology cache, and persisted RYHY pages with later exact
            // generation.
            const std::shared_ptr<worldgen::learned::WorldGenerationContext> finalContext =
                request.context->quality() == worldgen::learned::AuthorityQuality::FINAL
                    ? request.context
                    : request.context->withQuality(worldgen::learned::AuthorityQuality::FINAL);
            const std::shared_ptr<worldgen::learned::WorldGenerationContext> spawnContext =
                finalContext->withRequestPriority(
                    worldgen::learned::AuthorityRequestPriority::SPAWN);
            worldgen::MacroGenerationSampler sampler(spawnContext->identity().seed, spawnContext);
            ProgressSnapshot previousProgress = progressSnapshot(spawnContext);
            std::chrono::steady_clock::time_point lastStateOrCompletionAt = requestStartedAt;
            const int64_t ownerX = world_coord::floorDiv(
                request.worldX, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            const int64_t ownerZ = world_coord::floorDiv(
                request.worldZ, static_cast<int64_t>(worldgen::NATIVE_HYDROLOGY_PAGE_EDGE));
            const std::optional<V4SpawnAlignedCandidate> proposedCandidate =
                nearestAlignedSpawnCandidate(request.candidate, request.worldX, request.worldZ);
            const std::optional<worldgen::learned::TerrainPageCoordinate> proposedTerrainPage =
                proposedCandidate ? std::optional{worldgen::learned::terrainPageCoordinateFor(
                                        worldgen::learned::worldBlockToNative(
                                            proposedCandidate->worldX, proposedCandidate->worldZ))}
                                  : std::nullopt;
            bool ownerPrepared = false;
            std::optional<CertifiedOwnerMask> ownerMask;
            std::vector<V4SpawnAlignedCandidate> rankedCandidates;
            bool candidatesRanked = false;
            size_t exactCandidateIndex = 0;
            size_t exactCandidateAttempts = 0;
            size_t localCandidateIndex = 0;
            bool proposedExactTried = false;
            bool exactRelocationComplete = false;
            bool proposedLocalTried = false;
            bool provisionalLearnedTried = false;
            bool provisionalLearnedDry = false;
            std::optional<Vec3> resolvedCandidate;
            std::vector<V4SpawnAlignedCandidate> installedFootprint;

            while (!stopToken.stop_requested() && requestIsCurrent(request.epoch)) {
                try {
                    throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                    if (!ownerPrepared) {
                        // Direct preparation builds or loads exactly the
                        // requested 2,048-block owner. It neither opens a
                        // neighbor nor marks global semantic preparation.
                        sampler.prepareNativeHydrologyOwner(ownerX, ownerZ);
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        ownerPrepared = true;
                    }
                    if (!proposedExactTried && proposedCandidate) {
                        const std::optional<AcceptedCandidate> accepted =
                            certifyAndInstallLocalCandidate(
                                sampler, *proposedCandidate, stopToken, request.epoch,
                                requestDeadline,
                                {.coldFootprint = true, .exactSafetyFootprint = true});
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        proposedExactTried = true;
                        if (accepted) {
                            resolvedCandidate = accepted->resolved;
                            installedFootprint = accepted->footprint;
                            provisionalLearnedDry = accepted->provisionalLearnedDry;
                        }
                    } else if (!proposedExactTried) {
                        proposedExactTried = true;
                    }
                    if (!resolvedCandidate && !ownerMask) {
                        ownerMask = scanStrictOwner(sampler, ownerX, ownerZ, stopToken,
                                                    request.epoch, requestDeadline);
                        if (!ownerMask) {
                            if (stopToken.stop_requested() || !requestIsCurrent(request.epoch))
                                break;
                            throw std::runtime_error(
                                "The safe-spawn hydrology owner exceeded the coordinate range");
                        }
                    }
                    if (!resolvedCandidate && !candidatesRanked) {
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        rankedCandidates = v4RankCertifiedDrySpawnCandidates(
                            request.candidate, ownerMask->originWorldX, ownerMask->originWorldZ,
                            ownerMask->sampleWidth, ownerMask->sampleHeight, ownerMask->certified);
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        candidatesRanked = true;
                    }

                    // A dry exact-safety footprint prevents the radius-zero
                    // World check from resolving an open depression through
                    // cardinal owners. Prefer a bounded relocation inside the
                    // already materialized FINAL terrain page. The owner mask
                    // is a necessary four-block prefilter, while the installed
                    // one-block certificate remains the authoritative proof.
                    constexpr size_t MAXIMUM_EXACT_RELOCATION_ATTEMPTS = 64;
                    while (!resolvedCandidate && !exactRelocationComplete &&
                           exactCandidateIndex < rankedCandidates.size() &&
                           exactCandidateAttempts < MAXIMUM_EXACT_RELOCATION_ATTEMPTS) {
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        const V4SpawnAlignedCandidate candidate =
                            rankedCandidates[exactCandidateIndex];
                        if ((proposedCandidate && candidate == *proposedCandidate) ||
                            !proposedTerrainPage ||
                            !certificationFootprintFitsTerrainPage(candidate,
                                                                   *proposedTerrainPage) ||
                            !certificationFootprintPassesOwnerMask(candidate, *ownerMask)) {
                            ++exactCandidateIndex;
                            continue;
                        }
                        const std::optional<AcceptedCandidate> accepted =
                            certifyAndInstallLocalCandidate(sampler, candidate, stopToken,
                                                            request.epoch, requestDeadline,
                                                            {.exactSafetyFootprint = true});
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        ++exactCandidateIndex;
                        ++exactCandidateAttempts;
                        if (!accepted) continue;
                        resolvedCandidate = accepted->resolved;
                        installedFootprint = accepted->footprint;
                    }
                    if (!resolvedCandidate && !exactRelocationComplete &&
                        (exactCandidateIndex >= rankedCandidates.size() ||
                         exactCandidateAttempts >= MAXIMUM_EXACT_RELOCATION_ATTEMPTS)) {
                        exactRelocationComplete = true;
                    }

                    // Retain the former local-certificate behavior exactly as
                    // the semantic fallback. A river or shoreline in the
                    // wider safety footprint must not turn valid dry land into
                    // a false no-spawn result.
                    if (!resolvedCandidate && exactRelocationComplete && !proposedLocalTried) {
                        if (proposedCandidate) {
                            const std::optional<AcceptedCandidate> accepted =
                                certifyAndInstallLocalCandidate(
                                    sampler, *proposedCandidate, stopToken, request.epoch,
                                    requestDeadline, {.localFootprint = true});
                            throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                            proposedLocalTried = true;
                            if (accepted) {
                                resolvedCandidate = accepted->resolved;
                                installedFootprint = accepted->footprint;
                            }
                        } else {
                            proposedLocalTried = true;
                        }
                    }
                    while (!resolvedCandidate && exactRelocationComplete &&
                           localCandidateIndex < rankedCandidates.size()) {
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        const V4SpawnAlignedCandidate candidate =
                            rankedCandidates[localCandidateIndex];
                        if (proposedCandidate && candidate == *proposedCandidate) {
                            ++localCandidateIndex;
                            continue;
                        }
                        const std::optional<AcceptedCandidate> accepted =
                            certifyAndInstallLocalCandidate(sampler, candidate, stopToken,
                                                            request.epoch, requestDeadline,
                                                            {.localFootprint = true});
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        if (!accepted) {
                            ++localCandidateIndex;
                            continue;
                        }
                        resolvedCandidate = accepted->resolved;
                        installedFootprint = accepted->footprint;
                    }
                    if (!resolvedCandidate && !provisionalLearnedTried) {
                        const std::optional<AcceptedCandidate> provisional =
                            tryProvisionalLearnedCandidate(
                                spawnContext, request.candidate, request.worldX, request.worldZ,
                                stopToken, request.epoch, requestDeadline);
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        provisionalLearnedTried = true;
                        if (provisional) {
                            resolvedCandidate = provisional->resolved;
                            provisionalLearnedDry = true;
                        }
                    }
                    throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                    if (!resolvedCandidate) {
                        publish(request.epoch, {.status = V4SpawnWaterScreenStatus::Water});
                        break;
                    }

                    throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                    if (!provisionalLearnedDry) {
                        const worldgen::learned::AuthorityResult<bool> prepared =
                            prepareAcceptedV4SpawnAuthority(spawnContext, sampler,
                                                            *resolvedCandidate, installedFootprint);
                        throwIfRequestCannotContinue(stopToken, request.epoch, requestDeadline);
                        if (!prepared.isReady()) {
                            throw worldgen::learned::GenerationFailureException(
                                prepared.status(),
                                prepared.failure()
                                    ? *prepared.failure()
                                    : worldgen::learned::GenerationFailure{
                                          .code = worldgen::learned::GenerationFailureCode::
                                              INFERENCE_FAILED,
                                          .message =
                                              "The accepted safe-spawn authority did not return "
                                              "a value or failure",
                                          .retriable = true,
                                      });
                        }
                    }
                    publish(request.epoch, {.status = V4SpawnWaterScreenStatus::Dry,
                                            .resolvedCandidate = *resolvedCandidate,
                                            .provisionalLearnedDry = provisionalLearnedDry});
                    break;
                } catch (const ScreenRequestObsolete&) {
                    break;
                } catch (const ScreenDeadlineExceeded&) {
                    publish(request.epoch, screenDeadlineFailure());
                    break;
                } catch (const worldgen::learned::GenerationFailureException& failure) {
                    if (failure.status() == worldgen::learned::AuthorityStatus::DEFERRED) {
                        // Authority admission and model execution remain
                        // asynchronous. State transitions and completed work
                        // are observable progress. Repeated deferred replies
                        // or failed native-page attempts are not.
                        const auto now = std::chrono::steady_clock::now();
                        if (now >= requestDeadline) {
                            publish(request.epoch, screenDeadlineFailure());
                            break;
                        }
                        const ProgressSnapshot currentProgress = progressSnapshot(spawnContext);
                        if (screenDeadlineExceeded(requestDeadline)) {
                            publish(request.epoch, screenDeadlineFailure());
                            break;
                        }
                        const bool completedWork =
                            currentProgress.completedWorkSince(previousProgress);
                        const bool workStateChanged =
                            currentProgress.workStateChangedSince(previousProgress);
                        if (completedWork || workStateChanged) lastStateOrCompletionAt = now;
                        previousProgress = currentProgress;

                        const bool activeWork = currentProgress.hasAdmittedOrRunningWork();
                        if (!activeWork &&
                            now - lastStateOrCompletionAt >= timing_.noProgressTimeout) {
                            publish(
                                request.epoch,
                                {.status = V4SpawnWaterScreenStatus::Failed,
                                 .failure = worldgen::learned::GenerationFailure{
                                     .code =
                                         worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                                     .message =
                                         "Canonical safe-spawn water screening made no authority "
                                         "or hydrology progress within " +
                                         std::to_string(timing_.noProgressTimeout.count()) +
                                         " milliseconds",
                                     .retriable = true,
                                 }});
                            break;
                        }
                        std::unique_lock lock(mutex_);
                        const std::chrono::steady_clock::time_point retryAt =
                            std::min(requestDeadline,
                                     std::chrono::steady_clock::now() + timing_.retryInterval);
                        wake_.wait_until(lock, retryAt, [&] {
                            return stopToken.stop_requested() || !request_ ||
                                   request_->epoch != request.epoch;
                        });
                        continue;
                    }
                    publish(request.epoch, {.status = V4SpawnWaterScreenStatus::Failed,
                                            .failure = failure.failure()});
                    break;
                } catch (const std::exception& exception) {
                    publish(request.epoch,
                            {.status = V4SpawnWaterScreenStatus::Failed,
                             .failure = worldgen::learned::GenerationFailure{
                                 .code = worldgen::learned::GenerationFailureCode::INFERENCE_FAILED,
                                 .message = "Canonical safe-spawn water screening failed: " +
                                            std::string(exception.what()),
                                 .retriable = true,
                             }});
                    break;
                }
            }
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::optional<Request> request_;
    V4SpawnWaterScreenResult result_;
    uint64_t requestEpoch_ = 0;
    const V4SpawnWaterScreenTiming timing_;
    std::jthread worker_;
};

V4SpawnWaterScreen::V4SpawnWaterScreen(V4SpawnWaterScreenTiming timing)
    : impl_(std::make_unique<Impl>(timing)) {}

V4SpawnWaterScreen::~V4SpawnWaterScreen() = default;

V4SpawnWaterScreenResult V4SpawnWaterScreen::screen(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
    Vec3 candidate) {
    return impl_->screen(finalContext, candidate);
}

void V4SpawnWaterScreen::reset() {
    impl_->reset();
}

worldgen::learned::AuthorityResult<std::optional<Vec3>> findV4DryLandSpawnCandidate(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& finalContext,
    int64_t originWorldX, int64_t originWorldZ, uint32_t ordinal) {
    using namespace worldgen::learned;
    if (!finalContext) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            {.code = GenerationFailureCode::BACKEND_UNAVAILABLE,
             .message = "The learned terrain authority is unavailable for dry-land spawn selection",
             .retriable = true});
    }
    if (ordinal >= V4_DRY_SPAWN_SEARCH_MAX_CANDIDATES) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message =
                 "No dry learned-terrain candidate was found within the bounded spawn search",
             .retriable = false});
    }
    const TerrainPageCoordinate originPage =
        terrainPageCoordinateFor(worldBlockToNative(originWorldX, originWorldZ));
    const std::shared_ptr<WorldGenerationContext> finalAuthorityContext =
        finalContext->quality() == AuthorityQuality::FINAL
            ? finalContext
            : finalContext->withQuality(AuthorityQuality::FINAL);
    const std::optional<CoarseSpawnRegion> coarseRegion = coarseSpawnSearchRegion(
        originPage.row, originPage.column, V4_DRY_SPAWN_SEARCH_MAX_COARSE_EDGE);
    if (!coarseRegion) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message = "The dry-land spawn search exceeded the learned terrain coordinate range",
             .retriable = false});
    }
    const AuthorityResult<CoarseSpawnGrid> coarse =
        finalContext->queryCoarseSpawnGrid(*coarseRegion, AuthorityRequestPriority::SPAWN);
    if (coarse.status() == AuthorityStatus::FAILED) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            coarse.failure() ? *coarse.failure()
                             : GenerationFailure{
                                   .code = GenerationFailureCode::INFERENCE_FAILED,
                                   .message = "The coarse dry-land spawn map could not be prepared",
                                   .retriable = true});
    }
    if (!coarse.isReady()) {
        return AuthorityResult<std::optional<Vec3>>::deferred(
            coarse.failure()
                ? *coarse.failure()
                : GenerationFailure{.code = GenerationFailureCode::PAGE_NOT_FOUND,
                                    .message = "The coarse dry-land spawn map is still preparing",
                                    .retriable = true});
    }
    const std::optional<TerrainPageCoordinate> coordinate =
        coarseLandSpawnPage(*coarse.value(), originPage.row, originPage.column, ordinal);
    if (!coordinate) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message =
                 "No dry learned-terrain candidate was found within the bounded coarse search",
             .retriable = false});
    }

    // Coarse land only proposes where to spend final authority work. It is
    // never collision, water, or spawn authority: final terrain must
    // independently expose an inland candidate before exact cubes build.
    const std::optional<NativeRect> region = terrainPageNativeRect(*coordinate);
    if (!region) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            {.code = GenerationFailureCode::INVALID_REQUEST,
             .message = "The dry-land spawn page has an invalid learned terrain rectangle",
             .retriable = false});
    }
    // The selector needs only its one immutable final page. The canonical
    // water screen uses a separate exact transient raster, sharing tensor
    // windows without publishing mostly unused neighboring pages.
    const AuthorityResult<bool> finalRequested =
        requestSpawnCandidateFinalPage(finalAuthorityContext, *region);
    if (finalRequested.status() == AuthorityStatus::FAILED) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            finalRequested.failure()
                ? *finalRequested.failure()
                : GenerationFailure{.code = GenerationFailureCode::INFERENCE_FAILED,
                                    .message =
                                        "The final dry-land spawn page could not be prepared",
                                    .retriable = true});
    }
    if (!finalRequested.isReady()) {
        return AuthorityResult<std::optional<Vec3>>::deferred(
            finalRequested.failure()
                ? *finalRequested.failure()
                : GenerationFailure{.code = GenerationFailureCode::PAGE_NOT_FOUND,
                                    .message = "The final dry-land spawn page is still preparing",
                                    .retriable = true});
    }
    const AuthorityResult<PhysicalTerrainGrid> finalQueried =
        finalAuthorityContext->queryNative(*region, AuthorityRequestPriority::SPAWN);
    if (finalQueried.status() == AuthorityStatus::FAILED) {
        return AuthorityResult<std::optional<Vec3>>::failed(
            finalQueried.failure()
                ? *finalQueried.failure()
                : GenerationFailure{.code = GenerationFailureCode::INFERENCE_FAILED,
                                    .message = "The final dry-land spawn page could not be sampled",
                                    .retriable = true});
    }
    if (!finalQueried.isReady()) {
        return AuthorityResult<std::optional<Vec3>>::deferred(
            finalQueried.failure()
                ? *finalQueried.failure()
                : GenerationFailure{.code = GenerationFailureCode::PAGE_NOT_FOUND,
                                    .message = "The final dry-land spawn page is still preparing",
                                    .retriable = true});
    }
    return AuthorityResult<std::optional<Vec3>>::ready(
        inlandSpawnCandidate(*finalQueried.value(), originWorldX, originWorldZ));
}

V4SpawnAuthorityPrequeueResult prequeueV4SpawnAuthority(
    const std::shared_ptr<worldgen::learned::WorldGenerationContext>& generationContext,
    int64_t worldX, int64_t worldZ, int radiusChunks) {
    using namespace worldgen::learned;
    V4SpawnAuthorityPrequeueResult result;
    if (!generationContext) {
        result.status = V4SpawnAuthorityPrequeueStatus::Failed;
        result.failure = GenerationFailure{
            .code = GenerationFailureCode::BACKEND_UNAVAILABLE,
            .message = "The safe spawn terrain authority is unavailable",
            .retriable = true,
        };
        return result;
    }
    const std::optional<V4SpawnAuthorityPlan> plan =
        v4SpawnAuthorityPlan(worldX, worldZ, radiusChunks);
    if (!plan) {
        result.status = V4SpawnAuthorityPrequeueStatus::Failed;
        result.failure = GenerationFailure{
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "The safe spawn authority rectangle is out of range",
            .retriable = false,
        };
        return result;
    }
    const worldgen::NativeHydrologyAuthorityRequirements& requirements = plan->requirements;
    result.finalTopologyPageCount = requirements.finalTopologyPages.size();
    result.finalRefinementPageCount = requirements.finalRefinementPages.size();
    result.hydrologyOwnerCount = plan->hydrologyOwners.size();
    if (requirements.totalPageCount() > MAXIMUM_AUTHORITY_QUEUED_REQUESTS) {
        result.status = V4SpawnAuthorityPrequeueStatus::Failed;
        result.failure = GenerationFailure{
            .code = GenerationFailureCode::INVALID_REQUEST,
            .message = "The safe spawn authority exceeds the bounded inference queue",
            .retriable = false,
        };
        return result;
    }

    const std::shared_ptr<WorldGenerationContext> finalContext =
        generationContext->quality() == AuthorityQuality::FINAL
            ? generationContext
            : generationContext->withQuality(AuthorityQuality::FINAL);
    for (const V4SpawnHydrologyOwner owner : plan->hydrologyOwners) {
        if (finalContext->nativeHydrologyOwnerPrepared(owner.x, owner.z))
            ++result.preparedHydrologyOwnerCount;
    }
    const bool allOwnersPrepared =
        result.preparedHydrologyOwnerCount == result.hydrologyOwnerCount &&
        result.hydrologyOwnerCount != 0;
    if (boundedColdStartExactRadiusChunks(radiusChunks) == COLD_START_EXACT_CUBIC_DISTANCE_CHUNKS) {
        const std::optional<std::vector<V4SpawnAlignedCandidate>> footprint =
            v4ColdSpawnExactFootprintPoints(Chunk::worldToChunk(worldX),
                                            Chunk::worldToChunk(worldZ));
        if (footprint) {
            std::vector<ColumnPos> columns;
            columns.reserve(footprint->size());
            for (const V4SpawnAlignedCandidate point : *footprint)
                columns.emplace_back(point.worldX, point.worldZ);
            worldgen::MacroGenerationSampler sampler(finalContext->identity().seed, finalContext);
            result.reusedCertifiedDryFootprint =
                sampler.nativeHydrologyDryFootprintContains(columns);
        }
    }
    result.reusedPreparedHydrology = allOwnersPrepared || result.reusedCertifiedDryFootprint;

    std::set<TerrainPageCoordinate> finalPages;
    if (!result.reusedPreparedHydrology) {
        finalPages.insert(requirements.finalTopologyPages.begin(),
                          requirements.finalTopologyPages.end());
    }
    finalPages.insert(requirements.finalRefinementPages.begin(),
                      requirements.finalRefinementPages.end());
    const std::vector<TerrainPageCoordinate> closure(finalPages.begin(), finalPages.end());
    result.finalPageCount = closure.size();
    const AuthorityResult<bool> requested =
        finalContext->requestAuthorityPages(closure, AuthorityRequestPriority::SPAWN);
    if (requested.status() == AuthorityStatus::FAILED) {
        result.status = V4SpawnAuthorityPrequeueStatus::Failed;
        result.failure =
            requested.failure()
                ? *requested.failure()
                : GenerationFailure{
                      .code = GenerationFailureCode::INFERENCE_FAILED,
                      .message = "The safe spawn terrain authority could not be prepared",
                      .retriable = true,
                  };
        return result;
    }
    if (requested.status() == AuthorityStatus::DEFERRED) {
        result.status = V4SpawnAuthorityPrequeueStatus::Deferred;
        if (requested.failure()) result.failure = *requested.failure();
        return result;
    }
    result.status = V4SpawnAuthorityPrequeueStatus::Ready;
    return result;
}

std::optional<std::filesystem::path>
resolveV4LaunchProfilePath(const std::filesystem::path& applicationSupport,
                           std::string_view requestedPath) {
    if (requestedPath.empty()) return std::nullopt;
    std::filesystem::path resolved{requestedPath};
    if (resolved.is_relative()) resolved = applicationSupport / resolved;
    resolved = resolved.lexically_normal();
    if (resolved.filename().empty()) resolved = resolved.parent_path();
    return resolved;
}

V4WorldOpenResult openQualifiedV4World(worldgen::bootstrap::TerrainGenerationBootstrap& bootstrap,
                                       uint64_t seed, Vec3 initialSpawn, uint64_t initialWorldTime,
                                       std::shared_ptr<SaveManager::TestHooks> persistenceTestHooks,
                                       std::optional<std::filesystem::path> preferredProfilePath,
                                       std::optional<V4WorldCreationRequest> creationRequest) {
    const std::optional<std::filesystem::path> defaultWorldPath = bootstrap.worldPath();
    const std::optional<std::string> fingerprint = bootstrap.qualifiedGenerationFingerprint();
    const std::shared_ptr<worldgen::learned::WorldGenerationContext> generationContext =
        bootstrap.qualifiedGenerationContext();
    if (!defaultWorldPath || !fingerprint || !generationContext) {
        return {.status = V4WorldOpenStatus::BootstrapNotReady,
                .message = "Generator v4 model and runtime qualification is not ready"};
    }
    if (generationContext->identity().seed != seed ||
        worldgen::learned::sha256Hex(generationContext->fingerprint()) != *fingerprint) {
        return {.status = V4WorldOpenStatus::IdentityConflict,
                .message = "The requested v4 seed does not match the qualified generation "
                           "identity"};
    }

    if (preferredProfilePath && creationRequest) {
        return {.status = V4WorldOpenStatus::InvalidWorldDirectory,
                .profilePath = *preferredProfilePath,
                .message = "A new generator v4 world cannot reuse an existing profile path"};
    }

    std::filesystem::path worldPath = preferredProfilePath.value_or(*defaultWorldPath);
    if (preferredProfilePath) {
        const std::string preferredName = worldPath.filename().string();
        const std::string defaultName = defaultWorldPath->filename().string();
        const bool allowedName =
            preferredName == defaultName || preferredName.starts_with(defaultName + "-seed-");
        if (worldPath.parent_path() != defaultWorldPath->parent_path() || !allowedName) {
            return {.status = V4WorldOpenStatus::InvalidWorldDirectory,
                    .profilePath = worldPath,
                    .message = "The selected generator v4 profile is outside the profile root"};
        }
    }
    V4ProfileInspection inspection = inspectV4Profile(worldPath, seed, *fingerprint);
    if (preferredProfilePath && inspection.kind == V4ProfileInspectionKind::Available) {
        return {.status = V4WorldOpenStatus::MissingMetadata,
                .profilePath = worldPath,
                .usingSeparateProfile = worldPath != *defaultWorldPath,
                .message = inspection.existed
                               ? "The selected generator v4 profile has no valid metadata"
                               : "The selected generator v4 profile does not exist"};
    }
    if (creationRequest && inspection.existed) {
        bool selected = false;
        for (uint32_t collisionOrdinal = 0;
             collisionOrdinal < MAXIMUM_SEPARATE_V4_PROFILE_COLLISIONS; ++collisionOrdinal) {
            const std::filesystem::path candidate =
                separateV4ProfilePath(*defaultWorldPath, seed, *fingerprint, collisionOrdinal);
            V4ProfileInspection candidateInspection =
                inspectV4Profile(candidate, seed, *fingerprint);
            // Explicit creation never adopts an existing directory, including
            // an empty or interrupted one. A retry therefore cannot overwrite
            // another process's profile or an unrecognized filesystem entry.
            if (candidateInspection.kind != V4ProfileInspectionKind::Available ||
                candidateInspection.existed) {
                continue;
            }
            worldPath = candidate;
            inspection = std::move(candidateInspection);
            selected = true;
            break;
        }
        if (!selected) {
            return {.status = V4WorldOpenStatus::PersistenceFailure,
                    .message = "No collision-safe path is available for the new generator v4 "
                               "world"};
        }
    }
    bool usingSeparateProfile = worldPath != *defaultWorldPath;
    std::string profileMessage;
    if (inspection.kind == V4ProfileInspectionKind::IdentityConflict) {
        return {.status = V4WorldOpenStatus::IdentityConflict,
                .profilePath = worldPath,
                .usingSeparateProfile = usingSeparateProfile,
                .message = preferredProfilePath
                               ? "The selected generator v4 profile belongs to a different "
                                 "seed or generation fingerprint"
                               : "The default generator v4 profile belongs to a different seed "
                                 "or generation fingerprint; select a world or create a new one"};
    } else if (inspection.kind == V4ProfileInspectionKind::InvalidDirectory) {
        return {.status = V4WorldOpenStatus::InvalidWorldDirectory,
                .message = std::move(inspection.message)};
    } else if (inspection.kind == V4ProfileInspectionKind::MissingMetadata) {
        return {.status = V4WorldOpenStatus::MissingMetadata,
                .message = std::move(inspection.message)};
    }

    const bool existed = inspection.existed;
    const std::optional<SaveManager::WorldMetadata> existing = inspection.metadata;
    std::error_code error;
    const auto bindSelectedProfile = [&]() -> std::optional<V4WorldOpenResult> {
        if (bootstrap.bindWorldProfile(worldPath)) return std::nullopt;
        const worldgen::bootstrap::TerrainBootstrapSnapshot snapshot = bootstrap.snapshot();
        std::string message = snapshot.detail.empty()
                                  ? "The qualified terrain authority could not bind the selected "
                                    "generator v4 profile"
                                  : snapshot.detail;
        if (!profileMessage.empty()) message = profileMessage + ". " + message;
        return V4WorldOpenResult{.status = V4WorldOpenStatus::BootstrapNotReady,
                                 .profilePath = worldPath,
                                 .usingSeparateProfile = usingSeparateProfile,
                                 .message = std::move(message)};
    };

    if (existing) {
        if (creationRequest) {
            return {.status = V4WorldOpenStatus::PersistenceFailure,
                    .profilePath = worldPath,
                    .usingSeparateProfile = usingSeparateProfile,
                    .message = "The new generator v4 world path was claimed before publication"};
        }
        if (std::optional<V4WorldOpenResult> bindingFailure = bindSelectedProfile())
            return std::move(*bindingFailure);
        auto saves = std::make_unique<SaveManager>(
            worldPath.string(), SaveManager::Profile::GeneratorV4, std::move(persistenceTestHooks));
        V4WorldOpenResult result;
        result.status = V4WorldOpenStatus::Ready;
        result.saveManager = std::move(saves);
        result.metadata = *existing;
        result.profilePath = worldPath;
        result.usingSeparateProfile = usingSeparateProfile;
        result.fresh = v4SpawnRequiresStrictDryValidation(existing->spawnFinalized,
                                                          existing->spawnSafetyRevision,
                                                          existing->safeSpawnPos.has_value());
        result.message = std::move(profileMessage);
        return result;
    }

    // Publish a fresh profile as one directory rename. If a prior process
    // stopped after constructing only the expected empty directories, discard
    // that residue and repeat the same transaction. Arbitrary files are never
    // removed or overwritten.
    if (existed) {
        if (!isRecoverableV4Residue(worldPath)) {
            return {.status = V4WorldOpenStatus::MissingMetadata,
                    .message = "The existing generator v4 world metadata is missing or corrupt"};
        }
        std::filesystem::remove_all(worldPath, error);
        if (error) {
            return {.status = V4WorldOpenStatus::PersistenceFailure,
                    .message = "The incomplete generator v4 profile could not be repaired"};
        }
    }

    std::string stagingMessage;
    const std::optional<std::filesystem::path> stagingPath =
        createV4StagingDirectory(worldPath, stagingMessage);
    if (!stagingPath) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = std::move(stagingMessage)};
    }
    StagingWorldCleanup cleanup{*stagingPath};
    auto staging = std::make_unique<SaveManager>(
        stagingPath->string(), SaveManager::Profile::GeneratorV4, persistenceTestHooks);
    SaveManager::WorldMetadata freshMetadata;
    freshMetadata.seed = seed;
    freshMetadata.generationFingerprint = *fingerprint;
    freshMetadata.spawnFinalized = false;
    freshMetadata.spawnSafetyRevision = 0;
    freshMetadata.spawnPos = initialSpawn;
    freshMetadata.playerPos = initialSpawn;
    freshMetadata.safeSpawnPos.reset();
    freshMetadata.worldTime = initialWorldTime;
    freshMetadata.generatorVersion = SaveManager::GENERATOR_V4_VERSION;
    freshMetadata.name = creationRequest && !creationRequest->displayName.empty()
                             ? creationRequest->displayName
                             : worldPath.filename().string();
    if (creationRequest) {
        freshMetadata.gameMode = creationRequest->gameMode;
        freshMetadata.generation = creationRequest->generation;
        freshMetadata.player = creationRequest->player;
    }
    freshMetadata.createdMs =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count());
    if (!staging->saveMetadata(freshMetadata)) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = "The generator v4 identity metadata could not be published"};
    }
    const std::optional<SaveManager::WorldMetadata> staged = staging->loadMetadata();
    staging.reset();
    if (!staged) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = "The generator v4 identity metadata could not be validated"};
    }
    // Do not rely on directory rename replacing an empty destination. A
    // competing process or a manually created sibling must stay intact; the
    // caller can retry and select the resulting compatible profile.
    if (std::filesystem::exists(worldPath, error) || error) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = "The generator v4 profile path changed during publication"};
    }
    std::filesystem::rename(*stagingPath, worldPath, error);
    if (error) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = "The generator v4 profile could not be installed atomically"};
    }
    cleanup.committed = true;

    if (std::optional<V4WorldOpenResult> bindingFailure = bindSelectedProfile())
        return std::move(*bindingFailure);

    auto saves = std::make_unique<SaveManager>(
        worldPath.string(), SaveManager::Profile::GeneratorV4, std::move(persistenceTestHooks));
    const std::optional<SaveManager::WorldMetadata> created = saves->loadMetadata();
    if (!created) {
        return {.status = V4WorldOpenStatus::PersistenceFailure,
                .message = "The generator v4 identity metadata could not be validated"};
    }

    V4WorldOpenResult result;
    result.status = V4WorldOpenStatus::Ready;
    result.saveManager = std::move(saves);
    result.metadata = *created;
    result.profilePath = worldPath;
    result.usingSeparateProfile = usingSeparateProfile;
    result.newlyCreated = true;
    result.fresh = true;
    result.message = std::move(profileMessage);
    return result;
}
