#include "world/features.hpp"

#include "world/chunk_generator.hpp"
#include "world/surface_material.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

namespace {

constexpr int MAX_TREE_SPACING = 14;
constexpr uint64_t TREE_CANDIDATE_STREAM = 0x5452454543414E44ULL;
constexpr uint64_t TREE_PRIORITY_STREAM = 0x545245455052494FULL;
constexpr uint64_t TREE_SHAPE_STREAM = 0x5452454553484150ULL;
constexpr uint64_t TREE_BIOME_STREAM = 0x5452454542494F4DULL;
constexpr uint64_t FAR_CANOPY_CLUSTER_STREAM = 0x46415243414E4F50ULL;
constexpr uint64_t FLORA_STREAM = 0x464C4F524143454CULL;
constexpr uint64_t FLORA_BIOME_STREAM = 0x464C4F524142494FULL;

enum class TreeKind : uint8_t {
    OAK,
    LARGE_OAK,
    BIRCH,
    SPRUCE,
    ACACIA,
    JUNGLE,
    MANGROVE,
    PALM,
    WILLOW,
    ALPINE_SCRUB,
    FALLEN_LOG,
};

struct SpeciesTraits {
    double minimumTemperatureC;
    double maximumTemperatureC;
    double minimumMoisture;
    double minimumFertility;
    double maximumSlope;
    double minimumAltitude;
    double maximumAltitude;
    double minimumLight;
    bool toleratesFlooding;
    int spacing;
};

constexpr SpeciesTraits traitsFor(TreeKind kind) {
    switch (kind) {
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK:
            return {-8.0, 29.0, 0.28, 0.22, 1.20, 48.0, 190.0, 0.45, false, 9};
        case TreeKind::BIRCH:
            return {-14.0, 22.0, 0.30, 0.18, 1.30, 50.0, 210.0, 0.52, false, 8};
        case TreeKind::SPRUCE:
            return {-24.0, 31.0, 0.30, 0.12, 1.45, 45.0, 285.0, 0.35, false, 8};
        case TreeKind::ACACIA:
            return {15.0, 38.0, 0.12, 0.12, 1.10, 48.0, 170.0, 0.70, false, 11};
        case TreeKind::JUNGLE:
            return {18.0, 38.0, 0.65, 0.35, 1.05, 48.0, 175.0, 0.28, true, 12};
        case TreeKind::MANGROVE:
            return {16.0, 36.0, 0.72, 0.24, 0.75, 48.0, 82.0, 0.42, true, 9};
        case TreeKind::PALM:
            return {17.0, 39.0, 0.24, 0.10, 0.85, 48.0, 105.0, 0.64, true, 12};
        case TreeKind::WILLOW:
            return {-2.0, 29.0, 0.58, 0.24, 0.85, 48.0, 115.0, 0.34, true, 10};
        case TreeKind::ALPINE_SCRUB:
            return {-24.0, 13.0, 0.10, 0.05, 1.55, 82.0, 350.0, 0.58, false, 7};
        case TreeKind::FALLEN_LOG:
            return {-12.0, 34.0, 0.25, 0.10, 0.70, 48.0, 190.0, 0.18, true, 14};
    }
}

static_assert(traitsFor(TreeKind::SPRUCE).minimumTemperatureC <= 22.0 &&
              traitsFor(TreeKind::SPRUCE).maximumTemperatureC >= 22.0);

double rangeSuitability(double value, double minimum, double maximum) {
    if (value < minimum || value > maximum) return 0.0;
    const double middle = (minimum + maximum) * 0.5;
    const double halfRange = std::max((maximum - minimum) * 0.5, 0.001);
    return std::clamp(1.0 - std::abs(value - middle) / halfRange * 0.35, 0.0, 1.0);
}

double biomeTreeDensity(Biome biome) {
    switch (biome) {
        case Biome::TROPICAL_RAINFOREST:
            return 0.92;
        case Biome::TEMPERATE_RAINFOREST:
            return 0.82;
        case Biome::FOREST:
        case Biome::BIRCH_FOREST:
            return 0.72;
        case Biome::TAIGA:
            return 0.68;
        case Biome::TEMPERATE_CONIFER_FOREST:
            return 0.74;
        case Biome::TROPICAL_CONIFER_FOREST:
            return 0.62;
        case Biome::TROPICAL_DRY_FOREST:
            return 0.38;
        case Biome::MANGROVE:
            return 0.78;
        case Biome::SWAMP:
            return 0.52;
        case Biome::FLOODED_GRASSLAND:
            return 0.16;
        case Biome::SAVANNA:
            return 0.34;
        case Biome::MEDITERRANEAN_WOODLAND:
            return 0.28;
        case Biome::SHRUBLAND:
            return 0.18;
        case Biome::ALPINE:
            return 0.24;
        case Biome::MONTANE_GRASSLAND:
            return 0.14;
        case Biome::PLAINS:
        case Biome::FLOWER_FIELD:
        case Biome::BEACH:
            return 0.09;
        default:
            return 0.0;
    }
}

double blendedTreeDensity(const worldgen::SurfaceSample& surface) {
    const double primaryDensity = biomeTreeDensity(surface.biome.primary);
    const double secondaryDensity = biomeTreeDensity(surface.biome.secondary);
    const double secondaryWeight =
        worldgen::biomeBlendWeight(surface.biome, surface.biome.secondary);
    if (surface.biome.primary == surface.biome.secondary) return primaryDensity;
    return primaryDensity * (1.0 - secondaryWeight) + secondaryDensity * secondaryWeight;
}

std::pair<BlockType, BlockType> canopyBlocks(Biome biome) {
    switch (biome) {
        case Biome::TROPICAL_RAINFOREST:
        case Biome::TROPICAL_CONIFER_FOREST:
            return {BlockType::JUNGLE_LOG, BlockType::JUNGLE_LEAVES};
        case Biome::TEMPERATE_RAINFOREST:
        case Biome::SWAMP:
        case Biome::FLOODED_GRASSLAND:
            return {BlockType::WILLOW_LOG, BlockType::WILLOW_LEAVES};
        case Biome::MANGROVE:
            return {BlockType::MANGROVE_LOG, BlockType::MANGROVE_LEAVES};
        case Biome::SAVANNA:
        case Biome::TROPICAL_DRY_FOREST:
            return {BlockType::ACACIA_LOG, BlockType::ACACIA_LEAVES};
        case Biome::BEACH:
            return {BlockType::PALM_LOG, BlockType::PALM_LEAVES};
        case Biome::TAIGA:
        case Biome::TEMPERATE_CONIFER_FOREST:
        case Biome::MONTANE_GRASSLAND:
            return {BlockType::SPRUCE_LOG, BlockType::SPRUCE_LEAVES};
        case Biome::BIRCH_FOREST:
            return {BlockType::BIRCH_LOG, BlockType::BIRCH_LEAVES};
        default:
            return {BlockType::LOG, BlockType::LEAVES};
    }
}

Biome ditheredBiome(const worldgen::SurfaceSample& surface, const CounterRng& random,
                    uint64_t stream, int64_t x, int32_t y, int64_t z, uint32_t index = 0) {
    (void)y;
    const double secondaryWeight = std::clamp(surface.biome.transition, 0.0, 0.5);
    return worldgen::multiscaleDitherThreshold(random, stream, x, z, index) < secondaryWeight
               ? surface.biome.secondary
               : surface.biome.primary;
}

TreeKind pickKind(Biome biome, double roll) {
    if (roll > 0.975 && biomeTreeDensity(biome) > 0.2) return TreeKind::FALLEN_LOG;
    switch (biome) {
        case Biome::TROPICAL_RAINFOREST:
            return roll < 0.82 ? TreeKind::JUNGLE : TreeKind::PALM;
        case Biome::TROPICAL_CONIFER_FOREST:
            return roll < 0.58 ? TreeKind::SPRUCE : TreeKind::JUNGLE;
        case Biome::TROPICAL_DRY_FOREST:
            return roll < 0.58 ? TreeKind::ACACIA : TreeKind::JUNGLE;
        case Biome::TEMPERATE_RAINFOREST:
            return roll < 0.62 ? TreeKind::WILLOW : TreeKind::LARGE_OAK;
        case Biome::MANGROVE:
            return TreeKind::MANGROVE;
        case Biome::SAVANNA:
            return TreeKind::ACACIA;
        case Biome::BEACH:
            return TreeKind::PALM;
        case Biome::TAIGA:
        case Biome::TEMPERATE_CONIFER_FOREST:
            return TreeKind::SPRUCE;
        case Biome::BIRCH_FOREST:
            return roll < 0.86 ? TreeKind::BIRCH : TreeKind::OAK;
        case Biome::SWAMP:
        case Biome::FLOODED_GRASSLAND:
            return roll < 0.72 ? TreeKind::WILLOW : TreeKind::OAK;
        case Biome::ALPINE:
        case Biome::MONTANE_GRASSLAND:
            return TreeKind::ALPINE_SCRUB;
        case Biome::MEDITERRANEAN_WOODLAND:
            return roll < 0.18 ? TreeKind::ACACIA : TreeKind::OAK;
        case Biome::FOREST:
            return roll < 0.08   ? TreeKind::LARGE_OAK
                   : roll < 0.78 ? TreeKind::OAK
                                 : TreeKind::BIRCH;
        default:
            return roll < 0.12 ? TreeKind::LARGE_OAK : TreeKind::OAK;
    }
}

double availableSurfaceLight(const worldgen::SurfaceSample& surface) {
    const double canopyShade = blendedTreeDensity(surface) * 0.55;
    const double cloudShade = surface.climate.relativeHumidity * 0.10;
    const double terrainShade = std::min(surface.slope, 2.0) * 0.07;
    return std::clamp(1.0 - canopyShade - cloudShade - terrainShade, 0.18, 1.0);
}

struct TreeCandidate {
    int64_t cellX = 0;
    int64_t cellZ = 0;
    int64_t x = 0;
    int64_t z = 0;
    TreeKind kind = TreeKind::OAK;
    int sampledSurfaceY = SEA_LEVEL;
    double priority = 0.0;
    int spacing = 8;
};

std::optional<TreeCandidate> makeCandidate(const CounterRng& random, int64_t cellX, int64_t cellZ,
                                           const ChunkGenerator& generator) {
    const int offsetX = random.uniformInt(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 0, 0,
                                          feature_generation::TREE_CELL_EDGE - 1);
    const int offsetZ = random.uniformInt(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 1, 0,
                                          feature_generation::TREE_CELL_EDGE - 1);
    const int64_t x = cellX * feature_generation::TREE_CELL_EDGE + offsetX;
    const int64_t z = cellZ * feature_generation::TREE_CELL_EDGE + offsetZ;
    const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);
    const Biome biome = ditheredBiome(surface, random, TREE_BIOME_STREAM, x, 0, z);
    const Biome substrateBiome = worldgen::surface_material::materialBiome(surface, random, x, z);
    const double density = blendedTreeDensity(surface);
    if (density <= 0.0 || substrateBiome == Biome::VOLCANIC_BARREN ||
        worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL)) {
        return std::nullopt;
    }

    const double kindRoll = random.uniform01(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 2);
    const TreeKind kind = pickKind(biome, kindRoll);
    const SpeciesTraits traits = traitsFor(kind);
    const bool flooded =
        surface.hydrology.ocean || surface.hydrology.river || surface.hydrology.lake;
    if (flooded && !traits.toleratesFlooding) return std::nullopt;
    if (surface.slope > traits.maximumSlope || surface.terrainHeight < traits.minimumAltitude ||
        surface.terrainHeight > traits.maximumAltitude ||
        surface.soil.moisture < traits.minimumMoisture ||
        surface.soil.fertility < traits.minimumFertility ||
        availableSurfaceLight(surface) < traits.minimumLight) {
        return std::nullopt;
    }

    const double climateFit = rangeSuitability(
        surface.climate.temperatureC, traits.minimumTemperatureC, traits.maximumTemperatureC);
    const double fertilityFit =
        std::clamp((surface.soil.fertility - traits.minimumFertility) * 1.8 + 0.35, 0.0, 1.0);
    const double moistureFit =
        std::clamp((surface.soil.moisture - traits.minimumMoisture) * 1.5 + 0.4, 0.0, 1.0);
    const double floodplainInfluence =
        worldgen::MacroGenerationSampler::ecotopeInfluence(surface, worldgen::Ecotope::FLOODPLAIN);
    const double floodingFit =
        traits.toleratesFlooding ? 1.0 : std::clamp(1.0 - floodplainInfluence * 0.72, 0.0, 1.0);
    const double volcanicStress =
        std::clamp((surface.geology.volcanicActivity - 0.32) / 0.42, 0.0, 1.0);
    const double acceptance =
        density * climateFit * fertilityFit * moistureFit * floodingFit * (1.0 - volcanicStress);
    if (random.uniform01(TREE_CANDIDATE_STREAM, cellX, 0, cellZ, 3) >= acceptance) {
        return std::nullopt;
    }
    if (!worldgen::surface_material::supportsTreeRooting(generator.surfaceMaterialAt(x, z))) {
        return std::nullopt;
    }

    return TreeCandidate{
        .cellX = cellX,
        .cellZ = cellZ,
        .x = x,
        .z = z,
        .kind = kind,
        .sampledSurfaceY = static_cast<int>(std::floor(surface.terrainHeight)),
        .priority = random.uniform01(TREE_PRIORITY_STREAM, cellX, 0, cellZ),
        .spacing = traits.spacing,
    };
}

bool candidateWins(const TreeCandidate& candidate, const CounterRng& random,
                   const ChunkGenerator& generator,
                   std::unordered_map<ColumnPos, std::optional<TreeCandidate>>& cache) {
    const int search = (MAX_TREE_SPACING + feature_generation::TREE_CELL_EDGE - 1) /
                           feature_generation::TREE_CELL_EDGE +
                       1;
    for (int offsetZ = -search; offsetZ <= search; ++offsetZ) {
        for (int offsetX = -search; offsetX <= search; ++offsetX) {
            const ColumnPos cell{candidate.cellX + offsetX, candidate.cellZ + offsetZ};
            auto [found, inserted] = cache.try_emplace(cell);
            if (inserted) found->second = makeCandidate(random, cell.x, cell.z, generator);
            if (!found->second.has_value()) continue;
            const TreeCandidate& competitor = *found->second;
            if (competitor.cellX == candidate.cellX && competitor.cellZ == candidate.cellZ)
                continue;
            const int64_t dx = competitor.x - candidate.x;
            const int64_t dz = competitor.z - candidate.z;
            const int spacing = std::max(candidate.spacing, competitor.spacing);
            if (dx * dx + dz * dz >= static_cast<int64_t>(spacing * spacing)) continue;
            if (competitor.priority > candidate.priority ||
                (competitor.priority == candidate.priority &&
                 (competitor.cellX < candidate.cellX ||
                  (competitor.cellX == candidate.cellX && competitor.cellZ < candidate.cellZ)))) {
                return false;
            }
        }
    }
    return true;
}

struct TreeWriter {
    Chunk& chunk;
    int64_t baseX;
    int baseY;
    int64_t baseZ;

    void log(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return;
        }
        const BlockType current = chunk.getBlock(lx, ly, lz);
        if (current == BlockType::AIR || isLeafBlock(current) || isFlora(current)) {
            chunk.setBlock(lx, ly, lz, block);
        }
    }

    void leaves(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return;
        }
        if (chunk.getBlock(lx, ly, lz) == BlockType::AIR) chunk.setBlock(lx, ly, lz, block);
    }
};

struct TreeBoundsWriter {
    int64_t minimumLeafX = std::numeric_limits<int64_t>::max();
    int64_t maximumLeafX = std::numeric_limits<int64_t>::min();
    int minimumLeafY = std::numeric_limits<int>::max();
    int maximumLeafY = std::numeric_limits<int>::min();
    int64_t minimumLeafZ = std::numeric_limits<int64_t>::max();
    int64_t maximumLeafZ = std::numeric_limits<int64_t>::min();
    int topY = std::numeric_limits<int>::min();
    BlockType logBlock = BlockType::AIR;
    BlockType leafBlock = BlockType::AIR;
    bool hasFoliage = false;

    void log(int64_t, int y, int64_t, BlockType block) {
        if (logBlock == BlockType::AIR) logBlock = block;
        topY = std::max(topY, y);
    }

    void leaves(int64_t x, int y, int64_t z, BlockType block) {
        if (leafBlock == BlockType::AIR) leafBlock = block;
        hasFoliage = true;
        minimumLeafX = std::min(minimumLeafX, x);
        maximumLeafX = std::max(maximumLeafX, x);
        minimumLeafY = std::min(minimumLeafY, y);
        maximumLeafY = std::max(maximumLeafY, y);
        minimumLeafZ = std::min(minimumLeafZ, z);
        maximumLeafZ = std::max(maximumLeafZ, z);
        topY = std::max(topY, y);
    }
};

int shapeInt(const CounterRng& random, const TreeCandidate& candidate, uint32_t index, int minimum,
             int maximum) {
    return random.uniformInt(TREE_SHAPE_STREAM, candidate.cellX, 0, candidate.cellZ, index, minimum,
                             maximum);
}

double shapeUnit(const CounterRng& random, const TreeCandidate& candidate, uint32_t index) {
    return random.uniform01(TREE_SHAPE_STREAM, candidate.cellX, 0, candidate.cellZ, index);
}

template <typename Writer>
void roundedCanopy(Writer& out, const CounterRng& random, const TreeCandidate& candidate, int64_t x,
                   int top, int64_t z, int radius, BlockType leaves) {
    for (int dy = -2; dy <= 1; ++dy) {
        const int layerRadius = dy >= 1 ? std::max(1, radius - 1) : radius;
        for (int dz = -layerRadius; dz <= layerRadius; ++dz) {
            for (int dx = -layerRadius; dx <= layerRadius; ++dx) {
                const bool corner = std::abs(dx) == layerRadius && std::abs(dz) == layerRadius;
                const uint32_t shapeIndex = 512U + static_cast<uint32_t>(candidate.kind);
                if (corner && random.uniform01(TREE_SHAPE_STREAM, x + dx, top + dy, z + dz,
                                               shapeIndex) < 0.75) {
                    continue;
                }
                out.leaves(x + dx, top + dy, z + dz, leaves);
            }
        }
    }
    out.leaves(x, top + 2, z, leaves);
}

template <typename Writer>
void flatCanopy(Writer& out, int64_t x, int y, int64_t z, int radius, BlockType leaves) {
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz > radius * radius + 1) continue;
            out.leaves(x + dx, y, z + dz, leaves);
            if (std::abs(dx) + std::abs(dz) < radius) out.leaves(x + dx, y + 1, z + dz, leaves);
        }
    }
}

template <typename Writer>
void buildTree(const TreeCandidate& candidate, const CounterRng& random, int baseY, Writer& out) {
    const TreeKind kind = candidate.kind;
    const int64_t x = candidate.x;
    const int64_t z = candidate.z;
    switch (kind) {
        case TreeKind::OAK:
        case TreeKind::LARGE_OAK: {
            const bool large = kind == TreeKind::LARGE_OAK;
            const int height = large ? shapeInt(random, candidate, 0, 8, 11)
                                     : shapeInt(random, candidate, 0, 5, 7);
            const int top = baseY + height;
            roundedCanopy(out, random, candidate, x, top, z, large ? 3 : 2, BlockType::LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::LOG);
            if (large) {
                constexpr std::array<std::array<int, 2>, 4> branches{{
                    {{1, 0}},
                    {{-1, 0}},
                    {{0, 1}},
                    {{0, -1}},
                }};
                for (size_t index = 0; index < branches.size(); ++index) {
                    if (shapeUnit(random, candidate, 16U + static_cast<uint32_t>(index)) < 0.25) {
                        continue;
                    }
                    const int branchY = top - 3 + static_cast<int>(index & 1U);
                    out.log(x + branches[index][0], branchY, z + branches[index][1],
                            BlockType::LOG);
                    out.log(x + branches[index][0] * 2, branchY + 1, z + branches[index][1] * 2,
                            BlockType::LOG);
                }
            }
            break;
        }
        case TreeKind::BIRCH: {
            const int height = shapeInt(random, candidate, 0, 6, 9);
            const int top = baseY + height;
            roundedCanopy(out, random, candidate, x, top, z, 2, BlockType::BIRCH_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::BIRCH_LOG);
            break;
        }
        case TreeKind::SPRUCE: {
            const int height = shapeInt(random, candidate, 0, 8, 12);
            const int top = baseY + height;
            for (int dy = 0; dy <= height - 2; ++dy) {
                int radius = std::min(3, 1 + dy / 2);
                if ((dy & 1) != 0) radius = std::max(1, radius - 1);
                for (int dz = -radius; dz <= radius; ++dz) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (std::abs(dx) == radius && std::abs(dz) == radius && radius > 1)
                            continue;
                        out.leaves(x + dx, top - dy, z + dz, BlockType::SPRUCE_LEAVES);
                    }
                }
            }
            out.leaves(x, top + 1, z, BlockType::SPRUCE_LEAVES);
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::SPRUCE_LOG);
            break;
        }
        case TreeKind::ACACIA: {
            const int height = shapeInt(random, candidate, 0, 6, 9);
            const int bendX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
            const int bendZ = shapeUnit(random, candidate, 2) < 0.5 ? -1 : 1;
            for (int y = 0; y < height; ++y) {
                const int step = y > height / 2 ? (y - height / 2 + 1) / 2 : 0;
                out.log(x + bendX * step, baseY + y, z + bendZ * step, BlockType::ACACIA_LOG);
            }
            const int64_t crownX = x + bendX * ((height - height / 2) / 2);
            const int64_t crownZ = z + bendZ * ((height - height / 2) / 2);
            flatCanopy(out, crownX, baseY + height, crownZ, 3, BlockType::ACACIA_LEAVES);
            out.log(crownX - bendZ, baseY + height - 1, crownZ + bendX, BlockType::ACACIA_LOG);
            flatCanopy(out, crownX - bendZ * 2, baseY + height, crownZ + bendX * 2, 2,
                       BlockType::ACACIA_LEAVES);
            break;
        }
        case TreeKind::JUNGLE: {
            const int height = shapeInt(random, candidate, 0, 12, 18);
            for (int y = 0; y < height; ++y) {
                out.log(x, baseY + y, z, BlockType::JUNGLE_LOG);
                out.log(x + 1, baseY + y, z, BlockType::JUNGLE_LOG);
                out.log(x, baseY + y, z + 1, BlockType::JUNGLE_LOG);
                out.log(x + 1, baseY + y, z + 1, BlockType::JUNGLE_LOG);
            }
            for (int direction = 0; direction < 4; ++direction) {
                const int dx = direction == 0 ? 1 : direction == 1 ? -1 : 0;
                const int dz = direction == 2 ? 1 : direction == 3 ? -1 : 0;
                out.log(x + dx, baseY, z + dz, BlockType::JUNGLE_LOG);
                out.log(x + dx * 2, baseY - 1, z + dz * 2, BlockType::JUNGLE_LOG);
                const int branchY = baseY + height - 3 - (direction & 1);
                for (int step = 1; step <= 3; ++step) {
                    out.log(x + dx * step, branchY + step / 2, z + dz * step,
                            BlockType::JUNGLE_LOG);
                }
                flatCanopy(out, x + dx * 3, branchY + 2, z + dz * 3, 2, BlockType::JUNGLE_LEAVES);
            }
            flatCanopy(out, x, baseY + height, z, 4, BlockType::JUNGLE_LEAVES);
            break;
        }
        case TreeKind::MANGROVE: {
            const int height = shapeInt(random, candidate, 0, 7, 10);
            for (int y = 0; y < height; ++y)
                out.log(x, baseY + y, z, BlockType::MANGROVE_LOG);
            constexpr std::array<std::array<int, 2>, 8> roots{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& root : roots) {
                out.log(x + root[0], baseY, z + root[1], BlockType::MANGROVE_LOG);
                out.log(x + root[0] * 2, baseY - 1, z + root[1] * 2, BlockType::MANGROVE_LOG);
            }
            roundedCanopy(out, random, candidate, x, baseY + height, z, 3,
                          BlockType::MANGROVE_LEAVES);
            break;
        }
        case TreeKind::PALM: {
            const int height = shapeInt(random, candidate, 0, 8, 12);
            const int leanX = shapeUnit(random, candidate, 1) < 0.5 ? -1 : 1;
            for (int y = 0; y < height; ++y) {
                const int bend = y > height * 2 / 3 ? 1 : 0;
                out.log(x + leanX * bend, baseY + y, z, BlockType::PALM_LOG);
            }
            const int64_t topX = x + leanX;
            const int topY = baseY + height;
            out.leaves(topX, topY + 1, z, BlockType::PALM_LEAVES);
            constexpr std::array<std::array<int, 2>, 8> fronds{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& frond : fronds) {
                for (int step = 1; step <= 4; ++step) {
                    const int drop = step >= 3 ? 1 : 0;
                    out.leaves(topX + frond[0] * step, topY - drop, z + frond[1] * step,
                               BlockType::PALM_LEAVES);
                }
            }
            break;
        }
        case TreeKind::WILLOW: {
            const int height = shapeInt(random, candidate, 0, 7, 10);
            const int top = baseY + height;
            for (int y = baseY; y < top; ++y)
                out.log(x, y, z, BlockType::WILLOW_LOG);
            flatCanopy(out, x, top, z, 3, BlockType::WILLOW_LEAVES);
            constexpr std::array<std::array<int, 2>, 8> directions{{
                {{1, 0}},
                {{-1, 0}},
                {{0, 1}},
                {{0, -1}},
                {{1, 1}},
                {{-1, 1}},
                {{1, -1}},
                {{-1, -1}},
            }};
            for (const auto& direction : directions) {
                out.log(x + direction[0], top - 2, z + direction[1], BlockType::WILLOW_LOG);
                for (int drop = 0; drop < 4; ++drop) {
                    out.leaves(x + direction[0] * 3, top - drop, z + direction[1] * 3,
                               BlockType::WILLOW_LEAVES);
                }
            }
            break;
        }
        case TreeKind::ALPINE_SCRUB: {
            const int radius = shapeInt(random, candidate, 0, 1, 2);
            for (int dz = -radius; dz <= radius; ++dz) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dz * dz > radius * radius + 1) continue;
                    const uint32_t index = 160U + static_cast<uint32_t>((dx + 2) * 5 + dz + 2);
                    if (shapeUnit(random, candidate, index) < 0.28) continue;
                    out.leaves(x + dx, baseY, z + dz, BlockType::SHRUB);
                    if (dx == 0 && dz == 0 && shapeUnit(random, candidate, index + 32U) > 0.42) {
                        out.leaves(x, baseY + 1, z, BlockType::SHRUB);
                    }
                }
            }
            break;
        }
        case TreeKind::FALLEN_LOG: {
            const bool alongX = shapeUnit(random, candidate, 0) < 0.5;
            const int length = shapeInt(random, candidate, 1, 4, 7);
            const BlockType log =
                shapeUnit(random, candidate, 2) < 0.5 ? BlockType::LOG : BlockType::WILLOW_LOG;
            for (int step = 0; step < length; ++step) {
                out.log(x + (alongX ? step : 0), baseY, z + (alongX ? 0 : step), log);
            }
            break;
        }
    }
}

std::optional<int> acceptedTreeBaseY(const TreeCandidate& candidate,
                                     const ChunkGenerator& generator,
                                     const StructurePlacer& structures, GenScratch& scratch) {
    const int surfaceY = generator.surfaceYAt(candidate.x, candidate.z, scratch);
    if (std::abs(surfaceY - candidate.sampledSurfaceY) >
        feature_generation::TREE_MAXIMUM_SURFACE_DEVIATION) {
        return std::nullopt;
    }
    const int64_t anchorChunkX = Chunk::worldToChunk(candidate.x);
    const int64_t anchorChunkZ = Chunk::worldToChunk(candidate.z);
    if (structures.insideStructure(candidate.x, candidate.z, anchorChunkX, anchorChunkZ, generator,
                                   scratch, 1)) {
        return std::nullopt;
    }
    return surfaceY + 1;
}

struct DescribedFarCanopy {
    FarCanopy canopy;
    int64_t minimumX = 0;
    int64_t maximumX = 0;
    int64_t minimumZ = 0;
    int64_t maximumZ = 0;
};

std::optional<DescribedFarCanopy> describeFarCanopy(const TreeCandidate& candidate,
                                                    const CounterRng& random, int baseY) {
    TreeBoundsWriter bounds;
    buildTree(candidate, random, baseY, bounds);
    if (!bounds.hasFoliage) return std::nullopt;

    const int64_t canopyCenterX =
        bounds.minimumLeafX + (bounds.maximumLeafX - bounds.minimumLeafX) / 2;
    const int64_t canopyCenterZ =
        bounds.minimumLeafZ + (bounds.maximumLeafZ - bounds.minimumLeafZ) / 2;
    const int64_t radius =
        std::max({canopyCenterX - bounds.minimumLeafX, bounds.maximumLeafX - canopyCenterX,
                  canopyCenterZ - bounds.minimumLeafZ, bounds.maximumLeafZ - canopyCenterZ});

    DescribedFarCanopy result;
    result.canopy = {
        .x = candidate.x,
        .z = candidate.z,
        .baseY = baseY,
        .topY = bounds.topY,
        .canopyMinimumY = bounds.minimumLeafY,
        .canopyMaximumY = bounds.maximumLeafY,
        .canopyOffsetX = static_cast<int8_t>(canopyCenterX - candidate.x),
        .canopyOffsetZ = static_cast<int8_t>(canopyCenterZ - candidate.z),
        .canopyRadius = static_cast<uint8_t>(radius),
        .logBlock = bounds.logBlock,
        .leafBlock = bounds.leafBlock,
        .anchorId = random.u64(TREE_PRIORITY_STREAM, candidate.cellX, 0, candidate.cellZ),
    };
    result.minimumX = bounds.minimumLeafX;
    result.maximumX = bounds.maximumLeafX;
    result.minimumZ = bounds.minimumLeafZ;
    result.maximumZ = bounds.maximumLeafZ;
    return result;
}

bool intersectsCubeVertically(TreeKind kind, int baseY, int cubeBaseY) {
    int minimum = baseY + feature_generation::TREE_MINIMUM_VERTICAL_OFFSET;
    if (kind != TreeKind::JUNGLE && kind != TreeKind::MANGROVE) minimum = baseY;
    int maximum = baseY + feature_generation::TREE_MAXIMUM_VERTICAL_OFFSET;
    return maximum >= cubeBaseY && minimum < cubeBaseY + CHUNK_EDGE;
}

struct FloraWriter {
    Chunk& chunk;
    int64_t baseX;
    int baseY;
    int64_t baseZ;

    bool setIfAir(int64_t x, int y, int64_t z, BlockType block) const {
        const int lx = static_cast<int>(x - baseX);
        const int ly = y - baseY;
        const int lz = static_cast<int>(z - baseZ);
        if (lx < 0 || lx >= CHUNK_EDGE || ly < 0 || ly >= CHUNK_EDGE || lz < 0 ||
            lz >= CHUNK_EDGE) {
            return false;
        }
        if (chunk.getBlock(lx, ly, lz) != BlockType::AIR) return false;
        chunk.setBlock(lx, ly, lz, block);
        return true;
    }
};

} // namespace

FeaturePlacer::FeaturePlacer(uint32_t worldSeed) : random_(worldSeed) {}

void FeaturePlacer::placeTrees(Chunk& chunk, const ChunkGenerator& generator,
                               const StructurePlacer& structures, GenScratch& scratch) const {
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    TreeWriter writer{chunk, baseX, baseY, baseZ};
    std::unordered_map<ColumnPos, std::optional<TreeCandidate>> cache;

    const int64_t minimumCellX =
        world_coord::floorDiv(baseX - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellX = world_coord::floorDiv(
        baseX + CHUNK_EDGE - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
        static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t minimumCellZ =
        world_coord::floorDiv(baseZ - feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
                              static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellZ = world_coord::floorDiv(
        baseZ + CHUNK_EDGE - 1 + feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH,
        static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));

    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const ColumnPos cell{cellX, cellZ};
            auto [found, inserted] = cache.try_emplace(cell);
            if (inserted) found->second = makeCandidate(random_, cellX, cellZ, generator);
            if (!found->second.has_value()) continue;
            const TreeCandidate candidate = *found->second;
            if (!candidateWins(candidate, random_, generator, cache)) continue;

            const std::optional<int> treeBaseY =
                acceptedTreeBaseY(candidate, generator, structures, scratch);
            if (!treeBaseY.has_value()) continue;
            if (!intersectsCubeVertically(candidate.kind, *treeBaseY, baseY)) continue;
            buildTree(candidate, random_, *treeBaseY, writer);
        }
    }
}

std::vector<FarCanopy> FeaturePlacer::collectFarCanopies(int64_t minimumX, int64_t minimumZ,
                                                         int64_t maximumX, int64_t maximumZ,
                                                         const ChunkGenerator& generator,
                                                         const StructurePlacer& structures,
                                                         GenScratch& scratch) const {
    std::vector<FarCanopy> result;
    if (minimumX >= maximumX || minimumZ >= maximumZ) return result;

    constexpr int64_t REACH = feature_generation::TREE_MAXIMUM_HORIZONTAL_REACH;
    const int64_t expandedMinimumX = minimumX < std::numeric_limits<int64_t>::min() + REACH
                                         ? std::numeric_limits<int64_t>::min()
                                         : minimumX - REACH;
    const int64_t expandedMinimumZ = minimumZ < std::numeric_limits<int64_t>::min() + REACH
                                         ? std::numeric_limits<int64_t>::min()
                                         : minimumZ - REACH;
    const int64_t lastX = maximumX - 1;
    const int64_t lastZ = maximumZ - 1;
    const int64_t expandedMaximumX = lastX > std::numeric_limits<int64_t>::max() - REACH
                                         ? std::numeric_limits<int64_t>::max()
                                         : lastX + REACH;
    const int64_t expandedMaximumZ = lastZ > std::numeric_limits<int64_t>::max() - REACH
                                         ? std::numeric_limits<int64_t>::max()
                                         : lastZ + REACH;

    const int64_t minimumCellX = world_coord::floorDiv(
        expandedMinimumX, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellX = world_coord::floorDiv(
        expandedMaximumX, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t minimumCellZ = world_coord::floorDiv(
        expandedMinimumZ, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));
    const int64_t maximumCellZ = world_coord::floorDiv(
        expandedMaximumZ, static_cast<int64_t>(feature_generation::TREE_CELL_EDGE));

    std::unordered_map<ColumnPos, std::optional<TreeCandidate>> cache;
    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const ColumnPos cell{cellX, cellZ};
            auto [found, inserted] = cache.try_emplace(cell);
            if (inserted) found->second = makeCandidate(random_, cellX, cellZ, generator);
            if (!found->second.has_value()) continue;
            const TreeCandidate candidate = *found->second;
            if (!candidateWins(candidate, random_, generator, cache)) continue;

            const std::optional<int> treeBaseY =
                acceptedTreeBaseY(candidate, generator, structures, scratch);
            if (!treeBaseY.has_value()) continue;
            const std::optional<DescribedFarCanopy> described =
                describeFarCanopy(candidate, random_, *treeBaseY);
            if (!described.has_value()) continue;
            if (described->maximumX < minimumX || described->minimumX >= maximumX ||
                described->maximumZ < minimumZ || described->minimumZ >= maximumZ) {
                continue;
            }
            result.push_back(described->canopy);
        }
    }
    return result;
}

std::vector<FarCanopy>
FeaturePlacer::collectFarCanopyClusters(int64_t minimumX, int64_t minimumZ, int64_t maximumX,
                                        int64_t maximumZ, int lodStep,
                                        const ChunkGenerator& generator) const {
    std::vector<FarCanopy> result;
    if (minimumX >= maximumX || minimumZ >= maximumZ) return result;
    const int64_t cellEdge = lodStep >= 16 ? 64 : 32;
    const int64_t minimumCellX = world_coord::floorDiv(minimumX, cellEdge);
    const int64_t maximumCellX = world_coord::floorDiv(maximumX - 1, cellEdge);
    const int64_t minimumCellZ = world_coord::floorDiv(minimumZ, cellEdge);
    const int64_t maximumCellZ = world_coord::floorDiv(maximumZ - 1, cellEdge);
    result.reserve(
        static_cast<size_t>((maximumCellX - minimumCellX + 1) * (maximumCellZ - minimumCellZ + 1)));

    for (int64_t cellZ = minimumCellZ; cellZ <= maximumCellZ; ++cellZ) {
        for (int64_t cellX = minimumCellX; cellX <= maximumCellX; ++cellX) {
            const int offsetX = random_.uniformInt(FAR_CANOPY_CLUSTER_STREAM, cellX, 0, cellZ, 0, 3,
                                                   static_cast<int>(cellEdge) - 4);
            const int offsetZ = random_.uniformInt(FAR_CANOPY_CLUSTER_STREAM, cellX, 0, cellZ, 1, 3,
                                                   static_cast<int>(cellEdge) - 4);
            const int64_t x = cellX * cellEdge + offsetX;
            const int64_t z = cellZ * cellEdge + offsetZ;
            if (x < minimumX || x >= maximumX || z < minimumZ || z >= maximumZ) continue;

            const worldgen::SurfaceSample surface = generator.sampleFarSurface(x, z);
            const Biome biome = ditheredBiome(surface, random_, TREE_BIOME_STREAM, x, 0, z);
            const Biome substrateBiome =
                worldgen::surface_material::materialBiome(surface, random_, x, z);
            const double density = blendedTreeDensity(surface);
            if (density <= 0.0 || surface.hydrology.ocean || surface.hydrology.river ||
                surface.hydrology.lake || surface.slope > 1.55 ||
                substrateBiome == Biome::VOLCANIC_BARREN ||
                worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL) ||
                random_.uniform01(FAR_CANOPY_CLUSTER_STREAM, cellX, 0, cellZ, 2) >=
                    std::min(1.0, density * 1.18) ||
                !worldgen::surface_material::supportsTreeRooting(
                    generator.farSurfaceMaterialAt(x, z))) {
                continue;
            }

            const auto [logBlock, leafBlock] = canopyBlocks(biome);
            const int baseY = static_cast<int>(std::floor(surface.terrainHeight)) + 1;
            const int height = lodStep >= 16 ? 12 : 9;
            const int radius = static_cast<int>(
                std::lround((lodStep >= 16 ? 8.0 : 4.5) + density * (lodStep >= 16 ? 5.0 : 3.0)));
            result.push_back({
                .x = x,
                .z = z,
                .baseY = baseY,
                .topY = baseY + height,
                .canopyMinimumY = baseY + height / 3,
                .canopyMaximumY = baseY + height,
                .canopyOffsetX = 0,
                .canopyOffsetZ = 0,
                .canopyRadius = static_cast<uint8_t>(radius),
                .logBlock = logBlock,
                .leafBlock = leafBlock,
                .anchorId = random_.u64(FAR_CANOPY_CLUSTER_STREAM, cellX, 0, cellZ, 3),
                .aggregate = true,
            });
        }
    }
    return result;
}

void FeaturePlacer::placeFlora(Chunk& chunk, const ChunkGenerator& generator,
                               GenScratch& scratch) const {
    const int64_t baseX = chunk.chunkX * CHUNK_EDGE;
    const int baseY = chunk.chunkY * CHUNK_EDGE;
    const int64_t baseZ = chunk.chunkZ * CHUNK_EDGE;
    FloraWriter writer{chunk, baseX, baseY, baseZ};

    for (int localZ = 0; localZ < CHUNK_EDGE; ++localZ) {
        for (int localX = 0; localX < CHUNK_EDGE; ++localX) {
            const int64_t x = baseX + localX;
            const int64_t z = baseZ + localZ;
            const worldgen::SurfaceSample surface = generator.sampleSurface(x, z);
            const int terrainY = generator.surfaceYAt(x, z, scratch);
            const double roll = random_.uniform01(FLORA_STREAM, x, terrainY, z, 0);
            const double kindRoll = random_.uniform01(FLORA_STREAM, x, terrainY, z, 1);
            const Biome biome = ditheredBiome(surface, random_, FLORA_BIOME_STREAM, x, terrainY, z);
            const Biome substrateBiome =
                worldgen::surface_material::materialBiome(surface, random_, x, z);
            const double barrenWeight =
                worldgen::biomeBlendWeight(surface.biome, Biome::VOLCANIC_BARREN);
            const double volcanicStress =
                std::max(barrenWeight,
                         std::clamp((surface.geology.volcanicActivity - 0.32) / 0.42, 0.0, 1.0));
            const double growthFit = 1.0 - volcanicStress;
            if (growthFit <= 0.01 || substrateBiome == Biome::VOLCANIC_BARREN ||
                worldgen::hasEcotope(surface.ecotopes, worldgen::Ecotope::GEOTHERMAL)) {
                continue;
            }
            std::optional<bool> floraSupport;
            auto supportsFlora = [&] {
                if (!floraSupport.has_value()) {
                    floraSupport = worldgen::surface_material::supportsSurfaceFlora(
                        generator.surfaceMaterialAt(x, z));
                }
                return *floraSupport;
            };

            if ((surface.hydrology.lake || biome == Biome::SWAMP || biome == Biome::MANGROVE) &&
                !surface.hydrology.waterfall) {
                const int waterTop = static_cast<int>(std::ceil(surface.waterSurface)) - 1;
                if (roll < 0.055 * growthFit) {
                    writer.setIfAir(x, waterTop + 1, z, BlockType::LILY_PAD);
                }
            }

            const double riparianInfluence = std::max(
                {worldgen::MacroGenerationSampler::ecotopeInfluence(surface,
                                                                    worldgen::Ecotope::RIVERBANK),
                 worldgen::MacroGenerationSampler::ecotopeInfluence(surface,
                                                                    worldgen::Ecotope::LAKESHORE),
                 worldgen::MacroGenerationSampler::ecotopeInfluence(surface,
                                                                    worldgen::Ecotope::FLOODPLAIN),
                 worldgen::biomeBlendWeight(surface.biome, Biome::MANGROVE),
                 worldgen::biomeBlendWeight(surface.biome, Biome::FLOODED_GRASSLAND)});
            if (surface.soil.moisture > 0.50 && roll < 0.22 * riparianInfluence * growthFit) {
                if (!supportsFlora()) continue;
                const BlockType plant = kindRoll < 0.58 ? BlockType::CATTAIL : BlockType::REED;
                const int height =
                    plant == BlockType::REED ? 2 + static_cast<int>(kindRoll * 3.0) : 2;
                for (int offset = 1; offset <= height; ++offset) {
                    if (!writer.setIfAir(x, terrainY + offset, z, plant)) break;
                }
                continue;
            }

            if (biome == Biome::ALPINE || biome == Biome::MONTANE_GRASSLAND) {
                const double slopeFit = std::clamp(1.0 - surface.slope / 1.65, 0.0, 1.0);
                const double scrubSuitability =
                    std::clamp(surface.soil.moisture * 0.30 + surface.soil.fertility * 0.34 +
                                   slopeFit * 0.24 + availableSurfaceLight(surface) * 0.12,
                               0.0, 1.0);
                if (roll < scrubSuitability * 0.26 * growthFit && supportsFlora()) {
                    writer.setIfAir(x, terrainY + 1, z, BlockType::SHRUB);
                } else if (roll < scrubSuitability * 0.34 * growthFit && kindRoll > 0.55 &&
                           supportsFlora()) {
                    writer.setIfAir(x, terrainY + 1, z, BlockType::FERN);
                }
                continue;
            }

            if (biome == Biome::DESERT || biome == Biome::COLD_DESERT || biome == Biome::BADLANDS) {
                if (roll < 0.018 * growthFit && supportsFlora()) {
                    const int height = 1 + static_cast<int>(kindRoll * 3.0);
                    for (int offset = 1; offset <= height; ++offset) {
                        if (!writer.setIfAir(x, terrainY + offset, z, BlockType::CACTUS)) break;
                    }
                } else if (roll < 0.055 * growthFit && supportsFlora()) {
                    writer.setIfAir(x, terrainY + 1, z,
                                    kindRoll < 0.45 ? BlockType::SUCCULENT : BlockType::DEAD_BUSH);
                }
                continue;
            }

            const double vegetation =
                std::clamp(surface.soil.moisture * 0.45 + surface.soil.fertility * 0.45 +
                               surface.climate.relativeHumidity * 0.10 - surface.slope * 0.12,
                           0.0, 0.72) *
                growthFit;
            if (roll >= vegetation) continue;
            if (!supportsFlora()) continue;
            BlockType plant = BlockType::TALL_GRASS;
            if (biome == Biome::TROPICAL_RAINFOREST || biome == Biome::TEMPERATE_RAINFOREST ||
                biome == Biome::TAIGA || biome == Biome::TEMPERATE_CONIFER_FOREST ||
                biome == Biome::TROPICAL_CONIFER_FOREST) {
                plant = kindRoll < 0.62 ? BlockType::FERN : BlockType::SHRUB;
            } else if (biome == Biome::SHRUBLAND || biome == Biome::STEPPE ||
                       biome == Biome::SAVANNA || biome == Biome::MEDITERRANEAN_WOODLAND ||
                       biome == Biome::TROPICAL_DRY_FOREST) {
                plant = kindRoll < 0.52 ? BlockType::SHRUB : BlockType::TALL_GRASS;
            } else if (kindRoll > 0.87) {
                plant = BlockType::FLOWER_BLUE;
            } else if (kindRoll > 0.77) {
                plant = BlockType::FLOWER_RED;
            } else if (kindRoll > 0.67) {
                plant = BlockType::FLOWER_YELLOW;
            } else if (surface.climate.relativeHumidity > 0.78 && kindRoll < 0.08) {
                plant = BlockType::MUSHROOM_BROWN;
            }
            writer.setIfAir(x, terrainY + 1, z, plant);
        }
    }
}
