#include "world/alpine_morphology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace worldgen {
namespace {

constexpr uint64_t ALPINE_NOISE_STREAM = 0x414C'5049'4E45'4E4FULL;
constexpr uint64_t CIRQUE_CANDIDATE_STREAM = 0x4349'5251'5545'4341ULL;
constexpr double CIRQUE_CELL_EDGE = 640.0;

double clamp01(double value) noexcept {
    return std::clamp(value, 0.0, 1.0);
}

double smoothstep(double edge0, double edge1, double value) noexcept {
    if (edge0 == edge1) return value < edge0 ? 0.0 : 1.0;
    const double t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

double bell(double value, double center, double radius) noexcept {
    if (radius <= 0.0) return value == center ? 1.0 : 0.0;
    const double normalized = (value - center) / radius;
    return std::exp(-2.0 * normalized * normalized);
}

double unitFromWord(uint32_t value) noexcept {
    return static_cast<double>(value) * (1.0 / 4'294'967'296.0);
}

int64_t boundedFloor(double value) noexcept {
    constexpr double MINIMUM = -0x1p63;
    constexpr double MAXIMUM_EXCLUSIVE = 0x1p63;
    if (!std::isfinite(value)) return 0;
    if (value <= MINIMUM) return std::numeric_limits<int64_t>::min() + 2;
    if (value >= MAXIMUM_EXCLUSIVE) return std::numeric_limits<int64_t>::max() - 2;
    return static_cast<int64_t>(std::floor(value));
}

struct RotatedPosition {
    double along = 0.0;
    double across = 0.0;
};

RotatedPosition rotate(double x, double z, double angle) noexcept {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {
        .along = x * cosine + z * sine,
        .across = -x * sine + z * cosine,
    };
}

double ridgeKernel(double value) noexcept {
    // A zero-width crest avoids the mesas produced by clamping a finite noise
    // interval to one, while the smooth derivative prevents needle spikes.
    return 1.0 - smoothstep(0.0, 0.88, std::abs(value));
}

double geologicalGate(double uplift, double resistance, double continentalFraction) noexcept {
    const double resistantBedrock = smoothstep(0.30, 1.15, resistance);
    // Strong continental convergence can fold weaker sedimentary packages
    // into a recognizable range, but crystalline or volcanic rock retains
    // substantially more crest relief.
    const double competence = std::lerp(0.22, 1.0, resistantBedrock);
    return smoothstep(0.24, 0.82, uplift) * competence *
           smoothstep(0.38, 0.76, continentalFraction);
}

double glacierClimateGate(double terrainHeight, double temperatureC,
                          double precipitationMm) noexcept {
    const double elevation = smoothstep(132.0, 238.0, terrainHeight);
    const double cold = 1.0 - smoothstep(1.5, 7.0, temperatureC);
    const double snowSupply = smoothstep(220.0, 980.0, precipitationMm);
    return elevation * cold * snowSupply;
}

} // namespace

AlpineErosionResponse alpineErosionResponse(const AlpineErosionContext& context) noexcept {
    const double uplift = smoothstep(0.20, 0.82, context.uplift);
    const double resistance = smoothstep(0.42, 1.28, context.rockResistance);
    const double convergence = clamp01(context.drainageConvergence);
    const double moisture = smoothstep(180.0, 1'150.0, context.annualPrecipitationMm);
    const double cold = 1.0 - smoothstep(1.5, 7.0, context.temperatureC);
    const double alpineElevation = smoothstep(118.0, 220.0, context.terrainHeight);
    const double glacial =
        cold * moisture * alpineElevation * smoothstep(0.20, 0.85, context.uplift);
    const double frostBand = bell(context.temperatureC, -1.0, 5.5) * moisture * alpineElevation;
    const double exposedSlope = smoothstep(0.48, 1.10, context.slope);

    AlpineErosionResponse result;
    result.ridgePreservation = uplift * resistance * (1.0 - convergence * 0.82);
    result.glacialCompetition = glacial;
    result.periglacialWeathering = frostBand * exposedSlope;

    // Stream power remains effective in convergent drainage, but resistant
    // high divides are not eroded as though they were weak alluvium.
    result.streamIncisionScale = std::clamp(0.34 + convergence * 0.66 + (1.0 - resistance) * 0.28 -
                                                result.ridgePreservation * 0.22,
                                            0.22, 1.18);

    const double soilMantle = moisture * (1.0 - resistance * 0.78);
    const double channelColluvium = soilMantle * (0.18 + convergence * 0.82);
    const double frostRockfall = result.periglacialWeathering * (1.0 - glacial * 0.55) * 0.46;
    result.thermalRelaxationScale = std::clamp(
        (channelColluvium + frostRockfall) * (1.0 - result.ridgePreservation * 0.88), 0.0, 1.0);

    // Approximately 30 to 50 degrees at the 16-block basin raster, with
    // resistant exposed rock permitted to stand steeper than mobile talus.
    result.criticalSlope = std::lerp(0.58, 1.18, resistance);
    return result;
}

AlpineMorphologySampler::AlpineMorphologySampler(uint64_t worldSeed)
    : cacheTag_(worldSeed)
    , random_(worldSeed)
    , warpNoise_(random_.u32(ALPINE_NOISE_STREAM, 0, 0, 0, 0))
    , ridgeNoise_(random_.u32(ALPINE_NOISE_STREAM, 0, 0, 0, 1))
    , detailNoise_(random_.u32(ALPINE_NOISE_STREAM, 0, 0, 0, 2))
    , hornNoise_(random_.u32(ALPINE_NOISE_STREAM, 0, 0, 0, 3)) {}

AlpineTectonicSample
AlpineMorphologySampler::sampleTectonic(const AlpineTectonicContext& context) const noexcept {
    AlpineTectonicSample result;
    result.upliftGate =
        geologicalGate(context.uplift, context.rockResistance, context.continentalFraction);
    if (result.upliftGate <= 1.0e-6 || !std::isfinite(context.x) || !std::isfinite(context.z)) {
        return result;
    }

    // Rotate only globally defined coordinate fields. Rotating absolute world
    // coordinates by a query-local plate orientation would phase-shear the
    // noise wherever the orientation changed. Four fixed oblique networks
    // let the continuous tectonic uplift envelope supply the range direction
    // without introducing a storage-axis preference or a plate seam.
    const double warpedX =
        context.x + warpNoise_.noise2D(context.x / 3'700.0, context.z / 3'100.0) * 260.0;
    const double warpedZ =
        context.z +
        warpNoise_.noise2D(context.x / 2'900.0 + 31.0, context.z / 4'100.0 - 17.0) * 210.0;
    constexpr std::array<double, 4> NETWORK_ANGLES = {0.37, 1.1553981634, 1.9407963268,
                                                      2.7261944902};
    constexpr std::array<double, 4> NETWORK_OFFSETS = {0.0, 31.0, -47.0, 73.0};
    std::array<double, 4> networks{};
    for (size_t index = 0; index < networks.size(); ++index) {
        const RotatedPosition position = rotate(warpedX, warpedZ, NETWORK_ANGLES[index]);
        networks[index] = ridgeKernel(
            ridgeNoise_.noise2D(position.along / 1'420.0 + NETWORK_OFFSETS[index],
                                position.across / 380.0 - NETWORK_OFFSETS[index] * 0.61));
    }
    double networkMean = 0.0;
    double highOrderMean = 0.0;
    for (const double network : networks) {
        networkMean += network / static_cast<double>(networks.size());
        const double squared = network * network;
        highOrderMean += squared * squared / static_cast<double>(networks.size());
    }
    double pairMean = 0.0;
    size_t pairCount = 0;
    for (size_t first = 0; first < networks.size(); ++first) {
        for (size_t second = first + 1; second < networks.size(); ++second) {
            pairMean += networks[first] * networks[second];
            ++pairCount;
        }
    }
    pairMean /= static_cast<double>(pairCount);
    const double smoothStrongest = std::sqrt(std::sqrt(highOrderMean));
    const double intersection = std::sqrt(pairMean);
    const double connected = smoothStrongest * 0.58 + networkMean * 0.42;
    result.ridgeStrength = smoothstep(0.36, 1.0, connected);
    const double hornSelector =
        smoothstep(-0.18, 1.0, hornNoise_.noise2D(warpedX / 1'900.0, warpedZ / 1'700.0));
    result.hornStrength = smoothstep(0.48, 1.0, intersection) * hornSelector;
    result.elevationOffset =
        result.upliftGate * (result.ridgeStrength * 34.0 + result.hornStrength * 46.0);
    return result;
}

AlpineMorphologySampler::CirqueCandidate
AlpineMorphologySampler::cirqueCandidate(int64_t cellX, int64_t cellZ) const noexcept {
    struct CacheEntry {
        const AlpineMorphologySampler* owner = nullptr;
        uint64_t tag = 0;
        int64_t cellX = 0;
        int64_t cellZ = 0;
        CirqueCandidate candidate;
    };
    constexpr size_t CACHE_SIZE = 64;
    thread_local std::array<CacheEntry, CACHE_SIZE> cache;
    const uint64_t mixedX = static_cast<uint64_t>(cellX) * 0x9E37'79B9'7F4A'7C15ULL;
    const uint64_t mixedZ = static_cast<uint64_t>(cellZ) * 0xBF58'476D'1CE4'E5B9ULL;
    CacheEntry& entry = cache[static_cast<size_t>((mixedX ^ mixedZ) & (CACHE_SIZE - 1))];
    if (entry.owner == this && entry.tag == cacheTag_ && entry.cellX == cellX &&
        entry.cellZ == cellZ) {
        return entry.candidate;
    }

    const CounterRng::Block values = random_.block(CIRQUE_CANDIDATE_STREAM, cellX, 0, cellZ);
    const double jitterX = 0.18 + unitFromWord(values[0]) * 0.64;
    const double jitterZ = 0.18 + unitFromWord(values[1]) * 0.64;
    const CirqueCandidate candidate{
        .x = (static_cast<double>(cellX) + jitterX) * CIRQUE_CELL_EDGE,
        .z = (static_cast<double>(cellZ) + jitterZ) * CIRQUE_CELL_EDGE,
        .orientation = unitFromWord(values[2]) * 2.0 * std::numbers::pi,
        .radius = 112.0 + unitFromWord(values[3]) * 104.0,
        .strength = 0.62 + unitFromWord(values[0] ^ values[2]) * 0.38,
    };
    entry = {
        .owner = this,
        .tag = cacheTag_,
        .cellX = cellX,
        .cellZ = cellZ,
        .candidate = candidate,
    };
    return candidate;
}

double
AlpineMorphologySampler::filteredRidgeDetail(const AlpineSurfaceContext& context,
                                             const AlpineTectonicSample& tectonic) const noexcept {
    // The broad 256-block component is already represented by the warped
    // tectonic ridge sample. These four bands restore crest detail without
    // paying to reconstruct that macro term at every exact or step-2 point.
    constexpr std::array<double, ALPINE_RIDGE_DETAIL_BAND_COUNT> WAVELENGTHS = {128.0, 64.0, 32.0,
                                                                                16.0};
    constexpr std::array<double, ALPINE_RIDGE_DETAIL_BAND_COUNT> AMPLITUDES = {6.5, 3.5, 1.75,
                                                                               0.85};
    constexpr std::array<double, ALPINE_RIDGE_DETAIL_BAND_COUNT> ROTATIONS = {0.73, 1.31, 1.97,
                                                                              2.61};
    const double support = static_cast<double>(std::clamp(context.footprintWidth, 1, 32));
    double detail = 0.0;
    for (size_t index = 0; index < WAVELENGTHS.size(); ++index) {
        const double retained = smoothstep(support, support * 2.0, WAVELENGTHS[index]);
        if (retained <= 0.0) continue;
        const RotatedPosition position = rotate(context.x, context.z, ROTATIONS[index]);
        const double ridge = ridgeKernel(detailNoise_.noise2D(
            position.along / WAVELENGTHS[index], position.across / (WAVELENGTHS[index] * 0.62)));
        // Subtract the broad mean so detail sharpens the crest without
        // lifting every alpine sample or forming isolated positive needles.
        detail += (ridge - 0.42) * AMPLITUDES[index] * retained;
    }
    const double elevationGate = smoothstep(118.0, 220.0, context.terrainHeight);
    const double dryChannel =
        smoothstep(std::max(2.0, context.channelWidth * 1.15),
                   std::max(12.0, context.channelWidth * 2.8), context.channelDistance);
    return detail * tectonic.upliftGate * elevationGate * (0.34 + tectonic.ridgeStrength * 0.66) *
           dryChannel;
}

double
AlpineMorphologySampler::sampleRidgeDetail(const AlpineSurfaceContext& context,
                                           const AlpineTectonicSample& tectonic) const noexcept {
    if (context.ocean || context.lake || !std::isfinite(context.terrainHeight)) return 0.0;
    return filteredRidgeDetail(context, tectonic);
}

AlpineMorphologySample
AlpineMorphologySampler::sampleSurface(const AlpineSurfaceContext& context) const noexcept {
    return sampleSurface(context, sampleTectonic(context));
}

AlpineMorphologySample
AlpineMorphologySampler::sampleSurface(const AlpineSurfaceContext& context,
                                       const AlpineTectonicSample& tectonic) const noexcept {
    AlpineMorphologySample result;
    result.ridgeStrength = tectonic.ridgeStrength;
    result.hornStrength = tectonic.hornStrength;
    result.ridgeDetail = sampleRidgeDetail(context, tectonic);

    if (context.ocean || context.lake || !std::isfinite(context.terrainHeight)) {
        result.elevationOffset = 0.0;
        return result;
    }

    const double climate = glacierClimateGate(context.terrainHeight, context.temperatureC,
                                              context.annualPrecipitationMm);
    result.glacialInfluence = climate * smoothstep(0.18, 0.82, context.uplift) *
                              (0.48 + smoothstep(0.45, 1.20, context.rockResistance) * 0.52);
    result.periglacialInfluence = smoothstep(118.0, 220.0, context.terrainHeight) *
                                  bell(context.temperatureC, -1.0, 5.5) *
                                  smoothstep(160.0, 900.0, context.annualPrecipitationMm);

    const bool hasValley =
        std::isfinite(context.channelDistance) && context.channelDistance >= 0.0 &&
        (context.channelWidth > 0.0 || context.erosionDepth > 0.5 || context.discharge > 1.0);
    if (hasValley && result.glacialInfluence > 1.0e-5) {
        const double halfWidth = std::clamp(
            42.0 + std::sqrt(std::max(0.0, context.discharge)) * 0.30 + context.channelWidth * 1.15,
            48.0, 156.0);
        const double normalizedDistance = context.channelDistance / halfWidth;
        // The inner 40 percent is deliberately broad and nearly flat. The
        // outer transition steepens into the characteristic U-shaped wall.
        const double trough = 1.0 - smoothstep(0.40, 0.96, normalizedDistance);
        const double gradientDepth = smoothstep(0.002, 0.035, context.channelGradient) * 11.0;
        const double dischargeDepth =
            smoothstep(20.0, 8'000.0, std::max(0.0, context.discharge)) * 9.0;
        const double baseDepth = 9.0 + gradientDepth + dischargeDepth;
        result.valleyInfluence = result.glacialInfluence * trough;
        result.valleyCarve = std::clamp(baseDepth * result.valleyInfluence, 0.0, 30.0);

        const double talusBand = bell(normalizedDistance, 0.68, 0.24);
        result.talusInfluence =
            result.periglacialInfluence * talusBand * (0.42 + tectonic.ridgeStrength * 0.58);
        result.talusDeposit = std::min(6.5, result.talusInfluence * 5.2);
    }

    if (result.glacialInfluence > 0.015 && tectonic.ridgeStrength > 0.08) {
        const int64_t baseCellX = boundedFloor(context.x / CIRQUE_CELL_EDGE);
        const int64_t baseCellZ = boundedFloor(context.z / CIRQUE_CELL_EDGE);
        double strongestBowl = 0.0;
        for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                const CirqueCandidate candidate =
                    cirqueCandidate(baseCellX + offsetX, baseCellZ + offsetZ);
                const double dx = context.x - candidate.x;
                const double dz = context.z - candidate.z;
                // Orientation belongs to the global candidate. Query-local
                // flow can vary across the bowl and would shear its footprint.
                const RotatedPosition local = rotate(dx, dz, candidate.orientation);
                const double normalizedAlong = local.along / (candidate.radius * 1.12);
                const double normalizedAcross = local.across / (candidate.radius * 0.86);
                const double radial = std::hypot(normalizedAlong, normalizedAcross);
                if (radial >= 1.0) continue;
                const double bowl = (1.0 - smoothstep(0.18, 1.0, radial)) * candidate.strength;
                strongestBowl = std::max(strongestBowl, bowl);
            }
        }
        result.cirqueInfluence =
            strongestBowl * result.glacialInfluence * (0.35 + tectonic.ridgeStrength * 0.65);
        result.cirqueCarve = std::min(24.0, result.cirqueInfluence * 24.0);
    }

    // Horns survive where multiple warped ridges intersect. Glacial carving
    // around those intersections increases relative relief without applying a
    // global sharpening filter to unrelated terrain.
    const double hornAccent = tectonic.hornStrength * result.glacialInfluence * 9.0;
    result.elevationOffset = std::clamp(result.ridgeDetail + hornAccent + result.talusDeposit -
                                            result.valleyCarve - result.cirqueCarve,
                                        -42.0, 18.0);
    return result;
}

} // namespace worldgen
