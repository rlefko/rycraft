#pragma once

#include "world/learned_terrain.hpp"
#include "world/terrain_runtime.hpp"

#include <array>
#include <filesystem>
#include <memory>

namespace worldgen::runtime {

// Builds the production page backend around the single-flight typed model
// executor. The backend owns its deterministic window caches and keeps the
// executor alive for every WorldGenerationContext that references it.
// tensorWindowByteBudget of 0 keeps the production 384 MiB tensor-window cap.
// A smaller value is a test-only knob for exercising retention under pressure;
// it is not part of the generation identity.
[[nodiscard]] std::shared_ptr<learned::TerrainInferenceBackend>
makeInfiniteDiffusionTerrainBackend(uint64_t seed, std::filesystem::path modelPack,
                                    std::shared_ptr<TerrainModelExecutor> executor,
                                    size_t tensorWindowByteBudget = 0);

// Exposes the provider-independent scheduler vector used by compatibility
// tests and local model diagnostics.
[[nodiscard]] std::array<float, 21> infiniteDiffusionEdmSigmaSchedule() noexcept;

// Converts the six physical model outputs into the persisted RYTA channel
// units. BIO4 is already in centidegrees and BIO15 is reported as percent.
[[nodiscard]] learned::QuantizedTerrainSample quantizeInfiniteDiffusionSample(
    float elevationMeters, float meanTemperatureC, float temperatureVariabilityCentidegrees,
    float annualPrecipitationMillimeters, float precipitationCoefficientPercent,
    float lapseRateCPerMeter) noexcept;

inline constexpr size_t INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE = 4;
inline constexpr size_t INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS = 6;
inline constexpr size_t INFINITE_DIFFUSION_COARSE_CONDITIONING_VALUES =
    INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE * INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE *
    INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS;

// Interpolates one native pixel center from a channel-major 6 by 4 by 4
// physical coarse slice for conditioning and compatibility diagnostics. This
// helper never supplies drawable PREVIEW terrain. Slice index (0, 0) is the
// global coarse sample at (coarseRow - 1, coarseColumn - 1). The half-pixel
// registration matches the published Python and Java align_corners=false path
// at negative coordinates.
[[nodiscard]] std::array<float, INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS>
interpolateInfiniteDiffusionCoarseConditioning(
    std::span<const float, INFINITE_DIFFUSION_COARSE_CONDITIONING_VALUES> coarseFields,
    int64_t nativeRow, int64_t nativeColumn, int64_t coarseRow, int64_t coarseColumn) noexcept;

struct InfiniteDiffusionLaplacianResult {
    std::vector<float> lowFrequency;
    std::vector<float> decoded;
};

// Runs the exact postprocessing path for inspector and Python compatibility
// tests. Inputs and outputs are row-major float32 matrices.
[[nodiscard]] InfiniteDiffusionLaplacianResult
runInfiniteDiffusionLaplacian(std::span<const float> residual, int residualHeight,
                              int residualWidth, std::span<const float> lowFrequency,
                              int lowFrequencyHeight, int lowFrequencyWidth, float sigma);

} // namespace worldgen::runtime
