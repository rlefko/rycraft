#include "world/infinite_diffusion_backend.hpp"

#include "common/trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace worldgen::runtime {
namespace {

using learned::AuthorityQuality;
using learned::AuthorityRequestPriority;
using learned::AuthorityResult;
using learned::CoarseSpawnGrid;
using learned::CoarseSpawnRegion;
using learned::GenerationFailure;
using learned::GenerationFailureCode;
using learned::GenerationIdentity;
using learned::NativeRect;
using learned::TerrainAuthorityPage;
using learned::TerrainPageKey;
using learned::WindowIndex;

constexpr int COARSE_EDGE = 64;
constexpr int COARSE_STRIDE = 48;
constexpr int LATENT_EDGE = 64;
constexpr int LATENT_STRIDE = 32;
constexpr int DECODER_EDGE = 256;
constexpr int DECODER_STRIDE = 192;
constexpr int LATENT_COMPRESSION = 8;
constexpr int COARSE_CHANNELS = 7;
constexpr int LATENT_CHANNELS = 6;
constexpr int DECODER_CHANNELS = 2;
constexpr int MODEL_BATCH = 4;
static_assert(learned::LATENT_WINDOW.batchSize == MODEL_BATCH);
static_assert(learned::DECODER_WINDOW.batchSize == MODEL_BATCH);
constexpr float SIGMA_DATA = 0.5F;
constexpr float SIGMA_MIN = 0.002F;
constexpr float SIGMA_MAX = 80.0F;
constexpr float SIGMA_RHO = 7.0F;
constexpr float LOW_FREQUENCY_MEAN = -31.4F;
constexpr float LOW_FREQUENCY_STD = 38.6F;
constexpr float RESIDUAL_MEAN = 0.0F;
constexpr float RESIDUAL_STD = 0.7F;
constexpr size_t TENSOR_WINDOW_CACHE_BUDGET = 384ULL * 1'024 * 1'024;

constexpr std::array<float, 6> COARSE_MEANS{
    -37.7000079295F, 1.1403065256F,    18.1024865887F,
    332.8342598198F, 1332.2078969994F, 52.6600882069F,
};
constexpr std::array<float, 6> COARSE_STDS{
    39.7419997423F, 1.7681844105F, 8.9214691879F, 321.7660336396F, 842.9293648885F, 31.0799853187F,
};
constexpr std::array<float, 7> CONDITIONING_MEANS{
    14.99F, 11.65F, 15.87F, 619.26F, 833.12F, 69.40F, 0.66F,
};
constexpr std::array<float, 7> CONDITIONING_STDS{
    21.72F, 21.78F, 10.40F, 452.29F, 738.09F, 34.59F, 0.47F,
};

GenerationFailure inferenceFailure(std::string message, bool retriable = true) {
    return {.code = GenerationFailureCode::INFERENCE_FAILED,
            .message = std::move(message),
            .retriable = retriable};
}

int64_t ceilDivide(int64_t value, int64_t divisor) {
    return -learned::floorDivide(-value, divisor);
}

template <typename Integer>
Integer roundedClamp(double value) {
    if (!std::isfinite(value)) return Integer{};
    const double bounded =
        std::clamp(value, static_cast<double>(std::numeric_limits<Integer>::min()),
                   static_cast<double>(std::numeric_limits<Integer>::max()));
    return static_cast<Integer>(std::llround(bounded));
}

template <>
uint16_t roundedClamp<uint16_t>(double value) {
    if (!std::isfinite(value)) return 0;
    return static_cast<uint16_t>(std::llround(
        std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<uint16_t>::max()))));
}

struct Matrix {
    int height = 0;
    int width = 0;
    std::vector<float> values;

    Matrix() = default;
    Matrix(int requestedHeight, int requestedWidth)
        : height(requestedHeight)
        , width(requestedWidth)
        , values(static_cast<size_t>(requestedHeight) * requestedWidth) {}

    float& at(int row, int column) { return values[static_cast<size_t>(row) * width + column]; }
    float at(int row, int column) const {
        return values[static_cast<size_t>(row) * width + column];
    }
};

Matrix bilinearResize(const Matrix& source, int destinationHeight, int destinationWidth) {
    Matrix destination(destinationHeight, destinationWidth);
    for (int row = 0; row < destinationHeight; ++row) {
        const float sourceRow =
            (static_cast<float>(row) + 0.5F) * source.height / destinationHeight - 0.5F;
        int row0 = static_cast<int>(std::floor(sourceRow));
        int row1 = row0 + 1;
        const float rowWeight = sourceRow - static_cast<float>(row0);
        row0 = std::clamp(row0, 0, source.height - 1);
        row1 = std::clamp(row1, 0, source.height - 1);
        for (int column = 0; column < destinationWidth; ++column) {
            const float sourceColumn =
                (static_cast<float>(column) + 0.5F) * source.width / destinationWidth - 0.5F;
            int column0 = static_cast<int>(std::floor(sourceColumn));
            int column1 = column0 + 1;
            const float columnWeight = sourceColumn - static_cast<float>(column0);
            column0 = std::clamp(column0, 0, source.width - 1);
            column1 = std::clamp(column1, 0, source.width - 1);
            destination.at(row, column) =
                (1.0F - rowWeight) * (1.0F - columnWeight) * source.at(row0, column0) +
                (1.0F - rowWeight) * columnWeight * source.at(row0, column1) +
                rowWeight * (1.0F - columnWeight) * source.at(row1, column0) +
                rowWeight * columnWeight * source.at(row1, column1);
        }
    }
    return destination;
}

std::vector<std::vector<std::pair<int, float>>> antialiasedLinearWeights(int sourceSize,
                                                                         int destinationSize) {
    std::vector<std::vector<std::pair<int, float>>> weights(static_cast<size_t>(destinationSize));
    const double scale = static_cast<double>(sourceSize) / destinationSize;
    const double support = std::max(1.0, scale);
    for (int destination = 0; destination < destinationSize; ++destination) {
        const double center = (static_cast<double>(destination) + 0.5) * scale - 0.5;
        const int begin = static_cast<int>(std::ceil(center - support));
        const int end = static_cast<int>(std::floor(center + support));
        double total = 0.0;
        for (int source = begin; source <= end; ++source) {
            if (source < 0 || source >= sourceSize) continue;
            const double weight = std::max(0.0, 1.0 - std::abs(source - center) / support);
            if (weight == 0.0) continue;
            weights[static_cast<size_t>(destination)].push_back(
                {source, static_cast<float>(weight)});
            total += weight;
        }
        if (total > 0.0) {
            for (auto& [source, weight] : weights[static_cast<size_t>(destination)]) {
                static_cast<void>(source);
                weight = static_cast<float>(static_cast<double>(weight) / total);
            }
        }
    }
    return weights;
}

Matrix antialiasedBilinearResize(const Matrix& source, int destinationHeight,
                                 int destinationWidth) {
    if (destinationHeight >= source.height && destinationWidth >= source.width)
        return bilinearResize(source, destinationHeight, destinationWidth);
    const auto rowWeights = antialiasedLinearWeights(source.height, destinationHeight);
    const auto columnWeights = antialiasedLinearWeights(source.width, destinationWidth);
    Matrix destination(destinationHeight, destinationWidth);
    for (int row = 0; row < destinationHeight; ++row) {
        for (int column = 0; column < destinationWidth; ++column) {
            double value = 0.0;
            for (const auto& [sourceRow, rowWeight] : rowWeights[static_cast<size_t>(row)]) {
                for (const auto& [sourceColumn, columnWeight] :
                     columnWeights[static_cast<size_t>(column)]) {
                    value += static_cast<double>(source.at(sourceRow, sourceColumn)) * rowWeight *
                             columnWeight;
                }
            }
            destination.at(row, column) = static_cast<float>(value);
        }
    }
    return destination;
}

Matrix bilinearResizeExtrapolated(const Matrix& source, int destinationHeight,
                                  int destinationWidth) {
    Matrix padded(source.height + 2, source.width + 2);
    for (int row = 0; row < source.height; ++row)
        for (int column = 0; column < source.width; ++column)
            padded.at(row + 1, column + 1) = source.at(row, column);
    for (int column = 1; column <= source.width; ++column) {
        padded.at(0, column) = source.height > 1
                                   ? 2.0F * source.at(0, column - 1) - source.at(1, column - 1)
                                   : source.at(0, column - 1);
        padded.at(source.height + 1, column) =
            source.height > 1 ? 2.0F * source.at(source.height - 1, column - 1) -
                                    source.at(source.height - 2, column - 1)
                              : source.at(source.height - 1, column - 1);
    }
    for (int row = 0; row < source.height + 2; ++row) {
        padded.at(row, 0) =
            source.width > 1 ? 2.0F * padded.at(row, 1) - padded.at(row, 2) : padded.at(row, 1);
        padded.at(row, source.width + 1) = source.width > 1 ? 2.0F * padded.at(row, source.width) -
                                                                  padded.at(row, source.width - 1)
                                                            : padded.at(row, source.width);
    }
    const int enlargedHeight =
        static_cast<int>(std::lround(destinationHeight + 2.0 * destinationHeight / source.height));
    const int enlargedWidth =
        static_cast<int>(std::lround(destinationWidth + 2.0 * destinationWidth / source.width));
    const Matrix enlarged = bilinearResize(padded, enlargedHeight, enlargedWidth);
    const int padHeight =
        static_cast<int>(std::lround(static_cast<double>(destinationHeight) / source.height));
    const int padWidth =
        static_cast<int>(std::lround(static_cast<double>(destinationWidth) / source.width));
    Matrix destination(destinationHeight, destinationWidth);
    for (int row = 0; row < destinationHeight; ++row)
        for (int column = 0; column < destinationWidth; ++column)
            destination.at(row, column) = enlarged.at(row + padHeight, column + padWidth);
    return destination;
}

Matrix gaussianBlur(const Matrix& source, float sigma) {
    const int kernelSize = (static_cast<int>(sigma * 2.0F) / 2) * 2 + 1;
    const int radius = kernelSize / 2;
    std::vector<float> kernel(static_cast<size_t>(kernelSize));
    float total = 0.0F;
    for (int index = 0; index < kernelSize; ++index) {
        const float distance = static_cast<float>(index - radius);
        kernel[static_cast<size_t>(index)] =
            std::exp(-0.5F * distance * distance / (sigma * sigma));
        total += kernel[static_cast<size_t>(index)];
    }
    for (float& value : kernel)
        value /= total;
    auto reflected = [](int coordinate, int size) {
        if (size <= 1) return 0;
        while (coordinate < 0 || coordinate >= size) {
            if (coordinate < 0)
                coordinate = -coordinate;
            else
                coordinate = 2 * size - 2 - coordinate;
        }
        return coordinate;
    };
    Matrix horizontal(source.height, source.width);
    for (int row = 0; row < source.height; ++row) {
        for (int column = 0; column < source.width; ++column) {
            float sum = 0.0F;
            for (int offset = 0; offset < kernelSize; ++offset) {
                const int sampledColumn = reflected(column + offset - radius, source.width);
                sum += source.at(row, sampledColumn) * kernel[static_cast<size_t>(offset)];
            }
            horizontal.at(row, column) = sum;
        }
    }
    Matrix destination(source.height, source.width);
    for (int row = 0; row < source.height; ++row) {
        for (int column = 0; column < source.width; ++column) {
            float sum = 0.0F;
            for (int offset = 0; offset < kernelSize; ++offset) {
                const int sampledRow = reflected(row + offset - radius, source.height);
                sum += horizontal.at(sampledRow, column) * kernel[static_cast<size_t>(offset)];
            }
            destination.at(row, column) = sum;
        }
    }
    return destination;
}

Matrix laplacianDenoise(const Matrix& residual, const Matrix& lowFrequency, float sigma) {
    const Matrix expanded =
        bilinearResizeExtrapolated(lowFrequency, residual.height, residual.width);
    Matrix decoded(residual.height, residual.width);
    for (size_t index = 0; index < decoded.values.size(); ++index)
        decoded.values[index] = residual.values[index] + expanded.values[index];
    return gaussianBlur(antialiasedBilinearResize(decoded, lowFrequency.height, lowFrequency.width),
                        sigma);
}

Matrix laplacianDecode(const Matrix& residual, const Matrix& lowFrequency) {
    const Matrix expanded = bilinearResize(lowFrequency, residual.height, residual.width);
    Matrix decoded(residual.height, residual.width);
    for (size_t index = 0; index < decoded.values.size(); ++index)
        decoded.values[index] = residual.values[index] + expanded.values[index];
    return decoded;
}

float bilinearSample(const Matrix& source, float row, float column) {
    const float boundedRow = std::clamp(row, 0.0F, static_cast<float>(source.height - 1));
    const float boundedColumn = std::clamp(column, 0.0F, static_cast<float>(source.width - 1));
    const int row0 = static_cast<int>(boundedRow);
    const int row1 = std::min(source.height - 1, row0 + 1);
    const int column0 = static_cast<int>(boundedColumn);
    const int column1 = std::min(source.width - 1, column0 + 1);
    const float rowWeight = boundedRow - row0;
    const float columnWeight = boundedColumn - column0;
    return (1.0F - rowWeight) * (1.0F - columnWeight) * source.at(row0, column0) +
           (1.0F - rowWeight) * columnWeight * source.at(row0, column1) +
           rowWeight * (1.0F - columnWeight) * source.at(row1, column0) +
           rowWeight * columnWeight * source.at(row1, column1);
}

struct PipelineData {
    std::array<std::array<float, 64>, 5> noiseQuantiles{};
    std::array<std::array<float, 64>, 5> dataQuantiles{};
    float aTemperatureStd = 0.0F;
    float bTemperatureStd = 0.0F;
    float temperatureStdP1 = 0.0F;
    float temperatureStdP99 = 0.0F;
};

std::vector<double> parseJsonNumberArray(const std::string& json, std::string_view key) {
    const size_t keyPosition = json.find('"' + std::string(key) + '"');
    if (keyPosition == std::string::npos) return {};
    const size_t start = json.find('[', keyPosition);
    if (start == std::string::npos) return {};
    std::vector<double> values;
    int depth = 0;
    const char* cursor = json.c_str() + start;
    while (*cursor != '\0') {
        if (*cursor == '[') {
            ++depth;
            ++cursor;
            continue;
        }
        if (*cursor == ']') {
            --depth;
            ++cursor;
            if (depth == 0) break;
            continue;
        }
        if ((*cursor >= '0' && *cursor <= '9') || *cursor == '-' || *cursor == '+') {
            char* end = nullptr;
            const double value = std::strtod(cursor, &end);
            if (end != cursor) {
                values.push_back(value);
                cursor = end;
                continue;
            }
        }
        ++cursor;
    }
    return values;
}

std::optional<double> parseJsonNumber(const std::string& json, std::string_view key) {
    const size_t keyPosition = json.find('"' + std::string(key) + '"');
    if (keyPosition == std::string::npos) return std::nullopt;
    const size_t colon = json.find(':', keyPosition);
    if (colon == std::string::npos) return std::nullopt;
    const char* cursor = json.c_str() + colon + 1;
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n')
        ++cursor;
    char* end = nullptr;
    const double value = std::strtod(cursor, &end);
    if (end == cursor) return std::nullopt;
    return value;
}

std::optional<std::string> parseJsonString(const std::string& json, std::string_view key) {
    const size_t keyPosition = json.find('"' + std::string(key) + '"');
    if (keyPosition == std::string::npos) return std::nullopt;
    const size_t colon = json.find(':', keyPosition);
    if (colon == std::string::npos) return std::nullopt;
    const size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return std::nullopt;
    const size_t end = json.find('"', quote + 1);
    if (end == std::string::npos) return std::nullopt;
    return json.substr(quote + 1, end - quote - 1);
}

bool jsonFieldIsNull(const std::string& json, std::string_view key) {
    const size_t keyPosition = json.find('"' + std::string(key) + '"');
    if (keyPosition == std::string::npos) return false;
    const size_t colon = json.find(':', keyPosition);
    if (colon == std::string::npos) return false;
    const size_t value = json.find_first_not_of(" \t\r\n", colon + 1);
    return value != std::string::npos && json.compare(value, 4, "null") == 0;
}

bool matchesValues(std::span<const double> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - static_cast<double>(expected[index])) > 1.0E-4) {
            return false;
        }
    }
    return true;
}

bool validateModelConfiguration(const std::filesystem::path& path, std::string& error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "Could not open world_pipeline_config.json";
        return false;
    }
    const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::vector<double> means = parseJsonNumberArray(json, "coarse_means");
    const std::vector<double> standardDeviations = parseJsonNumberArray(json, "coarse_stds");
    const std::vector<double> conditioning = parseJsonNumberArray(json, "cond_snr");
    const std::vector<double> frequencies = parseJsonNumberArray(json, "frequency_mult");
    static constexpr std::array<float, 5> SUPPORTED_CONDITIONING{
        0.5F, 0.5F, 0.5F, 0.5F, 0.5F,
    };
    static constexpr std::array<float, 5> SUPPORTED_FREQUENCIES{
        1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
    };
    const auto className = parseJsonString(json, "_class_name");
    const auto elevationPool = parseJsonString(json, "elev_coarse_pool_mode");
    const auto percentilePool = parseJsonString(json, "p5_coarse_pool_mode");
    const auto coarsePooling = parseJsonNumber(json, "coarse_pooling");
    const auto dropWater = parseJsonNumber(json, "drop_water_pct");
    const auto latentCompression = parseJsonNumber(json, "latent_compression");
    const auto nativeResolution = parseJsonNumber(json, "native_resolution");
    const auto residualMean = parseJsonNumber(json, "residual_mean");
    const auto residualStd = parseJsonNumber(json, "residual_std");
    const bool supported =
        className == "WorldPipeline" && elevationPool == "avg" && percentilePool == "avg" &&
        coarsePooling == 1.0 && dropWater == 0.5 &&
        latentCompression == static_cast<double>(LATENT_COMPRESSION) && nativeResolution == 30.0 &&
        residualMean == static_cast<double>(RESIDUAL_MEAN) && residualStd &&
        std::abs(*residualStd - static_cast<double>(RESIDUAL_STD)) < 1.0E-6 &&
        matchesValues(means, COARSE_MEANS) && matchesValues(standardDeviations, COARSE_STDS) &&
        matchesValues(conditioning, SUPPORTED_CONDITIONING) &&
        matchesValues(frequencies, SUPPORTED_FREQUENCIES) && jsonFieldIsNull(json, "histogram_raw");
    if (!supported) error = "world_pipeline_config.json uses unsupported model pipeline settings";
    return supported;
}

std::optional<PipelineData> loadPipelineData(const std::filesystem::path& path,
                                             std::string& error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "Could not open pipeline_data.json";
        return std::nullopt;
    }
    const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::vector<double> noise = parseJsonNumberArray(json, "noise_quantile_tables");
    const std::vector<double> data = parseJsonNumberArray(json, "data_quantile_tables");
    if (noise.size() != 5 * 64 || data.size() != 5 * 64) {
        error = "pipeline_data.json has incompatible quantile tables";
        return std::nullopt;
    }
    const auto a = parseJsonNumber(json, "a_temp_std");
    const auto b = parseJsonNumber(json, "b_temp_std");
    const auto p1 = parseJsonNumber(json, "temp_std_p1");
    const auto p99 = parseJsonNumber(json, "temp_std_p99");
    if (!a || !b || !p1 || !p99) {
        error = "pipeline_data.json is missing temperature statistics";
        return std::nullopt;
    }
    PipelineData result;
    for (size_t channel = 0; channel < 5; ++channel) {
        for (size_t index = 0; index < 64; ++index) {
            result.noiseQuantiles[channel][index] = static_cast<float>(noise[channel * 64 + index]);
            result.dataQuantiles[channel][index] = static_cast<float>(data[channel * 64 + index]);
        }
    }
    result.aTemperatureStd = static_cast<float>(*a);
    result.bTemperatureStd = static_cast<float>(*b);
    result.temperatureStdP1 = static_cast<float>(*p1);
    result.temperatureStdP99 = static_cast<float>(*p99);
    return result;
}

float interpolateQuantiles(float value, std::span<const float> source,
                           std::span<const float> destination) {
    if (value <= source.front()) return destination.front();
    if (value >= source.back()) return destination.back();
    const auto upper = std::upper_bound(source.begin(), source.end(), value);
    const size_t high = static_cast<size_t>(upper - source.begin());
    const size_t low = high - 1;
    const float fraction = (value - source[low]) / (source[high] - source[low]);
    return destination[low] + fraction * (destination[high] - destination[low]);
}

class FastPerlin {
public:
    FastPerlin(int32_t seed, int octaves) : seed_(seed), octaves_(octaves) {
        float amplitude = 0.5F;
        float total = 1.0F;
        for (int octave = 1; octave < octaves_; ++octave) {
            total += amplitude;
            amplitude *= 0.5F;
        }
        bounding_ = 1.0F / total;
    }

    float sample(float x, float y) const {
        x *= 0.05F;
        y *= 0.05F;
        int32_t seed = seed_;
        float amplitude = bounding_;
        float result = 0.0F;
        for (int octave = 0; octave < octaves_; ++octave) {
            result += single(seed++, x, y) * amplitude;
            x *= 2.0F;
            y *= 2.0F;
            amplitude *= 0.5F;
        }
        return result;
    }

    std::array<float, 64> noiseQuantiles() const {
        constexpr int SAMPLE_EDGE = 1'024;
        constexpr float EPSILON = 1.0E-4F;
        std::vector<float> samples(static_cast<size_t>(SAMPLE_EDGE) * SAMPLE_EDGE);
        for (int row = 0; row < SAMPLE_EDGE; ++row) {
            for (int column = 0; column < SAMPLE_EDGE; ++column) {
                samples[static_cast<size_t>(row) * SAMPLE_EDGE + column] =
                    sample(static_cast<float>(column * 32), static_cast<float>(row * 32));
            }
        }
        std::sort(samples.begin(), samples.end());

        std::array<float, 64> quantiles{};
        for (size_t index = 0; index < quantiles.size(); ++index) {
            const float percentile = EPSILON + static_cast<float>(index) * (1.0F - 2.0F * EPSILON) /
                                                   static_cast<float>(quantiles.size() - 1);
            const float position = percentile * static_cast<float>(samples.size() - 1);
            const size_t low = static_cast<size_t>(position);
            const size_t high = std::min(low + 1, samples.size() - 1);
            quantiles[index] = samples[low] + (position - static_cast<float>(low)) *
                                                  (samples[high] - samples[low]);
        }
        float minimumDifference = std::numeric_limits<float>::max();
        for (size_t index = 1; index < quantiles.size(); ++index) {
            if (quantiles[index] > quantiles[index - 1]) {
                minimumDifference =
                    std::min(minimumDifference, quantiles[index] - quantiles[index - 1]);
            }
        }
        if (minimumDifference == std::numeric_limits<float>::max()) minimumDifference = 1.0E-10F;
        for (size_t index = 1; index < quantiles.size(); ++index) {
            if (quantiles[index] <= quantiles[index - 1])
                quantiles[index] = quantiles[index - 1] + minimumDifference * 0.1F;
        }
        return quantiles;
    }

private:
    static int floorFast(float value) {
        return value >= 0.0F ? static_cast<int>(value) : static_cast<int>(value) - 1;
    }
    static float quintic(float value) {
        return value * value * value * (value * (value * 6.0F - 15.0F) + 10.0F);
    }
    static float lerp(float first, float second, float weight) {
        return first + weight * (second - first);
    }
    static uint32_t hash(int32_t seed, int32_t x, int32_t y) {
        uint32_t value =
            static_cast<uint32_t>(seed) ^ static_cast<uint32_t>(x) ^ static_cast<uint32_t>(y);
        return value * 0x27D4EB2DU;
    }
    static std::pair<float, float> gradient(uint32_t hashed) {
        hashed ^= static_cast<uint32_t>(static_cast<int32_t>(hashed) >> 15);
        const unsigned pair = (hashed & 254U) / 2U;
        static constexpr std::array<unsigned, 8> SPECIAL{1, 4, 7, 10, 13, 16, 19, 22};
        const unsigned direction = pair < 120 ? pair % 24 : SPECIAL[pair - 120];
        static constexpr std::array<std::pair<float, float>, 24> GRADIENTS{{
            {0.130526192220052F, 0.99144486137381F},    {0.38268343236509F, 0.923879532511287F},
            {0.608761429008721F, 0.793353340291235F},   {0.793353340291235F, 0.608761429008721F},
            {0.923879532511287F, 0.38268343236509F},    {0.99144486137381F, 0.130526192220051F},
            {0.99144486137381F, -0.130526192220051F},   {0.923879532511287F, -0.38268343236509F},
            {0.793353340291235F, -0.60876142900872F},   {0.608761429008721F, -0.793353340291235F},
            {0.38268343236509F, -0.923879532511287F},   {0.130526192220052F, -0.99144486137381F},
            {-0.130526192220052F, -0.99144486137381F},  {-0.38268343236509F, -0.923879532511287F},
            {-0.608761429008721F, -0.793353340291235F}, {-0.793353340291235F, -0.608761429008721F},
            {-0.923879532511287F, -0.38268343236509F},  {-0.99144486137381F, -0.130526192220052F},
            {-0.99144486137381F, 0.130526192220051F},   {-0.923879532511287F, 0.38268343236509F},
            {-0.793353340291235F, 0.608761429008721F},  {-0.608761429008721F, 0.793353340291235F},
            {-0.38268343236509F, 0.923879532511287F},   {-0.130526192220052F, 0.99144486137381F},
        }};
        return GRADIENTS[direction];
    }
    static float dotGradient(int32_t seed, int32_t x, int32_t y, float dx, float dy) {
        const auto [gradientX, gradientY] = gradient(hash(seed, x, y));
        return dx * gradientX + dy * gradientY;
    }
    static float single(int32_t seed, float x, float y) {
        constexpr int32_t PRIME_X = 501'125'321;
        constexpr int32_t PRIME_Y = 1'136'930'381;
        const int xCoordinate = floorFast(x);
        const int yCoordinate = floorFast(y);
        const float dx0 = x - static_cast<float>(xCoordinate);
        const float dy0 = y - static_cast<float>(yCoordinate);
        const float dx1 = dx0 - 1.0F;
        const float dy1 = dy0 - 1.0F;
        const float xWeight = quintic(dx0);
        const float yWeight = quintic(dy0);
        const int32_t x0 = static_cast<int32_t>(static_cast<uint32_t>(xCoordinate) * PRIME_X);
        const int32_t y0 = static_cast<int32_t>(static_cast<uint32_t>(yCoordinate) * PRIME_Y);
        const int32_t x1 = static_cast<int32_t>(static_cast<uint32_t>(x0) + PRIME_X);
        const int32_t y1 = static_cast<int32_t>(static_cast<uint32_t>(y0) + PRIME_Y);
        const float first =
            lerp(dotGradient(seed, x0, y0, dx0, dy0), dotGradient(seed, x1, y0, dx1, dy0), xWeight);
        const float second =
            lerp(dotGradient(seed, x0, y1, dx0, dy1), dotGradient(seed, x1, y1, dx1, dy1), xWeight);
        return lerp(first, second, yWeight) * 1.4247691104677813F;
    }

    int32_t seed_;
    int octaves_;
    float bounding_ = 1.0F;
};

struct WindowKey {
    enum class Stage : uint8_t {
        Coarse,
        LatentInitial,
        LatentFinal,
        Decoder,
    };
    Stage stage = Stage::Coarse;
    int64_t row = 0;
    int64_t column = 0;
    auto operator<=>(const WindowKey&) const = default;
};

struct CachedWindow {
    std::vector<float> values;
    uint64_t lastAccess = 0;
};

class EdmScheduler {
public:
    EdmScheduler() {
        const float minimum = static_cast<float>(
            std::pow(static_cast<double>(SIGMA_MIN), 1.0 / static_cast<double>(SIGMA_RHO)));
        const float maximum = static_cast<float>(
            std::pow(static_cast<double>(SIGMA_MAX), 1.0 / static_cast<double>(SIGMA_RHO)));
        for (size_t index = 0; index < 20; ++index) {
            const float fraction = static_cast<float>(index) / 19.0F;
            const float inverseRho = maximum + fraction * (minimum - maximum);
            sigmas_[index] = static_cast<float>(
                std::pow(static_cast<double>(inverseRho), static_cast<double>(SIGMA_RHO)));
        }
        sigmas_[20] = 0.0F;
    }

    float sigma(size_t index) const { return sigmas_[index]; }
    const std::array<float, 21>& sigmas() const noexcept { return sigmas_; }

    std::vector<float> step(std::span<const float> model, std::span<const float> sample) {
        const float sigmaSource = sigmas_[step_];
        const float sigmaTarget = sigmas_[step_ + 1];
        const float sigmaDataSquared = SIGMA_DATA * SIGMA_DATA;
        const float sigmaSquared = sigmaSource * sigmaSource;
        const float skip = sigmaDataSquared / (sigmaSquared + sigmaDataSquared);
        const float output = sigmaSource * SIGMA_DATA / std::sqrt(sigmaSquared + sigmaDataSquared);
        std::vector<float> predicted(sample.size());
        for (size_t index = 0; index < sample.size(); ++index)
            predicted[index] = skip * sample[index] + output * model[index];

        std::vector<float> result(sample.size());
        if (step_ == 0 || step_ == 19) {
            const float ratio = sigmaTarget / sigmaSource;
            for (size_t index = 0; index < sample.size(); ++index)
                result[index] = ratio * sample[index] - (ratio - 1.0F) * predicted[index];
        } else {
            const double lambdaTarget = -std::log(static_cast<double>(sigmaTarget));
            const double lambdaSource = -std::log(static_cast<double>(sigmaSource));
            const double lambdaPrevious = -std::log(static_cast<double>(sigmas_[step_ - 1]));
            const float ratio =
                static_cast<float>((lambdaSource - lambdaPrevious) / (lambdaTarget - lambdaSource));
            const float exponential = sigmaTarget / sigmaSource;
            for (size_t index = 0; index < sample.size(); ++index) {
                const float derivative = (predicted[index] - previous_[index]) / ratio;
                result[index] = exponential * sample[index] -
                                (exponential - 1.0F) * predicted[index] -
                                0.5F * (exponential - 1.0F) * derivative;
            }
        }
        previous_ = std::move(predicted);
        ++step_;
        return result;
    }

private:
    std::array<float, 21> sigmas_{};
    size_t step_ = 0;
    std::vector<float> previous_;
};

class InfiniteDiffusionBackend final : public learned::TerrainInferenceBackend {
public:
    InfiniteDiffusionBackend(uint64_t requestedSeed, std::filesystem::path modelPack,
                             std::shared_ptr<TerrainModelExecutor> requestedExecutor)
        : seed_(requestedSeed)
        , modelPack_(std::move(modelPack))
        , executor_(std::move(requestedExecutor)) {
        std::string error;
        pipelineData_ = loadPipelineData(modelPack_ / "pipeline_data.json", error);
        if (!pipelineData_) initializationError_ = std::move(error);
        if (initializationError_.empty() &&
            !validateModelConfiguration(modelPack_ / "world_pipeline_config.json", error)) {
            initializationError_ = std::move(error);
        }
        for (size_t channel = 0; channel < syntheticNoise_.size(); ++channel) {
            const int32_t channelSeed =
                static_cast<int32_t>((seed_ + channel + 1) & 0x7FFF'FFFFULL);
            const int octaves = channel == 1 ? 2 : 4;
            syntheticNoise_[channel] = std::make_unique<FastPerlin>(channelSeed, octaves);
            if (pipelineData_)
                pipelineData_->noiseQuantiles[channel] = syntheticNoise_[channel]->noiseQuantiles();
        }
    }

    AuthorityResult<TerrainAuthorityPage> inferPage(const GenerationIdentity& identity,
                                                    TerrainPageKey key) override {
        return inferPageWithPhase(identity, key, TerrainRuntimeInferencePhase::Other);
    }

    AuthorityResult<TerrainAuthorityPage>
    inferPageForRequest(const GenerationIdentity& identity, TerrainPageKey key,
                        AuthorityRequestPriority priority) override {
        return inferPageWithPhase(identity, key, phaseForPage(key, priority));
    }

    AuthorityResult<TerrainAuthorityPage> inferPageWithPhase(const GenerationIdentity& identity,
                                                             TerrainPageKey key,
                                                             TerrainRuntimeInferencePhase phase) {
        std::lock_guard lock(mutex_);
        InferencePhaseScope phaseScope(activePhase_, phase);
        if (identity.seed != seed_ || !executor_) {
            return AuthorityResult<TerrainAuthorityPage>::failed(
                inferenceFailure("InfiniteDiffusion runtime identity mismatch", false));
        }
        if (!initializationError_.empty()) {
            return AuthorityResult<TerrainAuthorityPage>::failed(
                inferenceFailure(initializationError_, false));
        }
        try {
            TerrainAuthorityPage page = key.quality == AuthorityQuality::PREVIEW
                                            ? inferPreview(identity, key)
                                            : inferFinal(identity, key);
            return AuthorityResult<TerrainAuthorityPage>::ready(std::move(page));
        } catch (const std::exception& exception) {
            return AuthorityResult<TerrainAuthorityPage>::failed(inferenceFailure(
                std::string("InfiniteDiffusion page inference failed: ") + exception.what()));
        }
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPages(const GenerationIdentity& identity,
               std::span<const TerrainPageKey> requestedKeys) override {
        return inferPagesWithPhase(identity, requestedKeys, TerrainRuntimeInferencePhase::Other);
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPagesForRequest(const GenerationIdentity& identity,
                         std::span<const TerrainPageKey> requestedKeys,
                         AuthorityRequestPriority priority) override {
        const TerrainRuntimeInferencePhase phase =
            requestedKeys.empty() ? TerrainRuntimeInferencePhase::Other
                                  : phaseForPage(requestedKeys.front(), priority);
        return inferPagesWithPhase(identity, requestedKeys, phase);
    }

    AuthorityResult<std::vector<TerrainAuthorityPage>>
    inferPagesWithPhase(const GenerationIdentity& identity,
                        std::span<const TerrainPageKey> requestedKeys,
                        TerrainRuntimeInferencePhase phase) {
        std::lock_guard lock(mutex_);
        InferencePhaseScope phaseScope(activePhase_, phase);
        if (identity.seed != seed_ || !executor_) {
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                inferenceFailure("InfiniteDiffusion runtime identity mismatch", false));
        }
        if (requestedKeys.empty() ||
            requestedKeys.size() > learned::MAXIMUM_FINAL_AUTHORITY_BATCH_PAGES) {
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                inferenceFailure("InfiniteDiffusion page batch has an invalid size", false));
        }
        if (!initializationError_.empty()) {
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(
                inferenceFailure(initializationError_, false));
        }

        std::vector<TerrainPageKey> canonical(requestedKeys.begin(), requestedKeys.end());
        std::sort(canonical.begin(), canonical.end());
        if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end()) {
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(inferenceFailure(
                "InfiniteDiffusion page batch must contain unique page keys", false));
        }

        try {
            // PREVIEW and FINAL may share one coordinator admission. Union
            // preparation keeps the static four-window Base shape full across
            // page boundaries, while each immutable page remains reconstructed
            // and quantized independently in canonical order.
            std::vector<TerrainPageKey> previewKeys;
            std::vector<TerrainPageKey> finalKeys;
            previewKeys.reserve(canonical.size());
            finalKeys.reserve(canonical.size());
            for (const TerrainPageKey key : canonical) {
                (key.quality == AuthorityQuality::PREVIEW ? previewKeys : finalKeys).push_back(key);
            }
            std::map<TerrainPageKey, TerrainAuthorityPage> pagesByKey;
            const auto reconstruct = [&](std::span<const TerrainPageKey> keys,
                                         AuthorityQuality quality) {
                if (keys.empty()) return;
                if (quality == AuthorityQuality::PREVIEW)
                    prewarmPreviewPageGroup(keys);
                else
                    prewarmFinalPageGroup(keys);
                for (const TerrainPageKey key : keys) {
                    TerrainAuthorityPage page = quality == AuthorityQuality::PREVIEW
                                                    ? inferPreview(identity, key)
                                                    : inferFinal(identity, key);
                    auto [unused, inserted] = pagesByKey.emplace(key, std::move(page));
                    if (!inserted)
                        throw std::runtime_error("InfiniteDiffusion page batch repeated a key");
                    static_cast<void>(unused);
                }
            };
            reconstruct(previewKeys, AuthorityQuality::PREVIEW);
            reconstruct(finalKeys, AuthorityQuality::FINAL);
            std::vector<TerrainAuthorityPage> result;
            result.reserve(requestedKeys.size());
            for (const TerrainPageKey key : requestedKeys) {
                auto found = pagesByKey.find(key);
                if (found == pagesByKey.end()) {
                    throw std::runtime_error("InfiniteDiffusion page batch lost a page");
                }
                result.push_back(std::move(found->second));
            }
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::ready(std::move(result));
        } catch (const std::exception& exception) {
            return AuthorityResult<std::vector<TerrainAuthorityPage>>::failed(inferenceFailure(
                std::string("InfiniteDiffusion page batch inference failed: ") + exception.what()));
        }
    }

    AuthorityResult<learned::PhysicalTerrainGrid>
    inferFinalNativeGrid(const GenerationIdentity& identity, NativeRect region) override {
        return inferFinalNativeGridWithPhase(identity, region, TerrainRuntimeInferencePhase::Other);
    }

    AuthorityResult<learned::PhysicalTerrainGrid>
    inferFinalNativeGridForRequest(const GenerationIdentity& identity, NativeRect region,
                                   AuthorityRequestPriority priority) override {
        return inferFinalNativeGridWithPhase(identity, region, phaseForFinal(priority));
    }

    AuthorityResult<learned::PhysicalTerrainGrid>
    inferFinalNativeGridWithPhase(const GenerationIdentity& identity, NativeRect region,
                                  TerrainRuntimeInferencePhase phase) {
        std::lock_guard lock(mutex_);
        InferencePhaseScope phaseScope(activePhase_, phase);
        if (identity.seed != seed_ || !executor_) {
            return AuthorityResult<learned::PhysicalTerrainGrid>::failed(
                inferenceFailure("InfiniteDiffusion runtime identity mismatch", false));
        }
        // Reconstruction expands by one latent cell, adds the six-cell
        // cleanup stencil, then aligns outward to the latent lattice.
        // Reject coordinates whose exact expansion would overflow int64_t.
        constexpr int64_t RECONSTRUCTION_COORDINATE_MARGIN = LATENT_COMPRESSION * 8 - 1;
        if (!region.valid() ||
            region.rowBegin <
                std::numeric_limits<int64_t>::min() + RECONSTRUCTION_COORDINATE_MARGIN ||
            region.columnBegin <
                std::numeric_limits<int64_t>::min() + RECONSTRUCTION_COORDINATE_MARGIN ||
            region.rowEnd >
                std::numeric_limits<int64_t>::max() - RECONSTRUCTION_COORDINATE_MARGIN ||
            region.columnEnd >
                std::numeric_limits<int64_t>::max() - RECONSTRUCTION_COORDINATE_MARGIN ||
            region.height() > learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES ||
            region.width() > learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES ||
            (region.height() != 0 &&
             region.width() > learned::MAXIMUM_AUTHORITY_QUERY_SAMPLES / region.height())) {
            return AuthorityResult<learned::PhysicalTerrainGrid>::failed(
                inferenceFailure("InfiniteDiffusion transient final rectangle is invalid", false));
        }
        if (!initializationError_.empty()) {
            return AuthorityResult<learned::PhysicalTerrainGrid>::failed(
                inferenceFailure(initializationError_, false));
        }
        try {
            return AuthorityResult<learned::PhysicalTerrainGrid>::ready(inferFinalGrid(region));
        } catch (const std::exception& exception) {
            return AuthorityResult<learned::PhysicalTerrainGrid>::failed(inferenceFailure(
                std::string("InfiniteDiffusion transient final inference failed: ") +
                exception.what()));
        }
    }

    AuthorityResult<CoarseSpawnGrid> inferCoarseSpawnGrid(const GenerationIdentity& identity,
                                                          CoarseSpawnRegion region) override {
        return inferCoarseSpawnGridWithPhase(identity, region, TerrainRuntimeInferencePhase::Other);
    }

    AuthorityResult<CoarseSpawnGrid>
    inferCoarseSpawnGridForRequest(const GenerationIdentity& identity, CoarseSpawnRegion region,
                                   AuthorityRequestPriority) override {
        return inferCoarseSpawnGridWithPhase(identity, region,
                                             TerrainRuntimeInferencePhase::DrySpawnCoarseSearch);
    }

    AuthorityResult<CoarseSpawnGrid>
    inferCoarseSpawnGridWithPhase(const GenerationIdentity& identity, CoarseSpawnRegion region,
                                  TerrainRuntimeInferencePhase phase) {
        std::lock_guard lock(mutex_);
        InferencePhaseScope phaseScope(activePhase_, phase);
        if (identity.seed != seed_ || !executor_) {
            return AuthorityResult<CoarseSpawnGrid>::failed(
                inferenceFailure("InfiniteDiffusion runtime identity mismatch", false));
        }
        if (!region.valid() || region.height() > learned::MAXIMUM_COARSE_SPAWN_GRID_EDGE ||
            region.width() > learned::MAXIMUM_COARSE_SPAWN_GRID_EDGE) {
            return AuthorityResult<CoarseSpawnGrid>::failed(
                inferenceFailure("InfiniteDiffusion coarse spawn region is invalid", false));
        }
        if (!initializationError_.empty()) {
            return AuthorityResult<CoarseSpawnGrid>::failed(
                inferenceFailure(initializationError_, false));
        }
        try {
            const NativeRect coarseRegion{
                .rowBegin = region.rowBegin,
                .columnBegin = region.columnBegin,
                .rowEnd = region.rowEnd,
                .columnEnd = region.columnEnd,
            };
            const std::vector<float> weighted = slice(WindowKey::Stage::Coarse, coarseRegion,
                                                      COARSE_CHANNELS, COARSE_EDGE, COARSE_STRIDE);
            const size_t area =
                static_cast<size_t>(region.height()) * static_cast<size_t>(region.width());
            if (weighted.size() != static_cast<size_t>(COARSE_CHANNELS) * area) {
                return AuthorityResult<CoarseSpawnGrid>::failed(inferenceFailure(
                    "InfiniteDiffusion coarse spawn output has an incompatible shape"));
            }
            CoarseSpawnGrid result{
                .region = region,
                .elevationMeters = std::vector<float>(area),
            };
            for (size_t pixel = 0; pixel < area; ++pixel) {
                const float weight = weighted[6 * area + pixel];
                if (!std::isfinite(weight) || weight <= 1.0E-6F) {
                    return AuthorityResult<CoarseSpawnGrid>::failed(inferenceFailure(
                        "InfiniteDiffusion coarse spawn output does not cover its region"));
                }
                const float encodedElevation = weighted[pixel] / weight;
                if (!std::isfinite(encodedElevation)) {
                    return AuthorityResult<CoarseSpawnGrid>::failed(
                        inferenceFailure("InfiniteDiffusion coarse spawn elevation is not finite"));
                }
                result.elevationMeters[pixel] =
                    std::copysign(encodedElevation * encodedElevation, encodedElevation);
            }
            return AuthorityResult<CoarseSpawnGrid>::ready(std::move(result));
        } catch (const std::exception& exception) {
            return AuthorityResult<CoarseSpawnGrid>::failed(
                inferenceFailure(std::string("InfiniteDiffusion coarse spawn inference failed: ") +
                                 exception.what()));
        }
    }

private:
    class InferencePhaseScope {
    public:
        InferencePhaseScope(TerrainRuntimeInferencePhase& destination,
                            TerrainRuntimeInferencePhase phase)
            : destination_(destination)
            , previous_(destination) {
            destination_ = phase;
        }
        ~InferencePhaseScope() { destination_ = previous_; }

    private:
        TerrainRuntimeInferencePhase& destination_;
        TerrainRuntimeInferencePhase previous_;
    };

    static TerrainRuntimeInferencePhase phaseForFinal(AuthorityRequestPriority priority) noexcept {
        switch (priority) {
            case AuthorityRequestPriority::SPAWN:
                return TerrainRuntimeInferencePhase::FinalSpawnCertification;
            case AuthorityRequestPriority::EXPLORATION_EXACT:
                return TerrainRuntimeInferencePhase::ExplorationExact;
            case AuthorityRequestPriority::PROTECTED_HANDOFF:
                return TerrainRuntimeInferencePhase::ProtectedFinal;
            case AuthorityRequestPriority::VISIBLE_FINAL_REFINEMENT:
                return TerrainRuntimeInferencePhase::VisibleFinalRefinement;
            case AuthorityRequestPriority::COARSE_PREVIEW:
            case AuthorityRequestPriority::SPECULATIVE_PREFETCH:
                return TerrainRuntimeInferencePhase::Other;
        }
        return TerrainRuntimeInferencePhase::Other;
    }

    static TerrainRuntimeInferencePhase phaseForPage(TerrainPageKey key,
                                                     AuthorityRequestPriority priority) noexcept {
        if (key.quality == AuthorityQuality::PREVIEW ||
            priority == AuthorityRequestPriority::COARSE_PREVIEW) {
            return TerrainRuntimeInferencePhase::HorizonPreview;
        }
        return phaseForFinal(priority);
    }

    std::vector<float> syntheticMap(int64_t rowBegin, int64_t columnBegin, int edge) const {
        std::vector<float> result(5ULL * edge * edge);
        for (size_t channel = 0; channel < 5; ++channel) {
            for (int row = 0; row < edge; ++row) {
                for (int column = 0; column < edge; ++column) {
                    const float raw =
                        syntheticNoise_[channel]->sample(static_cast<float>(columnBegin + column),
                                                         static_cast<float>(rowBegin + row));
                    result[channel * edge * edge + static_cast<size_t>(row) * edge + column] =
                        interpolateQuantiles(raw, pipelineData_->noiseQuantiles[channel],
                                             pipelineData_->dataQuantiles[channel]);
                }
            }
        }
        const size_t area = static_cast<size_t>(edge) * edge;
        for (size_t index = 0; index < area; ++index) {
            const float elevation = result[index];
            float temperature = result[area + index];
            float temperatureStd = result[2 * area + index];
            const float precipitation = result[3 * area + index];
            float precipitationStd = result[4 * area + index];
            const float lapse = std::clamp(-6.5F + 0.0015F * precipitation, -9.8F, -4.0F) / 1000.0F;
            temperature =
                std::clamp(temperature + lapse * std::max(0.0F, elevation), -10.0F, 40.0F);
            const float baseline =
                pipelineData_->aTemperatureStd * temperature + pipelineData_->bTemperatureStd;
            const float normalized =
                (temperatureStd - pipelineData_->temperatureStdP1) /
                (pipelineData_->temperatureStdP99 - pipelineData_->temperatureStdP1);
            const float clipped = std::max(pipelineData_->temperatureStdP1, -baseline);
            temperatureStd =
                std::max(20.0F, normalized * (pipelineData_->temperatureStdP99 - clipped) +
                                    clipped + baseline);
            precipitationStd *= std::max(0.0F, (185.0F - 0.04111F * precipitation) / 185.0F);
            result[index] = std::copysign(std::sqrt(std::abs(elevation)), elevation);
            result[area + index] = temperature;
            result[2 * area + index] = temperatureStd;
            result[4 * area + index] = precipitationStd;
        }
        return result;
    }

    static std::vector<float> noisePatch(uint64_t seed, int64_t row, int64_t column, int edge,
                                         size_t channels) {
        const auto result = learned::gaussianNoisePatch(seed,
                                                        NativeRect{.rowBegin = row,
                                                                   .columnBegin = column,
                                                                   .rowEnd = row + edge,
                                                                   .columnEnd = column + edge},
                                                        channels, edge);
        if (!result.isReady()) throw std::runtime_error(result.failure()->message);
        return *result.value();
    }

    TerrainRuntimeTensor tensor(std::string name, std::vector<int64_t> shape,
                                std::vector<float> values) const {
        return {.name = std::move(name), .shape = std::move(shape), .values = std::move(values)};
    }

    std::vector<float> execute(TerrainRuntimeModel model, std::vector<TerrainRuntimeTensor> inputs,
                               std::vector<int64_t> expectedShape) const {
        TerrainRuntimeInferenceResult result = executor_->runModel(model, inputs, activePhase_);
        if (!result.succeeded) throw std::runtime_error(result.message);
        size_t expectedValues = 1;
        for (int64_t dimension : expectedShape)
            expectedValues *= static_cast<size_t>(dimension);
        if (result.output.shape != expectedShape || result.output.values.size() != expectedValues)
            throw std::runtime_error("Pinned terrain model returned an incompatible output shape");
        return std::move(result.output.values);
    }

    const std::vector<float>& coarseWindow(WindowIndex index) {
        const WindowKey key{WindowKey::Stage::Coarse, index.row, index.column};
        if (const std::vector<float>* cached = findWindow(key)) return *cached;
        const int64_t rowBegin = index.row * COARSE_STRIDE;
        const int64_t columnBegin = index.column * COARSE_STRIDE;
        constexpr size_t AREA = COARSE_EDGE * COARSE_EDGE;
        std::vector<float> synthetic = syntheticMap(rowBegin, columnBegin, COARSE_EDGE);
        for (size_t pixel = 0; pixel < AREA; ++pixel) {
            float& temperature = synthetic[AREA + pixel];
            if (temperature <= 20.0F) temperature = (temperature - 20.0F) * 1.25F + 20.0F;
        }
        constexpr std::array<size_t, 5> NORMALIZATION_INDICES{0, 2, 3, 4, 5};
        std::vector<float> conditioning(5 * AREA);
        for (size_t channel = 0; channel < 5; ++channel) {
            for (size_t pixel = 0; pixel < AREA; ++pixel) {
                conditioning[channel * AREA + pixel] =
                    (synthetic[channel * AREA + pixel] -
                     COARSE_MEANS[NORMALIZATION_INDICES[channel]]) /
                    COARSE_STDS[NORMALIZATION_INDICES[channel]];
            }
        }
        const std::vector<float> conditioningNoise =
            noisePatch(seed_, rowBegin, columnBegin, COARSE_EDGE, 5);
        const float conditioningAngle = std::atan(0.5F);
        for (size_t indexValue = 0; indexValue < conditioning.size(); ++indexValue) {
            conditioning[indexValue] = std::cos(conditioningAngle) * conditioning[indexValue] +
                                       std::sin(conditioningAngle) * conditioningNoise[indexValue];
        }

        EdmScheduler scheduler;
        std::vector<float> sample = noisePatch(seed_ + 1, rowBegin, columnBegin, COARSE_EDGE, 6);
        for (float& value : sample)
            value *= scheduler.sigma(0);
        constexpr std::array<float, 5> CONDITION_VALUES{
            -2.772588729F, -2.772588729F, -2.772588729F, -2.772588729F, -2.772588729F,
        };
        for (size_t step = 0; step < 20; ++step) {
            const float sigma = scheduler.sigma(step);
            const float inputScale = 1.0F / std::sqrt(sigma * sigma + SIGMA_DATA * SIGMA_DATA);
            std::vector<float> modelInput(11 * AREA);
            for (size_t value = 0; value < 6 * AREA; ++value)
                modelInput[value] = sample[value] * inputScale;
            std::copy(conditioning.begin(), conditioning.end(), modelInput.begin() + 6 * AREA);
            std::vector<TerrainRuntimeTensor> inputs;
            inputs.reserve(7);
            inputs.push_back(tensor("x", {1, 11, COARSE_EDGE, COARSE_EDGE}, std::move(modelInput)));
            inputs.push_back(tensor("noise_labels", {1}, {std::atan(sigma / SIGMA_DATA)}));
            for (size_t condition = 0; condition < CONDITION_VALUES.size(); ++condition)
                inputs.push_back(tensor("cond_" + std::to_string(condition), {1},
                                        {CONDITION_VALUES[condition]}));
            const std::vector<float> model =
                execute(TerrainRuntimeModel::Coarse, std::move(inputs),
                        {COARSE_MODEL_OUTPUT_SHAPE.begin(), COARSE_MODEL_OUTPUT_SHAPE.end()});
            sample = scheduler.step(model, sample);
        }
        std::vector<float> output(COARSE_CHANNELS * AREA);
        for (size_t channel = 0; channel < 6; ++channel) {
            for (size_t pixel = 0; pixel < AREA; ++pixel) {
                output[channel * AREA + pixel] =
                    sample[channel * AREA + pixel] / SIGMA_DATA * COARSE_STDS[channel] +
                    COARSE_MEANS[channel];
            }
        }
        for (size_t pixel = 0; pixel < AREA; ++pixel)
            output[AREA + pixel] = output[pixel] - output[AREA + pixel];
        applyWeight(output, 6, COARSE_EDGE);
        return insertWindow(key, std::move(output));
    }

    static void applyWeight(std::vector<float>& values, size_t valueChannels, int edge) {
        const size_t area = static_cast<size_t>(edge) * edge;
        for (int row = 0; row < edge; ++row) {
            const float rowWeight = learned::linearWindowWeight(static_cast<size_t>(row), edge);
            for (int column = 0; column < edge; ++column) {
                const float weight =
                    rowWeight * learned::linearWindowWeight(static_cast<size_t>(column), edge);
                const size_t pixel = static_cast<size_t>(row) * edge + column;
                for (size_t channel = 0; channel < valueChannels; ++channel)
                    values[channel * area + pixel] *= weight;
                values[valueChannels * area + pixel] = weight;
            }
        }
    }

    std::vector<float> slice(WindowKey::Stage stage, NativeRect region, int channels, int edge,
                             int stride) {
        const auto windows =
            learned::intersectingWindows(region, {.edge = static_cast<uint16_t>(edge),
                                                  .stride = static_cast<uint16_t>(stride),
                                                  .inferenceSteps = 1,
                                                  .batchSize = 1});
        if (stage == WindowKey::Stage::Coarse) {
            for (WindowIndex index : windows)
                coarseWindow(index);
        } else if (stage == WindowKey::Stage::LatentFinal) {
            ensureLatentWindows(windows);
        } else if (stage == WindowKey::Stage::Decoder) {
            ensureDecoderWindows(windows);
        }
        const int height = static_cast<int>(region.height());
        const int width = static_cast<int>(region.width());
        std::vector<float> output(static_cast<size_t>(channels) * height * width);
        for (WindowIndex index : windows) {
            const WindowKey key{stage, index.row, index.column};
            const std::vector<float>* window = findWindow(key);
            if (window == nullptr) throw std::runtime_error("A required tensor window was evicted");
            const int64_t windowRow = index.row * stride;
            const int64_t windowColumn = index.column * stride;
            const int64_t rowBegin = std::max(region.rowBegin, windowRow);
            const int64_t rowEnd = std::min(region.rowEnd, windowRow + edge);
            const int64_t columnBegin = std::max(region.columnBegin, windowColumn);
            const int64_t columnEnd = std::min(region.columnEnd, windowColumn + edge);
            for (int64_t row = rowBegin; row < rowEnd; ++row) {
                for (int64_t column = columnBegin; column < columnEnd; ++column) {
                    const size_t sourcePixel = static_cast<size_t>(row - windowRow) * edge +
                                               static_cast<size_t>(column - windowColumn);
                    const size_t destinationPixel =
                        static_cast<size_t>(row - region.rowBegin) * width +
                        static_cast<size_t>(column - region.columnBegin);
                    for (int channel = 0; channel < channels; ++channel) {
                        output[static_cast<size_t>(channel) * height * width + destinationPixel] +=
                            (*window)[static_cast<size_t>(channel) * edge * edge + sourcePixel];
                    }
                }
            }
        }
        return output;
    }

    std::array<float, 58> latentConditioning(WindowIndex index) {
        const NativeRect region{.rowBegin = index.row - 1,
                                .columnBegin = index.column - 1,
                                .rowEnd = index.row + 3,
                                .columnEnd = index.column + 3};
        const std::vector<float> coarse =
            slice(WindowKey::Stage::Coarse, region, COARSE_CHANNELS, COARSE_EDGE, COARSE_STRIDE);
        constexpr size_t AREA = 16;
        std::array<float, 6 * AREA> unweighted{};
        for (size_t pixel = 0; pixel < AREA; ++pixel) {
            const float weight = coarse[6 * AREA + pixel];
            for (size_t channel = 0; channel < 6; ++channel)
                unweighted[channel * AREA + pixel] =
                    weight > 1.0E-6F ? coarse[channel * AREA + pixel] / weight : 0.0F;
        }
        std::array<float, 7 * AREA> normalized{};
        for (size_t channel = 0; channel < 6; ++channel) {
            for (size_t pixel = 0; pixel < AREA; ++pixel) {
                const float value =
                    (unweighted[channel * AREA + pixel] - CONDITIONING_MEANS[channel]) /
                    CONDITIONING_STDS[channel];
                normalized[channel * AREA + pixel] = std::isnan(value) ? 0.0F : value;
            }
        }
        const float mask = (1.0F - CONDITIONING_MEANS[6]) / CONDITIONING_STDS[6];
        for (size_t pixel = 0; pixel < AREA; ++pixel)
            normalized[6 * AREA + pixel] = mask;
        constexpr std::array<int, 6> DIMENSIONS{16, 16, 4, 16, 5, 1};
        int totalDimensions = 0;
        for (int dimension : DIMENSIONS)
            totalDimensions += dimension;
        const float constant = std::sqrt(static_cast<float>(totalDimensions * DIMENSIONS.size()));
        std::array<float, 6> scales{};
        for (size_t indexScale = 0; indexScale < scales.size(); ++indexScale)
            scales[indexScale] = constant / std::sqrt(static_cast<float>(DIMENSIONS[indexScale])) /
                                 static_cast<float>(DIMENSIONS.size());
        std::array<float, 58> output{};
        size_t offset = 0;
        auto append = [&](std::span<const float> values, float scale) {
            for (float value : values)
                output[offset++] = value * scale;
        };
        append(std::span<const float>(normalized.data(), AREA), scales[0]);
        append(std::span<const float>(normalized.data() + AREA, AREA), scales[1]);
        std::array<float, 4> climate{};
        for (size_t channel = 0; channel < 4; ++channel) {
            for (int row = 1; row < 3; ++row)
                for (int column = 1; column < 3; ++column)
                    climate[channel] +=
                        normalized[(2 + channel) * AREA + static_cast<size_t>(row) * 4 + column];
            climate[channel] *= 0.25F;
            if (std::isnan(climate[channel])) climate[channel] = 0.0F;
        }
        append(climate, scales[2]);
        append(std::span<const float>(normalized.data() + 6 * AREA, AREA), scales[3]);
        const std::array<float, 5> histogram{};
        append(histogram, scales[4]);
        output[offset] = -0.5F * std::sqrt(12.0F) * scales[5];
        return output;
    }

    void ensureLatentWindows(std::span<const WindowIndex> requested) {
        std::vector<WindowIndex> missingInitial;
        std::vector<WindowIndex> missingFinal;
        for (WindowIndex index : requested) {
            if (!hasWindow({WindowKey::Stage::LatentInitial, index.row, index.column}))
                missingInitial.push_back(index);
            if (!hasWindow({WindowKey::Stage::LatentFinal, index.row, index.column}))
                missingFinal.push_back(index);
        }
        std::vector<WindowIndex> conditioning = missingInitial;
        conditioning.insert(conditioning.end(), missingFinal.begin(), missingFinal.end());
        std::sort(conditioning.begin(), conditioning.end());
        conditioning.erase(std::unique(conditioning.begin(), conditioning.end()),
                           conditioning.end());
        for (WindowIndex index : conditioning)
            static_cast<void>(latentConditioning(index));
        runLatentBatches(missingInitial, false);
        runLatentBatches(missingFinal, true);
    }

    void runLatentBatches(std::span<const WindowIndex> requested, bool finalStep) {
        for (size_t begin = 0; begin < requested.size(); begin += MODEL_BATCH) {
            const size_t count = std::min<size_t>(MODEL_BATCH, requested.size() - begin);
            std::array<WindowIndex, MODEL_BATCH> indices{};
            for (size_t batch = 0; batch < MODEL_BATCH; ++batch)
                indices[batch] = requested[begin + std::min(batch, count - 1)];
            constexpr size_t AREA = LATENT_EDGE * LATENT_EDGE;
            std::vector<float> modelInput(MODEL_BATCH * 5 * AREA);
            std::vector<float> conditioning(MODEL_BATCH * 58);
            std::array<std::vector<float>, MODEL_BATCH> noisySamples;
            const float angle =
                finalStep ? std::atan(0.35F / SIGMA_DATA) : std::atan(SIGMA_MAX / SIGMA_DATA);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            for (size_t batch = 0; batch < MODEL_BATCH; ++batch) {
                const WindowIndex index = indices[batch];
                const std::array<float, 58> condition = latentConditioning(index);
                std::copy(condition.begin(), condition.end(), conditioning.begin() + batch * 58);
                std::vector<float> sample(5 * AREA);
                if (finalStep) {
                    const auto* initial =
                        findWindow({WindowKey::Stage::LatentInitial, index.row, index.column});
                    if (initial == nullptr)
                        throw std::runtime_error("Initial latent window is missing");
                    for (size_t pixel = 0; pixel < AREA; ++pixel) {
                        const float weight = (*initial)[5 * AREA + pixel];
                        for (size_t channel = 0; channel < 5; ++channel)
                            sample[channel * AREA + pixel] =
                                weight > 1.0E-6F
                                    ? (*initial)[channel * AREA + pixel] / weight * SIGMA_DATA
                                    : 0.0F;
                    }
                }
                const int64_t rowBegin = index.row * LATENT_STRIDE;
                const int64_t columnBegin = index.column * LATENT_STRIDE;
                const std::vector<float> noise = noisePatch(seed_ + (finalStep ? 5820 : 5819),
                                                            rowBegin, columnBegin, LATENT_EDGE, 5);
                noisySamples[batch].resize(5 * AREA);
                for (size_t value = 0; value < 5 * AREA; ++value) {
                    noisySamples[batch][value] =
                        cosine * sample[value] + sine * noise[value] * SIGMA_DATA;
                    modelInput[batch * 5 * AREA + value] = noisySamples[batch][value] / SIGMA_DATA;
                }
            }
            std::vector<float> labels(MODEL_BATCH, angle);
            std::vector<TerrainRuntimeTensor> inputs;
            inputs.push_back(
                tensor("x", {MODEL_BATCH, 5, LATENT_EDGE, LATENT_EDGE}, std::move(modelInput)));
            inputs.push_back(tensor("noise_labels", {MODEL_BATCH}, std::move(labels)));
            inputs.push_back(tensor("cond_0", {MODEL_BATCH, 58}, std::move(conditioning)));
            const std::vector<float> prediction =
                execute(TerrainRuntimeModel::Base, std::move(inputs),
                        {BASE_MODEL_OUTPUT_SHAPE.begin(), BASE_MODEL_OUTPUT_SHAPE.end()});
            for (size_t batch = 0; batch < count; ++batch) {
                std::vector<float> output(LATENT_CHANNELS * AREA);
                for (size_t value = 0; value < 5 * AREA; ++value) {
                    const float predicted = -prediction[batch * 5 * AREA + value];
                    output[value] =
                        (cosine * noisySamples[batch][value] - sine * SIGMA_DATA * predicted) /
                        SIGMA_DATA;
                }
                applyWeight(output, 5, LATENT_EDGE);
                insertWindow(
                    {finalStep ? WindowKey::Stage::LatentFinal : WindowKey::Stage::LatentInitial,
                     indices[batch].row, indices[batch].column},
                    std::move(output));
            }
        }
    }

    void ensureDecoderWindows(std::span<const WindowIndex> requested) {
        std::vector<WindowIndex> missing;
        missing.reserve(requested.size());
        for (WindowIndex index : requested) {
            if (!hasWindow({WindowKey::Stage::Decoder, index.row, index.column}))
                missing.push_back(index);
        }
        std::sort(missing.begin(), missing.end());
        missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
        if (missing.empty()) return;

        std::vector<WindowIndex> requiredLatents;
        for (WindowIndex index : missing) {
            const int latentEdge = DECODER_EDGE / LATENT_COMPRESSION;
            const int64_t latentRow = index.row * DECODER_STRIDE / LATENT_COMPRESSION;
            const int64_t latentColumn = index.column * DECODER_STRIDE / LATENT_COMPRESSION;
            const std::vector<WindowIndex> dependencies =
                learned::intersectingWindows({.rowBegin = latentRow,
                                              .columnBegin = latentColumn,
                                              .rowEnd = latentRow + latentEdge,
                                              .columnEnd = latentColumn + latentEdge},
                                             learned::LATENT_WINDOW);
            requiredLatents.insert(requiredLatents.end(), dependencies.begin(), dependencies.end());
        }
        std::sort(requiredLatents.begin(), requiredLatents.end());
        requiredLatents.erase(std::unique(requiredLatents.begin(), requiredLatents.end()),
                              requiredLatents.end());
        ensureLatentWindows(requiredLatents);
        runDecoderBatches(missing);
    }

    void runDecoderBatches(std::span<const WindowIndex> requested) {
        constexpr int LATENT_DECODER_EDGE = DECODER_EDGE / LATENT_COMPRESSION;
        constexpr size_t LATENT_AREA =
            static_cast<size_t>(LATENT_DECODER_EDGE) * LATENT_DECODER_EDGE;
        constexpr size_t AREA = DECODER_EDGE * DECODER_EDGE;
        constexpr size_t INPUT_VALUES = 5 * AREA;
        const float angle = std::atan(SIGMA_MAX / SIGMA_DATA);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);

        for (size_t begin = 0; begin < requested.size(); begin += MODEL_BATCH) {
            const size_t count = std::min<size_t>(MODEL_BATCH, requested.size() - begin);
            std::array<WindowIndex, MODEL_BATCH> indices{};
            std::array<std::vector<float>, MODEL_BATCH> noisySamples;
            std::vector<float> modelInput(MODEL_BATCH * INPUT_VALUES);
            for (size_t batch = 0; batch < count; ++batch) {
                const WindowIndex index = requested[begin + batch];
                indices[batch] = index;
                const int64_t latentRow = index.row * DECODER_STRIDE / LATENT_COMPRESSION;
                const int64_t latentColumn = index.column * DECODER_STRIDE / LATENT_COMPRESSION;
                const std::vector<float> latent =
                    slice(WindowKey::Stage::LatentFinal,
                          {.rowBegin = latentRow,
                           .columnBegin = latentColumn,
                           .rowEnd = latentRow + LATENT_DECODER_EDGE,
                           .columnEnd = latentColumn + LATENT_DECODER_EDGE},
                          LATENT_CHANNELS, LATENT_EDGE, LATENT_STRIDE);
                const size_t inputOffset = batch * INPUT_VALUES;
                for (size_t channel = 0; channel < 4; ++channel) {
                    for (int row = 0; row < DECODER_EDGE; ++row) {
                        const int sourceRow = row * LATENT_DECODER_EDGE / DECODER_EDGE;
                        for (int column = 0; column < DECODER_EDGE; ++column) {
                            const int sourceColumn = column * LATENT_DECODER_EDGE / DECODER_EDGE;
                            const size_t sourcePixel =
                                static_cast<size_t>(sourceRow) * LATENT_DECODER_EDGE + sourceColumn;
                            const float weight = latent[5 * LATENT_AREA + sourcePixel];
                            modelInput[inputOffset + (channel + 1) * AREA +
                                       static_cast<size_t>(row) * DECODER_EDGE + column] =
                                weight > 1.0E-6F
                                    ? latent[channel * LATENT_AREA + sourcePixel] / weight
                                    : 0.0F;
                        }
                    }
                }
                const int64_t rowBegin = index.row * DECODER_STRIDE;
                const int64_t columnBegin = index.column * DECODER_STRIDE;
                const std::vector<float> noise =
                    noisePatch(seed_ + 5819, rowBegin, columnBegin, DECODER_EDGE, 1);
                noisySamples[batch].resize(AREA);
                for (size_t pixel = 0; pixel < AREA; ++pixel) {
                    noisySamples[batch][pixel] = sine * noise[pixel] * SIGMA_DATA;
                    modelInput[inputOffset + pixel] = noisySamples[batch][pixel] / SIGMA_DATA;
                }
            }
            for (size_t batch = count; batch < MODEL_BATCH; ++batch) {
                indices[batch] = indices[count - 1];
                noisySamples[batch] = noisySamples[count - 1];
                std::copy_n(modelInput.begin() +
                                static_cast<std::ptrdiff_t>((count - 1) * INPUT_VALUES),
                            INPUT_VALUES,
                            modelInput.begin() + static_cast<std::ptrdiff_t>(batch * INPUT_VALUES));
            }

            std::vector<TerrainRuntimeTensor> inputs;
            inputs.push_back(
                tensor("x", {MODEL_BATCH, 5, DECODER_EDGE, DECODER_EDGE}, std::move(modelInput)));
            inputs.push_back(
                tensor("noise_labels", {MODEL_BATCH}, std::vector<float>(MODEL_BATCH, angle)));
            const std::vector<float> prediction =
                execute(TerrainRuntimeModel::Decoder, std::move(inputs),
                        {DECODER_MODEL_OUTPUT_SHAPE.begin(), DECODER_MODEL_OUTPUT_SHAPE.end()});
            for (size_t batch = 0; batch < count; ++batch) {
                std::vector<float> output(DECODER_CHANNELS * AREA);
                for (size_t pixel = 0; pixel < AREA; ++pixel) {
                    const float predicted = -prediction[batch * AREA + pixel];
                    output[pixel] =
                        (cosine * noisySamples[batch][pixel] - sine * SIGMA_DATA * predicted) /
                        SIGMA_DATA;
                }
                applyWeight(output, 1, DECODER_EDGE);
                insertWindow({WindowKey::Stage::Decoder, indices[batch].row, indices[batch].column},
                             std::move(output));
            }
        }
    }

    struct ClimateResult {
        std::vector<float> temperature;
        std::vector<float> temperatureVariability;
        std::vector<float> precipitation;
        std::vector<float> precipitationVariability;
        std::vector<float> lapseRate;
    };

    ClimateResult climate(NativeRect region, std::span<const float> elevation) {
        const int height = static_cast<int>(region.height());
        const int width = static_cast<int>(region.width());
        constexpr int COARSE_NATIVE_STRIDE = LATENT_STRIDE * LATENT_COMPRESSION;
        const int64_t coarseRowBegin = learned::floorDivide(region.rowBegin, COARSE_NATIVE_STRIDE);
        const int64_t coarseColumnBegin =
            learned::floorDivide(region.columnBegin, COARSE_NATIVE_STRIDE);
        const int64_t coarseRowEnd = ceilDivide(region.rowEnd, COARSE_NATIVE_STRIDE);
        const int64_t coarseColumnEnd = ceilDivide(region.columnEnd, COARSE_NATIVE_STRIDE);
        constexpr int WINDOW = 15;
        constexpr int PAD = (WINDOW - 1) / 2 + 1;
        const NativeRect coarseRegion{.rowBegin = coarseRowBegin - PAD,
                                      .columnBegin = coarseColumnBegin - PAD,
                                      .rowEnd = coarseRowEnd + PAD,
                                      .columnEnd = coarseColumnEnd + PAD};
        const int coarseHeight = static_cast<int>(coarseRegion.height());
        const int coarseWidth = static_cast<int>(coarseRegion.width());
        const size_t coarseArea = static_cast<size_t>(coarseHeight) * coarseWidth;
        const std::vector<float> weighted = slice(WindowKey::Stage::Coarse, coarseRegion,
                                                  COARSE_CHANNELS, COARSE_EDGE, COARSE_STRIDE);
        std::array<Matrix, 6> coarse;
        for (Matrix& matrix : coarse)
            matrix = Matrix(coarseHeight, coarseWidth);
        for (size_t pixel = 0; pixel < coarseArea; ++pixel) {
            const float weight = weighted[6 * coarseArea + pixel];
            for (size_t channel = 0; channel < 6; ++channel)
                coarse[channel].values[pixel] =
                    weight > 1.0E-6F ? weighted[channel * coarseArea + pixel] / weight : 0.0F;
        }
        Matrix coarseElevation(coarseHeight, coarseWidth);
        for (size_t pixel = 0; pixel < coarseArea; ++pixel) {
            const float value = std::max(0.0F, coarse[0].values[pixel]);
            coarseElevation.values[pixel] = value * value;
        }

        const int regressionHeight = coarseHeight - WINDOW + 1;
        const int regressionWidth = coarseWidth - WINDOW + 1;
        Matrix baseline(regressionHeight, regressionWidth);
        Matrix lapse(regressionHeight, regressionWidth);
        for (int row = 0; row < regressionHeight; ++row) {
            for (int column = 0; column < regressionWidth; ++column) {
                double meanTemperature = 0.0;
                double meanElevation = 0.0;
                double meanElevationSquared = 0.0;
                double meanProduct = 0.0;
                double landCount = 0.0;
                for (int offsetRow = 0; offsetRow < WINDOW; ++offsetRow) {
                    for (int offsetColumn = 0; offsetColumn < WINDOW; ++offsetColumn) {
                        const float land =
                            coarseElevation.at(row + offsetRow, column + offsetColumn) > 0.0F
                                ? 1.0F
                                : 0.0F;
                        const float elevationValue =
                            coarseElevation.at(row + offsetRow, column + offsetColumn);
                        const float temperatureValue =
                            coarse[2].at(row + offsetRow, column + offsetColumn);
                        meanTemperature += temperatureValue * land;
                        meanElevation += elevationValue * land;
                        meanElevationSquared += elevationValue * elevationValue * land;
                        meanProduct += elevationValue * temperatureValue * land;
                        landCount += land;
                    }
                }
                const double denominator = landCount + 1.0E-6;
                meanTemperature /= denominator;
                meanElevation /= denominator;
                meanElevationSquared /= denominator;
                meanProduct /= denominator;
                const double variance = meanElevationSquared - meanElevation * meanElevation;
                const double covariance = meanProduct - meanElevation * meanTemperature;
                double beta = variance < 1.0 || landCount < 0.02 * WINDOW * WINDOW
                                  ? -0.0065
                                  : covariance / (variance + 1.0E-6);
                beta = std::clamp(beta, -0.012, 0.0);
                const int center = WINDOW / 2;
                baseline.at(row, column) =
                    coarse[2].at(row + center, column + center) -
                    static_cast<float>(beta) * coarseElevation.at(row + center, column + center);
                lapse.at(row, column) = static_cast<float>(beta);
            }
        }
        constexpr int CENTRAL_PAD = WINDOW / 2;
        const int centralHeight = coarseHeight - 2 * CENTRAL_PAD;
        const int centralWidth = coarseWidth - 2 * CENTRAL_PAD;
        std::array<Matrix, 3> central{
            Matrix(centralHeight, centralWidth),
            Matrix(centralHeight, centralWidth),
            Matrix(centralHeight, centralWidth),
        };
        for (size_t channel = 0; channel < central.size(); ++channel)
            for (int row = 0; row < centralHeight; ++row)
                for (int column = 0; column < centralWidth; ++column)
                    central[channel].at(row, column) =
                        coarse[channel + 3].at(row + CENTRAL_PAD, column + CENTRAL_PAD);

        const size_t area = static_cast<size_t>(height) * width;
        ClimateResult result{.temperature = std::vector<float>(area),
                             .temperatureVariability = std::vector<float>(area),
                             .precipitation = std::vector<float>(area),
                             .precipitationVariability = std::vector<float>(area),
                             .lapseRate = std::vector<float>(area)};
        for (int row = 0; row < height; ++row) {
            const float gridRow =
                (static_cast<float>(region.rowBegin + row) + 0.5F) / COARSE_NATIVE_STRIDE -
                static_cast<float>(coarseRowBegin) + 0.5F;
            for (int column = 0; column < width; ++column) {
                const float gridColumn = (static_cast<float>(region.columnBegin + column) + 0.5F) /
                                             COARSE_NATIVE_STRIDE -
                                         static_cast<float>(coarseColumnBegin) + 0.5F;
                const size_t pixel = static_cast<size_t>(row) * width + column;
                const float beta = bilinearSample(lapse, gridRow, gridColumn);
                result.temperature[pixel] = bilinearSample(baseline, gridRow, gridColumn) +
                                            beta * std::max(0.0F, elevation[pixel]);
                result.temperatureVariability[pixel] =
                    bilinearSample(central[0], gridRow, gridColumn);
                result.precipitation[pixel] = bilinearSample(central[1], gridRow, gridColumn);
                result.precipitationVariability[pixel] =
                    bilinearSample(central[2], gridRow, gridColumn);
                result.lapseRate[pixel] = beta;
            }
        }
        return result;
    }

    TerrainAuthorityPage makePage(const GenerationIdentity& identity, TerrainPageKey key,
                                  std::span<const float> elevation,
                                  const ClimateResult& climateResult) const {
        TerrainAuthorityPage page;
        page.key = key;
        page.generationSeed = identity.seed;
        page.generationFingerprint = identity.fingerprint();
        page.samples.resize(learned::AUTHORITY_PAGE_SAMPLE_COUNT);
        for (size_t pixel = 0; pixel < page.samples.size(); ++pixel) {
            learned::QuantizedTerrainSample& sample = page.samples[pixel];
            sample = quantizeInfiniteDiffusionSample(
                elevation[pixel], climateResult.temperature[pixel],
                climateResult.temperatureVariability[pixel], climateResult.precipitation[pixel],
                climateResult.precipitationVariability[pixel], climateResult.lapseRate[pixel]);
        }
        return page;
    }

    struct FinalFields {
        std::vector<float> elevation;
        ClimateResult climate;
    };

    FinalFields inferPreviewFieldsWithBoundary(NativeRect region) {
        // PREVIEW is the decoder-free low-frequency component of the FINAL
        // reconstruction. Running the exact cleanup operator with a zero
        // residual keeps it on LatentFinal channel 4's seeded lineage instead
        // of drawing a second, unrelated terrain surface from the coarse
        // conditioning model.
        constexpr int PADDING_LOW = 6;
        constexpr int PADDING_HIGH = PADDING_LOW * LATENT_COMPRESSION;
        const int64_t paddedRowBegin =
            learned::floorDivide(region.rowBegin - PADDING_HIGH, LATENT_COMPRESSION) *
            LATENT_COMPRESSION;
        const int64_t paddedColumnBegin =
            learned::floorDivide(region.columnBegin - PADDING_HIGH, LATENT_COMPRESSION) *
            LATENT_COMPRESSION;
        const int64_t paddedRowEnd =
            ceilDivide(region.rowEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION;
        const int64_t paddedColumnEnd =
            ceilDivide(region.columnEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION;
        const NativeRect padded{.rowBegin = paddedRowBegin,
                                .columnBegin = paddedColumnBegin,
                                .rowEnd = paddedRowEnd,
                                .columnEnd = paddedColumnEnd};
        const int paddedHeight = static_cast<int>(padded.height());
        const int paddedWidth = static_cast<int>(padded.width());
        const NativeRect latentRegion{.rowBegin = padded.rowBegin / LATENT_COMPRESSION,
                                      .columnBegin = padded.columnBegin / LATENT_COMPRESSION,
                                      .rowEnd = padded.rowEnd / LATENT_COMPRESSION,
                                      .columnEnd = padded.columnEnd / LATENT_COMPRESSION};
        const int latentHeight = static_cast<int>(latentRegion.height());
        const int latentWidth = static_cast<int>(latentRegion.width());
        const size_t latentArea = static_cast<size_t>(latentHeight) * latentWidth;
        const std::vector<float> latent = slice(WindowKey::Stage::LatentFinal, latentRegion,
                                                LATENT_CHANNELS, LATENT_EDGE, LATENT_STRIDE);
        Matrix lowFrequency(latentHeight, latentWidth);
        for (size_t pixel = 0; pixel < latentArea; ++pixel) {
            const float weight = latent[5 * latentArea + pixel];
            const float value = weight > 1.0E-6F ? latent[4 * latentArea + pixel] / weight : 0.0F;
            lowFrequency.values[pixel] = value * LOW_FREQUENCY_STD + LOW_FREQUENCY_MEAN;
        }
        const Matrix zeroResidual(paddedHeight, paddedWidth);
        const Matrix cleaned = laplacianDenoise(zeroResidual, lowFrequency, 5.0F);
        const Matrix encodedElevation = laplacianDecode(zeroResidual, cleaned);
        const int rowOffset = static_cast<int>(region.rowBegin - padded.rowBegin);
        const int columnOffset = static_cast<int>(region.columnBegin - padded.columnBegin);
        const int regionHeight = static_cast<int>(region.height());
        const int regionWidth = static_cast<int>(region.width());
        std::vector<float> elevation(static_cast<size_t>(regionHeight) * regionWidth);
        for (int row = 0; row < regionHeight; ++row) {
            for (int column = 0; column < regionWidth; ++column) {
                const float encoded = encodedElevation.at(rowOffset + row, columnOffset + column);
                elevation[static_cast<size_t>(row) * regionWidth + column] =
                    std::copysign(encoded * encoded, encoded);
            }
        }
        ClimateResult climateResult = climate(region, elevation);
        return {.elevation = std::move(elevation), .climate = std::move(climateResult)};
    }

    FinalFields inferPreviewFields(NativeRect region) {
        // Match FINAL's output apron so persisted pages and larger diagnostic
        // requests observe the same reconstruction at a shared coordinate.
        constexpr int64_t OUTPUT_APRON = LATENT_COMPRESSION;
        const NativeRect expanded{
            .rowBegin = region.rowBegin - OUTPUT_APRON,
            .columnBegin = region.columnBegin - OUTPUT_APRON,
            .rowEnd = region.rowEnd + OUTPUT_APRON,
            .columnEnd = region.columnEnd + OUTPUT_APRON,
        };
        FinalFields source = inferPreviewFieldsWithBoundary(expanded);
        const size_t sourceWidth = static_cast<size_t>(expanded.width());
        const size_t destinationWidth = static_cast<size_t>(region.width());
        const size_t destinationHeight = static_cast<size_t>(region.height());
        const size_t destinationCount = destinationWidth * destinationHeight;
        FinalFields destination{
            .elevation = std::vector<float>(destinationCount),
            .climate =
                {
                    .temperature = std::vector<float>(destinationCount),
                    .temperatureVariability = std::vector<float>(destinationCount),
                    .precipitation = std::vector<float>(destinationCount),
                    .precipitationVariability = std::vector<float>(destinationCount),
                    .lapseRate = std::vector<float>(destinationCount),
                },
        };
        const auto crop = [&](std::span<const float> input, std::span<float> output) {
            for (size_t row = 0; row < destinationHeight; ++row) {
                const size_t sourceOffset = (row + OUTPUT_APRON) * sourceWidth + OUTPUT_APRON;
                std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                            destinationWidth,
                            output.begin() + static_cast<std::ptrdiff_t>(row * destinationWidth));
            }
        };
        crop(source.elevation, destination.elevation);
        crop(source.climate.temperature, destination.climate.temperature);
        crop(source.climate.temperatureVariability, destination.climate.temperatureVariability);
        crop(source.climate.precipitation, destination.climate.precipitation);
        crop(source.climate.precipitationVariability, destination.climate.precipitationVariability);
        crop(source.climate.lapseRate, destination.climate.lapseRate);
        return destination;
    }

    TerrainAuthorityPage inferPreview(const GenerationIdentity& identity, TerrainPageKey key) {
        const int64_t rowBegin = key.coordinate.row * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        const int64_t columnBegin = key.coordinate.column * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        const NativeRect region{.rowBegin = rowBegin,
                                .columnBegin = columnBegin,
                                .rowEnd = rowBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE,
                                .columnEnd = columnBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE};
        FinalFields fields = inferPreviewFields(region);
        return makePage(identity, key, fields.elevation, fields.climate);
    }

    FinalFields inferFinalFieldsWithBoundary(NativeRect region) {
        constexpr int PADDING_LOW = 6;
        constexpr int PADDING_HIGH = PADDING_LOW * LATENT_COMPRESSION;
        const int64_t paddedRowBegin =
            learned::floorDivide(region.rowBegin - PADDING_HIGH, LATENT_COMPRESSION) *
            LATENT_COMPRESSION;
        const int64_t paddedColumnBegin =
            learned::floorDivide(region.columnBegin - PADDING_HIGH, LATENT_COMPRESSION) *
            LATENT_COMPRESSION;
        const int64_t paddedRowEnd =
            ceilDivide(region.rowEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION;
        const int64_t paddedColumnEnd =
            ceilDivide(region.columnEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION;
        const NativeRect padded{.rowBegin = paddedRowBegin,
                                .columnBegin = paddedColumnBegin,
                                .rowEnd = paddedRowEnd,
                                .columnEnd = paddedColumnEnd};
        const int paddedHeight = static_cast<int>(padded.height());
        const int paddedWidth = static_cast<int>(padded.width());
        const size_t paddedArea = static_cast<size_t>(paddedHeight) * paddedWidth;
        const std::vector<float> residualWeighted = slice(
            WindowKey::Stage::Decoder, padded, DECODER_CHANNELS, DECODER_EDGE, DECODER_STRIDE);
        Matrix residual(paddedHeight, paddedWidth);
        for (size_t pixel = 0; pixel < paddedArea; ++pixel) {
            const float weight = residualWeighted[paddedArea + pixel];
            const float value = weight > 1.0E-6F ? residualWeighted[pixel] / weight : 0.0F;
            residual.values[pixel] = value * RESIDUAL_STD + RESIDUAL_MEAN;
        }
        const NativeRect latentRegion{.rowBegin = padded.rowBegin / LATENT_COMPRESSION,
                                      .columnBegin = padded.columnBegin / LATENT_COMPRESSION,
                                      .rowEnd = padded.rowEnd / LATENT_COMPRESSION,
                                      .columnEnd = padded.columnEnd / LATENT_COMPRESSION};
        const int latentHeight = static_cast<int>(latentRegion.height());
        const int latentWidth = static_cast<int>(latentRegion.width());
        const size_t latentArea = static_cast<size_t>(latentHeight) * latentWidth;
        const std::vector<float> latent = slice(WindowKey::Stage::LatentFinal, latentRegion,
                                                LATENT_CHANNELS, LATENT_EDGE, LATENT_STRIDE);
        Matrix lowFrequency(latentHeight, latentWidth);
        for (size_t pixel = 0; pixel < latentArea; ++pixel) {
            const float weight = latent[5 * latentArea + pixel];
            const float value = weight > 1.0E-6F ? latent[4 * latentArea + pixel] / weight : 0.0F;
            lowFrequency.values[pixel] = value * LOW_FREQUENCY_STD + LOW_FREQUENCY_MEAN;
        }
        const Matrix cleaned = laplacianDenoise(residual, lowFrequency, 5.0F);
        const Matrix encodedElevation = laplacianDecode(residual, cleaned);
        const size_t sampleCount = static_cast<size_t>(region.height()) * region.width();
        std::vector<float> elevation(sampleCount);
        const int rowOffset = static_cast<int>(region.rowBegin - padded.rowBegin);
        const int columnOffset = static_cast<int>(region.columnBegin - padded.columnBegin);
        const int regionHeight = static_cast<int>(region.height());
        const int regionWidth = static_cast<int>(region.width());
        for (int row = 0; row < regionHeight; ++row) {
            for (int column = 0; column < regionWidth; ++column) {
                const float encoded = encodedElevation.at(rowOffset + row, columnOffset + column);
                elevation[static_cast<size_t>(row) * regionWidth + column] =
                    std::copysign(encoded * encoded, encoded);
            }
        }
        const ClimateResult climateResult = climate(region, elevation);
        return {.elevation = std::move(elevation), .climate = std::move(climateResult)};
    }

    FinalFields inferFinalFields(NativeRect region) {
        // The low-frequency cleanup has a six-cell latent stencil. Give the
        // requested output one additional latent cell before applying the
        // existing reconstruction padding, then crop it away. This makes a
        // sample independent of whether its caller requested one persisted
        // page or a larger transient hydrology rectangle.
        constexpr int64_t OUTPUT_APRON = LATENT_COMPRESSION;
        const NativeRect expanded{
            .rowBegin = region.rowBegin - OUTPUT_APRON,
            .columnBegin = region.columnBegin - OUTPUT_APRON,
            .rowEnd = region.rowEnd + OUTPUT_APRON,
            .columnEnd = region.columnEnd + OUTPUT_APRON,
        };
        FinalFields source = inferFinalFieldsWithBoundary(expanded);
        const size_t sourceWidth = static_cast<size_t>(expanded.width());
        const size_t destinationWidth = static_cast<size_t>(region.width());
        const size_t destinationHeight = static_cast<size_t>(region.height());
        const size_t destinationCount = destinationWidth * destinationHeight;
        FinalFields destination{
            .elevation = std::vector<float>(destinationCount),
            .climate =
                {
                    .temperature = std::vector<float>(destinationCount),
                    .temperatureVariability = std::vector<float>(destinationCount),
                    .precipitation = std::vector<float>(destinationCount),
                    .precipitationVariability = std::vector<float>(destinationCount),
                    .lapseRate = std::vector<float>(destinationCount),
                },
        };
        const auto crop = [&](std::span<const float> input, std::span<float> output) {
            for (size_t row = 0; row < destinationHeight; ++row) {
                const size_t sourceOffset = (row + OUTPUT_APRON) * sourceWidth + OUTPUT_APRON;
                std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(sourceOffset),
                            destinationWidth,
                            output.begin() + static_cast<std::ptrdiff_t>(row * destinationWidth));
            }
        };
        crop(source.elevation, destination.elevation);
        crop(source.climate.temperature, destination.climate.temperature);
        crop(source.climate.temperatureVariability, destination.climate.temperatureVariability);
        crop(source.climate.precipitation, destination.climate.precipitation);
        crop(source.climate.precipitationVariability, destination.climate.precipitationVariability);
        crop(source.climate.lapseRate, destination.climate.lapseRate);
        return destination;
    }

    TerrainAuthorityPage inferFinal(const GenerationIdentity& identity, TerrainPageKey key) {
        const int64_t rowBegin = key.coordinate.row * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        const int64_t columnBegin = key.coordinate.column * learned::AUTHORITY_PAGE_NATIVE_EDGE;
        const NativeRect region{.rowBegin = rowBegin,
                                .columnBegin = columnBegin,
                                .rowEnd = rowBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE,
                                .columnEnd = columnBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE};
        FinalFields fields = inferFinalFields(region);
        return makePage(identity, key, fields.elevation, fields.climate);
    }

    learned::PhysicalTerrainGrid inferFinalGrid(NativeRect region) {
        FinalFields fields = inferFinalFields(region);
        learned::PhysicalTerrainGrid grid{
            .region = region,
            .samples = std::vector<learned::PhysicalTerrainSample>(fields.elevation.size()),
        };
        for (size_t pixel = 0; pixel < grid.samples.size(); ++pixel) {
            const learned::QuantizedTerrainSample quantized = quantizeInfiniteDiffusionSample(
                fields.elevation[pixel], fields.climate.temperature[pixel],
                fields.climate.temperatureVariability[pixel], fields.climate.precipitation[pixel],
                fields.climate.precipitationVariability[pixel], fields.climate.lapseRate[pixel]);
            grid.samples[pixel] = learned::dequantizeTerrainSample(quantized);
        }
        return grid;
    }

    void prewarmPreviewPageGroup(std::span<const TerrainPageKey> keys) {
        std::vector<WindowIndex> requiredLatents;
        for (const TerrainPageKey key : keys) {
            const int64_t rowBegin = key.coordinate.row * learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const int64_t columnBegin = key.coordinate.column * learned::AUTHORITY_PAGE_NATIVE_EDGE;
            constexpr int64_t OUTPUT_APRON = LATENT_COMPRESSION;
            constexpr int64_t PADDING_HIGH = 6 * LATENT_COMPRESSION;
            const int64_t paddedRowBegin =
                learned::floorDivide(rowBegin - OUTPUT_APRON - PADDING_HIGH, LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const int64_t paddedColumnBegin =
                learned::floorDivide(columnBegin - OUTPUT_APRON - PADDING_HIGH,
                                     LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const int64_t paddedRowEnd = ceilDivide(rowBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE +
                                                        OUTPUT_APRON + PADDING_HIGH,
                                                    LATENT_COMPRESSION) *
                                         LATENT_COMPRESSION;
            const int64_t paddedColumnEnd =
                ceilDivide(columnBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE + OUTPUT_APRON +
                               PADDING_HIGH,
                           LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const NativeRect latentRegion{
                .rowBegin = paddedRowBegin / LATENT_COMPRESSION,
                .columnBegin = paddedColumnBegin / LATENT_COMPRESSION,
                .rowEnd = paddedRowEnd / LATENT_COMPRESSION,
                .columnEnd = paddedColumnEnd / LATENT_COMPRESSION,
            };
            const std::vector<WindowIndex> windows =
                learned::intersectingWindows(latentRegion, learned::LATENT_WINDOW);
            requiredLatents.insert(requiredLatents.end(), windows.begin(), windows.end());
        }
        std::sort(requiredLatents.begin(), requiredLatents.end());
        requiredLatents.erase(std::unique(requiredLatents.begin(), requiredLatents.end()),
                              requiredLatents.end());
        ensureLatentWindows(requiredLatents);
    }

    void prewarmFinalPageGroup(std::span<const TerrainPageKey> keys) {
        std::vector<WindowIndex> decoderWindows;
        for (const TerrainPageKey key : keys) {
            const int64_t rowBegin = key.coordinate.row * learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const int64_t columnBegin = key.coordinate.column * learned::AUTHORITY_PAGE_NATIVE_EDGE;
            const NativeRect region{.rowBegin = rowBegin,
                                    .columnBegin = columnBegin,
                                    .rowEnd = rowBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE,
                                    .columnEnd = columnBegin + learned::AUTHORITY_PAGE_NATIVE_EDGE};
            // Match inferFinalFields exactly. Its output apron is applied
            // before the reconstruction stencil, so omitting it here leaves
            // edge windows to later single-page batches and wastes the fixed
            // batch-of-four model shape.
            constexpr int64_t OUTPUT_APRON = LATENT_COMPRESSION;
            constexpr int PADDING_LOW = 6;
            constexpr int64_t PADDING_HIGH = PADDING_LOW * LATENT_COMPRESSION;
            const int64_t paddedRowBegin =
                learned::floorDivide(region.rowBegin - OUTPUT_APRON - PADDING_HIGH,
                                     LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const int64_t paddedColumnBegin =
                learned::floorDivide(region.columnBegin - OUTPUT_APRON - PADDING_HIGH,
                                     LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const int64_t paddedRowEnd =
                ceilDivide(region.rowEnd + OUTPUT_APRON + PADDING_HIGH, LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const int64_t paddedColumnEnd =
                ceilDivide(region.columnEnd + OUTPUT_APRON + PADDING_HIGH, LATENT_COMPRESSION) *
                LATENT_COMPRESSION;
            const NativeRect padded{.rowBegin = paddedRowBegin,
                                    .columnBegin = paddedColumnBegin,
                                    .rowEnd = paddedRowEnd,
                                    .columnEnd = paddedColumnEnd};
            const std::vector<WindowIndex> windows =
                learned::intersectingWindows(padded, learned::DECODER_WINDOW);
            decoderWindows.insert(decoderWindows.end(), windows.begin(), windows.end());
        }
        std::sort(decoderWindows.begin(), decoderWindows.end());
        decoderWindows.erase(std::unique(decoderWindows.begin(), decoderWindows.end()),
                             decoderWindows.end());

        ensureDecoderWindows(decoderWindows);
    }

    bool hasWindow(WindowKey key) const { return windows_.contains(key); }

    const std::vector<float>* findWindow(WindowKey key) {
        const auto found = windows_.find(key);
        if (found == windows_.end()) return nullptr;
        found->second.lastAccess = ++cacheClock_;
        return &found->second.values;
    }

    const std::vector<float>& insertWindow(WindowKey key, std::vector<float> values) {
        auto existing = windows_.find(key);
        if (existing != windows_.end()) return existing->second.values;
        // Observability only: one instant per actual window computation. A
        // window that reappears here after eviction is a recompute, which the
        // summarizer counts as a duplicate model window (see common/trace.hpp).
        const trace::Name windowName =
            key.stage == WindowKey::Stage::Coarse    ? trace::Name::ModelCoarse
            : key.stage == WindowKey::Stage::Decoder ? trace::Name::ModelDecoder
                                                     : trace::Name::ModelBase;
        trace::instant(
            trace::Track::ModelWindow, windowName,
            {.spatialKey = trace::packCoord(key.row, key.column, static_cast<uint8_t>(key.stage))});
        cacheBytes_ += values.size() * sizeof(float);
        auto [inserted, didInsert] = windows_.emplace(
            key, CachedWindow{.values = std::move(values), .lastAccess = ++cacheClock_});
        (void)didInsert;
        while (cacheBytes_ > TENSOR_WINDOW_CACHE_BUDGET && windows_.size() > 1) {
            auto oldest = windows_.begin();
            for (auto candidate = windows_.begin(); candidate != windows_.end(); ++candidate) {
                if (candidate == inserted) continue;
                if (oldest == inserted || candidate->second.lastAccess < oldest->second.lastAccess)
                    oldest = candidate;
            }
            cacheBytes_ -= oldest->second.values.size() * sizeof(float);
            windows_.erase(oldest);
        }
        return inserted->second.values;
    }

    uint64_t seed_;
    std::filesystem::path modelPack_;
    std::shared_ptr<TerrainModelExecutor> executor_;
    std::optional<PipelineData> pipelineData_;
    std::string initializationError_;
    std::array<std::unique_ptr<FastPerlin>, 5> syntheticNoise_;
    std::mutex mutex_;
    std::map<WindowKey, CachedWindow> windows_;
    size_t cacheBytes_ = 0;
    uint64_t cacheClock_ = 0;
    TerrainRuntimeInferencePhase activePhase_ = TerrainRuntimeInferencePhase::Other;
};

} // namespace

std::array<float, 21> infiniteDiffusionEdmSigmaSchedule() noexcept {
    return EdmScheduler{}.sigmas();
}

learned::QuantizedTerrainSample quantizeInfiniteDiffusionSample(
    float elevationMeters, float meanTemperatureC, float temperatureVariabilityCentidegrees,
    float annualPrecipitationMillimeters, float precipitationCoefficientPercent,
    float lapseRateCPerMeter) noexcept {
    return {
        .elevationMeters = roundedClamp<int16_t>(elevationMeters),
        .meanTemperatureCentidegrees =
            roundedClamp<int16_t>(static_cast<double>(meanTemperatureC) * 100.0),
        .temperatureVariabilityCentidegrees =
            roundedClamp<uint16_t>(std::abs(temperatureVariabilityCentidegrees)),
        .annualPrecipitationMillimeters =
            roundedClamp<uint16_t>(std::max(0.0F, annualPrecipitationMillimeters)),
        .precipitationCoefficientBasisPoints = roundedClamp<uint16_t>(
            static_cast<double>(std::abs(precipitationCoefficientPercent)) * 100.0),
        .lapseRateMicrodegreesPerMeter =
            roundedClamp<int16_t>(static_cast<double>(lapseRateCPerMeter) * 1'000'000.0),
    };
}

std::array<float, INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS>
interpolateInfiniteDiffusionCoarseConditioning(
    std::span<const float, INFINITE_DIFFUSION_COARSE_CONDITIONING_VALUES> coarseFields,
    int64_t nativeRow, int64_t nativeColumn, int64_t coarseRow, int64_t coarseColumn) noexcept {
    constexpr float NATIVE_PIXELS_PER_COARSE_SAMPLE = 256.0F;
    constexpr size_t COARSE_AREA =
        INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE * INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE;
    const float sampleRow =
        (static_cast<float>(nativeRow) + 0.5F) / NATIVE_PIXELS_PER_COARSE_SAMPLE -
        static_cast<float>(coarseRow) + 0.5F;
    const float sampleColumn =
        (static_cast<float>(nativeColumn) + 0.5F) / NATIVE_PIXELS_PER_COARSE_SAMPLE -
        static_cast<float>(coarseColumn) + 0.5F;
    const float boundedRow = std::clamp(
        sampleRow, 0.0F, static_cast<float>(INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE - 1));
    const float boundedColumn = std::clamp(
        sampleColumn, 0.0F, static_cast<float>(INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE - 1));
    const size_t row0 = static_cast<size_t>(boundedRow);
    const size_t row1 = std::min(INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE - 1, row0 + 1);
    const size_t column0 = static_cast<size_t>(boundedColumn);
    const size_t column1 = std::min(INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE - 1, column0 + 1);
    const float rowWeight = boundedRow - static_cast<float>(row0);
    const float columnWeight = boundedColumn - static_cast<float>(column0);

    std::array<float, INFINITE_DIFFUSION_COARSE_CONDITIONING_CHANNELS> sample{};
    for (size_t channel = 0; channel < sample.size(); ++channel) {
        const size_t offset = channel * COARSE_AREA;
        const auto at = [&](size_t row, size_t column) {
            return coarseFields[offset + row * INFINITE_DIFFUSION_COARSE_CONDITIONING_EDGE +
                                column];
        };
        sample[channel] = (1.0F - rowWeight) * (1.0F - columnWeight) * at(row0, column0) +
                          (1.0F - rowWeight) * columnWeight * at(row0, column1) +
                          rowWeight * (1.0F - columnWeight) * at(row1, column0) +
                          rowWeight * columnWeight * at(row1, column1);
    }
    return sample;
}

InfiniteDiffusionLaplacianResult
runInfiniteDiffusionLaplacian(std::span<const float> residual, int residualHeight,
                              int residualWidth, std::span<const float> lowFrequency,
                              int lowFrequencyHeight, int lowFrequencyWidth, float sigma) {
    if (residualHeight <= 0 || residualWidth <= 0 || lowFrequencyHeight <= 0 ||
        lowFrequencyWidth <= 0 || !std::isfinite(sigma) || sigma <= 0.0F ||
        residual.size() != static_cast<size_t>(residualHeight) * residualWidth ||
        lowFrequency.size() != static_cast<size_t>(lowFrequencyHeight) * lowFrequencyWidth) {
        throw std::invalid_argument("Invalid InfiniteDiffusion Laplacian diagnostic input");
    }
    Matrix residualMatrix(residualHeight, residualWidth);
    std::copy(residual.begin(), residual.end(), residualMatrix.values.begin());
    Matrix lowFrequencyMatrix(lowFrequencyHeight, lowFrequencyWidth);
    std::copy(lowFrequency.begin(), lowFrequency.end(), lowFrequencyMatrix.values.begin());
    Matrix cleaned = laplacianDenoise(residualMatrix, lowFrequencyMatrix, sigma);
    Matrix decoded = laplacianDecode(residualMatrix, cleaned);
    return {.lowFrequency = std::move(cleaned.values), .decoded = std::move(decoded.values)};
}

std::shared_ptr<learned::TerrainInferenceBackend>
makeInfiniteDiffusionTerrainBackend(uint64_t seed, std::filesystem::path modelPack,
                                    std::shared_ptr<TerrainModelExecutor> executor) {
    return std::make_shared<InfiniteDiffusionBackend>(seed, std::move(modelPack),
                                                      std::move(executor));
}

} // namespace worldgen::runtime
