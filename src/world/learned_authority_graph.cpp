#include "world/learned_authority_graph.hpp"

#include <algorithm>
#include <set>

// ---------------------------------------------------------------------------
// world/learned_authority_graph implementation. The enumeration below mirrors
// InfiniteDiffusionBackend's window flow exactly:
//
//   FINAL output R -> expand by OUTPUT_APRON -> pad to the latent lattice
//     -> Decoder windows over the padded rect
//     -> Latent windows = padded/8 latent rect  UNION  each decoder window's
//        {row*24, col*24, +32} latent dependency
//     -> Coarse windows = each latent's {row-1, col-1, row+3, col+3} region
//   PREVIEW output R -> same padding, Latent = padded/8 rect, no Decoder.
//
// The per-window byte sizes match the backend's cached tensor sizes
// (channels * edge^2 floats). A test cross-checks this closure against the
// windows the backend actually computes, recorded through the trace.
// ---------------------------------------------------------------------------

namespace worldgen::learned {
namespace {

// Mirrors the backend constants (infinite_diffusion_backend.cpp:42-52).
constexpr int64_t LATENT_COMPRESSION = 8;
constexpr int64_t OUTPUT_APRON = LATENT_COMPRESSION;
constexpr int64_t PADDING_HIGH = 6 * LATENT_COMPRESSION; // PADDING_LOW=6
constexpr int64_t DECODER_STRIDE = 192;
constexpr int64_t DECODER_EDGE = 256;

// Cached tensor floats per window: channels * edge^2 (7/6/2 channels).
constexpr size_t COARSE_WINDOW_FLOATS = 7ULL * 64 * 64;
constexpr size_t LATENT_WINDOW_FLOATS = 6ULL * 64 * 64;
constexpr size_t DECODER_WINDOW_FLOATS = 2ULL * 256 * 256;

int64_t ceilDivide(int64_t value, int64_t divisor) {
    return -floorDivide(-value, divisor);
}

NativeRect expandByApron(NativeRect region, int64_t apron) {
    return {.rowBegin = region.rowBegin - apron,
            .columnBegin = region.columnBegin - apron,
            .rowEnd = region.rowEnd + apron,
            .columnEnd = region.columnEnd + apron};
}

// The padded decoder region for an already-apron-expanded output rect, matching
// inferFinalFieldsWithBoundary / inferPreviewFieldsWithBoundary.
NativeRect paddedDecoderRegion(NativeRect expanded) {
    return {
        .rowBegin =
            floorDivide(expanded.rowBegin - PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION,
        .columnBegin = floorDivide(expanded.columnBegin - PADDING_HIGH, LATENT_COMPRESSION) *
                       LATENT_COMPRESSION,
        .rowEnd =
            ceilDivide(expanded.rowEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION,
        .columnEnd =
            ceilDivide(expanded.columnEnd + PADDING_HIGH, LATENT_COMPRESSION) * LATENT_COMPRESSION,
    };
}

NativeRect latentRegionFor(NativeRect padded) {
    return {.rowBegin = padded.rowBegin / LATENT_COMPRESSION,
            .columnBegin = padded.columnBegin / LATENT_COMPRESSION,
            .rowEnd = padded.rowEnd / LATENT_COMPRESSION,
            .columnEnd = padded.columnEnd / LATENT_COMPRESSION};
}

} // namespace

std::vector<WindowIndex> learnedDecoderWindows(NativeRect outputRegion) {
    const NativeRect padded = paddedDecoderRegion(expandByApron(outputRegion, OUTPUT_APRON));
    return intersectingWindows(padded, DECODER_WINDOW);
}

std::vector<WindowIndex> learnedLatentWindows(NativeRect outputRegion, AuthorityQuality quality) {
    const NativeRect padded = paddedDecoderRegion(expandByApron(outputRegion, OUTPUT_APRON));
    std::set<WindowIndex> latents;
    for (const WindowIndex window : intersectingWindows(latentRegionFor(padded), LATENT_WINDOW)) {
        latents.insert(window);
    }
    if (quality == AuthorityQuality::FINAL) {
        for (const WindowIndex decoder : intersectingWindows(padded, DECODER_WINDOW)) {
            const int64_t latentEdge = DECODER_EDGE / LATENT_COMPRESSION;
            const int64_t latentRow = decoder.row * DECODER_STRIDE / LATENT_COMPRESSION;
            const int64_t latentColumn = decoder.column * DECODER_STRIDE / LATENT_COMPRESSION;
            const NativeRect dependency{.rowBegin = latentRow,
                                        .columnBegin = latentColumn,
                                        .rowEnd = latentRow + latentEdge,
                                        .columnEnd = latentColumn + latentEdge};
            for (const WindowIndex window : intersectingWindows(dependency, LATENT_WINDOW)) {
                latents.insert(window);
            }
        }
    }
    return {latents.begin(), latents.end()};
}

std::vector<WindowIndex> learnedCoarseWindows(std::span<const WindowIndex> latentWindows) {
    std::set<WindowIndex> coarse;
    for (const WindowIndex latent : latentWindows) {
        const NativeRect region{.rowBegin = latent.row - 1,
                                .columnBegin = latent.column - 1,
                                .rowEnd = latent.row + 3,
                                .columnEnd = latent.column + 3};
        for (const WindowIndex window : intersectingWindows(region, COARSE_WINDOW)) {
            coarse.insert(window);
        }
    }
    return {coarse.begin(), coarse.end()};
}

std::vector<LearnedWindowRef>
learnedWindowClosure(std::span<const LearnedAuthorityRequest> requests) {
    std::set<LearnedWindowRef> refs;
    for (const LearnedAuthorityRequest& request : requests) {
        if (request.quality == AuthorityQuality::FINAL) {
            for (const WindowIndex decoder : learnedDecoderWindows(request.region)) {
                refs.insert({LearnedWindowStage::Decoder, decoder});
            }
        }
        const std::vector<WindowIndex> latents =
            learnedLatentWindows(request.region, request.quality);
        for (const WindowIndex latent : latents) {
            refs.insert({LearnedWindowStage::LatentInitial, latent});
            refs.insert({LearnedWindowStage::LatentFinal, latent});
        }
        for (const WindowIndex coarse : learnedCoarseWindows(latents)) {
            refs.insert({LearnedWindowStage::Coarse, coarse});
        }
    }
    return {refs.begin(), refs.end()};
}

size_t learnedRetentionBytes(std::span<const LearnedWindowRef> windows) {
    size_t bytes = 0;
    for (const LearnedWindowRef& window : windows) {
        switch (window.stage) {
            case LearnedWindowStage::Coarse:
                bytes += COARSE_WINDOW_FLOATS * sizeof(float);
                break;
            case LearnedWindowStage::LatentInitial:
            case LearnedWindowStage::LatentFinal:
                bytes += LATENT_WINDOW_FLOATS * sizeof(float);
                break;
            case LearnedWindowStage::Decoder:
                bytes += DECODER_WINDOW_FLOATS * sizeof(float);
                break;
        }
    }
    return bytes;
}

size_t learnedUniqueWindowCost(std::span<const NativeRect> finalRegions) {
    std::vector<LearnedAuthorityRequest> requests;
    requests.reserve(finalRegions.size());
    for (const NativeRect region : finalRegions) {
        requests.push_back({.quality = AuthorityQuality::FINAL, .region = region});
    }
    return learnedWindowClosure(requests).size();
}

bool learnedPreferGroupedRegion(NativeRect grouped, std::span<const NativeRect> members,
                                size_t maximumQuerySamples) {
    if (!grouped.valid()) {
        return false;
    }
    if (grouped.height() * grouped.width() > maximumQuerySamples) {
        return false;
    }
    // The independent baseline is the sum of each owner prepared on its own,
    // where seam windows are recomputed per request. Grouping wins only when the
    // combined rectangle's unique windows are fewer than that sum, so a bounding
    // box that merely spans a gap is rejected.
    const NativeRect groupedRegions[] = {grouped};
    const size_t groupedCost = learnedUniqueWindowCost(groupedRegions);
    size_t independentCost = 0;
    for (const NativeRect member : members) {
        const NativeRect single[] = {member};
        independentCost += learnedUniqueWindowCost(single);
    }
    return groupedCost < independentCost;
}

AuthorityResult<std::shared_ptr<const LearnedAuthorityGraph>>
LearnedAuthorityGraph::build(std::span<const LearnedAuthorityRequest> requests,
                             LearnedWindowPlanConfig config) {
    using Result = AuthorityResult<std::shared_ptr<const LearnedAuthorityGraph>>;
    for (const LearnedAuthorityRequest& request : requests) {
        if (!request.region.valid()) {
            return Result::failed({.code = GenerationFailureCode::INVALID_REQUEST,
                                   .message = "Authority request rectangle is empty",
                                   .retriable = false});
        }
        if (request.region.height() * request.region.width() > config.maximumQuerySamples) {
            return Result::failed({.code = GenerationFailureCode::INVALID_REQUEST,
                                   .message = "Authority request exceeds the sample bound",
                                   .retriable = false});
        }
    }
    auto graph = std::make_shared<LearnedAuthorityGraph>();
    graph->retained_ = learnedWindowClosure(requests);
    graph->retentionByteBound_ = learnedRetentionBytes(graph->retained_);
    if (graph->retentionByteBound_ > config.tensorWindowByteBudget) {
        return Result::failed({.code = GenerationFailureCode::INVALID_REQUEST,
                               .message = "Authority working set exceeds the tensor-window budget",
                               .retriable = false});
    }
    return Result::ready(std::move(graph));
}

} // namespace worldgen::learned
