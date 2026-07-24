#pragma once

#include "world/learned_terrain.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

// ---------------------------------------------------------------------------
// world/learned_authority_graph — one immutable dependency graph for learned
// authority (generator v4, issue #16).
//
// The planner replaces independent, overlapping page and protected-owner
// requests with a single immutable plan of the unique coarse, Base (latent),
// and Decoder model windows they need. It computes the working-set retention
// bound before inference begins so the backend can pin the closure and never
// recompute or evict a window that the active protected graph still references.
//
// This is a scheduling, dedup, and retention layer only. It changes which
// windows are grouped and retained, never how a window is computed. The window
// math (scalar Coarse, fixed four-window Base and Decoder batches, deterministic
// repeated-tail padding, lexicographic accumulation) stays in
// InfiniteDiffusionBackend, so authority bytes and qualification hashes are
// unchanged. The enumeration below mirrors that backend exactly and is
// cross-checked against the windows the backend actually computes.
// ---------------------------------------------------------------------------

namespace worldgen::learned {

// Mirrors InfiniteDiffusionBackend's private WindowKey::Stage ordering so a
// packed (row, column, stage) key matches the backend's trace records.
enum class LearnedWindowStage : uint8_t {
    Coarse = 0,
    LatentInitial = 1,
    LatentFinal = 2,
    Decoder = 3,
};

// One unique model window in the plan.
struct LearnedWindowRef {
    LearnedWindowStage stage = LearnedWindowStage::Coarse;
    WindowIndex index;

    auto operator<=>(const LearnedWindowRef&) const = default;
};

// One requested output. region is the unexpanded native output rectangle: a
// 256 by 256 authority page or a protected hydrology-owner rectangle.
struct LearnedAuthorityRequest {
    AuthorityQuality quality = AuthorityQuality::FINAL;
    NativeRect region;
};

struct LearnedWindowPlanConfig {
    size_t tensorWindowByteBudget = 384ULL * 1024 * 1024;
    size_t maximumQuerySamples = MAXIMUM_AUTHORITY_QUERY_SAMPLES;
};

// -- Pure enumeration helpers (single source of truth for planned windows) ---
// Each returns the exact window set InfiniteDiffusionBackend computes for the
// output rectangle, deduplicated and lexicographically sorted.
std::vector<WindowIndex> learnedDecoderWindows(NativeRect outputRegion);
std::vector<WindowIndex> learnedLatentWindows(NativeRect outputRegion, AuthorityQuality quality);
std::vector<WindowIndex> learnedCoarseWindows(std::span<const WindowIndex> latentWindows);

// The complete deduplicated window closure a request set requires.
std::vector<LearnedWindowRef>
learnedWindowClosure(std::span<const LearnedAuthorityRequest> requests);
inline std::vector<LearnedWindowRef>
learnedWindowClosure(std::initializer_list<LearnedAuthorityRequest> requests) {
    return learnedWindowClosure(
        std::span<const LearnedAuthorityRequest>(requests.begin(), requests.size()));
}

// Retained-window bytes for a closure, matching the backend's tensor-window
// cache accounting (value floats times sizeof(float)).
size_t learnedRetentionBytes(std::span<const LearnedWindowRef> windows);

// The number of unique model windows covering a set of FINAL output rectangles.
size_t learnedUniqueWindowCost(std::span<const NativeRect> finalRegions);

// Group only when the grouped rectangle stays within the sample bound AND costs
// strictly fewer unique windows than preparing the members independently. Union
// by bounding box alone is not sufficient.
bool learnedPreferGroupedRegion(NativeRect grouped, std::span<const NativeRect> members,
                                size_t maximumQuerySamples = MAXIMUM_AUTHORITY_QUERY_SAMPLES);

// The immutable plan. Built once, never mutated.
class LearnedAuthorityGraph {
public:
    static AuthorityResult<std::shared_ptr<const LearnedAuthorityGraph>>
    build(std::span<const LearnedAuthorityRequest> requests, LearnedWindowPlanConfig config = {});

    [[nodiscard]] std::span<const LearnedWindowRef> retainedWindows() const noexcept {
        return retained_;
    }
    [[nodiscard]] size_t uniqueWindowCount() const noexcept { return retained_.size(); }
    [[nodiscard]] size_t retentionByteBound() const noexcept { return retentionByteBound_; }

private:
    std::vector<LearnedWindowRef> retained_;
    size_t retentionByteBound_ = 0;
};

} // namespace worldgen::learned
