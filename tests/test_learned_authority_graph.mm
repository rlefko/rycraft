#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <world/learned_authority_graph.hpp>

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Learned-authority request-graph unit tests (issue #16). These are pure: the
// planner is a scheduling, dedup, and retention layer, so its window closure,
// unique-window cost gate, retention bound, and typed failure modes are all
// verifiable without a model. A separate cross-check in test_terrain_runtime.mm
// proves the closure equals the windows the real backend actually computes.
// ---------------------------------------------------------------------------

using namespace worldgen::learned;

namespace {

NativeRect pageRect(int64_t pageRow, int64_t pageColumn) {
    const int64_t edge = AUTHORITY_PAGE_NATIVE_EDGE;
    return {.rowBegin = pageRow * edge,
            .columnBegin = pageColumn * edge,
            .rowEnd = pageRow * edge + edge,
            .columnEnd = pageColumn * edge + edge};
}

LearnedAuthorityRequest finalPage(int64_t pageRow, int64_t pageColumn) {
    return {.quality = AuthorityQuality::FINAL, .region = pageRect(pageRow, pageColumn)};
}

} // namespace

TEST_CASE("Window closure is independent of request order", "[learned][window-plan]") {
    const std::vector<LearnedAuthorityRequest> forward = {finalPage(0, 0), finalPage(0, 1),
                                                          finalPage(1, 0)};
    std::vector<LearnedAuthorityRequest> reverse(forward.rbegin(), forward.rend());

    const std::vector<LearnedWindowRef> a = learnedWindowClosure(forward);
    const std::vector<LearnedWindowRef> b = learnedWindowClosure(reverse);
    REQUIRE(a == b);
    // The closure is sorted and free of duplicates.
    CHECK(std::is_sorted(a.begin(), a.end()));
    CHECK(std::adjacent_find(a.begin(), a.end()) == a.end());
}

TEST_CASE("Overlapping and duplicate requests do not inflate the window set",
          "[learned][window-plan]") {
    const std::vector<LearnedAuthorityRequest> single = {finalPage(0, 0)};
    const std::vector<LearnedAuthorityRequest> duplicated = {finalPage(0, 0), finalPage(0, 0)};
    CHECK(learnedWindowClosure(single).size() == learnedWindowClosure(duplicated).size());

    // Two adjacent pages share apron windows, so their combined closure is
    // strictly smaller than preparing each independently.
    const size_t adjacentTogether = learnedWindowClosure({finalPage(0, 0), finalPage(0, 1)}).size();
    const size_t separately = learnedWindowClosure({finalPage(0, 0)}).size() +
                              learnedWindowClosure({finalPage(0, 1)}).size();
    CHECK(adjacentTogether < separately);
}

TEST_CASE("FINAL closures carry decoder windows and PREVIEW closures do not",
          "[learned][window-plan]") {
    const std::vector<LearnedWindowRef> finalClosure = learnedWindowClosure({finalPage(0, 0)});
    const std::vector<LearnedWindowRef> previewClosure =
        learnedWindowClosure({{.quality = AuthorityQuality::PREVIEW, .region = pageRect(0, 0)}});

    const auto hasDecoder = [](const std::vector<LearnedWindowRef>& windows) {
        return std::any_of(windows.begin(), windows.end(), [](const LearnedWindowRef& window) {
            return window.stage == LearnedWindowStage::Decoder;
        });
    };
    CHECK(hasDecoder(finalClosure));
    CHECK_FALSE(hasDecoder(previewClosure));
    // Both carry coarse and both latent phases.
    const auto stageCount = [](const std::vector<LearnedWindowRef>& windows, LearnedWindowStage s) {
        return std::count_if(windows.begin(), windows.end(),
                             [s](const LearnedWindowRef& w) { return w.stage == s; });
    };
    CHECK(stageCount(finalClosure, LearnedWindowStage::Coarse) > 0);
    CHECK(stageCount(finalClosure, LearnedWindowStage::LatentInitial) ==
          stageCount(finalClosure, LearnedWindowStage::LatentFinal));
    CHECK(stageCount(previewClosure, LearnedWindowStage::LatentFinal) > 0);
}

TEST_CASE("Grouping is preferred only when it reduces unique windows", "[learned][window-plan]") {
    // Adjacent owners share aprons, so grouping their bounding box is cheaper.
    const NativeRect a = pageRect(0, 0);
    const NativeRect b = pageRect(0, 1);
    const NativeRect adjacentBox{.rowBegin = a.rowBegin,
                                 .columnBegin = a.columnBegin,
                                 .rowEnd = b.rowEnd,
                                 .columnEnd = b.columnEnd};
    const NativeRect adjacentMembers[] = {a, b};
    CHECK(learnedPreferGroupedRegion(adjacentBox, adjacentMembers));

    // Owners far apart on both axes: the bounding box spans the gap and adds
    // windows no member needs, so grouping is rejected.
    const NativeRect near = pageRect(0, 0);
    const NativeRect far = pageRect(6, 6);
    const NativeRect sparseBox{.rowBegin = near.rowBegin,
                               .columnBegin = near.columnBegin,
                               .rowEnd = far.rowEnd,
                               .columnEnd = far.columnEnd};
    const NativeRect sparseMembers[] = {near, far};
    CHECK_FALSE(learnedPreferGroupedRegion(sparseBox, sparseMembers));
}

TEST_CASE("Retention bytes account for every stage", "[learned][window-plan]") {
    const std::vector<LearnedWindowRef> closure = learnedWindowClosure({finalPage(0, 0)});
    const size_t bytes = learnedRetentionBytes(closure);

    size_t expected = 0;
    for (const LearnedWindowRef& window : closure) {
        switch (window.stage) {
            case LearnedWindowStage::Coarse:
                expected += 7ULL * 64 * 64 * sizeof(float);
                break;
            case LearnedWindowStage::LatentInitial:
            case LearnedWindowStage::LatentFinal:
                expected += 6ULL * 64 * 64 * sizeof(float);
                break;
            case LearnedWindowStage::Decoder:
                expected += 2ULL * 256 * 256 * sizeof(float);
                break;
        }
    }
    CHECK(bytes == expected);
    CHECK(bytes > 0);
}

TEST_CASE("build reports the retention bound and rejects out-of-bounds requests",
          "[learned][window-plan]") {
    const std::vector<LearnedAuthorityRequest> requests = {finalPage(0, 0), finalPage(0, 1)};
    const auto plan = LearnedAuthorityGraph::build(requests);
    REQUIRE(plan.isReady());
    const std::shared_ptr<const LearnedAuthorityGraph> graph = *plan.value();
    CHECK(graph->uniqueWindowCount() == learnedWindowClosure(requests).size());
    CHECK(graph->retentionByteBound() == learnedRetentionBytes(graph->retainedWindows()));

    // A rectangle larger than the sample bound is a typed INVALID_REQUEST.
    const std::vector<LearnedAuthorityRequest> oversized = {
        {.quality = AuthorityQuality::FINAL,
         .region = {.rowBegin = 0, .columnBegin = 0, .rowEnd = 2048, .columnEnd = 2048}}};
    const auto oversizedPlan = LearnedAuthorityGraph::build(oversized);
    CHECK(oversizedPlan.status() == AuthorityStatus::FAILED);
    REQUIRE(oversizedPlan.failure() != nullptr);
    CHECK(oversizedPlan.failure()->code == GenerationFailureCode::INVALID_REQUEST);

    // A working set larger than the tensor-window budget also fails typed.
    LearnedWindowPlanConfig tinyBudget;
    tinyBudget.tensorWindowByteBudget = 1;
    const auto budgetPlan = LearnedAuthorityGraph::build(requests, tinyBudget);
    CHECK(budgetPlan.status() == AuthorityStatus::FAILED);
    REQUIRE(budgetPlan.failure() != nullptr);
    CHECK(budgetPlan.failure()->code == GenerationFailureCode::INVALID_REQUEST);
}

TEST_CASE("Window closure is correct at negative coordinates", "[learned][window-plan][negative]") {
    const std::vector<LearnedAuthorityRequest> forward = {finalPage(-3, -2), finalPage(-3, -1)};
    std::vector<LearnedAuthorityRequest> reverse(forward.rbegin(), forward.rend());
    const std::vector<LearnedWindowRef> a = learnedWindowClosure(forward);
    CHECK(a == learnedWindowClosure(reverse));
    CHECK(std::is_sorted(a.begin(), a.end()));
    // Adjacent negative-coordinate owners still benefit from grouping.
    const size_t together = a.size();
    const size_t separate = learnedWindowClosure({finalPage(-3, -2)}).size() +
                            learnedWindowClosure({finalPage(-3, -1)}).size();
    CHECK(together < separate);
}
