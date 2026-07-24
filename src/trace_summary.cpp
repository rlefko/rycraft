#include "common/trace.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// rycraft_trace_summary: the checked-in Program 0 trace summarizer.
//
// It reads one binary trace written by trace::writeBinary (RYCRAFT_TRACE) and
// reports p50/p95/max, queue depth, cache reuse, cancellation, critical-path
// ownership, duplicate model windows, and omitted water/flora products. The
// summary math lives in trace::summarize (libcommon) so this tool, the game,
// and tests/test_trace.mm all compute identically.
// ---------------------------------------------------------------------------

namespace {

void usage() {
    std::fprintf(stderr, "usage: rycraft_trace_summary <trace.rytrace> [--json] [--route NAME]\n"
                         "  NAME is one of cold warm move hover reversal lod settle\n");
}

double ms(uint64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

void printText(const trace::Summary& summary) {
    std::printf("route=%s events=%llu dropped=%llu\n", trace::routeString(summary.route),
                static_cast<unsigned long long>(summary.eventCount),
                static_cast<unsigned long long>(summary.droppedEvents));
    std::printf("%-18s %8s %10s %10s %10s %10s\n", "track", "count", "p50ms", "p95ms", "maxms",
                "queueMax");
    for (const trace::TrackSummary& t : summary.tracks) {
        if (t.count == 0 && t.maxQueueDepth == 0) {
            continue;
        }
        std::printf("%-18s %8llu %10.4f %10.4f %10.4f %10llu\n", trace::trackString(t.track),
                    static_cast<unsigned long long>(t.count), ms(t.p50Ns), ms(t.p95Ns), ms(t.maxNs),
                    static_cast<unsigned long long>(t.maxQueueDepth));
    }
    const uint64_t lookups = summary.cacheHits + summary.cacheMisses;
    const double reuse =
        lookups > 0 ? static_cast<double>(summary.cacheHits) / static_cast<double>(lookups) : 0.0;
    std::printf("cacheHits=%llu cacheMisses=%llu evictions=%llu reuse=%.3f\n",
                static_cast<unsigned long long>(summary.cacheHits),
                static_cast<unsigned long long>(summary.cacheMisses),
                static_cast<unsigned long long>(summary.cacheEvictions), reuse);
    std::printf("duplicateModelWindows=%llu\n",
                static_cast<unsigned long long>(summary.duplicateModelWindows));
    std::printf("protectedWaitMs=%.4f optionalOwnedProtectedWaitMs=%.4f topOwner=%s\n",
                ms(summary.protectedWaitNs), ms(summary.optionalOwnedProtectedWaitNs),
                trace::trackString(summary.topProtectedWaitOwner));
    std::printf("omittedWater=%llu omittedFlora=%llu\n",
                static_cast<unsigned long long>(summary.omittedWater),
                static_cast<unsigned long long>(summary.omittedFlora));
    static const char* kCancellation[] = {"none",   "completed",     "deferred",
                                          "failed", "canceledEpoch", "canceledPriority"};
    std::printf("cancellation:");
    for (size_t i = 1; i < static_cast<size_t>(trace::Cancellation::Count); ++i) {
        std::printf(" %s=%llu", kCancellation[i],
                    static_cast<unsigned long long>(summary.cancellation[i]));
    }
    std::printf("\n");
}

void printJson(const trace::Summary& summary) {
    std::printf("{\"route\":\"%s\",\"events\":%llu,\"dropped\":%llu,\"tracks\":[",
                trace::routeString(summary.route),
                static_cast<unsigned long long>(summary.eventCount),
                static_cast<unsigned long long>(summary.droppedEvents));
    bool first = true;
    for (const trace::TrackSummary& t : summary.tracks) {
        if (t.count == 0 && t.maxQueueDepth == 0) {
            continue;
        }
        std::printf(
            "%s{\"track\":\"%s\",\"count\":%llu,\"p50Ns\":%llu,\"p95Ns\":%llu,"
            "\"maxNs\":%llu,\"queueMax\":%llu,\"canceled\":%llu}",
            first ? "" : ",", trace::trackString(t.track), static_cast<unsigned long long>(t.count),
            static_cast<unsigned long long>(t.p50Ns), static_cast<unsigned long long>(t.p95Ns),
            static_cast<unsigned long long>(t.maxNs),
            static_cast<unsigned long long>(t.maxQueueDepth),
            static_cast<unsigned long long>(t.canceled));
        first = false;
    }
    std::printf("],\"cacheHits\":%llu,\"cacheMisses\":%llu,\"evictions\":%llu,"
                "\"duplicateModelWindows\":%llu,\"protectedWaitNs\":%llu,"
                "\"optionalOwnedProtectedWaitNs\":%llu,\"topOwner\":\"%s\","
                "\"omittedWater\":%llu,\"omittedFlora\":%llu}\n",
                static_cast<unsigned long long>(summary.cacheHits),
                static_cast<unsigned long long>(summary.cacheMisses),
                static_cast<unsigned long long>(summary.cacheEvictions),
                static_cast<unsigned long long>(summary.duplicateModelWindows),
                static_cast<unsigned long long>(summary.protectedWaitNs),
                static_cast<unsigned long long>(summary.optionalOwnedProtectedWaitNs),
                trace::trackString(summary.topProtectedWaitOwner),
                static_cast<unsigned long long>(summary.omittedWater),
                static_cast<unsigned long long>(summary.omittedFlora));
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    bool json = false;
    bool filterRoute = false;
    trace::RouteTag route = trace::RouteTag::Unknown;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") {
            json = true;
        } else if (arg == "--route" && i + 1 < argc) {
            filterRoute = true;
            route = trace::routeFromName(argv[++i]);
        } else if (!arg.empty() && arg[0] != '-') {
            path = arg;
        } else {
            usage();
            return 2;
        }
    }
    if (path.empty()) {
        usage();
        return 2;
    }

    std::vector<trace::Event> events;
    if (!trace::readBinary(path, events)) {
        std::fprintf(stderr, "rycraft_trace_summary: cannot read trace '%s'\n", path.c_str());
        return 1;
    }

    if (filterRoute) {
        std::vector<trace::Event> filtered;
        filtered.reserve(events.size());
        for (const trace::Event& e : events) {
            if (static_cast<trace::RouteTag>(e.routeTag) == route) {
                filtered.push_back(e);
            }
        }
        events.swap(filtered);
    }

    const trace::Summary summary = trace::summarize(events);
    if (json) {
        printJson(summary);
    } else {
        printText(summary);
    }
    return 0;
}
