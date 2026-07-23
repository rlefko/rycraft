#pragma once

#include <functional>

struct ApplicationTerminationActions {
    std::function<bool()> saveDurableState;
    std::function<void()> cancelBootstrap;
    std::function<void()> stopRenderWorkers;
    std::function<void()> stopWorldAndGenerationWorkers;
    std::function<void()> releaseGenerationOwners;
    std::function<void()> releaseRuntime;
};

// AppKit terminates through exit and does not release the shared Engine
// singleton. This explicit sequence preserves dependency order and makes
// every quit route safe to call more than once. A failed durable save is the
// only cancelable boundary; teardown begins only after it succeeds.
class ApplicationTerminationQuiescence {
public:
    [[nodiscard]] bool quiesce(const ApplicationTerminationActions& actions,
                               bool requireDurableSave = true) {
        if (quiesced_) return true;
        if (!persistenceResolved_) {
            if (requireDurableSave && !actions.saveDurableState()) return false;
            persistenceResolved_ = true;
        }

        actions.cancelBootstrap();
        actions.stopRenderWorkers();
        actions.stopWorldAndGenerationWorkers();
        actions.releaseGenerationOwners();
        actions.releaseRuntime();
        quiesced_ = true;
        return true;
    }

    void resetForWorldSession() noexcept {
        persistenceResolved_ = false;
        quiesced_ = false;
    }

    [[nodiscard]] bool persistenceResolved() const noexcept { return persistenceResolved_; }
    [[nodiscard]] bool quiesced() const noexcept { return quiesced_; }

private:
    bool persistenceResolved_ = false;
    bool quiesced_ = false;
};
