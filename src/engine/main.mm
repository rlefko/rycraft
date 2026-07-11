#import <engine/engine.hpp>

#include <common/error.hpp>

// ---------------------------------------------------------------------------
// Entry point — Initialize and run the game engine
// ---------------------------------------------------------------------------
int main() {
    Engine* engine = [Engine sharedEngine];

    if (![engine initialize]) {
        RY_LOG_FATAL("Engine initialization failed");
        return 1;
    }

    [engine run];

    return 0;
}
