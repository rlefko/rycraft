#pragma once

// ---------------------------------------------------------------------------
// Game flow — which screen the player is on and how input transitions it.
//
// Pure C++ (no Cocoa/Metal) so every transition is unit-testable. The engine
// applies the returned effects: cursor capture/release, timing resets, and
// quitting. World simulation ticks only while the screen is Playing; menus
// freeze time.
// ---------------------------------------------------------------------------

enum class GameScreen {
    TITLE,          // launch screen: PLAY / QUIT, world visible behind
    PLAYING,        // normal gameplay, cursor captured
    PAUSED,         // ESC menu: RESUME / SETTINGS / QUIT
    SETTINGS,       // settings panel, reached from PAUSED
    VIDEO_SETTINGS, // per-effect video options, reached from SETTINGS
};

enum class MenuAction {
    NONE,
    PLAY,
    RESUME,
    OPEN_SETTINGS,
    CLOSE_SETTINGS,
    OPEN_VIDEO_SETTINGS,
    CLOSE_VIDEO_SETTINGS,
    QUIT,
    // Settings value steppers (handled by the engine; no screen change)
    VIEW_DISTANCE_DOWN,
    VIEW_DISTANCE_UP,
    FOG_DOWN,
    FOG_UP,
    SENSITIVITY_DOWN,
    SENSITIVITY_UP,
    VOLUME_DOWN,
    VOLUME_UP,
    // Video settings steppers/toggles (both arrows of a toggle row flip it)
    SHADOWS_DOWN,
    SHADOWS_UP,
    VL_TOGGLE,
    CLOUDS_DOWN,
    CLOUDS_UP,
    SSAO_TOGGLE,
    SSR_TOGGLE,
    WAVING_TOGGLE,
    LENS_FLARE_TOGGLE,
    BLOOM_DOWN,
    BLOOM_UP,
    VIBRANCE_DOWN,
    VIBRANCE_UP,
    SHARPEN_DOWN,
    SHARPEN_UP,
};

// What the engine must do after a transition.
struct GameFlowEffects {
    bool captureCursor = false; // hide + pointer-lock the mouse
    bool releaseCursor = false; // unhide + free the mouse
    bool resetTiming = false;   // zero the tick accumulator + mouse delta
    bool requestQuit = false;   // save and terminate
};

struct GameFlow {
    GameScreen screen = GameScreen::TITLE;

    // Menus freeze the world: ticks run only while Playing.
    constexpr bool inMenu() const { return screen != GameScreen::PLAYING; }

    // ESC: pause from gameplay, resume from pause, back out of settings.
    constexpr GameFlowEffects onEscape() {
        switch (screen) {
            case GameScreen::PLAYING:
                screen = GameScreen::PAUSED;
                return {.releaseCursor = true, .resetTiming = true};
            case GameScreen::PAUSED:
                screen = GameScreen::PLAYING;
                return {.captureCursor = true, .resetTiming = true};
            case GameScreen::SETTINGS:
                screen = GameScreen::PAUSED;
                return {};
            case GameScreen::VIDEO_SETTINGS:
                screen = GameScreen::SETTINGS;
                return {};
            case GameScreen::TITLE:
                return {};
        }
        return {};
    }

    constexpr GameFlowEffects onMenuAction(MenuAction action) {
        switch (action) {
            case MenuAction::PLAY:
                if (screen == GameScreen::TITLE) {
                    screen = GameScreen::PLAYING;
                    return {.captureCursor = true, .resetTiming = true};
                }
                return {};
            case MenuAction::RESUME:
                if (screen == GameScreen::PAUSED) {
                    screen = GameScreen::PLAYING;
                    return {.captureCursor = true, .resetTiming = true};
                }
                return {};
            case MenuAction::OPEN_SETTINGS:
                if (screen == GameScreen::PAUSED) screen = GameScreen::SETTINGS;
                return {};
            case MenuAction::CLOSE_SETTINGS:
                if (screen == GameScreen::SETTINGS) screen = GameScreen::PAUSED;
                return {};
            case MenuAction::OPEN_VIDEO_SETTINGS:
                if (screen == GameScreen::SETTINGS) screen = GameScreen::VIDEO_SETTINGS;
                return {};
            case MenuAction::CLOSE_VIDEO_SETTINGS:
                if (screen == GameScreen::VIDEO_SETTINGS) screen = GameScreen::SETTINGS;
                return {};
            case MenuAction::QUIT:
                if (screen == GameScreen::TITLE || screen == GameScreen::PAUSED) {
                    return {.requestQuit = true};
                }
                return {};
            default:
                // Value steppers change engine settings, not the screen
                return {};
        }
    }

    // Losing focus while playing auto-pauses: the pointer lock is global
    // system state and must never survive a Cmd-Tab.
    constexpr GameFlowEffects onFocusLost() {
        if (screen == GameScreen::PLAYING) {
            screen = GameScreen::PAUSED;
            return {.releaseCursor = true, .resetTiming = true};
        }
        return {};
    }
};
