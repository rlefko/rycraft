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
    Title,     // launch screen: PLAY / QUIT, world visible behind
    Playing,   // normal gameplay, cursor captured
    Paused,    // ESC menu: RESUME / SETTINGS / QUIT
    Settings,  // settings panel, reached from Paused
};

enum class MenuAction {
    None,
    Play,
    Resume,
    OpenSettings,
    CloseSettings,
    Quit,
    // Settings value steppers (handled by the engine; no screen change)
    ViewDistanceDown,
    ViewDistanceUp,
    FogDown,
    FogUp,
    SensitivityDown,
    SensitivityUp,
    VolumeDown,
    VolumeUp,
};

// What the engine must do after a transition.
struct GameFlowEffects {
    bool captureCursor = false;  // hide + pointer-lock the mouse
    bool releaseCursor = false;  // unhide + free the mouse
    bool resetTiming = false;    // zero the tick accumulator + mouse delta
    bool requestQuit = false;    // save and terminate
};

struct GameFlow {
    GameScreen screen = GameScreen::Title;

    // Menus freeze the world: ticks run only while Playing.
    constexpr bool inMenu() const { return screen != GameScreen::Playing; }

    // ESC: pause from gameplay, resume from pause, back out of settings.
    constexpr GameFlowEffects onEscape() {
        switch (screen) {
            case GameScreen::Playing:
                screen = GameScreen::Paused;
                return {.releaseCursor = true, .resetTiming = true};
            case GameScreen::Paused:
                screen = GameScreen::Playing;
                return {.captureCursor = true, .resetTiming = true};
            case GameScreen::Settings:
                screen = GameScreen::Paused;
                return {};
            case GameScreen::Title:
                return {};
        }
        return {};
    }

    constexpr GameFlowEffects onMenuAction(MenuAction action) {
        switch (action) {
            case MenuAction::Play:
                if (screen == GameScreen::Title) {
                    screen = GameScreen::Playing;
                    return {.captureCursor = true, .resetTiming = true};
                }
                return {};
            case MenuAction::Resume:
                if (screen == GameScreen::Paused) {
                    screen = GameScreen::Playing;
                    return {.captureCursor = true, .resetTiming = true};
                }
                return {};
            case MenuAction::OpenSettings:
                if (screen == GameScreen::Paused) screen = GameScreen::Settings;
                return {};
            case MenuAction::CloseSettings:
                if (screen == GameScreen::Settings) screen = GameScreen::Paused;
                return {};
            case MenuAction::Quit:
                if (screen == GameScreen::Title || screen == GameScreen::Paused) {
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
        if (screen == GameScreen::Playing) {
            screen = GameScreen::Paused;
            return {.releaseCursor = true, .resetTiming = true};
        }
        return {};
    }
};
