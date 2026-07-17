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
    TITLE,                // launch screen: PLAY / QUIT, backdrop only
    WORLD_SELECT,         // saved world list, reached from TITLE
    WORLD_CREATE,         // name/seed/toggles form, reached from WORLD_SELECT
    WORLD_DELETE_CONFIRM, // destructive-delete confirmation
    PLAYING,              // normal gameplay, cursor captured
    PAUSED,               // ESC menu: RESUME / SETTINGS / QUIT
    SETTINGS,             // settings panel, reached from PAUSED
    VIDEO_SETTINGS,       // per-effect video options, reached from SETTINGS
    INVENTORY,            // player inventory + 2x2 craft grid (world frozen)
    CRAFTING,             // crafting table 3x3 grid (world frozen)
    FURNACE,              // furnace slots + gauges (world frozen)
    DEATH,                // respawn / quit after health reaches zero
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
    // World flow. Side-effectful actions (selecting, playing, creating,
    // deleting, respawning, quitting to title) return no flow change from
    // onMenuAction; the engine performs the effect and then drives the flow
    // through onWorldStarted/onWorldStopped/onRespawn, mirroring how the
    // settings steppers already work.
    OPEN_WORLD_SELECT,
    OPEN_WORLD_CREATE,
    SELECT_WORLD, // MenuButton::payload = index into the cached world list
    WORLD_LIST_UP,
    WORLD_LIST_DOWN,
    PLAY_SELECTED_WORLD,
    CREATE_WORLD_CONFIRM,
    RANDOM_SEED,
    TOGGLE_GEN_STRUCTURES,
    TOGGLE_GEN_FAUNA,
    TOGGLE_GEN_WEATHER,
    TOGGLE_GEN_DAY_CYCLE,
    TOGGLE_CREATE_MODE,
    REQUEST_DELETE_WORLD,
    CONFIRM_DELETE,
    CANCEL_DELETE,
    WORLD_BACK,
    RESPAWN,
    SAVE_QUIT_TO_TITLE,
    TOGGLE_GAME_MODE,
    CREATIVE_PAGE_PREV,
    CREATIVE_PAGE_NEXT,
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

// Screens that run over a live world session (the HUD draws only there).
constexpr bool screenHasWorldSession(GameScreen screen) {
    return screen == GameScreen::PLAYING || screen == GameScreen::PAUSED ||
           screen == GameScreen::SETTINGS || screen == GameScreen::VIDEO_SETTINGS ||
           screen == GameScreen::INVENTORY || screen == GameScreen::CRAFTING ||
           screen == GameScreen::FURNACE || screen == GameScreen::DEATH;
}

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

    // Screens that require a live world session behind them.
    constexpr bool worldScreens() const { return screenHasWorldSession(screen); }

    // Container screens the inventory key and container blocks open.
    constexpr bool inContainer() const {
        return screen == GameScreen::INVENTORY || screen == GameScreen::CRAFTING ||
               screen == GameScreen::FURNACE;
    }

    // ESC: pause from gameplay, resume from pause, back out of settings,
    // close containers, back out of the world menus. Death ignores it.
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
            case GameScreen::INVENTORY:
            case GameScreen::CRAFTING:
            case GameScreen::FURNACE:
                screen = GameScreen::PLAYING;
                return {.captureCursor = true, .resetTiming = true};
            case GameScreen::WORLD_SELECT:
                screen = GameScreen::TITLE;
                return {};
            case GameScreen::WORLD_CREATE:
            case GameScreen::WORLD_DELETE_CONFIRM:
                screen = GameScreen::WORLD_SELECT;
                return {};
            case GameScreen::DEATH:
            case GameScreen::TITLE:
                return {};
        }
        return {};
    }

    // E toggles the inventory and closes any open container.
    constexpr GameFlowEffects onInventoryKey() {
        if (screen == GameScreen::PLAYING) {
            screen = GameScreen::INVENTORY;
            return {.releaseCursor = true, .resetTiming = true};
        }
        if (inContainer()) {
            screen = GameScreen::PLAYING;
            return {.captureCursor = true, .resetTiming = true};
        }
        return {};
    }

    // Right-clicking a crafting table or furnace while playing.
    constexpr GameFlowEffects onContainerOpened(GameScreen container) {
        if (screen != GameScreen::PLAYING ||
            (container != GameScreen::CRAFTING && container != GameScreen::FURNACE)) {
            return {};
        }
        screen = container;
        return {.releaseCursor = true, .resetTiming = true};
    }

    // The engine drives these after its side effect succeeds.
    constexpr GameFlowEffects onWorldStarted() {
        if (screen != GameScreen::TITLE && screen != GameScreen::WORLD_SELECT &&
            screen != GameScreen::WORLD_CREATE) {
            return {};
        }
        screen = GameScreen::PLAYING;
        return {.captureCursor = true, .resetTiming = true};
    }

    constexpr GameFlowEffects onWorldStopped() {
        screen = GameScreen::TITLE;
        return {.releaseCursor = true, .resetTiming = true};
    }

    constexpr GameFlowEffects onPlayerDied() {
        if (screen != GameScreen::PLAYING) return {};
        screen = GameScreen::DEATH;
        return {.releaseCursor = true, .resetTiming = true};
    }

    constexpr GameFlowEffects onRespawn() {
        if (screen != GameScreen::DEATH) return {};
        screen = GameScreen::PLAYING;
        return {.captureCursor = true, .resetTiming = true};
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
            case MenuAction::OPEN_WORLD_SELECT:
                if (screen == GameScreen::TITLE) screen = GameScreen::WORLD_SELECT;
                return {};
            case MenuAction::OPEN_WORLD_CREATE:
                if (screen == GameScreen::WORLD_SELECT) screen = GameScreen::WORLD_CREATE;
                return {};
            case MenuAction::REQUEST_DELETE_WORLD:
                if (screen == GameScreen::WORLD_SELECT) screen = GameScreen::WORLD_DELETE_CONFIRM;
                return {};
            case MenuAction::CANCEL_DELETE:
                if (screen == GameScreen::WORLD_DELETE_CONFIRM) screen = GameScreen::WORLD_SELECT;
                return {};
            case MenuAction::WORLD_BACK:
                if (screen == GameScreen::WORLD_SELECT) {
                    screen = GameScreen::TITLE;
                } else if (screen == GameScreen::WORLD_CREATE ||
                           screen == GameScreen::WORLD_DELETE_CONFIRM) {
                    screen = GameScreen::WORLD_SELECT;
                }
                return {};
            case MenuAction::QUIT:
                if (screen == GameScreen::TITLE || screen == GameScreen::PAUSED ||
                    screen == GameScreen::WORLD_SELECT) {
                    return {.requestQuit = true};
                }
                return {};
            default:
                // Value steppers and engine-side effects change no screen
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
