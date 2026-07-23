#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

class World;
struct LightningEvent;

// Refines a weather-grid strike only from already resident cubes. Remote
// strikes retain their learned coarse height and never construct an exact
// column plan or hydrology owner on the fixed tick.
void resolveLightningTerrainHeightIfLoaded(const World& world, LightningEvent& event);

// ---------------------------------------------------------------------------
// Game flow, which screen the player is on and how input transitions it.
//
// Pure C++ (no Cocoa/Metal) so every transition is unit-testable. The engine
// applies the returned effects: cursor capture/release, timing resets, and
// quitting. World simulation ticks only while the screen is Playing; menus
// freeze time.
// ---------------------------------------------------------------------------

// Parse an absolute tick count without accepting the sign, whitespace, base
// prefixes, or overflow behavior that strtoull permits. Capture tooling uses
// saved-world-scale uint64 values so a fixed time can select both time of day
// and a repeatable lunar phase.
inline std::optional<uint64_t> parseUnsignedDecimal(std::string_view text) noexcept {
    if (text.empty()) return std::nullopt;

    uint64_t result = 0;
    for (char character : text) {
        if (character < '0' || character > '9') return std::nullopt;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10U) {
            return std::nullopt;
        }
        result = result * 10U + digit;
    }
    return result;
}

struct CaptureLightningOverride {
    double x = 0.0;
    double z = 0.0;
    uint64_t id = 0;
    uint64_t ageTicks = 0;
};

namespace game_state_detail {

inline std::optional<double> parseFiniteDecimal(std::string_view text) noexcept {
    if (text.empty()) return std::nullopt;

    std::size_t cursor = 0;
    bool negative = false;
    if (text[cursor] == '+' || text[cursor] == '-') {
        negative = text[cursor] == '-';
        if (++cursor == text.size()) return std::nullopt;
    }

    long double significand = 0.0L;
    std::size_t fractionalDigits = 0;
    bool sawDigit = false;
    while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
        sawDigit = true;
        significand = significand * 10.0L + static_cast<int>(text[cursor] - '0');
        if (!std::isfinite(significand)) return std::nullopt;
        ++cursor;
    }
    if (cursor < text.size() && text[cursor] == '.') {
        ++cursor;
        while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
            sawDigit = true;
            significand = significand * 10.0L + static_cast<int>(text[cursor] - '0');
            if (!std::isfinite(significand)) return std::nullopt;
            ++fractionalDigits;
            ++cursor;
        }
    }
    if (!sawDigit) return std::nullopt;

    int explicitExponent = 0;
    if (cursor < text.size() && (text[cursor] == 'e' || text[cursor] == 'E')) {
        if (++cursor == text.size()) return std::nullopt;
        bool exponentNegative = false;
        if (text[cursor] == '+' || text[cursor] == '-') {
            exponentNegative = text[cursor] == '-';
            if (++cursor == text.size()) return std::nullopt;
        }
        bool sawExponentDigit = false;
        while (cursor < text.size() && text[cursor] >= '0' && text[cursor] <= '9') {
            sawExponentDigit = true;
            const int digit = text[cursor] - '0';
            if (explicitExponent > 10'000) return std::nullopt;
            explicitExponent = explicitExponent * 10 + digit;
            ++cursor;
        }
        if (!sawExponentDigit) return std::nullopt;
        if (exponentNegative) explicitExponent = -explicitExponent;
    }
    if (cursor != text.size() || fractionalDigits > 10'000) return std::nullopt;

    const int decimalExponent = explicitExponent - static_cast<int>(fractionalDigits);
    if (decimalExponent < -10'000 || decimalExponent > 10'000) return std::nullopt;
    const long double scaled = significand * std::pow(10.0L, decimalExponent);
    const double result = static_cast<double>(negative ? -scaled : scaled);
    if (!std::isfinite(result) || (significand != 0.0L && result == 0.0)) return std::nullopt;
    return result;
}

} // namespace game_state_detail

// Strict capture-only format: x,z,id,ageTicks. Coordinates are finite decimal
// values; the event ID and rendered age are unsigned decimal integers.
inline std::optional<CaptureLightningOverride>
parseCaptureLightningOverride(std::string_view text) noexcept {
    std::string_view fields[4];
    for (std::size_t field = 0; field < 3; ++field) {
        const std::size_t delimiter = text.find(',');
        if (delimiter == std::string_view::npos) return std::nullopt;
        fields[field] = text.substr(0, delimiter);
        text.remove_prefix(delimiter + 1);
    }
    if (text.find(',') != std::string_view::npos) return std::nullopt;
    fields[3] = text;

    const auto x = game_state_detail::parseFiniteDecimal(fields[0]);
    const auto z = game_state_detail::parseFiniteDecimal(fields[1]);
    const auto id = parseUnsignedDecimal(fields[2]);
    const auto ageTicks = parseUnsignedDecimal(fields[3]);
    if (!x || !z || !id || !ageTicks) return std::nullopt;
    const double minimumCoordinate = static_cast<double>(std::numeric_limits<int64_t>::min());
    const double maximumCoordinateExclusive = -minimumCoordinate;
    if (*x < minimumCoordinate || *x >= maximumCoordinateExclusive || *z < minimumCoordinate ||
        *z >= maximumCoordinateExclusive) {
        return std::nullopt;
    }
    return CaptureLightningOverride{*x, *z, *id, *ageTicks};
}

enum class GameScreen {
    TITLE,                   // launch screen: PLAY / QUIT, backdrop only
    WORLD_SELECT,            // saved world list, reached from TITLE
    WORLD_CREATE,            // name/seed/toggles form, reached from WORLD_SELECT
    WORLD_DELETE_CONFIRM,    // destructive-delete confirmation
    WORLD_SUCCESSOR_CONFIRM, // explicit legacy/stale v4 successor confirmation
    PLAYING,                 // normal gameplay, cursor captured
    PAUSED,                  // ESC menu: RESUME / SETTINGS / QUIT
    SETTINGS,                // settings panel, reached from PAUSED
    VIDEO_SETTINGS,          // per-effect video options, reached from SETTINGS
    INVENTORY,               // player inventory + 2x2 craft grid (world frozen)
    CRAFTING,                // crafting table 3x3 grid (world frozen)
    FURNACE,                 // furnace slots + gauges (world frozen)
    CHEST,                   // 27-slot storage block (world frozen)
    DEATH,                   // respawn / quit after health reaches zero
};

constexpr std::optional<GameScreen> gameScreenFromEnvironment(std::string_view name) {
    if (name == "title") return GameScreen::TITLE;
    if (name == "worlds") return GameScreen::WORLD_SELECT;
    if (name == "create") return GameScreen::WORLD_CREATE;
    if (name == "delete") return GameScreen::WORLD_DELETE_CONFIRM;
    if (name == "playing") return GameScreen::PLAYING;
    if (name == "paused") return GameScreen::PAUSED;
    if (name == "settings") return GameScreen::SETTINGS;
    if (name == "video") return GameScreen::VIDEO_SETTINGS;
    if (name == "inventory") return GameScreen::INVENTORY;
    if (name == "crafting") return GameScreen::CRAFTING;
    if (name == "furnace") return GameScreen::FURNACE;
    if (name == "chest") return GameScreen::CHEST;
    if (name == "death") return GameScreen::DEATH;
    return std::nullopt;
}

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
    REQUEST_V4_SUCCESSOR,
    CONFIRM_V4_SUCCESSOR,
    CANCEL_V4_SUCCESSOR,
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
    // Generator v4 installation and fail-closed recovery actions. These are
    // handled by the engine and deliberately do not change the current
    // gameplay screen by themselves.
    DOWNLOAD_MODEL,
    CANCEL_MODEL,
    RETRY_MODEL,
    REPAIR_MODEL,
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
    INDIRECT_DOWN,
    INDIRECT_UP,
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
           screen == GameScreen::FURNACE || screen == GameScreen::CHEST ||
           screen == GameScreen::DEATH;
}

// Ordinary launches stay on the title or Worlds screens without selecting,
// creating, or migrating a persistence profile. Capture and performance
// sessions may still request gameplay explicitly through automaticGameplay.
constexpr bool launchRequestsWorldSession(std::optional<GameScreen> requestedScreen,
                                          bool automaticGameplay) noexcept {
    return automaticGameplay || (requestedScreen && screenHasWorldSession(*requestedScreen));
}

// What the engine must do after a transition.
struct GameFlowEffects {
    bool captureCursor = false; // hide + pointer-lock the mouse
    bool releaseCursor = false; // unhide + free the mouse
    bool resetTiming = false;   // zero the tick accumulator + mouse delta
    bool requestQuit = false;   // save and terminate
};

struct FrameCaptureActions {
    bool capture = false;
    bool quit = false;
};

// Capture timing starts when the requested scene is actually drawable. This
// keeps asynchronous world preparation from consuming the requested frame
// budget before the first full scene can render.
struct FrameCaptureClock {
    uint64_t renderedFrames = 0;
    std::optional<uint64_t> capturedAt;
    bool quitRequested = false;

    constexpr FrameCaptureActions onRenderedFrame(uint64_t captureFrame,
                                                  uint64_t quitDelayFrames = 60) {
        FrameCaptureActions actions;
        if (!capturedAt && renderedFrames >= captureFrame) {
            capturedAt = renderedFrames;
            actions.capture = true;
        } else if (capturedAt && !quitRequested &&
                   renderedFrames >= *capturedAt + quitDelayFrames) {
            quitRequested = true;
            actions.quit = true;
        }
        ++renderedFrames;
        return actions;
    }
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
               screen == GameScreen::FURNACE || screen == GameScreen::CHEST;
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
            case GameScreen::CHEST:
                screen = GameScreen::PLAYING;
                return {.captureCursor = true, .resetTiming = true};
            case GameScreen::WORLD_SELECT:
                screen = GameScreen::TITLE;
                return {};
            case GameScreen::WORLD_CREATE:
            case GameScreen::WORLD_DELETE_CONFIRM:
            case GameScreen::WORLD_SUCCESSOR_CONFIRM:
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

    // Right-clicking a crafting table, furnace, or chest while playing.
    constexpr GameFlowEffects onContainerOpened(GameScreen container) {
        if (screen != GameScreen::PLAYING ||
            (container != GameScreen::CRAFTING && container != GameScreen::FURNACE &&
             container != GameScreen::CHEST)) {
            return {};
        }
        screen = container;
        return {.releaseCursor = true, .resetTiming = true};
    }

    // The engine drives these after its side effect succeeds.
    constexpr GameFlowEffects onWorldStarted() {
        if (screen != GameScreen::TITLE && screen != GameScreen::WORLD_SELECT &&
            screen != GameScreen::WORLD_CREATE && screen != GameScreen::WORLD_SUCCESSOR_CONFIRM) {
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

    // A generation failure temporarily replaces the live-world screen with a
    // recovery gate. Restore the exact prior screen only after the entry gate
    // is ready again, including paused and container screens.
    constexpr GameFlowEffects onGenerationRecovered(GameScreen priorScreen) {
        if (!screenHasWorldSession(priorScreen)) return {};
        screen = priorScreen;
        return {.captureCursor = priorScreen == GameScreen::PLAYING,
                .releaseCursor = priorScreen != GameScreen::PLAYING,
                .resetTiming = true};
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
            case MenuAction::REQUEST_V4_SUCCESSOR:
                if (screen == GameScreen::WORLD_SELECT)
                    screen = GameScreen::WORLD_SUCCESSOR_CONFIRM;
                return {};
            case MenuAction::CANCEL_DELETE:
                if (screen == GameScreen::WORLD_DELETE_CONFIRM) screen = GameScreen::WORLD_SELECT;
                return {};
            case MenuAction::CANCEL_V4_SUCCESSOR:
                if (screen == GameScreen::WORLD_SUCCESSOR_CONFIRM)
                    screen = GameScreen::WORLD_SELECT;
                return {};
            case MenuAction::WORLD_BACK:
                if (screen == GameScreen::WORLD_SELECT) {
                    screen = GameScreen::TITLE;
                } else if (screen == GameScreen::WORLD_CREATE ||
                           screen == GameScreen::WORLD_DELETE_CONFIRM ||
                           screen == GameScreen::WORLD_SUCCESSOR_CONFIRM) {
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
