#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

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
    double result = 0.0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, result, std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != end || !std::isfinite(result)) {
        return std::nullopt;
    }
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
