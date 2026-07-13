#pragma once

#include "engine/game_state.hpp"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Menu layout + hit testing — pure C++ (no Metal/Cocoa) so the geometry the
// mouse clicks is exactly the geometry the renderer draws, and both are
// unit-testable.
//
// Coordinates are normalized [0, 1] with a bottom-left origin (the same
// space UIOverlay draws in and locationInWindow/bounds produces). Sizes are
// specified in pixels at a 768-pt-tall reference window and scaled, so menus
// look identical at any resolution.
// ---------------------------------------------------------------------------

struct UIRect {
    float x = 0, y = 0, w = 0, h = 0;

    constexpr bool contains(float px, float py) const {
        return px >= x && px <= x + w && py >= y && py <= y + h;
    }
};

struct MenuText {
    float x, y;  // bottom-left of the text run
    float scale; // glyph scale (1.0 = 8 px tall)
    std::string text;
    float r = 1.f, g = 1.f, b = 1.f;
};

struct MenuButton {
    UIRect rect;
    std::string label;
    MenuAction action = MenuAction::NONE;
};

struct MenuLayout {
    float dimAlpha = 0.f; // full-screen darkening behind the menu
    UIRect panel{};       // w == 0 → no panel
    std::vector<MenuText> texts;
    std::vector<MenuButton> buttons;
};

// Live values the settings screen displays.
struct SettingsValues {
    int viewDistance = 12;    // chunks
    int fogLevel = 3;         // 0-10 (density = level * 0.0001 per block)
    int sensitivityLevel = 4; // 1-10 (sensitivity = level * 0.0005)
    int volumeLevel = 8;      // 0-10
};

// Performance HUD data (F3), filled by the engine with real measurements.
struct PerformanceStats {
    float fps = 0.f;
    uint32_t chunkCount = 0;
    uint32_t entityCount = 0;
    float frameTimeMs = 0.f;
    uint32_t pendingChunks = 0; // generation backlog + in-flight
    float genMsAvg = 0.f;       // EMA of per-chunk generation time
    float meshMsAvg = 0.f;      // EMA of per-chunk mesh build time
    uint32_t meshBuildsFrame = 0;
    float megaUsedMB = 0.f; // mega-buffer vertex bytes in use
    float megaCapMB = 0.f;
};

// Everything the UI pass needs to draw one frame.
struct UIFrameState {
    GameScreen screen = GameScreen::TITLE;
    int hoveredButton = -1; // index into menu.buttons, -1 = none
    bool showDebugHud = false;
    PerformanceStats stats{};
    MenuLayout menu;
};

// Normalized width of a string at the given glyph scale (8px glyphs plus
// 1px advance, matching UIOverlay's font metrics).
float menuTextWidth(const std::string& text, float scale, float pixelWidth);

// Build the layout for the current screen (Playing returns an empty layout).
MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values);

// Index of the button under the (normalized) mouse position, or -1.
int menuHitTest(const MenuLayout& layout, float mouseX, float mouseY);
