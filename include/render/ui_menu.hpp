#pragma once

#include "engine/game_state.hpp"
#include "world/macro_generation.hpp"
#include "world/view_distance.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Menu layout + hit testing, pure C++ (no Metal/Cocoa) so the geometry the
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
    static constexpr int MIN_VIEW_DISTANCE = MIN_RENDER_DISTANCE_CHUNKS;
    static constexpr int MAX_VIEW_DISTANCE = MAX_RENDER_DISTANCE_CHUNKS;
    static constexpr int DEFAULT_VIEW_DISTANCE = DEFAULT_RENDER_DISTANCE_CHUNKS;
    inline static constexpr std::array<int, 12> VIEW_DISTANCES = {
        MIN_VIEW_DISTANCE, 8, 12, 16, 24, 32, 64, 128, 192, 256, 384, MAX_VIEW_DISTANCE,
    };
    static_assert(VIEW_DISTANCES.front() == MIN_VIEW_DISTANCE);
    static_assert(VIEW_DISTANCES.back() == MAX_VIEW_DISTANCE);

    int viewDistance = DEFAULT_VIEW_DISTANCE; // chunks
    int fogLevel = 3;                         // 0-10
    int sensitivityLevel = 4;                 // 1-10
    int volumeLevel = 8;                      // 0-10
};

// Level → physical-unit conversions, defined once for the engine's init
// path and the stepper handlers (two independent copies once drifted).
constexpr float fogDensityForLevel(int level) {
    // The 512-chunk clear-weather horizon must remain legible. Weather still
    // raises this value dynamically, while level zero remains truly clear.
    return static_cast<float>(level) * 0.00005f;
}
constexpr float mouseSensitivityForLevel(int level) {
    return static_cast<float>(level) * 0.0005f;
}

// Performance HUD data (F3), filled by the engine with real measurements.
struct PerformanceStats {
    float fps = 0.f;
    uint32_t chunkCount = 0;
    uint32_t meshedCubeCount = 0;
    uint32_t entityCount = 0;
    float frameTimeMs = 0.f;
    float gpuFrameMs = 0.f;     // EMA of command-buffer GPUEndTime − GPUStartTime
    uint32_t pendingChunks = 0; // generation backlog + in-flight
    float genMsAvg = 0.f;       // EMA of per-chunk generation time
    float meshMsAvg = 0.f;      // EMA of per-chunk mesh build time
    uint32_t meshBuildsFrame = 0;
    float megaUsedMB = 0.f; // live combined vertex and index arena allocation
    float megaCapMB = 0.f;
    uint32_t exactSurfaceRequired = 0;
    uint32_t exactSurfaceReady = 0;
    uint32_t exactSurfaceUnresolvedColumns = 0;
    float exactSurfaceHandoffBlocks = 0.f;
    uint32_t farWantedTiles = 0;
    uint32_t farResidentTiles = 0;
    uint32_t farDrawnTiles = 0;
    uint32_t farBaseWantedTiles = 0;
    uint32_t farBaseResidentTiles = 0;
    uint32_t farBaseDrawnTiles = 0;
    uint32_t farBaseMissingTiles = 0;
    uint32_t farRefinementWantedTiles = 0;
    uint32_t farRefinementResidentTiles = 0;
    uint32_t farRefinementDrawnTiles = 0;
    uint32_t farFrustumCulledTiles = 0;
    uint32_t farOcclusionCulledTiles = 0;
    uint32_t farPendingTiles = 0;
    uint32_t farQueuedBaseTiles = 0;
    uint32_t farQueuedRefinementTiles = 0;
    uint32_t farActiveBaseWorkers = 0;
    uint32_t farReservedBaseWorkers = 0;
    uint32_t farActiveUrgentRefinements = 0;
    uint32_t farWorkerBudget = 0;
    uint32_t farCachedBaseTiles = 0;
    float farCoverageFrontierBlocks = 0.f;
    float farCacheMB = 0.f;
    float farMeshMB = 0.f;
    int64_t cubeX = 0;
    int32_t cubeY = 0;
    int64_t cubeZ = 0;
    uint64_t plateId = 0;
    worldgen::PlateBoundary boundary = worldgen::PlateBoundary::NONE;
    float temperatureC = 0.f;
    float precipitationMm = 0.f;
    Biome primaryBiome = Biome::PLAINS;
    Biome secondaryBiome = Biome::PLAINS;
    float biomeTransition = 0.f;
    uint8_t riverOrder = 0;
    uint32_t macroCacheEntries = 0;
    float macroCacheMB = 0.f;
    uint32_t pendingFluids = 0;
    uint64_t droppedFluidUpdates = 0;
    uint64_t droppedFluidFrontiers = 0;
    uint32_t shadowRefreshMask = 0;
    std::array<uint32_t, 5> shadowCasterCounts{};
    std::array<uint64_t, 5> shadowRefreshCounts{};
    uint32_t indirectHistoryResetMask = 0;
    bool indirectHistoryValid = false;
    bool cloudHistoryValid = false;
    bool froxelHistoryValid = false;
    uint64_t atmosphereSlowRefreshCount = 0;
    uint64_t atmosphereSkyRefreshCount = 0;
    float indirectPersistentMB = 0.0F;
    float cloudPersistentMB = 0.0F;
    float froxelPersistentMB = 0.0F;
    float integratedAtmosphericPersistentMB = 0.0F;
    uint64_t weatherRequests = 0;
    uint64_t weatherCoalescedRequests = 0;
    uint64_t weatherBuildsStarted = 0;
    uint64_t weatherSnapshotsPublished = 0;
    uint64_t weatherStaleBuildsDiscarded = 0;
    uint32_t weatherPendingRequests = 0;
    bool weatherWorkerBusy = false;
    uint32_t thunderPending = 0;
    float weatherPressureHpa = 0.0F;
    float weatherHumidity = 0.0F;
    float weatherTemperatureC = 0.0F;
    float weatherWindX = 0.0F;
    float weatherWindZ = 0.0F;
    float cloudCoverage = 0.0F;
    uint8_t cloudType = 0;
    float precipitationIntensity = 0.0F;
    uint8_t precipitationKind = 0;
    float stormPotential = 0.0F;
    float weatherFogExtinction = 0.0F;
    float aerosolDensity = 0.0F;
    uint64_t stormId = 0;
    float lunarPhaseEnergy = 0.0F;
    float lunarPhaseCycle = 0.0F;
};

// Everything the UI pass needs to draw one frame.
struct UIFrameState {
    GameScreen screen = GameScreen::TITLE;
    int hoveredButton = -1; // index into menu.buttons, -1 = none
    bool showDebugHud = false;
    bool cameraUnderwater = false; // drives the underwater veil + god rays
    PerformanceStats stats{};
    MenuLayout menu;
};

// Normalized width of a string at the given glyph scale (8px glyphs plus
// 1px advance, matching UIOverlay's font metrics).
float menuTextWidth(const std::string& text, float scale, float pixelWidth);

struct GraphicsSettings; // render/graphics_settings.hpp (video screen values)

// Build the layout for the current screen (Playing returns an empty layout).
MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values, const GraphicsSettings& gfx);

// Index of the button under the (normalized) mouse position, or -1.
int menuHitTest(const MenuLayout& layout, float mouseX, float mouseY);
