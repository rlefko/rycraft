#pragma once

#include "engine/game_state.hpp"
#include "world/item.hpp"
#include "world/macro_generation.hpp"
#include "world/view_distance.hpp"
#include "world/world_config.hpp"

#include <array>
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
    int payload = -1;        // row index for list actions (SELECT_WORLD)
    bool emphasized = false; // selected world row highlight
};

// One text-entry box: the label draws above it, the caret only while the
// engine has this field focused and the blink phase is on.
struct TextFieldWidget {
    UIRect rect;
    std::string label;
    std::string text;
    bool focused = false;
    bool caret = false;
};

// Which stack collection a slot addresses; index is domain-relative
// (INVENTORY 0-8 hotbar then 9-35 main, CREATIVE_PALETTE absolute).
enum class SlotDomain : uint8_t {
    NONE,
    INVENTORY,
    CRAFT_IN,
    CRAFT_OUT,
    FURNACE_INPUT,
    FURNACE_FUEL,
    FURNACE_OUTPUT,
    CREATIVE_PALETTE,
};

struct SlotRef {
    SlotDomain domain = SlotDomain::NONE;
    int index = 0;
};

struct SlotWidget {
    UIRect rect;
    SlotRef ref;
    ItemStack stack{}; // drawn snapshot (palette entries draw as full stacks)
};

// A filled gauge (furnace cook arrow, flame): fill is 0..1 along the axis.
struct MeterWidget {
    UIRect rect;
    float fill = 0.f;
    bool vertical = false;
    float r = 1.f, g = 1.f, b = 1.f;
};

struct MenuLayout {
    float dimAlpha = 0.f; // full-screen darkening behind the menu
    UIRect panel{};       // w == 0 → no panel
    std::vector<MenuText> texts;
    std::vector<MenuButton> buttons;
    std::vector<TextFieldWidget> textFields;
    std::vector<SlotWidget> slots;
    std::vector<MeterWidget> meters;
};

// Typed hit-testing across every widget kind. menuHitTest remains for
// button-only callers.
enum class UIHitKind : uint8_t { NONE, BUTTON, TEXT_FIELD, SLOT };
struct UIHit {
    UIHitKind kind = UIHitKind::NONE;
    int index = -1; // into the matching layout vector
};
UIHit uiHitTest(const MenuLayout& layout, float mouseX, float mouseY);

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
};

// Everything the UI pass needs to draw one frame.
struct UIFrameState {
    GameScreen screen = GameScreen::TITLE;
    int hoveredButton = -1; // index into menu.buttons, -1 = none
    int hoveredSlot = -1;   // index into menu.slots, -1 = none
    ItemStack cursorStack{};
    float mouseX = 0.f; // normalized, for the held stack and tooltip
    float mouseY = 0.f;
    float miningProgress = 0.f; // 0 = not mining, drives the break bar
    std::string tooltipText;    // hovered-slot display name, empty = none
    bool showDebugHud = false;
    bool cameraUnderwater = false; // drives the underwater veil + god rays
    PerformanceStats stats{};
    MenuLayout menu;

    // Hotbar snapshot the HUD draws; the engine copies it from the live
    // Inventory each frame so the render layer never touches engine state.
    struct HudHotbar {
        std::array<ItemStack, 9> slots{};
        int selected = 0;
    };
    HudHotbar hotbar;
};

// Normalized width of a string at the given glyph scale (8px glyphs plus
// 1px advance, matching UIOverlay's font metrics).
float menuTextWidth(const std::string& text, float scale, float pixelWidth);

struct GraphicsSettings; // render/graphics_settings.hpp (video screen values)

// Per-screen editable state the engine owns and the layouts render.
struct WorldSelectState {
    int selected = -1; // index into the cached world list, -1 = none
    int scroll = 0;    // first visible row
    static constexpr int VISIBLE_ROWS = 5;
};

struct WorldCreateState {
    std::string name;
    std::string seedText;  // digits only; empty creates a random seed
    int focusedField = -1; // 0 = name, 1 = seed, -1 = none
    bool structures = true;
    bool fauna = true;
    bool weather = true;
    bool dayCycle = true;
    bool creative = false;
    static constexpr size_t MAX_NAME_LENGTH = 24;
    static constexpr size_t MAX_SEED_LENGTH = 10;
};

// Field charset enforcement in one place: the world-name charset comes from
// world_list.hpp so typed names always render and persist escape-free.
std::string filterTextField(const std::string& raw, bool digitsOnly, size_t maxLength);

// Everything a container screen shows, snapshotted by the engine per frame.
struct ContainerView {
    std::array<ItemStack, 36> inventory{}; // 0-8 hotbar, 9-35 main
    std::array<ItemStack, 9> craftGrid{};  // first 4 used on INVENTORY
    int craftGridSize = 4;                 // 4 (2x2) or 9 (3x3)
    ItemStack craftResult{};
    ItemStack furnaceInput{};
    ItemStack furnaceFuel{};
    ItemStack furnaceOutput{};
    float furnaceCook = 0.f;     // 0..1 arrow fill
    float furnaceFuelLeft = 0.f; // 0..1 flame fill
    bool creative = false;       // palette instead of the craft grid
    int creativePage = 0;
};

// Everything any menu screen draws, filled by the engine each frame.
struct MenuContext {
    SettingsValues settings{};
    const GraphicsSettings* gfx = nullptr; // video screen only (non-owning)
    GameMode mode = GameMode::SURVIVAL;    // pause-screen mode row
    std::vector<std::string> worldRows;    // display labels, cached on entry
    WorldSelectState worldSelect{};
    WorldCreateState worldCreate{};
    bool caretVisible = true;
    std::string deleteWorldName;
    ContainerView container{};
};

// Creative palette paging: 45 palette slots per screenful.
inline constexpr int CREATIVE_PALETTE_COLUMNS = 9;
inline constexpr int CREATIVE_PALETTE_ROWS = 5;
inline constexpr int CREATIVE_PALETTE_PAGE_SIZE = CREATIVE_PALETTE_COLUMNS * CREATIVE_PALETTE_ROWS;

// Build the layout for any screen (Playing returns an empty layout).
MenuLayout buildScreenLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                             const MenuContext& ctx);

// Build the layout for the current screen (Playing returns an empty layout).
MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values, const GraphicsSettings& gfx);

// Index of the button under the (normalized) mouse position, or -1.
int menuHitTest(const MenuLayout& layout, float mouseX, float mouseY);
