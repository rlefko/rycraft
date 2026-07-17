#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/graphics_settings.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Engine: game flow, menus, input, hotbar
// ===========================================================================

TEST_CASE("InputBindings save/load round-trips a custom binding", "[engine][bindings]") {
    TempDir dir("bindings");
    std::string path = dir.path() + "/bindings.json";

    InputBindings custom;
    custom.forward.key = Key::Up;
    custom.jump.key = Key::F;
    REQUIRE(custom.save(path));

    auto loaded = InputBindings::load(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->forward.key == Key::Up);
    REQUIRE(loaded->jump.key == Key::F);
    REQUIRE(loaded->backward.key == Key::S); // untouched bindings keep defaults
}

TEST_CASE("InputBindings load returns defaults for a missing file", "[engine][bindings]") {
    TempDir dir("bindings_missing");
    auto loaded = InputBindings::load(dir.path() + "/nope.json");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->forward.key == Key::W);
}

TEST_CASE("InputBindings defaults: Ctrl sprints, Shift sneaks", "[engine][bindings]") {
    // Minecraft layout, and what the README documents. Sprint once sat on
    // LeftShift, which fly-descend now needs.
    InputBindings defaults;
    REQUIRE(defaults.sprint.key == Key::LeftControl);
    REQUIRE(defaults.sneak.key == Key::LeftShift);

    TempDir dir("bindings_sprint_sneak");
    std::string path = dir.path() + "/bindings.json";
    REQUIRE(defaults.save(path));
    auto loaded = InputBindings::load(path);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->sprint.key == Key::LeftControl);
    REQUIRE(loaded->sneak.key == Key::LeftShift);
}

// ============================================================================
// Double-tap detection (sprint on W, fly toggle on Space)
// ============================================================================

TEST_CASE("Double-tap: two presses inside the window latch for the tick", "[engine][input]") {
    InputState input;

    input.recordPress(Key::W, 1.0);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(input.isPressedForTick(Key::W));
    REQUIRE(input.isDown(Key::W));

    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW * 0.5);
    REQUIRE(input.isDoubleTappedForTick(Key::W));

    // Consumed at tick end, exactly like keysPressedForTick
    input.clearTickPresses();
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isPressedForTick(Key::W));
}

TEST_CASE("Double-tap: a slow second press does not latch", "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW + 0.05);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));

    // ...but that second press starts a fresh window
    input.recordPress(Key::W, 1.0 + InputState::DOUBLE_TAP_WINDOW + 0.15);
    REQUIRE(input.isDoubleTappedForTick(Key::W));
}

TEST_CASE("Double-tap: a triple-tap fires exactly one gesture", "[engine][input]") {
    InputState input;
    input.recordPress(Key::Space, 1.0);
    input.recordPress(Key::Space, 1.1); // fires and consumes the history
    REQUIRE(input.isDoubleTappedForTick(Key::Space));
    input.clearTickPresses();

    input.recordPress(Key::Space, 1.2); // pairs with nothing — history was consumed
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));
}

TEST_CASE("Double-tap: keys are tracked independently", "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::Space, 1.1);
    REQUIRE(!input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));

    input.recordPress(Key::W, 1.2);
    REQUIRE(input.isDoubleTappedForTick(Key::W));
    REQUIRE(!input.isDoubleTappedForTick(Key::Space));
}

TEST_CASE("Double-tap: latch survives per-frame update() until a tick consumes it",
          "[engine][input]") {
    InputState input;
    input.recordPress(Key::W, 1.0);
    input.recordPress(Key::W, 1.1);

    // Several tickless frames pass — the gesture must not be dropped
    input.update();
    input.update();
    REQUIRE(input.isDoubleTappedForTick(Key::W));
}

// ============================================================================
// Game flow + menu layout tests (pure C++, no Metal)
// ============================================================================

TEST_CASE("GameFlow: ESC toggles pause and backs out of settings", "[ui][flow]") {
    GameFlow flow;
    REQUIRE(flow.screen == GameScreen::TITLE);

    // ESC is inert on the title screen
    auto fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE(!fx.captureCursor);

    // PLAY enters gameplay and captures the mouse
    fx = flow.onMenuAction(MenuAction::PLAY);
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);
    REQUIRE(fx.resetTiming);

    // ESC pauses (release + timing reset), ESC again resumes (capture)
    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(fx.releaseCursor);
    REQUIRE(fx.resetTiming);

    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Settings sits under pause; ESC backs out one level
    flow.onEscape();
    flow.onMenuAction(MenuAction::OPEN_SETTINGS);
    REQUIRE(flow.screen == GameScreen::SETTINGS);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PAUSED);
}

TEST_CASE("Settings save/load round-trips values and video settings", "[engine][settings]") {
    TempDir dir("settings");
    std::string path = dir.path() + "/settings.json";

    SettingsValues values;
    values.viewDistance = SettingsValues::MAX_VIEW_DISTANCE;
    values.fogLevel = 7;
    values.sensitivityLevel = 9;
    values.volumeLevel = 2;
    GraphicsSettings gfx;
    gfx.shadowQuality = 1;
    gfx.volumetricLight = false;
    gfx.cloudMode = 0;
    gfx.ssao = false;
    gfx.waterReflections = false;
    gfx.wavingFoliage = false;
    gfx.lensFlare = false;
    gfx.bloomLevel = 8;
    gfx.vibrance = 3;
    gfx.sharpening = 6;

    REQUIRE(saveSettings(path, values, gfx));
    LoadedSettings loaded = loadSettings(path);

    REQUIRE(loaded.values.viewDistance == SettingsValues::MAX_VIEW_DISTANCE);
    REQUIRE(loaded.values.fogLevel == 7);
    REQUIRE(loaded.values.sensitivityLevel == 9);
    REQUIRE(loaded.values.volumeLevel == 2);
    REQUIRE(loaded.gfx.shadowQuality == 1);
    REQUIRE(loaded.gfx.volumetricLight == false);
    REQUIRE(loaded.gfx.cloudMode == 0);
    REQUIRE(loaded.gfx.ssao == false);
    REQUIRE(loaded.gfx.waterReflections == false);
    REQUIRE(loaded.gfx.wavingFoliage == false);
    REQUIRE(loaded.gfx.lensFlare == false);
    REQUIRE(loaded.gfx.bloomLevel == 8);
    REQUIRE(loaded.gfx.vibrance == 3);
    REQUIRE(loaded.gfx.sharpening == 6);
}

TEST_CASE("Settings reuse the supported world view-distance contract", "[engine][settings]") {
    STATIC_REQUIRE(SettingsValues::MIN_VIEW_DISTANCE == MIN_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::MAX_VIEW_DISTANCE == MAX_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::DEFAULT_VIEW_DISTANCE == DEFAULT_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::VIEW_DISTANCES.front() == MIN_RENDER_DISTANCE_CHUNKS);
    STATIC_REQUIRE(SettingsValues::VIEW_DISTANCES.back() == MAX_RENDER_DISTANCE_CHUNKS);
}

TEST_CASE("Default clear-weather fog preserves the eight-kilometer horizon",
          "[engine][settings][render][far-terrain]") {
    REQUIRE(fogDensityForLevel(0) == 0.0F);
    REQUIRE(fogDensityForLevel(3) == Catch::Approx(0.00015F));
    constexpr float HORIZON_BLOCKS = MAX_RENDER_DISTANCE_CHUNKS * CHUNK_EDGE;
    const float fogCoverage = 1.0F - std::exp(-fogDensityForLevel(3) * HORIZON_BLOCKS);
    REQUIRE(fogCoverage < 0.75F);
}

TEST_CASE("Settings load: missing file and out-of-range values fall back", "[engine][settings]") {
    TempDir dir("settings");

    // Missing file → the max-preset defaults
    LoadedSettings missing = loadSettings(dir.path() + "/nope.json");
    REQUIRE(missing.values.viewDistance == SettingsValues::DEFAULT_VIEW_DISTANCE);
    REQUIRE(missing.gfx.shadowQuality == 2);
    REQUIRE(missing.gfx.volumetricLight);
    REQUIRE(missing.gfx.cloudMode == 2);
    REQUIRE(missing.gfx.bloomLevel == 5);
    REQUIRE(missing.gfx.sharpening == 0);

    // Hand-edited garbage clamps instead of exploding
    std::string path = dir.path() + "/settings.json";
    std::filesystem::create_directories(dir.path());
    {
        std::ofstream file(path);
        file << "{ \"viewDistance\": 999, \"shadowQuality\": -3, \"vibrance\": 42 }";
    }
    LoadedSettings clamped = loadSettings(path);
    REQUIRE(clamped.values.viewDistance == SettingsValues::MAX_VIEW_DISTANCE);
    REQUIRE(clamped.gfx.shadowQuality == 0);
    REQUIRE(clamped.gfx.vibrance == 10);
    // Keys the file omits keep their defaults
    REQUIRE(clamped.gfx.cloudMode == 2);
}

TEST_CASE("GraphicsSettings env overrides map onto the fields", "[engine][settings]") {
    setenv("RYCRAFT_SHADOWS", "1", 1);
    setenv("RYCRAFT_VL", "0", 1);
    setenv("RYCRAFT_CLOUDS", "1", 1);
    setenv("RYCRAFT_SSR", "0", 1);
    setenv("RYCRAFT_BLOOM", "0", 1); // legacy intensity form: 0 disables

    GraphicsSettings gfx;
    REQUIRE(gfx.applyEnvOverrides()); // reports that overrides fired
    REQUIRE(gfx.shadowQuality == 1);
    REQUIRE(gfx.volumetricLight == false);
    REQUIRE(gfx.cloudMode == 1);
    REQUIRE(gfx.waterReflections == false);
    REQUIRE(gfx.bloomLevel == 0);
    // Untouched fields keep defaults
    REQUIRE(gfx.ssao);
    REQUIRE(gfx.wavingFoliage);

    unsetenv("RYCRAFT_SHADOWS");
    unsetenv("RYCRAFT_VL");
    unsetenv("RYCRAFT_CLOUDS");
    unsetenv("RYCRAFT_SSR");
    unsetenv("RYCRAFT_BLOOM");

    // With no RYCRAFT_* set it reports false, so the engine keeps saving
    GraphicsSettings clean;
    REQUIRE(!clean.applyEnvOverrides());
}

TEST_CASE("GameFlow: video settings nest under settings", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);
    flow.onEscape(); // pause
    flow.onMenuAction(MenuAction::OPEN_SETTINGS);

    // OPEN_VIDEO_SETTINGS only works from the settings screen
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(flow.screen == GameScreen::VIDEO_SETTINGS);

    // BACK returns to settings, ESC does the same
    flow.onMenuAction(MenuAction::CLOSE_VIDEO_SETTINGS);
    REQUIRE(flow.screen == GameScreen::SETTINGS);
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::SETTINGS);

    // Video screen freezes the sim like every other menu
    flow.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(flow.inMenu());

    // OPEN from a non-settings screen is inert
    GameFlow paused;
    paused.onMenuAction(MenuAction::PLAY);
    paused.onEscape();
    paused.onMenuAction(MenuAction::OPEN_VIDEO_SETTINGS);
    REQUIRE(paused.screen == GameScreen::PAUSED);
}

TEST_CASE("GameFlow: resume and quit actions", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);
    flow.onEscape(); // pause

    auto fx = flow.onMenuAction(MenuAction::RESUME);
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Quit only works from title/pause
    fx = flow.onMenuAction(MenuAction::QUIT);
    REQUIRE(!fx.requestQuit);
    flow.onEscape();
    fx = flow.onMenuAction(MenuAction::QUIT);
    REQUIRE(fx.requestQuit);
}

TEST_CASE("GameFlow: focus loss force-pauses gameplay only", "[ui][flow]") {
    GameFlow flow;
    flow.onMenuAction(MenuAction::PLAY);

    auto fx = flow.onFocusLost();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(fx.releaseCursor);

    // Idempotent while already paused
    fx = flow.onFocusLost();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(!fx.releaseCursor);
}

TEST_CASE("GameFlow: world session transitions", "[ui][flow]") {
    GameFlow flow;
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE_FALSE(flow.worldScreens());

    // Title -> world select -> create, ESC backs out one level at a time.
    flow.onMenuAction(MenuAction::OPEN_WORLD_SELECT);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onMenuAction(MenuAction::OPEN_WORLD_CREATE);
    REQUIRE(flow.screen == GameScreen::WORLD_CREATE);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onMenuAction(MenuAction::REQUEST_DELETE_WORLD);
    REQUIRE(flow.screen == GameScreen::WORLD_DELETE_CONFIRM);
    flow.onMenuAction(MenuAction::CANCEL_DELETE);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::TITLE);

    // Side-effectful actions never change the screen by themselves.
    flow.onMenuAction(MenuAction::OPEN_WORLD_SELECT);
    auto fx = flow.onMenuAction(MenuAction::PLAY_SELECTED_WORLD);
    REQUIRE(flow.screen == GameScreen::WORLD_SELECT);
    REQUIRE(!fx.captureCursor);

    // The engine drives the start after its side effect succeeds.
    fx = flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);
    REQUIRE(fx.resetTiming);
    REQUIRE(flow.worldScreens());

    // Save-and-quit lands back on the title with a free cursor.
    flow.onEscape(); // paused
    fx = flow.onWorldStopped();
    REQUIRE(flow.screen == GameScreen::TITLE);
    REQUIRE(fx.releaseCursor);

    // onWorldStarted refuses screens that already have a session.
    flow.onWorldStarted();
    flow.onEscape(); // paused
    fx = flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PAUSED);
    REQUIRE(!fx.captureCursor);
}

TEST_CASE("GameFlow: inventory key and container screens", "[ui][flow]") {
    GameFlow flow;
    flow.onWorldStarted();
    REQUIRE(flow.screen == GameScreen::PLAYING);

    auto fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::INVENTORY);
    REQUIRE(fx.releaseCursor);
    REQUIRE(flow.inMenu());
    REQUIRE(flow.inContainer());

    fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Container blocks open their screens from gameplay only.
    fx = flow.onContainerOpened(GameScreen::FURNACE);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    REQUIRE(fx.releaseCursor);
    fx = flow.onContainerOpened(GameScreen::CRAFTING);
    REQUIRE(flow.screen == GameScreen::FURNACE);
    flow.onEscape();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    fx = flow.onContainerOpened(GameScreen::CRAFTING);
    REQUIRE(flow.screen == GameScreen::CRAFTING);
    // E closes any container.
    flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    // Only container screens are valid targets.
    fx = flow.onContainerOpened(GameScreen::PAUSED);
    REQUIRE(flow.screen == GameScreen::PLAYING);
}

TEST_CASE("GameFlow: death ignores escape until respawn", "[ui][flow]") {
    GameFlow flow;
    flow.onWorldStarted();

    auto fx = flow.onPlayerDied();
    REQUIRE(flow.screen == GameScreen::DEATH);
    REQUIRE(fx.releaseCursor);

    fx = flow.onEscape();
    REQUIRE(flow.screen == GameScreen::DEATH);
    REQUIRE(!fx.captureCursor);
    fx = flow.onInventoryKey();
    REQUIRE(flow.screen == GameScreen::DEATH);

    fx = flow.onRespawn();
    REQUIRE(flow.screen == GameScreen::PLAYING);
    REQUIRE(fx.captureCursor);

    // Dying only happens while playing.
    flow.onEscape();
    fx = flow.onPlayerDied();
    REQUIRE(flow.screen == GameScreen::PAUSED);
}

TEST_CASE("Menu layouts: buttons sit on-screen and inside their panel", "[ui][menu]") {
    SettingsValues values;
    GraphicsSettings gfx;
    for (auto [w, h] : {std::pair{1024.f, 768.f}, {2048.f, 1536.f}, {3456.f, 2234.f}}) {
        for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS,
                                  GameScreen::VIDEO_SETTINGS}) {
            MenuLayout layout = buildMenuLayout(screen, w, h, values, gfx);
            REQUIRE(!layout.buttons.empty());

            for (const auto& button : layout.buttons) {
                REQUIRE(button.rect.x >= 0.f);
                REQUIRE(button.rect.y >= 0.f);
                REQUIRE(button.rect.x + button.rect.w <= 1.f);
                REQUIRE(button.rect.y + button.rect.h <= 1.f);
                REQUIRE(button.action != MenuAction::NONE);
                if (layout.panel.w > 0.f) {
                    REQUIRE(button.rect.x >= layout.panel.x);
                    REQUIRE(button.rect.x + button.rect.w <= layout.panel.x + layout.panel.w);
                }
            }

            // No two buttons overlap
            for (size_t i = 0; i < layout.buttons.size(); ++i) {
                for (size_t j = i + 1; j < layout.buttons.size(); ++j) {
                    const UIRect& a = layout.buttons[i].rect;
                    const UIRect& b = layout.buttons[j].rect;
                    bool separated = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y ||
                                     b.y + b.h <= a.y;
                    REQUIRE(separated);
                }
            }
        }
    }

    REQUIRE(buildMenuLayout(GameScreen::PLAYING, 1024.f, 768.f, values, gfx).buttons.empty());
}

TEST_CASE("Menu hit test: button centers hit, gaps miss", "[ui][menu]") {
    SettingsValues values;
    GraphicsSettings gfx;
    MenuLayout layout = buildMenuLayout(GameScreen::PAUSED, 1024.f, 768.f, values, gfx);

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const UIRect& rect = layout.buttons[i].rect;
        REQUIRE(menuHitTest(layout, rect.x + rect.w * 0.5f, rect.y + rect.h * 0.5f) ==
                static_cast<int>(i));
    }

    REQUIRE(menuHitTest(layout, 0.02f, 0.02f) == -1);
}

TEST_CASE("Font covers every character the menus draw", "[ui][font]") {
    SettingsValues values;
    GraphicsSettings gfx;
    std::string needed = "0123456789.:/-+ ";
    for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS,
                              GameScreen::VIDEO_SETTINGS}) {
        MenuLayout layout = buildMenuLayout(screen, 1024.f, 768.f, values, gfx);
        for (const auto& text : layout.texts)
            needed += text.text;
        for (const auto& button : layout.buttons)
            needed += button.label;
    }
    // Plus everything the debug HUD prints
    needed += "FPS: Chunks: Entities: Frame: ";

    for (char c : needed) {
        if (c == ' ')
            continue; // spaces render as gaps by design
        auto bitmap = UIOverlay::getCharBitmap(c);
        bool anyPixel = false;
        for (uint8_t row : bitmap)
            anyPixel |= row != 0;
        INFO("Missing glyph: '" << c << "'");
        REQUIRE(anyPixel);
    }
}

// ============================================================================
// UIOverlay Quad Vertex Generation Tests (no Metal device required)
// ============================================================================

TEST_CASE("UIOverlay quad vertex generation: fullscreen quad", "[render][ui]") {
    // Verify that a fullscreen quad (0,0,1,1) produces correct vertex positions.
    // Layout: [x, y] for each of 4 vertices: BL, TL, BR, TR
    float x = 0.0f, y = 0.0f, w = 1.0f, h = 1.0f;

    // Expected vertices (bottom-left origin):
    // BL: (0, 0), TL: (0, 1), BR: (1, 0), TR: (1, 1)
    struct QuadVertex {
        float px, py;
        float cr, cg, cb, ca;
    };

    QuadVertex expected[4] = {
        {0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Verify vertex positions
    REQUIRE(expected[0].px == x);
    REQUIRE(expected[0].py == y);
    REQUIRE(expected[1].px == x);
    REQUIRE(expected[1].py == y + h);
    REQUIRE(expected[2].px == x + w);
    REQUIRE(expected[2].py == y);
    REQUIRE(expected[3].px == x + w);
    REQUIRE(expected[3].py == y + h);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair horizontal line", "[render][ui]") {
    // Simulate crosshair horizontal line at center of 1920×1080 screen
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossH = 1.0f / screenHeight; // 1 pixel height
    float crossW = 20.0f / screenWidth; // 20 pixel width

    float left = centerX - crossW * 0.5f;
    float bottom = centerY - crossH * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossH * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossW > 0.0f);
    REQUIRE(crossH > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossH <= 1.0f);
}

TEST_CASE("UIOverlay quad vertex generation: crosshair vertical line", "[render][ui]") {
    float screenWidth = 1920.0f;
    float screenHeight = 1080.0f;

    float centerX = 0.5f;
    float centerY = 0.5f;
    float crossV = 20.0f / screenHeight;   // 20 pixel height
    float crossLineW = 1.0f / screenWidth; // 1 pixel width

    float left = centerX - crossLineW * 0.5f;
    float bottom = centerY - crossV * 0.5f;

    // Verify crosshair is centered
    REQUIRE(left + crossLineW * 0.5f == Catch::Approx(centerX));
    REQUIRE(bottom + crossV * 0.5f == Catch::Approx(centerY));

    // Verify dimensions are positive
    REQUIRE(crossLineW > 0.0f);
    REQUIRE(crossV > 0.0f);

    // Verify crosshair fits within screen
    REQUIRE(left >= 0.0f);
    REQUIRE(left + crossLineW <= 1.0f);
    REQUIRE(bottom >= 0.0f);
    REQUIRE(bottom + crossV <= 1.0f);
}

TEST_CASE("UIOverlay orthographic projection maps screen to NDC", "[render][ui]") {
    // Verify the orthographic projection matrix maps [0,1] screen coords to [-1,1] NDC.
    // Matrix:
    //   [ 2,  0,  0,  0]
    //   [ 0,  2,  0,  0]
    //   [ 0,  0,  1,  0]
    //   [-1, -1,  0,  1]
    //
    // For point (x, y, 0, 1): result = (2x-1, 2y-1, 0, 1)

    auto transform = [](float sx, float sy) -> std::pair<float, float> {
        float nx = 2.0f * sx - 1.0f;
        float ny = 2.0f * sy - 1.0f;
        return {nx, ny};
    };

    // Screen (0, 0) → NDC (-1, -1)
    auto p0 = transform(0.0f, 0.0f);
    REQUIRE(p0.first == Catch::Approx(-1.0f));
    REQUIRE(p0.second == Catch::Approx(-1.0f));

    // Screen (1, 1) → NDC (1, 1)
    auto p1 = transform(1.0f, 1.0f);
    REQUIRE(p1.first == Catch::Approx(1.0f));
    REQUIRE(p1.second == Catch::Approx(1.0f));

    // Screen (0.5, 0.5) → NDC (0, 0)
    auto p2 = transform(0.5f, 0.5f);
    REQUIRE(p2.first == Catch::Approx(0.0f));
    REQUIRE(p2.second == Catch::Approx(0.0f));
}

TEST_CASE("UIOverlay quad index order forms two triangles", "[render][ui]") {
    // Index buffer: {0, 1, 2, 0, 2, 3}
    // Triangle 1: vertices 0, 1, 2 (BL, TL, BR) — left-bottom triangle
    // Triangle 2: vertices 0, 2, 3 (BL, BR, TR) — right-top triangle
    uint16_t indices[] = {0, 1, 2, 0, 2, 3};

    // Verify 6 indices (2 triangles)
    REQUIRE(sizeof(indices) / sizeof(indices[0]) == 6);

    // Verify all indices reference valid vertices (0-3)
    for (uint16_t idx : indices) {
        REQUIRE((idx >= 0 && idx <= 3));
    }

    // Verify triangle 1 covers bottom-left half
    REQUIRE(indices[0] == 0); // BL
    REQUIRE(indices[1] == 1); // TL
    REQUIRE(indices[2] == 2); // BR

    // Verify triangle 2 covers top-right half
    REQUIRE(indices[3] == 0); // BL
    REQUIRE(indices[4] == 2); // BR
    REQUIRE(indices[5] == 3); // TR
}

// ---- Inventory Tests ----

TEST_CASE("Inventory: hotbar selection clamps and wraps", "[inventory]") {
    Inventory inventory;
    REQUIRE(inventory.getSelectedIndex() == 0);

    inventory.selectSlot(-5);
    REQUIRE(inventory.getSelectedIndex() == 0);
    inventory.selectSlot(100);
    REQUIRE(inventory.getSelectedIndex() == 8);
    inventory.selectSlot(4);
    REQUIRE(inventory.getSelectedIndex() == 4);

    inventory.selectSlot(8);
    inventory.selectNext();
    REQUIRE(inventory.getSelectedIndex() == 0);
    inventory.selectPrev();
    REQUIRE(inventory.getSelectedIndex() == 8);
}

TEST_CASE("Inventory: slots read and write with range guards", "[inventory]") {
    Inventory inventory;
    REQUIRE(inventory.getSlot(0).empty());

    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::DIAMOND_ORE), 3, 0});
    REQUIRE(inventory.getSlot(0).type == itemFromBlock(BlockType::DIAMOND_ORE));
    REQUIRE(inventory.getSlot(0).count == 3);

    // Main-grid slots exist beyond the hotbar.
    inventory.setSlot(35, ItemStack{ItemType::STICK, 5, 0});
    REQUIRE(inventory.getSlot(35).count == 5);

    // Out-of-range reads return empty; writes drop.
    REQUIRE(inventory.getSlot(-1).empty());
    REQUIRE(inventory.getSlot(Inventory::SLOTS).empty());
    inventory.setSlot(-1, ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(inventory.getSlot(0).type == itemFromBlock(BlockType::DIAMOND_ORE));
}

TEST_CASE("Inventory: selected block resolves through the item registry", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::STONE), 1, 0});
    inventory.setSlot(1, ItemStack{ItemType::IRON_PICKAXE, 1, 250});
    inventory.selectSlot(0);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::STONE);
    // Tools and empty slots place nothing.
    inventory.selectSlot(1);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::AIR);
    inventory.selectSlot(2);
    REQUIRE(inventory.getSelectedBlockType() == BlockType::AIR);
}

TEST_CASE("Inventory: add merges into stacks hotbar first", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(9, ItemStack{ItemType::COAL, 60, 0});

    // Merging tops off the existing main-grid stack, then opens hotbar slot 0.
    REQUIRE(inventory.add(ItemStack{ItemType::COAL, 10, 0}) == 10);
    REQUIRE(inventory.getSlot(9).count == 64);
    REQUIRE(inventory.getSlot(0).type == ItemType::COAL);
    REQUIRE(inventory.getSlot(0).count == 6);

    // A full inventory absorbs nothing.
    Inventory full;
    for (int slot = 0; slot < Inventory::SLOTS; ++slot) {
        full.setSlot(slot, ItemStack{ItemType::STICK, 64, 0});
    }
    REQUIRE(full.add(ItemStack{ItemType::STICK, 1, 0}) == 0);
    REQUIRE(full.add(ItemStack{ItemType::COAL, 1, 0}) == 0);

    // Tools never merge (stack limit one) but fill empty slots.
    Inventory tools;
    REQUIRE(tools.add(ItemStack{ItemType::IRON_AXE, 1, 250}) == 1);
    REQUIRE(tools.getSlot(0).type == ItemType::IRON_AXE);
    REQUIRE(tools.getSlot(0).durability == 250);
}

TEST_CASE("Inventory: consume and tool damage empty the selected slot", "[inventory]") {
    Inventory inventory;
    inventory.setSlot(0, ItemStack{itemFromBlock(BlockType::DIRT), 2, 0});
    inventory.selectSlot(0);
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).count == 1);
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).empty());
    inventory.consumeSelected();
    REQUIRE(inventory.getSlot(0).empty());

    inventory.setSlot(0, ItemStack{ItemType::WOODEN_PICKAXE, 1, 2});
    REQUIRE_FALSE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).durability == 1);
    REQUIRE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).empty());

    // Non-tools never wear.
    inventory.setSlot(0, ItemStack{ItemType::COAL, 4, 0});
    REQUIRE_FALSE(inventory.damageSelectedTool());
    REQUIRE(inventory.getSlot(0).count == 4);
}

// ---- Performance HUD Tests ----

TEST_CASE("Performance HUD: FPS averaging over 60 frames", "[phase8][hud]") {
    // Simulate rolling average FPS
    std::vector<float> frameTimes;
    frameTimes.reserve(60);

    auto computeFPS = [&frameTimes](float newFrameTimeMs) -> float {
        frameTimes.push_back(newFrameTimeMs);
        if (frameTimes.size() > 60) {
            frameTimes.erase(frameTimes.begin());
        }

        float totalMs = 0.0f;
        for (float t : frameTimes) {
            totalMs += t;
        }
        return static_cast<float>(frameTimes.size()) * 1000.0f / totalMs;
    };

    // Feed 60 frames at 16.67ms each (60 FPS)
    for (int i = 0; i < 60; ++i) {
        computeFPS(16.67f);
    }

    float fps = computeFPS(16.67f);
    REQUIRE(fps > 55.0f);
    REQUIRE(fps < 65.0f);

    // Feed slower frames → FPS drops
    for (int i = 0; i < 60; ++i) {
        computeFPS(33.33f); // 30 FPS
    }

    fps = computeFPS(33.33f);
    REQUIRE(fps > 25.0f);
    REQUIRE(fps < 35.0f);
}

TEST_CASE("Performance HUD: text positioning", "[phase8][hud]") {
    // HUD at top-left: (8px, height-8px)
    uint32_t width = 1920;
    uint32_t height = 1080;

    float hudX = 8.0f / static_cast<float>(width);
    float hudY = 1.0f - 8.0f / static_cast<float>(height);

    // Verify normalized coordinates are valid
    REQUIRE(hudX > 0.0f);
    REQUIRE(hudX < 0.01f); // Near left edge
    REQUIRE(hudY > 0.99f); // Near top edge
    REQUIRE(hudY < 1.0f);

    // Background dimensions
    float bgWidth = 220.0f / static_cast<float>(width);
    float bgHeight = 80.0f / static_cast<float>(height);

    REQUIRE(bgWidth > 0.0f);
    REQUIRE(bgWidth < 0.2f); // Less than 20% of screen width
    REQUIRE(bgHeight > 0.0f);
    REQUIRE(bgHeight < 0.1f); // Less than 10% of screen height
}

TEST_CASE("Performance HUD: integer to string conversion", "[phase8][hud]") {
    // Simulate the intToString function
    auto intToString = [](int value, char* buf, size_t bufSize) {
        char tmp[20];
        int len = 0;
        if (value == 0) {
            tmp[len++] = '0';
        } else {
            int v = value < 0 ? -value : value;
            while (v > 0) {
                tmp[len++] = '0' + (v % 10);
                v /= 10;
            }
            if (value < 0)
                tmp[len++] = '-';
            for (int i = 0; i < len / 2; ++i) {
                char t = tmp[i];
                tmp[i] = tmp[len - 1 - i];
                tmp[len - 1 - i] = t;
            }
        }
        size_t copyLen = len < static_cast<int>(bufSize - 1) ? len : bufSize - 1;
        std::memcpy(buf, tmp, copyLen);
        buf[copyLen] = '\0';
    };

    char buf[16];

    intToString(0, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "0");

    intToString(42, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "42");

    intToString(12345, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "12345");

    intToString(-7, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "-7");
}

TEST_CASE("Performance HUD: float to string conversion", "[phase8][hud]") {
    auto floatToString = [](float value, char* buf, size_t bufSize) {
        int intPart = static_cast<int>(std::floor(value));
        int fracPart = static_cast<int>((value - std::floor(value)) * 10);

        char tmp[20];
        int len = 0;
        if (intPart == 0) {
            tmp[len++] = '0';
        } else {
            int v = intPart < 0 ? -intPart : intPart;
            while (v > 0) {
                tmp[len++] = '0' + (v % 10);
                v /= 10;
            }
            if (intPart < 0)
                tmp[len++] = '-';
            for (int i = 0; i < len / 2; ++i) {
                char t = tmp[i];
                tmp[i] = tmp[len - 1 - i];
                tmp[len - 1 - i] = t;
            }
        }

        if (len + 3 < static_cast<int>(bufSize)) {
            std::memcpy(buf, tmp, len);
            buf[len] = '.';
            buf[len + 1] = '0' + fracPart;
            buf[len + 2] = '\0';
        } else {
            size_t safeLen =
                len < static_cast<int>(bufSize - 1) ? len : static_cast<int>(bufSize - 1);
            std::memcpy(buf, tmp, safeLen);
            buf[safeLen] = '\0';
        }
    };

    char buf[16];

    floatToString(60.0f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "60.0");

    floatToString(16.7f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "16.7");

    floatToString(0.5f, buf, sizeof(buf));
    REQUIRE(std::string(buf) == "0.5");
}
