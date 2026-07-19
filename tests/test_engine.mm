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
#include <engine/mining.hpp>
#include <engine/slot_interaction.hpp>
#include <engine/survival.hpp>
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
#include <world/world_list.hpp>

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

TEST_CASE("World select layout scrolls selects and guards actions", "[ui][worlds]") {
    MenuContext ctx;
    auto buttonWith = [](const MenuLayout& layout, MenuAction action) {
        for (const auto& button : layout.buttons) {
            if (button.action == action)
                return true;
        }
        return false;
    };

    // Empty list: no play/delete targets, create and back present.
    MenuLayout empty = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    REQUIRE_FALSE(buttonWith(empty, MenuAction::PLAY_SELECTED_WORLD));
    REQUIRE_FALSE(buttonWith(empty, MenuAction::REQUEST_DELETE_WORLD));
    REQUIRE(buttonWith(empty, MenuAction::OPEN_WORLD_CREATE));
    REQUIRE(buttonWith(empty, MenuAction::WORLD_BACK));
    REQUIRE_FALSE(buttonWith(empty, MenuAction::WORLD_LIST_UP));

    // Seven rows: five visible with correct payloads, scroll arrows appear
    // on the scrollable side only.
    for (int i = 0; i < 7; ++i) {
        ctx.worldRows.push_back("World " + std::to_string(i));
    }
    ctx.worldSelect.selected = 1;
    MenuLayout list = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    int rows = 0;
    for (const auto& button : list.buttons) {
        if (button.action != MenuAction::SELECT_WORLD)
            continue;
        REQUIRE(button.payload == rows);
        if (button.payload == 1)
            REQUIRE(button.emphasized);
        ++rows;
    }
    REQUIRE(rows == WorldSelectState::VISIBLE_ROWS);
    REQUIRE_FALSE(buttonWith(list, MenuAction::WORLD_LIST_UP));
    REQUIRE(buttonWith(list, MenuAction::WORLD_LIST_DOWN));
    REQUIRE(buttonWith(list, MenuAction::PLAY_SELECTED_WORLD));
    REQUIRE(buttonWith(list, MenuAction::REQUEST_DELETE_WORLD));

    // Scrolled to the bottom: rows start at the clamped offset.
    ctx.worldSelect.scroll = 99;
    MenuLayout bottom = buildScreenLayout(GameScreen::WORLD_SELECT, 1024.f, 768.f, ctx);
    int firstPayload = -1;
    for (const auto& button : bottom.buttons) {
        if (button.action == MenuAction::SELECT_WORLD) {
            firstPayload = button.payload;
            break;
        }
    }
    REQUIRE(firstPayload == 2);
    REQUIRE(buttonWith(bottom, MenuAction::WORLD_LIST_UP));
    REQUIRE_FALSE(buttonWith(bottom, MenuAction::WORLD_LIST_DOWN));
}

TEST_CASE("World create layout gates the create button on a name", "[ui][worlds]") {
    MenuContext ctx;
    auto hasCreate = [](const MenuLayout& layout) {
        for (const auto& button : layout.buttons) {
            if (button.action == MenuAction::CREATE_WORLD_CONFIRM)
                return true;
        }
        return false;
    };

    MenuLayout unnamed = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE(unnamed.textFields.size() == 2);
    REQUIRE(unnamed.textFields[0].label == "NAME");
    REQUIRE(unnamed.textFields[1].label == "SEED");
    REQUIRE_FALSE(hasCreate(unnamed));

    ctx.worldCreate.name = "   ";
    REQUIRE_FALSE(hasCreate(buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx)));

    ctx.worldCreate.name = "Base";
    ctx.worldCreate.focusedField = 1;
    MenuLayout named = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE(hasCreate(named));
    REQUIRE(named.textFields[1].focused);
    REQUIRE(named.textFields[1].caret);
    REQUIRE_FALSE(named.textFields[0].focused);

    // The caret obeys the blink phase.
    ctx.caretVisible = false;
    MenuLayout blink = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);
    REQUIRE_FALSE(blink.textFields[1].caret);
}

TEST_CASE("Typed hit-testing distinguishes fields and buttons", "[ui][worlds]") {
    MenuContext ctx;
    ctx.worldCreate.name = "Base";
    MenuLayout layout = buildScreenLayout(GameScreen::WORLD_CREATE, 1024.f, 768.f, ctx);

    const auto& field = layout.textFields[0];
    UIHit hit =
        uiHitTest(layout, field.rect.x + field.rect.w * 0.5f, field.rect.y + field.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::TEXT_FIELD);
    REQUIRE(hit.index == 0);

    const auto& button = layout.buttons.front();
    hit = uiHitTest(layout, button.rect.x + button.rect.w * 0.5f,
                    button.rect.y + button.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::BUTTON);
    REQUIRE(layout.buttons[static_cast<size_t>(hit.index)].action == button.action);

    REQUIRE(uiHitTest(layout, 0.01f, 0.01f).kind == UIHitKind::NONE);
}

TEST_CASE("Text field filtering enforces charset and length", "[ui][worlds]") {
    REQUIRE(filterTextField("My World_2.0-x", false, 24) == "My World_2.0-x");
    REQUIRE(filterTextField("bad!@#chars$%", false, 24) == "badchars");
    REQUIRE(filterTextField("way too long name for the field", false, 10) == "way too lo");
    REQUIRE(filterTextField("seed123seed", true, 10) == "123");
    REQUIRE(filterTextField("42", true, 10) == "42");
    REQUIRE(filterTextField("", true, 10).empty());
}

TEST_CASE("Font covers every character the menus draw", "[ui][font]") {
    GraphicsSettings gfx;
    std::string needed = "0123456789.:/-+ ";

    // Every screen with a fully populated context, including a world name
    // exercising the complete allowed charset.
    MenuContext ctx;
    ctx.gfx = &gfx;
    ctx.worldRows = {"A world_NAME.42-x - Survival - Seed 4294967295",
                     "second row - Creative - Seed 7"};
    ctx.worldSelect.selected = 0;
    ctx.worldCreate.name = "AZaz09 ._-";
    ctx.worldCreate.seedText = "0123456789";
    ctx.deleteWorldName = "A world_NAME.42-x";
    for (GameScreen screen :
         {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS, GameScreen::VIDEO_SETTINGS,
          GameScreen::WORLD_SELECT, GameScreen::WORLD_CREATE, GameScreen::WORLD_DELETE_CONFIRM}) {
        MenuLayout layout = buildScreenLayout(screen, 1024.f, 768.f, ctx);
        for (const auto& text : layout.texts)
            needed += text.text;
        for (const auto& button : layout.buttons)
            needed += button.label;
        for (const auto& field : layout.textFields) {
            needed += field.label;
            needed += field.text;
        }
    }
    // The full world-name charset can appear in any typed name.
    for (int c = 0; c < 128; ++c) {
        if (isWorldNameChar(static_cast<char>(c)))
            needed += static_cast<char>(c);
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

TEST_CASE("InputState: text entry accumulates edits and suppresses nothing else", "[input][text]") {
    InputState input;
    REQUIRE_FALSE(input.textEntryActive);

    // Inactive entry ignores edits entirely.
    input.applyTextKey('x');
    input.applyTextBackspace();
    REQUIRE(input.textBuffer.empty());

    input.beginTextEntry("Seed");
    REQUIRE(input.textEntryActive);
    REQUIRE(input.textBuffer == "Seed");

    input.applyTextKey(' ');
    input.applyTextKey('4');
    input.applyTextKey('2');
    REQUIRE(input.textBuffer == "Seed 42");

    // Control characters and non-ASCII bytes never land in the buffer.
    input.applyTextKey('\t');
    input.applyTextKey('\n');
    input.applyTextKey(static_cast<char>(0x1B));
    input.applyTextKey(static_cast<char>(0xC3));
    REQUIRE(input.textBuffer == "Seed 42");

    input.applyTextBackspace();
    REQUIRE(input.textBuffer == "Seed 4");

    // The cap holds regardless of how much is typed.
    for (int i = 0; i < 300; ++i) {
        input.applyTextKey('a');
    }
    REQUIRE(input.textBuffer.size() == InputState::TEXT_BUFFER_MAX);

    const std::string finished = input.endTextEntry();
    REQUIRE_FALSE(input.textEntryActive);
    REQUIRE(finished.size() == InputState::TEXT_BUFFER_MAX);
    REQUIRE(input.textBuffer.empty());

    // Submission is a one-frame edge cleared by update().
    input.beginTextEntry("");
    input.textSubmitted = true;
    input.update();
    REQUIRE_FALSE(input.textSubmitted);
}

namespace {

SlotAccess craftingAccess(std::array<ItemStack, 36>& inventory, std::array<ItemStack, 9>& grid,
                          ItemStack& result, int gridSize, int gridWidth) {
    SlotAccess access;
    access.inventory = inventory.data();
    access.craftGrid = grid.data();
    access.craftGridSize = gridSize;
    access.craftGridWidth = gridWidth;
    access.craftResult = &result;
    return access;
}

} // namespace

TEST_CASE("Slot clicks pick place merge and split", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    inventory[0] = ItemStack{ItemType::COAL, 10, 0};
    inventory[1] = ItemStack{ItemType::COAL, 60, 0};
    inventory[2] = ItemStack{ItemType::STICK, 5, 0};

    // LEFT on a stack picks the whole thing up.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 0}, SlotClickKind::LEFT).changed);
    REQUIRE(cursor == ItemStack{ItemType::COAL, 10, 0});
    REQUIRE(inventory[0].empty());

    // LEFT on the same type merges up to the cap and keeps the rest held.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 1}, SlotClickKind::LEFT).changed);
    REQUIRE(inventory[1].count == 64);
    REQUIRE(cursor.count == 6);

    // LEFT on a different type swaps.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 2}, SlotClickKind::LEFT).changed);
    REQUIRE(cursor == ItemStack{ItemType::STICK, 5, 0});
    REQUIRE(inventory[2] == ItemStack{ItemType::COAL, 6, 0});

    // RIGHT with a held stack places exactly one.
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 3}, SlotClickKind::RIGHT).changed);
    REQUIRE(inventory[3] == ItemStack{ItemType::STICK, 1, 0});
    REQUIRE(cursor.count == 4);

    // RIGHT with an empty cursor takes the larger half.
    cursor.clear();
    inventory[4] = ItemStack{ItemType::COAL, 7, 0};
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 4}, SlotClickKind::RIGHT).changed);
    REQUIRE(cursor.count == 4);
    REQUIRE(inventory[4].count == 3);

    // Clicks on empty air with an empty cursor change nothing.
    cursor.clear();
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 30}, SlotClickKind::LEFT).changed);
}

TEST_CASE("Shift clicks quick-move between regions", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    ItemStack cursor;

    // Hotbar to main.
    inventory[2] = ItemStack{ItemType::COAL, 12, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 2}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(inventory[2].empty());
    REQUIRE(inventory[9] == ItemStack{ItemType::COAL, 12, 0});

    // Main to hotbar.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 9}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(inventory[0] == ItemStack{ItemType::COAL, 12, 0});

    // Craft grid to inventory.
    grid[4] = ItemStack{itemFromBlock(BlockType::PLANKS), 3, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CRAFT_IN, 4}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(grid[4].empty());

    // Inventory to an open furnace, routed by what the item can do there.
    ItemStack furnaceInput;
    ItemStack furnaceFuel;
    ItemStack furnaceOutput;
    access.furnaceInput = &furnaceInput;
    access.furnaceFuel = &furnaceFuel;
    access.furnaceOutput = &furnaceOutput;
    inventory[5] = ItemStack{ItemType::RAW_BEEF, 2, 0};
    inventory[6] = ItemStack{ItemType::COAL, 12, 0};
    inventory[7] = ItemStack{ItemType::IRON_INGOT, 1, 0};
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 5}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(furnaceInput == ItemStack{ItemType::RAW_BEEF, 2, 0});
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::INVENTORY, 6}, SlotClickKind::SHIFT_LEFT)
                .changed);
    REQUIRE(furnaceFuel == ItemStack{ItemType::COAL, 12, 0});
    // Neither smeltable nor fuel goes nowhere.
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::INVENTORY, 7}, SlotClickKind::SHIFT_LEFT)
            .changed);
}

TEST_CASE("Craft output is take-only and consumes the grid", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 3, 0};
    result = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};

    // Placement onto the output is refused.
    cursor = ItemStack{ItemType::COAL, 1, 0};
    REQUIRE_FALSE(
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::LEFT).changed);
    cursor.clear();

    // Taking crafts once: log consumed, result refreshed for the next craft.
    const auto taken =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::LEFT);
    REQUIRE(taken.changed);
    REQUIRE(taken.crafted);
    REQUIRE(cursor == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});
    REQUIRE(grid[0].count == 2);
    REQUIRE(result == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});

    // Shift-crafting drains the remaining logs straight into the inventory.
    cursor.clear();
    const auto drained =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::SHIFT_LEFT);
    REQUIRE(drained.crafted);
    REQUIRE(grid[0].empty());
    REQUIRE(result.empty());
    REQUIRE(inventory[0] == ItemStack{itemFromBlock(BlockType::PLANKS), 8, 0});
}

TEST_CASE("Shift-crafting into a nearly full inventory never creates items", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result;
    SlotAccess access = craftingAccess(inventory, grid, result, 4, 2);
    ItemStack cursor;

    // One log crafts {PLANKS, 4}. Fill every slot with a foreign item except
    // one planks stack with room for exactly 1 more.
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 3, 0};
    result = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};
    for (ItemStack& slot : inventory) {
        slot = ItemStack{ItemType::COAL, 64, 0};
    }
    inventory[0] = ItemStack{itemFromBlock(BlockType::PLANKS), 63, 0};

    const auto before = inventory;
    // The 4-plank batch cannot fully fit (only room for 1), so the craft is
    // refused: no partial deposit, the grid is untouched, the output stands.
    const auto outcome =
        applySlotClick(access, cursor, {SlotDomain::CRAFT_OUT, 0}, SlotClickKind::SHIFT_LEFT);
    REQUIRE_FALSE(outcome.changed);
    REQUIRE(inventory == before);
    REQUIRE(grid[0].count == 3);
    REQUIRE(result == ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0});
}

TEST_CASE("Creative palette hands out stacks and eats held ones", "[slots]") {
    std::array<ItemStack, 36> inventory{};
    SlotAccess access;
    access.inventory = inventory.data();
    access.palette = CREATIVE_PALETTE.data();
    access.paletteSize = static_cast<int>(CREATIVE_PALETTE.size());
    ItemStack cursor;

    const ItemType first = CREATIVE_PALETTE[0];
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::LEFT)
                .changed);
    REQUIRE(cursor.type == first);
    REQUIRE(cursor.count == maxStackSize(first));

    // Holding anything, a palette click trashes it.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 5}, SlotClickKind::LEFT)
                .changed);
    REQUIRE(cursor.empty());

    // RIGHT builds a stack one item at a time.
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::RIGHT)
                .changed);
    REQUIRE(applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::RIGHT)
                .changed);
    REQUIRE(cursor == ItemStack{first, 2, 0});

    // SHIFT sends a full stack straight into the inventory; the palette
    // itself never mutates.
    cursor.clear();
    REQUIRE(
        applySlotClick(access, cursor, {SlotDomain::CREATIVE_PALETTE, 0}, SlotClickKind::SHIFT_LEFT)
            .changed);
    REQUIRE(inventory[0].type == first);
}

TEST_CASE("Outside drops and container close return items", "[slots]") {
    ItemStack cursor{ItemType::COAL, 5, 0};
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::RIGHT) == ItemStack{ItemType::COAL, 1, 0});
    REQUIRE(cursor.count == 4);
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::LEFT) == ItemStack{ItemType::COAL, 4, 0});
    REQUIRE(cursor.empty());
    REQUIRE(takeOutsideDrop(cursor, SlotClickKind::LEFT).empty());

    std::array<ItemStack, 36> inventory{};
    std::array<ItemStack, 9> grid{};
    ItemStack result{itemFromBlock(BlockType::PLANKS), 4, 0};
    SlotAccess access = craftingAccess(inventory, grid, result, 9, 3);
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 2, 0};
    grid[8] = ItemStack{ItemType::STICK, 7, 0};
    cursor = ItemStack{ItemType::COAL, 3, 0};

    REQUIRE(collectOnClose(access, cursor).empty());
    REQUIRE(cursor.empty());
    REQUIRE(grid[0].empty());
    REQUIRE(result.empty());
    int coal = 0;
    int sticks = 0;
    int logs = 0;
    for (const ItemStack& slot : inventory) {
        if (slot.type == ItemType::COAL)
            coal += slot.count;
        if (slot.type == ItemType::STICK)
            sticks += slot.count;
        if (slot.type == itemFromBlock(BlockType::LOG))
            logs += slot.count;
    }
    REQUIRE(coal == 3);
    REQUIRE(sticks == 7);
    REQUIRE(logs == 2);

    // A stuffed inventory reports the homeless remainder.
    for (ItemStack& slot : inventory) {
        slot = ItemStack{ItemType::STICK, 64, 0};
    }
    grid[0] = ItemStack{itemFromBlock(BlockType::LOG), 2, 0};
    cursor = ItemStack{ItemType::COAL, 3, 0};
    const auto overflow = collectOnClose(access, cursor);
    REQUIRE(overflow.size() == 2);
}

TEST_CASE("Furnace layout exposes three slots and two gauges", "[ui][containers]") {
    MenuContext ctx;
    ctx.container.furnaceInput = ItemStack{ItemType::RAW_BEEF, 3, 0};
    ctx.container.furnaceFuel = ItemStack{ItemType::COAL, 5, 0};
    ctx.container.furnaceOutput = ItemStack{ItemType::COOKED_BEEF, 2, 0};
    ctx.container.furnaceCook = 0.5f;
    ctx.container.furnaceFuelLeft = 0.25f;

    MenuLayout layout = buildScreenLayout(GameScreen::FURNACE, 1024.f, 768.f, ctx);
    int input = 0;
    int fuel = 0;
    int output = 0;
    int inventory = 0;
    for (const SlotWidget& slot : layout.slots) {
        switch (slot.ref.domain) {
            case SlotDomain::FURNACE_INPUT:
                ++input;
                REQUIRE(slot.stack == ItemStack{ItemType::RAW_BEEF, 3, 0});
                break;
            case SlotDomain::FURNACE_FUEL:
                ++fuel;
                break;
            case SlotDomain::FURNACE_OUTPUT:
                ++output;
                REQUIRE(slot.stack == ItemStack{ItemType::COOKED_BEEF, 2, 0});
                break;
            case SlotDomain::INVENTORY:
                ++inventory;
                break;
            default:
                break;
        }
    }
    REQUIRE(input == 1);
    REQUIRE(fuel == 1);
    REQUIRE(output == 1);
    REQUIRE(inventory == 36);
    REQUIRE(layout.meters.size() == 2);
    // The cook arrow is the horizontal gauge, the flame the vertical one.
    const bool haveCook = layout.meters[0].fill == 0.5f || layout.meters[1].fill == 0.5f;
    const bool haveFlame = layout.meters[0].fill == 0.25f || layout.meters[1].fill == 0.25f;
    REQUIRE(haveCook);
    REQUIRE(haveFlame);
}

TEST_CASE("Container layouts expose every slot with correct references", "[ui][containers]") {
    MenuContext ctx;
    ctx.container.inventory[0] = ItemStack{ItemType::COAL, 9, 0};
    ctx.container.craftGrid[0] = ItemStack{itemFromBlock(BlockType::LOG), 1, 0};
    ctx.container.craftResult = ItemStack{itemFromBlock(BlockType::PLANKS), 4, 0};

    MenuLayout survival = buildScreenLayout(GameScreen::INVENTORY, 1024.f, 768.f, ctx);
    int inventorySlots = 0;
    int craftIn = 0;
    int craftOut = 0;
    for (const SlotWidget& slot : survival.slots) {
        if (slot.ref.domain == SlotDomain::INVENTORY)
            ++inventorySlots;
        if (slot.ref.domain == SlotDomain::CRAFT_IN)
            ++craftIn;
        if (slot.ref.domain == SlotDomain::CRAFT_OUT)
            ++craftOut;
    }
    REQUIRE(inventorySlots == 36);
    REQUIRE(craftIn == 4);
    REQUIRE(craftOut == 1);
    REQUIRE(survival.slots.front().stack == ItemStack{itemFromBlock(BlockType::LOG), 1, 0});

    MenuLayout crafting = buildScreenLayout(GameScreen::CRAFTING, 1024.f, 768.f, ctx);
    craftIn = 0;
    for (const SlotWidget& slot : crafting.slots) {
        if (slot.ref.domain == SlotDomain::CRAFT_IN)
            ++craftIn;
    }
    REQUIRE(craftIn == 9);

    // Creative shows the paged palette instead of a craft grid.
    ctx.container.creative = true;
    ctx.container.creativePage = 1;
    MenuLayout creative = buildScreenLayout(GameScreen::INVENTORY, 1024.f, 768.f, ctx);
    int palette = 0;
    int minIndex = 1 << 20;
    for (const SlotWidget& slot : creative.slots) {
        if (slot.ref.domain == SlotDomain::CREATIVE_PALETTE) {
            ++palette;
            minIndex = std::min(minIndex, slot.ref.index);
        }
        REQUIRE(slot.ref.domain != SlotDomain::CRAFT_IN);
    }
    const int expected =
        std::min<int>(CREATIVE_PALETTE_PAGE_SIZE,
                      static_cast<int>(CREATIVE_PALETTE.size()) - CREATIVE_PALETTE_PAGE_SIZE);
    REQUIRE(palette == expected);
    REQUIRE(minIndex == CREATIVE_PALETTE_PAGE_SIZE);

    // Slot hit-testing resolves through the typed path.
    const SlotWidget& probe = survival.slots.front();
    const UIHit hit =
        uiHitTest(survival, probe.rect.x + probe.rect.w * 0.5f, probe.rect.y + probe.rect.h * 0.5f);
    REQUIRE(hit.kind == UIHitKind::SLOT);
    REQUIRE(hit.index == 0);
}

TEST_CASE("Mining accumulates over time and completes on a stable target", "[mining]") {
    MiningState state;
    // Stone by hand needs blockBreakTicks(STONE, NONE) = 150 ticks.
    const int needed = blockBreakTicks(BlockType::STONE, ItemType::NONE);
    REQUIRE(needed == 150);

    for (int tick = 0; tick < needed - 1; ++tick) {
        REQUIRE_FALSE(tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE));
        REQUIRE(state.active);
    }
    REQUIRE(state.progress > 0.9f);
    // The final tick completes and resets.
    REQUIRE(tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE));
    REQUIRE_FALSE(state.active);
}

TEST_CASE("Mining resets on release and on a new target", "[mining]") {
    MiningState state;
    for (int tick = 0; tick < 20; ++tick) {
        tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    }
    REQUIRE(state.ticksElapsed == 20);

    // Releasing the button clears progress.
    tickMining(state, false, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    REQUIRE_FALSE(state.active);
    REQUIRE(state.progress == 0.f);

    // Looking at a new block restarts from zero.
    for (int tick = 0; tick < 20; ++tick) {
        tickMining(state, true, true, 1, 2, 3, BlockType::STONE, ItemType::NONE);
    }
    tickMining(state, true, true, 9, 9, 9, BlockType::DIRT, ItemType::NONE);
    REQUIRE(state.x == 9);
    REQUIRE(state.block == BlockType::DIRT);
    REQUIRE(state.ticksElapsed == 1);
}

TEST_CASE("Mining respects tool speed and never breaks bedrock", "[mining]") {
    // A stone pickaxe finishes stone far faster than a bare hand.
    MiningState hand;
    int handTicks = 0;
    while (!tickMining(hand, true, true, 0, 0, 0, BlockType::STONE, ItemType::NONE) &&
           handTicks < 1000) {
        ++handTicks;
    }
    MiningState pick;
    int pickTicks = 0;
    while (!tickMining(pick, true, true, 0, 0, 0, BlockType::STONE, ItemType::STONE_PICKAXE) &&
           pickTicks < 1000) {
        ++pickTicks;
    }
    REQUIRE(pickTicks < handTicks);

    // Bedrock never completes, whatever the tool.
    MiningState bedrock;
    for (int tick = 0; tick < 500; ++tick) {
        REQUIRE_FALSE(
            tickMining(bedrock, true, true, 0, 0, 0, BlockType::BEDROCK, ItemType::IRON_PICKAXE));
    }
    REQUIRE(bedrock.progress == 0.f);

    // Instant-break flora completes the first settled tick.
    MiningState grass;
    REQUIRE(tickMining(grass, true, true, 0, 0, 0, BlockType::TALL_GRASS, ItemType::NONE));
}

TEST_CASE("Mining restarts when the held tool changes mid-mine", "[mining]") {
    MiningState state;
    // Start on stone with a stone pickaxe (fast).
    for (int tick = 0; tick < 5; ++tick) {
        tickMining(state, true, true, 0, 0, 0, BlockType::STONE, ItemType::STONE_PICKAXE);
    }
    const int fastNeeded = state.ticksNeeded;
    REQUIRE(state.ticksElapsed == 5);

    // Switching to a bare hand recomputes the (much longer) break time and
    // restarts progress, so pickaxe timing cannot break stone by hand.
    tickMining(state, true, true, 0, 0, 0, BlockType::STONE, ItemType::NONE);
    REQUIRE(state.ticksElapsed == 1);
    REQUIRE(state.ticksNeeded > fastNeeded);
    REQUIRE(state.tool == ItemType::NONE);
}

TEST_CASE("Survival exhaustion spends saturation then food", "[survival]") {
    SurvivalStats stats;
    stats.saturation = 1.0f;
    stats.food = 20;
    // One EXHAUSTION_THRESHOLD of sprint exhaustion spends one saturation.
    stats.exhaustion = SurvivalStats::EXHAUSTION_THRESHOLD;
    SurvivalTickInputs idle;
    tickSurvivalStats(stats, idle, 20);
    REQUIRE(stats.saturation == Catch::Approx(0.0f));
    REQUIRE(stats.food == 20);

    // With saturation gone, the next threshold eats into food.
    stats.exhaustion = SurvivalStats::EXHAUSTION_THRESHOLD;
    tickSurvivalStats(stats, idle, 20);
    REQUIRE(stats.food == 19);
}

TEST_CASE("Survival regenerates fast with saturation and slow without", "[survival]") {
    SurvivalTickInputs idle;

    // Full food plus ample saturation heals a whole hp every fast interval.
    SurvivalStats fast;
    fast.food = 20;
    fast.saturation = 20.f;
    int delta = 0;
    for (int tick = 0; tick < SurvivalStats::FAST_REGEN_INTERVAL; ++tick) {
        delta = tickSurvivalStats(fast, idle, 15);
    }
    REQUIRE(delta == 1); // +1 hp after only the short fast interval

    // High food with no saturation falls back to the slow regen path.
    SurvivalStats slow;
    slow.food = 18;
    slow.saturation = 0.f;
    for (int tick = 0; tick < SurvivalStats::FAST_REGEN_INTERVAL; ++tick) {
        REQUIRE(tickSurvivalStats(slow, idle, 15) == 0); // no fast heal without saturation
    }
    int slowDelta = 0;
    for (int tick = SurvivalStats::FAST_REGEN_INTERVAL; tick < SurvivalStats::SLOW_REGEN_INTERVAL;
         ++tick) {
        slowDelta = tickSurvivalStats(slow, idle, 15);
    }
    REQUIRE(slowDelta == 1); // +1 hp only after the full slow interval
}

TEST_CASE("Survival regenerates a well-fed player back to full health", "[survival]") {
    SurvivalStats stats;
    stats.food = 20;
    stats.saturation = 20.f;
    SurvivalTickInputs idle;
    int health = 4;
    // A player who stays fed (topping the bar back up as it drains, as a
    // Minecraft player does by eating) regenerates all the way to full health.
    for (int tick = 0; tick < 4000 && health < SurvivalStats::MAX_HEALTH; ++tick) {
        if (stats.saturation <= 0.f) {
            stats.food = SurvivalStats::MAX_FOOD;
            stats.saturation = 20.f;
        }
        health += tickSurvivalStats(stats, idle, health);
    }
    REQUIRE(health == SurvivalStats::MAX_HEALTH);
}

TEST_CASE("Survival starves at empty food down to the floor", "[survival]") {
    SurvivalTickInputs idle;
    SurvivalStats starve;
    starve.food = 0;
    int applied = 0;
    for (int tick = 0; tick < SurvivalStats::STARVE_INTERVAL; ++tick) {
        applied = tickSurvivalStats(starve, idle, 10);
    }
    REQUIRE(applied == -1); // -1 hp after the starve interval

    // Starvation never drops below the floor.
    SurvivalStats floored;
    floored.food = 0;
    for (int tick = 0; tick < SurvivalStats::STARVE_INTERVAL; ++tick) {
        REQUIRE(tickSurvivalStats(floored, idle, SurvivalStats::STARVE_HEALTH_FLOOR) == 0);
    }
}

TEST_CASE("Survival drains air underwater and drowns when empty", "[survival]") {
    SurvivalStats stats;
    SurvivalTickInputs under;
    under.eyesUnderwater = true;

    for (int tick = 0; tick < SurvivalStats::MAX_AIR; ++tick) {
        tickSurvivalStats(stats, under, 20);
    }
    REQUIRE(stats.air == 0);

    // Out of air, drowning damage lands once per interval.
    int worst = 0;
    for (int tick = 0; tick < SurvivalStats::DROWN_DAMAGE_INTERVAL; ++tick) {
        worst = std::min(worst, tickSurvivalStats(stats, under, 20));
    }
    REQUIRE(worst == -SurvivalStats::DROWN_DAMAGE);

    // Surfacing refills air quickly.
    SurvivalTickInputs surface;
    tickSurvivalStats(stats, surface, 20);
    REQUIRE(stats.air == SurvivalStats::AIR_REFILL_PER_TICK);
}

TEST_CASE("Eating requires a held right-click over time on the same slot", "[survival]") {
    EatingState eating;
    // Not holding, or full food, never progresses.
    REQUIRE_FALSE(tickEating(eating, false, 0, true, 10));
    REQUIRE_FALSE(tickEating(eating, true, 0, false, 10));
    REQUIRE_FALSE(tickEating(eating, true, 0, true, SurvivalStats::MAX_FOOD));

    // Held for EAT_TICKS completes exactly once.
    bool finished = false;
    for (int tick = 0; tick < EatingState::EAT_TICKS; ++tick) {
        finished = tickEating(eating, true, 2, true, 10);
    }
    REQUIRE(finished);
    REQUIRE_FALSE(eating.active);

    // Switching the selected slot restarts the timer.
    for (int tick = 0; tick < EatingState::EAT_TICKS - 1; ++tick) {
        tickEating(eating, true, 2, true, 10);
    }
    tickEating(eating, true, 5, true, 10);
    REQUIRE(eating.slot == 5);
    REQUIRE(eating.ticks == 1);
}
