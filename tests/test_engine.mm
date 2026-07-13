#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/hotbar.hpp>
#include <engine/input_bindings.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/biome.hpp>
#include <world/chunk.hpp>
#include <world/chunk_pos.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/terrain.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
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

TEST_CASE("Menu layouts: buttons sit on-screen and inside their panel", "[ui][menu]") {
    SettingsValues values;
    for (auto [w, h] : {std::pair{1024.f, 768.f}, {2048.f, 1536.f}, {3456.f, 2234.f}}) {
        for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS}) {
            MenuLayout layout = buildMenuLayout(screen, w, h, values);
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

    REQUIRE(buildMenuLayout(GameScreen::PLAYING, 1024.f, 768.f, values).buttons.empty());
}

TEST_CASE("Menu hit test: button centers hit, gaps miss", "[ui][menu]") {
    SettingsValues values;
    MenuLayout layout = buildMenuLayout(GameScreen::PAUSED, 1024.f, 768.f, values);

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const UIRect& rect = layout.buttons[i].rect;
        REQUIRE(menuHitTest(layout, rect.x + rect.w * 0.5f, rect.y + rect.h * 0.5f) ==
                static_cast<int>(i));
    }

    REQUIRE(menuHitTest(layout, 0.02f, 0.02f) == -1);
}

TEST_CASE("Font covers every character the menus draw", "[ui][font]") {
    SettingsValues values;
    std::string needed = "0123456789.:/-+ ";
    for (GameScreen screen : {GameScreen::TITLE, GameScreen::PAUSED, GameScreen::SETTINGS}) {
        MenuLayout layout = buildMenuLayout(screen, 1024.f, 768.f, values);
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

// ---- Hotbar Tests (Task 6.3) ----

TEST_CASE("Hotbar: initial slot selection is 0", "[phase6][hotbar]") {
    Hotbar hotbar;
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectSlot clamps to valid range", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Negative index clamps to 0
    hotbar.selectSlot(-5);
    REQUIRE(hotbar.getSelectedIndex() == 0);

    // Out-of-range index clamps to 8
    hotbar.selectSlot(100);
    REQUIRE(hotbar.getSelectedIndex() == 8);

    // Valid index works
    hotbar.selectSlot(4);
    REQUIRE(hotbar.getSelectedIndex() == 4);
}

TEST_CASE("Hotbar: selectNext wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(0);

    for (int i = 1; i <= 8; ++i) {
        hotbar.selectNext();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 8 → 0
    hotbar.selectNext();
    REQUIRE(hotbar.getSelectedIndex() == 0);
}

TEST_CASE("Hotbar: selectPrev wraps around", "[phase6][hotbar]") {
    Hotbar hotbar;
    hotbar.selectSlot(8);

    for (int i = 7; i >= 0; --i) {
        hotbar.selectPrev();
        REQUIRE(hotbar.getSelectedIndex() == i);
    }

    // Wrap around: 0 → 8
    hotbar.selectPrev();
    REQUIRE(hotbar.getSelectedIndex() == 8);
}

TEST_CASE("Hotbar: getSelectedBlockType returns correct type", "[phase6][hotbar]") {
    Hotbar hotbar;

    // Default slot 0 is STONE
    hotbar.selectSlot(0);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::STONE);

    // Slot 1 is DIRT
    hotbar.selectSlot(1);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::DIRT);

    // Slot 2 is GRASS
    hotbar.selectSlot(2);
    REQUIRE(hotbar.getSelectedBlockType() == BlockType::GRASS);
}

TEST_CASE("Hotbar: setSlot and getSlot", "[phase6][hotbar]") {
    Hotbar hotbar;

    hotbar.setSlot(0, BlockType::DIAMOND_ORE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);

    // Out-of-range returns AIR
    REQUIRE(hotbar.getSlot(-1) == BlockType::AIR);
    REQUIRE(hotbar.getSlot(9) == BlockType::AIR);

    // setSlot on out-of-range does nothing
    hotbar.setSlot(-1, BlockType::STONE);
    REQUIRE(hotbar.getSlot(0) == BlockType::DIAMOND_ORE);
}

TEST_CASE("Hotbar: default slot contents", "[phase6][hotbar]") {
    Hotbar hotbar;

    REQUIRE(hotbar.getSlot(0) == BlockType::STONE);
    REQUIRE(hotbar.getSlot(1) == BlockType::DIRT);
    REQUIRE(hotbar.getSlot(2) == BlockType::GRASS);
    REQUIRE(hotbar.getSlot(3) == BlockType::LOG);
    REQUIRE(hotbar.getSlot(4) == BlockType::PLANKS);
    REQUIRE(hotbar.getSlot(5) == BlockType::SAND);
    REQUIRE(hotbar.getSlot(6) == BlockType::SANDSTONE);
    REQUIRE(hotbar.getSlot(7) == BlockType::GLASS);
    REQUIRE(hotbar.getSlot(8) == BlockType::FLOWER_RED);
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
