#include "render/ui_menu.hpp"

#include "render/graphics_settings.hpp"
#include "world/world_list.hpp"

#include <algorithm>
#include <array>

// Font metrics mirrored from UIOverlay (8×8 glyphs, 1px advance gap)
static constexpr float FONT_CELL = 9.0f;
static constexpr float FONT_HEIGHT = 8.0f;

float menuTextWidth(const std::string& text, float scale, float pixelWidth) {
    return static_cast<float>(text.size()) * FONT_CELL * scale / pixelWidth;
}

namespace {

// Shared building blocks, all sized against a 768-pt-tall reference window.
struct LayoutContext {
    float w, h; // window size in points
    float s;    // reference scale: 1.0 at 768-pt-tall

    float px(float v) const { return v * s / w; } // ref-pixels → normalized X
    float py(float v) const { return v * s / h; } // ref-pixels → normalized Y
};

void addCenteredText(MenuLayout& layout, const LayoutContext& ctx, const std::string& text,
                     float centerY, float scale, float r = 1.f, float g = 1.f, float b = 1.f) {
    float width = menuTextWidth(text, scale * ctx.s, ctx.w);
    layout.texts.push_back(MenuText{0.5f - width * 0.5f,
                                    centerY - ctx.py(FONT_HEIGHT * scale) * 0.5f, scale * ctx.s,
                                    text, r, g, b});
}

void addButton(MenuLayout& layout, const LayoutContext& ctx, const std::string& label,
               MenuAction action, float centerX, float centerY, float refW, float refH) {
    UIRect rect{centerX - ctx.px(refW) * 0.5f, centerY - ctx.py(refH) * 0.5f, ctx.px(refW),
                ctx.py(refH)};
    layout.buttons.push_back(MenuButton{rect, label, action});
}

// One settings row: label on the left, [-] value [+] on the right.
void addSettingsRow(MenuLayout& layout, const LayoutContext& ctx, const std::string& label,
                    const std::string& value, MenuAction down, MenuAction up, float centerY) {
    const float labelScale = 2.0f;
    layout.texts.push_back(MenuText{0.5f - ctx.px(250.f),
                                    centerY - ctx.py(FONT_HEIGHT * labelScale) * 0.5f,
                                    labelScale * ctx.s, label});

    const float stepperX = 0.5f + ctx.px(120.f);
    addButton(layout, ctx, "-", down, stepperX - ctx.px(82.f), centerY, 32.f, 32.f);
    float valueWidth = menuTextWidth(value, labelScale * ctx.s, ctx.w);
    layout.texts.push_back(MenuText{stepperX - valueWidth * 0.5f,
                                    centerY - ctx.py(FONT_HEIGHT * labelScale) * 0.5f,
                                    labelScale * ctx.s, value, 1.f, 0.9f, 0.4f});
    addButton(layout, ctx, "+", up, stepperX + ctx.px(82.f), centerY, 32.f, 32.f);
}

MenuLayout buildTitleLayout(const LayoutContext& ctx) {
    MenuLayout layout;
    layout.dimAlpha = 0.25f;

    addCenteredText(layout, ctx, "rycraft", 0.68f, 8.0f);
    addCenteredText(layout, ctx, "A voxel world built on Metal", 0.58f, 1.5f, 0.85f, 0.85f, 0.9f);

    addButton(layout, ctx, "PLAY", MenuAction::OPEN_WORLD_SELECT, 0.5f, 0.42f, 320.f, 48.f);
    addButton(layout, ctx, "QUIT", MenuAction::QUIT, 0.5f, 0.42f - ctx.py(68.f), 320.f, 48.f);
    return layout;
}

MenuLayout buildPauseLayout(const LayoutContext& ctx, GameMode mode) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(240.f), 0.5f - ctx.py(220.f), ctx.px(480.f), ctx.py(440.f)};

    addCenteredText(layout, ctx, "PAUSED", 0.5f + ctx.py(170.f), 3.0f);

    addButton(layout, ctx, "RESUME", MenuAction::RESUME, 0.5f, 0.5f + ctx.py(96.f), 360.f, 44.f);
    addButton(layout, ctx, "SETTINGS", MenuAction::OPEN_SETTINGS, 0.5f, 0.5f + ctx.py(40.f), 360.f,
              44.f);
    addSettingsRow(layout, ctx, "MODE", mode == GameMode::CREATIVE ? "CREATIVE" : "SURVIVAL",
                   MenuAction::TOGGLE_GAME_MODE, MenuAction::TOGGLE_GAME_MODE, 0.5f - ctx.py(20.f));
    addButton(layout, ctx, "SAVE AND QUIT TO TITLE", MenuAction::SAVE_QUIT_TO_TITLE, 0.5f,
              0.5f - ctx.py(80.f), 360.f, 44.f);
    addButton(layout, ctx, "QUIT", MenuAction::QUIT, 0.5f, 0.5f - ctx.py(136.f), 360.f, 44.f);
    return layout;
}

MenuLayout buildSettingsLayout(const LayoutContext& ctx, const SettingsValues& values) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(300.f), 0.5f - ctx.py(235.f), ctx.px(600.f), ctx.py(470.f)};

    addCenteredText(layout, ctx, "SETTINGS", 0.5f + ctx.py(185.f), 3.0f);

    addSettingsRow(layout, ctx, "RENDER DIST", std::to_string(values.viewDistance),
                   MenuAction::VIEW_DISTANCE_DOWN, MenuAction::VIEW_DISTANCE_UP,
                   0.5f + ctx.py(108.f));
    addSettingsRow(layout, ctx, "FOG", std::to_string(values.fogLevel), MenuAction::FOG_DOWN,
                   MenuAction::FOG_UP, 0.5f + ctx.py(52.f));
    addSettingsRow(layout, ctx, "SENSITIVITY", std::to_string(values.sensitivityLevel),
                   MenuAction::SENSITIVITY_DOWN, MenuAction::SENSITIVITY_UP, 0.5f - ctx.py(4.f));
    addSettingsRow(layout, ctx, "VOLUME", std::to_string(values.volumeLevel),
                   MenuAction::VOLUME_DOWN, MenuAction::VOLUME_UP, 0.5f - ctx.py(60.f));

    addButton(layout, ctx, "VIDEO...", MenuAction::OPEN_VIDEO_SETTINGS, 0.5f, 0.5f - ctx.py(124.f),
              320.f, 44.f);
    addButton(layout, ctx, "BACK", MenuAction::CLOSE_SETTINGS, 0.5f, 0.5f - ctx.py(184.f), 320.f,
              44.f);
    return layout;
}

// The video screen's ten per-effect rows. Toggle rows reuse the stepper
// widget with both arrows bound to the same TOGGLE action — no new widget,
// no second hit-test path.
MenuLayout buildVideoSettingsLayout(const LayoutContext& ctx, const GraphicsSettings& gfx) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(320.f), 0.5f - ctx.py(320.f), ctx.px(640.f), ctx.py(640.f)};

    addCenteredText(layout, ctx, "VIDEO", 0.5f + ctx.py(276.f), 3.0f);

    auto onOff = [](bool on) { return std::string(on ? "ON" : "OFF"); };
    static constexpr const char* SHADOW_NAMES[] = {"OFF", "MEDIUM", "HIGH"};
    static constexpr const char* CLOUD_NAMES[] = {"OFF", "FLAT", "VOLUM"};

    float y = 0.5f + ctx.py(216.f);
    const float pitch = ctx.py(48.f);
    addSettingsRow(layout, ctx, "SHADOWS", SHADOW_NAMES[gfx.shadowQuality],
                   MenuAction::SHADOWS_DOWN, MenuAction::SHADOWS_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "VOLUM LIGHT", onOff(gfx.volumetricLight), MenuAction::VL_TOGGLE,
                   MenuAction::VL_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "CLOUDS", CLOUD_NAMES[gfx.cloudMode], MenuAction::CLOUDS_DOWN,
                   MenuAction::CLOUDS_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "AMBIENT OCCL", onOff(gfx.ssao), MenuAction::SSAO_TOGGLE,
                   MenuAction::SSAO_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "REFLECTIONS", onOff(gfx.waterReflections), MenuAction::SSR_TOGGLE,
                   MenuAction::SSR_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "WAVING", onOff(gfx.wavingFoliage), MenuAction::WAVING_TOGGLE,
                   MenuAction::WAVING_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "LENS FLARE", onOff(gfx.lensFlare), MenuAction::LENS_FLARE_TOGGLE,
                   MenuAction::LENS_FLARE_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "BLOOM", std::to_string(gfx.bloomLevel), MenuAction::BLOOM_DOWN,
                   MenuAction::BLOOM_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "VIBRANCE", std::to_string(gfx.vibrance), MenuAction::VIBRANCE_DOWN,
                   MenuAction::VIBRANCE_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "SHARPEN", std::to_string(gfx.sharpening), MenuAction::SHARPEN_DOWN,
                   MenuAction::SHARPEN_UP, y);

    addButton(layout, ctx, "BACK", MenuAction::CLOSE_VIDEO_SETTINGS, 0.5f, 0.5f - ctx.py(280.f),
              320.f, 44.f);
    return layout;
}

// One text-entry row: the box is the hit target; the label rides above it.
void addTextField(MenuLayout& layout, const LayoutContext& ctx, const std::string& label,
                  const std::string& text, bool focused, bool caret, float centerY) {
    UIRect rect{0.5f - ctx.px(180.f), centerY - ctx.py(22.f), ctx.px(360.f), ctx.py(44.f)};
    layout.textFields.push_back(TextFieldWidget{rect, label, text, focused, focused && caret});
}

MenuLayout buildWorldSelectLayout(const LayoutContext& ctx, const MenuContext& menu) {
    MenuLayout layout;
    layout.dimAlpha = 0.55f;
    layout.panel = UIRect{0.5f - ctx.px(310.f), 0.5f - ctx.py(280.f), ctx.px(620.f), ctx.py(560.f)};

    addCenteredText(layout, ctx, "SELECT WORLD", 0.5f + ctx.py(240.f), 3.0f);

    const int count = static_cast<int>(menu.worldRows.size());
    const int maxScroll = std::max(0, count - WorldSelectState::VISIBLE_ROWS);
    const int scroll = std::clamp(menu.worldSelect.scroll, 0, maxScroll);
    const bool hasSelection = menu.worldSelect.selected >= 0 && menu.worldSelect.selected < count;

    if (count == 0) {
        addCenteredText(layout, ctx, "NO WORLDS YET", 0.5f + ctx.py(60.f), 2.0f, 0.8f, 0.8f, 0.85f);
    }

    float y = 0.5f + ctx.py(180.f);
    for (int row = 0; row < WorldSelectState::VISIBLE_ROWS && scroll + row < count; ++row) {
        const int index = scroll + row;
        addButton(layout, ctx, menu.worldRows[static_cast<size_t>(index)], MenuAction::SELECT_WORLD,
                  0.5f - ctx.px(30.f), y, 480.f, 44.f);
        layout.buttons.back().payload = index;
        layout.buttons.back().emphasized = index == menu.worldSelect.selected;
        y -= ctx.py(52.f);
    }
    if (scroll > 0) {
        addButton(layout, ctx, "UP", MenuAction::WORLD_LIST_UP, 0.5f + ctx.px(260.f),
                  0.5f + ctx.py(180.f), 72.f, 44.f);
    }
    if (scroll < maxScroll) {
        addButton(layout, ctx, "DOWN", MenuAction::WORLD_LIST_DOWN, 0.5f + ctx.px(260.f),
                  0.5f - ctx.py(28.f), 72.f, 44.f);
    }

    float bottom = 0.5f - ctx.py(120.f);
    if (hasSelection) {
        addButton(layout, ctx, "PLAY SELECTED", MenuAction::PLAY_SELECTED_WORLD, 0.5f, bottom,
                  400.f, 44.f);
    }
    bottom -= ctx.py(52.f);
    addButton(layout, ctx, "CREATE NEW WORLD", MenuAction::OPEN_WORLD_CREATE, 0.5f, bottom, 400.f,
              44.f);
    bottom -= ctx.py(52.f);
    if (hasSelection) {
        addButton(layout, ctx, "DELETE", MenuAction::REQUEST_DELETE_WORLD, 0.5f - ctx.px(105.f),
                  bottom, 190.f, 44.f);
        addButton(layout, ctx, "BACK", MenuAction::WORLD_BACK, 0.5f + ctx.px(105.f), bottom, 190.f,
                  44.f);
    } else {
        addButton(layout, ctx, "BACK", MenuAction::WORLD_BACK, 0.5f, bottom, 400.f, 44.f);
    }
    return layout;
}

MenuLayout buildWorldCreateLayout(const LayoutContext& ctx, const MenuContext& menu) {
    MenuLayout layout;
    layout.dimAlpha = 0.55f;
    layout.panel = UIRect{0.5f - ctx.px(310.f), 0.5f - ctx.py(290.f), ctx.px(620.f), ctx.py(580.f)};

    const WorldCreateState& create = menu.worldCreate;
    addCenteredText(layout, ctx, "CREATE WORLD", 0.5f + ctx.py(250.f), 3.0f);

    addTextField(layout, ctx, "NAME", create.name, create.focusedField == 0, menu.caretVisible,
                 0.5f + ctx.py(170.f));
    addTextField(layout, ctx, "SEED", create.seedText, create.focusedField == 1, menu.caretVisible,
                 0.5f + ctx.py(92.f));
    addButton(layout, ctx, "RANDOM", MenuAction::RANDOM_SEED, 0.5f + ctx.px(250.f),
              0.5f + ctx.py(92.f), 110.f, 44.f);

    auto onOff = [](bool on) { return std::string(on ? "ON" : "OFF"); };
    float y = 0.5f + ctx.py(28.f);
    const float pitch = ctx.py(50.f);
    addSettingsRow(layout, ctx, "STRUCTURES", onOff(create.structures),
                   MenuAction::TOGGLE_GEN_STRUCTURES, MenuAction::TOGGLE_GEN_STRUCTURES, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "FAUNA", onOff(create.fauna), MenuAction::TOGGLE_GEN_FAUNA,
                   MenuAction::TOGGLE_GEN_FAUNA, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "WEATHER", onOff(create.weather), MenuAction::TOGGLE_GEN_WEATHER,
                   MenuAction::TOGGLE_GEN_WEATHER, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "DAY CYCLE", onOff(create.dayCycle),
                   MenuAction::TOGGLE_GEN_DAY_CYCLE, MenuAction::TOGGLE_GEN_DAY_CYCLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "MODE", create.creative ? "CREATIVE" : "SURVIVAL",
                   MenuAction::TOGGLE_CREATE_MODE, MenuAction::TOGGLE_CREATE_MODE, y);

    // CREATE appears only once the trimmed name is non-empty, so the click
    // target exists exactly when the action can succeed.
    const bool named = create.name.find_first_not_of(' ') != std::string::npos;
    if (named) {
        addButton(layout, ctx, "CREATE", MenuAction::CREATE_WORLD_CONFIRM, 0.5f,
                  0.5f - ctx.py(216.f), 400.f, 44.f);
    }
    addButton(layout, ctx, "BACK", MenuAction::WORLD_BACK, 0.5f, 0.5f - ctx.py(268.f), 400.f, 44.f);
    return layout;
}

MenuLayout buildDeleteConfirmLayout(const LayoutContext& ctx, const MenuContext& menu) {
    MenuLayout layout;
    layout.dimAlpha = 0.6f;
    layout.panel = UIRect{0.5f - ctx.px(280.f), 0.5f - ctx.py(130.f), ctx.px(560.f), ctx.py(260.f)};

    addCenteredText(layout, ctx, "DELETE WORLD?", 0.5f + ctx.py(70.f), 3.0f);
    addCenteredText(layout, ctx, menu.deleteWorldName + " WILL BE LOST FOREVER",
                    0.5f + ctx.py(16.f), 1.5f, 0.9f, 0.7f, 0.7f);

    addButton(layout, ctx, "DELETE", MenuAction::CONFIRM_DELETE, 0.5f - ctx.px(110.f),
              0.5f - ctx.py(64.f), 200.f, 44.f);
    addButton(layout, ctx, "CANCEL", MenuAction::CANCEL_DELETE, 0.5f + ctx.px(110.f),
              0.5f - ctx.py(64.f), 200.f, 44.f);
    return layout;
}

} // namespace

MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values, const GraphicsSettings& gfx) {
    if (pixelWidth <= 0.f || pixelHeight <= 0.f) return {};
    LayoutContext ctx{pixelWidth, pixelHeight, pixelHeight / 768.0f};

    switch (screen) {
        case GameScreen::TITLE:
            return buildTitleLayout(ctx);
        case GameScreen::PAUSED:
            return buildPauseLayout(ctx, GameMode::SURVIVAL);
        case GameScreen::SETTINGS:
            return buildSettingsLayout(ctx, values);
        case GameScreen::VIDEO_SETTINGS:
            return buildVideoSettingsLayout(ctx, gfx);
        case GameScreen::PLAYING:
        case GameScreen::WORLD_SELECT:
        case GameScreen::WORLD_CREATE:
        case GameScreen::WORLD_DELETE_CONFIRM:
        case GameScreen::INVENTORY:
        case GameScreen::CRAFTING:
        case GameScreen::FURNACE:
        case GameScreen::DEATH:
            return {};
    }
    return {};
}

int menuHitTest(const MenuLayout& layout, float mouseX, float mouseY) {
    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        if (layout.buttons[i].rect.contains(mouseX, mouseY)) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

MenuLayout buildScreenLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                             const MenuContext& ctx) {
    if (pixelWidth <= 0.f || pixelHeight <= 0.f) return {};
    LayoutContext layoutCtx{pixelWidth, pixelHeight, pixelHeight / 768.0f};
    static const GraphicsSettings DEFAULT_GRAPHICS{};

    switch (screen) {
        case GameScreen::PAUSED:
            return buildPauseLayout(layoutCtx, ctx.mode);
        case GameScreen::WORLD_SELECT:
            return buildWorldSelectLayout(layoutCtx, ctx);
        case GameScreen::WORLD_CREATE:
            return buildWorldCreateLayout(layoutCtx, ctx);
        case GameScreen::WORLD_DELETE_CONFIRM:
            return buildDeleteConfirmLayout(layoutCtx, ctx);
        case GameScreen::TITLE:
        case GameScreen::SETTINGS:
        case GameScreen::VIDEO_SETTINGS:
        case GameScreen::PLAYING:
        case GameScreen::INVENTORY:
        case GameScreen::CRAFTING:
        case GameScreen::FURNACE:
        case GameScreen::DEATH:
            return buildMenuLayout(screen, pixelWidth, pixelHeight, ctx.settings,
                                   ctx.gfx ? *ctx.gfx : DEFAULT_GRAPHICS);
    }
    return {};
}

std::string filterTextField(const std::string& raw, bool digitsOnly, size_t maxLength) {
    std::string filtered;
    filtered.reserve(std::min(raw.size(), maxLength));
    for (char c : raw) {
        if (filtered.size() >= maxLength) break;
        if (digitsOnly ? (c >= '0' && c <= '9') : isWorldNameChar(c)) {
            filtered.push_back(c);
        }
    }
    return filtered;
}

UIHit uiHitTest(const MenuLayout& layout, float mouseX, float mouseY) {
    for (size_t i = 0; i < layout.textFields.size(); ++i) {
        if (layout.textFields[i].rect.contains(mouseX, mouseY)) {
            return UIHit{UIHitKind::TEXT_FIELD, static_cast<int>(i)};
        }
    }
    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        if (layout.buttons[i].rect.contains(mouseX, mouseY)) {
            return UIHit{UIHitKind::BUTTON, static_cast<int>(i)};
        }
    }
    return {};
}
