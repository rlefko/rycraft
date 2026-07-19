#include "render/ui_menu.hpp"

#include "render/graphics_settings.hpp"

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
    addButton(layout, ctx, "-", down, stepperX - ctx.px(70.f), centerY, 32.f, 32.f);
    float valueWidth = menuTextWidth(value, labelScale * ctx.s, ctx.w);
    layout.texts.push_back(MenuText{stepperX - valueWidth * 0.5f,
                                    centerY - ctx.py(FONT_HEIGHT * labelScale) * 0.5f,
                                    labelScale * ctx.s, value, 1.f, 0.9f, 0.4f});
    addButton(layout, ctx, "+", up, stepperX + ctx.px(70.f), centerY, 32.f, 32.f);
}

MenuLayout buildTitleLayout(const LayoutContext& ctx) {
    MenuLayout layout;
    layout.dimAlpha = 0.25f;

    addCenteredText(layout, ctx, "rycraft", 0.68f, 8.0f);
    addCenteredText(layout, ctx, "A voxel world built on Metal", 0.58f, 1.5f, 0.85f, 0.85f, 0.9f);

    addButton(layout, ctx, "PLAY", MenuAction::PLAY, 0.5f, 0.42f, 320.f, 48.f);
    addButton(layout, ctx, "QUIT", MenuAction::QUIT, 0.5f, 0.42f - ctx.py(68.f), 320.f, 48.f);
    return layout;
}

MenuLayout buildPauseLayout(const LayoutContext& ctx) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(210.f), 0.5f - ctx.py(170.f), ctx.px(420.f), ctx.py(340.f)};

    addCenteredText(layout, ctx, "PAUSED", 0.5f + ctx.py(120.f), 3.0f);

    addButton(layout, ctx, "RESUME", MenuAction::RESUME, 0.5f, 0.5f + ctx.py(40.f), 320.f, 44.f);
    addButton(layout, ctx, "SETTINGS", MenuAction::OPEN_SETTINGS, 0.5f, 0.5f - ctx.py(24.f), 320.f,
              44.f);
    addButton(layout, ctx, "QUIT", MenuAction::QUIT, 0.5f, 0.5f - ctx.py(88.f), 320.f, 44.f);
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
// widget with both arrows bound to the same TOGGLE action, no new widget,
// no second hit-test path.
MenuLayout buildVideoSettingsLayout(const LayoutContext& ctx, const GraphicsSettings& gfx) {
    MenuLayout layout;
    layout.dimAlpha = 0.45f;
    layout.panel = UIRect{0.5f - ctx.px(320.f), 0.5f - ctx.py(320.f), ctx.px(640.f), ctx.py(640.f)};

    addCenteredText(layout, ctx, "VIDEO", 0.5f + ctx.py(276.f), 3.0f);

    auto onOff = [](bool on) { return std::string(on ? "ON" : "OFF"); };
    static constexpr const char* SHADOW_NAMES[] = {"OFF", "MEDIUM", "HIGH"};
    static constexpr const char* QUALITY_NAMES[] = {"OFF", "MEDIUM", "HIGH"};

    float y = 0.5f + ctx.py(216.f);
    const float pitch = ctx.py(48.f);
    addSettingsRow(layout, ctx, "SHADOWS", SHADOW_NAMES[gfx.shadowQuality],
                   MenuAction::SHADOWS_DOWN, MenuAction::SHADOWS_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "VOLUM LIGHT", onOff(gfx.volumetricLight), MenuAction::VL_TOGGLE,
                   MenuAction::VL_TOGGLE, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "CLOUDS", QUALITY_NAMES[gfx.cloudQuality], MenuAction::CLOUDS_DOWN,
                   MenuAction::CLOUDS_UP, y);
    y -= pitch;
    addSettingsRow(layout, ctx, "INDIRECT LIGHT", QUALITY_NAMES[gfx.indirectLightingQuality],
                   MenuAction::INDIRECT_DOWN, MenuAction::INDIRECT_UP, y);
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

} // namespace

MenuLayout buildMenuLayout(GameScreen screen, float pixelWidth, float pixelHeight,
                           const SettingsValues& values, const GraphicsSettings& gfx) {
    if (pixelWidth <= 0.f || pixelHeight <= 0.f) return {};
    LayoutContext ctx{pixelWidth, pixelHeight, pixelHeight / 768.0f};

    switch (screen) {
        case GameScreen::TITLE:
            return buildTitleLayout(ctx);
        case GameScreen::PAUSED:
            return buildPauseLayout(ctx);
        case GameScreen::SETTINGS:
            return buildSettingsLayout(ctx, values);
        case GameScreen::VIDEO_SETTINGS:
            return buildVideoSettingsLayout(ctx, gfx);
        case GameScreen::PLAYING:
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
