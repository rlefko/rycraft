#include "render/ui_hud.hpp"

#include "render/ui_overlay.hpp"
#include "world/block_properties.hpp"

void drawGameHud(UIOverlay& ui, const Hotbar& hotbar, const UIFrameState& frame,
                 uint32_t displayWidth, uint32_t displayHeight) {
    if (frame.screen == GameScreen::Title) return;

    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    // ---- Debug HUD (F3) ----
    if (frame.showDebugHud) {
        ui.drawPerformanceHUD(frame.stats);
    }

    // ---- Crosshair (gameplay only — menus have a real cursor) ----
    if (frame.screen == GameScreen::Playing) {
        float crossH = 2.0f / h;
        float crossW = 24.0f / w;
        float crossV = 24.0f / h;
        float crossLineW = 2.0f / w;
        ui.drawQuad(0.5f - crossW * 0.5f, 0.5f - crossH * 0.5f, crossW, crossH,
                    1.0f, 1.0f, 1.0f, 0.9f);
        ui.drawQuad(0.5f - crossLineW * 0.5f, 0.5f - crossV * 0.5f, crossLineW, crossV,
                    1.0f, 1.0f, 1.0f, 0.9f);
    }

    // ---- Hotbar (9 slots at bottom of screen) ----
    float slotSize = 48.0f / h;
    float slotGap = 2.0f / h;
    float hotbarY = 6.0f / h;
    float totalWidth = Hotbar::SLOTS * slotSize + (Hotbar::SLOTS - 1) * slotGap;
    float hotbarX = (1.0f - totalWidth) * 0.5f;

    int selectedIndex = hotbar.getSelectedIndex();

    for (int i = 0; i < Hotbar::SLOTS; ++i) {
        float slotX = hotbarX + i * (slotSize + slotGap);

        if (i == selectedIndex) {
            ui.drawQuad(slotX - 2.0f / w, hotbarY - 2.0f / h,
                        slotSize + 4.0f / w, slotSize + 4.0f / h,
                        1.0f, 1.0f, 1.0f, 0.8f);
        }

        ui.drawQuad(slotX, hotbarY, slotSize, slotSize, 0.3f, 0.3f, 0.3f, 0.6f);

        // Block type indicator (simplified: color per block type)
        BlockType type = hotbar.getSlot(i);
        float r = 0.5f, g = 0.5f, b = 0.5f;
        switch (type) {
            case BlockType::STONE:    r = 0.5f; g = 0.5f; b = 0.5f; break;
            case BlockType::DIRT:     r = 0.55f; g = 0.35f; b = 0.2f; break;
            case BlockType::GRASS:    r = 0.2f; g = 0.6f; b = 0.2f; break;
            case BlockType::LOG:      r = 0.4f; g = 0.25f; b = 0.15f; break;
            case BlockType::SAND:     r = 0.85f; g = 0.78f; b = 0.55f; break;
            case BlockType::PLANKS:   r = 0.65f; g = 0.45f; b = 0.25f; break;
            case BlockType::BEDROCK:  r = 0.2f; g = 0.2f; b = 0.2f; break;
            case BlockType::COAL_ORE: r = 0.15f; g = 0.15f; b = 0.15f; break;
            case BlockType::IRON_ORE: r = 0.6f; g = 0.5f; b = 0.45f; break;
            default:                  r = 0.5f; g = 0.5f; b = 0.5f; break;
        }

        float innerSize = slotSize * 0.7f;
        float innerOffset = (slotSize - innerSize) * 0.5f;
        ui.drawQuad(slotX + innerOffset, hotbarY + innerOffset, innerSize, innerSize,
                    r, g, b, 0.9f);
    }
}

void drawMenu(UIOverlay& ui, const MenuLayout& layout, int hoveredButton,
              uint32_t displayWidth, uint32_t displayHeight) {
    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    if (layout.dimAlpha > 0.f) {
        ui.drawQuad(0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, layout.dimAlpha);
    }

    if (layout.panel.w > 0.f) {
        // Border quad behind the panel gives a cheap Minecraft-style bevel
        float bx = 3.0f / w;
        float by = 3.0f / h;
        ui.drawQuad(layout.panel.x - bx, layout.panel.y - by,
                    layout.panel.w + 2 * bx, layout.panel.h + 2 * by,
                    0.02f, 0.02f, 0.03f, 0.95f);
        ui.drawQuad(layout.panel.x, layout.panel.y, layout.panel.w, layout.panel.h,
                    0.11f, 0.11f, 0.14f, 0.92f);
    }

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const MenuButton& button = layout.buttons[i];
        const bool hovered = static_cast<int>(i) == hoveredButton;

        float bx = 2.0f / w;
        float by = 2.0f / h;
        ui.drawQuad(button.rect.x - bx, button.rect.y - by,
                    button.rect.w + 2 * bx, button.rect.h + 2 * by,
                    0.02f, 0.02f, 0.03f, 0.95f);
        if (hovered) {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h,
                        0.45f, 0.52f, 0.82f, 0.95f);
        } else {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h,
                        0.33f, 0.33f, 0.38f, 0.92f);
        }

        const float labelScale = 2.0f * (h / 768.0f);
        float labelWidth = ui.measureString(button.label.c_str(), labelScale);
        float labelHeight = 8.0f * labelScale / h;
        float labelX = button.rect.x + (button.rect.w - labelWidth) * 0.5f;
        float labelY = button.rect.y + (button.rect.h - labelHeight) * 0.5f;
        float brightness = hovered ? 1.0f : 0.92f;
        ui.drawString(button.label.c_str(), labelX, labelY, labelScale,
                      brightness, brightness, brightness);
    }

    for (const MenuText& text : layout.texts) {
        ui.drawString(text.text.c_str(), text.x, text.y, text.scale, text.r, text.g, text.b);
    }
}
