#include "render/ui_hud.hpp"

#include "render/ui_overlay.hpp"
#include "world/item.hpp"

#include <algorithm>
#include <array>

namespace {

// 8x8 HUD icon bitmaps, drawn per pixel like the bitmap font (MSB = leftmost).
constexpr std::array<uint8_t, 8> HEART_ICON = {0x6C, 0xFE, 0xFE, 0xFE, 0x7C, 0x38, 0x10, 0x00};
constexpr std::array<uint8_t, 8> FOOD_ICON = {0x38, 0x7C, 0x7C, 0x3C, 0x18, 0x18, 0x30, 0x00};
constexpr std::array<uint8_t, 8> BUBBLE_ICON = {0x38, 0x7C, 0xDE, 0xDE, 0xFE, 0x7C, 0x38, 0x00};

// Draw one 8x8 icon at (x, y) in normalized coords. leftHalfOnly paints only
// the left four columns, giving a half-heart / half-drumstick.
void drawHudIcon(UIOverlay& ui, const std::array<uint8_t, 8>& icon, float x, float y, float scale,
                 float w, float h, float r, float g, float b, bool leftHalfOnly = false) {
    const float px = scale / w;
    const float py = scale / h;
    for (int row = 0; row < 8; ++row) {
        const uint8_t bits = icon[static_cast<size_t>(row)];
        for (int col = 0; col < (leftHalfOnly ? 4 : 8); ++col) {
            if (bits & (0x80 >> col)) {
                ui.drawQuad(x + col * px, y + (7 - row) * py, px, py, r, g, b, 1.0f);
            }
        }
    }
}

// A row of `count` icons filled to `value` (each icon is two units), starting
// at startX and stepping by `dir` (+1 rightward, -1 leftward).
void drawStatRow(UIOverlay& ui, const std::array<uint8_t, 8>& icon, int value, int count,
                 float startX, float y, float scale, float w, float h, float fillR, float fillG,
                 float fillB) {
    const float pitch = 9.0f * scale / w;
    for (int i = 0; i < count; ++i) {
        const float x = startX + i * pitch;
        drawHudIcon(ui, icon, x, y, scale, w, h, 0.15f, 0.05f, 0.05f); // empty backing
        const int covered = value - i * 2;
        if (covered >= 2) {
            drawHudIcon(ui, icon, x, y, scale, w, h, fillR, fillG, fillB);
        } else if (covered == 1) {
            drawHudIcon(ui, icon, x, y, scale, w, h, fillR, fillG, fillB, true);
        }
    }
}

} // namespace

void drawGameHud(UIOverlay& ui, const UIFrameState& frame, uint32_t displayWidth,
                 uint32_t displayHeight) {
    if (!screenHasWorldSession(frame.screen))
        return;

    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    // ---- Debug HUD (F3) ----
    if (frame.showDebugHud) {
        ui.drawPerformanceHUD(frame.stats);
    }

    // ---- Crosshair (gameplay only — menus have a real cursor) ----
    if (frame.screen == GameScreen::PLAYING) {
        float crossH = 2.0f / h;
        float crossW = 24.0f / w;
        float crossV = 24.0f / h;
        float crossLineW = 2.0f / w;
        ui.drawQuad(0.5f - crossW * 0.5f, 0.5f - crossH * 0.5f, crossW, crossH, 1.0f, 1.0f, 1.0f,
                    0.9f);
        ui.drawQuad(0.5f - crossLineW * 0.5f, 0.5f - crossV * 0.5f, crossLineW, crossV, 1.0f, 1.0f,
                    1.0f, 0.9f);
    }

    // ---- Break progress bar (survival mining, under the crosshair) ----
    if (frame.screen == GameScreen::PLAYING && frame.miningProgress > 0.f) {
        const float barW = 60.0f / w;
        const float barH = 6.0f / h;
        const float barX = 0.5f - barW * 0.5f;
        const float barY = 0.5f - 40.0f / h;
        ui.drawQuad(barX - 1.0f / w, barY - 1.0f / h, barW + 2.0f / w, barH + 2.0f / h, 0.05f,
                    0.05f, 0.05f, 0.85f);
        ui.drawQuad(barX, barY, barW * std::clamp(frame.miningProgress, 0.f, 1.f), barH, 0.9f, 0.9f,
                    0.9f, 0.95f);
    }

    // ---- Hotbar (9 slots at bottom of screen) ----
    // The UI projection has no aspect correction, so a square slot needs a
    // width fraction (over w) and a height fraction (over h) that map to the
    // same device pixels. slotH is the slot edge in height units; slotW is the
    // matching width so the slot, its icon, and its count all align.
    const int slotCount = static_cast<int>(frame.hotbar.slots.size());
    const float slotPixels = 58.0f;
    float slotH = slotPixels / h;
    float slotW = slotPixels / w;
    float slotGapW = 2.0f / w;
    float hotbarY = 8.0f / h;
    float totalWidth = slotCount * slotW + (slotCount - 1) * slotGapW;
    float hotbarX = (1.0f - totalWidth) * 0.5f;

    // ---- Survival stat rows (hearts, hunger, air) above the hotbar ----
    if (frame.mode == GameMode::SURVIVAL && frame.screen == GameScreen::PLAYING) {
        // scale = device pixels per 8x8 bitmap pixel (an ~14px icon at 2x).
        const float iconScale = 1.75f * (h / 768.0f);
        const float pitch = 9.0f * iconScale / w;
        const float heartsY = hotbarY + slotH + 8.0f / h;
        drawStatRow(ui, HEART_ICON, frame.health, 10, hotbarX, heartsY, iconScale, w, h, 0.85f,
                    0.12f, 0.12f);
        const float hungerStartX = hotbarX + totalWidth - 10 * pitch;
        drawStatRow(ui, FOOD_ICON, frame.food, 10, hungerStartX, heartsY, iconScale, w, h, 0.75f,
                    0.5f, 0.2f);
        // Air bubbles only while submerged, emptied as breath runs out.
        if (frame.air < frame.maxAir) {
            const int bubbles =
                std::clamp((frame.air * 20 + frame.maxAir - 1) / frame.maxAir, 0, 20);
            const float airY = heartsY + iconScale / h + 4.0f / h;
            drawStatRow(ui, BUBBLE_ICON, bubbles, 10, hungerStartX, airY, iconScale, w, h, 0.5f,
                        0.75f, 0.95f);
        }
    }

    int selectedIndex = frame.hotbar.selected;

    for (int i = 0; i < slotCount; ++i) {
        float slotX = hotbarX + i * (slotW + slotGapW);

        if (i == selectedIndex) {
            ui.drawQuad(slotX - 2.0f / w, hotbarY - 2.0f / h, slotW + 4.0f / w, slotH + 4.0f / h,
                        1.0f, 1.0f, 1.0f, 0.8f);
        }

        ui.drawQuad(slotX, hotbarY, slotW, slotH, 0.3f, 0.3f, 0.3f, 0.6f);

        // Item indicator: the real block/item texture, empty slots stay bare
        const ItemStack& stack = frame.hotbar.slots[static_cast<size_t>(i)];
        if (stack.empty())
            continue;
        const float insetW = slotW * 0.14f;
        const float insetH = slotH * 0.14f;
        drawItemIcon(ui, stack, slotX + insetW, hotbarY + insetH, slotW - 2 * insetW,
                     slotH - 2 * insetH);

        if (stack.count > 1) {
            char count[8];
            UIOverlay::intToString(stack.count, count, sizeof(count));
            const float countScale = 1.5f * (h / 768.0f);
            const float countWidth = ui.measureString(count, countScale);
            const float countX = slotX + slotW - countWidth - 2.0f / w;
            const float countY = hotbarY + 3.0f / h;
            ui.drawStringTop(count, countX + 1.0f / w, countY - 1.0f / h, countScale, 0.05f, 0.05f,
                             0.05f);
            ui.drawStringTop(count, countX, countY, countScale, 1.0f, 1.0f, 1.0f);
        }
    }
}

void drawMenu(UIOverlay& ui, const UIFrameState& frame, uint32_t displayWidth,
              uint32_t displayHeight) {
    const MenuLayout& layout = frame.menu;
    const int hoveredButton = frame.hoveredButton;
    const float w = static_cast<float>(displayWidth);
    const float h = static_cast<float>(displayHeight);

    if (layout.dimAlpha > 0.f) {
        ui.drawQuad(0.f, 0.f, 1.f, 1.f, layout.dimR, layout.dimG, layout.dimB, layout.dimAlpha);
    }

    if (layout.panel.w > 0.f) {
        // Border quad behind the panel gives a cheap Minecraft-style bevel
        float bx = 3.0f / w;
        float by = 3.0f / h;
        ui.drawQuad(layout.panel.x - bx, layout.panel.y - by, layout.panel.w + 2 * bx,
                    layout.panel.h + 2 * by, 0.02f, 0.02f, 0.03f, 0.95f);
        ui.drawQuad(layout.panel.x, layout.panel.y, layout.panel.w, layout.panel.h, 0.11f, 0.11f,
                    0.14f, 0.92f);
    }

    for (size_t i = 0; i < layout.buttons.size(); ++i) {
        const MenuButton& button = layout.buttons[i];
        const bool hovered = static_cast<int>(i) == hoveredButton;

        float bx = 2.0f / w;
        float by = 2.0f / h;
        ui.drawQuad(button.rect.x - bx, button.rect.y - by, button.rect.w + 2 * bx,
                    button.rect.h + 2 * by, 0.02f, 0.02f, 0.03f, 0.95f);
        if (hovered) {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h, 0.45f, 0.52f,
                        0.82f, 0.95f);
        } else if (button.emphasized) {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h, 0.30f, 0.48f,
                        0.36f, 0.95f);
        } else {
            ui.drawQuad(button.rect.x, button.rect.y, button.rect.w, button.rect.h, 0.33f, 0.33f,
                        0.38f, 0.92f);
        }

        // Long labels (world rows) shrink to fit instead of bleeding out.
        float labelScale = 2.0f * (h / 768.0f);
        float labelWidth = ui.measureString(button.label.c_str(), labelScale);
        const float maxWidth = button.rect.w * 0.94f;
        if (labelWidth > maxWidth && labelWidth > 0.f) {
            labelScale *= maxWidth / labelWidth;
            labelWidth = ui.measureString(button.label.c_str(), labelScale);
        }
        float labelHeight = 8.0f * labelScale / h;
        float labelX = button.rect.x + (button.rect.w - labelWidth) * 0.5f;
        float labelY = button.rect.y + (button.rect.h - labelHeight) * 0.5f;
        float brightness = hovered ? 1.0f : 0.92f;
        ui.drawString(button.label.c_str(), labelX, labelY, labelScale, brightness, brightness,
                      brightness);
    }

    for (const TextFieldWidget& field : layout.textFields) {
        float bx = 2.0f / w;
        float by = 2.0f / h;
        const float border = field.focused ? 0.85f : 0.35f;
        ui.drawQuad(field.rect.x - bx, field.rect.y - by, field.rect.w + 2 * bx,
                    field.rect.h + 2 * by, border, border, border + 0.05f, 0.95f);
        ui.drawQuad(field.rect.x, field.rect.y, field.rect.w, field.rect.h, 0.05f, 0.05f, 0.07f,
                    0.95f);

        const float labelScale = 1.5f * (h / 768.0f);
        ui.drawString(field.label.c_str(), field.rect.x, field.rect.y + field.rect.h + 4.0f / h,
                      labelScale, 0.8f, 0.8f, 0.85f);

        const float textScale = 2.0f * (h / 768.0f);
        const float textHeight = 8.0f * textScale / h;
        const float textX = field.rect.x + 8.0f / w;
        const float textY = field.rect.y + (field.rect.h - textHeight) * 0.5f;
        const float advance =
            ui.drawString(field.text.c_str(), textX, textY, textScale, 1.0f, 1.0f, 1.0f);
        if (field.caret) {
            ui.drawQuad(textX + advance + 1.0f / w, textY, 2.0f / w, textHeight, 1.0f, 1.0f, 1.0f,
                        0.9f);
        }
    }

    for (const MenuText& text : layout.texts) {
        ui.drawString(text.text.c_str(), text.x, text.y, text.scale, text.r, text.g, text.b);
    }

    // ---- Container slots: inset well, icon, count, hover overlay ----
    for (size_t i = 0; i < layout.slots.size(); ++i) {
        const SlotWidget& slot = layout.slots[i];
        const float bx = 2.0f / w;
        const float by = 2.0f / h;
        ui.drawQuad(slot.rect.x - bx, slot.rect.y - by, slot.rect.w + 2 * bx, slot.rect.h + 2 * by,
                    0.05f, 0.05f, 0.06f, 0.95f);
        ui.drawQuad(slot.rect.x, slot.rect.y, slot.rect.w, slot.rect.h, 0.22f, 0.22f, 0.26f, 0.95f);

        if (!slot.stack.empty()) {
            const float inset = slot.rect.w * 0.12f;
            const float insetY = slot.rect.h * 0.12f;
            drawItemIcon(ui, slot.stack, slot.rect.x + inset, slot.rect.y + insetY,
                         slot.rect.w - 2 * inset, slot.rect.h - 2 * insetY);
            if (slot.stack.count > 1) {
                char count[8];
                UIOverlay::intToString(slot.stack.count, count, sizeof(count));
                const float countScale = 1.5f * (h / 768.0f);
                const float countWidth = ui.measureString(count, countScale);
                const float countX = slot.rect.x + slot.rect.w - countWidth - 2.0f / w;
                const float countY = slot.rect.y + 2.0f / h;
                ui.drawStringTop(count, countX + 1.0f / w, countY - 1.0f / h, countScale, 0.05f,
                                 0.05f, 0.05f);
                ui.drawStringTop(count, countX, countY, countScale, 1.0f, 1.0f, 1.0f);
            }
        }
        if (static_cast<int>(i) == frame.hoveredSlot) {
            ui.drawQuadTop(slot.rect.x, slot.rect.y, slot.rect.w, slot.rect.h, 1.0f, 1.0f, 1.0f,
                           0.22f);
        }
    }

    // ---- Gauges (furnace flame and cook arrow) ----
    for (const MeterWidget& meter : layout.meters) {
        ui.drawQuad(meter.rect.x, meter.rect.y, meter.rect.w, meter.rect.h, 0.08f, 0.08f, 0.1f,
                    0.95f);
        const float fill = std::clamp(meter.fill, 0.f, 1.f);
        if (fill > 0.f) {
            if (meter.vertical) {
                ui.drawQuad(meter.rect.x, meter.rect.y, meter.rect.w, meter.rect.h * fill, meter.r,
                            meter.g, meter.b, 0.95f);
            } else {
                ui.drawQuad(meter.rect.x, meter.rect.y, meter.rect.w * fill, meter.rect.h, meter.r,
                            meter.g, meter.b, 0.95f);
            }
        }
    }

    // ---- Cursor-held stack rides the mouse above everything ----
    if (!frame.cursorStack.empty()) {
        const float iconH = 40.0f * (h / 768.0f) / h;
        const float iconW = iconH * (h / w);
        drawItemIcon(ui, frame.cursorStack, frame.mouseX - iconW * 0.5f,
                     frame.mouseY - iconH * 0.5f, iconW, iconH);
        if (frame.cursorStack.count > 1) {
            char count[8];
            UIOverlay::intToString(frame.cursorStack.count, count, sizeof(count));
            const float countScale = 1.5f * (h / 768.0f);
            ui.drawStringTop(count, frame.mouseX + iconW * 0.2f, frame.mouseY - iconH * 0.55f,
                             countScale, 1.0f, 1.0f, 1.0f);
        }
    }

    // ---- Tooltip: hovered item name near the cursor, top phase ----
    if (!frame.tooltipText.empty() && frame.cursorStack.empty()) {
        const float tipScale = 1.5f * (h / 768.0f);
        const float tipWidth = ui.measureString(frame.tooltipText.c_str(), tipScale);
        const float tipHeight = 10.0f * tipScale / h;
        float tipX = std::min(frame.mouseX + 14.0f / w, 1.0f - tipWidth - 4.0f / w);
        float tipY = std::min(frame.mouseY + 10.0f / h, 1.0f - tipHeight - 4.0f / h);
        ui.drawQuadTop(tipX - 4.0f / w, tipY - 3.0f / h, tipWidth + 8.0f / w, tipHeight + 6.0f / h,
                       0.05f, 0.05f, 0.08f, 0.92f);
        ui.drawStringTop(frame.tooltipText.c_str(), tipX, tipY, tipScale, 1.0f, 1.0f, 1.0f);
    }
}
