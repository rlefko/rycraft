#include "render/block_textures.hpp"
#include "render/ui_hud.hpp"
#include "render/ui_overlay.hpp"

// The one home for how a stack renders inside a slot rectangle: cube blocks
// draw as a shaded isometric mini-cube from their real face layers, flora
// and non-block items draw their flat texture layer. Counts are drawn by
// callers in the top phase so digits always sit above icons.
void drawItemIcon(UIOverlay& ui, const ItemStack& stack, float x, float y, float w, float h) {
    if (stack.empty())
        return;

    const BlockType block = isBlockItem(stack.type) ? blockFromItem(stack.type) : BlockType::AIR;
    if (block != BlockType::AIR && (rendersAsCube(block) || rendersAsLowBox(block))) {
        const float cx = x + w * 0.5f;
        const float shapeScale = rendersAsLowBox(block) ? blockCollisionHeight(block) : 1.0F;
        const float baseOffset = rendersAsLowBox(block) ? 0.15F : 0.0F;
        const auto iconY = [=](float normalized) {
            return y + h * (baseOffset + normalized * shapeScale);
        };
        const float topY = iconY(1.0F);
        const float midHighY = iconY(0.75F);
        const float centerY = iconY(0.5F);
        const float midLowY = iconY(0.25F);
        const float baseY = iconY(0.0F);

        const uint8_t topLayer = textureLayerFor(block, FaceNormal::PLUS_Y);
        const uint8_t leftLayer = textureLayerFor(block, FaceNormal::MINUS_X);
        const uint8_t rightLayer = textureLayerFor(block, itemIconRightFaceFor(block));

        // Corners are bottom-left, bottom-right, top-left, top-right.
        const float top[8] = {x, midHighY, cx, centerY, cx, topY, x + w, midHighY};
        const float left[8] = {x, midLowY, cx, baseY, x, midHighY, cx, centerY};
        const float right[8] = {cx, baseY, x + w, midLowY, cx, centerY, x + w, midHighY};
        ui.drawIconQuadCorners(top, topLayer, 1.0f, 1.0f);
        ui.drawIconQuadCorners(left, leftLayer, 0.8f, 1.0f);
        ui.drawIconQuadCorners(right, rightLayer, 0.6f, 1.0f);
        return;
    }

    const uint8_t layer = block != BlockType::AIR ? textureLayerFor(block, FaceNormal::PLUS_X)
                                                  : itemIconLayer(stack.type);
    ui.drawIconQuad(x, y, w, h, layer, 1.0f, 1.0f);
}
