#include "render/atmospheric_memory.hpp"

#include "render/atmosphere.hpp"
#include "render/cloud_renderer.hpp"
#include "render/lightning_renderer.hpp"
#include "render/screen_space_lighting.hpp"
#include "render/shadow_map.hpp"
#include "render/volumetrics.hpp"

#include <algorithm>

AtmosphericMemoryFootprint atmosphericMemoryFootprint(uint32_t width, uint32_t height,
                                                      int quality) noexcept {
    width = std::max(width, 1U);
    height = std::max(height, 1U);
    quality = std::clamp(quality, 0, 2);

    AtmosphericMemoryFootprint result;
    result.sceneTargetBytes = atmosphericSceneTargetMemoryBytes(width, height);
    result.shadowBytes = shadowMapMemoryBytes(static_cast<uint32_t>(std::max(quality, 1)));
    result.indirectBytes = screenSpaceLightingMemoryFootprint(width, height, quality).totalBytes();
    result.atmosphereBytes = atmosphereLutMemoryBytes();
    result.cloudBytes = cloudRendererMemoryFootprint(width, height, quality).totalBytes();
    result.volumetricBytes = volumetricMemoryBytes(width, height);
    result.lightningBytes = LIGHTNING_RENDERER_MEMORY_BYTES;
    return result;
}
