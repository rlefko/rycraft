#pragma once

#import <Metal/Metal.h>

// ---------------------------------------------------------------------------
// The frame's pixel formats, defined once. Every pipeline state and pass
// descriptor reads these constants, so a format change can never drift
// per-pipeline (the HDR migration touched seven pipelines at once — with
// literals, any one of them could have silently kept BGRA8 and faulted the
// pass it renders into).
//
//   SCENE_HDR — linear working space for everything up to the composite.
//               RGBA16Float: the sun disc, emissive blocks, and bloom carry
//               radiance well above 1.0.
//   BLOOM     — radiance-only intermediates (no alpha): RG11B10Float halves
//               the pyramid's bandwidth against RGBA16F.
//   DRAWABLE  — display-referred output after tonemapping, plus the UI.
// ---------------------------------------------------------------------------
namespace PixelFormats {
inline constexpr MTLPixelFormat SCENE_HDR = MTLPixelFormatRGBA16Float;
inline constexpr MTLPixelFormat SCENE_DEPTH = MTLPixelFormatDepth32Float;
inline constexpr MTLPixelFormat BLOOM = MTLPixelFormatRG11B10Float;
inline constexpr MTLPixelFormat DRAWABLE = MTLPixelFormatBGRA8Unorm;
} // namespace PixelFormats
