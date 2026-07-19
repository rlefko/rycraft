#pragma once

#import <Metal/Metal.h>

// Metal follows Objective-C ownership conventions. Most renderer code builds
// with ARC, but the supported manual-reference-counting configuration must
// release every object received through a create/copy/new method as well.
// Keeping this small bridge at ownership boundaries lets the same renderer
// sources support both configurations without retaining transient descriptors
// or retired textures indefinitely.
inline void releaseMetalObject(id object) noexcept {
#if __has_feature(objc_arc)
    (void)object;
#else
    [object release];
#endif
}

template <typename MetalObject>
inline void resetMetalObject(MetalObject& object) noexcept {
    releaseMetalObject(object);
    object = nil;
}
