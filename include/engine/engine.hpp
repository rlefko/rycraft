#pragma once

#include <common/error.hpp>
#include <common/math.hpp>

#include <cstdint>

#ifdef __OBJC__

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

// Forward declare the C++ state struct so the bridge function can return it
struct EngineState;

// ---------------------------------------------------------------------------
// Engine — NSObject subclass adopting NSApplicationDelegate,
//          NSWindowDelegate, and MTKViewDelegate protocols.
//
// Implements the game loop, Metal rendering, and Cocoa integration.
// Internal game logic uses C++ types (Mat4, Vec3, etc.).
// ---------------------------------------------------------------------------
@interface Engine : NSObject <NSApplicationDelegate, NSWindowDelegate, MTKViewDelegate>

+ (instancetype)sharedEngine;

// ---- Lifecycle ----
- (BOOL)initialize;
- (void)run;
- (void)terminate;

// ---- Accessors ----
@property (nonatomic, readonly) id<MTLDevice> metalDevice;
@property (nonatomic, readonly) id<MTLCommandQueue> commandQueue;
@property (nonatomic, readonly) NSWindow* window;
@property (nonatomic, readonly) MTKView* view;

// ---- Timing ----
@property (nonatomic, readonly) double deltaTime;
@property (nonatomic, readonly) uint64_t frameCount;

@end

// C++ bridge functions — call Obj-C methods internally
Mat4 engineProjectionMatrix(Engine* engine);
CGSize engineDrawableSize(Engine* engine);

#endif  // __OBJC__
