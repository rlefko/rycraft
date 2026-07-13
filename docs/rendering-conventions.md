# Rendering Conventions

The Metal rulebook. Every rule here was earned by a real, shipped defect from the great renderer repair — the "why" lines describe bugs confirmed in this codebase, not folklore. The `render-review` skill walks this file's checklist against any rendering diff.

## Prime directive

The frame must be **provably correct before it is clever**: shared struct layouts are compile-checked, every pipeline state matches the pass that runs it, and every change is verified in the running game with Metal validation on. The renderer once carried twelve simultaneous bugs — terrain never drew at all — because none of these held.

## 1. The C++/MSL boundary

- **Every struct both sides read lives in `include/render/shader_types.hpp`**, included by the C++ engine and the `.metal` sources alike, with `static_assert` sizeof/offsetof pins mirrored in `tests/test_render.mm`. Never redeclare a GPU struct locally. **Why:** five independent copies drifted — `Uniforms` had a phantom `float _padding` only on the MSL side (fog and camera position read 16 bytes off), `SkyUniforms` packed `float[3]` against MSL `float3`, and `GPUParticle` disagreed by 16 bytes — each producing garbage that looked like a different bug.
- **Use `simd` types, not float arrays with hand-padding.** `simd_float3` has identical layout in both languages; `float[3] + float _pad` invites the next drift.
- **A buffer's length comes from `sizeof(TheStruct)`, never a literal.** **Why:** the block-highlight buffer was allocated at a literal 256 bytes while the struct grew to 272 — a heap overflow on every highlight.

## 2. Pipelines match their passes

- **`rasterSampleCount` on a pipeline equals the sample count of the textures it renders into.** The scene pass is 4x MSAA; anything encoded into it (sky, chunks, highlight, particles, clouds, entities) declares 4. **Why:** the sky and cloud pipelines declared 4 while rendering into the 1-sample drawable, and the particle pipeline declared 1 inside the MSAA pass — every frame faulted and the screen went black.
- **`depthAttachmentPixelFormat` equals the depth texture's format** (`Depth32Float` everywhere). **Why:** pipelines declared `Depth32Float` against a `Depth32Float_Stencil8` texture.
- **Fullscreen sampling passes flip V.** Metal texture v runs downward while NDC y runs up, so a pass that samples a rendered texture with `pos * 0.5 + 0.5` flips the image. **Why:** bloom's thirteen passes left the whole frame upside down, with the bloom term misaligned against the scene.

## 3. Resolution and storage

- **Scene targets are sized from the drawable (pixels), never the view bounds (points)** — Retina drawables are 2x. `render()` re-checks the drawable size each frame. **Why:** point-sized targets left the scene in the top-left quarter of the window.
- **MSAA targets are memoryless**: color resolves (`MultisampleResolve` into `_colorResolve`), depth is `DontCare`. Their contents must never be read later.
- **CPU-visible buffers are `StorageModeShared`.** On Apple Silicon unified memory this is the correct default; `contents` on a `Private` buffer returns nil. **Why:** the mega-buffer was `Private` yet memcpy'd through `contents` — the moment the terrain path came alive, it dereferenced nil.
- **A dynamic buffer written per frame is ring-buffered (3 slots), and one draw call's data is never rewritten before the GPU reads it.** The GPU consumes vertex buffers at execution time, not encode time. **Why:** the UI overlay rewrote one 64-byte buffer per quad — every HUD quad rendered as the last one written.

## 4. Coordinate conventions

- **Column-major matrices, column vectors, right-handed, Metal [0,1] NDC depth.** `Mat4::perspective` maps near→0/far→1; `lookAt` puts basis vectors in rows and translation in column 3; `extractFrustumPlanes` uses the row-form Gribb-Hartmann with `near = row2` alone. All three are pinned by tests, including a memcpy-to-`simd_float4x4` equivalence test. **Why:** `lookAt` was transposed and `perspective` used the OpenGL [-1,1] convention; nothing consuming them had ever rendered, so both hid until terrain drew.
- **Mesh vertices are chunk-local** (fp16-exact in 0..256); the per-draw `ChunkOrigin` restores world space in the vertex shader. **Why:** fp16 world-space coordinates quantize beyond ±2048 blocks.
- **Face planes sit on block boundaries**: the -X face of block `lx` is at `x = lx`, its +X face at `lx + 1`. **Why:** four of six face emitters were off by one, drawing faces a block inside the neighbor.

## 5. Verification is part of the change

- **Run the game with validation for any rendering diff:** `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. Zero validation messages is the bar.
- **Capture and look at a frame** with `RYCRAFT_CAPTURE=/tmp/frame.png` (writes a PNG from inside the frame — no screen-recording permission needed). `RYCRAFT_CAPTURE_FRAME`, `RYCRAFT_BLOOM`, and `RYCRAFT_START_SCREEN` narrow a repro. The `playtest` skill runs this end to end.
- **A rendering claim without a frame to show for it is unverified.** Twelve bugs coexisted precisely because the pipeline was never watched while it ran.

## Review checklist

For any diff touching `src/render/`, `shaders/`, or Metal API calls, check in order:

1. Any struct read by both C++ and MSL: defined once in `shader_types.hpp`, with size/offset asserts updated on both sides (header + `tests/test_render.mm`)?
2. New or changed pipeline state: `rasterSampleCount` and `depthAttachmentPixelFormat` match every pass that uses it?
3. New texture/buffer: storage mode justified (memoryless for MSAA, shared for CPU-written), sized from the drawable or `sizeof`, never a literal?
4. New fullscreen sampling pass: V flipped?
5. Per-frame dynamic data: ring-buffered or otherwise safe against encode-vs-execute overwrite?
6. Matrix or coordinate math: consistent with column-vector/[0,1]-depth conventions, and covered by the math tests if a factory changed?
7. Was the game actually run with validation on, and a captured frame inspected? Say so in the PR.
