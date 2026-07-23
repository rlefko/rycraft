# Code Conventions

House style and structural rules. The repository review agents treat this file as their source of truth, and the checklist at the bottom is theirs to walk.

## One source of truth

A concept gets exactly one definition, in one header, and everyone imports it. Current canon:

| Concept | Home |
|---------|------|
| Block types + solidity/opacity/transparency | `world/block_properties.hpp` |
| Chunk keys | `world/chunk_pos.hpp` (`ChunkPos`) |
| Generator v4 version and profile directory names | `world/generator_v4.hpp` |
| Generator v4 identity, physical scale, and page schema | `world/learned_terrain.hpp` |
| Shared learned authority and failure latch | `world/learned_terrain.hpp` (`WorldGenerationContext`) |
| Model and runtime asset pins | `resources/config/terrain_model_manifest.json` and `world/terrain_bootstrap.hpp` |
| Production runtime version and qualification | `world/terrain_runtime.hpp` |
| Production InfiniteDiffusion page backend | `world/infinite_diffusion_backend.hpp` |
| Native hydrology page admission budget | `world/native_hydrology.hpp` |
| V4 startup, dry-land spawn selection, and entry gate | `engine/v4_world_startup.hpp` |
| V4 entry preparation renderer | `render/render_pipeline.hpp` (`RenderPipeline::renderV4Preparation`) |
| World vertical bounds and section mask | `world/chunk.hpp` |
| Packed derived skylight and block light | `world/chunk.hpp` (`packDerivedLight`, `derivedSkyLight`, `derivedBlockLight`) |
| Voxel-light flood and changed-face reporting | `world/light_engine.hpp` (`LightEngine::floodChunk`) |
| Surface sampling support | `world/macro_generation.hpp` (`SurfaceFootprint`) |
| Water-body identity | `world/basin_solver.hpp` (`WaterBodyId`) |
| Lithology transition data | `world/macro_generation.hpp` (`LithologyBlend`) |
| Weighted surface materials | `world/surface_material.hpp` (`SurfaceMaterialPalette`) |
| Tree species and habitat evaluation | `world/features.hpp` (`TreeSpecies`, `evaluateTreeHabitat`, `treeCoverDensity`) |
| Exact coverage requirements | `world/world.hpp` (`ExactSurfaceCoverageSnapshot`) |
| Exact generation and first-publication-light priority lanes | `world/world.hpp` (`exactStreamingSurfacePriorityLane`, `exactStreamingFloraPriorityLane`) |
| Exact mesh admission and upload priority | `render/render_pipeline.hpp` (`exactMeshCandidatePriority`, `exactMeshUploadPriority`) |
| Renderer-published exact collision ownership | `world/world.hpp` (`ExactCollisionOwnershipSnapshot`) |
| Per-column far ownership data | `render/far_terrain.hpp` (`FarTerrainExactHandoff`) |
| Far fragment visibility | `render/shader_types.hpp` |
| Far LOD tiers, parent residency, and transition timing | `render/far_terrain.hpp` |
| Far and canopy dynamic worker admission | `render/render_pipeline.hpp` (`farTerrainWorkerBudget`, `farTerrainCanopyWorkerBudget`) |
| Resolved per-world profile paths | `world/save_manager.hpp` (`SaveManager::Profile`) |
| Regional weather samples, snapshots, and storm events | `world/weather.hpp` |
| Regional weather grid resolution and spacing | `world/weather_grid.hpp` |
| Direct celestial selection, phases, true solar state, and day length | `render/celestial.hpp` (`CelestialState`, `computeCelestialState`) |
| Shadow cascade count, ranges, blends, and GPU records | `render/shader_types.hpp` and `render/shadow_map.hpp` |
| Indirect, atmosphere, cloud, lightning, and froxel GPU layouts | `render/shader_types.hpp` |
| Screen-space temporal reset reasons | `render/screen_space_lighting.hpp` |
| Coordinate-addressed generation randomness | `common/counter_rng.hpp` (`CounterRng`) |
| Seed hashing and serial visual-effect randomness | `common/random.hpp` |
| GPU-shared struct layouts | `render/shader_types.hpp` |
| Opaque and resolved scene pixel-format contracts | `render/pixel_formats.hpp` |
| Block-face → texture layer mapping | `render/block_textures.hpp` |
| Per-texel block emission masks | `render/block_textures.hpp` (`emissionMaskForTexel`) |
| Screen-space indirect-light history and reset policy | `render/screen_space_lighting.hpp` (`IndirectHistoryState`) |
| Menu geometry (drawn AND hit-tested) | `render/ui_menu.hpp` |
| Items, stacks, drops, item colors, and the mining-time formula | `world/item.hpp` |
| Crafting and smelting recipes and fuels | `world/recipes.hpp` |
| Furnace state and its 20 Hz step | `world/furnace.hpp` |
| Chest storage state (27 slots) | `world/chest.hpp` |
| Game mode rules and per-world generation toggles | `world/world_config.hpp` |
| World enumeration, creation, and deletion | `world/world_list.hpp` |
| Player inventory (36 slots, hotbar selection) | `engine/inventory.hpp` |
| Slot click interaction (pick/place/split/quick-move, drag distribution, double-click gather) | `engine/slot_interaction.hpp` |
| Survival food/air/regen timers and eating | `engine/survival.hpp` |
| Block mining progress and tracking | `engine/mining.hpp` |
| Melee entity picking | `entity/entity_picking.hpp` |
| Dropped item entities | `entity/item_entity.hpp` |
| Rideable boats (float physics, riding, picking) | `entity/boat.hpp` |
| Item-icon texture layers and slot drawing | `render/block_textures.hpp`, `render/ui_item_icon.mm` |
| Yaw/pitch → look direction | `common/math.hpp` (`directionFromYawPitch`) |

**Why:** this codebase once had four `isSolid` definitions, three chunk-key schemes, six RNG implementations, five copies of GPU structs, a hand-copied camera basis with inverted signs, and separate exact and far field rules. Duplicated definitions diverge, and each divergence becomes a bug.

The model manifest is serialized data for review and tooling. Runtime code still validates its own pinned `TerrainAssetSpec` list. A change to either location must update the other in the same commit and retain focused tests that compare all names, sizes, and hashes.

## Naming

- **Types:** `PascalCase`. **Methods and functions:** `camelCase`. **Constants:** `SCREAMING_SNAKE_CASE`.
- **Members:** trailing underscore (`chunksMutex_`) in C++ classes; leading underscore (`_device`) only in Objective-C++ classes, matching Apple convention. Do not convert existing members wholesale. Adopt the rule when a class is substantially rewritten.
- **Enum values:** `SCREAMING_SNAKE_CASE` (`BlockType::STONE`). A few older enums (`Biome`, `ChunkLOD`, `FaceNormal`, `Key`, `GameScreen`) use `PascalCase`; new enums follow SCREAMING, and migrations happen opportunistically, not as churn commits.
- Generic STL-style utilities in `common/` may use `snake_case` APIs (`ThreadPool::submit`) where they mirror the standard library.
- Generator v4 seeds remain `uint64_t` at public boundaries, constructors, metadata, cache keys, and diagnostics. A narrowing cast is permitted only at an explicitly named legacy subsystem adapter.

## Ownership and lifetimes

- `unique_ptr` by default; `shared_ptr` only when lifetime is genuinely shared (chunks handed to worker threads). Raw pointers mean **non-owning** and say so in a comment (`World::saveManager_`).
- No naked `new`/`delete`. **Why:** the render pipeline hand-deleted six subcomponents and still leaked the input manager.
- An object whose async work captures `this` stops admission and joins that work before member destruction (see `World::~World`). A worker task must not retain ownership of the pool that executes it. Pool ownership stays with the lifecycle owner, which invokes the idempotent shutdown boundary before releasing the pool.
- Application termination must use `ApplicationTerminationQuiescence`. Save failure remains cancelable before teardown begins. After persistence succeeds, stop render-owned workers, release `World` and its generation workers, join bootstrap, release generation contexts, and finally destroy the inference runtime. Do not rely on Objective-C singleton destruction or unload the process-lifetime ONNX image.

## Error idioms

Use the established boundary for each subsystem:

- Fatal log and termination for an unusable Metal device or pipeline
- `std::optional` or `bool` with logging for legacy local I/O
- `AuthorityResult<T>` with `READY`, `DEFERRED`, or `FAILED` for learned authority
- `TerrainRuntimeStepResult` and `TerrainBootstrapFailure` for startup preparation
- `GenerationFailureException` only when a synchronous legacy consumer cannot return a typed authority result

Generator v4 failures are fail-closed. Do not catch one and synthesize terrain, publish an empty cube, or select v3. The first production learned failure is latched in `WorldGenerationContext`.

## Comments

Comments state constraints the code can't (`// valid() stays true until get()`), the defect a guard exists for, or coordinate conventions. They don't narrate the next line.

Use American English. Do not use em dashes in repository text. Do not add tool attribution, generated boilerplate, or AI authorship. Required third-party technical and copyright notices remain factual notices, not commit co-authorship.

## Formatting

`.clang-format` is law (LLVM base, 4-space indent, 100 columns, left pointers). Run `clang-format` on touched files; CI checks the whole tree. Headers use `#pragma once`.

## Tests

- Behavior over structure: assert what a function returns, not how it's written. Pin conventions that broke before (matrix conventions, struct layouts, glass solidity, movement basis).
- Filesystem tests go through `tests/test_helpers.hpp`'s `TempDir`, never fixed `/tmp` paths or `std::system`.
- New module → tests in the matching `tests/test_<module>.mm`.
- Ordinary CI uses `DeterministicFakeTerrainBackend` and never downloads model assets. Real-model Core ML qualification is an explicit local suite with recorded hashes and hardware evidence.
- Water regressions reject wet-route deletion, dry-terrain raising, unowned stage jumps, unsupported water, step-32 topology loss, and nonzero production skirt quads.
- V4 surface regressions keep the legacy `BasinSolver` hydraulic-erosion and synthetic post-hydrology detail paths out of learned-authority sampling. A new residual-detail path needs topology-preservation coverage before it can run in v4.
- Near-field regressions keep every required exact surface through 32 chunks ahead of optional flora
  and broad work across generation, meshing, upload, and first-publication lighting.
- Collision regressions require a matching exact coverage epoch before loaded blocks or fluids become
  authoritative; unowned planned sections use canonical generated proxies and unresolved sections
  remain closed.
- Residency regressions permit only a requested protected FINAL role-selected key to reclaim optional
  distant CPU or GPU state and use the complete arena. Structural coverage and transition owners stay
  pinned.

## Review checklist

For any diff, the reuse/simplification/readability agents check in order:

1. Does the change re-implement something that already has a home (table above, or an existing helper)? Point to the existing one.
2. Does it introduce a second definition of any concept, such as a predicate, key, or constant, that exists elsewhere?
3. New state: is ownership expressed in the type (`unique_ptr`/value), and is any non-owning pointer labeled?
4. Does anything return a new error style instead of one of the three idioms?
5. Naming consistent with this file for NEW code (don't flag legacy names the diff merely touches)?
6. Is there dead code left behind, such as parameters consumed by `(void)`, unreachable branches, or stub functions?
7. Could the diff be smaller, such as a table entry instead of a switch case or a helper call instead of a copied block?
8. Does a v4 path preserve the unsigned 64-bit seed and complete generation fingerprint?
9. Can any authority failure silently enter legacy macro generation?
10. Does any water correction raise dry terrain, delete a wet route, or reintroduce a retaining wall?
11. Does any far path block terrain and water on flora work, retire far vegetation before exact flora sections are ready, or emit a downward skirt?
12. Does any v4 path reintroduce legacy hydraulic erosion or synthetic post-hydrology relief without a topology-preservation proof?
13. Can optional flora, canopy, prediction, or distant work pass required exact surfaces through 32
    chunks, current protected FINAL work, or urgent protected coverage?
14. Can exact collision publish from a stale coverage epoch, or can a first-visible mesh bypass its
    pending bounded lighting transaction?
15. Were changed text files checked for American English, em dashes, and authorship artifacts?
