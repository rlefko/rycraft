# Code Conventions

House style and structural rules. The reuse/simplification/readability review agents (see CLAUDE.md) treat this file as their source of truth; the checklist at the bottom is theirs to walk.

## One source of truth

A concept gets exactly one definition, in one header, and everyone imports it. Current canon:

| Concept | Home |
|---------|------|
| Block types + solidity/opacity/transparency | `world/block_properties.hpp` |
| Chunk keys | `world/chunk_pos.hpp` (`ChunkPos`) |
| Packed derived skylight and block light | `world/chunk.hpp` (`packDerivedLight`, `derivedSkyLight`, `derivedBlockLight`) |
| Voxel-light flood and changed-face reporting | `world/light_engine.hpp` (`LightEngine::floodChunk`) |
| Surface sampling support | `world/macro_generation.hpp` (`SurfaceFootprint`) |
| Water-body identity | `world/basin_solver.hpp` (`WaterBodyId`) |
| Lithology transition data | `world/macro_generation.hpp` (`LithologyBlend`) |
| Weighted surface materials | `world/surface_material.hpp` (`SurfaceMaterialPalette`) |
| Tree species and habitat evaluation | `world/features.hpp` (`TreeSpecies`, `evaluateTreeHabitat`, `treeCoverDensity`) |
| Exact coverage requirements | `world/world.hpp` (`ExactSurfaceCoverageSnapshot`) |
| Per-column far ownership data | `render/far_terrain.hpp` (`FarTerrainExactHandoff`) |
| Far fragment and paired skirt visibility | `render/shader_types.hpp` |
| Far LOD tiers, parent residency, and transition timing | `render/far_terrain.hpp` |
| Regional weather samples, snapshots, and storm events | `world/weather.hpp` |
| Regional weather grid resolution and spacing | `world/weather_grid.hpp` |
| Direct celestial selection, phases, true solar state, and day length | `render/celestial.hpp` (`CelestialState`, `computeCelestialState`) |
| Shadow cascade count, ranges, blends, and GPU records | `render/shader_types.hpp` and `render/shadow_map.hpp` |
| Indirect, atmosphere, cloud, lightning, and froxel GPU layouts | `render/shader_types.hpp` |
| Screen-space temporal reset reasons | `render/screen_space_lighting.hpp` |
| Coordinate-addressed generation randomness | `common/counter_rng.hpp` (`CounterRng`) |
| Seed hashing and serial visual-effect randomness | `common/random.hpp` |
| GPU-shared struct layouts | `render/shader_types.hpp` |
| Block-face → texture layer mapping | `render/block_textures.hpp` |
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

**Why:** this codebase once had four `isSolid` definitions (glass rendered solid, entities fell through it), three chunk-key schemes (one allocating strings per block access), six RNG implementations (one seeded from `random_device` in a deterministic game), five copies of GPU structs (all drifted), a hand-copied camera basis whose inverted signs walked W backwards, and separate exact and far field rules that exposed straight handoff boundaries. Duplicated definitions do not stay in sync; they diverge and each divergence is a bug.

## Naming

- **Types:** `PascalCase`. **Methods and functions:** `camelCase`. **Constants:** `SCREAMING_SNAKE_CASE`.
- **Members:** trailing underscore (`chunksMutex_`) in C++ classes; leading underscore (`_device`) only in Objective-C++ classes, matching Apple convention. Do not convert existing members wholesale. Adopt the rule when a class is substantially rewritten.
- **Enum values:** `SCREAMING_SNAKE_CASE` (`BlockType::STONE`). A few older enums (`Biome`, `ChunkLOD`, `FaceNormal`, `Key`, `GameScreen`) use `PascalCase`; new enums follow SCREAMING, and migrations happen opportunistically, not as churn commits.
- Generic STL-style utilities in `common/` may use `snake_case` APIs (`ThreadPool::submit`) where they mirror the standard library.

## Ownership and lifetimes

- `unique_ptr` by default; `shared_ptr` only when lifetime is genuinely shared (chunks handed to worker threads). Raw pointers mean **non-owning** and say so in a comment (`World::saveManager_`).
- No naked `new`/`delete`. **Why:** the render pipeline hand-deleted six subcomponents and still leaked the input manager.
- An object whose async work captures `this` joins that work in its destructor (see `World::~World`).

## Error idioms

Exactly three (details in [architecture.md](architecture.md)): fatal log-and-abort for unusable-GPU conditions, `try/catch`-with-fallback for chunk generation, and `std::optional`/`bool` + `RY_LOG_*` for I/O. No `Result` type, no error-code plumbing.

## Comments

Comments state constraints the code can't (`// valid() stays true until get()`), the defect a guard exists for, or coordinate conventions. They don't narrate the next line.

## Formatting

`.clang-format` is law (LLVM base, 4-space indent, 100 columns, left pointers). Run `clang-format` on touched files; CI checks the whole tree. Headers use `#pragma once`.

## Tests

- Behavior over structure: assert what a function returns, not how it's written. Pin conventions that broke before (matrix conventions, struct layouts, glass solidity, movement basis).
- Filesystem tests go through `tests/test_helpers.hpp`'s `TempDir`, never fixed `/tmp` paths or `std::system`.
- New module → tests in the matching `tests/test_<module>.mm`.

## Review checklist

For any diff, the reuse/simplification/readability agents check in order:

1. Does the change re-implement something that already has a home (table above, or an existing helper)? Point to the existing one.
2. Does it introduce a second definition of any concept, such as a predicate, key, or constant, that exists elsewhere?
3. New state: is ownership expressed in the type (`unique_ptr`/value), and is any non-owning pointer labeled?
4. Does anything return a new error style instead of one of the three idioms?
5. Naming consistent with this file for NEW code (don't flag legacy names the diff merely touches)?
6. Is there dead code left behind, such as parameters consumed by `(void)`, unreachable branches, or stub functions?
7. Could the diff be smaller, such as a table entry instead of a switch case or a helper call instead of a copied block?
