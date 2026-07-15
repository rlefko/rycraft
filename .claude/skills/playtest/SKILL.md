---
name: playtest
description: Build and run rycraft with Metal validation, use deterministic diagnostics to locate features, capture real PNG frames, inspect cubic terrain, adaptive far LOD, mipmapped textures, culling, HDR effects, partial water, flora, and fauna, and report what the game actually shows. Use for player-visible changes, rendering reviews, world-generation acceptance, menus, fauna, fluids, culling, LOD, textures, or screenshot requests.
---

# Playtest

Run the real game, inspect its logs and captures, and report observed behavior. Do not infer visual success from a passing build.

## 1. Build and test

```bash
ninja -C build
ninja -C build test
```

If `build` does not exist, run `meson setup build` first. Stop on a compile or test failure and report the first useful error.

For world-generation work, also locate the diagnostic target:

```bash
find build -type f -name 'rycraft_worldgen_inspect' -perm -111
```

If it is absent, report that deterministic feature discovery is unavailable and continue with fixed seed and spawn captures that do not require it.

## 2. Create an isolated run directory

Never playtest against the normal save directory.

```bash
rm -rf /tmp/rycraft-playtest
mkdir -p /tmp/rycraft-playtest/captures
```

Run the executable with `/tmp/rycraft-playtest` as the working directory so `./rycraft_world` remains disposable.

## 3. Fix the world and discover features

Choose and record one seed, normally 42 unless the change provides a regression seed. Pass it as the inspector's optional positional argument and save the JSON report. The positional seed takes precedence over `RYCRAFT_WORLD_SEED`.

```bash
RYCRAFT_WORLD_SEED=42 <repo>/build/src/rycraft_worldgen_inspect 42 > /tmp/rycraft-playtest/inspect-42-a.json
RYCRAFT_WORLD_SEED=42 <repo>/build/src/rycraft_worldgen_inspect 42 > /tmp/rycraft-playtest/inspect-42-b.json
jq 'del(.benchmark) | .far_terrain.tile_builds |= map(del(.milliseconds))' /tmp/rycraft-playtest/inspect-42-a.json > /tmp/rycraft-playtest/inspect-42-a-stable.json
jq 'del(.benchmark) | .far_terrain.tile_builds |= map(del(.milliseconds))' /tmp/rycraft-playtest/inspect-42-b.json > /tmp/rycraft-playtest/inspect-42-b-stable.json
cmp /tmp/rycraft-playtest/inspect-42-a-stable.json /tmp/rycraft-playtest/inspect-42-b-stable.json
```

The current inspector reports candidates and surface samples for mountain, cliff, canyon, river, confluence reach, lake, endorheic lake, waterfall, delta, volcano, oceanic island, biome transition, dense flora habitat, and deep generated fish water. It also reports column-plan counts and separate basin-cache bytes, hits, misses, builds, failures, active cold builds, peak cold builds, and throttled cold-build requests. Record every non-null coordinate relevant to the review. Pass an optional `sample_x sample_z` pair after the seed for another exact surface report. Find caves, aquifers, and lava tubes through the fixed fixtures or additional observation because surface sampling does not locate interiors directly.

The regression suite provides stable direct samples when a rare feature needs a known starting point:

- Seed 42 at X=-8235, Z=2976 is a shallow supported nonendorheic lake lip.
- Seed 42 crosses an incised river at the X=-12288 cube face at Z=2653 and Z=2654, while X=-23904, Z=0 provides a separate canyon ecotope.
- Seed 42 at X=-8272, Z=3056 is an elevated nonendorheic lake whose receiver-centered outlet fall is anchored at X=-8256, Z=3072 over a lower standing river.
- Seed 42 at X=-23904, Z=0 is a sediment-bearing delta with four distributaries.
- Seed 42 at X=-518996, Z=-385073 is an oceanic hotspot island with an accepted volcanic cone.
- Seed 42 at X=-81792, Z=126976 is an exposed snow peak above Y=340.
- Seed 764891 at X=23029, Z=-111486 has a volcanic conduit beneath a crater floor near Y=297 and a settled lake surface near Y=307. Its warped shoreline has a complete supported dry rim and at least one block of freeboard.
- Seed 764891 at X=-1443, Y=-84, Z=-1500 is inside a sealed aquifer pocket.

For the complete world-generation review, find candidates for:

- Mountain or cliff
- Canyon or gorge
- River and confluence
- Lake
- Waterfall
- Delta
- Volcanic island or strong hotspot
- Biome transition
- Dense flora
- Cave entrance or low subterranean section
- Deep generated water for fish

Feature coordinates, sample values, column-plan counts, basin-cache work counts, and route hashes must match exactly. Timing fields are machine-specific and may differ between identical runs.

## 4. Capture with Metal validation

The engine supports:

| Variable | Effect |
|---|---|
| `RYCRAFT_WORLD_SEED=N` | Override metadata seed for a repeatable world |
| `RYCRAFT_SPAWN=x,y,z` | Override the starting position |
| `RYCRAFT_WORLDGEN_OVERLAY=geology\|hydrology\|climate\|biome` | Show one of the four world-generation overlays |
| `RYCRAFT_SHOW_DEBUG=1` | Show F3 diagnostics without changing terrain colors |
| `RYCRAFT_CAPTURE=/absolute/path.png` | Write one frame to a PNG |
| `RYCRAFT_CAPTURE_FRAME=N` | Choose the capture frame; 400 through 600 usually allows streaming to settle |
| `RYCRAFT_NATIVE_WINDOW=1` | Fill the current display so captures and acceptance runs use its native backing resolution |
| `RYCRAFT_START_SCREEN=title\|playing\|paused\|settings` | Choose the initial screen |
| `RYCRAFT_BLOOM=0..1` | Scale or disable bloom |
| `RYCRAFT_VIEW_DISTANCE=4..256` | Override visible distance; exact simulation remains capped at radius 32 |
| `RYCRAFT_TIME=0..23999` and `RYCRAFT_TIME_FREEZE=1` | Pin time of day for repeatable shadow, sky, and flare captures |
| `RYCRAFT_WEATHER=rain\|clear` | Pin weather and its wetness state |
| `RYCRAFT_YAW=degrees` and `RYCRAFT_PITCH=degrees` | Point the capture camera after spawn validation |
| `RYCRAFT_AUTOPILOT=walk\|sprint\|fly` | Exercise a repeatable ground route or an obstacle-independent aerial streaming route |
| `RYCRAFT_AUTOPILOT_START_FRAME=N` and `RYCRAFT_AUTOPILOT_STOP_FRAME=N` | Bound movement to a fixed interval so queue settling can be measured afterward |
| `RYCRAFT_AUTOPAUSE_FRAME=N` | Enter the real paused screen at a fixed frame for a same-scene playing-versus-paused timing comparison |
| `RYCRAFT_PERF_WARMUP_FRAMES=N` and `RYCRAFT_PERF_FRAMES=N` | Exclude warmup, record a bounded performance window, print summary lines, and quit |
| `RYCRAFT_SHADOWS=0..2`, `RYCRAFT_CLOUDS=0..2` | Override shadow and cloud quality without saving preferences |
| `RYCRAFT_VL`, `RYCRAFT_SSAO`, `RYCRAFT_SSR`, `RYCRAFT_WAVING`, `RYCRAFT_LENS_FLARE` | Toggle individual graphics effects with 0 or 1 |
| `RYCRAFT_VIBRANCE=0..10`, `RYCRAFT_SHARPEN=0..10` | Override final-grade controls |
| `RYCRAFT_GPU_COUNTERS=1` | Enable diagnostic per-pass GPU timestamps |
| `RYCRAFT_SPAWN_LAVA=1`, `RYCRAFT_SPAWN_WATER=1` | Create disposable validation scenes near spawn |

For each location, launch from the scratch directory with validation:

```bash
cd /tmp/rycraft-playtest
MTL_DEBUG_LAYER=1 \
MTL_DEBUG_LAYER_ERROR_MODE=nslog \
MTL_SHADER_VALIDATION=1 \
RYCRAFT_WORLD_SEED=<seed> \
RYCRAFT_SPAWN=<x>,<y>,<z> \
RYCRAFT_CAPTURE=/tmp/rycraft-playtest/captures/<name>.png \
RYCRAFT_CAPTURE_FRAME=500 \
RYCRAFT_START_SCREEN=playing \
<repo>/build/src/rycraft > /tmp/rycraft-playtest/<name>.log 2>&1 &
```

Poll for the capture for no more than 45 seconds, then terminate only that recorded process ID. Do not use a broad `killall` that could stop another workspace's game.

## 5. World-generation capture matrix

Use inspector coordinates rather than wandering at random. Choose spawn Y based on the sampled terrain or water elevation.

### Aerial terrain

Spawn 40 to 100 blocks above the sampled surface with flight available. Capture a mountain or cliff, canyon or gorge, river confluence, lake, waterfall, delta, volcanic island, and biome transition. Use a wider view distance only when the feature needs it, and record that override.

Inspect for:

- Coherent large-scale relief rather than isolated noise spikes
- Rivers occupying eroded beds, increasing order at confluences, and meeting without one-block steps
- Flat spill and terminal lake surfaces with valid outlets where applicable
- Canonical lake shore water occupying supported columns rather than floating above land
- Waterfall sides reaching the plunge area without replacing the lower standing-water surface
- Two through four delta distributaries and sediment meeting a shallow receiving body
- Distinct shield and stratovolcano profiles, volcanic arcs, islands, calderas or craters, settled crater lakes, and basalt or ash fields. A crater lake must have an irregular shoreline, a complete supported dry rim, and at least one block of freeboard; a routed nonendorheic lake may retain its named outlet.
- Gradual climate-biome transitions without terrain walls

### Far horizon and LOD transitions

Use `RYCRAFT_VIEW_DISTANCE=256` for dedicated far-horizon captures. Exact editable cubic terrain stops at radius 32. Immutable 256 by 256-block tiles cover the half-open annulus `[32, 256)`. A narrow two-block sampling tier immediately outside the exact boundary samples exact emitted density heights as the topology bridge. Exact opaque terrain draws first, while positively depth-biased far tops remain behind it as lit fallback for cold exact meshes. Water and canopy summaries use a stable world-space dither over the following 16 blocks. Farther out, four-, eight-, and sixteen-block steps are selected per tile from distance plus immutable slope and hydrology complexity using bounded, tunable thresholds. Asymmetric refine and coarsen thresholds stabilize the selection, and a 0.4-second fog-hidden transition masks resident topology replacement.

Capture:

- Direct surface agreement between exact cubes and the two-block far tier at radius 32
- Lit opaque fallback during cold exact residency and identical 16-block dithered handoff coverage for water and canopies
- A dense forest spanning the exact boundary and all four far tiers
- The gradual taper from the two-block bridge into 4/8/16 tiers across flat terrain
- Complex terrain at similar distances to confirm that slope and hydrology retain detail farther out
- Refine and coarsen hysteresis crossings in both travel directions
- One resident topology replacement through its fog-hidden transition
- The visible horizon near radius 256
- A broad turn that reveals tiles behind the starting camera
- A nearer ridge in front of lower ground and a taller distant peak

At the exact-to-far handoff, compare matching terrain profiles and water boundaries on both sides of radius 32. Inspect for no height step, crack, duplicate surface, missing strip, water wall, material pop, or material-specific handoff ring through the following 16 blocks. Capture once while exact queues are still active and reject any black ring or vertical panel; the far top fallback must remain lit behind resident exact depth, and skirts must appear only on a finer edge beside a resident coarser tile. Generated source-water tops must meet at the same 0.875-block plane. In the forest capture, verify that steps 2 and 4 retain every accepted canopy and that steps 8 and 16 replace exact priority competition with grounded aggregate forest clusters rather than becoming barren. Confirm there is no forest-density cliff at any tier and no doubled far impostor over exact trees inside radius 32. Across the rest of the horizon, inspect for a visibly granular taper, continuous borders, no exposed skirts or cracks, no inward-wound missing faces, contour-clipped stable water coastlines, no ring-shaped material jumps, no topology popping or boundary chatter, and no false occlusion of the taller peak. Compare F3 wanted, resident, drawn, frustum-culled, horizon-culled, pending, cache, and arena counters before and after the turn.

Use oblique views of distant textured slopes and alpha-cutout flora to inspect the complete block-texture mip chain. Look for reduced shimmer and moire patterns, preserved foliage coverage, no sudden mip bands, and the expected crisp nearest-filtered appearance when magnified nearby.

The implementation uses adaptive immutable tile tiers, not a literal geometry clipmap. Its conservative 256-bin terrain-horizon test is not a hierarchical Z buffer. Draws are bounded direct indexed commands, not an indirect command buffer. Report only the mechanisms and evidence actually observed.

### Cubic vertical exploration

Capture one high section above Y=256 and one low section below Y=0. Capture a cave or overhang near a horizontal cube edge and another view looking across a top or bottom cube face. Underground, move laterally across the edge of the six-chunk exploration radius and vertically across the edge of its four-cube band, then wait for the queues to settle and capture both directions. For the complete world-generation review, also inspect a volcanic conduit or lava tube and the fixed sealed aquifer sample.

Inspect for:

- Correct Y origin and no repeated section at Y=0
- No missing slabs at cube boundaries
- No hidden interior face walls, light seams, or flora clipping
- Nearby underground cubes loading and meshing before distant exposed surface work
- Any temporarily unavailable aboveground boundary following a lit terrain silhouette without a full black panel
- Any temporarily unavailable underground opening remaining a dark inward cap rather than a bright or interactive void
- No skylight passing through an unloaded vertical gap above the camera and no bright void at an unloaded boundary below it
- Continuous conduit and lava-tube geometry across cube faces
- An aquifer water pocket enclosed by its clay or limestone shell rather than flooding a connected cave
- Bedrock behavior near Y=-128 and open headroom below Y=511 when reachable

Aim the block ray at a temporarily unavailable boundary, attempt both break and placement, and inspect the world before and after. The ray must stop, neither action may force-load or mutate the missing cube, and collision must remain closed. Repeat after the neighbor loads to confirm the cap disappears and ordinary interaction resumes against real blocks.

### Water and underwater

Capture a river or lake from above, a partial flowing edge when available, a waterfall from the side, and a deep-water view with the camera below the actual fluid surface. For seed 42, capture the supported lake lip at X=-8235, Z=2976, the incised river face near X=-12288, Z=2653, and the canyon ecotope at X=-23904, Z=0 as separate views. Also capture the elevated lake near X=-8272, Z=3056 and its receiver-centered outlet fall into the lower river at X=-8256, Z=3072 from exact range and from each far LOD. For seed 764891, circle the caldera at X=23029, Z=-111486 and capture its bank from water level.

Inspect for:

- Eighth-block level changes and smooth corners
- No vertical water walls at stable river, lake, ocean, delta, cube, or unavailable-neighbor edges
- No floating lake top over dry land, and a solid block below the lowest water voxel at shallow shore occupancy
- Standing generated water filling every voxel from its supported floor through the surface as implicit source state, including across a cube face
- A complete irregular dry caldera rim with at least one block of freeboard and no unsupported gap; named outlets remain valid for routed nonendorheic lakes
- Stable source and flowing cells showing top geometry only
- Explicit falling columns retaining vertical sides and consistent shading
- A narrow outlet fall centered on its lower receiver, reaching from the lower visible water plane to the upper lip without a long horizontal slab, a gap, or a raised receiving body
- One complete five-quad far outlet prism at steps 2, 4, 8, and 16, with half-open anchor ownership and no duplicate wall on a neighboring tile
- Far partially wet shoreline cells following contour-clipped edges rather than rectangular sheets
- Far generated source-water tops matching the exact 0.875-block plane at the handoff
- Back-to-front ordering that includes vertical distance
- Refraction, depth absorption, caustics, fog, and underwater overlay
- Fish remaining inside water and rendering below the surface

Generated water should show a zero pending-fluid count until disturbed. Runtime edits cannot be proven by a static startup capture alone. For disturbed-water acceptance, perform a manual edit near a loaded boundary, observe downward-first spread and level decay, leave and return to resume the frontier, restart to verify persistence, and capture before and after. Pair this manual result with the automated fluid-rule tests.

### Flora and fauna

Capture dense vegetation, land wildlife, wetland frogs, alpine goats, and underwater fish at deterministic habitat coordinates when those populations appear.

Inspect for cross-cube exact trunks and canopies, grounded plants, lily-pad orientation, distinct voxel models, fish confinement, and the absence of animals on invalid cliffs or dry fish spawns. At long range, confirm grounded trunk-and-crown canopy impostors persist at every tier. Steps 2 and 4 must match exact accepted anchors, while steps 8 and 16 must retain stable aggregate forest mass across tile ownership boundaries. Every far form must disappear beneath exact trees rather than overlap them inside radius 32.

### Diagnostics and F3

At one fixed coordinate, capture an F3 frame and compare every displayed value with the inspector or a direct sample at that block, allowing only documented display rounding. The F3 Cache entry count and MiB value combine column plans and basin solutions; reconcile them with the inspector's separate cache fields rather than comparing only one cache. Record exact loaded and meshed cubes, far resident and drawn tiles, frustum and horizon culls, far pending work, far cache MiB, far arena MiB, fluid work, pending-update drops, deferred-frontier drops, and mesh coalescing. Capture `RYCRAFT_WORLDGEN_OVERLAY=geology`, `hydrology`, `climate`, and `biome` separately. Verify each overlay is visibly distinct, agrees with the sampled field, and remains stable across cube and catchment boundaries. Any other nonempty value is invalid.

### M4 Max performance acceptance

Run performance acceptance separately from Metal validation with an optimized build. Use an identified Apple M4 Max, native display resolution, 4x MSAA, and the user-reported seed-764891 starting view:

```bash
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=256 \
<repo>/build/src/rycraft
```

After the static view settles, repeat it with `RYCRAFT_AUTOPILOT=fly` and record the warmup, movement start and stop frames, performance window, and resulting route. Record the M4 Max configuration, macOS version, resolution, commit, and settings. Confirm exact simulation remains capped at radius 32 throughout.

Allow streaming to settle, then measure the moving route long enough to expose repeated streaming work. Record frame p50 and p95, the lowest sustained one-second frame rate, queue maxima and settle time, loaded and mesh-resident maxima, peak process RSS, and peak Metal allocated or resident memory. Acceptance requires:

- Lowest sustained one-second frame rate of at least 60 FPS
- No sustained generation-related frame time above 20 ms
- At most 64 GB total unified-memory use
- No more than 32,768 loaded exact cubes or 16,384 exact mesh-resident cubes
- Far CPU cache at or below 1,024 tiles and 512 MiB
- Far GPU arenas at or below 256 MiB of vertices and 128 MiB of indices
- Exact generation, exact mesh, and far queues settling within five seconds after movement stops

Record frame p50 and p95, the lowest sustained one-second frame rate, exact loaded and meshed maxima, far wanted, resident, drawn, frustum-culled, horizon-culled, pending, cache, and arena maxima, upload maxima, and queue settle time. Apple Silicon uses unified memory. Report process RSS and Metal allocation or resident counters separately, then state the highest credible unified-memory total without adding overlapping counters.

At the same settled camera and settings, record a playing interval followed by a paused interval. Attribute fixed-tick CPU p50, p95, and maximum with a profiler or scoped measurement. Pausing may remove simulation work, but it must not conceal a material main-thread bottleneck or produce an unexplained frame-rate multiplication. Hardware results are acceptance evidence, not pass or fail CI checks. CI continues to enforce deterministic work limits, queue caps, cache bounds, and allocation invariants.

## 6. Inspect logs and images

For every run:

```bash
rg -n '\[MTLDebug\]|validation|\[ERROR\]|\[FATAL\]' /tmp/rycraft-playtest/*.log
rg -n 'Render:|pending|loaded' /tmp/rycraft-playtest/*.log
```

Any Metal validation message is a failure. Engine errors are separate failures. Confirm heartbeat frame numbers advance and generation and mesh queues settle within five seconds after the camera stops.

Open every PNG and actually inspect it. A file that exists but is black, frozen, pointed at empty sky, or captured before terrain settled is not evidence.

## 7. Report

Output in this order:

1. **Verdict:** works as intended, broken, or works with concerns
2. **Build and tests:** commands and result
3. **Validation:** exact validation-message count and engine errors
4. **Deterministic setup:** seed, inspector arguments, spawn coordinates, view distance, overlay, and capture frame
5. **Evidence:** each capture path and what it demonstrates
6. **Performance observations:** frame p50 and p95, lowest sustained one-second frame rate, playing-versus-paused comparison, fixed-tick CPU attribution, generation, mesh, queue, system memory, GPU memory, and fluid diagnostics that were actually recorded
7. **Unverified behavior:** audio, input feel, disturbed runtime water, rare feature classes, or anything not directly exercised

Do not claim a feature was inspected if its frame was not opened.
