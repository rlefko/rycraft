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

The current inspector reports candidates and surface samples for mountain, cliff, canyon, river, confluence reach, lake, endorheic lake, waterfall, delta, volcano, oceanic island, biome transition, dense flora habitat, and deep generated fish water. It also reports column-plan counts plus separate basin, shoreline-contour, and macro-control cache bytes, hits, misses, builds, failures, active cold builds, peak cold builds, and throttled cold-build requests. Record every non-null coordinate relevant to the review. Pass an optional `sample_x sample_z` pair after the seed for another exact surface report. Find caves, aquifers, and lava tubes through the fixed fixtures or additional observation because surface sampling does not locate interiors directly.

The regression suite provides stable direct samples when a rare feature needs a known starting point:

- Seed 42 at X=-557, Z=379 is the reported open-water level regression. Exact block-resolution sampling, every filtered footprint, the column plan, the solid floor, and the complete implicit-source volume must agree there.
- Seed 42 at X=-8235, Z=2976 is a shallow supported nonendorheic lake lip.
- Seed 42 crosses an incised river at the X=-12288 cube face at Z=2653 and Z=2654, while X=-23904, Z=0 provides a separate canyon ecotope.
- Seed 42 at X=-8272, Z=3056 is an elevated nonendorheic lake whose receiver-centered outlet fall is anchored at X=-8256, Z=3072 over a lower standing river.
- Seed 42 at X=-23904, Z=0 is a sediment-bearing delta with four distributaries.
- Seed 42 at X=-518996, Z=-385073 is an oceanic hotspot island with an accepted volcanic cone.
- Seed 42 at X=-81896, Z=126960 is an exposed snow peak above Y=340.
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
| `RYCRAFT_START_SCREEN=title\|worlds\|create\|delete\|playing\|paused\|settings\|video\|inventory\|crafting\|furnace\|death` | Choose the initial screen; the gameplay and container tokens auto-start the world in `RYCRAFT_WORLD_DIR`, and `furnace`/`death` seed sample state to capture |
| `RYCRAFT_WORLD_DIR=path` | World directory the auto-start tokens open (default: `rycraft_world`, else `saves/default`) |
| `RYCRAFT_GAME_MODE=survival\|creative` | Force the game mode for this session without touching the saved metadata |
| `RYCRAFT_SPAWN_ITEMS=N` | Scatter N dropped items on the loaded ground ahead of spawn to capture item entities |
| `RYCRAFT_BLOOM=0..1` | Scale or disable bloom |
| `RYCRAFT_VIEW_DISTANCE=4..512` | Override visible distance; exact simulation remains capped at radius 32 |
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

Use the inspector's former-grid report to choose views crossing 8-, 16-, 32-, 64-, 2,048-, and 8,192-block lines. Capture geology, hydrology, climate, and biome overlays on both sides of representative lines. Reject a lake edge, river, material contact, biome patch, or untagged geological contact that follows the line for more than 24 blocks. Compare nearby shifted lines so a real fault or cliff is not mistaken for a storage artifact.

The current categorical regression passes with a longest former-line run of 9 blocks, but the separate continuous-field matrix remains deferred with 15 failing assertions. Its terrain ratios reach 0.105649 at 2,048 blocks and 0.076197 at 8,192 blocks, aggregate shoreline energy is 0.194688, shoreline structured orientation is 1.674842, and biome suitability fails multiple spacings. Run `./build-release/tests/test_rycraft "[.known-continuity-debt]"` and record the expected failure. Do not mark procedural continuity accepted, and do not attribute these field measurements to the separate cold exact-to-far residency defect.

### Far horizon and LOD transitions

Use `RYCRAFT_VIEW_DISTANCE=512` for dedicated far-horizon captures. Exact editable cubic terrain has a nominal radius of 32. Every 256 by 256-block tile intersecting the complete visible disk requests an immutable step-32 parent before optional refinement, including tiles inside that nominal radius. A broad nearest-first lane advances missing parents, and resident active parents remain pinned. Eight far workers reserve four slots for missing parents and admit at most four connected urgent refinements while base work remains. Each connected coordinate requests its distance-selected step-16, step-8, step-4, or step-2 target before the complete parent disk is resident. Every far-owned fragment in the camera exploration band requires step 2 before any far topology can display. Every other far-owned fragment in the exact overlap requires step 8 or finer, including fragments in a fully ready tile whose exact requirements cover only part of a boundary. Their step-32 parents remain resident but hidden, and these protected jobs bypass ordinary grace and transition-cap admission. A refinement still requires its own parent to be resident. The parent frontier tracks missing step-32 dependencies, while a separate drawable frontier also treats protected base-only tiles as missing. The drawable frontier suppresses farther resident islands and fades the preceding 256 blocks into fog until each protected tile reaches its minimum tier. Published exact requirements and unresolved columns define one 256-bit ownership mask per far tile, with one bit per 16 by 16-block chunk column. Missing requirements keep a column far-owned, while empty completed meshes count as ready. Each draw also uses the eight neighboring tile masks so crossing canopies and waterfalls query their destination columns. A partially masked patch cannot become an occluder, while the nearest-gap distance remains a conservative fallback and diagnostic. One-, two-, four-, eight-, sixteen-, and thirty-two-block footprints filter detail without changing hydrology or feature ownership. Distance plus immutable slope and hydrology complexity selects refinements. Asymmetric refine and coarsen thresholds stabilize ordinary selection. Production filtered tiers can cross by several blocks, so a narrow terrain-only fog pulse hides one atomic complete-topology swap for ordinary replacements. Canopies use the full 0.65-second target-in, source-out exchange, and unswayed world coordinates govern both transition and coverage ownership. Skirts follow the complete terrain topology currently visible, while source water retains ownership until the transition completes.

Use the seed-42 view near X=69.7936, Y=85.7918, Z=-1472.94 to inspect cold exact-to-far ownership and exact missing-halo closure together. Missing exact lateral faces must show a lit planned continuation aboveground or a dark inward cap underground, and missing vertical faces must show bedrock caps until the real halo invalidates them. Record any straight frontier, crack, square hole, or cap that survives halo arrival as a failure. Separately time parent publication: terrain, water, and canopies still share one synchronous payload, and measured cold canopy work ranges from 250 to 1,165 milliseconds. Staged canopy attachment remains follow-up performance debt, so the two-second cold-horizon target still requires direct reference-route evidence.

Capture:

- Cold startup before the full-disk parent set is complete, with one connected foreground and no farther resident islands across a gap
- A rapid camera jump that cancels stale work and rebuilds the parent frontier nearest-first
- The 256-block drawable-coverage-frontier fade, with no partially faded terrain acting as an occluder
- Direct surface agreement between exact cubes and every filtered footprint
- Lit step-2 fallback for every far-owned exploration-band fragment and step-8-or-finer fallback for every other far-owned exact-overlap fragment, including a fully ready partial boundary tile, with step 32 hidden in both protected classes
- The seed-42 exact-to-far fixture near X=69.7936, Y=85.7918, Z=-1472.94, with ownership, closure-cap replacement, and synchronous parent-publication timing recorded separately
- A mixed far tile with ready and incomplete exact columns, plus a crossing canopy or waterfall at a tile face, confirming the 3 by 3 mask neighborhood and non-occluding partial patch
- A dense forest spanning the exact boundary and all five far tiers
- The gradual taper through the visible 32/16/8/4/2 chain across flat terrain before full-disk parent completion
- Protected fallback completion when ordinary grace and the topology-transition cap are both saturated
- Complex terrain at similar distances to confirm that slope and hydrology retain detail farther out
- Refine and coarsen hysteresis crossings in both travel directions
- One resident topology replacement through its terrain fog pulse and two-phase canopy exchange
- The visible horizon near radius 512
- A broad turn that reveals tiles behind the starting camera
- A nearer ridge in front of lower ground and a taller distant peak

Across the exact-to-far overlap, compare matching terrain profiles and water boundaries column by column rather than treating the reported nearest-gap distance as a radial ownership boundary. Inspect for height steps, cracks, duplicate surfaces, missing strips, water walls, material pops, and material-specific rings. Capture while exact queues are active and record any black ring or vertical panel as a failure. Every far-owned overlap fragment remains protected, even in a fully ready tile with only partial boundary requirements. Its tile must not draw until the exploration band has step 2 or the rest of the exact overlap has step 8 or finer. Confirm that a protected base-only tile stops the drawable frontier so no farther island shows through. No partially masked horizon patch may hide farther terrain. LOD skirts should appear only on a finer edge beside a resident coarser tile. Separately inspect missing exact halos for their explicit lit planned continuation, dark inward cap, or vertical bedrock cap, then confirm halo arrival removes that closure. Generated source-water tops must meet at the same full-block plane, while every standing water body remains source-filled from the first wet voxel above solid support through its exact surface. Explicit rapid and outlet levels must remain flowing, and waterfall curtains must retain falling state. In the forest capture, verify that step 2 reuses accepted exact anchors and that steps 4 through 32 use globally anchored 64-block aggregate cells with six fixed candidates and block-8 habitat and ground authority. The aggregate tiers must form strict stable subsets. An exact-anchor tree must retain its species and dimensions rather than flickering into an oversized generic form. At step 32, verify that the exact collector's habitat and root-water decision survives unrelated water elsewhere in the 32 by 32 cell and that the trunk grounds on the displayed voxel. During an ordinary tier replacement, verify that one complete terrain topology and its skirts swap beneath the narrow fog pulse, the target canopy is established before the source retires, and water remains source-owned until completion. Measure terrain and water parent work separately from the synchronous canopy stage. Confirm there is no forest-density cliff at any tier, no doubled far canopy over exact trees, no empty-forest blink, and no flicker where a crown crosses a tile face. Across the rest of the horizon, inspect for a visibly granular taper, continuous borders, exposed or orphan skirts, cracks, inward-wound missing faces, body-aware contour-clipped stable water coastlines, ring-shaped material jumps, topology popping, boundary chatter, and false occlusion of the taller peak. Compare F3 exact required and ready counts, the conservative nearest-gap distance, parent and refinement wanted, resident, drawn, missing, and queued counts, drawable frontier, culling, cache, and arena values before and after the turn.

Use oblique views of distant textured slopes and alpha-cutout flora to inspect the complete block-texture mip chain. Look for reduced shimmer and moire patterns, preserved foliage coverage, no sudden mip bands, and the expected crisp nearest-filtered appearance when magnified nearby.

The implementation uses adaptive immutable tile tiers, not a literal geometry clipmap. Its conservative 256-bin terrain-horizon test is not a hierarchical Z buffer. Draws are bounded direct indexed commands, not an indirect command buffer. Report only the mechanisms and evidence actually observed.

### Cubic vertical exploration

Capture one high section above Y=256 and one low section below Y=0. Capture a cave or overhang near a horizontal cube edge and another view looking across a top or bottom cube face. Underground, move laterally across the edge of the six-chunk exploration radius and vertically across the edge of its four-cube band, then wait for the queues to settle and capture both directions. For the complete world-generation review, also inspect a volcanic conduit or lava tube and the fixed sealed aquifer sample.

Expose a broad underground wall across at least one cube face, one chunk column boundary, one 64-block field line, and one plate contact. Capture it obliquely so bedding direction and thickness are visible. Ordinary beds must retain curved dip, variable thickness, and continuous deformation. Only a tagged fault may produce a sharp offset, and no ordinary stratum may reset into a vertical grid-aligned panel.

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

Capture a river or lake from above, a partial flowing edge when available, a waterfall from the side, and a deep-water view with the camera below the actual fluid surface. For seed 42, begin with the reported open-water regression at X=-557, Z=379. Confirm the direct sample, all footprints, and the column plan agree on topology and level, then inspect the floor and every implicit source voxel from that support through the top. Capture the supported lake lip at X=-8235, Z=2976, the incised river face near X=-12288, Z=2653, and the canyon ecotope at X=-23904, Z=0 as separate views. Also capture the elevated lake near X=-8272, Z=3056 and its receiver-centered outlet fall into the lower river at X=-8256, Z=3072 from exact range and from each far LOD. For seed 764891, circle the caldera at X=23029, Z=-111486 and capture its bank from water level.

Trace one lake shoreline across a catchment or 256-block contour-page boundary and one river through a shared portal. Inspect both the four-block broad contour controls and the two-block refined narrow band on either side of the page edge. Also find a competitive boundary between distinct lake authorities and an owned outlet or channel corridor crossing a supported bank. Record `WaterBodyId`, signed shoreline distance, surface level, channel gradient, and outlet state from the inspector on both sides. One lake identity and flat level must agree across its own boundary, distinct bodies must remain separated by a supported watershed, and an owned connector must stay open. Signed distance must cross zero continuously, and junction-to-portal channel water must not rise downstream unless a tagged explicit fall accounts for the drop.

Inspect for:

- Eighth-block level changes and smooth corners
- No vertical water walls at stable river, lake, ocean, delta, cube, or unavailable-neighbor edges
- No floating lake top over dry land, and a solid block below the lowest water voxel at shallow shore occupancy
- Standing generated water filling every wet voxel from the lowest one above solid support through the surface as implicit source state, including across a cube face
- Canonical ocean, river, lake, delta, waterfall, and supported-bank authority agreeing among direct samples, filtered footprints, column plans, and emitted blocks
- Distinct lake levels separated by an irregular supported competitive watershed, with no coarse far cell joining them and no divider closing an owned outlet or channel corridor
- Smooth monotonic water profiles from confluence junctions to shared portals, with abrupt changes represented only by explicit falls
- A complete irregular dry caldera rim with at least one block of freeboard and no unsupported gap; named outlets remain valid for routed nonendorheic lakes
- Stable source and flowing cells showing planar top geometry only, with no vertex displacement or geometric waves
- Filtered analytic fragment normals and caustics animating water shading without moving the source plane
- Explicit falling columns retaining vertical sides and consistent shading
- A narrow outlet fall centered on its lower receiver, reaching from the lower visible water plane to the upper lip without a long horizontal slab, a gap, or a raised receiving body
- One complete five-quad far outlet prism at steps 2, 4, 8, 16, and 32, with half-open anchor ownership and no duplicate wall on a neighboring tile
- Far partially wet shoreline cells following contour-clipped edges rather than rectangular sheets
- Far generated source-water tops matching the exact full-block plane at the handoff
- Back-to-front ordering that includes vertical distance
- Refraction, depth absorption, caustics, fog, and underwater overlay
- Fish remaining inside water and rendering below the surface

Generated water should show a zero pending-fluid count until disturbed. Runtime edits cannot be proven by a static startup capture alone. For disturbed-water acceptance, perform a manual edit near a loaded boundary, observe downward-first spread and level decay, leave and return to resume the frontier, restart to verify persistence, and capture before and after. Pair this manual result with the automated fluid-rule tests.

### Flora and fauna

Capture dense rainforest, temperate rainforest, broadleaf forest, conifer forest, taiga, mangrove habitat, sparse dry woodland, land wildlife, wetland frogs, alpine goats, and underwater fish at deterministic habitat coordinates when those populations appear.

Inspect exact oak, large oak, birch, spruce, acacia, jungle, mangrove, palm, willow, alpine scrub, and fallen-log forms. Their trunks and canopies must cross cube faces cleanly, and their species must fit the continuous biome suitability, temperature, precipitation, soil moisture, fertility, light, slope, altitude, lithology, tectonic stress, hydrology, and ecotope context. Suitable forests should produce dense cover with tighter deterministic spacing, while dry, steep, barren, geothermal, and actively volcanic ground should suppress it. Ordinary species must never root in standing generated water. Only mangroves in suitable water no more than three blocks deep and non-ocean willows in suitable water no more than two blocks deep may root while submerged. For either exception, inspect the complete column and require adapted logs to replace intervening source water from the solid floor through the trunk, with no floating base. Also inspect grounded plants, lily-pad orientation, distinct voxel models, fish confinement, and the absence of animals on invalid cliffs or dry fish spawns. At long range, confirm the strict aggregate canopy hierarchy persists at every tier with stable species silhouettes and disappears beneath latched exact ownership rather than overlapping exact trees inside radius 32.

### Diagnostics and F3

At one fixed coordinate, capture an F3 frame and compare every displayed value with the inspector or a direct sample at that block, allowing only documented display rounding. The F3 Cache entry count and MiB value combine exact and far column-plan, basin, shoreline-contour, and macro-control caches; reconcile them with the inspector's separate cache fields rather than comparing only one cache. Record exact required, ready, unresolved, loaded, and meshed counts, the conservative nearest-gap distance, far base and refinement wanted, resident, drawn, missing, and queued counts, the drawable coverage frontier, frustum and horizon culls, cache MiB, arena MiB, fluid work, pending-update drops, deferred-frontier drops, and mesh coalescing. Capture `RYCRAFT_WORLDGEN_OVERLAY=geology`, `hydrology`, `climate`, and `biome` separately. Verify each overlay is visibly distinct, agrees with the sampled field, has no straight former-control boundary, and remains stable across cube and catchment boundaries. Any other nonempty value is invalid.

### M4 Max performance acceptance

Run performance acceptance separately from Metal validation with an optimized build. Use an identified Apple M4 Max, native display resolution, 4x MSAA, and the user-reported seed-764891 starting view:

```bash
RYCRAFT_WORLD_SEED=764891 RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 RYCRAFT_VIEW_DISTANCE=512 \
<repo>/build-release/src/rycraft
```

After the static view settles, repeat it with `RYCRAFT_AUTOPILOT=fly` and record the warmup, movement start and stop frames, performance window, and resulting route. Record the M4 Max configuration, macOS version, resolution, commit, and settings. Confirm exact simulation remains capped at radius 32 throughout.

Allow streaming to settle, then measure the moving route long enough to expose repeated streaming work. Record frame p50 and p95, the lowest sustained one-second frame rate, queue maxima and settle time, loaded and mesh-resident maxima, active exact-generation, exact-mesh, and far-worker maxima, combined construction concurrency, peak process RSS, and peak Metal allocated or resident memory. Exact generation uses four latency-sensitive and two utility workers and may submit only seven cube tasks, six running plus one look-ahead, beneath the 64-job hard ceiling. Confirm stale retained-set tasks are skipped and still-required work returns through current plan dependencies. The six, four, and eight worker pools can expose 18 construction threads on the 16-core reference machine, so record CPU saturation while all three are active. Acceptance requires:

- Lowest sustained one-second frame rate of at least 60 FPS
- No sustained generation-related frame time above 20 ms
- At most 64 GB total unified-memory use
- No more than 32,768 loaded exact cubes or 16,384 exact mesh-resident cubes
- Far CPU cache at or below 9,280 tiles and 3 GiB
- Every cold step-32 horizon parent resident within two seconds
- No far-owned exact-overlap fragment displaying step 32; exploration-band fragments wait for step 2 and every other protected fragment waits for step 8 or finer, including fragments in fully ready partial boundary tiles, with protected base-only tiles stopping the drawable frontier
- Far GPU arena at or below 2 GiB of vertices and 1 GiB of indices, allocated lazily in paired 256 MiB vertex and 128 MiB index slabs
- Eight far workers remain available while exact streaming is busy; when parents are queued, dispatch reserves four worker slots for base work and admits no more than four urgent connected refinements after already running jobs finish
- While exact streaming is busy, no more than four refinement uploads advance per frame while the 32-parent lane remains available
- When exact streaming is idle, no more than 32 parent uploads and 12 refinement uploads advance per frame
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
