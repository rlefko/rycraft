---
name: playtest
description: Run a deterministic Rycraft generator v4 playtest with verified model setup, generation fingerprints, canonical water, expanded vertical range, cold-horizon and canopy staging checks, no-skirt LOD inspection, smooth packed light, physical atmosphere and weather, five-cascade shadows, Hi-Z indirect light, emissive materials, Metal validation, captures, and M4 Max performance evidence. Reads the architecture, world-generation, rendering, and performance documents first.
---

# Playtest

Use this workflow for visual or hardware validation. Do not claim a behavior was exercised unless the corresponding log, metric, or opened frame proves it.

## 1. Read the contracts

Read these files completely:

- `docs/architecture.md`
- `docs/world-generation.md`
- `docs/rendering-conventions.md`
- `docs/performance-conventions.md`
- `docs/generator-v4-follow-up.md` when qualifying the paged hierarchy, GPU construction, distant
  water, or distant flora

Note all implementation limitations, especially any incomplete first-entry or steady-state performance evidence and unqualified visual LOD evidence. The retained `BasinSolver` hydraulic erosion, alpine postprocessing, and analytical crater-lake overlay are v3-only compatibility machinery; finding one on a v4 path is a failure. V4 volcano relief must enter before canonical routing, and its separately bounded dry residual must not alter wet topology. Plant-functional-type equilibrium is deferred to PR 2, so PR 1 ecology evidence covers only the physical-climate adapter and existing placement consumers.

## 2. Build and run portable tests

```bash
meson setup build --buildtype=debugoptimized
ninja -C build tests/test_rycraft
ninja -C build test
./build/tests/test_rycraft "[learned]"
./build/tests/test_rycraft "[bootstrap]"
./build/tests/test_rycraft "[reported-water-continuity]"
./build/tests/test_rycraft "[render][far-terrain]"
```

Record exact pass and failure counts. A failing continuity case is a blocker, not a visual note.

## 3. Verify model setup

Generator v4 assets must live under `~/Library/Application Support/rycraft`, outside the repository and Conductor workspace. Verify the pinned revision and every size and SHA-256 against `resources/config/terrain_model_manifest.json`.

Confirm the compatibility references remain InfiniteDiffusion paper v4 and Minecraft implementation commit `23d3f50e5108882bb88a03c3ab048aa63633a02f`. These references define review inputs; only the model and ONNX Runtime archives are installed runtime assets.

For an isolated run, set `RYCRAFT_APPLICATION_SUPPORT_ROOT` to an absolute disposable directory outside the repository and Conductor workspace. Never point this override at an existing user world or a broad directory.

Record:

- macOS version and Apple Silicon model
- InfiniteDiffusion paper version and Minecraft reference commit
- ONNX Runtime 1.27.1 archive hash and loaded dylib path
- Model revision and five model-data hashes
- Core ML, CPU fallback, and other provider partitions and nodes
- Canonical qualification digest
- Full generation fingerprint
- Unsigned 64-bit world seed

Stop the production v4 playtest if `CANONICAL_QUALIFICATION_HASH` differs from the pinned recorded digest, if qualification fails, or if no complete production authority backend is wired into `WorldGenerationContext`. A diagnostic v3 frame is not v4 evidence.

After one verified installation, restart normally and confirm that launch verifies and reuses the installed pack without a full download. In a disposable root, remove only the completion marker and confirm that bootstrap restores it after local asset verification. Repair is the only route that may fetch a replacement asset.

## 4. Protect existing worlds

V4 normally writes beneath:

```text
~/Library/Application Support/rycraft/rycraft_world_v4
```

Before a destructive or repeatable test, use a separate user account, a separately configured Application Support root in a test harness, or a recoverable copy made outside the repository. Do not delete or modify `rycraft_world`, `regions-v3`, legacy manifests, edits, fluid frontiers, or player state.

Confirm that ordinary title startup performs no profile or model action. Confirm that a selected profile with a seed or fingerprint conflict returns to world selection without changing the source. Exercise explicit fresh creation and `CREATE V4 SUCCESSOR`, require the separate confirmation screen before successor setup begins, verify each confirmed action reserves and publishes a separate current-identity profile only after qualification, and confirm the source metadata and regions remain unchanged.

## 5. Exercise bootstrap UI

Capture and verify:

- Model required with Download and Quit
- Downloading with byte progress and Cancel
- Verifying
- Compiling Core ML partitions
- Loading and qualification
- Failed with Retry or Repair as appropriate
- Ready followed by creation or opening of `rycraft_world_v4`

Confirm no `SaveManager`, `World`, generation worker, or far worker begins before Ready. Cancel a staged download once, verify no partial pack is installed, and verify Retry continues from the staged byte count. Corrupt a disposable test asset once and verify Repair replaces only that asset while retaining valid files and compiled Core ML caches. Confirm that normal restart and Retry do not redownload a verified installed pack.

## 6. Fix deterministic views

Use recorded seed and camera coordinates. The established performance route starts with:

```bash
RYCRAFT_WORLD_SEED=764891 \
RYCRAFT_SPAWN=23029,225,-111726 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 \
RYCRAFT_VIEW_DISTANCE=512 \
./build-release/src/rycraft
```

For every capture, record:

- Generation fingerprint and authority quality
- Seed, spawn, camera position, yaw, pitch, and LOD step
- Frame number and time since launch
- Authority, hydrology, exact, far base, refinement, upload, and canopy queue counts
- Model-call, provider-partition, tensor-cache, and decoded-cache metrics

## 7. Cold horizon and canopy staging

Begin from a cache-cleared v4 start. Capture the first drawable surface stage and the settled canopy-complete scene.

Require:

- Complete connected coarse terrain and canonical water through the 96-chunk entry radius before gameplay entry
- While the entry UI remains visible, confirm exact mesh publication advances and the protected FINAL lane opens once the connected frontier reaches the near band
- Require the camera-aware protected FINAL closure before entry: 4 targets at step 1, 8 at step 2, 12 at step 4, 16 at step 8, and 20 at step 16
- Confirm the entry UI reports both the 96-chunk entry radius and configured radius, and cannot accept a frontier from a smaller or stale selection
- Keep the configured horizon selected after entry. Confirm ordinary outer filling pauses while
  exact 32-chunk publication or connected desired-LOD debt remains. Both ordinary outer submission
  and publication must pause, resume only after both debts clear, and never expose a farther
  resident island
- Test all four camera positions near the corners of a far tile. Protected diagonal targets beyond the 96-chunk radial prefix must remain hidden until the parent frontier advances, and frontier fog must remain opaque through at least 84 chunks.
- No farther resident island across a missing parent gap
- Exact safe spawn uses final authority. Its accepted four-block-aligned center normally has one atomically installed 5 by 5 native dry certificate, followed by exact support, headroom, slope, water-absence, and nearby-dry validation at the center.
- An all-positive continental owner may instead propose one learned dry site without installing a certificate. Confirm that the UI continues locating dry land and the far horizon remains dormant until the radius-zero exact plan accepts it. Record provisional selection time, exact-validation time, every rejection, and the accepted owner.
- Normal entry ranks at most one proposal per aligned 2,048-block hydrology owner and prepares only the selected owner before world construction. Cold world construction uses a zero nominal exact radius with its mandatory one-chunk active halo. The full 32-chunk exact band streams after entry, and only explicit repair or qualification may prequeue its complete authority closure.
- After entry, fly and walk across the full exact disk while recording required and ready surface
  sections, generation lanes, mesh candidates, uploads, deferred lighting, far worker budget, and
  canopy worker budget. Required surfaces through 32 chunks must stay ahead of optional flora and
  broad work. The camera column must rank first, the six-chunk exploration band second, and the rest
  of the required disk third. After the connected 96-chunk prefix is drawable, exact publication
  through 32 chunks or any connected visible desired-LOD miss must pause ordinary outer submission
  and publication. Near work must run nearest-first, rank horizontal distance before projected
  error within the nearby visible class, and may displace queued or dependency-parked outer parents.
  Local far admission must report 8 workers alongside exact debt, 12 after exact debt clears, and
  16 only after exact and local debt clear. Canopy admission must remain zero during entry
  preparation and until connected terrain is drawable. Gameplay must then retain exactly one
  low-priority canopy worker while protected, local, or exact publication debt continues. It must
  not open a second gameplay lane after stronger debt drains.
- Before control opens, confirm all 27 cubes in the finalized spawn's three-by-three horizontal by three-section collision halo are resident. Attempt immediate walking, sprinting, jumping, and flight in each horizontal direction. Missing cubes must remain closed, but the player must encounter no streaming-created invisible wall. Confirm the collision gate overlaps horizon work and adds no model inference.
- At an exact-to-far handoff, record the visual coverage epoch and collision ownership epoch. Verify
  that matching published exact sections use exact blocks and fluid heights, while unowned planned
  sections use canonical generated terrain and water and unresolved columns remain closed. Physics
  must not change early when a partially loaded exact cube arrives.
- Record per-phase coarse, Base, and decoder call counts on fresh and warm opens. A finalized warm
  profile must report zero dry-spawn, final-spawn, horizon-preview, and protected-handoff graph
  calls before entry, apart from calls caused by newly visible post-entry refinement.
- The isolated fresh seed-42 cold entry audit must report exactly 80 Coarse, 14 Base, and 5 Decoder
  calls for its audited spawn phases.
- Confirm that Coarse remains scalar, Base and Decoder use static batches of four, Decoder binds
  256 by 256 spatial dimensions and repeats the last real window in a short tail, the Core ML cache
  is `coreml-cache-v3-base4-decoder4x256`, and the qualification digest is
  `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df`.
- Restart after a protected FINAL rectangle has published and confirm the `RYTG` disk load avoids
  another backend inference. Corrupt one disposable `RYTG` payload and confirm one atomic repair.
- A fresh spawn is dry, supported, has headroom, and has a dry nearby safety neighborhood
- An older pre-safety-revision ocean spawn relocates through the bounded coarse search before entry
- Surface-stage terrain and water publish while flora callbacks are delayed
- A delayed or canceled flora attachment does not invalidate the resident surface stage
- Nearby and middle-distance PREVIEW surfaces receive grounded FINAL tree and ground-flora attachments before decoder promotion
- Exact terrain may hand off while far flora remains, and far flora retires only after every exact flora-bearing section is revision-ready
- Camera movement reprioritizes flora without leaving a bare transition band
- Cloud quality Off performs no noise generation or allocation, while High and Medium enter gameplay with neutral cloud shadow and accept the matching asynchronous noise upload without a frame stall
- Coarse parents remain until connected replacements are resident
- One-chunk movement recovers within five seconds without losing water connectivity
- Leaving the world while cube and column-plan work is active drains generation on the lifecycle owner and returns to the title or exits without a worker self-join, abort, or hang

Use a test hook with a blocked flora callback when available. Record surface-stage and flora-complete publication times separately. While the preparation screen is visible, confirm authority polling, base scheduling, result draining, and shared-buffer publication continue without the full exact-world scene draw. After a spawn relocation, require fresh matching world and view epochs before treating the horizon as complete. Force one generated tree or roof cutoff to exceed a single lighting transaction, confirm each transaction remains at or below 32 floods, and verify no pending cube becomes visible before the camera-ranked publication queue settles it.

```bash
RYCRAFT_WORLD_SEED=42 <repo>/build/src/rycraft_worldgen_inspect 42 > /tmp/rycraft-playtest/inspect-42-a.json
RYCRAFT_WORLD_SEED=42 <repo>/build/src/rycraft_worldgen_inspect 42 > /tmp/rycraft-playtest/inspect-42-b.json
jq 'del(.benchmark) | .far_terrain.tile_builds |= map(del(.milliseconds))' /tmp/rycraft-playtest/inspect-42-a.json > /tmp/rycraft-playtest/inspect-42-a-stable.json
jq 'del(.benchmark) | .far_terrain.tile_builds |= map(del(.milliseconds))' /tmp/rycraft-playtest/inspect-42-b.json > /tmp/rycraft-playtest/inspect-42-b-stable.json
cmp /tmp/rycraft-playtest/inspect-42-a-stable.json /tmp/rycraft-playtest/inspect-42-b-stable.json
```

The current inspector reports candidates and surface samples for mountain, cliff, canyon, river, confluence reach, lake, endorheic lake, waterfall, delta, estuary, wetland, groundwater interface, volcano, oceanic island, biome transition, dense flora habitat, and deep generated fish water. It also reports column-plan counts plus separate basin, shoreline-contour, and macro-control cache bytes, hits, misses, builds, failures, active cold builds, peak cold builds, and throttled cold-build requests. Record every non-null coordinate relevant to the review. Pass an optional `sample_x sample_z` pair after the seed for another exact surface report. Find caves, aquifers, and lava tubes through the fixed fixtures or additional observation because surface sampling does not locate interiors directly.

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

## 8. Capture controls

The engine supports:

| Variable | Effect |
|---|---|
| `RYCRAFT_WORLD_SEED=N` | Override metadata seed for a repeatable world |
| `RYCRAFT_SPAWN=x,y,z` | Override the starting position |
| `RYCRAFT_WORLDGEN_OVERLAY=geology\|hydrology\|climate\|biome\|lod\|authority` | Show a world-generation overlay; `lod` must distinguish exact ownership from all six far tiers |
| `RYCRAFT_SHOW_DEBUG=1` | Show F3 diagnostics without changing terrain colors |
| `RYCRAFT_CAPTURE=/absolute/path.png` | Write one frame to a PNG |
| `RYCRAFT_CAPTURE_FRAME=N` | Choose the capture frame; 400 through 600 usually allows streaming to settle |
| `RYCRAFT_CAPTURE_CAMERA=x,y,z` | Place only the capture camera after v4 entry, leaving the validated player and saved spawn unchanged |
| `RYCRAFT_NATIVE_WINDOW=1` | Fill the current display so captures and acceptance runs use its native backing resolution |
| `RYCRAFT_START_SCREEN=title\|worlds\|create\|delete\|playing\|paused\|settings\|video\|inventory\|crafting\|furnace\|chest\|death` | Choose the initial screen; the gameplay and container tokens auto-start the world in `RYCRAFT_WORLD_DIR`, and `furnace`/`chest`/`death` seed sample state to capture |
| `RYCRAFT_WORLD_DIR=path` | Exact generator v4 profile the auto-start tokens open; relative values resolve beneath Application Support, and an incompatible identity fails closed instead of redirecting or creating a sibling |
| `RYCRAFT_GAME_MODE=survival\|creative` | Force the game mode for this session without touching the saved metadata |
| `RYCRAFT_SPAWN_ITEMS=N` | Scatter N dropped items on the loaded ground ahead of spawn to capture item entities |
| `RYCRAFT_SPAWN_WATER=1` | Sink a pool ahead of spawn for water and reflection captures (pairs with `RYCRAFT_SPAWN_BOAT`) |
| `RYCRAFT_SPAWN_BOAT=1` | Drop a boat a few blocks ahead of spawn to capture it floating or beached |
| `RYCRAFT_SPAWN_MATERIALS=1` | With `RYCRAFT_CAPTURE`, place a capture-only bed, chest, floor torch, inactive furnace, and burning furnace lineup near validated spawn |
| `RYCRAFT_BLOOM=0..1` | Scale or disable bloom |
| `RYCRAFT_VIEW_DISTANCE=4..512` | Override visible distance; exact simulation remains capped at radius 32 |
| `RYCRAFT_TIME=unsignedTicks` and `RYCRAFT_TIME_FREEZE=1` | Pin an absolute unsigned 64-bit world tick for repeatable time of day, lunar phase, shadow, sky, and flare captures |
| `RYCRAFT_WEATHER=clear\|overcast\|rain\|storm\|snow` | Pin deterministic weather, clouds, precipitation, storm state, and wetness |
| `RYCRAFT_CAPTURE_LIGHTNING=x,z,id,ageTicks` | With `RYCRAFT_CAPTURE`, inject one deterministic visual strike after the weather snapshot is ready; age must not exceed the absolute world tick |
| `RYCRAFT_YAW=degrees` and `RYCRAFT_PITCH=degrees` | Point the capture camera after spawn validation |
| `RYCRAFT_AUTOPILOT=walk\|sprint\|fly` | Exercise a repeatable ground route or an obstacle-independent aerial streaming route |
| `RYCRAFT_AUTOPILOT_START_FRAME=N` and `RYCRAFT_AUTOPILOT_STOP_FRAME=N` | Bound movement to a fixed interval so queue settling can be measured afterward |
| `RYCRAFT_AUTOPAUSE_FRAME=N` | Enter the real paused screen at a fixed frame for a same-scene playing-versus-paused timing comparison |
| `RYCRAFT_PERF_WARMUP_FRAMES=N` and `RYCRAFT_PERF_FRAMES=N` | Exclude warmup, record a bounded performance window, print summary lines, and quit |
| `RYCRAFT_SHADOWS=0..2`, `RYCRAFT_CLOUD_QUALITY=0..2`, `RYCRAFT_INDIRECT_LIGHT=0..2` | Override shadow, cloud, and indirect-light quality without saving preferences |
| `RYCRAFT_CLOUDS=0..2`, `RYCRAFT_SSAO=0\|1` | Retained compatibility aliases for cloud quality and disabling or enabling High Hi-Z screen-space indirect lighting |
| `RYCRAFT_VL`, `RYCRAFT_SSR`, `RYCRAFT_WAVING`, `RYCRAFT_LENS_FLARE` | Toggle the remaining individual graphics effects with 0 or 1 |
| `RYCRAFT_VIBRANCE=0..10`, `RYCRAFT_SHARPEN=0..10` | Override final-grade controls |
| `RYCRAFT_GPU_COUNTERS=1` | Enable diagnostic per-pass GPU timestamps |
| `RYCRAFT_SPAWN_LAVA=1`, `RYCRAFT_SPAWN_WATER=1`, `RYCRAFT_SPAWN_MATERIALS=1` | Create disposable validation scenes near spawn |

Use `ci/run-v4-capture.sh` for real-model captures. It accepts a verified source model pack,
APFS-clones that pack into a disposable `/tmp` Application Support root, launches from a separate
scratch directory, records the exact capture identity and streaming state in the adjacent log, and
terminates only the process it started. It never removes the qualification root, so related captures
can reuse the cloned Core ML cache and authority pages without modifying the installed source pack
or any user world. It also redirects settings reads to the disposable root without changing
`HOME`, so the frame does not inherit or write the user's normal preferences. APFS clone stamps
differ from the installed files, so the first isolated launch
performs one local SHA-256 audit and refreshes the marker inside the clone. Later launches reuse that
marker. No launch downloads an asset unless the user explicitly invokes the application's Repair
action. The script refuses a qualification root outside `/tmp`. For example:

While entry is pending, the capture log records bootstrap and preparation progress once per second,
including installed-pack reuse, dry-spawn search state, exact and protected-near readiness,
full-horizon base residency, per-tier residency, worker queues, inference queues, authority builds,
and deferred lighting. Use those lines to locate a cold-start bottleneck even when the frame misses
its deadline.

```bash
RYCRAFT_WORLD_SEED=42 \
RYCRAFT_SPAWN=3200.5,215.05,-5307.5 \
RYCRAFT_YAW=0 RYCRAFT_PITCH=-17 \
ci/run-v4-capture.sh cold-handoff /absolute/path/to/the/verified/model-pack
```

Treat the helper's process exit as part of the evidence. A PNG followed by a signal, termination
exception, or ONNX static-finalizer crash fails the playtest. Reuse the same isolated root for one
follow-up launch and require another clean exit so application teardown is proven idempotent after
persisted authority reuse.

Pass the printed qualification root as the third argument on later calls. Set
`RYCRAFT_CAPTURE_METAL_VALIDATION=1` for a validation capture. The helper rejects a simultaneous
`RYCRAFT_PERF_FRAMES` run because validation and performance evidence are separate contracts.
The helper returns a failing status when the completed capture log contains a generation, runtime,
or Metal validation error. A PNG beside that failure is diagnostic evidence, not an accepted frame.

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

### Integrated lighting, atmosphere, and weather matrix

Use seed 764891, spawn `23029,225,-111726`, yaw 0, pitch -17, view distance 512, and explicit `RYCRAFT_TIME_FREEZE=1`. Capture at 3456 by 2234 when the native display route is available. Keep Metal validation and shader validation separate from the unvalidated performance run.

Capture settled stills and motion sequences for:

- Crooked forests, isolated logs, and broad overhangs at several solar angles. The underside may be in direct shadow, but propagated ambient light must not become a black vertical column merely because a solid block exists somewhere above it.
- A cave entrance from outside and inside, a sealed cave, and a lava-lit interior. The entrance should show propagated skylight and near-field screen-space bounce, the sealed cave should remain dark without an emitter, and lava block or emissive light must remain independent of ambient accessibility.
- Forward motion through endpoints 48, 160, 512, and 1,536 on High, then into the 8,192-block horizon map. Repeat Medium at 40, 128, 384, and 768. Inspect the final 12.5 percent of each range for continuous shadow blend, no camera-centered ring, no resolution pop, no double-cast exact or far silhouette, and no stale projection.
- Ordinary camera motion followed by teleport, resize, FOV change, indirect quality change, cloud quality change, forced time, forced weather, and world reload. Verify valid history settles, each discontinuity rejects old indirect, cloud, and froxel data, and no ghost survives disocclusion. Disoccluded regions must fill with a smooth bounce estimate within about eight frames rather than staying black or speckled, and no bright splotch may survive after its source leaves the frame during rotation.
- Place and break blocks indoors and under an overhang. The placed block must be lit correctly on the very next frame with no dark window, no lingering dark halo on the wall behind it, and no reconcile-queue delay visible as a black face.
- `clear`, `overcast`, `rain`, `storm`, and `snow` at representative dawn, noon, dusk, and night times. Compare pressure, humidity, temperature, wind, cloud coverage and type, precipitation, storm potential, fog, aerosol, and cloud bounds in F3. Rain or snow must follow temperature. Foliage and particles must move with the same wind vector.
- Views below, inside, and above stratus, cumulus, cumulonimbus, and cirrus layers. Fly through a layer and past a mountain intersection. Inspect density variation in all three dimensions, bounded physical wind speed, temporal stability, bilateral terrain edges, cloud shadow agreement, and no repeating slab or fast 80-block-per-second drift.
- Lightning before and behind clouds, the diffuse atmospheric flash, delayed thunder, repeated deterministic event IDs, and a reload after an older storm bucket. Confirm no backlog replay, block edit, fluid update, fire, or terrain change.
- Dawn, noon, dusk, and night physical atmosphere with clear and aerosol-heavy weather. Inspect horizon response, sun angular size, altitude response, cloud attenuation, shafts, fog silhouettes, finite color, and no old artistic daytime gradient.
- A clear, frozen daytime sky-and-ground coherence capture. Record the absolute time, cloud quality, weather preset, and relevant F3 weather and atmosphere diagnostics. Reject a dark or night-like sky above terrain receiving daylight illumination, or a daylight sky paired with stale nighttime direct illumination.
- Sunset and sunrise through civil twilight, then representative new, quarter, half, three-quarter, and full lunar phases. Only one directional light may contribute. The sun must disappear from direct light, flare, water glint, and reflection below the horizon. The moon must stay subdued through twilight, then scale its disc, direct light, and shadows by phase without blooming into a flat white circle.
- Water above and below the surface under clear daylight, moonlight, overcast sky, cloud shadow, fog, and shafts. Air fog must stop at the water surface, and underwater absorption must remain the sole submerged medium.

For every motion sequence, record cascade refreshes, indirect-history validity and reset reason, cloud and froxel history validity, atmosphere LUT refreshes, weather request and worker state, cloud type, storm or lightning ID, and the GPU timing for each new pass. Open every PNG and inspect the corresponding log. A stable event ID or zero validation count is not visual evidence by itself.

## 9. World-generation capture matrix

Use inspector coordinates rather than wandering at random. Choose spawn Y based on the sampled terrain or water elevation.

### Aerial terrain

Spawn 40 to 100 blocks above the sampled surface with flight available. Capture a mountain or cliff, canyon or gorge, river confluence, lake, waterfall, delta, brackish estuary, connected wetland and groundwater interface, volcanic island, and biome transition. Use a wider view distance only when the feature needs it, and record that override.

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

The continuous-field and categorical former-line regressions are ordinary CI tests. Run the
`[artifact][continuity]` matrix and require zero failures. Record the measured derivative and
structured-orientation ratios instead of relying on a hidden or expected-failure tag. Do not
attribute a field-continuity result to the separate cold exact-to-far residency checks.

### Far horizon and LOD transitions

Use `RYCRAFT_VIEW_DISTANCE=512` for dedicated far-horizon captures. Exact editable terrain has a
nominal radius of 32 chunks. Settled far geometry is step 2 through 64 chunks, step 4 through 128,
step 8 through 256, and step 16 through 512. Treat those bands as maximum-coarseness limits. Verify
that high-relief or strongly projected terrain retains a finer tier, including FINAL step 1 when
required, while its conservative screen-space error exceeds 0.55 pixels and coarsens outward only
after the next tier falls below 0.45 pixels. Step 1 is the irreducible voxel-grid floor. Repeat
representative captures after changing drawable size and FOV.

Step 32 is a coverage parent only. Every selected 256 by 256-block coordinate requests an immutable
step-32 parent before refinement. Cold entry publishes safe FINAL spawn terrain, revision-ready
exact meshes, the connected step-32 terrain-and-water prefix through 96 chunks, and the atomic
60-target protected FINAL closure. The camera-aware closure contains 4, 8, 12, 16, and 20 targets
at steps 1, 2, 4, 8, and 16 and begins during preparation when the connected frontier reaches the
near band.

After the connected prefix is drawable, exact publication through 32 chunks and every connected
visible desired-LOD miss pause ordinary outer-parent submission and publication. Confirm near work
is nearest-first, ranks horizontal distance before projected error within the nearby visible class,
and may displace queued or dependency-parked outer parents. Displayed parents, the connected prefix,
transition endpoints, exact fallbacks, active protected lineage, and requested critical keys remain
pinned. Local far work admits 8 of its 16 workers alongside exact debt, 12 after exact debt clears,
and all 16 only after exact and local debt clear. The canopy budget is zero during preparation and
exactly one low-priority worker during gameplay after the prefix is drawable. No second gameplay
canopy lane opens.

For protected FINAL authority, record the lexicographically grouped owner rectangles and compare
every 517 by 517 crop with an independently prepared owner, including negative coordinates and
shared aprons. Groups may span at most two by two adjacent owners and no combined request may exceed
1,048,576 samples. Compare samples or hashes exactly. Record fake-executor call reduction only as
structural evidence, not real-model performance evidence.

Record `NativeHydrologyCacheMetrics::deferredBuilds` as an interval delta of completed typed
deferrals. Record `activeBuilds` as a direct gauge. Do not interpret parked work, active work,
ordinary failures, or queue depth as deferred builds.

A missing protected PREVIEW parent uses urgent coverage admission and may displace lower-ranked
queued or dependency-parked ordinary coverage. Current protected FINAL children and parents receive
capacity before required PREVIEW bridges. Unused bridge capacity returns to current FINAL before
one directional prediction may use spare CPU-only capacity. Ready visible FINAL children, proxy
bridges, and visible FINAL parents each receive four ordinary urgent slots, with unused child slots
loaned to proxies. Coarse PREVIEW work leaves sixteen authority admissions for FINAL requests.
Every protected and ordinary target progresses through adjacent tiers. Confirm that a cached fine
target cannot starve the bridge needed to replace a displayed parent and that proxy work cannot
starve same-key FINAL detail.

A requested protected FINAL role-selected key may reclaim optional distant non-displayed CPU or GPU
residency and use the complete GPU arena, while structural coverage, displayed surfaces, transition
endpoints, exact fallbacks, active protected lineage, and requested critical keys remain pinned.
Moving inward adopts the finer required result immediately. Preview authority may supply nearby
geometry while final pages are cold. A same-key final promotion retains the preview terrain and
shadow source while a visible preview child depends on it, and exact ownership stays gated until
stable FINAL authority. Confirm matched terrain and water switch together. Treat topology-changing
per-tile water promotion as a failure until a complete hydrology owner is published atomically.
Parent and drawable frontiers prevent farther resident islands across a missing dependency.
Published exact requirements and unresolved columns define one 256-bit ownership mask per far tile,
with neighboring masks handling crossing canopies and falls. A partially masked patch cannot
occlude. Base terrain, standing water, and falls publish independently of the single-worker canopy
service. Production meshes emit no downward skirts, and shared canonical transition strips keep
displayed neighbors within a 2:1 ratio.

The four admitted slots for queued missing parents are a nominal reservation. Confirm that an urgent
camera-critical refinement starts before an unrelated distant parent when the reservation is the
only remaining capacity. Ordinary urgent work must continue to observe the reservation, and the
fixed worker and urgent-job caps must remain unchanged.

For warm captures, confirm current-frustum proxy refinement ranks ahead of work behind the camera. The capture log must report the worst visible projected error, its desired and displayed step and quality, both resident-quality masks, the violation count, and outstanding visible FINAL requests. Base-lineage PREVIEW steps 16, 8, 4, and 2 are temporary geometry, not acceptance evidence for canonical detail.

Use the validated seed-42 view near X=3200.5, Y=215.05, Z=-5307.5 to inspect cold exact-to-far ownership and exact missing-halo closure together. Missing exact lateral faces must show a lit planned continuation aboveground or a dark inward cap underground, and missing vertical faces must show bedrock caps until the real halo invalidates them. Record any straight frontier, crack, square hole, or cap that survives halo arrival as a failure. Block the flora callback and require the same terrain and water base to publish. Then release it and verify tree and ground-flora geometry arrives without replacing base geometry. Fly across the 16-chunk exact-flora priority boundary and confirm no bare band, duplicate vegetation, floating plants, or abrupt projected-cover collapse appears at steps 2, 4, 8, 16, or 32.

### Paged hierarchy and distant flora

When the deferred hierarchy path is enabled, record its schema revision and the exact Distant
Horizons, Voxy, and terrain-diffusion-mc reference revisions from
`docs/generator-v4-follow-up.md`. Exercise:

- A surface-only plain, high-relief mountain silhouette, overhang, cave mouth, deep cut, floating
  feature, and edited cubic exception. Confirm the surface quadtree remains the common path and only
  cubic exceptions allocate sparse volumetric bricks.
- Arbitrary parent and child completion order, a stationary camera at a threshold, repeated camera
  jitter, a 180-degree reversal, and high-speed flight. Require no uncovered frame, no visible
  oscillation, and no coarser replacement when resident fine data still violates the screen-error
  threshold. Screen-space error must choose desired quality, while the nearby visible scheduler
  ranks horizontal distance before projected error.
- Deletion of the derived hierarchy cache followed by a rebuild. Authority, exact meshes, water
  identities, flora anchors, and final visible hashes must remain unchanged.
- Narrow rivers, shorelines, falls, and broad standing water at every hierarchy level. Generic
  material aggregation may not erase, merge, restage, or disconnect them.
- Forest, riparian, wetland, savanna, and alpine scenes at exact-instance, crown-cluster, impostor,
  subpixel-cluster, and horizon-canopy tiers. Require nonzero distant flora wherever the exact
  reference has capacity. A source flora attachment must remain visible until its target attachment
  uploads, and no flora tier may change collision.
- CPU and Metal traversal of the same camera snapshots. Record selected-node hashes, projected
  error, request counts, overflow state, indirect command counts, page faults, hierarchy bytes, GPU
  heap bytes, and validation messages.

## 10. Expanded vertical range

Capture:

- Terrain or a test structure near Y=1407
- A high section above the former Y=511 ceiling
- A low section near Y=-128
- A sky path that crosses the boundary between the two vertical mask words
- A cave, water column, entity, save, and mesh near the top range

Inspect collision, skylight, fog, shadow coverage, water sorting, culling, and save reload. No loop may wrap, alias section 64, or stop at the former ceiling.

## 11. Canonical water matrix

Capture exact, handoff, step 1, step 2, step 4, step 8, step 16, and step 32 for:

- Ocean coast
- Flat lake interior and natural outlet
- Narrow river crossing a coarse cell with dry corners
- River junction and ordinary monotone descent
- Explicit rapid or waterfall
- Delta or distributary mouth with both branch ribbons visible
- Brackish estuary, connected wetland, and groundwater interface across a page seam

For each view, record body ID, stage, bed elevation, flow direction, discharge, signed shoreline distance, transition owner, and exact versus far mesh hash.

Reject:

- A deleted wet route
- Dry terrain raised only to hold water
- A straight retaining wall or long artificial ledge
- Full-height source shelves replacing an ordinary river's eighth-block descent
- Cardinal sawtooth or phase-zero staircase
- Step-32 route loss
- Different water stages joined by one surface
- An abrupt unowned stage jump
- Unsupported standing water
- Generated river tops that enqueue fluid work before a gameplay edit
- A far plane that differs from the exact source plane

The six supplied screenshots are anti-pattern references. Replacement captures must show the same classes of scene without empty water gaps, straight lines, sawteeth, or absurd banks.

## 12. No-skirt LOD matrix

Capture oblique joins for every adjacent pair:

- Step 2 to step 4
- Step 4 to step 8
- Step 8 to step 16
- Step 16 to step 32
- Exact to step 1

Confirm production mesh diagnostics report zero skirt quads and displayed-neighbor diagnostics never exceed a 2:1 step ratio. Inspect beneath and along each join for downward panels, open cracks, duplicate faces, and ledges.

Confirm both sides expose identical canonical boundary heights, positive-area terrain is owned by one half-open tile, and replacements advance through adjacent tiers. Transition-marked vertical geometry must be strictly internal and backed by a real source terrain discontinuity. Treat any crack, duplicate face, boundary panel, or height mismatch as a failure. Portable topology tests establish structural behavior but do not substitute for opened captures.

## 13. Gameplay rendering matrix

In a disposable qualified v4 profile, capture and exercise:

- A torch placed in a sealed dark room, including the placing frame, steady block-light gradient, flame-only emissive radiance, bloom, and indirect bounce while the wooden stick remains ordinarily lit
- Adjacent inactive and active furnaces, including the exact tick that cooking changes the block state, mouth-only emissive radiance, the nonemissive shell and top, and the return to inactive after fuel expires
- Lava with full-surface emission beside an ordinarily lit nonemissive block
- A bed from every exposed face, sleep interaction, safe-spawn update, death, respawn, save, and reload
- A chest and furnace at negative Y and near the upper world limit, including UI, persistence, targeting, collision, and relight
- A bucket edit at the boundary between canonical generated water and runtime fluid state
- A boat crossing generated water at an exact-to-far handoff without affecting far water ownership

Repeat the emissive captures with screen-space indirect lighting enabled and disabled. With it enabled, torch-flame and active-furnace-mouth radiance must enter resolved HDR, where the Hi-Z trace can sample it. The surface MRT must contain diffuse albedo and reduced ambient accessibility, not emitted radiance. Trace, temporal history, denoising, and the composite must not leak through closed walls. With indirect lighting disabled, direct block light and emissive HDR bloom must remain. An inactive furnace and a bed must not emit. Inspect the first changed frame and the settled frames so temporal accumulation cannot hide a one-frame state bug.

Exercise the temporal-history matrix explicitly:

- Require a reset after drawable resize, movement exceeding eight blocks between frames, world-instance change, FOV change above 0.5 degrees, quality change, session end, invalid prior depth, sun-to-moon source change, resident lighting edit, time rewind, or a jump exceeding eight ticks.
- Preserve history during ordinary fixed-tick progression, far LOD refinement, preview-to-final replacement, canopy attachment, and exact-to-far handoff.
- Capture rain and clouds crossing a bright emitter and an opaque silhouette. They must depth-test correctly in the resolved post-indirect pass without leaving indirect-history, accessibility, or bounce artifacts.

Record the block type, packed emissive flag, source block-light level, sampled neighboring light levels, screen-space-lighting quality, GPU pass time, and Metal validation count. Verify reload reconstructs derived light from blocks rather than persisting stale light values.

## 14. Metal validation

Run separately from performance:

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
RYCRAFT_WORLD_SEED=764891 \
RYCRAFT_VIEW_DISTANCE=512 \
./build/src/rycraft
```

Search logs:

```bash
rg -n '\[MTLDebug\]|validation|\[ERROR\]|\[FATAL\]' /tmp/rycraft-playtest/*.log
```

Any Metal API or shader validation message is a failure. Open every PNG. A black, frozen, empty-sky, or unsettled frame is not evidence.

## 15. Real-model determinism

Using identical identity and final quality, generate the same authority pages and camera route in these orders:

1. Fresh forward order
2. Reverse page order
3. Concurrent duplicate requests
4. Cache-cleared rebuild
5. Restart from persisted pages

Require identical quantized RYTA payload hashes and final mesh hashes. Record preview separately from final. Never combine qualities or fingerprints in one comparison.

## 16. M4 Max performance acceptance

Use a release build, native drawable, 4x MSAA, and view distance 512. Record a cold start, settled static view, movement route, one-chunk recovery, and paused comparison.

Acceptance requires:

- Safe final spawn, revision-ready exact meshes, connected coarse terrain and canonical water through 96 chunks, and the 60-target protected FINAL closure within 30 seconds
- Configured 512-chunk parent coverage and all remaining required queues settled within five minutes
- Lowest sustained one-second rate at least 60 FPS
- Frame-time p95 at most 16.67 ms
- No sustained generation interval above 20 ms
- Learned scheduling and lookup p95 below 0.25 ms and maximum below 1 ms on render and fixed-tick threads
- One-chunk recovery within five seconds
- Tensor windows at or below 384 MiB
- Decoded authority at or below 512 MiB
- Model, runtime, and scratch growth at or below 4 GiB
- Total unified memory at or below 64 GiB

Record process RSS and Metal allocated or resident memory separately. State the highest credible unified-memory total without summing overlapping counters.

Do not lower view distance, MSAA, or frame-rate gates to obtain a pass. Do not run performance with Metal validation enabled.

## 17. Report

Output:

1. **Verdict:** works as intended, broken, or works with concerns
2. **Build and tests:** commands and exact results
3. **Identity:** seed, model revision, runtime hash, provider configuration, qualification digest, and generation fingerprint
4. **Bootstrap:** states exercised, cancellation or repair result, and v3 isolation
5. **Validation:** exact Metal validation count and engine errors
6. **Evidence:** each opened capture path and what it demonstrates
7. **Water, LOD, and flora:** exact and far hashes, topology results, 2:1 neighbor and zero-skirt
   diagnostics, near-first queue evidence, nonzero flora coverage at every displayed tier, and
   opened transition captures
8. **Performance:** first entry, settlement, frame metrics, lookup metrics, queues, cache bytes, RSS, Metal memory, and unified-memory conclusion
9. **Unverified behavior:** every model, water-body type, transition, ecosystem feature, or gate not directly exercised

Do not claim a feature was inspected if its frame was not opened or its metric was not recorded.
