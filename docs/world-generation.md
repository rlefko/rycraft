# World Generation and Persistence

This document is the source of truth for generator v4 coordinates, learned terrain identity, generated water, and persistence. It also records which parts require real-model or visual qualification before merge.

## Platform and model setup

Generator v4 requires an Apple Silicon process on macOS 14 or newer. The in-app bootstrap installs assets below:

```text
~/Library/Application Support/rycraft/terrain-models/
  terrain-diffusion-30m-onnx/
    ad2df557eca5645f588766101cf3bc3682455c3e/
```

The installer downloads to a persistent revision-specific staging directory, reports byte progress, verifies exact size and SHA-256, writes a completion marker with the verified files' identity and change stamps, and atomically renames a fresh verified directory into place. A normal restart checks that local marker without rereading the 2.3 GB pack, then reuses the installed files. A changed, missing, or legacy marker triggers a full local SHA-256 audit and marker refresh, never another download. A short staged file continues with an HTTP range request after cancellation, failure, or restart. Repair verifies assets independently, replaces only missing or invalid files, and leaves the extracted runtime and Core ML caches in place. Models, the extracted runtime, and Core ML caches must never be committed or copied into a Conductor workspace.

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `base_model.onnx` | 2,029,994,361 | `543de788f73d0a4012685c908259f615601102aace4751aeccec64154ba145c0` |
| `coarse_model.onnx` | 22,497,125 | `d6ca15b21b2e35d5e594a9ac7a4249a2376590c0ad2b5b49a1e6e2d033450008` |
| `decoder_model.onnx` | 223,854,143 | `6473ae47ca6ec4d743d30fe4f5d381fe4158899714eff09b762005bdbdef68c1` |
| `pipeline_data.json` | 12,226 | `e3132c3ef0c65d8613615f9278ffe23bbd9363ddcd87f1cc6f18456bcc9efe5c` |
| `world_pipeline_config.json` | 774 | `c60f0b74d89317e64cfc623fbfdd828f1b5b2e50aa75020ac4001103381853bd` |
| `onnxruntime-osx-arm64-1.27.1.tgz` | 31,959,937 | `e42b77a7281cc6e55141bf44fcfbac2c782b823a491bbb6ac33c781dd991f8a6` |

All 65 `BlockType` values are append-only because saves persist raw bytes: ids are never reordered or reused, and the serializer validates every stored byte is below `BlockType::COUNT`, so growing the enum keeps every existing RYCH v4 cube valid with no chunk-version bump. A single compile-time `BlockDefinition` table exhaustively defines every block's render shape, collision solidity, opacity, targeting, liquid behavior, leaf behavior, sound, material, light emission, emissive state, wind-sway class, and the survival mining data (hardness, preferred tool class, minimum tool tier, interactability). `BlockType::COUNT` and compile-time assertions prevent a new block from omitting its traits. The crafting table, furnace, lit furnace, torch, chest, wool, and bed were appended this way. The byte-per-block format carries no facing metadata, so furnace and chest fronts use a documented fixed world -Z orientation until facing becomes persistent. A bed remains one block rather than a two-part Minecraft bed, but its rendered and collision volume stops at 9/16 height and it does not block skylight. A floor torch has a distinct support classification from flora: placement requires a full solid block below, breaking that support drops the torch, and water cannot replace it as vegetation. Targeting uses these authored occupied bounds as well: rays above a bed or beside a torch continue through the empty part of the voxel, while a hit carries matching bounds into the selection outline.

The model revision is `ad2df557eca5645f588766101cf3bc3682455c3e`. The runtime is ONNX Runtime 1.27.1 loaded through its pinned C API header and `dlopen`. Its verified dylib remains mapped for the process lifetime because ONNX static operator registries can outlive a runtime environment. Sessions, the environment, and its global worker pool are still released on every retry and application teardown, while one process-local owner prevents duplicate runtime images. AppKit termination explicitly joins render and world generation before releasing generation contexts and the runtime; it never depends on shared-singleton destruction at `exit`. Ordinary CI does not link the downloaded dylib and never downloads the 2,308,318,566-byte pack.

## Bootstrap and failure states

The title screen exposes these states before any world object exists:

1. Model required
2. Downloading
3. Verifying
4. Compiling Core ML partitions
5. Loading and qualifying
6. Ready
7. Failed

Cancellation and retry are explicit. Ordinary application startup performs no model or world work. Selecting or creating a v4 world starts local reuse and loading of an installed pack without another download action. The title screen states `LOCAL PACK REUSED - NO DOWNLOAD` whenever it has selected that path. Retry never replaces installed files. Repair is the only action that may replace an installed asset, and it downloads only assets that fail size and SHA-256 verification. The completion marker is recoverable state, not proof that the pack must be downloaded again: if it is missing or changed, bootstrap audits the installed files locally and restores the marker. A missing model, corrupt asset, unsupported platform, provider failure, or qualification mismatch is user-visible and fail-closed. It may not create a v4 world, publish an empty cube, or use the v3 generator.

The qualification query runs canonical input tensors through the coarse, base, and decoder graphs. Output values are quantized at a scale of 256 and hashed with SHA-256. Compilation retains one static session for each of those three graphs, so qualification and page inference never rebuild a session merely because the next call uses another graph. Coarse retains its qualified scalar batch. Base binds the model's symbolic `batch` dimension to the paper pipeline's fixed batch of four. Decoder binds batch four and the 256 by 256 spatial dimensions; a short lexicographic tail repeats its last real window and discards padded outputs. The runtime records Core ML, CPU fallback, and other provider partitions and nodes, the static Base and Decoder contracts, configured CPU fallback thread count, and resident-session count. The authority inspector also records final-page batch count and batched-page count, which must not imply more than one active model call. The pinned Core ML startup baseline is `6ccf5b56fc32d13df9e7a333a4e68f71c9a0f15191e57375a2e4785c463a41df`, and the provider-bound seed-42 final page containing block `(-4, -4)` has reference hash `d21220e869d92ad4c20201450bcaab05ae735b5657b26a502fd56a8b69c7896a`. These recorded values do not by themselves establish reverse-order determinism, full entry, visual, or performance qualification.

Graph execution remains sequential and the inference coordinator admits one model call. CPU fallback sets ONNX Runtime's intra-op total to `min(hw.physicalcpu, 16)`, which is 16 on the documented M4 Max. That parallelizes work inside one CPU operator without permitting another graph or call to execute concurrently. Dry-spawn ranking retains at most one proposal per aligned 2,048-block hydrology owner. The canonical screen checks the requested chunk, then scans the owner's globally aligned four-block native raster in bounded batches across up to `min(hw.physicalcpu, 16)` workers. A proven center first tries the complete cold footprint and then the exact 113 by 113 radius-zero safety footprint. A bounded same-owner and same-learned-page relocation can install that stronger certificate without additional graph calls. If wider water prevents it, the screen retains the 25-sample five-by-five canonical dry certificate as its strict fallback. A positive-elevation continental fallback installs no certificate and may start only radius-zero exact validation. It cannot admit far-horizon work or metadata publication until that exact plan proves canonical water absence, support, headroom, slope, and the nearby dry neighborhood. Rejected candidates never prepare the wider exact band before world construction.

The opt-in runtime lifetime regression removes one staged model link, lets compilation load the verified dylib and then fail, proves that the failure releases its partial sessions and environment, restores the link, retries successfully, constructs a second runtime, and then exits the isolated test process. Run it against an external verified pack with `RYCRAFT_TERRAIN_REAL_LIFETIME=1 RYCRAFT_TERRAIN_MODEL_PACK=/absolute/model/pack ./build-release/tests/test_rycraft "[.real-runtime-lifetime]"`. A nonzero process exit is a failure even if every in-process assertion completed.

A qualifying cold-start measurement must cover the safe final spawn, revision-ready exact spawn meshes, the connected canonical terrain-and-water parent frontier through 96 chunks, and the atomic 60-target protected FINAL closure for the 30-second gameplay gate. The protected closure begins during preparation once the connected frontier reaches the near band. The configured 512-chunk horizon remains selected and must finish, with every other required queue, before the five-minute settlement deadline. Partial-radius diagnostics do not establish either result, even when their individual setup, spawn, or page timings are within budget.

## Coordinates and physical scale

| Contract | Value |
|---|---:|
| Cube edge | 16 blocks |
| Minimum world Y | -128 |
| Maximum world Y | 1407 |
| Vertical section range | -8 through 87 |
| Vertical section count | 96 |
| Sea level | Y=64 |
| Native model pixel | 30 meters |
| Blocks per native pixel | 4 |
| Horizontal block scale | 7.5 meters |
| Positive-elevation block scale | 7.5 meters |

World X maps to the model column and world Z maps to the model row. Native coordinates use floor division by four, including negative positions. Rectangles and page ownership are half-open.

The height conversion matches the scale-four reference with Rycraft sea level at Y=64:

```text
meters >= 0: trunc(meters / 7.5) + 64
meters < 0:  trunc(-sqrt(abs(meters) + 10) + sqrt(10)) - 1 + 64
```

`VerticalSectionMask` uses two words, so section 64 is not a shift boundary. Density interpolation has 193 possible Y levels for the full range, but cube generation evaluates only the interval required by the requested cube and surface neighborhood.

## Generation identity

Every v4 world has one immutable `GenerationIdentity`. It includes generator version 4, the unsigned 64-bit seed, model-pack and runtime hashes, Core ML provider configuration including the scalar Coarse graph and static Base and Decoder batches, model scale, window geometry, RNG revision, quantization revision, hydrology revision, and postprocessing revision.

Hydrology revision 12 persists a frozen branch-specific fall partition and uses reach-scoped channel projection plus receiver-owned standing-water backwater. An immutable closed-depression outlet whose standing stage cannot grade into its routed receiver owns an explicit branch fall; spatial proximity and unresolved cross-page stages cannot synthesize one. Revision-11 RYHY pages therefore have a different generation fingerprint and cannot be mixed with newly routed pages.

Postprocessing revision 9 reconstructs PREVIEW elevation from the FINAL latent low-frequency channel with the same cleanup operator, then derives its physical climate from that elevation. It retains the published Python and Java half-pixel, `align_corners=false` registration introduced by revision 8. Earlier PREVIEW pages remain rejected as incompatible, and profile metadata must select a separate identity namespace before current revision pages are generated. PREVIEW and FINAL RYTA pages, protected RYTG rectangles, and RYHY hydrology all remain fail-closed on seed or fingerprint mismatch.

The identity's SHA-256 fingerprint is persisted in metadata and every terrain-authority page. A page, edit profile, or world path with a different identity cannot be combined with the current world. Cache eviction may affect latency only, never page bytes. A corrupt RYHY envelope is rebuilt and atomically replaced. An outer-valid RYHY native payload is replaced only after its exact bad bytes have been proved corrupt. Fingerprint, seed, quality, and other persistence failures remain fail-closed for both PREVIEW and FINAL authority.

`AuthorityQuality` is `PREVIEW` or `FINAL`. A typed authority result is `READY`, `DEFERRED`, or `FAILED`. `WorldGenerationContext` binds one identity, quality, and authority and latches the first production failure. The render and fixed-tick threads may read completed results or enqueue work, but they must not perform or wait on inference.

The bounded coordinator treats priority as an admission contract. A SPAWN or EXPLORATION_EXACT request may displace unstarted visible, preview, or speculative work when all 64 request slots are occupied. A current protected handoff may also replace lower work and stale protected work from an older camera epoch. An active model call and an active atomic publication finish normally. Decoded page and transient-grid caches evict only entries from the same or a weaker lane, then use recency within that lane. Distant decoding therefore cannot remove a stronger current-player input merely because it arrived later.

`PREVIEW` pages are reserved for the coarse far horizon. They are decoder-free low-frequency reconstructions from the same latent lineage as FINAL terrain, not downsampled copies of FINAL output. Safe-spawn hydrology, exact cubes, and the native-hydrology topology and refinement closure use `FINAL` authority only. A coarse spawn selector may propose locations, but it never authorizes collision or water. The decoder residual may still refine local terrain and water topology, so the renderer replaces preview terrain and water together instead of pairing authority from different qualities.

## InfiniteDiffusion authority contract

The locked window geometry is:

| Stage | Window | Stride | Solver steps | Batch |
|---|---:|---:|---:|---:|
| Coarse | 64 by 64 | 48 | 20 | 1 |
| Latent | 64 by 64 | 32 | T=2 | 4 |
| Decoder | 256 by 256 | 192 | T=1 | 4 |

The shared foundation implements global half-open window intersection, portable PCG64, Marsaglia Gaussian generation, coordinate-addressed noise patches, separable linear weights with epsilon 0.001, and insertion-order-independent accumulation. It also fixes model rows to world Z and columns to world X.

`InfiniteDiffusionBackend` implements the production scalar coarse denoising loop, fixed four-window latent batches, lexicographically grouped four-window Decoder batches with deterministic repeated-tail padding, signed-square-root elevation encoding, Laplacian low-frequency and residual reconstruction, half-pixel bilinear interpolation corresponding to `align_corners=false`, low-frequency cleanup, climate reconstruction, and page quantization. PREVIEW and FINAL share the same latent accumulation; PREVIEW substitutes a zero decoder residual before the common cleanup and crop. Coarse climate sampling retains the same pixel-center formula at positive and negative coordinates. Compatibility ramp goldens lock that registration against the published Python and Java implementations. Acceptance still requires repeatable real-model hashes. Do not report paper-faithful generation as qualified from source inspection or fake-backend tests alone.

## Learned fields and macro adaptation

Each native sample carries:

- Elevation in meters
- Mean temperature in degrees Celsius
- Temperature variability in degrees Celsius
- Annual precipitation in millimeters
- Precipitation coefficient of variation
- Lapse rate in degrees Celsius per meter

When a v4 `WorldGenerationContext` is present, learned elevation is the only macro elevation, and four learned climate variables plus the derived lapse-rate field are the only macro climate authority. Existing plate uplift and synthetic continental or climate fields do not modify those values. Temperature at a post-authority local height uses the derived lapse rate and the shared 7.5-meter scale.

Geology, lithology, rock resistance, strata, caves, ores, aquifers, structures, and volcanic primitives remain bounded procedural systems. V4 adds deterministic shield, stratovolcano, and warped-caldera relief to the learned physical elevation in meters before native routing. A crater becomes wet only when the canonical depression hierarchy finds a supported stage and spill; the analytical crater-lake overlay remains v3-only and cannot override that result. After routing, v4 may add at most 1.5 blocks of footprint-filtered local relief to a dry slope. Channel, outlet, lake-rim, coast, divide, transition-owner, and every wet category gate the residual to zero, so it cannot delete water, create a retaining wall, or change body topology. The old 16-block `BasinSolver` hydraulic-erosion passes and alpine postprocessing remain v3-only. V4 uses the learned elevation, pre-routed volcanic adjustment, canonical bed correction, and bounded dry residual without allowing a legacy raster to overwrite the learned surface or imprint its storage phase on water. The code retains the existing biome, flora, canopy, and fauna consumers through a physical-climate adapter. Its soil moisture uses a bounded annual precipitation and potential-evapotranspiration balance, with precipitation variability increasing effective demand without inventing a monthly phase. The model itself supplies no biome IDs. The ecosystem follow-up begins with a golden native-grid crosswalk against the pinned mod's hand-written `BiomeClassifier`, then replaces the temporary adapter with continuous PFT capacity informed by canonical water and soils instead of copying fixed noise or vanilla sea rules. Equilibrium plant-functional-type fractions are deferred to PR 2.

The context-free `MacroGenerationSampler` remains the legacy synthetic generator for tests and `RYCRAFT_DIAGNOSTIC_V3=1`. It is not a fallback for a failed v4 query.

## Regional weather boundary

Static learned climate remains immutable world-generation authority. `WeatherSystem` samples only learned elevation and climate in bounded spatial batches. It does not route hydrology or construct geology, soil, biomes, or ecology for its horizon grid. Weather is a separate presentation system reconstructed from the generation identity, full-width coordinates, static climate, and saved world time. Its latest-wins utility worker builds an 81 by 81 camera-centered grid at 256-block spacing. Each immutable snapshot carries two deterministic slices 200 ticks, or ten real seconds, apart. Gameplay retains the prior valid snapshot while a replacement is built and recenters only after 1,024 blocks of movement. Deferred authority leaves the prior snapshot resident and retries from the fixed tick without terminating the worker.

Each sample derives pressure, relative humidity, wind in blocks per second, temperature, cloud coverage and type, precipitation intensity and kind, storm potential, fog extinction, aerosol density, terrain height, and cloud base and top. Coordinate-pure pressure, moisture, temperature, instability, and front fields are biased by the learned climate. Rain or snow follows temperature rather than a biome label. Surface wetness integrates precipitation and dries from temperature, wind, and sunlight. Foliage, clouds, fog, precipitation, lightning, and thunder consume the same immutable snapshot.

Weather does not write blocks, fluids, terrain authority, column plans, or gameplay rules. Snow does not accumulate, rain does not create runoff or flooding, and lightning creates no fire or edits. Deterministic storm cells derive lightning IDs and positions from the seed, fixed time buckets, and cell coordinates. Thunder uses the same events with physical distance delay at 343 meters per second. Generator v4 converts horizontal and positive vertical distances at 7.5 meters per block. `RYCRAFT_WEATHER` provides stable `clear`, `overcast`, `rain`, `storm`, and `snow` capture authorities.

Continuity qualification compares each former storage line with eight nearby control lines. Its orientation histograms therefore use exposure-scaled Dirichlet pseudocounts. A control histogram with eight times as many samples receives eight times the prior mass, so mutually empty direction bins cannot create false grid evidence while an actual structured seam still exceeds the unchanged qualification threshold. Equal-exposure comparisons retain the original one-count-per-bin prior.

## Terrain-authority pages

One immutable page covers 256 by 256 native pixels, or 1,024 by 1,024 blocks. Preview and final pages are separate:

```text
terrain-authority-v1/
  preview/p.<row>.<column>.ryta
  final/p.<row>.<column>.ryta
  transient-final-v1/g.<row-begin>.<column-begin>.<row-end>.<column-end>.rytg
```

Each LZ4-compressed `RYTA` file records schema, quality, signed page coordinates, seed, full generation fingerprint, native dimensions, channel mask, payload lengths, payload CRC-32, and header CRC-32. The payload uses fixed 12-byte quantized samples. Publication uses a temporary file, `fsync`, rename, and directory synchronization.

An `RYTG` file uses the same 12-byte quantization for one exact half-open FINAL rectangle. Only
spawn and protected-handoff requests persist these rectangles. Restart loads validate the full
identity, rectangle, sizes, header checksum, and payload checksum before exposing samples. A corrupt
file is inferred again and atomically repaired. A fingerprint mismatch fails closed.

The decoded authority cache defaults to at most 1,024 entries and 512 MiB. A query is limited to 64 pages and 1,048,576 samples. At most 64 page requests may be outstanding and one build runs at a time. Equal cold requests share one flight. The production backend separately caps retained coarse, latent, and decoder tensor windows at 384 MiB. The coordinator orders spawn, exploration exact, protected exact handoff, visible final refinement, coarse preview, and speculative movement prefetch. Production final-parent requests use the protected lane for the exact-handoff prefix. Movement hints use the speculative lane only after the current visible preview closure is ready and are capped at eight pages.

Protected FINAL targets sort directly intersected native hydrology owners lexicographically and
group adjacent owners in sets of at most two by two. Every combined half-open FINAL rectangle stays
within the 1,048,576-sample query bound. Cropping the grouped output back to one 517 by 517 owner
produces exactly the same FINAL samples as preparing that owner separately, including negative
coordinates and shared aprons. The grouping changes request count and reuse, not terrain authority
or generation identity.

## Canonical generated water

Generated water is complete static geometry and implicit source state. It does not use runtime ticks to settle.

The governing invariants are:

- Hydrology may lower terrain to form a bed or open a route.
- Hydrology may not raise dry terrain only to hold water.
- A level conflict may not be resolved by deleting a wet cell.
- A standing body has one stable identity and one flat stage.
- A river stage is monotone along ordinary flow.
- An ordinary exposed river top uses the nearest eighth-block static fluid state, while every
  completely covered water voxel beneath it remains an implicit source.
- An abrupt stage change must be owned by an explicit rapid or waterfall.
- Every standing-water column has solid support and implicit source water through its surface.
- Exact cubes and far tiles consume the same body, stage, bed, flow, and shoreline authority.

The v4 path removes conflict-bank targets, categorical bank dilation, dry retaining-wall overrides, sealed artificial support columns, and shore-raising paths from its native hydrology and cube emission. The retained v3 `BasinSolver`, including its iterative hydraulic-erosion passes, is an explicit diagnostic compatibility path and is not used for v4 macro terrain. V4 derives a discharge-aware bed cut from learned elevation and canonical water stage rather than applying the legacy eroded raster. Conflicting connected stages reconcile by compatible stage relaxation, route opening, or explicit transition ownership rather than wall construction.

The implemented v4 route uses deterministic local native-page Priority-Flood and half-open shared-edge ownership. Its input is learned `elevationMeters`, not a converted block height: ocean classification uses values below 0 meters, Priority-Flood, lake components, and D-infinity retain those native meters, and gradients divide by the physical 30-meter raster spacing. The native lake-depth threshold is 0.9375 meters, while source stages, beds, raw terrain, and persisted sampling fields are emitted afterward in Rycraft block-height coordinates through the shared learned-elevation conversion. This preserves sub-block slopes in routing without changing the game-facing vertical contract. A one-component local lake retains its 256-block coarse anchor through ordinary preview and final refinement. If two disconnected local components collide in that anchor, each receives a separate exact local-component ID rather than sharing lake statistics or body ownership.

D-infinity keeps both physically significant receivers for runoff, but a receiver branch becomes visible geometry only when its target is a finished ocean, lake, river, or wetland cell with a canonical stage and body. A subthreshold branch therefore cannot terminate as a source-height puddle. Ordinary raster stages use a one-sixteenth-block-per-block upper grade so rounding exposed tops to eighth-block states cannot introduce a two-level step. Curved routes interpolate stage by sampled arc length, and overlap reconciliation is limited to one ordinary reach instead of borrowing a nearby body's stage. An abrupt descent is legal only on a frozen routed-fall branch or at an immutable closed-depression outlet whose standing stage cannot grade into its receiver. The latter owns the complete receiver edge as one half-open fall, while an edge-open depression waits for final cross-page spill authority and cannot emit a page-local curtain. River contacts lower smoothly into sea level, and a river entering a lower lake owns an explicit descent. When a small positive learned elevation quantizes to Y=64, the compatible downstream stage remains monotone and sampling cuts the supported bed below it instead of clamping the route to Y=63.875. These rules lower beds when required and never raise dry terrain or create a retaining wall.

Each immutable page also persists local depression anchors, stages, half-open core mass, natural outlets, edge masks, and opposing-edge samples. Before returning any sample of an edge-connected lake, the router follows the component's tiled spill summaries in lexicographic order. It merges components only when both pages report lake water above the identical substrate at the same native edge sample. Stage reduction monotonically removes a portal if its substrate would become dry, then recomputes the source component to a fixed point. The merged component uses one flat compatible stage, its lexicographically rooted stable identity, core-owned area, volume, and runoff without apron double counting, and either the lowest compatible interior outlet or an explicit endorheic result. A dry edge on either side remains a barrier. This prevents request order, concurrency, cache state, and restart state from changing a page-edge stage, mass, outlet, or identity. Hydrology may lower the bed of a proven connected fringe when it lowers a conflicting stage, but it never raises dry terrain.

The tiled spill traversal admits at most 64 connected owner pages and 256 component nodes, enough for a 131.072-kilometer one-page-wide chain. Reaching either limit fails closed instead of publishing a partial stage or identity, so no query performs a whole-world walk. Synthetic three-page and eight-page chains, including negative coordinates, qualify reverse order, concurrency, portal drying, aggregate mass, natural outlets, and endorheic closure.

Each page also persists physical native elevation, local runoff, and a four-bit receiving-edge hint derived from real escape paths across its core. When a locally wet component reaches an edge, or a locally dry page can receive a depression through one of those edges, the router combines the bounded owner rectangle and reruns the deterministic minimax flood on physical elevation. It expands only while the component can continue through an outer edge and fails closed at the same 64-page bound. The resulting immutable region owns its flat stage, mass, stable identity, and signed shoreline on both sides of the former page seam. Exact samples, dense grids, and compact far-topology cells share that region even when the first request arrives on the locally dry side or after a persisted restart. Exposure of a dry receiving page therefore cannot depend on query order, and a newly wet step-32 cell cannot disappear because its local page reduction was dry.

Shallow, low-variability groundwater cells enter a connected wetland graph instead of a one-cell decoration pass. The graph follows both significant D-infinity receivers through other candidates until it reaches a finished ocean, lake, river, or wetland. It crosses half-open owner pages, visits at most 64 pages and 262,144 native cells, and fails closed instead of publishing a partial identity. Every resolved cell inherits one parent stage and body ID, promotes hydraulic head to that stage, and receives a one-eighth-block supported bed by lowering terrain only. Exact columns, implicit fluids, materials, and every far LOD consume that same authority. Candidate cells set both wet and dry compact-topology evidence, so a four-block fringe cannot disappear at step 32 before the connected solve runs.

A low-gradient river that reaches the ocean becomes an estuary for up to 64 native cells of sea backwater. The category is explicitly brackish, cannot cross a waterfall, and follows receiver ownership across page boundaries. A qualifying mouth preserves its physical receiver and adds one deterministic secondary ocean receiver within four native cells. The stable seed chooses the branch side and a 0.32 through 0.42 secondary discharge fraction, the primary receives the remainder, and both ribbons retain the same junction identity. The secondary target is chosen for lateral separation, so the result is an actual visible distributary rather than two coincident labels on one raster edge. Schema-6 RYHY pages persist the receiver pair, weight, delta and estuary flags, and frozen two-bit fall-branch partition from which brackish and transition identity are derived. Exact and far sampling reconstruct the same splines, branch discharge, stage, and body ownership after cache clearing or restart.

The no-raising correction, tiled lake reconciliation, connected wetlands, coastal distributaries, and estuaries are implemented and covered by deterministic fake-authority tests. The complete locked hydrology design still requires the real-model visual matrix, including cold-start and step-32 captures, before it can be declared qualified for merge.

`NativeHydrologyCacheMetrics::deferredBuilds` counts completed cache build attempts that returned a
typed `DEFERRED` result because learned authority was not ready. It is monotonic and does not mean
currently parked pages, active builds, or failures. Phase logs subtract the preceding snapshot for
an interval count and report `activeBuilds` directly.

## Far terrain and water

Every visible 256 by 256-block tile requests a step-32 parent and can refine through steps 16, 8, 4, and 2 as required. Cold entry publishes the connected step-32 terrain-and-water parent prefix through 96 chunks. Once that frontier reaches the near band, preparation begins the camera-aware protected FINAL closure while exact publication continues. The closure contains 4 targets at step 1, 8 at step 2, 12 at step 4, 16 at step 8, and 20 at step 16. All 60 publish atomically before gameplay. A parent remains drawable until an eligible connected replacement is resident. Exact cubes own the first 32 chunks. The maximum-coarseness bands are step 2 through 64 chunks, step 4 through 128, step 8 through 256, and step 16 through 512. Gameplay may retain finer geometry above the 0.55-pixel refinement threshold and coarsens outward one adjacent tier at a time only below 0.45 pixels. After the connected 96-chunk entry prefix is drawable, unfinished exact publication through 32 chunks or any connected visible desired-LOD miss pauses ordinary outer-parent submission and publication. Near jobs proceed nearest-first, rank horizontal distance before projected error within the nearby visible class, and may displace queued or dependency-parked outer parents. Displayed parents, the connected prefix, active transition endpoints, exact fallbacks, and protected lineage remain pinned.

This is a hybrid hierarchy, not a literal sparse voxel octree. Mutable exact cubes remain sparse and hash-indexed for dense, write-heavy near simulation. Distant terrain and canonical water remain in the two-dimensional tile hierarchy because their visible authority is a single-valued surface. One frame-level coverage publication joins both systems while retaining parents until a connected child patch is ready.

The renderer recomputes movement hints only after the camera travels one chunk. It selects at most eight authority pages just beyond the visible closure in that direction. Visible preview pages and protected final handoff parents always outrank these hints, so prediction cannot consume the queue capacity needed to draw the current view.

`FarTerrainMesher::build` creates an immutable base mesh containing terrain, standing water, and falls without collecting vegetation anchors. `buildCanopyAttachment` produces a separate optional payload containing tree forms and deterministic ground-flora aggregates on the lower-priority flora lane. Ground flora uses globally anchored eight-block cells, the exact habitat and material rules, nested retention through steps 2, 4, 8, 16, and 32, and half-open tile ownership. Fine and middle tiers emit bounded clumps, while the two coarsest tiers widen repeated-texture billboards so projected plant cover does not collapse. Nearby displayed tiles publish PREVIEW ecology first, grounded against their displayed PREVIEW or FINAL terrain. Missing drawable attachments fill the bounded refresh batch before provisional FINAL promotions, then both classes remain nearest-first. The canopy budget is zero during preparation and until the connected 96-chunk prefix is drawable. Gameplay then guarantees exactly one low-priority canopy worker. Missing PREVIEW attachments remain ahead of FINAL ecology promotion on that worker. The provisional allocation remains drawable until its FINAL replacement is resident and exchanges atomically. Terrain and water publication therefore does not wait for vegetation discovery, and flora arrival never replaces or changes the base allocation or its deterministic hash. Presence of an attachment cache entry is the sole completion signal; an explicit empty attachment records that a tile has no flora geometry.

Coarse water uses canonical geometry callbacks and `waterTopologyPossible`. If a step-32 cell can contain a narrow channel, body contour, or volcanic water feature despite dry corners, the mesher performs bounded interior probes and resolves a finer contour. A narrow route may become subpixel-thin, but it must not disappear merely because corners miss it. Step-32 preparation preserves topology-certified sparse sample masks even above half-page occupancy and decomposes each mask into lexicographically discovered rectangles for canonical grid evaluation. It requests a complete 66 by 66 water page only when every sample is required.

For local real-model diagnosis, `rycraft_worldgen_inspect --v4-model MODEL_PACK SEED X Z preview --horizon-water-profile` reports the step-32 standing-water gate, topology-marked parent count, and native water-grid calls and samples. Add `--horizon-radius 64` only for a bounded diagnostic; without that option, the inspector measures the full 512-chunk production maximum. The JSON records the selected radius as `horizon.radius_chunks`. Before it requests authority, the inspector binds a deterministic temporary authority profile under the system temporary directory using the complete seed and fingerprint. It never opens or writes `rycraft_world_v4`. `--profile /absolute/external/directory` selects a durable external inspector profile instead; paths inside the configured user Application Support root are rejected. This opt-in mode instruments only the inspector's source callbacks; it does not add counters or synchronization to production far meshing.

All production skirt quads are disabled. Each far tile's outer cell ring uses a shared transition topology on a canonical two-block boundary lattice, so adjacent tiers use identical edge heights and positive-area geometry remains half-open to one tile. Displayed neighbors are limited to a 2:1 step ratio, including both endpoints of an active replacement. Ordinary replacements proceed through adjacent tiers. Interior transition-marked vertical geometry is permitted only across a real source-column height discontinuity and may not lie on a tile face or bridge an LOD mismatch. Portable tests cover every edge and corner orientation, steps 2 through 32, negative coordinates, duplicate ownership, winding, area, and fixed mesh budgets. Visual qualification must still inspect all joins for cracks, panels, and ledges.

Before entry, the preparation renderer advances exact mesh publication, far authority polling, base-mesh scheduling, result draining, and shared-buffer publication without rendering the full world scene. It completes a connected terrain-and-canonical-water step-32 prefix through 96 chunks. As soon as that connected frontier reaches the near band, the bounded protected FINAL near closure advances in the camera-critical lane. Entry requires the exact 4, 8, 12, 16, and 20 target distribution at steps 1, 2, 4, 8, and 16, all 60 matching FINAL parents and children, exact compatibility, legal shared boundaries, the 27-cube collision halo, and revision-ready exact spawn meshes. Its readiness identity cannot accept counters from a smaller or stale selection. Camera-critical jobs and CPU and GPU residency may evict optional distant work and cannot be evicted by it. Ordinary refinement and canopy work remain dormant until gameplay.

## Near-field publication contract

After entry, exact terrain owns a full 32-chunk radius. Every required surface section in that disk
keeps generation, meshing, completed-result upload, and deferred-lighting priority over optional
flora and broad work. The camera column is the highest lane, the six-chunk exploration band follows,
and the remaining required disk follows that. Flora-bearing exact sections within 16 chunks run only
after required terrain and before broad distant exact work. Far canopy workers remain at zero during
entry preparation and until the connected 96-chunk far-terrain prefix is drawable. Gameplay then guarantees one
low-priority canopy worker even while an unfinished exact surface or flora column remains inside the
16-chunk flora radius or protected local terrain debt persists. Continuously replenished exact work
therefore cannot leave every far attachment queued forever. Flora-only tree sections publish visual
and collision ownership together only after their complete vertical flora column is ready. The
canopy service remains capped at one gameplay worker after exact and local terrain debt drain. Canopy refresh skips
fully exact-owned tiles so its bounded request batch advances immediately to the nearest far-owned
vegetation gap.

The far scheduler does not compete for all physical cores while the near world is incomplete. It
pauses ordinary outer-parent submission and publication while exact publication or connected
visible desired-LOD debt remains. Local far work admits 8 of its 16 workers alongside exact debt,
12 after exact debt clears, and all 16 only after both debts clear. Nearby visible work ranks
distance before projected screen error. Within the protected patch, a missing PREVIEW step-32 parent uses urgent
coverage admission and may replace a lower-ranked queued or dependency-parked ordinary parent.
Current protected FINAL children and parents run before required PREVIEW bridges. Unused bridge
capacity returns to current FINAL work before at most one directional predicted anchor receives
CPU-only staging.

The requested protected FINAL key, and only that role-selected key, is a critical residency class.
It may reclaim optional distant non-displayed refinement or canopy entries from the CPU cache and GPU
arena, and it may use the complete GPU arena. Active coverage, displayed surfaces, transition
endpoints, exact fallbacks, active protected lineage, and the requested critical keys remain pinned.
The coarse parent stays drawable until the critical replacement commits.

Exact collision follows the renderer's visual publication boundary. Each frame publishes the set of
revision-ready exact sections with the matching exact-coverage epoch. A stale epoch is rejected.
Owned sections use their loaded exact blocks and fluid cells, and a missing owned cube remains closed.
Unowned sections use the immutable `ColumnPlan` terrain and canonical generated-fluid profile, while
an unresolved plan stays closed. Physics therefore cannot switch early merely because one cube
loaded before its visible column became authoritative.

Cube insertion also starts one bounded first-publication lighting transaction. It settles the
arriving cube, changed sky paths, and affected face neighbors under the world lock, with at most 32
floods in one transaction. Remaining work enters a camera-ranked bounded queue. A pending cube cannot
produce a mesh snapshot, so the first visible version already contains its settled packed light.

## Dry-land spawn selection

A new v4 world never finalizes an arbitrary requested coordinate as a spawn. Bootstrap samples the
coarse model directly and does not write preview authority pages while it searches. A coarse cell
is 256 native pixels, or 1,024 world blocks. At 7.5 meters per block it spans 7.68 kilometers.
The selector queries one stable page-aligned 16-cell square. Its 61.44-kilometer half-edge is the
largest representable bound below the 64-kilometer search contract. It ranks ocean-backed land
first, then inland and remaining coastal land, and retains only the best proposal from each aligned
two-by-two authority-page hydrology owner. At most 81 owners can intersect the odd-aligned search.
Final authority, canonical water, and exact collision still decide whether a proposed cell is legal.

For one proposal, the canonical screen directly prepares only its 2,048-block native hydrology
owner. It checks the requested chunk first, then scans the strict owner interior on the globally
aligned four-block native raster. Bounded batches use at most 16 workers. The nearest locally flat
center with a dry five-by-five canonical safety buffer first attempts the complete cold streaming
footprint. It next attempts the exact 113 by 113 radius-zero safety footprint, which is the union of
the center column's five-by-five `ColumnPlan` dependency apron and every plan's 49 by 49 hydrology
raster. If the original chunk cannot provide that proof, at most 64 deterministic candidates pass
through the four-block owner mask before exact certification. Candidates must remain inside the
already materialized learned page, so a successful relocation adds no terrain page or graph call.
The scan neither opens a neighboring hydrology owner nor prepares the wider exact band. If a river,
lake, wetland, or coast intersects either wider footprint, this optimization is rejected and the
screen installs only the original 25-sample local certificate. This fallback preserves valid dry
land near canonical water without weakening any water test.

A positive-elevation continental owner may have no permanent-ocean terminal from which a
conservative page-local dry proof can be derived. In that case the screen may offer one locally flat
learned site as explicitly provisional and installs no dry certificate. World construction then
starts only radius-zero exact generation. The UI continues reporting the land search, horizon work
remains dormant, and metadata remains unwritten until the exact plan proves canonical water absence,
solid support, two blocks of headroom, acceptable slope, and the nearby dry neighborhood. A
rejection advances to the next deduplicated owner. After acceptance, world construction starts with
a zero nominal exact radius. Its mandatory mesh halo keeps the camera column and four cardinal
neighbors active, while the full 32-chunk exact disk waits until entry. A bounded all-ocean result
is a visible failure and never falls back to an ocean coordinate.

For local real-model qualification, run
`rycraft_worldgen_inspect --v4-model MODEL_PACK SEED X Z final --dry-spawn`. The inspector invokes
the same bounded coarse-to-final proposal path and canonical-water screen as world bootstrap. Each
result comes from one directly prepared 2,048-block owner. The common path searches its globally
aligned four-block native raster and prefers a complete cold or 113 by 113 exact-safety certificate
before using the 25-sample local fallback. A continental provisional result is identified separately
and has no certificate. The JSON emits
the selected ordinal, coordinates, canonical-water rejection count, local relocation count,
duration, and isolated `authority_profile`. It deliberately does not substitute for the exact-plan
safety check, which still validates support, headroom, slope, and nearby dry columns before entry.

V4 metadata records `playerPos` separately from `safeSpawnPos` and records
`spawnSafetyRevision`. Normal saves update the resumable player location but never replace the
verified safe spawn. A profile written before the current dry-land rule, before the separate
safe-spawn field, or with an unknown safety revision is revalidated once. Its legacy `spawnPos` is
treated only as the old player location. A legacy `safeSpawnPos` is a deterministic first recovery
anchor, not unconditional proof of land. If its bounded search has no legal dry candidate, startup
retries once from the requested fresh-world anchor. If the proposed location is ocean, lake, river,
fall, wetland, delta, a water transition, too steep, unsupported, or lacks headroom, startup
searches and verifies a replacement before allowing entry. An exhausted bounded search fails closed
instead of placing the player into water.

## Native hydrology build admission

Preview and final native-hydrology routers retain independent immutable caches, but cold page construction shares one process-wide admission gate. The gate allows at most 16 page builds, further limited by reported hardware concurrency and the one-GiB aggregate scratch reservation. It prevents separate preview and final contexts from each oversubscribing the same CPU and unified-memory budget while allowing independent page builds to use available cores. Waiting requests are admitted by authority priority, and a duplicate exact request promotes the existing page flight. Active page solves are not interrupted. Native-page caches use the same priority-aware eviction rule as decoded learned authority, so distant hydrology cannot evict a stronger exact owner. Per-page single flight remains responsible for duplicate requests.

## Runtime water

A gameplay edit activates one fluid cell and its six neighbors. The fixed 20 Hz scheduler applies downward-first source and level rules with a five-tick delay, loaded-only reads, bounded pending work, and indexed deferred frontiers. Generated standing water remains an implicit source until an edit changes a cell. A generated partial river top is an immutable render and collision state until a gameplay edit activates its neighborhood. Generation and ordinary loading enqueue no fluid work.

Generated and runtime stable water emits planar top geometry. Animated fragment normals provide motion without changing ownership or surface position. Vertical water sides are reserved for explicit falling columns.

One fluid tick processes at most 1,024 deduplicated cells. Pending updates and deferred frontiers are each capped at 65,536, and catch-up is limited to eight ticks. A water update that reaches a missing cube records an activated frontier under that destination cube. Loading a cube makes only matching frontiers eligible for the bounded resume budget. Deferred frontiers persist through column manifests, while ordinary generated water without a frontier remains inactive on load.

Mesh snapshots carry blocks, fluid bytes, and packed smooth skylight and block light through the complete 18 by 18 by 18 halo. Water uses eighth-block flowing heights, flow-direction bits, and a falling flag while retaining the shared 16-byte vertex layout. Stable sources fill a complete voxel and meet far source-water geometry on the same full-block plane. Buoyancy and head-submersion checks use exact fluid state only for a renderer-published exact section. Before that ownership handoff they use the same canonical generated-fluid proxy as collision, so physics cannot disagree with the visible parent.

## Persistence profiles

Generator v4 uses:

```text
~/Library/Application Support/rycraft/rycraft_world_v4/
  metadata.json
  regions-v4/
  terrain-authority-v1/
  hydrology-authority-v1/
```

No v4 path is published before bootstrap qualification. Metadata records the unsigned 64-bit seed, 64-character generation fingerprint, and spawn-safety revision. The first explicit creation may use `rycraft_world_v4`; later creations reserve a collision-free identity-named sibling such as `rycraft_world_v4-seed-<16-hex-seed>-fingerprint-<64-hex-fingerprint>`, with a numeric suffix when needed. Opening a selected profile requires its exact metadata identity and never silently redirects to another path. After selection, the qualified terrain and hydrology authorities bind to that same profile's `terrain-authority-v1` and `hydrology-authority-v1` directories. Missing or corrupt metadata fails closed rather than being treated as a new world.

The legacy profile remains separate:

```text
rycraft_world/
  metadata.json
  regions-v3/
```

Ordinary v4 startup never examines, migrates, or rewrites the legacy directory. The Worlds screen may explicitly use a legacy or incompatible profile as a successor source. That action creates a separate current-v4 profile with compatible metadata and leaves the source regions, manifests, edits, and fluid frontiers untouched. `RYCRAFT_DIAGNOSTIC_V3=1` constructs a no-save v3 world.

Edited cubes remain LZ4-compressed RYCH files with CRC-32 validation and per-column manifests. Authority pages are not ordinary edit cubes and use their own `RYTA` schema and identity checks.

## Tests and qualification

Ordinary CI uses `DeterministicFakeTerrainBackend` and must remain network-free:

```bash
meson setup build --buildtype=debugoptimized
ninja -C build tests/test_rycraft
./build/tests/test_rycraft "[learned]"
./build/tests/test_rycraft "[bootstrap]"
./build/tests/test_rycraft "[generator-v4]"
./build/tests/test_rycraft "[chunk][coords]"
./build/tests/test_rycraft "[reported-water-continuity]"
```

Tag names should be checked against the built test list before relying on a focused command. The complete suite remains `ninja -C build test`.

Only modified cubes persist. Procedural cubes, far tiles, meshes, packed smooth voxel light, regional weather, lunar phase, and lightning buckets regenerate from the generation identity, coordinates, current loaded neighborhood, learned climate, and saved world time. The loader verifies the CRC-32 before deserializing block or fluid state. Unloading queues serialization and compression on the save thread. The queue is bounded to 32,768 cubic positions, and repeated snapshots for one queued position coalesce to its newest revision instead of growing the queue. Backpressure applies at the bound, while the quit path sweeps still-loaded modified cubes before flushing. Manifest serialization is ordered separately from the short in-memory manifest lock, so file I/O never holds the lookup lock. Metadata remains in `metadata.json` and records the seed, the safe world-start anchor, the current respawn anchor and bed provenance, the last player position and orientation, health, hunger, the selected hotbar slot, a 36-slot item-stack inventory as `[type, count, durability]` triples under `inventorySlots`, the cursor plus nine crafting-input stacks under `carriedStacks`, the display name, the game mode, the generation toggles, created and last-played timestamps, world time, chunk format version 4, and generator version 4.

V4 metadata extends the locked generation identity with the fixed safe-spawn anchor, current respawn anchor and bed provenance, resumable player position and orientation, health, hunger, selected hotbar slot, 36 inventory stacks, ten carried cursor and crafting stacks, display name, game mode, generation toggles, created and last-played timestamps, and world time. Metadata parsing remains tolerant of older gameplay fields, but it may not accept a mismatched v4 seed, generator fingerprint, or chunk format into the same profile.

Furnaces and chests are the stateful blocks. They persist in a per-world plaintext sidecar `block_entities.dat` with a versioned `RYBE 1` magic: one `furnace <x> <y> <z> <burn> <burnTotal> <cook> <in...> <fuel...> <out...>` line per furnace and one `chest <x> <y> <z> <slot0...> ... <slot26...>` line per chest, written atomically alongside metadata and loaded once at world start into a `BlockEntities` struct. The read is forward tolerant: unknown leading record types are skipped, and one malformed line is dropped with a single logged error. This keeps block entities off the RYCH cube format and the RYCM manifest with no version bump. Dropped item entities and boats are never persisted; they despawn on quit and on a world switch.

Real-model qualification must be explicit and local. Before reporting it as passing, record:

- Model revision, all asset hashes, runtime hash, machine, and macOS version
- Core ML and CPU fallback partition counts
- Canonical qualification digest and generation fingerprint
- Preview and final page hashes for fresh, reverse, concurrent, and cache-cleared requests
- Exact and far mesh hashes at all six footprints
- Queue and cache maxima
- Cold startup, spawn readiness, horizon completion, and five-minute settlement times
- Process RSS, Metal allocation or residency, and highest credible unified-memory total
- Metal validation logs and opened captures

Per-world generation toggles (structures, fauna, weather, day cycle) are stored in metadata and require no generator-version bump: the default toggle set produces byte-identical cubes to the settings-free path, verified by a serializer byte-comparison regression at seed 42. Only the structures toggle reaches the generator, where a disabled `StructurePlacer` deterministically reserves and emits nothing so trees fill former structure sites; fauna, weather, and the day cycle are engine-side gates. An old binary that opens a save containing a new block id rejects those cubes as incompatible and regenerates them, an unsupported downgrade path.

Weather tests cover deterministic fronts, spatial and temporal continuity, climate bias, physical wind units, precipitation type, cloud profiles, stable presets, latest-wins worker bounds, lightning IDs, and thunder delay. Rendering tests separately qualify five-cascade selection and refresh, smooth packed-light propagation, Hi-Z screen-space lighting history, atmosphere LUT finiteness, clouds, fog, and weather integration. These headless checks do not replace opened Metal captures.

The streaming active-set builder includes manifest `savedSections` for visible horizontal columns, so an off-surface build is rediscovered without generating every vertical section.

The opt-in entry audit exercises one process from runtime qualification through canonical dry-spawn
selection, the complete FINAL spawn authority closure, exact spawn readiness, the connected
96-chunk preview entry prefix with the 512-chunk horizon still selected, and the camera-aware
60-target protected FINAL closure. The closure begins during preparation when the connected
frontier reaches the near band. The 30-second gameplay gate ends only after the safe FINAL spawn,
revision-ready exact meshes, connected PREVIEW entry prefix, and atomic protected closure are ready.
PREVIEW parents remain drawable while exact 32-chunk publication and nearest desired-LOD debt run
first. Ordinary outer-horizon submission and publication resume only after those debts clear and
still must finish within the five-minute settlement deadline.
Point the audit at an existing empty external profile and
run it only while the machine is otherwise idle:

```bash
RYCRAFT_TERRAIN_MODEL_PACK=/absolute/path/to/the/verified/model/pack \
RYCRAFT_TERRAIN_REAL_ENTRY_PROFILE=/absolute/path/to/an/empty/external/profile \
RYCRAFT_TERRAIN_REAL_ENTRY_EXPECT_COLD=1 \
./build/tests/test_rycraft \
  "Seed 42 real-model entry audit completes exact and far coverage"
```

`RYCRAFT_TERRAIN_REAL_ENTRY_EXPECT_COLD=1` requires the fresh seed-42 spawn phases to use exactly 80
coarse, 14 Base, and 5 decoder calls. Omit it when inspecting a warm or intentionally interrupted
external profile; the audit still rejects any phase that exceeds those cold bounds.

The audit has one five-minute deadline for the entire route. Its warnings report phase time,
qualification, dry-spawn, final-spawn, horizon-preview, and protected-handoff model calls, authority
and hydrology cache activity, exact-plan progress, and far
scheduler progress. A run performed while another game, compiler, or benchmark is consuming the
CPU is diagnostic evidence only and may not be cited as a performance qualification.

The acceptance targets are a safe final spawn, revision-ready exact spawn meshes, connected coarse terrain and canonical water through 96 chunks, and the 60-target protected FINAL closure within 30 seconds; configured 512-chunk parent coverage and all remaining required queues within five minutes; at least 60 FPS at native resolution with 4x MSAA and view distance 512 on the documented M4 Max; and no more than 64 GiB total unified memory. These are hardware gates, not results established by fake-backend tests.

## Research boundary

Generator v4 follows [InfiniteDiffusion paper v4](https://arxiv.org/abs/2512.08309v4), pins the published [30-meter ONNX model at revision `ad2df557`](https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/tree/ad2df557eca5645f588766101cf3bc3682455c3e), and uses the [Minecraft implementation at commit `23d3f50`](https://github.com/xandergos/terrain-diffusion-mc/tree/23d3f50e5108882bb88a03c3ab048aa63633a02f) as a compatibility reference. The implementation must remain independently testable through public model behavior, fixed random vectors, window geometry, and quantized outputs. Existing geology, hydrology, ecology, rendering, and persistence systems are Rycraft extensions.

[National Weather Service guidance on fronts, precipitation, and thunderstorms](https://www.weather.gov/lmk/basic-fronts) informs the regional relationships among pressure, moisture, temperature, wind, precipitation, and instability. Rycraft implements a deterministic visual weather system, not a forecast model or terrain-changing climate simulation.

PR 2 adds equilibrium plant-functional-type fields and connects them to flora, canopy, and fauna capacity. Succession, animated seasons, fire spread, migration, predators, and food webs remain out of scope.
