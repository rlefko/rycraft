catch2_proj = subproject('catch2', default_options : ['tests=false'], required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

test_sources = files(
  'test_common.mm',
  'test_trace.mm',
  'test_world.mm',
  'test_weather.mm',
  'test_worldgen_advanced.mm',
  'test_geology_ecology.mm',
  'test_organic_geology.mm',
  'test_natural_bank_substrate.mm',
  'test_macro_control.mm',
  'test_final_field_artifacts.mm',
  'test_reported_water_continuity.mm',
  'test_v4_water_antipatterns.mm',
  'test_relief.mm',
  'test_alpine_morphology.mm',
  'test_tree_geometry.mm',
  'test_render.mm',
  'test_items.mm',
  'test_recipes.mm',
  'test_entity.mm',
  'test_fauna_habitat.mm',
  'test_fluid_persistence.mm',
  'test_saves.mm',
  'test_learned_terrain.mm',
  'test_hydrology_authority.mm',
  'test_native_hydrology.mm',
  'test_terrain_bootstrap.mm',
  'test_terrain_runtime.mm',
  'test_render_concurrency.mm',
  'test_engine.mm',
  'test_audio.mm',
  'test_lightning.mm',
)

test_exe = executable('test_rycraft',
  test_sources,
  include_directories : inc,
  link_with : [libcommon, libworld, librender, libentity, libengine, libaudio],
  dependencies : [catch2_with_main_dep, lz4_dep],
  objcpp_args : ['-DRYCRAFT_SOURCE_ROOT="' + meson.project_source_root() + '"'],
)

# The deterministic worldgen, full-disk coverage, and hydrology matrices
# dominate the suite. Catch2 shards partition every test case across several
# meson test targets that run in parallel, so the wall-clock (and the CI
# runner minutes billed for it) drops to the slowest shard instead of the
# sum, with no loss of coverage. Each shard finishes well inside its timeout.
# Eight shards with a wide per-shard timeout: the merged suite carries both
# the gameplay and the lighting-weather matrices, and the slowest worldgen
# fixtures exceeded the old budget on loaded CI runners. Catch2 randomizes test
# order by default, so every shard must use the same seed or their partitions
# overlap and leave other cases uncovered.
test_shard_count = '8'
foreach shard : ['0', '1', '2', '3', '4', '5', '6', '7']
  test('rycraft tests ' + shard, test_exe,
       args : ['--order', 'rand', '--rng-seed', '764891',
               '--shard-count', test_shard_count, '--shard-index', shard],
       timeout : 1800)
endforeach
