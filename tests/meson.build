catch2_proj = subproject('catch2', default_options : ['tests=false'], required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

test_sources = files(
  'test_common.mm',
  'test_world.mm',
  'test_weather.mm',
  'test_worldgen_advanced.mm',
  'test_geology_ecology.mm',
  'test_organic_geology.mm',
  'test_natural_bank_substrate.mm',
  'test_macro_control.mm',
  'test_final_field_artifacts.mm',
  'test_reported_water_continuity.mm',
  'test_relief.mm',
  'test_alpine_morphology.mm',
  'test_tree_geometry.mm',
  'test_render.mm',
  'test_entity.mm',
  'test_fauna_habitat.mm',
  'test_fluid_persistence.mm',
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
)

# The deterministic worldgen, full-disk coverage, and hydrology matrices run
# for more than eight minutes on the reference machine. Keep CI's process
# timeout above the measured suite duration so a real assertion, rather than
# SIGTERM in a late water fixture, reports any failure.
test('rycraft tests', test_exe, timeout : 1200)
