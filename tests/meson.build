catch2_proj = subproject('catch2', default_options : ['tests=false'], required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

test_sources = files(
  'test_common.mm',
  'test_world.mm',
  'test_worldgen_advanced.mm',
  'test_geology_ecology.mm',
  'test_relief.mm',
  'test_render.mm',
  'test_entity.mm',
  'test_fauna_habitat.mm',
  'test_fluid_persistence.mm',
  'test_render_concurrency.mm',
  'test_engine.mm',
  'test_audio.mm',
)

test_exe = executable('test_rycraft',
  test_sources,
  include_directories : inc,
  link_with : [libcommon, libworld, librender, libentity, libengine, libaudio],
  dependencies : [catch2_with_main_dep, lz4_dep],
)

test('rycraft tests', test_exe, timeout : 180)
