catch2_proj = subproject('catch2', default_options : ['tests=false'], required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

test_exe = executable('test_rycraft',
  'test_main.mm',
  include_directories : inc,
  link_with : [libcommon, libworld, librender, libentity, libengine, libaudio],
  dependencies : [catch2_with_main_dep, lz4_dep],
)

test('rycraft tests', test_exe)
