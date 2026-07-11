catch2_proj = subproject('catch2', default_options : ['tests=false'], required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

# LZ4 dependency for tests
lz4_dep = dependency('lz4', required : false)
if not lz4_dep.found()
  lz4_proj = subproject('lz4')
  lz4_dep = lz4_proj.get_variable('liblz4_dep')
endif

test_exe = executable('test_rycraft',
  'test_main.cpp',
  include_directories : inc,
  link_with : [libcommon, libworld, librender],
  dependencies : [catch2_with_main_dep, lz4_dep],
)

test('rycraft tests', test_exe)
