catch2_proj = subproject('catch2', required : false)

assert(catch2_proj.found(), 'Catch2 subproject failed to initialize')

catch2_with_main_dep = catch2_proj.get_variable('catch2_with_main_dep')

test_exe = executable('test_rycraft',
  'test_main.cpp',
  include_directories : inc,
  dependencies : [catch2_with_main_dep],
)

test('rycraft tests', test_exe)
