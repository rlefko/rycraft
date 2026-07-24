subdir('common')
subdir('engine')
subdir('render')
subdir('world')
subdir('entity')
subdir('audio')

# Copy compiled metallib next to the executable so it's discoverable at runtime
copy_metallib = custom_target(
  'copy_metallib',
  input : metallib,
  output : 'pipeline.metallib',
  command : ['cp', '@INPUT@', '@OUTPUT@'],
  build_by_default : true,
)

rycraft_exe = executable('rycraft',
  'engine/main.mm',
  include_directories : inc,
  link_with : [
    libcommon,
    libengine,
    librender,
    libworld,
    libentity,
    libaudio,
  ],
  dependencies : [appleframeworks],
  # A targeted `ninja src/rycraft` must refresh the runtime library beside
  # the executable, not only relink C++. Otherwise a newly added Metal entry
  # point can compile successfully while the launched binary sees a stale
  # metallib and aborts during renderer construction.
  link_depends : copy_metallib,
  build_by_default : true,
)

worldgen_inspect_exe = executable('rycraft_worldgen_inspect',
  'worldgen_inspect.cpp',
  include_directories : inc,
  link_with : [libcommon, libengine, libworld, librender],
  dependencies : [appleframeworks, lz4_dep],
  build_by_default : true,
)

# Program 0 critical-path trace summarizer. It only parses a binary trace, so
# it links libcommon alone and shares the record layout in common/trace.hpp.
trace_summary_exe = executable('rycraft_trace_summary',
  'trace_summary.cpp',
  include_directories : inc,
  link_with : [libcommon],
  dependencies : [appleframeworks],
  build_by_default : true,
)
