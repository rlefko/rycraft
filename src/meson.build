subdir('common')
subdir('engine')
subdir('render')
subdir('world')
subdir('entity')
subdir('audio')

# Copy compiled metallib next to the executable so it's discoverable at runtime
copy_metallib = run_command(
  'cp', metallib.full_path(), meson.current_build_dir() / 'pipeline.metallib',
  check : true,
)

rycraft_exe = executable('rycraft',
  'engine/main.cpp',
  link_with : [
    libcommon,
    libengine,
    librender,
    libworld,
    libentity,
    libaudio,
  ],
  dependencies : [appleframeworks],
)
