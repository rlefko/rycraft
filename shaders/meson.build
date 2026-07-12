# Metal shader compilation pipeline

# Compile pipeline.metal to .air intermediate representation
pipeline_air = custom_target(
  'pipeline.air',
  input : 'pipeline.metal',
  output : 'pipeline.air',
  command : [
    'xcrun', 'metal',
    '-c',
    '-mmacos-version-min=13.0',
    '-std=metal3.0',
    '-o', '@OUTPUT@',
    '@INPUT@',
  ],
)

# Compile ui_overlay.metal to .air intermediate representation
ui_overlay_air = custom_target(
  'ui_overlay.air',
  input : 'ui_overlay.metal',
  output : 'ui_overlay.air',
  command : [
    'xcrun', 'metal',
    '-c',
    '-mmacos-version-min=13.0',
    '-std=metal3.0',
    '-o', '@OUTPUT@',
    '@INPUT@',
  ],
)

# Compile sky.metal to .air intermediate representation
sky_air = custom_target(
  'sky.air',
  input : 'sky.metal',
  output : 'sky.air',
  command : [
    'xcrun', 'metal',
    '-c',
    '-mmacos-version-min=13.0',
    '-std=metal3.0',
    '-o', '@OUTPUT@',
    '@INPUT@',
  ],
)

# Combine all .air files into a single .metallib for runtime loading
metallib = custom_target(
  'pipeline.metallib',
  input : [pipeline_air, ui_overlay_air, sky_air],
  output : 'pipeline.metallib',
  command : [
    'xcrun', 'metallib',
    '@INPUT0@', '@INPUT1@', '@INPUT2@',
    '-o', '@OUTPUT@',
  ],
  build_by_default : true,
)

# metallib output is available at build/shaders/pipeline.metallib
# Copy to app bundle at runtime or during packaging
