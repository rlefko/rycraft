# Metal shader compilation pipeline

# Compile .metal source files to .air intermediate representation
metal_air = custom_target(
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

# Compile .air to .metallib for runtime loading
metallib = custom_target(
  'pipeline.metallib',
  input : metal_air,
  output : 'pipeline.metallib',
  command : [
    'xcrun', 'metallib',
    '@INPUT@',
    '-o', '@OUTPUT@',
  ],
  build_by_default : true,
)

# metallib output is available at build/shaders/pipeline.metallib
# Copy to app bundle at runtime or during packaging
