project('rycraft', 'cpp', 'objcpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++23',
    'warning_level=3',
    'werror=true',
  ],
)

cpp = meson.get_compiler('cpp')

# Central include directory (shared across all subdirs)
inc = include_directories('include')

# Apple frameworks
appleframeworks = dependency('appleframeworks',
  modules : [
    'Metal',
    'MetalKit',
    'Cocoa',
    'CoreAudio',
    'AudioToolbox',
    'QuartzCore',
    'CoreVideo',
    'ImageIO',
    'UniformTypeIdentifiers',
  ],
)

# LZ4 (world save compression): system copy if present, wrap subproject otherwise
lz4_dep = dependency('liblz4',
  fallback : ['lz4', 'liblz4_dep'],
  default_options : ['default_library=static', 'ossfuzz=false'],
)

# Xcode 15+ ld warns about the duplicate -lc++ that mixed cpp/objcpp linking produces
add_project_link_arguments('-Wl,-no_warn_duplicate_libraries', language : ['cpp', 'objcpp'])

# Shader compilation pipeline
subdir('shaders')

# Source modules
subdir('src')

# Tests
subdir('tests')
