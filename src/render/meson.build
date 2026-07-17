render_sources = files('lod_mesher.cpp', 'far_terrain.cpp', 'mesh_scheduler.cpp', 'block_texture_array.mm',
                       'mega_buffer.mm', 'render_pipeline.mm', 'ui_overlay.mm', 'ui_menu.cpp',
                       'ui_hud.mm', 'ui_item_icon.mm', 'entity_renderer.mm', 'item_entity_renderer.mm', 'bloom.mm', 'particles.mm',
                       'gpu_timer.mm', 'graphics_settings.cpp', 'post_stack.mm',
                       'shadow_map.mm', 'volumetrics.mm', 'ssao.mm')

    librender = static_library('render', render_sources, include_directories : inc,
                               dependencies : [appleframeworks], )
