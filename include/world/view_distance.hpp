#pragma once

// One supported horizontal view-distance contract serves world streaming,
// persisted settings, menu choices, and the far-terrain horizon. Consumers
// may retain domain-specific aliases, but they must derive from these values.
inline constexpr int MIN_RENDER_DISTANCE_CHUNKS = 4;
inline constexpr int MAX_RENDER_DISTANCE_CHUNKS = 512;
inline constexpr int DEFAULT_RENDER_DISTANCE_CHUNKS = MAX_RENDER_DISTANCE_CHUNKS;
