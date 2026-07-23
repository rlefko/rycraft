#pragma once

#include <cstdint>

namespace worldgen::learned {

inline constexpr uint32_t GENERATOR_V4_VERSION = 4;

} // namespace worldgen::learned

namespace worldgen::v4_profile {

inline constexpr char WORLD_DIRECTORY[] = "rycraft_world_v4";
inline constexpr char REGIONS_DIRECTORY[] = "regions-v4";
inline constexpr char TERRAIN_AUTHORITY_DIRECTORY[] = "terrain-authority-v1";
inline constexpr char HYDROLOGY_AUTHORITY_DIRECTORY[] = "hydrology-authority-v1";

} // namespace worldgen::v4_profile
