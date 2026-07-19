#pragma once

// Shared by weather simulation and Metal weather-map consumers. Keep these as
// preprocessor constants so the same contract is available to C++ and Metal
// without making world simulation depend on render headers.
#define WEATHER_GRID_EDGE 81
#define WEATHER_GRID_CELL_SPACING_BLOCKS 256.0f
