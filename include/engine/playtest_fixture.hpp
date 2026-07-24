#pragma once

#include "world/block_properties.hpp"

#include <array>
#include <string_view>

inline constexpr std::array<BlockType, 5> MATERIAL_PLAYTEST_BLOCKS = {
    BlockType::BED, BlockType::CHEST, BlockType::TORCH, BlockType::FURNACE, BlockType::FURNACE_LIT,
};

constexpr bool enabledEnvironmentValue(const char* value) {
    return value != nullptr && std::string_view(value) != "" && std::string_view(value) != "0";
}

// The fixture is deliberately capture-only. RYCRAFT_CAPTURE already selects
// the no-save teardown path, so an opt-in material lineup can never reach a
// normal profile's durable chunk or block-entity files.
constexpr bool materialPlaytestFixtureEnabled(const char* requested, const char* capturePath) {
    return enabledEnvironmentValue(requested) && enabledEnvironmentValue(capturePath);
}
