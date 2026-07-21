#include "test_helpers.hpp"

#include <audio/audio_engine.hpp>
#include <audio/sfx.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/math.hpp>
#include <common/random.hpp>
#include <common/thread_pool.hpp>
#include <engine/game_state.hpp>
#include <engine/input_bindings.hpp>
#include <engine/inventory.hpp>
#include <entity/ai.hpp>
#include <entity/entity.hpp>
#include <entity/physics.hpp>
#include <entity/player.hpp>
#include <entity/spatial_hash.hpp>
#include <entity/spawner.hpp>
#include <entity/voxel_traversal.hpp>
#include <render/block_texture_array.hpp>
#include <render/block_textures.hpp>
#include <render/lod_mesher.hpp>
#include <render/mega_buffer.hpp>
#include <render/shader_types.hpp>
#include <render/ui_menu.hpp>
#include <render/ui_overlay.hpp>
#include <render/vertex.hpp>
#include <world/chunk.hpp>
#include <world/chunk_generator.hpp>
#include <world/chunk_pos.hpp>
#include <world/climate.hpp>
#include <world/noise.hpp>
#include <world/save_manager.hpp>
#include <world/serialization.hpp>
#include <world/world.hpp>

#include <chrono>
#include <cmath>
#include <thread>

// ============================================================================
// Vec3 Tests
// ============================================================================
// ===========================================================================
// Audio: procedural sound effects and the mixer
// ===========================================================================

// ---- Audio Engine Tests ----

TEST_CASE("Audio engine: voice allocation and deallocation", "[phase8][audio]") {
    // Simulate voice allocation logic
    static constexpr int MAX_VOICES = 16;
    std::vector<bool> active(MAX_VOICES, false);

    auto allocateVoice = [&active]() -> int32_t {
        for (int i = 0; i < MAX_VOICES; ++i) {
            if (!active[i]) {
                active[i] = true;
                return i;
            }
        }
        return -1;
    };

    auto deallocateVoice = [&active](int32_t idx) {
        if (idx >= 0 && idx < MAX_VOICES) {
            active[idx] = false;
        }
    };

    // Allocate first voice
    int v1 = allocateVoice();
    REQUIRE(v1 == 0);

    // Allocate second voice
    int v2 = allocateVoice();
    REQUIRE(v2 == 1);

    // Allocate all 16 voices
    for (int i = 2; i < MAX_VOICES; ++i) {
        int v = allocateVoice();
        REQUIRE(v == i);
    }

    // 17th allocation should fail
    int vFail = allocateVoice();
    REQUIRE(vFail == -1);

    // Deallocate one voice
    deallocateVoice(0);

    // Next allocation should reuse slot 0
    int vReuse = allocateVoice();
    REQUIRE(vReuse == 0);
}

TEST_CASE("Audio engine: mixer gain calculation", "[phase8][audio]") {
    // Simple gain mixing: output = sample * voiceGain * masterVolume
    auto mixSample = [](float sample, float voiceGain, float masterVolume) -> float {
        return sample * voiceGain * masterVolume;
    };

    // Full gain
    REQUIRE(mixSample(1.0f, 1.0f, 1.0f) == Catch::Approx(1.0f));

    // Half voice gain
    REQUIRE(mixSample(1.0f, 0.5f, 1.0f) == Catch::Approx(0.5f));

    // Half master volume
    REQUIRE(mixSample(1.0f, 1.0f, 0.5f) == Catch::Approx(0.5f));

    // Quarter total gain
    REQUIRE(mixSample(1.0f, 0.5f, 0.5f) == Catch::Approx(0.25f));

    // Silent master volume
    REQUIRE(mixSample(1.0f, 1.0f, 0.0f) == Catch::Approx(0.0f));
}

TEST_CASE("Audio engine: master volume clamping", "[phase8][audio]") {
    auto clampVolume = [](float gain) -> float {
        return gain < 0.0f ? 0.0f : (gain > 1.0f ? 1.0f : gain);
    };

    REQUIRE(clampVolume(-0.5f) == Catch::Approx(0.0f));
    REQUIRE(clampVolume(0.0f) == Catch::Approx(0.0f));
    REQUIRE(clampVolume(0.5f) == Catch::Approx(0.5f));
    REQUIRE(clampVolume(1.0f) == Catch::Approx(1.0f));
    REQUIRE(clampVolume(2.0f) == Catch::Approx(1.0f));
}

TEST_CASE("Audio engine: real playSound path mixes and drains through the callback",
          "[phase8][audio]") {
    // Exercise the ACTUAL AudioEngine mixer (not a reimplemented copy). No
    // initialize(), the render callback never touches the AudioUnit, so this
    // runs headlessly. This is the path whose one-sided locking used to trap.
    AudioEngine engine;

    std::vector<float> buf = {0.5f, 0.5f, 0.5f, 0.5f};
    int32_t voice = engine.playSound(buf, SoundEffect::SAMPLE_RATE, 1.0f);
    REQUIRE(voice >= 0);

    // A stereo output block: 4 frames × 2 channels × float
    constexpr uint32_t kFrames = 4;
    float out[kFrames * 2] = {0.f};
    AudioBufferList abl;
    abl.mNumberBuffers = 1;
    abl.mBuffers[0].mNumberChannels = 2;
    abl.mBuffers[0].mDataByteSize = sizeof(out);
    abl.mBuffers[0].mData = out;

    engine.audioCallback(&abl);

    // Each sample lands in both channels at full gain
    REQUIRE(out[0] == Catch::Approx(0.5f)); // frame 0 left
    REQUIRE(out[1] == Catch::Approx(0.5f)); // frame 0 right
    REQUIRE(out[6] == Catch::Approx(0.5f)); // frame 3 left

    // The voice was fully consumed (4 samples) → it deallocates, so a second
    // callback produces silence.
    float out2[kFrames * 2] = {0.f};
    abl.mBuffers[0].mData = out2;
    engine.audioCallback(&abl);
    REQUIRE(out2[0] == Catch::Approx(0.f));

    // Master volume scales the mix.
    engine.setMasterVolume(0.5f);
    REQUIRE(engine.getMasterVolume() == Catch::Approx(0.5f));
    int32_t voice2 = engine.playSound(buf, SoundEffect::SAMPLE_RATE, 1.0f);
    REQUIRE(voice2 >= 0);
    float out3[kFrames * 2] = {0.f};
    abl.mBuffers[0].mData = out3;
    engine.audioCallback(&abl);
    REQUIRE(out3[0] == Catch::Approx(0.25f)); // 0.5 sample × 0.5 master

    engine.setMasterVolume(1.0F);
    int32_t voice3 = engine.playSound(buf, SoundEffect::SAMPLE_RATE, 1.0F, true);
    REQUIRE(voice3 >= 0);
    engine.setVoiceGain(voice3, 0.2F);
    float out4[kFrames * 2] = {0.f};
    abl.mBuffers[0].mData = out4;
    engine.audioCallback(&abl);
    REQUIRE(out4[0] == Catch::Approx(0.1F));
}

TEST_CASE("Audio engine: looping voices wrap without gaps and preserve phase",
          "[phase8][audio][weather]") {
    AudioEngine engine;
    const std::vector<float> loop = {0.125F, 0.25F, 0.5F};
    REQUIRE(engine.playSound(loop, SoundEffect::SAMPLE_RATE, 1.0F, true) >= 0);

    constexpr uint32_t kFirstFrames = 8;
    float first[kFirstFrames * 2] = {0.0F};
    AudioBufferList output;
    output.mNumberBuffers = 1;
    output.mBuffers[0].mNumberChannels = 2;
    output.mBuffers[0].mDataByteSize = sizeof(first);
    output.mBuffers[0].mData = first;

    engine.audioCallback(&output);

    const float expectedFirst[kFirstFrames] = {0.125F, 0.25F, 0.5F,   0.125F,
                                               0.25F,  0.5F,  0.125F, 0.25F};
    for (uint32_t frame = 0; frame < kFirstFrames; ++frame) {
        REQUIRE(first[frame * 2] == Catch::Approx(expectedFirst[frame]));
        REQUIRE(first[frame * 2 + 1] == Catch::Approx(expectedFirst[frame]));
    }

    // Eight frames leave a three-sample loop at phase two. The next callback
    // must continue there, wrap again, and fill its entire output block.
    constexpr uint32_t kSecondFrames = 5;
    float second[kSecondFrames * 2] = {0.0F};
    output.mBuffers[0].mDataByteSize = sizeof(second);
    output.mBuffers[0].mData = second;
    engine.audioCallback(&output);

    const float expectedSecond[kSecondFrames] = {0.5F, 0.125F, 0.25F, 0.5F, 0.125F};
    for (uint32_t frame = 0; frame < kSecondFrames; ++frame) {
        REQUIRE(second[frame * 2] == Catch::Approx(expectedSecond[frame]));
        REQUIRE(second[frame * 2 + 1] == Catch::Approx(expectedSecond[frame]));
    }
}

// ---- SFX Tests ----

TEST_CASE("SFX: block break generates non-empty buffer", "[phase8][sfx]") {
    auto samples = SoundEffect::generateBlockBreak();
    REQUIRE(samples.empty() == false);

    // Expected duration: 0.1s at 44100 Hz = 4410 samples
    uint32_t expected = static_cast<uint32_t>(0.1f * SoundEffect::SAMPLE_RATE);
    REQUIRE(samples.size() == expected);
}

TEST_CASE("SFX: block place generates non-empty buffer", "[phase8][sfx]") {
    auto samples = SoundEffect::generateBlockPlace();
    REQUIRE(samples.empty() == false);

    // Expected duration: 0.08s at 44100 Hz = 3528 samples
    uint32_t expected = static_cast<uint32_t>(0.08f * SoundEffect::SAMPLE_RATE);
    REQUIRE(samples.size() == expected);
}

TEST_CASE("SFX: footstep generates non-empty buffer", "[phase8][sfx]") {
    auto samples = SoundEffect::generateFootstep();
    REQUIRE(samples.empty() == false);

    // Expected duration: 0.12s at 44100 Hz = 5292 samples
    uint32_t expected = static_cast<uint32_t>(0.12f * SoundEffect::SAMPLE_RATE);
    REQUIRE(samples.size() == expected);
}

TEST_CASE("SFX: ambient wind generates correct duration", "[phase8][sfx]") {
    auto samples = SoundEffect::generateAmbientWind(2);
    REQUIRE(samples.empty() == false);

    // Expected duration: 2s at 44100 Hz = 88200 samples
    uint32_t expected = 2 * SoundEffect::SAMPLE_RATE;
    REQUIRE(samples.size() == expected);
}

TEST_CASE("SFX: procedural precipitation beds are bounded deterministic loops",
          "[phase8][sfx][weather]") {
    const auto rain = SoundEffect::generateRainAmbience(1);
    const auto rainAgain = SoundEffect::generateRainAmbience(1);
    const auto snow = SoundEffect::generateSnowAmbience(1);
    REQUIRE(rain.size() == SoundEffect::SAMPLE_RATE);
    REQUIRE(snow.size() == SoundEffect::SAMPLE_RATE);
    REQUIRE(rain == rainAgain);
    for (float sample : rain) {
        REQUIRE(sample >= -1.0F);
        REQUIRE(sample <= 1.0F);
    }
    for (float sample : snow) {
        REQUIRE(sample >= -1.0F);
        REQUIRE(sample <= 1.0F);
    }
}

TEST_CASE("SFX: animal sounds generate non-empty buffers", "[phase8][sfx]") {
    auto sheep = SoundEffect::generateSheepBaa();
    auto cow = SoundEffect::generateCowMoo();
    auto pig = SoundEffect::generatePigOink();
    auto chicken = SoundEffect::generateChickenCluck();
    auto deer = SoundEffect::generateDeerCall();
    auto goat = SoundEffect::generateGoatBleat();
    auto rabbit = SoundEffect::generateRabbitChirp();
    auto frog = SoundEffect::generateFrogCroak();
    auto fish = SoundEffect::generateFishSplash();

    REQUIRE(sheep.empty() == false);
    REQUIRE(cow.empty() == false);
    REQUIRE(pig.empty() == false);
    REQUIRE(chicken.empty() == false);
    REQUIRE(deer.empty() == false);
    REQUIRE(goat.empty() == false);
    REQUIRE(rabbit.empty() == false);
    REQUIRE(frog.empty() == false);
    REQUIRE(fish.empty() == false);

    for (size_t index = 0; index < ENTITY_TYPE_COUNT; ++index) {
        REQUIRE(SoundEffect::generateAnimalCall(static_cast<EntityType>(index)).empty() == false);
    }
}

TEST_CASE("SFX: sample values within [-1, 1] range", "[phase8][sfx]") {
    auto samples = SoundEffect::generateBlockBreak();

    for (float sample : samples) {
        REQUIRE(sample >= -1.0f);
        REQUIRE(sample <= 1.0f);
    }
}

TEST_CASE("SFX: footstep samples are low-frequency (few zero crossings)", "[phase8][sfx]") {
    auto samples = SoundEffect::generateFootstep();

    // Count zero crossings
    int zeroCrossings = 0;
    for (size_t i = 1; i < samples.size(); ++i) {
        if ((samples[i - 1] >= 0.f && samples[i] < 0.f) ||
            (samples[i - 1] < 0.f && samples[i] >= 0.f)) {
            ++zeroCrossings;
        }
    }

    // At 80-120Hz for 0.12s: expect roughly 10-15 zero crossings
    // (2 crossings per cycle, 10-14 cycles in 0.12s at 80-120Hz)
    REQUIRE(zeroCrossings > 0);
    REQUIRE(zeroCrossings < 50); // Much less than high-frequency sounds
}

TEST_CASE("SFX: deterministic output for same parameters", "[phase8][sfx]") {
    auto samples1a = SoundEffect::generateBlockBreak();
    auto samples1b = SoundEffect::generateBlockBreak();

    REQUIRE(samples1a.size() == samples1b.size());
    for (size_t i = 0; i < samples1a.size(); ++i) {
        REQUIRE(samples1a[i] == samples1b[i]);
    }
}

TEST_CASE("SFX: survival and interface sounds are non-empty and in range", "[phase8][sfx]") {
    const std::vector<std::vector<float>> buffers = {
        SoundEffect::generateHurt(),   SoundEffect::generateEat(),
        SoundEffect::generateDeath(),  SoundEffect::generateClick(),
        SoundEffect::generatePickup(), SoundEffect::generateFurnacePop(),
    };
    for (const auto& samples : buffers) {
        REQUIRE_FALSE(samples.empty());
        for (float s : samples) {
            REQUIRE(s >= -1.0f);
            REQUIRE(s <= 1.0f);
        }
    }
}
