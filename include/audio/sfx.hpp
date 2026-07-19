#pragma once

#include <cstdint>
#include <vector>

enum class EntityType : uint8_t;

// ---------------------------------------------------------------------------
// SoundEffect, Procedural sound effect generation.
//
// Generates PCM audio samples at 44100 Hz for various game events.
// All generation functions are pure: same parameters → same output.
//
// Sound types:
//   • Block break: noise burst with frequency sweep (200→800ms, 0.1s)
//   • Block place: noise burst with frequency sweep (800→200ms, 0.08s)
//   • Footstep: low-frequency thud (80-120Hz, 0.12s)
//   • Ambient wind: filtered noise, continuous loop
//   • Thunder: seeded broadband crack and rolling low-frequency tail
//   • Animal: simple tone sequences per entity type
// ---------------------------------------------------------------------------
class SoundEffect {
public:
    static constexpr uint32_t SAMPLE_RATE = 44100;

    // Block interaction sounds
    static std::vector<float> generateBlockBreak();
    static std::vector<float> generateBlockPlace();

    // Movement sounds
    static std::vector<float> generateFootstep();

    // Environmental sounds
    static std::vector<float> generateAmbientWind(uint32_t durationSeconds = 4);
    static std::vector<float> generateRainAmbience(uint32_t durationSeconds = 4);
    static std::vector<float> generateSnowAmbience(uint32_t durationSeconds = 4);
    static std::vector<float> generateThunder(uint64_t eventId, float intensity = 1.0F);

    // Entity sounds (tone sequences)
    static std::vector<float> generateSheepBaa();
    static std::vector<float> generateCowMoo();
    static std::vector<float> generatePigOink();
    static std::vector<float> generateChickenCluck();
    static std::vector<float> generateDeerCall();
    static std::vector<float> generateGoatBleat();
    static std::vector<float> generateRabbitChirp();
    static std::vector<float> generateFrogCroak();
    static std::vector<float> generateFishSplash();

    // Exhaustive dispatch used by the engine's EntityType-indexed table.
    static std::vector<float> generateAnimalCall(EntityType type);

    // Internal helpers
private:
    // Pure pseudo-random number generator (deterministic)
    static float randomNoise(uint32_t seed, uint32_t index);

    // Low-frequency oscillator
    static float sinOscillator(uint32_t index, uint32_t sampleRate, float frequency);

    // Frequency sweep (linear interpolation)
    static float frequencySweep(uint32_t index, uint32_t sampleRate, float startFreq, float endFreq,
                                uint32_t totalSamples);

    // Envelope: attack-decay-sustain-release
    static float adsrEnvelope(uint32_t index, uint32_t totalSamples, float attackTime,
                              float decayTime, float sustainLevel, float releaseTime);

    // Low-pass filter (simple one-pole)
    static float lowPassFilter(float sample, float* state, float cutoff, uint32_t sampleRate);
};
