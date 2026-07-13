#include "audio/sfx.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Pure pseudo-random number generator (deterministic, no state)
// Uses a simple LCG for reproducibility
// ---------------------------------------------------------------------------
float SoundEffect::randomNoise(uint32_t seed, uint32_t index) {
    uint32_t value = seed + index * 1664525u + 1013904223u;
    // Mix bits
    value ^= value >> 16;
    value *= 0x85ebca6bu;
    value ^= value >> 13;
    value *= 0xc2b2ae35u;
    value ^= value >> 16;
    // Convert to [-1, 1]
    return (static_cast<float>(value & 0x7FFFFFFFu) / static_cast<float>(0x7FFFFFFFu)) * 2.0f -
           1.0f;
}

// ---------------------------------------------------------------------------
// Low-frequency oscillator (pure sine)
// ---------------------------------------------------------------------------
float SoundEffect::sinOscillator(uint32_t index, uint32_t sampleRate, float frequency) {
    return std::sin(2.0f * static_cast<float>(M_PI) * frequency * static_cast<float>(index) /
                    static_cast<float>(sampleRate));
}

// ---------------------------------------------------------------------------
// Frequency sweep (linear interpolation between start and end frequency)
// ---------------------------------------------------------------------------
float SoundEffect::frequencySweep(uint32_t index, uint32_t sampleRate, float startFreq,
                                  float endFreq, uint32_t totalSamples) {
    float t = static_cast<float>(index) / static_cast<float>(totalSamples);
    float frequency = startFreq + (endFreq - startFreq) * t;
    return sinOscillator(index, sampleRate, frequency);
}

// ---------------------------------------------------------------------------
// ADSR envelope
// ---------------------------------------------------------------------------
float SoundEffect::adsrEnvelope(uint32_t index, uint32_t totalSamples, float attackTime,
                                float decayTime, float sustainLevel, float releaseTime) {
    float t = static_cast<float>(index) / static_cast<float>(totalSamples);
    float attackSamples = attackTime * SAMPLE_RATE / static_cast<float>(totalSamples);
    float decaySamples = decayTime * SAMPLE_RATE / static_cast<float>(totalSamples);
    float releaseSamples = releaseTime * SAMPLE_RATE / static_cast<float>(totalSamples);

    float attackEnd = attackSamples;
    float decayEnd = attackEnd + decaySamples;
    float releaseStart = 1.0f - releaseSamples;

    if (t < attackEnd) {
        // Attack phase: 0 → 1
        return t / attackEnd;
    } else if (t < decayEnd) {
        // Decay phase: 1 → sustainLevel
        float decayT = (t - attackEnd) / decaySamples;
        return 1.0f - (1.0f - sustainLevel) * decayT;
    } else if (t < releaseStart) {
        // Sustain phase
        return sustainLevel;
    } else {
        // Release phase: sustainLevel → 0
        float releaseT = (t - releaseStart) / releaseSamples;
        return sustainLevel * (1.0f - releaseT);
    }
}

// ---------------------------------------------------------------------------
// Low-pass filter (one-pole)
// ---------------------------------------------------------------------------
float SoundEffect::lowPassFilter(float sample, float* state, float cutoff, uint32_t sampleRate) {
    float rc = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff);
    float dt = 1.0f / static_cast<float>(sampleRate);
    float alpha = dt / (rc + dt);
    *state = *state + alpha * (sample - *state);
    return *state;
}

// ============================================================================
// Sound Effect Generators
// ============================================================================

// ---------------------------------------------------------------------------
// Block break: noise burst with frequency sweep (200→800Hz, 0.1s)
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateBlockBreak() {
    uint32_t duration = static_cast<uint32_t>(0.1f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    float filterState = 0.0f;

    for (uint32_t i = 0; i < duration; ++i) {
        // Noise burst mixed with frequency sweep
        float noise = randomNoise(42, i) * 0.6f;
        float sweep = frequencySweep(i, SAMPLE_RATE, 200.0f, 800.0f, duration) * 0.4f;
        float raw = noise + sweep;

        // Apply low-pass filter for crunchier sound
        raw = lowPassFilter(raw, &filterState, 1200.0f, SAMPLE_RATE);

        // Envelope: quick attack, fast decay
        float envelope = adsrEnvelope(i, duration, 0.005f, 0.05f, 0.0f, 0.045f);

        samples[i] = raw * envelope;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Block place: noise burst with frequency sweep (800→200Hz, 0.08s)
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateBlockPlace() {
    uint32_t duration = static_cast<uint32_t>(0.08f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    float filterState = 0.0f;

    for (uint32_t i = 0; i < duration; ++i) {
        // Reverse frequency sweep (descending)
        float noise = randomNoise(137, i) * 0.5f;
        float sweep = frequencySweep(i, SAMPLE_RATE, 800.0f, 200.0f, duration) * 0.5f;
        float raw = noise + sweep;

        // Low-pass filter
        raw = lowPassFilter(raw, &filterState, 1000.0f, SAMPLE_RATE);

        // Envelope: very quick attack, medium decay
        float envelope = adsrEnvelope(i, duration, 0.003f, 0.04f, 0.0f, 0.037f);

        samples[i] = raw * envelope;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Footstep: low-frequency thud (80-120Hz, 0.12s)
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateFootstep() {
    uint32_t duration = static_cast<uint32_t>(0.12f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    float filterState = 0.0f;

    for (uint32_t i = 0; i < duration; ++i) {
        // Low-frequency thud with slight frequency modulation
        float baseFreq =
            80.0f + 40.0f * std::exp(-static_cast<float>(i) / static_cast<float>(duration * 0.3f));
        float tone = sinOscillator(i, SAMPLE_RATE, baseFreq);

        // Add noise component for texture
        float noise = randomNoise(256, i) * 0.15f;

        float raw = tone * 0.8f + noise;
        raw = lowPassFilter(raw, &filterState, 300.0f, SAMPLE_RATE);

        // Envelope: medium attack, long decay
        float envelope = adsrEnvelope(i, duration, 0.01f, 0.04f, 0.2f, 0.07f);

        samples[i] = raw * envelope * 0.7f;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Ambient wind: filtered noise, continuous loop
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateAmbientWind(uint32_t durationSeconds) {
    uint32_t duration = durationSeconds * SAMPLE_RATE;
    std::vector<float> samples(duration);

    float filterState = 0.0f;

    for (uint32_t i = 0; i < duration; ++i) {
        // Slowly varying noise
        float noise = randomNoise(512, i / 4); // Lower effective rate for wind texture

        // Low-pass filter for wind character
        float cutoff =
            200.0f + 100.0f * std::sin(2.0f * static_cast<float>(M_PI) * 0.1f *
                                       static_cast<float>(i) / static_cast<float>(SAMPLE_RATE));
        noise = lowPassFilter(noise, &filterState, cutoff, SAMPLE_RATE);

        // Gentle volume modulation
        float modulation =
            0.3f + 0.1f * std::sin(2.0f * static_cast<float>(M_PI) * 0.05f * static_cast<float>(i) /
                                   static_cast<float>(SAMPLE_RATE));

        samples[i] = noise * modulation * 0.15f; // Low volume for ambient
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Sheep baa: short melodic chirp
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateSheepBaa() {
    uint32_t duration = static_cast<uint32_t>(0.3f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    for (uint32_t i = 0; i < duration; ++i) {
        // Two-tone "baa" sound
        float t = static_cast<float>(i) / static_cast<float>(duration);
        float freq = t < 0.5f ? 300.0f : 250.0f;
        float tone = sinOscillator(i, SAMPLE_RATE, freq);

        // Slight vibrato
        tone *= 1.0f + 0.05f * std::sin(2.0f * static_cast<float>(M_PI) * 5.0f *
                                        static_cast<float>(i) / static_cast<float>(SAMPLE_RATE));

        float envelope = adsrEnvelope(i, duration, 0.02f, 0.05f, 0.6f, 0.1f);

        samples[i] = tone * envelope * 0.5f;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Cow moo: low rumbling sound
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateCowMoo() {
    uint32_t duration = static_cast<uint32_t>(0.5f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    float filterState = 0.0f;

    for (uint32_t i = 0; i < duration; ++i) {
        // Low frequency with harmonics
        float fundamental = sinOscillator(i, SAMPLE_RATE, 80.0f);
        float harmonic = sinOscillator(i, SAMPLE_RATE, 160.0f) * 0.3f;
        float raw = fundamental + harmonic;

        // Add slight rumble
        raw += randomNoise(789, i / 8) * 0.1f;

        raw = lowPassFilter(raw, &filterState, 400.0f, SAMPLE_RATE);

        // Long envelope
        float envelope = adsrEnvelope(i, duration, 0.05f, 0.1f, 0.7f, 0.15f);

        samples[i] = raw * envelope * 0.5f;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Pig oink: short squeaky sound
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generatePigOink() {
    uint32_t duration = static_cast<uint32_t>(0.15f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    for (uint32_t i = 0; i < duration; ++i) {
        // Frequency sweep upward then down
        float t = static_cast<float>(i) / static_cast<float>(duration);
        float freq = 400.0f + 300.0f * std::sin(t * static_cast<float>(M_PI));
        float tone = sinOscillator(i, SAMPLE_RATE, freq);

        // Add slight noise
        tone += randomNoise(321, i) * 0.1f;

        float envelope = adsrEnvelope(i, duration, 0.01f, 0.03f, 0.4f, 0.04f);

        samples[i] = tone * envelope * 0.5f;
    }

    return samples;
}

// ---------------------------------------------------------------------------
// Chicken cluck: rapid short chirp
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateChickenCluck() {
    uint32_t duration = static_cast<uint32_t>(0.1f * SAMPLE_RATE);
    std::vector<float> samples(duration);

    for (uint32_t i = 0; i < duration; ++i) {
        // High frequency chirp
        float freq =
            800.0f + 200.0f * std::sin(2.0f * static_cast<float>(M_PI) * 15.0f *
                                       static_cast<float>(i) / static_cast<float>(SAMPLE_RATE));
        float tone = sinOscillator(i, SAMPLE_RATE, freq);

        // Rapid amplitude modulation for "cluck" character
        float am = 0.5f + 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 30.0f *
                                          static_cast<float>(i) / static_cast<float>(SAMPLE_RATE));

        float envelope = adsrEnvelope(i, duration, 0.005f, 0.02f, 0.0f, 0.05f);

        samples[i] = tone * am * envelope * 0.4f;
    }

    return samples;
}
