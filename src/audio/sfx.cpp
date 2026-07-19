#include "audio/sfx.hpp"
#include "entity/entity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
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
// Precipitation beds: deterministic filtered noise loops. Rain keeps a broad
// high-frequency hiss with individual drop ticks; snow is a much softer,
// lower band carried mostly by wind. Matching the loop endpoints avoids a
// click when the audio voice wraps.
// ---------------------------------------------------------------------------
static void closeProceduralLoop(std::vector<float>& samples) {
    const size_t blendSamples =
        std::min<size_t>(samples.size() / 8U, SoundEffect::SAMPLE_RATE / 40U);
    if (blendSamples == 0) return;
    for (size_t index = 0; index < blendSamples; ++index) {
        const size_t tail = samples.size() - blendSamples + index;
        const float amount = static_cast<float>(index) / static_cast<float>(blendSamples);
        const float joined = samples[tail] * (1.0F - amount) + samples[index] * amount;
        samples[index] = joined;
        samples[tail] = joined;
    }
}

std::vector<float> SoundEffect::generateRainAmbience(uint32_t durationSeconds) {
    const uint32_t duration = durationSeconds * SAMPLE_RATE;
    std::vector<float> samples(duration);
    float bodyState = 0.0F;
    for (uint32_t index = 0; index < duration; ++index) {
        const float noise = randomNoise(0x5241494EU, index);
        const float body = noise - lowPassFilter(noise, &bodyState, 1'800.0F, SAMPLE_RATE);
        const float drop = randomNoise(0x44524F50U, index / 37U) > 0.985F
                               ? std::exp(-static_cast<float>(index % 37U) * 0.18F)
                               : 0.0F;
        samples[index] = std::clamp(body * 0.16F + drop * 0.08F, -1.0F, 1.0F);
    }
    closeProceduralLoop(samples);
    return samples;
}

std::vector<float> SoundEffect::generateSnowAmbience(uint32_t durationSeconds) {
    const uint32_t duration = durationSeconds * SAMPLE_RATE;
    std::vector<float> samples(duration);
    float softState = 0.0F;
    for (uint32_t index = 0; index < duration; ++index) {
        const float noise = randomNoise(0x534E4F57U, index / 3U);
        const float soft = lowPassFilter(noise, &softState, 420.0F, SAMPLE_RATE);
        const float drift =
            0.65F + 0.35F * std::sin(2.0F * static_cast<float>(M_PI) * 0.13F *
                                     static_cast<float>(index) / static_cast<float>(SAMPLE_RATE));
        samples[index] = std::clamp(soft * drift * 0.045F, -1.0F, 1.0F);
    }
    closeProceduralLoop(samples);
    return samples;
}

// ---------------------------------------------------------------------------
// Thunder: deterministic broadband crack followed by a rolling low rumble.
// Event identity controls the timing and resonance, so repeated weather
// queries produce exactly the same sound without shipping external assets.
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateThunder(uint64_t eventId, float intensity) {
    constexpr float DURATION_SECONDS = 4.0F;
    const uint32_t duration = static_cast<uint32_t>(DURATION_SECONDS * SAMPLE_RATE);
    std::vector<float> samples(duration);
    const uint32_t seed =
        static_cast<uint32_t>(eventId) ^ static_cast<uint32_t>(eventId >> 32U) ^ 0x7468756EU;
    const float strength = std::isfinite(intensity) ? std::clamp(intensity, 0.0F, 1.25F) : 0.0F;
    const float resonance = 34.0F + static_cast<float>((seed >> 8U) % 18U);
    const float rollRate = 1.6F + static_cast<float>((seed >> 16U) % 11U) * 0.07F;
    float lowState = 0.0F;
    float bodyState = 0.0F;
    const double radiansPerSample = 2.0 * M_PI / static_cast<double>(SAMPLE_RATE);
    const std::array<double, 4> phaseStep = {
        radiansPerSample * static_cast<double>(rollRate),
        radiansPerSample * static_cast<double>(rollRate * 0.57F),
        radiansPerSample * static_cast<double>(resonance),
        radiansPerSample * static_cast<double>(resonance * 1.47F),
    };
    std::array<double, 4> phaseSine{};
    std::array<double, 4> phaseCosine{1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> stepSine{};
    std::array<double, 4> stepCosine{};
    for (size_t oscillator = 0; oscillator < phaseStep.size(); ++oscillator) {
        stepSine[oscillator] = std::sin(phaseStep[oscillator]);
        stepCosine[oscillator] = std::cos(phaseStep[oscillator]);
    }

    for (uint32_t i = 0; i < duration; ++i) {
        const float time = static_cast<float>(i) / static_cast<float>(SAMPLE_RATE);
        const float noise = randomNoise(seed, i);
        const float lowNoise = lowPassFilter(noise, &lowState, 125.0F, SAMPLE_RATE);
        const float bodyNoise = lowPassFilter(noise, &bodyState, 720.0F, SAMPLE_RATE);

        const float crackEnvelope = std::exp(-time * 31.0F) * std::min(time * 190.0F, 1.0F);
        const float bodyEnvelope = std::exp(-time * 0.72F) * std::min(time * 22.0F, 1.0F);
        std::array<float, 4> oscillatorSample{};
        for (size_t oscillator = 0; oscillator < phaseStep.size(); ++oscillator) {
            oscillatorSample[oscillator] = static_cast<float>(phaseSine[oscillator]);
            const double nextSine = phaseSine[oscillator] * stepCosine[oscillator] +
                                    phaseCosine[oscillator] * stepSine[oscillator];
            phaseCosine[oscillator] = phaseCosine[oscillator] * stepCosine[oscillator] -
                                      phaseSine[oscillator] * stepSine[oscillator];
            phaseSine[oscillator] = nextSine;
        }
        const float rolling = 0.48F + 0.32F * std::max(0.0F, oscillatorSample[0]) +
                              0.20F * std::max(0.0F, oscillatorSample[1]);
        const float resonanceTone = oscillatorSample[2] * 0.18F + oscillatorSample[3] * 0.08F;
        const float tailFade = std::clamp((DURATION_SECONDS - time) / 0.35F, 0.0F, 1.0F);
        const float raw =
            bodyNoise * crackEnvelope * 1.35F +
            (lowNoise * 1.75F + bodyNoise * 0.22F + resonanceTone) * bodyEnvelope * rolling;
        samples[i] = std::tanh(raw * strength * 1.45F) * tailFade;
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

// ---------------------------------------------------------------------------
// Deer call: a soft rising whistle
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateDeerCall() {
    uint32_t duration = static_cast<uint32_t>(0.35f * SAMPLE_RATE);
    std::vector<float> samples(duration);
    for (uint32_t i = 0; i < duration; ++i) {
        float tone = frequencySweep(i, SAMPLE_RATE, 180.0f, 260.0f, duration);
        tone += sinOscillator(i, SAMPLE_RATE, 360.0f) * 0.2f;
        float envelope = adsrEnvelope(i, duration, 0.04f, 0.08f, 0.55f, 0.12f);
        samples[i] = tone * envelope * 0.38f;
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Goat bleat: a nasal, rapidly modulated tone
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateGoatBleat() {
    uint32_t duration = static_cast<uint32_t>(0.3f * SAMPLE_RATE);
    std::vector<float> samples(duration);
    for (uint32_t i = 0; i < duration; ++i) {
        float vibrato = 32.0f * sinOscillator(i, SAMPLE_RATE, 8.0f);
        float tone = sinOscillator(i, SAMPLE_RATE, 310.0f + vibrato);
        tone += sinOscillator(i, SAMPLE_RATE, 620.0f + vibrato) * 0.22f;
        float envelope = adsrEnvelope(i, duration, 0.015f, 0.05f, 0.6f, 0.1f);
        samples[i] = tone * envelope * 0.4f;
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Rabbit chirp: a brief high-pitched squeak
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateRabbitChirp() {
    uint32_t duration = static_cast<uint32_t>(0.08f * SAMPLE_RATE);
    std::vector<float> samples(duration);
    for (uint32_t i = 0; i < duration; ++i) {
        float tone = frequencySweep(i, SAMPLE_RATE, 1050.0f, 720.0f, duration);
        float envelope = adsrEnvelope(i, duration, 0.004f, 0.015f, 0.25f, 0.04f);
        samples[i] = tone * envelope * 0.3f;
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Frog croak: low pulses with a resonant harmonic
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateFrogCroak() {
    uint32_t duration = static_cast<uint32_t>(0.45f * SAMPLE_RATE);
    std::vector<float> samples(duration);
    float filterState = 0.0f;
    for (uint32_t i = 0; i < duration; ++i) {
        float pulse = 0.35f + 0.65f * std::max(0.0f, sinOscillator(i, SAMPLE_RATE, 13.0f));
        float raw = sinOscillator(i, SAMPLE_RATE, 92.0f);
        raw += sinOscillator(i, SAMPLE_RATE, 184.0f) * 0.35f;
        raw += randomNoise(0xF206u, i / 3) * 0.08f;
        raw = lowPassFilter(raw, &filterState, 420.0f, SAMPLE_RATE);
        float envelope = adsrEnvelope(i, duration, 0.025f, 0.06f, 0.65f, 0.14f);
        samples[i] = raw * pulse * envelope * 0.42f;
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Fish splash: a short filtered water-noise burst
// ---------------------------------------------------------------------------
std::vector<float> SoundEffect::generateFishSplash() {
    uint32_t duration = static_cast<uint32_t>(0.12f * SAMPLE_RATE);
    std::vector<float> samples(duration);
    float filterState = 0.0f;
    for (uint32_t i = 0; i < duration; ++i) {
        float noise = randomNoise(0xF157u, i);
        float bubble = frequencySweep(i, SAMPLE_RATE, 520.0f, 180.0f, duration) * 0.25f;
        float raw = lowPassFilter(noise * 0.65f + bubble, &filterState, 900.0f, SAMPLE_RATE);
        float envelope = adsrEnvelope(i, duration, 0.003f, 0.025f, 0.12f, 0.07f);
        samples[i] = raw * envelope * 0.38f;
    }
    return samples;
}

std::vector<float> SoundEffect::generateAnimalCall(EntityType type) {
    switch (type) {
        case EntityType::SHEEP:
            return generateSheepBaa();
        case EntityType::COW:
            return generateCowMoo();
        case EntityType::PIG:
            return generatePigOink();
        case EntityType::CHICKEN:
            return generateChickenCluck();
        case EntityType::DEER:
            return generateDeerCall();
        case EntityType::GOAT:
            return generateGoatBleat();
        case EntityType::RABBIT:
            return generateRabbitChirp();
        case EntityType::FROG:
            return generateFrogCroak();
        case EntityType::FISH:
            return generateFishSplash();
        case EntityType::COUNT:
            std::unreachable();
    }
}
