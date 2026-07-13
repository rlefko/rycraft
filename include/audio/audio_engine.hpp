#pragma once

#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

// ---------------------------------------------------------------------------
// AudioEngine — Core Audio RemoteIO engine for rycraft.
//
// Responsibilities:
//   • Initialize RemoteIO AudioUnit at 44.1kHz stereo
//   • 512-sample buffer callback
//   • Simple gain mixer with max 16 concurrent voices
//   • Thread-safe voice allocation via mutex
// ---------------------------------------------------------------------------

struct AudioVoice {
    std::vector<float> samples;
    uint32_t sampleRate = 44100;
    float gain = 1.0f;
    uint32_t readPosition = 0;
    bool active = false;
    bool looping = false;
};

class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    // Initialize the audio unit and start playback.
    // Returns true on success, false on failure.
    bool initialize();

    // Stop playback and release audio resources.
    void stop();

    // Play a sound buffer. Returns voice index (0-15) or -1 if no slots available.
    // Looping voices repeat until stopVoice() is called (ambient beds).
    int32_t playSound(const std::vector<float>& buffer, uint32_t sampleRate, float gain,
                      bool looping = false);

    // Stop a specific voice by index.
    void stopVoice(int32_t voiceIndex);

    // Set master volume (0.0 = silent, 1.0 = full).
    void setMasterVolume(float gain);

    // Get current master volume.
    float getMasterVolume() const { return _masterVolume; }

    // Check if engine is initialized and running.
    bool isRunning() const { return _isRunning; }

    // Audio callback (public for C function bridge)
    void audioCallback(AudioBufferList* outputData);

private:
    static constexpr int MAX_VOICES = 16;
    static constexpr uint32_t BUFFER_SIZE = 512;
    static constexpr uint32_t SAMPLE_RATE = 44100;

    AudioVoice _voices[MAX_VOICES];
    // Read on the Core Audio render thread, written from the main thread —
    // atomic so the concurrent access is well-defined without the voice lock.
    std::atomic<float> _masterVolume{1.0f};
    std::atomic<bool> _isRunning{false};

    // AudioUnit
    AudioUnit _audioUnit = nullptr;
    AudioStreamBasicDescription _outputFormat;

    // Mutex for thread-safe voice allocation
    std::mutex _voiceMutex;

    // Find a free voice slot. Returns index or -1.
    int32_t allocateVoice();

    // Deallocate a voice slot.
    void deallocateVoice(int32_t voiceIndex);
};

// Audio callback wrapper (C function → C++ method bridge)
OSStatus audioRenderCallback(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags,
                             const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber,
                             UInt32 inNumberFrames, AudioBufferList* ioData);
