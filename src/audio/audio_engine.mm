#import "audio/audio_engine.hpp"

#include "common/error.hpp"

#import <AudioToolbox/AudioComponent.h>

#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Audio callback C function → C++ method bridge
// ---------------------------------------------------------------------------
OSStatus audioRenderCallback(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags,
                             const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber,
                             UInt32 /*inNumberFrames*/, AudioBufferList* ioData) {
    (void)ioActionFlags;
    (void)inTimeStamp;
    (void)inBusNumber;

    AudioEngine* engine = static_cast<AudioEngine*>(inRefCon);
    if (!engine)
        return -1;

    engine->audioCallback(ioData);
    return noErr;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
AudioEngine::AudioEngine() : _audioUnit(nullptr), _outputFormat{} {
    std::memset(_voices, 0, sizeof(_voices));
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
AudioEngine::~AudioEngine() {
    stop();
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------
bool AudioEngine::initialize() {
    if (_isRunning)
        return true;

    // Describe the RemoteIO AudioUnit
    AudioComponentDescription desc{};
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_DefaultOutput;
    desc.componentFlags = 0;
    desc.componentFlagsMask = 0;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;

    AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
    if (!comp) {
        RY_LOG_ERROR("Failed to find RemoteIO AudioComponent");
        return false;
    }

    OSStatus status = AudioComponentInstanceNew(comp, &_audioUnit);
    if (status != noErr) {
        RY_LOG_ERROR("Failed to create AudioUnit instance");
        return false;
    }

    // No EnableIO needed: kAudioUnitSubType_DefaultOutput has output-only
    // IO fixed on (EnableIO applies to AUHAL, and setting it here fails).

    // Set output format: 44.1kHz, stereo, 32-bit float
    std::memset(&_outputFormat, 0, sizeof(_outputFormat));
    _outputFormat.mSampleRate = SAMPLE_RATE;
    _outputFormat.mFormatID = kAudioFormatLinearPCM;
    _outputFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    _outputFormat.mFramesPerPacket = 1;
    _outputFormat.mChannelsPerFrame = 2;
    _outputFormat.mBytesPerFrame = 8; // 2 channels × 4 bytes
    _outputFormat.mBytesPerPacket = 8;
    _outputFormat.mBitsPerChannel = 32;

    // The format the render callback SUPPLIES: input scope of output bus 0
    status = AudioUnitSetProperty(_audioUnit, kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input, 0, &_outputFormat, sizeof(_outputFormat));
    if (status != noErr) {
        RY_LOG_ERROR("Failed to set audio output format");
        return false;
    }

    // Install render callback
    AURenderCallbackStruct callback{};
    callback.inputProc = audioRenderCallback;
    callback.inputProcRefCon = this;

    status = AudioUnitSetProperty(_audioUnit, kAudioUnitProperty_SetRenderCallback,
                                  kAudioUnitScope_Global, 0, &callback, sizeof(callback));
    if (status != noErr) {
        RY_LOG_ERROR("Failed to set audio render callback");
        return false;
    }

    // Initialize and start
    status = AudioUnitInitialize(_audioUnit);
    if (status != noErr) {
        RY_LOG_ERROR("Failed to initialize AudioUnit");
        return false;
    }

    status = AudioOutputUnitStart(_audioUnit);
    if (status != noErr) {
        RY_LOG_ERROR("Failed to start AudioUnit");
        return false;
    }

    _isRunning = true;
    return true;
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------
void AudioEngine::stop() {
    if (!_isRunning || !_audioUnit)
        return;

    OSStatus status = AudioOutputUnitStop(_audioUnit);
    if (status != noErr) {
        RY_LOG_ERROR("Failed to stop AudioUnit");
    }

    _isRunning = false;

    // Deactivate and dispose
    AudioUnitUninitialize(_audioUnit);
    AudioComponentInstanceDispose(_audioUnit);
    _audioUnit = nullptr;
}

// ---------------------------------------------------------------------------
// allocateVoice
// ---------------------------------------------------------------------------
int32_t AudioEngine::allocateVoice() {
    for (int i = 0; i < MAX_VOICES; ++i) {
        if (!_voices[i].active) {
            return i;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// deallocateVoice
// ---------------------------------------------------------------------------
void AudioEngine::deallocateVoice(int32_t voiceIndex) {
    if (voiceIndex < 0 || voiceIndex >= MAX_VOICES)
        return;

    _voices[voiceIndex].active = false;
    _voices[voiceIndex].samples.clear();
    _voices[voiceIndex].readPosition = 0;
    _voices[voiceIndex].gain = 1.0f;
    _voices[voiceIndex].looping = false;
}

// ---------------------------------------------------------------------------
// playSound
// ---------------------------------------------------------------------------
int32_t AudioEngine::playSound(const std::vector<float>& buffer, uint32_t sampleRate, float gain,
                               bool looping) {
    if (buffer.empty())
        return -1;

    std::lock_guard<std::mutex> lock(_voiceMutex);

    int32_t voiceIndex = allocateVoice();
    if (voiceIndex < 0)
        return -1;

    _voices[voiceIndex].samples = buffer;
    _voices[voiceIndex].sampleRate = sampleRate;
    _voices[voiceIndex].gain = gain;
    _voices[voiceIndex].readPosition = 0;
    _voices[voiceIndex].active = true;
    _voices[voiceIndex].looping = looping;

    return voiceIndex;
}

// ---------------------------------------------------------------------------
// stopVoice
// ---------------------------------------------------------------------------
void AudioEngine::stopVoice(int32_t voiceIndex) {
    if (voiceIndex < 0 || voiceIndex >= MAX_VOICES)
        return;

    std::lock_guard<std::mutex> lock(_voiceMutex);
    deallocateVoice(voiceIndex);
}

// ---------------------------------------------------------------------------
// setMasterVolume
// ---------------------------------------------------------------------------
void AudioEngine::setMasterVolume(float gain) {
    // Clamp to [0, 1]
    _masterVolume = gain < 0.0f ? 0.0f : (gain > 1.0f ? 1.0f : gain);
}

// ---------------------------------------------------------------------------
// audioCallback — called from audio thread
// ---------------------------------------------------------------------------
void AudioEngine::audioCallback(AudioBufferList* outputData) {
    if (!outputData)
        return;

    // Get output buffer
    float* outputBuffer = static_cast<float*>(outputData->mBuffers[0].mData);
    uint32_t frameCount = outputData->mBuffers[0].mDataByteSize / sizeof(float) / 2; // stereo

    // Clear output buffer
    std::memset(outputBuffer, 0, outputData->mBuffers[0].mDataByteSize);

    // Mix all active voices
    for (int i = 0; i < MAX_VOICES; ++i) {
        if (!_voices[i].active)
            continue;

        const auto& voice = _voices[i];
        if (voice.samples.empty())
            continue;

        uint32_t samplesAvailable = voice.samples.size();
        uint32_t framesToMix = std::min(frameCount, samplesAvailable - voice.readPosition);

        for (uint32_t f = 0; f < framesToMix; ++f) {
            float sample = voice.samples[voice.readPosition + f];
            float mixedSample = sample * voice.gain * _masterVolume;

            // Write to both channels (mono → stereo)
            outputBuffer[f * 2] += mixedSample;     // left
            outputBuffer[f * 2 + 1] += mixedSample; // right
        }

        // Advance read position
        if (!_voices[i].looping) {
            uint32_t newReadPos = voice.readPosition + framesToMix;
            if (newReadPos >= samplesAvailable) {
                // Voice finished — deallocate
                std::lock_guard<std::mutex> lock(_voiceMutex);
                deallocateVoice(i);
            } else {
                _voices[i].readPosition = newReadPos;
            }
        }
    }
}
