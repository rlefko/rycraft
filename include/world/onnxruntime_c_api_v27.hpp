#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// This is the small ABI surface Rycraft uses from the pinned ONNX Runtime
// 1.27.1 C API. Slot numbers come from OrtApi version 27 in the official
// onnxruntime_c_api.h. Keeping the declaration local avoids a link-time or
// build-time dependency on the downloaded architecture-specific archive.
namespace worldgen::runtime::ort_v27 {

inline constexpr uint32_t API_VERSION = 27;

struct OrtApiBase {
    const void* (*GetApi)(uint32_t version) noexcept;
    const char* (*GetVersionString)() noexcept;
};

using OrtGetApiBaseFunction = const OrtApiBase* (*)() noexcept;

struct OrtStatus;
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtThreadingOptions;
struct OrtRunOptions;
struct OrtMemoryInfo;
struct OrtValue;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtAllocator;
struct OrtEpAssignedSubgraph;
struct OrtEpAssignedNode;

enum class LoggingLevel : int {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
};

enum class ErrorCode : int {
    Ok = 0,
    Fail = 1,
};

enum class ExecutionMode : int {
    Sequential = 0,
    Parallel = 1,
};

enum class GraphOptimizationLevel : int {
    DisableAll = 0,
    EnableBasic = 1,
    EnableExtended = 2,
    EnableAll = 99,
};

enum class TensorElementDataType : int {
    Undefined = 0,
    Float = 1,
};

enum class AllocatorType : int {
    Invalid = -1,
    Device = 0,
    Arena = 1,
};

enum class MemoryType : int {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
};

// OrtApi is an append-only table of function pointers. These offsets are
// checked against the official version 27 header by the local qualification
// workflow whenever the runtime pin changes.
enum class ApiSlot : size_t {
    GetErrorCode = 1,
    GetErrorMessage = 2,
    CreateEnv = 3,
    CreateSession = 7,
    Run = 9,
    CreateSessionOptions = 10,
    SetSessionExecutionMode = 13,
    SetSessionGraphOptimizationLevel = 23,
    SetIntraOpNumThreads = 24,
    SetInterOpNumThreads = 25,
    SessionGetInputCount = 30,
    SessionGetOutputCount = 31,
    SessionGetInputTypeInfo = 33,
    SessionGetOutputTypeInfo = 34,
    SessionGetInputName = 36,
    SessionGetOutputName = 37,
    CreateRunOptions = 39,
    CreateTensorWithDataAsOrtValue = 49,
    GetTensorMutableData = 51,
    CastTypeInfoToTensorInfo = 55,
    GetTensorElementType = 60,
    GetDimensionsCount = 61,
    GetDimensions = 62,
    GetTensorShapeElementCount = 64,
    GetTensorTypeAndShape = 65,
    CreateCpuMemoryInfo = 69,
    AllocatorFree = 76,
    GetAllocatorWithDefaultOptions = 78,
    ReleaseEnv = 92,
    ReleaseStatus = 93,
    ReleaseMemoryInfo = 94,
    ReleaseSession = 95,
    ReleaseValue = 96,
    ReleaseRunOptions = 97,
    ReleaseTypeInfo = 98,
    ReleaseTensorTypeAndShapeInfo = 99,
    ReleaseSessionOptions = 100,
    CreateEnvWithGlobalThreadPools = 119,
    DisablePerSessionThreads = 120,
    CreateThreadingOptions = 121,
    ReleaseThreadingOptions = 122,
    AddFreeDimensionOverrideByName = 124,
    GetAvailableProviders = 125,
    ReleaseAvailableProviders = 126,
    AddSessionConfigEntry = 130,
    SetGlobalIntraOpNumThreads = 147,
    SetGlobalInterOpNumThreads = 148,
    SessionOptionsAppendExecutionProvider = 216,
    SetDeterministicCompute = 273,
    SessionGetEpGraphAssignmentInfo = 407,
    EpAssignedSubgraphGetEpName = 408,
    EpAssignedSubgraphGetNodes = 409,
    EpAssignedNodeGetName = 410,
    EpAssignedNodeGetDomain = 411,
    EpAssignedNodeGetOperatorType = 412,
};

template <typename Function>
[[nodiscard]] Function apiFunction(const void* api, ApiSlot slot) noexcept {
    static_assert(std::is_pointer_v<Function>);
    static_assert(sizeof(Function) == sizeof(void (*)()));
    Function function = nullptr;
    const auto* bytes = static_cast<const std::byte*>(api);
    std::memcpy(&function, bytes + static_cast<size_t>(slot) * sizeof(function), sizeof(function));
    return function;
}

} // namespace worldgen::runtime::ort_v27
