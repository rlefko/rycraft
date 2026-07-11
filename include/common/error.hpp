#pragma once

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <string_view>

// ---------------------------------------------------------------------------
// ErrorCode — Categorized error types for the engine
// ---------------------------------------------------------------------------
enum class ErrorCode {
  Success,
  Fatal,      // Unrecoverable — terminate
  NotFound,   // Resource or file missing
  Corrupt,    // Data integrity failure
  OutOfMemory,
};

// ---------------------------------------------------------------------------
// EngineError — Typed error with code + message
// ---------------------------------------------------------------------------
struct EngineError {
  ErrorCode code;
  std::string message;

  constexpr EngineError() : code(ErrorCode::Success), message() {}
  constexpr EngineError(ErrorCode code, std::string_view msg)
      : code(code), message(msg) {}
};

// ---------------------------------------------------------------------------
// Logging macros — timestamped output to stderr
// ---------------------------------------------------------------------------
inline std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return std::string(buf);
}

#define RY_LOG_FATAL(msg)                                                    \
  do {                                                                       \
    std::cerr << "[" << timestamp() << "] [FATAL] " << msg << std::endl;     \
    std::abort();                                                            \
  } while (0)

#define RY_LOG_ERROR(msg)                                                    \
  do {                                                                       \
    std::cerr << "[" << timestamp() << "] [ERROR] " << msg << std::endl;     \
  } while (0)
