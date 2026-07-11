#pragma once

#include <cassert>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

// ---------------------------------------------------------------------------
// Result<T, E> — std::expected-style error handling (C++23 compatible)
//
// Usage:
//   auto r = Result<int, std::string>::ok(42);
//   auto e = Result<int, std::string>::err("bad");
//
//   if (r.is_ok()) { ... }
//   int v = r.value();          // asserts if error
//   int v = r.value_or(-1);     // returns -1 if error
//   auto mapped = r.map([](int x){ return x * 2; });
//   auto chained = r.and_then([](int x){ return Result<int>::ok(x + 1); });
// ---------------------------------------------------------------------------
template <typename T, typename E = std::string>
class Result {
public:
  // ---- Static factories ----
  static Result ok(T value) {
    return Result{std::in_place_type_t<StoredValue>{}, std::move(value)};
  }

  static Result err(E error) {
    return Result{std::in_place_type_t<StoredError>{}, std::move(error)};
  }

  // ---- State queries ----
  [[nodiscard]] constexpr bool is_ok() const {
    return std::holds_alternative<StoredValue>(storage_);
  }
  [[nodiscard]] constexpr bool is_error() const {
    return std::holds_alternative<StoredError>(storage_);
  }

  // ---- Value access ----
  [[nodiscard]] T& value() & {
    assert(is_ok() && "Called value() on error Result");
    return std::get<StoredValue>(storage_).value;
  }
  [[nodiscard]] const T& value() const& {
    assert(is_ok() && "Called value() on error Result");
    return std::get<StoredValue>(storage_).value;
  }
  [[nodiscard]] T value() && {
    assert(is_ok() && "Called value() on error Result");
    return std::move(std::get<StoredValue>(storage_).value);
  }

  [[nodiscard]] T value_or(T defaultVal) const& {
    if (is_ok()) return std::get<StoredValue>(storage_).value;
    return std::move(defaultVal);
  }
  [[nodiscard]] T value_or(T defaultVal) && {
    if (is_ok()) return std::move(std::get<StoredValue>(storage_).value);
    return std::move(defaultVal);
  }

  // ---- Error access ----
  [[nodiscard]] E& error() & {
    assert(is_error() && "Called error() on ok Result");
    return std::get<StoredError>(storage_).error;
  }
  [[nodiscard]] const E& error() const& {
    assert(is_error() && "Called error() on ok Result");
    return std::get<StoredError>(storage_).error;
  }
  [[nodiscard]] E error() && {
    assert(is_error() && "Called error() on ok Result");
    return std::move(std::get<StoredError>(storage_).error);
  }

  // ---- Transformations ----
  template <typename Func>
  auto map(Func&& f) const -> Result<std::invoke_result_t<Func, T>, E> {
    if (!is_ok()) {
      return Result<std::invoke_result_t<Func, T>, E>::err(
          std::get<StoredError>(storage_).error);
    }
    return Result<std::invoke_result_t<Func, T>, E>::ok(
        std::invoke(f, std::get<StoredValue>(storage_).value));
  }

  template <typename Func>
  auto and_then(Func&& f) const -> std::invoke_result_t<Func, T> {
    if (!is_ok()) {
      using R = std::invoke_result_t<Func, T>;
      return R::err(std::get<StoredError>(storage_).error);
    }
    return std::invoke(f, std::get<StoredValue>(storage_).value);
  }

  // Alias for value_or (Ruby-style)
  [[nodiscard]] T unwrap_or(T defaultVal) const { return value_or(std::move(defaultVal)); }

private:
  // Tagged wrapper types to disambiguate when T == E
  struct StoredValue {
    T value;
  };
  struct StoredError {
    E error;
  };

  explicit Result(std::in_place_type_t<StoredValue>, T val)
      : storage_{StoredValue{std::move(val)}} {}
  explicit Result(std::in_place_type_t<StoredError>, E err)
      : storage_{StoredError{std::move(err)}} {}

  std::variant<StoredValue, StoredError> storage_;
};
