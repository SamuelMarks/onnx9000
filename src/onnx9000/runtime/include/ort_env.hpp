#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>

// C++23 std::expected or fallback
#if __cplusplus >= 202302L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202302L)
#include <expected>
namespace ort_compat {
template <typename T, typename E> using expected = std::expected<T, E>;
template <typename E> using unexpected = std::unexpected<E>;
} // namespace ort_compat
#else
// Minimal expected-like wrapper for older compilers
#include <stdexcept>
#include <variant>
namespace ort_compat {
template <typename E> struct unexpected {
  E error;
  constexpr explicit unexpected(E e) : error(std::move(e)) {}
};

template <typename T, typename E> class expected {
private:
  std::variant<T, unexpected<E>> m_data;

public:
  constexpr expected(T value) noexcept(std::is_nothrow_move_constructible_v<T>)
      : m_data(std::move(value)) {}
  constexpr expected(unexpected<E> error) noexcept(
      std::is_nothrow_move_constructible_v<E>)
      : m_data(std::move(error)) {}

  constexpr bool has_value() const noexcept { return m_data.index() == 0; }
  constexpr const T &value() const {
    if (has_value())
      return std::get<0>(m_data);
    throw std::runtime_error("Bad expected access");
  }
  constexpr const E &error() const { return std::get<1>(m_data).error; }
};
} // namespace ort_compat
#endif

// Platform support macros (GCC, Clang, MSVC, Emscripten)
#if defined(__EMSCRIPTEN__)
#define ORT_ENV_EXPORT __attribute__((used))
#elif defined(_MSC_VER)
#define ORT_ENV_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_ENV_EXPORT __attribute__((visibility("default")))
#else
#define ORT_ENV_EXPORT
#endif

namespace onnx9000 {

// Enumeration for logging levels mirroring OrtLoggingLevel
enum class LoggingLevel {
  ORT_LOGGING_LEVEL_VERBOSE = 0,
  ORT_LOGGING_LEVEL_INFO = 1,
  ORT_LOGGING_LEVEL_WARNING = 2,
  ORT_LOGGING_LEVEL_ERROR = 3,
  ORT_LOGGING_LEVEL_FATAL = 4
};

// Internal state structure
struct EnvState {
  LoggingLevel log_level{LoggingLevel::ORT_LOGGING_LEVEL_WARNING};
  std::string log_id;
  std::atomic<bool> is_released{false};
};

/**
 * @brief Thread-safe Environment encapsulating ONNX Runtime global state.
 */
class ORT_ENV_EXPORT OrtEnv {
public:
  OrtEnv(const OrtEnv &) = delete;
  OrtEnv &operator=(const OrtEnv &) = delete;

  /**
   * @brief Creates a new OrtEnv instance with thread-safe reference counting.
   * @return std::expected containing OrtEnv pointer or an error message.
   */
  static ort_compat::expected<OrtEnv *, std::string>
  Create(LoggingLevel default_logging_level, std::string log_id) noexcept {
    try {
      return new OrtEnv(default_logging_level, std::move(log_id));
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtEnv creation");
    }
  }

  /**
   * @brief Increments the reference count.
   */
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  /**
   * @brief Decrements the reference count and releases if 0.
   */
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_state.is_released.store(true, std::memory_order_release);
      delete this;
    }
  }

  /**
   * @brief Optionally gets the internal log id.
   */
  std::optional<std::string> GetLogId() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_state.is_released.load(std::memory_order_acquire)) {
      return std::nullopt;
    }
    return m_state.log_id;
  }

private:
  OrtEnv(LoggingLevel level, std::string id) noexcept {
    m_state.log_level = level;
    m_state.log_id = std::move(id);
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtEnv() noexcept = default;

  mutable std::mutex m_mutex;
  EnvState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
