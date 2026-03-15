#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>

#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct RunOptionsState {
  LoggingLevel run_log_severity_level{LoggingLevel::ORT_LOGGING_LEVEL_WARNING};
  int run_log_verbosity_level{0};
  std::optional<std::string> run_tag;
  bool terminate{false}; // Plain bool instead of atomic
  bool only_execute_path_to_fetches{false};
  bool is_released{false}; // Plain bool for copy state
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtRunOptions {
public:
  OrtRunOptions(const OrtRunOptions &) = delete;
  OrtRunOptions &operator=(const OrtRunOptions &) = delete;

  static ort_compat::expected<OrtRunOptions *, std::string> Create() noexcept {
    try {
      /// Implementation details and operations.
      return new OrtRunOptions();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtRunOptions creation");
    }
  }

  /// Implementation details and operations.

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  /// Implementation details and operations.

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_state.is_released = true;
      delete this;
    }
  }

  /// Implementation details and operations.

  void SetRunLogSeverityLevel(LoggingLevel level) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.run_log_severity_level = level;
  }

  /// Implementation details and operations.

  void SetRunLogVerbosityLevel(int level) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.run_log_verbosity_level = level;
  }

  /// Implementation details and operations.

  void SetRunTag(const std::string &run_tag) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.run_tag = run_tag;
  }

  ort_compat::expected<bool, std::string> SetTerminate() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.terminate = true;
    return true;
  }

  ort_compat::expected<bool, std::string> UnsetTerminate() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.terminate = false;
    return true;
  }

  // Usually cancellation is polled safely during compute
  /// Implementation details and operations.
  bool IsTerminate() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state.terminate;
  }

  /// Implementation details and operations.

  void SetOnlyExecutePathToFetches(bool enable) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.only_execute_path_to_fetches = enable;
  }

  // Provide a state copy (non-atomic variables are copied inside lock, atomic
  // polled)
  /// Implementation details and operations.
  RunOptionsState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtRunOptions() noexcept { m_ref_count.store(1, std::memory_order_relaxed); }

  ~OrtRunOptions() noexcept = default;

  mutable std::mutex m_mutex;
  RunOptionsState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
