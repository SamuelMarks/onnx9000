#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "ort_env.hpp"

namespace onnx9000 {

enum class GraphOptimizationLevel {
  ORT_DISABLE_ALL = 0,
  ORT_ENABLE_BASIC = 1,
  ORT_ENABLE_EXTENDED = 2,
  ORT_ENABLE_ALL = 99
};

enum class ExecutionMode { ORT_SEQUENTIAL = 0, ORT_PARALLEL = 1 };

/// Implementation details and operations.

struct SessionOptionsState {
  int intra_op_num_threads{0};
  int inter_op_num_threads{0};
  GraphOptimizationLevel graph_optimization_level{
      GraphOptimizationLevel::ORT_ENABLE_ALL};
  std::optional<std::string> optimized_model_filepath;
  ExecutionMode execution_mode{ExecutionMode::ORT_SEQUENTIAL};
  std::optional<std::string> log_id;
  LoggingLevel log_severity_level{LoggingLevel::ORT_LOGGING_LEVEL_WARNING};
  int log_verbosity_level{0};
  std::optional<std::string> custom_profiler_filepath;
  bool enable_cpu_mem_arena{true};
  bool enable_mem_pattern{true};
  std::unordered_map<std::string, std::string> config_entries;
  bool is_released{false}; // plain bool for the copy
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtSessionOptions {
public:
  OrtSessionOptions(const OrtSessionOptions &) = delete;
  OrtSessionOptions &operator=(const OrtSessionOptions &) = delete;

  static ort_compat::expected<OrtSessionOptions *, std::string>
  Create() noexcept {
    try {
      /// Implementation details and operations.
      return new OrtSessionOptions();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtSessionOptions creation");
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

  ort_compat::expected<bool, std::string>
  SetIntraOpNumThreads(int num_threads) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (num_threads < 0)
      return ort_compat::unexpected<std::string>(
          "num_threads cannot be negative");
    m_state.intra_op_num_threads = num_threads;
    return true;
  }

  ort_compat::expected<bool, std::string>
  SetInterOpNumThreads(int num_threads) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (num_threads < 0)
      return ort_compat::unexpected<std::string>(
          "num_threads cannot be negative");
    m_state.inter_op_num_threads = num_threads;
    return true;
  }

  /// Implementation details and operations.

  void SetGraphOptimizationLevel(GraphOptimizationLevel level) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.graph_optimization_level = level;
  }

  /// Implementation details and operations.

  void SetOptimizedModelFilePath(const std::string &path) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.optimized_model_filepath = path;
  }

  /// Implementation details and operations.

  void SetExecutionMode(ExecutionMode mode) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.execution_mode = mode;
  }

  /// Implementation details and operations.

  void SetLogId(const std::string &log_id) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.log_id = log_id;
  }

  /// Implementation details and operations.

  void SetLogSeverityLevel(LoggingLevel level) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.log_severity_level = level;
  }

  /// Implementation details and operations.

  void SetLogVerbosityLevel(int level) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.log_verbosity_level = level;
  }

  /// Implementation details and operations.

  void EnableProfiling(const std::string &profile_file_prefix) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.custom_profiler_filepath = profile_file_prefix;
  }

  /// Implementation details and operations.

  void DisableProfiling() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.custom_profiler_filepath = std::nullopt;
  }

  /// Implementation details and operations.

  void SetCpuMemArena(bool enable) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.enable_cpu_mem_arena = enable;
  }

  /// Implementation details and operations.

  void SetMemPattern(bool enable) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.enable_mem_pattern = enable;
  }

  ort_compat::expected<bool, std::string>
  AddConfigEntry(const std::string &key, const std::string &val) noexcept {
    try {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_state.config_entries[key] = val;
      return true;
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error adding config entry");
    }
  }

  /// Implementation details and operations.

  SessionOptionsState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtSessionOptions() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtSessionOptions() noexcept = default;

  mutable std::mutex m_mutex;
  SessionOptionsState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
