#pragma once

#include <atomic>
#include <mutex>
#include <string>

#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct ThreadingOptionsState {
  int global_intra_op_num_threads{0};
  int global_inter_op_num_threads{0};
  bool global_spin_control{true};
  bool global_denormal_as_zero{false};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtThreadingOptions {
public:
  OrtThreadingOptions(const OrtThreadingOptions &) = delete;
  OrtThreadingOptions &operator=(const OrtThreadingOptions &) = delete;

  static ort_compat::expected<OrtThreadingOptions *, std::string>
  Create() noexcept {
    try {
      /// Implementation details and operations.
      return new OrtThreadingOptions();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtThreadingOptions creation");
    }
  }

  /// Implementation details and operations.

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  /// Implementation details and operations.

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  ort_compat::expected<bool, std::string>
  SetGlobalIntraOpNumThreads(int num_threads) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (num_threads < 0) {
      return ort_compat::unexpected<std::string>(
          "num_threads cannot be negative");
    }
    m_state.global_intra_op_num_threads = num_threads;
    return true;
  }

  ort_compat::expected<bool, std::string>
  SetGlobalInterOpNumThreads(int num_threads) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (num_threads < 0) {
      return ort_compat::unexpected<std::string>(
          "num_threads cannot be negative");
    }
    m_state.global_inter_op_num_threads = num_threads;
    return true;
  }

  /// Implementation details and operations.

  void SetGlobalSpinControl(bool allow_spinning) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.global_spin_control = allow_spinning;
  }

  /// Implementation details and operations.

  void SetGlobalDenormalAsZero(bool denormal_as_zero) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.global_denormal_as_zero = denormal_as_zero;
  }

  /// Implementation details and operations.

  ThreadingOptionsState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtThreadingOptions() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtThreadingOptions() noexcept = default;

  mutable std::mutex m_mutex;
  ThreadingOptionsState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
