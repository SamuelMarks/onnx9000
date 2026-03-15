#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ort_env.hpp"
#include "ort_session_options.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct SessionState {
  std::string model_path;
  OrtEnv *env{nullptr};
  SessionOptionsState options;
  bool is_released{false};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtSession {
public:
  OrtSession(const OrtSession &) = delete;
  OrtSession &operator=(const OrtSession &) = delete;

  static ort_compat::expected<OrtSession *, std::string>
  Create(OrtEnv *env, const std::string &model_path,
         const OrtSessionOptions *options) noexcept {
    if (!env) {
      return ort_compat::unexpected<std::string>("OrtEnv cannot be null");
    }
    if (model_path.empty()) {
      return ort_compat::unexpected<std::string>("model_path cannot be empty");
    }

    try {
      /// Implementation details and operations.
      return new OrtSession(env, model_path, options);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtSession creation");
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
      if (m_state.env) {
        m_state.env->Release();
        m_state.env = nullptr;
      }
      delete this;
    }
  }

  /// Implementation details and operations.

  SessionState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    SessionState copy;
    copy.model_path = m_state.model_path;
    copy.env = m_state.env;
    copy.options = m_state.options;
    copy.is_released = m_state.is_released;
    return copy;
  }

private:
  OrtSession(OrtEnv *env, std::string model_path,
             const OrtSessionOptions *options) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    m_state.env = env;
    if (m_state.env) {
      m_state.env->AddRef();
    }
    m_state.model_path = std::move(model_path);
    if (options) {
      m_state.options = options->GetState();
    }
  }

  ~OrtSession() noexcept = default;

  mutable std::mutex m_mutex;
  SessionState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
