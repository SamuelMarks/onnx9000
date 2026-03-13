#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ort_env.hpp"
#include "ort_session.hpp"
#include "ort_value.hpp"

namespace onnx9000 {

struct IoBindingState {
  OrtSession *session{nullptr};
  std::unordered_map<std::string, OrtValue *> bound_inputs;
  std::unordered_map<std::string, OrtValue *> bound_outputs;
};

class ORT_ENV_EXPORT OrtIoBinding {
public:
  OrtIoBinding(const OrtIoBinding &) = delete;
  OrtIoBinding &operator=(const OrtIoBinding &) = delete;

  static ort_compat::expected<OrtIoBinding *, std::string>
  Create(OrtSession *session) noexcept {
    if (!session) {
      return ort_compat::unexpected<std::string>("session cannot be null");
    }
    try {
      return new OrtIoBinding(session);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtIoBinding creation");
    }
  }

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }

  ort_compat::expected<bool, std::string> BindInput(const std::string &name,
                                                    OrtValue *val) noexcept {
    if (!val)
      return ort_compat::unexpected<std::string>("OrtValue cannot be null");
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_state.bound_inputs.count(name)) {
      m_state.bound_inputs[name]->Release();
    }
    val->AddRef();
    m_state.bound_inputs[name] = val;
    return true;
  }

  ort_compat::expected<bool, std::string> BindOutput(const std::string &name,
                                                     OrtValue *val) noexcept {
    if (!val)
      return ort_compat::unexpected<std::string>("OrtValue cannot be null");
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_state.bound_outputs.count(name)) {
      m_state.bound_outputs[name]->Release();
    }
    val->AddRef();
    m_state.bound_outputs[name] = val;
    return true;
  }

  void ClearBoundInputs() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &pair : m_state.bound_inputs) {
      pair.second->Release();
    }
    m_state.bound_inputs.clear();
  }

  void ClearBoundOutputs() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto &pair : m_state.bound_outputs) {
      pair.second->Release();
    }
    m_state.bound_outputs.clear();
  }

  IoBindingState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    // Note: returning pointer copies in state is okay because this is just
    // observation for now. We do not manage the copied pointers.
    IoBindingState copy;
    copy.session = m_state.session;
    copy.bound_inputs = m_state.bound_inputs;
    copy.bound_outputs = m_state.bound_outputs;
    return copy;
  }

private:
  OrtIoBinding(OrtSession *session) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    m_state.session = session;
    if (m_state.session) {
      m_state.session->AddRef();
    }
  }

  ~OrtIoBinding() noexcept {
    ClearBoundInputs();
    ClearBoundOutputs();
    if (m_state.session) {
      m_state.session->Release();
      m_state.session = nullptr;
    }
  }

  mutable std::mutex m_mutex;
  IoBindingState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
