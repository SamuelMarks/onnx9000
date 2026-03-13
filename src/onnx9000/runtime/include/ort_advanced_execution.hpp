#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ort_env.hpp"
#include "ort_execution_provider.hpp"
#include "ort_io_binding.hpp"
#include "ort_value.hpp"

namespace onnx9000 {

class ORT_ENV_EXPORT ThreadpoolManager {
public:
  ThreadpoolManager(int intra_threads, int inter_threads)
      : m_intra_threads(intra_threads), m_inter_threads(inter_threads) {}

  int GetIntraThreads() const noexcept { return m_intra_threads; }
  int GetInterThreads() const noexcept { return m_inter_threads; }

  void Execute(void (*fn)(void *), void *ctx) noexcept {
    // Dummy execute
    if (fn)
      fn(ctx);
  }

private:
  int m_intra_threads;
  int m_inter_threads;
};

class ORT_ENV_EXPORT AdvancedSessionState {
public:
  AdvancedSessionState() = default;

  void AddInitializedWeight(const std::string &name, OrtValue *val) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_weights[name] = val;
    if (val)
      val->AddRef();
  }

  OrtValue *GetWeight(const std::string &name) const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_weights.find(name);
    return it != m_weights.end() ? it->second : nullptr;
  }

  ~AdvancedSessionState() {
    for (auto &pair : m_weights) {
      if (pair.second)
        pair.second->Release();
    }
  }

private:
  mutable std::mutex m_mutex;
  std::unordered_map<std::string, OrtValue *> m_weights;
};

class ORT_ENV_EXPORT DataTransferManager {
public:
  DataTransferManager() = default;

  ort_compat::expected<bool, std::string> CopyTensor(const OrtValue *src,
                                                     OrtValue *dst) noexcept {
    if (!src || !dst)
      return ort_compat::unexpected<std::string>("Null tensors for copy");
    // Dummy copy
    return true;
  }
};

class ORT_ENV_EXPORT ExecutionFrame {
public:
  ExecutionFrame(const std::vector<std::string> &fetch_names,
                 AdvancedSessionState *state)
      : m_fetches(fetch_names), m_session_state(state) {}

  ort_compat::expected<bool, std::string>
  AllocateIntermediate(const std::string &name, OrtValue *val) noexcept {
    if (!val)
      return ort_compat::unexpected<std::string>("Null intermediate");
    m_intermediates[name] = val;
    val->AddRef();
    return true;
  }

  OrtValue *GetIntermediate(const std::string &name) const noexcept {
    auto it = m_intermediates.find(name);
    return it != m_intermediates.end() ? it->second : nullptr;
  }

  ~ExecutionFrame() {
    for (auto &pair : m_intermediates) {
      if (pair.second)
        pair.second->Release();
    }
  }

private:
  std::vector<std::string> m_fetches;
  AdvancedSessionState *m_session_state;
  std::unordered_map<std::string, OrtValue *> m_intermediates;
};

} // namespace onnx9000
