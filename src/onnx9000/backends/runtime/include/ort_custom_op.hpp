#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct OrtCustomOp {
  virtual ~OrtCustomOp() = default;
  virtual const char *GetName() const noexcept = 0;
  virtual const char *GetExecutionProviderType() const noexcept = 0;
  virtual void *CreateKernel() const noexcept = 0;
};

/// Implementation details and operations.

struct CustomOpDomainState {
  std::string domain;
  std::vector<OrtCustomOp *> ops;
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtCustomOpDomain {
public:
  OrtCustomOpDomain(const OrtCustomOpDomain &) = delete;
  OrtCustomOpDomain &operator=(const OrtCustomOpDomain &) = delete;

  static ort_compat::expected<OrtCustomOpDomain *, std::string>
  Create(const char *domain) noexcept {
    if (!domain) {
      return ort_compat::unexpected<std::string>("domain cannot be null");
    }
    try {
      /// Implementation details and operations.
      return new OrtCustomOpDomain(domain);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtCustomOpDomain creation");
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

  ort_compat::expected<bool, std::string> Add(OrtCustomOp *op) noexcept {
    if (!op)
      return ort_compat::unexpected<std::string>("op cannot be null");
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.ops.push_back(op);
    return true;
  }

  /// Implementation details and operations.

  CustomOpDomainState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtCustomOpDomain(const char *domain) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    m_state.domain = domain;
  }

  ~OrtCustomOpDomain() noexcept = default;

  mutable std::mutex m_mutex;
  CustomOpDomainState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
