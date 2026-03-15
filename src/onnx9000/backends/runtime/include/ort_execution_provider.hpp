#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "ort_allocator.hpp"
#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct ComputeCapability {
  std::vector<std::string> sub_graph_node_names;
  std::string custom_op_name;
};

/// Implementation details and operations.

class ORT_ENV_EXPORT IExecutionProvider {
public:
  virtual ~IExecutionProvider() = default;

  virtual const char *Type() const noexcept = 0;

  virtual ort_compat::expected<std::vector<ComputeCapability>, std::string>
  GetCapability(const std::vector<std::string> &node_names) const noexcept = 0;

  virtual ort_compat::expected<bool, std::string>
  Compile(const std::vector<std::string> &sub_graph_node_names) noexcept = 0;

  virtual IAllocator *GetAllocator(int id,
                                   OrtMemType mem_type) const noexcept = 0;
};

/// Implementation details and operations.

class ORT_ENV_EXPORT CPUExecutionProvider : public IExecutionProvider {
public:
  static ort_compat::expected<CPUExecutionProvider *, std::string>
  Create() noexcept {
    try {
      auto alloc_res = OrtAllocator::CreateDefault();
      if (!alloc_res.has_value()) {
        return ort_compat::unexpected<std::string>(alloc_res.error());
      }
      /// Implementation details and operations.
      return new CPUExecutionProvider(alloc_res.value());
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during CPUExecutionProvider creation");
    }
  }

  /// Implementation details and operations.

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  /// Implementation details and operations.

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (m_allocator) {
        m_allocator->Release();
        m_allocator = nullptr;
      }
      delete this;
    }
  }

  const char *Type() const noexcept override { return "CPUExecutionProvider"; }

  ort_compat::expected<std::vector<ComputeCapability>, std::string>
  GetCapability(
      const std::vector<std::string> &node_names) const noexcept override {
    try {
      ComputeCapability cap;
      cap.sub_graph_node_names =
          node_names; // CPU EP can theoretically run anything as a fallback
      return std::vector<ComputeCapability>{cap};
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    }
  }

  ort_compat::expected<bool, std::string> Compile(
      const std::vector<std::string> &sub_graph_node_names) noexcept override {
    // Dummy compile
    if (sub_graph_node_names.empty()) {
      return ort_compat::unexpected<std::string>("Empty sub_graph");
    }
    return true;
  }

  IAllocator *GetAllocator(int id,
                           OrtMemType mem_type) const noexcept override {
    return m_allocator;
  }

private:
  CPUExecutionProvider(OrtAllocator *default_allocator) noexcept
      : m_allocator(default_allocator) {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~CPUExecutionProvider() noexcept = default;

  OrtAllocator *m_allocator{nullptr};
  std::atomic<int> m_ref_count{1};
};

// Generic Macro to generate dummy EPs for missing features/tests
#define DECLARE_DUMMY_EP(Name)                                                 \
  class ORT_ENV_EXPORT Name : public IExecutionProvider {                      \
  public:                                                                      \
    static ort_compat::expected<Name *, std::string> Create() noexcept {       \
      try {                                                                    \
        auto alloc_res = OrtAllocator::CreateDefault();                        \
        if (!alloc_res.has_value())                                            \
          return ort_compat::unexpected<std::string>(alloc_res.error());       \
        return new Name(alloc_res.value());                                    \
      } catch (const std::exception &e) {                                      \
        return ort_compat::unexpected<std::string>(e.what());                  \
      } catch (...) {                                                          \
        return ort_compat::unexpected<std::string>("Unknown error");           \
      }                                                                        \
    }                                                                          \
    void AddRef() noexcept {                                                   \
      m_ref_count.fetch_add(1, std::memory_order_relaxed);                     \
    }                                                                          \
    void Release() noexcept {                                                  \
      if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {          \
        if (m_allocator) {                                                     \
          m_allocator->Release();                                              \
          m_allocator = nullptr;                                               \
        }                                                                      \
        delete this;                                                           \
      }                                                                        \
    }                                                                          \
    const char *Type() const noexcept override { return #Name; }               \
    ort_compat::expected<std::vector<ComputeCapability>, std::string>          \
    GetCapability(                                                             \
        const std::vector<std::string> &node_names) const noexcept override {  \
      try {                                                                    \
        ComputeCapability cap;                                                 \
        cap.sub_graph_node_names = node_names;                                 \
        return std::vector<ComputeCapability>{cap};                            \
      } catch (const std::exception &e) {                                      \
        return ort_compat::unexpected<std::string>(e.what());                  \
      }                                                                        \
    }                                                                          \
    ort_compat::expected<bool, std::string>                                    \
    Compile(const std::vector<std::string> &sub_graph_node_names) noexcept     \
        override {                                                             \
      if (sub_graph_node_names.empty())                                        \
        return ort_compat::unexpected<std::string>("Empty sub_graph");         \
      return true;                                                             \
    }                                                                          \
    IAllocator *GetAllocator(int id,                                           \
                             OrtMemType mem_type) const noexcept override {    \
      return m_allocator;                                                      \
    }                                                                          \
                                                                               \
  private:                                                                     \
    Name(OrtAllocator *default_allocator) noexcept                             \
        : m_allocator(default_allocator) {                                     \
      m_ref_count.store(1, std::memory_order_relaxed);                         \
    }                                                                          \
    ~Name() noexcept = default;                                                \
    OrtAllocator *m_allocator{nullptr};                                        \
    std::atomic<int> m_ref_count{1};                                           \
  };

DECLARE_DUMMY_EP(CUDAExecutionProvider)
DECLARE_DUMMY_EP(TensorrtExecutionProvider)
DECLARE_DUMMY_EP(OpenVINOExecutionProvider)
DECLARE_DUMMY_EP(DmlExecutionProvider)
DECLARE_DUMMY_EP(CoreMLExecutionProvider)
DECLARE_DUMMY_EP(XnnpackExecutionProvider)
DECLARE_DUMMY_EP(NnapiExecutionProvider)
DECLARE_DUMMY_EP(QNNExecutionProvider)
DECLARE_DUMMY_EP(ROCmExecutionProvider)
DECLARE_DUMMY_EP(MIGraphXExecutionProvider)
DECLARE_DUMMY_EP(TvmExecutionProvider)
DECLARE_DUMMY_EP(WebNNExecutionProvider)

} // namespace onnx9000
