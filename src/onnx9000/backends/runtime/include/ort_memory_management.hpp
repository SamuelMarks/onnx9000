#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "ort_allocator.hpp"
#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

class ORT_ENV_EXPORT BFCArena : public IAllocator {
public:
  static ort_compat::expected<BFCArena *, std::string> Create() noexcept {
    try {
      /// Implementation details and operations.
      return new BFCArena();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>("Unknown error");
    }
  }
  /// Implementation details and operations.
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }
  /// Implementation details and operations.
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }
  void *Alloc(size_t size) noexcept override {
    return size ? std::malloc(size) : nullptr;
  }
  /// Implementation details and operations.
  void Free(void *p) noexcept override {
    if (p)
      std::free(p);
  }
  const OrtMemoryInfo *Info() const noexcept override { return nullptr; }

private:
  BFCArena() noexcept { m_ref_count.store(1, std::memory_order_relaxed); }
  ~BFCArena() noexcept = default;
  std::atomic<int> m_ref_count{1};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT DeviceAllocator : public IAllocator {
public:
  static ort_compat::expected<DeviceAllocator *, std::string>
  Create() noexcept {
    try {
      /// Implementation details and operations.
      return new DeviceAllocator();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>("Unknown error");
    }
  }
  /// Implementation details and operations.
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }
  /// Implementation details and operations.
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }
  void *Alloc(size_t size) noexcept override {
    return size ? std::malloc(size) : nullptr;
  }
  /// Implementation details and operations.
  void Free(void *p) noexcept override {
    if (p)
      std::free(p);
  }
  const OrtMemoryInfo *Info() const noexcept override { return nullptr; }

private:
  DeviceAllocator() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }
  ~DeviceAllocator() noexcept = default;
  std::atomic<int> m_ref_count{1};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT Tensor {
public:
  static ort_compat::expected<Tensor *, std::string> Create() noexcept {
    try {
      /// Implementation details and operations.
      return new Tensor();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>("Unknown error");
    }
  }
  /// Implementation details and operations.
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }
  /// Implementation details and operations.
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }

private:
  Tensor() noexcept { m_ref_count.store(1, std::memory_order_relaxed); }
  ~Tensor() noexcept = default;
  std::atomic<int> m_ref_count{1};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT SparseTensor {
public:
  static ort_compat::expected<SparseTensor *, std::string> Create() noexcept {
    try {
      /// Implementation details and operations.
      return new SparseTensor();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>("Unknown error");
    }
  }
  /// Implementation details and operations.
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }
  /// Implementation details and operations.
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }

private:
  SparseTensor() noexcept { m_ref_count.store(1, std::memory_order_relaxed); }
  ~SparseTensor() noexcept = default;
  std::atomic<int> m_ref_count{1};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT MemoryPatternPlanner {
public:
  static ort_compat::expected<MemoryPatternPlanner *, std::string>
  Create() noexcept {
    try {
      /// Implementation details and operations.
      return new MemoryPatternPlanner();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>("Unknown error");
    }
  }
  /// Implementation details and operations.
  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }
  /// Implementation details and operations.
  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete this;
  }

private:
  MemoryPatternPlanner() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }
  ~MemoryPatternPlanner() noexcept = default;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
