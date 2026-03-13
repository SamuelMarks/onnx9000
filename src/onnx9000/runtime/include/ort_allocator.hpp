#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>

#include "ort_env.hpp"

namespace onnx9000 {

enum class OrtMemType {
  OrtMemTypeDefault = 0,
  OrtMemTypeCPUInput = 1,
  OrtMemTypeCPUOutput = 2,
  OrtMemTypeCPU = 3
};

enum class OrtAllocatorType {
  OrtInvalidAllocator = 0,
  OrtDeviceAllocator = 1,
  OrtArenaAllocator = 2
};

struct OrtMemoryInfoState {
  std::string name;
  OrtAllocatorType alloc_type;
  int id;
  OrtMemType mem_type;
  bool is_released{false};
};

class ORT_ENV_EXPORT OrtMemoryInfo {
public:
  OrtMemoryInfo(const OrtMemoryInfo &) = delete;
  OrtMemoryInfo &operator=(const OrtMemoryInfo &) = delete;

  static ort_compat::expected<OrtMemoryInfo *, std::string>
  Create(const char *name, OrtAllocatorType alloc_type, int id,
         OrtMemType mem_type) noexcept {
    try {
      return new OrtMemoryInfo(name, alloc_type, id, mem_type);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtMemoryInfo creation");
    }
  }

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      m_state.is_released = true;
      delete this;
    }
  }

  OrtMemoryInfoState GetState() const noexcept { return m_state; }

private:
  OrtMemoryInfo(const char *name, OrtAllocatorType alloc_type, int id,
                OrtMemType mem_type) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    if (name)
      m_state.name = name;
    m_state.alloc_type = alloc_type;
    m_state.id = id;
    m_state.mem_type = mem_type;
  }

  ~OrtMemoryInfo() noexcept = default;

  OrtMemoryInfoState m_state;
  std::atomic<int> m_ref_count{1};
};

class ORT_ENV_EXPORT IAllocator {
public:
  virtual ~IAllocator() = default;
  virtual void *Alloc(size_t size) noexcept = 0;
  virtual void Free(void *p) noexcept = 0;
  virtual const OrtMemoryInfo *Info() const noexcept = 0;
};

class ORT_ENV_EXPORT OrtAllocator : public IAllocator {
public:
  OrtAllocator(const OrtAllocator &) = delete;
  OrtAllocator &operator=(const OrtAllocator &) = delete;

  static ort_compat::expected<OrtAllocator *, std::string>
  CreateDefault() noexcept {
    try {
      auto mem_info_res =
          OrtMemoryInfo::Create("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0,
                                OrtMemType::OrtMemTypeDefault);
      if (!mem_info_res.has_value()) {
        return ort_compat::unexpected<std::string>(mem_info_res.error());
      }
      return new OrtAllocator(mem_info_res.value());
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtAllocator creation");
    }
  }

  void AddRef() noexcept {
    m_ref_count.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() noexcept {
    if (m_ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      if (m_mem_info) {
        m_mem_info->Release();
        m_mem_info = nullptr;
      }
      delete this;
    }
  }

  void *Alloc(size_t size) noexcept override {
    if (size == 0)
      return nullptr;
    return std::malloc(size);
  }

  void Free(void *p) noexcept override {
    if (p) {
      std::free(p);
    }
  }

  const OrtMemoryInfo *Info() const noexcept override { return m_mem_info; }

private:
  OrtAllocator(OrtMemoryInfo *mem_info) noexcept : m_mem_info(mem_info) {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtAllocator() noexcept = default;

  OrtMemoryInfo *m_mem_info{nullptr};
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
