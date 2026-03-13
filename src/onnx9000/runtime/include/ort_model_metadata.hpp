#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ort_env.hpp"

namespace onnx9000 {

struct ModelMetadataState {
  std::string producer_name;
  std::string graph_name;
  std::string domain;
  std::string description;
  std::string graph_description;
  int64_t version{0};
  std::unordered_map<std::string, std::string> custom_metadata;
};

class ORT_ENV_EXPORT OrtModelMetadata {
public:
  OrtModelMetadata(const OrtModelMetadata &) = delete;
  OrtModelMetadata &operator=(const OrtModelMetadata &) = delete;

  static ort_compat::expected<OrtModelMetadata *, std::string>
  Create() noexcept {
    try {
      return new OrtModelMetadata();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtModelMetadata creation");
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

  // Setters
  void SetProducerName(const std::string &name) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.producer_name = name;
  }

  void SetGraphName(const std::string &name) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.graph_name = name;
  }

  void SetDomain(const std::string &domain) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.domain = domain;
  }

  void SetDescription(const std::string &desc) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.description = desc;
  }

  void SetGraphDescription(const std::string &desc) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.graph_description = desc;
  }

  void SetVersion(int64_t version) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.version = version;
  }

  ort_compat::expected<bool, std::string>
  AddCustomMetadata(const std::string &key, const std::string &value) noexcept {
    try {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_state.custom_metadata[key] = value;
      return true;
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error adding custom metadata");
    }
  }

  // Getters for internal usage
  ModelMetadataState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtModelMetadata() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtModelMetadata() noexcept = default;

  mutable std::mutex m_mutex;
  ModelMetadataState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
