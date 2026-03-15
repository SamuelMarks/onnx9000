#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "ort_env.hpp"

namespace onnx9000 {

enum class ONNXTensorElementDataType {
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtTensorTypeAndShapeInfo {
public:
  OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo &) = delete;
  OrtTensorTypeAndShapeInfo &
  operator=(const OrtTensorTypeAndShapeInfo &) = delete;

  static ort_compat::expected<OrtTensorTypeAndShapeInfo *, std::string>
  Create() noexcept {
    try {
      /// Implementation details and operations.
      return new OrtTensorTypeAndShapeInfo();
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtTensorTypeAndShapeInfo creation");
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

  /// Implementation details and operations.

  void SetTensorElementType(ONNXTensorElementDataType type) noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_type = type;
  }

  /// Implementation details and operations.

  ONNXTensorElementDataType GetTensorElementType() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_type;
  }

  ort_compat::expected<bool, std::string>
  SetDimensions(const int64_t *dims, size_t dim_count) noexcept {
    try {
      std::lock_guard<std::mutex> lock(m_mutex);
      if (dims == nullptr && dim_count > 0) {
        return ort_compat::unexpected<std::string>("dims pointer is null");
      }
      m_dims.assign(dims, dims + dim_count);
      return true;
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error in SetDimensions");
    }
  }

  /// Implementation details and operations.

  size_t GetDimensionsCount() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_dims.size();
  }

  ort_compat::expected<bool, std::string>
  GetDimensions(int64_t *dims, size_t dim_count) const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (dim_count < m_dims.size()) {
      return ort_compat::unexpected<std::string>("dim_count is too small");
    }
    if (dims == nullptr && m_dims.size() > 0) {
      return ort_compat::unexpected<std::string>("dims pointer is null");
    }
    for (size_t i = 0; i < m_dims.size(); ++i) {
      dims[i] = m_dims[i];
    }
    return true;
  }

  ort_compat::expected<int64_t, std::string> GetElementCount() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_dims.empty())
      return 1; // scalar
    int64_t count = 1;
    for (auto d : m_dims) {
      if (d < 0)
        return ort_compat::unexpected<std::string>("Dynamic dimension present");
      count *= d;
    }
    return count;
  }

private:
  OrtTensorTypeAndShapeInfo() noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
  }

  ~OrtTensorTypeAndShapeInfo() noexcept = default;

  mutable std::mutex m_mutex;
  ONNXTensorElementDataType m_type{
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::vector<int64_t> m_dims;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
