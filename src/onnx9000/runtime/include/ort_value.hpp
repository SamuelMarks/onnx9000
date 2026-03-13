#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "ort_allocator.hpp"
#include "ort_env.hpp"
#include "ort_tensor_type_and_shape_info.hpp"

namespace onnx9000 {

enum class OrtValueType {
  OrtValueTypeTensor = 0,
  OrtValueTypeSequence = 1,
  OrtValueTypeMap = 2,
  OrtValueTypeOpaque = 3,
  OrtValueTypeSparseTensor = 4
};

class ORT_ENV_EXPORT OrtValue {
public:
  OrtValue(const OrtValue &) = delete;
  OrtValue &operator=(const OrtValue &) = delete;

  static ort_compat::expected<OrtValue *, std::string>
  CreateTensor(OrtAllocator *allocator, const int64_t *shape, size_t shape_len,
               ONNXTensorElementDataType type) noexcept {
    if (!allocator) {
      return ort_compat::unexpected<std::string>("allocator cannot be null");
    }

    try {
      return new OrtValue(allocator, shape, shape_len, type);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtValue creation");
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

  bool IsTensor() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_type == OrtValueType::OrtValueTypeTensor;
  }

  ort_compat::expected<OrtTensorTypeAndShapeInfo *, std::string>
  GetTensorTypeAndShape() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_type != OrtValueType::OrtValueTypeTensor) {
      return ort_compat::unexpected<std::string>("OrtValue is not a tensor");
    }
    if (!m_info) {
      return ort_compat::unexpected<std::string>("Tensor shape info is null");
    }
    m_info->AddRef();
    return m_info;
  }

  ort_compat::expected<void *, std::string> GetTensorMutableData() noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_type != OrtValueType::OrtValueTypeTensor) {
      return ort_compat::unexpected<std::string>("OrtValue is not a tensor");
    }
    return m_data;
  }

private:
  OrtValue(OrtAllocator *allocator, const int64_t *shape, size_t shape_len,
           ONNXTensorElementDataType element_type) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    m_type = OrtValueType::OrtValueTypeTensor;
    m_allocator = allocator;
    if (m_allocator) {
      m_allocator->AddRef();
    }

    auto info_res = OrtTensorTypeAndShapeInfo::Create();
    if (info_res.has_value()) {
      m_info = info_res.value();
      m_info->SetTensorElementType(element_type);
      auto shape_res = m_info->SetDimensions(shape, shape_len);
      if (shape_res.has_value()) {
        auto count_res = m_info->GetElementCount();
        if (count_res.has_value() && m_allocator) {
          size_t elem_size = 4; // Default element size approximation
          switch (element_type) {
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            elem_size = 4;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            elem_size = 8;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            elem_size = 1;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            elem_size = 1;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            elem_size = 2;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            elem_size = 2;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            elem_size = 4;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            elem_size = 4;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            elem_size = 8;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            elem_size = 8;
            break;
          case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            elem_size = 1;
            break;
          default:
            break;
          }
          m_data = m_allocator->Alloc(count_res.value() * elem_size);
        }
      }
    }
  }

  ~OrtValue() noexcept {
    if (m_data && m_allocator) {
      m_allocator->Free(m_data);
      m_data = nullptr;
    }
    if (m_info) {
      m_info->Release();
      m_info = nullptr;
    }
    if (m_allocator) {
      m_allocator->Release();
      m_allocator = nullptr;
    }
  }

  mutable std::mutex m_mutex;
  OrtValueType m_type{OrtValueType::OrtValueTypeOpaque};
  void *m_data{nullptr};
  OrtAllocator *m_allocator{nullptr};
  OrtTensorTypeAndShapeInfo *m_info{nullptr};
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
