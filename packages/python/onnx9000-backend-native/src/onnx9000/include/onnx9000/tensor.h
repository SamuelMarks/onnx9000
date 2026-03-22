#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

/**
 * \brief Macro to calculate a 1D flat index from a 2D coordinate based on
 * strides.
 * \param strides A pointer to the strides array.
 * \param i The row index.
 * \param j The column index.
 * \return The flattened 1D index.
 */
#define ONNX9000_INDEX_2D(strides, i, j) ((i) * (strides)[0] + (j))

/**
 * \brief Macro to calculate a 1D flat index from a 3D coordinate based on
 * strides.
 * \param strides A pointer to the strides array.
 * \param i The depth index.
 * \param j The row index.
 * \param k The column index.
 * \return The flattened 1D index.
 */
#define ONNX9000_INDEX_3D(strides, i, j, k)                                    \
  ((i) * (strides)[0] + (j) * (strides)[1] + (k))

/**
 * \brief Macro to calculate a 1D flat index from a 4D coordinate based on
 * strides.
 * \param strides A pointer to the strides array.
 * \param n The batch index.
 * \param c The channel index.
 * \param h The height index.
 * \param w The width index.
 * \return The flattened 1D index.
 */
#define ONNX9000_INDEX_4D(strides, n, c, h, w)                                 \
  ((n) * (strides)[0] + (c) * (strides)[1] + (h) * (strides)[2] + (w))

namespace onnx9000 {

/**
 * \brief Multi-dimensional array representing an ONNX Tensor.
 * \tparam T Data type of the tensor elements.
 */
template <typename T> struct Tensor {
  /** \brief Pointer to the contiguous block of memory holding tensor data. */
  T *__restrict__ data = nullptr;
  /** \brief The shape of the tensor (sizes of each dimension). */
  std::vector<int64_t> shape;
  /** \brief The strides of the tensor (step sizes in memory for each
   * dimension). */
  std::vector<int64_t> strides;

  /**
   * \brief Default constructor. Creates an empty tensor.
   */
  Tensor() = default;

  /**
   * \brief Constructs a tensor with specific data and shape.
   * Computes the contiguous strides automatically based on the provided shape.
   * \param data Pointer to the tensor elements.
   * \param shape A vector of integers specifying the size of each dimension.
   */
  Tensor(T *__restrict__ data, const std::vector<int64_t> &shape)
      : data(data), shape(shape) {
    strides.resize(shape.size());
    if (!shape.empty()) {
      strides.back() = 1;
      for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }
  }

  /**
   * \brief Computes the total number of elements in the tensor.
   * \return The product of the dimension sizes. Returns 0 if shape is empty or
   * if any dimension is 0.
   */
  size_t size() const {
    if (shape.empty())
      return 0;
    size_t s = 1;
    for (auto d : shape) {
      if (d == 0)
        return 0;
      if (d > 0)
        s *= d;
    }
    return s;
  }
};

/**
 * \brief Computes the index in the input array for a given output flat index
 * during broadcasting.
 * \param out_index The flattened index in the output array.
 * \param out_shape The shape of the output array.
 * \param in_shape The shape of the input array.
 * \param in_strides The strides of the input array.
 * \return The corresponding flat index in the input array.
 */
inline int64_t broadcast_index(int64_t out_index,
                               const std::vector<int64_t> &out_shape,
                               const std::vector<int64_t> &in_shape,
                               const std::vector<int64_t> &in_strides) {
  int64_t in_index = 0;
  int64_t temp = out_index;

  int out_ndim = static_cast<int>(out_shape.size());
  int in_ndim = static_cast<int>(in_shape.size());

  for (int i = out_ndim - 1; i >= 0; --i) {
    int64_t coord = temp % out_shape[i];
    temp /= out_shape[i];

    int in_dim_idx = in_ndim - (out_ndim - i);
    if (in_dim_idx >= 0) {
      if (in_shape[in_dim_idx] > 1) {
        in_index += coord * in_strides[in_dim_idx];
      }
    }
  }
  return in_index;
}

} // namespace onnx9000
