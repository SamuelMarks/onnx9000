#pragma once

#include "tensor.h"
#include <algorithm>
#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#define ONNX9000_SIMD_WIDTH 16
#elif defined(__AVX2__)
#include <immintrin.h>
#define ONNX9000_SIMD_WIDTH 8
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ONNX9000_SIMD_WIDTH 4
#else
#define ONNX9000_SIMD_WIDTH 1
#endif

namespace onnx9000 {
namespace kernels {

template <typename T>
void add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) {
  size_t n = out.size();
  T *__restrict__ pa = a.data;
  T *__restrict__ pb = b.data;
  T *__restrict__ pout = out.data;

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
#pragma GCC unroll 4
    pout[i] = pa[i] + pb[i];
  }
}

template <typename T>
void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) {
  size_t n = out.size();
  T *__restrict__ pa = a.data;
  T *__restrict__ pb = b.data;
  T *__restrict__ pout = out.data;

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
#pragma GCC unroll 4
    pout[i] = pa[i] * pb[i];
  }
}

template <typename T> void relu(const Tensor<T> &in, Tensor<T> &out) {
  size_t n = out.size();
  T *__restrict__ pin = in.data;
  T *__restrict__ pout = out.data;

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
#pragma GCC unroll 4
    pout[i] = pin[i] > 0 ? pin[i] : 0;
  }
}

} // namespace kernels
} // namespace onnx9000
