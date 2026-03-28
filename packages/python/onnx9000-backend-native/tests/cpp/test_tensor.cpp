#include "../../src/onnx9000/include/onnx9000/tensor.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace onnx9000;

void test_tensor_macros() {
  std::vector<int64_t> strides_2d = {10, 1};
  assert(ONNX9000_INDEX_2D(strides_2d.data(), 2, 3) == 23);

  std::vector<int64_t> strides_3d = {100, 10, 1};
  assert(ONNX9000_INDEX_3D(strides_3d.data(), 1, 2, 3) == 123);

  std::vector<int64_t> strides_4d = {1000, 100, 10, 1};
  assert(ONNX9000_INDEX_4D(strides_4d.data(), 1, 2, 3, 4) == 1234);
}

void test_tensor_struct() {
  float data[] = {1, 2, 3, 4};
  std::vector<int64_t> shape = {2, 2};
  Tensor<float> t(data, shape);
  assert(t.data == data);
  assert(t.shape == shape);
  assert(t.strides.size() == 2);
  assert(t.strides[0] == 2);
  assert(t.strides[1] == 1);
  assert(t.size() == 4);

  Tensor<float> empty_t;
  assert(empty_t.data == nullptr);
  assert(empty_t.shape.empty());
  assert(empty_t.strides.empty());
  assert(empty_t.size() == 0);

  std::vector<int64_t> zeroshape = {0};
  Tensor<float> zero_t(data, zeroshape);
  assert(zero_t.size() == 0);
}

void test_broadcast_index() {
  std::vector<int64_t> out_shape = {2, 3, 4};
  std::vector<int64_t> in_shape = {1, 3, 1};
  std::vector<int64_t> in_strides = {3, 1, 1};

  // flat out_index for (1, 2, 3) in out_shape (strides: 12, 4, 1) is 1*12 + 2*4
  // + 3 = 12 + 8 + 3 = 23 coord is: n=1, c=2, h=3 broadcast mapping: n=0, c=2,
  // h=0 flat in_index for (0, 2, 0) in in_shape (strides: 3, 1, 1) is 0*3 + 2*1
  // + 0 = 2
  // Test case for in_ndim < out_ndim
  // out_shape = [2, 3, 4], in_shape = [3, 4], in_strides = [4, 1]
  // out_index 23 -> coords (1, 2, 3) -> in_coords (2, 3) -> in_index 11
  std::vector<int64_t> out_shape2 = {2, 3, 4};
  std::vector<int64_t> in_shape2 = {3, 4};
  std::vector<int64_t> in_strides2 = {4, 1};
  int64_t in_idx2 = broadcast_index(23, out_shape2, in_shape2, in_strides2);
  assert(in_idx2 == 11);

  // out_index 12 -> coords (1, 0, 0) -> in_coords (0, 0) -> in_index 0
  int64_t in_idx3 = broadcast_index(12, out_shape2, in_shape2, in_strides2);
  assert(in_idx3 == 0);
}

int main() {
  test_tensor_macros();
  test_tensor_struct();
  test_broadcast_index();
  std::cout << "All C++ tests passed!" << std::endl;
  return 0;
}
