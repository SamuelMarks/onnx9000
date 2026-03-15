#include <cstdint>
#include <expected> // C++23 feature, fallback to alternative if needed, assuming C++23 based on prompt
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/bind.h>
#include <wasm_simd128.h>
#endif

// Step 300: wasm_simd128.h wrapper
#if defined(__wasm_simd128__)
#define HAS_SIMD 1
#else
#define HAS_SIMD 0
#endif

namespace onnx9000 {

/// Manages and tracks WebAssembly linear memory allocations with 16-byte
/// alignment.

class WasmMemoryPlanner {
  // Step 302: WASM memory planner
  size_t current_offset = 0;

public:
  WasmMemoryPlanner() noexcept = default;

  std::expected<size_t, std::string> allocate(size_t size) noexcept {
    size_t align = size % 16 == 0 ? size : size + (16 - (size % 16));
    size_t start = current_offset;
    // Step 312: tracking
    current_offset += align;
    return start;
  }
};

/// Provides static SIMD-optimized vector math operations for WebAssembly
/// execution.

class WasmOps {
public:
  // Step 304: Vector Addition
  /// Performs element-wise addition on 32-bit float arrays, using 128-bit SIMD
  /// if available.
  static void add_f32(float *a, float *b, float *y, size_t size) noexcept {
#if HAS_SIMD
    size_t i = 0;
    // Step 313: loop unrolling natively
    for (; i + 3 < size; i += 4) {
      v128_t va = wasm_v128_load(&a[i]);
      v128_t vb = wasm_v128_load(&b[i]);
      v128_t vy = wasm_f32x4_add(va, vb);
      wasm_v128_store(&y[i], vy);
    }
    for (; i < size; ++i)
      y[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < size; ++i)
      y[i] = a[i] + b[i];
#endif
  }

  // Step 305: Vector Multiplication
  /// Performs element-wise multiplication on 32-bit float arrays, utilizing
  /// WASM SIMD where possible.
  static void mul_f32(float *a, float *b, float *y, size_t size) noexcept {
#if HAS_SIMD
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
      v128_t va = wasm_v128_load(&a[i]);
      v128_t vb = wasm_v128_load(&b[i]);
      v128_t vy = wasm_f32x4_mul(va, vb);
      wasm_v128_store(&y[i], vy);
    }
    for (; i < size; ++i)
      y[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < size; ++i)
      y[i] = a[i] * b[i];
#endif
  }

  // Step 308: Relu
  /// Applies the Rectified Linear Unit (ReLU) activation function element-wise
  /// on a float32 array.
  static void relu_f32(float *a, float *y, size_t size) noexcept {
#if HAS_SIMD
    v128_t vzero = wasm_f32x4_splat(0.0f);
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
      v128_t va = wasm_v128_load(&a[i]);
      v128_t vy = wasm_f32x4_max(va, vzero);
      wasm_v128_store(&y[i], vy);
    }
    for (; i < size; ++i)
      y[i] = a[i] > 0 ? a[i] : 0.0f;
#else
    for (size_t i = 0; i < size; ++i)
      y[i] = a[i] > 0 ? a[i] : 0.0f;
#endif
  }
};

} // namespace onnx9000

#if defined(__EMSCRIPTEN__)
EMSCRIPTEN_BINDINGS(onnx9000_wasm) {
  emscripten::class_<onnx9000::WasmMemoryPlanner>("WasmMemoryPlanner")
      .constructor<>()
      .function("allocate", &onnx9000::WasmMemoryPlanner::allocate);
}
#endif
