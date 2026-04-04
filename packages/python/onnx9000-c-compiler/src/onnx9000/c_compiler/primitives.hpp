
template <typename T, size_t N>
struct Tensor {
    std::array<size_t, N> shape;
    std::span<T> data;
};

// SIMD primitives
#if defined(__ARM_NEON)
#include <arm_neon.h>
#define VEC_ADD(a, b) vaddq_f32(a, b)
#elif defined(__AVX2__)
#include <immintrin.h>
#define VEC_ADD(a, b) _mm256_add_ps(a, b)
#elif defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define VEC_ADD(a, b) wasm_f32x4_add(a, b)
#else
// Fallback
#define VEC_ADD(a, b) (a + b)
#endif
