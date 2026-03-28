#include <wasm_simd128.h>
#include <stdint.h>

extern "C" {

/**
 * Perform Sparse Matrix - Dense Matrix Multiplication (CSR format).
 * Optimized with WASM SIMD128.
 */
void spmm_csr_simd(
    int M, int N,
    const float* values,
    const int* row_ptr,
    const int* col_indices,
    const float* dense_matrix,
    float* out_matrix
) noexcept {
    for (int i = 0; i < M; ++i) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        
        if (__builtin_expect(row_start == row_end, 0)) {
            continue;
        }

        float* out_row = out_matrix + i * N;

        for (int k = row_start; k < row_end; ++k) {
            float val = values[k];
            int col = col_indices[k];
            const float* dense_row = dense_matrix + col * N;

            v128_t val_vec = wasm_f32x4_splat(val);

            int j = 0;
            for (; j <= N - 4; j += 4) {
                v128_t dense_vec = wasm_v128_load(dense_row + j);
                v128_t out_vec = wasm_v128_load(out_row + j);
                v128_t res_vec = wasm_f32x4_add(out_vec, wasm_f32x4_mul(val_vec, dense_vec));
                wasm_v128_store(out_row + j, res_vec);
            }

            for (; j < N; ++j) {
                out_row[j] += val * dense_row[j];
            }
        }
    }
}

/**
 * Item 118: Handle INT8 sparse evaluation natively in WASM via DP4A-style emulation loops.
 * Assuming weights are INT8 and activations are INT8, outputting INT32.
 */
void spmm_csr_int8_simd(
    int M, int N,
    const int8_t* values,
    const int* row_ptr,
    const int* col_indices,
    const int8_t* dense_matrix,
    int32_t* out_matrix
) noexcept {
    for (int i = 0; i < M; ++i) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        
        if (__builtin_expect(row_start == row_end, 0)) {
            continue;
        }

        int32_t* out_row = out_matrix + i * N;

        for (int k = row_start; k < row_end; ++k) {
            int8_t val = values[k];
            int col = col_indices[k];
            const int8_t* dense_row = dense_matrix + col * N;

            v128_t val_vec = wasm_i8x16_splat(val);

            int j = 0;
            for (; j <= N - 16; j += 16) {
                v128_t dense_vec = wasm_v128_load(dense_row + j);
                // Simple emulation of DP4A: multiply and accumulate
                // Since we output i32, we need to widen.
                // This is a simplified version.
                for (int l = 0; l < 16; ++l) {
                    out_row[j + l] += (int32_t)val * (int32_t)dense_row[j + l];
                }
            }

            for (; j < N; ++j) {
                out_row[j] += (int32_t)val * (int32_t)dense_row[j];
            }
        }
    }
}

/**
 * Item 117: Implement multi-threaded (SharedArrayBuffer) sparse matrix chunking natively.
 * This function handles a chunk of rows.
 */
void spmm_csr_chunk(
    int start_row, int end_row, int N,
    const float* values,
    const int* row_ptr,
    const int* col_indices,
    const float* dense_matrix,
    float* out_matrix
) noexcept {
    for (int i = start_row; i < end_row; ++i) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        
        if (row_start == row_end) continue;

        float* out_row = out_matrix + i * N;

        for (int k = row_start; k < row_end; ++k) {
            float val = values[k];
            int col = col_indices[k];
            const float* dense_row = dense_matrix + col * N;
            for (int j = 0; j < N; ++j) {
                out_row[j] += val * dense_row[j];
            }
        }
    }
}

/**
 * Item 115: Implement specialized 2:4 block-sparse CPU kernels.
 */
void spmm_2_4_simd(
    int M, int N,
    const float* values,
    const uint8_t* metadata,
    const float* dense_matrix,
    float* out_matrix,
    int K
) noexcept {
    for (int i = 0; i < M; ++i) {
        float* out_row = out_matrix + i * N;
        for (int k_block = 0; k_block < K / 4; ++k_block) {
            float val0 = values[i * (K/2) + k_block * 2];
            float val1 = values[i * (K/2) + k_block * 2 + 1];
            
            uint8_t meta = metadata[i * (K/4) + k_block];
            int col0 = (meta & 0x03);
            int col1 = (meta & 0x0C) >> 2;
            
            const float* dense_row0 = dense_matrix + (k_block * 4 + col0) * N;
            const float* dense_row1 = dense_matrix + (k_block * 4 + col1) * N;

            v128_t v0 = wasm_f32x4_splat(val0);
            v128_t v1 = wasm_f32x4_splat(val1);

            for (int j = 0; j <= N - 4; j += 4) {
                v128_t d0 = wasm_v128_load(dense_row0 + j);
                v128_t d1 = wasm_v128_load(dense_row1 + j);
                v128_t out = wasm_v128_load(out_row + j);
                
                v128_t res = wasm_f32x4_add(out, wasm_f32x4_add(wasm_f32x4_mul(v0, d0), wasm_f32x4_mul(v1, d1)));
                wasm_v128_store(out_row + j, res);
            }
        }
    }
}

}
