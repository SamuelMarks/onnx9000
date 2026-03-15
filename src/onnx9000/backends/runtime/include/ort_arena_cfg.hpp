#pragma once

#include <atomic>
#include <mutex>
#include <string>

#include "ort_env.hpp"

namespace onnx9000 {

/// Implementation details and operations.

struct ArenaCfgState {
  size_t max_mem{0};
  int arena_extend_strategy{0};
  int initial_chunk_size_bytes{-1};
  int max_dead_bytes_per_chunk{-1};
  int initial_growth_chunk_size_bytes{-1};
};

/// Implementation details and operations.

class ORT_ENV_EXPORT OrtArenaCfg {
public:
  OrtArenaCfg(const OrtArenaCfg &) = delete;
  OrtArenaCfg &operator=(const OrtArenaCfg &) = delete;

  static ort_compat::expected<OrtArenaCfg *, std::string>
  Create(size_t max_mem, int arena_extend_strategy,
         int initial_chunk_size_bytes, int max_dead_bytes_per_chunk) noexcept {
    try {
      /// Implementation details and operations.
      return new OrtArenaCfg(max_mem, arena_extend_strategy,
                             initial_chunk_size_bytes,
                             max_dead_bytes_per_chunk);
    } catch (const std::exception &e) {
      return ort_compat::unexpected<std::string>(e.what());
    } catch (...) {
      return ort_compat::unexpected<std::string>(
          "Unknown error during OrtArenaCfg creation");
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

  ArenaCfgState GetState() const noexcept {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  OrtArenaCfg(size_t max_mem, int arena_extend_strategy,
              int initial_chunk_size_bytes,
              int max_dead_bytes_per_chunk) noexcept {
    m_ref_count.store(1, std::memory_order_relaxed);
    m_state.max_mem = max_mem;
    m_state.arena_extend_strategy = arena_extend_strategy;
    m_state.initial_chunk_size_bytes = initial_chunk_size_bytes;
    m_state.max_dead_bytes_per_chunk = max_dead_bytes_per_chunk;
  }

  ~OrtArenaCfg() noexcept = default;

  mutable std::mutex m_mutex;
  ArenaCfgState m_state;
  std::atomic<int> m_ref_count{1};
};

} // namespace onnx9000
