#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ort_allocator.hpp"
#include "ort_env.hpp"
#include "ort_io_binding.hpp"
#include "ort_model_metadata.hpp"
#include "ort_run_options.hpp"
#include "ort_session.hpp"
#include "ort_session_options.hpp"
#include "ort_value.hpp"

namespace Ort {

struct Exception : public std::runtime_error {
  explicit Exception(const std::string &msg) : std::runtime_error(msg) {}
};

inline void
ThrowOnError(const ort_compat::expected<bool, std::string> &status) {
  if (!status.has_value()) {
    throw Exception(status.error());
  }
}

template <typename T>
inline T ThrowOnError(const ort_compat::expected<T, std::string> &status) {
  if (!status.has_value()) {
    throw Exception(status.error());
  }
  return status.value();
}

struct Env {
  onnx9000::OrtEnv *p{nullptr};

  Env(onnx9000::LoggingLevel logging_level, const char *logid) {
    p = ThrowOnError(
        onnx9000::OrtEnv::Create(logging_level, logid ? logid : ""));
  }
  ~Env() {
    if (p)
      p->Release();
  }
};

struct SessionOptions {
  onnx9000::OrtSessionOptions *p{nullptr};

  SessionOptions() { p = ThrowOnError(onnx9000::OrtSessionOptions::Create()); }
  ~SessionOptions() {
    if (p)
      p->Release();
  }

  void SetIntraOpNumThreads(int intra_op_num_threads) {
    ThrowOnError(p->SetIntraOpNumThreads(intra_op_num_threads));
  }
  void SetInterOpNumThreads(int inter_op_num_threads) {
    ThrowOnError(p->SetInterOpNumThreads(inter_op_num_threads));
  }
  void SetGraphOptimizationLevel(
      onnx9000::GraphOptimizationLevel graph_optimization_level) {
    p->SetGraphOptimizationLevel(graph_optimization_level);
  }
};

struct RunOptions {
  onnx9000::OrtRunOptions *p{nullptr};

  RunOptions() { p = ThrowOnError(onnx9000::OrtRunOptions::Create()); }
  ~RunOptions() {
    if (p)
      p->Release();
  }

  void SetRunLogSeverityLevel(onnx9000::LoggingLevel level) {
    p->SetRunLogSeverityLevel(level);
  }
};

struct Session {
  onnx9000::OrtSession *p{nullptr};

  Session(Env &env, const char *model_path, const SessionOptions &options) {
    p = ThrowOnError(
        onnx9000::OrtSession::Create(env.p, model_path, options.p));
  }
  ~Session() {
    if (p)
      p->Release();
  }
};

struct MemoryInfo {
  onnx9000::OrtMemoryInfo *p{nullptr};

  MemoryInfo(const char *name, onnx9000::OrtAllocatorType alloc_type, int id,
             onnx9000::OrtMemType mem_type) {
    p = ThrowOnError(
        onnx9000::OrtMemoryInfo::Create(name, alloc_type, id, mem_type));
  }
  ~MemoryInfo() {
    if (p)
      p->Release();
  }
};

struct AllocatorWithDefaultOptions {
  onnx9000::OrtAllocator *p{nullptr};

  AllocatorWithDefaultOptions() {
    p = ThrowOnError(onnx9000::OrtAllocator::CreateDefault());
  }
  ~AllocatorWithDefaultOptions() {
    if (p)
      p->Release();
  }

  const onnx9000::OrtMemoryInfo *GetInfo() const { return p->Info(); }
  void *Alloc(size_t size) { return p->Alloc(size); }
  void Free(void *ptr) { p->Free(ptr); }
};

struct Value {
  onnx9000::OrtValue *p{nullptr};

  Value(onnx9000::OrtValue *val) : p(val) {}
  ~Value() {
    if (p)
      p->Release();
  }

  static Value CreateTensor(onnx9000::OrtAllocator *allocator,
                            const int64_t *shape, size_t shape_len,
                            onnx9000::ONNXTensorElementDataType type) {
    return Value(ThrowOnError(
        onnx9000::OrtValue::CreateTensor(allocator, shape, shape_len, type)));
  }

  bool IsTensor() const { return p->IsTensor(); }
  void *GetTensorMutableData() {
    return ThrowOnError(p->GetTensorMutableData());
  }
};

struct ModelMetadata {
  onnx9000::OrtModelMetadata *p{nullptr};

  ModelMetadata() { p = ThrowOnError(onnx9000::OrtModelMetadata::Create()); }
  ~ModelMetadata() {
    if (p)
      p->Release();
  }
};

struct IoBinding {
  onnx9000::OrtIoBinding *p{nullptr};

  IoBinding(Session &session) {
    p = ThrowOnError(onnx9000::OrtIoBinding::Create(session.p));
  }
  ~IoBinding() {
    if (p)
      p->Release();
  }

  void BindInput(const char *name, const Value &val) {
    ThrowOnError(p->BindInput(name, val.p));
  }
  void BindOutput(const char *name, const Value &val) {
    ThrowOnError(p->BindOutput(name, val.p));
  }
};

} // namespace Ort
