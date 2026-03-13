"""Test onnxruntime_cxx_api C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_cxx_api_cpp_compiles_and_runs():
    """Verify that onnxruntime_cxx_api.hpp compiles and behaves correctly in C++."""

    header_path = Path(
        "src/onnx9000/runtime/include/onnxruntime_cxx_api.hpp"
    ).absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        try {{
            Ort::Env env(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_INFO, "test");
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(4);
            opts.SetGraphOptimizationLevel(onnx9000::GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            Ort::Session session(env, "dummy.onnx", opts);
            Ort::RunOptions run_opts;
            run_opts.SetRunLogSeverityLevel(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_WARNING);

            Ort::MemoryInfo mem_info("Cpu", onnx9000::OrtAllocatorType::OrtDeviceAllocator, 0, onnx9000::OrtMemType::OrtMemTypeDefault);

            Ort::AllocatorWithDefaultOptions alloc;
            int64_t shape[] = {{2, 2}};
            Ort::Value tensor = Ort::Value::CreateTensor(alloc.p, shape, 2, onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            assert(tensor.IsTensor());
            assert(tensor.GetTensorMutableData() != nullptr);

            Ort::ModelMetadata metadata;
            
            Ort::IoBinding binding(session);
            binding.BindInput("input", tensor);
            binding.BindOutput("output", tensor);

            std::cout << "SUCCESS" << std::endl;
        }} catch (const Ort::Exception& e) {{
            std::cerr << "Caught Ort::Exception: " << e.what() << std::endl;
            return 1;
        }}
        return 0;
    }}
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = Path(temp_dir) / "main.cpp"
        src_file.write_text(cpp_code)

        out_bin = Path(temp_dir) / "test_bin"

        compile_cmd = [
            "c++",
            "-std=c++17",
            "-pthread",
            "-fprofile-arcs",
            "-ftest-coverage",
            str(src_file),
            "-o",
            str(out_bin),
        ]

        try:
            res = subprocess.run(
                compile_cmd, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Compile failed:\n{e.stderr}")
            raise Exception(f"Compile failed: {e.stderr}")

        run_res = subprocess.run(
            [str(out_bin)], check=True, capture_output=True, text=True
        )
        assert "SUCCESS" in run_res.stdout
