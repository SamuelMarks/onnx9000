"""Test OrtValue C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_value_cpp_compiles_and_runs():
    """Verify that ort_value.hpp compiles and behaves correctly in C++."""
    header_path = Path("src/onnx9000/backends/runtime/include/ort_value.hpp").absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto alloc_res = onnx9000::OrtAllocator::CreateDefault();
        assert(alloc_res.has_value());
        auto* alloc = alloc_res.value();

        int64_t shape[] = {{2, 3}};
        auto val_res = onnx9000::OrtValue::CreateTensor(alloc, shape, 2, onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        assert(val_res.has_value());
        auto* val = val_res.value();

        assert(val->IsTensor());

        auto info_res = val->GetTensorTypeAndShape();
        assert(info_res.has_value());
        auto* info = info_res.value();
        assert(info->GetTensorElementType() == onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        info->Release();

        auto data_res = val->GetTensorMutableData();
        assert(data_res.has_value());
        assert(data_res.value() != nullptr);

        // Fill data
        float* f_data = static_cast<float*>(data_res.value());
        f_data[0] = 1.0f;
        f_data[5] = 2.0f;

        val->AddRef();
        val->Release();
        val->Release();

        alloc->Release();

        std::cout << "SUCCESS" << std::endl;
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
