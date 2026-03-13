"""Test OrtTensorTypeAndShapeInfo C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_tensor_type_and_shape_info_cpp_compiles_and_runs():
    """Verify that ort_tensor_type_and_shape_info.hpp compiles and behaves correctly in C++."""

    header_path = Path(
        "src/onnx9000/runtime/include/ort_tensor_type_and_shape_info.hpp"
    ).absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto info_res = onnx9000::OrtTensorTypeAndShapeInfo::Create();
        assert(info_res.has_value());
        auto* info = info_res.value();

        info->SetTensorElementType(onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        assert(info->GetTensorElementType() == onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        int64_t dims[] = {{2, 3, 4}};
        assert(info->SetDimensions(dims, 3).has_value());
        assert(info->GetDimensionsCount() == 3);

        auto count_res = info->GetElementCount();
        assert(count_res.has_value());
        assert(count_res.value() == 24);

        int64_t out_dims[3];
        assert(info->GetDimensions(out_dims, 3).has_value());
        assert(out_dims[0] == 2);
        assert(out_dims[1] == 3);
        assert(out_dims[2] == 4);

        // Test error
        assert(!info->GetDimensions(out_dims, 2).has_value());

        info->AddRef();
        info->Release();
        info->Release();

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
