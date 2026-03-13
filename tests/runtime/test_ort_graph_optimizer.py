"""Test OrtGraphOptimizer C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_graph_optimizer_cpp_compiles_and_runs():
    """Verify that ort_graph_optimizer.hpp compiles and behaves correctly in C++."""

    header_path = Path(
        "src/onnx9000/runtime/include/ort_graph_optimizer.hpp"
    ).absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>
    #include <string>

    int main() {{
        onnx9000::ConstantFolding opt1;
        assert(std::string(opt1.PassName()) == "ConstantFolding");
        assert(opt1.Apply() == true);

        onnx9000::MatMulAddFusion opt2;
        assert(std::string(opt2.PassName()) == "MatMulAddFusion");
        assert(opt2.Apply() == true);

        onnx9000::NCHW_to_NHWC_Transformation opt3;
        assert(std::string(opt3.PassName()) == "NCHW_to_NHWC_Transformation");
        assert(opt3.Apply() == true);

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
