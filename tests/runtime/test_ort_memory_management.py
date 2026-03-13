"""Test memory management C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_memory_management_cpp_compiles_and_runs():
    """Verify that ort_memory_management.hpp compiles and behaves correctly in C++."""

    header_path = Path(
        "src/onnx9000/runtime/include/ort_memory_management.hpp"
    ).absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto arena = onnx9000::BFCArena::Create().value();
        void* p1 = arena->Alloc(100);
        assert(p1);
        arena->Free(p1);
        arena->Release();

        auto dev_alloc = onnx9000::DeviceAllocator::Create().value();
        void* p2 = dev_alloc->Alloc(200);
        assert(p2);
        dev_alloc->Free(p2);
        dev_alloc->Release();

        auto t = onnx9000::Tensor::Create().value();
        t->Release();

        auto st = onnx9000::SparseTensor::Create().value();
        st->Release();

        auto mpp = onnx9000::MemoryPatternPlanner::Create().value();
        mpp->Release();

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
