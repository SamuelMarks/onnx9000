"""Test OrtAllocator C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_allocator_cpp_compiles_and_runs():
    """Verify that ort_allocator.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_allocator.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        // Test MemoryInfo
        auto mem_info_res = onnx9000::OrtMemoryInfo::Create("Custom", onnx9000::OrtAllocatorType::OrtArenaAllocator, 1, onnx9000::OrtMemType::OrtMemTypeCPU);
        assert(mem_info_res.has_value());
        auto* mem_info = mem_info_res.value();
        
        auto state = mem_info->GetState();
        assert(state.name == "Custom");
        assert(state.alloc_type == onnx9000::OrtAllocatorType::OrtArenaAllocator);
        assert(state.id == 1);
        assert(state.mem_type == onnx9000::OrtMemType::OrtMemTypeCPU);
        
        mem_info->AddRef();
        mem_info->Release();
        mem_info->Release();

        // Test Default Allocator
        auto alloc_res = onnx9000::OrtAllocator::CreateDefault();
        assert(alloc_res.has_value());
        auto* alloc = alloc_res.value();

        auto* info = alloc->Info();
        assert(info != nullptr);
        assert(info->GetState().name == "Cpu");
        assert(info->GetState().alloc_type == onnx9000::OrtAllocatorType::OrtDeviceAllocator);

        // Test Alloc/Free
        void* p = alloc->Alloc(1024);
        assert(p != nullptr);
        alloc->Free(p);

        // Test null allocation
        assert(alloc->Alloc(0) == nullptr);
        alloc->Free(nullptr);

        alloc->AddRef();
        alloc->Release();
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
