"""Test OrtAdvancedExecution C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_advanced_execution_cpp_compiles_and_runs():
    """Verify that ort_advanced_execution.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_advanced_execution.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        onnx9000::ThreadpoolManager tm(4, 2);
        assert(tm.GetIntraThreads() == 4);
        assert(tm.GetInterThreads() == 2);
        tm.Execute(nullptr, nullptr);

        onnx9000::AdvancedSessionState state;
        auto alloc_res = onnx9000::OrtAllocator::CreateDefault();
        auto* alloc = alloc_res.value();
        
        int64_t shape[] = {{2, 2}};
        auto val_res = onnx9000::OrtValue::CreateTensor(alloc, shape, 2, onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto* val = val_res.value();

        state.AddInitializedWeight("w1", val);
        assert(state.GetWeight("w1") == val);
        assert(state.GetWeight("w2") == nullptr);

        onnx9000::DataTransferManager dtm;
        assert(dtm.CopyTensor(val, val).has_value());
        assert(!dtm.CopyTensor(nullptr, val).has_value());

        std::vector<std::string> fetches = {{"out"}};
        onnx9000::ExecutionFrame frame(fetches, &state);

        assert(frame.AllocateIntermediate("inter1", val).has_value());
        assert(frame.GetIntermediate("inter1") == val);
        assert(frame.GetIntermediate("inter2") == nullptr);

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
