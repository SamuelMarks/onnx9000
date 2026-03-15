"""Test OrtThreadingOptions C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_threading_options_cpp_compiles_and_runs():
    """Verify that ort_threading_options.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_threading_options.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto opts_res = onnx9000::OrtThreadingOptions::Create();
        assert(opts_res.has_value());
        auto* opts = opts_res.value();

        assert(opts->SetGlobalIntraOpNumThreads(4).has_value());
        assert(!opts->SetGlobalIntraOpNumThreads(-1).has_value());
        assert(opts->SetGlobalInterOpNumThreads(2).has_value());
        
        opts->SetGlobalSpinControl(false);
        opts->SetGlobalDenormalAsZero(true);

        auto state = opts->GetState();
        assert(state.global_intra_op_num_threads == 4);
        assert(state.global_inter_op_num_threads == 2);
        assert(state.global_spin_control == false);
        assert(state.global_denormal_as_zero == true);

        opts->AddRef();
        opts->Release();
        opts->Release();

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
