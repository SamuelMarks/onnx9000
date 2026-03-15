"""Test OrtRunOptions C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_run_options_cpp_compiles_and_runs():
    """Verify that ort_run_options.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_run_options.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto opts_result = onnx9000::OrtRunOptions::Create();
        assert(opts_result.has_value());
        
        onnx9000::OrtRunOptions* opts = opts_result.value();
        
        // Test Setters & State
        opts->SetRunLogSeverityLevel(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_ERROR);
        opts->SetRunLogVerbosityLevel(2);
        opts->SetRunTag("my_run");
        opts->SetOnlyExecutePathToFetches(true);

        assert(!opts->IsTerminate());
        assert(opts->SetTerminate().has_value());
        assert(opts->IsTerminate());
        assert(opts->UnsetTerminate().has_value());
        assert(!opts->IsTerminate());

        // Validate state
        auto state = opts->GetState();
        assert(state.run_log_severity_level == onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_ERROR);
        assert(state.run_log_verbosity_level == 2);
        assert(state.run_tag == "my_run");
        assert(state.terminate == false);
        assert(state.only_execute_path_to_fetches == true);

        // Add ref and release
        opts->AddRef();
        opts->Release(); // still 1 ref left
        opts->Release(); // this will delete it

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
