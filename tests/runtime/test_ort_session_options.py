"""Test OrtSessionOptions C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_session_options_cpp_compiles_and_runs():
    """Verify that ort_session_options.hpp compiles and behaves correctly in C++."""

    header_path = Path(
        "src/onnx9000/runtime/include/ort_session_options.hpp"
    ).absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto opts_result = onnx9000::OrtSessionOptions::Create();
        assert(opts_result.has_value());
        
        onnx9000::OrtSessionOptions* opts = opts_result.value();
        
        // Test Setters & State
        assert(opts->SetIntraOpNumThreads(4).has_value());
        assert(!opts->SetIntraOpNumThreads(-1).has_value());
        
        assert(opts->SetInterOpNumThreads(2).has_value());
        assert(!opts->SetInterOpNumThreads(-2).has_value());

        opts->SetGraphOptimizationLevel(onnx9000::GraphOptimizationLevel::ORT_ENABLE_BASIC);
        opts->SetExecutionMode(onnx9000::ExecutionMode::ORT_PARALLEL);
        opts->SetOptimizedModelFilePath("opt.onnx");
        opts->SetLogId("my_log");
        opts->SetLogSeverityLevel(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
        opts->SetLogVerbosityLevel(1);
        opts->EnableProfiling("prof_prefix");
        opts->SetCpuMemArena(false);
        opts->SetMemPattern(false);
        assert(opts->AddConfigEntry("key1", "val1").has_value());

        // Validate state
        auto state = opts->GetState();
        assert(state.intra_op_num_threads == 4);
        assert(state.inter_op_num_threads == 2);
        assert(state.graph_optimization_level == onnx9000::GraphOptimizationLevel::ORT_ENABLE_BASIC);
        assert(state.execution_mode == onnx9000::ExecutionMode::ORT_PARALLEL);
        assert(state.optimized_model_filepath == "opt.onnx");
        assert(state.log_id == "my_log");
        assert(state.log_severity_level == onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_VERBOSE);
        assert(state.log_verbosity_level == 1);
        assert(state.custom_profiler_filepath == "prof_prefix");
        assert(state.enable_cpu_mem_arena == false);
        assert(state.enable_mem_pattern == false);
        assert(state.config_entries.at("key1") == "val1");

        // Disable profiling
        opts->DisableProfiling();
        assert(!opts->GetState().custom_profiler_filepath.has_value());

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
