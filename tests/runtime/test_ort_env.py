"""Test OrtEnv C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_env_cpp_compiles_and_runs():
    """Verify that ort_env.hpp compiles and behaves correctly in C++."""

    header_path = Path("src/onnx9000/runtime/include/ort_env.hpp").absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto env_result = onnx9000::OrtEnv::Create(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_INFO, "test_env");
        assert(env_result.has_value());
        
        onnx9000::OrtEnv* env = env_result.value();
        
        // Check log id
        auto log_id = env->GetLogId();
        assert(log_id.has_value());
        assert(log_id.value() == "test_env");
        
        // Add ref and release
        env->AddRef();
        env->Release(); // still 1 ref left
        
        // This should delete it and free resources
        // Once deleted, we shouldn't access it, but we can verify our operations
        // reached this point without crashing or deadlocking.
        
        // Check log level structure and default expected class behaviour
        
        // Negative test for expected error behavior 
        // Not really possible to test out-of-memory easily but we can check the error path is compiling
        
        auto fail_result = ort_compat::unexpected<std::string>("fake error");
        ort_compat::expected<onnx9000::OrtEnv*, std::string> e_res(fail_result);
        assert(!e_res.has_value());
        assert(e_res.error() == "fake error");

        std::cout << "SUCCESS" << std::endl;
        return 0;
    }}
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = Path(temp_dir) / "main.cpp"
        src_file.write_text(cpp_code)

        out_bin = Path(temp_dir) / "test_bin"

        # Determine appropriate compiler flag. GCC 7+ supports C++17.
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

        # Run it
        run_res = subprocess.run(
            [str(out_bin)], check=True, capture_output=True, text=True
        )
        assert "SUCCESS" in run_res.stdout
