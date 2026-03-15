"""Test OrtSession C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_session_cpp_compiles_and_runs():
    """Verify that ort_session.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_session.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto env_result = onnx9000::OrtEnv::Create(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_INFO, "test_env");
        assert(env_result.has_value());
        onnx9000::OrtEnv* env = env_result.value();
        
        auto opts_result = onnx9000::OrtSessionOptions::Create();
        assert(opts_result.has_value());
        onnx9000::OrtSessionOptions* opts = opts_result.value();
        opts->SetIntraOpNumThreads(4);

        // Null env failure
        auto fail_env = onnx9000::OrtSession::Create(nullptr, "model.onnx", opts);
        assert(!fail_env.has_value());

        // Empty path failure
        auto fail_path = onnx9000::OrtSession::Create(env, "", opts);
        assert(!fail_path.has_value());

        // Success
        auto session_result = onnx9000::OrtSession::Create(env, "model.onnx", opts);
        assert(session_result.has_value());
        onnx9000::OrtSession* session = session_result.value();

        // Validate state
        auto state = session->GetState();
        assert(state.model_path == "model.onnx");
        assert(state.env != nullptr);
        assert(state.options.intra_op_num_threads == 4);

        // Add ref and release
        session->AddRef();
        session->Release(); // still 1 ref left
        session->Release(); // this will delete it

        opts->Release();
        env->Release();

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
