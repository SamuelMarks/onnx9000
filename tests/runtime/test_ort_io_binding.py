"""Test OrtIoBinding C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_io_binding_cpp_compiles_and_runs():
    """Verify that ort_io_binding.hpp compiles and behaves correctly in C++."""

    header_path = Path("src/onnx9000/runtime/include/ort_io_binding.hpp").absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto env_res = onnx9000::OrtEnv::Create(onnx9000::LoggingLevel::ORT_LOGGING_LEVEL_INFO, "env");
        auto* env = env_res.value();

        auto opts_res = onnx9000::OrtSessionOptions::Create();
        auto* opts = opts_res.value();

        auto sess_res = onnx9000::OrtSession::Create(env, "model.onnx", opts);
        auto* session = sess_res.value();

        auto io_res = onnx9000::OrtIoBinding::Create(session);
        assert(io_res.has_value());
        auto* io = io_res.value();

        auto alloc_res = onnx9000::OrtAllocator::CreateDefault();
        auto* alloc = alloc_res.value();

        int64_t shape[] = {{2}};
        auto val_res = onnx9000::OrtValue::CreateTensor(alloc, shape, 1, onnx9000::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto* val = val_res.value();

        assert(io->BindInput("input1", val).has_value());
        assert(io->BindOutput("output1", val).has_value());

        auto state = io->GetState();
        assert(state.session == session);
        assert(state.bound_inputs.count("input1"));
        assert(state.bound_outputs.count("output1"));

        io->ClearBoundInputs();
        io->ClearBoundOutputs();
        
        state = io->GetState();
        assert(state.bound_inputs.empty());
        assert(state.bound_outputs.empty());

        io->AddRef();
        io->Release();
        io->Release();

        val->Release();
        alloc->Release();
        session->Release();
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
