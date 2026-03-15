"""Test OrtCustomOp C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_custom_op_cpp_compiles_and_runs():
    """Verify that ort_custom_op.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_custom_op.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    struct MyCustomOp : public onnx9000::OrtCustomOp {{
        const char* GetName() const noexcept override {{ return "MyOp"; }}
        const char* GetExecutionProviderType() const noexcept override {{ return "CPUExecutionProvider"; }}
        void* CreateKernel() const noexcept override {{ return nullptr; }}
    }};

    int main() {{
        auto domain_res = onnx9000::OrtCustomOpDomain::Create("my.domain");
        assert(domain_res.has_value());
        auto* domain = domain_res.value();

        MyCustomOp op;
        assert(domain->Add(&op).has_value());

        auto state = domain->GetState();
        assert(state.domain == "my.domain");
        assert(state.ops.size() == 1);
        assert(std::string(state.ops[0]->GetName()) == "MyOp");

        domain->AddRef();
        domain->Release();
        domain->Release();

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
