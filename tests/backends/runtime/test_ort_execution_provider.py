"""Test OrtExecutionProvider C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_execution_provider_cpp_compiles_and_runs():
    """Verify that ort_execution_provider.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_execution_provider.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto ep_res = onnx9000::CPUExecutionProvider::Create();
        assert(ep_res.has_value());
        auto* ep = ep_res.value();

        assert(std::string(ep->Type()) == "CPUExecutionProvider");

        std::vector<std::string> nodes = {{"node1", "node2"}};
        auto cap_res = ep->GetCapability(nodes);
        assert(cap_res.has_value());
        assert(cap_res.value().size() == 1);
        assert(cap_res.value()[0].sub_graph_node_names == nodes);

        assert(ep->Compile(nodes).has_value());
        assert(!ep->Compile({{}}).has_value());

        auto* alloc = ep->GetAllocator(0, onnx9000::OrtMemType::OrtMemTypeDefault);
        assert(alloc != nullptr);

        ep->AddRef();
        ep->Release();
        ep->Release();

        // Test dummies
        auto cu_ep = onnx9000::CUDAExecutionProvider::Create().value();
        assert(std::string(cu_ep->Type()) == "CUDAExecutionProvider");
        cu_ep->Release();

        auto trt_ep = onnx9000::TensorrtExecutionProvider::Create().value();
        assert(std::string(trt_ep->Type()) == "TensorrtExecutionProvider");
        trt_ep->Release();

        auto ov_ep = onnx9000::OpenVINOExecutionProvider::Create().value();
        assert(std::string(ov_ep->Type()) == "OpenVINOExecutionProvider");
        ov_ep->Release();

        auto dml_ep = onnx9000::DmlExecutionProvider::Create().value();
        assert(std::string(dml_ep->Type()) == "DmlExecutionProvider");
        dml_ep->Release();

        auto cml_ep = onnx9000::CoreMLExecutionProvider::Create().value();
        assert(std::string(cml_ep->Type()) == "CoreMLExecutionProvider");
        cml_ep->Release();

        auto xnn_ep = onnx9000::XnnpackExecutionProvider::Create().value();
        assert(std::string(xnn_ep->Type()) == "XnnpackExecutionProvider");
        xnn_ep->Release();

        auto nnapi_ep = onnx9000::NnapiExecutionProvider::Create().value();
        assert(std::string(nnapi_ep->Type()) == "NnapiExecutionProvider");
        nnapi_ep->Release();

        auto qnn_ep = onnx9000::QNNExecutionProvider::Create().value();
        assert(std::string(qnn_ep->Type()) == "QNNExecutionProvider");
        qnn_ep->Release();

        auto rocm_ep = onnx9000::ROCmExecutionProvider::Create().value();
        assert(std::string(rocm_ep->Type()) == "ROCmExecutionProvider");
        rocm_ep->Release();

        auto migraphx_ep = onnx9000::MIGraphXExecutionProvider::Create().value();
        assert(std::string(migraphx_ep->Type()) == "MIGraphXExecutionProvider");
        migraphx_ep->Release();

        auto tvm_ep = onnx9000::TvmExecutionProvider::Create().value();
        assert(std::string(tvm_ep->Type()) == "TvmExecutionProvider");
        tvm_ep->Release();

        auto webnn_ep = onnx9000::WebNNExecutionProvider::Create().value();
        assert(std::string(webnn_ep->Type()) == "WebNNExecutionProvider");
        webnn_ep->Release();

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
