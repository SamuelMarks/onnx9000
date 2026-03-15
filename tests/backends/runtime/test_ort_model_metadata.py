"""Test OrtModelMetadata C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_model_metadata_cpp_compiles_and_runs():
    """Verify that ort_model_metadata.hpp compiles and behaves correctly in C++."""
    header_path = Path(
        "src/onnx9000/backends/runtime/include/ort_model_metadata.hpp"
    ).absolute()
    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto meta_res = onnx9000::OrtModelMetadata::Create();
        assert(meta_res.has_value());
        auto* meta = meta_res.value();

        meta->SetProducerName("onnx9000");
        meta->SetGraphName("test_graph");
        meta->SetDomain("ai.onnx");
        meta->SetDescription("Test desc");
        meta->SetGraphDescription("Graph desc");
        meta->SetVersion(42);
        
        assert(meta->AddCustomMetadata("author", "samuel").has_value());

        auto state = meta->GetState();
        assert(state.producer_name == "onnx9000");
        assert(state.graph_name == "test_graph");
        assert(state.domain == "ai.onnx");
        assert(state.description == "Test desc");
        assert(state.graph_description == "Graph desc");
        assert(state.version == 42);
        assert(state.custom_metadata.at("author") == "samuel");

        meta->AddRef();
        meta->Release();
        meta->Release();

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
