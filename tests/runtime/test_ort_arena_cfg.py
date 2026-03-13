"""Test OrtArenaCfg C++ implementation."""

import subprocess
import tempfile
from pathlib import Path


def test_ort_arena_cfg_cpp_compiles_and_runs():
    """Verify that ort_arena_cfg.hpp compiles and behaves correctly in C++."""

    header_path = Path("src/onnx9000/runtime/include/ort_arena_cfg.hpp").absolute()

    cpp_code = f"""
    #include "{header_path}"
    #include <iostream>
    #include <cassert>

    int main() {{
        auto cfg_res = onnx9000::OrtArenaCfg::Create(1024, 1, 256, 128);
        assert(cfg_res.has_value());
        auto* cfg = cfg_res.value();

        auto state = cfg->GetState();
        assert(state.max_mem == 1024);
        assert(state.arena_extend_strategy == 1);
        assert(state.initial_chunk_size_bytes == 256);
        assert(state.max_dead_bytes_per_chunk == 128);

        cfg->AddRef();
        cfg->Release();
        cfg->Release();

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
