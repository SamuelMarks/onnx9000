"""CUDA JIT Compiler."""

import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class CUDACompiler:
    """Dynamic kernel JIT compiler using nvcc."""

    @staticmethod
    def compile_kernel(
        kernel_code: str, kernel_name: str, cache_dir: Optional[str] = None
    ) -> bytes:
        """Compile CUDA C++ code to PTX."""
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        ptx_path = os.path.join(cache_dir, f"{kernel_name}.ptx")
        if os.path.exists(ptx_path):
            with open(ptx_path, "rb") as f:
                return f.read()
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(kernel_code)
            cu_path = f.name
        try:
            cmd = ["nvcc", "-ptx", cu_path, "-o", ptx_path]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(ptx_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("nvcc not found, cannot compile kernel.")
            return b""
        except subprocess.CalledProcessError as e:
            logger.error(f"nvcc compilation failed: {e.stderr.decode('utf-8')}")
            raise RuntimeError("CUDA Kernel compilation failed") from e
        finally:
            if os.path.exists(cu_path):
                os.remove(cu_path)

    @staticmethod
    def calculate_grid_block(total_elements: int, threads_per_block: int = 256) -> tuple[int, int]:
        """Calculate optimal grid and block dimensions."""
        blocks = (total_elements + threads_per_block - 1) // threads_per_block
        return (blocks, threads_per_block)
