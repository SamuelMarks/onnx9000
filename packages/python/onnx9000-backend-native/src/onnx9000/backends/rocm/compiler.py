"""ROCm JIT Compiler."""

import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class ROCmCompiler:
    """Dynamic kernel JIT compiler using hipcc."""

    @staticmethod
    def compile_kernel(
        kernel_code: str, kernel_name: str, cache_dir: Optional[str] = None
    ) -> bytes:
        """Compile HIP C++ code to HSA code object."""
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        hsaco_path = os.path.join(cache_dir, f"{kernel_name}.hsaco")
        if os.path.exists(hsaco_path):
            with open(hsaco_path, "rb") as f:
                return f.read()
        with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
            f.write(kernel_code)
            cpp_path = f.name
        try:
            cmd = ["hipcc", "--genco", cpp_path, "-o", hsaco_path]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(hsaco_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("hipcc not found, cannot compile kernel.")
            return b""
        except subprocess.CalledProcessError as e:
            logger.error(f"hipcc compilation failed: {e.stderr.decode('utf-8')}")
            raise RuntimeError("ROCm Kernel compilation failed") from e
        finally:
            if os.path.exists(cpp_path):
                os.remove(cpp_path)
