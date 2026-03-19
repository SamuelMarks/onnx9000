import time
import os
import tempfile
import numpy as np

try:
    from safetensors.numpy import save_file as rust_save_file, load_file as rust_load_file

    HAS_RUST_SAFETENSORS = True
except ImportError:
    HAS_RUST_SAFETENSORS = False

import pickle

from onnx9000.toolkit.safetensors.parser import save_file as py_save_file, load_file as py_load_file


def run_benchmark():
    print("Generating 100MB dummy tensor...")
    # 25,000,000 float32s = 100MB
    data = {"weight": np.random.rand(25000000).astype(np.float32)}

    with tempfile.TemporaryDirectory() as d:
        p_rust = os.path.join(d, "rust.safetensors")
        p_py = os.path.join(d, "py.safetensors")
        p_pkl = os.path.join(d, "model.pkl")

        print("\n--- Serialization Benchmark ---")

        # Pickle
        t0 = time.perf_counter()
        with open(p_pkl, "wb") as f:
            pickle.dump(data, f)
        t1 = time.perf_counter()
        print(f"Pickle save: {t1 - t0:.4f}s")

        # Py Safetensors
        t0 = time.perf_counter()
        py_save_file(data, p_py)
        t1 = time.perf_counter()
        print(f"Pure Python Safetensors save: {t1 - t0:.4f}s")

        # Rust Safetensors
        if HAS_RUST_SAFETENSORS:
            t0 = time.perf_counter()
            rust_save_file(data, p_rust)
            t1 = time.perf_counter()
            print(f"Rust Safetensors save: {t1 - t0:.4f}s")
        else:
            print("Rust Safetensors not installed, skipping save.")

        print("\n--- Deserialization Benchmark ---")

        # Pickle
        t0 = time.perf_counter()
        with open(p_pkl, "rb") as f:
            loaded_pkl = pickle.load(f)
        t1 = time.perf_counter()
        print(f"Pickle load: {t1 - t0:.4f}s")

        # Py Safetensors
        t0 = time.perf_counter()
        loaded_py = py_load_file(p_py)
        t1 = time.perf_counter()
        print(f"Pure Python Safetensors load (mmap): {t1 - t0:.4f}s")

        # Rust Safetensors
        if HAS_RUST_SAFETENSORS:
            t0 = time.perf_counter()
            loaded_rust = rust_load_file(p_rust)
            t1 = time.perf_counter()
            print(f"Rust Safetensors load: {t1 - t0:.4f}s")
        else:
            print("Rust Safetensors not installed, skipping load.")


if __name__ == "__main__":
    run_benchmark()
